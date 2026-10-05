#!/usr/bin/env python3
"""Operatori DiffusionNet col Laplaciano robusto, su mesh riscalate ad area totale 1 (ablazione C).

Variante d del brainstorm H3 (aau/brainstorm/h3_spectral.py), la piu' consistente fra le
topologie dello stesso soggetto: robust_laplacian.mesh_laplacian (Sharp & Crane 2020, Delaunay
intrinseco sul tufted cover, mollify di default) sulla mesh centrata e riscalata ad area 1.

Rispetto a diffusion_net.geometry.compute_operators (riga 276+) cambiano SOLO L e la massa: e'
il ramo commentato per il robusto, con la massa piu' eps*media come in h3_spectral.robust_pairs.
Frame tangenti, autosolver (L + eps I, sigma = eps, eps crescente se fallisce) e gradienti sono
quelli di compute_operators, riga per riga. I gradienti usano gli archi della sparsita' di L,
come li': col robusto sono gli archi della triangolazione intrinseca, che possono unire vertici
non adiacenti nella mesh (edge_tangent_vectors proietta qualunque coppia sul piano tangente).

Chiavi e formato come precompute_operators_npz.py (np.savez_compressed, float32, COO int64):
verts (area 1), faces, mass, evals, evecs, {L,gradX,gradY}_{indices,values,shape}. Il loader
congelato ricentra e divide per maxabs i vertici, quindi l'input del modello e' lo stesso che
con la cartella standard; cambiano solo gli operatori.

    aau/run.sh aau/models/robust_area1_operators.py --input-dir <mesh> --output-dir <out> --shard 3/24
    aau/run.sh aau/models/robust_area1_operators.py --check --output-dir <out> --ref-dir <withops>

Lo shard e' uno stride su una permutazione FISSA della lista ordinata: lo stride sulla lista
ordinata darebbe a ogni shard una sola topologia se n e' multiplo di 6 (vedi
aau/precompute_ops_areanorm.sbatch). Scrittura atomica (tmp + rename), file esistenti saltati.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "diffusion-net/src"))
sys.path.insert(0, str(THIS_DIR.parent / "brainstorm"))

from h3_spectral import EPS, mesh_area, unit_area  # noqa: E402

KEYS = ("verts", "faces", "mass", "evals", "evecs",
        "L_indices", "L_values", "L_shape", "gradX_indices", "gradX_values", "gradX_shape",
        "gradY_indices", "gradY_values", "gradY_shape")
PERM_SEED = 0


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        V = z["verts"] if "verts" in z.files else z["V"]
        F = z["faces"] if "faces" in z.files else z["F"]
        return np.asarray(V), np.asarray(F)


def robust_operators(verts, faces, k_eig: int):
    """compute_operators con L e massa del Laplaciano robusto. Torch in, torch out."""
    import robust_laplacian
    import scipy.sparse
    import scipy.sparse.linalg as sla
    import torch
    from diffusion_net import utils
    from diffusion_net.geometry import build_grad, build_tangent_frames, edge_tangent_vectors

    device, dtype = verts.device, verts.dtype
    verts_np = verts.cpu().numpy().astype(np.float64)
    faces_np = faces.cpu().numpy()
    frames = build_tangent_frames(verts, faces)

    L, M = robust_laplacian.mesh_laplacian(verts_np, faces_np)
    massvec_np = np.asarray(M.diagonal(), dtype=np.float64)
    massvec_np = massvec_np + EPS * np.mean(massvec_np)
    if np.isnan(L.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec_np).any():
        raise RuntimeError("NaN mass matrix")

    L_coo = L.tocoo()
    inds_row, inds_col = L_coo.row, L_coo.col

    L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * EPS).tocsc()
    Mmat = scipy.sparse.diags(massvec_np)
    failcount = 0
    while True:
        try:
            evals_np, evecs_np = sla.eigsh(L_eigsh, k=k_eig, M=Mmat, sigma=EPS)
            evals_np = np.clip(evals_np, a_min=0.0, a_max=float("inf"))
            break
        except Exception as e:  # noqa: BLE001  (stesso ciclo di compute_operators)
            print(e, flush=True)
            if failcount > 3:
                raise ValueError("failed to compute eigendecomp")
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount), flush=True)
            L_eigsh = L_eigsh + scipy.sparse.identity(L.shape[0]) * (EPS * 10 ** failcount)

    edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype)
    edge_vecs = edge_tangent_vectors(verts, frames, edges)
    grad_mat_np = build_grad(verts, edges, edge_vecs)

    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    L_t = utils.sparse_np_to_torch(L).to(device=device, dtype=dtype)
    evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=dtype)
    gradX = utils.sparse_np_to_torch(np.real(grad_mat_np)).to(device=device, dtype=dtype)
    gradY = utils.sparse_np_to_torch(np.imag(grad_mat_np)).to(device=device, dtype=dtype)
    return frames, massvec, L_t, evals, evecs, gradX, gradY


def coo(t, base: str) -> dict:
    c = t.coalesce()
    return {f"{base}_indices": c.indices().numpy(), f"{base}_values": c.values().numpy(),
            f"{base}_shape": np.array(c.shape, dtype=np.int64)}


def process(path: Path, out_dir: Path, k_eig: int) -> None:
    import torch

    V, F = load_mesh(path)
    A = mesh_area(V.astype(np.float64), F)
    if not np.isfinite(A) or A <= 0:
        raise ValueError(f"area non valida: {A}")
    V1 = unit_area(V.astype(np.float64), F).astype(np.float32)
    F64 = np.asarray(F, dtype=np.int64)
    k_eff = max(1, min(int(k_eig), V1.shape[0] - 2))
    ops = robust_operators(torch.from_numpy(V1), torch.from_numpy(F64), k_eff)
    data = {"verts": V1, "faces": F64, "mass": ops[1].numpy(), "evals": ops[3].numpy(),
            "evecs": ops[4].numpy()}
    data.update(coo(ops[2], "L"))
    data.update(coo(ops[5], "gradX"))
    data.update(coo(ops[6], "gradY"))
    out = out_dir / path.name
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **data)
    os.replace(tmp, out)


def run(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.input_dir.iterdir() if p.suffix == ".npz"
                   and fnmatch.fnmatch(p.name, args.pattern))
    perm = np.random.default_rng(PERM_SEED).permutation(len(files))
    si, sn = (int(x) for x in args.shard.split("/"))
    mine = [files[int(i)] for i in perm[si::sn]]
    todo = [p for p in mine if args.overwrite or not (args.output_dir / p.name).exists()]
    print(f"{len(files)} input, shard {si}/{sn}: {len(mine)}, da calcolare {len(todo)}", flush=True)
    t0, ok, fail = time.time(), 0, 0
    for i, p in enumerate(todo):
        if not args.overwrite and (args.output_dir / p.name).exists():
            continue                                 # scritto nel frattempo da un altro job
        try:
            process(p, args.output_dir, args.k_eig)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"  FAIL {p.name}: {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 10 == 0:
            r = (i + 1) / max(time.time() - t0, 1e-9)
            print(f"  {i+1}/{len(todo)} ok={ok} fail={fail} ({r:.3f}/s)", flush=True)
    print(f"done ok={ok} fail={fail} in {(time.time()-t0)/60:.1f} min", flush=True)
    if fail:
        sys.exit(1)


def check(args) -> None:
    """Chiavi = cartella di riferimento, area 1, stessa mesh (facce, vertici a meno di similitudine),
    k_eig, valori finiti; stampa la somma della massa e lo spostamento dello spettro. Esce 1 se un
    file non torna o se la cartella e' incompleta."""
    names = sorted(p.name for p in args.output_dir.iterdir()
                   if p.suffix == ".npz" and not p.name.startswith(".")
                   and fnmatch.fnmatch(p.name, args.pattern))
    n_in = len([p for p in args.input_dir.iterdir() if p.suffix == ".npz"]) if args.input_dir else None
    print(f"[check] {len(names)} file in {args.output_dir}" + (f" (input {n_in})" if n_in else ""))
    rng = np.random.default_rng(1)
    pick = names if args.check_all else sorted(rng.choice(names, size=min(args.n_check, len(names)),
                                                          replace=False).tolist())
    areas, devs, bad = [], [], 0
    for name in pick:
        with np.load(args.output_dir / name, allow_pickle=False) as z:
            keys = sorted(z.files)
            V, F = z["verts"].astype(np.float64), z["faces"]
            ev, evecs, mass = z["evals"], z["evecs"], z["mass"]
            finite = all(np.isfinite(z[k]).all() for k in ("mass", "evals", "evecs", "L_values",
                                                           "gradX_values", "gradY_values"))
        with np.load(args.ref_dir / name, allow_pickle=False) as r:
            keys_ref = sorted(r.files)
            Vr, Fr, ev_ref = r["verts"].astype(np.float64), r["faces"], r["evals"]
        A = mesh_area(V, F)
        # stessa mesh: stesse facce, e i vertici sono quelli di riferimento centrati e scalati
        Vr1 = unit_area(Vr, Fr)
        dv = float(np.abs(Vr1 - V).max()) if Vr1.shape == V.shape else float("inf")
        # Weyl: evals(area 1) ~ evals_ref * A_ref; il rapporto mediano dice quanto il robusto
        # sposta lo spettro rispetto al cotangente, a parita' di area
        r_med = float(np.median(ev[1:] / np.maximum(ev_ref[1:] * mesh_area(Vr, Fr), 1e-30)))
        # la massa non e' bloccante: col tufted cover la sua somma non e' per forza l'area
        ok = (keys == keys_ref and abs(A - 1.0) < 1e-4 and dv < 1e-5 and np.array_equal(F, Fr)
              and finite and ev.shape == (args.k_eig,) and evecs.shape == (V.shape[0], args.k_eig))
        areas.append(A)
        devs.append(r_med)
        flag = "OK" if ok else "FALLITO"
        print(f"  {name}: {flag} chiavi={'=' if keys == keys_ref else set(keys) ^ set(keys_ref)} "
              f"area={A:.6f} massa={float(mass.sum()):.5f} |dV|max={dv:.1e} k={ev.shape[0]} "
              f"evals/(ref*A_ref) mediana={r_med:.4f}")
        bad += 0 if ok else 1
    print(f"[check] area: min {min(areas):.6f} max {max(areas):.6f}; "
          f"robusto/cotangente (mediana degli autovalori a pari area): {np.median(devs):.4f}")
    if n_in is not None and len(names) != n_in:
        print(f"[check] INCOMPLETO: {len(names)}/{n_in}")
        bad += 1
    print("[check] OK" if bad == 0 else f"[check] FALLITO su {bad}")
    sys.exit(1 if bad else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--k-eig", type=int, default=128)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ref-dir", type=Path, default=None, help="cartella standard, per --check")
    ap.add_argument("--n-check", type=int, default=12)
    ap.add_argument("--check-all", action="store_true")
    args = ap.parse_args()
    if args.check:
        if args.ref_dir is None:
            raise SystemExit("--check vuole --ref-dir")
        check(args)
    else:
        if args.input_dir is None:
            raise SystemExit("serve --input-dir")
        run(args)


if __name__ == "__main__":
    main()
