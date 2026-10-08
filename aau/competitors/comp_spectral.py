#!/usr/bin/env python3
"""Descrittori spettrali globali (ShapeDNA, HKS, WKS) delle 600 mesh valutate di un dominio zero-shot.

    aau/run.sh aau/competitors/comp_spectral.py --view-dir datasets/HIFI3D/eval_view/npz \\
        --out aau/runs/competitors_hifi3d/desc_spectral.npz
    (comp_spectral.sbatch)

Parametri fissati in ``aau/runs/competitors_hifi3d/protocol.md`` prima del calcolo; qui solo
come costanti. Operatore = quello di ``diffusion_net.geometry.compute_operators`` (cotangente di
potpourri3d, massa concentrata + eps, eigsh shift-invert), ricopiato senza gli operatori gradiente
che qui non servono, sulla mesh scalata ad area 1.

Scrive per mesh: autovalori (101), HKS e WKS globali (100 ciascuno), numero di componenti
connesse, vertici tolti perche' non referenziati, e lo scarto massimo dall'identita'
media-per-area = somma spettrale (controllo: autovettori M-ortonormali).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "zs3dmm"))

from zs_stage import TOPOLOGIES, select_subjects  # noqa: E402

N_EIG = 101                 # lambda_0..lambda_100
EPS = 1e-8                  # come compute_operators
WEYL = 4.0 * np.pi          # lambda_k ~ 4 pi k su area 1
HKS_T = np.geomspace(4 * np.log(10) / (WEYL * 100), 4 * np.log(10) / WEYL, 100)
_E_MIN, _E_MAX = np.log(WEYL), np.log(WEYL * 100)
WKS_SIGMA = 7 * (_E_MAX - _E_MIN) / 100
WKS_E = np.linspace(_E_MIN + 2 * WKS_SIGMA, _E_MAX - 2 * WKS_SIGMA, 100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eval-seed", type=int, default=1234, help="WBES_EVAL_SEED: scelta dei soggetti")
    p.add_argument("--limit", type=int, default=0, help="solo le prime N mesh (prova)")
    return p.parse_args()


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path) as z:
        V, F = z["V"].astype(np.float64), z["F"].astype(np.int64)
    used = np.unique(F)
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F], len(V) - len(used)


def spectral(path: Path) -> dict:
    import potpourri3d as pp3d
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    from scipy.sparse.csgraph import connected_components

    V, F, n_dropped = load_mesh(path)
    tri = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = 0.5 * np.linalg.norm(tri, axis=1).sum()
    V = V / np.sqrt(area)                                   # area totale 1

    L = pp3d.cotan_laplacian(V, F, denom_eps=1e-10)
    mass = pp3d.vertex_areas(V, F)
    mass = mass + EPS * np.mean(mass)
    evals, evecs = sla.eigsh((L + sp.identity(L.shape[0]) * EPS).tocsc(), k=N_EIG, M=sp.diags(mass), sigma=EPS)
    order = np.argsort(evals)
    evals, evecs = np.clip(evals[order], 0.0, None), evecs[:, order]

    hks_pt = (evecs ** 2) @ np.exp(-np.outer(evals, HKS_T))                      # (n, T)
    coef = np.exp(-(WKS_E[None, :] - np.log(evals[1:, None])) ** 2 / (2 * WKS_SIGMA ** 2))  # (100, E)
    wks_pt = (evecs[:, 1:] ** 2) @ coef                                           # (n, E)
    m = mass / mass.sum()
    hks, wks = m @ hks_pt, m @ wks_pt
    # Identita': sum_x m_x phi_i(x)^2 = 1 -> media per area = somma spettrale / area della massa
    hks_id = np.exp(-np.outer(HKS_T, evals)).sum(1) / mass.sum()
    wks_id = coef.sum(0) / mass.sum()
    ident = max(np.abs(hks / hks_id - 1).max(), np.abs(wks / wks_id - 1).max())

    adj = sp.coo_matrix((np.ones(3 * len(F)), (F.ravel(), np.roll(F, 1, axis=1).ravel())), shape=(len(V),) * 2)
    n_comp = connected_components(adj, directed=False)[0]
    return {"evals": evals, "hks": hks, "wks": wks, "n_components": n_comp, "n_dropped": n_dropped,
            "area_raw": area, "n_verts": len(V), "identity_rel_err": ident}


def _task(item):
    k, path = item
    return k, spectral(path)


def main() -> None:
    args = parse_args()
    subjects = select_subjects(args.view_dir, args.eval_seed)
    keys = [(s, t) for s in subjects for t in TOPOLOGIES]
    if args.limit:
        keys = keys[: args.limit]
    items = [(k, args.view_dir / f"{k[0]}_GTready_{k[1]}.npz") for k in keys]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    print(f"[comp-spec] {len(items)} mesh, {len(subjects)} soggetti (primi {subjects[:3]}), {workers} processi",
          flush=True)
    res = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for n, (k, r) in enumerate(pool.imap_unordered(_task, items), 1):
            res[k] = r
            if n % 60 == 0 or n == len(items):
                print(f"[comp-spec] {n}/{len(items)}", flush=True)
    rows = [res[k] for k in keys]
    out = {"subjects": np.asarray([k[0] for k in keys]), "topologies": np.asarray([k[1] for k in keys]),
           "hks_t": HKS_T, "wks_e": WKS_E, "wks_sigma": WKS_SIGMA}
    for f in ("evals", "hks", "wks"):
        out[f] = np.stack([r[f] for r in rows])
    for f in ("n_components", "n_dropped", "area_raw", "n_verts", "identity_rel_err"):
        out[f] = np.asarray([r[f] for r in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"[comp-spec] lambda_0 max {out['evals'][:, 0].max():.2e}, lambda_1 min {out['evals'][:, 1].min():.3f}, "
          f"lambda_100 mediano {np.median(out['evals'][:, 100]):.1f} (Weyl {WEYL * 100:.1f})", flush=True)
    print(f"[comp-spec] componenti >1 in {(out['n_components'] > 1).sum()} mesh, vertici tolti in "
          f"{(out['n_dropped'] > 0).sum()} mesh, identita' media-per-area max rel err "
          f"{out['identity_rel_err'].max():.2e}", flush=True)
    for t in TOPOLOGIES:
        sel = out["topologies"] == t
        if sel.any():
            print(f"[comp-spec] {t:<8} n_verts mediano {int(np.median(out['n_verts'][sel])):>6} "
                  f"area mediana {np.median(out['area_raw'][sel]):.4f} "
                  f"lambda_1 mediano {np.median(out['evals'][sel, 1]):.2f}", flush=True)
    print(f"[comp-spec] scritto {args.out}", flush=True)


if __name__ == "__main__":
    main()
