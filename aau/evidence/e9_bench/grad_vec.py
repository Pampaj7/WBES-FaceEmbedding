#!/usr/bin/env python3
"""``build_grad`` vettorizzato (E9 punto d): stessa matrice di diffusion-net, senza loop Python.

L'originale (diffusion-net/src/diffusion_net/geometry.py:209-273) risolve per ogni vertice i un
minimo quadrato 2x2. Con e_j i vettori tangenti degli archi uscenti (coordinate nel frame di i):
A_i = sum_j e_j e_j^T + eps I, coefficiente dell'arco i->j = A_i^-1 e_j (parte reale = x, immaginaria
= y), coefficiente diagonale = -sum_j A_i^-1 e_j. Qui le somme per vertice sono ``np.bincount``
nell'ordine degli archi (lo stesso della lista per vertice dell'originale) e A_i^-1 e' la formula
chiusa 2x2. Struttura identica: una voce diagonale per OGNI vertice (0 se non ha archi) piu' gli
archi con coda != punta.

Non modifica diffusion-net: ``install()`` sostituisce ``geometry.build_grad`` nel solo processo
corrente (``compute_operators`` la risolve fra i globali del modulo a ogni chiamata).

    aau/run.sh aau/evidence/e9_bench/grad_vec.py --out-json aau/runs/evidence/e9/grad_check.json \
        --work-dir /tmp/$SLURM_JOB_ID/gradcheck

``main`` = controllo su 20 mesh: errore massimo assoluto e relativo di gradX/gradY contro
l'originale (stessi archi e vettori tangenti, quindi stesso eigsh), tempo dei due ``build_grad`` a
thread singolo, ed embedding del checkpoint e108 del run su scala con i due operatori, letti dal
loader congelato.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "diffusion-net" / "src"))

EPS_REG = 1e-5   # == eps_reg di geometry.build_grad


def build_grad_vec(verts, edges, edge_tangent_vectors):
    """Drop-in di ``geometry.build_grad``: stessi argomenti, stessa csc complessa (V, V)."""
    from diffusion_net.utils import toNP

    e = np.asarray(toNP(edges), dtype=np.int64)
    t = np.asarray(toNP(edge_tangent_vectors), dtype=np.float64)
    n = int(verts.shape[0])
    keep = e[0] != e[1]
    tail, tip, t = e[0, keep], e[1, keep], t[keep]
    tx, ty = t[:, 0], t[:, 1]
    a = np.bincount(tail, weights=tx * tx, minlength=n) + EPS_REG
    b = np.bincount(tail, weights=tx * ty, minlength=n)
    d = np.bincount(tail, weights=ty * ty, minlength=n) + EPS_REG
    det = a * d - b * b
    # A^-1 = [[d, -b], [-b, a]] / det, applicata a ogni e_j
    cx = (d[tail] * tx - b[tail] * ty) / det[tail]
    cy = (a[tail] * ty - b[tail] * tx) / det[tail]
    diag = -(np.bincount(tail, weights=cx, minlength=n) + 1j * np.bincount(tail, weights=cy, minlength=n))
    rows = np.concatenate([np.arange(n), tail])
    cols = np.concatenate([np.arange(n), tip])
    vals = np.concatenate([diag, cx + 1j * cy])
    return scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()


def install() -> None:
    """``compute_operators`` usera' ``build_grad_vec`` in questo processo."""
    from diffusion_net import geometry
    geometry.build_grad = build_grad_vec


# --- controllo ------------------------------------------------------------------------------

def _to_torch_coo(mat):
    """Come compute_operators: parte reale/immaginaria -> sparse torch float32 coalesced."""
    from diffusion_net import utils
    return (utils.sparse_np_to_torch(np.real(mat)).float().coalesce(),
            utils.sparse_np_to_torch(np.imag(mat)).float().coalesce())


def _errors(ref, got) -> dict:
    same = bool(ref.indices().shape == got.indices().shape and torch_equal(ref.indices(), got.indices()))
    r = ref.values().double().numpy()
    g = got.values().double().numpy()
    d = np.abs(r - g)
    nz = np.abs(r) > 0
    return {"same_structure": same, "nnz": int(r.size), "max_abs": float(d.max()),
            "max_abs_over_max_ref": float(d.max() / np.abs(r).max()),
            "max_rel_elementwise": float((d[nz] / np.abs(r[nz])).max()) if nz.any() else 0.0}


def torch_equal(a, b) -> bool:
    import torch
    return bool(torch.equal(a, b))


def check_one(src: Path, out_dir: Path) -> dict:
    """Operatori areanorm (come prepass_ops.ops_areanorm) con l'originale; poi il vettorizzato
    sugli STESSI archi e vettori tangenti. Scrive due npz che differiscono solo in gradX/gradY."""
    import torch
    from diffusion_net import geometry
    sys.path.insert(0, str(REPO_ROOT / "v2_work" / "potential"))
    from areanorm_operators import total_area
    from potential_operators import load_mesh

    orig = geometry.build_grad
    seen: dict = {}

    def spy(verts, edges, etv):
        seen["args"] = (verts, edges, etv)
        t0 = time.perf_counter()
        m = orig(verts, edges, etv)
        seen["t_orig"] = time.perf_counter() - t0
        seen["m"] = m
        return m

    V, F = load_mesh(src)
    V = (V - V.mean(0)) / np.sqrt(total_area(V, F))
    geometry.build_grad = spy
    try:
        _, mass, L, evals, evecs, gX, gY = geometry.compute_operators(
            torch.tensor(V, dtype=torch.float32), torch.tensor(F, dtype=torch.int32), k_eig=128)
    finally:
        geometry.build_grad = orig
    t0 = time.perf_counter()
    m_vec = build_grad_vec(*seen["args"])
    t_vec = time.perf_counter() - t0
    gXv, gYv = _to_torch_coo(m_vec)
    # anche in float64, prima del cast a float32 di compute_operators
    m_ref = seen["m"].tocsc()
    m_ref.sort_indices()
    m_vec.sort_indices()
    d64 = np.abs((m_ref - m_vec).data).max() if (m_ref - m_vec).nnz else 0.0
    rep = {"name": src.name, "V": int(V.shape[0]), "t_orig_s": seen["t_orig"], "t_vec_s": t_vec,
           "complex128": {"same_structure": bool(np.array_equal(m_ref.indptr, m_vec.indptr)
                                                 and np.array_equal(m_ref.indices, m_vec.indices)),
                          "max_abs": float(d64), "max_abs_over_max_ref": float(d64 / np.abs(m_ref.data).max())},
           "gradX": _errors(gX.coalesce(), gXv), "gradY": _errors(gY.coalesce(), gYv)}

    base = {"verts": V.astype(np.float32), "faces": F.astype(np.int32), "mass": mass.numpy(),
            "evals": evals.numpy(), "evecs": evecs.numpy()}
    for tag, gx, gy in (("orig", gX, gY), ("vec", gXv, gYv)):
        data = dict(base)
        for name, t in zip(("L", "gradX", "gradY"), (L, gx, gy)):
            c = t.coalesce()
            data[f"{name}_indices"] = c.indices().numpy().astype(np.int32)
            data[f"{name}_values"] = c.values().numpy()
            data[f"{name}_shape"] = np.array(c.shape)
        (out_dir / tag).mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / tag / src.name, **data)
    return rep


def embed_view(view: Path, ckpt: Path) -> tuple[list[str], np.ndarray]:
    """Embedding col loader congelato, come aau/data_scale/study_options.py::embed."""
    import torch
    from types import SimpleNamespace
    sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"))
    from robustness.data_utils import GTReadyDataset, sample_to_device
    from robustness.model_helpers import build_model, forward_model
    from robustness.posthoc_runner import load_checkpoint_bundle, merge_run_args

    dev = torch.device("cpu")
    model = build_model(args=SimpleNamespace(**merge_run_args(ckpt, "")), device=dev)
    model.load_state_dict(load_checkpoint_bundle(ckpt)["state_dict"], strict=True)
    model.eval()
    ds = GTReadyDataset(str(view))
    Z = []
    with torch.no_grad():
        for i in range(len(ds.files)):
            sd = sample_to_device(ds[i], dev)
            z, _ = forward_model(model, sd, sd["verts"], return_gate_info=False, add_noise=False)
            Z.append(z.reshape(-1).numpy())
    return list(ds.files), np.stack(Z)


def main() -> None:
    sys.path.insert(0, str(THIS_DIR))
    from ops_bench import sample_200, stage

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=REPO_ROOT / (
        "aau/runs/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411/"
        "mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_"
        "sig5e-4-2e-2_latentnoise_seed1234__3167d36d/checkpoints/epoch108.pth"))
    ap.add_argument("--n-meshes", type=int, default=20)
    a = ap.parse_args()

    # 20 mesh: le prime identita' di ogni categoria del campione di (a), a turno fra le 8 categorie
    by_cat: dict = {}
    for m in sample_200():
        by_cat.setdefault(m["cat"], []).append(m)
    picks = [by_cat[c][i] for i in range(25) for c in sorted(by_cat)][:a.n_meshes]
    geom = stage(picks, a.work_dir / "geom")
    meshes = [check_one(p, a.work_dir) for p in geom]
    for m in meshes:
        print(f"[grad] {m['name']} V={m['V']} orig {m['t_orig_s']:.3f}s vec {m['t_vec_s']:.4f}s "
              f"maxabs X {m['gradX']['max_abs']:.2e} Y {m['gradY']['max_abs']:.2e} "
              f"c128 {m['complex128']['max_abs']:.2e}", flush=True)

    f_o, Zo = embed_view(a.work_dir / "orig", a.ckpt)
    f_v, Zv = embed_view(a.work_dir / "vec", a.ckpt)
    assert f_o == f_v
    dz = np.linalg.norm(Zv - Zo, axis=1)
    iu = np.triu_indices(len(Zo), 1)
    med = float(np.median(np.linalg.norm(Zo[:, None] - Zo[None], axis=-1)[iu]))
    rep = {
        "ckpt": str(a.ckpt), "meshes": meshes,
        "worst": {k: max(m[g][k] for m in meshes for g in ("gradX", "gradY"))
                  for k in ("max_abs", "max_abs_over_max_ref", "max_rel_elementwise")},
        "all_same_structure": all(m[g]["same_structure"] for m in meshes for g in ("gradX", "gradY", "complex128")),
        "worst_complex128": {k: max(m["complex128"][k] for m in meshes) for k in ("max_abs", "max_abs_over_max_ref")},
        "speedup_total": sum(m["t_orig_s"] for m in meshes) / sum(m["t_vec_s"] for m in meshes),
        "embedding": {"max_abs_component": float(np.abs(Zv - Zo).max()),
                      "max_dz": float(dz.max()), "median_pairwise_dist": med,
                      "max_dz_rel": float(dz.max() / med), "files": f_o},
    }
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("meshes", "embedding")}, indent=1))
    print(json.dumps({k: v for k, v in rep["embedding"].items() if k != "files"}, indent=1))


if __name__ == "__main__":
    main()
