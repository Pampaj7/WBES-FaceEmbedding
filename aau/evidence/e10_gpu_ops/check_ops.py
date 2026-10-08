#!/usr/bin/env python3
"""E10 punto 3: operatori GPU contro CPU su 60 mesh, fino agli embedding del checkpoint e108.

    VENV=.venv_e10 aau/run.sh aau/evidence/e10_gpu_ops/check_ops.py \
        --work-dir /tmp/$SLURM_JOB_ID/check --out-json aau/runs/evidence/e10/check.json

Campione: BFM (datasets/REMESH/npz_data_topo_500) e ICT (datasets/ICT/topo, ricopiate col nome
id1NNNN come in e9_bench) x {original, remesh, down8k, up60k, noisy, crop} x 5 identita' = 60 mesh.

Varianti (una directory di npz ciascuna, stesso formato di prepass_ops.ops_areanorm):
  cpu        ops_areanorm con grad_vec (identico all'originale dopo il cast fp32, E9): riferimento;
  cpu_v0     idem ma eigsh con un altro vettore iniziale v0: il "rumore" del solo risolutore CPU;
  gpu_<nome> gpu_ops.compute_batch, una variante per risolutore/parametri (``VARIANTS``), batch =
             le 5 mesh di una categoria.

Per ogni mesh e variante: struttura ed errore di L, massa, gradX/gradY; errore relativo sugli
autovalori (i >= 1); angoli principali fra i sottospazi M-ortonormali (tutti i k, e i primi k-8 GPU
dentro i k CPU); embedding e108 col loader congelato su CPU (come e9_bench/grad_vec.embed_view):
|z_var - z_cpu| rispetto alla distanza mediana fra coppie di embedding CPU.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
DS = REPO_ROOT / "datasets"
LABELS = ("original", "remesh", "down8k", "up60k", "noisy", "crop")
CKPT = REPO_ROOT / ("aau/runs/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411/"
                    "mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_"
                    "sig5e-4-2e-2_latentnoise_seed1234__3167d36d/checkpoints/epoch108.pth")

# risolutori GPU confrontati: nome -> (metodo, opzioni)
VARIANTS = {
    "bk": ("bk", {}),                                    # Krylov a blocchi: bs 16, m max(4k, k+256), tol 1e-5
    "bk_1pass": ("bk", {"bs": 32, "m_factor": 3, "max_restart": 0}),     # bs 32, m 3k, una passata
    "bk_loose": ("bk", {"bs": 32, "m_factor": 2, "max_restart": 0}),     # bs 32, m 2k, una passata
    "si": ("si", {}),                                                    # shift-invert + Chebyshev, tol 1e-5
}


def sample(n_ids: int) -> list[dict]:
    """5 identita' per dominio con tutte e 6 le etichette, distanziate nell'elenco."""
    out = []
    for dom, d, pre in (("bfm", DS / "REMESH/npz_data_topo_500", "id"), ("ict", DS / "ICT/topo", "ict")):
        ids = sorted({p.name.split("_GTready_")[0] for p in d.glob(f"{pre}*_GTready_crop.npz")})
        ids = [i for i in ids if all((d / f"{i}_GTready_{lab}.npz").exists() for lab in LABELS)]
        pick = [ids[int(j * len(ids) / n_ids)] for j in range(n_ids)]
        for lab in LABELS:
            for i in pick:
                name = f"id1{i[3:]}_GTready_{lab}.npz" if dom == "ict" else f"{i}_GTready_{lab}.npz"
                out.append({"src": d / f"{i}_GTready_{lab}.npz", "name": name, "cat": f"{dom}_{lab}"})
    return out


# --- riferimento CPU (processi spawn, thread singolo) ----------------------------------------

def _cpu_init(v0_seed: int | None) -> None:
    sys.path.insert(0, str(REPO_ROOT / "aau" / "data_scale"))
    sys.path.insert(0, str(REPO_ROOT / "aau" / "evidence" / "e9_bench"))
    import grad_vec
    grad_vec.install()
    if v0_seed is not None:
        import scipy.sparse.linalg as sla
        orig = sla.eigsh

        def eigsh_v0(A, *a, **kw):
            kw.setdefault("v0", np.random.RandomState(v0_seed).rand(A.shape[0]))
            return orig(A, *a, **kw)
        sla.eigsh = eigsh_v0


def _cpu_work(task) -> tuple[str, float, str]:
    import prepass_ops
    src, out, k = task
    prepass_ops.K_EIG = k
    if os.path.exists(out):             # rilancio: le uscite sono scritte in modo atomico
        return Path(src).name, 0.0, ""
    t0 = time.perf_counter()
    try:
        prepass_ops.ops_areanorm(Path(src), Path(out))
        return Path(src).name, time.perf_counter() - t0, ""
    except Exception as exc:  # noqa: BLE001
        return Path(src).name, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"


def run_cpu(geom: list[Path], out_dir: Path, k: int, n_proc: int, v0_seed: int | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in out_dir.glob(".*.tmp.npz"):     # resti di un'esecuzione interrotta
        t.unlink()
    # il piu' grande prima; worker a thread singolo come il pre-pass (i processi spawn ereditano l'ambiente)
    tasks = sorted([(str(p), str(out_dir / p.name), k) for p in geom], key=lambda t: -os.path.getsize(t[0]))
    saved = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    os.environ.update({v: "1" for v in saved})
    try:
        with mp.get_context("spawn").Pool(n_proc, initializer=_cpu_init, initargs=(v0_seed,)) as pool:
            res = pool.map(_cpu_work, tasks, chunksize=1)
    finally:
        for v, x in saved.items():
            if x is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = x
    return {n: {"t": t, "err": e} for n, t, e in res}


# --- GPU --------------------------------------------------------------------------------------

def run_gpu(geom: list[Path], cats: dict, out_dir: Path, k: int, method: str, opts: dict) -> dict:
    import torch
    sys.path.insert(0, str(THIS_DIR))
    import gpu_ops

    out_dir.mkdir(parents=True, exist_ok=True)
    opts = dict(opts)
    if "m_factor" in opts:
        opts["m"] = opts.pop("m_factor") * k
    by_cat: dict = {}
    for p in geom:
        by_cat.setdefault(cats[p.name], []).append(p)
    rep = {}
    dev = torch.device("cuda")
    for cat, paths in sorted(by_cat.items()):
        meshes = [gpu_ops.prepare(p) for p in paths]
        st, tm = {}, {}
        t0 = time.perf_counter()
        outs = gpu_ops.compute_batch(meshes, k, method, dev, eig_opts=dict(opts, stats=st), timings=tm)
        dt = time.perf_counter() - t0
        for p, o in zip(paths, outs):
            gpu_ops.save_npz(o, out_dir / p.name)
            rep[p.name] = {"batch_s": dt, "B": len(paths), **{a: b for a, b in st.items() if a != "profile"}}
        print(f"[gpu {method}] {cat}: B={len(paths)} {dt:.2f}s {st.get('converged')} "
              f"res {st.get('max_res_rel', float('nan')):.1e}", flush=True)
    return rep


# --- confronto --------------------------------------------------------------------------------

def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b).max() / max(np.abs(b).max(), 1e-300))


def compare(ref: Path, got: Path) -> dict:
    with np.load(ref) as z:
        r = {k: z[k] for k in z.files}
    with np.load(got) as z:
        g = {k: z[k] for k in z.files}
    out = {"V": int(r["verts"].shape[0])}
    for name in ("L", "gradX", "gradY"):
        same = r[f"{name}_indices"].shape == g[f"{name}_indices"].shape and \
            np.array_equal(r[f"{name}_indices"], g[f"{name}_indices"])
        out[f"{name}_same_structure"] = bool(same)
        out[f"{name}_maxrel"] = _rel(g[f"{name}_values"], r[f"{name}_values"]) if same else float("nan")
    out["mass_maxrel"] = _rel(g["mass"], r["mass"])
    lr, lg = r["evals"].astype(np.float64), g["evals"].astype(np.float64)
    e = np.abs(lg[1:] - lr[1:]) / lr[1:]
    out.update({"evals_relerr_max": float(e.max()), "evals_relerr_median": float(np.median(e)),
                "eval0_abs": float(abs(lg[0] - lr[0])), "evals_identical_fp32": bool(np.array_equal(lr, lg))})
    M = r["mass"].astype(np.float64)
    Pr, Pg = r["evecs"].astype(np.float64), g["evecs"].astype(np.float64)
    k = Pr.shape[1]
    sv = np.linalg.svd(Pr.T @ (M[:, None] * Pg), compute_uv=False)
    ang = np.degrees(np.arccos(np.clip(sv, 0.0, 1.0)))
    # i primi k-8 GPU dentro lo span dei k CPU: insensibile al mescolamento al bordo dello spettro
    sv8 = np.linalg.svd(Pr.T @ (M[:, None] * Pg[:, :k - 8]), compute_uv=False)
    out.update({"angle_max_deg": float(ang.max()), "angle_median_deg": float(np.median(ang)),
                "angle_first_k-8_into_k_max_deg": float(np.degrees(np.arccos(np.clip(sv8.min(), 0.0, 1.0)))),
                "lambda_k_gap_rel": float((lr[k - 1] - lr[k - 2]) / lr[k - 1])})
    return out


def embed_view(view: Path, ckpt: Path, n_threads: int) -> tuple[list[str], np.ndarray]:
    """Embedding col loader congelato su CPU, come e9_bench/grad_vec.embed_view."""
    import torch
    from types import SimpleNamespace
    sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"))
    from robustness.data_utils import GTReadyDataset, sample_to_device
    from robustness.model_helpers import build_model, forward_model
    from robustness.posthoc_runner import load_checkpoint_bundle, merge_run_args

    torch.set_num_threads(n_threads)
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
    return [Path(f).name for f in ds.files], np.stack(Z)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n-ids", type=int, default=5)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--n-proc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    a = ap.parse_args()

    meshes = sample(a.n_ids)
    geom_dir = a.work_dir / "geom"
    geom_dir.mkdir(parents=True, exist_ok=True)
    geom = []
    for m in meshes:
        p = geom_dir / m["name"]
        if not p.exists():
            shutil.copyfile(m["src"], p)
        geom.append(p)
    cats = {m["name"]: m["cat"] for m in meshes}
    print(f"[check] {len(geom)} mesh, k={a.k}, varianti GPU {a.variants}", flush=True)

    rep: dict = {"k": a.k, "ckpt": str(a.ckpt), "meshes": {m["name"]: {"cat": m["cat"], "src": str(m["src"])}
                                                            for m in meshes}, "runs": {}}
    t0 = time.time()
    rep["runs"]["cpu"] = run_cpu(geom, a.work_dir / "cpu", a.k, a.n_proc, None)
    rep["runs"]["cpu_v0"] = run_cpu(geom, a.work_dir / "cpu_v0", a.k, a.n_proc, 1)
    print(f"[check] CPU in {time.time() - t0:.0f}s", flush=True)
    variants = ["cpu_v0"]
    for v in a.variants.split(","):
        method, opts = VARIANTS[v]
        rep["runs"][f"gpu_{v}"] = run_gpu(geom, cats, a.work_dir / f"gpu_{v}", a.k, method, opts)
        variants.append(f"gpu_{v}")
    import torch
    rep["gpu"] = torch.cuda.get_device_name()

    # operatori
    rep["ops"] = {v: {p.name: compare(a.work_dir / "cpu" / p.name, a.work_dir / v / p.name) for p in geom}
                  for v in variants}
    # embedding
    files, Zr = embed_view(a.work_dir / "cpu", a.ckpt, a.n_proc)
    iu = np.triu_indices(len(Zr), 1)
    med = float(np.median(np.linalg.norm(Zr[:, None] - Zr[None], axis=-1)[iu]))
    nn = np.sort(np.linalg.norm(Zr[:, None] - Zr[None], axis=-1), axis=1)[:, 1]
    rep["embedding"] = {"files": files, "median_pairwise_dist": med,
                        "nn_dist_min": float(nn.min()), "nn_dist_median": float(np.median(nn))}
    for v in variants:
        f2, Zv = embed_view(a.work_dir / v, a.ckpt, a.n_proc)
        assert f2 == files, (v, f2[:3], files[:3])
        dz = np.linalg.norm(Zv - Zr, axis=1)
        cos = (Zv * Zr).sum(1) / (np.linalg.norm(Zv, axis=1) * np.linalg.norm(Zr, axis=1))
        rep["embedding"][v] = {"dz": dz.tolist(), "dz_max": float(dz.max()), "dz_median": float(np.median(dz)),
                               "dz_max_rel_median_pair": float(dz.max() / med), "cos_min": float(cos.min()),
                               "max_abs_component": float(np.abs(Zv - Zr).max())}
        print(f"[check] embedding {v}: max |dz| {dz.max():.2e} (mediana {np.median(dz):.2e}), "
              f"relativo alla distanza mediana {dz.max() / med:.2e}, cos min {cos.min():.8f}", flush=True)
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(rep))
    for v in variants:
        o = rep["ops"][v].values()
        print(f"[check] {v}: evals relerr max {max(x['evals_relerr_max'] for x in o):.2e}, "
              f"angolo max {max(x['angle_max_deg'] for x in o):.2e} deg, primi k-8 {max(x['angle_first_k-8_into_k_max_deg'] for x in o):.2e} deg, "
              f"L/grad struttura {all(x['L_same_structure'] and x['gradX_same_structure'] for x in o)}, "
              f"grad maxrel {max(max(x['gradX_maxrel'], x['gradY_maxrel']) for x in o):.1e}", flush=True)


if __name__ == "__main__":
    main()
