#!/usr/bin/env python3
"""Studio misurato delle opzioni per moltiplicare i dati senza farli stare in 1 TB.

    aau/submit.sh data_scale/study_options.sbatch

Campione: 10 identita' ICT held-out (ict4500-ict4509) e 5 BFM (id0000-id0004), 6 topologie
ciascuna = 90 mesh, operatori ad area unitaria (la convenzione del modello congiunto). Tutto
cio' che pesa viene scritto in ``--work-dir`` (il /tmp del nodo); su CephFS va solo il JSON.

Varianti degli operatori, tutte lette dal loader CONGELATO (``GTReadyDatasetNPZ``):
  ref      i file in uso oggi (areanorm, k_eig 128, float32, indici int64)
  i32      ref con facce e indici COO in int32, npz compresso (senza perdita)
  e16      i32 con ``evecs`` in float16
  a16      e16 con anche i valori di L, gradX, gradY in float16
  rec128   operatori RICALCOLATI ora dalla geometria, k_eig 128: e' cio' che producono le
           opzioni (b) e (c); misura quanto un ricalcolo riproduce i file in uso
  k64      ricalcolati con k_eig 64
  k64e16   k64 con ``evecs`` in float16

Per ciascuna, contro ref: byte per mesh su disco, tempo del loader per campione, e l'uscita
del modello congiunto (``x3dmm_joint_bfm_ict_s1234_1019532``, epoch120):
  * ``dz_rel``: ||z_var - z_ref|| / mediana delle ||z_i - z_j|| di ref, per mesh;
  * Spearman fra le distanze latenti di ref e della variante, su tutte le coppie di mesh
    di identita' diverse dello stesso dominio;
  * Spearman latente-GT (la metrica del paper, a livello di coppia di mesh) con ref e con
    la variante: la differenza e' quanto cambierebbe un numero gia' pubblicato.

Tempi del calcolo al volo (b): ogni ricalcolo registra il tempo per mesh di un processo a
thread singolo; il throughput con N processi e' misurato a parte su 24 mesh ICT, N = 1, 4,
8, 16.  Geometria (c): byte per mesh di V float32 + F int32 compressi, e la rigenerazione
di 2 identita' ICT dai soli pesi (tempo e uguaglianza con la geometria su disco).
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"))
sys.path.insert(0, str(REPO_ROOT / "v2_work/genict"))
sys.path.insert(0, str(REPO_ROOT / "v2_work/potential"))
sys.path.insert(0, str(REPO_ROOT / "diffusion-net/src"))

DS = REPO_ROOT / "datasets"
TOPOS = ("original", "remesh", "crop", "noisy", "down8k", "up60k")
SAMPLE = {
    "ict": {"ids": [f"ict{i:04d}" for i in range(4500, 4510)],
            "ops": DS / "ICT/topo_withops", "geom": DS / "ICT/topo",
            "gt": DS / "ICT/gt/ict_matrix_distances_maxabs.npz"},
    "bfm": {"ids": [f"id{i:04d}" for i in range(5)],
            "ops": DS / "REMESH/npz_data_topo_500_withops_areanorm",
            "geom": DS / "REMESH/npz_data_topo_500",
            "gt": REPO_ROOT / "face_embedding/gt_encdec/autoencoder/latent_analysis"
                              "/gt_distance_matrix/normalized_matrix_distances.npz"},
}
CKPT_GLOB = "aau/runs/x3dmm_joint_bfm_ict_s1234_1019532/*/checkpoints/epoch120.pth"
SPARSE = ("L", "gradX", "gradY")


# --- codifiche ------------------------------------------------------------------------------

def load_dict(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def encode(d: dict, evecs16: bool = False, sparse16: bool = False) -> dict:
    d = dict(d)
    d["faces"] = np.asarray(d["faces"], dtype=np.int32)
    for b in SPARSE:
        d[f"{b}_indices"] = d[f"{b}_indices"].astype(np.int32)
        if sparse16:
            d[f"{b}_values"] = d[f"{b}_values"].astype(np.float16)
    if evecs16:
        d["evecs"] = d["evecs"].astype(np.float16)
    return d


def recompute(task: tuple[str, int]) -> tuple[str, dict, float]:
    """Operatori areanorm dalla geometria su disco, come v2_work/potential/areanorm_operators.py."""
    import torch
    from areanorm_operators import total_area
    from diffusion_net.geometry import compute_operators
    from potential_operators import load_mesh

    path, k = task
    t0 = time.time()
    V, F = load_mesh(Path(path))
    V = (V - V.mean(0)) / np.sqrt(total_area(V, F))
    _, mass, L, evals, evecs, gX, gY = compute_operators(
        torch.tensor(V, dtype=torch.float32), torch.tensor(F, dtype=torch.int32), k_eig=k)
    out = {"verts": V.astype(np.float32), "faces": F.astype(np.int64), "mass": mass.numpy(),
           "evals": evals.numpy(), "evecs": evecs.numpy()}
    for name, t in zip(SPARSE, (L, gX, gY)):
        c = t.coalesce()
        out[f"{name}_indices"] = c.indices().numpy()
        out[f"{name}_values"] = c.values().numpy()
        out[f"{name}_shape"] = np.array(c.shape)
    return path, out, time.time() - t0


def run_pool(tasks, n: int):
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(n) as pool:
        res = pool.map(recompute, tasks, chunksize=1)
    return res, time.time() - t0


# --- modello e metriche ---------------------------------------------------------------------

def embed(view: Path, model, dev) -> tuple[list[str], np.ndarray, float]:
    import torch
    from robustness.data_utils import GTReadyDataset, sample_to_device
    from robustness.model_helpers import forward_model

    ds = GTReadyDataset(str(view))
    Z, t_load = [], 0.0
    with torch.no_grad():
        for i in range(len(ds.files)):
            t0 = time.time()
            s = ds[i]
            t_load += time.time() - t0
            if s is None:
                raise RuntimeError(f"il loader congelato rifiuta {ds.files[i]}")
            sd = sample_to_device(s, dev)
            z, _ = forward_model(model, sd, sd["verts"], return_gate_info=False, add_noise=False)
            Z.append(z.reshape(-1).cpu().numpy())
    return ds.files, np.stack(Z), t_load / len(ds.files)


def gt_lookup(domain: str) -> tuple[dict, np.ndarray]:
    with np.load(SAMPLE[domain]["gt"]) as z:
        # BFM: "id0000_GTready"; ICT: "ict0000"
        names = [str(n).split("_GTready")[0] for n in z["names"]]
        D = z["D_orig"]
    pos = {n: i for i, n in enumerate(names)}
    ids = SAMPLE[domain]["ids"]
    idx = [pos[i] for i in ids]
    return {sid: j for j, sid in enumerate(ids)}, D[np.ix_(idx, idx)].astype(np.float64)


def metrics(files, Zr, Zv, gts) -> dict:
    from scipy.stats import spearmanr

    sid = [f.split("_GTready")[0] for f in files]
    dom = ["ict" if s.startswith("ict") else "bfm" for s in sid]
    Dr = np.linalg.norm(Zr[:, None] - Zr[None], axis=-1)
    Dv = np.linalg.norm(Zv[:, None] - Zv[None], axis=-1)
    iu = np.triu_indices(len(files), 1)
    med = float(np.median(Dr[iu]))
    dz = np.linalg.norm(Zv - Zr, axis=1) / med
    out = {"dz_rel_median": float(np.median(dz)), "dz_rel_max": float(dz.max())}
    for d in ("ict", "bfm"):
        pairs = [(i, j) for i, j in zip(*iu) if dom[i] == d and dom[j] == d and sid[i] != sid[j]]
        a = np.array([Dr[i, j] for i, j in pairs])
        b = np.array([Dv[i, j] for i, j in pairs])
        sidx, G = gts[d]
        g = np.array([G[sidx[sid[i]], sidx[sid[j]]] for i, j in pairs])
        out[d] = {"n_pairs": len(pairs),
                  "spearman_ref_vs_var": float(spearmanr(a, b).statistic),
                  "max_rel_dist_change": float(np.max(np.abs(b - a) / a)),
                  "spearman_gt_ref": float(spearmanr(a, g).statistic),
                  "spearman_gt_var": float(spearmanr(b, g).statistic)}
    return out


def write_view(view: Path, items: dict) -> dict:
    """items: nome -> dict di array. Scrive npz compressi (np.savez per 'ref'). Ritorna i byte."""
    view.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, d in items.items():
        np.savez_compressed(view / name, **d)
        sizes[name] = (view / name).stat().st_size
    return sizes


# --- (c) geometria ----------------------------------------------------------------------------

def regen_ict(sid: str) -> dict:
    """Rigenera le 6 topologie di una identita' ICT-5000 dai soli pesi e le confronta col disco."""
    import mesh_ops as mo
    from ict_model import ict_shape_mesh, load_ict
    from make_ict_topologies import (make_down8k, make_noisy, make_remesh, make_up60k,
                                     triangle_targets)

    w = json.loads((DS / "ICT/identities/identity_weights.json").read_text())[sid]
    t0 = time.time()
    V0, F0 = ict_shape_mesh(np.asarray(w), load_ict())
    base = mo.prepare_open_surface(*mo.as_arrays(V0, F0))
    dt, ut = triangle_targets(len(base[1]))
    built = {"original": base, "remesh": make_remesh(*base), "crop": mo.make_crop(*base),
             "noisy": make_noisy(*base, seed=int(sid[-4:])),
             "down8k": make_down8k(*base, target=dt), "up60k": make_up60k(*base, target=ut)}
    secs = time.time() - t0
    same = {}
    for t, (V, F) in built.items():
        with np.load(DS / f"ICT/topo/{sid}_GTready_{t}.npz") as z:
            same[t] = {"faces_equal": bool(np.array_equal(z["F"], F)),
                       "verts_maxdiff": float(np.abs(z["V"] - V.astype(np.float32)).max())
                       if z["V"].shape == V.shape else "shape diversa"}
    return {"sid": sid, "seconds": secs, "vs_disk": same}


def geom_bytes(path: Path) -> int:
    with np.load(path) as z:
        V = z["V"] if "V" in z.files else z["verts"]
        F = z["F"] if "F" in z.files else z["faces"]
    buf = io.BytesIO()
    np.savez_compressed(buf, V=np.asarray(V, np.float32), F=np.asarray(F, np.int32))
    return len(buf.getvalue())


def main() -> None:
    import glob
    import torch
    from types import SimpleNamespace
    from robustness.model_helpers import build_model
    from robustness.posthoc_runner import load_checkpoint_bundle, merge_run_args

    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--n-cores", type=int, default=16)
    a = ap.parse_args()
    a.work_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"sample": {d: SAMPLE[d]["ids"] for d in SAMPLE}}

    names, ops_src, geom_src = [], {}, {}
    for d, s in SAMPLE.items():
        for sid in s["ids"]:
            for t in TOPOS:
                n = f"{sid}_GTready_{t}.npz"
                names.append(n)
                ops_src[n] = s["ops"] / n
                geom_src[n] = s["geom"] / n

    # ref: symlink ai file in uso (zero byte scritti), byte su disco cosi' come sono oggi
    ref_view = a.work_dir / "ref"
    ref_view.mkdir()
    disk = {"ref": {}}
    for n in names:
        (ref_view / n).symlink_to(ops_src[n])
        disk["ref"][n] = ops_src[n].stat().st_size

    # ricalcoli: k 128 e 64 con n-cores processi; tempo per mesh a processo singolo
    rec, timing = {}, {}
    for k in (128, 64):
        res, wall = run_pool([(str(geom_src[n]), k) for n in names], a.n_cores)
        rec[k] = {Path(p).name: d for p, d, _ in res}
        timing[f"k{k}"] = {"wall_s": wall, "n_meshes": len(names), "n_proc": a.n_cores,
                           "per_mesh_s": {Path(p).name: s for p, _, s in res}}
        print(f"[rec k={k}] {len(names)} mesh in {wall:.0f}s con {a.n_cores} processi", flush=True)

    # throughput con N processi su 24 mesh ICT (4 identita' x 6 topologie)
    sub = [(str(geom_src[n]), 128) for n in names if n.startswith("ict")][:24]
    scaling = {}
    for nproc in (1, 4, 8, 16):
        _, wall = run_pool(sub, nproc)
        scaling[nproc] = {"wall_s": wall, "meshes_per_s": len(sub) / wall}
        print(f"[scaling] N={nproc}: {len(sub) / wall:.2f} mesh/s", flush=True)
    timing["scaling_ict_k128"] = scaling

    variants = {
        "i32": lambda n: encode(load_dict(ops_src[n])),
        "e16": lambda n: encode(load_dict(ops_src[n]), evecs16=True),
        "a16": lambda n: encode(load_dict(ops_src[n]), evecs16=True, sparse16=True),
        "rec128": lambda n: encode(rec[128][n]),
        "k64": lambda n: encode(rec[64][n]),
        "k64e16": lambda n: encode(rec[64][n], evecs16=True),
    }

    ckpt = Path(glob.glob(str(REPO_ROOT / CKPT_GLOB))[0])
    margs = SimpleNamespace(**merge_run_args(ckpt, ""))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args=margs, device=dev)
    model.load_state_dict(load_checkpoint_bundle(ckpt)["state_dict"], strict=True)
    model.eval()
    torch.set_num_threads(a.n_cores)   # i pool sopra sono a thread singolo, il forward no
    gts = {d: gt_lookup(d) for d in SAMPLE}

    files, Zr, tl = embed(ref_view, model, dev)
    report["model"] = {"ckpt": str(ckpt), "device": str(dev)}
    report["ref"] = {"loader_s_per_sample": tl}
    shutil.rmtree(ref_view)
    for v, make in variants.items():
        view = a.work_dir / v
        disk[v] = write_view(view, {n: make(n) for n in names})
        f2, Zv, tl = embed(view, model, dev)
        assert f2 == files
        report[v] = {"loader_s_per_sample": tl, **metrics(files, Zr, Zv, gts)}
        shutil.rmtree(view)
        print(f"[{v}] " + json.dumps(report[v]), flush=True)

    # byte per mesh e per identita' (somma delle 6 topologie), per dominio e variante
    by = {}
    for v, sizes in disk.items():
        for d in SAMPLE:
            per_id = [sum(sizes[f"{sid}_GTready_{t}.npz"] for t in TOPOS) for sid in SAMPLE[d]["ids"]]
            per_topo = {t: float(np.mean([sizes[f"{sid}_GTready_{t}.npz"] for sid in SAMPLE[d]["ids"]]))
                        for t in TOPOS}
            by.setdefault(v, {})[d] = {"bytes_per_identity": float(np.mean(per_id)),
                                       "bytes_per_mesh_mean": float(np.mean(per_id)) / 6,
                                       "bytes_per_topology": per_topo}
    for d in SAMPLE:
        g = [geom_bytes(geom_src[f"{sid}_GTready_{t}.npz"]) for sid in SAMPLE[d]["ids"] for t in TOPOS]
        by.setdefault("geom_only", {})[d] = {"bytes_per_identity": float(np.sum(g)) / len(SAMPLE[d]["ids"]),
                                             "bytes_per_mesh_mean": float(np.mean(g))}
    report["bytes"] = by

    # tempo per mesh del ricalcolo, per topologia e dominio (processo singolo dentro il pool)
    per = {}
    for k in ("k128", "k64"):
        for d in SAMPLE:
            per.setdefault(k, {})[d] = {
                t: float(np.mean([timing[k]["per_mesh_s"][f"{sid}_GTready_{t}.npz"]
                                  for sid in SAMPLE[d]["ids"]])) for t in TOPOS}
            per[k][d]["per_identity_s"] = float(sum(per[k][d][t] for t in TOPOS))
    timing["per_mesh_by_topology"] = per
    for k in ("k128", "k64"):
        timing[k].pop("per_mesh_s")
    report["timing"] = timing

    report["regen_geometry_ict"] = [regen_ict(s) for s in ("ict4500", "ict4501")]
    report["host"] = os.uname().nodename
    a.out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("bytes", "timing")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
