#!/usr/bin/env python3
"""E9 (a) e (d): costo degli operatori DiffusionNet sulle CPU di un nodo L40S.

    aau/run.sh aau/evidence/e9_bench/ops_bench.py --work-dir /tmp/$SLURM_JOB_ID/ops \
        --out-dir aau/runs/evidence/e9/ops [--only nome,nome]

Codice di calcolo: ``aau/data_scale/prepass_ops.py::ops_areanorm``, cioe' quello che produce gli
operatori del run su scala (job 1060130, convenzione ``areanorm``): mesh centrata ad area 1,
``diffusion_net.geometry.compute_operators`` (Laplaciano cotangente di potpourri3d: la riga di
``robust_laplacian`` e' commentata; eigsh shift-invert; ``build_grad``), npz NON compresso con
indici int32. Chiamata cosi' com'e': per configurazione cambiano solo ``prepass_ops.K_EIG`` (128 o
64) e ``geometry.build_grad`` (originale o ``grad_vec.build_grad_vec``).

Profilo per mesh: nei worker un cronometro avvolge ``pp3d.cotan_laplacian`` + ``pp3d.vertex_areas``
("lap"), eigsh ("eigsh"), ``build_grad`` ("grad") e ``np.savez`` ("save"); "rest" = totale meno
questi (lettura, area, frame tangenti, vettori degli archi, conversioni torch).

Throughput: pool spawn di P processi a thread singolo (OMP/MKL/OpenBLAS a 1, come il pre-pass del
trainer, train_steps.py:220), avviati e importati PRIMA del cronometro (barriera); compiti dal piu'
grande, chunksize 1; mesh/s = compiti / wall. Il campione si ripete R volte perche' ogni processo
riceva almeno ~6 compiti: con 2-3 compiti a testa il numero misurerebbe la coda, non il regime.
Le uscite si cancellano subito, tranne la prima ripetizione delle configurazioni ``keep``, che
serve per le dimensioni su disco.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import resource
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
DS = REPO_ROOT / "datasets"
LABELS = ("original", "remesh", "down8k", "up60k")

# campione di (a): 8 categorie x 25 identita' = 200 mesh, V da 3.3k a 60.4k
#   ICT-5000 (datasets/ICT/topo, ict0000..): original 9409, remesh ~6.6k, down8k ~3.3k, up60k ~24.1k
#   BFM-500 (datasets/REMESH/npz_data_topo_500): original 23470, remesh ~16.5k, down8k ~8.1k, up60k ~60.4k
# Le ICT si ricopiano col nome del run su scala (id1NNNN, come ICT/train_ready/npz_withops):
# prepass_ops.domain_of_name vuole un id numerico.


def _mesh(dom: str, i: int, label: str) -> dict:
    if dom == "ict":
        return {"src": DS / f"ICT/topo/ict{i:04d}_GTready_{label}.npz",
                "name": f"id1{i:04d}_GTready_{label}.npz", "cat": f"ict_{label}"}
    return {"src": DS / f"REMESH/npz_data_topo_500/id{i:04d}_GTready_{label}.npz",
            "name": f"id{i:04d}_GTready_{label}.npz", "cat": f"bfm_{label}"}


def sample_200() -> list[dict]:
    return ([_mesh("ict", 200 * j, lab) for lab in LABELS for j in range(25)]
            + [_mesh("bfm", 20 * j, lab) for lab in LABELS for j in range(25)])


def sample_le10k() -> list[dict]:
    """(d), V <= ~10k: ICT original 9409, remesh ~6.6k, down8k ~3.3k e BFM down8k ~8.1k, 100 identita' l'una."""
    return ([_mesh("ict", 50 * j, lab) for lab in ("original", "remesh", "down8k") for j in range(100)]
            + [_mesh("bfm", 5 * j, "down8k") for j in range(100)])


def subset_p1(sample: list[dict], per_cat: int = 5) -> list[dict]:
    """P=1 costa ~27 min sulle 200 mesh: 5 per categoria, stesse proporzioni, quindi mesh/s confrontabili."""
    seen: dict = {}
    out = []
    for m in sample:
        if seen.get(m["cat"], 0) < per_cat:
            seen[m["cat"]] = seen.get(m["cat"], 0) + 1
            out.append(m)
    return out


def stage(meshes: list[dict], dest: Path) -> list[Path]:
    """Geometria su /tmp (la lettura da CephFS non deve entrare nei tempi)."""
    dest.mkdir(parents=True, exist_ok=True)

    def cp(m):
        p = dest / m["name"]
        if not p.exists():
            shutil.copyfile(m["src"], p)
        return p
    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(cp, meshes))


# --- worker ---------------------------------------------------------------------------------

PH: dict = {}


def _timed(name, fn):
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            PH[name] = PH.get(name, 0.0) + time.perf_counter() - t0
    return w


def _init(k_eig: int, grad: str, barrier) -> None:
    sys.path.insert(0, str(REPO_ROOT / "aau" / "data_scale"))
    sys.path.insert(0, str(THIS_DIR))
    import potpourri3d as pp3d
    import scipy.sparse.linalg as sla
    import prepass_ops
    from diffusion_net import geometry

    prepass_ops.K_EIG = int(k_eig)
    if grad == "vec":
        from grad_vec import build_grad_vec
        geometry.build_grad = build_grad_vec
    pp3d.cotan_laplacian = _timed("lap", pp3d.cotan_laplacian)
    pp3d.vertex_areas = _timed("lap", pp3d.vertex_areas)
    sla.eigsh = _timed("eigsh", sla.eigsh)
    geometry.build_grad = _timed("grad", geometry.build_grad)
    np.savez = _timed("save", np.savez)
    barrier.wait()


def _work(task: tuple) -> dict:
    import prepass_ops
    src, out, keep, cat, nv = task
    PH.clear()
    t0 = time.perf_counter()
    err = ""
    try:
        prepass_ops.ops_areanorm(Path(src), Path(out))
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    t = time.perf_counter() - t0
    size = os.path.getsize(out) if os.path.exists(out) else 0
    if not keep and os.path.exists(out):
        os.unlink(out)
    ph = {k: PH.get(k, 0.0) for k in ("lap", "eigsh", "grad", "save")}
    ph["rest"] = t - sum(ph.values())
    return {"name": Path(out).name, "cat": cat, "V": nv, "t": t, **ph, "bytes": size, "err": err,
            "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def run_config(cfg: dict, geom: dict, nverts: dict, out_dir: Path) -> dict:
    meshes, P, R = cfg["meshes"], cfg["P"], cfg["R"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for r in range(R):
        for m in meshes:
            stem = m["name"][:-4]
            tasks.append((str(geom[m["name"]]), str(out_dir / f"{stem}_r{r}.npz"),
                          bool(cfg.get("keep")) and r == 0, m["cat"], nverts[m["name"]]))
    tasks.sort(key=lambda t: -t[4])
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(P + 1)
    t_start = time.time()
    with ctx.Pool(P, initializer=_init, initargs=(cfg["k"], cfg["grad"], barrier)) as pool:
        barrier.wait()
        t_ready = time.time()
        t0 = time.perf_counter()
        res = list(pool.imap_unordered(_work, tasks, chunksize=1))
        wall = time.perf_counter() - t0
    ts = np.array([r["t"] for r in res])
    rep = {k: v for k, v in cfg.items() if k != "meshes"}
    rep.update({"n_tasks": len(tasks), "n_distinct": len(meshes), "wall_s": wall,
                "pool_start_s": t_ready - t_start, "mesh_per_s": len(tasks) / wall,
                "steady_mesh_per_s": P / float(ts.mean()), "cpu_s_per_mesh": float(ts.mean()),
                "n_failed": sum(1 for r in res if r["err"]),
                "failures": [f"{r['name']}: {r['err']}" for r in res if r["err"]][:10],
                "max_worker_rss_mb": max(r["maxrss_mb"] for r in res), "per_mesh": res})
    return rep


# --- dimensioni -----------------------------------------------------------------------------

def _sizes(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    nb = lambda arrs: int(sum(a.nbytes for a in arrs))  # noqa: E731
    same = all(np.array_equal(d["L_indices"], d[f"{g}_indices"]) for g in ("gradX", "gradY"))
    i64 = [d[k].astype(np.int64) if k == "faces" or k.endswith("_indices") else d[k] for k in d]
    e16 = {k: (v.astype(np.float16) if k == "evecs" else v) for k, v in d.items()}
    buf = io.BytesIO()
    np.savez_compressed(buf, **e16)
    fwd = [d["verts"], d["mass"], d["evals"], e16["evecs"], d["gradX_indices"], d["gradX_values"],
           d["gradY_values"]] + ([] if same else [d["gradY_indices"]])
    return {"name": Path(path).name, "V": int(d["verts"].shape[0]), "k": int(d["evals"].shape[0]),
            "nnz": int(d["L_values"].size), "same_sparsity_L_gradX_gradY": bool(same),
            "file_fp32_i32_npz": os.path.getsize(path),
            "fp32_i64": nb(i64), "fp32_i32": nb(d.values()), "e16_i32": nb(e16.values()),
            "e16_i32_zlib": buf.getbuffer().nbytes, "forward_only_e16_i32": nb(fwd),
            "evecs_fp32": int(d["evecs"].nbytes)}


def sizes(out_dir: Path, n_proc: int) -> list[dict]:
    files = sorted(str(p) for p in out_dir.glob("*_r0.npz"))
    with mp.get_context("spawn").Pool(n_proc) as pool:
        return pool.map(_sizes, files, chunksize=1)


# --- configurazioni -------------------------------------------------------------------------

def configs() -> list[dict]:
    full, le10k = sample_200(), sample_le10k()
    p1 = subset_p1(full)
    R = lambda n, P: max(1, math.ceil(6 * P / n))  # noqa: E731
    c = []
    # (a) codice del run su scala, campione completo; P=64 tiene le uscite per le dimensioni
    for k in (128, 64):
        c.append({"name": f"a_orig_k{k}_full_P64", "k": k, "grad": "orig", "P": 64, "meshes": full,
                  "R": R(200, 64), "keep": True})
    # (d) build_grad vettorizzato + k 64 + V <= 10k, P = 32, 64, 100 (R=8: wall >= ~30 s anche a P=100)
    for P in (100, 64, 32):
        c.append({"name": f"d_vec_k64_le10k_P{P}", "k": 64, "grad": "vec", "P": P, "meshes": le10k, "R": 8})
    # (d) scomposizione: senza k 64 (k 128 cambia meno il modello), senza tetto su V, senza vettorizzazione
    c.append({"name": "d_vec_k128_le10k_P100", "k": 128, "grad": "vec", "P": 100, "meshes": le10k, "R": 8})
    c.append({"name": "d_vec_k64_full_P100", "k": 64, "grad": "vec", "P": 100, "meshes": full, "R": 3})
    c.append({"name": "d_vec_k128_full_P100", "k": 128, "grad": "vec", "P": 100, "meshes": full, "R": 3})
    c.append({"name": "d_orig_k64_le10k_P100", "k": 64, "grad": "orig", "P": 100, "meshes": le10k, "R": 3})
    c.append({"name": "a_orig_k128_full_P100", "k": 128, "grad": "orig", "P": 100, "meshes": full,
              "R": R(200, 100)})
    # (d) follow-up: k 128 e campione completo al P migliore di k 64 (P=64 rende piu' di P=100)
    for P in (64, 32):
        c.append({"name": f"d_vec_k128_le10k_P{P}", "k": 128, "grad": "vec", "P": P, "meshes": le10k, "R": 4})
    for k in (64, 128):
        c.append({"name": f"d_vec_k{k}_full_P64", "k": k, "grad": "vec", "P": 64, "meshes": full, "R": 2})
    # (a) scalatura
    for P in (32, 16):
        for k in (128, 64):
            c.append({"name": f"a_orig_k{k}_full_P{P}", "k": k, "grad": "orig", "P": P, "meshes": full,
                      "R": R(200, P)})
    for k in (128, 64):
        c.append({"name": f"a_orig_k{k}_p1sub_P1", "k": k, "grad": "orig", "P": 1, "meshes": p1, "R": 1})
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--only", default="", help="nomi di configurazione separati da virgola")
    ap.add_argument("--max-p", type=int, default=0, help="salta le configurazioni con P maggiore")
    ap.add_argument("--tag", default="", help="suffisso dei JSON (ripetizioni della stessa configurazione)")
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    a.out_dir.mkdir(parents=True, exist_ok=True)

    cfgs = configs()
    if a.only:
        keep = set(a.only.split(","))
        cfgs = [c for c in cfgs if c["name"] in keep]
    if a.max_p:
        cfgs = [c for c in cfgs if c["P"] <= a.max_p]
    allm = {m["name"]: m for c in cfgs for m in c["meshes"]}
    t0 = time.time()
    paths = stage(list(allm.values()), a.work_dir / "geom")
    geom = {p.name: p for p in paths}
    nverts = {}
    for p in paths:
        with np.load(p) as z:
            nverts[p.name] = int((z["verts"] if "verts" in z.files else z["V"]).shape[0])
    print(f"[ops] {len(paths)} mesh su /tmp in {time.time() - t0:.0f}s; configurazioni: "
          f"{[c['name'] for c in cfgs]}", flush=True)

    for cfg in cfgs:
        od = a.work_dir / cfg["name"]
        rep = run_config(cfg, geom, nverts, od)
        if cfg.get("keep"):
            rep["sizes"] = sizes(od, min(cfg["P"], os.cpu_count() or 1))
        shutil.rmtree(od, ignore_errors=True)
        (a.out_dir / f"{cfg['name']}{a.tag}.json").write_text(json.dumps(rep))
        print(f"[ops] {cfg['name']}: {rep['n_tasks']} compiti in {rep['wall_s']:.1f}s = "
              f"{rep['mesh_per_s']:.2f} mesh/s (regime {rep['steady_mesh_per_s']:.2f}), "
              f"{rep['cpu_s_per_mesh']:.2f} s/mesh, rss max {rep['max_worker_rss_mb']:.0f} MB, "
              f"fallite {rep['n_failed']}", flush=True)


if __name__ == "__main__":
    main()
