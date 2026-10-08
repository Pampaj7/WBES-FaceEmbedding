#!/usr/bin/env python3
"""E10 punto 4: throughput, latenza e memoria degli operatori DiffusionNet su GPU.

    VENV=.venv_e10 aau/run.sh aau/evidence/e10_gpu_ops/bench.py --work-dir /tmp/$SLURM_JOB_ID/bench \
        --out-dir aau/runs/evidence/e10/bench [--parts classes,methods,e2e] [--variant bk]

Parti:
  classes  V ~ 3.3k (ICT down8k), 9.4k (ICT original), 24.1k (ICT up60k), 60.4k (BFM up60k), k 64 e 128:
           batch di B mesh diverse della stessa categoria, B crescente fino all'OOM. mesh/s = B / tempo
           medio di R batch dopo uno di riscaldamento (geometria + autovettori + copia su host, mesh gia'
           lette e normalizzate); latenza = tempo per mesh a B=1; memoria = picco del dispositivo
           (torch + cuDSS, cudaMemGetInfo campionata ogni ~2 ms) e picco del solo allocatore torch.
  methods  le altre vie, una mesh per volta: (b) eigh denso fp64/fp32, (a) torch.lobpcg e cupyx lobpcg con
           Jacobi, Chebyshev senza shift, shift-invert + Chebyshev; tempo ed errore sugli autovalori
           rispetto all'eigh denso fp64 (solo V <= 10k).
  e2e      pipeline completa sugli STESSI campioni di E9 (e9_bench/ops_bench.py: sample_200, 3.3k-60k, e
           sample_le10k): lettura+normalizzazione in thread, batch per categoria sulla GPU, npz scritti
           su /tmp in thread e cancellati; mesh/s = mesh / wall, confrontabile con le tabelle E9 (a)/(d).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
DS = REPO_ROOT / "datasets"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "aau" / "evidence" / "e9_bench"))
import gpu_ops  # noqa: E402
from check_ops import VARIANTS  # noqa: E402

CLASSES = {   # nome -> (modello di percorso, n mesh distinte, batch provati)
    "ict_down8k": (DS / "ICT/topo/ict{:04d}_GTready_down8k.npz", 64, (1, 2, 4, 8, 16, 32, 64)),
    "ict_original": (DS / "ICT/topo/ict{:04d}_GTready_original.npz", 64, (1, 2, 4, 8, 16, 32, 64)),
    "ict_up60k": (DS / "ICT/topo/ict{:04d}_GTready_up60k.npz", 32, (1, 2, 4, 8, 16, 32)),
    "bfm_up60k": (DS / "REMESH/npz_data_topo_500/id{:04d}_GTready_up60k.npz", 24, (1, 2, 4, 8)),
}


class MemPeak:
    """Picco della memoria usata sul dispositivo (cudaMemGetInfo), campionato da un thread."""

    def __init__(self):
        self.peak, self._run = 0, False

    def __enter__(self):
        free, total = torch.cuda.mem_get_info()
        self.peak, self.total, self._run = total - free, total, True
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def _poll(self):
        while self._run:
            free, _ = torch.cuda.mem_get_info()
            self.peak = max(self.peak, self.total - free)
            time.sleep(0.002)

    def __exit__(self, *a):
        self._run = False
        self._t.join()


def variant_opts(variant: str, k: int) -> tuple[str, dict]:
    method, opts = VARIANTS[variant]
    opts = dict(opts)
    if "m_factor" in opts:
        opts["m"] = opts.pop("m_factor") * k
    return method, opts


def load_class(pattern: Path, n: int) -> list[dict]:
    """Le prime n identita' che esistono (ICT/topo e BFM hanno buchi nella numerazione)."""
    out, i = [], 0
    while len(out) < n and i < 10000:
        p = Path(str(pattern).format(i))
        if p.exists():
            out.append(gpu_ops.prepare(p))
        i += 1
    return out


def time_batch(meshes, k, method, opts, dev) -> dict:
    st, tm = {}, {}
    torch.cuda.reset_peak_memory_stats()
    with MemPeak() as mp_:
        gpu_ops._sync()
        t0 = time.perf_counter()
        gpu_ops.compute_batch(meshes, k, method, dev, eig_opts=dict(opts, stats=st), timings=tm)
        dt = time.perf_counter() - t0
    return {"s": dt, "dev_peak_gb": mp_.peak / 1e9, "torch_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            **tm, **{a: b for a, b in st.items() if a != "profile"}}


def part_classes(a, dev) -> list[dict]:
    rows = []
    for cname, (pat, n, Bs) in CLASSES.items():
        if a.classes and cname not in a.classes.split(","):
            continue
        t0 = time.perf_counter()
        meshes = load_class(pat, n)
        t_prep = (time.perf_counter() - t0) / len(meshes)
        V = int(np.median([m["verts"].shape[0] for m in meshes]))
        for k in (64, 128):
            method, opts = variant_opts(a.variant, k)
            for B in Bs:
                if B > len(meshes):
                    break
                torch.cuda.empty_cache()
                R = max(1, min(a.reps, len(meshes) // B))
                try:
                    time_batch(meshes[:B], k, method, opts, dev)        # riscaldamento
                    runs = [time_batch(meshes[r * B:(r + 1) * B], k, method, opts, dev) for r in range(R)]
                except torch.cuda.OutOfMemoryError:
                    rows.append({"class": cname, "V": V, "k": k, "B": B, "oom": True})
                    print(f"[classes] {cname} k={k} B={B}: OOM", flush=True)
                    break
                except RuntimeError as exc:
                    if "ALLOC" in str(exc) or "out of memory" in str(exc):
                        rows.append({"class": cname, "V": V, "k": k, "B": B, "oom": True, "err": str(exc)})
                        print(f"[classes] {cname} k={k} B={B}: OOM ({exc})", flush=True)
                        break
                    raise
                s = float(np.mean([r["s"] for r in runs]))
                row = {"class": cname, "V": V, "k": k, "B": B, "R": R, "variant": a.variant,
                       "mesh_per_s": B / s, "s_per_batch": s, "s_per_mesh": s / B, "prepare_s_per_mesh": t_prep,
                       "dev_peak_gb": max(r["dev_peak_gb"] for r in runs),
                       "torch_peak_gb": max(r["torch_peak_gb"] for r in runs),
                       "all_converged": all(r.get("converged", True) for r in runs),
                       "max_res_rel": max(r.get("max_res_rel", 0.0) for r in runs),
                       "phases": {key: float(np.mean([r[key] for r in runs])) for key in
                                  ("geom_s", "eig_s", "d2h_s", "build_s", "factor_s", "iter_s") if key in runs[0]},
                       "runs": runs}
                rows.append(row)
                print(f"[classes] {cname} V={V} k={k} B={B}: {row['mesh_per_s']:.1f} mesh/s, "
                      f"{s / B * 1e3:.0f} ms/mesh, picco {row['dev_peak_gb']:.2f} GB, conv {row['all_converged']}",
                      flush=True)
    return rows


def part_methods(a, dev) -> list[dict]:
    rows = []
    plan = [  # (classe, k, metodo, opzioni, etichetta)
        ("ict_down8k", 64, "dense", {}, "dense_fp64"), ("ict_down8k", 128, "dense", {}, "dense_fp64"),
        ("ict_down8k", 128, "dense", {"dtype": torch.float32}, "dense_fp32"),
        ("ict_original", 64, "dense", {}, "dense_fp64"), ("ict_original", 128, "dense", {}, "dense_fp64"),
        ("ict_original", 128, "dense", {"dtype": torch.float32}, "dense_fp32"),
        ("ict_down8k", 64, "lobpcg", {"niter": 1000}, "torch_lobpcg_jacobi"),
        ("ict_down8k", 128, "lobpcg", {"niter": 1000}, "torch_lobpcg_jacobi"),
        ("ict_original", 128, "lobpcg", {"niter": 1000}, "torch_lobpcg_jacobi"),
        ("ict_down8k", 128, "clobpcg", {"maxiter": 300}, "cupy_lobpcg_jacobi"),
        ("ict_down8k", 128, "chfsi", {"max_outer": 30}, "chebyshev_noshift"),
        ("ict_original", 128, "chfsi", {"max_outer": 30}, "chebyshev_noshift"),
        ("ict_down8k", 128, "si", {}, "shiftinvert_chebyshev"),
        ("ict_original", 128, "si", {}, "shiftinvert_chebyshev"),
        ("ict_down8k", 128, "bk", {}, "shiftinvert_blockkrylov"),
        ("ict_original", 128, "bk", {}, "shiftinvert_blockkrylov"),
    ]
    cache: dict = {}
    for cname, k, method, opts, label in plan:
        if cname not in cache:
            m = load_class(CLASSES[cname][0], 2)
            geo = gpu_ops.geometry_batch(m[1:2], dev)
            A, _, _ = gpu_ops.standard_form(geo)
            w = torch.linalg.eigvalsh(A.to_dense())
            cache[cname] = (m[1:2], w, float(A.to_dense().diagonal().max()))
        meshes, w, _ = cache[cname]
        method_opts = dict(opts)
        torch.cuda.empty_cache()
        try:
            for rep in range(2):            # la prima chiamata paga JIT/handle
                gpu_ops._sync()
                t0 = time.perf_counter()
                st: dict = {}
                if method in ("chfsi", "si", "bk"):
                    method_opts["stats"] = st
                geo = gpu_ops.geometry_batch(meshes, dev)
                out = gpu_ops.EIG[method](geo, k, **method_opts)
                gpu_ops._sync()
                dt = time.perf_counter() - t0
                if dt > 60:
                    break
            lam = out[0][0].double()
            e = ((lam[1:] - w[1:k]).abs() / w[1:k]).cpu().numpy()
            row = {"class": cname, "V": int(meshes[0]["verts"].shape[0]), "k": k, "method": label, "s": dt,
                   "evals_relerr_max": float(e.max()), "evals_relerr_median": float(np.median(e)),
                   **{x: y for x, y in st.items() if x != "profile"}}
        except Exception as exc:  # noqa: BLE001
            row = {"class": cname, "k": k, "method": label, "error": f"{type(exc).__name__}: {exc}"[:300]}
        rows.append(row)
        print(f"[methods] {row}", flush=True)
    for cname, (_, w, dmax) in cache.items():
        rows.append({"class": cname, "spectrum": True, "lambda_1": float(w[1]), "lambda_64": float(w[63]),
                     "lambda_128": float(w[127]), "lambda_max": float(w[-1]), "A_diag_max": dmax})
    return rows


def part_e2e(a, dev) -> list[dict]:
    from ops_bench import sample_200, sample_le10k
    rows = []
    for sname, smp in (("full200", sample_200()), ("le10k400", sample_le10k())):
        gdir = a.work_dir / "e2e_geom" / sname
        gdir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda m: (gdir / m["name"]).exists() or shutil.copyfile(m["src"], gdir / m["name"]), smp))
        cats: dict = {}
        for m in smp:
            cats.setdefault(m["cat"], []).append(gdir / m["name"])
        for k in (64, 128):
            method, opts = variant_opts(a.variant, k)
            # batch per categoria (V dal nome della categoria, misure di "classes"), il piu' grande prima
            bsize = {"down8k": 32, "remesh": 16, "original": 16, "up60k": 8}
            batches = []
            for cat, paths in cats.items():
                B = bsize[cat.split("_", 1)[1]]
                if cat == "bfm_up60k":
                    B = 4
                elif cat in ("bfm_original", "bfm_remesh", "ict_up60k"):
                    B = 8
                for i in range(0, len(paths), B):
                    batches.append(paths[i:i + B])
            batches.sort(key=lambda b: -os.path.getsize(b[0]))
            out_dir = a.work_dir / "e2e_out"
            out_dir.mkdir(parents=True, exist_ok=True)

            def save(o, name):
                p = out_dir / name
                gpu_ops.save_npz(o, p)
                os.unlink(p)
            gpu_ops.compute_batch([gpu_ops.prepare(batches[-1][0])], k, method, dev, eig_opts=dict(opts))  # warmup
            torch.cuda.empty_cache()
            wopts = dict(opts, host_threads=a.host_threads) if method in ("bk", "si") else dict(opts)
            stats_all: list = []
            with ThreadPoolExecutor(a.n_load) as loader, ThreadPoolExecutor(a.n_save) as saver, MemPeak() as mpk:
                t0 = time.perf_counter()
                futs = [[loader.submit(gpu_ops.prepare, p) for p in b] for b in batches]
                work: queue.Queue = queue.Queue()
                for item in zip(batches, futs):
                    work.put(item)
                saves: list = []
                lock = threading.Lock()

                def gpu_worker():
                    # uno stream per lavoratore: l'analisi su host (cuDSS) di uno copre la GPU dell'altro
                    with torch.cuda.stream(torch.cuda.Stream()):
                        while True:
                            try:
                                b, fb = work.get_nowait()
                            except queue.Empty:
                                return
                            meshes = [f.result() for f in fb]
                            st: dict = {}
                            outs = gpu_ops.compute_batch(meshes, k, method, dev, eig_opts=dict(wopts, stats=st))
                            with lock:
                                stats_all.append((len(b), bool(st.get("converged", True))))
                                saves.extend(saver.submit(save, o, p.name) for o, p in zip(outs, b))
                workers = [threading.Thread(target=gpu_worker) for _ in range(a.gpu_workers)]
                for w in workers:
                    w.start()
                for w in workers:
                    w.join()
                for f in saves:
                    f.result()
                wall = time.perf_counter() - t0
            n_mesh = sum(x for x, _ in stats_all)
            n_conv = sum(x for x, c in stats_all if c)
            row = {"sample": sname, "k": k, "variant": a.variant, "n_meshes": n_mesh, "wall_s": wall,
                   "mesh_per_s": n_mesh / wall, "n_batches": len(batches), "n_converged": n_conv,
                   "dev_peak_gb": mpk.peak / 1e9, "n_load_threads": a.n_load, "n_save_threads": a.n_save,
                   "gpu_workers": a.gpu_workers, "cudss_host_threads": a.host_threads,
                   "cpus_in_job": int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))}
            rows.append(row)
            print(f"[e2e] {sname} k={k}: {n_mesh} mesh in {wall:.1f}s = {row['mesh_per_s']:.2f} mesh/s, "
                  f"convergenti {n_conv}/{n_mesh}, picco {row['dev_peak_gb']:.1f} GB", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--parts", default="classes,methods,e2e")
    ap.add_argument("--classes", default="", help="sottoinsieme di CLASSES, separato da virgola")
    ap.add_argument("--variant", default="bk", help="variante di check_ops.VARIANTS per classes/e2e")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-load", type=int, default=6)
    ap.add_argument("--n-save", type=int, default=4)
    ap.add_argument("--gpu-workers", type=int, default=2, help="thread GPU (uno stream ciascuno) nella parte e2e")
    ap.add_argument("--host-threads", type=int, default=8, help="thread host dell'analisi cuDSS per lavoratore")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    gpu_ops.CuDSS.lib()                     # caricata una volta, prima dei thread
    info = {"gpu": torch.cuda.get_device_name(), "host": os.uname().nodename,
            "job": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
            "cpus": os.environ.get("SLURM_CPUS_PER_TASK"), "variant": a.variant}
    print(f"[bench] {info}", flush=True)
    for part in a.parts.split(","):
        rows = {"classes": part_classes, "methods": part_methods, "e2e": part_e2e}[part](a, dev)
        (a.out_dir / f"{part}_{a.variant}{a.tag}.json").write_text(json.dumps({"info": info, "args": vars(a) | {
            "work_dir": str(a.work_dir), "out_dir": str(a.out_dir)}, "rows": rows}))


if __name__ == "__main__":
    main()
