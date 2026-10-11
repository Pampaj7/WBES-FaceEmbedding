#!/usr/bin/env python3
"""Controlli delle viste di Ava-256 contro i domini di sviluppo: sola geometria delle viste, una mesh per volta (nessun
metodo, nessuna distanza fra soggetti, nessuna metrica di prestazione).

    v3_work/unified_gt/run.sh aau/ava256/ava_view_qc.py [--ava-topo datasets/AVA256/topo] [--out ...]

Le quantita' e le definizioni sono quelle dei controlli del critic dell'11 ottobre (check_ava3.py, check_ava6.py,
scratchpad della sessione), qui nel repo:
  - remesh: spostamento dei vertici interni della original dopo i 2 giri di smoothing di ``make_remesh``
    (``mesh_ops.smooth_simple``), scomposto sulla normale al vertice: RMS normale e tangenziale, mm;
  - lato medio (media dei tre lati di ogni triangolo) della original e di down8k, mm;
  - anelli di bordo per topologia.
Insiemi: HIFI3D, FaceVerse, FaceScape dev = le prime 10 original della vista di eval in ordine di nome (prime 5 per gli
anelli), coordinate x u del frame di E12 (``aau/runs/evidence/e12/frames.json``); Ava-256 = le prime 10 identita' della
cartella delle topologie, mm. Esito: ogni quantita' di Ava-256 dentro o fuori dall'intervallo [min, max] dei tre domini.
Uscita: ``aau/ava256/view_qc.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (THIS_DIR, REPO_ROOT / "v3_work" / "unified_gt", REPO_ROOT / "v2_work" / "genict"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ava_common as ac  # noqa: E402
import mesh_ops as mo  # noqa: E402
import ugt as C  # noqa: E402
from make_ict_topologies import REMESH_SMOOTH_ITERS, VARIANTS  # noqa: E402

FRAMES = REPO_ROOT / "aau" / "runs" / "evidence" / "e12" / "frames.json"
DEV = {"hifi3d": ("HIFI3D", "id9000"), "faceverse": ("FACEVERSE_ZS", "id9100"), "facescape": ("DEV_FACESCAPE", "id9300")}
N_MESH, N_LOOPS = 10, 5
OUT = THIS_DIR / "view_qc.json"


def vnormals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def mean_edge(V: np.ndarray, F: np.ndarray) -> float:
    e = np.concatenate([V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 1]], V[F[:, 0]] - V[F[:, 2]]])
    return float(np.linalg.norm(e, axis=1).mean())


def n_loops(F: np.ndarray) -> int:
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    _, inv, cnt = np.unique(np.sort(E, 1), axis=0, return_inverse=True, return_counts=True)
    nxt = {}
    for a, b in E[cnt[inv.ravel()] == 1]:
        nxt.setdefault(int(a), []).append(int(b))
    n, seen = 0, set()
    for s0 in nxt:
        if s0 in seen:
            continue
        v = s0
        while v not in seen:
            seen.add(v)
            v = nxt[v][0]
        n += 1
    return n


def remesh_smoothing(path: Path, u: float) -> tuple[float, float, float]:
    """(RMS normale, RMS tangenziale, lato medio) della original: i 2 giri di smoothing di make_remesh, vertici interni."""
    with np.load(path) as q:
        V, F = mo.as_arrays(q["V"], q["F"])
    V, F = mo.prepare_open_surface(V, F)
    Vs = mo.smooth_simple(V, F, REMESH_SMOOTH_ITERS)
    inner = ~C.boundary_vertices(F, len(V))
    d = (Vs - V)[inner] * u
    n = vnormals(V, F)[inner]
    dn = (d * n).sum(1)
    return (float(np.sqrt((dn ** 2).mean())), float(np.sqrt(((d - dn[:, None] * n) ** 2).sum(1).mean())),
            mean_edge(V, F) * u)


def measure(files: list[Path], u: float, name_of) -> dict:
    nrm, tan, e_o, e_d = [], [], [], []
    for f in files[:N_MESH]:
        a, b, c = remesh_smoothing(f, u)
        nrm.append(a)
        tan.append(b)
        e_o.append(c)
        with np.load(name_of(f, "down8k")) as q:
            e_d.append(mean_edge(q["V"].astype(np.float64), q["F"]) * u)
    loops = {}
    for t in VARIANTS:
        loops[t] = [n_loops(np.load(name_of(f, t))["F"]) for f in files[:N_LOOPS]]
    med = lambda x: round(float(np.median(x)), 3)  # noqa: E731
    return {"n": min(len(files), N_MESH), "smooth_normal_rms_mm": med(nrm), "smooth_tangential_rms_mm": med(tan),
            "mean_edge_original_mm": med(e_o), "mean_edge_down8k_mm": med(e_d), "boundary_loops": loops}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ava-topo", type=Path, default=ac.DATA_ROOT / "topo")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    frames = json.loads(FRAMES.read_text())["domains"]
    out = {"definition": __doc__.split("Uscita:")[0].strip(), "domains": {}}
    for d, (root, pre) in DEV.items():
        files = sorted((ac.REPO_ROOT / "datasets" / root / "eval_view" / "npz").glob(f"{pre}*_GTready_original.npz"))
        out["domains"][d] = measure(files, frames[d]["u"], lambda f, t: Path(str(f).replace("_original", f"_{t}")))
    files = sorted(a.ava_topo.glob("ava[0-9]*_GTready_original.npz"))
    out["domains"]["ava256"] = measure(files, 1.0, lambda f, t: Path(str(f).replace("_original", f"_{t}")))
    dev = [out["domains"][d] for d in DEV]
    checks = {}
    for k in ("smooth_normal_rms_mm", "mean_edge_original_mm", "mean_edge_down8k_mm", "smooth_tangential_rms_mm"):
        lo, hi = min(x[k] for x in dev), max(x[k] for x in dev)
        v = out["domains"]["ava256"][k]
        checks[k] = {"ava256": v, "dev_min": lo, "dev_max": hi, "inside": bool(lo <= v <= hi)}
    out["checks"] = checks
    out["ava_topo"] = str(a.ava_topo)
    C.save_json(a.out, out)
    print(json.dumps({"checks": checks, "loops": {d: v["boundary_loops"] for d, v in out["domains"].items()}}, indent=1),
          flush=True)


if __name__ == "__main__":
    main()
