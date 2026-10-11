#!/usr/bin/env python3
"""D4 (``aau/runs/evidence/diagnostics/PROTOCOL_D4.md``): ridiscretizzazione canonica delle viste di test.

    aau/diagnostics/d4_run.sh aau/diagnostics/d4_remesh.py remesh --workers 63      (d4.sbatch, passo prep; pymeshlab)
    aau/run.sh aau/diagnostics/d4_remesh.py check                                   (dopo gli operatori)
    aau/run.sh aau/diagnostics/d4_remesh.py tasks --ops-root <T>                    (d4_embed.sbatch, code GPU)

``remesh`` (sez. 1): per vista (FaceScape dev, HIFI3D, FaceVerse neutra) le 6 topologie dei 100 soggetti valutati
(``subjects.json`` dello store ufficiale di factorized s1234, ``bp_e3.store`` importato in sola lettura), remesh
isotropico esplicito di pymeshlab a lato fisso ``L_MM`` in mm (u_d di ``e12/frames.json``), componente piu' grande se
ne escono piu' d'una, vertici non referenziati tolti; FaceVerse con le facce invertite DOPO il remesh (``zs_stage.py
--flip-faces``). Mesh in ``datasets/DIAG_D4/<vista>/in`` (coordinate e unita' native), statistiche per mesh in
``d4/remesh_<vista>.npz`` (fuori da git) e ``d4/remesh_stats.json``; determinismo (K3) sulle prime ``N_DET`` mesh di
HIFI3D, ridiscretizzate due volte.

``check`` (K4): per mesh, operatori presenti con k = 128 autovalori finiti e lo stesso numero di vertici della mesh
canonica; validita' (una componente, nessun vertice isolato, operatori) in ``d4/remesh_stats.json``.

``tasks``: righe ``ops|tabella|dist_npz|checkpoint|uscita`` degli embedding (3 viste x 3 bracci, e il controllo della
catena K2: factorized s1234 sulla cache dello store di HIFI3D), con gli operatori copiati in ``<ops-root>/<vista>``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import diag  # noqa: E402

sys.path.insert(0, str(diag.REPO / "aau/baselines_param"))
import bp_e3  # noqa: E402  (sola lettura: store e chiavi degli store)

D4 = diag.EV / "d4"
STAGE = diag.REPO / "datasets/DIAG_D4"
OPS = diag.REPO / "datasets/V3_OPS_CACHE/diag_d4"
ST = diag.TV3 / "factorized/scale_tables"
# vista -> tabella di scala ufficiale (dominio), dominio dei frame di E12 (u_d)
VIEWS = {"facescape": {"table": ST / "devfs_eval.npz", "frame": "facescape"},
         "hifi3d": {"table": ST / "hifi3d_eval.npz", "frame": "hifi3d"},
         "faceverse_neutral": {"table": diag.REPO / "aau/runs/evidence/faceverse_neutral/scale_tables/fv_eval.npz",
                               "frame": "faceverse"}}
TOPOLOGIES = bp_e3.TOPOLOGIES                                  # crop, down8k, noisy, original, remesh, up60k
# (chiave, variante del tag dello store, epoca): i bracci di D4 (fact_paired.MODELS)
ARMS = (("factorized_s1234", "factorized", "072"), ("factorized_s2345", "factorizeds2345", "072"),
        ("ctrlfr_s1234", "ctrlfr", "072"))
L_MM = 2.0                    # lato bersaglio in mm (sez. 1: mediana del training 2.04 mm)
ITERS = 10
N_DET = 10
COLUMNS = ("n_vertices_in", "n_vertices_out", "n_faces_out", "components", "removed_vertices", "nonmanifold_edges",
           "boundary_edges", "edge_median_mm", "edge_cv", "area_in_mm2", "area_out_mm2", "hausdorff_mean_mm",
           "hausdorff_max_mm", "seconds")
_C: dict = {}


def u_of(view: str) -> float:
    """mm per unita' nativa (``e12/frames.json``, u_d)."""
    return float(json.loads(diag.FRAMES.read_text())["domains"][VIEWS[view]["frame"]]["u"])


def store_info(view: str) -> dict:
    """``subjects.json`` + ``eval_key.txt`` dello store di factorized s1234 (vista, soggetti, verso, GT)."""
    return bp_e3.store_key(bp_e3.store(view, "factorized", "072"))


def domain_of(view: str) -> str:
    with np.load(VIEWS[view]["table"], allow_pickle=True) as z:
        d = sorted({str(x) for x in z["domain"]})
    if len(d) != 1:
        raise SystemExit(f"{view}: domini {d} nella tabella ufficiale")
    return d[0]


def load_vf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        if "V" in z:
            return np.asarray(z["V"]), np.asarray(z["F"])
        return np.asarray(z["verts"]), np.asarray(z["faces"])


def area(V: np.ndarray, F: np.ndarray) -> float:
    t = V[F]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def edge_counts(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    return np.unique(E, axis=0, return_counts=True)


def components(F: np.ndarray, n: int) -> np.ndarray:
    """Etichetta di componente per vertice (grafo degli spigoli)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)), directed=False)[1]


def canonical(V: np.ndarray, F: np.ndarray, u: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remesh canonico (sez. 1) di una mesh in unita' native: (V, F, statistiche). Nessuna etichetta."""
    import pymeshlab as ml
    t0 = time.perf_counter()
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(V, np.float64), face_matrix=np.asarray(F, np.int32)))
    ms.meshing_isotropic_explicit_remeshing(iterations=ITERS, adaptive=False, selectedonly=False,
                                            targetlen=ml.PureValue(L_MM / u), featuredeg=30.0, checksurfdist=True,
                                            splitflag=True, collapseflag=True, swapflag=True, smoothflag=True,
                                            reprojectflag=True)
    m = ms.current_mesh()
    V2, F2 = m.vertex_matrix().astype(np.float64), m.face_matrix().astype(np.int64)
    lab = components(F2, len(V2))
    n_comp = len(np.unique(lab[np.unique(F2)]))
    if n_comp > 1:                                   # componente piu' grande per facce
        big = np.bincount(lab[F2[:, 0]]).argmax()
        F2 = F2[lab[F2[:, 0]] == big]
    used = np.unique(F2)
    removed = len(V2) - len(used)
    if removed:
        remap = -np.ones(len(V2), dtype=np.int64)
        remap[used] = np.arange(len(used))
        V2, F2 = V2[used], remap[F2]
    secs = time.perf_counter() - t0
    ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(V, np.float64), face_matrix=np.asarray(F, np.int32)))
    h = ms.get_hausdorff_distance(sampledmesh=1, targetmesh=0, samplevert=True, sampleedge=False, sampleface=False,
                                  samplenum=len(V))       # vertici d'ingresso -> superficie canonica (prima della pulizia)
    E, cnt = edge_counts(F2)
    el = np.linalg.norm(V2[E[:, 0]] - V2[E[:, 1]], axis=1) * u
    st = {"n_vertices_in": len(V), "n_vertices_out": len(V2), "n_faces_out": len(F2), "components": n_comp,
          "removed_vertices": removed, "nonmanifold_edges": int((cnt > 2).sum()), "boundary_edges": int((cnt == 1).sum()),
          "edge_median_mm": float(np.median(el)), "edge_cv": float(el.std() / el.mean()),
          "area_in_mm2": area(np.asarray(V, np.float64), np.asarray(F, np.int64)) * u * u,
          "area_out_mm2": area(V2, F2) * u * u, "hausdorff_mean_mm": float(h["mean"]) * u,
          "hausdorff_max_mm": float(h["max"]) * u, "seconds": secs}
    return V2, F2, st


def _one(task):
    src, dst = task
    try:
        V0, F0 = load_vf(Path(src))
        V, F, st = canonical(V0, F0, _C["u"])
        if _C["flip"]:                               # zs_stage.py --flip-faces, dopo il remesh
            F = np.ascontiguousarray(F[:, ::-1])
        np.savez(dst, V=V.astype(V0.dtype), F=F.astype(F0.dtype))
        return [st[c] for c in COLUMNS], ""
    except Exception as exc:  # noqa: BLE001  (mesh NaN, contata)
        return [np.nan] * len(COLUMNS), f"{type(exc).__name__}: {exc}"


def summarize(topo: np.ndarray, S: np.ndarray) -> dict:
    out = {}
    for t in TOPOLOGIES:
        k = topo == t
        col = {c: S[k, i] for i, c in enumerate(COLUMNS)}
        out[t] = {"n": int(k.sum()), "vertices_in_median": float(np.nanmedian(col["n_vertices_in"])),
                  "vertices_out_median": float(np.nanmedian(col["n_vertices_out"])),
                  "vertices_out_min": float(np.nanmin(col["n_vertices_out"])),
                  "vertices_out_max": float(np.nanmax(col["n_vertices_out"])),
                  "edge_median_mm": float(np.nanmedian(col["edge_median_mm"])),
                  "edge_cv_median": float(np.nanmedian(col["edge_cv"])),
                  "area_out_over_in_median": float(np.nanmedian(col["area_out_mm2"] / col["area_in_mm2"])),
                  "area_out_over_in_min": float(np.nanmin(col["area_out_mm2"] / col["area_in_mm2"])),
                  "hausdorff_mean_mm_median": float(np.nanmedian(col["hausdorff_mean_mm"])),
                  "hausdorff_max_mm_p90": float(np.nanpercentile(col["hausdorff_max_mm"], 90)),
                  "seconds_median": float(np.nanmedian(col["seconds"])),
                  "seconds_p90": float(np.nanpercentile(col["seconds"], 90)),
                  "multi_component": int((col["components"] > 1).sum()),
                  "with_nonmanifold_edges": int((col["nonmanifold_edges"] > 0).sum())}
    return out


def run_remesh(views: list, workers: int) -> None:
    D4.mkdir(parents=True, exist_ok=True)
    path = D4 / "remesh_stats.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary["definition"] = {"filter": "pymeshlab meshing_isotropic_explicit_remeshing", "L_mm": L_MM,
                             "iterations": ITERS, "featuredeg": 30.0, "maxsurfdist": "default (1% bbox)",
                             "pymeshlab": _pymeshlab_version()}
    for view in views:
        t0 = time.time()
        sk = store_info(view)
        vdir, flip = Path(sk["view_dir"]), bool(sk["flip_faces"])
        items = [(s, t) for s in sorted(sk["subjects"]) for t in TOPOLOGIES]
        out = STAGE / view / "in"
        out.mkdir(parents=True, exist_ok=True)
        for stale in out.glob("*.npz"):
            stale.unlink()
        (STAGE / view / "domain").write_text(domain_of(view) + "\n")
        _C.update(u=u_of(view), flip=flip)
        tasks = [(str(vdir / f"{s}_GTready_{t}.npz"), str(out / f"{s}_GTready_{t}.npz")) for s, t in items]
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_one, tasks, chunksize=1)
        S = np.asarray([r[0] for r in res], np.float64)
        failed = [f"{s}|{t}|{r[1]}" for (s, t), r in zip(items, res) if r[1]]
        np.savez(D4 / f"remesh_{view}.npz", subjects=np.asarray([s for s, _ in items]),
                 topologies=np.asarray([t for _, t in items]), stats=S, columns=np.asarray(COLUMNS),
                 failed=np.asarray(failed, dtype="U300"))
        summary[view] = {"n_meshes": len(items), "u_mm_per_unit": _C["u"], "flip_faces": flip, "view_dir": str(vdir),
                         "domain": domain_of(view), "failed": failed, "seconds": time.time() - t0,
                         "by_topology": summarize(np.asarray([t for _, t in items]), S)}
        if view == "hifi3d":                         # K3: determinismo
            again = [_one((src, dst + ".det.npz")) for src, dst in tasks[:N_DET]]
            same = []
            for (src, dst), r in zip(tasks[:N_DET], again):
                a, b = load_vf(Path(dst)), load_vf(Path(dst + ".det.npz"))
                same.append(bool(r[1] == "" and a[0].shape == b[0].shape and (a[0] == b[0]).all()
                                 and a[1].shape == b[1].shape and (a[1] == b[1]).all()))
                Path(dst + ".det.npz").unlink(missing_ok=True)
            summary["K3_determinism"] = {"meshes": [Path(d).name for _, d in tasks[:N_DET]], "identical": same,
                                         "pass": bool(all(same))}
        path.write_text(json.dumps(summary, indent=1) + "\n")
        bt = summary[view]["by_topology"]
        print(f"[d4-remesh] {view}: {len(items)} mesh in {time.time() - t0:.0f}s, fallite {len(failed)}; vertici "
              + ", ".join(f"{t} {bt[t]['vertices_out_median']:.0f}" for t in TOPOLOGIES) + "; lato mm "
              + ", ".join(f"{t} {bt[t]['edge_median_mm']:.2f}" for t in TOPOLOGIES)
              + f"; piu' componenti {sum(bt[t]['multi_component'] for t in TOPOLOGIES)}, con spigoli non-manifold "
              f"{sum(bt[t]['with_nonmanifold_edges'] for t in TOPOLOGIES)}", flush=True)
    if "K3_determinism" in summary:
        print(f"[d4-remesh] K3 determinismo: {summary['K3_determinism']['pass']}", flush=True)


def _pymeshlab_version() -> str:
    from importlib.metadata import version
    try:
        return version("pymeshlab")
    except Exception:  # noqa: BLE001
        return "?"


def _check_one(task):
    mesh, op = task
    try:
        V, F = load_vf(Path(mesh))
        with np.load(op) as z:
            ev = np.asarray(z["evals"])
            nv = len(z["verts"] if "verts" in z else z["V"])
        lab = components(np.asarray(F, np.int64), len(V))
        return {"ok_ops": bool(len(ev) == 128 and np.isfinite(ev).all() and nv == len(V)),
                "components": int(len(np.unique(lab[np.unique(F)]))), "isolated": int(len(V) - len(np.unique(F)))}
    except Exception as exc:  # noqa: BLE001
        return {"ok_ops": False, "components": -1, "isolated": -1, "error": f"{type(exc).__name__}: {exc}"}


def run_check(views: list, workers: int) -> None:
    path = D4 / "remesh_stats.json"
    summary = json.loads(path.read_text())
    k4 = {}
    for view in views:
        meshes = sorted((STAGE / view / "in").glob("*.npz"))
        tasks = [(str(m), str(OPS / view / "ops" / m.name)) for m in meshes]
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_check_one, tasks, chunksize=4)
        bad = [Path(m).name + " " + json.dumps(r) for (m, _), r in zip(tasks, res)
               if not (r["ok_ops"] and r["components"] == 1 and r["isolated"] == 0)]
        n_exp = summary[view]["n_meshes"]
        k4[view] = {"meshes": len(meshes), "expected": n_exp, "valid": len(meshes) - len(bad), "invalid": bad,
                    "fraction_invalid": (n_exp - (len(meshes) - len(bad))) / n_exp,
                    "pass": bool(len(meshes) == n_exp and not bad)}
        print(f"[d4-check] {view}: {len(meshes)} / {n_exp} mesh, valide {len(meshes) - len(bad)}; "
              f"non valide {bad[:3]}", flush=True)
    summary["K4_validity"] = k4
    path.write_text(json.dumps(summary, indent=1) + "\n")


def tasks(ops_root: Path) -> None:
    rows = []
    for view in VIEWS:
        dist = store_info(view)["dist_npz"]
        for pre, v, e in ARMS:
            ck = str(np.load(bp_e3.store(view, v, e), allow_pickle=True)["checkpoint"])
            out = D4 / "emb" / view / pre / "embeddings.npz"
            if not out.exists():
                rows.append((ops_root / view, D4 / view / "scale_table.npz", dist, ck, out))
    st = bp_e3.store("hifi3d", "factorized", "072")
    out = D4 / "emb_check" / "hifi3d_factorized_s1234" / "embeddings.npz"
    if not out.exists():                             # K2: cache dello store di HIFI3D, tabella ufficiale
        rows.append((ops_root / "hifi3d_store", bp_e3.CHECK_TABLE, store_info("hifi3d")["dist_npz"],
                     str(np.load(st, allow_pickle=True)["checkpoint"]), out))
    for r in rows:
        print("|".join(str(x) for x in r))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("remesh", "check", "tasks"))
    p.add_argument("--views", default=",".join(VIEWS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ops-root", type=Path)
    a = p.parse_args()
    if a.step == "remesh":
        run_remesh(a.views.split(","), a.workers)
    elif a.step == "check":
        run_check(a.views.split(","), a.workers)
    else:
        tasks(a.ops_root)


if __name__ == "__main__":
    main()
