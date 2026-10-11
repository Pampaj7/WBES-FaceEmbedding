#!/usr/bin/env python3
"""Emendamento 3 (POST HOC): ingresso dei bracci ritagliato alla regione di B, crop degli held-out sintetici
(PROTOCOL_emendamento_3.md, sez. 1, 2, 5).

    aau/outlineB/run_o3d.sh aau/baselines_param/bp_e3.py check-region
    aau/outlineB/run_o3d.sh aau/baselines_param/bp_e3.py crop --tmp <T> --workers 64
    aau/run.sh aau/baselines_param/bp_e3.py stage-heldout --tmp <T>
    aau/run.sh aau/baselines_param/bp_e3.py tasks --tmp <T>
    (bp_e3.sbatch; i delta in bp_paired_e3.py)

``check-region``: ``bp.region`` rieseguita per (vista, modello) ridà i vertici di ``region.npz`` (la regione non si
tocca). Uscita ``e3/check_region.json``.

``crop`` (sez. 1): per (vista, modello) la regione di ``region.npz`` collocata sul riferimento della vista come in
``bp.region`` (baricentri, ``bp.rigid_icp`` con soglia 1000 mm poi ``REGION_DIST_MM``), poi per ognuna delle 600 mesh
dello store dei bracci: regione portata sulla mesh in mm (``blmm.to_mm``) con ``bp.rigid_icp``, vertici della mesh
tenuti se il punto piu' vicino sulla regione (``bp.Surface``) dista meno di ``REGION_DIST_MM`` e cade su un triangolo
che non tocca il bordo della regione (l'esclusione di ``bp.loop_correspondences``), triangoli coi tre vertici tenuti,
componente connessa piu' grande, vertici non referenziati tolti. Coordinate grezze (unita' native) nella cartella
``<T>/crop/<vista>_<modello>/in``; FaceVerse con le facce invertite (``zs_stage.py --flip-faces``). Diagnostica per
mesh (vertici, aree in mm^2, copertura della regione con la regola di voto di ``bp.region``) in
``e3/crop/<vista>_<modello>.npz`` e il riepilogo in ``e3/crop_stats.json``.

``stage-heldout`` (sez. 2): ``fact_calib.stage`` (importato, in sola lettura) con le sole etichette ``crop`` in
``<T>/heldout/in``, tabella ``<T>/heldout/crop_table.npz``; poi la tabella unita (quella della cache della
calibrazione + i crop) in ``<T>/heldout/scale_table.npz``.

``tasks``: le righe ``ops|tabelle|dist_npz|checkpoint|uscita`` degli embedding (esperimento 1 su ogni ritaglio, controllo
della catena sulla cache dello store di HIFI3D, esperimento 2 sulle 1.800 held-out), per le code GPU dello sbatch.
Checkpoint e ``--dist_npz`` dall'``embeddings.npz`` e dall'``eval_key.txt`` dello store della vista.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
OUT = REPO / "aau" / "runs" / "evidence" / "baselines_param"
E3 = OUT / "e3"
EV = REPO / "aau" / "runs" / "evidence" / "trainer_v3"
EVAL = EV / "ablations" / "c3f_eval"
NEUTRAL_EMB = REPO / "aau" / "runs" / "evidence" / "faceverse_neutral" / "embed"
# vista -> cartella degli store dei bracci (fact_paired.FORM_DIR, piu' la neutra di bp_paired.NEUTRAL_EMB)
FORM_DIR = {"hifi3d": EVAL / "form_hifi", "facescape": EVAL / "form_devfs", "faceverse": EVAL / "form_fv_expr",
            "faceverse_neutral": NEUTRAL_EMB}
CROP_VIEWS = ("hifi3d", "facescape", "faceverse", "faceverse_neutral")      # bp.CROP_VIEWS
REGION_MODELS = ("gnm", "flame2023")                                        # bp.MODELS
FLIP = ("faceverse", "faceverse_neutral")             # facce invertite negli store (zs_stage.py --flip-faces)
# (prefisso delle colonne di fact_paired, variante del tag, epoca): i bracci dell'emendamento (fact_paired.MODELS)
ARMS = (("factorized_s1234", "factorized", "072"), ("factorized_s2345", "factorizeds2345", "072"),
        ("ctrlfr_s1234", "ctrlfr", "072"), ("ctrlfr_s2345", "ctrlfrs2345", "072"),
        ("factorizedc3m_e123", "factorizedc3m", "123"), ("factorizedc3m_e205", "factorizedc3m", "205"))
MIN_VERTICES = 500                                     # sotto: ritaglio fallito (sez. 1)
TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")    # zs_stage.TOPOLOGIES
HELDOUT_OPS = REPO / "datasets" / "V3_OPS_CACHE" / "heldout_calib"        # cache della calibrazione (fact_calib)
GT_SR_TRAIN = EV / "ablations" / "c3f" / "gt_sr.npz"
CHECK_OPS = REPO / "datasets" / "V3_OPS_CACHE" / "8e8f81d5f0204394"         # operatori dello store di HIFI3D
CHECK_TABLE = EV / "factorized" / "scale_tables" / "hifi3d_eval.npz"
_C: dict = {}          # regione collocata della (vista, modello), ereditata dai worker (fork)


def store(view: str, v: str, e: str) -> Path:
    """``embeddings.npz`` dello store dei bracci (``fact_paired.embeddings``, copiata; vista neutra compresa)."""
    hits = sorted(FORM_DIR[view].glob(f"data_*/scale_v3{v}fulle{e}*/zs_zeroshot/embeddings.npz"))
    if not hits:
        raise SystemExit(f"{view}: store di {v} e{e} assente")
    return hits[0]


def store_key(emb: Path) -> dict:
    """``eval_key.txt`` e ``subjects.json`` accanto allo store."""
    d = emb.parents[1]
    key = dict(ln.split("=", 1) for ln in (d / "eval_key.txt").read_text().split() if "=" in ln)
    return {**key, **json.loads((d / "subjects.json").read_text())}


# ------------------------------------------------------------------------------------------ regione

def check_region() -> None:
    import bp
    out = {}
    for view in CROP_VIEWS:
        for name in REGION_MODELS:
            with np.load(bp.out_dir(view, name) / "region.npz") as z:
                ref = np.asarray(z["vertices"])
            got = bp.region(view, bp.load_model(name))["vertices"]
            out[f"{view}|{name}"] = {"n_region_npz": int(len(ref)), "n_rerun": int(len(got)),
                                     "equal": bool(len(ref) == len(got) and (ref == got).all())}
            print(f"[bp-e3] regione {view} {name}: {out[f'{view}|{name}']}", flush=True)
    E3.mkdir(parents=True, exist_ok=True)
    (E3 / "check_region.json").write_text(json.dumps(out, indent=1) + "\n")


def placed_region(view: str, name: str) -> dict:
    """Regione di ``region.npz`` collocata sul riferimento della vista (le righe di ``bp.region``): vertici (mm, frame
    canonico), triangoli, bordo, pesi d'area della media del modello."""
    import bp
    import blmm
    m = bp.load_model(name)
    with np.load(bp.out_dir(view, name) / "region.npz") as z:
        used = np.asarray(z["vertices"], dtype=np.int64)
    M = bp.place_canonical(name, m["mu"])
    refs = bp.references(view)
    if len(refs) != 1:
        raise SystemExit(f"{view}: {len(refs)} riferimenti, atteso 1 (media del template)")
    _, V, _ = refs[0]
    T = np.eye(4)
    T[:3, 3] = V.mean(0) - M.mean(0)
    T = bp.rigid_icp(M, V, 1000.0, T)
    T = bp.rigid_icp(M, V, bp.REGION_DIST_MM, T)
    A = (M @ T[:3, :3].T + T[:3, 3])[used]
    F = bp.compact_faces(m["F"], used, len(m["mu"]))
    return {"A": A, "F": F, "bnd": bp.boundary_vertices(F, len(used)),
            "w": blmm.vertex_areas(m["mu"][used], F)}


def area(V: np.ndarray, F: np.ndarray) -> float:
    t = V[F]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def _crop(task):
    """Ritaglio di una mesh alla regione collocata: (statistiche, motivo del fallimento)."""
    import bp
    import blmm
    import mesh_ops as mo
    from scipy.spatial import cKDTree
    path, out_path = task
    st = np.full(6, np.nan)                      # vertici prima / dopo, area mm^2 prima / dopo, copertura, fitness
    try:
        with np.load(path) as z:
            V0, F0 = np.asarray(z["V"]), np.asarray(z["F"], dtype=np.int64)
        X = blmm.to_mm(_C["view"], np.asarray(V0, dtype=np.float64))
        R = _C["reg"]
        T = bp.rigid_icp(R["A"], X, bp.REGION_DIST_MM, np.eye(4))
        A = R["A"] @ T[:3, :3].T + T[:3, 3]
        d, i = cKDTree(X).query(A)                   # copertura: regola di voto di bp.region
        ok = (d < bp.REGION_DIST_MM) & ~bp.boundary_ring(F0, len(X))[i]
        st[[0, 2, 4]] = len(X), area(X, F0), R["w"][ok].sum() / R["w"].sum()
        st[5] = float(ok.mean())
        p, tri, _ = bp.Surface(A, R["F"]).closest(X)
        keep = (np.linalg.norm(p - X, axis=1) < bp.REGION_DIST_MM) & ~R["bnd"][R["F"][tri]].any(1)
        F = F0[keep[F0].all(1)]
        if len(F):
            F = mo.largest_component(F)
        used = np.unique(F)
        if len(used) < MIN_VERTICES:
            return st, f"{len(used)} vertici dopo il ritaglio"
        remap = -np.ones(len(X), dtype=np.int64)
        remap[used] = np.arange(len(used))
        F = remap[F]
        st[[1, 3]] = len(used), area(X[used], F)
        if _C["flip"]:
            F = np.ascontiguousarray(F[:, ::-1])
        np.savez(out_path, V=V0[used], F=F.astype(F0.dtype))
        return st, ""
    except Exception as exc:  # noqa: BLE001  (la mesh resta NaN per questo ritaglio, contata)
        return st, f"{type(exc).__name__}: {exc}"


def summarize_crop(view: str, name: str, subj: list, topo: list, S: np.ndarray) -> dict:
    """Riepilogo per topologia e crop contro original dopo il ritaglio (sez. 1, diagnostica)."""
    topo = np.asarray(topo)
    out = {"by_topology": {}}
    for t in TOPOLOGIES:
        k = topo == t
        cov, frac = S[k, 4], S[k, 3] / S[k, 2]
        out["by_topology"][t] = {"n": int(k.sum()), "coverage_median": float(np.nanmedian(cov)),
                                 "coverage_p5": float(np.nanpercentile(cov, 5)),
                                 "area_kept_median": float(np.nanmedian(frac)),
                                 "vertices_after_median": float(np.nanmedian(S[k, 1]))}
    pos = {(s, t): r for r, (s, t) in enumerate(zip(subj, topo))}
    us = sorted(set(subj))
    for lab, col in (("after", 3), ("before", 2)):
        r = np.asarray([0.5 * np.log(S[pos[(s, "crop")], col] / S[pos[(s, "original")], col]) for s in us])
        out[f"log_sqrt_area_crop_over_original_{lab}"] = {
            "median": float(np.nanmedian(r)), "q25": float(np.nanpercentile(r, 25)), "q75": float(np.nanpercentile(r, 75))}
    return out


def run_crop(tmp: Path, views: list, models: list, workers: int) -> None:
    import bp  # noqa: F401  (mette blmm su sys.path)
    import blmm
    E3.mkdir(parents=True, exist_ok=True)
    (E3 / "crop").mkdir(exist_ok=True)
    stats_path = E3 / "crop_stats.json"
    summary = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for view in views:
        sk = store_key(store(view, "factorized", "072"))
        vdir = Path(sk["view_dir"])
        if vdir.resolve() != blmm.VIEWS[view]["dir"].resolve():
            raise SystemExit(f"{view}: store su {vdir}, blmm su {blmm.VIEWS[view]['dir']}")
        if sorted(sk["subjects"]) != sorted(blmm.subjects(view)) or bool(sk["flip_faces"]) != (view in FLIP):
            raise SystemExit(f"{view}: soggetti o verso delle facce diversi dallo store")
        items = [(s, t) for s in sorted(sk["subjects"]) for t in TOPOLOGIES]
        for name in models:
            t0 = time.time()
            d = tmp / "crop" / f"{view}_{name}"
            (d / "in").mkdir(parents=True, exist_ok=True)
            (d / "domain").write_text(blmm.VIEWS[view]["domain"] + "\n")
            (d / "dist_npz").write_text(sk["dist_npz"] + "\n")
            _C.update(view=view, reg=placed_region(view, name), flip=view in FLIP)
            tasks = [(str(vdir / f"{s}_GTready_{t}.npz"), str(d / "in" / f"{s}_GTready_{t}.npz")) for s, t in items]
            with mp.get_context("fork").Pool(workers) as pool:
                res = pool.map(_crop, tasks, chunksize=2)
            S = np.stack([r[0] for r in res])
            failed = [f"{s}|{t}|{r[1]}" for (s, t), r in zip(items, res) if r[1]]
            np.savez(E3 / "crop" / f"{view}_{name}.npz", subjects=np.asarray([s for s, _ in items]),
                     topologies=np.asarray([t for _, t in items]), stats=S,
                     columns=np.asarray(["n_vertices_in", "n_vertices_out", "area_in_mm2", "area_out_mm2",
                                         "coverage_area", "coverage_vertices"]),
                     failed=np.asarray(failed, dtype="U300"), n_region_vertices=len(_C["reg"]["A"]))
            summary[f"{view}|{name}"] = {"n_meshes": len(items), "failed": failed, "seconds": time.time() - t0,
                                         "n_region_vertices": int(len(_C["reg"]["A"])),
                                         **summarize_crop(view, name, [s for s, _ in items], [t for _, t in items], S)}
            stats_path.write_text(json.dumps(summary, indent=1) + "\n")
            bt = summary[f"{view}|{name}"]["by_topology"]
            print(f"[bp-e3] ritaglio {view} {name}: {len(items)} mesh in {time.time() - t0:.0f}s, fallite {len(failed)}; "
                  "copertura mediana " + ", ".join(f"{t} {bt[t]['coverage_median']:.3f}" for t in TOPOLOGIES) +
                  "; area tenuta " + ", ".join(f"{t} {bt[t]['area_kept_median']:.2f}" for t in TOPOLOGIES), flush=True)


# ------------------------------------------------------------------------------------------ held-out

def stage_heldout(tmp: Path) -> None:
    """Le crop held-out con ``fact_calib.stage`` (etichette ridotte a ``crop`` in memoria), poi la tabella unita."""
    sys.path.insert(0, str(REPO / "v3_work" / "trainer" / "tools"))
    import fact_calib as fc
    fc.LABELS = ("crop",)
    d = tmp / "heldout"
    fc.stage(d / "in", d / "crop_table.npz")
    parts = []
    for p in (HELDOUT_OPS / "scale_table.npz", d / "crop_table.npz"):
        with np.load(p) as z:
            parts.append({k: np.asarray(z[k]) for k in ("names", "domain", "area_mm2", "area_raw", "maxabs_raw", "check")})
    names = np.concatenate([p["names"] for p in parts]).astype(str)
    if len(set(names)) != len(names):
        raise SystemExit("tabella unita: nomi ripetuti")
    np.savez(d / "scale_table.npz", **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    n_crop = len(parts[1]["names"])
    print(f"[bp-e3] held-out: {n_crop} crop messe in scena, tabella unita {len(names)} mesh", flush=True)


# ------------------------------------------------------------------------------------------ code GPU

def tasks(tmp: Path) -> None:
    """Righe ``ops|tabelle|dist_npz|checkpoint|uscita`` per le code di embedding."""
    rows = []
    for view in CROP_VIEWS:
        for name in REGION_MODELS:
            d = tmp / "crop" / f"{view}_{name}"
            todo = [pre for pre, v, e in ARMS if not (E3 / "embed" / view / name / pre / "embeddings.npz").exists()
                    and any(FORM_DIR[view].glob(f"data_*/scale_v3{v}fulle{e}*/zs_zeroshot/embeddings.npz"))]
            if not todo:
                continue                             # tutti gia' scritti (es. passo solo held-out)
            if not (d / "ops").is_dir():
                raise SystemExit(f"{d}/ops assente")
            dist = (d / "dist_npz").read_text().strip()
            for pre, v, e in ARMS:
                try:
                    ck = str(np.load(store(view, v, e), allow_pickle=True)["checkpoint"])
                except SystemExit as exc:            # es. C3M e123 senza store sulla vista neutra: dichiarato
                    print(f"[bp-e3] {exc}: saltato", file=sys.stderr)
                    continue
                out = E3 / "embed" / view / name / pre / "embeddings.npz"
                if not out.exists():
                    rows.append((d / "ops", d / "scale_table.npz", dist, ck, out))
    ck = str(np.load(store("hifi3d", "factorized", "072"), allow_pickle=True)["checkpoint"])
    out = E3 / "embed_check" / "hifi3d_store_factorized_s1234" / "embeddings.npz"
    if not out.exists():
        rows.append((CHECK_OPS, CHECK_TABLE, store_key(store("hifi3d", "factorized", "072"))["dist_npz"], ck, out))
    for pre, v, e in ARMS:
        ck = str(np.load(store("hifi3d", v, e), allow_pickle=True)["checkpoint"])
        out = E3 / "embed_heldout" / pre / "embeddings.npz"
        if not out.exists():
            rows.append((tmp / "heldout" / "ops", tmp / "heldout" / "scale_table.npz", GT_SR_TRAIN, ck, out))
    for r in rows:
        print("|".join(str(x) for x in r))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("check-region", "crop", "stage-heldout", "tasks"))
    p.add_argument("--tmp", type=Path)
    p.add_argument("--views", default=",".join(CROP_VIEWS))
    p.add_argument("--models", default=",".join(REGION_MODELS))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.step == "check-region":
        check_region()
    elif a.step == "crop":
        sys.path.insert(0, str(THIS))
        run_crop(a.tmp, a.views.split(","), a.models.split(","), a.workers)
    elif a.step == "stage-heldout":
        stage_heldout(a.tmp)
    else:
        tasks(a.tmp)


if __name__ == "__main__":
    main()
