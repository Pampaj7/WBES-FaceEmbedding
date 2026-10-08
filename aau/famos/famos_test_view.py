#!/usr/bin/env python3
"""Set di test reale "persone FaMoS mai viste": scansioni grezze dei soggetti di TEST, piu' le loro
registrazioni come controllo, nel formato delle viste di ``aau/zs3dmm``.

    aau/run.sh aau/famos/famos_test_view.py --workers 24     (dopo famos_subsample.py e famos_unified.py)

Selezione dei fotogrammi (solo fra quelli con una scansione di test, ognuno ha la registrazione dello
stesso fotogramma). Per ogni fotogramma, ``expr_mm`` = distanza g (GT unificata, mm) fra la sua
registrazione e la forma neutra di riferimento del soggetto (famos_subsample.py): misura quanto il volto
e' lontano dal neutro, posa esclusa (Procrustes).
  - galleria: il fotogramma con ``expr_mm`` minima fra tutti quelli del soggetto (uno per soggetto);
  - ``peak``: per ogni sequenza il fotogramma con ``expr_mm`` massima (l'espressione piu' marcata);
  - ``nearneutral``: per ogni sequenza il fotogramma con ``expr_mm`` minima, galleria e peak esclusi
    (un'altra cattura dello stesso volto quasi neutro).
Mesh: scansione (``scan``) per galleria, peak e nearneutral; registrazione FLAME (``reg``, il controllo:
stessa persona e stesso fotogramma, superficie ritopologizzata e pulita) per galleria e peak.

Pre-elaborazione: quella delle scansioni NoW, l'unico altro test su scansioni reali
(i passi di ``aau/recon/now_prepare_meshes.canonical_patch``, importati): frame canonico del template T7 di NoW
(``~/data/now_eval_work/template_lmk7.json``, copiato accanto) con la similarita' dai 7 landmark della
mesh, ritaglio ``compute_mask`` di NoW, pulizia manifold, decimazione quadrica a 5215 triangoli
(sopra il bersaglio: prima suddivisione 1->4, il caso delle registrazioni FLAME come di MICA), verso
dei triangoli uscente. I 7 landmark (iBUG 36, 39, 42, 45, 33, 48, 54, come NoW) vengono dalla
registrazione dello stesso fotogramma (embedding FLAME dei 68 landmark di DECA/MICA): registrazione e
scansione stanno nello stesso frame (controllo ``reg_to_scan_mm``: mediana della distanza dei vertici
della regione unificata della registrazione dal vertice piu' vicino della scansione).

UNA differenza da NoW, necessaria: i triangoli si orientano in modo coerente (``igl.bfs_orient``, solo
l'ordine dei vertici nei triangoli, nessun vertice si muove) PRIMA della decimazione, oltre che dopo.
Una scansione FaMoS con anche un solo triangolo girato dentro il ritaglio manda ``igl.qslim`` (che chiude
il bordo su un vertice all'infinito) oltre 48 GB: provato su FaMoS_subject_088/bareteeth.000044,
087/happiness.000160 e 088/surprise.000242, che con l'orientamento passano in 0.6 s. Sulle mesh gia'
coerenti (tutte le registrazioni, quasi tutte le scansioni) ``bfs_orient`` non cambia niente, quindi
la patch e' la stessa di ``canonical_patch``; quante facce gira per mesh sta in ``n_faces_reoriented``.
Ogni processo ha un tetto di memoria virtuale (``MEM_CAP_GB``): un caso patologico nuovo diventa un
errore con il nome della mesh, non un worker ucciso dall'OOM e un Pool fermo per sempre.

Uscite in ``datasets/FAMOS/test_view``: ``npz/id9400NN_GTready_<scan|reg>__<sequenza>__<fotogramma>.npz``
(V float32, F int32), ``manifest.csv`` (una riga per mesh: soggetto, etichetta, ruolo, ``expr_mm``,
controlli della preparazione), ``labels.json``, ``selection.json``, ``template_lmk7.json``.
``gt_matrix.npz`` la scrive famos_unified.py. Riassunto in ``aau/runs/evidence/famos/test_view.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parents[1] / "v3_work" / "unified_gt"))
sys.path.insert(0, str(THIS_DIR.parent / "recon"))

import famos_common as fc  # noqa: E402
import now_common  # noqa: E402
import now_prepare_meshes as nowprep  # noqa: E402
from shapes import Space, unified_distances  # noqa: E402

TEMPLATE = now_common.template_path()
FIELDS = ("name", "view_id", "subject", "kind", "role", "seq", "frame", "expr_mm", "reg_to_scan_mm",
          "n_vertices_raw", "n_faces_reoriented") + nowprep.PREP_FIELDS[3:]
MEM_CAP_GB = 12

_ST: dict = {}


def select(subject: str, sp: Space, F: np.ndarray) -> list[dict]:
    """Fotogrammi scelti del soggetto (vedi docstring)."""
    with np.load(fc.TEST_DIR / f"{subject}.npz") as z:
        Vn = z["V_neutral"].astype(np.float64)
    frames = []
    for sd in sorted(p for p in (fc.SCAN_DIR / subject).iterdir() if p.is_dir()):
        for f, _ in fc.frames_of(sd, "obj"):
            frames.append((sd.name, f))
    V = np.stack([fc.read_reg(fc.reg_path(subject, q, f)) for q, f in frames]).astype(np.float64) * 1000.0
    a, _, _ = sp.align(sp.map("flame", np.concatenate([Vn[None], V])))
    S = sp.svec(a)
    e = unified_distances(S[1:], S[:1], sp.A)[:, 0]
    g = int(np.argmin(e))
    out = [{"seq": frames[g][0], "frame": frames[g][1], "role": "gallery", "expr_mm": float(e[g])}]
    for q in sorted({q for q, _ in frames}):
        idx = [k for k, (qq, _) in enumerate(frames) if qq == q]
        pk = max(idx, key=lambda k: e[k])
        if pk != g:
            out.append({"seq": q, "frame": frames[pk][1], "role": "peak", "expr_mm": float(e[pk])})
        rest = [k for k in idx if k not in (g, pk)]
        if rest:
            nn = min(rest, key=lambda k: e[k])
            out.append({"seq": q, "frame": frames[nn][1], "role": "nearneutral", "expr_mm": float(e[nn])})
    return out


def _init(template: np.ndarray, F: np.ndarray, region_vidx: np.ndarray) -> None:
    import resource

    cap = MEM_CAP_GB * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    _ST.update(template=template, F=F, region=region_vidx, lmk=fc.flame_lmk68())


def canonical_patch(V, F, lmk, template, target_faces):
    """``now_prepare_meshes.canonical_patch`` (stessi passi, importati), con ``bfs_orient`` prima della
    decimazione (vedi docstring); restituisce anche il numero di facce girate."""
    import igl

    Vc, Lc, scale = now_common.to_canonical(V, lmk, template)
    centre, radius = now_common.now_mask(Lc)
    Vk, Fk = nowprep.crop_patch(Vc, F, centre, radius)
    Vm, Fm = nowprep.make_manifold(Vk, Fk)
    G = np.asarray(igl.bfs_orient(np.asarray(Fm, dtype=np.int64))[0], dtype=np.int64)
    n_flip = int((~(G == Fm).all(1)).sum())
    Fm = G
    if len(Fm) < target_faces:
        Vm, Fm = nowprep.mo.subdivide_midpoint(Vm, Fm, 1)
    Vd, Fd = nowprep.mo.decimate_to(Vm, Fm, target_faces)
    if len(Fd) < 0.9 * target_faces:
        raise RuntimeError(f"decimazione a {len(Fd)} triangoli invece di {target_faces}")
    Fd, outward_before = nowprep.orient_outward(Vd, Fd)
    outward, folds = nowprep.patch_quality(Vd, Fd)
    return Vd, Fd, {
        "outward_area_before": outward_before, "outward_area": outward, "n_folds": folds,
        "scale_to_mm": float(scale), "crop_radius_mm": float(radius),
        "n_vertices_crop": int(len(Vk)), "n_vertices": int(len(Vd)), "n_faces": int(len(Fd)),
        "lmk_residual_mm": float(np.linalg.norm(Lc - template, axis=1).mean()),
    }, n_flip


def prep(task: dict) -> dict:
    try:
        return _prep(task)
    except Exception as exc:                                  # noqa: BLE001
        return {"error": f"{task['subject']}/{task['kind']}/{task['seq']}.{task['frame']:06d}: "
                         f"{type(exc).__name__}: {exc}"}


def _prep(task: dict) -> dict:
    subject, q, f, kind = task["subject"], task["seq"], task["frame"], task["kind"]
    Vr = fc.read_reg(fc.reg_path(subject, q, f)).astype(np.float64) * 1000.0
    F = _ST["F"]
    lmk = fc.landmarks(Vr, F, *_ST["lmk"], now_common.LMK7_IBUG)
    if kind == "scan":
        V, Fm = fc.read_obj(fc.scan_path(subject, q, f))
        reg_to_scan = float(np.median(cKDTree(V).query(Vr[_ST["region"]])[0]))
    else:
        V, Fm, reg_to_scan = Vr, F.astype(np.int64), float("nan")
    Vd, Fd, row, n_flip = canonical_patch(V, Fm, lmk, _ST["template"], now_common.TARGET_FACES)
    name = f"{fc.view_id(subject)}_GTready_{kind}__{q}__{f:06d}"
    nowprep.mo.save_variant(Vd, Fd, fc.VIEW_DIR / "npz" / f"{name}.npz")
    return {"name": name, "view_id": fc.view_id(subject), "subject": subject, "kind": kind, "role": task["role"],
            "seq": q, "frame": f, "expr_mm": task["expr_mm"], "reg_to_scan_mm": reg_to_scan,
            "n_vertices_raw": int(len(V)), "n_faces_reoriented": n_flip, **row}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--subjects", default="", help="sottoinsieme (prova); vuoto = tutti i soggetti di TEST")
    a = p.parse_args()

    split = fc.load_split()
    subjects = split["test"] if not a.subjects else [s for s in split["test"] if s in a.subjects.split(",")]
    assert not set(subjects) & set(split["train"]), "soggetti di TRAIN nel set di test"
    sp = Space()
    F = np.load(fc.FLAME_FACES)
    template = np.asarray(json.loads(TEMPLATE.read_text())["landmarks_mm"], dtype=np.float64)
    fc.VIEW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE, fc.VIEW_DIR / "template_lmk7.json")
    out_npz = fc.VIEW_DIR / "npz"
    out_npz.mkdir(exist_ok=True)
    for stale in out_npz.glob("*.npz"):
        stale.unlink()
    # i latenti in cache (famos_eval.py, per nome di mesh) verrebbero da patch vecchie
    for stale in (fc.OUT_ROOT / "eval" / "embeddings").glob("*_test_view.npz"):
        stale.unlink()

    t0 = time.time()
    selection = {s: select(s, sp, F) for s in subjects}
    (fc.VIEW_DIR / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    tasks = []
    for s, rows in selection.items():
        for r in rows:
            tasks.append({"subject": s, "kind": "scan", **r})
            if r["role"] in ("gallery", "peak"):
                tasks.append({"subject": s, "kind": "reg", **r})
    print(f"[famos-view] selezione in {time.time() - t0:.0f}s: {len(subjects)} soggetti, {len(tasks)} mesh "
          f"({sum(t['kind'] == 'scan' for t in tasks)} scansioni, {sum(t['kind'] == 'reg' for t in tasks)} registrazioni)",
          flush=True)

    rows, errors = [], []
    with mp.get_context("fork").Pool(a.workers, initializer=_init,
                                     initargs=(template, F, sp.z["flame_vidx"])) as pool:
        for r in pool.imap_unordered(prep, tasks, chunksize=2):
            if "error" in r:
                errors.append(r["error"])
                print(f"[famos-view] ERRORE {r['error']}", flush=True)
                continue
            rows.append(r)
            if len(rows) % 100 == 0 or len(rows) == len(tasks):
                print(f"[famos-view] {len(rows)}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    if errors:
        raise SystemExit(f"ERRORE: {len(errors)} mesh non preparate: {errors[:5]}")
    rows.sort(key=lambda r: r["name"])
    now_common.write_rows(fc.VIEW_DIR / "manifest.csv", FIELDS, rows, template=str(TEMPLATE),
                          target_faces=now_common.TARGET_FACES, landmarks="FLAME full_lmk iBUG 36,39,42,45,33,48,54")
    (fc.VIEW_DIR / "labels.json").write_text(json.dumps(
        {"note": "etichetta d'identita' = view_id (id9400NN = 940000 + numero del soggetto FaMoS)",
         "labels": {r["name"]: r["view_id"] for r in rows},
         "subjects": {fc.view_id(s): s for s in subjects}}, indent=1) + "\n")

    def med(key, sel):
        x = np.array([float(r[key]) for r in rows if sel(r)], dtype=float)
        return {"median": float(np.nanmedian(x)), "min": float(np.nanmin(x)), "max": float(np.nanmax(x)), "n": int(len(x))}

    summary = {"n_subjects": len(subjects), "n_meshes": len(rows),
               "by_kind_role": {f"{k}/{ro}": sum(r["kind"] == k and r["role"] == ro for r in rows)
                                for k in ("scan", "reg") for ro in ("gallery", "peak", "nearneutral")},
               "expr_mm": {ro: med("expr_mm", lambda r, ro=ro: r["role"] == ro and r["kind"] == "scan")
                           for ro in ("gallery", "nearneutral", "peak")},
               "reg_to_scan_mm": med("reg_to_scan_mm", lambda r: r["kind"] == "scan"),
               "n_vertices_raw_scan": med("n_vertices_raw", lambda r: r["kind"] == "scan"),
               "meshes_reoriented_before_decimation": {k: sum(r["kind"] == k and r["n_faces_reoriented"] > 0
                                                              for r in rows) for k in ("scan", "reg")}}
    for key in ("n_vertices_crop", "n_vertices", "n_faces", "crop_radius_mm", "scale_to_mm", "lmk_residual_mm",
                "outward_area", "n_folds"):
        summary[key] = {k: med(key, lambda r, k=k: r["kind"] == k) for k in ("scan", "reg")}
    fc.EVID_DIR.mkdir(parents=True, exist_ok=True)
    (fc.EVID_DIR / "test_view.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)
    n_files = len(list(out_npz.glob("*.npz")))
    if n_files != len(tasks):
        raise SystemExit(f"ERRORE: {n_files} npz su {len(tasks)} attese")


if __name__ == "__main__":
    main()
