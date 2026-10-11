#!/usr/bin/env python3
"""Viste di Ava-256 (sola geometria, nessun operatore, nessun embedding).

    v3_work/unified_gt/run.sh aau/ava256/ava_views.py identities       (dopo ava_gt.py)
    aau/run.sh aau/zs3dmm/make_zs_topologies.py --prefix ava --in-dir datasets/AVA256/identities \\
        --out-dir datasets/AVA256/topo --n-cores 32 --overwrite        (lo stesso codice di HIFI3D e FaceVerse)
    aau/run.sh aau/zs3dmm/zs_check_crop.py --topo-dir datasets/AVA256/topo --prefix ava --write
    v3_work/unified_gt/run.sh aau/ava256/ava_views.py view              (viste di valutazione e di calibrazione)

CONFERMATIVO: non valutare prima del protocollo confermativo (README). Ava-256 non va aggiunto a nessuno strumento di
valutazione esistente; ogni vista porta un file ``CONFERMATIVO_NON_VALUTARE.txt``. Con ``datasets/AVA256/FROZEN.json``
presente (``ava_freeze.py``) lo script si rifiuta di scrivere.

``identities`` (README, scelta 7 e sezione "Modifiche"): per ogni soggetto valido la neutra di ava_neutral.py in mm
(u = 1, ava_gt.py) nel frame NATIVO dei dati (frame della testa del tracking di Meta: nessuna rigida della GT negli
ingressi), ristretta alla patch di ava_patch.py (impronta NoW, aperture palpebrali e bocca, stesso insieme di indici per
tutti), **suddivisa 1-a-4 a punto medio una volta** (``mesh_ops.subdivide_midpoint`` = ``igl.upsample``: la superficie non
cambia, come FLAME 2023 in D1): la original di Ava passa da un lato medio di circa 4 mm a circa 2 mm, dentro l'intervallo
delle original di HIFI3D, FaceVerse e FaceScape dev (1.4-2.1 mm), e remesh e down8k la perturbano quanto quelle
(``ava_view_qc.py``). ``identities/avaNNNN.npz`` (``V`` float32, ``F`` int32, le chiavi di ``zs_identities.py``); controlli
per mesh: ``now_prepare_meshes.patch_quality`` (frazione d'area con la normale verso +z, pieghe a piu' di 120 gradi).

``view``: split di ava_gt.py (``datasets/AVA256/gt/split.json``):
  - ``eval_view/npz/id<950000 + NNNN>_GTready_<topologia>.npz``: SOLO i 200 di valutazione (symlink a ``topo/``, come
    ``aau/zs3dmm/make_zs_view.py``, che non si usa tal quale: vuole GT maxabs e dei coefficienti, che per dati reali non
    esistono); ``gt_fr.npz`` / ``gt_sr.npz`` = le GT di valutazione di ava_gt.py, ``gt_matrix.npz`` -> ``gt_fr.npz``;
  - ``calib_view/npz/``: i 56 di calibrazione (template del NICP, regione del fit B, L_d, cs_ref), nessuna GT;
  - ``labels.json``, ``manifest.json`` con l'impronta del contenuto (sha256 di V float32 e F int32 di ogni mesh,
    ``mesh_sha256``; la riga di ogni mesh in ``content_sha256.txt``). Riassunto in ``aau/ava256/views_summary.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
for _p in (THIS_DIR, THIS_DIR.parents[1] / "v3_work" / "unified_gt", THIS_DIR.parent / "recon",
           THIS_DIR.parents[1] / "v2_work" / "genict"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ava_common as ac  # noqa: E402
import mesh_ops as mo  # noqa: E402
import ugt as C  # noqa: E402
from now_prepare_meshes import patch_quality  # noqa: E402

PREFIX = "ava"
IDENT_DIR = ac.DATA_ROOT / "identities"
TOPO_DIR = ac.DATA_ROOT / "topo"
VIEWS = {"eval": ac.DATA_ROOT / "eval_view", "calib": ac.DATA_ROOT / "calib_view"}
SUMMARY = ac.SUMMARY_DIR / "views_summary.json"
TOPOLOGIES = ("original", "remesh", "crop", "noisy", "down8k", "up60k")
SUBDIV = 1
MARKER = {"eval": "CONFERMATIVO: non valutare prima del protocollo confermativo (aau/ava256/README.md,\n"
                  "aau/ava256/PROTOCOL_CONFERMATIVO_bozza.md). Ava-256 non va aggiunto a nessuno strumento di valutazione.\n"
                  "Lo strumento che la valutera' deve chiamare ava_freeze.verify_frozen() prima di leggerla.\n",
          "calib": "CALIBRAZIONE di Ava-256: SOLO template del NICP, regione del fit B, L_d e cs_ref. Questi soggetti non si\n"
                   "valutano mai, con nessun metodo (aau/ava256/PROTOCOL_CONFERMATIVO_bozza.md).\n"}


def refuse_if_frozen() -> None:
    if ac.FROZEN.exists():
        raise SystemExit(f"{ac.FROZEN}: viste congelate, non le riscrivo (serve una decisione del PI)")


def mesh_sha256(path: Path) -> str:
    """sha256 del contenuto di una mesh (``V`` float32 e ``F`` int32): stabile fra riesecuzioni, il file npz no."""
    with np.load(path) as z:
        h = hashlib.sha256(np.ascontiguousarray(z["V"], dtype="<f4").tobytes())
        h.update(np.ascontiguousarray(z["F"], dtype="<i4").tobytes())
    return h.hexdigest()


def digest(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


def q(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"min": float(x.min()), "median": float(np.median(x)), "max": float(x.max())}


def stage_identities() -> None:
    refuse_if_frozen()
    u = json.loads((ac.GT_DIR / "frame.json").read_text())["u"]
    with np.load(ac.PATCH_NPZ) as z:
        patch_F = z["patch_F"].astype(np.int64)
    used = np.unique(patch_F)
    remap = -np.ones(ac.N_VERTS, dtype=np.int64)
    remap[used] = np.arange(len(used))
    Fp = remap[patch_F]
    IDENT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in IDENT_DIR.glob(f"{PREFIX}[0-9]*.npz"):
        stale.unlink()
    rows = []
    ids = json.loads((ac.GT_DIR / "ids.json").read_text())["all"]
    for r in ids:
        with np.load(ac.NEUTRAL_DIR / f"{r['ava_id']}.npz") as z:
            V = u * z["V_neutral"].astype(np.float64)[used]
        if not np.isfinite(V).all():
            raise SystemExit(f"{r['ava_id']}: vertici non finiti")
        Vs, Fs = mo.subdivide_midpoint(V, Fp, SUBDIV)
        out_frac, folds = patch_quality(Vs, Fs)
        np.savez_compressed(IDENT_DIR / f"{r['ava_id']}.npz", V=Vs.astype(np.float32), F=Fs.astype(np.int32))
        rows.append({**r, "outward_area": out_frac, "n_folds": folds})
    man = {"source": "datasets/AVA256/neutral (ava_neutral.py), patch_F di datasets/AVA256/corr/patch.npz (ava_patch.py)",
           "units": "mm", "u": u, "frame": "nativo dei dati: frame della testa del tracking di Meta (+y alto, +z avanti)",
           "subdivision": f"1-a-4 a punto medio, {SUBDIV} volta (mesh_ops.subdivide_midpoint = igl.upsample)",
           "n_identities": len(rows), "patch_vertices": int(len(used)), "patch_faces": int(len(Fp)),
           "n_vertices": int(len(Vs)), "n_faces": int(len(Fs)), "prefix": PREFIX, "identities": rows}
    C.save_json(IDENT_DIR / "manifest.json", man)
    print(f"[ava-views] {len(rows)} identita': patch {len(used)} v / {len(Fp)} t, suddivisa {len(Vs)} v / {len(Fs)} t; "
          f"pieghe {q([r['n_folds'] for r in rows])}, area verso +z {q([r['outward_area'] for r in rows])}", flush=True)


def build_view(kind: str, members: list[dict], name_re, want_gt: bool) -> dict:
    root = VIEWS[kind]
    data = root / "npz"
    data.mkdir(parents=True, exist_ok=True)
    for stale in data.glob("*.npz"):
        stale.unlink()
    want = {r["ava_id"]: r["view_id"] for r in members}
    lines, sizes, by_topo = [], {t: [] for t in TOPOLOGIES}, {t: [] for t in TOPOLOGIES}
    for p in sorted(TOPO_DIR.glob(f"{PREFIX}*_GTready_*.npz")):
        m = name_re.match(p.name)
        if not m or f"{PREFIX}{m['num']}" not in want:
            continue
        name = f"{want[PREFIX + m['num']]}_GTready_{m['variant']}"
        (data / f"{name}.npz").symlink_to(p.resolve())
        h = f"{name} {mesh_sha256(p)}"
        lines.append(h)
        by_topo[m["variant"]].append(h)
        with np.load(p) as z:
            sizes[m["variant"]].append((len(z["V"]), len(z["F"])))
    if len(lines) != 6 * len(members):
        raise SystemExit(f"{kind}: {len(lines)} symlink su {6 * len(members)} attesi")
    for stale in root.glob("gt_*"):
        stale.unlink()
    gt = {}
    if want_gt:
        names = [r["view_id"] for r in members]
        for k in ("fr", "sr"):
            src = ac.GT_DIR / f"ava256_eval_{k}.npz"
            with np.load(src) as z:
                if [str(s) for s in z["names"]] != names:
                    raise SystemExit(f"{src}: nomi diversi dalla lista di valutazione")
            shutil.copyfile(src, root / f"gt_{k}.npz")
            shutil.copyfile(src.with_suffix(".json"), root / f"gt_{k}.json")
        (root / "gt_matrix.npz").symlink_to("gt_fr.npz")
        gt = json.loads((ac.SUMMARY_DIR / "gt_summary.json").read_text())["gt"]["content_sha256"]
        gt = {"gt_fr": gt["eval_fr"], "gt_sr": gt["eval_sr"]}
    (root / ("CONFERMATIVO_NON_VALUTARE.txt" if kind == "eval" else "CALIBRAZIONE_NON_VALUTARE.txt")).write_text(MARKER[kind])
    (root / "content_sha256.txt").write_text("\n".join(sorted(lines)) + "\n")
    (root / "labels.json").write_text(json.dumps(
        {"note": "etichetta d'identita' = view_id (id950000 + riga di 256_ids.csv)",
         "labels": {f"{r['view_id']}_GTready_{t}": r["view_id"] for r in members for t in TOPOLOGIES},
         "subjects": {r["view_id"]: {"ava_id": r["ava_id"], "capture": r["capture"]} for r in members}}, indent=1) + "\n")
    dig = {"meshes": digest(lines), **{t: digest(v) for t, v in by_topo.items()}, **gt}
    man = {"kind": kind, "source_topo": str(TOPO_DIR.relative_to(ac.REPO_ROOT)), "id_offset": ac.ID_OFFSET,
           "n_subjects": len(members), "n_meshes": len(lines), "topologies": list(TOPOLOGIES), "units": "mm",
           "frame": "nativo dei dati (identities/manifest.json)", "content_digest": dig,
           "content_digest_rule": "sha256 delle righe '<mesh> <sha256(V float32, F int32)>' ordinate "
                                  "(content_sha256.txt); gt_*: ava_gt.content_sha256",
           "note": MARKER[kind].replace("\n", " ").strip()}
    if want_gt:
        man["gt"] = "gt_fr.npz / gt_sr.npz = datasets/AVA256/gt/ava256_eval_{fr,sr}.npz; gt_matrix.npz -> gt_fr.npz"
    (root / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    return {"n_subjects": len(members), "n_meshes": len(lines), "content_digest": dig,
            "topologies": {t: {"vertices": q([s[0] for s in v]), "faces": q([s[1] for s in v])} for t, v in sizes.items()}}


def stage_view() -> None:
    refuse_if_frozen()
    rows = {r["ava_id"]: r for r in json.loads((ac.GT_DIR / "ids.json").read_text())["all"]}
    split = json.loads((ac.GT_DIR / "split.json").read_text())
    name_re = re.compile(rf"^{PREFIX}(?P<num>\d+)_GTready_(?P<variant>.+)\.npz$")
    views = {"eval": build_view("eval", [rows[a] for a in split["evaluation"]], name_re, True),
             "calib": build_view("calib", [rows[a] for a in split["calibration"]], name_re, False)}
    ident = json.loads((IDENT_DIR / "manifest.json").read_text())
    crop = json.loads((TOPO_DIR / "crop_check.json").read_text())
    summ = {"split": {"evaluation": len(split["evaluation"]), "calibration": len(split["calibration"])},
            "views": views, "units": "mm", "frame": ident["frame"], "subdivision": ident["subdivision"],
            "patch": {"vertices_before_subdivision": ident["patch_vertices"], "faces_before_subdivision": ident["patch_faces"],
                      "n_vertices": ident["n_vertices"], "n_faces": ident["n_faces"],
                      "outward_area": q([r["outward_area"] for r in ident["identities"]]),
                      "n_folds": q([r["n_folds"] for r in ident["identities"]])},
            "crop_check": crop,
            "reference_famos_test_view": "aau/runs/evidence/famos/test_view.json: pieghe mediana 22 (scansioni) e 41 "
                                         "(registrazioni) su 5215 triangoli; area verso +z mediana 0.978 e 0.922"}
    C.save_json(SUMMARY, summ)
    print(json.dumps({k: (v if k != "views" else {kk: vv["content_digest"] for kk, vv in v.items()})
                      for k, v in summ.items()}, indent=1)[:3000], flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=("identities", "view"))
    a = p.parse_args()
    if a.stage == "identities":
        stage_identities()
    else:
        stage_view()


if __name__ == "__main__":
    main()
