#!/usr/bin/env python3
"""Viste di valutazione di Ava-256 (sola geometria, nessun operatore, nessun embedding).

    v3_work/unified_gt/run.sh aau/ava256/ava_views.py identities       (dopo ava_gt.py)
    aau/run.sh aau/zs3dmm/make_zs_topologies.py --prefix ava --in-dir datasets/AVA256/identities \\
        --out-dir datasets/AVA256/topo --n-cores 32                     (lo stesso codice di HIFI3D e FaceVerse)
    aau/run.sh aau/zs3dmm/zs_check_crop.py --topo-dir datasets/AVA256/topo --prefix ava --write
    v3_work/unified_gt/run.sh aau/ava256/ava_views.py view              (vista + GT rinominate)

CONFERMATIVO: non valutare prima del protocollo confermativo (README). Ava-256 non va aggiunto a nessuno strumento di
valutazione esistente; la vista porta un file ``CONFERMATIVO_NON_VALUTARE.txt``.

``identities`` (README, scelta 7): per ogni soggetto valido (ordine delle GT, ``datasets/AVA256/gt/ids.json``) la
neutra di ava_neutral.py in mm (``u`` di ava_gt.py) nel frame NATIVO dei dati (frame della testa del tracking di
Meta: nessuna rigida della GT negli ingressi), ristretta alla patch di ava_corr.py (``patch_F``, lo stesso insieme di
indici per tutti) e ricompattata: ``identities/avaNNNN.npz`` (``V`` float32, ``F`` int32, le chiavi di
``zs_identities.py``). Controlli per patch: ``now_prepare_meshes.patch_quality`` (frazione d'area con la normale verso
+z, pieghe: triangoli adiacenti a piu' di 120 gradi), il controllo delle viste FaMoS.

``view``: ``eval_view/npz/id<950000 + NNNN>_GTready_<topologia>.npz`` (symlink a ``topo/``, come
``aau/zs3dmm/make_zs_view.py``, che non si puo' usare tal quale: vuole le GT maxabs e dei coefficienti, che per
dati reali non esistono), ``gt_fr.npz`` / ``gt_sr.npz`` (copie delle GT di ava_gt.py, stessi nomi) e
``gt_matrix.npz`` -> ``gt_fr.npz`` (la GT di riferimento FR), ``labels.json``, ``manifest.json``; riassunto in
``aau/ava256/views_summary.json`` (conteggi, nessuna geometria).
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
for _p in (THIS_DIR, THIS_DIR.parents[1] / "v3_work" / "unified_gt", THIS_DIR.parent / "recon"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ava_common as ac  # noqa: E402
import ugt as C  # noqa: E402
from now_prepare_meshes import patch_quality  # noqa: E402

PREFIX = "ava"
IDENT_DIR = ac.DATA_ROOT / "identities"
TOPO_DIR = ac.DATA_ROOT / "topo"
VIEW_DIR = ac.DATA_ROOT / "eval_view"
CORR_NPZ = ac.DATA_ROOT / "corr" / "ava256.npz"
SUMMARY = ac.SUMMARY_DIR / "views_summary.json"
TOPOLOGIES = ("original", "remesh", "crop", "noisy", "down8k", "up60k")
MARKER = ("CONFERMATIVO: non valutare prima del protocollo confermativo (aau/ava256/README.md,\n"
          "aau/ava256/PROTOCOL_CONFERMATIVO_bozza.md). Ava-256 non va aggiunto a nessuno strumento di valutazione.\n")


def gt_ids() -> list[dict]:
    return json.loads((ac.GT_DIR / "ids.json").read_text())["ids"]


def q(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"min": float(x.min()), "median": float(np.median(x)), "max": float(x.max())}


def mesh_sha256(path: Path) -> str:
    """sha256 del contenuto di una mesh (``V`` float32 e ``F`` int32): stabile fra riesecuzioni, il file npz no."""
    with np.load(path) as z:
        h = hashlib.sha256(np.ascontiguousarray(z["V"], dtype="<f4").tobytes())
        h.update(np.ascontiguousarray(z["F"], dtype="<i4").tobytes())
    return h.hexdigest()


def stage_identities() -> None:
    u = json.loads((ac.GT_DIR / "frame.json").read_text())["u"]
    with np.load(CORR_NPZ) as z:
        patch_F = z["patch_F"].astype(np.int64)
    used = np.unique(patch_F)
    remap = -np.ones(ac.N_VERTS, dtype=np.int64)
    remap[used] = np.arange(len(used))
    Fp = remap[patch_F].astype(np.int32)
    IDENT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in IDENT_DIR.glob(f"{PREFIX}[0-9]*.npz"):
        stale.unlink()
    rows = []
    for r in gt_ids():
        with np.load(ac.NEUTRAL_DIR / f"{r['ava_id']}.npz") as z:
            V = u * z["V_neutral"].astype(np.float64)[used]
        if not np.isfinite(V).all():
            raise SystemExit(f"{r['ava_id']}: vertici non finiti")
        out_frac, folds = patch_quality(V, Fp)
        np.savez_compressed(IDENT_DIR / f"{r['ava_id']}.npz", V=V.astype(np.float32), F=Fp)
        rows.append({**r, "outward_area": out_frac, "n_folds": folds})
    man = {"source": "datasets/AVA256/neutral (ava_neutral.py), patch_F di datasets/AVA256/corr/ava256.npz",
           "units": "mm", "u": u, "frame": "nativo dei dati: frame della testa del tracking di Meta (+y alto, +z avanti)",
           "n_identities": len(rows), "n_vertices": int(len(used)), "n_faces": int(len(Fp)), "prefix": PREFIX,
           "identities": rows}
    C.save_json(IDENT_DIR / "manifest.json", man)
    print(f"[ava-views] {len(rows)} identita': patch {len(used)} vertici / {len(Fp)} triangoli; pieghe "
          f"{q([r['n_folds'] for r in rows])}, area verso +z {q([r['outward_area'] for r in rows])}", flush=True)


def stage_view() -> None:
    ids = gt_ids()
    name_re = re.compile(rf"^{PREFIX}(?P<num>\d+)_GTready_(?P<variant>.+)\.npz$")
    data = VIEW_DIR / "npz"
    data.mkdir(parents=True, exist_ok=True)
    for stale in data.glob("*.npz"):
        stale.unlink()
    want = {r["ava_id"]: r["view_id"] for r in ids}
    n_link, sizes, hashes = 0, {t: [] for t in TOPOLOGIES}, {t: [] for t in TOPOLOGIES}
    for p in sorted(TOPO_DIR.glob(f"{PREFIX}*_GTready_*.npz")):
        m = name_re.match(p.name)
        if not m or f"{PREFIX}{m['num']}" not in want:
            continue
        (data / f"{want[PREFIX + m['num']]}_GTready_{m['variant']}.npz").symlink_to(p.resolve())
        n_link += 1
        with np.load(p) as z:
            sizes[m["variant"]].append((len(z["V"]), len(z["F"])))
        hashes[m["variant"]].append(f"{want[PREFIX + m['num']]}_GTready_{m['variant']} {mesh_sha256(p)}")
    if n_link != 6 * len(ids):
        raise SystemExit(f"{n_link} symlink su {6 * len(ids)} attesi")
    names = [r["view_id"] for r in ids]
    for kind in ("fr", "sr"):
        src = ac.GT_DIR / f"ava256_{kind}.npz"
        with np.load(src) as z:
            if [str(s) for s in z["names"]] != names:
                raise SystemExit(f"{src}: nomi diversi da ids.json")
        shutil.copyfile(src, VIEW_DIR / f"gt_{kind}.npz")
        shutil.copyfile(src.with_suffix(".json"), VIEW_DIR / f"gt_{kind}.json")
    link = VIEW_DIR / "gt_matrix.npz"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to("gt_fr.npz")
    (VIEW_DIR / "CONFERMATIVO_NON_VALUTARE.txt").write_text(MARKER)
    (VIEW_DIR / "labels.json").write_text(json.dumps(
        {"note": "etichetta d'identita' = view_id (id950000 + riga di 256_ids.csv)",
         "labels": {f"{r['view_id']}_GTready_{t}": r["view_id"] for r in ids for t in TOPOLOGIES},
         "subjects": {r["view_id"]: {"ava_id": r["ava_id"], "capture": r["capture"]} for r in ids}}, indent=1) + "\n")
    crop = json.loads((TOPO_DIR / "crop_check.json").read_text())
    # impronta del contenuto: sha256 delle righe '<mesh> <sha256 di V e F>' ordinate, per topologia e in tutto
    digest = {t: hashlib.sha256(("\n".join(sorted(v)) + "\n").encode()).hexdigest() for t, v in hashes.items()}
    digest["all"] = hashlib.sha256(("\n".join(sorted(x for v in hashes.values() for x in v)) + "\n").encode()).hexdigest()
    (VIEW_DIR / "content_sha256.txt").write_text("\n".join(sorted(x for v in hashes.values() for x in v)) + "\n")
    man = {"source_topo": str(TOPO_DIR.relative_to(ac.REPO_ROOT)), "id_offset": ac.ID_OFFSET,
           "n_subjects": len(ids), "n_symlinks": n_link, "id_range": [names[0], names[-1]],
           "topologies": list(TOPOLOGIES), "units": "mm", "frame": "nativo dei dati (identities/manifest.json)",
           "gt": "gt_fr.npz / gt_sr.npz = datasets/AVA256/gt/ava256_{fr,sr}.npz (D_orig / massimo, json con "
                 "l'unita'); gt_matrix.npz -> gt_fr.npz", "crop_check": crop,
           "content_digest": digest, "content_digest_rule": "sha256 delle righe '<mesh> <sha256(V float32, F int32)>' "
                                                            "ordinate (content_sha256.txt)",
           "note": "CONFERMATIVO: vista di sola geometria, nessun operatore; non valutare prima del protocollo"}
    (VIEW_DIR / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    ident = json.loads((IDENT_DIR / "manifest.json").read_text())
    summ = {"n_subjects": len(ids), "n_meshes": n_link, "id_range": man["id_range"],
            "patch": {"n_vertices": ident["n_vertices"], "n_faces": ident["n_faces"],
                      "outward_area": q([r["outward_area"] for r in ident["identities"]]),
                      "n_folds": q([r["n_folds"] for r in ident["identities"]])},
            "topologies": {t: {"vertices": q([s[0] for s in v]), "faces": q([s[1] for s in v])}
                           for t, v in sizes.items()},
            "crop_check": crop, "units": "mm", "frame": man["frame"], "content_digest": digest,
            "content_digest_rule": man["content_digest_rule"],
            "reproducibility": "seconda esecuzione completa (job 1068023) contro la prima: GT entro 1.3e-4 mm, mappa "
                               "entro 0.0025 mm (3 punti su 1478 con indici diversi), original/remesh/crop/noisy entro "
                               "1e-5 mm, down8k entro 6e-5 mm, up60k con tassellazione diversa: le viste si congelano "
                               "per contenuto (content_digest), non per rigenerazione",
            "reference_famos_test_view": "aau/runs/evidence/famos/test_view.json: pieghe mediana 22 (scansioni) e 41 "
                                         "(registrazioni); area verso +z mediana 0.978 e 0.922"}
    C.save_json(SUMMARY, summ)
    print(json.dumps(summ, indent=1)[:2500], flush=True)


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
