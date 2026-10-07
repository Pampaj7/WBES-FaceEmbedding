#!/usr/bin/env python3
"""Insiemi del riconoscimento in dominio (protocollo: aau/runs/indomain_recog/protocol.md).

    python3 aau/indomain/ir_views.py      # solo stdlib: gira anche sul frontend

Riusa le viste WS2 di ``aau/cross3dmm/ws2_views.py`` (``joint__bfm``, ``joint__ict``) e ne
aggiunge una, ``datasets/INDOMAIN_RECOG/joint__rexpr6``: per ognuno degli 89 soggetti di
``joint__ict`` la neutra (``original`` di ICT) e le 5 espressioni casuali, a symlink, con
etichette ``neutral, rexpr1..5`` al posto delle topologie. Le due sottodir ``mixed/`` e
``neutral/`` di ``joint__rexpr`` restano come sono.

Il controllo che conta, ripetuto qui su ``splits.json`` invece di fidarsi delle viste: nessun
soggetto valutato e' nel training del congiunto; per il BFM-only, i soli 19 soggetti BFM che
non erano nel suo training (``bfm19``). Se no lo script si ferma.

Scrive ``aau/runs/indomain_recog/sets.json`` (insieme -> vista, etichette, soggetti) e
``ict992`` (revisione 1 del protocollo): i 992 held-out ICT del congiunto, vista
``datasets/INDOMAIN_RECOG/joint__ict992`` a symlink sulla data dir del suo training, con le 100 query
del protocollo in ``queries``.

``subjects/<insieme>.json`` nel formato di ``zs_stage.py`` (``view_dir``, ``subjects``), che
``zs_arcface_render.py`` rilegge.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
DATASETS = REPO_ROOT / "datasets"
OUT = AAU_DIR / "runs" / "indomain_recog"
SPLITS = AAU_DIR / "runs" / "ws2_cross3dmm" / "splits.json"
WS2 = DATASETS / "WS2_CROSS3DMM"
ICT_DIR = DATASETS / "ICT" / "train_ready" / "npz_withops"
ICT_REXPR_DIR = DATASETS / "ICT" / "expressions_random_withops"
REXPR6 = DATASETS / "INDOMAIN_RECOG" / "joint__rexpr6"
JOINT_DIR = DATASETS / "JOINT_BFM_ICT" / "npz_withops"
ICT992 = DATASETS / "INDOMAIN_RECOG" / "joint__ict992"
N_QUERIES_992 = 100

TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")
EXPR_LABELS = ("neutral", "rexpr1", "rexpr2", "rexpr3", "rexpr4", "rexpr5")
REXPR_RE = re.compile(r"^(id\d+)_rexpr_(\d+)\.npz$")


def view_subjects(view: Path, labels) -> list[str]:
    """Soggetti della vista, controllando che ognuno abbia esattamente le etichette attese."""
    found: dict[str, set[str]] = {}
    for p in view.glob("id*_GTready_*.npz"):
        sid, lab = p.name[:-4].split("_GTready_")
        found.setdefault(sid, set()).add(lab)
    bad = [s for s, labs in found.items() if labs != set(labels)]
    if not found or bad:
        raise SystemExit(f"{view}: etichette diverse da {labels} per {bad[:5]}")
    return sorted(found)


def build_rexpr6(subjects: list[str]) -> int:
    index: dict[str, list[str]] = {}
    for name in sorted(p.name for p in ICT_REXPR_DIR.iterdir()):
        m = REXPR_RE.match(name)
        if m:
            index.setdefault(m.group(1), []).append(m.group(2))
    REXPR6.mkdir(parents=True, exist_ok=True)
    for stale in REXPR6.glob("*.npz"):
        stale.unlink()
    n = 0
    for sid in subjects:
        if sorted(index.get(sid, [])) != ["1", "2", "3", "4", "5"]:
            raise SystemExit(f"{sid}: espressioni casuali {index.get(sid)} invece di 1..5")
        pairs = [("neutral", ICT_DIR / f"{sid}_GTready_original.npz")]
        pairs += [(f"rexpr{k}", ICT_REXPR_DIR / f"{sid}_rexpr_{k}.npz") for k in index[sid]]
        for lab, src in pairs:
            if not src.exists():
                raise SystemExit(f"mesh mancante: {src}")
            (REXPR6 / f"{sid}_GTready_{lab}.npz").symlink_to(src.resolve())
            n += 1
    return n


def build_ict992(subjects: list[str]) -> int:
    """Vista piatta dei 992 held-out ICT del congiunto, 6 topologie, dalla data dir del suo training."""
    ICT992.mkdir(parents=True, exist_ok=True)
    for stale in ICT992.glob("*.npz"):
        stale.unlink()
    for sid in subjects:
        for t in TOPOLOGIES:
            src = JOINT_DIR / f"{sid}_GTready_{t}.npz"
            if not src.exists():
                raise SystemExit(f"mesh mancante: {src}")
            (ICT992 / src.name).symlink_to(src.resolve())
    return len(subjects) * len(TOPOLOGIES)


def main() -> None:
    splits = json.loads(SPLITS.read_text())
    joint_train = set(splits["models"]["joint"]["train"])
    bfm_only_train = set(splits["models"]["bfm_only"]["train"])

    bfm = view_subjects(WS2 / "joint__bfm", TOPOLOGIES)
    ict = view_subjects(WS2 / "joint__ict", TOPOLOGIES)
    if bfm != sorted(splits["cells"]["joint__bfm"]["subjects"]) or ict != sorted(splits["cells"]["joint__ict"]["subjects"]):
        raise SystemExit("le viste WS2 non contengono i soggetti di splits.json")
    n_links = build_rexpr6(ict)
    rexpr = view_subjects(REXPR6, EXPR_LABELS)
    bfm19 = sorted(set(bfm) - bfm_only_train)
    # Galleria grande (revisione 1 del protocollo): TUTTI gli held-out ICT del congiunto. Query: 100
    # soggetti estratti con random.Random(1234) dalla lista ordinata (stdlib: lo script non usa numpy).
    ict992_all = sorted(s for s in splits["models"]["joint"]["heldout"] if int(s[2:]) >= 10000)
    if len(ict992_all) != 992 or not set(ict).issubset(ict992_all):
        raise SystemExit(f"held-out ICT del congiunto: {len(ict992_all)}, attesi 992 che contengano i 89")
    n_992 = build_ict992(ict992_all)
    ict992 = view_subjects(ICT992, TOPOLOGIES)
    queries_992 = sorted(random.Random(1234).sample(ict992, N_QUERIES_992))

    sets = {
        "bfm": {"view_dir": str(WS2 / "joint__bfm"), "labels": TOPOLOGIES, "subjects": bfm, "domain": "bfm"},
        "ict": {"view_dir": str(WS2 / "joint__ict"), "labels": TOPOLOGIES, "subjects": ict, "domain": "ict"},
        "rexpr": {"view_dir": str(REXPR6), "labels": EXPR_LABELS, "subjects": rexpr, "domain": "ict"},
        # Sottoinsieme di bfm, stesse mesh: niente vista propria.
        "bfm19": {"view_dir": str(WS2 / "joint__bfm"), "labels": TOPOLOGIES, "subjects": bfm19, "domain": "bfm",
                  "subset_of": "bfm"},
        "ict992": {"view_dir": str(ICT992), "labels": TOPOLOGIES, "subjects": ict992, "domain": "ict",
                   "queries": queries_992},
    }
    leak = {}
    for name, s in sets.items():
        subj = set(s["subjects"])
        leak[name] = {"n": len(subj), "in_joint_train": len(subj & joint_train),
                      "in_bfm_only_train": len(subj & bfm_only_train),
                      "in_joint_online_eval": len(subj & set(splits["models"]["joint"]["online_eval"])),
                      "in_bfm_only_online_eval": len(subj & set(splits["models"]["bfm_only"]["online_eval"]))}
        print(f"[ir-views] {name:<6} soggetti={len(subj):>3} nel training del congiunto={leak[name]['in_joint_train']} "
              f"del BFM-only={leak[name]['in_bfm_only_train']} (eval online del congiunto: "
              f"{leak[name]['in_joint_online_eval']}) primi={' '.join(s['subjects'][:4])}")
        if leak[name]["in_joint_train"]:
            raise SystemExit(f"{name}: LEAK nel training del congiunto")
    if leak["bfm19"]["in_bfm_only_train"] or len(bfm19) != 19:
        raise SystemExit(f"bfm19: {len(bfm19)} soggetti, {leak['bfm19']['in_bfm_only_train']} nel training del BFM-only")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sets.json").write_text(json.dumps({"sets": sets, "leak": leak, "splits": str(SPLITS)}, indent=1) + "\n")
    (OUT / "subjects").mkdir(exist_ok=True)
    for name in ("bfm", "ict", "rexpr"):
        (OUT / "subjects" / f"{name}.json").write_text(json.dumps(
            {"view_dir": sets[name]["view_dir"], "subjects": sets[name]["subjects"]}, indent=1) + "\n")
    print(f"[ir-views] vista rexpr6: {n_links} symlink in {REXPR6}; vista ict992: {n_992} symlink in {ICT992}, "
          f"{len(queries_992)} query (prime {queries_992[:3]}); scritto {OUT / 'sets.json'}")


if __name__ == "__main__":
    main()
