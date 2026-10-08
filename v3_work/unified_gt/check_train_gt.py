#!/usr/bin/env python3
"""Controllo della GT unificata di training col loader del trainer v2, in sola lettura.

    aau/run.sh v3_work/unified_gt/check_train_gt.py      (venv del trainer; train_gt.sbatch)

La catena di lettura del run su scala, importata e non modificata:
  - ``train_steps.check_gt_scale``: la guardia del json (con e senza ``--gt-keep-scale``);
  - ``train_steps.install_gt_keep_scale`` (il run usa ``--gt-keep-scale``), poi
    ``train_v2._nan_guarded_loader`` (``SUBJECT_RE_ANY`` e guardia NaN), chiamato come lo chiama
    ``train_runner`` (``load_gt_distance_matrix(args.dist_npz, dtype=np.float64)``).
Poi: ``name_to_idx`` identico a quello che lo stesso codice di parsing da' sulla GT del run;
held-out e soggetti della valutazione online (``split_scale_all.json``) sugli stessi indici; nessun
NaN, diagonale 0, simmetria; letture di blocchi monodominio attraverso la guardia; valori uguali alle
distanze ricalcolate da ``s_train.npz`` (in mm, via ``mm_per_unit``).
Scrive ``datasets/UNIFIED_GT/train/check.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "v2_work" / "fastio", REPO / "v2_work" / "train_v2",
          REPO / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"):
    sys.path.insert(0, str(p))

import train_steps  # noqa: E402
import robustness.train_runner as tr  # noqa: E402
from intrinsic_utils import SUBJECT_RE_ANY, extract_subject_id  # noqa: E402

TRAIN_DIR = REPO / "datasets" / "UNIFIED_GT" / "train"
NEW = TRAIN_DIR / "gt_unified_bfm_ict_gnm.npz"
RUN_GT = REPO / "datasets" / "SCALE_ALL" / "gt_joint_bfm_ict_gnm.npz"
SPLIT = REPO / "aau" / "data_scale" / "split_scale_all.json"


def main() -> None:
    out = {}
    # guardia del json, nei due modi
    train_steps.check_gt_scale(["--dist_npz", str(NEW)], True)
    train_steps.check_gt_scale(["--dist_npz", str(NEW)], False)
    out["check_gt_scale"] = "passa con e senza --gt-keep-scale"
    train_steps.install_gt_keep_scale()
    import train_v2
    loader = train_v2._nan_guarded_loader(tr.load_gt_distance_matrix)
    D, name_to_idx = loader(str(NEW), dtype=np.float64)
    out["shape"] = list(D.shape)
    with np.load(RUN_GT) as z:
        run_names = z["names"]
    run_idx = {}
    for i, n in enumerate(run_names):
        sid = extract_subject_id(str(n), subject_re=SUBJECT_RE_ANY)
        if sid is not None:
            run_idx[sid] = i
    out["name_to_idx_identical_to_run_gt"] = name_to_idx == run_idx
    split = json.loads(SPLIT.read_text())
    groups = {"heldout": split["heldout"], "online_eval": split["online_eval"],
              **{f"online_eval_extra_{k}": v for k, v in split["online_eval_extra"].items()},
              "train": split["train"]}
    out["index_match"] = {}
    for g, ids in groups.items():
        ok = all(s in name_to_idx and name_to_idx[s] == run_idx[s] for s in ids)
        out["index_match"][g] = {"n": len(ids), "all_same_index": bool(ok)}
    raw = np.asarray(D)
    out["n_nonfinite"] = int(np.count_nonzero(~np.isfinite(raw)))
    out["diag_max_abs"] = float(np.abs(np.diag(raw)).max())
    rng = np.random.default_rng(0)
    ii, jj = rng.integers(0, len(raw), 200000), rng.integers(0, len(raw), 200000)
    out["symmetry_max_abs_sample"] = float(np.abs(raw[ii, jj] - raw[jj, ii]).max())
    out["max"] = float(raw.max())
    # letture attraverso la guardia: blocchi monodominio (come i batch del trainer) e fra domini
    dom = {"bfm": [s for s in name_to_idx if int(s[2:]) < 1000],
           "ict": [s for s in name_to_idx if 10000 <= int(s[2:]) < 100000],
           "gnm": [s for s in name_to_idx if int(s[2:]) >= 100000]}
    for d, ids in dom.items():
        sel = np.array([name_to_idx[s] for s in rng.choice(ids, 5, replace=False)])
        _ = D[np.ix_(sel, sel)]
    sel = np.array([name_to_idx["id0001"], name_to_idx["id20000"], name_to_idx["id100000"]])
    _ = D[np.ix_(sel, sel)]
    out["guarded_reads"] = "blocchi monodominio e fra domini letti senza errori (nessun NaN)"
    # valori contro s_train.npz
    man = json.loads(NEW.with_suffix(".json").read_text())
    with np.load(TRAIN_DIR / "s_train.npz") as z:
        S = z["s"].astype(np.float64)
        names = [str(x) for x in z["names"]]
    with np.load(REPO / "datasets" / "UNIFIED_GT" / "unified_space.npz") as z:
        A = float(z["area_total"])
    probe = (groups["heldout"][:40] + groups["online_eval"] + groups["online_eval_extra_gnm"]
             + list(rng.choice(groups["train"], 200, replace=False)))
    pi = np.array([names.index(s) for s in probe])
    G = np.sqrt(((S[pi][:, None] - S[pi][None]) ** 2).sum(-1) / A)
    Dm = raw[np.ix_([name_to_idx[s] for s in probe], [name_to_idx[s] for s in probe])] * man["mm_per_unit"]
    out["values_vs_s_train_max_abs_mm"] = float(np.abs(G - Dm).max())
    out["mm_per_unit"] = man["mm_per_unit"]
    (TRAIN_DIR / "check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)
    bad = (not out["name_to_idx_identical_to_run_gt"] or out["n_nonfinite"] or out["diag_max_abs"] > 0
           or out["symmetry_max_abs_sample"] > 0 or not all(v["all_same_index"] for v in out["index_match"].values())
           or out["values_vs_s_train_max_abs_mm"] > 1e-3)
    if bad:
        raise SystemExit("[check-train-gt] FALLITO")
    print("[check-train-gt] OK", flush=True)


if __name__ == "__main__":
    main()
