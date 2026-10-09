#!/usr/bin/env python3
"""Controllo delle GT FR / SR di training col loader del trainer v2, in sola lettura; poi ``READY``.

    aau/run.sh v3_work/canonical_gt/check_train.py [--which calib|raw]     (venv del trainer; train.sbatch)

Come ``v3_work/unified_gt/check_train_gt.py`` e ``aau/evidence/e1_factorial/e1_calib_ugt.py check`` (catena
importata, non modificata): ``train_steps.check_gt_scale`` (nei modi ammessi dal ``global_max``),
``install_gt_keep_scale``, ``train_v2._nan_guarded_loader`` con ``dtype=np.float64`` come ``train_runner``.
Poi: ``name_to_idx`` identico a quello della GT del run; train, held-out ed eval online sugli stessi indici;
nessun valore non finito; diagonale 0; simmetria su un campione; valori contro le distanze ricalcolate da
``fr_train.npz`` / ``sr_train.npz`` (tarata: diviso per il fattore del blocco); mediane per dominio della tarata
contro quelle della maxabs (campione, entro il 2%); ``centroid_size_bfm_ict_gnm.npz``: stessi names, S finite > 0.
Con ``--which calib`` e tutto a posto scrive ``datasets/CANONICAL_GT/train/READY`` (una riga, sha256 di FR e SR
tarate e delle S_i). Uscita: ``datasets/CANONICAL_GT/train/check_<which>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "v2_work" / "fastio", REPO / "v2_work" / "train_v2",
          REPO / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"):
    sys.path.insert(0, str(p))

import train_steps  # noqa: E402
import robustness.train_runner as tr  # noqa: E402
from intrinsic_utils import SUBJECT_RE_ANY, extract_subject_id  # noqa: E402

TRAIN_DIR = REPO / "datasets" / "CANONICAL_GT" / "train"
RUN_GT = REPO / "datasets" / "SCALE_ALL" / "gt_joint_bfm_ict_gnm.npz"
SPLIT = REPO / "aau" / "data_scale" / "split_scale_all.json"
SPACE = REPO / "datasets" / "UNIFIED_GT" / "unified_space.npz"
SIZES = TRAIN_DIR / "centroid_size_bfm_ict_gnm.npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check_file(kind: str, which: str, run_idx: dict, groups: dict, rng) -> dict:
    path = TRAIN_DIR / f"gt_{kind}_bfm_ict_gnm{'_calib' if which == 'calib' else ''}.npz"
    man = json.loads(path.with_suffix(".json").read_text())
    out = {"file": str(path), "global_max": man["global_max"]}
    train_steps.check_gt_scale(["--dist_npz", str(path)], True)
    try:
        train_steps.check_gt_scale(["--dist_npz", str(path)], False)
        out["check_gt_scale"] = "passa con e senza --gt-keep-scale"
    except SystemExit:
        out["check_gt_scale"] = "passa solo con --gt-keep-scale (massimo > 1)"
    import train_v2
    loader = train_v2._nan_guarded_loader(tr.load_gt_distance_matrix)
    D, name_to_idx = loader(str(path), dtype=np.float64)
    out["name_to_idx_identical_to_run_gt"] = name_to_idx == run_idx
    out["index_match"] = {g: {"n": len(v), "all_same_index": all(name_to_idx.get(s) == run_idx.get(s) for s in v)}
                          for g, v in groups.items()}
    raw = np.asarray(D)
    out["n_nonfinite"] = int(np.count_nonzero(~np.isfinite(raw)))
    out["diag_max_abs"] = float(np.abs(np.diag(raw)).max())
    ii, jj = rng.integers(0, len(raw), 200000), rng.integers(0, len(raw), 200000)
    out["symmetry_max_abs_sample"] = float(np.abs(raw[ii, jj] - raw[jj, ii]).max())
    # valori contro le distanze ricalcolate dalle forme
    with np.load(TRAIN_DIR / f"{kind}_train.npz") as z:
        V = z["a" if kind == "fr" else "z"]
        names = [str(s) for s in z["names"]]
        dom = z["domain"].astype(str)
    with np.load(SPACE) as z:
        A = float(z["area_total"])
    probe = list(groups["heldout"][:40]) + list(groups["online_eval"]) + list(rng.choice(groups["train"], 200, replace=False))
    pos = {s: k for k, s in enumerate(names)}
    pi = np.array([pos[s] for s in probe])
    X = V[pi].astype(np.float64)
    G = np.sqrt(np.clip(((X[:, None] - X[None]) ** 2).sum(-1) / A, 0, None))
    Dm = raw[np.ix_([name_to_idx[s] for s in probe], [name_to_idx[s] for s in probe])]
    if which == "calib":
        f = man["factor"]
        unit_key = "mm_per_unit_by_domain" if kind == "fr" else "dP_per_unit_by_domain"
        d = dom[pi]
        fac = np.array([[f[a] if a == b else np.sqrt(f[a] * f[b]) for b in d] for a in d])
        gmax = man[unit_key][d[0]] * f[d[0]]
        Dm = Dm / fac * gmax
        # mediane per dominio contro la maxabs
        meds = {}
        for dd in ("bfm", "ict", "gnm"):
            a = np.array([name_to_idx[s] for s in np.array(names)[dom == dd]])
            r, c = rng.choice(a, 2_000_000), rng.choice(a, 2_000_000)
            keep = r != c
            meds[dd] = {"calib_sample": float(np.median(raw[r[keep], c[keep]])), "maxabs_exact": man["median_maxabs"][dd]}
        out["median_by_domain"] = meds
    else:
        Dm = Dm * man["mm_per_unit" if kind == "fr" else "dP_per_unit"]
    rel = np.abs(Dm - G)[G > 0] / G[G > 0]
    out["values_vs_shapes_max_rel"] = float(rel.max())
    ok = (out["name_to_idx_identical_to_run_gt"] and out["n_nonfinite"] == 0 and out["diag_max_abs"] == 0
          and out["symmetry_max_abs_sample"] == 0 and all(v["all_same_index"] for v in out["index_match"].values())
          and out["values_vs_shapes_max_rel"] < 1e-4)
    if which == "calib":
        ok = ok and all(abs(m["calib_sample"] / m["maxabs_exact"] - 1) < 0.02 for m in out["median_by_domain"].values())
    out["ok"] = bool(ok)
    del D, raw
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--which", choices=("calib", "raw"), default="calib")
    a = p.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(1234)
    with np.load(RUN_GT) as z:
        run_names = z["names"]
    run_idx = {}
    for i, n in enumerate(run_names):
        sid = extract_subject_id(str(n), subject_re=SUBJECT_RE_ANY)
        if sid is not None:
            run_idx[sid] = i
    split = json.loads(SPLIT.read_text())
    groups = {"train": split["train"], "heldout": split["heldout"], "online_eval": split["online_eval"],
              **{f"online_eval_extra_{k}": v for k, v in split["online_eval_extra"].items()}}
    train_steps.install_gt_keep_scale()
    out = {k: check_file(k, a.which, run_idx, groups, rng) for k in ("fr", "sr")}
    with np.load(SIZES) as z:
        S, names = z["S"], z["names"]
    out["centroid_size"] = {"names_identical_to_run_gt": bool(np.array_equal(names, run_names)),
                            "n": int(len(S)), "finite_positive": bool(np.isfinite(S).all() and (S > 0).all()),
                            "min_mm": float(S.min()), "max_mm": float(S.max())}
    out["ok"] = bool(out["fr"]["ok"] and out["sr"]["ok"] and out["centroid_size"]["names_identical_to_run_gt"]
                     and out["centroid_size"]["finite_positive"])
    out["seconds"] = time.time() - t0
    (TRAIN_DIR / f"check_{a.which}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)
    if not out["ok"]:
        raise SystemExit(f"[check-fr-sr] FALLITO ({a.which})")
    if a.which == "calib":
        files = [TRAIN_DIR / "gt_fr_bfm_ict_gnm_calib.npz", TRAIN_DIR / "gt_sr_bfm_ict_gnm_calib.npz", SIZES]
        line = (f"OK {time.strftime('%Y-%m-%dT%H:%M:%S%z')} verificato col loader del trainer (check_calib.json): "
                + " ".join(f"{f.name} sha256={sha256(f)}" for f in files))
        (TRAIN_DIR / "READY").write_text(line + "\n")
        print(line, flush=True)


if __name__ == "__main__":
    main()
