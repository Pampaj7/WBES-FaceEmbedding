#!/usr/bin/env python
"""Run the usable M3DFB error estimators over the v1 pair-level benchmark.

Inputs
  - pair tables : paper_artifacts/bootstrap_ci/table1_pairlevel_exact/<topoA>__to__<topoB>/pair_metrics.csv
                  (subject_a, subject_b, topology_a, topology_b, gt_distance,
                   latent_distance, raw_chamfer)
  - meshes      : datasets/REMESH/npz_data_topo_500/<sid>_GTready_<topo>.npz  (V/F)

Outputs
  - v2_work/m3dfb/pair_metrics/<pair_label>/pair_metrics.csv   (v1 columns + one
    column per estimator; resumable -- an existing file with all requested
    columns is skipped)
  - v2_work/m3dfb/m3dfb_summary.csv    Spearman vs gt_distance per estimator,
    per topology pair and OVERALL

Meshes are normalized per mesh exactly as the v1 benchmark did (centre on the
vertex mean, divide by max|coord| -- see v2_work/phase0/normalization_confound.py),
so the magnitudes are comparable with the stored raw_chamfer column.

Landmarks: M3DFB has no landmark predictor, and both of its rigid aligners are
landmark-driven, so landmarks are mandatory. Our `original`/`noisy` topologies ARE
BFM p23470 in M3DFB's own vertex order (verified: Procrustes residual d=0.0015 vs
0.9999 for a shuffled control), so their 51 iBUG landmarks are exact. For the
other four topologies the landmarks are transferred by nearest vertex from the
same subject's `original` mesh. This is auxiliary information a real
cross-topology evaluation would not have -- see INVENTORY.md.

Usage
  .conda_env/bin/python v2_work/m3dfb/run_m3dfb_pairs.py --n-subjects 5
  .conda_env/bin/python v2_work/m3dfb/run_m3dfb_pairs.py --n-subjects 30 --estimators E1,E9
"""
from __future__ import annotations

import argparse
import csv
import functools
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

import m3dfb_adapter as m3  # noqa: E402

PAIR_TABLE_ROOT = REPO_ROOT / "paper_artifacts" / "bootstrap_ci" / "table1_pairlevel_exact"
MESH_ROOT = REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500"
OUT_PAIRS = THIS_DIR / "pair_metrics"
OUT_SUMMARY = THIS_DIR / "m3dfb_summary.csv"

BFM_TOPOLOGIES = ("original", "noisy")  # the only ones with a shared template
V1_COLUMNS = ("subject_a", "subject_b", "topology_a", "topology_b",
              "gt_distance", "latent_distance", "raw_chamfer")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-subjects", type=int, default=30,
                   help="use the first N subject ids (full set is 100 -> 4950 pairs/topo-pair)")
    p.add_argument("--estimators", default=",".join(m3.estimator_names()),
                   help="comma list; default = all estimators usable at benchmark scale")
    p.add_argument("--pair-labels", default="", help="comma list of <topoA>__to__<topoB> (default: all)")
    p.add_argument("--out-pairs", default=str(OUT_PAIRS))
    p.add_argument("--out-summary", default=str(OUT_SUMMARY))
    p.add_argument("--force", action="store_true", help="ignore cached per-pair CSVs")
    p.add_argument("--timing", action="store_true", help="print ms/pair per estimator")
    return p.parse_args()


def normalize_maxabs(V: np.ndarray) -> np.ndarray:
    """v1 benchmark normalization: centre on vertex mean, divide by max|coord|."""
    Vn = V - V.mean(axis=0, keepdims=True)
    return Vn / max(float(np.abs(Vn).max()), 1e-9)


@functools.lru_cache(maxsize=256)
def _load_raw(sid: str, topo: str):
    with np.load(MESH_ROOT / f"{sid}_GTready_{topo}.npz") as d:
        return np.asarray(d["V"], dtype=np.float64), np.asarray(d["F"])


@functools.lru_cache(maxsize=256)
def load_mesh(sid: str, topo: str):
    """Normalized (V, F, landmark coords, landmark indices) for one mesh.

    The landmark transfer runs on the *raw* coordinates, where all six topology
    variants of a subject live in one common world frame; normalization is a
    per-mesh similarity transform and would otherwise perturb the matching.
    """
    V, F = _load_raw(sid, topo)
    if topo in BFM_TOPOLOGIES:
        idx = m3.bfm_landmark_indices()
    else:
        idx = m3.transfer_landmarks(V, _load_raw(sid, "original")[0])
    V = normalize_maxabs(V)
    return V, F, V[idx], idx


@functools.lru_cache(maxsize=64)
def read_pair_rows(label: str, n_subjects: int) -> tuple[dict, ...]:
    """Rows of one topology-pair table, restricted to the first N subjects.

    The 100 benchmark subjects are a sparse subset of id0000..id0499, so the
    subset is 'first N ids present in the table', matching
    v2_work/phase0/normalization_confound.py.
    """
    with open(PAIR_TABLE_ROOT / label / "pair_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    keep = set(sorted({r["subject_a"] for r in rows} | {r["subject_b"] for r in rows})[:n_subjects])
    return tuple(r for r in rows if r["subject_a"] in keep and r["subject_b"] in keep)


def applicable(name: str, topo_a: str) -> bool:
    """ETC-based estimators need a BFM template on the reconstruction side."""
    return (not m3.ESTIMATORS[name]["needs_bfm_template"]) or topo_a in BFM_TOPOLOGIES


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(spearmanr(x[m], y[m]).statistic)


def run_pair_label(label: str, estimators: list[str], n_subjects: int,
                   out_dir: Path, force: bool, timing: bool) -> list[dict]:
    out_csv = out_dir / label / "pair_metrics.csv"
    cols = list(V1_COLUMNS) + estimators
    if out_csv.exists() and not force:
        with open(out_csv, newline="") as f:
            rd = csv.DictReader(f)
            cached = list(rd)
        if cached and all(c in (rd.fieldnames or []) for c in cols) \
                and len(cached) == len(read_pair_rows(label, n_subjects)):
            print(f"  {label}: cached ({len(cached)} pairs)", flush=True)
            return cached

    rows = read_pair_rows(label, n_subjects)
    spent = {e: 0.0 for e in estimators}
    counted = {e: 0 for e in estimators}
    t_label = time.time()
    out_rows = []
    for i, r in enumerate(rows):
        VA, FA, LA, iA = load_mesh(r["subject_a"], r["topology_a"])
        VB, FB, LB, _ = load_mesh(r["subject_b"], r["topology_b"])
        rec = {c: r.get(c, "") for c in V1_COLUMNS}
        for name in estimators:
            if not applicable(name, r["topology_a"]):
                rec[name] = ""
                continue
            t0 = time.time()
            try:
                rec[name] = pretty(m3.pair_distance(
                    name, VA, FA, VB, FB,
                    lmks_a=LA, lmks_b=LB, lmk_indices_a=iA))
            except Exception as exc:  # keep the sweep going, record the hole
                print(f"    {label} {r['subject_a']}/{r['subject_b']} {name}: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                rec[name] = ""
            spent[name] += time.time() - t0
            counted[name] += 1
        out_rows.append(rec)
        if i and i % 100 == 0:
            print(f"    {label}: {i}/{len(rows)} ({(time.time()-t_label)/i:.2f} s/pair)",
                  flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    dt = time.time() - t_label
    print(f"  {label}: {len(out_rows)} pairs in {dt:.1f}s ({dt/max(len(out_rows),1):.2f} s/pair)",
          flush=True)
    if timing:
        for e in estimators:
            if counted[e]:
                print(f"      {e:6s} {1000*spent[e]/counted[e]:8.1f} ms/pair  "
                      f"({counted[e]} pairs)", flush=True)
    return out_rows


def pretty(x: float) -> str:
    return "" if not np.isfinite(x) else f"{x:.8g}"


def main():
    args = parse_args()
    estimators = [e.strip() for e in args.estimators.split(",") if e.strip()]
    unknown = [e for e in estimators if e not in m3.ESTIMATORS]
    if unknown:
        raise SystemExit(f"unknown estimators: {unknown}\navailable: {sorted(m3.ESTIMATORS)}")
    labels = ([l.strip() for l in args.pair_labels.split(",") if l.strip()]
              or sorted(p.name for p in PAIR_TABLE_ROOT.iterdir()
                        if p.is_dir() and (p / "pair_metrics.csv").exists()))
    out_dir = Path(args.out_pairs)

    print(f"M3DFB sweep: {len(labels)} topology pairs, {args.n_subjects} subjects, "
          f"estimators {estimators}", flush=True)
    all_rows = []
    summary = []
    for label in labels:
        rows = run_pair_label(label, estimators, args.n_subjects, out_dir,
                              args.force, args.timing)
        all_rows.extend(rows)
        gt = [float(r["gt_distance"]) for r in rows]
        rec = {"pair_label": label, "n_pairs": len(rows)}
        for col in ("latent_distance", "raw_chamfer", *estimators):
            vals = [float(r[col]) if r.get(col) not in (None, "") else np.nan for r in rows]
            rec[f"spearman_{col}"] = spearman(vals, gt)
            rec[f"n_{col}"] = int(np.isfinite(vals).sum())
        summary.append(rec)

    gt = [float(r["gt_distance"]) for r in all_rows]
    overall = {"pair_label": "OVERALL", "n_pairs": len(all_rows)}
    for col in ("latent_distance", "raw_chamfer", *estimators):
        vals = [float(r[col]) if r.get(col) not in (None, "") else np.nan for r in all_rows]
        overall[f"spearman_{col}"] = spearman(vals, gt)
        overall[f"n_{col}"] = int(np.isfinite(vals).sum())
    summary.insert(0, overall)

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {args.out_summary}")
    hdr = ["latent_distance", "raw_chamfer", *estimators]
    print("pair_label".ljust(24) + "n".rjust(7)
          + "".join(f"{h[:14]:>16s}" for h in hdr))
    for rec in summary:
        print(rec["pair_label"].ljust(24) + f"{rec['n_pairs']:7d}"
              + "".join(f"{rec[f'spearman_{h}']:16.3f}" for h in hdr))


if __name__ == "__main__":
    main()
