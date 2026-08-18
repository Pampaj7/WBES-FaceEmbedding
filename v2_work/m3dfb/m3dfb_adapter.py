#!/usr/bin/env python
"""Uniform wrapper around the M3DFB error estimators.

M3DFB = "A Modular 3D Face Reconstruction Benchmark" (Sariyanidi et al., IEEE FG
2025), cloned read-only at `external/M3DFB`.  Upstream is a *pipeline* framework:
an estimator is a recipe over five stages (mesh cropping, rigid alignment,
non-rigid warping, point correspondence, distance computation, correction).  The
paper's 16 estimators E1..E16 are the cross product

    rigid {ICP, RLR} x nonrigid {none, ELR, NICP, ELR+NICP} x corrector {none, ETC}

with Chamfer correspondence and dense point-to-point distance.  The repo ships
only 4 of the 16 as JSON (E01, E08, E12, E16); this module reconstructs all 16
from the same component classes, and records for each whether it can actually
run on cross-topology mesh pairs.  See INVENTORY.md for the per-estimator
verdict and the reasoning.

Public API
    estimator_names(include_slow=False, include_unusable=False) -> list[str]
    pair_distance(name, VA, FA, VB, FB, **kw) -> float
    ESTIMATORS: dict[str, dict]      # full table incl. unusable entries
    bfm_landmark_indices() -> np.ndarray        # 51 iBUG indices into BFM p23470
    transfer_landmarks(V, V_bfm) -> np.ndarray  # nearest-vertex landmark transfer

Convention: VA/FA is the *reconstruction* R, VB/FB the *ground truth* G, matching
M3DFB's `ErrorComputer.compute(R, G, Rlmks, Glmks)`.  The estimators are
asymmetric, and the error is reported per R-vertex.

Faces (FA/FB) are accepted but unused: M3DFB is point-based throughout
("we do not use triangles" -- upstream README).
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
M3DFB_ROOT = REPO_ROOT / "external" / "M3DFB"
if str(M3DFB_ROOT) not in sys.path:
    sys.path.insert(0, str(M3DFB_ROOT))

import facebenchmark.performance_reporters.error_computation as ecm  # noqa: E402
from facebenchmark.performance_reporters.error_computation import (  # noqa: E402
    distance_computers as _dc,
)

# --------------------------------------------------------------------------
# Two patches on the upstream code.  Both are documented in INVENTORY.md.
# --------------------------------------------------------------------------
# (1) ChamferCorrespondence.establish is a python loop over N vertices, each doing
#     a full N x M distance evaluation.  A KD-tree returns the identical argmin
#     (ties aside, which float coordinates do not produce) ~1000x faster.  Without
#     this, one 20k-vs-20k pair takes minutes instead of ~20 ms.
ecm.ChamferCorrespondence.establish = (
    lambda self, X, Y: cKDTree(Y).query(X, k=1)[1].astype(int)
)
# (2) DenseP2TriDistance.compute calls `ptd.pointTriangleDistance`, but the name
#     `ptd` is never imported anywhere in the repo -> NameError on every call.
#     Bind it to its own module so the function resolves.
_dc.ptd = _dc


# --------------------------------------------------------------------------
# Estimator table
# --------------------------------------------------------------------------
_RIGID = {
    "ICP": {"type": "ICPRigidAligner", "opts": {}},
    "RLR": {"type": "LandmarkBasedRigidAligner", "opts": {}},
}
_NONRIGID = [
    ("none", None),
    ("ELR", {"type": "LandmarkBasedElasticAligner", "opts": {"gamma": 1.0}}),
    ("NICP", {"type": "NonrigidICPAligner", "opts": {"epsilon": 0.01, "prealign_ELR": 0}}),
    ("ELR+NICP", {"type": "NonrigidICPAligner", "opts": {"epsilon": 0.01, "prealign_ELR": 1}}),
]
_CORRECTOR = [
    ("none", None),
    ("ETC", {"type": "TopologyConsistencyCorrector", "opts": {}}),
]

_ELR_REASON = (
    "LandmarkBasedElasticAligner builds squareform(pdist(R)), a dense N x N matrix, "
    "then solves 3 cvxpy/SCS programs. Measured 3.3 s / 1.1 GB at N=8k and 53 s / "
    "6.5 GB at N=23k; both cost and memory are O(N^2), so N=60k projects to ~350 s "
    "and ~43 GB. Runnable on single examples, not over 13k+ pairs on one core."
)
_NICP_REASON = (
    "NonrigidICPAligner is runnable (Delaunay on R's xy-projection, so it needs no "
    "faces) but measured 185 s per pair at N=8k on one core -- fine for a handful of "
    "examples, not for 13k+ pairs."
)
_ETC_NOTE = (
    "TopologyConsistencyCorrector needs a morphable-model template for the R side "
    "(mean_face_shape + lmk_indices, and its per-vertex weight vector must have "
    "length N_R). Only applicable when R is a BFM p23470 mesh."
)


def _build_estimators() -> dict:
    out = {}
    for ri, rname in enumerate(("ICP", "RLR")):
        for ni, (nname, nonrigid) in enumerate(_NONRIGID):
            for ci, (cname, corrector) in enumerate(_CORRECTOR):
                num = ri * 8 + ni * 2 + ci + 1
                if "ELR" in nname:
                    status, reason = "slow", _ELR_REASON
                elif "NICP" in nname:
                    status, reason = "slow", _NICP_REASON
                elif corrector is not None:
                    status, reason = "template", _ETC_NOTE
                else:
                    status, reason = "ok", ""
                out[f"E{num}"] = {
                    "recipe": {
                        "name": f"E{num}",
                        "mesh_cropper": None,
                        "rigid_aligner": _RIGID[rname],
                        "nonrigid_aligner": nonrigid,
                        "corr_establisher": {"type": "ChamferCorrespondence", "opts": {}},
                        "distance_computer": {"type": "DenseP2PDistance", "opts": {}},
                        "corrector": corrector,
                    },
                    "label": f"{rname}+{nname}+P2P+{cname}",
                    "rigid": rname,
                    "nonrigid": nname,
                    "corrector": cname,
                    "distance": "P2P",
                    "needs_landmarks": True,  # both rigid aligners are landmark-driven
                    "needs_bfm_template": corrector is not None,
                    "status": status,
                    "reason": reason,
                }
    # P2Tri variants of the two landmark-only-rigid estimators, for the record.
    for base, rname in (("E1", "ICP"), ("E9", "RLR")):
        spec = {k: v for k, v in out[base].items()}
        spec["recipe"] = dict(out[base]["recipe"])
        spec["recipe"] = {
            **spec["recipe"],
            "name": f"{base}_p2tri",
            "distance_computer": {"type": "DenseP2TriDistance", "opts": {}},
        }
        spec["label"] = f"{rname}+none+P2Tri+none"
        spec["distance"] = "P2Tri"
        spec["status"] = "unusable"
        spec["reason"] = (
            "DenseP2TriDistance is broken as shipped: compute() calls "
            "`ptd.pointTriangleDistance` and the name `ptd` is never imported anywhere "
            "in the repo, so every call raises NameError. Patched here, it still does a "
            "full argsort over all N corresponding points for each of the N points -- "
            "O(N^2 log N) in a python loop -- and it builds its 'triangle' from the "
            "3 nearest corresponding points rather than from a real mesh face."
        )
        out[f"{base}_p2tri"] = spec
    return out


ESTIMATORS = _build_estimators()

#: Component classes upstream ships that no recipe here can use at all.
INAPPLICABLE_COMPONENTS = {
    "IdentityCorrespondence": "Assumes R and G are vertex-wise the same mesh "
    "(pidx = arange(N)). This is M3DFB's oracle 'True' error for synthetic data; it "
    "cannot be evaluated on two meshes with different topologies.",
    "LandmarksDistance": "Reports the distance between the 51 landmark points only. "
    "Our landmarks are derived/transferred, not annotated, so this would measure the "
    "landmark transfer rather than the meshes.",
    "PointBasedCropper": "Broken as shipped: __init__ asserts opts['dist_threshold_ratio'] "
    "but crop() reads opts['dist_threshold'] -> KeyError. Also needs dataset-specific "
    "landmark indices with no defaults.",
}


def estimator_names(include_slow: bool = False, include_unusable: bool = False) -> list[str]:
    """Usable estimator names, in E-number order.

    Default = the ones runnable over the full pair sets on one core.
    `include_slow` adds the NICP recipes (minutes per pair).
    """
    keep = {"ok", "template"}
    if include_slow:
        keep.add("slow")
    if include_unusable:
        keep.add("unusable")
    names = [n for n, s in ESTIMATORS.items() if s["status"] in keep]
    return sorted(names, key=lambda n: (int(n[1:].split("_")[0]), n))


# --------------------------------------------------------------------------
# BFM p23470 template (the only shared-topology template we have)
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def bfm_mm() -> dict:
    """M3DFB's BFM p23470 morphable-model info, with mean_face_shape as ndarray."""
    with open(M3DFB_ROOT / "info" / "BFM-p23470.json") as f:
        mm = json.load(f)
    mm["mean_face_shape"] = np.asarray(mm["mean_face_shape"], dtype=np.float64)
    return mm


def bfm_landmark_indices() -> np.ndarray:
    """The 51 iBUG-51 vertex indices into a BFM p23470 mesh."""
    return np.asarray(bfm_mm()["lmk_indices"], dtype=np.int64)


def transfer_landmarks(V: np.ndarray, V_bfm: np.ndarray) -> np.ndarray:
    """Landmark vertex indices for `V`, by nearest vertex to the BFM landmarks.

    `V_bfm` must be a BFM p23470 mesh of the *same* subject in the same frame.
    Returns indices into `V` (so callers can cache indices, not coordinates).
    """
    li = bfm_landmark_indices()
    return cKDTree(V).query(V_bfm[li], k=1)[1].astype(np.int64)


# --------------------------------------------------------------------------
# Running an estimator on a pair
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _computer(name: str, n_a: int):
    spec = ESTIMATORS[name]
    if spec["needs_bfm_template"]:
        mm = bfm_mm()
        if n_a != mm["Npoints"]:
            raise ValueError(
                f"{name} needs a BFM p23470 R mesh (23470 vertices), got {n_a}. {_ETC_NOTE}"
            )
    else:
        # lmk_indices is only read by the non-rigid aligners; set per call below.
        mm = {"lmk_indices": None, "Npoints": n_a}
    return ecm.ErrorComputer(spec["recipe"], mm=mm)


def pair_distance(
    name: str,
    VA: np.ndarray,
    FA: np.ndarray | None,
    VB: np.ndarray,
    FB: np.ndarray | None,
    *,
    lmks_a: np.ndarray,
    lmks_b: np.ndarray,
    lmk_indices_a: np.ndarray | None = None,
    reduce: str = "mean",
) -> float:
    """Scalar M3DFB error for the ordered pair (A = reconstruction, B = ground truth).

    lmks_a / lmks_b : (51, 3) iBUG-51 landmark coordinates. Required -- both of
        M3DFB's rigid aligners are landmark-driven (ICPRigidAligner does a
        landmark Procrustes pre-alignment before calling open3d ICP), so there is
        no landmark-free path through the framework.
    lmk_indices_a : indices of those landmarks into VA. Only the non-rigid
        aligners need them.
    reduce : 'mean' (M3DFB's own reduction) or 'median'.
    """
    if name not in ESTIMATORS:
        raise KeyError(f"unknown estimator {name!r}; have {sorted(ESTIMATORS)}")
    if lmks_a is None or lmks_b is None:
        raise ValueError(f"{name} needs landmarks on both meshes (see docstring)")

    ec = _computer(name, int(len(VA)))
    ec.lmk_indices = None if lmk_indices_a is None else list(np.asarray(lmk_indices_a))

    # ErrorComputer mutates G in place when a corrector is active, and the
    # aligners are free to write into R -- hand them private copies.
    err = ec.compute(
        np.array(VA, dtype=np.float64, copy=True),
        np.array(VB, dtype=np.float64, copy=True),
        np.array(lmks_a, dtype=np.float64, copy=True),
        np.array(lmks_b, dtype=np.float64, copy=True),
    )
    err = np.asarray(err, dtype=np.float64).ravel()
    if not np.isfinite(err).any():
        return float("nan")
    return float(np.nanmedian(err) if reduce == "median" else np.nanmean(err))


def demo() -> None:
    """Self-check: E1/E9 are ~0 on a mesh against itself and larger across subjects."""
    mesh_root = REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500"

    def load(sid, topo):
        with np.load(mesh_root / f"{sid}_GTready_{topo}.npz") as d:
            return d["V"].astype(np.float64), d["F"]

    li = bfm_landmark_indices()
    VA, FA = load("id0000", "original")
    VB, FB = load("id0004", "original")
    VC, FC = load("id0000", "down8k")
    iC = transfer_landmarks(VC, VA)

    for name in ("E1", "E9", "E2", "E10"):
        self_err = pair_distance(name, VA, FA, VA, FA, lmks_a=VA[li], lmks_b=VA[li],
                                 lmk_indices_a=li)
        cross = pair_distance(name, VA, FA, VB, FB, lmks_a=VA[li], lmks_b=VB[li],
                              lmk_indices_a=li)
        scale = float(np.abs(VA - VA.mean(0)).max())
        assert self_err / scale < 1e-6, (name, self_err, scale)
        assert cross > 100 * self_err, (name, self_err, cross)
        print(f"  {name:4s} self={self_err:.4g} cross={cross:.4g}  ok")

    # cross-topology, landmarks transferred: A has 8129 vertices -> ETC must refuse
    xt = pair_distance("E1", VC, FC, VB, FB, lmks_a=VC[iC], lmks_b=VB[li], lmk_indices_a=iC)
    assert np.isfinite(xt) and xt > 0, xt
    print(f"  E1 cross-topology down8k->original = {xt:.4g}  ok")
    try:
        pair_distance("E2", VC, FC, VB, FB, lmks_a=VC[iC], lmks_b=VB[li], lmk_indices_a=iC)
    except ValueError:
        print("  E2 correctly refuses a non-BFM R mesh  ok")
    else:
        raise AssertionError("E2 should refuse a non-BFM R mesh")

    assert estimator_names() == ["E1", "E2", "E9", "E10"], estimator_names()
    assert len([n for n in ESTIMATORS if not n.endswith("_p2tri")]) == 16

    # the reconstructed E-numbering must reproduce the 4 recipes upstream ships
    for fname, name in (("E01", "E1"), ("E08", "E8"), ("E12", "E12"), ("E16", "E16")):
        with open(M3DFB_ROOT / "info" / "error_computers" / f"{fname}.json") as f:
            shipped = json.load(f)
        ours = ESTIMATORS[name]["recipe"]
        for stage in ("rigid_aligner", "nonrigid_aligner", "corr_establisher",
                      "distance_computer", "corrector"):
            assert shipped.get(stage) == ours[stage], (fname, stage, shipped.get(stage), ours[stage])
        print(f"  {name} matches shipped {fname}.json  ok")
    print(f"  {len(ESTIMATORS)} estimators registered, usable = {estimator_names()}  ok")


if __name__ == "__main__":
    demo()
