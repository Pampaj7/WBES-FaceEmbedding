"""Mesh-support statistics of the `original` topology across BFM / FLAME / FaceScape.

Answers: is FaceScape out-of-distribution w.r.t. the BFM training support in a way
FLAME is not, and along which axes?

    .conda_env/bin/python v2_work/transfer/support_stats.py

Writes v2_work/transfer/support_stats.csv (long format: domain, metric, mean, std,
p05, p95) and prints the same table.

Everything shape-related is measured AFTER the repo's per-mesh normalization
(centre on the vertex mean, divide by max|coord| -- `mesh_npz_utils.normalize_vertices`),
because that is what the encoder sees as input.  Two extra families are reported raw:

* `raw_*`: the pre-normalization scale, kept only to document how far apart the
  three domains' units are.
* `eval*`: the stored Laplace-Beltrami spectrum.  NOTE it is computed by
  `precompute_operators_npz.py` on the RAW vertices (no normalization anywhere in
  that script), so the numbers are in 1/length^2 of each domain's own unit and are
  NOT comparable across domains as stored -- which is itself a finding, since the
  network is fed normalized vertices together with these raw-scale operators.
  `eval1_x_area` = lambda_1 * raw surface area is the dimensionless version and is
  the comparable one.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "faceBench/latentVSpipeline"))
from mesh_npz_utils import normalize_vertices  # noqa: E402

DOMAINS = {
    # name: (dir, glob, n_meshes) -- the `original` topology of each domain, plus
    # (second block) the pair mates each domain's cross-topology eval actually uses.
    "BFM": (REPO_ROOT / "datasets/REMESH/npz_data_topo_500_withops", "*_GTready_original.npz", 100),
    "FLAME": (REPO_ROOT / "v2_work/genflame/flame_topo_600_withops", "flame*_GTready_original.npz", 100),
    "FaceScape": (REPO_ROOT / "datasets/FaceVerse/cross_topology_10k_with_ops", "*_01_original.npz", 110),
    "BFM_remesh": (REPO_ROOT / "datasets/REMESH/npz_data_topo_500_withops", "*_GTready_remesh.npz", 100),
    "FS_remesh10k_v1": (REPO_ROOT / "datasets/FaceVerse/remesh10k_with_ops", "*_01_remesh_10k.npz", 110),
    "FS_remesh_bfmrule": (REPO_ROOT / "v2_work/transfer/facescape_control_withops", "*_01_remesh.npz", 110),
    "FS_recrop": (REPO_ROOT / "v2_work/transfer/facescape_recrop_withops", "*_01_original.npz", 110),
    "FS_recrop_remesh": (REPO_ROOT / "v2_work/transfer/facescape_recrop_withops", "*_01_remesh.npz", 110),
    "FLAME_remesh": (REPO_ROOT / "v2_work/genflame/flame_topo_600_withops", "flame*_GTready_remesh.npz", 110),
    "FLAME_poisson": (REPO_ROOT / "v2_work/transfer/flame_poisson_withops", "flame*_poisson.npz", 110),
}
RADII = (0.2, 0.4, 0.6, 0.8)

# (original, remesh) pairs whose mutual support agreement decides whether a
# cross-topology pair is a mesh-processing perturbation or a change of region.
PAIRS = {
    "BFM original/remesh": ("BFM", "BFM_remesh"),
    "FaceScape original/remesh_10k (v1 protocol)": ("FaceScape", "FS_remesh10k_v1"),
    "FaceScape original/remesh BFM-rule": ("FaceScape", "FS_remesh_bfmrule"),
    "FaceScape recrop original/remesh": ("FS_recrop", "FS_recrop_remesh"),
    "FLAME original/remesh": ("FLAME", "FLAME_remesh"),
    "FLAME original/poisson (v1-style mate)": ("FLAME", "FLAME_poisson"),
}


def mesh_stats(V: np.ndarray, F: np.ndarray, evals: np.ndarray) -> dict[str, float]:
    """Support descriptors of one mesh; V raw, normalized internally."""
    F = np.asarray(F, dtype=np.int64)
    raw_ext = V.max(axis=0) - V.min(axis=0)
    tri_raw = V[F]
    raw_area = 0.5 * np.linalg.norm(
        np.cross(tri_raw[:, 1] - tri_raw[:, 0], tri_raw[:, 2] - tri_raw[:, 0]), axis=1
    ).sum()

    Vn = normalize_vertices(V)
    ext = Vn.max(axis=0) - Vn.min(axis=0)
    diag = float(np.linalg.norm(ext))
    tri = Vn[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()

    edges = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    counts = Counter(map(tuple, edges))
    uniq = np.array(list(counts.keys()))
    edge_len = np.linalg.norm(Vn[uniq[:, 0]] - Vn[uniq[:, 1]], axis=1)
    boundary = uniq[np.array([c == 1 for c in counts.values()])]
    n_boundary = len(boundary)

    r = np.linalg.norm(Vn, axis=1)  # centroid is the origin after normalization
    # Area-weighted radius: barycentric vertex areas, so the radial profile measures
    # the SUPPORT REGION and not the vertex sampling density (the plain vertex
    # fractions below conflate the two).
    tri_area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    vmass = np.zeros(len(Vn))
    np.add.at(vmass, F.ravel(), np.repeat(tri_area / 3.0, 3))
    vmass = vmass / vmass.sum()
    out = {
        "n_verts": float(len(Vn)),
        "n_faces": float(len(F)),
        "ext_x": float(ext[0]), "ext_y": float(ext[1]), "ext_z": float(ext[2]),
        "ext_x_over_y": float(ext[0] / ext[1]),
        "ext_z_over_y": float(ext[2] / ext[1]),
        "bbox_diag": diag,
        "area": float(area),
        "area_over_diag2": float(area / diag**2),
        "mean_edge_over_diag": float(edge_len.mean() / diag),
        "n_boundary_edges": float(n_boundary),
        "boundary_frac_of_verts": float(n_boundary / len(Vn)),
        "mean_radius": float(r.mean()),
        "max_radius": float(r.max()),
        "area_mean_radius": float((vmass * r).sum()),
        "boundary_mean_radius": float(r[np.unique(boundary)].mean()) if n_boundary else float("nan"),
        "boundary_min_radius": float(r[np.unique(boundary)].min()) if n_boundary else float("nan"),
        "raw_bbox_diag": float(np.linalg.norm(raw_ext)),
        "raw_area": float(raw_area),
        "eval1": float(evals[1]),
        "eval5": float(evals[5]),
        "eval20": float(evals[20]),
        "eval1_x_raw_area": float(evals[1] * raw_area),
        "eval5_x_raw_area": float(evals[5] * raw_area),
        "eval20_x_raw_area": float(evals[20] * raw_area),
    }
    out.update({f"frac_within_r{rr}": float((r <= rr).mean()) for rr in RADII})
    out.update({f"frac_area_within_r{rr}": float(vmass[r <= rr].sum()) for rr in RADII})
    return out


def pair_agreement(n: int = 25) -> list[dict]:
    """How far apart are the two supports the model is asked to match?

    Both mates are normalized independently (that is what the encoder sees), then
    one-sided nearest-neighbour distances are taken between their vertex sets.  A
    decimation/subdivision of the same surface leaves this near zero; a change of
    region or a Poisson re-reconstruction does not.
    """
    from scipy.spatial import cKDTree

    rows = []
    print(f"\n{'pair':<44} {'chamfer':>9} {'hausdorff':>10} {'area ratio':>11}")
    for label, (a, b) in PAIRS.items():
        da, pa, na = DOMAINS[a]
        db, pb, nb = DOMAINS[b]
        fa = {f.name.split("_")[0]: f for f in sorted(da.glob(pa))}
        fb = {f.name.split("_")[0]: f for f in sorted(db.glob(pb))}
        stats = []
        for key in sorted(set(fa) & set(fb))[:n]:
            Va, Fa_ = _load(fa[key])
            Vb, Fb_ = _load(fb[key])
            Va, Vb = normalize_vertices(Va), normalize_vertices(Vb)
            d_ab = cKDTree(Vb).query(Va)[0]
            d_ba = cKDTree(Va).query(Vb)[0]
            stats.append((0.5 * (d_ab.mean() + d_ba.mean()), max(d_ab.max(), d_ba.max()),
                          _area(Vb, Fb_) / _area(Va, Fa_)))
        m = np.mean(stats, axis=0)
        print(f"{label:<44} {m[0]:>9.4f} {m[1]:>10.4f} {m[2]:>11.3f}")
        rows.append({"domain": label, "n_meshes": len(stats), "metric": "pair_chamfer",
                     "mean": f"{m[0]:.6g}", "std": "", "p05": "", "p95": ""})
        rows.append({"domain": label, "n_meshes": len(stats), "metric": "pair_hausdorff",
                     "mean": f"{m[1]:.6g}", "std": "", "p05": "", "p95": ""})
        rows.append({"domain": label, "n_meshes": len(stats), "metric": "pair_area_ratio",
                     "mean": f"{m[2]:.6g}", "std": "", "p05": "", "p95": ""})
    return rows


def _load(f: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(f, allow_pickle=False) as z:
        return (np.asarray(z["verts"] if "verts" in z else z["V"], dtype=np.float64),
                np.asarray(z["faces"] if "faces" in z else z["F"], dtype=np.int64))


def _area(V: np.ndarray, F: np.ndarray) -> float:
    t = V[F]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def main() -> None:
    rows: list[dict] = []
    per_domain: dict[str, dict[str, list[float]]] = {}
    for domain, (d, pattern, n) in DOMAINS.items():
        files = sorted(d.glob(pattern))[:n]
        if not files:
            raise SystemExit(f"no files for {domain} in {d}")
        acc: dict[str, list[float]] = {}
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                V = np.asarray(z["verts"] if "verts" in z else z["V"], dtype=np.float64)
                F = np.asarray(z["faces"] if "faces" in z else z["F"])
                st = mesh_stats(V, F, np.asarray(z["evals"], dtype=np.float64))
            for k, v in st.items():
                acc.setdefault(k, []).append(v)
        per_domain[domain] = acc
        print(f"{domain}: {len(files)} meshes from {d.name}", flush=True)

    metrics = list(next(iter(per_domain.values())).keys())
    for metric in metrics:
        for domain, acc in per_domain.items():
            a = np.asarray(acc[metric])
            rows.append({
                "domain": domain, "n_meshes": len(a), "metric": metric,
                "mean": f"{a.mean():.6g}", "std": f"{a.std():.6g}",
                "p05": f"{np.percentile(a, 5):.6g}", "p95": f"{np.percentile(a, 95):.6g}",
            })

    rows += pair_agreement()

    out = Path(__file__).resolve().parent / "support_stats.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}\n")

    width = max(len(m) for m in metrics)
    print(f"{'metric':<{width}}  " + "  ".join(f"{d:>22}" for d in per_domain))
    for metric in metrics:
        cells = []
        for acc in per_domain.values():
            a = np.asarray(acc[metric])
            cells.append(f"{a.mean():>11.4g} +-{a.std():<9.3g}")
        print(f"{metric:<{width}}  " + "  ".join(cells))


def demo() -> None:
    """Self-check on a synthetic open disc: known boundary and known extents."""
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    V = np.vstack([[0, 0, 0], np.stack([np.cos(theta), np.sin(theta), np.zeros(12)], 1)])
    F = np.array([[0, i + 1, (i + 1) % 12 + 1] for i in range(12)])
    st = mesh_stats(V, F, np.zeros(64))
    assert st["n_boundary_edges"] == 12, st["n_boundary_edges"]
    assert abs(st["ext_x_over_y"] - 1.0) < 1e-9
    assert st["ext_z"] == 0.0
    assert abs(st["frac_within_r0.8"] - 1 / 13) < 1e-9  # only the centre vertex
    assert abs(st["area"] - np.pi) < 0.2  # 12-gon approximation of the unit disc
    print("demo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
