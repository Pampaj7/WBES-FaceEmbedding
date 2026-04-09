#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


BG = "#f7f4ef"
PANEL = "#fffdf8"
GRID = "#d8cfc4"
TEXT = "#1f2933"
MUTED = "#6b7280"
ACCENT = "#0f766e"
ACCENT_2 = "#c2410c"
ACCENT_3 = "#7c3aed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a compact report for pairwise operator similarity.")
    parser.add_argument("--summary-json", required=True, type=str)
    parser.add_argument("--pairwise-csv", required=True, type=str)
    parser.add_argument("--ops-dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--title", type=str, default="")
    return parser.parse_args()


def short_topology_label(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return stem


def load_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pairwise(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_matrices(
    query_files: List[str],
    pair_rows: List[Dict[str, str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(query_files)
    idx = {name: i for i, name in enumerate(query_files)}
    l2 = np.zeros((n, n), dtype=np.float64)
    cosine = np.ones((n, n), dtype=np.float64)
    pearson = np.ones((n, n), dtype=np.float64)

    for row in pair_rows:
        i = idx[row["a"]]
        j = idx[row["b"]]
        l2[i, j] = l2[j, i] = float(row["l2"])
        cosine[i, j] = cosine[j, i] = float(row["cosine"])
        pearson[i, j] = pearson[j, i] = float(row["pearson"])
    return l2, cosine, pearson


def load_mesh_stats(ops_dir: Path, query_files: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name in query_files:
        path = ops_dir / name
        with np.load(path, allow_pickle=False) as data:
            verts = np.asarray(data["verts"])
            faces = np.asarray(data["faces"])
            mass = np.asarray(data["mass"], dtype=np.float64).reshape(-1)
            out[name] = {
                "n_verts": float(verts.shape[0]),
                "n_faces": float(faces.shape[0]),
                "mass_sum": float(mass.sum()),
            }
    return out


def ratio_against_original(
    stats: Dict[str, Dict[str, float]],
    query_files: List[str],
    key: str,
) -> np.ndarray:
    original_name = next(name for name in query_files if short_topology_label(name) == "original")
    base = max(stats[original_name][key], 1e-12)
    return np.array([stats[name][key] / base for name in query_files], dtype=np.float64)


def nearest_positive_rank_map(summary: Dict) -> Dict[str, int]:
    return {row["query"]: int(row["nearest_positive_rank"]) for row in summary["retrieval"]}


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(1.2)


def add_panel_title(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=TEXT, pad=10)
    if subtitle:
        ax.text(
            0.0,
            1.005,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color=MUTED,
        )


def draw_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    labels: List[str],
    title: str,
    subtitle: str,
    cmap: LinearSegmentedColormap,
    fmt: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    style_axis(ax)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    add_panel_title(ax, title, subtitle)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10, color=TEXT)
    ax.set_yticklabels(labels, fontsize=10, color=TEXT)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=BG, linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    norm = im.norm
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            color = "white" if norm(value) > 0.55 else TEXT
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=9, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.tick_params(labelsize=9, colors=MUTED)


def add_header(fig: plt.Figure, summary: Dict, title: str) -> None:
    if not title:
        title = f"Operator Similarity Report · {summary['query_subject']}"

    fig.text(
        0.055,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.055,
        0.938,
        "Two complementary views of pairwise similarity for the six topology variants of the same identity.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )


def main() -> None:
    args = parse_args()
    summary = load_summary(Path(args.summary_json))
    pair_rows = load_pairwise(Path(args.pairwise_csv))
    ops_dir = Path(args.ops_dir)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
        }
    )

    query_files = summary["query_files"]
    labels = [short_topology_label(name) for name in query_files]
    l2, cosine, pearson = build_matrices(query_files, pair_rows)
    l2_cmap = LinearSegmentedColormap.from_list("l2_map", ["#f8fafc", "#94a3b8", "#0f766e", "#134e4a"])
    cos_cmap = LinearSegmentedColormap.from_list("cos_map", ["#fff7ed", "#fdba74", "#ea580c", "#7c2d12"])

    fig = plt.figure(figsize=(14, 7.4), facecolor=BG)
    gs = fig.add_gridspec(
        1,
        2,
        left=0.055,
        right=0.97,
        bottom=0.08,
        top=0.86,
        wspace=0.28,
    )

    ax_l2 = fig.add_subplot(gs[0, 0])
    ax_cos = fig.add_subplot(gs[0, 1])

    draw_heatmap(
        ax_l2,
        l2,
        labels,
        title="Pairwise Spectral Distance",
        subtitle="L2 distance on the normalized spectral vector. Diagonal = 0.",
        cmap=l2_cmap,
        fmt=".3f",
        vmin=0.0,
        vmax=float(np.max(l2)),
    )
    draw_heatmap(
        ax_cos,
        cosine,
        labels,
        title="Pairwise Cosine Similarity",
        subtitle="Values near 1 indicate nearly overlapping spectra.",
        cmap=cos_cmap,
        fmt=".4f",
        vmin=float(np.min(cosine)),
        vmax=1.0,
    )
    add_header(fig, summary, args.title)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
