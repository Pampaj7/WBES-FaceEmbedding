#!/usr/bin/env python3
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import torch

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

OUT_HTML = Path("grid_cell_errors.html")

GRID_SIZE = 8
BOUNDS_MIN = np.array([-1.2, -1.2, -1.2])
BOUNDS_MAX = np.array([ 1.2,  1.2,  1.2])

# -------------------------------------------------
# Utils
# -------------------------------------------------

def cell_bounds(cell_idx, grid_size, bounds_min, bounds_max):
    """Return min/max corner of a cell given linear index."""
    gs = grid_size
    ix = cell_idx // (gs * gs)
    iy = (cell_idx % (gs * gs)) // gs
    iz = cell_idx % gs

    step = (bounds_max - bounds_min) / gs

    cmin = bounds_min + step * np.array([ix, iy, iz])
    cmax = cmin + step
    return cmin, cmax


def cube_mesh(cmin, cmax):
    """Vertices and faces for a cube."""
    x0, y0, z0 = cmin
    x1, y1, z1 = cmax

    V = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ])

    F = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [1, 2, 6], [1, 6, 5],  # right
        [3, 0, 4], [3, 4, 7],  # left
    ])

    return V, F

# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main(grid_A, mask_A, grid_B, mask_B, V_A, V_B):
    """
    grid_* : [K, D]
    mask_* : [K]
    """

    valid = mask_A & mask_B
    idxs = torch.where(valid)[0].cpu().numpy()

    # per-cell distance
    dists = torch.norm(grid_A[valid] - grid_B[valid], dim=1).cpu().numpy()

    # normalize 0-1
    lo = np.percentile(dists, 20)
    hi = np.percentile(dists, 90)
    d_norm = np.clip((dists - lo) / (hi - lo), 0, 1)


    fig = go.Figure()
    # -------------------------------------------------
# Transparent faces (context)
# -------------------------------------------------
    fig.add_trace(go.Scatter3d(
        x=V_A[:, 0], y=V_A[:, 1], z=V_A[:, 2],
        mode="markers",
        marker=dict(size=1.8, color="blue", opacity=0.5),
        name="Subject A",
        showlegend=True
    ))

    fig.add_trace(go.Scatter3d(
        x=V_B[:, 0], y=V_B[:, 1], z=V_B[:, 2],
        mode="markers",
        marker=dict(size=1.8, color="orange", opacity=0.5),
        name="Subject B",
        showlegend=True
    ))

    for cell_idx, d in zip(idxs, d_norm):
        cmin, cmax = cell_bounds(cell_idx, GRID_SIZE, BOUNDS_MIN, BOUNDS_MAX)
        V, F = cube_mesh(cmin, cmax)

        fig.add_trace(go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=F[:, 0], j=F[:, 1], k=F[:, 2],
            color=f"rgb({int(255*d)}, {int(255*(1-d))}, 0)",
            opacity=0.2,
            showscale=False
        ))


    fig.update_layout(
        title="Per-cell Identity Difference (Green = Similar, Red = Different)",
        scene=dict(
            xaxis=dict(range=[BOUNDS_MIN[0], BOUNDS_MAX[0]]),
            yaxis=dict(range=[BOUNDS_MIN[1], BOUNDS_MAX[1]]),
            zaxis=dict(range=[BOUNDS_MIN[2], BOUNDS_MAX[2]]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(OUT_HTML)
    print(f"✅ Saved: {OUT_HTML.resolve()}")

# -------------------------------------------------
# ENTRY
# -------------------------------------------------
# Questo file va chiamato passando grid_A, grid_B
# oppure importato dentro run_grid_inference.py
