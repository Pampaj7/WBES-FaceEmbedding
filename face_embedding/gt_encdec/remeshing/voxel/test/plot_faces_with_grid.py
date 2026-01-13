#!/usr/bin/env python3
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

DATA_DIR = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/data_CANONICAL"
)

OUT_HTML = Path("faces_with_grid.html")

GRID_SIZE = 8
DOWNSAMPLE = 3000
SEED = 0

BOUNDS_MIN = np.array([-1.2, -1.2, -1.2])
BOUNDS_MAX = np.array([ 1.2,  1.2,  1.2])

np.random.seed(SEED)

# -------------------------------------------------
# Utils
# -------------------------------------------------

def load_vertices(path):
    return np.load(path)["V"]

def downsample(V, n):
    if V.shape[0] <= n:
        return V
    idx = np.random.choice(V.shape[0], n, replace=False)
    return V[idx]

def grid_lines(bounds_min, bounds_max, grid_size):
    xs = np.linspace(bounds_min[0], bounds_max[0], grid_size + 1)
    ys = np.linspace(bounds_min[1], bounds_max[1], grid_size + 1)
    zs = np.linspace(bounds_min[2], bounds_max[2], grid_size + 1)

    lines = []

    # vertical (z)
    for x in xs:
        for y in ys:
            lines.append(([x, x], [y, y], [zs[0], zs[-1]]))

    # y-lines
    for x in xs:
        for z in zs:
            lines.append(([x, x], [ys[0], ys[-1]], [z, z]))

    # x-lines
    for y in ys:
        for z in zs:
            lines.append(([xs[0], xs[-1]], [y, y], [z, z]))

    return lines

# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    subjects = sorted(
        p.stem.replace("_original", "")
        for p in DATA_DIR.glob("*_original.npz")
    )

    sid_A = subjects[0]
    sid_B = subjects[1]

    print(f"Plotting subjects: {sid_A} vs {sid_B}")
    print(f"Saving interactive HTML to: {OUT_HTML.resolve()}")

    V_A = downsample(load_vertices(DATA_DIR / f"{sid_A}_original.npz"), DOWNSAMPLE)
    V_B = downsample(load_vertices(DATA_DIR / f"{sid_B}_original.npz"), DOWNSAMPLE)

    fig = go.Figure()

    # -------------------------
    # Subject A
    # -------------------------
    fig.add_trace(go.Scatter3d(
        x=V_A[:, 0], y=V_A[:, 1], z=V_A[:, 2],
        mode="markers",
        marker=dict(size=2, color="blue", opacity=0.6),
        name=f"{sid_A}"
    ))

    # -------------------------
    # Subject B
    # -------------------------
    fig.add_trace(go.Scatter3d(
        x=V_B[:, 0], y=V_B[:, 1], z=V_B[:, 2],
        mode="markers",
        marker=dict(size=2, color="orange", opacity=0.6),
        name=f"{sid_B}"
    ))

    # -------------------------
    # Grid
    # -------------------------
    for x, y, z in grid_lines(BOUNDS_MIN, BOUNDS_MAX, GRID_SIZE):
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color="gray", width=1),
            opacity=0.2,
            showlegend=False
        ))

    # -------------------------
    # Layout
    # -------------------------
    fig.update_layout(
        title="Canonical Faces with 3D Spatial Grid",
        scene=dict(
            xaxis=dict(range=[BOUNDS_MIN[0], BOUNDS_MAX[0]], title="X"),
            yaxis=dict(range=[BOUNDS_MIN[1], BOUNDS_MAX[1]], title="Y"),
            zaxis=dict(range=[BOUNDS_MIN[2], BOUNDS_MAX[2]], title="Z"),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.02, y=0.98)
    )

    fig.write_html(OUT_HTML)
    print("✅ Interactive plot saved.")

# -------------------------------------------------
# ENTRY
# -------------------------------------------------

if __name__ == "__main__":
    main()
