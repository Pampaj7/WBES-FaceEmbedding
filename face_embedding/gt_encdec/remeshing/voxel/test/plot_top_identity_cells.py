#!/usr/bin/env python3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

CELL_CSV = Path("cell_identity_scores.csv")   # output del tuo script
OUT_HTML = Path("figure_identity_cells.html")

TOP_K = 20
GRID_SIZE = 8

BOUNDS_MIN = np.array([-1.2, -1.2, -1.2])
BOUNDS_MAX = np.array([ 1.2,  1.2,  1.2])

# un volto qualunque per overlay (meglio: volto medio, ma va bene anche uno)
FACE_NPZ = Path(
    "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/"
    "data_CANONICAL/id0000_GTready_original.npz"
)

# ============================================================
# GRID GEOMETRY
# ============================================================

def cell_bounds(cell_idx):
    gs = GRID_SIZE
    ix = cell_idx // (gs * gs)
    iy = (cell_idx % (gs * gs)) // gs
    iz = cell_idx % gs

    step = (BOUNDS_MAX - BOUNDS_MIN) / gs
    cmin = BOUNDS_MIN + step * np.array([ix, iy, iz])
    cmax = cmin + step
    return cmin, cmax


def cube_mesh(cmin, cmax):
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
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5],
        [3, 0, 4], [3, 4, 7],
    ])

    return V, F


# ============================================================
# MAIN
# ============================================================

def main():
    print("🧠 Plotting top identity cells")

    df = pd.read_csv(CELL_CSV)
    df = df.sort_values("score", ascending=False).head(TOP_K)

    scores = df["score"].values
    s_min, s_max = scores.min(), scores.max()

    # normalizza score → [0,1]
    def norm(s):
        return (s - s_min) / (s_max - s_min + 1e-12)

    # volto di riferimento
    face = np.load(FACE_NPZ)
    V_face = face["V"]

    fig = go.Figure()

    # ----------------------------
    # Volto (trasparente)
    # ----------------------------
    fig.add_trace(go.Scatter3d(
        x=V_face[:, 0],
        y=V_face[:, 1],
        z=V_face[:, 2],
        mode="markers",
        marker=dict(size=1.5, color="lightgray", opacity=0.15),
        name="Mean face (reference)"
    ))

    # ----------------------------
    # Celle identitarie
    # ----------------------------
    for _, row in df.iterrows():
        cell_idx = int(row["cell_idx"])
        score = row["score"]
        t = norm(score)

        cmin, cmax = cell_bounds(cell_idx)
        V, F = cube_mesh(cmin, cmax)

        color = f"rgb({int(255*t)}, {int(255*(1-t))}, 0)"

        fig.add_trace(go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=F[:, 0], j=F[:, 1], k=F[:, 2],
            color=color,
            opacity=0.6,
            name=f"cell {cell_idx} | score={score:.1f}",
            showscale=False
        ))

    # ----------------------------
    # Layout
    # ----------------------------
    fig.update_layout(
        title=f"Top-{TOP_K} identity-discriminative spatial cells",
        scene=dict(
            xaxis=dict(range=[BOUNDS_MIN[0], BOUNDS_MAX[0]], visible=False),
            yaxis=dict(range=[BOUNDS_MIN[1], BOUNDS_MAX[1]], visible=False),
            zaxis=dict(range=[BOUNDS_MIN[2], BOUNDS_MAX[2]], visible=False),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(itemsizing="constant")
    )

    fig.write_html(OUT_HTML)
    print(f"✅ Saved figure to: {OUT_HTML.resolve()}")


if __name__ == "__main__":
    main()
