#!/usr/bin/env python3
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = "recon_npz"
SUBJECT  = "id0000_GTready"   # cambia qui se vuoi

def load_points(path):
    d = np.load(path)
    V = d["V"]                 # Nx3 vertices only
    return V

def make_scatter_trace(V, name, color):
    return go.Scatter3d(
        x=V[:,0],
        y=V[:,1],
        z=V[:,2],
        mode='markers',
        marker=dict(size=2, color=color),
        name=name
    )

def main():
    variants = ["original_recon", "remesh_recon", "crop_recon", "noisy_recon"]
    colors   = ["blue", "red", "green", "orange"]

    # 2×2 subplot grid, all 3D scatter
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=variants
    )

    for idx, (var, col) in enumerate(zip(variants, colors)):
        row = idx // 2 + 1
        col_index = idx % 2 + 1

        path = os.path.join(BASE_DIR, f"{SUBJECT}_{var}.npz")
        print(f"Loading {path}")
        V = load_points(path)

        trace = make_scatter_trace(V, var, col)
        fig.add_trace(trace, row=row, col=col_index)

        scene_id = f"scene{idx+1}" if idx > 0 else "scene"
        fig.update_layout(**{
            scene_id: dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data"
            )
        })

    fig.update_layout(
        title_text=f"Point clouds for {SUBJECT}",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )

    out_path = f"preview_points_{SUBJECT}_recon.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"\nSaved HTML to: {out_path}")

if __name__ == "__main__":
    main()
