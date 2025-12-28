import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def load_mesh(path):
    return np.loadtxt(path)

def create_trace(mesh, name):
    return go.Scatter3d(
        x=mesh[:, 0], y=mesh[:, 1], z=mesh[:, 2],
        mode='markers',
        marker=dict(size=1.5, color='gray', opacity=0.85),
        name=name
    )

def load_meshes_for_subject(subject_id, method, base_dir, gt_dir):
    gt = load_mesh(gt_dir / f"{subject_id}.id.txt")
    f1 = load_mesh(base_dir / f"{subject_id}_000.txt")
    f15_list = []
    for i in range(15):
        p = base_dir / f"{subject_id}_{i:03}.txt"
        if p.exists():
            f15_list.append(load_mesh(p))
    if not f15_list:
        raise RuntimeError(f"No meshes to average for {subject_id}")
    f15 = np.mean(np.stack(f15_list), axis=0)
    return [gt, f1, f15]

# === CONFIG ===
method = "Smirk_cropped_neutral"
subject_ids = ["id0030", "id0078"]
base_dir = Path("/home/pampalonil/data") / method
gt_dir = Path("/home/pampalonil/data/GT/GT_BFM/")

# === LOAD ALL MESHES ===
rows = []
for sid in subject_ids:
    rows.append(load_meshes_for_subject(sid, method, base_dir, gt_dir))

# === PLOT ===
titles = ["GT (aligned)", "1 frame", "Average of 15 frames"]
fig = make_subplots(
    rows=2, cols=3,
    specs=[[{'type':'scene'}]*3, [{'type':'scene'}]*3],
    subplot_titles=[f"{t} — {sid}" for sid in subject_ids for t in titles],
    horizontal_spacing=0.01,
    vertical_spacing=0.01
)

for row_idx, meshes in enumerate(rows):
    for col_idx, mesh in enumerate(meshes):
        fig.add_trace(create_trace(mesh, f"{subject_ids[row_idx]}_{col_idx}"), row=row_idx+1, col=col_idx+1)
        fig.update_scenes(
            dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'
            ),
            row=row_idx+1, col=col_idx+1
        )

fig.update_layout(
    title_text=f"Birkan Plot — 2 Subjects ({method})",
    margin=dict(l=0, r=0, t=30, b=0),
    showlegend=False
)

# === SAVE TO HTML ===
out_html = f"faces_plot_{method}_2subjects.html"
fig.write_html(out_html)
print(f"✅ Saved: {out_html}")
