import numpy as np
import os
import glob
import plotly.graph_objects as go

# === CONFIG ===
INPUT_DIR = "GT_ready/npz_data"
OUTPUT_DIR = "GT_ready/npz_data_cropped"
INDEX_FILE = "/equilibrium/lpampaloni/WBES-FaceEmbedding/WBES/utils/ix_23470_relative_to_53215.txt"
GEN_HTML = False  # true se vuoi anteprime .html per controllo
# ===============

def load_crop_indices(index_path):
    return np.loadtxt(index_path, dtype=int)

def crop_mesh(V_full, F_full, idx_crop):
    V_crop = V_full[idx_crop]
    old_to_new = {old: new for new, old in enumerate(idx_crop)}
    F_crop = [ [old_to_new[v] for v in tri] for tri in F_full if all(v in old_to_new for v in tri) ]
    return V_crop, np.array(F_crop, dtype=int)

def render_html(V, F, out_path):
    x, y, z = V.T
    i, j, k = F.T
    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightblue', opacity=1.0)
    fig = go.Figure(mesh)
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0,r=0,t=0,b=0))
    fig.write_html(out_path)

def process_all(input_dir, output_dir, index_path):
    os.makedirs(output_dir, exist_ok=True)
    idx_crop = load_crop_indices(index_path)
    old_to_new = {old: new for new, old in enumerate(idx_crop)}

    files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    print(f"📁 Found {len(files)} files in: {input_dir}")

    for fpath in files:
        fname = os.path.basename(fpath)
        out_path = os.path.join(output_dir, fname)
        html_path = out_path.replace(".npz", ".html")

        try:
            data = np.load(fpath)
            if "verts" not in data or "faces" not in data:
                print(f"⚠️  Skipped {fname} (missing verts/faces)")
                continue

            V_full = data["verts"]
            F_full = data["faces"]
            V_crop, F_crop = crop_mesh(V_full, F_full, idx_crop)

            np.savez(out_path, V=V_crop, F=F_crop)
            print(f"✅ Cropped: {fname} → {out_path}")

            if GEN_HTML:
                render_html(V_crop, F_crop, html_path)
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

if __name__ == "__main__":
    process_all(INPUT_DIR, OUTPUT_DIR, INDEX_FILE)
