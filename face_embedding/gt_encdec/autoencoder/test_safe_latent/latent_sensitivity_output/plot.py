import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_obj(path):
    verts = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                verts.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                _, i, j, k = line.split()
                faces.append([int(i)-1, int(j)-1, int(k)-1])
    return np.array(verts), np.array(faces)


def rotate_vertices(verts):
    Ry = np.array([[0,0,1],[0,1,0],[-1,0,0]])
    Rx = np.array([[1,0,0],[0,0,1],[0,-1,0]])
    Rz = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    R = Rz @ Rx @ Ry
    return verts @ R.T


def plot_mesh(ax, verts, faces, title, text=None):
    center = verts.mean(axis=0)
    verts_zoom = (verts - center) * 3.8 + center

    ax.plot_trisurf(
        verts_zoom[:,0], verts_zoom[:,1], verts_zoom[:,2],
        triangles=faces, color="lightblue", edgecolor="none"
    )

    max_range = (verts_zoom.max(axis=0) - verts_zoom.min(axis=0)).max()
    mid = verts_zoom.mean(axis=0)
    ax.set_xlim(mid[0]-max_range/2, mid[0]+max_range/2)
    ax.set_ylim(mid[1]-max_range/2, mid[1]+max_range/2)
    ax.set_zlim(mid[2]-max_range/2, mid[2]+max_range/2)
    ax.dist = 5
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)

    if text:
        ax.text2D(0.5, -0.12, text, transform=ax.transAxes,
                  ha="center", va="top", fontsize=9)


def load_metrics(metric_file):
    metrics = {}
    if not os.path.exists(metric_file):
        return metrics
    with open(metric_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                nm, cham, cosid = parts
                metrics[nm] = (float(cham), float(cosid))
    return metrics


def make_grid(title, files, metrics, out_name):
    if len(files) == 0:
        print(f"No files for {title}")
        return

    cols = 3
    rows = (len(files) + cols - 1) // cols
    fig = plt.figure(figsize=(5 * cols, 5 * rows))

    for idx, fname in enumerate(files):
        verts, faces = load_obj(fname)
        verts = rotate_vertices(verts)

        txt = None
        key = os.path.basename(fname)
        if key in metrics:
            cham, cosid = metrics[key]
            txt = f"Chamfer={cham:.4f}\nCos={cosid:.3f}"

        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        plot_mesh(ax, verts, faces, key, text=txt)

    plt.tight_layout()
    plt.savefig(out_name, dpi=200)
    print(f"Saved {out_name}")


def main():
    BASE = "."
    metrics = load_metrics(os.path.join(BASE, "metrics.txt"))

    all_objs = [f for f in os.listdir(BASE) if f.endswith(".obj")]

    # split by prefix
    base_files = [f for f in all_objs if f == "base.obj"]
    local_files = [f for f in all_objs if f.startswith("local_noise")]
    global_files = [f for f in all_objs if f.startswith("global_shift")]
    nose_files = [f for f in all_objs if f.startswith("nose_")]

    # prepend path
    base_files = [os.path.join(BASE, f) for f in base_files]
    local_files = [os.path.join(BASE, f) for f in local_files]
    global_files = [os.path.join(BASE, f) for f in global_files]
    nose_files = [os.path.join(BASE, f) for f in nose_files]

    make_grid("Local Noise", local_files, metrics, "grid_local_noise.png")
    make_grid("Global Shift", global_files, metrics, "grid_global_shift.png")
    make_grid("Nose Perturbation", nose_files, metrics, "grid_nose_perturb.png")


if __name__ == "__main__":
    main()
