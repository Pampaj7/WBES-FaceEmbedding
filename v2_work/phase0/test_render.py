"""Self-check for render_mesh.py.  Run: .conda_env/bin/python v2_work/phase0/test_render.py"""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from render_mesh import render_npz

MESH_DIR = Path(__file__).resolve().parents[2] / "datasets/REMESH/npz_data_topo_500"
OUT_DIR = Path(__file__).resolve().parent / "test_renders"
TOPOS = ("original", "down8k", "crop")


def ncc(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    imgs = {}
    for topo in TOPOS:
        t0 = time.perf_counter()
        img = render_npz(MESH_DIR / f"id0000_GTready_{topo}.npz", size=512)
        dt = time.perf_counter() - t0
        imgs[topo] = img
        Image.fromarray(img).save(OUT_DIR / f"id0000_{topo}.png")

        fg = (img[:, :, 0] > 0).mean()
        mean = img.mean()
        print(f"{topo:9s} {dt:5.2f}s  fg={fg:.1%}  mean={mean:6.1f}")

        assert img.shape == (512, 512, 3), img.shape
        assert img.dtype == np.uint8, img.dtype
        assert (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 0] == img[:, :, 2]).all()
        assert dt < 2.0, f"{topo}: {dt:.2f}s > 2s budget"
        assert fg > 0.20, f"{topo}: only {fg:.1%} foreground"
        assert 20 < mean < 200, f"{topo}: mean intensity {mean:.1f} out of range"
        # face is centred: the foreground centroid sits near the image centre
        ys, xs = np.nonzero(img[:, :, 0])
        assert abs(xs.mean() - 256) < 40 and abs(ys.mean() - 256) < 40, (xs.mean(), ys.mean())

    r = ncc(imgs["original"], imgs["down8k"])
    print(f"NCC(original, down8k) = {r:.3f}")
    assert r > 0.7, r

    # non-uniform shading: the render is not a flat silhouette
    fg_vals = imgs["original"][:, :, 0][imgs["original"][:, :, 0] > 0]
    assert fg_vals.std() > 5, fg_vals.std()

    # --- v2: shared frame kills the per-mesh-bbox size bias, yaw is a real rotation
    from render_mesh import mesh_frame

    with np.load(MESH_DIR / "id0000_GTready_original.npz") as d:
        center, scale = mesh_frame(d["V"])

    def shared(topo, yaw=0.0):
        return render_npz(MESH_DIR / f"id0000_GTready_{topo}.npz", size=256,
                          scale=scale, center=center, yaw=yaw)

    def bbox(img):
        ys, xs = np.nonzero(img[:, :, 0])
        return np.ptp(xs), np.ptp(ys), xs.mean()

    w_o, _, cx_o = bbox(shared("original"))
    w_c, _, _ = bbox(shared("crop"))
    assert abs(w_o - w_c) <= 2, f"shared frame: crop width {w_c} vs original {w_o}"

    w_m, _, cx_m = bbox(shared("original", yaw=-30))
    w_p, _, cx_p = bbox(shared("original", yaw=+30))
    assert abs(w_m - w_p) <= 3, f"yaw asymmetric: {w_m} vs {w_p}"          # mirrored poses
    assert w_m < w_o - 5, f"yaw did not foreshorten: {w_m} vs {w_o}"        # rotation happened
    assert cx_m < cx_o < cx_p, (cx_m, cx_o, cx_p)                          # opposite directions
    # center/scale defaults must leave the v1 render untouched
    assert (render_npz(MESH_DIR / "id0000_GTready_original.npz", size=512) == imgs["original"]).all()

    print(f"v2 OK  crop_w={w_c} original_w={w_o}  yaw-30_w={w_m} yaw+30_w={w_p}")
    print("OK ->", OUT_DIR)


if __name__ == "__main__":
    main()
