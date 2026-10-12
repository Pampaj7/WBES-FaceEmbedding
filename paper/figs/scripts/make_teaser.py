#!/usr/bin/env python3
"""Teaser (Fig. 1): due identita' di GNM Head in discretizzazioni diverse, residuo dopo l'allineamento, schema per mesh.

    WBES_GNM_NPZ=~/data/gnm_head/gnm_head.npz aau/run.sh paper/figs/scripts/make_teaser.py   # nodo di calcolo

Licenza: SOLO GNM Head v3.0 (Google, Apache-2.0; ~/data/gnm_head/PROVENANCE.md), che permette di pubblicare render
delle sue mesh (literature/LICENZE_RILASCIO_2026-10-09.md, riga GNM). Nessuna mesh di FaceScape, HIFI3D, FaceVerse,
FaMoS, FLAME o BFM entra in questa figura. Nessuna mesh viene scritta nel repo: solo il PDF.

Pannelli:
  (a) identita' A nella discretizzazione originale, identita' B decimata (``down8k``, lo stesso generatore dei dati:
      v2_work/genict/make_ict_topologies.py), col reticolo visibile;
  (b) residuo per vertice |a_k - b_k| (mm) in corrispondenza densa nota (stessa topologia di GNM), dopo (i) una rigida
      per identita' verso la forma media (la posa: come il riferimento FR) e (ii) anche la taglia tolta (ogni mesh
      alla centroid size media, come il riferimento SR / la normalizzazione cs). E' un'illustrazione su due campioni:
      nessun numero di questa figura entra nel testo;
  (c) schema: ogni mesh codificata da sola, distanza fra embedding.
La coppia si sceglie in modo deterministico (seme fisso) fra 300 campioni N(0, I): A al quantile 0.85 della centroid
size, B al quantile 0.15, per rendere visibile la differenza di taglia.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[2]
sys.path.insert(0, str(REPO / "aau" / "zs3dmm"))
sys.path.insert(0, str(REPO / "v2_work" / "genict"))
sys.path.insert(0, str(THIS))

import gnm_model  # noqa: E402
import mesh_ops as mo  # noqa: E402
from make_ict_topologies import make_down8k, triangle_targets  # noqa: E402

SEED = 2027
N_SAMPLES = 300
OUT = REPO / "paper" / "figs" / "fig_teaser.pdf"
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def sample_faces(model: dict, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    k = model["shapedirs"].shape[2]
    W = np.random.default_rng(seed).normal(size=(n, k))
    Vs, F = [], None
    for w in W:
        V, F = gnm_model.face_patch(gnm_model.shape_mesh(w, model), model)
        Vs.append(1000.0 * V)  # metri -> mm
    return np.stack(Vs), F


def centroid_size(V: np.ndarray) -> float:
    return float(np.sqrt(((V - V.mean(0)) ** 2).sum(1).mean()))


def rigid_to(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """X allineata rigidamente (senza scala) a Y, corrispondenza per indice (Kabsch)."""
    mx, my = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((X - mx).T @ (Y - my))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return (X - mx) @ (U @ D @ Vt) + my


def project(V: np.ndarray, yaw_deg: float = 28.0, pitch_deg: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Proiezione ortografica: frame GNM +Y in alto, +Z il naso (verso l'osservatore)."""
    a, b = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    P = (V - V.mean(0)) @ (Rx @ Ry).T
    return P[:, :2], P[:, 2]


def draw_mesh(ax, V, F, face_values=None, cmap=None, vmin=None, vmax=None, edges=False, base=(0.80, 0.80, 0.84)):
    P, depth = project(V)
    tri = P[F]
    n = np.cross(V[F][:, 1] - V[F][:, 0], V[F][:, 2] - V[F][:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    a, b = np.radians(28.0), np.radians(6.0)
    light = np.array([0.35, 0.45, 1.0]); light /= np.linalg.norm(light)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    nv = n @ (Rx @ Ry).T
    shade = np.clip(0.25 + 0.75 * np.abs(nv @ light), 0, 1)
    if face_values is None:
        rgb = np.outer(shade, np.asarray(base))
    else:
        cm = plt.get_cmap(cmap)
        rgb = cm((face_values - vmin) / (vmax - vmin))[:, :3] * (0.45 + 0.55 * shade)[:, None]
    order = np.argsort(depth[F].mean(1))  # dal fondo verso l'osservatore
    pc = PolyCollection(tri[order], facecolors=rgb[order],
                        edgecolors=(0.15, 0.15, 0.2, 0.55) if edges else "face",
                        linewidths=0.08 if edges else 0.05)
    pc.set_rasterized(True)  # 17k triangoli per pannello: raster a 300 dpi dentro il PDF, testo vettoriale
    ax.add_collection(pc)
    ax.set_xlim(P[:, 0].min() - 2, P[:, 0].max() + 2)
    ax.set_ylim(P[:, 1].min() - 2, P[:, 1].max() + 2)
    ax.set_aspect("equal"); ax.axis("off")


def main() -> None:
    model = gnm_model.load_gnm(os.environ.get("WBES_GNM_NPZ", str(Path.home() / "data" / "gnm_head" / "gnm_head.npz")))
    Vs, F = sample_faces(model, N_SAMPLES, SEED)
    cs = np.array([centroid_size(V) for V in Vs])
    order = np.argsort(cs)
    ia, ib = order[int(0.85 * N_SAMPLES)], order[int(0.15 * N_SAMPLES)]
    A, B = Vs[ia], Vs[ib]
    mean = Vs.mean(0)
    print(f"A = campione {ia} (CS {cs[ia]:.1f} mm), B = campione {ib} (CS {cs[ib]:.1f} mm); V={len(A)} F={len(F)}")

    # (b) una rigida per identita' verso la media (posa), poi anche la taglia (ogni mesh alla CS media)
    Ar, Br = rigid_to(A, mean), rigid_to(B, mean)
    cs_ref = centroid_size(mean)
    As = rigid_to((A - A.mean(0)) * cs_ref / cs[ia], mean)
    Bs = rigid_to((B - B.mean(0)) * cs_ref / cs[ib], mean)
    r_pose = np.linalg.norm(Ar - Br, axis=1)
    r_size = np.linalg.norm(As - Bs, axis=1)
    vmax = float(np.quantile(r_pose, 0.98))

    # (a) B decimata con il generatore down8k dei dati
    Fa = F
    Vb, Fb = make_down8k(*mo.as_arrays(B, F), triangle_targets(len(F))[0])
    print(f"down8k di B: V={len(Vb)} F={len(Fb)}")

    fig = plt.figure(figsize=(6.875, 2.0))
    # posizioni esplicite [x, y, larghezza, altezza] in frazione della figura
    ax = [fig.add_axes(r) for r in ([0.000, 0.13, 0.145, 0.70], [0.150, 0.13, 0.145, 0.70],
                                     [0.320, 0.13, 0.145, 0.70], [0.470, 0.13, 0.145, 0.70],
                                     [0.690, 0.10, 0.310, 0.80])]
    cax = fig.add_axes([0.625, 0.22, 0.007, 0.52])
    draw_mesh(ax[0], A, Fa, edges=True)
    draw_mesh(ax[1], Vb, Fb, edges=True)
    ax[0].set_title(f"identity A\noriginal ({len(A) // 1000}k vertices)", fontsize=6)
    ax[1].set_title(f"identity B\ndecimated ({len(Vb) // 1000}k vertices)", fontsize=6)
    draw_mesh(ax[2], Ar, Fa, r_pose[Fa].mean(1), "magma", 0.0, vmax)
    draw_mesh(ax[3], As, Fa, r_size[Fa].mean(1), "magma", 0.0, vmax)
    ax[2].set_title("pose removed\n(identity kept)", fontsize=6)
    ax[3].set_title("pose and size removed\n(size difference erased)", fontsize=6)
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("|A - B| (mm)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fig.text(0.148, 0.02, "(a) same pair, different meshes", ha="center", fontsize=6.5)
    fig.text(0.47, 0.02, "(b) what alignment removes is not compared", ha="center", fontsize=6.5)

    # (c) schema per mesh
    s = ax[4]
    s.set_xlim(-0.2, 10.5); s.set_ylim(0, 6); s.axis("off")
    for y, name in ((4.4, "mesh of A"), (1.6, "mesh of B")):
        s.add_patch(FancyBboxPatch((0.0, y - 0.55), 2.3, 1.1, boxstyle="round,pad=0.05", fc="#e8e8f0", ec="#666"))
        s.text(1.15, y, name, ha="center", va="center", fontsize=6.5)
        s.add_patch(FancyArrowPatch((2.4, y), (3.3, y), arrowstyle="-|>", mutation_scale=7, lw=0.8))
        s.add_patch(FancyBboxPatch((3.4, y - 0.55), 2.2, 1.1, boxstyle="round,pad=0.05", fc="#efe6f7", ec="#5b2c83"))
        s.text(4.5, y, "encoder $f_\\theta$", ha="center", va="center", fontsize=6.5)
        s.add_patch(FancyArrowPatch((5.7, y), (6.8, y), arrowstyle="-|>", mutation_scale=7, lw=0.8))
        s.text(7.5, y, "$(s, u)$", ha="center", va="center", fontsize=7)
    s.add_patch(FancyArrowPatch((7.5, 3.8), (7.5, 2.2), arrowstyle="<|-|>", mutation_scale=7, lw=0.8, color="#5b2c83"))
    s.text(7.8, 3.0, "$d_F$: size\nand shape", ha="left", va="center", fontsize=6.2,
           color="#5b2c83")
    fig.text(0.845, 0.02, "(c) one embedding per mesh, no pairwise alignment", ha="center", fontsize=6.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01, dpi=300)
    (OUT.parent / "fig_teaser.sources.txt").write_text(
        "stato: DATI (illustrazione: GNM Head v3.0, Apache-2.0, campioni N(0, I) con seme "
        f"{SEED}, coppia {ia} / {ib}); nessun numero entra nel testo\n")
    if os.environ.get("FIG_PREVIEW"):
        fig.savefig(Path(os.environ["FIG_PREVIEW"]) / "fig_teaser.png", dpi=150, bbox_inches="tight")
    print(f"scritto {OUT}")


if __name__ == "__main__":
    main()
