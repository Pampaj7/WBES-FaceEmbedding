#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from collections import defaultdict

SRC = Path("/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500")
DST = Path("/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/data_CANONICAL")

DST.mkdir(parents=True, exist_ok=True)

REF_VARIANT = "original"

def load_npz(path):
    d = np.load(path)
    return d["V"], d["F"]

def save_npz(path, V, F):
    np.savez(path, V=V, F=F)

def compute_transform(V_ref):
    # center at zero
    center = V_ref.mean(axis=0)
    Vc = V_ref - center

    # isotropic scale: max bbox extent -> 2
    extent = (Vc.max(axis=0) - Vc.min(axis=0)).max()
    scale = 2.0 / extent

    return center, scale

def apply_transform(V, center, scale):
    return (V - center) * scale

# -------------------------------------------------
# Collect files by subject
# -------------------------------------------------

files = sorted(SRC.glob("*.npz"))
subjects = defaultdict(dict)

for f in files:
    # es: id0000_GTready_original.npz
    name = f.stem
    parts = name.split("_")
    sid = parts[0]
    variant = parts[-1]
    subjects[sid][variant] = f

print(f"Found {len(subjects)} subjects")

# -------------------------------------------------
# Canonicalize per subject
# -------------------------------------------------

for sid, variants in subjects.items():
    if REF_VARIANT not in variants:
        print(f"[WARN] {sid}: missing '{REF_VARIANT}', skipping")
        continue

    # reference
    V_ref, _ = load_npz(variants[REF_VARIANT])
    center, scale = compute_transform(V_ref)

    for variant, path in variants.items():
        V, F = load_npz(path)
        Vc = apply_transform(V, center, scale)

        out_path = DST / path.name
        save_npz(out_path, Vc, F)

print("Canonical dataset written to:", DST)
