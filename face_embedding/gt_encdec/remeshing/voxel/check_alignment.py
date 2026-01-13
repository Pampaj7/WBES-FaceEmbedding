import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.spatial import cKDTree

ROOT = Path("/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/data_CANONICAL")
REF_VARIANT = "original"
MAX_SAMPLES = 5000

def load_vertices(npz_path):
    d = np.load(npz_path)
    return d["V"]

def centroid(V):
    return V.mean(axis=0)

def bbox_extent(V):
    return V.max(axis=0) - V.min(axis=0)

def icp_rms(src, tgt, max_samples=MAX_SAMPLES):
    if src.shape[0] > max_samples:
        src = src[np.random.choice(src.shape[0], max_samples, replace=False)]
    if tgt.shape[0] > max_samples:
        tgt = tgt[np.random.choice(tgt.shape[0], max_samples, replace=False)]
    tree = cKDTree(tgt)
    dists, _ = tree.query(src)
    return np.sqrt(np.mean(dists ** 2))

files = sorted(ROOT.glob("*.npz"))
subjects = defaultdict(dict)

for f in files:
    parts = f.stem.split("_")
    sid = parts[0]
    variant = parts[-1]
    subjects[sid][variant] = f

report = []

for sid, variants in subjects.items():
    if REF_VARIANT not in variants:
        continue

    V_ref = load_vertices(variants[REF_VARIANT])
    c_ref = centroid(V_ref)
    bb_ref = bbox_extent(V_ref)

    for vname, vpath in variants.items():
        V = load_vertices(vpath)
        ce = np.linalg.norm(centroid(V) - c_ref)
        se = np.linalg.norm(bbox_extent(V) - bb_ref)
        rms = icp_rms(V, V_ref)
        report.append((sid, vname, ce, se, rms))

print("\n=== ALIGNMENT CHECK (CANONICAL) ===\n")
for sid, vname, ce, se, rms in report[:20]:
    print(f"{sid:>6} | {vname:<8} | centroid_err: {ce:.2e} | scale_err: {se:.2e} | ICP RMS: {rms:.2e}")

print("\nSUMMARY")
print("Max centroid error:", max(r[2] for r in report))
print("Max scale error   :", max(r[3] for r in report))
print("Max ICP RMS       :", max(r[4] for r in report))
