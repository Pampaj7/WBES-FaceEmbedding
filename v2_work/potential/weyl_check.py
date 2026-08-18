#!/usr/bin/env python
"""Does area normalisation make the spectrum comparable across topologies?

Weyl: lambda_k ~ 4*pi*k/A, so cropping (which removes area) shifts every eigenvalue even
though the identity is unchanged. The pipeline currently normalises VERTICES by maxabs and
leaves the OPERATORS at raw scale, so the effective spectral scale is lambda*maxabs^2.
This compares, per identity, the spread across its own topologies under three conventions.
Lower spread = the same identity looks the same whatever the topology.
"""
import numpy as np, glob, re, collections

files = sorted(glob.glob("datasets/REMESH/npz_data_topo_500_withops/*.npz"))
by_subj = collections.defaultdict(dict)
for f in files:
    m = re.match(r".*/(id\d+)_GTready_(\w+)\.npz", f)
    if m: by_subj[m.group(1)][m.group(2)] = f
subs = sorted(by_subj)[:40]

rows = {"raw lambda": [], "lambda*maxabs^2 (attuale)": [], "lambda*A (Weyl)": []}
K = 30
for s in subs:
    vals = collections.defaultdict(list)
    for topo, f in sorted(by_subj[s].items()):
        d = np.load(f); V = d["verts"].astype(np.float64); F = d["faces"]
        ev = np.asarray(d["evals"], dtype=np.float64)[1:K+1]
        Vc = V - V.mean(0); ma = float(np.abs(Vc).max())
        tri = V[F]; A = float(0.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1).sum())
        vals["raw lambda"].append(ev)
        vals["lambda*maxabs^2 (attuale)"].append(ev*ma*ma)
        vals["lambda*A (Weyl)"].append(ev*A)
    for k, v in vals.items():
        M = np.stack(v)                      # (n_topo, K)
        # dispersione relativa fra topologie della STESSA identità, mediata sui primi K modi
        rows[k].append(float(np.mean(M.std(0)/(np.abs(M.mean(0))+1e-30))))

print(f"{'convenzione':28s} {'dispersione media':>18s}  (piu' basso = meglio)")
for k, v in rows.items():
    print(f"{k:28s} {np.mean(v):18.4f}")
