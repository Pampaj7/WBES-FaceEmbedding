import sys, numpy as np, glob
sys.path.insert(0, "/dtu/p1/leopam/WBES-FaceEmbedding/v2_work/potential")
from potential_operators import load_mesh, geodesic_from_centre, boundary_vertices
from pathlib import Path
import torch
sys.path.insert(0, "/dtu/p1/leopam/WBES-FaceEmbedding/diffusion-net/src")
from diffusion_net.geometry import compute_operators
print(f"{'mesh':34s} {'alpha_permesh':>14s} {'bordo_piu_vicino':>17s}  violazione")
for f in sorted(glob.glob("/dtu/p1/leopam/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500/id000[0-2]_*.npz"))[:8]:
    V,F = load_mesh(Path(f))
    d,_ = geodesic_from_centre(V,F); d = d/max(float(d.max()),1e-12)
    _,mass,_,_,_,_,_ = compute_operators(torch.tensor(V,dtype=torch.float32), torch.tensor(F,dtype=torch.int32), k_eig=16)
    m = mass.numpy().astype(np.float64)
    o = np.argsort(d); cw = np.cumsum(m[o])/max(m.sum(),1e-12)
    alpha = float(d[o][int(np.searchsorted(cw,0.75))])
    b = boundary_vertices(F); bmin = float(d[b].min()) if len(b) else float('nan')
    print(f"{Path(f).stem:34s} {alpha:14.3f} {bmin:17.3f}  {'SI  bordo dentro' if alpha>bmin else 'no'}")
