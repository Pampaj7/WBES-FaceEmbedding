import sys, json, numpy as np, glob, torch
sys.path.insert(0,"/dtu/p1/leopam/WBES-FaceEmbedding/v2_work/potential")
sys.path.insert(0,"/dtu/p1/leopam/WBES-FaceEmbedding/diffusion-net/src")
from potential_operators import load_mesh, geodesic_from_centre
from diffusion_net.geometry import compute_operators
from pathlib import Path
cfg = json.load(open("/dtu/p1/leopam/WBES-FaceEmbedding/v2_work/potential/alpha_global.json"))
A, S = 0.55, cfg["scale"]
print(f"alpha_global={A:.4f} scala={S:.0f}\n")
print(f"{'mesh':30s} {'area tenuta (global)':>21s} {'area tenuta (per-mesh)':>23s}")
for f in sorted(glob.glob("/dtu/p1/leopam/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500/id000[01]_*.npz"))[:6]:
    V,F = load_mesh(Path(f))
    d,_ = geodesic_from_centre(V,F)
    _,mass,_,_,_,_,_ = compute_operators(torch.tensor(V,dtype=torch.float32), torch.tensor(F,dtype=torch.int32), k_eig=16)
    m = mass.numpy().astype(np.float64); tot = m.sum()
    # global: d/S contro alpha
    keep_g = m[(d/S) < A].sum()/tot
    # per-mesh: d/dmax contro quantile 0.75 pesato per area
    dn = d/max(d.max(),1e-12); o=np.argsort(dn); cw=np.cumsum(m[o])/tot
    a_pm = float(dn[o][int(np.searchsorted(cw,0.75))])
    keep_p = m[dn < a_pm].sum()/tot
    print(f"{Path(f).stem:30s} {keep_g*100:19.1f}% {keep_p*100:22.1f}%")
