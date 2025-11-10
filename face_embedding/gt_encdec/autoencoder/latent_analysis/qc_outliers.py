import os, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

GT_DIR   = "../../../datasets/GT_ready/npz_data"
LAT_DIR  = "latents_full"
IN_MATS  = "dist_matrices_fields/distance_matrices_fields.npz"
OUT_DIR  = "dist_analysis/qc"
os.makedirs(OUT_DIR, exist_ok=True)

def load_gt(p):
    d = np.load(p)
    V = d["verts"].astype(np.float64)
    F = d["faces"].astype(np.int64)
    evals = d["evals"].astype(np.float64)
    evecs = d["evecs"].astype(np.float64)
    # area triangoli
    a = V[F[:,1]] - V[F[:,0]]
    b = V[F[:,2]] - V[F[:,0]]
    area = 0.5 * np.linalg.norm(np.cross(a,b), axis=1).sum()
    # varianza vertici
    vvar = V.var()
    # Laplacian energy (approssimata via evals & proiezione)
    # proietta coordinate su basi: C (k x 3)
    k = min(32, evecs.shape[1])
    Phi = evecs[:,:k]                       # (n,k)
    C = Phi.T @ V                           # (k,3)
    lap_energy = float(np.sum(evals[:k] * (C**2).sum(axis=1)))
    return area, vvar, lap_energy

def load_lat(p):
    d = np.load(p)
    zg = d["Z_global"].astype(np.float64)
    zgnorm = float(np.linalg.norm(zg))
    return zgnorm

# allineamento per nome
lat_files = sorted([f for f in os.listdir(LAT_DIR) if f.endswith(".npz")])
gt_files  = [f for f in lat_files if os.path.exists(os.path.join(GT_DIR, f))]

feat = []
names = []
for fn in gt_files:
    area, vvar, lapE = load_gt(os.path.join(GT_DIR, fn))
    znorm = load_lat(os.path.join(LAT_DIR, fn))
    feat.append([area, vvar, lapE, znorm])
    names.append(fn)
X = np.array(feat)

# Isolation Forest + LOF
iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=0).fit(X)
iso_score = -iso.score_samples(X)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
lof_score = -lof.fit_predict(X)  # labels; need negative_outlier_factor_
lof_raw = -lof.negative_outlier_factor_

rank = np.argsort(0.5*iso_score + 0.5*lof_raw)[::-1]
topN = rank[:20]

# CSV
with open(os.path.join(OUT_DIR, "qc_outliers.csv"), "w") as f:
    f.write("name,area,verts_var,lap_energy,z_norm,iso_score,lof_score\n")
    for i in rank:
        f.write(f"{names[i]},{X[i,0]:.6e},{X[i,1]:.6e},{X[i,2]:.6e},{X[i,3]:.6e},{iso_score[i]:.6f},{lof_raw[i]:.6f}\n")

# Proiezione nel piano D_orig vs D_lat_field con outlier evidenziati
mats = np.load(IN_MATS)
Dg = mats["D_orig"]; Df = mats["D_lat_field"]
def norm01(v): return (v-v.min())/(v.max()-v.min()+1e-12)
x = norm01(Dg[np.triu_indices_from(Dg,1)])
y = norm01(Df[np.triu_indices_from(Df,1)])
plt.figure(figsize=(6,6)); hb = plt.hexbin(x,y,gridsize=80,cmap="viridis")
plt.colorbar(label="Density")
plt.plot([0,1],[0,1],"k:",lw=1)

# evidenzia coppie che coinvolgono outlier
idx_out = set(topN.tolist())
N = len(names)
pairs = []
for i in topN:
    for j in range(N):
        if j==i: continue
        a,b = min(i,j), max(i,j)
        pairs.append(a*N + b)  # indice nel vettore triu “compatto” (non banale)

# fallback: disegna solo etichette testuali degli outlier
for i in topN[:10]:
    plt.scatter(0.02, 0.95-0.06*(list(topN).index(i)%10), c="r", s=30)
    plt.text(0.05, 0.95-0.06*(list(topN).index(i)%10), names[i].replace("_GTready",""),
             color="w", fontsize=8)

plt.xlim(0,1); plt.ylim(0,1); plt.gca().set_aspect('equal', 'box')
plt.xlabel("D_orig (normalized)"); plt.ylabel("D_lat_field (normalized)")
plt.title("Outlier QC (top-20 evidenziati)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"qc_outliers_scatter.png"), dpi=220); plt.close()

print("✅ QC completato in", OUT_DIR)
print("Top outlier:", [names[i] for i in topN[:10]])
