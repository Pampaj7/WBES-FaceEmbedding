import os, numpy as np, matplotlib.pyplot as plt

IN = "dist_matrices_fields/distance_matrices_fields.npz"
OUT_DIR = "dist_analysis/subject_views"
os.makedirs(OUT_DIR, exist_ok=True)

data = np.load(IN, allow_pickle=True)
Dg = data["D_orig"]
Df = data["D_lat_field"]
names = [n.decode() if isinstance(n, bytes) else n for n in data["names"]]

def norm01(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-12)

def plot_subject(i):
    x = norm01(Dg[i])
    y = norm01(Df[i])

    idx = np.arange(len(x))
    mask = idx != i
    idx = idx[mask]; x = x[mask]; y = y[mask]

    # top-5 vicini in GT e in latente
    top5_gt   = idx[np.argsort(x)[:5]]
    top5_lat  = idx[np.argsort(y)[:5]]
    special   = set(top5_gt) | set(top5_lat)

    plt.figure(figsize=(6,4.5))
    plt.scatter(x, y, s=10, alpha=0.6)
    for j in special:
        jpos = np.where(idx==j)[0][0]
        plt.scatter([x[jpos]],[y[jpos]], s=40, edgecolor="k",
                    c="tab:red" if j in top5_gt else "tab:green", zorder=3)
        lbl = names[j].replace("_GTready","")
        plt.annotate(lbl, (x[jpos], y[jpos]), fontsize=7,
                     xytext=(3,3), textcoords="offset points")

    lim=(0,1); plt.xlim(*lim); plt.ylim(*lim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("D_orig (normalized)"); plt.ylabel("D_lat_field (normalized)")
    plt.title(f"Subject-level view — {names[i]}")
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"subject_{i:03d}.png"), dpi=220); plt.close()

# scegli 3 soggetti rappresentativi: min/med/max correlazione locale
corrs = []
for i in range(Dg.shape[0]):
    idx = np.arange(Dg.shape[0]) != i
    xi = Dg[i, idx]; yi = Df[i, idx]
    xi = (xi-xi.min())/(xi.max()-xi.min()+1e-12)
    yi = (yi-yi.min())/(yi.max()-yi.min()+1e-12)
    c = np.corrcoef(xi, yi)[0,1]
    corrs.append(c)

order = np.argsort(corrs)
for i in [order[0], order[len(order)//2], order[-1]]:
    plot_subject(int(i))

print("✅ Subject-level views salvate in", OUT_DIR)
