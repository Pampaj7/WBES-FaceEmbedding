import os, numpy as np, matplotlib.pyplot as plt

IN = "dist_matrices_fields/distance_matrices_fields.npz"
OUT_DIR = "dist_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

data = np.load(IN)
Dg = data["D_orig"]
Df = data["D_lat_field"]
Dm = data["D_lat_mean"]

def ranking(D):
    N = D.shape[0]
    order = np.argsort(D, axis=1)   # crescente
    order = np.array([row[row != i] for i,row in enumerate(order)])  # rimuovi self
    return order  # shape (N, N-1)

Rg = ranking(Dg)
Rf = ranking(Df)
Rm = ranking(Dm)

def precision_at_k(R_pred, R_gt, k):
    N = R_pred.shape[0]
    p = 0.0
    for i in range(N):
        pred = set(R_pred[i, :k])
        gt   = set(R_gt[i, :k])
        p += len(pred & gt) / float(k)
    return p / N

def average_precision(pred_rank, rel_set):
    score = 0.0; hits = 0
    for r, j in enumerate(pred_rank, start=1):
        if j in rel_set:
            hits += 1
            score += hits / r
    return score / max(len(rel_set), 1)

def mean_average_precision(R_pred, R_gt):
    N = R_pred.shape[0]
    ap = 0.0
    for i in range(N):
        rel = set(R_gt[i])  # tutto l'ordine GT come "rilevante" (ranking supervision)
        ap += average_precision(R_pred[i], rel)
    return ap / N

Ks = [1,3,5]
metrics = []
for name, R in [("lat_field", Rf), ("lat_mean", Rm)]:
    row = {"name": name}
    for k in Ks: row[f"P@{k}"] = precision_at_k(R, Rg, k)
    row["mAP"] = mean_average_precision(R, Rg)
    metrics.append(row)

# CSV
with open(os.path.join(OUT_DIR, "retrieval_metrics.csv"), "w") as f:
    f.write("name," + ",".join([f"P@{k}" for k in Ks]) + ",mAP\n")
    for r in metrics:
        f.write(r["name"] + "," + ",".join(f"{r[f'P@{k}']:.4f}" for k in Ks) + f",{r['mAP']:.4f}\n")

# Bar plot
labels = [f"P@{k}" for k in Ks] + ["mAP"]
x = np.arange(len(labels))
w = 0.35
f_vals = [metrics[0][lab] for lab in labels]
m_vals = [metrics[1][lab] for lab in labels]

plt.figure(figsize=(7,4))
plt.bar(x - w/2, f_vals, width=w, label="lat_field")
plt.bar(x + w/2, m_vals, width=w, label="lat_mean")
plt.xticks(x, labels); plt.ylim(0,1)
plt.ylabel("Score"); plt.title("Retrieval: precision@k & mAP")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "retrieval_bars.png"), dpi=200); plt.close()

print("✅ Retrieval metrics in", OUT_DIR)
print(metrics)
