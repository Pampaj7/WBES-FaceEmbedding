import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
CSV_PATH = "cosine_wbesstyle_disjoint.csv"
METRICS = ["cosine_corr", "l2_corr", "complex_corr"]
COLORS = {
    "3DDFAV2_23470_neutral": "#1f77b4",
    "3DDFAV3_neutral": "#ff7f0e",
    "Deep3DFace_23470_neutral": "#2ca02c",
    "SynergyNet_neutral": "#d62728",
    "INORig_23470_neutral": "#9467bd",
    "3DI_neutral": "#8c564b"
}
SCALE_BY_METHOD = {
    "INORig_23470_neutral": 5,
    "3DI_neutral": 10,
}

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)

# === PLOT ===
for metric in METRICS:
    plt.figure(figsize=(10, 6))
    for method in df['method'].unique():
        df_method = df[df['method'] == method].copy()
        scale = SCALE_BY_METHOD.get(method, 1)
        df_method["F_scaled"] = df_method["F"] * scale
        label = f"{method} (×{scale})" if scale > 1 else method

        plt.plot(df_method["F_scaled"], df_method[metric], marker="o",
                 label=label, color=COLORS.get(method, None))

    plt.title(f"Correlation with GT: {metric}")
    plt.xlabel("Number of frames averaged (adjusted scale)")
    plt.ylabel("Pearson Correlation")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_{metric}_scaled.png")
    plt.show()
