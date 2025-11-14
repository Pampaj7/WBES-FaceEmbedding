# ==========================================
# plot_train_log_final.py
# Visualizzazione pulita e automatica del training log
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

BASE_DIR = "./"
CSV_PATH = os.path.join(BASE_DIR, "train_log.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "train_log_final.png")

# === Load CSV ===
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded train log with {len(df)} rows")

# --- Correzione automatica della colonna epoch ---
if "epoch" not in df.columns:
    df["epoch"] = np.arange(1, len(df) + 1)
else:
    # Se i valori di epoch sono molto piccoli o frazionari (es. 0.1–0.9)
    if df["epoch"].max() < 2:
        df["epoch"] = np.arange(1, len(df) + 1)
        print("⚙️  Epoch column normalized to integer indices")

# === Setup figure ===
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
epochs = df["epoch"]

# --- 1️⃣ Totale Train vs Val ---
axes[0].plot(epochs, df["train_loss"], label="Train Loss", color="tab:blue", linewidth=2)
axes[0].plot(epochs, df["val_loss"], label="Val Loss", color="tab:orange", linewidth=2, linestyle="--")
axes[0].set_ylabel("Total Loss")
axes[0].set_title("Training and Validation Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

# --- 2️⃣ Componenti della loss ---
if {"train_l1", "val_l1", "train_normal", "train_laplacian"}.issubset(df.columns):
    axes[1].plot(epochs, df["train_l1"], label="Train L1", color="tab:blue", linestyle="-")
    axes[1].plot(epochs, df["val_l1"], label="Val L1", color="tab:orange", linestyle="--")
    axes[1].plot(epochs, df["train_normal"], label="Normal", color="tab:green", linestyle="-.")
    axes[1].plot(epochs, df["train_laplacian"], label="Laplacian", color="tab:red", linestyle=":")
    axes[1].set_ylabel("Component Loss")
    axes[1].set_title("Loss Components Breakdown")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
else:
    axes[1].text(0.5, 0.5, "Loss components not found", ha="center", va="center")
    axes[1].set_axis_off()

# --- 3️⃣ Learning rate ---
if "current_lr" in df.columns:
    axes[2].plot(epochs, df["current_lr"], color="gray", linewidth=2)
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(alpha=0.3)
else:
    axes[2].text(0.5, 0.5, "No LR data", ha="center", va="center")
    axes[2].set_axis_off()

axes[2].set_xlabel("Epoch")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)
plt.close()

print(f"💾 Saved clean plot → {OUTPUT_PATH}")
