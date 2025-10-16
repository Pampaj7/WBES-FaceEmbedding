import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Carica i CSV ===
within = pd.read_csv("3DDFAV3_neutral-within_per_subject_v2.csv")
wbes = pd.read_csv("3DDFAV3_neutral-wbes_per_subject_v2.csv")
geom = pd.read_csv("3DDFAV3_neutral-geom_error_per_subject.csv")

# === Unisci per soggetto e F ===
df = within.merge(wbes, on=["subject", "F"]).merge(geom, on=["subject", "F"])

# === Scala l’errore geometrico ===
df["error"] *= 1000

# === Plotta linee per soggetto ===
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="F", y="error", hue="subject",
             alpha=0.2, legend=False, units="subject")
sns.lineplot(data=df, x="F", y="wbse", hue="subject",
             alpha=0.2, legend=False, units="subject")
sns.lineplot(data=df, x="F", y="within", hue="subject",
             alpha=0.2, legend=False, units="subject")

# Aggiungi medie visibili
sns.lineplot(data=df, x="F", y="error",
             label="Geometric Error (x1000)", color="red", ci="sd")
sns.lineplot(data=df, x="F", y="wbse", label="WBES", color="blue", ci="sd")
sns.lineplot(data=df, x="F", y="within",
             label="Within-subject", color="green", ci="sd")

plt.xlabel("Frame Aggregation Level (F)")
plt.ylabel("Metric Value")
plt.title("Metric Trajectories per Subject 3DDFAV3")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.ylim(0, 4)
plt.savefig("lineplot_per_subject.png", dpi=300)
plt.show()
