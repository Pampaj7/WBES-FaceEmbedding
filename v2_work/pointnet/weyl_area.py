#!/usr/bin/env python
"""Area SPETTRALE contro area geometrica.

Il check precedente ha trovato che sotto crop un solo scalare spiega il 99.95% dello
spostamento dello spettro, e che quello scalare e' il rapporto di aree all'1.1%. Sotto noisy
invece l'area geometrica sbaglia del 14%: il rumore aggiunge triangoli che contano nella somma
delle aree ma che il calore quasi non sente.

Se e' cosi', l'area giusta per normalizzare non e' quella geometrica ma quella che lo spettro
implica. Weyl: lambda_k ~ 4 pi k / A, quindi A_weyl = 4 pi k / lambda_k, stimata per regressione
sui modi di mezzo (i primi sono dominati dalla forma globale, gli ultimi dal troncamento).

La domanda non e' se A_weyl riproduce alpha -- lo fa quasi per costruzione, essendo derivata
dagli stessi autovalori. La domanda e' se A_weyl coincide con l'area GEOMETRICA dove ci
aspettiamo che coincida (crop, resample: superficie vera) e diverge dove sospettiamo che l'area
geometrica menta (noisy). Se il quadro e' questo, allora "il rumore gonfia l'area senza gonfiare
la superficie efficace" e' un'affermazione misurata e non una congettura, e A_weyl e' una
normalizzazione migliore di sqrt(A) per il nostro insieme di topologie.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "datasets/REMESH/npz_data_topo_500_withops"
TOPOS = ["original", "crop", "noisy", "down8k", "up60k", "remesh"]
N_ID = 60
KLO, KHI = 20, 110      # banda di mezzo: sotto la forma globale, sopra il troncamento


def geo_area(V, F):
    tri = V[F]
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def weyl_area(evals):
    # lambda_k = 4 pi k / A  =>  regressione senza intercetta di lambda su k, A = 4 pi / pendenza
    k = np.arange(KLO, KHI, dtype=np.float64)
    lam = evals[KLO:KHI].astype(np.float64)
    slope = float((k @ lam) / (k @ k))
    return 4.0 * np.pi / slope if slope > 0 else np.nan


ids = sorted({p.name.split("_")[0] for p in D.glob("*_original.npz")})[:N_ID]
rows = {t: {"g": [], "w": []} for t in TOPOS}
n = 0
for i in ids:
    vals = {}
    for t in TOPOS:
        p = D / f"{i}_GTready_{t}.npz"
        if not p.exists():
            break
        z = np.load(p)
        vals[t] = (geo_area(z["verts"].astype(np.float64), z["faces"]), weyl_area(z["evals"]))
    if len(vals) != len(TOPOS):
        continue
    n += 1
    for t in TOPOS:
        rows[t]["g"].append(vals[t][0] / vals["original"][0])
        rows[t]["w"].append(vals[t][1] / vals["original"][1])

print(f"{n} identita', banda di Weyl k={KLO}-{KHI}\n")
print(f"{'topologia':10s} {'A_geom / orig':>20s} {'A_weyl / orig':>20s} {'weyl/geom':>18s}")
for t in TOPOS:
    g = np.array(rows[t]["g"]); w = np.array(rows[t]["w"]); q = w / g
    print(f"{t:10s} {g.mean():11.4f} +-{g.std():.4f} {w.mean():11.4f} +-{w.std():.4f} "
          f"{q.mean():9.4f} +-{q.std():.4f}")

print("\nLettura: weyl/geom ~ 1 significa che l'area geometrica e' fedele alla superficie che")
print("il calore percepisce. Uno scostamento significa che i triangoli ci sono ma non contano.")
