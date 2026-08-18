#!/usr/bin/env python
"""Il calore vede la stessa forma, dopo aver risolto la scala?

Idea di Leonardo: sistemare la scala in modo che, guardata col calore, la versione crop
coincida con l'originale. La versione ingenua non e' testabile come metodo, perche' a test time
l'originale non c'e': ogni mesh deve normalizzarsi da sola. Quindi la domanda giusta e' se una
normalizzazione PER-MESH faccia collassare le topologie della stessa identita' senza far
collassare identita' diverse fra loro.

Descrittore senza corrispondenze: la traccia del kernel del calore, Tr(e^{-t Delta}) = sum_k
e^{-lambda_k t}, campionata su tempi log-spaziati. E' globale, non richiede vertici appaiati, ed
e' esattamente "cosa vede il calore".

Tre convenzioni, tutte calcolabili da una mesh sola:
    raw     lambda            (nessuna correzione)
    geom    lambda * A_geom   (quello che fa bfm_areanorm)
    weyl    lambda * A_weyl   (l'area che lo spettro implica, 4 pi k / lambda_k)

Metriche: `within` = distanza media fra topologie della STESSA identita'; `between` = fra
identita' DIVERSE nella stessa topologia; il rapporto between/within e' quanto segnale di
identita' sopravvive per unita' di rumore di topologia. Riportato anche il solo crop, che e'
l'asse rotto.

AVVERTENZA, scritta qui perche' e' costata cara. Questo e' un proxy senza training. Lo sweep di
alpha del pozzo migliorava del 46% su una misura come questa e il modello allenato e' peggiorato
sullo stesso gruppo. Serve a sapere se il calore vede la stessa forma, NON a predire lo Spearman.
"""
from pathlib import Path
import numpy as np
from itertools import combinations

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "datasets/REMESH/npz_data_topo_500_withops"
TOPOS = ["original", "crop", "noisy", "down8k", "up60k", "remesh"]
N_ID = 40
KLO, KHI = 20, 110
TS = np.logspace(-2, 0.5, 24)
# I tempi vanno tarati sulla scala degli autovalori di CIASCUNA convenzione, altrimenti la
# traccia satura a 0 o a 1 e il descrittore diventa una costante -- il primo tentativo aveva
# esattamente questo difetto sulla riga `raw`, che dava distanze nulle. La taratura e' un solo
# numero per convenzione, calcolato sull'intero insieme: globale, quindi non introduce
# normalizzazione per-mesh dalla porta di servizio.


def geo_area(V, F):
    t = V[F]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def weyl_area(ev):
    k = np.arange(KLO, KHI, dtype=np.float64)
    lam = ev[KLO:KHI].astype(np.float64)
    s = float((k @ lam) / (k @ k))
    return 4.0 * np.pi / s if s > 0 else np.nan


def descriptor(ev, scale, tscale=1.0):
    lam = ev[1:].astype(np.float64) * scale * tscale
    # normalizzato per il numero di modi: la traccia troncata dipende da quanti ne teniamo
    tr = np.exp(-np.outer(TS, lam)).sum(1) / len(lam)
    return np.log(tr + 1e-12)


data = {c: {} for c in ("raw", "geom", "weyl")}
ids = []
for i in sorted({p.name.split("_")[0] for p in D.glob("*_original.npz")})[:N_ID]:
    d = {}
    for t in TOPOS:
        p = D / f"{i}_GTready_{t}.npz"
        if not p.exists():
            break
        z = np.load(p)
        ev = z["evals"]
        d[t] = {"raw": (ev, 1.0),
                "geom": (ev, geo_area(z["verts"].astype(np.float64), z["faces"])),
                "weyl": (ev, weyl_area(ev))}
    if len(d) != len(TOPOS):
        continue
    ids.append(i)
    for c in data:
        data[c][i] = {t: d[t][c] for t in TOPOS}   # (evals, scale), descrittori dopo la taratura

# una sola costante per convenzione: porta la mediana di lambda_med*scale a 1, cosi' la finestra
# dei tempi cade dove la traccia varia davvero
for c in data:
    med = np.median([np.median(data[c][i][t][0][1:]) * data[c][i][t][1]
                     for i in ids for t in TOPOS])
    for i in ids:
        for t in TOPOS:
            ev, sc = data[c][i][t]
            data[c][i][t] = descriptor(ev, sc, 1.0 / med)

print(f"{len(ids)} identita', {len(TS)} tempi\n")
print(f"{'conv.':6s} {'within':>10s} {'within crop':>13s} {'between':>10s} {'between/within':>16s} {'b/w crop':>10s}")
for c in ("raw", "geom", "weyl"):
    W = [np.linalg.norm(data[c][i][a] - data[c][i][b])
         for i in ids for a, b in combinations(TOPOS, 2)]
    Wc = [np.linalg.norm(data[c][i]["original"] - data[c][i]["crop"]) for i in ids]
    B = [np.linalg.norm(data[c][i][t] - data[c][j][t])
         for t in TOPOS for i, j in combinations(ids, 2)]
    w, wc, b = np.mean(W), np.mean(Wc), np.mean(B)
    print(f"{c:6s} {w:10.4f} {wc:13.4f} {b:10.4f} {b/w:16.2f} {b/wc:10.2f}")
