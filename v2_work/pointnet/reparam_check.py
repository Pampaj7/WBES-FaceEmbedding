#!/usr/bin/env python
"""Il crop riparametrizza lo spettro: quanto bene, e con quale numero?

Rodola et al. (Partial Functional Correspondence): la mappa funzionale fra una forma e una sua
parte ha diagonale INCLINATA, con pendenza pari al rapporto di aree. Bracha et al. (ACCV 2024):
l'errore sotto parzialita' e' proporzionale all'area mancante. Weyl: lambda_k ~ 4 pi k / A.

Tutte e tre dicono la stessa cosa in forme diverse -- il taglio non distrugge lo spettro, lo
RIPARAMETRIZZA. Se e' vero nel nostro caso discreto, allora:

  (1) esiste un solo scalare alpha_t per cui lambda^t_k ~ alpha_t * lambda^orig_k per ogni k;
  (2) quello scalare e' il rapporto di aree A_orig / A_t;
  (3) il residuo dopo aver tolto alpha_t e' rumore, non struttura.

Le tre cose sono separabili e vanno separate: (1) dice se la riparametrizzazione e' pura, (2) se
la spiega l'area, (3) se resta qualcosa da spiegare. Un fallimento di (2) con successo di (1)
sarebbe il caso interessante -- vorrebbe dire che alpha e' stimabile dallo spettro stesso, senza
conoscere il taglio.

AVVERTENZA. Questo e' meccanicismo, non metrica. Lo sweep di alpha del pozzo si muoveva nella
direzione giusta su un proxy e il modello allenato e' andato dall'altra parte. Serve a decidere
se la normalizzazione d'area puo' essere una correzione COMPLETA o solo parziale, non a
predire lo Spearman.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "datasets/REMESH/npz_data_topo_500_withops"
TOPOS = ["crop", "noisy", "down8k", "up60k", "remesh"]
N_ID = 60
KLO, KHI = 1, 128          # lambda_0 = 0 va escluso: non porta informazione e rompe i rapporti


def area(V, F):
    tri = V[F]
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def load(p):
    z = np.load(p)
    return z["evals"][KLO:KHI].astype(np.float64), area(z["verts"].astype(np.float64), z["faces"])


ids = sorted({p.name.split("_")[0] for p in D.glob("*_original.npz")})[:N_ID]
res = {t: {"alpha": [], "arearatio": [], "r2": [], "tilt": []} for t in TOPOS}
n = 0
for i in ids:
    po = D / f"{i}_GTready_original.npz"
    if not po.exists():
        continue
    lo, Ao = load(po)
    ok = False
    for t in TOPOS:
        pt = D / f"{i}_GTready_{t}.npz"
        if not pt.exists():
            continue
        lt, At = load(pt)
        # (1) miglior scalare nel senso dei minimi quadrati su lambda^t = alpha * lambda^orig
        alpha = float((lt @ lo) / (lo @ lo))
        resid = lt - alpha * lo
        r2 = 1.0 - float(resid @ resid) / float(((lt - lt.mean()) ** 2).sum())
        # (3) il residuo e' piatto in k o ha una pendenza? una pendenza significa che un solo
        #     scalare NON basta e la riparametrizzazione dipende dall'indice.
        k = np.arange(len(lt), dtype=np.float64)
        tilt = float(np.polyfit(k, resid / max(lt.mean(), 1e-12), 1)[0] * len(lt))
        res[t]["alpha"].append(alpha)
        res[t]["arearatio"].append(Ao / At)
        res[t]["r2"].append(r2)
        res[t]["tilt"].append(tilt)
        ok = True
    n += ok

print(f"{n} identita', modi {KLO}-{KHI}\n")
print(f"{'topologia':10s} {'alpha (fit)':>18s} {'A_orig/A_t':>18s} {'alpha/rapporto':>16s} {'R^2':>8s} {'tilt resid':>11s}")
for t in TOPOS:
    a = np.array(res[t]["alpha"]); r = np.array(res[t]["arearatio"])
    q = a / r
    print(f"{t:10s} {a.mean():9.4f} +-{a.std():.4f} {r.mean():9.4f} +-{r.std():.4f} "
          f"{q.mean():9.4f} +-{q.std():.4f} {np.mean(res[t]['r2']):8.4f} {np.mean(res[t]['tilt']):11.4f}")

print("\ncorrelazione fra alpha stimato e rapporto di aree, per topologia:")
for t in TOPOS:
    a = np.array(res[t]["alpha"]); r = np.array(res[t]["arearatio"])
    if a.std() < 1e-9 or r.std() < 1e-9:
        print(f"  {t:10s} varianza nulla, correlazione non definita")
        continue
    print(f"  {t:10s} Pearson {np.corrcoef(a, r)[0,1]:+.4f}   (n={len(a)})")

# La domanda che conta per il metodo: se il taglio e' ignoto a test time, alpha e' recuperabile
# dallo spettro stesso? Un alpha con dispersione fra identita' MOLTO piu' piccola del suo valore
# medio si puo' stimare; uno che varia quanto varia il segnale, no.
print("\ndispersione relativa di alpha fra identita' (std/mean): piu' bassa = piu' stimabile")
for t in TOPOS:
    a = np.array(res[t]["alpha"])
    print(f"  {t:10s} {a.std()/abs(a.mean()):.4f}")
