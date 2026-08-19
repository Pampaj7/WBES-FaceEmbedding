#!/usr/bin/env python
"""Il 2x2 e le repliche di seed, in una tabella sola.

Il guadagno da spiegare e' +0.022 su crop e +0.0085 su all, misurato su un seed. Il criterio
fissato prima di vederlo era che una differenza piccola va replicata, quindi questa tabella
riporta media e deviazione standard sui seed disponibili e dice esplicitamente quanti sono: una
media su un solo seed va letta come un singolo campione, non come una media, e viene marcata.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--latex", action="store_true",
                help="emette anche la tabella LaTeX, cosi' i numeri entrano nel supplementary "
                     "senza essere ricopiati a mano: una cifra trascritta male non solleva "
                     "errori e sopravvive fino alla revisione")
ARGS = ap.parse_args()

R = Path(__file__).resolve().parent / "results"
GROUPS = ["crop", "noisy", "resample", "all"]

# braccio -> (frame atteso, elenco dei tag delle repliche)
ARMS = [
    ("pot_plain",    "current", ["pot_plain", "pot_plain_s2", "pot_plain_s3"]),
    ("pot_area",     "current", ["pot_area", "pot_area_s2", "pot_area_s3"]),
    ("pot_rms",      "rms",     ["pot_rms"]),
    ("pot_rms_area", "rms",     ["pot_rms_area"]),
    # operatori sulla stessa normalizzazione che la rete vede sull'xyz: testa se
    # l'antagonismo fra frame rms e area unitaria sia incoerenza di unita'
    ("pot_rmsops",   "rms",     ["pot_rmsops"]),
]

print(f"{'braccio':14s} {'n':>2s} " + " ".join(f"{g:>16s}" for g in GROUPS))
base = {}
for name, frame, tags in ARMS:
    vals = {g: [] for g in GROUPS}
    n = 0
    for t in tags:
        f = R / f"{t}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        # Un disallineamento fra frame di training e di valutazione non solleva errori e produce
        # numeri plausibili. Il frame e' scritto nel JSON apposta: qui si controlla.
        got = d.get("frame", "current")
        if got != frame:
            print(f"  ATTENZIONE: {t} valutato con frame '{got}', atteso '{frame}' -- ESCLUSO")
            continue
        n += 1
        for g in GROUPS:
            vals[g].append(d["groups"][g]["spearman"])
    if n == 0:
        print(f"{name:14s} {'-':>2s} " + " ".join(f"{'in attesa':>16s}" for _ in GROUPS))
        continue
    cells = []
    for g in GROUPS:
        a = np.array(vals[g])
        cells.append(f"{a.mean():.4f}" + (f" +-{a.std(ddof=1):.4f}" if n > 1 else "  (1 seed)"))
    print(f"{name:14s} {n:2d} " + " ".join(f"{c:>16s}" for c in cells))
    if name == "pot_plain":
        base = {g: np.array(vals[g]) for g in GROUPS}

if base:
    print(f"\ndelta contro pot_plain:")
    for name, frame, tags in ARMS[1:]:
        vals = {}
        for t in tags:
            f = R / f"{t}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            if d.get("frame", "current") != frame:
                continue
            for g in GROUPS:
                vals.setdefault(g, []).append(d["groups"][g]["spearman"])
        if not vals:
            continue
        row = " ".join(f"{np.mean(vals[g]) - base[g].mean():+16.4f}" for g in GROUPS)
        print(f"{name:14s} {len(vals['all']):2d} {row}")

nseed = len([t for t in ("pot_plain", "pot_plain_s2", "pot_plain_s3") if (R / f"{t}.json").exists()])
if nseed < 3:
    print(f"\nPROVVISORIO: {nseed} seed su 3 per il braccio di controllo. Con guadagni dell'ordine")
    print("di 0.02 una sola replica non distingue l'effetto dal rumore. Niente di questo va in un")
    print("paper prima delle tre repliche.")


if ARGS.latex:
    LBL = {"pot_plain":    (r"\texttt{current}", r"$L$"),
           "pot_area":     (r"\texttt{current}", r"$L$, area unitaria"),
           "pot_rms":      (r"\texttt{rms}",     r"$L$"),
           "pot_rms_area": (r"\texttt{rms}",     r"$L$, area unitaria"),
           "pot_rmsops":   (r"\texttt{rms}",     r"$L$, raggio rms unitario")}
    rows, seeds_min = [], None
    for name, frame, tags in ARMS:
        vals, n = {g: [] for g in GROUPS}, 0
        for t in tags:
            f = R / f"{t}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            if d.get("frame", "current") != frame:
                continue
            n += 1
            for g in GROUPS:
                vals[g].append(d["groups"][g]["spearman"])
        if n == 0:
            continue
        seeds_min = n if seeds_min is None else min(seeds_min, n)
        cells = []
        for g in GROUPS:
            a = np.array(vals[g])
            cells.append("$%.4f$" % a.mean() if n == 1
                         else "$%.4f \\pm %.4f$" % (a.mean(), a.std(ddof=1)))
        fr, op = LBL.get(name, (name, ""))
        rows.append(fr + " & " + op + " & " + " & ".join(cells) + r" \\")

    note = ("Ogni cella e' un singolo seed: la differenza fra bracci non e' ancora "
            "distinguibile dal rumore di inizializzazione."
            if seeds_min == 1 else "Media e deviazione standard su %d seed." % seeds_min)
    print("\n% ---- generato da aggregate_2x2.py --latex, non modificare a mano ----")
    print(r"\begin{tabular}{ll" + "c" * len(GROUPS) + "}")
    print(r"\toprule")
    print("frame & operatori & " + " & ".join(r"\texttt{%s}" % g for g in GROUPS) + r" \\")
    print(r"\midrule")
    for r_ in rows:
        print(r_)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print("% " + note)


# ---------------------------------------------------------------------------
# Analisi APPAIATA. E' la statistica giusta per questo disegno e non un extra:
# bracci che condividono il seed condividono i pesi iniziali e l'ordine dei dati,
# quindi l'effetto del seed e' comune e la differenza appaiata lo elimina. Confrontare
# un guadagno contro la dispersione NON appaiata del controllo -- come ho fatto nella
# ritrattazione del 19 agosto -- usa un denominatore troppo grande e fa sembrare rumore
# un effetto reale. Il segno concorde fra seed vale piu' della media quando n e' 2.
SEEDS = {"1234": "", "1235": "_s2", "1236": "_s3"}
BASE = "pot_plain"

print("\n=== differenze appaiate contro pot_plain, per seed ===")
for arm, frame, _ in ARMS:
    if arm == BASE:
        continue
    rows, seeds = [], []
    for seed, suf in SEEDS.items():
        fa, fb = R / f"{BASE}{suf}.json", R / f"{arm}{suf}.json"
        if not (fa.exists() and fb.exists()):
            continue
        da, db = json.loads(fa.read_text()), json.loads(fb.read_text())
        if db.get("frame", "current") != frame:
            continue
        rows.append([db["groups"][g]["spearman"] - da["groups"][g]["spearman"] for g in GROUPS])
        seeds.append(seed)
    if not rows:
        continue
    a = np.array(rows)
    print(f"\n{arm}  ({len(seeds)} seed: {', '.join(seeds)})")
    print(f"  {'':10s}" + " ".join(f"{g:>11s}" for g in GROUPS))
    for s, r_ in zip(seeds, a):
        print(f"  seed {s:5s}" + " ".join(f"{v:+11.4f}" for v in r_))
    if len(a) > 1:
        print(f"  {'media':10s}" + " ".join(f"{v:+11.4f}" for v in a.mean(0)))
        print(f"  {'spread':10s}" + " ".join(f"{v:11.4f}" for v in (a.max(0) - a.min(0))))
        conc = ["si" if (r_ > 0).all() or (r_ < 0).all() else "NO" for r_ in a.T]
        print(f"  {'segno':10s}" + " ".join(f"{c:>11s}" for c in conc))
        print("  (segno 'NO' = la direzione cambia fra seed, cioe' non e' un effetto)")


# Con tre differenze appaiate si puo' dire qualcosa di quantitativo, purche' si dica anche
# quanto e' debole: n=3 significa 2 gradi di liberta', e un t-test qui e' un indizio, non una
# prova. Il test dei segni e' riportato accanto perche' non assume nulla sulla distribuzione:
# con tre campioni concordi da' p = 1/8, che e' il massimo che tre punti possono offrire.
try:
    from scipy import stats as _st
except ImportError:
    _st = None

print("\n=== quanto reggono, per gruppo (n=3 => 2 gradi di liberta', indizio non prova) ===")
for arm, frame, _ in ARMS:
    if arm == BASE:
        continue
    rows = []
    for seed, suf in SEEDS.items():
        fa, fb = R / f"{BASE}{suf}.json", R / f"{arm}{suf}.json"
        if not (fa.exists() and fb.exists()):
            continue
        da, db = json.loads(fa.read_text()), json.loads(fb.read_text())
        if db.get("frame", "current") != frame:
            continue
        rows.append([db["groups"][g]["spearman"] - da["groups"][g]["spearman"] for g in GROUPS])
    a = np.array(rows)
    if len(a) < 3:
        print(f"\n{arm}: {len(a)} seed, servono 3 repliche appaiate")
        continue
    print(f"\n{arm} ({len(a)} seed appaiati)")
    for i, g in enumerate(GROUPS):
        d = a[:, i]
        conc = (d > 0).all() or (d < 0).all()
        line = f"  {g:10s} media {d.mean():+.4f}  ds {d.std(ddof=1):.4f}  segno {'concorde' if conc else 'DISCORDE'}"
        if _st is not None and d.std(ddof=1) > 0:
            t, p = _st.ttest_1samp(d, 0.0)
            line += f"  t={t:+.2f} p={p:.3f}"
        if conc:
            line += "  (test dei segni: p=0.125, il minimo possibile con 3 campioni)"
        print(line)
