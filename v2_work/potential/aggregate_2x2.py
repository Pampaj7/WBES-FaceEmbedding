#!/usr/bin/env python
"""Il 2x2 e le repliche di seed, in una tabella sola.

Il guadagno da spiegare e' +0.022 su crop e +0.0085 su all, misurato su un seed. Il criterio
fissato prima di vederlo era che una differenza piccola va replicata, quindi questa tabella
riporta media e deviazione standard sui seed disponibili e dice esplicitamente quanti sono: una
media su un solo seed va letta come un singolo campione, non come una media, e viene marcata.
"""
import json, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent / "results"
GROUPS = ["crop", "noisy", "resample", "all"]

# braccio -> (frame atteso, elenco dei tag delle repliche)
ARMS = [
    ("pot_plain",    "current", ["pot_plain", "pot_plain_s2", "pot_plain_s3"]),
    ("pot_area",     "current", ["pot_area", "pot_area_s2", "pot_area_s3"]),
    ("pot_rms",      "rms",     ["pot_rms"]),
    ("pot_rms_area", "rms",     ["pot_rms_area"]),
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
