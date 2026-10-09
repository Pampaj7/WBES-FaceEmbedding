#!/usr/bin/env python3
"""Studio umano v2: potenza del test primario (quota delle risposte con X nello strato ``X_vs_Y``) per simulazione.

    v3_work/unified_gt/run.sh aau/human_study_v2/power_v2.py      (run.sbatch, passo ``power``)

Modello generativo, uno strato alla volta: ogni partecipante vede ``--trials`` triplette dello strato (9 = 36 test
/ 4 tipi, come la pagina) estratte dalle ``--items`` dello strato (60); la risposta sta con X con probabilita'
expit(b + u_p + v_t), u_p ~ N(0, sd_p) per partecipante, v_t ~ N(0, sd_t) per tripletta (fissi per studio, come
nella realta': le triplette sono quelle). ``b`` si tara perche' la quota MARGINALE sia ``q``; il differenziale di
accordo fra le due GT nello strato e' 2q - 1.

Test (``analyze_v2.py``): d_p = risposte con X - risposte con Y del partecipante p, statistica sum_p d_p,
permutazione a segni ribaltati; qui con l'approssimazione normale della distribuzione di permutazione (varianza
condizionale sum_p d_p^2), che evita 10.000 permutazioni per ognuna delle migliaia di repliche. Soglie: alfa 0.05 e
0.05 / 4 = 0.0125 (Holm sui 4 tipi: chi ha p <= 0.0125 e' rifiutato qualunque sia il resto, quindi e' la soglia
prudente per UN tipo).

Taratura (``--v1``): dai partecipanti della v1 (``aau/human_study/responses_*``) l'accordo per risposta di ogni
metrica (il "differenziale realistico") e la varianza fra partecipanti oltre quella binomiale, che fissa l'ordine di
grandezza di sd_p. Scrive ``power_v2.md`` e ``power_v2.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

THIS_DIR = Path(__file__).resolve().parent
V1_DIR = THIS_DIR.parent / "human_study"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--q", default="0.55,0.575,0.60,0.65", help="quote marginali con X nello strato")
    p.add_argument("--n", default="10,15,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180,200,250,300,400")
    p.add_argument("--trials", type=int, default=9)
    p.add_argument("--items", type=int, default=60)
    p.add_argument("--sd-p", default="0.5", help="dev. std. logit fra partecipanti (lista per la sensibilita')")
    p.add_argument("--sd-t", default="0.0,0.8", help="dev. std. logit fra triplette")
    p.add_argument("--sims", type=int, default=4000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-v1", action="store_true")
    p.add_argument("--out", type=Path, default=THIS_DIR / "power_v2.md")
    return p.parse_args()


def calibrate_b(q: float, sd_p: float, sd_t: float) -> float:
    """b tale che E[expit(b + u + v)] = q (quadratura di Gauss-Hermite sulla somma normale)."""
    x, w = np.polynomial.hermite_e.hermegauss(80)
    s = math.hypot(sd_p, sd_t)
    lo, hi = -5.0, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (w * expit(mid + s * x)).sum() / w.sum() < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def power(n: int, q: float, sd_p: float, sd_t: float, args, rng) -> dict:
    b = calibrate_b(q, sd_p, sd_t)
    rej05 = rej0125 = 0
    for _ in range(args.sims):
        v = rng.normal(0.0, sd_t, args.items)
        u = rng.normal(0.0, sd_p, n)
        items = np.argsort(rng.random((n, args.items)), axis=1)[:, : args.trials]
        x = rng.random((n, args.trials)) < expit(b + u[:, None] + v[items])
        d = 2.0 * x.sum(1) - args.trials
        den = math.sqrt((d ** 2).sum())
        z = abs(d.sum()) / den if den > 0 else 0.0
        p = 2.0 * norm.sf(z)
        rej05 += p <= 0.05
        rej0125 += p <= 0.0125
    return {"n": n, "q": q, "diff": 2 * q - 1, "sd_p": sd_p, "sd_t": sd_t,
            "power_05": rej05 / args.sims, "power_0125": rej0125 / args.sims}


def v1_anchor() -> dict | None:
    """Accordo per risposta delle metriche della v1 e varianza fra partecipanti oltre la binomiale."""
    tp = V1_DIR / "triplets.json"
    files = sorted(V1_DIR.glob("responses_manual/*.json")) + sorted(V1_DIR.glob("responses_export/responses/*.json"))
    if not tp.exists() or not files:
        return None
    T = {t["id"]: t for t in json.loads(tp.read_text())["triplets"]}
    metrics = json.loads(tp.read_text())["meta"]["metrics"]
    rows = {m: [] for m in metrics}
    for f in files:
        doc = json.loads(f.read_text())
        trials = [x for x in doc.get("trials", doc.get("responses", [])) if T[x["triplet_id"]]["kind"] == "test"]
        for m in metrics:
            hits = sum(x.get("choice", x.get("chosen")) == T[x["triplet_id"]]["metrics"][m]["expected"]
                       for x in trials)
            rows[m].append((hits, len(trials)))
    out = {"n_participants": len(files), "metrics": {}}
    for m, r in rows.items():
        h, n = np.array(r, dtype=float).T
        pp = h / n
        q = h.sum() / n.sum()
        excess = max(pp.var(ddof=1) - (q * (1 - q) / n).mean(), 0.0)
        out["metrics"][m] = {"agreement": float(q), "per_participant": pp.round(3).tolist(),
                             "sd_between_prob": math.sqrt(excess),
                             "sd_between_logit": math.sqrt(excess) / (q * (1 - q))}
    return out


def needed(rows: list[dict], q: float, sd_p: float, sd_t: float, key: str, target: float):
    ok = [r["n"] for r in rows if r["q"] == q and r["sd_p"] == sd_p and r["sd_t"] == sd_t and r[key] >= target]
    return min(ok) if ok else None


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    qs = [float(x) for x in args.q.split(",")]
    ns = [int(x) for x in args.n.split(",")]
    sdps = [float(x) for x in args.sd_p.split(",")]
    sdts = [float(x) for x in args.sd_t.split(",")]
    rows = []
    for sd_p in sdps:
        for sd_t in sdts:
            for q in qs:
                for n in ns:
                    rows.append(power(n, q, sd_p, sd_t, args, rng))
                print(f"[power] sd_p={sd_p} sd_t={sd_t} q={q}: fatto", flush=True)
    anchor = None if args.no_v1 else v1_anchor()
    L = ["# Studio umano v2: calcolo di potenza (simulazione)", "",
         f"Generato da `aau/human_study_v2/power_v2.py`, {args.sims} studi simulati per cella, seed {args.seed}. "
         f"{args.trials} prove per strato e partecipante, {args.items} triplette per strato.", "",
         "q = quota marginale delle risposte che stanno con X nello strato `X_vs_Y`; differenziale d'accordo fra le "
         "due GT nello strato = 2q - 1. Potenza del test a segni ribaltati per partecipante, a due code.", ""]
    for sd_p in sdps:
        for sd_t in sdts:
            L += [f"## sd fra partecipanti {sd_p} logit, sd fra triplette {sd_t} logit", "",
                  "| q | differenziale | N per 80% (alfa 0.05) | N per 80% (alfa 0.0125) | N per 90% (alfa 0.0125) |",
                  "|---:|---:|---:|---:|---:|"]
            for q in qs:
                cells = [needed(rows, q, sd_p, sd_t, k, t) for k, t in
                         (("power_05", 0.8), ("power_0125", 0.8), ("power_0125", 0.9))]
                L.append(f"| {q:.3f} | {2 * q - 1:+.2f} | " +
                         " | ".join(f"{c}" if c else f"> {max(ns)}" for c in cells) + " |")
            L.append("")
    if anchor:
        L += ["## Taratura sulla v1", "",
              f"{anchor['n_participants']} partecipanti della v1 (36 test ciascuno, triplette dove le metriche "
              "litigano). Accordo per risposta e deviazione fra partecipanti oltre la binomiale:", "",
              "| metrica | accordo | sd fra partecipanti (prob.) | (logit) |", "|---|---:|---:|---:|"]
        for m, r in anchor["metrics"].items():
            L.append(f"| {m} | {r['agreement']:.3f} | {r['sd_between_prob']:.3f} | {r['sd_between_logit']:.2f} |")
        L.append("")
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                         "rows": rows, "v1": anchor}, indent=1), encoding="utf-8")
    print(args.out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
