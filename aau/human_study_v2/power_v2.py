#!/usr/bin/env python3
"""Studio umano v2: alfa empirico e potenza del test primario (quota con X nello strato ``X_vs_Y``) per simulazione.

    v3_work/unified_gt/run.sh aau/human_study_v2/power_v2.py      (run.sbatch, passo ``power``)

Modello generativo, uno strato alla volta (``--strata`` nome:prove:triplette, come la pagina: F_vs_S 12 prove su
200 triplette, F_vs_size 3 su 80, S_vs_maxabs 3 su 80). Ogni partecipante vede le sue prove, estratte senza
ripetizione dalle triplette dello strato. La risposta sta con X con probabilita' expit(b + u_p + v_t), con:
  - u_p ~ N(0, sd_p) per partecipante;
  - v_t ~ N(0, sd_t) per tripletta, fisso per studio come nella realta' (le triplette sono quelle).
``b`` si tara perche' la quota MARGINALE (sulla popolazione di partecipanti e di triplette) sia ``q``; il
differenziale d'accordo fra le due GT nello strato e' 2q - 1.

Test, lo stesso di ``analyze_v2.py``:
  - **primario:** errore standard per disegno incrociato, V = V_P + V_T - V_0 (bootstrap sui soli partecipanti,
    sulle sole triplette, varianza binomiale; ``analyze_v2.crossed_se``), p a due code su una t con gradi di
    liberta' di Satterthwaite;
  - **confronto:** bootstrap "pigeonhole" (righe e colonne insieme), che conta due volte il rumore binomiale;
  - **secondario:** segni ribaltati per partecipante (approssimazione normale della distribuzione di permutazione,
    varianza condizionale sum d_p^2), che ricampiona solo i partecipanti.

**Taratura dell'alfa, prima dei dati.** Sotto H0 (q = 0.5, ``--alpha-sims`` studi per condizione) si misurano i
rifiuti del primario alle soglie nominali di ``ALPHA_GRID``; la soglia TARATA e' la piu' grande per cui l'alfa
empirico resta <= 0.05 in TUTTE le condizioni (strati, sd, N). La potenza si calcola a quella soglia. I test sono a
cascata (gatekeeping: ogni strato si testa solo se il precedente ha rifiutato), ciascuno alla soglia tarata.
Per confronto, alla soglia 0.05: bootstrap "pigeonhole" e segni ribaltati.

Taratura (``--v1``): dai partecipanti della v1 (``aau/human_study/responses_*``) l'accordo per risposta di ogni
metrica e la varianza fra partecipanti oltre quella binomiale. Scrive ``power_v2.md`` e ``power_v2.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from scipy.stats import t as student_t

from analyze_v2 import satterthwaite

THIS_DIR = Path(__file__).resolve().parent
V1_DIR = THIS_DIR.parent / "human_study"
ALPHA_GRID = (0.025, 0.03, 0.035, 0.04, 0.045, 0.05)
ALPHA_TARGET = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--q", default="0.575,0.60,0.65,0.70", help="quote marginali con X nello strato")
    p.add_argument("--n", default="10,15,20,25,30,40,50,60,70,80,100,120,150,200,250,300")
    p.add_argument("--strata", default="F_vs_S:12:200,F_vs_size:3:80,S_vs_maxabs:3:80",
                   help="nome:prove:triplette")
    p.add_argument("--sd-p", default="0.5,1.0", help="dev. std. logit fra partecipanti")
    p.add_argument("--sd-t", default="0.0,0.8", help="dev. std. logit fra triplette")
    p.add_argument("--sims", type=int, default=1000)
    p.add_argument("--boot", type=int, default=200, help="repliche del bootstrap incrociato per studio simulato")
    p.add_argument("--alpha-sims", type=int, default=4000)
    p.add_argument("--alpha-n", default="15,20,30,40,60,80,120")
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


def one_study(n: int, trials: int, n_items: int, b: float, sd_p: float, sd_t: float, boot: int, rng) -> tuple:
    """(p incrociato, p segni) di uno studio simulato."""
    v = rng.normal(0.0, sd_t, n_items)
    u = rng.normal(0.0, sd_p, n)
    items = np.argsort(rng.random((n, n_items)), axis=1)[:, :trials]
    x = rng.random((n, trials)) < expit(b + u[:, None] + v[items])
    H = np.zeros((n, n_items))
    S = np.zeros((n, n_items))
    rows = np.repeat(np.arange(n), trials)
    np.add.at(H, (rows, items.ravel()), x.ravel())
    np.add.at(S, (rows, items.ravel()), 1.0)
    seen = S.sum(0) > 0                              # come crossed_se: solo le triplette viste
    H, S = H[:, seen], S[:, seen]
    n_seen = int(seen.sum())
    CP = rng.multinomial(n, np.full(n, 1.0 / n), size=boot).astype(float)
    CT = rng.multinomial(n_seen, np.full(n_seen, 1.0 / n_seen), size=boot).astype(float)
    q0 = H.sum() / S.sum()
    # primario, come analyze_v2.crossed_se: V_P + V_T - V_0
    vp = np.var((CP @ H.sum(1)) / (CP @ S.sum(1)), ddof=1)
    vt = np.var((CT @ H.sum(0)) / (CT @ S.sum(0)), ddof=1)
    v = max(vp + vt - q0 * (1.0 - q0) / S.sum(), vp, vt)
    df = satterthwaite(v, vp, vt, n, n_seen)
    p_x = 2.0 * student_t.sf(abs(q0 - 0.5) / math.sqrt(v), df) if v > 0 else 0.0
    # confronto: bootstrap pigeonhole (righe e colonne insieme) e segni ribaltati
    qb = np.einsum("rp,pt,rt->r", CP, H, CT) / np.einsum("rp,pt,rt->r", CP, S, CT)
    se = qb.std(ddof=1)
    p_g = 2.0 * norm.sf(abs(q0 - 0.5) / se) if se > 0 else 0.0
    d = 2.0 * x.sum(1) - trials
    den = math.sqrt((d ** 2).sum())
    p_s = 2.0 * norm.sf(abs(d.sum()) / den) if den > 0 else 1.0
    return p_x, p_s, p_g


def cell(task) -> dict:
    name, trials, n_items, n, q, sd_p, sd_t, sims, boot, seed = task
    rng = np.random.default_rng(seed)
    b = calibrate_b(q, sd_p, sd_t)
    P = np.array([one_study(n, trials, n_items, b, sd_p, sd_t, boot, rng) for _ in range(sims)])
    return {"stratum": name, "trials": trials, "items": n_items, "n": n, "q": q, "diff": 2 * q - 1, "sd_p": sd_p,
            "sd_t": sd_t, "sims": sims, "crossed": {f"{t:g}": float((P[:, 0] <= t).mean()) for t in ALPHA_GRID},
            "signflip_05": float((P[:, 1] <= 0.05).mean()), "pigeonhole_05": float((P[:, 2] <= 0.05).mean())}


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


def needed(rows, name, q, sd_p, sd_t, alpha, target):
    ok = [r["n"] for r in rows if r["stratum"] == name and r["q"] == q and r["sd_p"] == sd_p and r["sd_t"] == sd_t
          and r["crossed"][f"{alpha:g}"] >= target]
    return min(ok) if ok else None


def calibrate_alpha(arows: list[dict]) -> float:
    """Soglia nominale piu' grande di ALPHA_GRID con alfa empirico <= ALPHA_TARGET in tutte le condizioni."""
    ok = [t for t in ALPHA_GRID if max(r["crossed"][f"{t:g}"] for r in arows) <= ALPHA_TARGET]
    if not ok:
        raise SystemExit(f"nessuna soglia di {ALPHA_GRID} tiene l'alfa empirico <= {ALPHA_TARGET}")
    return max(ok)


def main() -> None:
    args = parse_args()
    fl = lambda s: [float(x) for x in s.split(",")]  # noqa: E731
    qs, sdps, sdts = fl(args.q), fl(args.sd_p), fl(args.sd_t)
    ns = [int(x) for x in args.n.split(",")]
    strata = [(nm, int(t), int(i)) for nm, t, i in (x.split(":") for x in args.strata.split(","))]
    ss = np.random.SeedSequence(args.seed)
    tasks, atasks = [], []
    for nm, t, i in strata:
        for sd_p in sdps:
            for sd_t in sdts:
                for n in [int(x) for x in args.alpha_n.split(",")]:
                    atasks.append((nm, t, i, n, 0.5, sd_p, sd_t, args.alpha_sims, args.boot, None))
                for q in qs:
                    for n in ns:
                        tasks.append((nm, t, i, n, q, sd_p, sd_t, args.sims, args.boot, None))
    seeds = ss.spawn(len(tasks) + len(atasks))
    tasks = [t[:-1] + (s,) for t, s in zip(tasks, seeds)]
    atasks = [t[:-1] + (s,) for t, s in zip(atasks, seeds[len(tasks):])]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    with mp.get_context("fork").Pool(workers) as pool:
        arows = pool.map(cell, atasks, chunksize=1)
        print("[power] alfa empirico: fatto", flush=True)
        rows = pool.map(cell, tasks, chunksize=1)
    anchor = None if args.no_v1 else v1_anchor()
    a_cal = calibrate_alpha(arows)
    L = ["# Studio umano v2: taratura dell'alfa e potenza (simulazione)", "",
         f"Generato da `aau/human_study_v2/power_v2.py`, seed {args.seed}; {args.sims} studi per cella di potenza, "
         f"{args.alpha_sims} per cella di alfa, {args.boot} repliche di bootstrap per studio. Strati "
         "(prove per partecipante / triplette): " + ", ".join(f"{s} {t}/{i}" for s, t, i in strata) + ".", "",
         f"**Soglia nominale tarata: {a_cal:g}** (la piu' grande di {list(ALPHA_GRID)} con alfa empirico <= "
         f"{ALPHA_TARGET} in tutte le {len(arows)} condizioni sotto H0).", "",
         "## Alfa empirico sotto H0 (q = 0.5)", "",
         "| strato | sd partecipanti | sd triplette | N | primario a "
         + " | primario a ".join(f"{t:g}" for t in ALPHA_GRID) + " | pigeonhole a 0.05 | segni a 0.05 |",
         "|---|---:|---:|---:|" + "---:|" * (len(ALPHA_GRID) + 2)]
    for r in arows:
        L.append(f"| {r['stratum']} | {r['sd_p']} | {r['sd_t']} | {r['n']} | "
                 + " | ".join(f"{r['crossed'][f'{t:g}']:.3f}" for t in ALPHA_GRID)
                 + f" | {r['pigeonhole_05']:.3f} | {r['signflip_05']:.3f} |")
    L.append("| massimo | | | | " + " | ".join(f"{max(r['crossed'][f'{t:g}'] for r in arows):.3f}"
                                               for t in ALPHA_GRID) + " | | |")
    L += ["", f"## Potenza del test primario alla soglia tarata {a_cal:g}: N tenuti per l'80% e il 90%", "",
          "q = quota marginale con X nello strato, differenziale = 2q - 1.", ""]
    for nm, t, _ in strata:
        L += [f"### {nm} ({t} prove per partecipante)", "",
              "| sd partecipanti | sd triplette | q | differenziale | 80% | 90% |", "|---:|---:|---:|---:|---:|---:|"]
        for sd_p in sdps:
            for sd_t in sdts:
                for q in qs:
                    c = [needed(rows, nm, q, sd_p, sd_t, a_cal, tg) for tg in (0.8, 0.9)]
                    L.append(f"| {sd_p} | {sd_t} | {q:.3f} | {2 * q - 1:+.2f} | " +
                             " | ".join(f"{x}" if x else f"> {max(ns)}" for x in c) + " |")
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
                                                         "alpha_calibrated": a_cal, "alpha": arows, "rows": rows,
                                                         "v1": anchor}, indent=1),
                                             encoding="utf-8")
    print(args.out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
