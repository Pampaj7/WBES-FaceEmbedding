#!/usr/bin/env python3
"""H3: consistenza spettrale fra topologie per variante di operatore, da h3_parts/.

Legge i npz di ``h3_spectral.py`` (autovalori e phi^2 nei punti campione) e calcola, per
variante e per coppia di topologie:

  disp   dispersione relativa degli autovalori lambda_1..lambda_64 fra le topologie dello
         stesso soggetto: std/media per indice (std di popolazione, su 2 topologie e'
         |a-b|/(a+b)), mediana sugli indici, media sui soggetti.  ``all6`` = tutte e 6.
  hks    errore relativo mediano fra le HKS dello stesso soggetto in due topologie, nei
         punti corrispondenti validi in entrambe: 2|hA-hB|/(hA+hB), mediana su punti e
         tempi, media sui soggetti.
  sep    hks fra topologie dello stesso soggetto diviso la stessa distanza fra soggetti
         diversi nella stessa topologia (media delle due topologie della coppia; i soggetti
         sono in corrispondenza perche' i punti campione sono indici dell'original BFM, che
         ha connettivita' comune).  Piu' basso = meglio.

I tempi HKS sono 8, log-spaziati in [4 ln10 / lambda_64, 4 ln10 / lambda_1] (Sun et al.
2009), con lambda presi come mediana sugli ``original`` della variante: FISSI per variante,
non per mesh, altrimenti la scelta del tempo normalizzerebbe la scala e (a) diventerebbe (c).

``e_roi`` e' il pozzo con l'HKS ristretta ai punti dentro la regione d'interesse (roi>0.5 in
entrambe le mesh): fuori dal pozzo phi ~ 0 e l'errore relativo e' rumore numerico.

    AAU_NV="" srun -p cpu -c 4 --mem=16G aau/run.sh aau/brainstorm/h3_table.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "baselines"))

import common  # noqa: E402

RUN_DIR = common.AAU_DIR / "runs" / "brainstorm"
VARIANTS = ("a", "b", "c", "d", "e", "e_roi")
LABELS = {"a": "a cotan", "b": "b robusto", "c": "c cotan+area1", "d": "d robusto+area1",
          "e": "e pozzo 0.55", "e_roi": "e pozzo 0.55, solo ROI"}
N_TIMES = 8
PAIRS = list(itertools.combinations(common.TOPOLOGIES, 2))
FOCUS = ("down8k", "noisy", "crop", "remesh", "up60k")


def relerr(hA: np.ndarray, hB: np.ndarray, mask: np.ndarray) -> float:
    a, b = hA[mask], hB[mask]
    if a.size == 0:
        return float("nan")
    return float(np.median(2.0 * np.abs(a - b) / (a + b + 1e-300)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", type=Path, default=RUN_DIR / "h3_parts")
    ap.add_argument("--out", type=Path, default=RUN_DIR / "h3")
    args = ap.parse_args()

    files = sorted(args.parts.glob("id*.npz"))
    data = [dict(np.load(f)) for f in files]
    subjects = [f.stem for f in files]
    print(f"[h3-table] {len(subjects)} soggetti", flush=True)
    T = common.TOPOLOGIES

    def ev(var, s, t):
        return data[s][f"{t}/{var[0]}/evals"]

    rows = []
    summary: dict = {"n_subjects": len(subjects), "times": {}}
    for var in VARIANTS:
        base = var[0]
        # --- autovalori
        lam = np.stack([np.stack([ev(var, s, t)[1:65] for t in T]) for s in range(len(data))])
        rel = lam.std(1) / lam.mean(1)                          # (S, 64)
        disp = {"all6": float(np.median(rel, 1).mean())}
        for tA, tB in PAIRS:
            a, b = lam[:, T.index(tA)], lam[:, T.index(tB)]
            disp[f"{tA}__{tB}"] = float(np.median(np.abs(a - b) / (a + b), 1).mean())
        # controllo con la statistica dell'autore (weyl_check.py: primi 30 modi, media)
        author30 = float((lam[:, :, :30].std(1) / lam[:, :, :30].mean(1)).mean())

        # --- HKS a tempi fissi per variante
        orig = np.stack([ev(var, s, "original") for s in range(len(data))])
        l1, lk = float(np.median(orig[:, 1])), float(np.median(orig[:, 64]))
        times = np.geomspace(4 * np.log(10) / lk, 4 * np.log(10) / l1, N_TIMES)
        summary["times"][var] = times.tolist()
        H, M = {}, {}
        for s in range(len(data)):
            for t in T:
                e = ev(var, s, t)
                H[s, t] = data[s][f"{t}/{base}/phi2"].astype(np.float64) @ np.exp(-np.outer(e, times))
                m = data[s][f"{t}/valid"].copy()
                if var == "e_roi":
                    m &= data[s][f"{t}/e/roi"] > 0.5
                M[s, t] = m[:, None].repeat(N_TIMES, 1)
        within = {(tA, tB): float(np.nanmean([relerr(H[s, tA], H[s, tB], M[s, tA] & M[s, tB])
                                              for s in range(len(data))]))
                  for tA, tB in PAIRS}
        between = {t: float(np.nanmean([relerr(H[i, t], H[j, t], M[i, t] & M[j, t])
                                        for i, j in itertools.combinations(range(len(data)), 2)]))
                   for t in T}
        hks_all = float(np.mean(list(within.values())))
        sep_all = hks_all / float(np.mean(list(between.values())))
        rows.append({"variant": var, "pair": "all", "disp": disp["all6"], "hks": hks_all,
                     "sep": sep_all, "author30": author30})
        for tA, tB in PAIRS:
            w = within[tA, tB]
            rows.append({"variant": var, "pair": f"{tA}__{tB}", "disp": disp[f"{tA}__{tB}"],
                         "hks": w, "sep": w / (0.5 * (between[tA] + between[tB]))})
        summary[var] = {"between": between, "lambda1_med": l1, "lambda64_med": lk,
                        "author30": author30}
        print(f"[h3-table] {var}: disp {disp['all6']:.4f} hks {hks_all:.4f} sep {sep_all:.4f} "
              f"(autore-30 {author30:.4f})", flush=True)

    import csv
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "pair", "disp", "hks", "sep", "author30"])
        w.writeheader()
        w.writerows(rows)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))

    get = {(r["variant"], r["pair"]): r for r in rows}

    def pk(t):
        return "__".join(sorted(("original", t), key=T.index))

    lines = [(RUN_DIR / "PREDICTIONS.md").read_text(), "",
             f"# H3: consistenza spettrale per variante di operatore ({len(subjects)} soggetti "
             "held-out BFM, k=64 modi non banali)", ""]
    for key, title in (("disp", "Dispersione relativa autovalori 1..64 (piu' basso = meglio)"),
                       ("hks", "Errore relativo mediano HKS, stesso soggetto fra topologie"),
                       ("sep", "Separabilita' HKS: stesso soggetto / soggetti diversi (piu' basso = meglio)")):
        lines += [f"## {title}", "",
                  "| variante | tutte | " + " | ".join(f"orig–{t}" for t in FOCUS) + " |",
                  "|---|---:|" + "---:|" * len(FOCUS)]
        for var in VARIANTS:
            vals = [get[var, "all"][key]] + [get[var, pk(t)][key] for t in FOCUS]
            lines.append(f"| {LABELS[var]} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
        lines.append("")
    lines += ["Controllo con la statistica dell'autore (STATUS.md: raw 0.2202, λ·A 0.0577; media "
              "sui primi 30 modi): " + ", ".join(f"{v} {get[v, 'all']['author30']:.4f}"
                                                for v in ("a", "b", "c", "d", "e")), ""]

    # verdetto H3: (b) contro (a) su original–down8k e original–noisy, entrambe le misure
    checks = []
    for t in ("down8k", "noisy"):
        for key in ("disp", "hks"):
            a, b = get["a", pk(t)][key], get["b", pk(t)][key]
            checks.append((t, key, (a - b) / a))
    ok = all(red >= 0.20 for _, _, red in checks)
    lines += ["## Verdetto H3", "",
              "Riduzione relativa (a−b)/a: " + ", ".join(f"{t} {k} {r:+.1%}" for t, k, r in checks),
              "", f"**H3 {'CONFERMATA' if ok else 'NON confermata'}** (soglia 20% su tutte e quattro).",
              "", "Stesso confronto (c→d), entrambe ad area 1: " + ", ".join(
                  f"{t} {k} {(get['c', pk(t)][k] - get['d', pk(t)][k]) / get['c', pk(t)][k]:+.1%}"
                  for t in ("down8k", "noisy") for k in ("disp", "hks")), "",
              f"Tutte le 15 coppie in `{args.out.with_suffix('.csv').name}`; tempi HKS e distanze "
              f"fra soggetti in `{args.out.with_suffix('.json').name}`."]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[2:]))


if __name__ == "__main__":
    main()
