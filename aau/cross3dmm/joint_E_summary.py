#!/usr/bin/env python3
"""Tabella del confronto E congiunto / controllo congiunto: aau/runs/joint_E/summary.md.

Non calcola embedding: legge gli output di aau/cross3dmm/joint_E_eval.sbatch per i due bracci e i due
domini (eval_cells.py: gruppi; breakdown del repo: pair_metrics.csv) e, se presente, il csv WS3a
Multiface. Il criterio sta in HEADER ed e' stato scritto prima dei numeri: senza risultati lo script
scrive solo quello (e gira anche sul frontend, senza numpy).

    aau/run.sh aau/cross3dmm/joint_E_summary.py --ctrl-root aau/runs/joint_E/joint_E_ctrl_s1234_<job> \
        --e-root aau/runs/joint_E/joint_E_e_s1234_<job> [--ws3a-csv aau/runs/multiface_ws3a_hard/summary_hard_jointE.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
RUNS = AAU_DIR / "runs" / "joint_E"

BFM_MIN_GAIN = 0.03
ICT_MAX_DROP = 0.02
DOMAINS = ("bfm", "ict")
GROUPS = ("crop", "noisy", "resample", "all")
MF_PAIRS = (("tracked", "tracked"), ("tracked", "crop"), ("remesh", "crop"), ("tracked", "noisy"),
            ("down", "up"), ("crop", "crop"))
MF_METRICS = {"ctrl": ("latent_jointE_ctrl",), "e": ("latent_jointE_e_tokA", "latent_jointE_e_tok0")}

HEADER = f"""# E congiunto BFM+ICT contro controllo congiunto

## Criterio, scritto il 5 ottobre prima di qualunque numero

**E congiunto passa se batte il controllo congiunto di almeno +{BFM_MIN_GAIN:.2f} sul margine medio BFM e non
perde piu' di {ICT_MAX_DROP:.2f} su ICT.** Reso operativo cosi', Delta = E - controllo:

1. **BFM**: margine medio = media sulle 30 coppie ordinate di topologie (protocollo mesh-pair
   cross-topologia: `compare_model_vs_chamfer_topology_breakdown.py --ordered_topology_pairs
   --topology_pair_mode cross_only`, scenario clean) di (Spearman latent - Spearman Chamfer) contro la GT,
   sui 104 soggetti BFM held-out dello split. Passa se Delta >= +{BFM_MIN_GAIN:.2f}. Chamfer e' lo stesso nei
   due bracci (stessi soggetti, stesse mesh), quindi il Delta del margine e' il Delta del latent.
2. **ICT**: Spearman mesh-pair `all_cross` (stesso protocollo, tutte le coppie cross-topologia) sui 100
   soggetti ICT held-out estratti con seed 1234 dagli held-out ICT dello split. Passa se
   Delta >= -{ICT_MAX_DROP:.2f}.
3. Servono entrambe. Un seed solo: un passaggio va confermato su altri seed.

E' il criterio delle ablazioni v3 (`aau/runs/ablations_v3/summary.md`) con lo zero-shot ICT sostituito
dall'ICT in dominio, perche' qui ICT e' nel training. Le altre righe (gruppi crop/noisy/resample/all
di eval_by_topology, IC bootstrap per soggetto, WS3a Multiface) sono descrittive e non entrano nel verdetto.

## Disegno

- Sottoinsieme: BFM 500 soggetti (tutti) + ICT 1500 (`np.random.default_rng(1234).choice` sui 5000);
  split del trainer sull'unione (rebuild_subject_split, eval_fraction 0.2, seed 1234, come train_v2):
  BFM 396 train / 104 held-out, ICT 1204 / 296. GT del congiunto attuale (`datasets/JOINT_BFM_ICT/gt_matrix.npz`).
- Controllo: frame current, operatori del congiunto attuale (BFM cotangente area 1, ICT `train_ready`),
  niente token. E: frame rms, operatori robusti ad area 1 per entrambi i domini, token di taglia
  standardizzato per dominio sui soggetti di training del dominio (`size_token_joint_{{bfm,ict}}_s1234.json`).
- Ricetta v1, 120 epoche, cache in RAM (train_fast.py), batch a dominio singolo ed eval online su BFM
  (train_v2), seed 1234. Codice: `aau/cross3dmm/{{joint_E_prep.py,train_joint_E.py,train_joint_E.sbatch,joint_E_eval.sbatch}}`.
"""


def eval_dir(root: Path, dom: str) -> Path:
    return RUNS / "eval" / f"{root.name}__{dom}"


def eval_complete(d: Path) -> bool:
    """cells e i 15 shard del breakdown (joint_E_eval.sbatch), 30 pair_metrics.csv."""
    return ((d / ".done_cells").is_file() and len(list(d.glob(".done_topo__*"))) == 15
            and len(list((d / "topology").glob("*/pair_metrics.csv"))) == 30)


def load_pair_metrics(d: Path):
    import pandas as pd
    return pd.concat([pd.read_csv(p) for p in sorted((d / "topology").glob("*/pair_metrics.csv"))],
                     ignore_index=True)


def paired_delta_ci(pm_c, pm_e, dom: str, bm, args) -> dict:
    """IC bootstrap APPAIATO per soggetto del Delta e - ctrl: stesse estrazioni di soggetti per i due bracci,
    pesi count[a] * count[b] per coppia come weighted_bootstrap_spearman del repo (con np.repeat e la sua
    finite_spearman), Delta calcolato replica per replica. Statistiche:
      - margine: media sulle 30 celle di (latent - Chamfer); il Chamfer e' identico riga per riga nei due
        bracci (controllato qui), quindi il Delta del margine e' la media sulle celle del Delta latent;
      - all_cross: uno Spearman mesh-pair su tutte le coppie cross-topologia;
      - no_crop: protocollo primario, uno Spearman mesh-pair sulle coppie cross-topologia senza crop
        (ne' topology_a ne' topology_b = crop, 20 celle ordinate)."""
    import numpy as np

    sys.path.insert(0, str(AAU_DIR / "ict"))
    import ict_summarize as base

    key = ["subject_a", "subject_b", "topology_a", "topology_b"]
    cols = key + ["gt_distance", "raw_chamfer", "latent_distance"]
    m = pm_c[cols].merge(pm_e[cols], on=key, suffixes=("_c", "_e"), validate="one_to_one")
    if not len(m) == len(pm_c) == len(pm_e):
        raise SystemExit(f"{dom}: righe non appaiabili ({len(pm_c)} ctrl, {len(pm_e)} e, {len(m)} comuni)")
    for c in ("gt_distance", "raw_chamfer"):
        if not np.array_equal(m[f"{c}_c"].to_numpy(), m[f"{c}_e"].to_numpy()):
            raise SystemExit(f"{dom}: {c} diverso fra i bracci, il Delta non e' appaiato")
    subjects = np.array(sorted(set(m["subject_a"]) | set(m["subject_b"])))
    idx = {s: i for i, s in enumerate(subjects)}
    sa = m["subject_a"].map(idx).to_numpy()
    sb = m["subject_b"].map(idx).to_numpy()
    gt = m["gt_distance_c"].to_numpy(dtype=np.float64)
    ch = m["raw_chamfer_c"].to_numpy(dtype=np.float64)
    lat = {"ctrl": m["latent_distance_c"].to_numpy(dtype=np.float64),
           "e": m["latent_distance_e"].to_numpy(dtype=np.float64)}
    ta, tb = m["topology_a"].to_numpy(), m["topology_b"].to_numpy()
    cells = [(ta == a) & (tb == b) for a, b in sorted(set(zip(ta, tb)))]
    no_crop = (ta != "crop") & (tb != "crop")

    def sp(mask, values, w):
        k = mask & (w > 0)
        return bm.finite_spearman(np.repeat(gt[k], w[k]), np.repeat(values[k], w[k]))

    everything = np.ones(len(m), dtype=bool)

    def stats(w):
        out = {}
        for arm in ("ctrl", "e"):
            out[f"margin_{arm}"] = float(np.mean([sp(c, lat[arm], w) - sp(c, ch, w) for c in cells]))
            out[f"all_cross_{arm}"] = sp(everything, lat[arm], w)
            out[f"no_crop_{arm}"] = sp(no_crop, lat[arm], w)
        return {k: out[f"{k}_e"] - out[f"{k}_ctrl"] for k in ("margin", "all_cross", "no_crop")} | out

    point = stats(np.ones(len(m), dtype=np.int64))
    rng = np.random.default_rng(base.stable_seed(args.seed, dom, "paired_delta"))
    boot = []
    for _ in range(args.n_bootstrap):
        counts = np.bincount(rng.integers(0, len(subjects), size=len(subjects)), minlength=len(subjects))
        boot.append(stats(counts[sa] * counts[sb]))
    out = {"n_subjects": len(subjects), "n_pairs": len(m), "n_cells": len(cells),
           "n_pairs_no_crop": int(no_crop.sum()), "n_bootstrap": args.n_bootstrap}
    for k in ("margin", "all_cross", "no_crop"):
        v = np.array([b[k] for b in boot])
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[k] = {"delta": point[k], "ci_low": float(lo), "ci_high": float(hi), "frac_pos": float((v > 0).mean()),
                  "ctrl": point[f"{k}_ctrl"], "e": point[f"{k}_e"]}
        if k == "margin":
            out[k]["frac_ge_threshold"] = float((v >= BFM_MIN_GAIN).mean())
    return out


def arm_tables(root: Path, dom: str, splits: dict, bm, args) -> dict:
    """Margine per cella, mesh-pair all_cross con IC, gruppi, controllo leak di una (braccio, dominio)."""
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(AAU_DIR / "ict"))
    import ict_summarize as base

    d = eval_dir(root, dom)
    cells = json.loads((d / "cells.json").read_text())
    pm = load_pair_metrics(d)
    expected = set(splits["eval_subjects"][dom])
    seen = set(pm["subject_a"]) | set(pm["subject_b"])
    leak = {"pair_metrics": len(seen & set(splits["train"])), "cells": len(set(cells["subjects"]) & set(splits["train"])),
            "pair_metrics_eq_split": seen == expected, "cells_eq_split": set(cells["subjects"]) == expected}
    per_cell = []
    for (a, b), g in pm.groupby(["topology_a", "topology_b"], sort=True):
        lat = float(g["gt_distance"].corr(g["latent_distance"], method="spearman"))
        ch = float(g["gt_distance"].corr(g["raw_chamfer"], method="spearman"))
        per_cell.append({"a": a, "b": b, "n": len(g), "latent": lat, "chamfer": ch, "margin": lat - ch})
    out = {"cells": per_cell, "n_cells": len(per_cell),
           "margin_mean": float(np.mean([c["margin"] for c in per_cell])),
           "latent_mean": float(np.mean([c["latent"] for c in per_cell])),
           "chamfer_mean": float(np.mean([c["chamfer"] for c in per_cell])),
           "groups": {g: cells["groups"][g]["spearman"] for g in GROUPS if g in cells["groups"]},
           "n_subjects": len(seen), "leak": leak}
    # Stesse coppie (soggetto minore in A) in eval_cells: controllo indipendente degli shard del breakdown.
    ref = {(c["a"], c["b"]): c["latent_spearman"] for c in cells["cells"]}
    out["cells_vs_breakdown"] = max(abs(c["latent"] - ref[(c["a"], c["b"])]) for c in per_cell)
    for metric, col in base.METRICS:
        rng = np.random.default_rng(base.stable_seed(args.seed, dom, "all_cross", metric))
        out[f"all_cross_{metric}"] = base.bootstrap_row(pm, col, args.n_bootstrap, rng, bm)
    return out


def ws3a_rows(path: Path) -> dict:
    import csv
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["comparison"] == "auc_b_vs_c":
                out[(r["metric"], r["topology_a"], r["topology_b"])] = (float(r["auc"]), float(r["ci_low"]),
                                                                       float(r["ci_high"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ctrl-root", type=Path, default=None)
    ap.add_argument("--e-root", type=Path, default=None)
    ap.add_argument("--ws3a-csv", type=Path, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=RUNS / "summary.md")
    args = ap.parse_args()

    roots = {"ctrl": args.ctrl_root, "e": args.e_root}
    missing = [f"{arm} {dom}: {eval_dir(r, dom) if r else 'run non indicato'}"
               for arm, r in roots.items() for dom in DOMAINS
               if r is None or not eval_complete(eval_dir(r, dom))]
    lines = [HEADER, "## Risultati", ""]
    if missing:
        lines += ["(in attesa delle eval)", "", "## Mancanti", ""] + [f"- {m}" for m in missing]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"scritto {args.out} (solo criterio: {len(missing)} eval mancanti)")
        return 0

    sys.path.insert(0, str(AAU_DIR / "ict"))
    import ict_summarize as base

    bm = base.load_bootstrap_module()
    splits = json.loads((RUNS / "splits.json").read_text())
    res = {arm: {dom: arm_tables(r, dom, splits, bm, args) for dom in DOMAINS} for arm, r in roots.items()}

    d_bfm = res["e"]["bfm"]["margin_mean"] - res["ctrl"]["bfm"]["margin_mean"]
    d_ict = res["e"]["ict"]["all_cross_latent"]["spearman"] - res["ctrl"]["ict"]["all_cross_latent"]["spearman"]
    ok_bfm, ok_ict = d_bfm >= BFM_MIN_GAIN, d_ict >= -ICT_MAX_DROP
    verdict = ("PASSA" if ok_bfm and ok_ict else "NON PASSA") + \
              f" (BFM Delta margine {d_bfm:+.4f} {'>=' if ok_bfm else '<'} +{BFM_MIN_GAIN:.2f}; " \
              f"ICT Delta all_cross {d_ict:+.4f} {'>=' if ok_ict else '<'} -{ICT_MAX_DROP:.2f})"

    lines += [f"**Verdetto: {verdict}.**", "",
              "| braccio | dominio | soggetti | margine medio 30 celle (Δ) | latent medio 30 celle | "
              "Chamfer medio | mesh-pair all_cross latent [IC 95%] (Δ) | Chamfer all_cross | "
              "crop | noisy | resample | all |", "|" + "---|" * 12]
    for arm in ("ctrl", "e"):
        for dom in DOMAINS:
            r, c = res[arm][dom], res["ctrl"][dom]
            dm = "" if arm == "ctrl" else f" ({r['margin_mean'] - c['margin_mean']:+.4f})"
            lat = r["all_cross_latent"]
            dl = "" if arm == "ctrl" else f" ({lat['spearman'] - c['all_cross_latent']['spearman']:+.4f})"
            grp = " | ".join(f"{r['groups'].get(g, float('nan')):.4f}" for g in GROUPS)
            lines.append(f"| {arm} | {dom} | {r['n_subjects']} | {r['margin_mean']:.4f}{dm} | {r['latent_mean']:.4f} | "
                         f"{r['chamfer_mean']:.4f} | {lat['spearman']:.4f} [{lat['ci_low']:.3f}, {lat['ci_high']:.3f}]{dl} | "
                         f"{r['all_cross_chamfer']['spearman']:.4f} | {grp} |")
    lines += ["", "Gruppi crop/noisy/resample/all: uno Spearman su tutte le coppie del gruppo "
              "(eval_cells.py, aggregazione di eval_by_topology), stessi soggetti.", ""]

    paired = {dom: paired_delta_ci(load_pair_metrics(eval_dir(roots["ctrl"], dom)),
                                   load_pair_metrics(eval_dir(roots["e"], dom)), dom, bm, args) for dom in DOMAINS}
    (RUNS / "paired_delta_ci.json").write_text(json.dumps(paired, indent=1))
    lines += ["## IC bootstrap appaiato del Delta e - ctrl", "",
              f"{args.n_bootstrap} repliche, ricampionamento dei soggetti con reinserimento, STESSE repliche per i due "
              "bracci, pesi count[a]·count[b] per coppia (come weighted_bootstrap_spearman del repo); IC percentile "
              "95%. Protocollo primario: mesh-pair cross-topologia senza crop (coppie con ne' A ne' B = crop, 20 celle "
              "ordinate). Il margine usa le 30 celle; il Chamfer e' identico riga per riga nei bracci.", "",
              "| dominio | statistica | ctrl | e | Δ | IC 95% del Δ | repliche con Δ > 0 |", "|---|---|---|---|---|---|---|"]
    names = {"margin": "margine medio 30 celle", "all_cross": "mesh-pair all_cross",
             "no_crop": "mesh-pair cross senza crop (primario)"}
    for dom, k in (("bfm", "margin"), ("bfm", "all_cross"), ("bfm", "no_crop"), ("ict", "all_cross"), ("ict", "no_crop")):
        r = paired[dom][k]
        lines.append(f"| {dom} | {names[k]} | {r['ctrl']:.4f} | {r['e']:.4f} | {r['delta']:+.4f} | "
                     f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['frac_pos']:.1%} |")
    pm_ = paired["bfm"]["margin"]
    lines += ["", f"Margine BFM: repliche con Δ >= +{BFM_MIN_GAIN:.2f} (soglia del criterio) {pm_['frac_ge_threshold']:.1%}. "
              f"Soggetti {paired['bfm']['n_subjects']} BFM / {paired['ict']['n_subjects']} ICT; coppie mesh "
              f"{paired['bfm']['n_pairs']} / {paired['ict']['n_pairs']}, senza crop {paired['bfm']['n_pairs_no_crop']} / "
              f"{paired['ict']['n_pairs_no_crop']}. Numeri in `paired_delta_ci.json`.", ""]

    lines += ["## Controlli", ""]
    for dom in DOMAINS:
        dch = max(abs(a["chamfer"] - b["chamfer"]) for a, b in zip(res["ctrl"][dom]["cells"], res["e"][dom]["cells"]))
        lines.append(f"- {dom}: Chamfer per cella fra i due bracci, differenza massima {dch:.2e} "
                     f"({res['ctrl'][dom]['n_cells']} celle); latent per cella breakdown contro eval_cells, differenza "
                     f"massima ctrl {res['ctrl'][dom]['cells_vs_breakdown']:.1e}, e {res['e'][dom]['cells_vs_breakdown']:.1e}; "
                     f"leak ctrl {res['ctrl'][dom]['leak']}, e {res['e'][dom]['leak']}")
    lines.append("")

    if args.ws3a_csv is not None and args.ws3a_csv.is_file():
        mf = ws3a_rows(args.ws3a_csv)
        ref = MF_METRICS["ctrl"][0]
        metrics = [m for arm in ("ctrl", "e") for m in MF_METRICS[arm]]
        lines += ["## WS3a Multiface duro, AUC b_vs_c (Δ rispetto al controllo congiunto)", "",
                  f"Csv `{args.ws3a_csv}`. Token di E su Multiface: `tokA` = raggio convertito in unita' BFM "
                  "e standardizzato con le statistiche BFM del training, `tok0` = token neutro (0).", "",
                  "| metrica | " + " | ".join(f"{a}->{b}" for a, b in MF_PAIRS) + " |", "|---|" + "---|" * len(MF_PAIRS)]
        for m in metrics:
            cells = []
            for a, b in MF_PAIRS:
                v = mf.get((m, a, b))
                if v is None:
                    cells.append("--")
                    continue
                d = "" if m == ref or (ref, a, b) not in mf else f" ({v[0] - mf[(ref, a, b)][0]:+.3f})"
                cells.append(f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]{d}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Sorgenti", ""] + [f"- `{arm}`: training `{r}`; eval `{eval_dir(r, 'bfm')}`, `{eval_dir(r, 'ict')}`"
                                    for arm, r in roots.items()]
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"scritto {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
