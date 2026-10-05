#!/usr/bin/env python3
"""Tabella appaiata per seed: Spearman per gruppo, frame rms contro maxabs, e il Delta.

Legge le out dir di aau/eval_frame_topology.sbatch (aau/runs/eval_frame/<run>_<frame>/), solo
quelle con .done: e' la sentinella che python e' uscito 0, il json da solo potrebbe essere di
un'eval non verificata. Solo stdlib: gira sul frontend e su qualunque nodo.

  aau/eval_frame_table.py --out-dir aau/runs/eval_frame/_paired \
      --pair 1234 <dir rms s1234> <dir v1 maxabs> \
      --pair 2345 <dir rms s2345> <dir controllo s2345> ...

Delta = rms - maxabs, per seed; in fondo media e deviazione standard campionaria (ddof=1) dei
Delta sui seed. ``--reference LABEL DIR`` (ripetibile) aggiunge righe di riferimento, p.es. il v1
pubblicato quando il controllo del seed 1234 e' un retraining: stampate a parte, fuori dal Delta.

Budget: per ogni braccio "epoche fatte/previste (best a epoca N)", dalla run dir del checkpoint
valutato (riga ckpt= di eval_key.txt): ultima riga di train_log.csv, epochs di config.json,
best_epoch di best_by_xtopo_mesh_clean.txt. Un braccio con partial= nella .done (training
interrotto, aau/eval_frame_queue.sh) e' marcato PARZIALE nella tabella. Prima di appaiare controlla che i due bracci dello stesso seed abbiano valutato
gli stessi soggetti (riga "primi N valutati" di split_info.log e n_pairs per gruppo): un
seed di split diverso fra i due bracci renderebbe il Delta un confronto fra soggetti diversi.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

GROUPS = ("crop", "noisy", "resample", "all")


def budget_of(out_dir: Path) -> str:
    key = dict(line.split("=", 1) for line in (out_dir / "eval_key.txt").read_text().splitlines()
               if "=" in line)
    run_dir = Path(key["ckpt"]).parent.parent
    try:
        last = (run_dir / "train_log.csv").read_text().strip().splitlines()[-1].split(",")[0]
        cfg = json.loads((run_dir / "config.json").read_text())
        planned = cfg.get("args", cfg)["epochs"]
        best = dict(t.split("=", 1) for t in
                    (run_dir / "best_by_xtopo_mesh_clean.txt").read_text().split())["best_epoch"]
    except (OSError, KeyError, IndexError, ValueError) as exc:
        return f"? ({type(exc).__name__})"
    return f"{last}/{planned} (best {best})"


def load_arm(out_dir: Path) -> dict:
    done = out_dir / ".done"
    if not done.is_file():
        raise FileNotFoundError(f"manca {done}: eval non completata")
    fields = dict(line.split("=", 1) for line in done.read_text().splitlines() if "=" in line)
    blob = json.loads(Path(fields["result"]).read_text())
    split_line = ""
    split_log = out_dir / "split_info.log"
    if split_log.is_file():
        for line in split_log.read_text().splitlines():
            if "valutati:" in line:
                split_line = line.split("valutati:", 1)[1].strip()
    budget = budget_of(out_dir)
    if "partial" in fields:
        budget = f"PARZIALE {budget}"
    return {"dir": out_dir, "frame": blob.get("frame"), "groups": blob["groups"],
            "subjects_head": split_line, "job": fields.get("job", ""), "budget": budget}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pair", nargs=3, action="append", required=True,
                    metavar=("SEED", "RMS_DIR", "MAXABS_DIR"))
    ap.add_argument("--reference", nargs=2, action="append", default=[],
                    metavar=("LABEL", "DIR"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rows, missing, problems = [], [], []
    for seed, rms_dir, max_dir in args.pair:
        arms = {}
        for name, d in (("rms", Path(rms_dir)), ("maxabs", Path(max_dir))):
            try:
                arms[name] = load_arm(d)
            except FileNotFoundError as exc:
                missing.append(str(exc))
        if len(arms) < 2:
            continue
        r, m = arms["rms"], arms["maxabs"]
        if (r["frame"], m["frame"]) != ("rms", "current"):
            problems.append(f"seed {seed}: frame nei json {r['frame']}/{m['frame']}, attesi rms/current")
        if r["subjects_head"] != m["subjects_head"]:
            problems.append(f"seed {seed}: soggetti diversi fra i bracci: "
                            f"[{r['subjects_head']}] contro [{m['subjects_head']}]")
        for g in GROUPS:
            if r["groups"][g]["n_pairs"] != m["groups"][g]["n_pairs"]:
                problems.append(f"seed {seed} {g}: n_pairs {r['groups'][g]['n_pairs']} "
                                f"contro {m['groups'][g]['n_pairs']}")
            rows.append({"seed": seed, "group": g,
                         "rms": r["groups"][g]["spearman"], "maxabs": m["groups"][g]["spearman"],
                         "delta": r["groups"][g]["spearman"] - m["groups"][g]["spearman"],
                         "n_pairs": r["groups"][g]["n_pairs"],
                         "rms_dir": r["dir"].name, "maxabs_dir": m["dir"].name,
                         "rms_budget": r["budget"], "maxabs_budget": m["budget"]})

    refs = []
    for label, d in args.reference:
        try:
            refs.append((label, load_arm(Path(d))))
        except FileNotFoundError as exc:
            missing.append(str(exc))

    for msg in missing:
        print(f"MANCA: {msg}", file=sys.stderr)
    for msg in problems:
        print(f"PROBLEMA: {msg}", file=sys.stderr)
    if missing or problems:
        print("Tabella NON scritta: servono tutte le eval, appaiate sugli stessi soggetti.",
              file=sys.stderr)
        return 1

    seeds = [p[0] for p in args.pair]
    summary = []
    for g in GROUPS:
        sel = [row for row in rows if row["group"] == g]
        deltas = [row["delta"] for row in sel]
        summary.append({
            "group": g,
            "rms_mean": statistics.mean(row["rms"] for row in sel),
            "maxabs_mean": statistics.mean(row["maxabs"] for row in sel),
            "delta_mean": statistics.mean(deltas),
            "delta_std": statistics.stdev(deltas) if len(deltas) > 1 else float("nan"),
            "n_seeds": len(deltas),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "rms_vs_maxabs_by_seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(args.out_dir / "rms_vs_maxabs_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    lines = ["# Frame rms contro maxabs, Spearman vs D_GT per gruppo (eval_by_topology.py)", "",
             "Delta = rms - maxabs, appaiato per seed. Ultime due righe: media e dev. std "
             f"campionaria (ddof=1) dei Delta su {len(seeds)} seed.", "",
             "| seed | budget rms | budget maxabs | "
             + " | ".join(f"{g} rms | {g} maxabs | {g} Δ" for g in GROUPS) + " |",
             "|---|---|---|" + "---|" * (3 * len(GROUPS))]
    for seed in seeds:
        cells = []
        for g in GROUPS:
            row = next(r for r in rows if r["seed"] == seed and r["group"] == g)
            cells += [f"{row['rms']:.4f}", f"{row['maxabs']:.4f}", f"{row['delta']:+.4f}"]
        lines.append(f"| {seed} | {row['rms_budget']} | {row['maxabs_budget']} | "
                     + " | ".join(cells) + " |")
    lines.append("| media | | | " + " | ".join(
        f"{s['rms_mean']:.4f} | {s['maxabs_mean']:.4f} | {s['delta_mean']:+.4f}" for s in summary) + " |")
    lines.append("| dev.std Δ | | | " + " | ".join(
        x for s in summary for x in ("", "", f"{s['delta_std']:.4f}")) + " |")
    if refs:
        lines += ["", "Riferimento, fuori dal Delta appaiato:", "",
                  "| riferimento | frame | budget | " + " | ".join(GROUPS) + " |",
                  "|---|---|---|" + "---|" * len(GROUPS)]
        for label, arm in refs:
            lines.append(f"| {label} | {arm['frame']} | {arm['budget']} | " + " | ".join(
                f"{arm['groups'][g]['spearman']:.4f}" for g in GROUPS) + " |")
        with open(args.out_dir / "reference.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["label", "frame", "budget", "dir"] + list(GROUPS))
            for label, arm in refs:
                w.writerow([label, arm["frame"], arm["budget"], arm["dir"].name]
                           + [arm["groups"][g]["spearman"] for g in GROUPS])
    partial = [f"seed {r['seed']}" for r in rows if r["group"] == "all"
               and "PARZIALE" in r["rms_budget"] + r["maxabs_budget"]]
    if partial:
        lines += ["", f"ATTENZIONE: budget ridotto (training interrotto) in {', '.join(partial)}: "
                      "il Delta di quei seed confronta budget diversi."]
    lines += ["", "Bracci:"] + [f"- seed {r['seed']}: rms `{r['rms_dir']}`, maxabs `{r['maxabs_dir']}`"
                                 for r in rows if r["group"] == "all"]
    (args.out_dir / "rms_vs_maxabs.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nscritti in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
