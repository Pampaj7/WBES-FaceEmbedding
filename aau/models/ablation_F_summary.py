#!/usr/bin/env python3
"""Tabella della variante F (crop casuali in training) contro il controllo s1234: aau/runs/ablation_F/summary.md.

Non calcola embedding: legge
  - REMESH: le out dir di aau/eval_frame_topology.sbatch (frame current) di F e del controllo,
    Spearman per gruppo crop / noisy / resample / all di eval_by_topology.py dell'autore, con
    load_arm di aau/eval_frame_table.py (stessi soggetti valutati, budget del training);
  - Multiface WS3a duro: il csv di aau/multiface/ws3a_analysis.py con le due metriche latent
    (ws3a_latent.py --metric ...), AUC b_vs_c (stesso soggetto con espressione diversa contro
    soggetti diversi con la stessa espressione) per coppia di topologie.
Solo stdlib: gira sul frontend.

    python3 aau/models/ablation_F_summary.py --f-eval <eval_frame F> --ctrl-eval <eval_frame controllo> \
        --ws3a-csv aau/runs/multiface_ws3a_hard/summary_hard_F.csv --out aau/runs/ablation_F/summary.md
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
sys.path.insert(0, str(AAU_DIR))
from eval_frame_table import GROUPS, load_arm  # noqa: E402

CROP_MIN_GAIN = 0.05
MAX_DROP = 0.02
MF_CROP = (("tracked", "crop"), ("remesh", "crop"), ("crop", "crop"))
MF_OTHER = (("tracked", "tracked"), ("tracked", "noisy"), ("down", "up"))
COMPARISON = "auc_b_vs_c"

HEADER = f"""# Variante F: crop casuali in training (ipotesi H6)

**Criterio, fissato prima dei numeri (BOARD_DIARY, "Ipotesi H6"):** almeno +{CROP_MIN_GAIN:.2f} di AUC sulle
coppie con crop di Multiface **oppure** +{CROP_MIN_GAIN:.2f} di Spearman sulle coppie con crop di REMESH, senza
perdere piu' di {MAX_DROP:.2f} altrove. Resi operativi cosi':
- Multiface: media dei Delta di AUC b_vs_c sulle tre coppie con crop ({", ".join(f"{a}->{b}" for a, b in MF_CROP)});
- REMESH: Delta Spearman del gruppo `crop` di eval_by_topology.py (100 soggetti held-out, seed 1234);
- altrove: Delta >= -{MAX_DROP:.2f} su REMESH noisy e resample e su Multiface {", ".join(f"{a}->{b}" for a, b in MF_OTHER)}.
Delta = F - controllo. Controllo appaiato: `remesh_v1recipe_current_s1234_1055026` (stessa ricetta v1, stesso
wrapper con cache, seed 1234, frame current). F differisce solo per la vista dati `datasets/REMESH/view_F`:
5 crop casuali in piu' per soggetto di training, che il trainer etichetta `crop` (statistiche in
`aau/runs/ablation_F/crops_stats.md`). Multiface ha 13 soggetti: gli IC delle AUC sono larghi.
"""


def ws3a_rows(path: Path, metric: str) -> dict:
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["metric"] == metric and r["comparison"] == COMPARISON:
                out[(r["topology_a"], r["topology_b"])] = (float(r["auc"]), float(r["ci_low"]), float(r["ci_high"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--f-eval", type=Path, required=True)
    ap.add_argument("--ctrl-eval", type=Path, required=True)
    ap.add_argument("--ws3a-csv", type=Path, required=True)
    ap.add_argument("--f-metric", default="latent_abl_F")
    ap.add_argument("--ctrl-metric", default="latent_ctrl_s1234")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    missing, problems, lines = [], [], [HEADER]
    arms = {}
    for name, d in (("F", args.f_eval), ("controllo", args.ctrl_eval)):
        try:
            arms[name] = load_arm(d)
        except (FileNotFoundError, KeyError) as exc:
            missing.append(f"REMESH {name}: {exc}")
    remesh = {}
    if len(arms) == 2:
        f, c = arms["F"], arms["controllo"]
        if (f["frame"], c["frame"]) != ("current", "current"):
            problems.append(f"frame nei json {f['frame']}/{c['frame']}, attesi current/current")
        if f["subjects_head"] != c["subjects_head"]:
            problems.append(f"soggetti diversi: [{f['subjects_head']}] contro [{c['subjects_head']}]")
        lines += ["## REMESH, Spearman vs D_GT per gruppo (eval_by_topology.py, frame current)", "",
                  f"Budget: F {f['budget']}, controllo {c['budget']}.", "",
                  "| gruppo | coppie | F | controllo | Δ |", "|---|---|---|---|---|"]
        for g in GROUPS:
            a, b = f["groups"][g], c["groups"][g]
            if a["n_pairs"] != b["n_pairs"]:
                problems.append(f"{g}: n_pairs {a['n_pairs']} contro {b['n_pairs']}")
            remesh[g] = a["spearman"] - b["spearman"]
            lines.append(f"| {g} | {a['n_pairs']} | {a['spearman']:.4f} | {b['spearman']:.4f} | {remesh[g]:+.4f} |")
        lines.append("")

    mf = {}
    if args.ws3a_csv.is_file():
        fr, cr = ws3a_rows(args.ws3a_csv, args.f_metric), ws3a_rows(args.ws3a_csv, args.ctrl_metric)
        lines += [f"## Multiface WS3a duro, AUC {COMPARISON} [IC 95% bootstrap sui soggetti]", "",
                  f"Metriche `{args.f_metric}` e `{args.ctrl_metric}` in `{args.ws3a_csv}`.", "",
                  "| topologie | F | controllo | Δ |", "|---|---|---|---|"]
        for pair in MF_OTHER[:1] + MF_CROP + MF_OTHER[1:]:
            if pair not in fr or pair not in cr:
                missing.append(f"WS3a {pair[0]}->{pair[1]}: riga assente per F o controllo")
                continue
            mf[pair] = fr[pair][0] - cr[pair][0]
            lines.append(f"| {pair[0]}->{pair[1]} | {fr[pair][0]:.3f} [{fr[pair][1]:.3f}, {fr[pair][2]:.3f}] | "
                         f"{cr[pair][0]:.3f} [{cr[pair][1]:.3f}, {cr[pair][2]:.3f}] | {mf[pair]:+.3f} |")
        lines.append("")
    else:
        missing.append(f"WS3a: {args.ws3a_csv} assente")

    lines += ["## Verdetto", ""]
    if not missing and not problems:
        mf_crop = sum(mf[p] for p in MF_CROP) / len(MF_CROP)
        gain = remesh["crop"] >= CROP_MIN_GAIN or mf_crop >= CROP_MIN_GAIN
        drops = {f"REMESH {g}": remesh[g] for g in ("noisy", "resample")}
        drops.update({f"Multiface {a}->{b}": mf[(a, b)] for a, b in MF_OTHER})
        lost = {k: v for k, v in drops.items() if v < -MAX_DROP}
        lines += [f"- Guadagno sul crop: REMESH {remesh['crop']:+.4f}, Multiface (media 3 coppie) {mf_crop:+.3f}: "
                  f"{'raggiunto' if gain else 'NON raggiunto'} (soglia +{CROP_MIN_GAIN:.2f}).",
                  "- Perdite altrove oltre -%.2f: %s." % (MAX_DROP, ", ".join(f"{k} {v:+.3f}" for k, v in lost.items())
                                                        if lost else "nessuna"),
                  f"- **{'PASSA' if gain and not lost else 'NON PASSA'}** il criterio fissato."]
    else:
        lines.append("n/d: tabella incompleta.")
    if missing or problems:
        lines += ["", "## Mancanti / problemi", ""] + [f"- {m}" for m in missing + problems]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nscritto {args.out}")
    return 1 if missing or problems else 0


if __name__ == "__main__":
    sys.exit(main())
