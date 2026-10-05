#!/usr/bin/env python3
"""Tabella del pilota pot: Delta per gruppo di coppie contro il controllo s1234, e zero-shot ICT.

Non calcola embedding: legge
  - in-domain BFM: il json di eval_by_topology.py scritto da eval_frame_topology.sbatch
    (riga result= della sentinella .done), gruppi crop / noisy / resample / all;
  - zero-shot ICT: le pair_metrics.csv del breakdown di ict_zeroshot_rank.sbatch (scenario
    clean, protocollo mesh-pair, gruppo all_cross come aau/ict/ict_summarize.py), con CI
    bootstrap subject-level della stessa funzione del paper (scripts/compute_bootstrap_ci.py).
La out dir ICT si trova da eval_key.txt (checkpoint, data dir, eval_seed), non dal nome: il
nome della run dir col fingerprint e' lo stesso per v1 e controllo.

    aau/run.sh aau/models/pilot_summary.py --out aau/runs/pilot_pot/summary.md \
        --row controllo <eval_frame dir> <ckpt> <ict data dir> \
        --row pot_m55 ... --row dual ...        (la prima riga e' il controllo)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
sys.path.insert(0, str(AAU_DIR / "ict"))
import ict_summarize as ict  # noqa: E402

GROUPS = ("crop", "noisy", "resample", "all")
CROP_MIN_GAIN = 0.05
ICT_MAX_DROP = 0.02

HEADER = f"""# Pilota pot: operatori DiffusionNet col pozzo di potenziale (alpha 0.55) per la robustezza al crop

## Criterio di successo (fissato il 2026-10-04, prima di qualunque risultato)

Un braccio passa se, rispetto al controllo appaiato `remesh_v1recipe_current_s1234_1055026`
(rilancio a --mem=180G di 1054511, cancellato per thrashing di memoria a fine run;
stessa ricetta v1, stesso wrapper train_fast.py con cache, `--frame current`, seed 1234):

1. Spearman sulle coppie con `crop` (eval_by_topology, 100 soggetti held-out) **>= controllo + {CROP_MIN_GAIN:.2f}**;
2. zero-shot ICT held-out (WBES_EVAL_SEED=1234, scenario clean, protocollo mesh-pair, all_cross)
   **>= controllo - {ICT_MAX_DROP:.2f}**.

Entrambe le condizioni. Un seed solo: e' un pilota, un passaggio va confermato su altri seed.

## Bracci

- `pot_m55`: operatori col pozzo (alpha 0.55, scala comune BFM 127507, `potential_operators.py
  --alpha-mode global`), pooling mean+max ristretto alla ROI del pozzo (roi_mask > 0.5).
- `dual`: due rami di operatori sullo stesso input (`aau/models/dn_dual_ops.py`): ogni blocco
  diffonde con la base standard e con quella del pozzo e concatena prima della MLP; width 103
  invece di 128 per avere gli stessi parametri (693140 contro 691584, +0.2%). Pooling pieno.
- Su ICT gli operatori col pozzo usano la scala comune di ICT (`calib_ict.json`), stesso alpha.
"""


def bfm_groups(eval_dir: Path) -> dict:
    done = eval_dir / ".done"
    if not done.exists():
        raise FileNotFoundError(f"eval non completa: manca {done}")
    result = next(l.split("=", 1)[1] for l in done.read_text().splitlines() if l.startswith("result="))
    import json
    payload = json.loads(Path(result).read_text())
    return {g: payload["groups"][g] for g in GROUPS}


def resolve_ckpt(path: Path) -> Path:
    """Il checkpoint, o il runs_root del training (una sola run dir con checkpoints/ dentro)."""
    if path.is_file():
        return path
    hits = sorted(path.glob("*/checkpoints/best_by_xtopo_mesh_clean.pth"))
    if len(hits) != 1:
        raise FileNotFoundError(f"atteso un solo best_by_xtopo_mesh_clean.pth sotto {path}, trovati {hits}")
    return hits[0]


def ict_stage(ckpt: Path, data_dir: Path, stage: str, seed: str) -> Path:
    want = {"ckpt": str(ckpt.resolve()), "data_dir": str(data_dir.resolve()), "eval_seed": seed}
    hits = []
    for key in sorted((AAU_DIR / "runs").glob("eval_*/eval_key.txt")):
        kv = dict(l.split("=", 1) for l in key.read_text().splitlines() if "=" in l)
        if all(kv.get(k) == v for k, v in want.items()):
            hits.append(key.parent / stage)
    hits = [h for h in hits if (h / ".done").exists()]
    if len(hits) != 1:
        raise FileNotFoundError(f"attesa una sola out dir ICT completa per {want} stage {stage}, trovate {hits}")
    return hits[0]


def ict_all_cross(stage: Path, boot, n_bootstrap: int, seed: int) -> dict:
    pm = ict.read_pair_metrics(stage)
    rng = np.random.default_rng(ict.stable_seed(seed, "pilot_pot", "all_cross"))
    row = ict.bootstrap_row(pm, "latent_distance", n_bootstrap, rng, boot)
    row["stage"] = str(stage)
    return row


def fmt(x: float, d: int = 4) -> str:
    return "n/d" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--row", nargs=4, action="append", required=True,
                    metavar=("NOME", "EVAL_FRAME_DIR", "CKPT_O_RUNS_ROOT", "ICT_DATA_DIR"))
    ap.add_argument("--ict-stage", default="ict_zeroshot_pilot_clean")
    ap.add_argument("--eval-seed", default="1234")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234, help="seme del ricampionamento")
    args = ap.parse_args()

    boot = ict.load_bootstrap_module()
    rows, problems = [], []
    for name, eval_dir, ckpt, ict_data in args.row:
        r = {"name": name}
        try:
            r["bfm"] = bfm_groups(Path(eval_dir))
        except Exception as exc:  # noqa: BLE001  (riga mancante, non tabella mancante)
            problems.append(f"{name} BFM: {exc}")
        try:
            r["ict"] = ict_all_cross(ict_stage(resolve_ckpt(Path(ckpt)), Path(ict_data), args.ict_stage, args.eval_seed),
                                     boot, args.n_bootstrap, args.seed)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name} ICT: {exc}")
        rows.append(r)

    ctrl = rows[0]
    lines = [HEADER, "## Risultati\n"]
    head = ["braccio"] + [f"{g} (Δ)" for g in GROUPS] + ["ICT all_cross [CI 95%] (Δ)", "esito"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("| " + " | ".join("---" for _ in head) + " |")
    for r in rows:
        cells = [r["name"]]
        for g in GROUPS:
            v = r.get("bfm", {}).get(g, {}).get("spearman")
            c = ctrl.get("bfm", {}).get(g, {}).get("spearman")
            if v is None:
                cells.append("n/d")
            elif r is ctrl or c is None:
                cells.append(fmt(v))
            else:
                cells.append(f"{fmt(v)} ({v - c:+.4f})")
        iv, ic = r.get("ict"), ctrl.get("ict")
        if iv is None:
            cells.append("n/d")
        else:
            s = f"{fmt(iv['spearman'])} [{fmt(iv['ci_low'], 3)}, {fmt(iv['ci_high'], 3)}]"
            if r is not ctrl and ic is not None:
                s += f" ({iv['spearman'] - ic['spearman']:+.4f})"
            cells.append(s)
        if r is ctrl:
            cells.append("controllo")
        else:
            try:
                d_crop = r["bfm"]["crop"]["spearman"] - ctrl["bfm"]["crop"]["spearman"]
                d_ict = r["ict"]["spearman"] - ctrl["ict"]["spearman"]
                ok = d_crop >= CROP_MIN_GAIN and d_ict >= -ICT_MAX_DROP
                cells.append(("PASSA" if ok else "NON PASSA") + f" (crop {d_crop:+.3f}, ICT {d_ict:+.3f})")
            except (KeyError, TypeError):
                cells.append("incompleto")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\nCoppie per gruppo (BFM): "
                 + ", ".join(f"{g} {ctrl['bfm'][g]['n_pairs']}" for g in GROUPS) if "bfm" in ctrl else "")
    lines.append("\n## Sorgenti\n")
    for (name, eval_dir, ckpt, ict_data), r in zip(args.row, rows):
        lines.append(f"- `{name}`: checkpoint `{ckpt}`; eval BFM `{eval_dir}`; "
                     f"ICT `{r.get('ict', {}).get('stage', 'n/d')}` (data `{ict_data}`)")
    if problems:
        lines.append("\n## Mancanti\n")
        lines += [f"- {p}" for p in problems]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
