#!/usr/bin/env python3
"""Tabella delle ablazioni v3 (B, C, E) contro il controllo s1234: aau/runs/ablations_v3/summary.md.

Non calcola embedding: legge
  - in-domain BFM: il json di aau/models/eval_cells.py (eval_ablation_topology.sbatch): gruppi
    crop / noisy / resample / all dell'autore e le 30 celle ordinate del repo;
  - margine latent - Chamfer: media sulle 30 celle del latent meno la media delle stesse celle di
    Chamfer, prese dallo stage topology del v1 (seed 1234, stessi 100 soggetti: verificato qui
    sulla lista dei soggetti). Chamfer non dipende dal modello, quindi il Delta del margine e' il
    Delta della media latent;
  - zero-shot ICT: le pair_metrics.csv del breakdown di ict_ablation_rank.sbatch, all_cross con CI
    bootstrap subject-level, con le stesse funzioni di aau/models/pilot_summary.py.

    aau/run.sh aau/models/ablation_summary.py --out aau/runs/ablations_v3/summary.md \
        --row controllo <eval_bfm dir> <runs_root> <ict data dir> --row B ... --row C ... --row E ...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
import pilot_summary as ps  # noqa: E402  (ict_stage, ict_all_cross, resolve_ckpt, fmt)

GROUPS = ("crop", "noisy", "resample", "all")
MARGIN_MIN_GAIN = 0.03
CROP_MIN_GAIN = 0.05
ICT_MAX_DROP = 0.02
V1_TOPO = AAU_DIR / "runs" / "eval_mixed_xtopo_rank0p5_id0p25_bs5_best_57bad1df" / "topology"

HEADER = f"""# Ablazioni v3: frame rms + token di taglia (B), operatori robusti ad area 1 (C), combinazione (E)

## Criterio di successo (BOARD_DIARY, "Progetto della versione nuova e ablazioni", fissato prima dei risultati)

> Criterio per ciascuna, fissato ora: almeno +0.03 sul margine latent − Chamfer medio sulle 30 coppie
> o +0.05 sulle coppie con crop, senza perdere più di 0.02 sullo zero-shot ICT.

Operativamente, rispetto al controllo appaiato `remesh_v1recipe_current_s1234_1055026` (stessa ricetta v1,
stesso wrapper train_fast.py con cache in RAM, `--frame current`, operatori standard, seed 1234):

1. Δ margine medio sulle 30 celle ordinate (eval_cells.py, 100 held-out) **>= +{MARGIN_MIN_GAIN:.2f}**,
   oppure Δ Spearman del gruppo `crop` (uno Spearman su tutte le coppie con crop, come il pilota pot)
   **>= +{CROP_MIN_GAIN:.2f}**;
2. **e** zero-shot ICT held-out (WBES_EVAL_SEED=1234, scenario clean, all_cross) **>= controllo − {ICT_MAX_DROP:.2f}**.

Un seed solo: un passaggio va confermato su altri seed.

## Bracci

- `B`: frame rms dell'input + token di taglia (log del raggio rms della mesh grezza, pesato per area,
  standardizzato su media e std delle mesh dei 400 soggetti di training BFM), concatenato dopo il pooling
  mean+max, prima della proiezione a 256 (`aau/models/ablation_hooks.py`). Su ICT il token e' standardizzato
  sui 4500 soggetti di training ICT (le coordinate grezze ICT non sono in unita' BFM).
- `C`: operatori robusti ad area 1 (`robust_laplacian.mesh_laplacian`, mollify di default, mesh centrata e
  riscalata ad area 1, k_eig 128; frame tangenti e gradienti come `compute_operators`), frame standard.
- `E`: B + C.
"""


def bfm_eval(eval_dir: Path) -> dict:
    done = eval_dir / ".done"
    if not done.exists():
        raise FileNotFoundError(f"eval non completa: manca {done}")
    result = next(l.split("=", 1)[1] for l in done.read_text().splitlines() if l.startswith("result="))
    return json.loads(Path(result).read_text())


def chamfer_cells() -> tuple[dict, list]:
    """Celle di Chamfer dello stage topology del v1, e i suoi soggetti."""
    with open(V1_TOPO / "topology_breakdown_summary.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cells = {(r["topology_a"], r["topology_b"]): float(r["chamfer_spearman"]) for r in rows}
    sel = json.loads((V1_TOPO / "crop__to__down8k" / "ranking_summary.json").read_text())["selected_subjects"]
    return cells, sel


def margin(ev: dict, ch: dict) -> dict:
    lat = {(c["a"], c["b"]): c["latent_spearman"] for c in ev["cells"]}
    if set(lat) != set(ch):
        raise ValueError(f"celle diverse: {sorted(set(lat) ^ set(ch))}")
    keys = sorted(lat)
    crop = [k for k in keys if "crop" in k]
    return {"latent": float(np.mean([lat[k] for k in keys])),
            "chamfer": float(np.mean([ch[k] for k in keys])),
            "margin": float(np.mean([lat[k] - ch[k] for k in keys])),
            "margin_crop": float(np.mean([lat[k] - ch[k] for k in crop])),
            "n_cells": len(keys)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--row", nargs=4, action="append", required=True,
                    metavar=("NOME", "EVAL_BFM_DIR", "CKPT_O_RUNS_ROOT", "ICT_DATA_DIR"))
    ap.add_argument("--ict-stage", default="ict_zeroshot_abl3_clean")
    ap.add_argument("--eval-seed", default="1234")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234, help="seme del ricampionamento")
    args = ap.parse_args()

    boot = ps.ict.load_bootstrap_module()
    ch, ch_subjects = chamfer_cells()
    rows, problems = [], []
    for name, eval_dir, ckpt, ict_data in args.row:
        r = {"name": name}
        try:
            ev = bfm_eval(Path(eval_dir))
            r["bfm"] = ev["groups"]
            if ev["subjects"] != ch_subjects:
                raise ValueError("soggetti diversi da quelli dello stage topology del v1: margine non calcolabile")
            r["margin"] = margin(ev, ch)
        except Exception as exc:  # noqa: BLE001  (riga mancante, non tabella mancante)
            problems.append(f"{name} BFM: {exc}")
        try:
            r["ict"] = ps.ict_all_cross(ps.ict_stage(ps.resolve_ckpt(Path(ckpt)), Path(ict_data),
                                                     args.ict_stage, args.eval_seed),
                                        boot, args.n_bootstrap, args.seed)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name} ICT: {exc}")
        rows.append(r)

    ctrl = rows[0]
    lines = [HEADER, "## Risultati\n"]
    head = (["braccio"] + [f"{g} (Δ)" for g in GROUPS]
            + ["latent medio 30 celle (Δ)", "margine − Chamfer 30 celle (Δ)", "ICT all_cross [CI 95%] (Δ)", "esito"])
    lines.append("| " + " | ".join(head) + " |")
    lines.append("| " + " | ".join("---" for _ in head) + " |")
    for r in rows:
        cells = [r["name"]]
        for g in GROUPS:
            v = r.get("bfm", {}).get(g, {}).get("spearman")
            c = ctrl.get("bfm", {}).get(g, {}).get("spearman")
            cells.append("n/d" if v is None else fmt_delta(v, c, r is ctrl))
        for k in ("latent", "margin"):
            v, c = r.get("margin", {}).get(k), ctrl.get("margin", {}).get(k)
            cells.append("n/d" if v is None else fmt_delta(v, c, r is ctrl))
        iv, ic = r.get("ict"), ctrl.get("ict")
        if iv is None:
            cells.append("n/d")
        else:
            s = f"{ps.fmt(iv['spearman'])} [{ps.fmt(iv['ci_low'], 3)}, {ps.fmt(iv['ci_high'], 3)}]"
            if r is not ctrl and ic is not None:
                s += f" ({iv['spearman'] - ic['spearman']:+.4f})"
            cells.append(s)
        if r is ctrl:
            cells.append("controllo")
        else:
            try:
                d_m = r["margin"]["margin"] - ctrl["margin"]["margin"]
                d_crop = r["bfm"]["crop"]["spearman"] - ctrl["bfm"]["crop"]["spearman"]
                d_ict = r["ict"]["spearman"] - ctrl["ict"]["spearman"]
                ok = (d_m >= MARGIN_MIN_GAIN or d_crop >= CROP_MIN_GAIN) and d_ict >= -ICT_MAX_DROP
                cells.append(("PASSA" if ok else "NON PASSA")
                             + f" (margine {d_m:+.3f}, crop {d_crop:+.3f}, ICT {d_ict:+.3f})")
            except (KeyError, TypeError):
                cells.append("incompleto")
        lines.append("| " + " | ".join(cells) + " |")

    if "bfm" in ctrl:
        lines.append("\nCoppie per gruppo (BFM): " + ", ".join(f"{g} {ctrl['bfm'][g]['n_pairs']}" for g in GROUPS)
                     + "; 30 celle da 4950 coppie. Chamfer medio sulle 30 celle (v1, stessi soggetti): "
                     + (f"{ctrl['margin']['chamfer']:.4f}." if "margin" in ctrl else "n/d."))
    lines.append("\n## Sorgenti\n")
    for (name, eval_dir, ckpt, ict_data), r in zip(args.row, rows):
        lines.append(f"- `{name}`: training `{ckpt}`; eval BFM `{eval_dir}`; "
                     f"ICT `{r.get('ict', {}).get('stage', 'n/d')}` (data `{ict_data}`)")
    lines.append(f"- Chamfer per cella: `{V1_TOPO / 'topology_breakdown_summary.csv'}`")
    if problems:
        lines.append("\n## Mancanti\n")
        lines += [f"- {p}" for p in problems]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if problems:
        sys.exit(1)


def fmt_delta(v: float, c: float | None, is_ctrl: bool) -> str:
    if is_ctrl or c is None or (isinstance(c, float) and math.isnan(c)):
        return ps.fmt(v)
    return f"{ps.fmt(v)} ({v - c:+.4f})"


if __name__ == "__main__":
    main()
