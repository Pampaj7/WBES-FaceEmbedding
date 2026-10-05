#!/usr/bin/env python3
"""Token di taglia per Multiface (WS3a duro), nelle due varianti chieste per B ed E.

Il token dei bracci B/E e' log del raggio rms della mesh GREZZA, standardizzato con media e std del
training BFM (aau/models/size_token.py). Le mesh Multiface sono in mm, quelle BFM in un'altra unita'
(~1e5), quindi il log r grezzo di Multiface non e' confrontabile con le statistiche BFM. Due modi:

  a_bfmunits  log r_MF + log k, con k = mediana(r BFM original) / mediana(r Multiface tracked):
              Multiface portato in unita' BFM con UN fattore per tutte le topologie, poi
              standardizzato con le statistiche BFM. Il crop resta piu' piccolo del tracked, come
              in training il crop BFM e' piu' piccolo dell'original.
  b_neutral   token = media del training (valore standardizzato 0) per ogni mesh: il modello senza
              informazione di taglia.

Le topologie Multiface hanno gli STESSI nomi di file (manifest.csv), quindi una tabella sola per
nome non basta: si scrive una tabella per topologia, nel formato di SizeTokenTable
(aau/models/ablation_hooks.py), e aau/multiface/ws3a_latent_arm.py la sceglie per cartella.

    aau/run.sh aau/multiface/mf_size_token.py --bfm-table aau/runs/ablations_v3/size_token_bfm_s1234.json \
        --out-dir aau/runs/multiface_ws3a_hard/size_token/bfm_s1234
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "models"))

import ws3a_common as common  # noqa: E402
from size_token import log_rms_radius  # noqa: E402

TOPOLOGIES = ("tracked", "remesh", "crop", "noisy", "down", "up")
MODES = ("a_bfmunits", "b_neutral")


def read_one(path: Path) -> tuple[str, float]:
    with np.load(path, allow_pickle=False) as z:
        V = z["verts"] if "verts" in z.files else z["V"]
        F = z["faces"] if "faces" in z.files else z["F"]
        return path.name[:-len(".npz")], log_rms_radius(V, F)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bfm-table", type=Path, required=True,
                    help="tabella BFM di size_token.py: log r grezzi e statistiche del training")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    bfm = json.loads(args.bfm_table.read_text())
    mean, std = float(bfm["train"]["mean"]), float(bfm["train"]["std"])
    bfm_orig = np.array([v for n, v in bfm["log_r"].items() if n.endswith("_GTready_original")])
    if bfm_orig.size == 0:
        raise SystemExit(f"nessuna mesh original in {args.bfm_table}")

    log_r: dict[str, dict[str, float]] = {}
    for topo in TOPOLOGIES:
        files = sorted(p for p in (common.PREP_DIR / topo).iterdir()
                       if p.suffix == ".npz" and not p.name.startswith("."))
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            log_r[topo] = dict(ex.map(read_one, files))
        print(f"[mf-token] {topo}: {len(log_r[topo])} mesh", flush=True)

    # Mediana del raggio = exp(mediana del log), il log e' monotono.
    med_bfm = float(np.median(bfm_orig))
    med_mf = float(np.median(list(log_r["tracked"].values())))
    log_k = med_bfm - med_mf
    print(f"[mf-token] mediana r BFM original {math.exp(med_bfm):.6g} ({bfm_orig.size} mesh), "
          f"Multiface tracked {math.exp(med_mf):.6g} mm ({len(log_r['tracked'])} mesh): "
          f"k = {math.exp(log_k):.6g} (log k {log_k:+.5f})", flush=True)

    rows = []
    for mode in MODES:
        for topo in TOPOLOGIES:
            if mode == "a_bfmunits":
                vals = {n: v + log_k for n, v in log_r[topo].items()}
            else:
                vals = {n: mean for n in log_r[topo]}
            tok = (np.array(list(vals.values())) - mean) / std
            payload = {
                "collection": f"multiface_{topo}_{mode}",
                "source_dir": str((common.PREP_DIR / topo).resolve()),
                "definition": ("log rms radius, area-weighted, raw coordinates (mm) + log k, k = BFM "
                               "original / Multiface tracked median radius" if mode == "a_bfmunits"
                               else "neutral: every mesh at the BFM training mean (standardized token 0)"),
                "conversion": {"k": math.exp(log_k), "log_k": log_k,
                               "median_r_bfm_original": math.exp(med_bfm),
                               "median_r_multiface_tracked_mm": math.exp(med_mf),
                               "n_bfm_original": int(bfm_orig.size)},
                "train": dict(bfm["train"], source_table=str(args.bfm_table.resolve())),
                "log_r": vals,
            }
            out = args.out_dir / mode / f"{topo}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=1))
            rows.append((mode, topo, len(vals), float(tok.mean()), float(tok.std()),
                         float(tok.min()), float(tok.max())))

    lines = ["# Token di taglia Multiface (WS3a duro)\n",
             f"Statistiche del training da `{args.bfm_table}`: media {mean:.5f}, std {std:.5f} "
             f"(log r in unita' BFM).\n",
             f"Fattore di conversione (a): k = mediana r BFM original / mediana r Multiface tracked = "
             f"{math.exp(med_bfm):.6g} / {math.exp(med_mf):.6g} mm = **{math.exp(log_k):.6g}** "
             f"({bfm_orig.size} mesh BFM original, {len(log_r['tracked'])} mesh tracked).\n",
             "Token standardizzato per variante e topologia:\n",
             "| variante | topologia | mesh | media | std | min | max |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {m} | {t} | {n} | {mu:+.3f} | {sd:.3f} | {lo:+.3f} | {hi:+.3f} |"
              for m, t, n, mu, sd, lo, hi in rows]
    md = "\n".join(lines) + "\n"
    (args.out_dir / "README.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
