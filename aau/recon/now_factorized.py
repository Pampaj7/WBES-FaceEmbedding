#!/usr/bin/env python3
"""d_P e d_F del modello fattorizzato su NoW: ogni ricostruzione contro la scansione del suo soggetto (NoW ranking).

    WBES_V3_FACTORIZED_OUT=full WBES_V3_SCALE_TABLES=<t1:t2:...> \
        aau/run.sh v3_work/trainer/eval_v3.py -- aau/recon/now_factorized.py --checkpoint <pth> --key factorized_s1234

Regole: ``aau/runs/evidence/now_ranking/REGOLE.md``, sez. 2.  La catena e' quella di ``now_latent.py`` (``build_v1_model``
ed ``encode`` di ``ws3b_latent.py``, operatori ad area unitaria k_eig 128), lanciata attraverso ``eval_v3.py`` come il
passo ``now`` di ``eval_body.sh``; con ``WBES_V3_FACTORIZED_OUT=full`` il modello restituisce z = [s, u].
Per ricostruzione r e scansione s del soggetto:

    d_P = c x dp_per_unit x ||u_r - u_s||          (c e dp_per_unit: riga ``--key`` di factorized_calibration.csv)
    d_F = sqrt((S_r - S_s)^2 + S_r S_s d_P^2),  S = exp(s)   (factorized_v3.form_distance, d_F calibrata)

Le ricostruzioni ``<metodo>_metric`` (``now_prepare_meshes.py --metric``) si confrontano con ``scan_face_metric``, le
altre con ``scan_face``.  Uscite: ``gt_fact_<key>_<metodo>.csv`` (``now_common.OUT_ROOT``), cache degli embedding in
``<--cache-root>/embeddings/fact_<key>_<tag>.npz`` (fuori dal repo).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import now_common as common  # noqa: E402
from ws3b_latent import build_v1_model, encode  # noqa: E402

CALIB = common.AAU_DIR / "runs" / "evidence" / "trainer_v3" / "factorized_calibration.csv"
ITEM_FIELDS = ("name", "subject", "challenge")
FIELDS = ITEM_FIELDS + ("s_rec", "s_scan", "du", "d_P", "d_F")
METRIC_SUFFIX = "_metric"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", type=str, default="3ddfa_v2,synergynet,prnet,3ddfa_v3,3ddfa_v3_metric")
    p.add_argument("--checkpoint", type=str, default=os.environ.get("WBES_CKPT", ""))
    p.add_argument("--key", type=str, required=True, help="riga di factorized_calibration.csv (es. factorized_s1234)")
    p.add_argument("--cache-root", type=Path, default=Path.home() / "data" / "now_ranking_work")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def calibration(key: str, checkpoint: Path) -> tuple[float, float]:
    """(c, dp_per_unit) della riga ``key``; il checkpoint deve essere quello della calibrazione."""
    rows = {r["key"]: r for r in csv.DictReader(open(CALIB))}
    if key not in rows:
        raise SystemExit(f"{key}: assente da {CALIB}")
    r = rows[key]
    if Path(r["checkpoint"]).resolve() != checkpoint:
        raise SystemExit(f"{key}: checkpoint {checkpoint} diverso da quello calibrato {r['checkpoint']}")
    return float(r["c_median"]), float(r["dp_per_unit"])


def main() -> None:
    import torch

    args = parse_args()
    if os.environ.get("WBES_V3_FACTORIZED_OUT") != "full":
        raise SystemExit("serve WBES_V3_FACTORIZED_OUT=full (embedding [s, u]) e il lancio attraverso eval_v3.py")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    c, dpu = calibration(args.key, checkpoint)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    metric = f"fact_{args.key}"
    cache = SimpleNamespace(out_root=args.cache_root, metric=metric, overwrite=args.overwrite)
    print(f"[now-fact] device={device} ckpt={checkpoint} c={c:.6f} dp_per_unit={dpu:.6f}", flush=True)

    items = common.load_items()
    subjects = common.subjects_of(items)
    model = build_v1_model(checkpoint, "", device)
    scans = {}
    for tag in sorted({"scan_face_metric" if m.endswith(METRIC_SUFFIX) else "scan_face" for m in methods}):
        scans[tag] = encode(model, tag, common.WORK_ROOT / f"{tag}{common.OPS_SUFFIX}", subjects, device, cache)

    for method in methods:
        ops = common.recon_face_ops_dir(method)
        present = {p.stem for p in ops.glob("*.npz")}
        mine = [it for it in items if it.name in present]
        if len(mine) != len(items):
            print(f"[now-fact] ATTENZIONE {method}: {len(mine)}/{len(items)} con operatori", flush=True)
        z = encode(model, method, ops, [it.name for it in mine], device, cache)
        Zr = np.stack([z[it.name] for it in mine]).astype(np.float64)
        sc = scans["scan_face_metric" if method.endswith(METRIC_SUFFIX) else "scan_face"]
        Zs = np.stack([sc[it.subject] for it in mine]).astype(np.float64)

        du = np.linalg.norm(Zr[:, 1:] - Zs[:, 1:], axis=1)
        dP = c * dpu * du
        Sr, Ss = np.exp(Zr[:, 0]), np.exp(Zs[:, 0])
        dF = np.sqrt((Sr - Ss) ** 2 + Sr * Ss * dP ** 2)
        rows = [{**{k: getattr(it, k) for k in ITEM_FIELDS}, "s_rec": float(Zr[i, 0]), "s_scan": float(Zs[i, 0]),
                 "du": float(du[i]), "d_P": float(dP[i]), "d_F": float(dF[i])} for i, it in enumerate(mine)]
        common.write_rows(common.OUT_ROOT / f"gt_fact_{args.key}_{method}.csv", FIELDS, rows,
                          method=method, key=args.key, checkpoint=str(checkpoint), c=c, dp_per_unit=dpu,
                          ops=str(ops), scale_tables=os.environ.get("WBES_V3_SCALE_TABLES", ""))
        print(f"[now-fact] {method}: {len(rows)} ricostruzioni, d_P mediana {np.median(dP):.4f}, d_F mediana "
              f"{np.median(dF):.2f} mm, S_rec mediana {np.median(Sr):.1f} mm", flush=True)


if __name__ == "__main__":
    main()
