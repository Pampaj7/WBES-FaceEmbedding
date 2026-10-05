#!/usr/bin/env python3
"""Il latent di WS3a/WS3b riproduce l'eval di WS2 sul congiunto BFM+ICT?  Controllo prima dei dati reali.

    WBES_CKPT=<congiunto> sbatch --gres=gpu:t4:1 aau/recon/ws3b_joint_check.sbatch

``ws3b_latent.py`` e ``ws3a_latent.py`` caricano il checkpoint e codificano le mesh con la
loro catena (``build_v1_model`` + ``GTReadyDataset`` + ``forward_model``), non con lo script
di breakdown che ha fatto la tabella di WS2.  Prima di fidarsi dei loro numeri sul congiunto
si rifa' qui una cella di WS2 con QUELLA catena -- le funzioni di ``ws3b_latent.py``
importate, non copiate -- e la si confronta con le ``pair_metrics.csv`` del breakdown:

* vista ``datasets/WS2_CROSS3DMM/joint__bfm`` (BFM held-out del congiunto, operatori ad
  area unitaria), primi ``--n-subjects`` soggetti in ordine, una mesh per topologia;
* scenario clean (``add_noise=False``), mesh-pair sulle 30 coppie ordinate di topologie;
* per ogni riga delle pair_metrics fra quei soggetti: distanza latente ricalcolata contro
  ``latent_distance``, e Spearman con ``gt_distance`` dalle due colonne.

Esce con 1 se lo scarto massimo supera ``--tol`` o se lo Spearman differisce di piu' di
``--tol-spearman``: cosi' un ``--dependency=afterok`` ferma i job sui dati reali.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ws3b_common as common  # noqa: E402
import ws3b_latent  # noqa: E402

sys.path.insert(0, str(common.AAU_DIR / "cross3dmm"))
from ws2_summarize import find_eval_dir, read_pair_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", type=str, default=os.environ.get("WBES_CKPT", ""))
    p.add_argument("--view", type=Path,
                   default=common.REPO_ROOT / "datasets" / "WS2_CROSS3DMM" / "joint__bfm")
    p.add_argument("--n-subjects", type=int, default=10)
    p.add_argument("--out-root", type=Path, default=common.AAU_DIR / "runs" / "ws3b_joint_check",
                   help="cache degli embedding e csv del confronto")
    p.add_argument("--tol", type=float, default=1e-3, help="scarto massimo sulla distanza")
    p.add_argument("--tol-spearman", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    import torch
    from scipy.stats import spearmanr

    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint non trovato: {checkpoint_path}")
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available())
                          else "cpu")

    stage = find_eval_dir(common.AAU_DIR / "runs", checkpoint_path, str(args.view)) / "ws2_cell"
    if not (stage / ".done").exists():
        raise SystemExit(f"stage WS2 incompleta (manca .done): {stage}")
    pm = read_pair_metrics(stage / "topology")
    subjects = sorted(set(pm["subject_a"]) | set(pm["subject_b"]))[: args.n_subjects]
    pm = pm[pm["subject_a"].isin(subjects) & pm["subject_b"].isin(subjects)].reset_index(drop=True)
    if (pm["n_mesh_pairs"] != 1).any():
        raise SystemExit("pair_metrics con piu' mesh per (soggetto, topologia): confronto ambiguo")
    print(f"[joint-check] ckpt={checkpoint_path}\n[joint-check] ws2={stage}\n"
          f"[joint-check] soggetti={subjects} righe={len(pm)} device={device}", flush=True)

    # Le mesh della vista si chiamano <soggetto>_GTready_<topologia>.npz.
    names = sorted({f"{s}_GTready_{t}" for s, t in zip(pm["subject_a"], pm["topology_a"])}
                   | {f"{s}_GTready_{t}" for s, t in zip(pm["subject_b"], pm["topology_b"])})
    enc_args = SimpleNamespace(out_root=args.out_root.resolve(), overwrite=True, metric="check")
    model = ws3b_latent.build_v1_model(checkpoint_path, "", device)
    latents = ws3b_latent.encode(model, args.view.name, args.view, names, device, enc_args)

    A = np.stack([latents[f"{s}_GTready_{t}"] for s, t in zip(pm["subject_a"], pm["topology_a"])])
    B = np.stack([latents[f"{s}_GTready_{t}"] for s, t in zip(pm["subject_b"], pm["topology_b"])])
    pm["latent_recomputed"] = np.linalg.norm(A.astype(np.float64) - B.astype(np.float64), axis=1)
    diff = (pm["latent_recomputed"] - pm["latent_distance"]).abs()
    rho_ws2 = spearmanr(pm["latent_distance"], pm["gt_distance"]).statistic
    rho_new = spearmanr(pm["latent_recomputed"], pm["gt_distance"]).statistic
    rho_cham = spearmanr(pm["raw_chamfer"], pm["gt_distance"]).statistic
    pm.to_csv(enc_args.out_root / "pair_check.csv", index=False)

    print(f"[joint-check] |delta distanza| max={diff.max():.3g} mediana={diff.median():.3g} "
          f"(distanza mediana {pm['latent_distance'].median():.4f})")
    print(f"[joint-check] Spearman mesh-pair cross-topology clean, {len(subjects)} soggetti: "
          f"WS2 {rho_ws2:.4f}, ricalcolato {rho_new:.4f}, chamfer {rho_cham:.4f}")
    ok = diff.max() <= args.tol and abs(rho_new - rho_ws2) <= args.tol_spearman
    print(f"[joint-check] {'RIPRODOTTO' if ok else 'NON RIPRODOTTO'} "
          f"(tol {args.tol} sulla distanza, {args.tol_spearman} sullo Spearman)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
