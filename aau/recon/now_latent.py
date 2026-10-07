#!/usr/bin/env python3
"""Distanza latente del modello congiunto BFM+ICT su NoW: ricostruzione-scansione e fra
ricostruzioni.

    WBES_CKPT=<congiunto> aau/run.sh aau/recon/now_latent.py

La catena e' quella di ``ws3b_latent.py`` (``build_v1_model`` ed ``encode`` importate tali e
quali): ``merge_run_args -> build_model -> GTReadyDataset -> forward_model(add_noise=False)``,
distanza = norma L2 fra i latenti.  Il congiunto e' addestrato ad AREA UNITARIA, quindi gli
operatori sono quelli di ``now_ops_latent.sbatch`` (``*_withops_areanorm``, k_eig 128).

Uscite:
  ``gt_latent_joint_<metodo>.csv``   (repo) una riga per ricostruzione, contro scan_face
  ``pairs/latent_joint_<metodo>.npz`` (fuori) D (n, n) fra le ricostruzioni del metodo
  ``embeddings/latent_joint_*.npz``   (fuori) cache dei latenti, per mesh
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import now_common as common  # noqa: E402
from ws3b_latent import build_v1_model, encode  # noqa: E402

METRIC = "latent_joint"
ITEM_FIELDS = ("name", "subject", "challenge")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", type=str, default=",".join(common.METHODS))
    p.add_argument("--checkpoint", type=str, default=os.environ.get("WBES_CKPT", ""))
    p.add_argument("--config-json", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def pairwise_l2(Z: np.ndarray) -> np.ndarray:
    sq = (Z ** 2).sum(1)
    return np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T, 0.0, None))


def main() -> None:
    import torch

    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint non trovato: {checkpoint} (serve --checkpoint o WBES_CKPT)")
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    # encode() di ws3b_latent legge la cache in <out_root>/embeddings/<metric>_<tag>.npz
    cache = SimpleNamespace(out_root=common.WORK_ROOT, metric=METRIC, overwrite=args.overwrite)
    print(f"[now-latent] device={device} ckpt={checkpoint}", flush=True)

    items = common.load_items()
    model = build_v1_model(checkpoint, args.config_json, device)
    subjects = common.subjects_of(items)
    scans = encode(model, "scan_face", common.scan_face_ops_dir(), subjects, device, cache)

    for method in methods:
        present = {p.stem for p in common.recon_face_ops_dir(method).glob("*.npz")}
        mine = [it for it in items if it.name in present]
        if len(mine) != len(items):
            print(f"[now-latent] ATTENZIONE {method}: {len(mine)}/{len(items)} con operatori", flush=True)
        z = encode(model, method, common.recon_face_ops_dir(method), [it.name for it in mine], device, cache)
        Z = np.stack([z[it.name] for it in mine]).astype(np.float64)

        S = np.stack([scans[it.subject] for it in mine]).astype(np.float64)
        values = np.linalg.norm(Z - S, axis=1)
        rows = [{**{k: getattr(it, k) for k in ITEM_FIELDS}, METRIC: float(v)} for it, v in zip(mine, values)]
        common.write_rows(common.gt_csv_path(METRIC, method), ITEM_FIELDS + (METRIC,), rows,
                          method=method, checkpoint=str(checkpoint), latent_dim=int(Z.shape[1]),
                          ops="k_eig 128, area unitaria", recon_dir=str(common.recon_face_ops_dir(method)))

        path = common.pair_matrix_path(METRIC, method)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, D=pairwise_l2(Z), names=np.asarray([it.name for it in mine]))
        print(f"[now-latent] {method}: {len(rows)} ricostruzioni, distanza dalla scansione mediana "
              f"{np.median(values):.4f}; matrice {len(mine)}x{len(mine)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
