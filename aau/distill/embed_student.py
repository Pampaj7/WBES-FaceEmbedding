#!/usr/bin/env python3
"""Embedding dello studente distillato su ogni mesh di una data dir con operatori (zs_stage + areanorm_operators).

    aau/run.sh aau/distill/embed_student.py --ckpt <run>/checkpoints/last.pth --data-dir /tmp/.../ops \\
        --out <stage>/embeddings.npz
    (eval_distill.sbatch)

Scrive ``embeddings.npz`` nel formato di ``aau/zs3dmm/zs_embed.py`` (``Z`` (n, 512) float32,
``subjects``, ``topologies``, ``files``, ``checkpoint``), cosi' che ``zs_expr_summarize.model_distances``
lo legga senza adattatori. ``Z`` e' normalizzato L2: la distanza euclidea che ne calcola il
summarizer e' monotona nel coseno (||a - b||^2 = 2 - 2 cos), quindi rank-1, mAP e AUC coincidono con
quelli di 1 - coseno. Loader congelato (``GTReadyDatasetNPZ``), nessun rumore, ``model.eval()``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import train_distill as td  # noqa: E402  (mette sul path robustness e il loader)

from dataset_gtready import GTReadyDatasetNPZ  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    bundle = torch.load(a.ckpt, map_location="cpu")
    model = td.build_model(argparse.Namespace(**bundle["args"]), device)
    model.load_state_dict(bundle["state_dict"], strict=True)
    dataset = GTReadyDatasetNPZ(str(a.data_dir))
    t0 = time.time()
    Z = td.embed(model, dataset, range(len(dataset)), device).cpu().numpy().astype(np.float32)
    names = [Path(f).stem for f in dataset.files]
    parsed = [td.NAME_RE.match(n) for n in names]
    if any(m is None for m in parsed):
        raise SystemExit(f"nomi inattesi in {a.data_dir}: {[n for n, m in zip(names, parsed) if m is None][:5]}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.stem + ".tmp.npz")
    np.savez(tmp, Z=Z, subjects=np.asarray([m.group(1) for m in parsed], dtype="U16"),
             topologies=np.asarray([m.group(2) for m in parsed], dtype="U16"),
             files=np.asarray([str(a.data_dir / f) for f in dataset.files]), checkpoint=str(a.ckpt))
    tmp.replace(a.out)
    dt = time.time() - t0
    print(f"[embed-student] {Z.shape} da {a.ckpt} (epoca {bundle.get('epoch')}) in {dt:.0f}s "
          f"({dt / len(Z) * 1e3:.0f} ms/mesh, caricamento compreso) -> {a.out}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
