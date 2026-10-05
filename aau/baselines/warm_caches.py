#!/usr/bin/env python3
"""Scarica una volta sola i pesi dei quattro estrattori percettivi nelle cache di default.

Va lanciato da un nodo di calcolo (il frontend non ha python): il probe ha verificato che
i nodi vedono pypi/github/huggingface.  Le cache stanno in home (CephFS), quindi sono
condivise da tutti i job successivi, che possono girare anche senza rete.

  aau/baselines/run_bl.sh aau/baselines/warm_caches.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

DUMMY = np.full((512, 512, 3), 128, dtype=np.uint8)


def warm(name: str, fn) -> bool:
    print(f"--- {name} ---", flush=True)
    try:
        fn()
        print(f"[ok] {name}", flush=True)
        return True
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"[KO] {name}", flush=True)
        return False


def warm_arcface() -> None:
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(512, 512))
    emb = app.models["recognition"].get_feat(DUMMY[:112, :112, ::-1].copy()).flatten()
    print(f"    embedding arcface: {emb.shape}")


def warm_clip() -> None:
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"
    )
    print(f"    clip ok: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parametri")


def warm_dinov2() -> None:
    import torch

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    print(f"    dinov2 ok: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parametri")


def warm_lpips() -> None:
    import lpips

    model = lpips.LPIPS(net="alex")
    print(f"    lpips ok: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parametri")


def print_versions() -> None:
    import cv2
    import torch

    print(f"numpy={np.__version__} ({np.__file__})")
    print(f"cv2={cv2.__version__} ({cv2.__file__})")
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    # Il giro numpy<->torch e' il primo a rompersi se nel venv finisce numpy 2.x.
    print(f"torch->numpy: {torch.arange(3).numpy()}")


def main() -> None:
    print_versions()
    results = {
        "arcface (insightface buffalo_l)": warm("arcface", warm_arcface),
        "clip (open_clip ViT-B-32)": warm("clip", warm_clip),
        "dinov2 (torch.hub ViT-S/14)": warm("dinov2", warm_dinov2),
        "lpips (alexnet)": warm("lpips", warm_lpips),
    }
    print("--- riepilogo ---")
    for name, ok in results.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
