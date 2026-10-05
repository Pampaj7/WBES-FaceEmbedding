#!/usr/bin/env python3
"""Prepara le 3 immagini frontali su cui si prova la pipeline di ricostruzione (WS3b).

  aau/recon/run_recon.sh ddfa aau/recon/make_test_images.py --mode samples
  aau/recon/run_recon.sh ddfa aau/recon/make_test_images.py --mode remesh

``samples`` (default) copia tre foto gia' incluse nei cloni: sono ritratti reali con
sfondo, cioe' l'input per cui i tre metodi sono stati addestrati, e servono a provare
anche il detector.  ``remesh`` renderizza invece tre identita' REMESH con il renderer
del repo (``v2_work/phase0/render_mesh.py``): utile per vedere come reagiscono a una
faccia sintetica grigia senza texture, che e' un input fuori distribuzione per tutti e
tre e su cui FaceBoxes puo' benissimo non trovare niente.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

# Tre ritratti a volto singolo presi dai cloni, con un nome esplicito sulla provenienza.
SAMPLE_IMAGES = (
    ("sample0_emma.jpg", common.EXTERNAL / "3DDFA_V2" / "examples" / "inputs" / "emma.jpg"),
    ("sample1_jianzhuguo.jpg",
     common.EXTERNAL / "3DDFA_V2" / "examples" / "inputs" / "JianzhuGuo.jpg"),
    ("sample2_prnet0.jpg", common.EXTERNAL / "PRNet" / "TestImages" / "0.jpg"),
)
REMESH_SUBJECTS = ("id0000", "id0001", "id0002")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", type=str, default="samples", choices=("samples", "remesh"))
    p.add_argument("--out", type=Path, default=common.OUT_ROOT / "test_images")
    p.add_argument("--size", type=int, default=512, help="solo per --mode remesh")
    return p.parse_args()


def from_samples(out: Path) -> list[Path]:
    written = []
    for name, src in SAMPLE_IMAGES:
        if not src.is_file():
            raise SystemExit(f"ERRORE: immagine campione mancante: {src}")
        dst = out / name
        shutil.copyfile(src, dst)
        written.append(dst)
    return written


def from_remesh(out: Path, size: int) -> list[Path]:
    from PIL import Image

    sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "phase0"))
    from render_mesh import render_npz  # noqa: E402

    mesh_dir = common.REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500"
    written = []
    for sid in REMESH_SUBJECTS:
        src = mesh_dir / f"{sid}_GTready_original.npz"
        if not src.is_file():
            raise SystemExit(f"ERRORE: mesh REMESH mancante: {src}")
        img = render_npz(str(src), size=size)
        dst = out / f"remesh_{sid}.png"
        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(dst)
        written.append(dst)
    return written


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    written = from_samples(out) if args.mode == "samples" else from_remesh(out, args.size)
    for p in written:
        print(f"[test-images] {p} ({p.stat().st_size / 1024:.0f} KB)")
    print(f"[test-images] {len(written)} immagini in {out}")


if __name__ == "__main__":
    main()
