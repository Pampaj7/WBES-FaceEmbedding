#!/usr/bin/env python3
"""CLIP ViT-L/14 e DINOv2 ViT-B/14 sui render di sola geometria della pipeline ArcFace.

    aau/baselines/run_bl.sh aau/baselines_extra/vfm_embed.py \\
        --render-root aau/runs/arcface_render_zs/fv_expr/normals --out-dir aau/runs/baselines_extra/fv_expr
    (be_embed.sbatch)

Protocollo: ``aau/runs/baselines_extra/protocol.md``. I render NON si rifanno: si leggono quelli di
``zs_arcface_render.py`` (``<render-root>/renders/<topologia>/``, nomi di ``ws3a_render.render_path``),
con le mesh e l'ordine di ``<render-root>/arcface_views.npz``, cosi' le righe coincidono con quelle di
ArcFace e il summarizer le legge con la stessa funzione (``zs_arcface_summarize.arcface_distances``).

Due ritagli:

1. ``crop``: la similarita' 2x3 congelata di ``arcface_align.json`` (per yaw) moltiplicata per 2, cioe'
   lo stesso riquadro del volto che ArcFace vede a 112x112, a 224x224. Bordo nero come nei png di
   controllo di zs_arcface_render.py. Nessun resize dopo.
2. ``full``: il render intero 512 -> 224 bicubico, come CLIP/DINOv2 in WS1 (ablazione).

Modelli (pesi locali, scaricati dal frontend; sha256 nel protocollo):
- ``clip_l14``: open_clip ``ViT-L-14``, pesi OpenAI da ``--clip-weights``; ``encode_image``, mean/std di CLIP;
- ``dinov2_b14``: ``dinov2_vitb14`` dal clone di torch hub in cache (``source="local"``, nessuna rete),
  pesi da ``~/.cache/torch/hub/checkpoints``; token CLS (uscita di ``forward``), mean/std ImageNet.

Scrive ``<out-dir>/<modello>_<ritaglio>_views.npz``: ``E`` (n_mesh, n_yaw, d) float32 L2 per vista,
``subjects``, ``topologies``, ``yaws`` (stesso formato di ``arcface_views.npz``), e in
``<out-dir>/control/`` i ritagli 224 dei primi 2 soggetti.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
sys.path.insert(0, str(AAU_DIR / "multiface"))

import ws3a_render as render  # noqa: E402

MODELS = ("clip_l14", "dinov2_b14")
CROPS = ("crop", "full")
SIZE = 224
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--render-root", type=Path, required=True, help="<dominio>/normals di zs_arcface_render.py")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", type=str, default=",".join(MODELS))
    p.add_argument("--crops", type=str, default=",".join(CROPS))
    p.add_argument("--clip-weights", type=Path,
                   default=Path.home() / ".cache/wbes_weights/clip_vitl14_openai/open_clip_model.safetensors")
    p.add_argument("--hub-dir", type=Path, default=Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_model(name: str, args):
    """(modello, mean, std, funzione immagini normalizzate -> embedding)."""
    import torch

    if name == "clip_l14":
        import open_clip

        # I pesi OpenAI vogliono QuickGELU: col tag "openai" open_clip lo mette da solo, da file locale no.
        model = open_clip.create_model("ViT-L-14", pretrained=str(args.clip_weights), device=args.device,
                                       force_quick_gelu=True).eval()
        return model, CLIP_MEAN, CLIP_STD, model.encode_image
    if name == "dinov2_b14":
        model = torch.hub.load(str(args.hub_dir), "dinov2_vitb14", source="local").to(args.device).eval()
        return model, IMAGENET_MEAN, IMAGENET_STD, model
    raise ValueError(name)


def crop_224(img: np.ndarray, M112: np.ndarray | None) -> np.ndarray:
    import cv2

    if M112 is None:
        return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    return cv2.warpAffine(img, 2.0 * M112, (SIZE, SIZE), borderValue=0.0)


def main() -> None:
    import torch
    from PIL import Image

    args = parse_args()
    ref = np.load(args.render_root / "arcface_views.npz")
    subjects, topologies = [str(s) for s in ref["subjects"]], [str(t) for t in ref["topologies"]]
    yaws = [float(y) for y in ref["yaws"]]
    cal = json.loads((args.render_root / "renders" / "arcface_align.json").read_text())
    transforms = {float(v["yaw"]): np.asarray(v["transform"], np.float64) for v in cal["views"].values()}
    print(f"[vfm] {args.render_root}: {len(subjects)} mesh x {len(yaws)} viste, yaw {yaws}, device {args.device}",
          flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "control").mkdir(exist_ok=True)

    crops = [c for c in args.crops.split(",") if c]
    # Immagini 224 in RAM una volta sola per ritaglio: 1800 x 224 x 224 x 3 = 270 MB.
    t0 = time.time()
    imgs = {c: np.zeros((len(subjects), len(yaws), SIZE, SIZE, 3), np.uint8) for c in crops}
    for i, (s, t) in enumerate(zip(subjects, topologies)):
        for k, y in enumerate(yaws):
            img = np.asarray(Image.open(render.render_path(args.render_root, t, s, y)).convert("RGB"))
            for c in crops:
                imgs[c][i, k] = crop_224(img, transforms[y] if c == "crop" else None)
    print(f"[vfm] {len(subjects) * len(yaws)} render letti e ritagliati in {time.time() - t0:.0f}s", flush=True)
    first = sorted(set(subjects))[:2]
    for c in crops:
        rows = [np.concatenate(list(imgs[c][i]), axis=1) for i, s in enumerate(subjects) if s in first]
        Image.fromarray(np.concatenate(rows, axis=0)).save(args.out_dir / "control" / f"{c}224_rows-mesh_cols-yaws.png")

    for name in [m for m in args.models.split(",") if m]:
        todo = [c for c in crops if args.overwrite or not (args.out_dir / f"{name}_{c}_views.npz").exists()]
        if not todo:
            print(f"[vfm] {name}: gia' fatto", flush=True)
            continue
        model, mean, std, fn = load_model(name, args)
        mean_t = torch.tensor(mean, device=args.device).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=args.device).view(1, 3, 1, 1)
        for c in todo:
            t1 = time.time()
            flat = imgs[c].reshape(-1, SIZE, SIZE, 3)
            out = []
            with torch.no_grad():
                for b in range(0, len(flat), args.batch):
                    x = torch.from_numpy(flat[b:b + args.batch]).to(args.device).permute(0, 3, 1, 2).float() / 255.0
                    out.append(fn((x - mean_t) / std_t).float().cpu().numpy())
            E = np.concatenate(out).reshape(len(subjects), len(yaws), -1).astype(np.float64)
            E /= np.maximum(np.linalg.norm(E, axis=2, keepdims=True), 1e-9)
            if not np.isfinite(E).all():
                raise SystemExit(f"{name} {c}: embedding non finiti")
            path = args.out_dir / f"{name}_{c}_views.npz"
            np.savez(path, E=E.astype(np.float32), subjects=np.asarray(subjects), topologies=np.asarray(topologies),
                     yaws=np.asarray(yaws), model=name, crop=c, render_root=str(args.render_root))
            print(f"[vfm] {name} {c}: {E.shape} in {time.time() - t1:.0f}s -> {path}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
