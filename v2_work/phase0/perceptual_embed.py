"""Perceptual 2D embeddings for rendered face meshes.

Extractors (all CPU-capable):
  - arcface : insightface buffalo_l recognition model (512-d, identity-specific)
  - clip    : open_clip ViT-B-32 laion2b image encoder (512-d)
  - dinov2  : torch.hub facebookresearch/dinov2 ViT-S/14 (384-d)

Each extractor maps a HxWx3 uint8 render -> 1D float32 numpy embedding.
Distances between meshes are cosine distances between embeddings
(1 - cos_sim), computed downstream.

LPIPS is pairwise on images and lives in the driver, not here.

Model files are downloaded on first use to the default caches
(~/.insightface, ~/.cache). Run once interactively to warm caches.
"""
from __future__ import annotations

import numpy as np


class ArcFaceExtractor:
    """insightface recognition embedding. Uses the detector when it fires,
    falls back to a center crop of the render otherwise (synthetic shaded
    renders sometimes miss the detector's training distribution)."""

    def __init__(self):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=-1, det_size=(512, 512))
        # direct handle to the recognition model for the no-detection fallback
        self.rec_model = self.app.models["recognition"]
        self.n_fallback = 0

    def __call__(self, img: np.ndarray) -> np.ndarray:
        bgr = img[:, :, ::-1]  # insightface expects BGR
        faces = self.app.get(bgr)
        if faces:
            return np.asarray(faces[0].normed_embedding, dtype=np.float32)
        # fallback: center square crop -> 112x112 -> recognition model directly
        self.n_fallback += 1
        h, w = bgr.shape[:2]
        s = min(h, w)
        crop = bgr[(h - s) // 2 : (h + s) // 2, (w - s) // 2 : (w + s) // 2]
        import cv2

        crop = cv2.resize(crop, (112, 112))
        emb = self.rec_model.get_feat(crop).flatten().astype(np.float32)
        n = np.linalg.norm(emb)
        return emb / max(n, 1e-9)


class ClipExtractor:
    def __init__(self, device: str = "cpu"):
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
        )
        self.model.eval()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image

        with self.torch.no_grad():
            t = self.preprocess(Image.fromarray(img)).unsqueeze(0).to(self.device)
            emb = self.model.encode_image(t).squeeze(0).cpu().numpy().astype(np.float32)
        return emb / max(np.linalg.norm(emb), 1e-9)


class Dinov2Extractor:
    def __init__(self, device: str = "cpu"):
        import torch

        self.torch = torch
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
        self.model.eval()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        import torch

        # resize to multiple of 14 (224), imagenet normalize
        from PIL import Image

        pil = Image.fromarray(img).resize((224, 224))
        x = torch.from_numpy(np.asarray(pil)).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = ((x - mean) / std).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(x).squeeze(0).cpu().numpy().astype(np.float32)
        return emb / max(np.linalg.norm(emb), 1e-9)


EXTRACTORS = {
    "arcface": ArcFaceExtractor,
    "clip": ClipExtractor,
    "dinov2": Dinov2Extractor,
}


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))
