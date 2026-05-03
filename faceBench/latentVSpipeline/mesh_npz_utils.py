from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def normalize_vertices(V: np.ndarray) -> np.ndarray:
    """Center mesh vertices and scale them to the unit max-abs box.

    This mirrors GTReadyDatasetNPZ in face_embedding/gt_encdec/autoencoder/dataset_gtready.py.
    """
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"Expected (n, 3) vertices, got shape {V.shape}")
    V = V - V.mean(axis=0, keepdims=True)
    scale = float(np.max(np.abs(V)))
    if scale > 1e-6:
        V = V / scale
    else:
        V = V * 0.0
    return V


def load_normalized_vertices_npz(path: Path, scale: float = 1.0) -> np.ndarray:
    """Load vertices from an NPZ and apply the original per-mesh normalization."""
    data = np.load(path, allow_pickle=True)
    try:
        for key in ("V", "verts", "vertices"):
            if key in data.files:
                V = np.asarray(data[key], dtype=np.float64)
                if V.ndim != 2 or V.shape[1] != 3:
                    raise ValueError(f"Bad shape {V.shape} in {path}")
                V = V * scale
                return normalize_vertices(V)
        raise KeyError(f"No vertex key in {path}; keys={data.files}")
    finally:
        try:
            if hasattr(data, "close"):
                data.close()
        except Exception:
            pass


def load_withops_sample_npz(npz_path: Path, device) -> Optional[Tuple[object, dict]]:
    """Load normalized verts and operator tensors from a withops NPZ."""
    if not npz_path.exists():
        return None

    import torch

    data = np.load(npz_path, allow_pickle=True)
    try:
        if "verts" not in data.files or "faces" not in data.files:
            raise RuntimeError(f"Missing verts/faces in {npz_path}")

        verts = torch.tensor(normalize_vertices(np.asarray(data["verts"])), dtype=torch.float32).to(device)
        sample_dict = {
            "faces": torch.tensor(data["faces"], dtype=torch.long).to(device),
            "mass": torch.tensor(data["mass"], dtype=torch.float32).to(device),
            "L": _load_coo_from_npz(data, "L", device),
            "evals": torch.tensor(data["evals"], dtype=torch.float32).to(device),
            "evecs": torch.tensor(data["evecs"], dtype=torch.float32).to(device),
            "gradX": _load_coo_from_npz(data, "gradX", device),
            "gradY": _load_coo_from_npz(data, "gradY", device),
        }
        return verts, sample_dict
    finally:
        try:
            if hasattr(data, "close"):
                data.close()
        except Exception:
            pass


def _load_coo_from_npz(data, key: str, device):
    import torch

    indices = torch.tensor(data[f"{key}_indices"], dtype=torch.long)
    values = torch.tensor(data[f"{key}_values"], dtype=torch.float32)
    shape = tuple(int(x) for x in data[f"{key}_shape"])
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(device)
