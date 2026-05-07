from typing import Optional

import warnings
import numpy as np

_emitted_warnings = set()


def warn_once(key: str, message: str):
    if key not in _emitted_warnings:
        _emitted_warnings.add(key)
        warnings.warn(message, stacklevel=2)


def warn_if_icp_no_prealign(prealign: Optional[str]):
    if prealign is None or prealign.lower() == "none":
        warn_once("icp_no_prealign",
                  "⚠️ ICP is being used without any pre-alignment. This may result in poor convergence."
                  )


def warn_if_scale_mismatch(R: np.ndarray, G: np.ndarray):
    norm_r, norm_g = np.linalg.norm(R), np.linalg.norm(G)
    ratio = norm_r / norm_g if norm_g != 0 else 0
    if ratio < 0.01 or ratio > 100:
        msg = (f"⚠️ Possible scale mismatch between R and G (norm ratio: {ratio:.2e}). "
               "Check mesh units.")
        warn_once("scale_mismatch", msg)


def warn_if_shape_mismatch(R: np.ndarray, G: np.ndarray):
    if R.ndim != 2 or G.ndim != 2 or R.shape[1] != 3 or G.shape[1] != 3:
        msg = f"⚠️ Suspicious input shape: R={R.shape}, G={G.shape}. Expected (N, 3)."
        warn_once("shape_mismatch", msg)


def warn_if_landmarks_invalid(Rlmks: np.ndarray, Glmks: np.ndarray):
    if Rlmks.shape != Glmks.shape:
        msg = (f"⚠️ Landmark mismatch: R landmarks shape={Rlmks.shape}, "
               f"G landmarks shape={Glmks.shape}. They must match.")
        warn_once("landmark_mismatch", msg)
    elif Rlmks.ndim != 2 or Rlmks.shape[1] != 3:
        msg = f"⚠️ Invalid landmark shape: Rlmks.shape={Rlmks.shape}. Expected (N, 3)."
        warn_once("landmark_shape", msg)
