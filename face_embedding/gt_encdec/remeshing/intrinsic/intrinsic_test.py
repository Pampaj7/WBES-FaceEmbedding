#!/usr/bin/env python3
import os
import re
import math
import numpy as np
import torch

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500_withops"
DEVICE = "cuda"
N_SAMPLES = 5

N_HKS = 16
N_WKS = 16
K_USE = 50

T_MIN = 1e-6
T_MAX = 1e-2

# ============================================================

import sys
sys.path.append("/equilibrium/lpampaloni/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder")
from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset


def compute_hks(evals, evecs):
    evals_k = evals[:K_USE]
    evecs_k = evecs[:, :K_USE]

    t = torch.logspace(
        math.log10(T_MIN),
        math.log10(T_MAX),
        N_HKS,
        device=evals.device,
        dtype=evals.dtype,
    )

    exp_kt = torch.exp(-evals_k[:, None] * t[None, :])
    hks = (evecs_k ** 2) @ exp_kt
    return hks


def compute_wks(evals, evecs):
    evals_k = evals[:K_USE]
    evecs_k = evecs[:, :K_USE]

    log_ev = torch.log(evals_k.clamp_min(1e-12))

    e_min = log_ev[1]
    e_max = log_ev[-1]

    energies = torch.linspace(
        float(e_min.detach().cpu()),
        float(e_max.detach().cpu()),
        N_WKS,
        device=evals.device,
        dtype=evals.dtype,
    )

    sigma = (energies[1] - energies[0]).abs()

    diff = log_ev[:, None] - energies[None, :]
    kernel = torch.exp(-(diff ** 2) / (2 * sigma ** 2 + 1e-12))
    kernel = kernel / (kernel.sum(dim=0, keepdim=True) + 1e-8)

    wks = (evecs_k ** 2) @ kernel
    return wks


def mass_weighted_standardize(x, mass, eps=1e-8):
    """
    Mass-weighted z-score per channel.
    Keeps operator consistency.
    """
    w = mass / mass.sum()

    mean = (w[:, None] * x).sum(dim=0)
    var = (w[:, None] * (x - mean[None, :]) ** 2).sum(dim=0)
    std = torch.sqrt(var + eps)

    x_norm = (x - mean[None, :]) / std[None, :]
    return x_norm, mean, std


def main():

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ds = GTReadyDataset(DATA_DIR)

    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds.files),
                      size=min(N_SAMPLES, len(ds.files)),
                      replace=False)

    for idx in idxs:

        sample = ds[int(idx)]
        fname = ds.files[int(idx)]

        print("\n===================================================")
        print("File:", fname)

        mass = sample["mass"].to(device)
        evals = sample["evals"].to(device)
        evecs = sample["evecs"].to(device)

        print("   evals min/max:",
              evals.min().item(),
              evals.max().item())

        # ------------------------------------------------
        # Operator sanity check (no touching mass!)
        # ------------------------------------------------
        Phi = evecs[:, :K_USE]
        M = torch.diag(mass)

        I_test = Phi.T @ M @ Phi
        orth_err = torch.norm(
            I_test - torch.eye(I_test.size(0), device=device)
        ).item()

        print("   orth_err:", orth_err)

        # ------------------------------------------------
        # HKS / WKS
        # ------------------------------------------------
        hks = compute_hks(evals, evecs)
        wks = compute_wks(evals, evecs)

        print("   raw HKS std:", hks.std().item())
        print("   raw WKS std:", wks.std().item())

        # ------------------------------------------------
        # Standardization
        # ------------------------------------------------
        x = torch.cat([hks, wks], dim=1)

        x_norm, mean, std = mass_weighted_standardize(x, mass)

        print("   normalized std (should be ~1):",
              x_norm.std().item())

        print("   min/max after norm:",
              x_norm.min().item(),
              x_norm.max().item())

        print("===================================================\n")


if __name__ == "__main__":
    main()