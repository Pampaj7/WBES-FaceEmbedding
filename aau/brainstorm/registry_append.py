#!/usr/bin/env python3
"""Aggiunge in CODA a checking_assumptions/experiment_registry.csv le righe del brainstorm
sugli operatori (H1, H3).  Solo stdlib: gira sul frontend.  Rifiuta experiment_id gia' presenti.

    python3 aau/brainstorm/registry_append.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
from rev2_registry_append import append  # noqa: E402

ROWS = [
    {
        "experiment_id": "brainstorm_h1_size_invisible",
        "branch": "both",
        "status": "completed",
        "question": "H1 (BOARD_DIARY brainstorm): does the absolute head size, which raw D_GT contains and per-mesh maxabs input removes, correlate with D_GT, and does the latent v1 residual (rank D_GT - rank latent) correlate with |dlog size|? Held-out BFM, 100 subjects, all 21 topology pairs. Prediction fixed beforehand: rho(D_GT,|dlog size|) > 0.3 and rho(resid,|dlog size|) > 0.1 on original->original for the same size measure.",
        "script_or_source": "aau/brainstorm/h1_size.py; latents for 6 topologies via aau/human_study/latent_matrix.py --out-root aau/runs/brainstorm/latent_v1 (job 1054992, T4 unprivileged, 5m26s); analysis job 1055005 (cpu, 15s)",
        "output_path": "aau/runs/brainstorm/h1.{md,csv,json}; aau/runs/brainstorm/latent_v1/; predictions aau/runs/brainstorm/PREDICTIONS.md",
        "notes": "NOT confirmed by the pre-set rule. rho(D_GT,|dlog size|): maxabs 0.332, sqrt-area 0.189, rms 0.235. original->original: rho(GT,latent) 0.830; partial given size maxabs 0.808 / sqrt-area 0.825 / rms 0.849; rho(resid,|dlog size|) maxabs -0.034 / sqrt-area 0.044 / rms 0.360; rho(latent,|dlog size|) maxabs 0.346 / sqrt-area 0.153 / rms 0.018. Maxabs size passes the first threshold but the latent already tracks it (shape proxy); rms size is the one the latent misses (resid 0.36, cross mean 0.31, crop pairs 0.28) but alone correlates only 0.235 with D_GT. Resid vs dlog maxabs rises on crop pairs (mean 0.111 vs 0.019 no-crop cross). Crop maxabs divisor / original: 0.869 +- 0.032 (0.810-0.947), std log 0.037, equal to the between-subject std of log maxabs (0.033); sqrt-area ratio std log 0.003. Recomputed original latents reproduce the published latent_v1 matrix exactly (max diff 0).",
    },
    {
        "experiment_id": "brainstorm_h3_operator_spectral_consistency",
        "branch": "both",
        "status": "completed",
        "question": "H3 (BOARD_DIARY brainstorm): does the robust Laplacian (Sharp & Crane 2020) make spectra and HKS of the same subject more consistent across the 6 topologies than DiffusionNet's cotan Laplacian? 50 held-out BFM subjects, k=64 non-trivial modes, variants a cotan, b robust, c cotan+unit area, d robust+unit area, e potential well 0.55 (v2_work/potential/potential_operators.py unmodified, global alpha 0.55, scale 127507). Prediction fixed beforehand: b reduces eigenvalue dispersion and HKS error vs a by >=20% on original-down8k and original-noisy.",
        "script_or_source": "aau/brainstorm/h3_spectral.py + h3_spectral.sbatch (array 1054997, 16 tasks cpu unprivileged, 5-8 min each, 34 min wall; subject id0001 from smoke job 1054995), aau/brainstorm/h3_table.py (job 1055033, 20s); HKS 8 times log-spaced in [4ln10/lambda64, 4ln10/lambda1], medians over originals, fixed per variant; 2000 sampled original vertices, nearest vertex in the shared frame, crop restricted to points within 3 median edges",
        "output_path": "aau/runs/brainstorm/h3.{md,csv,json}; per-subject eigenvalues and phi^2 at samples in aau/runs/brainstorm/h3_parts/",
        "notes": "NOT confirmed. (a->b) reduction on orig-down8k: dispersion +28.5% (0.0015->0.0011), HKS +7.8%; orig-noisy: dispersion -0.8%, HKS 0.0%. In raw coordinates noisy's area x2.3 dominates (HKS rel err ~2.0 saturated for a and b). At unit area (c->d): down8k disp +27.8% HKS +8.8%, noisy disp +16.3% HKS +22.7%; all-6 dispersion a 0.137 b 0.138 c 0.056 d 0.048 e 0.100; HKS all pairs c 0.0226 d 0.0197; separability all pairs a 5.93 b 6.15 c 2.35 d 2.04 e 0.98 (e within ROI 0.95). Well e: crop dispersion 0.0080 (a 0.072, c 0.011) but remesh 0.108 and down8k 0.006 (worse); HKS error large even inside ROI (0.11-0.83). Author statistic reproduced (first 30 modes, mean): a 0.227 vs 0.2202, c 0.0574 vs 0.0577 (different subject sample).",
    },
]

if __name__ == "__main__":
    append(ROWS)
