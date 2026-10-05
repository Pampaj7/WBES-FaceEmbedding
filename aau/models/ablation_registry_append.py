#!/usr/bin/env python3
"""Aggiunge in CODA a checking_assumptions/experiment_registry.csv le righe delle ablazioni v3
(B, C, E, BOARD_DIARY "Progetto della versione nuova e ablazioni").  Solo stdlib: gira sul
frontend.  Rifiuta experiment_id gia' presenti.

    python3 aau/models/ablation_registry_append.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
from rev2_registry_append import append  # noqa: E402

CRITERION = ("Criterion fixed beforehand (BOARD_DIARY): >= +0.03 on the latent - Chamfer margin averaged over "
             "the 30 ordered topology pairs, or >= +0.05 on crop pairs, without losing more than 0.02 on "
             "zero-shot ICT. Paired control remesh_v1recipe_current_s1234_1055026 (same wrapper, cache, seed 1234).")
COMMON = ("v1 recipe, 120 epochs, seed 1234, v2_work/fastio/train_fast.py with RAM cache via "
          "aau/models/train_ablation.py + aau/models/train_ablation.sbatch (A10, --mem=180G); hooks "
          "aau/models/ablation_hooks.py; evals aau/models/eval_ablation_topology.sbatch (eval_cells.py: author "
          "groups + 30 repo cells) and aau/models/ict_ablation_rank.sbatch (WBES_EVAL_SEED=1234, clean), queue "
          "aau/models/ablation_queue.sh, table aau/models/ablation_summary.py")

ROWS = [
    {
        "experiment_id": "ablation_v3_B_rms_frame_size_token_s1234",
        "branch": "both",
        "status": "running",
        "question": "Ablation B: does rms input frame + a size token (log rms radius of the RAW mesh, area-weighted, "
                    "standardized on the 400 BFM training subjects, concatenated after mean+max pooling before the "
                    "256-d projection) beat the paired control? " + CRITERION,
        "script_or_source": COMMON + "; size token table aau/models/size_token.py (BFM seed 1234; ICT standardized "
                            "on the 4500 ICT training subjects from datasets/ICT/topo raw cm); training job 1055113 (V100 nv-ai-03; A10 had no room for 180G; first launch 1055097 cancelled at epoch 2, see notes); "
                            "smoke 1055060 (+ token checks 1055067); evals 1055117/1055118, control 1055119/1055120, table 1055125",
        "output_path": "aau/runs/abl3_B_s1234_1055113; aau/runs/ablations_v3/summary.md; "
                       "aau/runs/ablations_v3/size_token_{bfm_s1234,ict}.json(.check.md)",
        "notes": "Token consistency across topologies (BFM, dlog r vs original, mean/std): remesh -0.0034/0.0005, "
                 "noisy -0.0058/0.0057, down8k +0.0001/0.0001, up60k +0.0014/0.0003, crop -0.0879/0.0064 "
                 "(radius x0.916, -2.48 token std; between-subject std of log r on originals 0.0156, training std "
                 "0.0354 incl. all topologies). ICT: crop -0.100, noisy -0.042, remesh -0.017. Token weight column "
                 "zero-initialized (rest of the init and torch RNG identical to the control): with default init the "
                 "first launch (1055097) had epoch-1 loss 0.266 vs 0.080 control, xtopo 0.11 at epoch 2; relaunched.",
    },
    {
        "experiment_id": "ablation_v3_C_robust_laplacian_unit_area_s1234",
        "branch": "both",
        "status": "running",
        "question": "Ablation C: do robust-Laplacian operators on unit-area meshes (robust_laplacian.mesh_laplacian, "
                    "default mollify, k_eig 128; tangent frames and gradients as compute_operators, only L and mass "
                    "replaced), standard input frame, beat the paired control? " + CRITERION,
        "script_or_source": COMMON + "; operators aau/models/robust_area1_operators.py + "
                            "aau/models/precompute_robust_area1.sbatch (1055041_0, 1055050 BFM; 1055051 ICT held-out), "
                            "full checks 1055053 (BFM) / 1055054 (ICT); training job 1055099 (V100 nv-ai-03); evals 1055121/1055122",
        "output_path": "datasets/REMESH/npz_data_topo_500_withops_robust_area1; "
                       "datasets/ICT/eval_view_heldout_robust_area1; aau/runs/abl3_C_s1234_1055099; "
                       "aau/runs/ablations_v3/summary.md",
        "notes": "Smoke subjects: area 1.000000, same keys as the standard npz, mass sums to 1, robust/cotan median "
                 "eigenvalue ratio at equal area 0.9995 (BFM) / 0.9987 (ICT). The frozen loader divides evals by "
                 "their max and gradients by its sqrt, which already removes a global scale: the unit-area rescaling "
                 "is expected to change the trained model very little, so C mostly tests robust vs cotan.",
    },
    {
        "experiment_id": "ablation_v3_E_rms_token_robust_unit_area_s1234",
        "branch": "both",
        "status": "running",
        "question": "Ablation E: B + C combined (rms frame + size token + robust unit-area operators). " + CRITERION,
        "script_or_source": COMMON + "; training job 1055114 (V100; first launch 1055100 cancelled at epoch 2); smoke 1055060; evals 1055123/1055124",
        "output_path": "aau/runs/abl3_E_s1234_1055114; aau/runs/ablations_v3/summary.md",
        "notes": "Launched together with B and C as requested (not conditional on B/C passing).",
    },
]

if __name__ == "__main__":
    append(ROWS)
