#!/usr/bin/env python3
"""Aggiunge in CODA a checking_assumptions/experiment_registry.csv la riga della variante F (BOARD_DIARY,
"Ipotesi H6: varieta' del crop in training").  Solo stdlib: gira sul frontend.  Rifiuta experiment_id
gia' presenti.

    python3 aau/models/ablation_F_registry_append.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
from rev2_registry_append import append  # noqa: E402

ROWS = [
    {
        "experiment_id": "ablation_F_random_crops_s1234",
        "branch": "both",
        "status": "running",
        "question": "Hypothesis H6: does training with varied random crops (5 per training subject, 60-90% of the "
                    "original's vertices, random lower/lateral cut plane, irregular border, optional second lateral "
                    "cut, eyes and nose tip always kept) improve crop pairs without losing elsewhere? Criterion fixed "
                    "beforehand (BOARD_DIARY): >= +0.05 AUC (b_vs_c, mean of tracked->crop, remesh->crop, crop->crop) "
                    "on Multiface WS3a hard, or >= +0.05 Spearman on the REMESH crop group (eval_by_topology.py), "
                    "without losing more than 0.02 on REMESH noisy/resample or Multiface tracked->tracked, "
                    "tracked->noisy, down->up. Paired control remesh_v1recipe_current_s1234_1055026.",
        "script_or_source": "crops aau/models/random_crops_F.py + aau/models/view_F.sbatch (gen 1055158, standard "
                            "operators k_eig 128 with precompute_operators_npz.py array 1055159, check + view 1055160); "
                            "smoke aau/models/ablation_F_smoke.sbatch 1055161; training aau/train_remesh_frame.sbatch "
                            "(v1 recipe, train_fast.py RAM cache, frame current, seed 1234, 120 epochs, L40S, "
                            "--mem=180G) with WBES_DATA_DIR=datasets/REMESH/view_F, job 1055162; evals "
                            "aau/eval_frame_topology.sbatch 1055167 (standard data dir), WS3a hard latent "
                            "(aau/multiface/ws3a_latent.sbatch, standard ops) F 1055168 and control 1055169, analysis "
                            "1055170 (summary_hard_F), table aau/models/ablation_F_summary.py 1055171",
        "output_path": "datasets/REMESH/npz_data_topo_500_cropF{,_withops}; datasets/REMESH/view_F; "
                       "aau/runs/abl_F_s1234_1055162; aau/runs/eval_frame/abl_F_s1234_1055162_maxabs; "
                       "aau/runs/multiface_ws3a_hard/{latent_abl_F,latent_ctrl_s1234}_*.csv, summary_hard_F.md; "
                       "aau/runs/ablation_F/{crops_stats.md,crops_meta.csv,summary.md}",
        "notes": "Crops only for the 400 training subjects of the seed-1234 split (100 held-out untouched; the "
                 "control's 16 online-eval subjects verified held-out). Files id<NNNN>_GTready_crop_r1..r5: the "
                 "trainer's infer_topology_label_from_name labels them 'crop', so they enter as variants of the "
                 "canonical crop with no wrapper: per subject and step still 6 meshes (one per label), the crop slot "
                 "drawn uniformly among canonical + 5 random (canonical 1/6 as often as in the control). Online and "
                 "final evals on the 6 standard topologies (held-out subjects have no extra files; final evals read "
                 "the standard data dir). Kept fraction: mean 0.752, min 0.598, median 0.755, max 0.900, roughly "
                 "uniform over 0.60-0.90 (318-374 crops per 0.05 bin); 50.3% with two cuts; first cut |phi| from chin "
                 "<30 deg 1041, 30-60 700, 60-90 251, 90-110 8 (eye protection makes deep lateral cuts infeasible: "
                 "protect radius 0.15 x outer-eye distance around nose tip and the 4 eye corners, BFM p23470 "
                 "landmarks). 0 empty crops, 14044-21120 vertices, largest edge-connected component, no isolated "
                 "vertices.",
    },
]

if __name__ == "__main__":
    append(ROWS)
