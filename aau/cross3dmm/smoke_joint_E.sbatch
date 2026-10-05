#!/usr/bin/env bash
#SBATCH --job-name=wbes-jointE-smoke
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#
# Smoke di aau/cross3dmm/train_joint_E.py, entrambi i bracci, 1 epoca su una vista minima in /tmp:
# 15 soggetti BFM + 15 ICT del sottoinsieme, scelti fra gli ICT che hanno gia' gli operatori robusti
# (id14500-id14999), cosi' il braccio e non deve calcolarne. Controlla che partano cache, frame, token
# per dominio e batch a dominio singolo; i numeri non contano.
#
#   sbatch ... aau/cross3dmm/smoke_joint_E.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net

STAGE="/tmp/${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE"
SUBS=$("$AAU_DIR/run.sh" -c "
import json
s = json.load(open('aau/runs/joint_E/splits.json'))['subjects']
ict = [x for x in s['ict'] if int(x[2:]) >= 14500][:15]
print(' '.join(s['bfm'][:15] + ict))")
echo "[smoke] soggetti: $SUBS"
D="$WBES_ROOT/datasets"
for arm in ctrl e; do
    mkdir -p "$STAGE/$arm"
    for sid in $SUBS; do
        for t in crop down8k noisy original remesh up60k; do
            n="${sid}_GTready_${t}.npz"
            if [[ "$arm" == ctrl ]]; then
                src="$D/JOINT_E/ctrl/npz_withops/$n"
            elif (( 10#${sid#id} >= 10000 )); then
                src="$D/ICT/eval_view_heldout_robust_area1/$n"
            else
                src="$D/REMESH/npz_data_topo_500_withops_robust_area1/$n"
            fi
            ln -s "$(realpath "$src")" "$STAGE/$arm/$n"
        done
    done
done

for arm in ctrl e; do
    ARGS=(--frame current)
    if [[ "$arm" == e ]]; then
        ARGS=(--frame rms
              --size-token-json-bfm "$AAU_DIR/runs/joint_E/size_token_joint_bfm_s1234.json"
              --size-token-json-ict "$AAU_DIR/runs/joint_E/size_token_joint_ict_s1234.json")
    fi
    echo "[smoke] ===== $arm ====="
    "$AAU_DIR/run.sh" aau/cross3dmm/train_joint_E.py --eval_domain bfm "${ARGS[@]}" \
      --cache-residency ram --cache-workers 8 \
      --data_dir "$STAGE/$arm" --dist_npz "$D/JOINT_BFM_ICT/gt_matrix.npz" \
      --runs_root "$STAGE/runs_$arm" --device cuda --model xyz_dn \
      --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1 \
      --pool_mode meanmax --k_spec 0 --no-log_spec --eig_k 300 \
      --epochs 2 --batch_subjects 5 --lr 1e-4 --weight_decay 1e-6 --grad_clip 1.0 \
      --train_level mixed --train_pair_mode cross_topology \
      --lambda_subject 1.0 --lambda_mesh 1.0 --lambda_rank 0.5 --use_id_loss --lambda_id 0.25 \
      --rank_margin 0.05 --rank_pairs 1024 --rank_tau 0.02 --rank_hard_frac 0.7 \
      --max_meshes_per_subject_train 6 --max_meshes_per_subject_eval 6 --max_subjects_eval_train 16 \
      --preload_eval_workers_train 2 --p_noise 0.6 --sigma_min 5e-4 --sigma_max 2e-2 \
      --noise_modes translation,rotation,jitter --noise_mode_weights "translation=4,rotation=2,jitter=1" \
      --rigid_rot_deg 12.0 --rigid_rot_deg_min 0.5 --rigid_trans_scale 0.03 --rigid_trans_scale_min 0.001 \
      --sigma_min_eval 1e-3 --sigma_max_eval 0.1 --n_sigma_eval 6 \
      --eval_mode average --eval_every 2 --save_every 5 --seed 1234 2>&1 | grep -v "it/s\|s/it"
done
echo "[smoke] OK"
