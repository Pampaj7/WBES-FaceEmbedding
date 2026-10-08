#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-eq1
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=34G
#SBATCH --time=01:00:00
#
# Equivalenza v2 = v3, livello 1 (stesso batch): v3_work/trainer/tests/test_equivalence.py.
# Dati: aau/runs/evidence/trainer_v3/testdata (make_testdata.py), pre-pass su /tmp (conta contro --mem).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
STAGE="/tmp/${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$STAGE/eq" "$STAGE/eq32"' EXIT
mkdir -p "$STAGE"
EV="$AAU_RUNS/evidence/trainer_v3"
OUT="$EV${WBES_V3_SUITE_TAG:+/$WBES_V3_SUITE_TAG}"
mkdir -p "$OUT"
CKPT="$AAU_RUNS/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411/mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__3167d36d/checkpoints/epoch108.pth"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$AAU_DIR/run.sh" v3_work/trainer/tests/test_equivalence.py \
    --testdata "$EV/testdata" --stage-root "$STAGE/eq" --ckpt "$CKPT" --out "$OUT/equivalence_l1.json"
# packed contro sequenziale in fp32 VERO (il container tiene il TF32 acceso con un override d'ambiente)
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$AAU_DIR/run.sh" v3_work/trainer/tests/test_equivalence.py \
    --testdata "$EV/testdata" --stage-root "$STAGE/eq32" --ckpt "$CKPT" --out "$OUT/equivalence_packed_fp32.json" \
    --only-packed --fp32
