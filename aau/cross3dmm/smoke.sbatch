#!/usr/bin/env bash
#SBATCH --job-name=wbes-x3dmm-smoke
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili, e un
# --output relativo viene risolto nella submit dir. Sottometti con aau/submit.sh.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Prova a vuoto della tabella cross-3DMM prima di bruciare 48 h di L40S: vista congiunta
# minuscola (6 BFM + 6 ICT, 2 topologie), 2 epoche su CPU con modello ridotto, per ciascuno
# dei due regimi che poi girano davvero (ICT da solo e BFM+ICT).
# Quello che deve rompersi qui e non li': il parsing degli id ICT a 5 cifre nella GT, i NaN
# fuori blocco letti da un batch misto, l'eval online che spanna i domini.
#
#   aau/submit.sh cross3dmm/smoke.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net
export AAU_NV=""   # partizione cpu: --nv su un nodo senza GPU NVIDIA non ha senso

TMP_DIR="${WBES_X3DMM_SMOKE_DIR:-$WBES_ROOT/datasets/JOINT_BFM_ICT_smoke}"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

echo "[smoke] host=$(hostname) job=${SLURM_JOB_ID:-none} tmp=$TMP_DIR"

echo "--- 1. vista congiunta 10 BFM + 10 ICT, topologie original+remesh ---"
"$AAU_DIR/run.sh" v2_work/train_v2/make_joint_view.py \
    --domains bfm,ict --n-bfm 10 --n-ict 10 --variants original,remesh --support-bank "" \
    --data-dir "bfm=$WBES_ROOT/datasets/REMESH/npz_data_topo_500_withops_areanorm" \
    --out-dir "$TMP_DIR/joint"

echo "--- 2. vista ICT da sola, stesse 10 identita' ---"
"$AAU_DIR/run.sh" v2_work/train_v2/make_joint_view.py \
    --domains ict --n-ict 10 --variants original,remesh --support-bank "" \
    --out-dir "$TMP_DIR/ict"

# Modello ridotto e 2 epoche: qui interessa che il giro si chiuda, non che impari.
smoke_train() {
    echo "--- 3.$1 due epoche su $2 (eval_domain='$3') ---"
    "$AAU_DIR/run.sh" v2_work/train_v2/train_v2.py \
        --eval_domain "$3" \
        --data_dir "$2/npz_withops" --dist_npz "$2/gt_matrix.npz" \
        --runs_root "$TMP_DIR/runs_$1" --device cpu \
        --model xyz_dn --k_spec 0 --no-log_spec \
        --latent_dim 64 --width 32 --n_blocks 2 --eig_k 64 \
        --epochs 2 --batch_subjects 2 --max_meshes_per_subject_train 2 \
        --train_level mixed --train_pair_mode cross_topology \
        --eval_every 1 --max_subjects_eval_train 3 --no-preload_eval_samples_train \
        --seed 1234
}
smoke_train ict "$TMP_DIR/ict" ""
smoke_train joint "$TMP_DIR/joint" bfm

echo "[smoke] loss finite e nessun NaN letto dalla GT: OK"
