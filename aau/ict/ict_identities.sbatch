#!/usr/bin/env bash
#SBATCH --job-name=wbes-ict-identities
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# ICT-5000, passo 1: 5000 identita' neutre della sola regione volto (9409 vertici),
# seed 1234, pesi N(0,1) sui 100 modi PCA come `FaceModel.randomize_identity`.
#
#   aau/submit.sh ict/ict_identities.sbatch
#
# Perche' una GPU per un passo che genera 5000 matvec: il controllo di distinzione Z1mX
# chiede la distanza minima FRA TUTTE LE COPPIE anche in spazio forma, cioe' 12,5 milioni
# di coppie x 9409 vertici. E' l'unica parte cara (il doppio ciclo numpy del gemello FLAME
# ci metterebbe ore) e su L40S sono minuti. Con --skip-shape-check il job gira anche su cpu.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

# `ls dir/*.npz` esplode con "Argument list too long" a 30.000 file (misurato: job 1019283
# e' fallito cosi' DOPO aver scritto correttamente tutte le mesh). find non espande niente.
count_npz() { find "$1" -maxdepth 1 -name "$2" -type f -o -maxdepth 1 -name "$2" -type l | wc -l; }
aau_require_venv

OUT_DIR="${WBES_ICT_IDENTITIES_DIR:-$WBES_ROOT/datasets/ICT/identities}"
N_ID="${WBES_ICT_N_IDENTITIES:-5000}"
SEED="${WBES_ICT_SEED:-1234}"

echo "[ict-identities] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[ict-identities] out=$OUT_DIR n=$N_ID seed=$SEED"

"$AAU_DIR/run.sh" v2_work/genict/generate_identities.py \
    --n-identities "$N_ID" --seed "$SEED" --out-dir "$OUT_DIR"

echo "[ict-identities] file: $(count_npz "$OUT_DIR" "ict*.npz")  dimensione: $(du -sh "$OUT_DIR" | cut -f1)"
cat "$OUT_DIR/manifest.json"
