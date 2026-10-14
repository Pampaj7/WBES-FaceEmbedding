#!/usr/bin/env bash
#SBATCH --job-name=wbes-now-rank-embed
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# NoW ranking (aau/runs/evidence/now_ranking/REGOLE.md, sez. 2): d_P e d_F dei factorized s1234 e s2345.
#   1. operatori ad area unitaria (k_eig 128, areanorm_operators.py come now_ops_latent.sbatch) delle patch nuove:
#      recon_face/<m> e recon_face/<m>_metric dei metodi WBES_RANKING_NEW, e scan_face_metric (gli altri ci sono);
#   2. tabelle di scala (tools/build_scale_table.py, dominio "now", u = 1 mm) delle stesse patch, in
#      ~/data/now_ranking_work/scale_tables; quelle dei tre metodi e di scan_face sono quelle del passo now di
#      eval_body.sh (trainer_v3/factorized/scale_tables);
#   3. now_factorized.py attraverso eval_v3.py con WBES_V3_FACTORIZED_OUT=full, per i due semi.
# Ripartibile: operatori ed embedding gia' presenti si riusano.
#
#   aau/submit.sh recon/now_ranking_embed.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net

WORK="${WBES_NOW_WORK:-$HOME/data/now_eval_work}"
RW="${WBES_RANKING_WORK:-$HOME/data/now_ranking_work}"
export WBES_NOW_OUT="${WBES_NOW_OUT:-$AAU_RUNS/evidence/now_ranking/tables}"
OLD="${WBES_RANKING_OLD_METHODS:-3ddfa_v2 synergynet prnet}"
NEW="${WBES_RANKING_NEW:-3ddfa_v3}"
METRIC="${WBES_RANKING_METRIC:-3ddfa_v3}"
ST="$AAU_RUNS/evidence/trainer_v3/factorized/scale_tables"
RUNS="$AAU_RUNS/evidence/trainer_v3/ablations/c3f_runs"
SUFFIX="_withops_areanorm"
N_SHARDS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$RW/scale_tables" "$WBES_NOW_OUT"
echo "[now-rank-embed] host=$(hostname) job=${SLURM_JOB_ID:-none} nuovi=$NEW metrici=$METRIC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

SHARD_LOGS="$AAU_LOGS/${SLURM_JOB_NAME:-wbes-now-rank-embed}-${SLURM_JOB_ID:-manual}.shards"
mkdir -p "$SHARD_LOGS"

# coppie patch:operatori:gruppo della tabella (nome della cartella degli operatori, come la legge eval_v3)
SETS="scan_face_metric:scan_face_metric$SUFFIX:scan_face_metric$SUFFIX"
for m in $NEW; do SETS="$SETS recon_face/$m:recon_face$SUFFIX/$m:$m"; done
for m in $METRIC; do SETS="$SETS recon_face/${m}_metric:recon_face$SUFFIX/${m}_metric:${m}_metric"; done

TABLES="$ST/now_scan.npz"
for m in $OLD; do TABLES="$TABLES:$ST/now_$m.npz"; done
for set in $SETS; do
    IFS=: read -r in_rel out_rel group <<< "$set"
    IN_DIR="$WORK/$in_rel"; OUT_DIR="$WORK/$out_rel"
    [[ -d "$IN_DIR" ]] || { echo "ERRORE: mancano le patch in $IN_DIR (now_ranking_prep.sbatch)" >&2; exit 1; }
    mkdir -p "$OUT_DIR"
    t0=$(date +%s); pids=()
    for (( i = 0; i < N_SHARDS; i++ )); do
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AAU_NV= \
            "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
            --input-dir "$IN_DIR" --output-dir "$OUT_DIR" --k-eig 128 --shard "$i/$N_SHARDS" \
            > "$SHARD_LOGS/$group-shard$i.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "ERRORE: shard $i di $group (vedi $SHARD_LOGS/$group-shard$i.log)" >&2; exit 1; }
    done
    N_IN=$(find "$IN_DIR" -maxdepth 1 -name '*.npz' | wc -l)
    N_OUT=$(find "$OUT_DIR" -maxdepth 1 -name '*.npz' | wc -l)
    echo "[now-rank-embed] operatori $in_rel: $N_OUT/$N_IN in $(( $(date +%s) - t0 )) s"
    (( N_OUT == N_IN )) || { echo "ERRORE: operatori incompleti per $in_rel" >&2; exit 1; }

    T="$RW/scale_tables/now_$group.npz"
    AAU_NV= "$AAU_DIR/run.sh" v3_work/trainer/tools/build_scale_table.py --view-dir "$IN_DIR" --domain now \
        --group "$group" --out "$T" --workers "$N_SHARDS"
    TABLES="$TABLES:$T"
done

for seed in s1234 s2345; do
    sfx=""; [[ "$seed" != s1234 ]] && sfx="_$seed"
    CK="$(ls "$RUNS/factorized$sfx"/v3_*/checkpoints/epoch072_ema.pth | head -1)"
    [[ -f "$CK" ]] || { echo "ERRORE: checkpoint factorized$sfx assente" >&2; exit 1; }
    METHODS="$(echo $OLD $NEW $(for m in $METRIC; do echo "${m}_metric"; done) | tr ' ' ',')"
    WBES_V3_FACTORIZED_OUT=full WBES_V3_SCALE_TABLES="$TABLES" AAU_NV="--nv" \
        "$AAU_DIR/run.sh" v3_work/trainer/eval_v3.py -- aau/recon/now_factorized.py \
        --checkpoint "$CK" --key "factorized_$seed" --methods "$METHODS" --cache-root "$RW"
done

echo "[now-rank-embed] OK"
