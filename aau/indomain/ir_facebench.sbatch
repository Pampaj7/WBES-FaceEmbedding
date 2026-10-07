#!/usr/bin/env bash
#SBATCH --job-name=wbes-ir-fb
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#
# Riconoscimento in dominio: baseline faceBench su tutte le coppie di mesh con etichette diverse
# dei tre insiemi (aau/indomain/ir_facebench.py). Uno shard delle 45 unita' (insieme x coppia di
# etichette) per job. Partizione prioritized senza GPU, come zs_baselines.sbatch.
#
#   for k in $(seq 0 14); do WBES_IR_FB_SHARD=$k/15 aau/submit.sh indomain/ir_facebench.sbatch \
#       --exclude=a768-l40s-0[1-6]; done
#   WBES_IR_FB_SHARD="7/15 8/15" aau/submit.sh indomain/ir_facebench.sbatch --cpus-per-task=64 --mem=96G ...
#
# --exclude: i core dei nodi L40S restano ai job GPU (la distillazione in coda).
# TEMPI: misurati ~6-7 s per coppia-core con 16 worker (smoke, 64 coppie BFM e ICT; zs_baselines
# diceva 2.7 s a un core). 15 x (108^2 + 2 x 89^2) = 412k coppie, ~700 ore-core: 3 unita' per shard,
# un'unita' BFM (11.664 coppie) ~40 min a 32 worker. Riprendibile per unita'.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

# WBES_IR_FB_SHARD puo' contenere piu' shard ("7/15 8/15"): girano in parallelo nello stesso job,
# i core divisi fra loro, ognuno con la sua copia della geometria (niente corse sui file). Serve a
# stare sotto il limite di 12 job per utente.
SHARDS=(${WBES_IR_FB_SHARD:-0/1})
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} / ${#SHARDS[@]} ))
STAGE="${TMPDIR:-/tmp}/wbes_ir_fb_${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$STAGE"' EXIT
echo "[ir-fb] host=$(hostname) job=${SLURM_JOB_ID:-none} shard=${SHARDS[*]} worker per shard=$WORKERS stage=$STAGE"
pids=()
for shard in "${SHARDS[@]}"; do
    "$AAU_DIR/outlineB/run_o3d.sh" aau/indomain/ir_facebench.py --out-root "${WBES_IR_FB_OUT:-$AAU_RUNS/indomain_recog/facebench}" \
        --stage-dir "$STAGE/${shard%%/*}" --workers "$WORKERS" --shard "$shard" ${WBES_IR_FB_ARGS:-} &
    pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
(( rc == 0 )) || { echo "[ir-fb] ERRORE: almeno uno shard e' fallito" >&2; exit 1; }
echo "[ir-fb] OK"
