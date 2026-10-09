#!/usr/bin/env bash
#SBATCH --job-name=wbes-stream-xprod
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#
# Produttori SOLO CPU per l'anello CONDIVISO su CephFS (letto dai rank di massive.sbatch con STREAM_EXTRA=<anello>):
#   sbatch v3_work/stream/slurm/extra_producers.sbatch                                  (prova, 32 CPU)
#   sbatch -p aicentre-a100 --qos=unprivileged --nodelist=nv-ai-04 --cpus-per-task=128 --mem=200G \
#          --time=2-00:00:00 v3_work/stream/slurm/extra_producers.sbatch                (prelazionabile: va bene)
# Piu' job scrivono nello stesso anello: il contatore delle sequenze e' sotto flock, il budget FIFO e' comune (ogni
# produttore cancella gli shard piu' vecchi dell'intero anello, ogni XPROD_EVICT_EVERY scritture).
# Vincoli di disco (utente, 9 ottobre): anello al massimo 150 GiB; la quota deve restare con >= 300 GB liberi anche
# con l'anello pieno: il job lo controlla prima di partire. L'anello NON si cancella alla fine (lo leggono altri job):
# va tolto a mano a fine run (rm -rf $XPROD_RING).
# Variabili: XPROD_RING (datasets/STREAM/shared_ring), XPROD_GB (<= 150), XPROD_K (128), XPROD_SOURCES (massive),
#   XPROD_EXPR_FRAC (0.5), XPROD_DURATION (s, 0 = fino al limite), XPROD_GROUPS_PER_SHARD (4), XPROD_EVICT_EVERY (8),
#   XPROD_MM_AUG (come STREAM_MM_AUG di massive.sbatch; vuoto = puri).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
RING="${XPROD_RING:-$WBES_ROOT/datasets/STREAM/shared_ring}"
XPROD_MM_AUG="${XPROD_MM_AUG-hybrid=0.3,expr_transfer=0.15,rbf=0.15}"
GB="${XPROD_GB:-150}"
QUOTA_GB=2199
python3 - "$GB" "$QUOTA_GB" <<'PY'
import os, subprocess, sys
gb, quota = float(sys.argv[1]), float(sys.argv[2])
if gb > 150:
    sys.exit(f"XPROD_GB {gb} > 150 GiB (vincolo dell'utente)")
used = int(subprocess.run(["getfattr", "-n", "ceph.dir.rbytes", "--only-values", os.path.expanduser("~")],
                          capture_output=True, text=True).stdout.strip() or 0) / 1e9
free = quota - used
print(f"[xprod] quota: usati {used:.0f} GB su {quota:.0f}, liberi {free:.0f} GB; anello fino a {gb * 1.074:.0f} GB")
if free - gb * 1.074 < 300:
    sys.exit(f"[xprod] FERMO: con l'anello pieno resterebbero {free - gb * 1.074:.0f} GB < 300 GB liberi")
PY
RST="${SLURM_RESTART_COUNT:-0}"
SEED=$(( 30000000 + ${SLURM_JOB_ID:-0} % 100000 * 100 + RST ))
E="$AAU_RUNS/evidence/stream/xprod_${SLURM_JOB_ID:-manual}"
mkdir -p "$E" "$RING"
echo "[xprod] $(date +%F_%T) host=$(hostname) cpu=$(nproc) anello=$RING (${GB} GiB) seme=$SEED riavvio=$RST"
AAU_NV= "$AAU_DIR/run.sh" v3_work/stream/producer.py --ring "$RING" --ring-gb "$GB" --n-proc "$(nproc)" \
  --k-eig "${XPROD_K:-128}" --evecs-dtype fp32 --sources "${XPROD_SOURCES:-massive}" --provenance --canonical-gt \
  --expr-frac "${XPROD_EXPR_FRAC:-0.5}" --seed "$SEED" --groups-per-shard "${XPROD_GROUPS_PER_SHARD:-4}" \
  --evict-every "${XPROD_EVICT_EVERY:-8}" --duration "${XPROD_DURATION:-0}" --stats-every 60 \
  ${XPROD_MM_AUG:+--mm-aug "$XPROD_MM_AUG"} \
  --stats-dir "$E/stats" --summary "$E/producers.json" 2>&1 | tee "$E/producers.log" | grep -v "Warning\|FloatTensor"
echo "[xprod] $(date +%F_%T) fine; anello: $(ls "$RING/shards" | wc -l) shard, $(du -sh "$RING" | cut -f1)"
