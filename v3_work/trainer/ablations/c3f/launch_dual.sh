#!/usr/bin/env bash
# Bracci ``dual`` (testa doppia, factorized.md sez. 7) con il protocollo dei bracci factorized/factorized2/ctrlfr
# (factorized_protocol.md + emendamenti 1-3): semi 1234 e 2345, 21.096 passi, checkpoint EMA alle epoche 36 e 72,
# stesso store C3F (/tmp/wbes_v3_store_c3f su nv-ai-04, custode 1062762 finche' esiste HOLD_STORE), stessi dati.
# Catena: training (train.sbatch, A100 prelazionabile, --requeue + --resume auto) -> eval (eval.sbatch: form, FaMoS,
# NoW su u e z_F) -> delta appaiati e riepilogo (fact_paired.py, fact_summary.py) dopo le due eval. Job in c3f_jobs.txt.
# Tempi da sacct dei bracci gemelli: training 5h05-5h08 (1062768-1062989), eval 56 min senza intoppi, 2h30 con i
# tentativi per "unknown userid" (1066193), riepilogo 1h05 (1066330).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/aau/env.sh"
D="$WBES_ROOT/v3_work/trainer/ablations/c3f"
LOG="$AAU_RUNS/evidence/trainer_v3/ablations/c3f_jobs.txt"
[[ -e "$AAU_RUNS/evidence/trainer_v3/factorized/HOLD_STORE" ]] || { echo "ERRORE: HOLD_STORE assente, lo store puo' sparire" >&2; exit 3; }
EVALS=()
for seed in ${WBES_V3_DUAL_SEEDS:-1234 2345}; do
  T=$(WBES_V3_ARM=dual WBES_V3_SEED="$seed" "$AAU_DIR/submit.sh" "$D/train.sbatch" --time=07:00:00 \
      --job-name="wbes-v3-c3f-dual-s$seed" --parsable)
  E=$(WBES_V3_ARM=dual WBES_V3_SEED="$seed" "$AAU_DIR/submit.sh" "$D/eval.sbatch" --dependency="afterok:$T" \
      --kill-on-invalid-dep=yes --time=04:00:00 --mem=70G --job-name="wbes-v3-c3f-eval-dual-s$seed" --parsable)
  EVALS+=("$E")
  echo "$(date +%F_%T) dual s$seed training=$T eval=$E" | tee -a "$LOG"
done
S=$(sbatch --parsable -p prioritized --gres=NONE -c 32 --mem=96G -t 02:00:00 -J wbes-fact-final-dual \
    -o "$AAU_LOGS/%x-%j.out" -e "$AAU_LOGS/%x-%j.err" --chdir="$WBES_ROOT" \
    --dependency="afterany:$(IFS=:; echo "${EVALS[*]}")" \
    --wrap "export AAU_NV= UGT_THREADS=1; v3_work/unified_gt/run.sh v3_work/trainer/tools/fact_paired.py --workers 32 && aau/run.sh v3_work/trainer/tools/fact_summary.py")
echo "$(date +%F_%T) dual: riepilogo (delta appaiati + factorized_results.md)=$S, dopo ${EVALS[*]}" | tee -a "$LOG"
