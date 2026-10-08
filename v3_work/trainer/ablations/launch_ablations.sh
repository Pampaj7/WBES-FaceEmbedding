#!/usr/bin/env bash
# Sottomette i bracci delle ablazioni v3, uno per job, con gli STESSI passi. NON lanciato dal coder: lo
# lancia il PI dopo il critic.
#
#   WBES_V3_STEPS=<T> v3_work/trainer/ablations/launch_ablations.sh ctrl area arearobust     (E7, priorita' 1)
#   WBES_V3_STEPS=<T> WBES_V3_BLOCKS=16 v3_work/trainer/ablations/launch_ablations.sh ...   (--mem 150G)
#   WBES_V3_A100=1 ...   sulle A100 prelazionabili, con --requeue
#   WBES_V3_A100=1 WBES_V3_LOCAL_OPS=/tmp/wbes_v3_ablops ...   dopo precompute_ops_local.sbatch (niente pre-pass)
# Variabili lette dallo sbatch: WBES_V3_STEPS (obbligatoria), WBES_V3_BLOCKS (8), WBES_V3_FORWARD (sequential; groups su A100),
# WBES_V3_STEPS_PER_EPOCH, WBES_V3_EVAL_EVERY (24), WBES_V3_SEED (1234), WBES_V3_MAX_HOURS (5.1).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/aau/env.sh"
: "${WBES_V3_STEPS:?WBES_V3_STEPS (passi totali, uguali per tutti i bracci)}"
export WBES_V3_STEPS WBES_V3_BLOCKS="${WBES_V3_BLOCKS:-8}"
MEM=(); [[ "$WBES_V3_BLOCKS" == 16 ]] && MEM=(--mem=150G)
# WBES_V3_A100=1: A100 di nv-ai-04 (eccezione dell'utente, prelazionabili): ripresa da last.pth al requeue
[[ "${WBES_V3_A100:-0}" == 1 ]] && MEM+=(-p aicentre-a100 --qos=unprivileged --gres=gpu:a100:1 --requeue)
# WBES_V3_LOCAL_OPS=/tmp/wbes_v3_ablops: operatori precalcolati sul disco di nv-ai-04 (precompute_ops_local.sbatch)
[[ -n "${WBES_V3_LOCAL_OPS:-}" ]] && MEM+=(--nodelist=nv-ai-04) && export WBES_V3_LOCAL_OPS
for arm in "$@"; do
  id=$(WBES_V3_ARM="$arm" "$AAU_DIR/submit.sh" "$WBES_ROOT/v3_work/trainer/ablations/train_ablation_v3.sbatch" \
       "${MEM[@]}" --job-name="wbes-v3-abl-$arm" --parsable)
  echo "$arm=$id"
done
