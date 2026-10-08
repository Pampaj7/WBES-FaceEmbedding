#!/usr/bin/env bash
#SBATCH --job-name=wbes-zs-expr-summarize
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#
# Modalita' espressioni (WBES_ZS_EXPR=1), tabella finale: riconoscimento d'identita' (primario),
# ranking con GT neutra (secondario), crop a parte, sul protocollo di $ZS_RUNS/protocol.md.
# Legge $ZS_RUNS/data_<fp>/ e riscrive $ZS_RUNS/summary.md (protocollo in testa).
#
#   WBES_ZS_DOMAIN=fv WBES_ZS_EXPR=1 aau/submit.sh zs3dmm/zs_expr_summarize.sbatch --dependency=afterok:<...>
#
# WBES_ZS_SUM_BOOTSTRAP=200 per CI piu' grossolani e veloci.
set -euo pipefail

export AAU_NV=""
export WBES_ZS_EXPR="${WBES_ZS_EXPR:-1}"
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
cd "$WBES_ROOT"
aau_require_venv
zs_require_view
if [[ ! -f "$ZS_RUNS/protocol.md" ]]; then
    echo "ERRORE: $ZS_RUNS/protocol.md assente: il protocollo va dichiarato prima" >&2
    exit 1
fi

opts=()
[[ -n "${WBES_ZS_SUM_BOOTSTRAP:-}" ]] && opts+=(--n-bootstrap "$WBES_ZS_SUM_BOOTSTRAP")
ZS_FP="$(zs_data_fp)"
echo "[zs-expr-sum] dominio=$WBES_ZS_DOMAIN host=$(hostname) job=${SLURM_JOB_ID:-none} risultati da $ZS_RUNS/data_$ZS_FP"
"$AAU_DIR/run.sh" aau/zs3dmm/zs_expr_summarize.py --runs "$ZS_RUNS/data_$ZS_FP" --summary-dir "${WBES_ZS_SUMMARY_DIR:-$ZS_RUNS}" \
    --gt "$ZS_DIST_NPZ" --view-dir "$ZS_DATA_DIR" --eval-seed "${WBES_EVAL_SEED:-1234}" "${opts[@]}"
