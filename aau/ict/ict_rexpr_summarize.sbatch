#!/usr/bin/env bash
#SBATCH --job-name=wbes-ict-rexpr-sum
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# Tabella finale di WS5 rifatto + CI bootstrap. Niente GPU: legge solo csv e json, quindi
# non occupa una delle 12 GPU del tetto QoS.
#
#   aau/submit.sh ict/ict_rexpr_summarize.sbatch --dependency=afterok:<job rexpr rank>
#
# Come in ict_summarize.sbatch le opzioni passano da variabili d'ambiente e non da riga di
# comando, perche' aau/submit.sh mette tutto quello che segue il nome dello script fra le
# opzioni di sbatch, non fra gli argomenti dello script:
#   WBES_ICT_SUM_RUNS=<runs root> WBES_ICT_SUM_OUT=<out dir> WBES_ICT_SUM_BOOTSTRAP=200 \
#     aau/submit.sh ict/ict_rexpr_summarize.sbatch
#
# Il bootstrap e' la parte cara: 3 regimi x (fino a 5 condizioni + 1 aggregato) x 2
# metriche x 1000 ricampionamenti, con 4950 righe per condizione same_k e 99.000 per mixed.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv

# Nodo senza GPU: --nv farebbe fallire singularity per mancanza di libnvidia.
export AAU_NV=""

opts=()
[[ -n "${WBES_ICT_SUM_RUNS:-}" ]] && opts+=(--runs-root "$WBES_ICT_SUM_RUNS")
[[ -n "${WBES_ICT_SUM_OUT:-}" ]] && opts+=(--out-dir "$WBES_ICT_SUM_OUT")
[[ -n "${WBES_ICT_SUM_BOOTSTRAP:-}" ]] && opts+=(--n-bootstrap "$WBES_ICT_SUM_BOOTSTRAP")

echo "[ict-rexpr-sum] host=$(hostname) job=${SLURM_JOB_ID:-none} opts=${opts[*]:-(default)} $*"
"$AAU_DIR/run.sh" aau/ict/ict_rexpr_summarize.py "${opts[@]}" "$@"
