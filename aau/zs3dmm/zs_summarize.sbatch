#!/usr/bin/env bash
#SBATCH --job-name=wbes-zs-summarize
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#
# Zero-shot su un 3DMM mai visto (WBES_ZS_DOMAIN=hifi|fv), tabella finale: tre bracci +
# baseline, due GT, CI bootstrap del paper. Niente GPU: legge solo csv, json e le matrici delle
# baseline. Legge $ZS_RUNS/data_<fp>/ (impronta dei dati, zs_env.sh) e scrive summary.md
# in $ZS_RUNS (aau/runs/ws_hifi3d o aau/runs/ws_faceverse).
#
#   WBES_ZS_DOMAIN=hifi aau/submit.sh zs3dmm/zs_summarize.sbatch --dependency=afterok:<eval e baseline>
#
# WBES_ZS_SUM_BOOTSTRAP=200 per CI piu' grossolani e veloci.
# WBES_ZS_SUM_VARIANT=_frame-xmymz (o _evalframe-rms, ...): tabella della variante, con l'effetto
# appaiato rispetto alla base e il controllo che la Chamfer non cambi -> summary<variante>.md.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
cd "$WBES_ROOT"
aau_require_venv
zs_require_view

opts=()
[[ -n "${WBES_ZS_SUM_BOOTSTRAP:-}" ]] && opts+=(--n-bootstrap "$WBES_ZS_SUM_BOOTSTRAP")

echo "[zs-sum] dominio=$WBES_ZS_DOMAIN host=$(hostname) job=${SLURM_JOB_ID:-none} opts=${opts[*]:-(default)}"
ZS_FP="$(zs_data_fp)"
echo "[zs-sum] risultati da $ZS_RUNS/data_$ZS_FP"
"$AAU_DIR/run.sh" aau/zs3dmm/zs_summarize.py --label "$ZS_LABEL" --runs "$ZS_RUNS/data_$ZS_FP" \
    --summary-dir "$ZS_RUNS" --variant "${WBES_ZS_SUM_VARIANT:-}" \
    --gt "$ZS_DIST_NPZ" --gt-coef "$ZS_COEF_NPZ" --root "$ZS_ROOT" "${opts[@]}"
