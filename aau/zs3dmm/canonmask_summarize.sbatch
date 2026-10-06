#!/usr/bin/env bash
#SBATCH --job-name=wbes-canon-summarize
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# Maschera canonica, tabelle (a)-(d) di aau/runs/canonmask/PROTOCOL.md: canonmask_summarize.py su BFM
# (congiunto sui soli 19 fuori training), ICT (congiunto sui soli 20 fuori training) e HIFI3D, poi
# aau/runs/canonmask/summary.md. Niente GPU.
#
#   aau/submit.sh zs3dmm/canonmask_summarize.sbatch --dependency=afterok:<eval e baseline>
# WBES_CANON_SUM_DOMAINS="hifi" per un sottoinsieme, WBES_CANON_SUM_BOOTSTRAP=200 per una prova.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv

OUT="$AAU_RUNS/canonmask"
mkdir -p "$OUT"
SPLITS="$AAU_RUNS/ws2_cross3dmm/splits.json"
read -ra DOMAINS <<< "${WBES_CANON_SUM_DOMAINS:-bfm ict hifi}"
opts=(--workers "${SLURM_CPUS_PER_TASK:-8}")
[[ -n "${WBES_CANON_SUM_BOOTSTRAP:-}" ]] && opts+=(--n-bootstrap "$WBES_CANON_SUM_BOOTSTRAP")

for dom in "${DOMAINS[@]}"; do
    (
        export WBES_ZS_DOMAIN="$dom"
        source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
        runs="$ZS_RUNS/data_$(zs_data_fp)"
        extra=()
        case "$dom" in
            bfm)  bl_base="$AAU_RUNS/outlineB/bfm_heldout"
                  extra=(--train-split "$SPLITS" --train-name-offset=0 --heldout-only joint) ;;
            ict)  bl_base="$runs/baselines"
                  extra=(--train-split "$SPLITS" --train-name-offset=-910000 --heldout-only joint) ;;
            hifi) bl_base="$runs/baselines" ;;
            *) echo "ERRORE: dominio $dom (bfm|ict|hifi)" >&2; exit 2 ;;
        esac
        echo "[canon-sum] $dom: runs=$runs baseline prima=$bl_base"
        "$AAU_DIR/run.sh" aau/zs3dmm/canonmask_summarize.py --tag "$dom" --label "$ZS_LABEL" --runs "$runs" \
            --gt "$ZS_DIST_NPZ" --root "$ZS_ROOT" --base-baselines "$bl_base" --out-dir "$OUT" "${extra[@]}" "${opts[@]}"
    )
done

{
    echo "# Maschera canonica: tabelle (a)-(d) del protocollo"
    echo
    echo "Protocollo: aau/runs/canonmask/PROTOCOL.md. Generato da aau/zs3dmm/canonmask_summarize.sbatch (job ${SLURM_JOB_ID:-manual}, $(date '+%Y-%m-%d %H:%M'))."
    echo
    for dom in "${DOMAINS[@]}"; do cat "$OUT/$dom.md"; echo; done
} > "$OUT/summary.md"
echo "[canon-sum] scritto $OUT/summary.md"
