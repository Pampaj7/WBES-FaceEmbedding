#!/usr/bin/env bash
#SBATCH --job-name=wbes-eq-summarize
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#
# Protocollo equalize-support, tabella finale: eqsupport_summarize.py su BFM held-out, ICT
# held-out (pipeline zs3dmm) e HIFI3D (nativo e frame BFM = Rx180 + facce invertite), poi
# aau/runs/eqsupport/summary.md. Niente GPU: legge pair_metrics e matrici.
#
#   aau/submit.sh zs3dmm/eqsupport_summarize.sbatch --dependency=afterok:<eval e baseline>
#
# Prima (base) per dominio:
#   bfm   modelli: aau/runs/ws_bfm/data_<fp>/<arm>_topology (job di questa serie);
#         baseline: aau/runs/outlineB/bfm_heldout (stessa pipeline, stessi 100, 4096 pt)
#   ict   modelli: aau/runs/ws_ictzs/data_<fp>/<arm>; baseline: .../baselines (solo coppie con crop)
#   hifi  modelli: aau/runs/ws_hifi3d/data_<fp>/<arm> e <arm>_frame-xmymz_flip; baseline: .../baselines
# WBES_EQ_SUM_DOMAINS="bfm ict hifi" per sceglierne un sottoinsieme, WBES_EQ_SUM_BOOTSTRAP=200 per
# CI piu' grossolani.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv

OUT="$AAU_RUNS/eqsupport"
mkdir -p "$OUT"
SPLITS="$AAU_RUNS/ws2_cross3dmm/splits.json"
read -ra DOMAINS <<< "${WBES_EQ_SUM_DOMAINS:-bfm ict hifi}"
opts=(--workers "${SLURM_CPUS_PER_TASK:-8}")
[[ -n "${WBES_EQ_SUM_BOOTSTRAP:-}" ]] && opts+=(--n-bootstrap "$WBES_EQ_SUM_BOOTSTRAP")

for dom in "${DOMAINS[@]}"; do
    (
        export WBES_ZS_DOMAIN="$dom"
        source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
        fp="$(zs_data_fp)"
        runs="$ZS_RUNS/data_$fp"
        extra=()
        case "$dom" in
            bfm)  variants=("") ; bl_base="$AAU_RUNS/outlineB/bfm_heldout"
                  extra=(--train-split "$SPLITS" --train-name-offset=0) ;;
            ict)  variants=("") ; bl_base="$runs/baselines"
                  extra=(--train-split "$SPLITS" --train-name-offset=-910000) ;;
            hifi) variants=("" "_frame-xmymz_flip") ; bl_base="$runs/baselines" ;;
            *) echo "ERRORE: dominio $dom (bfm|ict|hifi)" >&2; exit 2 ;;
        esac
        echo "[eq-sum] $dom: runs=$runs eq=$ZS_EQ_DIR baseline prima=$bl_base"
        "$AAU_DIR/run.sh" aau/zs3dmm/eqsupport_summarize.py --tag "$dom" --label "$ZS_LABEL" \
            --runs "$runs" --gt "$ZS_DIST_NPZ" --eq-dir "$ZS_EQ_DIR" --variants "${variants[@]}" \
            --base-baselines "$bl_base" --eq-baselines "$runs/baselines_eqsupport" \
            --out-dir "$OUT" "${extra[@]}" "${opts[@]}"
    )
done

{
    echo "# Equalize support: righe con crop prima e dopo, per metrica e dominio"
    echo
    echo "Generato da aau/zs3dmm/eqsupport_summarize.sbatch (job ${SLURM_JOB_ID:-manual}, $(date '+%Y-%m-%d %H:%M'))."
    echo "Regola di equalizzazione: aau/zs3dmm/eqsupport_view.py. Modelli: congiunto 1019532, BFM-only 1019310."
    echo
    for dom in "${DOMAINS[@]}"; do
        cat "$OUT/$dom.md"
        echo
    done
} > "$OUT/summary.md"
echo "[eq-sum] scritto $OUT/summary.md"
