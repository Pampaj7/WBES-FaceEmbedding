#!/usr/bin/env bash
# Sottomette le eval del pilota pot e il job della tabella aau/runs/pilot_pot/summary.md.
# Si lancia sul frontend DOPO che i due training sono partiti (serve la run dir):
#
#   aau/pilot_pot_queue.sh <runs_root m55> <runs_root dual> <jobid precompute ICT> [--dry-run]
#
# Per braccio (controllo s1234 compreso):
#   1. eval per gruppo BFM, eval_frame_topology.sbatch (T4), afterok sul training. Per il
#      controllo c'e' gia' 1055030 (eval_frame_queue.sh): si riusa, non si duplica.
#   2. zero-shot ICT, ict/ict_zeroshot_rank.sbatch, solo clean, WBES_EVAL_SEED=1234, stage
#      ict_zeroshot_pilot_clean, A10. afterok sul training (e sul precompute ICT per i bracci
#      col pozzo) E afterany sull'eval BFM dello stesso braccio: cosi' ogni braccio usa una GPU
#      alla volta e il pilota resta dentro le 3 GPU anche mentre l'altro training gira.
#   3. tabella (cpu), afterany su tutte le eval: con un'eval mancante scrive la riga "n/d",
#      lo dice nella sezione Mancanti ed esce 1.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$WBES_ROOT"

M55_ROOT="$(realpath "${1:?runs_root di pot_m55}")"
DUAL_ROOT="$(realpath "${2:?runs_root del dual}")"
ICT_OPS_JOB="${3:?job id del precompute ICT}"
DRY=0
[[ "${4:-}" == "--dry-run" ]] && DRY=1

CTRL_ROOT="$AAU_RUNS/remesh_v1recipe_current_s1234_1055026"
CTRL_TRAIN_JOB=1055026
CTRL_EVAL_JOB=1055030
ICT_STD="$WBES_ROOT/datasets/ICT/eval_view_heldout"
ICT_POT="$WBES_ROOT/datasets/ICT/eval_view_heldout_pot055"
ICT_DIST="$WBES_ROOT/datasets/ICT/train_ready/gt_matrix.npz"
STAGE=ict_zeroshot_pilot_clean

ckpt_of() {   # il best_by_xtopo_mesh_clean.pth (puo' non esistere ancora) dell'unica run dir
    local runs=("$1"/*/config.json)
    if [[ ${#runs[@]} -ne 1 || ! -f "${runs[0]}" ]]; then
        echo "ERRORE: attesa una sola run dir con config.json in $1, trovate: ${runs[*]}" >&2
        return 1
    fi
    echo "$(dirname "${runs[0]}")/checkpoints/best_by_xtopo_mesh_clean.pth"
}
job_of() {    # job id dal nome del runs_root (..._<jobid>)
    echo "${1##*_}"
}
live() {      # il job id se e' ancora da aspettare, niente se COMPLETED, errore altrimenti
    local state
    state="$(sacct -j "$1" -X -n -o State | head -1 | tr -d ' ')"
    case "$state" in
        PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING) echo "$1" ;;
        COMPLETED) ;;
        *) echo "ERRORE: job $1 in stato '${state:-sconosciuto}'" >&2; return 1 ;;
    esac
}
deps() {      # deps afterok "a b" afterany "c" -> --dependency=afterok:a:b,afterany:c (o niente)
    local out=() kind ids j list
    while (( $# )); do
        kind="$1"; ids="$2"; shift 2
        list=""
        for j in $ids; do
            j="$(live "$j")" || return 1      # mai una dipendenza persa in silenzio
            [[ -n "$j" ]] && list="$list:$j"
        done
        [[ -n "$list" ]] && out+=("$kind$list")
    done
    (( ${#out[@]} )) && echo "--dependency=$(IFS=,; echo "${out[*]}")"
    return 0
}
submit() {    # stampa il job id
    if (( DRY )); then
        echo "[dry] $*" >&2
        echo "DRY"
        return
    fi
    local out
    out="$("$@")"
    echo "${out%%;*}"
}

[[ "$(squeue -h -j "$CTRL_EVAL_JOB" -o %j 2>/dev/null)" == "wbes-evalframe-s1234-maxabs" ]] \
    || [[ -f "$AAU_RUNS/eval_frame/remesh_v1recipe_current_s1234_1055026_maxabs/.done" ]] \
    || { echo "ERRORE: l'eval del controllo $CTRL_EVAL_JOB non e' in coda e non e' completa" >&2; exit 1; }

declare -a ALL_EVALS=("$CTRL_EVAL_JOB")
declare -a ROWS=()
CTRL_CKPT="$(ckpt_of "$CTRL_ROOT")"
ROWS+=(--row controllo "$AAU_RUNS/eval_frame/$(basename "$CTRL_ROOT")_maxabs" "$CTRL_ROOT" "$ICT_STD")

# controllo: solo ICT
d="$(deps afterok "$CTRL_TRAIN_JOB" afterany "$CTRL_EVAL_JOB")"
j="$(WBES_CKPT="$CTRL_CKPT" WBES_DATA_DIR="$ICT_STD" WBES_DIST_NPZ="$ICT_DIST" WBES_EVAL_SEED=1234 \
     WBES_EVAL_SCENARIOS=clean WBES_EVAL_STAGE="$STAGE" \
     submit "$AAU_DIR/submit.sh" ict/ict_zeroshot_rank.sbatch --parsable \
       --job-name=wbes-pilot-ict-ctrl --gres=gpu:a10:1 \
       $d)"
echo "[ict] controllo -> $j"
ALL_EVALS+=("$j")

for spec in "m55|$M55_ROOT|$ICT_POT" "dual|$DUAL_ROOT|$ICT_STD"; do
    IFS='|' read -r tag root ict_data <<< "$spec"
    tjob="$(job_of "$root")"
    ckpt="$(ckpt_of "$root")"
    grep -q "^arm=" "$root/pilot_arm.txt"
    d="$(deps afterok "$tjob")"
    j_bfm="$(WBES_CKPT="$ckpt" WBES_FRAME=maxabs \
        submit "$AAU_DIR/submit.sh" eval_frame_topology.sbatch --parsable \
          "--job-name=wbes-pilot-evalframe-$tag" $d)"
    echo "[bfm] $tag -> $j_bfm (afterok:$tjob)"
    if (( DRY )); then d="(afterok $tjob $ICT_OPS_JOB, afterany bfm)"; else d="$(deps afterok "$tjob $ICT_OPS_JOB" afterany "$j_bfm")"; fi
    j_ict="$(WBES_CKPT="$ckpt" WBES_DATA_DIR="$ICT_STD" WBES_POT_DIR="$ICT_POT" \
        WBES_DIST_NPZ="$ICT_DIST" WBES_EVAL_SEED=1234 WBES_EVAL_SCENARIOS=clean \
        WBES_EVAL_STAGE="$STAGE" \
        submit "$AAU_DIR/submit.sh" ict/ict_zeroshot_rank.sbatch --parsable \
          "--job-name=wbes-pilot-ict-$tag" --gres=gpu:a10:1 \
          $d)"
    echo "[ict] $tag -> $j_ict"
    ALL_EVALS+=("$j_bfm" "$j_ict")
    ROWS+=(--row "pot_$tag" "$AAU_RUNS/eval_frame/$(basename "$root")_maxabs" "$root" "$ict_data")
done

wrap="AAU_NV= $(printf '%q ' "$AAU_DIR/run.sh" aau/models/pilot_summary.py \
    --out "$AAU_RUNS/pilot_pot/summary.md" "${ROWS[@]}")"
j_tab="$(submit sbatch --parsable --job-name=wbes-pilot-summary --partition=cpu \
    --cpus-per-task=2 --mem=8G --time=00:30:00 \
    "--chdir=$WBES_ROOT" "--output=$AAU_LOGS/%x-%j.out" "--error=$AAU_LOGS/%x-%j.err" \
    "--dependency=afterany:$(IFS=:; echo "${ALL_EVALS[*]}")" --wrap "$wrap")"
echo "[table] $j_tab dopo: ${ALL_EVALS[*]}"
