#!/usr/bin/env bash
# Sottomette le eval delle ablazioni v3 e il job della tabella aau/runs/ablations_v3/summary.md.
# Si lancia sul frontend DOPO che i tre training sono partiti (serve la run dir):
#
#   aau/models/ablation_queue.sh <runs_root B> <runs_root C> <runs_root E> <job verifica ops ICT> [--dry-run]
#
# GPU V100 (nv-ai-03, dove girano i training): sugli A10 la RAM allocabile non basta (5 ottobre).
# Tre corsie, una GPU ciascuna, cosi' le ablazioni restano dentro 3 GPU anche quando un training
# finisce e gli altri girano ancora:
#   B: eval BFM (afterok training) -> ICT (afterany) -> eval BFM del controllo (afterok 1055026)
#      -> ICT del controllo
#   C, E: eval BFM (afterok training) -> ICT (afterok verifica degli operatori ICT, afterany BFM)
# Il controllo passa dagli stessi script dei bracci (aau/models/eval_ablation_topology.sbatch,
# ict_ablation_rank.sbatch), cosi' i Delta sono fra numeri dello stesso codice. Poi la tabella
# (cpu), afterany su tutte: con un'eval mancante scrive "n/d", lo dice in Mancanti ed esce 1.
# --kill-on-invalid-dep=yes: se un training muore le sue eval vengono cancellate.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
cd "$WBES_ROOT"

B_ROOT="$(realpath "${1:?runs_root di B}")"
C_ROOT="$(realpath "${2:?runs_root di C}")"
E_ROOT="$(realpath "${3:?runs_root di E}")"
ICT_CHECK_JOB="${4:?job id della verifica degli operatori ICT}"
DRY=0
[[ "${5:-}" == "--dry-run" ]] && DRY=1

CTRL_ROOT="$AAU_RUNS/remesh_v1recipe_current_s1234_1055026"
CTRL_TRAIN_JOB=1055026
ICT_STD="$WBES_ROOT/datasets/ICT/eval_view_heldout"
ICT_RA1="$WBES_ROOT/datasets/ICT/eval_view_heldout_robust_area1"
STAGE=ict_zeroshot_abl3_clean
EVAL_BFM="$AAU_RUNS/ablations_v3/eval_bfm"

ckpt_of() {   # il best_by_xtopo_mesh_clean.pth (puo' non esistere ancora) dell'unica run dir
    local runs=("$1"/*/config.json)
    if [[ ${#runs[@]} -ne 1 || ! -f "${runs[0]}" ]]; then
        echo "ERRORE: attesa una sola run dir con config.json in $1, trovate: ${runs[*]}" >&2
        return 1
    fi
    echo "$(dirname "${runs[0]}")/checkpoints/best_by_xtopo_mesh_clean.pth"
}
job_of() { echo "${1##*_}"; }
live() {      # il job id se e' ancora da aspettare, niente se COMPLETED, errore altrimenti
    local state
    state="$(sacct -j "$1" -X -n -o State | head -1 | tr -d ' ')"
    # un job appena sottomesso puo' non essere ancora in sacct: senza questo la dipendenza cade
    [[ -z "$state" ]] && state="$(squeue -j "$1" -h -o %T 2>/dev/null | head -1)"
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
            [[ "$j" == DRY ]] && continue
            j="$(live "$j")"; [[ -n "$j" ]] && list="$list:$j"
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

declare -a ALL_EVALS=()
declare -a ROWS=()

bfm_eval() {  # <tag> <runs_root> <dipendenze...>
    local tag="$1" root="$2"; shift 2
    WBES_CKPT="$root" WBES_EXPECT_SEED=1234 \
        submit "$AAU_DIR/submit.sh" models/eval_ablation_topology.sbatch --parsable \
          "--job-name=wbes-abl-evalbfm-$tag" --gres=gpu:v100:1 --kill-on-invalid-dep=yes $(deps "$@")
}
ict_eval() {  # <tag> <ckpt> <dipendenze...>
    local tag="$1" ckpt="$2"; shift 2
    WBES_CKPT="$ckpt" WBES_EVAL_SEED=1234 WBES_EVAL_STAGE="$STAGE" \
        submit "$AAU_DIR/submit.sh" models/ict_ablation_rank.sbatch --parsable \
          "--job-name=wbes-abl-ict-$tag" --gres=gpu:v100:1 --time=04:00:00 --kill-on-invalid-dep=yes $(deps "$@")
}

# corsia B, poi il controllo
B_JOB="$(job_of "$B_ROOT")"
# checkpoint risolti PRIMA di sottomettere: dentro un argomento $(...) un errore non fermerebbe set -e
B_CK="$(ckpt_of "$B_ROOT")"; C_CK="$(ckpt_of "$C_ROOT")"; E_CK="$(ckpt_of "$E_ROOT")"; CTRL_CK="$(ckpt_of "$CTRL_ROOT")"
declare -A CK=([C]="$C_CK" [E]="$E_CK")
j1="$(bfm_eval B "$B_ROOT" afterok "$B_JOB")";                         echo "[bfm] B -> $j1"
j2="$(ict_eval B "$B_CK" afterok "$B_JOB" afterany "$j1")"; echo "[ict] B -> $j2"
j3="$(bfm_eval ctrl "$CTRL_ROOT" afterok "$CTRL_TRAIN_JOB" afterany "$j2")"; echo "[bfm] controllo -> $j3"
j4="$(ict_eval ctrl "$CTRL_CK" afterok "$CTRL_TRAIN_JOB" afterany "$j3")"; echo "[ict] controllo -> $j4"
ALL_EVALS+=("$j1" "$j2" "$j3" "$j4")
ROWS+=(--row controllo "$EVAL_BFM/$(basename "$CTRL_ROOT")" "$CTRL_ROOT" "$ICT_STD")
ROWS+=(--row B "$EVAL_BFM/$(basename "$B_ROOT")" "$B_ROOT" "$ICT_STD")

for spec in "C|$C_ROOT" "E|$E_ROOT"; do
    IFS='|' read -r tag root <<< "$spec"
    tjob="$(job_of "$root")"
    grep -q "^arm=$tag$" "$root/ablation_arm.txt"
    jb="$(bfm_eval "$tag" "$root" afterok "$tjob")";                                   echo "[bfm] $tag -> $jb"
    ji="$(ict_eval "$tag" "${CK[$tag]}" afterok "$tjob $ICT_CHECK_JOB" afterany "$jb")"; echo "[ict] $tag -> $ji"
    ALL_EVALS+=("$jb" "$ji")
    ROWS+=(--row "$tag" "$EVAL_BFM/$(basename "$root")" "$root" "$ICT_RA1")
done

wrap="AAU_NV= $(printf '%q ' "$AAU_DIR/run.sh" aau/models/ablation_summary.py \
    --out "$AAU_RUNS/ablations_v3/summary.md" "${ROWS[@]}")"
dep_ids="$(printf '%s\n' "${ALL_EVALS[@]}" | grep -v '^DRY$' | paste -sd: || true)"
j_tab="$(submit sbatch --parsable --job-name=wbes-abl-summary --partition=cpu \
    --cpus-per-task=2 --mem=8G --time=00:30:00 \
    "--chdir=$WBES_ROOT" "--output=$AAU_LOGS/%x-%j.out" "--error=$AAU_LOGS/%x-%j.err" \
    ${dep_ids:+"--dependency=afterany:$dep_ids"} --wrap "$wrap")"
echo "[table] $j_tab dopo: ${ALL_EVALS[*]}"
