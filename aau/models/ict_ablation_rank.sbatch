#!/usr/bin/env bash
#SBATCH --job-name=wbes-abl-ict
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#
# Zero-shot ICT delle ablazioni v3, adattato da aau/ict/ict_zeroshot_rank.sbatch: stessi due
# script del repo (ranking aggregato + breakdown con pair_metrics.csv), stessi common_args di
# eval_common.sh, WBES_EVAL_SEED=1234, solo scenario clean. Il modello passa da
# aau/models/eval_ablation.py con frame, token e operatori del braccio (<runs_root>/ablation_arm.txt):
#   - frame rms: V_in ri-inquadrato nel forward (per questo SOLO clean);
#   - token: tabella ICT (WBES_ICT_TOKEN_JSON), standardizzata sui 4500 soggetti di training ICT;
#   - operatori robusti ad area 1 (C, E): la data dir diventa WBES_ICT_RA1_DIR, la vista
#     held-out con gli operatori robusti (stessi nomi; Chamfer sugli stessi vertici, perche' il
#     loader ricentra e divide per maxabs).
# Senza ablation_arm.txt (controllo) e' ict_zeroshot_rank.sbatch con lo scenario clean.
#
#   WBES_CKPT=<best_by_xtopo_mesh_clean.pth> WBES_EVAL_STAGE=ict_zeroshot_abl3_clean \
#     aau/submit.sh aau/models/ict_ablation_rank.sbatch
set -euo pipefail

# Variabili PROPRIE, non WBES_DATA_DIR/WBES_DIST_NPZ: aau/submit.sh sourcea env.sh, che le esporta
# sempre coi percorsi BFM, e il job le eredita. Con ${WBES_DATA_DIR:-...} l'eval "ICT" girava sui dati
# e sulla matrice BFM (job 1055122: 0 soggetti in comune fra la vista ICT robusta e la matrice BFM).
export WBES_DATA_DIR="${WBES_ICT_DATA_DIR:-$PWD/datasets/ICT/eval_view_heldout}"
export WBES_DIST_NPZ="${WBES_ICT_DIST_NPZ:-$PWD/datasets/ICT/train_ready/gt_matrix.npz}"
export WBES_EVAL_SEED="${WBES_EVAL_SEED:-1234}"
WBES_EVAL_STAGE="${WBES_EVAL_STAGE:-ict_zeroshot_abl3_clean}"
ICT_TOKEN_JSON="${WBES_ICT_TOKEN_JSON:-$PWD/aau/runs/ablations_v3/size_token_ict.json}"
ICT_RA1_DIR="${WBES_ICT_RA1_DIR:-$PWD/datasets/ICT/eval_view_heldout_robust_area1}"

: "${WBES_CKPT:?WBES_CKPT = il checkpoint del braccio}"
_runs_root="$(dirname "$(dirname "$(dirname "$(realpath "$WBES_CKPT")")")")"
FRAME=current
LAUNCH=()
if [[ -f "$_runs_root/ablation_arm.txt" ]]; then
    ARM="$(sed -n 's/^arm=//p' "$_runs_root/ablation_arm.txt")"
    FRAME="$(sed -n 's/^frame=//p' "$_runs_root/ablation_arm.txt")"
    TOK="$(sed -n 's/^size_token_json=//p' "$_runs_root/ablation_arm.txt")"
    LAUNCH=(aau/models/eval_ablation.py --frame "$FRAME")
    if [[ -n "$TOK" ]]; then
        [[ -f "$ICT_TOKEN_JSON" ]] || { echo "ERRORE: tabella token ICT assente: $ICT_TOKEN_JSON" >&2; exit 1; }
        LAUNCH+=(--size-token-json "$ICT_TOKEN_JSON")
    fi
    case "$ARM" in
        B) ;;
        C|E) export WBES_DATA_DIR="$ICT_RA1_DIR" ;;
        *) echo "ERRORE: arm '$ARM' in $_runs_root/ablation_arm.txt" >&2; exit 2 ;;
    esac
    LAUNCH+=(--)
    echo "[abl] arm=$ARM frame=$FRAME token=${TOK:+$ICT_TOKEN_JSON} data_dir=$WBES_DATA_DIR"
fi
unset _runs_root

source "${WBES_ROOT:-$PWD}/aau/eval_common.sh"

aau_eval_begin "$WBES_EVAL_STAGE"

echo "[$(timestamp)] 1/2 ranking aggregato, scenario clean"
"$AAU_DIR/run.sh" "${LAUNCH[@]}" "$PERT/compare_model_vs_chamfer_rankings.py" \
  "${common_args[@]}" \
  --out_dir "$STAGE_OUT/ranking" \
  --aggregation_level subject_pair_mean \
  --scenarios clean \
  2>&1 | tee "$STAGE_OUT/ranking.log"

echo "[$(timestamp)] 2/2 breakdown per coppia di topologie (clean) + pair_metrics"
"$AAU_DIR/run.sh" "${LAUNCH[@]}" "$PERT/compare_model_vs_chamfer_topology_breakdown.py" \
  "${common_args[@]}" \
  --out_dir "$STAGE_OUT/topology" \
  --topology_pair_mode cross_only \
  --ordered_topology_pairs \
  --write_pair_metrics \
  2>&1 | tee "$STAGE_OUT/topology.log"

aau_eval_done "$WBES_EVAL_STAGE"
