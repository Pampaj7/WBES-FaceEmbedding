#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws2-eval
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili, e un
# --output relativo viene risolto nella submit dir. Sottometti con aau/submit.sh, che li
# ricalcola e li passa da riga di comando.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#
# WS2, tabella cross-3DMM: una cella (modello x dominio) o la riga espressioni, sulle viste
# di aau/cross3dmm/ws2_views.py. Gli script sono gli stessi di ict_zeroshot_rank.sbatch e
# ict_rexpr_rank.sbatch, con UNA differenza: la vista contiene gia' e solo i soggetti
# held-out scelti per la cella, quindi si valutano tutti con
# `--subject_split all --max_subjects 0` invece di farli riestrarre allo script.
# Con `--subject_split eval` lo script rifarebbe uno split SUO sui soggetti della vista,
# che non e' quello del training (vedi ws2_views.py: per ICT-only e congiunto lo split del
# training e' sull'intera data dir, 5000 e 5500 soggetti).
#
# WBES_WS2_MODE=cell (default): come ict_zeroshot_rank.sbatch
#   1. compare_model_vs_chamfer_rankings.py           -> Spearman aggregato, clean e mixed
#   2. compare_model_vs_chamfer_topology_breakdown.py -> per coppia di topologie (clean)
#      + pair_metrics.csv, da cui ws2_summarize.py ricava riga mesh-pair e CI bootstrap
#   WBES_WS2_SKIP_RANKING=1 salta il passo 1 (BFM-only su BFM: il ranking esiste gia',
#   manca solo il breakdown con le pair_metrics).
#
#   V=$PWD/datasets/WS2_CROSS3DMM
#   CK=aau/runs/x3dmm_ict_only_s1234_1019531/mixed_xtopo_*_seed1234__9a81466d/checkpoints/best_by_xtopo_mesh_clean.pth
#   WBES_DATA_DIR=$V/ict_only__ict WBES_DIST_NPZ=$PWD/datasets/ICT/train_ready/gt_matrix.npz \
#   WBES_CKPT=$CK aau/submit.sh cross3dmm/ws2_eval.sbatch
#
# WBES_WS2_MODE=rexpr: regime (c) misto di ict_rexpr_rank.sbatch (mesh A con espressione
# k, mesh B con k' != k, cross_topology) piu' la baseline neutra-contro-neutra sugli stessi
# soggetti (within_topology), che per il BFM-only era il job 1019710. WBES_DATA_DIR e' la
# radice <modello>__rexpr con le sottodir mixed/ e neutral/; GT delle identita' NEUTRE.
#
# TEMPI: ICT ~1h10m per cella (zero-shot, job 1019695: 148.500 pair, ranking clean+mixed +
# breakdown); BFM ~2.3 h per scenario di ranking + ~1 h di breakdown (README), cioe' ~6 h:
# per le celle BFM passare --time=09:00:00. rexpr: ~35 min per mixed (job 1019721).
WBES_WS2_MODE="${WBES_WS2_MODE:-cell}"
WBES_EVAL_STAGE="${WBES_EVAL_STAGE:-ws2_${WBES_WS2_MODE}}"
WBES_EVAL_SCENARIOS="${WBES_EVAL_SCENARIOS:-clean,mixed}"

source "${WBES_ROOT:-$PWD}/aau/eval_common.sh"

# La vista e' gia' il set di eval: tutti i suoi soggetti, nessun ricampionamento.
# Dopo common_args, quindi vince (argparse tiene l'ultima occorrenza), anche per il
# referto di eval_split_info.py in aau_eval_begin.
common_args+=(--subject_split all --max_subjects 0)

aau_eval_begin "$WBES_EVAL_STAGE"

ranking_run() {  # <data_dir> <out_dir> <log> [args...]
    local data_dir="$1" out_dir="$2" log="$3"
    shift 3
    mkdir -p "$out_dir"
    "$AAU_DIR/run.sh" "$PERT/compare_model_vs_chamfer_rankings.py" \
      "${common_args[@]}" \
      --data_dir "$data_dir" \
      --out_dir "$out_dir" \
      --aggregation_level subject_pair_mean \
      "$@" \
      2>&1 | tee "$log"
}

breakdown_run() {  # <data_dir> <out_dir> <log> [args...]
    local data_dir="$1" out_dir="$2" log="$3"
    shift 3
    mkdir -p "$out_dir"
    "$AAU_DIR/run.sh" "$PERT/compare_model_vs_chamfer_topology_breakdown.py" \
      "${common_args[@]}" \
      --data_dir "$data_dir" \
      --out_dir "$out_dir" \
      --ordered_topology_pairs \
      --write_pair_metrics \
      "$@" \
      2>&1 | tee "$log"
}

if [[ "$WBES_WS2_MODE" == "cell" ]]; then
    if [[ "${WBES_WS2_SKIP_RANKING:-0}" == "1" ]]; then
        echo "[$(timestamp)] 1/2 ranking saltato (WBES_WS2_SKIP_RANKING=1)"
    else
        echo "[$(timestamp)] 1/2 ranking aggregato, scenari: $WBES_EVAL_SCENARIOS"
        ranking_run "$WBES_DATA_DIR" "$STAGE_OUT/ranking" "$STAGE_OUT/ranking.log" \
          --scenarios "$WBES_EVAL_SCENARIOS" \
          --jitter_sigma 0.05 \
          --translation_sigma 0.05 \
          --rotation_sigma 0.05 \
          --mixed_jitter_sigma 0.05 \
          --mixed_translation_sigma 0.05 \
          --mixed_rotation_sigma 0.05
    fi
    echo "[$(timestamp)] 2/2 breakdown per coppia di topologie (clean) + pair_metrics"
    breakdown_run "$WBES_DATA_DIR" "$STAGE_OUT/topology" "$STAGE_OUT/topology.log" \
      --topology_pair_mode cross_only
elif [[ "$WBES_WS2_MODE" == "rexpr" ]]; then
    for view in mixed neutral; do
        if [[ ! -d "$WBES_DATA_DIR/$view" ]]; then
            echo "ERRORE: vista mancante: $WBES_DATA_DIR/$view (lancia ws2_views.sbatch)" >&2
            exit 1
        fi
    done
    # mixed: coppie a etichetta diversa (k != k'); neutral: una sola etichetta, quindi
    # within_topology e la diagonale nel breakdown, come same_k in ict_rexpr_rank.sbatch.
    for spec in "mixed|cross_topology|cross_only" "neutral|within_topology|all"; do
        IFS='|' read -r view pair_mode topo_mode <<< "$spec"
        echo "[$(timestamp)] $view 1/2 ranking"
        ranking_run "$WBES_DATA_DIR/$view" "$STAGE_OUT/$view/ranking" "$STAGE_OUT/$view/ranking.log" \
          --scenarios clean --pair_mode "$pair_mode"
        echo "[$(timestamp)] $view 2/2 breakdown + pair_metrics"
        breakdown_run "$WBES_DATA_DIR/$view" "$STAGE_OUT/$view/topology" "$STAGE_OUT/$view/topology.log" \
          --pair_mode "$pair_mode" --topology_pair_mode "$topo_mode"
    done
else
    echo "ERRORE: WBES_WS2_MODE=$WBES_WS2_MODE (cell|rexpr)" >&2
    exit 1
fi

aau_eval_done "$WBES_EVAL_STAGE"
