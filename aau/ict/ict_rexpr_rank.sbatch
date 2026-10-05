#!/usr/bin/env bash
#SBATCH --job-name=wbes-ict-rexpr
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
#SBATCH --time=06:00:00
#
# WS5 rifatto: gli stessi script di ranking del repo sulle espressioni CASUALI per
# soggetto (aau/ict/ict_expressions_random.py), sulle viste di aau/ict/ict_rexpr_views.py.
# Tre regimi, 11 condizioni in tutto:
#
#   same_k/k1..k5           --pair_mode within_topology
#                           ogni mesh porta la PROPRIA espressione k, che e' un vettore
#                           diverso da soggetto a soggetto -- non e' piu' la deformazione
#                           di modo comune del vecchio ict_expr_rank.sbatch
#   expr_vs_neutral/k1..k5  --pair_mode cross_topology sulla vista a due etichette
#                           (espressione k contro neutra)
#   mixed/all               --pair_mode cross_topology sulla vista a cinque etichette:
#                           mesh A con espressione k, mesh B con k' != k. E' il caso
#                           reale, ed e' il regime che il WS5 vecchio non aveva.
#
# Per ogni condizione girano i DUE script di WS2 (come ict_zeroshot_rank.sbatch):
#   1. compare_model_vs_chamfer_rankings.py           -> Spearman aggregato (clean)
#   2. compare_model_vs_chamfer_topology_breakdown.py -> pair_metrics.csv, da cui
#      aau/ict/ict_rexpr_summarize.py ricava i CI bootstrap con la funzione del paper.
# Senza il secondo la tabella non avrebbe intervalli di confidenza: lo script di ranking
# scrive solo il punto.
#
# La GT resta quella delle identita' NEUTRE: la domanda e' se il ranking di identita'
# sopravvive all'espressione, quindi il riferimento non deve muoversi con l'espressione.
#
#   CK=aau/runs/remesh_v1recipe_areanorm_s1234_1019310/mixed_xtopo_*_seed1234__9a81466d
#   WBES_DATA_DIR=$PWD/datasets/ICT/rexpr_view \
#   WBES_DIST_NPZ=$PWD/datasets/ICT/train_ready/gt_matrix.npz \
#   WBES_CKPT=$CK/checkpoints/best_by_xtopo_mesh_clean.pth \
#     aau/submit.sh ict/ict_rexpr_rank.sbatch
#
# WBES_DATA_DIR e' la RADICE delle viste: serve solo alla chiave e ai preflight di
# eval_common.sh, la data dir vera di ogni run e' la sottodir della condizione, passata
# come --data_dir DOPO "${common_args[@]}" (argparse tiene l'ultima occorrenza).
# Ogni run e' indipendente e riparte da sola: risottomettere salta le condizioni gia'
# chiuse, cosi' un kill per walltime non butta via le altre.
#
# TEMPI: 4950 coppie di soggetti (100 held-out, pool 500, seed 1234), quindi 4950 mesh
# pair per same_k, 9900 per expr_vs_neutral e 99.000 per mixed (20 coppie di etichette
# per coppia di soggetti). Il vecchio job WS5 ha misurato ~1 min per run da 4950-9900
# coppie su L40S (job 1019698, 30 run in 38m44s) e il breakdown di WS2 23 min per
# 148.500 coppie (job 1019695): la stima e' ~25 min per le 10 condizioni piccole x 2
# script, piu' ~35 min per mixed. --time=6h e' largo, resta corto per il backfill.
WBES_EVAL_STAGE="${WBES_EVAL_STAGE:-ict_rexpr}"
WBES_ICT_REXPR_VIEW="${WBES_ICT_REXPR_VIEW:-${WBES_DATA_DIR:-}}"

# Stesso seed fisso del job WS2 (aau/ict/ict_zeroshot_rank.sbatch, dove il perche' e'
# scritto per esteso): ICT non e' nel training di nessuno dei modelli BFM, e le 11
# condizioni di questo job devono guardare gli STESSI 100 soggetti del WS2 e del WS5
# vecchio, altrimenti la tabella confronterebbe campioni diversi da riga a riga.
export WBES_EVAL_SEED="${WBES_EVAL_SEED:-1234}"

source "${WBES_ROOT:-$PWD}/aau/eval_common.sh"

aau_eval_begin "$WBES_EVAL_STAGE"

if [[ ! -d "$WBES_ICT_REXPR_VIEW/mixed" ]]; then
    echo "ERRORE: viste non trovate sotto $WBES_ICT_REXPR_VIEW (lancia aau/ict/ict_rexpr_views.py)" >&2
    exit 1
fi

# regime|condizione|sottodir della vista. Una riga per run: il ciclo sotto non deve
# indovinare niente dai nomi delle directory.
runs=()
for k in 1 2 3 4 5; do
    runs+=("same_k|k$k|same_k$k" "expr_vs_neutral|k$k|neutral_k$k")
done
runs+=("mixed|all|mixed")

echo "[$(timestamp)] ${#runs[@]} condizioni su $WBES_ICT_REXPR_VIEW"

n_run=0
n_skip=0
for spec in "${runs[@]}"; do
    IFS='|' read -r regime cond view <<< "$spec"
    data_dir="$WBES_ICT_REXPR_VIEW/$view"
    out_dir="$STAGE_OUT/$regime/$cond"
    if [[ ! -d "$data_dir" ]]; then
        echo "ERRORE: vista mancante: $data_dir" >&2
        exit 1
    fi

    # same_k ha una sola etichetta per vista, quindi le coppie sono tutte diagonali e il
    # breakdown deve enumerare anche la diagonale (topology_pair_mode all). Negli altri
    # due regimi le coppie sono per costruzione a etichette diverse: cross_only.
    if [[ "$regime" == "same_k" ]]; then
        mode_args=(--pair_mode within_topology)
        topo_pair_args=(--topology_pair_mode all)
    else
        mode_args=(--pair_mode cross_topology)
        topo_pair_args=(--topology_pair_mode cross_only)
    fi

    if [[ -f "$out_dir/ranking/ranking_summary.json" ]]; then
        echo "[$(timestamp)] skip $regime/$cond ranking (ranking_summary.json presente)"
        n_skip=$(( n_skip + 1 ))
    else
        mkdir -p "$out_dir/ranking"
        echo "[$(timestamp)] $regime/$cond 1/2 ranking -> $out_dir/ranking"
        "$AAU_DIR/run.sh" "$PERT/compare_model_vs_chamfer_rankings.py" \
          "${common_args[@]}" \
          --data_dir "$data_dir" \
          --out_dir "$out_dir/ranking" \
          --aggregation_level subject_pair_mean \
          --scenarios clean \
          "${mode_args[@]}" \
          2>&1 | tee "$out_dir/ranking.log"
        n_run=$(( n_run + 1 ))
    fi

    if [[ -f "$out_dir/topology/topology_breakdown_summary.csv" ]]; then
        echo "[$(timestamp)] skip $regime/$cond breakdown (summary csv presente)"
        n_skip=$(( n_skip + 1 ))
    else
        mkdir -p "$out_dir/topology"
        echo "[$(timestamp)] $regime/$cond 2/2 breakdown + pair_metrics -> $out_dir/topology"
        "$AAU_DIR/run.sh" "$PERT/compare_model_vs_chamfer_topology_breakdown.py" \
          "${common_args[@]}" \
          --data_dir "$data_dir" \
          --out_dir "$out_dir/topology" \
          "${mode_args[@]}" \
          "${topo_pair_args[@]}" \
          --ordered_topology_pairs \
          --write_pair_metrics \
          2>&1 | tee "$out_dir/topology.log"
        n_run=$(( n_run + 1 ))
    fi
done

echo "[$(timestamp)] run eseguite: $n_run, saltate: $n_skip"
aau_eval_done "$WBES_EVAL_STAGE"
