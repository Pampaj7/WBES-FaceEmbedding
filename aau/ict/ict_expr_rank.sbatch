#!/usr/bin/env bash
#SBATCH --job-name=wbes-ict-expr
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili, e un
# --output relativo viene risolto nella submit dir. Sottometti con aau/submit.sh, che li
# ricalcola e li passa da riga di comando.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#
# WS5, espressioni: lo stesso script di ranking del repo, una volta per ogni combinazione
# (espressione, intensita') x regime, sulle viste di aau/ict/make_ict_eval_views.py.
# 15 condizioni x 2 regimi = 30 run:
#
#   same-expression      --pair_mode within_topology --topology_labels original
#                        entrambe le mesh della coppia portano la STESSA espressione
#   expression-vs-neutral --pair_mode cross_topology
#                        le sole coppie a etichetta diversa sono espressiva_i x neutra_j
#                        e neutra_i x espressiva_j, cioe' esattamente questo regime
#
# La GT resta quella delle identita' NEUTRE: la domanda e' se il ranking di identita'
# sopravvive all'espressione, quindi il riferimento non deve muoversi con l'espressione.
#
#   WBES_DATA_DIR=$PWD/datasets/ICT/expr_view \
#   WBES_DIST_NPZ=$PWD/datasets/ICT/train_ready/gt_matrix.npz \
#   WBES_CKPT=<...>/checkpoints/best_by_xtopo_mesh_clean.pth \
#     aau/submit.sh ict/ict_expr_rank.sbatch
#
# WBES_DATA_DIR e' la RADICE delle viste: serve solo alla chiave e ai preflight di
# eval_common.sh, la data dir vera di ogni run e' la sottodir della condizione, passata
# come --data_dir DOPO "${common_args[@]}" (argparse tiene l'ultima occorrenza).
# Ogni run e' indipendente e riparte da sola: risottomettere salta le condizioni che
# hanno gia' ranking_summary.json, cosi' un kill per walltime non butta via le altre.
#
# TEMPI: 4950 coppie di soggetti per condizione (100 soggetti held-out, gli stessi del
# job WS2 perche' il pool e' sempre 500 e il seed e' 1234), cioe' 4950 mesh pair per
# same-expression e 9900 per expression-vs-neutral: due ordini di grandezza meno delle
# 148.500 di uno scenario di ranking completo. --time=10h e' largo di proposito, il costo
# per run non era misurato prima di questo job.
WBES_EVAL_STAGE="${WBES_EVAL_STAGE:-ict_expr}"
WBES_ICT_EXPR_VIEW="${WBES_ICT_EXPR_VIEW:-${WBES_DATA_DIR:-}}"

# Stesso seed fisso del job WS2 (aau/ict/ict_zeroshot_rank.sbatch, dove il perche' e'
# scritto per esteso): ICT non e' nel training di nessuno dei modelli BFM, e le 30 run di
# questo job devono guardare gli STESSI 100 soggetti, altrimenti la tabella espressione x
# intensita' confronterebbe campioni diversi da cella a cella.
export WBES_EVAL_SEED="${WBES_EVAL_SEED:-1234}"

source "${WBES_ROOT:-$PWD}/aau/eval_common.sh"

aau_eval_begin "$WBES_EVAL_STAGE"

mapfile -t conditions < <(find "$WBES_ICT_EXPR_VIEW" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if (( ${#conditions[@]} == 0 )); then
    echo "ERRORE: nessuna vista sotto $WBES_ICT_EXPR_VIEW (lancia aau/ict/make_ict_eval_views.py)" >&2
    exit 1
fi
echo "[$(timestamp)] ${#conditions[@]} condizioni: ${conditions[*]}"

n_run=0
n_skip=0
for cond in "${conditions[@]}"; do
    for mode in same_expression expression_vs_neutral; do
        out_dir="$STAGE_OUT/$cond/$mode"
        if [[ -f "$out_dir/ranking_summary.json" ]]; then
            echo "[$(timestamp)] skip $cond/$mode (ranking_summary.json presente)"
            n_skip=$(( n_skip + 1 ))
            continue
        fi
        if [[ "$mode" == "same_expression" ]]; then
            mode_args=(--pair_mode within_topology --topology_labels original)
        else
            mode_args=(--pair_mode cross_topology)
        fi
        mkdir -p "$out_dir"
        echo "[$(timestamp)] $cond/$mode -> $out_dir"
        "$AAU_DIR/run.sh" "$PERT/compare_model_vs_chamfer_rankings.py" \
          "${common_args[@]}" \
          --data_dir "$WBES_ICT_EXPR_VIEW/$cond" \
          --out_dir "$out_dir" \
          --aggregation_level subject_pair_mean \
          --scenarios clean \
          "${mode_args[@]}" \
          2>&1 | tee "$out_dir/run.log"
        n_run=$(( n_run + 1 ))
    done
done

echo "[$(timestamp)] run eseguite: $n_run, saltate: $n_skip"
aau_eval_done "$WBES_EVAL_STAGE"
