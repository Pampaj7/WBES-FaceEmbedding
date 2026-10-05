#!/usr/bin/env bash
#SBATCH --job-name=wbes-ict-zeroshot
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
#SBATCH --time=08:00:00
#
# WS2, cross-3DMM zero-shot: un modello addestrato su BFM/REMESH valutato sui dati ICT.
# Due script del repo nello stesso job, sulla stessa selezione di soggetti:
#   1. compare_model_vs_chamfer_rankings.py           -> Spearman aggregato, clean e mixed
#   2. compare_model_vs_chamfer_topology_breakdown.py -> stesso numero per coppia di
#      topologie, piu' pair_metrics.csv (una riga per coppia di soggetti) da cui
#      aau/ict/ict_summarize.py ricava i CI bootstrap con la funzione del paper.
# Il breakdown gira solo sullo scenario CLEAN: non applica perturbazioni (usa
# _collect_clean_embeddings_and_vertices). Per `mixed` resta il numero aggregato di 1.
#
# Tutto passa da WBES_DATA_DIR / WBES_DIST_NPZ / WBES_CKPT, come i tre eval_*.sbatch:
#
#   CK=aau/runs/remesh_v1recipe_areanorm_s1234_1019310/mixed_xtopo_*_seed1234__9a81466d
#   WBES_DATA_DIR=$PWD/datasets/ICT/eval_view_heldout \
#   WBES_DIST_NPZ=$PWD/datasets/ICT/train_ready/gt_matrix.npz \
#   WBES_CKPT=$CK/checkpoints/best_by_xtopo_mesh_clean.pth \
#     aau/submit.sh ict/ict_zeroshot_rank.sbatch
#
# La data dir e' la VISTA HELD-OUT (aau/ict/make_ict_eval_views.py), non
# train_ready/npz_withops: gli script rifanno il loro split sui soggetti che trovano, e
# sui 5000 di train_ready l'80% dei soggetti valutati sarebbe di training ICT. Con la
# vista il pool e' 500 e i common_args di eval_common.sh (--max_subjects 500
# --eval_fraction 0.2 --subject_split eval) piu' il WBES_EVAL_SEED=1234 di qui sotto ne
# estraggono 100 in modo deterministico: 148.500 mesh pair cross-topology, lo stesso
# conteggio del numero storico BFM->FLAME 0.478 (v2_work/STATUS.md:398).
#
# TEMPI: non misurati su ICT prima di questo job. Il riferimento e' la run REMESH (2.3 h
# per scenario di ranking + 1 h di breakdown su L40S), ma le mesh ICT hanno 2,5x meno
# vertici (original 9409 contro 23470) e il Chamfer e' quadratico nei vertici: la stima e'
# ben sotto l'ora per scenario. --time=8h lascia margine restando corto per il backfill.
WBES_EVAL_SCENARIOS="${WBES_EVAL_SCENARIOS:-clean,mixed}"
WBES_EVAL_STAGE="${WBES_EVAL_STAGE:-ict_zeroshot}"

# Seed dello split FISSATO a 1234 per tutti e tre i checkpoint, al contrario degli eval
# in-domain, dove eval_common.sh lo lascia prendere dal config.json del checkpoint perche'
# con un seed fisso i modelli 2345 e 3456 verrebbero valutati su soggetti BFM che hanno
# visto in training. Qui il dataset e' ICT e nessuno dei tre modelli ne ha visto un
# vertice: non c'e' niente da tenere fuori, e un seed fisso serve, perche' da' ai tre
# modelli LA STESSA selezione di 100 soggetti (e la stessa, deterministica, del job
# WS5 sulle espressioni). Senza, ogni seed guarderebbe 100 soggetti diversi e la
# differenza fra i tre numeri sarebbe in parte differenza di campione.
#
# NOTA: WBES_EVAL_SEED entra nell'hash della out dir (eval_common.sh:54). I risultati dei
# job 1019695/6/7, girati mentre eval_common.sh aveva ancora --seed 1234 dentro
# common_args, stanno nelle out dir con l'hash SENZA seed; i numeri sono gli stessi (stesso
# seed, stesso split), ma una risottomissione oggi scrive in una out dir nuova invece di
# fermarsi sulla sentinella .done di quelle.
export WBES_EVAL_SEED="${WBES_EVAL_SEED:-1234}"

# WBES_CKPT puo' essere anche il runs_root di un training, come in eval_frame_topology.sbatch:
# serve alla coda per i training ancora in attesa, la cui run dir col fingerprint non esiste
# ancora. Si risolve qui, all'avvio, nell'unico <rundir>/checkpoints/best_by_xtopo_mesh_clean.pth.
if [[ -n "${WBES_CKPT:-}" && -d "$WBES_CKPT" ]]; then
    _matches=("$WBES_CKPT"/*/checkpoints/best_by_xtopo_mesh_clean.pth)
    if [[ ${#_matches[@]} -ne 1 || ! -f "${_matches[0]}" ]]; then
        echo "ERRORE: in $WBES_CKPT attesa una sola */checkpoints/best_by_xtopo_mesh_clean.pth, trovate: ${_matches[*]}" >&2
        exit 2
    fi
    export WBES_CKPT="${_matches[0]}"
    unset _matches
fi

# Bracci del pilota pot (aau/train_pilot_pot.sbatch scrive <runs_root>/pilot_arm.txt): il
# modello passa da aau/models/eval_pilot.py con gli operatori col pozzo DI QUESTO dataset,
# WBES_POT_DIR (per ICT datasets/ICT/eval_view_heldout_pot055, aau/precompute_pot.sbatch).
# masked: la data dir diventa WBES_POT_DIR (stessi nomi, stessi vertici, e Chamfer sugli stessi
# vertici); dual: data dir standard + --dual-ops-dir. Nessun pilot_arm.txt = come prima.
PILOT=()
_runs_root=""
# WBES_CKPT vuota = default di env.sh (il v1), che non e' un braccio del pilota.
[[ -n "${WBES_CKPT:-}" ]] && _runs_root="$(dirname "$(dirname "$(dirname "$(realpath "$WBES_CKPT")")")")"
if [[ -n "$_runs_root" && -f "$_runs_root/pilot_arm.txt" ]]; then
    PILOT_ARM="$(sed -n 's/^arm=//p' "$_runs_root/pilot_arm.txt")"
    : "${WBES_POT_DIR:?checkpoint di un braccio del pilota pot: serve WBES_POT_DIR}"
    case "$PILOT_ARM" in
        masked) export WBES_DATA_DIR="$WBES_POT_DIR"
                PILOT=(aau/models/eval_pilot.py --pilot-arm masked --) ;;
        dual)   PILOT=(aau/models/eval_pilot.py --pilot-arm dual --dual-ops-dir "$WBES_POT_DIR" --) ;;
        *) echo "ERRORE: arm '$PILOT_ARM' in $_runs_root/pilot_arm.txt" >&2; exit 2 ;;
    esac
    echo "[pilot] arm=$PILOT_ARM pot_dir=$WBES_POT_DIR data_dir=$WBES_DATA_DIR"
fi
unset _runs_root

source "${WBES_ROOT:-$PWD}/aau/eval_common.sh"

# Un solo stage per i due script: aau_eval_begin esce 0 se trova <stage>/.done, quindi due
# stage nello stesso job si sarebbero interrotti a vicenda al primo gia' fatto.
aau_eval_begin "$WBES_EVAL_STAGE"

echo "[$(timestamp)] 1/2 ranking aggregato, scenari: $WBES_EVAL_SCENARIOS"
"$AAU_DIR/run.sh" "${PILOT[@]}" "$PERT/compare_model_vs_chamfer_rankings.py" \
  "${common_args[@]}" \
  --out_dir "$STAGE_OUT/ranking" \
  --aggregation_level subject_pair_mean \
  --scenarios "$WBES_EVAL_SCENARIOS" \
  --jitter_sigma 0.05 \
  --translation_sigma 0.05 \
  --rotation_sigma 0.05 \
  --mixed_jitter_sigma 0.05 \
  --mixed_translation_sigma 0.05 \
  --mixed_rotation_sigma 0.05 \
  2>&1 | tee "$STAGE_OUT/ranking.log"

echo "[$(timestamp)] 2/2 breakdown per coppia di topologie (clean) + pair_metrics"
"$AAU_DIR/run.sh" "${PILOT[@]}" "$PERT/compare_model_vs_chamfer_topology_breakdown.py" \
  "${common_args[@]}" \
  --out_dir "$STAGE_OUT/topology" \
  --topology_pair_mode cross_only \
  --ordered_topology_pairs \
  --write_pair_metrics \
  2>&1 | tee "$STAGE_OUT/topology.log"

aau_eval_done "$WBES_EVAL_STAGE"
