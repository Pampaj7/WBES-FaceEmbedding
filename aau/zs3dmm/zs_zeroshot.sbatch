#!/usr/bin/env bash
#SBATCH --job-name=wbes-zs-zeroshot
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili. Sottometti
# con aau/submit.sh, che li ricalcola e li passa da riga di comando.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#
# Zero-shot su un 3DMM mai visto (WBES_ZS_DOMAIN=hifi|fv): un modello mai addestrato su quel
# dominio valutato sulle identita' di zs_build.sbatch. Protocollo IDENTICO a
# aau/ict/ict_zeroshot_rank.sbatch (stessi due script, stessi common_args di eval_common.sh,
# stesse sigma):
#   1. compare_model_vs_chamfer_rankings.py           -> Spearman aggregato, clean e mixed
#   2. compare_model_vs_chamfer_topology_breakdown.py -> per coppia ordinata di topologie
#      (clean, cross_only) + pair_metrics.csv per i CI bootstrap (zs_summarize.py)
#
# UNA differenza, imposta dallo spazio (<= 15 GB nuovi nella home): gli operatori NON stanno
# nella home. Prima dell'eval:
#   a. zs_stage.py estrae dal pool di 500 i 100 soggetti che lo script di ranking estrarrebbe
#      (rebuild_subject_split, --eval_fraction 0.2 --max_subjects 500, WBES_EVAL_SEED=1234) e
#      mette le loro 600 mesh in /tmp del nodo;
#   b. v2_work/potential/areanorm_operators.py (k_eig 128, area unitaria: gli operatori dei tre
#      modelli) le porta in /tmp/.../ops;
#   c. l'eval gira su quella dir con --subject_split all --max_subjects 0 (come ws2_eval.sbatch):
#      stessi 100 soggetti, stesse 148.500 mesh pair cross.
# /tmp conta contro --mem: ~60 MB di operatori per identita' a risoluzione ICT/HIFI3D, ~3x per
# FaceVerse (passare --mem=160G). Si cancella all'uscita.
#
# Un braccio per job, scelto da WBES_ZS_ARM:
#   joint     BFM+ICT congiunto, job 1019532 (WBES_ZS_CKPT_JOINT)
#   bfm_only  solo BFM, job 1019310 (WBES_ZS_CKPT_BFM_ONLY)
#   ict_only  solo ICT, job 1019531 (WBES_ZS_CKPT_ICT_ONLY)
# Output in $ZS_RUNS/data_<fp>/<arm>/zs_zeroshot (fp = impronta dei manifest dei dati, zs_env.sh).
#
#   WBES_ZS_DOMAIN=hifi WBES_ZS_ARM=joint aau/submit.sh zs3dmm/zs_zeroshot.sbatch
#
# TEMPI: riferimento ICT ~1h10m per cella (job 1019695) con patch da 9409 vertici; HIFI3D ha
# 9518 vertici. FaceVerse sta alla risoluzione di BFM (~2.3 h per scenario di ranking + ~1 h di
# breakdown, README): passare --time=09:00:00.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net
zs_require_view

WBES_ZS_ARM="${WBES_ZS_ARM:-joint}"
case "$WBES_ZS_ARM" in
    joint)    CKPT="$WBES_ZS_CKPT_JOINT" ;;
    bfm_only) CKPT="$WBES_ZS_CKPT_BFM_ONLY" ;;
    ict_only) CKPT="$WBES_ZS_CKPT_ICT_ONLY" ;;
    *) echo "ERRORE: WBES_ZS_ARM=$WBES_ZS_ARM (joint|bfm_only|ict_only)" >&2; exit 2 ;;
esac
WBES_EVAL_STAGE="${WBES_EVAL_STAGE:-zs_zeroshot}"
WBES_EVAL_SCENARIOS="${WBES_EVAL_SCENARIOS:-clean,mixed}"
export WBES_EVAL_SEED="${WBES_EVAL_SEED:-1234}"
ZS_FP="$(zs_data_fp)"
# WBES_ZS_PART: all (default) = ranking + breakdown nello stesso job; ranking / topology = una
# sola delle due, in una out dir propria (<arm>_ranking, <arm>_topology) e con una /tmp propria,
# cosi' i due pezzi girano in parallelo su due GPU. Serve a FaceVerse, dove il Chamfer su tutti
# i vertici va a ~12 coppie/s (HIFI3D ~110) e ranking clean+mixed + breakdown in serie passano
# le 10 h. zs_summarize.py prende <arm>/ se completo, altrimenti i due pezzi.
WBES_ZS_PART="${WBES_ZS_PART:-all}"
case "$WBES_ZS_PART" in
    all) PART_SUFFIX="" ;;
    ranking|topology) PART_SUFFIX="_$WBES_ZS_PART" ;;
    *) echo "ERRORE: WBES_ZS_PART=$WBES_ZS_PART (all|ranking|topology)" >&2; exit 2 ;;
esac
export WBES_EVAL_OUT="$ZS_RUNS/data_$ZS_FP/$WBES_ZS_ARM$PART_SUFFIX"

# Nessun braccio di ablazione o di pilota: questi tre checkpoint non hanno
# ablation_arm.txt/pilot_arm.txt, e un checkpoint che li avesse vorrebbe frame, token o
# operatori propri che qui non ci sono.
_runs_root="$(dirname "$(dirname "$(dirname "$(realpath "$CKPT")")")")"
for f in ablation_arm.txt pilot_arm.txt; do
    if [[ -f "$_runs_root/$f" ]]; then
        echo "ERRORE: $_runs_root/$f presente: questo sbatch non sa valutare quel braccio" >&2
        exit 2
    fi
done
unset _runs_root

# Stesso controllo di aau_eval_begin, ma PRIMA dei minuti di operatori.
if [[ -f "$WBES_EVAL_OUT/$WBES_EVAL_STAGE/.done" ]]; then
    echo "[zs-zs] stage gia' completo: $WBES_EVAL_OUT/$WBES_EVAL_STAGE/.done"
    exit 0
fi

# Percorso FISSO (niente job id): eval_common.sh mette la data dir nella chiave della out dir,
# e una risottomissione deve ritrovare la stessa chiave.
STAGE_TMP="/tmp/wbes_zs_${WBES_ZS_DOMAIN}_${WBES_ZS_ARM}${PART_SUFFIX}"
rm -rf "$STAGE_TMP"
mkdir -p "$STAGE_TMP"
trap 'rm -rf "$STAGE_TMP"' EXIT

echo "[zs-zs] dominio=$WBES_ZS_DOMAIN ($ZS_LABEL) arm=$WBES_ZS_ARM dati=data_$ZS_FP host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[zs-zs] a. soggetti valutati -> $STAGE_TMP/in"
"$AAU_DIR/run.sh" aau/zs3dmm/zs_stage.py --view-dir "$ZS_DATA_DIR" \
    --out-dir "$STAGE_TMP/in" --seed "$WBES_EVAL_SEED"
mkdir -p "$WBES_EVAL_OUT"
cp "$STAGE_TMP/subjects.json" "$WBES_EVAL_OUT/subjects.json"

echo "[zs-zs] b. operatori (k_eig 128, area unitaria) su /tmp"
N_SHARDS="${SLURM_CPUS_PER_TASK:-8}"
# n coprimo con 6: i nomi sono soggetti x 6 topologie (vedi aau/ict/ict_ops.sbatch).
if (( N_SHARDS % 6 == 0 )); then
    N_SHARDS=$(( N_SHARDS - 1 ))
fi
SHARD_LOGS="$AAU_LOGS/${SLURM_JOB_NAME:-wbes-zs-zeroshot}-${SLURM_JOB_ID:-manual}.shards"
mkdir -p "$SHARD_LOGS" "$STAGE_TMP/ops"
pids=()
for (( i = 0; i < N_SHARDS; i++ )); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
      --input-dir "$STAGE_TMP/in" --output-dir "$STAGE_TMP/ops" \
      --k-eig 128 --shard "$i/$N_SHARDS" \
      > "$SHARD_LOGS/shard$i.log" 2>&1 &
    pids+=($!)
done
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "ERRORE: shard $i uscito != 0 (vedi $SHARD_LOGS/shard$i.log)" >&2
        exit 1
    fi
done
N_IN=$(find "$STAGE_TMP/in" -name "*.npz" | wc -l)
N_OPS=$(find "$STAGE_TMP/ops" -name "*.npz" | wc -l)
echo "[zs-zs] operatori: $N_OPS / $N_IN, $(du -sh "$STAGE_TMP/ops" | cut -f1) su /tmp"
if (( N_OPS != N_IN )); then
    echo "ERRORE: operatori incompleti" >&2
    exit 1
fi

# Assegnazioni SENZA ${...:-}: env.sh ha esportato i percorsi BFM e il checkpoint v1, e un
# default qui li lascerebbe passare (vedi la nota in hifi_env.sh).
export WBES_DATA_DIR="$STAGE_TMP/ops"
export WBES_DIST_NPZ="$ZS_DIST_NPZ"
export WBES_CKPT="$CKPT"

source "$WBES_ROOT/aau/eval_common.sh"
# La data dir e' gia' il set di eval: tutti i suoi soggetti, nessun ricampionamento.
common_args+=(--subject_split all --max_subjects 0)

echo "[zs-zs] c. eval ckpt=$WBES_CKPT"
echo "[zs-zs] data=$WBES_DATA_DIR gt=$WBES_DIST_NPZ out=$OUT_ROOT"
aau_eval_begin "$WBES_EVAL_STAGE"

if [[ "$WBES_ZS_PART" != topology ]]; then
echo "[$(timestamp)] 1/2 ranking aggregato, scenari: $WBES_EVAL_SCENARIOS"
"$AAU_DIR/run.sh" "$PERT/compare_model_vs_chamfer_rankings.py" \
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
fi

if [[ "$WBES_ZS_PART" != ranking ]]; then
echo "[$(timestamp)] 2/2 breakdown per coppia di topologie (clean) + pair_metrics"
"$AAU_DIR/run.sh" "$PERT/compare_model_vs_chamfer_topology_breakdown.py" \
  "${common_args[@]}" \
  --out_dir "$STAGE_OUT/topology" \
  --topology_pair_mode cross_only \
  --ordered_topology_pairs \
  --write_pair_metrics \
  2>&1 | tee "$STAGE_OUT/topology.log"
fi

aau_eval_done "$WBES_EVAL_STAGE"
