#!/usr/bin/env bash
#SBATCH --job-name=wbes-flame-zeroshot
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili. Sottometti
# con aau/submit.sh, che li ricalcola e li passa da riga di comando.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#
# WS-FLAME, zero-shot su un terzo 3DMM: un modello mai addestrato su FLAME valutato sulle
# identita' FLAME di flame_build.sbatch. Protocollo IDENTICO a aau/ict/ict_zeroshot_rank.sbatch
# (stessi due script, stessi common_args di eval_common.sh, stesse sigma):
#   1. compare_model_vs_chamfer_rankings.py           -> Spearman aggregato, clean e mixed
#   2. compare_model_vs_chamfer_topology_breakdown.py -> per coppia ordinata di topologie
#      (clean, cross_only) + pair_metrics.csv per i CI bootstrap
# Pool di 500 identita', `--max_subjects 500 --eval_fraction 0.2 --subject_split eval` con
# WBES_EVAL_SEED=1234: 100 soggetti, gli stessi per i due modelli (148.500 mesh pair cross).
#
# Un braccio per job, scelto da WBES_FLAME_ARM:
#   joint     BFM+ICT congiunto, job 1019532 (WBES_FLAME_CKPT_JOINT)
#   bfm_only  solo BFM, job 1019310 (WBES_FLAME_CKPT_BFM_ONLY)
# Output in $WBES_FLAME_RUNS/<impronta>/<arm>/flame_zeroshot (default aau/runs/ws_flame/...),
# con <impronta> = flame_fingerprint, cioe' i dati su cui e' stato valutato.
#
#   WBES_FLAME_ARM=joint    aau/submit.sh flame/flame_zeroshot.sbatch
#   WBES_FLAME_ARM=bfm_only aau/submit.sh flame/flame_zeroshot.sbatch
#
# SPAZIO: gli operatori non stanno in home. Il job li calcola dalle topologie compresse di
# $WBES_FLAME_TOPO_DIR sul /tmp del nodo ($WBES_FLAME_TMP/<arm>-<impronta>, flame_stage_ops in
# flame_env.sh), ci valuta sopra e li cancella all'uscita. /tmp e' RAM e conta contro --mem:
# su ICT erano ~60 MB per identita' a 9409 vertici, la patch FLAME e' piu' piccola.
#
# TEMPI: non misurati su FLAME. Riferimento ICT ~1h10m per cella (job 1019695) con patch da
# 9409 vertici, piu' gli operatori (~3 s per mesh a 9409 vertici su un core, job 1019281);
# la patch FLAME e' piu' piccola, quindi --time=3h e' largo.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/flame/flame_env.sh"
flame_require_view

WBES_FLAME_ARM="${WBES_FLAME_ARM:-joint}"
case "$WBES_FLAME_ARM" in
    joint)    CKPT="$WBES_FLAME_CKPT_JOINT" ;;
    bfm_only) CKPT="$WBES_FLAME_CKPT_BFM_ONLY" ;;
    *) echo "ERRORE: WBES_FLAME_ARM=$WBES_FLAME_ARM (joint|bfm_only)" >&2; exit 2 ;;
esac

WBES_EVAL_SCENARIOS="${WBES_EVAL_SCENARIOS:-clean,mixed}"
WBES_EVAL_STAGE="${WBES_EVAL_STAGE:-flame_zeroshot}"
# Impronta dei dati (flame_env.sh): nella out dir e nella data dir su /tmp, che eval_common.sh
# mette nell'hash e in eval_key.txt. Dati rigenerati con altri input => tutto rieseguito.
FP="$(flame_fingerprint)"
export WBES_EVAL_OUT="$WBES_FLAME_RUNS/$FP/$WBES_FLAME_ARM"
mkdir -p "$WBES_FLAME_RUNS/$FP"
cp "$WBES_FLAME_TOPO_DIR/$FLAME_STAMP_NAME" "$WBES_FLAME_RUNS/$FP/identities_manifest.json"
# Prima degli operatori: eval_common.sh lo controllerebbe anche lui, ma dopo minuti di calcolo.
if [[ -f "$WBES_EVAL_OUT/$WBES_EVAL_STAGE/.done" ]]; then
    echo "[flame-zs] $WBES_EVAL_OUT/$WBES_EVAL_STAGE gia' completo, niente da fare"
    exit 0
fi

STAGE_TMP="$WBES_FLAME_TMP/$WBES_FLAME_ARM-$FP"
trap 'rm -rf "$STAGE_TMP"' EXIT
cd "$WBES_ROOT"
flame_stage_ops "$STAGE_TMP"

# Assegnazioni SENZA ${...:-}: env.sh ha appena esportato i percorsi BFM e il checkpoint v1,
# e un default qui li lascerebbe passare (vedi la nota in flame_env.sh).
export WBES_DATA_DIR="$STAGE_TMP/view/npz_withops"
export WBES_DIST_NPZ="$WBES_FLAME_DIST_NPZ"
export WBES_CKPT="$CKPT"
export WBES_EVAL_SEED="${WBES_EVAL_SEED:-1234}"

# Nessun braccio di ablazione o di pilota: questi due checkpoint non hanno
# ablation_arm.txt/pilot_arm.txt, e un checkpoint che li avesse vorrebbe frame, token o
# operatori propri che qui non ci sono.
_runs_root="$(dirname "$(dirname "$(dirname "$(realpath "$WBES_CKPT")")")")"
for f in ablation_arm.txt pilot_arm.txt; do
    if [[ -f "$_runs_root/$f" ]]; then
        echo "ERRORE: $_runs_root/$f presente: questo sbatch non sa valutare quel braccio" >&2
        exit 2
    fi
done
unset _runs_root

source "$WBES_ROOT/aau/eval_common.sh"

echo "[flame-zs] arm=$WBES_FLAME_ARM impronta=$FP ckpt=$WBES_CKPT"
echo "[flame-zs] data=$WBES_DATA_DIR gt=$WBES_DIST_NPZ out=$OUT_ROOT"
aau_eval_begin "$WBES_EVAL_STAGE"

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

echo "[$(timestamp)] 2/2 breakdown per coppia di topologie (clean) + pair_metrics"
"$AAU_DIR/run.sh" "$PERT/compare_model_vs_chamfer_topology_breakdown.py" \
  "${common_args[@]}" \
  --out_dir "$STAGE_OUT/topology" \
  --topology_pair_mode cross_only \
  --ordered_topology_pairs \
  --write_pair_metrics \
  2>&1 | tee "$STAGE_OUT/topology.log"

aau_eval_done "$WBES_EVAL_STAGE"
