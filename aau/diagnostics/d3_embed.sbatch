#!/usr/bin/env bash
#SBATCH --job-name=wbes-d3-embed
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#
# Diagnostica D3 (PROTOCOL_D3.md sez. 3 e 9, emendamento 1 sez. 3): embedding [s, u] nuovi, per parte (D3_EMBED_PARTS,
# default tutte) e braccio (D3_ARMS, default i tre bracci):
#   famos  FaMoS TRAIN: eval_v3 + zs_embed con gli argomenti di diagnostics/embed.sbatch (D1), dopo D3_STEP=famos
#          (operatori in datasets/V3_OPS_CACHE/diag_d3/famos/ops, tabella e GT in d3/famos);
#   fvn    C3M e123 su FaceVerse NEUTRA (lo store neutro ha solo e205): il passo ``fvn`` di trainer/ablations/c3f/
#          eval_body.sh, come v3_work/faceverse_neutral/fvn_embed.sbatch per e205, con la tabella di scala ufficiale della
#          vista neutra e le uscite in d3/fvn (WBES_FVN_OUT); operatori dalla cache condivisa della vista;
#   pools  i pool NON valutati di FaceScape, HIFI3D, FaceVerse (emendamento 1), dopo D3_STEP=pools: stessi argomenti di
#          famos, tabelle di scala ufficiali delle viste, --dist_npz = la gt_matrix della vista (come zs_zeroshot).
# Job brevi sulle A100 di nv-ai-04 (unprivileged, autorizzato per job brevi): ripartibili, gli embedding gia' scritti si
# saltano (--requeue). Un job per braccio:
#   for a in factorized_s1234 factorized_s2345 factorizedc3m_e123; do
#     D3_ARMS=$a D3_EMBED_PARTS=pools sbatch --time=01:30:00 aau/diagnostics/d3_embed.sbatch; done
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
PARTS=" ${D3_EMBED_PARTS:-famos fvn pools} "
ARMS="${D3_ARMS:-factorized_s1234 factorized_s2345 factorizedc3m_e123}"
D3="$AAU_RUNS/evidence/diagnostics/d3"
ST="$AAU_RUNS/evidence/trainer_v3/factorized/scale_tables"
echo "[d3-embed] host=$(hostname) job=${SLURM_JOB_ID:-none} $(date +%T) parti=$PARTS bracci=$ARMS"
# Fino a 3 tentativi a 2 minuti di distanza: su nv-ai-04 la risoluzione dell'utente di dominio cade a tratti ("unknown
# userid" all'avvio di singularity, come in eval_body.sh); il primo tentativo del job 1068061 e' caduto cosi' (127).
retry() {
  local t
  for t in 1 2 3; do
    "$@" && return 0
    (( t < 3 )) && { echo "[d3-embed] $(date +%T) $1: tentativo $t fallito, riprovo fra 2 minuti" >&2; sleep 120; }
  done
  return 1
}
ckpt_of() { "$AAU_DIR/run.sh" v3_work/trainer/tools/fact_calib.py ckpt "$1" | tail -1; }
embed_one() {  # checkpoint, operatori, tabella di scala, GT per la scelta dei soggetti, uscita
  WBES_V3_SCALE_TABLES="$3" WBES_V3_FACTORIZED_OUT=full WBES_EVAL_SCENARIOS=clean \
    "$AAU_DIR/run.sh" v3_work/trainer/eval_v3.py -- aau/zs3dmm/zs_embed.py \
      --model_path "$1" --checkpoint_selector best_by_clean --data_dir "$2" --dist_npz "$4" \
      --subject_split all --eval_fraction 0.2 --max_subjects 0 --max_meshes_per_subject_eval 10 --preload_workers 8 \
      --pair_mode cross_topology --seed 1234 --embeddings_out "$5" 2>&1 | tee "$(dirname "$5")/embed.log"
}
embed_to() {  # chiave, operatori, tabella, GT, uscita: salta se c'e' gia'
  local ck
  [[ -f "$5" ]] && { echo "[d3-embed] $5: gia' presente"; return 0; }
  ck=$(retry ckpt_of "$1")
  [[ -f "$ck" ]] || { echo "ERRORE: checkpoint di $1 non trovato ($ck)"; return 1; }
  echo "[d3-embed] $(date +%T) $1 -> $5"
  mkdir -p "$(dirname "$5")"
  retry embed_one "$ck" "$2" "$3" "$4" "$5" || { echo "ERRORE: embedding $5"; return 1; }
  [[ -f "$5" ]] || { echo "ERRORE: $5 assente"; return 1; }
}
if [[ "$PARTS" == *" famos "* ]]; then
  [[ -f "$D3/famos/scale_table.npz" && -f "$D3/famos/gt_names.npz" ]] || { echo "ERRORE: prima D3_STEP=famos"; exit 1; }
  for key in $ARMS; do
    embed_to "$key" "$WBES_ROOT/datasets/V3_OPS_CACHE/diag_d3/famos/ops" "$D3/famos/scale_table.npz" \
      "$D3/famos/gt_names.npz" "$D3/famos/emb/$key/embeddings.npz"
  done
fi
if [[ "$PARTS" == *" fvn "* && " $ARMS " == *" factorizedc3m_e123 "* ]]; then
  FO="$D3/fvn"
  if find "$FO/embed" -path "*scale_v3factorizedc3mfulle123_flip_embed*" -name embeddings.npz 2>/dev/null | grep -q .; then
    echo "[d3-embed] C3M e123 FaceVerse neutra: embedding gia' presenti"
  else
    mkdir -p "$FO/scale_tables"
    cp "$AAU_RUNS/evidence/faceverse_neutral/scale_tables/fv_eval.npz" "$FO/scale_tables/fv_eval.npz"
    echo "[d3-embed] $(date +%T) C3M e123 FaceVerse neutra"
    B="/tmp/v3c3f_eval_body_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}.sh"
    cp v3_work/trainer/ablations/c3f/eval_body.sh "$B"
    WBES_V3_EVAL_STEPS=fvn WBES_FVN_OUT="$FO" WBES_V3_ARM=factorizedc3m WBES_V3_EVAL_EPOCHS=123 \
      WBES_V3_EVAL_RR="$AAU_RUNS/evidence/trainer_v3/factorized/c3m/runs" bash "$B"
  fi
fi
if [[ "$PARTS" == *" pools "* ]]; then
  for p in facescape hifi3d faceverse; do
    case "$p" in
      facescape) tab="$ST/devfs_eval.npz"; gt="$WBES_ROOT/datasets/DEV_FACESCAPE/eval_view/gt_matrix.npz" ;;
      hifi3d)    tab="$ST/hifi3d_eval.npz"; gt="$WBES_ROOT/datasets/HIFI3D/eval_view/gt_matrix.npz" ;;
      faceverse) tab="$AAU_RUNS/evidence/faceverse_neutral/scale_tables/fv_eval.npz"
                 gt="$WBES_ROOT/datasets/FACEVERSE_ZS/eval_view/gt_matrix.npz" ;;
    esac
    ops="$WBES_ROOT/datasets/V3_OPS_CACHE/diag_d3/pools/$p/ops"
    [[ -f "$D3/pools/$p/subjects.json" && -d "$ops" ]] || { echo "ERRORE: prima D3_STEP=pools ($p)"; exit 1; }
    for key in $ARMS; do
      embed_to "$key" "$ops" "$tab" "$gt" "$D3/pools/$p/emb/$key/embeddings.npz"
    done
  done
fi
echo "[d3-embed] $(date +%T) fatto"
