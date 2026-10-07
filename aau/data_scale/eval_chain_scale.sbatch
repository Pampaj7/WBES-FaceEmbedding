#!/usr/bin/env bash
#SBATCH --job-name=wbes-ds-evalchain
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#
# Catena di eval del run grande, accodata da launch_scale.sh con --dependency=afterok sul training:
# trova il checkpoint (best_by_xtopo_mesh_clean.pth, come le celle ws2) in WBES_RUNS_ROOT e sottomette
# le TRE celle con cui e' stato valutato il congiunto x3dmm_joint_bfm_ict_s1234_1019532, sulle stesse
# viste e con la stessa GT (aau/runs/ws2_cross3dmm, job 1054450/1054451/1054454):
#   joint__bfm   108 held-out BFM   GT BFM in uso          ~10 h (job 1054450)
#   joint__ict    89 held-out ICT   GT ICT in uso          ~1.3 h
#   joint__rexpr espressioni ICT    GT ICT, modo rexpr     ~35 min
# Poi lo zero-shot sui 3DMM mai visti, con la pipeline aau/zs3dmm e il braccio `scale` (checkpoint in
# WBES_ZS_CKPT_SCALE; i bracci joint/bfm_only/ict_only e le baseline esistono gia'):
#   HIFI3D, FaceVerse     zs_zeroshot.sbatch (Spearman) -> zs_summarize.sbatch
#   FaceVerse espressioni zs_zeroshot.sbatch nelle due convenzioni di frame (come il congiunto in
#                         ws_zs_expr.sh) -> zs_expr_summarize.sbatch (riconoscimento d'identita')
# I soggetti delle tre viste sono held-out anche nel run grande (sono nel set congelato). Lo Spearman
# non dipende dalla scala della GT, quindi la GT estesa non serve qui.
# Con WBES_DS_CANON attivo le viste hanno gli operatori del frame NON canonico: la cella BFM non e'
# confrontabile e non viene lanciata (va rifatta la vista dalla geometria specchiata).
#
# WBES_DS_DRY=1: stampa i tre comandi senza sottometterli (prova della catena dal frontend).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
SUBMIT=("$AAU_DIR/submit.sh")
[[ "${WBES_DS_DRY:-0}" == 1 ]] && SUBMIT=(echo "[dry]" "$AAU_DIR/submit.sh")

ROOT="${WBES_RUNS_ROOT:?WBES_RUNS_ROOT}"
# il training scrive la run dir effettiva (con eventuale suffisso _rerunN/_restartN) accanto al suo log
RR="$AAU_LOGS/wbes-ds-train-scale-${WBES_DS_TRAIN_JOB:-none}.runs_root"
[[ -f "$RR" ]] && ROOT="$(cat "$RR")"
CK=$(find "$ROOT" -path '*/checkpoints/best_by_xtopo_mesh_clean.pth' | head -1)
[[ -n "$CK" ]] || { echo "ERRORE: nessun best_by_xtopo_mesh_clean.pth sotto $ROOT" >&2; exit 1; }
V="$WBES_ROOT/datasets/WS2_CROSS3DMM"
BFM_GT="$WBES_ROOT/face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz"
ICT_GT="$WBES_ROOT/datasets/ICT/train_ready/gt_matrix.npz"
echo "[evalchain] ckpt=$CK"
if [[ "${WBES_DS_CANON:-null}" == "null" ]]; then
  [[ -d "$V/joint__bfm" ]] || { echo "ERRORE: vista $V/joint__bfm assente" >&2; exit 1; }
  WBES_CKPT="$CK" WBES_DATA_DIR="$V/joint__bfm" WBES_DIST_NPZ="$BFM_GT" \
    "${SUBMIT[@]}" cross3dmm/ws2_eval.sbatch --time=14:00:00 --job-name=wbes-ds-eval-bfm
else
  echo "[evalchain] WBES_DS_CANON=$WBES_DS_CANON: cella BFM NON lanciata (vista nel frame non canonico)"
fi
WBES_CKPT="$CK" WBES_DATA_DIR="$V/joint__ict" WBES_DIST_NPZ="$ICT_GT" \
  "${SUBMIT[@]}" cross3dmm/ws2_eval.sbatch --time=04:00:00 --job-name=wbes-ds-eval-ict
WBES_CKPT="$CK" WBES_WS2_MODE=rexpr WBES_DATA_DIR="$V/joint__rexpr" WBES_DIST_NPZ="$ICT_GT" \
  "${SUBMIT[@]}" cross3dmm/ws2_eval.sbatch --time=03:00:00 --job-name=wbes-ds-eval-rexpr

# --- zero-shot (aau/zs3dmm), braccio scale ---------------------------------------------------------
export WBES_ZS_CKPT_SCALE="$CK"
zs=()
for dom in hifi fv; do
  opts=(); [[ "$dom" == fv ]] && opts=(--time=09:00:00 --mem=160G)
  j=$(WBES_ZS_DOMAIN=$dom WBES_ZS_ARMS=scale "${SUBMIT[@]}" zs3dmm/zs_zeroshot.sbatch --parsable \
      --job-name="wbes-ds-zs-$dom" "${opts[@]}")
  echo "[evalchain] zero-shot $dom: job $j"
  sj=$(WBES_ZS_DOMAIN=$dom "${SUBMIT[@]}" zs3dmm/zs_summarize.sbatch --parsable --job-name="wbes-ds-zs-sum-$dom" \
       --dependency="afterok:$j")
  echo "[evalchain] tabella $dom: job $sj"
done
# FaceVerse con espressioni: solo breakdown + embedding (come ws_zs_expr.sh), nelle due convenzioni
ex=()
for conv in bfm ict; do
  if [[ "$conv" == bfm ]]; then fr=(WBES_ZS_FLIP_FACES=1 WBES_ZS_FRAME=); else fr=(WBES_ZS_FRAME=x,-y,-z WBES_ZS_FLIP_FACES=0); fi
  j=$(env WBES_ZS_DOMAIN=fv WBES_ZS_EXPR=1 WBES_ZS_PART=topology WBES_ZS_EMBED=1 WBES_ZS_ARMS=scale "${fr[@]}" \
      "${SUBMIT[@]}" zs3dmm/zs_zeroshot.sbatch --parsable --job-name="wbes-ds-zs-fvexpr-$conv" \
      --mem=160G --time=05:00:00)
  echo "[evalchain] FaceVerse espressioni, convenzione $conv: job $j"
  ex+=("$j")
done
if [[ "${WBES_DS_DRY:-0}" != 1 ]]; then
  sj=$(WBES_ZS_DOMAIN=fv WBES_ZS_EXPR=1 "$AAU_DIR/submit.sh" zs3dmm/zs_expr_summarize.sbatch --parsable \
       --job-name=wbes-ds-zs-expr-sum --dependency="afterok:$(IFS=:; echo "${ex[*]}")")
  echo "[evalchain] riconoscimento FaceVerse espressioni (zs_expr_summarize): job $sj"
else
  echo "[dry] zs_expr_summarize.sbatch dopo ${ex[*]}"
fi

