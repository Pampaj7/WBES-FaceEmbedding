#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-abl-eval
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#
# Eval fuori dominio dei bracci di ablazione v3 con la pipeline zero-shot del run su scala
# (aau/zs3dmm/zs_zeroshot.sbatch, NON modificato): se ne esegue una copia su /tmp in cui il prefisso LAUNCH
# punta a v3_work/trainer/eval_v3.py, che ricostruisce i modelli v3 dagli args del checkpoint (pooling
# area_attn, sqrt(area) nel forward). Bracci "scale_v3<tag>", quindi le uscite stanno accanto a quelle del
# run su scala (scale_e036...) in $ZS_RUNS/data_<impronta>/.
#
#   WBES_V3_EVAL="ctrl=<ckpt> pool=<ckpt> norm=<ckpt>"          tag minuscoli alfanumerici
#   HIFI3D (distanza graduata senza crop + breakdown + embedding per il rank-1):
#     WBES_ZS_DOMAIN=hifi WBES_V3_EVAL=... aau/submit.sh v3_work/trainer/ablations/eval_ablation_v3.sbatch
#   FaceVerse con espressioni, convenzione BFM (come fvexpr del run su scala) o ICT:
#     WBES_ZS_DOMAIN=fv WBES_ZS_EXPR=1 WBES_ZS_FLIP_FACES=1 ...        (BFM, suffisso _flip)
#     WBES_ZS_DOMAIN=fv WBES_ZS_EXPR=1 WBES_ZS_FRAME=x,-y,-z ...       (ICT, suffisso _frame-xmymz)
#   FaceScape dev: datasets/DEV_FACESCAPE/ (agente D1) non e' ancora un dominio di zs_env.sh: da agganciare
#   quando esiste.
# Default: WBES_ZS_PART=topology (breakdown per coppia di topologie, scenario clean, pair_metrics) e
# WBES_ZS_EMBED=1 (embedding di ogni mesh, per retrieval e verifica). Scenari: solo clean (sqrt(area) nel
# forward non vale su input perturbati). Checkpoint consigliato: l'ultimo EMA (epochNNN_ema.pth).
# Con piu' GPU (--gres=gpu:l40s:N) i bracci girano in parallelo, uno per GPU, come in zs_zeroshot.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
ARMS=()
for pair in ${WBES_V3_EVAL:?WBES_V3_EVAL=\"tag=checkpoint ...\"}; do
  tag="${pair%%=*}"; ck="${pair#*=}"
  [[ "$tag" =~ ^[a-z0-9]+$ ]] || { echo "ERRORE: tag '$tag' (minuscole e cifre)" >&2; exit 2; }
  [[ -f "$ck" ]] || { echo "ERRORE: checkpoint $ck assente" >&2; exit 2; }
  export "WBES_ZS_CKPT_SCALE_V3$(echo "$tag" | tr 'a-z' 'A-Z')=$ck"
  ARMS+=("scale_v3$tag")
done
export WBES_ZS_ARMS="${ARMS[*]}"
export WBES_ZS_PART="${WBES_ZS_PART:-topology}" WBES_EVAL_SCENARIOS=clean WBES_ZS_EMBED="${WBES_ZS_EMBED:-1}"
COPY="/tmp/zs_v3_${SLURM_JOB_ID:-manual}.sh"
sed 's|^LAUNCH=()$|LAUNCH=(v3_work/trainer/eval_v3.py --)|' "$AAU_DIR/zs3dmm/zs_zeroshot.sbatch" > "$COPY"
grep -qx 'LAUNCH=(v3_work/trainer/eval_v3.py --)' "$COPY" || { echo "ERRORE: LAUNCH non sostituito" >&2; exit 1; }
trap 'rm -f "$COPY"' EXIT
echo "[v3-eval] dominio=$WBES_ZS_DOMAIN bracci=$WBES_ZS_ARMS part=$WBES_ZS_PART embed=$WBES_ZS_EMBED"
bash "$COPY"
