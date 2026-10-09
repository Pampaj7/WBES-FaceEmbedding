#!/usr/bin/env bash
# Eval di un braccio C3F del trainer v3 (protocollo ablation_protocol.md), ai passi 10.548 e 21.096 con i pesi EMA
# (epoch036_ema.pth, epoch072_ema.pth). Pipeline esistenti come in aau/evidence/e1_factorial/e1_eval_body.sh, ma
# lanciate da una copia di aau/zs3dmm/zs_zeroshot.sbatch con LAUNCH=(v3_work/trainer/eval_v3.py --) (modelli v3:
# pooling e frame per area) e scenario clean:
#   hifi    HIFI3D, breakdown + embedding (graduata con GT maxabs e unificata, rank-1)
#   devfs   dev FaceScape vista neutra (breakdown + embedding) e vista con espressioni (embedding): punteggio dev
#   fv      FaceVerse con espressioni, convenzione BFM (_flip), embedding
#   now     NoW, latenti dalla catena di aau/recon/now_ops_latent.sbatch (copia con eval_v3 davanti a now_latent.py)
# Uscite in aau/runs/evidence/trainer_v3/ablations/c3f_eval/. Ripartibile: i bracci con .done si saltano.
set -uo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
ARM="${WBES_V3_ARM:?WBES_V3_ARM}"
STEPS=" ${WBES_V3_EVAL_STEPS:-hifi devfs fv now} "
OUT="$AAU_RUNS/evidence/trainer_v3/ablations/c3f_eval"
RR="$AAU_RUNS/evidence/trainer_v3/ablations/c3f_runs/$ARM"
CK_DIR="$(ls -d "$RR"/v3_*/checkpoints | head -1)"
declare -A CK TAG
for e in 036 072; do
  CK[$e]="$CK_DIR/epoch${e}_ema.pth"
  [[ -f "${CK[$e]}" ]] || { echo "ERRORE: ${CK[$e]} assente" >&2; exit 1; }
  TAG[$e]="v3${ARM}e$e"
  export "WBES_ZS_CKPT_SCALE_${TAG[$e]^^}=${CK[$e]}"
done
ARMS="scale_${TAG[036]} scale_${TAG[072]}"
ZS="/tmp/zs_v3c3f_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}.sh"
sed 's|^LAUNCH=()$|LAUNCH=(v3_work/trainer/eval_v3.py --)|' "$AAU_DIR/zs3dmm/zs_zeroshot.sbatch" > "$ZS"
grep -qx 'LAUNCH=(v3_work/trainer/eval_v3.py --)' "$ZS" || { echo "ERRORE: LAUNCH non sostituito" >&2; exit 1; }
NOWS="/tmp/now_v3c3f_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}.sh"
sed 's|"\$AAU_DIR/run.sh" aau/recon/now_latent.py|"$AAU_DIR/run.sh" v3_work/trainer/eval_v3.py -- aau/recon/now_latent.py|' \
  "$AAU_DIR/recon/now_ops_latent.sbatch" > "$NOWS"
grep -q 'eval_v3.py -- aau/recon/now_latent.py' "$NOWS" || { echo "ERRORE: now_latent non agganciato" >&2; exit 1; }
trap 'rm -f "$ZS" "$NOWS"' EXIT
export WBES_EVAL_SCENARIOS=clean
echo "[v3-eval] braccio=$ARM bracci=$ARMS host=$(hostname) job=${SLURM_JOB_ID:-none} riavvii=${SLURM_RESTART_COUNT:-0}"
FAILED=()
# Ogni passo fino a 3 tentativi a 10 minuti di distanza: su nv-ai-04 la risoluzione dell'utente di dominio cade a
# tratti ("unknown userid" all'avvio di singularity: 9 ottobre 05:33-05:51 e 11:06-11:09) e i passi sono ripartibili
# (le stage con .done si saltano).
retry() {  # $1 nome, resto: comando
  local name="$1" t; shift
  for t in 1 2 3; do
    "$@" && return 0
    (( t < 3 )) && { echo "[v3-eval] $(date +%T) $name: tentativo $t fallito, riprovo fra 10 minuti"; sleep 600; }
  done
  return 1
}
step_hifi() {
  WBES_ZS_DOMAIN=hifi WBES_HIFI_RUNS="$OUT/hifi_runs" WBES_ZS_ARMS="$ARMS" WBES_ZS_PART=topology WBES_ZS_EMBED=1 bash "$ZS"
}
step_devfs() {
  ( source aau/zs3dmm/dev_facescape_env.sh
    export WBES_FV_RUNS="$OUT/devfs" WBES_ZS_ARMS="$ARMS" WBES_ZS_PART=topology WBES_ZS_EMBED=1
    unset WBES_ZS_EXPR WBES_ZS_FLIP_FACES
    bash "$ZS" )
}
step_devfs_expr() {
  ( source aau/zs3dmm/dev_facescape_env.sh
    export WBES_FV_RUNS="$OUT/devfs" WBES_ZS_EXPR=1 WBES_ZS_ARMS="$ARMS" WBES_ZS_PART=embed
    unset WBES_ZS_FLIP_FACES
    bash "$ZS" )
}
step_fv() {
  WBES_ZS_DOMAIN=fv WBES_ZS_EXPR=1 WBES_FV_RUNS="$OUT/fv" WBES_ZS_FLIP_FACES=1 WBES_ZS_ARMS="$ARMS" \
    WBES_ZS_PART=embed bash "$ZS"
}
step_now() {  # $1 W, $2 O, $3 checkpoint
  WBES_NOW_WORK="$1" WBES_NOW_OUT="$2" WBES_CKPT="$3" bash "$NOWS" \
    && WBES_NOW_WORK="$1" WBES_NOW_OUT="$2" AAU_NV= "$AAU_DIR/run.sh" aau/recon/now_summarize.py
}
if [[ "$STEPS" == *" hifi "* ]]; then
  echo "[v3-eval] $(date +%T) HIFI3D"
  retry hifi step_hifi || FAILED+=(hifi)
fi
if [[ "$STEPS" == *" devfs "* ]]; then
  echo "[v3-eval] $(date +%T) dev FaceScape, vista neutra"
  retry devfs step_devfs || FAILED+=(devfs)
  echo "[v3-eval] $(date +%T) dev FaceScape, vista con espressioni"
  retry devfs_expr step_devfs_expr || FAILED+=(devfs_expr)
fi
if [[ "$STEPS" == *" fv "* ]]; then
  echo "[v3-eval] $(date +%T) FaceVerse con espressioni"
  retry fv step_fv || FAILED+=(fv)
fi
if [[ "$STEPS" == *" now "* ]]; then
  SRC_W="$HOME/data/now_eval_work"
  SRC_O="$AAU_RUNS/now_eval"
  for e in 036 072; do
    t="${TAG[$e]}"
    W="$HOME/data/v3c3f_now/$t"
    O="$OUT/now/$t"
    mkdir -p "$W/embeddings" "$W/pairs" "$O"
    for x in arcface mica_assets official pred recon_face recon_face_withops_areanorm scan_face \
             scan_face_withops_areanorm template_lmk7.json; do
      ln -sfn "$SRC_W/$x" "$W/$x"
    done
    for f in "$SRC_W"/pairs/*.npz; do
      case "$(basename "$f")" in latent_*) ;; *) ln -sf "$f" "$W/pairs/" ;; esac
    done
    for f in "$SRC_O"/gt_arcface_* "$SRC_O"/gt_geometric_* "$SRC_O"/now_official_* "$SRC_O"/prep_* \
             "$SRC_O"/official_selfcheck.json "$SRC_O"/protocol.md; do
      ln -sf "$f" "$O/"
    done
    echo "ckpt=${CK[$e]}" > "$O/checkpoint.txt"
    echo "[v3-eval] $(date +%T) NoW $t"
    retry "now_$t" step_now "$W" "$O" "${CK[$e]}" || FAILED+=("now_$t")
  done
fi
echo "[v3-eval] $(date +%T) fine: falliti=${FAILED[*]:-nessuno}"
[[ "$0" == /tmp/v3c3f_eval_body_* ]] && rm -f "$0"
(( ${#FAILED[@]} == 0 ))
