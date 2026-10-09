#!/usr/bin/env bash
# Corpo di e1_eval.sbatch (copiato all'avvio del job). Checkpoint epoch036/epoch072 di una cella E1 sui domini
# di test, con le pipeline esistenti invariate e le uscite in aau/runs/evidence/e1/:
#   hifi   aau/zs3dmm/zs_zeroshot.sbatch, HIFI3D, braccio scale_<tag> (WBES_ZS_CKPT_SCALE_<TAG>),
#          WBES_ZS_PART=topology + WBES_ZS_EMBED=1: pair_metrics (Spearman, GT maxabs e unificata) ed embedding
#          (riconoscimento), come i bracci scale_e0NN di aau/runs/data_scale_ood; WBES_HIFI_RUNS=hifi_runs
#   devfs  la stessa sul dev FaceScape (aau/zs3dmm/dev_facescape_env.sh, emendamento del protocollo),
#          uscita in devfs/ invece di ws_dev_facescape
#   fv     la stessa, FaceVerse con espressioni in convenzione BFM (_flip, quella di curve.md), solo embedding
#          (WBES_ZS_PART=embed): il riconoscimento non usa il breakdown; WBES_FV_RUNS=fv
#   flame  aau/flame/flame_zeroshot.sbatch con il checkpoint al posto del congiunto, una cartella per
#          checkpoint (WBES_FLAME_RUNS=flame/<tag>)
#   now    come aau/data_scale/ood_now.sbatch, cartelle separate: lavoro ~/data/e1_now/<tag> (dati NoW fuori
#          dal repo), uscita now/<tag>
# Celle: quelle di e1_train_body.sh (checkpoint dal puntatore train_<cella>.runs_root) e c3ml = il run su scala
# su L40S (1060130), solo riferimento: rivalutato qui, sullo stesso hardware di valutazione delle altre celle.
# Tag dei bracci: e1<cella>e036 / e1<cella>e072. Ripartibile (requeue): i bracci con .done si saltano.
set -uo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
CELL="${WBES_E1_CELL:?WBES_E1_CELL=<cella di e1_train_body.sh>|c3ml}"
STEPS=" ${WBES_E1_EVAL_STEPS:-hifi devfs fv flame now} "
OUT="${WBES_E1_OUT:-$AAU_RUNS/evidence/e1}"        # WBES_E1_OUT: solo per le prove
if [[ "$CELL" == c3ml ]]; then
    RR="$AAU_RUNS/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411"
else
    RR="$(cat "$AAU_RUNS/evidence/e1/train_${CELL}.runs_root")"
fi
CK_DIR="$(ls -d "$RR"/mixed_*/checkpoints | head -1)"
declare -A CK TAG
for e in 036 072; do
    CK[$e]="$CK_DIR/epoch$e.pth"
    [[ -f "${CK[$e]}" ]] || { echo "ERRORE: ${CK[$e]} assente" >&2; exit 1; }
    TAG[$e]="e1${CELL}e$e"
    export "WBES_ZS_CKPT_SCALE_${TAG[$e]^^}=${CK[$e]}"
done
ARMS="scale_${TAG[036]} scale_${TAG[072]}"
echo "[e1-eval] cella=$CELL passi='$STEPS' bracci=$ARMS host=$(hostname) job=${SLURM_JOB_ID:-none} riavvii=${SLURM_RESTART_COUNT:-0}"
echo "[e1-eval] checkpoint ${CK[036]} ${CK[072]}"
nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | head -1
FAILED=()
# ritentativi: fino a 3, a 10 minuti di distanza (nv-ai-04 a tratti non riconosce l'utente, "unknown userid":
# eval di C2F 1061844 fallita cosi' alle 11:06 del 9 ottobre); come v3_work/trainer/ablations/c3f/eval_body.sh.
# Le pipeline saltano i bracci gia' completi (.done), quindi un nuovo tentativo rifa' solo quello che manca
retry() {  # $1 nome, resto: comando
    local name="$1" t; shift
    for t in 1 2 3; do
        "$@" && return 0
        (( t < 3 )) && { echo "[e1-eval] $(date +%T) $name: tentativo $t fallito, riprovo fra 10 minuti"; sleep 600; }
    done
    return 1
}
step_hifi() {
    WBES_ZS_DOMAIN=hifi WBES_HIFI_RUNS="$OUT/hifi_runs" WBES_ZS_ARMS="$ARMS" WBES_ZS_PART=topology WBES_ZS_EMBED=1 \
        bash aau/zs3dmm/zs_zeroshot.sbatch
}
step_devfs() {
    ( source aau/zs3dmm/dev_facescape_env.sh
      export WBES_FV_RUNS="$OUT/devfs" WBES_ZS_ARMS="$ARMS" WBES_ZS_PART=topology WBES_ZS_EMBED=1
      unset WBES_ZS_EXPR WBES_ZS_FLIP_FACES
      bash aau/zs3dmm/zs_zeroshot.sbatch )
}
step_fv() {
    WBES_ZS_DOMAIN=fv WBES_ZS_EXPR=1 WBES_FV_RUNS="$OUT/fv" WBES_ZS_FLIP_FACES=1 WBES_ZS_ARMS="$ARMS" \
        WBES_ZS_PART=embed bash aau/zs3dmm/zs_zeroshot.sbatch
}
step_flame() {  # $1 checkpoint, $2 tag
    WBES_FLAME_ARM=joint WBES_FLAME_CKPT_JOINT="$1" WBES_FLAME_RUNS="$OUT/flame/$2" \
        WBES_FLAME_TMP="/tmp/e1flame_${SLURM_JOB_ID:-manual}_$2" bash aau/flame/flame_zeroshot.sbatch
}
step_now() {  # $1 W, $2 O, $3 checkpoint
    WBES_NOW_WORK="$1" WBES_NOW_OUT="$2" WBES_CKPT="$3" bash aau/recon/now_ops_latent.sbatch \
        && WBES_NOW_WORK="$1" WBES_NOW_OUT="$2" AAU_NV= "$AAU_DIR/run.sh" aau/recon/now_summarize.py
}

if [[ "$STEPS" == *" hifi "* ]]; then
    echo "[e1-eval] $(date +%T) HIFI3D"
    retry hifi step_hifi || FAILED+=(hifi)
fi
if [[ "$STEPS" == *" devfs "* ]]; then
    echo "[e1-eval] $(date +%T) dev FaceScape"
    retry devfs step_devfs || FAILED+=(devfs)
fi
if [[ "$STEPS" == *" fv "* ]]; then
    echo "[e1-eval] $(date +%T) FaceVerse con espressioni, embedding"
    retry fv step_fv || FAILED+=(fv)
fi
if [[ "$STEPS" == *" flame "* ]]; then
    for e in 036 072; do
        echo "[e1-eval] $(date +%T) FLAME ${TAG[$e]}"
        retry "flame_${TAG[$e]}" step_flame "${CK[$e]}" "${TAG[$e]}" || FAILED+=("flame_${TAG[$e]}")
    done
fi
if [[ "$STEPS" == *" now "* ]]; then
    SRC_W="$HOME/data/now_eval_work"
    SRC_O="$AAU_RUNS/now_eval"
    for e in 036 072; do
        t="${TAG[$e]}"
        W="${WBES_E1_NOW_WORK:-$HOME/data/e1_now}/$t"
        O="$OUT/now/$t"
        mkdir -p "$W/embeddings" "$W/pairs" "$O"
        # come ood_now.sbatch: patch, operatori e metriche non latenti come symlink, latenti nuovi
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
        echo "[e1-eval] $(date +%T) NoW $t"
        retry "now_$t" step_now "$W" "$O" "${CK[$e]}" || FAILED+=("now_$t")
    done
fi

echo "[e1-eval] $(date +%T) fine: falliti=${FAILED[*]:-nessuno}"
[[ "$0" == /tmp/e1_eval_body_* ]] && rm -f "$0"
(( ${#FAILED[@]} == 0 ))
