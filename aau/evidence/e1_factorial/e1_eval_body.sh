#!/usr/bin/env bash
# Corpo di e1_eval.sbatch (letto all'avvio del job). Checkpoint epoch036/epoch072 di una cella E1 sui
# domini di test, con le pipeline esistenti invariate e le uscite in aau/runs/evidence/e1/:
#   hifi   aau/zs3dmm/zs_zeroshot.sbatch, HIFI3D, braccio scale_<tag> (WBES_ZS_CKPT_SCALE_<TAG>),
#          WBES_ZS_PART=topology + WBES_ZS_EMBED=1: pair_metrics (Spearman) ed embedding (riconoscimento),
#          come i bracci scale_e0NN di aau/runs/data_scale_ood; WBES_HIFI_RUNS=hifi_runs (ws_hifi3d intatta)
#   fv     la stessa, FaceVerse con espressioni in convenzione BFM (_flip, quella di curve.md), solo
#          embedding (WBES_ZS_PART=embed): il riconoscimento non usa il breakdown; WBES_FV_RUNS=fv
#   flame  aau/flame/flame_zeroshot.sbatch con il checkpoint al posto del congiunto, una cartella per
#          checkpoint (WBES_FLAME_RUNS=flame/<tag>), /tmp proprio del job
#   now    come aau/data_scale/ood_now.sbatch, cartelle separate: lavoro ~/data/e1_now/<tag> (dati NoW
#          fuori dal repo), uscita now/<tag>
# Tag dei bracci: scale_e1<cella>e036 / ...e072; per c3m (run su scala) scale_e036 / scale_e072, gli
# stessi nomi dei riepiloghi di data_scale_ood (HIFI3D e NoW di c3m si riusano: WBES_E1_EVAL_STEPS="fv flame").
set -uo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
CELL="${WBES_E1_CELL:?WBES_E1_CELL=c2m|c2f|c3f|g1|c3m}"
STEPS=" ${WBES_E1_EVAL_STEPS:-hifi fv flame now} "
OUT="${WBES_E1_OUT:-$AAU_RUNS/evidence/e1}"        # WBES_E1_OUT: solo per le prove
if [[ "$CELL" == c3m ]]; then
    RR="$AAU_RUNS/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411"
else
    RR="$(cat "$AAU_RUNS/evidence/e1/train_${CELL}_s1234.runs_root")"
fi
CK_DIR="$(ls -d "$RR"/mixed_*/checkpoints | head -1)"
declare -A CK TAG
for e in 036 072; do
    CK[$e]="$CK_DIR/epoch$e.pth"
    [[ -f "${CK[$e]}" ]] || { echo "ERRORE: ${CK[$e]} assente" >&2; exit 1; }
    if [[ "$CELL" == c3m ]]; then TAG[$e]="e$e"; else TAG[$e]="e1${CELL}e$e"; fi
    export "WBES_ZS_CKPT_SCALE_${TAG[$e]^^}=${CK[$e]}"
done
ARMS="scale_${TAG[036]} scale_${TAG[072]}"
echo "[e1-eval] cella=$CELL passi='$STEPS' bracci=$ARMS host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[e1-eval] checkpoint ${CK[036]} ${CK[072]}"
FAILED=()

if [[ "$STEPS" == *" hifi "* ]]; then
    echo "[e1-eval] $(date +%T) HIFI3D"
    WBES_ZS_DOMAIN=hifi WBES_HIFI_RUNS="$OUT/hifi_runs" WBES_ZS_ARMS="$ARMS" WBES_ZS_PART=topology WBES_ZS_EMBED=1 \
        bash aau/zs3dmm/zs_zeroshot.sbatch || FAILED+=(hifi)
fi
if [[ "$STEPS" == *" fv "* ]]; then
    echo "[e1-eval] $(date +%T) FaceVerse con espressioni, embedding"
    WBES_ZS_DOMAIN=fv WBES_ZS_EXPR=1 WBES_FV_RUNS="$OUT/fv" WBES_ZS_FLIP_FACES=1 WBES_ZS_ARMS="$ARMS" \
        WBES_ZS_PART=embed bash aau/zs3dmm/zs_zeroshot.sbatch || FAILED+=(fv)
fi
if [[ "$STEPS" == *" flame "* ]]; then
    for e in 036 072; do
        echo "[e1-eval] $(date +%T) FLAME ${TAG[$e]}"
        WBES_FLAME_ARM=joint WBES_FLAME_CKPT_JOINT="${CK[$e]}" WBES_FLAME_RUNS="$OUT/flame/${TAG[$e]}" \
            WBES_FLAME_TMP="/tmp/e1flame_${SLURM_JOB_ID:-manual}_${TAG[$e]}" \
            bash aau/flame/flame_zeroshot.sbatch || FAILED+=("flame_${TAG[$e]}")
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
        { WBES_NOW_WORK="$W" WBES_NOW_OUT="$O" WBES_CKPT="${CK[$e]}" bash aau/recon/now_ops_latent.sbatch \
          && WBES_NOW_WORK="$W" WBES_NOW_OUT="$O" AAU_NV= "$AAU_DIR/run.sh" aau/recon/now_summarize.py; } \
            || FAILED+=("now_$t")
    done
fi

echo "[e1-eval] $(date +%T) fine: falliti=${FAILED[*]:-nessuno}"
(( ${#FAILED[@]} == 0 ))
