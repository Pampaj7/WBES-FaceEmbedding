#!/usr/bin/env bash
#SBATCH --job-name=wbes-ava-conf
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#
# Ava-256, prova confermativa: la catena da spec (aau/ava256/conf_*.py, blmm/bp con --spec), passi CPU in sequenza nello
# stesso job; il primo passo e' sempre ``conf_common.py verify`` (sez. 11: con ``freeze`` nello spec, manifest,
# impronte, checkpoint, pesi del rilevatore; un solo scarto ferma il job). STESSI passi per Ava-256 e per i collaudi su
# HIFI3D (spec/hifi3d_full.json, spec/hifi3d_det.json), cambia solo CONF_SPEC.
#
#   S=aau/ava256/spec/ava256.json
#   CONF_SPEC=$S CONF_STEPS="versions gate define prep render detect detprep" sbatch --nodelist=a256-t4-04 aau/ava256/conf.sbatch
#   CONF_SPEC=$S CONF_STEPS="run:0/4" sbatch --time=24:00:00 aau/ava256/conf.sbatch      (4 job, run:k/4; poi merge)
#   CONF_SPEC=$S CONF_STEPS="blmm-scalars blmm-template blmm-pairs:mm:fast blmm-pairs:cs:fast" sbatch aau/ava256/conf.sbatch
#   CONF_SPEC=$S CONF_STEPS="blmm-pairs:mm:nicp blmm-pairs:cs:nicp" sbatch --time=24:00:00 aau/ava256/conf.sbatch
#   CONF_SPEC=$S CONF_STEPS="bp-fit bp-fit-e1 bp-crop" sbatch aau/ava256/conf.sbatch
#   CONF_SPEC=$S CONF_STEPS="head-fit" sbatch --nodelist=i256-a10-10 --cpus-per-task=4 --mem=64G aau/ava256/conf.sbatch
#   CONF_SPEC=$S CONF_STEPS="eval" sbatch aau/ava256/conf.sbatch                          (LEGGE LE GT)
# Passi: versions (venv di MICA), gate (gate-render, rilevatore, gate-compare: tolleranza 0, se fallisce il job si
# ferma), define, prep, render, detect (venv di MICA), detprep, run[:k/n], merge (conf_protocols.py); blmm-scalars,
# blmm-template, blmm-pairs:<modo>:<passo> (con --claim: piu' job svuotano la stessa coda); bp-fit, bp-fit-e1, bp-crop
# (sigma 2, LOOP congelato); head-block, head-fit (conf_head.py; nodo i256-a10-10); eval (conf_eval.py);
# export-template e collaudo:<passo> (conf_collaudo.py, solo HIFI3D). Il rilevatore gira come in pr_noise.sbatch
# (onnxruntime su CPU, OMP_NUM_THREADS = core del job); gate e render dello stesso dominio nello stesso job, sullo stesso
# nodo (a256-t4-04, quello di sviluppo, job 1071153).
set -euo pipefail
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
SPEC="${CONF_SPEC:?CONF_SPEC: spec json}"
W="${SLURM_CPUS_PER_TASK:-8}"
NAME=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["name"])' "$SPEC")
WORK=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["out"]["work"])' "$SPEC")
A=aau/ava256
RUN=(v3_work/unified_gt/run.sh -u)
O3D=(aau/outlineB/run_o3d.sh -u)
MICA=(aau/recon/run_recon.sh mica -u)
DETDIR="$WORK/protocolli/noise/det"
echo "[conf] host=$(hostname) job=${SLURM_JOB_ID:-none} spec=$SPEC vista=$NAME passi=${CONF_STEPS:?CONF_STEPS} cpu=$W $(date +%T)"
"${RUN[@]}" $A/conf_common.py verify --spec "$SPEC"
for tok in $CONF_STEPS; do
    step="${tok%%:*}"; arg=""; [[ "$tok" == *:* ]] && arg="${tok#*:}"
    t0=$(date +%s)
    echo "[conf] passo $step $arg $(date +%T)"
    case "$step" in
        versions)  "${MICA[@]}" $A/conf_common.py versions --spec "$SPEC" ;;
        gate)      "${RUN[@]}" $A/conf_protocols.py gate-render --spec "$SPEC" --workers "$W"
                   OMP_NUM_THREADS="$W" "${MICA[@]}" aau/competitors/protocolli/pr_noise_detect.py --dir "$DETDIR/gate_hifi3d"
                   "${RUN[@]}" $A/conf_protocols.py gate-compare --spec "$SPEC" ;;
        define|prep|render|detprep|merge)
                   "${RUN[@]}" $A/conf_protocols.py "$step" --spec "$SPEC" --workers "$W" ;;
        detect)    OMP_NUM_THREADS="$W" "${MICA[@]}" aau/competitors/protocolli/pr_noise_detect.py --dir "$DETDIR/$NAME" ;;
        run)       "${RUN[@]}" $A/conf_protocols.py run --spec "$SPEC" --workers "$W" --shard "${arg:-0/1}" \
                       ${CONF_REVERSE:+--reverse} ;;
        blmm-scalars)  "${O3D[@]}" aau/baselines_mm/blmm_scalars.py --spec "$SPEC" --workers "$W" ;;
        blmm-template) "${O3D[@]}" aau/baselines_mm/blmm_template.py --spec "$SPEC" --modes mm,cs --workers "$W" ;;
        blmm-pairs)    BLMM_CLAIM_TAG="${BLMM_CLAIM_TAG:-conf}" "${O3D[@]}" aau/baselines_mm/blmm_pairs.py "$NAME" \
                           ${arg//:/ } --spec "$SPEC" --claim --workers "$W" ;;
        bp-fit)    "${O3D[@]}" aau/baselines_param/bp_fit.py --spec "$SPEC" --workers "$W" ;;
        bp-fit-e1) "${O3D[@]}" aau/baselines_param/bp_fit_e1.py --spec "$SPEC" --workers "$W" ;;
        bp-crop)   "${O3D[@]}" aau/baselines_param/bp_e2.py crop --spec "$SPEC" --workers "$W" ;;
        head-block|head-fit)
                   UGT_THREADS=1 "${RUN[@]}" $A/conf_head.py "${step#head-}" --spec "$SPEC" ;;
        eval)      UGT_THREADS=1 "${RUN[@]}" $A/conf_eval.py --spec "$SPEC" --workers "$W" ;;
        export-template) "${RUN[@]}" $A/conf_collaudo.py export-template ;;
        collaudo)  if [[ "$arg" == blmm || "$arg" == bp ]]; then R=("${O3D[@]}"); else R=("${RUN[@]}"); fi
                   "${R[@]}" $A/conf_collaudo.py "$arg" --workers "$W" ;;
        *) echo "passo sconosciuto: $step" >&2; exit 2 ;;
    esac
    echo "[conf] OK $step $arg in $(( $(date +%s) - t0 ))s"
done
echo "[conf] OK tutti i passi $(date +%T)"
