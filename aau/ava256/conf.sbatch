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
#   CONF_SPEC=$S CONF_STEPS="bp-fit bp-fit-e1 bp-crop" sbatch --nodelist=i256-a10-08 aau/ava256/conf.sbatch
#   CONF_SPEC=$S CONF_STEPS="head-fit" sbatch --nodelist=i256-a10-10 --cpus-per-task=4 --mem=64G aau/ava256/conf.sbatch
#   CONF_SPEC=$S CONF_STEPS="eval" sbatch aau/ava256/conf.sbatch                          (LEGGE LE GT)
#   v4, embedding e descrittivi fuori da conf_eval (conf_embed.py, conf_desc.py):
#   CONF_SPEC=$S CONF_STEPS="embed-stage" sbatch aau/ava256/conf.sbatch                   (link e operatori, CPU)
#   CONF_SPEC=$S CONF_STEPS="embed" sbatch -p aicentre-a100 --qos=unprivileged --requeue --gres=gpu:a100:1 \
#       --cpus-per-task=16 --mem=96G aau/ava256/conf.sbatch                 (v5: A100, la classe di GPU dello sviluppo)
#   CONF_SPEC=$S CONF_STEPS="desc-varifold" sbatch --gres=gpu:l40s:1 --cpus-per-task=16 --mem=64G aau/ava256/conf.sbatch
#   CONF_SPEC=$S CONF_STEPS="desc-spectral desc-icp-same desc-arcface desc-gw-prep desc-perc" sbatch --nodelist=i256-a10-09 \
#       aau/ava256/conf.sbatch                                                 (ArcFace: nodo obbligato i256-a10-09)
#   CONF_SPEC=$S CONF_STEPS="desc-gw-run:0/4" sbatch --time=24:00:00 aau/ava256/conf.sbatch   (4 job; poi desc-gw-merge)
#   CONF_SPEC=$S CONF_STEPS="desc-ulr-run:0/1" sbatch --gres=gpu:l40s:1 --mem=160G --time=06:00:00 aau/ava256/conf.sbatch
#   CONF_SPEC=$S CONF_STEPS="desc-ulr-merge" sbatch aau/ava256/conf.sbatch               (v6: ULRSSM, poi desc-eval)
#   CONF_SPEC=$S CONF_STEPS="desc-eval" sbatch aau/ava256/conf.sbatch                     (LEGGE LE GT)
# Passi: versions (venv di MICA), gate (gate-render, rilevatore, gate-compare: tolleranza 0, se fallisce il job si
# ferma), define, prep, render, detect (venv di MICA), detprep, run[:k/n], merge (conf_protocols.py); blmm-scalars,
# blmm-template, blmm-pairs:<modo>:<passo> (con --claim: piu' job svuotano la stessa coda); bp-fit, bp-fit-e1, bp-crop
# (sigma 2, LOOP congelato); head-block, head-fit (conf_head.py; nodo i256-a10-10); eval (conf_eval.py);
# export-template e collaudo:<passo> (conf_collaudo.py, solo HIFI3D); v4: embed-stage (conf_embed.py stage, poi gli
# operatori areanorm k_eig 128 in W-1 shard come d3.sbatch, controllo del conteggio), embed (GPU, AAU_NV=--nv), desc-spectral,
# desc-varifold[:k/n] (GPU), desc-icp-same, desc-arcface (render su /tmp del job), desc-gw-prep, desc-gw-run[:k/n],
# desc-gw-merge, desc-perc (venv di gw_percettivi), desc-eval (conf_desc.py). Il rilevatore gira come in pr_noise.sbatch
# (onnxruntime su CPU, OMP_NUM_THREADS = core del job); gate e render dello stesso dominio nello stesso job, sullo stesso
# nodo (a256-t4-04, quello di sviluppo, job 1071153).
# v7: su Ava-256 si sottomette dal worktree del commit congelato (aau/ava256/conf_worktree.sh), con --chdir=<worktree> e
# CONF_FROZEN_COMMIT=<commit> (conf_common.verify controlla HEAD e albero pulito; conf_run.py i moduli importati).
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
REPOPY=(aau/run.sh -u)
CR=$A/conf_run.py      # v7: ogni passo Python passa da qui (moduli importati e confronto col commit congelato)
BL=(aau/baselines/run_bl.sh -u)
GP=(aau/competitors/gw_percettivi/gp_run.sh -u)
ops() {  # <cartella delle mesh> <cartella degli operatori>: areanorm k_eig 128 in W-1 shard (d3.sbatch), conteggio
  local src="$1" out="$2" i N M pids=()
  N=$(find "$src" -name "*.npz" | wc -l)
  mkdir -p "$out"
  for (( i = 0; i < W - 1; i++ )); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "${REPOPY[@]}" $CR v2_work/potential/areanorm_operators.py \
      --input-dir "$src" --output-dir "$out" --k-eig 128 --shard "$i/$((W - 1))" \
      > "/tmp/wbes_conf_ops_${SLURM_JOB_ID:-x}.$i.log" 2>&1 & pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "ERRORE: shard $i"; cat "/tmp/wbes_conf_ops_${SLURM_JOB_ID:-x}.$i.log"; return 1; }
  done
  rm -f /tmp/wbes_conf_ops_"${SLURM_JOB_ID:-x}".*.log
  M=$(find "$out" -name "*.npz" ! -name ".*" | wc -l)
  echo "[conf] $(date +%T) operatori $M / $N in $out, $(du -sh "$out" | cut -f1)"
  (( M == N )) || { echo "ERRORE: operatori incompleti"; return 1; }
}
DETDIR="$WORK/protocolli/noise/det"
echo "[conf] host=$(hostname) job=${SLURM_JOB_ID:-none} spec=$SPEC vista=$NAME passi=${CONF_STEPS:?CONF_STEPS} cpu=$W $(date +%T)"
"${RUN[@]}" $CR $A/conf_common.py verify --spec "$SPEC"
for tok in $CONF_STEPS; do
    step="${tok%%:*}"; arg=""; [[ "$tok" == *:* ]] && arg="${tok#*:}"
    t0=$(date +%s)
    echo "[conf] passo $step $arg $(date +%T)"
    case "$step" in
        versions)  "${MICA[@]}" $CR $A/conf_common.py versions --spec "$SPEC" ;;
        gate)      "${RUN[@]}" $CR $A/conf_protocols.py gate-render --spec "$SPEC" --workers "$W"
                   OMP_NUM_THREADS="$W" "${MICA[@]}" $CR aau/competitors/protocolli/pr_noise_detect.py --dir "$DETDIR/gate_hifi3d"
                   "${RUN[@]}" $CR $A/conf_protocols.py gate-compare --spec "$SPEC" ;;
        define|prep|render|detprep|merge)
                   "${RUN[@]}" $CR $A/conf_protocols.py "$step" --spec "$SPEC" --workers "$W" ;;
        detect)    OMP_NUM_THREADS="$W" "${MICA[@]}" $CR aau/competitors/protocolli/pr_noise_detect.py --dir "$DETDIR/$NAME" ;;
        run)       "${RUN[@]}" $CR $A/conf_protocols.py run --spec "$SPEC" --workers "$W" --shard "${arg:-0/1}" \
                       ${CONF_REVERSE:+--reverse} ;;
        blmm-scalars)  "${O3D[@]}" $CR aau/baselines_mm/blmm_scalars.py --spec "$SPEC" --workers "$W" ;;
        blmm-template) "${O3D[@]}" $CR aau/baselines_mm/blmm_template.py --spec "$SPEC" --modes mm,cs --workers "$W" ;;
        blmm-pairs)    BLMM_CLAIM_TAG="${BLMM_CLAIM_TAG:-conf}" "${O3D[@]}" $CR aau/baselines_mm/blmm_pairs.py "$NAME" \
                           ${arg//:/ } --spec "$SPEC" --claim --workers "$W" ;;
        bp-fit|bp-fit-e1|bp-crop)                                     # v7: nodo obbligato del fit B
                   "${RUN[@]}" $A/conf_common.py host --spec "$SPEC" --role bp ;;&
        bp-fit)    "${O3D[@]}" $CR aau/baselines_param/bp_fit.py --spec "$SPEC" --workers "$W" ;;
        bp-fit-e1) "${O3D[@]}" $CR aau/baselines_param/bp_fit_e1.py --spec "$SPEC" --workers "$W" ;;
        bp-crop)   "${O3D[@]}" $CR aau/baselines_param/bp_e2.py crop --spec "$SPEC" --workers "$W" ;;
        head-block|head-fit)
                   UGT_THREADS=1 "${RUN[@]}" $CR $A/conf_head.py "${step#head-}" --spec "$SPEC" ;;
        eval)      UGT_THREADS=1 "${RUN[@]}" $CR $A/conf_eval.py --spec "$SPEC" --workers "$W" ;;
        embed-stage)
                   "${REPOPY[@]}" $CR $A/conf_embed.py stage --spec "$SPEC"
                   while read -r IN OUT; do ops "$IN" "$OUT"; done < <(python3 -c 'import json, sys
s = json.load(open(sys.argv[1])); r = s["embed"]["ops_root"]
for k in s["embed"]["sets"]: print(f"{r}/{k}/in {r}/{k}/ops")' "$SPEC") ;;
        embed)     AAU_NV=--nv "${REPOPY[@]}" $CR $A/conf_embed.py run --spec "$SPEC" ;;
        desc-spectral) "${REPOPY[@]}" $CR $A/conf_desc.py spectral --spec "$SPEC" --workers "$W" ;;
        desc-varifold) AAU_NV=--nv "${O3D[@]}" $CR $A/conf_desc.py varifold --spec "$SPEC" --device cuda \
                           --shard "${arg:-0/1}" --workers "$W" ;;
        desc-icp-same) "${O3D[@]}" $CR $A/conf_desc.py icp-same --spec "$SPEC" --workers "$W" ;;
        desc-arcface)  ST="/tmp/wbes_conf_arc_${SLURM_JOB_ID:-x}"
                       "${BL[@]}" $CR $A/conf_desc.py arcface --spec "$SPEC" --stage "$ST" --workers "$W"; rm -rf "$ST" ;;
        desc-gw-prep|desc-gw-merge)
                   "${GP[@]}" $CR $A/conf_desc.py "${step#desc-}" --spec "$SPEC" --workers "$W" ;;
        desc-gw-run) "${GP[@]}" $CR $A/conf_desc.py gw-run --spec "$SPEC" --shard "${arg:-0/1}" --workers "$W" ;;
        desc-perc) GP_THREADS="$W" "${GP[@]}" $CR $A/conf_desc.py perc --spec "$SPEC" --workers "$W" ;;
        desc-eval) UGT_THREADS=1 "${RUN[@]}" $CR $A/conf_desc.py eval --spec "$SPEC" --workers "$W" ;;
        desc-ulr-run)  ST="/tmp/wbes_conf_ulr_${SLURM_JOB_ID:-x}"           # operatori di ULRSSM: /tmp (conta su --mem)
                       AAU_NV=--nv "${REPOPY[@]}" $CR $A/conf_desc.py ulr-run --spec "$SPEC" --stage "$ST" \
                           --shard "${arg:-0/1}" --workers "$W"; rm -rf "$ST" ;;
        desc-ulr-merge) "${RUN[@]}" $CR $A/conf_desc.py ulr-merge --spec "$SPEC" ;;
        export-template) "${RUN[@]}" $CR $A/conf_collaudo.py export-template ;;
        collaudo)  if [[ "$arg" == blmm || "$arg" == bp ]]; then R=("${O3D[@]}"); else R=("${RUN[@]}"); fi
                   G=(); [[ "$arg" == guard ]] && G=(--worktree "$WBES_ROOT")       # v7: dal worktree
                   "${R[@]}" $CR $A/conf_collaudo.py "$arg" --workers "$W" "${G[@]}" ;;
        *) echo "passo sconosciuto: $step" >&2; exit 2 ;;
    esac
    echo "[conf] OK $step $arg in $(( $(date +%s) - t0 ))s"
done
echo "[conf] OK tutti i passi $(date +%T)"
