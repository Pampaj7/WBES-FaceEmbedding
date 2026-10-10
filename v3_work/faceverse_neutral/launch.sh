#!/usr/bin/env bash
# FaceVerse neutra contro espressioni (aau/runs/evidence/faceverse_neutral/PROTOCOL.md): tutti i job, in ordine.
#
#   v3_work/faceverse_neutral/launch.sh
#
#   embed     fvn_embed.sbatch (L40S): bracci, C3M, e108 sulla eval_view
#   scalars   aau/baselines_mm/blmm.sbatch, vista faceverse_neutral (CS robusta per mesh; serve al modo cs)
#   fast      Chamfer e ICP, modi mm e cs (cs dopo scalars)
#   template  NICP su template, modo mm
#   nicp      NICP per coppia, modi mm e cs: blmm_multi.sbatch, coda condivisa (BLMM_CLAIM_TAG=fvn), dopo scalars
#   summary   fvn_summary.sbatch, dopo tutti (afterany: si legge results.md, che elenca i mancanti)
# Ogni job e' ripartibile (uscite gia' su disco saltate). Le dipendenze degli scalari sono afterany: blmm_scalars.py
# scrive scalars.npz prima di confrontare L e CS_ref con params.json.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
sub() { sbatch --parsable "$@"; }
BL=aau/baselines_mm/blmm.sbatch
E=$(sub v3_work/faceverse_neutral/fvn_embed.sbatch)
S=$(BLMM_STEP=scalars BLMM_ARGS="--views faceverse_neutral" sub -J wbes-fvn-scalars --time=00:45:00 "$BL")
FM=$(BLMM_STEP=pairs BLMM_ARGS="faceverse_neutral mm fast" sub -J wbes-fvn-fast-mm --time=01:00:00 "$BL")
FC=$(BLMM_STEP=pairs BLMM_ARGS="faceverse_neutral cs fast" sub -J wbes-fvn-fast-cs --time=01:00:00 \
  --dependency="afterany:$S" "$BL")
T=$(BLMM_STEP=template BLMM_ARGS="--views faceverse_neutral --modes mm" sub -J wbes-fvn-tpl-mm --time=01:00:00 "$BL")
N=()
for k in 1 2; do
  N+=("$(BLMM_CLAIM_TAG=fvn BLMM_JOBS="faceverse_neutral:mm faceverse_neutral:cs" sub -J "wbes-fvn-nicp$k" \
    --ntasks=12 --time=03:00:00 --dependency="afterany:$S" aau/baselines_mm/blmm_multi.sbatch)")
done
ALL="$E:$S:$FM:$FC:$T:${N[0]}:${N[1]}"
J="embed $E, scalari $S, fast mm $FM, fast cs $FC, template $T, nicp ${N[*]}"
R=$(FVN_JOBS="$J" sub --dependency="afterany:$ALL" v3_work/faceverse_neutral/fvn_summary.sbatch)
echo "$J, riepilogo $R"
