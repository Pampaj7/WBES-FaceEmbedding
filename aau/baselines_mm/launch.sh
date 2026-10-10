#!/usr/bin/env bash
# Tutti i job delle baseline con normalizzazione coerente (blmm.sbatch), in ordine di priorita' (la coda e' FIFO).
#
#   aau/baselines_mm/launch.sh fast  [dipendenza]     Chamfer, ICP, template (veloci)
#   aau/baselines_mm/launch.sh nicp  [dipendenza]     NICP per coppia, job multi-task a coda condivisa
#
# ``dipendenza``: job id da attendere (afterok), p.es. quello degli scalari. Ogni job e' ripartibile: le coppie di
# topologie (o gli shard FaMoS) gia' su disco si saltano.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
what="${1:?fast | nicp}"
dep=()
[[ -n "${2:-}" ]] && dep=(--dependency="afterok:$2")
sub() { sbatch --parsable "${dep[@]}" "$@" aau/baselines_mm/blmm.sbatch; }

case "$what" in
    fast)
        for v in hifi3d facescape faceverse facescape_expr; do
            for m in mm cs; do
                BLMM_STEP=pairs BLMM_ARGS="$v $m fast" sub -J "wbes-blmm-f-$v-$m" --time=01:00:00
            done
        done
        for v in facescape facescape_expr; do
            BLMM_STEP=pairs BLMM_ARGS="$v maxabs fast" sub -J "wbes-blmm-f-$v-maxabs" --time=01:00:00
        done
        for m in mm cs maxabs; do
            BLMM_STEP=pairs BLMM_ARGS="famos $m fast" sub -J "wbes-blmm-f-famos-$m" --time=01:00:00 --cpus-per-task=16 --mem=32G
        done
        for v in hifi3d facescape faceverse facescape_expr; do
            BLMM_STEP=template BLMM_ARGS="--views $v" sub -J "wbes-blmm-t-$v" --time=01:00:00
        done
        ;;
    nicp)
        # NICP per coppia: pochi job multi-task (blmm_multi.sbatch) con la coda condivisa dei reclami
        # (BLMM_CLAIM_TAG): ogni job svuota le (vista, modo) nell'ordine dato, i task prendono la prossima coppia
        # di topologie libera. Pochi job perche' la QoS normal ha MaxJobsPU = 12, condiviso con gli altri lavori.
        list="hifi3d:mm hifi3d:cs facescape:mm facescape:cs faceverse:mm faceverse:cs famos:mm famos:cs famos:maxabs"
        list+=" facescape:maxabs facescape_expr:mm facescape_expr:cs facescape_expr:maxabs"
        for ((k = 0; k < ${BLMM_NICP_JOBS:-4}; k++)); do
            BLMM_CLAIM_TAG="${BLMM_CLAIM_TAG:-pool}" BLMM_JOBS="$list" sbatch --parsable "${dep[@]}" \
                --ntasks="${BLMM_NICP_TASKS:-5}" aau/baselines_mm/blmm_multi.sbatch
        done
        ;;
    *) echo "uso: $0 fast|nicp [dipendenza]" >&2; exit 2 ;;
esac
