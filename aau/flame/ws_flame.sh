#!/usr/bin/env bash
# WS-FLAME in un comando: sottomette dati -> (zero-shot congiunto, zero-shot solo BFM,
# baseline geometriche) con afterok. Gira sul frontend.
#
#   aau/flame/ws_flame.sh      # file della licenza in v2_work/genflame/official/:
#                              #   FLAME2020/generic_model.pkl e FLAME_masks.pkl
#   WBES_FLAME_MODEL=/percorso/generic_model.pkl WBES_FLAME_MASKS=/percorso/FLAME_masks.pkl \
#     aau/flame/ws_flame.sh    # oppure altrove, indicati a mano
#
#   aau/flame/ws_flame.sh --skip-build   # dati gia' pronti: solo i tre eval
#
# Opzioni sbatch in piu' per job (separate da spazi), p.es. per una prova corta:
#   WBES_FLAME_BUILD_OPTS="--time=00:20:00" WBES_FLAME_ZS_OPTS="--time=00:30:00" \
#   WBES_FLAME_BL_OPTS="--time=00:30:00" aau/flame/ws_flame.sh
# Tutte le WBES_FLAME_* esportate qui arrivano ai job (sbatch esporta l'ambiente).
set -euo pipefail

AAU="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$AAU/env.sh"
source "$AAU/flame/flame_env.sh"

SKIP_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        *) echo "uso: aau/flame/ws_flame.sh [--skip-build]" >&2; exit 2 ;;
    esac
done

# shellcheck disable=SC2206
BUILD_OPTS=(${WBES_FLAME_BUILD_OPTS:-})
# shellcheck disable=SC2206
ZS_OPTS=(${WBES_FLAME_ZS_OPTS:-})
# shellcheck disable=SC2206
BL_OPTS=(${WBES_FLAME_BL_OPTS:-})

DEP=()
if (( SKIP_BUILD )); then
    flame_require_view
else
    flame_require_model
    build=$("$AAU/submit.sh" flame/flame_build.sbatch --parsable "${BUILD_OPTS[@]}")
    echo "[ws-flame] dati: job $build"
    DEP=(--dependency="afterok:$build")
fi

joint=$(WBES_FLAME_ARM=joint "$AAU/submit.sh" flame/flame_zeroshot.sbatch --parsable "${DEP[@]}" "${ZS_OPTS[@]}")
bfm=$(WBES_FLAME_ARM=bfm_only "$AAU/submit.sh" flame/flame_zeroshot.sbatch --parsable "${DEP[@]}" "${ZS_OPTS[@]}")
bl=$("$AAU/submit.sh" flame/flame_baselines.sbatch --parsable "${DEP[@]}" "${BL_OPTS[@]}")
echo "[ws-flame] zero-shot congiunto: job $joint"
echo "[ws-flame] zero-shot solo BFM:  job $bfm"
echo "[ws-flame] baseline:            job $bl"
echo "[ws-flame] output in $WBES_FLAME_RUNS/<impronta dei dati>/ (l'impronta la stampa il log del build)"
