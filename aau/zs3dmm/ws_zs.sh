#!/usr/bin/env bash
# Zero-shot su un 3DMM mai visto, in un comando: sottomette dati -> (zero-shot dei tre modelli,
# baseline in due shard) -> rank delle baseline -> tabella, con afterok. Gira sul frontend.
#
#   aau/zs3dmm/ws_zs.sh hifi                 # HIFI3D   -> aau/runs/ws_hifi3d/summary.md
#   aau/zs3dmm/ws_zs.sh fv                   # FaceVerse -> aau/runs/ws_faceverse/summary.md
#   aau/zs3dmm/ws_zs.sh gnm                  # GNM Head  -> aau/runs/ws_gnm/summary.md
#   aau/zs3dmm/ws_zs.sh hifi --skip-build    # dati gia' pronti: solo eval e tabella
#
# Opzioni sbatch in piu' per job (separate da spazi), p.es. per FaceVerse, che sta alla
# risoluzione di BFM:
#   WBES_ZS_ZS_OPTS="--time=09:00:00 --mem=160G" aau/zs3dmm/ws_zs.sh fv --skip-build
# Tutte le WBES_ZS_* / WBES_<DOM>_* esportate qui arrivano ai job (sbatch esporta l'ambiente).
set -euo pipefail

AAU="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WBES_ZS_DOMAIN="${1:-}"
shift || true
source "$AAU/env.sh"
source "$AAU/zs3dmm/zs_env.sh"

SKIP_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        *) echo "uso: aau/zs3dmm/ws_zs.sh hifi|fv|gnm [--skip-build]" >&2; exit 2 ;;
    esac
done

# shellcheck disable=SC2206
BUILD_OPTS=(${WBES_ZS_BUILD_OPTS:-})
# shellcheck disable=SC2206
ZS_OPTS=(${WBES_ZS_ZS_OPTS:-})
# shellcheck disable=SC2206
BL_OPTS=(${WBES_ZS_BL_OPTS:-})
TAG="$WBES_ZS_DOMAIN"

DEP=()
if (( SKIP_BUILD )); then
    zs_require_view
else
    zs_require_model
    build=$("$AAU/submit.sh" zs3dmm/zs_build.sbatch --parsable --job-name="wbes-zs-build-$TAG" "${BUILD_OPTS[@]}")
    echo "[ws-zs] $TAG dati: job $build"
    DEP=(--dependency="afterok:$build")
fi

jobs=()
for arm in joint bfm_only ict_only; do
    j=$(WBES_ZS_ARM=$arm "$AAU/submit.sh" zs3dmm/zs_zeroshot.sbatch --parsable \
        --job-name="wbes-zs-$TAG-$arm" "${DEP[@]}" "${ZS_OPTS[@]}")
    echo "[ws-zs] $TAG zero-shot $arm: job $j"
    jobs+=("$j")
done
for shard in 0/2 1/2; do
    j=$(WBES_ZS_BL_STEPS=align WBES_ZS_BL_SHARD=$shard "$AAU/submit.sh" zs3dmm/zs_baselines.sbatch \
        --parsable --job-name="wbes-zs-bl-$TAG" "${DEP[@]}" "${BL_OPTS[@]}")
    echo "[ws-zs] $TAG baseline align $shard: job $j"
    jobs+=("$j")
done
after="$(IFS=:; echo "${jobs[*]}")"
rank=$(WBES_ZS_BL_STEPS=rank "$AAU/submit.sh" zs3dmm/zs_baselines.sbatch --parsable \
       --job-name="wbes-zs-blrank-$TAG" --dependency="afterok:${jobs[3]}:${jobs[4]}" \
       --cpus-per-task=4 --time=00:30:00)
echo "[ws-zs] $TAG baseline rank: job $rank"
sum=$("$AAU/submit.sh" zs3dmm/zs_summarize.sbatch --parsable --job-name="wbes-zs-sum-$TAG" \
      --dependency="afterok:$after:$rank")
echo "[ws-zs] $TAG tabella: job $sum"
echo "[ws-zs] output in $ZS_RUNS/summary.md"
