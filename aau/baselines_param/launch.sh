#!/usr/bin/env bash
# Tutti i job dei concorrenti parametrici e del varifold (bp.sbatch), con le dipendenze.
#
#   aau/baselines_param/launch.sh            fit (un job per modello, CPU), varifold (una L40S per vista a 500 mesh),
#                                            paired dopo tutti (afterok)
#
# Ogni passo e' ripartibile: le (vista, modello) e gli shard del varifold gia' su disco si saltano. FaMoS (15 mesh) si
# fa dentro i job di fit e, per il varifold, su CPU nel job del paired se manca.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
sub() { sbatch --parsable "$@" aau/baselines_param/bp.sbatch; }

ids=()
for m in gnm flame2023; do
    ids+=("$(BP_STEP=fit BP_ARGS="--models $m" sub -J "wbes-bp-fit-$m" --cpus-per-task=48 --mem=96G --time=01:00:00)")
done
for v in hifi3d facescape faceverse faceverse_neutral; do
    ids+=("$(BP_STEP=varifold BP_ARGS="$v --device cuda" sub -J "wbes-bp-vf-$v" --gres=gpu:l40s:1 --cpus-per-task=8 \
        --mem=32G --time=01:00:00)")
done
dep=$(IFS=:; echo "${ids[*]}")
BP_STEP=paired sub -J wbes-bp-paired --dependency="afterok:$dep" --cpus-per-task=32 --mem=96G --time=01:00:00
echo "job: ${ids[*]}"
