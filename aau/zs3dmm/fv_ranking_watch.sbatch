#!/usr/bin/env bash
#SBATCH --job-name=wbes-zs-fv-watch
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=08:00:00
#
# Una tantum (5 ottobre 2026): i tre job FaceVerse 1055801-3 sono partiti come WBES_ZS_PART=all
# con --time=10h, che non basta a ranking clean+mixed + breakdown a ~12 coppie/s. Il breakdown
# e' rifatto in parallelo come pezzo topology (1055931-3); qui si aspetta che il ranking di
# ciascun job abbia scritto i suoi tre file e lo si cancella, invece di lasciargli bruciare
# ~3 h di L40S su un breakdown che il --time ucciderebbe comunque.
#   WBES_FV_WATCH_RUNS=<runs/data_fp> WBES_FV_WATCH="joint:1055801 ..." aau/submit.sh zs3dmm/fv_ranking_watch.sbatch
set -euo pipefail
RUNS="${WBES_FV_WATCH_RUNS:?}"
pending="${WBES_FV_WATCH:?}"
while [[ -n "$pending" ]]; do
    left=""
    for spec in $pending; do
        arm="${spec%%:*}"; job="${spec##*:}"
        r="$RUNS/$arm/zs_zeroshot/ranking"
        if [[ -f "$r/ranking_summary.json" && -f "$r/ranking_summary.csv" && -f "$r/ranking_summary.md" ]]; then
            echo "[fv-watch] $(date '+%F %T') ranking $arm completo: cancello $job"
            scancel "$job" || true
        elif ! squeue -j "$job" -h 2>/dev/null | grep -q .; then
            echo "[fv-watch] ERRORE: $job ($arm) finito senza ranking completo" >&2
            exit 1
        else
            left="$left $spec"
        fi
    done
    pending="$(echo $left)"
    [[ -n "$pending" ]] && sleep 60
done
echo "[fv-watch] OK"
