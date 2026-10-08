#!/usr/bin/env bash
# E1: sottomette i quattro training (c2m, c2f, c3f, g1), la valutazione di ognuno (afterok) e quella dei
# due checkpoint di C3M che mancano (FaceVerse con espressioni e FLAME: HIFI3D e NoW di C3M esistono gia',
# aau/runs/data_scale_ood), poi il riepilogo (afterany). Dal frontend:
#
#   aau/evidence/e1_factorial/e1_launch.sh             # tutto
#   aau/evidence/e1_factorial/e1_launch.sh c2f g1      # solo queste celle (training + eval)
#
# --mem per cella: picco rss+shmem previsto da e1_design.py (cache esatta del blocco + /tmp del blocco
# successivo + RSS di base, misurati sul run su scala; il run su scala supera la previsione di 9.8 GiB),
# piu' quel 9.8, piu' il 15% circa. Il picco vero lo registra mem_job.log di ogni run.
# --time: ~1.3-1.4 volte la stima (c2m come i primi 72 epoche di 1060130: 9 h 28 min).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"
OUT="$AAU_RUNS/evidence/e1"
mkdir -p "$OUT"
declare -A MEM=([c2m]=430G [c2f]=430G [c3f]=410G [g1]=270G)
declare -A TIME=([c2m]=12:00:00 [c2f]=10:00:00 [c3f]=10:00:00 [g1]=10:00:00)
CELLS=("$@")
(( ${#CELLS[@]} )) || CELLS=(c2m c2f c3f g1)
EVALS=()
{
    echo
    echo "## Sottomissione $(date +%F_%T)"
    for c in "${CELLS[@]}"; do
        t=$(WBES_E1_CELL=$c "$AAU_DIR/submit.sh" evidence/e1_factorial/e1_train.sbatch --parsable \
            --job-name="wbes-e1-train-$c" --mem="${MEM[$c]}" --time="${TIME[$c]}")
        v=$(WBES_E1_CELL=$c "$AAU_DIR/submit.sh" evidence/e1_factorial/e1_eval.sbatch --parsable \
            --job-name="wbes-e1-eval-$c" --dependency="afterok:$t")
        EVALS+=("$v")
        echo "- $c: training $t (--mem ${MEM[$c]}, --time ${TIME[$c]}), eval $v (afterok:$t)"
    done
    if (( $# == 0 )); then
        v=$(WBES_E1_CELL=c3m WBES_E1_EVAL_STEPS="fv flame" "$AAU_DIR/submit.sh" evidence/e1_factorial/e1_eval.sbatch \
            --parsable --job-name="wbes-e1-eval-c3m" --time=01:30:00)
        EVALS+=("$v")
        echo "- c3m: eval FaceVerse + FLAME $v (HIFI3D e NoW: quelli di aau/runs/data_scale_ood)"
    fi
    s=$("$AAU_DIR/submit.sh" evidence/e1_factorial/e1_summarize.sbatch --parsable \
        --dependency="afterany:$(IFS=:; echo "${EVALS[*]}")")
    echo "- riepilogo $s (afterany sulle eval)"
} | tee -a "$OUT/jobs.md"
