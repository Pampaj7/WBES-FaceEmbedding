#!/usr/bin/env bash
# E1: sottomette la catena corretta (emendamento del protocollo, 8 ottobre sera), nell'ordine di priorita' del PI
# (in FIFO l'ordine di sottomissione e' la priorita'):
#   1. C2F, C3F, C2F-GNM e C3F-UGT (seme 1234; C3F-UGT parte quando il cancello e1_gate_ugt.sbatch trova la GT
#      unificata verificata)
#   2. C2F e C3F, seme 2345
#   3. C3M rifatta su V100, C2M
#   4. C2F40, C3F40, G1
# Training su V100 (e1_train.sbatch), valutazioni sulle A100 di nv-ai-04 (e1_eval.sbatch, QoS unprivileged,
# --requeue), una per cella con afterok e --kill-on-invalid-dep; valutazione di C3M L40S (c3ml) subito; riepilogo
# con afterany su tutte le valutazioni. Dal frontend:
#
#   aau/evidence/e1_factorial/e1_launch.sh                 # tutto
#   aau/evidence/e1_factorial/e1_launch.sh c2f c3f         # solo queste celle (training + eval), senza riepilogo
#
# --mem (V100: lo staging del pre-pass va su /raid, fuori da --mem): picco previsto rss = RSS di base 40 GiB
# (misurato su 1060130: GT 65.600^2 in float64 piu' il resto; la GT unificata di C3F-UGT ha la stessa forma) + cache
# esatta del blocco piu' grande (design.md: <= 219.8 GiB per le celle con ICT, <= 146.3 per C2F-GNM, 112 per G1),
# +20%: 320G e 240G. Il picco vero lo registra mem_job.log di ogni run.
# --time: su V100 il pre-pass e' il collo di bottiglia (misurato: 10.0 CPU-s per mesh con 16 processi su nv-ai-03
# contro 4.9 su L40S, ~2 mesh/s per job): blocco F da ~17.500 mesh ~2.5 h, quindi celle F ~12-15 h, celle M (10
# blocchi fino all'epoca 72) ~25-30 h. Limiti larghi: 30 h e 60 h.
# Cancelli: tutti i training aspettano lo smoke V100 (e1_gate_smoke.sbatch: loss dell'epoca 1 entro il 5% di quella
# su L40S); C3F-UGT anche la GT unificata verificata (e1_gate_ugt.sbatch).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"
OUT="$AAU_RUNS/evidence/e1"
mkdir -p "$OUT"
SMOKE_JOB="${WBES_E1_SMOKE_JOB:?job dello smoke V100}"
ORDER=(c2f c3f c2fgnm c3fugt c2fs2 c3fs2 c3mv c2m c2f40 c3f40 g1)
declare -A MEM TIME
for c in "${ORDER[@]}"; do MEM[$c]=320G; TIME[$c]=30:00:00; done
MEM[c2fgnm]=240G; MEM[g1]=240G
TIME[c3mv]=60:00:00; TIME[c2m]=60:00:00
CELLS=("$@")
(( ${#CELLS[@]} )) || CELLS=("${ORDER[@]}")
EVALS=()
{
    echo
    echo "## Sottomissione $(date +%F_%T) (catena corretta: emendamento del protocollo)"
    if (( $# == 0 )); then
        v=$(WBES_E1_CELL=c3ml "$AAU_DIR/submit.sh" evidence/e1_factorial/e1_eval.sbatch --parsable \
            --job-name="wbes-e1-eval-c3ml")
        EVALS+=("$v")
        echo "- c3ml (C3M L40S, riferimento): eval $v su A100, subito"
    fi
    gs=$(WBES_E1_SMOKE_RUN="$OUT/train_c3mv_s1234_smokev100_$SMOKE_JOB" "$AAU_DIR/submit.sh" \
         evidence/e1_factorial/e1_gate_smoke.sbatch --parsable --dependency="afterany:$SMOKE_JOB")
    echo "- cancello smoke V100 $gs (afterany:$SMOKE_JOB): tutti i training ne dipendono"
    for c in "${CELLS[@]}"; do
        dep="afterok:$gs"
        if [[ "$c" == c3fugt ]]; then
            g=$("$AAU_DIR/submit.sh" evidence/e1_factorial/e1_gate_ugt.sbatch --parsable)
            dep="$dep:$g"
            echo "- c3fugt: cancello GT unificata $g"
        fi
        t=$(WBES_E1_CELL=$c "$AAU_DIR/submit.sh" evidence/e1_factorial/e1_train.sbatch \
            --parsable --job-name="wbes-e1-train-$c" --mem="${MEM[$c]}" --time="${TIME[$c]}" \
            --dependency="$dep" --kill-on-invalid-dep=yes)
        v=$(WBES_E1_CELL=$c "$AAU_DIR/submit.sh" evidence/e1_factorial/e1_eval.sbatch --parsable \
            --job-name="wbes-e1-eval-$c" --dependency="afterok:$t" --kill-on-invalid-dep=yes)
        EVALS+=("$v")
        echo "- $c: training $t (V100, --mem ${MEM[$c]}, --time ${TIME[$c]}, 32 CPU, pre-pass 22), eval $v (A100, afterok:$t)"
    done
    if (( $# == 0 )); then
        s=$("$AAU_DIR/submit.sh" evidence/e1_factorial/e1_summarize.sbatch --parsable \
            --dependency="afterany:$(IFS=:; echo "${EVALS[*]}")")
        echo "- riepilogo $s (afterany sulle eval)"
    fi
} | tee -a "$OUT/jobs.md"
