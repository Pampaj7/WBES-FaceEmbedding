#!/usr/bin/env bash
# Test con espressioni (GT d'identita' neutra) su FaceVerse, in un comando: dati -> eval dei modelli
# nelle due convenzioni di frame -> baseline -> tabella, con afterok. Gira sul frontend.
#
#   aau/zs3dmm/ws_zs_expr.sh                 # -> aau/runs/ws_faceverse_expr/summary.md
#   aau/zs3dmm/ws_zs_expr.sh --skip-build    # vista con espressioni gia' pronta
#   aau/zs3dmm/ws_zs_expr.sh --skip-build --skip-baselines   # baseline gia' lanciate a parte
#
# Il protocollo ($ZS_RUNS/protocol.md) va dichiarato PRIMA: senza, lo script si ferma.
#
# Eval: solo breakdown (WBES_ZS_PART=topology) + embedding di ogni mesh (WBES_ZS_EMBED=1), come i
# test del frame. Righe di riferimento fissate dal protocollo -- BFM-only in convenzione BFM
# (rotazione nativa + facce invertite, WBES_ZS_FLIP_FACES=1), ICT-only in convenzione ICT
# (WBES_ZS_FRAME=x,-y,-z), congiunto in entrambe -- su 4 GPU in due job da 2; poi le due
# combinazioni secondarie (BFM-only in ICT, ICT-only in BFM), un job da 1 GPU ciascuna dopo i
# primi: mai piu' di 4 GPU insieme.
set -euo pipefail

AAU="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WBES_ZS_DOMAIN=fv
export WBES_ZS_EXPR=1
source "$AAU/env.sh"
source "$AAU/zs3dmm/zs_env.sh"

SKIP_BUILD=0
SKIP_BL=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --skip-baselines) SKIP_BL=1 ;;
        *) echo "uso: aau/zs3dmm/ws_zs_expr.sh [--skip-build] [--skip-baselines]" >&2; exit 2 ;;
    esac
done
if [[ ! -f "$ZS_RUNS/protocol.md" ]]; then
    echo "ERRORE: $ZS_RUNS/protocol.md assente: dichiara il protocollo prima delle eval" >&2
    exit 1
fi

DEP=()
if (( SKIP_BUILD )); then
    zs_require_view
else
    zs_require_model
    build=$("$AAU/submit.sh" zs3dmm/zs_build_expr.sbatch --parsable --job-name=wbes-zs-build-fvexpr)
    echo "[ws-zs-expr] dati: job $build"
    DEP=(--dependency="afterok:$build")
fi

# GPU: L40S di default; WBES_ZS_EXPR_GPU=a10 quando le L40S sono tutte occupate (FaceVerse su A10:
# ~2x il tempo, nodi da 244 GB: --mem 120G basta, MaxRSS 58 GB per 3 bracci + 18 GB di operatori
# su /tmp, job 1056092).
GPU="${WBES_ZS_EXPR_GPU:-l40s}"
export WBES_ZS_PART=topology WBES_ZS_EMBED=1
if [[ "$GPU" == l40s ]]; then EVAL_OPTS=(--mem=160G --time=05:00:00); else EVAL_OPTS=(--mem=120G --time=10:00:00); fi
evals=()
submit_eval() {  # <convenzione bfm|ict> <bracci> <n gpu> [dipendenza]
    local conv="$1" arms="$2" ngpu="$3" dep="${4:-}"
    local env=(WBES_ZS_ARMS="$arms")
    if [[ "$conv" == bfm ]]; then env+=(WBES_ZS_FLIP_FACES=1 WBES_ZS_FRAME=); else env+=(WBES_ZS_FRAME=x,-y,-z WBES_ZS_FLIP_FACES=0); fi
    local d=("${DEP[@]}")
    [[ -n "$dep" ]] && d=(--dependency="$dep")
    env "${env[@]}" "$AAU/submit.sh" zs3dmm/zs_zeroshot.sbatch --parsable \
        --job-name="wbes-zs-fvexpr-$conv-${arms// /-}" --gres="gpu:$GPU:$ngpu" \
        --cpus-per-task=$(( 10 * ngpu )) "${d[@]}" "${EVAL_OPTS[@]}"
}
a=$(submit_eval bfm "joint bfm_only" 2); echo "[ws-zs-expr] convenzione BFM, joint+bfm_only: job $a"
b=$(submit_eval ict "joint ict_only" 2); echo "[ws-zs-expr] convenzione ICT, joint+ict_only: job $b"
c=$(submit_eval bfm "ict_only" 1 "afterany:$a"); echo "[ws-zs-expr] convenzione BFM, ict_only (secondaria): job $c"
d=$(submit_eval ict "bfm_only" 1 "afterany:$b"); echo "[ws-zs-expr] convenzione ICT, bfm_only (secondaria): job $d"
evals=("$a" "$b" "$c" "$d")

bl=()
if (( ! SKIP_BL )); then
    for shard in 0/2 1/2; do
        j=$(WBES_ZS_BL_STEPS=align WBES_ZS_BL_SHARD=$shard "$AAU/submit.sh" zs3dmm/zs_baselines.sbatch \
            --parsable --job-name=wbes-zs-bl-fvexpr "${DEP[@]}")
        echo "[ws-zs-expr] baseline align $shard: job $j"
        bl+=("$j")
    done
    j=$(WBES_ZS_BL_STEPS=rank "$AAU/submit.sh" zs3dmm/zs_baselines.sbatch --parsable --job-name=wbes-zs-blrank-fvexpr \
        --dependency="afterok:${bl[0]}:${bl[1]}" --cpus-per-task=4 --time=00:30:00)
    bl+=("$j")
    j=$("$AAU/submit.sh" zs3dmm/zs_expr_extra.sbatch --parsable --job-name=wbes-zs-expr-extra-fv "${DEP[@]}")
    echo "[ws-zs-expr] baseline stesso soggetto + regione stabile: job $j"
    bl+=("$j")
fi
after="$(IFS=:; echo "${evals[*]}${bl[*]:+:}${bl[*]}")"
after="${after// /:}"
sum=$("$AAU/submit.sh" zs3dmm/zs_expr_summarize.sbatch --parsable --job-name=wbes-zs-expr-sum-fv \
      --dependency="afterok:$after")
echo "[ws-zs-expr] tabella: job $sum"
echo "[ws-zs-expr] output in $ZS_RUNS/summary.md"
