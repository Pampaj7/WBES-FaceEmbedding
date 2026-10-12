#!/usr/bin/env bash
#SBATCH --job-name=wbes-pr
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#
# Protocolli della comunita' (aau/competitors/protocolli/): passi in sequenza nello stesso job, solo CPU (un job con
# molte CPU invece di tanti job: il tetto e' sul numero di job per utente).
#
#   PR_STEPS="define prep:hifi3d selftest:hifi3d sample:hifi3d official:hifi3d" sbatch pr.sbatch
#   PR_STEPS="run:hifi3d" PR_METHODS=now_original,now_metrical,stirling PR_SHARD=0 PR_NSHARDS=2 sbatch pr.sbatch
#   PR_STEPS="run:famos@now_original,now_metrical,stirling,realy merge:famos" sbatch pr.sbatch   (metodi per passo)
#   PR_STEPS="merge:hifi3d" sbatch --cpus-per-task=2 --mem=32G pr.sbatch
#   PR_STEPS="eval" sbatch --mem=192G pr.sbatch                         (Spearman con IC; LEGGE LE GT di test)
# Dominio nuovo (es. Ava-256, NON prima del protocollo confermativo): PR_SPEC=<json> (pr_common.py), e "define" usa il
# blocco "landmarks" dello spec. Opzioni: PR_NOW_OPT (punti dell'ICP di NoW, 0 = tutti), PR_REALY_CAP, PR_N.
set -euo pipefail
export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
W="${SLURM_CPUS_PER_TASK:-8}"
D=aau/competitors/protocolli
RUN=(v3_work/unified_gt/run.sh -u)
SPEC=()
[[ -n "${PR_SPEC:-}" ]] && SPEC=(--spec "$PR_SPEC")
echo "[pr] host=$(hostname) job=${SLURM_JOB_ID:-none} passi=${PR_STEPS:?PR_STEPS} cpu=$W $(date +%T)"
for tok in $PR_STEPS; do
    step="${tok%%:*}"; view=""; [[ "$tok" == *:* ]] && view="${tok#*:}"
    meth="${PR_METHODS:-}"; [[ "$view" == *@* ]] && { meth="${view#*@}"; view="${view%%@*}"; }
    t0=$(date +%s)
    echo "[pr] passo $step $view $(date +%T)"
    case "$step" in
        define)
            if [[ -n "${PR_SPEC:-}" ]]; then "${RUN[@]}" $D/pr_landmarks.py define "${SPEC[@]}"
            else for d in hifi3d facescape faceverse; do "${RUN[@]}" $D/pr_landmarks.py define --domain "$d"; done; fi ;;
        prep)     "${RUN[@]}" $D/pr_landmarks.py prep --view "$view" "${SPEC[@]}" --workers "$W" \
                      --realy-cap "${PR_REALY_CAP:-1500}" ;;
        selftest) "${RUN[@]}" $D/pr_selftest.py --view "$view" || echo "[pr] ATTENZIONE: selftest FALLITO (vedi sopra)" ;;
        sample)   "${RUN[@]}" $D/pr_run.py sample --view "$view" "${SPEC[@]}" --n "${PR_N:-200}" --workers "$W" \
                      --now-opt "${PR_NOW_OPT_SAMPLE:-4000}" ;;
        official)
            "${RUN[@]}" $D/pr_official_check.py export --view "$view" --n "${PR_N_OFFICIAL:-20}"
            VENV="$WBES_ROOT/external/venvs/now" MPLBACKEND=Agg OMP_NUM_THREADS=1 singularity exec "$CONTAINER" \
                bash -c 'source "$VENV/bin/activate"; exec python3 -u "$@"' _ $D/pr_official_check.py official --view "$view"
            "${RUN[@]}" $D/pr_official_check.py compare --view "$view" ;;
        run)      "${RUN[@]}" $D/pr_run.py run --view "$view" "${SPEC[@]}" --methods "${meth:?PR_METHODS o run:vista@metodi}" \
                      --shard "${PR_SHARD:-0}" --nshards "${PR_NSHARDS:-1}" --workers "$W" --now-opt "${PR_NOW_OPT:-0}" ;;
        merge)    "${RUN[@]}" $D/pr_run.py merge --view "$view" "${SPEC[@]}" ;;
        eval)     UGT_THREADS=1 "${RUN[@]}" $D/pr_eval.py spearman --workers "$W" ;;
        *) echo "passo sconosciuto: $step" >&2; exit 2 ;;
    esac
    echo "[pr] OK $step $view in $(( $(date +%s) - t0 ))s"
done
echo "[pr] OK tutti i passi $(date +%T)"
