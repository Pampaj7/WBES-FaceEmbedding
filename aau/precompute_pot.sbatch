#!/usr/bin/env bash
#SBATCH --job-name=wbes-precompute-pot
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#
# Operatori DiffusionNet col "pozzo di potenziale" (pilota pot, aau/runs/pilot_pot/), con lo
# script dell'autore v2_work/potential/potential_operators.py invariato: offset GLOBALE
# (--alpha-mode global) al valore del ginocchio dello sweep, alpha=0.55 (v2_work/STATUS.md,
# sweep 2026-08-17 ~23:10), e scala comune = mediana della geodetica massima dal centro,
# calcolata da v2_work/potential/calibrate_alpha.py sulla collezione stessa (120 mesh, seed 0,
# come il 127507 di STATUS.md per BFM). Il suo "alpha" di sicurezza non si usa: si usa 0.55.
#
#   aau/submit.sh precompute_pot.sbatch                                   # BFM/REMESH
#   WBES_POT_IN=$PWD/datasets/ICT/eval_view_heldout \
#   WBES_POT_OUT=$PWD/datasets/ICT/eval_view_heldout_pot055 \
#   WBES_POT_CALIB=$PWD/aau/runs/pilot_pot/calib_ict.json \
#     aau/submit.sh precompute_pot.sbatch --cpus-per-task=6 --job-name=wbes-precompute-pot-ict
#
# La scala e' per collezione e non quella BFM anche su ICT: e' in unita' delle coordinate
# grezze (BFM ~1e5, ICT no), quindi la scala BFM piazzerebbe il pozzo a caso sulle mesh ICT.
#
# Un processo per core (--shard i/N, OMP=1), come precompute_ops.sbatch. Lo script salta i file
# gia' presenti e scrive con rename atomico: se il job muore basta risottometterlo. Alla fine
# il job fallisce se i file in uscita non sono quanti quelli in ingresso o se un worker ha
# scritto [fail], e lancia aau/models/check_pot_ops.py (chiavi, vertici, spettro diverso).
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net

# Nodo cpu senza GPU, come setup_env.sbatch.
export AAU_NV=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

IN_DIR="${WBES_POT_IN:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500}"
OUT_DIR="${WBES_POT_OUT:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500_withops_pot055}"
REF_DIR="${WBES_POT_REF:-$WBES_DATA_DIR}"   # operatori standard, per il controllo finale
CALIB="${WBES_POT_CALIB:-$AAU_RUNS/pilot_pot/calib_bfm.json}"
ALPHA="${WBES_POT_ALPHA:-0.55}"
N="${WBES_POT_SHARDS:-${SLURM_CPUS_PER_TASK:-8}}"
LOGDIR="$AAU_LOGS/pot_shards_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT_DIR" "$(dirname "$CALIB")" "$LOGDIR"

n_in="$(ls -1 "$IN_DIR"/*.npz | wc -l)"
echo "[pot] host=$(hostname) job=${SLURM_JOB_ID:-none} shards=$N"
echo "[pot] in=$IN_DIR ($n_in npz)"
echo "[pot] out=$OUT_DIR  ref=$REF_DIR  alpha=$ALPHA"

if [[ ! -f "$CALIB" ]]; then
    echo "[pot] calibrazione della scala -> $CALIB"
    # run.sh fa cd nella root: il path completo mette v2_work/potential su sys.path[0],
    # da cui calibrate_alpha importa potential_operators.
    "$AAU_DIR/run.sh" v2_work/potential/calibrate_alpha.py \
        --input-dir "$IN_DIR" --n-sample 120 --seed 0 --out "$CALIB"
fi
SCALE="$("$AAU_DIR/run.sh" -c "import json,sys; print(json.load(open(sys.argv[1]))['scale'])" "$CALIB")"
echo "[pot] scala comune=$SCALE (da $CALIB), alpha=$ALPHA"

pids=()
for ((i = 0; i < N; i++)); do
    "$AAU_DIR/run.sh" v2_work/potential/potential_operators.py \
        --input-dir "$IN_DIR" --output-dir "$OUT_DIR" --k-eig 128 --shard "$i/$N" \
        --alpha-mode global --alpha-value "$ALPHA" --scale-value "$SCALE" \
        > "$LOGDIR/shard_$i.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done

n_out="$(ls -1 "$OUT_DIR"/*.npz | wc -l)"
n_fail="$(cat "$LOGDIR"/shard_*.log | grep -c '^\[fail\]' || true)"
echo "[pot] prodotti $n_out/$n_in, [fail]=$n_fail, rc worker=$rc"
grep -h '^done' "$LOGDIR"/shard_*.log | sed 's/^/    /' || true
if (( rc != 0 || n_fail != 0 || n_out != n_in )); then
    grep -h '^\[fail\]' "$LOGDIR"/shard_*.log | head -20 >&2 || true
    echo "ERRORE: operatori incompleti, vedi $LOGDIR" >&2
    exit 1
fi
printf '{"alpha": %s, "scale": %s, "calib": "%s", "in": "%s", "job": "%s"}\n' \
    "$ALPHA" "$SCALE" "$CALIB" "$IN_DIR" "${SLURM_JOB_ID:-none}" > "$OUT_DIR.params.json"

"$AAU_DIR/run.sh" aau/models/check_pot_ops.py --pot-dir "$OUT_DIR" --ref-dir "$REF_DIR" --n 60
