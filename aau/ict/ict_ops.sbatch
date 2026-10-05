#!/usr/bin/env bash
#SBATCH --job-name=wbes-ict-ops
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#
# ICT-5000, passo 4: operatori DiffusionNet con k_eig 128 e convenzione ad AREA UNITARIA
# (lo standard deciso in paper/PLAN_CVPR2027.md:48 dopo l'esperimento pot_area).
#
#   aau/submit.sh ict/ict_ops.sbatch                                   # 30.000 topologie
#   WBES_ICT_OPS_IN=$PWD/datasets/ICT/expressions \
#   WBES_ICT_OPS_OUT=$PWD/datasets/ICT/expressions_withops \
#     aau/submit.sh ict/ict_ops.sbatch                                 # 7.500 espressioni
#
# Gemello di aau/precompute_ops_areanorm.sbatch: stesso script di calcolo
# (`v2_work/potential/areanorm_operators.py`, usato verbatim), stesso sharding su lista
# ordinata completa, stessa ripartibilita' (risottometti e riprende dai mancanti).
# Cambia solo la coppia di directory.
#
# TEMPI: misurati nello smoke (job 1019281), 6 topologie di 1 identita' in 18 s a
# processo singolo, cioe' ~3 s per mesh => 25 CPU-ore per le 30.000, ~65 min su 24 core.
# Le 7.500 espressioni sono tutte grandi come `original`, ~7 CPU-ore, ~20 min.
#
# SPAZIO: `potential_operators.save_npz` usa np.savez NON compresso. Misurato nello stesso
# smoke: 59 MB per identita' (le 6 topologie insieme) => ~295 GB per le 30.000 mesh, piu'
# ~90 GB per le espressioni. La quota e' 1 TB e la home ne usava 148 GB prima di questo
# dataset: ci sta, ma non e' molto quello che avanza.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

# `ls dir/*.npz` esplode con "Argument list too long" a 30.000 file (misurato: job 1019283
# e' fallito cosi' DOPO aver scritto correttamente tutte le mesh). find non espande niente.
count_npz() { find "$1" -maxdepth 1 -name "$2" -type f -o -maxdepth 1 -name "$2" -type l | wc -l; }

aau_require_venv
aau_require_diffusion_net

IN_DIR="${WBES_ICT_OPS_IN:-$WBES_ROOT/datasets/ICT/topo}"
OUT_DIR="${WBES_ICT_OPS_OUT:-$WBES_ROOT/datasets/ICT/topo_withops}"
K_EIG="${WBES_K_EIG:-128}"
N_SHARDS="${WBES_OPS_CORES:-${SLURM_CPUS_PER_TASK:-8}}"
# Stesso motivo del gemello aau/precompute_ops_areanorm.sbatch: lo shard e' uno stride
# `files[i::n]` sulla lista ordinata, e i nomi sono ordinati per identita' x 6 topologie,
# quindi con n multiplo di 6 ogni worker riceve UNA SOLA topologia (125 `up60k` da 24k
# vertici contro 125 `down8k` da 3k) e il wall del job e' quello dello shard piu' lento.
# Un n coprimo con 6 mescola le topologie. Innocuo sulle espressioni, che sono tutte uguali.
if (( N_SHARDS % 6 == 0 )); then
    N_SHARDS=$(( N_SHARDS - 1 ))
fi

if [[ ! -d "$IN_DIR" ]]; then
    echo "ERRORE: mesh non trovate in $IN_DIR" >&2
    exit 1
fi

# Un processo per core, ognuno single-thread: senza questo torch/BLAS aprono un pool di
# thread per processo e i worker si contendono i core a vicenda.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

N_IN=$(count_npz "$IN_DIR" "*.npz")
echo "[ict-ops] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[ict-ops] in=$IN_DIR ($N_IN npz)"
echo "[ict-ops] out=$OUT_DIR  k_eig=$K_EIG  shard=$N_SHARDS"

mkdir -p "$OUT_DIR"

# Un log per shard: N processi che scrivono le stesse righe di progresso sullo stesso file
# le renderebbero illeggibili (nessuna riga porta l'indice dello shard).
SHARD_LOGS="$AAU_LOGS/${SLURM_JOB_NAME:-wbes-ict-ops}-${SLURM_JOB_ID:-manual}.shards"
mkdir -p "$SHARD_LOGS"

pids=()
for (( i = 0; i < N_SHARDS; i++ )); do
    "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
      --input-dir "$IN_DIR" \
      --output-dir "$OUT_DIR" \
      --k-eig "$K_EIG" \
      --shard "$i/$N_SHARDS" \
      > "$SHARD_LOGS/shard$i.log" 2>&1 &
    pids+=($!)
done
echo "[ict-ops] $N_SHARDS shard avviati, log in $SHARD_LOGS"

rc=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "[ict-ops] ERRORE: lo shard $i e' uscito != 0 (vedi $SHARD_LOGS/shard$i.log)" >&2
        rc=1
    fi
done

grep -H '^done ' "$SHARD_LOGS"/shard*.log || true
n_fail=$(grep -h '^done ' "$SHARD_LOGS"/shard*.log | sed 's/.*fail=\([0-9]*\).*/\1/' | awk '{s+=$1} END {print s+0}')
N_OUT=$(count_npz "$OUT_DIR" "*.npz")
echo "[ict-ops] file prodotti: $N_OUT / $N_IN  (fail totali riportati: ${n_fail:-?})"
echo "[ict-ops] dimensione: $(du -sh "$OUT_DIR" | cut -f1)"

if (( N_OUT != N_IN )); then
    echo "ERRORE: output incompleto ($N_OUT/$N_IN). Risottometti: riprende dai mancanti." >&2
    rc=1
fi
if (( rc != 0 )); then
    exit 1
fi

# --- verifica: area totale 1 e chiavi complete su 3 file di topologie diverse ----------
# ATTENZIONE: il python qui sotto e' UN SOLO argomento single-quoted della shell, quindi non
# puo' contenere apostrofi (usa le doppie). Il job 1019307 e' morto cosi', con
# "NameError: name 'nessuna' is not defined", DOPO aver scritto correttamente tutti i 30.000
# file: l'apostrofo di 'nessuna' chiudeva la stringa e la parola finiva nuda nel codice.
echo "[ict-ops] verifica su 3 file (chiavi, area totale, k_eig)"
WBES_CHECK_DIR="$OUT_DIR" "$AAU_DIR/run.sh" -c '
import glob, os
import numpy as np

out = os.environ["WBES_CHECK_DIR"]
files = sorted(glob.glob(os.path.join(out, "*.npz")))
picks = [files[0], files[len(files) // 2], files[-1]]
need = {"verts", "faces", "mass", "evals", "evecs",
        "L_indices", "L_values", "L_shape",
        "gradX_indices", "gradX_values", "gradX_shape",
        "gradY_indices", "gradY_values", "gradY_shape"}
bad = 0
for p in picks:
    with np.load(p, allow_pickle=False) as z:
        keys = set(z.files)
        V, F = z["verts"].astype(np.float64), z["faces"]
        t = V[F]
        A = float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())
        k = int(z["evals"].shape[0])
        nv, nf = len(V), len(F)
    miss = need - keys
    miss_txt = ", ".join(sorted(miss)) if miss else "nessuna"
    print(f"  {os.path.basename(p)}: verts={nv} faces={nf} k_eig={k} area={A:.6f} "
          f"chiavi_mancanti={miss_txt}")
    if miss or abs(A - 1.0) > 1e-3 or k != 128:
        bad += 1
print("  verifica OK" if bad == 0 else f"  VERIFICA FALLITA su {bad} file")
raise SystemExit(1 if bad else 0)
'
