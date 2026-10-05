#!/usr/bin/env bash
#SBATCH --job-name=wbes-precompute-areanorm
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili, e un
# --output relativo viene risolto nella submit dir (se non esiste, Slurm scarta il job in
# un secondo senza scrivere niente). Se sposti il repo, aggiorna queste tre righe oppure
# sottometti con aau/submit.sh, che le ricalcola e le passa da riga di comando.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#
# Come precompute_ops.sbatch, ma con la convenzione WS2: ogni mesh viene centrata e
# riscalata ad AREA TOTALE 1 prima di costruire gli operatori (legge di Weyl,
# lambda_k ~ 4*pi*k/A). npz_data_topo_500 -> npz_data_topo_500_withops_areanorm.
#
#   aau/submit.sh precompute_ops_areanorm.sbatch
#
# Lo script di calcolo e' `v2_work/potential/areanorm_operators.py`, lo stesso che ha
# prodotto gli operatori dell'esperimento `pot_area`: usato verbatim, non ha percorsi
# hardcodati (risolve diffusion-net da `parents[1]`, cioe' la root del repo) e prende
# input/output da riga di comando. Nessun file di v2_work/ viene toccato.
#
# Parallelismo: quello script e' single-process con `--shard i/n`, non ha una Pool come
# `precompute_operators_npz.py`, quindi qui si lanciano N processi in background, uno per
# core allocato. Niente `taskset` (lo usa `v2_work/potential/shard_areanorm.sh` perche'
# li' il nodo era condiviso e non gestito): sotto Slurm il job e' gia' confinato nel suo
# cpuset. Lo sharding e' fatto sulla lista ORDINATA COMPLETA e poi filtrato per i file
# gia' esistenti, quindi gli stripe sono disgiunti anche se i worker partono in momenti
# diversi, ed e' RIPARTIBILE: se il job muore basta risottometterlo.
#
# Tempi: il precompute della cartella di controllo (job 1019208, stesso k_eig, stessi
# 3000 npz, 24 core) ha misurato 25m56s con MaxRSS 11.9 GB. Qui il lavoro per mesh e' lo
# stesso piu' il calcolo dell'area; --time=3h e' la stessa abbondanza del gemello.
#
# ATTENZIONE alla dimensione: `potential_operators.save_npz` usa `np.savez` (non
# compresso) mentre `precompute_operators_npz.py` usa `np.savez_compressed`. Stesse
# chiavi e stessi valori, ma su disco questa cartella pesa ~1.65x l'altra (~71 GB contro
# 43 GB): tienine conto nel --mem del training con WBES_STAGE_TMP=1, perche' /tmp e' RAM.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net

IN_DIR="${WBES_REMESH_NOOPS_DIR:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500}"
# NON si usa WBES_DATA_DIR come default (env.sh lo punta sempre alla cartella withops di
# controllo: ci scriveremmo dentro gli operatori area-norm).
OUT_DIR="${WBES_AREANORM_DIR:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500_withops_areanorm}"
REF_DIR="${WBES_REF_OPS_DIR:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500_withops}"
K_EIG="${WBES_K_EIG:-128}"
N_SHARDS="${WBES_OPS_CORES:-${SLURM_CPUS_PER_TASK:-8}}"
# Lo shard e' uno stride `files[i::n]` sulla lista ordinata, e i nomi sono ordinati per
# identita' x 6 topologie: con n multiplo di 6 ogni worker riceve UNA SOLA topologia, cioe'
# 125 `up60k` (60k vertici) contro 125 `down8k`. Misurato sul job 1019247 con n=24: shard da
# 6.2 min accanto a shard da 22+ min, e il wall del job e' quello del piu' lento. Un n
# coprimo con 6 mescola le topologie e pareggia gli stripe; togliere un processo su 24 costa
# molto meno dello sbilanciamento.
if (( N_SHARDS % 6 == 0 )); then
    N_SHARDS=$(( N_SHARDS - 1 ))
fi

if [[ ! -d "$IN_DIR" ]]; then
    echo "ERRORE: dataset mesh-only non trovato in $IN_DIR" >&2
    echo "  Estrailo prima con: aau/submit.sh extract_remesh.sbatch" >&2
    exit 1
fi

# Un processo per core, ognuno single-thread: senza questo torch/BLAS aprono un pool di
# thread per processo e i worker si contendono i core a vicenda.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

N_IN=$(ls -1 "$IN_DIR"/*.npz | wc -l)
echo "[areanorm] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[areanorm] in=$IN_DIR ($N_IN npz)"
echo "[areanorm] out=$OUT_DIR  k_eig=$K_EIG  shard=$N_SHARDS"

mkdir -p "$OUT_DIR"

# Un log per shard: 24 processi che scrivono le stesse righe di progresso sullo stesso
# file le renderebbero illeggibili (nessuna delle righe porta l'indice dello shard).
SHARD_LOGS="$AAU_LOGS/${SLURM_JOB_NAME:-wbes-precompute-areanorm}-${SLURM_JOB_ID:-manual}.shards"
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
echo "[areanorm] $N_SHARDS shard avviati, log in $SHARD_LOGS"

rc=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "[areanorm] ERRORE: lo shard $i e' uscito != 0 (vedi $SHARD_LOGS/shard$i.log)" >&2
        rc=1
    fi
done

# Il riepilogo di ogni shard e' l'ultima riga del suo log: "done ok=N fail=M in T min".
grep -H '^done ' "$SHARD_LOGS"/shard*.log || true
n_fail=$(grep -h '^done ' "$SHARD_LOGS"/shard*.log | sed 's/.*fail=\([0-9]*\).*/\1/' | awk '{s+=$1} END {print s+0}')
N_OUT=$(ls -1 "$OUT_DIR"/*.npz 2>/dev/null | wc -l)
echo "[areanorm] file prodotti: $N_OUT / $N_IN  (fail totali riportati: ${n_fail:-?})"
echo "[areanorm] dimensione: $(du -sh "$OUT_DIR" | cut -f1)"

if (( N_OUT != N_IN )); then
    echo "ERRORE: output incompleto ($N_OUT/$N_IN). Risottometti: lo script riprende i mancanti." >&2
    rc=1
fi
if (( rc != 0 )); then
    exit 1
fi

# --- verifica: stesso schema di chiavi della cartella di controllo, area totale = 1 ----
echo "[areanorm] verifica su 3 file (chiavi vs $REF_DIR, area totale, evals ~ evals_ref*A)"
WBES_CHECK_DIR="$OUT_DIR" WBES_CHECK_REF="$REF_DIR" WBES_CHECK_RAW="$IN_DIR" \
"$AAU_DIR/run.sh" -c '
import os
import numpy as np

out = os.environ["WBES_CHECK_DIR"]
ref = os.environ["WBES_CHECK_REF"]
raw = os.environ["WBES_CHECK_RAW"]

def area(V, F):
    t = V[F]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())

bad = 0
# tre topologie con comportamenti di area diversi: original (riferimento), crop (-15%),
# up60k (retessellazione fitta).
for name in ("id0000_GTready_original.npz", "id0000_GTready_crop.npz", "id0000_GTready_up60k.npz"):
    with np.load(os.path.join(out, name), allow_pickle=False) as z:
        ko = sorted(z.files)
        A = area(z["verts"].astype(np.float64), z["faces"])
        ev = np.asarray(z["evals"], dtype=np.float64)
        nv, nf = z["verts"].shape[0], z["faces"].shape[0]
    with np.load(os.path.join(ref, name), allow_pickle=False) as z:
        kr = sorted(z.files)
        ev_ref = np.asarray(z["evals"], dtype=np.float64)
    with np.load(os.path.join(raw, name), allow_pickle=False) as z:
        A_raw = area((z["verts"] if "verts" in z else z["V"]).astype(np.float64),
                     z["faces"] if "faces" in z else z["F"])
    same = "OK" if ko == kr else "DIVERSE: " + str(set(ko) ^ set(kr))
    # Weyl: gli autovalori della mesh ad area 1 devono valere quelli grezzi per A.
    r = float(np.median(ev[1:] / np.maximum(ev_ref[1:] * A_raw, 1e-30)))
    print(f"  {name}: verts={nv} faces={nf} chiavi={same} area={A:.6f} "
          f"area_grezza={A_raw:.4g} evals/(evals_ref*A_grezza) mediana={r:.4f}")
    if ko != kr or abs(A - 1.0) > 1e-3:
        bad += 1
print("  verifica OK" if bad == 0 else f"  VERIFICA FALLITA su {bad} file")
raise SystemExit(1 if bad else 0)
'
