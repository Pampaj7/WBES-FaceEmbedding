#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws3b-ops-areanorm
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# WS3b, operatori ad AREA UNITARIA per il latent di un modello areanorm (il congiunto
# BFM+ICT di WS2): gli stessi tre insiemi di mesh di ws3b_prep.sbatch e
# ws3b_gtclip_prep.sbatch, cioe' gt_face, recon_face/<metodo> e recon_face_gtclip/<metodo>,
# in cartelle gemelle col suffisso `_withops_areanorm` (le `_withops` restano al v1).
#
#   aau/submit.sh recon/ws3b_ops_areanorm.sbatch
#
# Poi il latent, con argomenti per lo script (sbatch diretto, vedi ws3b_gtclip_prep.sbatch):
#   WBES_CKPT=<congiunto> sbatch --gres=gpu:t4:1 aau/recon/ws3b_latent.sbatch \
#       --metric latent_joint --ops-suffix _withops_areanorm
#
# Lo script di calcolo e' `v2_work/potential/areanorm_operators.py`, usato verbatim come in
# `aau/multiface/multiface_ops_areanorm.sbatch`: single-process con `--shard i/n`, quindi
# N processi in background, uno per core.  Sharding sulla lista completa e poi filtro sui
# file gia' presenti: il job e' ripartibile.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net

OUT="${WBES_WS3B_OUT:-$AAU_RUNS/multiface_ws3b}"
K_EIG="${WBES_K_EIG:-128}"
N_SHARDS="${WBES_WS3B_CORES:-${SLURM_CPUS_PER_TASK:-8}}"
METHODS="${WBES_WS3B_METHODS:-3ddfa_v2 synergynet prnet}"
SUFFIX="_withops_areanorm"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[ws3b-areanorm] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT k_eig=$K_EIG shard=$N_SHARDS"

SHARD_LOGS="$AAU_LOGS/${SLURM_JOB_NAME:-wbes-ws3b-ops-areanorm}-${SLURM_JOB_ID:-manual}.shards"
mkdir -p "$SHARD_LOGS"

# <mesh>:<operatori>, relativi a $OUT, nello stesso layout delle cartelle `_withops`.
PAIRS="gt_face:gt_face$SUFFIX"
for m in $METHODS; do
    PAIRS="$PAIRS recon_face/$m:recon_face$SUFFIX/$m recon_face_gtclip/$m:recon_face_gtclip$SUFFIX/$m"
done

rc=0
for pair in $PAIRS; do
    IN_DIR="$OUT/${pair%%:*}"
    OUT_DIR="$OUT/${pair##*:}"
    tag="$(echo "${pair%%:*}" | tr '/' '_')"
    if [[ ! -d "$IN_DIR" ]]; then
        echo "ERRORE: mesh non trovate in $IN_DIR (gira prima ws3b_prep / ws3b_gtclip_prep)" >&2
        exit 1
    fi
    N_IN=$(find "$IN_DIR" -maxdepth 1 -name '*.npz' -type f | wc -l)
    mkdir -p "$OUT_DIR"
    echo "[ws3b-areanorm] ${pair%%:*}: $N_IN mesh -> $OUT_DIR"

    t0=$(date +%s)
    pids=()
    for (( i = 0; i < N_SHARDS; i++ )); do
        "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
            --input-dir "$IN_DIR" --output-dir "$OUT_DIR" \
            --k-eig "$K_EIG" --shard "$i/$N_SHARDS" \
            > "$SHARD_LOGS/$tag-shard$i.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "[ws3b-areanorm] ERRORE: shard $i di $tag != 0 (vedi $SHARD_LOGS/$tag-shard$i.log)" >&2
            rc=1
        fi
    done

    N_OUT=$(find "$OUT_DIR" -maxdepth 1 -name '*.npz' -type f | wc -l)
    echo "[ws3b-areanorm] ${pair%%:*}: $N_OUT / $N_IN npz in $(( ($(date +%s) - t0) / 60 )) min, $(du -sh "$OUT_DIR" | cut -f1)"
    if (( N_OUT != N_IN )); then
        echo "ERRORE: output incompleto per ${pair%%:*} ($N_OUT/$N_IN). Risottometti: riprende." >&2
        rc=1
    fi
done
if (( rc != 0 )); then
    exit 1
fi

# --- verifica: chiavi, k_eig, area totale 1, stessi vertici e facce a meno della scala ---
# check_ops.py vuole il layout di datasets/Multiface/prep (<topo> e <topo><suffisso>
# fratelli), che qui non vale per le ricostruzioni: stessi controlli, su un file per cartella.
echo "[ws3b-areanorm] verifica su una mesh per cartella"
WBES_CHECK_OUT="$OUT" WBES_CHECK_PAIRS="$PAIRS" "$AAU_DIR/run.sh" - <<'EOF'
import os, sys
from pathlib import Path
import numpy as np

NEEDED = {"verts", "faces", "mass", "evals", "evecs", "L_indices", "L_values", "L_shape",
          "gradX_indices", "gradX_values", "gradX_shape",
          "gradY_indices", "gradY_values", "gradY_shape"}
K_EIG = int(os.environ.get("WBES_K_EIG", "128"))

def area(V, F):
    t = V[F]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())

out, bad = Path(os.environ["WBES_CHECK_OUT"]), 0
for pair in os.environ["WBES_CHECK_PAIRS"].split():
    mesh_dir, ops_dir = (out / p for p in pair.split(":"))
    files = sorted(ops_dir.glob("*.npz"))
    pick = files[len(files) // 2]
    with np.load(pick, allow_pickle=False) as z:
        missing = NEEDED - set(z.files)
        V, F, k = z["verts"].astype(np.float64), z["faces"], int(z["evals"].shape[0])
        mass_min = float(np.asarray(z["mass"]).min())
    with np.load(mesh_dir / pick.name, allow_pickle=False) as z:
        V0, F0 = z["V"].astype(np.float64), z["F"]
    A, A0 = area(V, F), area(V0, F0)
    # La mesh deve essere quella di partenza centrata e divisa per sqrt(area): niente altro.
    same = bool(np.array_equal(F, F0)) and np.allclose(V, (V0 - V0.mean(0)) / np.sqrt(A0), atol=1e-5)
    print(f"  {ops_dir.relative_to(out)}: {len(files)} file, {pick.name} verts={len(V)} k_eig={k} "
          f"area={A:.6f} area_mesh={A0:.6g} mass_min={mass_min:.3g} stessa_mesh={same} "
          f"chiavi_mancanti={sorted(missing) if missing else 'nessuna'}")
    if missing or k != K_EIG or mass_min <= 0.0 or abs(A - 1.0) > 1e-3 or not same:
        bad += 1
print("  verifica OK" if bad == 0 else f"  VERIFICA FALLITA su {bad} cartelle")
sys.exit(1 if bad else 0)
EOF

echo "[ws3b-areanorm] OK"
