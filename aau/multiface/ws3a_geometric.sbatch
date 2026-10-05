#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws3a-geom
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# WS3a, distanze geometriche (chamfer, rigid ICP + chamfer, varifold, currents) e la riga
# di controllo bbox_proxy, sulle 8000 coppie del protocollo Multiface, per le configurazioni
# di topologia di WBES_WS3A_PAIRS (4 per `clean`, il default, 6 per `hard`).
#
#   aau/submit.sh multiface/ws3a_geometric.sbatch
#   WBES_WS3A_METRICS=varifold aau/submit.sh multiface/ws3a_geometric.sbatch   # una sola
#   WBES_WS3A_PAIRS=hard aau/submit.sh multiface/ws3a_geometric.sbatch --time=08:00:00
#
# Niente GPU: sono tutte cKDTree e prodotti kernel in torch CPU.  16 core e non 24 perche'
# la partizione cpu ha UN SOLO nodo da 32 core, condiviso con il precompute ICT.
#
# --time: stimato dal test su 200 coppie (--max-pairs-per-class 50) su 8 core, 169 s in
# tutto, di cui 61 s di caricamento (642 mesh) e 45 s di auto-correlazioni varifold.
# Estrapolando a 8520 mesh e 96000 coppie-metrica veniva ~1 h su 16 core.  Misurato poi
# sul giro vero (job 1019518, 16 core): 28m56s, cioe' 178 s di caricamento, 291 s di
# auto-correlazioni, 16 s di chamfer, 155 s di rigid ICP, 1084 s di varifold.  Le 4 h
# restano perche' il nodo cpu e' uno solo ed e' condiviso.
#
# Quel 28m56s NON vale piu' per il giro `hard`, per due motivi che si moltiplicano:
# `--max-tris` e' passato da 4000 a 12000 (il kernel e' quadratico nei triangoli, cioe'
# ~7.5x per distanza sulle topologie a ~11k facce) e le metriche di misura sono due invece
# di una, su 6 coppie di topologie invece di 4. Misurato sul giro hard (job 1019643, 24
# core): 4h17m, di cui 2h05m di varifold e 2h05m di currents. Il passaggio a
# --normalize maxabs non cambia il costo (il numero di triangoli e' lo stesso, cambia solo
# la similarita' applicata prima del kernel) e bbox_proxy costa secondi.
#
# Il job e' ripartibile: ogni csv gia' presente viene saltato (senza --overwrite).
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv

# bbox_proxy e' nel default: e' la riga con cui si leggono tutte le altre (quanto della
# separazione si ottiene con quattro numeri banali sulla mesh gia' normalizzata), costa
# secondi, e lasciarla fuori vuol dire pubblicare una tabella senza il suo metro.
METRICS="${WBES_WS3A_METRICS:-chamfer,rigid_icp,varifold,currents,bbox_proxy}"
WORKERS="${WBES_WS3A_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[ws3a-geom] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[ws3a-geom] metriche=$METRICS workers=$WORKERS out=${WBES_WS3A_OUT:-$AAU_RUNS/multiface_ws3a}"

"$AAU_DIR/run.sh" aau/multiface/ws3a_geometric.py \
    --metrics "$METRICS" --workers "$WORKERS" "$@"

echo "[ws3a-geom] OK"
