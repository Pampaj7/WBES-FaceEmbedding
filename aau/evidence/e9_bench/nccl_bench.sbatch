#!/usr/bin/env bash
#SBATCH --job-name=wbes-e9-nccl
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#
# E9 (c): all_reduce NCCL. Di default 2 nodi L40S, 1 GPU ciascuno. Stesso nodo, 2 GPU:
#   sbatch --nodes=1 --ntasks-per-node=2 --gres=gpu:l40s:2 aau/evidence/e9_bench/nccl_bench.sbatch
# Rete dei nodi L40S (sonda dell'8 ottobre): bond0 = 2 x 25 GbE Mellanox, dispositivo RDMA mlx5_bond_0
# (RoCE, /dev/infiniband/uverbs0). NCCL_DEBUG=INFO dice se NCCL usa NET/IB (RoCE) o NET/Socket.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
export MASTER_PORT=$(( 29500 + ${SLURM_JOB_ID:-0} % 1000 ))
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
TAG="${SLURM_JOB_NUM_NODES}nodes_${SLURM_NTASKS}gpu_${SLURM_JOB_ID}"
echo "[nccl] nodi=$SLURM_JOB_NODELIST task=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT tag=$TAG"
# ripiego senza GPU (backend gloo, tensori CPU), per la sola rete fra due nodi L40S:
#   E9_BACKEND=gloo sbatch --gres=none --exclude=<nodi non L40S> aau/evidence/e9_bench/nccl_bench.sbatch
BACKEND="${E9_BACKEND:-nccl}"
if [[ "$BACKEND" == gloo ]]; then
    export AAU_NV="" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
    # NCCL potra' usare RoCE? dispositivi RDMA visti dentro il container
    srun --ntasks-per-node=1 bash -c 'echo "== $(hostname)"; ls /dev/infiniband; singularity exec "$CONTAINER" ibv_devinfo 2>&1 | grep -E "hca_id|transport|state|active_mtu|link_layer"' || true
fi
srun --kill-on-bad-exit=1 "$AAU_DIR/run.sh" aau/evidence/e9_bench/nccl_bench.py --backend "$BACKEND" \
    --out-json "$AAU_RUNS/evidence/e9/nccl/${BACKEND}_${TAG}.json"
echo "[e9-nccl] OK"
