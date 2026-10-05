#!/usr/bin/env bash
#SBATCH --job-name=wbes-bl-geom
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#
# Varifold e currents senza sottocampionamento casuale: misura quantizzata su griglia (vedi
# geometric_matrix.py). Il kernel di phase0 non gira su CUDA com'e' scritto (somma dentro un
# tensore CPU), ed e' per questo che c'e' geometric_kernel.py.
#   aau/submit.sh baselines/geometric.sbatch
#   WBES_BL_ARGS="--subject-set facebench_first100 --out-root aau/runs/baselines_fb100" \
#     aau/submit.sh baselines/geometric.sbatch
#
# A40 e non T4, e 20 h e non 8: col passaggio a --normalize maxabs la mesh normalizzata e'
# ~1.6x piu' grande, quindi alla stessa cella (sigma_min/4) gli atomi passano da ~8000 a
# ~21000 e il costo, che e' O(N*M), da 122 a 890 ms/coppia su T4 (misurato, job 1019724).
# 21 coppie di topologie x 4950 coppie di soggetti fanno 26 h di T4 per set; su A40, che e'
# ~2.8x piu' veloce, sono ~9 h. Se le A40 sono occupate si spezza per coppie di topologie,
# e il job e' comunque ripartibile (salta le matrici gia' su disco):
#   WBES_BL_ARGS="--topology-pairs original:noisy,noisy:original" \
#     aau/submit.sh baselines/geometric.sbatch --gres=gpu:t4:1 --time=04:00:00
#
# Variante a massa unitaria nel frame maxabs (matrici in varifold_maxabs_unitmass e
# currents_maxabs_unitmass, accanto alle altre). Le sigma sono 1.848x quelle di phase0, quindi
# la cella e' piu' grossa e gli atomi meno: basta una A10.
#   WBES_BL_ARGS="--normalize maxabs_unitmass" \
#     aau/submit.sh baselines/geometric.sbatch --gres=gpu:a10:1 --time=06:00:00
#
# Solo CPU (24 core, giorni invece di ore): non praticabile a maxabs, resta per la variante
# area.
#   WBES_BL_ARGS="--device cpu --normalize area" aau/submit.sh baselines/geometric.sbatch \
#     --partition=cpu --gres=NONE --cpus-per-task=24
set -euo pipefail

if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${GPU_DEVICE_ORDINAL:-}" ]]; then
    export AAU_NV=""
fi
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

DEVICE_DEFAULT="cuda"
if [[ -z "${AAU_NV}" ]]; then DEVICE_DEFAULT="cpu"; fi

"$AAU_DIR/run.sh" aau/baselines/geometric_matrix.py \
    --device "$DEVICE_DEFAULT" --block 4096 \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${WBES_BL_ARGS:-}
