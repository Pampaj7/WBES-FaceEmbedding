#!/usr/bin/env bash
#SBATCH --job-name=wbes-famos-build
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# FaMoS, dati: dall'estrazione verificata al set di test reale. Dopo l'estrazione delle registrazioni
# (7z, job 1062020):
#
#   aau/submit.sh famos/famos_build.sbatch --dependency=afterok:1062020
#
#   1. verifica + inventario  famos_inventory.py  -> aau/runs/evidence/famos/inventory.{json,md}
#   2. sottocampionamento     famos_subsample.py  -> datasets/FAMOS/{train,test}/<soggetto>.npz
#   3. GT unificata           famos_unified.py    -> datasets/FAMOS/unified, test_view/gt_matrix.npz
#   4. set di test reale      famos_test_view.py  -> datasets/FAMOS/test_view
#
# Lo split (aau/famos/split.json) e' dichiarato prima, da famos_split.py: qui si legge soltanto. Il
# passo 1 esce con errore se l'estrazione non coincide con l'elenco dell'archivio, e allora non parte
# nient'altro. Riprendibile: il passo 2 salta i soggetti gia' scritti (WBES_FAMOS_OVERWRITE=1 per
# rifarli). Nessuna cancellazione qui: archivi e grezzi non usati si tolgono a mano dopo le verifiche.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
export AAU_NV=""
cd "$WBES_ROOT"
aau_require_venv

N_CORES="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LISTING="${WBES_FAMOS_LISTING:-$WBES_ROOT/datasets/FAMOS/reg_archive_listing.tsv}"
LOG7Z="${WBES_FAMOS_7Z_LOG:-$WBES_ROOT/external_data/famos/extracted/x_reg_1062020.log}"
OVERWRITE=(); [[ "${WBES_FAMOS_OVERWRITE:-0}" == 1 ]] && OVERWRITE=(--overwrite)

echo "[famos-build] host=$(hostname) job=${SLURM_JOB_ID:-none} cores=$N_CORES"
echo "[famos-build] 1/4 verifica dell'estrazione e inventario"
"$AAU_DIR/run.sh" aau/famos/famos_inventory.py --listing "$LISTING" --log "$LOG7Z" --workers "$N_CORES"

echo "[famos-build] 2/4 sottocampionamento e forma neutra"
"$AAU_DIR/run.sh" aau/famos/famos_subsample.py --stride "${WBES_FAMOS_STRIDE:-10}" --workers "$N_CORES" "${OVERWRITE[@]}"

echo "[famos-build] 3/4 GT unificata"
"$AAU_DIR/run.sh" aau/famos/famos_unified.py

echo "[famos-build] 4/4 set di test reale"
"$AAU_DIR/run.sh" aau/famos/famos_test_view.py --workers "$N_CORES"

echo "[famos-build] spazio:"; du -sh "$WBES_ROOT"/datasets/FAMOS/{train,test,unified,test_view}
echo "[famos-build] OK"
