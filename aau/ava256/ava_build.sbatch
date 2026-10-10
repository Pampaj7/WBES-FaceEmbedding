#!/usr/bin/env bash
#SBATCH --job-name=wbes-ava256
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#
# Ava-256, test CONFERMATIVO: dati, GT e viste su CPU (nessun modello, nessun operatore, nessun embedding).
# Il download (ava_download.py) va fatto prima, sul frontend: serve la rete.
#
#   sbatch aau/ava256/ava_build.sbatch                          # tutto
#   AVA_STEPS="identities topo crop view" sbatch aau/ava256/ava_build.sbatch
#
#   template    ava_neutral.py template   template per la corrispondenza
#   corr        ava_corr.py               regione FLAME / unificata sulla topologia, patch, controllo Multiface
#   neutral     ava_neutral.py neutral    neutre e neutri ripetuti (regola FaMoS), controlli dei frame
#   gt          ava_gt.py                 unita', taglia, qualita', GT FR / SR, affidabilita' della GT
#   identities  ava_views.py identities   patch della neutra in mm, frame nativo
#   topo        make_zs_topologies.py     le 6 topologie (codice di HIFI3D / FaceVerse / FaceScape dev)
#   crop        zs_check_crop.py          il crop differisce dalla original per ogni identita'
#   view        ava_views.py view         eval_view (symlink id950000-, GT rinominate)
set -euo pipefail

if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${GPU_DEVICE_ORDINAL:-}" ]]; then
    export AAU_NV=""
fi
STEPS="${AVA_STEPS:-template corr neutral gt identities topo crop view}"
UGT="v3_work/unified_gt/run.sh"
AAU="aau/run.sh"
N_CORES="${SLURM_CPUS_PER_TASK:-8}"
export UGT_THREADS="${UGT_THREADS:-8}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[ava256] host=$(hostname) job=${SLURM_JOB_ID:-none} steps: $STEPS"
for step in $STEPS; do
    t0=$(date +%s)
    case "$step" in
        template)   UGT_THREADS=1 $UGT aau/ava256/ava_neutral.py template --workers "$N_CORES" ;;
        corr)       $UGT aau/ava256/ava_corr.py ;;
        neutral)    UGT_THREADS=1 $UGT aau/ava256/ava_neutral.py neutral --workers "$N_CORES" ;;
        gt)         $UGT aau/ava256/ava_gt.py ;;
        identities) $UGT aau/ava256/ava_views.py identities ;;
        topo)       $AAU aau/zs3dmm/make_zs_topologies.py --prefix ava --in-dir datasets/AVA256/identities \
                        --out-dir datasets/AVA256/topo --n-cores "$N_CORES" --overwrite
                    $AAU aau/zs3dmm/make_zs_topologies.py --demo --prefix ava --in-dir datasets/AVA256/identities \
                        --out-dir datasets/AVA256/topo ;;
        crop)       $AAU aau/zs3dmm/zs_check_crop.py --topo-dir datasets/AVA256/topo --prefix ava --write ;;
        view)       $UGT aau/ava256/ava_views.py view ;;
        *) echo "passo sconosciuto: $step" >&2; exit 2 ;;
    esac
    echo "[ava256] $step: $(( $(date +%s) - t0 )) s"
done
echo "[ava256] OK"
