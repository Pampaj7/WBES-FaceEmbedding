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
#   sbatch aau/ava256/ava_build.sbatch                          # tutto, fino al congelamento
#   AVA_STEPS="verify" sbatch aau/ava256/ava_build.sbatch       # solo il controllo delle impronte
#
#   template    ava_neutral.py template   template per la corrispondenza
#   corr        ava_corr.py               regione FLAME / unificata sulla topologia, controllo Multiface
#   patch       ava_patch.py              patch delle viste: impronta NoW, aperture palpebrali e bocca
#   neutral     ava_neutral.py neutral    neutre e neutri ripetuti (regola FaMoS), controlli dei frame
#   gt          ava_gt.py                 unita', taglia, qualita', split 56 + 200, GT FR / SR, affidabilita'
#   identities  ava_views.py identities   patch della neutra in mm, frame nativo, suddivisa 1-a-4
#   topo        make_zs_topologies.py     le 6 topologie (codice di HIFI3D / FaceVerse / FaceScape dev)
#   crop        zs_check_crop.py          il crop differisce dalla original per ogni identita'
#   view        ava_views.py view         eval_view (200 valutati, GT) e calib_view (56 di calibrazione)
#   scale       ava_scale_table.py        tabella di scala del modello (definizione di build_scale_table.py)
#   qc          ava_view_qc.py            remesh, lati, anelli contro i domini dev
#   freeze      ava_freeze.py freeze      impronte in git, dati in sola lettura
#   verify      ava_freeze.py verify      controllo delle impronte (lo chiama anche lo strumento di valutazione)
#
# GUARDIA: con datasets/AVA256/FROZEN.json presente i passi che scrivono dati si rifiutano (up60k non e' riproducibile
# bit per bit: rigenerare cambierebbe le viste congelate). Si puo' solo verificare.
set -euo pipefail

if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${GPU_DEVICE_ORDINAL:-}" ]]; then
    export AAU_NV=""
fi
STEPS="${AVA_STEPS:-template corr patch neutral gt identities topo crop view scale qc freeze}"
UGT="v3_work/unified_gt/run.sh"
AAU="aau/run.sh"
N_CORES="${SLURM_CPUS_PER_TASK:-8}"
FROZEN="datasets/AVA256/FROZEN.json"
export UGT_THREADS="${UGT_THREADS:-8}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[ava256] host=$(hostname) job=${SLURM_JOB_ID:-none} steps: $STEPS"
if [[ -e "$FROZEN" ]]; then
    for step in $STEPS; do
        if [[ "$step" != "verify" && "$step" != "qc" ]]; then
            echo "ERRORE: $FROZEN presente, GT e viste congelate: il passo '$step' riscriverebbe i dati." >&2
            echo "  Rigenerare richiede una decisione del PI (togliere a mano FROZEN.json e la sola lettura)." >&2
            exit 3
        fi
    done
fi
for step in $STEPS; do
    t0=$(date +%s)
    case "$step" in
        template)   UGT_THREADS=1 $UGT aau/ava256/ava_neutral.py template --workers "$N_CORES" ;;
        corr)       $UGT aau/ava256/ava_corr.py ;;
        patch)      $UGT aau/ava256/ava_patch.py ;;
        neutral)    UGT_THREADS=1 $UGT aau/ava256/ava_neutral.py neutral --workers "$N_CORES" ;;
        gt)         $UGT aau/ava256/ava_gt.py ;;
        identities) $UGT aau/ava256/ava_views.py identities ;;
        topo)       $AAU aau/zs3dmm/make_zs_topologies.py --prefix ava --in-dir datasets/AVA256/identities \
                        --out-dir datasets/AVA256/topo --n-cores "$N_CORES" --overwrite
                    $AAU aau/zs3dmm/make_zs_topologies.py --demo --prefix ava --in-dir datasets/AVA256/identities \
                        --out-dir datasets/AVA256/topo ;;
        crop)       $AAU aau/zs3dmm/zs_check_crop.py --topo-dir datasets/AVA256/topo --prefix ava --write ;;
        view)       $UGT aau/ava256/ava_views.py view ;;
        scale)      $AAU aau/ava256/ava_scale_table.py ;;
        qc)         $UGT aau/ava256/ava_view_qc.py ;;
        freeze)     $UGT aau/ava256/ava_freeze.py freeze ;;
        verify)     $UGT aau/ava256/ava_freeze.py verify ;;
        *) echo "passo sconosciuto: $step" >&2; exit 2 ;;
    esac
    echo "[ava256] $step: $(( $(date +%s) - t0 )) s"
done
echo "[ava256] OK"
