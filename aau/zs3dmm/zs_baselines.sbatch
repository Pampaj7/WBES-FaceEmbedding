#!/usr/bin/env bash
#SBATCH --job-name=wbes-zs-bl
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Zero-shot su un 3DMM mai visto (WBES_ZS_DOMAIN=hifi|fv), baseline geometriche sugli stessi
# 100 soggetti dello zero-shot (set <dominio>_heldout di aau/zs3dmm/zs_bl.py, stesso
# rebuild_subject_split col seed 1234). Sono le righe che su ICT stanno in
# aau/runs/outlineB/ict_heldout:
#   align  aau/outlineB/alignment_matrix.py (pipeline faceBench importata): chamfer,
#          rigid_icp_chamfer, nicp_p2p, nicp_p2tri; original->original + 30 coppie cross
#   rank   aau/baselines/rank_from_matrix.py: Spearman/Pearson vs D_GT del dominio, CI bootstrap
#          per soggetto, sui setting della Tabella 2 estesa piu' crop/all_cross
# La Chamfer dell'eval del repo (chamfer_sq) esce gia' dagli script di zs_zeroshot.sbatch.
# La GT nei coefficienti la calcola zs_summarize.py dalle stesse matrici.
# Legge solo la geometria della vista (V/F): niente operatori.
#
#   WBES_ZS_DOMAIN=hifi aau/submit.sh zs3dmm/zs_baselines.sbatch
#   WBES_ZS_DOMAIN=hifi WBES_ZS_BL_SHARD=1/2 WBES_ZS_BL_STEPS=align aau/submit.sh zs3dmm/zs_baselines.sbatch
#   WBES_ZS_DOMAIN=hifi WBES_ZS_BL_STEPS=rank aau/submit.sh zs3dmm/zs_baselines.sbatch --cpus-per-task=4 --time=00:30:00
#
# Partizione prioritized senza GPU e non cpu: la cpu ha UN nodo da 32 core, condiviso col
# lavoro FLAME; i nodi GPU hanno decine di core liberi.
#
# WBES_ZS_EQSUPPORT=1: protocollo equalize-support. Mesh dalla vista equalizzata
# ($ZS_EQ_DATA_DIR, eqsupport_view.py), solo le 10 coppie con crop (--settings
# crop_cross_topology: le altre non cambiano nel protocollo), out in baselines_eqsupport/.
# Dominio bfm: set "heldout" di common (i 100 standard), vedi zs_bl.py.
#
#   WBES_ZS_DOMAIN=hifi WBES_ZS_EQSUPPORT=1 WBES_ZS_BL_STEPS=align aau/submit.sh zs3dmm/zs_baselines.sbatch
#
# NON incluse: M3DFB RLR (vuole 51 landmark iBUG sulla topologia del modello), i controlli di
# frame di outlineB e le metriche percettive (su ICT non sono state calcolate).
#
# TEMPI: su ICT 2.5-2.9 s per coppia a un core, quasi tutto NICP (job 1054623): 31 coppie di
# topologie x 4950 = ~115 ore-core, ~2 h a 64 worker. NICP lavora su 4096 punti campionati,
# quindi la mesh FaceVerse piu' grande non lo allunga di molto. Ripartibile: salta le coppie
# di topologie gia' su disco.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
cd "$WBES_ROOT"
zs_require_view

# Assegnazioni SENZA default (vedi hifi_env.sh): common.py legge queste due all'import.
export WBES_REMESH_NOOPS_DIR="$ZS_DATA_DIR"
export WBES_DIST_NPZ="$ZS_DIST_NPZ"
export WBES_EVAL_SEED="${WBES_EVAL_SEED:-1234}"

ZS_FP="$(zs_data_fp)"
OUT="$ZS_RUNS/data_$ZS_FP/baselines"
EQ_ARGS=()
if [[ "${WBES_ZS_EQSUPPORT:-0}" == 1 ]]; then
    zs_require_eq_view
    export WBES_REMESH_NOOPS_DIR="$ZS_EQ_DATA_DIR"
    OUT="$ZS_RUNS/data_$ZS_FP/baselines_eqsupport"
    EQ_ARGS=(--settings crop_cross_topology)
    mkdir -p "$OUT"
    if [[ -f "$OUT/eqsupport_manifest.json" ]] && ! cmp -s "$ZS_EQ_DIR/manifest.json" "$OUT/eqsupport_manifest.json"; then
        echo "ERRORE: $OUT viene da un'altra vista equalizzata (eqsupport_manifest.json diverso)" >&2
        exit 1
    fi
    cp "$ZS_EQ_DIR/manifest.json" "$OUT/eqsupport_manifest.json"
fi
STEPS="${WBES_ZS_BL_STEPS:-align rank}"
SHARD="${WBES_ZS_BL_SHARD:-0/1}"
echo "[zs-bl] dominio=$WBES_ZS_DOMAIN host=$(hostname) job=${SLURM_JOB_ID:-none} passi=$STEPS shard=$SHARD out=$OUT"
echo "[zs-bl] mesh=$WBES_REMESH_NOOPS_DIR gt=$WBES_DIST_NPZ seed=$WBES_EVAL_SEED"

for step in $STEPS; do
    case "$step" in
        align)
            "$AAU_DIR/outlineB/run_o3d.sh" aau/zs3dmm/zs_bl.py "$WBES_ZS_DOMAIN" "$ZS_ID_OFFSET" align \
                --out-root "$OUT" --shard "$SHARD" \
                --workers "${SLURM_CPUS_PER_TASK:-8}" "${EQ_ARGS[@]}" ${WBES_ZS_BL_ARGS:-}
            ;;
        rank)
            # Il .tex e l'avviso finale del log di rank_from_matrix chiamano "gate di
            # riproduzione con soggetti di training" ogni set diverso da heldout: qui e' falso,
            # nessun modello ha visto il dominio. Idem per il confronto col paper.
            "$AAU_DIR/run.sh" aau/zs3dmm/zs_bl.py "$WBES_ZS_DOMAIN" "$ZS_ID_OFFSET" rank --out-root "$OUT" \
                --settings original_to_original,nocrop_cross_topology,tessellation_cross_topology,perturbation_cross_topology,crop_cross_topology,all_cross_topology \
                ${WBES_ZS_BL_RANK_ARGS:-}
            ;;
        *) echo "ERRORE: passo sconosciuto: $step (align|rank)" >&2; exit 2 ;;
    esac
done
echo "[zs-bl] OK"
