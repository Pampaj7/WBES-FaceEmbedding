#!/usr/bin/env bash
#SBATCH --job-name=wbes-flame-bl
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=30
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#
# WS-FLAME, baseline geometriche sugli stessi 100 soggetti dello zero-shot (subject set
# flame_heldout di aau/baselines/common.py, stesso rebuild_subject_split col seed 1234).
# Sono le righe che su ICT stanno in aau/runs/outlineB/ict_heldout:
#   align  aau/outlineB/alignment_matrix.py (pipeline faceBench importata): chamfer,
#          rigid_icp_chamfer, nicp_p2p, nicp_p2tri; original->original + 30 coppie cross
#   rank   aau/baselines/rank_from_matrix.py: Spearman/Pearson vs D_GT FLAME, CI bootstrap
#          per soggetto, sui setting della Tabella 2 estesa piu' crop/all_cross
# La Chamfer dell'eval del repo (chamfer_sq) esce gia' dagli script di flame_zeroshot.sbatch.
#
#   aau/submit.sh flame/flame_baselines.sbatch
#   WBES_FLAME_BL_STEPS=rank aau/submit.sh flame/flame_baselines.sbatch --time=00:30:00
#   WBES_FLAME_BL_SHARD=1/4  aau/submit.sh flame/flame_baselines.sbatch   # solo align, a pezzi
#
# NON incluse: M3DFB RLR (m3dfb_matrix.py vuole 51 landmark iBUG sulla topologia del modello,
# e per FLAME stanno nell'embedding dei landmark FLAME, un altro file della licenza), i
# controlli di frame di outlineB e le metriche percettive (su ICT non sono state calcolate).
#
# TEMPI: su ICT 2.5-2.9 s per coppia a un core, quasi tutto NICP (job 1054623): 31 coppie
# di topologie x 4950 = ~115 ore-core, ~4 h a 30 worker. NICP lavora su 4096 punti
# campionati, quindi la patch FLAME piu' piccola non lo accorcia di molto. Ripartibile:
# salta le coppie di topologie gia' su disco.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/flame/flame_env.sh"
cd "$WBES_ROOT"
flame_require_view

# Assegnazioni SENZA default (vedi flame_env.sh): common.py legge queste due all'import.
# Vista SENZA operatori: le baseline leggono solo vertici e facce, e la normalizzazione maxabs
# di common.py rende indifferente che la mesh sia quella grezza o quella ad area unitaria.
export WBES_REMESH_NOOPS_DIR="$WBES_FLAME_MESH_DIR"
export WBES_DIST_NPZ="$WBES_FLAME_DIST_NPZ"
export WBES_EVAL_SEED="${WBES_EVAL_SEED:-1234}"

# Sotto l'impronta dei dati: alignment_matrix.py e rank_from_matrix.py riusano le matrici gia'
# su disco per nome, e dati nuovi devono trovare una dir vuota.
FP="$(flame_fingerprint)"
OUT="$WBES_FLAME_RUNS/$FP/baselines"
mkdir -p "$WBES_FLAME_RUNS/$FP"
cp "$WBES_FLAME_TOPO_DIR/$FLAME_STAMP_NAME" "$WBES_FLAME_RUNS/$FP/identities_manifest.json"
STEPS="${WBES_FLAME_BL_STEPS:-align rank}"
SHARD="${WBES_FLAME_BL_SHARD:-0/1}"
echo "[flame-bl] host=$(hostname) job=${SLURM_JOB_ID:-none} passi=$STEPS shard=$SHARD out=$OUT"
echo "[flame-bl] mesh=$WBES_REMESH_NOOPS_DIR gt=$WBES_DIST_NPZ seed=$WBES_EVAL_SEED"

for step in $STEPS; do
    case "$step" in
        align)
            "$AAU_DIR/outlineB/run_o3d.sh" aau/outlineB/alignment_matrix.py \
                --subject-set flame_heldout --out-root "$OUT" --shard "$SHARD" \
                --workers "${SLURM_CPUS_PER_TASK:-8}" ${WBES_FLAME_BL_ARGS:-}
            ;;
        rank)
            # Si leggono csv/json. Il .tex e l'avviso finale del log di rank_from_matrix
            # chiamano "gate di riproduzione con soggetti di training" ogni set diverso da
            # heldout (succede anche con ict_heldout): per FLAME e' falso, nessun modello
            # ne ha visto un soggetto. Idem per il confronto col paper in coda al log.
            "$AAU_DIR/run.sh" aau/baselines/rank_from_matrix.py \
                --subject-set flame_heldout --out-root "$OUT" \
                --settings original_to_original,nocrop_cross_topology,tessellation_cross_topology,perturbation_cross_topology,crop_cross_topology,all_cross_topology \
                ${WBES_FLAME_BL_RANK_ARGS:-}
            ;;
        *) echo "ERRORE: passo sconosciuto: $step (align|rank)" >&2; exit 2 ;;
    esac
done
echo "[flame-bl] OK"
