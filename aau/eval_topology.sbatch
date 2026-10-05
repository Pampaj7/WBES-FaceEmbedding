#!/usr/bin/env bash
#SBATCH --job-name=wbes-eval-topo
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili, e un
# --output relativo viene risolto nella submit dir (se non esiste, Slurm scarta il job in
# un secondo senza scrivere niente). Se sposti il repo, aggiorna queste tre righe oppure
# sottometti con aau/submit.sh, che le ricalcola e le passa da riga di comando.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#
# Stage 2/3 della robustness eval: breakdown per coppia di topologie.
# Misurato 1h03m nel log della run originale, --time=3h per stare larghi.
#   aau/submit.sh eval_topology.sbatch

source "${WBES_ROOT:-$PWD}/aau/eval_common.sh"

aau_eval_begin topology

echo "[$(timestamp)] Starting topology breakdown"
"$AAU_DIR/run.sh" "$PERT/compare_model_vs_chamfer_topology_breakdown.py" \
  "${common_args[@]}" \
  --out_dir "$STAGE_OUT" \
  --topology_pair_mode cross_only \
  --ordered_topology_pairs \
  2>&1 | tee "$STAGE_OUT/run.log"

aau_eval_done topology
