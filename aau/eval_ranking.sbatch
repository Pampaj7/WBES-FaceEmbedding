#!/usr/bin/env bash
#SBATCH --job-name=wbes-eval-rank
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
#SBATCH --time=10:00:00
#
# Stage 1/3 della robustness eval: ranking del modello vs Chamfer su 5 scenari.
# Misurato 5h13m nel log della run originale, --time=10h per stare larghi.
#   aau/submit.sh eval_ranking.sbatch
#
# WBES_EVAL_SCENARIOS spezza la run in piu' job quando il walltime non basta.
# Serve perche' compare_model_vs_chamfer_rankings.py accumula le righe in memoria e scrive
# ranking_summary.json solo DOPO l'ultimo scenario (riga 790): un job ucciso a meta' non
# lascia NIENTE di parziale, nemmeno gli scenari gia' finiti. Misurato 2.3h per scenario su
# L40S (148500 mesh pair a ~18.5 pair/s), cioe' ~11.8h per i cinque: con --time=10h muore
# dentro 'mixed' e butta via tutto.
# Un job per gruppo di scenari, ognuno con la sua WBES_EVAL_STAGE (= sottocartella sotto
# OUT_ROOT, altrimenti i job si sovrascriverebbero run.log e .done a vicenda), poi
# aau/merge_ranking_scenarios.py rimette insieme la tabella:
#   WBES_EVAL_SCENARIOS=rotation,mixed WBES_EVAL_STAGE=ranking_rotation_mixed \
#     aau/submit.sh eval_ranking.sbatch --time=07:00:00
WBES_EVAL_SCENARIOS="${WBES_EVAL_SCENARIOS:-clean,jitter,translation,rotation,mixed}"
WBES_EVAL_STAGE="${WBES_EVAL_STAGE:-ranking}"

source "${WBES_ROOT:-$PWD}/aau/eval_common.sh"

aau_eval_begin "$WBES_EVAL_STAGE"

echo "[$(timestamp)] Starting ranking benchmark"
"$AAU_DIR/run.sh" "$PERT/compare_model_vs_chamfer_rankings.py" \
  "${common_args[@]}" \
  --out_dir "$STAGE_OUT" \
  --aggregation_level subject_pair_mean \
  --scenarios "$WBES_EVAL_SCENARIOS" \
  --jitter_sigma 0.05 \
  --translation_sigma 0.05 \
  --rotation_sigma 0.05 \
  --mixed_jitter_sigma 0.05 \
  --mixed_translation_sigma 0.05 \
  --mixed_rotation_sigma 0.05 \
  2>&1 | tee "$STAGE_OUT/run.log"

aau_eval_done "$WBES_EVAL_STAGE"
