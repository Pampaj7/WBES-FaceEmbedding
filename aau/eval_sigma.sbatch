#!/usr/bin/env bash
#SBATCH --job-name=wbes-eval-sigma
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
#SBATCH --time=36:00:00
#
# Stage 3/3 della robustness eval: sigma sweep, 4 scenari x 6 sigma.
# Misurato 21h54m nel log della run originale: e' lo stage che domina le 28h totali.
# --time=36h (MaxWall su prioritized/QoS normal e' 6 giorni).
#   aau/submit.sh eval_sigma.sbatch
#
# 36 h NON bastano: misurato 2.3h per blocco scenario/sigma (148500 mesh pair a ~18.5
# pair/s su L40S). I blocchi pagati sono 21 — 1 clean + 4 scenari x 5 sigma non nulle,
# perche' con --include_clean_once le sigma=0.00 vengono clonate dal clean e costano zero —
# cioe' ~48 h. Il job muore dentro 'mixed'.
# Qui il taglio e' possibile: --sweep_scenarios e --sigma_values accettano sottoinsiemi, e
# con --progressive_output_layout ogni coppia scenario/sigma ha gia' scritto la sua
# sottocartella, quindi un job ucciso conserva tutto quello che aveva finito. Quello che
# NON si puo' condividere e' la dir dello stage: due job nella stessa scriverebbero lo
# stesso run.log (tee lo tronca) e la stessa .done. Da qui WBES_EVAL_STAGE.
#   WBES_SIGMA_SCENARIOS=mixed WBES_EVAL_STAGE=sigma_mixed \
#     aau/submit.sh eval_sigma.sbatch --time=17:00:00
WBES_SIGMA_SCENARIOS="${WBES_SIGMA_SCENARIOS:-jitter,translation,rotation,mixed}"
WBES_SIGMA_VALUES="${WBES_SIGMA_VALUES:-0.00,0.02,0.05,0.10,0.15,0.20}"
WBES_EVAL_STAGE="${WBES_EVAL_STAGE:-sigma}"

source "${WBES_ROOT:-$PWD}/aau/eval_common.sh"

aau_eval_begin "$WBES_EVAL_STAGE"

echo "[$(timestamp)] Starting sigma sweep"
"$AAU_DIR/run.sh" "$PERT/compare_model_vs_chamfer_rankings_sigma_sweep.py" \
  "${common_args[@]}" \
  --out_dir "$STAGE_OUT" \
  --aggregation_level subject_pair_mean \
  --sweep_scenarios "$WBES_SIGMA_SCENARIOS" \
  --sigma_values "$WBES_SIGMA_VALUES" \
  --include_clean_once \
  --progressive_output_layout \
  2>&1 | tee "$STAGE_OUT/run.log"

aau_eval_done "$WBES_EVAL_STAGE"
