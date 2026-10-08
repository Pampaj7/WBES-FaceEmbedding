#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-suite
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --gres-flags=disable-binding
#SBATCH --time=02:00:00
#
# Tutti i test del trainer v3 in UN job (il tetto di 12 job per utente e' condiviso fra gli agenti), in
# sequenza, ognuno col suo log in aau/runs/evidence/trainer_v3/: equivalenza livello 1, DDP su 2 GPU,
# throughput del forward, equivalenza livello 2. Uno che fallisce non ferma i successivi.
#   WBES_V3_TESTS="l1 ddp bench l2"   sottoinsieme e ordine
set -uo pipefail
cd /home/create.aau.dk/ga41wf/WBES-FaceEmbedding
S=v3_work/trainer/slurm
OUT=aau/runs/evidence/trainer_v3${WBES_V3_SUITE_TAG:+/$WBES_V3_SUITE_TAG}
mkdir -p "$OUT"
for t in ${WBES_V3_TESTS:-l1 ddp bench l2}; do
  echo "[suite] $(date +%T) $t"
  case "$t" in
    l1) bash "$S/test_equivalence_l1.sbatch" > "$OUT/suite_l1.log" 2>&1 ;;
    ddp) bash "$S/test_ddp.sbatch" > "$OUT/suite_ddp.log" 2>&1 ;;
    bench) bash "$S/bench_forward.sbatch" > "$OUT/suite_bench.log" 2>&1 ;;
    l2) bash "$S/test_equivalence_l2.sbatch" > "$OUT/suite_l2.log" 2>&1 ;;
    ac) bash "$S/anti_collapse.sbatch" > "$OUT/suite_ac.log" 2>&1 ;;
    micro) source aau/env.sh && "$AAU_DIR/run.sh" v3_work/trainer/tests/micro_serve.py \
             --out "$OUT/micro_serve_${SLURM_JOB_ID:-manual}.json" > "$OUT/suite_micro.log" 2>&1 ;;
  esac
  echo "[suite] $(date +%T) $t rc=$?"
done
