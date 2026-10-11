#!/usr/bin/env bash
#SBATCH --job-name=wbes-abl2x2
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=prioritized
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:2
#SBATCH --cpus-per-gpu=20
#SBATCH --mem-per-gpu=80G
#SBATCH --time=10:00:00
#SBATCH --requeue
#
# Ablazione 2x2 dello stream (aau/runs/evidence/stream/ablation_2x2/PROTOCOL.md, scritto prima dei numeri): FLAME 2023
# Open come quarto generatore x parzialita' variabile, ricetta c3m (testa factorized, massive_node.sh), 2 L40S per cella,
# QoS normal. Una cella per job; A e D su un nodo, B e C sull'altro (il nodo si cancella negli effetti principali):
#   CELL=A sbatch --nodelist=a768-l40s-05 v3_work/stream/slurm/ablation_2x2.sbatch
# Celle (tutto il resto identico: seme, passi, lr costante, batch per rank, hardware):
#   A  validated (bfm2019, ict, gnm), parzialita' spenta: il controllo
#   B  validated + flame2023, parzialita' spenta
#   C  validated, parzialita' p = 0.5 per vista (partial_aug.py, area di default 0.03-0.40)
#   D  validated + flame2023, parzialita' p = 0.5
# STREAM_RECIPE=c3m esplicito (con fonti diverse da "validated" il default di massive_node.sh sarebbe max); quota
# d'espressioni del C3M per bfm2019, ict, gnm e, con flame2023, 0.2142 = media delle quote di ICT e GNM.
# T = 60.000 passi da 600 (100 epoche): checkpoint ed eval online ogni 10 epoche (10%), lettura su epoch100_ema.pth.
# Uscite in aau/runs/evidence/stream/ablation_2x2/<CELL> (STREAM_OUT: run dir in runs/, codice congelato in code/).
# Continuazione dopo un TIMEOUT: stessa riga di lancio (stesso STREAM_OUT: massive.sbatch riprende da last.pth con il
# codice congelato della prima partenza).
set -euo pipefail
case "${CELL:-}" in
  A) SRC=validated;           PART="" ;;
  B) SRC=validated,flame2023; PART="" ;;
  C) SRC=validated;           PART=0.5 ;;
  D) SRC=validated,flame2023; PART=0.5 ;;
  *) echo "[abl2x2] ERRORE: CELL=A|B|C|D" >&2; exit 2 ;;
esac
EF="bfm2019=0,ict=0.2315,gnm=0.1968"
[[ "$SRC" == *flame2023* ]] && EF+=",flame2023=0.2142"
export STREAM_SOURCES="$SRC" STREAM_RECIPE=c3m STREAM_EXPR_FRAC="$EF" STREAM_PARTIAL="$PART"
export STREAM_ARM=factorized STREAM_STEPS=60000 STREAM_SPE=600 STREAM_SEED=1234
export STREAM_OUT="/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/stream/ablation_2x2/$CELL"
mkdir -p "$STREAM_OUT"
echo "[abl2x2] $(date +%F_%T) cella $CELL job=${SLURM_JOB_ID:-} riavvio=${SLURM_RESTART_COUNT:-0}" \
     "nodo=$(hostname -s) fonti=$SRC quota=$EF parzialita'=${PART:-spenta} T=$STREAM_STEPS S=$STREAM_SPE" \
  | tee -a "$STREAM_OUT/ablation.log"
exec bash v3_work/stream/slurm/massive.sbatch
