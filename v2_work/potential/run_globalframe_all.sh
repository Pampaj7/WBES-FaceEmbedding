#!/usr/bin/env bash
# I tre seed del braccio global frame, in sequenza.
#
# Sequenziale e non in parallelo: le due H100 del nodo sono condivise con altri utenti, e un
# nostro braccio che ne occupa due lascerebbe il nodo senza margine. A ~3h per run sono ~9h
# totali, che e' una notte -- non vale il rischio di contendere.
#
# Non si uccide niente e non si sovrascrive niente: ogni run scrive in v2_work/runs/
# pot_globalframe_s<seed>_<JOBID>, quindi un rilancio accidentale crea una cartella nuova
# invece di rovinare quella precedente.
set -u
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
export ESUB_BYPASS=1 ESUB_QUIET=1

for SEED in 1234 1235 1236; do
  echo "=== seed ${SEED} — $(date -Is) ==="
  bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=64GB]" \
       -gpu "num=1:mode=shared" -W 720 -J "gframe_s${SEED}" \
       bash "$ROOT/v2_work/potential/_node_pot_globalframe_s${SEED}.sh" \
    2>&1 | grep -v "^Epoch .*it/s" | tail -25
  echo "=== seed ${SEED} finito — $(date -Is) ==="
done
echo "=== tutti e tre i seed completati — $(date -Is) ==="
ls -d "$ROOT"/v2_work/runs/pot_globalframe_* 2>/dev/null
