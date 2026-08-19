#!/usr/bin/env bash
# Il test diretto di "il problema e' dare xyz a DiffusionNet".
#
#   abl_intrinsic   --no-use_xyz --n_hks 16 --n_wks 16   nessuna coordinata, solo descrittori
#                                                        intrinseci sullo spettro
#
# Ricetta identica a pot_plain in tutto il resto: stessi dati, stesso frame corrente, stesso
# seed 1234, 60 epoche, bs5. La differenza e' il solo canale di ingresso, quindi il confronto
# contro pot_plain (crop 0.7072, all 0.7347) isola esattamente il contributo dell'xyz.
#
# PREVISIONE REGISTRATA PRIMA DI LANCIARLO. Il braccio intrinseco sara' PIU' ROBUSTO in termini
# relativi sul crop -- non ha un frame di coordinate da rompere -- ma NETTAMENTE PEGGIORE su
# `all`, perche' l'identita' di un volto e' in buona parte estrinseca (proporzioni, profondita',
# sporgenze) e i descrittori intrinseci sono invarianti alle isometrie, cioe' per costruzione
# non distinguono volti che differiscono per come sono piegati nello spazio.
#
# Due indizi gia' misurati che sostengono questa previsione:
#   - togliendo gli operatori e lasciando solo xyz, crop crolla a 0.3155: xyz da solo non basta,
#     ma questo non dice se sia necessario;
#   - FLAME ha consistenza spettrale 3.5x migliore di BFM e ranking molto peggiore (0.478 contro
#     0.751): piu' invarianza intrinseca non ha significato metrica migliore.
#
# SE INVECE abl_intrinsic batte pot_plain su `all`, la previsione e' sbagliata, l'ipotesi di
# Leonardo e' giusta nella sua forma forte, e la direzione del lavoro cambia: non si tratterebbe
# di riparare il frame ma di togliere le coordinate. Va detto senza attenuarlo.
set -u
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
cd "$ROOT"
TAG=abl_intrinsic

while [ "$(bjobs -w 2>/dev/null | awk '$3=="RUN" && $7 ~ /^(pot_|pt_|abl_)/' | wc -l)" -ge 2 ]; do sleep 120; done
if bjobs -w 2>/dev/null | grep -qE "[[:space:]]$TAG$"; then echo "[$(date +%T)] $TAG gia' in coda"; exit 0; fi

grep -q -- "--no-use_xyz" "$ROOT/v2_work/ablation/_node_$TAG.sh" || { echo "ABORT: il braccio non e' senza xyz"; exit 1; }

export ESUB_BYPASS=1 ESUB_QUIET=1
setsid nohup bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=24GB]" \
  -gpu "num=1:mode=shared" -W 720 -J "$TAG" \
  "bash $ROOT/v2_work/ablation/_node_$TAG.sh" \
  > "$ROOT/v2_work/logs/runs/$TAG.log" 2>&1 < /dev/null &
echo "[$(date +%T)] $TAG sottomesso (nessun xyz, HKS+WKS)"
