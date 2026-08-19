#!/usr/bin/env bash
# Il braccio che testa l'ipotesi della coerenza di unita': operatori calcolati sulla STESSA
# normalizzazione che la rete vede sull'xyz (centroide pesato per massa, raggio rms unitario),
# quindi anche --frame rms in training. Bar: pot_rms, crop 0.7637.
#
# Attende due condizioni, entrambe sul CONTEGGIO e mai sulla presenza di un job con un certo
# nome: 3000 operatori, e meno di 3 nostri training vivi. Un training su operatori incompleti
# sarebbe silenziosamente non confrontabile, quindi uno stallo ABORTISCE invece di procedere.
set -u
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
cd "$ROOT"
TAG=pot_rmsops
OPS=v2_work/potential/bfm_rmsnorm

stall=0; last=-1
while true; do
  n=$(ls "$OPS" 2>/dev/null | wc -l)
  [ "$n" -ge 3000 ] && break
  if [ "$n" -eq "$last" ]; then stall=$((stall+1)); else stall=0; fi
  last=$n
  if [ "$stall" -ge 20 ]; then
    echo "[$(date +%T)] ABORT: operatori fermi a $n/3000 da 40 minuti. NON allena su dati parziali."
    exit 1
  fi
  sleep 120
done
echo "[$(date +%T)] operatori rms completi: $n/3000"

while [ "$(bjobs -w 2>/dev/null | awk '$3=="RUN" && $7 ~ /^(pot_|pt_)/' | wc -l)" -ge 3 ]; do sleep 120; done

if bjobs -w 2>/dev/null | grep -qE "[[:space:]]$TAG$"; then
  echo "[$(date +%T)] $TAG gia' in coda, esco"; exit 0
fi

sed -e "s|pot_plain|$TAG|g" \
    -e "s|--cache-residency ram --cache-workers 16|--cache-residency ram --cache-workers 8 --frame rms|" \
    -e "s|--data_dir datasets/REMESH/npz_data_topo_500_withops|--data_dir $OPS|" \
    "$ROOT/v2_work/potential/_node_pot_plain.sh" > "$ROOT/v2_work/potential/_node_$TAG.sh"
chmod +x "$ROOT/v2_work/potential/_node_$TAG.sh"
grep -q -- "--frame rms" "$ROOT/v2_work/potential/_node_$TAG.sh" || { echo "ABORT: --frame rms non applicato"; exit 1; }
grep -q -- "--data_dir $OPS" "$ROOT/v2_work/potential/_node_$TAG.sh" || { echo "ABORT: data dir non applicata"; exit 1; }

export ESUB_BYPASS=1 ESUB_QUIET=1
setsid nohup bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=24GB]" \
  -gpu "num=1:mode=shared" -W 720 -J "$TAG" \
  "bash $ROOT/v2_work/potential/_node_$TAG.sh" \
  > "$ROOT/v2_work/logs/runs/$TAG.log" 2>&1 < /dev/null &
echo "[$(date +%T)] $TAG sottomesso"
