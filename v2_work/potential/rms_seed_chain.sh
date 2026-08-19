#!/usr/bin/env bash
# Repliche di seed per pot_rms, accodate dietro ai quattro training in corso.
#
# Perche' non c'erano gia': la catena originale replicava pot_plain e pot_area, decisa quando
# quelli erano gli unici due bracci. pot_rms e' arrivato dopo ed e' risultato il migliore, e
# quindi e' rimasto senza repliche proprio il braccio su cui poggia la conclusione.
#
# Perche' adesso e' urgente: il controllo su due seed da' crop 0.7072 e 0.7333, cioe' una
# dispersione di 0.0261 -- piu' grande del +0.0220 che avevo attribuito alla normalizzazione
# d'area. Il +0.0435 di pot_rms e' oltre il doppio di quella dispersione, ma con n=1 non e'
# ancora una misura.
set -u
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
cd "$ROOT"
MAXRUN=3

ours () { bjobs -w 2>/dev/null | awk '$3=="RUN" && $7 ~ /^(pot_|pt_)/' | wc -l; }

launch () {   # $1 tag, $2 seed
  if bjobs -w 2>/dev/null | grep -qE "[[:space:]]$1$"; then echo "[$(date +%T)] $1 gia' in coda"; return; fi
  sed -e "s|pot_plain|$1|g" \
      -e "s|--cache-residency ram --cache-workers 16|--cache-residency ram --cache-workers 8 --frame rms|" \
      -e "s|--seed 1234|--seed $2|" \
      "$ROOT/v2_work/potential/_node_pot_plain.sh" > "$ROOT/v2_work/potential/_node_$1.sh"
  chmod +x "$ROOT/v2_work/potential/_node_$1.sh"
  # Un seed non applicato darebbe repliche identiche che sembrano una conferma perfetta, e il
  # frame mancante renderebbe il braccio un altro braccio. Entrambi verificati, non dedotti.
  grep -q -- "--seed $2" "$ROOT/v2_work/potential/_node_$1.sh" || { echo "ABORT: seed $2 non applicato a $1"; exit 1; }
  grep -q -- "--frame rms" "$ROOT/v2_work/potential/_node_$1.sh" || { echo "ABORT: --frame rms non applicato a $1"; exit 1; }
  export ESUB_BYPASS=1 ESUB_QUIET=1
  setsid nohup bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=24GB]" \
    -gpu "num=1:mode=shared" -W 720 -J "$1" \
    "bash $ROOT/v2_work/potential/_node_$1.sh" \
    > "$ROOT/v2_work/logs/runs/$1.log" 2>&1 < /dev/null &
  echo "[$(date +%T)] $1 sottomesso (seed $2, frame rms)"
  sleep 60
}

i=0; stall=0
for spec in "pot_rms_s2 1235" "pot_rms_s3 1236"; do
  set -- $spec
  while [ "$(ours)" -ge "$MAXRUN" ]; do
    stall=$((stall+1))
    [ "$stall" -ge 360 ] && { echo "[$(date +%T)] ABORT: nessuno slot da 12 ore"; exit 1; }
    sleep 120
  done
  stall=0
  launch "$1" "$2"
done
echo "[$(date +%T)] repliche di pot_rms sottomesse"
