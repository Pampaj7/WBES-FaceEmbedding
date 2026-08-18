#!/usr/bin/env bash
# Le repliche di seed per pot_plain e pot_area, in coda dietro al 2x2.
#
# Perche' servono: il guadagno misurato e' +0.022 su crop e +0.0085 su all. Il criterio fissato
# la mattina del 18, prima di vedere quei numeri, era che un seed basta per una differenza come
# 0.7072 contro 0.3155 e non per una piccola. Questa e' piccola. Senza repliche non si puo' dire
# se il guadagno e' reale, ne' se il calo di 1.4 punti su resample e' un effetto o rumore.
#
# Perche' una catena e non quattro bsub subito: p1i ha due H100 condivise con ~19 job altrui, e
# tre training nostri insieme li rallentano tutti senza anticipare nulla. La catena tiene al
# massimo due nostri training vivi e lancia il successivo quando uno finisce.
#
# La condizione di avanzamento e' il CONTEGGIO dei nostri training in RUN, mai la presenza di un
# job con un certo nome: e' l'errore che una volta ha fatto partire un training su 2792 mesh
# su 3000. E questo script non uccide nulla, mai: il finisher che uccideva i bracci che stava
# aspettando lo ha fatto tre volte prima che il difetto fosse capito.
set -u
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
cd "$ROOT"
MAXRUN=2
STALL_LIMIT=180        # 6 ore senza uno slot libero = qualcosa non va, esci invece di insistere

ours () { bjobs -w 2>/dev/null | awk '$3=="RUN" && $7 ~ /^(pot_|pt_)/' | wc -l; }

launch () {   # $1 tag, $2 seed, $3 data dir, $4 flag extra
  if bjobs -w 2>/dev/null | grep -qE "[[:space:]]$1$"; then
    echo "[$(date +%T)] $1 gia' in coda, salto"
    return
  fi
  sed -e "s|pot_plain|$1|g" \
      -e "s|--cache-residency ram --cache-workers 16|--cache-residency ram --cache-workers 8 $4|" \
      -e "s|--data_dir datasets/REMESH/npz_data_topo_500_withops|--data_dir $3|" \
      -e "s|--seed 1234|--seed $2|" \
      "$ROOT/v2_work/potential/_node_pot_plain.sh" > "$ROOT/v2_work/potential/_node_$1.sh"
  chmod +x "$ROOT/v2_work/potential/_node_$1.sh"
  grep -q -- "--seed $2" "$ROOT/v2_work/potential/_node_$1.sh" || { echo "ABORT: seed non applicato a $1"; exit 1; }
  grep -q -- "--data_dir $3" "$ROOT/v2_work/potential/_node_$1.sh" || { echo "ABORT: data dir non applicata a $1"; exit 1; }
  export ESUB_BYPASS=1 ESUB_QUIET=1
  setsid nohup bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=24GB]" \
    -gpu "num=1:mode=shared" -W 720 -J "$1" \
    "bash $ROOT/v2_work/potential/_node_$1.sh" \
    > "$ROOT/v2_work/logs/runs/$1.log" 2>&1 < /dev/null &
  echo "[$(date +%T)] $1 sottomesso (seed $2, dati $3)"
  sleep 60
}

AREA=v2_work/potential/bfm_areanorm
PLAIN=datasets/REMESH/npz_data_topo_500_withops

# ordine: alterna i due bracci, cosi' se la catena si interrompe a meta' restano comunque
# coppie confrontabili invece di tre repliche di un braccio solo
run_next () {
  case "$1" in
    0) launch pot_plain_s2 1235 "$PLAIN" "" ;;
    1) launch pot_area_s2  1235 "$AREA"  "" ;;
    2) launch pot_plain_s3 1236 "$PLAIN" "" ;;
    3) launch pot_area_s3  1236 "$AREA"  "" ;;
  esac
}

i=0; stall=0
while [ "$i" -lt 4 ]; do
  n=$(ours)
  if [ "$n" -lt "$MAXRUN" ]; then
    run_next "$i"; i=$((i + 1)); stall=0
  else
    stall=$((stall + 1))
    [ "$stall" -ge "$STALL_LIMIT" ] && { echo "[$(date +%T)] ABORT: nessuno slot da 6 ore, mi fermo a $i/4"; exit 1; }
    sleep 120
  fi
done
echo "[$(date +%T)] tutte e quattro le repliche sottomesse"
