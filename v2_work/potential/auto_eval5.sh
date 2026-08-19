#!/usr/bin/env bash
# Valuta ogni braccio appena e' finito, con i flag giusti per quel braccio.
#
# Serve perche' i flag NON sono deducibili dal checkpoint: un modello allenato col frame rms
# caricato e valutato col frame corrente non solleva nessun errore, non cambia nessuna forma di
# tensore, e produce numeri plausibili e sbagliati. La tabella qui sotto e' l'unica fonte di
# verita' su quale flag va con quale braccio, ed e' per questo che i bracci si aggiungono QUI e
# non a mano sulla riga di comando.
#
# I run dir sono <tag>_<JOBID> con JOBID numerico. Il glob deve essere ${tag}_[0-9]* e non
# ${tag}_*, altrimenti pot_rms cattura anche pot_rms_area, e pot_plain cattura pot_plain_s2 e
# pot_plain_s3: il controllo "esiste epoch060" diventerebbe vero appena una qualsiasi delle
# omonime finisce, e un braccio verrebbe valutato a meta' training col checkpoint di un altro.
# Nessun errore sollevato, numeri plausibili.
#
# Due regole non negoziabili, entrambe nate da errori:
#   - si valuta solo se esiste epoch060.pth. Una volta un risultato fu scritto da un modello
#     all'epoca 5 di 60, con numeri credibili che sarebbero finiti nel paper.
#   - non si uccide niente, mai.
set -u
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
cd "$ROOT"
DIST=face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz
AREA=v2_work/potential/bfm_areanorm
PLAIN=datasets/REMESH/npz_data_topo_500_withops

# tag|data dir|flag extra
ARMS="
pot_rmsops|v2_work/potential/bfm_rmsnorm|--frame rms
pot_rms|$PLAIN|--frame rms
pot_rms_area|$AREA|--frame rms
pot_plain_s2|$PLAIN|
pot_area_s2|$AREA|
pot_plain_s3|$PLAIN|
pot_area_s3|$AREA|
pot_rms_s2|$PLAIN|--frame rms
pot_rms_s3|$PLAIN|--frame rms
abl_intrinsic|$PLAIN|
"

evaluate () {
  local tag=$1 data=$2 extra=$3
  local ck
  ck=$(ls v2_work/runs/${tag}_[0-9]*/*/checkpoints/best_by_xtopo_mesh_clean.pth 2>/dev/null | head -1)
  [ -n "$ck" ] || { echo "[$(date +%T)] $tag: manca il checkpoint di selezione"; return 1; }
  cat > "v2_work/potential/_eval_$tag.sh" <<NODE
#!/bin/bash
set -u
cd $ROOT
export WBES_DIFFUSION_NET_SRC=\$PWD/diffusion-net/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec .conda_env/bin/python v2_work/potential/eval_by_topology.py \\
  --checkpoint $ck --data-dir $data --dist-npz $DIST \\
  --tag $tag --use-eval-split --n-subjects 100 $extra
NODE
  chmod +x "v2_work/potential/_eval_$tag.sh"
  echo "[$(date +%T)] valuto $tag (dati $data, flag '${extra:-nessuno}')"
  export ESUB_BYPASS=1 ESUB_QUIET=1
  bsub -I -q p1i -n 4 -R "span[hosts=1] rusage[mem=24GB]" -W 120 -J "ev_$tag" \
    "bash $ROOT/v2_work/potential/_eval_$tag.sh" > "v2_work/logs/eval_$tag.log" 2>&1
  if [ -f "v2_work/potential/results/$tag.json" ]; then
    echo "[$(date +%T)] $tag FATTO"
    .conda_env/bin/python -c "
import json;d=json.load(open('v2_work/potential/results/$tag.json'))
g=d['groups'];print('   frame=%s  crop %.4f  noisy %.4f  resample %.4f  all %.4f' % (
  d.get('frame','?'), g['crop']['spearman'], g['noisy']['spearman'],
  g['resample']['spearman'], g['all']['spearman']))"
  else
    echo "[$(date +%T)] $tag: valutazione FALLITA, vedi v2_work/logs/eval_$tag.log"
  fi
}

while true; do
  pending=0
  while IFS='|' read -r tag data extra; do
    [ -z "$tag" ] && continue
    [ -f "v2_work/potential/results/$tag.json" ] && continue
    pending=$((pending + 1))
    # la condizione e' il checkpoint dell'ultima epoca, non l'assenza del job dalla coda
    if ls v2_work/runs/${tag}_[0-9]*/*/checkpoints/epoch060.pth >/dev/null 2>&1; then
      evaluate "$tag" "$data" "$extra"
    fi
  done <<< "$ARMS"
  [ "$pending" -eq 0 ] && { echo "[$(date +%T)] tutti i bracci valutati"; break; }
  sleep 300
done
