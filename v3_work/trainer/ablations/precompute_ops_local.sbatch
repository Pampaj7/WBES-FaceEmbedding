#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-ablops
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --nodelist=nv-ai-04
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#
# NON lanciato dal coder. Operatori di TUTTE le mesh dagli shard del sottoinsieme delle ablazioni (4.064 ICT_SCALE +
# 810 GNM, ~63.000 mesh, ~380 GB compressi), UNA volta, sul disco locale di nv-ai-04 (/tmp e' il disco md0 da
# 1.8 TB, non RAM). Poi i bracci su A100 li leggono come vista (WBES_V3_LOCAL_OPS, train_ablation_v3.sbatch):
# niente pre-pass per braccio. Misurato nello smoke 1061910: un blocco da 7.829 mesh = 46 min con 22 processi
# su nv-ai-04 (7.7 CPU-s per mesh), troppo per 8 blocchi in 5 h. Rilanciabile: i file gia' scritti si saltano.
# Da cancellare a fine ablazioni: rm -rf /tmp/wbes_v3_ablops (sul nodo).
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV=
D="$AAU_RUNS/evidence/trainer_v3/ablations/data"
OUTD="${WBES_V3_LOCAL_OPS:-/tmp/wbes_v3_ablops}"
mkdir -p "$OUTD"
python3 -c "import json; s=json.load(open('$D/split.json')); print('\n'.join(x for x in s['train'] if int(x[2:]) >= 20000))" \
  > "$OUTD.subjects.txt" 2> /dev/null || true
[[ -s "$OUTD.subjects.txt" ]] || "$AAU_DIR/run.sh" -c "import json; s=json.load(open('$D/split.json')); open('$OUTD.subjects.txt','w').write('\n'.join(x for x in s['train'] if int(x[2:]) >= 20000)+'\n')"
TARS=$("$AAU_DIR/run.sh" -c "import json; s=json.load(open('$D/spec.json')); print(' '.join(s['tars']))")
IDX=$("$AAU_DIR/run.sh" -c "import json; print(json.load(open('$D/spec.json'))['tar_index'])")
echo "[ablops] $(wc -l < "$OUTD.subjects.txt") soggetti dagli shard -> $OUTD ($(hostname))"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$AAU_DIR/run.sh" v3_work/trainer/prepass_v3.py \
  --compress --out-dir "$OUTD" --subjects "$OUTD.subjects.txt" --convention areanorm --n-proc 120 \
  --tars $TARS --tar-index "$IDX"
echo "[ablops] $(ls "$OUTD" | wc -l) file, $(du -sh "$OUTD" | cut -f1)"
