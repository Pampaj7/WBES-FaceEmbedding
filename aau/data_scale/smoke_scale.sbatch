#!/usr/bin/env bash
#SBATCH --job-name=wbes-ds-smoke
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#
# Smoke del percorso congiunto del run grande (aau/data_scale/PLAN.md), corto. WBES_DS_SMOKE:
#   S1  default del run grande: BFM e ICT-5000 dalle viste con operatori, ICT nuove LETTE DAGLI SHARD
#       (pre-pass dentro il trainer), --domain-blocked --eval_domain bfm, gruppi di espressioni al 25%,
#       GT estesa a scala invariata, guardia held-out (politica joint_compare), BFM residente in ogni
#       blocco con il 9% dei passi, 2 blocchi, 60 passi
#   S2  le opzioni di frame (spente nel run grande): BFM dalla GEOMETRIA specchiata x->-x nel pre-pass,
#       aug rotazione <= 180 gradi e riflessione p=0.5, 1 blocco, 30 passi
#   S3  (dopo il BLOCCANTE del critic, 6 ottobre) come il run grande: 8 thread di torch al training e
#       22 processi al pre-pass, lr a passi fissi, quota BFM 78/879, eval online del congiunto, 2
#       blocchi da 300 passi ciascuno, cosi' il training del blocco 0 si SOVRAPPONE al pre-pass del
#       blocco 1: misura s/passo con e senza pre-pass in corso, RAM del job e /tmp ogni 15 s (mem.log)
#   S4  (secondo giro del critic) come S3 con indice dei tar, proiezione esatta della cache, cache NON
#       pinned; il campionatore registra il cgroup scomposto (anon / shmem = /tmp / page cache) e il picco
#   S5  blocchi a GRANDEZZA REALE: i soggetti dei blocchi 0 e 1 della partizione del run grande
#       (spec_scale_default.json, 1.742 soggetti per blocco), S=293, 2.930 passi (5 epoche per blocco,
#       il pre-pass del blocco 1 si sovrappone al training del blocco 0), da lanciare con il --mem del
#       run grande: verifica il bilancio di memoria su un blocco vero
#         WBES_DS_SMOKE=S5 WBES_DS_MEM_EVERY=5 aau/submit.sh data_scale/smoke_scale.sbatch --mem=600G --time=04:00:00
#       (--mem largo apposta: si misura il picco vero attraverso il cambio di blocco, poi si dimensiona)
# Split ridotto aau/data_scale/split_smoke.json (40 BFM + 60 ICT-5000 + 500 nuove in training).
#
#   WBES_DS_SMOKE=S1 aau/submit.sh data_scale/smoke_scale.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net
source "$AAU_DIR/data_scale/recipe_v1.sh"

ARM="${WBES_DS_SMOKE:?S1|S2}"
ROOT="$AAU_RUNS/data_scale_smoke/${ARM}_${SLURM_JOB_ID:-manual}"
STAGE="/tmp/wbes-ds-smoke-${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$ROOT" "$STAGE"
D="$WBES_ROOT/datasets"
SH="$D/ICT_SCALE/shards"
EXPR_GROUPS='{"rexprA": ["rexpr1","rexpr2","rexpr3","rexpr4"], "rexprB": ["rexpr5","rexpr6","rexpr7","rexpr8"]}'
case "$ARM" in
  S1) BFM_SRC="\"views\": [\"$D/REMESH/npz_data_topo_500_withops_areanorm\", \"$D/ICT/train_ready/npz_withops\"], \"geom_dirs\": []"
      FRAME='"canon": null, "aug": null'; NB=2; T=60 ;;
  S3) BFM_SRC="\"views\": [\"$D/REMESH/npz_data_topo_500_withops_areanorm\", \"$D/ICT/train_ready/npz_withops\"], \"geom_dirs\": []"
      FRAME='"canon": null, "aug": null'; NB=2; T=600 ;;
  S4) BFM_SRC="\"views\": [\"$D/REMESH/npz_data_topo_500_withops_areanorm\", \"$D/ICT/train_ready/npz_withops\"], \"geom_dirs\": [], \"tar_index\": \"$SH/index.npz\""
      FRAME='"canon": null, "aug": null'; NB=2; T=600 ;;
  S5) BFM_SRC="\"views\": [\"$D/REMESH/npz_data_topo_500_withops_areanorm\", \"$D/ICT/train_ready/npz_withops\"], \"geom_dirs\": [], \"tar_index\": \"$SH/index.npz\""
      FRAME='"canon": null, "aug": null'; NB=2; T=2930 ;;
  S2) BFM_SRC="\"views\": [\"$D/ICT/train_ready/npz_withops\"], \"geom_dirs\": [\"$D/REMESH/npz_data_topo_500\"]"
      FRAME='"canon": {"bfm": [[-1,0,0],[0,1,0],[0,0,1]]}, "aug": {"rot_deg": 180, "reflect_p": 0.5}'; NB=1; T=30 ;;
  *) echo "WBES_DS_SMOKE=$ARM, attesi S1|S2|S3|S4|S5" >&2; exit 2 ;;
esac
TARLIST="\"$SH/shard_00000.tar\", \"$SH/shard_00001.tar\""
SPLIT="$AAU_DIR/data_scale/split_smoke.json"; S_EP=15; EV=10; LRB=300
if [[ "$ARM" == S5 ]]; then
  TARLIST=$(find "$SH" -maxdepth 1 -name 'shard_*.tar' | sort | sed 's/.*/"&"/' | paste -sd, -)
  SPLIT="$ROOT/split_fullblock.json"; S_EP=293; EV=5; LRB=1465
  AAU_NV="" "$AAU_DIR/run.sh" -c '
import json, sys
sys.path.insert(0, "v2_work/fastio")
from train_steps import partition_blocks
spec = json.load(open("aau/data_scale/spec_scale_default.json"))
split = json.load(open("aau/data_scale/split_scale.json"))
blocks = partition_blocks(sorted(split["train"]), spec)
keep = sorted(set(blocks[0]) | set(blocks[1]))
json.dump({"source": "blocchi 0 e 1 della partizione del run grande", "train": keep,
           "heldout": split["heldout"], "online_eval": split["online_eval"]}, open(sys.argv[1], "w"))
print("[smoke] S5:", len(blocks[0]), len(blocks[1]), "soggetti nei blocchi reali 0 e 1 ->", len(keep))
' "$SPLIT"
fi
cat > "$ROOT/spec.json" <<JSON
{$BFM_SRC, "tars": [$TARLIST],
 "labels": null, "label_groups": $EXPR_GROUPS, $FRAME,
 "pin_domains": ["bfm"], "domain_step_share": {"bfm": 0.0887372},
 "convention": "areanorm", "n_blocks": $NB, "block_seed": 0}
JSON
echo "[smoke] host=$(hostname) job=${SLURM_JOB_ID:-none} arm=$ARM root=$ROOT"
cat "$ROOT/spec.json"
# campionatore: memoria del cgroup del job (v1 o v2) e /tmp dello stage, ogni 15 s
cg_file() {  # cgroup del JOB (dove Slurm applica --mem), non dello step o del task
  local p d; p=$(awk -F: '$2 ~ /memory/ || $2 == "" {print $3; exit}' /proc/self/cgroup)
  for base in /sys/fs/cgroup/memory /sys/fs/cgroup; do
    d="$base$p"
    while [[ "$d" != "$base" && "$(basename "$d")" != job_* ]]; do d=$(dirname "$d"); done
    for f in "$d/memory.usage_in_bytes" "$d/memory.current"; do
      [[ -r "$f" ]] && { echo "$f"; return; }
    done
  done
}
CG=$(cg_file || true)
CGD=$(dirname "${CG:-/nonexistent}")
cg_stat() {  # anon / shmem (tmpfs) / page cache in GiB, v1 (total_*) o v2
  awk '$1=="total_rss"||$1=="anon"{a=$2} $1=="total_shmem"||$1=="shmem"{s=$2}
       $1=="total_cache"||$1=="file"{c=$2} END{printf "anon_gb=%.1f shmem_gb=%.1f cache_gb=%.1f", a/2^30, s/2^30, c/2^30}' \
      "$CGD/memory.stat" 2> /dev/null
}
( while sleep "${WBES_DS_MEM_EVERY:-15}"; do
    printf '%s mem_gb=%s tmp_gb=%s %s\n' "$(date +%T)" \
      "$( [[ -n "$CG" ]] && awk '{printf "%.1f", $1/2^30}' "$CG" || echo NA)" \
      "$(du -sk "$STAGE" 2>/dev/null | awk '{printf "%.1f", $1/2^20}')" "$(cg_stat)"
  done ) > "$ROOT/mem.log" 2>&1 &
SAMPLER=$!
T_THREADS=(); P_PROC=30; LR=(--plateau-patience 8)
if [[ "$ARM" == S3 || "$ARM" == S4 || "$ARM" == S5 ]]; then
  T_THREADS=(--train-threads 8); P_PROC=22; LR=(--lr-steps "$LRB:5e-5"); export OMP_NUM_THREADS=8
fi
t0=$SECONDS
rc=0
"$AAU_DIR/run.sh" v2_work/fastio/train_steps.py --total-steps "$T" --steps-per-epoch "$S_EP" \
    --split-json "$SPLIT" \
    --frozen-heldout "$AAU_DIR/data_scale/heldout_frozen.json" \
    --data-spec "$ROOT/spec.json" --stage-root "$STAGE/stage" --prepass-proc "$P_PROC" "${T_THREADS[@]}" \
    --domain-blocked --eval_domain bfm --gt-keep-scale "${LR[@]}" \
    --cache-residency ram --cache-workers 16 --cache-max-gb 240 --frame current \
    "${RECIPE_V1[@]}" --epochs $(( T / S_EP )) --eval_every "$EV" --save_every $(( T / S_EP )) --seed 1234 \
    --data_dir "$STAGE/stage" --dist_npz "$D/ICT_SCALE/gt_joint_bfm_ict.npz" --runs_root "$ROOT" \
    > "$ROOT/train.log" 2>&1 || rc=$?
kill "$SAMPLER" 2> /dev/null || true
for f in "$CGD/memory.max_usage_in_bytes" "$CGD/memory.peak"; do
  [[ -r "$f" ]] && awk -v f="$f" '{printf "[smoke] picco del cgroup (%s): %.1f GiB\n", f, $1/2^30}' "$f" | tee -a "$ROOT/mem.log"
done
tr '\r' '\n' < "$ROOT/train.log" | grep -E "^\[steps\]|^\[v2\]|^Epoch [0-9]{3}|cache\] ready|Error|Traceback" | tail -60
sort -t= -k2 -n "$ROOT/mem.log" | tail -1 | sed 's/^/[smoke] picco campionato: /' 
echo "[smoke] rc=$rc in $(( SECONDS - t0 )) s"
cp "$STAGE"/stage/*.prepass.log "$ROOT/" 2>/dev/null || true
exit $rc
