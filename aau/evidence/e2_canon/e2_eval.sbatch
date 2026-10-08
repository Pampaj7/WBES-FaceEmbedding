#!/usr/bin/env bash
#SBATCH --job-name=wbes-e2-eval
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# E2: e108 con canonicalizzazione rigida al test (canon.py), su un dominio (WBES_E2_DOMAIN):
#   hifi  HIFI3D, 100 soggetti x 6 topologie: canon -> operatori su /tmp -> embedding e108; Chamfer
#         faceBench sulle stesse mesh canonicalizzate (+ controllo sulla vista originale)
#   fv    FaceVerse con espressioni, idem (frame atteso Rx180; facce NON invertite: dopo la rotazione
#         le mesh sono nella convenzione ICT completa, la stessa del riferimento)
#   now   NoW validation: patch scan_face e recon_face canonicalizzate fuori dal repo (licenza), operatori,
#         latenti e108 (now_latent.py) e Chamfer grezza / ICP + Chamfer (now_geometric.py --mode gt)
# Uscite in aau/runs/evidence/e2/<dominio>/. Riferimento e parametri: protocol.md (dichiarato prima).
#
#   WBES_E2_DOMAIN=hifi aau/submit.sh evidence/e2_canon/e2_eval.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/evidence/e2_canon/ops_embed.sh"
# Su CPU (sottomesso con --partition=cpu --gres=none): niente --nv a singularity.
[[ -n "${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" ]] || export AAU_NV=
# Un thread BLAS per processo: gli stadi in Python sono pool da un processo per core (canon.py, remesh.py,
# chamfer_canon.py, now_geometric.py); senza, ogni processo apre un thread per core del nodo. L'embedding
# (ev_embed) e i latenti NoW rialzano i thread al numero di CPU del job.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
aau_require_venv
aau_require_diffusion_net

DOM="${WBES_E2_DOMAIN:?WBES_E2_DOMAIN=hifi|fv|now}"
OUT="$AAU_RUNS/evidence/e2"
REF="$OUT/reference_ict_mean.npz"
[[ -f "$OUT/protocol.md" ]] || { echo "ERRORE: $OUT/protocol.md assente: il protocollo va dichiarato prima" >&2; exit 2; }
NW="${SLURM_CPUS_PER_TASK:-8}"
TMPD="/tmp/wbes_e2_${SLURM_JOB_ID:-manual}_$DOM"
rm -rf "$TMPD"; mkdir -p "$TMPD"
trap 'rm -rf "$TMPD"' EXIT
echo "[e2] dominio=$DOM host=$(hostname) job=${SLURM_JOB_ID:-none} ckpt=$E108_CKPT"

case "$DOM" in
  hifi|fv)
    if [[ "$DOM" == hifi ]]; then
        VIEW="$WBES_ROOT/datasets/HIFI3D/eval_view"; CONV=I
        CTL_ROOT="$AAU_RUNS/ws_hifi3d/data_328f2bfc1a/baselines"
    else
        VIEW="$WBES_ROOT/datasets/FACEVERSE_ZS/expr_view"; CONV=Rx180
        CTL_ROOT="$AAU_RUNS/ws_faceverse_expr/data_736f96956a/baselines"
    fi
    D="$OUT/$DOM"; mkdir -p "$D"
    echo "[e2] a. canonicalizzazione di $VIEW/npz"
    AAU_NV= "$AAU_DIR/run.sh" aau/evidence/e2_canon/canon.py stage --reference "$REF" --view-dir "$VIEW/npz" \
        --seed 1234 --out-dir "$TMPD/in" --records "$D/canon_records.csv" --convention "$CONV" --workers "$NW"
    cp "$TMPD/subjects.json" "$D/subjects.json"
    echo "[e2] b. operatori"
    ev_ops "$TMPD/in" "$TMPD/ops" "${DOM}_canon"
    echo "[e2] c. embedding e108"
    ev_embed "$TMPD/ops" "$VIEW/gt_matrix.npz" "$D/canon_e108"
    rm -rf "$TMPD/ops"
    echo "[e2] d. Chamfer faceBench sulle mesh canonicalizzate (+ controllo sulla vista originale)"
    AAU_NV= "$AAU_DIR/outlineB/run_o3d.sh" aau/evidence/e2_canon/chamfer_canon.py --mesh-dir "$TMPD/in" \
        --subjects-json "$TMPD/subjects.json" --out-root "$D/canon_chamfer" --workers "$NW" \
        --control-view "$VIEW/npz" --control-root "$CTL_ROOT" 2>&1 | grep -v "valid test operator"
    ;;
  now)
    D="$OUT/now"; mkdir -p "$D"
    W="$TMPD/work"; KEEP="$HOME/data/now_eval_e2canon"
    SRC_W="$HOME/data/now_eval_work"; SRC_O="$AAU_RUNS/now_eval"
    mkdir -p "$W/embeddings" "$W/pairs" "$KEEP"
    for x in official template_lmk7.json; do ln -sfn "$SRC_W/$x" "$W/$x"; done
    for f in "$SRC_O"/now_official_* "$SRC_O"/protocol.md; do ln -sf "$f" "$D/"; done
    echo "ckpt=$E108_CKPT" > "$D/checkpoint.txt"
    echo "[e2] a. canonicalizzazione delle patch NoW"
    AAU_NV= "$AAU_DIR/run.sh" aau/evidence/e2_canon/canon.py now --reference "$REF" --src-work "$SRC_W" \
        --dst-work "$W" --records "$D/canon_records.csv" --workers "$NW"
    echo "[e2] b. operatori"
    ev_ops "$W/scan_face" "$W/scan_face_withops_areanorm" now_scan
    for m in 3ddfa_v2 synergynet prnet mica; do
        ev_ops "$W/recon_face/$m" "$W/recon_face_withops_areanorm/$m" "now_$m"
    done
    echo "[e2] c. latenti e108 (now_latent.py) e Chamfer (now_geometric.py --mode gt)"
    WBES_NOW_WORK="$W" WBES_NOW_OUT="$D" WBES_CKPT="$E108_CKPT" OMP_NUM_THREADS="$NW" MKL_NUM_THREADS="$NW" \
        "$AAU_DIR/run.sh" aau/recon/now_latent.py --methods 3ddfa_v2,synergynet,prnet,mica 2>&1 | grep -v "valid test operator"
    WBES_NOW_WORK="$W" WBES_NOW_OUT="$D" AAU_NV= \
        "$AAU_DIR/run.sh" aau/recon/now_geometric.py --mode gt --workers "$NW" 2>&1 | grep -v "valid test operator"
    # latenti e matrici (nomi NoW: fuori dal repo), per rifare i conti senza GPU
    cp -r "$W/embeddings" "$W/pairs" "$KEEP/"
    ;;
  *) echo "ERRORE: WBES_E2_DOMAIN=$DOM" >&2; exit 2 ;;
esac
echo "[e2] $DOM OK"
