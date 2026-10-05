#!/usr/bin/env bash
#SBATCH --job-name=wbes-abl-smoke
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#
# Smoke delle ablazioni v3 B ed E su 10 soggetti BFM (id0000-id0009, 60 npz), sullo stesso A10
# dei run veri. Prima:
#   0. tabelle dei token BFM (seed 1234) e ICT, se mancano (aau/models/size_token.py), con la
#      verifica di consistenza fra topologie (<json>.check.md);
#   1. operatori robusti ad area 1 dei 10 soggetti BFM e di 10 held-out ICT, scritti nelle
#      cartelle VERE (il precompute completo li salta), con --check contro quelle standard.
# Per braccio: training di 2 epoche con aau/models/train_ablation.sbatch, eval in-domain
# (eval_cells.py) e breakdown zero-shot del repo su 10 soggetti ICT, entrambi da eval_ablation.py.
# Poi i controlli sul token: entra nell'embedding, il forward senza token si ferma.
#   WBES_SMOKE_ARMS="B E" (default)
set -euo pipefail
source aau/env.sh
OUT=$AAU_DIR/scratch/abl3_smoke_runs/${SLURM_JOB_ID:-manual}
STD_VIEW=$AAU_DIR/scratch/rms_smoke_view
RA1_FULL=$WBES_ROOT/datasets/REMESH/npz_data_topo_500_withops_robust_area1
RA1_VIEW=$AAU_DIR/scratch/abl3_smoke_view_ra1
ICT_STD_FULL=$WBES_ROOT/datasets/ICT/eval_view_heldout
ICT_RA1_FULL=$WBES_ROOT/datasets/ICT/eval_view_heldout_robust_area1
ICT_STD_VIEW=$AAU_DIR/scratch/abl3_smoke_view_ict
ICT_RA1_VIEW=$AAU_DIR/scratch/abl3_smoke_view_ict_ra1
ICT_DIST=$WBES_ROOT/datasets/ICT/train_ready/gt_matrix.npz
TOK_BFM=$AAU_RUNS/ablations_v3/size_token_bfm_s1234.json
TOK_ICT=$AAU_RUNS/ablations_v3/size_token_ict.json
PERT=face_embedding/gt_encdec/remeshing/intrinsic/perturbated
mkdir -p "$OUT" "$AAU_RUNS/ablations_v3"

nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 2 > "$OUT/gpu_mem.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT

echo "================ 0. token di taglia"
[[ -f "$TOK_BFM" ]] || "$AAU_DIR/run.sh" aau/models/size_token.py bfm \
    --raw-dir datasets/REMESH/npz_data_topo_500 --dist-npz "$WBES_DIST_NPZ" --seed 1234 \
    --workers 8 --out "$TOK_BFM"
[[ -f "$TOK_ICT" ]] || "$AAU_DIR/run.sh" aau/models/size_token.py ict \
    --raw-dir datasets/ICT/topo --id-offset 10000 --train-list datasets/ICT/train_ready/split_train.txt \
    --workers 8 --out "$TOK_ICT"

# La tabella ICT viene dalla cartella mesh-only datasets/ICT/topo (coordinate grezze, cm): i vertici
# della vista held-out (topo_withops) NON sono grezzi: sono la stessa mesh centrata e riscalata per
# mesh (fattore ~0.034, non maxabs; misurato con job 1055059). Qui si verifica che la vista e' una
# similitudine della mesh grezza e che la tabella e' il log r della grezza.
SMOKE_TOK_ICT="$TOK_ICT" "$AAU_DIR/run.sh" - <<'PY'
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "aau/models")
from size_token import read_one
t = json.loads(Path(os.environ["SMOKE_TOK_ICT"]).read_text())["log_r"]
for raw, n in (("ict4500_GTready_original", "id14500_GTready_original"),
               ("ict4500_GTready_crop", "id14500_GTready_crop"), ("ict4999_GTready_up60k", "id14999_GTready_up60k")):
    _, v_raw = read_one(Path("datasets/ICT/topo") / f"{raw}.npz")
    with np.load(Path("datasets/ICT/topo") / f"{raw}.npz") as z:
        Vr = (z["verts"] if "verts" in z.files else z["V"]).astype(np.float64)
    with np.load(Path("datasets/ICT/eval_view_heldout") / f"{n}.npz") as z:
        Vv = z["verts"].astype(np.float64)
    X, Y = Vr - Vr.mean(0), Vv - Vv.mean(0)
    a = float((X * Y).sum() / (X * X).sum())
    dv = float(np.abs(a * X - Y).max())
    print(f"[token] {n}: tabella {t[n]:.6f}, grezzo {v_raw:.6f}; vista = grezzo centrato x {a:.5f} "
          f"a meno di {dv:.1e}")
    assert abs(v_raw - t[n]) < 1e-9 and dv < 1e-4, n
PY

echo "================ 1. operatori robusti ad area 1 per lo smoke"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pids=()
for s in 0 1 2 3 4 5 6 7; do
    "$AAU_DIR/run.sh" aau/models/robust_area1_operators.py --input-dir datasets/REMESH/npz_data_topo_500 \
        --output-dir "$RA1_FULL" --pattern 'id000[0-9]_*' --shard "$s/8" > "$OUT/ra1_bfm_$s.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
pids=()
for s in 0 1 2 3 4 5 6 7; do
    "$AAU_DIR/run.sh" aau/models/robust_area1_operators.py --input-dir "$ICT_STD_FULL" \
        --output-dir "$ICT_RA1_FULL" --pattern 'id1450[0-9]_*' --shard "$s/8" > "$OUT/ra1_ict_$s.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
mkdir -p "$RA1_VIEW" "$ICT_STD_VIEW" "$ICT_RA1_VIEW"
for f in "$STD_VIEW"/*.npz; do ln -sfn "$RA1_FULL/$(basename "$f")" "$RA1_VIEW/$(basename "$f")"; done
for f in "$ICT_STD_FULL"/id1450[0-9]_*.npz; do
    ln -sfn "$(realpath "$f")" "$ICT_STD_VIEW/$(basename "$f")"
    ln -sfn "$ICT_RA1_FULL/$(basename "$f")" "$ICT_RA1_VIEW/$(basename "$f")"
done
"$AAU_DIR/run.sh" aau/models/robust_area1_operators.py --check --check-all --input-dir "$STD_VIEW" \
    --output-dir "$RA1_VIEW" --ref-dir "$STD_VIEW"
"$AAU_DIR/run.sh" aau/models/robust_area1_operators.py --check --check-all --input-dir "$ICT_STD_VIEW" \
    --output-dir "$ICT_RA1_VIEW" --ref-dir "$ICT_STD_VIEW"

for ARM in ${WBES_SMOKE_ARMS:-B E}; do
    echo "================ $ARM"
    if [[ $ARM == B ]]; then DATA=$STD_VIEW; ICT_DATA=$ICT_STD_VIEW; else DATA=$RA1_VIEW; ICT_DATA=$ICT_RA1_VIEW; fi
    t0=$(date +%s)
    WBES_ABL_ARM=$ARM WBES_DATA_DIR=$STD_VIEW WBES_RA1_DIR=$RA1_VIEW WBES_EPOCHS=2 \
      WBES_RUNS_ROOT=$OUT/$ARM bash aau/models/train_ablation.sbatch
    echo "[smoke] $ARM training 2 epoche: $(( $(date +%s) - t0 )) s"
    CK=$(ls -d "$OUT/$ARM"/*/checkpoints/best_by_xtopo_mesh_clean.pth)
    cat "$OUT/$ARM/ablation_arm.txt"

    "$AAU_DIR/run.sh" aau/models/eval_ablation.py --frame rms --size-token-json "$TOK_BFM" -- \
      aau/models/eval_cells.py --checkpoint "$CK" --data-dir "$DATA" --dist-npz "$WBES_DIST_NPZ" \
      --subject-split all --n-subjects 10 --device cuda --tag "smoke_$ARM" --out-dir "$OUT/eval"

    "$AAU_DIR/run.sh" aau/models/eval_ablation.py --frame rms --size-token-json "$TOK_ICT" -- \
      "$PERT/compare_model_vs_chamfer_topology_breakdown.py" --model_path "$CK" \
      --checkpoint_selector best_by_clean --data_dir "$ICT_DATA" --dist_npz "$ICT_DIST" \
      --subject_split all --max_subjects 10 --max_meshes_per_subject_eval 10 --seed 1234 \
      --preload_workers 4 --pair_mode cross_topology --chamfer_batch_pairs 256 \
      --out_dir "$OUT/ict_breakdown_$ARM" --topology_pair_mode cross_only \
      --ordered_topology_pairs --write_pair_metrics
done

echo "================ controlli sul token"
CK=$(ls -d "$OUT/B"/*/checkpoints/best_by_xtopo_mesh_clean.pth)
"$AAU_DIR/run.sh" aau/models/check_size_token.py --checkpoint "$CK" --data-dir "$STD_VIEW" \
    --size-token-json "$TOK_BFM"

kill $SMI 2>/dev/null || true
echo "[smoke] picco memoria GPU: $(cut -d, -f2 "$OUT/gpu_mem.csv" | tr -dc '0-9\n' | sort -n | tail -1) MiB"
echo "[smoke] fatto: $OUT"
