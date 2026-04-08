#!/usr/bin/env bash
set -euo pipefail

ROOT=/deck/datasets/WBES-FaceEmbedding
FACEVERSE_ROOT="$ROOT/datasets/FaceVerse"
PYTHON="$ROOT/.venv_twotower_robust_312/bin/python"
PRECOMP="$ROOT/face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py"
REMESH_SCRIPT="$FACEVERSE_ROOT/remesh_faceverse_from_npz.py"
ASSEMBLE_SCRIPT="$FACEVERSE_ROOT/assemble_faceverse_cross_topology_dataset.py"

GEOM_DIR="$FACEVERSE_ROOT/remesh10k_geometry"
OPS_DIR="$FACEVERSE_ROOT/remesh10k_with_ops"
XTOPO_DIR="$FACEVERSE_ROOT/cross_topology_10k_with_ops"
LOG="$FACEVERSE_ROOT/prepare_faceverse_remesh10k_xtopo.log"

mkdir -p "$GEOM_DIR" "$OPS_DIR" "$XTOPO_DIR"
exec > >(tee -a "$LOG") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] Remeshing FaceVerse from existing 10k operator meshes"
export VIRTUAL_ENV="$ROOT/.venv_twotower_robust_312"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME
export WBES_DIFFUSION_NET_SRC="$ROOT/diffusion-net/src"
export PYTHONUNBUFFERED=1

"$PYTHON" "$REMESH_SCRIPT" \
  --input_dir "$FACEVERSE_ROOT/downsampled_with_ops" \
  --output_dir "$GEOM_DIR" \
  --pattern '*_01.npz'

echo "[$(timestamp)] Precomputing DiffusionNet operators for remeshed FaceVerse"
"$PYTHON" "$PRECOMP" \
  --input-dir "$GEOM_DIR" \
  --output-dir "$OPS_DIR" \
  --input-kind npz \
  --pattern '*_remesh_10k.npz' \
  --k-eig 128 \
  --n-cores 8

echo "[$(timestamp)] Assembling cross-topology dataset directory"
"$PYTHON" "$ASSEMBLE_SCRIPT" \
  --original_dir "$FACEVERSE_ROOT/downsampled_with_ops" \
  --remesh_dir "$OPS_DIR" \
  --output_dir "$XTOPO_DIR" \
  --pattern '*_01.npz' \
  --overwrite

echo "[$(timestamp)] FaceVerse remesh10k cross-topology dataset ready"
