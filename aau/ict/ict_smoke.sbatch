#!/usr/bin/env bash
#SBATCH --job-name=wbes-ict-smoke
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili, e un
# --output relativo viene risolto nella submit dir. Sottometti con aau/submit.sh, che li
# ricalcola e li passa da riga di comando.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#
# Prova a vuoto della pipeline ICT-5000 su 8 identita', in una dir usa-e-getta:
# self-check del modello, generazione, 6 topologie, demo delle espressioni.
# Serve a misurare il costo per identita' e a beccare gli errori prima dei 5000.
#
#   aau/submit.sh ict/ict_smoke.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv

TMP_DIR="${WBES_ICT_SMOKE_DIR:-$WBES_ROOT/datasets/ICT/_smoke}"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

echo "[smoke] host=$(hostname) job=${SLURM_JOB_ID:-none} tmp=$TMP_DIR"

echo "--- 1. modello ICT (parser OBJ, formula, regione volto) ---"
time "$AAU_DIR/run.sh" v2_work/genict/ict_model.py

echo "--- 2. 8 identita' (senza check di forma: e' O(n^2) e qui n=8) ---"
time "$AAU_DIR/run.sh" v2_work/genict/generate_identities.py \
    --n-identities 8 --seed 1234 --out-dir "$TMP_DIR/identities" --device cpu
cat "$TMP_DIR/identities/manifest.json"

echo "--- 3. 6 topologie ---"
time "$AAU_DIR/run.sh" v2_work/genict/make_ict_topologies.py \
    --in-dir "$TMP_DIR/identities" --out-dir "$TMP_DIR/topo" --n-cores 4

echo "--- 4. verifica delle topologie su 3 identita' ---"
"$AAU_DIR/run.sh" v2_work/genict/make_ict_topologies.py --demo \
    --in-dir "$TMP_DIR/identities" --out-dir "$TMP_DIR/topo"

echo "--- 5. matrice GT (8x8) ---"
"$AAU_DIR/run.sh" v2_work/genict/build_ict_gt_matrix.py \
    --topo-dir "$TMP_DIR/topo" --out-dir "$TMP_DIR/gt" --device cpu

echo "--- 6. espressioni: demo ---"
"$AAU_DIR/run.sh" v2_work/genict/make_expressions.py --demo

echo "--- 7. operatori area-norm su 6 mesh (1 identita', tutte le topologie) ---"
mkdir -p "$TMP_DIR/one" "$TMP_DIR/one_withops"
cp "$TMP_DIR"/topo/ict0000_GTready_*.npz "$TMP_DIR/one/"
time "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
    --input-dir "$TMP_DIR/one" --output-dir "$TMP_DIR/one_withops" --k-eig 128
T="$TMP_DIR" "$AAU_DIR/run.sh" -c "
import numpy as np, glob, os
for p in sorted(glob.glob(os.environ['T'] + '/one_withops/*.npz')):
    with np.load(p) as z:
        t = z['verts'][z['faces']]
        A = 0.5*np.linalg.norm(np.cross(t[:,1]-t[:,0], t[:,2]-t[:,0]), axis=1).sum()
        print(f\"  {os.path.basename(p)}: verts={len(z['verts'])} faces={len(z['faces'])} \"
              f\"k={z['evals'].shape[0]} area={A:.6f}\")
        assert abs(A-1) < 1e-3, 'area non unitaria'
print('  operatori OK')
"

echo "[smoke] dimensione operatori di 1 identita' (6 topologie): $(du -sh "$TMP_DIR/one_withops" | cut -f1)"
echo "[smoke] OK"
