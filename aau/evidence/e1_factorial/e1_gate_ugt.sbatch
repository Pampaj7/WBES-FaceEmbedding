#!/usr/bin/env bash
#SBATCH --job-name=wbes-e1-gate-ugt
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=36:00:00
#
# E1, cancello di C3F-UGT: esce 0 solo quando la GT unificata di training esiste ed e' VERIFICATA col loader
# del trainer (datasets/UNIFIED_GT/train/check.json di v3_work/unified_gt/check_train_gt.py, stesse condizioni
# del suo "FALLITO"). Il training di C3F-UGT dipende da questo job (afterok): se la GT non arriva in 36 h il
# cancello fallisce e il training si cancella da solo (--kill-on-invalid-dep). Controllo ogni 10 minuti.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV=""
D="$WBES_ROOT/datasets/UNIFIED_GT/train"
while true; do
    if [[ -f "$D/gt_unified_bfm_ict_gnm.npz" && -f "$D/gt_unified_bfm_ict_gnm.json" && -f "$D/check.json" ]] \
       && "$AAU_DIR/run.sh" -c "
import json, sys
c = json.load(open(sys.argv[1]))
bad = (not c['name_to_idx_identical_to_run_gt'] or c['n_nonfinite'] or c['diag_max_abs'] > 0
       or c['symmetry_max_abs_sample'] > 0 or not all(v['all_same_index'] for v in c['index_match'].values())
       or c['values_vs_s_train_max_abs_mm'] > 1e-3)
sys.exit(int(bad))
" "$D/check.json" 2> /dev/null; then
        # il json di check deve essere piu' recente della GT (una GT riscritta va ricontrollata)
        if [[ "$D/check.json" -nt "$D/gt_unified_bfm_ict_gnm.npz" ]]; then
            echo "[e1-gate] $(date +%F_%T) GT unificata verificata: $D/check.json"
            exit 0
        fi
    fi
    sleep 600
done
