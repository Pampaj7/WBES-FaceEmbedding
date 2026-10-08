#!/usr/bin/env bash
# E1: tabella standard HIFI3D di aau/zs3dmm sui bracci delle celle (WBES_ZS_SUMMARY_DIR=aau/runs/evidence/e1/hifi).
# I bracci E1 stanno in aau/runs/evidence/e1/hifi_runs (WBES_HIFI_RUNS, ws_hifi3d resta intatta); zs_summarize.py
# vuole anche i tre bracci di riferimento e le baseline nella stessa cartella: qui come symlink a ws_hifi3d (sola
# lettura), insieme ai bracci di C3M di curve.md (scale_e036/e072_topology). Checkpoint attesi esportati per
# braccio (WBES_ZS_CKPT_SCALE_<TAG>): zs_summarize controlla che siano quelli degli eval_key.txt.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV="" WBES_ZS_DOMAIN=hifi WBES_HIFI_RUNS="$AAU_RUNS/evidence/e1/hifi_runs"
source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
FP="$(zs_data_fp)"
R="$ZS_RUNS/data_$FP"
SRC="$AAU_RUNS/ws_hifi3d/data_$FP"
[[ -d "$R" ]] || { echo "ERRORE: nessun braccio E1 in $R" >&2; exit 1; }
for d in joint bfm_only ict_only baselines scale_e036_topology scale_e072_topology; do
    [[ -e "$R/$d" ]] || ln -s "$SRC/$d" "$R/$d"
done
SCALE_RUN="$AAU_RUNS/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411"
for e in 036 072; do
    export "WBES_ZS_CKPT_SCALE_E$e=$(ls "$SCALE_RUN"/mixed_*/checkpoints/epoch$e.pth)"
    for c in c2m c2f c3f g1; do
        ptr="$AAU_RUNS/evidence/e1/train_${c}_s1234.runs_root"
        [[ -f "$ptr" ]] || continue
        ck=$(ls "$(cat "$ptr")"/mixed_*/checkpoints/epoch$e.pth 2> /dev/null | head -1)
        [[ -n "$ck" ]] && export "WBES_ZS_CKPT_SCALE_E1${c^^}E$e=$ck"
    done
done
"$AAU_DIR/run.sh" aau/evidence/e1_factorial/e1_zs_summarize.py --label "$ZS_LABEL" --runs "$R" \
    --summary-dir "${WBES_ZS_SUMMARY_DIR:-$AAU_RUNS/evidence/e1/hifi}" --variant "" \
    --gt "$ZS_DIST_NPZ" --gt-coef "$ZS_COEF_NPZ" --root "$ZS_ROOT"
