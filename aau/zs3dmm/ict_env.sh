#!/usr/bin/env bash
# Dominio "ict" di aau/zs3dmm: NON un dominio nuovo, ma il controllo della pipeline. Le 500
# identita' held-out di ICT-5000 (ict4500-ict4999, il pool di datasets/ICT/eval_view_heldout)
# rifatte da capo con zs3dmm (topologie, GT, vista, operatori su /tmp, eval) come se fossero
# un 3DMM mai visto. Con lo stesso seed lo split estrae gli STESSI 100 soggetti dello zero-shot
# ICT esistente (rebuild_subject_split dipende solo dall'ordine dei nomi, e id924500.. si
# ordina come id14500..), quindi:
#   - BFM-only deve riprodurre lo zero-shot BFM->ICT esistente (0.294 mesh-pair, job 1019695);
#   - congiunto e ICT-only ~0.98, ma NON e' generalizzazione: ~80 di questi 100 soggetti erano
#     nel loro training (ws2_views.py: dello stesso pool solo 89/95 sono held-out).
# Serve anche al test del frame su ICT (WBES_ZS_FRAME=x,-y,-z, frame dei dati BFM).
#
# Solo variabili WBES_ICTZS_*: non tocca WBES_ICT_* (la pipeline ICT vera) ne' le altre.

export WBES_ICTZS_SOURCE_DIR="${WBES_ICTZS_SOURCE_DIR:-$WBES_ROOT/datasets/ICT/identities}"
export WBES_ICTZS_FIRST="${WBES_ICTZS_FIRST:-4500}"
export WBES_ICTZS_ROOT="${WBES_ICTZS_ROOT:-$WBES_ROOT/datasets/ICT_ZS}"
export WBES_ICTZS_IDENTITIES_DIR="${WBES_ICTZS_IDENTITIES_DIR:-$WBES_ICTZS_ROOT/identities}"
export WBES_ICTZS_TOPO_DIR="${WBES_ICTZS_TOPO_DIR:-$WBES_ICTZS_ROOT/topo}"
export WBES_ICTZS_GT_DIR="${WBES_ICTZS_GT_DIR:-$WBES_ICTZS_ROOT/gt}"
export WBES_ICTZS_VIEW_DIR="${WBES_ICTZS_VIEW_DIR:-$WBES_ICTZS_ROOT/eval_view}"
export WBES_ICTZS_DATA_DIR="${WBES_ICTZS_DATA_DIR:-$WBES_ICTZS_VIEW_DIR/npz}"
export WBES_ICTZS_DIST_NPZ="${WBES_ICTZS_DIST_NPZ:-$WBES_ICTZS_VIEW_DIR/gt_matrix.npz}"
export WBES_ICTZS_COEF_NPZ="${WBES_ICTZS_COEF_NPZ:-$WBES_ICTZS_VIEW_DIR/gt_coef_matrix.npz}"
export WBES_ICTZS_RUNS="${WBES_ICTZS_RUNS:-$AAU_RUNS/ws_ictzs}"
export WBES_ICTZS_ID_OFFSET="${WBES_ICTZS_ID_OFFSET:-920000}"

ZS_LABEL="ICT held-out (controllo della pipeline)"
ZS_PREFIX="ict"
ZS_MODEL_FILE="$WBES_ICTZS_SOURCE_DIR/manifest.json"
ZS_ROOT="$WBES_ICTZS_ROOT"
ZS_IDENTITIES_DIR="$WBES_ICTZS_IDENTITIES_DIR"
ZS_TOPO_DIR="$WBES_ICTZS_TOPO_DIR"
ZS_GT_DIR="$WBES_ICTZS_GT_DIR"
ZS_VIEW_DIR="$WBES_ICTZS_VIEW_DIR"
ZS_DATA_DIR="$WBES_ICTZS_DATA_DIR"
ZS_DIST_NPZ="$WBES_ICTZS_DIST_NPZ"
ZS_COEF_NPZ="$WBES_ICTZS_COEF_NPZ"
ZS_RUNS="$WBES_ICTZS_RUNS"
ZS_N_IDENTITIES=500
ZS_SEED=""
ZS_N_SHAPE=""
ZS_TRUNC=""
ZS_ID_OFFSET="$WBES_ICTZS_ID_OFFSET"
