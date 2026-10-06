#!/usr/bin/env bash
# Dominio GNM Head v3.0 (Google) del test zero-shot su 3DMM mai visti (aau/zs3dmm/). Si sourcea
# DOPO aau/env.sh, di solito attraverso zs_env.sh (WBES_ZS_DOMAIN=gnm):
#
#   source aau/env.sh; source aau/zs3dmm/gnm_env.sh
#
# Solo variabili WBES_GNM_*: non tocca WBES_DATA_DIR / WBES_DIST_NPZ / WBES_ICT_* / WBES_FLAME_* /
# WBES_HIFI_* / WBES_FV_*. Stessa regola di hifi_env.sh per la data dir degli eval.
#
# GNM resta SOLO dominio di test: nessuno script di training legge queste variabili. Regione del
# volto (hockey_mask, quad a ventaglio) e posa neutra: gnm_model.py.
#
# LICENZA: codice e pesi Apache-2.0 (~/data/gnm_head/PROVENANCE.md). Il .npz sta fuori dal repo
# e non si copia: si salvano solo le mesh generate.

export WBES_GNM_NPZ="${WBES_GNM_NPZ:-$HOME/data/gnm_head/gnm_head.npz}"

export WBES_GNM_ROOT="${WBES_GNM_ROOT:-$WBES_ROOT/datasets/GNM_ZS}"
export WBES_GNM_IDENTITIES_DIR="${WBES_GNM_IDENTITIES_DIR:-$WBES_GNM_ROOT/identities}"
export WBES_GNM_TOPO_DIR="${WBES_GNM_TOPO_DIR:-$WBES_GNM_ROOT/topo}"
export WBES_GNM_GT_DIR="${WBES_GNM_GT_DIR:-$WBES_GNM_ROOT/gt}"
# Offset 920000: vedi hifi_env.sh. Lo stesso offset del controllo ICT (ict_env.sh), ma intervalli
# disgiunti: GNM id920000-id920499, ICT held-out id924500-id924999; e le viste sono separate.
export WBES_GNM_VIEW_DIR="${WBES_GNM_VIEW_DIR:-$WBES_GNM_ROOT/eval_view}"
export WBES_GNM_DATA_DIR="${WBES_GNM_DATA_DIR:-$WBES_GNM_VIEW_DIR/npz}"
export WBES_GNM_DIST_NPZ="${WBES_GNM_DIST_NPZ:-$WBES_GNM_VIEW_DIR/gt_matrix.npz}"
export WBES_GNM_COEF_NPZ="${WBES_GNM_COEF_NPZ:-$WBES_GNM_VIEW_DIR/gt_coef_matrix.npz}"
export WBES_GNM_RUNS="${WBES_GNM_RUNS:-$AAU_RUNS/ws_gnm}"

# Campionamento: come HIFI3D, FaceVerse e ICT (pool 500, seed 1234, N(0,1) non troncata), sulle
# 170 basi head_* (n_shape 0 = tutte e 170; occhi e denti a zero, gnm_model.py).
export WBES_GNM_N_IDENTITIES="${WBES_GNM_N_IDENTITIES:-500}"
export WBES_GNM_SEED="${WBES_GNM_SEED:-1234}"
export WBES_GNM_N_SHAPE="${WBES_GNM_N_SHAPE:-0}"
export WBES_GNM_TRUNC="${WBES_GNM_TRUNC:-0}"
export WBES_GNM_ID_OFFSET="${WBES_GNM_ID_OFFSET:-920000}"

ZS_LABEL="GNM Head v3.0"
ZS_PREFIX="gnm"
ZS_MODEL_FILE="$WBES_GNM_NPZ"
ZS_ROOT="$WBES_GNM_ROOT"
ZS_IDENTITIES_DIR="$WBES_GNM_IDENTITIES_DIR"
ZS_TOPO_DIR="$WBES_GNM_TOPO_DIR"
ZS_GT_DIR="$WBES_GNM_GT_DIR"
ZS_VIEW_DIR="$WBES_GNM_VIEW_DIR"
ZS_DATA_DIR="$WBES_GNM_DATA_DIR"
ZS_DIST_NPZ="$WBES_GNM_DIST_NPZ"
ZS_COEF_NPZ="$WBES_GNM_COEF_NPZ"
ZS_RUNS="$WBES_GNM_RUNS"
ZS_N_IDENTITIES="$WBES_GNM_N_IDENTITIES"
ZS_SEED="$WBES_GNM_SEED"
ZS_N_SHAPE="$WBES_GNM_N_SHAPE"
ZS_TRUNC="$WBES_GNM_TRUNC"
ZS_ID_OFFSET="$WBES_GNM_ID_OFFSET"
