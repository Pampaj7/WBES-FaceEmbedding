#!/usr/bin/env bash
# Dominio FaceVerse v2 del test zero-shot su 3DMM mai visti (aau/zs3dmm/). Si sourcea DOPO
# aau/env.sh, di solito attraverso zs_env.sh (WBES_ZS_DOMAIN=fv):
#
#   source aau/env.sh; source aau/zs3dmm/fv_env.sh
#
# Solo variabili WBES_FV_*: non tocca WBES_DATA_DIR / WBES_DIST_NPZ / WBES_ICT_* / WBES_FLAME_* /
# WBES_HIFI_*. Stessa regola di hifi_env.sh per la data dir degli eval.
#
# FaceVerse resta SOLO dominio di test: nessuno script di training legge queste variabili, e
# datasets/FaceVerse (i dati FaceVerse preesistenti del repo) non viene toccato: le mesh di qui
# stanno in datasets/FACEVERSE_ZS.
#
# LICENZA: codice FaceVerse BSD-2-Clause; il file del modello non dichiara una licenza propria
# (~/data/faceverse/PROVENANCE.md). Il .npy sta fuori dal repo e non si copia.

export WBES_FV_NPY="${WBES_FV_NPY:-$HOME/data/faceverse/faceverse_simple_v2.npy}"

export WBES_FV_ROOT="${WBES_FV_ROOT:-$WBES_ROOT/datasets/FACEVERSE_ZS}"
export WBES_FV_IDENTITIES_DIR="${WBES_FV_IDENTITIES_DIR:-$WBES_FV_ROOT/identities}"
export WBES_FV_TOPO_DIR="${WBES_FV_TOPO_DIR:-$WBES_FV_ROOT/topo}"
export WBES_FV_GT_DIR="${WBES_FV_GT_DIR:-$WBES_FV_ROOT/gt}"
# Offset 910000: vedi hifi_env.sh.
export WBES_FV_VIEW_DIR="${WBES_FV_VIEW_DIR:-$WBES_FV_ROOT/eval_view}"
export WBES_FV_DATA_DIR="${WBES_FV_DATA_DIR:-$WBES_FV_VIEW_DIR/npz}"
export WBES_FV_DIST_NPZ="${WBES_FV_DIST_NPZ:-$WBES_FV_VIEW_DIR/gt_matrix.npz}"
export WBES_FV_COEF_NPZ="${WBES_FV_COEF_NPZ:-$WBES_FV_VIEW_DIR/gt_coef_matrix.npz}"
export WBES_FV_RUNS="${WBES_FV_RUNS:-$AAU_RUNS/ws_faceverse}"

# Campionamento: come HIFI3D e ICT (pool 500, seed 1234, tutti i 150 modi, N(0,1) non troncata).
export WBES_FV_N_IDENTITIES="${WBES_FV_N_IDENTITIES:-500}"
export WBES_FV_SEED="${WBES_FV_SEED:-1234}"
export WBES_FV_N_SHAPE="${WBES_FV_N_SHAPE:-0}"
export WBES_FV_TRUNC="${WBES_FV_TRUNC:-0}"
export WBES_FV_ID_OFFSET="${WBES_FV_ID_OFFSET:-910000}"

ZS_LABEL="FaceVerse v2"
ZS_PREFIX="fv"
ZS_MODEL_FILE="$WBES_FV_NPY"
ZS_ROOT="$WBES_FV_ROOT"
ZS_IDENTITIES_DIR="$WBES_FV_IDENTITIES_DIR"
ZS_TOPO_DIR="$WBES_FV_TOPO_DIR"
ZS_GT_DIR="$WBES_FV_GT_DIR"
ZS_VIEW_DIR="$WBES_FV_VIEW_DIR"
ZS_DATA_DIR="$WBES_FV_DATA_DIR"
ZS_DIST_NPZ="$WBES_FV_DIST_NPZ"
ZS_COEF_NPZ="$WBES_FV_COEF_NPZ"
ZS_RUNS="$WBES_FV_RUNS"
ZS_N_IDENTITIES="$WBES_FV_N_IDENTITIES"
ZS_SEED="$WBES_FV_SEED"
ZS_N_SHAPE="$WBES_FV_N_SHAPE"
ZS_TRUNC="$WBES_FV_TRUNC"
ZS_ID_OFFSET="$WBES_FV_ID_OFFSET"
