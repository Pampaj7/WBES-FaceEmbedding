#!/usr/bin/env bash
# Dominio HIFI3D del test zero-shot su 3DMM mai visti (aau/zs3dmm/). Si sourcea DOPO aau/env.sh,
# di solito attraverso zs_env.sh (WBES_ZS_DOMAIN=hifi):
#
#   source aau/env.sh; source aau/zs3dmm/hifi_env.sh
#
# Solo variabili WBES_HIFI_*: non tocca WBES_DATA_DIR / WBES_DIST_NPZ / WBES_ICT_* / WBES_FLAME_*.
# Gli sbatch che chiamano eval_common.sh ricopiano a mano la data dir in WBES_DATA_DIR, SEMPRE e
# senza ${...:-}: env.sh esporta i percorsi BFM e il job li eredita (job 1055122, eval "ICT"
# girato sui dati BFM).
#
# LICENZA: HIFI3D (Tencent AI-NExT, repo MIT, solo ricerca). Il .mat sta fuori dal repo, in
# ~/data/hifi3d, e non si copia nei dataset: si salvano solo le mesh generate.

export WBES_HIFI_MAT="${WBES_HIFI_MAT:-$HOME/data/hifi3d/files/AI-NEXT-Shape.mat}"

export WBES_HIFI_ROOT="${WBES_HIFI_ROOT:-$WBES_ROOT/datasets/HIFI3D}"
export WBES_HIFI_IDENTITIES_DIR="${WBES_HIFI_IDENTITIES_DIR:-$WBES_HIFI_ROOT/identities}"
export WBES_HIFI_TOPO_DIR="${WBES_HIFI_TOPO_DIR:-$WBES_HIFI_ROOT/topo}"
export WBES_HIFI_GT_DIR="${WBES_HIFI_GT_DIR:-$WBES_HIFI_ROOT/gt}"
# Vista idNNNNNN sulla SOLA geometria (symlink a topo/) + GT rinominate. Gli operatori non stanno
# nella home: zs_zeroshot.sbatch li calcola su /tmp del nodo per i soli soggetti valutati.
# Offset 900000: 6 cifre, fuori da BFM id0000-, FLAME id1000-, ICT id10000-14999 e dalle
# identita' ICT nuove di aau/data_scale (ID_BASE 20000, dati di TRAINING).
export WBES_HIFI_VIEW_DIR="${WBES_HIFI_VIEW_DIR:-$WBES_HIFI_ROOT/eval_view}"
export WBES_HIFI_DATA_DIR="${WBES_HIFI_DATA_DIR:-$WBES_HIFI_VIEW_DIR/npz}"
export WBES_HIFI_DIST_NPZ="${WBES_HIFI_DIST_NPZ:-$WBES_HIFI_VIEW_DIR/gt_matrix.npz}"
export WBES_HIFI_COEF_NPZ="${WBES_HIFI_COEF_NPZ:-$WBES_HIFI_VIEW_DIR/gt_coef_matrix.npz}"
export WBES_HIFI_RUNS="${WBES_HIFI_RUNS:-$AAU_RUNS/ws_hifi3d}"

# Campionamento: pool di 500 come eval_view_heldout di ICT (100 valutati, seed 1234);
# coefficienti come ICT: tutti i modi (500 in AI-NEXT-Shape.mat), N(0,1) senza troncamento.
export WBES_HIFI_N_IDENTITIES="${WBES_HIFI_N_IDENTITIES:-500}"
export WBES_HIFI_SEED="${WBES_HIFI_SEED:-1234}"
export WBES_HIFI_N_SHAPE="${WBES_HIFI_N_SHAPE:-0}"
export WBES_HIFI_TRUNC="${WBES_HIFI_TRUNC:-0}"
export WBES_HIFI_ID_OFFSET="${WBES_HIFI_ID_OFFSET:-900000}"

# Variabili generiche lette dagli sbatch di aau/zs3dmm (non esportate: ognuno risourcea).
ZS_LABEL="HIFI3D"
ZS_PREFIX="hifi"
ZS_MODEL_FILE="$WBES_HIFI_MAT"
ZS_ROOT="$WBES_HIFI_ROOT"
ZS_IDENTITIES_DIR="$WBES_HIFI_IDENTITIES_DIR"
ZS_TOPO_DIR="$WBES_HIFI_TOPO_DIR"
ZS_GT_DIR="$WBES_HIFI_GT_DIR"
ZS_VIEW_DIR="$WBES_HIFI_VIEW_DIR"
ZS_DATA_DIR="$WBES_HIFI_DATA_DIR"
ZS_DIST_NPZ="$WBES_HIFI_DIST_NPZ"
ZS_COEF_NPZ="$WBES_HIFI_COEF_NPZ"
ZS_RUNS="$WBES_HIFI_RUNS"
ZS_N_IDENTITIES="$WBES_HIFI_N_IDENTITIES"
ZS_SEED="$WBES_HIFI_SEED"
ZS_N_SHAPE="$WBES_HIFI_N_SHAPE"
ZS_TRUNC="$WBES_HIFI_TRUNC"
ZS_ID_OFFSET="$WBES_HIFI_ID_OFFSET"
