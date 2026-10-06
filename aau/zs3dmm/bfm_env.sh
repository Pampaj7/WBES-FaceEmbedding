#!/usr/bin/env bash
# Dominio "bfm" di aau/zs3dmm: NON zero-shot. I 100 soggetti BFM held-out standard (seed 1234, gli
# stessi di common.heldout_subjects e delle matrici di aau/runs/outlineB/bfm_heldout, verificato)
# fatti passare dalla pipeline zs3dmm (operatori su /tmp, breakdown per topologie) per il
# protocollo equalize-support: serve la base e la variante _eqsupport con lo STESSO percorso.
# Si sourcea DOPO aau/env.sh, attraverso zs_env.sh (WBES_ZS_DOMAIN=bfm).
#
# Niente zs_build.sbatch: la vista e' la dir REMESH com'e' (id0000_GTready_<topologia>.npz,
# chiavi V/F, solo geometria) e la GT e' quella del repo. L'impronta dei dati (zs_data_fp) legge
# un solo manifest, $WBES_BFMZS_ROOT/manifest.json (sorgente, GT e sha1 della GT), scritto da
# eqsupport_view.sbatch; li' sta anche eqsupport_view/.
#
# ATTENZIONE: il congiunto 1019532 ha 81 di questi 100 soggetti nel suo training (split in
# aau/runs/ws2_cross3dmm/splits.json); per lui non sono held-out. BFM-only 1019310 nessuno.
#
# Solo variabili WBES_BFMZS_*: non tocca WBES_DATA_DIR / WBES_DIST_NPZ (vedi hifi_env.sh).

export WBES_BFMZS_SOURCE_DIR="${WBES_BFMZS_SOURCE_DIR:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500}"
export WBES_BFMZS_DIST_NPZ="${WBES_BFMZS_DIST_NPZ:-$WBES_ROOT/face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz}"
export WBES_BFMZS_ROOT="${WBES_BFMZS_ROOT:-$WBES_ROOT/datasets/BFM_ZS}"
export WBES_BFMZS_RUNS="${WBES_BFMZS_RUNS:-$AAU_RUNS/ws_bfm}"

ZS_LABEL="BFM held-out (100 soggetti standard)"
ZS_PREFIX="bfm"
ZS_MODEL_FILE="$WBES_BFMZS_DIST_NPZ"
ZS_ROOT="$WBES_BFMZS_ROOT"
ZS_IDENTITIES_DIR="$WBES_BFMZS_ROOT"
ZS_TOPO_DIR="$WBES_BFMZS_SOURCE_DIR"
ZS_GT_DIR="$WBES_BFMZS_ROOT"
ZS_VIEW_DIR="$WBES_BFMZS_ROOT"
ZS_DATA_DIR="$WBES_BFMZS_SOURCE_DIR"
ZS_DIST_NPZ="$WBES_BFMZS_DIST_NPZ"
ZS_COEF_NPZ=""
ZS_RUNS="$WBES_BFMZS_RUNS"
ZS_N_IDENTITIES=500
ZS_SEED=""
ZS_N_SHAPE=""
ZS_TRUNC=""
ZS_ID_OFFSET=0
