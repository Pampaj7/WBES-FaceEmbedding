#!/usr/bin/env bash
# Set di SVILUPPO FaceScape bilineare (D1, paper/PLAN_MASSIVE.md sez. 9 e 14.5) dentro la pipeline
# zero-shot ESISTENTE, senza toccarne i file. Si sourcea al posto di WBES_ZS_DOMAIN=...:
#
#   ( source aau/zs3dmm/dev_facescape_env.sh
#     WBES_ZS_ARMS=scale_e108 WBES_ZS_CKPT_SCALE_E108=<.../checkpoints/epoch108.pth> WBES_ZS_PART=topology \
#       aau/submit.sh zs3dmm/zs_zeroshot.sbatch )
#   ( source aau/zs3dmm/dev_facescape_env.sh; export WBES_ZS_EXPR=1; ... )    # la variante con espressioni
#
# Come: zs_env.sh accetta solo i domini hifi|fv|gnm|ict|bfm e ne sourcea il <dominio>_env.sh, che
# definisce ogni percorso come ${WBES_<DOM>_*:-default}. Qui si sceglie lo slot "fv" (l'unico, con
# gnm, dove WBES_ZS_EXPR=1 apre la vista con espressioni) e si ridirigono TUTTE le sue variabili
# su datasets/DEV_FACESCAPE e aau/runs/ws_dev_facescape. Assegnazioni esplicite, non ${...:-}: un
# WBES_FV_* gia' nell'ambiente (il FaceVerse vero) non deve passare. Gli sbatch girano col loro
# ambiente (sbatch esporta tutto), quindi la ridirezione arriva nel job.
#
# Cosa resta di FaceVerse e non conta: ZS_LABEL "FaceVerse v2" nelle righe di log e "_fv" nel
# nome della dir su /tmp. ZS_PREFIX "fv" lo usano solo i builder dello zero-shot, che qui NON
# vanno lanciati (zs_build.sbatch / zs_build_expr.sbatch chiamerebbero il loader FaceVerse sul file
# FaceScape e fallirebbero): i dati li fa dev_fs_build.sbatch, coi file fsNNNN.
# Provato: zs_zeroshot.sbatch con WBES_ZS_PART=embed su CPU (job 1061747/1061748, D1); il breakdown
# su L40S (job 1061735/1061736) era in coda alla consegna. Dovrebbero andare, non lanciati in D1:
# zs_baselines.sbatch e zs_expr_extra.sbatch passo same (faceBench, set "fv_heldout" di zs_bl.py
# con offset 930000). NON zs_expr_extra.sbatch passo region: passerebbe --domain fv, cioe' gli
# assi di FaceVerse (alto -y), mentre FaceScape ha quelli di HIFI3D: per la Chamfer su regione si
# usa dev_fs_chamfer.sbatch, che passa --domain hifi.
#
# LICENZA: FaceScape, solo ricerca non commerciale, niente ridistribuzione. Il .npz del modello sta
# in ~/data/facescape_bilinear e non si copia; in datasets/ (ignorata da git) solo mesh generate.

source "${WBES_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/aau/env.sh"

export WBES_ZS_DOMAIN=fv
export WBES_DEV_FACESCAPE=1
export WBES_FV_NPY="${WBES_DEV_FS_MODEL:-$HOME/data/facescape_bilinear/facescape_bm_v1.6_847_300_52_id.npz}"
export WBES_FV_ROOT="$WBES_ROOT/datasets/DEV_FACESCAPE"
export WBES_FV_IDENTITIES_DIR="$WBES_FV_ROOT/identities"
export WBES_FV_TOPO_DIR="$WBES_FV_ROOT/topo"
export WBES_FV_GT_DIR="$WBES_FV_ROOT/gt"
export WBES_FV_VIEW_DIR="$WBES_FV_ROOT/eval_view"
export WBES_FV_DATA_DIR="$WBES_FV_VIEW_DIR/npz"
export WBES_FV_DIST_NPZ="$WBES_FV_VIEW_DIR/gt_matrix.npz"
export WBES_FV_COEF_NPZ="$WBES_FV_VIEW_DIR/gt_coef_matrix.npz"
export WBES_FV_RUNS="$AAU_RUNS/ws_dev_facescape"
# Pool di 500 (zs_stage.py ne estrae i soliti 100 col seed 1234), N(0,1) non troncata su tutti i
# 300 modi; offset 930000: fuori da BFM, FLAME, ICT, ICT nuovi, GNM di training (100000+) e dagli
# altri zero-shot (900000 HIFI3D, 910000 FaceVerse, 920000 GNM / ICT held-out).
export WBES_FV_N_IDENTITIES=500
export WBES_FV_SEED=1234
export WBES_FV_N_SHAPE=0
export WBES_FV_TRUNC=0
export WBES_FV_ID_OFFSET=930000
DEV_FS_PREFIX=fs
