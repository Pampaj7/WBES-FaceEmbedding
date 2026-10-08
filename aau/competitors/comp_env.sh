#!/usr/bin/env bash
# Competitori diretti sulla verita' geometrica di HIFI3D (literature/COMPETITORS_GEOM_2026-10-08.md).
# Si sourcea DOPO aau/env.sh:
#
#   source aau/env.sh; source aau/competitors/comp_env.sh
#
# Codice e pesi di terzi stanno in external_models/competitors_geom/ (gitignored, mai in git):
#   uni3d_src/    clone di github.com/baaivision/Uni3D        (commit 64e03c3c42c1)
#   openshape_src/ clone di github.com/Colin97/OpenShape_code (commit abe5aa42b7c9)
#   weights_uni3d_g/model.pt                      HF BAAI/Uni3D modelzoo/uni3d-g/model.pt
#   weights_openshape_pointbert_vitg14_rgb/model.pt HF OpenShape/openshape-pointbert-vitg14-rgb
#   venv/         venv nel container (--system-site-packages) con timm, easydict, torch_redstone, einops
# Gli spettrali girano nel venv del repo ($VENV): l'operatore e' quello di diffusion_net.

export COMP_EXT="${COMP_EXT:-$WBES_ROOT/external_models/competitors_geom}"
export COMP_VENV="${COMP_VENV:-$COMP_EXT/venv}"
export COMP_UNI3D_SRC="${COMP_UNI3D_SRC:-$COMP_EXT/uni3d_src}"
export COMP_UNI3D_CKPT="${COMP_UNI3D_CKPT:-$COMP_EXT/weights_uni3d_g/model.pt}"
export COMP_OPENSHAPE_SRC="${COMP_OPENSHAPE_SRC:-$COMP_EXT/openshape_src}"
export COMP_OPENSHAPE_CKPT="${COMP_OPENSHAPE_CKPT:-$COMP_EXT/weights_openshape_pointbert_vitg14_rgb/model.pt}"

# Stesso dominio, stessi soggetti e stessi risultati di aau/runs/data_scale_ood/hifi/summary.md.
export COMP_VIEW_DIR="${COMP_VIEW_DIR:-$WBES_ROOT/datasets/HIFI3D/eval_view/npz}"
export COMP_ZS_RUNS="${COMP_ZS_RUNS:-$AAU_RUNS/ws_hifi3d/data_328f2bfc1a}"
export COMP_OUT="${COMP_OUT:-$AAU_RUNS/competitors_hifi3d}"
