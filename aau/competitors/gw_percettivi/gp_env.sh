#!/usr/bin/env bash
# Gromov-Wasserstein (POT) e feature percettive su render (DINOv2, CLIP, LPIPS, DreamSim). Si sourcea DOPO aau/env.sh:
#
#   source aau/env.sh; source aau/competitors/gw_percettivi/gp_env.sh
#
# Venv e pesi di terzi FUORI dal repo (mai in git): venv in $GP_EXT/venv (gp_setup.sbatch), pesi in $GP_WEIGHTS
# (cache di torch.hub, Hugging Face, open_clip, DreamSim).
export GP_EXT="${GP_EXT:-$HOME/ext/gwperc}"
export GP_VENV="${GP_VENV:-$GP_EXT/venv}"
export GP_WEIGHTS="${GP_WEIGHTS:-$HOME/ext/weights}"
export TORCH_HOME="${TORCH_HOME:-$GP_WEIGHTS/torch}"
export HF_HOME="${HF_HOME:-$GP_WEIGHTS/hf}"
export GP_DREAMSIM_CACHE="${GP_DREAMSIM_CACHE:-$GP_WEIGHTS/dreamsim}"
