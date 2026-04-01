#!/usr/bin/env bash

_wbes_activate_twotower_robust_env() {
    local script_dir repo_root venv_path diffusion_src
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/.." && pwd)"
    venv_path="${repo_root}/.venv_twotower_robust_312"
    diffusion_src="${repo_root}/diffusion-net/src"

    if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
        echo "Use: source ${repo_root}/scripts/activate_twotower_robust_env.sh"
        return 1
    fi

    if [[ ! -x "${venv_path}/bin/python" ]]; then
        echo "Missing virtualenv python: ${venv_path}/bin/python"
        return 1
    fi

    if [[ ! -d "${diffusion_src}/diffusion_net" ]]; then
        echo "Missing diffusion-net source: ${diffusion_src}"
        return 1
    fi

    export VIRTUAL_ENV="${venv_path}"
    export PATH="${venv_path}/bin:${PATH}"
    unset PYTHONHOME
    export WBES_DIFFUSION_NET_SRC="${diffusion_src}"

    echo "Activated ${venv_path}"
    echo "WBES_DIFFUSION_NET_SRC=${WBES_DIFFUSION_NET_SRC}"
    python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"
}

_wbes_activate_twotower_robust_env
