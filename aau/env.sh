#!/usr/bin/env bash
# Sorgente unica di configurazione per il porting sul cluster AAU AI Cloud.
# Si sourcea sia sul frontend (senza singularity) sia dentro un job Slurm.
#
#   source aau/env.sh
#
# Ogni variabile e' sovrascrivibile dall'esterno (export prima del source).

_aau_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WBES_ROOT="${WBES_ROOT:-$(cd "${_aau_dir}/.." && pwd)}"
export AAU_DIR="${_aau_dir}"
export AAU_LOGS="${AAU_LOGS:-$AAU_DIR/logs}"
export AAU_RUNS="${AAU_RUNS:-$AAU_DIR/runs}"

# Container NGC: pytorch_24.10.sif = python 3.10.12 / torch 2.5.0a0+e000cf0ad9.nv24.10,
# che soddisfa python=3.10 e pytorch=2.5.* di environment.twotower_robust.yml.
export CONTAINER="${CONTAINER:-/home/container/pytorch/pytorch_24.10.sif}"

# Venv creato DENTRO il container con --system-site-packages: eredita torch/numpy/scipy
# del container e ci aggiunge solo cio' che manca (igl, potpourri3d, robust-laplacian).
export VENV="${VENV:-$WBES_ROOT/.venv_aau}"

# Pacchetto esterno diffusion_net (clonato sul frontend, vedi aau/README.md).
export WBES_DIFFUSION_NET_SRC="${WBES_DIFFUSION_NET_SRC:-$WBES_ROOT/diffusion-net/src}"

# Il codice del repo risolve diffusion_net da solo via path_setup.py, ma un
# `python3 -c "import diffusion_net"` no: lo mettiamo anche su PYTHONPATH.
case ":${PYTHONPATH:-}:" in
    *":$WBES_DIFFUSION_NET_SRC:"*) ;;
    *) export PYTHONPATH="$WBES_DIFFUSION_NET_SRC${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

# Dati: default ai percorsi attesi dal repo (robustness/paths.py).
export WBES_DATA_DIR="${WBES_DATA_DIR:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500_withops}"
export WBES_DIST_NPZ="${WBES_DIST_NPZ:-$WBES_ROOT/face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz}"

# Checkpoint del modello top incluso nel repo.
export WBES_CKPT="${WBES_CKPT:-$WBES_ROOT/face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth}"

# Flag GPU per singularity: su un nodo senza GPU (partizione cpu) va messo a vuoto.
export AAU_NV="${AAU_NV---nv}"

export PYTHONUNBUFFERED=1

unset _aau_dir

# --- preflight -------------------------------------------------------------
# Falliscono presto e con un messaggio leggibile, invece che in fondo a uno stack trace.

aau_require_diffusion_net() {
    if [[ ! -d "$WBES_DIFFUSION_NET_SRC/diffusion_net" ]]; then
        echo "ERRORE: diffusion-net non trovato in $WBES_DIFFUSION_NET_SRC" >&2
        echo "  Clonalo sul frontend (il nodo di calcolo non serve che abbia rete):" >&2
        echo "    git clone --depth 1 https://github.com/nmwsharp/diffusion-net.git \"$WBES_ROOT/diffusion-net\"" >&2
        return 1
    fi
}

aau_require_venv() {
    if [[ ! -x "$VENV/bin/python3" ]]; then
        echo "ERRORE: venv assente in $VENV" >&2
        echo "  Crealo con: sbatch aau/setup_env.sbatch" >&2
        return 1
    fi
}

aau_require_dataset() {
    local missing=0
    if [[ ! -d "$WBES_DATA_DIR" ]]; then
        echo "ERRORE: dataset non trovato in WBES_DATA_DIR=$WBES_DATA_DIR" >&2
        missing=1
    fi
    if [[ ! -f "$WBES_DIST_NPZ" ]]; then
        echo "ERRORE: matrice GT non trovata in WBES_DIST_NPZ=$WBES_DIST_NPZ" >&2
        missing=1
    fi
    if (( missing )); then
        echo "  Nessuno dei due e' incluso nel repo: vanno copiati a mano." >&2
        echo "  Attesi: datasets/REMESH/npz_data_topo_500_withops/ (npz con operatori)" >&2
        echo "          face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz" >&2
        echo "  Oppure esporta WBES_DATA_DIR / WBES_DIST_NPZ su percorsi alternativi." >&2
        return 1
    fi
}

aau_require_checkpoint() {
    if [[ ! -f "$WBES_CKPT" ]]; then
        echo "ERRORE: checkpoint non trovato in WBES_CKPT=$WBES_CKPT" >&2
        return 1
    fi
}
