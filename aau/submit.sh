#!/usr/bin/env bash
# Sottomette uno degli sbatch di aau/ da qualunque directory.
#
#   aau/submit.sh train_remesh.sbatch
#   ~/WBES-FaceEmbedding/aau/submit.sh eval_sigma.sbatch --time=24:00:00
#
# Serve perche' le direttive #SBATCH sono statiche: --output/--error/--chdir nei file
# valgono per il checkout in $WBES_ROOT, e se qualcuno sposta il repo o sottomette da
# un'altra root Slurm scarta il job in un secondo senza scrivere nessun log.
# Qui i tre path vengono ricalcolati e passati da riga di comando, che ha la precedenza.
# Le opzioni extra vanno dopo il nome dello script e sovrascrivono anche queste.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if [[ $# -lt 1 ]]; then
    echo "uso: aau/submit.sh <script.sbatch> [opzioni sbatch...]" >&2
    exit 2
fi

script="$1"
shift

# Accetta sia un percorso vero sia il solo nome del file dentro aau/.
if [[ ! -f "$script" ]]; then
    for candidate in "$WBES_ROOT/$script" "$AAU_DIR/$script" "$AAU_DIR/$(basename "$script")"; do
        if [[ -f "$candidate" ]]; then
            script="$candidate"
            break
        fi
    done
fi
if [[ ! -f "$script" ]]; then
    echo "ERRORE: script sbatch non trovato: $script" >&2
    exit 2
fi

# Le dir di log e di output devono esistere PRIMA della sottomissione: Slurm non le crea.
mkdir -p "$AAU_LOGS" "$AAU_RUNS"

exec sbatch \
    --chdir="$WBES_ROOT" \
    --output="$AAU_LOGS/%x-%j.out" \
    --error="$AAU_LOGS/%x-%j.err" \
    "$@" \
    "$script"
