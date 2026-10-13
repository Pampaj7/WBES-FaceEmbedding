#!/usr/bin/env bash
# Ava-256, prova confermativa (v7): worktree di git al commit congelato, da cui si sottomettono i job della catena.
#
#   aau/ava256/conf_worktree.sh <commit congelato> <cartella nuova>
#   cd <cartella> && CONF_FROZEN_COMMIT=<commit> CONF_SPEC=aau/ava256/spec/ava256.json CONF_STEPS="..." \
#       sbatch --chdir="$PWD" aau/ava256/conf.sbatch
#
# Il codice viene SOLO dal commit (checkout pulito, staccato). I dati non tracciati o ignorati del repo principale
# (datasets/, aau/runs/ con checkpoint, matrici e uscite, venv, diffusion-net, cloni esterni) entrano come link
# simbolici: aau/runs/ e datasets/ sono fuori dallo sparse-checkout (i loro file tracciati non sono codice), ogni
# altra voce non tracciata o ignorata del repo principale e' un link al suo percorso, mai un file copiato. I moduli
# Python importati restano controllati uno per uno contro il commit (conf_run.py); conf_common.verify controlla HEAD
# e che nessun file tracciato sia modificato.
set -euo pipefail
COMMIT="${1:?commit congelato}"
DEST="${2:?cartella del worktree}"
MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -e "$DEST" ]] && { echo "ERRORE: $DEST esiste gia'" >&2; exit 1; }
git -C "$MAIN" worktree add --detach --no-checkout "$DEST" "$COMMIT"
cd "$DEST"
mkdir -p "$(git rev-parse --git-dir)/info"                    # git 2.25: la cartella info del worktree manca
git sparse-checkout set --no-cone '/*' '!/aau/runs/' '!/datasets/'
git checkout --detach "$COMMIT"
ln -s "$MAIN/aau/runs" aau/runs
ln -s "$MAIN/datasets" datasets
# voci non tracciate o ignorate del repo principale (cartelle intere o file), fuori da aau/runs e datasets
git -C "$MAIN" status --porcelain --ignored --untracked-files=normal | sed -n 's/^\(??\|!!\) //p' \
  | grep -v '^aau/runs/\|^datasets/\|__pycache__\|\.pyc$' | while read -r p; do
    p="${p%/}"
    [[ -e "$p" || -L "$p" ]] && continue
    mkdir -p "$(dirname "$p")"
    ln -s "$MAIN/$p" "$p"
  done
echo "[conf-worktree] $DEST al commit $(git rev-parse HEAD); modifiche tracciate: $(git status --porcelain --untracked-files=no | wc -l)"
