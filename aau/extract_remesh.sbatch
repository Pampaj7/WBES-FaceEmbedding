#!/usr/bin/env bash
#SBATCH --job-name=wbes-extract
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili, e un
# --output relativo viene risolto nella submit dir (se non esiste, Slurm scarta il job in
# un secondo senza scrivere niente). Se sposti il repo, aggiorna queste tre righe oppure
# sottometti con aau/submit.sh, che le ricalcola e le passa da riga di comando.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#
# Spacchetta REMESH_npz_data_topo_500_noops.tar.zst (Hugging Face, Pampaj) in
# datasets/REMESH/npz_data_topo_500. Layout piatto, 3000 npz id<NNNN>_GTready_<topo>.npz,
# che e' quello che si aspettano precompute_operators_npz.py (Path.iterdir, non ricorsivo)
# e GTReadyDatasetNPZ (os.listdir). Job CPU: niente GPU, serve solo zstandard.
#
#   aau/submit.sh extract_remesh.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

ARCHIVE="${WBES_REMESH_ARCHIVE:-$WBES_ROOT/hf_downloads/REMESH_npz_data_topo_500_noops.tar.zst}"
DEST="${WBES_REMESH_NOOPS_DIR:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500}"

if [[ ! -f "$ARCHIVE" ]]; then
    echo "ERRORE: archivio non trovato: $ARCHIVE" >&2
    exit 1
fi

echo "[extract] archivio: $ARCHIVE"
echo "[extract] destinazione: $DEST"

# --include scarta .cache/huggingface/upload/, 4000 file di metadati dell'export HF.
"$AAU_DIR/run.sh" "$AAU_DIR/tar_zst.py" extract \
    "$ARCHIVE" "$DEST" \
    --strip 2 \
    --include 'REMESH/npz_data_topo_500/*.npz'

echo "[extract] file estratti: $(ls -1 "$DEST"/*.npz | wc -l)"
echo "[extract] per topologia:"
ls -1 "$DEST" | sed 's/.*_//; s/\.npz$//' | sort | uniq -c | sed 's/^/    /'
