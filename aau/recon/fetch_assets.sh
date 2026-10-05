#!/usr/bin/env bash
# Scarica e piazza gli asset che i tre repo di ricostruzione 3D NON includono (WS3b).
#
#   aau/recon/fetch_assets.sh          # frontend (git + internet) oppure nodo di calcolo
#   RECON_FORCE_FETCH=1 aau/recon/fetch_assets.sh
#
# Nessuno di questi file richiede registrazione o licenza firmata: sono link pubblici.
# Cosa manca a ciascun repo, e da dove si recupera:
#
#   3DDFA_V2   niente. Pesi (weights/*.pth), BFM ridotto (configs/bfm_noneck_v3.pkl) e
#              FaceBoxes sono gia' nel clone.
#   SynergyNet pesi + 3dmm_data. I pesi stanno su Google Drive: dei quattro link del
#              README ne risponde UNO SOLO, il re-upload di feb 2026 (gli altri danno
#              404, compreso quello di 3dmm_data). 3dmm_data si ricostruisce dal repo
#              3DDFA v1 dello stesso filone (cleardusk/3DDFA, MIT): i sei file che
#              utils/params.py carica stanno in train.configs/ e tri.mat in visualize/.
#   PRNet      solo il .data del checkpoint TF (160 MB, Google Drive); il .index e i
#              file uv-data sono nel clone.
#
# I download da Drive passano per la pagina "Virus scan warning": si prende il campo
# uuid dal form e si ripete la richiesta su drive.usercontent.google.com.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

EXTERNAL="$WBES_ROOT/external"
FORCE="${RECON_FORCE_FETCH:-0}"

# id Drive -> file di destinazione.
SYNERGY_WEIGHTS_ID=1RUOOWT1oSLOJYFpkWF_st2ZrVs8dpEcg   # zip: pretrained/{best,best_pose}.pth.tar
PRNET_WEIGHTS_ID=1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH     # 256_256_resfcn256_weight.data-00000-of-00001

gdrive_download() {
    # gdrive_download <file-id> <output>
    local id="$1" out="$2"
    local tmp cookie uuid
    tmp="$(mktemp -d)"
    cookie="$tmp/cookie.txt"
    curl -sL -c "$cookie" "https://docs.google.com/uc?export=download&id=$id" -o "$tmp/page.html"
    uuid="$(grep -o 'name="uuid" value="[^"]*"' "$tmp/page.html" | sed 's/.*value="//;s/"//')"
    if [[ -z "$uuid" ]]; then
        echo "ERRORE: Drive non ha dato il form di conferma per l'id $id" >&2
        echo "  (link scaduto o file rimosso: la pagina scaricata e' in $tmp/page.html)" >&2
        return 1
    fi
    curl -fL -b "$cookie" \
        "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t&uuid=$uuid" \
        -o "$out"
    rm -rf "$tmp"
}

echo "[fetch] EXTERNAL=$EXTERNAL"

# --- SynergyNet: pesi -------------------------------------------------------
syn_w="$EXTERNAL/SynergyNet/pretrained/best.pth.tar"
if [[ "$FORCE" == "1" || ! -f "$syn_w" ]]; then
    echo "[fetch] SynergyNet: pesi da Google Drive ($SYNERGY_WEIGHTS_ID)"
    tmp_zip="$EXTERNAL/SynergyNet/pretrained/_weights.zip"
    gdrive_download "$SYNERGY_WEIGHTS_ID" "$tmp_zip"
    # -j no: lo zip contiene gia' pretrained/, si estrae sulla radice del clone.
    python3 -c "
import zipfile, sys
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
assert names == ['pretrained/', 'pretrained/best.pth.tar', 'pretrained/best_pose.pth.tar'], names
z.extractall(sys.argv[2])
" "$tmp_zip" "$EXTERNAL/SynergyNet"
    rm -f "$tmp_zip"
else
    echo "[fetch] SynergyNet: pesi gia' presenti"
fi

# --- SynergyNet: 3dmm_data dal repo 3DDFA v1 --------------------------------
syn_data="$EXTERNAL/SynergyNet/3dmm_data"
if [[ "$FORCE" == "1" || ! -f "$syn_data/w_shp_sim.npy" ]]; then
    ddfa1="$EXTERNAL/3DDFA_v1"
    if [[ ! -d "$ddfa1/.git" ]]; then
        echo "[fetch] clono cleardusk/3DDFA (v1) per i file 3dmm_data"
        git clone --depth 1 https://github.com/cleardusk/3DDFA.git "$ddfa1"
    fi
    mkdir -p "$syn_data"
    # I sei file che utils/params.py cerca, piu' tri.mat (le facce della mesh densa).
    for f in keypoints_sim.npy w_shp_sim.npy w_exp_sim.npy param_whitening.pkl u_shp.npy u_exp.npy; do
        cp -f "$ddfa1/train.configs/$f" "$syn_data/$f"
    done
    cp -f "$ddfa1/visualize/tri.mat" "$syn_data/tri.mat"
    echo "[fetch] SynergyNet: 3dmm_data ricostruito da $ddfa1"
else
    echo "[fetch] SynergyNet: 3dmm_data gia' presente"
fi

# --- PRNet: il .data del checkpoint TF --------------------------------------
prn_w="$EXTERNAL/PRNet/Data/net-data/256_256_resfcn256_weight.data-00000-of-00001"
if [[ "$FORCE" == "1" || ! -f "$prn_w" ]]; then
    echo "[fetch] PRNet: checkpoint da Google Drive ($PRNET_WEIGHTS_ID)"
    gdrive_download "$PRNET_WEIGHTS_ID" "$prn_w"
else
    echo "[fetch] PRNet: checkpoint gia' presente"
fi

echo "[fetch] --- stato finale ---"
ls -l "$EXTERNAL/SynergyNet/pretrained/" "$EXTERNAL/SynergyNet/3dmm_data/" \
      "$EXTERNAL/PRNet/Data/net-data/" "$EXTERNAL/3DDFA_V2/weights/"
echo "[fetch] OK"
