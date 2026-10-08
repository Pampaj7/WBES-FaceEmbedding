#!/usr/bin/env bash
#SBATCH --job-name=wbes-ood-now
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#
# NoW sui checkpoint intermedi del run grande (segnale fuori dominio anticipato, aau/data_scale/PLAN.md):
# latenti del checkpoint sulle patch NoW e concordanza con NoW (tau per immagine sui 3 metodi
# pre-registrati), con la pipeline aau/recon/now_* invariata.
#
#   WBES_OOD_CKPTS="e036:<ckpt> e072:<ckpt> ..." aau/submit.sh data_scale/ood_now.sbatch
#
# Niente --overwrite nella cartella del congiunto: la cache dei latenti ha un nome fisso
# (embeddings/latent_joint_*.npz, gt_latent_joint_*.csv) e un modello nuovo la' sovrascriverebbe quelli
# del congiunto. Per ogni checkpoint una cartella di lavoro ~/data/now_eval_scale_<tag> e un'uscita
# aau/runs/now_eval_scale_<tag>: patch, operatori (gia' calcolati, dipendono solo dalle patch) e csv
# delle ALTRE metriche come symlink; embeddings/, pairs/ (solo le matrici non latenti collegate) e le
# uscite del riassunto come file veri, cosi' nessuna scrittura passa da un link verso l'originale.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
SRC_W="$HOME/data/now_eval_work"
SRC_O="$AAU_RUNS/now_eval"
read -ra PAIRS <<< "${WBES_OOD_CKPTS:?WBES_OOD_CKPTS}"
for pair in "${PAIRS[@]}"; do
  tag="${pair%%:*}"; ck="${pair#*:}"
  [[ -f "$ck" ]] || { echo "ERRORE: checkpoint $ck assente" >&2; exit 1; }
  W="$HOME/data/now_eval_scale_$tag"; O="$AAU_RUNS/now_eval_scale_$tag"
  mkdir -p "$W/embeddings" "$W/pairs" "$O"
  for x in arcface mica_assets official pred recon_face recon_face_withops_areanorm scan_face \
           scan_face_withops_areanorm template_lmk7.json; do
    ln -sfn "$SRC_W/$x" "$W/$x"
  done
  for f in "$SRC_W"/pairs/*.npz; do
    case "$(basename "$f")" in latent_*) ;; *) ln -sf "$f" "$W/pairs/" ;; esac
  done
  for f in "$SRC_O"/gt_arcface_* "$SRC_O"/gt_geometric_* "$SRC_O"/now_official_* "$SRC_O"/prep_* \
           "$SRC_O"/official_selfcheck.json "$SRC_O"/protocol.md; do
    ln -sf "$f" "$O/"
  done
  echo "[ood-now] $tag: ckpt=$ck work=$W out=$O"
  echo "ckpt=$ck" > "$O/checkpoint.txt"
  # operatori: le cartelle sono symlink a quelli gia' calcolati, che si saltano; latenti nuovi
  WBES_NOW_WORK="$W" WBES_NOW_OUT="$O" WBES_CKPT="$ck" bash aau/recon/now_ops_latent.sbatch
  WBES_NOW_WORK="$W" WBES_NOW_OUT="$O" AAU_NV= "$AAU_DIR/run.sh" aau/recon/now_summarize.py
  grep -E "latente congiunto|Chamfer grezza \(maxabs\)" "$O/summary.md" | head -2 || true
done
