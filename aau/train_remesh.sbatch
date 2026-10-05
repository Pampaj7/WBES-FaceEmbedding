#!/usr/bin/env bash
#SBATCH --job-name=wbes-train
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili, e un
# --output relativo viene risolto nella submit dir (se non esiste, Slurm scarta il job in
# un secondo senza scrivere niente). Se sposti il repo, aggiorna queste tre righe oppure
# sottometti con aau/submit.sh, che le ricalcola e le passa da riga di comando.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#
# Porting della ricetta del modello top (v2_work/train_flame_v1config.sh, che a sua volta
# copia dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/config.json) sul dataset
# REMESH. Gli iperparametri sono identici alla v1: qualunque differenza nel risultato e'
# attribuibile ai dati, non alla ricetta.
#
#   aau/submit.sh train_remesh.sbatch
#   WBES_STAGE_TMP=1 aau/submit.sh train_remesh.sbatch
#
# Variabili riconosciute: WBES_DATA_DIR, WBES_DIST_NPZ, WBES_RUNS_ROOT, WBES_RUN_NAME,
# WBES_EPOCHS, WBES_PRELOAD_WORKERS, WBES_STAGE_TMP, WBES_SEED.
#
# --time=72h, stima conservativa: la run v1 con questo stesso fingerprint ha misurato
# ~11.5 min/epoca sul solo loop (80 iterazioni a ~8.6 s), cioe' ~23.3 h per 120 epoche, a
# cui vanno aggiunte ~60 eval online (--eval_every 2) di durata non misurata. MaxWall su
# prioritized/QoS normal e' 6 giorni, quindi non c'e' motivo di stringere finche' il dato
# vero non c'e'. Dopo la prima run reale su L40S, tara questo numero con `seff <jobid>`.
# Attenzione: train_runner.py NON ha resume, se il job muore per walltime si riparte da zero.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net
aau_require_dataset

# Il fingerprint della run dir NON include i percorsi dei dati ne' il job id: due
# sottomissioni della stessa config scriverebbero negli stessi checkpoint. Il job id
# nel runs_root le tiene separate.
RUN_NAME="${WBES_RUN_NAME:-remesh_v1recipe}"
RUNS_ROOT="${WBES_RUNS_ROOT:-$AAU_DIR/runs/${RUN_NAME}_${SLURM_JOB_ID:-manual}}"
mkdir -p "$RUNS_ROOT"

echo "[train] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[train] data_dir=$WBES_DATA_DIR"
echo "[train] dist_npz=$WBES_DIST_NPZ"
echo "[train] runs_root=$RUNS_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- staging opzionale su /tmp (WBES_STAGE_TMP=1) ---------------------------
# dataset_gtready.py fa un np.load per ogni __getitem__ e non usa worker di DataLoader:
# tutto l'I/O passa da un solo processo, e CephFS mono-processo fa 69 MB/s. Copiare gli
# npz su /tmp (4.2 GB/s in lettura) toglie di mezzo il problema.
# /tmp e' RAM: quello che ci copi CONTA CONTRO --mem. Qui teniamo 32 GB di margine per
# il training vero e proprio e abortiamo se il dataset non ci sta.
DATA_ARG="$WBES_DATA_DIR"
DIST_ARG="$WBES_DIST_NPZ"

if [[ "${WBES_STAGE_TMP:-0}" == "1" ]]; then
    STAGE_DIR="/tmp/${SLURM_JOB_ID:-manual}"
    trap 'rm -rf "$STAGE_DIR"' EXIT

    mem_mb="${SLURM_MEM_PER_NODE:-0}"
    if (( mem_mb <= 0 )); then
        # --mem=0 significa "tutta la RAM del nodo": non sappiamo quanto possiamo
        # prenderci per /tmp, quindi non azzardiamo. Non e' un errore: si continua da
        # CephFS, che e' lento ma sempre meglio di un OOM a meta' training.
        echo "[train] ATTENZIONE: SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-<vuoto>}, nessun budget calcolabile"
        echo "[train] staging saltato, il dataset si legge da CephFS"
    else
        budget_mb=$(( mem_mb - 32 * 1024 ))
        need_mb=$(( $(du -sm "$WBES_DATA_DIR" | cut -f1) + $(du -sm "$WBES_DIST_NPZ" | cut -f1) ))
        echo "[train] staging: servono ${need_mb} MB, budget ${budget_mb} MB (--mem ${mem_mb} MB meno 32 GB)"
        if (( need_mb > budget_mb )); then
            echo "ERRORE: il dataset non ci sta in /tmp senza rischiare l'OOM." >&2
            echo "  Rilancia con --mem=$(( (need_mb + 32 * 1024 + 1023) / 1024 ))G, oppure senza WBES_STAGE_TMP." >&2
            exit 1
        fi

        mkdir -p "$STAGE_DIR/data"
        echo "[train] copio gli npz su $STAGE_DIR/data"
        find "$WBES_DATA_DIR" -maxdepth 1 -type f -name '*.npz' -print0 \
            | xargs -0 -r -P 8 -I{} cp {} "$STAGE_DIR/data/"
        cp "$WBES_DIST_NPZ" "$STAGE_DIR/$(basename "$WBES_DIST_NPZ")"

        DATA_ARG="$STAGE_DIR/data"
        DIST_ARG="$STAGE_DIR/$(basename "$WBES_DIST_NPZ")"
        echo "[train] staging fatto: $(ls -1 "$DATA_ARG" | wc -l) file, $(du -sh "$STAGE_DIR" | cut -f1)"
    fi
fi

# Iperparametri copiati verbatim dalla ricetta v1, fingerprint incluso. Unico parametro
# della ricetta reso variabile: WBES_SEED (default 1234, il valore della v1), che serve per
# le repliche a piu' semi. Il seme entra sia nel NOME della run dir (`...seed<N>...`) sia
# nel fingerprint, quindi due semi diversi non si sovrascrivono mai i checkpoint.
# ATTENZIONE su --preload_eval_workers_train: entra nel fingerprint della run dir, quindi
# alzarlo (WBES_PRELOAD_WORKERS=8) cambia il nome della run e la rende non confrontabile
# a colpo d'occhio con la v1. Vale poco: il preload tocca 16 soggetti x 6 mesh una volta
# sola all'avvio, non e' li che CephFS fa male.
# Niente exec: il trap EXIT che ripulisce /tmp deve poter scattare.
"$AAU_DIR/run.sh" \
  face_embedding/gt_encdec/remeshing/intrinsic/train_twotower_dn_spec_robust.py \
  --data_dir "$DATA_ARG" \
  --dist_npz "$DIST_ARG" \
  --runs_root "$RUNS_ROOT" \
  --device cuda \
  --model xyz_dn \
  --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1 \
  --pool_mode meanmax --k_spec 0 --no-log_spec --eig_k 300 \
  --epochs "${WBES_EPOCHS:-120}" --batch_subjects 5 --lr 1e-4 --weight_decay 1e-6 --grad_clip 1.0 \
  --train_level mixed --train_pair_mode cross_topology \
  --lambda_subject 1.0 --lambda_mesh 1.0 --lambda_rank 0.5 \
  --use_id_loss --lambda_id 0.25 \
  --rank_margin 0.05 --rank_pairs 1024 --rank_tau 0.02 --rank_hard_frac 0.7 \
  --max_meshes_per_subject_train 6 --max_meshes_per_subject_eval 6 \
  --max_subjects_eval_train 16 \
  --preload_eval_workers_train "${WBES_PRELOAD_WORKERS:-2}" \
  --p_noise 0.6 --sigma_min 5e-4 --sigma_max 2e-2 \
  --noise_modes translation,rotation,jitter \
  --noise_mode_weights "translation=4,rotation=2,jitter=1" \
  --rigid_rot_deg 12.0 --rigid_rot_deg_min 0.5 \
  --rigid_trans_scale 0.03 --rigid_trans_scale_min 0.001 \
  --sigma_min_eval 1e-3 --sigma_max_eval 0.1 --n_sigma_eval 6 \
  --eval_mode average --eval_every 2 --save_every 5 --seed "${WBES_SEED:-1234}"
