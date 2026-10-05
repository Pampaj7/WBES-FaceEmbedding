#!/usr/bin/env bash
#SBATCH --job-name=wbes-x3dmm-train
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#
# Ricetta v1 verbatim (la stessa di aau/train_remesh.sbatch, iperparametro per iperparametro)
# sui dataset della tabella cross-3DMM di WS2: ICT da solo, oppure BFM+ICT insieme.
#
#   aau/submit.sh cross3dmm/train_x3dmm.sbatch --job-name=wbes-x3dmm-ict --mem=352G   # con WBES_* sotto
#
# L'entrypoint NON e' train_twotower_dn_spec_robust.py ma v2_work/train_v2/train_v2.py, che
# importa il trainer v1 senza modificarlo e gli installa tre patch per la durata della run
# (batch a dominio singolo, GT NaN-guarded, eval online su un dominio solo). Serve anche per
# ICT da solo, e non per il multi-dominio: il trainer legge la GT con SUBJECT_RE_4DIGIT mentre
# costruisce la subject map con SUBJECT_RE_ANY, e sugli id a 5 cifre di ICT (id10000) i due non
# coincidono ("id1000" contro "id10000") -- l'intersezione e' vuota e la run muore subito con
# "Need at least 6 overlapping subjects, found 0". train_v2 passa SUBJECT_RE_ANY anche alla GT.
#
# Variabili riconosciute: WBES_DATA_DIR, WBES_DIST_NPZ, WBES_RUNS_ROOT, WBES_RUN_NAME,
# WBES_EPOCHS, WBES_PRELOAD_WORKERS, WBES_STAGE_TMP, WBES_SEED, WBES_EVAL_DOMAIN.
#
# --time=48h. ATTENZIONE: il costo per epoca scala con il numero di soggetti di training, e
# ICT ne ha 4000 contro i 400 di BFM (10x le iterazioni per epoca). train_runner.py non ha
# resume: se il walltime scade si riparte da zero. Misura s/epoca dal log prima di fidarti.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net
aau_require_dataset

RUN_NAME="${WBES_RUN_NAME:-x3dmm}"
RUNS_ROOT="${WBES_RUNS_ROOT:-$AAU_DIR/runs/${RUN_NAME}_${SLURM_JOB_ID:-manual}}"
mkdir -p "$RUNS_ROOT"

echo "[x3dmm] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[x3dmm] data_dir=$WBES_DATA_DIR"
echo "[x3dmm] dist_npz=$WBES_DIST_NPZ"
echo "[x3dmm] runs_root=$RUNS_ROOT"
echo "[x3dmm] eval_domain=${WBES_EVAL_DOMAIN:-<dominio piu numeroso>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- staging opzionale su /tmp (WBES_STAGE_TMP=1) ---------------------------
# Come in aau/train_remesh.sbatch, ma con -L ovunque: qui la data_dir e' fatta di SYMLINK
# (le viste di datasets/ICT/train_ready e datasets/JOINT_BFM_ICT), e senza -L `du` misura i
# link invece dei target e `find -type f` non ne trova nessuno -- lo staging passerebbe il
# controllo di budget e poi copierebbe zero file.
# Vincoli: /tmp e' RAM e conta contro --mem, ed e' un tmpfs con una dimensione propria
# (378 GB sui nodi L40S da 755 GB, misurato). Servono entrambi i margini.
DATA_ARG="$WBES_DATA_DIR"
DIST_ARG="$WBES_DIST_NPZ"

if [[ "${WBES_STAGE_TMP:-0}" == "1" ]]; then
    STAGE_DIR="/tmp/${SLURM_JOB_ID:-manual}"
    trap 'rm -rf "$STAGE_DIR"' EXIT

    mem_mb="${SLURM_MEM_PER_NODE:-0}"
    if (( mem_mb <= 0 )); then
        echo "[x3dmm] ATTENZIONE: SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-<vuoto>}, nessun budget calcolabile"
        echo "[x3dmm] staging saltato, il dataset si legge da CephFS"
    else
        budget_mb=$(( mem_mb - 32 * 1024 ))
        need_mb=$(( $(du -sLm "$WBES_DATA_DIR" | cut -f1) + $(du -sLm "$WBES_DIST_NPZ" | cut -f1) ))
        tmpfs_mb=$(df -Pm /tmp | awk 'NR==2 {print $4}')
        echo "[x3dmm] staging: servono ${need_mb} MB, budget ${budget_mb} MB (--mem ${mem_mb} MB meno 32 GB), /tmp libero ${tmpfs_mb} MB"
        if (( need_mb > budget_mb )); then
            echo "ERRORE: il dataset non ci sta in /tmp senza rischiare l'OOM." >&2
            echo "  Rilancia con --mem=$(( (need_mb + 32 * 1024 + 1023) / 1024 ))G, oppure senza WBES_STAGE_TMP." >&2
            exit 1
        fi
        if (( need_mb > tmpfs_mb )); then
            echo "ERRORE: /tmp e' un tmpfs da ${tmpfs_mb} MB liberi, non bastano per ${need_mb} MB." >&2
            echo "  Rilancia senza WBES_STAGE_TMP: si legge da CephFS." >&2
            exit 1
        fi

        mkdir -p "$STAGE_DIR/data"
        echo "[x3dmm] copio gli npz su $STAGE_DIR/data"
        t0=$SECONDS
        find -L "$WBES_DATA_DIR" -maxdepth 1 -type f -name '*.npz' -print0 \
            | xargs -0 -r -P 8 -I{} cp -L {} "$STAGE_DIR/data/"
        cp -L "$WBES_DIST_NPZ" "$STAGE_DIR/$(basename "$WBES_DIST_NPZ")"

        DATA_ARG="$STAGE_DIR/data"
        DIST_ARG="$STAGE_DIR/$(basename "$WBES_DIST_NPZ")"
        echo "[x3dmm] staging fatto in $(( SECONDS - t0 )) s: $(find "$DATA_ARG" -maxdepth 1 -name '*.npz' | wc -l) file, $(du -sh "$STAGE_DIR" | cut -f1)"
    fi
fi

# Iperparametri identici a aau/train_remesh.sbatch: qualunque differenza nel risultato e'
# attribuibile ai dati (BFM / ICT / BFM+ICT), non alla ricetta. Il fingerprint della run dir
# non include i percorsi dei dati, quindi a parita' di seme le tre run hanno lo STESSO nome di
# run dir (`...seed1234__9a81466d`): a tenerle separate e' il job id dentro runs_root.
# Niente exec: il trap EXIT che ripulisce /tmp deve poter scattare.
"$AAU_DIR/run.sh" \
  v2_work/train_v2/train_v2.py \
  --eval_domain "${WBES_EVAL_DOMAIN:-}" \
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
