#!/usr/bin/env bash
#SBATCH --job-name=wbes-train-jointE
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=20:00:00
#
# E congiunto BFM+ICT contro controllo congiunto, stesso sottoinsieme (BFM 500 + ICT 1500, vedi
# aau/cross3dmm/joint_E_prep.py), ricetta v1 IDENTICA a aau/models/train_ablation.sbatch (cache in RAM
# di train_fast.py, 120 epoche, seed 1234) piu' le patch di train_v2 (batch a dominio singolo, eval
# online su BFM), attraverso aau/cross3dmm/train_joint_E.py:
#
#   WBES_JE_ARM=ctrl  --frame current  operatori del congiunto attuale          niente token
#   WBES_JE_ARM=e     --frame rms      robusti ad area 1 per BFM e ICT           token per dominio
#
#   WBES_JE_ARM=ctrl sbatch ... aau/cross3dmm/train_joint_E.sbatch
#   WBES_JE_ARM=e    sbatch ... --cpus-per-task=48 --mem=360G aau/cross3dmm/train_joint_E.sbatch
#
# Braccio e: gli operatori robusti ICT dei 1500 soggetti non stanno nella quota CephFS (piena), quindi
# si calcolano qui su /tmp (robust_area1_operators.py, un processo per core, 8016 npz in 1381 s con 32
# core nel job 1055578) e la vista del training vive in /tmp. /tmp e' RAM e conta contro --mem (53 GB).
#
# Run dir e log su /tmp, non su CephFS: il 5 ottobre alle 11:39 la quota della home si e' saturata e
# ogni job che ci scriveva e' morto senza messaggio alla prima scrittura (1055577 e 1055578 a epoca
# 9 e 6, tqdm su stderr). Ogni 10 minuti la run dir e il log senza barre si copiano in $RUNS_ROOT con
# un sync che non puo' far fallire il job; a fine job il sync si ritenta per 30 minuti. Nella home
# finiscono solo l'ultimo checkpoint e i tre best_by_* (--save_every = epoche, ~32 MB) e i log.
#
# --mem: la cache stima ~19.5 MB a campione BFM (log 1055114) e ~12.5 MB a campione ICT (rapporto
# delle dimensioni degli npz), cioe' ~170 GB per 12000 campioni; l'RSS misurato dei training con
# cache e' ~0.65 volte la stima (MaxRSS 37-53 GB per 57-80 GB stimati), piu' il margine che ha
# evitato il thrashing del 5 ottobre (100G -> 180G per 57 GB stimati). 300G (ctrl), 360G (e).
# --time: ~118 s/epoca su L40S per 400 soggetti BFM (abl_F 1055162), qui 400 BFM + 1200 ICT con mesh
# ICT ~0.64 volte piu' leggere: ~300 s/epoca, ~10-11 h per 120 epoche, piu' cache e staging.
# train_runner.py non ha resume: 20 h di tetto.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net

ARM="${WBES_JE_ARM:?WBES_JE_ARM=ctrl|e}"
SEED=1234
SPLITS="$AAU_DIR/runs/joint_E/splits.json"
[[ -f "$SPLITS" ]] || { echo "ERRORE: $SPLITS assente (joint_E_prep.py build)" >&2; exit 1; }
DIST="$WBES_ROOT/datasets/JOINT_BFM_ICT/gt_matrix.npz"
RUN_NAME="${WBES_RUN_NAME:-joint_E_${ARM}_s${SEED}}"
RUNS_ROOT="${WBES_RUNS_ROOT:-$AAU_DIR/runs/joint_E/${RUN_NAME}_${SLURM_JOB_ID:-manual}}"
EPOCHS="${WBES_EPOCHS:-120}"
mkdir -p "$RUNS_ROOT"
STAGE="/tmp/${SLURM_JOB_ID:-manual}"
RUN_TMP="$STAGE/run"
LOG_TMP="$STAGE/train.log"
mkdir -p "$RUN_TMP"

# Copia run dir e log filtrato nella home. Ritorna lo stato di rsync, ma chi la chiama in background
# lo ignora: una quota piena non deve fermare il training. Il filtro toglie le barre tqdm (stdout e
# stderr sono nello stesso file, quindi "Epoch NNN |" puo' finire attaccata in coda a una barra).
sync_home() {
    [[ -f "$LOG_TMP" ]] && { tr '\r' '\n' < "$LOG_TMP" | sed 's/\(Epoch [0-9]\{3\} |\)/\n\1/g' \
        | grep -v '%|.*\(it/s\|s/it\)\|^[[:space:]]*$' > "$RUNS_ROOT/train.log.part" \
        && mv -f "$RUNS_ROOT/train.log.part" "$RUNS_ROOT/train.log"; } 2> /dev/null
    rsync -a "$RUN_TMP/" "$RUNS_ROOT/" 2> /dev/null
}
final_sync() {
    local i
    for (( i = 0; i < 30; i++ )); do
        sync_home && { echo "[jointE] run dir copiata in $RUNS_ROOT" || true; return 0; }
        sleep 60
    done
    echo "ERRORE: sync finale verso $RUNS_ROOT fallito per 30 minuti" >&2 || true
    return 1
}
SYNC_PID=""
cleanup() {
    [[ -n "$SYNC_PID" ]] && kill "$SYNC_PID" 2> /dev/null
    final_sync || true
    rm -rf "$STAGE"
}
trap cleanup EXIT

echo "[jointE] host=$(hostname) job=${SLURM_JOB_ID:-none} arm=$ARM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ARM_ARGS=()
case "$ARM" in
    ctrl)
        FRAME=current
        DATA_DIR="$WBES_ROOT/datasets/JOINT_E/ctrl/npz_withops"
        ;;
    e)
        FRAME=rms
        AAU_NV="" "$AAU_DIR/run.sh" aau/cross3dmm/joint_E_prep.py stage-input --dest "$STAGE" --what train
        PROCS="${SLURM_CPUS_PER_TASK:-8}"
        t0=$SECONDS
        pids=()
        for (( p = 0; p < PROCS; p++ )); do
            OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AAU_NV="" \
            "$AAU_DIR/run.sh" aau/models/robust_area1_operators.py \
                --input-dir "$STAGE/ict_in" --output-dir "$STAGE/ict_ra1" --k-eig 128 --shard "$p/$PROCS" \
                > "$STAGE/ra1_$p.log" 2>&1 &
            pids+=($!)
        done
        for i in "${!pids[@]}"; do
            wait "${pids[$i]}" || { echo "ERRORE: operatori robusti ICT, shard $i" >&2; tail -5 "$STAGE/ra1_$i.log" >&2; exit 1; }
        done
        echo "[jointE] operatori robusti ICT: $(find "$STAGE/ict_ra1" -maxdepth 1 -name 'id*.npz' | wc -l) npz" \
             "($(find "$STAGE/ict_in" -maxdepth 1 -name 'id*.npz' | wc -l) calcolati in $(( SECONDS - t0 )) s)," \
             "$(du -shL "$STAGE/ict_ra1" | cut -f1)"
        AAU_NV="" "$AAU_DIR/run.sh" aau/cross3dmm/joint_E_prep.py stage-view --dest "$STAGE" --what train
        DATA_DIR="$STAGE/view"
        ARM_ARGS=(--size-token-json-bfm "$AAU_DIR/runs/joint_E/size_token_joint_bfm_s${SEED}.json"
                  --size-token-json-ict "$AAU_DIR/runs/joint_E/size_token_joint_ict_s${SEED}.json")
        ;;
    *) echo "ERRORE: WBES_JE_ARM=$ARM, attesi ctrl|e" >&2; exit 2 ;;
esac
N_DATA=$(find "$DATA_DIR" -maxdepth 1 -name 'id*.npz' | wc -l)
[[ "$N_DATA" == 12000 ]] || { echo "ERRORE: $DATA_DIR ha $N_DATA npz, attesi 12000" >&2; exit 1; }
echo "[jointE] frame=$FRAME data_dir=$DATA_DIR ($N_DATA npz) dist=$DIST runs_root=$RUNS_ROOT"

CMD=(
  aau/cross3dmm/train_joint_E.py --eval_domain bfm "${ARM_ARGS[@]}"
  --cache-residency ram --cache-workers "${WBES_CACHE_WORKERS:-8}" --frame "$FRAME"
  --data_dir "$DATA_DIR"
  --dist_npz "$DIST"
  --runs_root "$RUN_TMP"
  --device cuda
  --model xyz_dn
  --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1
  --pool_mode meanmax --k_spec 0 --no-log_spec --eig_k 300
  --epochs "$EPOCHS" --batch_subjects 5 --lr 1e-4 --weight_decay 1e-6 --grad_clip 1.0
  --train_level mixed --train_pair_mode cross_topology
  --lambda_subject 1.0 --lambda_mesh 1.0 --lambda_rank 0.5
  --use_id_loss --lambda_id 0.25
  --rank_margin 0.05 --rank_pairs 1024 --rank_tau 0.02 --rank_hard_frac 0.7
  --max_meshes_per_subject_train 6 --max_meshes_per_subject_eval 6
  --max_subjects_eval_train 16
  --preload_eval_workers_train "${WBES_PRELOAD_WORKERS:-2}"
  --p_noise 0.6 --sigma_min 5e-4 --sigma_max 2e-2
  --noise_modes translation,rotation,jitter
  --noise_mode_weights "translation=4,rotation=2,jitter=1"
  --rigid_rot_deg 12.0 --rigid_rot_deg_min 0.5
  --rigid_trans_scale 0.03 --rigid_trans_scale_min 0.001
  --sigma_min_eval 1e-3 --sigma_max_eval 0.1 --n_sigma_eval 6
  --eval_mode average --eval_every 2 --save_every "$EPOCHS" --seed "$SEED"
)
printf '%q ' "${CMD[@]}" > "$RUNS_ROOT/wrapper_launch.txt"
echo >> "$RUNS_ROOT/wrapper_launch.txt"
printf 'arm=%s\nframe=%s\ndata_dir=%s\nsize_token_json=%s\n' \
    "$ARM" "$FRAME" "$DATA_DIR" "${ARM_ARGS[*]:-}" > "$RUNS_ROOT/joint_E_arm.txt"

( while sleep 600; do sync_home || true; done ) &
SYNC_PID=$!
# Niente exec: il trap EXIT che copia la run dir e ripulisce /tmp deve poter scattare.
"$AAU_DIR/run.sh" "${CMD[@]}" > "$LOG_TMP" 2>&1
kill "$SYNC_PID" 2> /dev/null || true
SYNC_PID=""
final_sync
tail -4 "$RUNS_ROOT/train.log"

AAU_NV="" "$AAU_DIR/run.sh" aau/cross3dmm/joint_E_prep.py verify --run-root "$RUNS_ROOT"
echo "[jointE] OK"
