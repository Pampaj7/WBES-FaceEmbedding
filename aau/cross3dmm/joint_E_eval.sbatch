#!/usr/bin/env bash
#SBATCH --job-name=wbes-jointE-eval
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Eval di un braccio del confronto E congiunto / controllo congiunto su un dominio, sui soggetti
# held-out dello split ricostruito (aau/runs/joint_E/splits.json): BFM 104, ICT 100.
#
#   WBES_CKPT=aau/runs/joint_E/joint_E_e_s1234_<job> WBES_JE_DOMAIN=ict sbatch ... aau/cross3dmm/joint_E_eval.sbatch
#
#   0. joint_E_prep.py verify: seed ed eval online del run = split ricostruito, nessun soggetto di eval
#      nel training (lo split e' sull'unione dei due domini, come train_v2);
#   1. aau/models/eval_cells.py: Spearman per gruppo crop/noisy/resample/all (eval_by_topology) e 30 celle;
#   2. compare_model_vs_chamfer_topology_breakdown.py (repo), clean, coppie ordinate cross-topologia,
#      con pair_metrics.csv: protocollo mesh-pair di WS2 e Chamfer per cella (margine del criterio).
# Entrambi lanciati da aau/models/eval_ablation.py con frame e token del braccio (joint_E_arm.txt).
# La vista di eval contiene SOLO i soggetti scelti, quindi --subject_split all (come ws2_eval.sbatch).
#
# Variabili PROPRIE per dati e GT: aau/submit.sh e env.sh esportano sempre i percorsi BFM (bug del
# 5 ottobre). Il dominio dei soggetti valutati viene controllato prima delle ore di GPU.
# Braccio e su ICT: gli operatori robusti dei 100 soggetti si calcolano su /tmp
# (quota CephFS piena), come in train_joint_E.sbatch.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

RUNS_ROOT="$(realpath "${WBES_CKPT:?WBES_CKPT = runs_root del braccio}")"
_m=("$RUNS_ROOT"/*/checkpoints/best_by_xtopo_mesh_clean.pth)
[[ ${#_m[@]} -eq 1 && -f "${_m[0]}" ]] || { echo "ERRORE: checkpoint in $RUNS_ROOT: ${_m[*]}" >&2; exit 2; }
CKPT="${_m[0]}"
DOM="${WBES_JE_DOMAIN:?WBES_JE_DOMAIN=bfm|ict}"
ARM="$(sed -n 's/^arm=//p' "$RUNS_ROOT/joint_E_arm.txt")"
FRAME="$(sed -n 's/^frame=//p' "$RUNS_ROOT/joint_E_arm.txt")"
case "$DOM" in
    bfm) GT="$WBES_ROOT/face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz" ;;
    ict) GT="$WBES_ROOT/datasets/ICT/train_ready/gt_matrix.npz" ;;
    *) echo "ERRORE: WBES_JE_DOMAIN=$DOM" >&2; exit 2 ;;
esac
LAUNCH=(aau/models/eval_ablation.py --frame "$FRAME")
[[ "$ARM" == e ]] && LAUNCH+=(--size-token-json "$AAU_DIR/runs/joint_E/size_token_joint_${DOM}_s1234.json")
LAUNCH+=(--)

if [[ "$ARM" == e && "$DOM" == ict ]]; then
    STAGE="/tmp/${SLURM_JOB_ID:-manual}"
    trap 'rm -rf "$STAGE"' EXIT
    mkdir -p "$STAGE"
    AAU_NV="" "$AAU_DIR/run.sh" aau/cross3dmm/joint_E_prep.py stage-input --dest "$STAGE" --what eval_ict
    PROCS="${SLURM_CPUS_PER_TASK:-8}"
    pids=()
    for (( p = 0; p < PROCS; p++ )); do
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AAU_NV="" \
        "$AAU_DIR/run.sh" aau/models/robust_area1_operators.py \
            --input-dir "$STAGE/ict_in" --output-dir "$STAGE/ict_ra1" --k-eig 128 --shard "$p/$PROCS" \
            > "$STAGE/ra1_$p.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "ERRORE: operatori robusti, shard $i" >&2; tail -5 "$STAGE/ra1_$i.log" >&2; exit 1; }
    done
    AAU_NV="" "$AAU_DIR/run.sh" aau/cross3dmm/joint_E_prep.py stage-view --dest "$STAGE" --what eval_ict
    VIEW="$STAGE/view"
else
    VIEW="$WBES_ROOT/datasets/JOINT_E/eval/${ARM}__${DOM}"
fi
OUT="$AAU_DIR/runs/joint_E/eval/$(basename "$RUNS_ROOT")__${DOM}"
mkdir -p "$OUT"

echo "[$(timestamp)] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[$(timestamp)] arm=$ARM frame=$FRAME dominio=$DOM ckpt=$CKPT"
echo "[$(timestamp)] view=$VIEW gt=$GT out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

AAU_NV="" "$AAU_DIR/run.sh" aau/cross3dmm/joint_E_prep.py verify --run-root "$RUNS_ROOT"
# Dominio dei soggetti della vista: tutti BFM (id < 10000) o tutti ICT (id >= 10000).
N_FILES=$(find "$VIEW" -maxdepth 1 -name 'id*.npz' | wc -l)
N_BAD=$(find "$VIEW" -maxdepth 1 -name 'id*.npz' -printf '%f\n' | sed 's/^id\([0-9]*\)_.*/\1/' \
        | awk -v d="$DOM" '{ if ((d == "bfm") != ($1 + 0 < 10000)) n++ } END { print n + 0 }')
echo "[$(timestamp)] vista: $N_FILES npz, $N_BAD fuori dal dominio $DOM"
[[ "$N_BAD" == 0 && "$N_FILES" -gt 0 ]] || { echo "ERRORE: vista non tutta $DOM" >&2; exit 1; }

echo "[$(timestamp)] 1/2 eval_cells (gruppi e 30 celle)"
"$AAU_DIR/run.sh" "${LAUNCH[@]}" aau/models/eval_cells.py \
  --checkpoint "$CKPT" --data-dir "$VIEW" --dist-npz "$GT" --subject-split all --n-subjects 0 \
  --device cuda --tag cells --out-dir "$OUT" 2>&1 | tee "$OUT/cells.log"

echo "[$(timestamp)] 2/2 breakdown per coppia ordinata di topologie (clean) + pair_metrics"
"$AAU_DIR/run.sh" "${LAUNCH[@]}" \
  face_embedding/gt_encdec/remeshing/intrinsic/perturbated/compare_model_vs_chamfer_topology_breakdown.py \
  --model_path "$CKPT" --checkpoint_selector best_by_clean \
  --data_dir "$VIEW" --dist_npz "$GT" \
  --subject_split all --max_subjects 0 --eval_fraction 0.2 \
  --max_meshes_per_subject_eval 10 --preload_workers 8 --pair_mode cross_topology \
  --rigid_rot_deg 20 --rigid_trans_scale 0.05 --rigid_rot_deg_min 1 --rigid_trans_scale_min 0.002 \
  --chamfer_batch_pairs 256 --chamfer_cache_verts force --chamfer_cache_verts_max_mb 4096 \
  --out_dir "$OUT/topology" --topology_pair_mode cross_only --ordered_topology_pairs --write_pair_metrics \
  2>&1 | tee "$OUT/topology.log"

printf 'completed=%s\njob=%s\nhost=%s\narm=%s\ndomain=%s\nckpt=%s\nview=%s\ngt=%s\n' "$(timestamp)" \
    "${SLURM_JOB_ID:-none}" "$(hostname)" "$ARM" "$DOM" "$CKPT" "$VIEW" "$GT" > "$OUT/.done"
echo "[$(timestamp)] OK, scritta $OUT/.done"
