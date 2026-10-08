#!/usr/bin/env bash
# Funzioni comuni a E2 ed E3b, da sourceare DOPO aau/env.sh dentro un job. Sono i passi b. e c. di
# aau/zs3dmm/zs_zeroshot.sbatch con WBES_ZS_PART=embed, ricopiati (lo sbatch non si tocca):
#   ev_ops   <in_dir> <ops_dir> <tag>     operatori ad area unitaria (k_eig 128) di
#                                         v2_work/potential/areanorm_operators.py, un processo per core
#   ev_embed <ops_dir> <gt_npz> <out_dir> zs_embed.py coi common_args di eval_common.sh
#                                         (--subject_split all --max_subjects 0, WBES_EVAL_SEED=1234),
#                                         embedding di ogni mesh in <out_dir>/embeddings.npz
# Il checkpoint e' E108_CKPT (o WBES_E108_CKPT). Gira su GPU (job con --gres) o su CPU (partizione cpu,
# AAU_NV vuota): gli script scelgono cuda solo se c'e'.

E108_CKPT="${WBES_E108_CKPT:-$AAU_RUNS/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411/mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__3167d36d/checkpoints/epoch108.pth}"

ev_ops() {
    local in_dir="$1" ops_dir="$2" tag="$3"
    local n_shards="${SLURM_CPUS_PER_TASK:-8}" pids=() i
    (( n_shards % 6 == 0 )) && n_shards=$(( n_shards - 1 ))   # coprimo con 6 topologie, come zs_zeroshot
    local logs="$AAU_LOGS/${SLURM_JOB_NAME:-wbes-ev}-${SLURM_JOB_ID:-manual}.shards"
    mkdir -p "$logs" "$ops_dir"
    local t0=$SECONDS
    for (( i = 0; i < n_shards; i++ )); do
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
          --input-dir "$in_dir" --output-dir "$ops_dir" --k-eig 128 --shard "$i/$n_shards" \
          > "$logs/$tag-shard$i.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "ERRORE: shard $i di $tag (vedi $logs/$tag-shard$i.log)" >&2; return 1; }
    done
    local n_in n_ops
    n_in=$(find "$in_dir" -maxdepth 1 -name "*.npz" | wc -l)
    n_ops=$(find "$ops_dir" -maxdepth 1 -name "*.npz" | wc -l)
    echo "[ev] operatori $tag: $n_ops / $n_in in $(( SECONDS - t0 )) s, $(du -sh "$ops_dir" | cut -f1) su /tmp"
    (( n_ops == n_in )) || { echo "ERRORE: operatori incompleti ($tag)" >&2; return 1; }
}

ev_embed() {
    local ops_dir="$1" gt="$2" out_dir="$3"
    shift 3
    (
        export WBES_DATA_DIR="$ops_dir" WBES_DIST_NPZ="$gt" WBES_CKPT="$E108_CKPT" WBES_EVAL_SEED=1234
        export WBES_EVAL_OUT="$out_dir"
        source "$WBES_ROOT/aau/eval_common.sh"
        common_args+=(--subject_split all --max_subjects 0)
        mkdir -p "$out_dir"
        printf 'ckpt=%s\ndata_dir=%s\ndist_npz=%s\neval_seed=1234\njob=%s\nhost=%s\n' \
            "$CKPT_REAL" "$DATA_REAL" "$DIST_REAL" "${SLURM_JOB_ID:-none}" "$(hostname)" > "$out_dir/eval_key.txt"
        # GPU se il job ne ha una (AAU_NV ereditato), altrimenti CPU: device() degli script ricade su cpu
        OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
        "$AAU_DIR/run.sh" "${@:-aau/zs3dmm/zs_embed.py}" "${common_args[@]}" \
            --embeddings_out "$out_dir/embeddings.npz" 2>&1 | grep -v "valid test operator" | tee "$out_dir/embed.log"
    )
}
