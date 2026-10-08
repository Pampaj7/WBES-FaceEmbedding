#!/usr/bin/env bash
#SBATCH --job-name=wbes-devfs-chamfer
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#
# Set di sviluppo FaceScape: Chamfer su tutte le coppie delle 600 mesh valutate (mesh intera e
# regione stabile all'espressione), sulle due viste, neutra e con espressioni. E' il passo "region"
# di zs_expr_extra.sbatch (zs_region_chamfer.py, importato com'e'), lanciato qui perche' quello
# passerebbe --domain fv, cioe' gli assi di FaceVerse (alto -y, naso -z); FaceScape ha quelli di
# HIFI3D e ICT (alto +y, naso +z), quindi --domain hifi. La Chamfer intera non dipende dal frame.
#
#   aau/submit.sh zs3dmm/dev_fs_chamfer.sbatch
#
# Output: $ZS_RUNS[_expr]/data_<fp>/baselines/region_chamfer.npz, accanto ai bracci dei modelli,
# dove li legge dev_fs_summarize.py. E' la "Chamfer" del riconoscimento: quella degli script di
# eval (raw_chamfer delle pair_metrics) non ha le coppie stesso soggetto.
set -euo pipefail

export AAU_NV=""
cd "${WBES_ROOT:-$PWD}"
for expr in 0 1; do
    (
        export WBES_ZS_EXPR="$expr"
        source aau/zs3dmm/dev_facescape_env.sh
        source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
        zs_require_view
        out="$ZS_RUNS/data_$(zs_data_fp)/baselines"
        mkdir -p "$out"
        if [[ -f "$out/region_chamfer.npz" ]]; then
            echo "[devfs-chamfer] espressioni=$expr: gia' fatto ($out/region_chamfer.npz)"
            exit 0
        fi
        echo "[devfs-chamfer] espressioni=$expr vista=$ZS_DATA_DIR -> $out host=$(hostname) job=${SLURM_JOB_ID:-none}"
        "$AAU_DIR/run.sh" aau/zs3dmm/zs_region_chamfer.py --domain hifi --view-dir "$ZS_DATA_DIR" \
            --out "$out/region_chamfer.npz" --seed "${WBES_EVAL_SEED:-1234}" --workers "${SLURM_CPUS_PER_TASK:-8}"
    )
done
echo "[devfs-chamfer] OK"
