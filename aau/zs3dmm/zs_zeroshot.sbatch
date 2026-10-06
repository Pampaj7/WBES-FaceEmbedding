#!/usr/bin/env bash
#SBATCH --job-name=wbes-zs-zeroshot
# --chdir e i log sono path ASSOLUTI: le direttive #SBATCH non espandono variabili. Sottometti
# con aau/submit.sh, che li ricalcola e li passa da riga di comando.
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#
# Zero-shot su un 3DMM mai visto (WBES_ZS_DOMAIN=hifi|fv|ict|bfm): un modello mai addestrato su quel
# dominio valutato sulle identita' di zs_build.sbatch. Protocollo IDENTICO a
# aau/ict/ict_zeroshot_rank.sbatch (stessi due script, stessi common_args di eval_common.sh,
# stesse sigma):
#   1. compare_model_vs_chamfer_rankings.py           -> Spearman aggregato, clean e mixed
#   2. compare_model_vs_chamfer_topology_breakdown.py -> per coppia ordinata di topologie
#      (clean, cross_only) + pair_metrics.csv per i CI bootstrap (zs_summarize.py)
#
# UNA differenza, imposta dallo spazio (<= 15 GB nuovi nella home): gli operatori NON stanno
# nella home. Prima dell'eval:
#   a. zs_stage.py estrae dal pool di 500 i 100 soggetti che lo script di ranking estrarrebbe
#      (rebuild_subject_split, --eval_fraction 0.2 --max_subjects 500, WBES_EVAL_SEED=1234) e
#      mette le loro 600 mesh in /tmp del nodo;
#   b. v2_work/potential/areanorm_operators.py (k_eig 128, area unitaria: gli operatori dei tre
#      modelli) le porta in /tmp/.../ops, UNA volta per job;
#   c. l'eval gira su quella dir con --subject_split all --max_subjects 0 (come ws2_eval.sbatch):
#      stessi 100 soggetti, stesse 148.500 mesh pair cross.
# /tmp conta contro --mem: ~60 MB di operatori per identita' a risoluzione ICT/HIFI3D, ~3x per
# FaceVerse (passare --mem=160G). Si cancella all'uscita.
#
# Bracci: WBES_ZS_ARMS (lista, default = WBES_ZS_ARM o joint), fra
#   joint     BFM+ICT congiunto, job 1019532 (WBES_ZS_CKPT_JOINT)
#   bfm_only  solo BFM, job 1019310 (WBES_ZS_CKPT_BFM_ONLY)
#   ict_only  solo ICT, job 1019531 (WBES_ZS_CKPT_ICT_ONLY)
# Con piu' bracci e piu' GPU (--gres=gpu:l40s:N) ogni GPU prende un braccio, in parallelo, sugli
# stessi operatori: un job invece di N, per stare nel tetto di job dell'utente.
#
# Varianti (si sommano nel nome della out dir del braccio, <arm><varianti>):
#   WBES_ZS_FRAME=x,-y,-z      vertici trasformati prima degli operatori (zs_stage.py --transform):
#                              test del frame. Suffisso _frame-<segni>.
#   WBES_ZS_FLIP_FACES=1       verso dei triangoli invertito (le mesh BFM di training hanno le
#                              normali verso l'interno). Suffisso _flip.
#   WBES_ZS_EQSUPPORT=1        le mesh vengono dalla vista a supporto equalizzato ($ZS_EQ_DATA_DIR,
#                              eqsupport_view.py): ogni topologia ritagliata sulla regione del crop del
#                              suo soggetto. Valgono SOLO le righe con crop (crop -> X equalizzata); le
#                              righe senza crop del protocollo restano quelle della base
#                              (eqsupport_summarize.py). Si somma al frame (si ritaglia, poi si ruota).
#                              Suffisso _eqsupport.
#   WBES_ZS_EVAL_FRAME=rms     il forward ri-inquadra V_in nel frame rms (aau/models/eval_ablation.py,
#                              ablation_hooks.py); solo scenario clean. Suffisso _evalframe-rms.
#   WBES_ZS_PART=ranking|topology  uno solo dei due script (FaceVerse: in serie passano le 10 h).
#                              Suffisso _ranking / _topology.
#   WBES_ZS_EMBED=1            in piu', l'embedding di ogni mesh in <stage>/embeddings.npz (zs_embed.py),
#                              per retrieval e verifica d'identita'. Nessun suffisso.
#   WBES_ZS_EXPR=1             vista con espressioni (zs_env.sh): output in ${ZS_RUNS}_expr.
# Output in $ZS_RUNS/data_<fp>/<arm><varianti>/zs_zeroshot (fp = impronta dei manifest dei dati).
#
#   WBES_ZS_DOMAIN=hifi WBES_ZS_ARMS="joint bfm_only ict_only" WBES_ZS_FRAME=x,-y,-z \
#     aau/submit.sh zs3dmm/zs_zeroshot.sbatch --gres=gpu:l40s:3 --cpus-per-task=30 --mem=128G
#
# TEMPI: ~1h15 per braccio su HIFI3D (job 1055756); FaceVerse ~3.3 h per scenario a ~12.6
# coppie/s (job 1055801).
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net
zs_require_view

read -ra ARMS <<< "${WBES_ZS_ARMS:-${WBES_ZS_ARM:-joint}}"
for arm in "${ARMS[@]}"; do
    case "$arm" in
        joint|bfm_only|ict_only) ;;
        *) echo "ERRORE: braccio '$arm' (joint|bfm_only|ict_only)" >&2; exit 2 ;;
    esac
done
ckpt_of() {
    case "$1" in
        joint)    echo "$WBES_ZS_CKPT_JOINT" ;;
        bfm_only) echo "$WBES_ZS_CKPT_BFM_ONLY" ;;
        ict_only) echo "$WBES_ZS_CKPT_ICT_ONLY" ;;
    esac
}

WBES_EVAL_STAGE="${WBES_EVAL_STAGE:-zs_zeroshot}"
WBES_EVAL_SCENARIOS="${WBES_EVAL_SCENARIOS:-clean,mixed}"
export WBES_EVAL_SEED="${WBES_EVAL_SEED:-1234}"
ZS_FP="$(zs_data_fp)"

WBES_ZS_PART="${WBES_ZS_PART:-all}"
case "$WBES_ZS_PART" in
    all) PART_SUFFIX="" ;;
    ranking|topology) PART_SUFFIX="_$WBES_ZS_PART" ;;
    *) echo "ERRORE: WBES_ZS_PART=$WBES_ZS_PART (all|ranking|topology)" >&2; exit 2 ;;
esac
FRAME_SUFFIX=""
if [[ -n "${WBES_ZS_FRAME:-}" ]]; then
    FRAME_SUFFIX="_frame-$(echo "$WBES_ZS_FRAME" | tr -d ',' | tr '-' 'm')"   # x,-y,-z -> _frame-xmymz
fi
STAGE_FLIP=()
if [[ "${WBES_ZS_FLIP_FACES:-0}" == 1 ]]; then
    FRAME_SUFFIX="${FRAME_SUFFIX}_flip"
    STAGE_FLIP=(--flip-faces)
fi
LAUNCH=()
EVALFRAME_SUFFIX=""
if [[ -n "${WBES_ZS_EVAL_FRAME:-}" ]]; then
    [[ "$WBES_ZS_EVAL_FRAME" == rms ]] || { echo "ERRORE: WBES_ZS_EVAL_FRAME=$WBES_ZS_EVAL_FRAME (rms)" >&2; exit 2; }
    LAUNCH=(aau/models/eval_ablation.py --frame rms --)
    EVALFRAME_SUFFIX="_evalframe-rms"
    WBES_EVAL_SCENARIOS=clean   # il ri-inquadramento annullerebbe le perturbazioni del mixed
fi
EQ_SUFFIX=""
STAGE_VIEW="$ZS_DATA_DIR"
if [[ "${WBES_ZS_EQSUPPORT:-0}" == 1 ]]; then
    zs_require_eq_view
    EQ_SUFFIX="_eqsupport"
    STAGE_VIEW="$ZS_EQ_DATA_DIR"
fi
SUFFIX="$FRAME_SUFFIX$EQ_SUFFIX$EVALFRAME_SUFFIX$PART_SUFFIX"

# Nessun braccio di ablazione o di pilota: questi tre checkpoint non hanno
# ablation_arm.txt/pilot_arm.txt, e un checkpoint che li avesse vorrebbe frame, token o
# operatori propri che qui non ci sono. Bracci gia' completi: saltati PRIMA degli operatori.
TODO=()
for arm in "${ARMS[@]}"; do
    _runs_root="$(dirname "$(dirname "$(dirname "$(realpath "$(ckpt_of "$arm")")")")")"
    for f in ablation_arm.txt pilot_arm.txt; do
        if [[ -f "$_runs_root/$f" ]]; then
            echo "ERRORE: $_runs_root/$f presente: questo sbatch non sa valutare quel braccio" >&2
            exit 2
        fi
    done
    done_file="$ZS_RUNS/data_$ZS_FP/$arm$SUFFIX/$WBES_EVAL_STAGE/.done"
    if [[ -f "$done_file" ]]; then
        echo "[zs-zs] $arm$SUFFIX gia' completo: $done_file"
    else
        TODO+=("$arm")
    fi
done
unset _runs_root
if (( ${#TODO[@]} == 0 )); then
    exit 0
fi

# Il job id nel percorso: due job dello stesso dominio e della stessa variante sullo stesso nodo
# si cancellerebbero /tmp a vicenda (rm -rf qui sotto e trap all'uscita). Il percorso cambia a ogni
# sottomissione, quindi la data dir in eval_key.txt anche: vedi run_arm.
STAGE_TMP="/tmp/wbes_zs_${SLURM_JOB_ID:-manual}_${WBES_ZS_DOMAIN}${SUFFIX}"
rm -rf "$STAGE_TMP"
mkdir -p "$STAGE_TMP"
trap 'rm -rf "$STAGE_TMP"' EXIT

echo "[zs-zs] dominio=$WBES_ZS_DOMAIN ($ZS_LABEL) bracci=${TODO[*]} varianti='$SUFFIX' dati=data_$ZS_FP host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[zs-zs] a. soggetti valutati da $STAGE_VIEW -> $STAGE_TMP/in"
"$AAU_DIR/run.sh" aau/zs3dmm/zs_stage.py --view-dir "$STAGE_VIEW" \
    --out-dir "$STAGE_TMP/in" --seed "$WBES_EVAL_SEED" --transform="${WBES_ZS_FRAME:-}" "${STAGE_FLIP[@]}"  # "=": "-x,y,-z" sembrerebbe un flag

echo "[zs-zs] b. operatori (k_eig 128, area unitaria) su /tmp"
N_SHARDS="${SLURM_CPUS_PER_TASK:-8}"
# n coprimo con 6: i nomi sono soggetti x 6 topologie (vedi aau/ict/ict_ops.sbatch).
if (( N_SHARDS % 6 == 0 )); then
    N_SHARDS=$(( N_SHARDS - 1 ))
fi
SHARD_LOGS="$AAU_LOGS/${SLURM_JOB_NAME:-wbes-zs-zeroshot}-${SLURM_JOB_ID:-manual}.shards"
mkdir -p "$SHARD_LOGS" "$STAGE_TMP/ops"
pids=()
for (( i = 0; i < N_SHARDS; i++ )); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
      --input-dir "$STAGE_TMP/in" --output-dir "$STAGE_TMP/ops" \
      --k-eig 128 --shard "$i/$N_SHARDS" \
      > "$SHARD_LOGS/shard$i.log" 2>&1 &
    pids+=($!)
done
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "ERRORE: shard $i uscito != 0 (vedi $SHARD_LOGS/shard$i.log)" >&2
        exit 1
    fi
done
N_IN=$(find "$STAGE_TMP/in" -name "*.npz" | wc -l)
N_OPS=$(find "$STAGE_TMP/ops" -name "*.npz" | wc -l)
echo "[zs-zs] operatori: $N_OPS / $N_IN, $(du -sh "$STAGE_TMP/ops" | cut -f1) su /tmp"
if (( N_OPS != N_IN )); then
    echo "ERRORE: operatori incompleti" >&2
    exit 1
fi

# c. Un braccio per processo, in una subshell: eval_common.sh fissa variabili globali (checkpoint,
# out dir, common_args) e va sourceato una volta per braccio. Con G GPU i bracci si dividono in
# G code, una per GPU; ogni coda li valuta in serie.
run_arm() {  # <arm> <gpu>
    local arm="$1" gpu="$2"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        # Assegnazioni SENZA ${...:-}: env.sh ha esportato i percorsi BFM e il checkpoint v1, e
        # un default qui li lascerebbe passare (vedi la nota in hifi_env.sh).
        export WBES_DATA_DIR="$STAGE_TMP/ops"
        export WBES_DIST_NPZ="$ZS_DIST_NPZ"
        export WBES_CKPT="$(ckpt_of "$arm")"
        export WBES_EVAL_OUT="$ZS_RUNS/data_$ZS_FP/$arm$SUFFIX"
        # eval_key.txt di una sottomissione precedente NON completata (niente .done: chi e' completo
        # e' gia' saltato sopra) ha la data dir /tmp del job di allora, che non esiste piu': la si
        # toglie, altrimenti aau_eval_check_key rifiuterebbe la out dir. Le altre righe (checkpoint,
        # GT, seed) restano controllate. Una chiave con una data dir NON in /tmp/wbes_zs_ si lascia.
        key="$WBES_EVAL_OUT/eval_key.txt"
        if [[ -f "$key" ]] && grep -q "^data_dir=/tmp/wbes_zs_" "$key" \
           && ! grep -qx "data_dir=$(realpath "$WBES_DATA_DIR")" "$key"; then
            echo "[zs-zs] $arm$SUFFIX: eval_key.txt di un job precedente non completato, rigenerata"
            grep -v "^data_dir=" "$key" > "$key.prev_ckpt_gt"
            rm -f "$key"
        fi
        source "$WBES_ROOT/aau/eval_common.sh"
        # La data dir e' gia' il set di eval: tutti i suoi soggetti, nessun ricampionamento.
        common_args+=(--subject_split all --max_subjects 0)
        mkdir -p "$WBES_EVAL_OUT"
        cp "$STAGE_TMP/subjects.json" "$WBES_EVAL_OUT/subjects.json"
        # La vista equalizzata non entra nell'impronta dei dati: una vista ricostruita con un'altra
        # regola finirebbe nella stessa out dir e lo skip-if-done riuserebbe i numeri vecchi.
        if [[ -n "$EQ_SUFFIX" ]]; then
            if [[ -f "$WBES_EVAL_OUT/eqsupport_manifest.json" ]] \
               && ! cmp -s "$ZS_EQ_DIR/manifest.json" "$WBES_EVAL_OUT/eqsupport_manifest.json"; then
                echo "ERRORE: $WBES_EVAL_OUT viene da un'altra vista equalizzata (eqsupport_manifest.json diverso)" >&2
                exit 1
            fi
            cp "$ZS_EQ_DIR/manifest.json" "$WBES_EVAL_OUT/eqsupport_manifest.json"
        fi

        echo "[zs-zs] c. $arm$SUFFIX su GPU $gpu ckpt=$WBES_CKPT"
        echo "[zs-zs] data=$WBES_DATA_DIR gt=$WBES_DIST_NPZ out=$OUT_ROOT launch=${LAUNCH[*]:-}"
        aau_eval_begin "$WBES_EVAL_STAGE"
        if [[ -f "$key.prev_ckpt_gt" ]]; then
            # La chiave rigenerata deve coincidere con la vecchia a meno della data dir /tmp.
            if ! diff -q <(grep -v "^data_dir=" "$key") "$key.prev_ckpt_gt" >/dev/null; then
                echo "ERRORE: $WBES_EVAL_OUT conteneva un'altra combinazione (checkpoint/GT/seed): vedi $key.prev_ckpt_gt" >&2
                exit 1
            fi
            rm -f "$key.prev_ckpt_gt"
        fi

        if [[ "${WBES_ZS_EMBED:-0}" == 1 ]]; then
            # Embedding di ogni mesh (retrieval e verifica d'identita', zs_expr_summarize.py):
            # pochi minuti, prima delle ore di Chamfer del breakdown.
            echo "[$(timestamp)] 0/2 embedding di tutte le mesh valutate"
            "$AAU_DIR/run.sh" "${LAUNCH[@]}" aau/zs3dmm/zs_embed.py "${common_args[@]}" \
              --embeddings_out "$STAGE_OUT/embeddings.npz" \
              2>&1 | tee "$STAGE_OUT/embed.log"
        fi

        if [[ "$WBES_ZS_PART" != topology ]]; then
            echo "[$(timestamp)] 1/2 ranking aggregato, scenari: $WBES_EVAL_SCENARIOS"
            "$AAU_DIR/run.sh" "${LAUNCH[@]}" "$PERT/compare_model_vs_chamfer_rankings.py" \
              "${common_args[@]}" \
              --out_dir "$STAGE_OUT/ranking" \
              --aggregation_level subject_pair_mean \
              --scenarios "$WBES_EVAL_SCENARIOS" \
              --jitter_sigma 0.05 \
              --translation_sigma 0.05 \
              --rotation_sigma 0.05 \
              --mixed_jitter_sigma 0.05 \
              --mixed_translation_sigma 0.05 \
              --mixed_rotation_sigma 0.05 \
              2>&1 | tee "$STAGE_OUT/ranking.log"
        fi

        if [[ "$WBES_ZS_PART" != ranking ]]; then
            echo "[$(timestamp)] 2/2 breakdown per coppia di topologie (clean) + pair_metrics"
            "$AAU_DIR/run.sh" "${LAUNCH[@]}" "$PERT/compare_model_vs_chamfer_topology_breakdown.py" \
              "${common_args[@]}" \
              --out_dir "$STAGE_OUT/topology" \
              --topology_pair_mode cross_only \
              --ordered_topology_pairs \
              --write_pair_metrics \
              2>&1 | tee "$STAGE_OUT/topology.log"
        fi

        aau_eval_done "$WBES_EVAL_STAGE"
    )
}

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
N_Q=${#GPUS[@]}
(( N_Q > ${#TODO[@]} )) && N_Q=${#TODO[@]}
qpids=()
for (( q = 0; q < N_Q; q++ )); do
    (
        for (( k = q; k < ${#TODO[@]}; k += N_Q )); do
            arm="${TODO[$k]}"
            run_arm "$arm" "${GPUS[$q]}" > "$AAU_LOGS/${SLURM_JOB_NAME:-wbes-zs}-${SLURM_JOB_ID:-manual}.$arm.log" 2>&1
        done
    ) &
    qpids+=($!)
done
echo "[zs-zs] ${#TODO[@]} bracci su $N_Q GPU (${GPUS[*]}), log per braccio in $AAU_LOGS/${SLURM_JOB_NAME:-wbes-zs}-${SLURM_JOB_ID:-manual}.<braccio>.log"
rc=0
for p in "${qpids[@]}"; do
    wait "$p" || rc=1
done
for arm in "${TODO[@]}"; do
    f="$ZS_RUNS/data_$ZS_FP/$arm$SUFFIX/$WBES_EVAL_STAGE/.done"
    if [[ -f "$f" ]]; then echo "[zs-zs] $arm$SUFFIX OK"; else echo "[zs-zs] ERRORE: $arm$SUFFIX senza .done" >&2; rc=1; fi
done
exit "$rc"
