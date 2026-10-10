#!/usr/bin/env bash
# Eval di un braccio C3F del trainer v3 (protocollo ablation_protocol.md), ai passi 10.548 e 21.096 con i pesi EMA
# (epoch036_ema.pth, epoch072_ema.pth). Pipeline esistenti come in aau/evidence/e1_factorial/e1_eval_body.sh, ma
# lanciate da una copia di aau/zs3dmm/zs_zeroshot.sbatch con LAUNCH=(v3_work/trainer/eval_v3.py --) (modelli v3:
# pooling e frame per area) e scenario clean:
#   hifi    HIFI3D, breakdown + embedding (graduata con GT maxabs e unificata, rank-1)
#   devfs   dev FaceScape vista neutra (breakdown + embedding) e vista con espressioni (embedding): punteggio dev
#   fv      FaceVerse con espressioni, convenzione BFM (_flip), embedding
#   now     NoW, latenti dalla catena di aau/recon/now_ops_latent.sbatch (copia con eval_v3 davanti a now_latent.py)
#   fvn     (solo su richiesta, bracci a ingresso globale) FaceVerse NEUTRA come ``form``: uscite in
#           aau/runs/evidence/faceverse_neutral (WBES_FVN_OUT), protocollo PROTOCOL.md li'
# Uscite in aau/runs/evidence/trainer_v3/ablations/c3f_eval/. Ripartibile: i bracci con .done si saltano.
set -uo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
ARM="${WBES_V3_ARM:?WBES_V3_ARM}"
DEF_STEPS="hifi devfs fv now"
if [[ "$ARM" == factorized* || "$ARM" == ctrlfr* || "$ARM" == dual* ]]; then
  # ingresso globale: la scala delle mesh di eval viene dalle tabelle (eval_v3.py); gli script ricevono u (forma).
  # ``form``: embedding [s, u] e GT di E12 (datasets/CANONICAL_GT); ``famos``: FaMoS TEST con la scala metrica
  # (tools/eval_famos_v3.py); ``now``: la pipeline di e108, patch NoW alla taglia del template (le ricostruzioni
  # monoculari non sono metriche: per i fattorizzati conta u), factorized.md. ``dual``: ``form`` da [z_F, u] (le due
  # distanze in fact_summary), FaMoS con z_F e u, NoW su u e su z_F (now/<tag> e now/<tag>_zf).
  ST="$AAU_RUNS/evidence/trainer_v3/factorized/scale_tables"
  # UNA tabella per passo (dev FaceScape neutra ed espressioni hanno gli stessi nomi di file: insieme si
  # contraddicono, ScaleTable si ferma). Metriche dagli embedding (graduata per coppia di mesh come E12, rank-1):
  # niente breakdown con la Chamfer per braccio. Operatori dei set in cache condivisa.
  NOW_TABLES="$ST/now_scan.npz"
  for m in 3ddfa_v2 synergynet prnet mica mica_loop2; do NOW_TABLES+=":$ST/now_$m.npz"; done
  export WBES_V3_FACTORIZED_OUT=u
  export WBES_ZS_OPS_CACHE="$WBES_ROOT/datasets/V3_OPS_CACHE"
  mkdir -p "$WBES_ZS_OPS_CACHE"
  DEF_STEPS="form famos now"
fi
STEPS=" ${WBES_V3_EVAL_STEPS:-$DEF_STEPS} "
OUT="$AAU_RUNS/evidence/trainer_v3/ablations/c3f_eval"
SEED="${WBES_V3_SEED:-1234}"
SFX=""; [[ "$SEED" != 1234 ]] && SFX="_s$SEED"
RR="${WBES_V3_EVAL_RR:-$AAU_RUNS/evidence/trainer_v3/ablations/c3f_runs/$ARM$SFX}"   # C3M: la sua run dir
EPS="${WBES_V3_EVAL_EPOCHS:-036 072}"                                               # epoche dei checkpoint EMA
CK_DIR="$(ls -d "$RR"/v3_*/checkpoints | head -1)"
declare -A CK TAG
for e in $EPS; do
  CK[$e]="$CK_DIR/epoch${e}_ema.pth"
  [[ -f "${CK[$e]}" ]] || { echo "ERRORE: ${CK[$e]} assente" >&2; exit 1; }
  TAG[$e]="v3${ARM}${SFX#_}e$e"
  export "WBES_ZS_CKPT_SCALE_${TAG[$e]^^}=${CK[$e]}"
done
ARMS=""; for e in $EPS; do ARMS+="${ARMS:+ }scale_${TAG[$e]}"; done
ZS="/tmp/zs_v3c3f_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}.sh"
sed 's|^LAUNCH=()$|LAUNCH=(v3_work/trainer/eval_v3.py --)|' "$AAU_DIR/zs3dmm/zs_zeroshot.sbatch" > "$ZS"
grep -qx 'LAUNCH=(v3_work/trainer/eval_v3.py --)' "$ZS" || { echo "ERRORE: LAUNCH non sostituito" >&2; exit 1; }
# WBES_ZS_OPS_CACHE (solo bracci a ingresso globale): gli operatori del set valutato non dipendono dal braccio; si
# calcolano UNA volta in $WBES_ZS_OPS_CACHE/<sha1 di subjects.json: vista, soggetti, frame, flip> (lock), poi si riusano
python3 - "$ZS" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = 'echo "[zs-zs] b. operatori (k_eig 128, area unitaria) su /tmp"\n'
b = (a + 'OPS_DIR="$STAGE_TMP/ops"\nif [[ -n "${WBES_ZS_OPS_CACHE:-}" ]]; then\n'
     '    OPS_KEY=$(python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])), sort_keys=True))" '
     '"$STAGE_TMP/subjects.json" | sha1sum | cut -c1-16)\n'
     '    OPS_DIR="$WBES_ZS_OPS_CACHE/$OPS_KEY"; mkdir -p "$OPS_DIR"; cp "$STAGE_TMP/subjects.json" "$OPS_DIR.subjects.json"\n'
     '    exec 9> "$OPS_DIR.lock"; flock 9; find "$OPS_DIR" -name ".*.tmp.npz" -delete; echo "[zs-zs] cache degli operatori $OPS_DIR"\nfi\n')
c = '--input-dir "$STAGE_TMP/in" --output-dir "$STAGE_TMP/ops"'
d = 'N_OPS=$(find "$STAGE_TMP/ops" -name "*.npz" | wc -l)'
e = 'echo "ERRORE: operatori incompleti" >&2\n    exit 1\nfi\n'
for x in (a, c, d, e):
    assert s.count(x) == 1, x
s = s.replace(a, b).replace(c, '--input-dir "$STAGE_TMP/in" --output-dir "$OPS_DIR"')
s = s.replace(d, 'N_OPS=$(find "$OPS_DIR" -name "*.npz" | wc -l)')
s = s.replace(e, e + 'if [[ "$OPS_DIR" != "$STAGE_TMP/ops" ]]; then flock -u 9; rm -rf "$STAGE_TMP/ops"; ln -s "$OPS_DIR" "$STAGE_TMP/ops"; fi\n')
open(p, "w").write(s)
PY
grep -q 'WBES_ZS_OPS_CACHE' "$ZS" || { echo "ERRORE: cache degli operatori non agganciata" >&2; exit 1; }
NOWS="/tmp/now_v3c3f_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}.sh"
sed 's|"\$AAU_DIR/run.sh" aau/recon/now_latent.py|"$AAU_DIR/run.sh" v3_work/trainer/eval_v3.py -- aau/recon/now_latent.py|' \
  "$AAU_DIR/recon/now_ops_latent.sbatch" > "$NOWS"
grep -q 'eval_v3.py -- aau/recon/now_latent.py' "$NOWS" || { echo "ERRORE: now_latent non agganciato" >&2; exit 1; }
trap 'rm -f "$ZS" "$NOWS"' EXIT
export WBES_EVAL_SCENARIOS=clean
echo "[v3-eval] braccio=$ARM bracci=$ARMS host=$(hostname) job=${SLURM_JOB_ID:-none} riavvii=${SLURM_RESTART_COUNT:-0}"
FAILED=()
# Ogni passo fino a 3 tentativi a 10 minuti di distanza: su nv-ai-04 la risoluzione dell'utente di dominio cade a
# tratti ("unknown userid" all'avvio di singularity: 9 ottobre 05:33-05:51 e 11:06-11:09) e i passi sono ripartibili
# (le stage con .done si saltano).
retry() {  # $1 nome, resto: comando
  local name="$1" t; shift
  for t in 1 2 3; do
    "$@" && return 0
    (( t < 3 )) && { echo "[v3-eval] $(date +%T) $name: tentativo $t fallito, riprovo fra 10 minuti"; sleep 600; }
  done
  return 1
}
step_hifi() {
  WBES_ZS_DOMAIN=hifi WBES_HIFI_RUNS="$OUT/hifi_runs" WBES_ZS_ARMS="$ARMS" WBES_ZS_PART=topology WBES_ZS_EMBED=1 bash "$ZS"
}
step_devfs() {
  ( source aau/zs3dmm/dev_facescape_env.sh
    export WBES_FV_RUNS="$OUT/devfs" WBES_ZS_ARMS="$ARMS" WBES_ZS_PART=topology WBES_ZS_EMBED=1
    unset WBES_ZS_EXPR WBES_ZS_FLIP_FACES
    bash "$ZS" )
}
step_devfs_expr() {
  ( source aau/zs3dmm/dev_facescape_env.sh
    export WBES_FV_RUNS="$OUT/devfs" WBES_ZS_EXPR=1 WBES_ZS_ARMS="$ARMS" WBES_ZS_PART=embed
    unset WBES_ZS_FLIP_FACES
    bash "$ZS" )
}
step_fv() {
  WBES_ZS_DOMAIN=fv WBES_ZS_EXPR=1 WBES_FV_RUNS="$OUT/fv" WBES_ZS_FLIP_FACES=1 WBES_ZS_ARMS="$ARMS" \
    WBES_ZS_PART=embed bash "$ZS"
}
step_form() {  # bracci a ingresso globale: [s, u] (o z) di ogni mesh, poi tools/eval_factorized.py con le GT di E12
  local e dom set g emb V="${ARM}${SFX#_}" FARMS
  FARMS=""; for e in $EPS; do FARMS+="${FARMS:+ }scale_v3${V}fulle$e"; done
  for e in $EPS; do export "WBES_ZS_CKPT_SCALE_V3${V^^}FULLE$e=${CK[$e]}"; done
  WBES_V3_SCALE_TABLES="$ST/hifi3d_eval.npz" WBES_V3_FACTORIZED_OUT=full WBES_ZS_DOMAIN=hifi \
    WBES_HIFI_RUNS="$OUT/form_hifi" WBES_ZS_ARMS="$FARMS" WBES_ZS_PART=embed bash "$ZS" || return 1
  ( source aau/zs3dmm/dev_facescape_env.sh
    export WBES_FV_RUNS="$OUT/form_devfs" WBES_ZS_ARMS="$FARMS" WBES_ZS_PART=embed WBES_V3_FACTORIZED_OUT=full
    export WBES_V3_SCALE_TABLES="$ST/devfs_eval.npz"
    unset WBES_ZS_EXPR WBES_ZS_FLIP_FACES
    bash "$ZS" ) || return 1
  ( source aau/zs3dmm/dev_facescape_env.sh
    export WBES_FV_RUNS="$OUT/form_devfs" WBES_ZS_EXPR=1 WBES_ZS_ARMS="$FARMS" WBES_ZS_PART=embed
    export WBES_V3_FACTORIZED_OUT=full WBES_V3_SCALE_TABLES="$ST/devfs_expr.npz"
    unset WBES_ZS_FLIP_FACES
    bash "$ZS" ) || return 1
  WBES_V3_SCALE_TABLES="$ST/fv_expr.npz" WBES_V3_FACTORIZED_OUT=full WBES_ZS_DOMAIN=fv WBES_ZS_EXPR=1 \
    WBES_FV_RUNS="$OUT/form_fv" WBES_ZS_FLIP_FACES=1 WBES_ZS_ARMS="$FARMS" WBES_ZS_PART=embed bash "$ZS" || return 1
  local G=datasets/CANONICAL_GT/eval
  for pair in hifi:hifi3d:HIFI3D/eval_view devfs:facescape:DEV_FACESCAPE/eval_view fv:faceverse:FACEVERSE_ZS/expr_view; do
    IFS=: read -r dom set view <<< "$pair"
    gts=(--gt "fr=$G/${set}_fr.npz" --gt "sr=$G/${set}_sr.npz" --gt "maxabs=datasets/$view/gt_matrix.npz")
    for e in $EPS; do
      emb=$(find "$OUT/form_${dom}$([[ $dom == fv ]] && echo _expr)" -path "*scale_v3${V}fulle${e}*" -name embeddings.npz | head -1)
      [[ -n "$emb" ]] || { echo "[v3-eval] ERRORE: embedding [s, u] di $dom e$e assenti"; return 1; }
      AAU_NV= "$AAU_DIR/run.sh" v3_work/trainer/tools/eval_factorized.py --embeddings "$emb" "${gts[@]}" \
        --size-table "$G/${set}_centroid_size.npz" --out-dir "$OUT/form/$dom/v3${V}e$e" || return 1
    done
  done
}
step_fvn() {  # FaceVerse NEUTRA (aau/runs/evidence/faceverse_neutral/PROTOCOL.md): come ``form`` sulla sola eval_view
  local e emb V="${ARM}${SFX#_}" FARMS FO="${WBES_FVN_OUT:-$AAU_RUNS/evidence/faceverse_neutral}"
  local G=datasets/CANONICAL_GT/eval
  FARMS=""; for e in $EPS; do FARMS+="${FARMS:+ }scale_v3${V}fulle$e"; done
  for e in $EPS; do export "WBES_ZS_CKPT_SCALE_V3${V^^}FULLE$e=${CK[$e]}"; done
  ( unset WBES_ZS_EXPR
    WBES_V3_SCALE_TABLES="$FO/scale_tables/fv_eval.npz" WBES_V3_FACTORIZED_OUT=full WBES_ZS_DOMAIN=fv \
      WBES_FV_RUNS="$FO/embed" WBES_ZS_FLIP_FACES=1 WBES_ZS_ARMS="$FARMS" WBES_ZS_PART=embed bash "$ZS" ) || return 1
  for e in $EPS; do
    emb=$(find "$FO/embed" -path "*scale_v3${V}fulle${e}_flip_embed*" -name embeddings.npz | head -1)
    [[ -n "$emb" ]] || { echo "[v3-eval] ERRORE: embedding [s, u] di FaceVerse neutra e$e assenti"; return 1; }
    AAU_NV= "$AAU_DIR/run.sh" v3_work/trainer/tools/eval_factorized.py --embeddings "$emb" \
      --gt "fr=$G/faceverse_fr.npz" --gt "sr=$G/faceverse_sr.npz" --gt "maxabs=datasets/FACEVERSE_ZS/eval_view/gt_matrix.npz" \
      --size-table "$G/faceverse_centroid_size.npz" --out-dir "$FO/form/v3${V}e$e" || return 1
  done
}
step_famos() {  # FaMoS TEST: operatori ad area unitaria delle patch su /tmp, poi eval_famos_v3.py per ogni checkpoint
  local e T="/tmp/v3famos_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}" V="$WBES_ROOT/datasets/FAMOS/test_view" i pids=()
  mkdir -p "$T/ops"
  for (( i = 0; i < 12; i++ )); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py --input-dir "$V/npz" \
      --output-dir "$T/ops" --k-eig 128 --shard "$i/12" > /dev/null 2>&1 & pids+=($!)
  done
  for i in "${pids[@]}"; do wait "$i" || return 1; done
  for e in $EPS; do
    WBES_V3_FACTORIZED_OUT=full WBES_V3_SCALE_TABLES="$ST/famos_test.npz" "$AAU_DIR/run.sh" v3_work/trainer/eval_v3.py -- \
      v3_work/trainer/tools/eval_famos_v3.py --ops-dir "$T/ops" --checkpoint "${CK[$e]}" --tag "${TAG[$e]}full" \
      --out-dir "$OUT/famos/${TAG[$e]}" --workers 12 || { rm -rf "$T"; return 1; }
  done
  rm -rf "$T"
}
step_now() {  # $1 W, $2 O, $3 checkpoint, $4 uscita del modello (dual: u|zf; vuoto: quella esportata)
  WBES_V3_FACTORIZED_OUT="${4:-${WBES_V3_FACTORIZED_OUT:-u}}" WBES_V3_SCALE_TABLES="${NOW_TABLES:-}" \
    WBES_NOW_WORK="$1" WBES_NOW_OUT="$2" WBES_CKPT="$3" bash "$NOWS" \
    && WBES_NOW_WORK="$1" WBES_NOW_OUT="$2" AAU_NV= "$AAU_DIR/run.sh" aau/recon/now_summarize.py
}
if [[ "$STEPS" == *" hifi "* ]]; then
  echo "[v3-eval] $(date +%T) HIFI3D"
  retry hifi step_hifi || FAILED+=(hifi)
fi
if [[ "$STEPS" == *" devfs "* ]]; then
  echo "[v3-eval] $(date +%T) dev FaceScape, vista neutra"
  retry devfs step_devfs || FAILED+=(devfs)
  echo "[v3-eval] $(date +%T) dev FaceScape, vista con espressioni"
  retry devfs_expr step_devfs_expr || FAILED+=(devfs_expr)
fi
if [[ "$STEPS" == *" fv "* ]]; then
  echo "[v3-eval] $(date +%T) FaceVerse con espressioni"
  retry fv step_fv || FAILED+=(fv)
fi
if [[ "$STEPS" == *" form "* ]]; then
  echo "[v3-eval] $(date +%T) form: embedding [s, u] e GT di E12"
  retry form step_form || FAILED+=(form)
fi
if [[ "$STEPS" == *" fvn "* ]]; then
  echo "[v3-eval] $(date +%T) FaceVerse neutra: embedding [s, u] e GT di E12"
  retry fvn step_fvn || FAILED+=(fvn)
fi
if [[ "$STEPS" == *" famos "* ]]; then
  echo "[v3-eval] $(date +%T) FaMoS TEST"
  retry famos step_famos || FAILED+=(famos)
fi
if [[ "$STEPS" == *" now "* ]]; then
  SRC_W="$HOME/data/now_eval_work"
  SRC_O="$AAU_RUNS/now_eval"
  NOW_OUTS=""; [[ "$ARM" == dual* ]] && NOW_OUTS="u zf"
  for e in $EPS; do
   for o in ${NOW_OUTS:-.}; do
    t="${TAG[$e]}"; [[ "$o" == zf ]] && t+="_zf"
    W="$HOME/data/v3c3f_now/$t"
    O="$OUT/now/$t"
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
    echo "ckpt=${CK[$e]}" > "$O/checkpoint.txt"
    [[ "$o" != . ]] && echo "uscita=$o" >> "$O/checkpoint.txt"
    echo "[v3-eval] $(date +%T) NoW $t"
    retry "now_$t" step_now "$W" "$O" "${CK[$e]}" "${o#.}" || FAILED+=("now_$t")
   done
  done
fi
echo "[v3-eval] $(date +%T) fine: falliti=${FAILED[*]:-nessuno}"
[[ "$0" == /tmp/v3c3f_eval_body_* ]] && rm -f "$0"
(( ${#FAILED[@]} == 0 ))
