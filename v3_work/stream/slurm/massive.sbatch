#!/usr/bin/env bash
#SBATCH --job-name=wbes-stream-massive
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=prioritized
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-gpu=20
#SBATCH --mem-per-gpu=80G
#SBATCH --time=02:00:00
#SBATCH --requeue
#
# Run a streaming (P1) con la GT di E12, su uno o piu' nodi: per OGNI nodo produttori + anello in RAM + i rank DDP
# del nodo (torchrun con rendezvous c10d fra i nodi), corpo in massive_node.sh (uno srun per nodo).
#   prova, 1 GPU:     STREAM_SOURCES=validated sbatch v3_work/stream/slurm/massive.sbatch   (STREAM_STEPS=2000 ...)
#   prove brevi della configurazione (2 e 6 + 6 L40S): trial_validated.sbatch
#   run principale, 12 L40S = 6 + 6 (STESSO numero di GPU per nodo):
#     STREAM_SOURCES=<preset> sbatch --nodes=2 --gpus-per-node=l40s:6 --time=2-00:00:00 v3_work/stream/slurm/massive.sbatch
#   NIENTE 8 + 4: con un numero di GPU diverso fra i nodi la prima all_reduce di NCCL non finisce mai (misurato il 9
#   ottobre con 2 + 1 GPU: RoCE, socket, Ring, Tree, PXN/P2P/SHM spenti, NCCL 2.22 e quello di pytorch_25.09; 1 + 1,
#   1 + 1 + 1 e 2 + 2 funzionano). Un job eterogeneo (componenti separati da ':') parte solo con STREAM_ALLOW_HETERO=1
#   e GPU uguali per componente (es. 2 A100 di nv-ai-04 + 2 L40S, provato).
#   A100 (secondari): sbatch -p aicentre-a100 --qos=unprivileged --gpus-per-node=a100:8 --cpus-per-gpu=28 \
#            --mem-per-gpu=110G --time=2-00:00:00 v3_work/stream/slurm/massive.sbatch
# Ripresa: --requeue + --resume auto del trainer + checkpoint al passo in corso su SIGTERM (STREAM_PREEMPTIBLE=1, il
# default, massive_node.sh). La run dir dipende dal job (l'anello /tmp/<job>_stream/r0 e' nell'hash) ma non dai riavvii;
# l'anello si riempie da capo, i produttori ripartono con semi nuovi (SEME + 1000 x nodo + 100000 x segmento, scritti in
# ogni gruppo; segmento = partenze precedenti del run, righe di $STREAM_OUT/segments.log).
# Codice congelato: alla prima partenza slurm/code_snapshot.py copia il codice del repo (*.py, *.sh, *.sbatch) in
# $STREAM_OUT/code e questo script si riesegue da li': massive_node.sh, produttori e trainer usano la copia, i dati
# restano quelli del repo (stessi percorsi nella riga di lancio, quindi stessa run dir). A ogni riavvio la copia si
# verifica (sha256 del manifesto) e la ripresa e' obbligatoria: senza last.pth nella run dir errore esplicito
# (massive_node.sh, STREAM_EXPECT_RESUME), mai ripartenza da zero; un riavvio senza la copia e' un errore.
# Continuazione dopo un TIMEOUT con un job nuovo: lo stesso STREAM_OUT (il job della prima partenza, STREAM_RUN_TAG, si
# legge da $STREAM_OUT/run_tag) e le stesse variabili:
#   STREAM_OUT=<STREAM_OUT del run> STREAM_SOURCES=<le stesse> sbatch <le stesse opzioni> v3_work/stream/slurm/massive.sbatch
# Variabili (default = la prova breve; ricetta e differenze dai bracci C3F/C3M in testa a massive_node.sh):
#   STREAM_SOURCES  OBBLIGATORIA, nessun default (senza, il job esce con errore). Preset (sources.SOURCE_PRESETS):
#                     validated = bfm2019, ict, gnm: i domini dei bracci decisivi C3F/C3M (BFM REMESH, ICT, GNM), con
#                                 BFM 2019 al posto di BFM REMESH (3DDFA), che lo stream non ha;
#                     max       = bfm2019, ict, gnm, flame2023, famos (= massive, il preset di prima: + FLAME 2023 Open
#                                 e FaMoS TRAIN, 80 persone);
#                     open_core = gnm, ict, flame2023 (nucleo aperto); oppure una lista di domini
#   STREAM_ARM      factorized | factorized2 | ctrlfr (testa standard su GT-FR) | unified (la GT unificata di prima)
#   STREAM_IDS_PER_RANK (5), STREAM_MESHES_PER_ID (6), STREAM_VIEWS (viste per identita' dei produttori, = MESHES)
#   STREAM_BATCH_DOMAINS (single; mixed = domini mescolati nel batch), STREAM_SIZE_MASK (vuoto; es. bfm2019)
#   STREAM_RECIPE   default dei dati: c3m (con validated: quote d'espressioni del C3M per dominio, niente mm_aug,
#                   niente rotazione a ogni uso, discretizzazioni senza reinserimento) | max (con gli altri preset:
#                   le deviazioni); dettagli e numeri in testa a massive_node.sh. Ognuno sovrascrivibile da:
#   STREAM_EXPR_FRAC (c3m bfm2019=0,ict=0.2315,gnm=0.1968; max 0.5), STREAM_ROT (c3m 0; max 30,15,10),
#   STREAM_LABEL_DRAW (c3m perm; max replace), STREAM_MM_AUG (sotto)
#   STREAM_STEPS, STREAM_SPE (passi per epoca), STREAM_K (64|128), STREAM_REUSE (tetto morbido, 4),
#   STREAM_SAVE_EVERY, STREAM_EVAL_EVERY (epoche; default ~10% del run), STREAM_LOG_BATCHES (200: dominio e fonti dei
#   primi batch per rank, runs/*/batch_domains_rank<r>.csv)
#   STREAM_FORWARD (groups; sequential = il percorso v2), WBES_STREAM_PROFILE=1 (tempi per fase)
#   STREAM_MM_AUG   probabilita' dei moltiplicatori (c3m vuoto = puri; max hybrid=0.3,expr_transfer=0.15,rbf=0.15)
#   STREAM_RING_GB_PER_GPU (10), STREAM_TRAIN_CPUS_PER_RANK (8), STREAM_EXTRA (anelli condivisi, es. CephFS),
#   STREAM_MIRROR_THREADS (2)
#   STREAM_OUT (aau/runs/evidence/stream/massive_<job>), STREAM_SEED (1234), STREAM_TEST_CRASH (passo, solo test)
#   STREAM_PREEMPTIBLE (1: su SIGTERM il trainer salva al passo in corso; 0 = trainer in primo piano, come prima),
#   STREAM_CKPT_MIN (20: checkpoint di ripresa ogni N minuti; entra nell'hash della run dir: uguale fra le partenze)
set -euo pipefail
if [[ -z "${STREAM_SOURCES:-}" ]]; then
  echo "[massive] ERRORE: STREAM_SOURCES obbligatorio (validated | max | open_core | lista di domini), nessun default" >&2
  exit 2
fi
if [[ -z "${STREAM_CODE:-}" ]]; then
  # --- prima del resto: copia del codice (prima partenza) o sua verifica (riavvio, continuazione), poi si riesegue
  # questo script dalla copia. WBES_ROOT e AAU_RUNS restano il repo vivo: dati e percorsi della riga di lancio
  export WBES_ROOT="${WBES_ROOT:-$PWD}"
  export AAU_RUNS="${AAU_RUNS:-$WBES_ROOT/aau/runs}" AAU_LOGS="${AAU_LOGS:-$WBES_ROOT/aau/logs}"
  export STREAM_OUT="${STREAM_OUT:-$AAU_RUNS/evidence/stream/massive_${SLURM_JOB_ID:-manual}}"
  mkdir -p "$STREAM_OUT"
  STREAM_OUT="$(cd "$STREAM_OUT" && pwd)"
  export STREAM_CODE="$STREAM_OUT/code"
  if [[ -f "$STREAM_CODE/MANIFEST.json" ]]; then
    python3 "$STREAM_CODE/v3_work/stream/slurm/code_snapshot.py" verify "$STREAM_CODE" 2>&1 | tee -a "$STREAM_OUT/massive.log"
    export STREAM_RUN_TAG="${STREAM_RUN_TAG:-$(cat "$STREAM_OUT/run_tag" 2>/dev/null || echo "${SLURM_JOB_ID:-manual}")}"
    export STREAM_EXPECT_RESUME=1
  elif [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]]; then
    echo "[massive] ERRORE: riavvio ${SLURM_RESTART_COUNT} senza la copia del codice ($STREAM_CODE/MANIFEST.json)" >&2
    exit 2
  else
    python3 "$WBES_ROOT/v3_work/stream/slurm/code_snapshot.py" make "$WBES_ROOT" "$STREAM_CODE" 2>&1 \
      | tee -a "$STREAM_OUT/massive.log"
    export STREAM_RUN_TAG="${STREAM_RUN_TAG:-${SLURM_JOB_ID:-manual}}"
    echo "$STREAM_RUN_TAG" > "$STREAM_OUT/run_tag"
  fi
  exec bash "$STREAM_CODE/v3_work/stream/slurm/massive.sbatch"
fi
source "$STREAM_CODE/aau/env.sh"
cd "$WBES_ROOT"
# segmento = partenze precedenti del run (semi dei produttori, massive_node.sh); versione del codice nelle ricette
SEG=0
[[ -f "$STREAM_OUT/segments.log" ]] && SEG=$(wc -l < "$STREAM_OUT/segments.log")
echo "$SEG job=${SLURM_JOB_ID:-} riavvio=${SLURM_RESTART_COUNT:-0} $(date +%F_%T)" >> "$STREAM_OUT/segments.log"
export STREAM_SEGMENT="$SEG"
export WBES_CODE_VERSION="$(python3 "$STREAM_CODE/v3_work/stream/slurm/code_snapshot.py" version "$STREAM_CODE")"
# nodi: quelli del job o, in un job eterogeneo, dei suoi componenti in ordine (massive_node.sh ne ricava indice e numero)
if [[ -n "${SLURM_HET_SIZE:-}" ]]; then
  NL=""; for i in $(seq 0 $(( SLURM_HET_SIZE - 1 ))); do v="SLURM_JOB_NODELIST_HET_GROUP_$i"; NL+="${NL:+,}${!v}"; done
  SRUN=(--het-group="$(seq -s, 0 $(( SLURM_HET_SIZE - 1 )))")
else
  NL="$SLURM_JOB_NODELIST"; SRUN=(--ntasks-per-node=1)
fi
export STREAM_NODES=$(scontrol show hostnames "$NL" | awk '!s[$0]++' | xargs)
export STREAM_MASTER=${STREAM_NODES%% *}
if [[ -n "${SLURM_HET_SIZE:-}" && $(wc -w <<< "$STREAM_NODES") -ne $SLURM_HET_SIZE ]]; then
  echo "[massive] ERRORE: due componenti sullo stesso nodo ($NL): usare --nodelist o --exclude per componente" >&2
  exit 2
fi
if [[ -n "${SLURM_HET_SIZE:-}" && "${STREAM_ALLOW_HETERO:-0}" != 1 ]]; then
  echo "[massive] ERRORE: job eterogeneo: NCCL si blocca se le GPU per nodo sono diverse (vedi sopra);" \
       "STREAM_ALLOW_HETERO=1 solo con lo stesso numero di GPU in ogni componente" >&2
  exit 2
fi
export STREAM_PORT=$(( 29400 + ${SLURM_JOB_ID:-0} % 500 ))
echo "[massive] $(date +%F_%T) job=${SLURM_JOB_ID:-} riavvio=${SLURM_RESTART_COUNT:-0} segmento=$SEG" \
     "nodi=$STREAM_NODES rendezvous=$STREAM_MASTER:$STREAM_PORT arm=${STREAM_ARM:-factorized} fonti=$STREAM_SOURCES" \
     "codice=$STREAM_CODE ($WBES_CODE_VERSION) tag=${STREAM_RUN_TAG:-${SLURM_JOB_ID:-manual}} ripresa=${STREAM_EXPECT_RESUME:-0}" \
  | tee -a "$STREAM_OUT/massive.log"
# un task per nodo, con tutte le CPU e le GPU del nodo nel job
srun "${SRUN[@]}" --kill-on-bad-exit=1 bash "$STREAM_CODE/v3_work/stream/slurm/massive_node.sh"
echo "[massive] $(date +%F_%T) fine" | tee -a "$STREAM_OUT/massive.log"
