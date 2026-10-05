#!/usr/bin/env bash
# Configurazione del test zero-shot FLAME (WS-FLAME). Si sourcea DOPO aau/env.sh:
#
#   source aau/env.sh; source aau/flame/flame_env.sh
#
# Solo variabili WBES_FLAME_*: non tocca WBES_DATA_DIR / WBES_DIST_NPZ / WBES_ICT_*. Gli sbatch
# FLAME che chiamano eval_common.sh assegnano a mano WBES_DATA_DIR (la vista con operatori su /tmp),
# SEMPRE e senza ${...:-}: env.sh esporta i percorsi BFM e il job li eredita (job 1055122,
# eval "ICT" girato sui dati BFM).
#
# LICENZA: il modello FLAME non e' nel repo, non si copia da altri utenti e non si scarica:
# lo fornisce l'utente con la sua licenza MPI. Default = la cartella official/ della pipeline
# storica v2_work/genflame (oggi vuota), dove l'utente mette i suoi file; nessun altro
# ripiego. flame_require_model esce se i file non ci sono.
export WBES_FLAME_MODEL="${WBES_FLAME_MODEL:-$WBES_ROOT/v2_work/genflame/official/FLAME2020/generic_model.pkl}"
export WBES_FLAME_MASKS="${WBES_FLAME_MASKS:-$WBES_ROOT/v2_work/genflame/official/FLAME_masks.pkl}"

# SPAZIO: in home SOLO geometria compressa (npz compressi float32) e GT, come per HIFI3D. Gli
# operatori (np.savez non compresso, ~60 MB per identita' su ICT) NON vanno in home: li calcola
# ogni job di eval sul /tmp del suo nodo (flame_stage_ops qui sotto) e li cancella all'uscita.
# Le teste intere del generatore storico stanno anch'esse solo su /tmp del job di build.
export WBES_FLAME_ROOT="${WBES_FLAME_ROOT:-$WBES_ROOT/datasets/FLAME}"
export WBES_FLAME_IDENTITIES_DIR="${WBES_FLAME_IDENTITIES_DIR:-$WBES_FLAME_ROOT/identities}"
export WBES_FLAME_TOPO_DIR="${WBES_FLAME_TOPO_DIR:-$WBES_FLAME_ROOT/topo}"
export WBES_FLAME_GT_DIR="${WBES_FLAME_GT_DIR:-$WBES_FLAME_ROOT/gt}"
# Vista idNNNN (offset 1000, convenzione di v2_work/genflame/make_train_ready.py) sulle mesh
# SENZA operatori, piu' la GT rinominata. La sottocartella si chiama npz_withops perche' il
# nome e' fisso in make_train_ready.py, ma contiene solo symlink alla geometria di topo/:
# la leggono le baseline (che usano solo vertici e facce) e la usa la GT degli eval.
export WBES_FLAME_VIEW_DIR="${WBES_FLAME_VIEW_DIR:-$WBES_FLAME_ROOT/eval_view}"
export WBES_FLAME_MESH_DIR="${WBES_FLAME_MESH_DIR:-$WBES_FLAME_VIEW_DIR/npz_withops}"
export WBES_FLAME_DIST_NPZ="${WBES_FLAME_DIST_NPZ:-$WBES_FLAME_VIEW_DIR/gt_matrix.npz}"
# Output per impronta dei dati: $WBES_FLAME_RUNS/<impronta>/{joint,bfm_only,baselines}.
export WBES_FLAME_RUNS="${WBES_FLAME_RUNS:-$AAU_RUNS/ws_flame}"
# Radice su /tmp del nodo (RAM: conta contro --mem). Percorso STABILE, senza job id: la data
# dir entra nell'hash e in eval_key.txt di eval_common.sh, e un percorso diverso a ogni job
# farebbe rifiutare la risottomissione. Una sottocartella per braccio, cosi' due bracci sullo
# stesso nodo non si pestano.
export WBES_FLAME_TMP="${WBES_FLAME_TMP:-/tmp/wbes-flame-$USER}"

# Campionamento: pool di 500 come eval_view_heldout di ICT, 100 modi di forma e seed 1234,
# generatore storico v2_work/genflame/generate_identities.py. WBES_FLAME_SAMPLING sceglie la
# distribuzione dei coefficienti:
#   ict     (default) N(0,1) SENZA troncamento, come ICT (v2_work/genict/generate_identities.py,
#           il campionatore di ICT-FaceKit): stessa regola per tutti i domini della tabella.
#   legacy  N(0,1) troncata a +-2.5 sigma per rifiuto, come build_flame_5000.sh e il numero
#           storico BFM->FLAME 0.478. Quel confronto e' comunque perso, perche' il crop qui e'
#           la maschera ufficiale e non la corrispondenza BFM->FLAME: serve solo a riprodurre.
# Le shapedirs FLAME sono gia' scalate sulla deviazione standard, quindi N(0,1) e' il prior.
export WBES_FLAME_N_IDENTITIES="${WBES_FLAME_N_IDENTITIES:-500}"
export WBES_FLAME_SEED="${WBES_FLAME_SEED:-1234}"
export WBES_FLAME_N_SHAPE="${WBES_FLAME_N_SHAPE:-100}"
export WBES_FLAME_SAMPLING="${WBES_FLAME_SAMPLING:-ict}"
case "$WBES_FLAME_SAMPLING" in
    ict)    WBES_FLAME_TRUNC=0 ;;
    legacy) WBES_FLAME_TRUNC=2.5 ;;
    *) echo "ERRORE: WBES_FLAME_SAMPLING=$WBES_FLAME_SAMPLING (ict|legacy)" >&2; return 2 ;;
esac
export WBES_FLAME_TRUNC
export WBES_FLAME_CROP="${WBES_FLAME_CROP:-mask}"
export WBES_FLAME_MASK_REGION="${WBES_FLAME_MASK_REGION:-face}"
export WBES_FLAME_ID_OFFSET="${WBES_FLAME_ID_OFFSET:-1000}"

# Modelli da valutare (gli stessi della tabella WS2, seed 1234, ricetta v1, area unitaria).
_flame_run_dir="mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d"
export WBES_FLAME_CKPT_JOINT="${WBES_FLAME_CKPT_JOINT:-$AAU_RUNS/x3dmm_joint_bfm_ict_s1234_1019532/$_flame_run_dir/checkpoints/best_by_xtopo_mesh_clean.pth}"
export WBES_FLAME_CKPT_BFM_ONLY="${WBES_FLAME_CKPT_BFM_ONLY:-$AAU_RUNS/remesh_v1recipe_areanorm_s1234_1019310/$_flame_run_dir/checkpoints/best_by_xtopo_mesh_clean.pth}"
unset _flame_run_dir

flame_require_model() {
    local bad=0
    if [[ ! -f "$WBES_FLAME_MODEL" ]]; then
        echo "ERRORE: modello FLAME assente: WBES_FLAME_MODEL=$WBES_FLAME_MODEL" >&2
        echo "  Metti li' il generic_model.pkl della licenza ufficiale, o esporta WBES_FLAME_MODEL." >&2
        bad=1
    fi
    if [[ "$WBES_FLAME_CROP" == "mask" && ! -f "$WBES_FLAME_MASKS" ]]; then
        echo "ERRORE: maschere FLAME assenti: WBES_FLAME_MASKS=$WBES_FLAME_MASKS" >&2
        echo "  Serve FLAME_masks.pkl (stessa pagina di download del modello) per il crop del volto." >&2
        echo "  WBES_FLAME_CROP=head valuta la testa intera: NON confrontabile con BFM/ICT." >&2
        bad=1
    fi
    return "$bad"
}

# Timbro dei dati: copia del manifest delle identita' (modello, maschera, campionamento, seed,
# sha256 dei file). flame_build.sbatch lo scrive in topo/ PRIMA delle topologie e nella vista
# DOPO l'ultimo passo: vista e topo con lo stesso timbro = build completa sugli stessi input.
FLAME_STAMP_NAME=".identities_manifest.json"

flame_require_view() {
    if [[ ! -d "$WBES_FLAME_MESH_DIR" || ! -f "$WBES_FLAME_DIST_NPZ" || ! -d "$WBES_FLAME_TOPO_DIR" ]]; then
        echo "ERRORE: dati FLAME assenti (WBES_FLAME_MESH_DIR=$WBES_FLAME_MESH_DIR," >&2
        echo "  WBES_FLAME_DIST_NPZ=$WBES_FLAME_DIST_NPZ). Lancia prima aau/flame/flame_build.sbatch." >&2
        return 1
    fi
    if ! cmp -s "$WBES_FLAME_TOPO_DIR/$FLAME_STAMP_NAME" "$WBES_FLAME_VIEW_DIR/$FLAME_STAMP_NAME"; then
        echo "ERRORE: timbro della vista assente o diverso da quello delle topologie: build" >&2
        echo "  interrotta o in corso. Rilancia aau/flame/flame_build.sbatch." >&2
        return 1
    fi
}

# flame_fingerprint: 12 caratteri dello sha256 del timbro. Entra nel nome delle dir di output
# ($WBES_FLAME_RUNS/<impronta>/...) e nella data dir su /tmp, che eval_common.sh mette nel suo
# hash e in eval_key.txt: dati rigenerati con altri input = output nuovi, mai riuso di matrici
# o sentinelle .done vecchie (alignment_matrix.py e rank_from_matrix.py saltano le matrici
# gia' su disco, e lo fanno per nome di file).
flame_fingerprint() {
    sha256sum "$WBES_FLAME_TOPO_DIR/$FLAME_STAMP_NAME" | cut -c1-12
}

# flame_stage_ops <dest>
# Operatori DiffusionNet (v2_work/potential/areanorm_operators.py, k_eig 128, area unitaria:
# quelli dei modelli valutati) di tutte le topologie di $WBES_FLAME_TOPO_DIR in <dest>/withops,
# e la vista idNNNN di make_train_ready.py su di essi in <dest>/view. <dest> sta su /tmp: il
# chiamante lo cancella con un trap EXIT. Sharding di aau/ict/ict_ops.sbatch: stride sulla lista
# ordinata, n coprimo con 6 (i nomi sono identita' x 6 topologie).
flame_stage_ops() {
    local dest="$1" n="${SLURM_CPUS_PER_TASK:-8}" i rc=0 n_in n_out
    local pids=()
    if (( n % 6 == 0 )); then
        n=$(( n - 1 ))
    fi
    rm -rf "$dest"
    mkdir -p "$dest/withops" "$dest/logs"
    n_in=$(find "$WBES_FLAME_TOPO_DIR" -maxdepth 1 -name "flame*_GTready_*.npz" | wc -l)
    echo "[flame-ops] $n_in topologie -> $dest/withops, $n shard"
    for (( i = 0; i < n; i++ )); do
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
          --input-dir "$WBES_FLAME_TOPO_DIR" --output-dir "$dest/withops" \
          --k-eig 128 --shard "$i/$n" > "$dest/logs/shard$i.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || rc=1
    done
    n_out=$(find "$dest/withops" -maxdepth 1 -name "flame*_GTready_*.npz" | wc -l)
    echo "[flame-ops] operatori: $n_out / $n_in, $(du -sh "$dest/withops" | cut -f1) su /tmp"
    if (( rc != 0 || n_out != n_in )); then
        echo "ERRORE: operatori incompleti; coda dei log degli shard:" >&2
        tail -n 3 "$dest"/logs/shard*.log >&2
        return 1
    fi
    "$AAU_DIR/run.sh" v2_work/genflame/make_train_ready.py \
        --withops-dir "$dest/withops" \
        --gt-npz "$WBES_FLAME_GT_DIR/flame_matrix_distances_maxabs.npz" \
        --out-dir "$dest/view" --id-offset "$WBES_FLAME_ID_OFFSET" > "$dest/logs/view.log"
    # make_train_ready scrive anche <dest>/view/gt_matrix.npz: gli eval NON la usano, leggono
    # $WBES_FLAME_DIST_NPZ in home (stessi dati, stesso script; percorso stabile per l'hash).
}
