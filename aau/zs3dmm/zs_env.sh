#!/usr/bin/env bash
# Test zero-shot su 3DMM mai visti in training (WS-HIFI3D, WS-FaceVerse). Si sourcea DOPO
# aau/env.sh; il dominio arriva da WBES_ZS_DOMAIN:
#
#   WBES_ZS_DOMAIN=hifi  -> aau/zs3dmm/hifi_env.sh (WBES_HIFI_*, output aau/runs/ws_hifi3d)
#   WBES_ZS_DOMAIN=fv    -> aau/zs3dmm/fv_env.sh   (WBES_FV_*,   output aau/runs/ws_faceverse)
#   WBES_ZS_DOMAIN=gnm   -> aau/zs3dmm/gnm_env.sh  (WBES_GNM_*,  output aau/runs/ws_gnm)
#   WBES_ZS_DOMAIN=ict   -> aau/zs3dmm/ict_env.sh  (WBES_ICTZS_*, output aau/runs/ws_ictzs):
#                           controllo della pipeline sulle held-out di ICT, non un dominio nuovo
#   WBES_ZS_DOMAIN=bfm   -> aau/zs3dmm/bfm_env.sh  (WBES_BFMZS_*, output aau/runs/ws_bfm): i 100
#                           soggetti BFM standard, per il protocollo equalize-support (eqsupport_*)
#
# Il file del dominio esporta le sue WBES_<DOM>_* e riempie le ZS_* che usano gli sbatch.
# Qui le parti comuni: i tre modelli valutati e i preflight.

case "${WBES_ZS_DOMAIN:-}" in
    hifi|fv|gnm|ict|bfm) ;;
    *) echo "ERRORE: WBES_ZS_DOMAIN='${WBES_ZS_DOMAIN:-}' (hifi|fv|gnm|ict|bfm)" >&2; return 2 ;;
esac
source "$WBES_ROOT/aau/zs3dmm/${WBES_ZS_DOMAIN}_env.sh"

# Vista a supporto equalizzato (eqsupport_view.py): stessi nomi e stesso pool della vista, ogni
# topologia ritagliata sulla regione del crop del suo soggetto. La usano zs_zeroshot.sbatch e
# zs_baselines.sbatch con WBES_ZS_EQSUPPORT=1.
ZS_EQ_DIR="$ZS_ROOT/eqsupport_view"
ZS_EQ_DATA_DIR="$ZS_EQ_DIR/npz"

# WBES_ZS_EXPR=1: modalita' espressioni (GT d'identita'). Le mesh valutate vengono dalla vista con
# un'espressione casuale per mesh (make_zs_expr_topologies.py, zs_build_expr.sbatch); GT e
# identita' restano quelle NEUTRE del dominio (la vista ne copia le GT). Stessi nomi id e stesso
# pool, quindi zs_stage.py e zs_bl.py estraggono gli stessi 100 soggetti. Risultati in
# ${ZS_RUNS}_expr (aau/runs/ws_faceverse_expr, aau/runs/ws_gnm_expr). Solo fv e gnm: gli unici con
# una base d'espressione.
ZS_NEUTRAL_VIEW_DIR="$ZS_VIEW_DIR"
if [[ "${WBES_ZS_EXPR:-0}" == 1 ]]; then
    if [[ "$WBES_ZS_DOMAIN" != fv && "$WBES_ZS_DOMAIN" != gnm ]]; then
        echo "ERRORE: WBES_ZS_EXPR=1 solo con WBES_ZS_DOMAIN=fv|gnm (dato '$WBES_ZS_DOMAIN')" >&2
        return 2
    fi
    ZS_LABEL="$ZS_LABEL con espressioni"
    ZS_EXPR_TOPO_DIR="$ZS_ROOT/expr_topo"
    ZS_VIEW_DIR="$ZS_ROOT/expr_view"
    ZS_DATA_DIR="$ZS_VIEW_DIR/npz"
    ZS_DIST_NPZ="$ZS_VIEW_DIR/gt_matrix.npz"
    ZS_COEF_NPZ="$ZS_VIEW_DIR/gt_coef_matrix.npz"
    ZS_RUNS="${ZS_RUNS}_expr"
fi

# Modelli da valutare: gli stessi tre della tabella WS2 (seed 1234, ricetta v1, area unitaria).
# Non WBES_<DOM>_*: sono gli stessi per ogni dominio.
_zs_run_dir="mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d"
export WBES_ZS_CKPT_JOINT="${WBES_ZS_CKPT_JOINT:-$AAU_RUNS/x3dmm_joint_bfm_ict_s1234_1019532/$_zs_run_dir/checkpoints/best_by_xtopo_mesh_clean.pth}"
export WBES_ZS_CKPT_BFM_ONLY="${WBES_ZS_CKPT_BFM_ONLY:-$AAU_RUNS/remesh_v1recipe_areanorm_s1234_1019310/$_zs_run_dir/checkpoints/best_by_xtopo_mesh_clean.pth}"
export WBES_ZS_CKPT_ICT_ONLY="${WBES_ZS_CKPT_ICT_ONLY:-$AAU_RUNS/x3dmm_ict_only_s1234_1019531/$_zs_run_dir/checkpoints/best_by_xtopo_mesh_clean.pth}"
unset _zs_run_dir

zs_require_model() {
    if [[ ! -f "$ZS_MODEL_FILE" ]]; then
        echo "ERRORE: modello $ZS_LABEL assente: '$ZS_MODEL_FILE'" >&2
        return 1
    fi
}

zs_require_view() {
    # ZS_COEF_NPZ vuota solo per bfm (nessuna GT nei coefficienti): la usa solo zs_summarize.py.
    if [[ ! -d "$ZS_DATA_DIR" || ! -f "$ZS_DIST_NPZ" || ( -n "$ZS_COEF_NPZ" && ! -f "$ZS_COEF_NPZ" ) ]]; then
        echo "ERRORE: vista $ZS_LABEL assente (data=$ZS_DATA_DIR, gt=$ZS_DIST_NPZ)." >&2
        echo "  Lancia prima WBES_ZS_DOMAIN=$WBES_ZS_DOMAIN aau/submit.sh zs3dmm/zs_build.sbatch" >&2
        return 1
    fi
}

zs_require_eq_view() {
    if [[ ! -f "$ZS_EQ_DIR/manifest.json" || ! -d "$ZS_EQ_DATA_DIR" ]]; then
        echo "ERRORE: vista a supporto equalizzato assente ($ZS_EQ_DIR)." >&2
        echo "  Lancia prima WBES_ZS_DOMAIN=$WBES_ZS_DOMAIN aau/submit.sh zs3dmm/eqsupport_view.sbatch" >&2
        return 1
    fi
}

# Vista alternativa generica, $ZS_ROOT/<nome>_view (npz/ + manifest.json): eqsupport (eqsupport_view.py),
# canonmask, canonmask_offcenter, canonmask_identity (canonmask.py). WBES_ZS_ALTVIEW=<nome>.
zs_require_alt_view() {
    local name="$1"
    if [[ ! "$name" =~ ^[a-z_]+$ ]]; then
        echo "ERRORE: nome di vista alternativa non valido: '$name'" >&2
        return 1
    fi
    if [[ ! -f "$ZS_ROOT/${name}_view/manifest.json" || ! -d "$ZS_ROOT/${name}_view/npz" ]]; then
        echo "ERRORE: vista $ZS_ROOT/${name}_view assente o senza manifest.json" >&2
        return 1
    fi
}

# Impronta dei dati: sha1 dei manifest di identita' (file e sha256 del modello, seed, modi,
# troncamento, layout, regione), GT e vista. Entra nel percorso di TUTTI i risultati
# ($ZS_RUNS/data_<fp>/): eval_key.txt di eval_common.sh e lo skip-if-exists di
# alignment_matrix.py / rank_from_matrix.py guardano solo i percorsi, quindi dati ricostruiti
# con altri parametri negli stessi percorsi riuserebbero in silenzio i risultati vecchi.
# Con l'impronta nel percorso una ricostruzione diversa scrive altrove. summary.md resta in
# $ZS_RUNS e dice da quale data_<fp> viene.
zs_data_fp() {
    local f
    for f in "$ZS_IDENTITIES_DIR/manifest.json" "$ZS_GT_DIR/manifest.json" "$ZS_VIEW_DIR/manifest.json"; do
        if [[ ! -f "$f" ]]; then
            echo "ERRORE: manifest assente: $f" >&2
            return 1
        fi
    done
    cat "$ZS_IDENTITIES_DIR/manifest.json" "$ZS_GT_DIR/manifest.json" "$ZS_VIEW_DIR/manifest.json" \
        | sha1sum | cut -c1-10
}
