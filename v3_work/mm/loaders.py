"""Un caricatore per 3DMM, tutti verso ``MorphableModel`` (``model.py``).

Dove c'e' gia' un loader nel repo lo si importa e non lo si riscrive: ``ict_model``
(v2_work/genict), ``flame_model`` (v2_work/genflame), ``hifi_model`` / ``fv_model`` /
``gnm_model`` (aau/zs3dmm). Nuovi solo BFM 2019, BFM 3DDFA e FaceScape. I fatti marcati
"misurato" vengono da ``aau/scratch/d1_mm/probe_all.py``, ``probe2.py`` e
``inspect_bfm2019_npz.py`` (8 ottobre) e li ricontrolla ``selftest.py``.

Modello           ruolo  regione                                 k id / k espr  unita'  frame (sx, alto, avanti; normali)
bfm2019           train  mesh di model2019_bfm (47.439 v.)       199 / 100 pca  mm      +x +y +z; uscenti
bfm2019_face12    train  mesh di model2019_face12 (27.657 v.)    199 / 100 pca  mm      +x +y +z; uscenti
bfm2019_fullhead  train  mesh di model2019_fullHead (58.203 v.)  199 / 100 pca  mm      +x +y +z; uscenti
bfm3ddfa          train  crop p23470 dei dati REMESH             40 / 10 pca    um      +x +y +z; INTERNE
ict               train  geometria #0 "Face" del README          100 / 53 bs    cm      +x +y +z; uscenti
gnm               train  hockey_mask, quad a ventaglio           170 / 383 pca  m       +x +y +z; uscenti
flame2020         train  maschera "face" di FLAME_masks.pkl      300 / 100 pca  m       +x +y +z; uscenti
flame2023         train  idem (FLAME 2023 Open)                  300 / 100 pca  m       +x +y +z; uscenti
facescape         dev    fv_indices_front del toolkit v1.6       300 / 51 bs    mm      +x +y +z; uscenti
facescape50       dev    idem, file 50_52_id_exp                 50 / 51 bs     mm      +x +y +z; uscenti
hifi3d            test   mask_face del .mat                      500 / 199 pca  cm (*)  +x +y +z; uscenti
faceverse         test   mesh piena                              150 / 52 bs    (**)    +x -y -z; uscenti
(*) stimato dalla bbox della patch; (**) unita' proprie, 162.6 mm per unita' stimati dagli angoli
esterni degli occhi (90 mm).

Percorsi: ``WBES_MM_<NOME>`` (p.es. ``WBES_MM_FACESCAPE``) sovrascrive il default. NON si leggono
le variabili dei domini zero-shot (``WBES_FV_NPY`` & co.): ``dev_facescape_env.sh`` ridirige
proprio quelle sui dati FaceScape.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from .model import BilinearModel, ExpressionSpace, Frame, MorphableModel

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "aau" / "zs3dmm", REPO_ROOT / "v2_work" / "genict", REPO_ROOT / "v2_work" / "genflame"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mesh_ops as mo  # noqa: E402  (genict: igl, niente open3d)

HOME_DATA = Path.home() / "data"
BFM2019_DIR = REPO_ROOT / "external_data" / "bfm2019"
DEFAULT_PATHS = {
    "bfm2019": BFM2019_DIR / "model2019_bfm.npz",
    "bfm2019_face12": BFM2019_DIR / "model2019_face12.npz",
    "bfm2019_fullhead": BFM2019_DIR / "model2019_fullHead.npz",
    "bfm3ddfa": REPO_ROOT / "external" / "3DDFA_v1" / "train.configs",
    "ict": REPO_ROOT / "external" / "ICT-FaceKit" / "FaceXModel",
    "gnm": HOME_DATA / "gnm_head" / "gnm_head.npz",
    "flame2020": REPO_ROOT / "v2_work" / "genflame" / "official" / "FLAME2020" / "generic_model.pkl",
    "flame2023": REPO_ROOT / "v2_work" / "genflame" / "official" / "FLAME2023Open" / "flame2023_Open.pkl",
    "flame_masks": REPO_ROOT / "v2_work" / "genflame" / "official" / "FLAME_masks.pkl",
    "facescape": HOME_DATA / "facescape_bilinear" / "facescape_bm_v1.6_847_300_52_id.npz",
    "facescape50": HOME_DATA / "facescape_bilinear" / "facescape_bm_v1.6_847_50_52_id_exp.npz",
    "hifi3d": HOME_DATA / "hifi3d" / "files" / "AI-NEXT-Shape.mat",
    "faceverse": HOME_DATA / "faceverse" / "faceverse_simple_v2.npy",
}
ROLES = {"bfm2019": "train", "bfm2019_face12": "train", "bfm2019_fullhead": "train", "bfm3ddfa": "train",
         "ict": "train", "gnm": "train", "flame2020": "train", "flame2023": "train",
         "facescape": "dev", "facescape50": "dev", "hifi3d": "test", "faceverse": "test"}
ALIASES = {"bfm2019": ("BFM2019", "bfm2019_bfm"), "bfm2019_face12": ("BFM2019_face12",),
           "bfm2019_fullhead": ("BFM2019_fullHead", "bfm2019_fullHead"),
           # NON "bfm": in canonical_transforms.json "bfm" e' il frame dei dati REMESH (y in basso, naso -z),
           # non quello del modello 3DDFA (+y alto, +z naso) su cui la libreria costruisce le mesh
           "bfm3ddfa": ("bfm_3ddfa", "BFM3DDFA"), "ict": ("ICT", "ict_facekit"), "gnm": ("GNM", "gnm_head"),
           # flame2023 e facescape50 prendono la voce JSON del gemello: stesso frame, stesse unita'
           # (FLAME: m -> mm puro; FaceScape: medie neutre dei due file uguali entro 0.001 mm)
           "flame2020": ("flame", "FLAME2020"), "flame2023": ("FLAME2023", "flame2023_open", "flame"),
           "facescape": ("facescape300", "FaceScape"), "facescape50": ("FaceScape50", "facescape"),
           "hifi3d": ("hifi", "HIFI3D"), "faceverse": ("fv", "FaceVerse")}

# Sigma dei coefficienti d'espressione pca (in unita' di sigma del modo). Regola unica, quella di
# aau/distill/gen_gnm_shard.py --calibrate: griglia (0.25, 0.5, 0.75, 1, 1.5, 2), mediana su 50
# identita' dello spostamento medio per vertice dopo maxabs, valore piu' vicino alla ricetta ICT
# (0.017). GNM 0.5 e' quello dei dati di training (datasets/GNM_DISTILL/shards/calibration.json);
# gli altri: selftest.py --calibrate (job 1061652, che su GNM ritrova 0.5: mediana 0.0187), mediane
# 0.0182 (BFM, 1.0), 0.0156 (FLAME 2020, 0.75), 0.0188 (FLAME 2023, 1.0), 0.0183 (HIFI3D, 0.5).
# BFM 2019 (job 1061712, s scala anche la deformazione media): 0.5 per tutte e tre le varianti,
# mediane 0.0176 (bfm), 0.0211 (face12), 0.0175 (fullHead).
EXPR_SCALE = {"bfm2019": 0.5, "bfm2019_face12": 0.5, "bfm2019_fullhead": 0.5, "gnm": 0.5, "bfm3ddfa": 1.0,
              "flame2020": 0.75, "flame2023": 1.0, "hifi3d": 0.5}
GAZE_PREFIX = "eyeLook"
GNM_POOL_GROUPS = ("lower_face_region", "left_eye_region", "right_eye_region")   # make_zs_expr_topologies
FACESCAPE_PATCH = "fv_indices_front"
EYE_OUTER_REFERENCE_MM = 90.0

ICT_LMK68 = (1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443,
             268, 493, 1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 4857, 1140, 2075, 1147,
             4269, 3360, 1507, 1542, 1537, 1528, 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695,
             2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841, 5702, 5711, 5533, 6216, 6207, 6470, 5517,
             5966)  # README di ICT-FaceKit, "Multi-PIE 68 point facial landmarks", base 0
BFM_LMK_JSON = REPO_ROOT / "WBES" / "utils" / "BFM-p23470.json"     # 51 iBUG (senza mandibola) sul crop p23470
FLAME_LMK_JSON = REPO_ROOT / "WBES" / "utils" / "FLAME-face.json"   # 51 iBUG sulla regione "face"
BFM_CROP = REPO_ROOT / "WBES" / "utils" / "ix_23470_relative_to_53215.txt"
BFM_REMESH_ORIGINALS = REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500"


def model_path(name: str) -> Path:
    env = os.environ.get(f"WBES_MM_{name.upper()}", "").strip()
    path = Path(env) if env else DEFAULT_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"{name}: file del modello assente: {path} (WBES_MM_{name.upper()})")
    return path


def _compact(F_full: np.ndarray, n_full: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertici usati (ordine nativo) e triangoli rimappati."""
    used = np.unique(F_full)
    remap = -np.ones(n_full, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return used, remap[F_full].astype(np.int32)


def _rows(used: np.ndarray) -> np.ndarray:
    """Righe xyz interlacciate dei vertici ``used``."""
    return (3 * used[:, None] + np.arange(3)[None, :]).ravel()


def _pool_without_gaze(names: list[str]) -> np.ndarray:
    gaze = [n for n in names if n.startswith(GAZE_PREFIX)]
    if len(gaze) != 8:
        raise ValueError(f"attesi 8 blendshape {GAZE_PREFIX}*, trovati {gaze}")
    return np.asarray([i for i, n in enumerate(names) if not n.startswith(GAZE_PREFIX)])


def _lmk_in_patch(lmk_native: np.ndarray, used: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(indici nella patch, numero del landmark) dei soli landmark che cadono nella patch."""
    remap = {int(v): i for i, v in enumerate(used)}
    keep = [(remap[int(v)], k) for k, v in enumerate(lmk_native) if int(v) in remap]
    return np.asarray([a for a, _ in keep]), np.asarray([b for _, b in keep])


# ----------------------------------------------------------------------------- BFM 2019

# Landmark con nome del file (metadata/landmarks/json) che hanno un corrispondente iBUG 68.
BFM2019_IBUG = {"center.chin.tip": 8, "center.nose.tip": 30, "right.eye.corner_outer": 36,
                "right.eye.corner_inner": 39, "left.eye.corner_inner": 42, "left.eye.corner_outer": 45,
                "right.lips.corner": 48, "center.lips.upper.outer": 51, "left.lips.corner": 54,
                "center.lips.lower.outer": 57, "center.lips.upper.inner": 62, "center.lips.lower.inner": 66}


def load_bfm2019(name: str = "bfm2019") -> MorphableModel:
    """Basel Face Model 2019 (Statismo HDF5), letto dall'npz di ``bfm2019_convert.py``.

    Tre file, stessa parametrizzazione: ``model2019_bfm`` (principale, la maschera del BFM originale
    con orecchie: 47.439 vertici, 94.464 triangoli), ``model2019_face12`` (solo volto: 27.657 /
    55.040) e ``model2019_fullHead`` (testa intera: 58.203 / 116.160), per la variazione di
    supporto. Misurato (aau/scratch/d1_mm/inspect_bfm2019_npz.py): ``pcaBasis`` ortonormale
    (gram fuori diagonale < 1e-9), quindi base a +1 sigma = ``pcaBasis * sqrt(pcaVariance)``;
    199 modi d'identita', 100 d'espressione; mm; +x sinistra del soggetto (i landmark ``right.*``
    stanno a -x), +y alto, +z naso; normali uscenti (91% dei triangoli). La regione e' la mesh
    del file com'e'.

    Espressione: il modello d'espressione ha una media NON nulla (deformazione media delle
    espressioni di training, fino a 16.7 mm), mentre la neutra e' ``shape/model/mean`` senza
    deformazione. La media entra come coefficiente 0 della base, fisso a ``s`` in ogni espressione
    campionata (``ExpressionSpace.fixed``), e i 100 modi seguono N(0, s^2) in unita' di sigma: ``s``
    scala tutto il prior N(media, Sigma) verso la neutra, cioe' e' l'intensita' dell'espressione
    (la sola media sposta gia' 0.022-0.029 dopo maxabs, oltre lo 0.017 della taratura). Cosi'
    ``mesh(z)`` resta la neutra e ``mesh`` resta lineare nei coefficienti.
    Landmark: i ``metadata/landmarks/json`` (coordinate) portati sul vertice piu' vicino della
    media; ``landmark_ids`` = numero iBUG dove esiste (``BFM2019_IBUG``), -1 altrove.
    """
    path = model_path(name)
    if path.suffix != ".npz":
        raise ValueError(f"{name}: serve l'npz di bfm2019_convert.py, non {path.name}")
    with np.load(path) as z:
        mean = z["shape__model__mean"].astype(np.float64).reshape(-1, 3)
        F = z["shape__representer__cells"].T.astype(np.int64)
        B = z["shape__model__pcaBasis"].astype(np.float64)
        sig = np.sqrt(z["shape__model__pcaVariance"].astype(np.float64))
        E = z["expression__model__pcaBasis"].astype(np.float64)
        sig_e = np.sqrt(z["expression__model__pcaVariance"].astype(np.float64))
        e_mean = z["expression__model__mean"].astype(np.float64).reshape(-1, 1)
        if not np.array_equal(z["expression__representer__cells"], z["shape__representer__cells"]):
            raise ValueError(f"{path.name}: triangoli diversi fra forma ed espressione")
        lm = json.loads(str(z["metadata__landmarks__json"].ravel()[0]))
    n = len(mean)
    norms = np.linalg.norm(B, axis=0)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(f"{path.name}: pcaBasis non ortonormale (norme {norms[:3]})")
    id_basis = (B * sig).reshape(n, 3, -1)
    ex_basis = np.concatenate([e_mean, E * sig_e], axis=1).reshape(n, 3, -1)
    names = [l["id"] for l in lm]
    P = np.asarray([l["coordinates"] for l in lm], dtype=np.float64)
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(mean).query(P)
    return MorphableModel(
        name=name, role=ROLES[name], faces=F, mean=mean, id_basis=id_basis, id_sigma=sig,
        expr=ExpressionSpace("pca", ["mean_expression"] + [f"exp{i:03d}" for i in range(E.shape[1])],
                             np.arange(1, E.shape[1] + 1), basis=ex_basis, scale=EXPR_SCALE[name],
                             fixed={0: 1.0}, note="coefficiente 0 = deformazione media, fisso a s"),
        frame=Frame("+x", "+y", "+z", "outward", "mm", 1.0, "misurato: landmark con nome sulla media"),
        landmarks=idx, landmark_ids=np.asarray([BFM2019_IBUG.get(nm, -1) for nm in names]),
        landmark_scheme="bfm2019", landmark_names=names, region=f"mesh di {path.stem}.h5",
        region_vertices=np.arange(n),
        info={"file": str(path), "n_verts": n, "n_faces": int(len(F)), "n_id": int(id_basis.shape[2]),
              "n_expr_pca": int(E.shape[1]), "landmark_snap_mm": {"median": float(np.median(dist)),
                                                                  "max": float(dist.max())}},
        aliases=ALIASES[name])


# ----------------------------------------------------------------------------- BFM 3DDFA

def load_bfm3ddfa(name: str = "bfm3ddfa") -> MorphableModel:
    """BFM di 3DDFA v1 (``train.configs``): media ``u_shp + u_exp``, 40 modi ``w_shp_sim``, 10 ``w_exp_sim``.

    Misurato: xyz interlacciati; ``w_shp_sim`` ortonormale (norme 0.993-1.000, gram fuori
    diagonale < 0.01), quindi sigma = ``param_std[12:52]`` di ``param_whitening.pkl`` (deviazione
    dei coefficienti stimati da 3DDFA su 300W-LP: il prior empirico; la loro media, 0.5 sigma sul
    primo modo, e' un bias dei fit e non si usa). ``w_exp_sim`` non e' ortonormale (norme ~2e5) e
    ``param_std[52:62]`` vale ~0.15-1.5: base d'espressione = ``w_exp_sim`` x quella sigma.
    Unita' micrometri (angoli esterni degli occhi a 86 mm). Regione: il crop p23470
    (``WBES/utils/ix_23470_relative_to_53215.txt``) dei dati REMESH, coi triangoli delle loro
    ``original``: contengono tutti i 46.326 triangoli di ``visualize/tri.mat`` ristretti al crop,
    stesso verso, piu' 114 che chiudono la fessura della bocca. Senza i dati REMESH si usano
    quelli di ``tri.mat``. Verso INTERNO (3% dei triangoli uscenti), come i dati BFM di training.
    """
    import scipy.io
    cfg = model_path(name)
    u = (np.load(cfg / "u_shp.npy") + np.load(cfg / "u_exp.npy")).astype(np.float64).ravel()
    W_id = np.load(cfg / "w_shp_sim.npy").astype(np.float64)
    W_ex = np.load(cfg / "w_exp_sim.npy").astype(np.float64)
    with open(cfg / "param_whitening.pkl", "rb") as fh:
        pw = pickle.load(fh, encoding="latin1")
    std = np.asarray(pw["param_std"], dtype=np.float64)
    sig_id, sig_ex = std[12:52], std[52:62]
    n_full = u.size // 3

    ix = np.loadtxt(BFM_CROP, dtype=np.int64)
    T = np.asarray(scipy.io.loadmat(cfg.parent / "visualize" / "tri.mat")["tri"]).T.astype(np.int64) - 1
    inv = -np.ones(n_full, dtype=np.int64)
    inv[ix] = np.arange(len(ix))
    T_crop = inv[T][(inv[T] >= 0).all(axis=1)]
    originals = sorted(BFM_REMESH_ORIGINALS.glob("id*_GTready_original.npz"))
    if originals:
        with np.load(originals[0]) as d:
            F = np.asarray(d["F"], dtype=np.int64)
        oriented = {tuple(t) for t in F.tolist()} | {(b, c, a) for a, b, c in F.tolist()} \
            | {(c, a, b) for a, b, c in F.tolist()}
        missing = sum(1 for t in T_crop.tolist() if tuple(t) not in oriented)
        if missing:
            raise ValueError(f"{originals[0].name}: {missing} triangoli di tri.mat assenti o capovolti")
        faces_source = f"{originals[0]} (contiene i {len(T_crop)} di tri.mat, stesso verso)"
    else:
        F, faces_source = T_crop, "external/3DDFA_v1/visualize/tri.mat ristretto al crop (bocca aperta)"
    if F.max() != len(ix) - 1:
        raise ValueError("triangoli del crop BFM non compatti")

    rows = _rows(ix)
    mean = u[rows].reshape(-1, 3)
    id_basis = (W_id[rows] * sig_id).reshape(len(ix), 3, -1)
    ex_basis = (W_ex[rows] * sig_ex).reshape(len(ix), 3, -1)
    js = json.loads(BFM_LMK_JSON.read_text())
    if js["Npoints"] != len(ix):
        raise ValueError(f"{BFM_LMK_JSON.name}: {js['Npoints']} punti, crop da {len(ix)}")
    return MorphableModel(
        name=name, role=ROLES[name], faces=F, mean=mean, id_basis=id_basis, id_sigma=sig_id,
        expr=ExpressionSpace("pca", [f"exp{i:02d}" for i in range(ex_basis.shape[2])], np.arange(ex_basis.shape[2]),
                             basis=ex_basis, scale=EXPR_SCALE[name],
                             note="w_exp_sim x param_std[52:62]; media u_exp = espressione 0"),
        frame=Frame("+x", "+y", "+z", "inward", "um", 1e-3, "misurato: landmark 68 di keypoints_sim sulla media"),
        landmarks=np.asarray(js["lmk_indices"]), landmark_ids=np.arange(17, 68), landmark_scheme="ibug51",
        region="crop p23470 (dati REMESH)", region_vertices=ix,
        info={"file": str(cfg), "faces_source": faces_source,
              "param_mean_id_first3_not_used": np.asarray(pw["param_mean"])[12:15].tolist()},
        aliases=ALIASES[name])


# ----------------------------------------------------------------------------- ICT-FaceKit

def load_ict(name: str = "ict") -> MorphableModel:
    """ICT-FaceKit light: ``v2_work/genict/ict_model.load_ict`` con tutti i 53 blendshape.

    Regione: geometria #0 "Face" (9409 vertici, quad spezzati in due). Pool d'espressione: i 53
    meno gli 8 ``eyeLook*`` (sguardo, WS5). Unita' ~cm (angoli esterni degli occhi a 8.92).
    """
    from ict_model import load_ict as _load
    md = model_path(name)
    names = json.loads((md / "vertex_indices.json").read_text())["expressions"]
    m = _load(md, n_shape=100, expressions=tuple(names))
    ex = np.stack([m["exprdirs"][n] for n in names], axis=-1)
    return MorphableModel(
        name=name, role=ROLES[name], faces=m["f"], mean=m["v_template"], id_basis=m["shapedirs"],
        id_sigma=np.ones(m["shapedirs"].shape[2]),
        expr=ExpressionSpace("blendshape", list(names), _pool_without_gaze(names), basis=ex,
                             note="ricetta WS5, pool senza eyeLook*"),
        frame=Frame("+x", "+y", "+z", "outward", "cm", 10.0, "misurato: landmark 68 del README"),
        landmarks=np.asarray(ICT_LMK68), landmark_ids=np.arange(68), landmark_scheme="ibug68",
        region="geometria #0 Face del README", region_vertices=np.arange(len(m["v_template"])),
        info={"file": str(md), "identity": "deltas identityNNN - neutro, gia' a +1 sigma (randomize_identity)"},
        aliases=ALIASES[name])


# ----------------------------------------------------------------------------- GNM Head

def load_gnm(name: str = "gnm") -> MorphableModel:
    """GNM Head v3.0: ``aau/zs3dmm/gnm_model`` (hockey_mask, quad a ventaglio, posa neutra verificata).

    Espressione: la base PCA (383) col pool di ``make_zs_expr_topologies`` (350: parte bassa del
    volto e regioni degli occhi; lingua e pupille a zero), sigma 0.5 dei dati di training. Nessun
    landmark nel file; il verso destra/sinistra si controlla qui sul gruppo ``left``.
    """
    import gnm_model
    path = model_path(name)
    m = gnm_model.load_gnm(path)
    ex = gnm_model.load_gnm_expressions(path)
    used = m["face_vertices"].astype(np.int64)
    _, F = gnm_model.face_patch(m["v_template"], m)
    with np.load(path) as z:
        groups = [str(n) for n in z["vertex_group_names"]]
        left = z["vertex_groups"][groups.index("left")] > 1e-4
        nv = len(z["template_vertex_positions"])
    if not m["v_template"][:nv][left, 0].mean() > 0:
        raise ValueError("GNM: il gruppo 'left' non sta a +x")
    pool = np.asarray([i for g in GNM_POOL_GROUPS for i in ex["groups"][g]])
    return MorphableModel(
        name=name, role=ROLES[name], faces=F, mean=m["v_template"][used], id_basis=m["shapedirs"][used],
        id_sigma=np.ones(m["shapedirs"].shape[2]),
        expr=ExpressionSpace("pca", ex["names"], pool, basis=ex["exprdirs"][used], scale=EXPR_SCALE[name],
                             note="pool di 350 (lower_face, occhi); sigma 0.5 = GNM_DISTILL"),
        frame=Frame("+x", "+y", "+z", "outward", "m", 1000.0, "gnm_model (+Y alto, +Z naso), gruppo 'left' a +x"),
        region="hockey_mask, quad a ventaglio dal centro", region_vertices=used,
        info={"file": str(path), **{k: m["info"][k] for k in ("n_head_basis_in_file", "pose_neutral_max_dev_m")}},
        aliases=ALIASES[name])


# ----------------------------------------------------------------------------- FLAME

def load_flame(name: str = "flame2020") -> MorphableModel:
    """FLAME 2020 / 2023 Open: ``v2_work/genflame/flame_model.load_flame`` (pickle senza chumpy).

    ``shapedirs`` (5023, 3, 400): 300 identita' poi 100 espressioni, gia' a +1 sigma. Regione: la
    maschera "face" di ``FLAME_masks.pkl``, triangoli con i tre vertici dentro, componente piu'
    grande (``aau/flame/flame_mask.crop_faces``): 1787 vertici, 3408 triangoli, quelli del set
    zero-shot FLAME. Landmark: i 51 di ``WBES/utils/FLAME-face.json`` sulla stessa regione.
    """
    from flame_model import load_flame as _load
    path = model_path(name)
    m = _load(str(path))
    with open(model_path("flame_masks"), "rb") as fh:
        masks = pickle.load(fh, encoding="latin1")
    n_full = len(m["v_template"])
    keep = np.zeros(n_full, dtype=bool)
    keep[np.asarray(masks["face"], dtype=np.int64)] = True
    F_full = mo.largest_component(m["f"][keep[m["f"]].all(axis=1)])
    used, F = _compact(F_full, n_full)
    js = json.loads(FLAME_LMK_JSON.read_text())
    if js["Npoints"] != len(used):
        raise ValueError(f"{FLAME_LMK_JSON.name}: {js['Npoints']} punti, regione da {len(used)}")
    return MorphableModel(
        name=name, role=ROLES[name], faces=F, mean=m["v_template"][used], id_basis=m["shapedirs"][used, :, :300],
        id_sigma=np.ones(300),
        expr=ExpressionSpace("pca", [f"exp{i:03d}" for i in range(100)], np.arange(100),
                             basis=m["shapedirs"][used, :, 300:400], scale=EXPR_SCALE[name]),
        frame=Frame("+x", "+y", "+z", "outward", "m", 1000.0, "misurato: landmark 51 sulla media"),
        landmarks=np.asarray(js["lmk_indices"]), landmark_ids=np.arange(17, 68), landmark_scheme="ibug51",
        region="maschera face di FLAME_masks.pkl", region_vertices=used,
        info={"file": str(path), "masks": str(model_path("flame_masks"))}, aliases=ALIASES[name])


# ----------------------------------------------------------------------------- FaceScape

def _cache_dir() -> Path:
    return Path(os.environ.get("WBES_MM_CACHE", Path.home() / ".cache" / "wbes_mm"))


def _facescape50_expr_mix(path50: Path, core50: np.ndarray, id_mean50_0: float) -> tuple[np.ndarray, dict]:
    """U_exp^T del file 50_52_id_exp: core50[:, :, 0] = +-core300[:, :, 0] @ U (residuo 2e-7, misurato).

    Il file 50 ha la PCA anche sulle espressioni: la neutra non e' un asse, e' la riga 0 di U.
    U si ricava UNA volta dal file 300 (che serve solo per questo) e si mette in cache fuori dal
    repo (``WBES_MM_CACHE``, default ~/.cache/wbes_mm): sono dati derivati da FaceScape. Il
    segno: il primo fattore d'identita' (la forma media, 1/sqrt(847)) ha segno opposto nei due
    file (``id_mean[0]`` -0.0344 nel 300, +0.0344 nel 50), e il minimo quadrato lo scarica su U;
    senza la correzione la neutra del file 50 esce capovolta (selftest, job 1061653)."""
    p300 = model_path("facescape")
    key = f"facescape50_umix_signed_{path50.stat().st_size}_{p300.stat().st_size}.npz"
    cache = _cache_dir() / key
    if cache.is_file():
        with np.load(cache) as c:
            return c["mix"], {"expr_mix_cache": str(cache), "expr_mix_rel_residual": float(c["residual"])}
    with np.load(p300) as z:
        A = np.asarray(z["shape_bm_core"][:, :, 0], dtype=np.float64)
        sign = float(np.sign(z["id_mean"][0]) * np.sign(id_mean50_0))
    B = np.asarray(core50[:, :, 0], dtype=np.float64)
    U, *_ = np.linalg.lstsq(A, sign * B, rcond=None)
    residual = float(np.linalg.norm(A @ U - sign * B) / np.linalg.norm(B))
    if residual > 1e-4 or np.abs(U @ U.T - np.eye(len(U))).max() > 1e-3:
        raise ValueError(f"FaceScape 50: U_exp non ortogonale o residuo {residual:.2e}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, mix=U.T, residual=residual)
    return U.T, {"expr_mix_cache": str(cache), "expr_mix_rel_residual": residual}


def load_facescape(name: str = "facescape") -> BilinearModel:
    """FaceScape bilineare v1.6, ``shape_bm_core`` (3 x 26278, 52 espressioni, k identita').

    Forward del toolkit (``facescape_bm.gen_full``): ``core . id . exp`` col core convertito in
    residui lungo le espressioni. Misurato sul file 300: le fette grezze ``core[:, e, :]`` sono
    forme ASSOLUTE (la 0 e' la neutra, le altre ne distano 0.05-4.4 mm in media), quindi i 51 pesi
    d'espressione sono residui contro la neutra. Identita': i fattori di Tucker non sono centrati
    (``id_mean``, la prima componente vale 1/sqrt(847)) e hanno varianza ``id_var`` (= 1/847 tranne
    la prima): z = (w - id_mean) / sqrt(id_var) ~ N(0, 1) e' il prior (il campionatore del
    toolkit e' ``np.random.normal(id_mean, sqrt(id_var))``). Unita' mm, +y alto, +z naso
    (angoli esterni degli occhi a 88 mm). Regione: ``fv_indices_front`` (base 1), il volto dalla
    fronte al mento senza orecchie, occhi aperti: 12.596 vertici, 24.765 triangoli, una
    componente, tutti i 68 landmark ``lm_list_v16`` dentro. Il core si tiene solo per la patch
    (2.4 GB float32 col file 300; il caricamento ne legge 4.9 e impiega ~75 s).

    Pool d'espressione: tutti i 51 blendshape. Il file non ha i nomi, quindi l'esclusione dello
    sguardo di ICT e FaceVerse qui non si puo' fare (blendshape 1-19 spostano 0.1-0.4 mm in media
    sulla regione, la bocca aperta 4.4).
    """
    path = model_path(name)
    with np.load(path, allow_pickle=True) as z:
        n_full = int(z["vert_num"])
        F_front = z[FACESCAPE_PATCH].astype(np.int64) - 1
        core_full = z["shape_bm_core"]                              # (3N, 52, k) float32
        id_mean = z["id_mean"].astype(np.float64)
        id_sigma = np.sqrt(z["id_var"].astype(np.float64))
        lm_native = z["lm_list_v16"].astype(np.int64)
    if F_front.min() < 0 or F_front.max() != n_full - 1:
        raise ValueError(f"{FACESCAPE_PATCH}: indici in [{F_front.min()}, {F_front.max()}], attesi base 1")
    used, F = _compact(mo.largest_component(F_front), n_full)
    core = np.ascontiguousarray(core_full[_rows(used)])            # (3n, 52, k)
    info = {"file": str(path), "core_shape_file": list(core_full.shape)}
    if core.shape[1] != 52:
        raise ValueError(f"shape_bm_core {core_full.shape}: attese 52 espressioni")
    if name == "facescape50":
        mix, extra = _facescape50_expr_mix(path, core_full, float(id_mean[0]))
        info.update(extra)
    else:
        mix = np.eye(52)
    del core_full
    # (3n, k): la neutra, in float32 come il core (una copia float64 del core sarebbe 4.7 GB)
    C_n = np.tensordot(core, mix[:, 0].astype(np.float32), axes=([1], [0])).astype(np.float64)
    n = len(used)
    lmk, lmk_ids = _lmk_in_patch(lm_native, used)
    if len(lmk) != 68:
        raise ValueError(f"landmark fuori dalla regione: {68 - len(lmk)}")
    model = BilinearModel(
        name=name, role=ROLES[name], faces=F, mean=(C_n @ id_mean).reshape(n, 3),
        id_basis=(C_n * id_sigma).reshape(n, 3, -1), id_sigma=id_sigma,
        expr=ExpressionSpace("blendshape", [f"bs{k:02d}" for k in range(1, 52)], np.arange(51),
                             note="ricetta WS5 sui 51 blendshape (nessun nome nel file: niente esclusione dello sguardo)"),
        frame=Frame("+x", "+y", "+z", "outward", "mm", 1.0, "misurato: landmark 68 lm_list_v16 sulla media"),
        landmarks=lmk, landmark_ids=lmk_ids, landmark_scheme="ibug68", id_mean_coef=id_mean,
        region=f"{FACESCAPE_PATCH} (toolkit FaceScape v1.6)", region_vertices=used, info=info,
        aliases=ALIASES[name])
    model.core = core
    model.expr_mix = mix
    return model


# ----------------------------------------------------------------------------- HIFI3D, FaceVerse

def load_hifi3d(name: str = "hifi3d") -> MorphableModel:
    """HIFI3D (AI-NExT-Shape.mat): ``aau/zs3dmm/hifi_model`` (regione ``mask_face``, 500 modi).

    Espressione: ``basis_exp`` (199 modi, norme decrescenti 37.6 ... 0.19: gia' scalata come
    ``basis_shape``), stesso layout dei vertici. Landmark: ``keypoints`` (86, schema HIFI3D,
    base 0: 78 dentro la patch, gli altri sul contorno fuori). Unita' stimate cm (patch 15.9 x
    17.2 x 10.8, contro 15.1 x 17.9 x 11.5 cm della regione FLAME); frame ICT (zs_identities.py).
    """
    import hifi_model
    path = model_path(name)
    m = hifi_model.load_hifi(path)
    mat = hifi_model.read_mat(path)
    E = np.asarray(mat["basis_exp"], dtype=np.float64)
    if E.shape[1] != 3 * len(m["v_template"]):
        E = E.T
    ex = np.moveaxis(hifi_model._to_vertices(E, m["info"]["layout"]), 0, -1)   # (nv, 3, 199)
    used = m["face_vertices"].astype(np.int64)
    _, F = hifi_model.face_patch(m["v_template"], m)
    lmk, lmk_ids = _lmk_in_patch(np.asarray(mat["keypoints"]).ravel().astype(np.int64), used)
    return MorphableModel(
        name=name, role=ROLES[name], faces=F, mean=m["v_template"][used], id_basis=m["shapedirs"][used],
        id_sigma=np.ones(m["shapedirs"].shape[2]),
        expr=ExpressionSpace("pca", [f"exp{i:03d}" for i in range(ex.shape[2])], np.arange(ex.shape[2]),
                             basis=ex[used], scale=EXPR_SCALE[name]),
        frame=Frame("+x", "+y", "+z", "outward", "cm", 10.0, "frame ICT (zs_identities.py); cm stimati dalla bbox"),
        landmarks=lmk, landmark_ids=lmk_ids, landmark_scheme="hifi86",
        region="mask_face del .mat", region_vertices=used, info={"file": str(path)}, aliases=ALIASES[name])


def load_faceverse(name: str = "faceverse") -> MorphableModel:
    """FaceVerse v2: ``aau/zs3dmm/fv_model`` (mesh piena, 150 modi, 52 blendshape ARKit).

    Frame misurato sui 66 ``keypoints`` (i primi 48 nell'ordine iBUG): alto -y, naso -z, angolo
    esterno dell'occhio sinistro (45) a +x. Unita' proprie: angoli esterni degli occhi a 0.554,
    quindi 90 / 0.554 = 162.6 mm per unita' (stima). Pool d'espressione: i 52 meno gli 8
    ``eyeLook*``, come il set FaceVerse con espressioni.
    """
    import fv_model
    path = model_path(name)
    m = fv_model.load_fv(path)
    ex = fv_model.load_fv_expressions(path)
    d = np.load(path, allow_pickle=True).item()
    used = m["face_vertices"].astype(np.int64)
    _, F = fv_model.face_patch(m["v_template"], m)
    kp = np.asarray(d["keypoints"]).astype(np.int64)
    eye = float(np.linalg.norm(m["v_template"][kp[45]] - m["v_template"][kp[36]]))
    lmk, lmk_ids = _lmk_in_patch(kp, used)
    return MorphableModel(
        name=name, role=ROLES[name], faces=F, mean=m["v_template"][used], id_basis=m["shapedirs"][used],
        id_sigma=np.ones(m["shapedirs"].shape[2]),
        expr=ExpressionSpace("blendshape", ex["names"], _pool_without_gaze(ex["names"]), basis=ex["exprdirs"][used],
                             note="ricetta WS5, pool senza eyeLook* (come FACEVERSE_ZS/expr_topo)"),
        frame=Frame("+x", "-y", "-z", "outward", "faceverse", EYE_OUTER_REFERENCE_MM / eye,
                    f"misurato sui keypoints; mm per unita' = {EYE_OUTER_REFERENCE_MM:.0f} mm / {eye:.4f}"),
        landmarks=lmk, landmark_ids=lmk_ids, landmark_scheme="faceverse66",
        region="mesh piena", region_vertices=used, info={"file": str(path)}, aliases=ALIASES[name])


LOADERS = {"bfm2019": load_bfm2019, "bfm2019_face12": load_bfm2019, "bfm2019_fullhead": load_bfm2019,
           "bfm3ddfa": load_bfm3ddfa, "ict": load_ict, "gnm": load_gnm, "flame2020": load_flame,
           "flame2023": load_flame, "facescape": load_facescape, "facescape50": load_facescape,
           "hifi3d": load_hifi3d, "faceverse": load_faceverse}
