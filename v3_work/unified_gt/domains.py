"""Template medio di ogni dominio, nel frame e nelle unita' dei SUOI dati, e identita' neutre.

    template(name) -> dict(name, V, F_target, F_render, basis, units, frame, note)

  - ``V``: forma media (nv, 3) float64, frame e unita' dei dati del dominio su disco (la
    trasformazione canonica si applica a quelli);
  - ``F_target``: i triangoli della pelle su cui si fanno NICP e mappa baricentrica (la sola
    componente connessa piu' grande, niente bulbi oculari, denti, lingua);
  - ``F_render``: tutti i triangoli, per il render dei landmark (con i bulbi il detector vede occhi);
  - ``basis``: (nv, 3, k) se le identita' sono combinazioni lineari ``V + basis @ w`` (la mappa
    baricentrica e' lineare, quindi s_i = mappa(V) + mappa(basis) @ w, senza generare mesh), altrimenti
    None.

I vertici sono quelli del MODELLO intero quando le identita' sono definite dai pesi (GNM, HIFI3D: la
testa intera, non la patch del volto delle viste di eval, cosi' la copertura della regione FLAME non
dipende dal crop di ciascun protocollo); per BFM e Multiface, che esistono solo come mesh, la
topologia delle mesh.

Domini: flame, bfm, ict, gnm, facescape, hifi3d, faceverse, multiface.
"""

from __future__ import annotations

import json
import tarfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from ugt import REPO_ROOT, largest_component, umeyama, apply_sim

HOME = Path.home()
DATASETS = REPO_ROOT / "datasets"
DOMAINS = ("flame", "bfm", "ict", "gnm", "facescape", "hifi3d", "faceverse", "multiface")
TRAIN_DOMAINS = ("flame", "bfm", "ict", "gnm", "facescape")

GNM_NPZ = HOME / "data" / "gnm_head" / "gnm_head.npz"
HIFI_MAT = HOME / "data" / "hifi3d" / "files" / "AI-NEXT-Shape.mat"
HIFI_NOAUG_MAT = HOME / "data" / "hifi3d" / "files" / "AI-NEXT-Shape-NoAug.mat"
FV_NPY = HOME / "data" / "faceverse" / "faceverse_simple_v2.npy"
FS_NPZ = HOME / "data" / "facescape_bilinear" / "facescape_bm_v1.6_847_300_52_id.npz"
BFM_REMESH = DATASETS / "REMESH" / "npz_data_topo_500"
BFM_3DDFA = REPO_ROOT / "external" / "3DDFA_v1" / "train.configs"
BFM_CROP_IDX = REPO_ROOT / "WBES" / "utils" / "ix_23470_relative_to_53215.txt"
MULTIFACE_TRACKED = DATASETS / "Multiface" / "prep" / "tracked"
MULTIFACE_NEUTRAL = ("E001_Neutral_Eyes_Open", "EXP_eye_neutral")
FS_CACHE = REPO_ROOT / "datasets" / "UNIFIED_GT" / "cache" / "facescape_neutral_template.npz"


def _pack(name, V, F_target, F_render=None, basis=None, units="", frame="", note="") -> dict:
    V = np.ascontiguousarray(V, dtype=np.float64)
    F_target = np.ascontiguousarray(largest_component(np.asarray(F_target, dtype=np.int64)))
    F_render = F_target if F_render is None else np.ascontiguousarray(F_render, dtype=np.int64)
    return {"name": name, "V": V, "F_target": F_target, "F_render": F_render, "basis": basis,
            "units": units, "frame": frame, "note": note}


# ------------------------------------------------------------------------------ FLAME

def flame() -> dict:
    from flame_model import N_SHAPE, load_flame, model_path

    m = load_flame(model_path())
    return _pack("flame", m["v_template"], m["f"], m["f"], m["shapedirs"][:, :, :N_SHAPE],
                 units="metri", frame="FLAME: +y alto, +z naso",
                 note="FLAME 2020 generic_model.pkl ufficiale (v2_work/genflame/flame_model.py), 300 modi")


# -------------------------------------------------------------------------------- BFM

def bfm_original_paths() -> list[Path]:
    return sorted(BFM_REMESH.glob("id*_GTready_original.npz"))


def load_bfm_original(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as d:
        return np.asarray(d["V"], dtype=np.float64), np.asarray(d["F"], dtype=np.int64)


@lru_cache(maxsize=1)
def bfm() -> dict:
    """Media delle 500 ``original`` REMESH (crop p23470 del BFM a 53.215 vertici), nel loro frame.

    Le 500 mesh sono state allineate una per una (similarita') alla stessa mesh di riferimento,
    quindi hanno un frame comune: la loro media vertice per vertice e' il template nel frame dei
    dati. I coefficienti non esistono (data_scale/PLAN.md:155: fuori dal sottospazio 3DDFA), quindi
    niente base. Controllo: la media di 3DDFA (``u_shp``, ``u_shp + u_exp``) ritagliata col crop
    p23470 e portata sulla media con una similarita' (``bfm_3ddfa_check``).
    """
    Vs = []
    F0 = None
    for p in bfm_original_paths():
        V, F = load_bfm_original(p)
        if F0 is None:
            F0 = F
        elif not np.array_equal(F, F0):
            raise ValueError(f"{p.name}: facce diverse dalla prima original")
        Vs.append(V)
    V = np.mean(Vs, axis=0)
    return _pack("bfm", V, F0, units="micrometri (BFM)",
                 frame="frame delle original REMESH (similarita' verso una mesh di riferimento)",
                 note=f"media delle {len(Vs)} original di datasets/REMESH/npz_data_topo_500 (crop p23470)")


def bfm_3ddfa_check() -> dict:
    """RMS (unita' BFM) fra la media REMESH e la media di 3DDFA ritagliata, dopo similarita'."""
    ix = np.loadtxt(BFM_CROP_IDX, dtype=np.int64)
    V = bfm()["V"]
    out = {}
    for tag, files in (("u_shp", ("u_shp.npy",)), ("u_shp+u_exp", ("u_shp.npy", "u_exp.npy"))):
        u = sum(np.load(BFM_3DDFA / f).astype(np.float64).ravel() for f in files).reshape(-1, 3)[ix]
        s, R, t = umeyama(u, V)
        out[tag] = float(np.sqrt(((apply_sim(u, s, R, t) - V) ** 2).sum(1).mean()))
    out["template_rms_radius"] = float(np.sqrt(((V - V.mean(0)) ** 2).sum(1).mean()))
    return out


# -------------------------------------------------------------------------------- ICT

def ict() -> dict:
    from ict_model import load_ict

    m = load_ict()
    return _pack("ict", m["v_template"], m["f"], m["f"], m["shapedirs"],
                 units="ICT (circa centimetri)", frame="ICT: +y alto, +z naso",
                 note="patch Face (geometria #0, 9409 vertici) di v2_work/genict/ict_model.py, 100 modi")


# -------------------------------------------------------------------------------- GNM

@lru_cache(maxsize=1)
def _gnm_raw() -> dict:
    with np.load(GNM_NPZ) as z:
        return {k: z[k] for k in ("template_vertex_positions", "vertex_identity_basis", "identity_names",
                                  "triangles", "vertex_group_names", "vertex_groups")}


def gnm() -> dict:
    """Testa GNM intera (17.821 vertici, metri), triangoli di ``skin_exterior`` come bersaglio.

    Le identita' (GNM_DISTILL, aau/zs3dmm/gnm_model.py) sono ``template + basi head_* @ w``: la
    patch dei dati (hockey_mask con i centri dei quad) e' un sottoinsieme lineare degli stessi
    vertici, quindi la mappa sulla testa intera vale per gli stessi pesi.
    """
    z = _gnm_raw()
    names = [str(n) for n in z["identity_names"]]
    head = [i for i, n in enumerate(names) if n.startswith("head_")]
    groups = [str(n) for n in z["vertex_group_names"]]
    skin = z["vertex_groups"][groups.index("skin_exterior")] > 1e-4
    F = z["triangles"].astype(np.int64)
    basis = np.moveaxis(z["vertex_identity_basis"][head].astype(np.float64), 0, -1)  # (nv, 3, 170)
    return _pack("gnm", z["template_vertex_positions"], F[skin[F].all(1)], F, basis,
                 units="metri", frame="GNM: +y alto, +z naso",
                 note="testa intera di gnm_head.npz, bersaglio skin_exterior, 170 basi head_*")


# ----------------------------------------------------------------------------- HIFI3D

def hifi3d() -> dict:
    """Testa intera HIFI3D: vertici e base di AI-NEXT-Shape.mat (il modello delle identita'),
    triangoli della testa intera da AI-NEXT-Shape-NoAug.mat (stessi 20.481 vertici; il ``tri`` di
    AI-NEXT-Shape.mat e' solo il volto, hifi_model.py)."""
    import scipy.io
    from hifi_model import load_hifi

    m = load_hifi(HIFI_MAT)
    tri = np.asarray(scipy.io.loadmat(str(HIFI_NOAUG_MAT))["tri"]).astype(np.int64)
    if tri.shape[1] != 3:
        tri = tri.T
    if tri.max() == len(m["v_template"]):
        tri = tri - 1
    face = {tuple(sorted(t)) for t in m["f"].tolist()}
    head = {tuple(sorted(t)) for t in tri.tolist()}
    if not face <= head:
        raise ValueError("i triangoli del volto di AI-NEXT-Shape non stanno nella testa di NoAug")
    return _pack("hifi3d", m["v_template"], tri, tri, m["shapedirs"],
                 units="unita' del .mat", frame="HIFI3D: +y alto, +z naso",
                 note="AI-NEXT-Shape.mat (500 modi), triangoli della testa da AI-NEXT-Shape-NoAug.mat")


# -------------------------------------------------------------------------- FaceVerse

def faceverse() -> dict:
    from fv_model import load_fv

    m = load_fv(FV_NPY)
    return _pack("faceverse", m["v_template"], m["f_face"], m["f"], m["shapedirs"],
                 units="unita' del .npy", frame="nativo di FaceVerse v2",
                 note="mesh piena di faceverse_simple_v2.npy (fv_model.py), 150 modi")


# -------------------------------------------------------------------------- FaceScape

def facescape() -> dict:
    """Bilineare FaceScape v1.6 (300 identita' PCA x 52 espressioni), espressione neutra (indice 0)
    e identita' media ``id_mean``: ``V = core[:, 0, :] @ id_mean`` (gen_full del toolkit con
    exp_vec = e_0). Il core da 4.9 GB si legge una volta: la media si salva in cache (solo la
    media, non il modello)."""
    if FS_CACHE.exists():
        with np.load(FS_CACHE) as d:
            V, F = d["V"], d["F"]
    else:
        with np.load(FS_NPZ, allow_pickle=True) as d:
            core = d["shape_bm_core"]                     # (3n, 52, 300)
            id_mean = d["id_mean"].astype(np.float64)
            F = d["fv_indices"].astype(np.int64) - 1      # base 1 nel file
        V = (core[:, 0, :].astype(np.float64) @ id_mean).reshape(-1, 3)
        del core
        FS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(FS_CACHE, V=V, F=F)
    return _pack("facescape", V, F, F, None, units="FaceScape (millimetri)",
                 frame="nativo del bilineare v1.6",
                 note="facescape_bm_v1.6_847_300_52_id.npz, neutra, id_mean")


# -------------------------------------------------------------------------- Multiface

def multiface_neutral_paths() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for p in sorted(MULTIFACE_TRACKED.glob("*.npz")):
        subj, seg, _ = p.stem.split("__")
        if seg in MULTIFACE_NEUTRAL:
            out.setdefault(subj, []).append(p)
    return out


@lru_cache(maxsize=1)
def multiface() -> dict:
    """Topologia tracked di Multiface (5.471 vertici, mm). Template: per soggetto la media dei
    frame neutri (E001_Neutral_Eyes_Open / EXP_eye_neutral), poi Procrustes generalizzato fra i 13
    soggetti, nel frame del primo (le mesh tracked hanno la posa della testa di ogni frame)."""
    per_subj = []
    F0 = None
    for subj, paths in sorted(multiface_neutral_paths().items()):
        Vs = []
        for p in paths:
            with np.load(p) as d:
                V, F = np.asarray(d["V"], np.float64), np.asarray(d["F"], np.int64)
            if F0 is None:
                F0 = F
            elif not np.array_equal(F, F0):
                raise ValueError(f"{p.name}: facce diverse")
            if Vs:
                V = apply_sim(V, *umeyama(V, Vs[0], scale=False))
            Vs.append(V)
        per_subj.append(np.mean(Vs, axis=0))
    mean = per_subj[0]
    for _ in range(10):
        aligned = [apply_sim(X, *umeyama(X, mean)) for X in per_subj]
        new = np.mean(aligned, axis=0)
        new = apply_sim(new, *umeyama(new, per_subj[0]))
        if np.abs(new - mean).max() < 1e-6:
            break
        mean = new
    return _pack("multiface", mean, F0, units="millimetri",
                 frame="frame del primo soggetto (002421669); le mesh tracked hanno la posa di ogni frame",
                 note=f"media GPA dei frame neutri di {len(per_subj)} soggetti, topologia tracked")


def template(name: str) -> dict:
    return {"flame": flame, "bfm": bfm, "ict": ict, "gnm": gnm, "facescape": facescape,
            "hifi3d": hifi3d, "faceverse": faceverse, "multiface": multiface}[name]()


# ------------------------------------------------------------------------- identita'

def _shard_manifest(tar_path: Path) -> dict:
    with tarfile.open(tar_path) as t:
        return json.load(t.extractfile("manifest.json"))


def ict_weights(sids: list[str]) -> np.ndarray:
    """Pesi (n, 100) delle identita' ICT: ICT-5000 (id10000-14999, datasets/ICT/identities) e ICT
    nuove (id20000-, manifest degli shard di datasets/ICT_SCALE)."""
    want = {s: i for i, s in enumerate(sids)}
    W = np.full((len(sids), 100), np.nan)
    for s, i in want.items():
        g = int(s[2:])
        if 10000 <= g < 15000:
            with np.load(DATASETS / "ICT" / "identities" / f"ict{g - 10000:04d}.npz") as d:
                W[i] = d["weights"]
    rest = {s for s in want if int(s[2:]) >= 20000}
    if rest:
        for tar in sorted((DATASETS / "ICT_SCALE" / "shards").glob("shard_*.tar")):
            for rec in _shard_manifest(tar)["identities"]:
                if rec["sid"] in rest:
                    W[want[rec["sid"]]] = rec["weights"]
    if np.isnan(W).any():
        raise ValueError(f"pesi ICT mancanti per {int(np.isnan(W).any(1).sum())} identita'")
    return W


def gnm_weights(sids: list[str]) -> np.ndarray:
    """Pesi (n, 170) delle identita' GNM_DISTILL (manifest degli shard)."""
    want = {s: i for i, s in enumerate(sids)}
    W = None
    for tar in sorted((DATASETS / "GNM_DISTILL" / "shards").glob("gnm_shard_*.tar")):
        for rec in _shard_manifest(tar)["identities"]:
            if rec["sid"] in want:
                w = np.asarray(rec["weights"], dtype=np.float64)
                if W is None:
                    W = np.full((len(sids), len(w)), np.nan)
                W[want[rec["sid"]]] = w
    if W is None or np.isnan(W).any():
        raise ValueError("pesi GNM mancanti")
    return W


def zs_weights(domain: str, sids: list[str]) -> np.ndarray:
    """Pesi delle identita' zero-shot: HIFI3D id900000+i -> hifiNNNN, FaceVerse id910000+i -> fvNNNN."""
    root, prefix, off = {"hifi3d": ("HIFI3D", "hifi", 900000), "faceverse": ("FACEVERSE_ZS", "fv", 910000)}[domain]
    out = []
    for s in sids:
        with np.load(DATASETS / root / "identities" / f"{prefix}{int(s[2:]) - off:04d}.npz") as d:
            out.append(np.asarray(d["weights"], dtype=np.float64))
    return np.stack(out)


def zs_stored_vertices(domain: str, sid: str) -> np.ndarray:
    """La forma salvata (patch) di un'identita' zero-shot, per il controllo pesi -> mesh."""
    root, prefix, off = {"hifi3d": ("HIFI3D", "hifi", 900000), "faceverse": ("FACEVERSE_ZS", "fv", 910000)}[domain]
    with np.load(DATASETS / root / "identities" / f"{prefix}{int(sid[2:]) - off:04d}.npz") as d:
        return np.asarray(d["V"], dtype=np.float64)
