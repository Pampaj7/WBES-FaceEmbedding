#!/usr/bin/env python3
"""Landmark e regioni per i protocolli: definizione per generatore (``define``) e trasferimento a ogni mesh di una
vista (``prep``). Nessuna GT.

    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_landmarks.py define --domain hifi3d
    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_landmarks.py prep --view hifi3d --workers 32
    (pr.sbatch, passi define e prep; dominio nuovo: --spec <json>, pr_common.py)

``define`` (una volta per generatore, sulla topologia original): i 68 landmark iBUG dell'embedding FLAME 2020 di
DECA/MICA (``famos_common.flame_lmk68``, quello delle viste FaMoS) e gli 85 keypoint di REALY su FLAME
(``REALY/data/FLAME.txt``, baricentrici sui triangoli di ``REALY/data/FLAME.obj``, che sono quelli di FLAME 2020:
controllato) portati sul template del generatore con la mappa FLAME -> template della GT unificata
(``datasets/UNIFIED_GT/corr/<dominio>.npz``: NICP della regione FLAME guidato da landmark, ``correspond.py``), come
``aau/ava256/ava_patch.py:183`` (baricentrico se il triangolo FLAME sta nella regione, altrimenti l'immagine del
vertice di regione piu' vicino); poi il vertice piu' vicino della media delle original di 100 soggetti NON valutati
(``blmm.template_subjects``, nel frame dei dati del template: controllo, mediana della distanza media-template). Uscite:
``landmarks/<dominio>.json`` (indici, distanze di aggancio in mm; in git) e ``aau/runs/protocolli/defs/<dominio>.npz``
(etichette di regione per vertice). Regioni REALY (nose, mouth, forehead, cheek): le 4 maschere di REALY sono definite
sul template HIFI3D di REALY e NON sono pubbliche (solo le regioni ritagliate delle scansioni del benchmark); qui
l'approssimazione dalle maschere ufficiali di FLAME (``FLAME_masks.pkl``), con priorita' nose > lips (mouth) >
forehead U eye_region (forehead: REALY la ancora ai keypoint di occhi e sopracciglia) > cheek = resto della regione
del volto FLAME (``flame_region.npz``); etichetta di un
vertice = quella del vertice FLAME di peso massimo nel punto piu' vicino della regione FLAME deformata, 0 oltre 3 mm.
Su HIFI3D, controllo indipendente: i keypoint NATIVI di REALY per la topologia HIFI3D (``REALY/data/HIFI3D.txt``)
contro quelli trasferiti da FLAME.

``prep`` (per vista): per ogni soggetto, i landmark della original dagli indici; sulle altre topologie il punto piu'
vicino della superficie perturbata alla posizione sulla original dello stesso soggetto (quello che un utente del
benchmark fornirebbe); etichette per vertice dal vertice piu' vicino della original. Crop: un landmark che cade fuori
(distanza > ``OUT_MM`` dalla superficie ritagliata) si conta e si usa comunque il punto piu' vicino. FaMoS TEST: i
landmark dalla registrazione FLAME dello STESSO fotogramma (dati grezzi), portati nel frame della patch con la stessa
similarita' di ``famos_test_view.canonical_patch`` (7 landmark -> template T7), poi sul punto piu' vicino della
scansione; etichette dal vertice piu' vicino della registrazione. Riferimento di REALY per ogni mesh
(``realy_reference``): la mesh nel frame del template HIFI3D di REALY con la similarita' dei 7 keypoint (come REALY
riscala e allinea le scansioni al template, paper Sez. 4), scala metrica = mm per unita' ([R] :259); regioni = le
etichette dentro il disco centrato sulla punta del naso (paper Sez. 4: "a disk centered on the nose tip", raggio non
dato: quello di Stirling, 1.2 (ex-ex + naso) / 2), componente connessa piu' grande (paper: "return the largest
connected region"), decimate a ``--realy-cap`` vertici se piu' grandi (costo del nICP, dichiarato). Uscita
``aau/runs/protocolli/prep/<vista>.npz``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as pc  # noqa: E402
import pr_metrics as pm  # noqa: E402

blmm = pc.blmm
COVER_MM = 3.0                 # etichetta di regione: distanza massima dalla regione FLAME deformata / registrazione
OUT_MM = 0.5                   # landmark "fuori" dalla mesh perturbata (crop)
FRAME_MAX_MM = 2.0             # controllo del frame: mediana media-original -> template
REALY_CAP = 1500               # vertici massimi per regione del riferimento di REALY (default; --realy-cap)
MASKS = pc.REPO / "v2_work" / "genflame" / "official" / "FLAME_masks.pkl"
REGION_NPZ = pc.REPO / "datasets" / "UNIFIED_GT" / "flame_region.npz"


# ------------------------------------------------------------------------------------------- FLAME

def read_obj_tri(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Obj come lo legge REALY (utils/io_obj.py:44-90): i quad diventano (a, b, c), (a, c, d)."""
    V, F = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            q = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            F.append(q[:3])
            if len(q) == 4:
                F.append([q[0], q[2], q[3]])
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


def realy_kp(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(triangoli (85,), pesi (85, 3)) di ``REALY/data/<name>.txt`` (utils/util.py:56-66: [tri, b1, b2] ->
    pesi (b1, b2, 1 - b1 - b2))."""
    k = np.asarray(json.loads((pc.REALY_DATA / f"{name}.txt").read_text()), np.float64)
    return k[:, 0].astype(np.int64), np.stack([k[:, 1], k[:, 2], 1.0 - k[:, 1] - k[:, 2]], axis=1)


def bary_points(V: np.ndarray, F: np.ndarray, tri: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.einsum("kvd,kv->kd", V[F[tri]], w)


def flame_assets() -> dict:
    """FLAME 2020 (mm), embedding iBUG 68 di DECA/MICA, 85 keypoint di REALY, regione del volto, etichette."""
    import domains
    import famos_common as fc
    fl = domains.flame()
    Vf, Ff = fl["V"] * 1000.0, fl["F_render"]
    _, rF = read_obj_tri(pc.REALY_DATA / "FLAME.obj")
    if rF.shape != Ff.shape or not (rF == Ff).all():
        raise SystemExit("REALY/data/FLAME.obj: triangoli diversi da FLAME 2020")
    fi, bc = fc.flame_lmk68()
    t85, w85 = realy_kp("FLAME")
    reg = np.load(REGION_NPZ)
    with open(MASKS, "rb") as fh:
        masks = {k: np.asarray(v, np.int64) for k, v in pickle.load(fh, encoding="latin1").items()}
    lab = np.zeros(len(Vf), np.int8)
    inreg = np.zeros(len(Vf), bool)
    inreg[reg["vidx"]] = True
    lab[inreg] = 4                                                    # cheek = resto della regione del volto
    # eye_region nella fronte: REALY ancora la regione "forehead" ai keypoint di occhi e sopracciglia (iBUG 17-26,
    # 36-47, utils/util.py:31); con la sola maschera forehead di FLAME (sopra le sopracciglia) quei keypoint cadono
    # fuori dalla regione e il nICP ne trascina il bordo (controllo 6 di pr_selftest)
    for name, k in (("eye_region", 3), ("forehead", 3), ("lips", 2), ("nose", 1)):   # l'ultimo vince
        m = np.zeros(len(Vf), bool)
        m[masks[name]] = True
        lab[m & inreg] = k
    return {"V": Vf, "F": Ff, "fi": fi, "bc": bc, "t85": t85, "w85": w85, "vidx": reg["vidx"], "Fr": reg["F"],
            "labels": lab}


def transfer_points(img: np.ndarray, fa: dict, tri: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, list]:
    """``ava_patch.transferred_landmarks`` (aau/ava256/ava_patch.py:183-205) per punti baricentrici su FLAME:
    ``img`` = immagini dei vertici della regione FLAME sul template."""
    pos = -np.ones(len(fa["V"]), np.int64)
    pos[fa["vidx"]] = np.arange(len(fa["vidx"]))
    out, how = [], []
    for i in range(len(tri)):
        t = fa["F"][tri[i]]
        if (pos[t] >= 0).all():
            out.append((w[i][:, None] * img[pos[t]]).sum(0))
            how.append("baricentrico")
        else:
            pf = (w[i][:, None] * fa["V"][t]).sum(0)
            j = int(np.linalg.norm(fa["V"][fa["vidx"]] - pf, axis=1).argmin())
            out.append(img[j])
            how.append(f"vicino {np.linalg.norm(fa['V'][fa['vidx']][j] - pf):.2f} mm")
    return np.asarray(out), how


# ------------------------------------------------------------------------------------------ define

def define_inputs(domain: str, spec: Path | None) -> dict:
    """Template (nel frame dei dati), mappa FLAME -> template, original da mediare (raw), unita' (mm per unita')."""
    import domains
    if spec is not None:
        s = json.loads(Path(spec).read_text())
        L = s["landmarks"]
        rp = lambda p: Path(p) if Path(p).is_absolute() else pc.REPO / p  # noqa: E731
        with np.load(rp(L["template"])) as z:
            T = {"V": np.asarray(z["V"], np.float64), "F": np.asarray(z["F"], np.int64)}
        corr = rp(L["corr"])
        snap = [rp(L["snap_dir"]) / f"{x}_GTready_original.npz" for x in L["snap_subjects"]]
        return {"T": T, "corr": corr, "snap": snap, "u": float(s["frame"]["u"])}
    t = domains.template(domain)
    view = {"hifi3d": "hifi3d", "facescape": "facescape", "faceverse": "faceverse"}[domain]
    snap = [blmm.mesh_path(view, x, "original", template=True) for x in blmm.template_subjects(view)]
    u = float(json.loads(blmm.FRAMES_JSON.read_text())["domains"][domain]["u"])
    return {"T": {"V": t["V"], "F": t["F_target"]}, "corr": pc.REPO / "datasets" / "UNIFIED_GT" / "corr" /
            f"{domain}.npz", "snap": snap, "u": u}


def define(domain: str, spec: Path | None) -> None:
    import ugt as C
    t0 = time.time()
    fa = flame_assets()
    inp = define_inputs(domain, spec)
    T, u = inp["T"], inp["u"]
    with np.load(inp["corr"]) as z:
        img = C.bary_interp(T["V"], z["vidx"], z["bary"])
    P68, how68 = transfer_points(img, fa, fa["fi"], fa["bc"])
    P85, how85 = transfer_points(img, fa, fa["t85"], fa["w85"])
    Vs, F0 = [], None
    for p in inp["snap"]:
        V, F = blmm.load_raw(p)
        if F0 is None:
            F0 = F
        elif F.shape != F0.shape or not (F == F0).all():
            raise SystemExit(f"{p}: topologia original diversa")
        Vs.append(V)
    M = np.mean(Vs, axis=0)
    used = np.unique(F0)
    frame = C.Surface(T["V"], T["F"]).closest(M[used])["dist"] * u
    if np.median(frame) > FRAME_MAX_MM:
        raise SystemExit(f"{domain}: media delle original lontana dal template (mediana {np.median(frame):.2f} mm): "
                         "frame diversi, la mappa non vale")
    tree = cKDTree(M[used])
    d68, i68 = tree.query(P68)
    d85, i85 = tree.query(P85)
    idx68, idx85 = used[i68], used[i85]
    # regioni: etichetta FLAME nel punto piu' vicino della regione FLAME deformata sul template
    cl = C.Surface(img, fa["Fr"]).closest(M[used])
    vmax = fa["Fr"][cl["tri"], np.argmax(cl["bary"], axis=1)]
    lab_used = fa["labels"][fa["vidx"][vmax]].copy()
    lab_used[cl["dist"] * u > COVER_MM] = 0
    labels = np.zeros(len(M), np.int8)
    labels[used] = lab_used
    out = {"domain": domain, "n_vertices_original": int(len(M)), "n_referenced": int(len(used)),
           "snap_subjects": len(inp["snap"]), "mm_per_unit": u,
           "lmk7_names": list(pc.LMK7_NAMES), "lmk7_ibug": list(pc.LMK7),
           "lmk7": [int(idx68[k]) for k in pc.LMK7], "ibug68": idx68.tolist(), "realy85": idx85.tolist(),
           "snap_mm_lmk7": [round(float(d68[k] * u), 3) for k in pc.LMK7],
           "snap_mm_ibug68": {"median": float(np.median(d68) * u), "max": float(d68.max() * u)},
           "snap_mm_realy85": {"median": float(np.median(d85) * u), "max": float(d85.max() * u)},
           "transfer_ibug68_nonbary": {str(k): h for k, h in enumerate(how68) if h != "baricentrico"},
           "transfer_realy85_nonbary": {str(k): h for k, h in enumerate(how85) if h != "baricentrico"},
           "frame_check_mm": {"median": float(np.median(frame)), "p95": float(np.percentile(frame, 95))},
           "region_vertices": {r: int((labels == k + 1).sum()) for k, r in enumerate(pc.REGIONS)},
           "sources": {"ibug68": "external/MICA/data/FLAME2020/landmark_embedding.npy (full_lmk, DECA/MICA)",
                       "realy85": "REALY/data/FLAME.txt", "corr": pc.blmm.rel(inp["corr"]),
                       "masks": "v2_work/genflame/official/FLAME_masks.pkl", "clones": pc.CLONE_COMMITS},
           **pc.stamp()}
    if domain == "hifi3d" and spec is None:
        hV, hF = read_obj_tri(pc.REALY_DATA / "HIFI3D.obj")
        th, wh = realy_kp("HIFI3D")
        if hV.shape == T["V"].shape:
            H85 = bary_points(T["V"], hF, th, wh)
            dd = np.linalg.norm(H85 - P85, axis=1) * u
            out["check_realy_native_hifi3d_mm"] = {
                "note": "keypoint nativi di REALY per la topologia HIFI3D (HIFI3D.txt) contro quelli trasferiti da FLAME, "
                        "sul template HIFI3D",
                "lmk7": [round(float(dd[k]), 2) for k in pc.LMK7], "median_85": float(np.median(dd)),
                "max_85": float(dd.max()), "argmax_85": int(dd.argmax())}
    pc.atomic_json(pc.DEFS / f"{domain}.json", out)
    blmm.atomic_savez(pc.DEFS_NPZ / f"{domain}.npz", labels=labels, ibug68=idx68, realy85=idx85)
    print(f"[pr-lmk] {domain}: aggancio lmk7 {out['snap_mm_lmk7']} mm, frame {out['frame_check_mm']}, regioni "
          f"{out['region_vertices']}, {time.time() - t0:.0f}s", flush=True)


# -------------------------------------------------------------------------------------- riferimento REALY

_TPL: dict = {}


def realy_template7() -> np.ndarray:
    """I 7 keypoint del template HIFI3D di REALY (``data/HIFI3D.obj`` + ``HIFI3D.txt``)."""
    if not _TPL:
        hV, hF = read_obj_tri(pc.REALY_DATA / "HIFI3D.obj")
        _TPL["k7"] = bary_points(hV, hF, *realy_kp("HIFI3D"))[pm.KP_REGION["seven_keypoints"]]
    return _TPL["k7"]


def largest_component(F: np.ndarray, n: int) -> np.ndarray:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    if not len(F):
        return F
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]])
    G = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    _, comp = connected_components(G, directed=False)
    c = comp[F[:, 0]]
    return F[c == np.bincount(c).argmax()]


def compact(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(F)
    m = -np.ones(len(V), np.int64)
    m[used] = np.arange(len(used))
    return V[used], m[F]


def realy_reference(V: np.ndarray, F: np.ndarray, labels: np.ndarray, K85: np.ndarray, cap: int) -> dict:
    """Riferimento di REALY per una mesh (mm): frame e unita' del template (``fit_icp_RT`` con scala sui 7 keypoint,
    la funzione di REALY), regioni nel disco della punta del naso, componente piu' grande, decimazione a ``cap``."""
    import open3d as o3d
    s, R, t = pm.fit_icp_RT(K85[pm.KP_REGION["seven_keypoints"]], realy_template7(), with_scale=True)
    Vr, Kr = s * V @ R.T + t, s * K85 @ R.T + t
    centre = Kr[30]
    _, rad = pm.now_mask(Kr[list(pc.LMK7)], 1.2)
    inside = np.linalg.norm(Vr - centre, axis=1) <= rad
    regions, info = {}, {}
    for k, name in enumerate(pc.REGIONS):
        sel = (labels == k + 1) & inside
        Fk = largest_component(F[sel[F].all(1)], len(Vr))
        if len(Fk) < 4:
            info[name] = 0
            continue
        Vk, Fk = compact(Vr, Fk)
        n0 = len(Vk)
        if len(Vk) > cap:
            m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vk), o3d.utility.Vector3iVector(Fk.astype(np.int32)))
            m = m.simplify_quadric_decimation(int(round(len(Fk) * cap / len(Vk))))
            Vd, Fd = np.asarray(m.vertices), np.asarray(m.triangles, np.int64)
            Fd = largest_component(Fd[(Fd[:, 0] != Fd[:, 1]) & (Fd[:, 1] != Fd[:, 2]) & (Fd[:, 0] != Fd[:, 2])], len(Vd))
            Vk, Fk = compact(Vd, Fd)
        regions[name] = (Vk.astype(np.float64), Fk.astype(np.int32))
        info[name] = [n0, int(len(Vk))]
    return {"K": Kr, "regions": regions, "metrical": float(1.0 / s), "info": info}


# ------------------------------------------------------------------------------------------- prep

_ST: dict = {}
FV_EXPR_TOPO = blmm.DATASETS / "FACEVERSE_ZS" / "expr_topo"
FV_IDENTITIES = blmm.DATASETS / "FACEVERSE_ZS" / "identities"


def fv_expr_original(subject: str, topology: str) -> np.ndarray:
    """FaceVerse con espressioni: ogni (identita', topologia) ha un'espressione PROPRIA
    (``aau/zs3dmm/make_zs_expr_topologies.py``), quindi la original dello stesso soggetto non e' la stessa superficie
    delle altre topologie. Qui la patch original con l'espressione della mesh (pesi d'identita' di
    ``identities/identity_weights.json``, vettore di ``expr_topo/expression_vectors.json``), con le funzioni di quel
    modulo (importate): ``expressed_patch`` + ``mesh_ops.prepare_open_surface`` (la variante "original"). Vertici
    nativi (n, 3), stessa topologia della original su disco."""
    import make_zs_expr_topologies as mz
    if "fv" not in _ST:
        mf = json.loads((blmm.VIEWS["faceverse"]["dir"].parent / "manifest.json").read_text())["source_manifest"]["model_file"]
        _, expr = mz._load("fv", mf)
        pool, _ = mz.expression_pool("fv", expr)
        mz._init("fv", mf, pool)
        _ST["fv"] = {"w": json.loads((FV_IDENTITIES / "identity_weights.json").read_text()),
                     "vec": json.loads((FV_EXPR_TOPO / "expression_vectors.json").read_text())["vectors"]}
    fid = f"fv{int(subject[2:]) - 910000:04d}"                # domains.zs_weights: id910000 + i -> fvNNNN
    names = mz._EXPR["names"]
    coef = np.zeros(mz._EXPR["exprdirs"].shape[2])
    for n, c in _ST["fv"]["vec"][fid][topology].items():
        coef[names.index(n)] = c
    V, Fp = mz.expressed_patch(np.asarray(_ST["fv"]["w"][fid], np.float64), coef)
    return np.asarray(mz.mo.prepare_open_surface(*mz.mo.as_arrays(V, Fp))[0], np.float64)


def _prep_subject(task: tuple) -> list:
    """Le mesh di un soggetto: landmark (68 + 85, mm), distanze di trasferimento, etichette, riferimento REALY.
    Sorgente dei landmark di una mesh: la original del soggetto; su FaceVerse con espressioni la original con la STESSA
    espressione della mesh (``fv_expr_original``; controllo: rigenerata contro la original su disco)."""
    subject, items = task
    v, d, cap = _ST["view"], _ST["def"], _ST["cap"]
    expr = v["name"] == "faceverse"
    orig = [p for _, t, p in items if t == "original"][0]
    Vraw, Forig = blmm.load_raw(orig)
    used_o = np.unique(Forig)
    out = []
    for _, t, p in items:
        Vsrc = fv_expr_original(subject, t) if expr else Vraw
        regen = float(np.abs(Vsrc - Vraw).max() * pc.unit_of(v)) if expr and t == "original" else np.nan
        Vo = pc.to_mm(v, Vsrc, orig)
        P = np.vstack([Vo[d["ibug68"]], Vo[d["realy85"]]])
        lab_o = d["labels"][used_o]
        V, F, _ = pc.mm_mesh(v, p)
        if t == "original":
            Q, dist = P.copy(), np.zeros(len(P))
            lab = lab_o
        else:
            Q = pm.Surf(V, F).closest(P)
            dist = np.linalg.norm(Q - P, axis=1)
            lab = lab_o[cKDTree(Vo[used_o]).query(V)[1]]
        ref = realy_reference(V, F, lab, Q[68:], cap)
        out.append({"subject": subject, "topology": t, "path": blmm.rel(p), "L68": Q[:68], "K85": Q[68:],
                    "d68": dist[:68], "d85": dist[68:], "labels": lab.astype(np.int8), "realy": ref,
                    "n_vertices": len(V), "regen_mm": regen})
    return out


def _prep_famos(task: tuple) -> list:
    """Scansione di galleria FaMoS: landmark dalla registrazione FLAME dello stesso fotogramma (dati grezzi)."""
    import famos_common as fc
    import now_common
    row, p = task
    fa, tpl7, cap = _ST["flame"], _ST["tpl7"], _ST["cap"]
    Vr = fc.read_reg(fc.reg_path(row["subject"], row["seq"], int(row["frame"]))).astype(np.float64) * 1000.0
    L68 = bary_points(Vr, fa["F"], fa["fi"], fa["bc"])
    K85 = bary_points(Vr, fa["F"], fa["t85"], fa["w85"])
    s, R, t = now_common.similarity_to(L68[list(pc.LMK7)], tpl7)        # famos_test_view.canonical_patch
    to = lambda X: (s * X @ R + t) / float(row["scale_to_mm"])         # noqa: E731  (blmm.to_mm: / scale_to_mm)
    P = np.vstack([to(L68), to(K85)])
    V, F, _ = pc.mm_mesh(_ST["view"], p)
    Q = pm.Surf(V, F).closest(P)
    dist = np.linalg.norm(Q - P, axis=1)
    dd, ii = cKDTree(to(Vr)).query(V)
    lab = fa["labels"][ii].copy()
    lab[dd > COVER_MM] = 0
    ref = realy_reference(V, F, lab, Q[68:], cap)
    return [{"subject": row["view_id"], "topology": "scan_gallery", "path": blmm.rel(p), "L68": Q[:68],
             "K85": Q[68:], "d68": dist[:68], "d85": dist[68:], "labels": lab.astype(np.int8), "realy": ref,
             "n_vertices": len(V), "scale_check": float(s / float(row["scale_to_mm"]))}]


def prep(vname: str, spec: Path | None, workers: int, cap: int) -> None:
    t0 = time.time()
    v = pc.view(vname, spec)
    _ST.update(view=v, cap=cap)
    if v["name"] == "famos":
        _ST["flame"] = flame_assets()
        _ST["tpl7"] = np.asarray(json.loads((blmm.DATASETS / "FAMOS" / "test_view" / "template_lmk7.json").read_text())
                                 ["landmarks_mm"], np.float64)
        rows = {r["name"]: r for r in blmm.famos_manifest()}
        tasks = [(rows[Path(p).stem], p) for _, _, p in v["items"]]
        fn = _prep_famos
    else:
        _ST["def"] = pc.lm_def(v["lm_domain"])
        by = {}
        for s, t, p in v["items"]:
            by.setdefault(s, []).append((s, t, p))
        tasks = list(by.items())
        fn = _prep_subject
    with mp.get_context("fork").Pool(workers) as pool:
        res = [r for rr in pool.map(fn, tasks, chunksize=1) for r in rr]
    pos = {(r["subject"], r["topology"]): r for r in res}
    res = [pos[(s, t)] for s, t, _ in v["items"]]
    obj = lambda xs: np.asarray(xs + [None], dtype=object)[:-1]       # noqa: E731
    d68 = np.stack([r["d68"] for r in res])
    d85 = np.stack([r["d85"] for r in res])
    topo = np.asarray([r["topology"] for r in res])
    out = {"subjects": np.asarray([r["subject"] for r in res]), "topologies": topo,
           "paths": np.asarray([r["path"] for r in res]), "L68": np.stack([r["L68"] for r in res]),
           "K85": np.stack([r["K85"] for r in res]), "d68": d68, "d85": d85,
           "labels": obj([r["labels"] for r in res]), "realy": obj([r["realy"] for r in res]),
           "n_vertices": np.asarray([r["n_vertices"] for r in res])}
    pc.WORK.joinpath("prep").mkdir(parents=True, exist_ok=True)
    np.savez(pc.prep_path(v["name"]), **out)
    lm7 = d68[:, list(pc.LMK7)]
    summary = {"view": v["name"], "meshes": len(res), "realy_cap": cap, "seconds": time.time() - t0,
               "lmk7_transfer_mm_by_topology": {
                   t: {"median": float(np.median(lm7[topo == t])), "max": float(lm7[topo == t].max())}
                   for t in sorted(set(topo))},
               "lmk7_outside": {t: {n: int((lm7[topo == t][:, k] > OUT_MM).sum()) for k, n in enumerate(pc.LMK7_NAMES)}
                                for t in sorted(set(topo))},
               "realy85_outside": {t: int((d85[topo == t] > OUT_MM).sum()) for t in sorted(set(topo))},
               "realy_region_vertices": {r: {"median_before": float(np.median([x["realy"]["info"][r][0] for x in res
                                                                               if x["realy"]["info"].get(r)])),
                                             "median_after": float(np.median([x["realy"]["info"][r][1] for x in res
                                                                              if x["realy"]["info"].get(r)])),
                                             "missing": int(sum(1 for x in res if not x["realy"]["info"].get(r)))}
                                         for r in pc.REGIONS},
               "realy_metrical_mm_per_unit": {"median": float(np.median([x["realy"]["metrical"] for x in res])),
                                              "min": float(min(x["realy"]["metrical"] for x in res)),
                                              "max": float(max(x["realy"]["metrical"] for x in res))},
               **pc.stamp()}
    if v["name"] == "faceverse":
        rg = [r["regen_mm"] for r in res if np.isfinite(r.get("regen_mm", np.nan))]
        summary["fv_expr_original_regenerated_vs_disk_max_mm"] = float(max(rg))
    if v["name"] == "famos":
        summary["famos_similarity_scale_over_manifest"] = [round(r["scale_check"], 6) for r in res]
    pc.atomic_json(pc.EVID / "prep" / f"{v['name']}.json", summary)
    print(f"[pr-lmk] prep {v['name']}: {len(res)} mesh, {time.time() - t0:.0f}s; landmark fuori "
          f"{summary['lmk7_outside']}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("define", "prep"))
    p.add_argument("--domain", default="")
    p.add_argument("--view", default="")
    p.add_argument("--spec", type=Path, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--realy-cap", type=int, default=REALY_CAP)
    a = p.parse_args()
    if a.step == "define":
        define(a.domain or json.loads(a.spec.read_text())["name"], a.spec)
    else:
        prep(a.view or json.loads(a.spec.read_text())["name"], a.spec, a.workers, a.realy_cap)


if __name__ == "__main__":
    main()
