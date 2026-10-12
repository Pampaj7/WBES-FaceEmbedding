#!/usr/bin/env python3
"""Matrice completa e confrontabile: ogni metodo sulle STESSE righe, stessa maschera comune, stesse repliche.

    v3_work/unified_gt/run.sh aau/full_matrix/fm_eval.py --workers 64      (fm.sbatch, passo eval; LEGGE LE GT di test)
    v3_work/unified_gt/run.sh aau/full_matrix/fm_eval.py --summary-only    (solo results.md dai CSV)

Importati in sola lettura: ``fact_paired`` (GT, colonne dei bracci, ``_pearson``, FaMoS TEST), ``bp_paired`` /
``bp_paired_e2`` (righe all_cross, B sulle 600 mesh, composizione, la frame della T1 per il controllo), ``blmm`` /
``blmm_eval`` (matrici geometriche, scalari, template), ``bp`` (distanze delle mesh d'identita' dei fit).

Righe per dominio (HIFI3D, FaceScape dev, FaceVerse con espressioni, FaceVerse neutra: 100 soggetti x 6 topologie):
le 148.500 di ``bp_paired_e2.all_cross_rows`` (coppie di soggetti diversi, topologie diverse, crop compreso; per la
neutra le stesse chiavi e GT di FaceVerse) + le 6 x 4.950 di stessa topologia costruite sulle stesse coppie di soggetti
(come ``bp_paired_e3.run_topo``, qui anche crop - crop). Maschera COMUNE = righe dove TUTTE le colonne del dominio
sono finite (una colonna intera NaN sul dominio non entra ed e' dichiarata non applicabile). Insiemi (sottoinsiemi della
stessa maschera): ``nocrop`` (topologie diverse, nessuna crop), ``crop`` (una crop), ``all_cross``, per perturbazione
P ``con_perturbazione`` (P su un lato, l'altro una topologia diversa senza crop: la scomposizione di ``nocrop``; per P =
crop e' l'insieme ``crop``), ``coppia_topologie`` (coppia non ordinata), ``stessa_topologia``
(per P e tutte), ``coppia_ordinata``; le medie ``media_20_ordinate_senza_crop`` e ``media_5_stessa_topologia`` (le
grandezze di topo_pairs_e3, per il controllo). FaMoS TEST: il blocco scan gallery -> scan di ``fact_paired.famos_frame``
(105 righe), un solo insieme (``nocrop``); crop e perturbazioni non esistono.

Repliche: 1000 per soggetto, ``default_rng(seme)`` sui soggetti ordinati, seme = quello della T1 (``fact_paired.rows_for``
del dominio: HIFI3D 990708, FaceScape 796786, FaceVerse e neutra 271049; FaMoS ``famos_eval.bootstrap_counts(15, n,
1234)``), lo stesso per tutti gli insiemi del dominio. Spearman pesato c_a c_b come ``fact_paired._rep`` (ranghi delle
righe ripetute, Pearson). IC 95% percentile; delta appaiati = differenza replica per replica.

Delta (``deltas.csv``): ``vs_nostro`` = braccio - concorrente (positivo = il nostro e' davanti) per ogni concorrente e
riferimento: FR contro factorized s1234/s2345 d_F cal. e d_F non cal. e ctrlfr s1234/s2345; SR contro factorized
s1234/s2345 d_P e ctrlfr; ``scala_allineamento`` (a - b, ``LADDER``); ``migliore_vs_migliore`` = miglior nostro - miglior
concorrente fra le letture dichiarate (scelti sul punto: selezione post hoc, descrittiva). Scala: passi principali
e secondari (``LADDER``: 2b e 3 in mm confusi dalla taglia; con SR i passi 4 e 5 anche contro ICP cs e NICP cs) e il
conteggio per famiglia sintetica nel verso "piu' allineamento = peggio". Sensibilita' ai fallimenti del NICP per coppia
(HIFI3D, FaceScape): maschera senza il NICP e fallimento = distanza massima (``passo`` dedicato).

Plug-in (concorrenti nuovi): ogni cartella ``aau/runs/evidence/competitors_new/<metodo>/`` o ``<metodo>/<tag>/`` (o
``--plugins``) MARCATA da ``fm_plugin.json`` (le cartelle senza, es. pilot/ o prep/, si ignorano e si elencano in
controls.json; un file senza nessuna mesh del dominio e' un errore col nome del plug-in) con, per
dominio, ``<dominio>.npz`` (``subjects``, ``topologies``, una o piu' matrici ``D`` / ``D_<lettura>`` (n, n) sulle mesh
elencate; FaMoS: ``subjects`` = view_id, ``topologies`` = ``scan_gallery``) oppure ``<dominio>.csv`` (``subject_a``,
``topology_a``, ``subject_b``, ``topology_b`` e una colonna per lettura; la coppia mancante si cerca invertita), e
``fm_plugin.json`` (obbligatorio, anche ``{}``; facoltativi ``label``, ``family``, ``readings``: {lettura: {``label``,
``gt``: ["fr", "sr"]}}). Un
plug-in finito su tutte le righe della maschera comune entra nello stesso passo; se ha buchi va in un passo a parte sulla
maschera comune ristretta alle sue righe finite, con i bracci di riferimento sulle stesse righe (``n_righe`` lo dice).

Uscite in ``aau/runs/evidence/full_matrix/``: ``spearman.csv``, ``deltas.csv``, ``non_applicabili.csv``,
``controls.json``, ``results.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
REPO = AAU.parent
sys.path.insert(0, str(AAU / "baselines_param"))

import bp_paired as bpp  # noqa: E402  (sola lettura: fact_paired, blmm_eval, blmm, bp)
import bp_paired_e2 as bpe2  # noqa: E402

fp, be, blmm, bp = bpp.fp, bpp.be, bpp.blmm, bpp.bp
EV = AAU / "runs" / "evidence"
OUT = EV / "full_matrix"
PLUGIN_ROOT = EV / "competitors_new"
OUT_W = OUT                                  # uscite (--out per le prove)
VIEWS = ("hifi3d", "facescape", "faceverse", "faceverse_neutral", "famos")
VIEW6 = VIEWS[:4]
DOMAIN = {"hifi3d": "HIFI3D", "facescape": "FaceScape (dev)", "faceverse": "FaceVerse con espressioni",
          "faceverse_neutral": "FaceVerse neutra", "famos": "FaMoS TEST"}
TOPOS = ("original", "remesh", "down8k", "up60k", "noisy", "crop")
NOCROP = TOPOS[:-1]
GTS = ("fr", "sr")
SPEC = AAU / "runs" / "competitors_hifi3d" / "desc_spectral.npz"
ARC_FV = AAU / "runs" / "arcface_render_zs"

# ------------------------------------------------------------------------------------------ metodi
# colonna -> (etichetta, famiglia, GT dichiarate, nostro)
FR, SR, BOTH = ("fr",), ("sr",), ("fr", "sr")
METHODS: dict = {}
for _s in ("s1234", "s2345"):
    METHODS[f"factorized_{_s}|form_cal"] = (f"factorized {_s}, d_F cal.", "nostri", FR, True)
    METHODS[f"factorized_{_s}|form"] = (f"factorized {_s}, d_F non cal.", "nostri", FR, True)
    METHODS[f"factorized_{_s}|shape"] = (f"factorized {_s}, d_P", "nostri", SR, True)
for _s in ("s1234", "s2345"):
    METHODS[f"ctrlfr_{_s}|z"] = (f"ctrlfr {_s}", "nostri", FR, True)
for _e in ("e123", "e205"):
    METHODS[f"factorizedc3m_{_e}|form_cal"] = (f"C3M {_e}, d_F cal.", "nostri", FR, True)
    METHODS[f"factorizedc3m_{_e}|form"] = (f"C3M {_e}, d_F non cal.", "nostri", FR, True)
    METHODS[f"factorizedc3m_{_e}|shape"] = (f"C3M {_e}, d_P", "nostri", SR, True)
METHODS.update({
    "mm_rigid_icp_chamfer": ("ICP + Chamfer in mm (per coppia)", "geometrici", BOTH, False),
    "cs_rigid_icp_chamfer": ("ICP + Chamfer cs (per coppia, scala tolta)", "geometrici", BOTH, False),
    "mm_nicp_p2tri": ("NICP per coppia in mm", "geometrici", BOTH, False),
    "cs_nicp_p2tri": ("NICP per coppia cs", "geometrici", BOTH, False),
    "mm_nicp_template": ("NICP su template in mm", "geometrici", BOTH, False),
    "cs_nicp_template": ("NICP su template cs", "geometrici", BOTH, False),
    "varifold": ("varifold in mm (massa unitaria)", "geometrici", BOTH, False)})
_MN = {"gnm": "GNM (visto)", "flame2023": "FLAME 2023 Open"}
_VN = {"orig": "preregistrato", "va": "vA", "vb": "vB"}
for _m in ("gnm", "flame2023"):
    for _v in ("orig", "va", "vb"):
        METHODS[f"{_m}_{_v}_coef"] = (f"{_MN[_m]} {_VN[_v]}, coefficienti", "fit 3DMM", BOTH, False)
        METHODS[f"{_m}_{_v}_fr"] = (f"{_MN[_m]} {_VN[_v]}, mesh d'identita' FR", "fit 3DMM", FR, False)
        METHODS[f"{_m}_{_v}_sr"] = (f"{_MN[_m]} {_VN[_v]}, mesh d'identita' SR", "fit 3DMM", SR, False)
    METHODS[f"{_m}_vb_comp"] = (f"{_MN[_m]} vB, composizione S_B + k d_P", "fit 3DMM", FR, False)
METHODS.update({
    "arcface_normals": ("ArcFace-render, normal map 3 viste (G0)", "percettivi", BOTH, False),
    "arcface_shaded": ("ArcFace-render, ombreggiato 3 viste (E12)", "percettivi", BOTH, False),
    "shapedna_k50": ("ShapeDNA k=50", "spettrali", BOTH, False),
    "hks": ("HKS globale", "spettrali", BOTH, False),
    "wks": ("WKS globale", "spettrali", BOTH, False),
    "openshape": ("OpenShape PointBERT", "reti 3D", BOTH, False),
    "uni3d": ("Uni3D-g", "reti 3D", BOTH, False),
    "est_cs": ("taglia stimata dalla mesh (non oracolo)", "riferimenti", BOTH, False),
    "oracle_size": ("taglia oracolo (S di FR)", "riferimenti", BOTH, False)})
REFERENCES = ("est_cs", "oracle_size")
# riferimenti dei delta vs_nostro per GT (ctrlfr non e' una lettura dichiarata per SR: solo FR)
REF = {"fr": ("factorized_s1234|form_cal", "factorized_s2345|form_cal", "factorized_s1234|form",
              "factorized_s2345|form", "ctrlfr_s1234|z", "ctrlfr_s2345|z"),
       "sr": ("factorized_s1234|shape", "factorized_s2345|shape")}


def post_hoc(col: str, view: str = "") -> bool:
    """Letture POST HOC: d_F calibrata (emendamento 4), varianti A e B dei fit e la composizione (emendamenti 1 e 2 dei
    concorrenti parametrici), il template di FaMoS costruito qui; C3M e' preregistrato come descrittivo (le sue d_F
    non calibrata e d_P non sono post hoc, la d_F calibrata si')."""
    return ("|form_cal" in col or "_va_" in col or "_vb_" in col or
            (view == "famos" and col == "mm_nicp_template"))


# scala dell'allineamento: (gt, a, b, passo, ruolo, lato piu' allineato); delta = a - b. Ruolo: principale o secondario
# (2b e 3 in mm sono confusi dalla taglia: il NICP su template in mm segue la taglia oracolo, il NICP per coppia in mm
# no; con SR il riferimento corretto e' senza taglia, ICP mm e' secondario). Lato piu' allineato (per il conteggio): b
# nei passi 1-3 (cs = taglia tolta per mesh; NICP = deformazione; NICP per coppia = deformazione specifica della
# coppia), a nel 4 (fit del modello per mesh contro la rigida), b nel 5 (ICP: allineamento esplicito; il braccio nessuno).
LADDER = []
_P1 = "1. togliere la taglia (normalizzazione per mesh alla cs_ref): ICP mm - ICP cs"
for _g in GTS:
    LADDER += [(_g, "mm_rigid_icp_chamfer", "cs_rigid_icp_chamfer", _P1, "principale", "b"),
               (_g, "cs_rigid_icp_chamfer", "cs_nicp_p2tri", "2. deformazione non rigida per coppia, cs: ICP cs - NICP cs",
                "principale", "b"),
               (_g, "mm_rigid_icp_chamfer", "mm_nicp_p2tri", "2b. idem in mm: ICP mm - NICP mm (confuso dalla taglia)",
                "secondario", "b"),
               (_g, "mm_nicp_template", "mm_nicp_p2tri",
                "3. per mesh contro per coppia, mm: NICP template mm - NICP per coppia mm (confuso dalla taglia)",
                "secondario", "b"),
               (_g, "cs_nicp_template", "cs_nicp_p2tri",
                "3b. per mesh contro per coppia, cs: NICP template cs - NICP per coppia cs", "principale", "b")]
    _r, _a = ("fr", "form") if _g == "fr" else ("sr", "shape")
    _refs = (("mm_rigid_icp_chamfer", "ICP mm", "principale"),) if _g == "fr" else \
        (("cs_rigid_icp_chamfer", "ICP cs", "principale"), ("cs_nicp_p2tri", "NICP cs", "principale"),
         ("mm_rigid_icp_chamfer", "ICP mm (riferimento con la taglia)", "secondario"))
    for _b, _lb, _ru in _refs:
        LADDER += [(_g, f"{m}_vb_{_r}", _b, f"4. fit 3DMM vB per mesh (mesh {_r.upper()}) - {_lb}", _ru, "a")
                   for m in ("gnm", "flame2023")]
        LADDER += [(_g, f"factorized_{s}|{_a}", _b,
                    f"5. nostro ({'d_F non cal.' if _g == 'fr' else 'd_P'}) - {_lb}", _ru, "b") for s in ("s1234", "s2345")]
NICP = ("mm_nicp_p2tri", "cs_nicp_p2tri")
SYNTH = VIEWS[:4]                                     # famiglie sintetiche (crop e perturbazioni)
# motivi dei non applicabili noti (colonna -> {vista: motivo}); il resto lo scopre il caricamento
NA_REASON = {
    "factorizedc3m_e123": {"faceverse_neutral": "embedding di C3M e123 mai calcolato sulla vista neutra (store assente; "
                                                "bp_e3 e faceverse_neutral lo saltano)"},
    "cs_nicp_template": {"faceverse_neutral": "template cs mai iscritto sulla neutra (faceverse_neutral/blmm ha solo mm)",
                         "famos": "template cs non calcolato su FaMoS (qui solo mm)"},
    "arcface_shaded": {v: "render ombreggiato calcolato solo da E12 (HIFI3D, FaceVerse con espressioni)"
                       for v in ("facescape", "faceverse_neutral", "famos")},
    "openshape": {v: "OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d)" for v in VIEWS[1:]},
    "uni3d": {v: "OpenShape/Uni3D: solo dove gia' presenti (competitors_hifi3d)" for v in VIEWS[1:]},
    "_vb_comp": {}, "_va_": {}, "_orig_": {}}
SET_NA = {"famos": "FaMoS TEST: scansioni reali, nessuna crop ne' perturbazione sintetica (solo il blocco scan gallery -> "
                   "scan)"}
_J: dict = {}


def label(c: str) -> str:
    return METHODS[c][0] if c in METHODS else c


# ----------------------------------------------------------------------------------------- colonne

def log_abs(x) -> np.ndarray:
    return be.log_abs(x)


def cosine(Z: np.ndarray) -> np.ndarray:
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return 1.0 - Z @ Z.T


def ordered(keys: list, idx) -> np.ndarray:
    pos = {k: i for i, k in enumerate(keys)}
    if sorted(pos) != sorted(idx.keys):
        raise SystemExit(f"chiavi diverse da quelle attese ({len(pos)} contro {len(idx.keys)})")
    return np.asarray([pos[k] for k in idx.keys])


def keys_of(z) -> list:
    return list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))


def geo_columns(view: str, idx) -> dict:
    """ICP + Chamfer e NICP per coppia (mm, cs), NICP su template (mm, cs), dalle matrici di blmm."""
    out = {}
    for mode in ("mm", "cs"):
        root = blmm.view_root(view) / mode
        for m in ("rigid_icp_chamfer", "nicp_p2tri"):
            if (root / "matrices" / m).is_dir():
                out[f"{mode}_{m}"] = be.zes.facebench_distances(root, m, idx)
        same = OUT / "icp_same" / f"{view}_{mode}.npz"     # soggetti diversi, stessa topologia (fm_compute icp-same)
        if same.exists() and f"{mode}_rigid_icp_chamfer" in out:
            M = out[f"{mode}_rigid_icp_chamfer"]
            with np.load(same) as z:
                if [str(x) for x in z["subjects"]] != idx.subjects:
                    raise SystemExit(f"{same}: soggetti diversi")
                iu, ju = np.asarray(z["iu"]), np.asarray(z["ju"])
                for k, t in enumerate(str(x) for x in z["topologies"]):
                    r = idx.rows(t)
                    M[r[iu], r[ju]] = M[r[ju], r[iu]] = np.asarray(z["values_rigid_icp_chamfer"])[k]
        if (root / "template.npz").exists():
            with np.load(root / "template.npz") as z:
                R = np.asarray(z["R"])[ordered(keys_of(z), idx)].astype(np.float64)
            out[f"{mode}_nicp_template"] = be.irt.template_distances(R, R)
    return out


def varifold_column(view: str, idx) -> tuple[np.ndarray, float]:
    """Gram 600: 500 x 500 degli shard di bp_varifold.py + righe crop di fm_varifold.py."""
    G5, k5 = bpp.varifold_gram(view)
    with np.load(OUT / "varifold" / f"{view}.npz") as z:
        G, k6, chk = np.asarray(z["G"]), keys_of(z), float(z["check_rel_max"])
    if k6[:500] != k5:
        raise SystemExit(f"{view}: varifold, chiavi senza crop diverse")
    G[:, :500, :500] = G5
    if not np.isfinite(G).all():
        raise SystemExit(f"{view}: varifold incompleto")
    o = ordered(k6, idx)
    return bpp.varifold_distances(G[:, o][:, :, o]), chk


def fit_columns(view: str, idx) -> tuple[dict, dict]:
    """Preregistrato, vA, vB sulle 600 mesh (beta: fit.npz / fit_e1.npz senza crop + crop), distanze ``bp.mesh_distances``;
    composizione di B (k di calib_e2). Controllo: blocco senza crop contro le D salvate."""
    out, ctrl, K = {}, {}, bpe2.k_values()
    for m in bp.MODELS:
        root = bp.out_dir(view, m)
        ctx = bpe2.fit_context(view, m)
        for v, srcs in (("orig", ((root / "fit.npz", "beta", "D_"), (OUT / "fits" / f"{view}_{m}_orig_crop.npz", "beta", None))),
                        ("va", ((root / "fit_e1.npz", "va_beta", "D_va_"), (root / "fit_e2_crop.npz", "va_beta", None))),
                        ("vb", ((root / "fit_e1.npz", "vb_beta", "D_vb_"), (root / "fit_e2_crop.npz", "vb_beta", None)))):
            beta = np.full((len(idx.keys), ctx["k_id"]), np.nan)
            ref = None
            for path, key, dpre in srcs:
                with np.load(path) as z:
                    pos = np.asarray([idx.pos[k] for k in keys_of(z)])
                    beta[pos] = np.asarray(z[key])
                    if dpre:
                        ref = (pos, {d: np.asarray(z[f"{dpre}{d}"]) for d in ("coef", "fr", "sr")})
            D = bp.mesh_distances(beta, ctx)
            for d, M in D.items():
                out[f"{m}_{v}_{d}"] = M
                ctrl[f"{m}_{v}_{d}"] = float(np.nanmax(np.abs(M[np.ix_(ref[0], ref[0])] - ref[1][d])))
            if v == "vb" and m in K:
                out[f"{m}_vb_comp"] = bp.composition(bp.identity_sizes(beta, ctx), D["sr"], ctx, K[m])
    return out, ctrl


def arcface_columns(view: str, idx) -> dict:
    """normal map: G0 (``g0/teacher/<vista>/teacher.npz``, Z = media delle 3 viste, L2); FaceVerse con espressioni:
    ``arcface_render_zs/fv_expr/normals`` (stessa configurazione, la G0 non ha l'insieme), Z come in G0. Ombreggiato:
    E12 (HIFI3D, FaceVerse con espressioni). Distanza 1 - coseno."""
    out = {}
    g0 = EV / "diagnostics" / "g0" / "teacher" / view / "teacher.npz"
    src = {"hifi3d": "hifi3d", "faceverse": "fv_expr"}.get(view)
    if g0.exists():
        with np.load(g0) as z:
            out["arcface_normals"] = cosine(np.asarray(z["Z"], np.float64)[ordered(keys_of(z), idx)])
    elif src:
        with np.load(ARC_FV / src / "normals" / "arcface_views.npz") as z:
            out["arcface_normals"] = cosine(np.asarray(z["E"], np.float64).mean(1)[ordered(keys_of(z), idx)])
    if src and view != "faceverse_neutral":
        with np.load(ARC_FV / src / "shaded" / "arcface_views.npz") as z:
            out["arcface_shaded"] = cosine(np.asarray(z["E"], np.float64).mean(1)[ordered(keys_of(z), idx)])
    return out


def spectral_columns(path: Path, order) -> dict:
    """ShapeDNA k=50 (autovalori 1..50), HKS e WKS globali, L2 (``comp_summarize.competitor_distances``)."""
    with np.load(path) as z:
        ev = np.asarray(z["evals"], np.float64)[order]
        desc = {"shapedna_k50": ev[:, 1:51], "hks": np.asarray(z["hks"], np.float64)[order],
                "wks": np.asarray(z["wks"], np.float64)[order]}
    return {k: np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)) for k, X in desc.items()}


def view_columns(view: str, idx) -> tuple[dict, dict]:
    """{colonna: D (600, 600)} di una vista a 6 topologie, controlli."""
    ctrl = {}
    arms = fp.model_columns(view, idx)
    D = {c: M for c, M in arms.items() if c in METHODS}
    D.update(geo_columns(view, idx))
    D["varifold"], ctrl["varifold_check_rel_max_vs_shards"] = varifold_column(view, idx)
    F, ctrl["fits_nocrop_vs_saved_max_abs"] = fit_columns(view, idx)
    D.update(F)
    D.update(arcface_columns(view, idx))
    sp = SPEC if view == "hifi3d" else OUT / "spectral" / f"{view}.npz"
    with np.load(sp) as z:
        o = ordered(keys_of(z), idx)
    D.update(spectral_columns(sp, o))
    if view == "hifi3d":
        for name in ("openshape", "uni3d"):
            with np.load(AAU / "runs" / "competitors_hifi3d" / f"emb_{name}.npz") as z:
                D[name] = cosine(np.asarray(z["E"], np.float64)[ordered(keys_of(z), idx)])
    sc = blmm.scalars_of(view)
    D["est_cs"] = log_abs([sc[blmm.rel(blmm.mesh_path(view, s, t))]["cs"] for s, t in idx.keys])
    orc = fp.oracle_S(blmm.VIEWS[view]["gt"])
    D["oracle_size"] = log_abs([orc[s] for s, _ in idx.keys])
    return D, ctrl


# ------------------------------------------------------------------------------------------- FaMoS

def famos_extra(view: str, keys: list) -> dict:
    """``bp_paired.EXTRA`` per FaMoS (15 x 15 sulle chiavi di fit.npz): NICP per coppia in mm (shard di blmm), NICP su
    template in mm, spettrali, ArcFace (fm_compute / fm_arcface)."""
    if view != "famos":
        return {}
    rows = blmm.famos_manifest()
    names = [r["name"] for r in rows]
    vid = [s for s, _ in keys]
    with np.load(blmm.DATASETS / "FAMOS" / "test_view" / "gt_matrix.npz") as z:
        gt_names = [str(x) for x in z["names"]]
    s2i = {s: k for k, s in enumerate(gt_names)}
    subj = np.asarray([s2i[r["view_id"]] for r in rows])
    kind, role = np.asarray([r["kind"] for r in rows]), np.asarray([r["role"] for r in rows])
    gal = {k: (lambda g: g[np.argsort(subj[g])])(np.flatnonzero((kind == k) & (role == "gallery"))) for k in ("scan", "reg")}
    gal_all = np.concatenate([gal["scan"], gal["reg"]])
    row_of = {rows[i]["view_id"]: i for i in gal["scan"]}
    col_of = {rows[i]["view_id"]: j for j, i in enumerate(gal_all) if kind[i] == "scan"}
    out = {}
    for mode in ("mm", "cs"):
        M = np.full((len(rows), len(gal_all)), np.nan)
        for p in sorted((blmm.OUT_ROOT / "famos" / mode).glob("nicp_*of*.npz")):
            with np.load(p) as z:
                if [str(x) for x in z["gallery"]] != [names[i] for i in gal_all]:
                    raise SystemExit(f"{p}: galleria diversa")
                M[z["rows"]] = z["nicp_p2tri"]
        # cs: gia' in fact_paired.famos_frame; qui solo come controllo (``fmchk_``)
        out[("" if mode == "mm" else "fmchk_") + f"{mode}_nicp_p2tri"] = \
            np.asarray([[M[row_of[a], col_of[b]] for b in vid] for a in vid])
    with np.load(OUT / "famos_template" / "mm_template.npz") as z:
        pos = {k: i for i, k in enumerate(keys_of(z))}
        R = np.asarray(z["R"], np.float64)[[pos[k] for k in keys]]
    out["mm_nicp_template"] = be.irt.template_distances(R, R)
    with np.load(OUT / "spectral" / "famos.npz") as z:
        pos = {k: i for i, k in enumerate(keys_of(z))}
    out.update(spectral_columns(OUT / "spectral" / "famos.npz", np.asarray([pos[k] for k in keys])))
    with np.load(OUT / "arcface" / "teacher" / "famos_test" / "teacher.npz") as z:
        pos = {str(s): i for i, s in enumerate(z["subjects"])}
        out["arcface_normals"] = cosine(np.asarray(z["Z"], np.float64)[[pos[s] for s in vid]])
    return out


# ------------------------------------------------------------------------------------------- plug-in

def plugin_dirs(roots: list) -> tuple[list, list]:
    """Cartelle dei plug-in DICHIARATE: ``<radice>/<metodo>/`` o ``<radice>/<metodo>/<tag>/`` che contengono
    ``fm_plugin.json`` (il file di marcatura). Le altre (pilot/, prep/, cartelle di lavoro) si ignorano e si elencano."""
    out, ignored = [], []
    for r in roots:
        if not r.is_dir():
            continue
        for d in sorted(x for x in r.iterdir() if x.is_dir() and not x.name.startswith(".")):
            subs = [t for t in sorted(d.iterdir()) if t.is_dir() and not t.name.startswith(".")]
            for name, x in [(d.name, d)] + [(f"{d.name}/{t.name}", t) for t in subs]:
                if (x / "fm_plugin.json").exists():
                    out.append((name, x))
                elif x is d or any(x.glob("*.npz")) or any(x.glob("*.csv")):
                    ignored.append(name)
    return out, ignored


def reading_gts(r: str, info: dict) -> tuple:
    """GT dichiarate di una lettura: ``fm_plugin.json``, altrimenti dal nome (``fr`` / ``*_fr`` -> FR, ``sr`` / ``*_sr``
    -> SR), altrimenti entrambe."""
    if "gt" in info:
        return tuple(info["gt"])
    if r == "fr" or r.endswith("_fr"):
        return FR
    if r == "sr" or r.endswith("_sr"):
        return SR
    return BOTH


def plugin_columns(view: str, keys: list, pdirs: list) -> tuple[dict, dict, list]:
    """{colonna: D sulle ``keys``} dei plug-in per una vista, {colonna: (etichetta, famiglia, GT, False)}, plug-in senza
    file per la vista; NaN dove manca la mesh o la coppia."""
    D, meta, missing = {}, {}, []
    pos = {k: i for i, k in enumerate(keys)}
    for name, d in pdirs:
        cfg = json.loads((d / "fm_plugin.json").read_text()) if (d / "fm_plugin.json").exists() else {}
        rd = cfg.get("readings", {})
        mats = {}
        csvp = next((q for q in (d / f"{view}.csv", d / f"pairs_{view}.csv") if q.exists()), None)
        if (d / f"{view}.npz").exists():
            with np.load(d / f"{view}.npz", allow_pickle=True) as z:
                kz = keys_of(z)
                sel = [(i, pos[k]) for i, k in enumerate(kz) if k in pos]
                if not sel:
                    raise SystemExit(f"plug-in {name}: {d / f'{view}.npz'} non contiene nessuna mesh del dominio {view} "
                                     f"(es. {kz[:2]} contro {keys[:2]})")
                a = np.asarray([x for x, _ in sel], dtype=np.int64)
                b = np.asarray([y for _, y in sel], dtype=np.int64)
                for f in z.files:
                    if f == "D" or f.startswith("D_"):
                        M = np.full((len(keys), len(keys)), np.nan)
                        M[np.ix_(b, b)] = np.asarray(z[f], np.float64)[np.ix_(a, a)]
                        mats[f[2:] if f.startswith("D_") else "-"] = M
        elif csvp is not None:
            df = pd.read_csv(csvp, dtype={"subject_a": str, "subject_b": str, "topology_a": str, "topology_b": str})
            ia = df.apply(lambda r: pos.get((r.subject_a, r.topology_a), -1), axis=1).to_numpy()
            ib = df.apply(lambda r: pos.get((r.subject_b, r.topology_b), -1), axis=1).to_numpy()
            ia, ib = ia.astype(np.int64), ib.astype(np.int64)
            ok = (ia >= 0) & (ib >= 0)
            if not ok.any():
                raise SystemExit(f"plug-in {name}: {csvp} non contiene nessuna coppia di mesh del dominio {view}")
            for c in df.columns:
                if c in ("subject_a", "topology_a", "subject_b", "topology_b"):
                    continue
                M = np.full((len(keys), len(keys)), np.nan)
                x = df[c].to_numpy(np.float64)
                M[ib[ok], ia[ok]] = x[ok]                    # verso invertito solo dove manca quello diretto
                M[ia[ok], ib[ok]] = x[ok]
                mats[c] = M
        else:
            missing.append(name)
        for r, M in mats.items():
            col = f"plugin:{name}" + ("" if r == "-" else f"|{r}")
            info = rd.get(r, {})
            D[col] = M
            meta[col] = (info.get("label", f"{cfg.get('label', name)}" + ("" if r == "-" else f", {r}")),
                         cfg.get("family", "plug-in"), reading_gts(r, info), False)
    return D, meta, missing


# ------------------------------------------------------------------------------------------- righe

def rows_view(view: str):
    """(df con GT, idx, maschera crop/stessa) = all_cross + stessa topologia sulle stesse coppie di soggetti."""
    df, idx, _ = bpe2.all_cross_rows(view)
    key = ["subject_a", "subject_b"]
    g = df.groupby(key)[["gt_fr", "gt_sr", "oracle_size"]]
    spread = float((g.max() - g.min())[["gt_fr", "gt_sr"]].to_numpy().max())
    pairs = g.first().reset_index()
    same = pd.concat([pairs.assign(topology_a=t, topology_b=t) for t in TOPOS], ignore_index=True)
    cols = key + ["topology_a", "topology_b", "gt_fr", "gt_sr"]
    rows = pd.concat([df[cols], same[cols]], ignore_index=True)
    return rows, idx, spread


def groups_of(ta: np.ndarray, tb: np.ndarray, mask: np.ndarray) -> dict:
    """{(insieme_righe, perturbazione): righe} dentro la maschera."""
    cross = (ta != tb) & mask
    crop = (ta == "crop") | (tb == "crop")
    G = {("nocrop", "-"): cross & ~crop, ("crop", "crop"): cross & crop, ("all_cross", "-"): cross,
         ("stessa_topologia", "tutte"): (ta == tb) & mask}
    for t in TOPOS:                                       # P senza crop: contro le altre senza crop; crop: contro tutte
        G[("con_perturbazione", t)] = cross & ((ta == t) | (tb == t)) & (~crop if t != "crop" else True)
        G[("stessa_topologia", t)] = (ta == t) & (tb == t) & mask
    for i, a in enumerate(TOPOS):
        for b in TOPOS[i + 1:]:
            G[("coppia_topologie", f"{a}~{b}")] = cross & (((ta == a) & (tb == b)) | ((ta == b) & (tb == a)))
    for a in TOPOS:
        for b in TOPOS:
            if a != b:
                G[("coppia_ordinata", f"{a}>{b}")] = (ta == a) & (tb == b) & mask
    return {k: np.flatnonzero(v) for k, v in G.items()}


# ----------------------------------------------------------------------------------------- repliche

def _rep(k: int) -> np.ndarray:
    """rho (gruppo, gt, metodo) di una replica: ``fact_paired._rep`` (ranghi delle righe ripetute c_a c_b, Pearson)."""
    from scipy.stats import rankdata
    J = _J
    c = J["counts"][k]
    wt = c[J["sa"]].astype(np.int64) * c[J["sb"]]
    out = np.full((len(J["groups"]), len(GTS), len(J["methods"])), np.nan)
    for gi, rows in enumerate(J["groups"]):
        w = wt[rows]
        keep = w > 0
        if keep.sum() < 3:
            continue
        sel, w = rows[keep], w[keep]
        rg = [rankdata(np.repeat(J["gt"][g][sel], w)) for g in GTS]
        for mi, m in enumerate(J["methods"]):
            r = rankdata(np.repeat(J["cols"][m][sel], w))
            for ti in range(len(GTS)):
                out[gi, ti, mi] = fp._pearson(r, rg[ti])
    return out


def replicas(cols: dict, gt: dict, sa, sb, counts, groups: dict, methods: list, workers: int) -> np.ndarray:
    """V (1 + repliche, gruppi, gt, metodi)."""
    global _J
    _J = {"counts": counts, "sa": sa, "sb": sb, "cols": cols, "gt": gt, "groups": list(groups.values()),
          "methods": methods}
    with mp.get_context("fork").Pool(workers) as pool:
        reps = pool.map(_rep, range(len(counts)), chunksize=2)
    return np.stack(reps)


def ci(x: np.ndarray) -> tuple[float, float]:
    b = x[1:][np.isfinite(x[1:])]
    return tuple(np.percentile(b, [2.5, 97.5])) if len(b) else (np.nan, np.nan)


# ------------------------------------------------------------------------------------------ record

def records(view: str, V: np.ndarray, gkeys: list, groups: dict, methods: list, meta: dict, seed: int,
            n_subj: int, tag: str = "") -> tuple[list, list]:
    """Righe di spearman.csv e deltas.csv da V (con le medie di topo_pairs_e3)."""
    S, Dl = [], []
    mi = {m: k for k, m in enumerate(methods)}
    Vd = {g: V[:, k] for k, g in enumerate(gkeys)}
    nrows = {g: len(groups[g]) for g in gkeys}
    xo = [g for g in gkeys if g[0] == "coppia_ordinata" and "crop" not in g[1]]
    so = [g for g in gkeys if g[0] == "stessa_topologia" and g[1] in NOCROP]
    if len(xo) == 20 and len(so) == 5:
        Vd[("media_20_ordinate_senza_crop", "-")] = np.mean([Vd[g] for g in xo], axis=0)
        Vd[("media_5_stessa_topologia", "-")] = np.mean([Vd[g] for g in so], axis=0)
        nrows[("media_20_ordinate_senza_crop", "-")] = sum(nrows[g] for g in xo)
        nrows[("media_5_stessa_topologia", "-")] = sum(nrows[g] for g in so)

    def base(g, gt):
        return {"dominio": view, "gt": gt, "insieme_righe": g[0], "perturbazione": g[1], "n_righe": nrows[g],
                "n_soggetti": n_subj, "seme": seed, "passo": tag or "comune"}

    for g, A in Vd.items():
        for ti, gt in enumerate(GTS):
            for m in methods:
                x = A[:, ti, mi[m]]
                if not np.isfinite(x[0]):
                    continue
                lo, hi = ci(x)
                lab, fam, dec, ours = meta[m]
                S.append({**base(g, gt), "metodo": lab, "colonna": m, "famiglia": fam, "nostro": ours,
                          "dichiarata": gt in dec, "post_hoc": post_hoc(m, view), "rho": x[0], "ic_lo": lo, "ic_hi": hi})

            def delta(a, b, tipo, conf, ruolo="-", piu="-"):
                if a not in mi or b not in mi:
                    return
                xa, xb = A[:, ti, mi[a]], A[:, ti, mi[b]]
                d = xa - xb
                if not np.isfinite(d[0]):
                    return
                lo, hi = ci(d)
                bb = d[1:][np.isfinite(d[1:])]
                Dl.append({**base(g, gt), "tipo": tipo, "confronto": conf, "ruolo": ruolo, "piu_allineato": piu,
                           "metodo_a": meta[a][0], "colonna_a": a, "post_hoc_a": post_hoc(a, view),
                           "metodo_b": meta[b][0], "colonna_b": b, "post_hoc_b": post_hoc(b, view), "rho_a": xa[0],
                           "rho_b": xb[0], "delta": d[0], "ic_lo": lo, "ic_hi": hi,
                           "p_le0": float((bb <= 0).mean()) if len(bb) else np.nan})

            comps = [m for m in methods if not meta[m][3] and gt in meta[m][2]]
            if g[0] in ("coppia_ordinata",):
                continue
            for r in REF[gt]:
                for c in comps:
                    delta(r, c, "vs_nostro", f"{meta[r][0]} - {meta[c][0]}")
            for lg, a, b, step, ruolo, piu in LADDER:
                if lg == gt:
                    delta(a, b, "scala_allineamento", step, ruolo, piu)
            ours = [m for m in methods if meta[m][3] and gt in meta[m][2] and np.isfinite(A[0, ti, mi[m]])]
            rivals = [m for m in comps if m not in REFERENCES and np.isfinite(A[0, ti, mi[m]])]
            if ours and rivals:
                bo = max(ours, key=lambda m: A[0, ti, mi[m]])
                bc = max(rivals, key=lambda m: A[0, ti, mi[m]])
                delta(bo, bc, "migliore_vs_migliore", f"miglior nostro ({meta[bo][0]}) - miglior concorrente "
                                                      f"({meta[bc][0]})")
    return S, Dl


# ----------------------------------------------------------------------------------------- controlli

def size_confound(cols: dict, core: list, groups: dict) -> dict:
    """Spearman (punto, righe della maschera) di ogni colonna con la taglia oracolo e con la taglia stimata: quanto una
    distanza segue la taglia (confusione dei passi in mm della scala dell'allineamento)."""
    from scipy.stats import rankdata
    out = {}
    for g in (("nocrop", "-"), ("crop", "crop"), ("all_cross", "-")):
        if g not in groups:
            continue
        r = groups[g]
        ref = {k: rankdata(cols[k][r]) for k in ("oracle_size", "est_cs") if k in cols}
        out[g[0]] = {c: {k: round(fp._pearson(rankdata(cols[c][r]), v), 4) for k, v in ref.items()} for c in core}
    return out


def t1_control(view: str, my: dict, idx, n_boot: int, workers: int, sens: list) -> dict:
    """C1: le colonne di qui sulle righe, maschera e repliche della T1 (``bp_paired.frame`` con le colonne di
    ``bp_paired_e2``) contro spearman_e2.csv: scarto massimo (punto, IC)."""
    bpp.COLS[:] = bpp.NEW + bpp.NEW_E1 + bpe2.COMP + sens
    bpp.EXTRA[:] = [bpe2.extra_matrices]
    cols, M, bl, sa, sb, counts, seed, _ = bpp.frame(view, n_boot)
    mask = (sa != sb) & np.all([np.isfinite(v) for v in cols.values()], axis=0)
    df, idx2, _ = fp.rows_for("faceverse" if view == "faceverse_neutral" else view)
    if idx2.keys != idx.keys:
        raise SystemExit(f"{view}: indice della T1 diverso")
    mapping = {"mm_rigid_icp_chamfer": ("baseline", "ICP + Chamfer in mm"),
               "cs_rigid_icp_chamfer": ("baseline", "ICP + Chamfer (cs)"),
               "cs_nicp_p2tri": ("baseline", "NICP per coppia (cs)"),
               "mm_nicp_template": ("baseline", "NICP su template in mm"), "est_cs": ("baseline", "taglia stimata"),
               "oracle_size": ("baseline", "taglia oracolo"), "varifold": ("baseline", bpp.LABEL["varifold"])}
    for m in bp.MODELS:
        for d in ("coef", "fr", "sr"):
            mapping[f"{m}_orig_{d}"] = ("baseline", bpp.LABEL[f"{m}_{d}"])
            for v in ("va", "vb"):
                mapping[f"{m}_{v}_{d}"] = ("baseline", bpp.LABEL[f"{m}_{v}_{d}"])
        mapping[f"{m}_vb_comp"] = ("baseline", bpe2.LABEL[f"{m}_vb_comp"])
    use = [c for c in my if c in mapping or (c in METHODS and "|" in c)]
    ia = np.asarray([idx.pos[k] for k in zip(df["subject_a"].astype(str), df["topology_a"].astype(str))])
    ib = np.asarray([idx.pos[k] for k in zip(df["subject_b"].astype(str), df["topology_b"].astype(str))])
    sel = np.flatnonzero(mask)
    cc = {c: my[c][ia, ib][sel] for c in use}
    gt = {g: cols[f"gt_{g}"][sel] for g in GTS}
    V = replicas(cc, gt, sa[sel], sb[sel], counts, {"t1": np.arange(len(sel))}, use, workers)
    ref = pd.read_csv(EV / "baselines_param" / "spearman_e2.csv")
    ref = ref[(ref["domain"] == view) & (ref["kind"] == "rho") & ~ref["group"].str.startswith("all_cross")]
    diffs, n = [], 0
    for k, c in enumerate(use):
        arm, dist = mapping[c] if c in mapping else c.split("|")
        for ti, g in enumerate(GTS):
            r = ref[(ref["arm"] == arm) & (ref["distance"] == dist) & (ref["gt"] == g)]
            if not len(r):
                continue
            x = V[:, 0, ti, k]
            lo, hi = ci(x)
            r = r.iloc[0]
            diffs.append(max(abs(x[0] - r.arm_point), abs(lo - r.arm_ci_low), abs(hi - r.arm_ci_high)))
            n += 1
    return {"n_compared": n, "max_abs_diff_point_ci": float(max(diffs)) if diffs else None,
            "rows_t1_mask": int(mask.sum()), "seed": int(seed)}


def point_controls(S: pd.DataFrame) -> dict:
    """C2: punti della matrice (maschera comune di qui) contro le fonti precedenti; lo scarto misura l'effetto della
    maschera (e del seme sugli IC, qui non confrontati)."""
    out = {}
    e2 = pd.read_csv(EV / "baselines_param" / "spearman_e2.csv")
    e2 = e2[(e2["kind"] == "rho") & ~e2["group"].str.startswith("all_cross")]
    e2c = {(r.domain, r.gt, f"{r.arm}|{r.distance}" if r.arm != "baseline" else r.distance): (r.arm_point, r.n_rows)
           for r in e2.itertuples()}
    inv = {"ICP + Chamfer in mm": "mm_rigid_icp_chamfer", "ICP + Chamfer (cs)": "cs_rigid_icp_chamfer",
           "NICP per coppia (cs)": "cs_nicp_p2tri", "NICP su template in mm": "mm_nicp_template",
           "taglia stimata": "est_cs", "taglia oracolo": "oracle_size", bpp.LABEL["varifold"]: "varifold"}
    for m in bp.MODELS:
        for d in ("coef", "fr", "sr"):
            inv[bpp.LABEL[f"{m}_{d}"]] = f"{m}_orig_{d}"
            for v in ("va", "vb"):
                inv[bpp.LABEL[f"{m}_{v}_{d}"]] = f"{m}_{v}_{d}"
        inv[bpe2.LABEL[f"{m}_vb_comp"]] = f"{m}_vb_comp"
    e2c = {(a, b, inv.get(c, c)): v for (a, b, c), v in e2c.items()}
    x = S[(S["insieme_righe"] == "nocrop") & (S["passo"] == "comune")]
    d = [(abs(r.rho - e2c[(r.dominio, r.gt, r.colonna)][0]), r.dominio, r.colonna, r.gt, r.n_righe,
          e2c[(r.dominio, r.gt, r.colonna)][1]) for r in x.itertuples() if (r.dominio, r.gt, r.colonna) in e2c]
    out["spearman_e2_nocrop"] = summarize_diffs(d)
    e3 = pd.read_csv(EV / "baselines_param" / "spearman_e3.csv")
    e3 = e3[(e3["kind"] == "rho") & (e3["region"].isin(["intero", "-"]))]
    for mine, theirs in (("crop", "righe col crop"), ("all_cross", "all_cross")):
        ref = {(r.domain, r.gt, r.column): (r.point, r.n_rows) for r in e3[e3["group"] == theirs].itertuples()}
        x = S[(S["insieme_righe"] == mine) & (S["passo"] == "comune")]
        d = [(abs(r.rho - ref[(r.dominio, r.gt, r.colonna)][0]), r.dominio, r.colonna, r.gt, r.n_righe,
              ref[(r.dominio, r.gt, r.colonna)][1]) for r in x.itertuples() if (r.dominio, r.gt, r.colonna) in ref]
        out[f"spearman_e3_{mine}"] = summarize_diffs(d)
    g0 = pd.read_csv(EV / "diagnostics" / "g0_spearman.csv")
    for mine, theirs in (("nocrop", "nocrop_cross"), ("crop", "righe col crop")):
        ref = {(r.set, r.gt): (r.rho, r.n_rows) for r in g0[(g0["group"] == theirs) & (g0["column"] == "teacher|d_T") &
                                                              (g0["set"] != "famos")].itertuples()}   # G0 famos = TRAIN
        x = S[(S["insieme_righe"] == mine) & (S["colonna"] == "arcface_normals") & (S["passo"] == "comune")]
        d = [(abs(r.rho - ref[(r.dominio, r.gt)][0]), r.dominio, r.colonna, r.gt, r.n_righe, ref[(r.dominio, r.gt)][1])
             for r in x.itertuples() if (r.dominio, r.gt) in ref]
        out[f"g0_arcface_{mine}"] = summarize_diffs(d)
    tp = pd.read_csv(EV / "baselines_param" / "topo_pairs_e3.csv")
    for mine, theirs in (("media_20_ordinate_senza_crop", "media 20 senza crop"),
                         ("media_5_stessa_topologia", "media 5 stessa topologia")):
        ref = {(r.domain, r.gt, r.method): (r.point, r.n_rows) for r in tp[tp["quantity"] == theirs].itertuples()}
        x = S[(S["insieme_righe"] == mine) & (S["passo"] == "comune")]
        d = [(abs(r.rho - ref[(r.dominio, r.gt, r.colonna)][0]), r.dominio, r.colonna, r.gt, r.n_righe,
              ref[(r.dominio, r.gt, r.colonna)][1]) for r in x.itertuples() if (r.dominio, r.gt, r.colonna) in ref]
        out[f"topo_pairs_e3_{mine}"] = summarize_diffs(d)
    e12 = pd.read_csv(EV / "e12" / "methods_spearman.csv")
    e12 = e12[(e12["domain"] == "hifi3d") & (e12["group"] == "nocrop_cross") & (e12["gt"] == "F_rig_rob")]
    ref = {r.method: (r.point, r.n_rows) for r in e12.itertuples()}
    x = S[(S["insieme_righe"] == "nocrop") & (S["dominio"] == "hifi3d") & (S["gt"] == "fr") & (S["passo"] == "comune")]
    d = [(abs(r.rho - ref[r.colonna][0]), r.dominio, r.colonna, r.gt, r.n_righe, ref[r.colonna][1])
         for r in x.itertuples() if r.colonna in ref and r.colonna in ("shapedna_k50", "hks", "wks", "openshape", "uni3d")]
    out["e12_hifi3d_fr_spettrali_reti"] = summarize_diffs(d)
    return out


def summarize_diffs(d: list) -> dict:
    if not d:
        return {"n": 0}
    w = max(d)
    return {"n": len(d), "max_abs_diff_point": float(w[0]),
            "worst": f"{w[1]} {w[2]} {w[3]}: righe qui {w[4]}, fonte {w[5]}",
            "n_equal_1e-12": int(sum(x[0] < 1e-12 for x in d)), "n_same_rows": int(sum(x[4] == x[5] for x in d))}


# ---------------------------------------------------------------------------------------------- dominio

def run_view(view: str, n_boot: int, workers: int, pdirs: list, sens: list) -> tuple[list, list, list, dict]:
    t0 = time.time()
    NA, info = [], {}
    if view == "famos":
        bpp.COLS[:] = bpp.NEW + bpp.NEW_E1 + bpe2.COMP + sens
        bpp.EXTRA[:] = [bpe2.extra_matrices, famos_extra]
        cols, M, bl, sa, sb, counts, seed, ctrl = bpp.frame("famos", n_boot)
        ren = {**{f"{m}_{d}": f"{m}_orig_{d}" for m in bp.MODELS for d in ("coef", "fr", "sr")}}
        cols = {ren.get(c, c): v for c, v in cols.items()}
        info["famos_controls"] = [f"{k}: {d:.1e}" for k, d in ctrl]
        tj = OUT / "arcface" / "teacher" / "famos_test" / "timing.json"
        info["famos_arcface_detector"] = json.loads(tj.read_text()).get("n_detected_by_yaw") if tj.exists() else None
        info["famos_cs_nicp_vs_famos_frame"] = float(np.nanmax(np.abs(cols.pop("fmchk_cs_nicp_p2tri") -
                                                                      cols["cs_nicp_p2tri"])))
        _, keys = bpp.new_matrices("famos")
        Pd, pmeta, pmiss = plugin_columns("famos", keys, pdirs)
        if Pd:
            with np.load(blmm.DATASETS / "FAMOS" / "test_view" / "gt_matrix.npz") as z:
                gt_names = [str(x) for x in z["names"]]
            order = np.asarray([[s for s, _ in keys].index(n) for n in gt_names])
            cols.update({c: Mx[np.ix_(order, order)][sa, sb] for c, Mx in Pd.items()})
        gt = {g: cols[f"gt_{g}"] for g in GTS}
        n_subj = 15
        groups_all = None
    else:
        rows, idx, spread = rows_view(view)
        D, info["load"] = view_columns(view, idx)
        Pd, pmeta, pmiss = plugin_columns(view, idx.keys, pdirs)
        D.update(Pd)
        ia = np.asarray([idx.pos[k] for k in zip(rows["subject_a"].astype(str), rows["topology_a"].astype(str))])
        ib = np.asarray([idx.pos[k] for k in zip(rows["subject_b"].astype(str), rows["topology_b"].astype(str))])
        cols = {c: Mx[ia, ib] for c, Mx in D.items()}
        gt = {g: rows[f"gt_{g}"].to_numpy(np.float64) for g in GTS}
        subjects = np.array(sorted(set(rows["subject_a"]) | set(rows["subject_b"])))
        s2i = {s: i for i, s in enumerate(subjects)}
        sa, sb = rows["subject_a"].map(s2i).to_numpy(), rows["subject_b"].map(s2i).to_numpy()
        info["t1_control"] = t1_control(view, D, idx, n_boot, workers, sens)
        seed = info["t1_control"]["seed"]
        rng = np.random.default_rng(seed)
        counts = [np.ones(len(subjects), dtype=np.int64)] + \
            [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(n_boot)]
        n_subj = len(subjects)
        info.update(rows=int(len(rows)), gt_spread_within_subject_pair=spread)
        groups_all = (rows["topology_a"].to_numpy(str), rows["topology_b"].to_numpy(str))
    meta = {**{c: METHODS[c] for c in METHODS}, **pmeta}
    NA += [{"dominio": view, "metodo": f"plug-in {n}", "colonna": f"plugin:{n}", "insieme_righe": "tutti",
            "motivo": "nessun file per questo dominio al momento del calcolo"} for n in pmiss]
    core = [c for c in METHODS if c in cols]
    for c in METHODS:                                     # colonne assenti o tutte NaN: non applicabili
        if c not in core or not np.isfinite(cols[c][sa != sb]).any():
            why = next((w[view] for k, w in NA_REASON.items() if k in c and view in w), "non calcolato per questo dominio")
            NA.append({"dominio": view, "metodo": label(c), "colonna": c, "insieme_righe": "tutti", "motivo": why})
            if c in core:
                core.remove(c)
    gok = np.isfinite(gt["fr"]) & np.isfinite(gt["sr"]) & (sa != sb)
    same = np.zeros(len(sa), bool) if view == "famos" else groups_all[0] == groups_all[1]
    fin = {c: np.isfinite(cols[c]) for c in core}
    # righe a topologie diverse: tutte le colonne; stessa topologia: le colonne definite li' (le distanze per coppia di
    # blmm non hanno soggetti diversi con la stessa topologia; ICP l'aggiunge fm_compute icp-same, il NICP no)
    core_same = [c for c in core if fin[c][same & gok].any()]
    mask = (gok & ~same & np.all(list(fin.values()), axis=0)) | \
        (gok & same & np.all([fin[c] for c in core_same] or [same], axis=0))
    for c in core:
        if same.any() and c not in core_same:
            NA.append({"dominio": view, "metodo": label(c), "colonna": c, "insieme_righe": "stessa_topologia",
                       "motivo": "distanza per coppia mai calcolata fra soggetti diversi con la stessa topologia (NICP per "
                                 "coppia: 2-4 s a coppia, circa 200 ore-core per le 4 viste e i 2 modi; non calcolato)"})
    info["mask"] = {"rows_mask": int(mask.sum()), "rows_mask_cross": int((mask & ~same).sum()),
                    "rows_mask_same": int((mask & same).sum()), "rows_valid_gt": int(gok.sum()),
                    "columns_missing_on_same": [c for c in core if c not in core_same] if same.any() else [],
                    "rows_dropped_by_column_cross": {c: int((gok & ~same & ~f).sum()) for c, f in fin.items()
                                                     if (gok & ~same & ~f).any()}}
    pfull = [c for c in Pd if np.isfinite(cols[c][mask]).all()]
    ppart = [c for c in Pd if c not in pfull and np.isfinite(cols[c][mask]).any()]
    for c in Pd:
        if c not in pfull and c not in ppart:
            NA.append({"dominio": view, "metodo": meta[c][0], "colonna": c, "insieme_righe": "tutti",
                       "motivo": "plug-in senza righe finite nella maschera comune"})
    if view == "famos":
        groups = {("nocrop", "-"): np.flatnonzero(mask)}
        for s in ("crop", "all_cross", "con_perturbazione", "coppia_topologie", "stessa_topologia"):
            NA.append({"dominio": view, "metodo": "tutti", "colonna": "-", "insieme_righe": s, "motivo": SET_NA["famos"]})
    else:
        groups = groups_of(*groups_all, mask)
    methods = core + pfull
    V = replicas({m: cols[m] for m in methods}, gt, sa, sb, counts, groups, methods, workers)
    S, Dl = records(view, V, list(groups), groups, methods, meta, seed, n_subj)
    info["spearman_con_taglia"] = size_confound(cols, core, groups)
    if view == "famos":
        det = info.get("famos_arcface_detector") or {}
        NA.append({"dominio": view, "metodo": label("arcface_normals"), "colonna": "arcface_normals",
                   "insieme_righe": "nocrop", "motivo": "CAUTELA (calcolato, non escluso): sulle scansioni parziali il "
                   f"detector trova il volto in {sum(det.values())}/45 render ({det}); il ritaglio a +30 gradi e' tarato su "
                   f"{det.get('+030.00', '?')} render su 15, fuori dal regime validato in G0"})
    else:                                                 # NICP: selezione delle righe per i suoi fallimenti
        others = [c for c in core if c not in NICP]
        m_nf = (gok & ~same & np.all([fin[c] for c in others], axis=0)) | (mask & same)
        lost = m_nf & ~same & ~mask
        ta, tb = groups_all
        info["nicp_failures"] = {"rows_lost": int(lost.sum()), "by_topology": {
            t: {"lost": int((lost & ((ta == t) | (tb == t))).sum()),
                "rows_without_nicp_mask": int((m_nf & ~same & ((ta == t) | (tb == t))).sum())} for t in TOPOS}}
        if lost.any():                                    # sensibilita': fallimento = distanza massima, righe tutte
            filled = {c: (np.where(np.isfinite(cols[c]), cols[c], np.nanmax(cols[c][gok]) * (1 + 1e-9))
                          if c in NICP else cols[c]) for c in methods}
            g3 = groups_of(ta, tb, m_nf)
            V3 = replicas(filled, gt, sa, sb, counts, g3, methods, workers)
            s3, d3 = records(view, V3, list(g3), g3, methods, meta, seed, n_subj,
                             tag="sensibilita': NICP fallito = distanza massima")
            S += s3
            Dl += d3
    for c in ppart:                                       # plug-in con buchi: passo a parte, righe proprie
        m2 = mask & np.isfinite(cols[c])
        g2 = {("nocrop", "-"): np.flatnonzero(m2)} if view == "famos" else groups_of(*groups_all, m2)
        refs = sorted({r for g in GTS for r in REF[g] if r in core})
        V2 = replicas({m: cols[m] for m in refs + [c]}, gt, sa, sb, counts, g2, refs + [c], workers)
        s2, d2 = records(view, V2, list(g2), g2, refs + [c], meta, seed, n_subj, tag=f"righe proprie di {c}")
        S += [r for r in s2 if r["colonna"] == c]
        Dl += [r for r in d2 if r["tipo"] == "vs_nostro"]
    info.update(seed=int(seed), n_methods=len(methods), plugins_full=pfull, plugins_partial=ppart,
                seconds=time.time() - t0)
    print(f"[fm-eval] {view}: {len(methods)} metodi, {int(mask.sum())} righe nella maschera, {len(groups)} insiemi, "
          f"seme {seed}, {time.time() - t0:.0f}s", flush=True)
    return S, Dl, NA, info


# ------------------------------------------------------------------------------------------- markdown

def f3(r, lo, hi, sign=False) -> str:
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"{f.format(r)} [{f.format(lo)}, {f.format(hi)}]"


SETS_MD = [("nocrop", "-"), ("crop", "crop"), ("all_cross", "-")] + [("con_perturbazione", t) for t in NOCROP] + \
    [("stessa_topologia", "tutte")]
SENS = "sensibilita': NICP fallito = distanza massima"


def sname(s: str, p: str) -> str:
    return s if p in ("-", "crop") and s in ("nocrop", "crop", "all_cross") else f"{s} {p}"


def best_table(S: pd.DataFrame, Dl: pd.DataFrame, views: tuple) -> list:
    B = Dl[(Dl["tipo"] == "migliore_vs_migliore") & (Dl["passo"] == "comune")]
    md = ["| dominio | GT | insieme | righe | miglior nostro | rho | miglior concorrente | rho | delta [IC] |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v in views:
        for g in GTS:
            for s, p in SETS_MD:
                r = B[(B["dominio"] == v) & (B["gt"] == g) & (B["insieme_righe"] == s) & (B["perturbazione"] == p)]
                if not len(r):
                    continue
                r = r.iloc[0]
                x = S[(S["dominio"] == v) & (S["gt"] == g) & (S["insieme_righe"] == s) & (S["perturbazione"] == p) &
                      (S["passo"] == "comune")]
                ra, rb = (x[x["colonna"] == c].iloc[0] for c in (r.colonna_a, r.colonna_b))
                md.append(f"| {v} | {g.upper()} | {sname(s, p)} | {int(r.n_righe)} | {r.metodo_a} | "
                          f"{f3(ra.rho, ra.ic_lo, ra.ic_hi)} | {r.metodo_b} | {f3(rb.rho, rb.ic_lo, rb.ic_hi)} | "
                          f"{f3(r.delta, r.ic_lo, r.ic_hi, True)} |")
    return md + [""]


def ladder_table(Dl: pd.DataFrame, views: tuple, ruolo: str, passo: str = "comune") -> list:
    L = Dl[(Dl["tipo"] == "scala_allineamento") & (Dl["passo"] == passo) & (Dl["ruolo"] == ruolo)]
    md = []
    for g in GTS:
        x = L[L["gt"] == g]
        sets = [sp for sp in SETS_MD if len(x[(x["insieme_righe"] == sp[0]) & (x["perturbazione"] == sp[1])])]
        if x.empty:
            continue
        md += [f"GT {g.upper()}:", "", "| dominio | passo | " + " | ".join(sname(s, p) for s, p in sets) + " |",
               "| --- | --- |" + " --- |" * len(sets)]
        for v in views:
            y = x[x["dominio"] == v]
            for (st, a, b) in dict.fromkeys(zip(y["confronto"], y["colonna_a"], y["colonna_b"])):
                cells = []
                for s, p in sets:
                    r = y[(y["confronto"] == st) & (y["colonna_a"] == a) & (y["colonna_b"] == b) &
                          (y["insieme_righe"] == s) & (y["perturbazione"] == p)]
                    cells.append(f3(r.iloc[0].delta, r.iloc[0].ic_lo, r.iloc[0].ic_hi, True) if len(r) else "-")
                who = st if not st.startswith(("4.", "5.")) else f"{st} [{label(a)}]"
                md.append(f"| {v} | {who} | " + " | ".join(cells) + " |")
        md.append("")
    return md


def count_table(Dl: pd.DataFrame) -> list:
    """Per passo (riga di confronto) e GT, in nocrop, crop, all_cross: famiglie sintetiche con IC appaiato che esclude
    lo 0 nel verso "piu' allineamento = peggio" / nel verso opposto / non risolte."""
    L = Dl[(Dl["tipo"] == "scala_allineamento") & (Dl["passo"] == "comune") & Dl["dominio"].isin(SYNTH)]
    sets = (("nocrop", "-"), ("crop", "crop"), ("all_cross", "-"))
    md = ["| GT | passo | ruolo | " + " | ".join(f"{s}: peggio / meglio / n.r." for s, _ in sets) + " |",
          "| --- | --- | --- |" + " --- |" * len(sets)]
    for g in GTS:
        x = L[L["gt"] == g]
        for st, a, b, ru, piu in dict.fromkeys(zip(x["confronto"], x["colonna_a"], x["colonna_b"], x["ruolo"],
                                                   x["piu_allineato"])):
            cells = []
            for s, p in sets:
                y = x[(x["confronto"] == st) & (x["colonna_a"] == a) & (x["colonna_b"] == b) &
                      (x["insieme_righe"] == s) & (x["perturbazione"] == p)]
                worse = int(((y.ic_lo > 0) if piu == "b" else (y.ic_hi < 0)).sum())
                better = int(((y.ic_hi < 0) if piu == "b" else (y.ic_lo > 0)).sum())
                cells.append(f"{worse} / {better} / {len(y) - worse - better} (su {len(y)})")
            who = st if not st.startswith(("4.", "5.")) else f"{st} [{label(a)}]"
            md.append(f"| {g.upper()} | {who} | {ru} | " + " | ".join(cells) + " |")
    return md + [""]


def size_table(ctrl: dict, views: tuple) -> list:
    cols = ("mm_rigid_icp_chamfer", "cs_rigid_icp_chamfer", "mm_nicp_p2tri", "cs_nicp_p2tri", "mm_nicp_template",
            "cs_nicp_template", "gnm_vb_fr", "gnm_vb_sr", "factorized_s1234|form", "factorized_s1234|shape")
    md = ["| dominio | " + " | ".join(label(c) for c in cols) + " |", "| --- |" + " --- |" * len(cols)]
    for v in views:
        t = ctrl["views"].get(v, {}).get("spearman_con_taglia", {}).get("nocrop", {})
        md.append(f"| {v} | " + " | ".join(f"{t[c]['oracle_size']:+.2f} / {t[c]['est_cs']:+.2f}" if c in t else "-"
                                           for c in cols) + " |")
    return md + [""]


def domain_tables(S: pd.DataFrame, Dl: pd.DataFrame, v: str) -> list:
    x = S[(S["dominio"] == v) & (S["passo"] == "comune")]
    if x.empty:
        return []
    vs = [(s, p) for s, p in SETS_MD if len(x[(x["insieme_righe"] == s) & (x["perturbazione"] == p)])]
    nr = {sp: int(x[(x["insieme_righe"] == sp[0]) & (x["perturbazione"] == sp[1])]["n_righe"].iloc[0]) for sp in vs}
    md = [f"### {DOMAIN[v]}: Spearman per insieme di righe (seme {int(x['seme'].iloc[0])})", ""]
    for g in GTS:
        md += [f"GT {g.upper()} (grassetto = lettura dichiarata; * = post hoc):", "",
               "| metodo | " + " | ".join(f"{sname(s, p)} ({nr[(s, p)]})" for s, p in vs) + " |",
               "| --- |" + " --- |" * len(vs)]
        for c in dict.fromkeys(x["colonna"]):
            y = x[(x["colonna"] == c) & (x["gt"] == g)]
            if y.empty:
                continue
            b0 = "**" if bool(y["dichiarata"].iloc[0]) else ""
            ph = "*" if bool(y["post_hoc"].iloc[0]) else ""
            cells = []
            for s, p in vs:
                r = y[(y["insieme_righe"] == s) & (y["perturbazione"] == p)]
                cells.append(f3(r.iloc[0].rho, r.iloc[0].ic_lo, r.iloc[0].ic_hi) if len(r) else "-")
            md.append(f"| {b0}{y['metodo'].iloc[0]}{ph}{b0} | " + " | ".join(cells) + " |")
        md.append("")
    for g, ref in (("fr", "factorized_s1234|form_cal"), ("fr", "factorized_s1234|form"), ("sr", "factorized_s1234|shape")):
        y = Dl[(Dl["dominio"] == v) & (Dl["gt"] == g) & (Dl["tipo"] == "vs_nostro") & (Dl["colonna_a"] == ref) &
               (Dl["passo"] == "comune")]
        if y.empty:
            continue
        md += [f"Delta appaiati, {label(ref)} - concorrente, GT {g.upper()} (tutti i riferimenti in `deltas.csv`):", "",
               "| concorrente | " + " | ".join(sname(s, p) for s, p in vs) + " |", "| --- |" + " --- |" * len(vs)]
        for c in dict.fromkeys(y["colonna_b"]):
            cells = []
            for s, p in vs:
                r = y[(y["colonna_b"] == c) & (y["insieme_righe"] == s) & (y["perturbazione"] == p)]
                cells.append(f3(r.iloc[0].delta, r.iloc[0].ic_lo, r.iloc[0].ic_hi, True) if len(r) else "-")
            md.append(f"| {label(c)} | " + " | ".join(cells) + " |")
        md.append("")
    return md


def nicp_section(S: pd.DataFrame, Dl: pd.DataFrame, ctrl: dict) -> list:
    md = ["## Fallimenti del NICP per coppia: selezione delle righe", "",
          "La maschera comune toglie le righe dove il NICP per coppia (mm o cs) fallisce (matrice singolare). La selezione "
          "dipende dalla topologia, non dalla GT: le righe perse sono soprattutto quelle con up60k, che quindi pesa meno "
          "negli insiemi nocrop e all_cross; e il NICP non e' penalizzato per i propri fallimenti. Righe perse per "
          "topologia (righe con quella topologia su un lato, senza la maschera del NICP):", "",
          "| dominio | righe perse | " + " | ".join(TOPOS) + " |", "| --- | --- |" + " --- |" * len(TOPOS)]
    for v in SYNTH:
        nf = ctrl["views"].get(v, {}).get("nicp_failures")
        if nf:
            md.append(f"| {v} | {nf['rows_lost']} | " + " | ".join(
                f"{b['lost']} / {b['rows_without_nicp_mask']} ({b['lost'] / max(b['rows_without_nicp_mask'], 1):.1%})"
                for b in (nf["by_topology"][t] for t in TOPOS)) + " |")
    md += ["", f"Sensibilita' (`passo` = \"{SENS}\" in entrambi i CSV): stesse colonne, maschera senza il NICP (le righe "
               "perse rientrano per tutti), NICP fallito = distanza massima della colonna sul dominio (il fallimento conta "
               "come la coppia piu' lontana). Il NICP con e senza la sostituzione e i passi della scala che lo usano:", "",
           "| dominio | GT | insieme | metodo / passo | maschera comune | fallimento = distanza massima |",
           "| --- | --- | --- | --- | --- | --- |"]
    for v in SYNTH:
        for g in GTS:
            for s, p in (("nocrop", "-"), ("crop", "crop")):
                for c in NICP:
                    a = [S[(S["dominio"] == v) & (S["gt"] == g) & (S["insieme_righe"] == s) & (S["perturbazione"] == p) &
                           (S["colonna"] == c) & (S["passo"] == ps)] for ps in ("comune", SENS)]
                    if len(a[1]):
                        md.append(f"| {v} | {g.upper()} | {s} | {label(c)} | " + " | ".join(
                            f"{f3(z.iloc[0].rho, z.iloc[0].ic_lo, z.iloc[0].ic_hi)} ({int(z.iloc[0].n_righe)})" if len(z)
                            else "-" for z in a) + " |")
                for st in ("2. ", "3b."):
                    a = [Dl[(Dl["dominio"] == v) & (Dl["gt"] == g) & (Dl["insieme_righe"] == s) & (Dl["perturbazione"] == p)
                            & (Dl["tipo"] == "scala_allineamento") & Dl["confronto"].str.startswith(st) &
                            (Dl["passo"] == ps)] for ps in ("comune", SENS)]
                    if len(a[1]):
                        md.append(f"| {v} | {g.upper()} | {s} | {a[1].iloc[0].confronto} | " + " | ".join(
                            f3(z.iloc[0].delta, z.iloc[0].ic_lo, z.iloc[0].ic_hi, True) if len(z) else "-" for z in a) + " |")
    return md + [""]


def write_results(S: pd.DataFrame, Dl: pd.DataFrame, NA: pd.DataFrame, ctrl: dict) -> None:
    md = ["# Matrice completa e confrontabile: Spearman con FR e SR, delta appaiati", "",
          "Codice `aau/full_matrix/` (`fm_eval.py`, definizioni nel docstring), numeri in questa cartella. Ogni metodo "
          "di un dominio e' valutato sulle STESSE righe (maschera comune: righe con tutte le colonne del dominio finite), "
          "con le STESSE 1000 repliche per soggetto (seme della T1 del dominio, lo stesso per tutti gli insiemi). IC 95% "
          "percentile. Delta appaiati = differenza replica per replica. FaMoS TEST sta in una sezione descrittiva a "
          "parte in fondo. Ava-256 non e' toccato.", "",
          "## File e colonne (nomi stabili)", "",
          "- `spearman.csv`: `metodo` (etichetta), `colonna` (id stabile), `famiglia`, `nostro`, `dichiarata` (lettura "
          "dichiarata per quella GT), `post_hoc` (d_F calibrata, varianti A e B dei fit e composizione, template FaMoS "
          "costruito qui), `dominio`, `gt` (fr, sr), `insieme_righe` (nocrop, crop, all_cross, con_perturbazione, "
          "coppia_topologie, stessa_topologia, coppia_ordinata, media_20_ordinate_senza_crop, media_5_stessa_topologia), "
          "`perturbazione` (`-`, una topologia, `a~b` coppia non ordinata, `a>b` ordinata, `tutte`; `con_perturbazione` "
          "P = P contro le altre topologie SENZA crop, la scomposizione di nocrop; per P = crop coincide con `crop`; "
          "`stessa_topologia` = soggetti diversi con la stessa topologia, crop - crop compresa), `rho`, `ic_lo`, `ic_hi`, "
          "`n_righe`, `n_soggetti`, `seme`, `passo` (`comune`; la sensibilita' del NICP; le righe proprie di un plug-in "
          "con buchi).",
          "- `deltas.csv`: le stesse chiavi + `tipo` (vs_nostro = braccio - concorrente; scala_allineamento = a - b; "
          "migliore_vs_migliore), `confronto`, `ruolo` (scala: principale o secondario), `piu_allineato` (scala: a o b), "
          "`metodo_a`, `colonna_a`, `post_hoc_a`, `metodo_b`, `colonna_b`, `post_hoc_b`, `rho_a`, `rho_b`, `delta`, "
          "`ic_lo`, `ic_hi`, `p_le0` (frazione delle repliche con delta <= 0).",
          "- `non_applicabili.csv`: `dominio`, `metodo`, `colonna`, `insieme_righe`, `motivo` (anche le CAUTELE su celle "
          "calcolate).",
          "- `controls.json`: controlli di riproduzione, maschera e righe tolte per colonna, fallimenti del NICP per "
          "topologia, Spearman di ogni colonna con la taglia oracolo e stimata (`spearman_con_taglia`), plug-in.", "",
          "Letture dichiarate: nostri FR = d_F cal. (post hoc, emendamento 4), d_F non cal. (preregistrata), ctrlfr; SR = "
          "d_P (ctrlfr non e' un braccio per SR: e' in `spearman.csv` ma non e' un riferimento dei delta SR). Fit 3DMM: "
          "coefficienti (FR e SR), mesh FR con FR, mesh SR con SR, composizione di B con FR (k calibrato solo per B: per "
          "preregistrato e vA la composizione non esiste). Geometrici, percettivi, spettrali, reti e riferimenti: "
          "entrambe le GT. Tutte le letture sono comunque in `spearman.csv` (`dichiarata` = False).", ""]
    md += ["## Come aggiungere un concorrente (plug-in)", "",
           "Una cartella `aau/runs/evidence/competitors_new/<metodo>/` (o `<metodo>/<tag>/`) con, per dominio (`hifi3d`, "
           "`facescape`, `faceverse`, `faceverse_neutral`, `famos`), `<dominio>.npz` (`subjects`, `topologies`, matrici "
           "`D` o `D_<lettura>` (n, n) sulle mesh elencate; FaMoS: `subjects` = view_id, `topologies` = `scan_gallery`) "
           "oppure `<dominio>.csv` / `pairs_<dominio>.csv` (`subject_a`, `topology_a`, `subject_b`, `topology_b`, una "
           "colonna per lettura), e il file di marcatura OBBLIGATORIO `fm_plugin.json` (anche `{}`; le cartelle senza, es. "
           "`pilot/` o `prep/`, si ignorano; facoltativi `label`, `family`, `readings`: {lettura: {`label`, "
           "`gt`}}; senza, una lettura `fr` / `*_fr` e' dichiarata per FR, `sr` / `*_sr` per SR, le altre per entrambe). "
           "Poi solo lo stadio finale: `FM_STEP=eval sbatch --cpus-per-task=64 --mem=192G --time=02:00:00 "
           "aau/full_matrix/fm.sbatch`. Un plug-in finito su tutta la maschera comune entra nello stesso passo e non cambia "
           "le righe degli altri; uno con buchi va in un passo a parte (`passo` = righe proprie), con i bracci di "
           "riferimento sulle sue righe.", ""]
    if len(NA):
        md += ["## Non applicabili e cautele", "", "| dominio | metodo | insieme | motivo |", "| --- | --- | --- | --- |"]
        md += [f"| {r.dominio} | {r.metodo} | {r.insieme_righe} | {r.motivo} |" for r in NA.itertuples()]
        md.append("")
    md += ["## Miglior nostro contro miglior concorrente (famiglie sintetiche; letture dichiarate, scelti sul punto: "
           "post hoc)", ""] + best_table(S, Dl, SYNTH)
    md += ["## Scala dell'allineamento", "",
           "Delta appaiati a - b (stesse righe, stesse repliche) lungo i gradi di liberta' dell'allineamento. PRINCIPALI: "
           "1 (togliere la taglia: la modalita' cs scala ogni mesh alla cs_ref del dominio, `blmm.work_coords`; non e' una "
           "scala per coppia), 2 (ICP cs - NICP cs), 3b (NICP su template cs - NICP per coppia cs), 4 e 5 con FR contro "
           "ICP mm, con SR contro ICP cs e NICP cs (con SR il riferimento corretto e' senza taglia). SECONDARI: 2b e 3 in "
           "mm, CONFUSI DALLA TAGLIA (in mm il NICP su template segue la taglia, il NICP per coppia la assorbe: vedi la "
           "tabella sotto), e 4 e 5 con SR contro ICP mm. 4: fit B con la mesh della GT (FR con FR, SR con SR); 5: nostro "
           "preregistrato (d_F non cal. con FR, d_P con SR), entrambi i semi.", "",
           "Spearman (punto, righe nocrop della maschera) di ogni colonna della scala con la taglia oracolo / la taglia "
           "stimata (`est_cs`); tutte le colonne e gli insiemi crop e all_cross in `controls.json`, "
           "`spearman_con_taglia`:", ""] + size_table(ctrl, VIEWS)
    md += ["### Confronti principali", ""] + ladder_table(Dl, SYNTH, "principale")
    md += ["### Confronti secondari (2b e 3 confusi dalla taglia; 4 e 5 con SR contro un riferimento con la taglia)", ""] + \
        ladder_table(Dl, SYNTH, "secondario")
    md += ["## Conteggio", "",
           "Per ogni passo e GT, in nocrop, crop e all_cross: numero di famiglie sintetiche (HIFI3D, FaceScape, FaceVerse "
           "con espressioni, FaceVerse neutra) con IC 95% appaiato che esclude lo 0 nel verso \"piu' allineamento = "
           "peggio\" / nel verso opposto / non risolte. Lato piu' allineato: b nei passi 1-3 (cs: taglia tolta per mesh; "
           "NICP: deformazione; NICP per coppia: deformazione specifica della coppia contro l'iscrizione su un template "
           "comune), a nel 4 (fit del modello per mesh contro la rigida), b nel 5 (ICP: allineamento esplicito; il braccio "
           "non allinea). Le righe 4 e 5 contano ogni modello / seme a parte.", ""] + count_table(Dl)
    md += nicp_section(S, Dl, ctrl)
    md += ["## Tabelle per dominio (famiglie sintetiche)", ""]
    for v in SYNTH:
        md += domain_tables(S, Dl, v)
    md += ["## FaMoS TEST (descrittivo)", "",
           "**Non usare per scegliere il setting di Ava-256** ne' per nessuna selezione: 15 persone, 105 righe (blocco "
           "scan gallery -> scan), IC larghissimi; nessuna crop, nessuna perturbazione. Cautele: ArcFace-render (ritaglio "
           "tarato su pochi render, vedi le cautele sopra), NICP su template con un template costruito qui dalle persone "
           "TRAIN (post hoc).", ""] + best_table(S, Dl, ("famos",))
    md += ["Scala dell'allineamento, FaMoS (tutti i passi, principali e secondari):", ""] + \
        ladder_table(Dl, ("famos",), "principale") + ladder_table(Dl, ("famos",), "secondario") + domain_tables(S, Dl, "famos")
    md += ["## Controlli", "", "```", json.dumps({k: v for k, v in ctrl.items()}, indent=1, default=str), "```", ""]
    (OUT_W / "results.md").write_text("\n".join(md) + "\n")


# ----------------------------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--views", default=",".join(VIEWS))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--plugins", default=str(PLUGIN_ROOT), help="cartelle radice dei plug-in, separate da virgola")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--out", type=Path, default=OUT, help="cartella delle uscite (prove: fuori dall'evidenza)")
    a = p.parse_args()
    global OUT_W
    OUT_W = a.out
    if a.summary_only:
        write_results(pd.read_csv(OUT_W / "spearman.csv"), pd.read_csv(OUT_W / "deltas.csv"),
                      pd.read_csv(OUT_W / "non_applicabili.csv"), json.loads((OUT_W / "controls.json").read_text()))
        return
    fp.VIEWS["faceverse_neutral"] = ("fvn", "mesh_pair_nocrop")
    fp.FORM_DIR["fvn"] = os.path.relpath(bpp.NEUTRAL_EMB, fp.EVAL)
    fp.BASELINES.update(bpe2.LABEL)
    _rows_for, cache = fp.rows_for, {}

    def rows_for(v):                                      # la T1 e la sua chiave righe calcolate una volta
        if v not in cache:
            cache[v] = _rows_for(v)
        return cache[v]

    fp.rows_for = rows_for
    sens = [c for c in bpe2.SENS if any((bp.out_dir(v, c.split("_vbs_")[0]) / "fit_e2_sens.npz").exists()
                                        for v in bpp.VIEW_ORDER)]
    pdirs, pignored = plugin_dirs([Path(x) for x in a.plugins.split(",") if x])
    S, Dl, NA, info = [], [], [], {}
    for view in a.views.split(","):
        s, d, n, info[view] = run_view(view, a.n_boot, a.workers, pdirs, sens)
        S += s
        Dl += d
        NA += n
    S, Dl, NA = pd.DataFrame(S), pd.DataFrame(Dl), pd.DataFrame(NA)
    OUT_W.mkdir(parents=True, exist_ok=True)
    S.to_csv(OUT_W / "spearman.csv", index=False)
    Dl.to_csv(OUT_W / "deltas.csv", index=False)
    NA.to_csv(OUT_W / "non_applicabili.csv", index=False)
    ctrl = {"views": info, "points_vs_sources": point_controls(S), "plugins": [str(d) for _, d in pdirs],
            "plugin_dirs_ignored_without_fm_plugin_json": pignored}
    (OUT_W / "controls.json").write_text(json.dumps(ctrl, indent=1, default=str) + "\n")
    print(json.dumps(ctrl["points_vs_sources"], indent=1), flush=True)
    write_results(S, Dl, NA, ctrl)


if __name__ == "__main__":
    main()
