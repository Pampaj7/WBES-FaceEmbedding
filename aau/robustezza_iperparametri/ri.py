#!/usr/bin/env python3
"""Robustezza agli iperparametri: i metodi con allineamento per coppia (ICP + Chamfer in mm, NICP per coppia in mm, NoW
metrical col rilevatore) su griglie di configurazioni ragionevoli, ciascuna con una fonte, contro la variabilita' da
seme del nostro (factorized s1234 / s2345, d_F non cal. e cal.). Regole in
``aau/runs/evidence/robustezza_iperparametri/REGOLE.md`` (committate prima dei numeri).

    aau/outlineB/run_o3d.sh      aau/robustezza_iperparametri/ri.py run --view hifi3d --method icp --workers 64
    aau/outlineB/run_o3d.sh      aau/robustezza_iperparametri/ri.py run --view hifi3d --method nicp --workers 64
    v3_work/unified_gt/run.sh    aau/robustezza_iperparametri/ri.py run --view hifi3d --method now --workers 64
    v3_work/unified_gt/run.sh    aau/robustezza_iperparametri/ri.py merge --view hifi3d --method now
    v3_work/unified_gt/run.sh    aau/robustezza_iperparametri/ri.py check-now-node --view hifi3d --workers 12  (a256-t4)
    v3_work/unified_gt/run.sh    aau/robustezza_iperparametri/ri.py check --view hifi3d          (nessuna GT)
    v3_work/unified_gt/run.sh    aau/robustezza_iperparametri/ri.py eval --workers 64           (LEGGE LE GT di test)
    (ri.sbatch; ``--limit N``: pilota dei tempi su N coppie / mesh A, uscite in ``<WORK>/pilot``, mai valutate)

Le implementazioni sono quelle della matrice, importate; qui cambiano solo i parametri (``GRID``):

- ICP (``blmm.pair_metrics`` passo ``fast``, modo ``mm``): campioni di 4096 punti coi semi della coppia,
  ``registration_icp`` punto-punto di Open3D, lettura = media delle distanze dal campione di X allineato al punto piu'
  vicino del campione di Y (``rigid_icp_chamfer``), x L_d in mm. Manopole: iterazioni (``ICPConvergenceCriteria``),
  soglia delle corrispondenze (mm / L_d nelle coordinate di lavoro), inizializzazione (centro = ``blmm.center_prealign``;
  identita' nel frame canonico = coordinate di lavoro + offset del centro; rigida sui 7 landmark NoW rilevati, Kabsch
  senza scala ne' riflessione), rigetto robusto (a ogni iterazione la soglia di Open3D = quantile 0,9 delle distanze
  dal piu' vicino, o 2,5 sigma con sigma = RMS delle distanze; una iterazione di Open3D per volta, arresto quando l'RMS
  di tutte le distanze cambia meno di 1e-6 unita' di lavoro o dopo 30 iterazioni).
- NICP per coppia (``blmm.pair_metrics`` passo ``nicp``): campioni e ICP rigido della matrice (``blmm.icp`` modo
  ``mm``), poi ``nicp``: il ciclo di ``facebench.nonrigid_icp_align`` (stessi aiutanti privati importati: Delaunay
  sul piano xy, incidenza, sistema ``_spsolve_system``, pesi = distanza < max(2 x media, 1)) con il programma della
  rigidezza come parametro: per stadio (alpha, soglia), arresto per stadio quando ||X - X_prec||_F <= soglia (``fro``)
  o quando ||X - X_prec||_F / sqrt(numero di elementi) < soglia (``rms``, menpo3d; ``dowhile``: controllo dopo
  l'iterazione, almeno una per stadio, come menpo3d; senza, controllo in testa come facebench e REALY, che con lo
  scarto dello stadio precedente puo' saltare uno stadio), tetto di iterazioni per stadio,
  termine dei landmark (righe ``beta D_L``, ``beta U_L`` come menpo3d), normalizzazione di menpo3d (centro della
  sorgente, scala 1 / diagonale del riquadro della sorgente; la deformata torna nelle coordinate di lavoro). Il
  programma di default riproduce ``nonrigid_icp_align`` (alpha = 50 e^(-0,5 d), soglia = 1 x 0,5^d x 6, d = 0, 1, 2).
  Lettura = media P2Tri (``fb.p2tri_distance`` sulla corrispondenza Chamfer), x L_d.
- NoW metrical (``pr_metrics``: ``procrustes``, ``now_icp``, ``scan_to_mesh``; ``pr_run.one_pair`` per la
  configurazione della matrice): landmark del rilevatore (``pr_noise`` condizione ``det`` sulle viste sintetiche; patch
  di ``esterna_famos_id`` su FaMoS), media dei due versi. Manopole: raggio del ritaglio (fattore di ``pm.crop`` sui 7
  landmark NoW, che restano quelli del ritaglio), iterazioni dell'ICP (``e_3`` del dogleg; 0 = nessun ICP, solo il
  Procrustes), sottoinsieme dei landmark del Procrustes (7 NoW, 5 di ArcFace/insightface: centri degli occhi = media
  di iBUG 36-41 e 42-47, punta del naso 30, angoli della bocca 48, 54; 51 interni iBUG 17-67; 68). I parametri del
  dogleg passano da un involucro di ``pm.dogleg`` (``now_icp`` lo chiama per nome): senza parametri e' l'originale.

Viste: HIFI3D e FaceScape dev, righe SENZA crop del sottocampione del rumore landmark (``pr_noise.subsample``: 8% delle
coppie non ordinate, seme 20261013 e nome della vista; restano quelle senza crop), X = mesh del soggetto di indice
minore e seme = indice della coppia di soggetti in ``np.triu_indices`` (la regola di ``blmm_pairs``); FaMoS: le 150
patch e le 11.175 coppie di ``esterna_famos_id`` (X = indice minore, seme ``blmm.mesh_seed`` dei nomi). Parti per
blocchi di coppie (ICP, NICP: ``c<k>.npz``) o per mesh A (NoW: ``A<k>.npz``) in ``<WORK>/parts``; una ripartenza salta
quelle salvate. ``merge``: ``<WORK>/dist/<vista>/<metodo>.npz`` (``i``, ``j``, ``D_<config>``) e
``<EVID>/controls/run_<vista>_<metodo>.json``. ``check``: la configurazione di default contro la matrice (sintetiche:
``mm_rigid_icp_chamfer``, ``mm_nicp_p2tri``, plug-in ``now_metrical/lmk_det``) o contro ``esterna_famos_id`` (ICP mm e
NoW metrical), tolleranze in ``TOL``; ``eval`` rifiuta di partire se un controllo non passa.

``eval``: sintetiche = Spearman con GT FR sulle righe del sottocampione senza crop con tutte le colonne finite (UNA
maschera per vista, per tutte le configurazioni e i metodi), 1000 repliche per soggetto col seme della T1
(``pr_eval.t1_seed``, ``bp_paired_e3.replicas``); FaMoS = rank-1 e mAP (``ei.retrieval``, query ricampionate per
soggetto con ``famos_eval.bootstrap_counts(15, 1000, 1234)``, distanza non finita = +inf). Misure (``measures``):
intervallo e deviazione standard (ddof 1) fra configurazioni con IC per replica; migliore - peggiore (scelte sul punto)
e default contro entrambe, delta appaiati; scarto fra i semi |s1234 - s2345| per lettura; intervallo del metodo -
scarto fra i semi (per replica); inversioni fra coppie di metodi su tutte le combinazioni di configurazioni (sul punto
e con IC del delta che esclude 0); classifiche dei quattro metodi su tutte le combinazioni. IC 95% percentile.
Uscite in ``aau/runs/evidence/robustezza_iperparametri/``: ``spearman.csv``, ``famos.csv``, ``spread.csv``,
``confronto_semi.csv``, ``inversioni.csv``, ``classifiche.csv``, ``controls.json``, ``results.md`` (generato; il
testo dopo ``MARK`` si conserva).
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
REPO = AAU.parent
PROT = AAU / "competitors" / "protocolli"
for _p in (PROT, AAU / "esterna_famos_id", AAU / "famos", AAU / "baselines_mm", AAU / "baselines_param"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pr_common as pc  # noqa: E402

blmm = pc.blmm
VIEWS = ("hifi3d", "facescape", "famos")
SYNTH = VIEWS[:2]
METHODS = ("icp", "nicp", "now")
WORK = AAU / "runs" / "robustezza_iperparametri"              # parti e distanze (npz fuori da git)
EVID = AAU / "runs" / "evidence" / "robustezza_iperparametri"
N_BOOT = 1000
CHUNK = 32                                                    # coppie per parte (ICP, NICP)
TOL = {"icp_mm": 1e-6, "nicp_mm": 1e-4, "now_mm": 1e-4}       # come esterna_famos_id; NICP: REGOLE.md, emendamento 2
MARK = "<!-- da qui in giu': scritto a mano, conservato da ri.py results -->"
LMK7 = list(pc.LMK7)
_ST: dict = {}

# ------------------------------------------------------------------------------------------ griglie
# metodo -> configurazione -> (parametri, manopola, fonte). "default" = la configurazione della matrice.
_A0 = [50.0 * np.exp(-0.5 * d) for d in range(3)]                       # facebench nonrigid_icp_align
MENPO_ALPHA = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2]
GRID = {
    "icp": {
        "default": ({}, "-", "matrice: blmm.icp modo mm (centro, registration_icp punto-punto, soglia 1000 = infinita, "
                             "criteri di default di Open3D: 30 iterazioni, 1e-6, 1e-6)"),
        "it10": ({"max_iter": 10}, "iterazioni", "PCL, Registration: max_iterations_ = 10 (default)"),
        "it2000": ({"max_iter": 2000}, "iterazioni",
                   "tutorial ICP di Open3D 0.18: ICPConvergenceCriteria(max_iteration=2000)"),
        "thr20": ({"thr_mm": 20.0}, "soglia", "tutorial ICP di Open3D 0.18: threshold = 0.02 (m)"),
        "thr10": ({"thr_mm": 10.0}, "soglia",
                  "tutorial colored ICP di Open3D 0.18: voxel_radius = [0.04, 0.02, 0.01] (m), l'ultimo"),
        "init_id": ({"init": "identity"}, "inizializzazione",
                    "Open3D registration_icp: init = identita' (default), frame canonico senza centrare"),
        "init_lmk": ({"init": "landmark"}, "inizializzazione",
                     "rigida sui 7 landmark NoW rilevati (Procrustes senza scala, come l'avvio di NoW metrical; "
                     "facebench icp_align prealign='landmark')"),
        "rej_pulli": ({"reject": "q90"}, "rigetto",
                      "Pulli 1999 in Rusinkiewicz e Levoy 2001: scarta il 10% delle coppie peggiori"),
        "rej_masuda": ({"reject": "sigma2.5"}, "rigetto",
                       "Masuda 1996 in Rusinkiewicz e Levoy 2001: scarta oltre 2,5 sigma"),
    },
    "nicp": {
        "default": ({"alpha": _A0, "eps": [1 * (0.5 ** d) * 6 for d in range(3)], "crit": "fro"}, "-",
                    "matrice: facebench nonrigid_icp_align (alpha 50 e^(-0,5 d), soglia 6 x 0,5^d, 3 stadi, gamma 1)"),
        "eps_realy": ({"alpha": _A0, "eps": [1.0 * 0.5 ** d for d in range(3)], "crit": "fro"}, "iterazioni",
                      "REALY utils/NICP.py nICP_without_keypoints con eval.py epsilon = 1.0 (soglia 0,5^d, senza x 6)"),
        "cap10": ({"alpha": _A0, "eps": [1 * (0.5 ** d) * 6 for d in range(3)], "crit": "fro", "cap": 10},
                  "iterazioni", "trimesh 5.1 nricp_amberg: max_iter = 10 per passo (default)"),
        "menpo3d": ({"alpha": MENPO_ALPHA, "eps": [1e-3] * 8, "crit": "rms", "norm": "menpo", "dowhile": True},
                    "rigidezza",
                    "menpo3d non_rigid_icp: stiffness_weights = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2], eps = 1e-3 "
                    "(default; 'empirically found to perform well for well rigidly aligned facial meshes')"),
        "menpo3d_lmk": ({"alpha": MENPO_ALPHA, "eps": [1e-3] * 8, "crit": "rms", "norm": "menpo", "dowhile": True,
                         "beta": [5, 2, 0.5, 0, 0, 0, 0, 0]}, "landmark",
                        "menpo3d non_rigid_icp con landmark: landmark_weights = [5, 2, 0.5, 0, 0, 0, 0, 0] (default), "
                        "7 landmark NoW rilevati"),
    },
    "now": {
        "default": ({}, "-", "matrice: plug-in now_metrical/lmk_det (7 landmark NoW, raggio 1,4, dogleg delta_0 0,1, "
                             "e_3 1e-4, maxiter 200)"),
        "r12": ({"radius": 1.2}, "raggio", "Stirling/ESRC, fg2018-competition compute_vertices_to_mesh_distances.py:46 "
                                           "(raggio 1,2)"),
        "icp0": ({"icp": False}, "iterazioni", "Stirling/ESRC (fg2018): solo il Procrustes, nessun ICP"),
        "e3_0": ({"dogleg": {"e_3": 0.0}}, "iterazioni",
                 "chumpy minimize_dogleg: e_3 = 0 (default; fino a maxiter = 200 o e_1 / e_2)"),
        "lmk5": ({"lmk": "5"}, "landmark", "5 punti di ArcFace / insightface (centri degli occhi, punta del naso, angoli "
                                           "della bocca), dai 68 rilevati"),
        "lmk51": ({"lmk": "51"}, "landmark", "51 interni iBUG (300-W, senza il contorno: 17-67)"),
        "lmk68": ({"lmk": "68"}, "landmark", "68 iBUG (Sagonas et al.)"),
    },
}
LABEL = {"icp": "ICP + Chamfer in mm", "nicp": "NICP per coppia in mm", "now": "NoW metrical (rilevatore)"}
SEEDS = ("s1234", "s2345")
READ = {"form": "d_F non cal.", "form_cal": "d_F cal."}
OURS = {f"nostro_{r}": (READ[r], [(s, f"factorized_{s}|{r}") for s in SEEDS]) for r in READ}


def col(m: str, c: str) -> str:
    return f"{m}@{c}"


# ------------------------------------------------------------------------------------------ dati

def pair_seeds(subjects: list, items: list, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Seme della coppia = indice di (soggetto di i, soggetto di j) in ``np.triu_indices`` (``blmm_pairs.topo_view``)."""
    n = len(subjects)
    s2k = {s: k for k, s in enumerate(subjects)}
    Q = -np.ones((n, n), np.int64)
    iu, ju = np.triu_indices(n, 1)
    Q[iu, ju] = np.arange(len(iu))
    si = np.asarray([s2k[items[a][0]] for a in i])
    sj = np.asarray([s2k[items[b][0]] for b in j])
    if (si >= sj).any():
        raise SystemExit("coppie con il soggetto di X non minore di quello di Y")
    return Q[si, sj]


def view_data(vname: str) -> dict:
    """Elementi (soggetto, topologia, percorso), coppie (i, j) con i < j, semi, 68 landmark rilevati in mm, nome della
    vista per ``blmm`` e vista di ``pr_common``."""
    if vname == "famos":
        import ei
        it = ei.items("eval")
        i, j = ei.pairs(len(it))
        names = [p.stem for _, _, p in it]
        seeds = np.asarray([blmm.mesh_seed(names[a], names[b]) for a, b in zip(i, j)])
        with np.load(ei.DATA / "eval" / "lmk.npz") as z:
            if list(z["names"].astype(str)) != names:
                raise SystemExit("famos: landmark con patch diverse")
            L68 = np.asarray(z["L68"], np.float64)
        return {"items": it, "i": i, "j": j, "seed": seeds, "L68": L68, "bview": ei.inject_view("eval"),
                "pview": pc.view(ei.VNAME["eval"], ei.spec_path("eval"))}
    import pr_noise as prn
    pv = pc.view(vname)
    it = pv["items"]
    i, j = prn.subsample(vname, it)
    topo = np.asarray([t for _, t, _ in it])
    keep = (topo[i] != "crop") & (topo[j] != "crop")
    i, j = i[keep], j[keep]
    with np.load(prn.prep_path(vname, "det"), allow_pickle=True) as z:
        if list(zip(z["subjects"].astype(str), z["topologies"].astype(str))) != [(s, t) for s, t, _ in it]:
            raise SystemExit(f"{vname}: landmark del rilevatore con mesh diverse dalla vista")
        L68 = np.asarray(z["L68"], np.float64)
    return {"items": it, "i": i, "j": j, "seed": pair_seeds(blmm.subjects(vname), it, i, j), "L68": L68,
            "bview": vname, "pview": pv}


def lmk_work(L: np.ndarray, off: np.ndarray, unit: float) -> np.ndarray:
    """Landmark in mm -> coordinate di lavoro del modo mm: (L - centro) / L_d = L / L_d - offset."""
    return L / unit - off


# ------------------------------------------------------------------------------------------ ICP

def kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) rigida ai minimi quadrati con B ~ A @ R.T + t, senza riflessione."""
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ D @ Vt).T
    return R, mb - ma @ R.T


def _pcd(P: np.ndarray):
    import open3d as o3d
    c = o3d.geometry.PointCloud()
    c.points = o3d.utility.Vector3dVector(P)
    return c


def rigid(Xs: np.ndarray, Ys: np.ndarray, p: dict, unit: float, off: tuple, lmk: tuple) -> tuple[np.ndarray, np.ndarray, int]:
    """X allineato, Y nello stesso frame, iterazioni (ICP rigido della configurazione ``p``; ``{}`` = ``blmm.icp``)."""
    import open3d as o3d
    reg = o3d.pipelines.registration
    init = p.get("init", "centroid")
    if init == "centroid":
        src, tgt = blmm.center_prealign(Xs, Ys), Ys
    elif init == "identity":
        src, tgt = Xs + off[0], Ys + off[1]
    else:
        R, t = kabsch(*lmk)
        src, tgt = Xs @ R.T + t, Ys
    thr = p["thr_mm"] / unit if "thr_mm" in p else 1000.0
    s, g = _pcd(src), _pcd(tgt)
    est = reg.TransformationEstimationPointToPoint()
    if "reject" not in p:
        kw = {"criteria": reg.ICPConvergenceCriteria(max_iteration=p["max_iter"])} if "max_iter" in p else {}
        res = reg.registration_icp(s, g, thr, estimation_method=est, **kw)
        s.transform(res.transformation)
        return np.asarray(s.points), tgt, -1
    from scipy.spatial import cKDTree
    tree = cKDTree(tgt)
    prev, it = None, 0
    for it in range(1, 31):
        d = tree.query(np.asarray(s.points))[0]
        thr = float(np.quantile(d, 0.9)) if p["reject"] == "q90" else 2.5 * float(np.sqrt((d ** 2).mean()))
        res = reg.registration_icp(s, g, thr, estimation_method=est, criteria=reg.ICPConvergenceCriteria(max_iteration=1))
        s.transform(res.transformation)
        rmse = float(np.sqrt((tree.query(np.asarray(s.points))[0] ** 2).mean()))
        if prev is not None and abs(prev - rmse) < 1e-6:
            break
        prev = rmse
    return np.asarray(s.points), tgt, it


def chamfer_read(Xa: np.ndarray, Y: np.ndarray) -> float:
    import facebench as fb
    return float(np.mean(fb.p2p_distance(Xa, Y, fb.chamfer_correspondence(Xa, Y))))


# ------------------------------------------------------------------------------------------ NICP

def nicp(source_points: np.ndarray, target_points: np.ndarray, p: dict, lmk=None) -> tuple[np.ndarray, list]:
    """Il ciclo di ``facebench.nonrigid_icp_align`` con il programma ``p`` (alpha, eps, crit, cap, beta, norm) e, con
    ``beta``, il termine dei landmark ``lmk`` = (indici nei punti della sorgente, posizioni sul bersaglio). Ritorna la
    sorgente deformata e le iterazioni per stadio."""
    import scipy
    from scipy.sparse import csc_matrix
    from scipy.spatial import Delaunay
    from facebench.nonrigid_aligners import nonrigid_icp as fnr
    c, s = np.zeros(3), 1.0
    if p.get("norm") == "menpo":                         # menpo3d: centro della sorgente, diagonale del riquadro = 1
        c = source_points.mean(0)
        s = 1.0 / float(np.sqrt((np.ptp(source_points, axis=0) ** 2).sum()))
    src_pts, tgt_pts = (source_points - c) * s, (target_points - c) * s
    source_tri = Delaunay(src_pts[:, :2]).simplices.T
    source, target = src_pts.T, tgt_pts.T
    G = np.diag([1.0, 1.0, 1.0, p.get("gamma", 1.0)]).astype(np.float32)
    M = fnr._triangles_to_edge_vertex_adjacent_matrix(source_tri)
    A1 = scipy.sparse.kron(M, G)
    B1 = csc_matrix((A1.shape[0], 3), dtype=np.float32)
    cur_src = source
    cur_X = np.zeros((source.shape[1] * 4, 3))
    X = np.ones_like(cur_X)
    beta = p.get("beta")
    if beta is not None:
        kidx, U_L = lmk[0], (lmk[1] - c) * s
    iters = []

    def delta(X, cur_X):
        d = np.linalg.norm(X - cur_X)
        return d / np.sqrt(X.size) if p["crit"] == "rms" else d

    dw = p.get("dowhile", False)
    for st, (cur_alpha, eps) in enumerate(zip(p["alpha"], p["eps"])):
        k = 0
        while True:
            if not dw and delta(X, cur_X) <= eps:      # facebench, REALY: controllo in testa (scarto dello stadio prima)
                break
            if p.get("cap") and k >= p["cap"]:
                break
            cur_X = X.copy()
            nn_indices, nn_distances = fnr._find_nearest_neighbors(cur_src, target)
            cur_nn_dst = target[:, nn_indices]
            D = fnr._sparse_matrix_from_vertices(cur_src)
            weights = (nn_distances < max(np.mean(nn_distances) * 2, 1)).astype(np.float32)
            A2 = D.multiply(weights[:, np.newaxis])
            B2 = csc_matrix(np.multiply(cur_nn_dst.T, weights[:, np.newaxis]))
            As, Bs = [A1.multiply(cur_alpha), A2], [B1, B2]
            if beta is not None and beta[st] > 0:
                As.append(D[kidx] * beta[st])
                Bs.append(csc_matrix(U_L * beta[st]))
            A = scipy.sparse.vstack(As)
            B = scipy.sparse.vstack(Bs)
            X = fnr._spsolve_system(A, B)
            cur_src = (D @ X).T
            k += 1
            if dw and delta(X, cur_X) < eps:           # menpo3d: almeno una iterazione per stadio, controllo dopo
                break
        iters.append(k)
    return cur_src.T / s + c, iters


# ------------------------------------------------------------------------------------------ NoW

SUBSETS = {"7": LMK7, "51": list(range(17, 68)), "68": list(range(68))}


def lmk_subset(L68: np.ndarray, name: str) -> np.ndarray:
    if name == "5":                                      # ArcFace / insightface: occhi, naso, angoli della bocca
        return np.stack([L68[36:42].mean(0), L68[42:48].mean(0), L68[30], L68[48], L68[54]])
    return L68[SUBSETS[name]]


def _dogleg(fun, jac, x0, **kw):
    """``pm.dogleg`` con i parametri della configurazione corrente (``_ST['dogleg']``); vuoti = l'originale."""
    kw.update(_ST.get("dogleg") or {})
    return _ST["dogleg_orig"](fun, jac, x0, **kw)


def now_cfg(surf, LA68: np.ndarray, LB68: np.ndarray, B_crop: np.ndarray, p: dict) -> tuple[float, int]:
    """e(A, B) di NoW metrical (``pm.now_pair`` con ``metrical=True``) nella configurazione ``p``."""
    import pr_metrics as pm
    sub = p.get("lmk", "7")
    tf = pm.procrustes(lmk_subset(LB68, sub), lmk_subset(LA68, sub), scaling=True)
    T, c = tf["rotation"], tf["translation"]
    if p.get("icp", True):
        _ST["dogleg"] = p.get("dogleg")
        (sig, Q, tau), info = pm.now_icp(surf, B_crop, 1.0, T, c, with_scale=False)
        _ST["dogleg"] = None
        it = int(info["iterations"])
    else:
        sig, Q, tau, it = 1.0, T, c, 0
    return float(pm.scan_to_mesh(surf, B_crop, sig, Q, tau).mean()), it


# ------------------------------------------------------------------------------------------ run

def _pair_task(k: int) -> int:
    """Una parte di coppie (ICP o NICP): tutte le configurazioni del metodo."""
    import run_facebench_remesh as rfr
    out_p = _ST["ckdir"] / f"c{k}.npz"
    if out_p.exists():
        return k
    t0 = time.perf_counter()
    V, m, cfgs = _ST["V"], _ST["method"], GRID[_ST["method"]]
    pos = _ST["todo_pos"][k]
    vals = {c: np.full(len(pos), np.nan) for c in cfgs}
    secs = {c: np.zeros(len(pos)) for c in cfgs}
    its = {c: np.full((len(pos), 8), -1, np.int64) for c in cfgs}
    errs = {c: [""] * len(pos) for c in cfgs}
    for q, r in enumerate(pos):
        a, b, seed = int(V["i"][r]), int(V["j"][r]), int(V["seed"][r])
        (X, ox), (Y, oy) = V["coords"][a], V["coords"][b]
        Xs, Ys = rfr.sample_pts(X, blmm.N_POINTS, seed), rfr.sample_pts(Y, blmm.N_POINTS, seed + 1)
        lx, ly = (lmk_work(V["L68"][x][LMK7], o, V["unit"]) for x, o in ((a, ox), (b, oy)))
        if m == "nicp":
            t0r = time.perf_counter()
            try:
                X_rig = blmm.icp(Xs, Ys, "mm")
                # landmark di X = il punto del campione piu' vicino (come REALY), posizioni = landmark di Y
                lmk = (np.argmin(((Xs[:, None] - lx[None]) ** 2).sum(-1), axis=0), ly)
            except Exception as exc:  # noqa: BLE001  (l'ICP della matrice fallito: NaN per tutte)
                X_rig, lmk = None, None
                for c in cfgs:
                    errs[c][q] = f"{type(exc).__name__}: {exc}"
            t_rig = time.perf_counter() - t0r
        for c, (p, _, _) in cfgs.items():
            t1 = time.perf_counter()
            try:
                if m == "icp":
                    Xa, Yt, it = rigid(Xs, Ys, p, V["unit"], (ox, oy), (lx, ly))
                    vals[c][q] = chamfer_read(Xa, Yt) * V["unit"]
                    its[c][q, 0] = it
                elif X_rig is not None:
                    import facebench as fb
                    Xn, it = nicp(X_rig, Ys, p, lmk)
                    corr = fb.chamfer_correspondence(Xn, Ys)
                    vals[c][q] = float(np.mean(fb.p2tri_distance(Xn, Ys, corr))) * V["unit"]
                    its[c][q, :len(it)] = it
            except Exception as exc:  # noqa: BLE001  (come blmm.pair_metrics: la coppia resta NaN)
                vals[c][q] = np.nan
                errs[c][q] = f"{type(exc).__name__}: {exc}"
            secs[c][q] = time.perf_counter() - t1 + (t_rig if m == "nicp" else 0.0)
    blmm.atomic_savez(out_p, pos=np.asarray(pos), task_s=time.perf_counter() - t0, job=pc.stamp()["job"],
                      **{f"val:{c}": vals[c] for c in cfgs}, **{f"sec:{c}": secs[c] for c in cfgs},
                      **{f"it:{c}": its[c] for c in cfgs}, **{f"err:{c}": np.asarray(errs[c]) for c in cfgs})
    return k


def _now_task(a: int) -> int:
    """Una mesh A di NoW: e(A, B) per ogni B partner, tutte le configurazioni."""
    import pr_metrics as pm
    out_p = _ST["ckdir"] / f"A{a}.npz"
    if out_p.exists():
        return a
    t0 = time.perf_counter()
    V, cfgs = _ST["V"], GRID["now"]
    bs = _ST["partners"][a]
    Va, Fa = V["mesh"][a]
    surf = pm.Surf(Va, Fa)
    vals = {c: np.full(len(bs), np.nan) for c in cfgs}
    secs = {c: np.zeros(len(bs)) for c in cfgs}
    its = {c: np.full(len(bs), -1, np.int64) for c in cfgs}
    errs = {c: [""] * len(bs) for c in cfgs}
    for q, b in enumerate(bs):
        for c, (p, _, _) in cfgs.items():
            t1 = time.perf_counter()
            try:
                vals[c][q], its[c][q] = now_cfg(surf, V["L68"][a], V["L68"][b], V["crop"][p.get("radius", 1.4)][b], p)
            except Exception as exc:  # noqa: BLE001  (pr_run.one_pair: la coppia resta NaN)
                errs[c][q] = f"{type(exc).__name__}: {exc}"
            secs[c][q] = time.perf_counter() - t1
    blmm.atomic_savez(out_p, a=a, b=np.asarray(bs), task_s=time.perf_counter() - t0, job=pc.stamp()["job"],
                      **{f"val:{c}": vals[c] for c in cfgs}, **{f"sec:{c}": secs[c] for c in cfgs},
                      **{f"it:{c}": its[c] for c in cfgs}, **{f"err:{c}": np.asarray(errs[c]) for c in cfgs})
    return a


def ckdir_of(vname: str, method: str, limit: int) -> Path:
    return WORK / ("pilot" if limit else "parts") / vname / method


def partners_of(i: np.ndarray, j: np.ndarray, n: int) -> list:
    out = [[] for _ in range(n)]
    for a, b in zip(i, j):
        out[a].append(b)
        out[b].append(a)
    return [np.asarray(sorted(x), np.int64) for x in out]


def run(vname: str, method: str, workers: int, limit: int) -> None:
    t0 = time.time()
    V = view_data(vname)
    ck = ckdir_of(vname, method, limit)
    ck.mkdir(parents=True, exist_ok=True)
    _ST.update(V=V, method=method, ckdir=ck)
    if method == "now":
        import pr_metrics as pm
        import pr_run as prr
        _ST["dogleg_orig"] = pm.dogleg
        pm.dogleg = _dogleg                               # now_icp chiama dogleg per nome nel modulo
        V["mesh"] = [pc.mm_mesh(V["pview"], p)[:2] for _, _, p in V["items"]]
        radii = sorted({p.get("radius", prr.NOW_R) for p, _, _ in GRID["now"].values()})
        V["crop"] = {r: [pm.crop(Vm, V["L68"][k][LMK7], r) for k, (Vm, _) in enumerate(V["mesh"])] for r in radii}
        P = partners_of(V["i"], V["j"], len(V["items"]))
        _ST["partners"] = P
        todo = [a for a in range(len(P)) if len(P[a])]
        if limit:
            todo = todo[:limit]
        fn = _now_task
    else:
        prm = blmm.params()
        V["unit"] = blmm.unit(V["bview"], "mm", prm)
        V["coords"] = [blmm.work_coords(V["bview"], "mm", p, prm) for _, _, p in V["items"]]
        npairs = min(limit, len(V["i"])) if limit else len(V["i"])
        _ST["todo_pos"] = [list(range(s, min(s + CHUNK, npairs))) for s in range(0, npairs, CHUNK)]
        todo = list(range(len(_ST["todo_pos"])))
        fn = _pair_task
    print(f"[ri] run {vname} {method}: {len(V['i'])} coppie, {len(todo)} parti, configurazioni {list(GRID[method])}, "
          f"{time.time() - t0:.0f}s di caricamento", flush=True)
    tl = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(fn, todo, chunksize=1)):
            if (k + 1) % max(1, len(todo) // 10) == 0:
                print(f"[ri] run {vname} {method}: {k + 1}/{len(todo)} parti, {time.time() - tl:.0f}s", flush=True)
    if limit:
        pilot_report(vname, method, ck)
    print(f"[ri] run {vname} {method}: fatto in {time.time() - t0:.0f}s", flush=True)


def pilot_report(vname: str, method: str, ck: Path) -> None:
    """Tempi e iterazioni del pilota (nessuna GT, nessuna distanza riportata)."""
    rows = {c: {"sec": [], "it": [], "nan": 0} for c in GRID[method]}
    for f in sorted(ck.glob("*.npz")):
        with np.load(f) as z:
            for c in GRID[method]:
                rows[c]["sec"] += list(z[f"sec:{c}"])
                rows[c]["it"] += [x.tolist() if np.ndim(x) else int(x) for x in z[f"it:{c}"]]
                rows[c]["nan"] += int(np.isnan(z[f"val:{c}"]).sum())
    out = {c: {"n": len(r["sec"]), "s_median": float(np.median(r["sec"])), "s_mean": float(np.mean(r["sec"])),
               "nan": r["nan"], "it_median": np.median(np.asarray(r["it"]), axis=0).tolist()} for c, r in rows.items()}
    print(f"[ri] pilota {vname} {method}: {json.dumps(out)}", flush=True)


# ------------------------------------------------------------------------------------------ merge

def dist_path(vname: str, method: str) -> Path:
    return WORK / "dist" / vname / f"{method}.npz"


def merge(vname: str, method: str) -> None:
    V = view_data(vname)
    i, j = V["i"], V["j"]
    ck = ckdir_of(vname, method, 0)
    cfgs = list(GRID[method])
    D = {c: np.full(len(i), np.nan) for c in cfgs}
    S = {c: np.full(len(i), np.nan) for c in cfgs}
    IT = {c: [] for c in cfgs}
    errs = {c: [] for c in cfgs}
    core_s, jobs = 0.0, set()
    if method == "now":
        P = partners_of(i, j, len(V["items"]))
        E = {c: {} for c in cfgs}
        Es = {c: {} for c in cfgs}
        need = [a for a in range(len(P)) if len(P[a])]
        miss = [a for a in need if not (ck / f"A{a}.npz").exists()]
        if miss:
            raise SystemExit(f"{vname} now: {len(miss)} mesh A mancanti (es. {miss[:5]})")
        for a in need:
            with np.load(ck / f"A{a}.npz") as z:
                if not np.array_equal(np.sort(z["b"]), P[a]):
                    raise SystemExit(f"{vname} now A{a}: partner diversi dal sottocampione")
                for c in cfgs:
                    for b, x, s in zip(z["b"], z[f"val:{c}"], z[f"sec:{c}"]):
                        E[c][(a, int(b))], Es[c][(a, int(b))] = float(x), float(s)
                    IT[c] += list(z[f"it:{c}"])
                    e = z[f"err:{c}"]
                    errs[c] += [str(x) for x in e[e != ""]]
                core_s += float(z["task_s"])
                jobs.add(str(z["job"]))
        for c in cfgs:
            D[c] = np.asarray([(E[c][(a, b)] + E[c][(b, a)]) / 2 for a, b in zip(i, j)])
            S[c] = np.asarray([Es[c][(a, b)] + Es[c][(b, a)] for a, b in zip(i, j)])
    else:
        nch = (len(i) + CHUNK - 1) // CHUNK
        miss = [k for k in range(nch) if not (ck / f"c{k}.npz").exists()]
        if miss:
            raise SystemExit(f"{vname} {method}: {len(miss)} parti mancanti (es. {miss[:5]})")
        for k in range(nch):
            with np.load(ck / f"c{k}.npz") as z:
                pos = z["pos"]
                if not np.array_equal(pos, np.arange(k * CHUNK, min((k + 1) * CHUNK, len(i)))):
                    raise SystemExit(f"{vname} {method} c{k}: coppie diverse dalla parte")
                for c in cfgs:
                    D[c][pos], S[c][pos] = z[f"val:{c}"], z[f"sec:{c}"]
                    IT[c] += [x for x in z[f"it:{c}"]]
                    e = z[f"err:{c}"]
                    errs[c] += [str(x) for x in e[e != ""]]
                core_s += float(z["task_s"])
                jobs.add(str(z["job"]))
    blmm.atomic_savez(dist_path(vname, method), i=i, j=j, names=np.asarray([f"{s}|{t}" for s, t, _ in V["items"]]),
                      **{f"D_{c}": D[c] for c in cfgs})
    info = {"view": vname, "method": method, "pairs": int(len(i)), "core_hours": core_s / 3600, "jobs": sorted(jobs),
            "configs": {}}
    for c in cfgs:
        it = np.asarray(IT[c])
        if it.ndim == 2:
            it = it[:, (it >= 0).any(0)]
        info["configs"][c] = {"nan": int(np.isnan(D[c]).sum()), "errors": len(errs[c]),
                              "error_examples": sorted(set(errs[c]))[:5], "median_mm": float(np.nanmedian(D[c])),
                              "seconds_per_pair_median": float(np.nanmedian(S[c])),
                              "iterations_median": np.median(it, axis=0).tolist() if it.size else None,
                              "iterations_max": np.max(it, axis=0).tolist() if it.size else None}
    info.update(pc.stamp())
    pc.atomic_json(EVID / "controls" / f"run_{vname}_{method}.json", info)
    print(f"[ri] merge {vname} {method}: {len(i)} coppie, NaN per configurazione "
          f"{ {c: x['nan'] for c, x in info['configs'].items()} }, {info['core_hours']:.1f} ore-core", flush=True)


def load_dist(vname: str, method: str, V: dict) -> dict:
    with np.load(dist_path(vname, method)) as z:
        if not (np.array_equal(z["i"], V["i"]) and np.array_equal(z["j"], V["j"])):
            raise SystemExit(f"{vname} {method}: coppie diverse dalla vista")
        return {c: np.asarray(z[f"D_{c}"], np.float64) for c in GRID[method]}


# ------------------------------------------------------------------------------------------ check

def maxdiff(x, y) -> dict:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    d = np.abs(x[ok] - y[ok])
    return {"n": int(ok.sum()), "nan_new": int((~np.isfinite(x)).sum()), "nan_ref": int((~np.isfinite(y)).sum()),
            "nan_same_pairs": bool(np.array_equal(~np.isfinite(x), ~np.isfinite(y))),
            "max_abs": float(d.max()) if len(d) else None, "median_abs": float(np.median(d)) if len(d) else None,
            "equal": int((d == 0).sum())}


def _now_default_task(a: int) -> tuple:
    import pr_metrics as pm
    V = _ST["V"]
    Va, Fa = V["mesh"][a]
    surf = pm.Surf(Va, Fa)
    return a, [now_cfg(surf, V["L68"][a], V["L68"][b], V["crop"][b], {})[0] for b in _ST["partners"][a]]


def now_matrix_ref(vname: str, V: dict) -> np.ndarray:
    """Plug-in ``now_metrical/lmk_det`` della matrice sulle coppie della vista."""
    import bp_paired as bpp
    import pr_eval as pe
    idx = bpp.be.zes.Index(blmm.subjects(vname))
    pi = np.asarray([idx.pos[(s, t)] for s, t, _ in V["items"]])
    return pe.plugin_matrix("now_metrical/lmk_det", vname, list(idx.keys))[pi[V["i"]], pi[V["j"]]]


def check_now_node(vname: str, workers: int) -> None:
    """NoW metrical di default ricalcolato su QUESTO nodo per tutte le coppie della vista (media dei due versi) contro il
    plug-in della matrice: il controllo del codice sul tipo di nodo della matrice (a256-t4; REGOLE.md, emendamento 3).
    I valori non entrano nella valutazione. Nessuna GT."""
    import pr_metrics as pm
    import pr_run as prr
    t0 = time.time()
    V = view_data(vname)
    _ST["dogleg_orig"] = pm.dogleg
    pm.dogleg = _dogleg
    V["mesh"] = [pc.mm_mesh(V["pview"], p)[:2] for _, _, p in V["items"]]
    V["crop"] = [pm.crop(Vm, V["L68"][k][LMK7], prr.NOW_R) for k, (Vm, _) in enumerate(V["mesh"])]
    P = partners_of(V["i"], V["j"], len(V["items"]))
    _ST.update(V=V, partners=P)
    E = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for a, vals in pool.imap_unordered(_now_default_task, [a for a in range(len(P)) if len(P[a])]):
            E.update({(a, int(b)): x for b, x in zip(P[a], vals)})
    here = np.asarray([(E[(a, b)] + E[(b, a)]) / 2 for a, b in zip(V["i"], V["j"])])
    ref = now_matrix_ref(vname, V)
    grid = load_dist(vname, "now", V)["default"]
    out = {"view": vname, "here_vs_matrix": maxdiff(here, ref), "grid_vs_matrix": maxdiff(grid, ref),
           "grid_vs_matrix_over_tol": int((np.abs(grid - ref) > TOL["now_mm"]).sum()), "tol": TOL["now_mm"],
           "grid_host": json.loads((EVID / "controls" / f"run_{vname}_now.json").read_text())["host"],
           "seconds": time.time() - t0, **pc.stamp()}
    out["ok"] = bool(out["here_vs_matrix"]["max_abs"] <= TOL["now_mm"] and out["here_vs_matrix"]["nan_same_pairs"])
    pc.atomic_json(EVID / "controls" / f"check_now_node_{vname}.json", out)
    print(f"[ri] check-now-node {vname}: {json.dumps(out)}", flush=True)


def check(vname: str) -> None:
    """Default contro la matrice (sintetiche) o contro esterna_famos_id (FaMoS). Nessuna GT. NoW sulle sintetiche: il
    controllo e' ``check_now_node`` sul tipo di nodo della matrice (il default della griglia si riporta)."""
    V = view_data(vname)
    i, j = V["i"], V["j"]
    out = {"view": vname, "pairs": int(len(i)), "checks": {}}
    if vname == "famos":
        import ei
        for m, src, key, tol in (("icp", "icp", "D_mm", "icp_mm"), ("now", "now_metrical", "D_mm", "now_mm")):
            with np.load(ei.dist_path("eval", src)) as z:
                if not (np.array_equal(z["i"], i) and np.array_equal(z["j"], j)):
                    raise SystemExit(f"famos {src}: coppie diverse")
                ref = np.asarray(z[key], np.float64)
            out["checks"][m] = {**maxdiff(load_dist(vname, m, V)["default"], ref), "ref": f"esterna_famos_id {src}",
                                "tol": TOL[tol]}
        out["checks"]["nicp"] = {"ref": "nessuno (il NICP per coppia non e' fra i metodi di esterna_famos_id)"}
    else:
        import bp_paired as bpp
        idx = bpp.be.zes.Index(blmm.subjects(vname))
        pi = np.asarray([idx.pos[(s, t)] for s, t, _ in V["items"]])
        root = blmm.view_root(vname) / "mm"
        refs = {"icp": (bpp.be.zes.facebench_distances(root, "rigid_icp_chamfer", idx), "icp_mm", "mm_rigid_icp_chamfer"),
                "nicp": (bpp.be.zes.facebench_distances(root, "nicp_p2tri", idx), "nicp_mm", "mm_nicp_p2tri")}
        for m, (M, tol, name) in refs.items():
            out["checks"][m] = {**maxdiff(load_dist(vname, m, V)["default"], M[pi[i], pi[j]]), "ref": name,
                                "tol": TOL[tol]}
        p = EVID / "controls" / f"check_now_node_{vname}.json"
        node = json.loads(p.read_text()) if p.exists() else {"ok": False, "here_vs_matrix": None}
        out["checks"]["now"] = {"ref": "plugin now_metrical/lmk_det", "tol": TOL["now_mm"], "ok": bool(node["ok"]),
                                "default_on_matrix_node": node["here_vs_matrix"], "node": node.get("host"),
                                "grid_vs_matrix": maxdiff(load_dist(vname, "now", V)["default"], now_matrix_ref(vname, V))}
    for c in out["checks"].values():
        if "tol" in c and "ok" not in c:
            c["ok"] = bool(c["max_abs"] is not None and c["max_abs"] <= c["tol"] and c["nan_same_pairs"])
    out["ok"] = all(c.get("ok", True) for c in out["checks"].values())
    out.update(pc.stamp())
    pc.atomic_json(EVID / "controls" / f"check_{vname}.json", out)
    print(f"[ri] check {vname}: {json.dumps(out['checks'])}", flush=True)


# ------------------------------------------------------------------------------------------ eval

def ci(x: np.ndarray) -> tuple[float, float]:
    b = x[1:][np.isfinite(x[1:])]
    return tuple(float(v) for v in np.percentile(b, [2.5, 97.5])) if len(b) else (np.nan, np.nan)


def groups_of() -> dict:
    """{metodo: (etichetta, [(configurazione, colonna)])}: i tre concorrenti e il nostro per lettura."""
    G = {m: (LABEL[m], [(c, col(m, c)) for c in GRID[m]]) for m in METHODS}
    G.update(OURS)
    return G


def synth_view(vname: str, workers: int) -> tuple[dict, dict]:
    """{colonna: Spearman con GT FR (1 + repliche)}, controlli."""
    import bp_paired as bpp
    import bp_paired_e2 as bpe2
    import bp_paired_e3 as bpe3
    import pr_eval as pe
    fp = bpp.fp
    V = view_data(vname)
    df, idx, _ = bpe2.all_cross_rows(vname)
    seed = pe.t1_seed(vname)
    keys = list(idx.keys)
    n = len(keys)
    pi = np.asarray([idx.pos[(s, t)] for s, t, _ in V["items"]])
    a, b = pi[V["i"]], pi[V["j"]]
    D = {}
    for m in METHODS:
        for c, x in load_dist(vname, m, V).items():
            M = np.full((n, n), np.nan)
            M[a, b] = M[b, a] = x
            D[col(m, c)] = M
    arms = fp.model_columns(vname, idx)
    D.update({c: arms[c] for _, cs in OURS.values() for _, c in cs})
    S = np.zeros((n, n))
    S[a, b] = S[b, a] = 1.0
    D["_sub"] = S
    df = bpe3.be.add_columns(df, D, idx)
    cols = [c for c in D if c != "_sub"]
    X = {m: df[m].to_numpy(np.float64) for m in cols + ["oracle_size"]}
    X.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in ("fr", "sr")})
    ta, tb = df["topology_a"].to_numpy(str), df["topology_b"].to_numpy(str)
    sub = (df["_sub"].to_numpy() > 0) & (ta != tb) & (ta != "crop") & (tb != "crop")
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
    s2i = {s: k for k, s in enumerate(subjects)}
    sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
    rng = np.random.default_rng(seed)
    counts = [np.ones(len(subjects), dtype=np.int64)] + \
        [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(N_BOOT)]
    mask = sub & (sa != sb) & np.all([np.isfinite(v) for v in X.values()], axis=0)
    R = bpe3.replicas(X, sa, sb, mask, cols, counts, workers)
    out = {c: np.asarray(R[("fr", "rho", c)], np.float64) for c in cols}
    info = {"seed": seed, "rows_subsample_nocrop": int(sub.sum()), "rows_mask": int(mask.sum()),
            "pairs_subsample_nocrop": int(len(V["i"])), "subjects": int(len(subjects)),
            "rows_nan_in_subsample_by_column": {c: int((~np.isfinite(X[c]) & sub).sum()) for c in cols}}
    print(f"[ri] eval {vname}: {int(mask.sum())}/{int(sub.sum())} righe, seme {seed}", flush=True)
    return out, info


def famos_view() -> tuple[dict, dict]:
    """{(metrica, colonna): rank-1 / mAP (1 + repliche)}, controlli. Regole di ``ei.evaluate``."""
    import ei
    import famos_eval as fe
    V = view_data("famos")
    it = V["items"]
    iu, ju = V["i"], V["j"]
    cols = {}
    for m in METHODS:
        cols.update({col(m, c): x for c, x in load_dist("famos", m, V).items()})
    for _, cs in OURS.values():
        for s, c in cs:
            m, r = c.split("|")
            with np.load(ei.dist_path("eval", m)) as z:
                if not (np.array_equal(z["i"], iu) and np.array_equal(z["j"], ju)):
                    raise SystemExit(f"famos {m}: coppie diverse")
                cols[c] = np.asarray(z[f"D_{r}"], np.float64)
    subj_names = sorted({x for x, _, _ in it})
    subj = np.asarray([subj_names.index(x) for x, _, _ in it])
    n, ns = len(it), len(subj_names)
    counts = np.concatenate([np.ones((1, ns), np.int64), fe.bootstrap_counts(ns, N_BOOT, ei.BOOT_SEED)])
    wq = counts[:, subj].astype(np.float64)
    out, failed = {}, {}
    for c, x in cols.items():
        failed[c] = int((~np.isfinite(x)).sum())
        Dm = np.zeros((n, n))
        Dm[iu, ju] = Dm[ju, iu] = np.where(np.isfinite(x), x, np.inf)
        r1, ap = ei.retrieval(Dm, subj)
        out[("rank1", c)] = (wq @ r1) / wq.sum(1)
        out[("mAP", c)] = (wq @ ap) / wq.sum(1)
    info = {"seed": ei.BOOT_SEED, "scans": n, "subjects": ns, "pairs": int(len(iu)), "non_finite_by_column": failed}
    print(f"[ri] eval famos: {n} scansioni, {len(iu)} coppie", flush=True)
    return out, info


def measures(R: dict, dom: str, mis: str) -> tuple[list, list, list, list]:
    """Righe di spread.csv, confronto_semi.csv, inversioni.csv, classifiche.csv da {colonna: (1 + repliche)}."""
    G = groups_of()
    SP, CS, INV, CL = [], [], [], []
    base = {"dominio": dom, "misura": mis}

    def stat(x):
        lo, hi = ci(x)
        return float(x[0]), lo, hi

    rng_of = {}
    for m, (lab, cs) in list(G.items()) + [("nostro_4", ("nostro, 4 letture (2 semi x cal./non cal.)",
                                                          G["nostro_form"][1] + G["nostro_form_cal"][1]))]:
        A = np.stack([R[c] for _, c in cs])                  # (configurazioni, 1 + repliche)
        rg = A.max(0) - A.min(0)
        sd = A.std(0, ddof=1)
        rng_of[m] = rg
        best, worst = int(np.argmax(A[:, 0])), int(np.argmin(A[:, 0]))
        names = [c for c, _ in cs]
        row = {**base, "metodo": m, "etichetta": lab, "n_config": len(cs)}
        row.update(zip(("intervallo", "intervallo_ic_lo", "intervallo_ic_hi"), stat(rg)))
        row.update(zip(("sd", "sd_ic_lo", "sd_ic_hi"), stat(sd)))
        row.update({"migliore": names[best], "peggiore": names[worst], "val_migliore": float(A[best, 0]),
                    "val_peggiore": float(A[worst, 0])})
        row.update(zip(("migliore_meno_peggiore", "mp_ic_lo", "mp_ic_hi"), stat(A[best] - A[worst])))
        if "default" in names:
            d = names.index("default")
            row["val_default"] = float(A[d, 0])
            row.update(zip(("migliore_meno_default", "md_ic_lo", "md_ic_hi"), stat(A[best] - A[d])))
            row.update(zip(("default_meno_peggiore", "dp_ic_lo", "dp_ic_hi"), stat(A[d] - A[worst])))
        SP.append(row)
    for r in READ:
        cs = G[f"nostro_{r}"][1]
        seed_gap = np.abs(R[cs[0][1]] - R[cs[1][1]])
        for m in METHODS:
            d = rng_of[m] - seed_gap
            lo, hi = ci(d)
            verdict = "piu' instabile del seme" if lo > 0 else "piu' stabile del seme" if hi < 0 else "non distinguibile"
            CS.append({**base, "metodo": m, "lettura_nostra": READ[r], "intervallo_metodo": float(rng_of[m][0]),
                       "scarto_semi": float(seed_gap[0]), "scarto_semi_ic_lo": ci(seed_gap)[0],
                       "scarto_semi_ic_hi": ci(seed_gap)[1], "s1234_meno_s2345": float(R[cs[0][1]][0] - R[cs[1][1]][0]),
                       "differenza": float(d[0]), "ic_lo": lo, "ic_hi": hi,
                       "p_le0": float((d[1:] <= 0).mean()), "lettura": verdict})
    order = list(METHODS) + [f"nostro_{r}" for r in READ]
    for ma, mb in itertools.combinations(order, 2):
        if ma.startswith("nostro") and mb.startswith("nostro"):
            continue
        n_pos = n_neg = s_pos = s_neg = 0
        deltas = []
        for (ca, xa), (cb, xb) in itertools.product(G[ma][1], G[mb][1]):
            d = R[xa] - R[xb]
            lo, hi = ci(d)
            deltas.append(d[0])
            n_pos += d[0] > 0
            n_neg += d[0] < 0
            s_pos += lo > 0
            s_neg += hi < 0
        INV.append({**base, "metodo_a": ma, "metodo_b": mb, "combinazioni": len(deltas), "a_meglio": int(n_pos),
                    "b_meglio": int(n_neg), "a_meglio_ic": int(s_pos), "b_meglio_ic": int(s_neg),
                    "inversione_punto": bool(n_pos and n_neg), "inversione_ic": bool(s_pos and s_neg),
                    "delta_min": float(min(deltas)), "delta_max": float(max(deltas))})
    for r in READ:
        ms = list(METHODS) + [f"nostro_{r}"]
        cnt = Counter()
        combos = list(itertools.product(*[G[m][1] for m in ms]))
        dflt = None
        for combo in combos:
            v = {m: R[c][0] for m, (_, c) in zip(ms, combo)}
            key = " > ".join(sorted(ms, key=lambda m: -v[m]))
            cnt[key] += 1
            if all(cf == "default" for m, (cf, _) in zip(ms[:3], combo[:3])) and combo[3][0] == "s1234":
                dflt = key
        for key, k in cnt.most_common():
            CL.append({**base, "lettura_nostra": READ[r], "classifica": key, "combinazioni": int(k),
                       "frazione": k / len(combos), "default_s1234": key == dflt, "classifiche_distinte": len(cnt)})
    return SP, CS, INV, CL


def evaluate(workers: int) -> None:
    import pandas as pd
    t0 = time.time()
    for v in VIEWS:
        p = EVID / "controls" / f"check_{v}.json"
        if not p.exists() or not json.loads(p.read_text())["ok"]:
            raise SystemExit(f"{v}: controllo di riproduzione assente o non passato ({p})")
    S, F, SP, CS, INV, CL, info = [], [], [], [], [], [], {}
    G = groups_of()
    for v in SYNTH:
        R, info[v] = synth_view(v, workers)
        for m, (lab, cs) in G.items():
            for c, x in cs:
                lo, hi = ci(R[x])
                S.append({"dominio": v, "gt": "fr", "metodo": m, "etichetta": lab, "config": c, "colonna": x,
                          "rho": float(R[x][0]), "ic_lo": lo, "ic_hi": hi, "n_righe": info[v]["rows_mask"],
                          "seme": info[v]["seed"], "manopola": GRID[m][c][1] if m in GRID else "seme",
                          "fonte": GRID[m][c][2] if m in GRID else "-"})
        for out, new in zip((SP, CS, INV, CL), measures(R, v, "rho_fr")):
            out += new
    Rf, info["famos"] = famos_view()
    for met in ("rank1", "mAP"):
        Rm = {c: x for (mm, c), x in Rf.items() if mm == met}
        for m, (lab, cs) in G.items():
            for c, x in cs:
                lo, hi = ci(Rm[x])
                F.append({"dominio": "famos", "metrica": met, "metodo": m, "etichetta": lab, "config": c, "colonna": x,
                          "valore": float(Rm[x][0]), "ic_lo": lo, "ic_hi": hi, "seme": info["famos"]["seed"]})
        for out, new in zip((SP, CS, INV, CL), measures(Rm, "famos", met)):
            out += new
    EVID.mkdir(parents=True, exist_ok=True)
    for name, rows in (("spearman", S), ("famos", F), ("spread", SP), ("confronto_semi", CS), ("inversioni", INV),
                       ("classifiche", CL)):
        pd.DataFrame(rows).to_csv(EVID / f"{name}.csv", index=False)
    pc.atomic_json(EVID / "controls.json", {**info, "n_boot": N_BOOT, "wall_s": time.time() - t0,
                                            "grid": {m: {c: {"params": p, "manopola": k, "fonte": f}
                                                         for c, (p, k, f) in g.items()} for m, g in GRID.items()},
                                            **pc.stamp()})
    results()


# ------------------------------------------------------------------------------------------ results

def f3(r, lo, hi, sign=False) -> str:
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"{f.format(r)} [{f.format(lo)}, {f.format(hi)}]"


def results() -> None:
    import pandas as pd
    S = pd.read_csv(EVID / "spearman.csv")
    F = pd.read_csv(EVID / "famos.csv")
    SP = pd.read_csv(EVID / "spread.csv")
    CS = pd.read_csv(EVID / "confronto_semi.csv")
    INV = pd.read_csv(EVID / "inversioni.csv")
    CL = pd.read_csv(EVID / "classifiche.csv")
    ctrl = json.loads((EVID / "controls.json").read_text())
    G = groups_of()
    L = ["# Robustezza agli iperparametri: allineamento per coppia contro la variabilita' da seme del nostro", "",
         "Generato da `aau/robustezza_iperparametri/ri.py` (regole in `REGOLE.md`, committate prima dei numeri; "
         "definizioni nel docstring). Lettura DESCRITTIVA. HIFI3D e FaceScape: Spearman con GT FR sulle righe senza "
         "crop del sottocampione del rumore landmark (8%, seme 20261013), una maschera per vista; FaMoS: rank-1 e mAP "
         f"sulle 150 scansioni di `esterna_famos_id`. {N_BOOT} repliche per soggetto, IC 95% percentile.", "",
         "## Griglie", "", "| metodo | configurazione | manopola | fonte |", "|---|---|---|---|"]
    for m, g in GRID.items():
        for c, (_, k, f) in g.items():
            L.append(f"| {LABEL[m]} | {c} | {k} | {f} |")
    L += ["", "Righe: " + "; ".join(f"{v} {ctrl[v]['rows_mask']} / {ctrl[v]['rows_subsample_nocrop']} del sottocampione "
                                    "senza crop" for v in SYNTH) + "; FaMoS "
          f"{ctrl['famos']['scans']} scansioni, {ctrl['famos']['pairs']} coppie.", ""]
    for dom, mis, X, vc in [(v, "rho_fr", S[S["dominio"] == v], "rho") for v in SYNTH] + \
            [("famos", met, F[F["metrica"] == met], "valore") for met in ("rank1", "mAP")]:
        L += [f"## {dom}, {mis}", "", "| metodo | configurazione | valore [IC] |", "|---|---|---|"]
        for _, r in X.iterrows():
            L.append(f"| {r['etichetta']} | {r['config']} | {f3(r[vc], r['ic_lo'], r['ic_hi'])} |")
        Y = SP[(SP["dominio"] == dom) & (SP["misura"] == mis)]
        L += ["", "Spread fra configurazioni (migliore e peggiore scelte sul punto: post hoc):", "",
              "| metodo | n | intervallo [IC] | SD [IC] | migliore - peggiore [IC] | default - peggiore [IC] | "
              "migliore / peggiore |", "|---|---|---|---|---|---|---|"]
        for _, r in Y.iterrows():
            dp = f3(r["default_meno_peggiore"], r["dp_ic_lo"], r["dp_ic_hi"], True) \
                if pd.notna(r.get("default_meno_peggiore")) else "-"
            L.append(f"| {r['etichetta']} | {r['n_config']} | {f3(r['intervallo'], r['intervallo_ic_lo'], r['intervallo_ic_hi'])} | "
                     f"{f3(r['sd'], r['sd_ic_lo'], r['sd_ic_hi'])} | "
                     f"{f3(r['migliore_meno_peggiore'], r['mp_ic_lo'], r['mp_ic_hi'], True)} | {dp} | "
                     f"{r['migliore']} / {r['peggiore']} |")
        Y = CS[(CS["dominio"] == dom) & (CS["misura"] == mis)]
        L += ["", "Intervallo del metodo - scarto fra i nostri semi (stesse repliche):", "",
              "| metodo | lettura nostra | scarto fra i semi [IC] | intervallo - scarto [IC] | lettura |",
              "|---|---|---|---|---|"]
        for _, r in Y.iterrows():
            L.append(f"| {LABEL[r['metodo']]} | {r['lettura_nostra']} | "
                     f"{f3(r['scarto_semi'], r['scarto_semi_ic_lo'], r['scarto_semi_ic_hi'])} | "
                     f"{f3(r['differenza'], r['ic_lo'], r['ic_hi'], True)} | {r['lettura']} |")
        Y = INV[(INV["dominio"] == dom) & (INV["misura"] == mis)]
        L += ["", "Inversioni (tutte le combinazioni di configurazioni; IC = delta appaiato con IC che esclude 0):", "",
              "| a | b | combinazioni | a meglio | b meglio | a meglio (IC) | b meglio (IC) | delta min / max |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in Y.iterrows():
            L.append(f"| {G[r['metodo_a']][0]} | {G[r['metodo_b']][0]} | {r['combinazioni']} | {r['a_meglio']} | "
                     f"{r['b_meglio']} | {r['a_meglio_ic']} | {r['b_meglio_ic']} | "
                     f"{r['delta_min']:+.3f} / {r['delta_max']:+.3f} |")
        Y = CL[(CL["dominio"] == dom) & (CL["misura"] == mis)]
        L += ["", "Classifiche dei quattro metodi (sul punto, tutte le combinazioni; * = configurazioni di default e "
                  "s1234):", ""]
        for r_ in READ.values():
            Z = Y[Y["lettura_nostra"] == r_]
            if Z.empty:
                continue
            L.append(f"- nostro {r_}: {int(Z['classifiche_distinte'].iloc[0])} classifiche distinte; " + "; ".join(
                f"{r['classifica']}{' *' if r['default_s1234'] else ''} ({r['frazione']:.0%})"
                for _, r in Z.head(4).iterrows()))
        L.append("")
    p = EVID / "results.md"
    tail = p.read_text().split(MARK, 1)[1] if p.exists() and MARK in p.read_text() else "\n"
    p.write_text("\n".join(L) + "\n" + MARK + tail)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("run", "merge", "check", "check-now-node", "eval", "results"))
    p.add_argument("--view", default="", choices=("",) + VIEWS)
    p.add_argument("--method", default="", choices=("",) + METHODS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="run: pilota dei tempi su N coppie / mesh A (<WORK>/pilot)")
    a = p.parse_args()
    if a.step == "run":
        run(a.view, a.method, a.workers, a.limit)
    elif a.step == "merge":
        merge(a.view, a.method)
    elif a.step == "check":
        check(a.view)
    elif a.step == "check-now-node":
        check_now_node(a.view, a.workers)
    elif a.step == "eval":
        evaluate(a.workers)
    else:
        results()


if __name__ == "__main__":
    main()
