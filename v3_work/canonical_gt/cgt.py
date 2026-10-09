"""GT canoniche di E12: forma metrica (F), EDMA (EDM), forma senza taglia (S), varianti e arbitro.

Definizioni e scelte: ``aau/runs/evidence/e12/protocol.md`` (sezioni 0-5 ed Emendamento 1, scritti prima dei
calcoli). Tutte le GT per vertici stanno sulla regione unificata (``datasets/UNIFIED_GT/unified_space.npz``, 1478
punti, mappe baricentriche per dominio) con i pesi d'area FISSI ``w`` di mu: g_ij = sqrt(sum_v w_v ||x_i,v -
x_j,v||^2 / A), distanza euclidea pesata, in mm.

  - F:       f_i = u_d R_d p_i + t_d, una sola rigida per dominio (media del dominio -> media FLAME) dopo la
             conversione delle unita' (``u_d``: unita' dichiarata, o 63 mm / IPD per le unita' ignote);
             sulle catture reali (FaMoS, Multiface) rigida robusta per identita' verso mu.
  - F_centered, F_rig_ls, F_rig_rob: traslazione, rigida LS, rigida robusta (IRLS Tukey) per identita'.
  - S:       m + (f_i - m) CS(mu) / CS_i, una sola scala per identita' attorno al centroide fisso m di mu.
  - EDM, EDM_s: distanze interne fra K = 400 punti campionati per area (EDMA), Frobenius a pesi uniformi;
             EDM_s divide ogni matrice per la sua media geometrica.
  - unified: Procrustes di similarita' verso mu (``shapes.Space.align``, la GT di E8), ricalcolata per controllo.
  - F_json:  la canonica della prima versione del protocollo (similarita' del json, scala inclusa): controllo.
Baseline banali: |log CS_i - log CS_j| e |log H_i - log H_j| (H = estensione verticale della regione in F).

Il codice della GT unificata (``v3_work/unified_gt``) e' importato, non modificato.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (REPO_ROOT / "v3_work" / "unified_gt", REPO_ROOT / "aau" / "famos"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ugt as C  # noqa: E402
import domains  # noqa: E402
from shapes import Space, ids_range  # noqa: E402

OUT_DIR = REPO_ROOT / "datasets" / "CANONICAL_GT"          # matrici GT (fuori da git)
EVID_DIR = REPO_ROOT / "aau" / "runs" / "evidence" / "e12"
FRAMES_JSON = EVID_DIR / "frames.json"                       # u_d, R_d, t_d di GT-F (scritto da gt.py)
TRANSFORMS = C.DATA_ROOT / "canonical_transforms.json"
DATASETS = REPO_ROOT / "datasets"
# set di valutazione: (cartella, prefisso della GT raw di build_zs_gt.py, offset degli id)
EVAL_SETS = {"hifi3d": ("HIFI3D", "hifi", 900000), "faceverse": ("FACEVERSE_ZS", "fv", 910000),
             "facescape": ("DEV_FACESCAPE", "fs", 930000)}
# Unita' dichiarate come unita' fisiche in domains.py (mm per unita'); None = ignota -> 63 mm / IPD
UNITS_DECLARED = {"flame": 1000.0, "bfm": 1e-3, "ict": 10.0, "gnm": 1000.0, "facescape": 1.0, "multiface": 1.0,
                  "hifi3d": None, "faceverse": None}
# GT salvate per i set di valutazione (lette da methods.py)
SAVED = ("F", "F_centered", "F_rig_ls", "F_rig_rob", "S", "EDM", "EDM_s", "F_json", "unified")
# iBUG 68 (base 0): contorno dell'occhio destro e sinistro dell'immagine
EYE_R, EYE_L = tuple(range(36, 42)), tuple(range(42, 48))
IPD_REF_MM = 63.0
UNIT_CANDIDATES = {"micrometri": 1e-3, "millimetri": 1.0, "centimetri": 10.0, "decimetri": 100.0, "metri": 1000.0}
UNIT_TOL = 0.15
EDM_K, EDM_SEED = 400, 1234
TUKEY_C, IRLS_MAX_IT, IRLS_TOL_R, IRLS_TOL_T = 3.0, 100, 1e-9, 1e-7


# ---------------------------------------------------------------------------- spazio comune

class Canon:
    """Spazio unificato (mappe, mu, pesi), trasformazioni del json e frame di GT-F (se gia' calcolati)."""

    def __init__(self, frames: dict | None = None):
        self.sp = Space()
        self.ct = json.loads(TRANSFORMS.read_text())["domains"]
        self.W = self.sp.W                       # pesi normalizzati (somma 1)
        self.mu = self.sp.mu
        if frames is None and FRAMES_JSON.exists():
            frames = json.loads(FRAMES_JSON.read_text())["domains"]
        self.frames = frames or {}
        self._edm = None

    # -------------------------------------------------------------- trasformazioni di dominio
    def json_transform(self, d: str) -> tuple[float, np.ndarray, np.ndarray]:
        c = self.ct[d]
        return float(c["s"]), np.asarray(c["R"], dtype=np.float64), np.asarray(c["t"], dtype=np.float64)

    def to_json_frame(self, d: str, P: np.ndarray) -> np.ndarray:
        s, R, t = self.json_transform(d)
        return s * np.asarray(P, dtype=np.float64) @ R.T + t

    def to_F(self, d: str, P: np.ndarray) -> np.ndarray:
        """GT-F sui 3DMM: u_d R_d p + t_d. ``famos``: neutre FLAME gia' in mm, nessuna trasformazione."""
        if d == "famos":
            return np.asarray(P, dtype=np.float64).copy()
        f = self.frames[d]
        return f["u"] * np.asarray(P, dtype=np.float64) @ np.asarray(f["R"]).T + np.asarray(f["t"])

    # ---------------------------------------------------------------- varianti per identita'
    def centroid(self, X: np.ndarray) -> np.ndarray:
        return np.einsum("v,nvd->nd", self.W, X)

    def centered(self, X: np.ndarray) -> np.ndarray:
        return X - self.centroid(X)[:, None]

    def centroid_size(self, X: np.ndarray) -> np.ndarray:
        """sqrt(sum_v W_v ||x_v - centroide||^2), mm (W normalizzati)."""
        return np.sqrt(np.einsum("v,nv->n", self.W, (self.centered(X) ** 2).sum(-1)))

    def height(self, X: np.ndarray) -> np.ndarray:
        return X[..., 1].max(-1) - X[..., 1].min(-1)

    def rigid_ls(self, X: np.ndarray) -> dict:
        R, t = rigid_fit(X, self.mu, np.broadcast_to(self.W, X.shape[:2]))
        return _rigid_out(X, R, t, self.mu, self.W)

    def rigid_robust(self, X: np.ndarray) -> dict:
        """Resistant fit: IRLS con Tukey biweight sui residui per punto verso mu (protocollo, Emendamento 1)."""
        w = self.W
        R, t = rigid_fit(X, self.mu, np.broadcast_to(w, X.shape[:2]))
        it_done = np.zeros(len(X), dtype=np.int32)
        conv = np.zeros(len(X), dtype=bool)
        for it in range(1, IRLS_MAX_IT + 1):
            r = np.linalg.norm(np.einsum("nvb,nab->nva", X, R) + t[:, None] - self.mu[None], axis=-1)
            c = np.maximum(TUKEY_C * weighted_median(r, w), 1e-12)
            psi = np.clip(1.0 - (r / c[:, None]) ** 2, 0.0, None) ** 2
            R2, t2 = rigid_fit(X, self.mu, w[None] * psi)
            dR = np.abs(R2 - R).max(axis=(1, 2))
            dt = np.abs(t2 - t).max(axis=1)
            R, t = R2, t2
            it_done[~conv] = it
            conv |= (dR < IRLS_TOL_R) & (dt < IRLS_TOL_T)
            if conv.all():
                break
        out = _rigid_out(X, R, t, self.mu, w)
        r = np.linalg.norm(out["a"] - self.mu[None], axis=-1)
        c = np.maximum(TUKEY_C * weighted_median(r, w), 1e-12)
        psi = np.clip(1.0 - (r / c[:, None]) ** 2, 0.0, None) ** 2
        out.update(iterations=it_done, converged=conv, downweighted_area=np.einsum("v,nv->n", w, psi < 0.5))
        return out

    def shape_S(self, Xf: np.ndarray) -> np.ndarray:
        m = self.W @ self.mu
        ref = float(self.centroid_size(self.mu[None])[0])
        return m + (Xf - m) * (ref / self.centroid_size(Xf))[:, None, None]

    def unified(self, X: np.ndarray) -> np.ndarray:
        return self.sp.align(X)[0]

    # ------------------------------------------------------------------------------- EDMA
    def edm_points(self) -> tuple[np.ndarray, np.ndarray]:
        """(vidx (K, 3), bary (K, 3)): K punti uniformi per area sui triangoli della regione su mu, seme fisso."""
        if self._edm is None:
            F = self.sp.z["F"]
            area = C.face_areas(self.mu, F)
            rng = np.random.default_rng(EDM_SEED)
            tri = rng.choice(len(F), size=EDM_K, p=area / area.sum())
            r1, r2 = rng.random(EDM_K), rng.random(EDM_K)
            s1 = np.sqrt(r1)
            self._edm = (F[tri], np.stack([1.0 - s1, s1 * (1.0 - r2), s1 * r2], axis=1))
        return self._edm

    def edm_features(self, X: np.ndarray, scale_free: bool = False, chunk: int = 64) -> np.ndarray:
        """(n, K(K-1)/2) distanze interne fra i K punti; ``scale_free``: divise per la media geometrica e
        moltiplicate per quella di mu."""
        vidx, bary = self.edm_points()
        iu = np.triu_indices(EDM_K, 1)
        out = np.empty((len(X), len(iu[0])))
        for k in range(0, len(X), chunk):
            out[k:k + chunk] = _internal(np.einsum("nkcd,kc->nkd", X[k:k + chunk][:, vidx], bary), iu)
        if scale_free:
            ref = np.exp(np.log(_internal(np.einsum("kcd,kc->kd", self.mu[vidx], bary)[None], iu)).mean())
            out = out / np.exp(np.log(np.clip(out, 1e-12, None)).mean(1, keepdims=True)) * ref
        return out

    # ---------------------------------------------------------------------------- distanze
    def distances(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """g_ij = sqrt(sum_v w_v ||x_i,v - y_j,v||^2 / A), mm, in float64."""
        sq = np.sqrt(self.sp.w)[None, :, None]
        S1 = (sq * X).reshape(len(X), -1)
        S2 = S1 if Y is None else (sq * Y).reshape(len(Y), -1)
        return _euclid(S1, S2, self.sp.A, Y is None)


def _internal(Q: np.ndarray, iu) -> np.ndarray:
    """Distanze interne (n, len(iu)) di punti (n, K, 3), dalla matrice di Gram dei punti centrati."""
    Q = Q - Q.mean(1, keepdims=True)
    sq = (Q ** 2).sum(-1)
    G = sq[:, :, None] + sq[:, None, :] - 2.0 * np.einsum("nkd,nld->nkl", Q, Q)
    return np.sqrt(np.clip(G[:, iu[0], iu[1]], 0.0, None))


def edm_distances(E: np.ndarray) -> np.ndarray:
    """sqrt(mean_ab (FM_i - FM_j)^2): Frobenius a pesi uniformi, nelle unita' delle distanze interne."""
    return _euclid(E, E, E.shape[1], True)


def _euclid(S1: np.ndarray, S2: np.ndarray, denom: float, square: bool) -> np.ndarray:
    G = (S1 ** 2).sum(1)[:, None] + (S2 ** 2).sum(1)[None, :] - 2.0 * S1 @ S2.T
    D = np.sqrt(np.clip(G, 0.0, None) / denom)
    if square:
        np.fill_diagonal(D, 0.0)
        D = 0.5 * (D + D.T)
    return D


def scalar_distances(x: np.ndarray) -> np.ndarray:
    """|log x_i - log x_j| (baseline "solo taglia" / "solo altezza")."""
    lx = np.log(np.asarray(x, dtype=np.float64))
    return np.abs(lx[:, None] - lx[None, :])


# ---------------------------------------------------------------------------- rigide

def rigid_fit(X: np.ndarray, Y: np.ndarray, Wt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Umeyama pesato SENZA scala di ogni forma X[n] (m, 3) verso Y (m, 3), pesi per forma Wt (n, m):
    (R (n, 3, 3), t (n, 3)) con R x + t ~ y."""
    Wn = Wt / Wt.sum(1, keepdims=True)
    mx = np.einsum("nv,nvd->nd", Wn, X)
    my = Wn @ Y
    Cm = np.einsum("nv,nva,nvb->nab", Wn, Y[None] - my[:, None], X - mx[:, None])
    U, _, Vt = np.linalg.svd(Cm)
    D = np.ones((len(X), 3))
    D[:, 2] = np.sign(np.linalg.det(U @ Vt))
    R = np.einsum("nab,nb,nbc->nac", U, D, Vt)
    return R, my - np.einsum("nab,nb->na", R, mx)


def _rigid_out(X, R, t, mu, W) -> dict:
    a = np.einsum("nvb,nab->nva", X, R) + t[:, None]
    mx = np.einsum("v,nvd->nd", W, X)
    return {"a": a, "R": R, "t": t, "angle": rot_angle(R), "offset": np.linalg.norm(mx - W @ mu, axis=1)}


def weighted_median(r: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Mediana pesata di ogni riga di r (n, m) con pesi w (m,)."""
    o = np.argsort(r, axis=1)
    cw = np.cumsum(w[o], axis=1)
    k = (cw < 0.5 * cw[:, -1:]).sum(1)
    return np.take_along_axis(r, o, axis=1)[np.arange(len(r)), k]


def rot_angle(R: np.ndarray) -> np.ndarray:
    """Angolo (gradi) di rotazioni (n, 3, 3)."""
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))


def chordal_mean(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R.mean(0))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


# ------------------------------------------------------------------------- tutte le GT di un set

def all_gts(cn: Canon, d: str, P: np.ndarray, real: bool, extra: dict | None = None) -> tuple[dict, dict]:
    """Matrici (n, n) di tutte le GT dei punti nativi ``P`` del dominio ``d`` e i diagnostici.

    ``real``: catture reali (posa della testa per cattura): F = rigida robusta, F pura = frame di cattura, niente
    F_centered ne' F_json."""
    X0 = cn.to_F(d, P)
    rob = cn.rigid_robust(X0)
    ls = cn.rigid_ls(X0)
    Xf = rob["a"] if real else X0
    D = {"F": cn.distances(Xf), "F_rig_ls": cn.distances(ls["a"]), "F_rig_rob": cn.distances(rob["a"]),
         "S": cn.distances(cn.shape_S(Xf)), "EDM": edm_distances(cn.edm_features(X0)),
         "EDM_s": edm_distances(cn.edm_features(X0, scale_free=True)), "unified": cn.distances(cn.unified(X0))}
    if real:
        D["F_pure"] = cn.distances(X0)
    else:
        D["F_centered"] = cn.distances(cn.centered(X0))
        D["F_json"] = cn.distances(cn.to_json_frame(d, P))
    cs, h = cn.centroid_size(Xf), cn.height(Xf)
    D["size_only"], D["height_only"] = scalar_distances(cs), scalar_distances(h)
    Rm = chordal_mean(ls["R"])
    diag = {"centroid_size_mm": cs, "height_mm": h, "Xf": Xf,
            "rigid_ls": {"angle_deg": ls["angle"], "offset_mm": ls["offset"]},
            "rigid_rob": {"angle_deg": rob["angle"], "iterations": rob["iterations"], "converged": rob["converged"],
                          "downweighted_area": rob["downweighted_area"],
                          "angle_ls_vs_rob_deg": rot_angle(np.einsum("nab,ncb->nac", rob["R"], ls["R"]))},
            "pose_spread": {"angle_from_mean_rotation_deg": rot_angle(np.einsum("nab,cb->nac", ls["R"], Rm)),
                            "centroid_from_median_mm": np.linalg.norm(cn.centroid(X0) - np.median(cn.centroid(X0), 0),
                                                                      axis=1)},
            "rob": rob}
    return D, diag


# ------------------------------------------------------------------------------ landmark occhi

def eye_proxies(cn: Canon) -> dict:
    """Indici (12,) nella regione unificata dei vertici piu' vicini ai 12 landmark del contorno degli occhi
    sulla media FLAME (mm), con la distanza del sostituto e l'IPD esatta e sostituita della media FLAME."""
    import famos_common as fc

    flame = domains.flame()
    Vf = flame["V"] * 1000.0
    fi, bc = fc.flame_lmk68()
    L = fc.landmarks(Vf, flame["F_render"], fi, bc, EYE_R + EYE_L)
    P = Vf[cn.sp.z["flame_vidx"]]
    d = np.linalg.norm(L[:, None] - P[None], axis=-1)
    li = d.argmin(1)
    exact = float(np.linalg.norm(L[:6].mean(0) - L[6:].mean(0)))
    return {"idx": li, "proxy_mm": d[np.arange(len(li)), li], "flame_ipd_exact_mm": exact,
            "flame_ipd_proxy_mm": float(ipd(P[None], li)[0])}


def ipd(X: np.ndarray, li: np.ndarray) -> np.ndarray:
    """Distanza fra i centri dei due occhi (media dei 6 punti del contorno), (n,)."""
    return np.linalg.norm(X[:, li[:6]].mean(1) - X[:, li[6:]].mean(1), axis=1)


def nearest_unit(mm_per_unit: float) -> tuple[str, float, float]:
    """(nome, mm per unita', rapporto stimato / candidato) della potenza di 10 piu' vicina in log; nome
    ``arbitraria`` se il rapporto sta fuori da 1 +- UNIT_TOL."""
    name, u = min(UNIT_CANDIDATES.items(), key=lambda kv: abs(np.log(mm_per_unit / kv[1])))
    r = mm_per_unit / u
    return (name if abs(r - 1.0) <= UNIT_TOL else "arbitraria"), u, r


# ----------------------------------------------------------------------------- identificabilita'

class Ident:
    """Arbitro (protocollo, Emendamento 1, E2): AUC di verifica, rank-1, rapporto intra / inter, con le stesse
    repliche bootstrap per persona per tutte le GT. ``person`` (N,) etichette delle catture."""

    def __init__(self, person: np.ndarray, n_boot: int, seed: int):
        self.person = np.asarray(person)
        self.persons, self.pidx = np.unique(self.person, return_inverse=True)
        n = len(self.persons)
        self.iu = np.triu_indices(len(self.person), 1)
        self.gen = self.pidx[self.iu[0]] == self.pidx[self.iu[1]]
        self.n_per = np.bincount(self.pidx, minlength=n)
        rng = np.random.default_rng(seed)
        self.counts = [np.ones(n, dtype=np.int64)] + [np.bincount(rng.integers(0, n, size=n), minlength=n)
                                                      for _ in range(n_boot)]

    def evaluate(self, D: np.ndarray) -> dict:
        """Repliche (punto in posizione 0) di ``auc``, ``rank1``, ``ratio``, e i conteggi."""
        d = D[self.iu]
        pa, pb = self.pidx[self.iu[0]], self.pidx[self.iu[1]]
        dg, ga = d[self.gen], pa[self.gen]
        oi = np.argsort(d[~self.gen], kind="stable")
        di, ia, ib = d[~self.gen][oi], pa[~self.gen][oi], pb[~self.gen][oi]
        lo = np.searchsorted(di, dg, side="left")
        hi = np.searchsorted(di, dg, side="right")
        og = np.argsort(dg, kind="stable")
        dgs, gas = dg[og], ga[og]
        Dn = D + np.diag(np.full(len(D), np.inf))
        hit = self.pidx[Dn.argmin(1)] == self.pidx
        hits = np.bincount(self.pidx, weights=hit, minlength=len(self.persons))
        auc, r1, ratio = [], [], []
        for c in self.counts:
            wi = (c[ia] * c[ib]).astype(np.float64)
            cw = np.concatenate([[0.0], np.cumsum(wi)])
            wg = c[ga].astype(np.float64)
            below, upto, tot = cw[lo], cw[hi], cw[-1]
            auc.append(float((wg * (tot - upto + 0.5 * (upto - below))).sum() / (wg.sum() * tot)))
            r1.append(float((c * hits).sum() / (c * self.n_per).sum()))
            cg = np.cumsum(c[gas].astype(np.float64))
            med_g = dgs[np.searchsorted(cg, 0.5 * cg[-1])]
            med_i = di[np.searchsorted(cw[1:], 0.5 * tot)]
            ratio.append(float(med_g / med_i))
        return {"auc": np.asarray(auc), "rank1": np.asarray(r1), "ratio": np.asarray(ratio),
                "n_genuine": int(self.gen.sum()), "n_impostor": int((~self.gen).sum()),
                "n_persons": len(self.persons), "n_captures": len(self.person)}


# ---------------------------------------------------------------------------- punti nativi

def view_names(name: str) -> list[str]:
    root = EVAL_SETS[name][0]
    with np.load(DATASETS / root / "eval_view" / "gt_matrix.npz") as z:
        return [str(s).split("_GTready")[0] for s in z["names"]]


def native_points(name: str, cn: Canon) -> dict:
    """Punti della regione (n, m, 3) nel frame e nelle unita' dei DATI, per il set ``name``.

    ``domain`` = la chiave delle trasformazioni. Eval: hifi3d, faceverse, facescape (pool di 500 della vista,
    nell'ordine dei ``names``), famos (95 neutre registrate, mm, ``V_full`` la mesh FLAME intera). Training (solo
    dimensione e unita'): flame (1000 N(0, 1), seme 1234, come shapes.py), bfm (500 REMESH original), ict
    (ICT-5000), gnm (GNM_DISTILL), multiface (13, media dei frame neutri come shapes.py)."""
    sp = cn.sp
    out = {"checks": {}}
    if name in EVAL_SETS:
        import make_eval_gt as meg
        ids = view_names(name)
        P, checks = meg.mapped_points(name, ids, sp)
        out.update(ids=ids, P=P, domain=name, checks=checks)
    elif name == "famos":
        import famos_common as fc
        split = fc.load_split()
        ids, splits, V = [], [], []
        for sub, d in (("train", fc.TRAIN_DIR), ("test", fc.TEST_DIR)):
            for s in split[sub]:
                with np.load(d / f"{s}.npz") as z:
                    if str(z["subject"]) != s or str(z["units"]) != "mm":
                        raise SystemExit(f"{d / s}.npz: soggetto o unita' inattesi")
                    V.append(z["V_neutral"].astype(np.float64))
                ids.append(s)
                splits.append(sub)
        V = np.stack(V)
        out.update(ids=ids, split=np.asarray(splits), P=sp.map("flame", V), V_full=V, domain="famos")
    elif name in ("flame", "ict", "gnm"):
        tpl = domains.template(name)
        M0, B = sp.linear(name, tpl)
        if name == "flame":
            ids = [f"flame{k:04d}" for k in range(1000)]
            W = np.random.default_rng(1234).normal(size=(1000, 300))
        elif name == "ict":
            ids = ids_range(10000, 15000)
            W = domains.ict_weights(ids)
        else:
            ids = ids_range(100000, 110100)
            W = domains.gnm_weights(ids)
        P = M0[None] + np.einsum("nk,kvd->nvd", W[:, : B.shape[0]], B)
        out.update(ids=ids, P=P, domain=name)
    elif name == "bfm":
        paths = domains.bfm_original_paths()
        P = np.stack([sp.map("bfm", domains.load_bfm_original(p)[0]) for p in paths])
        out.update(ids=[p.name.split("_GTready_")[0] for p in paths], P=P, domain="bfm")
    elif name == "multiface":
        ids, P = [], []
        for subj, paths in sorted(domains.multiface_neutral_paths().items()):
            Vs = []
            for p in paths:
                with np.load(p) as z:
                    V = np.asarray(z["V"], np.float64)
                if Vs:
                    V = C.apply_sim(V, *C.umeyama(V, Vs[0], scale=False))
                Vs.append(V)
            ids.append(subj)
            P.append(sp.map("multiface", np.mean(Vs, axis=0)))
        out.update(ids=ids, P=np.stack(P), domain="multiface")
    else:
        raise SystemExit(f"set sconosciuto: {name}")
    return out


def famos_captures(cn: Canon, kept_only: bool = False) -> dict:
    """Primo fotogramma di ogni sequenza di ogni persona FaMoS (``first_frames``; con ``kept_only`` solo quelli
    tenuti per la neutra, ``neutral_from``): punti della regione (N, m, 3) in mm, mesh intere, persona, sequenza."""
    import famos_common as fc
    split = fc.load_split()
    P, Vfull, person, seq, sp_of = [], [], [], [], []
    for sub, d in (("train", fc.TRAIN_DIR), ("test", fc.TEST_DIR)):
        for s in split[sub]:
            with np.load(d / f"{s}.npz") as z:
                want = [str(x) for x in (z["neutral_from"] if kept_only else z["first_frames"])]
                key = {f"{q}.{int(f):06d}": i for i, (q, f) in enumerate(zip(z["seq"].astype(str), z["frame"]))}
                miss = [w for w in want if w not in key]
                if miss:
                    raise SystemExit(f"{s}: primi fotogrammi assenti dai fotogrammi salvati: {miss[:3]}")
                V = z["V"][[key[w] for w in want]].astype(np.float64)
            Vfull.append(V)
            P.append(cn.sp.map("flame", V))
            person += [s] * len(want)
            seq += [w.rsplit(".", 1)[0] for w in want]
            sp_of += [sub] * len(want)
    return {"P": np.concatenate(P), "V_full": np.concatenate(Vfull), "person": np.asarray(person),
            "seq": np.asarray(seq), "split": np.asarray(sp_of)}


def multiface_captures(cn: Canon) -> dict:
    """Tutti i fotogrammi neutri tracked di Multiface (una sola ripresa per persona), punti in mm."""
    P, person, seg = [], [], []
    for subj, paths in sorted(domains.multiface_neutral_paths().items()):
        for p in paths:
            with np.load(p) as z:
                P.append(cn.sp.map("multiface", np.asarray(z["V"], np.float64)))
            person.append(subj)
            seg.append(p.stem.split("__")[1])
    return {"P": np.stack(P), "person": np.asarray(person), "seq": np.asarray(seg)}


# ------------------------------------------------------------------------------------- I/O

def save_gt(name: str, variant: str, D: np.ndarray, ids: list[str], extra: dict | None = None) -> Path:
    """``D_orig`` in mm (float64, NON diviso per il massimo: ``load_gt_distance_matrix`` lo divide), ``names``."""
    path = OUT_DIR / f"{name}_{variant}.npz"
    C.save_npz(path, D_orig=D, names=np.asarray(ids))
    iu = np.triu_indices(len(D), 1)
    man = {"set": name, "variant": variant, "n": len(ids), "median": float(np.median(D[iu])),
           "min_offdiag": float(D[iu].min()), "max": float(D.max()),
           "definition": "aau/runs/evidence/e12/protocol.md (Emendamento 1)", **(extra or {})}
    path.with_suffix(".json").write_text(json.dumps(man, indent=1) + "\n")
    return path
