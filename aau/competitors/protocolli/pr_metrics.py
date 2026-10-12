"""I quattro protocolli come errore di una coppia ORDINATA (A predetta, B riferimento): logica portata dal codice
ufficiale, riscritta (nessuna riga di terzi copiata; now_evaluation non ha licenza nel repo, REALY e fg2018 sono MIT,
chumpy e' MIT). Riferimenti ai cloni (``pr_common.CLONE_COMMITS``):

  [N]  now_evaluation/scan2mesh_computations.py, [Nm] .../scan2mesh_computations_metrical.py
  [S]  now_evaluation/sbody/mesh_distance.py (ScanToMesh), [CH] chumpy/optimization_internal.py (dogleg, chumpy 0.70
       del venv external/venvs/now)
  [R]  REALY/main.py, [Ru] REALY/utils/util.py, [Rg] gICP.py, [Rr] rICP.py, [Re] eval.py, [Rn] NICP.py
  [F]  fg2018-competition/compute_vertices_to_mesh_distances.py

NoW originale ([N] compute_errors :146-171): riferimento ritagliato col cerchio di ``compute_mask`` (:56-72, centro
subnasale + 0.3 (ponte - subnasale), raggio 1.4 (ex-ex + naso) / 2) sui landmark di B; Procrustes a 7 punti con scala e
riflessione 'best' (:126, :173-285); ICP scan-to-mesh con s, r (Rodrigues), t liberi (:23-54): dogleg di chumpy con
``delta_0 = 0.1``, ``e_3 = 1e-4`` (:24-27), residui = distanza di ogni vertice del riferimento ritagliato dalla
superficie di A trasformata ([S] ScanToMesh, ``signed=False, normalize=False``) + ``s_reg = 100 (|s| - s)``; errore =
le stesse distanze dopo l'allineamento (:169-171), qui la MEDIA (mm). NoW metrical ([Nm]): Procrustes ancora con scala
ma applicato con il fattore d'unita' 1 (mm) al posto della scala stimata e con la traslazione calcolata con la scala
(:117-125, riprodotto com'e'); ICP su [r, t] soltanto (:53).

Stirling/ESRC ([F] :25-62): Procrustes come NoW, riferimento ritagliato col raggio 1.2 (ex-ex + naso) / 2 (:46), nessun
ICP, distanze vertice -> superficie di A; errore = RMSE (paper, Sez. 3: "3D-RMSE").

REALY ([R] :155-262): gICP = similarita' sui 7 keypoint con la scala come rapporto di norme L1 ([Ru] fit_icp_RT
:76-109, [Rg] :9-29); per ognuna delle 4 regioni del riferimento un ICP rigido separato con i keypoint della regione
ripetuti come peso ([Rr] :11-85; pesi :43-51; al massimo 100 iterazioni, arresto se |errore - precedente| < 1e-6, la
trasformazione dell'ultima iterazione NON si applica, :69-70); poi nICP che deforma la regione del RIFERIMENTO verso i
suoi punti piu' vicini su A ([Re] :43-81; [Rn] Amberg con rigidezza su trasformazioni affini per vertice, keypoint
pesati 50 con decadimento) e nuove corrispondenze (:88); errore della regione = media di ||corrispondenza aggiornata su
A - vertice ORIGINALE del riferimento|| (:92-94), riportato in mm con la scala metrica del riferimento ([R] :259);
totale = media delle 4 regioni ([R] :85-97). Il sistema di [Rn] si risolve con le equazioni normali come
``cholesky_AAt`` di CHOLMOD, qui con ``scipy.sparse.linalg.splu`` (stessa soluzione).

Convenzioni: vertici per riga; similarita' ``X' = s X @ Q + c`` (Q ortogonale, anche con riflessione). La superficie di
A si costruisce UNA volta nelle coordinate di A (``Surf``); ogni allineamento si applica ai punti di B con l'inversa:
la distanza di p dalla superficie di (s A Q + c) e' s volte la distanza di (p - c) Q^T / s dalla superficie di A.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------------------- superficie


class Surf:
    """Punto piu' vicino sulla superficie (open3d RaycastingScene, Embree su CPU, float32 interno; il punto si
    ricalcola in float64 dalle baricentriche, come ``ugt.Surface``). ``nthreads`` = 1: si parallelizza per processo."""

    def __init__(self, V: np.ndarray, F: np.ndarray, nthreads: int = 1):
        import open3d as o3d
        self.V = np.asarray(V, dtype=np.float64)
        self.F = np.asarray(F, dtype=np.int64)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(self.V.astype(np.float32)), o3d.core.Tensor(self.F.astype(np.uint32)))
        self.nthreads = nthreads
        self._o3d = o3d

    def closest(self, P: np.ndarray) -> np.ndarray:
        """Punti piu' vicini (n, 3), float64."""
        r = self.scene.compute_closest_points(self._o3d.core.Tensor(np.asarray(P, dtype=np.float32)),
                                              nthreads=self.nthreads)
        tri = r["primitive_ids"].numpy().astype(np.int64)
        uv = r["primitive_uvs"].numpy().astype(np.float64)
        b = np.clip(np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=1), 0.0, 1.0)
        b /= b.sum(1, keepdims=True)
        return np.einsum("nkd,nk->nd", self.V[self.F[tri]], b)


def closest_sim(surf: Surf, P: np.ndarray, s: float, Q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Punti piu' vicini a P sulla superficie di (s V Q + c), V = vertici di ``surf``."""
    return s * surf.closest((P - c) @ Q.T / s) @ Q + c


# ------------------------------------------------------------------------------------- Procrustes [N]

def procrustes(X: np.ndarray, Y: np.ndarray, scaling: bool = True) -> dict:
    """Port di [N] procrustes :173-285 (MATLAB ``procrustes``, via fg2018) con ``reflection='best'``: la condizione
    ``reflection is not 'best'`` (:248) e' falsa, quindi la soluzione puo' includere una riflessione. Trasforma Y
    verso X: Z = b Y T + c. Ritorna ``scale`` b, ``rotation`` T, ``translation`` c, ``d``, ``reflection``."""
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    muX, muY = X.mean(0), Y.mean(0)
    X0, Y0 = X - muX, Y - muY
    ssX, ssY = (X0 ** 2).sum(), (Y0 ** 2).sum()
    normX, normY = np.sqrt(ssX), np.sqrt(ssY)
    X0, Y0 = X0 / normX, Y0 / normY
    U, s, Vt = np.linalg.svd(X0.T @ Y0, full_matrices=False)
    T = Vt.T @ U.T
    traceTA = s.sum()
    if scaling:
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
    else:
        b = 1.0
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
    c = muX - b * muY @ T
    return {"scale": float(b), "rotation": T, "translation": c, "d": float(d),
            "reflection": bool(np.linalg.det(T) < 0)}


def now_mask(L7: np.ndarray, factor: float) -> tuple[np.ndarray, float]:
    """[N] compute_mask :56-72 (``factor`` 1.4) / [F] :38-46 (``factor`` 1.2): centro e raggio sui 7 landmark."""
    nose_bottom = L7[4]
    nose_bridge = (L7[1] + L7[2]) / 2
    centre = nose_bottom + 0.3 * (nose_bridge - nose_bottom)
    outer_eye_dist = np.linalg.norm(L7[0] - L7[3])
    nose_dist = np.linalg.norm(nose_bridge - nose_bottom)
    return centre, float(factor * (outer_eye_dist + nose_dist) / 2)


def crop(V: np.ndarray, L7: np.ndarray, factor: float) -> np.ndarray:
    """Vertici di V nel cerchio (``dist <= raggio``, [N] :87-88, [F] :52-53)."""
    c, r = now_mask(L7, factor)
    return V[np.linalg.norm(V - c, axis=1) <= r]


# --------------------------------------------------------------------------------- Rodrigues e dogleg

def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rodrigues(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R(r) e dR/dr_k (3, 3, 3): chumpy ``Rodrigues`` usa cv2.Rodrigues e il suo jacobiano; qui la formula chiusa di
    Gallego e Yezzi (2015): dR/dr_k = (r_k [r]x + [r x (I - R) e_k]x) R / |r|^2, e [e_k]x in r = 0."""
    th = float(np.linalg.norm(r))
    E = np.eye(3)
    if th < 1e-12:
        return E + _skew(r), np.stack([_skew(E[k]) for k in range(3)])
    k = r / th
    K = _skew(k)
    R = np.cos(th) * E + (1 - np.cos(th)) * np.outer(k, k) + np.sin(th) * K
    Kr = _skew(r)
    dR = np.stack([(r[i] * Kr + _skew(np.cross(r, (E - R) @ E[i]))) @ R / th ** 2 for i in range(3)])
    return R, dR


def dogleg(fun, jac, x0: np.ndarray, delta_0: float = 0.1, e_1: float = 1e-15, e_2: float = 1e-15,
           e_3: float = 1e-4, maxiter: int = 200) -> tuple[np.ndarray, dict]:
    """Port di [CH] ``minimize_dogleg`` :358-455 con ``DoglegState`` :158-356 (Powell, Lourakis 2005): passo di
    Cauchy, Gauss-Newton (sistema denso 7 x 7: ``np.linalg.solve``; NoW passa un ``cg`` con 2000 iterazioni, che
    su 7 incognite da' la stessa soluzione), interpolazione ``beta_multiplier``, rapporto rho, raggio x2.5 sopra 0.9 e
    x0.25 sotto 0.05, arresto per gradiente (e_1), passo (e_2), miglioramento relativo (e_3) o ``maxiter``.
    ``fun(x)`` -> stato con ``.r`` (residui); ``jac(stato, x)`` -> J (n, p)."""
    x = np.asarray(x0, np.float64).copy()
    ev = fun(x)
    fevals, it = 1, 0
    J, r = jac(ev, x), ev["r"]
    A, g = J.T @ J, J.T @ (-r)
    delta = delta_0
    done = bool(np.linalg.norm(g, np.inf) < e_1)
    why = "gradiente" if done else ""
    while not done:
        it += 1
        Jg = J @ g
        d_sd = (g @ g) / (Jg @ Jg) * g
        d_gn = None
        while True:
            if delta is not None and np.linalg.norm(d_sd) >= delta:
                d_dl = delta / np.linalg.norm(d_sd) * d_sd
            else:
                if d_gn is None:
                    try:
                        d_gn = np.linalg.solve(A, g)
                    except np.linalg.LinAlgError:
                        d_gn = np.linalg.lstsq(A, g, rcond=None)[0]
                if delta is None or np.linalg.norm(d_gn) <= delta:
                    d_dl = d_gn
                    if delta is None:
                        delta = np.linalg.norm(d_gn)
                else:
                    diff = d_gn - d_sd
                    sq_sd = d_sd @ d_sd
                    pnow = (diff @ diff) * delta ** 2 + (d_gn @ d_sd) ** 2 - (d_gn @ d_gn) * sq_sd
                    beta = (delta ** 2 - sq_sd) / (diff @ d_sd + np.sqrt(pnow))
                    d_dl = d_sd + beta * diff
            improved = False
            if np.linalg.norm(d_dl) <= e_2 * np.linalg.norm(x):
                done, why = True, "passo"
            else:
                ev_new = fun(x + d_dl)
                fevals += 1
                sse, sse_new = r @ r, ev_new["r"] @ ev_new["r"]
                rho = sse - sse_new
                if rho > 0:
                    with np.errstate(all="ignore"):
                        rho /= 2.0 * (g @ d_dl) - d_dl @ (A @ d_dl)
                improved = bool(rho > 0)
                if improved:
                    x = x + d_dl
                    if e_3 > 0 and (sse - sse_new) / sse < e_3:
                        done, why = True, "miglioramento"
                    else:
                        J, r = jac(ev_new, x), ev_new["r"]
                        A, g = J.T @ J, J.T @ (-r)
                        if np.linalg.norm(g, np.inf) < e_1:
                            done, why = True, "gradiente"
                if rho > 0.9:
                    delta = max(delta, 2.5 * np.linalg.norm(d_dl))
                elif rho < 0.05:
                    delta *= 0.25
                if delta <= e_2 * np.linalg.norm(x):
                    done, why = True, "raggio"
            if done or improved:
                break
        if not done and it >= maxiter:
            done, why = True, "maxiter"
    return x, {"iterations": it, "fevals": fevals, "stop": why}


# ------------------------------------------------------------------------------------------ NoW [N]

def now_icp(surf: Surf, P: np.ndarray, b: float, T: np.ndarray, c: np.ndarray, with_scale: bool) -> tuple:
    """[N] rigid_scan_2_mesh_alignment :23-54 sulla mesh A' = b A T + c (A = vertici di ``surf``): A'' = s R A' + t
    (:33), minimizza la somma dei quadrati di [distanze dei punti P dalla superficie di A'', 100 (|s| - s)]. Ritorna la
    similarita' complessiva (sigma, Q, tau) con A'' = sigma A Q + tau e le informazioni del dogleg."""

    def unpack(x):
        if with_scale:
            return x[0], x[1:4], x[4:7]
        return 1.0, x[0:3], x[3:6]

    def fun(x):
        s, rv, t = unpack(x)
        R, dR = rodrigues(rv)
        sig, Q, tau = s * b, T @ R.T, s * c @ R.T + t
        qA = surf.closest((P - tau) @ Q.T / sig)               # punto piu' vicino nelle coordinate di A
        a1 = b * qA @ T + c                                     # lo stesso punto su A' (baricentriche fisse)
        q = s * a1 @ R.T + t
        dvec = P - q
        d = np.linalg.norm(dvec, axis=1)
        res = d if not with_scale else np.r_[d, 100.0 * (abs(s) - s)]
        return {"r": res, "a1": a1, "dvec": dvec, "d": d, "R": R, "dR": dR, "s": s}

    def jac(ev, x):
        d = ev["d"]
        u = np.where(d[:, None] > 1e-12, -ev["dvec"] / np.maximum(d, 1e-300)[:, None], 0.0)   # d d / d q
        a1, R, dR, s = ev["a1"], ev["R"], ev["dR"], ev["s"]
        cols = []
        if with_scale:
            cols.append((u * (a1 @ R.T)).sum(1))
        cols += [s * (u * (a1 @ dR[k].T)).sum(1) for k in range(3)]
        cols += [u[:, k] for k in range(3)]
        J = np.stack(cols, axis=1)
        if with_scale:
            row = np.zeros((1, J.shape[1]))
            row[0, 0] = 100.0 * (np.sign(s) - 1.0)
            J = np.vstack([J, row])
        return J

    x0 = np.r_[1.0, np.zeros(6)] if with_scale else np.zeros(6)
    x, info = dogleg(fun, jac, x0, delta_0=0.1, e_3=1e-4)
    s, rv, t = unpack(x)
    R, _ = rodrigues(rv)
    info["s_icp"] = float(s)
    return (s * b, T @ R.T, s * c @ R.T + t), info


def scan_to_mesh(surf: Surf, P: np.ndarray, sig: float, Q: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Distanze dei punti P dalla superficie di (sig A Q + tau): [S] ScanToMesh, ``signed=False, normalize=False``."""
    return np.linalg.norm(P - closest_sim(surf, P, sig, Q, tau), axis=1)


def now_pair(surf: Surf, LA: np.ndarray, B_crop: np.ndarray, LB: np.ndarray, metrical: bool,
             P_opt: np.ndarray | None = None) -> dict:
    """Errore NoW (originale o metrical) di A rispetto al riferimento B gia' ritagliato (``B_crop``). ``P_opt``:
    sottoinsieme dei punti di ``B_crop`` per le iterazioni dell'ICP (None = tutti, come [N]); l'errore finale usa
    sempre tutti i punti."""
    tf = procrustes(LB, LA, scaling=True)
    T, c = tf["rotation"], tf["translation"]
    b = 1.0 if metrical else tf["scale"]              # [Nm] :125: fattore d'unita' (mm -> 1) al posto della scala
    (sig, Q, tau), info = now_icp(surf, B_crop if P_opt is None else P_opt, b, T, c, with_scale=not metrical)
    d = scan_to_mesh(surf, B_crop, sig, Q, tau)
    return {"mean": float(d.mean()), "median": float(np.median(d)), "scale": float(sig),
            "procrustes_scale": tf["scale"], "reflection": tf["reflection"], "n_ref": int(len(B_crop)), **info}


# ------------------------------------------------------------------------------------- Stirling [F]

def stirling_pair(surf: Surf, LA: np.ndarray, B_crop: np.ndarray, LB: np.ndarray) -> dict:
    """[F] :25-62: Procrustes a 7 punti con scala, distanze dei vertici del riferimento ritagliato (raggio 1.2) dalla
    superficie di A allineata; RMSE."""
    tf = procrustes(LB, LA, scaling=True)
    d = scan_to_mesh(surf, B_crop, tf["scale"], tf["rotation"], tf["translation"])
    return {"rmse": float(np.sqrt((d ** 2).mean())), "mean": float(d.mean()), "scale": tf["scale"],
            "reflection": tf["reflection"], "n_ref": int(len(B_crop))}


# ---------------------------------------------------------------------------------------- REALY [R]

# [Ru] keypoints_region_map :30-52
KP_REGION = {"forehead": [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
             "mouth": [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 64],
             "nose": [27, 28, 29, 30, 31, 32, 33, 34, 35],
             "nose_part": [27, 28, 29, 30, 31, 35, 32, 33, 34, 70, 71],
             "cheek": [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 76, 77, 36, 41, 40, 39, 42, 47, 46, 45,
                       7, 8, 9],
             "seven_keypoints": [36, 39, 42, 45, 33, 48, 54]}
RIGID_KP = {"forehead": "forehead", "mouth": "mouth", "nose": "nose_part", "cheek": "cheek"}   # [Rr] :123-124
EVAL_KP = {"forehead": "forehead", "mouth": "mouth", "nose": "nose", "cheek": None}           # [R] :220-225


def fit_icp_RT(source: np.ndarray, target: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """[Ru] fit_icp_RT :76-109 (vertici per riga qui, per colonna li'): rotazione U V dalla SVD di
    W = t1 s1^T SENZA controllo della riflessione, scala = sum|t1| / sum|R s1| (norme L1), traslazione = media di
    (target - scale R source). Ritorna (scale, R, t) con target ~ scale source @ R.T + t."""
    s1 = source - source.mean(0)
    t1 = target - target.mean(0)
    U, _, Vt = np.linalg.svd(t1.T @ s1)
    R = U @ Vt
    scale = float(np.abs(t1).sum() / np.abs(s1 @ R.T).sum()) if with_scale else 1.0
    t = (target - scale * source @ R.T).mean(0)
    return scale, R, t


def edge_incidence(F: np.ndarray, n: int) -> sp.csc_matrix:
    """[Rn] triangles_to_edge_vertex_adjacent_matrix :15-49: una riga per spigolo non orientato, +1 / -1."""
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [0, 2]], F[:, [1, 2]]]), axis=1), axis=0)
    k = np.arange(len(E))
    return sp.csc_matrix((np.r_[np.ones(len(E)), -np.ones(len(E))], (np.r_[k, k], np.r_[E[:, 0], E[:, 1]])),
                         shape=(len(E), n))


def _D(X: np.ndarray) -> sp.csc_matrix:
    """[Rn] sparse_matrix_from_vertices :52-66: riga i = [x y z 1] nelle colonne 4i..4i+3."""
    n = len(X)
    return sp.csc_matrix((np.c_[X, np.ones(n)].ravel(), (np.repeat(np.arange(n), 4), np.arange(4 * n))),
                         shape=(n, 4 * n))


def _lsq(A: sp.spmatrix, B: np.ndarray) -> np.ndarray:
    """min ||A X - B||: [Rn] spsolve :10-12 (CHOLMOD ``cholesky_AAt(A.T)`` = Cholesky di A^T A) con splu."""
    AtA = (A.T @ A).tocsc()
    return spl.splu(AtA, permc_spec="COLAMD").solve(np.asarray(A.T @ B))


def nicp(src: np.ndarray, dst: np.ndarray, F: np.ndarray, kp_idx=None, kp_tgt=None, max_steps: int = 200) -> tuple:
    """[Rn] nICP_with_keypoints :153-265 (``kp_idx`` dato) / nICP_without_keypoints :82-150, con i parametri di [Re]
    :32-41 (epsilon 1, gamma 1, alpha 50, alpha2 150, kp_w1 50). Deforma ``src`` (vertici del riferimento) verso i
    punti ``dst`` (nn su KD-tree). Senza keypoint il codice ufficiale non ha tetto di passi per decadimento: qui
    ``max_steps`` (contato in ``hit_cap``)."""
    eps, gamma, alpha, alpha2, kp_w1 = 1.0, 1.0, 50.0, 150.0, 50.0
    n = len(src)
    M = edge_incidence(F, n)
    A1 = sp.kron(M, sp.diags([1.0, 1.0, 1.0, gamma])).tocsr()
    tree = cKDTree(dst)
    cur = np.asarray(src, np.float64)
    cur_X, X = np.zeros((4 * n, 3)), np.ones((4 * n, 3))
    zeros = np.zeros((A1.shape[0], 3))
    solves, hit_cap = 0, 0
    if kp_idx is None:
        for decay in range(3):
            a = alpha * np.exp(-0.5 * decay)
            e = eps * 0.5 ** decay
            step = 0
            while np.linalg.norm(X - cur_X) > e:
                if step >= max_steps:
                    hit_cap += 1
                    break
                cur_X = X.copy()
                nd, ni = tree.query(cur)
                w = (nd < max(nd.mean() * 2, 1)).astype(np.float64)
                D = _D(cur)
                A = sp.vstack([A1 * a, sp.diags(w) @ D]).tocsc()
                X = _lsq(A, np.vstack([zeros, dst[ni] * w[:, None]]))
                cur = D @ X
                step += 1
                solves += 1
        return cur, {"solves": solves, "hit_cap": hit_cap}
    A1_KP = sp.kron(M, sp.diags([2.0, 2.0, 3.0, gamma * 2])).tocsr()
    kp_idx = np.asarray(kp_idx)
    count = 0
    for decay in range(4):
        a = alpha2 if decay == 0 else alpha * np.exp(-0.5 * (decay - 1))
        e = eps * 0.5 ** (decay - 1)
        step = 0
        while True:
            if np.linalg.norm(X - cur_X) <= e or step > 10:
                break
            cur_X = X.copy()
            nd, ni = tree.query(cur)
            D = _D(cur)
            w = (nd < max(nd.mean() * 2, 1)).astype(np.float64)
            kw = np.zeros(n)
            kw[kp_idx] = kp_w1 if decay == 0 else kp_w1 * np.exp(-0.9 * count) / 10.0
            tgt = dst[ni].copy()
            tgt[kp_idx] = kp_tgt
            A3, B3 = sp.diags(kw) @ D, tgt * kw[:, None]
            if decay == 0:
                A, B = sp.vstack([A1_KP * a, A3]), np.vstack([zeros, B3])
            else:
                A, B = sp.vstack([A1 * a, sp.diags(w) @ D, A3]), np.vstack([zeros, dst[ni] * w[:, None], B3])
            X = _lsq(A.tocsc(), B)
            cur = D @ X
            step += 1
            count += 1
            solves += 1
            if decay == 0:
                break
    return cur, {"solves": solves, "hit_cap": hit_cap}


def realy_pair(surf: Surf, KA: np.ndarray, ref: dict, max_iteration: int = 100) -> dict:
    """Errore REALY di A (vertici di ``surf`` in mm, 85 keypoint ``KA`` in mm) rispetto al riferimento ``ref``
    (``pr_run.realy_reference``: 85 keypoint ``K`` e regioni ``regions`` {nome: (V, F)} nel frame del template di
    REALY, in unita' di REALY; ``metrical`` = mm per unita'). Ritorna l'errore per regione e il totale in mm."""
    K, regions, m = ref["K"], ref["regions"], ref["metrical"]
    seven = KP_REGION["seven_keypoints"]
    # gICP ([Rg] :22-27): A in unita' di REALY sul riferimento
    g, Rg, tg = fit_icp_RT(KA[seven], K[seven], with_scale=True)
    out = {"gicp_scale": g}
    errs = {}
    for reg in ("forehead", "mouth", "nose", "cheek"):                       # ordine di [Rr] :123
        if reg not in regions:
            errs[reg] = np.nan
            continue
        P, Fr = regions[reg]
        # rICP ([Rr] :11-85), similarita' corrente di A: X -> sig X Q + tau (Q = Rg^T poi rigide)
        sig, Q, tau = g, Rg.T, tg
        kidx = KP_REGION[RIGID_KP[reg]]
        rate = len(P) / len(kidx) / (4 if reg == "cheek" else 1)
        kp_pred = np.vstack([(sig * KA[kidx] @ Q + tau)] * int(rate)) if int(rate) > 0 else np.zeros((0, 3))
        kp_gt = np.vstack([K[kidx]] * int(rate)) if int(rate) > 0 else np.zeros((0, 3))
        err_pre, it = 0.0, 0
        while True:
            corr = closest_sim(surf, P, sig, Q, tau)
            cp, cg = np.vstack([corr, kp_pred]), np.vstack([P, kp_gt])
            _, R, t = fit_icp_RT(cp, cg, with_scale=False)
            err = float((((cp @ R.T + t) - cg) ** 2).sum(1).mean())
            if abs(err - err_pre) < 1e-6 or it >= max_iteration:
                break
            it += 1
            err_pre = err
            Q, tau = Q @ R.T, tau @ R.T + t
            kp_pred = kp_pred @ R.T + t
        # bICP ([Re] :43-94) sulla A allineata per regione
        corr = closest_sim(surf, P, sig, Q, tau)
        ek = EVAL_KP[reg]
        if ek is None:
            deformed, ninfo = nicp(P, corr, Fr)
        else:
            kidx_scan = cKDTree(P).query(K[KP_REGION[ek]])[1]
            kp_tgt = sig * KA[KP_REGION[ek]] @ Q + tau
            deformed, ninfo = nicp(P, corr, Fr, kidx_scan, kp_tgt)
        corr2 = closest_sim(surf, deformed, sig, Q, tau)
        errs[reg] = float(np.linalg.norm(corr2 - P, axis=1).mean()) * m
        out[f"{reg}_ricp_iter"] = it
        out[f"{reg}_solves"] = ninfo["solves"]
        out[f"{reg}_hit_cap"] = ninfo["hit_cap"]
        out[f"{reg}_n"] = int(len(P))
    out.update({f"err_{k}": v for k, v in errs.items()})
    out["total"] = float(np.mean([errs[r] for r in ("nose", "mouth", "forehead", "cheek")]))
    return out
