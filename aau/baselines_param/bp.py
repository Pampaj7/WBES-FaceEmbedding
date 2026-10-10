"""Concorrenti parametrici (fit di un 3DMM e distanza d'identita') e distanza varifold in mm: libreria.

Protocollo: ``aau/runs/evidence/baselines_param/PROTOCOL.md`` (scritto prima dei numeri). Infrastruttura di
``aau/baselines_mm`` importata, non copiata: viste, mm nel frame canonico (``blmm.to_mm``), unita' di lavoro L_d
(``blmm.params``), semi per mesh (``blmm.mesh_seed``), iscrizione ICP di similarita' + NICP faceBench
(``ir_simicp.similarity_icp``, ``facebench.nonrigid_icp_align``, come ``blmm.enroll``), ``view_root``.

Modelli (``v3_work/mm/loaders.py``, regione del volto, basi d'identita' a +1 sigma: il prior e' N(0, 1)):
  - ``gnm``: GNM Head v3.0 (Apache-2.0), hockey_mask, 170 identita', pool d'espressione di 350 (parte bassa del
    volto e occhi). E' fra i generatori del training (C3F: BFM, GNM, ICT): PRIOR VISTO.
  - ``flame2023``: FLAME 2023 Open (CC BY 4.0), maschera face, 300 identita', 100 espressioni, piu' il pitch della
    mandibola (LBS col giunto 2, correttivi di posa, giunto regredito dalla forma). Prior NON visto.

Per (vista, modello):
  1. regione (``region``): i vertici della regione del modello che cadono DENTRO la superficie del dominio. Per ogni
     mesh di riferimento (vista a 6 topologie: la media vertice per vertice delle original dei 100 soggetti NON
     valutati del template, ``blmm.template_subjects``; FaMoS: le 15 scansioni di galleria) la media del modello,
     nel frame canonico (``frames.json``), si porta sul riferimento con baricentri + ICP rigido (soglia infinita,
     poi 10 mm); un vertice vota si' se il vertice piu' vicino del riferimento dista meno di ``REGION_DIST_MM`` e
     non e' sul bordo ne' a un anello dal bordo. Tenuti i vertici col voto >= ``REGION_VOTE``, poi i triangoli
     coi tre vertici tenuti e la componente connessa piu' grande.
  2. iscrizione (``enroll_model``): template = media del modello sulla regione (al piu' 4096 vertici ``rng(0)``,
     come ``blmm.build_template``), nelle unita' di lavoro del modo ``mm`` (centrato, / L_d); 4096 punti della mesh
     col seme ``blmm.mesh_seed``; ICP di similarita' + NICP del template sulla mesh; uscita x L_d = i vertici del
     template registrati sulla mesh, in mm.
  3. regressione (``fit_points``): MAP con rumore isotropo ``SIGMA_NOISE_MM`` e prior N(0, 1) su identita' ed
     espressione (espressione libera anche sulle viste neutre), alternata con la rigida (Umeyama senza scala):
     min sum_v ||R x_v(c, theta) + t - y_v||^2 + SIGMA^2 ||c||^2. Nessuna scala per mesh: la taglia resta
     nell'identita'. FLAME: theta (pitch, rad) in ``JAW_BOUNDS`` per ricerca scalare con c risolto in forma chiusa.
  4. fallimento: eccezione, valori non finiti o RMS finale > ``FAIL_RMS_MM`` -> coefficienti NaN (righe escluse).

Distanze: ``coef`` = ||beta_i - beta_j|| (euclidea sui coefficienti a +1 sigma, cioe' scalati per autovalore);
``fr`` / ``sr`` = sulle mesh d'identita' ricostruite mu + B beta (espressione e mandibola a zero) sulla regione, in
mm, pesi d'area di mu: rigida ai minimi quadrati pesata verso mu, sqrt(sum_v w_v ||a_i - a_j||^2 / sum w) (FR);
SR = lo stesso dopo la scala a centroid size di mu attorno al baricentro (come ``cgt.Canon.shape_S``).

Varifold (``varifold_measure``): il kernel di ``v2_work/phase0/measure_distances.py`` nella versione a blocchi di
``aau/baselines/geometric_kernel.py`` (gaussiana sui centroidi x (n_i . n_j)^2, somma su ``VARIFOLD_SIGMAS_MM``),
misura in mm (``blmm.to_mm``) centrata sul baricentro pesato per area, nessuna rotazione ne' scala per mesh, aree
divise per l'area totale (massa unitaria), quantizzata su una griglia di lato ``VARIFOLD_CELL_MM``
(``geometric_matrix.quantize_measure``, deterministica).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
for _p in (AAU_DIR / "baselines_mm", REPO_ROOT, REPO_ROOT / "v2_work" / "genflame", REPO_ROOT / "v2_work" / "genict"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import blmm  # noqa: E402  (mette faceBench, zs3dmm, indomain su sys.path)

OUT_ROOT = AAU_DIR / "runs" / "evidence" / "baselines_param"
VIEWS = ("hifi3d", "facescape", "faceverse", "faceverse_neutral", "famos")
TOPOLOGIES = ("original", "remesh", "noisy", "down8k", "up60k")   # senza crop: le righe di fact_paired
# nome -> (loader di v3_work/mm, voce di frames.json, prior visto nel training, mandibola articolata)
MODELS = {"gnm": ("gnm", "gnm", True, False), "flame2023": ("flame2023", "flame", False, True)}
N_POINTS = blmm.N_POINTS                 # 4096: template e campione della mesh, come blmm
SIGMA_NOISE_MM = 1.0                     # rumore della regressione: lambda = SIGMA^2 col prior N(0, 1)
FIT_ITERS, FIT_TOL_MM = 30, 1e-4         # alternanza rigida / coefficienti, arresto sulla variazione dell'RMS
JAW_BOUNDS = (-0.1, 0.6)                 # pitch della mandibola FLAME (rad), giunto 2, asse x nativo
FAIL_RMS_MM = 5.0                        # RMS finale oltre cui il fit e' fallito
REGION_DIST_MM, REGION_VOTE = 10.0, 0.5
VARIFOLD_SIGMAS_MM = (10.0, 20.0, 40.0)
VARIFOLD_CELL_MM = 2.5                   # un quarto della sigma piu' piccola (geometric_matrix.calibrate_cell)
FLAME_JAW = 2                            # indice del giunto nella kintree di FLAME (root, neck, jaw, occhi)
# emendamento 1 (PROTOCOL_emendamento_1.md, POST HOC): varianti A (``va``) e B (``vb``), errore di superficie
NEUTRAL_VIEWS = ("hifi3d", "facescape", "faceverse_neutral", "famos")   # A: niente espressione, mandibola a 0
VARIANTS = ("va", "vb")
LOOP_INNER = 5                           # giri interni (rigida / coefficienti) per iterazione esterna di B
LOOP_MIN_KEPT = 0.25                     # frazione minima di corrispondenze tenute per verso (B)
LOOP_GRID = {"sigma": (0.5, 1.0, 2.0), "tau": (2.0, 5.0, 10.0), "iters": (5, 10)}   # pilota (sez. 2)
PILOT_VIEWS, PILOT_SUBJECTS = ("hifi3d", "facescape", "faceverse", "faceverse_neutral"), 10
# {sigma, tau, iters} di B congelati dal pilota (job 1067652, 1067653; sez. 8 dell'emendamento, prima delle viste valutate)
LOOP = {"gnm": {"sigma": 2.0, "tau": 10.0, "iters": 5}, "flame2023": {"sigma": 2.0, "tau": 5.0, "iters": 5}}


# ------------------------------------------------------------------------------------------ modelli

def load_model(name: str) -> dict:
    """Il modello ``name`` in mm, frame nativo: ``mu`` (n, 3), ``B`` (n, 3, k), ``E`` (n, 3, e) (solo il pool),
    ``F``, ``k_id``, ``expr_scale`` (sigma dichiarata dei coefficienti d'espressione, ``loaders.EXPR_SCALE``: la usa
    solo l'emendamento 1); FLAME anche ``jaw`` (pesi, correttivi, giunto lineare nei coefficienti)."""
    from v3_work.mm import loaders
    loader, _, seen, jaw = MODELS[name]
    m = loaders.LOADERS[loader](loader)
    u = float(m.frame.mm_per_unit)
    pool = np.asarray(m.expr.pool, dtype=np.int64)
    out = {"name": name, "mu": m.mean * u, "B": m.id_basis * u, "E": m.expr.basis[:, :, pool] * u,
           "F": np.asarray(m.faces, dtype=np.int64), "k_id": int(m.id_basis.shape[2]), "seen": seen,
           "file": str(loaders.model_path(loader)), "region": m.region, "expr_scale": float(m.expr.scale)}
    if jaw:
        out["jaw"] = flame_jaw(loaders.model_path(loader), np.asarray(m.region_vertices), u, out)
    return out


def flame_jaw(path: Path, used: np.ndarray, u: float, m: dict) -> dict:
    """Dati della mandibola FLAME sulla regione: peso di skinning del giunto 2, correttivi di posa dei 9 valori di
    (R_jaw - I) (posti 9..17 del vettore di posa: giunti 1..4), giunto = j0 + Jc c (J_regressor sulla testa intera,
    lineare in [identita', espressione])."""
    import scipy.sparse as sp
    from flame_model import _dechumpy, _Unpickler
    with open(path, "rb") as fh:
        raw = _Unpickler(fh, encoding="latin1").load()
    vt = _dechumpy(raw["v_template"]).astype(np.float64) * u
    sd = _dechumpy(raw["shapedirs"]).astype(np.float64) * u
    if np.abs(vt[used] - m["mu"]).max() > 1e-6 or np.abs(sd[used, :, :m["k_id"]] - m["B"]).max() > 1e-6:
        raise ValueError(f"{path.name}: media o basi diverse da quelle del loader")
    Jr = raw["J_regressor"]
    Jr = Jr.toarray() if sp.issparse(Jr) else np.asarray(_dechumpy(Jr), dtype=np.float64)
    kin = _dechumpy(raw["kintree_table"])
    if int(kin[0, FLAME_JAW]) != 1:
        raise ValueError(f"kintree inattesa: {kin.tolist()}")
    r = Jr[FLAME_JAW]
    C_full = np.concatenate([sd[:, :, :m["k_id"]], sd[:, :, 300:400]], axis=2)       # (5023, 3, 400)
    pd = _dechumpy(raw["posedirs"]).astype(np.float64)[used] * u                     # (n, 3, 36)
    return {"w": _dechumpy(raw["weights"]).astype(np.float64)[used, FLAME_JAW], "P": pd[:, :, 9:18],
            "j0": r @ vt, "Jc": np.einsum("v,vdk->dk", r, C_full)}


def rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def place_canonical(name: str, X: np.ndarray) -> np.ndarray:
    """Punti del modello (mm, frame nativo) nel frame canonico di E12: R_d x + t_d (u_d gia' applicata)."""
    f = json.loads(blmm.FRAMES_JSON.read_text())["domains"][MODELS[name][1]]
    return X @ np.asarray(f["R"]).T + np.asarray(f["t"])


# ------------------------------------------------------------------------------------------ regione

def boundary_ring(F: np.ndarray, n: int) -> np.ndarray:
    """Vertici di bordo (spigoli con un solo triangolo) e il loro primo anello."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    e, c = np.unique(E, axis=0, return_counts=True)
    b = np.zeros(n, dtype=bool)
    b[e[c == 1].ravel()] = True
    ring = b.copy()
    hit = b[E[:, 0]] | b[E[:, 1]]
    ring[E[hit].ravel()] = True
    return ring


def rigid_icp(source: np.ndarray, target: np.ndarray, threshold: float, init: np.ndarray) -> np.ndarray:
    """Trasformazione 4x4 dell'ICP rigido punto-punto di open3d (senza scala)."""
    import open3d as o3d
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target)
    return np.asarray(o3d.pipelines.registration.registration_icp(
        src, tgt, threshold, init, o3d.pipelines.registration.TransformationEstimationPointToPoint()).transformation)


def references(view: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Mesh di riferimento della regione, in mm nel frame canonico: (nome, V, F)."""
    if view == "famos":
        rows = [r for r in blmm.famos_manifest() if r["kind"] == "scan" and r["role"] == "gallery"]
        out = []
        for r in rows:
            V, F = blmm.load_raw(blmm.VIEWS["famos"]["dir"] / f"{r['name']}.npz")
            out.append((r["name"], blmm.to_mm("famos", V, r["name"]), F))
        return out
    Vs, F = [], None
    for s in blmm.template_subjects(view):
        V, F = blmm.load_raw(blmm.mesh_path(view, s, "original", template=True))
        Vs.append(blmm.to_mm(view, V))
    return [(f"media di {len(Vs)} original del template", np.mean(np.stack(Vs), axis=0), F)]


def region(view: str, m: dict) -> dict:
    """Vertici della regione del modello dentro il dominio (indici nella regione del modello), voti, diagnostica."""
    import mesh_ops as mo
    from scipy.spatial import cKDTree
    M = place_canonical(m["name"], m["mu"])
    refs = references(view)
    votes = np.zeros(len(M))
    diag = []
    for name, V, F in refs:
        T = np.eye(4)
        T[:3, 3] = V.mean(0) - M.mean(0)
        T = rigid_icp(M, V, 1000.0, T)
        T = rigid_icp(M, V, REGION_DIST_MM, T)
        A = M @ T[:3, :3].T + T[:3, 3]
        d, i = cKDTree(V).query(A)
        ok = (d < REGION_DIST_MM) & ~boundary_ring(F, len(V))[i]
        votes += ok
        diag.append({"reference": name, "kept": float(ok.mean()), "median_mm": float(np.median(d[ok]))})
    keep = votes >= REGION_VOTE * len(refs)
    Fk = mo.largest_component(m["F"][keep[m["F"]].all(axis=1)])
    used = np.unique(Fk)
    return {"vertices": used, "votes": votes, "n_refs": len(refs), "diag": diag}


# --------------------------------------------------------------------------------------- iscrizione

def compact_faces(F: np.ndarray, used: np.ndarray, n: int) -> np.ndarray:
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(len(used))
    Fr = remap[F]
    return Fr[(Fr >= 0).all(axis=1)]


def context(view: str, name: str, reg_vertices: np.ndarray, prm: dict | None = None) -> dict:
    """Tutto cio' che serve al fit di una (vista, modello): template di lavoro, basi sui vertici del campione e
    della regione, pesi d'area di mu, mandibola."""
    m = load_model(name)
    prm = prm or blmm.params()
    L = float(prm["domains"][blmm.VIEWS[view]["domain"]]["L"])
    used = np.asarray(reg_vertices, dtype=np.int64)
    sub = np.arange(len(used)) if len(used) <= N_POINTS else \
        np.sort(np.random.default_rng(0).choice(len(used), N_POINTS, replace=False))
    vs = used[sub]                                   # vertici del modello del template
    C = np.concatenate([m["B"], m["E"]], axis=2)     # (n, 3, K): [identita', espressione]
    T = m["mu"][vs]
    ctx = {"view": view, "model": name, "L": L, "used": used, "sub": sub, "k_id": m["k_id"],
           "T_work": (T - T.mean(0)) / L, "mu_s": m["mu"][vs], "C_s": C[vs], "mu_r": m["mu"][used],
           "B_r": m["B"][used], "F_r": compact_faces(m["F"], used, len(m["mu"])), "file": m["file"]}
    ctx["w_r"] = blmm.vertex_areas(ctx["mu_r"], ctx["F_r"])
    Cs = ctx["C_s"].reshape(-1, C.shape[2])
    ctx["CtC"] = Cs.T @ Cs
    # emendamento 1: base completa sulla regione, bordo della regione, sigma dichiarata dell'espressione
    ctx.update(C_r=C[used], expr_scale=m["expr_scale"], bnd_r=boundary_vertices(ctx["F_r"], len(used)))
    if "jaw" in m:
        j = m["jaw"]
        ctx["jaw"] = {"w": j["w"][vs], "P": j["P"][vs], "j0": j["j0"], "Jc": j["Jc"]}
        ctx["jaw_r"] = {"w": j["w"][used], "P": j["P"][used], "j0": j["j0"], "Jc": j["Jc"]}
    return ctx


def enroll_model(X: np.ndarray, seed: int, ctx: dict) -> tuple[np.ndarray, float]:
    """(vertici del template registrati sulla mesh, mm; distanza media dal campione della mesh, mm)."""
    import facebench as fb
    import ir_simicp
    import run_facebench_remesh as rfr
    from scipy.spatial import cKDTree
    Xs = rfr.sample_pts(X, N_POINTS, seed)
    T_nicp = np.asarray(fb.nonrigid_icp_align(ir_simicp.similarity_icp(ctx["T_work"], Xs), Xs), dtype=np.float64)
    return T_nicp * ctx["L"], float(cKDTree(Xs).query(T_nicp)[0].mean() * ctx["L"])


# --------------------------------------------------------------------------------------- regressione

def rigid_fit(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) con R x + t ~ y ai minimi quadrati (Umeyama senza scala)."""
    mx, my = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((Y - my).T @ (X - mx))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, my - R @ mx


def jaw_system(ctx: dict, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """(A (3m, K), b (3m,)) con x(c) = A c + b: LBS del solo giunto della mandibola ruotato di ``theta``,
    x_v = (I + w_v Q)(mu_v + C_v c + P_v q) - w_v Q (j0 + Jc c), Q = R - I, q = Q in riga."""
    j = ctx["jaw"]
    Q = rot_x(theta) - np.eye(3)
    p = j["P"] @ Q.ravel()                                                   # (m, 3)
    C = ctx["C_s"]
    WQ = j["w"][:, None, None] * Q[None]                                     # (m, 3, 3)
    A = C + np.einsum("vab,vbk->vak", WQ, C) - np.einsum("vab,bk->vak", WQ, j["Jc"])
    base = ctx["mu_s"] + p
    b = base + np.einsum("vab,vb->va", WQ, base) - WQ @ j["j0"]
    return A.reshape(-1, C.shape[2]), b.ravel()


def solve_coef(A: np.ndarray, b: np.ndarray, y: np.ndarray, AtA: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """MAP: argmin ||A c + b - y||^2 + SIGMA^2 ||c||^2, e il valore dell'obiettivo."""
    AtA = A.T @ A if AtA is None else AtA
    c = np.linalg.solve(AtA + SIGMA_NOISE_MM ** 2 * np.eye(len(AtA)), A.T @ (y - b))
    r = A @ c + b - y
    return c, float(r @ r + SIGMA_NOISE_MM ** 2 * c @ c)


def fit_points(Y: np.ndarray, ctx: dict) -> dict:
    """Regressione alternata (rigida, coefficienti, mandibola) dei vertici registrati ``Y`` (m, 3) in mm."""
    from scipy.optimize import minimize_scalar
    K = ctx["C_s"].shape[2]
    A0, b0 = ctx["C_s"].reshape(-1, K), ctx["mu_s"].ravel()
    c, theta, prev = np.zeros(K), 0.0, np.inf
    X = ctx["mu_s"]
    for it in range(1, FIT_ITERS + 1):
        R, t = rigid_fit(X, Y)
        y = ((Y - t) @ R).ravel()                                            # Y nel frame del modello
        if "jaw" in ctx:
            def obj(th):
                return solve_coef(*jaw_system(ctx, th), y)[1]
            theta = float(minimize_scalar(obj, bounds=JAW_BOUNDS, method="bounded", options={"xatol": 1e-4}).x)
            A, b = jaw_system(ctx, theta)
            c, _ = solve_coef(A, b, y)
        else:
            A, b = A0, b0
            c, _ = solve_coef(A, b, y, ctx["CtC"])
        X = (A @ c + b).reshape(-1, 3)
        rms = float(np.sqrt((((X @ R.T + t) - Y) ** 2).sum(1).mean()))
        if abs(prev - rms) < FIT_TOL_MM:
            break
        prev = rms
    R, t = rigid_fit(X, Y)
    rms = float(np.sqrt((((X @ R.T + t) - Y) ** 2).sum(1).mean()))
    return {"beta": c[:ctx["k_id"]], "psi": c[ctx["k_id"]:], "theta": theta, "R": R, "t": t, "rms": rms,
            "iters": it}


# ------------------------------------------------------------------------------------------ distanze

def identity_meshes(beta: np.ndarray, ctx: dict) -> np.ndarray:
    """(n, nr, 3) mesh d'identita' mu + B beta sulla regione, mm (righe NaN restano NaN)."""
    return ctx["mu_r"][None] + np.einsum("vdk,nk->nvd", ctx["B_r"], beta)


def weighted_rigid_to(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """X (n, v, 3) portate su Y (v, 3) con la rigida pesata (``cgt.rigid_fit``, pesi w)."""
    wn = w / w.sum()
    my = wn @ Y
    out = np.full_like(X, np.nan)
    for k, x in enumerate(X):
        if not np.isfinite(x).all():
            continue
        mx = wn @ x
        U, _, Vt = np.linalg.svd((wn[:, None] * (Y - my)).T @ (x - mx))
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        out[k] = (x - mx) @ (U @ D @ Vt).T + my
    return out


def mesh_distances(beta: np.ndarray, ctx: dict) -> dict:
    """{coef, fr, sr}: matrici (n, n) fra le mesh dei coefficienti ``beta`` (n, k); NaN dove il fit e' fallito."""
    w = ctx["w_r"]
    a = weighted_rigid_to(identity_meshes(beta, ctx), ctx["mu_r"], w)
    m = (w / w.sum()) @ ctx["mu_r"]

    def cs(X):
        return np.sqrt(np.einsum("v,nv->n", w / w.sum(), ((X - m) ** 2).sum(-1)))

    s = m + (a - m) * (cs(ctx["mu_r"][None])[0] / cs(a))[:, None, None]

    def euclid(X):
        S = (np.sqrt(w)[None, :, None] * X).reshape(len(X), -1)
        G = (S ** 2).sum(1)[:, None] + (S ** 2).sum(1)[None, :] - 2.0 * S @ S.T
        D = np.sqrt(np.clip(G, 0.0, None) / w.sum())
        np.fill_diagonal(D, 0.0)
        return 0.5 * (D + D.T)

    b = np.asarray(beta, dtype=np.float64)
    Gb = (b ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * b @ b.T
    coef = np.sqrt(np.clip(Gb, 0.0, None))
    np.fill_diagonal(coef, 0.0)
    return {"coef": 0.5 * (coef + coef.T), "fr": euclid(a), "sr": euclid(s)}


# --------------------------------------------------------------------- emendamento 1 (post hoc)

def boundary_vertices(F: np.ndarray, n: int) -> np.ndarray:
    """Maschera dei vertici di bordo (spigoli con un solo triangolo)."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    e, c = np.unique(E, axis=0, return_counts=True)
    b = np.zeros(n, dtype=bool)
    b[e[c == 1].ravel()] = True
    return b


def n_coef(ctx: dict, free_expr: bool) -> int:
    """Coefficienti risolti: solo identita' (A sulle viste neutre) o [identita', espressione]."""
    return ctx["C_r"].shape[2] if free_expr else ctx["k_id"]


def region_system(ctx: dict, theta: float, k: int, idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(A (n, 3, k), b (n, 3)) con x = A c + b sui vertici ``idx`` della regione (tutti se None), primi k
    coefficienti; FLAME: la LBS della mandibola di ``jaw_system`` a ``theta`` (theta = 0: A = C, b = mu)."""
    sel = slice(None) if idx is None else idx
    C, mu = ctx["C_r"][sel, :, :k], ctx["mu_r"][sel]
    if "jaw_r" not in ctx or theta == 0.0:
        return C, mu
    j = ctx["jaw_r"]
    Q = rot_x(theta) - np.eye(3)
    WQ = j["w"][sel][:, None, None] * Q[None]
    A = C + np.einsum("vab,vbk->vak", WQ, C) - np.einsum("vab,bk->vak", WQ, j["Jc"][:, :k])
    base = mu + j["P"][sel] @ Q.ravel()
    return A, base + np.einsum("vab,vb->va", WQ, base) - WQ @ j["j0"]


def prior_diag(ctx: dict, k: int, sigma: float) -> np.ndarray:
    """sigma^2 (||beta||^2 + ||psi / s||^2): s = sigma dichiarata dell'espressione (``loaders.EXPR_SCALE``)."""
    return sigma ** 2 * np.r_[np.ones(ctx["k_id"]), np.full(k - ctx["k_id"], ctx["expr_scale"] ** -2.0)]


def weighted_rigid(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) con R x + t ~ y ai minimi quadrati pesati (Umeyama senza scala; w = 1 da' ``rigid_fit``)."""
    wn = w / w.sum()
    mx, my = wn @ X, wn @ Y
    U, _, Vt = np.linalg.svd((wn[:, None] * (Y - my)).T @ (X - mx))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, my - R @ mx


def fit_corr(S, Y: np.ndarray, w: np.ndarray, ctx: dict, k: int, sigma: float, jaw_free: bool,
             c0: np.ndarray | None = None, theta0: float = 0.0, iters: int = FIT_ITERS) -> dict:
    """Regressione alternata (rigida pesata, coefficienti MAP, mandibola se ``jaw_free``) sulle corrispondenze
    S x(c) ~ Y: S (m, n_r) sparsa (righe one-hot o baricentriche sui vertici della regione), Y (m, 3) mm, pesi w.
    min sum_i w_i ||R S_i x(c, theta) + t - y_i||^2 + c' diag(``prior_diag``) c; arresto come ``fit_points``."""
    import scipy.sparse as sp
    from scipy.optimize import minimize_scalar
    S = sp.csr_matrix(S)
    nz = np.unique(S.indices)                        # solo i vertici toccati entrano nel sistema
    S = S[:, nz]
    G = (S.T @ sp.diags(w) @ S).tocsr()
    lam = prior_diag(ctx, k, sigma)
    c = np.zeros(k) if c0 is None else np.asarray(c0, dtype=np.float64)[:k].copy()
    theta = float(theta0) if jaw_free else 0.0
    cache = {}

    def system(th):
        if th not in cache:
            A, b = region_system(ctx, th, k, nz)
            GA = (G @ A.reshape(len(nz), -1)).reshape(A.shape)
            cache.clear()
            cache[th] = (A, b, A.reshape(-1, k).T @ GA.reshape(-1, k) + np.diag(lam), G @ b)
        return cache[th]

    def solve(th, y):
        A, b, H, Gb = system(th)
        cc = np.linalg.solve(H, A.reshape(-1, k).T @ (S.T @ (w[:, None] * y) - Gb).ravel())
        r = S @ (A @ cc + b) - y
        return cc, float(w @ (r * r).sum(1) + cc @ (lam * cc))

    def points(cc, th):
        A, b = system(th)[:2]
        return S @ (A @ cc + b)

    X, prev = points(c, theta), np.inf
    it = 0
    for it in range(1, iters + 1):
        R, t = weighted_rigid(X, Y, w)
        y = (Y - t) @ R                                                      # Y nel frame del modello
        if jaw_free:
            theta = float(minimize_scalar(lambda th: solve(th, y)[1], bounds=JAW_BOUNDS, method="bounded",
                                          options={"xatol": 1e-4}).x)
        c, _ = solve(theta, y)
        X = points(c, theta)
        rms = float(np.sqrt(w @ (((X @ R.T + t) - Y) ** 2).sum(1) / w.sum()))
        if abs(prev - rms) < FIT_TOL_MM:
            break
        prev = rms
    R, t = weighted_rigid(X, Y, w)
    rms = float(np.sqrt(w @ (((X @ R.T + t) - Y) ** 2).sum(1) / w.sum()))
    return {"c": c, "beta": c[:ctx["k_id"]], "psi": c[ctx["k_id"]:], "theta": theta, "R": R, "t": t, "rms": rms,
            "iters": it}


def fit_registered(Y: np.ndarray, ctx: dict, free_expr: bool, sigma: float = SIGMA_NOISE_MM) -> dict:
    """Variante A sui punti registrati dal NICP (riga v = vertice ``sub[v]`` del template, peso 1): ``fit_points``
    con psi = 0 e mandibola a 0 se ``free_expr`` e' falso, col prior d'espressione dichiarato altrimenti."""
    import scipy.sparse as sp
    m, n = len(ctx["sub"]), len(ctx["used"])
    S = sp.csr_matrix((np.ones(m), (np.arange(m), ctx["sub"])), shape=(m, n))
    return fit_corr(S, Y, np.ones(m), ctx, n_coef(ctx, free_expr), sigma, free_expr and "jaw_r" in ctx)


class Surface:
    """Punto piu' vicino su una mesh triangolata (open3d RaycastingScene, float32): (punti, triangoli, (u, v)) con
    p = (1 - u - v) V[F0] + u V[F1] + v V[F2]."""

    def __init__(self, V: np.ndarray, F: np.ndarray):
        import open3d as o3d
        self.F = np.asarray(F, dtype=np.int64)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(V, dtype=np.float32)),
                                 o3d.core.Tensor(self.F.astype(np.uint32)))

    def closest(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import open3d as o3d
        a = self.scene.compute_closest_points(o3d.core.Tensor(np.asarray(Q, dtype=np.float32)))
        return (a["points"].numpy().astype(np.float64), a["primitive_ids"].numpy().astype(np.int64),
                a["primitive_uvs"].numpy().astype(np.float64))


def posed_model(c: np.ndarray, theta: float, R: np.ndarray, t: np.ndarray, ctx: dict) -> np.ndarray:
    """(n_r, 3) mesh del modello posata come fittata, su tutti i vertici della regione (mm, frame di lavoro)."""
    A, b = region_system(ctx, float(theta) if np.isfinite(theta) else 0.0, len(c))
    return (A @ c + b) @ R.T + t


def surface_error(Vin: np.ndarray, Fin: np.ndarray, Xs: np.ndarray, M: np.ndarray, ctx: dict,
                  surf_in: "Surface | None" = None) -> np.ndarray:
    """Sez. 3: [mediana, p95 dell'unione, mediana M -> ingresso, mediana ingresso -> M, frazione ingresso -> M
    tenuta] in mm. M -> ingresso: tutti i vertici della regione; ingresso -> M: i punti ``Xs`` il cui punto piu'
    vicino su M non cade su un triangolo col bordo della regione."""
    surf_in = surf_in or Surface(Vin, Fin)
    d1 = np.linalg.norm(surf_in.closest(M)[0] - M, axis=1)
    p, tri, _ = Surface(M, ctx["F_r"]).closest(Xs)
    keep = ~ctx["bnd_r"][ctx["F_r"][tri]].any(1)
    d2 = np.linalg.norm(p - Xs, axis=1)[keep]
    u = np.concatenate([d1, d2])
    return np.array([np.median(u), np.percentile(u, 95), np.median(d1),
                     np.median(d2) if len(d2) else np.nan, keep.mean()])


def loop_correspondences(Vin_bnd: np.ndarray, surf_in: Surface, Xs: np.ndarray, M: np.ndarray, ctx: dict,
                         tau: float):
    """Corrispondenze della variante B: (S sparsa (m, n_r), Y (m, 3), w (m,), frazioni tenute (2,))."""
    import scipy.sparse as sp
    sub, n, N = ctx["sub"], len(ctx["used"]), len(ctx["sub"])
    p, tri, _ = surf_in.closest(M[sub])                                       # modello -> ingresso
    k1 = (np.linalg.norm(p - M[sub], axis=1) <= tau) & ~Vin_bnd[surf_in.F[tri]].any(1)
    q, tri2, uv = Surface(M, ctx["F_r"]).closest(Xs)                          # ingresso -> modello
    k2 = (np.linalg.norm(q - Xs, axis=1) <= tau) & ~ctx["bnd_r"][ctx["F_r"][tri2]].any(1)
    n1, n2 = int(k1.sum()), int(k2.sum())
    frac = np.array([n1 / N, n2 / len(Xs)])
    if n1 == 0 or n2 == 0:
        return None, None, None, frac
    bary = np.c_[1.0 - uv[k2].sum(1), uv[k2]]
    rows = np.r_[np.arange(n1), np.repeat(n1 + np.arange(n2), 3)]
    cols = np.r_[sub[k1], ctx["F_r"][tri2[k2]].ravel()]
    vals = np.r_[np.ones(n1), bary.ravel()]
    S = sp.csr_matrix((vals, (rows, cols)), shape=(n1 + n2, n))
    w = np.r_[np.full(n1, 0.5 * N / n1), np.full(n2, 0.5 * N / n2)]
    return S, np.r_[p[k1], Xs[k2]], w, frac


def fit_loop(Vin: np.ndarray, Fin: np.ndarray, Xs: np.ndarray, f0: dict, ctx: dict, free_expr: bool,
             sigma: float, tau: float, iters: int, record: tuple = ()) -> dict:
    """Variante B: ``iters`` iterazioni esterne (corrispondenze punto-superficie nei due versi sul modello
    corrente, poi ``fit_corr`` con al piu' ``LOOP_INNER`` giri) a partire dal fit ``f0`` (variante A). ``record``:
    iterazioni di cui tenere lo stato (pilota). Uscita: lo stato finale (+ ``kept``, ``states``)."""
    k = n_coef(ctx, free_expr)
    jaw = free_expr and "jaw_r" in ctx
    surf_in = Surface(Vin, Fin)
    bnd = boundary_vertices(np.asarray(Fin, dtype=np.int64), len(Vin))
    f = {**f0, "c": np.r_[f0["beta"], f0["psi"]][:k]}
    states = {}
    frac = np.full(2, np.nan)
    for it in range(1, iters + 1):
        M = posed_model(f["c"], f["theta"], f["R"], f["t"], ctx)
        S, Y, w, frac = loop_correspondences(bnd, surf_in, Xs, M, ctx, tau)
        if S is None:
            raise ValueError(f"nessuna corrispondenza tenuta (frazioni {frac})")
        f = fit_corr(S, Y, w, ctx, k, sigma, jaw, f["c"], f["theta"], LOOP_INNER)
        f["kept"] = frac
        if it in record:
            states[it] = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in f.items()}
    f["states"] = states
    return f


# ------------------------------------------------------------------------------------------ varifold

def varifold_measure(view: str, path: Path) -> dict:
    """Misura varifold di una mesh in mm: centroidi (centrati sul baricentro pesato per area), normali, aree / area
    totale; quantizzata su ``VARIFOLD_CELL_MM`` (``geometric_matrix.quantize_measure``)."""
    import torch
    sys.path.insert(0, str(AAU_DIR / "baselines"))
    from geometric_matrix import quantize_measure
    V, F = blmm.load_raw(path)
    V = blmm.to_mm(view, V, Path(path).stem)
    t = V[F]
    cr = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    nrm = np.linalg.norm(cr, axis=1)
    ok = nrm > 0
    c, n, a = t[ok].mean(1), cr[ok] / nrm[ok, None], 0.5 * nrm[ok]
    c = c - (a[:, None] * c).sum(0) / a.sum()
    f = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    return quantize_measure({"centroids": f(c), "normals": f(n), "areas": f(a / a.sum()), "_cache": {}},
                            VARIFOLD_CELL_MM)


# ----------------------------------------------------------------------------------------- mesh

def meshes(view: str) -> list[tuple[str, str, Path]]:
    """(soggetto, topologia, percorso) delle mesh da iscrivere: 100 soggetti x 5 topologie senza crop; FaMoS: le 15
    scansioni di galleria (soggetto = view_id, topologia = ``scan_gallery``)."""
    if view == "famos":
        return [(r["view_id"], "scan_gallery", blmm.VIEWS["famos"]["dir"] / f"{r['name']}.npz")
                for r in blmm.famos_manifest() if r["kind"] == "scan" and r["role"] == "gallery"]
    return [(s, t, blmm.mesh_path(view, s, t)) for s in blmm.subjects(view) for t in TOPOLOGIES]


def out_dir(view: str, name: str | None = None) -> Path:
    return OUT_ROOT / view / name if name else OUT_ROOT / view


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

