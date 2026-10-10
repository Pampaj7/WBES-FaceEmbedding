"""Baseline geometriche con la rimozione dei disturbi coerente con la GT (FR in mm, SR senza taglia): libreria.

Le baseline geometriche pubblicate (Chamfer, ICP + Chamfer, NICP per coppia, NICP su template) girano su mesh
normalizzate maxabs una per una (``mesh_npz_utils.normalize_vertices``) e l'ICP prealinea col bbox, che scala la
sorgente sul bersaglio (``facebench/rigid_aligners/icp.py:prealign_by_bbox``): sono cieche alla taglia per
costruzione, mentre la GT FR (E12, ``datasets/CANONICAL_GT/eval/<set>_fr.npz``) la conserva. Qui la stessa
pipeline faceBench (``run_facebench_remesh``: 4096 vertici campionati coi semi della coppia, ``icp_align``,
``nonrigid_icp_align``, ``p2p``/``p2tri``; NICP su template di ``aau/indomain/ir_template.py``) con UNA modifica,
la normalizzazione della mesh, in tre modi:

  - ``maxabs``: quella pubblicata (centro dei vertici, / max|coordinata|, coordinate native, ICP con il
    prealineamento bbox che scala). E' il controllo di riproduzione: deve ridare le matrici esistenti.
  - ``mm`` (coerente con FR): V_mm = u_d R_d V + t_d, il frame di GT-F di E12 (``aau/runs/evidence/e12/frames.json``,
    UNA trasformazione per dominio; FaMoS: V / scale_to_mm, patch T7 gia' nel frame canonico). Nessuna scala per
    mesh: le coordinate di lavoro sono (V_mm - centro dei vertici) / L_d, con L_d UNA costante per dominio (mediana
    del max|coordinata| delle original valutate, mm), cosi' NICP vede coordinate dell'ordine di quelle per cui e'
    tarato (gamma, epsilon e la soglia dei pesi di ``nonrigid_icp_align`` non sono invarianti alla scala). Le
    distanze escono x L_d, in mm. ICP rigido: il prealineamento centra soltanto (prealign_by_bbox senza la scala),
    poi ``registration_icp`` punto-punto senza scala, soglia 1000 (infinita in entrambe le unita').
    Il centraggio per mesh tocca solo l'origine (Chamfer e ICP sono invarianti; NICP no, e cosi' ha l'origine
    della pipeline pubblicata); ``chamfer_pure`` e' la Chamfer sulle coordinate NON centrate.
  - ``cs`` (coerente con SR): come ``mm``, ma ogni mesh e' scalata a centroid size CS_ref:
    X = (V_mm - centro) / CS_i x CS_ref / L_d. CS_i = centroid size robusta: pesi = aree baricentriche della
    geometria passa-basso (64 autofunzioni del Laplaciano, il modo ``smooth`` di ``v3_work/trainer/area_v3.py``),
    attorno al baricentro pesato; invariante alla tassellazione e poco sensibile al rumore, che gonfia le aree
    della mesh com'e'. CS_ref = mediana delle original valutate del dominio. Distanze in "mm alla taglia di
    riferimento".

NICP su template: template = media vertice per vertice delle original di 100 soggetti NON valutati del pool
(``rng(1234)``, come ``aau/competitors/comp_template.py``) nella normalizzazione del modo, 4096 vertici con
``rng(0)``; iscrizione = ICP di similarita' del template sulla mesh + NICP (come ``ir_template.enroll``), poi
ritorno nel frame del template con Procrustes di SIMILARITA' (``maxabs``, quello pubblicato) o RIGIDO (``mm``,
``cs``: nessuna scala per identita'); distanza = L2 media per vertice (``ir_template.template_distances``).

Per mesh (``mesh_scalars``): centroid size pesata per area, sqrt(area), altezza (estensione in y nel frame
canonico, +y alto), mm: le baseline banali NON oracolo |log x_a - log x_b|.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
FB_DIR = REPO_ROOT / "faceBench" / "latentVSpipeline"
for _p in (FB_DIR, FB_DIR.parent, AAU_DIR / "zs3dmm", AAU_DIR / "indomain"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FRAMES_JSON = AAU_DIR / "runs" / "evidence" / "e12" / "frames.json"
OUT_ROOT = AAU_DIR / "runs" / "evidence" / "baselines_mm"
PARAMS_JSON = OUT_ROOT / "params.json"
DATASETS = REPO_ROOT / "datasets"
TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")
MODES = ("maxabs", "mm", "cs")
N_POINTS = 4096
SAME_SEED_OFFSET = 1_000_000          # zs_bl_same.SEED_OFFSET
N_TEMPLATE_SUBJECTS = 100             # ir_template.N_TEMPLATE_SUBJECTS
SMOOTH_K = 64                         # area_v3.CFG["k"]: autofunzioni del passa-basso delle aree robuste
# metriche per passo: ``fast`` (Chamfer e ICP), ``nicp`` (NICP per coppia)
METRICS = {"fast": ("chamfer", "chamfer_pure", "rigid_icp_chamfer"), "nicp": ("nicp_p2tri",)}

# viste: cartella delle mesh, dominio del frame, set della GT, vista neutra per il template, baseline pubblicate
VIEWS = {
    "hifi3d": {"dir": DATASETS / "HIFI3D" / "eval_view" / "npz", "domain": "hifi3d", "gt": "hifi3d",
               "template_dir": DATASETS / "HIFI3D" / "eval_view" / "npz",
               "fb_root": AAU_DIR / "runs" / "ws_hifi3d" / "data_328f2bfc1a" / "baselines",
               "template_ref": AAU_DIR / "runs" / "competitors_hifi3d" / "template_hifi3d.npz"},
    "faceverse": {"dir": DATASETS / "FACEVERSE_ZS" / "expr_view" / "npz", "domain": "faceverse", "gt": "faceverse",
                  "template_dir": DATASETS / "FACEVERSE_ZS" / "eval_view" / "npz",
                  "fb_root": AAU_DIR / "runs" / "ws_faceverse_expr" / "data_736f96956a" / "baselines",
                  "template_ref": None},
    "facescape": {"dir": DATASETS / "DEV_FACESCAPE" / "eval_view" / "npz", "domain": "facescape", "gt": "facescape",
                  "template_dir": DATASETS / "DEV_FACESCAPE" / "eval_view" / "npz", "fb_root": None,
                  "template_ref": None},
    "facescape_expr": {"dir": DATASETS / "DEV_FACESCAPE" / "expr_view" / "npz", "domain": "facescape",
                       "gt": "facescape", "template_dir": DATASETS / "DEV_FACESCAPE" / "eval_view" / "npz",
                       "fb_root": None, "template_ref": None},
    "famos": {"dir": DATASETS / "FAMOS" / "test_view" / "npz", "domain": "famos", "gt": "famos_test",
              "template_dir": None, "fb_root": None, "template_ref": None},
}
FAMOS_MANIFEST = DATASETS / "FAMOS" / "test_view" / "manifest.csv"


# ------------------------------------------------------------------------------------- soggetti e mesh

def subjects(view: str) -> list[str]:
    """I 100 soggetti valutati (``zs_stage.select_subjects``, seme 1234), gli stessi di tutti i summary."""
    from zs_stage import select_subjects
    return select_subjects(VIEWS[view]["dir"], 1234)


def mesh_path(view: str, subject: str, topology: str, template: bool = False) -> Path:
    root = VIEWS[view]["template_dir"] if template else VIEWS[view]["dir"]
    return root / f"{subject}_GTready_{topology}.npz"


def load_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return np.asarray(z["V"], dtype=np.float64), np.asarray(z["F"], dtype=np.int64)


def famos_manifest() -> list[dict]:
    with open(FAMOS_MANIFEST, newline="") as fh:
        return list(csv.DictReader(fh))


_FAMOS_SCALE: dict = {}


def to_mm(view: str, V: np.ndarray, name: str | None = None) -> np.ndarray:
    """V nativa -> mm nel frame canonico, UNA trasformazione per dominio (FaMoS: la scala della patch T7)."""
    dom = VIEWS[view]["domain"]
    if dom == "famos":
        if not _FAMOS_SCALE:
            _FAMOS_SCALE.update({r["name"]: float(r["scale_to_mm"]) for r in famos_manifest()})
        return V / _FAMOS_SCALE[name]
    f = json.loads(FRAMES_JSON.read_text())["domains"][dom]
    return f["u"] * V @ np.asarray(f["R"]).T + np.asarray(f["t"])


def vertex_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Massa baricentrica: un terzo dell'area di ogni triangolo a ciascun vertice."""
    t = V[F]
    a = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    return np.bincount(F.ravel(), weights=np.repeat(a / 3.0, 3), minlength=len(V))


def lowpass(V: np.ndarray, F: np.ndarray, k: int = SMOOTH_K) -> np.ndarray:
    """V proiettata sulle prime k autofunzioni del Laplaciano (M-ortonormali): ``area_v3.lowpass`` in numpy.
    Le autofunzioni non dipendono dalla discretizzazione, quindi il filtro toglie il rumore allo stesso modo su
    mesh dense e rade. Laplaciano e massa robusti (``robust_laplacian.mesh_laplacian``, come
    ``aau/models/robust_area1_operators.py``): le up60k hanno triangoli di area nulla, che la cotangente non regge."""
    import robust_laplacian
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    L, M = robust_laplacian.mesh_laplacian(np.ascontiguousarray(V), np.ascontiguousarray(F))
    m = np.asarray(M.diagonal(), dtype=np.float64)
    m = m + 1e-8 * m.mean()
    eps = 1e-8 * float(np.median(np.abs(L.diagonal())))
    _, phi = eigsh((L + eps * sp.identity(len(V))).tocsc(), k=min(k, len(V) - 2), M=sp.diags(m), sigma=eps)
    return phi @ (phi.T @ (m[:, None] * V))


def mesh_scalars(Vmm: np.ndarray, F: np.ndarray) -> dict:
    """Taglia stimata dalla mesh osservata (mm). ``cs``: centroid size con le aree ROBUSTE (aree baricentriche
    della geometria passa-basso, ``area_v3`` modo ``smooth``, k = 64) attorno al baricentro pesato; ``sqrt_area``:
    radice dell'area robusta; ``*_mass``: le stesse con le aree della mesh com'e' (il rumore le gonfia);
    ``height``: estensione in y (frame canonico, +y alto); ``maxabs``: max|coordinata| dopo il centro dei vertici."""
    out = {"height": float(np.ptp(Vmm[:, 1])), "maxabs": float(np.abs(Vmm - Vmm.mean(0)).max())}
    for tag, w in (("", vertex_areas(lowpass(Vmm, F), F)), ("_mass", vertex_areas(Vmm, F))):
        m = (w[:, None] * Vmm).sum(0) / w.sum()
        out[f"cs{tag}"] = float(np.sqrt((w * ((Vmm - m) ** 2).sum(1)).sum() / w.sum()))
        out[f"sqrt_area{tag}"] = float(np.sqrt(w.sum()))
    return out


def params() -> dict:
    """L_d e CS_ref per dominio (``blmm_scalars.py``)."""
    return json.loads(PARAMS_JSON.read_text())


_SCAL: dict = {}


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT)) if Path(path).is_absolute() else str(path)


def scalars_of(view: str) -> dict:
    """{percorso relativo alla radice -> scalari} dalla cache ``<OUT_ROOT>/<vista>/scalars.npz``."""
    if view not in _SCAL:
        with np.load(OUT_ROOT / view / "scalars.npz") as z:
            cols = [c for c in z.files if c != "paths"]
            _SCAL[view] = {str(p): {c: float(z[c][k]) for c in cols} for k, p in enumerate(z["paths"])}
    return _SCAL[view]


def work_coords(view: str, mode: str, path: Path, prm: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(coordinate di lavoro (n, 3), offset del centro (3,)) di una mesh nel modo; offset = 0 tranne ``mm``."""
    V, F = load_raw(path)
    if mode == "maxabs":
        from mesh_npz_utils import normalize_vertices
        return normalize_vertices(V), np.zeros(3)
    prm = prm or params()
    dom = prm["domains"][VIEWS[view]["domain"]]
    Vmm = to_mm(view, V, Path(path).stem)
    c = Vmm.mean(0)
    X = (Vmm - c) / dom["L"]
    if mode == "mm":
        return X, c / dom["L"]
    if mode == "cs":
        return X * (dom["cs_ref"] / scalars_of(view)[rel(path)]["cs"]), np.zeros(3)
    raise ValueError(mode)


def unit(view: str, mode: str, prm: dict | None = None) -> float:
    """Moltiplicatore delle distanze di lavoro: 1 in maxabs, L_d (mm) negli altri modi."""
    return 1.0 if mode == "maxabs" else float((prm or params())["domains"][VIEWS[view]["domain"]]["L"])


# ----------------------------------------------------------------------------------- pipeline per coppia

def center_prealign(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``prealign_by_bbox`` senza la scala: solo la traslazione dei baricentri."""
    return source - source.mean(0) + target.mean(0)


def icp(source: np.ndarray, target: np.ndarray, mode: str) -> np.ndarray:
    """``facebench.icp_align``: in ``maxabs`` quella pubblicata (bbox, con scala), altrimenti centro + ICP rigido."""
    import facebench as fb
    if mode == "maxabs":
        return fb.icp_align(source, target, prealign="bbox")[0]
    import open3d as o3d
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(center_prealign(source, target))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target)
    res = o3d.pipelines.registration.registration_icp(
        src, tgt, 1000.0, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
    src.transform(res.transformation)
    return np.asarray(src.points)


def pair_metrics(X: tuple, Y: tuple, seed: int, mode: str, step: str) -> dict:
    """Metriche di ``METRICS[step]`` per la coppia (X, Y) = (coordinate, offset), unita' di lavoro.

    Stessi passi di ``run_facebench_remesh.run_geometry_pipeline``: campioni coi semi ``seed`` e ``seed + 1``,
    Chamfer simmetrica, ICP del campione di X su quello di Y e p2p media (``rigid_p2p``), NICP sul risultato
    dell'ICP e p2tri media. Una coppia che fallisce resta NaN, con l'errore."""
    import facebench as fb
    import run_facebench_remesh as rfr
    out = {m: np.nan for m in METRICS[step]}
    try:
        Xs = rfr.sample_pts(X[0], N_POINTS, seed)
        Ys = rfr.sample_pts(Y[0], N_POINTS, seed + 1)
        X_rig = icp(Xs, Ys, mode)
        if step == "fast":
            out["chamfer"] = rfr.symmetric_chamfer(Xs, Ys)
            if mode == "mm":
                out["chamfer_pure"] = rfr.symmetric_chamfer(Xs + X[1], Ys + Y[1])
            corr = fb.chamfer_correspondence(X_rig, Ys)
            out["rigid_icp_chamfer"] = float(np.mean(fb.p2p_distance(X_rig, Ys, corr)))
        else:
            X_nicp = fb.nonrigid_icp_align(X_rig, Ys)
            corr = fb.chamfer_correspondence(X_nicp, Ys)
            out["nicp_p2tri"] = float(np.mean(fb.p2tri_distance(X_nicp, Ys, corr)))
        out["error"] = ""
    except Exception as exc:  # noqa: BLE001  (come run_geometry_pipeline: la coppia resta NaN)
        out = {m: np.nan for m in METRICS[step]}
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------------------- NICP su template

def template_subjects(view: str) -> list[str]:
    """100 soggetti NON valutati del pool della vista neutra, ``rng(1234)`` (``comp_template.build_template``)."""
    tdir = VIEWS[view]["template_dir"]
    pool = sorted({p.name.split("_GTready_")[0] for p in tdir.glob("id*_GTready_original.npz")})
    others = sorted(set(pool) - set(subjects(view)))
    return sorted(np.random.default_rng(1234).choice(others, N_TEMPLATE_SUBJECTS, replace=False).tolist())


def build_template(view: str, mode: str, prm: dict | None = None) -> tuple[np.ndarray, list[str], int]:
    """Template del modo: media delle original non valutate (normalizzate nel modo), 4096 vertici ``rng(0)``.
    In ``maxabs`` e' ``comp_template.build_template`` (media di maxabs, poi di nuovo maxabs); negli altri modi la
    media e' centrata e basta (nessuna scala)."""
    from mesh_npz_utils import normalize_vertices
    chosen = template_subjects(view)
    Xs = [work_coords(view, mode, mesh_path(view, s, "original", template=True), prm)[0] for s in chosen]
    if len({x.shape for x in Xs}) != 1:
        raise SystemExit(f"{view}: original con numeri di vertici diversi")
    mean = np.mean(np.stack(Xs), axis=0)
    mean = normalize_vertices(mean) if mode == "maxabs" else mean - mean.mean(0)
    idx = np.sort(np.random.default_rng(0).choice(len(mean), N_POINTS, replace=False))
    return mean[idx], chosen, len(mean)


def rigid_to(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A portata su B con la rigida ai minimi quadrati (Umeyama senza scala)."""
    mu_a, mu_b = A.mean(0), B.mean(0)
    A0, B0 = A - mu_a, B - mu_b
    U, _, Vt = np.linalg.svd(A0.T @ B0)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return A0 @ (U @ D @ Vt) + mu_b


def enroll(X: np.ndarray, seed: int, T: np.ndarray, mode: str) -> np.ndarray:
    """``ir_template.enroll`` sulle coordinate del modo; ritorno al template con similarita' (maxabs) o rigida."""
    import facebench as fb
    import ir_simicp
    import ir_template as irt
    import run_facebench_remesh as rfr
    Xs = rfr.sample_pts(X, N_POINTS, seed)
    T_nicp = np.asarray(fb.nonrigid_icp_align(ir_simicp.similarity_icp(T, Xs), Xs), dtype=np.float64)
    return irt.procrustes_to(T_nicp, T) if mode == "maxabs" else rigid_to(T_nicp, T)


def mesh_seed(subject: str, label: str) -> int:
    """``ir_template.mesh_seed``."""
    return zlib.crc32(f"{subject}|{label}".encode()) % 2_000_000_000


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
