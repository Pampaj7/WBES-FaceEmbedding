"""Moltiplicatori di dati: ibridi d'identita' fra 3DMM, trasferimento d'espressioni, bump RBF, al volo.

    from v3_work.mm_aug import AugConfig, sample_view_spec
    cfg = AugConfig()                                  # probabilita' dei tipi in cfg.p_kind
    spec = sample_view_spec(np.random.default_rng(0), cfg)
    spec["V"], spec["F"]          # vista nel frame canonico dello stream (sources.MMSource.view_mesh), float64 / int32
    spec["neutral_points"]        # (1478, 3) forma NEUTRA sulla regione unificata, unita' native del template
    spec["gt_domain"]             # chiave per stream/targets.CanonTargets: FR, SR e S_i da neutral_points
    spec["provenance"]            # tutto per rigenerare il campione (rebuild_view_spec) e filtrarlo per licenza

Tipi (``KINDS``), per vista; il template A e' sempre un modello di training dello stream (``TEMPLATES``):
  pure           identita' di A (code larghe della libreria) e, con ``p_expr``, un'espressione dal prior di A;
                 ``famos`` (se in ``templates``): una persona TRAIN, neutra o un fotogramma registrato;
  hybrid         X = mean_A + alpha Delta_A + beta T_{B->A}(Delta_B), B != A: Delta = identita' meno media;
                 (alpha, beta) = (cos theta, sin theta), theta ~ U(``theta_deg``): la varianza di due deformazioni
                 indipendenti resta quella di una. Espressione nativa di A con ``p_expr``;
  expr_transfer  identita' di A e un'espressione di B != A (FLAME 100, GNM 383, ICT 53, BFM 2019 100 col loro
                 prior) o di FaMoS TRAIN (fotogramma meno neutra della stessa persona, rigida robusta sul neutro);
  rbf            identita' di A piu' 1-3 bump RBF della libreria (ampiezza 1-3 mm, raggio 10-30 mm) con centro
                 nella regione: il bump fa parte dell'identita', quindi anche della neutra della GT.

Trasporto (``transfer.RegionTransfer``): il campo di B sui punti della regione unificata, nel frame FLAME in mm
veri, portato sui vertici di A (baricentrico dentro la regione, raccordo biarmonico in una banda di 40 mm fuori,
zero oltre). GT: la neutra sui punti della regione e' ESATTA, mean_u + alpha id_u z_A + beta g_B (il trasporto e'
interpolante sui punti), quindi FR/SR del campione sono quelli della forma generata sul template A. Un'espressione
(nativa o trasferita) non entra mai nella neutra.

Validita' (``cfg.validate``): ``cheap`` = triangoli capovolti e degeneri contro il template, ``full`` = anche le
auto-intersezioni NUOVE, cioe' con almeno un triangolo fuori dalla zona gia' intersecata nel template, dilatata di 2
anelli (le labbra della media si compenetrano agli angoli della bocca); soglie per template = p99 dei campioni puri dello stesso template (``thresholds.json``, ``stats.py
--calibrate``): un campione non deve essere peggiore di quelli che il modello nativo genera. Un campione
scartato si riestrae (stesso generatore, fino a ``max_tries``), poi si ripiega sul puro.

Determinismo e provenienza: ``sample_view_spec`` estrae dal generatore un solo intero, il seme del campione;
tutto il resto viene da ``default_rng(seme)`` (``view_spec_from_seed``). ``provenance`` ha il seme, il tipo, le
fonti con alpha e beta, tutti i coefficienti, i bump, la persona e il fotogramma FaMoS, i tentativi, e le
licenze: ``rebuild_view_spec(provenance)`` ricostruisce la vista dai coefficienti senza generatore (stessi
array, bit per bit). Licenza del campione = la piu' restrittiva delle fonti (``LICENSES``, ``redistributable``).
Niente si scrive su disco.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "v3_work" / "stream", REPO_ROOT / "v3_work" / "unified_gt",
           REPO_ROOT / "v3_work" / "canonical_gt"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from . import transfer as TR  # noqa: E402

AUG_VERSION = "mm_aug-1"
TEMPLATES = ("bfm2019", "ict", "gnm", "flame2020")          # i 3DMM di training dello stream
EXPR_SOURCES = TEMPLATES + ("famos",)
KINDS = ("pure", "hybrid", "expr_transfer", "rbf")
THRESHOLDS_JSON = THIS_DIR / "thresholds.json"
PHYSICAL_UNITS = {"mm", "cm", "m", "um"}                     # unita' dichiarate (la taglia della GT e' affidabile)
SI_ZONE_RINGS = 2                                            # dilatazione della zona gia' intersecata nel template

# Licenze delle fonti. ``redistributable``: un derivato si puo' pubblicare. ``evidence``: dove l'abbiamo letto;
# "non verificato nel repo" = da controllare prima di un rilascio. Un campione eredita la piu' restrittiva.
LICENSES = {
    "ict": {"license": "MIT", "redistributable": True, "evidence": "external/ICT-FaceKit/LICENSE"},
    "gnm": {"license": "Apache-2.0", "redistributable": True, "evidence": "~/data/gnm_head/PROVENANCE.md"},
    "flame2020": {"license": "FLAME (MPI-IS), solo ricerca non commerciale", "redistributable": False,
                  "evidence": "non verificato nel repo"},
    "bfm2019": {"license": "BFM 2019 (Univ. Basel), solo ricerca non commerciale", "redistributable": False,
                "evidence": "non verificato nel repo"},
    "famos": {"license": "FaMoS (MPI-IS), solo ricerca non commerciale", "redistributable": False,
              "evidence": "aau/runs/evidence/famos/NOTES.md (licenza MPI)"},
}


@dataclass(frozen=True)
class AugConfig:
    p_kind: tuple = (("pure", 0.4), ("hybrid", 0.3), ("expr_transfer", 0.15), ("rbf", 0.15))
    templates: tuple = TEMPLATES          # template A; "famos" qui = FaMoS anche come dominio puro
    hybrid_sources: tuple = TEMPLATES     # B degli ibridi
    expr_sources: tuple = EXPR_SOURCES    # B dei trasferimenti d'espressione
    p_expr: float = 0.6                   # espressione nativa nelle viste pure, ibride, rbf
    theta_deg: tuple = (0.0, 90.0)        # (alpha, beta) = (cos, sin) theta
    n_bumps: tuple = (1, 3)
    validate: str = "cheap"               # none | cheap | full
    max_tries: int = 10
    band_mm: float = TR.BAND_MM

    def kind_probs(self) -> dict:
        p = {k: float(v) for k, v in self.p_kind}
        bad = set(p) - set(KINDS)
        if bad:
            raise ValueError(f"tipi sconosciuti: {bad}")
        tot = sum(p.values())
        return {k: v / tot for k, v in p.items() if v > 0}

    def to_dict(self) -> dict:
        return {k: (list(map(list, v)) if k == "p_kind" else list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}


def config_from_dict(d: dict) -> AugConfig:
    d = dict(d)
    d["p_kind"] = tuple((k, float(v)) for k, v in d["p_kind"])
    for k in ("templates", "hybrid_sources", "expr_sources", "theta_deg", "n_bumps"):
        d[k] = tuple(d[k])
    return AugConfig(**d)


# ------------------------------------------------------------------------------------------- domini

def _project(basis: np.ndarray, tri: np.ndarray, bary: np.ndarray) -> np.ndarray:
    if basis.ndim == 2:
        return np.einsum("qkd,qk->qd", basis[tri], bary)
    return np.einsum("qkdj,qk->qdj", basis[tri], bary)


class Template:
    """Un 3DMM dello stream (``sources.MMSource``) con frame in mm veri, trasporto e mappe verso la regione."""

    def __init__(self, src, F_u: np.ndarray, band_mm: float):
        self.src, self.name, self.model = src, src.name, src.model
        self.u = float(self.model.frame.mm_per_unit)
        self.R = np.asarray(src.canon["R"], dtype=np.float64)
        self.faces = np.asarray(src.faces)
        # base d'espressione sui punti della regione (MMSource proietta solo media e identita')
        vidx, bary = _UNI().map_of(self.name)
        rv = np.asarray(self.model.region_vertices if self.model.region_vertices is not None
                        else np.arange(self.model.n_verts))
        pos = np.searchsorted(rv, vidx)
        self.ex_u = _project(self.model.expr.basis, pos, bary)
        self.ref_mm = self.to_mm(src.mean_w)
        self.rt = TR.RegionTransfer(self.ref_mm, self.faces, self.to_mm(src.mean_u), F_u, band_mm=band_mm)
        # punti della regione sulla topologia di lavoro (per i bump: GT dalla mesh)
        self.u2w_tri, self.u2w_bary, self.u2w_dist = TR.closest_on_mesh(self.to_mm(src.mean_u), self.ref_mm,
                                                                         self.faces)
        self.inside_idx = np.flatnonzero(self.rt.inside)
        self.size_ok = self.model.frame.units in PHYSICAL_UNITS

    def to_mm(self, X: np.ndarray) -> np.ndarray:
        """Nativo -> frame FLAME in mm veri (rotazione canonica dello stream, unita' dichiarata; niente traslazione:
        vale per posizioni e spostamenti)."""
        return self.u * (np.asarray(X, dtype=np.float64) @ self.R.T)

    def from_mm(self, Y: np.ndarray) -> np.ndarray:
        return (np.asarray(Y, dtype=np.float64) @ self.R) / self.u

    def u2w(self, D: np.ndarray) -> np.ndarray:
        """Campo (n_w, 3) sui vertici di lavoro -> punti della regione (baricentriche sulla topologia di lavoro)."""
        return np.einsum("qkd,qk->qd", np.asarray(D)[self.faces[self.u2w_tri]], self.u2w_bary)

    def canonical(self, V: np.ndarray) -> np.ndarray:
        """La vista nel frame dello stream (MMSource.view_mesh): similarita' del json."""
        c = self.src.canon
        return c["scale"] * (V @ np.asarray(c["R"]).T) + np.asarray(c["t"])

    def mm_factor(self) -> float:
        """mm veri per unita' della vista canonica (= targets.CanonTargets.mm_factor)."""
        return self.u / float(self.src.canon["scale"])


class Famos:
    """FaMoS TRAIN (``sources.FamosSource``): espressioni reali = fotogramma - neutra della stessa persona."""

    def __init__(self, src, flame_ref_u_mm: np.ndarray, w: np.ndarray):
        self.src, self.persons = src, src.persons
        self.ref, self.w = flame_ref_u_mm, w

    def n_frames(self, person: str) -> int:
        return int(len(self.src._data(person)[0]))

    def points(self, X: np.ndarray) -> np.ndarray:
        return np.einsum("nkd,nk->nd", np.asarray(X, dtype=np.float64)[self.src.vidx], self.src.bary)

    def expression_mm(self, person: str, frame: int) -> np.ndarray:
        """g (n_u, 3) nel frame FLAME in mm: fotogramma allineato alla neutra con la rigida robusta (Tukey, come la
        GT FR: l'espressione non trascina la posa), meno la neutra, ruotato con la rigida neutra -> media FLAME."""
        V, Vn = self.src._data(person)
        Pn, Pf = self.points(Vn), self.points(V[int(frame)])
        A = robust_rigid(Pf, Pn, self.w)
        _, R0, _ = rigid(Pn, self.ref, self.w)
        return (A - Pn) @ R0.T


def rigid(X: np.ndarray, Y: np.ndarray, w: np.ndarray):
    import cgt
    R, t = cgt.rigid_fit(X[None], Y, w[None])
    return np.einsum("vb,ab->va", X, R[0]) + t[0], R[0], t[0]


def robust_rigid(X: np.ndarray, Y: np.ndarray, w: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """X allineata rigidamente su Y: IRLS con Tukey (c = 3 mediane pesate), la regola di ``cgt.rigid_robust``."""
    import cgt
    A, _, _ = rigid(X, Y, w)
    for _ in range(n_iter):
        r = np.linalg.norm(A - Y, axis=1)
        c = max(cgt.TUKEY_C * float(cgt.weighted_median(r[None], w)[0]), 1e-12)
        psi = np.clip(1.0 - (r / c) ** 2, 0.0, None) ** 2
        A2, _, _ = rigid(X, Y, w * psi)
        done = np.abs(A2 - A).max() < 1e-9
        A = A2
        if done:
            break
    return A


@lru_cache(maxsize=1)
def _UNI():
    import sources as S
    return S.Unified()


class AugLibrary:
    """Modelli, trasporti e soglie, caricati una volta per processo (i produttori fanno fork dopo)."""

    def __init__(self, cfg: AugConfig):
        import sources as S
        need = set(cfg.templates) | set(cfg.hybrid_sources) | set(cfg.expr_sources)
        bad = need - set(EXPR_SOURCES)
        if bad:
            raise ValueError(f"fonti non di training o sconosciute: {bad} (ammesse {EXPR_SOURCES})")
        self.uni = _UNI()
        self.F_u = self.uni.sp.z["F"]
        mm = sorted(need - {"famos"}, key=TEMPLATES.index)
        srcs = S.build_sources(mm + (["famos"] if "famos" in need else []), self.uni)
        self.srcs = srcs
        self.tpl = {d: Template(srcs[d], self.F_u, cfg.band_mm) for d in mm}
        self.famos = None
        if "famos" in need:
            fl = self.tpl.get("flame2020") or Template(S.MMSource("flame2020", self.uni), self.F_u, cfg.band_mm)
            self.famos = Famos(srcs["famos"], fl.to_mm(fl.src.mean_u), self.uni.sp.W)
        self.thresholds = json.loads(THRESHOLDS_JSON.read_text()) if THRESHOLDS_JSON.exists() else {}
        self._targets = None

    def targets(self):
        """``stream/targets.CanonTargets`` con i frame di questi modelli (FR, SR, S_i dalla neutra)."""
        if self._targets is None:
            from targets import CanonTargets
            self._targets = CanonTargets(self.srcs)
        return self._targets


@lru_cache(maxsize=4)
def get_library(cfg: AugConfig) -> AugLibrary:
    return AugLibrary(cfg)


# ------------------------------------------------------------------------------------------ estrazioni

def _lst(x) -> list:
    return [float(v) for v in np.asarray(x, dtype=np.float64).ravel()]


def draw_identity(rng: np.random.Generator, kind: str, A: str, cfg: AugConfig, lib: AugLibrary) -> dict:
    """Parametri espliciti dell'identita' (tipo ``pure`` | ``hybrid`` | ``rbf``) sul template A."""
    if A == "famos":
        return {"kind": "pure", "A": "famos", "person": lib.famos.persons[int(rng.integers(len(lib.famos.persons)))]}
    t = lib.tpl[A]
    ident = {"kind": kind, "A": A, "z_A": _lst(t.model.sample_identity(rng))}
    if kind == "hybrid":
        Bs = [b for b in cfg.hybrid_sources if b != A]
        B = Bs[int(rng.integers(len(Bs)))]
        theta = float(np.radians(rng.uniform(*cfg.theta_deg)))
        ident.update(B=B, z_B=_lst(lib.tpl[B].model.sample_identity(rng)), theta_deg=float(np.degrees(theta)),
                     alpha=float(np.cos(theta)), beta=float(np.sin(theta)))
    elif kind == "rbf":
        from v3_work.mm.model import RBF_AMPLITUDE_MM, RBF_RADIUS_MM
        n = int(rng.integers(cfg.n_bumps[0], cfg.n_bumps[1] + 1))
        V = t.ref_mm + t.to_mm(t.src.id_w @ np.asarray(ident["z_A"]))
        area = _vertex_areas(V, t.faces)[t.inside_idx]
        bumps = []
        for _ in range(n):
            bumps.append({"center": int(t.inside_idx[int(rng.choice(len(area), p=area / area.sum()))]),
                          "radius_mm": float(rng.uniform(*RBF_RADIUS_MM)),
                          "amplitude_mm": float(rng.uniform(*RBF_AMPLITUDE_MM)) * (1.0 if rng.random() < 0.5 else -1.0)})
        ident["bumps"] = bumps
    elif kind != "pure":
        raise ValueError(kind)
    return ident


def draw_expression(rng: np.random.Generator, A: str, mode: str, cfg: AugConfig, lib: AugLibrary) -> dict | None:
    """``mode``: ``none`` | ``native`` (prior di A) | ``transfer`` (B != A dalle ``expr_sources``)."""
    if mode == "none":
        return None
    if mode == "native":
        return {"mode": "native", "source": A, "coef": _lst(lib.tpl[A].model.sample_expression(rng))}
    Bs = [b for b in cfg.expr_sources if b != A]
    B = Bs[int(rng.integers(len(Bs)))]
    if B == "famos":
        person = lib.famos.persons[int(rng.integers(len(lib.famos.persons)))]
        return {"mode": "transfer", "source": "famos", "person": person,
                "frame": int(rng.integers(lib.famos.n_frames(person)))}
    return {"mode": "transfer", "source": B, "coef": _lst(lib.tpl[B].model.sample_expression(rng))}


def _vertex_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    a = 0.5 * np.linalg.norm(TR.face_normals(V, F, unit=False), axis=1) / 3.0
    out = np.zeros(len(V))
    for k in range(3):
        np.add.at(out, F[:, k], a)
    return out


# ------------------------------------------------------------------------------------------ costruzione

def build_identity(ident: dict, lib: AugLibrary) -> tuple[np.ndarray, np.ndarray]:
    """(neutra sui vertici di lavoro, neutra sui punti della regione), unita' native di A. Deterministica."""
    t = lib.tpl[ident["A"]]
    s = t.src
    z = np.asarray(ident["z_A"], dtype=np.float64)
    if ident["kind"] == "hybrid":
        tb = lib.tpl[ident["B"]]
        g = tb.to_mm(tb.src.id_u @ np.asarray(ident["z_B"], dtype=np.float64))     # Delta_B, mm, frame FLAME
        a, b = float(ident["alpha"]), float(ident["beta"])
        Vw = s.mean_w + a * (s.id_w @ z) + b * t.from_mm(t.rt.apply(g))
        P = s.mean_u + a * (s.id_u @ z) + b * t.from_mm(g)                        # esatto sui punti
        return Vw, P
    Vw, P = s.mean_w + s.id_w @ z, s.mean_u + s.id_u @ z
    if ident["kind"] == "rbf":
        from v3_work.mm.model import rbf_bump, vertex_normals
        X0 = t.to_mm(Vw)
        X = X0
        for bp in ident["bumps"]:
            X = rbf_bump(X, vertex_normals(X, t.faces), bp["center"], bp["radius_mm"], bp["amplitude_mm"])
        D = t.from_mm(X - X0)
        Vw, P = Vw + D, P + t.u2w(D)
    return Vw, P


def build_expression(expr: dict | None, A: str, lib: AugLibrary) -> np.ndarray | None:
    """Spostamento d'espressione (n_w, 3) sui vertici di lavoro di A, unita' native."""
    if expr is None:
        return None
    t = lib.tpl[A]
    if expr["mode"] == "native":
        return t.src.ex_w @ np.asarray(expr["coef"], dtype=np.float64)
    if expr["source"] == "famos":
        g = lib.famos.expression_mm(expr["person"], expr["frame"])
    else:
        tb = lib.tpl[expr["source"]]
        g = tb.to_mm(tb.ex_u @ np.asarray(expr["coef"], dtype=np.float64))
    return t.from_mm(t.rt.apply(g))


def _famos_view(ident: dict, expr: dict | None, lib: AugLibrary) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ugt import apply_sim, umeyama
    src = lib.famos.src
    V, Vn = src._data(ident["person"])
    X = np.asarray(V[expr["frame"]] if expr else Vn, dtype=np.float64)[src.rv]
    return apply_sim(X, *umeyama(X, src.target, scale=False)), src.faces, src.neutral_points(ident["person"])


# ------------------------------------------------------------------------------------------ licenze

def license_of(sources: list[str]) -> dict:
    lic = {s: LICENSES[s] for s in sources}
    red = all(v["redistributable"] for v in lic.values())
    restrictive = [s for s in sources if not LICENSES[s]["redistributable"]]
    return {"sources": lic, "redistributable": red,
            "license": "; ".join(sorted({LICENSES[s]["license"] for s in (restrictive or sources)}))}


def _sources_of(ident: dict, expr: dict | None) -> list[str]:
    out = [ident["A"]]
    if ident.get("B"):
        out.append(ident["B"])
    if expr is not None and expr.get("source") and expr["source"] not in out:
        out.append(expr["source"])
    return out


# ------------------------------------------------------------------------------------------ API

def assemble(prov: dict, lib: AugLibrary, check: str = "none") -> dict:
    """La vista dai parametri espliciti di ``prov`` (identita', espressione): nessun generatore."""
    ident, expr = prov["identity"], prov["expression"]
    A = ident["A"]
    if A == "famos":
        V, F, P = _famos_view(ident, expr, lib)
        out = {"V": V, "F": F, "neutral_points": P, "gt_domain": "famos", "mm_factor": 1.0, "size_mask": True,
               "checks": None}
    else:
        t = lib.tpl[A]
        Vw, P = build_identity(ident, lib)
        D = build_expression(expr, A, lib)
        Vn = Vw if D is None else Vw + D
        checks = None
        if check != "none":
            X = t.to_mm(Vn)
            checks = TR.mesh_checks(X, t.faces, t.ref_mm, full=False)
            if check == "full" and checks["finite"]:
                checks["self_intersections_new"] = _new_intersections(t, X)
        size_ok = t.size_ok and (ident.get("B") is None or lib.tpl[ident["B"]].size_ok)
        out = {"V": t.canonical(Vn), "F": t.faces, "neutral_points": P, "gt_domain": A,
               "mm_factor": t.mm_factor(), "size_mask": bool(size_ok), "checks": checks}
    out.update(template=A, kind=prov["kind"], expr="neutral" if expr is None else "expr", provenance=prov)
    return out


def _new_intersections(t: Template, X: np.ndarray) -> int:
    """Auto-intersezioni NUOVE: coppie con almeno un triangolo fuori dalla zona gia' intersecata nel template (i
    vertici dei triangoli che si intersecano nella media, dilatati di ``SI_ZONE_RINGS`` anelli: le labbra della
    media si compenetrano agli angoli della bocca, e ogni identita' sposta un po' quelle coppie)."""
    if not hasattr(t, "_si_zone"):
        si = TR.self_intersections(t.ref_mm, t.faces)
        zone = np.zeros(len(t.ref_mm), dtype=bool)
        zone[t.faces[si.ravel()].ravel()] = True
        E = t.rt.edges
        for _ in range(SI_ZONE_RINGS):
            z = zone.copy()
            z[E[:, 0][zone[E[:, 1]]]] = True
            z[E[:, 1][zone[E[:, 0]]]] = True
            zone = z
        t._si_zone = zone[t.faces].any(1)                 # triangoli che toccano la zona
    si = TR.self_intersections(X, t.faces)
    return int((~(t._si_zone[si[:, 0]] & t._si_zone[si[:, 1]])).sum()) if len(si) else 0


def valid(spec: dict, lib: AugLibrary) -> bool:
    c = spec["checks"]
    if c is None:
        return True
    th = lib.thresholds.get(spec["template"], {})
    if not c["finite"]:
        return False
    if c["flipped"] > th.get("flipped", 0) or c["degenerate"] > th.get("degenerate", 0):
        return False
    if "self_intersections_new" in c and c["self_intersections_new"] > th.get("self_intersections_new", 0):
        return False
    return True


def draw_view(rng: np.random.Generator, kind: str, A: str, cfg: AugConfig, lib: AugLibrary) -> dict:
    """Parametri espliciti (identita', espressione) di una vista del tipo ``kind`` sul template A."""
    if A == "famos":
        ident = draw_identity(rng, "pure", A, cfg, lib)
        expr = None
        if rng.random() < cfg.p_expr:
            expr = {"mode": "frame", "frame": int(rng.integers(lib.famos.n_frames(ident["person"])))}
        return {"identity": ident, "expression": expr}
    ident = draw_identity(rng, "pure" if kind == "expr_transfer" else kind, A, cfg, lib)
    if kind == "expr_transfer":
        mode = "transfer"
    else:
        mode = "native" if rng.random() < cfg.p_expr else "none"
    return {"identity": ident, "expression": draw_expression(rng, A, mode, cfg, lib)}


def view_spec_from_seed(seed: int, cfg: AugConfig, lib: AugLibrary | None = None) -> dict:
    """Il campione del seme ``seed``: tipo, template e coefficienti da ``default_rng(seed)``, riestratti finche' la
    vista e' valida (al massimo ``max_tries``; poi il puro dello stesso template)."""
    lib = lib or get_library(cfg)
    rng = np.random.default_rng(int(seed))
    probs = cfg.kind_probs()
    kinds = list(probs)
    kind = kinds[int(rng.choice(len(kinds), p=np.asarray([probs[k] for k in kinds])))]
    tpls = [a for a in cfg.templates if a != "famos" or kind == "pure"]
    A = tpls[int(rng.integers(len(tpls)))]
    for attempt in range(cfg.max_tries + 1):
        k = kind if attempt < cfg.max_tries else "pure"     # ultimo tentativo: il puro
        params = draw_view(rng, k, A, cfg, lib)
        prov = _provenance(seed, kind, k, attempt, params, cfg)
        spec = assemble(prov, lib, check=("none" if A == "famos" else cfg.validate))
        if cfg.validate == "none" or valid(spec, lib) or attempt == cfg.max_tries:
            prov["valid"] = bool(cfg.validate == "none" or valid(spec, lib))
            return spec
    raise AssertionError("non raggiunto")


def _provenance(seed: int, kind_drawn: str, kind: str, attempt: int, params: dict, cfg: AugConfig) -> dict:
    srcs = _sources_of(params["identity"], params["expression"])
    return {"version": AUG_VERSION, "seed": int(seed), "kind": kind, "kind_drawn": kind_drawn, "attempt": attempt,
            "template": params["identity"]["A"], "identity": params["identity"], "expression": params["expression"],
            "sources_used": srcs, **license_of(srcs), "config": cfg.to_dict()}


def sample_view_spec(rng: np.random.Generator, cfg: AugConfig, lib: AugLibrary | None = None) -> dict:
    """Una vista: tipo secondo ``cfg.p_kind``; dal generatore si estrae SOLO il seme del campione.

    Ritorna ``V`` (frame canonico dello stream), ``F``, ``template``, ``kind``, ``expr``, ``neutral_points`` e
    ``gt_domain`` (bersagli FR/SR: ``targets.CanonTargets()(gt_domain, neutral_points)``, o ``gt_targets``),
    ``mm_factor`` (mm veri per unita' della vista, per ``area_mm2``), ``size_mask`` (True = la taglia della GT e'
    affidabile: unita' dichiarate in tutte le fonti d'identita'), ``checks`` e ``provenance``."""
    seed = int(rng.integers(0, 2 ** 63 - 1))
    return view_spec_from_seed(seed, cfg, lib)


def rebuild_view_spec(prov: dict, lib: AugLibrary | None = None, check: str | None = None) -> dict:
    """La vista dai coefficienti della provenienza (non dal seme): stessi array di quando e' stata generata."""
    cfg = config_from_dict(prov["config"])
    lib = lib or get_library(cfg)
    return assemble(prov, lib, check=cfg.validate if check is None else check)


def gt_targets(spec: dict, lib: AugLibrary) -> dict:
    """FR, SR, S_i della neutra (la funzione dello stream, ``targets.CanonTargets``)."""
    return lib.targets()(spec["gt_domain"], spec["neutral_points"])


def provenance_json(prov: dict) -> str:
    return json.dumps(prov, sort_keys=True)


__all__ = ["AUG_VERSION", "AugConfig", "AugLibrary", "EXPR_SOURCES", "KINDS", "LICENSES", "TEMPLATES",
           "assemble", "build_expression", "build_identity", "config_from_dict", "draw_view", "get_library",
           "gt_targets", "rebuild_view_spec", "sample_view_spec", "view_spec_from_seed"]
