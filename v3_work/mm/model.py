"""Interfaccia comune dei 3DMM (``MorphableModel``) e campionatori condivisi.

Un modello e' sempre ristretto alla sua REGIONE DEL VOLTO (la patch dei set REMESH-2 e
zero-shot): ``faces`` sono i triangoli della patch a indici compattati, ``mean`` la patch media
neutra, tutto nelle unita' e nel frame NATIVI del file del modello. Il frame canonico (FLAME:
+x sinistra del soggetto, +y alto, +z fuori dal volto, normali uscenti, millimetri) lo da'
``canonical_transform``, che legge ``datasets/UNIFIED_GT/canonical_transforms.json`` se c'e'.

Coefficienti
------------
Identita': SEMPRE standardizzati, ``z ~ N(0, 1)`` e' il prior del modello. ``id_basis`` (n, 3, k)
e' lo spostamento per +1 sigma, ``id_sigma`` (k,) la sigma del coefficiente nativo (1 dove la base
del file e' gia' scalata), ``id_mean_coef`` la media del coefficiente nativo (zero tranne
FaceScape, i cui fattori di Tucker non sono centrati). Espressione: coefficienti NATIVI, perche'
la semantica cambia fra modelli:
  - ``pca`` (BFM, FLAME, GNM, HIFI3D): in unita' di sigma come l'identita', campionati
    N(0, s^2) sul pool, ``s`` tarato (``expr.scale``, vedi ``loaders``);
  - ``blendshape`` (ICT, FaceVerse, FaceScape): pesi in [0, 1], ricetta WS5
    (``aau/ict/ict_expressions_random.py``): 3-8 blendshape attivi dal pool, U(0.3, 1.0).

Ruoli
-----
``role`` = ``train`` | ``dev`` | ``test``. Ogni campionatore ha ``purpose`` (default
``"train"``): con ``purpose="train"`` un modello ``dev`` o ``test`` solleva ``RoleError``
(``assert_trainable``, un'eccezione vera: un ``assert`` sparirebbe con ``python -O``). Chi genera
un set di valutazione lo dice esplicitamente con ``purpose="eval"``. ``mesh`` non e' protetta: non
campiona niente.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_JSON = REPO_ROOT / "datasets" / "UNIFIED_GT" / "canonical_transforms.json"

ROLES = ("train", "dev", "test")
PURPOSES = ("train", "eval")

# Code pesanti (PLAN_MASSIVE sez. 2, punto 1): per COEFFICIENTE, 80% N(0,1) e 20% N(0,2^2),
# troncata a 3.5 sigma per rifiuto (la sigma del modo, cioe' |z| <= 3.5 in unita' standard).
TAIL_FRACTION = 0.2
TAIL_SIGMA = 2.0
TRUNC_SIGMA = 3.5

# Ricetta WS5 delle espressioni a blendshape (aau/ict/ict_expressions_random.py), estremi inclusi.
N_ACTIVE_MIN = 3
N_ACTIVE_MAX = 8
COEF_LOW = 0.3
COEF_HIGH = 1.0

# Deformazioni locali (sez. 2, punto 2) e coppie di perturbazione (punto 3).
RBF_AMPLITUDE_MM = (1.0, 3.0)
RBF_RADIUS_MM = (10.0, 30.0)
DELTA_SIGMA_RANGE = (0.2, 0.5)


class RoleError(RuntimeError):
    """Un modello ``dev``/``test`` chiesto da un campionatore di training."""


def assert_trainable(model: "MorphableModel") -> None:
    if model.role != "train":
        raise RoleError(f"{model.name}: role={model.role!r}, vietato nei campionatori di training "
                        f"(PLAN_MASSIVE sez. 9 e 14.5); per un set di valutazione passa purpose='eval'")


def sample_coefficients(rng: np.random.Generator, k: int, tails: bool = True,
                        trunc: float = TRUNC_SIGMA) -> np.ndarray:
    """(k,) coefficienti standardizzati.

    ``tails=True``: mistura per coefficiente (``TAIL_FRACTION`` a sigma ``TAIL_SIGMA``);
    ``tails=False``: N(0, 1). ``trunc`` > 0 tronca |z| <= trunc per rifiuto, sulla STESSA
    componente della mistura; ``trunc=0`` niente troncamento. Con ``tails=False, trunc=0`` il
    flusso del generatore e' quello di ``rng.normal(size=k)``, cioe' il campionatore dei set
    zero-shot esistenti (``zs_identities.py``: N(0, 1) non troncata).
    """
    scale = np.where(rng.random(k) < TAIL_FRACTION, TAIL_SIGMA, 1.0) if tails else np.ones(k)
    z = rng.normal(0.0, 1.0, size=k) * scale
    if trunc > 0:
        bad = np.abs(z) > trunc
        while bad.any():
            z[bad] = rng.normal(0.0, 1.0, size=int(bad.sum())) * scale[bad]
            bad = np.abs(z) > trunc
    return z


def sample_expression_coefficients(rng: np.random.Generator, expr: "ExpressionSpace") -> np.ndarray:
    """Il campionatore di ``MorphableModel.sample_expression`` senza il modello (e senza il
    controllo del ruolo, che fa il metodo): serve ai worker che non hanno il core di FaceScape."""
    e = np.zeros(len(expr.names))
    pool = expr.pool
    if expr.kind == "blendshape":
        # stesso ordine di chiamate di make_zs_expr_topologies.sample_expression
        n_active = int(rng.integers(N_ACTIVE_MIN, N_ACTIVE_MAX + 1))
        idx = rng.choice(len(pool), size=n_active, replace=False)
        e[pool[idx]] = rng.uniform(COEF_LOW, COEF_HIGH, size=n_active)
    elif expr.kind == "pca":
        if expr.scale is None:
            raise ValueError("scala d'espressione non tarata (loaders.EXPR_SCALE)")
        e[pool] = rng.normal(0.0, expr.scale, size=len(pool))
    else:
        raise ValueError("nessuna base d'espressione")
    for i, v in expr.fixed.items():     # BFM 2019: la deformazione media delle espressioni,
        e[i] = v * (expr.scale if expr.kind == "pca" else 1.0)   # scalata come il resto
    return e


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Normali per vertice, media dei triangoli pesata per area, nel verso dei triangoli."""
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])   # norma = 2 x area
    vn = np.zeros_like(V)
    for c in range(3):
        np.add.at(vn, F[:, c], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.where(n > 0, n, 1.0)


def vertex_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area baricentrica per vertice (un terzo di ogni triangolo incidente)."""
    tri = V[F]
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    out = np.zeros(len(V))
    for c in range(3):
        np.add.at(out, F[:, c], a / 3.0)
    return out


def rbf_bump(V: np.ndarray, normals: np.ndarray, center: int, radius: float, amplitude: float) -> np.ndarray:
    """Bump lungo la normale, RBF di Wendland C2 a supporto compatto: picco ``amplitude`` sul
    centro, zero oltre ``radius`` (distanza euclidea). Unita' di ``V``."""
    d = np.linalg.norm(V - V[center], axis=1) / radius
    phi = np.where(d < 1.0, (1.0 - d) ** 4 * (4.0 * d + 1.0), 0.0)
    return V + (amplitude * phi)[:, None] * normals


@dataclass(frozen=True)
class Frame:
    """Frame NATIVO: quale asse (con segno) punta a sinistra del soggetto, in alto, fuori dal volto."""

    left: str
    up: str
    forward: str
    normals: str          # "outward" | "inward": verso dei triangoli di ``faces``
    units: str
    mm_per_unit: float
    note: str = ""

    def rotation(self) -> np.ndarray:
        """R (3, 3) con x_canonico = R @ x_nativo: righe = assi nativi di sinistra, alto, avanti."""
        def axis(s: str) -> np.ndarray:
            v = np.zeros(3)
            v["xyz".index(s[-1])] = -1.0 if s.startswith("-") else 1.0
            return v
        R = np.stack([axis(self.left), axis(self.up), axis(self.forward)])
        if not np.isclose(np.linalg.det(R), 1.0):
            raise ValueError(f"frame {self}: assi non destrorsi (det {np.linalg.det(R):+.0f})")
        return R


@dataclass
class ExpressionSpace:
    kind: str                         # "pca" | "blendshape" | "none"
    names: list[str]
    pool: np.ndarray                  # indici campionabili
    basis: np.ndarray | None = None   # (n, 3, ke) per i modelli lineari, nativo per coefficiente
    scale: float = 1.0                # pca: sigma dei coefficienti campionati
    note: str = ""
    fixed: dict = field(default_factory=dict)   # coefficienti fissi (x scale se pca) in ogni espressione


@dataclass
class MorphableModel:
    name: str
    role: str
    faces: np.ndarray                 # (m, 3) int32, patch compattata
    mean: np.ndarray                  # (n, 3) float64, patch media neutra, unita' native
    id_basis: np.ndarray              # (n, 3, k): +1 sigma per coefficiente standardizzato
    id_sigma: np.ndarray              # (k,) sigma del coefficiente nativo
    expr: ExpressionSpace
    frame: Frame
    landmarks: np.ndarray | None = None        # indici nella patch
    landmark_ids: np.ndarray | None = None     # numero del landmark nello schema
    landmark_scheme: str | None = None
    landmark_names: list[str] | None = None    # schemi con nomi (BFM 2019)
    id_mean_coef: np.ndarray | None = None     # media del coefficiente nativo (FaceScape)
    region: str = ""
    region_vertices: np.ndarray | None = None  # indici nella topologia nativa completa
    info: dict = field(default_factory=dict)
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"{self.name}: role {self.role!r} non in {ROLES}")
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int32)
        if self.faces.min() < 0 or self.faces.max() != len(self.mean) - 1:
            raise ValueError(f"{self.name}: faces in [{self.faces.min()}, {self.faces.max()}] "
                             f"per {len(self.mean)} vertici")

    # ------------------------------------------------------------------ dimensioni
    @property
    def n_verts(self) -> int:
        return len(self.mean)

    @property
    def n_id(self) -> int:
        return int(self.id_basis.shape[2])

    @property
    def n_expr(self) -> int:
        return len(self.expr.names)

    # ------------------------------------------------------------------ forward
    def native_identity(self, id_z: np.ndarray) -> np.ndarray:
        """Coefficienti nel formato del file: media + sigma * z."""
        z = np.asarray(id_z, dtype=np.float64)
        mu = 0.0 if self.id_mean_coef is None else self.id_mean_coef
        return mu + self.id_sigma * z

    def expr_basis_at(self, id_z: np.ndarray | None = None) -> np.ndarray:
        """(n, 3, ke) base d'espressione; nei modelli lineari non dipende dall'identita'."""
        if self.expr.basis is None:
            raise ValueError(f"{self.name}: nessuna base d'espressione")
        return self.expr.basis

    def mesh(self, id_z: np.ndarray | None = None, expr: np.ndarray | None = None) -> np.ndarray:
        """Vertici (n, 3) float64 della patch, unita' e frame nativi. ``None`` = media / neutra."""
        V = self.mean.copy()
        if id_z is not None:
            z = np.asarray(id_z, dtype=np.float64).ravel()
            if len(z) != self.n_id:
                raise ValueError(f"{self.name}: {len(z)} coefficienti d'identita', attesi {self.n_id}")
            V += self.id_basis @ z
        if expr is not None:
            e = np.asarray(expr, dtype=np.float64).ravel()
            if len(e) != self.n_expr:
                raise ValueError(f"{self.name}: {len(e)} coefficienti d'espressione, attesi {self.n_expr}")
            V += self.expr_basis_at(id_z) @ e
        return V

    def landmark_positions(self, V: np.ndarray) -> np.ndarray | None:
        return None if self.landmarks is None else np.asarray(V)[self.landmarks]

    # ------------------------------------------------------------------ campionatori
    def _guard(self, purpose: str) -> None:
        if purpose not in PURPOSES:
            raise ValueError(f"purpose {purpose!r} non in {PURPOSES}")
        if purpose == "train":
            assert_trainable(self)

    def sample_identity(self, rng: np.random.Generator, tails: bool = True, trunc: float = TRUNC_SIGMA,
                        purpose: str = "train") -> np.ndarray:
        """(k,) coefficienti standardizzati; vedi ``sample_coefficients``."""
        self._guard(purpose)
        return sample_coefficients(rng, self.n_id, tails=tails, trunc=trunc)

    def sample_expression(self, rng: np.random.Generator, purpose: str = "train") -> np.ndarray:
        """(ke,) coefficienti d'espressione NATIVI, densi (zero fuori dagli attivi / dal pool)."""
        self._guard(purpose)
        return sample_expression_coefficients(rng, self.expr)

    def local_rbf_deform(self, verts: np.ndarray, rng: np.random.Generator, n_bumps: int = 1,
                         purpose: str = "train", return_params: bool = False):
        """Bump RBF lungo la normale: ampiezza U(1, 3) mm col segno a caso, raggio di supporto
        U(10, 30) mm, centro su un vertice scelto per area. ``verts`` nella topologia della patch
        e nelle unita' native (le misure in mm passano per ``frame.mm_per_unit``)."""
        self._guard(purpose)
        V = np.asarray(verts, dtype=np.float64)
        if V.shape != self.mean.shape:
            raise ValueError(f"{self.name}: verts {V.shape}, attesi {self.mean.shape} (topologia della patch)")
        params = []
        for _ in range(n_bumps):
            area = vertex_areas(V, self.faces)
            center = int(rng.choice(len(V), p=area / area.sum()))
            radius_mm = float(rng.uniform(*RBF_RADIUS_MM))
            amp_mm = float(rng.uniform(*RBF_AMPLITUDE_MM)) * (1.0 if rng.random() < 0.5 else -1.0)
            V = rbf_bump(V, vertex_normals(V, self.faces), center,
                         radius_mm / self.frame.mm_per_unit, amp_mm / self.frame.mm_per_unit)
            params.append({"center": center, "radius_mm": radius_mm, "amplitude_mm": amp_mm})
        return (V, params) if return_params else V

    def perturbation_pair(self, id_z: np.ndarray, delta_sigma: float | None, rng: np.random.Generator,
                          expr: np.ndarray | None = None, purpose: str = "train"):
        """Coppia (X, X + delta): ``delta`` ~ N(0, delta_sigma^2) per coefficiente standardizzato,
        ``delta_sigma`` in [0.2, 0.5] (``None`` = estratto uniforme). Per coefficiente, non in
        norma: la distanza attesa nello spazio dei coefficienti e' ``delta_sigma / sqrt(2)`` di
        quella fra due identita' indipendenti (14-35%), qualunque sia il numero di modi.
        Ritorna ``(V_a, V_b, id_b, delta_sigma)``; stessa espressione ``expr`` sui due lati."""
        self._guard(purpose)
        lo, hi = DELTA_SIGMA_RANGE
        if delta_sigma is None:
            delta_sigma = float(rng.uniform(lo, hi))
        if not lo <= delta_sigma <= hi:
            raise ValueError(f"delta_sigma {delta_sigma} fuori da [{lo}, {hi}]")
        id_a = np.asarray(id_z, dtype=np.float64)
        id_b = id_a + rng.normal(0.0, delta_sigma, size=self.n_id)
        return self.mesh(id_a, expr), self.mesh(id_b, expr), id_b, delta_sigma

    # ------------------------------------------------------------------ frame canonico
    def canonical_params(self, json_path: Path | str | None = None) -> dict:
        """``{scale, R, t, flip_faces, source}`` con x_can = scale * R @ x + t (mm).

        Dal JSON di ``datasets/UNIFIED_GT/canonical_transforms.json`` (o ``WBES_MM_CANONICAL_JSON``)
        se esiste e ha una voce per il modello (nome o alias), altrimenti dal frame nativo: solo
        assi e unita', nessuna traslazione. Attenzione: la similarita' del JSON (unified.py) porta
        il template medio del dominio SULLA media FLAME, scala compresa, quindi non e' un puro
        cambio di unita' (BFM 1.04e-3 invece di 1e-3); le misure in mm della libreria (bump RBF)
        usano sempre ``frame.mm_per_unit``."""
        path = Path(json_path or os.environ.get("WBES_MM_CANONICAL_JSON", "") or CANONICAL_JSON)
        entry = _json_entry(path, (self.name,) + tuple(self.aliases)) if path.is_file() else None
        if entry is not None:
            p = _parse_entry(entry, path)
            if p.get("flip_faces") is None:
                p["flip_faces"] = self.frame.normals == "inward"
            p["source"] = f"json:{path}"
            return p
        return {"scale": float(self.frame.mm_per_unit), "R": self.frame.rotation(), "t": np.zeros(3),
                "flip_faces": self.frame.normals == "inward", "source": "fallback (assi e unita' del frame nativo)"}

    def canonical_transform(self, verts: np.ndarray, faces: np.ndarray | None = None,
                            json_path: Path | str | None = None):
        """Vertici nel frame canonico in mm; con ``faces``, anche i triangoli col verso uscente."""
        p = self.canonical_params(json_path)
        Vc = p["scale"] * (np.asarray(verts, dtype=np.float64) @ p["R"].T) + p["t"]
        if faces is None:
            return Vc
        F = np.asarray(faces)
        return Vc, (np.ascontiguousarray(F[:, ::-1]) if p["flip_faces"] else F)

    # ------------------------------------------------------------------ descrizione
    def describe(self) -> dict:
        return {
            "name": self.name, "role": self.role, "region": self.region,
            "n_verts": self.n_verts, "n_faces": int(len(self.faces)),
            "n_id": self.n_id, "n_expr": self.n_expr, "expr_kind": self.expr.kind,
            "expr_pool": int(len(self.expr.pool)), "expr_scale": self.expr.scale, "expr_note": self.expr.note,
            "id_sigma_first5": [float(x) for x in self.id_sigma[:5]],
            "landmark_scheme": self.landmark_scheme,
            "n_landmarks": None if self.landmarks is None else int(len(self.landmarks)),
            "frame": {k: getattr(self.frame, k) for k in ("left", "up", "forward", "normals", "units",
                                                          "mm_per_unit", "note")},
            "info": self.info,
        }


@dataclass
class BilinearModel(MorphableModel):
    """FaceScape: V = sum_e a_e C_e @ (mu + sigma * z), espressione e identita' non separabili.

    ``core`` (3n, 52, k_file) e' ristretto alla patch; ``expr_mix`` (52, 52) porta le colonne
    della base d'espressione del file sulle 52 espressioni GREZZE (la 0 e' la neutra): identita'
    per il file 300, U_exp^T per il file 50 con PCA sulle espressioni. ``mean`` e ``id_basis``
    sono quelli della neutra (lineari nell'identita'). I pesi d'espressione dell'API sono i 51
    RESIDUI del toolkit FaceScape: a = (1 - sum w) e_0 + sum_k w_k e_k."""

    core: np.ndarray | None = None
    expr_mix: np.ndarray | None = None

    def _abs_weights(self, w: np.ndarray) -> np.ndarray:
        a = np.zeros(self.n_expr + 1)
        a[0] = 1.0 - float(np.sum(w))
        a[1:] = w
        return a

    def _identity_core(self, id_z: np.ndarray | None) -> np.ndarray:
        """(3n, 52) = core @ coefficienti nativi, nella base d'espressione GREZZA (52)."""
        w = self.native_identity(np.zeros(self.n_id) if id_z is None else id_z).astype(np.float32)
        M = (self.core @ w).astype(np.float64)                    # (3n, 52) nella base del file
        return M @ self.expr_mix                                  # colonne = espressioni grezze

    def expr_basis_at(self, id_z: np.ndarray | None = None) -> np.ndarray:
        M = self._identity_core(id_z)
        D = M[:, 1:] - M[:, :1]                                   # residui contro la neutra
        return D.reshape(self.n_verts, 3, self.n_expr)

    def mesh(self, id_z: np.ndarray | None = None, expr: np.ndarray | None = None) -> np.ndarray:
        if expr is None:
            return MorphableModel.mesh(self, id_z, None)
        e = np.asarray(expr, dtype=np.float64).ravel()
        if len(e) != self.n_expr:
            raise ValueError(f"{self.name}: {len(e)} coefficienti d'espressione, attesi {self.n_expr}")
        if id_z is not None and len(np.ravel(id_z)) != self.n_id:
            raise ValueError(f"{self.name}: {len(np.ravel(id_z))} coefficienti d'identita', attesi {self.n_id}")
        return (self._identity_core(id_z) @ self._abs_weights(e)).reshape(self.n_verts, 3)


# ---------------------------------------------------------------------- JSON dei frame

@lru_cache(maxsize=8)
def _read_json(path: str, mtime_ns: int, size: int) -> dict:
    """Il JSON dei frame, riletto solo se cambia (``canonical_transform`` si chiama per ogni mesh)."""
    return json.loads(Path(path).read_text())


def _json_entry(path: Path, names: tuple[str, ...]) -> dict | None:
    """Voce del modello in ``{"domains": {nome: voce}}`` (il formato di v3_work/unified_gt/unified.py),
    ``{"models": {...}}`` o ``{nome: voce}``."""
    st = Path(path).stat()
    data = _read_json(str(path), st.st_mtime_ns, st.st_size)
    table = (data.get("domains") or data.get("models") or data) if isinstance(data, dict) else {}
    for n in names:
        if isinstance(table.get(n), dict):
            return table[n]
    return None


def _parse_entry(e: dict, path: Path) -> dict:
    """Similarita' dal coordinate NATIVE al frame canonico in mm: ``matrix`` (4x4 o 3x4, A = s R)
    oppure ``R``/``rotation`` + ``t``/``translation`` + ``s``/``scale``; ``flip_faces`` facoltativo."""
    if "matrix" in e:
        M = np.asarray(e["matrix"], dtype=np.float64)
        A, t = M[:3, :3], M[:3, 3]
        det = float(np.linalg.det(A))
        if det <= 0:
            raise ValueError(f"{path}: matrice con determinante {det:+.3g} (riflessione o degenere)")
        s = det ** (1.0 / 3.0)
        R = A / s
    else:
        R = np.asarray(e.get("R", e.get("rotation", np.eye(3))), dtype=np.float64)
        t = np.asarray(e.get("t", e.get("translation", np.zeros(3))), dtype=np.float64)
        s = float(e.get("s", e.get("scale", 1.0)))
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-4):
        raise ValueError(f"{path}: R non e' una rotazione propria")
    flip = e.get("flip_faces")
    return {"scale": float(s), "R": R, "t": np.asarray(t, dtype=np.float64).reshape(3),
            "flip_faces": None if flip is None else bool(flip)}
