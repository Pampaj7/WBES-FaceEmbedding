"""Sorgenti delle identita' dei produttori: un dominio di training = come si campiona un'identita', le sue
espressioni e la sua forma neutra nello spazio della GT unificata.

Domini (``DOMAINS``) e taglia n_d per il bilanciamento con temperatura (p_d ~ n_d^alpha, ``domain_probs``):
  bfm2019    BFM 2019 (mesh model2019_bfm, 47.439 v.), 199 modi; topologia di lavoro decimata a ~``V_WORK``
  ict        ICT-FaceKit, patch Face (9.409 v.), 100 modi, 53 blendshape
  gnm        GNM Head, hockey_mask (9.022 v.), 170 modi, 383 espressioni pca
  flame2020  FLAME 2020, maschera "face" (1.787 v.), 300 modi, 100 espressioni pca
  famos      FaMoS, le 80 persone TRAIN di aau/famos/split.json: fotogrammi reali (espressioni vere) e neutra
n_d = modi d'identita' per i 3DMM (la dimensione del sottospazio), persone per FaMoS: con alpha = 0 (default)
i domini sono uniformi, con alpha = 1 proporzionali a n_d.

Identita' dei 3DMM: ``MorphableModel.sample_identity`` (code larghe, PLAN_MASSIVE sez. 2), espressioni dal
prior del modello (``sample_expression``). Tutti i modelli sono lineari: le basi si proiettano UNA volta sulla
topologia di lavoro e sui punti della regione unificata, quindi una mesh costa due prodotti matrice-vettore.

GT: s_i (``shapes.Space``, la stessa funzione della GT unificata in datasets/UNIFIED_GT) dalla forma NEUTRA
dell'identita': punti della mappa baricentrica del dominio (``unified_space.npz``; BFM 2019 da
``datasets/STREAM/maps/bfm2019.npz``), Procrustes di similarita' pesato verso mu, sqrt(w) * a. FaMoS: dalla
neutra di riferimento di aau/famos/famos_subsample.py (``V_neutral``), mappa ``flame``.

Frame delle viste: canonico FLAME in mm (``MorphableModel.canonical_transform``; BFM 2019 con la similarita'
della sua corrispondenza). FaMoS: ogni fotogramma allineato rigidamente (senza scala) alla patch media FLAME.

Modello da rilasciare (``build_sources(..., release=True)``, ``producer.py --release``; PLAN_MASSIVE sez. 23): anche i
3DMM di test e di sviluppo, ``RELEASE_ONLY`` (HIFI3D, FaceVerse; FaceScape, bilineare: ``MMSource`` porta sulla
topologia di lavoro la mesh completa del modello, identita' e espressione insieme), campionati con
``purpose="release"``; senza ``release`` sono vietati (ValueError) come prima (RoleError della libreria). FaMoS sul
template di FLAME 2023 Open (maschera, facce e media per l'allineamento rigido; mai FLAME 2020) e senza le persone
held-out del monitoraggio (``famos_exclude``). Preset ``release``: tutte le fonti del modello rilasciato.

Suddivisione (opt-in, ``build_sources(..., subdiv={dominio: n})``, ``producer.py --subdiv``; spenta di default): la
mesh di ogni vista di un 3DMM, nelle unita' native e prima del frame canonico, suddivisa 1-a-4 n volte coi punti
medi (``igl.upsample``, lineare: come gli insiemi FLAME di D1, ``aau/diagnostics/d1_gen.work_mesh``); la GT resta
quella dei punti della regione unificata (invariata). Nessuna estrazione dal generatore in piu'.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
UGT_DIR = REPO_ROOT / "v3_work" / "unified_gt"
for _p in (REPO_ROOT, UGT_DIR, REPO_ROOT / "v2_work" / "genict"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

STREAM_DATA = REPO_ROOT / "datasets" / "STREAM"
CACHE_DIR = STREAM_DATA / "cache"
MAPS_DIR = STREAM_DATA / "maps"
FAMOS_TRAIN = REPO_ROOT / "datasets" / "FAMOS" / "train"
FAMOS_SPLIT = REPO_ROOT / "aau" / "famos" / "split.json"
TRAIN_GT_JSON = REPO_ROOT / "datasets" / "UNIFIED_GT" / "train" / "gt_unified_bfm_ict_gnm.json"

DOMAINS = ("bfm2019", "ict", "gnm", "flame2020", "famos")
# flame2023 = FLAME 2023 Open (CC BY 4.0), stessa topologia di FLAME 2020: stessa mappa unificata
MAP_KEY = {"ict": "ict", "gnm": "gnm", "flame2020": "flame", "flame2023": "flame", "famos": "flame",
           "hifi3d": "hifi3d", "faceverse": "faceverse", "facescape": "facescape"}   # bfm2019: MAPS_DIR
# 3DMM di test (HIFI3D, FaceVerse) e di sviluppo (FaceScape): solo nel modello da rilasciare (release=True)
RELEASE_ONLY = ("hifi3d", "faceverse", "facescape")
ALL_DOMAINS = DOMAINS + ("flame2023",) + RELEASE_ONLY
# --sources: preset (literature/LICENZE_RILASCIO_2026-10-09.md). massive = max = tutte le fonti del run massivo (FLAME
# 2023 Open al posto di FLAME 2020, non ridistribuibile); validated = i domini dei bracci decisivi C3F/C3M (BFM, ICT,
# GNM; BFM 2019 al posto di BFM REMESH, che lo stream non ha); open_core = il nucleo aperto, per i pesi rilasciabili
SOURCE_PRESETS = {"massive": ("bfm2019", "ict", "gnm", "flame2023", "famos"),
                  "max": ("bfm2019", "ict", "gnm", "flame2023", "famos"), "validated": ("bfm2019", "ict", "gnm"),
                  "open_core": ("gnm", "ict", "flame2023"), "legacy": DOMAINS,
                  "release": ("bfm2019", "ict", "gnm", "flame2023", "hifi3d", "faceverse", "facescape", "famos")}
# licenza di ogni fonte e rango (0 = mesh ridistribuibili; 1 = solo ricetta e semi): una vista eredita la piu'
# restrittiva fra le fonti coinvolte (inherited_license). NON e' un parere legale.
LICENSES = {"gnm": ("Apache-2.0 (google/GNM, NOTICE)", 0), "ict": ("MIT (ICT-FaceKit Light, commit da5f95a)", 0),
            "flame2023": ("CC-BY-4.0 (FLAME 2023 Open, restrizioni d'uso)", 0),
            "flame2020": ("FLAME 2020 (MPI): solo ricetta", 1), "bfm2019": ("BFM 2019 (Basilea): solo ricetta", 1),
            "famos": ("FaMoS (MPI): solo ricetta, persone reali", 2),
            "hifi3d": ("HIFI3D++ (Tencent AI-NEXT): solo ricetta", 1),
            "faceverse": ("FaceVerse v2 (Tsinghua): solo ricetta", 1),
            "facescape": ("FaceScape bilineare (Nanjing Univ.): solo ricetta", 1)}


def parse_sources(txt: str) -> list[str]:
    """Preset (``validated``, ``max`` = ``massive``, ``open_core``, ``legacy``) o domini separati da virgola."""
    out = []
    for item in (x for x in txt.split(",") if x):
        for d in SOURCE_PRESETS.get(item, (item,)):
            if d not in ALL_DOMAINS:
                raise ValueError(f"fonte {d!r} non in {ALL_DOMAINS} ne' fra i preset {tuple(SOURCE_PRESETS)}")
            if d not in out:
                out.append(d)
    return out


def inherited_license(domains) -> tuple[str, int]:
    """La licenza piu' restrittiva fra le fonti coinvolte (le fonti senza voce valgono come le piu' restrittive)."""
    tags = [LICENSES.get(d, (f"{d}: non verificata", 9)) for d in domains]
    return max(tags, key=lambda t: t[1])
V_WORK = 10_000          # vertici della topologia di lavoro quando la patch nativa supera V_MAX
V_MAX = 12_000


def domain_probs(sizes: dict, alpha: float) -> dict:
    """p_d proporzionale a n_d^alpha (sampler_v3.domain_probs, qui sui domini dello stream)."""
    w = {d: float(n) ** float(alpha) for d, n in sizes.items() if n > 0}
    tot = sum(w.values())
    return {d: v / tot for d, v in sorted(w.items())}


def gt_scale_mm() -> float:
    """mm per unita' della GT unificata di training (il massimo di D_orig): la GT al volo usa la stessa scala."""
    return float(json.loads(TRAIN_GT_JSON.read_text())["mm_per_unit"])


def _space():
    from shapes import Space   # v3_work/unified_gt/shapes.py: la stessa funzione delle GT unificate
    return Space()


class Unified:
    """Spazio della GT unificata: mappe per dominio e s_i di una forma (n, 3) di punti mappati."""

    def __init__(self) -> None:
        self.sp = _space()
        self.n = int(len(self.sp.mu))
        self.A = float(self.sp.A)

    def map_of(self, domain: str) -> tuple[np.ndarray, np.ndarray]:
        """(vidx (n, 3), bary (n, 3)) nella numerazione NATIVA completa del modello del dominio."""
        if domain in MAP_KEY:
            k = MAP_KEY[domain]
            return self.sp.z[f"vidx_{k}"], self.sp.z[f"bary_{k}"]
        path = MAPS_DIR / f"{domain}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{domain}: mappa unificata assente ({path}); per BFM 2019: "
                                    "v3_work/stream/bfm2019_map.py")
        with np.load(path) as z:
            return z["vidx"], z["bary"]

    def svec(self, P: np.ndarray) -> np.ndarray:
        """s_i (3n,) float32 dai punti mappati (n, 3), in qualunque frame e unita' (il Procrustes li toglie)."""
        a, _, _ = self.sp.align(np.asarray(P, dtype=np.float64)[None])
        return self.sp.svec(a)[0]


def _bary_embed(Vq: np.ndarray, V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Punto piu' vicino di ogni Vq sulla superficie (V, F): (vertici del triangolo (q, 3), baricentriche)."""
    import igl
    _, I, C = igl.point_mesh_squared_distance(Vq, V, F.astype(np.int64))
    tri = F[I].astype(np.int64)
    # baricentriche del punto piu' vicino C nel suo triangolo (formula delle aree via prodotti scalari)
    a, b, c = (V[tri[:, k]] for k in range(3))
    v0, v1, v2 = b - a, c - a, C - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    B = np.clip(np.stack([1.0 - w1 - w2, w1, w2], axis=1), 0.0, 1.0)
    return tri, B / B.sum(1, keepdims=True)


def work_topology(model, v_work: int = V_WORK) -> dict:
    """Topologia di lavoro FISSA di un modello con patch troppo densa: la media decimata (igl.qslim, come i
    generatori) a ~2 v_work triangoli, i vertici incollati alla superficie media con baricentriche. Una mesh
    del modello si porta sulla topologia con le stesse baricentriche (lineare, quindi anche le basi).
    In cache in datasets/STREAM/cache."""
    import mesh_ops as mo
    path = CACHE_DIR / f"{model.name}_work{v_work}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    U, G = mo.decimate_to(model.mean, model.faces, 2 * v_work)
    U, G = mo.prepare_open_surface(U, G)
    tri, B = _bary_embed(U, model.mean, model.faces)
    out = {"tri": tri, "bary": B, "faces": G.astype(np.int32), "n_full": np.int64(model.n_verts)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(path)
    return out


def _full_unified(name: str, vidx: np.ndarray, bary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(media (n, 3), base d'identita' (n, 3, k)) sui punti della regione unificata dal modello INTERO (la mappa e'
    nella numerazione nativa completa, come le GT di E12), nella convenzione dei coefficienti della libreria
    (v3_work/mm/loaders.py). Solo i 3DMM del modello da rilasciare la cui patch non copre la regione."""
    from v3_work.mm import loaders as LD
    path = LD.model_path(name)
    u, inv = np.unique(vidx, return_inverse=True)
    inv = inv.reshape(vidx.shape)
    if name == "hifi3d":
        import hifi_model
        m = hifi_model.load_hifi(path)
        mean, basis = m["v_template"][u], m["shapedirs"][u]
    elif name == "faceverse":
        import fv_model
        m = fv_model.load_fv(path)
        mean, basis = m["v_template"][u], m["shapedirs"][u]
    elif name == "facescape":              # load_facescape: neutra = fetta 0 del core, z = (w - id_mean) / sigma
        with np.load(path, allow_pickle=True) as z:
            C = np.asarray(z["shape_bm_core"][LD._rows(u)][:, 0, :], dtype=np.float64)     # (3u, k)
            id_mean, id_sigma = z["id_mean"].astype(np.float64), np.sqrt(z["id_var"].astype(np.float64))
        mean, basis = (C @ id_mean).reshape(len(u), 3), (C * id_sigma).reshape(len(u), 3, -1)
    else:
        raise ValueError(f"{name}: regione unificata fuori dalla patch e nessun modello intero")
    return _project(np.asarray(mean, np.float64), inv, bary), _project(np.asarray(basis, np.float64), inv, bary)


def _project(basis: np.ndarray, tri: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """Base (n, 3, k) (o media (n, 3)) portata sui punti baricentrici (q, 3, k)."""
    if basis.ndim == 2:
        return np.einsum("qkd,qk->qd", basis[tri], bary)
    return np.einsum("qkdj,qk->qdj", basis[tri], bary)


class MMSource:
    """Un 3DMM lineare della libreria v3_work/mm, ruolo train. ``identity`` -> (z, s_i); ``view`` -> mesh canonica."""

    kind = "mm"

    def __init__(self, name: str, unified: Unified, v_max: int = V_MAX, v_work: int = V_WORK, subdiv: int = 0,
                 purpose: str = "train") -> None:
        from v3_work.mm import BilinearModel, load_for_training, load_model
        if purpose == "release":       # modello da rilasciare: anche i 3DMM dev e test (RELEASE_ONLY)
            m = load_model(name)
        elif purpose == "train":
            m = load_for_training(name)
        else:
            raise ValueError(f"{name}: purpose {purpose!r} (train | release)")
        # bilineare (FaceScape): neutra lineare nell'identita' (mean, id_basis), espressione dalla mesh completa
        self.bilinear = isinstance(m, BilinearModel)
        if self.bilinear and subdiv:
            raise ValueError(f"{name}: suddivisione non supportata sul modello bilineare")
        self.purpose = purpose
        self._core = None                  # bilineare: (chiave dell'identita', core dell'identita') dell'ultima vista
        self.name, self.model = name, m
        self.n_id = m.n_id
        self.subdiv = int(subdiv)          # suddivisioni 1-a-4 di ogni vista (0 = nessuna, il default)
        # punti della regione unificata: dalla numerazione nativa alla posizione nella patch
        vidx, bary = unified.map_of(name)
        rv = np.asarray(m.region_vertices if m.region_vertices is not None else np.arange(m.n_verts))
        pos = np.searchsorted(rv, vidx)
        inside = (pos < len(rv)) & (rv[np.minimum(pos, len(rv) - 1)] == vidx)
        if inside.all():
            self.mean_u, self.id_u = _project(m.mean, pos, bary), _project(m.id_basis, pos, bary)
        elif purpose == "release":     # HIFI3D, FaceScape: la regione unificata esce dalla patch (mappa sul modello intero)
            self.mean_u, self.id_u = _full_unified(name, vidx, bary)
            ok = inside.all(axis=1)    # controllo: i punti coi tre vertici nella patch uguali al calcolo sulla patch
            pm = _project(m.mean, np.minimum(pos[ok], len(rv) - 1), bary[ok])
            pb = _project(m.id_basis, np.minimum(pos[ok], len(rv) - 1), bary[ok])
            tol = 1e-6 * float(np.abs(m.mean).max())
            if not ok.any() or np.abs(pm - self.mean_u[ok]).max() > tol or np.abs(pb - self.id_u[ok]).max() > tol:
                raise ValueError(f"{name}: punti unificati dal modello intero diversi da quelli della patch")
            self.unified_outside = int((~ok).sum())
        else:
            raise ValueError(f"{name}: vertici della mappa unificata fuori dalla patch del modello")
        # topologia di lavoro: la patch se e' abbastanza piccola, altrimenti la media decimata
        if m.n_verts <= v_max:
            self.work = None
            self.faces = m.faces
            self.mean_w, self.id_w, self.ex_w = m.mean, m.id_basis, (None if self.bilinear else m.expr.basis)
        else:
            wt = work_topology(m, v_work)
            self.work = wt
            self.faces = wt["faces"]
            self.mean_w = _project(m.mean, wt["tri"], wt["bary"])
            self.id_w = _project(m.id_basis, wt["tri"], wt["bary"])
            self.ex_w = None if self.bilinear else _project(m.expr.basis, wt["tri"], wt["bary"])
        self.canon = self._canonical(unified)
        if self.canon["flip_faces"]:
            self.faces = np.ascontiguousarray(self.faces[:, ::-1])

    def _canonical(self, unified: Unified) -> dict:
        p = self.model.canonical_params()
        if p["source"].startswith("fallback") and (MAPS_DIR / f"{self.name}.npz").exists():
            with np.load(MAPS_DIR / f"{self.name}.npz") as z:      # BFM 2019: similarita' della corrispondenza
                if "canon_s" in z.files:
                    p = {"scale": float(z["canon_s"]), "R": z["canon_R"], "t": z["canon_t"],
                         "flip_faces": bool(z["canon_flip"]), "source": f"corrispondenza {MAPS_DIR / self.name}.npz"}
        return p

    def identity(self, rng: np.random.Generator) -> np.ndarray:
        return self.model.sample_identity(rng, purpose=self.purpose)

    def neutral_points(self, z: np.ndarray) -> np.ndarray:
        """Punti della regione unificata della forma neutra (n, 3), unita' native."""
        return self.mean_u + self.id_u @ np.asarray(z, dtype=np.float64)

    def view_mesh(self, z: np.ndarray, rng: np.random.Generator, expr: bool) -> tuple[np.ndarray, np.ndarray, str]:
        """(V canonica in mm, F, etichetta d'espressione) di una vista dell'identita' ``z``."""
        V = self.mean_w + self.id_w @ np.asarray(z, dtype=np.float64)
        tag = "neutral"
        if expr:
            e = self.model.sample_expression(rng, purpose=self.purpose)
            V = self._bilinear_expr(z, e) if self.bilinear else V + self.ex_w @ e
            tag = "expr"
        F = self.faces
        if self.subdiv:
            V, F = self.upsample(V)
        c = self.canon
        return c["scale"] * (V @ np.asarray(c["R"]).T) + np.asarray(c["t"]), F, tag

    def _bilinear_expr(self, z: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Bilineare: la mesh completa del modello (identita' ``z``, espressione ``e``) sulla topologia di lavoro. Il core
        dell'identita' (BilinearModel._identity_core) si calcola una volta per identita' (le viste del gruppo)."""
        m = self.model
        key = np.asarray(z, dtype=np.float64).tobytes()
        if self._core is None or self._core[0] != key:
            self._core = (key, m._identity_core(z))
        V = (self._core[1] @ m._abs_weights(np.asarray(e, dtype=np.float64))).reshape(m.n_verts, 3)
        return V if self.work is None else _project(V, self.work["tri"], self.work["bary"])

    def upsample(self, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(V, F) suddivisi 1-a-4 ``self.subdiv`` volte coi punti medi (igl.upsample, come d1_gen.work_mesh)."""
        import igl
        Vs, Fs = igl.upsample(np.asarray(V, dtype=np.float64), np.asarray(self.faces, dtype=np.int64), self.subdiv)
        return Vs, np.ascontiguousarray(Fs, dtype=np.asarray(self.faces).dtype)

    def describe(self) -> dict:
        out = {"name": self.name, "kind": self.kind, "n_id": self.n_id, "n_verts_work": int(len(self.mean_w)),
               "n_faces_work": int(len(self.faces)), "work_topology": "patch" if self.work is None else "decimata",
               "canonical": self.canon["source"]}
        if self.purpose != "train":
            out.update(purpose=self.purpose, role=self.model.role, bilinear=self.bilinear,
                       unified_outside_patch=getattr(self, "unified_outside", 0))
        if self.subdiv:
            Vs, Fs = self.upsample(self.mean_w)
            out.update(subdiv=self.subdiv, n_verts_view=int(len(Vs)), n_faces_view=int(len(Fs)))
        return out


def _npz_memmap(path: Path, key: str) -> np.ndarray:
    """Memmap di un membro NON compresso di un npz (data_v3.npz_member_memmap): la page cache e' condivisa fra
    i processi del nodo, quindi 80 persone FaMoS (3 GB) stanno in RAM una volta sola."""
    import zipfile
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f"{key}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            with np.load(path) as z:
                return z[key]
        with open(path, "rb") as fh:
            fh.seek(info.header_offset)
            local = fh.read(30)
            n_name, n_extra = int.from_bytes(local[26:28], "little"), int.from_bytes(local[28:30], "little")
            fh.seek(info.header_offset + 30 + n_name + n_extra)
            version = np.lib.format.read_magic(fh)
            read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = read(fh)
            off = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=off, shape=shape, order="F" if fortran else "C")


class FamosSource:
    """FaMoS, sole persone TRAIN: un'identita' e' una persona, le sue viste sono la neutra di riferimento o un
    fotogramma registrato (espressione reale), sulla maschera "face" di FLAME 2020 (``template``; il modello da
    rilasciare usa FLAME 2023 Open, stessa topologia e maschera). ``exclude``: persone TRAIN tenute fuori (held-out
    del monitoraggio del modello da rilasciare)."""

    kind = "famos"

    def __init__(self, unified: Unified, template: str = "flame2020", exclude=()) -> None:
        from v3_work.mm import load_model
        if template not in ("flame2020", "flame2023"):
            raise ValueError(f"FaMoS: template {template!r} (flame2020 | flame2023)")
        split = json.loads(FAMOS_SPLIT.read_text())
        exclude = sorted(set(exclude))
        bad = sorted(set(exclude) - set(split["train"]))
        if bad:
            raise ValueError(f"FaMoS: persone da escludere fuori da TRAIN: {bad}")
        self.template, self.excluded = template, exclude
        self.persons = sorted(set(split["train"]) - set(exclude))
        test = set(split["test"])
        files = {p: FAMOS_TRAIN / f"{p}.npz" for p in self.persons}
        missing = [p for p, f in files.items() if not f.exists()]
        if missing:
            raise FileNotFoundError(f"FaMoS: mancano {missing[:3]} in {FAMOS_TRAIN}")
        if test & set(self.persons):
            raise SystemExit("FaMoS: persone di TEST nello split di training")
        self.files = files
        self.n_id = len(self.persons)
        flame = load_model(template)
        self.rv = np.asarray(flame.region_vertices)
        p = flame.canonical_params()
        self.faces = flame.faces[:, ::-1].copy() if p["flip_faces"] else flame.faces
        self.target = p["scale"] * (flame.mean @ np.asarray(p["R"]).T) + np.asarray(p["t"])   # patch media, mm
        self.vidx, self.bary = unified.map_of("famos")
        self._mm: dict = {}

    def _data(self, person: str) -> tuple[np.ndarray, np.ndarray]:
        d = self._mm.get(person)
        if d is None:
            d = (_npz_memmap(self.files[person], "V"), _npz_memmap(self.files[person], "V_neutral"))
            self._mm[person] = d
        return d

    def identity(self, rng: np.random.Generator) -> str:
        return self.persons[int(rng.integers(len(self.persons)))]

    def neutral_points(self, person: str) -> np.ndarray:
        Vn = np.asarray(self._data(person)[1], dtype=np.float64)
        return np.einsum("nkd,nk->nd", Vn[self.vidx], self.bary)

    def view_mesh(self, person: str, rng: np.random.Generator, expr: bool) -> tuple[np.ndarray, np.ndarray, str]:
        from ugt import apply_sim, umeyama
        V, Vn = self._data(person)
        if expr:
            i = int(rng.integers(len(V)))
            X, tag = np.asarray(V[i], dtype=np.float64)[self.rv], "expr"
        else:
            i = -1
            X, tag = np.asarray(Vn, dtype=np.float64)[self.rv], "neutral"
        self.last_frame = i            # provenienza: indice del fotogramma registrato (-1 = neutra di riferimento)
        return apply_sim(X, *umeyama(X, self.target, scale=False)), self.faces, tag

    def describe(self) -> dict:
        out = {"name": "famos", "kind": self.kind, "n_id": self.n_id, "n_verts_work": int(len(self.rv)),
               "n_faces_work": int(len(self.faces)), "persons": len(self.persons)}
        if self.template != "flame2020" or self.excluded:
            out.update(template=self.template, excluded=self.excluded)
        return out


def build_sources(domains, unified: Unified | None = None, v_max: int = V_MAX, v_work: int = V_WORK,
                  subdiv: dict | None = None, release: bool = False, famos_exclude=()) -> dict:
    """Sorgenti dei domini; ``subdiv`` {dominio: n} suddivide le viste di quei 3DMM (opt-in, vuoto = nessuna).
    ``release``: modello da rilasciare (RELEASE_ONLY con purpose release, FaMoS su FLAME 2023 Open senza
    ``famos_exclude``); senza, i domini RELEASE_ONLY sono un errore."""
    gated = sorted(set(domains) & set(RELEASE_ONLY))
    if gated and not release:
        raise ValueError(f"{gated}: 3DMM di test/sviluppo, solo nel modello da rilasciare (release=True, "
                         "producer.py --release)")
    if famos_exclude and not release:
        raise ValueError("famos_exclude solo con release=True")
    unified = unified or Unified()
    subdiv = {d: int(n) for d, n in (subdiv or {}).items() if int(n) > 0}
    bad = sorted(set(subdiv) - (set(domains) - {"famos"}))
    if bad:
        raise ValueError(f"suddivisione per {bad}: non sono 3DMM fra le fonti {list(domains)}")
    out = {}
    for d in domains:
        if d not in ALL_DOMAINS:
            raise ValueError(f"dominio {d!r} non in {ALL_DOMAINS}")
        if d == "famos":
            out[d] = FamosSource(unified, **({"template": "flame2023", "exclude": famos_exclude} if release else {}))
        else:
            out[d] = MMSource(d, unified, v_max=v_max, v_work=v_work, **({"subdiv": subdiv[d]} if d in subdiv else {}),
                              **({"purpose": "release"} if release and d in RELEASE_ONLY else {}))
    return out
