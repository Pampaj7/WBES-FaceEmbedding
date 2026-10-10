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
MAP_KEY = {"ict": "ict", "gnm": "gnm", "flame2020": "flame", "flame2023": "flame", "famos": "flame"}   # bfm2019: MAPS_DIR
ALL_DOMAINS = DOMAINS + ("flame2023",)
# --sources: preset (literature/LICENZE_RILASCIO_2026-10-09.md). massive = max = tutte le fonti del run massivo (FLAME
# 2023 Open al posto di FLAME 2020, non ridistribuibile); validated = i domini dei bracci decisivi C3F/C3M (BFM, ICT,
# GNM; BFM 2019 al posto di BFM REMESH, che lo stream non ha); open_core = il nucleo aperto, per i pesi rilasciabili
SOURCE_PRESETS = {"massive": ("bfm2019", "ict", "gnm", "flame2023", "famos"),
                  "max": ("bfm2019", "ict", "gnm", "flame2023", "famos"), "validated": ("bfm2019", "ict", "gnm"),
                  "open_core": ("gnm", "ict", "flame2023"), "legacy": DOMAINS}
# licenza di ogni fonte e rango (0 = mesh ridistribuibili; 1 = solo ricetta e semi): una vista eredita la piu'
# restrittiva fra le fonti coinvolte (inherited_license). NON e' un parere legale.
LICENSES = {"gnm": ("Apache-2.0 (google/GNM, NOTICE)", 0), "ict": ("MIT (ICT-FaceKit Light, commit da5f95a)", 0),
            "flame2023": ("CC-BY-4.0 (FLAME 2023 Open, restrizioni d'uso)", 0),
            "flame2020": ("FLAME 2020 (MPI): solo ricetta", 1), "bfm2019": ("BFM 2019 (Basilea): solo ricetta", 1),
            "famos": ("FaMoS (MPI): solo ricetta, persone reali", 2)}


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


def _project(basis: np.ndarray, tri: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """Base (n, 3, k) (o media (n, 3)) portata sui punti baricentrici (q, 3, k)."""
    if basis.ndim == 2:
        return np.einsum("qkd,qk->qd", basis[tri], bary)
    return np.einsum("qkdj,qk->qdj", basis[tri], bary)


class MMSource:
    """Un 3DMM lineare della libreria v3_work/mm, ruolo train. ``identity`` -> (z, s_i); ``view`` -> mesh canonica."""

    kind = "mm"

    def __init__(self, name: str, unified: Unified, v_max: int = V_MAX, v_work: int = V_WORK) -> None:
        from v3_work.mm import BilinearModel, load_for_training
        m = load_for_training(name)
        if isinstance(m, BilinearModel):
            raise ValueError(f"{name}: modello bilineare, non supportato dallo stream")
        self.name, self.model = name, m
        self.n_id = m.n_id
        # punti della regione unificata: dalla numerazione nativa alla posizione nella patch
        vidx, bary = unified.map_of(name)
        rv = np.asarray(m.region_vertices if m.region_vertices is not None else np.arange(m.n_verts))
        pos = np.searchsorted(rv, vidx)
        if (pos >= len(rv)).any() or not np.array_equal(rv[np.minimum(pos, len(rv) - 1)], vidx):
            raise ValueError(f"{name}: vertici della mappa unificata fuori dalla patch del modello")
        self.mean_u, self.id_u = _project(m.mean, pos, bary), _project(m.id_basis, pos, bary)
        # topologia di lavoro: la patch se e' abbastanza piccola, altrimenti la media decimata
        if m.n_verts <= v_max:
            self.work = None
            self.faces = m.faces
            self.mean_w, self.id_w, self.ex_w = m.mean, m.id_basis, m.expr.basis
        else:
            wt = work_topology(m, v_work)
            self.work = wt
            self.faces = wt["faces"]
            self.mean_w = _project(m.mean, wt["tri"], wt["bary"])
            self.id_w = _project(m.id_basis, wt["tri"], wt["bary"])
            self.ex_w = _project(m.expr.basis, wt["tri"], wt["bary"])
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
        return self.model.sample_identity(rng)

    def neutral_points(self, z: np.ndarray) -> np.ndarray:
        """Punti della regione unificata della forma neutra (n, 3), unita' native."""
        return self.mean_u + self.id_u @ np.asarray(z, dtype=np.float64)

    def view_mesh(self, z: np.ndarray, rng: np.random.Generator, expr: bool) -> tuple[np.ndarray, np.ndarray, str]:
        """(V canonica in mm, F, etichetta d'espressione) di una vista dell'identita' ``z``."""
        V = self.mean_w + self.id_w @ np.asarray(z, dtype=np.float64)
        tag = "neutral"
        if expr:
            V = V + self.ex_w @ self.model.sample_expression(rng)
            tag = "expr"
        c = self.canon
        return c["scale"] * (V @ np.asarray(c["R"]).T) + np.asarray(c["t"]), self.faces, tag

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind, "n_id": self.n_id, "n_verts_work": int(len(self.mean_w)),
                "n_faces_work": int(len(self.faces)), "work_topology": "patch" if self.work is None else "decimata",
                "canonical": self.canon["source"]}


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
    fotogramma registrato (espressione reale), sulla maschera "face" di FLAME 2020."""

    kind = "famos"

    def __init__(self, unified: Unified) -> None:
        from v3_work.mm import load_model
        split = json.loads(FAMOS_SPLIT.read_text())
        self.persons = sorted(split["train"])
        test = set(split["test"])
        files = {p: FAMOS_TRAIN / f"{p}.npz" for p in self.persons}
        missing = [p for p, f in files.items() if not f.exists()]
        if missing:
            raise FileNotFoundError(f"FaMoS: mancano {missing[:3]} in {FAMOS_TRAIN}")
        if test & set(self.persons):
            raise SystemExit("FaMoS: persone di TEST nello split di training")
        self.files = files
        self.n_id = len(self.persons)
        flame = load_model("flame2020")
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
        return {"name": "famos", "kind": self.kind, "n_id": self.n_id, "n_verts_work": int(len(self.rv)),
                "n_faces_work": int(len(self.faces)), "persons": len(self.persons)}


def build_sources(domains, unified: Unified | None = None, v_max: int = V_MAX, v_work: int = V_WORK) -> dict:
    unified = unified or Unified()
    out = {}
    for d in domains:
        if d not in ALL_DOMAINS:
            raise ValueError(f"dominio {d!r} non in {ALL_DOMAINS}")
        out[d] = FamosSource(unified) if d == "famos" else MMSource(d, unified, v_max=v_max, v_work=v_work)
    return out
