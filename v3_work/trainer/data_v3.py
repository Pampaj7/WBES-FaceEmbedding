"""Dati del trainer v3: sorgenti, blocchi residenti in RAM, cache, GT, trasformazioni al servizio.

Fork di v2_work/fastio/train_steps.py (BlockedDataset, collect_sources, partition_blocks, mem_log) e di
v2_work/fastio/fast_data.py (CachedDataset). Con i flag di default ogni campione servito e' IDENTICO a
quello della catena v2 (stesso loader congelato GTReadyDatasetNPZ, stessa ricostruzione degli sparsi).

Aggiunte, tutte spente di default:
  * ``input_norm="sqrt_area"``: vertici nel frame centroide pesato per massa + sqrt(area totale), applicato
    al campione PULITO (come ``train_fast --frame area``); le perturbazioni del training arrivano dopo,
    con le stesse sigma, quindi in unita' del nuovo frame;
  * ``input_norm="global"``: mm nel frame canonico divisi per una costante (global_v3.GlobalFrame), stesso punto;
  * ``compact=True``: nella cache facce e indici COO in int32 e niente L (la diffusione spettrale non lo
    legge). Senza perdita: al servizio gli indici tornano int64 e L e' uno sparso vuoto (n, n);
  * ``shard=(rank, world)``: ogni processo DDP tiene in RAM solo i soggetti del proprio shard.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch

from common import PREPASS, REPO_ROOT, domain_of, log0, split_name

from dataset_gtready import GTReadyDatasetNPZ  # noqa: E402  (loader congelato, sola lettura)
from intrinsic_utils import SUBJECT_RE_ANY, extract_subject_id  # noqa: E402

TENSOR_KEYS = ("verts", "mass", "evals", "evecs", "faces", "gradX", "gradY", "L")
THIS_PREPASS_VEC = Path(__file__).resolve().parent / "prepass_v3.py"   # prepass_ops con grad_vec (E9)
LIVE_PROCS: set = set()


def kill_proc(proc) -> None:
    """Chiude un pre-pass in corso con tutto il suo gruppo (il Pool di prepass_ops ha processi figli)."""
    if proc is None or proc.poll() is not None:
        LIVE_PROCS.discard(proc)
        return
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    LIVE_PROCS.discard(proc)


def kill_all() -> None:
    for p in list(LIVE_PROCS):
        kill_proc(p)
SPARSE_KEYS = ("L", "gradX", "gradY")


# --- sorgenti e blocchi (come train_steps.py) -----------------------------------------------------

def collect_sources(spec: dict) -> dict[str, tuple[str, str]]:
    """nome file -> (tipo, sorgente) per ogni mesh della spec, filtrata per etichetta."""
    labels = set(spec["labels"]) if spec.get("labels") else None
    out: dict[str, tuple[str, str]] = {}

    def add(name: str, kind: str, src: str) -> None:
        if not name.endswith(".npz") or "_GTready_" not in name:
            return
        if labels is not None and split_name(name)[1] not in labels:
            return
        if name in out:
            raise SystemExit(f"{name} compare in due sorgenti: {out[name][1]} e {src}")
        out[name] = (kind, src)

    for d in spec.get("views", []):
        for n in sorted(os.listdir(d)):
            add(n, "view", str(Path(d) / n))
    if spec.get("tars") and spec.get("tar_index"):
        with np.load(spec["tar_index"]) as z:
            tar_names, tar_id, names = [str(x) for x in z["tars"]], z["tar_id"], z["names"]
        wanted_tars = {Path(t).name: str(t) for t in spec["tars"]}
        missing = sorted(set(wanted_tars) - set(tar_names))
        if missing:
            raise SystemExit(f"tar_index {spec['tar_index']}: mancano {missing[:3]}")
        for n, i in zip(names, tar_id):
            t = tar_names[int(i)]
            if t in wanted_tars:
                add(str(n), "tar", wanted_tars[t])
    else:
        for t in spec.get("tars", []):
            with tarfile.open(t) as tar:
                for n in tar.getnames():
                    add(n, "tar", str(t))
    for d in spec.get("geom_dirs", []):
        for n in sorted(os.listdir(d)):
            add(n, "geom", str(Path(d) / n))
    return out


def partition_blocks(train: list[str], spec: dict) -> list[list[str]]:
    """Blocchi di soggetti di training, IDENTICA a train_steps.partition_blocks (domini pinned in ogni
    blocco, il resto permutato con block_seed, a strisce, eventualmente stratificato per dominio)."""
    rng = np.random.default_rng(int(spec.get("block_seed", 0)))
    pinned_dom = set(spec.get("pin_domains") or [])
    pinned = sorted(s for s in train if domain_of(s) in pinned_dom)
    pinned_set = set(pinned)
    rest = [s for s in train if s not in pinned_set]
    K = int(spec.get("n_blocks", 1))
    if not spec.get("stratify_blocks"):
        perm = rng.permutation(np.array(rest, dtype=object)).tolist()
        return [sorted(perm[k::K] + pinned) for k in range(K)]
    blocks: list[list[str]] = [list(pinned) for _ in range(K)]
    for dom in sorted({domain_of(s) for s in rest}):
        pool = sorted(s for s in rest if domain_of(s) == dom)
        perm = rng.permutation(np.array(pool, dtype=object)).tolist()
        for k in range(K):
            blocks[k] += perm[k::K]
    return [sorted(b) for b in blocks]


def shard_subjects(subjects: Sequence[str], rank: int, world: int, seed: int) -> list[str]:
    """Shard DDP stratificato per dominio: ogni rank riceve ~1/world dei soggetti di OGNI dominio.
    Con world=1 restituisce la lista invariata (stesso ordine)."""
    if world <= 1:
        return list(subjects)
    rng = np.random.default_rng(int(seed) + 7_919)
    out: list[str] = []
    for dom in sorted({domain_of(s) for s in subjects}):
        pool = sorted(s for s in subjects if domain_of(s) == dom)
        perm = rng.permutation(np.array(pool, dtype=object)).tolist()
        out += perm[rank::world]
    return sorted(out)


# --- memoria ----------------------------------------------------------------------------------------

def _cgroup_dir() -> Path | None:
    """Cgroup di memoria del JOB (quello su cui Slurm applica --mem), v1 o v2."""
    try:
        for line in open("/proc/self/cgroup"):
            _, ctrl, path = line.strip().split(":", 2)
            if ctrl in ("memory", ""):
                for base in (Path("/sys/fs/cgroup/memory"), Path("/sys/fs/cgroup")):
                    p = base / path.lstrip("/")
                    while p != base and not p.name.startswith("job_"):
                        p = p.parent
                    if p.name.startswith("job_") and p.exists():
                        return p
    except OSError:
        pass
    return None


def mem_log(what: str) -> None:
    rss = float("nan")
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) / 2 ** 20
    cg, cur, peak = _cgroup_dir(), float("nan"), float("nan")
    if cg is not None:
        for f, var in (("memory.usage_in_bytes", "cur"), ("memory.current", "cur"),
                       ("memory.max_usage_in_bytes", "peak"), ("memory.peak", "peak")):
            q = cg / f
            if q.exists():
                v = int(q.read_text().split()[0]) / 2 ** 30
                if var == "cur":
                    cur = v
                else:
                    peak = v
    print(f"[mem] {what}: RSS processo {rss:.1f} GiB, cgroup job {cur:.1f} GiB (picco {peak:.1f})", flush=True)


def malloc_trim() -> None:
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


# --- cache in RAM (fork di fast_data.CachedDataset) --------------------------------------------------

class _CompactSparse:
    """COO con indici int32 (senza perdita: gli indici sono < 2^31) per la cache compatta."""

    __slots__ = ("indices", "values", "shape")

    def __init__(self, t: torch.Tensor | None) -> None:
        if t is None:
            return
        c = t.coalesce()
        self.indices = c.indices().to(torch.int32)
        self.values = c.values()
        self.shape = tuple(c.shape)

    @classmethod
    def from_parts(cls, indices: torch.Tensor, values: torch.Tensor, shape) -> "_CompactSparse":
        o = cls(None)
        o.indices, o.values, o.shape = indices, values, tuple(shape)
        return o

    def nbytes(self) -> int:
        return self.indices.numel() * 4 + self.values.numel() * self.values.element_size()


def _sample_bytes(sample: Dict[str, object]) -> int:
    total = 0
    for k in TENSOR_KEYS:
        t = sample.get(k)
        if isinstance(t, _CompactSparse):
            total += t.nbytes()
        elif torch.is_tensor(t):
            if t.is_sparse:
                c = t.coalesce()
                total += c.indices().numel() * c.indices().element_size()
                total += c.values().numel() * c.values().element_size()
            else:
                total += t.numel() * t.element_size()
    return total


def _to_residency(sample: Dict[str, torch.Tensor], pin: bool, compact: bool) -> Dict[str, object]:
    out: Dict[str, object] = dict(sample)
    if compact:
        out["faces"] = sample["faces"].to(torch.int32)
        out.pop("L", None)
        for k in ("gradX", "gradY"):
            out[k] = _CompactSparse(sample[k])
        return out
    for k in TENSOR_KEYS:
        t = out.get(k)
        if torch.is_tensor(t) and pin and not t.is_sparse:
            out[k] = t.pin_memory()
    return out


FAST = {"on": False}   # --fast-data: sparsi ricostruiti senza riordino (gli indici in cache sono gia' coalescenti)


def _coo(indices: torch.Tensor, values: torch.Tensor, shape) -> torch.Tensor:
    if FAST["on"]:
        return torch.sparse_coo_tensor(indices, values, shape, is_coalesced=True)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def _serve(sample: Dict[str, object]) -> Dict[str, torch.Tensor]:
    """Campione servito al trainer: sparsi ricostruiti (vedi fast_data._rebuild_sparse: serve perche'
    l'eval gira in inference_mode) e, dalla cache compatta, int64 e L vuoto. Con FAST gli stessi tensori
    senza il .coalesce() (un ordinamento per matrice a ogni lettura)."""
    out = dict(sample)
    if isinstance(sample.get("gradX"), _CompactSparse):
        out["faces"] = sample["faces"].long()
        for k in ("gradX", "gradY"):
            c = sample[k]
            out[k] = _coo(c.indices.long(), c.values, c.shape)
        n = int(sample["verts"].shape[0])
        out["L"] = torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long), torch.zeros(0), (n, n)).coalesce()
        return out
    for k in SPARSE_KEYS:
        t = sample.get(k)
        if torch.is_tensor(t) and t.is_sparse:
            out[k] = _coo(t.indices(), t.values(), t.shape)
    return out


class CachedDataset:
    """Campioni preparati dal loader congelato, residenti in RAM. Superficie: files, __len__, __getitem__."""

    def __init__(self, data_dir: str | Path, indices: Iterable[int] | None = None, workers: int = 16,
                 pin: bool = False, compact: bool = False, verbose: bool = True) -> None:
        self._base = GTReadyDatasetNPZ(str(data_dir))
        self.files: Sequence[str] = self._base.files
        self.data_dir = str(data_dir)
        want = sorted(set(range(len(self.files)) if indices is None else (int(i) for i in indices)))
        self._cache: Dict[int, Dict[str, object]] = {}
        if not want:
            self.load_seconds = 0.0
            return
        t0 = time.time()

        def load(i: int):
            return i, _to_residency(self._base[i], pin, compact)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for n, (i, s) in enumerate(ex.map(load, want), start=1):
                self._cache[i] = s
                if verbose and n % 2000 == 0:
                    rate = n / (time.time() - t0)
                    print(f"[cache] {n}/{len(want)} ({rate:.0f}/s, eta {(len(want) - n) / max(rate, 1e-9):.0f}s)",
                          flush=True)
        self.load_seconds = time.time() - t0
        if verbose:
            gib = sum(_sample_bytes(s) for s in self._cache.values()) / 2 ** 30
            print(f"[cache] {len(self._cache)} campioni da {data_dir} in {self.load_seconds:.0f}s, "
                  f"{gib:.1f} GiB in RAM ({'compatta' if compact else 'come v2'})", flush=True)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self._cache.get(int(idx))
        if s is None:
            return self._base[int(idx)]
        return _serve(s)


# --- store in mmap, condiviso fra processi (tools/build_store.py) ------------------------------------

STORE_FIELDS = (("verts", np.float32), ("faces", np.int32), ("mass", np.float32), ("evals", np.float32),
                ("evecs", np.float32), ("gxi", np.int32), ("gxv", np.float32), ("gyi", np.int32), ("gyv", np.float32))


def store_shapes(n: int, m: int, k: int, nx: int, ny: int, evecs_f: bool = False) -> dict:
    """Forme su disco. ``evecs_f``: evecs scritto come (k, n) contiguo, cioe' la memoria di un (n, k) in ordine
    colonna, che e' il layout del loader (stride (1, n)); si serve con .t(), senza copia."""
    return {"verts": (n, 3), "faces": (m, 3), "mass": (n,), "evals": (k,), "evecs": (k, n) if evecs_f else (n, k),
            "gxi": (2, nx), "gxv": (nx,), "gyi": (2, ny), "gyv": (ny,)}


class MmapStore:
    """Campioni GIA' preparati dal loader congelato, in formato compatto, in shard binari letti con memmap.

    Piu' processi sullo stesso nodo leggono gli stessi file: il kernel ne tiene UNA copia nella page cache.
    ``compact_sample(i)`` restituisce viste numpy zero-copy (memmap in copy-on-write, mai scritto) nel formato
    della cache compatta, che data_v3._serve trasforma nei tensori del loader (identici, tools/build_store.py
    li verifica a campione, valori e stride).

    Lo stride conta: sotto TF32 (acceso dal container) cuBLAS sceglie il kernel dal layout, quindi evecs in ordine
    riga invece che colonna cambia gli arrotondamenti e la traiettoria si separa da v2 (misurato: Z del passo 1
    a 2e-5). Gli store senza ``evecs_f`` nell'indice (scritti prima della correzione) servono evecs in ordine riga."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        z = np.load(self.root / "index.npz")
        self.names = [str(x) for x in z["names"]]
        self.shard = z["shard"]
        self.dims = {k: z[k] for k in ("n", "m", "k", "nx", "ny")}
        self.off = {f: z[f"off_{f}"] for f, _ in STORE_FIELDS}
        if "evecs_f" in z.files:
            self.evecs_f = z["evecs_f"].astype(bool)
        else:
            self.evecs_f = np.zeros(len(self.names), dtype=bool)
            print(f"[store] AVVISO: {self.root} senza evecs_f (formato vecchio): evecs in ordine riga, non nel layout "
                  "del loader; la traiettoria non e' identica a v2 sotto TF32", flush=True)
        self._mm: dict = {}

    def _map(self, k: int):
        mm = self._mm.get(k)
        if mm is None:
            mm = np.memmap(self.root / "shards" / f"shard_{k:03d}.bin", dtype=np.uint8, mode="c")
            self._mm[k] = mm
        return mm

    def compact_sample(self, i: int) -> Dict[str, object]:
        mm = self._map(int(self.shard[i]))
        ef = bool(self.evecs_f[i])
        sh = store_shapes(*(int(self.dims[k][i]) for k in ("n", "m", "k", "nx", "ny")), evecs_f=ef)
        a = {f: torch.from_numpy(np.ndarray(sh[f], dtype=dt, buffer=mm, offset=int(self.off[f][i])))
             for f, dt in STORE_FIELDS}
        n = sh["verts"][0]
        evecs = a["evecs"].t() if ef else a["evecs"]
        return {"verts": a["verts"], "faces": a["faces"], "mass": a["mass"], "evals": a["evals"], "evecs": evecs,
                "gradX": _CompactSparse.from_parts(a["gxi"], a["gxv"], (n, n)),
                "gradY": _CompactSparse.from_parts(a["gyi"], a["gyv"], (n, n)),
                "name": self.names[i][:-4]}


class StoreDataset:
    """Dataset sullo store: tutte le mesh sempre disponibili (nessuno staging). ``resident_block`` resta come
    stato del campionatore a blocchi, che sceglie i soggetti dell'epoca come in v2."""

    def __init__(self, store: MmapStore, transform) -> None:
        self.store = store
        self.files = list(store.names)
        self.transform = transform
        self.resident_block = -1

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        return self.transform(self.files[int(idx)], _serve(self.store.compact_sample(int(idx))))


# --- trasformazioni al servizio ------------------------------------------------------------------

def _total_area(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    tri = V[F.long()]
    return 0.5 * torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1).sum()


def reframe_sqrt_area(V: torch.Tensor, mass: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Centroide pesato per massa e scala sqrt(area totale): copia di v2_work/pointnet/frames.reframe('area').

    Non dipende dal frame d'ingresso (il loader ha gia' fatto centro-vertici e maxabs: traslazione e
    scala uniforme si cancellano), quindi e' uguale a ri-inquadrare la mesh grezza."""
    w = mass.reshape(-1).to(V.dtype).clamp_min(0)
    tot = w.sum()
    if not torch.isfinite(tot) or float(tot) <= 0:
        raise ValueError("massa degenere: impossibile il frame sqrt(area)")
    w = w / tot
    X = V - (w.unsqueeze(1) * V).sum(0, keepdim=True)
    s = torch.sqrt(_total_area(V, faces))
    if not torch.isfinite(s) or float(s) <= 1e-12:
        raise ValueError("area degenere: impossibile il frame sqrt(area)")
    return X / s


def augment_verts(V, aug: dict, rng: np.random.Generator):
    """Rotazione casuale (asse uniforme, angolo <= rot_deg) e riflessione x->-x (train_steps.augment_verts)."""
    R = np.eye(3)
    a = float(aug.get("rot_deg", 0.0))
    if a > 0:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        th = np.deg2rad(rng.uniform(-a, a))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    if rng.uniform() < float(aug.get("reflect_p", 0.0)):
        R = R @ np.diag([-1.0, 1.0, 1.0])
    return V @ torch.as_tensor(R.T, dtype=V.dtype, device=V.device)


class ServeTransform:
    """input_norm (tutti i campioni) e aug della spec (solo soggetti di training), in quest'ordine.

    sqrt_area con ``area_weights`` mass: data_v3.reframe_sqrt_area (centroide pesato per massa, sqrt dell'area
    totale). Con smooth/winsor: area_v3.area_frame (pesi robusti per centro e scala); centro e scala si
    calcolano una volta per campione e si memorizzano (dipendono solo dalla mesh pulita servita)."""

    def __init__(self, input_norm: str = "maxabs", aug: dict | None = None, aug_seed: int = 0,
                 area_weights: str = "mass", global_frame=None) -> None:
        if input_norm not in ("maxabs", "sqrt_area", "global"):
            raise ValueError(f"input_norm {input_norm!r}")
        if (input_norm == "global") != (global_frame is not None):
            raise ValueError("input_norm global va con un global_v3.GlobalFrame (e solo con quello)")
        self.input_norm = input_norm
        self.area_weights = area_weights
        self.global_frame = global_frame
        self.aug = aug
        self.aug_rng = np.random.default_rng(int(aug_seed))
        self.train_set: set[str] = set()
        self._frames: dict = {}

    @property
    def identity(self) -> bool:
        return self.input_norm == "maxabs" and not self.aug

    def __call__(self, name: str, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.identity:
            return sample
        sample = dict(sample)
        if self.input_norm == "sqrt_area" and self.area_weights == "mass":
            sample["verts"] = reframe_sqrt_area(sample["verts"], sample["mass"], sample["faces"])
        elif self.input_norm == "sqrt_area":
            import area_v3
            fr = self._frames.get(name)
            if fr is None:
                V = sample["verts"]
                w = area_v3.area_weights(self.area_weights, V, sample["faces"], sample["mass"], sample["evecs"])
                fr = ((w.unsqueeze(1) * V).sum(0, keepdim=True) / w.sum(), torch.sqrt(w.sum()))
                self._frames[name] = fr
            sample["verts"] = (sample["verts"] - fr[0]) / fr[1]
        elif self.input_norm == "global":       # mm nel frame canonico / L0 (global_v3.py)
            sample = self.global_frame(name, sample)
        if self.aug and split_name(name)[0] in self.train_set:
            sample["verts"] = augment_verts(sample["verts"], self.aug, self.aug_rng)
        return sample


class DirDataset:
    """Una directory piatta di npz con operatori (``--data_dir`` senza data-spec), in RAM o letta dal disco."""

    def __init__(self, data_dir: str, transform: ServeTransform, cache: bool, workers: int, compact: bool,
                 subjects: set[str] | None = None) -> None:
        base = GTReadyDatasetNPZ(str(data_dir))
        self.files = base.files
        self.transform = transform
        if cache:
            idx = None if subjects is None else [i for i, f in enumerate(self.files) if split_name(f)[0] in subjects]
            self._ds = CachedDataset(data_dir, indices=idx, workers=workers, compact=compact)
        else:
            self._ds = base

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        return self.transform(self.files[int(idx)], self._ds[int(idx)])


class BlockedDataset:
    """Fork di train_steps.BlockedDataset: ``files`` e' l'elenco COMPLETO delle mesh, in RAM c'e' solo il
    blocco residente piu' l'insieme dell'eval online. Un campione fuori da entrambi e' un errore."""

    def __init__(self, sources: dict[str, tuple[str, str]], cfg: dict, transform: ServeTransform) -> None:
        self.files = sorted(sources)
        self.sources = sources
        self.cfg = cfg
        self.transform = transform
        self._pos = {n: i for i, n in enumerate(self.files)}
        self._parts: list = []        # [(CachedDataset, {idx_globale: idx_locale}), ...]
        self.resident_block = -1

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        for ds, local in self._parts:
            j = local.get(int(idx))
            if j is not None:
                return self.transform(self.files[int(idx)], ds[j])
        raise KeyError(f"{self.files[int(idx)]}: non e' nel blocco residente ({self.resident_block}) "
                       "ne' nell'eval online")

    def names_of(self, subjects: Iterable[str]) -> list[str]:
        subj = set(subjects)
        return [n for n in self.files if split_name(n)[0] in subj]

    def stage(self, names: list[str], dest: Path, wait: bool = True):
        """Vista piatta in ``dest``: symlink per le viste, pre-pass per tar e geometria."""
        dest.mkdir(parents=True, exist_ok=True)
        # temporanei di un pre-pass ucciso a meta' scrittura (riavvio elastico di torchrun): il loader li leggerebbe
        # come mesh (finiscono in .npz); il pre-pass scrive tmp + rename, quindi sono solo detriti
        for stale in dest.glob(".*.tmp.npz"):
            stale.unlink(missing_ok=True)
        need_tars, need_geom, subjects = set(), set(), set()
        canon = self.cfg.get("canon") or {}
        for n in names:
            kind, src = self.sources[n]
            if kind == "view" and domain_of(split_name(n)[0]) in canon:
                raise SystemExit(f"canon per {domain_of(split_name(n)[0])}: {n} arriva da una vista")
            if kind == "view":
                p = dest / n
                if not p.exists():
                    p.symlink_to(Path(src).resolve())
            else:
                (need_tars if kind == "tar" else need_geom).add(src if kind == "tar" else str(Path(src).parent))
                subjects.add(split_name(n)[0])
        if not subjects:
            return None
        subj_file = dest.parent / f"{dest.name}.subjects.txt"
        subj_file.write_text("\n".join(sorted(subjects)) + "\n")
        labels = ",".join(self.cfg["labels"]) if self.cfg.get("labels") else ""
        script = THIS_PREPASS_VEC if self.cfg.get("prepass_grad") == "vec" else PREPASS
        cmd = [sys.executable, str(script), "--compress", "--out-dir", str(dest), "--subjects", str(subj_file),
               "--convention", self.cfg.get("convention", "areanorm"),
               "--n-proc", str(self.cfg["prepass_proc"])]
        if need_tars:
            cmd += ["--tars", *sorted(need_tars)]
        if need_geom:
            cmd += ["--geom-dirs", *sorted(need_geom)]
        if labels:
            cmd += ["--labels", labels]
        if canon:
            cmd += ["--transform", json.dumps(canon)]
        if self.cfg.get("tar_index") and need_tars:
            cmd += ["--tar-index", str(self.cfg["tar_index"])]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        log = open(dest.parent / f"{dest.name}.prepass.log", "w")
        # gruppo di processi proprio: kill_proc lo chiude con i figli del Pool (riavvii di torchrun, SIGTERM)
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        LIVE_PROCS.add(proc)
        if wait:
            self.finish(proc, dest)
            return None
        return proc

    def finish(self, proc, dest: Path) -> None:
        if proc is None:
            return
        t0 = time.time()
        rc = proc.wait()
        LIVE_PROCS.discard(proc)
        if rc == 1 and int(self.cfg.get("tolerate", 0)) > 0:
            print(f"[v3] AVVISO: pre-pass {dest.name} con mesh fallite (rc=1): si contano al caricamento "
                  f"(tolleranza {self.cfg['tolerate']}), log in {dest.parent / (dest.name + '.prepass.log')}",
                  flush=True)
            return
        if rc != 0:
            raise SystemExit(f"pre-pass di {dest} fallito (rc={rc}), log in {dest.parent / (dest.name + '.prepass.log')}")
        print(f"[v3] pre-pass {dest.name} pronto (atteso {time.time() - t0:.0f}s)", flush=True)

    def block_gib(self, dest: Path, names: list[str]) -> float:
        """GiB della cache del blocco nel formato v2 (indice per i tar, header per le viste): la compatta
        ne occupa meno, quindi come tetto e' conservativo."""
        sys.path.insert(0, str(REPO_ROOT / "aau/data_scale"))
        from cache_budget import load_index, npz_sample_bytes, predicted_bytes
        if "view_bytes" not in self.cfg and self.cfg.get("view_bytes_json"):
            self.cfg["view_bytes"] = {k: int(v) for k, v in json.loads(Path(self.cfg["view_bytes_json"]).read_text()).items()}
        vb = self.cfg.setdefault("view_bytes", {})
        if any(self.sources[n][0] == "tar" for n in names) and "index" not in self.cfg:
            idx = load_index(Path(self.cfg["tar_index"]))
            self.cfg["index"] = (idx, {str(n): i for i, n in enumerate(idx["names"])})
        total = 0
        for n in names:
            if self.sources[n][0] == "tar":
                idx, pos = self.cfg["index"]
                i = pos[n]
                total += predicted_bytes(int(idx["n"][i]), int(idx["m"][i]), int(idx["E"][i]))
            else:
                if n not in vb:
                    vb[n] = npz_sample_bytes(dest / n)
                total += vb[n]
        return total / 2 ** 30

    def load(self, dest: Path, names: list[str]):
        """(cache, mappa indici, mesh mancanti scartate). Mancanti oltre la tolleranza: errore, come v2."""
        c = self.cfg
        missing = [n for n in names if not (dest / n).exists()]
        if missing and len(missing) > int(c.get("tolerate", 0)):
            raise SystemExit(f"{dest}: {len(missing)} mesh mancanti dopo il pre-pass (prima {missing[0]})")
        if missing:
            log = dest.parent / f"{dest.name}.prepass.log"
            tail = log.read_text()[-3000:] if log.exists() else ""
            print(f"[v3] AVVISO: {len(missing)} mesh mancanti dopo il pre-pass, scartate: {missing[:10]}\n"
                  f"[v3] coda del log del pre-pass: {tail}", flush=True)
            names = [n for n in names if n not in set(missing)]
        gib = self.block_gib(dest, names)
        print(f"[v3] {dest.name}: {len(names)} campioni, cache {gib:.1f} GiB nel formato v2 "
              f"(tetto {c['cache_max_gb']:.0f} GiB)", flush=True)
        if gib > c["cache_max_gb"]:
            raise SystemExit(f"{dest.name}: cache {gib:.1f} GiB > --cache-max-gb {c['cache_max_gb']:.0f}")
        ds = CachedDataset(dest, workers=c["cache_workers"], pin=c["pin"], compact=c["compact"])
        local = {self._pos[n]: j for j, n in enumerate(ds.files) if n in self._pos}
        return ds, local, missing

    def drop_parts_after(self, keep: int) -> None:
        """Libera i blocchi oltre i primi ``keep`` (l'eval online resta): gc + malloc_trim come in v2."""
        while len(self._parts) > keep:
            old = self._parts.pop()
            del old
        gc.collect()
        malloc_trim()


# --- GT ----------------------------------------------------------------------------------------------

class NanGuardedMatrix(np.ndarray):
    """GT le cui voci non definite (NaN, fra domini) non si possono leggere in silenzio (train_v2)."""

    def __getitem__(self, key):
        out = super().__getitem__(key)
        arr = np.asarray(out, dtype=np.float64)
        if arr.size and not np.isfinite(arr).all():
            raise AssertionError("read an undefined (NaN) GT entry: the batch or eval set spans domains "
                                 f"(index={key!r})")
        return np.asarray(out) if isinstance(out, np.ndarray) else out


class VectorGT:
    """GT come vettori per identita' (formato PROVVISORIO, in attesa di datasets/UNIFIED_GT/): npz con
    ``names`` [n] e ``S`` [n, d] float32, g_ij = ||S_i - S_j|| * ``scale`` (scalare opzionale, default 1).
    Si indicizza come la matrice (``gt[np.ix_(a, b)]``), quindi passa da tutto il codice v1 di eval."""

    def __init__(self, S: np.ndarray, scale: float = 1.0) -> None:
        self.S = np.asarray(S, dtype=np.float64)
        self.scale = float(scale)
        self.shape = (self.S.shape[0], self.S.shape[0])

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("VectorGT si legge solo come gt[np.ix_(righe, colonne)]")
        r, c = (np.asarray(k).reshape(-1) for k in key)
        diff = self.S[r][:, None, :] - self.S[c][None, :, :]
        return np.sqrt((diff * diff).sum(-1)) * self.scale


def npz_member_memmap(path: Path, key: str) -> np.ndarray:
    """Memmap di un membro NON compresso di un npz (D_orig da 17 GB senza leggerlo tutto)."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f"{key}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{key} e' compresso: niente memmap")
        with open(path, "rb") as fh:
            fh.seek(info.header_offset)
            local = fh.read(30)
            n_name, n_extra = int.from_bytes(local[26:28], "little"), int.from_bytes(local[28:30], "little")
            data_off = info.header_offset + 30 + n_name + n_extra
            fh.seek(data_off)
            version = np.lib.format.read_magic(fh)
            read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                    else np.lib.format.read_array_header_2_0)
            shape, fortran, dtype = read(fh)
            arr_off = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=arr_off, shape=shape, order="F" if fortran else "C")


def load_gt(path: str, keep_scale: bool, vector: bool = False):
    """(gt, name_to_idx). Matrice: D_orig con nomi letti da SUBJECT_RE_ANY (come il wrapper NaN di train_v2).

    keep_scale: D_orig cosi' com'e' (train_steps --gt-keep-scale), tenuta in float32: lo stesso valore che
    v2 leggeva in float64 e poi riportava a float32 nel tensore della loss (conversione esatta, meta' RAM).
    Altrimenti la lettura v1 (divisione per il massimo delle voci positive) in float64.
    """
    pack = np.load(path, allow_pickle=True)
    name_to_idx = {}
    for i, n in enumerate(pack["names"]):
        n = n.decode("utf-8", errors="ignore") if isinstance(n, bytes) else n
        sid = extract_subject_id(str(n), subject_re=SUBJECT_RE_ANY)
        if sid is not None:
            name_to_idx[sid] = i
    if not name_to_idx:
        raise RuntimeError(f"Could not parse subject ids from names in {path}")
    if vector:
        gt = VectorGT(pack["S"], float(pack["scale"]) if "scale" in pack.files else 1.0)
        log0(f"[v3] GT vettoriale {Path(path).name}: {gt.S.shape} (formato provvisorio)")
        return gt, name_to_idx
    if keep_scale:
        D = np.ascontiguousarray(pack["D_orig"], dtype=np.float32)
    else:
        D = pack["D_orig"].astype(np.float64)
        mask = D > 0
        if mask.any():
            D = D / float(D[mask].max())
    n_nan = int(np.count_nonzero(~np.isfinite(D)))
    log0(f"[v3] GT {Path(path).name}: {D.shape} {'scala invariata' if keep_scale else 'divisa per il massimo'}, "
         f"max={float(np.nanmax(D)):.4f}, voci non definite (NaN)={n_nan}")
    return D.view(NanGuardedMatrix), name_to_idx


def check_gt_scale(dist_npz: str, keep_scale: bool) -> None:
    """Come train_steps.check_gt_scale: GT con massimo > 1 (scala ICT-5000) richiede --gt-keep-scale."""
    side = Path(dist_npz).with_suffix(".json")
    if not side.exists():
        return
    gmax = float(json.loads(side.read_text()).get("global_max", 1.0))
    if gmax > 1.0 + 1e-6 and not keep_scale:
        raise SystemExit(f"GT {Path(dist_npz).name}: massimo {gmax:.4f} > 1; serve --gt-keep-scale")
