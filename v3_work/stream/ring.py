"""Buffer ad anello degli shard: una directory su /tmp (tmpfs, cioe' RAM del job), un file per shard.

    <root>/shards/<seq 10 cifre>.shard     shard completi, comparsi con rename atomico
    <root>/tmp/                            shard in scrittura
    <root>/seq                             contatore globale dei numeri di sequenza (flock)
    <root>/stats/                          statistiche dei produttori (producer.py)

Anello = FIFO a budget di byte: dopo ogni scrittura il produttore cancella gli shard piu' vecchi finche' il
totale sta nel budget. Niente letture strappate e niente lock lato lettore: lo shard compare intero (tmp +
rename) e chi l'ha gia' mappato continua a leggerlo anche dopo la cancellazione (l'inode vive finche' resta
mappato); la RAM torna libera quando l'ultimo lettore lo lascia.

Formato: ``MAGIC`` (8 byte), lunghezza dell'header (8 byte, little endian), header JSON, poi gli array allineati
a 64 byte. Header: ``seq``, ``producer``, ``t_created``, ``groups``: per gruppo (un'identita') ``key``,
``domain``, ``s`` (offset di s_i, float32 (3n,)), ``views``: per vista i metadati di views.make_view e gli
offset degli array (``off``). Con ``producer.py --canonical-gt`` il gruppo ha anche ``fr`` e ``sr`` (offset,
float32 (3n,)) e ``S`` (centroid size in mm), e ogni vista ``area_mm2``. Con ``--provenance``: ``recipe`` nello
header, ``prov`` (seme, fonti, licenza ereditata, persona) e ``zid`` (coefficienti, float64) per gruppo, ``vi``,
``noise_seed``, ``frame`` per vista; ``origin`` (pura, ibrido, trasferimento) sempre. Con ``--partial-p`` ogni vista ha
``partial`` (partial_aug.apply: seme, modo, parametri, perdita d'area).
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np

MAGIC = b"WBESSTR1"
ALIGN = 64
DTYPES = {"verts": np.float32, "faces": np.int32, "mass": np.float32, "evals": np.float32, "gxi": np.int32,
          "gxv": np.float32, "gyi": np.int32, "gyv": np.float32}
EVECS = {"fp16": np.float16, "fp32": np.float32}
TARGETS = ("fr", "sr")       # vettori per gruppo (float32, lunghi come s), opzionali


def shapes(v: dict) -> dict:
    n, m, k, nx, ny = (int(v[x]) for x in ("n", "m", "k", "nx", "ny"))
    return {"verts": (n, 3), "faces": (m, 3), "mass": (n,), "evals": (k,), "evecs": (k, n) if v["evecs_f"] else (n, k),
            "gxi": (2, nx), "gxv": (nx,), "gyi": (2, ny), "gyv": (ny,)}


def dtype_of(field: str, v: dict):
    return EVECS[v["evecs_dtype"]] if field == "evecs" else DTYPES[field]


class Ring:
    def __init__(self, root: str | Path, budget_bytes: int = 0, evict_every: int = 1, by_count: bool = False) -> None:
        self.root = Path(root)
        self.shards = self.root / "shards"
        self.tmp = self.root / "tmp"
        self.budget = int(budget_bytes)
        # su CephFS (anello condiviso) il listing con stat di migliaia di shard costa: il budget si applica ogni N
        # scritture di questo processo (lo sforamento resta di qualche shard per produttore)
        self.evict_every = max(1, int(evict_every))
        self._writes = 0
        # by_count (anello condiviso da centinaia di produttori): budget come NUMERO di shard = budget / dimensione
        # media degli shard scritti da questo processo, contato coi soli nomi (niente stat): si applica a ogni
        # scrittura e lo sforamento resta di pochi shard (con evict_every su CephFS arrivava a ~7 GiB su 40)
        self.by_count = bool(by_count)
        self._avg = 0.0
        for d in (self.shards, self.tmp, self.root / "stats"):
            d.mkdir(parents=True, exist_ok=True)

    # --- scrittura -------------------------------------------------------------------------------------
    def next_seq(self) -> int:
        with open(self.root / "seq", "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            txt = fh.read().strip()
            seq = int(txt) + 1 if txt else 0
            fh.seek(0)
            fh.truncate()
            fh.write(str(seq))
            fh.flush()
        return seq

    def write(self, groups: list[dict], producer: int, recipe: dict | None = None) -> tuple[int, int]:
        """Scrive uno shard (``groups``: key, domain, s, views [(array, meta)]) e applica il budget. (seq, byte).
        ``recipe``: i parametri per rigenerare i gruppi dal seme (producer.py --provenance)."""
        seq = self.next_seq()
        pos = 0
        head = {"seq": seq, "producer": int(producer), "t_created": time.time(), "groups": []}
        if recipe is not None:
            head["recipe"] = recipe
        pending: list[tuple[int, bytes]] = []

        def put(a: np.ndarray) -> int:
            nonlocal pos
            pos += (-pos) % ALIGN
            off = pos
            b = np.ascontiguousarray(a).tobytes()
            pending.append((off, b))
            pos += len(b)
            return off

        for g in groups:
            hg = {"key": g["key"], "domain": g["domain"], "s": put(np.asarray(g["s"], dtype=np.float32)),
                  "s_len": int(len(g["s"])), "views": []}
            for f in TARGETS:          # bersagli della GT di E12 (targets.py), solo se il produttore li calcola
                if f in g:
                    hg[f] = put(np.asarray(g[f], dtype=np.float32))
            if "S" in g:
                hg["S"] = float(g["S"])
            if "origin" in g:          # pura, ibrido, trasferimento (producer.py --mm-aug)
                hg["origin"] = str(g["origin"])
            if "prov" in g:            # seme, fonti, licenza ereditata, persona (producer.py --provenance)
                hg["prov"] = g["prov"]
            if "zid" in g:             # coefficienti d'identita' del 3DMM (float64)
                hg["zid"] = put(np.asarray(g["zid"], dtype=np.float64))
                hg["zid_len"] = int(len(g["zid"]))
            for arr, meta in g["views"]:
                v = {k: meta[k] for k in ("label", "expr", "n", "m", "k", "nx", "ny", "evecs_f", "evecs_dtype")}
                if "area_mm2" in meta:
                    v["area_mm2"] = float(meta["area_mm2"])
                for k in ("vi", "noise_seed", "frame"):
                    if k in meta:
                        v[k] = int(meta[k])
                if "partial" in meta:      # parzialita' variabile (producer.py --partial-p)
                    v["partial"] = meta["partial"]
                v["off"] = {}
                for f in ("verts", "faces", "mass", "evals", "evecs", "gxi", "gxv", "gyi", "gyv"):
                    a = np.asarray(arr[f])
                    if a.dtype != dtype_of(f, v) or tuple(a.shape) != shapes(v)[f]:
                        raise ValueError(f"{f}: {a.dtype} {a.shape}, attesi {dtype_of(f, v)} {shapes(v)[f]}")
                    v["off"][f] = put(a)
                hg["views"].append(v)
            head["groups"].append(hg)
        hb = json.dumps(head).encode()
        data_start = 16 + len(hb)
        data_start += (-data_start) % ALIGN
        tmp = self.tmp / f"{seq:010d}.{os.getpid()}"
        with open(tmp, "wb") as fh:
            fh.write(MAGIC + int(data_start).to_bytes(8, "little") + hb)
            fh.write(b"\0" * (data_start - 16 - len(hb)))
            cur = 0
            for off, b in pending:
                if off > cur:
                    fh.write(b"\0" * (off - cur))
                fh.write(b)
                cur = off + len(b)
            nbytes = data_start + cur
        os.replace(tmp, self.shards / f"{seq:010d}.shard")
        self._writes += 1
        self._avg = nbytes if self._avg == 0 else 0.9 * self._avg + 0.1 * nbytes
        if self._writes % self.evict_every == 0:
            self.evict()
        return seq, nbytes

    def listing(self) -> list[tuple[int, int, Path]]:
        """(seq, byte, percorso) degli shard presenti, dal piu' vecchio."""
        out = []
        for e in os.scandir(self.shards):
            if e.name.endswith(".shard"):
                try:
                    out.append((int(e.name[:-6]), e.stat().st_size, Path(e.path)))
                except FileNotFoundError:     # cancellato da un altro produttore fra scandir e stat
                    continue
        return sorted(out)

    def seqs(self) -> dict:
        """{seq: percorso} degli shard presenti, senza stat (il consumatore non ha bisogno delle dimensioni)."""
        return {int(e.name[:-6]): Path(e.path) for e in os.scandir(self.shards) if e.name.endswith(".shard")}

    def evict(self) -> int:
        """Cancella gli shard piu' vecchi oltre il budget (ne resta sempre almeno uno). Shard cancellati."""
        if self.budget <= 0:
            return 0
        if self.by_count and self._avg > 0:
            names = sorted(self.seqs().items())
            extra = len(names) - max(1, int(self.budget / self._avg))
            n = 0
            for _, p in names[:max(0, min(extra, len(names) - 1))]:
                try:
                    p.unlink()
                    n += 1
                except FileNotFoundError:
                    pass
            return n
        lst = self.listing()
        total = sum(b for _, b, _ in lst)
        n = 0
        for seq, b, p in lst[:-1]:
            if total <= self.budget:
                break
            try:
                p.unlink()
                n += 1
            except FileNotFoundError:
                pass
            total -= b
        return n


class ShardReader:
    """Uno shard in mmap (sola lettura). ``view(g, v)`` restituisce viste numpy sulla mappa: chi le usa le copia."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        if bytes(self.mm[:8]) != MAGIC:
            raise ValueError(f"{self.path}: non e' uno shard dello stream")
        start = int.from_bytes(bytes(self.mm[8:16]), "little")
        self.head = json.loads(bytes(self.mm[16:start]).rstrip(b"\0").decode())
        self.start = start
        self.seq = int(self.head["seq"])

    @property
    def groups(self) -> list[dict]:
        return self.head["groups"]

    def _arr(self, off: int, dtype, shape) -> np.ndarray:
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        a = self.start + int(off)
        return np.ndarray(shape, dtype=dtype, buffer=self.mm[a:a + n])

    def s(self, g: int) -> np.ndarray:
        hg = self.groups[g]
        return self._arr(hg["s"], np.float32, (hg["s_len"],))

    def target(self, g: int, field: str) -> np.ndarray:
        """``fr`` o ``sr`` del gruppo (targets.py); KeyError se il produttore non li ha calcolati."""
        hg = self.groups[g]
        if field not in hg:
            raise KeyError(f"{self.path}: lo shard non ha il bersaglio {field!r} (producer.py --canonical-gt)")
        return self._arr(hg[field], np.float32, (hg["s_len"],))

    def zid(self, g: int) -> np.ndarray | None:
        hg = self.groups[g]
        return self._arr(hg["zid"], np.float64, (hg["zid_len"],)) if "zid" in hg else None

    def view(self, g: int, v: int) -> dict:
        meta = self.groups[g]["views"][v]
        sh = shapes(meta)
        return {f: self._arr(off, dtype_of(f, meta), sh[f]) for f, off in meta["off"].items()}
