"""Consumatore della pipeline P1: batch del trainer v3 dagli shard dell'anello, GT del batch al volo.

Un ``StreamConsumer`` per rank DDP. Legge gli shard in mmap (``ring.ShardReader``): sono file su tmpfs, quindi
gli 8 rank di un nodo condividono UNA copia (la page cache), e nessuno copia lo shard intero, solo le viste
che usa. I rank si dividono gli shard (``seq % world == rank``): ogni identita' finisce a un solo rank.

Riuso: ogni gruppo (un'identita' con le sue viste) si estrae al massimo ``reuse`` volte; a ogni uso le sue
viste ricevono una rotazione (yaw, pitch, roll uniformi entro ``rot_deg``) e una scala (1 +- ``scale``) nuove,
senza ricalcolare gli operatori, che sono intrinseci. Se gli eleggibili non bastano per un batch si prendono i
gruppi meno usati oltre il tetto (``over_reuse``, contato): la GPU non aspetta mai i produttori, salvo
all'avvio (``min_groups``). Il fattore di riuso misurato e' usi / viste distinte usate.

Ogni vista servita e' il campione del loader (centro e maxabs, autovalori normalizzati, gradienti scalati: li
ha gia' calcolati il produttore, views.serve_like_loader), negli stessi tipi e layout (autovettori in ordine
colonna, sparsi COO int64), poi ``input_norm`` del trainer e l'augmentation. I campioni del batch successivo si
preparano in thread mentre la GPU lavora sul corrente (``prefetch``).

GT: ``StreamGT`` ha l'interfaccia della matrice GT del trainer (``gt[np.ix_(righe, colonne)]`` e
``name_to_idx``), con g_ij = ||s_i - s_j|| / sqrt(A) / ``gt_mm``: la GT unificata di datasets/UNIFIED_GT
(train_gt.py), nella sua scala (mm per unita' = il suo massimo, 14.16 mm) se ``gt_mm`` non e' dato.
Con ``gt_kind`` fr o sr (shard di ``producer.py --canonical-gt``) il vettore e' quello della GT di E12
(targets.py: FR in mm, SR adimensionale), stessa formula; ``StreamGT.log_s`` tiene log S_i (testa fattorizzata).

Ingresso globale (``input_norm`` global): X = ((V - c) f) / L0 come global_v3 (centro coi pesi ``area_weights``,
f = sqrt(area_mm2 / area(V)) con ``area_mm2`` dai metadati della vista), senza R_d: le viste sono gia' nel frame
canonico. Poi la rotazione; la scala dell'augmentation deve essere 0 (cambierebbe la taglia senza il bersaglio).

Su piu' nodi ogni nodo ha il suo anello: gli shard si dividono fra i rank DEL NODO (``shard_rank``,
``shard_world`` = rank e numero di rank locali), non fra tutti.

Anelli in piu' (``extra_rings``: [(directory, rank, world)]): per esempio un anello CONDIVISO su CephFS, scritto da
produttori solo CPU su altri nodi e letto da tutti i rank di tutti i nodi (diviso col rank globale). I loro shard
entrano nello stesso pool con un numero di sequenza negativo, -(i * EXTRA_OFFSET + seq), e la directory si rilegge
al massimo ogni ``extra_refresh_s`` secondi (scandir su CephFS). Le statistiche seq_* restano dell'anello locale.
Con ``extra_mirror`` (una directory locale, su /tmp) un thread copia in RAM gli shard del rank (i piu' nuovi per
primi) e il consumatore legge le copie: letto direttamente da CephFS, il primo accesso a ogni gruppo (readahead del
client) costava ~2 s per passo nel thread principale (misurato il 9 ottobre, trial_2gpu).

Fonti (``domains``): solo i gruppi con TUTTE le fonti fra quelle (template, seconda identita' degli ibridi, fonte
dell'espressione trasferita) entrano nel pool (es. il nucleo aperto da un anello con tutto).
Registro delle viste usate (``log_views``): alla PRIMA estrazione di ogni vista una riga con dominio, origine,
licenza, seme del gruppo, indice nella ricetta, discretizzazione, espressione, fotogramma, seme del rumore,
vertici, area, S_i; ``flush_log`` scrive un npz compresso a colonne compatte (~64 byte per vista prima della
compressione).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (THIS_DIR, REPO_ROOT / "v3_work" / "trainer"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ring import Ring, ShardReader  # noqa: E402

EXTRA_OFFSET = 10 ** 12


class StreamGT:
    """GT unificata dai vettori s_i delle identita' del batch, con l'interfaccia della matrice del trainer."""

    def __init__(self, area_total: float, gt_mm: float, keep: int = 8192) -> None:
        self.norm = float(np.sqrt(area_total)) * float(gt_mm)
        self.keep = int(keep)
        self.name_to_idx: dict = {}
        self.domain: dict = {}
        self.log_s: dict = {}                    # chiave -> log S_i (mm), se il produttore l'ha calcolato
        self._key_of: dict = {}                  # riga -> chiave
        self._s: OrderedDict = OrderedDict()     # riga -> s_i float64, in ordine d'uso (si potano le piu' vecchie)
        self._next = 0
        self.shape = (np.inf, np.inf)

    def register(self, key: str, s: np.ndarray, domain: str, log_s: float | None = None) -> int:
        row = self.name_to_idx.get(key)
        if row is None:
            row = self._next
            self._next += 1
            self.name_to_idx[key] = row
            self._key_of[row] = key
            self.domain[key] = domain
            self._s[row] = np.asarray(s, dtype=np.float64).copy()
            if log_s is not None:
                self.log_s[key] = float(log_s)
        self._s.move_to_end(row)
        while len(self._s) > self.keep:
            old, _ = self._s.popitem(last=False)
            k = self._key_of.pop(old)
            del self.name_to_idx[k]
            self.domain.pop(k, None)
            self.log_s.pop(k, None)
        return row

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("StreamGT si legge solo come gt[np.ix_(righe, colonne)]")
        r, c = (np.asarray(k).reshape(-1) for k in key)
        u, inv = np.unique(np.concatenate([r, c]), return_inverse=True)
        S = np.stack([self._s[int(i)] for i in u])
        D = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)) / self.norm     # diagonale esattamente 0
        return D[np.ix_(inv[:len(r)], inv[len(r):])]


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """R = Ry(yaw) Rx(pitch) Rz(roll), gradi, nel frame canonico (+y alto, +z fuori dal volto)."""
    a, b, c = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


class StreamConsumer:
    def __init__(self, root: str | Path, rank: int = 0, world: int = 1, reuse: float = 4.0, seed: int = 0,
                 input_norm: str = "maxabs", area_weights: str = "mass", rot_deg=(30.0, 15.0, 10.0),
                 scale: float = 0.1, min_groups: int = 0, wait_s: float = 900.0, prefetch: int = 4,
                 gt: StreamGT | None = None, gt_kind: str = "unified", global_unit_mm: float = 100.0,
                 global_ops: str = "areanorm", shard_rank: int | None = None, shard_world: int | None = None,
                 extra_rings=(), extra_refresh_s: float = 10.0, domains=None, log_views: bool = False,
                 extra_mirror: str | Path = "") -> None:
        self.ring = Ring(root)
        self.extra = [(Ring(r), int(er), int(ew)) for r, er, ew in extra_rings]
        self.extra_refresh_s = float(extra_refresh_s)
        self._extra_present: dict = {}
        self._extra_t = -1e18
        self.domains = set(domains) if domains else None
        self.log_views = bool(log_views)
        self._log_rows: list = []
        self.rank, self.world = int(rank), int(world)
        self.mirror = None
        if extra_mirror and self.extra:
            self.mirror = [Ring(Path(extra_mirror) / f"rank{self.rank:03d}_x{i}") for i in range(1, len(self.extra) + 1)]
            self.mstat = {"copied": 0, "bytes": 0, "seconds": 0.0, "missed": 0, "removed": 0}
            threading.Thread(target=self._mirror_loop, daemon=True).start()
        self.shard_rank = self.rank if shard_rank is None else int(shard_rank)
        self.shard_world = self.world if shard_world is None else int(shard_world)
        if gt_kind not in ("unified", "fr", "sr"):
            raise ValueError(f"gt_kind {gt_kind!r} (unified|fr|sr)")
        self.gt_kind = gt_kind
        self.global_unit_mm, self.global_ops = float(global_unit_mm), global_ops
        if input_norm == "global" and scale > 0:
            raise ValueError("ingresso globale: la scala dell'augmentation deve essere 0 (cambia la taglia, non il "
                             "bersaglio; la scala con bersaglio e' --scale-aug del trainer)")
        self.reuse = float(reuse)
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7_001, self.rank]))
        self.input_norm, self.area_weights = input_norm, area_weights
        self.rot_deg, self.scale = tuple(float(x) for x in rot_deg), float(scale)
        self.min_groups, self.wait_s = int(min_groups), float(wait_s)
        self.gt = gt
        import data_v3  # noqa: F401  (importato qui, non per la prima volta in un thread di materialize)
        self.readers: dict = {}            # seq -> ShardReader
        self.pool: dict = {}               # (seq, g) -> stato del gruppo
        self.pending: dict = {}            # seq -> viste prenotate e non ancora servite
        self.handles: dict = {}            # handle -> (seq, Future)
        self.plan_handles: list = []       # handle per piano, per buttare quelli mai serviti
        self._next_handle = 0
        self.pool_ex = ThreadPoolExecutor(max_workers=max(1, int(prefetch))) if prefetch > 0 else None
        self.c = {"plans": 0, "uses": 0, "unique_views_used": 0, "views_seen": 0, "groups_seen": 0,
                  "groups_evicted_unused": 0, "views_evicted_unused": 0, "over_reuse_groups": 0, "wait_s": 0.0,
                  "serve_s": 0.0, "plan_s": 0.0, "age_sum_s": 0.0, "age_max_s": 0.0, "by_domain": {}, "by_label": {},
                  "seq_used_min": None, "seq_used_max": None, "seq_mod_seen": set(), "uses_by_ring": {}}

    # --- anello -----------------------------------------------------------------------------------------
    def refresh(self) -> None:
        present = {seq: p for seq, p in self.ring.seqs().items() if seq % self.shard_world == self.shard_rank}
        if self.extra:
            if time.time() - self._extra_t >= self.extra_refresh_s:
                src = [(m, er, ew) for m, (_, er, ew) in zip(self.mirror, self.extra)] if self.mirror else self.extra
                self._extra_present = {-(i * EXTRA_OFFSET + seq): p for i, (rg, er, ew) in enumerate(src, 1)
                                       for seq, p in rg.seqs().items() if seq % ew == er}
                self._extra_t = time.time()
            present.update(self._extra_present)
        for seq, p in present.items():
            if seq in self.readers:
                continue
            try:
                rd = ShardReader(p)
            except (FileNotFoundError, ValueError):     # cancellato fra listing e apertura
                continue
            self.readers[seq] = rd
            for g, hg in enumerate(rd.groups):
                # tutte le fonti del gruppo (un ibrido o un trasferimento ne hanno piu' d'una), non solo il template
                if self.domains is not None and not set((hg.get("prov") or {}).get("sources") or [hg["domain"]]) \
                        <= self.domains:
                    continue
                self.pool[(seq, g)] = {"key": hg["key"], "domain": hg["domain"], "uses": 0,
                                       "view_uses": np.zeros(len(hg["views"]), dtype=np.int64),
                                       "t_created": rd.head["t_created"]}
                self.c["views_seen"] += len(hg["views"])
                self.c["groups_seen"] += 1
        for seq in [s for s in self.readers if s not in present]:      # uscito dall'anello
            for g in range(len(self.readers[seq].groups)):
                st = self.pool.pop((seq, g), None)
                if st is not None:
                    self.c["views_evicted_unused"] += int((st["view_uses"] == 0).sum())
                    self.c["groups_evicted_unused"] += int(st["uses"] == 0)
            if not self.pending.get(seq):
                del self.readers[seq]
                self.pending.pop(seq, None)

    def _mirror_loop(self) -> None:
        """Copia locale degli shard del rank negli anelli in piu' (i piu' nuovi per primi, al massimo 16 per giro,
        poi si rilegge l'anello); toglie le copie degli shard usciti dall'anello condiviso (chi li ha in mappa
        continua a leggerli)."""
        while True:
            busy = False
            for (rg, er, ew), dst in zip(self.extra, self.mirror):
                try:
                    want = {seq: p for seq, p in rg.seqs().items() if seq % ew == er}
                except OSError:
                    continue
                have = dst.seqs()
                todo = sorted(set(want) - set(have), reverse=True)
                busy |= len(todo) > 16
                for seq in todo[:16]:
                    t0 = time.time()
                    tmp = dst.tmp / f"{seq:010d}.{os.getpid()}"
                    out = dst.shards / f"{seq:010d}.shard"
                    try:
                        shutil.copyfile(want[seq], tmp)
                        os.replace(tmp, out)
                    except OSError:                    # uscito dall'anello condiviso prima della copia
                        self.mstat["missed"] += 1
                        tmp.unlink(missing_ok=True)
                        continue
                    self.mstat["copied"] += 1
                    self.mstat["bytes"] += out.stat().st_size
                    self.mstat["seconds"] += time.time() - t0
                for seq in set(have) - set(want):
                    try:
                        have[seq].unlink(missing_ok=True)
                    except OSError:
                        continue
                    self.mstat["removed"] += 1
            if not busy:
                time.sleep(self.extra_refresh_s)

    def _wait_for(self, n: int) -> None:
        t0 = time.time()
        while len({st["key"] for st in self.pool.values()}) < n:
            if time.time() - t0 > self.wait_s:
                raise TimeoutError(f"anello {self.ring.root}: meno di {n} identita' per il rank {self.rank} dopo "
                                   f"{self.wait_s:.0f}s (produttori fermi?)")
            time.sleep(0.5)
            self.refresh()
        self.c["wait_s"] += time.time() - t0

    # --- batch ------------------------------------------------------------------------------------------
    def pick(self, B: int, max_views: int, rng: np.random.Generator) -> list:
        """B gruppi con chiavi distinte: eleggibili (usi < reuse) in ordine casuale, poi i meno usati.
        Ritorna [(seq, g, [viste])]; aggiorna i contatori d'uso."""
        self.refresh()
        self._wait_for(max(B, self.min_groups))
        keys = list(self.pool)
        out, seen = [], set()

        def take(k) -> bool:
            st = self.pool[k]
            if st["key"] in seen:
                return False
            seen.add(st["key"])
            nv = len(st["view_uses"])
            vs = list(range(nv)) if nv <= max_views else sorted(rng.choice(nv, size=max_views, replace=False).tolist())
            out.append((k[0], k[1], vs))
            return len(out) == B

        # eleggibili in ordine casuale, fermandosi al B-esimo (il pool per rank e' di migliaia di gruppi)
        done = False
        for i in rng.permutation(len(keys)):
            k = keys[int(i)]
            if self.pool[k]["uses"] < self.reuse and take(k):
                done = True
                break
        if not done:          # non bastano: i meno usati oltre il tetto
            for k in sorted((k for k in keys if self.pool[k]["uses"] >= self.reuse), key=lambda k: self.pool[k]["uses"]):
                if self.pool[k]["key"] not in seen:
                    self.c["over_reuse_groups"] += 1
                if take(k):
                    break
        now = time.time()
        for seq, g, vs in out:
            st = self.pool[(seq, g)]
            st["uses"] += 1
            for v in vs:
                if st["view_uses"][v] == 0:
                    self.c["unique_views_used"] += 1
                    if self.log_views:
                        self._log(seq, g, v, now)
                st["view_uses"][v] += 1
            self.c["uses"] += len(vs)
            age = now - st["t_created"]
            self.c["age_sum_s"] += age * len(vs)
            self.c["age_max_s"] = max(self.c["age_max_s"], age)
            self.c["by_domain"][st["domain"]] = self.c["by_domain"].get(st["domain"], 0) + len(vs)
            ri = str(-seq // EXTRA_OFFSET) if seq < 0 else "0"
            self.c["uses_by_ring"][ri] = self.c["uses_by_ring"].get(ri, 0) + len(vs)
            if seq < 0:
                continue
            self.c["seq_used_min"] = seq if self.c["seq_used_min"] is None else min(self.c["seq_used_min"], seq)
            self.c["seq_used_max"] = seq if self.c["seq_used_max"] is None else max(self.c["seq_used_max"], seq)
            self.c["seq_mod_seen"].add(seq % self.shard_world)
        return out

    # --- registro delle viste usate ------------------------------------------------------------------------
    # colonne del registro: categoriche (codici + ``<nome>_vocab``) e numeriche coi loro tipi; ~64 byte per vista
    LOG_CAT = ("domain", "origin", "license", "label", "expr", "person")
    LOG_NUM = {"license_rank": np.int8, "seed0": np.int64, "seed1": np.int32, "seed2": np.int32, "vi": np.int8,
               "frame": np.int32, "noise_seed": np.int64, "n": np.int32, "area_mm2": np.float32, "S": np.float32,
               "ring": np.int8, "seq": np.int64, "t_first_use": np.float64}

    def _log(self, seq: int, g: int, v: int, now: float) -> None:
        rd = self.readers[seq]
        hg = rd.groups[g]
        m = hg["views"][v]
        pr = hg.get("prov") or {}
        sd = (list(pr.get("seed") or []) + [-1, -1, -1])[:3]
        self._log_rows.append({
            "key": "" if pr.get("seed") else hg["key"], "domain": hg["domain"], "origin": hg.get("origin", ""),
            "license": pr.get("license", ""), "person": pr.get("person") or "", "label": m["label"], "expr": m["expr"],
            "license_rank": pr.get("license_rank", -1), "seed0": sd[0], "seed1": sd[1], "seed2": sd[2],
            "vi": m.get("vi", v), "frame": m.get("frame", -1), "noise_seed": m.get("noise_seed", -1), "n": m["n"],
            "area_mm2": m.get("area_mm2", np.nan), "S": hg.get("S", np.nan),
            "ring": (-seq // EXTRA_OFFSET) if seq < 0 else 0, "seq": (-seq) % EXTRA_OFFSET if seq < 0 else seq,
            "t_first_use": now})

    def flush_log(self, path: Path) -> int:
        """Scrive le righe accumulate (npz compresso, colonne compatte) e le svuota; ritorna il numero di righe.
        L'identita' e' il seme (seed0, seed1, seed2) del gruppo, FaMoS anche la persona: i coefficienti si
        rigenerano dal seme (regen.py) e restano negli shard (``zid``), non nel registro. ``key`` solo per i gruppi
        senza provenienza."""
        rows = self._log_rows
        self._log_rows = []
        if not rows:
            return 0
        out = {}
        for k in self.LOG_CAT + ("key",):
            vocab, codes = np.unique(np.asarray([r[k] for r in rows], dtype=str), return_inverse=True)
            if k == "key" and len(vocab) == 1 and vocab[0] == "":
                continue
            out[k] = codes.astype(np.uint8 if len(vocab) < 256 else np.uint32)
            out[f"{k}_vocab"] = vocab
        for k, dt in self.LOG_NUM.items():
            out[k] = np.asarray([r[k] for r in rows], dtype=dt)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **out)
        tmp.replace(path)
        return len(rows)

    def plan(self, B: int, max_views: int, drawcfg, rng: np.random.Generator):
        """Un BatchPlan del trainer (sampler_v3): soggetti = chiavi delle identita', entries con handle allo
        stream; rumore del batch come _draw_batch (probabilita', sigma log-uniforme, modo per mesh)."""
        from robustness.noise import sample_log_uniform_sigma
        from sampler_v3 import BatchPlan
        t0 = time.perf_counter()
        picked = self.pick(B, max_views, rng)
        do_noise = bool(rng.uniform() < float(drawcfg.p_noise))
        sigma = sample_log_uniform_sigma(drawcfg.sigma_min, drawcfg.sigma_max, rng) if do_noise else 0.0
        probs = np.asarray(drawcfg.noise_mode_probs, dtype=np.float64)
        plan = BatchPlan(subjects=[], sigma=float(sigma))
        hs = []
        for seq, g, vs in picked:
            rd = self.readers[seq]
            hg = rd.groups[g]
            key = hg["key"]
            if self.gt is not None:
                vec = rd.s(g) if self.gt_kind == "unified" else rd.target(g, self.gt_kind)
                self.gt.register(key, vec, hg["domain"], float(np.log(hg["S"])) if "S" in hg else None)
            plan.subjects.append(key)
            for v in vs:
                meta = hg["views"][v]
                topo = meta["label"] + ("+x" if meta["expr"] == "expr" else "")
                self.c["by_label"][topo] = self.c["by_label"].get(topo, 0) + 1
                h = self._book(seq, g, v)
                hs.append(h)
                mode = ""
                if sigma > 0.0:
                    mode = drawcfg.noise_modes[0] if len(drawcfg.noise_modes) == 1 else \
                        drawcfg.noise_modes[int(rng.choice(len(drawcfg.noise_modes), p=probs))]
                plan.entries.append((key, h, topo, mode))
        self.plan_handles.append(hs)
        while len(self.plan_handles) > 3:          # piani mai serviti (ripresa a meta' epoca): si liberano
            for h in self.plan_handles.pop(0):
                self._drop(h)
        self.c["plans"] += 1
        self.c["plan_s"] += time.perf_counter() - t0
        return plan

    def _book(self, seq: int, g: int, v: int) -> int:
        h = self._next_handle
        self._next_handle += 1
        rot = [float(self.rng.uniform(-a, a)) if a > 0 else 0.0 for a in self.rot_deg]
        sc = float(self.rng.uniform(1.0 - self.scale, 1.0 + self.scale)) if self.scale > 0 else 1.0
        self.pending[seq] = self.pending.get(seq, 0) + 1
        args = (self.readers[seq], g, v, rot, sc)
        fut = self.pool_ex.submit(self.materialize, *args) if self.pool_ex is not None else args
        self.handles[h] = (seq, fut)
        return h

    def _drop(self, h: int) -> None:
        item = self.handles.pop(h, None)
        if item is None:
            return
        seq, fut = item
        if hasattr(fut, "cancel"):
            fut.cancel()
        self.pending[seq] -= 1

    # --- servizio ---------------------------------------------------------------------------------------
    def materialize(self, rd: ShardReader, g: int, v: int, rot, sc: float) -> dict:
        """Il campione del loader della vista (g, v), copiato fuori dalla mappa, con input_norm e augmentation."""
        import torch
        import data_v3 as dv
        a = rd.view(g, v)
        meta = rd.groups[g]["views"][v]
        n = int(meta["n"])
        t = {f: torch.from_numpy(np.array(a[f])) for f in a}
        ev = t["evecs"].float()
        s = dv._serve({"verts": t["verts"], "faces": t["faces"], "mass": t["mass"], "evals": t["evals"],
                       "evecs": ev.t() if meta["evecs_f"] else ev,
                       "gradX": dv._CompactSparse.from_parts(t["gxi"], t["gxv"], (n, n)),
                       "gradY": dv._CompactSparse.from_parts(t["gyi"], t["gyv"], (n, n))})
        V = s["verts"]
        if self.input_norm == "global":     # mm veri / L0 (global_v3), gia' nel frame canonico
            import global_v3
            if "area_mm2" not in meta:
                raise KeyError(f"{rd.path}: vista senza area_mm2 (producer.py --canonical-gt)")
            c, f = global_v3.frame_params(V, s["faces"], s["mass"], s["evecs"], float(meta["area_mm2"]),
                                          self.area_weights)
            V = global_v3.apply(V, c, f, np.eye(3), self.global_unit_mm)
            if self.global_ops == "mm":
                s["mass"], s["evecs"] = global_v3.ops_to_mm(s["mass"], s["evecs"], float(meta["area_mm2"]),
                                                            self.global_unit_mm)
        elif self.input_norm == "sqrt_area":
            if self.area_weights == "mass":
                V = dv.reframe_sqrt_area(V, s["mass"], s["faces"])
            else:
                import area_v3
                w = area_v3.area_weights(self.area_weights, V, s["faces"], s["mass"], s["evecs"])
                V = (V - (w.unsqueeze(1) * V).sum(0, keepdim=True) / w.sum()) / torch.sqrt(w.sum())
        if any(rot) or sc != 1.0:
            R = torch.as_tensor(sc * rotation(*rot).T, dtype=V.dtype)
            V = V @ R
        s["verts"] = V.contiguous()
        s["name"] = f"{rd.groups[g]['key']}#{rd.seq}.{v}"
        return s

    def __getitem__(self, h: int) -> dict:
        t0 = time.perf_counter()
        seq, fut = self.handles.pop(int(h))
        s = fut.result() if hasattr(fut, "result") else self.materialize(*fut)
        self.pending[seq] -= 1
        self.c["serve_s"] += time.perf_counter() - t0
        return s

    def __len__(self) -> int:
        return self._next_handle

    # --- statistiche ------------------------------------------------------------------------------------
    def stats(self) -> dict:
        c = dict(self.c)
        c["seq_mod_seen"] = sorted(c["seq_mod_seen"])     # sempre [shard_rank]: gli shard degli altri non si toccano
        c["reuse_factor"] = c["uses"] / max(c["unique_views_used"], 1)
        c["mean_age_s"] = c["age_sum_s"] / max(c["uses"], 1)
        c["pool_groups"] = len(self.pool)
        c["pool_shards"] = len(self.readers)
        c["rank"], c["world"], c["reuse_cap"] = self.rank, self.world, self.reuse
        c["shard_rank"], c["shard_world"] = self.shard_rank, self.shard_world
        if self.mirror:
            c["mirror"] = dict(self.mstat)
        return c


class StreamPlans:
    """Le ``steps`` BatchPlan di un'epoca, estratte una alla volta (la lista del trainer v3 e' deterministica,
    lo stream no). Il piano successivo nasce mentre il corrente e' in GPU, cosi' i suoi campioni si preparano in
    anticipo. A fine iterazione (anche interrotta) una riga di statistiche in ``log_path``."""

    def __init__(self, consumer: StreamConsumer, steps: int, B: int, max_views: int, drawcfg, rng, epoch: int,
                 log_path: Path | None = None) -> None:
        self.consumer, self.steps, self.B, self.max_views = consumer, int(steps), int(B), int(max_views)
        self.drawcfg, self.rng, self.epoch, self.log_path = drawcfg, rng, int(epoch), log_path

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        c = self.consumer
        before = c.stats()
        t0 = time.time()
        c.c["seq_used_min"] = c.c["seq_used_max"] = None
        make = lambda: c.plan(self.B, self.max_views, self.drawcfg, self.rng)  # noqa: E731
        n = 0
        try:
            nxt = make() if self.steps > 0 else None
            for i in range(self.steps):
                cur = nxt
                nxt = make() if i + 1 < self.steps else None
                n += 1
                yield cur
        finally:
            if self.log_path is not None:
                after = c.stats()
                row = {"epoch": self.epoch, "plans_yielded": n, "seconds": time.time() - t0,
                       **{k: after[k] for k in ("seq_used_min", "seq_used_max", "seq_mod_seen", "pool_groups", "pool_shards",
                                                "reuse_factor", "mean_age_s", "age_max_s", "rank", "reuse_cap")},
                       **{f"d_{k}": after[k] - before[k] for k in ("uses", "unique_views_used", "views_seen",
                                                                    "groups_seen", "views_evicted_unused",
                                                                    "groups_evicted_unused", "over_reuse_groups",
                                                                    "wait_s", "serve_s", "plan_s")},
                       "by_domain": after["by_domain"], "by_label": after["by_label"]}
                if after["uses_by_ring"]:
                    row["uses_by_ring"] = after["uses_by_ring"]
                if "mirror" in after:
                    row["mirror"] = after["mirror"]
                if c.log_views:
                    p = self.log_path.parent / "views_used" / f"rank{c.rank:03d}_e{self.epoch:05d}_{int(t0)}.npz"
                    row["views_logged"] = c.flush_log(p)
                with open(self.log_path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
