"""Piano di un'epoca: quali soggetti, in quali batch, quali mesh, quale rumore.

Il trainer v2 estraeva tutto questo DENTRO il ciclo dei batch (train_runner._train_epoch_mixed), con il
generatore numpy dell'epoca, intrecciato ai forward. I forward non toccano numpy, quindi estrarre il piano
in anticipo, con le stesse chiamate nello stesso ordine, da' gli stessi valori: e' la proprieta' su cui
poggia l'equivalenza (test_equivalence.py la verifica contro la funzione v2 vera).

Campionatori (``--sampler``):
  * ``v2`` (default): ``epoch_subset`` di train_steps (quota dei passi per dominio, ``domain_step_share``),
    poi ``domain_blocked_order`` di train_v2 (batch a dominio singolo, blocchi mescolati);
  * ``balanced``: p_d proporzionale a n_d^alpha (n_d = soggetti di training del dominio nello split
    intero); alpha=1 proporzionale ai soggetti, alpha=0 uniforme fra domini. Passi per dominio col resto
    maggiore, soggetti di ogni dominio da permutazioni successive del pool residente (un soggetto non
    compare due volte nello stesso batch; nella stessa epoca si', se il pool non basta).
Batch (``--batch-domains``): ``single`` (default, l'unico possibile con la GT di oggi, NaN fra domini) o
``mixed``: il dominio di ogni soggetto del batch estratto da p_d. Richiede una GT definita fra domini (la
matrice con NaN fallisce alla prima lettura, NanGuardedMatrix).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from common import domain_of

from intrinsic_utils import sample_mesh_indices  # noqa: E402
from robustness.noise import sample_log_uniform_sigma  # noqa: E402
from robustness.train_runner import _sample_subject_mesh_entries  # noqa: E402

SUBJECT_ID_RE = re.compile(r"^id\d+$", re.IGNORECASE)


@dataclass
class BatchPlan:
    subjects: List[str]
    sigma: float
    # (soggetto, indice nel dataset, etichetta di topologia, modo di rumore o "" se sigma == 0)
    entries: List[tuple] = field(default_factory=list)

    def valid(self) -> bool:
        """Le condizioni con cui v2 salta un batch (len(per_subject_latents) < 2 o < 3 mesh)."""
        return len({e[0] for e in self.entries}) >= 2 and len(self.entries) >= 3


# --- v2: train_steps.epoch_subset + train_v2.domain_blocked_order -----------------------------------

def domain_blocked_order(subjects: Sequence[str], batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """IDENTICA a train_v2.domain_blocked_order: ogni blocco consecutivo di batch_size e' a dominio singolo."""
    blocks: list[list[str]] = []
    for dom in sorted({domain_of(s) for s in subjects}):
        pool = [s for s in subjects if domain_of(s) == dom]
        if len(pool) < batch_size:
            raise ValueError(f"domain {dom} has {len(pool)} train subjects < batch_subjects={batch_size}; "
                             "lower --batch_subjects or drop the domain")
        pool = rng.permutation(np.array(pool, dtype=object)).tolist()
        n_used = (len(pool) // batch_size) * batch_size
        blocks.extend([pool[i: i + batch_size] for i in range(0, n_used, batch_size)])
    for block in blocks:
        assert len({domain_of(s) for s in block}) == 1, f"batch spans domains: {block}"
    order = rng.permutation(len(blocks))
    return np.array([s for i in order for s in blocks[int(i)]], dtype=object)


def epoch_subset(pool: list[str], steps: int, batch: int, seed: int, domain_blocked: bool,
                 share: dict | None = None) -> list[str]:
    """IDENTICA a train_steps.epoch_subset: ``steps * batch`` soggetti che diventano esattamente ``steps``
    batch, quota per dominio dai soggetti del blocco o da ``share`` (resto maggiore)."""
    rng = np.random.default_rng(seed)
    if not domain_blocked:
        groups = {"": list(pool)}
        alloc = {"": steps}
    else:
        groups: Dict[str, list] = {}
        for s in pool:
            groups.setdefault(domain_of(s), []).append(s)
        share = {d: float(f) for d, f in (share or {}).items() if d in groups}
        rest = [d for d in groups if d not in share]
        n_rest = sum(len(groups[d]) for d in rest)
        quota = {d: steps * f for d, f in share.items()}
        quota.update({d: steps * (1 - sum(share.values())) * len(groups[d]) / max(n_rest, 1) for d in rest})
        alloc = {d: int(q) for d, q in quota.items()}
        for d in sorted(quota, key=lambda d: quota[d] - alloc[d], reverse=True)[: steps - sum(alloc.values())]:
            alloc[d] += 1
    out = []
    for d in sorted(groups):
        n_need = alloc[d] * batch
        if n_need == 0:
            continue
        perm = rng.permutation(np.array(sorted(groups[d]), dtype=object)).tolist()
        out += (perm * math.ceil(n_need / len(perm)))[:n_need]
    if len(set(out)) != len(out):
        raise SystemExit(f"blocco troppo piccolo: {len(set(out))} soggetti per {len(out)} posti; "
                         "riduci --steps-per-epoch o i blocchi")
    return sorted(out)


def natural_steps(train_subjects: list[str], batch: int, domain_blocked: bool) -> int:
    """Passi di un'epoca v1 sugli stessi soggetti (train_steps.natural_steps)."""
    if domain_blocked:
        counts: dict[str, int] = {}
        for s in train_subjects:
            counts[domain_of(s)] = counts.get(domain_of(s), 0) + 1
        return sum(n // batch for n in counts.values())
    n = len(train_subjects)
    return n // batch + (1 if n % batch >= 2 else 0)


# --- estrazioni per batch, nell'ordine di _train_epoch_mixed ----------------------------------------

@dataclass
class DrawCfg:
    p_noise: float
    sigma_min: float
    sigma_max: float
    noise_modes: Sequence[str]
    noise_mode_probs: Sequence[float]
    max_meshes: int


def _draw_batch(rng: np.random.Generator, subjects: List[str], topo_map, subj_map, cfg: DrawCfg) -> BatchPlan:
    """Le chiamate a ``rng`` di un batch di _train_epoch_mixed (train_runner.py:1025-1064), nello stesso
    ordine: rumore del batch, sigma, poi per ogni soggetto le sue mesh e per ogni mesh il modo di rumore."""
    do_noise = bool(rng.uniform() < float(cfg.p_noise))
    sigma = sample_log_uniform_sigma(cfg.sigma_min, cfg.sigma_max, rng) if do_noise else 0.0
    probs = np.asarray(cfg.noise_mode_probs, dtype=np.float64)
    plan = BatchPlan(subjects=list(subjects), sigma=float(sigma))
    for sid in subjects:
        entries = _sample_subject_mesh_entries(sid=str(sid), subject_topology_map=topo_map,
                                               max_meshes=int(cfg.max_meshes), rng=rng)
        if not entries:
            fallback = sample_mesh_indices(subj_map[str(sid)], max_meshes=cfg.max_meshes,
                                           seed=int(rng.integers(0, 2_000_000_000)))
            entries = [(int(i), "unknown") for i in fallback]
        for idx, topo in entries:
            mode = ""
            if sigma > 0.0:
                if len(cfg.noise_modes) == 1:
                    mode = cfg.noise_modes[0]
                else:
                    mode = cfg.noise_modes[int(rng.choice(len(cfg.noise_modes), p=probs))]
            plan.entries.append((str(sid), int(idx), str(topo), mode))
    return plan


def plan_epoch_v2(epoch_subjects: Sequence[str], batch: int, seed: int, epoch: int, domain_blocked: bool,
                  topo_map, subj_map, cfg: DrawCfg) -> List[BatchPlan]:
    """Piano di _train_epoch_mixed: rng(seed+977+epoch), permutazione (bloccata per dominio), batch."""
    rng = np.random.default_rng(int(seed) + 977 + int(epoch))
    arr = np.array(list(epoch_subjects), dtype=object)
    vals = [str(v) for v in arr.ravel().tolist()]
    if domain_blocked and vals and all(SUBJECT_ID_RE.match(v) for v in vals):
        perm = domain_blocked_order(vals, batch, rng)     # train_v2._BlockedRng.permutation
    else:
        perm = rng.permutation(arr)
    plans = []
    for start in range(0, len(perm), batch):
        subjects = perm[start: start + batch].tolist()
        if len(subjects) < 2:
            continue
        plans.append(_draw_batch(rng, [str(s) for s in subjects], topo_map, subj_map, cfg))
    return plans


# --- bilanciato con temperatura ---------------------------------------------------------------------

def domain_probs(counts: Dict[str, int], alpha: float) -> Dict[str, float]:
    """p_d proporzionale a n_d^alpha."""
    w = {d: float(n) ** float(alpha) for d, n in counts.items() if n > 0}
    tot = sum(w.values())
    return {d: v / tot for d, v in sorted(w.items())}


def _largest_remainder(total: int, probs: Dict[str, float]) -> Dict[str, int]:
    quota = {d: total * p for d, p in probs.items()}
    alloc = {d: int(q) for d, q in quota.items()}
    for d in sorted(quota, key=lambda d: quota[d] - alloc[d], reverse=True)[: total - sum(alloc.values())]:
        alloc[d] += 1
    return alloc


class _Stream:
    """Soggetti di un dominio da permutazioni successive; un batch non contiene mai due volte lo stesso."""

    def __init__(self, pool: Sequence[str], rng: np.random.Generator) -> None:
        self.pool = sorted(pool)
        self.rng = rng
        self.buf: list[str] = []

    def take(self, k: int, exclude: set[str] = frozenset()) -> list[str]:
        out: list[str] = []
        while len(out) < k:
            if not self.buf:
                self.buf = self.rng.permutation(np.array(self.pool, dtype=object)).tolist()
            s = self.buf.pop()
            if s in exclude or s in out:
                if not set(self.pool) - set(exclude) - set(out):
                    raise ValueError(f"pool di {len(self.pool)} soggetti troppo piccolo per un batch di {k}")
                continue
            out.append(s)
        return out


def plan_epoch_balanced(pool: Sequence[str], steps: int, batch: int, seed: int, epoch: int, probs: Dict[str, float],
                        mixed: bool, topo_map, subj_map, cfg: DrawCfg) -> List[BatchPlan]:
    """``steps`` batch esatti dal pool residente con le probabilita' di dominio ``probs``."""
    rng = np.random.default_rng(int(seed) + 977 + int(epoch))
    by_dom: Dict[str, list] = {}
    for s in pool:
        by_dom.setdefault(domain_of(s), []).append(s)
    probs = {d: p for d, p in probs.items() if d in by_dom}
    tot = sum(probs.values())
    probs = {d: p / tot for d, p in probs.items()}
    streams = {d: _Stream(by_dom[d], rng) for d in sorted(probs)}
    doms = sorted(probs)
    batches: list[list[str]] = []
    if not mixed:
        alloc = _largest_remainder(int(steps), probs)
        for d in doms:
            if alloc[d] and len(by_dom[d]) < batch:
                raise ValueError(f"dominio {d}: {len(by_dom[d])} soggetti < batch {batch}")
            for _ in range(alloc[d]):
                batches.append(streams[d].take(batch))
        order = rng.permutation(len(batches))
        batches = [batches[int(i)] for i in order]
    else:
        p = np.asarray([probs[d] for d in doms])
        for _ in range(int(steps)):
            ks = rng.multinomial(batch, p)
            b: list[str] = []
            for d, k in zip(doms, ks):
                if k:
                    b += streams[d].take(int(k), exclude=set(b))
            batches.append(b)
    return [_draw_batch(rng, b, topo_map, subj_map, cfg) for b in batches]
