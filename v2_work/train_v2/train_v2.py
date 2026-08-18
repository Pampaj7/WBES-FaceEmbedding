#!/usr/bin/env python
"""Joint multi-domain training (BFM + FLAME [+ FaceScape]) on top of the v1 trainer.

The v1 package (`face_embedding/gt_encdec/remeshing/intrinsic/robustness/`) is
imported and **never modified**: v1 results stay reproducible. This module runs
`robustness.train_runner.run_training` unchanged and installs three narrow
patches on that module for the duration of the run (restored in a `finally`):

1. **Domain-homogeneous batching.** There is no defined identity distance
   between a BFM and a FLAME subject, so the joint GT is block-diagonal with
   *undefined* (not zero) inter-domain blocks. All three v1 epoch functions
   (`subject_mean`, `mesh_pair`, `mixed`) start with the same two lines:

       rng = np.random.default_rng(args.seed + 777 + epoch)
       train_perm = rng.permutation(np.array(train_subjects, dtype=object))

   and then cut `train_perm` into consecutive chunks of `batch_subjects`. So the
   *only* thing that decides batch composition is that permutation. We swap the
   generator for one whose `permutation()` — and nothing else — returns a
   domain-blocked order: per-domain shuffle, cut into batch-sized blocks, blocks
   shuffled across domains. Every chunk v1 then slices out is single-domain, so
   the undefined GT blocks are never read, and gradient steps still alternate
   between domains at *batch* granularity (not epoch granularity, which would
   let the model drift to whichever domain ran last).

   Cost of correctness: each domain's subject list is truncated to a multiple of
   `batch_subjects` per epoch (< batch_subjects subjects dropped per domain per
   epoch, re-shuffled each epoch), because a short block would misalign every
   chunk boundary after it. A domain with fewer than `batch_subjects` train
   subjects raises instead of being silently dropped.

2. **NaN-guarded GT.** The loaded GT matrix is returned as an ndarray subclass
   that raises on any read containing NaN. That covers every read path at once
   (all three epoch functions, both eval paths) instead of one guard per call
   site, so a mixed-domain batch fails loudly rather than poisoning the loss.
   Undefined inter-domain entries are stored as NaN precisely so they can never
   be mistaken for "distance 0" (see `make_joint_view.py`).

3. **Single-domain online eval.** Spearman over a pair set that includes
   undefined pairs is undefined; with guard (2) it would also raise. The online
   training eval is therefore restricted to one domain (`--eval_domain`, default
   = the domain with most eval subjects). Per-domain eval of a joint model is a
   post-hoc job (`posthoc_runner.py`), not something the training loop needs.

Support augmentation needs no patch at all: the offline bank
(`make_support_bank.py`) writes support variants as extra
`idNNNN_GTready_supp<k>.npz` meshes of the same subject, so v1's own per-subject
mesh sampler (`sample_mesh_indices` / `--max_meshes_per_subject_train`) is the
train-time sampler.

Usage (all v1 flags, plus --eval_domain):

    .conda_env/bin/python v2_work/train_v2/train_v2.py \
        --data_dir <joint view>/npz_withops --dist_npz <joint view>/gt_matrix.npz \
        --runs_root v2_work/runs/joint_bfm_flame --device cuda ...
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
INTRINSIC_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"
if str(INTRINSIC_DIR) not in sys.path:
    sys.path.insert(0, str(INTRINSIC_DIR))

from robustness import train_runner as v1  # noqa: E402  (v1 package, imported read-only)

# id-offset convention of the symlink views (v2_work/genflame/make_train_ready.py).
# FaceScape appears at 2000 (v2_work/transfer/facescape_train_ready) and 3000 (the offset the
# re-cropped rebuild uses); both are listed so neither can be silently read as FLAME.
DOMAIN_OFFSETS = ((3000, "facescape"), (2000, "facescape"), (1000, "flame"), (0, "bfm"))
SUBJECT_ID_RE = re.compile(r"^id\d+$", re.IGNORECASE)

STATS = {"blocked_permutations": 0, "batches": 0}


def domain_of(subject_id: str) -> str:
    """bfm/flame/facescape from the id offset (BFM 0, FLAME 1000, FaceScape 3000)."""
    num = int(str(subject_id).lower().lstrip("id"))
    for offset, name in DOMAIN_OFFSETS:
        if num >= offset:
            return name
    raise ValueError(f"no domain for subject {subject_id!r}")


def domain_counts(subjects: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(domain_of(s) for s in subjects).items()))


def domain_blocked_order(
    subjects: Sequence[str],
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subject order whose every consecutive `batch_size` chunk is single-domain."""
    blocks: list[list[str]] = []
    for dom in sorted({domain_of(s) for s in subjects}):
        pool = [s for s in subjects if domain_of(s) == dom]
        if len(pool) < batch_size:
            raise ValueError(
                f"domain {dom} has {len(pool)} train subjects < batch_subjects={batch_size}; "
                "lower --batch_subjects or drop the domain"
            )
        pool = rng.permutation(np.array(pool, dtype=object)).tolist()
        n_used = (len(pool) // batch_size) * batch_size  # short tail would misalign chunks
        blocks.extend([pool[i : i + batch_size] for i in range(0, n_used, batch_size)])

    for block in blocks:
        assert len({domain_of(s) for s in block}) == 1, f"batch spans domains: {block}"
    STATS["blocked_permutations"] += 1
    STATS["batches"] += len(blocks)
    order = rng.permutation(len(blocks))
    return np.array([s for i in order for s in blocks[int(i)]], dtype=object)


class _Fwd:
    """Attribute forwarder: everything goes to `target` except the given overrides."""

    def __init__(self, target, **overrides):
        self._target = target
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._target, name)


class _BlockedRng:
    """np.random.Generator look-alike; only `permutation` of a subject list differs."""

    def __init__(self, rng: np.random.Generator, batch_size: int):
        self._rng = rng
        self._batch_size = int(batch_size)

    def __getattr__(self, name):
        return getattr(self._rng, name)

    def permutation(self, x):
        vals = [str(v) for v in np.asarray(x, dtype=object).ravel().tolist()]
        if vals and all(SUBJECT_ID_RE.match(v) for v in vals):
            return domain_blocked_order(vals, self._batch_size, self._rng)
        return self._rng.permutation(x)


class NanGuardedMatrix(np.ndarray):
    """GT matrix whose undefined (NaN) inter-domain entries cannot be read silently."""

    def __getitem__(self, key):
        out = super().__getitem__(key)
        arr = np.asarray(out, dtype=np.float64)
        if arr.size and not np.isfinite(arr).all():
            raise AssertionError(
                "read an undefined (NaN) GT entry: the batch or eval set spans domains "
                f"(index={key!r})"
            )
        return np.asarray(out) if isinstance(out, np.ndarray) else out


def _nan_guarded_loader(orig):
    def wrapped(path, *a, **kw):
        D, name_to_idx = orig(path, *a, **kw)
        n_nan = int(np.count_nonzero(~np.isfinite(D)))
        print(f"[v2] GT {Path(path).name}: {D.shape} undefined(NaN) entries={n_nan}")
        return D.view(NanGuardedMatrix), name_to_idx

    return wrapped


def _single_domain_eval(orig, eval_domain: str):
    def wrapped(eval_subjects, max_subjects_eval_train, seed):
        subjects = [str(s) for s in eval_subjects]
        counts = domain_counts(subjects)
        dom = eval_domain or max(counts, key=lambda d: (counts[d], d))
        if dom not in counts:
            raise ValueError(f"--eval_domain {dom} absent from eval subjects {counts}")
        kept = [s for s in subjects if domain_of(s) == dom]
        print(f"[v2] online eval domain={dom} subjects={len(kept)}/{len(subjects)} all={counts}")
        return orig(eval_subjects=kept, max_subjects_eval_train=max_subjects_eval_train, seed=seed)

    return wrapped


@contextmanager
def _patched_v1(batch_size: int, eval_domain: str):
    saved = (v1.np, v1.load_gt_distance_matrix, v1._select_online_eval_subjects)
    v1.np = _Fwd(
        np,
        random=_Fwd(
            np.random,
            default_rng=lambda seed: _BlockedRng(np.random.default_rng(seed), batch_size),
        ),
    )
    v1.load_gt_distance_matrix = _nan_guarded_loader(saved[1])
    v1._select_online_eval_subjects = _single_domain_eval(saved[2], eval_domain)
    try:
        yield
    finally:
        v1.np, v1.load_gt_distance_matrix, v1._select_online_eval_subjects = saved


def run_training_v2(args: argparse.Namespace) -> None:
    with _patched_v1(int(args.batch_subjects), str(getattr(args, "eval_domain", ""))):
        v1.run_training(args)
    print(f"[v2] domain-blocked permutations={STATS['blocked_permutations']} batches={STATS['batches']}")


def parse_args_v2() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--eval_domain", type=str, default="", help="bfm|flame|facescape, '' = most frequent")
    extra, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = v1.parse_args()
    args.eval_domain = str(extra.eval_domain)
    return args


if __name__ == "__main__":
    run_training_v2(parse_args_v2())
