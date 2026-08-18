"""RAM-resident dataset for training: kill the per-iteration NFS read.

Measured on this cluster, per mesh, inside the training loop:
    np.load + decompress   48 ms   (BFM 23k verts; 264 ms cold, NFS)
    normalize + rebuild sparse ops   41 ms
    ---------------------------------------
    total per mesh         89 ms  ->  ~2.7 s per iteration of 30 meshes

The v1 trainer calls `dataset[idx]` inside the batch loop with no cache (only the
*eval* path preloads), so that cost is paid on every epoch for every mesh. This
module preloads the fully prepared samples once and serves them from RAM, which
removes both the decompression and the preprocessing from the training loop.

Memory, measured shapes, float32:
    BFM   23470 verts  ~22 MB/sample  ->  3000 samples ~ 66 GB
    FLAME  1930 verts   ~3 MB/sample  -> 30000 samples ~ 90 GB
Both fit on a large-memory node; `max_gb` refuses to start rather than swapping.

Two residency modes:
    "ram"    prepared tensors in host memory (pinned when a GPU is present, so the
             per-iteration H2D copy is a DMA rather than a pageable copy)
    "device" prepared tensors already on the GPU, so the training loop's
             `sample_to_device` becomes a no-op and there is no transfer at all

`install()` monkeypatches the name `GTReadyDataset` inside the frozen v1
`robustness` modules, so nothing under face_embedding/ has to be edited and v1
runs stay bit-reproducible when the cache is not installed.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/autoencoder"))
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"))

from dataset_gtready import GTReadyDatasetNPZ  # noqa: E402

TENSOR_KEYS = ("verts", "mass", "evals", "evecs", "faces", "gradX", "gradY", "L")


def _sample_bytes(sample: Dict[str, torch.Tensor]) -> int:
    total = 0
    for k in TENSOR_KEYS:
        t = sample.get(k)
        if not torch.is_tensor(t):
            continue
        if t.is_sparse:
            c = t.coalesce()
            total += c.indices().numel() * c.indices().element_size()
            total += c.values().numel() * c.values().element_size()
        else:
            total += t.numel() * t.element_size()
    return total


def _to_residency(sample: Dict[str, torch.Tensor], device: torch.device | None,
                  pin: bool) -> Dict[str, torch.Tensor]:
    out = dict(sample)
    for k in TENSOR_KEYS:
        t = out.get(k)
        if not torch.is_tensor(t):
            continue
        if device is not None:
            out[k] = t.to(device)
        elif pin and not t.is_sparse:
            # sparse tensors cannot be pinned; dense ones can, and they are the
            # bulk of the per-sample bytes (evecs dominates)
            out[k] = t.pin_memory()
    return out


SPARSE_KEYS = ("L", "gradX", "gradY")


def _with_roi(sample: Dict[str, torch.Tensor], path: str) -> Dict[str, torch.Tensor]:
    """Attach the well's region-of-interest mask, which the v1 dataset does not know about.

    The frozen loader builds a fixed key set, so `roi_mask` -- written alongside the operators
    by potential_operators.py -- would be dropped before the cache ever sees it. Reading it
    here keeps the change inside v2_work instead of editing the frozen dataset, and datasets
    without the key are returned untouched, so the masked and unmasked arms share this path.
    """
    if "roi_mask" in sample:
        return sample
    try:
        with np.load(path, allow_pickle=False) as z:
            if "roi_mask" not in z.files:
                return sample
            out = dict(sample)
            out["roi_mask"] = torch.as_tensor(z["roi_mask"], dtype=torch.float32)
            return out
    except (OSError, ValueError, KeyError):
        # Only I/O-shaped failures are tolerated. A broad `except Exception` here would also
        # swallow programming errors (a missing import, a renamed key) and silently disable
        # masking, which would look exactly like a masked run that simply did not help.
        return sample


def _rebuild_sparse(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return the sample with its sparse operators rebuilt from cached components.

    Necessary, not cosmetic. The evaluation path runs under `torch.inference_mode()`
    and diffusion-net does `L.unsqueeze(0)` there. A *cached* sparse tensor was created
    outside inference mode, so that view op tries to track a version counter for a
    tensor the inference block treats as an inference tensor, and torch raises
    "Cannot set version_counter for inference tensor". The v1 path never hits this
    because it constructs the tensors inside the inference block.

    Rebuilding here makes the constructed tensor inherit the *caller's* mode: an
    inference tensor under eval, an ordinary tensor under training (where inference
    tensors would be rejected by autograd). Cost is a COO wrapper around tensors that
    are already resident — microseconds, and no decompression or NFS read.
    """
    out = dict(sample)
    for k in SPARSE_KEYS:
        t = sample.get(k)
        if torch.is_tensor(t) and t.is_sparse:
            out[k] = torch.sparse_coo_tensor(t.indices(), t.values(), t.shape).coalesce()
    return out


class CachedDataset:
    """Drop-in replacement for GTReadyDatasetNPZ that serves prepared samples from RAM.

    Exposes the same surface the trainer touches: `files`, `__len__`, `__getitem__`.
    Indices are the same as the wrapped dataset's, so subject maps built from
    `files` stay valid.
    """

    def __init__(
        self,
        data_dir: str | Path,
        indices: Iterable[int] | None = None,
        workers: int = 16,
        residency: str = "ram",
        device: str | torch.device | None = None,
        max_gb: float = 900.0,
        verbose: bool = True,
    ) -> None:
        self._base = GTReadyDatasetNPZ(str(data_dir))
        self.files: Sequence[str] = self._base.files
        self.data_dir = str(data_dir)
        if residency not in ("ram", "device"):
            raise ValueError("residency must be 'ram' or 'device'")
        self._device = torch.device(device) if (residency == "device" and device) else None
        pin = residency == "ram" and torch.cuda.is_available()

        want = sorted(set(range(len(self.files)) if indices is None else (int(i) for i in indices)))
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

        # probe one sample to project memory before committing to the full load
        probe = self._base[want[0]]
        projected_gb = _sample_bytes(probe) * len(want) / 1024 ** 3
        if projected_gb > max_gb:
            raise MemoryError(
                f"cache would need ~{projected_gb:.0f} GB for {len(want)} samples "
                f"(limit {max_gb:.0f} GB). Reduce the subject count or raise --cache-max-gb."
            )
        if verbose:
            print(f"[cache] {len(want)} samples from {data_dir}", flush=True)
            print(f"[cache] ~{_sample_bytes(probe)/1024**2:.1f} MB/sample "
                  f"-> ~{projected_gb:.1f} GB, residency={residency}"
                  f"{'' if self._device is None else f' on {self._device}'}", flush=True)

        t0 = time.time()
        self._cache[want[0]] = _to_residency(probe, self._device, pin)

        def load(i: int) -> tuple[int, Dict[str, torch.Tensor]]:
            return i, _to_residency(_with_roi(self._base[i], self.files[i]), self._device, pin)

        rest = want[1:]
        if rest:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                for n, (i, s) in enumerate(ex.map(load, rest), start=2):
                    self._cache[i] = s
                    if verbose and n % 500 == 0:
                        rate = n / (time.time() - t0)
                        print(f"[cache] {n}/{len(want)} ({rate:.0f}/s, "
                              f"eta {(len(want)-n)/max(rate,1e-9):.0f}s)", flush=True)
        self.load_seconds = time.time() - t0
        if verbose:
            print(f"[cache] ready in {self.load_seconds:.0f}s "
                  f"({len(self._cache)/max(self.load_seconds,1e-9):.0f} samples/s)", flush=True)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self._cache.get(int(idx))
        if s is None:  # not requested at build time: fall back rather than crash
            return self._base[int(idx)]
        return _rebuild_sparse(s)


def install(dataset: CachedDataset) -> None:
    """Make the frozen v1 robustness modules use `dataset` instead of loading from disk."""
    import robustness.data_utils as du
    import robustness.train_runner as tr

    factory = lambda *_a, **_k: dataset  # noqa: E731  (the trainer calls GTReadyDataset(args.data_dir))
    du.GTReadyDataset = factory
    tr.GTReadyDataset = factory


def demo() -> None:
    """Self-check: cached reads must be far cheaper than uncached, and identical."""
    d = REPO_ROOT / "v2_work/genflame/flame_train_ready/npz_withops"
    base = GTReadyDatasetNPZ(str(d))
    n = 24
    base[0]  # warm page cache so the comparison is not flattered by cold NFS

    t0 = time.time()
    for i in range(n):
        base[i]
    uncached = (time.time() - t0) / n

    ds = CachedDataset(d, indices=range(n), workers=8, verbose=False)
    t0 = time.time()
    for i in range(n):
        ds[i]
    cached = (time.time() - t0) / n

    a, b = base[3], ds[3]
    assert torch.allclose(a["verts"], b["verts"]), "cached verts differ"
    assert torch.allclose(a["evals"], b["evals"]), "cached evals differ"
    assert a["name"] == b["name"], "cached name differs"
    assert cached < uncached / 20, f"cache not fast enough: {cached*1e3:.2f} vs {uncached*1e3:.1f} ms"
    print(f"OK uncached {uncached*1e3:.1f} ms/sample | cached {cached*1e3:.3f} ms/sample "
          f"| speedup {uncached/cached:.0f}x")


if __name__ == "__main__":
    demo()
