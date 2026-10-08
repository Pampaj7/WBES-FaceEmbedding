#!/usr/bin/env python
"""Traccia del primo passo di train_v3 (pesi, RNG, campioni serviti con dtype e stride, Z, loss): si lancia in
modo staging e in modo store con gli stessi argomenti e si confrontano le due tracce riga per riga.

    aau/run.sh v3_work/trainer/tests/debug_store_step.py <argomenti di train_v3> > trace.txt
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))
import train_v3 as T3  # noqa: E402
import model_v3 as mv  # noqa: E402


def h(t) -> str:
    if t is None:
        return "None"
    if t.is_sparse:
        t = t.coalesce() if not t.is_coalesced() else t
        return f"sp[{h(t.indices())}|{h(t.values())}]"
    return f"{hashlib.md5(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:8]}" \
           f"{tuple(t.shape)}{str(t.dtype)[6:]}s{tuple(t.stride())}"


def rng() -> str:
    c = hashlib.md5(torch.get_rng_state().numpy().tobytes()).hexdigest()[:8]
    g = hashlib.md5(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()[:8] if torch.cuda.is_available() else "-"
    return f"cpu={c} cuda={g}"


_fwd = mv.StepEmbedder.forward
CALLS = {"n": 0}


def forward(self, entries, sigma, add_noise):
    CALLS["n"] += 1
    if CALLS["n"] == 1:
        print("TRACE pesi", hashlib.md5(b"".join(p.detach().cpu().numpy().tobytes()
                                                 for p in self.model.parameters())).hexdigest()[:8], flush=True)
        print("TRACE rng", rng(), "sigma", sigma, "add_noise", add_noise, flush=True)
        for _sid, idx, topo, mode in entries:
            s = self.dataset[int(idx)]
            print("TRACE in", self.dataset.files[int(idx)], topo, mode,
                  " ".join(f"{k}={h(s.get(k))}" for k in ("verts", "mass", "evals", "evecs", "gradX", "gradY")),
                  flush=True)
            d = mv.to_device(s, next(self.model.parameters()).device, self.fast_data)
            print("TRACE dev", " ".join(f"{k}={h(d.get(k))}" for k in ("verts", "mass", "evals", "evecs",
                                                                        "gradX", "gradY")), flush=True)
    Z = _fwd(self, entries, sigma, add_noise)
    if CALLS["n"] <= 3:
        print(f"TRACE Z{CALLS['n']}", h(Z), f"{float(Z.double().sum()):.9e}", "rng dopo", rng(), flush=True)
    return Z


mv.StepEmbedder.forward = forward

if __name__ == "__main__":
    T3.main()
