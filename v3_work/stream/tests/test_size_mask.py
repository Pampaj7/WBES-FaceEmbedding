#!/usr/bin/env python
"""Correttezza (f): ``--size-mask-domains`` e log S_i con le chiavi dello stream (train_stream.install).

    aau/run.sh v3_work/stream/tests/test_size_mask.py

Un batch finto con chiavi dello stream di due domini (bfm2019 e ict) registrate nella StreamGT con il loro log S:
  * senza maschera la MSE di s e' la media su tutte le mesh;
  * con ``--size-mask-domains bfm2019`` e' la media sulle sole mesh ict (peso 0 alle altre), e un dominio che lo
    stream non ha (``bfm``, BFM REMESH) non maschera nulla;
  * StreamLogCS da' il log S della vista per le chiavi dello stream e la tabella per gli altri soggetti.
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
STREAM = THIS.parent
for _p in (STREAM, STREAM.parent / "trainer"):
    sys.path.insert(0, str(_p))


def main() -> None:
    import torch
    import train_stream as TS
    import factorized_v3 as fz
    from consumer import StreamGT
    from losses_v3 import StepBatch

    gt = StreamGT(1.0, 1.0)
    TS.STREAM["gt"] = gt
    TS.install()
    rng = np.random.default_rng(0)
    keys = [("bfm2019/a", "bfm2019", 4.0), ("bfm2019/b", "bfm2019", 4.1), ("ict/c", "ict", 4.05), ("ict/d", "ict", 3.95)]
    for k, d, ls in keys:
        gt.register(k, rng.normal(size=12), d, ls)
    logcs = TS.StreamLogCS({"id0001": 3.9})
    assert logcs["ict/c"] == 4.05 and logcs["id0001"] == 3.9 and "bfm2019/a" in logcs
    subj = [k for k, _, _ in keys for _ in range(2)]
    s_pred = torch.tensor([4.2, 4.2, 4.3, 4.3, 4.0, 4.0, 3.9, 3.9], dtype=torch.float64)
    Z = torch.cat([s_pred[:, None], torch.randn(8, 4, dtype=torch.float64)], dim=1)
    batch = StepBatch(Z=Z, mesh_subjects=subj, mesh_topos=["original", "remesh"] * 4,
                      batch_subjects=[k for k, _, _ in keys], gt=gt, name_to_idx=gt.name_to_idx)
    base = dict(loss="v2", lambda_size=1.0, lambda_subject=1.0, lambda_mesh=1.0, lambda_rank=0.5, use_id_loss=True,
                lambda_id=0.25, rank_margin=0.05, rank_pairs=64, rank_tau=0.02, rank_hard_frac=0.7,
                train_pair_mode="cross_topology", train_level="mixed")
    tgt = np.array([logcs[k] for k in subj])
    err2 = (s_pred.numpy() - tgt) ** 2
    out = {}
    for mask, want in (("", err2.mean()), ("bfm2019", err2[4:].mean()), ("bfm", err2.mean())):
        _, terms = fz.factorized_loss(Namespace(**base, size_mask_domains=mask), batch, logcs, None)
        out[mask or "nessuna"] = (terms["size_mse"], float(want))
        print(f"[size-mask] maschera {mask or '-'}: size_mse {terms['size_mse']:.6f}, atteso {want:.6f}", flush=True)
        assert abs(terms["size_mse"] - want) < 1e-9, mask
    print("[size-mask] ESITO PASSA", flush=True)


if __name__ == "__main__":
    main()
