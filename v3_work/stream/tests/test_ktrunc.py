#!/usr/bin/env python
"""Correttezza (e): la troncatura di ktrunc.py (k 128 -> 64) e' la pipeline con k_eig 64.

    aau/run.sh v3_work/stream/tests/test_ktrunc.py --out aau/runs/evidence/stream/ktrunc_check.json

Per ogni dominio del preset massive un'identita' e due discretizzazioni (original, remesh): la stessa mesh va per
views.operators con k 128 (poi ktrunc.truncate a 64) e con k 64, entrambe servite come il loader
(serve_like_loader). Confronto: autovalori (relativo), autovettori a meno del segno (1 - |coseno| per colonna,
nella metrica della massa), valori dei gradienti, embedding del checkpoint e108 (forward congelato, CPU fp32).
Pavimento: k 64 calcolato due volte (eigsh parte da un vettore casuale).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

THIS = Path(__file__).resolve().parent
STREAM = THIS.parent
REPO = STREAM.parents[1]
for _p in (THIS, STREAM, REPO / "v3_work" / "trainer"):
    sys.path.insert(0, str(_p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    import sources as S
    import views as VW
    from ktrunc import truncate
    from model_v3 import build_model_v3
    from test_ops_equiv import CKPT, embed
    VW.install_grad_vec()
    torch.set_num_threads(4)
    pack = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model_v3(SimpleNamespace(**pack["args"]), torch.device("cpu"))
    model.load_state_dict(pack["state_dict"])
    model.eval()
    rng = np.random.default_rng(11)
    uni = S.Unified()
    srcs = S.build_sources(S.parse_sources("massive"), uni)
    rows = []
    for d, src in srcs.items():
        ident = src.identity(rng)
        V, F, _ = src.view_mesh(ident, rng, False)
        for label in ("original", "remesh"):
            Vd, Fd = VW.discretize(V, F, label, 0)
            def serve(k):    # come il consumatore (data_v3._serve): facce int64, L vuoto (lo spettrale non lo legge)
                s = VW.serve_like_loader(VW.operators(Vd, Fd, k))
                n = int(s["verts"].shape[0])
                s["faces"] = s["faces"].long()
                s["L"] = torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long), torch.zeros(0), (n, n)).coalesce()
                return s
            s128, s64, s64b = serve(128), serve(64), serve(64)
            t = truncate(s128, 64)
            M = s64["mass"].double()

            def cmp(x, y) -> dict:
                ev = float(((x["evals"].double() - y["evals"].double()).abs() / y["evals"].double().clamp_min(1e-12))[1:].max())
                cos = (x["evecs"].double() * M[:, None] * y["evecs"].double()).sum(0)
                g = max(float((x[k].coalesce().values().double() - y[k].coalesce().values().double()).abs().max())
                        for k in ("gradX", "gradY"))
                zx, zy = embed(model, x), embed(model, y)
                return {"evals_rel_max": ev, "evecs_1_minus_abs_cos_max": float((1 - cos.abs()).max()),
                        "grad_values_max_abs": g, "embed_max_abs": float((zx - zy).abs().max()),
                        "embed_rel": float((zx - zy).norm() / zy.norm())}
            row = {"domain": d, "label": label, "n": int(len(Vd)), "trunc_vs_k64": cmp(t, s64), "k64_vs_k64": cmp(s64b, s64)}
            rows.append(row)
            print(f"[ktrunc] {row}", flush=True)
    worst = max(r["trunc_vs_k64"]["embed_rel"] for r in rows)
    floor = max(r["k64_vs_k64"]["embed_rel"] for r in rows)
    out = {"rows": rows, "embed_rel_max": worst, "floor_embed_rel_max": floor,
           "pass": bool(worst < max(10 * floor, 1e-5))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[ktrunc] ESITO {'PASSA' if out['pass'] else 'FALLISCE'}: embedding relativo {worst:.2e} (pavimento {floor:.2e})")


if __name__ == "__main__":
    main()
