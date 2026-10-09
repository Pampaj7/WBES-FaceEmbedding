#!/usr/bin/env python
"""``--forward groups`` con le teste fattorizzate: equivalenza col sequenziale e velocita' (10 ottobre).

    aau/run.sh v3_work/trainer/tests/test_groups_factorized.py --out aau/runs/evidence/trainer_v3/groups_factorized.json

Viste vere dello stream (v3_work/stream: un gruppo per dominio del preset massive, ingresso globale in mm / L0, pesi
smooth), 64 mesh come un passo. Per factorized e factorized2 (pesi casuali, ultimo strato della testa della taglia
NON nullo, dropout 0, nessun rumore) lo stesso StepEmbedder in ``sequential`` e in ``groups``:
  * Z = [s, u]: scarto massimo assoluto e relativo (norma di Frobenius);
  * gradienti di una loss fissa (somma dei quadrati di Z): scarto relativo massimo fra i parametri;
  * tempo di forward + backward (mediana di ``--reps``), in fp32 stretto e con TF32 (come il training).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
STREAM = TRAINER.parent / "stream"
for _p in (TRAINER, STREAM):
    sys.path.insert(0, str(_p))


def stream_samples(n_groups: int, seed: int) -> list:
    """Campioni serviti dal consumatore (ingresso globale, smooth, nessuna rotazione) di gruppi veri."""
    import producer as PR
    import sources as S
    import views as VW
    from consumer import StreamConsumer, StreamGT
    from ring import Ring
    from sampler_v3 import DrawCfg
    from targets import CanonTargets
    VW.install_grad_vec()
    rng = np.random.default_rng(seed)
    uni = S.Unified()
    srcs = S.build_sources(S.parse_sources("massive"), uni)
    tg = CanonTargets(srcs)
    cfg = argparse.Namespace(views=4, p_expr=0.6, expr_frac=0.5, k_eig=128, evecs_dtype="fp32", labels=list(VW.LABELS),
                             label_p=np.asarray([VW.LABEL_WEIGHTS[k] for k in VW.LABELS]) /
                             sum(VW.LABEL_WEIGHTS.values()), mm_factor={d: tg.mm_factor(d, s) for d, s in srcs.items()})
    st = {k: 0.0 for k in ("t_ident", "t_mesh", "t_gen", "t_ops", "t_pack")}
    st.update(views=0, failures=0, verts=0, by_label={}, by_domain={})
    doms = list(srcs)
    groups = [PR.make_group(srcs[doms[i % len(doms)]], doms[i % len(doms)], rng, cfg, uni, f"g{i}", st, tg)
              for i in range(n_groups)]
    tmp = tempfile.mkdtemp()
    Ring(tmp, 0).write(groups, 0)
    c = StreamConsumer(tmp, reuse=10, prefetch=0, gt=StreamGT(uni.A, 1.0), gt_kind="sr", input_norm="global",
                       area_weights="smooth", rot_deg=(0, 0, 0), scale=0.0)
    plan = c.plan(n_groups, 4, DrawCfg(p_noise=0.0, sigma_min=5e-4, sigma_max=2e-2, noise_modes=["translation"],
                                       noise_mode_probs=[1.0], max_meshes=4), np.random.default_rng(0))
    return [c[e[1]] for e in plan.entries]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    import torch
    import area_v3
    import factorized_v3 as fz
    import data_v3 as dv
    from model_v3 import StepEmbedder
    area_v3.CFG.update(k=64)
    dv.FAST["on"] = True
    dev = torch.device("cuda")
    samples = stream_samples(a.groups, a.seed)
    ns = [int(s["verts"].shape[0]) for s in samples]
    entries = [(f"s{i // 4}", i, "original", "") for i in range(len(samples))]
    out = {"meshes": len(samples), "verts": {"min": min(ns), "median": int(np.median(ns)), "max": max(ns)}, "heads": {}}
    for head in ("factorized", "factorized2"):
        args = SimpleNamespace(head=head, latent_dim=256, width=128, n_blocks=4, dropout=0.0, pooling="area_meanmax",
                               attn_heads=1, area_weights="smooth", size_hidden=64, size_init=4.0, global_unit_mm=100.0)
        torch.manual_seed(a.seed)
        model = fz.build(args, dev)
        with torch.no_grad():
            model.size_head[-1].weight.normal_(0, 0.05)
        model.train()
        res = {}
        for prec in ("fp32", "tf32"):
            torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
            torch.backends.cudnn.allow_tf32 = prec == "tf32"
            Z, G, T = {}, {}, {}
            for mode in ("sequential", "groups"):
                emb = StepEmbedder(model, mode).to(dev)
                emb.fast_data = True
                emb.bind(samples, None)
                times = []
                for r in range(a.reps + 1):
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    z = emb(entries, 0.0, False)
                    (z.double() ** 2).sum().backward()
                    torch.cuda.synchronize()
                    if r:
                        times.append(time.perf_counter() - t0)
                Z[mode] = z.detach().double().cpu()
                G[mode] = {n: p.grad.detach().double().cpu() for n, p in model.named_parameters() if p.grad is not None}
                T[mode] = float(np.median(times))
            dz = (Z["groups"] - Z["sequential"]).abs()
            grel = max(float((G["groups"][n] - g).norm() / g.norm().clamp_min(1e-30)) for n, g in G["sequential"].items())
            res[prec] = {"z_max_abs": float(dz.max()), "z_rel": float(dz.norm() / Z["sequential"].norm()),
                         "s_max_abs": float(dz[:, 0].max()), "grad_rel_max": grel,
                         "s_fwd_bwd_sequential": T["sequential"], "s_fwd_bwd_groups": T["groups"],
                         "mesh_per_s_sequential": len(samples) / T["sequential"],
                         "mesh_per_s_groups": len(samples) / T["groups"]}
            print(f"[groups] {head} {prec}: {res[prec]}", flush=True)
        out["heads"][head] = res
    out["pass"] = all(r["fp32"]["z_rel"] < 1e-4 and r["fp32"]["grad_rel_max"] < 1e-3 for r in out["heads"].values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[groups] ESITO {'PASSA' if out['pass'] else 'FALLISCE'} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
