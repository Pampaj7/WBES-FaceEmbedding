#!/usr/bin/env python
"""Throughput su GPU del passo di training: forward sequenziale (v2) contro packed (v3), su batch REALI.

Passo completo = embedding delle mesh del batch (5 soggetti x <= 6 mesh) + loss v2 + backward + Adam, con
TF32 come nel training. Per dominio (BFM ha mesh da 8k a 60k vertici, ICT e GNM piu' piccole) e per pooling.
In piu', solo forward: il vecchio v2_work/fastio/batched.py (gruppi con pad_slack 0.05, importato in sola
lettura) per vedere quanti gruppi produce su V continuo.

    aau/run.sh v3_work/trainer/tests/bench_forward_v3.py --testdata ... --stage-root /tmp/... --out bench.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
sys.path.insert(0, str(TRAINER))

import train_v3 as T3  # noqa: E402
import sampler_v3 as sv  # noqa: E402
from losses_v3 import StepBatch, compute_loss  # noqa: E402
from model_v3 import StepEmbedder, build_model_v3  # noqa: E402
from intrinsic_utils import seed_everything  # noqa: E402
from robustness.noise import PerturbationParams, parse_noise_mode_weights, parse_noise_modes  # noqa: E402

from test_equivalence import RECIPE  # noqa: E402


def _sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def step_time(args, model, emb, plans, data, n_warm: int, reps: int) -> dict:
    """Passo completo e fasi (sincronizzate): dati = serve dal dataset + copia sulla GPU delle mesh del batch
    (misurata a parte, senza forward); embed = dati + forward; poi loss, backward (+ clip), Adam."""
    from model_v3 import to_device
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    emb.train()
    dev = next(model.parameters()).device
    ph = {"data": 0.0, "embed": 0.0, "loss": 0.0, "backward": 0.0, "optim": 0.0}
    times, meshes = [], 0
    torch.cuda.reset_peak_memory_stats()
    for i, plan in enumerate((plans * ((n_warm + reps) // len(plans) + 1))[: n_warm + reps]):
        t0 = _sync()
        for e in plan.entries:                    # solo i dati, come li prepara il percorso sequenziale
            to_device(data.dataset[e[1]], dev, emb.fast_data)
        t1 = _sync()
        opt.zero_grad(set_to_none=True)
        Z = emb(plan.entries, plan.sigma, True)
        t2 = _sync()
        b = StepBatch(Z, [e[0] for e in plan.entries], [e[2] for e in plan.entries], list(plan.subjects), data.gt,
                      data.name_to_idx)
        loss, _ = compute_loss("v2", b, args)
        t3 = _sync()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        t4 = _sync()
        opt.step()
        t5 = _sync()
        if i >= n_warm:
            times.append(t5 - t1)
            meshes += len(plan.entries)
            for k, d in zip(ph, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                ph[k] += d
    t = sum(times)
    n = len(times)
    return {"s_per_step": t / n, "meshes_per_s": meshes / t, "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
            "steps": n, "phase_s": {k: v / n for k, v in ph.items()}}


@torch.no_grad()
def forward_only(model, emb, plans, data, mode: str, reps: int) -> dict:
    """Solo forward in eval, sequenziale / packed / gruppi di batched.py (pad_slack 0.05)."""
    sys.path.insert(0, str(REPO / "v2_work/fastio"))
    from batched import embed_samples, size_groups
    model.eval()
    dev = next(model.parameters()).device
    n_groups, t, meshes = [], 0.0, 0
    for plan in (plans * (reps // len(plans) + 1))[:reps]:
        samples = [data.dataset[e[1]] for e in plan.entries]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if mode == "groups":
            embed_samples(model, samples, dev)
            n_groups.append(len(size_groups(samples, 0.05)))
        else:
            emb.eval()
            emb(plan.entries, 0.0, False)
        torch.cuda.synchronize()
        t += time.perf_counter() - t0
        meshes += len(plan.entries)
    out = {"s_per_batch": t / reps, "meshes_per_s": meshes / t}
    if n_groups:
        out["groups_per_batch"] = sum(n_groups) / len(n_groups)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdata", type=Path, required=True)
    ap.add_argument("--stage-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--poolings", default="meanmax,area_attn")
    ap.add_argument("--bucket-ratios", default="1.5")
    a = ap.parse_args()
    spec = json.loads((a.testdata / "spec.json").read_text())
    spec["n_blocks"] = 1
    a.stage_root.mkdir(parents=True, exist_ok=True)
    (a.stage_root / "spec.json").write_text(json.dumps(spec))
    args = T3.build_parser().parse_args(RECIPE + [
        "--seed", "1234", "--dist_npz", str(a.testdata / "gt.npz"), "--runs_root", str(a.stage_root / "r"),
        "--data-spec", str(a.stage_root / "spec.json"), "--split-json", str(a.testdata / "split.json"),
        "--stage-root", str(a.stage_root / "stage"), "--total-steps", "1000", "--steps-per-epoch", "10",
        "--gt-keep-scale", "--eval_domain", "bfm", "--prepass-proc", "7"])
    device = T3.configure_device(args, 0, False)
    data = T3.Data(args, 0, 1)
    data.setup(epochs=100)
    data.switch_block(0)
    if data.pending:
        T3.dv.kill_proc(data.pending[1])
    modes = parse_noise_modes(args.noise_modes)
    probs = parse_noise_mode_weights(args.noise_mode_weights, modes)
    pert = PerturbationParams.from_namespace(args)
    cfg = sv.DrawCfg(args.p_noise, args.sigma_min, args.sigma_max, modes, probs, args.max_meshes_per_subject_train)
    by_dom = {}
    for s in data.train:
        by_dom.setdefault(sv.domain_of(s), []).append(s)
    plans = {d: [p for e in range(1, 4) for p in sv.plan_epoch_v2(sorted(ss), 5, 1234, e, True, data.topo_map,
                                                                    data.subj_map, cfg)]
             for d, ss in sorted(by_dom.items())}
    gpu = torch.cuda.get_device_name(0)
    res = {"gpu": gpu, "tf32": bool(torch.backends.cuda.matmul.allow_tf32), "train_step": [], "forward_only": []}
    for pooling in a.poolings.split(","):
        args.pooling = pooling
        seed_everything(1234)
        base = build_model_v3(args, device)
        for dom, pl in plans.items():
            nv = sorted({int(data.dataset[e[1]]["verts"].shape[0]) for p in pl[:3] for e in p.entries})
            for mode, ratio in [("sequential", 1.5), ("sequential+fast", 1.5), ("groups", 1.5), ("groups+fast", 1.5)] + \
                    [("packed", float(r)) for r in a.bucket_ratios.split(",")]:
                m = copy.deepcopy(base)
                emb = StepEmbedder(m, mode.split("+")[0], ratio, 0.05).to(device)
                emb.fast_data = mode.endswith("+fast")
                T3.dv.FAST["on"] = emb.fast_data
                emb.bind(data.dataset, pert)
                r = step_time(args, m, emb, pl, data, n_warm=3, reps=a.reps)
                r.update(pooling=pooling, domain=dom, forward=mode, bucket_ratio=ratio, vertex_counts=nv[:12])
                res["train_step"].append(r)
                print(f"[bench] passo {pooling:12s} {dom:4s} {mode:10s} r={ratio:<4} {r['s_per_step'] * 1e3:8.1f} ms/passo "
                      f"{r['meshes_per_s']:7.1f} mesh/s picco {r['peak_gib']:.1f} GiB | fasi ms: "
                      + " ".join(f"{k}={v * 1e3:.0f}" for k, v in r["phase_s"].items()), flush=True)
            if pooling == "meanmax":
                for mode in ("sequential", "packed", "groups"):
                    m = copy.deepcopy(base)
                    emb = StepEmbedder(m, "packed" if mode == "packed" else "sequential", 1.5).to(device)
                    emb.bind(data.dataset, pert)
                    r = forward_only(m, emb, pl, data, mode, a.reps)
                    r.update(domain=dom, forward=mode)
                    res["forward_only"].append(r)
                    print(f"[bench] forward {dom:4s} {mode:10s} {r['s_per_batch'] * 1e3:8.1f} ms/batch "
                          f"{r['meshes_per_s']:7.1f} mesh/s"
                          + (f" gruppi/batch {r['groups_per_batch']:.1f}" if "groups_per_batch" in r else ""),
                          flush=True)
    for pooling in a.poolings.split(","):
        for dom in plans:
            rows = [r for r in res["train_step"] if r["pooling"] == pooling and r["domain"] == dom]
            seq = [r for r in rows if r["forward"] == "sequential"][0]
            for r in rows:
                if r["forward"] != "sequential":
                    r["speedup_vs_sequential"] = seq["s_per_step"] / r["s_per_step"]
                    print(f"[bench] {pooling} {dom}: {r['forward']} r={r['bucket_ratio']} x{r['speedup_vs_sequential']:.2f}")
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
