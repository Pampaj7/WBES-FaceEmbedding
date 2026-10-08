#!/usr/bin/env python
"""Test anti-collasso di una loss (richiesto dal critic l'8 ottobre): N passi su batch REALI ripetuti.

Tre batch fissi (uno per dominio: BFM, ICT, GNM; stesse mesh a ogni giro, rumore del training ri-estratto a
ogni passo), ricetta del run su scala (Adam 1e-4, wd 1e-6, clip 1, dropout e rumore latente accesi), un
modello fresco. Ogni ``--every`` passi, in eval e senza rumore, sui tre batch:
  * scala: mediana e media della distanza latente fra mesh di identita' diverse;
  * Spearman fra distanza latente e GT sulle coppie di identita' diverse (livello mesh);
  * NaN / inf nella loss o nei pesi.
Esito (criteri dichiarati prima di guardare i numeri):
  PASSA se nessun NaN/inf, la mediana finale e' fra 0.2x e 5x quella al passo di riferimento (``--ref-step``,
  dopo il transitorio) e sopra 1e-3, e lo Spearman finale non e' sceso di oltre 0.1 rispetto al riferimento.

    aau/run.sh v3_work/trainer/tests/anti_collapse.py --losses v2,log,log+inv --steps 6000 --testdata ... --out x.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
sys.path.insert(0, str(TRAINER))

import train_v3 as T3  # noqa: E402
import sampler_v3 as sv  # noqa: E402
from losses_v3 import StepBatch, compute_loss  # noqa: E402
from model_v3 import StepEmbedder, build_model_v3  # noqa: E402
from intrinsic_utils import seed_everything, spearman_corr  # noqa: E402
from robustness.noise import PerturbationParams, parse_noise_mode_weights, parse_noise_modes  # noqa: E402

from test_equivalence import RECIPE  # noqa: E402


@torch.no_grad()
def probe(emb, plans, data) -> dict:
    emb.eval()
    meds, means, sps = [], [], []
    for p in plans:
        Z = emb(p.entries, 0.0, False)
        subj = np.asarray([e[0] for e in p.entries], dtype=object)
        diff = (subj[:, None] != subj[None, :]) & np.triu(np.ones((len(subj),) * 2, dtype=bool), 1)
        d = torch.cdist(Z, Z).cpu().numpy()[diff]
        idx = np.asarray([data.name_to_idx[s] for s in subj])
        g = np.asarray(data.gt[np.ix_(idx, idx)])[diff]
        meds.append(float(np.median(d)))
        means.append(float(d.mean()))
        sps.append(float(spearman_corr(g.astype(np.float64), d.astype(np.float64))))
    emb.train()
    return {"median_d": float(np.mean(meds)), "mean_d": float(np.mean(means)), "spearman": float(np.mean(sps)),
            "finite_weights": bool(all(torch.isfinite(q).all() for q in emb.model.parameters()))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--losses", default="v2,log,log+inv,log+inv+nbr", help="varianti, in sequenza, stessi dati")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--ref-step", type=int, default=500)
    ap.add_argument("--forward", default="sequential")
    ap.add_argument("--testdata", type=Path, required=True)
    ap.add_argument("--stage-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra", default="", help="flag v3 in piu', es. '--lambda-scale 0.1'")
    a = ap.parse_args()
    spec = json.loads((a.testdata / "spec.json").read_text())
    spec["n_blocks"] = 1
    a.stage_root.mkdir(parents=True, exist_ok=True)
    (a.stage_root / "spec.json").write_text(json.dumps(spec))
    args = T3.build_parser().parse_args(RECIPE + [
        "--seed", "1234", "--dist_npz", str(a.testdata / "gt.npz"), "--runs_root", str(a.stage_root / "r"),
        "--data-spec", str(a.stage_root / "spec.json"), "--split-json", str(a.testdata / "split.json"),
        "--stage-root", str(a.stage_root / "stage"), "--total-steps", "1000", "--steps-per-epoch", "10",
        "--gt-keep-scale", "--eval_domain", "bfm", "--prepass-proc", "8"] + a.extra.split())
    device = T3.configure_device(args, 0, False)
    data = T3.Data(args, 0, 1)
    data.setup(epochs=100)
    data.switch_block(0)
    if data.pending:
        T3.dv.kill_proc(data.pending[1])
    modes = parse_noise_modes(args.noise_modes)
    probs = parse_noise_mode_weights(args.noise_mode_weights, modes)
    cfg = sv.DrawCfg(args.p_noise, args.sigma_min, args.sigma_max, modes, probs, args.max_meshes_per_subject_train)
    by_dom = {}
    for s in data.train:
        by_dom.setdefault(sv.domain_of(s), []).append(s)
    fixed = [sv.plan_epoch_v2(sorted(ss)[:5], 5, 1234, 1, True, data.topo_map, data.subj_map, cfg)[0]
             for _, ss in sorted(by_dom.items())]
    results = []
    for loss_name in a.losses.split(","):
        args.loss = loss_name
        results.append(run_one(a, args, data, fixed, device, modes, probs))
    summary = {r["loss"]: {"pass": r["pass"], "median_ratio": r["median_ratio_last_over_ref"], "nan_at": r["nan_at"],
                           "spearman_ref": r["ref"]["spearman"], "spearman_last": r["last"]["spearman"]}
               for r in results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "runs": results}, indent=1))
    print(json.dumps(summary, indent=1))


def run_one(a, args, data, fixed, device, modes, probs) -> dict:
    seed_everything(1234)
    model = build_model_v3(args, device)
    emb = StepEmbedder(model, a.forward, args.bucket_ratio).to(device)
    emb.bind(data.dataset, PerturbationParams.from_namespace(args))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(99)
    hist, nan_at = [], None
    t0 = time.time()
    hist.append({"step": 0, **probe(emb, fixed, data)})
    for step in range(1, a.steps + 1):
        p = fixed[(step - 1) % len(fixed)]
        do_noise = rng.uniform() < args.p_noise
        sigma = float(math.exp(rng.uniform(math.log(args.sigma_min), math.log(args.sigma_max)))) if do_noise else 0.0
        entries = [(e[0], e[1], e[2], modes[int(rng.choice(len(modes), p=probs))] if sigma > 0 else "")
                   for e in p.entries]
        opt.zero_grad(set_to_none=True)
        Z = emb(entries, sigma, True)
        loss, terms = compute_loss(args.loss, StepBatch(Z, [e[0] for e in entries], [e[2] for e in entries],
                                                        list(p.subjects), data.gt, data.name_to_idx), args)
        if not torch.isfinite(loss):
            nan_at = step
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step % a.every == 0:
            row = {"step": step, "loss": float(loss.item()), **{f"t_{k}": v for k, v in terms.items()},
                   **probe(emb, fixed, data)}
            hist.append(row)
            print(f"[ac {args.loss}] passo {step}: loss={row['loss']:.4f} mediana_d={row['median_d']:.4g} "
                  f"spearman={row['spearman']:.3f} ({(time.time() - t0) / step:.3f} s/passo)", flush=True)
            if not row["finite_weights"]:
                nan_at = step
                break
    ref = min(hist, key=lambda r: abs(r["step"] - a.ref_step))
    last = hist[-1]
    ratio = last["median_d"] / max(ref["median_d"], 1e-30)
    passed = (nan_at is None and 0.2 <= ratio <= 5.0 and last["median_d"] > 1e-3
              and last["spearman"] >= ref["spearman"] - 0.1)
    print(f"[ac {args.loss}] {'PASSA' if passed else 'FALLISCE'}: mediana {ref['median_d']:.4g} -> "
          f"{last['median_d']:.4g} (x{ratio:.3f}), spearman {ref['spearman']:.3f} -> {last['spearman']:.3f}, "
          f"NaN al passo {nan_at}", flush=True)
    return {"loss": args.loss, "extra": a.extra, "steps": a.steps, "forward": a.forward, "nan_at": nan_at,
            "ref": ref, "last": last, "median_ratio_last_over_ref": ratio, "pass": bool(passed),
            "criterion": "nessun NaN; mediana finale in [0.2, 5] x riferimento e > 1e-3; Spearman finale >= rif - 0.1",
            "history": hist}


if __name__ == "__main__":
    main()
