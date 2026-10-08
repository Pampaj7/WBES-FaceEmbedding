#!/usr/bin/env python
"""Test unitari del trainer v3 che non hanno bisogno di GPU.

  1. cache compatta: stessi tensori della cache v2 (int32 e niente L sono senza perdita), su mesh vere;
  2. campionatore bilanciato: quota di passi per dominio con alpha 0, 0.5, 1 contro p_d ~ n_d^alpha, e un
     batch non contiene mai due volte lo stesso soggetto;
  3. batch misti: con la GT a matrice (NaN fra domini) la guardia scatta; con una GT a vettori la loss si
     calcola e i batch contengono piu' domini;
  4. shard DDP: disgiunti, coprono tutto, stratificati per dominio;
  5. sqrt(area): non dipende dal frame d'ingresso, area totale 1;
  6. loss: tutte le varianti finite e con gradiente su un batch sintetico.

    aau/run.sh v3_work/trainer/tests/test_units_cpu.py --out aau/runs/evidence/trainer_v3/units_cpu.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
sys.path.insert(0, str(TRAINER))

import data_v3 as dv  # noqa: E402
import sampler_v3 as sv  # noqa: E402
from common import domain_of  # noqa: E402
from losses_v3 import LOSSES, StepBatch, compute_loss  # noqa: E402

VIEW_FILES = ["datasets/REMESH/npz_data_topo_500_withops_areanorm/id0001_GTready_up60k.npz",
              "datasets/REMESH/npz_data_topo_500_withops_areanorm/id0001_GTready_noisy.npz",
              "datasets/ICT/train_ready/npz_withops/id10000_GTready_down8k.npz"]


def t_compact() -> dict:
    tmp = Path(tempfile.mkdtemp())
    for f in VIEW_FILES:
        (tmp / Path(f).name).symlink_to((REPO / f).resolve())
    a = dv.CachedDataset(tmp, workers=2, compact=False, verbose=False)
    b = dv.CachedDataset(tmp, workers=2, compact=True, verbose=False)
    worst, bytes_a, bytes_b = 0.0, 0, 0
    for i in range(len(a)):
        sa, sb = a[i], b[i]
        for k in ("verts", "mass", "evals", "evecs"):
            worst = max(worst, float((sa[k] - sb[k]).abs().max()))
        assert torch.equal(sa["faces"], sb["faces"]) and sb["faces"].dtype == torch.long
        for k in ("gradX", "gradY"):
            assert torch.equal(sa[k].indices(), sb[k].indices()) and torch.equal(sa[k].values(), sb[k].values())
            assert sb[k].indices().dtype == torch.long
        assert sb["L"].shape == sa["L"].shape and sb["L"]._nnz() == 0
        bytes_a += dv._sample_bytes(a._cache[i])
        bytes_b += dv._sample_bytes(b._cache[i])
    return {"pass": worst == 0.0, "max_abs_dense": worst, "ram_ratio_compact_over_v2": bytes_b / bytes_a}


def t_fast_data() -> dict:
    """--fast-data (sparsi senza riordino, L e facce non copiati) deve dare loss e gradienti IDENTICI."""
    import copy
    from types import SimpleNamespace as NS
    from model_v3 import StepEmbedder, build_model_v3
    tmp = Path(tempfile.mkdtemp())
    for f in VIEW_FILES:
        (tmp / Path(f).name).symlink_to((REPO / f).resolve())
    ds = dv.CachedDataset(tmp, workers=2, compact=True, verbose=False)
    files = ds.files
    entries = [(f.split("_")[0], i, "t", "") for i, f in enumerate(files)]
    subjects = sorted({e[0] for e in entries})
    n2i = {s_: i for i, s_ in enumerate(subjects)}
    D = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float32)
    args = NS(model="xyz_dn", latent_dim=256, width=128, n_blocks=4, dropout=0.1, pool_mode="meanmax",
              pooling="meanmax", **vars(_loss_args()))
    torch.manual_seed(0)
    base = build_model_v3(args, torch.device("cpu"))
    out = []
    for fast in (False, True):
        m = copy.deepcopy(base)
        emb = StepEmbedder(m, "sequential")
        emb.fast_data = fast
        dv.FAST["on"] = fast
        emb.bind(ds, None)
        emb.train()
        torch.manual_seed(1)
        Z = emb(entries, 0.0, True)
        loss, _ = compute_loss("v2", StepBatch(Z, [e[0] for e in entries], ["original", "up60k", "down8k"], subjects,
                                               D.view(dv.NanGuardedMatrix), n2i), args)
        loss.backward()
        out.append((float(loss), [p.grad.clone() for p in m.parameters()]))
    dv.FAST["on"] = False
    dg = max(float((a - b).abs().max()) for a, b in zip(out[0][1], out[1][1]))
    return {"pass": out[0][0] == out[1][0] and dg == 0.0, "loss": [out[0][0], out[1][0]], "max_abs_grad_diff": dg}


def _fake_maps(subjects):
    files, subj_map, topo = [], {}, {}
    for s in subjects:
        for lab in ("original", "remesh", "down8k", "crop", "noisy", "up60k"):
            subj_map.setdefault(s, []).append(len(files))
            topo.setdefault(s, {}).setdefault(lab, []).append(len(files))
            files.append(f"{s}_GTready_{lab}.npz")
    return files, subj_map, topo


def t_balanced() -> dict:
    subjects = [f"id{i:04d}" for i in range(392)] + [f"id{20000 + i}" for i in range(4390)] + \
               [f"id{100000 + i}" for i in range(810)]
    _, subj_map, topo = _fake_maps(subjects)
    counts = Counter(domain_of(s) for s in subjects)
    cfg = sv.DrawCfg(0.6, 5e-4, 2e-2, ["translation", "rotation", "jitter"], [4 / 7, 2 / 7, 1 / 7], 6)
    out = {}
    ok = True
    for alpha in (0.0, 0.5, 1.0):
        probs = sv.domain_probs(counts, alpha)
        plans = sv.plan_epoch_balanced(subjects, 1000, 5, 1234, 1, probs, False, topo, subj_map, cfg)
        got = Counter(domain_of(p.subjects[0]) for p in plans)
        dup = any(len(set(p.subjects)) != len(p.subjects) for p in plans)
        single = all(len({domain_of(s) for s in p.subjects}) == 1 for p in plans)
        err = max(abs(got[d] / 1000 - probs[d]) for d in probs)
        ok &= (not dup) and single and err <= 0.001 and len(plans) == 1000
        out[f"alpha={alpha}"] = {"p_d": {d: round(v, 4) for d, v in probs.items()},
                                 "quota_passi": {d: got[d] / 1000 for d in sorted(got)}, "duplicati": dup,
                                 "batch_monodominio": single}
    out["pass"] = bool(ok)
    return out


def t_mixed() -> dict:
    subjects = [f"id{i:04d}" for i in range(20)] + [f"id{20000 + i}" for i in range(20)] + \
               [f"id{100000 + i}" for i in range(20)]
    _, subj_map, topo = _fake_maps(subjects)
    cfg = sv.DrawCfg(0.0, 5e-4, 2e-2, ["jitter"], [1.0], 6)
    probs = sv.domain_probs(Counter(domain_of(s) for s in subjects), 0.0)
    plans = sv.plan_epoch_balanced(subjects, 50, 5, 1, 1, probs, True, topo, subj_map, cfg)
    multi = float(np.mean([len({domain_of(s) for s in p.subjects}) > 1 for p in plans]))
    dup = any(len(set(p.subjects)) != len(p.subjects) for p in plans)
    n2i = {s: i for i, s in enumerate(subjects)}
    D = np.random.default_rng(0).uniform(0.1, 1.0, (60, 60)).astype(np.float32)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    for a in range(60):
        for b in range(60):
            if domain_of(subjects[a]) != domain_of(subjects[b]):
                D[a, b] = np.nan
    guard = D.view(dv.NanGuardedMatrix)
    p = next(p for p in plans if len({domain_of(s) for s in p.subjects}) > 1)
    Z = torch.randn(len(p.entries), 8, requires_grad=True)
    args = _loss_args()
    raised = False
    try:
        compute_loss("v2", StepBatch(Z, [e[0] for e in p.entries], [e[2] for e in p.entries], p.subjects, guard, n2i), args)
    except AssertionError:
        raised = True
    vec = dv.VectorGT(np.random.default_rng(1).normal(size=(60, 16)))
    loss, _ = compute_loss("log+inv", StepBatch(Z, [e[0] for e in p.entries], [e[2] for e in p.entries], p.subjects,
                                                 vec, n2i), args)
    loss.backward()
    ok = raised and torch.isfinite(loss).item() and multi > 0.5 and not dup and Z.grad is not None
    return {"pass": bool(ok), "frazione_batch_multidominio": multi, "guardia_nan_scatta": raised,
            "loss_vettoriale": float(loss.item())}


def t_shard() -> dict:
    subjects = [f"id{i:04d}" for i in range(392)] + [f"id{20000 + i}" for i in range(1000)] + \
               [f"id{100000 + i}" for i in range(201)]
    shards = [dv.shard_subjects(subjects, r, 4, 1234) for r in range(4)]
    allx = [s for sh in shards for s in sh]
    disjoint = len(allx) == len(set(allx)) == len(subjects)
    per = [Counter(domain_of(s) for s in sh) for sh in shards]
    balanced = all(max(c[d] for c in per) - min(c[d] for c in per) <= 1 for d in ("bfm", "ict", "gnm"))
    same1 = dv.shard_subjects(subjects, 0, 1, 1234) == list(subjects)
    return {"pass": bool(disjoint and balanced and same1), "per_rank": [dict(c) for c in per]}


def t_sqrt_area() -> dict:
    g = torch.Generator().manual_seed(0)
    V = torch.randn(400, 3, generator=g)
    F = torch.randint(0, 400, (700, 3), generator=g)
    mass = torch.rand(400, generator=g) + 1e-3
    A = dv.reframe_sqrt_area(V, mass, F)
    B = dv.reframe_sqrt_area(V * 3.7 + 1.3, mass, F)
    area = float(dv._total_area(A, F))
    ok = torch.allclose(A, B, atol=1e-5) and abs(area - 1.0) < 1e-4
    return {"pass": bool(ok), "max_diff_frame": float((A - B).abs().max()), "area": area}


def _loss_args():
    return SimpleNamespace(lambda_rank=0.5, rank_pairs=1024, rank_margin=0.05, rank_tau=0.02, rank_hard_frac=0.7,
                           train_pair_mode="cross_topology", use_id_loss=True, lambda_id=0.25, lambda_subject=1.0,
                           lambda_mesh=1.0, log_pair_mode="all", log_eps=1e-6, log_eps_gt=1e-4, log_huber_delta=0.25,
                           w_cross=1.0, lambda_scale=1.0, scale_target=1.0, lambda_inv=1.0, lambda_nbr=0.5)


def t_losses() -> dict:
    subjects = [f"id{20000 + i}" for i in range(5)]
    mesh_s = [s for s in subjects for _ in range(6)]
    topos = ["original", "remesh", "down8k", "crop", "noisy", "up60k"] * 5
    n2i = {s: i for i, s in enumerate(subjects)}
    D = np.random.default_rng(0).uniform(0.1, 1.0, (5, 5)).astype(np.float32)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    out, ok = {}, True
    for name in LOSSES:
        Z = torch.randn(30, 16, requires_grad=True)
        loss, terms = compute_loss(name, StepBatch(Z, mesh_s, topos, subjects, D.view(dv.NanGuardedMatrix), n2i),
                                   _loss_args())
        loss.backward()
        good = bool(torch.isfinite(loss) and torch.isfinite(Z.grad).all() and Z.grad.abs().sum() > 0)
        # invarianza di scala della parte log (b e denominatori con gradiente): L_grad e L_inv non cambiano
        ok &= good
        out[name] = {"loss": float(loss.item()), "terms": terms, "finite_grad": good}
    Z = torch.randn(30, 16)
    a1 = compute_loss("log+inv", StepBatch(Z, mesh_s, topos, subjects, D.view(dv.NanGuardedMatrix), n2i), _loss_args())[1]
    a2 = compute_loss("log+inv", StepBatch(Z * 7.0, mesh_s, topos, subjects, D.view(dv.NanGuardedMatrix), n2i),
                      _loss_args())[1]
    inv = abs(a1["grad"] - a2["grad"]) < 1e-5 and abs(a1["inv"] - a2["inv"]) < 1e-5 and a2["scale"] > a1["scale"] - 1e-9
    out["scale_invariance_grad_inv"] = {"grad": [a1["grad"], a2["grad"]], "inv": [a1["inv"], a2["inv"]],
                                        "scale_anchor": [a1["scale"], a2["scale"]], "pass": bool(inv)}
    out["pass"] = bool(ok and inv)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {}
    for name, fn in (("fast_data", t_fast_data), ("compact_cache", t_compact), ("balanced_sampler", t_balanced), ("mixed_batches", t_mixed),
                     ("ddp_shards", t_shard), ("sqrt_area", t_sqrt_area), ("losses", t_losses)):
        try:
            res[name] = fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            res[name] = {"pass": False, "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()}
        print(name, json.dumps(res[name], default=str)[:600], flush=True)
    res["all_pass"] = all(r.get("pass") for r in res.values())
    a.out.write_text(json.dumps(res, indent=1, default=str))
    print("TEST UNITARI:", "PASSATI" if res["all_pass"] else "FALLITI")


if __name__ == "__main__":
    main()
