#!/usr/bin/env python
"""Staging a blocchi contro store in mmap: stessi soggetti, blocchi, piano dell'epoca (in nomi di mesh) e tensori?

    aau/run.sh v3_work/trainer/tests/debug_store.py --testdata ... --split split_views.json --store <dir> --stage <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))
import train_v3 as T3  # noqa: E402
import sampler_v3 as sv  # noqa: E402
from robustness.noise import parse_noise_mode_weights, parse_noise_modes  # noqa: E402
from test_equivalence import RECIPE  # noqa: E402


def make(a, extra):
    args = T3.build_parser().parse_args(RECIPE + [
        "--seed", "1234", "--dist_npz", str(a.testdata / "gt.npz"), "--runs_root", "/tmp/x", "--split-json",
        str(a.testdata / a.split), "--data-spec", str(a.testdata / "spec.json"), "--total-steps", "24",
        "--steps-per-epoch", "3", "--gt-keep-scale", "--eval_domain", "bfm", "--fast-data", "--compact-cache",
        "--prepass-proc", "8"] + extra)
    T3.check_args(args)
    T3.dv.FAST["on"] = True
    d = T3.Data(args, 0, 1)
    d.setup(8)
    d.switch_block(0)
    return args, d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdata", type=Path, required=True)
    ap.add_argument("--split", default="split_views.json")
    ap.add_argument("--store", required=True)
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()
    argsA, A = make(a, ["--stage-root", a.stage])
    argsB, B = make(a, ["--store", a.store])
    print("train uguale:", A.train == B.train, len(A.train), "| blocchi uguali:", A.blocks == B.blocks)
    print("online uguale:", A.online == B.online)
    modes = parse_noise_modes(argsA.noise_modes)
    probs = parse_noise_mode_weights(argsA.noise_mode_weights, modes)
    cfg = sv.DrawCfg(argsA.p_noise, argsA.sigma_min, argsA.sigma_max, modes, probs, argsA.max_meshes_per_subject_train)
    tm_a = {s: {k: [A.files[i] for i in v] for k, v in A.topo_map[s].items()} for s in A.train}
    tm_b = {s: {k: [B.files[i] for i in v] for k, v in B.topo_map[s].items()} for s in B.train}
    print("mappa topologie uguale (in nomi):", tm_a == tm_b)
    if tm_a != tm_b:
        for s in A.train:
            if tm_a[s] != tm_b.get(s):
                print("  primo diverso", s, tm_a[s], tm_b.get(s))
                break
    pa = T3.epoch_plans(argsA, A, 1, 3, 3, 5, cfg, 1234)
    pb = T3.epoch_plans(argsB, B, 1, 3, 3, 5, cfg, 1234)
    na = [[(e[0], A.files[e[1]], e[3]) for e in p.entries] for p in pa]
    nb = [[(e[0], B.files[e[1]], e[3]) for e in p.entries] for p in pb]
    print("piano dell'epoca 1 uguale:", na == nb, "| sigma:", [p.sigma for p in pa], [p.sigma for p in pb])
    if na != nb:
        print("  A:", na[0][:6])
        print("  B:", nb[0][:6])
    worst = 0.0
    for p in pa:
        for e in p.entries:
            sa = A.dataset[e[1]]
            sb = B.dataset[B.files.index(A.files[e[1]])]
            for k in ("verts", "mass", "evals", "evecs"):
                worst = max(worst, float((sa[k] - sb[k]).abs().max()))
            for k in ("gradX", "gradY"):
                assert torch.equal(sa[k].indices(), sb[k].indices()) and torch.equal(sa[k].values(), sb[k].values())
    print("tensori del piano: max |delta| denso", worst)


if __name__ == "__main__":
    main()
