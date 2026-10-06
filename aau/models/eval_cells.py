#!/usr/bin/env python3
"""Eval in-domain BFM delle ablazioni v3: gruppi dell'autore e 30 celle ordinate del repo, da un
solo set di embedding.

Stessi record, stesso split e stesse coppie di v2_work/potential/eval_by_topology.py (100
held-out del seed del checkpoint, 6 topologie, coppie triu cross-soggetto), da cui:
  - groups: crop / noisy / resample / all, UNO Spearman su tutte le coppie del gruppo, come
    eval_by_topology (numeri confrontabili con aau/runs/eval_frame/*);
  - cells: le 30 celle ordinate A->B dello stage topology del repo (soggetto minore in A,
    maggiore in B, 4950 coppie), cioe' la definizione di aau/scratch/eval_frame_crosscheck.py,
    validata contro il repo a 4e-4 sul v1. La media delle celle e' il termine latent del
    margine latent - Chamfer del criterio (Chamfer e' lo stesso per tutti i bracci: stessi
    soggetti, stesse mesh).

Il frame e il token li mettono gli hook: si lancia da aau/models/eval_ablation.py.

    aau/run.sh aau/models/eval_ablation.py --frame rms --size-token-json J -- \
        aau/models/eval_cells.py --checkpoint <pth> --data-dir <ops> --dist-npz <gt> --tag T --out-dir D
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2_work/potential"))
sys.path.insert(0, str(THIS_DIR))

import eval_by_topology as ebt  # noqa: E402  (fa da se' i sys.path del repo)
from robustness.data_utils import rebuild_subject_split  # noqa: E402

GROUPS = ("crop", "noisy", "resample", "all")
TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--dist-npz", type=Path, required=True)
    ap.add_argument("--n-subjects", type=int, default=100)
    ap.add_argument("--subject-split", default="eval", choices=["eval", "all"],
                    help="all: tutti i soggetti della data dir (solo smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    import ablation_hooks
    frame = ablation_hooks._STATE["eval_frame"] or "current"
    table = ablation_hooks._STATE["table"]

    ckpt, cfg = ebt.resolve_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    model = ebt.build_model(args=SimpleNamespace(**cfg), device=device)
    pack = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(pack["state_dict"] if "state_dict" in pack else pack, strict=True)
    model.eval()
    print(f"[{args.tag}] modello {type(model).__name__}, frame {frame}, "
          f"token {None if table is None else table.path}", flush=True)

    dataset = ebt.GTReadyDataset(str(args.data_dir))
    subj_map = ebt.build_subject_map(dataset.files, subject_re=ebt.SUBJECT_RE_ANY)
    # SUBJECT_RE_ANY: il default a 4 cifre tronca gli id ICT (id14500 -> id1450) e l'intersezione con
    # la vista resta vuota (jobs 1055742/1055744); sulla GT BFM i due regex danno lo stesso mapping.
    gt, name_to_idx = ebt.load_gt_distance_matrix(str(args.dist_npz), subject_re=ebt.SUBJECT_RE_ANY,
                                                  dtype=np.float64)
    subjects = sorted(set(subj_map) & set(name_to_idx))
    if not subjects:
        raise SystemExit(f"nessun soggetto in comune fra {args.data_dir} e {args.dist_npz}")
    if args.subject_split == "eval":
        _, subjects = rebuild_subject_split(subjects=subjects, eval_fraction=0.2,
                                            seed=int(cfg["seed"]), max_subjects=0)
    rng = np.random.default_rng(1234)                   # come eval_by_topology (--seed 1234)
    if 0 < args.n_subjects < len(subjects):
        pick = np.sort(rng.choice(len(subjects), size=args.n_subjects, replace=False))
        subjects = [subjects[int(i)] for i in pick]
    plan = ebt.build_eval_plan(subj_map=subj_map, eval_subjects=subjects,
                               max_meshes_per_subject_eval=0, seed=1234)
    records = ebt.build_sample_eval_records(dataset=dataset, eval_plan=plan,
                                            eval_subjects=subjects, sample_cache=None)
    print(f"[{args.tag}] seed={cfg['seed']} {len(subjects)} soggetti, {len(records)} mesh", flush=True)

    t0 = time.time()
    Z = []
    with torch.inference_mode():
        for i, rec in enumerate(records):
            s = ebt.sample_to_device(dataset[int(rec.dataset_idx)], device=device)
            z, _ = ebt.forward_model(model=model, sample_dict=s, V_in=s["verts"],
                                     return_gate_info=False, add_noise=False)
            Z.append(z.squeeze(0))
            if (i + 1) % 200 == 0:
                print(f"[{args.tag}] {i+1}/{len(records)} ({(i+1)/(time.time()-t0):.1f}/s)", flush=True)
    Z = torch.stack(Z, dim=0).cpu().numpy().astype(np.float64)

    subj = np.array([r.subject_id for r in records], dtype=object)
    topo = np.array([r.topology_label for r in records], dtype=object)
    gt_idx = np.array([name_to_idx[r.subject_id] for r in records], dtype=int)
    iu, ju = np.triu_indices(len(records), 1)
    keep = subj[iu] != subj[ju]
    iu, ju = iu[keep], ju[keep]
    d_lat = np.linalg.norm(Z[iu] - Z[ju], axis=1)
    d_gt = gt[gt_idx[iu], gt_idx[ju]]
    groups = np.array([ebt.group_of(a, b) for a, b in zip(topo[iu], topo[ju])], dtype=object)

    out = {"tag": args.tag, "checkpoint": str(ckpt), "data_dir": str(args.data_dir), "frame": frame,
           "size_token_json": None if table is None else str(table.path), "model": type(model).__name__,
           "seed": int(cfg["seed"]), "n_subjects": len(subjects), "subjects": list(subjects),
           "groups": {}, "cells": []}
    print(f"\n{'group':10s} {'pairs':>9s} {'Spearman':>10s}")
    for g in GROUPS[:3]:
        m = groups == g
        if m.sum() < 3:
            continue
        rho = float(ebt.spearman_corr(d_gt[m], d_lat[m]))
        out["groups"][g] = {"n_pairs": int(m.sum()), "spearman": rho}
        print(f"{g:10s} {int(m.sum()):9d} {rho:10.4f}")
    m_all = groups != None  # noqa: E711
    out["groups"]["all"] = {"n_pairs": int(m_all.sum()),
                            "spearman": float(ebt.spearman_corr(d_gt[m_all], d_lat[m_all]))}
    print(f"{'all':10s} {int(m_all.sum()):9d} {out['groups']['all']['spearman']:10.4f}")

    for a in TOPOLOGIES:
        for b in TOPOLOGIES:
            if a == b:
                continue
            m = (topo[iu] == a) & (topo[ju] == b)
            if m.sum() < 3:
                continue
            out["cells"].append({"a": a, "b": b, "group": ebt.group_of(a, b), "n": int(m.sum()),
                                 "latent_spearman": float(ebt.spearman_corr(d_gt[m], d_lat[m]))})
    cells = out["cells"]
    out["mean_cells"] = {
        "all": float(np.mean([c["latent_spearman"] for c in cells])),
        "crop": float(np.mean([c["latent_spearman"] for c in cells if c["group"] == "crop"])),
        "n_cells": len(cells)}
    print(f"\n{len(cells)} celle ordinate: media {out['mean_cells']['all']:.4f}, "
          f"con crop {out['mean_cells']['crop']:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.tag}.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(args.out_dir / f"{args.tag}_latents.npz", Z=Z, subject=subj.astype(str),
                        topology=topo.astype(str))
    print(f"\nscritto {args.out_dir / f'{args.tag}.json'}")


if __name__ == "__main__":
    main()
