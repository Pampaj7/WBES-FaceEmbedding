#!/usr/bin/env python
"""Correttezza (b): la GT al volo dello stream coincide con datasets/UNIFIED_GT per le identita' che vi compaiono.

    aau/run.sh v3_work/stream/tests/test_gt.py --out aau/runs/evidence/stream/gt_check.json

Identita' con coefficienti noti nei dati della GT unificata:
  ict    ICT-5000 (id10000-14999, datasets/ICT/identities) e ICT nuove di due shard di datasets/ICT_SCALE;
  gnm    GNM_DISTILL (id100000-), le identita' dei primi due shard;
  flame  le 1000 identita' di shapes.py (seme 1234, N(0, 1) sui 300 modi).
Per ognuna la strada dei produttori: MMSource(dominio).neutral_points(z) (basi proiettate sulla regione
unificata dalla patch della libreria v3_work/mm) -> Unified.svec; poi StreamGT, cioe' la GT che il trainer
calcola sul batch. Confronti:
  * s_i contro datasets/UNIFIED_GT/shapes/<dominio>.npz e (ict, gnm) contro train/s_train.npz: ||ds|| / sqrt(A), mm;
  * GT: tutte le coppie fra le identita' ict + gnm scelte (anche fra domini) contro D_orig di
    train/gt_unified_bfm_ict_gnm.npz (letto in memmap), in unita' della GT e in mm.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
STREAM = THIS.parent
REPO = STREAM.parents[1]
for _p in (STREAM, REPO / "v3_work" / "trainer"):
    sys.path.insert(0, str(_p))

UGT = REPO / "datasets" / "UNIFIED_GT"


def ict_ids_weights(n_old: int, shards: list[int], rng) -> tuple[list[str], np.ndarray]:
    import domains as DM
    sids, W = [], []
    for g in sorted(rng.choice(np.arange(10000, 15000), n_old, replace=False).tolist()):
        with np.load(DM.DATASETS / "ICT" / "identities" / f"ict{g - 10000:04d}.npz") as d:
            sids.append(f"id{g}")
            W.append(np.asarray(d["weights"], dtype=np.float64))
    for k in shards:
        for rec in DM._shard_manifest(DM.DATASETS / "ICT_SCALE" / "shards" / f"shard_{k:05d}.tar")["identities"]:
            sids.append(rec["sid"])
            W.append(np.asarray(rec["weights"], dtype=np.float64))
    return sids, np.stack(W)


def gnm_ids_weights(n_tars: int) -> tuple[list[str], np.ndarray]:
    import domains as DM
    sids, W = [], []
    for tar in sorted((DM.DATASETS / "GNM_DISTILL" / "shards").glob("gnm_shard_*.tar"))[:n_tars]:
        for rec in DM._shard_manifest(tar)["identities"]:
            sids.append(rec["sid"])
            W.append(np.asarray(rec["weights"], dtype=np.float64))
    return sids, np.stack(W)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-ict-old", type=int, default=100)
    ap.add_argument("--ict-shards", default="0,117")
    ap.add_argument("--gnm-tars", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    import sources as S
    from consumer import StreamGT
    sys.path.insert(0, str(REPO / "v3_work" / "trainer"))
    import data_v3 as dv

    rng = np.random.default_rng(a.seed)
    uni = S.Unified()
    sqA = np.sqrt(uni.A)
    gt_mm = S.gt_scale_mm()
    sets = {"ict": ict_ids_weights(a.n_ict_old, [int(x) for x in a.ict_shards.split(",")], rng),
            "gnm": gnm_ids_weights(a.gnm_tars),
            "flame2020": ([f"flame{k:04d}" for k in range(1000)], np.random.default_rng(1234).normal(size=(1000, 300)))}
    shapes_file = {"ict": "ict", "gnm": "gnm", "flame2020": "flame"}
    with np.load(UGT / "train" / "s_train.npz") as z:
        train_pos = {str(n): i for i, n in enumerate(z["names"])}
        S_train = z["s"]
    out = {"area_total": uni.A, "gt_mm_per_unit": gt_mm, "domains": {}}
    S_stream = {}
    for d, (sids, W) in sets.items():
        src = S.MMSource(d, uni)
        Sx = np.stack([uni.svec(src.neutral_points(w)) for w in W]).astype(np.float64)
        S_stream[d] = (sids, Sx)
        with np.load(UGT / "shapes" / f"{shapes_file[d]}.npz") as z:
            pos = {str(n): i for i, n in enumerate(z["ids"])}
            ref = z["s"][[pos[s] for s in sids]].astype(np.float64)
        dmm = np.linalg.norm(Sx - ref, axis=1) / sqA
        row = {"n": len(sids), "vs_shapes_mm": {"max": float(dmm.max()), "median": float(np.median(dmm))},
               "ids_first_last": [sids[0], sids[-1]]}
        if d in ("ict", "gnm"):
            reft = S_train[[train_pos[s] for s in sids]].astype(np.float64)
            dtr = np.linalg.norm(Sx - reft, axis=1) / sqA
            row["vs_s_train_mm"] = {"max": float(dtr.max()), "median": float(np.median(dtr))}
        out["domains"][d] = row
        print(f"[gt] {d}: {row}", flush=True)
    # GT del batch, come la calcola il trainer, contro D_orig (anche le coppie fra domini)
    gt = StreamGT(uni.A, gt_mm, keep=10 ** 6)
    names = []
    for d in ("ict", "gnm"):
        sids, Sx = S_stream[d]
        for s, v in zip(sids, Sx):
            gt.register(s, v, d)
            names.append(s)
    D = dv.npz_member_memmap(UGT / "train" / "gt_unified_bfm_ict_gnm.npz", "D_orig")
    idx = np.asarray([train_pos[s] for s in names])
    order = np.argsort(idx)
    ref = np.asarray(D[np.ix_(idx[order], idx[order])], dtype=np.float64)
    rows = np.asarray([gt.name_to_idx[names[i]] for i in order])
    got = gt[np.ix_(rows, rows)]
    diff = np.abs(got - ref)
    doms = np.asarray(["ict" if int(names[i][2:]) < 100000 else "gnm" for i in order])
    cross = doms[:, None] != doms[None, :]
    out["gt_pairs"] = {"n_identities": len(names), "n_pairs": int(len(names) * (len(names) - 1) / 2),
                       "n_cross_domain_pairs": int(cross.sum() // 2),
                       "max_abs_units": float(diff.max()), "max_abs_mm": float(diff.max() * gt_mm),
                       "max_abs_cross_domain_mm": float(diff[cross].max() * gt_mm),
                       "max_rel_offdiag": float((diff / np.where(ref > 0, ref, 1))[~np.eye(len(ref), dtype=bool)].max()),
                       "ref_median_units": float(np.median(ref[~np.eye(len(ref), dtype=bool)]))}
    print(f"[gt] coppie: {out['gt_pairs']}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
