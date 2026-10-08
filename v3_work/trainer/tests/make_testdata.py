#!/usr/bin/env python3
"""Dati di prova per i test del trainer v3: un sottoinsieme REALE del run su scala, piccolo.

Stesse sorgenti del run su scala (la sua spec.json: viste BFM e ICT-5000, shard ICT_SCALE e GNM con
l'indice), uno split ridotto e la GT ritagliata dalla matrice congiunta (letta in memmap, valori identici).

    aau/run.sh v3_work/trainer/tests/make_testdata.py --out-dir aau/runs/evidence/trainer_v3/testdata \
        --n-bfm 15 --n-ict-view 30 --n-ict-tar 30 --n-gnm 30 --n-blocks 2

Scrive: split.json (train ridotto; heldout = online_eval + online_eval_extra dello split del run su scala),
spec.json (la spec del run su scala con n_blocks cambiato), gt.npz (D_orig ritagliata + names) e gt.json
(con il global_max della GT intera, cosi' la guardia --gt-keep-scale si comporta come nel run vero).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "v3_work/trainer"))
from data_v3 import npz_member_memmap  # noqa: E402

SCALE_RUN = REPO / "aau/runs/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411"
SPLIT = REPO / "aau/data_scale/split_scale_all.json"
GT = REPO / "datasets/SCALE_ALL/gt_joint_bfm_ict_gnm.npz"


def num(s: str) -> int:
    return int(s[2:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-bfm", type=int, default=15)
    ap.add_argument("--n-ict-view", type=int, default=30)
    ap.add_argument("--n-ict-tar", type=int, default=30)
    ap.add_argument("--n-gnm", type=int, default=30)
    ap.add_argument("--n-blocks", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="0 = i primi in ordine; altrimenti scelta casuale")
    ap.add_argument("--gt-src", type=Path, default=GT, help="GT da ritagliare (stesso formato e stessi names)")
    ap.add_argument("--gt-name", default="gt", help="nome dell'uscita: <gt-name>.npz e .json")
    ap.add_argument("--gt-only", action="store_true", help="solo la GT (split e spec gia' scritti in --out-dir)")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.gt_only:
        tiny = json.loads((a.out_dir / "split.json").read_text())
        cut_gt(a, tiny["train"], tiny["heldout"])
        return
    split = json.loads(SPLIT.read_text())
    train = split["train"]
    kinds = {
        "bfm": [s for s in train if num(s) < 1000],
        "ict_view": [s for s in train if 10000 <= num(s) < 20000],
        "ict_tar": [s for s in train if 20000 <= num(s) < 100000],
        "gnm": [s for s in train if 100000 <= num(s) < 200000],
    }
    want = {"bfm": a.n_bfm, "ict_view": a.n_ict_view, "ict_tar": a.n_ict_tar, "gnm": a.n_gnm}
    rng = np.random.default_rng(a.seed)
    picked = []
    for k, pool in kinds.items():
        pool = sorted(pool)
        sel = pool[: want[k]] if a.seed == 0 else sorted(rng.choice(pool, size=want[k], replace=False).tolist())
        picked += sel
        print(f"{k}: {len(sel)} di {len(pool)} (primi {sel[:3]})")
    extra = split.get("online_eval_extra") or {}
    held = sorted(set(split["online_eval"]) | {s for ids in extra.values() for s in ids})
    tiny = {"train": sorted(picked), "heldout": held, "online_eval": split["online_eval"],
            "online_eval_extra": extra,
            "note": f"sottoinsieme di prova di {SPLIT.name} (make_testdata.py, seed {a.seed})"}
    (a.out_dir / "split.json").write_text(json.dumps(tiny, indent=1))

    spec = json.loads((SCALE_RUN / "spec.json").read_text())
    spec["n_blocks"] = int(a.n_blocks)
    (a.out_dir / "spec.json").write_text(json.dumps(spec, indent=1))

    cut_gt(a, picked, held)


def cut_gt(a, picked, held) -> None:
    GT = a.gt_src
    with np.load(GT, allow_pickle=True) as z:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in z["names"]]
    pos = {}
    for i, n in enumerate(names):
        m = re.search(r"(id\d+)", n, re.IGNORECASE)          # come extract_subject_id(SUBJECT_RE_ANY)
        if m:
            pos[m.group(1).lower()] = i
    ids = sorted(set(picked) | set(held), key=num)
    missing = [s for s in ids if s not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} soggetti senza riga nella GT (primo {missing[0]})")
    idx = np.asarray([pos[s] for s in ids])
    D = npz_member_memmap(GT, "D_orig")
    order = np.argsort(idx)
    rows = np.asarray(D[idx[order]])                     # letture ordinate per riga
    sub = np.empty((len(ids), len(ids)), dtype=D.dtype)
    sub[order] = rows[:, idx]
    np.savez(a.out_dir / f"{a.gt_name}.npz", D_orig=sub, names=np.asarray([names[i] for i in idx]))
    side = json.loads(GT.with_suffix(".json").read_text())
    side.update(note=f"ritaglio di {GT.name} su {len(ids)} soggetti (make_testdata.py); valori identici",
                n_total=len(ids))
    (a.out_dir / f"{a.gt_name}.json").write_text(json.dumps(side, indent=1))
    print(f"GT ritagliata {sub.shape}, NaN {int(np.isnan(sub).sum())}, max {np.nanmax(sub):.4f}")


if __name__ == "__main__":
    main()
