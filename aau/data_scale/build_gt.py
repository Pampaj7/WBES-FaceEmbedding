#!/usr/bin/env python3
"""Matrice GT congiunta BFM + ICT esteso (ICT-5000 + identita' nuove degli shard).

    aau/run.sh aau/data_scale/build_gt.py --shards-dir datasets/ICT_SCALE/shards \
        --out datasets/ICT_SCALE/gt_joint_bfm_ict.npz

Stessa GT di sempre: vertex-mean-L2 fra le ``original`` normalizzate maxabs
(``v2_work/genict/build_ict_gt_matrix.py``, via ``pairdist.vertex_mean_l2_matrix`` su GPU).
Stessa convenzione del congiunto attuale (``datasets/JOINT_BFM_ICT``, ``make_joint_view.py``):
  * blocco BFM: i valori del congiunto attuale, invariati (max 0.834);
  * blocco ICT: diviso per il SUO massimo, come ogni file GT del repo ("divided by its own max
    so the largest pair is exactly 1"). Con piu' identita' il massimo cresce, quindi le coppie
    di ICT-5000 si riscalano di ``old_max / new_max``: il fattore e' nel manifest. Il massimo
    globale resta 1, quindi ``load_gt_distance_matrix`` non riscala niente;
  * coppie fra domini: NaN (train_v2 rifiuta di leggerle).
Controllo: le coppie di ICT-5000 ricalcolate qui coincidono con la matrice in uso (prima del
riscalamento), altrimenti il file non viene scritto.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "v2_work/genict"))
from pairdist import vertex_mean_l2_matrix  # noqa: E402

DS = REPO_ROOT / "datasets"
JOINT = DS / "JOINT_BFM_ICT/gt_matrix.npz"
ICT_GT = DS / "ICT/gt/ict_matrix_distances_maxabs.npz"


def maxabs(V: np.ndarray) -> np.ndarray:
    Vc = V - V.mean(0, keepdims=True)
    return Vc / max(float(np.abs(Vc).max()), 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    t0 = time.time()
    with np.load(ICT_GT) as z:
        old_raw = [str(n) for n in z["names"]]                       # ict0000..ict4999
        D_old_stored = z["D_orig"].astype(np.float64)
    names, V = [], []
    for r in old_raw:
        with np.load(DS / f"ICT/topo/{r}_GTready_original.npz") as z:
            V.append(maxabs(z["V"].astype(np.float64)))
        names.append(f"id{10000 + int(r[3:]):05d}")
    n_old = len(names)
    tars = sorted(a.shards_dir.glob("shard_*.tar"))
    for t in tars:
        with tarfile.open(t) as tar:
            for m in tar:
                if m.name.endswith("_GTready_original.npz"):
                    with np.load(io.BytesIO(tar.extractfile(m).read())) as z:
                        V.append(maxabs(z["V"].astype(np.float64)))
                    names.append(m.name.split("_GTready_")[0])
    if len(set(names)) != len(names):
        raise SystemExit("nomi duplicati fra ICT-5000 e shard")
    print(f"{n_old} ICT-5000 + {len(names) - n_old} nuove da {len(tars)} shard, "
          f"lette in {time.time() - t0:.0f}s", flush=True)

    D = vertex_mean_l2_matrix(np.stack(V).astype(np.float32), device=a.device, block=1)
    man_scale = json.loads((DS / "ICT/gt/manifest.json").read_text())["normalization_scale"]["maxabs"]
    rel = np.abs(D[:n_old, :n_old] / man_scale - D_old_stored).max()
    if rel > 1e-4:
        raise SystemExit(f"le coppie di ICT-5000 ricalcolate differiscono dalla GT in uso: {rel:.2e}")
    new_max = float(D.max())
    D_ict = (D / new_max).astype(np.float32)

    with np.load(JOINT) as z:
        jn = [str(n) for n in z["names"]]
        DJ = z["D_orig"]
    bfm_idx = [i for i, n in enumerate(jn) if int(n[2:]) < 1000]
    bfm_names = [jn[i] for i in bfm_idx]
    D_bfm = DJ[np.ix_(bfm_idx, bfm_idx)].astype(np.float32)

    nb, ni = len(bfm_names), len(names)
    out = np.full((nb + ni, nb + ni), np.nan, dtype=np.float32)
    out[:nb, :nb] = D_bfm
    out[nb:, nb:] = D_ict
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, D_orig=out, names=np.array(bfm_names + names))
    man = {"n_bfm": nb, "n_ict_old": n_old, "n_ict_new": ni - n_old,
           "bfm_block_max": float(np.nanmax(D_bfm)), "ict_block_max": 1.0,
           "ict_raw_max_old": float(man_scale), "ict_raw_max_new": new_max,
           "ict5000_rescale_factor": float(man_scale / new_max),
           "ict5000_recompute_max_rel_diff": float(rel), "shards": [t.name for t in tars],
           "seconds": time.time() - t0}
    a.out.with_suffix(".json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
