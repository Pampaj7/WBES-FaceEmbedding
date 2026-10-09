#!/usr/bin/env python3
"""Valutazione "form" di un checkpoint fattorizzato contro le GT di E12 (form, shape, EDM, ...): l'aggancio.

Ingresso: gli embedding [s, u] di ogni mesh di una vista di eval (``embeddings.npz`` di aau/zs3dmm/zs_embed.py,
lanciato attraverso eval_v3 con ``WBES_V3_FACTORIZED_OUT=full`` e ``WBES_V3_SCALE_TABLES``: vedi
ablations/c3f/eval_body.sh, passo ``form``), e una o piu' GT ``--gt NOME=file.npz`` nel formato di
load_gt_distance_matrix (``D_orig``, ``names``; per E12: datasets/CANONICAL_GT/<set>_<variante>.npz, mm).

Distanze del modello per coppia di mesh di soggetti diversi e topologie diverse (``--pairs nocrop_cross``: senza
crop, come il primario del protocollo; ``all_cross``: tutte). Due livelli: ``mesh_pair`` (ogni coppia di mesh una riga
con la GT della sua coppia di soggetti: e' il ``nocrop_cross`` dei csv di E12, 99.000 righe su HIFI3D) e
``subject_pair_mean`` (media per coppia di soggetti). Distanze:
  * ``form``  d_F = sqrt((S_i - S_j)^2 + S_i S_j dP^2), S = exp(s) in mm (factorized_v3.form_distance);
  * ``shape`` dP = ||u_i - u_j|| * dp_per_unit (corda fra pre-forme a centroid size unitaria; rho = 2 asin(dP/2));
  * ``size``  |s_i - s_j| (differenza di log centroid size).
``dp_per_unit`` = 1/kappa della GT shape del training (``gt_shape.json`` accanto a ``--dist_npz`` del checkpoint),
oppure ``--dp-per-unit``. Per ogni (distanza, GT): Spearman sulle coppie di soggetti, IC 95% bootstrap per soggetto
(stesse repliche per tutte, ``--boot`` 1000, ``--seed`` 1234; replica = soggetti ricampionati, peso di una coppia =
prodotto dei conteggi, come evidence.boot_spearman di E8). Con ``--size-table`` (log S vero degli stessi soggetti)
anche l'accuratezza di s. Uscite: ``<out>/form_spearman.csv`` e ``<out>/form_eval.json``.

    aau/run.sh v3_work/trainer/tools/eval_factorized.py --embeddings <stage>/embeddings.npz \
        --gt form=datasets/CANONICAL_GT/hifi3d_canonical_centered.npz --gt shape=... --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
sys.path.insert(0, str(TRAINER))

import factorized_v3 as fz  # noqa: E402

ID_RE = re.compile(r"(id\d+)", re.IGNORECASE)


def load_gt(path: Path, subjects: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in z["names"]]
        D = np.asarray(z["D_orig"], dtype=np.float64)
    pos = {ID_RE.search(n).group(1).lower(): i for i, n in enumerate(names)}
    miss = [s for s in subjects if s not in pos]
    if miss:
        raise SystemExit(f"{path}: {len(miss)} soggetti assenti dalla GT (primo {miss[0]})")
    ii = np.asarray([pos[s] for s in subjects])
    return D[np.ix_(ii, ii)]


def ranks(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(x)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def boot_rows(model: dict, gts: dict, si: np.ndarray, sj: np.ndarray, n: int, n_boot: int, seed: int) -> list[dict]:
    """Come ``boot`` a livello di coppia di mesh: riga (i, j) con la GT G[s_i, s_j], replica per soggetto con peso
    di riga = conteggio(s_i) x conteggio(s_j)."""
    rng = np.random.default_rng(seed)
    gv = {k: G[si, sj] for k, G in gts.items()}
    reps = {(a, b): [] for a in model for b in gv}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        wt = counts[si].astype(np.int64) * counts[sj]
        keep = wt > 0
        for a, b in reps:
            reps[(a, b)].append(spearman(np.repeat(model[a][keep], wt[keep]), np.repeat(gv[b][keep], wt[keep])))
    out = []
    for (a, b), r in reps.items():
        lo, hi = np.percentile(r, [2.5, 97.5])
        out.append({"level": "mesh_pair", "distance": a, "gt": b, "point": spearman(model[a], gv[b]),
                    "ci_low": float(lo), "ci_high": float(hi), "n_subjects": int(n), "n_pairs": int(len(si))})
    return out


def boot(model: dict, gts: dict, n_boot: int, seed: int) -> list[dict]:
    """Spearman (punto e IC) per ogni coppia (distanza del modello, GT), repliche per soggetto condivise."""
    n = len(next(iter(model.values())))
    iu, ju = np.triu_indices(n, 1)
    mv = {k: m[iu, ju] for k, m in model.items()}
    gv = {k: m[iu, ju] for k, m in gts.items()}
    rng = np.random.default_rng(seed)
    reps = {(a, b): [] for a in mv for b in gv}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        wt = counts[iu].astype(np.int64) * counts[ju]
        keep = wt > 0
        for a, b in reps:
            reps[(a, b)].append(spearman(np.repeat(mv[a][keep], wt[keep]), np.repeat(gv[b][keep], wt[keep])))
    out = []
    for (a, b), r in reps.items():
        lo, hi = np.percentile(r, [2.5, 97.5])
        out.append({"level": "subject_pair_mean", "distance": a, "gt": b, "point": spearman(mv[a], gv[b]),
                    "ci_low": float(lo), "ci_high": float(hi),
                    "n_subjects": int(n), "n_pairs": int(len(iu))})
    return out


def dp_from_ckpt(ckpt: Path) -> float:
    import torch
    args = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
    side = Path(args["dist_npz"]).with_suffix(".json")
    info = json.loads(side.read_text())
    key = "dp_per_unit" if "dp_per_unit" in info else "dP_per_unit"     # nostro / E12 (GT-SR)
    if key not in info:
        raise SystemExit(f"{side}: senza dp_per_unit (la GT del checkpoint non e' una GT shape)")
    return float(info[key]) / float(args.get("gt_scale", 1.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--gt", action="append", default=[], help="NOME=file.npz (ripetibile)")
    ap.add_argument("--dp-per-unit", type=float, default=0.0, help="0 = dal checkpoint degli embedding")
    ap.add_argument("--pairs", default="nocrop_cross", choices=["nocrop_cross", "all_cross"])
    ap.add_argument("--size-table", type=Path, default=None, help="log S vero (names, log_cs_mm) dei soggetti valutati")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    with np.load(a.embeddings, allow_pickle=True) as z:
        Z = np.asarray(z["Z"], np.float64)
        subj = [str(s) for s in z["subjects"]]
        topo = [str(t) for t in z["topologies"]]
        ckpt = Path(str(z["checkpoint"])) if "checkpoint" in z.files else None
    import torch
    head = torch.load(ckpt, map_location="cpu", weights_only=False)["args"].get("head", "embed") if ckpt else "factorized"
    factorized = head in ("factorized", "factorized2")
    if factorized and Z.shape[1] != 257:
        raise SystemExit("embedding senza s: rilanciare zs_embed con WBES_V3_FACTORIZED_OUT=full")
    dpu = (a.dp_per_unit if a.dp_per_unit > 0 else dp_from_ckpt(ckpt)) if factorized else float("nan")
    subjects = sorted(set(subj))
    pos = {s: k for k, s in enumerate(subjects)}
    i, j = np.triu_indices(len(Z), 1)
    si, sj = np.asarray([pos[s] for s in subj])[i], np.asarray([pos[s] for s in subj])[j]
    ti, tj = np.asarray(topo)[i], np.asarray(topo)[j]
    keep = (si != sj) & (ti != tj)
    if a.pairs == "nocrop_cross":
        keep &= (ti != "crop") & (tj != "crop")
    i, j, si, sj = i[keep], j[keep], si[keep], sj[keep]
    d = fz.pair_distances(Z, i, j, dpu) if factorized else {"z": np.linalg.norm(Z[i] - Z[j], axis=1)}
    n = len(subjects)
    rows_model = {k_out: d[k_in] for k_out, k_in in (
        (("form", "form_mm"), ("shape", "dP"), ("size", "size_abs")) if factorized else (("z", "z"),))}
    model = {}
    pairs_out = (("form", "form_mm"), ("shape", "dP"), ("size", "size_abs")) if factorized else (("z", "z"),)
    for k_out, k_in in pairs_out:
        acc, cnt = np.zeros((n, n)), np.zeros((n, n))
        np.add.at(acc, (si, sj), d[k_in])
        np.add.at(cnt, (si, sj), 1.0)
        M = acc + acc.T
        C = cnt + cnt.T
        if (C[np.triu_indices(n, 1)] == 0).any():
            raise SystemExit("coppie di soggetti senza coppie di mesh: vista incompleta")
        np.fill_diagonal(C, 1.0)
        model[k_out] = M / C
    gts = {}
    for spec in a.gt:
        name, path = spec.split("=", 1)
        gts[name] = load_gt(Path(path), subjects)
    rows = (boot_rows(rows_model, gts, si, sj, n, a.boot, a.seed) + boot(model, gts, a.boot, a.seed)) if gts else []
    s_subj = np.asarray([Z[[k for k, x in enumerate(subj) if x == s], 0].mean() for s in subjects])
    out = {"embeddings": str(a.embeddings), "checkpoint": str(ckpt), "head": head, "dp_per_unit": dpu, "pairs": a.pairs,
           "n_subjects": n, "n_mesh_pairs": int(len(i)), "gts": {s.split("=", 1)[0]: s.split("=", 1)[1] for s in a.gt},
           "median_model": {k: float(np.median(m[np.triu_indices(n, 1)])) for k, m in model.items()},
           "spearman": rows}
    if factorized:
        out["S_mm_subject"] = {"median": float(np.median(np.exp(s_subj))),
                               "cv": float(np.exp(s_subj).std() / np.exp(s_subj).mean())}
    if a.size_table and factorized:
        lcs = fz.load_log_cs(a.size_table)
        have = [k for k, s in enumerate(subjects) if s in lcs]
        if have:
            t = np.asarray([lcs[subjects[k]] for k in have])
            out["size_accuracy"] = {"n_subjects": len(have), "spearman_s_vs_log_cs": spearman(s_subj[have], t),
                                    "err_mean": float((s_subj[have] - t).mean()),
                                    "err_median_abs": float(np.median(np.abs(s_subj[have] - t)))}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "form_eval.json").write_text(json.dumps(out, indent=1))
    if rows:
        import csv
        with open(a.out_dir / "form_spearman.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps({k: v for k, v in out.items() if k != "spearman"}, indent=1))
    for r in rows:
        print(f"{r['level']:17s} {r['distance']:6s} vs {r['gt']:12s} {r['point']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
