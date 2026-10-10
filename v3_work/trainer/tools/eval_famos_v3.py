#!/usr/bin/env python3
"""FaMoS TEST (persone reali mai viste, scansioni grezze) per i checkpoint v3 a ingresso globale: riconoscimento e
graduata con le GT di E12 (FR, SR) e l'unificata.

Gira DENTRO eval_v3 (agganci del modello e della scala):

    WBES_V3_SCALE_TABLES=<famos.npz> WBES_V3_FACTORIZED_OUT=full \
    aau/run.sh v3_work/trainer/eval_v3.py -- v3_work/trainer/tools/eval_famos_v3.py --ops-dir <ops> \
        --checkpoint <epochNNN_ema.pth> --tag <tag> --out-dir <dir>

Stessi dati, blocchi, bootstrap e Chamfer di aau/famos/famos_eval.py (funzioni importate): le patch T7 di
``datasets/FAMOS/test_view`` con la scala METRICA (la tabella annulla la similarita' verso il template, area /
scale_to_mm^2: build_scale_table.py --scale-csv). Distanze del modello: ``factorized`` -> d_F (form, mm), d_P
(forma) e |delta s|; ``dual`` -> ||delta z_F|| e ||delta u|| (latenti [z_F, u]); testa standard -> ||z_a - z_b||.
GT: ``fr``/``sr`` (``datasets/CANONICAL_GT/eval/famos_test_*``) e ``unified`` (``test_view/gt_matrix.npz``).
Con ``famos_test_centroid_size.npz``: s medio per persona contro log S.
Uscite: recognition.csv, graded.csv, results.json in ``--out-dir``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
for _p in (TRAINER, REPO / "aau/famos", REPO / "aau/recon"):
    sys.path.insert(0, str(_p))

import factorized_v3 as fz  # noqa: E402
import famos_common as fc  # noqa: E402
import famos_eval as fe  # noqa: E402  (aau/famos, sola lettura)

GT_DIR = REPO / "datasets/CANONICAL_GT/eval"


def gt_by_id(path: Path, ids: list[str], scale: float = 1.0) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        pos = {str(n): k for k, n in enumerate(z["names"])}
        D = np.asarray(z["D_orig"], np.float64) * scale
    ii = np.asarray([pos[s] for s in ids])
    return D[np.ix_(ii, ii)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ops-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-chamfer", action="store_true")
    a = p.parse_args()
    import torch
    cargs = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["args"]
    factorized = cargs.get("head", "embed") in ("factorized", "factorized2")

    rows = fe.read_manifest()
    names = [r["name"] for r in rows]
    kind = np.asarray([r["kind"] for r in rows])
    role = np.asarray([r["role"] for r in rows])
    with np.load(fc.VIEW_DIR / "gt_matrix.npz") as z:
        ids = [str(x) for x in z["names"]]
    s2i = {s: k for k, s in enumerate(ids)}
    subj = np.asarray([s2i[r["view_id"]] for r in rows])
    gal = {k: np.flatnonzero((kind == k) & (role == "gallery")) for k in ("scan", "reg")}
    gal = {k: g[np.argsort(subj[g])] for k, g in gal.items()}
    gal_all = np.concatenate([gal["scan"], gal["reg"]])
    gts = {"fr": gt_by_id(GT_DIR / "famos_test_fr.npz", ids), "sr": gt_by_id(GT_DIR / "famos_test_sr.npz", ids),
           "unified": gt_by_id(fc.VIEW_DIR / "gt_matrix.npz", ids)}

    Z = fe.embed(a.ops_dir, names, a.checkpoint, a.tag, a.device)
    ii = np.repeat(np.arange(len(Z)), len(gal_all))
    jj = np.tile(gal_all, len(Z))
    if factorized:
        if Z.shape[1] != 257:
            raise SystemExit("servono i latenti [s, u]: WBES_V3_FACTORIZED_OUT=full")
        side = json.loads(Path(cargs["dist_npz"]).with_suffix(".json").read_text())
        dpu = float(side.get("dp_per_unit", side.get("dP_per_unit"))) / float(cargs.get("gt_scale", 1.0))
        d = fz.pair_distances(Z, ii, jj, dpu)
        D = {f"{a.tag}_form": d["form_mm"], f"{a.tag}_shape": d["dP"], f"{a.tag}_size": d["size_abs"]}
    elif cargs.get("head", "embed") == "dual":
        if Z.shape[1] != 2 * int(cargs["latent_dim"]):
            raise SystemExit("servono i latenti [z_F, u]: WBES_V3_FACTORIZED_OUT=full")
        D = {f"{a.tag}_{k}": v for k, v in fz.model_distances(Z, ii, jj, "dual").items()}
    else:
        D = {a.tag: np.linalg.norm(Z[ii] - Z[jj], axis=1)}
    D = {k: v.reshape(len(Z), len(gal_all)) for k, v in D.items()}
    if not a.no_chamfer:
        ch = fe.chamfer_to_gallery([fc.VIEW_DIR / "npz" / f"{n}.npz" for n in names], gal_all, a.workers)
        D.update({c: ch[c] for c in fe.CHAMFERS})
    col = {k: {int(i): j for j, i in enumerate(gal_all) if kind[i] == k} for k in ("scan", "reg")}
    gcols = lambda gk: np.asarray([col[gk][int(i)] for i in gal[gk]])  # noqa: E731
    counts = fe.bootstrap_counts(len(ids), a.n_bootstrap, a.seed)
    rec_rows, gr_rows = [], []
    for qk, qr, gk in fe.RECOG_BLOCKS:
        q_idx = np.flatnonzero((kind == qk) & (role == qr))
        for m, Dm in D.items():
            v = fe.recognition(Dm[:, gcols(gk)], q_idx, subj[q_idx], np.arange(len(ids)), counts)
            rec_rows.append({"block": f"{qk} {qr} -> {gk}", "method": m, "n_queries": len(q_idx),
                             **{f"{k}{s}": x for k in ("rank1", "auc") for s, x in
                                zip(("", "_ci_low", "_ci_high"), (v[k][0], *fe.ci(v[k])))}})
    for qk, qr, gk in fe.GRADED_BLOCKS:
        q_idx = np.flatnonzero((kind == qk) & (role == qr))
        gc = gcols(gk)
        qa = np.repeat(q_idx, len(ids))
        gb = np.tile(np.arange(len(ids)), len(q_idx))
        sa, sb = subj[qa], gb
        keep = sa != sb
        if qr == "gallery" and qk == gk:
            keep &= sa < sb
        qa, gb, sa, sb = qa[keep], gb[keep], sa[keep], sb[keep]
        for gname, G in gts.items():
            for m, Dm in D.items():
                v = fe.spearman_reps(Dm[qa, gc[gb]], G[sa, sb], sa, sb, counts)
                gr_rows.append({"block": f"{qk} {qr} -> {gk}", "gt": gname, "method": m, "n_pairs": int(len(qa)),
                                "spearman": v[0], "ci_low": fe.ci(v)[0], "ci_high": fe.ci(v)[1]})
    out = {"checkpoint": str(a.checkpoint), "tag": a.tag, "head": cargs.get("head", "embed"),
           "n_meshes": len(rows), "n_subjects": len(ids), "latent_finite": bool(np.isfinite(Z).all())}
    if factorized and (GT_DIR / "famos_test_centroid_size.npz").exists():
        from scipy.stats import spearmanr
        lcs = fz.load_log_cs(GT_DIR / "famos_test_centroid_size.npz")
        sm = np.asarray([Z[subj == k, 0].mean() for k in range(len(ids))])
        t = np.asarray([lcs[s] for s in ids])
        out["size_accuracy"] = {"spearman_subject_s_vs_log_cs": float(spearmanr(sm, t).correlation),
                                "err_mean": float((sm - t).mean()), "err_median_abs": float(np.median(np.abs(sm - t)))}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for fname, rr in (("recognition.csv", rec_rows), ("graded.csv", gr_rows)):
        with open(a.out_dir / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    (a.out_dir / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    for r in gr_rows:
        if r["block"].startswith("scan gallery"):
            print(f"graduata {r['block']:28s} {r['gt']:8s} {r['method']:28s} {r['spearman']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
