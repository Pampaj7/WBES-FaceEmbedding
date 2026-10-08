#!/usr/bin/env python3
"""Prova della pipeline sul set di test reale FaMoS: e108 e Chamfer, riconoscimento e distanza graduata.

    aau/run.sh aau/famos/famos_eval.py --ops-dir /tmp/<job>/ops --checkpoint <epoch108.pth> --tag e108 \\
        --out-dir aau/runs/evidence/famos/eval_e108 --workers 16
    (famos_eval.sbatch, che calcola prima gli operatori su /tmp)

Metodi e distanze, su tutte le mesh di ``datasets/FAMOS/test_view/npz``:
  - modello: ||z_a - z_b|| fra i latenti, la catena di NoW (``ws3b_latent.build_v1_model`` ed
    ``encode`` importate: merge_run_args -> build_model -> GTReadyDataset -> forward_model senza rumore)
    sugli operatori ad area unitaria k_eig 128 (``v2_work/potential/areanorm_operators.py``, come
    zs_zeroshot.sbatch e now_ops_latent.sbatch). Cache in ``datasets/FAMOS/eval/embeddings``;
  - ``chamfer_full`` e ``chamfer_stable``: ``aau/zs3dmm/zs_region_chamfer.py`` importata (maxabs per
    mesh, 4096 punti per area col seme del nome, Chamfer simmetrica dei quadrati; la regione stabile col
    frame ``hifi``: +y alto, +z naso, che e' il frame T7 delle patch). Nessun allineamento oltre il
    frame canonico, applicato uguale a modello e Chamfer.
  Distanze calcolate solo verso le 30 mesh di galleria (15 scansioni, 15 registrazioni), che bastano
  a tutti i blocchi.

RICONOSCIMENTO (etichette d'identita'): query -> galleria di 15 (una mesh per persona), blocchi
``<tipo query> <ruolo> -> <tipo galleria>``: scan peak/nearneutral -> scan (il test vero: scansione
grezza contro scansione grezza, espressione contro neutro); reg peak -> reg (controllo); scan -> reg e
reg peak -> scan (incrociati). rank-1 (pari a meta' strada, come zs_expr_summarize) e AUC di verifica
(-distanza, coppie query-galleria stessa persona contro persone diverse); CI 95% bootstrap sui soggetti
(query ricampionate, galleria fissa; AUC pesata come ``zs_expr_summarize.recognition_values``), delta
appaiati modello - Chamfer sulle stesse repliche.

GRADUATA (GT unificata dalle neutre registrate, ``test_view/gt_matrix.npz``): Spearman fra distanza e
GT sulle coppie di persone diverse, blocchi: galleria scan-scan (105 coppie), reg-reg, scan-reg (210
ordinate), scan peak -> galleria scan e scan nearneutral -> galleria scan. CI bootstrap sui soggetti
(righe pesate per il prodotto dei conteggi, come ``dev_fs_summarize.spearman_replicates``), delta
appaiati. Con 15 persone i CI sono larghi: e' una prova della pipeline, non un risultato.

Uscite in ``--out-dir``: ``recognition.csv``, ``recognition_paired.csv``, ``graded.csv``,
``results.json``, ``results.md``. Solo numeri aggregati.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "zs3dmm"))
sys.path.insert(0, str(THIS_DIR.parent / "recon"))

import famos_common as fc  # noqa: E402
from zs_region_chamfer import load_points  # noqa: E402

CHAMFERS = ("chamfer_full", "chamfer_stable")
RECOG_BLOCKS = (("scan", "peak", "scan"), ("scan", "nearneutral", "scan"), ("reg", "peak", "reg"),
                ("scan", "peak", "reg"), ("scan", "nearneutral", "reg"), ("reg", "peak", "scan"))
GRADED_BLOCKS = (("scan", "gallery", "scan"), ("reg", "gallery", "reg"), ("scan", "gallery", "reg"),
                 ("scan", "peak", "scan"), ("scan", "nearneutral", "scan"))

_PTS: list = []
_TREES: list = []


def read_manifest() -> list[dict]:
    with open(fc.VIEW_DIR / "manifest.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def _init_cols(cols: list) -> None:
    global _TREES, _PTS
    _PTS = cols
    _TREES = [cKDTree(p) for p in cols]


def _chamfer_row(pts: np.ndarray) -> np.ndarray:
    tree = cKDTree(pts)
    out = np.zeros(len(_PTS))
    for j, (q, t) in enumerate(zip(_PTS, _TREES)):
        out[j] = 0.5 * (np.mean(t.query(pts)[0] ** 2) + np.mean(tree.query(q)[0] ** 2))
    return out


def chamfer_to_gallery(paths: list[Path], gal: np.ndarray, workers: int) -> dict:
    """{chamfer_full, chamfer_stable}: (n, n_gallery), e la frazione di vertici nella regione stabile."""
    with mp.get_context("fork").Pool(workers) as pool:
        loaded = pool.map(load_points, [(str(q), "hifi") for q in paths], chunksize=8)
    out = {"kept_vertex_fraction": np.asarray([x["kept_vertex_fraction"] for x in loaded])}
    for tag in ("full", "stable"):
        cols = [loaded[g][tag] for g in gal]
        with mp.get_context("fork").Pool(workers, initializer=_init_cols, initargs=(cols,)) as pool:
            out[f"chamfer_{tag}"] = np.stack(pool.map(_chamfer_row, [x[tag] for x in loaded], chunksize=8))
    return out


def embed(ops_dir: Path, names: list[str], checkpoint: Path, tag: str, device_name: str) -> np.ndarray:
    import torch

    from ws3b_latent import build_v1_model, encode

    device = torch.device(device_name if (device_name == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[famos-eval] device={device} ckpt={checkpoint}", flush=True)
    model = build_v1_model(checkpoint, "", device)
    cache = SimpleNamespace(out_root=fc.OUT_ROOT / "eval", metric=f"latent_{tag}", overwrite=False)
    z = encode(model, "test_view", ops_dir, names, device, cache)
    return np.stack([z[n] for n in names]).astype(np.float64)


# ------------------------------------------------------------------------------ metriche

def weighted_auc(score: np.ndarray, genuine: np.ndarray, w: np.ndarray) -> float:
    """Come ``zs_expr_summarize.weighted_auc`` (copiata: importarla tira dentro tutta la pipeline di eval)."""
    g, i = genuine & (w > 0), ~genuine & (w > 0)
    si, wi = score[i], w[i]
    order = np.argsort(si, kind="stable")
    si, cw = si[order], np.concatenate([[0.0], np.cumsum(wi[order])])
    lo = np.searchsorted(si, score[g], side="left")
    hi = np.searchsorted(si, score[g], side="right")
    below = cw[lo] + 0.5 * (cw[hi] - cw[lo])
    return float((w[g] * below).sum() / (w[g].sum() * wi.sum()))


def bootstrap_counts(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)])


def ci(x: np.ndarray) -> tuple[float, float]:
    return tuple(float(v) for v in np.percentile(x[1:], [2.5, 97.5]))


def recognition(Dg: np.ndarray, q_idx: np.ndarray, q_subj: np.ndarray, g_subj: np.ndarray,
                counts: np.ndarray) -> dict:
    """[punto, repliche] di rank-1 e AUC; ``Dg`` (n, n_gal) gia' ristretta alla galleria del blocco."""
    R = Dg[q_idx]
    R = np.where(np.isfinite(R), R, np.inf)
    true = (q_subj[:, None] == g_subj[None, :])
    gd = R[true]
    rank = 1 + (R < gd[:, None]).sum(1) + 0.5 * ((R == gd[:, None]).sum(1) - 1)
    hit = (rank == 1).astype(float)
    sa = np.repeat(q_subj, len(g_subj))
    sb = np.tile(g_subj, len(q_subj))
    sc, gen = -R.ravel(), true.ravel()
    out = {"rank1": [hit.mean()], "auc": [weighted_auc(sc, gen, np.ones(len(sc)))]}
    for c in counts:
        wq = c[q_subj].astype(float)
        out["rank1"].append((wq * hit).sum() / wq.sum())
        out["auc"].append(weighted_auc(sc, gen, np.where(gen, c[sa], c[sa] * c[sb]).astype(float)))
    return {k: np.asarray(v) for k, v in out.items()}


def spearman_reps(x: np.ndarray, gt: np.ndarray, sa: np.ndarray, sb: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = [spearmanr(x, gt)[0]]
    for c in counts:
        w = c[sa].astype(np.int64) * c[sb].astype(np.int64)
        k = w > 0
        out.append(spearmanr(np.repeat(x[k], w[k]), np.repeat(gt[k], w[k]))[0])
    return np.asarray(out)


def fmt(v: np.ndarray, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    lo, hi = ci(v)
    return f"{f.format(v[0])} [{f.format(lo)}, {f.format(hi)}]"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ops-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", default="e108")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()

    rows = read_manifest()
    names = [r["name"] for r in rows]
    kind = np.asarray([r["kind"] for r in rows])
    role = np.asarray([r["role"] for r in rows])
    with np.load(fc.VIEW_DIR / "gt_matrix.npz") as z:
        gt_names = [str(x) for x in z["names"]]
        G = z["D_orig"].astype(np.float64) * json.loads((fc.VIEW_DIR / "gt_matrix.json").read_text())["mm_per_unit"]
    s2i = {s: k for k, s in enumerate(gt_names)}
    subj = np.asarray([s2i[r["view_id"]] for r in rows])
    gal = {k: np.flatnonzero((kind == k) & (role == "gallery")) for k in ("scan", "reg")}
    for k, g in gal.items():
        g = g[np.argsort(subj[g])]
        assert np.array_equal(subj[g], np.arange(len(gt_names))), f"galleria {k}: una mesh per soggetto attesa"
        gal[k] = g
    gal_all = np.concatenate([gal["scan"], gal["reg"]])
    print(f"[famos-eval] {len(rows)} mesh, {len(gt_names)} soggetti, galleria {len(gal_all)}", flush=True)

    t0 = time.time()
    Z = embed(a.ops_dir, names, a.checkpoint, a.tag, a.device)
    D = {a.tag: np.sqrt(np.clip(((Z[:, None, :] - Z[None, gal_all, :]) ** 2).sum(-1), 0.0, None))}
    print(f"[famos-eval] latenti {Z.shape} in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    ch = chamfer_to_gallery([fc.VIEW_DIR / "npz" / f"{n}.npz" for n in names], gal_all, a.workers)
    D.update({c: ch[c] for c in CHAMFERS})
    print(f"[famos-eval] Chamfer verso la galleria in {time.time() - t0:.0f}s; frazione nella regione stabile "
          f"mediana {np.median(ch['kept_vertex_fraction']):.3f}", flush=True)
    col = {k: {int(i): j for j, i in enumerate(gal_all) if kind[i] == k} for k in ("scan", "reg")}

    def gcols(gk: str) -> np.ndarray:
        return np.asarray([col[gk][int(i)] for i in gal[gk]])

    methods = list(D)
    counts = bootstrap_counts(len(gt_names), a.n_bootstrap, a.seed)
    # ------------------------------------------------ riconoscimento
    rec, rec_rows, paired_rows = {}, [], []
    for qk, qr, gk in RECOG_BLOCKS:
        q_idx = np.flatnonzero((kind == qk) & (role == qr))
        blk = f"{qk} {qr} -> {gk}"
        for m in methods:
            rec[(blk, m)] = recognition(D[m][:, gcols(gk)], q_idx, subj[q_idx], np.arange(len(gt_names)), counts)
            v = rec[(blk, m)]
            rec_rows.append({"block": blk, "method": m, "n_queries": len(q_idx),
                             **{f"{k}{s}": x for k in ("rank1", "auc") for s, x in
                                zip(("", "_ci_low", "_ci_high"), (v[k][0], *ci(v[k])))}})
        for c in CHAMFERS:
            d = {k: rec[(blk, a.tag)][k] - rec[(blk, c)][k] for k in ("rank1", "auc")}
            paired_rows.append({"block": blk, "model": a.tag, "baseline": c,
                                **{f"{k}{s}": x for k in ("rank1", "auc") for s, x in
                                   zip(("", "_ci_low", "_ci_high"), (d[k][0], *ci(d[k])))},
                                **{f"{k}_p_le0": float((d[k][1:] <= 0).mean()) for k in ("rank1", "auc")}})
            rec[(blk, f"{a.tag}-{c}")] = d
    # ------------------------------------------------ graduata
    gr, gr_rows = {}, []
    for qk, qr, gk in GRADED_BLOCKS:
        q_idx = np.flatnonzero((kind == qk) & (role == qr))
        gc = gcols(gk)
        qa = np.repeat(q_idx, len(gt_names))
        gb = np.tile(np.arange(len(gt_names)), len(q_idx))
        sa, sb = subj[qa], gb
        keep = sa != sb
        if qr == "gallery" and qk == gk:
            keep &= sa < sb                                     # coppie non ordinate
        qa, gb, sa, sb = qa[keep], gb[keep], sa[keep], sb[keep]
        gt = G[sa, sb]
        blk = f"{qk} {qr} -> {gk}"
        for m in methods:
            gr[(blk, m)] = spearman_reps(D[m][qa, gc[gb]], gt, sa, sb, counts)
        for m in methods:
            v = gr[(blk, m)]
            r = {"block": blk, "method": m, "n_pairs": int(len(gt)), "spearman": v[0], "ci_low": ci(v)[0],
                 "ci_high": ci(v)[1]}
            if m != a.tag:
                d = gr[(blk, a.tag)] - v
                r.update({"delta_model_minus": d[0], "delta_ci_low": ci(d)[0], "delta_ci_high": ci(d)[1],
                          "delta_p_le0": float((d[1:] <= 0).mean())})
            gr_rows.append(r)

    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for fname, rr in (("recognition.csv", rec_rows), ("recognition_paired.csv", paired_rows), ("graded.csv", gr_rows)):
        keys = list(dict.fromkeys(k for r in rr for k in r))
        with open(out / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rr)
    checks = {"n_meshes": len(rows), "n_subjects": len(gt_names), "latent_dim": int(Z.shape[1]),
              "latent_finite": bool(np.isfinite(Z).all()), "chamfer_finite": bool(all(np.isfinite(D[c]).all() for c in CHAMFERS)),
              "kept_vertex_fraction_min_median": [float(ch["kept_vertex_fraction"].min()),
                                                  float(np.median(ch["kept_vertex_fraction"]))],
              "gt_mm_median_offdiag": float(np.median(G[np.triu_indices(len(G), 1)]))}
    (out / "results.json").write_text(json.dumps({"checkpoint": str(a.checkpoint), "tag": a.tag, "checks": checks,
                                                  "n_bootstrap": a.n_bootstrap, "seed": a.seed}, indent=1) + "\n")
    label = {a.tag: a.tag, "chamfer_full": "Chamfer intera", "chamfer_stable": "Chamfer regione stabile"}
    md = [f"# FaMoS, persone mai viste: {a.tag} e Chamfer", "",
          f"{len(gt_names)} persone di TEST, {len(rows)} mesh (patch NoW, 5215 triangoli). CI 95% bootstrap sui "
          f"soggetti ({a.n_bootstrap} repliche). Con 15 persone: prova della pipeline, non risultato.", "",
          "## Riconoscimento (galleria di 15, una mesh per persona)", "",
          "| blocco | query | metodo | rank-1 | AUC verifica |", "| --- | --- | --- | --- | --- |"]
    for qk, qr, gk in RECOG_BLOCKS:
        blk = f"{qk} {qr} -> {gk}"
        nq = int(((kind == qk) & (role == qr)).sum())
        for m in methods:
            v = rec[(blk, m)]
            md.append(f"| {blk} | {nq} | {label[m]} | {fmt(v['rank1'])} | {fmt(v['auc'])} |")
        for c in CHAMFERS:
            d = rec[(blk, f"{a.tag}-{c}")]
            md.append(f"| {blk} | | {a.tag} - {label[c]} | {fmt(d['rank1'], True)} | {fmt(d['auc'], True)} |")
    md += ["", "## Distanza graduata: Spearman con la GT unificata (mm, neutre registrate)", "",
           "| blocco | coppie | metodo | Spearman | delta modello - metodo |", "| --- | --- | --- | --- | --- |"]
    for r in gr_rows:
        dl = (f"{r['delta_model_minus']:+.3f} [{r['delta_ci_low']:+.3f}, {r['delta_ci_high']:+.3f}] "
              f"(P<=0 {r['delta_p_le0']:.3f})") if "delta_model_minus" in r else "-"
        md.append(f"| {r['block']} | {r['n_pairs']} | {label[r['method']]} | {r['spearman']:.3f} "
                  f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {dl} |")
    md += ["", "## Controlli", "", "```", json.dumps(checks, indent=1), "```"]
    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
