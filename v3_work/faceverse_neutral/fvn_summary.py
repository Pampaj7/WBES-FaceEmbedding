#!/usr/bin/env python3
"""FaceVerse neutra contro FaceVerse con espressioni: graduata con FR, SR, maxabs di bracci, C3M, e108 e baseline in mm.

    AAU_NV= aau/run.sh v3_work/faceverse_neutral/fvn_summary.py --workers 16 [--jobs "id ..."]
    (fvn_summary.sbatch; protocollo aau/runs/evidence/faceverse_neutral/PROTOCOL.md)

Per ENTRAMBE le viste (``neutra`` = FACEVERSE_ZS/eval_view, ``espressioni`` = expr_view) e con lo stesso codice:
righe = coppie di mesh dei 100 soggetti valutati (``zs_stage.select_subjects``, seme 1234) con soggetti diversi,
topologie diverse e niente crop (99.000, il ``nocrop_cross`` dei bracci e il ``mesh_pair_nocrop`` delle baseline); GT
FR e SR (``datasets/CANONICAL_GT/eval/faceverse_{fr,sr}.npz``) e maxabs (``gt_matrix.npz`` della vista, lo stesso
file); Spearman e IC 95% bootstrap per soggetto con ``eval_factorized.boot_rows`` (1000 repliche, seme 1234) per ogni
metodo. Distanze:
  * bracci e C3M: ``factorized_v3.model_distances`` dagli embedding [s, u] (o z) del passo ``form`` (espressioni,
    ``c3f_eval/form_fv_expr``) e ``fvn`` (neutra, ``faceverse_neutral/embed``); FR e maxabs con d_F (||z|| per ctrlfr),
    SR con d_P, come ``tools/fact_summary.py``;
  * e108: distanza euclidea degli embedding (``zs_expr_summarize.model_distances``);
  * baseline (``aau/baselines_mm``, viste ``faceverse`` e ``faceverse_neutral``): ICP + Chamfer in mm, NICP per coppia
    in mm e in cs (``zs_expr_summarize.facebench_distances``), NICP su template in mm (``ir_template.template_distances``),
    taglia stimata |log CS robusta| (``scalars.npz``) e taglia oracolo |log S| (``faceverse_centroid_size.npz``).
Righe con distanza non finita escluse per quel metodo (contate). Controlli contro i numeri pubblicati della vista con
espressioni e contro ``eval_factorized.py`` della neutra. Uscite in ``aau/runs/evidence/faceverse_neutral/``:
``graded.csv``, ``controls.csv``, ``results.md``.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
for _p in (REPO / "v3_work/trainer", REPO / "v3_work/trainer/tools", REPO / "aau/zs3dmm", REPO / "aau/baselines_mm",
           REPO / "aau/indomain"):
    sys.path.insert(0, str(_p))

import blmm  # noqa: E402
import ir_template as irt  # noqa: E402
import zs_expr_summarize as zes  # noqa: E402  (con ``common`` di aau/baselines, legato all'import)
from zs_stage import select_subjects  # noqa: E402

sys.modules.pop("common", None)   # il trainer ha un suo ``common`` (factorized_v3: domain_of), stesso nome
import eval_factorized as ef  # noqa: E402  (mette v3_work/trainer in testa a sys.path)
import factorized_v3 as fz  # noqa: E402

EV = REPO / "aau/runs/evidence"
OUT = EV / "faceverse_neutral"
G = REPO / "datasets/CANONICAL_GT/eval"
NB, SEED = 1000, 1234
GTS = ("fr", "sr", "maxabs")
VIEWS = {
    "neutra": {"dir": REPO / "datasets/FACEVERSE_ZS/eval_view", "emb": OUT / "embed", "blmm": "faceverse_neutral",
               "e108": OUT / "embed", "e108_stage": "scale_e108_flip_embed"},
    "espressioni": {"dir": REPO / "datasets/FACEVERSE_ZS/expr_view", "emb": EV / "trainer_v3/ablations/c3f_eval/form_fv_expr",
                    "blmm": "faceverse", "e108": REPO / "aau/runs/ws_faceverse_expr", "e108_stage": "scale_e108_flip_topology"},
}
# (braccio, seme, epoca, etichetta); C3M: run dir propria, epoca 205
ARMS = tuple((a, s, "072", f"{a} s{s}") for a in ("factorized", "factorized2", "ctrlfr") for s in (1234, 2345)) \
    + (("factorizedc3m", 1234, "205", "C3M e205"),)
BASELINES = (("mm_rigid_icp_chamfer", "ICP + Chamfer in mm"), ("mm_nicp_p2tri", "NICP per coppia in mm"),
             ("mm_nicp_template", "NICP su template in mm"), ("cs_nicp_p2tri", "NICP per coppia, modo cs"),
             ("est_cs", "taglia stimata (non oracolo)"), ("oracle_size", "taglia oracolo"))
MODELS = tuple(f"{a}_s{s}_e{e}" for a, s, e, _ in ARMS)
LABEL = {**{m: a[3] for m, a in zip(MODELS, ARMS)}, "scale_e108": "e108 (cieco alla taglia)", **dict(BASELINES)}


def vname(arm: str, seed: int) -> str:
    return arm + ("" if seed == 1234 else f"s{seed}")


def order_of(subjects, topologies, idx: zes.Index) -> np.ndarray:
    """Indici delle righe di un file per mesh nell'ordine di ``idx.keys``."""
    keys = list(zip([str(s) for s in subjects], [str(t) for t in topologies]))
    if sorted(keys) != sorted(idx.keys):
        raise SystemExit("mesh diverse da quelle attese")
    pos = {k: i for i, k in enumerate(keys)}
    return np.asarray([pos[k] for k in idx.keys])


def arm_distances(view: str, arm: str, seed: int, e: str, idx: zes.Index, i, j) -> dict:
    """{gt: distanza per riga} di un braccio: d_F per FR e maxabs, d_P per SR (fattorizzati), ||z|| (ctrlfr)."""
    import torch
    hits = sorted(VIEWS[view]["emb"].glob(f"data_*/scale_v3{vname(arm, seed)}fulle{e}_flip_embed/zs_zeroshot/embeddings.npz"))
    if not hits:
        raise FileNotFoundError(f"{view}: embedding di {arm} s{seed} e{e} assenti")
    with np.load(hits[0], allow_pickle=True) as z:
        Z = np.asarray(z["Z"], np.float64)[order_of(z["subjects"], z["topologies"], idx)]
        ckpt = Path(str(z["checkpoint"]))
    head = torch.load(ckpt, map_location="cpu", weights_only=False)["args"].get("head", "embed")
    dpu = ef.dp_from_ckpt(ckpt) if head in fz.FACTORIZED else float("nan")
    D = fz.model_distances(Z, i, j, head, dpu)
    if head in fz.FACTORIZED:
        return {"fr": D["form"], "sr": D["shape"], "maxabs": D["form"]}
    return {g: D["z"] for g in GTS}


def log_abs(x) -> np.ndarray:
    lx = np.log(np.asarray(x, dtype=np.float64))
    return np.abs(lx[:, None] - lx[None, :])


def baseline_matrices(view: str, idx: zes.Index) -> dict:
    """{metodo: D (600, 600)} delle baseline nella vista, nell'ordine di ``idx.keys``."""
    bv = VIEWS[view]["blmm"]
    root = blmm.view_root(bv)
    D = {}
    for m in ("mm_rigid_icp_chamfer", "mm_nicp_p2tri", "cs_nicp_p2tri"):
        mode, _, metric = m.partition("_")
        D[m] = zes.facebench_distances(root / mode, metric, idx)
    with np.load(root / "mm" / "template.npz") as z:
        R = z["R"][order_of(z["subjects"], z["topologies"], idx)].astype(np.float64)
    D["mm_nicp_template"] = irt.template_distances(R, R)
    sc = blmm.scalars_of(bv)
    D["est_cs"] = log_abs([sc[blmm.rel(blmm.mesh_path(bv, s, t))]["cs"] for s, t in idx.keys])
    with np.load(G / "faceverse_centroid_size.npz") as z:
        S = dict(zip([str(s) for s in z["names"]], z["S"]))
    D["oracle_size"] = log_abs([S[s] for s, _ in idx.keys])
    return D


def _boot(task):
    view, method, cols, gts, si, sj, n = task
    out = []
    for g in GTS:
        d = cols[g]
        ok = np.isfinite(d)
        r = ef.boot_rows({"d": d[ok]}, {g: gts[g]}, si[ok], sj[ok], n, NB, SEED)[0]
        out.append({"view": view, "method": method, "label": LABEL[method], "gt": g, "point": r["point"],
                    "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n_subjects": n, "n_rows": int(ok.sum()),
                    "n_nan": int((~ok).sum())})
    return out


def tasks_of(view: str, missing: list) -> list:
    subjects = select_subjects(VIEWS[view]["dir"] / "npz", 1234)
    idx = zes.Index(subjects)
    nc = np.asarray([k for k, (_, t) in enumerate(idx.keys) if t != "crop"])
    a, b = np.triu_indices(len(nc), 1)
    i, j = nc[a], nc[b]
    subj = np.asarray([subjects.index(s) for s, _ in idx.keys])
    topo = np.asarray([t for _, t in idx.keys])
    keep = (subj[i] != subj[j]) & (topo[i] != topo[j])
    i, j = i[keep], j[keep]
    si, sj, n = subj[i], subj[j], len(subjects)
    gts = {"fr": ef.load_gt(G / "faceverse_fr.npz", subjects), "sr": ef.load_gt(G / "faceverse_sr.npz", subjects),
           "maxabs": ef.load_gt(VIEWS[view]["dir"] / "gt_matrix.npz", subjects)}
    cols = {}
    for m, (arm, seed, e, _) in zip(MODELS, ARMS):
        try:
            cols[m] = arm_distances(view, arm, seed, e, idx, i, j)
        except FileNotFoundError as exc:
            missing.append(str(exc))
    stage = sorted(VIEWS[view]["e108"].glob(f"data_*/{VIEWS[view]['e108_stage']}/zs_zeroshot"))
    if stage and (stage[0] / "embeddings.npz").exists():
        d = zes.model_distances(stage[0], idx)[i, j]
        cols["scale_e108"] = {g: d for g in GTS}
    else:
        missing.append(f"{view}: embedding di e108 assenti")
    try:
        for m, M in baseline_matrices(view, idx).items():
            cols[m] = {g: M[i, j] for g in GTS}
    except (FileNotFoundError, OSError) as exc:
        missing.append(f"{view}: baseline incomplete ({type(exc).__name__}: {exc})")
    print(f"[fvn] {view}: {n} soggetti, {len(i)} righe, {len(cols)} metodi", flush=True)
    return [(view, m, c, gts, si, sj, n) for m, c in cols.items()]


def controls(rows: list) -> list:
    """Vista con espressioni ricalcolata contro i numeri pubblicati; neutra contro eval_factorized.py (passo fvn)."""
    get = {(r["view"], r["method"], r["gt"]): r for r in rows}
    out = []

    def add(what, src, ref, key, cols=("point", "ci_low", "ci_high")):
        if key in get:
            for c in cols:
                out.append({"what": f"{what} ({c})", "source": src, "published": ref[c], "recomputed": get[key][c],
                            "abs_diff": abs(ref[c] - get[key][c])})

    fr = EV / "trainer_v3/factorized_results.csv"
    pub = list(csv.DictReader(open(fr))) if fr.exists() else []
    for m, (arm, seed, e, lab) in zip(MODELS, ARMS):
        for g in GTS:
            dist = ("shape" if g == "sr" else "form") if arm != "ctrlfr" else "z"
            hit = [r for r in pub if r["arm"] == arm and r["seed"] == str(seed) and r["steps"] == "21096"
                   and r["dom"] == "fv" and r["kind"] == "graded" and r["distance"] == dist and r["gt"] == g]
            if arm == "factorizedc3m":
                p = EV / f"trainer_v3/ablations/c3f_eval/form/fv/v3{vname(arm, seed)}e{e}/form_spearman.csv"
                hit = [r for r in csv.DictReader(open(p)) if r["level"] == "mesh_pair" and r["distance"] == dist
                       and r["gt"] == g] if p.exists() else []
            if hit:
                add(f"espressioni {lab} {dist} GT {g}", "factorized_results.csv / form_spearman.csv",
                    {c: float(hit[0][c]) for c in ("point", "ci_low", "ci_high")}, ("espressioni", m, g))
            p = OUT / f"form/v3{vname(arm, seed)}e{e}/form_spearman.csv"
            if p.exists():
                hit = [r for r in csv.DictReader(open(p)) if r["level"] == "mesh_pair" and r["distance"] == dist
                       and r["gt"] == g]
                add(f"neutra {lab} {dist} GT {g}", "faceverse_neutral/form (eval_factorized.py)",
                    {c: float(hit[0][c]) for c in ("point", "ci_low", "ci_high")}, ("neutra", m, g))
    bl = list(csv.DictReader(open(EV / "baselines_mm/spearman.csv")))
    for m in ["scale_e108"] + [b for b, _ in BASELINES]:
        for g in GTS:
            hit = [r for r in bl if r["domain"] == "faceverse" and r["group"] == "mesh_pair_nocrop" and r["method"] == m
                   and r["gt"] == g]
            if hit:
                ref = {c: float(hit[0][c]) for c in ("point", "ci_low", "ci_high")}
                add(f"espressioni {LABEL[m]} GT {g}", "baselines_mm/spearman.csv (punto)", ref, ("espressioni", m, g),
                    ("point",))
                add(f"espressioni {LABEL[m]} GT {g}", "baselines_mm/spearman.csv (IC, seme di E12 contro 1234)", ref,
                    ("espressioni", m, g), ("ci_low", "ci_high"))
    return out


def verdict(rows: list, gt: str) -> tuple[str, str]:
    """Regola di PROTOCOL.md, sezione 2, sui punti della vista neutra."""
    neu = {r["method"]: r["point"] for r in rows if r["view"] == "neutra" and r["gt"] == gt}
    models = list(MODELS)
    expected = models + ["scale_e108"] + [b for b, _ in BASELINES]
    absent = [LABEL[m] for m in expected if m not in neu]
    if absent:
        return "INCOMPLETO", "metodi assenti: " + ", ".join(absent)
    top = max(expected, key=lambda m: neu[m])
    low = min(models, key=lambda m: neu[m])
    why = f"massimo neutro {LABEL[top]} {neu[top]:.3f}; minimo dei modelli {LABEL[low]} {neu[low]:.3f}"
    if all(neu[m] <= 0.35 for m in expected):
        return "DOMINIO", why
    if all(neu[m] > 0.5 for m in models):
        return "ESPRESSIONI", why
    return "INTERMEDIO", why


def fmt(r) -> str:
    return f"{r['point']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if r else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--jobs", default="", help="job id da citare in results.md")
    a = ap.parse_args()
    missing = []
    tasks = tasks_of("espressioni", missing) + tasks_of("neutra", missing)
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = [r for block in pool.map(_boot, tasks, chunksize=1) for r in block]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "graded.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    C = controls(rows)
    if C:
        with open(OUT / "controls.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(C[0]))
            w.writeheader()
            w.writerows(C)
    get = {(r["view"], r["method"], r["gt"]): r for r in rows}
    n_subj = sorted({r["n_subjects"] for r in rows})
    n_rows = sorted({r["n_rows"] for r in rows})
    md = ["# FaceVerse neutra contro FaceVerse con espressioni, graduata", "",
          "Generato da `v3_work/faceverse_neutral/fvn_summary.py`; protocollo `PROTOCOL.md` (sha256 in `PROTOCOL.sha256`). "
          f"Soggetti {n_subj} (gli stessi nelle due viste), righe per metodo {n_rows} (coppie di mesh senza crop, "
          "topologie diverse). IC 95% bootstrap per soggetto (1000 repliche, seme 1234, `eval_factorized.boot_rows`), lo "
          "stesso codice per le due viste. GT identiche nelle due viste (definite sull'identita' neutra): cambia solo la "
          "geometria osservata. FR e maxabs con d_F (||z|| per ctrlfr), SR con d_P.", ""]
    if a.jobs:
        md += [f"Job: {a.jobs}.", ""]
    if missing:
        md += ["**Mancanti** (righe assenti): " + "; ".join(missing), ""]
    md += ["## Regola preregistrata (PROTOCOL.md, sezione 2)", ""]
    for g in ("fr", "sr"):
        v, why = verdict(rows, g)
        md.append(f"- GT {g.upper()}{' (primaria)' if g == 'fr' else ' (secondaria)'}: **{v}** ({why})")
    md += ["", "## Tabella", "",
           "| metodo | FR neutra | FR espressioni | delta FR | SR neutra | SR espressioni | delta SR | maxabs neutra | maxabs espressioni |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in list(MODELS) + ["scale_e108"] + [b for b, _ in BASELINES]:
        cells = []
        for g in GTS:
            rn, rx = get.get(("neutra", m, g)), get.get(("espressioni", m, g))
            cells += [fmt(rn), fmt(rx)]
            if g != "maxabs":
                cells.append(f"{rn['point'] - rx['point']:+.3f}" if rn and rx else "-")
        md.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    nan = [r for r in rows if r["n_nan"]]
    if nan:
        md += ["", "Righe escluse per distanza non finita: " + "; ".join(
            f"{r['view']} {r['label']} GT {r['gt']}: {r['n_nan']}" for r in nan if r["gt"] == "fr")]
    if C:
        md += ["", "## Controlli", "", "| controllo | sorgente | confronti | max abs diff |", "| --- | --- | --- | --- |"]
        for src in dict.fromkeys(c["source"] for c in C):
            x = [c["abs_diff"] for c in C if c["source"] == src]
            md.append(f"| {'neutra' if 'faceverse_neutral' in src else 'espressioni'} | {src} | {len(x)} | {max(x):.2e} |")
    (OUT / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
