#!/usr/bin/env python3
"""Tabelle del riconoscimento in dominio e dei tempi (protocollo: aau/runs/indomain_recog/protocol.md).

    aau/run.sh aau/indomain/ir_summarize.py      (ir_summarize.sbatch)

Tutto il calcolo del riconoscimento viene da ``aau/zs3dmm/zs_expr_summarize.py``, importato e non
riscritto: ``retrieval_queries``, ``verification_pairs``, ``recognition_values`` (via
``_recog_task``), ``bootstrap_counts``, ``ci``, ``fmt``. Cambia solo l'indice: qui le etichette
sono quelle dell'insieme (``sets.json``: 6 topologie, o neutral + rexpr1..5) e i soggetti possono
essere un sottoinsieme delle mesh calcolate (``bfm19`` dentro ``bfm``).

Distanze, matrice (N x 6, N x 6) indicizzata da (soggetto, etichetta):
  - modelli: ||z_i - z_j|| da ``embeddings/<modello>__<insieme>.npz`` (zs_embed.py);
  - faceBench: ``facebench/<insieme>/<metrica>/<a>__<b>.npz`` di ir_facebench.py, matrici piene N x N
    per coppia non ordinata di etichette (riga = soggetto in a, colonna = soggetto in b);
  - ArcFace: ``arcface/<insieme>/<modo>/arcface_views.npz``, media delle 3 viste rinormalizzata,
    1 - coseno (come ``zs_arcface_summarize.arcface_distances``).
Controllo: le distanze del congiunto (e del BFM-only) contro ``latent_distance`` delle pair_metrics
delle eval WS2 sugli stessi soggetti.

Tempi: ``timing/{model,facebench,arcface}.json`` di ir_timing.py, mediana e IQR.

Scrive ``recognition.csv``, ``recognition_paired.csv``, ``timing.csv`` e ``summary.md`` = protocol.md +
risultati + ``lettura.md`` (se c'e').
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
sys.path.insert(0, str(AAU_DIR / "zs3dmm"))

import zs_expr_summarize as zes  # noqa: E402

base = zes.base
RUNS = AAU_DIR / "runs"
OUT = RUNS / "indomain_recog"
FB_METRICS = ("chamfer", "rigid_icp_chamfer", "nicp_p2tri")
ARC_MODES = ("normals", "shaded")
YAWS = (0.0, -30.0, 30.0)
LABEL = {"joint": "BFM+ICT congiunto", "bfm_only": "BFM-only (1019310)",
         "chamfer": zes.BL_LABEL.get("chamfer", "Chamfer"),
         "rigid_icp_chamfer": zes.BL_LABEL.get("rigid_icp_chamfer", "ICP rigido + Chamfer"),
         "nicp_p2tri": zes.BL_LABEL.get("nicp_p2tri", "NICP P2Tri"),
         "arcface_normals": "ArcFace, normal map, 3 viste",
         "arcface_shaded": "ArcFace, ombreggiato, 3 viste (secondaria)"}
SET_LABEL = {"bfm": "BFM, 108 held-out del congiunto", "ict": "ICT, 89 held-out del congiunto",
             "rexpr": "ICT con espressioni casuali, 89 held-out del congiunto",
             "bfm19": "BFM-19, held-out di congiunto E BFM-only"}
# Pair_metrics delle eval WS2 (ws2_eval.sbatch) per il controllo delle distanze dei modelli.
_EV = "eval_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d"
WS2_STAGES = {("joint", "bfm"): [RUNS / f"{_EV}_13da6115" / "ws2_cell"],
              ("joint", "ict"): [RUNS / f"{_EV}_60d50ab6" / "ws2_cell"],
              ("joint", "rexpr"): [RUNS / f"{_EV}_8afd021f" / "ws2_rexpr" / v for v in ("mixed", "neutral")],
              ("bfm_only", "bfm19"): [RUNS / f"{_EV}_16aab77d" / "ws2_cell"]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234, help="Seme del ricampionamento")
    p.add_argument("--sets", default="bfm,ict,rexpr,bfm19")
    return p.parse_args()


class LIndex(zes.Index):
    """``zes.Index`` con etichette date invece di TOPOLOGIES."""

    def __init__(self, subjects: list[str], labels):
        self.subjects, self.labels = list(subjects), tuple(labels)
        self.keys = [(s, t) for s in self.subjects for t in self.labels]
        self.pos = {k: i for i, k in enumerate(self.keys)}


# ------------------------------------------------------------------------------- distanze

def _place(keys, rows: np.ndarray, idx: LIndex, src: Path) -> np.ndarray:
    have = {k: r for k, r in zip(keys, rows)}
    missing = [k for k in idx.keys if k not in have]
    if missing:
        raise SystemExit(f"{src}: mancano {len(missing)} mesh, p.es. {missing[:3]}")
    return np.stack([have[k] for k in idx.keys])


def model_distances(path: Path, idx: LIndex) -> np.ndarray:
    z = np.load(path)
    keys = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))
    Z = _place(keys, z["Z"].astype(np.float64), idx, path)
    sq = (Z ** 2).sum(1)
    return np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0, None))


def arcface_distances(path: Path, idx: LIndex) -> np.ndarray:
    z = np.load(path)
    keys = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))
    norms = np.linalg.norm(z["E"].astype(np.float64), axis=2)
    if not np.isfinite(norms).all() or np.abs(norms - 1).max() > 1e-3:
        raise SystemExit(f"{path}: embedding non finiti o non normalizzati")
    cols = [list(z["yaws"].astype(float)).index(y) for y in YAWS]
    E = z["E"].astype(np.float64)[:, cols].mean(axis=1)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    Z = _place(keys, E, idx, path)
    return 1.0 - Z @ Z.T


def facebench_distances(root: Path, metric: str, idx: LIndex) -> np.ndarray:
    D = np.full((len(idx.keys), len(idx.keys)), np.nan)
    for i, a in enumerate(idx.labels):
        for b in idx.labels[i + 1:]:
            with np.load(root / metric / f"{a}__{b}.npz") as z:
                subj = [str(s) for s in z["subjects"]]
                sel = np.asarray([subj.index(s) for s in idx.subjects])
                M = z["values"][np.ix_(sel, sel)]
            ra, rb = idx.rows(a), idx.rows(b)
            D[np.ix_(ra, rb)] = M
            D[np.ix_(rb, ra)] = M.T
    return D


def ws2_check(stages: list[Path], D: np.ndarray, idx: LIndex) -> tuple[float, int, float]:
    """Max |distanza dagli embedding - latent_distance delle pair_metrics WS2|, sulle coppie dell'indice,
    e la mediana di latent_distance per la scala."""
    pm = pd.concat([base.read_pair_metrics(s) for s in stages], ignore_index=True)
    ka = list(zip(pm["subject_a"].astype(str), pm["topology_a"].astype(str)))
    kb = list(zip(pm["subject_b"].astype(str), pm["topology_b"].astype(str)))
    ok = np.asarray([a in idx.pos and b in idx.pos for a, b in zip(ka, kb)])
    if not ok.any():
        return float("nan"), 0, float("nan")
    ia = np.asarray([idx.pos[a] for a, k in zip(ka, ok) if k])
    ib = np.asarray([idx.pos[b] for b, k in zip(kb, ok) if k])
    ref = pm.loc[ok, "latent_distance"].to_numpy()
    return float(np.abs(D[ia, ib] - ref).max()), int(ok.sum()), float(np.median(ref))


# ---------------------------------------------------------------------------------- blocchi

def blocks_for(name: str, labels) -> dict:
    """blocco -> (coppie ordinate del retrieval (query, galleria), coppie della verifica)."""
    if name == "rexpr":
        ex = [t for t in labels if t != "neutral"]
        return {"expr": ([(a, b) for a in ex for b in ex if a != b],
                         [(a, b) for i, a in enumerate(ex) for b in ex[i + 1:]]),
                "neutral_gallery": ([(a, "neutral") for a in ex], [("neutral", a) for a in ex])}
    nocrop = [t for t in labels if t != "crop"]
    return {"nocrop": ([(a, b) for a in nocrop for b in nocrop if a != b],
                       [(a, b) for i, a in enumerate(nocrop) for b in nocrop[i + 1:]]),
            "crop": ([(a, b) for a in labels for b in labels if a != b and "crop" in (a, b)],
                     [("crop", b) for b in nocrop])}


BLOCK_LABEL = {"nocrop": "PRIMARIO, 5 topologie senza crop", "crop": "a parte: crop da un lato",
               "expr": "PRIMARIO, espressione contro espressione (k != k')",
               "neutral_gallery": "secondario: galleria neutra, query con espressione"}


# ------------------------------------------------------------------------------------ tempi

def timing_table() -> pd.DataFrame:
    rows = []
    specs = [("model", "operators_cpu", "congiunto: operatori DiffusionNet (k=128), CPU 1 thread", "mesh"),
             ("model", "embed_cuda", "congiunto: embedding (caricamento + forward), GPU", "mesh"),
             ("model", "embed_cpu", "congiunto: embedding (caricamento + forward), CPU 1 thread", "mesh"),
             ("model", "compare_cpu", "congiunto: confronto ||z_a - z_b||, CPU", "coppia di embedding"),
             *[("model", f"retrieval_{d}_N{n}", f"congiunto: retrieval 1:{n:,} (distanze + argsort), "
                + ("CPU 1 thread" if d == "cpu" else "GPU"), "query")
               for n in (100, 1000, 10000) for d in ("cpu", "gpu")],
             ("facebench", "icp_chamfer", "ICP rigido + Chamfer, CPU 1 thread", "coppia di mesh"),
             ("facebench", "nicp_p2tri", "NICP P2Tri (con l'ICP rigido che la precede), CPU 1 thread", "coppia di mesh"),
             ("arcface", "render", "ArcFace normal map: lettura + 3 render, CPU 1 thread", "mesh"),
             ("arcface", "embed", "ArcFace normal map: 3 embedding + media, CPU 1 thread", "mesh"),
             ("arcface", "total", "ArcFace normal map: totale", "mesh")]
    data = {}
    for part in ("model", "facebench", "arcface"):
        p = OUT / "timing" / f"{part}.json"
        if p.exists():
            data[part] = json.loads(p.read_text())
    for part, key, lab, unit in specs:
        x = np.asarray(data.get(part, {}).get(key, []), dtype=np.float64)
        if not len(x):
            continue
        q25, q50, q75 = np.percentile(x, [25, 50, 75])
        rows.append({"part": part, "key": key, "label": lab, "unit": unit, "n": len(x),
                     "median_s": q50, "q25_s": q25, "q75_s": q75})
    if "model" in data and "operators_cpu" in data["model"]:
        ops, emb = np.asarray(data["model"]["operators_cpu"]), np.asarray(data["model"].get("embed_cuda", []))
        if len(emb) == len(ops):
            t = ops + emb
            q25, q50, q75 = np.percentile(t, [25, 50, 75])
            rows.append({"part": "model", "key": "ops_plus_embed_gpu", "unit": "mesh", "n": len(t),
                         "label": "congiunto: operatori + embedding GPU (totale per mesh)",
                         "median_s": q50, "q25_s": q25, "q75_s": q75})
    df = pd.DataFrame(rows)
    df.attrs["hw"] = {k: {h: v.get(h) for h in ("host", "cpu", "gpu", "job", "omp_threads")} for k, v in data.items()}
    df.attrs["n_vertices"] = data.get("model", {}).get("n_vertices", [])
    return df


def fmt_time(s: float) -> str:
    if s >= 1:
        return f"{s:.2f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.2f} ms"
    return f"{s * 1e6:.1f} us"


# ------------------------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    sets = json.loads((OUT / "sets.json").read_text())["sets"]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    rec_rows, delta_rows, checks, info = [], [], [], {}

    for name in args.sets.split(","):
        spec = sets[name]
        src = spec.get("subset_of", name)
        idx = LIndex(spec["subjects"], spec["labels"])
        D = {}
        models = ["joint"] + (["bfm_only"] if name == "bfm19" else [])
        for m in models:
            D[m] = model_distances(OUT / "embeddings" / f"{m}__{src}.npz", idx)
            if (m, name) in WS2_STAGES or (m, src) in WS2_STAGES:
                diff, n, med = ws2_check(WS2_STAGES.get((m, name)) or WS2_STAGES[(m, src)], D[m], idx)
                checks.append(f"- {SET_LABEL[name]}, {LABEL[m]}: max |diff| contro `latent_distance` WS2 = "
                              f"{diff:.2e} su {n} coppie (mediana di latent_distance {med:.3f})")
        for m in FB_METRICS:
            try:
                D[m] = facebench_distances(OUT / "facebench" / src, m, idx)
            except FileNotFoundError as exc:
                print(f"[ir-sum] ATTENZIONE: {name} faceBench {m} incompleto ({exc}), riga saltata", flush=True)
        for mode in ARC_MODES:
            p = OUT / "arcface" / src / mode / "arcface_views.npz"
            if p.exists():
                D[f"arcface_{mode}"] = arcface_distances(p, idx)
            else:
                print(f"[ir-sum] ATTENZIONE: {p} assente, riga saltata", flush=True)
        blocks = blocks_for(name, spec["labels"])
        counts = zes.bootstrap_counts(len(idx.subjects), args.n_bootstrap,
                                      base.stable_seed(args.seed, f"indomain_recognition:{name}"))
        print(f"[ir-sum] {name}: {len(idx.subjects)} soggetti, metodi {list(D)}", flush=True)
        rec = {}
        with mp.get_context("fork").Pool(min(workers, 16)) as pool:
            tasks = [((blk, m), D[m], idx, pr, pv, counts) for blk, (pr, pv) in blocks.items() for m in D]
            for key, vals, n_nan in pool.imap_unordered(zes._recog_task, tasks):
                rec[key] = (vals, n_nan)
        for (blk, m), (vals, n_nan) in rec.items():
            r = {"set": name, "block": blk, "method": m, "n_subjects": len(idx.subjects), "n_nan_distances": n_nan}
            for k in ("rank1", "map", "auc"):
                r[k], (r[f"{k}_ci_low"], r[f"{k}_ci_high"]) = float(vals[k][0]), zes.ci(vals[k])
            rec_rows.append(r)
        for blk in blocks:
            for m in D:
                if m == "joint":
                    continue
                a, b = rec[(blk, "joint")][0], rec[(blk, m)][0]
                r = {"set": name, "block": blk, "model": "joint", "other": m}
                for k in ("rank1", "map", "auc"):
                    d = a[k] - b[k]
                    r[k], (r[f"{k}_ci_low"], r[f"{k}_ci_high"]) = float(d[0]), zes.ci(d)
                    r[f"{k}_p_le0"] = float((d[1:] <= 0).mean())
                delta_rows.append(r)
        info[name] = {"blocks": {b: (len(pr), len(pv)) for b, (pr, pv) in blocks.items()}, "n": len(idx.subjects)}

    rec_table, delta_table = pd.DataFrame(rec_rows), pd.DataFrame(delta_rows)
    timing = timing_table()
    rec_table.to_csv(OUT / "recognition.csv", index=False)
    delta_table.to_csv(OUT / "recognition_paired.csv", index=False)
    timing.to_csv(OUT / "timing.csv", index=False)

    def verdict(lo: float, hi: float) -> str:
        return "sopra" if lo > 0 else ("sotto" if hi < 0 else "pari")

    lines = [(OUT / "protocol.md").read_text().rstrip(), "\n---\n", "# Risultati\n",
             f"CI 95% bootstrap per soggetto, {args.n_bootstrap} repliche, le stesse per tutti i metodi di un "
             "insieme. mAP = MRR (un solo rilevante). Distanze NaN (coppie faceBench fallite) = +inf.\n"]
    for name, inf in info.items():
        n = inf["n"]
        lines.append(f"## {SET_LABEL[name]}\n")
        for blk, (n_r, n_v) in inf["blocks"].items():
            lines += [f"### {BLOCK_LABEL[blk]}\n",
                      f"Retrieval: {n_r * n} query ({n_r} coppie ordinate x {n}), galleria di {n}. Verifica: "
                      f"{n_v * n} coppie stessa persona, {n_v * n * (n - 1)} persone diverse.\n",
                      "| metodo | rank-1 | mAP | AUC verifica | distanze NaN |", "| --- | --- | --- | --- | --- |"]
            sub = rec_table[(rec_table["set"] == name) & (rec_table["block"] == blk)].set_index("method")
            for m in [x for x in LABEL if x in sub.index]:
                r = sub.loc[m]
                lines.append(f"| {LABEL[m]} | " + " | ".join(zes.fmt(r[k], r[f"{k}_ci_low"], r[f"{k}_ci_high"])
                                                            for k in ("rank1", "map", "auc"))
                             + f" | {int(r['n_nan_distances'])} |")
            lines += ["", "Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):\n",
                      "| metodo | rank-1: delta [CI] (P<=0) | mAP: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | lettura rank-1 / AUC |",
                      "| --- | --- | --- | --- | --- |"]
            dsub = delta_table[(delta_table["set"] == name) & (delta_table["block"] == blk)].set_index("other")
            for m in [x for x in LABEL if x in dsub.index]:
                r = dsub.loc[m]
                lines.append(f"| {LABEL[m]} | " + " | ".join(
                    f"{zes.fmt(r[k], r[f'{k}_ci_low'], r[f'{k}_ci_high'], True)} ({r[f'{k}_p_le0']:.3f})"
                    for k in ("rank1", "map", "auc"))
                    + f" | {verdict(r['rank1_ci_low'], r['rank1_ci_high'])} / {verdict(r['auc_ci_low'], r['auc_ci_high'])} |")
            lines.append("")

    lines.append("## Tempi\n")
    hw = timing.attrs.get("hw", {})
    for part, h in hw.items():
        lines.append(f"- {part}: host `{h.get('host')}`, CPU {h.get('cpu')}"
                     + (f", GPU {h.get('gpu')}" if h.get("gpu") else "") + f", job {h.get('job')}, "
                     f"OMP_NUM_THREADS={h.get('omp_threads')}")
    nv = timing.attrs.get("n_vertices", [])
    if nv:
        lines.append(f"- campione: {len(nv)} mesh, vertici da {min(nv)} a {max(nv)} (mediana {int(np.median(nv))})")
    lines += ["", "| voce | per | n | mediana | IQR (25-75%) |", "| --- | --- | --- | --- | --- |"]
    for r in timing.itertuples():
        lines.append(f"| {r.label} | {r.unit} | {r.n} | {fmt_time(r.median_s)} | {fmt_time(r.q25_s)} - {fmt_time(r.q75_s)} |")
    lines += ["", "## Controlli\n", *checks,
              "- faceBench: NICP asimmetrico, orientazione della coppia = ordine delle etichette, non query -> galleria.",
              f"- leak (`sets.json`): {json.dumps(json.loads((OUT / 'sets.json').read_text())['leak'])}", ""]
    if (OUT / "lettura.md").exists():
        lines += ["---\n", (OUT / "lettura.md").read_text().rstrip(), ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ir-sum] scritto {OUT / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
