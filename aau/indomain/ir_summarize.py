#!/usr/bin/env python3
"""Tabelle del riconoscimento in dominio e dei tempi (protocollo: aau/runs/indomain_recog/protocol.md).

    aau/run.sh aau/indomain/ir_summarize.py      (ir_summarize.sbatch)

Tutto il calcolo del riconoscimento viene da ``aau/zs3dmm/zs_expr_summarize.py``, importato e non
riscritto: ``retrieval_queries``, ``verification_pairs``, ``recognition_values`` (via
``_recog_task``), ``weighted_auc``, ``bootstrap_counts``, ``ci``, ``fmt``. Cambia solo l'indice: qui le
etichette sono quelle dell'insieme (``sets.json``: 6 topologie, o neutral + rexpr1..5) e i soggetti
possono essere un sottoinsieme delle mesh calcolate (``bfm19`` dentro ``bfm``).

Distanze, matrice (N x 6, N x 6) indicizzata da (soggetto, etichetta):
  - modelli: ||z_i - z_j|| da ``embeddings/<modello>__<insieme>.npz`` (zs_embed.py);
  - faceBench: ``facebench/<insieme>/<metrica>/<a>__<b>.npz`` di ir_facebench.py, matrici piene N x N
    per coppia non ordinata di etichette (riga = soggetto in a, colonna = soggetto in b); ICP rigido
    (prima tornata) e ICP di similarita' (revisione 1, ``--variant sim``);
  - NICP su template: distanza L2 media per vertice fra iscrizioni (``template/<insieme>.npz``,
    ir_template.py; ``ict`` viene da ``ict992``, stesse mesh e stessi semi);
  - ArcFace: ``arcface/`` (prima tornata) e ``arcface_best/`` (revisione 1), media delle 3 viste
    rinormalizzata, 1 - coseno (come ``zs_arcface_summarize.arcface_distances``).
Revisione 1, in piu':
  - TAR@FAR 1e-3 e 1e-4 su ogni blocco (``tar_values``: soglia = quantile pesato degli impostori,
    TAR = frazione pesata dei genuini sopra soglia; stessi pesi bootstrap dell'AUC);
  - galleria grande ``ict992`` (``gallery_block``): 100 query x 992 in galleria, matrici rettangolari,
    bootstrap sulle query con galleria fissa; e, solo per congiunto e template, tutte le 992 query sui
    blocchi senza crop / crop (secondario, 200 repliche: 9.8 milioni di coppie di verifica per blocco).
Controlli: le distanze del congiunto (e del BFM-only) contro ``latent_distance`` delle pair_metrics
delle eval WS2; ``chamfer_sim`` (rifatta dalla variante sim) contro ``chamfer`` della prima tornata.

Tempi: ``timing/{model,facebench,template,arcface}.json`` di ir_timing.py, mediana e IQR, divisi fra
iscrizione (per mesh), ricerca su galleria iscritta (per query) e metodi a coppie (per coppia, 1:N
stimato come N x mediana).

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
sys.path.insert(0, str(THIS_DIR))

import zs_expr_summarize as zes  # noqa: E402
from ir_template import template_distances  # noqa: E402

base = zes.base
RUNS = AAU_DIR / "runs"
OUT = RUNS / "indomain_recog"
FB_METRICS = ("chamfer", "rigid_icp_chamfer", "nicp_p2tri", "sim_icp_chamfer", "sim_nicp_p2tri")
# metodo -> (cartella, modo) degli embedding ArcFace
ARC = {"arcface_best_normals": ("arcface_best", "normals"), "arcface_best_shaded": ("arcface_best", "shaded"),
       "arcface_normals": ("arcface", "normals"), "arcface_shaded": ("arcface", "shaded")}
TEMPLATE_SRC = {"bfm": "bfm", "bfm19": "bfm", "ict": "ict992", "rexpr": "rexpr", "ict992": "ict992"}
YAWS = (0.0, -30.0, 30.0)
FARS = (1e-3, 1e-4)
METRICS = ("rank1", "map", "auc") + tuple(f"tar{f:g}" for f in FARS)
# Ordine delle righe nelle tabelle.
LABEL = {"joint": "BFM+ICT congiunto", "bfm_only": "BFM-only (1019310)",
         "sim_nicp_p2tri": "ICP di similarita' + NICP P2Tri",
         "sim_icp_chamfer": "ICP di similarita' + Chamfer",
         "template": "NICP su template (iscrizione)",
         "nicp_p2tri": "ICP rigido + NICP P2Tri (prima tornata)",
         "rigid_icp_chamfer": "ICP rigido + Chamfer (prima tornata)",
         "chamfer": "Chamfer faceBench",
         "chamfer_sim": "Chamfer faceBench",
         "arcface_best_normals": "ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste",
         "arcface_best_shaded": "ArcFace, ombreggiato, inquadratura per mesh (secondaria)",
         "arcface_normals": "ArcFace, normal map, camera di dominio (prima tornata)",
         "arcface_shaded": "ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria)"}
SET_LABEL = {"bfm": "BFM, 108 held-out del congiunto", "ict": "ICT, 89 held-out del congiunto",
             "rexpr": "ICT con espressioni casuali, 89 held-out del congiunto",
             "bfm19": "BFM-19, held-out di congiunto E BFM-only",
             "ict992": "ICT, galleria grande: 992 held-out del congiunto"}
# Pair_metrics delle eval WS2 (ws2_eval.sbatch) per il controllo delle distanze dei modelli.
_EV = "eval_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d"
WS2_STAGES = {("joint", "bfm"): [RUNS / f"{_EV}_13da6115" / "ws2_cell"],
              ("joint", "ict"): [RUNS / f"{_EV}_60d50ab6" / "ws2_cell"],
              ("joint", "rexpr"): [RUNS / f"{_EV}_8afd021f" / "ws2_rexpr" / v for v in ("mixed", "neutral")],
              ("bfm_only", "bfm19"): [RUNS / f"{_EV}_16aab77d" / "ws2_cell"]}
GALLERY_BLOCKS = {"g_remesh": ("remesh", "original"), "g_noisy": ("noisy", "original"), "g_crop": ("crop", "original")}
GALLERY_METHODS = ("joint", "sim_nicp_p2tri", "template", "chamfer_sim")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-bootstrap-992-full", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234, help="Seme del ricampionamento")
    p.add_argument("--sets", default="bfm,ict,rexpr,bfm19,ict992")
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


def model_embeddings(path: Path, idx: LIndex) -> np.ndarray:
    z = np.load(path)
    keys = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))
    return _place(keys, z["Z"].astype(np.float64), idx, path)


def l2(Za: np.ndarray, Zb: np.ndarray) -> np.ndarray:
    sa, sb = (Za ** 2).sum(1), (Zb ** 2).sum(1)
    return np.sqrt(np.clip(sa[:, None] + sb[None, :] - 2 * Za @ Zb.T, 0, None))


def model_distances(path: Path, idx: LIndex) -> np.ndarray:
    Z = model_embeddings(path, idx)
    return l2(Z, Z)


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


def template_enrollments(path: Path, idx: LIndex) -> np.ndarray:
    z = np.load(path)
    keys = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["labels"]]))
    return _place(keys, z["R"].astype(np.float32), idx, path)


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


def template_distances_parallel(R: np.ndarray, workers: int, chunk: int = 64) -> np.ndarray:
    """``template_distances(R, R)`` a blocchi di righe su piu' processi (ict992: 5952 x 5952 x 4096)."""
    global _TD_R
    _TD_R = R
    with mp.get_context("fork").Pool(max(1, min(workers, 16))) as pool:
        parts = pool.map(_tdist_rows, [(s, min(s + chunk, len(R))) for s in range(0, len(R), chunk)])
    return np.concatenate(parts).astype(np.float64)


_TD_R: np.ndarray | None = None


def _tdist_rows(span):
    s, e = span
    return template_distances(_TD_R[s:e], _TD_R).astype(np.float32)


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


# ----------------------------------------------------------------------------------- misure

def tar_values(score: np.ndarray, genuine: np.ndarray, weights, n_reps: int) -> dict:
    """TAR@FAR per ogni FAR: punto (pesi 1) e ``n_reps`` repliche; ``weights(r)`` da' i pesi di TUTTE le
    coppie nella replica r (calcolati uno alla volta: con 9.8 milioni di coppie non stanno tutti in RAM).

    Soglia = punteggio del primo impostore (in ordine decrescente) oltre il quale la frazione pesata
    degli impostori accettati supererebbe FAR; accettato = punteggio strettamente sopra soglia.
    """
    imp = ~genuine
    order = np.argsort(-score[imp], kind="stable")
    si, gs = score[imp][order], score[genuine]
    out = {f"tar{f:g}": [] for f in FARS}
    for r in range(n_reps + 1):
        if r == 0:
            wi, wg = np.ones(len(si)), np.ones(len(gs))
        else:
            w = weights(r - 1)
            wi, wg = w[imp][order], w[genuine]
        cum = np.cumsum(wi) / wi.sum()
        for f in FARS:
            k = int(np.searchsorted(cum, f, side="right"))
            thr = si[k] if k < len(si) else -np.inf
            out[f"tar{f:g}"].append(float((wg * (gs > thr)).sum() / wg.sum()))
    return {k: np.asarray(v) for k, v in out.items()}


def _recog_task(task):
    """``zes._recog_task`` piu' TAR@FAR sulle stesse coppie di verifica e con gli stessi pesi."""
    key, D, idx, pairs_r, pairs_v, counts = task
    name, vals, n_nan = zes._recog_task(task)
    v = zes.verification_pairs(D, idx, pairs_v)
    sa, sb, gen, sc = v["sa"].to_numpy(), v["sb"].to_numpy(), v["genuine"].to_numpy(), v["score"].to_numpy()
    # Stessi pesi di zes.recognition_values: stessa persona c[sa], persone diverse c[sa] * c[sb].
    vals.update(tar_values(sc, gen, lambda r: np.where(gen, counts[r][sa], counts[r][sa] * counts[r][sb])
                           .astype(np.float64), len(counts)))
    return name, vals, n_nan


def gallery_values(M: np.ndarray, queries: list[str], gallery: list[str], counts: np.ndarray) -> tuple[dict, int]:
    """Retrieval e verifica su una matrice rettangolare query x galleria; bootstrap sulle query."""
    n_nan = int(np.isnan(M).sum())
    M = np.where(np.isfinite(M), M, np.inf)
    gpos = np.asarray([gallery.index(q) for q in queries])
    nq = len(queries)
    g = M[np.arange(nq), gpos][:, None]
    rank = 1 + (M < g).sum(1) + 0.5 * ((M == g).sum(1) - 1)
    hit, ap = (rank == 1).astype(np.float64), 1.0 / rank
    score = -M.ravel()
    gen = (np.arange(M.shape[1])[None, :] == gpos[:, None]).ravel()
    qidx = np.repeat(np.arange(nq), M.shape[1])
    out = {"rank1": [hit.mean()], "map": [ap.mean()], "auc": [zes.weighted_auc(score, gen, np.ones(len(score)))]}
    for c in counts:
        wq = c.astype(np.float64)
        out["rank1"].append((wq * hit).sum() / wq.sum())
        out["map"].append((wq * ap).sum() / wq.sum())
        out["auc"].append(zes.weighted_auc(score, gen, wq[qidx]))
    out = {k: np.asarray(x) for k, x in out.items()}
    out.update(tar_values(score, gen, lambda r: counts[r][qidx].astype(np.float64), len(counts)))
    return out, n_nan


def _gallery_task(task):
    key, M, queries, gallery, counts = task
    vals, n_nan = gallery_values(M, queries, gallery, counts)
    return key, vals, n_nan


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
               "neutral_gallery": "secondario: galleria neutra, query con espressione",
               "g_remesh": "PRIMARIO della galleria grande: 100 query in remesh, galleria di 992 in original",
               "g_noisy": "galleria grande, blocco degenere (noisy = original perturbata): 100 query in noisy, 992 in original",
               "g_crop": "a parte: 100 query in crop, galleria di 992 in original",
               "full_nocrop": "secondario: tutte le 992 query, 20 coppie di topologie senza crop (solo metodi a iscrizione)",
               "full_crop": "secondario: tutte le 992 query, coppie con crop (solo metodi a iscrizione)"}


# ------------------------------------------------------------------------------------ tempi

def timing_table() -> pd.DataFrame:
    data = {}
    for part in ("model", "facebench", "template", "arcface"):
        p = OUT / "timing" / f"{part}.json"
        if p.exists():
            data[part] = json.loads(p.read_text())
    E, S, P = "iscrizione (per mesh)", "ricerca su galleria iscritta (per query)", "a coppie (per coppia)"
    specs = [(E, "model", "operators_cpu", "congiunto: operatori DiffusionNet (k=128), CPU 1 thread"),
             (E, "model", "embed_cuda", "congiunto: embedding (lettura + forward), GPU"),
             (E, "model", "embed_cpu", "congiunto: embedding (lettura + forward), CPU 1 thread"),
             (E, "template", "enroll", "NICP su template: ICP di similarita' + NICP + Procrustes, CPU 1 thread"),
             (E, "arcface", "best_total", "ArcFace migliore: inquadratura + 3 render + 3 embedding, CPU 1 thread"),
             (E, "arcface", "best_render", "  di cui inquadratura + 3 render"),
             (E, "arcface", "first_total", "ArcFace prima tornata (camera di dominio), CPU 1 thread"),
             (S, "model", "compare_cpu", "congiunto: un confronto ||z_a - z_b|| (1:1), CPU"),
             *[(S, "model", f"retrieval_{d}_N{n}", f"congiunto 1:{n:,} (256-d, distanze + argsort), "
                + ("CPU 1 thread" if d == "cpu" else "GPU")) for n in (100, 1000, 10000) for d in ("cpu", "gpu")],
             *[(S, "template", f"search_N{n}", f"NICP su template 1:{n:,} (4096 x 3, L2 media per vertice), CPU 1 thread")
               for n in (100, 1000, 10000)],
             *[(S, "arcface", f"search_N{n}", f"ArcFace 1:{n:,} (512-d, coseno), CPU 1 thread") for n in (100, 1000, 10000)],
             (P, "facebench", "sim_icp_chamfer", "ICP di similarita' + Chamfer, CPU 1 thread"),
             (P, "facebench", "sim_nicp_p2tri", "ICP di similarita' + NICP P2Tri, CPU 1 thread"),
             (P, "facebench", "icp_chamfer", "ICP rigido + Chamfer (prima tornata), CPU 1 thread"),
             (P, "facebench", "nicp_p2tri", "ICP rigido + NICP P2Tri (prima tornata), CPU 1 thread")]
    rows = []

    def add(phase, part, key, lab, x, estimate=False):
        q25, q50, q75 = np.percentile(x, [25, 50, 75])
        rows.append({"phase": phase, "part": part, "key": key, "label": lab, "n": len(x), "estimate": estimate,
                     "median_s": q50, "q25_s": q25, "q75_s": q75})

    for phase, part, key, lab in specs:
        x = np.asarray(data.get(part, {}).get(key, []), dtype=np.float64)
        if len(x):
            add(phase, part, key, lab, x)
        if part == "model" and key == "embed_cuda" and len(x):
            ops = np.asarray(data["model"]["operators_cpu"])
            if len(ops) == len(x):
                add(E, "model", "ops_plus_embed_gpu", "congiunto: totale (operatori CPU + embedding GPU)", ops + x)
    for key in ("sim_icp_chamfer", "sim_nicp_p2tri"):
        x = np.asarray(data.get("facebench", {}).get(key, []), dtype=np.float64)
        for n in (100, 1000, 10000):
            if len(x):
                add(S, "facebench", f"{key}_N{n}", f"{LABEL[key]} 1:{n:,}: STIMA = N x tempo per coppia", n * x, True)
    df = pd.DataFrame(rows)
    df.attrs["hw"] = {k: {h: v.get(h) for h in ("host", "cpu", "gpu", "job", "omp_threads")} for k, v in data.items()}
    df.attrs["n_vertices"] = data.get("model", {}).get("n_vertices", [])
    return df


def fmt_time(s: float) -> str:
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 1:
        return f"{s:.2f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.2f} ms"
    return f"{s * 1e6:.1f} us"


# ------------------------------------------------------------------------------------- main

def record(rows, deltas, set_name, blk, rec, n_subjects):
    for (b, m), (vals, n_nan) in rec.items():
        if b != blk:
            continue
        r = {"set": set_name, "block": blk, "method": m, "n_subjects": n_subjects, "n_nan_distances": n_nan}
        for k in METRICS:
            r[k], (r[f"{k}_ci_low"], r[f"{k}_ci_high"]) = float(vals[k][0]), zes.ci(vals[k])
        rows.append(r)
    if (blk, "joint") not in rec:
        return
    for (b, m), (vals, _) in rec.items():
        if b != blk or m == "joint":
            continue
        a = rec[(blk, "joint")][0]
        r = {"set": set_name, "block": blk, "model": "joint", "other": m}
        for k in METRICS:
            d = a[k] - vals[k]
            r[k], (r[f"{k}_ci_low"], r[f"{k}_ci_high"]) = float(d[0]), zes.ci(d)
            r[f"{k}_p_le0"] = float((d[1:] <= 0).mean())
        deltas.append(r)


def square_set(name, spec, sets, args, workers, rows, deltas, checks, info):
    src = spec.get("subset_of", name)
    idx = LIndex(spec["subjects"], spec["labels"])
    D = {}
    for m in ["joint"] + (["bfm_only"] if name == "bfm19" else []):
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
    if name in ("bfm", "ict") and "chamfer" in D:
        try:
            Dc = facebench_distances(OUT / "facebench" / src, "chamfer_sim", idx)
            ok = np.isfinite(Dc) & np.isfinite(D["chamfer"])
            checks.append(f"- {SET_LABEL[name]}: `chamfer_sim` (variante sim) contro `chamfer` (prima tornata), "
                          f"max |diff| = {np.abs(Dc - D['chamfer'])[ok].max():.2e} su {int(ok.sum())} distanze")
        except FileNotFoundError:
            pass
    tp = OUT / "template" / f"{TEMPLATE_SRC[name]}.npz"
    if tp.exists():
        R = template_enrollments(tp, idx)
        D["template"] = template_distances(R, R)
    else:
        print(f"[ir-sum] ATTENZIONE: {tp} assente, riga saltata", flush=True)
    for m, (root, mode) in ARC.items():
        p = OUT / root / src / mode / "arcface_views.npz"
        if p.exists():
            D[m] = arcface_distances(p, idx)
        else:
            print(f"[ir-sum] ATTENZIONE: {p} assente, riga saltata", flush=True)
    blocks = blocks_for(name, spec["labels"])
    counts = zes.bootstrap_counts(len(idx.subjects), args.n_bootstrap,
                                  base.stable_seed(args.seed, f"indomain_recognition:{name}"))
    print(f"[ir-sum] {name}: {len(idx.subjects)} soggetti, metodi {list(D)}", flush=True)
    rec = {}
    with mp.get_context("fork").Pool(min(workers, 16)) as pool:
        tasks = [((blk, m), D[m], idx, pr, pv, counts) for blk, (pr, pv) in blocks.items() for m in D]
        for key, vals, n_nan in pool.imap_unordered(_recog_task, tasks):
            rec[key] = (vals, n_nan)
    for blk in blocks:
        record(rows, deltas, name, blk, rec, len(idx.subjects))
    n = len(idx.subjects)
    info[name] = {blk: (f"Retrieval: {len(pr) * n} query ({len(pr)} coppie ordinate x {n}), galleria di {n}. "
                        f"Verifica: {len(pv) * n} coppie stessa persona, {len(pv) * n * (n - 1)} persone diverse.")
                  for blk, (pr, pv) in blocks.items()}


def gallery_set(name, spec, args, workers, rows, deltas, info):
    subj, queries = spec["subjects"], spec["queries"]
    full = LIndex(subj, spec["labels"])
    Z = model_embeddings(OUT / "embeddings" / f"joint__{name}.npz", full)
    tp = OUT / "template" / f"{TEMPLATE_SRC[name]}.npz"
    R = template_enrollments(tp, full) if tp.exists() else None
    tasks = []
    counts = zes.bootstrap_counts(len(queries), args.n_bootstrap,
                                  base.stable_seed(args.seed, f"indomain_recognition:{name}:gallery"))
    for blk, (lq, lg) in GALLERY_BLOCKS.items():
        rq = np.asarray([full.pos[(s, lq)] for s in queries])
        rg = full.rows(lg)
        tasks.append(((blk, "joint"), l2(Z[rq], Z[rg]), queries, subj, counts))
        if R is not None:
            tasks.append(((blk, "template"), template_distances(R[rq], R[rg]), queries, subj, counts))
        for m in ("sim_nicp_p2tri", "chamfer_sim"):
            p = OUT / "facebench" / name / m / f"{lq}__to__{lg}.npz"
            if not p.exists():
                print(f"[ir-sum] ATTENZIONE: {p} assente, riga saltata", flush=True)
                continue
            with np.load(p) as z:
                if [str(s) for s in z["subjects_a"]] != queries or [str(s) for s in z["subjects_b"]] != subj:
                    raise SystemExit(f"{p}: query o galleria diverse da sets.json")
                tasks.append(((blk, m), z["values"].astype(np.float64), queries, subj, counts))
        info.setdefault(name, {})[blk] = (f"Retrieval: {len(queries)} query in `{lq}`, galleria di {len(subj)} in "
                                          f"`{lg}`. Verifica: {len(queries)} coppie stessa persona, "
                                          f"{len(queries) * (len(subj) - 1)} persone diverse.")
    rec = {}
    with mp.get_context("fork").Pool(min(workers, 8)) as pool:
        for key, vals, n_nan in pool.imap_unordered(_gallery_task, tasks):
            rec[key] = (vals, n_nan)
    for blk in GALLERY_BLOCKS:
        record(rows, deltas, name, blk, rec, len(queries))

    # Secondario: tutte le 992 query, solo metodi a iscrizione (congiunto, template), zes come per gli
    # insiemi quadrati ma con meno repliche (9.8 milioni di coppie di verifica per blocco).
    D = {"joint": l2(Z, Z)}
    if R is not None:
        D["template"] = template_distances_parallel(R, workers)
    counts_full = zes.bootstrap_counts(len(subj), args.n_bootstrap_992_full,
                                       base.stable_seed(args.seed, f"indomain_recognition:{name}:full"))
    rec = {}
    blocks = {f"full_{k}": v for k, v in blocks_for(name, spec["labels"]).items()}
    with mp.get_context("fork").Pool(min(workers, 4)) as pool:
        tasks = [((blk, m), D[m], full, pr, pv, counts_full) for blk, (pr, pv) in blocks.items() for m in D]
        for key, vals, n_nan in pool.imap_unordered(_recog_task, tasks):
            rec[key] = (vals, n_nan)
    n = len(subj)
    for blk, (pr, pv) in blocks.items():
        record(rows, deltas, name, blk, rec, n)
        info[name][blk] = (f"Retrieval: {len(pr) * n} query ({len(pr)} coppie ordinate x {n}), galleria di {n}. "
                           f"Verifica: {len(pv) * n} coppie stessa persona, {len(pv) * n * (n - 1)} persone diverse. "
                           f"{args.n_bootstrap_992_full} repliche bootstrap.")


def main() -> None:
    args = parse_args()
    sets = json.loads((OUT / "sets.json").read_text())["sets"]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    rows, deltas, checks, info = [], [], [], {}
    for name in args.sets.split(","):
        if name == "ict992":
            gallery_set(name, sets[name], args, workers, rows, deltas, info)
        else:
            square_set(name, sets[name], sets, args, workers, rows, deltas, checks, info)

    rec_table, delta_table = pd.DataFrame(rows), pd.DataFrame(deltas)
    timing = timing_table()
    rec_table.to_csv(OUT / "recognition.csv", index=False)
    delta_table.to_csv(OUT / "recognition_paired.csv", index=False)
    timing.to_csv(OUT / "timing.csv", index=False)

    def verdict(lo: float, hi: float) -> str:
        return "sopra" if lo > 0 else ("sotto" if hi < 0 else "pari")

    tar_h = " | ".join(f"TAR@FAR {f:g}" for f in FARS)
    lines = [(OUT / "protocol.md").read_text().rstrip(), "\n---\n", "# Risultati\n",
             f"CI 95% bootstrap per soggetto, {args.n_bootstrap} repliche (salvo dove detto), le stesse per tutti i "
             "metodi di un blocco. mAP = MRR (un solo rilevante). Distanze NaN (coppie fallite) = +inf.\n",
             "**Nota (revisione 1):** il congiunto ha visto in training la topologia crop (e noisy, down8k, remesh, "
             "up60k) dei soggetti di training, cioe' un'augmentation che nessuna baseline ha avuto. Le righe ICP "
             "rigido (prima tornata) portano l'artefatto di scala del crop; le righe ICP di similarita' no.\n"]
    for name, blks in info.items():
        lines.append(f"## {SET_LABEL[name]}\n")
        for blk, text in blks.items():
            sub = rec_table[(rec_table["set"] == name) & (rec_table["block"] == blk)].set_index("method")
            if sub.empty:
                continue
            lines += [f"### {BLOCK_LABEL[blk]}\n", text + "\n",
                      f"| metodo | rank-1 | mAP | AUC verifica | {tar_h} | NaN |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for m in [x for x in LABEL if x in sub.index]:
                r = sub.loc[m]
                lines.append(f"| {LABEL[m]} | " + " | ".join(zes.fmt(r[k], r[f"{k}_ci_low"], r[f"{k}_ci_high"])
                                                            for k in METRICS) + f" | {int(r['n_nan_distances'])} |")
            dsub = delta_table[(delta_table["set"] == name) & (delta_table["block"] == blk)] \
                if not delta_table.empty else delta_table
            if not dsub.empty:
                dsub = dsub.set_index("other")
                lines += ["", "Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):\n",
                          f"| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | "
                          + " | ".join(f"TAR@FAR {f:g}: delta [CI]" for f in FARS) + " | lettura rank-1 / AUC |",
                          "| --- | --- | --- | --- | --- | --- |"]
                for m in [x for x in LABEL if x in dsub.index]:
                    r = dsub.loc[m]
                    cells = [f"{zes.fmt(r[k], r[f'{k}_ci_low'], r[f'{k}_ci_high'], True)} ({r[f'{k}_p_le0']:.3f})"
                             for k in ("rank1", "auc")]
                    cells += [zes.fmt(r[k], r[f"{k}_ci_low"], r[f"{k}_ci_high"], True) for k in METRICS[3:]]
                    lines.append(f"| {LABEL[m]} | " + " | ".join(cells)
                                 + f" | {verdict(r['rank1_ci_low'], r['rank1_ci_high'])} / "
                                   f"{verdict(r['auc_ci_low'], r['auc_ci_high'])} |")
            lines.append("")

    lines.append("## Tempi\n")
    for part, h in timing.attrs.get("hw", {}).items():
        lines.append(f"- {part}: host `{h.get('host')}`, CPU {h.get('cpu')}"
                     + (f", GPU {h.get('gpu')}" if h.get("gpu") else "") + f", job {h.get('job')}, "
                     f"OMP_NUM_THREADS={h.get('omp_threads')}")
    nv = timing.attrs.get("n_vertices", [])
    if nv:
        lines.append(f"- campione: {len(nv)} mesh, vertici da {min(nv)} a {max(nv)} (mediana {int(np.median(nv))})")
    for phase in dict.fromkeys(timing["phase"]) if not timing.empty else []:
        lines += ["", f"### {phase}\n", "| voce | n | mediana | IQR (25-75%) |", "| --- | --- | --- | --- |"]
        for r in timing[timing["phase"] == phase].itertuples():
            iqr = "-" if r.estimate else f"{fmt_time(r.q25_s)} - {fmt_time(r.q75_s)}"
            lines.append(f"| {r.label} | {r.n} | {fmt_time(r.median_s)} | {iqr} |")
    lines += ["", "## Controlli\n", *checks,
              "- faceBench: NICP asimmetrico, orientazione della coppia = ordine delle etichette (insiemi quadrati) "
              "o query -> galleria (ict992), non sempre query -> galleria.",
              f"- leak (`sets.json`): {json.dumps(json.loads((OUT / 'sets.json').read_text())['leak'])}", ""]
    if (OUT / "lettura.md").exists():
        lines += ["---\n", (OUT / "lettura.md").read_text().rstrip(), ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ir-sum] scritto {OUT / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
