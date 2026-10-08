#!/usr/bin/env python3
"""Tabella del test con espressioni (WBES_ZS_EXPR=1): riconoscimento d'identita' e ranking con GT neutra.

    aau/run.sh aau/zs3dmm/zs_expr_summarize.py --runs aau/runs/ws_faceverse_expr/data_<fp> \\
        --summary-dir aau/runs/ws_faceverse_expr --gt <expr_view>/gt_matrix.npz --view-dir <expr_view>/npz
    (zs_expr_summarize.sbatch)

Protocollo: ``<summary-dir>/protocol.md``, dichiarato prima delle eval e copiato in testa al
summary. In breve:

PRIMARIO, riconoscimento d'identita' sulle 5 topologie senza crop (solo etichette):
  - retrieval: query (A, t1), galleria = le 100 mesh in t2 != t1 (una per soggetto); rank-1 e
    mAP (un solo rilevante: AP = 1/rank). 20 coppie ordinate x 100 query. Distanze NaN (coppie
    faceBench fallite) contano come +inf. CI: soggetti query ricampionati, galleria fissa.
  - verifica: AUC di -distanza, coppie (A, t1)-(B, t2) con t1 < t2 nell'ordine di TOPOLOGIES:
    stessa persona (A = B) contro persone diverse. CI: soggetti ricampionati, coppia pesata per
    il prodotto dei conteggi (stessa persona: il conteggio).
  - delta APPAIATI modello - baseline sulle stesse repliche.
  Crop a parte: le stesse misure sulle coppie di topologie con crop da un lato.
SECONDARIO: Spearman con la GT d'identita' neutra, mesh-pair cross-topologia senza crop, dalle
pair_metrics dello script di breakdown (righe soggetto a < b); modello - Chamfer eval e modello -
Chamfer regione stabile appaiati (``paired_bootstrap`` di zs_summarize.py). Terziario:
subject-pair-mean.

Distanze per metodo, su una matrice (600, 600) indicizzata da (soggetto, topologia):
  - modelli: ||z_i - z_j|| dagli ``embeddings.npz`` di zs_embed.py (controllo: coincide con
    ``latent_distance`` delle pair_metrics);
  - faceBench: matrici i<j di alignment_matrix.py per (ta, tb) e (tb, ta), piu' le coppie
    stesso soggetto di zs_bl_same.py. Per NICP (asimmetrico) l'orientazione della coppia e'
    quella della matrice i<j, non quella query -> galleria;
  - Chamfer regione stabile e intero (stessa implementazione): zs_region_chamfer.py.

Frame dei modelli: suffisso ``_flip`` = convenzione BFM (rotazione nativa + facce invertite),
``_frame-xmymz`` = convenzione ICT. Righe di riferimento fissate dal protocollo: BFM-only in
BFM, ICT-only in ICT, congiunto in entrambe; le altre due sono secondarie.
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
sys.path.insert(0, str(THIS_DIR))

import zs_summarize as zsum  # noqa: E402
from zs_stage import TOPOLOGIES, select_subjects  # noqa: E402

base = zsum.base
common = zsum.common
NOCROP = tuple(t for t in TOPOLOGIES if t != "crop")
CONVENTIONS = {"bfm": "_flip", "ict": "_frame-xmymz"}
CONV_LABEL = {"bfm": "convenzione BFM (nativa + facce invertite)", "ict": "convenzione ICT (Rx 180)"}
# (braccio, convenzione, riferimento secondo il protocollo)
MODEL_ROWS = (("joint", "bfm", True), ("joint", "ict", True), ("bfm_only", "bfm", True),
              ("ict_only", "ict", True), ("bfm_only", "ict", False), ("ict_only", "bfm", False),
)
BL_FACEBENCH = zsum.BL_METRICS
BL_REGION = ("chamfer_stable", "chamfer_full")
BL_LABEL = {**zsum.BL_LABEL, "chamfer_stable": "Chamfer regione stabile",
            "chamfer_full": "Chamfer intero (stessa implementazione)"}
STAGE = "zs_zeroshot"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, required=True, help="$ZS_RUNS/data_<fp>")
    p.add_argument("--summary-dir", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True, help="GT d'identita' (neutra)")
    p.add_argument("--view-dir", type=Path, required=True, help="vista con espressioni (npz/), per i soggetti")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234, help="Seme del ricampionamento, non del modello")
    p.add_argument("--eval-seed", type=int, default=1234, help="WBES_EVAL_SEED: scelta dei soggetti")
    return p.parse_args()


# ------------------------------------------------------------------- distanze (600 x 600)

class Index:
    def __init__(self, subjects: list[str]):
        self.subjects = subjects
        self.keys = [(s, t) for s in subjects for t in TOPOLOGIES]
        self.pos = {k: i for i, k in enumerate(self.keys)}

    def rows(self, topology: str) -> np.ndarray:
        return np.asarray([self.pos[(s, topology)] for s in self.subjects])


def model_distances(stage: Path, idx: Index) -> np.ndarray:
    z = np.load(stage / "embeddings.npz")
    keys = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))
    if sorted(keys) != sorted(idx.keys):
        raise SystemExit(f"{stage}: embedding di mesh diverse da quelle attese")
    Z = np.zeros((len(idx.keys), z["Z"].shape[1]))
    for k, row in zip(keys, z["Z"].astype(np.float64)):
        Z[idx.pos[k]] = row
    sq = (Z ** 2).sum(1)
    return np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0, None))


def facebench_distances(root: Path, metric: str, idx: Index) -> np.ndarray:
    n = len(idx.subjects)
    D = np.full((len(idx.keys), len(idx.keys)), np.nan)
    iu, ju = np.triu_indices(n, 1)
    for ta in TOPOLOGIES:
        for tb in TOPOLOGIES:
            if ta == tb:
                continue
            M, subj, _, _, _ = common.load_matrix(common.matrix_path(metric, ta, tb, root))
            if subj != idx.subjects:
                raise SystemExit(f"faceBench {metric} {ta}->{tb}: soggetti diversi dal set")
            ra, rb = idx.rows(ta), idx.rows(tb)
            D[ra[iu], rb[ju]] = M[iu, ju]   # X = soggetto i in ta, Y = j in tb
            D[rb[ju], ra[iu]] = M[iu, ju]   # stessa coppia vista da (j, tb)
            with np.load(root / "matrices_same" / metric / f"{ta}__to__{tb}.npz") as z:
                if [str(s) for s in z["subjects"]] != idx.subjects:
                    raise SystemExit(f"faceBench stesso soggetto {metric} {ta}->{tb}: soggetti diversi")
                D[ra, rb] = z["values"]
    return D


def region_distances(path: Path, idx: Index) -> dict:
    z = np.load(path)
    keys = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))
    order = np.asarray([keys.index(k) for k in idx.keys])
    out = {m: z[m][np.ix_(order, order)] for m in BL_REGION}
    out["_kept"] = z["kept_vertex_fraction"]
    return out


# --------------------------------------------------------------- riconoscimento d'identita'

def retrieval_queries(D: np.ndarray, idx: Index, topo_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Una riga per query: soggetto, (t1, t2), rank del soggetto giusto, hit@1, AP = 1/rank."""
    rows = []
    for t1, t2 in topo_pairs:
        R = D[np.ix_(idx.rows(t1), idx.rows(t2))]
        R = np.where(np.isfinite(R), R, np.inf)
        g = np.diag(R)[:, None]
        rank = 1 + (R < g).sum(1) + 0.5 * ((R == g).sum(1) - 1)   # pari: a meta' strada
        for k, s in enumerate(idx.subjects):
            rows.append({"subject": s, "t1": t1, "t2": t2, "rank": float(rank[k]),
                         "hit1": float(rank[k] == 1), "ap": 1.0 / float(rank[k])})
    return pd.DataFrame(rows)


def verification_pairs(D: np.ndarray, idx: Index, topo_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    n = len(idx.subjects)
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    frames = []
    for t1, t2 in topo_pairs:
        M = D[np.ix_(idx.rows(t1), idx.rows(t2))]
        frames.append(pd.DataFrame({"sa": a.ravel(), "sb": b.ravel(), "t1": t1, "t2": t2,
                                    "score": -M.ravel(), "genuine": (a == b).ravel()}))
    out = pd.concat(frames, ignore_index=True)
    out["score"] = out["score"].fillna(-np.inf)
    return out


def weighted_auc(score: np.ndarray, genuine: np.ndarray, w: np.ndarray) -> float:
    g, i = genuine & (w > 0), ~genuine & (w > 0)
    si, wi = score[i], w[i]
    order = np.argsort(si, kind="stable")
    si, cw = si[order], np.concatenate([[0.0], np.cumsum(wi[order])])
    lo = np.searchsorted(si, score[g], side="left")
    hi = np.searchsorted(si, score[g], side="right")
    below = cw[lo] + 0.5 * (cw[hi] - cw[lo])
    return float((w[g] * below).sum() / (w[g].sum() * wi.sum()))


def bootstrap_counts(n_subjects: int, n_bootstrap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_subjects, n_subjects), minlength=n_subjects)
                     for _ in range(n_bootstrap)])


def recognition_values(q: pd.DataFrame, v: pd.DataFrame, idx: Index, counts: np.ndarray) -> dict:
    """Punto e repliche bootstrap di rank-1, mAP, AUC (repliche: righe di ``counts``)."""
    s2i = {s: k for k, s in enumerate(idx.subjects)}
    qs = q["subject"].map(s2i).to_numpy()
    hit, ap = q["hit1"].to_numpy(), q["ap"].to_numpy()
    sa, sb = v["sa"].to_numpy(), v["sb"].to_numpy()
    sc, gen = v["score"].to_numpy(), v["genuine"].to_numpy()
    out = {"rank1": [hit.mean()], "map": [ap.mean()], "auc": [weighted_auc(sc, gen, np.ones(len(sc)))]}
    for c in counts:
        wq = c[qs].astype(np.float64)
        out["rank1"].append((wq * hit).sum() / wq.sum())
        out["map"].append((wq * ap).sum() / wq.sum())
        wv = np.where(gen, c[sa], c[sa] * c[sb]).astype(np.float64)
        out["auc"].append(weighted_auc(sc, gen, wv))
    return {k: np.asarray(x) for k, x in out.items()}


def _recog_task(task):
    name, D, idx, pairs_r, pairs_v, counts = task
    topo = np.asarray([t for _, t in idx.keys])
    n_nan = int(np.isnan(D)[topo[:, None] != topo[None, :]].sum())   # solo coppie di topologie diverse
    return name, recognition_values(retrieval_queries(D, idx, pairs_r), verification_pairs(D, idx, pairs_v),
                                    idx, counts), n_nan


def ci(x: np.ndarray) -> tuple[float, float]:
    return tuple(float(v) for v in np.percentile(x[1:], [2.5, 97.5]))


# --------------------------------------------------------------------------- secondario

def secondary_frame(stage: Path, D_by_method: dict, idx: Index, latent_name: str) -> pd.DataFrame:
    pm = base.read_pair_metrics(stage)
    pm = pm[pm["topology_a"].isin(NOCROP) & pm["topology_b"].isin(NOCROP)].reset_index(drop=True)
    ia = np.asarray([idx.pos[(s, t)] for s, t in zip(pm["subject_a"].astype(str), pm["topology_a"])])
    ib = np.asarray([idx.pos[(s, t)] for s, t in zip(pm["subject_b"].astype(str), pm["topology_b"])])
    check = float(np.abs(D_by_method[latent_name][ia, ib] - pm["latent_distance"]).max())
    for m, D in D_by_method.items():
        pm[m] = D[ia, ib]
    pm.attrs["latent_check"] = check
    return pm


def _sec_task(task):
    key, df, col_a, col_b, n, seed = task
    bm = base.load_bootstrap_module()
    rng = np.random.default_rng(seed)
    if col_b is None:
        return key, base.bootstrap_row(df, col_a, n, rng, bm)
    return key, zsum.paired_bootstrap(df, col_a, col_b, n, rng, bm)


# --------------------------------------------------------------------------- markdown

def fmt(point: float, lo: float, hi: float, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(point)} [{f.format(lo)}, {f.format(hi)}]"


def main() -> None:
    args = parse_args()
    subjects = select_subjects(args.view_dir, args.eval_seed)
    idx = Index(subjects)
    print(f"[zs-expr-sum] {len(subjects)} soggetti (primi {subjects[:3]})", flush=True)

    # run grande BFM+ICT+GNM (aau/data_scale), anche checkpoint intermedi: righe aggiunte solo se
    # hanno risultati, nelle due convenzioni (senza, la tabella e' quella di sempre)
    global MODEL_ROWS
    for conv, suf in CONVENTIONS.items():
        for arm in zsum.discover_scale_arms(args.runs, suf):
            MODEL_ROWS = MODEL_ROWS + ((arm, conv, True),)
    # Distanze per metodo
    D, stages, sources = {}, {}, {}
    for arm, conv, _ in MODEL_ROWS:
        name = f"{arm}@{conv}"
        stage = args.runs / f"{arm}{CONVENTIONS[conv]}_topology" / STAGE
        if not (stage / "embeddings.npz").exists():
            print(f"[zs-expr-sum] ATTENZIONE: {name}: embeddings assenti ({stage}), riga saltata", flush=True)
            continue
        D[name] = model_distances(stage, idx)
        stages[name] = stage if (stage / ".done").exists() else None
        staged = json.loads((stage.parent / "subjects.json").read_text())
        if sorted(staged["subjects"]) != subjects:
            raise SystemExit(f"{name}: zs_stage ha valutato soggetti diversi")
        sources[name] = f"{stage.relative_to(args.runs)}" + ("" if stages[name] else " (breakdown non finito)")
    bl_root = args.runs / "baselines"
    for m in BL_FACEBENCH:
        try:
            D[m] = facebench_distances(bl_root, m, idx)
        except FileNotFoundError as exc:
            print(f"[zs-expr-sum] ATTENZIONE: faceBench {m} incompleto ({exc}), riga saltata", flush=True)
    kept = None
    if (bl_root / "region_chamfer.npz").exists():
        reg = region_distances(bl_root / "region_chamfer.npz", idx)
        kept = reg.pop("_kept")
        D.update(reg)
    models = [n for n in D if "@" in n]
    baselines = [n for n in D if "@" not in n]
    print(f"[zs-expr-sum] modelli {models}, baseline {baselines}", flush=True)

    # PRIMARIO e crop
    counts = bootstrap_counts(len(subjects), args.n_bootstrap, base.stable_seed(args.seed, "expr_recognition"))
    blocks = {
        "nocrop": ([(a, b) for a in NOCROP for b in NOCROP if a != b],
                   [(a, b) for i, a in enumerate(NOCROP) for b in NOCROP[i + 1:]]),
        "crop": ([(a, b) for a in TOPOLOGIES for b in TOPOLOGIES if a != b and "crop" in (a, b)],
                 [("crop", b) for b in NOCROP]),
    }
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    rec = {}
    with mp.get_context("fork").Pool(min(workers, 16)) as pool:
        tasks = [((blk, name), D[name], idx, pr, pv, counts) for blk, (pr, pv) in blocks.items() for name in D]
        for key, vals, n_nan in pool.imap_unordered(_recog_task, tasks):
            rec[key] = (vals, n_nan)
    rec_rows = []
    for (blk, name), (vals, n_nan) in rec.items():
        r = {"block": blk, "method": name, "n_nan_distances": n_nan}
        for m in ("rank1", "map", "auc"):
            r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(vals[m][0]), ci(vals[m])
        rec_rows.append(r)
    rec_table = pd.DataFrame(rec_rows)
    deltas = []
    for blk in blocks:
        for name in models:
            for bl in baselines:
                a, b = rec[(blk, name)][0], rec[(blk, bl)][0]
                r = {"block": blk, "model": name, "baseline": bl}
                for m in ("rank1", "map", "auc"):
                    d = a[m] - b[m]
                    r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(d[0]), ci(d)
                    r[f"{m}_p_le0"] = float((d[1:] <= 0).mean())
                deltas.append(r)
    delta_table = pd.DataFrame(deltas)

    # SECONDARIO / terziario
    sec_tasks, sec_meta, checks = [], {}, {}
    baselines_queued = False
    for name in models:
        if stages[name] is None:
            continue
        df = secondary_frame(stages[name], D, idx, name)
        checks[name] = df.attrs["latent_check"]
        spm = df.groupby(["subject_a", "subject_b"], as_index=False)[["gt_distance", "latent_distance", "raw_chamfer"]
                                                                      + baselines].mean()
        for proto, frame in (("mesh_pair_nocrop", df), ("subject_pair_mean_nocrop", spm)):
            for col_b in ("raw_chamfer",) + (("chamfer_stable",) if "chamfer_stable" in D else ()):
                key = (name, proto, "latent_distance", col_b)
                sec_tasks.append((key, frame, "latent_distance", col_b, args.n_bootstrap,
                                  base.stable_seed(args.seed, "expr_sec", *key)))
            if not baselines_queued:  # le baseline sono le stesse righe per ogni braccio
                for col in ["raw_chamfer"] + baselines:
                    key = ("baseline", proto, col, None)
                    sec_tasks.append((key, frame, col, None, args.n_bootstrap,
                                      base.stable_seed(args.seed, "expr_sec", *map(str, key))))
        baselines_queued = True
    with mp.get_context("fork").Pool(min(workers, len(sec_tasks) or 1)) as pool:
        for key, res in pool.imap_unordered(_sec_task, sec_tasks):
            sec_meta[key] = res
    sec_rows = [{"method": k[0], "protocol": k[1], "col_a": k[2], "col_b": k[3], **v} for k, v in sec_meta.items()]
    sec_table = pd.DataFrame(sec_rows)

    # Uscite
    out = args.summary_dir
    out.mkdir(parents=True, exist_ok=True)
    rec_table.to_csv(out / "recognition.csv", index=False)
    delta_table.to_csv(out / "recognition_paired.csv", index=False)
    sec_table.to_csv(out / "secondary.csv", index=False)
    expr_manifest = json.loads((args.view_dir.parent / "manifest.json").read_text()).get("source_manifest", {})

    def label(name: str) -> str:
        if "@" in name:
            arm, conv = name.split("@")
            ref = next(r for a, c, r in MODEL_ROWS if a == arm and c == conv)
            return f"{zsum.ARM_LABEL[arm]}, {CONV_LABEL[conv]}" + ("" if ref else " (secondaria)")
        return BL_LABEL.get(name, name)

    def rec_md(blk: str) -> list[str]:
        lines = ["| metodo | rank-1 | mAP | AUC verifica | distanze NaN |", "| --- | --- | --- | --- | --- |"]
        sub = rec_table[rec_table["block"] == blk].set_index("method")
        for name in models + baselines:
            r = sub.loc[name]
            lines.append(f"| {label(name)} | " + " | ".join(fmt(r[m], r[f"{m}_ci_low"], r[f"{m}_ci_high"])
                                                           for m in ("rank1", "map", "auc"))
                         + f" | {r['n_nan_distances']} |")
        return lines

    def delta_md(blk: str) -> list[str]:
        lines = ["| modello | baseline | rank-1: delta [CI 95%] (P<=0) | mAP: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) |",
                 "| --- | --- | --- | --- | --- |"]
        sub = delta_table[delta_table["block"] == blk] if not delta_table.empty else delta_table
        for r in sub.itertuples():
            lines.append(f"| {label(r.model)} | {label(r.baseline)} | "
                         + " | ".join(f"{fmt(getattr(r, m), getattr(r, m + '_ci_low'), getattr(r, m + '_ci_high'), True)} "
                                      f"({getattr(r, m + '_p_le0'):.3f})" for m in ("rank1", "map", "auc")) + " |")
        return lines

    def sec_md(proto: str) -> list[str]:
        lines = ["| metodo | Spearman [CI 95%] | delta vs Chamfer eval [CI] (P<=0) | delta vs Chamfer regione stabile [CI] (P<=0) |",
                 "| --- | --- | --- | --- |"]
        s = sec_table[sec_table["protocol"] == proto] if not sec_table.empty else sec_table
        for name in models:
            rr = s[(s["method"] == name)] if not s.empty else s
            if rr.empty:
                continue
            cells = [label(name)]
            first = rr.iloc[0]
            cells.append(f"{first['a']:.3f}")
            for col_b in ("raw_chamfer", "chamfer_stable"):
                x = rr[rr["col_b"] == col_b]
                cells.append("-" if x.empty else f"{fmt(x.iloc[0]['diff'], x.iloc[0]['ci_low'], x.iloc[0]['ci_high'], True)} "
                                                 f"({x.iloc[0]['p_boot_le0']:.3f})")
            lines.append("| " + " | ".join(cells) + " |")
        b = s[s["method"] == "baseline"] if not s.empty else s
        for r in b.itertuples():
            lines.append(f"| {label('Chamfer eval' if r.col_a == 'raw_chamfer' else r.col_a)} | "
                         f"{fmt(r.spearman, r.ci_low, r.ci_high)} | - | - |")
        return lines

    sh = expr_manifest.get("shift_maxabs", {})
    ex = expr_manifest.get("expression", {})
    parts = [(out / "protocol.md").read_text().rstrip(), "\n---\n",
             "# Risultati: FaceVerse v2 con espressioni casuali, GT d'identita' neutra\n",
             f"Soggetti: {len(subjects)} (`select_subjects`, seed {args.eval_seed}, gli stessi dello zero-shot "
             f"FaceVerse neutro), 6 topologie, un'espressione casuale per mesh: pool {ex.get('pool_size')} blendshape "
             f"ARKit (esclusi {len(ex.get('excluded', []))} eyeLook*), {ex.get('n_active')} attivi, coefficienti "
             f"U{tuple(ex.get('coef_uniform', []))}. Spostamento maxabs medio {sh.get('mean', float('nan')):.4f} "
             f"(sd fra soggetti {sh.get('between_subject_sd', float('nan')):.4f}) su diametro "
             f"{sh.get('diameter_mean', float('nan')):.2f}; jawOpen a 1.0: {sh.get('jawOpen_1.00', {}).get('mean', float('nan')):.4f}. "
             f"Crop riestratti: {sh.get('crop_draws')}. CI 95% bootstrap per soggetto, {args.n_bootstrap} repliche.\n",
             f"Sorgenti dei modelli: " + "; ".join(f"{label(k)}: `{v}`" for k, v in sources.items()) + ".\n",
             "## PRIMARIO: riconoscimento d'identita', 5 topologie senza crop\n",
             "Retrieval: 2000 query (20 coppie ordinate di topologie x 100), galleria di 100 mesh in un'altra "
             "topologia; mAP = MRR (un solo rilevante). Verifica: 1000 coppie stessa persona, 99.000 persone diverse.\n",
             *rec_md("nocrop"),
             "\n### Delta appaiati modello - baseline (stesse repliche)\n", *delta_md("nocrop"),
             "\n## SECONDARIO: Spearman con la GT d'identita' neutra, mesh-pair senza crop\n", *sec_md("mesh_pair_nocrop"),
             "\n### Terziario: subject-pair-mean (senza crop)\n", *sec_md("subject_pair_mean_nocrop"),
             "\n## A parte: crop (coppie di topologie con crop da un lato)\n", *rec_md("crop"),
             "\n### Delta appaiati, crop\n", *delta_md("crop"),
             "\n## Controlli\n",
             f"- latent dagli embedding contro `latent_distance` delle pair_metrics, max |diff|: "
             + ", ".join(f"{label(k)} {v:.2e}" for k, v in checks.items()),
             (f"- regione stabile: frazione di vertici tenuti min {kept.min():.3f}, mediana {np.median(kept):.3f}, "
              f"max {kept.max():.3f}" if kept is not None else "- regione stabile: non calcolata"),
             "- NICP (asimmetrico): l'orientazione della coppia e' quella delle matrici i<j di alignment_matrix.py, "
             "non query -> galleria.",
             ]
    (out / "summary.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"[zs-expr-sum] scritto {out / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
