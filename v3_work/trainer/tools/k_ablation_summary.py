#!/usr/bin/env python3
"""Ablazione k_eig 64 contro 128 (robal, seme 1234) con le righe, le GT e le repliche di fact_paired.py: graduata
FR, SR e maxabs, IC bootstrap per soggetto, delta appaiati e regola di aau/runs/evidence/trainer_v3/ablations/
k_ablation/PROTOCOL_emendamento_1.md (k128 di base; k64 se non perde piu' di 0.03 in nessuna cella).

    v3_work/unified_gt/run.sh v3_work/trainer/tools/k_ablation_summary.py [--workers 32]

Embedding dei bracci robal_k<K> (v3_work/stream/ablation_k/eval_k*.sbatch, passi hifi, devfs, fv di
ablations/c3f/eval_body.sh, epoche 036 e 072): ``hifi_runs/<data>/scale_v3robal_k<K>e<E>_topology``, ``devfs/...``
(vista neutra), ``fv_expr/.../_flip_embed``; distanza ||z_i - z_j||. Righe, GT e seme per dominio da
``fact_paired.rows_for`` (HIFI3D e dev FaceScape ``nocrop_cross``, FaceVerse ``mesh_pair_nocrop``), maschera comune
con le colonne di fact_paired (bracci fattorizzati e baseline) piu' quelle di qui e GT maxabs, repliche come
``fact_paired.main`` (1.000, soggetti ricampionati, seme del dominio). Delta k64 - k128.
Tempo per passo dai ``train.log`` (media delle epoche 2-72). fact_paired.py importato, mai modificato.
Uscite in aau/runs/evidence/trainer_v3/ablations/k_ablation/: k_ablation_paired.csv, results.md.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

import fact_paired as fp  # noqa: E402  (sola lettura: righe, GT, baseline, colonne dei bracci)

be = fp.be
OUT = fp.EV / "ablations/k_ablation"
RUNS = fp.EV / "ablations/c3f_runs"
KS, EPOCHS, PRIMARY = (64, 128), ("036", "072"), "072"
GTS = ("fr", "sr", "maxabs")
PAIRS = ((64, 128),)
# (cartella delle eval, suffisso del tag) per dominio: le stesse viste e soggetti dei passi ``form`` (data_<hash>)
EMB = {"hifi3d": ("hifi_runs", "_topology"), "facescape": ("devfs", "_topology"), "faceverse": ("fv_expr", "_flip_embed")}
LOSS = -0.03                                      # PROTOCOL_emendamento_1.md: margine di non inferiorita'
_J: dict = {}


def embeddings(view: str, k: int, e: str):
    d, sfx = EMB[view]
    hits = sorted((fp.EVAL / d).glob(f"data_*/scale_v3robal_k{k}e{e}{sfx}/zs_zeroshot/embeddings.npz"))
    return hits[0] if hits else None


def k_columns(view: str, idx) -> dict:
    """{k<K>_e<E>: D (n, n) sulle chiavi di idx} per i bracci k presenti."""
    from scipy.spatial.distance import cdist
    out = {}
    for k in KS:
        for e in EPOCHS:
            p = embeddings(view, k, e)
            if p is None:
                continue
            with np.load(p, allow_pickle=True) as z:
                Z = np.asarray(z["Z"], np.float64)
                keys = list(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))
            pos = {kk: r for r, kk in enumerate(keys)}
            miss = [kk for kk in idx.keys if kk not in pos]
            if miss:
                raise SystemExit(f"{p}: {len(miss)} mesh delle righe assenti (prima {miss[0]})")
            Z = Z[[pos[kk] for kk in idx.keys]]
            out[f"k{k}_e{e}"] = cdist(Z, Z)
    return out


def _rep(r: int) -> dict:
    """Spearman di ogni metodo con FR, SR e maxabs su una replica (come fact_paired._rep, parte ``rho``)."""
    from scipy.stats import rankdata
    J = _J
    c = J["counts"][r]
    wt = c[J["sa"]].astype(np.int64) * c[J["sb"]]
    keep = wt > 0
    w = wt[keep]
    rg = {g: rankdata(np.repeat(J["cols"][f"gt_{g}"][keep], w)) for g in GTS}
    out = {}
    for m in J["methods"]:
        rm = rankdata(np.repeat(J["cols"][m][keep], w))
        for g in GTS:
            out[(g, m)] = fp._pearson(rm, rg[g])
    return out


def ci(x: np.ndarray) -> tuple[float, float]:
    b = x[1:][np.isfinite(x[1:])]
    return tuple(np.percentile(b, [2.5, 97.5])) if len(b) else (np.nan, np.nan)


def sec_per_step(k: int) -> float:
    """Media dei s/passo per epoca (2-72, l'ultima riga di ogni epoca: le riprese la riscrivono) da train.log."""
    p = RUNS / f"robal_k{k}" / "train.log"
    if not p.exists():
        return float("nan")
    ep = {}
    for m in re.finditer(r"\[v3\] epoca (\d+): .*\(([\d.]+) s/passo\)", p.read_text()):
        ep[int(m.group(1))] = float(m.group(2))
    v = [s for e, s in ep.items() if 2 <= e <= 72]
    return float(np.mean(v)) if len(v) == 71 else float("nan")


def rule(df: pd.DataFrame, sps: dict) -> list[str]:
    """PROTOCOL_emendamento_1.md sul checkpoint primario: k128 di base; k64 si adotta se in TUTTE le celle (HIFI3D,
    dev FaceScape, FaceVerse x FR, SR, maxabs) il delta k64 - k128 e' >= -0.03 (stima puntuale)."""
    d = df[(df["kind"] == "delta") & (df["arm"] == "k64-k128") & (df["epoch"] == PRIMARY)]
    if d.empty:
        return ["- delta k64 - k128 non calcolato"]
    bad = d[d["delta"] < LOSS]
    worst = d.loc[d["delta"].idxmin()]
    ok = bad.empty and len(d) == 9
    lines = [f"- celle valutate: {len(d)} di 9; peggiore: {worst['domain']} {worst['gt'].upper()} "
             f"{worst['delta']:+.3f} [{worst['ci_low']:+.3f}, {worst['ci_high']:+.3f}]",
             f"- celle sotto {LOSS}: " + (", ".join(f"{r.domain} {r.gt.upper()} {r.delta:+.3f}" for r in bad.itertuples())
                                          or "nessuna"),
             f"- velocita' (motivo dell'adozione, non criterio): {sps.get(64, np.nan):.3f} contro "
             f"{sps.get(128, np.nan):.3f} s/passo nel train.log",
             f"\nk adottato = **{64 if ok else 128}**" + ("" if ok else " (k64 perde piu' di 0.03 in almeno una cella)")]
    return lines


def main() -> None:
    global _J
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    recs, notes = [], []
    for view in ("hifi3d", "facescape", "faceverse"):
        df, idx, seed = fp.rows_for(view)
        Kc = k_columns(view, idx)
        if not Kc:
            notes.append(f"{view}: nessun braccio k")
            continue
        Mc = fp.model_columns(view, idx)               # solo per la maschera comune di fact_paired
        df = be.add_columns(df, {**Mc, **Kc}, idx)
        base = [b for b in fp.BASELINES if b in df]
        cols = {m: df[m].to_numpy(np.float64) for m in list(Mc) + base + list(Kc)}
        cols.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in GTS})
        subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
        s2i = {s: i for i, s in enumerate(subjects)}
        sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
        rng = np.random.default_rng(seed)
        counts = [np.ones(len(subjects), dtype=np.int64)] + \
            [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(a.n_boot)]
        fin = {k: np.isfinite(v) for k, v in cols.items()}
        mask_fp = (sa != sb) & np.all([fin[k] for k in list(Mc) + base + ["gt_fr", "gt_sr"]], axis=0)
        mask = mask_fp & np.all([fin[k] for k in list(Kc) + ["gt_maxabs"]], axis=0)
        n = int(mask.sum())
        if n != int(mask_fp.sum()):
            notes.append(f"{view}: {n} righe contro {int(mask_fp.sum())} della maschera di fact_paired")
        _J = {"counts": counts, "sa": sa[mask], "sb": sb[mask], "cols": {k: cols[k][mask] for k in list(Kc) + [
              f"gt_{g}" for g in GTS]}, "methods": list(Kc)}
        with mp.get_context("fork").Pool(a.workers) as pool:
            reps = pool.map(_rep, range(len(counts)), chunksize=4)
        V = {key: np.array([r[key] for r in reps]) for key in reps[0]}
        row = {"domain": view, "group": fp.VIEWS[view][1], "n_rows": n, "seed": int(seed)}
        for (g, m), x in V.items():
            lo, hi = ci(x)
            arm, e = m.split("_e")
            recs.append({**row, "gt": g, "epoch": e, "kind": "value", "arm": arm, "point": x[0], "ci_low": lo,
                         "ci_high": hi, "delta": np.nan, "p_le0": np.nan})
        for e in EPOCHS:
            for k1, k0 in PAIRS:
                m1, m0 = f"k{k1}_e{e}", f"k{k0}_e{e}"
                if m1 not in Kc or m0 not in Kc:
                    continue
                for g in GTS:
                    d = V[(g, m1)] - V[(g, m0)]
                    lo, hi = ci(d)
                    b = d[1:][np.isfinite(d[1:])]
                    recs.append({**row, "gt": g, "epoch": e, "kind": "delta", "arm": f"k{k1}-k{k0}", "point": np.nan,
                                 "delta": d[0], "ci_low": lo, "ci_high": hi, "p_le0": float((b <= 0).mean())})
        print(f"[k-abl] {view}: {sorted(Kc)}, {n} righe, seme {seed}", flush=True)
    df = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "k_ablation_paired.csv", index=False)
    sps = {k: sec_per_step(k) for k in KS}
    md = ["# Ablazione k_eig 64 contro 128: risultati (regola di PROTOCOL_emendamento_1.md)", "",
          "Generato da `v3_work/trainer/tools/k_ablation_summary.py`; numeri in `k_ablation_paired.csv`. Un solo seme "
          "(1234): gli IC coprono i soggetti di eval, non la variabilita' fra semi.", "",
          "## Spearman graduato (IC 95% bootstrap per soggetto)", "",
          "| dominio | gruppo | righe | GT | epoca | " + " | ".join(f"k{k}" for k in KS) + " |",
          "|---|---|---|---|---|" + "---|" * len(KS)]
    v = df[df["kind"] == "value"]
    for (dom, g, e), s in v.groupby(["domain", "gt", "epoch"], sort=False):
        cell = {r.arm: f"{r.point:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]" for r in s.itertuples()}
        md.append(f"| {dom} | {s['group'].iloc[0]} | {s['n_rows'].iloc[0]} | {g.upper()} | e{e} | "
                  + " | ".join(cell.get(f"k{k}", "-") for k in KS) + " |")
    md += ["", "## Delta appaiati (stesse repliche)", "", "| dominio | GT | epoca | confronto | delta | IC 95% | P(delta<=0) |",
           "|---|---|---|---|---|---|---|"]
    for r in df[df["kind"] == "delta"].itertuples():
        md.append(f"| {r.domain} | {r.gt.upper()} | e{r.epoch} | {r.arm} | {r.delta:+.3f} | "
                  f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.p_le0:.3f} |")
    md += ["", "## Tempo per passo (train.log, media delle epoche 2-72)", "",
           "| k | s/passo |", "|---|---|"] + [f"| {k} | {sps[k]:.3f} |" for k in KS]
    md += ["", f"## Regola (checkpoint e{PRIMARY})", ""] + rule(df, sps)
    if notes:
        md += ["", "## Note", ""] + [f"- {x}" for x in notes]
    (OUT / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(notes), flush=True)
    print(f"[k-abl] {len(df)} righe -> {OUT / 'k_ablation_paired.csv'}, {OUT / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
