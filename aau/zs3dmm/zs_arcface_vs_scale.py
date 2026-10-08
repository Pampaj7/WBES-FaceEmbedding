#!/usr/bin/env python3
"""ArcFace su render e il run grande (BFM+ICT+GNM, checkpoint intermedi) sullo STESSO protocollo, HIFI3D.

    aau/run.sh aau/zs3dmm/zs_arcface_vs_scale.py
    (zs_arcface_vs_scale.sbatch)

Due protocolli, pubblicati ciascuno senza l'altro braccio:
  (a) Spearman con la GT ``maxabs`` (``aau/runs/data_scale_ood/hifi/summary.md``, zs_summarize.py),
      ``nocrop_cross`` (mesh-pair, 20 coppie ordinate di topologie) e ``subject_pair_mean`` (clean,
      media sulle 30), sulle pair_metrics del breakdown. Qui entra ArcFace: 1 - coseno di
      ``zs_arcface_summarize.arcface_distances`` letto sulle STESSE righe (soggetto, topologia).
  (b) riconoscimento d'identita' (``aau/runs/arcface_render_zs/results_hifi3d.md``,
      zs_arcface_summarize.py), blocco ``nocrop``. Qui entrano i checkpoint, dagli
      ``embeddings.npz`` di ``zs_zeroshot.sbatch`` con WBES_ZS_PART=embed (stesso frame dei bracci
      ``scale_<tag>_topology`` di (a): nessuna rotazione, facce come sono).

Niente calcolo nuovo: distanze, bootstrap e misure sono le funzioni dei due summarizer, importate.
Repliche, nessun seme nuovo per le righe gia' pubblicate:
  - (a), righe singole: ogni riga gia' pubblicata col SUO seme (``stable_seed(seed, arm, gt, gruppo,
    metrica)`` di zs_summarize.boot) -> deve tornare identica; ArcFace col seme della Chamfer eval del
    congiunto, cioe' sulle stesse repliche della riga Chamfer di riferimento (0.372).
  - (a), differenze appaiate: tutte col seme di ``BFM+ICT+GNM (10^5), e108 - Chamfer eval`` (lo stesso
    di zs_summarize.pboot): ArcFace - e108, ArcFace - Chamfer ed e108 - Chamfer sulle stesse repliche,
    l'ultima e' il controllo.
  - (b): ``bootstrap_counts`` con ``stable_seed(seed, "expr_recognition")``, unico per tutte le righe
    come in zs_arcface_summarize.main.
  - breakdown sulle coppie con ``noisy``: seme per gruppo (``arcface_vs_scale``, gruppo), uguale per
    tutte le righe e le differenze del gruppo.

Scrive ``<out-dir>/*.csv`` e ``<out-md>``.
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

import zs_arcface_summarize as zas  # noqa: E402
import zs_expr_summarize as zes  # noqa: E402
import zs_summarize as zsum  # noqa: E402
from zs_stage import TOPOLOGIES, select_subjects  # noqa: E402

base = zsum.base
NOCROP = zes.NOCROP
TAGS = ("e036", "e072", "e108")
ARCFACE = ("arcface_shaded_3v", "arcface_normals_3v")
REFERENCE = zas.REFERENCE
# Confronti appaiati, uguali nei due protocolli ("chamfer": Chamfer eval in (a), faceBench in (b))
COMPARISONS = (("arcface_shaded_3v", "scale_e108"), ("arcface_shaded_3v", "chamfer"),
               ("arcface_normals_3v", "scale_e108"), ("arcface_normals_3v", "chamfer"),
               ("scale_e108", "chamfer"))
NOISY_PAIRS = tuple(t for t in NOCROP if t != "noisy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("aau/runs/ws_hifi3d/data_328f2bfc1a"))
    p.add_argument("--view-dir", type=Path, default=Path("datasets/HIFI3D/eval_view/npz"))
    p.add_argument("--arcface-root", type=Path, default=Path("aau/runs/arcface_render_zs/hifi3d"))
    p.add_argument("--joint-stage", type=Path,
                   default=Path("aau/runs/ws_hifi3d/data_328f2bfc1a/joint_frame-xmymz_flip_ranking/zs_zeroshot"),
                   help="congiunto in convenzione BFM, la riga di results_hifi3d.md")
    p.add_argument("--ref-a", type=Path, default=Path("aau/runs/data_scale_ood/hifi"),
                   help="table_cells.csv e paired.csv del summary (a), per il controllo")
    p.add_argument("--ref-b", type=Path, default=Path("aau/runs/arcface_render_zs/hifi3d/recognition.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("aau/runs/data_scale_ood/arcface_vs_scale_hifi3d"))
    p.add_argument("--out-md", type=Path, default=Path("aau/runs/data_scale_ood/arcface_vs_scale_hifi3d.md"))
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234, help="Seme del ricampionamento (quello dei due summary)")
    p.add_argument("--eval-seed", type=int, default=1234, help="WBES_EVAL_SEED: scelta dei soggetti")
    return p.parse_args()


def label(name: str) -> str:
    return f"BFM+ICT+GNM (10^5), {name[6:]}" if name.startswith("scale_") else zas.label(name)


def label_a(name: str) -> str:
    return "Chamfer eval" if name == "chamfer" else label(name)


# ---------------------------------------------------------------------------- (a) Spearman

def frame_a(args, idx: zes.Index, D_arc: dict) -> tuple[pd.DataFrame, dict]:
    """Le pair_metrics dei tre checkpoint e del congiunto, allineate per riga, + ArcFace sulle stesse righe."""
    pms = {"joint": base.read_pair_metrics(args.runs / "joint" / zsum.STAGE)}
    for t in TAGS:
        pms[f"scale_{t}"] = base.read_pair_metrics(args.runs / f"scale_{t}_topology" / zsum.STAGE)
    keys = zsum.PAIR_KEYS
    df = pms["scale_e108"][keys + ["gt_distance", "raw_chamfer"]].rename(columns={"raw_chamfer": "chamfer"})
    df["subject_a"], df["subject_b"] = df["subject_a"].astype(str), df["subject_b"].astype(str)
    checks = {}
    for name, pm in pms.items():
        pm = pm.assign(subject_a=pm["subject_a"].astype(str), subject_b=pm["subject_b"].astype(str))
        m = df[keys + ["gt_distance"]].merge(pm[keys + ["gt_distance", "latent_distance", "raw_chamfer"]], on=keys,
                                             how="left", validate="one_to_one", suffixes=("", "_arm"))
        if len(pm) != len(df) or m["latent_distance"].isna().any():
            raise SystemExit(f"{name}: righe delle pair_metrics diverse da quelle di scale_e108")
        if not np.array_equal(m["gt_distance"].to_numpy(), m["gt_distance_arm"].to_numpy()):
            raise SystemExit(f"{name}: GT diversa sulle stesse righe")
        df[name] = m["latent_distance"].to_numpy()
        if name == "joint":
            df["chamfer_joint"] = m["raw_chamfer"].to_numpy()
        if name != "scale_e108":
            checks[f"Chamfer eval delle pair_metrics di {name} contro quelle di scale_e108, max |diff|"] = \
                float(np.abs(m["raw_chamfer"] - df["chamfer"]).max())
    subj = sorted(set(df["subject_a"]) | set(df["subject_b"]))
    if subj != idx.subjects:
        raise SystemExit("pair_metrics: soggetti diversi da select_subjects")
    ia = np.asarray([idx.pos[k] for k in zip(df["subject_a"], df["topology_a"])])
    ib = np.asarray([idx.pos[k] for k in zip(df["subject_b"], df["topology_b"])])
    for name, D in D_arc.items():
        df[name] = D[ia, ib]
    return df, checks


def groups_a() -> list[tuple[str, callable]]:
    """(gruppo, filtro delle righe); i due scenari del summary (a) e il breakdown su noisy."""
    nocrop = lambda d: d[d["topology_a"].ne("crop") & d["topology_b"].ne("crop")]  # noqa: E731
    out = [("nocrop_cross", nocrop)]
    for t in NOISY_PAIRS:
        out.append((f"noisy|{t}", lambda d, t=t: d[((d["topology_a"] == "noisy") & (d["topology_b"] == t))
                                                    | ((d["topology_a"] == t) & (d["topology_b"] == "noisy"))]))
    out.append(("con noisy", lambda d: nocrop(d)[nocrop(d)[["topology_a", "topology_b"]].eq("noisy").any(axis=1)]))
    out.append(("senza noisy", lambda d: nocrop(d)[~nocrop(d)[["topology_a", "topology_b"]].eq("noisy").any(axis=1)]))
    return out


def tasks_a(df: pd.DataFrame, args) -> list:
    """Compiti per zsum._boot_task: (chiave, frame, colonna o coppia di colonne, n, seme)."""
    n, s = args.n_bootstrap, args.seed
    methods = list(ARCFACE) + [f"scale_{t}" for t in TAGS] + ["chamfer"]
    spm_cols = ["gt_distance"] + methods + ["joint", "chamfer_joint"]
    spm = df.groupby(["subject_a", "subject_b"], as_index=False)[spm_cols].mean()
    e108_name = f"{zsum.ARM_LABEL['scale']}, e108 - Chamfer eval"
    tasks = []
    for group, frame, tok in (("nocrop_cross", groups_a()[0][1](df), "nocrop_cross"),
                              ("subject_pair_mean", spm, "spm_clean")):
        # righe singole: i semi di zs_summarize.boot
        tasks.append(((group, "chamfer", "ref"), frame, "chamfer_joint",
                      n, base.stable_seed(s, "joint", "maxabs", tok, "chamfer")))
        for t in TAGS:
            tasks.append(((group, f"scale_{t}", "ref"), frame, f"scale_{t}",
                          n, base.stable_seed(s, f"scale_{t}", "maxabs", tok, "latent")))
        for a in ARCFACE:
            tasks.append(((group, a, "ref"), frame, a, n, base.stable_seed(s, "joint", "maxabs", tok, "chamfer")))
        # differenze: il seme di zs_summarize.pboot per e108 - Chamfer eval
        seed = base.stable_seed(s, "paired", "", "maxabs", e108_name, group)
        for a, b in COMPARISONS:
            tasks.append(((group, a, b), frame, (a, b), n, seed))
    for group, filt in groups_a()[1:]:
        frame = filt(df)
        seed = base.stable_seed(s, "arcface_vs_scale", "maxabs", group)
        for m in methods:
            tasks.append(((group, m, "ref"), frame, m, n, seed))
        for a, b in COMPARISONS:
            tasks.append(((group, a, b), frame, (a, b), n, seed))
    return tasks


# --------------------------------------------------------------------- (b) riconoscimento

def distances_b(args, idx: zes.Index, D_arc: dict) -> tuple[dict, dict]:
    D, checks = dict(D_arc), {}
    for t in TAGS:
        stage = args.runs / f"scale_{t}_embed" / zsum.STAGE
        if not (stage / ".done").exists():
            raise SystemExit(f"{stage}: embedding assenti (zs_zeroshot.sbatch, WBES_ZS_PART=embed)")
        staged = json.loads((stage.parent / "subjects.json").read_text())["subjects"]
        if sorted(staged) != idx.subjects:
            raise SystemExit(f"scale_{t}: zs_stage ha valutato soggetti diversi")
        # stesso checkpoint dei numeri di (a)
        key = lambda p: dict(l.split("=", 1) for l in p.read_text().splitlines() if "=" in l)  # noqa: E731
        ck_emb = key(stage.parent / "eval_key.txt")["ckpt"]
        ck_top = key(args.runs / f"scale_{t}_topology" / "eval_key.txt")["ckpt"]
        if os.path.realpath(ck_emb) != os.path.realpath(ck_top):
            raise SystemExit(f"scale_{t}: checkpoint degli embedding {ck_emb} != breakdown {ck_top}")
        D[f"scale_{t}"] = zes.model_distances(stage, idx)
        # ||z_i - z_j|| deve coincidere con latent_distance delle pair_metrics di (a), su tutte le 30 coppie
        pm = base.read_pair_metrics(args.runs / f"scale_{t}_topology" / zsum.STAGE)
        ia = np.asarray([idx.pos[k] for k in zip(pm["subject_a"].astype(str), pm["topology_a"])])
        ib = np.asarray([idx.pos[k] for k in zip(pm["subject_b"].astype(str), pm["topology_b"])])
        checks[f"scale_{t}: distanze dagli embedding contro latent_distance delle pair_metrics di (a) "
               f"({len(pm)} righe), max |diff|"] = float(np.abs(D[f"scale_{t}"][ia, ib] - pm["latent_distance"]).max())
    D["joint@bfm"] = zes.model_distances(args.joint_stage, idx)
    bl_root = args.runs / "baselines"
    for m in zes.BL_FACEBENCH:
        D[m] = zes.facebench_distances(bl_root, m, idx)
    return D, checks


def blocks_b() -> dict:
    """gruppo -> (coppie ordinate del retrieval, coppie non ordinate della verifica)."""
    out = {"nocrop": ([(a, b) for a in NOCROP for b in NOCROP if a != b],
                      [(a, b) for i, a in enumerate(NOCROP) for b in NOCROP[i + 1:]])}
    for t in NOISY_PAIRS:
        out[f"noisy|{t}"] = ([("noisy", t), (t, "noisy")],
                             [tuple(sorted(("noisy", t), key=NOCROP.index))])
    pr, pv = out["nocrop"]
    out["con noisy"] = ([p for p in pr if "noisy" in p], [p for p in pv if "noisy" in p])
    out["senza noisy"] = ([p for p in pr if "noisy" not in p], [p for p in pv if "noisy" not in p])
    return out


# --------------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    subjects = select_subjects(args.view_dir, args.eval_seed)
    idx = zes.Index(subjects)
    print(f"[arc-vs-scale] {len(subjects)} soggetti (primi {subjects[:3]})", flush=True)
    D_arc = {}
    for mode, name in (("shaded", "arcface_shaded_3v"), ("normals", "arcface_normals_3v")):
        D_arc[name] = zas.arcface_distances(args.arcface_root / mode / "arcface_views.npz", idx, zas.VIEWS["3v"])
    cal = {m: json.loads((args.arcface_root / m / "renders" / "arcface_align.json").read_text())
           for m in ("shaded", "normals")}
    same_cal = cal["shaded"]["views"] == cal["normals"]["views"]

    # (a) --------------------------------------------------------------------------------
    df, checks_a = frame_a(args, idx, D_arc)
    tasks = tasks_a(df, args)
    workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), 16)
    print(f"[arc-vs-scale] (a): {len(df)} righe, {len(tasks)} bootstrap su {workers} processi", flush=True)
    res = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for key, r in pool.imap_unordered(zsum._boot_task, tasks):
            res[key] = r
    rows_a, pair_a = [], []
    for (group, a, b), r in res.items():
        if b == "ref":
            rows_a.append({"group": group, "method": a, "point": r["spearman"], "ci_low": r["ci_low"],
                           "ci_high": r["ci_high"], "n_subjects": r["n_subjects"], "n_pairs": r["n_pairs"]})
        else:
            pair_a.append({"group": group, "a": a, "b": b, "a_point": r["a"], "b_point": r["b"], "point": r["diff"],
                           "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_le0": r["p_boot_le0"],
                           "n_pairs": r["n_pairs"], "n_bootstrap": r["n_bootstrap"]})
    rows_a, pair_a = pd.DataFrame(rows_a), pd.DataFrame(pair_a)

    # controllo (a): righe e differenza gia' pubblicate
    cells = pd.read_csv(args.ref_a / "table_cells.csv")
    paired_ref = pd.read_csv(args.ref_a / "paired.csv")
    ctrl_a = []
    for group, proto in (("nocrop_cross", "mesh_pair_nocrop_cross"), ("subject_pair_mean", "subject_pair_mean")):
        for name, arm, metric in [("chamfer", "joint", "chamfer")] + [(f"scale_{t}", f"scale_{t}", "latent")
                                                                      for t in TAGS]:
            ref = cells[(cells["model"] == arm) & (cells["gt"] == "maxabs") & (cells["protocol"] == proto)
                        & (cells["scenario"] == "clean")].iloc[0]
            new = rows_a[(rows_a["group"] == group) & (rows_a["method"] == name)].iloc[0]
            # subject_pair_mean: il punto pubblicato del congiunto viene dallo script di ranking, quello
            # dalle pair_metrics e' point_check (lo stesso del bootstrap); dei bracci scale coincidono.
            p_ref = ref[f"{metric}_point_check"] if np.isfinite(ref[f"{metric}_point_check"]) else ref[f"{metric}_spearman"]
            d = max(abs(new["point"] - p_ref), abs(new["ci_low"] - ref[f"{metric}_ci_low"]),
                    abs(new["ci_high"] - ref[f"{metric}_ci_high"]))
            ctrl_a.append({"riga": f"{label_a(name)}, {group}",
                           "pubblicato": zes.fmt(ref[f"{metric}_spearman"], ref[f"{metric}_ci_low"], ref[f"{metric}_ci_high"]),
                           "ricalcolato": zes.fmt(new["point"], new["ci_low"], new["ci_high"]), "max_abs_diff": d})
        ref = paired_ref[(paired_ref["gt"] == "maxabs") & (paired_ref["scenario"] == group)
                         & (paired_ref["comparison"] == f"{zsum.ARM_LABEL['scale']}, e108 - Chamfer eval")].iloc[0]
        new = pair_a[(pair_a["group"] == group) & (pair_a["a"] == "scale_e108") & (pair_a["b"] == "chamfer")].iloc[0]
        d = max(abs(new[c] - ref[c2]) for c, c2 in (("point", "diff"), ("ci_low", "ci_low"), ("ci_high", "ci_high"),
                                                     ("p_le0", "p_boot_le0")))
        ctrl_a.append({"riga": f"e108 - Chamfer eval (appaiata), {group}",
                       "pubblicato": zes.fmt(ref["diff"], ref["ci_low"], ref["ci_high"], True),
                       "ricalcolato": zes.fmt(new["point"], new["ci_low"], new["ci_high"], True), "max_abs_diff": d})
    ctrl_a = pd.DataFrame(ctrl_a)
    print(ctrl_a.to_string(index=False), flush=True)

    # (b) --------------------------------------------------------------------------------
    D, checks_b = distances_b(args, idx, D_arc)
    counts = zes.bootstrap_counts(len(subjects), args.n_bootstrap, base.stable_seed(args.seed, "expr_recognition"))
    blocks = blocks_b()
    methods_b = list(ARCFACE) + [f"scale_{t}" for t in TAGS] + ["joint@bfm"] + list(zes.BL_FACEBENCH)
    rec = {}
    with mp.get_context("fork").Pool(workers) as pool:
        tasks = [((blk, name), D[name], idx, pr, pv, counts) for blk, (pr, pv) in blocks.items()
                 for name in (methods_b if blk == "nocrop" else list(ARCFACE) + [f"scale_{t}" for t in TAGS] + ["chamfer"])]
        for key, vals, n_nan in pool.imap_unordered(zes._recog_task, tasks):
            rec[key] = (vals, n_nan)
    rows_b, pair_b = [], []
    for (blk, name), (vals, n_nan) in rec.items():
        r = {"block": blk, "method": name, "n_nan_distances": n_nan}
        for m in ("rank1", "map", "auc"):
            r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(vals[m][0]), zes.ci(vals[m])
        rows_b.append(r)
    for blk in blocks:
        for a, b in COMPARISONS:
            r = {"block": blk, "a": a, "b": b}
            for m in ("rank1", "map", "auc"):
                d = rec[(blk, a)][0][m] - rec[(blk, b)][0][m]
                r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(d[0]), zes.ci(d)
                r[f"{m}_p_le0"] = float((d[1:] <= 0).mean())
            pair_b.append(r)
    rows_b, pair_b = pd.DataFrame(rows_b), pd.DataFrame(pair_b)

    # controllo (b): righe di results_hifi3d.md (nocrop)
    ref = pd.read_csv(args.ref_b).set_index(["block", "method"])
    cols = [f"{m}{s}" for m in ("rank1", "map", "auc") for s in ("", "_ci_low", "_ci_high")]
    ctrl_b = []
    for r in rows_b[rows_b["block"] == "nocrop"].itertuples():
        if ("nocrop", r.method) in ref.index:
            x = ref.loc[("nocrop", r.method)]
            ctrl_b.append({"riga": label(r.method), "pubblicato rank-1": zes.fmt(x["rank1"], x["rank1_ci_low"], x["rank1_ci_high"]),
                           "ricalcolato rank-1": zes.fmt(r.rank1, r.rank1_ci_low, r.rank1_ci_high),
                           "max_abs_diff (rank-1, mAP, AUC, CI)": max(abs(getattr(r, c) - x[c]) for c in cols)})
    ctrl_b = pd.DataFrame(ctrl_b)
    print(ctrl_b.to_string(index=False), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_a.to_csv(args.out_dir / "spearman.csv", index=False)
    pair_a.to_csv(args.out_dir / "spearman_paired.csv", index=False)
    rows_b.to_csv(args.out_dir / "recognition.csv", index=False)
    pair_b.to_csv(args.out_dir / "recognition_paired.csv", index=False)
    ctrl_a.to_csv(args.out_dir / "control_a.csv", index=False)
    ctrl_b.to_csv(args.out_dir / "control_b.csv", index=False)
    write_md(args, subjects, rows_a, pair_a, rows_b, pair_b, ctrl_a, ctrl_b, {**checks_a, **checks_b}, same_cal, blocks)
    print(f"[arc-vs-scale] scritto {args.out_md}", flush=True)


def write_md(args, subjects, rows_a, pair_a, rows_b, pair_b, ctrl_a, ctrl_b, checks, same_cal, blocks) -> None:
    order = list(ARCFACE) + [f"scale_{t}" for t in TAGS] + ["chamfer"]

    def a_cell(group, m):
        r = rows_a[(rows_a["group"] == group) & (rows_a["method"] == m)]
        return "-" if r.empty else zes.fmt(r.iloc[0]["point"], r.iloc[0]["ci_low"], r.iloc[0]["ci_high"])

    def a_delta(group, a, b):
        r = pair_a[(pair_a["group"] == group) & (pair_a["a"] == a) & (pair_a["b"] == b)].iloc[0]
        return f"{zes.fmt(r['point'], r['ci_low'], r['ci_high'], True)} ({r['p_le0']:.3f})"

    def b_row(blk, m, metrics=("rank1", "map", "auc")):
        r = rows_b[(rows_b["block"] == blk) & (rows_b["method"] == m)].iloc[0]
        return [zes.fmt(r[x], r[f"{x}_ci_low"], r[f"{x}_ci_high"]) for x in metrics]

    def b_delta(blk, a, b, metrics=("rank1", "map", "auc")):
        r = pair_b[(pair_b["block"] == blk) & (pair_b["a"] == a) & (pair_b["b"] == b)].iloc[0]
        return [f"{zes.fmt(r[x], r[f'{x}_ci_low'], r[f'{x}_ci_high'], True)} ({r[f'{x}_p_le0']:.3f})" for x in metrics]

    def table(header, rows):
        return ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]

    n_q = len(blocks["nocrop"][0]) * len(subjects)
    parts = [
        "# ArcFace su render contro il run grande (BFM+ICT+GNM 10^5, e036/e072/e108): HIFI3D, stesso protocollo\n",
        f"Soggetti: {len(subjects)} (`select_subjects`, seed {args.eval_seed}; primi {', '.join(subjects[:3])}), gli stessi "
        "dei due summary. CI 95% bootstrap per soggetto, "
        f"{args.n_bootstrap} repliche; P(<=0) = frazione di repliche con differenza <= 0. Script: "
        "`aau/zs3dmm/zs_arcface_vs_scale.py` (importa zs_summarize, zs_expr_summarize, zs_arcface_summarize: "
        f"nessuna misura riscritta). CSV in `{args.out_dir}`.\n",
        "ArcFace = 1 - coseno della media degli embedding delle 3 viste (yaw 0, +-30), rinormalizzata, dagli "
        f"`arcface_views.npz` gia' calcolati in `{args.arcface_root}` (ombreggiato = riga di riferimento di "
        "results_hifi3d.md; normal map = stesso crop, calcolata dopo quel summary e qui alla prima apparizione). "
        "Convenzione di results_hifi3d.md invariata: crop fisso ricalibrato (similarita' congelata dalle mediane per "
        "yaw dei landmark) applicato a TUTTI i render, detector solo per la calibrazione; su `noisy` il detector "
        f"fallisce 300/300 e il crop fisso vale come sulle altre. Calibrazione normal map = ombreggiato: {same_cal}.\n",
        "Checkpoint: `epoch036/072/108.pth` del run 1060130 (gli stessi di `scale_eNNN_topology`, controllato dagli "
        "`eval_key.txt`), frame nativo HIFI3D come in (a) (nessuna rotazione, facce come sono). Per (b) gli embedding "
        f"vengono da `{args.runs}/scale_eNNN_embed` (zs_zeroshot.sbatch, WBES_ZS_PART=embed).\n",
        "## (a) Spearman con la GT `maxabs` (protocollo di `aau/runs/data_scale_ood/hifi/summary.md`)\n",
        "Righe: le pair_metrics del breakdown (coppie di soggetti diversi, coppie ordinate di topologie, clean); "
        "ArcFace letto sulle stesse righe. `chamfer` = Chamfer eval degli script di eval (la colonna del summary (a)), "
        "non la Chamfer faceBench. Righe singole: Chamfer ed e0NN col loro seme del summary (riprodotte, vedi Controlli), "
        "ArcFace col seme della riga Chamfer (stesse repliche del riferimento 0.372).\n",
        *table(["metodo", "nocrop_cross (20 coppie ordinate)", "subject_pair_mean (clean, 30)"],
               [[label_a(m), a_cell("nocrop_cross", m), a_cell("subject_pair_mean", m)] for m in order]),
        "\n### (a) Differenze appaiate\n",
        "Tutte sulle STESSE repliche: il seme della differenza `e108 - Chamfer eval` del summary (a), che qui torna "
        "identica (ultima riga). Chamfer eval presa dalle pair_metrics di e108, come nel summary.\n",
        *table(["A - B", "nocrop_cross [CI] (P<=0)", "subject_pair_mean [CI] (P<=0)"],
               [[f"{label_a(a)} - {label_a(b)}", a_delta("nocrop_cross", a, b), a_delta("subject_pair_mean", a, b)]
                for a, b in COMPARISONS]),
        "\n## (b) Riconoscimento d'identita' (protocollo di `aau/runs/arcface_render_zs/results_hifi3d.md`)\n",
        f"5 topologie senza crop: {n_q} query di retrieval (galleria di {len(subjects)} mesh in un'altra topologia; "
        "mAP = MRR), verifica su 1000 coppie stessa persona e 99000 diverse. Repliche: `bootstrap_counts` col seme "
        "`expr_recognition`, le stesse per tutte le righe e le differenze (come nel summary (b)). `chamfer` qui = "
        "Chamfer faceBench 4096 pt (la riga di results_hifi3d.md). Il congiunto e' in convenzione BFM come in (b); "
        "i checkpoint su scala nel frame nativo di (a).\n",
        *table(["metodo", "rank-1", "mAP", "AUC verifica", "distanze NaN"],
               [[label(m), *b_row("nocrop", m),
                 str(int(rows_b[(rows_b['block'] == 'nocrop') & (rows_b['method'] == m)].iloc[0]['n_nan_distances']))]
                for m in list(ARCFACE) + [f"scale_{t}" for t in TAGS] + ["chamfer", "joint@bfm", "rigid_icp_chamfer",
                                                                          "nicp_p2p", "nicp_p2tri"]]),
        "\n### (b) Differenze appaiate (stesse repliche)\n",
        *table(["A - B", "rank-1 [CI] (P<=0)", "mAP [CI] (P<=0)", "AUC [CI] (P<=0)"],
               [[f"{label(a)} - {label(b)}", *b_delta("nocrop", a, b)] for a, b in COMPARISONS]),
        "\n## Breakdown sulle coppie di topologie con `noisy` (senza crop)\n",
        "Per coppia NON ordinata {noisy, X}: in (a) le due coppie ordinate insieme, in (b) retrieval nei due versi "
        "(200 query) e verifica su quella coppia. `con noisy` = le 4 coppie insieme, `senza noisy` = le altre 6 "
        "(complemento nel protocollo senza crop). Repliche: in (a) un seme per gruppo, uguale per righe e differenze "
        "del gruppo; in (b) le stesse di sopra.\n",
        "### (a) Spearman, GT maxabs\n",
        *table(["coppia"] + [label_a(m) for m in order] + ["ArcFace ombr. - e108", "ArcFace ombr. - Chamfer eval"],
               [[g.replace("|", " / ")] + [a_cell(g, m) for m in order] + [a_delta(g, "arcface_shaded_3v", "scale_e108"),
                                                      a_delta(g, "arcface_shaded_3v", "chamfer")]
                for g in [f"noisy|{t}" for t in NOISY_PAIRS] + ["con noisy", "senza noisy"]]),
        "\n### (b) rank-1\n",
        *table(["coppia"] + [label(m) for m in order] + ["ArcFace ombr. - e108", "ArcFace ombr. - Chamfer"],
               [[g.replace("|", " / ")] + [b_row(g, m, ("rank1",))[0] for m in order]
                + [b_delta(g, "arcface_shaded_3v", "scale_e108", ("rank1",))[0],
                   b_delta(g, "arcface_shaded_3v", "chamfer", ("rank1",))[0]]
                for g in [f"noisy|{t}" for t in NOISY_PAIRS] + ["con noisy", "senza noisy"]]),
        "\n### (b) AUC verifica\n",
        *table(["coppia"] + [label(m) for m in order] + ["ArcFace ombr. - e108", "ArcFace ombr. - Chamfer"],
               [[g.replace("|", " / ")] + [b_row(g, m, ("auc",))[0] for m in order]
                + [b_delta(g, "arcface_shaded_3v", "scale_e108", ("auc",))[0],
                   b_delta(g, "arcface_shaded_3v", "chamfer", ("auc",))[0]]
                for g in [f"noisy|{t}" for t in NOISY_PAIRS] + ["con noisy", "senza noisy"]]),
        "\n## Controlli\n",
        "Riproduzione del summary (a) (`table_cells.csv`, `paired.csv`; max |diff| su punto e CI, e P per la differenza; "
        "per subject_pair_mean il punto si confronta con `point_check`, cioe' lo Spearman sulle pair_metrics mediate: "
        "il punto pubblicato del congiunto viene dallo script di ranking e ne differisce di ~1e-6):\n",
        *table(["riga", "pubblicato", "ricalcolato", "max |diff|"],
               [[r["riga"], r["pubblicato"], r["ricalcolato"], f"{r['max_abs_diff']:.2e}"] for _, r in ctrl_a.iterrows()]),
        f"\nRiproduzione di results_hifi3d.md (`{args.ref_b}`, blocco nocrop; max |diff| su punto e CI di rank-1, mAP, AUC):\n",
        *table(["riga", "pubblicato rank-1", "ricalcolato rank-1", "max |diff|"],
               [[r["riga"], r["pubblicato rank-1"], r["ricalcolato rank-1"],
                 f"{r['max_abs_diff (rank-1, mAP, AUC, CI)']:.2e}"] for _, r in ctrl_b.iterrows()]),
        "\nAltri:\n",
        *[f"- {k}: {v:.2e}" for k, v in checks.items()],
        "- soggetti: pair_metrics, `subjects.json` degli embedding e `select_subjects` coincidono (altrimenti lo script esce)",
        "- embedding ArcFace: tutti finiti e L2 (|norma - 1| <= 1e-3, controllo di `arcface_distances`)",
    ]
    args.out_md.write_text("\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
