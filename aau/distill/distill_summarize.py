#!/usr/bin/env python3
"""Riconoscimento d'identita' dello studente distillato, con le misure e le repliche del gate ArcFace.

    aau/run.sh aau/distill/distill_summarize.py --domain fv_expr
    (distill_summarize.sbatch; protocollo in aau/runs/distill_pilot/protocol.md, scritto prima delle eval)

Tutto il calcolo viene da ``aau/zs3dmm/zs_expr_summarize.py`` e ``zs_arcface_summarize.py``, importati
e non riscritti: ``Index``, ``_recog_task`` (retrieval, verifica, repliche), ``bootstrap_counts`` con
lo STESSO seme del gate (``stable_seed(1234, "expr_recognition")``), ``model_distances``,
``facebench_distances``, ``arcface_distances``. Le righe che il gate ha gia' (insegnante, NICP, ICP,
Chamfer, congiunto) devono quindi coincidere con il suo ``recognition.csv``: e' il controllo,
riportato in fondo.

Righe: studente in convenzione BFM (riferimento, stesso ingresso della riga del congiunto) e in
convenzione ICT (secondaria); insegnante ArcFace 3 viste ombreggiato e normal map; NICP P2Tri; ICP
rigido + Chamfer; Chamfer faceBench; congiunto BFM+ICT (convenzione BFM). Delta APPAIATI studente -
ciascuna, e il rapporto rank-1 studente / insegnante normal map con il suo CI (stesse repliche).

Scrive ``<out-dir>/<domain>/recognition{,_paired}.csv``, ``<out-dir>/results_<domain>.md`` e ricompone
``<out-dir>/summary.md`` = protocol.md + i results_*.md presenti + giudizio.md (se c'e').
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

import zs_arcface_summarize as zas  # noqa: E402
import zs_expr_summarize as zes  # noqa: E402
from zs_stage import TOPOLOGIES, select_subjects  # noqa: E402

base = zes.base
DOMAINS = {
    # dominio: (vista, baseline faceBench, stage del congiunto in convenzione BFM, recognition.csv del gate)
    "fv_expr": ("datasets/FACEVERSE_ZS/expr_view/npz", "aau/runs/ws_faceverse_expr/data_736f96956a",
                "joint_flip_topology/zs_zeroshot"),
    "hifi3d": ("datasets/HIFI3D/eval_view/npz", "aau/runs/ws_hifi3d/data_328f2bfc1a",
               "joint_frame-xmymz_flip_ranking/zs_zeroshot"),
}
REFERENCE = "student@bfm"
BASELINES = ("arcface_normals_3v", "arcface_shaded_3v", "nicp_p2tri", "rigid_icp_chamfer", "chamfer", "joint@bfm")
SECONDARY = (("student@ict", "joint@bfm"), ("student@ict", "student@bfm"), ("student@ict", "arcface_normals_3v"))
TEACHER = "arcface_normals_3v"
LABELS = {"student@bfm": "Studente distillato, convenzione BFM (riferimento)",
          "student@ict": "Studente distillato, convenzione ICT"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", choices=sorted(DOMAINS), required=True)
    p.add_argument("--student-root", type=Path, default=None, help="default <out-dir>/eval/<domain>")
    p.add_argument("--arcface-root", type=Path, default=None, help="default aau/runs/arcface_render_zs/<domain>")
    p.add_argument("--out-dir", type=Path, default=Path("aau/runs/distill_pilot"))
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eval-seed", type=int, default=1234)
    return p.parse_args()


def label(name: str) -> str:
    return LABELS.get(name) or zas.label(name).replace(" (riferimento)", "")


def main() -> None:
    args = parse_args()
    view_dir, runs, joint_rel = (Path(x) for x in DOMAINS[args.domain])
    student_root = args.student_root or args.out_dir / "eval" / args.domain
    arc_root = args.arcface_root or Path("aau/runs/arcface_render_zs") / args.domain
    subjects = select_subjects(view_dir, args.eval_seed)
    idx = zes.Index(subjects)
    print(f"[distill-sum] {args.domain}: {len(subjects)} soggetti (primi {subjects[:3]})", flush=True)

    D = {}
    for conv in ("bfm", "ict"):
        stage = student_root / conv
        if not (stage / "embeddings.npz").exists():
            print(f"[distill-sum] ATTENZIONE: {stage}/embeddings.npz assente, riga student@{conv} saltata", flush=True)
            continue
        if sorted(json.loads((stage / "subjects.json").read_text())["subjects"]) != subjects:
            raise SystemExit(f"{stage}: zs_stage ha valutato soggetti diversi")
        D[f"student@{conv}"] = zes.model_distances(stage, idx)
    if REFERENCE not in D:
        raise SystemExit(f"riga di riferimento {REFERENCE} assente")
    for mode in ("normals", "shaded"):
        D[f"arcface_{mode}_3v"] = zas.arcface_distances(arc_root / mode / "arcface_views.npz", idx, zas.VIEWS["3v"])
    D["joint@bfm"] = zes.model_distances(runs / joint_rel, idx)
    for m in ("nicp_p2tri", "rigid_icp_chamfer", "chamfer"):
        D[m] = zes.facebench_distances(runs / "baselines", m, idx)
    print(f"[distill-sum] metodi {list(D)}", flush=True)

    counts = zes.bootstrap_counts(len(subjects), args.n_bootstrap, base.stable_seed(args.seed, "expr_recognition"))
    blocks = {
        "nocrop": ([(a, b) for a in zes.NOCROP for b in zes.NOCROP if a != b],
                   [(a, b) for i, a in enumerate(zes.NOCROP) for b in zes.NOCROP[i + 1:]]),
        "crop": ([(a, b) for a in TOPOLOGIES for b in TOPOLOGIES if a != b and "crop" in (a, b)],
                 [("crop", b) for b in zes.NOCROP]),
    }
    rec = {}
    with mp.get_context("fork").Pool(min(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), 16)) as pool:
        tasks = [((blk, name), D[name], idx, pr, pv, counts) for blk, (pr, pv) in blocks.items() for name in D]
        for key, vals, n_nan in pool.imap_unordered(zes._recog_task, tasks):
            rec[key] = (vals, n_nan)
    rows = []
    for (blk, name), (vals, n_nan) in rec.items():
        r = {"block": blk, "method": name, "n_nan_distances": n_nan}
        for m in ("rank1", "map", "auc"):
            r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(vals[m][0]), zes.ci(vals[m])
        rows.append(r)
    rec_table = pd.DataFrame(rows)
    comparisons = [(REFERENCE, b) for b in BASELINES] + [c for c in SECONDARY if c[0] in D]
    deltas = []
    for blk in blocks:
        for a_name, b_name in comparisons:
            a, b = rec[(blk, a_name)][0], rec[(blk, b_name)][0]
            r = {"block": blk, "model": a_name, "baseline": b_name}
            for m in ("rank1", "map", "auc"):
                d = a[m] - b[m]
                r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(d[0]), zes.ci(d)
                r[f"{m}_p_le0"] = float((d[1:] <= 0).mean())
            deltas.append(r)
    delta_table = pd.DataFrame(deltas)
    # Criterio: rank-1 studente / insegnante normal map, punto e CI sulle stesse repliche
    ratios = {}
    for blk in blocks:
        for s_name in [n for n in D if n.startswith("student@")]:
            rr = rec[(blk, s_name)][0]["rank1"] / rec[(blk, TEACHER)][0]["rank1"]
            ratios[(blk, s_name)] = (float(rr[0]), *zes.ci(rr))

    out = args.out_dir / args.domain
    out.mkdir(parents=True, exist_ok=True)
    rec_table.to_csv(out / "recognition.csv", index=False)
    delta_table.to_csv(out / "recognition_paired.csv", index=False)
    pd.DataFrame([{"block": b, "model": s, "ratio_rank1_vs_teacher_normals": v[0], "ci_low": v[1], "ci_high": v[2]}
                  for (b, s), v in ratios.items()]).to_csv(out / "ratio_teacher.csv", index=False)

    order = [n for n in D if n.startswith("student@")] + [n for n in BASELINES if n in D]

    def rec_md(blk: str) -> list[str]:
        lines = ["| metodo | rank-1 | mAP | AUC verifica | distanze NaN |", "| --- | --- | --- | --- | --- |"]
        sub = rec_table[rec_table["block"] == blk].set_index("method")
        for name in order:
            r = sub.loc[name]
            lines.append(f"| {label(name)} | " + " | ".join(zes.fmt(r[m], r[f"{m}_ci_low"], r[f"{m}_ci_high"])
                                                           for m in ("rank1", "map", "auc"))
                         + f" | {r['n_nan_distances']} |")
        return lines

    def delta_md(blk: str, pairs) -> list[str]:
        lines = ["| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |",
                 "| --- | --- | --- | --- | --- |"]
        sub = delta_table[delta_table["block"] == blk]
        for a_name, b_name in pairs:
            r = sub[(sub["model"] == a_name) & (sub["baseline"] == b_name)].iloc[0]
            lines.append(f"| {label(a_name)} | {label(b_name)} | "
                         + " | ".join(f"{zes.fmt(r[m], r[m + '_ci_low'], r[m + '_ci_high'], True)} ({r[m + '_p_le0']:.3f})"
                                      for m in ("rank1", "map", "auc")) + " |")
        return lines

    def ratio_md(blk: str) -> list[str]:
        return [f"- {label(s)}: rank-1 / insegnante normal map = {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
                for (b, s), v in ratios.items() if b == blk]

    # Controllo: le righe gia' nel gate devono coincidere con il suo recognition.csv
    ref = pd.read_csv(Path("aau/runs/arcface_render_zs") / args.domain / "recognition.csv").set_index(["block", "method"])
    cols = [f"{m}{s}" for m in ("rank1", "map", "auc") for s in ("", "_ci_low", "_ci_high")]
    diffs = {(r["block"], r["method"]): max(abs(r[c] - ref.loc[(r["block"], r["method"]), c]) for c in cols)
             for r in rows if (r["block"], r["method"]) in ref.index}
    check = (f"- riproduzione di `aau/runs/arcface_render_zs/{args.domain}/recognition.csv` (stesse repliche): "
             f"max |diff| su punto e CI = {max(diffs.values()):.2e} su {len(diffs)} righe "
             f"({', '.join(sorted({m for _, m in diffs}))})" if diffs else "- nessuna riga in comune col gate")
    missing = sorted({m for m in D if not m.startswith("student@")} - {m for _, m in diffs})
    if missing:
        check += f"; righe non presenti nel csv del gate (calcolate qui): {', '.join(missing)}"
    primary = [(REFERENCE, b) for b in BASELINES]
    secondary = [c for c in SECONDARY if c[0] in D]
    parts = [f"# Risultati: {zas.DOMAIN_LABEL[args.domain]}\n",
             f"Soggetti: {len(subjects)} (`select_subjects`, seed {args.eval_seed}), mesh da `{view_dir}`; studente "
             f"da `{student_root}`, insegnante da `{arc_root}`, baseline da `{runs / 'baselines'}`, congiunto da "
             f"`{runs / joint_rel}`. CI 95% bootstrap per soggetto, {args.n_bootstrap} repliche (le stesse del gate).\n",
             "## PRIMARIO: riconoscimento d'identita', 5 topologie senza crop\n", *rec_md("nocrop"),
             "\n### Delta appaiati, studente (convenzione BFM) - baseline\n", *delta_md("nocrop", primary),
             "\n### Criterio: rapporto con l'insegnante a normal map\n", *ratio_md("nocrop"),
             "\n### Secondari\n", *delta_md("nocrop", secondary),
             "\n## A parte: crop (coppie di topologie con crop da un lato)\n", *rec_md("crop"),
             "\n### Delta appaiati, crop\n", *delta_md("crop", primary + secondary), "", *ratio_md("crop"),
             "\n## Controlli\n", check]
    (args.out_dir / f"results_{args.domain}.md").write_text("\n".join(parts) + "\n", encoding="utf-8")

    sections = [(args.out_dir / "protocol.md").read_text().rstrip()]
    for name in ("results_fv_expr.md", "results_hifi3d.md", "giudizio.md"):
        if (args.out_dir / name).exists():
            sections += ["\n---\n", (args.out_dir / name).read_text().rstrip()]
    (args.out_dir / "summary.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"[distill-sum] scritto {args.out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
