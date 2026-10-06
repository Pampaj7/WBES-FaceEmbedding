#!/usr/bin/env python3
"""Riconoscimento d'identita' di ArcFace su render di sola geometria, con le misure di zs_expr_summarize.py.

    aau/run.sh aau/zs3dmm/zs_arcface_summarize.py --domain fv_expr \\
        --runs aau/runs/ws_faceverse_expr/data_736f96956a --view-dir datasets/FACEVERSE_ZS/expr_view/npz \\
        --joint-stage aau/runs/ws_faceverse_expr/data_736f96956a/joint_flip_topology/zs_zeroshot \\
        --reference-csv aau/runs/ws_faceverse_expr/recognition.csv
    (zs_arcface_summarize.sbatch)

Protocollo: ``aau/runs/arcface_render_zs/protocol.md``, dichiarato prima dei render e copiato in
testa al summary. Tutto il calcolo viene da ``zs_expr_summarize.py``, importato e non riscritto:
``Index``, ``retrieval_queries``, ``verification_pairs``, ``recognition_values`` (via
``_recog_task``), ``bootstrap_counts`` con lo STESSO seme (``stable_seed(seed, "expr_recognition")``),
``facebench_distances``, ``region_distances``, ``model_distances``. Le righe delle baseline e del
congiunto devono quindi coincidere con quelle di ``--reference-csv`` (il ``recognition.csv`` del
summary esistente): e' il controllo, riportato in fondo.

Distanze ArcFace, come in ws3a_perceptual.py: embedding per vista (gia' L2), media sulle viste
scelte, rinormalizzazione, 1 - coseno. Righe: ombreggiato e normal map (se c'e'), 3 viste e 1 vista
(yaw 0). Riferimento: ombreggiato, 3 viste.

Scrive ``<summary-dir>/<domain>/recognition{,_paired}.csv`` e ``<summary-dir>/results_<domain>.md``,
poi ricompone ``<summary-dir>/summary.md`` = protocol.md + i results_*.md presenti + giudizio.md
(se c'e').
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

import zs_expr_summarize as zes  # noqa: E402
from zs_stage import TOPOLOGIES, select_subjects  # noqa: E402

base = zes.base
MODES = {"shaded": "ombreggiato", "normals": "normal map"}
VIEWS = {"3v": (0.0, -30.0, 30.0), "1v": (0.0,)}
REFERENCE = "arcface_shaded_3v"
# Confronti appaiati del primario (fissati nel protocollo) e delle ablazioni.
PRIMARY_BASELINES = ("nicp_p2tri", "rigid_icp_chamfer", "chamfer", "joint@bfm")
ABLATIONS = (("arcface_shaded_3v", "arcface_shaded_1v"), ("arcface_normals_3v", "arcface_shaded_3v"),
             ("arcface_normals_1v", "arcface_shaded_1v"), ("arcface_normals_3v", "arcface_normals_1v"))
DOMAIN_LABEL = {"fv_expr": "FaceVerse v2 con espressioni casuali", "hifi3d": "HIFI3D, neutre (senza espressioni)"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", choices=sorted(DOMAIN_LABEL), required=True)
    p.add_argument("--runs", type=Path, required=True, help="data_<fp> del dominio (baseline faceBench)")
    p.add_argument("--view-dir", type=Path, required=True, help="<vista>/npz, per i soggetti")
    p.add_argument("--arcface-root", type=Path, default=None, help="default <summary-dir>/<domain>")
    p.add_argument("--joint-stage", type=Path, default=None, help="stage con embeddings.npz del congiunto (BFM)")
    p.add_argument("--reference-csv", type=Path, default=None, help="recognition.csv esistente, per il controllo")
    p.add_argument("--summary-dir", type=Path, default=Path("aau/runs/arcface_render_zs"))
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234, help="Seme del ricampionamento (lo stesso del summary esistente)")
    p.add_argument("--eval-seed", type=int, default=1234, help="WBES_EVAL_SEED: scelta dei soggetti")
    return p.parse_args()


def arcface_distances(path: Path, idx: zes.Index, yaws) -> np.ndarray:
    """1 - coseno della media sulle viste ``yaws`` degli embedding per vista, rinormalizzata."""
    z = np.load(path)
    keys = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))
    if sorted(keys) != sorted(idx.keys):
        raise SystemExit(f"{path}: embedding di mesh diverse da quelle attese")
    # Il job che li ha scritti ha avuto oom-kill (1057301): ogni embedding per vista deve esserci, L2.
    norms = np.linalg.norm(z["E"].astype(np.float64), axis=2)
    if not np.isfinite(norms).all() or np.abs(norms - 1).max() > 1e-3:
        raise SystemExit(f"{path}: embedding non finiti o non normalizzati (|norma - 1| max {np.abs(norms - 1).max():.2e})")
    cols = [list(z["yaws"].astype(float)).index(float(y)) for y in yaws]
    E = z["E"].astype(np.float64)[:, cols].mean(axis=1)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    Z = np.zeros((len(idx.keys), E.shape[1]))
    for k, row in zip(keys, E):
        Z[idx.pos[k]] = row
    return 1.0 - Z @ Z.T


def label(name: str) -> str:
    if name.startswith("arcface_"):
        _, mode, views = name.split("_")
        return f"ArcFace, {MODES[mode]}, {views[0]} vist{'e' if views == '3v' else 'a'}" + \
            (" (riferimento)" if name == REFERENCE else "")
    if name == "joint@bfm":
        return "BFM+ICT, convenzione BFM"
    return zes.BL_LABEL.get(name, name)


def calibration_md(root: Path) -> list[str]:
    path = root / "shaded" / "renders" / "arcface_align.json"
    data = json.loads(path.read_text())
    det = data["detector"]
    lines = [f"Crop fisso ricalibrato su questi render (`{path}`): detector su {det['total']['n']} render, "
             f"fallito su {det['total']['n_failed']}. Per topologia (falliti/render): "
             + ", ".join(f"{t} {v['n_failed']}/{v['n']}" for t, v in sorted(det.items()) if t != "total") + ".",
             "Landmark mediani per vista (5 punti, px su 512) e IQR massimo fra i 5 punti:"]
    for key, v in sorted(data["views"].items()):
        lines.append(f"- yaw {key}: {v['n_detected']} detection, IQR max {np.max(v['kps_iqr_px']):.1f} px, "
                     f"kps {np.round(v['kps_median'], 1).tolist()}")
    cam = json.loads((root / "shaded" / "renders" / "camera.json").read_text())
    lines.append(f"Camera unica: base_rotation `{cam['base_rotation']}`, center "
                 f"{np.round(cam['center'], 4).tolist()}, scale {cam['scale']:.4f}, su {cam['n_meshes']} mesh.")
    return lines


def main() -> None:
    args = parse_args()
    root = args.arcface_root or args.summary_dir / args.domain
    subjects = select_subjects(args.view_dir, args.eval_seed)
    idx = zes.Index(subjects)
    print(f"[arcface-sum] {args.domain}: {len(subjects)} soggetti (primi {subjects[:3]})", flush=True)

    D = {}
    for mode in MODES:
        path = root / mode / "arcface_views.npz"
        if not path.exists():
            print(f"[arcface-sum] ATTENZIONE: {path} assente, righe {mode} saltate", flush=True)
            continue
        for tag, yaws in VIEWS.items():
            D[f"arcface_{mode}_{tag}"] = arcface_distances(path, idx, yaws)
    if REFERENCE not in D:
        raise SystemExit(f"riga di riferimento {REFERENCE} assente")
    if args.joint_stage is not None and (args.joint_stage / "embeddings.npz").exists():
        D["joint@bfm"] = zes.model_distances(args.joint_stage, idx)
        staged = json.loads((args.joint_stage.parent / "subjects.json").read_text())
        if sorted(staged["subjects"]) != subjects:
            raise SystemExit("congiunto: zs_stage ha valutato soggetti diversi")
    else:
        print(f"[arcface-sum] ATTENZIONE: embedding del congiunto assenti ({args.joint_stage})", flush=True)
    bl_root = args.runs / "baselines"
    for m in zes.BL_FACEBENCH:
        try:
            D[m] = zes.facebench_distances(bl_root, m, idx)
        except FileNotFoundError as exc:
            print(f"[arcface-sum] ATTENZIONE: faceBench {m} incompleto ({exc}), riga saltata", flush=True)
    if (bl_root / "region_chamfer.npz").exists():
        reg = zes.region_distances(bl_root / "region_chamfer.npz", idx)
        reg.pop("_kept")
        D.update(reg)
    print(f"[arcface-sum] metodi {list(D)}", flush=True)

    # Stesse repliche e stessi blocchi di zs_expr_summarize.main
    counts = zes.bootstrap_counts(len(subjects), args.n_bootstrap, base.stable_seed(args.seed, "expr_recognition"))
    blocks = {
        "nocrop": ([(a, b) for a in zes.NOCROP for b in zes.NOCROP if a != b],
                   [(a, b) for i, a in enumerate(zes.NOCROP) for b in zes.NOCROP[i + 1:]]),
        "crop": ([(a, b) for a in TOPOLOGIES for b in TOPOLOGIES if a != b and "crop" in (a, b)],
                 [("crop", b) for b in zes.NOCROP]),
    }
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    rec = {}
    with mp.get_context("fork").Pool(min(workers, 16)) as pool:
        tasks = [((blk, name), D[name], idx, pr, pv, counts) for blk, (pr, pv) in blocks.items() for name in D]
        for key, vals, n_nan in pool.imap_unordered(zes._recog_task, tasks):
            rec[key] = (vals, n_nan)
    rec_rows = []
    for (blk, name), (vals, n_nan) in rec.items():
        r = {"block": blk, "method": name, "n_nan_distances": n_nan}
        for m in ("rank1", "map", "auc"):
            r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(vals[m][0]), zes.ci(vals[m])
        rec_rows.append(r)
    rec_table = pd.DataFrame(rec_rows)
    comparisons = [(REFERENCE, b) for b in PRIMARY_BASELINES if b in D] + [(a, b) for a, b in ABLATIONS
                                                                           if a in D and b in D]
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

    out = args.summary_dir / args.domain
    out.mkdir(parents=True, exist_ok=True)
    rec_table.to_csv(out / "recognition.csv", index=False)
    delta_table.to_csv(out / "recognition_paired.csv", index=False)

    order = [n for n in D if n.startswith("arcface_")] + [n for n in D if not n.startswith("arcface_")]

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
            r = sub[(sub["model"] == a_name) & (sub["baseline"] == b_name)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"| {label(a_name)} | {label(b_name)} | "
                         + " | ".join(f"{zes.fmt(r[m], r[m + '_ci_low'], r[m + '_ci_high'], True)} ({r[m + '_p_le0']:.3f})"
                                      for m in ("rank1", "map", "auc")) + " |")
        return lines

    # Controllo: baseline e congiunto contro il summary esistente (stesse repliche -> stessi numeri)
    check = ["- riproduzione del summary esistente: nessun `--reference-csv`"]
    if args.reference_csv is not None and args.reference_csv.exists():
        ref = pd.read_csv(args.reference_csv).set_index(["block", "method"])
        cols = [f"{m}{s}" for m in ("rank1", "map", "auc") for s in ("", "_ci_low", "_ci_high")]
        diffs = {}
        for r in rec_rows:
            if (r["block"], r["method"]) in ref.index:
                diffs[(r["block"], r["method"])] = max(abs(r[c] - ref.loc[(r["block"], r["method"]), c]) for c in cols)
        check = [f"- riproduzione di `{args.reference_csv}` (stesse repliche bootstrap): max |diff| su punto e CI "
                 f"di rank-1, mAP, AUC = {max(diffs.values()):.2e} su {len(diffs)} righe "
                 f"({', '.join(sorted({m for _, m in diffs}))})" if diffs else
                 f"- `{args.reference_csv}`: nessuna riga in comune"]
    n_q = len(blocks["nocrop"][0]) * len(subjects)
    n_gen = len(blocks["nocrop"][1]) * len(subjects)
    n_imp = len(blocks["nocrop"][1]) * len(subjects) * (len(subjects) - 1)
    primary_pairs = [c for c in comparisons if c[0] == REFERENCE and c[1] in PRIMARY_BASELINES]
    ablation_pairs = [c for c in comparisons if c not in primary_pairs]
    parts = [f"# Risultati: {DOMAIN_LABEL[args.domain]}\n",
             f"Soggetti: {len(subjects)} (`select_subjects`, seed {args.eval_seed}), mesh da `{args.view_dir}`, "
             f"baseline da `{bl_root}`" + (f", congiunto da `{args.joint_stage}`" if "joint@bfm" in D else
                                           ", congiunto: embedding assenti, riga mancante")
             + f". CI 95% bootstrap per soggetto, {args.n_bootstrap} repliche (le stesse per tutte le righe).\n",
             *calibration_md(root), "",
             "## PRIMARIO: riconoscimento d'identita', 5 topologie senza crop\n",
             f"Retrieval: {n_q} query, galleria di {len(subjects)} mesh in un'altra topologia; mAP = MRR. "
             f"Verifica: {n_gen} coppie stessa persona, {n_imp} persone diverse.\n",
             *rec_md("nocrop"),
             "\n### Delta appaiati, riferimento ArcFace - baseline (stesse repliche)\n",
             *delta_md("nocrop", primary_pairs),
             "\n### Ablazioni (secondarie), delta appaiati\n", *delta_md("nocrop", ablation_pairs),
             "\n## A parte: crop (coppie di topologie con crop da un lato)\n", *rec_md("crop"),
             "\n### Delta appaiati, crop\n", *delta_md("crop", primary_pairs + ablation_pairs),
             "\n## Controlli\n", *check,
             f"- png di controllo: `{root / 'shaded' / 'control'}`, `{root / 'normals' / 'control'}`; "
             f"frame: `{root / 'shaded' / 'frame_check'}`",
             ]
    (args.summary_dir / f"results_{args.domain}.md").write_text("\n".join(parts) + "\n", encoding="utf-8")

    sections = [(args.summary_dir / "protocol.md").read_text().rstrip()]
    for name in ("results_fv_expr.md", "results_hifi3d.md", "giudizio.md"):
        if (args.summary_dir / name).exists():
            sections += ["\n---\n", (args.summary_dir / name).read_text().rstrip()]
    (args.summary_dir / "summary.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"[arcface-sum] scritto {args.summary_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
