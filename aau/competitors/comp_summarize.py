#!/usr/bin/env python3
"""Competitori diretti sulla verita' geometrica di HIFI3D: Spearman con la GT, delta appaiati, riconoscimento.

    aau/run.sh aau/competitors/comp_summarize.py --out-dir aau/runs/competitors_hifi3d \\
        --zs-runs aau/runs/ws_hifi3d/data_328f2bfc1a --view-dir datasets/HIFI3D/eval_view/npz
    (comp_summarize.sbatch)

Protocollo: ``<out-dir>/protocol.md``, dichiarato prima del calcolo e copiato in testa al summary.
Nessuna funzione di misura e' riscritta:
  - Spearman con CI: ``zs_summarize.boot`` (``weighted_bootstrap_spearman`` del paper) con i semi
    di zs_summarize.py; delta appaiati: ``zs_summarize.pboot`` / ``paired_bootstrap``, scenari di
    ``zs_summarize.scenario_frame``. Le righe sono quelle delle pair_metrics di e108 (soggetto
    a < b, 30 coppie ordinate di topologie), a cui si aggiunge una colonna per competitore presa
    dalla sua matrice (600, 600). Righe di riferimento (e108, Chamfer eval, e108 - Chamfer eval)
    ricalcolate con gli STESSI semi del summary esistente: devono coincidere con
    ``<reference-dir>/table_cells.csv`` e ``paired.csv`` (controllo nel summary).
  - Riconoscimento: ``zs_expr_summarize`` (Index, _recog_task, bootstrap_counts, ci) con le
    repliche di ``aau/runs/arcface_render_zs/results_hifi3d.md``; righe di riferimento confrontate
    con ``<arcface-root>/recognition.csv``.

Competitori (presenti solo se c'e' il file): ``desc_spectral.npz`` di comp_spectral.py (ShapeDNA
k=50 e k=100, HKS e WKS globali, distanza L2), ``template_hifi3d.npz`` di comp_template.py (NICP su
template, distanza ``ir_template.template_distances``), ``emb_<modello>[_rx90].npz`` di comp_embed.py
(Uni3D, OpenShape, distanza 1 - coseno). Le righe ``_rx90`` sono l'ablazione del frame.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "zs3dmm"))
sys.path.insert(0, str(THIS_DIR.parent / "indomain"))

import ir_template as irt  # noqa: E402  (solo template_distances; open3d non serve)
import zs_arcface_summarize as zas  # noqa: E402
import zs_expr_summarize as zes  # noqa: E402
import zs_summarize as zsum  # noqa: E402
from zs_stage import TOPOLOGIES, select_subjects  # noqa: E402

base = zsum.base
REF_ARM = "scale_e108"
REF_LABEL = "BFM+ICT+GNM (10^5), e108"
CHAMFER_LABEL = "Chamfer eval"
SPECTRAL = (("shapedna_k50", "ShapeDNA k=50"), ("shapedna_k100", "ShapeDNA k=100"),
            ("hks", "HKS globale"), ("wks", "WKS globale"))
TEMPLATE = ("nicp_template", "NICP su template (iscrizione)")
NEURAL = (("uni3d", "Uni3D-g"), ("openshape", "OpenShape PointBERT"))
GROUPS = ("nocrop_cross", "all_cross")
SCENARIOS = ("nocrop_cross", "all_cross", "subject_pair_mean")
# Riconoscimento: righe di riferimento da recognition.csv di arcface_render_zs (stesse repliche)
RECOG_REFS = ("arcface_shaded_3v", "chamfer", "rigid_icp_chamfer", "nicp_p2tri", "joint@bfm")
RECOG_BASELINES = ("chamfer", "arcface_shaded_3v")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True, help="protocol.md, descrittori, embedding; scrive summary.md")
    p.add_argument("--zs-runs", type=Path, required=True, help="data_<fp> di HIFI3D: e108, baseline, congiunto")
    p.add_argument("--view-dir", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, default=zsum.RUNS / "data_scale_ood" / "hifi")
    p.add_argument("--arcface-root", type=Path, default=zsum.RUNS / "arcface_render_zs" / "hifi3d")
    p.add_argument("--joint-stage", type=Path, default=None,
                   help="default <zs-runs>/joint_frame-xmymz_flip_ranking/zs_zeroshot (come zs_arcface_summarize)")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234, help="Seme del ricampionamento (quello del summary esistente)")
    p.add_argument("--eval-seed", type=int, default=1234, help="WBES_EVAL_SEED: scelta dei soggetti")
    return p.parse_args()


# ------------------------------------------------------------------- distanze (600 x 600)

def ordered(z, idx: zes.Index) -> np.ndarray:
    keys = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))
    if sorted(keys) != sorted(idx.keys):
        raise SystemExit("descrittori di mesh diverse da quelle attese")
    pos = {k: i for i, k in enumerate(keys)}
    return np.asarray([pos[k] for k in idx.keys])


def l2(X: np.ndarray) -> np.ndarray:
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))


def competitor_distances(out: Path, idx: zes.Index) -> tuple[dict, dict, dict]:
    """{nome: D}, {nome: etichetta}, {nome: info per i controlli}."""
    D, labels, info = {}, {}, {}
    spec = out / "desc_spectral.npz"
    if spec.exists():
        z = np.load(spec)
        o = ordered(z, idx)
        ev = z["evals"][o]
        desc = {"shapedna_k50": ev[:, 1:51], "shapedna_k100": ev[:, 1:101], "hks": z["hks"][o], "wks": z["wks"][o]}
        for name, lab in SPECTRAL:
            D[name], labels[name] = l2(desc[name].astype(np.float64)), lab
        info["spectral"] = {"identity_rel_err_max": float(z["identity_rel_err"].max()),
                            "n_multi_component": int((z["n_components"] > 1).sum()),
                            "n_dropped_vertices_meshes": int((z["n_dropped"] > 0).sum()),
                            "lambda0_max": float(z["evals"][:, 0].max()),
                            "lambda1_min": float(z["evals"][:, 1].min()),
                            "lambda100_median": float(np.median(z["evals"][:, 100]))}
    else:
        print(f"[comp-sum] ATTENZIONE: {spec} assente, righe spettrali saltate", flush=True)
    tpl = out / "template_hifi3d.npz"
    if tpl.exists():
        z = np.load(tpl)
        R = z["R"][ordered(z, idx)].astype(np.float64)
        name, lab = TEMPLATE
        D[name], labels[name] = irt.template_distances(R, R), lab
        info[name] = {"n_failed": len(z["failed"]), "median_seconds": float(np.median(z["seconds"])),
                      "n_template_subjects": len(z["template_subjects"]),
                      "n_template_vertices": int(z["n_template_vertices"])}
    else:
        print(f"[comp-sum] ATTENZIONE: {tpl} assente, riga NICP su template saltata", flush=True)
    for name, lab in NEURAL:
        for rot, suffix in (("none", ""), ("rx90", "_rx90")):
            path = out / f"emb_{name}{suffix}.npz"
            if not path.exists():
                print(f"[comp-sum] ATTENZIONE: {path} assente, riga saltata", flush=True)
                continue
            z = np.load(path)
            if str(z["rotation"]) != rot:
                raise SystemExit(f"{path}: rotazione {z['rotation']} != {rot}")
            E = z["E"][ordered(z, idx)].astype(np.float64)
            E /= np.linalg.norm(E, axis=1, keepdims=True)
            D[name + suffix] = 1.0 - E @ E.T
            labels[name + suffix] = lab + (", Rx(+90) (ablazione)" if suffix else "")
            info[name + suffix] = {"dim": E.shape[1], "n_points": int(z["n_points"])}
    return D, labels, info


# ---------------------------------------------------------------- Spearman con la GT

def pair_frame(zs_runs: Path, idx: zes.Index, D: dict) -> pd.DataFrame:
    """pair_metrics di e108 (con gt_distance maxabs, latent e Chamfer eval) + una colonna per competitore."""
    stage = zs_runs / f"{REF_ARM}_topology" / zsum.STAGE
    pm = base.read_pair_metrics(stage)
    seen = sorted(set(pm["subject_a"].astype(str)) | set(pm["subject_b"].astype(str)))
    if seen != idx.subjects:
        raise SystemExit("pair_metrics di e108 e select_subjects: soggetti diversi")
    if pm.duplicated(zsum.PAIR_KEYS).any() or len(pm) != 30 * 4950:
        raise SystemExit(f"pair_metrics di e108: {len(pm)} righe o duplicati")
    ia = np.asarray([idx.pos[k] for k in zip(pm["subject_a"].astype(str), pm["topology_a"])])
    ib = np.asarray([idx.pos[k] for k in zip(pm["subject_b"].astype(str), pm["topology_b"])])
    pm = pm.copy()
    for name, M in D.items():
        pm[name] = M[ia, ib]
    return pm


def spearman_tasks(pm: pd.DataFrame, names: list[str], args, bm) -> tuple[list[dict], list[dict]]:
    """Celle (righe di tabella) e delta appaiati; con zsum._STATE["collect"] raccoglie solo i compiti."""
    cells, paired = [], []
    per_sp = pm.groupby(["subject_a", "subject_b"], as_index=False)[
        ["gt_distance", "latent_distance", "raw_chamfer"] + names].mean()
    groups = {"nocrop_cross": pm[pm["topology_a"].ne("crop") & pm["topology_b"].ne("crop")], "all_cross": pm}
    # riferimenti: stessi token di zs_summarize.arm_tables (braccio, gt, gruppo, metrica)
    for metric, col in base.METRICS:
        for g in GROUPS:
            cells.append({"method": f"ref_{metric}", "scenario": g,
                          **zsum.boot(groups[g], col, args, bm, REF_ARM, "maxabs", g, metric)})
        cells.append({"method": f"ref_{metric}", "scenario": "subject_pair_mean",
                      **zsum.boot(per_sp, col, args, bm, REF_ARM, "maxabs", "spm_clean", metric)})
    for name in names:
        for g in GROUPS:
            cells.append({"method": name, "scenario": g, **zsum.boot(groups[g], name, args, bm, "comp", name, "maxabs", g)})
        cells.append({"method": name, "scenario": "subject_pair_mean",
                      **zsum.boot(per_sp, name, args, bm, "comp", name, "maxabs", "spm_clean")})

    specs = [(f"{REF_LABEL} - {CHAMFER_LABEL}", "latent_distance", "raw_chamfer")]  # stesso nome -> stesso seme
    for name in names:
        specs += [(f"{name} - {REF_LABEL}", name, "latent_distance"), (f"{name} - {CHAMFER_LABEL}", name, "raw_chamfer")]
        if name.endswith("_rx90"):
            specs.append((f"{name} - {name[:-5]}", name, name[:-5]))
    for cmp_name, ca, cb in specs:
        for sc in SCENARIOS:
            r = zsum.pboot(zsum.scenario_frame(pm, sc, [ca, cb]), ca, cb, args, bm, "", "maxabs", cmp_name, sc)
            paired.append({"comparison": cmp_name, "a_col": ca, "b_col": cb, "scenario": sc, **r})
    return cells, paired


def reference_check(cells: pd.DataFrame, paired: pd.DataFrame, ref_dir: Path) -> pd.DataFrame:
    """Righe di riferimento ricalcolate contro i csv del summary esistente: max |diff| su punto e CI."""
    tc = pd.read_csv(ref_dir / "table_cells.csv")
    tc = tc[(tc["model"] == REF_ARM) & (tc["gt"] == "maxabs") & (tc["scenario"] == "clean")]
    proto = {"nocrop_cross": "mesh_pair_nocrop_cross", "all_cross": "mesh_pair_all_cross",
             "subject_pair_mean": "subject_pair_mean"}
    rows = []
    for metric, _ in base.METRICS:
        for sc in SCENARIOS:
            ref = tc[tc["protocol"] == proto[sc]].iloc[0]
            new = cells[(cells["method"] == f"ref_{metric}") & (cells["scenario"] == sc)].iloc[0]
            d = max(abs(new["spearman"] - ref[f"{metric}_spearman"]), abs(new["ci_low"] - ref[f"{metric}_ci_low"]),
                    abs(new["ci_high"] - ref[f"{metric}_ci_high"]))
            rows.append({"riga": f"{REF_LABEL if metric == 'latent' else CHAMFER_LABEL}", "scenario": sc,
                         "esistente": f"{ref[f'{metric}_spearman']:.4f} [{ref[f'{metric}_ci_low']:.4f}, {ref[f'{metric}_ci_high']:.4f}]",
                         "ricalcolato": f"{new['spearman']:.4f} [{new['ci_low']:.4f}, {new['ci_high']:.4f}]",
                         "max_abs_diff": d})
    pr = pd.read_csv(ref_dir / "paired.csv")
    name = f"{REF_LABEL} - {CHAMFER_LABEL}"
    for sc in SCENARIOS:
        ref = pr[(pr["comparison"] == name) & (pr["scenario"] == sc) & (pr["gt"] == "maxabs")].iloc[0]
        new = paired[(paired["comparison"] == name) & (paired["scenario"] == sc)].iloc[0]
        d = max(abs(new[c] - ref[c]) for c in ("diff", "ci_low", "ci_high", "p_boot_le0"))
        rows.append({"riga": f"delta {name}", "scenario": sc,
                     "esistente": f"{ref['diff']:+.4f} [{ref['ci_low']:+.4f}, {ref['ci_high']:+.4f}]",
                     "ricalcolato": f"{new['diff']:+.4f} [{new['ci_low']:+.4f}, {new['ci_high']:+.4f}]",
                     "max_abs_diff": d})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- riconoscimento

def recognition(D: dict, idx: zes.Index, args, comp_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = zes.bootstrap_counts(len(idx.subjects), args.n_bootstrap, base.stable_seed(args.seed, "expr_recognition"))
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
    rows = []
    for (blk, name), (vals, n_nan) in rec.items():
        r = {"block": blk, "method": name, "n_nan_distances": n_nan}
        for m in ("rank1", "map", "auc"):
            r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(vals[m][0]), zes.ci(vals[m])
        rows.append(r)
    deltas = []
    for blk in blocks:
        for a_name in comp_names:
            for b_name in RECOG_BASELINES:
                if b_name not in D:
                    continue
                a, b = rec[(blk, a_name)][0], rec[(blk, b_name)][0]
                r = {"block": blk, "model": a_name, "baseline": b_name}
                for m in ("rank1", "map", "auc"):
                    d = a[m] - b[m]
                    r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(d[0]), zes.ci(d)
                    r[f"{m}_p_le0"] = float((d[1:] <= 0).mean())
                deltas.append(r)
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def recognition_check(rec: pd.DataFrame, ref_csv: Path) -> list[str]:
    ref = pd.read_csv(ref_csv).set_index(["block", "method"])
    cols = [f"{m}{s}" for m in ("rank1", "map", "auc") for s in ("", "_ci_low", "_ci_high")]
    diffs = {}
    for r in rec.itertuples():
        if (r.block, r.method) in ref.index:
            diffs[(r.block, r.method)] = max(abs(getattr(r, c) - ref.loc[(r.block, r.method), c]) for c in cols)
    if not diffs:
        return [f"- `{ref_csv}`: nessuna riga in comune"]
    return [f"- riconoscimento, riproduzione di `{ref_csv}` (stesse repliche): max |diff| su punto e CI di rank-1, "
            f"mAP, AUC = {max(diffs.values()):.2e} su {len(diffs)} righe ({', '.join(sorted({m for _, m in diffs}))})"]


# --------------------------------------------------------------------------- markdown

def fmt(r) -> str:
    return f"{r['spearman']:.3f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"


def main() -> None:
    args = parse_args()
    args.variant = ""
    joint_stage = args.joint_stage or args.zs_runs / "joint_frame-xmymz_flip_ranking" / zsum.STAGE
    subjects = select_subjects(args.view_dir, args.eval_seed)
    idx = zes.Index(subjects)
    print(f"[comp-sum] {len(subjects)} soggetti (primi {subjects[:3]})", flush=True)
    D, labels, info = competitor_distances(args.out_dir, idx)
    names = list(D)
    if not names:
        raise SystemExit("nessun competitore con risultati")
    print(f"[comp-sum] competitori {names}", flush=True)

    bm = base.load_bootstrap_module()
    pm = pair_frame(args.zs_runs, idx, D)
    zsum._STATE["collect"] = True
    spearman_tasks(pm, names, args, bm)
    zsum._STATE["collect"] = False
    zsum.run_boot_tasks(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    cells, paired = (pd.DataFrame(x) for x in spearman_tasks(pm, names, args, bm))
    check = reference_check(cells, paired, args.reference_dir)
    print(f"[comp-sum] riproduzione riferimenti:\n{check.to_string(index=False)}", flush=True)

    # Riconoscimento: competitori + righe di riferimento
    R = dict(D)
    R["arcface_shaded_3v"] = zas.arcface_distances(args.arcface_root / "shaded" / "arcface_views.npz", idx,
                                                   zas.VIEWS["3v"])
    for m in ("chamfer", "rigid_icp_chamfer", "nicp_p2tri"):
        R[m] = zes.facebench_distances(args.zs_runs / "baselines", m, idx)
    R["joint@bfm"] = zes.model_distances(joint_stage, idx)
    rec, rec_delta = recognition(R, idx, args, names)
    rec_check = recognition_check(rec, args.arcface_root / "recognition.csv")
    print("[comp-sum] " + rec_check[0], flush=True)

    out = args.out_dir
    cells.to_csv(out / "spearman.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    check.to_csv(out / "reference_check.csv", index=False)
    rec.to_csv(out / "recognition.csv", index=False)
    rec_delta.to_csv(out / "recognition_paired.csv", index=False)

    lab = {"ref_latent": f"{REF_LABEL} (riferimento)", "ref_chamfer": f"{CHAMFER_LABEL} (riferimento)", **labels}
    rlab = {**labels, "arcface_shaded_3v": "ArcFace, ombreggiato, 3 viste (riferimento)",
            "joint@bfm": "BFM+ICT, convenzione BFM", **{k: v for k, v in zsum.BL_LABEL.items()}}
    primary = [n for n in names if not n.endswith("_rx90")]
    ablation = [n for n in names if n.endswith("_rx90")]

    def sp_table(methods: list[str]) -> list[str]:
        lines = ["| metodo | nocrop_cross (PRIMARIO) | all_cross | subject_pair_mean |", "| --- | --- | --- | --- |"]
        for mth in methods:
            sel = cells[cells["method"] == mth].set_index("scenario")
            lines.append(f"| {lab[mth]} | " + " | ".join(fmt(sel.loc[sc]) for sc in SCENARIOS) + " |")
        return lines

    def paired_table(comps: list[str]) -> list[str]:
        lines = ["| confronto | scenario | differenza [CI 95%] (a / b) | P(boot <= 0) |", "| --- | --- | --- | --- |"]
        for c in comps:
            for sc in SCENARIOS:
                r = paired[(paired["comparison"] == c) & (paired["scenario"] == sc)].iloc[0]
                a, b = c.split(" - ", 1)
                name = f"{lab.get(a, a)} - {lab.get(b, b) if b in lab else b}"
                lines.append(f"| {name} | {sc} | {r['diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] "
                             f"({r['a']:.3f} / {r['b']:.3f}) | {r['p_boot_le0']:.3f} |")
        return lines

    def rec_table(blk: str, methods: list[str]) -> list[str]:
        lines = ["| metodo | rank-1 | mAP | AUC verifica | distanze NaN |", "| --- | --- | --- | --- | --- |"]
        sub = rec[rec["block"] == blk].set_index("method")
        for m in methods:
            r = sub.loc[m]
            lines.append(f"| {rlab.get(m, m)} | " + " | ".join(zes.fmt(r[k], r[f"{k}_ci_low"], r[f"{k}_ci_high"])
                                                              for k in ("rank1", "map", "auc"))
                         + f" | {r['n_nan_distances']} |")
        return lines

    def rec_delta_table(blk: str) -> list[str]:
        lines = ["| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |",
                 "| --- | --- | --- | --- | --- |"]
        for r in rec_delta[rec_delta["block"] == blk].itertuples():
            r = r._asdict()
            lines.append(f"| {rlab.get(r['model'], r['model'])} | {rlab.get(r['baseline'], r['baseline'])} | "
                         + " | ".join(f"{zes.fmt(r[m], r[m + '_ci_low'], r[m + '_ci_high'], True)} ({r[m + '_p_le0']:.3f})"
                                      for m in ("rank1", "map", "auc")) + " |")
        return lines

    comps_main = [f"{n} - {REF_LABEL}" for n in primary] + [f"{n} - {CHAMFER_LABEL}" for n in primary]
    comps_abl = [f"{n} - {n[:-5]}" for n in ablation] + [f"{n} - {REF_LABEL}" for n in ablation]
    sp = info.get("spectral")
    parts = [
        "# Risultati: competitori diretti su HIFI3D (verita' geometrica)\n",
        f"Soggetti: {len(subjects)} (`select_subjects`, seed {args.eval_seed}; gli stessi delle pair_metrics di e108, "
        f"controllato), mesh da `{args.view_dir}`, e108 e Chamfer eval da `{args.zs_runs}/{REF_ARM}_topology`. GT `maxabs`. "
        f"CI 95% bootstrap per soggetto, {args.n_bootstrap} repliche; delta appaiati sulle stesse repliche per i due lati.\n",
        "## Spearman con la GT maxabs (mesh-pair, clean)\n",
        *sp_table(["ref_latent", "ref_chamfer"] + primary),
        "\n### Delta appaiati: competitore - e108 e competitore - Chamfer eval\n",
        *paired_table(comps_main),
        f"\nRiferimento, stessi semi del summary esistente: {REF_LABEL} - {CHAMFER_LABEL}\n",
        *paired_table([f"{REF_LABEL} - {CHAMFER_LABEL}"]),
    ]
    if ablation:
        parts += ["\n## Ablazione del frame: rotazione nota Rx(+90) (alto +y -> +z), solo neurali\n",
                  "La riga primaria resta quella del frame nativo.\n",
                  *sp_table(ablation), "", *paired_table(comps_abl)]
    parts += [
        "\n## Riconoscimento d'identita' (protocollo di `aau/runs/arcface_render_zs/results_hifi3d.md`)\n",
        "### PRIMARIO: 5 topologie senza crop\n",
        *rec_table("nocrop", list(RECOG_REFS) + names),
        "\n#### Delta appaiati, competitore - riferimento (stesse repliche)\n", *rec_delta_table("nocrop"),
        "\n### A parte: crop (coppie con crop da un lato)\n", *rec_table("crop", list(RECOG_REFS) + names),
        "\n#### Delta appaiati, crop\n", *rec_delta_table("crop"),
        "\nE108 non ha una riga di riconoscimento: per i bracci di scala esistono solo le pair_metrics del breakdown "
        "(coppie fra soggetti diversi), non gli embedding per mesh che servono per le coppie stessa persona.\n",
        "\n## Controlli\n",
        "Righe di riferimento ricalcolate con gli stessi semi contro `" + str(args.reference_dir) + "` "
        "(table_cells.csv, paired.csv; diff massima su punto e CI):\n",
        base.md_table(check, list(check.columns), floats=6),
        "", *rec_check,
    ]
    if sp:
        parts.append(f"- spettrali: identita' media-per-area = somma spettrale, errore relativo max "
                     f"{sp['identity_rel_err_max']:.1e}; mesh con piu' componenti connesse {sp['n_multi_component']}, "
                     f"con vertici non referenziati tolti {sp['n_dropped_vertices_meshes']}; lambda_0 max "
                     f"{sp['lambda0_max']:.1e}, lambda_1 min {sp['lambda1_min']:.2f}, lambda_100 mediano "
                     f"{sp['lambda100_median']:.0f} (Weyl, area 1: {400 * np.pi:.0f})")
    if TEMPLATE[0] in info:
        t = info[TEMPLATE[0]]
        parts.append(f"- {TEMPLATE[1]}: template = media di {t['n_template_subjects']} original non valutate "
                     f"({t['n_template_vertices']} vertici, 4096 tenuti); iscrizioni fallite {t['n_failed']} su 600 "
                     f"(NaN, contate come +inf nel riconoscimento), mediana {t['median_seconds']:.2f} s per mesh")
    for n in names:
        if n in info and "dim" in info[n]:
            parts.append(f"- {labels[n]}: embedding di dimensione {info[n]['dim']}, {info[n]['n_points']} punti per mesh")
    (out / "results.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    sections = [(out / "protocol.md").read_text().rstrip(), "\n---\n", (out / "results.md").read_text().rstrip()]
    (out / "summary.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"[comp-sum] scritto {out / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
