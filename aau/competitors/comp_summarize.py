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
template, distanza ``ir_template.template_distances``), ``emb_<modello>[_rx90|_yzswap].npz`` di comp_embed.py
(Uni3D, OpenShape, distanza 1 - coseno). Le righe ``_rx90`` e ``_yzswap`` sono le ablazioni del frame.

Aggiunta del 9 ottobre (protocollo, "Aggiunta -- 2026-10-09"): riga di riconoscimento di e108 (embedding di
``scale_e108_embed`` via ``zs_arcface_vs_scale.distances_b``), ShapeDNA con autovalori / lambda_1, OpenShape con
lo scambio y/z del training, e tutte le righe di Spearman anche con la GT unificata
(``datasets/UNIFIED_GT/eval/hifi3d_gt_matrix.npz``, ``zs_summarize.with_gt``, stessi semi delle righe maxabs).
Le sezioni dell'8 ottobre del summary restano com'erano; le righe nuove vanno in una sezione datata in fondo.
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
import zs_arcface_vs_scale as zavs  # noqa: E402
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

# --- aggiunta del 9 ottobre (protocollo, "Aggiunta -- 2026-10-09")
ADDED_ON = "2026-10-09"
# ShapeDNA con autovalori / lambda_1 (Reuter et al. 2006): nome -> (riga grezza, k)
SPECTRAL_L1 = {"shapedna_k50_l1": ("shapedna_k50", 50), "shapedna_k100_l1": ("shapedna_k100", 100)}
SPECTRAL_L1_LABEL = {"shapedna_k50_l1": "ShapeDNA k=50, lambda_i / lambda_1",
                     "shapedna_k100_l1": "ShapeDNA k=100, lambda_i / lambda_1"}
# Frame dei neurali: (valore di ``rotation`` nel npz, suffisso, etichetta); yzswap e' nuova (solo OpenShape:
# per Uni3D la trasformazione del training e' l'identita', cioe' la riga primaria)
FRAMES = (("none", "", ""), ("rx90", "_rx90", ", Rx(+90) (ablazione)"),
          ("yzswap", "_yzswap", ", scambio y/z del training (riflessione, ablazione)"))
ADDED = set(SPECTRAL_L1) | {"uni3d_yzswap", "openshape_yzswap"}
UNIFIED_GT = zsum.REPO_ROOT / "datasets" / "UNIFIED_GT" / "eval" / "hifi3d_gt_matrix.npz"
E8_DIR = zsum.RUNS / "evidence" / "e8"
ARCVS_CSV = zsum.RUNS / "data_scale_ood" / "arcface_vs_scale_hifi3d" / "recognition.csv"
# Riconoscimento: e108 contro queste righe (delta appaiati sulle stesse repliche)
RECOG_E108_VS = (TEMPLATE[0], "rigid_icp_chamfer", "arcface_shaded_3v", "chamfer")


def variant_of(name: str, names: list[str]) -> list[str]:
    """Righe di base contro cui si confronta una variante (ablazione del frame o ShapeDNA / lambda_1)."""
    if name in SPECTRAL_L1:
        return [SPECTRAL_L1[name][0]]
    if name.endswith("_rx90"):
        return [name[:-5]]
    if name.endswith("_yzswap"):
        b = name[:-7]
        return [b] + ([b + "_rx90"] if b + "_rx90" in names else [])
    return []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True, help="protocol.md, descrittori, embedding; scrive summary.md")
    p.add_argument("--zs-runs", type=Path, required=True, help="data_<fp> di HIFI3D: e108, baseline, congiunto")
    p.add_argument("--view-dir", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, default=zsum.RUNS / "data_scale_ood" / "hifi")
    p.add_argument("--arcface-root", type=Path, default=zsum.RUNS / "arcface_render_zs" / "hifi3d")
    p.add_argument("--joint-stage", type=Path, default=None,
                   help="default <zs-runs>/joint_frame-xmymz_flip_ranking/zs_zeroshot (come zs_arcface_summarize)")
    p.add_argument("--unified-gt", type=Path, default=UNIFIED_GT, help="GT unificata (aggiunta del 9 ottobre)")
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
        for name, (_, k) in SPECTRAL_L1.items():
            desc[name] = ev[:, 1:k + 1].astype(np.float64) / ev[:, 1:2].astype(np.float64)
        for name, lab in SPECTRAL + tuple(SPECTRAL_L1_LABEL.items()):
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
        for rot, suffix, frame_lab in FRAMES:
            path = out / f"emb_{name}{suffix}.npz"
            if not path.exists():
                if name + suffix != "uni3d_yzswap":  # non prevista dal protocollo
                    print(f"[comp-sum] ATTENZIONE: {path} assente, riga saltata", flush=True)
                continue
            z = np.load(path)
            if str(z["rotation"]) != rot:
                raise SystemExit(f"{path}: rotazione {z['rotation']} != {rot}")
            E = z["E"][ordered(z, idx)].astype(np.float64)
            E /= np.linalg.norm(E, axis=1, keepdims=True)
            D[name + suffix] = 1.0 - E @ E.T
            labels[name + suffix] = lab + frame_lab
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


def unified_frame(pm: pd.DataFrame, names: list[str], gt_path: Path) -> pd.DataFrame:
    """Le stesse righe con la GT unificata (``zs_summarize.with_gt``) e le colonne di distanza rinominate
    ``unified|<colonna>``: la chiave della cache di boot/pboot contiene la colonna, il seme no, quindi le righe
    unificate stanno sulle STESSE repliche delle maxabs senza collidere in cache."""
    cols = ["latent_distance", "raw_chamfer"] + names
    return zsum.with_gt(pm, zsum.load_gt(gt_path)).rename(columns={c: f"unified|{c}" for c in cols})


def spearman_tasks(pm: pd.DataFrame, names: list[str], args, bm, gt: str = "maxabs") -> tuple[list[dict], list[dict]]:
    """Celle (righe di tabella) e delta appaiati; con zsum._STATE["collect"] raccoglie solo i compiti.

    ``gt="unified"``: ``pm`` e' quello di ``unified_frame``; i token (quindi i semi) restano quelli maxabs."""
    c = (lambda col: col) if gt == "maxabs" else (lambda col: f"{gt}|{col}")  # noqa: E731
    cells, paired = [], []
    per_sp = pm.groupby(["subject_a", "subject_b"], as_index=False)[
        ["gt_distance", c("latent_distance"), c("raw_chamfer")] + [c(n) for n in names]].mean()
    groups = {"nocrop_cross": pm[pm["topology_a"].ne("crop") & pm["topology_b"].ne("crop")], "all_cross": pm}
    # riferimenti: stessi token di zs_summarize.arm_tables (braccio, gt, gruppo, metrica)
    for metric, col in base.METRICS:
        for g in GROUPS:
            cells.append({"gt": gt, "method": f"ref_{metric}", "scenario": g,
                          **zsum.boot(groups[g], c(col), args, bm, REF_ARM, "maxabs", g, metric)})
        cells.append({"gt": gt, "method": f"ref_{metric}", "scenario": "subject_pair_mean",
                      **zsum.boot(per_sp, c(col), args, bm, REF_ARM, "maxabs", "spm_clean", metric)})
    for name in names:
        for g in GROUPS:
            cells.append({"gt": gt, "method": name, "scenario": g,
                          **zsum.boot(groups[g], c(name), args, bm, "comp", name, "maxabs", g)})
        cells.append({"gt": gt, "method": name, "scenario": "subject_pair_mean",
                      **zsum.boot(per_sp, c(name), args, bm, "comp", name, "maxabs", "spm_clean")})

    specs = [(f"{REF_LABEL} - {CHAMFER_LABEL}", "latent_distance", "raw_chamfer")]  # stesso nome -> stesso seme
    for name in names:
        specs += [(f"{name} - {REF_LABEL}", name, "latent_distance"), (f"{name} - {CHAMFER_LABEL}", name, "raw_chamfer")]
        specs += [(f"{name} - {b}", name, b) for b in variant_of(name, names)]
    for cmp_name, ca, cb in specs:
        for sc in SCENARIOS:
            r = zsum.pboot(zsum.scenario_frame(pm, sc, [c(ca), c(cb)]), c(ca), c(cb), args, bm, "", "maxabs",
                           cmp_name, sc)
            paired.append({"gt": gt, "comparison": cmp_name, "a_col": ca, "b_col": cb, "scenario": sc, **r})
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

def recognition(D: dict, idx: zes.Index, args, pairs: list[tuple[str, str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Righe di tutti i metodi in ``D`` e delta appaiati ``(A, B, aggiunta)`` (aggiunta: "" o la data)."""
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
        for a_name, b_name, added in pairs:
            if a_name not in D or b_name not in D:
                continue
            a, b = rec[(blk, a_name)][0], rec[(blk, b_name)][0]
            r = {"block": blk, "model": a_name, "baseline": b_name, "added": added}
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


def e8_check(cells_u: pd.DataFrame, paired_u: pd.DataFrame, e8_dir: Path) -> pd.DataFrame:
    """Righe con la GT unificata contro ``eval_methods.py`` (``<e8_dir>/methods_*.csv``, GT ``unified``).

    Righe singole: solo il punto (eval_methods usa per tutte le righe il seme della differenza, qui ogni riga ha
    il suo seme maxabs). Differenza e108 - Chamfer eval: stesso seme (``zs_summarize.pboot``), punto, CI e P."""
    sp = pd.read_csv(e8_dir / "methods_spearman.csv")
    sp = sp[(sp["domain"] == "hifi3d") & (sp["gt"] == "unified")]
    pr = pd.read_csv(e8_dir / "methods_paired.csv")
    pr = pr[(pr["domain"] == "hifi3d") & (pr["gt"] == "unified") & (pr["a"] == "scale_e108")
            & (pr["b"] == "chamfer_eval")]
    rows = []
    for sc in ("nocrop_cross", "subject_pair_mean"):
        for meth, e8m, lab in (("ref_latent", "scale_e108", REF_LABEL), ("ref_chamfer", "chamfer_eval", CHAMFER_LABEL)):
            ref = sp[(sp["group"] == sc) & (sp["method"] == e8m)].iloc[0]
            new = cells_u[(cells_u["method"] == meth) & (cells_u["scenario"] == sc)].iloc[0]
            rows.append({"riga": f"{lab} (solo punto)", "scenario": sc, "e8": f"{ref['point']:.4f}",
                         "ricalcolato": f"{new['spearman']:.4f}", "max_abs_diff": abs(new["spearman"] - ref["point"])})
        ref = pr[pr["group"] == sc].iloc[0]
        new = paired_u[(paired_u["comparison"] == f"{REF_LABEL} - {CHAMFER_LABEL}") & (paired_u["scenario"] == sc)].iloc[0]
        d = max(abs(new["diff"] - ref["point"]), abs(new["ci_low"] - ref["ci_low"]), abs(new["ci_high"] - ref["ci_high"]),
                abs(new["p_boot_le0"] - ref["p_le0"]))
        rows.append({"riga": f"delta {REF_LABEL} - {CHAMFER_LABEL} (punto, CI, P)", "scenario": sc,
                     "e8": f"{ref['point']:+.4f} [{ref['ci_low']:+.4f}, {ref['ci_high']:+.4f}]",
                     "ricalcolato": f"{new['diff']:+.4f} [{new['ci_low']:+.4f}, {new['ci_high']:+.4f}]",
                     "max_abs_diff": d})
    return pd.DataFrame(rows)


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
    names_old = [n for n in names if n not in ADDED]  # le righe delle sezioni dell'8 ottobre
    print(f"[comp-sum] competitori {names}", flush=True)

    bm = base.load_bootstrap_module()
    pm = pair_frame(args.zs_runs, idx, D)
    pm_u = unified_frame(pm, names, args.unified_gt)
    zsum._STATE["collect"] = True
    spearman_tasks(pm, names, args, bm)
    spearman_tasks(pm_u, names, args, bm, gt="unified")
    zsum._STATE["collect"] = False
    zsum.run_boot_tasks(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    cells, paired = (pd.DataFrame(x) for x in spearman_tasks(pm, names, args, bm))
    cells_u, paired_u = (pd.DataFrame(x) for x in spearman_tasks(pm_u, names, args, bm, gt="unified"))
    check = reference_check(cells, paired, args.reference_dir)
    print(f"[comp-sum] riproduzione riferimenti:\n{check.to_string(index=False)}", flush=True)
    check_e8 = e8_check(cells_u, paired_u, E8_DIR)
    print(f"[comp-sum] GT unificata contro e8:\n{check_e8.to_string(index=False)}", flush=True)

    # Riconoscimento: competitori + righe di riferimento + e108 (aggiunta del 9 ottobre)
    R = dict(D)
    R["arcface_shaded_3v"] = zas.arcface_distances(args.arcface_root / "shaded" / "arcface_views.npz", idx,
                                                   zas.VIEWS["3v"])
    for m in ("chamfer", "rigid_icp_chamfer", "nicp_p2tri"):
        R[m] = zes.facebench_distances(args.zs_runs / "baselines", m, idx)
    R["joint@bfm"] = zes.model_distances(joint_stage, idx)
    # e108 dagli embedding di scale_e108_embed, con i controlli di zs_arcface_vs_scale (checkpoint = quello del
    # breakdown; ||z_i - z_j|| = latent_distance delle pair_metrics)
    D_b, checks_b = zavs.distances_b(argparse.Namespace(runs=args.zs_runs, joint_stage=joint_stage), idx, {})
    R[REF_ARM] = D_b[REF_ARM]
    checks_b = {k: v for k, v in checks_b.items() if k.startswith(REF_ARM)}
    pairs = [(a, b, "") for a in names_old for b in RECOG_BASELINES]
    pairs += [(REF_ARM, b, ADDED_ON) for b in RECOG_E108_VS]
    pairs += [(a, b, ADDED_ON) for a in names if a in ADDED for b in RECOG_BASELINES + (REF_ARM,)]
    pairs += [(a, b, ADDED_ON) for a in names if a in ADDED for b in variant_of(a, names)]
    rec, rec_delta = recognition(R, idx, args, pairs)
    rec_check = recognition_check(rec, args.arcface_root / "recognition.csv")
    print("[comp-sum] " + rec_check[0], flush=True)
    rec_check_vs = recognition_check(rec, ARCVS_CSV)
    print("[comp-sum] " + rec_check_vs[0], flush=True)

    out = args.out_dir
    pd.concat([cells, cells_u]).to_csv(out / "spearman.csv", index=False)
    pd.concat([paired, paired_u]).to_csv(out / "paired.csv", index=False)
    check.to_csv(out / "reference_check.csv", index=False)
    check_e8.to_csv(out / "reference_check_unified_e8.csv", index=False)
    rec.to_csv(out / "recognition.csv", index=False)
    rec_delta.to_csv(out / "recognition_paired.csv", index=False)

    lab = {"ref_latent": f"{REF_LABEL} (riferimento)", "ref_chamfer": f"{CHAMFER_LABEL} (riferimento)", **labels}
    rlab = {**labels, "arcface_shaded_3v": "ArcFace, ombreggiato, 3 viste (riferimento)",
            "joint@bfm": "BFM+ICT, convenzione BFM", **{k: v for k, v in zsum.BL_LABEL.items()},
            REF_ARM: REF_LABEL}
    primary = [n for n in names_old if not n.endswith("_rx90")]
    ablation = [n for n in names_old if n.endswith("_rx90")]

    def sp_table(methods: list[str], gt: str = "maxabs") -> list[str]:
        lines = ["| metodo | nocrop_cross (PRIMARIO) | all_cross | subject_pair_mean |", "| --- | --- | --- | --- |"]
        tab = cells if gt == "maxabs" else cells_u
        for mth in methods:
            sel = tab[tab["method"] == mth].set_index("scenario")
            lines.append(f"| {lab[mth]} | " + " | ".join(fmt(sel.loc[sc]) for sc in SCENARIOS) + " |")
        return lines

    def paired_table(comps: list[str], gt: str = "maxabs") -> list[str]:
        lines = ["| confronto | scenario | differenza [CI 95%] (a / b) | P(boot <= 0) |", "| --- | --- | --- | --- |"]
        tab = paired if gt == "maxabs" else paired_u
        for c in comps:
            for sc in SCENARIOS:
                r = tab[(tab["comparison"] == c) & (tab["scenario"] == sc)].iloc[0]
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

    def rec_delta_table(blk: str, models: list[str] | None = None) -> list[str]:
        """Senza ``models``: i delta dell'8 ottobre; con ``models``: quelli aggiunti il 9 con A in ``models``."""
        lines = ["| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |",
                 "| --- | --- | --- | --- | --- |"]
        sel = rec_delta[rec_delta["block"] == blk]
        sel = sel[sel["added"] == ""] if models is None else sel[(sel["added"] == ADDED_ON) & sel["model"].isin(models)]
        for r in sel.itertuples():
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
        *rec_table("nocrop", list(RECOG_REFS) + names_old),
        "\n#### Delta appaiati, competitore - riferimento (stesse repliche)\n", *rec_delta_table("nocrop"),
        "\n### A parte: crop (coppie con crop da un lato)\n", *rec_table("crop", list(RECOG_REFS) + names_old),
        "\n#### Delta appaiati, crop\n", *rec_delta_table("crop"),
        "\n~~E108 non ha una riga di riconoscimento: per i bracci di scala esistono solo le pair_metrics del breakdown "
        "(coppie fra soggetti diversi), non gli embedding per mesh che servono per le coppie stessa persona.~~ "
        f"**Corretto il {ADDED_ON}: la frase era falsa.** Gli embedding per mesh di e108 esistono "
        f"(`{args.zs_runs}/{REF_ARM}_embed`); la riga di riconoscimento e' nella sezione datata in fondo.\n",
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
    for n in names_old:
        if n in info and "dim" in info[n]:
            parts.append(f"- {labels[n]}: embedding di dimensione {info[n]['dim']}, {info[n]['n_points']} punti per mesh")

    # ------------------------------------------------------------ sezione datata del 9 ottobre
    added = [n for n in names if n in ADDED]
    all_rows = [n for n in names if not variant_of(n, names) and n not in SPECTRAL_L1] \
        + [n for n in names if variant_of(n, names) or n in SPECTRAL_L1]
    parts += [
        f"\n## Aggiunta -- {ADDED_ON}: revisione del critic (protocollo, aggiunta della stessa data)\n",
        "Le sezioni sopra sono quelle dell'8 ottobre, rigenerate con gli stessi semi e invariate, salvo la frase "
        "corretta sul riconoscimento di e108. Qui: (1) riconoscimento di e108; (2) tutte le righe di Spearman con la GT "
        "unificata; (3) OpenShape con lo scambio y/z esatto del training; (4) ShapeDNA con autovalori / lambda_1. "
        "Stesse repliche delle altre righe.\n",
        f"### 1. Riconoscimento di e108 (embedding di `{args.zs_runs}/{REF_ARM}_embed`)\n",
        "Stesso checkpoint del breakdown e stesse distanze latenti delle pair_metrics (controlli in fondo); stesse "
        "repliche (`expr_recognition`) di tutte le altre righe di riconoscimento.\n",
        "#### PRIMARIO: 5 topologie senza crop\n",
        *rec_table("nocrop", [REF_ARM] + list(RECOG_E108_VS)),
        "", *rec_delta_table("nocrop", [REF_ARM]),
        "\n#### A parte: crop\n",
        *rec_table("crop", [REF_ARM] + list(RECOG_E108_VS)),
        "", *rec_delta_table("crop", [REF_ARM]),
        "\n### 2. Spearman con la GT unificata (tutte le righe)\n",
        f"GT `{args.unified_gt}` (v3_work/unified_gt/make_eval_gt.py: RMS pesato per area in mm sulla regione FLAME "
        "comune dopo Procrustes di similarita'), sostituita per nome di soggetto (`zs_summarize.with_gt`) nelle STESSE "
        "righe delle pair_metrics di e108; stessi scenari e STESSI semi delle righe maxabs (cambia solo la GT).\n",
        *sp_table(["ref_latent", "ref_chamfer"] + all_rows, gt="unified"),
        "\n#### Delta appaiati con la GT unificata: competitore - e108 e competitore - Chamfer eval\n",
        *paired_table([f"{n} - {REF_LABEL}" for n in all_rows] + [f"{n} - {CHAMFER_LABEL}" for n in all_rows],
                      gt="unified"),
        f"\nRiferimento con la GT unificata: {REF_LABEL} - {CHAMFER_LABEL}\n",
        *paired_table([f"{REF_LABEL} - {CHAMFER_LABEL}"], gt="unified"),
        "\n#### Varianti contro la loro riga di base, GT unificata\n",
        *paired_table([f"{n} - {b}" for n in all_rows for b in variant_of(n, names)], gt="unified"),
    ]
    if "openshape_yzswap" in D:
        yz = "openshape_yzswap"
        parts += [
            "\n### 3. Frame esatto del training: OpenShape con lo scambio y/z (ablazione)\n",
            "OpenShape (`src/data.py`, `y_up`) scambia y e z, (x, y, z) -> (x, z, y): una riflessione (det -1), "
            "non la rotazione Rx(+90) delle righe dell'8 ottobre; poi `normalize_pc` e, in training, una rotazione "
            "casuale attorno a z. Uni3D (`Ensembled_embedding` del pre-training) non scambia gli assi e ruota attorno a "
            "y: la sua trasformazione esatta e' l'identita', cioe' la riga primaria (nessuna riga nuova). "
            "La riga primaria di OpenShape resta quella del frame nativo.\n",
            "#### Spearman, GT maxabs\n",
            *sp_table([b for b in ("openshape", "openshape_rx90") if b in D] + [yz]), "",
            *paired_table([f"{yz} - {b}" for b in variant_of(yz, names)] + [f"{yz} - {REF_LABEL}", f"{yz} - {CHAMFER_LABEL}"]),
            "\n#### Riconoscimento\n",
            *rec_table("nocrop", [b for b in ("openshape", "openshape_rx90") if b in D] + [yz]),
            "", *rec_delta_table("nocrop", [yz]),
            "\nCrop:\n", *rec_table("crop", [b for b in ("openshape", "openshape_rx90") if b in D] + [yz]),
            "", *rec_delta_table("crop", [yz]),
        ]
    l1 = [n for n in SPECTRAL_L1 if n in D]
    if l1:
        parts += [
            "\n### 4. ShapeDNA con autovalori / lambda_1\n",
            "(lambda_1, ..., lambda_k) / lambda_1 dagli stessi autovalori (normalizzazione di scala di Reuter et al. "
            "2006), distanza L2. Le righe grezze sono gia' riscalate per area (mesh ad area 1).\n",
            "#### Spearman, GT maxabs\n",
            *sp_table([x for n in l1 for x in (SPECTRAL_L1[n][0], n)]), "",
            *paired_table([f"{n} - {SPECTRAL_L1[n][0]}" for n in l1] + [f"{n} - {REF_LABEL}" for n in l1]
                          + [f"{n} - {CHAMFER_LABEL}" for n in l1]),
            "\n#### Riconoscimento\n",
            *rec_table("nocrop", [x for n in l1 for x in (SPECTRAL_L1[n][0], n)]),
            "", *rec_delta_table("nocrop", l1),
            "\nCrop:\n", *rec_table("crop", [x for n in l1 for x in (SPECTRAL_L1[n][0], n)]),
            "", *rec_delta_table("crop", l1),
        ]
    parts += [
        "\n### Controlli dell'aggiunta\n",
        f"GT unificata contro `{E8_DIR}` (eval_methods.py, GT `unified`; righe singole: solo il punto, perche' "
        "eval_methods usa per tutte le righe il seme della differenza; differenza appaiata: stesso seme):\n",
        base.md_table(check_e8, list(check_e8.columns), floats=6),
        "", *rec_check_vs,
        *[f"- {k}: {v:.2e}" for k, v in checks_b.items()],
    ]
    for n in added:
        if n in info and "dim" in info[n]:
            parts.append(f"- {labels[n]}: embedding di dimensione {info[n]['dim']}, {info[n]['n_points']} punti per mesh")
    (out / "results.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    sections = [(out / "protocol.md").read_text().rstrip(), "\n---\n", (out / "results.md").read_text().rstrip()]
    (out / "summary.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"[comp-sum] scritto {out / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
