#!/usr/bin/env python3
"""Protocollo equalize-support: righe con crop prima e dopo l'equalizzazione, per ogni metrica, con CI appaiati.

    aau/run.sh aau/zs3dmm/eqsupport_summarize.py --tag hifi --label HIFI3D \\
        --runs aau/runs/ws_hifi3d/data_328f2bfc1a --gt datasets/HIFI3D/eval_view/gt_matrix.npz \\
        --eq-dir datasets/HIFI3D/eqsupport_view --variants "" _frame-xmymz_flip \\
        --base-baselines aau/runs/ws_hifi3d/data_328f2bfc1a/baselines \\
        --eq-baselines aau/runs/ws_hifi3d/data_328f2bfc1a/baselines_eqsupport --out-dir aau/runs/eqsupport
    (eqsupport_summarize.sbatch, tutti i domini)

Il protocollo: in una coppia con un ``crop`` l'altra mesh e' quella della vista equalizzata
(``eqsupport_view.py``: la stessa topologia ritagliata sulla regione del crop del suo soggetto);
nelle coppie senza crop non cambia niente. Quindi la tabella del protocollo e' la base per le
righe senza crop e la variante ``_eqsupport`` per quelle con crop, unite per (soggetti,
topologie). Nessuna distanza calcolata qui: si leggono

  - modelli e Chamfer eval: ``pair_metrics.csv`` del breakdown di ``zs_zeroshot.sbatch``,
    ``<arm><variante>`` (prima) e ``<arm><variante>_eqsupport`` (dopo);
  - baseline faceBench (Chamfer 4096 pt, rigid ICP, NICP P2P/P2Tri): le matrici di
    ``zs_baselines.sbatch`` (prima: ``baselines``/``outlineB``; dopo: ``baselines_eqsupport``).

Per ogni metrica e gruppo di righe (crop<->X per ogni X, entrambi i versi; tutte le crop; senza
crop; tutte le cross) Spearman con la GT del dominio prima e dopo, e la differenza dopo - prima
con CI 95% bootstrap per soggetto, appaiato sulle stesse repliche (``paired_bootstrap`` di
zs_summarize.py: soggetti con reinserimento, peso della coppia = prodotto dei conteggi).

Controlli scritti nel summary:
  - le righe senza crop del protocollo sono quelle della base, riga per riga (differenza 0);
  - le righe con crop di base e variante sono le stesse coppie, con la stessa GT;
  - ``original`` equalizzata == ``crop`` (bit per bit su HIFI3D/ICT): latente e Chamfer di
    (crop_a, original_b) e (original_a, crop_b) devono coincidere nella variante;
  - la Chamfer eval della variante e' la stessa nei due bracci (non dipende dal modello);
  - la vista equalizzata delle out dir e' quella attuale (eqsupport_manifest.json).

``--train-split`` (splits.json di WS2): per il congiunto, che su BFM e ICT ha in training 81 e
80 dei 100 soggetti, la riga "tutte le crop" ripetuta sui soli soggetti fuori dal suo training.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import zs_summarize as zs  # noqa: E402  (bootstrap appaiato, sorgenti dei bracci, GT)

base = zs.base
common = zs.common
OTHERS = ("original", "noisy", "remesh", "down8k", "up60k")
KEYS = zs.PAIR_KEYS
BL_METRICS = zs.BL_METRICS
BL_LABEL = zs.BL_LABEL
VARIANT_LABEL = {"": "nativo", "_frame-xmymz_flip": "frame BFM (Rx180 + facce invertite)",
                 "_frame-xmymz": "Rx(180)"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="nome breve del dominio nei file, p.es. hifi")
    p.add_argument("--label", required=True)
    p.add_argument("--runs", type=Path, required=True, help="$ZS_RUNS/data_<fp>")
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--eq-dir", type=Path, required=True, help="vista equalizzata (manifest.json, region_stats.csv)")
    p.add_argument("--arms", nargs="+", default=["joint", "bfm_only"])
    p.add_argument("--variants", nargs="+", default=[""], help="suffissi di frame dei bracci; il primo e' quello "
                   "da cui si prende la Chamfer eval")
    p.add_argument("--base-baselines", type=Path, required=True, help="dir con matrices/ (prima)")
    p.add_argument("--eq-baselines", type=Path, required=True, help="dir con matrices/ (dopo)")
    p.add_argument("--train-split", type=Path, default=None)
    p.add_argument("--train-name-offset", type=int, default=0,
                   help="nome nello split = id<int(id) + offset>, p.es. -910000 per ICT_ZS")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234, help="seme del ricampionamento")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def has_crop(df: pd.DataFrame) -> pd.Series:
    return df["topology_a"].eq("crop") | df["topology_b"].eq("crop")


def groups_of(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for x in OTHERS:
        sel = (df["topology_a"].eq("crop") & df["topology_b"].eq(x)) | (df["topology_a"].eq(x) & df["topology_b"].eq("crop"))
        out.append((f"crop<->{x}", df[sel]))
    out.append(("crop (tutte)", df[has_crop(df)]))
    if (~has_crop(df)).any():
        out.append(("senza crop (controllo)", df[~has_crop(df)]))
        out.append(("tutte le cross", df))
    return out


# --------------------------------------------------------------------------- modelli

def read_arm(runs: Path, name: str) -> tuple[pd.DataFrame, Path]:
    src = zs.arm_sources(runs, name)
    pm = base.read_pair_metrics(src["topology"])
    if pm.duplicated(KEYS).any():
        raise SystemExit(f"{name}: righe duplicate in pair_metrics")
    return pm, src["topology"]


def protocol_frame(pm_base: pd.DataFrame, pm_eq: pd.DataFrame, cols: list[str], who: str) -> tuple[pd.DataFrame, dict]:
    """Righe del protocollo con ``<col>_before`` (base) e ``<col>_after`` (eq sulle crop, base altrove)."""
    b_crop, e_crop = pm_base[has_crop(pm_base)], pm_eq[has_crop(pm_eq)]
    m = b_crop[KEYS + ["gt_distance"] + cols].merge(e_crop[KEYS + ["gt_distance"] + cols], on=KEYS,
                                                    suffixes=("_before", "_after"), validate="one_to_one")
    if not (len(m) == len(b_crop) == len(e_crop)):
        raise SystemExit(f"{who}: righe con crop non allineate (base {len(b_crop)}, eq {len(e_crop)}, comuni {len(m)})")
    gt_diff = float((m["gt_distance_before"] - m["gt_distance_after"]).abs().max())
    if gt_diff > 0:
        raise SystemExit(f"{who}: GT diversa sulle stesse righe con crop (max {gt_diff})")
    m = m.rename(columns={"gt_distance_before": "gt_distance"}).drop(columns=["gt_distance_after"])
    nocrop = pm_base[~has_crop(pm_base)][KEYS + ["gt_distance"] + cols].copy()
    for c in cols:
        nocrop[f"{c}_before"] = nocrop[c]
        nocrop[f"{c}_after"] = nocrop[c]   # il protocollo non tocca le righe senza crop
    nocrop = nocrop.drop(columns=cols)
    prot = pd.concat([m, nocrop], ignore_index=True)
    # Controllo: le righe senza crop del protocollo sono quelle della base, riga per riga.
    chk = prot[~has_crop(prot)].merge(pm_base[~has_crop(pm_base)][KEYS + cols], on=KEYS, validate="one_to_one")
    check = {"who": who, "n_rows": len(prot), "n_crop_rows": len(m), "n_nocrop_rows": len(chk),
             "nocrop_rows_match_base": len(chk) == int((~has_crop(pm_base)).sum()),
             "nocrop_max_abs_diff": max(float((chk[f"{c}_after"] - chk[c]).abs().max()) for c in cols),
             "crop_gt_max_abs_diff": gt_diff}
    return prot, check


def original_is_crop_check(pm_eq: pd.DataFrame, who: str) -> dict:
    """Nella variante, (crop_a, original_b) e (original_a, crop_b) sono la stessa coppia di crop."""
    a = pm_eq[pm_eq["topology_a"].eq("crop") & pm_eq["topology_b"].eq("original")]
    b = pm_eq[pm_eq["topology_a"].eq("original") & pm_eq["topology_b"].eq("crop")]
    m = a.merge(b, on=["subject_a", "subject_b"], suffixes=("_co", "_oc"), validate="one_to_one")
    out = {"who": who, "n_pairs": len(m)}
    for c in ("latent_distance", "raw_chamfer"):
        d = (m[f"{c}_co"] - m[f"{c}_oc"]).abs()
        out[f"{c}_max_abs_diff"] = float(d.max())
        out[f"{c}_max_rel_diff"] = float((d / m[f"{c}_oc"].abs().clip(lower=1e-12)).max())
    return out


# -------------------------------------------------------------------------- baseline

def baseline_rows(root: Path, metric: str, gt: tuple, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    D_gt, idx = gt
    parts = []
    for ta, tb in pairs:
        path = common.matrix_path(metric, ta, tb, root)
        if not path.exists():
            raise SystemExit(f"matrice assente: {path}")
        D, subjects, _, _, _ = common.load_matrix(path)
        i, j = common.subject_pair_indices(len(subjects))
        sa, sb = np.asarray(subjects)[i], np.asarray(subjects)[j]
        parts.append(pd.DataFrame({"subject_a": sa, "subject_b": sb, "topology_a": ta, "topology_b": tb,
                                   "gt_distance": D_gt[[idx[s] for s in sa], [idx[s] for s in sb]],
                                   "value": D[i, j]}))
    return pd.concat(parts, ignore_index=True)


def baseline_frame(args, gt, metric: str) -> pd.DataFrame:
    crop_pairs = [(a, b) for a in zs.TOPOLOGIES for b in zs.TOPOLOGIES if a != b and "crop" in (a, b)]
    b = baseline_rows(args.base_baselines, metric, gt, crop_pairs)
    e = baseline_rows(args.eq_baselines, metric, gt, crop_pairs)
    m = b.merge(e[KEYS + ["value"]], on=KEYS, suffixes=("_before", "_after"), validate="one_to_one")
    if len(m) != len(b) or len(m) != len(e):
        raise SystemExit(f"baseline {metric}: righe non allineate")
    for c in ("value_before", "value_after"):
        if not np.isfinite(m[c]).all():
            print(f"[eq-sum] ATTENZIONE baseline {metric} {c}: {int((~np.isfinite(m[c])).sum())} coppie NaN "
                  "(fallite in faceBench), escluse dal bootstrap", flush=True)
    return m


def nocrop_reference(root: Path, metric: str, gt) -> float:
    """Spearman della baseline sulle 20 coppie senza crop della base, se le matrici ci sono."""
    pairs = common.setting_topology_pairs("nocrop_cross_topology")
    if not all(common.matrix_path(metric, a, b, root).exists() for a, b in pairs):
        return np.nan
    df = baseline_rows(root, metric, gt, pairs)
    df = df[np.isfinite(df["value"])]
    return float(zs._BM.finite_spearman(df["gt_distance"].to_numpy(), df["value"].to_numpy()))


# ------------------------------------------------------------------------- bootstrap

def run_tasks(tasks: list, workers: int) -> dict:
    print(f"[eq-sum] {len(tasks)} bootstrap appaiati su {workers} processi", flush=True)
    out = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for key, res in pool.imap_unordered(zs._boot_task, tasks):
            out[key] = res
    return out


def fmt(x: float) -> str:
    return "n/d" if not np.isfinite(x) else f"{x:.3f}"


def fmt_diff(r: dict) -> str:
    if not np.isfinite(r["diff"]):
        return "n/d"
    return f"{r['diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"


def main() -> None:
    args = parse_args()
    zs._BM = base.load_bootstrap_module()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt = zs.load_gt(args.gt)
    eq_manifest = (args.eq_dir / "manifest.json").read_text()

    frames = {}   # metrica -> frame con gt_distance, <col>_before, <col>_after
    checks, ident_checks, sources = [], [], []
    cham_eval = {}
    for variant in args.variants:
        for arm in args.arms:
            name = f"{arm}{variant}"
            pm_b, src_b = read_arm(args.runs, name)
            pm_e, src_e = read_arm(args.runs, f"{name}_eqsupport")
            man = src_e.parent / "eqsupport_manifest.json"
            if not man.exists() or man.read_text() != eq_manifest:
                raise SystemExit(f"{name}_eqsupport: eqsupport_manifest.json assente o diverso dalla vista attuale")
            sources.append({"arm": name, "prima": str(src_b), "dopo": str(src_e)})
            prot, chk = protocol_frame(pm_b, pm_e, ["latent_distance", "raw_chamfer"], name)
            checks.append(chk)
            ident_checks.append(original_is_crop_check(pm_e, f"{name}_eqsupport"))
            label = f"latente {zs.ARM_LABEL[arm]}" + ("" if variant == args.variants[0] and len(args.variants) == 1
                                                      else f", {VARIANT_LABEL.get(variant, variant)}")
            frames[label] = prot.rename(columns={"latent_distance_before": "before", "latent_distance_after": "after"})
            cham_eval[name] = prot
    # Chamfer eval: non dipende dal modello ne' dal frame; si prende dal primo braccio del primo frame,
    # e si controlla che gli altri bracci la riproducano sulle righe con crop della variante.
    first = f"{args.arms[0]}{args.variants[0]}"
    frames["Chamfer eval (repo)"] = cham_eval[first].rename(columns={"raw_chamfer_before": "before",
                                                                     "raw_chamfer_after": "after"})
    cham_rows = []
    for name, prot in cham_eval.items():
        m = prot.merge(cham_eval[first], on=KEYS, suffixes=("", "_ref"), validate="one_to_one")
        for col in ("raw_chamfer_before", "raw_chamfer_after"):
            d = (m[col] - m[f"{col}_ref"]).abs()
            rel = d / m[f"{col}_ref"].abs().clip(lower=1e-12)
            cham_rows.append({"arm": name, "colonna": col, "max_rel_diff": float(rel.max()),
                              "n_rows_rel_gt_1e-4": int((rel > 1e-4).sum()), "n_rows": len(m)})
    for metric in BL_METRICS:
        frames[BL_LABEL[metric]] = baseline_frame(args, gt, metric).rename(
            columns={"value_before": "before", "value_after": "after"})

    # soggetti fuori dal training del congiunto
    heldout_sets = {}
    if args.train_split is not None and "joint" in args.arms:
        train = set(json.loads(args.train_split.read_text())["models"]["joint"]["train"])
        subs = sorted(set(frames[next(iter(frames))]["subject_a"]) | set(frames[next(iter(frames))]["subject_b"]))
        to_name = (lambda s: f"id{int(s[2:]) + args.train_name_offset:04d}")
        heldout_sets["joint"] = [s for s in subs if to_name(s) not in train]
        print(f"[eq-sum] congiunto: {len(subs) - len(heldout_sets['joint'])}/{len(subs)} soggetti nel training, "
              f"{len(heldout_sets['joint'])} fuori", flush=True)

    tasks, layout = [], []
    for metric, df in frames.items():
        for group, sub in groups_of(df):
            key = (args.tag, metric, group)
            tasks.append((key, sub, ("after", "before"), args.n_bootstrap, base.stable_seed(args.seed, *key)))
            layout.append((metric, group, key))
        if metric.startswith("latente BFM+ICT") and "joint" in heldout_sets:
            keep = set(heldout_sets["joint"])
            sub = df[has_crop(df) & df["subject_a"].isin(keep) & df["subject_b"].isin(keep)]
            group = f"crop (tutte), {len(keep)} soggetti fuori dal training del congiunto"
            key = (args.tag, metric, group)
            tasks.append((key, sub, ("after", "before"), args.n_bootstrap, base.stable_seed(args.seed, *key)))
            layout.append((metric, group, key))
    res = run_tasks(tasks, args.workers)

    refs = {}
    for metric, df in frames.items():
        nc = df[~has_crop(df)]
        refs[metric] = (float(zs._BM.finite_spearman(nc["gt_distance"].to_numpy(), nc["before"].to_numpy()))
                        if len(nc) else np.nan)
    for metric in BL_METRICS:
        refs[BL_LABEL[metric]] = nocrop_reference(args.base_baselines, metric, gt)

    rows = []
    for metric, group, key in layout:
        r = res[key]
        rows.append({"dominio": args.label, "metrica": metric, "gruppo": group, "prima": r["b"], "dopo": r["a"],
                     "diff": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                     "p_boot_le0": r["p_boot_le0"], "rif_senza_crop": refs[metric],
                     "n_subjects": r["n_subjects"], "n_pairs": r["n_pairs"], "n_bootstrap": r["n_bootstrap"]})
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / f"{args.tag}_paired.csv", index=False)
    pd.DataFrame(checks).to_csv(args.out_dir / f"{args.tag}_checks.csv", index=False)

    region = pd.read_csv(args.eq_dir / "region_stats.csv")
    reg = (region.groupby("topology")
           .agg(soggetti=("subject", "nunique"), facce_tenute=("kept_face_frac", "median"),
                area_su_crop_mediana=("area_ratio_to_crop", "median"), area_su_crop_min=("area_ratio_to_crop", "min"),
                area_su_crop_max=("area_ratio_to_crop", "max"), eq_su_crop_p99=("eq_to_crop_p99", "median"),
                crop_su_eq_p99=("crop_to_eq_p99", "median"), crop_su_eq_max=("crop_to_eq_max", "max"),
                pavimento_p99=("full_to_original_p99", "median"))
           .reset_index())
    orig = region[region["topology"] == "original"]
    n_ident = int((orig["original_is_crop"].astype(str) == "True").sum())
    n_regdiff = int((orig["region_faces_of_original"] != orig["crop_faces"]).sum())

    md = [f"## {args.label}", ""]
    md.append(f"Prima = protocollo attuale, dopo = equalize-support (righe con crop dalla vista equalizzata). "
              f"Spearman con la GT del dominio (`{args.gt}`), mesh-pair, clean; differenza dopo - prima con CI 95% "
              f"bootstrap per soggetto appaiato ({args.n_bootstrap} repliche). `rif. senza crop`: Spearman della "
              f"stessa metrica sulle 20 coppie senza crop (base), il livello a cui il gap si misura.")
    md.append("")
    for metric in frames:
        sub = table[table["metrica"] == metric]
        md.append(f"### {metric}  (rif. senza crop {fmt(refs[metric])})")
        md.append("")
        md.append("| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |")
        md.append("| --- | --- | --- | --- | --- | --- |")
        for _, r in sub.iterrows():
            md.append(f"| {r['gruppo']} | {fmt(r['prima'])} | {fmt(r['dopo'])} | {fmt_diff(r)} | "
                      f"{r['n_subjects']} | {r['n_pairs']} |")
        md.append("")
    md += ["### Controlli", "",
           "Righe senza crop del protocollo contro la base (devono coincidere, differenza 0):", "",
           base.md_table(pd.DataFrame(checks), list(checks[0].keys()), floats=6), "",
           "Nella variante, (crop_a, original_b) contro (original_a, crop_b): la original equalizzata e' il "
           "crop, quindi le due righe sono la stessa coppia di crop:", "",
           base.md_table(pd.DataFrame(ident_checks), list(ident_checks[0].keys()), floats=8), "",
           "Chamfer eval nei bracci e nei frame contro il riferimento "
           f"`{first}` (deve coincidere: non dipende dal modello ne' da una rotazione):", "",
           base.md_table(pd.DataFrame(cham_rows), list(cham_rows[0].keys()), floats=8), "",
           f"Vista equalizzata `{args.eq_dir}`: original equalizzata identica al crop bit per bit in {n_ident}/"
           f"{len(orig)} soggetti; regione letta dal crop diversa dalle facce del crop in {n_regdiff} soggetti "
           "(BFM: vertici di bordo spostati da open3d, vedi eqsupport_view.py). Distanze in frazione della "
           "diagonale della original; `pavimento` = la stessa distanza fra la topologia intera e la original "
           "(rumore, smoothing), sotto la quale la regione non si puo' distinguere:", "",
           base.md_table(reg, list(reg.columns), floats=5), "",
           "Sorgenti:", "", base.md_table(pd.DataFrame(sources), ["arm", "prima", "dopo"]), ""]
    (args.out_dir / f"{args.tag}.md").write_text("\n".join(md) + "\n")
    print(f"[eq-sum] scritto {args.out_dir / f'{args.tag}.md'}", flush=True)


if __name__ == "__main__":
    main()
