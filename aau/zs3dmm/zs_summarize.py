#!/usr/bin/env python3
"""Tabella zero-shot su un 3DMM mai visto: tre modelli e le baseline geometriche, con CI bootstrap.

    aau/run.sh aau/zs3dmm/zs_summarize.py --label HIFI3D --runs aau/runs/ws_hifi3d \
        --gt <vista>/gt_matrix.npz --gt-coef <vista>/gt_coef_matrix.npz --root datasets/HIFI3D
    (zs_summarize.sbatch, WBES_ZS_DOMAIN=hifi|fv)

Non calcola nessuna distanza fra mesh: legge quello che hanno scritto gli script del repo --
``compare_model_vs_chamfer_rankings.py`` / ``..._topology_breakdown.py`` per i tre bracci di
``zs_zeroshot.sbatch`` e le matrici di ``zs_baselines.sbatch`` -- e ci mette i CI
bootstrap per soggetto con ``weighted_bootstrap_spearman`` del paper (via
``aau/ict/ict_summarize.py``, importata, come fa ws2_summarize.py).

Protocolli, gli stessi della tabella WS2 (``aau/runs/ws2_cross3dmm/summary.md``):
  - ``mesh_pair`` ``all_cross`` / ``nocrop_cross``: una osservazione per (coppia di soggetti,
    coppia ordinata di topologie), clean, dalle pair_metrics del breakdown;
  - ``subject_pair_mean`` clean (CI dalle pair_metrics mediate per coppia di soggetti, il punto
    deve coincidere con quello dello script, colonna ``point_check``) e mixed (solo punto).

Due GT, stesse coppie: ``maxabs`` (protocollo ICT, quella con cui girano gli script, colonna
``gt_distance``) e ``coef`` (distanza nei coefficienti di identita' standardizzati,
``gt_coef_matrix.npz`` della vista), sostituita per nome soggetto con lo stesso loader del repo.

Controlli scritti nel summary: la GT ha varianza non nulla e diagonale zero (stesso soggetto
-> distanza zero), i tre bracci e le baseline guardano gli stessi soggetti, il checkpoint
dichiarato dai json e' quello del braccio.

Differenze appaiate: per modello - Chamfer eval (stesse righe) e fra bracci (congiunto - ICT-only,
congiunto - BFM-only, righe allineate per soggetti e topologie), Spearman delle due colonne
ricalcolato sulle STESSE repliche bootstrap per soggetto, CI della differenza. Stessa
ricampionatura di ``weighted_bootstrap_spearman`` (``paired_bootstrap`` qui sotto la replica per
due colonne alla volta; la correlazione e' ``finite_spearman`` del modulo del paper).

``--variant <suffisso>`` (p.es. ``_frame-xmymz``, ``_evalframe-rms``): i bracci sono
``<arm><suffisso>`` e in piu' si stima, appaiato, l'effetto della variante (variante - base sullo
stesso braccio) e si verifica che la Chamfer eval sia la stessa della base riga per riga. Scrive
``summary<suffisso>.md`` e i csv col suffisso.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
RUNS = AAU_DIR / "runs"
sys.path.insert(0, str(AAU_DIR / "ict"))
sys.path.insert(0, str(AAU_DIR / "baselines"))
sys.path.insert(0, str(REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"))

import ict_summarize as base  # noqa: E402
import common  # noqa: E402  (solo I/O delle matrici e setting: legge env all'import, innocuo)
from intrinsic_utils import SUBJECT_RE_ANY, load_gt_distance_matrix  # noqa: E402

ARMS = ("joint", "bfm_only", "ict_only")
# bracci opzionali: entrano nelle tabelle SOLO se hanno risultati (senza, il summary e' quello di sempre)
OPTIONAL_ARMS = ("scale",)
ARM_LABEL = {"joint": "BFM+ICT", "bfm_only": "BFM-only", "ict_only": "ICT-only",
             "scale": "BFM+ICT+GNM (10^5)"}
ARM_CKPT_ENV = {"joint": "WBES_ZS_CKPT_JOINT", "bfm_only": "WBES_ZS_CKPT_BFM_ONLY",
                "ict_only": "WBES_ZS_CKPT_ICT_ONLY", "scale": "WBES_ZS_CKPT_SCALE"}


def arm_present(runs: Path, name: str) -> bool:
    return ((runs / name / STAGE / ".done").exists()
            or (runs / f"{name}_topology" / STAGE / ".done").exists())
STAGE = "zs_zeroshot"
TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")
BL_METRICS = ("chamfer", "rigid_icp_chamfer", "nicp_p2p", "nicp_p2tri")
BL_LABEL = {"chamfer": "Chamfer (faceBench, 4096 pt)", "rigid_icp_chamfer": "Rigid ICP + Chamfer",
            "nicp_p2p": "Rigid ICP + NICP + P2P", "nicp_p2tri": "Rigid ICP + NICP + P2Tri"}
BL_SETTINGS = ("original_to_original", "nocrop_cross_topology", "all_cross_topology")
GTS = ("maxabs", "coef")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="nome del dominio nel titolo, p.es. HIFI3D")
    p.add_argument("--runs", type=Path, required=True, help="$ZS_RUNS/data_<fp>: bracci e baseline")
    p.add_argument("--summary-dir", type=Path, required=True, help="dove scrivere summary.md e i csv")
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--gt-coef", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True, help="radice dei dati: identities/ e gt/ coi manifest")
    p.add_argument("--ws2", type=Path, default=RUNS / "ws2_cross3dmm" / "table_cells.csv")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234, help="Seme del ricampionamento, non del modello")
    p.add_argument("--variant", default="", help="suffisso delle out dir dei bracci, p.es. _frame-xmymz")
    return p.parse_args()


# ------------------------------------------------------------------------------ GT

def load_gt(path: Path) -> tuple[np.ndarray, dict]:
    D, name_to_idx = load_gt_distance_matrix(str(path), subject_re=SUBJECT_RE_ANY, dtype=np.float64)
    return np.asarray(D, dtype=np.float64), name_to_idx


def gt_checks(gts: dict) -> list[dict]:
    rows = []
    for tag, (D, idx) in gts.items():
        off = D[np.triu_indices(len(D), 1)]
        rows.append({"gt": tag, "n_subjects": len(D), "diag_max_abs": float(np.abs(np.diag(D)).max()),
                     "symmetric_max_abs": float(np.abs(D - D.T).max()),
                     "offdiag_min": float(off.min()), "offdiag_median": float(np.median(off)),
                     "offdiag_std": float(off.std()), "offdiag_max": float(off.max())})
    return rows


def with_gt(df: pd.DataFrame, gt: tuple[np.ndarray, dict]) -> pd.DataFrame:
    """Le stesse righe con ``gt_distance`` presa da un'altra matrice, per nome di soggetto."""
    D, idx = gt
    a = df["subject_a"].astype(str).map(idx)
    b = df["subject_b"].astype(str).map(idx)
    if a.isna().any() or b.isna().any():
        raise SystemExit("soggetti assenti dalla GT nei coefficienti")
    out = df.copy()
    out["gt_distance"] = D[a.to_numpy(int), b.to_numpy(int)]
    return out


# --------------------------------------------------------------------------- modelli

def arm_sources(runs: Path, arm: str) -> dict:
    """Dove stanno ranking e breakdown di un braccio.

    ``<arm>/zs_zeroshot/.done`` (WBES_ZS_PART=all): entrambi da li'. Altrimenti i due pezzi di
    WBES_ZS_PART=ranking/topology, ognuno col suo .done. Unica eccezione, per i job FaceVerse
    partiti come ``all`` e fermati dopo il ranking (il breakdown e' rifatto come pezzo
    ``topology``): il ranking di ``<arm>/`` vale se lo script ha scritto tutti e tre i suoi file
    (json, csv, md: il json e' il primo, quindi un kill a meta' lascerebbe mancare gli altri).
    La sorgente usata finisce nel summary.
    """
    full = runs / arm / STAGE
    if (full / ".done").exists():
        return {"ranking": full / "ranking", "topology": full, "source": f"{arm}/ (completo)"}
    out = {}
    for part in ("ranking", "topology"):
        stage = runs / f"{arm}_{part}" / STAGE
        if (stage / ".done").exists():
            out[part] = stage / "ranking" if part == "ranking" else stage
    if "ranking" not in out:
        r = full / "ranking"
        if all((r / f"ranking_summary.{ext}").exists() for ext in ("json", "csv", "md")):
            out["ranking"] = r
    if "topology" not in out:
        raise SystemExit(f"{arm}: manca il breakdown (ne' {full}/.done ne' {arm}_topology/)")
    if "ranking" not in out:
        # Varianti girate col solo breakdown (FaceVerse, per le ore GPU): il clean del ranking si
        # ricalcola esatto dalle pair_metrics (point_check), il mixed non c'e'.
        out["ranking"] = None
        out["source"] = f"solo breakdown da {out['topology'].relative_to(runs)} (niente ranking: punto clean dalle pair_metrics, niente mixed)"
        return out
    out["source"] = (f"ranking da {out['ranking'].relative_to(runs)}, "
                     f"breakdown da {out['topology'].relative_to(runs)}")
    return out


# Bootstrap in due passate: la prima (``_STATE["collect"]``) raccoglie i compiti senza calcolare
# niente, ``run_boot_tasks`` li esegue in parallelo, la seconda rilegge i risultati. Seriale, il
# bootstrap degli aggregati (60 x 1000 ricampionamenti su fino a 148.500 righe) stava ben oltre
# l'ora. Seme e funzione sono quelli di ws2_summarize.py: cambia solo dove gira.
_STATE = {"collect": False, "tasks": [], "cache": {}}
_BM = None


def boot(df: pd.DataFrame, value_col: str, args, bm, *tokens: str, n_bootstrap: int | None = None) -> dict:
    n = args.n_bootstrap if n_bootstrap is None else n_bootstrap
    key = (value_col, n) + tokens
    if key in _STATE["cache"]:
        return _STATE["cache"][key]
    seed = base.stable_seed(args.seed, *tokens)
    if _STATE["collect"]:
        _STATE["tasks"].append((key, df, value_col, n, seed))
        return {"spearman": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_subjects": 0, "n_pairs": 0,
                "n_bootstrap": n}
    return base.bootstrap_row(df, value_col, n, np.random.default_rng(seed), bm)


def paired_bootstrap(df: pd.DataFrame, col_a: str, col_b: str, n_bootstrap: int, rng, bm) -> dict:
    """Spearman(gt, a) - Spearman(gt, b) con CI, sulle stesse repliche bootstrap per soggetto.

    Ricampionamento identico a ``weighted_bootstrap_spearman`` (scripts/compute_bootstrap_ci.py):
    soggetti estratti con reinserimento, peso di una coppia = prodotto dei conteggi dei suoi due
    soggetti, righe ripetute per il peso. Cambia solo che le colonne sono due.
    """
    w = df.loc[:, ["subject_a", "subject_b", "gt_distance", col_a, col_b]].copy()
    for c in ("gt_distance", col_a, col_b):
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[np.isfinite(w["gt_distance"]) & np.isfinite(w[col_a]) & np.isfinite(w[col_b])]
    w = w[w["subject_a"].astype(str) != w["subject_b"].astype(str)]
    subjects = np.array(sorted(set(w["subject_a"].astype(str)) | set(w["subject_b"].astype(str))))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa = w["subject_a"].astype(str).map(s2i).to_numpy(np.int32)
    sb = w["subject_b"].astype(str).map(s2i).to_numpy(np.int32)
    gt, a, b = (w[c].to_numpy(np.float64) for c in ("gt_distance", col_a, col_b))
    pa, pb = bm.finite_spearman(gt, a), bm.finite_spearman(gt, b)
    diffs = []
    for _ in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, len(subjects), size=len(subjects)), minlength=len(subjects))
        wt = counts[sa].astype(np.int64) * counts[sb].astype(np.int64)
        keep = wt > 0
        if int(keep.sum()) < 3:
            continue
        x = np.repeat(gt[keep], wt[keep])
        d = bm.finite_spearman(x, np.repeat(a[keep], wt[keep])) - bm.finite_spearman(x, np.repeat(b[keep], wt[keep]))
        if np.isfinite(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (np.nan, np.nan))
    return {"a": pa, "b": pb, "diff": pa - pb, "ci_low": float(lo), "ci_high": float(hi),
            "p_boot_le0": float((diffs <= 0).mean()) if len(diffs) else np.nan,
            "n_subjects": len(subjects), "n_pairs": len(w), "n_bootstrap": len(diffs)}


def pboot(df: pd.DataFrame, col_a: str, col_b: str, args, bm, *tokens: str) -> dict:
    key = ("paired", col_a, col_b, args.n_bootstrap) + tokens
    if key in _STATE["cache"]:
        return _STATE["cache"][key]
    seed = base.stable_seed(args.seed, "paired", *tokens)
    if _STATE["collect"]:
        _STATE["tasks"].append((key, df, (col_a, col_b), args.n_bootstrap, seed))
        return {"a": np.nan, "b": np.nan, "diff": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_boot_le0": np.nan, "n_subjects": 0, "n_pairs": 0, "n_bootstrap": 0}
    return paired_bootstrap(df, col_a, col_b, args.n_bootstrap, np.random.default_rng(seed), bm)


def _boot_task(task):
    global _BM
    if _BM is None:
        _BM = base.load_bootstrap_module()
    key, df, value_col, n, seed = task
    rng = np.random.default_rng(seed)
    if isinstance(value_col, tuple):
        return key, paired_bootstrap(df, value_col[0], value_col[1], n, rng, _BM)
    return key, base.bootstrap_row(df, value_col, n, rng, _BM)


def run_boot_tasks(n_workers: int) -> None:
    import multiprocessing as mp

    tasks = _STATE["tasks"]
    print(f"[zs-sum] {len(tasks)} bootstrap su {n_workers} processi", flush=True)
    with mp.get_context("fork").Pool(n_workers) as pool:
        for key, res in pool.imap_unordered(_boot_task, tasks):
            _STATE["cache"][key] = res
    _STATE["tasks"] = []


def arm_tables(args, bm, gts, variant: str = "") -> tuple:
    cells, topo, subjects, sources, pms = [], [], {}, {}, {}
    for arm in ARMS:
        name = arm + variant
        src = arm_sources(args.runs, name)
        sources[arm] = src["source"]
        want = os.environ.get(ARM_CKPT_ENV[arm])
        payload = None
        if src["ranking"] is not None:
            payload = json.loads((src["ranking"] / "ranking_summary.json").read_text())
            if want and os.path.realpath(payload["checkpoint"]) != os.path.realpath(want):
                raise SystemExit(f"{arm}: checkpoint {payload['checkpoint']} != {want}")
        key = dict(l.split("=", 1) for l in (src["topology"].parent / "eval_key.txt").read_text().splitlines()
                   if "=" in l)
        if want and os.path.realpath(key["ckpt"]) != os.path.realpath(want):
            raise SystemExit(f"{arm}: breakdown col checkpoint {key['ckpt']} != {want}")
        ranking = {} if payload is None else {row["scenario"]: row for row in payload["rows"]}
        pm_max = base.read_pair_metrics(src["topology"])  # legge da se' <stage>/topology/*/pair_metrics.csv
        seen = sorted(set(pm_max["subject_a"].astype(str)) | set(pm_max["subject_b"].astype(str)))
        subjects[arm] = seen if payload is None else sorted(set(payload["selected_subjects"]))
        staged_file = next(p for p in (args.runs / name / "subjects.json",
                                       args.runs / f"{name}_topology" / "subjects.json") if p.exists())
        staged = json.loads(staged_file.read_text())["subjects"]
        if seen != subjects[arm] or sorted(staged) != subjects[arm]:
            raise SystemExit(f"{name}: pair_metrics, ranking e zs_stage guardano soggetti diversi")
        keys = ["subject_a", "subject_b", "topology_a", "topology_b"]
        if pm_max.duplicated(keys).any():
            raise SystemExit(f"{name}: righe duplicate in pair_metrics per {keys}")
        pms[arm] = pm_max

        for gt_tag in GTS:
            pm = pm_max if gt_tag == "maxabs" else with_gt(pm_max, gts["coef"])
            per_sp = (pm.groupby(["subject_a", "subject_b"], as_index=False)
                      [["gt_distance", "latent_distance", "raw_chamfer"]].mean())
            for scenario in ("clean", "mixed"):
                if scenario == "mixed" and (gt_tag != "maxabs" or "mixed" not in ranking):
                    continue  # il mixed: solo punto dello script, sulla GT dello script, se c'e' (rms: no)
                row = ranking.get(scenario)
                out = {"model": arm, "gt": gt_tag, "protocol": "subject_pair_mean", "scenario": scenario,
                       "n_subjects": len(seen) if row is None else int(row["n_subjects"]),
                       "n_mesh_pairs": len(pm_max) if row is None else int(row["n_mesh_pairs"])}
                for metric, col in base.METRICS:
                    if scenario == "clean":
                        b = boot(per_sp, col, args, bm, arm, gt_tag, "spm_clean", metric)
                        point = (float(row[f"{metric}_spearman"]) if gt_tag == "maxabs" and row is not None
                                 else b["spearman"])
                        out.update({f"{metric}_spearman": point, f"{metric}_ci_low": b["ci_low"],
                                    f"{metric}_ci_high": b["ci_high"], f"{metric}_point_check": b["spearman"]})
                    else:
                        out.update({f"{metric}_spearman": float(row[f"{metric}_spearman"]),
                                    f"{metric}_ci_low": np.nan, f"{metric}_ci_high": np.nan,
                                    f"{metric}_point_check": np.nan})
                cells.append(out)

            groups = [(f"{a}__to__{b}", sub) for (a, b), sub in pm.groupby(["topology_a", "topology_b"], sort=True)]
            groups.append(("all_cross", pm))
            groups.append(("nocrop_cross", pm[pm["topology_a"].ne("crop") & pm["topology_b"].ne("crop")]))
            for group, sub in groups:
                # CI solo sugli aggregati: le 30 celle per coppia di topologie vanno in tabella
                # come punto, e un bootstrap ciascuna costerebbe 30x il resto.
                n_b = None if group in ("all_cross", "nocrop_cross") else 0
                res = {metric: boot(sub, col, args, bm, arm, gt_tag, group, metric, n_bootstrap=n_b)
                       for metric, col in base.METRICS}
                for metric in res:
                    topo.append({"model": arm, "gt": gt_tag, "topology_pair": group, "metric": metric, **res[metric]})
                if group in ("all_cross", "nocrop_cross"):
                    out = {"model": arm, "gt": gt_tag, "protocol": f"mesh_pair_{group}", "scenario": "clean",
                           "n_subjects": res["latent"]["n_subjects"], "n_mesh_pairs": res["latent"]["n_pairs"]}
                    for metric in res:
                        out.update({f"{metric}_spearman": res[metric]["spearman"],
                                    f"{metric}_ci_low": res[metric]["ci_low"],
                                    f"{metric}_ci_high": res[metric]["ci_high"], f"{metric}_point_check": np.nan})
                    cells.append(out)
                    if not _STATE["collect"]:
                        print(f"[zs-sum] {arm:<9} gt={gt_tag:<6} {group:<12} "
                              f"latent={res['latent']['spearman']:.4f} "
                              f"[{res['latent']['ci_low']:.3f},{res['latent']['ci_high']:.3f}] "
                              f"chamfer={res['chamfer']['spearman']:.4f} n_pairs={res['latent']['n_pairs']}",
                              flush=True)
    table = pd.DataFrame(cells)
    table["delta_spearman"] = table["latent_spearman"] - table["chamfer_spearman"]
    return table, pd.DataFrame(topo), subjects, sources, pms


# ---------------------------------------------------------------- differenze appaiate

PAIR_KEYS = ["subject_a", "subject_b", "topology_a", "topology_b"]
SCENARIOS = ("all_cross", "nocrop_cross", "subject_pair_mean")


def scenario_frame(df: pd.DataFrame, scenario: str, cols: list[str]) -> pd.DataFrame:
    if scenario == "all_cross":
        return df
    if scenario == "nocrop_cross":
        return df[df["topology_a"].ne("crop") & df["topology_b"].ne("crop")]
    return df.groupby(["subject_a", "subject_b"], as_index=False)[["gt_distance"] + cols].mean()


def merge_arms(pa: pd.DataFrame, pb: pd.DataFrame, col: str, ta: str, tb: str) -> pd.DataFrame:
    """Righe allineate (stessi soggetti, stesse topologie) con ``col`` dei due lati in ``<col>_<ta>``/``_<tb>``."""
    m = pa[PAIR_KEYS + ["gt_distance", col]].merge(pb[PAIR_KEYS + ["gt_distance", col]], on=PAIR_KEYS,
                                                   suffixes=(f"_{ta}", f"_{tb}"), validate="one_to_one")
    if len(m) != len(pa) or len(m) != len(pb):
        raise SystemExit(f"{ta} / {tb}: righe non allineate ({len(pa)}, {len(pb)}, comuni {len(m)})")
    if not np.allclose(m[f"gt_distance_{ta}"], m[f"gt_distance_{tb}"]):
        raise SystemExit(f"{ta} / {tb}: GT diversa sulle stesse righe")
    return m.rename(columns={f"gt_distance_{ta}": "gt_distance"}).drop(columns=[f"gt_distance_{tb}"])


def paired_tables(args, bm, gts, pms: dict, base_pms: dict | None) -> pd.DataFrame:
    rows = []
    for gt_tag in GTS:
        conv = (lambda d: d) if gt_tag == "maxabs" else (lambda d: with_gt(d, gts["coef"]))
        specs = []  # (confronto, frame con le due colonne, col_a, col_b, colonne da mediare)
        for arm in ARMS:
            specs.append((f"{ARM_LABEL[arm]} - Chamfer eval", conv(pms[arm]), "latent_distance", "raw_chamfer",
                          ["latent_distance", "raw_chamfer"]))
        for x, y in (("joint", "ict_only"), ("joint", "bfm_only")):
            m = conv(merge_arms(pms[x], pms[y], "latent_distance", x, y))
            specs.append((f"{ARM_LABEL[x]} - {ARM_LABEL[y]}", m, f"latent_distance_{x}", f"latent_distance_{y}",
                          [f"latent_distance_{x}", f"latent_distance_{y}"]))
        if base_pms is not None:
            for arm in ARMS:
                m = conv(merge_arms(pms[arm], base_pms[arm], "latent_distance", "var", "base"))
                specs.append((f"{ARM_LABEL[arm]}: variante - base", m, "latent_distance_var", "latent_distance_base",
                              ["latent_distance_var", "latent_distance_base"]))
        for name, df, ca, cb, cols in specs:
            for scenario in SCENARIOS:
                r = pboot(scenario_frame(df, scenario, cols), ca, cb, args, bm, args.variant, gt_tag, name, scenario)
                rows.append({"gt": gt_tag, "comparison": name, "scenario": scenario, **r})
    return pd.DataFrame(rows)


def chamfer_vs_base(pms: dict, base_pms: dict) -> pd.DataFrame:
    """Controllo di sanita' della variante: la Chamfer eval non deve cambiare, riga per riga."""
    rows = []
    for arm in ARMS:
        m = merge_arms(pms[arm], base_pms[arm], "raw_chamfer", "var", "base")
        d = (m["raw_chamfer_var"] - m["raw_chamfer_base"]).abs()
        rel = d / m["raw_chamfer_base"].abs().clip(lower=1e-12)
        rows.append({"model": arm, "n_rows": len(m), "max_abs_diff": float(d.max()),
                     "max_rel_diff": float(rel.max()), "n_rows_rel_gt_1e-4": int((rel > 1e-4).sum())})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- baseline

def baseline_table(args, bm, gts) -> tuple[pd.DataFrame, list[str]]:
    root = args.runs / "baselines"
    rows, bl_subjects = [], None
    if not root.is_dir():
        return pd.DataFrame(), None  # dominio "ict": le baseline ICT esistono gia' (aau/runs/outlineB)
    for metric in BL_METRICS:
        for setting in BL_SETTINGS:
            frames = []
            for ta, tb in common.setting_topology_pairs(setting):
                path = common.matrix_path(metric, ta, tb, root)
                if not path.exists():
                    # Baseline incomplete (p.es. un altro job le sta ancora scrivendo): tabella dei
                    # modelli senza baseline, invece di fallire.
                    print(f"[zs-sum] ATTENZIONE: baseline incomplete ({path} manca): sezione saltata", flush=True)
                    return pd.DataFrame(), None
                D, subj, _, _, _ = common.load_matrix(path)
                if bl_subjects is None:
                    bl_subjects = subj
                elif subj != bl_subjects:
                    raise SystemExit(f"{path}: soggetti diversi dalle altre matrici")
                i, j = common.subject_pair_indices(len(subj))
                names = np.asarray(subj)
                frames.append(pd.DataFrame({"subject_a": names[i], "subject_b": names[j],
                                            "topology_a": ta, "topology_b": tb, "gt_distance": 0.0,
                                            metric: D[i, j]}))
            df = pd.concat(frames, ignore_index=True)
            for gt_tag in GTS:
                b = boot(with_gt(df, gts[gt_tag]), metric, args, bm, "bl", metric, setting, gt_tag)
                rows.append({"metric": metric, "setting": setting, "gt": gt_tag,
                             "n_nan": int(df[metric].isna().sum()), **b})
                if not _STATE["collect"]:
                    print(f"[zs-sum] bl {metric:<18} {setting:<22} gt={gt_tag:<6} "
                          f"{b['spearman']:.4f} [{b['ci_low']:.3f},{b['ci_high']:.3f}] n={b['n_pairs']}", flush=True)
    return pd.DataFrame(rows), bl_subjects


# --------------------------------------------------------------------------- markdown

def fmt_ci(point: float, low: float, high: float) -> str:
    if not np.isfinite(low):
        return f"{point:.3f}"
    return f"{point:.3f} [{low:.2f}, {high:.2f}]"


def model_grid(table: pd.DataFrame, protocol: str, scenario: str) -> str:
    lines = ["| modello | latent (GT maxabs) | chamfer eval (GT maxabs) | latent (GT coef) | chamfer eval (GT coef) |",
             "| --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        cells = [ARM_LABEL[arm]]
        for gt_tag in GTS:
            sel = table[(table["model"] == arm) & (table["gt"] == gt_tag) & (table["protocol"] == protocol)
                        & (table["scenario"] == scenario)]
            if sel.empty:
                cells += ["-", "-"]
                continue
            r = sel.iloc[0]
            cells += [fmt_ci(r.latent_spearman, r.latent_ci_low, r.latent_ci_high),
                      fmt_ci(r.chamfer_spearman, r.chamfer_ci_low, r.chamfer_ci_high)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ws2_reference(path: Path) -> str:
    """Le stesse celle su ICT (tabella WS2), per leggere il dominio nuovo accanto a uno visto/non visto."""
    if not path.exists():
        return "(tabella WS2 non trovata)"
    t = pd.read_csv(path)
    t = t[t["domain"].eq("ict") & t["scenario"].eq("clean")]
    lines = ["| modello | ICT mesh-pair all cross, latent | ICT nocrop, latent | ICT subject-pair-mean, latent |",
             "| --- | --- | --- | --- |"]
    for arm in ARMS:
        cells = [ARM_LABEL[arm]]
        for proto in ("mesh_pair_all_cross", "mesh_pair_nocrop_cross", "subject_pair_mean"):
            sel = t[(t["model"] == arm) & (t["protocol"] == proto)]
            cells.append("-" if sel.empty else fmt_ci(sel.iloc[0].latent_spearman, sel.iloc[0].latent_ci_low,
                                                     sel.iloc[0].latent_ci_high))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def topology_matrix(topo: pd.DataFrame, arm: str, metric: str) -> str:
    sub = topo[(topo["model"] == arm) & (topo["metric"] == metric) & (topo["gt"] == "maxabs")]
    value = {r.topology_pair: r.spearman for r in sub.itertuples()}
    lines = ["| A \\ B | " + " | ".join(TOPOLOGIES) + " |", "| --- |" + " --- |" * len(TOPOLOGIES)]
    for a in TOPOLOGIES:
        cells = [a] + ["-" if a == b else f"{value.get(f'{a}__to__{b}', np.nan):.3f}" for b in TOPOLOGIES]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_md(paired: pd.DataFrame, comparisons: list[str]) -> str:
    lines = ["| confronto | scenario | GT maxabs: differenza [CI 95%] (a / b) | P(boot <= 0) | GT coef: differenza [CI 95%] |",
             "| --- | --- | --- | --- | --- |"]
    for name in comparisons:
        for scenario in SCENARIOS:
            r = {g: paired[(paired["comparison"] == name) & (paired["scenario"] == scenario) & (paired["gt"] == g)]
                 for g in GTS}
            if r["maxabs"].empty:
                continue
            m, c = r["maxabs"].iloc[0], r["coef"].iloc[0]
            lines.append(f"| {name} | {scenario} | {m['diff']:+.3f} [{m['ci_low']:+.3f}, {m['ci_high']:+.3f}] "
                         f"({m['a']:.3f} / {m['b']:.3f}) | {m['p_boot_le0']:.3f} | "
                         f"{c['diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] |")
    return "\n".join(lines)


def write_markdown(path: Path, table, topo, bl, checks, meta, args, paired, cham=None) -> None:
    title = f"# Zero-shot su un 3DMM mai visto: {args.label}" + (f", variante `{args.variant}`" if args.variant else "")
    parts = [
        f"{title}\n",
        f"Modelli: BFM+ICT congiunto (job 1019532), BFM-only (1019310), ICT-only (1019531): gli stessi "
        f"della tabella WS2, seed 1234, ricetta v1, operatori ad area unitaria. Nessuno ha visto {args.label}. "
        f"Dati: {meta['n_identities']} identita' dal 3DMM `{meta['model_file']}` "
        f"(sha256 {meta['model_sha256'][:12]}..), {meta['n_shape']} modi, z ~ N(0,1) senza troncamento, "
        f"seed {meta['seed']}; scala: {meta['model']['scaling']}. Regione: "
        f"{meta['model'].get('region', '`mask_face` del .mat')}, "
        f"{meta['n_verts']} vertici / {meta['n_faces']} triangoli (modello intero "
        f"{meta['model']['n_verts_head']} vertici). Stesse 6 topologie e stessi operatori di ICT "
        f"(calcolati su /tmp nel job di eval, per i soli soggetti valutati). "
        f"Valutati {meta['n_eval']} soggetti estratti dal pool con `rebuild_subject_split` seed 1234, "
        f"gli stessi per i tre modelli e per le baseline. CI 95% bootstrap per soggetto, "
        f"{args.n_bootstrap} ricampionamenti. Risultati da `{args.runs}`.\n",
        "GT `maxabs` = protocollo ICT (vertex-mean-L2 fra le `original` dopo la normalizzazione maxabs, "
        "quella di `datasets/ICT/train_ready/gt_matrix.npz`). GT `coef` = distanza L2 fra i coefficienti di "
        f"identita' standardizzati z. Spearman fra le due GT sul pool: {meta['spearman_coef_vs_maxabs']:.3f}.\n",
        "## Mesh-pair cross-topology, clean (protocollo dello 0.301 zero-shot)\n",
        model_grid(table, "mesh_pair_all_cross", "clean"),
        "\n### Senza crop (colonna della Tabella 2 del paper)\n",
        model_grid(table, "mesh_pair_nocrop_cross", "clean"),
        "\n## Subject-pair-mean (script di ranking), clean\n",
        model_grid(table, "subject_pair_mean", "clean"),
        "\n## Subject-pair-mean, mixed (solo punto, solo GT dello script)\n",
        model_grid(table, "subject_pair_mean", "mixed"),
        "\n## Differenze appaiate (stesse repliche bootstrap per soggetto)\n",
        "Differenza degli Spearman con la GT, a - b, CI 95% sulle stesse 1000 repliche; P(boot <= 0) e' la "
        "frazione di repliche con differenza <= 0. Scenari: `all_cross` = 30 coppie ordinate di topologie, "
        "`nocrop_cross` = 20, `subject_pair_mean` = media per coppia di soggetti sulle 30 (clean).\n",
        paired_md(paired, [c for c in dict.fromkeys(paired["comparison"]) if "variante" not in c]),
    ]
    if args.variant:
        parts += [f"\n## Effetto della variante `{args.variant}` (variante - base, stesso braccio, appaiato)\n",
                  paired_md(paired, [c for c in dict.fromkeys(paired["comparison"]) if "variante" in c]),
                  "\nControllo di sanita': Chamfer eval della variante contro quella della base, riga per riga "
                  "(deve essere identica: la Chamfer non dipende dal frame ne' dal ri-inquadramento del modello).\n",
                  base.md_table(cham, list(cham.columns), floats=6)]
    parts += ["\n## Riferimento: le stesse celle su ICT (tabella WS2)\n", ws2_reference(args.ws2)]
    if bl is not None and not bl.empty:
        parts.append("\n## Baseline geometriche (pipeline faceBench, stessi soggetti)\n")
        if args.variant:
            parts.append("Le baseline sono quelle della base: allineamento e Chamfer su mesh normalizzate maxabs, "
                         "invarianti per il cambio di frame (stessa trasformazione sulle due mesh).\n")
        lines = ["| metodo | setting | GT maxabs | GT coef | NaN |", "| --- | --- | --- | --- | --- |"]
        for metric in BL_METRICS:
            for setting in BL_SETTINGS:
                r = {g: bl[(bl["metric"] == metric) & (bl["setting"] == setting) & (bl["gt"] == g)].iloc[0] for g in GTS}
                lines.append(f"| {BL_LABEL[metric]} | {setting} | {fmt_ci(r['maxabs'].spearman, r['maxabs'].ci_low, r['maxabs'].ci_high)} | "
                             f"{fmt_ci(r['coef'].spearman, r['coef'].ci_low, r['coef'].ci_high)} | {r['maxabs'].n_nan} |")
        parts.append("\n".join(lines))
        parts.append("\n`chamfer eval` nelle tabelle dei modelli e' la Chamfer degli script di eval (media delle "
                     "distanze al quadrato su tutti i vertici); la riga `Chamfer (faceBench)` qui sopra e' quella "
                     "della Tabella 2 (4096 punti campionati). `all_cross_topology` = le 30 coppie ordinate cross.\n")
    parts.append("\n## Controlli\n")
    parts.append(base.md_table(pd.DataFrame(checks), list(checks[0].keys()), floats=4))
    parts.append(f"\nSoggetti: identici nei tre bracci (e nelle baseline, se ci sono) = {meta['same_subjects']} "
                 f"({meta['n_eval']}, primi {', '.join(meta['subjects'][:3])}).\n")
    parts.append("\nSorgenti dei bracci: " + "; ".join(f"{ARM_LABEL[a]}: {meta['sources'][a]}" for a in ARMS)
                 + ".\n")
    parts.append("\n## Matrici per coppia di topologie (clean, mesh-pair, GT maxabs; righe A, colonne B)\n")
    parts.append("SOLO PUNTO, senza CI: per le 30 celle n_bootstrap=0 (i CI sono sugli aggregati qui sopra).\n")
    for arm in ARMS:
        parts.append(f"\n### {ARM_LABEL[arm]}, latent\n")
        parts.append(topology_matrix(topo, arm, "latent"))
    parts.append("\n### Chamfer eval (uguale per i tre bracci a meno del campione: si riporta il congiunto)\n")
    parts.append(topology_matrix(topo, "joint", "chamfer"))
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    global ARMS
    args = parse_args()
    extra = tuple(a for a in OPTIONAL_ARMS if arm_present(args.runs, a + (args.variant or "")))
    if extra:
        ARMS = ARMS + extra
        print(f"[zs-sum] bracci opzionali con risultati: {extra}", flush=True)
    bm = base.load_bootstrap_module()
    gts = {"maxabs": load_gt(args.gt), "coef": load_gt(args.gt_coef)}
    checks = gt_checks(gts)
    for c in checks:
        print(f"[zs-sum] GT {c}", flush=True)
        if c["diag_max_abs"] != 0.0 or not c["offdiag_std"] > 0 or not c["offdiag_min"] > 0:
            raise SystemExit(f"GT non valida: {c}")

    _STATE["collect"] = True
    # Della base servono solo le pair_metrics: in modalita' collect arm_tables non calcola niente,
    # e i compiti raccolti per la base si buttano.
    base_pms = arm_tables(args, bm, gts, "")[4] if args.variant else None
    _STATE["tasks"] = []
    pms = arm_tables(args, bm, gts, args.variant)[4]
    baseline_table(args, bm, gts)
    paired_tables(args, bm, gts, pms, base_pms)
    _STATE["collect"] = False
    run_boot_tasks(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    table, topo, subjects, sources, pms = arm_tables(args, bm, gts, args.variant)
    bl, bl_subjects = baseline_table(args, bm, gts)
    paired = paired_tables(args, bm, gts, pms, base_pms)
    cham = chamfer_vs_base(pms, base_pms) if base_pms is not None else None
    same = all(subjects[a] == subjects["joint"] for a in ARMS) and (
        bl_subjects is None or sorted(bl_subjects) == subjects["joint"])

    ident = json.loads((args.root / "identities" / "manifest.json").read_text())
    gtman = json.loads((args.root / "gt" / "manifest.json").read_text())
    meta = {**ident, "n_eval": len(subjects["joint"]), "subjects": subjects["joint"], "same_subjects": same,
            "sources": sources,
            "spearman_coef_vs_maxabs": gtman["spearman_coef_vs_maxabs"]}

    out = args.summary_dir
    out.mkdir(parents=True, exist_ok=True)
    v = args.variant
    table.to_csv(out / f"table_cells{v}.csv", index=False)
    topo.to_csv(out / f"topology_pairs{v}.csv", index=False)
    paired.to_csv(out / f"paired{v}.csv", index=False)
    if cham is not None:
        cham.to_csv(out / f"chamfer_vs_base{v}.csv", index=False)
        print(f"[zs-sum] Chamfer variante vs base:\n{cham.to_string(index=False)}", flush=True)
    if bl is not None and not bl.empty:
        bl.to_csv(out / f"baselines{v}.csv", index=False)
    pd.DataFrame(checks).to_csv(out / f"gt_checks{v}.csv", index=False)
    write_markdown(out / f"summary{v}.md", table, topo, bl, checks, meta, args, paired, cham)
    print(f"[zs-sum] soggetti identici fra bracci e baseline: {same}", flush=True)
    print(f"[zs-sum] scritto {out / f'summary{v}.md'}", flush=True)
    if not same:
        raise SystemExit("i tre bracci e le baseline non guardano gli stessi soggetti")


if __name__ == "__main__":
    main()
