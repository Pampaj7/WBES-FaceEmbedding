#!/usr/bin/env python3
"""Collaudo della catena confermativa di Ava-256 su HIFI3D (dominio di sviluppo): le uscite dello STESSO codice da spec
(``spec/hifi3d_full.json``, ``spec/hifi3d_det.json``) contro i risultati di sviluppo gia' in git. Nessun dato di
Ava-256.

    v3_work/unified_gt/run.sh aau/ava256/conf_collaudo.py <passo> [--workers 64]     (conf.sbatch, CONF_STEPS)

Passi (ognuno scrive ``<evidence>/collaudo/<passo>.json`` con gli scarti misurati e ``pass`` contro le tolleranze
``TOL``, fissate prima di eseguire):
  export-template  template di HIFI3D (``domains.template``) in un npz, ingresso di ``pr_landmarks define --spec``.
  landmarks   definizione (indici 68 + 85, etichette di regione) e prep dello spec contro ``landmarks/hifi3d.json``,
              ``aau/runs/protocolli/defs/hifi3d.npz`` e ``prep/hifi3d.npz``.
  det         render, punti 2D, ripieghi e landmark del rilevatore delle 600 mesh e ``D_mm`` dei 4 protocolli sulle
              11.880 coppie del sottocampione contro ``noise/det/hifi3d``, ``noise/prep/hifi3d_det.npz`` e i plug-in
              ``competitors_new/<metodo>/lmk_det/hifi3d.npz`` (job 1071153).
  eval-full   ``conf_eval.py`` (hifi3d_full, seme della T1 990708) contro ``full_matrix`` (Spearman e delta: versione
              mascherata = passo ``comune``, lettura confermativa +inf = passo "NICP fallito = distanza massima") e contro
              ``protocolli`` (R7 = -(NoW originale - NoW metrical), Spearman dei 4 protocolli, landmark esatti).
  eval-det    ``conf_eval.py`` (hifi3d_det) contro ``rumore_landmark``: Spearman dei 4 protocolli col rilevatore e
              NoW metrical - d_F non cal. s1234 (delta "protocollo - nostro").
  blmm        ``blmm_scalars.py`` e ``blmm_template.py`` con ``--spec`` contro ``baselines_mm/hifi3d`` (scalari, L_d,
              cs_ref, R del NICP su template mm e cs), ``blmm.pair_metrics`` sulla vista da spec per ``N_PAIRS``
              coppie (Chamfer + ICP) e ``N_NICP`` (NICP per coppia) per modo contro le matrici.
  bp          ``bp_fit.py``, ``bp_fit_e1.py`` e ``bp_e2.py crop`` con ``--spec`` contro ``baselines_param/hifi3d``:
              regione, beta e matrici della variante B (``D_vb_*``) senza crop e col crop.
  head        ``conf_head.py block`` / ``fit`` (K0) dagli json di ``<evidence>/head``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(AAU / "competitors" / "protocolli"))
import conf_common as cc  # noqa: E402

SPEC_FULL = THIS / "spec" / "hifi3d_full.json"
SPEC_DET = THIS / "spec" / "hifi3d_det.json"
EV = AAU / "runs" / "evidence"
# tolleranze, fissate PRIMA di eseguire: 0 = uguale; "num" = stesso calcolo sugli stessi ingressi (solo l'ordine delle
# righe cambia); "geom" = calcolo geometrico rifatto (il controllo di sviluppo di pr_noise.check: <= 1e-4 mm su REALY)
TOL = {"exact": 0.0, "num": 1e-12, "geom_mm": 1e-4, "rel": 1e-9, "spearman_geom": 1e-4}
N_PAIRS, N_NICP = 1000, 100
SENS = "sensibilita': NICP fallito = distanza massima"


def out_path(step: str) -> Path:
    return EV / "ava256_conf" / "collaudo" / f"{step}.json"


def finish(step: str, rep: dict) -> None:
    rep.update(tolerances=TOL, **cc.stamp())
    cc.atomic_json(out_path(step), rep)
    print(f"[conf-collaudo] {step}: pass {rep['pass']} {json.dumps(rep)[:1500]}", flush=True)


def maxabs(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    if (np.isfinite(a) != np.isfinite(b)).any():
        return float("inf")
    return float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0


# ------------------------------------------------------------------------------------------ protocolli

def export_template() -> None:
    sys.path.insert(0, str(AAU.parent / "v3_work" / "unified_gt"))
    import domains
    s = cc.load(SPEC_DET)
    t = domains.template("hifi3d")
    p = cc.rp(s["landmarks"]["template"])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, V=np.asarray(t["V"], np.float64), F=np.asarray(t["F_target"], np.int64))
    print(f"[conf-collaudo] template di HIFI3D in {p}", flush=True)


def landmarks() -> None:
    import pr_common as pc
    s = cc.load(SPEC_DET)
    dev_def = json.loads((pc.DEFS / "hifi3d.json").read_text())
    with np.load(pc.DEFS_NPZ / "hifi3d.npz") as z:
        dev_lab = np.asarray(z["labels"])
    dev_prep = pc.read_prep("hifi3d")
    pc.use_roots(Path(s["_path"]))
    new_def = json.loads((pc.DEFS / f"{s['name']}.json").read_text())
    with np.load(pc.DEFS_NPZ / f"{s['name']}.npz") as z:
        new_lab = np.asarray(z["labels"])
    new_prep = pc.read_prep(s["name"])
    rep = {"indices_equal": {k: dev_def[k] == new_def[k] for k in ("lmk7", "ibug68", "realy85")},
           "labels_equal": bool(np.array_equal(dev_lab, new_lab)),
           "frame_check_mm": [dev_def["frame_check_mm"], new_def["frame_check_mm"]],
           "prep_keys_equal": bool(all(np.array_equal(dev_prep[k], new_prep[k]) for k in ("subjects", "topologies"))),
           "prep_L68_max_abs_mm": maxabs(dev_prep["L68"], new_prep["L68"]),
           "prep_K85_max_abs_mm": maxabs(dev_prep["K85"], new_prep["K85"]),
           "prep_labels_equal": bool(all(np.array_equal(a, b) for a, b in zip(dev_prep["labels"], new_prep["labels"])))}
    rep["pass"] = all(rep["indices_equal"].values()) and rep["labels_equal"] and rep["prep_keys_equal"] and \
        rep["prep_labels_equal"] and max(rep["prep_L68_max_abs_mm"], rep["prep_K85_max_abs_mm"]) <= TOL["geom_mm"]
    finish("landmarks", rep)


def det() -> None:
    from PIL import Image
    import pr_common as pc
    import pr_noise as pn
    s = cc.load(SPEC_DET)
    ref = pn.NWORK / "det" / "hifi3d"
    dev_prep = np.load(pn.NWORK / "prep" / "hifi3d_det.npz", allow_pickle=True)
    pc.use_roots(Path(s["_path"]))
    new = pc.WORK / "noise" / "det" / s["name"]
    rep = {}
    with np.load(new / "cams.npz") as z, np.load(ref / "cams.npz") as r:
        rep["cams_paths_equal"] = bool(np.array_equal(z["paths"], r["paths"]))
        rep["cams_max_abs"] = max(maxabs(z[k], r[k]) for k in ("o", "f", "box"))
        n = len(z["box"])
    rep["pixels_different"] = int(sum((np.asarray(Image.open(new / f"{k:04d}.png")) !=
                                       np.asarray(Image.open(ref / f"{k:04d}.png"))).sum() for k in range(n)))
    with np.load(new / "lmk2d.npz") as z, np.load(ref / "lmk2d.npz") as r:
        rep["points2d_max_abs_px"] = maxabs(z["P"][:, :, :2], r["P"][:, :, :2])
        rep["fallback_equal"] = bool(np.array_equal(z["fallback"], r["fallback"]))
        rep["fallback"] = int(z["fallback"].sum())
    with np.load(pc.WORK / "noise" / "prep" / f"{s['name']}_det.npz", allow_pickle=True) as z:
        rep["detprep_L68_max_abs_mm"] = maxabs(z["L68"], dev_prep["L68"])
        rep["detprep_K85_max_abs_mm"] = maxabs(z["K85"], dev_prep["K85"])
    rep["D_mm_max_abs"], rep["pairs_compared"] = {}, {}
    for m in pc.METHODS:
        with np.load(pc.PLUGIN_ROOT / m / "lmk_det" / f"{s['name']}.npz") as z, \
                np.load(EV / "competitors_new" / m / "lmk_det" / "hifi3d.npz") as r:
            A, B = np.asarray(z["D_mm"]), np.asarray(r["D_mm"])
            rep["D_mm_max_abs"][m] = maxabs(A, B)
            rep["pairs_compared"][m] = int(np.isfinite(B).sum() - len(B)) // 2
    rep["detprep_summary"] = json.loads((pc.EVID / f"detprep_{s['name']}.json").read_text())["det_fallback"]
    rep["pass"] = rep["cams_paths_equal"] and rep["cams_max_abs"] == 0 and rep["pixels_different"] == 0 and \
        rep["points2d_max_abs_px"] == 0 and rep["fallback_equal"] and \
        max(rep["detprep_L68_max_abs_mm"], rep["detprep_K85_max_abs_mm"]) <= TOL["geom_mm"] and \
        max(rep["D_mm_max_abs"].values()) <= TOL["geom_mm"]
    finish("det", rep)


# ------------------------------------------------------------------------------------------ valutatore

GMAP = {"nocrop": ("nocrop", "-"), "crop": ("crop", "crop"), "all_cross": ("all_cross", "-")}


def cmp_rows(mine: pd.DataFrame, theirs: pd.DataFrame, keys_mine, keys_theirs, vals: dict) -> dict:
    """Unione per chiave e scarto massimo per valore ({mio: (loro, segno, scambia IC)})."""
    a = mine.set_index(keys_mine)
    b = theirs.set_index(keys_theirs)
    b.index.names = keys_mine
    j = a.join(b, how="inner", lsuffix="_m", rsuffix="_t")
    out = {"n": int(len(j))}
    for c, (t, sign) in vals.items():
        x = j[f"{c}_m"] if f"{c}_m" in j else j[c]
        y = sign * (j[f"{t}_t"] if f"{t}_t" in j else j[t])
        out[c] = float(np.abs(x.to_numpy(float) - y.to_numpy(float)).max()) if len(j) else None
    return out


def max_diff(rep: dict, keys: list) -> float:
    """Scarto massimo sui valori confrontati (non sui conteggi); inf se un confronto non ha righe."""
    v = [x for k in keys for c, x in rep[k].items() if c not in ("n", "n_righe")]
    return float("inf") if any(x is None for x in v) else max(v)


def eval_full() -> None:
    s = cc.load(SPEC_FULL)
    d = cc.evidence(s) / "conf"
    S, D = pd.read_csv(d / "spearman.csv"), pd.read_csv(d / "deltas.csv")
    ctrl = json.loads((d / "controls.json").read_text())
    fmS, fmD = pd.read_csv(EV / "full_matrix" / "spearman.csv"), pd.read_csv(EV / "full_matrix" / "deltas.csv")
    fmS, fmD = fmS[fmS["dominio"] == "hifi3d"], fmD[fmD["dominio"] == "hifi3d"]
    rep = {"rows": ctrl["rows"], "rows_by_group": {"confermativa": ctrl["rows_by_group_confirmatory"],
                                                   "mascherata": ctrl["rows_by_group_masked"]}}
    for mine, passo in (("mascherata (descrittiva)", "comune"), ("confermativa (+inf)", SENS)):
        x = S[S["lettura"] == mine].copy()
        x["ins"], x["per"] = zip(*x["gruppo"].map(GMAP))
        y = fmS[fmS["passo"] == passo]
        rep[f"spearman {mine} vs full_matrix {passo}"] = cmp_rows(
            x, y, ["ins", "per", "gt", "colonna"], ["insieme_righe", "perturbazione", "gt", "colonna"],
            {"rho": ("rho", 1), "ic_lo": ("ic_lo", 1), "ic_hi": ("ic_hi", 1), "n_righe": ("n_righe", 1)})
        x = D[(D["lettura"] == mine) & (D["regola"] != "R7")].copy()
        x["ins"], x["per"] = zip(*x["gruppo"].map(GMAP))
        y = fmD[(fmD["passo"] == passo) & fmD["tipo"].isin(["vs_nostro", "scala_allineamento"])]
        y = y.drop_duplicates(["insieme_righe", "perturbazione", "gt", "colonna_a", "colonna_b"])
        rep[f"delta {mine} vs full_matrix {passo}"] = cmp_rows(
            x, y, ["ins", "per", "gt", "colonna_a", "colonna_b"],
            ["insieme_righe", "perturbazione", "gt", "colonna_a", "colonna_b"],
            {"delta": ("delta", 1), "ic_lo": ("ic_lo", 1), "ic_hi": ("ic_hi", 1), "p_le0": ("p_le0", 1)})
    pS, pD = pd.read_csv(EV / "protocolli" / "spearman.csv"), pd.read_csv(EV / "protocolli" / "deltas.csv")
    pS = pS[(pS["dominio"] == "hifi3d") & (pS["kind"] == "rho")]
    pD = pD[(pD["dominio"] == "hifi3d") & (pD["tipo"] == "NoW originale - NoW metrical")]
    x = S[(S["lettura"] == "confermativa (+inf)") & S["colonna"].isin(["now_original", "now_metrical", "realy",
                                                                       "stirling"])].copy()
    x["ins"], x["per"] = zip(*x["gruppo"].map(GMAP))
    rep["spearman protocolli (confermativa) vs protocolli/spearman.csv"] = cmp_rows(
        x, pS, ["ins", "per", "gt", "colonna"], ["insieme_righe", "perturbazione", "gt", "colonna"],
        {"rho": ("rho", 1), "ic_lo": ("ic_lo", 1), "ic_hi": ("ic_hi", 1), "n_righe": ("n_righe", 1)})
    x = D[(D["lettura"] == "confermativa (+inf)") & (D["regola"] == "R7")].copy()
    x["ins"], x["per"] = zip(*x["gruppo"].map(GMAP))
    rep["R7 (= -(NoW originale - NoW metrical)) vs protocolli/deltas.csv"] = cmp_rows(
        x, pD, ["ins", "per", "gt"], ["insieme_righe", "perturbazione", "gt"],
        {"delta": ("delta", -1), "ic_lo": ("ic_hi", -1), "ic_hi": ("ic_lo", -1)})
    keys = [k for k in rep if k.startswith(("spearman", "delta", "R7"))]
    worst = max_diff(rep, keys)
    rep["max_abs_diff"] = worst
    rep["n_righe_max_abs_diff"] = max(rep[k].get("n_righe", 0) or 0 for k in keys)
    rep["compared"] = {k: rep[k]["n"] for k in keys}
    rep["pass"] = worst <= TOL["num"] and rep["n_righe_max_abs_diff"] == 0 and all(rep[k]["n"] > 0 for k in keys)
    rep["rules_hifi3d_seed_990708"] = pd.read_csv(d / "rules.csv")[
        ["regola", "seme", "delta", "ic_lo", "ic_hi", "k_le0", "non_risolto", "esito"]].to_dict("records")
    finish("eval-full", rep)


def eval_det() -> None:
    s = cc.load(SPEC_DET)
    d = cc.evidence(s) / "conf"
    S, D = pd.read_csv(d / "spearman.csv"), pd.read_csv(d / "deltas.csv")
    rS = pd.read_csv(EV / "protocolli" / "rumore_landmark" / "spearman.csv")
    rD = pd.read_csv(EV / "protocolli" / "rumore_landmark" / "deltas.csv")
    rS = rS[(rS["dominio"] == "hifi3d") & (rS["cond"] == "det")].assign(colonna=lambda q: q["metodo"])
    rD = rD[(rD["dominio"] == "hifi3d") & (rD["cond"] == "det") & (rD["metodo"] == "now_metrical") &
            (rD["tipo"] == "protocollo - nostro") & (rD["gt"] == "fr")]
    x = S[(S["lettura"] == "confermativa (+inf)") & S["colonna"].isin(["now_original", "now_metrical", "realy",
                                                                       "stirling"])]
    rep = {"spearman protocolli col rilevatore vs rumore_landmark": cmp_rows(
        x, rS, ["gruppo", "gt", "colonna"], ["insieme_righe", "gt", "colonna"],
        {"rho": ("rho", 1), "ic_lo": ("ic_lo", 1), "ic_hi": ("ic_hi", 1), "n_righe": ("n_righe", 1)})}
    x = D[(D["regola"] == "NoW metrical - d_F") & (D["colonna_b"] == "factorized_s1234|form")]
    rep["NoW metrical - d_F non cal. s1234 vs rumore_landmark"] = cmp_rows(
        x, rD, ["gruppo"], ["insieme_righe"], {"delta": ("delta", 1), "ic_lo": ("ic_lo", 1), "ic_hi": ("ic_hi", 1),
                                               "p_le0": ("p_le0", 1)})
    keys = list(rep)
    rep["max_abs_diff"] = max_diff(rep, keys)
    rep["n_righe_max_abs_diff"] = max(rep[k].get("n_righe", 0) or 0 for k in keys)
    rep["controls"] = {k: v for k, v in json.loads((d / "controls.json").read_text()).items()
                       if k in ("rows", "rows_by_group_confirmatory", "subsample_pairs", "fallback")}
    rep["pass"] = rep["max_abs_diff"] <= TOL["spearman_geom"] and rep["n_righe_max_abs_diff"] == 0 and \
        all(rep[k]["n"] > 0 for k in keys)
    finish("eval-det", rep)


# ------------------------------------------------------------------------------------------ blmm, bp, testa

def _pair(task):
    import blmm
    view, mode, step, ka, kb, seed = task
    X = blmm.work_coords(view, mode, blmm.mesh_path(view, *ka))
    Y = blmm.work_coords(view, mode, blmm.mesh_path(view, *kb))
    return blmm.pair_metrics(X, Y, seed, mode, step)


def blmm_check(workers: int) -> None:
    import multiprocessing as mp
    sys.path.insert(0, str(AAU / "baselines_mm"))
    sys.path.insert(0, str(AAU / "baselines"))
    import blmm
    import common
    s = cc.load(SPEC_FULL)
    name = blmm.register_spec(s["_path"])
    dev, new = blmm.OUT_ROOT / "hifi3d", blmm.view_root(name)
    rep = {}
    with np.load(dev / "scalars.npz") as a, np.load(new / "scalars.npz") as b:
        pa, pb = list(a["paths"].astype(str)), list(b["paths"].astype(str))
        rep["scalars_paths"] = [len(pa), len(pb), sorted(pa) == sorted(pb)]
        o = [pa.index(q) for q in pb]
        rep["scalars_max_rel"] = max(float(np.max(np.abs(b[c] / a[c][o] - 1))) for c in a.files if c != "paths")
    dp = json.loads(blmm.PARAMS_JSON.read_text())["domains"]["hifi3d"]
    npj = json.loads(blmm.params_json(name).read_text())["domains"][name]
    rep["params"] = {"dev": dp, "spec": npj, "L_rel": abs(npj["L"] / dp["L"] - 1), "cs_ref_rel": abs(npj["cs_ref"] /
                                                                                                     dp["cs_ref"] - 1)}
    rep["template"] = {}
    for mode in ("mm", "cs"):
        with np.load(dev / mode / "template.npz") as a, np.load(new / mode / "template.npz") as b:
            ka = list(zip(a["subjects"].astype(str), a["topologies"].astype(str)))
            kb = list(zip(b["subjects"].astype(str), b["topologies"].astype(str)))
            o = [ka.index(k) for k in kb]
            rep["template"][mode] = {"R_max_abs_mm": maxabs(b["R"], a["R"][o]), "T_max_abs_mm": maxabs(b["T"], a["T"]),
                                     "template_subjects_equal": bool(np.array_equal(a["template_subjects"],
                                                                                    b["template_subjects"])),
                                     "failed": [len(a["failed"]), len(b["failed"])]}
    subj = blmm.subjects(name)
    iu, ju = np.triu_indices(len(subj), 1)
    rng = np.random.default_rng(20261013)
    tps = [(x, y) for x in blmm.TOPOLOGIES for y in blmm.TOPOLOGIES if x != y]
    rep["pairs"] = {}
    for mode in ("mm", "cs"):
        for step, n, metric in (("fast", N_PAIRS, "rigid_icp_chamfer"), ("nicp", N_NICP, "nicp_p2tri")):
            pick = [(tps[int(rng.integers(len(tps)))], int(rng.integers(len(iu)))) for _ in range(n)]
            tasks = [(name, mode, step, (subj[iu[q]], ta), (subj[ju[q]], tb), q) for (ta, tb), q in pick]
            with mp.get_context("fork").Pool(workers) as pool:
                res = pool.map(_pair, tasks, chunksize=1)
            unit = blmm.unit(name, mode)
            diffs = []
            for ((ta, tb), q), r in zip(pick, res):
                M, sj, _, _, _ = common.load_matrix(common.matrix_path(metric, ta, tb, dev / mode))
                diffs.append((r[metric] * unit, M[iu[q], ju[q]]))
            a_, b_ = np.asarray(diffs).T
            rep["pairs"][f"{mode}_{metric}"] = {"n": n, "max_abs_mm": maxabs(a_, b_),
                                                "nan": [int(np.isnan(a_).sum()), int(np.isnan(b_).sum())]}
    rep["pass"] = rep["scalars_paths"][2] and rep["scalars_max_rel"] <= TOL["rel"] and \
        max(rep["params"]["L_rel"], rep["params"]["cs_ref_rel"]) <= TOL["rel"] and \
        all(v["R_max_abs_mm"] <= TOL["geom_mm"] and v["template_subjects_equal"] for v in rep["template"].values()) and \
        all(v["max_abs_mm"] <= TOL["geom_mm"] for v in rep["pairs"].values())
    finish("blmm", rep)


def bp_check() -> None:
    sys.path.insert(0, str(AAU / "baselines_param"))
    import bp
    s = cc.load(SPEC_FULL)
    name = bp.register_spec(s["_path"])
    rep = {}
    for m in bp.MODELS:
        dev, new = bp.OUT_ROOT / "hifi3d" / m, bp.out_dir(name, m)
        r = {}
        with np.load(dev / "region.npz") as a, np.load(new / "region.npz") as b:
            r["region_equal"] = bool(np.array_equal(a["vertices"], b["vertices"]))
        for f, keys in (("fit.npz", ("beta", "D_fr", "D_sr")), ("fit_e1.npz", ("vb_beta", "D_vb_fr", "D_vb_sr")),
                        ("fit_e2_crop.npz", ("vb_beta",))):
            with np.load(dev / f) as a, np.load(new / f) as b:
                same = np.array_equal(a["subjects"], b["subjects"]) and np.array_equal(a["topologies"], b["topologies"])
                r[f] = {"keys_equal": bool(same), **{k: maxabs(b[k], a[k]) for k in keys}}
        # riferimento (non un criterio): due esecuzioni di sviluppo dello stesso codice, time_e2.npz (emendamento 2) contro
        # fit_e1.npz sulle stesse mesh
        with np.load(dev / "time_e2.npz", allow_pickle=True) as t, np.load(dev / "fit_e1.npz") as a:
            ka = list(zip(a["subjects"].astype(str), a["topologies"].astype(str)))
            o = [ka.index(k) for k in zip(t["subjects"].astype(str), t["topologies"].astype(str))]
            r["dev_rerun_vb_beta_max_abs"] = maxabs(t["vb_beta"], a["vb_beta"][o])
        rep[m] = r
    rep["pass"] = all(rep[m]["region_equal"] and all(v["keys_equal"] and max(x for k, x in v.items() if k != "keys_equal")
                                                     <= TOL["geom_mm"] for f, v in rep[m].items() if f.endswith(".npz"))
                      for m in bp.MODELS)
    rep["note"] = ("beta: coefficienti a +1 sigma (adimensionali); D_*: mm. La tolleranza geom_mm si applica a entrambi, "
                   "come scritta prima di eseguire; dev_rerun_vb_beta_max_abs e' solo il riferimento")
    finish("bp", rep)


def head_check() -> None:
    s = cc.load(SPEC_FULL)
    d = cc.evidence(s) / "head"
    blk = json.loads((d / "block.json").read_text())
    k0 = {a: json.loads((d / f"{a}.json").read_text())["K0"] for a in s["head"]["arms"]}
    rep = {"block": blk["arms"], "K0": k0,
           "pass": all(all(v["equal"].values()) for v in blk["arms"].values()) and all(v["pass"] for v in k0.values())}
    finish("head", rep)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("export-template", "landmarks", "det", "eval-full", "eval-det", "blmm", "bp", "head"))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    {"export-template": export_template, "landmarks": landmarks, "det": det, "eval-full": eval_full,
     "eval-det": eval_det, "blmm": lambda: blmm_check(a.workers), "bp": bp_check, "head": head_check}[a.step]()


if __name__ == "__main__":
    main()
