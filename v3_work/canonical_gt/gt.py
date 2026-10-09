#!/usr/bin/env python3
"""E12, passi 1-3 e arbitro: GT canoniche, verifiche, Spearman fra GT, identificabilita' su catture reali.

    v3_work/unified_gt/run.sh v3_work/canonical_gt/gt.py          (run.sbatch, passo ``gt``)

Protocollo: ``aau/runs/evidence/e12/protocol.md`` (sezioni 0-5 ed Emendamento 1). Definizioni in ``cgt.py``.

1. Frame di GT-F per ogni dominio: unita' (dichiarata, o 63 mm / IPD se ignota), rigida media -> media FLAME;
   controlli: la similarita' del json ricalcolata dal template (e' per dominio), la sua scala ``k_d = s_d / u_d``,
   la rotazione uguale a quella del json, l'IPD contro 63 mm. Scrive ``frames.json``, ``transforms_check.csv``,
   ``units.csv``.
2. Tutte le GT dei set di valutazione (hifi3d, faceverse, facescape: pool di 500; famos: 95 neutre) in
   ``datasets/CANONICAL_GT/<set>_<gt>.npz`` (mm); dimensione del volto (centroid size, IPD, altezza) e CV anche sui
   set di training; diagnostici delle rigide (effetto Pinocchio). ``size_cv.csv``.
3. Spearman fra tutte le GT e le baseline banali: 100 soggetti valutati (IC bootstrap per soggetto, 1000 repliche,
   seme 1234, ``evidence.boot_spearman`` di E8) e pool intero; FaMoS TEST (15) e tutti (95). ``gt_corr.csv``.
4. Arbitro (Emendamento 1, E2): AUC di verifica, rank-1, rapporto intra / inter sulle catture FaMoS (primi
   fotogrammi di ogni sequenza, 95 persone; sensibilita': 15 di TEST, fotogrammi filtrati) e Multiface
   (secondario); regola di selezione. ``ident.csv``, ``ident_paired.csv``, ``ident.json``.
Riassunto in ``gt.json``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import pandas as pd

import cgt
from cgt import C, Canon, domains

import evidence as e8ev  # noqa: E402  (v3_work/unified_gt: boot_spearman, spearman, gt_on)

TRAIN_SETS = ("flame", "bfm", "ict", "gnm", "multiface")
N_BOOT, SEED = 1000, 1234
FLAME_MASKS = C.REPO_ROOT / "v2_work" / "genflame" / "official" / "FLAME_masks.pkl"
CANDIDATES = ("F", "S", "EDM", "EDM_s")                     # in ordine di semplicita' (protocollo, E2)
IDENT_GTS = ("F", "F_rig_ls", "S", "EDM", "EDM_s", "unified", "maxabs", "F_pure", "size_only", "height_only")


def q(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "median": float(np.median(x)),
            "p5": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)), "min": float(x.min()),
            "max": float(x.max())}


# --------------------------------------------------------------------------- 1. frame di GT-F

def frames(cn: Canon) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    sp = cn.sp
    flame = domains.flame()
    P_flame = flame["V"][sp.z["flame_vidx"]] * 1000.0
    w_flame = C.vertex_areas(P_flame, sp.z["F"])
    eyes = cgt.eye_proxies(cn)
    fr, rows_t, rows_u, M_dom = {}, [], [], {}
    for d in domains.DOMAINS:
        tpl = domains.template(d)
        M = C.bary_interp(tpl["V"], sp.z[f"vidx_{d}"], sp.z[f"bary_{d}"])
        M_dom[d] = M
        s0, R0, t0 = cn.json_transform(d)
        if d == "flame":                     # unified.py fissa FLAME a m -> mm esatto
            s, R, t = 1000.0, np.eye(3), np.zeros(3)
        else:
            s, R, t = C.umeyama(M, P_flame, w_flame)
        rows_t.append({"domain": d, "s_json": s0, "s_recomputed": s, "rel_diff_s": abs(s - s0) / s0,
                       "max_abs_diff_R": float(np.abs(R - R0).max()), "max_abs_diff_t_mm": float(np.abs(t - t0).max()),
                       "rotation_deg": float(cgt.rot_angle(R0[None])[0]), "det_R": float(np.linalg.det(R0)),
                       "units_declared": cn.ct[d]["units_in"]})
        ipd_nat = float(cgt.ipd(M[None], eyes["idx"])[0])
        est = cgt.IPD_REF_MM / ipd_nat
        name, u_pow, r = cgt.nearest_unit(est)
        declared = cgt.UNITS_DECLARED[d]
        u = declared if declared is not None else est
        _, Rf, tf = C.umeyama(u * M, P_flame, w_flame, scale=False)
        res = np.sqrt(np.average(((u * M @ Rf.T + tf - P_flame) ** 2).sum(1), weights=w_flame))
        fr[d] = {"u": u, "R": Rf.tolist(), "t": tf.tolist(),
                 "unit_source": "dichiarata (domains.py)" if declared is not None else "IPD: 63 mm / IPD della media"}
        rows_u.append({"domain": d, "units_declared": cn.ct[d]["units_in"], "u_declared_mm": declared,
                       "ipd_template_native": ipd_nat, "mm_per_unit_from_ipd63": est,
                       "mm_per_unit_from_flame_ipd": eyes["flame_ipd_proxy_mm"] / ipd_nat,
                       "nearest_power_of_ten": name, "ratio_ipd_estimate_over_power": r, "u_used_mm": u,
                       "unit_source": fr[d]["unit_source"], "ipd_template_mm_with_u": ipd_nat * u,
                       "ipd_disagrees_15pct": bool(abs(ipd_nat * u / cgt.IPD_REF_MM - 1.0) > cgt.UNIT_TOL),
                       "s_json": s0, "k_json_over_u": s0 / u,
                       "rotation_F_vs_json_max_abs": float(np.abs(Rf - R0).max()),
                       "rigid_mean_to_flame_mean_rms_mm": float(res)})
    eyes_info = {"proxy_distance_mm": [float(x) for x in eyes["proxy_mm"]],
                 "flame_ipd_exact_mm": eyes["flame_ipd_exact_mm"], "flame_ipd_proxy_mm": eyes["flame_ipd_proxy_mm"],
                 "region_idx": [int(i) for i in eyes["idx"]]}
    return fr, pd.DataFrame(rows_t), pd.DataFrame(rows_u), {"eyes": eyes_info, "M": M_dom, "P_flame": P_flame,
                                                            "eye_idx": eyes["idx"]}


# ------------------------------------------------------------------------------- maxabs

def face_mask() -> np.ndarray:
    with open(FLAME_MASKS, "rb") as fh:
        return np.asarray(pickle.load(fh, encoding="latin1")["face"], dtype=np.int64)


def maxabs_normalize(V: np.ndarray) -> np.ndarray:
    """Copia di ``aau/zs3dmm/make_zs_expr_topologies.maxabs_normalize`` (centro sulla media, divisione per
    max|coordinata|), per non importare il generatore delle viste."""
    Vc = np.asarray(V, dtype=np.float64)
    Vc = Vc - Vc.mean(axis=0, keepdims=True)
    scale = float(np.abs(Vc).max())
    return Vc / scale if scale > 1e-6 else Vc * 0.0


def maxabs_matrix(X: np.ndarray) -> np.ndarray:
    """Media per vertice della distanza L2 fra forme normalizzate maxabs (``vertex_mean_l2_matrix``)."""
    X = np.stack([maxabs_normalize(x) for x in X])
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i + 1:] = np.linalg.norm(X[i][None] - X[i + 1:], axis=-1).mean(1)
    return D + D.T


def real_maxabs(V_full: np.ndarray, rob: dict, mask: np.ndarray) -> np.ndarray:
    """Maxabs delle catture FLAME (FaMoS): mesh intera con la rigida robusta della regione, maschera ``face``."""
    return maxabs_matrix(np.einsum("nvb,nab->nva", V_full[:, mask], rob["R"]) + rob["t"][:, None])


# ----------------------------------------------------------------------------- dimensione

def size_rows(tag: str, cn: Canon, Xf: np.ndarray, eye_idx: np.ndarray, groups: dict) -> list[dict]:
    cs, ip, h = cn.centroid_size(Xf), cgt.ipd(Xf, eye_idx), cn.height(Xf)
    out = []
    for g, m in groups.items():
        rec = {"set": tag if g == "all" else f"{tag}_{g}", "n": int(m.sum())}
        for k, v in (("centroid_size", cs), ("ipd", ip), ("height", h)):
            rec[f"{k}_mean_mm"] = float(v[m].mean())
            rec[f"{k}_sd_mm"] = float(v[m].std(ddof=1))
            rec[f"{k}_cv"] = float(v[m].std(ddof=1) / v[m].mean())
        rec["corr_cs_ipd"] = float(np.corrcoef(cs[m], ip[m])[0, 1])
        out.append(rec)
    return out


def rigid_diag(diag: dict) -> dict:
    """Riassunto delle rigide per identita' (effetto Pinocchio) e della dispersione della posa."""
    r = diag["rigid_rob"]
    return {"rigid_ls": {k: q(v) for k, v in diag["rigid_ls"].items()},
            "rigid_rob": {"angle_deg": q(r["angle_deg"]), "angle_ls_vs_rob_deg": q(r["angle_ls_vs_rob_deg"]),
                          "iterations": q(r["iterations"]), "converged_fraction": float(np.mean(r["converged"])),
                          "downweighted_area": q(r["downweighted_area"])},
            "pose_spread": {k: q(v) for k, v in diag["pose_spread"].items()}}


# ------------------------------------------------------------------------- 3. correlazioni

def raw_gt(name: str, ids: list[str]) -> np.ndarray:
    """GT ``raw`` di build_zs_gt.py (patch nativa, media delle norme, nessuna normalizzazione)."""
    root, prefix, off = cgt.EVAL_SETS[name]
    with np.load(cgt.DATASETS / root / "gt" / f"{prefix}_matrix_distances_raw.npz") as z:
        D, names = z["D_orig"].astype(np.float64), [str(s) for s in z["names"]]
    pos = {f"id{off + int(n[len(prefix):])}": i for i, n in enumerate(names)}
    ii = np.array([pos[s] for s in ids])
    return D[np.ix_(ii, ii)]


def sub(D: np.ndarray, ids: list[str], subjects: list[str]) -> np.ndarray:
    pos = {s: i for i, s in enumerate(ids)}
    ii = np.array([pos[s] for s in subjects])
    return D[np.ix_(ii, ii)]


def corr_rows(tag: str, mats_eval: dict, mats_pool: dict | None) -> list[dict]:
    r = e8ev.boot_spearman(mats_eval, N_BOOT, SEED)
    rows = []
    for key, v in r.items():
        a, b = key.split("|")
        rec = {"set": tag, "gt_a": a, "gt_b": b, "n_subjects": len(next(iter(mats_eval.values()))), **v}
        if mats_pool is not None and a in mats_pool and b in mats_pool:
            n = len(mats_pool[a])
            iu = np.triu_indices(n, 1)
            rec["pool_point"] = e8ev.spearman(mats_pool[a][iu], mats_pool[b][iu])
            rec["pool_n"] = n
        rows.append(rec)
    return rows


# ------------------------------------------------------------------------------ 4. arbitro

_ID = {}


def _ident_task(name: str):
    return name, _ID["ident"].evaluate(_ID["D"][name])


def ident_set(tag: str, D: dict, person: np.ndarray, workers: int) -> tuple[list, list, dict]:
    """Misure per GT e differenze appaiate (stesse repliche) per un insieme di catture."""
    _ID["ident"] = cgt.Ident(person, N_BOOT, SEED)
    _ID["D"] = D
    with mp.get_context("fork").Pool(min(workers, len(D))) as pool:
        res = dict(pool.map(_ident_task, list(D)))
    rows, prs = [], []
    for g, r in res.items():
        for m in ("auc", "rank1", "ratio"):
            v = r[m]
            lo, hi = np.percentile(v[1:], [2.5, 97.5])
            rows.append({"set": tag, "gt": g, "metric": m, "point": float(v[0]), "ci_low": float(lo),
                         "ci_high": float(hi), "n_persons": r["n_persons"], "n_captures": r["n_captures"],
                         "n_genuine": r["n_genuine"], "n_impostor": r["n_impostor"]})
    for a in res:
        for b in res:
            if a == b:
                continue
            for m in ("auc", "rank1", "ratio"):
                dv = res[a][m] - res[b][m]
                lo, hi = np.percentile(dv[1:], [2.5, 97.5])
                prs.append({"set": tag, "a": a, "b": b, "metric": m, "diff": float(dv[0]), "ci_low": float(lo),
                            "ci_high": float(hi), "p_le0": float((dv[1:] <= 0).mean())})
    return rows, prs, res


def select(prs: pd.DataFrame, rows: pd.DataFrame, tag: str) -> dict:
    """Regola del protocollo (Emendamento 1, E2): AUC, poi rapporto intra / inter, poi semplicita'."""
    def point(g, m):
        return float(rows[(rows["set"] == tag) & (rows["gt"] == g) & (rows["metric"] == m)]["point"].iloc[0])

    def tied(best, cands, m):
        out = [best]
        for g in cands:
            if g == best:
                continue
            r = prs[(prs["set"] == tag) & (prs["a"] == best) & (prs["b"] == g) & (prs["metric"] == m)].iloc[0]
            if r["ci_low"] <= 0.0 <= r["ci_high"]:
                out.append(g)
        return out

    best_auc = max(CANDIDATES, key=lambda g: point(g, "auc"))
    t1 = tied(best_auc, CANDIDATES, "auc")
    steps = {"auc_best": best_auc, "auc_tied": t1}
    if len(t1) > 1:
        best_ratio = min(t1, key=lambda g: point(g, "ratio"))
        t2 = tied(best_ratio, t1, "ratio")
        steps.update(ratio_best=best_ratio, ratio_tied=t2)
    else:
        t2 = t1
    steps["selected"] = min(t2, key=CANDIDATES.index)
    steps["strict_rule_auc_then_simplicity"] = min(t1, key=CANDIDATES.index)
    return steps


def capture_gts(cn: Canon, d: str, cap: dict, maxabs_fn) -> dict:
    D, diag = cgt.all_gts(cn, d, cap["P"], real=True)
    D["maxabs"] = maxabs_fn(cap, diag)
    return {g: D[g] for g in IDENT_GTS}, diag


# -------------------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train-sets", default=",".join(TRAIN_SETS), help="set di training per unita' e CV ('' = nessuno)")
    a = p.parse_args()
    workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")), 32)
    t0 = time.time()
    cgt.EVID_DIR.mkdir(parents=True, exist_ok=True)
    cn = Canon(frames={})
    out = {"protocol": "aau/runs/evidence/e12/protocol.md (sezioni 0-5 ed Emendamento 1)"}

    # 1. frame di GT-F
    fr, tr, un, tinfo = frames(cn)
    cn.frames = fr
    C.save_json(cgt.FRAMES_JSON, {"description": "GT-F: f = u * R @ p + t (p nel frame e nelle unita' dei dati del "
                                                 "dominio; mm, frame della media FLAME)", "domains": fr})
    tr.to_csv(cgt.EVID_DIR / "transforms_check.csv", index=False)
    un.to_csv(cgt.EVID_DIR / "units.csv", index=False)
    out["eyes"] = tinfo["eyes"]
    print(tr.to_string(index=False), flush=True)
    print(un.to_string(index=False), flush=True)
    eye_idx = tinfo["eye_idx"]
    mask = face_mask()

    # 2. GT dei set di valutazione, dimensione, diagnostici
    srows, sets_info, gts = [], {}, {}
    for name in ("hifi3d", "faceverse", "facescape", "famos"):
        t1 = time.time()
        nat = cgt.native_points(name, cn)
        d = nat["domain"]
        real = name == "famos"
        D, diag = cgt.all_gts(cn, d, nat["P"], real=real)
        if real:
            D["maxabs"] = real_maxabs(nat["V_full"], diag["rob"], mask)
        Xf = diag["Xf"]
        groups = {"all": np.ones(len(Xf), bool)}
        if real:
            groups["test"] = nat["split"] == "test"
        srows += size_rows(name, cn, Xf, eye_idx, groups)
        info = {"n": len(nat["ids"]), "checks": nat["checks"], **rigid_diag(diag)}
        if d in tinfo["M"]:
            Tf = cn.to_F(d, tinfo["M"][d][None])
            info["mean_identity_vs_template_mm"] = float(cn.distances(Xf.mean(0)[None], Tf)[0, 0])
        info["rms_to_flame_mean_mm"] = q(cn.distances(Xf, tinfo["P_flame"][None])[:, 0])
        iu = np.triu_indices(len(Xf), 1)
        info["median_pair"] = {g: float(np.median(M[iu])) for g, M in D.items()}
        for g in cgt.SAVED + ("F_pure", "maxabs"):
            if g in D and not (g == "maxabs" and not real):
                cgt.save_gt(name, g, D[g], nat["ids"])
        if real:
            info["split"] = [str(s) for s in nat["split"]]
        gts[name] = {"ids": nat["ids"], "D": D, "split": nat.get("split")}
        sets_info[name] = info
        print(f"[e12-gt] {name}: n={info['n']} {time.time() - t1:.0f}s", flush=True)
        del nat, diag
    for name in [s for s in a.train_sets.split(",") if s]:
        t1 = time.time()
        nat = cgt.native_points(name, cn)
        Xf = cn.to_F(nat["domain"], nat["P"])
        srows += size_rows(name, cn, Xf, eye_idx, {"all": np.ones(len(Xf), bool)})
        sets_info[name] = {"n": len(nat["ids"]), "checks": nat["checks"]}
        print(f"[e12-gt] {name}: n={len(nat['ids'])} {time.time() - t1:.0f}s", flush=True)
        del nat, Xf
    size = pd.DataFrame(srows)
    size.to_csv(cgt.EVID_DIR / "size_cv.csv", index=False)
    print(size.to_string(index=False), flush=True)
    out["sets"] = sets_info

    # controlli: unificata ricalcolata contro quella su disco
    checks = {}
    for name in cgt.EVAL_SETS:
        g = gts[name]
        Dd = e8ev.gt_on(C.DATA_ROOT / "eval" / f"{name}_gt_matrix.npz", g["ids"])
        Du = g["D"]["unified"]
        iu = np.triu_indices(len(Du), 1)
        checks[f"{name}: unificata ricalcolata vs disco, max |diff| dopo /max"] = float(
            np.abs(Dd[iu] / Dd[iu].max() - Du[iu] / Du[iu].max()).max())
    import famos_common as fc
    g = gts["famos"]
    te = [i for i, s in enumerate(g["split"]) if s == "test"]
    Dd = e8ev.gt_on(fc.VIEW_DIR / "gt_matrix.npz", [fc.view_id(g["ids"][i]) for i in te])
    Du = g["D"]["unified"][np.ix_(te, te)]
    iu = np.triu_indices(len(te), 1)
    checks["famos TEST: unificata ricalcolata vs disco, max |diff| dopo /max"] = float(
        np.abs(Dd[iu] / Dd[iu].max() - Du[iu] / Du[iu].max()).max())
    out["checks"] = checks
    print(checks, flush=True)

    # 3. Spearman fra GT
    from zs_stage import select_subjects
    rows = []
    for name in cgt.EVAL_SETS:
        view = cgt.DATASETS / cgt.EVAL_SETS[name][0] / "eval_view"
        subj = select_subjects(view / "npz", SEED)
        g = gts[name]
        pool = dict(g["D"])
        pool["maxabs"] = e8ev.gt_on(view / "gt_matrix.npz", g["ids"])
        pool["raw"] = raw_gt(name, g["ids"])
        rows += corr_rows(name, {k: sub(D, g["ids"], subj) for k, D in pool.items()}, pool)
        print(f"[e12-gt] Spearman fra GT, {name}: fatto", flush=True)
    allm = gts["famos"]["D"]
    rows += corr_rows("famos_test", {k: D[np.ix_(te, te)] for k, D in allm.items()}, allm)
    corr = pd.DataFrame(rows)
    corr.to_csv(cgt.EVID_DIR / "gt_corr.csv", index=False)
    del gts

    # 4. arbitro: catture reali ripetute
    irows, iprs, cap_info = [], [], {}
    for tag, kept in (("famos95_first", False), ("famos95_kept", True)):
        t1 = time.time()
        cap = cgt.famos_captures(cn, kept_only=kept)
        D, diag = capture_gts(cn, "famos", cap, lambda c, dg: real_maxabs(c["V_full"], dg["rob"], mask))
        r, pr, _ = ident_set(tag, D, cap["person"], workers)
        irows += r
        iprs += pr
        cap_info[tag] = {"n_captures": int(len(cap["person"])), "n_persons": int(len(set(cap["person"]))),
                         "captures_per_person": q(np.unique(cap["person"], return_counts=True)[1]),
                         **rigid_diag(diag)}
        if not kept:
            m = cap["split"] == "test"
            r, pr, _ = ident_set("famos15_test_first", {k: v[np.ix_(m, m)] for k, v in D.items()},
                                 cap["person"][m], workers)
            irows += r
            iprs += pr
        print(f"[e12-gt] arbitro {tag}: {len(cap['person'])} catture, {time.time() - t1:.0f}s", flush=True)
        del cap, D, diag
    cap = cgt.multiface_captures(cn)
    D, diag = capture_gts(cn, "multiface", cap, lambda c, dg: maxabs_matrix(dg["Xf"]))
    r, pr, _ = ident_set("multiface_take", D, cap["person"], workers)
    irows += r
    iprs += pr
    cap_info["multiface_take"] = {"n_captures": int(len(cap["person"])), "n_persons": int(len(set(cap["person"]))),
                                  "segments": sorted(set(cap["seq"])), **rigid_diag(diag)}
    irows, iprs = pd.DataFrame(irows), pd.DataFrame(iprs)
    irows.to_csv(cgt.EVID_DIR / "ident.csv", index=False)
    iprs.to_csv(cgt.EVID_DIR / "ident_paired.csv", index=False)
    decision = {tag: select(iprs, irows, tag) for tag in ("famos95_first", "famos95_kept", "famos15_test_first")}
    C.save_json(cgt.EVID_DIR / "ident.json", {"rule": "protocol.md, Emendamento 1, E2", "primary_set": "famos95_first",
                                              "candidates_by_simplicity": list(CANDIDATES), "decision": decision,
                                              "captures": cap_info})
    print(irows[irows["metric"] == "auc"].to_string(index=False), flush=True)
    print(decision, flush=True)
    out["seconds"] = time.time() - t0
    C.save_json(cgt.EVID_DIR / "gt.json", out)


if __name__ == "__main__":
    main()
