#!/usr/bin/env python3
"""GT di riferimento scelta dal PI dopo E12 (F + rigida robusta, "form") per il training e per le valutazioni.

    v3_work/unified_gt/run.sh v3_work/canonical_gt/train_fr_sr.py          (train.sbatch; dopo gt.py)

Per ogni identita' (punti della regione unificata, frame di GT-F in mm, ``cgt.Canon.to_F``):
  - a_i = rigida robusta verso mu (``Canon.rigid_robust``, IRLS Tukey: la GT-F-rig-rob di E12);
  - S_i = centroid size pesata della regione (mm), c_i = centroide pesato di a_i;
  - z_i = (a_i - c_i) / S_i, la pre-forma a taglia unitaria con la stessa rotazione.
GT:
  - **FR** (form): d_FR(i, j) = sqrt(sum_v W_v ||a_i,v - a_j,v||^2), mm (W pesi d'area normalizzati di mu);
  - **SR** (shape): d_P(i, j) = sqrt(sum_v W_v ||z_i,v - z_j,v||^2), adimensionale.
Identita' esatta (pesi normalizzati, sum_v W_v z_v = 0):
    d_FR^2 = ||c_i - c_j||^2 + (S_i - S_j)^2 + S_i S_j d_P^2,
cioe' la formula d_F^2 = (S_i - S_j)^2 + S_i S_j d_P^2 piu' il termine dei centroidi, che la rigida robusta non
annulla (la traslazione robusta non e' quella dei centroidi). Verificata su un campione, insieme alla distanza
di Procrustes con la rotazione ottima per coppia (``procrustes_pair``).

Training: le 65.600 identita' del run su scala, ``names`` e ordine di ``datasets/SCALE_ALL/gt_joint_bfm_ict_gnm.npz``
(la GT di ``--dist_npz``). Forme native: le sorgenti di ``shapes.py`` (BFM: original REMESH, gia' allineate per
similarita' una per una; ICT e GNM: modello dai pesi). Controllo: la GT unificata ricalcolata da queste forme
contro ``datasets/UNIFIED_GT/train/s_train.npz`` (le viste del run). Formato di ``train_gt.py``: ``D_orig`` float32
diviso per il massimo, ``names`` copiati; versione TARATA come ``aau/evidence/e1_factorial/e1_calib_ugt.py``:
blocco (d, d) x f_d, f_d = mediana(maxabs, d) / mediana(GT, d), mediane esatte sulle coppie i < j (quelle della
maxabs dal json della taratura di C3F-UGT, stesso file sorgente), blocchi fra domini x sqrt(f_d1 f_d2).

Valutazione: hifi3d, faceverse, facescape (pool di 500, nomi della vista), famos_test (15, id9400NN), formato
delle viste (``D_orig`` / massimo, ``names``, json con l'unita').

Uscite: ``datasets/CANONICAL_GT/train/`` (gt_{fr,sr}_bfm_ict_gnm[_calib].npz + json, centroid_size_bfm_ict_gnm.npz,
fr_train.npz con a_i pesate) e ``datasets/CANONICAL_GT/eval/`` (<set>_{fr,sr}.npz + json, <set>_centroid_size.npz);
verifiche in ``aau/runs/evidence/e12/train_gt.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import cgt
from cgt import C, Canon, domains

sys.path.insert(0, str(C.REPO_ROOT / "aau" / "evidence" / "e1_factorial"))
import e1_calib_ugt as calib  # noqa: E402  (block_median, domain: le funzioni della taratura di C3F-UGT)
import train_gt as tg  # noqa: E402  (v3_work/unified_gt: distances a blocchi, simmetria esatta)
from shapes import ids_range  # noqa: E402

RUN_GT = C.REPO_ROOT / "datasets" / "SCALE_ALL" / "gt_joint_bfm_ict_gnm.npz"
UGT_CALIB_JSON = C.DATA_ROOT / "train" / "gt_unified_bfm_ict_gnm_calib.json"
S_TRAIN = C.DATA_ROOT / "train" / "s_train.npz"
TRAIN_DIR = cgt.OUT_DIR / "train"
EVAL_DIR = cgt.OUT_DIR / "eval"
DOMS = ("bfm", "ict", "gnm")
CHUNK = 2000
UNIT = {"fr": ("mm_per_unit", "mm"), "sr": ("dP_per_unit", "distanza di Procrustes (pre-forme a centroid size 1)")}


# --------------------------------------------------------------------------------- forme

_FJ = {}


def _factor_chunk(k: int) -> dict:
    cn, d, P = _FJ["cn"], _FJ["d"], _FJ["P"]
    sq = np.sqrt(cn.sp.w)[None, :, None]
    X = cn.to_F(d, P[k:k + CHUNK])
    rb = cn.rigid_robust(X)
    a = rb["a"]
    S = cn.centroid_size(a)
    c = cn.centroid(a)
    z = (a - c[:, None]) / S[:, None, None]
    return {"A": (sq * a).reshape(len(a), -1).astype(np.float32), "Z": (sq * z).reshape(len(a), -1).astype(np.float32),
            "S": S, "c": c, "angle": rb["angle"], "it": rb["iterations"], "conv": rb["converged"]}


def factor(cn: Canon, d: str, P: np.ndarray, workers: int = 1) -> dict:
    """a_i, S_i, c_i, z_i (pesati sqrt(w), appiattiti), a blocchi di CHUNK identita' (in parallelo con fork)."""
    import multiprocessing as mp
    _FJ.update(cn=cn, d=d, P=P)
    ks = list(range(0, len(P), CHUNK))
    if workers > 1 and len(ks) > 1:
        with mp.get_context("fork").Pool(min(workers, len(ks))) as pool:
            parts = pool.map(_factor_chunk, ks, chunksize=1)
    else:
        parts = [_factor_chunk(k) for k in ks]
    return {key: np.concatenate([q[key] for q in parts]) for key in parts[0]}


def _manifest_weights(task) -> dict:
    """Pesi delle identita' volute da un tar di shard (``manifest.json``, come ``domains._shard_manifest``)."""
    tar, want = task
    out = {}
    for rec in domains._shard_manifest(tar)["identities"]:
        if rec["sid"] in want:
            out[rec["sid"]] = np.asarray(rec["weights"], dtype=np.float64)
    return out


def shard_weights(d: str, sids: list[str], workers: int) -> np.ndarray:
    """Come ``domains.ict_weights`` / ``gnm_weights``, ma con i manifest letti in parallelo (CephFS scala coi
    processi; in sequenza i 400 tar ICT richiedevano 37 min in shapes.py)."""
    import multiprocessing as mp
    want = set(sids)
    W = {}
    if d == "ict":
        for s in sids:
            g = int(s[2:])
            if 10000 <= g < 15000:
                with np.load(domains.DATASETS / "ICT" / "identities" / f"ict{g - 10000:04d}.npz") as z:
                    W[s] = np.asarray(z["weights"], dtype=np.float64)
        tars = sorted((domains.DATASETS / "ICT_SCALE" / "shards").glob("shard_*.tar"))
    else:
        tars = sorted((domains.DATASETS / "GNM_DISTILL" / "shards").glob("gnm_shard_*.tar"))
    rest = want - set(W)
    if rest:
        with mp.get_context("fork").Pool(min(workers, len(tars))) as pool:
            for part in pool.imap_unordered(_manifest_weights, [(t, rest) for t in tars]):
                W.update(part)
    miss = [s for s in sids if s not in W]
    if miss:
        raise SystemExit(f"pesi {d} mancanti per {len(miss)} identita' (prima {miss[0]})")
    return np.stack([W[s] for s in sids])


def train_points(cn: Canon, names: list[str], workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Punti nativi della regione (n, m, 3) nell'ordine di ``names`` e dominio di ciascuno."""
    sp = cn.sp
    dom = np.array([calib.domain(s) for s in names])
    if set(dom) - set(DOMS):
        raise SystemExit(f"domini inattesi: {set(dom) - set(DOMS)}")
    P = np.empty((len(names), len(sp.mu), 3))
    pos = {s: i for i, s in enumerate(names)}
    # BFM: original REMESH (gli id del run sono id0000-0499)
    paths = {p.name.split("_GTready_")[0]: p for p in domains.bfm_original_paths()}
    for s in np.array(names)[dom == "bfm"]:
        P[pos[s]] = sp.map("bfm", domains.load_bfm_original(paths[s])[0])
    for d in ("ict", "gnm"):
        t0 = time.time()
        sids = [str(s) for s in np.array(names)[dom == d]]
        W = shard_weights(d, sids, workers)
        M0, B = sp.linear(d, domains.template(d))
        ii = np.array([pos[s] for s in sids])
        for k in range(0, len(sids), 4000):
            P[ii[k:k + 4000]] = M0[None] + np.einsum("nk,kvd->nvd", W[k:k + 4000, : B.shape[0]], B)
        print(f"[fr-sr] {d}: {len(sids)} pesi in {time.time() - t0:.0f}s", flush=True)
    return P, dom


# ---------------------------------------------------------------------------- verifiche

def procrustes_pair(zi: np.ndarray, zj: np.ndarray, W: np.ndarray) -> float:
    """Distanza fra pre-forme centrate a taglia 1 con la rotazione ottima per la coppia: 2 sin(rho / 2)."""
    Cm = (zj * W[:, None]).T @ zi
    U, s, Vt = np.linalg.svd(Cm)
    s[2] *= np.sign(np.linalg.det(U @ Vt))
    return float(np.sqrt(max(2.0 - 2.0 * s.sum(), 0.0)))


def identity_check(F: dict, W: np.ndarray, pairs: np.ndarray) -> dict:
    i, j = pairs[:, 0], pairs[:, 1]
    A64, Z64 = F["A"].astype(np.float64), F["Z"].astype(np.float64)
    dfr = np.sqrt(((A64[i] - A64[j]) ** 2).sum(1) / W.size_A)
    dp = np.sqrt(((Z64[i] - Z64[j]) ** 2).sum(1) / W.size_A)
    Si, Sj = F["S"][i], F["S"][j]
    cen = ((F["c"][i] - F["c"][j]) ** 2).sum(1)
    exact = np.sqrt(cen + (Si - Sj) ** 2 + Si * Sj * dp ** 2)
    pd_formula = np.sqrt((Si - Sj) ** 2 + Si * Sj * dp ** 2)
    zi = Z64[i].reshape(len(i), -1, 3) / np.sqrt(W.w)[None, :, None]
    zj = Z64[j].reshape(len(j), -1, 3) / np.sqrt(W.w)[None, :, None]
    dpair = np.array([procrustes_pair(a, b, W.Wn) for a, b in zip(zi, zj)])
    rel = lambda x, y: np.abs(x / y - 1.0)  # noqa: E731
    q = lambda x: {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "max": float(np.max(x))}  # noqa: E731
    return {"n_pairs": int(len(pairs)),
            "with_centroid_term_rel_err": q(rel(exact, dfr)),
            "formula_without_centroid_term_rel_err": q(rel(pd_formula, dfr)),
            "centroid_term_share_of_dFR2": q(cen / dfr ** 2),
            "size_term_share_of_dFR2": q((Si - Sj) ** 2 / dfr ** 2),
            "shape_term_share_of_dFR2": q(Si * Sj * dp ** 2 / dfr ** 2),
            "dP_common_rotation_over_pairwise_optimal": q(dp / dpair),
            "spearman_dP_common_vs_pairwise": float(np.corrcoef(np.argsort(np.argsort(dp)),
                                                                np.argsort(np.argsort(dpair)))[0, 1])}


class _W:
    def __init__(self, cn: Canon):
        self.w, self.Wn, self.size_A = cn.sp.w, cn.W, cn.sp.A
        self.size = len(cn.sp.w)


# ------------------------------------------------------------------------------- uscite

def save_train(kind: str, V: np.ndarray, A: float, names_arr, dom: np.ndarray, med_max: dict, extra: dict) -> dict:
    """Distanze (mm o d_P) di tutte le coppie; scrive PRIMA la versione tarata, poi la grezza (divisa per il massimo)."""
    t0 = time.time()
    D = tg.distances(V, A)                       # float32, unita' fisiche (mm per fr, d_P per sr)
    gmax = float(D.max())
    print(f"[fr-sr] {kind}: distanze in {time.time() - t0:.0f}s, massimo {gmax:.4g}", flush=True)
    key, unit = UNIT[kind]
    idx = {d: np.flatnonzero(dom == d) for d in DOMS}
    med = {d: calib.block_median(D, ix) / gmax for d, ix in idx.items()}   # mediane della versione grezza (max 1)
    f = {d: med_max[d] / med[d] for d in DOMS}
    fac = {(d1, d2): (f[d1] if d1 == d2 else float(np.sqrt(f[d1] * f[d2]))) / gmax for d1 in DOMS for d2 in DOMS}

    def scale_blocks(mult):
        for d1, ix1 in idx.items():
            for d2, ix2 in idx.items():
                m = np.float32(mult(fac[(d1, d2)]))
                for s0 in range(0, len(ix1), 2048):
                    r = ix1[s0:s0 + 2048]
                    D[np.ix_(r, ix2)] *= m

    tmp = TRAIN_DIR / f".{kind}.tmp.npz"
    scale_blocks(lambda x: x)                    # unita' fisiche -> tarata
    cal = TRAIN_DIR / f"gt_{kind}_bfm_ict_gnm_calib.npz"
    np.savez(tmp, D_orig=D, names=names_arr)
    os.replace(tmp, cal)
    C.save_json(cal.with_suffix(".json"), {
        "scale": f"canonical_{kind}_calibrated_to_maxabs_medians", "global_max": float(np.nanmax(D)),
        "n_total": int(len(names_arr)), "n_by_domain": {d: int(len(ix)) for d, ix in idx.items()},
        "names_source": str(RUN_GT), "names_identical_to_run_gt": True,
        "median_maxabs": med_max, f"median_{kind}_normalized": med, "factor": f,
        "cross_domain_factor": "sqrt(f_d1 * f_d2) (media geometrica; il trainer a batch monodominio non le legge)",
        f"{key}_by_domain": {d: gmax / f[d] for d in DOMS}, "units": unit,
        "median_maxabs_source": f"{UGT_CALIB_JSON} (mediane esatte di {RUN_GT}, aau/evidence/e1_factorial/e1_calib_ugt.py)",
        "definition": "v3_work/canonical_gt/train_fr_sr.py; blocco (d, d) della versione grezza (max 1) x f_d, "
                      "f_d = mediana(maxabs, d) / mediana(grezza, d); names e ordine identici", **extra})
    print(f"[fr-sr] {kind}: tarata scritta ({time.time() - t0:.0f}s), fattori {f}", flush=True)
    scale_blocks(lambda x: 1.0 / (x * gmax))     # tarata -> grezza (max 1)
    base = TRAIN_DIR / f"gt_{kind}_bfm_ict_gnm.npz"
    np.savez(tmp, D_orig=D, names=names_arr)
    os.replace(tmp, base)
    C.save_json(base.with_suffix(".json"), {
        "scale": f"canonical_{kind}", "global_max": float(D.max()), key: gmax, "units": unit,
        "n_total": int(len(names_arr)), "n_by_domain": {d: int(len(ix)) for d, ix in idx.items()},
        "names_source": str(RUN_GT), "names_identical_to_run_gt": True, "median_by_domain": med,
        "definition": "v3_work/canonical_gt/train_fr_sr.py; D_orig * " + key + " = " + unit, **extra})
    del D
    print(f"[fr-sr] {kind}: grezza scritta ({time.time() - t0:.0f}s)", flush=True)
    return {"max": gmax, "median": med, "factor": f}


def save_eval(name: str, ids: list[str], F: dict, A: float) -> dict:
    out = {}
    for kind, V in (("fr", F["A"]), ("sr", F["Z"])):
        X = V.astype(np.float64)
        G = (X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2.0 * X @ X.T
        D = np.sqrt(np.clip(G, 0.0, None) / A)
        np.fill_diagonal(D, 0.0)
        D = 0.5 * (D + D.T)
        gmax = float(D.max())
        p = EVAL_DIR / f"{name}_{kind}.npz"
        C.save_npz(p, D_orig=(D / gmax).astype(np.float32), names=np.asarray(ids))
        key, unit = UNIT[kind]
        iu = np.triu_indices(len(D), 1)
        man = {"set": name, "gt": kind, "n": len(ids), key: gmax, "units": unit, "median": float(np.median(D[iu])),
               "definition": "v3_work/canonical_gt/train_fr_sr.py; D_orig * " + key + " = " + unit}
        p.with_suffix(".json").write_text(json.dumps(man, indent=1) + "\n")
        out[kind] = D
    C.save_npz(EVAL_DIR / f"{name}_centroid_size.npz", S=F["S"], names=np.asarray(ids))
    return out


# ---------------------------------------------------------------------------------- main

def stage_eval(cn: Canon, rep: dict, rng) -> None:
    Wc = _W(cn)
    ev = {}
    for name in ("hifi3d", "faceverse", "facescape", "famos"):
        nat = cgt.native_points(name, cn)
        F = factor(cn, nat["domain"], nat["P"])
        ids = list(nat["ids"])
        e12_file, e12_ids = cgt.OUT_DIR / f"{name}_F_rig_rob.npz", ids
        if name == "famos":
            import famos_common as fc
            te = np.flatnonzero(nat["split"] == "test")
            te = te[np.argsort([fc.view_id(ids[i]) for i in te])]
            F = {k: v[te] for k, v in F.items()}
            e12_ids = [ids[i] for i in te]
            ids, name = [fc.view_id(s) for s in e12_ids], "famos_test"
        D = save_eval(name, ids, F, cn.sp.A)
        info = {"n": len(ids), "S_mean_mm": float(F["S"].mean()), "S_cv": float(F["S"].std(ddof=1) / F["S"].mean()),
                "robust_converged": float(F["conv"].mean())}
        with np.load(e12_file) as z:
            Do, pos = z["D_orig"], {str(s): k for k, s in enumerate(z["names"])}
        ii = np.array([pos[s] for s in e12_ids])
        info["fr_vs_e12_F_rig_rob_max_abs_mm"] = float(np.abs(Do[np.ix_(ii, ii)] - D["fr"]).max())
        pr = rng.choice(len(ids), size=(400, 2))
        info["identity"] = identity_check(F, Wc, pr[pr[:, 0] != pr[:, 1]])
        ev[name] = info
        print(f"[fr-sr] eval {name}: S cv {info['S_cv']:.3f}, FR vs E12 {info['fr_vs_e12_F_rig_rob_max_abs_mm']:.1e}", flush=True)
    rep["eval"] = ev


def stage_shapes(cn: Canon, rep: dict, rng, workers: int) -> None:
    """Forme di training: a_i, z_i, S_i nell'ordine dei names del run."""
    t0 = time.time()
    with np.load(RUN_GT) as z:
        names_arr = z["names"]
    names = [str(s) for s in names_arr]
    P, dom = train_points(cn, names, workers)
    print(f"[fr-sr] {len(names)} forme native in {time.time() - t0:.0f}s", flush=True)
    with np.load(S_TRAIN) as z:
        if [str(s) for s in z["names"]] != names:
            raise SystemExit("s_train.npz: names diversi")
        probe = np.sort(rng.choice(len(names), 3000, replace=False))
        St = z["s"][probe].astype(np.float64)
    a_u, _, _ = cn.sp.align(np.stack([cn.to_F(dom[k], P[k][None])[0] for k in probe]))
    Su = (np.sqrt(cn.sp.w)[None, :, None] * a_u).reshape(len(probe), -1)
    rep["unified_from_native_vs_s_train_max_mm"] = float(np.sqrt(((Su - St) ** 2).sum(1) / cn.sp.A).max())
    print(f"[fr-sr] unificata da forme native vs s_train: {rep['unified_from_native_vs_s_train_max_mm']:.2e} mm", flush=True)
    if rep["unified_from_native_vs_s_train_max_mm"] > 1e-3:
        raise SystemExit("forme native diverse da quelle del run")
    F = {}
    for d in DOMS:
        ii = np.flatnonzero(dom == d)
        Fd = factor(cn, d, P[ii], workers)
        for k, v in Fd.items():
            if k not in F:
                F[k] = np.empty((len(names),) + v.shape[1:], v.dtype)
            F[k][ii] = v
        print(f"[fr-sr] {d}: rigida robusta in {time.time() - t0:.0f}s", flush=True)
    del P
    rep["train_robust"] = {"converged_fraction": float(F["conv"].mean()),
                           "iterations_p95": float(np.percentile(F["it"], 95)),
                           "angle_deg_median": float(np.median(F["angle"]))}
    rep["train_centroid_size"] = {d: {"n": int((dom == d).sum()), "mean_mm": float(F["S"][dom == d].mean()),
                                      "cv": float(F["S"][dom == d].std(ddof=1) / F["S"][dom == d].mean())} for d in DOMS}
    C.save_npz(TRAIN_DIR / "centroid_size_bfm_ict_gnm.npz", S=F["S"], centroid_mm=F["c"], names=names_arr, domain=dom)
    C.save_npz(TRAIN_DIR / "fr_train.npz", a=F["A"], names=names_arr, domain=dom)
    C.save_npz(TRAIN_DIR / "sr_train.npz", z=F["Z"], names=names_arr, domain=dom)
    pr = np.concatenate([np.stack([rng.choice(np.flatnonzero(dom == d), 700),
                                   rng.choice(np.flatnonzero(dom == d), 700)], 1) for d in DOMS])
    rep["train_identity"] = identity_check(F, _W(cn), pr[pr[:, 0] != pr[:, 1]])


def stage_dist(cn: Canon, kind: str) -> dict:
    src = TRAIN_DIR / f"{kind}_train.npz"
    with np.load(src) as z:
        V, names_arr, dom = z["a" if kind == "fr" else "z"], z["names"], z["domain"].astype(str)
    with np.load(RUN_GT) as z:
        if not np.array_equal(z["names"], names_arr):
            raise SystemExit(f"{src}: names diversi dalla GT del run")
    med_max = json.loads(UGT_CALIB_JSON.read_text())["median_maxabs"]
    note = {"bfm_caveat": "BFM: original REMESH gia' allineate per similarita' una per una (domains.bfm): la taglia "
                          "di BFM non e' quella del modello", "shapes": str(src)}
    return save_train(kind, V, cn.sp.A, names_arr, dom, med_max, note)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=("shapes", "eval", "dist"))
    p.add_argument("--kind", choices=("fr", "sr"), default="fr")
    a = p.parse_args()
    workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")), 32)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cn = Canon()
    if not cn.frames:
        raise SystemExit(f"{cgt.FRAMES_JSON} assente: prima gt.py")
    rng = np.random.default_rng(1234)
    out = cgt.EVID_DIR / f"train_gt_{a.stage}{'_' + a.kind if a.stage == 'dist' else ''}.json"
    rep = {"definition": __doc__.split("Training:")[0].strip()}
    t0 = time.time()
    if a.stage == "eval":
        stage_eval(cn, rep, rng)
    elif a.stage == "shapes":
        stage_shapes(cn, rep, rng, workers)
    else:
        rep[a.kind] = stage_dist(cn, a.kind)
    rep["seconds"] = time.time() - t0
    C.save_json(out, rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "definition"}, indent=1, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
