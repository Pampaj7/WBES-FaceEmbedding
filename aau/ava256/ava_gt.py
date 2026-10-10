#!/usr/bin/env python3
"""Unita', taglia, qualita' e GT FR / SR di Ava-256.

    v3_work/unified_gt/run.sh aau/ava256/ava_gt.py          (dopo ava_neutral.py neutral)

CONFERMATIVO: non valutare prima del protocollo confermativo (README). Solo dati e GT: nessun modello, nessuna
baseline (nemmeno "solo taglia"), nessuna metrica di prestazione. Le sole statistiche riguardano la GT stessa.

1. Unita' (README, scelta 4; come E12, ``v3_work/canonical_gt/gt.py`` sez. 2b): IPD in unita' native dai 12 punti
   del contorno degli occhi (``cgt.eye_proxies``: iBUG 36-47 sulla media FLAME, punto piu' vicino della regione
   unificata). La forma media del dominio e' rappresentata dalla mediana delle IPD delle neutre: il template ha la
   taglia della prima cattura (``domains.multiface`` fa lo stesso), la sua IPD si riporta a parte. ``u`` = potenza di
   10 piu' vicina a 63 / IPD entro +-15% (``cgt.nearest_unit``), altrimenti 63 / IPD. Controlli: larghezza
   intercantale (iBUG 39-42) e biorbitale (36-45) con gli stessi sostituti, confrontate con la media FLAME (stessa
   costruzione, mm) e con i valori antropometrici; centroid size della regione contro gli altri domini (E12
   ``size_cv.csv``); scala della similarita' canonica di ``correspond``.
2. GT (scelta 5): ``train_fr_sr.factor`` sui punti della regione unificata delle neutre (frame di dominio: ``u``,
   R = I, t = 0; segue la rigida robusta per identita' verso mu), scritte da ``train_fr_sr.save_eval`` nella cartella
   delle GT di Ava-256 (NON ``datasets/CANONICAL_GT/eval``); ``identity_check`` su 400 coppie (seme 1234).
3. Taglia: centroid size (S di FR), IPD e altezza della regione allineata; CV con IC 95% bootstrap per soggetto
   (1000 repliche, seme 1234); una riga nel formato di ``aau/runs/evidence/e12/size_cv.csv``.
4. Qualita' (scelta 6): soggetti validi; anomalie (centroid size, IPD, distanza da mu dopo Procrustes, rumore della
   neutra oltre mediana +- 5 x 1.4826 MAD) elencate e NON escluse.
5. Affidabilita' della GT (solo GT contro GT): FR / SR della neutra ripetuta (EXP_eye_neutral) e con la mappa di
   Multiface, Spearman sulle coppie i < j; distanza FR fra neutra e neutro ripetuto della stessa persona contro la
   mediana delle distanze FR fra persone, e quante persone hanno il proprio neutro ripetuto come vicino piu' prossimo.

Uscite: ``datasets/AVA256/gt/ava256_{fr,sr}.npz`` + json (nomi = id delle viste, id950000-), ``ava256_centroid_size.npz``,
``ids.json`` (id della vista, id Ava, cattura), ``frame.json`` (u, R, t di dominio come ``frames.json`` di E12, per
strumenti futuri), ``diagnostics/`` (GT delle varianti); in git ``aau/ava256/gt_summary.json`` e
``aau/ava256/size_cv.csv`` (numeri aggregati e id, nessuna geometria).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (THIS_DIR, REPO_ROOT / "v3_work" / "canonical_gt"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ava_common as ac  # noqa: E402
import cgt  # noqa: E402
import train_fr_sr as tfs  # noqa: E402
from cgt import C, Canon, domains  # noqa: E402

NAME = "ava256"
CORR_NPZ = ac.DATA_ROOT / "corr" / f"{NAME}.npz"
TEMPLATE = ac.DATA_ROOT / "template.npz"
DIAG_DIR = ac.GT_DIR / "diagnostics"
SUMMARY = ac.SUMMARY_DIR / "gt_summary.json"
SIZE_CSV = ac.SUMMARY_DIR / "size_cv.csv"
E12_SIZE_CSV = REPO_ROOT / "aau" / "runs" / "evidence" / "e12" / "size_cv.csv"
N_BOOT, SEED = 1000, 1234
MAD_K = 5.0
# iBUG 36, 39, 42, 45 nei 12 sostituti di cgt.eye_proxies (EYE_R = 36-41, EYE_L = 42-47)
EX_R, EN_R, EN_L, EX_L = 0, 3, 6, 9
ANTHRO_MM = {"ipd": 63.0, "intercanthal": (31.0, 33.0), "biocular": (87.0, 91.0)}


def q(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "min": float(x.min()),
            "p5": float(np.percentile(x, 5)), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "max": float(x.max())}


def widths(P: np.ndarray, idx: np.ndarray) -> dict:
    """IPD, larghezza intercantale e biorbitale (n,) dai 12 sostituti degli occhi."""
    d = lambda a, b: np.linalg.norm(P[:, idx[a]] - P[:, idx[b]], axis=1)  # noqa: E731
    return {"ipd": cgt.ipd(P, idx), "intercanthal": d(EN_R, EN_L), "biocular": d(EX_R, EX_L)}


def spearman(A: np.ndarray, B: np.ndarray) -> float:
    iu = np.triu_indices(len(A), 1)
    ra = np.argsort(np.argsort(A[iu])).astype(float)
    rb = np.argsort(np.argsort(B[iu])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def mad_flags(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    return np.abs(x - med) > MAD_K * mad


def boot_cv(x: np.ndarray, rng) -> list[float]:
    n = len(x)
    cv = [float(x[i].std(ddof=1) / x[i].mean()) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return [float(np.percentile(cv, 2.5)), float(np.percentile(cv, 97.5))]


def content_sha256(path: Path) -> str:
    """sha256 del contenuto di una GT (``D_orig`` float32 e nomi): stabile fra riesecuzioni, il file npz no (date
    nello zip). E' l'impronta da citare nel protocollo."""
    with np.load(path) as z:
        h = hashlib.sha256(np.ascontiguousarray(z["D_orig"], dtype="<f4").tobytes())
        h.update("\n".join(str(s) for s in z["names"]).encode())
    return h.hexdigest()


def load() -> dict:
    """Punti della regione unificata (mappa nuova e di Multiface) delle neutre e dei neutri ripetuti, unita' native."""
    with np.load(CORR_NPZ) as z:
        maps = {"new": (z["vidx_unified"], z["bary_unified"]), "mf": (z["vidx_multiface"], z["bary_multiface"])}
    caps = ac.captures()
    out = {"caps": caps, "valid": [], "rep_ok": [], "halves": [], "P": [], "P_mf": [], "P_rep": []}
    for c in caps:
        with np.load(ac.NEUTRAL_DIR / f"{c['ava_id']}.npz") as z:
            ok = bool(z["valid"])
            rep = "V_repeat" in z.files
            nan = np.full((ac.N_VERTS, 3), np.nan)
            Vn = z["V_neutral"].astype(np.float64) if ok else nan
            Vr = z["V_repeat"].astype(np.float64) if rep else nan
            out["halves"].append(float(z["neutral_halves_mm"]) if ok else np.nan)
        out["valid"].append(ok)
        out["rep_ok"].append(rep and ok)
        out["P"].append(C.bary_interp(Vn, *maps["new"]))
        out["P_mf"].append(C.bary_interp(Vn, *maps["mf"]))
        out["P_rep"].append(C.bary_interp(Vr, *maps["new"]))
    for k in ("P", "P_mf", "P_rep"):
        out[k] = np.stack(out[k])
    for k in ("valid", "rep_ok", "halves"):
        out[k] = np.asarray(out[k])
    return out


def units(cn: Canon, P: np.ndarray, eyes: dict) -> dict:
    """Unita' native (scelta 4): IPD e controlli; ``u`` in mm per unita'."""
    W = widths(P, eyes["idx"])
    ipd_med = float(np.median(W["ipd"]))
    est = cgt.IPD_REF_MM / ipd_med
    name, u_pow, ratio = cgt.nearest_unit(est)
    u = u_pow if name != "arbitraria" else est
    flame = domains.flame()
    P_fl = (flame["V"] * 1000.0)[cn.sp.z["flame_vidx"]][None]
    Wf = {k: float(v[0]) for k, v in widths(P_fl, eyes["idx"]).items()}
    with np.load(TEMPLATE) as z, np.load(CORR_NPZ) as zc:
        M = C.bary_interp(z["V"], zc["vidx_unified"], zc["bary_unified"])
    Wt = {k: float(v[0]) for k, v in widths(M[None], eyes["idx"]).items()}
    s_canon = json.loads((ac.DATA_ROOT / "corr" / f"{NAME}.json").read_text())["canonical"]["s"]
    return {"u": float(u), "unit": name if name != "arbitraria" else "arbitraria (63 / IPD)",
            "ipd_native_median": ipd_med, "mm_per_unit_from_ipd63": est, "ratio_to_power_of_ten": ratio,
            "tolerance": cgt.UNIT_TOL,
            "widths_mm_with_u": {k: q(v * u) for k, v in W.items()},
            "widths_flame_mean_same_proxies_mm": Wf,
            "template_over_median_width": {k: Wt[k] / float(np.median(W[k])) for k in Wt},
            "anthropometric_reference_mm": ANTHRO_MM,
            "eye_proxy_distance_on_flame_mm": [float(x) for x in eyes["proxy_mm"]],
            "canonical_similarity_s_template_to_flame": s_canon,
            "note": "proxy = punto piu' vicino della regione unificata al landmark iBUG sulla media FLAME (E12 2b); "
                    "i valori FLAME con gli stessi proxy danno il riferimento della stessa costruzione; il template "
                    "ha la taglia della prima cattura (solo il rapporto con la mediana: nessuna misura di un singolo "
                    "soggetto in git)"}


def domain_frame(cn: Canon, u: float) -> dict:
    """(u, R, t) come ``frames.json`` di E12: Umeyama senza scala da u * mappa(template) alla media FLAME (mm)."""
    flame = domains.flame()
    P_flame = flame["V"][cn.sp.z["flame_vidx"]] * 1000.0
    w_flame = C.vertex_areas(P_flame, cn.sp.z["F"])
    with np.load(TEMPLATE) as z, np.load(CORR_NPZ) as zc:
        M = C.bary_interp(z["V"], zc["vidx_unified"], zc["bary_unified"])
    _, R, t = C.umeyama(u * M, P_flame, w_flame, scale=False)
    return {"u": u, "R": R.tolist(), "t": t.tolist(),
            "description": "f = u * R @ p + t (p frame e unita' dei dati, le viste); come aau/runs/evidence/e12/"
                           "frames.json; la GT FR non dipende da R, t (segue la rigida robusta per identita'). "
                           "NON usato per GT o viste: serve a strumenti futuri, dopo il protocollo"}


def factor_set(cn: Canon, P: np.ndarray) -> dict:
    return tfs.factor(cn, NAME, P)


def aligned(F: dict, cn: Canon) -> np.ndarray:
    return F["A"].astype(np.float64).reshape(len(F["A"]), -1, 3) / np.sqrt(cn.sp.w)[None, :, None]


def main() -> None:
    rng = np.random.default_rng(SEED)
    cn = Canon()
    data = load()
    caps, valid = data["caps"], data["valid"]
    iv = np.flatnonzero(valid)
    eyes = cgt.eye_proxies(cn)
    un = units(cn, data["P"][iv], eyes)
    u = un["u"]
    cn.frames[NAME] = {"u": u, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}
    ids = [caps[i]["view_id"] for i in iv]

    # GT FR / SR
    ac.GT_DIR.mkdir(parents=True, exist_ok=True)
    F = factor_set(cn, data["P"][iv])
    tfs.EVAL_DIR = ac.GT_DIR                       # stesso scrittore, cartella di Ava-256
    D = tfs.save_eval(NAME, ids, F, cn.sp.A)
    C.save_json(ac.GT_DIR / "ids.json", {"note": "ordine delle righe delle GT", "ids": [
        {"view_id": caps[i]["view_id"], "ava_id": caps[i]["ava_id"], "capture": caps[i]["capture"]} for i in iv]})
    C.save_json(ac.GT_DIR / "frame.json", domain_frame(cn, u))
    pr = rng.choice(len(iv), size=(400, 2))
    ident = tfs.identity_check(F, tfs._W(cn), pr[pr[:, 0] != pr[:, 1]])
    iu = np.triu_indices(len(iv), 1)

    # taglia (formato di size_cv.csv di E12)
    a = aligned(F, cn)
    cs, ip, h = F["S"], cgt.ipd(a, eyes["idx"]), cn.height(a)
    row = {"set": NAME, "n": len(iv)}
    for k, v in (("centroid_size", cs), ("ipd", ip), ("height", h)):
        row.update({f"{k}_mean_mm": float(v.mean()), f"{k}_sd_mm": float(v.std(ddof=1)),
                    f"{k}_cv": float(v.std(ddof=1) / v.mean())})
    row["corr_cs_ipd"] = float(np.corrcoef(cs, ip)[0, 1])
    pd.DataFrame([row]).to_csv(SIZE_CSV, index=False)
    e12 = pd.read_csv(E12_SIZE_CSV)
    size = {"row": row, "cv_ci95": {k: boot_cv(v, np.random.default_rng(SEED)) for k, v in
                                    (("centroid_size", cs), ("ipd", ip), ("height", h))},
            "e12_centroid_size_mean_mm": {r["set"]: float(r["centroid_size_mean_mm"]) for _, r in e12.iterrows()},
            "e12_centroid_size_cv": {r["set"]: float(r["centroid_size_cv"]) for _, r in e12.iterrows()}}

    # qualita': anomalie elencate, non escluse
    _, _, rms_mu = cn.sp.align(data["P"][iv])
    crit = {"centroid_size": cs, "ipd": ip, "rms_to_mu_after_procrustes": rms_mu,
            "neutral_halves_mm": data["halves"][iv]}
    flags = {k: mad_flags(np.asarray(v, dtype=float)) for k, v in crit.items()}
    flagged = sorted({caps[iv[j]]["ava_id"] for f in flags.values() for j in np.flatnonzero(f)})
    quality = {"n_captures": len(caps), "n_valid": int(valid.sum()),
               "invalid": [c["ava_id"] for c, ok in zip(caps, valid) if not ok],
               "robust_rigid": {"converged_fraction": float(F["conv"].mean()),
                                "iterations": q(F["it"]), "angle_deg": q(F["angle"])},
               "rule": f"anomalo = oltre mediana +- {MAD_K:g} x 1.4826 MAD (elencati, NON esclusi)",
               "flagged": {k: [caps[iv[j]]["ava_id"] for j in np.flatnonzero(f)] for k, f in flags.items()},
               "n_flagged": len(flagged), "criteria_stats": {k: q(v) for k, v in crit.items()},
               "non_finite_points": int((~np.isfinite(data["P"][iv])).sum())}

    # affidabilita' della GT (GT contro GT)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    tfs.EVAL_DIR = DIAG_DIR
    rel = {}
    ir = np.flatnonzero(data["rep_ok"][iv])               # posizioni (fra i validi) con il neutro ripetuto
    F_rep = factor_set(cn, data["P_rep"][iv[ir]])
    D_rep = tfs.save_eval(f"{NAME}_repeat", [ids[k] for k in ir], F_rep, cn.sp.A)
    a_rep = aligned(F_rep, cn)
    within = np.sqrt(np.einsum("v,nv->n", cn.W, ((a[ir] - a_rep) ** 2).sum(-1)))
    # vicino piu' prossimo del neutro ripetuto fra le neutre di tutti (FR, stessa formula)
    sq = np.sqrt(cn.W)[None, :, None]
    X, Y = (sq * a).reshape(len(a), -1), (sq * a_rep).reshape(len(a_rep), -1)
    cross = np.sqrt(np.clip((Y ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2 * Y @ X.T, 0, None))
    nn_self = float(np.mean(cross.argmin(1) == ir))
    between = D["fr"][iu]
    rel["repeat_eye_neutral"] = {
        "n": int(len(ir)),
        "spearman_fr": spearman(D["fr"][np.ix_(ir, ir)], D_rep["fr"]),
        "spearman_sr": spearman(D["sr"][np.ix_(ir, ir)], D_rep["sr"]),
        "within_person_fr_mm": q(within), "between_person_fr_mm": q(between),
        "within_over_between_median": float(np.median(within) / np.median(between)),
        "nearest_neutral_is_own_fraction": nn_self}
    F_mf = factor_set(cn, data["P_mf"][iv])
    D_mf = tfs.save_eval(f"{NAME}_mfmap", ids, F_mf, cn.sp.A)
    rel["multiface_index_map"] = {"spearman_fr": spearman(D["fr"], D_mf["fr"]),
                                  "spearman_sr": spearman(D["sr"], D_mf["sr"]),
                                  "abs_diff_fr_mm": q(np.abs(D["fr"] - D_mf["fr"])[iu])}
    rel["neutral_split_halves_unified_mm"] = q(data["halves"][iv])
    rel["note"] = ("statistiche della sola GT: nessun modello ne' baseline; within = FR fra neutra (EXP_neutral_peak) "
                   "e neutro ripetuto (EXP_eye_neutral) della stessa persona, entrambi con la rigida robusta verso mu")

    gt = {"n": len(iv), "ids": "datasets/AVA256/gt/ids.json (id950000 + riga di 256_ids.csv)",
          "fr_mm": q(D["fr"][iu]), "sr_dP": q(D["sr"][iu]), "identity_check": ident,
          "content_sha256": {k: content_sha256(ac.GT_DIR / f"{NAME}_{k}.npz") for k in ("fr", "sr")},
          "files": "datasets/AVA256/gt/ava256_{fr,sr}.npz (D_orig / massimo, json con l'unita'), "
                   "ava256_centroid_size.npz, frame.json"}
    summ = {"definition": __doc__.split("Uscite:")[0].strip(), "units": un, "size": size, "quality": quality,
            "gt": gt, "gt_reliability": rel}
    C.save_json(SUMMARY, summ)
    print(json.dumps({"u": u, "unit": un["unit"], "ipd_native_median": un["ipd_native_median"],
                      "n_valid": quality["n_valid"], "n_flagged": quality["n_flagged"], "size_row": row,
                      "cv_ci95": size["cv_ci95"], "fr_mm_median": gt["fr_mm"]["median"],
                      "reliability": {k: v for k, v in rel.items() if k != "note"}}, indent=1, default=float)[:4000],
          flush=True)


if __name__ == "__main__":
    main()
