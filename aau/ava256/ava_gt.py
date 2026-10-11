#!/usr/bin/env python3
"""Unita', taglia, qualita', split e GT FR / SR di Ava-256.

    v3_work/unified_gt/run.sh aau/ava256/ava_gt.py          (dopo ava_neutral.py neutral)

CONFERMATIVO: non valutare prima del protocollo confermativo (README). Solo dati e GT: nessun modello, nessuna
baseline (nemmeno "solo taglia"), nessuna metrica di prestazione. Le sole statistiche riguardano la GT stessa.

Scelte: README, scelte 4-6 e sezione "Modifiche dell'11 ottobre" (decisioni del PI PRIMA di qualsiasi valutazione).
1. Unita': **u = 1 (mm), dichiarata** per analogia con Multiface (stesso laboratorio e stessa famiglia di topologia, mm
   in ``domains.py``) e per la coerenza della centroid size della regione con la media FLAME in mm (E12 ``size_cv.csv``).
   Si riporta anche la regola letterale di E12 (63 mm / IPD mediana, ``cgt.nearest_unit``): una scala globale, che dentro
   la tolleranza non cambia gli Spearman. Controlli: IPD, larghezza intercantale (iBUG 39-42) e biorbitale (36-45) dai
   sostituti di ``cgt.eye_proxies`` contro la media FLAME con la stessa costruzione; scala della similarita' canonica di
   ``correspond`` (template della prima cattura -> media FLAME).
2. Split fisso (``ava_common.split``): 56 soggetti di calibrazione (template del NICP, regione del fit B, L_d, cs_ref) e
   200 di valutazione, per sha256 del sid con un sale dichiarato.
3. GT: ``train_fr_sr.factor`` sui punti della regione unificata delle neutre (frame di dominio ``u``, R = I, t = 0; segue
   la rigida robusta per identita' verso mu), scritte da ``train_fr_sr.save_eval`` nella cartella delle GT di Ava-256
   (NON ``datasets/CANONICAL_GT/eval``): ``ava256_eval_*`` sui 200 di valutazione (la GT della prova), ``ava256_all_*`` su
   tutti i 256 (affidabilita' e taglia); ``identity_check`` su 400 coppie (seme 1234).
4. Taglia (256): centroid size (S di FR), IPD e altezza della regione allineata; CV con IC 95% bootstrap per soggetto
   (1000 repliche, seme 1234); una riga nel formato di ``aau/runs/evidence/e12/size_cv.csv``.
5. Qualita': soggetti validi; **stabilita' della neutra su TUTTI i frame** del segmento (distanza g dal medoide, mm;
   segnalato chi ha un frame oltre 1 mm); anomali (centroid size, IPD, distanza da mu dopo Procrustes oltre mediana +-
   5 x 1.4826 MAD). Tolta la regola MAD sulla distanza fra le meta' pari e dispari della neutra (frame consecutivi: non
   informativa). I segnalati restano nella GT; i loro id stanno in ``datasets/AVA256/gt/quality_flags.json`` (fuori da
   git), in git solo i conteggi.
6. Affidabilita' della GT (solo GT contro GT, 256 soggetti): FR / SR della neutra ripetuta (EXP_eye_neutral) e con la mappa
   di Multiface, Spearman sulle coppie i < j; distanza FR fra neutra e neutro ripetuto della stessa persona contro la
   mediana delle distanze fra persone; quante persone hanno il proprio neutro ripetuto come vicino piu' prossimo. E' una
   stima del rumore fra segmenti della stessa sessione, NON un tetto per gli Spearman dei metodi: viste e GT vengono dalla
   stessa neutra.

Uscite: ``datasets/AVA256/gt/`` (``ava256_{eval,all}_{fr,sr}.npz`` + json, nomi = id delle viste, id950000-;
``*_centroid_size.npz``; ``ids.json``; ``split.json``; ``frame.json`` con u, R, t di dominio come ``frames.json`` di E12;
``quality_flags.json``; ``diagnostics/``); in git ``aau/ava256/{gt_summary.json, size_cv.csv, split.json}`` (numeri
aggregati e id, nessuna geometria).
"""

from __future__ import annotations

import hashlib
import json
import shutil
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
U_MM = 1.0                                   # dichiarata (scelta 1 della docstring)
CORR_NPZ = ac.DATA_ROOT / "corr" / f"{NAME}.npz"
TEMPLATE = ac.DATA_ROOT / "template.npz"
DIAG_DIR = ac.GT_DIR / "diagnostics"
SUMMARY = ac.SUMMARY_DIR / "gt_summary.json"
SIZE_CSV = ac.SUMMARY_DIR / "size_cv.csv"
SPLIT_JSON = ac.SUMMARY_DIR / "split.json"
E12_SIZE_CSV = REPO_ROOT / "aau" / "runs" / "evidence" / "e12" / "size_cv.csv"
N_BOOT, SEED = 1000, 1234
MAD_K = 5.0
STABILITY_MM = 1.0
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
    nello zip). E' l'impronta da citare nel protocollo (e quella che controlla ``ava_freeze.verify_frozen``)."""
    with np.load(path) as z:
        h = hashlib.sha256(np.ascontiguousarray(z["D_orig"], dtype="<f4").tobytes())
        h.update("\n".join(str(s) for s in z["names"]).encode())
    return h.hexdigest()


def load() -> dict:
    """Punti della regione unificata (mappa nuova e di Multiface) delle neutre e dei neutri ripetuti (unita' native),
    stabilita' della neutra (distanze dal medoide di tutti i frame finiti)."""
    with np.load(CORR_NPZ) as z:
        maps = {"new": (z["vidx_unified"], z["bary_unified"]), "mf": (z["vidx_multiface"], z["bary_multiface"])}
    caps = ac.captures()
    out = {"caps": caps, "valid": [], "rep_ok": [], "stab_max": [], "kept_frac": [], "P": [], "P_mf": [], "P_rep": []}
    nan = np.full((ac.N_VERTS, 3), np.nan)
    for c in caps:
        with np.load(ac.NEUTRAL_DIR / f"{c['ava_id']}.npz") as z:
            ok = bool(z["valid"])
            rep = "V_repeat" in z.files
            Vn = z["V_neutral"].astype(np.float64) if ok else nan
            Vr = z["V_repeat"].astype(np.float64) if rep else nan
            out["stab_max"].append(float(z["neutral_dist_to_medoid_mm"].max()) if ok else np.nan)
            out["kept_frac"].append(len(z["neutral_kept"]) / max(int(z["neutral_finite"].sum()), 1) if ok else np.nan)
        out["valid"].append(ok)
        out["rep_ok"].append(rep and ok)
        out["P"].append(C.bary_interp(Vn, *maps["new"]))
        out["P_mf"].append(C.bary_interp(Vn, *maps["mf"]))
        out["P_rep"].append(C.bary_interp(Vr, *maps["new"]))
    for k in ("P", "P_mf", "P_rep"):
        out[k] = np.stack(out[k])
    for k in ("valid", "rep_ok", "stab_max", "kept_frac"):
        out[k] = np.asarray(out[k])
    return out


def units(cn: Canon, P: np.ndarray, eyes: dict) -> dict:
    """u = 1 dichiarata; controlli anatomici e regola letterale di E12 (riportata, non usata)."""
    W = widths(P, eyes["idx"])
    ipd_med = float(np.median(W["ipd"]))
    est = cgt.IPD_REF_MM / ipd_med
    name, _, ratio = cgt.nearest_unit(est)
    flame = domains.flame()
    P_fl = (flame["V"] * 1000.0)[cn.sp.z["flame_vidx"]][None]
    Wf = {k: float(v[0]) for k, v in widths(P_fl, eyes["idx"]).items()}
    with np.load(TEMPLATE) as z, np.load(CORR_NPZ) as zc:
        M = C.bary_interp(z["V"], zc["vidx_unified"], zc["bary_unified"])
    Wt = {k: float(v[0]) for k, v in widths(M[None], eyes["idx"]).items()}
    s_canon = json.loads((ac.DATA_ROOT / "corr" / f"{NAME}.json").read_text())["canonical"]["s"]
    return {"u": U_MM, "unit": "millimetri (dichiarata)",
            "reason": "analogia con Multiface (stesso laboratorio, stessa famiglia di topologia, mm in domains.py) e "
                      "coerenza della centroid size della regione con la media FLAME in mm (size, sotto)",
            "e12_literal_rule": {"ipd_native_median": ipd_med, "u_63_over_ipd": est, "nearest_power_of_ten": name,
                                 "ratio_to_power_of_ten": ratio, "tolerance": cgt.UNIT_TOL,
                                 "note": "scala globale entro la tolleranza: non cambia gli Spearman dentro il dominio"},
            "widths_mm": {k: q(v * U_MM) for k, v in W.items()},
            "widths_flame_mean_same_proxies_mm": Wf,
            "template_over_median_width": {k: Wt[k] / float(np.median(W[k])) for k in Wt},
            "anthropometric_reference_mm": ANTHRO_MM,
            "eye_proxy_distance_on_flame_mm": [float(x) for x in eyes["proxy_mm"]],
            "canonical_similarity_s_template_to_flame": s_canon,
            "note": "proxy = punto piu' vicino della regione unificata al landmark iBUG sulla media FLAME (E12 2b); "
                    "il template ha la taglia della prima cattura (in git solo il rapporto con la mediana)"}


def domain_frame(cn: Canon, u: float) -> dict:
    """(u, R, t) come ``frames.json`` di E12: Umeyama senza scala da u * mappa(template) alla media FLAME (mm)."""
    flame = domains.flame()
    P_flame = flame["V"][cn.sp.z["flame_vidx"]] * 1000.0
    w_flame = C.vertex_areas(P_flame, cn.sp.z["F"])
    with np.load(TEMPLATE) as z, np.load(CORR_NPZ) as zc:
        M = C.bary_interp(z["V"], zc["vidx_unified"], zc["bary_unified"])
    _, R, t = C.umeyama(u * M, P_flame, w_flame, scale=False)
    return {"u": u, "R": R.tolist(), "t": t.tolist(), "unit_source": "dichiarata: mm (ava_gt.py, scelta 1)",
            "description": "f = u * R @ p + t (p frame e unita' dei dati, le viste); come aau/runs/evidence/e12/"
                           "frames.json; la GT FR non dipende da R, t (segue la rigida robusta per identita'). Serve agli "
                           "strumenti in mm (baselines_mm, tabella di scala), dopo il protocollo"}


def aligned(F: dict, cn: Canon) -> np.ndarray:
    return F["A"].astype(np.float64).reshape(len(F["A"]), -1, 3) / np.sqrt(cn.sp.w)[None, :, None]


def subset(F: dict, rows: np.ndarray) -> dict:
    return {k: v[rows] for k, v in F.items()}


def clean_gt_dir() -> None:
    """Toglie le GT di un'esecuzione precedente (nomi vecchi compresi): la cartella contiene solo quelle di questa."""
    if ac.GT_DIR.exists():
        for p in ac.GT_DIR.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            elif p.suffix in (".npz", ".json"):
                p.unlink()


def main() -> None:
    if ac.FROZEN.exists():
        raise SystemExit(f"{ac.FROZEN}: GT e viste congelate, non le riscrivo (serve una decisione del PI)")
    rng = np.random.default_rng(SEED)
    cn = Canon()
    data = load()
    caps, valid = data["caps"], data["valid"]
    iv = np.flatnonzero(valid)
    eyes = cgt.eye_proxies(cn)
    un = units(cn, data["P"][iv], eyes)
    cn.frames[NAME] = {"u": U_MM, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}
    sp = ac.split()
    pos = {caps[i]["ava_id"]: k for k, i in enumerate(iv)}
    ie = np.array([pos[a] for a in sp["evaluation"] if a in pos])
    ic = np.array([pos[a] for a in sp["calibration"] if a in pos])
    ids = [caps[i]["view_id"] for i in iv]

    # GT: tutti i validi (affidabilita', taglia) e i 200 di valutazione (la GT della prova)
    clean_gt_dir()
    ac.GT_DIR.mkdir(parents=True, exist_ok=True)
    F = tfs.factor(cn, NAME, data["P"][iv])
    tfs.EVAL_DIR = ac.GT_DIR                       # stesso scrittore, cartella di Ava-256
    D = tfs.save_eval(f"{NAME}_all", ids, F, cn.sp.A)
    D_eval = tfs.save_eval(f"{NAME}_eval", [ids[k] for k in ie], subset(F, ie), cn.sp.A)
    rows = [{"view_id": caps[i]["view_id"], "ava_id": caps[i]["ava_id"], "capture": caps[i]["capture"]} for i in iv]
    C.save_json(ac.GT_DIR / "ids.json", {"note": "righe di ava256_all_* (tutti i validi) e di ava256_eval_* (valutazione)",
                                         "all": rows, "eval": [rows[k] for k in ie]})
    split = {"rule": f"sha256(SPLIT_SALT + sid) in ordine crescente: i primi {ac.N_CALIB} sono di calibrazione",
             "salt": ac.SPLIT_SALT, "n_calibration": int(len(ic)), "n_evaluation": int(len(ie)),
             "calibration": [rows[k]["ava_id"] for k in ic], "evaluation": [rows[k]["ava_id"] for k in ie],
             "calibration_view_ids": [rows[k]["view_id"] for k in ic],
             "evaluation_view_ids": [rows[k]["view_id"] for k in ie],
             "use": "calibrazione: SOLO template del NICP, regione del fit B, L_d e cs_ref; mai valutati, da nessun metodo"}
    C.save_json(ac.GT_DIR / "split.json", split)
    C.save_json(SPLIT_JSON, split)
    C.save_json(ac.GT_DIR / "frame.json", domain_frame(cn, U_MM))
    pr = rng.choice(len(iv), size=(400, 2))
    ident = tfs.identity_check(F, tfs._W(cn), pr[pr[:, 0] != pr[:, 1]])
    iu = np.triu_indices(len(iv), 1)
    iue = np.triu_indices(len(ie), 1)

    # taglia (formato di size_cv.csv di E12), tutti i validi
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
            "centroid_size_vs_flame_mean": float(cs.mean() / float(e12.set_index("set").loc["flame", "centroid_size_mean_mm"])),
            "e12_centroid_size_mean_mm": {r["set"]: float(r["centroid_size_mean_mm"]) for _, r in e12.iterrows()},
            "e12_centroid_size_cv": {r["set"]: float(r["centroid_size_cv"]) for _, r in e12.iterrows()}}

    # qualita': stabilita' su tutti i frame e anomali; segnalati elencati fuori da git, non esclusi
    _, _, rms_mu = cn.sp.align(data["P"][iv])
    crit = {"centroid_size": cs, "ipd": ip, "rms_to_mu_after_procrustes": rms_mu}
    flags = {k: mad_flags(np.asarray(v, dtype=float)) for k, v in crit.items()}
    flags["neutral_frame_over_1mm"] = data["stab_max"][iv] > STABILITY_MM
    flagged = sorted({caps[iv[j]]["ava_id"] for f in flags.values() for j in np.flatnonzero(f)})
    C.save_json(ac.GT_DIR / "quality_flags.json", {k: [caps[iv[j]]["ava_id"] for j in np.flatnonzero(f)]
                                                   for k, f in flags.items()})
    quality = {"n_captures": len(caps), "n_valid": int(valid.sum()),
               "n_invalid": int((~valid).sum()),
               "robust_rigid": {"converged_fraction": float(F["conv"].mean()),
                                "iterations": q(F["it"]), "angle_deg": q(F["angle"])},
               "stability_rule": f"distanza g (GT unificata, mm) di TUTTI i frame finiti di EXP_neutral_peak dal medoide; "
                                 f"segnalato chi ne ha uno oltre {STABILITY_MM:g} mm",
               "stability_max_dist_mm": q(data["stab_max"][iv]), "kept_fraction": q(data["kept_frac"][iv]),
               "anomaly_rule": f"oltre mediana +- {MAD_K:g} x 1.4826 MAD (centroid size, IPD, distanza da mu)",
               "n_flagged_by_rule": {k: int(f.sum()) for k, f in flags.items()}, "n_flagged": len(flagged),
               "n_flagged_in_evaluation": int(len(set(flagged) & set(split["evaluation"]))),
               "flagged_ids": "datasets/AVA256/gt/quality_flags.json (fuori da git)",
               "criteria_stats": {k: q(v) for k, v in crit.items()},
               "non_finite_points": int((~np.isfinite(data["P"][iv])).sum())}

    # affidabilita' della GT (GT contro GT, tutti i validi)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    tfs.EVAL_DIR = DIAG_DIR
    rel = {}
    ir = np.flatnonzero(data["rep_ok"][iv])               # posizioni (fra i validi) con il neutro ripetuto
    F_rep = tfs.factor(cn, NAME, data["P_rep"][iv[ir]])
    D_rep = tfs.save_eval(f"{NAME}_repeat", [ids[k] for k in ir], F_rep, cn.sp.A)
    a_rep = aligned(F_rep, cn)
    within = np.sqrt(np.einsum("v,nv->n", cn.W, ((a[ir] - a_rep) ** 2).sum(-1)))
    sq = np.sqrt(cn.W)[None, :, None]
    X, Y = (sq * a).reshape(len(a), -1), (sq * a_rep).reshape(len(a_rep), -1)
    cross = np.sqrt(np.clip((Y ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2 * Y @ X.T, 0, None))
    between = D["fr"][iu]
    rel["repeat_eye_neutral"] = {
        "n": int(len(ir)),
        "spearman_fr": spearman(D["fr"][np.ix_(ir, ir)], D_rep["fr"]),
        "spearman_sr": spearman(D["sr"][np.ix_(ir, ir)], D_rep["sr"]),
        "within_person_fr_mm": q(within), "between_person_fr_mm": q(between),
        "within_over_between_median": float(np.median(within) / np.median(between)),
        "nearest_neutral_is_own_fraction": float(np.mean(cross.argmin(1) == ir))}
    F_mf = tfs.factor(cn, NAME, data["P_mf"][iv])
    D_mf = tfs.save_eval(f"{NAME}_mfmap", ids, F_mf, cn.sp.A)
    rel["multiface_index_map"] = {"spearman_fr": spearman(D["fr"], D_mf["fr"]),
                                  "spearman_sr": spearman(D["sr"], D_mf["sr"]),
                                  "abs_diff_fr_mm": q(np.abs(D["fr"] - D_mf["fr"])[iu])}
    rel["note"] = ("statistiche della sola GT, nessun modello ne' baseline. Stima del rumore fra segmenti diversi della "
                   "stessa sessione (EXP_neutral_peak contro EXP_eye_neutral, che contiene anche espressione): NON e' un "
                   "tetto per gli Spearman dei metodi, perche' viste e GT vengono dalla stessa neutra")

    gt = {"n_all": len(iv), "n_eval": int(len(ie)), "n_calibration": int(len(ic)),
          "ids": "datasets/AVA256/gt/ids.json (id950000 + riga di 256_ids.csv)",
          "eval_fr_mm": q(D_eval["fr"][iue]), "eval_sr_dP": q(D_eval["sr"][iue]),
          "all_fr_mm": q(D["fr"][iu]), "all_sr_dP": q(D["sr"][iu]), "identity_check": ident,
          "content_sha256": {f"{s}_{k}": content_sha256(ac.GT_DIR / f"{NAME}_{s}_{k}.npz")
                             for s in ("eval", "all") for k in ("fr", "sr")},
          "files": "datasets/AVA256/gt/ava256_{eval,all}_{fr,sr}.npz (D_orig / massimo, json con l'unita'), "
                   "*_centroid_size.npz, split.json, frame.json"}
    summ = {"definition": __doc__.split("Uscite:")[0].strip(), "units": un, "size": size, "quality": quality,
            "gt": gt, "gt_reliability": rel}
    C.save_json(SUMMARY, summ)
    print(json.dumps({"u": U_MM, "e12_literal_u": un["e12_literal_rule"]["u_63_over_ipd"], "n_valid": quality["n_valid"],
                      "split": [len(ic), len(ie)], "n_flagged": quality["n_flagged_by_rule"], "size_row": row,
                      "cv_ci95": size["cv_ci95"], "eval_fr_mm_median": gt["eval_fr_mm"]["median"],
                      "content_sha256": gt["content_sha256"],
                      "reliability": {k: v for k, v in rel.items() if k != "note"}}, indent=1, default=float)[:4000],
          flush=True)


if __name__ == "__main__":
    main()
