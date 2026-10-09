#!/usr/bin/env python3
"""Bersagli della testa fattorizzata (``--head factorized``): GT "shape" e centroid size per identita'.

Per l'identita' i del dominio d (BFM, ICT, GNM), con le definizioni di E12 (aau/runs/evidence/e12/protocol.md):
  p_i = mappa baricentrica del dominio sulla regione unificata (1478 punti, ``unified_space.npz``), dalla forma
        neutra nativa (BFM: original REMESH; ICT e GNM: pesi del modello, come v3_work/unified_gt/shapes.py);
  c_i = u_d R_d p_i + t_d, il frame di GT-F di E12 (aau/runs/evidence/e12/frames.json: unita' fisiche, UNA rigida
        per dominio), nessun allineamento per identita'; W = pesi d'area di mu normalizzati (somma 1);
  S_i = sqrt(sum_v W_v ||c_i,v - m_i||^2)          centroid size in mm (m_i centroide pesato);
  z_i = (c_i - m_i) / S_i                           pre-forma a centroid size UNITARIA;
  dP_ij = sqrt(sum_v W_v ||z_i,v - z_j,v||^2) = 2 sin(rho_ij / 2)   corda fra pre-forme (rho = angolo).
Allora (S_i - S_j)^2 + S_i S_j dP_ij^2 = sum_v W_v ||(c_i - m_i) - (c_j - m_j)||^2, cioe' la distanza form di
Dryden & Mardia ricostruisce ESATTAMENTE la GT ``F_centered`` di E12 (controllo qui sotto). Con la rotazione
ottimizzata per coppia (Procrustes) la stessa formula da' la distanza size-and-shape di Dryden & Mardia; si
riportano entrambe.

Uscite in ``--out-dir``:
  * ``gt_shape.npz``: ``D_orig`` = kappa * dP (float32), ``names`` come la GT di riferimento (stesso ordine): la
    loss v2 ha margini fissi in unita' di GT, quindi kappa porta la mediana di dP sulle coppie di training dentro
    il dominio sulla mediana della GT maxabs di riferimento (``--kappa`` per fissarlo). ``dp_per_unit`` = 1/kappa
    nel .json: dP = ||u_i - u_j|| * dp_per_unit;
  * ``size.npz``: ``names`` (id), ``cs_mm``, ``log_cs_mm``, ``domain``; statistiche per dominio nel .json.

    v3_work/unified_gt/run.sh v3_work/trainer/tools/build_factorized_targets.py --ref-gt <gt.npz> \
        --split <split.json> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[2]
sys.path.insert(0, str(REPO / "v3_work/unified_gt"))

import domains  # noqa: E402  (v3_work/unified_gt, sola lettura)
from shapes import Space  # noqa: E402

FRAMES = REPO / "aau/runs/evidence/e12/frames.json"   # u_d, R_d, t_d di GT-F (E12)
N_CHECK, SEED = 300, 1234


def domain_of(sid: str) -> str:
    g = int(sid[2:])
    if g < 1000:
        return "bfm"
    if 10000 <= g < 15000 or 20000 <= g < 100000:
        return "ict"
    if 100000 <= g < 200000:
        return "gnm"
    raise SystemExit(f"{sid}: dominio senza forma nativa (solo bfm, ict, gnm)")


def native_points(sp: Space, d: str, sids: list[str]) -> np.ndarray:
    """(n, 1478, 3) nelle unita' e nel frame dei dati del dominio."""
    if d == "bfm":
        return np.stack([sp.map("bfm", domains.load_bfm_original(domains.BFM_REMESH / f"{s}_GTready_original.npz")[0])
                         for s in sids])
    M0, B = sp.linear(d, domains.template(d))
    W = domains.ict_weights(sids) if d == "ict" else domains.gnm_weights(sids)
    return M0[None] + np.einsum("nk,kvd->nvd", W[:, : B.shape[0]], B)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def procrustes_dp(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """dP con la rotazione ottimizzata per coppia (pre-forme centrate a CS 1): 2 sin(rho/2), cos rho = somma dei
    valori singolari di sum_v W_v z_j z_i^T (col segno del determinante)."""
    zw = Z * np.sqrt(W)[None, :, None]
    n = len(Z)
    out = np.zeros((n, n))
    for i in range(n):
        Cm = np.einsum("jva,vb->jab", zw, zw[i])
        sv = np.linalg.svd(Cm, compute_uv=False)
        T = sv[:, 0] + sv[:, 1] + np.sign(np.linalg.det(Cm)) * sv[:, 2]
        out[i] = np.sqrt(np.clip(2.0 - 2.0 * T, 0.0, None))
    np.fill_diagonal(out, 0.0)
    return 0.5 * (out + out.T)


def checks(C: np.ndarray, S: np.ndarray, Z: np.ndarray, m: np.ndarray, W: np.ndarray, dP: np.ndarray) -> dict:
    """Coerenza della formula form con le GT di E12, su un sottoinsieme di un dominio."""
    iu = np.triu_indices(len(S), 1)
    Cc = C - m[:, None]
    sq = np.sqrt(W)[None, :, None]
    F = (sq * Cc).reshape(len(C), -1)
    G = (F ** 2).sum(1)[:, None] + (F ** 2).sum(1)[None] - 2 * F @ F.T
    d_cc = np.sqrt(np.clip(G, 0, None))[iu]                                   # E12 F_centered
    d_c = np.sqrt(d_cc ** 2 + ((m[:, None] - m[None]) ** 2).sum(-1)[iu])      # E12 F
    Si, Sj = S[iu[0]], S[iu[1]]
    dF = np.sqrt((Si - Sj) ** 2 + Si * Sj * dP[iu] ** 2)
    dP_proc = procrustes_dp(Z, W)[iu]
    dSS = np.sqrt((Si - Sj) ** 2 + Si * Sj * dP_proc ** 2)
    return {"n_subjects": len(S), "n_pairs": len(d_cc),
            "form_vs_F_centered_max_abs_mm": float(np.abs(dF - d_cc).max()),
            "form_vs_F_centered_median_mm": [float(np.median(dF)), float(np.median(d_cc))],
            "spearman_form_vs_F": spearman(dF, d_c),
            "median_F_mm": float(np.median(d_c)),
            "spearman_dP_vs_dP_procrustes": spearman(dP[iu], dP_proc),
            "median_dP": float(np.median(dP[iu])), "median_dP_procrustes": float(np.median(dP_proc)),
            "spearman_form_vs_size_and_shape_procrustes": spearman(dF, dSS),
            "median_size_and_shape_procrustes_mm": float(np.median(dSS)),
            "share_of_size_in_form2": float(np.median((Si - Sj) ** 2 / np.maximum(dF ** 2, 1e-30)))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref-gt", type=Path, required=True, help="GT maxabs del run: nomi, ordine e mediana per kappa")
    ap.add_argument("--split", type=Path, required=True, help="soggetti di training per la taratura di kappa")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--kappa", type=float, default=0.0, help=">0: kappa fisso invece della taratura")
    a = ap.parse_args()
    t0 = time.time()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(a.ref_gt, allow_pickle=True) as z:
        raw_names = [n.decode() if isinstance(n, bytes) else str(n) for n in z["names"]]
        Dref = np.asarray(z["D_orig"], dtype=np.float32)
    sids = [re.search(r"(id\d+)", n, re.IGNORECASE).group(1).lower() for n in raw_names]
    if len(set(sids)) != len(sids):
        raise SystemExit("--ref-gt: id ripetuti")
    sp = Space()
    tf = json.loads(FRAMES.read_text())["domains"]
    dom = np.asarray([domain_of(s) for s in sids])
    n = len(sids)
    C = np.zeros((n, len(sp.W), 3))
    for d in ("bfm", "ict", "gnm"):
        ii = np.flatnonzero(dom == d)
        if len(ii) == 0:
            continue
        P = native_points(sp, d, [sids[i] for i in ii])
        u, R, t = float(tf[d]["u"]), np.asarray(tf[d]["R"]), np.asarray(tf[d]["t"])
        C[ii] = u * P @ R.T + t
        print(f"[targets] {d}: {len(ii)} identita' ({time.time() - t0:.0f}s)", flush=True)
    W = sp.W
    m = np.einsum("v,nvd->nd", W, C)
    Cc = C - m[:, None]
    S = np.sqrt(np.einsum("v,nv->n", W, (Cc ** 2).sum(-1)))
    Z = Cc / S[:, None, None]
    Zf = (np.sqrt(W)[None, :, None] * Z).reshape(n, -1)
    G = Zf @ Zf.T
    dP = np.sqrt(np.clip(2.0 - 2.0 * G, 0.0, None))
    np.fill_diagonal(dP, 0.0)
    dP = 0.5 * (dP + dP.T)

    # kappa: mediana della GT di riferimento / mediana di dP, coppie di training dentro il dominio (insieme)
    train = set(json.loads(a.split.read_text())["train"])
    meds = {}
    ref_all, dp_all = [], []
    for d in ("bfm", "ict", "gnm"):
        ii = np.asarray([i for i in np.flatnonzero(dom == d) if sids[i] in train])
        if len(ii) < 2:
            continue
        iu = np.triu_indices(len(ii), 1)
        r = Dref[np.ix_(ii, ii)][iu].astype(np.float64)
        p = dP[np.ix_(ii, ii)][iu]
        ref_all.append(r)
        dp_all.append(p)
        meds[d] = {"n_train": int(len(ii)), "median_ref": float(np.median(r)), "median_dP": float(np.median(p))}
    kappa = float(a.kappa) if a.kappa > 0 else float(np.median(np.concatenate(ref_all)) / np.median(np.concatenate(dp_all)))
    for d in meds:
        meds[d]["median_kappa_dP"] = kappa * meds[d]["median_dP"]

    rng = np.random.default_rng(SEED)
    chk = {}
    for d in ("bfm", "ict", "gnm"):
        ii = np.flatnonzero(dom == d)
        if len(ii) >= 3:
            ii = np.sort(rng.choice(ii, size=min(N_CHECK, len(ii)), replace=False))
            chk[d] = checks(C[ii], S[ii], Z[ii], m[ii], W, dP[np.ix_(ii, ii)])
    D = (kappa * dP).astype(np.float32)
    np.savez(a.out_dir / "gt_shape.npz", D_orig=D, names=np.asarray(raw_names))
    gt_info = {"scale": "shape_dP", "global_max": float(D.max()), "n_total": n, "kappa": kappa, "dp_per_unit": 1.0 / kappa,
               "definition": "D_orig = kappa * dP, dP = 2 sin(rho/2) = corda fra pre-forme a centroid size unitaria nel "
                             "frame di GT-F di E12 (u_d R_d p + t_d, nessuna rotazione per identita'), pesi d'area di mu; form: "
                             "d_F^2 = (S_i - S_j)^2 + S_i S_j dP^2 (= GT F_centered di E12)",
               "names_source": str(a.ref_gt), "names_identical_to_ref": True, "medians_train_within_domain": meds,
               "kappa_rule": "fisso (--kappa)" if a.kappa > 0 else "mediana(ref) / mediana(dP), coppie di training dentro il dominio",
               "coherence_checks": chk, "frames": str(FRAMES),
               "frames_used": {d: tf[d] for d in ("bfm", "ict", "gnm")}, "seconds": time.time() - t0,
               "built_by": "v3_work/trainer/tools/build_factorized_targets.py"}
    (a.out_dir / "gt_shape.json").write_text(json.dumps(gt_info, indent=1))
    np.savez(a.out_dir / "size.npz", names=np.asarray(sids), cs_mm=S, log_cs_mm=np.log(S), domain=dom)
    size_info = {"definition": "S_i = centroid size pesata (pesi d'area di mu normalizzati) della regione unificata nel "
                               "frame di GT-F di E12 (unita' fisiche), mm; bersaglio della testa: log S_i (stesso per ogni mesh dell'identita', "
                               "crop ed espressioni compresi)",
                 "by_domain": {d: {"n": int((dom == d).sum()), "mean_mm": float(S[dom == d].mean()),
                                   "sd_mm": float(S[dom == d].std(ddof=1)) if (dom == d).sum() > 1 else 0.0,
                                   "cv": float(S[dom == d].std(ddof=1) / S[dom == d].mean()) if (dom == d).sum() > 1 else 0.0,
                                   "log_sd": float(np.log(S[dom == d]).std(ddof=1)) if (dom == d).sum() > 1 else 0.0}
                               for d in ("bfm", "ict", "gnm") if (dom == d).any()},
                 "log_mean_train": float(np.mean([np.log(S[i]) for i in range(n) if sids[i] in train]))}
    (a.out_dir / "size.json").write_text(json.dumps(size_info, indent=1))
    print(json.dumps({"kappa": kappa, "medians": meds, "size": size_info["by_domain"]}, indent=1), flush=True)
    print(json.dumps(chk, indent=1), flush=True)
    print(f"[targets] OK in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
