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
  v4:
  embed       ``conf_embed.py`` (operatori rifatti; v5: embedding su A100, la classe dello sviluppo, tolleranze
              ``embed_*_v5``; l'esito v4 su L40S con 1e-4 e' in ``embed_l40s.json``) contro gli embedding di sviluppo: ``form_hifi``
              (600 mesh, factorized s1234 e s2345, C3M e123) e il pool NON valutato di D3 (200 x 5, i due semi): chiavi,
              Z, d_F cal., d_F non cal., d_P su tutte le coppie; Spearman (punto e IC, nocrop / crop / all_cross, GT FR
              e SR) di quelle distanze sulle righe e repliche di ``conf_eval`` col nuovo embedding contro lo stesso con
              quello di sviluppo.
  desc        ``conf_desc.py`` (calcolo) contro le colonne di sviluppo della matrice: spettrali contro
              ``competitors_hifi3d/desc_spectral.npz``, Gram varifold contro gli shard di ``bp_varifold.py`` + le righe crop
              di ``fm_varifold.py``, ICP + Chamfer di stessa topologia contro ``full_matrix/icp_same``, ArcFace contro
              ``g0/teacher/hifi3d`` (Z, camera, ritaglio); GW e percettivi sul SOTTOCAMPIONE dichiarato
              (``desc.subsample_subjects``: i primi 12 soggetti per id, 72 mesh) contro i plug-in di
              ``competitors_new`` (varianti dichiarate), sulle stesse coppie.
  ulr         (v6) ``conf_desc.py ulr-run`` / ``ulr-merge`` (L40S, la classe della valutazione di sviluppo) sul
              sottocampione dichiarato contro ``competitors_new/ulrssm/faces_xyz/hifi3d.npz``: le 4 letture sulle stesse
              coppie, scarto massimo relativo alla mediana di sviluppo (``ulr_rel``), stesse mesh fallite.
  guard       (v7) i controlli del codice congelato e dei nodi, eseguiti DA un worktree (``--worktree``, creato con
              ``conf_worktree.sh`` al commit da provare, su un nodo diverso da quelli obbligati): ``verify`` passa sul
              worktree pulito al suo commit e fallisce con un altro commit, con un file tracciato modificato e su Ava-256
              senza ``CONF_FROZEN_COMMIT`` (prima di leggere qualunque dato); ``conf_run.py`` esce con 3 se un modulo
              importato differisce dal commit; ``host --role bp``, ``conf_head.py fit`` e ``conf_desc.py arcface``
              falliscono sul nodo sbagliato e ``host`` passa sul nodo giusto (spec copia con quel nodo);
              ``check_host`` rifiuta. v8: ``verify`` fallisce con una copia del clone diffusion-net con ``geometry.py``
              modificato al posto del link del worktree, e ripassa col link originale; ripresa di ``conf_head``
              (``resume``): npz intero con sha registrato -> salta, npz senza json o con sha diverso -> fermo;
              ``conf_desc.fit_hosts`` rifiuta un fit B di un altro nodo. Esito per caso: atteso, codice d'uscita, passa.
  desc-eval   ``conf_desc.py eval`` (hifi3d_full) contro ``full_matrix`` (mascherata = passo ``comune``, confermativa =
              "NICP fallito = distanza massima"; Spearman con IC e righe, delta ``vs_nostro`` e ``scala_allineamento``,
              in tutti i gruppi della lettura: nocrop, crop, all_cross, per perturbazione, stessa topologia) e contro
              ``protocolli/spearman.csv`` (protocolli, anche per perturbazione); plug-in GW e percettivi: rho puntuale su
              tutte le righe dei gruppi di ``competitors_new/spearman.csv`` (le loro righe: nessuna maschera; gli IC
              hanno un altro seme).
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
TOL = {"exact": 0.0, "num": 1e-12, "geom_mm": 1e-4, "rel": 1e-9, "spearman_geom": 1e-4,
       # v4 (embedding e descrittivi), fissate PRIMA di eseguire. embed_abs: Z = [s, u] e distanze d_F (mm), d_P
       # ricalcolate (operatori rifatti su un'altra CPU, embedding su un'altra GPU: L40S contro A100 di sviluppo);
       # desc_rel: autovalori / HKS / WKS e Gram varifold ricalcolati, relativo (fm_varifold.py misura 1,2e-5 fra due
       # device); perc_abs: ArcFace e percettivi ricalcolati (1 - coseno, LPIPS); p_geom: p delle colonne ricalcolate,
       # una replica su 1000; GW e ICP + Chamfer ricalcolati: geom_mm
       "embed_abs": 1e-4, "desc_rel": 1e-4, "perc_abs": 1e-4, "p_geom": 1e-3,
       # v5 (decisione del PI), fissate PRIMA del collaudo su A100: la embed_abs della v4 (1e-4) era sotto il rumore di
       # riproduzione gia' noto in sviluppo (A100 contro A100, operatori ricalcolati: max |dz| 2,8e-4 sul pool di D3,
       # 5,2e-4 nel K7 di D3); l'esito della v4 su L40S resta registrato (collaudo/embed_l40s.json). Z: 1e-3 (~3,5 volte
       # 2,8e-4); d_F (cal. e non cal.): 0,1 mm (1e-3 su s = log S, S ~ 100 mm); d_P: 1e-3 (stessa scala di u);
       # Spearman e IC delle distanze: 1e-3
       "embed_z_v5": 1e-3, "embed_dF_mm_v5": 0.1, "embed_dP_v5": 1e-3, "embed_spearman_v5": 1e-3,
       # v6, fissata PRIMA del collaudo di ULRSSM: scarto massimo / mediana delle distanze di sviluppo confrontate, per
       # ognuna delle 4 letture (per coppia e via template, FR e SR); la mappa punto-a-punto e' un argmax discreto su
       # feature calcolate con operatori rifatti (cache del job), quindi qualche vertice puo' cambiare immagine
       "ulr_rel": 1e-3}
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


# ------------------------------------------------------------------------------------------ v4: embedding

def _keys_Z(path: Path) -> tuple[list, np.ndarray, Path]:
    with np.load(path, allow_pickle=True) as z:
        if "subjects" in z.files:
            keys = list(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))
        else:
            keys = [tuple(Path(str(f)).name[:-4].split("_GTready_", 1)) for f in z["files"]]
        return keys, np.asarray(z["Z"], np.float64), Path(str(z["checkpoint"]))


def embed_check(workers: int) -> None:
    import conf_embed as cem
    import conf_eval as cev
    sys.path.insert(0, str(AAU / "full_matrix"))
    import fm_eval as fm
    s = cc.load(SPEC_FULL)
    arms = cem.arms_of(s)
    rep = {"sets": {}}
    ok = True
    for name, st in s["embed"]["sets"].items():
        for arm in st["arms"]:
            ref = arms[arm]["embeddings"] if st["role"] == "eval" else s["head"]["embeddings"][arm]
            kn, Zn, ckn = _keys_Z(cem.out_path(s, name, arm))
            kr, Zr, ckr = _keys_Z(cc.rp(ref))
            pos = {k: i for i, k in enumerate(kr)}
            want = sorted((x, t) for x, t, _ in cem.items_of(s, name))
            r = {"reference": ref, "meshes": len(kn), "keys_equal_staged": sorted(kn) == want,
                 "keys_in_reference": all(k in pos for k in kn), "checkpoint_equal": ckn.resolve() == ckr.resolve()}
            if r["keys_in_reference"]:
                Zr = Zr[[pos[k] for k in kn]]
                r["Z_max_abs"] = maxabs(Zn, Zr)
                i, j = np.triu_indices(len(kn), 1)
                c = float(arms[arm]["c"])
                dn = fm.fp.distances(Zn, i, j, ckn, (c, c))
                dr = fm.fp.distances(Zr, i, j, ckr, (c, c))
                r["dist_max_abs"] = {k: maxabs(dn[k], dr[k]) for k in ("form_cal", "form", "shape")}
                r["pairs"] = int(len(i))
            r["pass"] = r["keys_equal_staged"] and r["keys_in_reference"] and r["checkpoint_equal"] and \
                r["Z_max_abs"] <= TOL["embed_z_v5"] and r["dist_max_abs"]["shape"] <= TOL["embed_dP_v5"] and \
                max(r["dist_max_abs"]["form_cal"], r["dist_max_abs"]["form"]) <= TOL["embed_dF_mm_v5"]
            ok &= r["pass"]
            rep["sets"][f"{name}|{arm}"] = r
            gj = cc.evidence(s) / "embed" / name / f"{arm}.json"
            if gj.exists():
                r["gpu"] = json.loads(gj.read_text()).get("gpu")
    # Spearman delle distanze col nuovo embedding contro lo stesso con quello di sviluppo (righe e repliche di conf_eval)
    be = fm.be
    R = cev.build_rows(s, be.zes)
    gt = cev.gt_columns(s, R, be.zsum)
    X = {}
    for name, st in s["embed"]["sets"].items():
        if st["role"] != "eval":
            continue
        for arm in st["arms"]:
            c = float(arms[arm]["c"])
            for tag, path in (("new", cem.out_path(s, name, arm)), ("dev", cc.rp(arms[arm]["embeddings"]))):
                Z, ck = cev.embeddings(path, R["idx"])
                for k, d in fm.fp.distances(Z, R["ia"], R["ib"], ck, (c, c)).items():
                    if k in ("form_cal", "form", "shape"):
                        X[f"{tag}:{arm}|{k}"] = d
    mask = np.isfinite(gt["fr"]) & np.isfinite(gt["sr"]) & np.all([np.isfinite(v) for v in X.values()], axis=0)
    n = len(R["subjects"])
    rng = np.random.default_rng(int(s["boot_seed"]))
    counts = [np.ones(n, dtype=np.int64)] + [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(1000)]
    meth = sorted(X)
    V, nr = cev.reading(X, gt, R, mask, meth, counts, workers, fm)
    mi = {m: k for k, m in enumerate(meth)}
    sp = {}
    for g in cev.GROUPS:
        for ti, gtn in enumerate(("fr", "sr")):
            for m in meth:
                if not m.startswith("new:"):
                    continue
                a_, b_ = V[g][:, ti, mi[m]], V[g][:, ti, mi["dev:" + m[4:]]]
                pa, pb = [a_[0], *np.percentile(a_[1:], [2.5, 97.5])], [b_[0], *np.percentile(b_[1:], [2.5, 97.5])]
                sp[f"{g}|{gtn}|{m[4:]}"] = {"rho_new": pa[0], "rho_dev": pb[0], "max_abs": maxabs(pa, pb)}
    rep["spearman"] = sp
    rep["spearman_max_abs"] = max(v["max_abs"] for v in sp.values())
    rep["rows_by_group"] = nr
    rep["pass"] = bool(ok and rep["spearman_max_abs"] <= TOL["embed_spearman_v5"])
    finish("embed", rep)


# ------------------------------------------------------------------------------------------ v4: descrittivi

def desc_check() -> None:
    import conf_desc as cd
    sys.path.insert(0, str(AAU / "full_matrix"))
    import bp_paired as bpp
    import fm_compute as fcm
    s, name = cd.setup(SPEC_FULL)
    W = cd.desc_dir(s)
    rep = {}

    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        if (np.isfinite(a) != np.isfinite(b)).any():
            return float("inf")
        f = np.isfinite(b)
        return float((np.abs(a[f] - b[f]) / np.maximum(np.abs(b[f]), 1e-300)).max())

    with np.load(W / "spectral.npz") as a, np.load(fcm.HIFI_SPECTRAL) as b:     # spettrali
        ka, kb = cd_keys(a), cd_keys(b)
        o = [kb.index(k) for k in ka]
        rep["spectral"] = {"meshes": len(ka), "keys_in_reference": all(k in kb for k in ka),
                           "evals_rel": rel(a["evals"][:, 1:], np.asarray(b["evals"])[o][:, 1:]),
                           "hks_rel": rel(a["hks"], np.asarray(b["hks"])[o]),
                           "wks_rel": rel(a["wks"], np.asarray(b["wks"])[o]),
                           "n_components_equal": bool(np.array_equal(a["n_components"], np.asarray(b["n_components"])[o]))}
    Gn, kn = cd.varifold_gram(s)                                                # varifold
    G5, k5 = bpp.varifold_gram("hifi3d")
    with np.load(EV / "full_matrix" / "varifold" / "hifi3d.npz") as z:
        G6, k6 = np.asarray(z["G"]), fm_keys(z)
    if k6[:500] != k5:
        raise SystemExit("varifold di sviluppo: chiavi senza crop diverse")
    G6[:, :500, :500] = G5
    o = [k6.index(k) for k in kn]
    Gd = G6[:, o][:, :, o]
    rep["varifold"] = {"meshes": len(kn), "G_rel": rel(Gn, Gd),
                       "D_rel": rel(bpp.varifold_distances(Gn) + np.eye(len(kn)), bpp.varifold_distances(Gd) +
                                    np.eye(len(kn)))}
    rep["icp_same"] = {}                                                        # ICP, stessa topologia
    for mode in ("mm", "cs"):
        with np.load(W / f"icp_same_{mode}.npz") as a, np.load(EV / "full_matrix" / "icp_same" / f"hifi3d_{mode}.npz") as b:
            same = list(a["subjects"]) == list(b["subjects"]) and np.array_equal(a["iu"], b["iu"])
            tb = [str(t) for t in b["topologies"]]
            o = [tb.index(str(t)) for t in a["topologies"]]
            rep["icp_same"][mode] = {"keys_equal": bool(same), "pairs": int(a["values_rigid_icp_chamfer"].size),
                                     "max_abs_mm": maxabs(a["values_rigid_icp_chamfer"],
                                                          np.asarray(b["values_rigid_icp_chamfer"])[o]),
                                     "failed": [int(a["n_failed"]), int(b["n_failed"])]}
    ga = cd.arcface_path(s, name)                                               # ArcFace
    gb = EV / "diagnostics" / "g0" / "teacher" / "hifi3d"
    with np.load(ga) as a, np.load(gb / "teacher.npz") as b:
        ka, kb = cd_keys(a), cd_keys(b)
        o = [kb.index(k) for k in ka]
        Za, Zb = np.asarray(a["Z"], np.float64), np.asarray(b["Z"], np.float64)[o]
        rep["arcface"] = {"meshes": len(ka), "keys_equal": ka == kb, "Z_max_abs": maxabs(Za, Zb),
                          "D_max_abs": maxabs(1 - Za @ Za.T, 1 - Zb @ Zb.T),
                          "E_max_abs": maxabs(a["E"], np.asarray(b["E"])[o])}
    ca = json.loads((ga.parent / "camera.json").read_text())
    cb = json.loads((gb / "camera.json").read_text())
    rep["arcface"]["camera_max_abs"] = max(maxabs(ca[k], cb[k]) for k in ("center", "scale"))
    aa = json.loads((ga.parent / "arcface_align.json").read_text())
    ab = json.loads((gb / "arcface_align.json").read_text())
    rep["arcface"]["align_equal"] = aa == ab
    rep["arcface"]["frame_note"] = json.loads((ga.parent / "frame_note.json").read_text())
    rep["gp"] = {}                                                              # GW e percettivi, sottocampione
    gpv = {m: v for m, v in s["columns"]["desc_plugins"]["variants"].items() if m != "ulrssm"}   # ULRSSM: passo ulr
    for m, v in gpv.items():
        with np.load(W / "gp" / m / v / f"{name}.npz") as a, \
                np.load(EV / "competitors_new" / m / v / "hifi3d.npz") as b:
            ka, kb = cd_keys(a), cd_keys(b)
            o = np.asarray([kb.index(k) for k in ka])
            r = {"meshes": len(ka)}
            for d in ("fr", "sr"):
                A, B = np.asarray(a[f"D_{d}"], np.float64), np.asarray(b[f"D_{d}"], np.float64)[np.ix_(o, o)]
                f = np.isfinite(A) & ~np.eye(len(ka), dtype=bool)
                r[f"pairs_{d}"] = int(f.sum() // 2)
                r[f"max_abs_{d}"] = maxabs(A[f], B[f])
            rep["gp"][f"{m}/{v}"] = r
    gp_ok = all(max(r["max_abs_fr"], r["max_abs_sr"]) <= (TOL["geom_mm"] if k.startswith("gw/") else TOL["perc_abs"])
                and r["pairs_fr"] > 0 for k, r in rep["gp"].items())
    rep["pass"] = bool(rep["spectral"]["keys_in_reference"] and rep["spectral"]["n_components_equal"] and
                       max(rep["spectral"][k] for k in ("evals_rel", "hks_rel", "wks_rel")) <= TOL["desc_rel"] and
                       rep["varifold"]["G_rel"] <= TOL["desc_rel"] and
                       all(v["keys_equal"] and v["max_abs_mm"] <= TOL["geom_mm"] for v in rep["icp_same"].values()) and
                       rep["arcface"]["keys_equal"] and rep["arcface"]["D_max_abs"] <= TOL["perc_abs"] and
                       rep["arcface"]["align_equal"] and gp_ok)
    finish("desc", rep)


def ulr_check() -> None:
    import conf_desc as cd
    s, name = cd.setup(SPEC_FULL)
    v = s["desc"]["ulrssm"]["variant"]
    pa = cd.desc_dir(s) / "gp" / "ulrssm" / v
    rep = {"variant": v, "info": json.loads((pa / f"{name}.json").read_text())}
    with np.load(pa / f"{name}.npz") as a, np.load(EV / "competitors_new" / "ulrssm" / v / "hifi3d.npz") as b:
        ka, kb = cd_keys(a), cd_keys(b)
        o = np.asarray([kb.index(k) for k in ka])
        rep["meshes"] = len(ka)
        for d in ("pair_fr", "pair_sr", "tpl_fr", "tpl_sr"):
            A, B = np.asarray(a[f"D_{d}"], np.float64), np.asarray(b[f"D_{d}"], np.float64)[np.ix_(o, o)]
            f = np.isfinite(A) & np.isfinite(B) & ~np.eye(len(ka), dtype=bool)
            med = float(np.median(B[f]))
            rep[d] = {"pairs": int(f.sum() // 2), "nan_mismatch": int((np.isfinite(A) != np.isfinite(B)).sum()),
                      "max_abs": maxabs(A[f], B[f]), "median_dev": med, "max_rel": maxabs(A[f], B[f]) / med}
    rep["gpu"] = rep["info"].get("gpu")
    rep["pass"] = all(rep[d]["pairs"] > 0 and rep[d]["nan_mismatch"] == 0 and rep[d]["max_rel"] <= TOL["ulr_rel"]
                      for d in ("pair_fr", "pair_sr", "tpl_fr", "tpl_sr"))
    finish("ulr", rep)


def cd_keys(z) -> list:
    return list(zip([str(x) for x in z["subjects"]], [str(t) for t in z["topologies"]]))


fm_keys = cd_keys
RECOMPUTED = ("varifold", "shapedna_k50", "hks", "wks", "arcface_normals", "est_cs")
ICP = ("mm_rigid_icp_chamfer", "cs_rigid_icp_chamfer")


def desc_eval_check() -> None:
    import conf_desc as cd
    import conf_eval as cev
    from scipy.stats import rankdata
    sys.path.insert(0, str(AAU / "full_matrix"))
    import fm_eval as fm
    s = cc.load(SPEC_FULL)
    d = cc.evidence(s) / "desc"
    S, D = pd.read_csv(d / "spearman.csv"), pd.read_csv(d / "deltas.csv")
    fmS, fmD = pd.read_csv(EV / "full_matrix" / "spearman.csv"), pd.read_csv(EV / "full_matrix" / "deltas.csv")
    fmS, fmD = fmS[fmS["dominio"] == "hifi3d"], fmD[fmD["dominio"] == "hifi3d"]

    def recomputed(cols, ins) -> pd.Series:
        r = False
        for c in cols:
            r = r | c.isin(RECOMPUTED) | (c.isin(ICP) & (ins == "stessa_topologia"))
        return r

    rep, worst = {}, {"num": 0.0, "geom": 0.0, "p_geom": 0.0}
    for mine, passo in ((cd.MASKED, "comune"), (cd.CONF, SENS)):
        keys = ["insieme_righe", "perturbazione", "gt", "colonna"]
        x = S[(S["passo"] == mine) & S["colonna"].isin(list(fm.METHODS))]
        j = x.merge(fmS[fmS["passo"] == passo], on=keys, suffixes=("_m", "_t"))
        g = recomputed([j["colonna"]], j["insieme_righe"])
        dv = np.max(np.abs(j[["rho_m", "ic_lo_m", "ic_hi_m"]].to_numpy(float) -
                           j[["rho_t", "ic_lo_t", "ic_hi_t"]].to_numpy(float)), axis=1)
        rep[f"spearman {mine} vs full_matrix {passo}"] = {
            "n": int(len(j)), "n_mine": int(len(x)), "n_geom": int(g.sum()),
            "max_abs_num": float(dv[~g].max()) if (~g).any() else 0.0,
            "max_abs_geom": float(dv[g].max()) if g.any() else 0.0,
            "n_righe_max_abs": int(np.abs(j["n_righe_m"] - j["n_righe_t"]).max()) if len(j) else None,
            "groups": sorted({f"{a}|{b}" for a, b in zip(j["insieme_righe"], j["perturbazione"])})}
        keys = ["insieme_righe", "perturbazione", "gt", "tipo", "colonna_a", "colonna_b"]
        x = D[(D["passo"] == mine) & D["tipo"].isin(["vs_nostro", "scala_allineamento"])]
        y = fmD[(fmD["passo"] == passo) & fmD["tipo"].isin(["vs_nostro", "scala_allineamento"])].drop_duplicates(keys)
        j = x.drop_duplicates(keys).merge(y, on=keys, suffixes=("_m", "_t"))
        g = recomputed([j["colonna_a"], j["colonna_b"]], j["insieme_righe"])
        dv = np.max(np.abs(j[["delta_m", "ic_lo_m", "ic_hi_m"]].to_numpy(float) -
                           j[["delta_t", "ic_lo_t", "ic_hi_t"]].to_numpy(float)), axis=1)
        dp = np.abs(j["p_le0_m"].to_numpy(float) - j["p_le0_t"].to_numpy(float))
        rep[f"delta {mine} vs full_matrix {passo}"] = {
            "n": int(len(j)), "n_mine": int(len(x)), "n_geom": int(g.sum()),
            "max_abs_num": float(max(dv[~g].max(), dp[~g].max())) if (~g).any() else 0.0,
            "max_abs_geom": float(dv[g].max()) if g.any() else 0.0,
            "p_max_abs_geom": float(dp[g].max()) if g.any() else 0.0}
    for k, v in rep.items():
        worst["num"] = max(worst["num"], v["max_abs_num"])
        worst["geom"] = max(worst["geom"], v["max_abs_geom"])
        worst["p_geom"] = max(worst["p_geom"], v.get("p_max_abs_geom", 0.0))
    pS = pd.read_csv(EV / "protocolli" / "spearman.csv")                       # protocolli, anche per perturbazione
    pS = pS[(pS["dominio"] == "hifi3d") & (pS["kind"] == "rho")]
    x = S[(S["passo"] == cd.CONF) & S["colonna"].isin(["now_original", "now_metrical", "realy", "stirling"])]
    j = x.merge(pS, on=["insieme_righe", "perturbazione", "gt", "colonna"], suffixes=("_m", "_t"))
    dv = np.max(np.abs(j[["rho_m", "ic_lo_m", "ic_hi_m"]].to_numpy(float) - j[["rho_t", "ic_lo_t", "ic_hi_t"]].to_numpy(
        float)), axis=1)
    rep["protocolli (confermativa) vs protocolli/spearman.csv"] = {
        "n": int(len(j)), "max_abs_num": float(dv.max()) if len(j) else None,
        "n_righe_max_abs": int(np.abs(j["n_righe_m"] - j["n_righe_t"]).max()) if len(j) else None,
        "groups": sorted({f"{a}|{b}" for a, b in zip(j["insieme_righe"], j["perturbazione"])})}
    worst["num"] = max(worst["num"], rep["protocolli (confermativa) vs protocolli/spearman.csv"]["max_abs_num"] or 0.0)
    # plug-in GW e percettivi: rho puntuale su tutte le righe dei gruppi di competitors_new (nessuna maschera)
    be = fm.be
    R = cev.build_rows(s, be.zes)
    gt = cev.gt_columns(s, R, be.zsum)
    gp = s["columns"]["desc_plugins"]
    dirs = [(f"{m}/{v}", cc.rp(gp["root"]) / m / v) for m, v in gp["variants"].items()]
    P, _, _ = fm.plugin_columns(gp["view"], R["idx"].keys, dirs)
    cS = pd.read_csv(EV / "competitors_new" / "spearman.csv")
    cS = cS[(cS["domain"] == "hifi3d") & (cS["kind"] == "rho")]
    crop = (R["ta"] == "crop") | (R["tb"] == "crop")
    gsel = {"senza crop": ~crop, "righe col crop": crop, "all_cross": np.ones(len(crop), bool)}
    gsel.update({f"con {t}": ~crop & ((R["ta"] == t) | (R["tb"] == t)) for t in cd.NOCROP})
    pr, pmax = [], 0.0
    for r in cS.itertuples():
        col = f"plugin:{r.method}/{r.variant}|{r.mode}"
        if col not in P or r.group not in gsel:
            continue
        rows = np.flatnonzero(gsel[r.group])
        x = P[col][R["ia"][rows], R["ib"][rows]]
        rho = fm.fp._pearson(rankdata(x), rankdata(gt[r.gt][rows]))
        pr.append({"group": r.group, "gt": r.gt, "column": col, "rho_mine": rho, "rho_dev": r.point, "n": int(len(rows)),
                   "n_dev": int(r.n_rows)})
        pmax = max(pmax, abs(rho - r.point))
    rep["plugin GW e percettivi: rho puntuale vs competitors_new/spearman.csv"] = {
        "n": len(pr), "max_abs_num": pmax, "rows_equal": all(q["n"] == q["n_dev"] for q in pr)}
    worst["num"] = max(worst["num"], pmax)
    # v6: ULRSSM, rho puntuale delle 4 letture su tutte le righe dei gruppi di ulrssm/<variante>/spearman.csv
    v = gp["variants"].get("ulrssm")
    if v:
        lab = {"ULRSSM per coppia, FR (mm)": "pair_fr", "ULRSSM per coppia, SR (CS_ref)": "pair_sr",
               "ULRSSM via template, FR (GT)": "tpl_fr", "ULRSSM via template, SR (GT)": "tpl_sr"}
        uS = pd.read_csv(EV / "competitors_new" / "ulrssm" / v / "spearman.csv")
        uS = uS[(uS["domain"] == "hifi3d") & (uS["kind"] == "rho") & (uS["arm"] == "baseline")]
        usel = {"nocrop": ~crop, "crop": crop, "all_cross": np.ones(len(crop), bool)}
        ur, umax = [], 0.0
        for r in uS.itertuples():
            col = f"plugin:ulrssm/{v}|{lab[r.distance]}"
            rows = np.flatnonzero(usel[r.group])
            rho = fm.fp._pearson(rankdata(P[col][R["ia"][rows], R["ib"][rows]]), rankdata(gt[r.gt][rows]))
            ur.append({"group": r.group, "gt": r.gt, "column": col, "n": int(len(rows)), "n_dev": int(r.n_rows)})
            umax = max(umax, abs(rho - r.arm_point))
        rep["ULRSSM: rho puntuale vs ulrssm/spearman.csv"] = {"n": len(ur), "max_abs_num": umax,
                                                              "rows_equal": all(q["n"] == q["n_dev"] for q in ur)}
        worst["num"] = max(worst["num"], umax)
    rep["worst"] = worst
    nr = [v.get("n_righe_max_abs") for v in rep.values() if isinstance(v, dict) and v.get("n_righe_max_abs") is not None]
    rep["n_righe_max_abs"] = max(nr) if nr else None
    rep["pass"] = bool(worst["num"] <= TOL["num"] and worst["geom"] <= TOL["spearman_geom"] and
                       worst["p_geom"] <= TOL["p_geom"] and rep["n_righe_max_abs"] == 0 and
                       all(v["n"] > 0 for k, v in rep.items() if isinstance(v, dict) and "n" in v) and
                       rep["plugin GW e percettivi: rho puntuale vs competitors_new/spearman.csv"]["rows_equal"] and
                       rep.get("ULRSSM: rho puntuale vs ulrssm/spearman.csv", {"rows_equal": True})["rows_equal"])
    finish("desc-eval", rep)


def guard_check(worktree: Path) -> None:
    """Casi del passo ``guard`` (docstring del modulo), ognuno un processo nuovo nel worktree."""
    import os
    import subprocess
    import tempfile
    wt = worktree.resolve()
    head = subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    # un commit precedente in cui conf_common.py era diverso (HEAD~1 puo' essere un commit d'altro con gli stessi moduli)
    prev = subprocess.run(["git", "-C", str(wt), "log", "-2", "--format=%H", "--", "aau/ava256/conf_common.py"],
                          capture_output=True, text=True).stdout.split()[1]
    py = [sys.executable]
    spec_full = "aau/ava256/spec/hifi3d_full.json"

    def run(argv, env=None):
        e = {k: v for k, v in os.environ.items() if k not in ("CONF_FROZEN_COMMIT", "CONF_IMPORT_LOG", "CONF_SPEC")}
        e.update(env or {})
        r = subprocess.run(py + argv, cwd=wt, env=e, capture_output=True, text=True)
        return r.returncode, (r.stderr or r.stdout).strip().splitlines()[-1:] if (r.stderr or r.stdout) else []

    cases = []

    def case(name, expect_ok, argv, env=None):
        code, tail = run(argv, env)
        cases.append({"case": name, "expect_ok": expect_ok, "exit": code, "tail": tail,
                      "pass": (code == 0) == expect_ok})

    verify = ["aau/ava256/conf_common.py", "verify", "--spec", spec_full]
    case("verify: worktree pulito al suo commit", True, verify, {"CONF_FROZEN_COMMIT": head})
    case("verify: commit diverso", False, verify, {"CONF_FROZEN_COMMIT": prev})
    case("verify: Ava-256 senza CONF_FROZEN_COMMIT (prima di leggere dati)", False,
         ["aau/ava256/conf_common.py", "verify", "--spec", "aau/ava256/spec/ava256.json"])
    target = wt / "aau" / "ava256" / "conf_eval.py"
    orig = target.read_bytes()
    try:
        target.write_bytes(orig + b"# modifica di prova (collaudo guard)\n")
        case("verify: file tracciato modificato", False, verify, {"CONF_FROZEN_COMMIT": head})
    finally:
        target.write_bytes(orig)
    case("verify: di nuovo pulito", True, verify, {"CONF_FROZEN_COMMIT": head})
    sha = ["aau/ava256/conf_run.py", "aau/ava256/conf_common.py", "sha"]
    case("conf_run: moduli uguali al commit", True, sha, {"CONF_FROZEN_COMMIT": head})
    case("conf_run: moduli diversi dal commit (precedente di conf_common.py)", False, sha, {"CONF_FROZEN_COMMIT": prev})
    import conf_common
    me = conf_common.short_host()
    case("host --role bp sul nodo sbagliato", False, ["aau/ava256/conf_common.py", "host", "--spec", spec_full,
                                                      "--role", "bp"])
    s = json.loads((wt / spec_full).read_text())
    s["nodes"]["bp"] = me
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=str(Path.home())) as fh:
        json.dump(s, fh)
    try:
        case("host --role bp sul nodo giusto (spec copia)", True, ["aau/ava256/conf_common.py", "host", "--spec",
                                                                  fh.name, "--role", "bp"])
    finally:
        os.unlink(fh.name)
    case("conf_head.py fit sul nodo sbagliato", False, ["aau/ava256/conf_head.py", "fit", "--spec", spec_full])
    case("conf_desc.py arcface sul nodo sbagliato", False, ["aau/ava256/conf_desc.py", "arcface", "--spec", spec_full,
                                                            "--stage", "/tmp/wbes_guard_arc"])
    # v8: clone diffusion-net modificato (copia del clone al posto del link del worktree)
    link = wt / "diffusion-net"
    target = link.resolve()
    tmp = Path(tempfile.mkdtemp(prefix="wbes_guard_dn_", dir=str(Path.home() / "scratch_claude")))
    copy = tmp / "diffusion-net"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(target), str(copy)], check=True)
    subprocess.run(["git", "-C", str(copy), "checkout", "-q", conf_common.EXTERNAL_CLONES["diffusion-net"]], check=True)
    geo = copy / "src" / "diffusion_net" / "geometry.py"
    try:
        link.unlink()
        link.symlink_to(copy)
        case("verify: copia del clone diffusion-net pulita", True, verify, {"CONF_FROZEN_COMMIT": head})
        geo.write_bytes(geo.read_bytes() + b"# modifica di prova (collaudo guard)\n")
        case("verify: geometry.py modificato nel clone diffusion-net", False, verify, {"CONF_FROZEN_COMMIT": head})
    finally:
        link.unlink()
        link.symlink_to(target)
        subprocess.run(["rm", "-rf", str(tmp)])
    case("verify: link originale al clone", True, verify, {"CONF_FROZEN_COMMIT": head})
    # v8: ripresa di conf_head (resume) su un npz finto nella cartella temporanea
    pre = ("import sys, json, tempfile; from pathlib import Path; import numpy as np; sys.path.insert(0, 'aau/ava256'); "
           "import conf_head as ch, conf_common as cc\nd = Path(tempfile.mkdtemp()); n = d / 'a.npz'; j = d / 'a.json'\n"
           "np.savez(n, alpha=1.0, W=np.eye(2), r=2, lam=1e-6, alpha0=1.0, nit=1, success=True, loss=0.0, c=0.4, "
           "dpu=0.2, checkpoint='x', host='h')\n")
    case("resume: npz intero e sha registrato -> salta", True,
         ["-c", pre + "j.write_text(json.dumps({'npz_sha256': cc.sha256(n)}))\nsys.exit(0 if ch.resume(n, j, True) else 2)"])
    case("resume: npz senza json -> fermo", False, ["-c", pre + "ch.resume(n, j, True)"])
    case("resume: sha diverso -> fermo", False,
         ["-c", pre + "j.write_text(json.dumps({'npz_sha256': '0' * 64}))\nch.resume(n, j, True)"])
    case("fit B: uscita da un altro nodo rifiutata", False,
         ["-c", "import sys, tempfile; from pathlib import Path; import numpy as np; sys.path.insert(0, 'aau/ava256'); "
                "import conf_desc as cd\nd = Path(tempfile.mkdtemp())\n"
                "[np.savez(d / f, host='altro-nodo.srv') for f in ('fit.npz', 'fit_e1.npz', 'fit_e2_crop.npz')]\n"
                "cd.fit_hosts({'nodes': {'bp': 'i256-a10-08'}}, d)"])
    code = ("import sys; sys.path.insert(0, 'aau/ava256'); import conf_head as ch; "
            "ch.check_host({'host': 'altro-nodo.srv'}, 'i256-a10-10')")
    case("check_host di R6 con la testa di un altro nodo", False, ["-c", code])
    rep_ = {"worktree": str(wt), "head": head, "prev": prev, "node": me, "cases": cases,
            "pass": all(c["pass"] for c in cases)}
    finish("guard", rep_)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("export-template", "landmarks", "det", "eval-full", "eval-det", "blmm", "bp", "head",
                                    "embed", "desc", "desc-eval", "ulr", "guard"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--worktree", type=Path, default=None)
    a = p.parse_args()
    {"export-template": export_template, "landmarks": landmarks, "det": det, "eval-full": eval_full,
     "eval-det": eval_det, "blmm": lambda: blmm_check(a.workers), "bp": bp_check, "head": head_check,
     "embed": lambda: embed_check(a.workers), "desc": desc_check, "desc-eval": desc_eval_check,
     "ulr": ulr_check, "guard": lambda: guard_check(a.worktree)}[a.step]()


if __name__ == "__main__":
    main()
