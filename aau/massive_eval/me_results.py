#!/usr/bin/env python3
"""Modelli massivi: riassunto (``results.md``, ``metrics.csv``, ``controls.json``) dalle uscite delle tre catene, con
la fonte (file:riga) di ogni numero e la contaminazione di ogni cella. Nessun calcolo nuovo.

    aau/run.sh aau/massive_eval/me_results.py           (dopo i passi matrix, famos_id_eval, ava_eval di me.sbatch)

Letture (come la riga "10x identities" del paper): matrice di sviluppo (``matrix/``, ``fm_eval``), insieme ``nocrop``,
passo ``comune``: FR con d_F cal. (c di ``calib/``) e d_F non cal. (c = 1), SR con d_P; riconoscimento su FaMoS TEST
(``famos_id/``, ``ei.py``): rank-1, mAP, AUC (ed EER) per le tre letture; Ava-256 (``ava256/desc``, ``conf_desc.py``),
SOLO DESCRITTIVO, passo ``confermativa (+inf)`` (quello della riga di C3M e123 nella tabella), insieme ``nocrop``.
Delta appaiati sulle stesse repliche: massivo - modello del paper (factorized s1234, stessa lettura) e massivo - ICP +
Chamfer in mm (matrice e Ava-256: ``tipo`` = ``massivo`` di ``me_matrix``; FaMoS: ``deltas.csv`` di ei, positivo = il
massivo meglio, EER compreso). Controlli: le colonne gia' esistenti devono uscire identiche alle fonti del paper
(matrice completa, ``ava256_conf/desc``, ``esterna_famos_id``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
EV = REPO / "aau/runs/evidence"
OUT = EV / "massive_eval"
RUNS = REPO / "aau/runs/massive_release"
MODELS = {"massive_main": ("main", "main_a100", "principale (release, 8 fonti), 330k passi"),
          "massive_open": ("open", "release_open", "aperto (GNM, ICT, FLAME 2023 Open), 230k passi")}
DOMS = ("hifi3d", "facescape", "faceverse", "faceverse_neutral", "famos")
DNAME = {"hifi3d": "HIFI3D", "facescape": "FaceScape", "faceverse": "FaceVerse", "faceverse_neutral": "FV neutral",
         "famos": "FaMoS", "ava256": "Ava-256 (descr.)"}
READS = (("fr", "form_cal", "d_F cal."), ("fr", "form", "d_F c=1"), ("sr", "shape", "d_P"))
PAPER, ICP = "factorized_s1234", "mm_rigid_icp_chamfer"
CONF = "confermativa (+inf)"
# dominio di valutazione -> fonte di training che lo rende in-domain (sources.SOURCE_PRESETS["release"])
IN_DOMAIN = {"hifi3d": "hifi3d", "facescape": "facescape", "faceverse": "faceverse", "faceverse_neutral": "faceverse",
             "famos": "famos"}


def rel(p: Path) -> str:
    return str(Path(p).relative_to(REPO))


def src(path: Path, i: int) -> str:
    return f"{rel(path)}:{int(i) + 2}"                    # riga del file (intestazione = 1)


def training(model: str) -> dict:
    """Fonti del training e viste per fonte (``node0/producers.json``), riga del producer con fonti ed esclusioni."""
    run = RUNS / MODELS[model][1]
    pj = json.loads((run / "node0" / "producers.json").read_text())
    log = (run / "node0" / "producers.log").read_text().splitlines()
    k = next(i for i, x in enumerate(log) if x.startswith("[producer]") and "fonti" in x)
    return {"views_by_domain": pj["views_by_domain"], "views": pj["views"],
            "fonte_viste": f"{rel(run / 'node0' / 'producers.json')} (views_by_domain)",
            "riga_producer": log[k], "fonte_producer": f"{rel(run / 'node0' / 'producers.log')}:{k + 1}"}


def contamination(model: str, tr: dict) -> dict:
    """{dominio: (etichetta breve, motivo)}: IN-DOMAIN se il generatore / dataset del dominio e' fra le fonti."""
    split = json.loads((REPO / "aau/famos/split.json").read_text())
    seen = set(tr["views_by_domain"])
    out = {}
    for d, s in IN_DOMAIN.items():
        if s in seen and d == "famos":
            ov = sorted(set(split["train"]) & set(split["test"]))
            out[d] = ("IN-DOMAIN", f"FaMoS TRAIN nel training ({tr['views_by_domain']['famos']} viste; 74 persone, 6 "
                      f"held-out); stesso sistema di cattura e registrazione. Persone di TEST escluse (FamosSource; "
                      f"persone TRAIN e TEST in comune: {len(ov)}, aau/famos/split.json): nessuna identita' condivisa")
        elif s in seen:
            out[d] = ("IN-DOMAIN", f"generatore '{s}' nel training ({tr['views_by_domain'][s]} viste): stesso 3DMM delle "
                      "identita' di valutazione; identita' esatte non condivise per costruzione (coefficienti continui, "
                      "semi del training indipendenti; non verificato coefficiente per coefficiente)")
        elif d == "famos" and "flame2023" in seen:
            out[d] = ("vicino", "zero-shot per persone e dataset (nessuna vista FaMoS), ma FLAME 2023 Open nel training "
                      f"({tr['views_by_domain']['flame2023']} viste): FaMoS e' registrato su FLAME (topologia e GT)")
        else:
            out[d] = ("zero-shot", f"generatore del dominio non nel training (fonti: {', '.join(sorted(seen))})")
    out["ava256"] = ("zero-shot", "nessuna sorgente Ava-256 (esclusa sempre, PLAN_MASSIVE sez. 23.1)")
    return out


def pick(df: pd.DataFrame, **kw) -> pd.DataFrame:
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def one(df: pd.DataFrame, **kw):
    x = pick(df, **kw)
    if len(x) != 1:
        raise SystemExit(f"{len(x)} righe per {kw}")
    return x.index[0], x.iloc[0]


def control_rows(new: pd.DataFrame, old: pd.DataFrame, key: list, vals: tuple, skip: str) -> dict:
    """Righe di ``new`` con le stesse chiavi in ``old`` (colonne non dei modelli massivi): scarto massimo."""
    new = new[~new["colonna"].astype(str).str.startswith(skip)]
    old = old[old["dominio"].isin(set(new["dominio"]))]
    m = new.merge(old, on=key, suffixes=("", "_ref"))
    out = {"rows_matched": int(len(m)), "rows_ref": int(len(old)), "rows_new_without_ref": int(len(new) - len(m)),
           "n_righe_diverse": int((m["n_righe"] != m["n_righe_ref"]).sum())}
    for v in vals:
        out[f"max_abs_diff_{v}"] = float(np.nanmax(np.abs(m[v].to_numpy(float) - m[f"{v}_ref"].to_numpy(float))))
    return out


def main() -> None:
    rows, ctrl = [], {"calibrazione": {}, "training": {}}
    tr = {m: training(m) for m in MODELS}
    con = {m: contamination(m, tr[m]) for m in MODELS}
    cal = {m: json.loads((OUT / "calib" / f"{k}.json").read_text()) for m, (k, _, _) in MODELS.items()}
    for m, (k, _, _) in MODELS.items():
        p = OUT / "calib_paper" / k / "calib.json"
        ctrl["calibrazione"][m] = {"c": cal[m]["c_median"], "c_ls": cal[m]["c_ls"],
                                   "c_con_espressioni": cal[m]["con_espressioni"]["c_median"],
                                   "c_per_dominio": {d: cal[m][f"c_median_{d}"] for d in cal[m]["generators"]},
                                   "fonte": rel(OUT / "calib" / f"{k}.json")}
        if p.exists():
            cp = json.loads(p.read_text())
            ctrl["calibrazione"][m]["catena_paper_heldout_C3M"] = {
                "c": cp["c_median"], "c_ls": cp["c_ls"], **{f"c_{d}": cp[f"c_median_{d}"] for d in ("bfm", "ict", "gnm")},
                "fonte": rel(p)}
        ctrl["training"][m] = {kk: v for kk, v in tr[m].items()}

    def add(model, analisi, dom, gt, metrica, lettura, tipo, r, i, path, extra=None):
        rows.append({"modello": model, "analisi": analisi, "dominio": dom, "gt": gt, "metrica": metrica,
                     "lettura": lettura, "tipo": tipo, "valore": float(r[extra or ("rho" if tipo == "valore" else "delta")]),
                     "ic_lo": float(r["ic_lo"]), "ic_hi": float(r["ic_hi"]),
                     "p_le0": float(r["p_le0"]) if "p_le0" in r and tipo != "valore" else np.nan,
                     "n_righe": int(r["n_righe"]) if "n_righe" in r else np.nan,
                     "n_soggetti": int(r["n_soggetti"]) if "n_soggetti" in r else np.nan,
                     "seme": int(r["seme"]) if "seme" in r else np.nan,
                     "contaminazione": con[model][dom][0], "fonte": src(path, i)})

    # --- matrice di sviluppo
    ps, pd_ = OUT / "matrix" / "spearman.csv", OUT / "matrix" / "deltas.csv"
    S = pd.read_csv(ps, dtype={"perturbazione": str})
    D = pd.read_csv(pd_, dtype={"perturbazione": str})
    base = dict(insieme_righe="nocrop", perturbazione="-", passo="comune")
    for m in MODELS:
        for dom in DOMS:
            for gt, rd, lab in READS:
                i, r = one(S, dominio=dom, gt=gt, colonna=f"{m}|{rd}", **base)
                add(m, "matrice_dev", dom, gt, "spearman", lab, "valore", r, i, ps)
                for b, t in ((f"{PAPER}|{rd}", "delta_vs_paper_s1234"), (ICP, "delta_vs_icp_mm")):
                    i, r = one(D, dominio=dom, gt=gt, tipo="massivo", colonna_a=f"{m}|{rd}", colonna_b=b, **base)
                    add(m, "matrice_dev", dom, gt, "spearman", lab, t, r, i, pd_)
    S0 = pd.read_csv(EV / "full_matrix" / "spearman.csv", dtype={"perturbazione": str})
    ctrl["matrice_vs_full_matrix"] = control_rows(
        S, S0, ["dominio", "gt", "insieme_righe", "perturbazione", "passo", "colonna"], ("rho", "ic_lo", "ic_hi"), "massive_")
    # --- Ava-256, descrittivo
    pa, pad = OUT / "ava256" / "desc" / "spearman.csv", OUT / "ava256" / "desc" / "deltas.csv"
    A = pd.read_csv(pa, dtype={"perturbazione": str})
    AD = pd.read_csv(pad, dtype={"perturbazione": str})
    abase = dict(insieme_righe="nocrop", perturbazione="-", passo=CONF)
    for m in MODELS:
        for gt, rd, lab in READS:
            i, r = one(A, dominio="ava256", gt=gt, colonna=f"{m}|{rd}", **abase)
            add(m, "ava256_descrittivo", "ava256", gt, "spearman", lab, "valore", r, i, pa)
            for b, t in ((f"{PAPER}|{rd}", "delta_vs_paper_s1234"), (ICP, "delta_vs_icp_mm")):
                i, r = one(AD, dominio="ava256", gt=gt, tipo="massivo", colonna_a=f"{m}|{rd}", colonna_b=b, **abase)
                add(m, "ava256_descrittivo", "ava256", gt, "spearman", lab, t, r, i, pad)
    A0 = pd.read_csv(EV / "ava256_conf" / "desc" / "spearman.csv", dtype={"perturbazione": str})
    ctrl["ava256_vs_ava256_conf_desc"] = control_rows(
        A, A0, ["dominio", "gt", "insieme_righe", "perturbazione", "passo", "colonna"], ("rho", "ic_lo", "ic_hi"), "massive_")
    # --- FaMoS TEST, riconoscimento
    pf, pfd = OUT / "famos_id" / "metrics.csv", OUT / "famos_id" / "deltas.csv"
    F = pd.read_csv(pf)
    FD = pd.read_csv(pfd)
    for m in MODELS:
        for gt, rd, lab in READS:
            for met in ("rank1", "mAP", "AUC", "EER"):
                i, r = one(F, colonna=f"{m}|{rd}", metrica=met)
                add(m, "famos_id", "famos", "-", met, lab, "valore", r, i, pf, extra="valore")
                rows[-1].update(n_righe=int(r["n_coppie"]), n_soggetti=int(r["n_soggetti"]), seme=int(r["seme"]))
                for b, t in ((f"{PAPER}|{rd}", "delta_vs_paper_s1234"), ("icp|mm", "delta_vs_icp_mm")):
                    i, r = one(FD, nostro=f"{m}|{rd}", concorrente=b, metrica=met)
                    add(m, "famos_id", "famos", "-", met, lab, t, r, i, pfd)
                    rows[-1].update(seme=int(r["seme"]))
    F0 = pd.read_csv(EV / "esterna_famos_id" / "metrics.csv")
    mm = F[F["colonna"].str.startswith(PAPER) | ~F["nostro"].astype(bool)].merge(
        F0, on=["colonna", "metrica"], suffixes=("", "_ref"))
    ctrl["famos_id_vs_esterna_famos_id"] = {"rows_matched": int(len(mm)), "max_abs_diff": float(np.nanmax(np.abs(
        mm[["valore", "ic_lo", "ic_hi"]].to_numpy(float) - mm[["valore_ref", "ic_lo_ref", "ic_hi_ref"]].to_numpy(float))))}
    ctrl["famos_id_embed"] = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "famos_id" / "controls").glob("embed_*.json"))}
    # --- uscite
    with open(OUT / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ctrl["contaminazione"] = {m: {d: {"etichetta": a, "motivo": b} for d, (a, b) in c.items()} for m, c in con.items()}
    (OUT / "controls.json").write_text(json.dumps(ctrl, indent=1, default=str) + "\n")
    write_md(pd.DataFrame(rows), ctrl, cal, con, tr)


def f3(v, lo, hi, sign=False) -> str:
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def write_md(R: pd.DataFrame, ctrl: dict, cal: dict, con: dict, tr: dict) -> None:
    L = ["# Modelli massivi: valutazione con la pipeline della riga \"10x identities\"", "",
         "Generato da `aau/massive_eval/me_results.py` (fonti: `metrics.csv`, colonna `fonte` = file:riga di ogni "
         "numero). Checkpoint finali EMA: principale `main_a100` epoch330 (330k passi, job 1075583 + 1078908), aperto "
         "`release_open` epoch230 (230k passi, job 1079557). Nessun numero di questi modelli e' nelle affermazioni del "
         "paper (PLAN_MASSIVE sez. 23); Ava-256 SOLO DESCRITTIVO, fuori dalla confermativa (prova congelata non toccata).",
         "", "## Calibrazione c (fissata e committata prima di ogni valutazione)", "",
         "`fact_calib.calib_one` sull'held-out sintetico dei generatori di training del modello (`me_calib.py`): "
         "coppie stesso dominio, identita' ed etichette diverse, senza crop, viste neutre; c = mediana d_P GT / "
         "mediana d_P modello.", "", "| modello | c | c_LS | c con espressioni | c per dominio | catena del paper "
         "(held-out C3M bfm/ict/gnm), controllo | fonte |", "|---|---|---|---|---|---|---|"]
    for m, c in ctrl["calibrazione"].items():
        cp = c.get("catena_paper_heldout_C3M")
        cps = (f"{cp['c']:.4f} (bfm {cp['c_bfm']:.3f}, ict {cp['c_ict']:.3f}, gnm {cp['c_gnm']:.3f})" if cp else "-")
        L.append(f"| {m} | {c['c']:.4f} | {c['c_ls']:.4f} | {c['c_con_espressioni']:.4f} | "
                 + ", ".join(f"{d} {v:.3f}" for d, v in c["c_per_dominio"].items()) + f" | {cps} | `{c['fonte']}` |")
    L += ["", "Riferimento: modello del paper s1234 c = 0.4053, C3M e123 c = 0.2573 (`factorized_calibration.csv`).", "",
          "## Contaminazione (dalla configurazione del training)", ""]
    for m in MODELS:
        L.append(f"- **{m}** ({MODELS[m][2]}): `{tr[m]['riga_producer']}` (`{tr[m]['fonte_producer']}`); viste per "
                 f"fonte in `{tr[m]['fonte_viste']}`.")
        for d in DOMS + ("ava256",):
            L.append(f"  - {DNAME[d]}: **{con[m][d][0]}**. {con[m][d][1]}.")
    L += ["- Per entrambi (e per i modelli del paper): HIFI3D, FaceVerse e dev FaceScape sono stati usati nella scelta "
          "della ricetta (PLAN_MASSIVE sez. 22.6).", "",
          "## Matrice di sviluppo (nocrop, passo comune; IC 95%, 1000 repliche per soggetto, semi della matrice)", "",
          "Celle marcate: `[IN]` = in-domain, `[~]` = dominio vicino; senza marca = zero-shot.", ""]
    for m in MODELS:
        L += [f"### {m}", "", "| lettura | GT | " + " | ".join(DNAME[d] for d in DOMS) + " | " + DNAME["ava256"] + " |",
              "|---|---|" + "---|" * (len(DOMS) + 1)]
        for gt, rd, lab in READS:
            for t, tl in (("valore", "rho"), ("delta_vs_paper_s1234", "delta vs paper s1234"),
                          ("delta_vs_icp_mm", "delta vs ICP mm")):
                cells = []
                for d in DOMS + ("ava256",):
                    an = "ava256_descrittivo" if d == "ava256" else "matrice_dev"
                    x = R[(R.modello == m) & (R.analisi == an) & (R.dominio == d) & (R.lettura == lab) & (R.tipo == t)]
                    mk = {"IN-DOMAIN": " [IN]", "vicino": " [~]"}.get(con[m][d][0], "")
                    cells.append(f3(x.valore.iloc[0], x.ic_lo.iloc[0], x.ic_hi.iloc[0], t != "valore") + mk if len(x) else "-")
                L.append(f"| {lab}, {tl} | {gt.upper()} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## FaMoS TEST, riconoscimento (150 scansioni grezze, 15 persone; `ei.py`)", "",
          "Delta: positivo = massivo meglio (EER: concorrente - massivo).", ""]
    for m in MODELS:
        L += [f"### {m} ({con[m]['famos'][0]})", "", "| lettura | | rank-1 | mAP | AUC | EER |", "|---|---|---|---|---|---|"]
        for _, rd, lab in READS:
            for t, tl in (("valore", "valore"), ("delta_vs_paper_s1234", "delta vs paper s1234"),
                          ("delta_vs_icp_mm", "delta vs ICP mm")):
                cells = []
                for met in ("rank1", "mAP", "AUC", "EER"):
                    x = R[(R.modello == m) & (R.analisi == "famos_id") & (R.lettura == lab) & (R.tipo == t) & (R.metrica == met)]
                    cells.append(f3(x.valore.iloc[0], x.ic_lo.iloc[0], x.ic_hi.iloc[0], t != "valore"))
                L.append(f"| {lab} | {tl} | " + " | ".join(cells) + " |")
        L.append("")
    c1, c2, c3 = ctrl["matrice_vs_full_matrix"], ctrl["ava256_vs_ava256_conf_desc"], ctrl["famos_id_vs_esterna_famos_id"]
    L += ["## Controlli di riproduzione", "",
          f"- Matrice: {c1['rows_matched']} righe di `matrix/spearman.csv` (colonne non massive) con le stesse chiavi di "
          f"`full_matrix/spearman.csv`; n_righe diversi {c1['n_righe_diverse']}; scarto massimo punto "
          f"{c1['max_abs_diff_rho']:.2e}, IC {max(c1['max_abs_diff_ic_lo'], c1['max_abs_diff_ic_hi']):.2e}; righe senza "
          f"riferimento {c1['rows_new_without_ref']} (colonne nuove di `celle_mancanti`).",
          f"- Ava-256: {c2['rows_matched']} righe contro `ava256_conf/desc/spearman.csv`; n_righe diversi "
          f"{c2['n_righe_diverse']}; scarto massimo punto {c2['max_abs_diff_rho']:.2e}, IC "
          f"{max(c2['max_abs_diff_ic_lo'], c2['max_abs_diff_ic_hi']):.2e}.",
          f"- FaMoS riconoscimento: {c3['rows_matched']} righe (paper s1234 e concorrenti) contro "
          f"`esterna_famos_id/metrics.csv`, scarto massimo {c3['max_abs_diff']:.2e}; embedding contro i latenti della "
          "matrice: `" + json.dumps({k: v.get("ok") for k, v in ctrl["famos_id_embed"].items()}) + "`.", "",
          "Tutto in `controls.json`.", ""]
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
