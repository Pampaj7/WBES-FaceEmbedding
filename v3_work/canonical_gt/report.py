#!/usr/bin/env python3
"""E12: ``aau/runs/evidence/e12/summary.md`` dai csv di gt.py e methods.py.

    v3_work/unified_gt/run.sh v3_work/canonical_gt/report.py          (run.sbatch, passo ``report``)

Se c'e' ``conclusions.md`` (scritto a mano, sui numeri sotto) va in testa. Solo tabelle: nessun numero
ricalcolato qui.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import cgt

E = cgt.EVID_DIR
GT_LABEL = {"maxabs": "maxabs (legacy)", "unified": "unificata (Procrustes completo)", "F": "F (forma, mm)",
            "F_centered": "F centrata", "F_rig_ls": "F + rigida LS", "F_rig_rob": "F + rigida robusta",
            "S": "S (senza taglia)", "EDM": "EDM", "EDM_s": "EDM senza scala", "F_json": "canonica col json",
            "F_pure": "F pura (frame di cattura)", "raw": "raw (build_zs_gt)", "size_only": "solo taglia",
            "height_only": "solo altezza"}
MAIN_GTS = ("maxabs", "unified", "F", "S", "EDM", "EDM_s")
VAR_GTS = ("F_centered", "F_rig_ls", "F_rig_rob")
METHOD_LABEL = {
    "arcface_shaded_3v": "ArcFace ombreggiato (3 viste)", "arcface_normals_3v": "ArcFace normal map (3 viste)",
    "scale_e036": "BFM+ICT+GNM e036", "scale_e072": "BFM+ICT+GNM e072", "scale_e108": "BFM+ICT+GNM e108",
    "joint": "BFM+ICT congiunto 1019532", "joint@bfm": "congiunto 1019532 (conv. BFM)",
    "joint@ict": "congiunto 1019532 (conv. ICT)", "chamfer_eval": "Chamfer eval",
    "fb_chamfer": "Chamfer faceBench 4096 pt", "fb_rigid_icp_chamfer": "ICP rigido + Chamfer",
    "fb_nicp_p2tri": "ICP + NICP P2Tri", "nicp_template": "NICP su template", "uni3d": "Uni3D-g",
    "openshape": "OpenShape PointBERT", "shapedna_k50": "ShapeDNA k=50", "shapedna_k100": "ShapeDNA k=100",
    "chamfer_full": "Chamfer intera (4096 pt)", "chamfer_stable": "Chamfer regione stabile"}
# righe solo nei csv (ablazioni dei competitori)
CSV_ONLY = ("uni3d_rx90", "openshape_rx90", "openshape_yzswap", "shapedna_k50_l1", "shapedna_k100_l1", "hks", "wks")
GROUPS = {"hifi3d": ("nocrop_cross", "all_cross", "subject_pair_mean"),
          "faceverse": ("mesh_pair_nocrop", "subject_pair_mean_nocrop"),
          "facescape": ("nocrop_cross", "all_cross", "subject_pair_mean")}
DOM_LABEL = {"hifi3d": "HIFI3D", "faceverse": "FaceVerse con espressioni (secondario)", "facescape": "FaceScape dev"}


def ci(r, signed=False, digits=3) -> str:
    f = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return f"{f.format(r['point'])} [{f.format(r['ci_low'])}, {f.format(r['ci_high'])}]"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"] + \
           ["| " + " | ".join(r) + " |" for r in rows]


def section_frames(md: list) -> None:
    un = pd.read_csv(E / "units.csv")
    tr = pd.read_csv(E / "transforms_check.csv")
    g = json.loads((E / "gt.json").read_text())
    md += ["## 1. Frame di GT-F e unita' (verifiche a e b)", "",
           "`canonical_transforms.json` ricalcolato dal template medio di ogni dominio (stessa procedura di "
           "`unified.py`): la trasformazione dipende solo dal dominio.", ""]
    md += table(["dominio", "unita' dichiarata", "s json", "s ricalcolata (diff. rel.)", "max |diff R|", "max |diff t| mm",
                 "rotazione (gradi)"],
                [[r.domain, r.units_declared, f"{r.s_json:.6g}", f"{r.s_recomputed:.6g} ({r.rel_diff_s:.1e})",
                  f"{r.max_abs_diff_R:.1e}", f"{r.max_abs_diff_t_mm:.1e}", f"{r.rotation_deg:.1f}"] for r in tr.itertuples()])
    e = g["eyes"]
    md += ["", f"IPD: centri degli occhi = media dei 6 landmark iBUG del contorno, portati per vertice piu' vicino "
               f"della regione (distanza del sostituto sulla media FLAME: mediana "
               f"{np.median(e['proxy_distance_mm']):.2f} mm, massimo {max(e['proxy_distance_mm']):.2f} mm). IPD della "
               f"media FLAME: {e['flame_ipd_exact_mm']:.1f} mm coi landmark esatti, {e['flame_ipd_proxy_mm']:.1f} mm coi "
               f"sostituti. La scala del json e' `s = u * k`: `k` = taglia della media FLAME / taglia della media del dominio.", ""]
    md += table(["dominio", "IPD media (unita' native)", "mm/unita' da 63 mm", "potenza di 10 piu' vicina (rapporto)",
                 "u usata (mm/unita')", "fonte", "IPD media in mm con u", "k = s json / u", "rigida media -> media FLAME, RMS mm"],
                [[r.domain, f"{r.ipd_template_native:.5g}", f"{r.mm_per_unit_from_ipd63:.4g}",
                  f"{r.nearest_power_of_ten} ({r.ratio_ipd_estimate_over_power:.3f})", f"{r.u_used_mm:.6g}", r.unit_source,
                  f"{r.ipd_template_mm_with_u:.1f}" + (" (oltre il 15%)" if r.ipd_disagrees_15pct else ""),
                  f"{r.k_json_over_u:.3f}", f"{r.rigid_mean_to_flame_mean_rms_mm:.2f}"] for r in un.itertuples()])
    md.append("")


def section_size(md: list) -> None:
    s = pd.read_csv(E / "size_cv.csv")
    g = json.loads((E / "gt.json").read_text())
    md += ["## 2. Dimensione del volto dentro ogni dominio (verifica c)", "",
           "Nel frame di GT-F (mm fisici; sulle catture reali dopo la rigida robusta). Centroid size pesata per area "
           "della regione, IPD, altezza (max y - min y della regione). CV = dev. std. / media; atteso 4-8%.", ""]
    md += table(["set", "n", "centroid size mm (CV)", "IPD mm (CV)", "altezza mm (CV)", "corr(CS, IPD)"],
                [[r.set, str(r.n), f"{r.centroid_size_mean_mm:.1f} ({100 * r.centroid_size_cv:.1f}%)",
                  f"{r.ipd_mean_mm:.1f} ({100 * r.ipd_cv:.1f}%)", f"{r.height_mean_mm:.1f} ({100 * r.height_cv:.1f}%)",
                  f"{r.corr_cs_ipd:.2f}"] for r in s.itertuples()])
    md += ["", "Rigide per identita' (effetto Pinocchio) e posa, set di valutazione: mediana [p95].", ""]
    rows = []
    for name in ("hifi3d", "faceverse", "facescape", "famos"):
        x = g["sets"][name]
        rows.append([name, f"{x['rigid_ls']['angle_deg']['median']:.2f} [{x['rigid_ls']['angle_deg']['p95']:.2f}]",
                     f"{x['rigid_ls']['offset_mm']['median']:.2f} [{x['rigid_ls']['offset_mm']['p95']:.2f}]",
                     f"{x['rigid_rob']['angle_ls_vs_rob_deg']['median']:.2f} [{x['rigid_rob']['angle_ls_vs_rob_deg']['p95']:.2f}]",
                     f"{x['rigid_rob']['downweighted_area']['median']:.3f}",
                     f"{x['rigid_rob']['converged_fraction']:.3f}",
                     f"{x['pose_spread']['angle_from_mean_rotation_deg']['median']:.2f} [{x['pose_spread']['angle_from_mean_rotation_deg']['p95']:.2f}]",
                     f"{x['pose_spread']['centroid_from_median_mm']['median']:.2f} [{x['pose_spread']['centroid_from_median_mm']['p95']:.2f}]"])
    md += table(["set", "rigida LS: angolo (gradi)", "rigida LS: centroide da mu (mm)", "LS contro robusta (gradi)",
                 "area con peso ridotto (robusta)", "robusta convergente", "dispersione della rotazione (gradi)",
                 "dispersione del centroide (mm)"], rows)
    md += ["", "Mediana delle distanze fra coppie del pool (unita' di ogni GT):", ""]
    gts = ["maxabs", "unified", "F", "F_centered", "F_rig_ls", "F_rig_rob", "S", "EDM", "EDM_s", "F_pure"]
    md += table(["set"] + [GT_LABEL[k] for k in gts],
                [[n] + [f"{g['sets'][n]['median_pair'][k]:.3g}" if k in g["sets"][n]["median_pair"] else "-" for k in gts]
                 for n in ("hifi3d", "faceverse", "facescape", "famos")])
    md += ["", "Controlli: " + "; ".join(f"{k} = {v:.1e}" for k, v in g["checks"].items()) + ".", ""]


def section_corr(md: list) -> None:
    c = pd.read_csv(E / "gt_corr.csv")
    md += ["## 3. Spearman fra GT", "",
           "100 soggetti valutati (4.950 coppie; FaMoS: 15 di TEST, 105 coppie), IC 95% bootstrap per soggetto "
           "(1000 repliche, seme 1234); fra parentesi quadre l'IC, dopo la barra il punto sul pool intero (500; "
           "FaMoS 95).", ""]
    cols = ("maxabs", "unified", "F", "S", "EDM", "EDM_s", "size_only", "height_only")
    for s in ("hifi3d", "faceverse", "facescape", "famos_test"):
        x = c[c["set"] == s]
        names = sorted(set(x["gt_a"]) | set(x["gt_b"]), key=lambda k: list(GT_LABEL).index(k))

        def cell(a, b):
            if a == b:
                return "1"
            r = x[((x["gt_a"] == a) & (x["gt_b"] == b)) | ((x["gt_a"] == b) & (x["gt_b"] == a))]
            if not len(r):
                return "-"
            r = r.iloc[0]
            return f"{r['point']:.3f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] / {r['pool_point']:.3f}"
        md += [f"**{s}**", ""]
        md += table(["GT"] + [GT_LABEL[k] for k in cols if k in names],
                    [[GT_LABEL[a]] + [cell(a, b) for b in cols if b in names] for a in names])
        md.append("")


def section_ident(md: list) -> None:
    i = pd.read_csv(E / "ident.csv")
    p = pd.read_csv(E / "ident_paired.csv")
    d = json.loads((E / "ident.json").read_text())
    md += ["## 4. Arbitro: identificabilita' su catture reali ripetute", "",
           "AUC di verifica (genuine = stessa persona, sequenze diverse), rank-1 (ogni cattura contro tutte le altre), "
           "rapporto intra / inter (mediane; piu' basso = meglio). IC 95% bootstrap per persona (1000 repliche, seme "
           "1234), le stesse repliche per tutte le GT.", ""]
    for s in ("famos95_first", "famos15_test_first", "famos95_kept", "multiface_take"):
        x = i[i["set"] == s]
        if not len(x):
            continue
        r0 = x.iloc[0]
        md += [f"**{s}**: {r0['n_persons']} persone, {r0['n_captures']} catture, {r0['n_genuine']} coppie genuine, "
               f"{r0['n_impostor']} impostore.", ""]
        gts = list(dict.fromkeys(x["gt"]))
        md += table(["GT", "AUC", "rank-1", "intra / inter"],
                    [[GT_LABEL[g]] + [ci(x[(x["gt"] == g) & (x["metric"] == m)].iloc[0]) for m in ("auc", "rank1", "ratio")]
                     for g in gts])
        if s in d["decision"]:
            dec = d["decision"][s]
            best = dec["auc_best"]
            md += ["", f"Differenze appaiate contro la migliore per AUC ({GT_LABEL[best]}):", ""]
            rows = []
            for g in gts:
                if g == best:
                    continue
                cells = []
                for m in ("auc", "ratio"):
                    r = p[(p["set"] == s) & (p["a"] == best) & (p["b"] == g) & (p["metric"] == m)].iloc[0]
                    cells.append(f"{r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
                rows.append([GT_LABEL[g]] + cells)
            md += table([f"{GT_LABEL[best]} - GT", "delta AUC", "delta intra / inter"], rows)
            md += ["", f"Regola: {json.dumps(dec, ensure_ascii=False)}", ""]
        md.append("")


def section_methods(md: list) -> None:
    R = pd.read_csv(E / "methods_spearman.csv")
    P = pd.read_csv(E / "methods_paired.csv")
    md += ["## 5. Metodi esistenti con tutte le GT", "",
           "Stesse righe, stessi soggetti, stesse repliche dei summary esistenti (un seme per dominio e gruppo, quello "
           "di e108 - Chamfer eval); cambia solo la GT. IC 95% bootstrap per soggetto, 1000 repliche. Righe solo nei "
           f"csv: {', '.join(CSV_ONLY)}.", ""]
    for dom, groups in GROUPS.items():
        for grp in groups:
            x = R[(R["domain"] == dom) & (R["group"] == grp)]
            if not len(x):
                continue
            methods = [m for m in dict.fromkeys(x["method"]) if m not in CSV_ONLY]
            methods.sort(key=lambda m: -float(x[(x["method"] == m) & (x["gt"] == "F")]["point"].iloc[0]))
            md += [f"### {DOM_LABEL[dom]}, {grp}", "", "Spearman con la GT (ordinati per GT-F):", ""]
            md += table(["metodo"] + [GT_LABEL[g] for g in MAIN_GTS],
                        [[METHOD_LABEL.get(m, m)] + [ci(x[(x["method"] == m) & (x["gt"] == g)].iloc[0]) for g in MAIN_GTS]
                         for m in methods])
            md += ["", "Varianti di F:", ""]
            md += table(["metodo"] + [GT_LABEL[g] for g in VAR_GTS],
                        [[METHOD_LABEL.get(m, m)] + [ci(x[(x["method"] == m) & (x["gt"] == g)].iloc[0]) for g in VAR_GTS]
                         for m in methods])
            for ref in ("scale_e108", "chamfer_eval"):
                y = P[(P["domain"] == dom) & (P["group"] == grp) & (P["b"] == ref)]
                if not len(y):
                    continue
                for gts, what in ((MAIN_GTS, ""), (VAR_GTS, ", varianti di F")):
                    md += ["", f"Delta appaiati metodo - {METHOD_LABEL[ref]}{what} (P(delta <= 0) fra parentesi):", ""]
                    rows = []
                    for m in methods:
                        if m == ref:
                            continue
                        cells = []
                        for g in gts:
                            r = y[(y["a"] == m) & (y["gt"] == g)].iloc[0]
                            cells.append(f"{ci(r, True)} ({r['p_le0']:.3f})")
                        rows.append([METHOD_LABEL.get(m, m)] + cells)
                    md += table([f"metodo - {METHOD_LABEL[ref]}"] + [GT_LABEL[g] for g in gts], rows)
            md.append("")


def section_controls(md: list) -> None:
    c = pd.read_csv(E / "methods_controls.csv")
    agg = c.groupby("source")["abs_diff"].agg(["count", "max"]).reset_index()
    md += ["## 6. Controlli di riproduzione (GT maxabs e unificata)", ""]
    md += table(["sorgente", "numeri confrontati", "max |diff|"],
                [[r.source, str(int(r.count)), f"{r.max:.1e}"] for r in agg.itertuples()])
    md.append("")


def main() -> None:
    md = ["# E12: GT canonica (9 ottobre 2026)", "",
          "Protocollo: `protocol.md` (sezioni 0-5 ed Emendamento 1, scritti prima dei numeri; hash in "
          "`protocol.sha256` e `protocol_amend1.sha256`). Codice: `v3_work/canonical_gt/` (gt.py, methods.py, "
          "report.py; controlli in check_cgt.py). Matrici GT: `datasets/CANONICAL_GT/` (fuori da git).", ""]
    if (E / "conclusions.md").exists():
        md += [(E / "conclusions.md").read_text().strip(), ""]
    for f in (section_frames, section_size, section_corr, section_ident, section_methods, section_controls):
        try:
            f(md)
        except FileNotFoundError as e:
            md += [f"({f.__name__}: manca {e.filename})", ""]
    (E / "summary.md").write_text("\n".join(md) + "\n")
    print(f"[e12-report] {E / 'summary.md'}: {len(md)} righe")


if __name__ == "__main__":
    main()
