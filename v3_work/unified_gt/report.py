#!/usr/bin/env python3
"""Scrive ``aau/runs/evidence/e8/summary.md`` dai json e csv dei passi precedenti.

    v3_work/unified_gt/run.sh v3_work/unified_gt/report.py

Le conclusioni stanno in ``aau/runs/evidence/e8/conclusions.md`` (scritte a mano dopo aver letto i
numeri) e vengono copiate in testa; tutto il resto e' generato, nessun numero scritto a mano.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

import ugt as C
import domains

E = C.EVID_DIR


def fmt(p, lo=None, hi=None, signed=False):
    f = "{:+.3f}" if signed else "{:.3f}"
    if lo is None or not np.isfinite(lo):
        return f.format(p)
    return f"{f.format(p)} [{f.format(lo)}, {f.format(hi)}]"


def table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return out


def rot_angle(R) -> float:
    R = np.asarray(R)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def section_corr(L: list[str]) -> None:
    reg = json.loads((C.DATA_ROOT / "flame_region.json").read_text())
    ct = json.loads((C.DATA_ROOT / "canonical_transforms.json").read_text())
    R = ct["region"]
    L += ["## 1. Regione del volto e corrispondenze", ""]
    L += [f"- **Maschere ufficiali trovate:** `v2_work/genflame/official/FLAME_masks.pkl`. Maschera `face`: "
          f"{reg['n_face_mask']} vertici, intersezione nulla con orecchie, collo, bulbi oculari e `boundary` "
          f"(23 vertici in comune con `scalp`, sul bordo della fronte).",
          f"- **Interno di occhi e bocca** tolto con una regola geometrica (visibilita' da un cono di 75 gradi "
          f"attorno a +z, occlusori = testa intera con i bulbi): {reg['n_interior']} vertici "
          f"({reg['interior_in_lips']} nelle labbra, {reg['interior_in_eye_region']} attorno agli occhi). "
          f"Regione FLAME: **{reg['n_region']} vertici**, {reg['area_mm2']:.0f} mm^2. Figura: `flame_region.png`.",
          f"- **Regione unificata** (coperta da tutti gli 8 domini): **{R['n_unified']} vertici**, "
          f"{R['area_unified_on_flame_mm2']:.0f} mm^2 sulla media FLAME "
          f"({R['area_unified_on_flame_mm2'] / R['area_flame_region_mm2']:.0%} dell'area della regione FLAME). "
          f"Vertici persi solo per colpa di un dominio: "
          + ", ".join(f"{d} {n}" for d, n in R["uncovered_only_by_domain"].items() if n)
          + f". Senza il vincolo di BFM: {R['variant_without_bfm']['n_vertices']} vertici, "
          f"{R['variant_without_bfm']['area_on_flame_mm2']:.0f} mm^2 (salvata come `ridx_nobfm`, non usata). "
          f"Figura: `unified_region.png`.", ""]
    rows = []
    for d in domains.DOMAINS:
        if d == "flame":
            continue
        j = json.loads((C.CORR_DIR / f"{d}.json").read_text())
        lm = j["landmarks"]
        c = j["chamfer_mm"]
        rows.append([d, j["n_template_vertices"], lm["n_used"],
                     f"{lm['residual_after_similarity_mm']['median']:.2f} -> {lm['residual_after_nicp_mm']['median']:.2f} "
                     f"(p95 {lm['residual_after_nicp_mm']['p95']:.2f})",
                     f"{lm['held_out_before_mm']['median']:.2f} -> {lm['held_out_after_nicp_mm']['median']:.2f} "
                     f"(p95 {lm['held_out_after_nicp_mm']['p95']:.2f}, n={lm['n_held_out']})",
                     f"{c['region_to_template']['mean']:.2f} / {c['region_to_template']['p95']:.2f}",
                     f"{c['template_to_region']['median']:.2f} / {c['template_to_region']['p95']:.2f}",
                     f"{j['coverage']['n_covered']}/{j['coverage']['n_region']}",
                     f"{j['distortion']['log_area_ratio_abs_median']:.3f} / {j['distortion']['flipped_triangles']}"])
    L += ["Controlli delle corrispondenze (mm sulla scala della media FLAME). Landmark: mediana del residuo "
          "dopo la sola similarita' -> dopo NICP, sui landmark usati e su un 20% tenuto FUORI da un secondo NICP. "
          "Chamfer: regione deformata -> template (media / p95, vertici coperti) e template -> regione "
          "(mediana / p95, vertici del template nell'impronta della regione; la coda viene da narici e "
          "orbite profonde, che la regione FLAME non ha). Distorsione: mediana di |log rapporto d'area| dei "
          "triangoli FLAME / triangoli capovolti. Figure: `corr_<dominio>.png`.", ""]
    L += table(["dominio", "vertici template", "landmark", "residuo landmark", "landmark tenuti fuori",
                "Chamfer regione->template", "Chamfer template->regione", "coperti", "distorsione / capovolti"], rows)
    L += [""]


def section_canon(L: list[str]) -> None:
    ct = json.loads((C.DATA_ROOT / "canonical_transforms.json").read_text())
    L += ["## 2. Trasformazioni canoniche", "",
          "`datasets/UNIFIED_GT/canonical_transforms.json`: " + ct["description"], ""]
    rows = []
    for d, c in ct["domains"].items():
        rows.append([d, f"{c['s']:.6g}", f"{rot_angle(c['R']):.1f}", "si'" if c["flip_faces"] else "no",
                     f"{c['residual_to_flame_mean_mm']['rms_area_weighted']:.2f}", c["units_in"]])
    L += table(["dominio", "scala s", "rotazione (gradi)", "flip_faces", "residuo verso la media FLAME (mm, rms)",
                "unita' dei dati"], rows)
    L += ["", "Il residuo e' la differenza di forma fra la media del dominio e quella di FLAME dopo la similarita' "
          "(non un errore): e' la parte che la GT unificata vede come distanza fra le medie.", ""]
    chk = E / "canonical_check.json"
    if chk.exists():
        ck = json.loads(chk.read_text())
        L += ["Controllo da lettore (`check_canonical.py`): una mesh di DATI per dominio presa dal disco, trasformata "
              "col json (e facce invertite se `flip_faces`), nessun altro allineamento. Normale media dei triangoli "
              "davanti lungo +z (1 = uscente), RMS dalla media FLAME sulla regione unificata (include la differenza "
              "d'identita'; un asse sbagliato darebbe decine di mm).", ""]
        L += table(["dominio", "mesh", "normale z", "RMS dalla media FLAME (mm)"],
                   [[d, r["mesh"], f"{r['normal_z']:.4f}", f"{r['rms_to_flame_mm']:.2f}"] for d, r in ck.items()])
        L += [""]


def section_gt(L: list[str], ev: dict) -> None:
    man = json.loads((C.DATA_ROOT / "shapes" / "manifest.json").read_text())
    sp = ev["space"]
    L += ["## 3. GT unificata", "",
          f"- s_i: forma neutra -> mappa baricentrica sulla regione unificata ({sp['n_vertices']} vertici) -> "
          f"Procrustes di similarita' pesato per area verso la media globale mu -> pesi sqrt(area). mu = Procrustes "
          f"generalizzato delle medie dei domini di training (flame, bfm, ict, gnm, facescape), a scala e frame FLAME. "
          f"Area totale A = {sp['area_mm2']:.0f} mm^2. g_ij = ||s_i - s_j|| / sqrt(A): RMS pesato per area, in mm.",
          "- Dati: `datasets/UNIFIED_GT/` (`unified_space.npz`, `shapes/<set>.npz`, `gt/<set>_unified*.npz` nel formato "
          "di `load_gt_distance_matrix`). Codice: `v3_work/unified_gt/`.", ""]
    rows = []
    for name, m in man.items():
        rows.append([name, m["n"], f"{m['rms_to_mu_mm']['median']:.2f}",
                     "-" if "linear_vs_mesh_max_abs" not in m else f"{m['linear_vs_mesh_max_abs']:.1e}",
                     "-" if "weights_vs_stored_max_abs" not in m else f"{m['weights_vs_stored_max_abs']:.1e}"])
    L += table(["set", "identita'", "RMS da mu (mm, mediana)", "mappa lineare vs mesh generata (max |diff|)",
                "mesh dai pesi vs mesh su disco (max |diff|, unita' del dominio)"], rows)
    L += [""]


def section_a(L: list[str], ev: dict) -> None:
    L += ["## 4. (a) Accordo fra GT sugli stessi soggetti di valutazione", "",
          "Spearman sulle 4.950 coppie dei 100 soggetti valutati (IC 95% bootstrap per soggetto, 1000 repliche, "
          "seme 1234) e, solo punto, sulle 124.750 coppie del pool di 500. Le righe 1-5 cambiano un ingrediente "
          "alla volta da maxabs all'unificata (GT intermedie di `native_gt.py`, sulla patch `original` delle viste); "
          "tutte le coppie in `evidence.json`.", ""]
    # la catena: un ingrediente alla volta da maxabs all'unificata (native_gt.py), piu' coef e la variante
    chain = [("maxabs", "coef", "maxabs contro coefficienti"),
             ("unified", "coef", "unificata contro coefficienti"),
             ("unified", "maxabs", "**unificata contro maxabs**"),
             ("maxabs", "native_maxabs_rms", "1. media delle norme -> RMS (stessa normalizzazione maxabs)"),
             ("native_maxabs_rms", "native_sim", "2. maxabs -> Procrustes di similarita' (patch nativa)"),
             ("native_sim", "native_sim_area", "3. vertici uniformi -> pesi d'area"),
             ("native_sim_area", "native_sim_area_region", "4. patch intera -> impronta della regione unificata"),
             ("native_sim_area_region", "unified", "5. vertici nativi -> mappa FLAME (stessa regione)"),
             ("unified", "unified_pairwise", "variante: Procrustes per coppia invece che verso mu")]
    rows = []
    for dom, r in ev["a"].items():
        for a, b, lab in chain:
            k = f"{a}|{b}" if f"{a}|{b}" in r["spearman_eval"] else f"{b}|{a}"
            v = r["spearman_eval"][k]
            rows.append([dom, lab, fmt(v["point"], v["ci_low"], v["ci_high"]), f"{r['spearman_pool500'][k]:.3f}"])
    L += table(["dominio", "GT", "Spearman (100 soggetti)", "Spearman (pool 500)"], rows)
    L += [""]
    for dom, r in ev["a"].items():
        u = r["unified_mm"]
        L.append(f"- {dom}: g unificata sui 100 soggetti mediana {u['median']:.2f} mm (min {u['min']:.2f}, "
                 f"p5 {u['p5']:.2f}, max {u['max']:.2f}); variante per coppia / globale, rapporto mediano "
                 f"{r['pairwise_over_global_median']:.3f}, Pearson {r['pearson_unified_vs_pairwise']:.4f}.")
    v = ev["variant_cross_domain"]
    L += [f"- Variante con Procrustes per coppia contro globale, campione fra domini ({v['n_shapes']} forme, 100 per "
          f"set): Spearman {v['spearman_all']:.4f} su tutte le coppie, {v['spearman_cross_domain_pairs']:.4f} sulle "
          f"coppie fra domini diversi; rapporto mediano {v['pairwise_over_global_median']:.3f}.", ""]


def section_b(L: list[str], ev: dict) -> None:
    b = ev["b"]
    L += ["## 5. (b) Quasi-duplicati fra domini (training del run su scala contro test)", "",
          f"Pool di training (split `aau/data_scale/split_scale_all.json`): " +
          ", ".join(f"{d} {n}" for d, n in b["pool_sizes"].items()) +
          ". Distanze unificate in mm; per identita' di test il vicino piu' prossimo in ogni pool.", ""]
    for dom in ("hifi3d", "faceverse"):
        r = b[dom]
        rows = [["dentro il test (99 altri)", fmt(r["nn_within_test100"]["median"]), fmt(r["nn_within_test100"]["min"]),
                 fmt(r["nn_within_test100"]["p5"])],
                ["dentro il pool di 500 (499 altri)", fmt(r["nn_within_pool500"]["median"]),
                 fmt(r["nn_within_pool500"]["min"]), fmt(r["nn_within_pool500"]["p5"])]]
        for d, q in r["nn_train_full"].items():
            rows.append([f"training {d} (tutto)", fmt(q["median"]), fmt(q["min"]), fmt(q["p5"])])
        for n, per in r["nn_train_eqsize"].items():
            for d, q in per.items():
                rows.append([f"training {d}, {n} identita' (mediana su 50 sottocampioni)", fmt(q["median"]),
                             fmt(q["min"]), fmt(q["p5"])])
        L += [f"**{dom}** ({r['n_test']} identita' di test)", ""]
        L += table(["vicino piu' prossimo in", "mediana", "minimo", "p5"], rows)
        L += ["", f"- identita' di test col vicino di training piu' vicino del vicino dentro il pool di 500: "
              f"{r['n_train_closer_than_within500']}/{r['n_test']}; piu' vicino del minimo di tutto il test: "
              f"{r['n_train_below_min_within500']}; dominio del vicino piu' prossimo (pool interi): "
              + ", ".join(f"{d} {n}" for d, n in r["argmin_domain_full"].items()) + "."]
        rows = []
        for k, v in r["paired_eqsize"].items():
            rows.append([k, f"{v['ratio_geomean']:.3f}", fmt(v["mean_log_ratio"], v["ci_low"], v["ci_high"], True),
                         f"{v['frac_first_closer']:.2f}"])
        L += ["", "A taglia uguale, per identita' di test: rapporto fra le distanze dal vicino (primo / secondo "
              "dominio), media geometrica e IC bootstrap sui soggetti di test del log-rapporto medio.", ""]
        L += table(["confronto", "rapporto (media geom.)", "log-rapporto medio [IC 95%]",
                    "frazione col primo piu' vicino"], rows)
        L += [""]


def section_c(L: list[str], ev: dict) -> None:
    c = ev["c"]
    doms = c["domains"]
    M = np.array(c["means_mm"])
    L += ["## 6. (c) Distanze fra le medie dei domini (mm, spazio unificato)", ""]
    L += table([""] + doms, [[d] + [f"{M[i, j]:.2f}" for j in range(len(doms))] for i, d in enumerate(doms)])
    L += ["", "Dispersione interna (stessa metrica):", ""]
    rows = [[n, s["n"], f"{s['pairwise_median']:.2f}", f"{s['nn_median']:.2f}", f"{s['to_template_mean_median']:.2f}",
             f"{s['empirical_mean_vs_template_mean']:.2f}"] for n, s in c["spread"].items()]
    L += table(["set", "identita'", "distanza mediana fra coppie", "vicino piu' prossimo (mediana)",
                "distanza dalla media del template (mediana)", "media empirica vs media del template"], rows)
    L += ["", "Distanza mediana delle identita' di test dalle medie dei domini:", ""]
    rows = [[t] + [f"{v[d]:.2f}" for d in doms] for t, v in c["test_to_means_median"].items()]
    L += table(["test"] + doms, rows)
    L += [""]


def section_methods(L: list[str]) -> None:
    R = pd.read_csv(E / "methods_spearman.csv")
    P = pd.read_csv(E / "methods_paired.csv")
    G = pd.read_csv(E / "methods_gt_diff.csv")
    K = pd.read_csv(E / "methods_controls.csv")
    from eval_methods import LABEL
    L += ["## 7. Metodi esistenti con la GT unificata (richiesta del critic, priorita' alta)", "",
          "GT primaria = maxabs (dichiarata); l'unificata e' secondaria. Stessi soggetti, stesse righe e stesse "
          "repliche bootstrap dei summary esistenti (un seme per dominio e gruppo: quello della differenza "
          "pubblicata e108 - Chamfer eval); cambia solo la colonna della GT. Codice: `eval_methods.py`; csv: "
          "`methods_*.csv`. IC 95% bootstrap per soggetto, 1000 repliche.", ""]
    for (dom, group), sub in R.groupby(["domain", "group"], sort=False):
        piv = {g: sub[sub["gt"] == g].set_index("method")
               for g in ("maxabs", "unified", "unified_pairwise", "coef", "native_sim", "native_sim_area",
                         "native_sim_area_region")}
        order_u = piv["unified"]["point"].sort_values(ascending=False)
        rk_m = piv["maxabs"]["point"].rank(ascending=False).astype(int)
        rk_u = piv["unified"]["point"].rank(ascending=False).astype(int)
        gd = G[(G["domain"] == dom) & (G["group"] == group) & (G["gt_a"] == "unified") & (G["gt_b"] == "maxabs")]
        gd = gd.set_index("method")
        rows = []
        for m in order_u.index:
            x, u, c, q = piv["maxabs"].loc[m], piv["unified"].loc[m], piv["coef"].loc[m], piv["unified_pairwise"].loc[m]
            ns, na, nr = piv["native_sim"].loc[m], piv["native_sim_area"].loc[m], piv["native_sim_area_region"].loc[m]
            d = gd.loc[m]
            rows.append([LABEL.get(m, m), fmt(x.point, x.ci_low, x.ci_high), fmt(u.point, u.ci_low, u.ci_high),
                         fmt(d.point, d.ci_low, d.ci_high, True), f"{q.point:.3f}", f"{ns.point:.3f}",
                         f"{na.point:.3f}", f"{nr.point:.3f}", f"{c.point:.3f}", f"{rk_m[m]} -> {rk_u[m]}",
                         int(u.n_nan)])
        L += [f"### {dom}, {group}", ""]
        L += table(["metodo", "GT maxabs", "GT unificata", "unificata - maxabs (appaiata)", "unificata per coppia",
                    "nativa sim.", "nativa sim. + area", "nativa sim. + area, regione unificata", "GT coef",
                    "rango maxabs -> unificata", "NaN"], rows)
        for ref in ("chamfer_eval", "scale_e108"):
            pp = P[(P["domain"] == dom) & (P["group"] == group) & (P["b"] == ref)]
            if not len(pp):
                continue
            rows = []
            for m in order_u.index:
                if m == ref:
                    continue
                a = pp[(pp["a"] == m) & (pp["gt"] == "maxabs")]
                b = pp[(pp["a"] == m) & (pp["gt"] == "unified")]
                if not len(a):
                    continue
                a, b = a.iloc[0], b.iloc[0]
                rows.append([LABEL.get(m, m), f"{fmt(a.point, a.ci_low, a.ci_high, True)} ({a.p_le0:.3f})",
                             f"{fmt(b.point, b.ci_low, b.ci_high, True)} ({b.p_le0:.3f})"])
            L += ["", f"Delta appaiati contro {LABEL.get(ref, ref)} (P(<=0) fra parentesi):", ""]
            L += table(["metodo - " + LABEL.get(ref, ref), "GT maxabs", "GT unificata"], rows)
        L += [""]
    L += ["### Controlli (GT maxabs e coef: i numeri pubblicati devono tornare)", ""]
    L += table(["dominio", "riga", "pubblicato", "ricalcolato", "|diff|"],
               [[r.domain, r.what, f"{r.published:.4f}", f"{r.recomputed:.4f}", f"{r.abs_diff:.1e}"]
                for r in K.itertuples()])
    L += [""]


def main() -> None:
    ev = json.loads((E / "evidence.json").read_text())
    L = [f"# E8: GT unificata e controlli fra domini ({date.today().isoformat()})", "",
         "Piano: `paper/PLAN_MASSIVE.md`, sezioni 3 e 13 (E8). Generato da `v3_work/unified_gt/report.py`; "
         "le conclusioni in testa da `conclusions.md`.", ""]
    concl = E / "conclusions.md"
    if concl.exists():
        L += [concl.read_text().strip(), ""]
    section_corr(L)
    section_canon(L)
    section_gt(L, ev)
    section_a(L, ev)
    section_b(L, ev)
    section_c(L, ev)
    if (E / "methods_spearman.csv").exists():
        section_methods(L)
    (E / "summary.md").write_text("\n".join(L) + "\n")
    print(f"[report] scritto {E / 'summary.md'}")


if __name__ == "__main__":
    main()
