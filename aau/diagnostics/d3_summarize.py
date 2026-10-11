#!/usr/bin/env python3
"""D3 con l'emendamento 1: ``aau/runs/evidence/diagnostics/results_d3.md`` dalle uscite di d3_stats.py (solo
presentazione, nessun numero nuovo).

    AAU_NV="" aau/run.sh aau/diagnostics/d3_summarize.py          (d3.sbatch, passo summary)
"""
from __future__ import annotations

import csv
import json

import numpy as np

import diag

SHORT = {"factorized_s1234": "fact. s1234", "factorized_s2345": "fact. s2345", "factorizedc3m_e123": "C3M e123"}
METH = {"i": "(i) appresa, tutte tranne T", "i_nf": "(i) senza FaMoS", "ii": "(ii) intra-soggetto, tutte tranne T",
        "ii_nf": "(ii) senza FaMoS", "coral": "CORAL (esplorativa)"}
TESTS = (("facescape", "FaceScape (3DMM bilineare)"), ("hifi3d", "HIFI3D (AI-NEXT)"),
         ("faceverse_neutral", "FaceVerse neutra"), ("flame2023_s1", "FLAME 2023 (D1)"),
         ("faceverse", "FaceVerse con espressioni (secondaria)"), ("famos", "FaMoS TRAIN, reale (secondaria)"))
PRIMARY = ("facescape", "hifi3d", "faceverse_neutral", "flame2023_s1")
SOURCES = ("bfm", "ict", "gnm", "flame2023_s1", "facescape", "hifi3d", "faceverse", "famos")
NOTES: list[str] = []      # note del coder, scritte DOPO i numeri (commento, non regole)


def cell(p, lo, hi, sign=True) -> str:
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"{f.format(float(p))} [{f.format(float(lo))}, {f.format(float(hi))}]"


def rd_cell(r: dict) -> str:
    return cell(r["delta"], r["ci_low"], r["ci_high"])


def verdict_lines(name: str, v: dict) -> list[str]:
    out = [f"- **{name}: {v['verdict']}**"]
    for arm, x in v["arms"].items():
        cells = "; ".join(f"{t} SR {rd_cell(c['sr'])}, FR {float(c['fr']['delta']):+.3f}" for t, c in x["targets"].items())
        out.append(f"  - {SHORT[arm]}: {x['verdict']} (n_SR {x['n_sr']} su 4, FR >= -0.03 ovunque: {x['fr_ok']}); {cells}")
    return out


def main() -> None:
    rd = json.loads((diag.EV / "d3" / "readings.json").read_text())
    ctrl = json.loads((diag.EV / "d3" / "controls.json").read_text())
    sha = (diag.EV / "PROTOCOL_D3.sha256").read_text().split()[0]
    sha1 = (diag.EV / "PROTOCOL_D3_emendamento_1.sha256").read_text().split()[0]
    rho = {(r["test"], r["rows"], r["column"], r["gt"]): r for r in csv.DictReader(open(diag.EV / "d3_spearman.csv"))}
    dl = list(csv.DictReader(open(diag.EV / "d3_delta.csv")))
    ref = {(r["test"], r["rows"], r["arm"], r["method"], r["gt"]): r for r in dl if r["kind"] == "ref"}
    fam = {(r["test"], r["arm"], r["method"], r["gt"]): r for r in dl if r["kind"] == "famos" and r["rows"] == "cross"}
    out = ["# Diagnostica D3: risultati (emendamento 1, leave-one-generator-out)", "",
           f"Protocollo `PROTOCOL_D3.md` (sha256 {sha[:8]}..., commit 21afea8) ed emendamento 1 "
           f"(`PROTOCOL_D3_emendamento_1.md`, sha256 {sha1[:8]}..., commit 49fead6), scritti prima dei numeri; le "
           "anteprime del critic sono dichiarate nell'emendamento (sez. 1). FaceScape, HIFI3D e FaceVerse sono campioni di "
           "3DMM non visti dai bracci, non dati reali; reale e' solo FaMoS. Generato da `aau/diagnostics/d3_summarize.py`; "
           "numeri in `d3_spearman.csv`, `d3_delta.csv`, `d3_curve.csv`, `d3_heads.csv`, `d3_cv.csv`. IC 95% percentile, "
           "1000 repliche per soggetto di test, condizionati alle teste addestrate.", "", "## Letture", ""]
    out += verdict_lines("R_LOGO, testa (i), tutte le sorgenti tranne T (primaria)", rd["R_LOGO"])
    out += verdict_lines("Secondaria: (ii) Mahalanobis intra-soggetto", rd["secondary"]["ii"])
    out += verdict_lines("Secondaria: (i) senza FaMoS", rd["secondary"]["i_nofamos"])
    out += ["", "Delta medio sui 4 bersagli primari (replica per replica):", "",
            "| braccio | testa | SR | FR |", "|---|---|---|---|"]
    md = rd["mean_delta"]
    for arm in diag.ARMS:
        for m in ("i", "i_nf", "ii", "ii_nf"):
            if f"{arm}|{m}|sr" in md:
                out.append(f"| {SHORT[arm]} | {METH[m]} | {rd_cell(md[f'{arm}|{m}|sr'])} | {rd_cell(md[f'{arm}|{m}|fr'])} |")
    cv = rd["curve"]
    out += ["", f"Curva sul numero di sorgenti: il guadagno cresce (media non decrescente da k = 1 a 7 su >= 3 bersagli, "
            f"entrambi i semi)? (i): **{cv['i']['grows']}**; (ii): **{cv['ii']['grows']}**.", ""]
    if NOTES:
        out += ["Note del coder, scritte DOPO i numeri (commento, non regole):", ""] + NOTES + [""]
    out += ["## Delta contro d_P per bersaglio (righe incrociate)", "",
            "SR: d_h - d_P; FR: d_F,h - d_F calibrata; FR senza taglia: d_h - d_P letti con la GT FR. \"fold\" = minimo e "
            "massimo del delta SR puntuale delle 15 teste di fold (4/5 dei soggetti delle sorgenti).", ""]
    for t, tname in TESTS:
        tc = ctrl["tests"][t]
        out += [f"### {tname} ({tc['rows']} righe, {tc['subjects']} soggetti, seme {tc['seed']})", "",
                "| testa | braccio | rho SR | delta SR [IC] | fold SR | delta FR [IC] | delta FR senza taglia [IC] |",
                "|---|---|---|---|---|---|---|"]
        for m in ("i", "i_nf", "ii", "ii_nf", "coral"):
            for a in diag.ARMS:
                s = ref.get((t, "cross", a, m, "sr"))
                if s is None:
                    continue
                f, n = ref[(t, "cross", a, m, "fr")], ref[(t, "cross", a, m, "fr_nosize")]
                fs = f"[{float(s['fold_min']):+.3f}, {float(s['fold_max']):+.3f}]" if s.get("fold_min") else "-"
                out.append(f"| {METH[m]} | {SHORT[a]} | {float(s['rho_method']):.3f} | {rd_cell(s)} | {fs} | "
                           f"{rd_cell(f)} | {rd_cell(n)} |")
        out += ["", "| riferimento | SR | FR | FR senza taglia |", "|---|---|---|---|"]
        for a in diag.ARMS:
            x = rho.get((t, "cross", f"{a}|dP|shape", "sr"))
            if x is None:
                continue
            y, z = rho[(t, "cross", f"{a}|dP|form", "fr")], rho[(t, "cross", f"{a}|dP|shape", "fr")]
            out.append(f"| {SHORT[a]}: d_P / d_F cal. | {cell(x['rho'], x['ci_low'], x['ci_high'], False)} | "
                       f"{cell(y['rho'], y['ci_low'], y['ci_high'], False)} | "
                       f"{cell(z['rho'], z['ci_low'], z['ci_high'], False)} |")
        for mm, lab in (("gnm", "B-GNM (vB)"), ("flame2023", "B-FLAME (vB)")):
            x = rho.get((t, "cross", f"B-{mm}|vb_sr", "sr"))
            if x is not None:
                y = rho[(t, "cross", f"B-{mm}|vb_fr", "fr")]
                out.append(f"| {lab} | {cell(x['rho'], x['ci_low'], x['ci_high'], False)} | "
                           f"{cell(y['rho'], y['ci_low'], y['ci_high'], False)} | - |")
        out.append("")
    out += ["## Contributo del dato reale: (i) - (i) senza FaMoS", "", "| bersaglio | braccio | SR | FR |",
            "|---|---|---|---|"]
    for t in PRIMARY:
        for a in diag.ARMS:
            if (t, a, "i", "sr") in fam:
                out.append(f"| {t} | {SHORT[a]} | {rd_cell(fam[(t, a, 'i', 'sr')])} | {rd_cell(fam[(t, a, 'i', 'fr')])} |")
    out += ["", "## Invarianza contro pesatura (SR)", "",
            "Righe incrociate (topologie diverse), original-original e mediate sulle 5 etichette: rho di d_P e delta delle "
            "teste.", "", "| bersaglio | braccio | d_P incr. / orig. / mediate | (i) delta incr. / orig. / mediate | "
            "(ii) delta incr. / orig. / mediate |", "|---|---|---|---|---|"]
    for t in PRIMARY:
        for a in diag.ARMS:
            if (t, "cross", f"{a}|dP|shape", "sr") not in rho:
                continue
            dp = " / ".join(f"{float(rho[(t, rs, f'{a}|dP|shape', 'sr')]['rho']):.3f}" for rs in ("cross", "orig", "lavg"))
            cells = [" / ".join(f"{float(ref[(t, rs, a, m, 'sr')]['delta']):+.3f}" for rs in ("cross", "orig", "lavg"))
                     for m in ("i", "ii")]
            out.append(f"| {t} | {SHORT[a]} | {dp} | {cells[0]} | {cells[1]} |")
    out += ["", "## Curva sul numero di sorgenti (delta SR puntuale, righe incrociate)", "",
            "Testa (i) con gli iperparametri scelti per \"tutte tranne T\", rifittata su ogni sottoinsieme di k sorgenti, e "
            "testa (ii); media [minimo, massimo] sui sottoinsiemi (k = 7: le teste di R_LOGO).", "",
            "| bersaglio | braccio | testa | k = 1 | k = 2 | k = 4 | k = 7 |", "|---|---|---|---|---|---|---|"]
    curve = list(csv.DictReader(open(diag.EV / "d3_curve.csv")))
    for t in ("facescape", "hifi3d", "faceverse", "flame2023_s1"):
        for a in diag.ARMS:
            for m in ("i", "ii"):
                cells = []
                for k in ("1", "2", "4", "7"):
                    x = np.asarray([float(r[f"delta_{m}"]) for r in curve if r["target"] == t and r["arm"] == a
                                    and r["k"] == k])
                    cells.append("-" if not len(x) else (f"{x.mean():+.3f}" if len(x) == 1 else
                                                         f"{x.mean():+.3f} [{x.min():+.3f}, {x.max():+.3f}]"))
                out.append(f"| {t} | {SHORT[a]} | ({m}) | " + " | ".join(cells) + " |")
    out += ["", "Sottoinsiemi di una sola sorgente (k = 1), delta SR della testa (i):", "",
            "| bersaglio | braccio | " + " | ".join(SOURCES) + " |", "|---" * (len(SOURCES) + 2) + "|"]
    for t in ("facescape", "hifi3d", "faceverse", "flame2023_s1"):
        for a in diag.ARMS:
            one = {r["sources"]: float(r["delta_i"]) for r in curve if r["target"] == t and r["arm"] == a and r["k"] == "1"}
            out.append(f"| {t} | {SHORT[a]} | " + " | ".join(f"{one[s]:+.3f}" if s in one else "-" for s in SOURCES) + " |")
    out += ["", "## Teste (i) scelte (CV per soggetto dentro le sorgenti)", "",
            "| impostazione | braccio | r | lambda | CV testa | CV d_P | bordo | alpha / alpha0 | c | convergenza |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in csv.DictReader(open(diag.EV / "d3_heads.csv")):
        if r["method"] != "i":
            continue
        out.append(f"| {r['setting']} | {SHORT[r['arm']]} | {r['r']} | {float(r['lambda']):g} | "
                   f"{float(r['cv_score']):.4f} | {float(r['cv_score_dP']):.4f} | {r['edge']} | "
                   f"{float(r['alpha_over_alpha0']):.3f} | {float(r['c']):.3f} | {r['success']} ({r['nit']} it.) |")
    k1, k7 = ctrl["K1"], ctrl["K7"]
    out += ["", "## Controlli", "",
            f"- K1 (riferimenti dei bracci e B = pubblicati, punto e IC; FLAME = D1): max |scarto| "
            f"{k1['max_abs_diff']:.1e}, righe uguali {k1['rows_equal']}, **{'passa' if k1['pass'] else 'NON passa'}**; "
            f"celle senza riferimento: {[k for k, x in k1['cells'].items() if not isinstance(x, dict)]}",
            "- K2 (c dei bracci dagli held-out): " + ", ".join(f"{SHORT[a]} {x['c']:.6f} contro {x['c_median']:.6f}"
                                                              for a, x in ctrl["K2"]["arms"].items()) +
            f", **{'passa' if ctrl['K2']['pass'] else 'NON passa'}**",
            f"- K3 (GT FaMoS TRAIN contro la GT-F di E12): max |scarto| {ctrl['K3']['max_abs_mm']:.1e} mm, "
            f"**{'passa' if ctrl['K3']['pass'] else 'NON passa'}**",
            f"- K4 (sorgenti: soggetti, mesh): {json.dumps(ctrl['K4']['source_sizes'])}; embedding mancanti "
            f"{json.dumps(ctrl['K4']['missing'])}",
            f"- K5 (r = 0 contro d_P in CV): max |scarto| {ctrl['K5_r0_max_abs_diff']:.1e}",
            f"- K6: {ctrl['K6']}",
            f"- K7 (embedding dei pool contro gli store ufficiali, 5 soggetti di controllo per pool): max |dz| "
            f"{k7['max_abs']:.1e}, **{'passa' if k7['pass'] else 'NON passa'}**",
            f"- Righe per insieme: {json.dumps({t: x['rowsets'] for t, x in ctrl['tests'].items()})}",
            f"- Tempo: CV {ctrl['seconds_cv']:.0f} s, totale {ctrl['seconds']:.0f} s ({ctrl['workers']} processi)", ""]
    (diag.EV / "results_d3.md").write_text("\n".join(out) + "\n")
    print(f"[d3-summary] {diag.EV / 'results_d3.md'}", flush=True)


if __name__ == "__main__":
    main()
