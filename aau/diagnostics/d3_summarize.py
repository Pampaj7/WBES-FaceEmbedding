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
NOTES = [                  # note del coder, scritte DOPO i numeri (commento, non regole); corrette dopo il critic
    "- R_LOGO e' NO in entrambi i semi: nessun bersaglio arriva a +0.05. Il guadagno piu' grande e' FaceScape s1234, "
    "+0.038 [+0.014, +0.065], sotto la soglia; la media sui 4 bersagli e' +0.016 [+0.003, +0.029] (s1234) e -0.001 "
    "[-0.012, +0.011] (s2345). FR non peggiora oltre -0.03 da nessuna parte con la testa (i). Verdetto preregistrato, "
    "invariato dopo il critic e dopo l'emendamento 2 (POST HOC, sotto).",
    "- La CV dentro le sorgenti sceglie r = 0 (d_P invariata) in 3 impostazioni su 9 per s1234 e in 6 su 9 per s2345; dove "
    "sceglie una correzione, lambda e' sempre sul bordo basso (1e-4). **ARTEFATTO DELLA GRIGLIA DI LAMBDA** (critic, "
    "emendamento 2): la penalita' agisce su x non standardizzate e con lambda >= 1e-2 azzera W, quindi dei 5 valori "
    "preregistrati solo il bordo lasciava una correzione. Con lambda fino a 1e-6 e la perdita riscalata la CV sceglie "
    "lambda = 1e-6 in 30 impostazioni su 30 e r = 64 in 27 (r = 32 nelle altre), con guadagno di CV sulle sorgenti da +0.045 "
    "a +0.064; di nuovo sul bordo della griglia estesa (riportato, griglia non allargata).",
    "- Sorgenti singole nella catena preregistrata (iperparametri fissi, k = 1): FLAME da solo porta FaceScape a +0.060 "
    "(s1234) e +0.086 (C3M), FaMoS da solo a +0.068 e +0.071, HIFI3D da solo a -0.074 (s1234); tutte insieme danno meno "
    "della sorgente migliore. **LETTURA CORRETTA** (critic, emendamento 2): ne avevo dedotto che una correzione lineare "
    "comune non esiste, ed e' FALSO. La correzione comune esiste se il generatore e' visto: la testa congiunta su tutte le 8 "
    "sorgenti, compresi i pool non valutati dei bersagli, migliora i 4 bersagli insieme (media +0.120 [+0.098, +0.142] "
    "s1234, +0.102 [+0.076, +0.130] s2345). Fallisce l'estrapolazione a un generatore mai visto, che resta piccola (LOGO "
    "con la catena corretta: media +0.025 [+0.004, +0.045] e +0.011 [-0.013, +0.037]). La curva piatta di s2345 e il suo "
    "\"non risolto\" erano un **ARTEFATTO DELLA GRIGLIA** (r = 0 scelto): con la catena corretta il guadagno cresce col "
    "numero di sorgenti su tutti i bersagli (una sola sorgente da -0.03 a -0.08 in media, tutte e 7 da -0.03 a +0.06).",
    "- (ii) intra-soggetto stimata sulle sorgenti peggiora HIFI3D (-0.069 e -0.059 in SR con IC sotto 0; FR -0.031 e "
    "-0.026: per s1234 oltre la soglia), e' nulla su FaceScape e +0.056 / +0.025 su FaceVerse (IC che tocca lo 0): la "
    "covarianza intra-soggetto delle sorgenti non e' quella del bersaglio (l'anteprima P4, +0.05-0.07 su FaceScape, la "
    "stimava sul bersaglio stesso). Non ha lambda ne' perdita: l'emendamento 2 non la tocca.",
    "- CORAL (esplorativa, covarianza dal pool non valutato): da -0.12 a -0.21 su FaceScape e HIFI3D, circa 0 su FaceVerse, "
    "come l'anteprima P4.",
    "- La media sulle etichette alza d_P su FaceScape (0.747 -> 0.802, s1234) e il guadagno della testa (i) scende da "
    "+0.038 (righe incrociate) a +0.019 (mediate). **LETTA MALE** (critic, emendamento 2): non e' riduzione del rumore ne' "
    "un effetto della original. Il guadagno viene dall'avere la STESSA discretizzazione ai due lati della riga: d_P "
    "FaceScape s1234 remesh-remesh 0.800, up60k-up60k 0.802, original-original 0.778, contro 0.747 incrociate "
    "(remesh-remesh - incrociate +0.053 [+0.041, +0.068]); HIFI3D +0.020, FaceVerse +0.012, FLAME +0.026. L'aumento al "
    "test da provare e' una ridiscretizzazione canonica (sempre lo stesso remesher ai due lati): colpisce il gap di "
    "discretizzazione, non quello di generatore.",
    "- Dato reale (FaMoS): nella catena preregistrata il contributo passava dalla scelta r = 0 / r = 32 (+0.038 su "
    "FaceScape s1234, zero altrove): **ARTEFATTO DELLA GRIGLIA**. Con la catena corretta (i-e2) - (i-e2 senza FaMoS) vale "
    "+0.025 [+0.015, +0.036] su FaceScape (s1234), +0.023 / +0.013 / +0.031 su FLAME, +0.030 [+0.004, +0.055] su FaceVerse "
    "(s1234), circa 0 su HIFI3D: il reale aggiunge poco, ma non zero. FaMoS come bersaglio (secondario): +0.021 (s1234), "
    "-0.018 (s2345), +0.026 (C3M) preregistrati; +0.053, +0.029, +0.043 con la catena corretta.",
    "- C3M e123 (descrittivo): media +0.007; su HIFI3D la testa preregistrata peggiora (-0.031 [-0.053, -0.011]), con la "
    "catena corretta +0.061 [+0.026, +0.098]. B-GNM resta sopra tutte le teste LOGO su FaceScape SR (0.854).",
]


E2_NOTES = [               # note del coder sull'emendamento 2, scritte DOPO i suoi numeri
    "- Il verdetto preregistrato resta NO e la catena corretta non lo ribalta: n_SR = 0 in entrambi i semi, media sui 4 "
    "bersagli +0.025 [+0.004, +0.045] (s1234) e +0.011 [-0.013, +0.037] (s2345). Lettura: **la testa lineare trasferisce "
    "poco (+0.01 / +0.03) a un generatore mai visto, sotto la soglia**; non \"non trasferisce\".",
    "- **La correzione comune esiste se il generatore e' visto**: la testa congiunta (descrittiva, non LOGO) migliora tutti "
    "e 4 i bersagli insieme, media +0.120 [+0.098, +0.142] (s1234), +0.102 [+0.076, +0.130] (s2345), +0.122 (C3M); s1234 "
    "FaceScape +0.142, HIFI3D +0.153, FaceVerse +0.112, FLAME +0.075, tutti con IC sopra 0 (FaceVerse s2345 +0.059 con IC "
    "che tocca lo 0). E' l'estrapolazione a un generatore mai visto che resta piccola.",
    "- Confronto col critic. Alla sua configurazione fissa (r = 32, lambda = 1e-5, perdita riscalata) i 16 delta LOGO e "
    "congiunti coincidono coi suoi (scarto 0, tabella sotto). La sua catena (CV ridotta: r in {8, 32, 64}, lambda in {1e-5, "
    "1e-4, 1e-3}, 2 ripetizioni) da' medie piu' alte della nostra (+0.033 [+0.016, +0.051] e +0.022 [+0.001, +0.043] "
    "contro +0.025 e +0.011) perche' la nostra CV completa sceglie lambda = 1e-6, il bordo della griglia estesa, che sulle "
    "sorgenti vince (+0.045 / +0.064 di CV) ma estrapola peggio di 1e-5: FaceScape s1234 +0.050 a (32, 1e-5) contro +0.034 a "
    "(64, 1e-6), FaceVerse s2345 +0.008 contro -0.029. La CV sulle sorgenti premia l'adattamento ai generatori visti, non il "
    "trasferimento. In entrambe le catene n_SR = 0. Sulla testa congiunta la nostra scelta (64, 1e-6) da' piu' del critic "
    "su FaceScape, HIFI3D e FaceVerse (+0.142 / +0.153 / +0.112 contro +0.133 / +0.122 / +0.091), uguale su FLAME (+0.075 "
    "contro +0.076).",
    "- Curva: con la catena corretta il guadagno cresce col numero di sorgenti su tutti i bersagli (verdetto \"si\"); una "
    "sola sorgente in media peggiora (da -0.03 a -0.08), tutte e 7 vanno da -0.03 a +0.06. Il contributo di FaMoS e' "
    "piccolo ma positivo su FaceScape (s1234), FLAME e FaceVerse (s1234), nullo su HIFI3D.",
    "- d_P con la stessa discretizzazione ai due lati: su FaceScape tutte le righe con la stessa etichetta (0.767-0.808) "
    "stanno sopra le incrociate (0.746-0.753) e la original non e' la migliore (0.778-0.792); remesh-remesh - incrociate "
    "+0.050 / +0.062 su FaceScape, +0.02 su HIFI3D, +0.01 / +0.02 su FaceVerse, +0.03 su FLAME. Conferma il punto 4 del "
    "critic (i suoi 0.800 e 0.802 per s1234 sono i nostri).",
    "- Bordi e riproducibilita': tutte le scelte (i-e2) hanno lambda = 1e-6 e 27 su 30 r = 64; la CV potrebbe salire ancora "
    "con meno cresta (griglia non allargata, come da protocollo). Le CV di s2345 e C3M sono state finite da due aiutanti su "
    "altri nodi: i 952 e 903 fit calcolati due volte differiscono fino a 0.008 nel punteggio (BLAS diverse fra nodi; con "
    "lambda = 1e-6 la perdita e' quasi senza cresta); la valutazione gira sullo stesso nodo del run preregistrato (K_pre = 0).",
]


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


def e2_section() -> list[str]:
    """Sezione POST HOC dell'emendamento 2 (d3_e2.py), se le sue uscite esistono."""
    p = diag.EV / "d3" / "e2" / "readings.json"
    if not p.exists():
        return []
    rd = json.loads(p.read_text())
    ctrl = json.loads((diag.EV / "d3" / "e2" / "controls.json").read_text())
    sha2 = (diag.EV / "PROTOCOL_D3_emendamento_2.sha256").read_text().split()[0]
    D = {(r["test"], r["rows"], r["arm"], r["method"], r["gt"]): r for r in csv.DictReader(open(diag.EV / "d3_e2_delta.csv"))}
    rho = {(r["test"], r["rows"], r["column"], r["gt"]): r for r in csv.DictReader(open(diag.EV / "d3_e2_spearman.csv"))}
    tgt = {"facescape": "facescape", "hifi3d": "hifi3d", "faceverse_neutral": "faceverse", "flame2023_s1": "flame2023_s1",
           "faceverse": "faceverse", "famos": "famos"}
    out = ["## POST HOC, emendamento 2: griglia di lambda estesa, perdita riscalata per sorgente, testa congiunta", "",
           f"`PROTOCOL_D3_emendamento_2.md` (sha256 {sha2[:8]}..., commit 36fe43e), scritto DOPO i numeri sopra e dopo il "
           "critic (RISERVE). Non cambia il verdetto preregistrato (R_LOGO: NO). (i-e2) = testa (i) con lambda in {1e-6, "
           "..., 1} e la GT di ogni sorgente divisa per il suo alpha0 nella perdita; stessa CV, scelta, rifit, righe e "
           "repliche. Numeri in `d3_e2_delta.csv`, `d3_e2_spearman.csv`, `d3_e2_curve.csv`, `d3_e2_heads.csv`, "
           "`d3_e2_cv.csv`.", ""]
    out += verdict_lines("(i-e2), regola di R_LOGO applicata POST HOC (descrittiva)", rd["R_LOGO_i_e2"])
    out += ["", "Note del coder sull'emendamento 2, scritte DOPO i numeri (commento, non regole):", ""] + E2_NOTES
    out += ["", "LOGO, preregistrata contro post hoc (righe incrociate; delta contro d_P dello stesso braccio):", "",
            "| bersaglio | braccio | scelta e2 (r, lambda) | SR preregistrata | SR (i-e2) | (i-e2) - preregistrata | "
            "FR (i-e2) |", "|---|---|---|---|---|---|---|"]
    for t in PRIMARY + ("faceverse", "famos"):
        for a in diag.ARMS:
            e2 = D.get((t, "cross", a, "i_e2-dP", "sr"))
            if e2 is None:
                continue
            ch = rd["chosen"][f"{a}|{tgt[t]}|all"]
            out.append(f"| {t} | {SHORT[a]} | {ch[0]}, {float(ch[1]):g} | {rd_cell(D[(t, 'cross', a, 'i_pre-dP', 'sr')])} | "
                       f"{rd_cell(e2)} | {rd_cell(D[(t, 'cross', a, 'i_e2-i_pre', 'sr')])} | "
                       f"{rd_cell(D[(t, 'cross', a, 'i_e2-dP', 'fr')])} |")
    md = rd["mean"]
    out += ["", "Medie sui 4 bersagli primari (replica per replica; testa congiunta su FLAME sulle righe 100-199):", "",
            "| braccio | SR preregistrata | SR (i-e2) | FR (i-e2) | SR congiunta | FR congiunta |", "|---|---|---|---|---|---|"]
    for a in diag.ARMS:
        if f"{a}|i_e2-dP|sr" in md:
            out.append(f"| {SHORT[a]} | {rd_cell(md[f'{a}|i_pre-dP|sr'])} | {rd_cell(md[f'{a}|i_e2-dP|sr'])} | "
                       f"{rd_cell(md[f'{a}|i_e2-dP|fr'])} | {rd_cell(md[f'{a}|joint-dP|sr'])} | "
                       f"{rd_cell(md[f'{a}|joint-dP|fr'])} |")
    out += ["", "Testa congiunta (DESCRITTIVA, non LOGO: tutte le 8 sorgenti, compresi i pool non valutati dei bersagli; FLAME "
            "soggetti 0-99 in training, righe 100-199 in test con bootstrap sui 100 di test):", "",
            "| bersaglio | braccio | scelta (r, lambda) | SR [IC] | FR [IC] | congiunta - (i-e2) SR [IC] |",
            "|---|---|---|---|---|---|"]
    for t in PRIMARY + ("faceverse",):
        rows = "cross_half" if t == "flame2023_s1" else "cross"
        for a in diag.ARMS:
            j = D.get((t, rows, a, "joint-dP", "sr"))
            if j is None:
                continue
            ch = rd["chosen"][f"{a}|joint"]
            out.append(f"| {t} | {SHORT[a]} | {ch[0]}, {float(ch[1]):g} | {rd_cell(j)} | "
                       f"{rd_cell(D[(t, rows, a, 'joint-dP', 'fr')])} | {rd_cell(D[(t, rows, a, 'joint-i_e2', 'sr')])} |")
    out += ["", "Contributo di FaMoS con la catena e2: (i-e2) - (i-e2 senza FaMoS):", "",
            "| bersaglio | braccio | SR | FR |", "|---|---|---|---|"]
    for t in PRIMARY:
        for a in diag.ARMS:
            x = D.get((t, "cross", a, "i_e2-i_nf_e2", "sr"))
            if x is not None:
                out.append(f"| {t} | {SHORT[a]} | {rd_cell(x)} | {rd_cell(D[(t, 'cross', a, 'i_e2-i_nf_e2', 'fr')])} |")
    cv = rd["curve_i_e2"]
    curve = list(csv.DictReader(open(diag.EV / "d3_e2_curve.csv")))
    out += ["", f"Curva sul numero di sorgenti con (i-e2) (iperparametri di \"tutte tranne T\"): cresce? **{cv['grows']}**",
            "", "| bersaglio | braccio | k = 1 | k = 2 | k = 4 | k = 7 |", "|---|---|---|---|---|---|"]
    for t in ("facescape", "hifi3d", "faceverse", "flame2023_s1"):
        for a in diag.ARMS:
            cells = []
            for k in ("1", "2", "4", "7"):
                x = np.asarray([float(r["delta_i_e2"]) for r in curve if r["target"] == t and r["arm"] == a and r["k"] == k])
                cells.append("-" if not len(x) else (f"{x.mean():+.3f}" if len(x) == 1 else
                                                     f"{x.mean():+.3f} [{x.min():+.3f}, {x.max():+.3f}]"))
            out.append(f"| {t} | {SHORT[a]} | " + " | ".join(cells) + " |")
    out += ["", "d_P con la STESSA discretizzazione ai due lati della riga (una riga per coppia di soggetti di test), SR:", "",
            "| bersaglio | braccio | incrociate | " + " | ".join(f"{lab}-{lab}" for lab in diag.LABELS) +
            " | mediate | remesh-remesh - incrociate [IC] |", "|---" * (len(diag.LABELS) + 5) + "|"]
    for t in PRIMARY:
        for a in diag.ARMS:
            if (t, "cross", f"{a}|dP|shape", "sr") not in rho:
                continue
            vals = [float(rho[(t, rs, f"{a}|dP|shape", "sr")]["rho"]) for rs in
                    ("cross",) + tuple(f"same_{lab}" for lab in diag.LABELS) + ("lavg",)]
            dr = D[(t, "same_remesh", a, "dP", "sr")]
            out.append(f"| {t} | {SHORT[a]} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {rd_cell(dr)} |")
    kc = ctrl["K_critic"]
    out += ["", "Confronto col critic alla sua configurazione fissa (r = 32, lambda = 1e-5, perdita riscalata), delta SR "
            "puntuale:", "", "| cella | critic | nostro | scarto |", "|---|---|---|---|"]
    for k, x in kc["cells"].items():
        out.append(f"| {k.replace('|', ' ')} | {x['critic']:+.4f} | {x['ours']:+.4f} | {x['abs_diff']:.4f} |")
    out += ["", f"Controlli: K1 (riferimenti = pubblicati) {ctrl['K1']['max_abs_diff']:.1e}, "
            f"**{'passa' if ctrl['K1']['pass'] else 'NON passa'}**; K_pre (teste preregistrate rifittate = delta "
            f"preregistrati, {ctrl['K_pre']['n_cells']} celle, punto e IC) {ctrl['K_pre']['max_abs_diff']:.1e}, "
            f"**{'passa' if ctrl['K_pre']['pass'] else 'NON passa'}**; K_critic max scarto {kc['max_abs_diff']:.4f}, "
            f"**{'passa' if kc['pass'] else 'NON passa'}**; K5 {ctrl['K5_r0_max_abs_diff']:.1e}; {ctrl['n_cv_fits']} fit di "
            f"CV ({ctrl['n_configs']} configurazioni); tempo della valutazione {ctrl['seconds']:.0f} s.", ""]
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
    out += e2_section()
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
    out += ["## Contributo del dato reale: (i) - (i) senza FaMoS", "",
            "Catena preregistrata: ARTEFATTO DELLA GRIGLIA di lambda (vedi le note e la sezione POST HOC, dove si "
            "ricalcola con la catena corretta).", "", "| bersaglio | braccio | SR | FR |", "|---|---|---|---|"]
    for t in PRIMARY:
        for a in diag.ARMS:
            if (t, a, "i", "sr") in fam:
                out.append(f"| {t} | {SHORT[a]} | {rd_cell(fam[(t, a, 'i', 'sr')])} | {rd_cell(fam[(t, a, 'i', 'fr')])} |")
    out += ["", "## Invarianza contro pesatura (SR)", "",
            "Righe incrociate (topologie diverse), original-original e mediate sulle 5 etichette: rho di d_P e delta delle "
            "teste. Lettura corretta nelle note: conta avere la stessa discretizzazione ai due lati (tabella delle righe con "
            "la stessa etichetta nella sezione POST HOC), non la original ne' la riduzione del rumore.", "", "| bersaglio | braccio | d_P incr. / orig. / mediate | (i) delta incr. / orig. / mediate | "
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
            "testa (ii); media [minimo, massimo] sui sottoinsiemi (k = 7: le teste di R_LOGO). Catena preregistrata: le "
            "righe piatte a 0 (s2345, e s1234 su HIFI3D) sono un ARTEFATTO DELLA GRIGLIA (r = 0 scelto); la curva con la "
            "catena corretta e' nella sezione POST HOC.", "",
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
