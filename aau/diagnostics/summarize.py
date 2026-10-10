#!/usr/bin/env python3
"""Diagnostica D: ``aau/runs/evidence/diagnostics/results.md`` dalle uscite di d1_gen, d1_bp, d1_stats e d2_probe.

    AAU_NV="" aau/run.sh aau/diagnostics/summarize.py          (diag.sbatch, passo summary)

Tabelle: rho per insieme (SR e FR) dei bracci e di B, letture R1 / R2 / controllo di difficolta' / C1, sonda di D2
(R3), controlli, costi dei fit B. Solo presentazione: nessun numero nuovo.
"""
from __future__ import annotations

import csv
import json

import numpy as np

import diag

SET_ORDER = ("bfm", "ict", "gnm", "regen_ict", "regen_gnm", "flame2023_s1", "flame2023")
SHORT = {"factorized_s1234": "fact. s1234", "factorized_s2345": "fact. s2345", "factorizedc3m_e123": "C3M e123"}


def f3(x) -> str:
    return "-" if x is None or not np.isfinite(float(x)) else f"{float(x):.3f}"


def cell(p, lo, hi, sign=False) -> str:
    fmt = "{:+.3f}" if sign else "{:.3f}"
    return f"{fmt.format(float(p))} [{fmt.format(float(lo))}, {fmt.format(float(hi))}]"


def d1_tables(out: list) -> None:
    rho = list(csv.DictReader(open(diag.EV / "d1_spearman.csv")))
    get = {(r["set"], r["gt"], r["method"]): r for r in rho}
    for g, dist in (("sr", "shape"), ("fr", "form_cal")):
        out += [f"### D1, GT {g.upper()} ({'d_P' if g == 'sr' else 'd_F calibrata'} dei bracci, B `vb_{g}`)", "",
                "| insieme | " + " | ".join(SHORT[a] for a in diag.ARMS) + " | B-FLAME | B-GNM | righe |",
                "|---" * (len(diag.ARMS) + 4) + "|"]
        for s in SET_ORDER:
            cells = []
            for a in diag.ARMS:
                r = get.get((s, g, f"{a}|{dist}"))
                cells.append(cell(r["rho"], r["ci_low"], r["ci_high"]) if r else "-")
            for m in diag.BP_MODELS:
                r = next((x for k, x in get.items() if k[0] == s and k[1] == g and k[2].startswith(f"B-{m}|vb_{g}")),
                         None)
                tag = "" if r is None else (" (esatto)" if "esatto" in r["method"] else "")
                cells.append(cell(r["rho"], r["ci_low"], r["ci_high"]) + tag if r else "-")
            n = next((x["n_rows"] for k, x in get.items() if k[0] == s), "-")
            out.append(f"| {s} | " + " | ".join(cells) + f" | {n} |")
        out.append("")


def d1_deltas(out: list) -> None:
    rows = list(csv.DictReader(open(diag.EV / "d1_delta.csv")))
    out += ["### D1, delta (stesse righe e repliche dentro un insieme; insiemi diversi indipendenti)", "",
            "| tipo | braccio | GT | confronto | delta [IC 95%] | P(<= 0) |", "|---|---|---|---|---|---|"]
    order = {"gen": 0, "read": 1, "difficulty": 2, "resolution": 3, "within": 4, "C1": 5}
    for r in sorted(rows, key=lambda r: (order[r["kind"]], r["gt"], r["arm"])):
        out.append(f"| {r['kind']} | {SHORT.get(r['arm'], r['arm'])} | {r['gt']} | {r['what']} | "
                   f"{cell(r['delta'], r['ci_low'], r['ci_high'], True)} | {float(r['p_le0']):.3f} |")
    out.append("")


def d2_table(out: list) -> None:
    rows = list(csv.DictReader(open(diag.EV / "d2_probe.csv")))
    out += ["### D2, sonda lineare (diagnostica, non un metodo)", "",
            "| dominio | braccio | GT | sonda [IC] | riferimento [IC] | delta [IC protocollo] | IC sola valutazione | "
            "P(<= 0) | permutazione (min, max) | lambda mediano (bordo) | R3 |", "|---" * 11 + "|"]
    for r in rows:
        r3 = {"True": "passa", "False": "no"}.get(r["r3_pass"], "-")
        out.append(f"| {r['domain']} | {SHORT[r['arm']]} | {r['gt']} ({r['reference']}) | "
                   f"{cell(r['probe'], r['probe_ci_low'], r['probe_ci_high'])} | "
                   f"{cell(r['ref'], r['ref_ci_low'], r['ref_ci_high'])} | "
                   f"{cell(r['delta'], r['delta_ci_low'], r['delta_ci_high'], True)} | "
                   f"[{float(r['delta_evalci_low']):+.3f}, {float(r['delta_evalci_high']):+.3f}] | "
                   f"{float(r['p_le0']):.3f} | {float(r['perm_mean']):+.3f} ({float(r['perm_min']):+.3f}, "
                   f"{float(r['perm_max']):+.3f}) | {float(r['lambda_median']):g} ({float(r['lambda_edge_frac']):.2f}) "
                   f"| {r3} |")
    out.append("")


def main() -> None:
    out = ["# Diagnostica D: risultati", "",
           "Protocollo `PROTOCOL_D.md` (sha256 b9f5378a..., commit 251bdd6) ed emendamento 1. Generato da "
           "`aau/diagnostics/summarize.py`; numeri in `d1_spearman.csv`, `d1_delta.csv`, `d2_probe.csv`.", ""]
    rd = json.loads((diag.D1 / "readings.json").read_text())
    d2c = json.loads((diag.EV / "d2_controls.json").read_text())
    out += ["## Letture", "",
            f"- C1 (held-out statici compatibili col codice di D1): **{'si' if rd['verdict']['C1_compatible'] else 'NO'}**",
            f"- R1 (varieta' leva forte, FLAME suddiviso): **{rd['verdict']['R1_variety_strong']}**; "
            f"con FLAME nativo: {rd['verdict']['R1_native_variety_strong']}",
            f"- R2 (limite di lettura sui visti, contro B-FLAME): **{rd['verdict']['R2_reading_limit']}**",
            f"- R3 (informazione nell'embedding, FaceScape SR): **{d2c['R3_information_in_embedding']}**", "",
            "| braccio | Delta_gen (suddiviso) | Delta_gen (nativo) | Delta_read | DiD difficolta' |", "|---|---|---|---|---|"]
    for a in diag.ARMS:
        x = [rd["R1"][a], rd["R1_native"][a], rd["R2"][a], rd["difficulty"][a]]
        out.append(f"| {SHORT[a]} | " + " | ".join(cell(r["delta"], r["ci_low"], r["ci_high"], True) for r in x) + " |")
    out += ["", "## D1", ""]
    d1_tables(out)
    d1_deltas(out)
    out += ["## D2", ""]
    d2_table(out)
    ctrl = json.loads((diag.D1 / "controls.json").read_text())
    gen = json.loads((diag.D1 / "gen.json").read_text())
    out += ["## Controlli", "", f"- C1-GT: {json.dumps(ctrl['c1_gt'])}"]
    for s in ("regen_ict", "regen_gnm"):
        out.append(f"- C1-emb {s}: " + "; ".join(
            f"{SHORT[a]} " + ", ".join(f"{k} {v:.1e}" for k, v in m.items()) +
            f" (scala {ctrl['sets'][s]['c1_emb_scale'][a]:.2f})" for a, m in ctrl["sets"][s]["c1_emb_max_abs"].items()))
    for s in SET_ORDER:
        c = ctrl["sets"][s]
        out.append(f"- {s}: {c['n_subjects']} soggetti, {c['n_rows']} righe, {c['n_rows_masked_out']} fuori maschera; "
                   f"d_P GT fra soggetti mediana {c['dP_gt_subject_pairs']['median']:.4f}, IQR "
                   f"[{c['dP_gt_subject_pairs']['iqr'][0]:.4f}, {c['dP_gt_subject_pairs']['iqr'][1]:.4f}]")
    for v, c in d2c["views"].items():
        out.append(f"- D2 {v}: {c['n_rows']} righe, K2 {json.dumps(c['K2'])}, K3 max |rho - pubblicato| "
                   f"{max(x['abs_diff'] for x in c['K3'].values()):.1e}")
    out.append(f"- generazione: {json.dumps({s: v.get('verts_by_label') for s, v in gen['sets'].items()})}")
    for p in sorted((diag.D1 / "bp").glob("flame2023*_*.npz")):
        with np.load(p, allow_pickle=True) as z:
            out.append(f"- B {p.stem}: fallite {len(z['failed_vb'])}, regione {json.loads(str(z['region_diag']))}, "
                       f"{float(z['wall_s']):.0f} s ({int(z['workers'])} processi), s/mesh mediana per passo "
                       f"{dict(zip([str(x) for x in z['stages']], np.round(np.nanmedian(z['t_stage'], 0), 2).tolist()))}")
    (diag.EV / "results.md").write_text("\n".join(out) + "\n")
    print(f"[summary] {diag.EV / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
