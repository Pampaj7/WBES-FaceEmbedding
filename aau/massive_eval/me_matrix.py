#!/usr/bin/env python3
"""Modelli massivi: la matrice di sviluppo FR / SR con ``aau/full_matrix/fm_eval.py`` (importato, invariato).

    v3_work/unified_gt/run.sh aau/massive_eval/me_matrix.py --workers 64     (me.sbatch, passo matrix; LEGGE LE GT)

Qui solo la registrazione dei due modelli come la riga "10x identities" (C3M): ``fact_paired.MODELS`` (colonne
``massive_main|form``, ``|form_cal``, ``|shape`` e ``massive_open|...``: embedding del passo form / fvn / famos di
eval_body.sh sotto i tag ``factorizedmassmain`` e ``factorizedmassopen``), ``fact_paired.calibration`` (c di
``massive_eval/calib/<modello>.json``, fissata prima), ``fm_eval.METHODS`` (nostri; d_F cal. post hoc come C3M). Il
resto e' ``fm_eval.main`` (stesse righe, maschera comune, repliche e semi della matrice completa; uscite in
``massive_eval/matrix``; ``blmm.METRICS`` senza il passo ``nicp_corr``, aggiunto dopo la matrice completa, che su
FaMoS azzererebbe ``cs_nicp_p2tri``). In piu', da ogni chiamata di ``fm_eval.records`` e sulle STESSE repliche, i delta appaiati
``tipo`` = ``massivo``: massivo - modello del paper (factorized s1234, stessa lettura) e massivo - ICP + Chamfer in mm
(FR: d_F cal. e d_F non cal.; SR: d_P). Le colonne della matrice completa devono uscire identiche (controllo in
``me_results.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
sys.path.insert(0, str(REPO / "aau/full_matrix"))

import fm_eval as fm  # noqa: E402  (sola lettura)

fp = fm.fp
EV = REPO / "aau/runs/evidence/massive_eval"
# colonna -> (chiave di me_calib, variante del tag di eval_body.sh, epoca, etichetta)
MASSIVE = {"massive_main": ("main", "factorizedmassmain", "330", "massivo principale"),
           "massive_open": ("open", "factorizedmassopen", "230", "massivo aperto")}
PAPER = "factorized_s1234"
ICP = "mm_rigid_icp_chamfer"
PAIRS = {"fr": [(f"{m}|{r}", b) for m in MASSIVE for r in ("form_cal", "form") for b in (f"{PAPER}|{r}", ICP)],
         "sr": [(f"{m}|shape", b) for m in MASSIVE for b in (f"{PAPER}|shape", ICP)]}


def register() -> None:
    # i passi di blmm come quando e' stata calcolata la matrice completa: ``nicp_corr`` (commit 21bfa5e, dopo) contiene
    # ``nicp_p2tri`` e in ``fact_paired.famos_frame`` (nessuno shard nicp_corr su FaMoS) azzera ``cs_nicp_p2tri``
    fm.blmm.METRICS = {k: v for k, v in fm.blmm.METRICS.items() if k != "nicp_corr"}
    cal = {m: json.loads((EV / "calib" / f"{k}.json").read_text()) for m, (k, _, _, _) in MASSIVE.items()}
    fp.MODELS += [(m, v, e) for m, (_, v, e, _) in MASSIVE.items()]
    _calibration = fp.calibration

    def calibration() -> dict:
        return {**_calibration(), **{m: (c["c_median"], c["c_ls"]) for m, c in cal.items()}}

    fp.calibration = calibration
    for m, (_, _, _, lab) in MASSIVE.items():
        fm.METHODS[f"{m}|form_cal"] = (f"{lab}, d_F cal.", "nostri", fm.FR, True)
        fm.METHODS[f"{m}|form"] = (f"{lab}, d_F non cal.", "nostri", fm.FR, True)
        fm.METHODS[f"{m}|shape"] = (f"{lab}, d_P", "nostri", fm.SR, True)
    patch_records(fm)


def patch_records(fm) -> None:
    """``fm_eval.records`` + i delta ``massivo`` di ``PAIRS`` sulle stesse repliche (anche per ``me_ava.py``)."""
    _records = fm.records

    def records(view, V, gkeys, groups, methods, meta, seed, n_subj, tag=""):
        S, Dl = _records(view, V, gkeys, groups, methods, meta, seed, n_subj, tag)
        mi = {m: k for k, m in enumerate(methods)}
        for gi, g in enumerate(gkeys):
            for ti, gt in enumerate(fm.GTS):
                for a, b in PAIRS[gt]:
                    if a not in mi or b not in mi:
                        continue
                    xa, xb = V[:, gi, ti, mi[a]], V[:, gi, ti, mi[b]]
                    d = xa - xb
                    if not np.isfinite(d[0]):
                        continue
                    lo, hi = fm.ci(d)
                    bb = d[1:][np.isfinite(d[1:])]
                    Dl.append({"dominio": view, "gt": gt, "insieme_righe": g[0], "perturbazione": g[1],
                               "n_righe": len(groups[g]), "n_soggetti": n_subj, "seme": seed, "passo": tag or "comune",
                               "tipo": "massivo", "confronto": f"{meta[a][0]} - {meta[b][0]}", "ruolo": "-",
                               "piu_allineato": "-", "metodo_a": meta[a][0], "colonna_a": a,
                               "post_hoc_a": fm.post_hoc(a, view), "metodo_b": meta[b][0], "colonna_b": b,
                               "post_hoc_b": fm.post_hoc(b, view), "rho_a": xa[0], "rho_b": xb[0], "delta": d[0],
                               "ic_lo": lo, "ic_hi": hi, "p_le0": float((bb <= 0).mean()) if len(bb) else np.nan})
        return S, Dl

    fm.records = records


def main() -> None:
    register()
    if "--out" not in sys.argv:
        sys.argv += ["--out", str(EV / "matrix")]
    fm.main()


if __name__ == "__main__":
    main()
