#!/usr/bin/env python3
"""Le 4 celle di sviluppo mancanti della matrice completa, dalla STESSA ``fm_eval.py`` rilanciata su FaceVerse neutra e
FaMoS con le colonne che mancavano (lettura dei CSV, nessun calcolo).

    FM_STEP=eval FM_ARGS="--views faceverse_neutral,famos --out aau/runs/evidence/celle_mancanti/fm" \
        sbatch --cpus-per-task=64 --mem=192G --time=02:00:00 aau/full_matrix/fm.sbatch
    python3 aau/full_matrix/fm_celle.py                     (results.md e metrics.csv in celle_mancanti/)

Colonne nuove: embedding di C3M e123 sulla FaceVerse neutra (``eval_body.sh`` passo ``fvn``, epoca 123), template cs
della neutra (``blmm_template.py --views faceverse_neutral --modes cs``) e di FaMoS (``fm_compute.py famos-template
--mode cs``). Celle: riga "10x identities" = C3M e123 (``paper/tables/cvpr_supp_*.tex``), d_F cal. con FR e d_P con SR;
"NICP to a template, size-normalized" = ``cs_nicp_template`` con SR; FaceVerse neutra e FaMoS; insieme ``nocrop``,
passo ``comune``. Controllo: ogni riga di ``spearman.csv`` e ``deltas.csv`` della rilettura che esiste anche nella
matrice completa (stesse chiavi) deve avere gli stessi n_righe e lo stesso punto e IC: la maschera, le repliche e i semi
non sono cambiati, quindi le celle nuove sono confrontabili con le altre della stessa colonna.
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
FULL = EV / "full_matrix"
OUT = EV / "celle_mancanti"
NEW = OUT / "fm"
CELLS = [("faceverse_neutral", "fr", "factorizedc3m_e123|form_cal", "10x identities, d_F cal."),
         ("faceverse_neutral", "sr", "factorizedc3m_e123|shape", "10x identities, d_P"),
         ("faceverse_neutral", "sr", "cs_nicp_template", "NICP to a template, size-normalized"),
         ("famos", "sr", "cs_nicp_template", "NICP to a template, size-normalized")]
KEY = ["dominio", "gt", "insieme_righe", "perturbazione", "passo", "colonna"]
DKEY = ["dominio", "gt", "insieme_righe", "perturbazione", "passo", "tipo", "confronto", "colonna_a", "colonna_b"]


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def control(new: pd.DataFrame, old: pd.DataFrame, key: list, vals: tuple) -> dict:
    """Righe con le stesse chiavi: scarto massimo dei valori e righe con n_righe diverso."""
    old = old[old["dominio"].isin(set(new["dominio"]))]
    m = new.merge(old, on=key, suffixes=("", "_full"))
    gone = old.merge(new[key], on=key, how="left", indicator=True)
    gone = gone[gone["_merge"] == "left_only"]
    out = {"rows_new": int(len(new)), "rows_full_same_domains": int(len(old)), "rows_matched": int(len(m)),
           "rows_full_not_in_new": int(len(old) - len(m)),
           "full_not_in_new": sorted({" ".join(str(r[k]) for k in ("dominio", "gt", "colonna", "colonna_a", "colonna_b")
                                     if k in r) for _, r in gone.iterrows()}),
           "n_righe_diverse": int((m["n_righe"] != m["n_righe_full"]).sum())}
    for v in vals:
        out[f"max_abs_diff_{v}"] = float(np.nanmax(np.abs(m[v].to_numpy(float) - m[f"{v}_full"].to_numpy(float))))
    return out


def main() -> None:
    S = pd.read_csv(NEW / "spearman.csv", dtype={"perturbazione": str})
    D = pd.read_csv(NEW / "deltas.csv", dtype={"perturbazione": str})
    S0 = pd.read_csv(FULL / "spearman.csv", dtype={"perturbazione": str})
    D0 = pd.read_csv(FULL / "deltas.csv", dtype={"perturbazione": str})
    best = "migliore_vs_migliore"                # scelto sul punto: con una colonna nuova puo' cambiare per costruzione
    ctrl = {"spearman": control(S, S0, KEY, ("rho", "ic_lo", "ic_hi")),
            "deltas": control(D[D["tipo"] != best], D0[D0["tipo"] != best], DKEY, ("delta", "ic_lo", "ic_hi"))}
    rows = []
    for dom, gt, col, lab in CELLS:
        hit = S[(S["dominio"] == dom) & (S["gt"] == gt) & (S["colonna"] == col) & (S["insieme_righe"] == "nocrop") &
                (S["passo"] == "comune")]
        if len(hit) != 1:
            raise SystemExit(f"{dom} {gt} {col}: {len(hit)} righe nocrop / comune")
        r = hit.iloc[0]
        old = S0[(S0["dominio"] == dom) & (S0["gt"] == gt) & (S0["colonna"] == col)]
        rows.append({"riga_paper": lab, "dominio": dom, "gt": gt, "colonna": col, "metodo": r["metodo"],
                     "rho": r["rho"], "ic_lo": r["ic_lo"], "ic_hi": r["ic_hi"], "n_righe": int(r["n_righe"]),
                     "n_soggetti": int(r["n_soggetti"]), "seme": int(r["seme"]), "post_hoc": bool(r["post_hoc"]),
                     "gia_nella_matrice": bool(len(old)),
                     "fonte": f"{rel(NEW / 'spearman.csv')}:{int(hit.index[0]) + 2}"})
    with open(OUT / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "controls.json").write_text(json.dumps(ctrl, indent=1) + "\n")
    f3 = lambda r: f"{r['rho']:.3f} [{r['ic_lo']:.3f}, {r['ic_hi']:.3f}]"      # noqa: E731
    L = ["# Celle di sviluppo mancanti (matrice completa, solo dev, niente Ava-256)", "",
         "Generato da `aau/full_matrix/fm_celle.py` dai CSV di `fm/` (= `fm_eval.py` rilanciato su FaceVerse neutra e "
         "FaMoS con le colonne che mancavano; uscite complete in `fm/`). Insieme `nocrop`, passo `comune`, IC 95% "
         "percentile, 1000 repliche per soggetto col seme della matrice (FaceVerse neutra 271049; FaMoS "
         "`famos_eval.bootstrap_counts(15, n, 1234)`).", "",
         "| riga del paper | colonna | dominio | GT | Spearman [IC 95%] | righe | soggetti | fonte |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['riga_paper']}{' (post hoc)' if r['post_hoc'] else ''} | `{r['colonna']}` | {r['dominio']} | "
                 f"{r['gt'].upper()} | {f3(r)} | {r['n_righe']} | {r['n_soggetti']} | `{r['fonte']}` |")
    c = ctrl["spearman"]
    L += ["", "## Calcolo delle colonne nuove", "",
          "- C3M e123 sulla FaceVerse neutra: `v3_work/trainer/ablations/c3f/eval_body.sh`, passo `fvn`, checkpoint "
          "`epoch123_ema.pth` di `trainer_v3/factorized/c3m/runs` (come e205), d_F cal. con la c di "
          "`factorized_calibration.csv` (`factorizedc3m_e123`, la stessa delle altre colonne di C3M e123).",
          "- NICP su template cs, FaceVerse neutra: `aau/baselines_mm/blmm_template.py --views faceverse_neutral "
          "--modes cs` (definizione di `blmm.build_template` / `blmm.enroll`, come le altre viste).",
          "- NICP su template cs, FaMoS: `aau/full_matrix/fm_compute.py famos-template --mode cs` (lo stesso template "
          "delle 80 persone FaMoS TRAIN del modo mm, nelle coordinate cs di `blmm.work_coords`); post hoc come il "
          "template mm di FaMoS (costruito qui).", "",
          "## Controllo: le altre celle non cambiano", "",
          f"Righe di `spearman.csv` con le stesse chiavi nella matrice completa: {c['rows_matched']} su "
          f"{c['rows_full_same_domains']} delle due viste; n_righe diversi {c['n_righe_diverse']}; scarto massimo "
          f"punto {c['max_abs_diff_rho']:.2e}, IC {max(c['max_abs_diff_ic_lo'], c['max_abs_diff_ic_hi']):.2e}. "
          f"`deltas.csv` (senza `migliore_vs_migliore`, scelto sul punto): {ctrl['deltas']['rows_matched']} righe "
          "confrontate, scarto massimo "
          f"{max(ctrl['deltas'][k] for k in ('max_abs_diff_delta', 'max_abs_diff_ic_lo', 'max_abs_diff_ic_hi')):.2e}. "
          "Dettagli in `controls.json`.", "",
          f"Righe della matrice completa assenti dalla rilettura: {', '.join(c['full_not_in_new']) or 'nessuna'}. "
          "Causa (non legata alle colonne nuove): dal commit 21bfa5e (15 ottobre, dopo la matrice completa) "
          "`blmm.METRICS` ha il passo `nicp_corr`, che contiene `nicp_p2tri`; `fact_paired.famos_frame` scorre i passi e "
          "per `nicp_corr` (nessuno shard su FaMoS) sovrascrive `cs_nicp_p2tri` con una matrice tutta NaN, che "
          "`fm_eval` dichiara non applicabile. La maschera di FaMoS resta 105 righe (la colonna esce intera, non le "
          "righe), quindi le altre celle, comprese le 4 nuove, non cambiano.", ""]
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
