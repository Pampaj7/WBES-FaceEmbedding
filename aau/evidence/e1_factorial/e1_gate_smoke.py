#!/usr/bin/env python3
"""E1, cancello dei training su V100: la loss dell'epoca 1 dello smoke V100 contro quella del run su scala su L40S.

    python3 aau/evidence/e1_factorial/e1_gate_smoke.py <run dir dello smoke>     (solo stdlib; e1_gate_smoke.sbatch)

Lo smoke (WBES_E1_SMOKE=v100) e' la configurazione di C3M (split, 46 blocchi, semi) fermata dopo l'epoca 1:
stessi soggetti, stesso ordine, stessi semi dell'epoca 1 di 1060130. Le differenze vengono solo dall'aritmetica
(V100 fp32 contro L40S, TF32 e non determinismo della GPU) mediata su 293 passi. Tolleranza dichiarata prima del
confronto: 5% relativo su loss, stress, rank e sulle quattro componenti di mixed_train_log.csv, e nessun NaN.
Scrive aau/runs/evidence/e1/smoke_v100.json ed esce 0 solo se tutto torna.
"""
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REF = REPO / "aau/runs/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411"
TOL = 0.05


def epoch1(run: Path) -> dict:
    d = next(run.glob("mixed_*"))
    out = {}
    for f, keys in (("train_log.csv", ("loss", "stress", "rank")),
                    ("mixed_train_log.csv", ("subject_stress", "subject_rank", "mesh_stress", "mesh_rank"))):
        row = next(r for r in csv.DictReader(open(d / f)) if r["epoch"] == "1")
        out.update({k: float(row[k]) for k in keys})
    return out


def main() -> None:
    run = Path(sys.argv[1]).resolve()
    ref, new = epoch1(REF), epoch1(run)
    rel = {k: abs(new[k] - ref[k]) / abs(ref[k]) for k in ref}
    ok = all(math.isfinite(v) for v in new.values()) and max(rel.values()) <= TOL
    log = (run / "train.log").read_text() if (run / "train.log").exists() else ""
    speed = [l for l in log.splitlines() if l.startswith("[steps] epoca 1:")]
    rep = {"smoke_run": str(run.relative_to(REPO)), "reference": str(REF.relative_to(REPO)), "tolerance_rel": TOL,
           "l40s_epoch1": ref, "v100_epoch1": new, "rel_diff": rel, "max_rel_diff": max(rel.values()), "ok": ok,
           "epoch1_speed_line": speed[0] if speed else None}
    (REPO / "aau/runs/evidence/e1/smoke_v100.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
