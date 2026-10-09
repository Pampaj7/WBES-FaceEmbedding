#!/usr/bin/env python3
"""Prova di e1_summarize.py (regola a tre esiti, domanda su C3F-UGT, differenze, controlli) sui soli dati esistenti.

    aau/run.sh aau/evidence/e1_factorial/scratch/test_summarize_alias.py

Le celle puntano a valutazioni gia' fatte (bracci HIFI3D di aau/runs/ws_hifi3d, NoW di C3M, FLAME di ws_flame),
con i checkpoint attesi mappati per la prova: C3M L40S = scale_e036/e072 (deve riprodurre curve.md ed E8),
"C3F" = scale_e036, "C2F" = scale_e072, "C2F-GNM" = congiunto 1019532, "C3F-UGT" = scale_e036. I NUMERI NON
SONO DI E1: servono solo a provare il codice. Scrive in scratch/alias_out, non tocca aau/runs/evidence/e1.
"""
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import e1_summarize as s  # noqa: E402

out = HERE / "alias_out"
shutil.rmtree(out, ignore_errors=True)
out.mkdir()
for f in ("protocol.md", "protocol.sha256", "protocol_amendment.md", "protocol_amendment.sha256", "design.json"):
    shutil.copy(s.OUT / f, out / f)
H = s.RUNS / "ws_hifi3d/data_328f2bfc1a"
alias = {"c3ml": {"036": "scale_e036", "072": "scale_e072"}, "c3f": {"036": "scale_e036", "072": "scale_e036"},
         "c2f": {"036": "scale_e072", "072": "scale_e072"}, "c2fgnm": {"036": "joint", "072": "joint"},
         "c3fugt": {"036": "scale_e036", "072": "scale_e036"}}
d = out / "hifi_runs/data_328f2bfc1a"
d.mkdir(parents=True)
ck = {}
for cell, m in alias.items():
    for e, arm in m.items():
        src = H / (arm if arm == "joint" else f"{arm}_topology")
        (d / f"scale_{s.tag(cell, e)}_topology").symlink_to(src)
        ck[(cell, e)] = s.eval_ckpt(src / "zs_zeroshot")
for cell in ("c3ml", "c3f"):
    for e in ("036", "072"):
        (out / "now").mkdir(exist_ok=True)
        (out / "now" / s.tag(cell, e)).symlink_to(s.RUNS / f"now_eval_scale_e{alias[cell][e][-3:]}")
s.expected_ckpt = lambda c, e: ck.get((c, e))
s.OUT = out
sys.argv = [sys.argv[0], "--workers", "24"]
s.main()
