#!/usr/bin/env python3
"""La tabella standard di aau/zs3dmm (zs_summarize.py, invariato) sui bracci HIFI3D di E1, con etichette giuste.

    bash aau/evidence/e1_factorial/e1_zs_summarize.sh       (da e1_summarize.sbatch)

``zs_summarize.discover_scale_arms`` etichetta ogni braccio scale_<tag> come "BFM+ICT+GNM (10^5), <tag>":
per C2M o G1 sarebbe falso. Qui le etichette delle celle sono messe PRIMA (discover usa setdefault e non le
sovrascrive), poi si chiama ``zs_summarize.main()`` con gli argomenti di zs_summarize.sbatch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "zs3dmm"))
import zs_summarize as zsum  # noqa: E402

CELL = {"c2m": "C2M, BFM+ICT", "c2f": "C2F, BFM+ICT/10", "c3f": "C3F, BFM+ICT+GNM a pari identita' di C2F",
        "g1": "G1, solo GNM"}
STEPS = {"036": 10548, "072": 21096}
for e, n in STEPS.items():
    zsum.ARM_LABEL[f"scale_e{e}"] = f"E1 C3M (run su scala 1060130), {n} passi"
    for c, lab in CELL.items():
        zsum.ARM_LABEL[f"scale_e1{c}e{e}"] = f"E1 {lab}, {n} passi"

if __name__ == "__main__":
    zsum.main()
