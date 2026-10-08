#!/usr/bin/env python3
"""Split per persona di FaMoS, dichiarato PRIMA di qualunque uso dei dati (solo nomi di cartelle).

    python3 aau/famos/famos_split.py --reg-listing <7z l -slt delle registrazioni, tsv size\\tpath> \\
        --test-scans external_data/famos/extracted/test_scans/scans --out aau/famos/split.json

Regola (PLAN_MASSIVE sez. 2 e 14.5):
  - TEST  = i soggetti delle scansioni di test di TEMPEH (cartelle ``FaMoS_subject_NNN`` di
    ``test_scans/scans``). Se fossero piu' di 20 se ne estrarrebbero 20 col seme ``--seed``;
    sono 15, quindi tutti, senza estrazione;
  - TRAIN = tutti gli altri soggetti con registrazioni.
Controlli: TEST interamente coperto dalle registrazioni, TRAIN e TEST disgiunti, TRAIN + TEST =
tutti i soggetti registrati. Solo libreria standard: gira anche sul frontend. Il file contiene solo
identificativi di soggetto, nessun dato FaMoS.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

SUBJ_RE = re.compile(r"^FaMoS_subject_\d{3}$")
MAX_TEST = 20


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reg-listing", type=Path, required=True)
    p.add_argument("--test-scans", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()

    reg = set()
    for line in a.reg_listing.read_text().splitlines():
        parts = line.split("\t")[-1].split("/")
        if len(parts) >= 2 and SUBJ_RE.match(parts[1]):
            reg.add(parts[1])
    test_all = sorted(d.name for d in a.test_scans.iterdir() if d.is_dir() and SUBJ_RE.match(d.name))
    if not reg or not test_all:
        raise SystemExit(f"soggetti vuoti: registrazioni {len(reg)}, scansioni di test {len(test_all)}")
    missing = sorted(set(test_all) - reg)
    if missing:
        raise SystemExit(f"soggetti di test senza registrazioni: {missing}")
    if len(test_all) > MAX_TEST:
        test = sorted(random.Random(a.seed).sample(test_all, MAX_TEST))
        rule = f"{MAX_TEST} di {len(test_all)} estratti con random.Random({a.seed}).sample"
    else:
        test = test_all
        rule = f"tutti i {len(test_all)} soggetti delle scansioni di test (<= {MAX_TEST}: nessuna estrazione)"
    train = sorted(reg - set(test))
    assert not set(train) & set(test), "TRAIN e TEST non disgiunti"
    assert set(train) | set(test) == reg, "TRAIN + TEST diverso dai soggetti registrati"
    out = {
        "declared_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "split per persona. TEST = soggetti delle scansioni di test di TEMPEH (" + rule + "); "
                "TRAIN = tutti gli altri soggetti registrati. Le persone di TEST non entrano MAI nei dati di training.",
        "n_registered": len(reg), "n_test_scan_subjects": len(test_all),
        "test": test, "train": train,
        "test_scan_subjects_not_in_test": sorted(set(test_all) - set(test)),
        "source": {"reg_listing": str(a.reg_listing), "test_scans": str(a.test_scans)},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[famos-split] registrati {len(reg)}, TEST {len(test)} ({test[0]}..{test[-1]}), TRAIN {len(train)} -> {a.out}")


if __name__ == "__main__":
    main()
