#!/usr/bin/env python3
"""Modelli massivi: riconoscimento su FaMoS TEST con la catena di ``aau/esterna_famos_id/ei.py`` (importata, invariata).

    aau/run.sh v3_work/trainer/eval_v3.py -- aau/massive_eval/me_famos_id.py embed --model massive_main  (GPU)
    v3_work/unified_gt/run.sh aau/massive_eval/me_famos_id.py ours
    v3_work/unified_gt/run.sh aau/massive_eval/me_famos_id.py eval
    (me.sbatch, passi famos_id_embed e famos_id_eval)

Qui solo i nomi: ``ei.CKPT`` / ``ei.EMB_REF`` coi checkpoint finali EMA dei due modelli (quelli della calibrazione) e i loro
latenti della matrice (passo famos di eval_body.sh, controllo dell'embedding sulle 15 scansioni di galleria), ``ei.EVID``
-> ``aau/runs/evidence/massive_eval/famos_id`` (``selection.json`` e ``prep_eval.json`` copiati dall'evidenza di ei,
servono a ``ei.write_results``). Stesse 150 scansioni, stesse patch e operatori, stesse coppie, stesse 1000 repliche
(seme 1234). ``ours``: ``fact_paired.distances`` con la c di ``massive_eval/calib/<modello>.json`` (fissata prima).
``eval``: ``ei.evaluate`` con i due modelli come "nostri" e, fra i concorrenti, le tre letture del modello del paper
(factorized s1234, distanze gia' su disco) accanto a quelli di ei: i delta appaiati massivo - paper e massivo - ICP mm
escono da ``deltas.csv`` (positivo = massivo meglio). Distanze e latenti in ``datasets/FAMOS/esterna_id`` (fuori da git).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
for _p in (REPO / "aau/esterna_famos_id", REPO / "v3_work/trainer/tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ei  # noqa: E402  (sola lettura)

EV = REPO / "aau/runs/evidence/massive_eval"
EVID0 = ei.EVID
# modello -> (chiave di me_calib, tag dei latenti di FaMoS TEST della matrice, etichetta)
MASSIVE = {"massive_main": ("main", "v3factorizedmassmaine330full", "massivo principale (330k passi)"),
           "massive_open": ("open", "v3factorizedmassopene230full", "massivo aperto (230k passi)")}
PAPER = "factorized_s1234"
READ = (("form", "d_F non cal."), ("form_cal", "d_F cal."), ("shape", "d_P"))


def calib(m: str) -> dict:
    return json.loads((EV / "calib" / f"{MASSIVE[m][0]}.json").read_text())


def setup() -> None:
    ei.CKPT = {PAPER: ei.CKPT[PAPER], **{m: REPO / calib(m)["checkpoint"] for m in MASSIVE}}
    ei.EMB_REF = {**ei.EMB_REF, **{m: t for m, (_, t, _) in MASSIVE.items()}}
    ei.OURS = [(f"{m}|{r}", f"{lab}, {rl}", "nostro", True) for m, (_, _, lab) in MASSIVE.items() for r, rl in READ]
    ei.COMPETITORS = [(f"{PAPER}|{r}", f"paper s1234, {rl}", "nostro (paper)", False) for r, rl in READ] + \
        list(ei.COMPETITORS)
    ei.COLUMNS = ei.OURS + ei.COMPETITORS
    ei.EVID = EV / "famos_id"
    ei.EVID.mkdir(parents=True, exist_ok=True)
    for f in ("selection.json", "prep_eval.json"):
        if not (ei.EVID / f).exists():
            shutil.copy2(EVID0 / f, ei.EVID / f)


def ours(s: str) -> None:
    """``ei.ours`` per i due modelli, con la c della loro calibrazione."""
    import numpy as np
    import fact_paired as fp
    it = ei.items(s)
    iu, ju = ei.pairs(len(it))
    for m in MASSIVE:
        cb = calib(m)
        ck = ei.CKPT[m]
        with np.load(ei.DATA / s / "emb" / f"{m}.npz") as z:
            if list(z["names"].astype(str)) != [p.stem for _, _, p in it]:
                raise SystemExit(f"{m}: latenti con mesh diverse")
            Z = np.asarray(z["Z"], np.float64)
        d = fp.distances(Z, iu, ju, ck, (cb["c_median"], cb["c_ls"]))
        ei.save_dist(s, m, {k: d[k] for k in ("form", "form_cal", "shape")},
                     {"c": cb["c_median"], "checkpoint": str(ck.relative_to(REPO)),
                      "calibration": str((EV / "calib" / f"{MASSIVE[m][0]}.json").relative_to(REPO))})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("embed", "ours", "eval"))
    p.add_argument("--set", default="eval", choices=ei.SETS)
    p.add_argument("--model", default="", choices=("",) + tuple(MASSIVE))
    a = p.parse_args()
    setup()
    if a.step == "embed":
        ei.embed(argparse.Namespace(set=a.set, model=a.model, checkpoint=str(ei.CKPT[a.model])))
    elif a.step == "ours":
        ours(a.set)
    else:
        ei.evaluate(a.set)


if __name__ == "__main__":
    main()
