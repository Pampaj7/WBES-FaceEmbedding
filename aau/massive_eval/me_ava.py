#!/usr/bin/env python3
"""Modelli massivi su Ava-256: SOLO DESCRITTIVO, fuori dalla confermativa, con la catena dei descrittivi della prova
(``aau/ava256/conf_embed.py`` e ``conf_desc.py``, importati e invariati), senza toccare la prova congelata.

    python3 aau/massive_eval/me_ava.py prep                                               (spec, cartelle, link)
    AAU_NV=--nv aau/run.sh aau/ava256/conf_embed.py run --spec <spec> --set eval_view     (A100, come C3M e123)
    v3_work/unified_gt/run.sh aau/massive_eval/me_ava.py eval --workers 64                (LEGGE LE GT di Ava-256)
    (me.sbatch, passi ava_embed e ava_eval)

``prep``: lo spec ``aau/runs/evidence/massive_eval/ava256/spec.json`` = ``aau/ava256/spec/ava256.json`` (soggetti di
valutazione, 6 topologie, GT, seme delle repliche, colonne della prova: tutto invariato) con: ``freeze`` false (non e' la
prova: nessun ``verify``); ``columns.desc_models`` = C3M e123 (invariato, controllo di riproduzione) + i due modelli
massivi (checkpoint finale EMA e sha256 di ``massive_eval/calib/<modello>.json``, c di quella calibrazione, fissata
prima) + ``c3m_e123_a10`` (C3M e123 rifatto qui: controllo della classe di GPU); ``embed.sets`` = solo ``eval_view``
con questi tre; ``embed.gpu`` = ``A10`` (le A100 della prova sono occupate da job di giorni: deviazione dichiarata,
misurata dal controllo: C3M e123 su A10 contro le sue colonne su A100); ``embed.ops_root`` e ``out.work`` / ``out.evidence``
NUOVI: ``datasets/V3_OPS_CACHE/massive_eval_ava`` (link agli operatori della prova e a ``names.npz``, nessun ricalcolo),
``aau/runs/massive_eval/ava256`` (link in sola lettura a ``baselines_param``, ``blmm``, ``desc``, ``head``,
``protocolli`` della prova; ``embed`` nuovo) e ``aau/runs/evidence/massive_eval/ava256``. Nessuna scrittura sotto
``aau/runs/ava256_conf`` o ``aau/runs/evidence/ava256_conf`` (``eval`` scrive solo in ``<evidence>/desc``).
``eval``: ``conf_desc.step_eval`` sullo spec nuovo (righe, maschere confermativa (+inf) e mascherata, 1000 repliche col
seme dello spec, ``fm_eval.records``), con i due modelli fra i nostri di ``fm_eval.METHODS`` e i delta appaiati
``tipo`` = ``massivo`` di ``me_matrix`` (massivo - factorized s1234, massivo - ICP + Chamfer in mm). Le colonne della
prova devono uscire identiche a ``aau/runs/evidence/ava256_conf/desc/spearman.csv`` (controllo in ``me_results.py``).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
AVA = REPO / "aau/ava256"
SPEC0 = AVA / "spec/ava256.json"
EV = REPO / "aau/runs/evidence/massive_eval"
SPEC = EV / "ava256" / "spec.json"
WORK = "aau/runs/massive_eval/ava256"
OPS = "datasets/V3_OPS_CACHE/massive_eval_ava"
FROZEN_WORK = REPO / "aau/runs/ava256_conf"
LINKED = ("baselines_param", "blmm", "desc", "head", "protocolli")
MASSIVE = {"massive_main": "main", "massive_open": "open"}       # colonna -> chiave di me_calib
C3M, CTRL = "factorizedc3m_e123", "c3m_e123_a10"                  # controllo della classe di GPU
GPU = "A10"


def prep() -> None:
    s = json.loads(SPEC0.read_text())
    if not s.get("freeze") or s["out"]["work"] != str(FROZEN_WORK.relative_to(REPO)):
        raise SystemExit(f"{SPEC0}: non e' lo spec della prova congelata")
    n = copy.deepcopy(s)
    n["name"] = s["name"]                                   # stessa vista (blmm / bp): righe, scalari, params
    n["freeze"] = False
    n["note"] = ("Modelli massivi, SOLO DESCRITTIVO (aau/massive_eval/me_ava.py): spec della prova con freeze false, "
                 "desc_models = C3M e123 + i due modelli massivi, uscite nuove; prova congelata non toccata.")
    for col, k in MASSIVE.items():
        cb = json.loads((EV / "calib" / f"{k}.json").read_text())
        n["columns"]["desc_models"][col] = {"checkpoint": cb["checkpoint"], "checkpoint_sha256": cb["sha256_checkpoint"],
                                            "c": cb["c_median"],
                                            "embeddings": f"{WORK}/embed/eval_view/{col}/embeddings.npz",
                                            "calibration": str((EV / "calib" / f"{k}.json").relative_to(REPO))}
    c3m = copy.deepcopy(s["columns"]["desc_models"][C3M])
    n["columns"]["desc_models"][CTRL] = {**c3m, "embeddings": f"{WORK}/embed/eval_view/{CTRL}/embeddings.npz"}
    st = copy.deepcopy(s["embed"]["sets"]["eval_view"])
    st["arms"] = list(MASSIVE) + [CTRL]
    n["embed"]["sets"] = {"eval_view": st}
    n["embed"]["gpu"] = GPU
    n["embed"]["ops_root"] = OPS
    n["out"] = {"work": WORK, "evidence": str((EV / "ava256").relative_to(REPO))}
    work = REPO / WORK
    (work / "embed").mkdir(parents=True, exist_ok=True)
    for d in LINKED:
        if not (work / d).is_symlink():
            (work / d).symlink_to(FROZEN_WORK / d)
    src = REPO / s["embed"]["ops_root"] / "eval_view"
    dst = REPO / OPS / "eval_view"
    (dst / "ops").mkdir(parents=True, exist_ok=True)
    for p in sorted((src / "ops").glob("*.npz")):
        if not p.name.startswith(".") and not (dst / "ops" / p.name).exists():
            (dst / "ops" / p.name).symlink_to(p)
    if not (dst / "names.npz").exists():
        (dst / "names.npz").symlink_to(src / "names.npz")
    SPEC.parent.mkdir(parents=True, exist_ok=True)
    SPEC.write_text(json.dumps(n, indent=1) + "\n")
    print(f"[me-ava] spec {SPEC}; work {work} ({', '.join(LINKED)} in sola lettura dalla prova); operatori "
          f"{len(list((dst / 'ops').glob('*.npz')))} link in {dst}", flush=True)


def evaluate(workers: int) -> None:
    for _p in (AVA, REPO / "aau/full_matrix", THIS):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import conf_desc
    import fm_eval as fm
    import me_matrix
    for col, (_, _, _, lab) in me_matrix.MASSIVE.items():
        fm.METHODS[f"{col}|form_cal"] = (f"{lab}, d_F cal.", "nostri", fm.FR, True)
        fm.METHODS[f"{col}|form"] = (f"{lab}, d_F non cal.", "nostri", fm.FR, True)
        fm.METHODS[f"{col}|shape"] = (f"{lab}, d_P", "nostri", fm.SR, True)
    for k, lab in (("form_cal", "d_F cal."), ("form", "d_F non cal."), ("shape", "d_P")):
        fm.METHODS[f"{CTRL}|{k}"] = (f"controllo: C3M e123 su A10, {lab}", "nostri", fm.SR if k == "shape" else fm.FR,
                                     True)
    me_matrix.patch_records(fm)
    conf_desc.step_eval(SPEC, workers, 1000)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("prep", "eval"))
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    prep() if a.step == "prep" else evaluate(a.workers)


if __name__ == "__main__":
    main()
