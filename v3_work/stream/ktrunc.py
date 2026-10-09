#!/usr/bin/env python
"""Ablazione k_eig 64 contro 128 senza ricostruire lo store: gli autovettori serviti troncati ai primi k.

    WBES_K_TRUNC=64 aau/run.sh v3_work/stream/ktrunc.py v3_work/trainer/train_v3.py <argomenti>
    WBES_K_TRUNC=64 aau/run.sh v3_work/stream/ktrunc.py v3_work/trainer/eval_v3.py -- <script> <argomenti>

Esegue lo script indicato (runpy, come __main__) dopo aver installato, se ``WBES_K_TRUNC`` = k > 0, la troncatura
su ogni campione servito: lo store e la cache del trainer (``data_v3._serve``) e il loader congelato degli script
di eval (``GTReadyDatasetNPZ.__getitem__``). Senza la variabile non cambia nulla (stesso percorso).

Perche' e' la pipeline con k_eig = k: eigsh con k = 64 e i primi 64 di k = 128 sono le stesse autocoppie (entro la
tolleranza di ARPACK, a meno del segno, che la diffusione spettrale non vede). Il loader normalizza per lambda_max
dei SUOI autovalori: servito = (lambda / lambda_128, grad / sqrt(lambda_128)); con k:
    evals_k = evals[:k] / evals[k-1],  evecs_k = evecs[:, :k],  grad_k = grad / sqrt(evals[k-1])
cioe' esattamente (lambda / lambda_k, grad / sqrt(lambda_k)). I pesi robusti (area_v3, passa-basso coi primi 64
autovettori) non cambiano con k >= 64.
"""
from __future__ import annotations

import math
import os
import runpy
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TRAINER = THIS_DIR.parent / "trainer"


def truncate(sample: dict, k: int) -> dict:
    ev = sample["evals"]
    if int(ev.numel()) <= k:
        return sample
    lam = float(ev.reshape(-1)[k - 1])
    if not lam > 0:
        raise ValueError(f"k_eig {k}: autovalore servito {lam} non positivo")
    out = dict(sample)
    out["evals"] = ev[:k] / lam
    out["evecs"] = sample["evecs"][:, :k]
    f = 1.0 / math.sqrt(lam)
    for key in ("gradX", "gradY"):
        out[key] = sample[key] * f
    return out


def install(k: int) -> None:
    sys.path.insert(0, str(TRAINER))
    import common  # noqa: F401  (percorsi del pacchetto congelato)
    import data_v3 as dv
    from dataset_gtready import GTReadyDatasetNPZ
    serve, getitem = dv._serve, GTReadyDatasetNPZ.__getitem__
    dv._serve = lambda sample: truncate(serve(sample), k)
    GTReadyDatasetNPZ.__getitem__ = lambda self, idx: truncate(getitem(self, idx), k)
    print(f"[ktrunc] autovettori serviti troncati ai primi {k} (store, cache e loader congelato)", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    k = int(os.environ.get("WBES_K_TRUNC", "0") or 0)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(Path(script).resolve().parent))
    if k > 0:
        install(k)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
