#!/usr/bin/env python3
"""Statismo HDF5 (BFM 2019) -> npz, una volta per file: il venv del repo non ha h5py.

    singularity exec /home/container/tensorflow/tensorflow_24.10.sif \\
        python3 -I v3_work/mm/bfm2019_convert.py external_data/bfm2019/model2019_bfm.h5 [...]

Il container TensorFlow ha h5py 3.10 e numpy; quello PyTorch dei job (aau/env.sh) no, e non si
installa niente nel venv condiviso. Scrive ``<nome>.npz`` accanto all'``.h5`` (``external_data/``
e' ignorata da git: e' un derivato del modello, licenza non commerciale, mai nel repo). Ogni
dataset diventa una chiave col percorso HDF5 e ``/`` -> ``__`` (``shape__model__mean``); le
stringhe restano stringhe. ``v3_work/mm/loaders.load_bfm2019`` legge solo l'npz.
"""

import sys
from pathlib import Path

import h5py
import numpy as np


def convert(path: Path) -> Path:
    out = {}
    with h5py.File(path, "r") as h:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                v = obj[()]
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="replace")
                v = np.asarray(v)
                if v.dtype.kind in "SO":     # stringhe di h5py: dtype con metadati, che np.load rifiuta
                    v = np.asarray([x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x)
                                    for x in v.ravel()]).reshape(v.shape)
                out[name.replace("/", "__")] = np.asarray(v, dtype=np.dtype(v.dtype.str))
        h.visititems(visit)
    dst = path.with_suffix(".npz")
    np.savez(dst, **out)
    print(f"{path.name}: {len(out)} dataset -> {dst}")
    for k in sorted(out):
        v = out[k]
        print(f"  {k}: {v.dtype} {v.shape}" + (f" = {str(v)[:120]!r}" if v.dtype.kind in "US" else ""))
    return dst


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        convert(Path(arg))
