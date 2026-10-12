#!/usr/bin/env python3
"""Congelamento di GT e viste di Ava-256 e controllo dell'impronta (decisione del PI dell'11 ottobre, PRIMA di qualsiasi
valutazione).

    v3_work/unified_gt/run.sh aau/ava256/ava_freeze.py freeze      (ultimo passo di ava_build.sbatch)
    v3_work/unified_gt/run.sh aau/ava256/ava_freeze.py verify
    v3_work/unified_gt/run.sh aau/ava256/ava_freeze.py supp        (una volta: impronte supplementari, 12 ottobre)

``freeze``: impronte del CONTENUTO (le npz hanno le date nello zip, i file cambiano a ogni scrittura): sha256 di ``V``
float32 e ``F`` int32 di ogni mesh delle viste ``eval_view`` e ``calib_view`` (``ava_views.mesh_sha256``), sha256 di
``D_orig`` float32 e dei nomi di ogni GT (``ava_gt.content_sha256``), sha256 di nomi e ``area_mm2`` della tabella di scala.
Le scrive in ``aau/ava256/freeze_manifest.json`` (in git: il riferimento) e in ``datasets/AVA256/FROZEN.json``; poi rende
in sola lettura file E cartelle dei dati (con la cartella scrivibile, ``tmp.replace(file)`` sovrascriverebbe anche un file
in sola lettura). Da quel momento ``ava_build.sbatch``, ``ava_gt.py``, ``ava_views.py`` e ``ava_scale_table.py`` si
rifiutano di scrivere: rigenerare richiede una decisione del PI (togliere a mano ``FROZEN.json`` e la sola lettura).

``verify_frozen()``: la funzione che lo strumento di valutazione DEVE chiamare prima di leggere viste, GT o tabella di
scala di Ava-256: ricalcola le impronte e le confronta con quelle in git; un solo scarto (mesh mancante, in piu' o
diversa) -> RuntimeError.

``supp`` / ``verify_frozen_supp()`` (critic del 12 ottobre: le impronte del contenuto non coprono l'unita' dei json delle
GT ne' ``ids.json``): sha256 dei BYTE di ogni file di ``gt/`` (sottocartelle comprese: json con ``mm_per_unit``,
``ids.json``, ``split.json``, ``frame.json`` con u, R, t, centroid size, diagnostici) e dei file in cima a ``eval_view/``,
``calib_view/`` (json delle GT, etichette, manifest; ``npz/`` e' gia' coperta dalle mesh) e ``scale/``; i symlink per
bersaglio. Calcolate dai file congelati cosi' come sono, senza rigenerarli, in ``aau/ava256/freeze_manifest_supp.json``:
``freeze_manifest.json`` e ``FROZEN.json`` non si toccano. Lo strumento di valutazione chiama ``verify_frozen()`` E
``verify_frozen_supp()``; un file mancante, in piu' o diverso -> RuntimeError.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import ava_common as ac  # noqa: E402

MANIFEST = ac.SUMMARY_DIR / "freeze_manifest.json"
SUPP_MANIFEST = ac.SUMMARY_DIR / "freeze_manifest_supp.json"
SUPP_DIRS = {"gt": True, "eval_view": False, "calib_view": False, "scale": False}   # cartella: ricorsiva
VIEW_KINDS = ("eval_view", "calib_view")
GT_FILES = ("ava256_eval_fr", "ava256_eval_sr", "ava256_all_fr", "ava256_all_sr")
FROZEN_DIRS = ("raw", "corr", "neutral", "gt", "identities", "topo", "eval_view", "calib_view", "scale")
FROZEN_FILES = ("template.npz",)


def mesh_sha256(path: Path) -> str:
    """Come ``ava_views.mesh_sha256`` (copiata per non importare il generatore delle viste nello strumento di valutazione)."""
    with np.load(path) as z:
        h = hashlib.sha256(np.ascontiguousarray(z["V"], dtype="<f4").tobytes())
        h.update(np.ascontiguousarray(z["F"], dtype="<i4").tobytes())
    return h.hexdigest()


def gt_sha256(path: Path) -> str:
    """Come ``ava_gt.content_sha256``."""
    with np.load(path) as z:
        h = hashlib.sha256(np.ascontiguousarray(z["D_orig"], dtype="<f4").tobytes())
        h.update("\n".join(str(s) for s in z["names"]).encode())
    return h.hexdigest()


def scale_sha256(path: Path) -> str:
    with np.load(path) as z:
        h = hashlib.sha256("\n".join(str(s) for s in z["names"]).encode())
        h.update(np.ascontiguousarray(z["area_mm2"], dtype="<f8").tobytes())
    return h.hexdigest()


def _digest(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


def fingerprints(root: Path = ac.DATA_ROOT) -> dict:
    """Impronte attuali dei dati (stessa struttura di ``freeze_manifest.json``)."""
    out = {"views": {}, "gt": {}, "scale": {}}
    for kind in VIEW_KINDS:
        files = sorted((root / kind / "npz").glob("*.npz"))
        lines = [f"{f.stem} {mesh_sha256(f)}" for f in files]
        by_topo = {}
        for ln in lines:
            by_topo.setdefault(ln.split(" ")[0].split("_GTready_")[1], []).append(ln)
        out["views"][kind] = {"n_meshes": len(lines), "meshes": _digest(lines),
                              **{t: _digest(v) for t, v in sorted(by_topo.items())}}
        for k in ("fr", "sr"):
            p = root / kind / f"gt_{k}.npz"
            if p.exists():
                out["views"][kind][f"gt_{k}"] = gt_sha256(p)
    for name in GT_FILES:
        out["gt"][name] = gt_sha256(root / "gt" / f"{name}.npz")
    for kind in VIEW_KINDS:
        out["scale"][kind] = scale_sha256(root / "scale" / f"{kind}.npz")
    return out


def verify_frozen(root: Path = ac.DATA_ROOT, manifest: Path = MANIFEST) -> dict:
    """Controllo dell'impronta da chiamare PRIMA di qualunque valutazione su Ava-256. Ritorna le impronte; se una sola
    differisce da quelle registrate in git (``aau/ava256/freeze_manifest.json``) solleva RuntimeError."""
    ref = json.loads(Path(manifest).read_text())["fingerprints"]
    cur = fingerprints(root)
    diff = []
    for sec in ("views", "gt", "scale"):
        for key, val in ref[sec].items():
            got = cur[sec].get(key)
            if isinstance(val, dict):
                diff += [f"{sec}/{key}/{k}" for k, v in val.items() if (got or {}).get(k) != v]
            elif got != val:
                diff.append(f"{sec}/{key}")
    if diff:
        raise RuntimeError(f"Ava-256: dati diversi da quelli congelati ({manifest}): {diff}")
    return cur


def file_sha256(path: Path) -> str:
    """sha256 dei byte di un file; un symlink vale il suo bersaglio (``symlink:<bersaglio>``)."""
    if path.is_symlink():
        return f"symlink:{os.readlink(path)}"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path, dirs: dict) -> dict:
    """{percorso relativo a ``root``: ``file_sha256``} dei file delle cartelle (ricorsive se ``dirs[d]``), cartelle escluse."""
    out = {}
    for d, rec in dirs.items():
        base = root / d
        items = base.rglob("*") if rec else base.iterdir()
        for p in items:
            if p.is_symlink() or p.is_file():
                out[str(p.relative_to(root))] = file_sha256(p)
    return dict(sorted(out.items()))


def tree_diff(ref: dict, cur: dict) -> list[str]:
    return ([f"mancante {k}" for k in ref if k not in cur] + [f"in piu' {k}" for k in cur if k not in ref] +
            [f"diverso {k}" for k in ref if k in cur and cur[k] != ref[k]])


def verify_frozen_supp(root: Path = ac.DATA_ROOT, manifest: Path = SUPP_MANIFEST) -> dict:
    """Impronte supplementari (byte dei json delle GT, ids.json, split, frame, manifest delle viste, scala): da chiamare
    con ``verify_frozen()`` prima di qualunque valutazione. Uno scarto -> RuntimeError."""
    ref = json.loads(Path(manifest).read_text())["files"]
    cur = tree_sha256(root, SUPP_DIRS)
    diff = tree_diff(ref, cur)
    if diff:
        raise RuntimeError(f"Ava-256: dati diversi dalle impronte supplementari ({manifest}): {diff}")
    return cur


def freeze_supp() -> None:
    if SUPP_MANIFEST.exists():
        raise SystemExit(f"{SUPP_MANIFEST} esiste gia': impronte supplementari registrate")
    if not ac.FROZEN.exists():
        raise SystemExit(f"{ac.FROZEN} mancante: i dati non sono congelati")
    verify_frozen()                                  # i file sono quelli congelati l'11 ottobre
    files = tree_sha256(ac.DATA_ROOT, SUPP_DIRS)
    man = {"recorded_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "rule": __doc__.split("``supp`` / ``verify_frozen_supp()``")[1].strip(),
           "dirs": {d: ("ricorsiva" if r else "solo i file in cima") for d, r in SUPP_DIRS.items()},
           "n_files": len(files), "files": files}
    SUPP_MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
    verify_frozen_supp()
    print(f"[ava-freeze] impronte supplementari: {len(files)} file in {SUPP_MANIFEST}", flush=True)


def _readonly(path: Path) -> int:
    """Toglie la scrittura a un file o a una cartella e a tutto il suo contenuto (i symlink si saltano: il bersaglio sta
    in topo/). Ritorna il numero di voci toccate."""
    n = 0
    items = [path] if path.is_file() else sorted(path.rglob("*"), key=lambda p: -len(p.parts)) + [path]
    for p in items:
        if p.is_symlink():
            continue
        mode = p.stat().st_mode
        p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        n += 1
    return n


def freeze() -> None:
    if ac.FROZEN.exists():
        raise SystemExit(f"{ac.FROZEN} esiste gia': i dati sono congelati")
    t0 = time.time()
    fp = fingerprints()
    man = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "rule": __doc__.split("``verify_frozen()``")[0].split("``freeze``:")[1].strip(),
           "split": {"n_calibration": len(json.loads((ac.GT_DIR / "split.json").read_text())["calibration"]),
                     "n_evaluation": len(json.loads((ac.GT_DIR / "split.json").read_text())["evaluation"])},
           "fingerprints": fp}
    MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
    ac.FROZEN.write_text(json.dumps(man, indent=1) + "\n")
    n = sum(_readonly(ac.DATA_ROOT / d) for d in FROZEN_DIRS if (ac.DATA_ROOT / d).exists())
    n += sum(_readonly(ac.DATA_ROOT / f) for f in FROZEN_FILES) + _readonly(ac.FROZEN)
    verify_frozen()
    print(f"[ava-freeze] congelati: {n} voci in sola lettura; impronte in {MANIFEST} ({time.time() - t0:.0f}s)", flush=True)
    print(json.dumps(fp, indent=1), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=("freeze", "verify", "supp"))
    a = p.parse_args()
    if a.stage == "freeze":
        freeze()
    elif a.stage == "supp":
        freeze_supp()
    else:
        print(json.dumps(verify_frozen(), indent=1))
        print(f"[ava-freeze] supplementari: {len(verify_frozen_supp())} file uguali")
        print("[ava-freeze] impronte uguali a quelle congelate")


if __name__ == "__main__":
    main()
