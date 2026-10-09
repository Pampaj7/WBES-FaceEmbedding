#!/usr/bin/env python3
"""Tabella di scala per mesh, per l'ingresso a scala globale (``--input-norm global``, global_v3.py).

Lo store e le viste servono vertici del loader congelato (centro dei vertici e maxabs per mesh) calcolati da npz
gia' normalizzati per area (``areanorm``): la dimensione in mm non sopravvive. Qui, per ogni mesh, si legge la
geometria GREZZA da cui quel campione deriva e si scrive l'area totale nel frame canonico, in mm^2:

    area_mm2 = u_d^2 * area(V_grezza)

con u_d (mm per unita' nativa) e R_d del frame di GT-F di E12 (aau/runs/evidence/e12/frames.json, emendamento 1 di
aau/runs/evidence/e12/protocol.md): u_d = unita' fisica dichiarata (BFM um, ICT cm, GNM m, FaceScape mm) o 63 mm / IPD
della media per le unita' ignote (HIFI3D, FaceVerse); R_d = la rotazione di canonical_transforms.json. NON la scala
s_d = u_d k_d del json, che porta la media di ogni dominio sulla taglia della media FLAME.
Al servizio: fattore = sqrt(area_mm2 / area(V_loader)) (l'area non dipende da centro e rotazione), poi R_d.

Sorgenti:
  * ``--spec`` + ``--split``: le mesh dello store (train + online_eval + online_eval_extra, come build_store.py).
    Tar: il membro grezzo (V, F) letto per offset dall'indice. Viste con operatori: la geometria grezza con lo
    stesso nome in RAW_OF_VIEW; per ogni mesh si VERIFICA che maxabs(grezza) coincida con maxabs(vertici della vista)
    (stesse facce, scarto < 1e-4), cioe' che la vista sia una similarita' senza rotazione della grezza;
  * ``--view-dir`` + ``--domain``: tutti gli npz di una vista di eval (geometria grezza V/F, per es. HIFI3D eval_view).

    aau/run.sh v3_work/trainer/tools/build_scale_table.py --spec S --split P --out T.npz [--workers 16]
    aau/run.sh v3_work/trainer/tools/build_scale_table.py --view-dir datasets/HIFI3D/eval_view/npz --domain hifi3d --out T.npz

Uscita: ``names``, ``domain``, ``area_mm2``, ``area_raw``, ``maxabs_raw`` (unita' native), ``check`` (scarto della
verifica, NaN se non applicabile); accanto un .json con trasformazioni, conteggi e mediane di sqrt(area_mm2).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
sys.path.insert(0, str(TRAINER))
sys.path.insert(0, str(REPO / "aau/data_scale"))

FRAMES = REPO / "aau/runs/evidence/e12/frames.json"   # u_d, R_d, t_d di GT-F (E12)
# vista con operatori (directory reale) -> geometria grezza con gli stessi nomi di file
RAW_OF_VIEW = {REPO / "datasets/REMESH/npz_data_topo_500_withops_areanorm": REPO / "datasets/REMESH/npz_data_topo_500",
               REPO / "datasets/ICT/topo_withops": REPO / "datasets/ICT/topo"}
CHECK_TOL = 1e-4


def total_area(V: np.ndarray, F: np.ndarray) -> float:
    t = V[F]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def maxabs_norm(V: np.ndarray) -> np.ndarray:
    """Come GTReadyDatasetNPZ: centro dei vertici, divisione per max|coordinata|."""
    V = np.asarray(V, dtype=np.float64)
    V = V - V.mean(axis=0, keepdims=True)
    return V / np.abs(V).max()


def _vf(z) -> tuple[np.ndarray, np.ndarray]:
    if "V" in z.files:
        return np.asarray(z["V"], np.float64), np.asarray(z["F"], np.int64)
    return np.asarray(z["verts"], np.float64), np.asarray(z["faces"], np.int64)


def raw_of_view(path: Path) -> tuple[Path, bool]:
    """(file grezzo, True se il file della vista ha operatori da verificare)."""
    real = Path(os.path.realpath(path))
    with np.load(real) as z:
        has_ops = "verts" in z.files and "V" not in z.files
    if not has_ops:
        return real, False
    raw_dir = RAW_OF_VIEW.get(real.parent)
    if raw_dir is None:
        raise SystemExit(f"{path}: vista con operatori senza geometria grezza nota (RAW_OF_VIEW): {real.parent}")
    return raw_dir / real.name, True


def _work(task):
    """Un blocco di mesh: (nome, tipo, sorgente) -> righe. I tar si leggono per offset (indice)."""
    items, tar_index = task
    from cache_budget import read_member
    idx = handles = pos = None
    if tar_index and any(k == "tar" for _, k, _ in items):
        from cache_budget import load_index
        idx = load_index(Path(tar_index))
        pos = {str(n): i for i, n in enumerate(idx["names"])}
        handles = {}
    rows = []
    try:
        for name, kind, src in items:
            check = float("nan")
            if kind == "tar":
                with np.load(io.BytesIO(read_member(idx, pos[name], handles))) as z:
                    V, F = _vf(z)
            else:
                raw, verify = raw_of_view(Path(src))
                with np.load(raw) as z:
                    V, F = _vf(z)
                if verify:
                    with np.load(os.path.realpath(src)) as z:
                        Vv, Fv = _vf(z)
                    if Vv.shape != V.shape or not np.array_equal(Fv, F):
                        raise ValueError(f"{name}: geometria grezza {raw} con vertici/facce diversi dalla vista")
                    check = float(np.abs(maxabs_norm(Vv) - maxabs_norm(V)).max())
                    if not check < CHECK_TOL:
                        raise ValueError(f"{name}: maxabs(grezza) != maxabs(vista), scarto {check:.2e} ({raw})")
            Vc = V - V.mean(0)
            rows.append((name, total_area(V, F), float(np.abs(Vc).max()), check))
    finally:
        for fh in (handles or {}).values():
            fh.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--split", type=Path)
    ap.add_argument("--view-dir", type=Path)
    ap.add_argument("--domain", default="", help="--view-dir: chiave di canonical_transforms.json (hifi3d, faceverse, ...)")
    ap.add_argument("--scale-csv", default="", help="--view-dir: CSV:colonna_nome:colonna_scala; area / scala^2 (annulla la "
                                                    "similarita' delle patch T7 di NoW/FaMoS, scale_to_mm)")
    ap.add_argument("--group", default="", help="--view-dir: prefisso dei nomi (<group>/<file>), es. la cartella degli operatori")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    import global_v3
    tf = {**json.loads(FRAMES.read_text())["domains"], **global_v3.EXTRA_FRAMES}
    tar_index = None
    if a.view_dir:
        if a.domain not in tf:
            raise SystemExit(f"--domain {a.domain!r} non in {sorted(tf)}")
        names = sorted(p.name for p in a.view_dir.glob("*.npz"))
        todo = [(n, "view", str(a.view_dir / n)) for n in names]
        dom = {n: a.domain for n in names}
        source = {"view_dir": str(a.view_dir), "domain": a.domain}
    else:
        import data_v3 as dv
        from common import domain_of, split_name
        spec = json.loads(a.spec.read_text())
        if spec.get("canon"):
            raise SystemExit("spec con 'canon' (rotazione nel pre-pass): non supportata, R_d sarebbe composta")
        split = json.loads(a.split.read_text())
        subjects = set(split["train"]) | set(split.get("online_eval", [])) | \
            {s for ids in (split.get("online_eval_extra") or {}).values() for s in ids}
        sources = dv.collect_sources(spec)
        names = sorted(n for n in sources if split_name(n)[0] in subjects)
        todo = [(n, sources[n][0], sources[n][1]) for n in names]
        dom = {n: domain_of(split_name(n)[0]) for n in names}
        tar_index = spec.get("tar_index")
        source = {"spec": str(a.spec), "split": str(a.split), "n_subjects": len(subjects)}
    missing_dom = sorted({d for d in dom.values() if d not in tf})
    if missing_dom:
        raise SystemExit(f"domini senza frame di GT-F in {FRAMES}: {missing_dom}")
    chunks = [todo[k::max(1, a.workers * 8)] for k in range(max(1, a.workers * 8))]
    chunks = [c for c in chunks if c]
    print(f"[scale] {len(todo)} mesh ({sum(k == 'tar' for _, k, _ in todo)} dai tar) -> {a.out}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_work, [(c, tar_index) for c in chunks]):
            rows += r
    rows.sort(key=lambda r: r[0])
    nm = [r[0] for r in rows]
    if nm != sorted(n for n, _, _ in todo):
        raise SystemExit("mesh mancanti nelle righe")
    area_raw = np.asarray([r[1] for r in rows])
    if a.scale_csv:
        import csv
        path, ncol, scol = a.scale_csv.split(":")
        sc = {r[ncol] + ("" if r[ncol].endswith(".npz") else ".npz"): float(r[scol]) for r in csv.DictReader(open(path))}
        miss = [n for n in nm if n not in sc]
        if miss:
            raise SystemExit(f"--scale-csv: {len(miss)} mesh senza scala (prima {miss[0]})")
        area_raw = area_raw / np.asarray([sc[n] for n in nm]) ** 2
        source["scale_csv"] = a.scale_csv
    if a.group:
        nm = [f"{a.group}/{n}" for n in nm]
        dom = {f"{a.group}/{n}": d for n, d in dom.items()}
    u = np.asarray([float(tf[dom[n]]["u"]) for n in nm])
    area_mm2 = u ** 2 * area_raw
    check = np.asarray([r[3] for r in rows])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, names=np.asarray(nm), domain=np.asarray([dom[n] for n in nm]), area_mm2=area_mm2,
             area_raw=area_raw, maxabs_raw=np.asarray([r[2] for r in rows]), check=check)
    from collections import defaultdict
    by = defaultdict(list)
    for n, v in zip(nm, np.sqrt(area_mm2)):
        lab = n[:-4].split("_GTready_", 1)[1].split("__")[0] if "_GTready_" in n else n.split("/")[0]
        by[(dom[n], lab)].append(v)
    info = {**source, "n_meshes": len(nm), "frames": str(FRAMES),
            "domains": {d: {"u": tf[d]["u"], "R": tf[d]["R"], "unit_source": tf[d]["unit_source"]}
                        for d in sorted(set(dom.values()))},
            "definition": "area_mm2 = u_d^2 * area totale della geometria grezza (unita' native); fattore al servizio "
                          "sqrt(area_mm2 / area(V_loader)), poi rotazione R_d (global_v3.py)",
            "check_maxabs_vs_view": {"n_checked": int(np.isfinite(check).sum()),
                                     "max_abs": float(np.nanmax(check)) if np.isfinite(check).any() else None},
            "sqrt_area_mm_median": {f"{d}|{lab}": float(np.median(v)) for (d, lab), v in sorted(by.items())},
            "seconds": time.time() - t0, "host": os.uname().nodename}
    a.out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("n_meshes", "check_maxabs_vs_view", "seconds")}), flush=True)
    print("[scale] OK", flush=True)


if __name__ == "__main__":
    main()
