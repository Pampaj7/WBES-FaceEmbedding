#!/usr/bin/env python3
"""Pre-pass degli operatori DiffusionNet su /tmp, dalla sola geometria (opzione (c)).

    aau/run.sh aau/data_scale/prepass_ops.py --out-dir /tmp/$SLURM_JOB_ID/view \
        --tars shard_00000.tar shard_00001.tar --geom-dirs datasets/ICT/topo --n-proc 16 \
        [--subjects lista.txt] [--labels original,remesh,...] [--convention areanorm]

Sorgenti: tar di gen_ict_shard.py (membri ``id<g>_GTready_<etichetta>.npz`` con V/F) e/o
directory di geometria gia' esistenti. ``--subjects`` (un id per riga) e ``--labels``
restringono a un blocco. Uscita: layout PIATTO letto dal loader congelato
``GTReadyDatasetNPZ``, cioe' una vista pronta per il trainer.

Convenzioni, ognuna IDENTICA allo script che ha prodotto gli operatori in uso:
  ``areanorm``      v2_work/potential/areanorm_operators.py (mesh centrata, area totale 1,
                    ``compute_operators`` di diffusion-net, k_eig 128): le stesse cinque righe;
  ``robust_area1``  aau/models/robust_area1_operators.py::process, chiamata cosi' com'e'.
Codifica ``areanorm``: le chiavi di ``potential_operators.save_npz``, con facce e indici COO
in int32 e npz NON compresso. Il loader li converte (``.long()``), quindi i tensori che
escono dal loader sono quelli di sempre (misurato: scarto 0 sull'embedding,
aau/data_scale/study_options.json), e la lettura costa quanto oggi (51 ms contro 55 per
campione, loader_timing.json): la compressione costerebbe +36 ms a ogni lettura.

``--transform`` (spento di default): per dominio {"R": 3x3, "flip_faces": bool}, applicato alla
geometria PRIMA degli operatori per canonicalizzare il frame (``apply_frame``). Misurato dal PI su dati
reali: BFM ha il naso verso -z e l'alto verso -y, cioe' R = Rx(180) = diag(1,-1,-1), e normali verso
l'interno (ICT verso l'esterno), cioe' flip_faces true. Solo ``areanorm``.

La geometria estratta dai tar e' cancellata a fine pre-pass (``keep_geom`` per tenerla).

Scrittura atomica (tmp + rename) e file esistenti saltati: si puo' rilanciare. Un processo
per core, a thread singolo (OMP/MKL a 1 vanno messi dal chiamante).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "potential"))
sys.path.insert(0, str(REPO_ROOT / "aau" / "models"))
sys.path.insert(0, str(REPO_ROOT / "diffusion-net" / "src"))

K_EIG = 128
SPARSE = ("L", "gradX", "gradY")


def domain_of_name(name: str) -> str:
    """bfm/flame/ict dall'id, come train_v2.domain_of (BFM <1000, FLAME 1000-9999, ICT >=10000)."""
    num = int(name.split("_GTready_")[0][2:])
    if 100000 <= num < 200000:          # GNM Head, come train_v2.GNM_RANGE
        return "gnm"
    return "bfm" if num < 1000 else ("ict" if num >= 10000 else "flame")


def apply_frame(V: np.ndarray, F: np.ndarray, spec) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalizzazione del frame, PRIMA degli operatori.

    ``spec`` = {"R": matrice 3x3, "flip_faces": bool} (una matrice nuda vale {"R": matrice}).
    I vertici diventano ``V @ R.T``. L'ordine dei vertici di ogni triangolo si inverte se
    ``det(R) < 0`` XOR ``flip_faces``: una riflessione da sola inverte l'orientazione delle
    normali e la si ricompensa; ``flip_faces`` la inverte in piu', indipendentemente da R, per
    portare le normali BFM (verso l'interno) dalla parte di quelle ICT (verso l'esterno). Conta
    perche' i frame tangenti, e quindi gradX/gradY, seguono le normali.
    """
    if not isinstance(spec, dict):
        spec = {"R": spec}
    R = np.asarray(spec.get("R", np.eye(3)), dtype=np.float64)
    V = V @ R.T
    if (np.linalg.det(R) < 0) != bool(spec.get("flip_faces", False)):
        F = F[:, [0, 2, 1]]
    return V, F


COMPRESS = {"on": False}   # impostato da run(): il trainer decomprime una volta sola, al caricamento


def ops_areanorm(src: Path, out: Path, transform: dict | None = None) -> None:
    import torch
    from areanorm_operators import total_area
    from diffusion_net.geometry import compute_operators
    from potential_operators import load_mesh

    # == v2_work/potential/areanorm_operators.py::main, corpo del ciclo
    V, F = load_mesh(src)
    spec = (transform or {}).get(domain_of_name(src.name))
    if spec is not None:
        V, F = apply_frame(V, F, spec)
    A = total_area(V, F)
    if not np.isfinite(A) or A <= 0:
        raise ValueError(f"area non valida: {A}")
    V = (V - V.mean(0)) / np.sqrt(A)
    Vt = torch.tensor(V, dtype=torch.float32)
    Ft = torch.tensor(F, dtype=torch.int32)
    _, mass, L, evals, evecs, gX, gY = compute_operators(Vt, Ft, k_eig=K_EIG)
    # == potential_operators.save_npz, con int32 al posto di int64
    data = {"verts": V.astype(np.float32), "faces": F.astype(np.int32), "mass": mass.numpy(),
            "evals": evals.numpy(), "evecs": evecs.numpy()}
    for name, t in zip(SPARSE, (L, gX, gY)):
        c = t.coalesce()
        data[f"{name}_indices"] = c.indices().numpy().astype(np.int32)
        data[f"{name}_values"] = c.values().numpy()
        data[f"{name}_shape"] = np.array(c.shape)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp.npz")
    (np.savez_compressed if COMPRESS["on"] else np.savez)(tmp, **data)
    os.replace(tmp, out)


def ops_robust_area1(src: Path, out: Path, transform: dict | None = None) -> None:
    from robust_area1_operators import process
    if transform:
        raise ValueError("robust_area1: canonicalizzazione del frame non supportata")
    if src.name != out.name:
        raise ValueError("robust_area1: il nome di uscita e' quello della sorgente")
    process(src, out.parent, K_EIG)


CONVENTIONS = {"areanorm": ops_areanorm, "robust_area1": ops_robust_area1}


def _work(task: tuple) -> tuple[str, float, str]:
    src, out, conv, transform, compress = task
    COMPRESS["on"] = bool(compress)
    t0 = time.time()
    try:
        CONVENTIONS[conv](Path(src), Path(out), transform)
        return Path(src).name, time.time() - t0, ""
    except Exception as exc:  # noqa: BLE001  (una mesh rotta non deve fermare il blocco)
        return Path(src).name, time.time() - t0, f"{type(exc).__name__}: {exc}"


def wanted(name: str, subjects: set[str] | None, labels: set[str] | None) -> bool:
    if not name.endswith(".npz") or "_GTready_" not in name:
        return False
    sid, label = name[:-4].split("_GTready_", 1)
    return (subjects is None or sid in subjects) and (labels is None or label in labels)


def stage_geometry(tars: list[Path], geom_dirs: list[Path], dest: Path,
                   subjects: set[str] | None, labels: set[str] | None,
                   tar_index: Path | None = None) -> list[Path]:
    """Geometria selezionata in ``dest`` (estratta dai tar, symlink per le directory).

    Con ``tar_index`` (cache_budget.py index) i membri si leggono per offset: niente scansione
    dei 3.500 header sparsi di ogni tar, che su CephFS a freddo costa ~10 s a tar.
    """
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    if tar_index is not None and tars:
        sys.path.insert(0, str(THIS_DIR))
        from cache_budget import load_index, read_member
        idx = load_index(tar_index)
        keep_tars = {Path(t).name for t in tars}
        tar_names = [str(x) for x in idx["tars"]]
        handles: dict = {}
        try:
            for i, n in enumerate(idx["names"]):
                n = str(n)
                if tar_names[int(idx["tar_id"][i])] in keep_tars and wanted(n, subjects, labels):
                    p = dest / n
                    if not p.exists():
                        p.write_bytes(read_member(idx, i, handles))
                    out.append(p)
        finally:
            for fh in handles.values():
                fh.close()
        tars = []
    for t in tars:
        with tarfile.open(t) as tar:
            for m in tar:
                if m.isfile() and wanted(m.name, subjects, labels):
                    p = dest / m.name
                    if not p.exists():
                        with tar.extractfile(m) as fh:
                            p.write_bytes(fh.read())
                    out.append(p)
    for d in geom_dirs:
        for p in sorted(d.iterdir()):
            if wanted(p.name, subjects, labels):
                q = dest / p.name
                if not q.exists():
                    q.symlink_to(p.resolve())
                out.append(q)
    return sorted(out)


def run(out_dir: Path, tars: list[Path], geom_dirs: list[Path], n_proc: int,
        subjects: set[str] | None = None, labels: set[str] | None = None,
        convention: str = "areanorm", geom_stage: Path | None = None,
        transform: dict | None = None, compress: bool = False, keep_geom: bool = False,
        tar_index: Path | None = None) -> dict:
    """Pre-pass. La geometria estratta in ``geom_stage`` (~3 GB per blocco di shard) e'
    cancellata alla fine, salvo ``keep_geom``: lasciata li' si accumulerebbe su /tmp, che e' RAM."""
    out_dir.mkdir(parents=True, exist_ok=True)
    geom_stage = geom_stage or out_dir.parent / f"{out_dir.name}_geom"
    t0 = time.time()
    srcs = stage_geometry(tars, geom_dirs, geom_stage, subjects, labels, tar_index)
    t_stage = time.time() - t0
    todo = [(str(p), str(out_dir / p.name), convention, transform, compress) for p in srcs
            if not (out_dir / p.name).exists()]
    # il piu' grande prima: up60k costa 10x down8k, in coda allungherebbe il blocco
    todo.sort(key=lambda t: -Path(t[0]).stat().st_size)
    t1 = time.time()
    if n_proc > 1 and todo:
        with mp.get_context("spawn").Pool(n_proc) as pool:
            res = pool.map(_work, todo, chunksize=1)
    else:
        res = [_work(t) for t in todo]
    wall = time.time() - t1
    if not keep_geom:
        import shutil
        shutil.rmtree(geom_stage, ignore_errors=True)
    fails = [f"{n}: {e}" for n, _, e in res if e]
    return {"n_meshes": len(srcs), "n_computed": len(todo), "n_failed": len(fails),
            "failures": fails[:20], "stage_seconds": t_stage, "wall_seconds": wall,
            "cpu_seconds": float(sum(s for _, s, _ in res)), "n_proc": n_proc,
            "convention": convention}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tars", type=Path, nargs="*", default=[])
    ap.add_argument("--geom-dirs", type=Path, nargs="*", default=[])
    ap.add_argument("--subjects", type=Path, default=None, help="un id per riga")
    ap.add_argument("--labels", default="", help="etichette separate da virgola; vuoto = tutte")
    ap.add_argument("--convention", choices=sorted(CONVENTIONS), default="areanorm")
    ap.add_argument("--n-proc", type=int, default=1)
    ap.add_argument("--tar-index", type=Path, default=None,
                    help="indice dei membri dei tar (cache_budget.py index): lettura per offset")
    ap.add_argument("--compress", action="store_true",
                    help="npz compresso (6.2 contro 8.2 MB/mesh ICT): per il trainer, che decomprime "
                         "una volta al caricamento in cache; letto a ogni campione costerebbe +36 ms")
    ap.add_argument("--transform", default="",
                    help='JSON {dominio: {"R": 3x3, "flip_faces": bool}} applicato alla geometria prima '
                         'degli operatori, es. \'{"bfm": {"R": [[1,0,0],[0,-1,0],[0,0,-1]], '
                         '"flip_faces": true}}\' (Rx 180 gradi, BFM nel frame ICT); vuoto = nessuna')
    a = ap.parse_args()
    subjects = set(a.subjects.read_text().split()) if a.subjects else None
    labels = set(a.labels.split(",")) if a.labels else None
    transform = json.loads(a.transform) if a.transform else None
    rep = run(a.out_dir, a.tars, a.geom_dirs, a.n_proc, subjects, labels, a.convention,
              transform=transform, compress=a.compress, tar_index=a.tar_index)
    print(rep, flush=True)
    raise SystemExit(1 if rep["n_failed"] else 0)


if __name__ == "__main__":
    main()
