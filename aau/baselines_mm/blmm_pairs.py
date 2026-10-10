#!/usr/bin/env python3
"""Matrici delle baseline per coppia (Chamfer, ICP + Chamfer; NICP P2Tri) in un modo di normalizzazione.

    aau/outlineB/run_o3d.sh aau/baselines_mm/blmm_pairs.py <vista> <modo> <passo> --shard k/n --workers 32
    (blmm.sbatch; modi e pipeline: blmm.py)

Viste a 6 topologie (hifi3d, faceverse, facescape, facescape_expr): le stesse coppie e gli stessi semi di
``aau/outlineB/alignment_matrix.py`` e ``aau/zs3dmm/zs_bl_same.py``, cioe' per ogni coppia ordinata di topologie
(ta, tb), le 4950 coppie di soggetti i < j (X = i in ta, Y = j in tb, seme = indice della coppia in
``np.triu_indices``) e le 100 coppie stesso soggetto (seme 1000000 + i). Uscite nel formato di quelle
pubblicate, cosi' le legge ``zs_expr_summarize.facebench_distances``: ``<OUT_ROOT>/<vista>/<modo>/matrices/
<metrica>/<ta>__to__<tb>.npz`` (``aau/baselines/common.save_matrix``) e ``.../matrices_same/...`` (``values``).
Distanze in mm (modi ``mm``, ``cs``) o nelle unita' maxabs. Riprendibile per coppia di topologie.

FaMoS: ogni patch di ``test_view`` contro le 30 di galleria (15 scansioni, 15 registrazioni, l'ordine di
``aau/famos/famos_eval.py``): ``<OUT_ROOT>/famos/<modo>/<passo>_<k>of<n>.npz`` con le righe dello shard.

``--check N``: solo le prime N coppie i < j di ogni coppia di topologie dello shard, confrontate con le matrici
pubblicate (``blmm.VIEWS[vista]["fb_root"]``): il controllo di riproduzione del modo ``maxabs``. Scrive
``<OUT_ROOT>/<vista>/<modo>/check_<passo>_<k>of<n>.json`` e nient'altro.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

import blmm

sys.path.insert(0, str(blmm.AAU_DIR / "baselines"))

_C: dict = {}          # (soggetto, topologia) o nome -> (coordinate, offset); ereditato dai worker (fork)


def _run_chunk(task):
    """Worker: ``blmm.pair_metrics`` su un blocco di coppie (chiave X, chiave Y, seme)."""
    mode, step, items = task
    out = []
    for ka, kb, seed in items:
        t0 = time.time()
        out.append((blmm.pair_metrics(_C[ka], _C[kb], seed, mode, step), time.time() - t0))
    return out


def metrics_of(mode: str, step: str) -> list[str]:
    return [m for m in blmm.METRICS[step] if m != "chamfer_pure" or mode == "mm"]


def run_items(pool, items: list, mode: str, step: str, chunk: int) -> tuple[dict, np.ndarray, list]:
    chunks = [(mode, step, items[s:s + chunk]) for s in range(0, len(items), chunk)]
    res = [r for block in pool.map(_run_chunk, chunks, chunksize=1) for r in block]
    vals = {m: np.asarray([r[0][m] for r in res], dtype=np.float64) for m in metrics_of(mode, step)}
    errors = [r[0]["error"] for r in res if r[0]["error"]]
    return vals, np.asarray([r[1] for r in res]), errors


def topo_view(a, prm) -> None:
    import common
    subjects = blmm.subjects(a.view)
    n = len(subjects)
    pairs = [(x, y) for x in blmm.TOPOLOGIES for y in blmm.TOPOLOGIES if x != y]
    k, nsh = (0, 1) if a.claim else (int(x) for x in a.shard.split("/"))
    mine = pairs[k::nsh]
    root = blmm.OUT_ROOT / a.view / a.mode
    mets = metrics_of(a.mode, a.step)
    todo = [p for p in mine if a.overwrite or a.check or not all(
        (root / sub / m / f"{p[0]}__to__{p[1]}.npz").exists() for m in mets for sub in ("matrices", "matrices_same"))]
    print(f"[blmm-pairs] {a.view} {a.mode} {a.step} shard {a.shard}: {len(mine)} coppie di topologie, da fare "
          f"{len(todo)}: {todo}", flush=True)
    if not todo:
        return
    if a.check:
        todo_iter = iter(todo)
    else:
        # Reclami: prima di calcolare una coppia di topologie il task crea un file con O_EXCL nella cartella della
        # sua etichetta (BLMM_CLAIM_TAG, o il job), e salta le coppie gia' su disco o reclamate da altre etichette:
        # job diversi sulla stessa (vista, modo) non rifanno lo stesso lavoro. Con --claim tutti i task di
        # un'etichetta svuotano la stessa coda (i nodi veloci ne fanno di piu'); senza, ogni task ha il suo shard. Un
        # task ucciso lascia il reclamo: alla fine si cancella .claims e si rilancia, si rifa' solo cio' che manca.
        tag = os.environ.get("BLMM_CLAIM_TAG") or os.environ.get("SLURM_JOB_ID", "nojob")
        croot = root / ".claims"
        cdir = croot / f"{tag}_{a.step}"
        cdir.mkdir(parents=True, exist_ok=True)

        def claimed():
            for ta, tb in todo:
                name = f"{ta}__to__{tb}"
                if all((root / sub / m / f"{name}.npz").exists() for m in mets for sub in ("matrices", "matrices_same")):
                    continue
                if any((d / name).exists() for d in croot.glob(f"*_{a.step}") if d != cdir):
                    continue
                try:
                    os.close(os.open(cdir / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    continue
                yield ta, tb
        todo_iter = claimed()
    topos = sorted({t for p in todo for t in p})
    t0 = time.time()
    for s in subjects:
        for t in topos:
            _C[(s, t)] = blmm.work_coords(a.view, a.mode, blmm.mesh_path(a.view, s, t), prm)
    print(f"[blmm-pairs] {len(_C)} mesh caricate in {time.time() - t0:.0f}s", flush=True)
    unit = blmm.unit(a.view, a.mode, prm)
    iu, ju = np.triu_indices(n, 1)
    report = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for ta, tb in todo_iter:
            t1 = time.time()
            n_ij = a.check if a.check else len(iu)
            items = [((subjects[i], ta), (subjects[j], tb), q) for q, (i, j) in enumerate(zip(iu[:n_ij], ju[:n_ij]))]
            if not a.check:
                items += [((s, ta), (s, tb), blmm.SAME_SEED_OFFSET + q) for q, s in enumerate(subjects)]
            vals, sec, errors = run_items(pool, items, a.mode, a.step, a.chunk)
            if a.check:
                rep = {"pair": f"{ta}->{tb}", "n": n_ij}
                for m in mets:
                    ref, subj, _, _, _ = common.load_matrix(common.matrix_path(m, ta, tb, blmm.VIEWS[a.view]["fb_root"]))
                    if subj != subjects:
                        raise SystemExit("soggetti diversi dalle matrici pubblicate")
                    new, old = vals[m] * unit, ref[iu[:n_ij], ju[:n_ij]]
                    both = np.isfinite(new) & np.isfinite(old)
                    rep[m] = {"max_abs_diff": float(np.abs(new[both] - old[both]).max()) if both.any() else None,
                              "n_equal": int((new[both] == old[both]).sum()), "n_both_finite": int(both.sum()),
                              "nan_new": int((~np.isfinite(new)).sum()), "nan_old": int((~np.isfinite(old)).sum())}
                report.append(rep)
                print(f"[blmm-pairs] check {ta}->{tb}: {json.dumps(rep)}", flush=True)
                continue
            for m in mets:
                D = common.empty_matrix(n)
                D[iu, ju] = vals[m][: len(iu)] * unit
                common.save_matrix(root / "matrices" / m / f".{ta}__to__{tb}.tmp.npz", D, subjects, m, ta, tb,
                                   max_sample_points=blmm.N_POINTS, mode=a.mode, unit_mm=unit, n_failed=len(errors))
                (root / "matrices" / m / f".{ta}__to__{tb}.tmp.npz").replace(root / "matrices" / m / f"{ta}__to__{tb}.npz")
                blmm.atomic_savez(root / "matrices_same" / m / f"{ta}__to__{tb}.npz", values=vals[m][len(iu):] * unit,
                                  subjects=np.asarray(subjects, dtype="U16"), metric=m, topology_a=ta, topology_b=tb,
                                  seed_offset=blmm.SAME_SEED_OFFSET, n_failed=len(errors), mode=a.mode)
            print(f"[blmm-pairs] {ta}->{tb}: {len(items)} coppie in {time.time() - t1:.0f}s, "
                  f"{np.mean(sec):.2f}s/coppia, fallite {len(errors)}" + (f" (p.es. {errors[:2]})" if errors else ""),
                  flush=True)
    if a.check:
        (root / f"check_{a.step}_{k}of{nsh}.json").parent.mkdir(parents=True, exist_ok=True)
        (root / f"check_{a.step}_{k}of{nsh}.json").write_text(json.dumps(report, indent=1) + "\n")


def famos_gallery() -> tuple[list[dict], list[int]]:
    """Righe del manifest e indici delle 30 mesh di galleria nell'ordine di ``famos_eval.py`` (scan, poi reg,
    ciascuna per indice di soggetto della GT della vista)."""
    rows = blmm.famos_manifest()
    with np.load(blmm.DATASETS / "FAMOS" / "test_view" / "gt_matrix.npz") as z:
        s2i = {str(s): k for k, s in enumerate(z["names"])}
    gal = []
    for kind in ("scan", "reg"):
        g = [k for k, r in enumerate(rows) if r["kind"] == kind and r["role"] == "gallery"]
        gal += sorted(g, key=lambda k: s2i[rows[k]["view_id"]])
    return rows, gal


def famos(a, prm) -> None:
    rows, gal = famos_gallery()
    k, nsh = (int(x) for x in a.shard.split("/"))
    mine = list(range(len(rows)))[k::nsh]
    out = blmm.OUT_ROOT / "famos" / a.mode / f"{a.step}_{k}of{nsh}.npz"
    if out.exists() and not a.overwrite:
        print(f"[blmm-pairs] {out}: gia' presente", flush=True)
        return
    names = [r["name"] for r in rows]
    for q in sorted(set(mine) | set(gal)):
        _C[names[q]] = blmm.work_coords("famos", a.mode, blmm.VIEWS["famos"]["dir"] / f"{names[q]}.npz", prm)
    items = [(names[q], names[g], blmm.mesh_seed(names[q], names[g])) for q in mine for g in gal]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        vals, sec, errors = run_items(pool, items, a.mode, a.step, a.chunk)
    unit = blmm.unit("famos", a.mode, prm)
    blmm.atomic_savez(out, rows=np.asarray(mine), names=np.asarray(names)[mine], gallery=np.asarray(names)[gal],
                      n_failed=len(errors), errors=np.asarray(errors[:20], dtype="U200"), unit_mm=unit,
                      **{m: v.reshape(len(mine), len(gal)) * unit for m, v in vals.items()})
    print(f"[blmm-pairs] famos {a.mode} {a.step} shard {a.shard}: {len(items)} coppie in {time.time() - t0:.0f}s, "
          f"{np.mean(sec):.2f}s/coppia, fallite {len(errors)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("view", choices=list(blmm.VIEWS))
    p.add_argument("mode", choices=blmm.MODES)
    p.add_argument("step", choices=list(blmm.METRICS))
    p.add_argument("--shard", default="0/1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--check", type=int, default=0)
    p.add_argument("--claim", action="store_true", help="coda condivisa fra i task (ignora --shard, vedi i reclami)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    prm = None if a.mode == "maxabs" else blmm.params()
    if a.view == "famos":
        famos(a, prm)
    else:
        topo_view(a, prm)


if __name__ == "__main__":
    main()
