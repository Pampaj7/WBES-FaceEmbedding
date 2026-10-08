#!/usr/bin/env python3
"""E11, dati: la GT unificata delle 65.600 identita' del run su scala, nel formato della GT del trainer v2.

    v3_work/unified_gt/run.sh v3_work/unified_gt/train_gt.py          (train_gt.sbatch; dopo unified.py)

Il trainer (``v2_work/fastio/train_steps.py --dist_npz ... --gt-keep-scale``, run
``scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411``) legge ``datasets/SCALE_ALL/gt_joint_bfm_ict_gnm.npz``:
``D_orig`` float32 (65.600 x 65.600) e ``names`` (<U8, ``id0000`` ...), piu' il json accanto con
``global_max`` (la guardia ``check_gt_scale``). Qui lo STESSO formato, con gli STESSI ``names`` nello
STESSO ordine, copiati dal file del run: per usarla basta cambiare ``--dist_npz``.

Forme: la ``original`` neutra di ogni identita' dalle stesse sorgenti del run (``spec.json``):
  - BFM id0000-0499: ``datasets/REMESH/npz_data_topo_500_withops_areanorm`` (``verts``);
  - ICT-5000 id10000-14999: ``datasets/ICT/train_ready/npz_withops`` (``verts``);
  - ICT nuove id20000-69999 e GNM id100000-110099: i tar di ``datasets/SCALE_ALL/shards``, letti per
    offset con ``index.npz`` (``V``). La original GNM e' la patch hockey_mask: i suoi vertici di testa
    tornano agli indici della testa intera, su cui sta la mappa (``gnm_embedding``).
Le viste del run sono normalizzate per mesh (area, maxabs): sono similarita' della forma, e il
Procrustes di similarita' verso mu le toglie. Controllo: s_i da queste sorgenti contro s_i di
``shapes.py`` (pesi del modello o original grezze), su tutte le identita'.

Valori: g_ij = ||s_i - s_j|| / sqrt(A) in mm, poi divisi per il massimo (``D_orig`` max 1, la convenzione
dei file GT del repo; ``mm_per_unit`` nel json riporta ai mm). Coppie fra domini RIEMPITE (la GT
unificata e' definita fra domini; il trainer v2 a blocchi monodominio non le legge, e la guardia NaN
di train_v2 non scatta). Diagonale 0, simmetrica esatta (blocco (j, i) = trasposto di (i, j)).

Uscite: ``datasets/UNIFIED_GT/train/gt_unified_bfm_ict_gnm.npz`` + ``.json``,
``datasets/UNIFIED_GT/train/s_train.npz`` (s_i float32 nell'ordine di ``names``).
"""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import time

import numpy as np

import ugt as C
from shapes import SHAPES_DIR, Space

RUN_GT = C.REPO_ROOT / "datasets" / "SCALE_ALL" / "gt_joint_bfm_ict_gnm.npz"
TAR_INDEX = C.REPO_ROOT / "datasets" / "SCALE_ALL" / "shards" / "index.npz"
BFM_VIEW = C.REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500_withops_areanorm"
ICT_VIEW = C.REPO_ROOT / "datasets" / "ICT" / "train_ready" / "npz_withops"
OUT_DIR = C.DATA_ROOT / "train"
OUT = OUT_DIR / "gt_unified_bfm_ict_gnm.npz"
BLOCK = 8192

_SP = None
_GNM_EMBED = None   # (posizioni nella patch, indici di testa) dei vertici GNM nativi della patch


def domain_of(sid: str) -> str:
    g = int(sid[2:])
    if g < 1000:
        return "bfm"
    if g < 100000:
        return "ict"
    return "gnm"


def gnm_embedding(sp: Space) -> tuple[np.ndarray, np.ndarray, int]:
    """La ``original`` GNM e' la patch hockey_mask (vertici di testa + centri dei quad, compattati,
    gnm_model.face_patch): i vertici di testa tornano al loro indice, i centri si scartano. Tutti i
    vertici della mappa devono stare nella patch (verificato qui)."""
    import gnm_model
    import domains as dm
    m = gnm_model.load_gnm(dm.GNM_NPZ)
    nv = len(dm.gnm()["V"])
    fv = m["face_vertices"]
    keep = np.flatnonzero(fv < nv)
    if not np.isin(sp.z["vidx_gnm"], fv[keep]).all():
        raise SystemExit("vertici della mappa GNM fuori dalla patch hockey_mask")
    return keep, fv[keep], nv


def _init():
    global _SP, _GNM_EMBED
    _SP = Space()
    _GNM_EMBED = gnm_embedding(_SP)


def _map_view(task):
    """Worker: forme dai file npz delle viste (BFM, ICT-5000) -> punti mappati (n, 3)."""
    dom, paths = task
    out = []
    for p in paths:
        with np.load(p) as z:
            V = np.asarray(z["verts"], np.float64)
        out.append(_SP.map(dom, V))
    return np.stack(out)


def _map_tar(task):
    """Worker: membri ``<sid>_GTready_original.npz`` di un tar, letti per offset."""
    tar, items = task
    out = []
    with open(tar, "rb") as fh:
        for sid, off, size in items:
            fh.seek(off)
            with np.load(io.BytesIO(fh.read(size))) as z:
                V = np.asarray(z["V"], np.float64)
            if domain_of(sid) == "gnm":
                pos, head, nv = _GNM_EMBED
                Vh = np.zeros((nv, 3))
                Vh[head] = V[pos]
                V = Vh
            out.append(_SP.map(domain_of(sid), V))
    return np.stack(out)


def load_points(names: list[str], workers: int) -> np.ndarray:
    pos = {s: i for i, s in enumerate(names)}
    P = None
    tasks, where = [], []
    for dom, view in (("bfm", BFM_VIEW), ("ict", ICT_VIEW)):
        sids = [s for s in names if domain_of(s) == dom and (dom == "bfm" or int(s[2:]) < 15000)]
        for k in range(0, len(sids), 250):
            chunk = sids[k:k + 250]
            tasks.append(("view", (dom, [view / f"{s}_GTready_original.npz" for s in chunk])))
            where.append(chunk)
    with np.load(TAR_INDEX) as z:
        tars, tar_id, mnames, off, size = z["tars"], z["tar_id"], z["names"], z["offset"], z["size"]
    want = {s for s in names if int(s[2:]) >= 20000}
    by_tar: dict[int, list] = {}
    for t, n, o, sz in zip(tar_id, mnames, off, size):
        n = str(n)
        if n.endswith("_GTready_original.npz"):
            sid = n.split("_GTready_")[0]
            if sid in want:
                by_tar.setdefault(int(t), []).append((sid, int(o), int(sz)))
    for t, items in sorted(by_tar.items()):
        tasks.append(("tar", (str(TAR_INDEX.parent / str(tars[t])), items)))
        where.append([s for s, _, _ in items])
    found = sum(len(w) for w in where)
    if found != len(names):
        raise SystemExit(f"forme trovate {found} su {len(names)} nomi")
    with mp.get_context("fork").Pool(workers, initializer=_init) as pool:
        res = pool.map(_run, tasks, chunksize=1)
    for chunk, Pc in zip(where, res):
        if P is None:
            P = np.empty((len(names), Pc.shape[1], 3))
        P[[pos[s] for s in chunk]] = Pc
    return P


def _run(task):
    kind, payload = task
    return _map_view(payload) if kind == "view" else _map_tar(payload)


def distances(S: np.ndarray, A: float) -> np.ndarray:
    """g (mm) per blocchi, float64 sui vettori centrati; simmetrica esatta, diagonale 0."""
    X = S.astype(np.float64)
    X -= X.mean(0)
    sq = (X ** 2).sum(1)
    n = len(X)
    D = np.empty((n, n), dtype=np.float32)
    for i in range(0, n, BLOCK):
        for j in range(i, n, BLOCK):
            G = sq[i:i + BLOCK, None] + sq[None, j:j + BLOCK] - 2.0 * X[i:i + BLOCK] @ X[j:j + BLOCK].T
            B = np.sqrt(np.clip(G, 0.0, None) / A).astype(np.float32)
            D[i:i + BLOCK, j:j + BLOCK] = B
            if j != i:
                D[j:j + BLOCK, i:i + BLOCK] = B.T
    i = np.arange(n)
    D[i, i] = 0.0
    # blocchi diagonali: simmetria esatta anche dentro il blocco
    for k in range(0, n, BLOCK):
        Bk = D[k:k + BLOCK, k:k + BLOCK]
        D[k:k + BLOCK, k:k + BLOCK] = 0.5 * (Bk + Bk.T)
    return D


def compare_with_shapes(names: list[str], S: np.ndarray, sp: Space) -> dict:
    """s_i dalle sorgenti del run contro s_i di shapes.py (pesi / original grezze), in mm di g."""
    pos = {s: i for i, s in enumerate(names)}
    out = {}
    for dom in ("bfm", "ict", "gnm"):
        with np.load(SHAPES_DIR / f"{dom}.npz") as z:
            ids, Sref = [str(x) for x in z["ids"]], z["s"]
        both = [k for k, s in enumerate(ids) if s in pos]
        d = np.sqrt(((S[[pos[ids[k]] for k in both]].astype(np.float64)
                      - Sref[both].astype(np.float64)) ** 2).sum(1) / sp.A)
        out[dom] = {"n": len(both), "max_mm": float(d.max()), "median_mm": float(np.median(d))}
    return out


def main() -> None:
    t0 = time.time()
    workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")), 32)
    with np.load(RUN_GT) as z:
        names_arr = z["names"]
    names = [str(s) for s in names_arr]
    if len(set(names)) != len(names):
        raise SystemExit("nomi duplicati nella GT del run")
    sp = Space()
    P = load_points(names, workers)
    print(f"[train-gt] {len(names)} forme mappate in {time.time() - t0:.0f}s", flush=True)
    S = np.empty((len(names), P.shape[1] * 3), dtype=np.float32)
    for k in range(0, len(names), 4000):
        a, _, _ = sp.align(P[k:k + 4000])
        S[k:k + 4000] = sp.svec(a)
    del P
    cmp = compare_with_shapes(names, S, sp)
    print(f"[train-gt] confronto con shapes.py: {cmp}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.save_npz(OUT_DIR / "s_train.npz", s=S, names=names_arr)
    D = distances(S, sp.A)
    gmax = float(D.max())
    D /= gmax
    print(f"[train-gt] distanze in {time.time() - t0:.0f}s; massimo {gmax:.3f} mm", flush=True)
    tmp = OUT_DIR / ".gt.tmp.npz"
    np.savez(tmp, D_orig=D, names=names_arr)
    os.replace(tmp, OUT)
    doms = np.array([domain_of(s) for s in names])
    blocks = {}
    rng = np.random.default_rng(1234)
    for a in ("bfm", "ict", "gnm"):
        for b in ("bfm", "ict", "gnm"):
            if a > b:
                continue
            ia, ib = np.flatnonzero(doms == a), np.flatnonzero(doms == b)
            sa, sb = rng.choice(ia, min(2000, len(ia)), replace=False), rng.choice(ib, min(2000, len(ib)), replace=False)
            v = D[np.ix_(sa, sb)].astype(np.float64) * gmax
            v = v[v > 0]
            blocks[f"{a}|{b}"] = {"median_mm": float(np.median(v)), "max_mm_sample": float(v.max())}
    man = {
        "scale": "unified", "global_max": float(D.max()), "mm_per_unit": gmax, "n_total": len(names),
        "n_by_domain": {d: int((doms == d).sum()) for d in ("bfm", "ict", "gnm")},
        "names_source": str(RUN_GT), "names_identical_to_run_gt": True,
        "cross_domain": "riempite (GT unificata, stessa scala per tutti i domini); nessun NaN",
        "definition": "g_ij = ||s_i - s_j|| / sqrt(A) (RMS pesato per area in mm sulla regione unificata, dopo "
                      "Procrustes di similarita' verso mu), diviso per il massimo: D_orig * mm_per_unit = mm",
        "space": str(C.DATA_ROOT / "unified_space.npz"),
        "sources": {"bfm": str(BFM_VIEW), "ict5000": str(ICT_VIEW), "ict_new_gnm": str(TAR_INDEX)},
        "check_vs_shapes_py": cmp, "blocks_sample": blocks,
        "note": "stesso formato e stessi names di datasets/SCALE_ALL/gt_joint_bfm_ict_gnm.npz: si usa cambiando "
                "solo --dist_npz (con o senza --gt-keep-scale: il massimo e' 1)",
        "seconds": time.time() - t0,
    }
    C.save_json(OUT.with_suffix(".json"), man)
    print(json.dumps(man, indent=1), flush=True)


if __name__ == "__main__":
    main()
