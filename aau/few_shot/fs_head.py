#!/usr/bin/env python3
"""Protocollo adattabile: la testa adattata di R6 (PROTOCOL_CONFERMATIVO sez. 10, procedura di D5) con k identita'
etichettate di un dominio nuovo. Encoder congelato, solo la testa.

    v3_work/unified_gt/run.sh aau/few_shot/fs_head.py fit --workers 64      (fs.sbatch, nodo i256-a10-10; ripartibile)

Importati in sola lettura: ``d3_stats`` (sorgenti di D3, ``settings``, ``store_by_names``, ``gt_matrix``), ``d3_e2``
(``fit_e2``: perdita riscalata per blocco e ``d3_head.fit``; ``blocks_for`` per il controllo K0), ``d3_head``
(``Block``, ``calib_c``), ``diag`` (checkpoint, ``dp_per_unit``), ``fact_paired`` (store degli embedding della matrice).

Per (braccio, dominio, k, q): sorgenti = le 7 di D3 tranne il generatore del dominio (``d3_stats.settings()['<T>|all']``,
T = hifi3d, facescape, faceverse per le due viste FaceVerse, famos per FaMoS); blocco del dominio = i k soggetti di
adattamento (``fs_common.drawn``) x le 5 etichette senza crop (FaMoS: x le 10 sequenze), x = u x dp_per_unit dello store
della matrice (FaMoS: i latenti di ``esterna_famos_id``), GT SR del dominio (``CANONICAL_GT/eval/<gt>_sr.npz`` x
dP_per_unit, le GT delle sorgenti di D3), coppie di soggetti diversi ed etichette diverse; in coda alle sorgenti.
Configurazione fissa r = 64, lambda = 1e-6, pesi EQ (``d3_e2.fit_e2``), nessuna CV ne' rifit; c della testa con
``d3_head.calib_c`` sulle SOLE sorgenti (la GT del dominio non entra in c). k = 0: le sole sorgenti (una testa per
braccio e generatore escluso). Controllo K0: la testa ``joint`` (8 sorgenti, FLAME 0-99) con la stessa configurazione
riproduce c e alpha / alpha0 di ``d3_e2_heads.csv`` entro ``K0_REL`` relativo (sullo stesso nodo).

Uscite: ``HEADS/<braccio>__<dominio>__k<k>__q<q>.npz`` e ``HEADS/<braccio>__k0__<T>.npz`` (``alpha``, ``W``, ``r``,
``lam``, ``alpha0``, ``nit``, ``success``, ``loss``, ``c``, ``dpu``, ``subjects`` = soggetti di adattamento, ``host``,
``seconds``); ``EVID/heads.json`` (K0, nodo, esiti).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import fs_common as fc  # noqa: E402
import d3_e2 as E2  # noqa: E402
import d3_head as H  # noqa: E402
import d3_stats as S  # noqa: E402
import diag  # noqa: E402

K0_REL = 1e-6
FAMOS_EMB = fc.REPO / "datasets" / "FAMOS" / "esterna_id" / "eval" / "emb"
FAMOS_GT = S.GT_EVAL / "famos_test_sr.npz"
_DB: dict = {}          # blocchi completi del dominio per (braccio, dominio), ereditati dai worker (fork)
_DPU: dict = {}         # dp_per_unit per braccio


def arm_info() -> dict:
    """Checkpoint e dp_per_unit dei bracci: quelli della calibrazione (come ``d5_fewshot.main_fit``)."""
    info = {"ckpt": {}, "dpu": {}}
    for arm in fc.ARMS:
        _, _, ck = S.store_by_names(diag.CALIB_EMB / arm / "embeddings.npz", diag.set_names("bfm"))
        info["ckpt"][arm], info["dpu"][arm] = ck, diag.dp_per_unit(ck)
    return info


def patch_fact_paired():
    """La vista neutra di FaceVerse in ``fact_paired`` (come ``fm_eval.main``: in memoria, il modulo non si tocca)."""
    import fact_paired as fp
    fp.VIEWS["faceverse_neutral"] = ("fvn", "mesh_pair_nocrop")
    fp.FORM_DIR["fvn"] = os.path.relpath(S.NEUTRAL_EMB, fp.EVAL)
    return fp


def store_Z(domain: str, keys: list) -> dict:
    """{braccio: (Z sulle ``keys`` (soggetto, topologia), checkpoint)} dallo store della matrice
    (``fact_paired.embeddings``, come ``fact_paired.model_columns``)."""
    fp = patch_fact_paired()
    dom = fp.VIEWS[domain][0]
    out = {}
    for pre, v, e in fp.MODELS:
        if pre not in fc.ARMS:
            continue
        p = fp.embeddings(dom, v, e)
        with np.load(p, allow_pickle=True) as z:
            Z = np.asarray(z["Z"], np.float64)
            kz = list(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))
            ck = Path(str(z["checkpoint"]))
        pos = {k: r for r, k in enumerate(kz)}
        miss = [k for k in keys if k not in pos]
        if miss:
            raise SystemExit(f"{domain} {pre}: {len(miss)} mesh assenti dallo store {p} (es. {miss[:2]})")
        out[pre] = (Z[[pos[k] for k in keys]], ck)
    return out


def domain_subjects(domain: str) -> list[str]:
    """Soggetti del dominio in ordine di stringa (l'ordine delle righe e dei conteggi della matrice)."""
    if domain == fc.FAMOS:
        with np.load(FAMOS_EMB / f"{fc.ARMS[0]}.npz") as z:
            return sorted({str(n).split("_GTready_")[0] for n in z["names"]})
    import blmm
    return sorted(blmm.subjects(domain))


def gt_sub(domain: str, subjects: list[str]) -> np.ndarray:
    """GT SR (d_P fisica) fra i soggetti, nell'ordine dato; simmetrica, diagonale 0, positiva fuori."""
    path = FAMOS_GT if domain == fc.FAMOS else S.GT_EVAL / f"{S.GT_SET[domain]}_sr.npz"
    gn, G = S.gt_matrix(path, "sr")
    pos = {str(s): k for k, s in enumerate(gn)}
    ii = np.asarray([pos[s] for s in subjects])
    Gs = np.asarray(G, np.float64)[np.ix_(ii, ii)]
    if not (np.allclose(Gs, Gs.T) and np.all(np.diag(Gs) == 0) and np.all(Gs[~np.eye(len(ii), dtype=bool)] > 0)):
        raise SystemExit(f"{domain}: GT non simmetrica o con zeri fuori diagonale")
    return Gs


def domain_block(arm: str, domain: str, info: dict) -> H.Block:
    """Il blocco di TUTTI i soggetti del dominio (``take`` lo restringe all'adattamento): mesh senza crop
    (soggetto x etichetta), x = u x dp_per_unit, G = GT SR fra i soggetti ordinati."""
    subjects = domain_subjects(domain)
    if domain == fc.FAMOS:
        with np.load(FAMOS_EMB / f"{arm}.npz") as z:
            names = [str(n) for n in z["names"]]
            Z = np.asarray(z["Z"], np.float64)
        import ei
        if ei.CKPT[arm].resolve() != info["ckpt"][arm].resolve():           # K4: stesso checkpoint della calibrazione
            raise SystemExit(f"{arm}: latenti di FaMoS da {ei.CKPT[arm]}, atteso {info['ckpt'][arm]}")
        keys = [tuple(n.split("_GTready_")) for n in names]
        labels = sorted({t for _, t in keys})
    else:
        keys = [(s, t) for s in subjects for t in fc.NOCROP]
        Zs = store_Z(domain, keys)
        Z, ck = Zs[arm]
        if ck.resolve() != info["ckpt"][arm].resolve():
            raise SystemExit(f"{domain} {arm}: checkpoint {ck}, atteso {info['ckpt'][arm]}")
        labels = list(fc.NOCROP)
    if not np.isfinite(Z).all() or Z.shape[1] != 257:
        raise SystemExit(f"{domain} {arm}: latenti non finiti o non [s, u]")
    spos = {s: k for k, s in enumerate(subjects)}
    subj = np.asarray([spos[s] for s, _ in keys])
    lab = np.asarray([labels.index(t) for _, t in keys])
    if np.bincount(subj, minlength=len(subjects)).min() != len(labels):
        raise SystemExit(f"{domain} {arm}: non tutte le etichette per ogni soggetto")
    return H.Block(Z[:, 1:] * info["dpu"][arm], subj, lab, gt_sub(domain, subjects), domain)


def build(info: dict) -> None:
    for arm in fc.ARMS:
        S.build_sources(arm, info)
        for d in fc.DOMAINS + (fc.FAMOS,):
            _DB[(arm, d)] = domain_block(arm, d, info)


def sources(arm: str, domain: str) -> list[H.Block]:
    srcs = S.settings()[f"{fc.TARGET[domain]}|all"]
    if fc.TARGET[domain] in srcs:
        raise SystemExit(f"{domain}: il generatore del dominio fra le sorgenti")
    return [S._B[(arm, s)] for s in srcs]


def _fit(task):
    arm, domain, k, q = task
    t0 = time.time()
    src = sources(arm, domain)
    if k:
        A, _ = fc.drawn(domain, k, q)
        blocks = src + [_DB[(arm, domain)].take(A)]
    else:
        A, blocks = np.zeros(0, dtype=int), src
    head = E2.fit_e2(blocks, fc.R_HEAD, fc.LAM_HEAD)                              # EQ, perdita riscalata per blocco
    c = H.calib_c(lambda b, i, j: head.dist(b.X[i], b.X[j]), src)               # SOLO le sorgenti
    nt = int(len(np.unique(blocks[-1].subj))) if k else 0
    fc.atomic_savez(fc.head_path(arm, domain, k, q), alpha=head.alpha, W=head.W, r=head.r, lam=head.lam,
                    alpha0=head.alpha0, nit=head.nit, success=head.success, loss=head.loss, c=c,
                    dpu=_DPU[arm], subjects=A, n_target=nt,
                    sources=np.asarray([b.name for b in src]), host=socket.gethostname(), seconds=time.time() - t0)
    return task, time.time() - t0


def _k0(arm):
    """K0: la testa ``joint`` di D3 e2 con la configurazione di R6 (come ``conf_head.step_fit``)."""
    blocks = E2.blocks_for(arm, "joint")
    h = E2.fit_e2(blocks, fc.R_HEAD, fc.LAM_HEAD)
    c = H.calib_c(lambda b, i, j: h.dist(b.X[i], b.X[j]), blocks)
    return arm, {"c": c, "alpha_over_alpha0": h.alpha / h.alpha0, "nit": h.nit, "success": h.success}


def k0_reference(arm: str) -> dict:
    for r in csv.DictReader(open(diag.EV / "d3_e2_heads.csv")):
        if r["kind"] == "e2" and r["arm"] == arm and r["setting"] == "joint":
            if (int(r["r"]), float(r["lambda"])) != (fc.R_HEAD, fc.LAM_HEAD):
                raise SystemExit(f"{arm}: configurazione joint {r['r']}, {r['lambda']} diversa da quella di R6")
            return {"c": float(r["c"]), "alpha_over_alpha0": float(r["alpha_over_alpha0"]), "nit": int(r["nit"])}
    raise SystemExit(f"{arm}: riga e2 / joint assente da d3_e2_heads.csv")


def main_fit(a) -> None:
    t0 = time.time()
    info = arm_info()
    _DPU.update(info["dpu"])
    build(info)
    fc.check_draws()
    tasks = fc.head_tasks()
    todo = [x for x in tasks if not fc.head_path(*x).exists()]
    todo.sort(key=lambda x: -x[2])                                              # i blocchi piu' grandi prima
    print(f"[fs-head] {len(tasks)} teste, {len(todo)} da fare, host {socket.gethostname()} "
          f"({time.time() - t0:.0f}s)", flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        k0 = pool.map_async(_k0, fc.ARMS, chunksize=1)
        for n, (task, sec) in enumerate(pool.imap_unordered(_fit, todo, chunksize=1), 1):
            if n % 10 == 0 or n == len(todo):
                print(f"[fs-head] {n} / {len(todo)} (ultima {task}, {sec:.0f}s; {time.time() - t0:.0f}s)", flush=True)
        k0 = dict(k0.get())
    rep = {"K0": {}, "host": socket.gethostname(), "node_required": fc.NODE_HEAD, "r": fc.R_HEAD, "lambda": fc.LAM_HEAD,
           "dpu": info["dpu"], "checkpoints": {k: str(v) for k, v in info["ckpt"].items()}}
    for arm, x in k0.items():
        ref = k0_reference(arm)
        x.update(reference=ref, rel_c=abs(x["c"] / ref["c"] - 1.0),
                 rel_alpha=abs(x["alpha_over_alpha0"] / ref["alpha_over_alpha0"] - 1.0))
        x["pass"] = bool(x["rel_c"] <= K0_REL and x["rel_alpha"] <= K0_REL)
        rep["K0"][arm] = x
    fits = []
    for task in tasks:
        with np.load(fc.head_path(*task)) as z:
            fits.append({"task": list(task), "success": bool(z["success"]), "nit": int(z["nit"]), "c": float(z["c"]),
                         "alpha_over_alpha0": float(z["alpha"] / z["alpha0"]), "seconds": float(z["seconds"]),
                         "host": str(z["host"]), "n_target": int(z["n_target"])})
    rep.update(n=len(fits), success=sum(f["success"] for f in fits), hosts=sorted({f["host"] for f in fits}),
               nit_min=min(f["nit"] for f in fits), nit_max=max(f["nit"] for f in fits),
               seconds_median=float(np.median([f["seconds"] for f in fits])), fits=fits,
               sources={d: list(S.settings()[f"{fc.TARGET[d]}|all"]) for d in fc.DOMAINS + (fc.FAMOS,)},
               seconds=time.time() - t0, **fc.stamp())
    fc.atomic_json(fc.EVID / "heads.json", rep)
    print(f"[fs-head] finito: {rep['success']} / {rep['n']} con success, K0 "
          f"{ {a_: (x['pass'], x['rel_c'], x['rel_alpha']) for a_, x in rep['K0'].items()} } ({time.time() - t0:.0f}s)",
          flush=True)


def load_head(path: Path) -> tuple[H.Head, float, dict]:
    with np.load(path) as z:
        h = H.Head(float(z["alpha"]), np.asarray(z["W"]), int(z["r"]), float(z["lam"]), float(z["alpha0"]),
                   int(z["nit"]), bool(z["success"]), float(z["loss"]))
        meta = {"c": float(z["c"]), "subjects": np.asarray(z["subjects"]), "host": str(z["host"]),
                "success": bool(z["success"]), "nit": int(z["nit"]), "dpu": float(z["dpu"])}
    return h, meta["c"], meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("fit",))
    p.add_argument("--workers", type=int, default=64)
    a = p.parse_args()
    main_fit(a)


if __name__ == "__main__":
    main()
