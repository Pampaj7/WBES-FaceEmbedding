#!/usr/bin/env python3
"""Addestramento NON supervisionato di ULRSSM sulle mesh dei generatori di training (``ulr_data.py``).

    aau/run.sh aau/competitors/ulrssm/ulr_train.py --input xyz --iters 100000 --cache /tmp/<job>/ulr --workers 32
    (ulr.sbatch, passo train; una L40S)

Ricetta = ``FMNetModel`` di ULRSSM con ``options/train/faust.yaml`` (moduli importati da ``ULRSSM_DIR``, non copiati):
DiffusionNet (out 256) -> RegularizedFMNet bidirezionale (lambda 100, gamma 0.5) -> perdite SURFMNet (biiettivita' 1,
ortogonalita' 1, Laplaciano 0) + allineamento Cxy ~ evecs_trans_y Pyx evecs_x nei due versi (``SquaredFrobeniusLoss``,
peso 1), Pxy / Pyx = ``Similarity`` (softmax delle similarita' coseno / tau 0.07); Adam lr 1e-3, clip della norma del
gradiente a 1, CosineAnnealingLR (T_max 15 epoche, eta_min 1e-4) passo per epoca, batch di una coppia.

Il ciclo e' quello di ``FMNetModel.feed_data`` + ``BaseModel.optimize_parameters`` riscritto in poche righe (il loro
``train.py`` vuole i dataset di ``datasets/shape_dataset.py``, che mettono L DENSA in ogni campione: 26k vertici =
2.7 GB per mesh); stesse operazioni nello stesso ordine. Coppie: (i, j) uniformi con reinserimento su tutte le mesh di
training (come ``PairShapeDataset``: il prodotto cartesiano, coppie con se stessa comprese), generatore col seme
``CFG["seed"]``. Budget: ``--iters`` iterazioni divise in 15 epoche; il default di FAUST e' 80^2 x 15 = 96.000.

Ingresso (``--input``): ``wks`` (il default di ULRSSM per forme complete: 128 WKS dagli autovettori, intrinseco, quindi
cieco alla simmetria destra-sinistra di un volto) o ``xyz`` (coordinate normalizzate con l'augmentation di ULRSSM:
rotazione attorno a y fino a 90 gradi, rumore 0.01, scala 0.9-1.1; la loro scelta per il parziale di SHREC'16). Si
addestrano entrambi; la scelta per la tabella e' la regola di ``ulr_eval.py select``: errore di validazione minore.

Validazione (ogni ``val_freq`` iterazioni e alla fine; solo identita' di validazione dei generatori, MAI domini di
valutazione): coppie dello stesso dominio con la STESSA connettivita', quindi corrispondenza nota = identita' dei
vertici: original di g con original di g+1 e con noisy di g+1 (vertici con rumore). Errore = distanza euclidea media
(pesi d'area del bersaglio) fra il vertice mappato e quello vero, nelle coordinate normalizzate all'area 1, x 100
(la scala dell'errore geodetico di ULRSSM); ``flip`` = quota d'area con il vertice mappato dall'altra parte del piano
sagittale (|x| > 0.05) rispetto a quello vero.

Uscite in ``aau/runs/competitors_ulrssm/<tag>/``: ``last.pth`` (ripartenza: rete, ottimizzatore, scheduler, stato dei
generatori, iterazione), ``final.pth`` a fine budget, ``log.jsonl`` (perdite ogni 100 iterazioni, validazione),
``train.json`` (ricetta, tempi, GPU).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")       # un thread per processo di preparazione (torch.set_num_threads nel padre)

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import ulr_common as U  # noqa: E402

_P: dict = {}


def _prep(task):
    """``ulr_common.prep`` di una mesh del dataset (operatori nella cache): (nome, vertici, facce, flipped, s)."""
    import torch
    torch.set_num_threads(1)                         # worker del Pool: niente thread annidati
    name, path = task
    with np.load(path) as z:
        V, F = U.referenced(np.asarray(z["V"], np.float64), np.asarray(z["F"], np.int64))
    r = U.prep(V, F, _P["cache"])
    return name, r["verts"], r["faces"], r["flipped"], r["seconds"]


def prepare(split: str, cache: str, workers: int, subset: int = 0) -> list[dict]:
    man = json.loads((U.DATA / "manifest.json").read_text())
    rows = man["files"][split][:subset] if subset > 0 else man["files"][split]
    _P["cache"] = cache
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_prep, [(r["name"], U.DATA / split / f"{r['name']}.npz") for r in rows], chunksize=4)
    meta = {r["name"]: r for r in rows}
    return [{**meta[n], "verts": v, "faces": f, "flipped": fl, "prep_s": s} for n, v, f, fl, s in res]


class Ops:
    """Operatori dalla cache di ULRSSM (``get_operators``, hit per hash): evals, evecs, evecs_trans in torch."""

    def __init__(self, cache: str):
        self.cache = cache
        self.get = U.ulrssm()["get_operators"]

    def __call__(self, item: dict, device) -> dict:
        import torch
        v, f = torch.from_numpy(item["verts"]), torch.from_numpy(item["faces"]).long()
        _, mass, _, evals, evecs, _, _ = self.get(v, f, k=U.K_FMAP, cache_dir=self.cache)
        return {"verts": v[None].to(device), "faces": f[None].to(device), "evals": evals[None, :U.K_FMAP].to(device),
                "evecs": evecs[None, :, :U.K_FMAP].to(device), "mass": mass.to(device),
                "evecs_trans": (evecs[:, :U.K_FMAP].T * mass[None])[None].to(device)}


def losses(nets: dict, crit: dict, x: dict, y: dict) -> dict:
    """``FMNetModel.feed_data`` (non parziale, senza Dirichlet)."""
    import torch
    import torch.nn.functional as Fn
    feat_x = nets["feature_extractor"](x["verts"], x["faces"])
    feat_y = nets["feature_extractor"](y["verts"], y["faces"])
    Cxy, Cyx = nets["fmap_net"](feat_x, feat_y, x["evals"], y["evals"], x["evecs_trans"], y["evecs_trans"])
    L = crit["surfmnet"](Cxy, Cyx, x["evals"], y["evals"])
    fx, fy = Fn.normalize(feat_x, dim=-1, p=2), Fn.normalize(feat_y, dim=-1, p=2)
    sim = torch.bmm(fx, fy.transpose(1, 2))
    Pxy, Pyx = nets["permutation"](sim), nets["permutation"](sim.transpose(1, 2))
    Cxy_est = torch.bmm(y["evecs_trans"], torch.bmm(Pyx, x["evecs"]))
    Cyx_est = torch.bmm(x["evecs_trans"], torch.bmm(Pxy, y["evecs"]))
    L["l_align"] = crit["align"](Cxy, Cxy_est) + crit["align"](Cyx, Cyx_est)
    return L


def val_pairs(items: list[dict]) -> list[tuple[int, int]]:
    """(sorgente, bersaglio) con la stessa connettivita': original g -> original g+1 e -> noisy g+1, per dominio."""
    pos = {(r["domain"], r["group"], r["label"]): k for k, r in enumerate(items)}
    out = []
    for d in sorted({r["domain"] for r in items}):
        groups = sorted({r["group"] for r in items if r["domain"] == d})
        for g, h in zip(groups, groups[1:] + groups[:1]):
            for lab in ("original", "noisy"):
                a, b = pos.get((d, g, "original")), pos.get((d, h, lab))
                if a is not None and b is not None:
                    out.append((a, b))
    return out


def validate(net, ops: Ops, items: list[dict], vp: list, device) -> dict:
    """Errore medio (x 100, coordinate ad area 1) e quota d'area capovolta sulle coppie di ``val_pairs``, per dominio."""
    import torch
    was = net.training
    net.eval()
    feats, cache = {}, {}
    for k in sorted({k for p in vp for k in p}):
        o = ops(items[k], device)
        cache[k] = o
        feats[k] = U.features(net, items[k]["verts"], items[k]["faces"], device)
    by = {}
    for a, b in vp:
        A, B = cache[a], cache[b]
        m = U.p2p(feats[a], feats[b], A["evecs"][0], B["evecs"][0], B["mass"])      # ogni vertice di b -> uno di a
        Xa = torch.from_numpy(items[a]["verts"]).to(device)
        Xb = torch.from_numpy(items[b]["verts"]).to(device)
        if len(Xa) != len(Xb):
            raise SystemExit("validazione: connettivita' diversa")
        w = B["mass"] / B["mass"].sum()
        # corrispondenza vera: vertice k di b <-> vertice k di a; errore misurato su a (sorgente della mappa)
        err = float((w * torch.linalg.norm(Xa[m] - Xa, dim=1)).sum()) * 100.0
        flip = float((w * ((Xa[m, 0] * Xa[:, 0] < 0) & (Xa[:, 0].abs() > 0.05))).sum())
        d = items[a]["domain"]
        by.setdefault(d, []).append((err, flip))
    net.train(was)
    out = {d: {"err": float(np.mean([e for e, _ in v])), "flip": float(np.mean([f for _, f in v])), "n": len(v)}
           for d, v in by.items()}
    out["all"] = {"err": float(np.mean([x["err"] for x in out.values()])),
                  "flip": float(np.mean([x["flip"] for x in out.values()])), "n": len(vp)}
    return out


def main() -> None:
    import torch
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", choices=("xyz", "wks"), required=True)
    p.add_argument("--iters", type=int, default=100_000)
    p.add_argument("--tag", default="")
    p.add_argument("--cache", required=True, help="cache degli operatori (su /tmp del nodo: conta contro --mem)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-hours", type=float, default=0.0, help="salva last.pth ed esce dopo tante ore (0 = mai)")
    p.add_argument("--subset", type=int, default=0, help="solo le prime N mesh di training (prova di velocita')")
    a = p.parse_args()
    tag = a.tag or f"faces_{a.input}"
    out = U.RUNS / tag
    out.mkdir(parents=True, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    t0 = time.time()
    torch.set_num_threads(max(1, min(8, a.workers)))
    train = prepare("train", a.cache, a.workers, a.subset)
    val = prepare("val", a.cache, a.workers)
    t_prep = time.time() - t0
    print(f"[ulr-train] {len(train)} mesh di training, {len(val)} di validazione, operatori in {t_prep:.0f}s; facce "
          f"invertite {sum(r['flipped'] for r in train + val)}", flush=True)
    m = U.ulrssm()
    device = torch.device("cuda")
    random.seed(U.CFG["seed"])
    np.random.seed(U.CFG["seed"])
    torch.manual_seed(U.CFG["seed"])
    nets = {"feature_extractor": U.feature_net(a.input, a.cache).to(device),
            "fmap_net": m["RegularizedFMNet"](bidirectional=U.CFG["fmap_bidirectional"]).to(device),
            "permutation": m["Similarity"](tau=U.CFG["tau"]).to(device)}
    crit = {"surfmnet": m["SURFMNetLoss"](w_bij=U.CFG["w_bij"], w_orth=U.CFG["w_orth"], w_lap=U.CFG["w_lap"]),
            "align": m["SquaredFrobeniusLoss"](loss_weight=U.CFG["w_align"])}
    opt = torch.optim.Adam(nets["feature_extractor"].parameters(), lr=U.CFG["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=U.CFG["epochs"], eta_min=U.CFG["eta_min"])
    per_epoch = -(-a.iters // U.CFG["epochs"])
    rng = np.random.default_rng(U.CFG["seed"])
    it, info = 0, {"segments": []}
    last = out / "last.pth"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        nets["feature_extractor"].load_state_dict(ck["networks"]["feature_extractor"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        random.setstate(ck["py_rng"])
        it, info = int(ck["iter"]), ck["info"]
        print(f"[ulr-train] ripresa da {last} all'iterazione {it}", flush=True)
    if it >= a.iters:
        print("[ulr-train] budget gia' completato", flush=True)
        return
    ops = Ops(a.cache)
    vp = val_pairs(val)
    log = open(out / "log.jsonl", "a")

    def save(path: Path) -> None:
        tmp = path.with_name(f".{path.name}.tmp")
        torch.save({"networks": {"feature_extractor": nets["feature_extractor"].state_dict()}, "input_type": a.input,
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                    "py_rng": random.getstate(), "iter": it, "info": info, "cfg": U.CFG,
                    "ulrssm_commit": U.ULRSSM_COMMIT}, tmp)
        os.replace(tmp, path)

    nets["feature_extractor"].train()
    seg = {"start_iter": it, "t_prep_s": t_prep, **U.stamp(), "gpu": torch.cuda.get_device_name(0)}
    t1, acc, n_acc = time.time(), {}, 0
    while it < a.iters:
        i, j = (int(x) for x in rng.integers(0, len(train), 2))
        x, y = ops(train[i], device), ops(train[j], device)
        L = losses(nets, crit, x, y)
        loss = sum(L.values())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nets["feature_extractor"].parameters(), U.CFG["clip"])
        opt.step()
        it += 1
        for k, v in L.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        n_acc += 1
        if it % per_epoch == 0:
            sched.step()
        if it % 100 == 0:
            rec = {"iter": it, "lr": opt.param_groups[0]["lr"], **{k: v / n_acc for k, v in acc.items()},
                   "s_per_iter": (time.time() - t1) / (it - seg["start_iter"])}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            acc, n_acc = {}, 0
            if it % 2000 == 0:
                print(f"[ulr-train] {json.dumps(rec)}", flush=True)
        if it % U.CFG["val_freq"] == 0 or it == a.iters:
            v = validate(nets["feature_extractor"], ops, val, vp, device)
            log.write(json.dumps({"iter": it, "val": v}) + "\n")
            log.flush()
            print(f"[ulr-train] val {it}: {json.dumps(v)}", flush=True)
            info["last_val"] = {"iter": it, **v}
            save(last)
        if a.max_hours > 0 and time.time() - t0 > a.max_hours * 3600 and it < a.iters:
            save(last)
            print(f"[ulr-train] limite di {a.max_hours} h: salvato a {it}, si riprende col prossimo job", flush=True)
            break
    seg.update(end_iter=it, seconds=time.time() - t1, max_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    info["segments"].append(seg)
    save(last)
    if it >= a.iters:
        save(out / "final.pth")
    U.atomic_json(out / "train.json", {"input_type": a.input, "iters": a.iters, "iters_done": it,
                                       "per_epoch": per_epoch, "n_train": len(train), "n_val": len(val),
                                       "n_val_pairs": len(vp), "cfg": U.CFG, "ulrssm_commit": U.ULRSSM_COMMIT,
                                       "data_manifest": str((U.DATA / "manifest.json").relative_to(U.REPO)),
                                       **info, "gpu_hours": sum(s["seconds"] for s in info["segments"]) / 3600})
    print(f"[ulr-train] fine segmento: iterazione {it}/{a.iters}, {seg['seconds'] / 3600:.2f} h, "
          f"memoria max {seg['max_mem_gb']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
