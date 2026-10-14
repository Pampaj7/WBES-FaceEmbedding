#!/usr/bin/env python3
"""Held-out di MONITORAGGIO del modello da rilasciare (PLAN_MASSIVE sez. 23): identita' fuori dal training, per
dominio, valutate su ogni checkpoint EMA durante il run. Solo sorveglianza: nessuna selezione, nessun numero del paper.

    AAU_NV= aau/run.sh aau/massive/heldout.py make --out datasets/MASSIVE_HELDOUT --workers 32   (aau/massive/heldout.sbatch)
    aau/run.sh aau/massive/heldout.py eval --heldout datasets/MASSIVE_HELDOUT --ckpt <run>/checkpoints/epoch008_ema.pth
    aau/run.sh aau/massive/heldout.py watch --heldout ... --runs <STREAM_OUT>/runs --out <STREAM_OUT>/monitor/heldout.jsonl

``make``: per ogni 3DMM delle fonti ``release`` ``--ids`` identita' col seme RISERVATO SeedSequence([HSEED, d, i]) (d =
indice del dominio in sources.SOURCE_PRESETS["release"]; i produttori usano [20261009 + ..., processo, gruppo]:
flussi indipendenti, coefficienti continui, quindi nessuna identita' del training coincide); per FaMoS ``--famos``
persone TRAIN estratte con rng(HSEED) e TOLTE dal training (massive_node.sh le passa a producer.py --famos-exclude).
Viste come i produttori del run (producer.make_group, provenienza compresa): ``--views`` 6 discretizzazioni senza
reinserimento (crop compreso), quota d'espressioni ``--expr-frac``, operatori k_eig 128 fp32, GT di E12 (fr, sr, S).
Uscita: shard nel formato dell'anello (``<out>/ring/shards``) e ``<out>/heldout.json`` (dichiarazione: semi, persone,
ricetta, conteggi). Mai Ava-256, mai FaMoS TEST (FamosSource lo vieta).

``eval``: embedding [s, u] di ogni vista col checkpoint (stesso servizio dei campioni del training,
consumer.StreamConsumer.materialize, senza rotazione ne' scala; modello in eval, senza rumore latente). Per dominio:
  * ``sp_sr``: Spearman fra d_P del modello (||u_i - u_j|| x c) e d_P della GT (||sr_i - sr_j|| / sqrt(A));
  * ``sp_fr``: Spearman fra d_F (factorized_v3.form_distance con S = exp(s)) e d_FR della GT (mm);
    coppie di viste di identita' diverse, senza crop (come nocrop);
  * ``rank1_fr`` / ``rank1_sr``: per ogni vista (crop compreso) la vista piu' vicina fra le altre del dominio per d_F
    (d_P) e' della stessa identita';
  * ``size_mape``: errore relativo medio della taglia, |exp(s - log S) - 1|.
c = mediana(d_P GT) / mediana(||u_i - u_j||) sulle coppie senza crop di tutti i domini, calcolata sullo stesso
held-out (in campione: e' un monitoraggio, non la calibrazione del protocollo); ``c_over_unit`` = c / targets.gt_unit
("sr"), 1 se la scala di u e' quella della GT del training. Medie fra domini in ``mean``.
``watch``: valuta ogni ``epoch*_ema.pth`` nuovo sotto ``--runs`` e aggiunge una riga a ``--out`` (jsonl); ``--once``
una passata sola (fine run). Un checkpoint fallito si registra con ``error`` e non si ritenta.
``--domains`` (eval, watch): solo le viste di quei domini, es. ``gnm,ict,flame2023`` per il modello a nucleo aperto
(PLAN_MASSIVE sez. 24): stessi semi e stesse viste del held-out completo, c e medie sui soli domini scelti.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[1]
TRAINER, STREAM = REPO / "v3_work" / "trainer", REPO / "v3_work" / "stream"
for _p in (TRAINER, STREAM):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sources as S  # noqa: E402

HSEED = 20261013            # seme riservato dell'held-out (giorno della decisione del PI)
DOMAINS = S.SOURCE_PRESETS["release"]


# --- make ------------------------------------------------------------------------------------------------

def famos_heldout(n: int) -> list[str]:
    """``n`` persone FaMoS TRAIN, rng(HSEED) sull'elenco ordinato di aau/famos/split.json."""
    train = sorted(json.loads(S.FAMOS_SPLIT.read_text())["train"])
    rng = np.random.default_rng(HSEED)
    return sorted(train[int(i)] for i in rng.choice(len(train), size=int(n), replace=False))


class _Person:
    """FamosSource con l'identita' fissata (la persona held-out): ``identity`` non consuma il generatore."""

    def __init__(self, src, person: str) -> None:
        self._src, self._person = src, person

    def identity(self, rng) -> str:
        return self._person

    def __getattr__(self, k):
        return getattr(self._src, k)


def _cfg(a) -> argparse.Namespace:
    """La configurazione di producer.make_group per le viste dell'held-out (quella del run, massive_node.sh)."""
    import views as VW
    lab = list(VW.LABELS)
    w = np.asarray([VW.LABEL_WEIGHTS[k] for k in lab])
    return argparse.Namespace(views=int(a.views), p_expr=0.6, expr_frac=float(a.expr_frac), labels=lab,
                              label_p=w / w.sum(), label_draw="perm", k_eig=int(a.k_eig), evecs_dtype="fp32",
                              v_max=S.V_MAX, v_work=S.V_WORK, partial=None, subdiv={}, release=True,
                              famos_exclude_list=[], code_version=os.environ.get("WBES_CODE_VERSION", ""),
                              mm_aug_probs={})


_G: dict = {}


def _group(task):
    """Un gruppo held-out (processo figlio: sorgenti e bersagli ereditati dal padre con fork)."""
    import producer as P
    d, i, person = task
    cfg, srcs, uni, tg = _G["cfg"], _G["srcs"], _G["uni"], _G["tg"]
    di = DOMAINS.index(d)
    seed = (HSEED, di, i)
    rg = np.random.default_rng(np.random.SeedSequence(list(seed)))
    st = {"views": 0, "groups": 0, "verts": 0, "failures": 0, "by_domain": {}, "by_label": {},
          **{f"t_{k}": 0.0 for k in ("ident", "mesh", "gen", "ops", "pack", "write")}}
    src = _Person(srcs[d], person) if person else srcs[d]
    g = P.make_group(src, d, rg, cfg, uni, f"heldout/{d}/{i:03d}", st, tg, seed)
    g["key"] = f"heldout/{d}/{person or f'{i:03d}'}"
    return g, st["failures"], st.get("last_failure", "")


def make(a) -> None:
    import multiprocessing as mp
    import producer as P
    import views as VW
    from ring import Ring
    from targets import CanonTargets
    out = Path(a.out)
    if (out / "heldout.json").exists() and not a.force:
        raise SystemExit(f"{out}/heldout.json esiste gia' (--force per rifarlo)")
    persons = famos_heldout(a.famos)
    cfg = _cfg(a)
    VW.install_grad_vec()
    uni = S.Unified()
    srcs = S.build_sources(list(DOMAINS), uni, release=True)       # FaMoS su FLAME 2023, tutte le persone TRAIN
    tg = CanonTargets(srcs)
    cfg.mm_factor = {d: tg.mm_factor(d, srcs[d]) for d in DOMAINS}
    cfg.recipe = P.recipe(cfg)
    _G.update(cfg=cfg, srcs=srcs, uni=uni, tg=tg)
    tasks = [(d, i, None) for d in DOMAINS if d != "famos" for i in range(int(a.ids))]
    tasks += [("famos", i, p) for i, p in enumerate(persons)]
    ring = Ring(out / "ring")
    for f in (out / "ring" / "shards").glob("*.shard"):
        f.unlink()
    t0 = time.time()
    counts, fails = {}, []
    with mp.get_context("fork").Pool(int(a.workers)) as pool:
        batch = []
        for g, nf, lf in pool.imap(_group, tasks, chunksize=1):
            if nf:
                fails.append(f"{g['key']}: {nf} viste fallite ({lf})")
            counts[g["domain"]] = counts.get(g["domain"], 0) + len(g["views"])
            batch.append(g)
            if len(batch) == 4:
                ring.write(batch, 0, cfg.recipe)
                batch = []
        if batch:
            ring.write(batch, 0, cfg.recipe)
    decl = {"purpose": "monitoraggio del modello da rilasciare (PLAN_MASSIVE sez. 23): mai selezione, mai paper",
            "seed": HSEED, "seed_rule": "SeedSequence([seed, indice del dominio in domains, identita'])",
            "domains": list(DOMAINS), "ids_per_mm": int(a.ids), "famos_heldout_persons": persons,
            "famos_rule": f"rng({HSEED}).choice sulle persone TRAIN ordinate di aau/famos/split.json, TOLTE dal training",
            "views_per_id": int(a.views), "expr_frac": float(a.expr_frac), "k_eig": int(a.k_eig),
            "views_by_domain": counts, "failures": fails, "recipe": cfg.recipe,
            "sources": {d: srcs[d].describe() for d in DOMAINS}, "mm_factor": cfg.mm_factor,
            "excluded_always": ["Ava-256 (nessuna sorgente)", "FaMoS TEST (FamosSource lo vieta)"],
            "seconds": time.time() - t0}
    (out / "heldout.json").write_text(json.dumps(decl, indent=1, default=str) + "\n")
    print(f"[heldout] {sum(counts.values())} viste in {time.time() - t0:.0f}s: {counts}; FaMoS {persons}; "
          f"fallite {len(fails)}", flush=True)


# --- eval ------------------------------------------------------------------------------------------------

def load_heldout(root: Path, domains=None) -> dict:
    """Viste dell'held-out (dei soli ``domains``, se dati): lettori, (seq, g, v), dominio, identita', etichetta, GT."""
    from ring import Ring, ShardReader
    rows, readers, gt = [], {}, {}
    for seq, p in sorted(Ring(root / "ring").seqs().items()):
        rd = ShardReader(p)
        readers[seq] = rd
        for g, hg in enumerate(rd.groups):
            if domains and hg["domain"] not in domains:
                continue
            gt[hg["key"]] = {"fr": np.asarray(rd.target(g, "fr"), np.float64),
                             "sr": np.asarray(rd.target(g, "sr"), np.float64), "S": float(hg["S"])}
            for v, m in enumerate(hg["views"]):
                rows.append({"seq": seq, "g": g, "v": v, "domain": hg["domain"], "key": hg["key"],
                             "label": m["label"], "expr": m["expr"]})
    if not rows:
        raise SystemExit(f"{root}: held-out vuoto (heldout.py make)")
    return {"rows": rows, "readers": readers, "gt": gt}


def embed(ckpt: Path, H: dict, device: str) -> tuple[np.ndarray, dict]:
    """[s, u] di ogni vista dell'held-out col checkpoint (pesi EMA in state_dict dei *_ema.pth)."""
    import torch
    import factorized_v3 as fz
    from consumer import StreamConsumer
    from model_v3 import build_model_v3, to_device
    from robustness.model_helpers import forward_model
    pack = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**pack["args"])
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = build_model_v3(args, dev)
    model.load_state_dict(pack["state_dict"], strict=True)
    model.eval()
    fz.set_output(model, "full")
    c = StreamConsumer(H["root"] / "ring", input_norm=args.input_norm, area_weights=args.area_weights,
                       rot_deg=(0.0,), scale=0.0, prefetch=0, global_unit_mm=args.global_unit_mm,
                       global_ops=args.global_ops)
    smooth = getattr(model, "area_weights", "mass") == "smooth"
    Z = []
    with torch.no_grad():
        for r in H["rows"]:
            s = c.materialize(H["readers"][r["seq"]], r["g"], r["v"], [0.0, 0.0, 0.0], 1.0)
            sd = to_device(s, dev, True, smooth)
            z, _ = forward_model(model=model, sample_dict=sd, V_in=sd["verts"], return_gate_info=False,
                                 add_noise=False)
            Z.append(z.squeeze(0).double().cpu().numpy())
    return np.stack(Z), {"epoch": int(pack.get("epoch", -1)), "head": str(getattr(args, "head", "")),
                         "steps_per_epoch": int(getattr(args, "steps_per_epoch", 0) or 0)}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation) if len(a) > 2 else float("nan")


def metrics(Z: np.ndarray, H: dict) -> dict:
    import factorized_v3 as fz
    import targets
    rows = H["rows"]
    sqA = float(np.sqrt(S.Unified().A))
    keys = np.asarray([r["key"] for r in rows])
    dom = np.asarray([r["domain"] for r in rows])
    crop = np.asarray([r["label"] == "crop" for r in rows])
    s, u = Z[:, 0], Z[:, 1:]
    logS = np.asarray([np.log(H["gt"][k]["S"]) for k in keys])
    fr = np.stack([H["gt"][k]["fr"] for k in keys])
    sr = np.stack([H["gt"][k]["sr"] for k in keys])
    # coppie senza crop, identita' diverse, di tutti i domini: la c in campione
    pairs = {}
    for d in [d for d in DOMAINS if d in set(dom)]:
        idx = np.flatnonzero(dom == d)
        if len(idx) < 2:
            continue
        i, j = np.triu_indices(len(idx), 1)
        i, j = idx[i], idx[j]
        ok = (keys[i] != keys[j]) & ~crop[i] & ~crop[j]
        pairs[d] = (i[ok], j[ok], idx)
    du_all = np.concatenate([np.linalg.norm(u[i] - u[j], axis=1) for i, j, _ in pairs.values()])
    gp_all = np.concatenate([np.linalg.norm(sr[i] - sr[j], axis=1) / sqA for i, j, _ in pairs.values()])
    c = float(np.median(gp_all) / max(np.median(du_all), 1e-30))
    out = {"c": c, "c_over_unit": c / targets.gt_unit("sr"), "domains": {}}
    for d, (i, j, idx) in pairs.items():
        dP = np.linalg.norm(u[i] - u[j], axis=1) * c
        dF = fz.form_distance(np.exp(s[i]), np.exp(s[j]), dP)
        g_fr = np.linalg.norm(fr[i] - fr[j], axis=1) / sqA
        g_sr = np.linalg.norm(sr[i] - sr[j], axis=1) / sqA
        # rank-1 su tutte le viste del dominio (crop compreso)
        Ss = np.exp(s[idx])
        DU = np.linalg.norm(u[idx][:, None] - u[idx][None], axis=2) * c
        DF = fz.form_distance(Ss[:, None], Ss[None], DU)
        np.fill_diagonal(DU, np.inf)
        np.fill_diagonal(DF, np.inf)
        same = keys[idx][:, None] == keys[idx][None]
        nn_f, nn_p = DF.argmin(1), DU.argmin(1)
        ar = np.arange(len(idx))
        out["domains"][d] = {
            "sp_fr": _spearman(dF, g_fr), "sp_sr": _spearman(dP, g_sr),
            "rank1_fr": float(same[ar, nn_f].mean()), "rank1_sr": float(same[ar, nn_p].mean()),
            "size_mape": float(np.mean(np.abs(np.exp(s[idx] - logS[idx]) - 1.0))),
            "n_views": int(len(idx)), "n_ids": int(len(set(keys[idx]))), "n_pairs": int(len(i)),
            "finite": bool(np.isfinite(Z[idx]).all())}
    ks = ("sp_fr", "sp_sr", "rank1_fr", "rank1_sr", "size_mape")
    out["mean"] = {k: float(np.nanmean([v[k] for v in out["domains"].values()])) for k in ks}
    out["finite"] = bool(np.isfinite(Z).all())
    return out


def eval_one(ckpt: Path, H: dict, device: str) -> dict:
    t0 = time.time()
    Z, info = embed(ckpt, H, device)
    m = metrics(Z, H) if np.isfinite(Z).all() else {"finite": False, "error": "embedding non finiti"}
    step = info["epoch"] * info["steps_per_epoch"] if info["steps_per_epoch"] else None
    return {"t": time.time(), "ckpt": str(ckpt), "epoch": info["epoch"], "step_end_of_epoch": step,
            "seconds": round(time.time() - t0, 1), **m}


def watch(a) -> None:
    H = load_heldout(Path(a.heldout), [d for d in a.domains.split(",") if d])
    H["root"] = Path(a.heldout)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for ln in out.read_text().splitlines():
            try:
                done.add(json.loads(ln)["ckpt"])
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"[heldout] watch {a.runs}: {len(H['rows'])} viste, {len(done)} checkpoint gia' valutati", flush=True)
    while True:
        new = sorted(str(p) for p in Path(a.runs).glob("*/checkpoints/epoch*_ema.pth") if str(p) not in done)
        for p in new:
            time.sleep(5)                     # scrittura atomica (tmp + rename), ma il file e' appena comparso
            try:
                row = eval_one(Path(p), H, a.device)
            except Exception as exc:  # noqa: BLE001  (il sorvegliante non deve morire su un checkpoint)
                row = {"t": time.time(), "ckpt": p, "error": f"{type(exc).__name__}: {exc}"}
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            done.add(p)
            print(f"[heldout] {Path(p).name}: " + (json.dumps(row.get("mean")) if "mean" in row else row.get("error", "")),
                  flush=True)
        if a.once:
            return
        time.sleep(float(a.interval))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", required=True)
    m.add_argument("--ids", type=int, default=32, help="identita' per 3DMM")
    m.add_argument("--famos", type=int, default=6, help="persone FaMoS TRAIN tenute fuori")
    m.add_argument("--views", type=int, default=6)
    m.add_argument("--expr-frac", type=float, default=0.5)
    m.add_argument("--k-eig", type=int, default=128)
    m.add_argument("--workers", type=int, default=16)
    m.add_argument("--force", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--heldout", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--domains", default="", help="solo questi domini (vuoto = tutti)")
    w = sub.add_parser("watch")
    w.add_argument("--heldout", required=True)
    w.add_argument("--runs", required=True, help="directory delle run dir (STREAM_OUT/runs)")
    w.add_argument("--out", required=True)
    w.add_argument("--device", default="cuda:0")
    w.add_argument("--domains", default="", help="solo questi domini (vuoto = tutti)")
    w.add_argument("--interval", type=float, default=120.0)
    w.add_argument("--once", action="store_true")
    p = sub.add_parser("persons", help="stampa le persone FaMoS held-out (virgole)")
    p.add_argument("--famos", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "make":
        make(a)
    elif a.cmd == "eval":
        H = load_heldout(Path(a.heldout), [d for d in a.domains.split(",") if d])
        H["root"] = Path(a.heldout)
        print(json.dumps(eval_one(Path(a.ckpt), H, a.device), indent=1))
    elif a.cmd == "watch":
        watch(a)
    else:
        print(",".join(famos_heldout(a.famos)))


if __name__ == "__main__":
    main()
