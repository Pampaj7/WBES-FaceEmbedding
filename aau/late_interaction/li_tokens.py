#!/usr/bin/env python3
"""Token per vertice dei due semi factorized e distanze di interazione tardiva per insieme di mesh (nessuna GT).

    v3_work/unified_gt/run.sh aau/late_interaction/li_tokens.py stage            (insiemi; link delle mesh del pilota)
    (li.sbatch, passo ops: operatori areanorm k_eig 128 delle mesh del pilota)
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 AAU_NV=--nv aau/competitors/gw_percettivi/gp_run.sh \
        aau/late_interaction/li_tokens.py run --seed s1234                         (li.sbatch, passo run; GPU L40S)

``stage``: ``WORK/sets.json`` con, per insieme, mesh (soggetto, topologia, file), cartella degli operatori, tabella di
scala e (viste valutate) gli embedding della matrice per il controllo; link delle mesh del pilota in
``OPS_ROOT/pilot_<vista>/in``. Mesh in ordine soggetto per soggetto (soggetti ordinati, topologie ``NOCROP``): le
mesh dello stesso soggetto sono contigue (``li_dist.chamfer_set``).

``run`` (un processo per seme; GPU obbligatoria): sha256 del checkpoint, poi il modello con gli agganci di
``eval_v3.install`` e gli argomenti di ``d3_embed.sbatch`` (come ``aau/costo_confronto/cc.py ours``: stesso loader,
stesso ``forward_model``, ``WBES_V3_FACTORIZED_OUT=full``). Per ogni mesh UNA forward senza rumore, col TF32 acceso come
negli embedding di sviluppo, e due agganci che non cambiano nulla: un pre-hook del modello legge X (vertici nel frame
globale, l'ingresso di ``EncoderV3.forward``) e un involucro di ``model.pool`` legge Z (le feature per vertice,
``model_v3.py:92-93``) e w (i pesi del pooling, ``model_v3.py:97``) prima del pooling (``model_v3.py:98``). Token =
(Z_i, w_i / sum w) sui vertici con w_i > 0. Sottocampione (``N_SUB`` vertici): farthest point sampling su X dal vertice
piu' vicino al baricentro pesato, masse = somma dei pesi dei vertici piu' vicini (in X) a ogni campione (Voronoi; come
``gp_gw``). Per mesh si salvano [s, u] del pooling, indici e masse del sottocampione (``meta_path``); i token restano
sulla GPU. Distanze (TF32 spento, ``li_dist``): ``chamfer`` su tutte le coppie di soggetti diversi
(``chamfer_set``), ``chamfer_sub`` e ``sinkhorn`` sulle coppie dell'insieme (pilota e viste: soggetti diversi;
held-out: stesso generatore, soggetti ed etichette diversi, come ``fact_calib.calib_one``). Epsilon = ``EPS_REL`` x
s0, s0 = mediana del costo c sui sottocampioni di ``S0_PAIRS`` coppie a caso (``rng(0)``) dei due piloti (nessuna GT).
Tempi per coppia isolata su ``TIME_PAIRS`` coppie del pilota (regola di parita' del pilota). Uscite:
``dist_path`` (matrici (M, M) per variante, iterazioni e convergenza di Sinkhorn, secondi), ``meta_path``,
``WORK/dist/<seme>/run.json`` (controlli: Z contro gli embedding della matrice, Chamfer contro float64, Sinkhorn contro
POT, s0 ed epsilon, ambiente).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import li_common as lc

S0_PAIRS, TIME_PAIRS, CHECK_PAIRS, POT_PAIRS = 200, 100, 4, 2
ORDER = ("pilot_hifi3d", "pilot_facescape", "heldout", "hifi3d", "facescape")


def log(msg: str) -> None:
    print(f"[li] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# ------------------------------------------------------------------------------------------ messa in scena

def stage() -> None:
    sys.path.insert(0, str(lc.AAU / "baselines_mm"))
    sys.path.insert(0, str(lc.TRAINER / "tools"))
    import blmm
    import fact_paired as fp
    sets = {}
    for v in lc.VIEWS:
        ev = sorted(blmm.subjects(v))
        pil = sorted(blmm.template_subjects(v)[:lc.PILOT_SUBJECTS])
        if set(pil) & set(ev):
            raise SystemExit(f"{v}: soggetti del pilota fra i valutati")
        store = {s: str(fp.embeddings(lc.FP_DOM[v], m["store_variant"], "072")) for s, m in lc.SEEDS.items()}
        key = Path(store["s1234"]).parents[1] / "eval_key.txt"
        ops = Path(dict(x.split("=", 1) for x in key.read_text().split())["data_dir"])
        table = lc.SCALE_TABLES / lc.TABLE[v]
        for name, subj, d, st in ((v, ev, ops, store), (f"pilot_{v}", pil, lc.pilot_ops(v) / "ops", None)):
            items = [(s, t, blmm.mesh_path(v, s, t).name) for s in subj for t in lc.NOCROP]
            sets[name] = {"view": v, "domain": blmm.VIEWS[v]["domain"], "items": items, "ops": str(d),
                          "table": str(table), "store": st}
        src = lc.pilot_ops(v) / "in"
        src.mkdir(parents=True, exist_ok=True)
        for s in pil:
            for t in lc.NOCROP:
                p = blmm.mesh_path(v, s, t)
                if not (src / p.name).exists():
                    (src / p.name).symlink_to(p.resolve())
        log(f"{v}: {len(ev)} valutati, pilota {pil}, operatori di sviluppo {ops}")
    for name, s in sets.items():
        missing = [f for _, _, f in s["items"] if not (Path(s["ops"]) / f).exists()]
        s["ops_missing"] = len(missing)
    lc.atomic_json(lc.WORK / "sets.json", {**sets, "_stamp": lc.stamp()})
    log("sets.json: " + ", ".join(f"{k} {len(s['items'])} mesh" for k, s in sets.items()))


def heldout_set() -> dict:
    """Sottoinsieme degli held-out della calibrazione di c (``fact_calib``): ``HELDOUT_SUBJECTS`` soggetti per
    generatore (``rng(HELDOUT_SEED)``), tutte le etichette; mesh per generatore, poi soggetto (blocchi contigui)."""
    p = lc.WORK / "heldout.json"
    if p.exists():
        return json.loads(p.read_text())
    sys.path.insert(0, str(lc.TRAINER / "tools"))
    import fact_calib
    from common import domain_of
    held = fact_calib.heldout_subjects()
    pick = {d: sorted(np.random.default_rng(lc.HELDOUT_SEED).choice(held[d], lc.HELDOUT_SUBJECTS, replace=False).tolist())
            for d in fact_calib.DOMS}
    names = [f"{s}_GTready_{t}.npz" for d in fact_calib.DOMS for s in pick[d] for t in sorted(lc.HELDOUT_LABELS)]
    have = {p.name for p in (lc.HELDOUT_OPS / "ops").glob("*.npz")}
    if set(names) - have:
        raise SystemExit(f"held-out: {len(set(names) - have)} mesh senza operatori")
    out = {"view": "heldout", "domain": "heldout", "ops": str(lc.HELDOUT_OPS / "ops"),
           "table": str(lc.HELDOUT_OPS / "scale_table.npz"), "store": None, "subjects": pick,
           "items": [(n[:-4].split("_GTready_")[0], n[:-4].split("_GTready_")[1], n) for n in names],
           "generator": {n[:-4].split("_GTready_")[0]: domain_of(n[:-4].split("_GTready_")[0]) for n in names}}
    lc.atomic_json(p, out)
    return out


def set_pairs(name: str, st: dict) -> tuple[np.ndarray, np.ndarray]:
    subj = np.asarray([s for s, _, _ in st["items"]])
    i, j = lc.pair_index(subj)
    if name == "heldout":                                       # fact_calib.calib_one: stesso generatore, etichette diverse
        gen = np.asarray([st["generator"][s] for s in subj])
        lab = np.asarray([t for _, t, _ in st["items"]])
        keep = (gen[i] == gen[j]) & (lab[i] != lab[j])
        i, j = i[keep], j[keep]
    return i, j


# ------------------------------------------------------------------------------------------ modello

class Embedder:
    """Il modello di un seme con gli agganci di eval_v3 (come ``cc.ours``) e la lettura di X, Z, w."""

    def __init__(self, seed: str, tables: list[str]) -> None:
        import torch
        os.environ.update(WBES_V3_FACTORIZED_OUT="full", WBES_EVAL_SCENARIOS="clean",
                          WBES_V3_SCALE_TABLES=":".join(tables))
        sys.path.insert(0, str(lc.TRAINER))
        sys.path.insert(0, str(lc.AAU / "ava256"))
        import eval_v3
        from conf_embed import ARGS                                     # gli argomenti di d3_embed.sbatch
        ck = lc.checkpoint(seed)
        got = lc.sha256(ck)
        if got != lc.SEEDS[seed]["sha256"]:
            raise SystemExit(f"{seed}: sha256 del checkpoint {got}, atteso {lc.SEEDS[seed]['sha256']}")
        _, cargs = eval_v3._ckpt_args(["--model_path", str(ck)])
        import robustness.model_helpers  # noqa: F401
        import robustness.data_utils  # noqa: F401
        eval_v3.install(cargs)
        pert = lc.REPO / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic" / "perturbated"
        sys.path.insert(0, str(pert))
        sys.path.insert(0, str(pert.parent))
        import compare_model_vs_chamfer_topology_breakdown as bd
        self.base = base = bd.base
        names = lc.WORK / "tmp" / f"names_{os.getpid()}.npz"
        names.parent.mkdir(parents=True, exist_ok=True)
        np.savez(names, names=np.asarray(["id000000"]), D_orig=np.zeros((1, 1)))
        sys.argv = ["li", "--model_path", str(ck), "--data_dir", str(lc.HELDOUT_OPS / "ops"), "--dist_npz", str(names),
                    *ARGS]
        cli = bd.parse_args()
        _, ckpt = base._resolve_run_dir_and_checkpoint(cli.model_path, selector=str(cli.checkpoint_selector))
        margs = base._resolve_runtime_args(cli, base.merge_run_args(ckpt, explicit_config_json=cli.config_json))
        base.seed_everything(int(margs.seed))
        self.device = base._resolve_device(cli.device)
        if self.device.type != "cuda":
            raise SystemExit("run: serve la GPU")
        model = base.build_model(args=margs, device=self.device)
        model.load_state_dict(base.load_checkpoint_bundle(ckpt)["state_dict"], strict=True)
        model.eval()
        names.unlink()
        self.cap: dict = {}
        orig_pool = model.pool

        def pool(Z, mass, w_raw=None):                                  # FactorizedEncoderV3.pool, invariato
            self.cap["Z"], self.cap["w"] = Z, (mass if w_raw is None else w_raw)
            return orig_pool(Z, mass, w_raw)

        def pre(_mod, args):
            self.cap["X"] = args[0]

        model.pool = pool
        model.register_forward_pre_hook(pre)
        self.model, self.torch, self.checkpoint = model, torch, ckpt

    def __call__(self, dataset, idx: int):
        """([s, u] (257,), token Z (n, 256) float32, pesi a (n,) a somma 1, X (n, 3)) sui vertici con peso > 0."""
        torch = self.torch
        from li_dist import tf32
        self.cap.clear()
        with torch.no_grad(), tf32(True):
            sd = self.base.sample_to_device(dataset[idx], device=self.device)
            z, _ = self.base.forward_model(model=self.model, sample_dict=sd, V_in=sd["verts"], return_gate_info=False,
                                           add_noise=False)
        Z, w, X = self.cap["Z"], self.cap["w"].reshape(-1), self.cap["X"]
        keep = w > 0
        w = w[keep].double()
        return z.squeeze(0).double().cpu().numpy(), Z[keep].float(), (w / w.sum()).float(), X[keep].double()


def subsample(X, a, n: int):
    """FPS su X dal vertice piu' vicino al baricentro pesato; masse di Voronoi (somma di a sul campione piu' vicino)."""
    import torch
    c = (a.double()[:, None] * X).sum(0) / a.double().sum()
    idx = torch.empty(min(n, len(X)), dtype=torch.long, device=X.device)
    idx[0] = int(((X - c) ** 2).sum(1).argmin())
    d = ((X - X[idx[0]]) ** 2).sum(1)
    for k in range(1, len(idx)):
        idx[k] = d.argmax()
        d = torch.minimum(d, ((X - X[idx[k]]) ** 2).sum(1))
    near = torch.cdist(X, X[idx], compute_mode="donot_use_mm_for_euclid_dist").argmin(1)
    m = torch.zeros(len(idx), dtype=torch.float64, device=X.device).index_add_(0, near, a.double())
    return idx, (m / m.sum()).float()


# ------------------------------------------------------------------------------------------ passo GPU

def embed_set(E: Embedder, name: str, st: dict, seed: str) -> dict:
    """Token di tutte le mesh dell'insieme (GPU), sottocampione, [s, u]; meta e controllo contro la matrice."""
    torch = E.torch
    ds = E.base.GTReadyDataset(st["ops"])
    pos = {Path(f).name: k for k, f in enumerate(ds.files)}
    toks, ws, S, Sm, zs, nt, sidx = [], [], [], [], [], [], []
    t0 = time.time()
    for k, (_, _, fn) in enumerate(st["items"]):
        z, Z, a, X = E(ds, pos[fn])
        idx, m = subsample(X, a, lc.N_SUB)
        toks.append(Z)
        ws.append(a)
        S.append(Z[idx])
        Sm.append(m)
        zs.append(z)
        nt.append(len(Z))
        sidx.append(idx.cpu().numpy())
        if k % 100 == 0:
            log(f"{name}: mesh {k + 1}/{len(st['items'])}, {time.time() - t0:.0f}s")
    t_emb = time.time() - t0
    T = torch.cat(toks)
    mu = T.mean(0, keepdim=True)                                       # centratura: c non cambia
    T -= mu
    out = {"T": T, "w": torch.cat(ws), "off": np.concatenate([[0], np.cumsum(nt)]), "S": torch.stack(S) - mu,
           "m": torch.stack(Sm), "Z": np.stack(zs), "n_tokens": np.asarray(nt), "seconds_embed": t_emb}
    del toks
    lc.atomic_savez(lc.meta_path(seed, name), Z=out["Z"], n_tokens=out["n_tokens"], sub_idx=np.stack(sidx),
                    sub_mass=out["m"].cpu().numpy(), subjects=np.asarray([s for s, _, _ in st["items"]]),
                    topologies=np.asarray([t for _, t, _ in st["items"]]), files=np.asarray([f for _, _, f in st["items"]]),
                    checkpoint=str(E.checkpoint))
    if st.get("store"):                                                # [s, u] contro gli embedding della matrice (A100)
        with np.load(st["store"][seed], allow_pickle=True) as z:
            p = {(str(s), str(t)): r for r, (s, t) in enumerate(zip(z["subjects"], z["topologies"]))}
            ref = np.asarray(z["Z"], np.float64)[[p[(s, t)] for s, t, _ in st["items"]]]
        out["check_Z"] = {"max_abs": float(np.abs(out["Z"] - ref).max()),
                          "max_abs_s": float(np.abs(out["Z"][:, 0] - ref[:, 0]).max()),
                          "max_rel_u": float((np.linalg.norm(out["Z"][:, 1:] - ref[:, 1:], axis=1) /
                                              np.linalg.norm(ref[:, 1:], axis=1)).max())}
    log(f"{name}: {len(nt)} mesh in {t_emb:.0f}s, token {int(sum(nt))} (mediana {int(np.median(nt))}), "
        f"controllo Z {out.get('check_Z')}")
    return out


def s0_of(sets_tok: dict) -> float:
    """Mediana del costo c sui sottocampioni di ``S0_PAIRS`` coppie a caso dei due piloti (nessuna GT)."""
    import li_dist as ld
    vals = []
    for name in ("pilot_hifi3d", "pilot_facescape"):
        st, tk = sets_tok[name]
        i, j = set_pairs(name, st)
        sel = np.random.default_rng(0).choice(len(i), S0_PAIRS // 2, replace=False)
        for q in sel:
            C = ld.cost_batch(tk["S"][[i[q]]], tk["S"][[j[q]]])
            vals.append(C.flatten().cpu().numpy())
    return float(np.median(np.concatenate(vals)))


def dist_set(name: str, st: dict, tk: dict, eps: float, seed: str) -> dict:
    import li_dist as ld
    torch = __import__("torch")
    subj = np.asarray([s for s, _, _ in st["items"]])
    M = len(subj)
    i, j = set_pairs(name, st)
    t0 = time.time()
    torch.cuda.synchronize()
    stop = None
    if name == "heldout":                                       # solo dentro il generatore (mesh ordinate per generatore)
        gen = np.asarray([st["generator"][s] for s in subj])
        if any(gen[k] in gen[:k] and gen[k - 1] != gen[k] for k in range(1, M)):
            raise SystemExit("held-out: generatori non contigui nell'ordine delle mesh")
        stop = np.asarray([np.flatnonzero(gen == g).max() + 1 for g in gen])
    Dc = ld.chamfer_set(tk["T"], tk["w"], tk["off"], subj, stop=stop, log=lambda m: log(f"{name}: {m}"))
    torch.cuda.synchronize()
    t_ch = time.time() - t0
    r = ld.sub_pairs(tk["S"], tk["m"], i, j, eps, log=lambda m: log(f"{name}: {m}"))
    D = {"chamfer": Dc}
    for v in ("chamfer_sub", "sinkhorn"):
        D[v] = np.full((M, M), np.nan)
        np.fill_diagonal(D[v], 0.0)
        D[v][i, j] = D[v][j, i] = r[v]
    keep = np.zeros((M, M), bool)
    keep[i, j] = keep[j, i] = True
    D["chamfer"] = np.where(keep | np.eye(M, dtype=bool), D["chamfer"], np.nan)
    sec = {"chamfer": t_ch, **r["seconds"]}
    lc.atomic_savez(lc.dist_path(seed, name), subjects=subj, topologies=np.asarray([t for _, t, _ in st["items"]]),
                    pair_i=i, pair_j=j, sk_iter=r["sk_iter"], sk_err=r["sk_err"], eps=eps,
                    **{f"D_{v}": D[v] for v in lc.VARIANTS})
    info = {"meshes": M, "pairs": int(len(i)), "seconds": sec, "s_per_pair_batched": {v: s / len(i) for v, s in sec.items()},
            "sinkhorn_not_converged": int((np.abs(r["sk_iter"]) >= lc.SK_MAX_ITER).sum()),
            "sinkhorn_log_fallback": int((r["sk_iter"] < 0).sum()),
            "sinkhorn_iter_median": float(np.median(np.abs(r["sk_iter"]))),
            "sinkhorn_iter_max": int(np.abs(r["sk_iter"]).max()),
            "nan": {v: int(np.isnan(D[v][i, j]).sum()) for v in lc.VARIANTS}}
    log(f"{name}: distanze {info}")
    return info


def checks(name: str, st: dict, tk: dict, eps: float, seed: str) -> dict:
    """Chamfer di ``chamfer_set`` contro float64 su CPU e contro ``chamfer_pair``; Sinkhorn a blocchi contro POT."""
    import li_dist as ld
    with np.load(lc.dist_path(seed, name)) as z:
        i, j, Dc, Ds = z["pair_i"], z["pair_j"], z["D_chamfer"], z["D_sinkhorn"]
    rng = np.random.default_rng(1)
    q = rng.choice(len(i), CHECK_PAIRS, replace=False)
    off, T, w = tk["off"], tk["T"], tk["w"]
    ch = []
    for a_, b_ in zip(i[q], j[q]):
        X, Y = T[off[a_]:off[a_ + 1]], T[off[b_]:off[b_ + 1]]
        ref = ld.chamfer_f64(X.double().cpu().numpy(), w[off[a_]:off[a_ + 1]].double().cpu().numpy(),
                             Y.double().cpu().numpy(), w[off[b_]:off[b_ + 1]].double().cpu().numpy())
        pair = ld.chamfer_pair(X, w[off[a_]:off[a_ + 1]], Y, w[off[b_]:off[b_ + 1]])
        ch.append((abs(Dc[a_, b_] - ref) / ref, abs(pair - Dc[a_, b_]) / ref))
    pot = []                                                     # POT in float64 su CPU e' lento: solo sui piloti
    q2 = rng.choice(len(i), POT_PAIRS if name.startswith("pilot_") else 0, replace=False)
    for a_, b_ in zip(i[q2], j[q2]):
        C = ld.cost_batch(tk["S"][[a_]], tk["S"][[b_]])[0].double().cpu().numpy()
        ref = ld.check_pot(C, tk["m"][a_].double().cpu().numpy(), tk["m"][b_].double().cpu().numpy(), eps)
        pot.append(abs(Ds[a_, b_] - ref) / ref)
    return {"chamfer_vs_float64_rel_max": float(max(c[0] for c in ch)),
            "chamfer_pair_vs_set_rel_max": float(max(c[1] for c in ch)), "n_chamfer": len(ch),
            "sinkhorn_vs_pot_rel_max": float(max(pot)) if pot else None, "n_pot": len(pot)}


def timing(name: str, st: dict, tk: dict, eps: float) -> dict:
    """Secondi per coppia ISOLATA (una alla volta, ``torch.cuda.synchronize``) su ``TIME_PAIRS`` coppie del pilota:
    mediana per variante (regola di parita' del pilota); il riscaldamento (le prime 3 coppie) e' escluso."""
    import li_dist as ld
    torch = __import__("torch")
    i, j = set_pairs(name, st)
    q = np.random.default_rng(2).choice(len(i), TIME_PAIRS + 3, replace=False)
    off, T, w, S, m = tk["off"], tk["T"], tk["w"], tk["S"], tk["m"]
    t = {v: [] for v in lc.VARIANTS}
    for a_, b_ in zip(i[q], j[q]):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ld.chamfer_pair(T[off[a_]:off[a_ + 1]], w[off[a_]:off[a_ + 1]], T[off[b_]:off[b_ + 1]], w[off[b_]:off[b_ + 1]])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        C = ld.cost_batch(S[[a_]], S[[b_]])
        ld.chamfer_batch(C, m[[a_]], m[[b_]]).cpu()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        C = ld.cost_batch(S[[a_]], S[[b_]])
        ld.sinkhorn_batch(C, m[[a_]], m[[b_]], eps)[0].cpu()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        for v, s in zip(lc.VARIANTS, (t1 - t0, t2 - t1, t3 - t2)):
            t[v].append(s)
    return {v: {"median_s": float(np.median(x[3:])), "p25_s": float(np.percentile(x[3:], 25)),
                "p75_s": float(np.percentile(x[3:], 75)), "n": len(x) - 3} for v, x in t.items()}


def run(seed: str, only: list[str]) -> None:
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") != "0":
        raise SystemExit("run: serve TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 (il TF32 si accende solo nella forward)")
    sets = lc.load_sets()
    sets["heldout"] = heldout_set()
    tables = sorted({sets[n]["table"] for n in ORDER})
    E = Embedder(seed, tables)
    import li_dist as ld
    torch = E.torch
    rep_p = lc.WORK / "dist" / seed / "run.json"
    rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
    rep.update(tf32_check_rel=ld.check_no_tf32(E.device), gpu=torch.cuda.get_device_name(0),
               checkpoint=str(E.checkpoint.relative_to(lc.REPO)), sha256=lc.SEEDS[seed]["sha256"], n_sub=lc.N_SUB,
               eps_rel=lc.EPS_REL, sk_tol=lc.SK_TOL, sk_max_iter=lc.SK_MAX_ITER, **lc.stamp())
    tok = {}
    for name in ("pilot_hifi3d", "pilot_facescape"):                  # s0 dai piloti, prima di ogni distanza
        tok[name] = (sets[name], embed_set(E, name, sets[name], seed))
    rep["s0"] = s0_of(tok)
    eps = lc.EPS_REL * rep["s0"]
    rep["eps"] = eps
    log(f"s0 {rep['s0']:.5g}, epsilon {eps:.5g}")
    for name in ORDER:
        if only and name not in only:
            continue
        st = sets[name]
        tk = tok.pop(name)[1] if name in tok else embed_set(E, name, st, seed)
        info = dist_set(name, st, tk, eps, seed)
        info["embed_seconds"] = tk["seconds_embed"]
        info["check_Z_vs_matrice"] = tk.get("check_Z")
        info["checks"] = checks(name, st, tk, eps, seed)
        if name.startswith("pilot_"):
            info["timing_isolated"] = timing(name, st, tk, eps)
        rep.setdefault("sets", {})[name] = info
        lc.atomic_json(rep_p, rep)
        del tk
        torch.cuda.empty_cache()
    log(f"fatto: {rep_p}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("stage", "run"))
    p.add_argument("--seed", choices=list(lc.SEEDS))
    p.add_argument("--only", default="", help="run: solo questi insiemi (virgole; i piloti si incorporano sempre per s0)")
    a = p.parse_args()
    if a.step == "stage":
        stage()
    else:
        run(a.seed, [x for x in a.only.split(",") if x])


if __name__ == "__main__":
    main()
