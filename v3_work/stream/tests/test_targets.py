#!/usr/bin/env python
"""Correttezza (c): i bersagli della GT di E12 calcolati al volo dallo stream (targets.py) e l'ingresso globale.

    aau/run.sh v3_work/stream/tests/test_targets.py --out aau/runs/evidence/stream/targets_check.json

1. **Contro i file di E12** (``datasets/CANONICAL_GT/train``: fr_train, sr_train, centroid_size): per le identita'
   con forma nota, la strada dei produttori (ICT e GNM: ``MMSource.neutral_points`` dai pesi, come test_gt.py;
   BFM REMESH: le original di train_fr_sr, non e' un dominio dello stream) -> ``CanonTargets`` una identita' alla
   volta. Scarti: ||fr - fr_file|| / sqrt(A) in mm, ||sr - sr_file|| / sqrt(A), |S - S_file| in mm; soglia 1e-5.
   Poi le coppie: d_FR e d_P della GT dello stream (``StreamGT``) contro D_orig x unita' di
   gt_{fr,sr}_bfm_ict_gnm.npz (grezze, lette in memmap).
2. **Domini dello stream** (``--n-per-domain`` identita' campionate): S_i (media, CV), convergenza della rigida
   robusta, secondi per identita'; ``mm_factor`` (u_d / s_d) verificato: i punti della regione nel frame canonico
   della libreria x mm_factor hanno la stessa centroid size dei punti nel frame di GT-F e differiscono per una
   rigida (residuo).
3. **Espressioni:** per ogni dominio una vista con espressione si sposta dalla neutra (mm).
4. **Ruoli:** FaceScape, HIFI3D, FaceVerse vietati ai campionatori di training e allo stream.
5. **Anello e consumatore:** un gruppo per dominio scritto in un anello temporaneo (``make_group`` con i bersagli),
   letto da ``StreamConsumer`` con GT fr / sr e ingresso globale: GT del batch = distanze dai bersagli, log S nelle
   viste, area di X = area_mm2 / L0^2, X = vertici veri in mm centrati / L0.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
STREAM = THIS.parent
REPO = STREAM.parents[1]
for _p in (THIS, STREAM, REPO / "v3_work" / "trainer"):
    sys.path.insert(0, str(_p))

CGT_TRAIN = REPO / "datasets" / "CANONICAL_GT" / "train"
TOL = 1e-5


def q(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"max": float(x.max()), "median": float(np.median(x))}


def against_files(tg, uni, sets: dict, out: dict) -> dict:
    """Bersagli al volo contro fr_train / sr_train / centroid_size; ritorna {sid: bersagli} per le coppie."""
    import data_v3 as dv
    with np.load(CGT_TRAIN / "centroid_size_bfm_ict_gnm.npz") as z:
        names = [str(s) for s in z["names"]]
        S_file = z["S"]
    pos = {s: i for i, s in enumerate(names)}
    FR = dv.npz_member_memmap(CGT_TRAIN / "fr_train.npz", "a")
    SR = dv.npz_member_memmap(CGT_TRAIN / "sr_train.npz", "z")
    sqA = np.sqrt(tg.A)
    got = {}
    for d, (sids, P) in sets.items():
        t0 = time.time()
        T = [tg(d, p) for p in P]
        dt = (time.time() - t0) / len(P)
        idx = np.asarray([pos[s] for s in sids])
        o = np.argsort(idx)
        fr_ref = np.asarray(FR[idx[o]], dtype=np.float64)[np.argsort(o)]
        sr_ref = np.asarray(SR[idx[o]], dtype=np.float64)[np.argsort(o)]
        fr = np.stack([t["fr"] for t in T]).astype(np.float64)
        sr = np.stack([t["sr"] for t in T]).astype(np.float64)
        S = np.asarray([t["S"] for t in T])
        e_fr = np.linalg.norm(fr - fr_ref, axis=1) / sqA
        e_sr = np.linalg.norm(sr - sr_ref, axis=1) / sqA
        e_S = np.abs(S - S_file[idx])
        row = {"n": len(sids), "ids_first_last": [sids[0], sids[-1]], "fr_mm": q(e_fr), "sr": q(e_sr), "S_mm": q(e_S),
               "s_per_identity": dt, "converged": float(np.mean([t["converged"] for t in T])),
               "pass": bool(max(e_fr.max(), e_sr.max(), e_S.max()) < TOL)}
        out["files"][d] = row
        print(f"[targets] {d}: {row}", flush=True)
        for s, t in zip(sids, T):
            got[s] = t
    return got


def pair_check(tg, got: dict, out: dict) -> None:
    """d_FR e d_P dalla GT dello stream contro le GT grezze di E12 (D_orig x unita'), tutte le coppie."""
    import data_v3 as dv
    from consumer import StreamGT
    with np.load(CGT_TRAIN / "centroid_size_bfm_ict_gnm.npz") as z:
        pos = {str(s): i for i, s in enumerate(z["names"])}
    sids = sorted(got, key=lambda s: pos[s])
    idx = np.asarray([pos[s] for s in sids])
    for kind, key in (("fr", "mm_per_unit"), ("sr", "dP_per_unit")):
        unit = float(json.loads((CGT_TRAIN / f"gt_{kind}_bfm_ict_gnm.json").read_text())[key])
        D = dv.npz_member_memmap(CGT_TRAIN / f"gt_{kind}_bfm_ict_gnm.npz", "D_orig")
        ref = np.asarray(D[np.ix_(idx, idx)], dtype=np.float64) * unit
        gt = StreamGT(tg.A, 1.0, keep=10 ** 6)
        rows = np.asarray([gt.register(s, got[s][kind], "x") for s in sids])
        diff = np.abs(gt[np.ix_(rows, rows)] - ref)
        off = ~np.eye(len(ref), dtype=bool)
        out["pairs"][kind] = {"n_identities": len(sids), "n_pairs": int(off.sum() // 2), "unit": unit,
                              "max_abs": float(diff.max()), "ref_median": float(np.median(ref[off])),
                              "max_rel": float((diff[off] / ref[off]).max()),
                              "note": "float32 della matrice di E12: risoluzione ~6e-8 x massimo"}
        print(f"[targets] coppie {kind}: {out['pairs'][kind]}", flush=True)


def domains_check(tg, srcs: dict, n: int, rng, out: dict) -> None:
    for d, src in srcs.items():
        t0 = time.time()
        P = np.stack([src.neutral_points(src.identity(rng)) for _ in range(n)]) if src.kind == "mm" else \
            np.stack([src.neutral_points(p) for p in src.persons])
        T = tg(d, P)
        dt = (time.time() - t0) / len(P)
        S = T["S"]
        row = {"n": int(len(P)), "S_mean_mm": float(S.mean()), "S_cv": float(S.std(ddof=1) / S.mean()),
               "S_p5_p95_mm": [float(np.percentile(S, 5)), float(np.percentile(S, 95))],
               "converged": float(T["converged"].mean()), "iterations_p95": float(np.percentile(T["iterations"], 95)),
               "s_per_identity_batch": dt}
        # mm_factor: frame canonico della libreria x mm_factor contro il frame di GT-F (stessa taglia, rigida)
        f = tg.mm_factor(d, src)
        row["mm_factor"] = f
        out["domains"][d] = row
        if src.kind != "mm":       # FaMoS: viste allineate con una rigida SENZA scala, fattore 1 per costruzione
            print(f"[targets] dominio {d}: {row}", flush=True)
            continue
        c = src.canon
        Xc = f * (c["scale"] * P[:8] @ np.asarray(c["R"]).T + np.asarray(c["t"]))
        XF = tg.cn.to_F(d, P[:8])
        cs_ratio = tg.cn.centroid_size(Xc) / tg.cn.centroid_size(XF)
        from cgt import rigid_fit
        res = []
        for k in range(len(Xc)):
            Rk, tk = rigid_fit(Xc[k:k + 1], XF[k], tg.cn.W[None])
            a = Xc[k] @ Rk[0].T + tk[0]
            res.append(float(np.sqrt((tg.cn.W * ((a - XF[k]) ** 2).sum(1)).sum())))
        row["view_frame_vs_F"] = {"cs_ratio_max_dev": float(np.abs(cs_ratio - 1).max()), "rigid_residual_mm_max": max(res)}
        out["domains"][d] = row
        print(f"[targets] dominio {d}: {row}", flush=True)


def expr_check(srcs: dict, rng, out: dict) -> None:
    for d, src in srcs.items():
        ident = src.identity(rng)
        Vn, _, tn = src.view_mesh(ident, rng, False)
        disp = []
        for _ in range(5):
            Ve, _, te = src.view_mesh(ident, rng, True)     # FaMoS: fotogramma reale, entrambe allineate
            disp.append(float(np.linalg.norm(Ve - Vn, axis=1).max()))
        out["expressions"][d] = {"tag": te, "max_disp_mm_over_5": [round(x, 3) for x in disp],
                                 "pass": bool(te == "expr" and min(disp) > 0.5)}
        print(f"[targets] espressioni {d}: {out['expressions'][d]}", flush=True)


def roles_check(out: dict) -> None:
    import sources as S
    from v3_work.mm import RoleError, load_for_training
    res = {}
    for name in ("facescape", "facescape50", "hifi3d", "faceverse"):
        try:
            load_for_training(name)
            res[name] = "AMMESSO (errore)"
        except RoleError as exc:
            res[name] = f"RoleError: {str(exc)[:60]}"
        except Exception as exc:  # noqa: BLE001  (file assente: il ruolo si controlla prima? lo si riporta)
            res[name] = f"{type(exc).__name__}: {str(exc)[:80]}"
        try:
            S.build_sources([name])
            res[name + "_stream"] = "AMMESSO (errore)"
        except ValueError as exc:
            res[name + "_stream"] = f"ValueError: {str(exc)[:60]}"
    out["roles"] = {"results": res, "pass": all("AMMESSO" not in v for v in res.values()),
                    "stream_domains": list(S.DOMAINS)}
    print(f"[targets] ruoli: {out['roles']}", flush=True)


def ring_check(tg, srcs: dict, uni, rng, out: dict) -> None:
    """make_group -> anello -> StreamConsumer (fr e sr, ingresso globale) contro i bersagli."""
    import torch
    import area_v3
    import producer as PR
    import views as VW
    from consumer import StreamConsumer, StreamGT
    from ring import Ring
    from sampler_v3 import DrawCfg
    VW.install_grad_vec()
    cfg = argparse.Namespace(views=2, p_expr=0.6, expr_frac=0.5, k_eig=64, evecs_dtype="fp32", labels=list(VW.LABELS),
                             label_p=np.asarray([1, 1, 1, 1, 1, 0], float) / 5,
                             mm_factor={d: tg.mm_factor(d, s) for d, s in srcs.items()})
    st = {k: 0.0 for k in ("t_ident", "t_mesh", "t_gen", "t_ops", "t_pack")}
    st.update(views=0, failures=0, verts=0, by_label={}, by_domain={})
    groups = [PR.make_group(src, d, rng, cfg, uni, f"{d}/test-{i}", st, tg) for i, (d, src) in enumerate(srcs.items())]
    res = {"failures": st["failures"], "expr_views": int(st.get("expr_views", 0)), "views": int(st["views"])}
    with tempfile.TemporaryDirectory() as tmp:
        ring = Ring(tmp, 0)
        ring.write(groups, 0)
        drawcfg = DrawCfg(p_noise=0.0, sigma_min=5e-4, sigma_max=2e-2, noise_modes=["translation"],
                          noise_mode_probs=[1.0], max_meshes=2)
        for kind in ("fr", "sr"):
            gt = StreamGT(tg.A, 1.0)
            c = StreamConsumer(tmp, reuse=10, prefetch=0, gt=gt, gt_kind=kind, input_norm="global",
                               area_weights="smooth", rot_deg=(0, 0, 0), scale=0.0, global_unit_mm=100.0)
            plan = c.plan(len(groups), 2, drawcfg, np.random.default_rng(0))
            keys = list(plan.subjects)
            rows = np.asarray([gt.name_to_idx[k] for k in keys])
            G = gt[np.ix_(rows, rows)]
            by_key = {g["key"]: g for g in groups}
            V = np.stack([by_key[k][kind] for k in keys]).astype(np.float64)
            ref = np.sqrt(((V[:, None] - V[None]) ** 2).sum(-1)) / np.sqrt(tg.A)
            ls = max(abs(gt.log_s[k] - np.log(by_key[k]["S"])) for k in keys)
            area_err, x_err = [], []
            for key, h, _topo, _m in plan.entries:
                s = c[h]
                X = s["verts"].double()
                F = s["faces"].long()
                tri = X[F]
                area = 0.5 * torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1).sum()
                g = by_key[key]
                vi = int(s["name"].rsplit(".", 1)[1])
                arr, meta = g["views"][vi]
                area_err.append(abs(float(area) * 1e4 / meta["area_mm2"] - 1))
                # X contro i vertici veri in mm (vista del loader x f), centrati coi pesi robusti, / L0
                Vt = torch.as_tensor(arr["verts"], dtype=torch.float64)
                w = area_v3.area_weights("smooth", Vt, F, torch.as_tensor(arr["mass"]).double(),
                                         torch.as_tensor(arr["evecs"]).double().t() if meta["evecs_f"]
                                         else torch.as_tensor(arr["evecs"]).double())
                tri0 = Vt[F]
                a0 = 0.5 * torch.linalg.cross(tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 0], dim=-1).norm(dim=-1).sum()
                Vmm = Vt * np.sqrt(meta["area_mm2"] / float(a0))
                Xd = (Vmm - (w[:, None] * Vmm).sum(0) / w.sum()) / 100.0
                x_err.append(float((X - Xd).abs().max()))
            res[kind] = {"gt_vs_targets_max_abs": float(np.abs(G - ref).max()), "log_s_max_abs": float(ls),
                         "area_rel_err_max": max(area_err), "X_vs_direct_max_abs_L0": max(x_err),
                         "n_views": len(plan.entries)}
            res[kind]["pass"] = bool(res[kind]["gt_vs_targets_max_abs"] < 1e-9 and ls < 1e-9 and max(area_err) < 1e-5
                                     and max(x_err) < 1e-5)
    out["ring"] = res
    print(f"[targets] anello: {res}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-ict-old", type=int, default=100)
    ap.add_argument("--ict-shards", default="0,117")
    ap.add_argument("--gnm-tars", type=int, default=1)
    ap.add_argument("--n-bfm", type=int, default=200)
    ap.add_argument("--n-per-domain", type=int, default=500)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    import sources as S
    from targets import CanonTargets
    import test_gt as TG
    import domains as DM

    rng = np.random.default_rng(a.seed)
    uni = S.Unified()
    srcs = S.build_sources(S.DOMAINS, uni)
    tg = CanonTargets(srcs)
    out = {"tolerance": TOL, "frames": tg.describe(), "files": {}, "pairs": {}, "domains": {}, "expressions": {}}
    sets = {}
    for d, (sids, W) in (("ict", TG.ict_ids_weights(a.n_ict_old, [int(x) for x in a.ict_shards.split(",")], rng)),
                         ("gnm", TG.gnm_ids_weights(a.gnm_tars))):
        sets[d] = (sids, np.stack([srcs[d].neutral_points(w) for w in W]))
    paths = DM.bfm_original_paths()[:a.n_bfm]
    sets["bfm"] = ([p.name.split("_GTready_")[0] for p in paths],
                   np.stack([uni.sp.map("bfm", DM.load_bfm_original(p)[0]) for p in paths]))
    got = against_files(tg, uni, sets, out)
    pair_check(tg, got, out)
    domains_check(tg, srcs, a.n_per_domain, rng, out)
    expr_check(srcs, rng, out)
    roles_check(out)
    ring_check(tg, srcs, uni, rng, out)
    out["pass"] = bool(all(r["pass"] for r in out["files"].values()) and out["roles"]["pass"]
                       and all(r["pass"] for r in out["expressions"].values())
                       and out["ring"]["fr"]["pass"] and out["ring"]["sr"]["pass"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"[targets] ESITO {'PASSA' if out['pass'] else 'FALLISCE'} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
