#!/usr/bin/env python3
"""Soglie di validita', controlli, statistiche nello spazio FR e render dei moltiplicatori (``aug.py``).

    srun -p prioritized --gres=NONE -c 16 --mem=64G -t 02:00:00 env AAU_NV= \\
        aau/run.sh v3_work/mm_aug/stats.py --calibrate            # soglie -> v3_work/mm_aug/thresholds.json
    ... v3_work/mm_aug/stats.py --checks --fr --render            # -> aau/runs/evidence/mm_aug/

  --calibrate  per template, ``N_CAL`` viste PURE come le genera lo stream (espressione nativa con p 0.6): triangoli
               capovolti, degeneri e auto-intersezioni nuove; soglia = p99 (per eccesso). Un campione aumentato non
               deve essere peggiore di quelli che il modello nativo genera;
  --checks     per tipo e template, viste GREZZE (prima del rifiuto): gli stessi controlli, la quota oltre soglia,
               la continuita' del raccordo (strain sugli spigoli per zona, contro lo strain di Delta_A nativo e
               contro il taglio netto senza raccordo), la GT esatta contro quella letta dalla mesh, e per i
               trasferimenti d'espressione che la neutra (FR) resti quella di A;
  --fr         nello spazio FR (GT di E12, ``targets.CanonTargets``): distanze degli ibridi dalle medie e dai pool
               puri dei domini, e la distanza minima dai pool di TEST (HIFI3D, FaceVerse, FaceScape: le 500
               identita' delle viste, FR di E12) di puri, ibridi, bump;
  --subspace   residuo degli ibridi fuori dallo span della base d'identita' di A (piu' i moti rigidi): la parte
               nuova della forma, che nessun coefficiente di A genera;
  --render     10 ibridi e 10 trasferimenti d'espressione in PNG.
Niente dataset su disco: solo JSON e PNG in ``--out``.
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

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "phase0"))

from v3_work.mm_aug import aug as AG  # noqa: E402
from v3_work.mm_aug import transfer as TR  # noqa: E402

OUT = REPO_ROOT / "aau" / "runs" / "evidence" / "mm_aug"
N_CAL = 400
SEED = 20261009
TEST_POOLS = ("hifi3d", "faceverse", "facescape")
CFG = AG.AugConfig(templates=AG.TEMPLATES + ("famos",), validate="none")
_LIB = None


def lib() -> AG.AugLibrary:
    global _LIB
    if _LIB is None:
        _LIB = AG.get_library(CFG)
    return _LIB


def raw_view(args) -> dict:
    """Una vista grezza (nessun rifiuto) del tipo ``kind`` sul template A, coi controlli completi."""
    kind, A, seed, extra = args
    L = lib()
    rng = np.random.default_rng([SEED, seed])
    params = AG.draw_view(rng, kind, A, CFG, L)
    prov = AG._provenance(seed, kind, kind, 0, params, CFG)
    spec = AG.assemble(prov, L, check="full")
    out = {"kind": kind, "A": A, "B": params["identity"].get("B") or (params["expression"] or {}).get("source"),
           "expr": spec["expr"], **{k: spec["checks"][k] for k in ("flipped", "degenerate", "self_intersections_new",
                                                                       "area_ratio_min")}}
    if extra:
        out.update(extra_checks(spec, params, L))
    return out


def extra_checks(spec: dict, params: dict, L: AG.AugLibrary) -> dict:
    """Raccordo, GT esatta contro mesh, neutra invariata nei trasferimenti d'espressione."""
    ident, expr = params["identity"], params["expression"]
    t = L.tpl[ident["A"]]
    out = {}
    Vw, P = AG.build_identity(ident, L)
    # GT esatta (punti della regione) contro quella letta dalla mesh neutra (baricentriche di lavoro)
    out["gt_vs_mesh_mm"] = float(np.linalg.norm(t.to_mm(P - t.u2w(Vw)), axis=1).max())
    if ident["kind"] == "hybrid":
        tb = L.tpl[ident["B"]]
        g = tb.to_mm(tb.src.id_u @ np.asarray(ident["z_B"]))
        out["strain"] = TR.edge_strain(t.rt.apply(g), t.rt)
        out["strain_hard_cut"] = TR.edge_strain(t.rt.hard_cut(g), t.rt)
        out["strain_native_A"] = TR.edge_strain(t.to_mm(t.src.id_w @ np.asarray(ident["z_A"])), t.rt)
    if expr is not None and expr["mode"] == "transfer":
        pure = AG.build_identity({"kind": "pure", "A": ident["A"], "z_A": ident["z_A"]}, L)[1]
        tg = L.targets()
        out["neutral_gt_equal"] = bool(np.array_equal(spec["neutral_points"], pure)
                                       and np.array_equal(tg(ident["A"], spec["neutral_points"])["fr"],
                                                          tg(ident["A"], pure)["fr"]))
        D = t.to_mm(AG.build_expression(expr, ident["A"], L))
        out["expr_disp_mm_region_mean"] = float(np.linalg.norm(D[t.rt.inside], axis=1).mean())
        out["strain"] = TR.edge_strain(D, t.rt)
    if expr is not None and expr["mode"] == "native":
        D = t.to_mm(AG.build_expression(expr, ident["A"], L))
        out["expr_disp_mm_region_mean"] = float(np.linalg.norm(D[t.rt.inside], axis=1).mean())
    return out


def run_pool(jobs: list, n_proc: int) -> list:
    lib()                                                  # carica prima del fork: pagine condivise
    with mp.get_context("fork").Pool(n_proc) as pool:
        return pool.map(raw_view, jobs, chunksize=4)


def q(x, ps=(50, 90, 99, 100)) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} | {"mean": float(x.mean()), "n": int(len(x))}


# ------------------------------------------------------------------------------------------ taratura

def calibrate(n_proc: int) -> dict:
    jobs = [("pure", A, 1_000_000 * i + k, False) for i, A in enumerate(AG.TEMPLATES) for k in range(N_CAL)]
    res = run_pool(jobs, n_proc)
    out = {"_meta": {"n_per_template": N_CAL, "seed": SEED, "rule": "p99 (per eccesso) delle viste pure grezze, "
                     "espressione nativa con p 0.6", "date": time.strftime("%Y-%m-%d")}}
    for A in AG.TEMPLATES:
        r = [x for x in res if x["A"] == A]
        out[A] = {k: int(np.ceil(np.percentile([x[k] for x in r], 99)))
                  for k in ("flipped", "degenerate", "self_intersections_new")}
        out[A]["quantiles"] = {k: q([x[k] for x in r]) for k in ("flipped", "degenerate", "self_intersections_new")}
    AG.THRESHOLDS_JSON.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({A: {k: v for k, v in out[A].items() if k != "quantiles"} for A in AG.TEMPLATES}), flush=True)
    return out


# ------------------------------------------------------------------------------------------ controlli

def checks(n_proc: int, n: int) -> dict:
    th = json.loads(AG.THRESHOLDS_JSON.read_text())
    jobs = [(kind, A, 10_000_000 * (j + 1) + 100_000 * i + k, True)
            for j, kind in enumerate(AG.KINDS) for i, A in enumerate(AG.TEMPLATES) for k in range(n)]
    res = run_pool(jobs, n_proc)
    out = {"thresholds": {A: {k: th[A][k] for k in ("flipped", "degenerate", "self_intersections_new")}
                          for A in AG.TEMPLATES}, "n_per_kind_template": n, "by_kind": {}}
    for kind in AG.KINDS:
        rk = [x for x in res if x["kind"] == kind]
        by_t = {}
        for A in AG.TEMPLATES:
            r = [x for x in rk if x["A"] == A]
            T = th[A]
            over = [x["flipped"] > T["flipped"] or x["degenerate"] > T["degenerate"]
                    or x["self_intersections_new"] > T["self_intersections_new"] for x in r]
            by_t[A] = {"frac_over_threshold": float(np.mean(over)),
                       "frac_over_cheap": float(np.mean([x["flipped"] > T["flipped"] or x["degenerate"] > T["degenerate"]
                                                         for x in r])),
                       **{k: q([x[k] for x in r]) for k in ("flipped", "degenerate", "self_intersections_new")},
                       "gt_vs_mesh_mm": q([x["gt_vs_mesh_mm"] for x in r])}
            if kind == "expr_transfer":
                by_t[A]["neutral_gt_equal_all"] = bool(all(x["neutral_gt_equal"] for x in r))
                by_t[A]["expr_disp_mm_region_mean"] = q([x["expr_disp_mm_region_mean"] for x in r])
                by_t[A]["by_source"] = {B: int(sum(x["B"] == B for x in r)) for B in AG.EXPR_SOURCES}
            if kind == "pure":
                e = [x["expr_disp_mm_region_mean"] for x in r if "expr_disp_mm_region_mean" in x]
                if e:
                    by_t[A]["native_expr_disp_mm_region_mean"] = q(e)
            if kind in ("hybrid", "expr_transfer"):
                by_t[A]["strain"] = strain_summary(r)
        out["by_kind"][kind] = by_t
    return out


def strain_summary(r: list) -> dict:
    """Max per zona su tutti i campioni: il bordo del raccordo (seam) non deve superare l'interno della regione."""
    out = {}
    for key in ("strain", "strain_hard_cut", "strain_native_A"):
        rows = [x[key] for x in r if key in x]
        if not rows:
            continue
        out[key] = {z: {"p99_median": float(np.median([s[z]["p99"] for s in rows if z in s])),
                        "max": float(max(s[z]["max"] for s in rows if z in s))}
                    for z in ("region", "seam", "band", "edge_out") if any(z in s for s in rows)}
    return out


# ------------------------------------------------------------------------------------------ spazio FR

def fr_vectors(L: AG.AugLibrary, groups: dict) -> dict:
    """{nome: (dominio GT, punti (n, 1478, 3))} -> {nome: vettori FR (n, 3m) float64}."""
    tg = L.targets()
    out = {}
    for name, (dom, P) in groups.items():
        out[name] = np.concatenate([tg(dom, P[k:k + 500])["fr"] for k in range(0, len(P), 500)]).astype(np.float64)
    return out


def test_pools(L: AG.AugLibrary) -> dict:
    """FR di E12 dei pool di test (le 500 identita' delle viste): F + rigida robusta, mm."""
    import cgt
    cn = L.targets().cn
    out = {}
    for name in TEST_POOLS:
        P = cgt.native_points(name, cn)
        a = cn.rigid_robust(cn.to_F(P["domain"], P["P"]))["a"]
        out[name] = (L.targets().sq[None] * a).reshape(len(a), -1)
    return out


def dmat(X: np.ndarray, Y: np.ndarray, A: float) -> np.ndarray:
    G = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None] - 2.0 * X @ Y.T
    return np.sqrt(np.clip(G, 0.0, None) / A)


def identities(kind: str, A: str, n: int, seed: int) -> tuple[np.ndarray, list]:
    L = lib()
    rng = np.random.default_rng([SEED, seed])
    P, meta = [], []
    for _ in range(n):
        ident = AG.draw_identity(rng, kind, A, CFG, L)
        P.append(AG.build_identity(ident, L)[1])
        meta.append({k: ident.get(k) for k in ("B", "alpha", "beta")})
    return np.stack(P), meta


def boot_ci(x: np.ndarray, f, n_boot: int = 1000, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    v = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def fr_stats(n: int, n_rbf: int) -> dict:
    L = lib()
    Aw = L.targets().A
    groups, meta = {}, {}
    for i, A in enumerate(AG.TEMPLATES):
        groups[f"pure/{A}"] = (A, identities("pure", A, n, 100 + i)[0])
        P, m = identities("hybrid", A, n, 200 + i)
        groups[f"hybrid/{A}"], meta[f"hybrid/{A}"] = (A, P), m
        groups[f"rbf/{A}"] = (A, identities("rbf", A, n_rbf, 300 + i)[0])
        groups[f"mean/{A}"] = (A, L.tpl[A].src.mean_u[None])
    fam = L.famos.src
    groups["pure/famos"] = ("famos", np.stack([fam.neutral_points(p) for p in fam.persons]))
    Z = fr_vectors(L, groups)
    tests = test_pools(L)
    doms = list(AG.TEMPLATES)
    means = np.concatenate([Z[f"mean/{A}"] for A in doms])
    pure_all = np.concatenate([Z[f"pure/{A}"] for A in doms])
    pure_lab = np.repeat(np.arange(len(doms)), n)
    out = {"n_per_template": n, "n_rbf_per_template": n_rbf, "units": "mm (GT FR di E12)"}
    # distanze fra le medie dei domini (scala di riferimento)
    Dm = dmat(means, means, Aw)
    out["domain_means_mm"] = {f"{a}-{b}": float(Dm[i, j]) for i, a in enumerate(doms) for j, b in enumerate(doms) if i < j}
    # ibridi: distanza dalla media di A, di B, dalla media piu' vicina; puri come riferimento
    hyb = {}
    for i, A in enumerate(doms):
        Zh, m = Z[f"hybrid/{A}"], meta[f"hybrid/{A}"]
        Bi = np.asarray([doms.index(x["B"]) for x in m])
        beta = np.asarray([x["beta"] for x in m])
        Dh = dmat(Zh, means, Aw)
        Dp = dmat(Z[f"pure/{A}"], means, Aw)
        nearest = Dh.argmin(1)
        # NN fra i puri di TUTTI i domini (escluso se stesso per i puri): "fuori" = piu' lontano dei puri fra loro
        nn_h = dmat(Zh, pure_all, Aw).min(1)
        Dpp = dmat(Z[f"pure/{A}"], pure_all, Aw)
        Dpp[np.arange(n), i * n + np.arange(n)] = np.inf
        nn_p = Dpp.min(1)
        hyb[A] = {"d_mean_A_hybrid": q(Dh[:, i]), "d_mean_A_pure": q(Dp[:, i]),
                  "d_mean_B_hybrid": q(Dh[np.arange(n), Bi]),
                  "d_nearest_mean_hybrid": q(Dh.min(1)), "d_nearest_mean_pure": q(Dp.min(1)),
                  "nearest_mean_is_A": float(np.mean(nearest == i)), "nearest_mean_is_B": float(np.mean(nearest == Bi)),
                  "nearest_mean_is_A_pure": float(np.mean(Dp.argmin(1) == i)),
                  "nn_to_pure_pools_hybrid": q(nn_h), "nn_to_pure_pools_pure_loo": q(nn_p),
                  "nn_ratio_median_hybrid_over_pure": float(np.median(nn_h) / np.median(nn_p)),
                  "nn_ratio_ci95": boot_ci(np.stack([nn_h, nn_p], 1),
                                           lambda x: float(np.median(x[:, 0]) / np.median(x[:, 1]))),
                  "frac_closer_to_B_than_A_by_beta": {
                      f"beta<{b1:.1f}" if b0 == 0 else f"{b0:.1f}-{b1:.1f}": float(np.mean(
                          Dh[np.arange(n), Bi][(beta >= b0) & (beta < b1)] < Dh[(beta >= b0) & (beta < b1), i]))
                      for b0, b1 in ((0.0, 0.5), (0.5, 0.8), (0.8, 1.01)) if ((beta >= b0) & (beta < b1)).any()}}
    out["hybrids"] = hyb
    # vicinanza ai pool di test: distanza minima per campione
    sets = {"pure": pure_all, "hybrid": np.concatenate([Z[f"hybrid/{A}"] for A in doms]),
            "rbf": np.concatenate([Z[f"rbf/{A}"] for A in doms]), "famos_pure": Z["pure/famos"]}
    for A in doms:
        sets[f"pure/{A}"] = Z[f"pure/{A}"]
        sets[f"hybrid/{A}"] = Z[f"hybrid/{A}"]
    prox = {}
    for tname, T in tests.items():
        mins = {k: dmat(v, T, Aw).min(1) for k, v in sets.items()}
        p1 = float(np.percentile(mins["pure"], 1))
        pt = {k: {"min": float(v.min()), "p1": float(np.percentile(v, 1)), "p5": float(np.percentile(v, 5)),
                  "median": float(np.median(v)), "n": int(len(v))} for k, v in mins.items()}
        for k in ("hybrid", "rbf"):
            fr = mins[k] < p1
            pt[k]["frac_below_pure_p1"] = float(fr.mean())
            pt[k]["frac_below_pure_p1_ci95"] = boot_ci(fr.astype(float), np.mean)
            pt[k]["min_minus_pure_min_mm"] = float(mins[k].min() - mins["pure"].min())
        Tt = dmat(T, T, Aw)
        np.fill_diagonal(Tt, np.inf)
        pt["test_internal_nn"] = {"median": float(np.median(Tt.min(1))), "min": float(Tt.min())}
        prox[tname] = pt
    out["test_proximity"] = prox
    figure_pca(Z, tests, doms, OUT / "fr_pca.png")
    return out


def subspace_stats(n: int) -> dict:
    """Residuo della neutra ibrida (punti della regione, mm) fuori dallo span della base d'identita' di A piu' i 6 moti
    rigidi infinitesimi (minimi quadrati pesati per area): 0 per i puri di A; per gli ibridi, la parte della
    deformazione che A da solo non genera. Rapporto = RMS residuo / RMS di (neutra - media di A)."""
    L = lib()
    W = L.uni.sp.W
    out = {"n_per_template": n}
    for i, A in enumerate(AG.TEMPLATES):
        t = L.tpl[A]
        M = t.to_mm(t.src.mean_u)
        Bid = np.einsum("vdk,ed->vek", t.src.id_u, t.R) * t.u                      # (n, 3, k) mm, frame FLAME
        rot = np.stack([np.cross(np.eye(3)[a], M) for a in range(3)], axis=-1)    # (n, 3, 3)
        tr = np.broadcast_to(np.eye(3)[None], (len(M), 3, 3))
        Bf = np.concatenate([Bid, rot, tr], axis=-1).reshape(-1, Bid.shape[2] + 6)
        sw = np.repeat(np.sqrt(W), 3)[:, None]
        Q, _ = np.linalg.qr(sw * Bf)
        res = {}
        for kind in ("pure", "hybrid"):
            P, meta = identities(kind, A, n, 400 + 10 * i + (kind == "hybrid"))
            D = (t.to_mm(P) - M[None]).reshape(n, -1) * sw[:, 0][None]
            R = D - (D @ Q) @ Q.T
            rms_r, rms_d = np.sqrt((R ** 2).sum(1)), np.sqrt((D ** 2).sum(1))
            res[kind] = {"residual_rms_mm": q(rms_r), "ratio": q(rms_r / rms_d)}
            if kind == "hybrid":
                beta = np.asarray([m["beta"] for m in meta])
                res[kind]["ratio_by_beta"] = {f"{b0:.1f}-{b1:.1f}": float(np.median((rms_r / rms_d)[(beta >= b0) & (beta < b1)]))
                                              for b0, b1 in ((0.0, 0.5), (0.5, 0.8), (0.8, 1.01))}
                res[kind]["by_B"] = {B: float(np.median((rms_r / rms_d)[[m["B"] == B for m in meta]]))
                                     for B in AG.TEMPLATES if B != A}
        out[A] = res
    return out


def figure_pca(Z: dict, tests: dict, doms: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = np.concatenate([Z[f"pure/{A}"] for A in doms] + [Z[f"hybrid/{A}"] for A in doms])
    mu = base.mean(0)
    _, _, Vt = np.linalg.svd(base - mu, full_matrices=False)
    pr = lambda X: (X - mu) @ Vt[:2].T  # noqa: E731
    cols = {"bfm2019": "tab:red", "ict": "tab:cyan", "gnm": "tab:purple", "flame2020": "tab:green"}
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    for ax, which in zip(axs, ("pure", "hybrid")):
        for A in doms:
            Y = pr(Z[f"pure/{A}"])
            ax.scatter(Y[:, 0], Y[:, 1], s=2, c=cols[A], alpha=0.25 if which == "hybrid" else 0.5, label=f"puro {A}")
        if which == "hybrid":
            for A in doms:
                Y = pr(Z[f"hybrid/{A}"])
                ax.scatter(Y[:, 0], Y[:, 1], s=4, marker="x", c=cols[A], alpha=0.6, label=f"ibrido su {A}")
        for name, T in tests.items():
            Y = pr(T)
            ax.scatter(Y[:, 0], Y[:, 1], s=6, c="k", marker={"hifi3d": "^", "faceverse": "s", "facescape": "o"}[name],
                       alpha=0.5, label=f"test {name}")
        for A in doms:
            Y = pr(Z[f"mean/{A}"])
            ax.scatter(Y[:, 0], Y[:, 1], s=150, c=cols[A], edgecolors="k", marker="*")
        ax.set_title("FR (mm), PCA sui puri + ibridi: " + ("solo puri" if which == "pure" else "puri e ibridi"))
        ax.legend(markerscale=3, fontsize=7, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ------------------------------------------------------------------------------------------ render

def render_examples(out_dir: Path, n: int = 10) -> None:
    from PIL import Image, ImageDraw
    from render_mesh import _yaw_rotate, render_mesh
    L = lib()

    def tile(spec, yaw=0.0, size=224):
        V = spec["V"] * np.array([1.0, -1.0, -1.0])
        if yaw:
            V = _yaw_rotate(V, yaw, (V.max(0) + V.min(0)) / 2)
        return render_mesh(V, spec["F"], size=size)

    def label(img, txt):
        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((4, 4), txt, fill=(255, 255, 0))
        return np.asarray(im)

    def prov(ident, expr):
        return AG._provenance(0, "render", "render", 0, {"identity": ident, "expression": expr}, CFG)

    rng = np.random.default_rng([SEED, 77])
    rows = []
    for k in range(n):
        A = AG.TEMPLATES[k % 4]
        while True:
            p = AG.draw_view(rng, "hybrid", A, AG.AugConfig(p_expr=0.0), L)
            ident = p["identity"]
            spec = AG.assemble(prov(ident, None), L, check="cheap")
            if AG.valid(spec, L):
                break
        pa = AG.assemble(prov({"kind": "pure", "A": A, "z_A": ident["z_A"]}, None), L)
        pb = AG.assemble(prov({"kind": "pure", "A": ident["B"], "z_A": ident["z_B"]}, None), L)
        rows.append(np.concatenate([label(tile(pa), f"A={A} (z_A)"), label(tile(pb), f"B={ident['B']} (z_B)"),
                                    label(tile(spec), f"ibrido, a={ident['alpha']:.2f} b={ident['beta']:.2f}"),
                                    label(tile(spec, 60.0), "ibrido, yaw 60")], axis=1))
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_dir / "hybrids_10.png")
    rows = []
    for k in range(n):
        A = AG.TEMPLATES[k % 4]
        while True:
            p = AG.draw_view(rng, "expr_transfer", A, CFG, L)
            spec = AG.assemble(prov(p["identity"], p["expression"]), L, check="cheap")
            if AG.valid(spec, L) and (k < 6 or p["expression"]["source"] == "famos"):
                break
        e = p["expression"]
        neutral = AG.assemble(prov(p["identity"], None), L)
        if e["source"] == "famos":
            src = AG.assemble(prov({"kind": "pure", "A": "famos", "person": e["person"]},
                                   {"mode": "frame", "frame": e["frame"]}), L)
            txt = f"FaMoS {e['person'][-3:]} fot. {e['frame']}"
        else:
            src = AG.assemble(prov({"kind": "pure", "A": e["source"], "z_A": [0.0] * L.tpl[e["source"]].model.n_id},
                                   {"mode": "native", "source": e["source"], "coef": e["coef"]}), L)
            txt = f"espressione di {e['source']} (media)"
        rows.append(np.concatenate([label(tile(neutral), f"A={A} neutra"), label(tile(src), txt),
                                    label(tile(spec), "trasferita su A"), label(tile(spec, 60.0), "yaw 60")], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_dir / "expr_transfer_10.png")


# ------------------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--fr", action="store_true")
    ap.add_argument("--subspace", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--n-checks", type=int, default=150)
    ap.add_argument("--n-fr", type=int, default=2000)
    ap.add_argument("--n-rbf", type=int, default=500)
    ap.add_argument("--n-sub", type=int, default=500)
    ap.add_argument("--n-proc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.calibrate:
        calibrate(a.n_proc)
        print(f"[mm_aug] soglie in {AG.THRESHOLDS_JSON} ({time.time() - t0:.0f}s)", flush=True)
    if a.checks:
        r = checks(a.n_proc, a.n_checks)
        (a.out / "checks.json").write_text(json.dumps(r, indent=1) + "\n")
        print(f"[mm_aug] controlli in {a.out / 'checks.json'} ({time.time() - t0:.0f}s)", flush=True)
    if a.fr:
        r = fr_stats(a.n_fr, a.n_rbf)
        (a.out / "fr_stats.json").write_text(json.dumps(r, indent=1) + "\n")
        print(f"[mm_aug] statistiche FR in {a.out / 'fr_stats.json'} ({time.time() - t0:.0f}s)", flush=True)
    if a.subspace:
        r = subspace_stats(a.n_sub)
        (a.out / "subspace.json").write_text(json.dumps(r, indent=1) + "\n")
        print(f"[mm_aug] sottospazi in {a.out / 'subspace.json'} ({time.time() - t0:.0f}s)", flush=True)
    if a.render:
        render_examples(a.out)
        print(f"[mm_aug] render in {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
