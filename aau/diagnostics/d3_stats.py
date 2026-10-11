#!/usr/bin/env python3
"""D3 con l'emendamento 1 (PROTOCOL_D3.md + PROTOCOL_D3_emendamento_1.md): teste metriche sull'embedding congelato in
uno schema leave-one-generator-out (LOGO), controlli e letture. DIAGNOSTICA, non un metodo.

    v3_work/unified_gt/run.sh aau/diagnostics/d3_stats.py --workers 64          (d3.sbatch, passo stats)

Sorgenti (``SOURCES``, un ``d3_head.Block`` per sorgente, x = u x dp_per_unit, senza crop): bfm, ict, gnm (held-out
della calibrazione), flame2023_s1 (D1), i pool NON valutati di FaceScape, HIFI3D, FaceVerse (d3_pools.py +
d3_embed.sbatch), famos (FaMoS TRAIN). Per bersaglio T (``TARGETS`` + famos) e variante (tutte tranne T, e senza FaMoS):
(i) testa appresa con CV per soggetto dentro le sorgenti (fold per sorgente), scelta, rifit, c; (ii) Mahalanobis
intra-soggetto (Ledoit-Wolf dei residui dalla media del soggetto, media sulle sorgenti); (iii) d_P e d_F calibrata;
CORAL esplorativa (covarianza del bersaglio dal suo pool non valutato).

Insiemi di test (``TESTS``): FaceScape, HIFI3D, FaceVerse neutra e con espressioni (righe, GT, semi e conteggi di
``fact_paired``, importato), FLAME (righe e conteggi di D1), FaMoS (righe di ``diag.rows``, conteggi per persona).
Righe incrociate, original-original e mediate sulle etichette; Spearman pesato c_a c_b per replica; delta appaiati;
curva sul numero di sorgenti (sottoinsiemi di 1, 2, 4 sorgenti, iperparametri della testa con tutte le sorgenti).

Uscite: ``d3_cv.csv``, ``d3_heads.csv``, ``d3_spearman.csv``, ``d3_delta.csv``, ``d3_curve.csv``,
``d3/controls.json``, ``d3/readings.json``, ``d3/heads.npz``, ``d3/reps.npz``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import d3_head as H  # noqa: E402
import diag  # noqa: E402

sys.path.insert(0, str(diag.REPO / "v3_work/trainer/tools"))

D3 = diag.EV / "d3"
SOURCES = ("bfm", "ict", "gnm", "flame2023_s1", "facescape", "hifi3d", "faceverse", "famos")
POOLS = ("facescape", "hifi3d", "faceverse")
TARGETS = ("facescape", "hifi3d", "faceverse", "flame2023_s1")              # bersagli primari
# insieme di test -> bersaglio (la sorgente esclusa); i primari sono le prime quattro chiavi
TESTS = {"facescape": "facescape", "hifi3d": "hifi3d", "faceverse_neutral": "faceverse", "flame2023_s1": "flame2023_s1",
         "faceverse": "faceverse", "famos": "famos"}
PRIMARY = ("facescape", "hifi3d", "faceverse_neutral", "flame2023_s1")
FP_VIEWS = ("facescape", "hifi3d", "faceverse_neutral", "faceverse")
GT_SET = {"facescape": "facescape", "hifi3d": "hifi3d", "faceverse_neutral": "faceverse", "faceverse": "faceverse"}
METHODS = ("dP", "i", "i_nf", "ii", "ii_nf", "coral")
SEED_FAMOS_BOOT = 20261123
GT_EVAL = diag.REPO / "datasets/CANONICAL_GT/eval"
NEUTRAL_EMB = diag.REPO / "aau/runs/evidence/faceverse_neutral/embed"
BP_OUT = diag.REPO / "aau/runs/evidence/baselines_param"
B_MODELS = ("gnm", "flame2023")
B_LABEL = {("gnm", "sr"): "GNM (visto) vB, mesh d'identita' SR", ("gnm", "fr"): "GNM (visto) vB, mesh d'identita' FR",
           ("flame2023", "sr"): "FLAME 2023 Open vB, mesh d'identita' SR",
           ("flame2023", "fr"): "FLAME 2023 Open vB, mesh d'identita' FR"}
SR_MIN, FR_MIN, N_SR = 0.05, -0.03, 3                         # letture (emendamento 1, sez. 6)
CURVE_K = (1, 2, 4)
_B: dict = {}           # blocchi delle sorgenti per (braccio, sorgente), ereditati dai worker (fork)
_TS: dict = {}          # insiemi di test
_J: dict = {}           # colonne per il bootstrap


def settings() -> dict:
    """{chiave: sorgenti} delle teste: per ogni bersaglio primario tutte tranne T (``T|all``) e senza FaMoS
    (``T|nofamos``); FaMoS come bersaglio secondario (``famos|all``)."""
    out = {}
    for t in TARGETS:
        out[f"{t}|all"] = tuple(s for s in SOURCES if s != t)
        out[f"{t}|nofamos"] = tuple(s for s in SOURCES if s not in (t, "famos"))
    out["famos|all"] = tuple(s for s in SOURCES if s != "famos")
    for k, srcs in out.items():                               # K6: la sorgente del bersaglio fuori dalla sua testa
        if k.split("|")[0] in srcs:
            raise RuntimeError(f"{k}: il bersaglio fra le sorgenti")
    return out


def gt_of(col: str) -> tuple[str, ...]:
    """Le GT con cui si legge una colonna: d_h (``shape``) con SR e FR (FR senza taglia), d_F e vb_fr con FR."""
    if col.endswith("shape"):
        return ("sr", "fr")
    return ("sr",) if col.endswith("vb_sr") else ("fr",)


# ------------------------------------------------------------------------------------------ embedding e GT

def arm_store(arm: str, dom: str) -> Path | None:
    """embeddings.npz dello store ufficiale di ``fact_paired`` (dom = hifi, devfs, fv, fvn) per il braccio."""
    import fact_paired as fp
    for pre, v, e in fp.MODELS:
        if pre == arm:
            return fp.embeddings(dom, v, e)
    raise KeyError(arm)


def arm_tag(arm: str) -> str:
    """``factorizedc3m_e123`` -> ``factorizedc3mfulle123`` (nome dello store di zs_zeroshot)."""
    import fact_paired as fp
    for pre, v, e in fp.MODELS:
        if pre == arm:
            return f"{v}fulle{e}"
    raise KeyError(arm)


def store_by_names(path: Path, names: list[str] | None = None) -> tuple[list[str], np.ndarray, Path]:
    """(nomi ``<sid>_GTready_<etichetta>.npz``, Z, checkpoint) di uno store; senza ``names`` tutte le mesh senza crop."""
    with np.load(path, allow_pickle=True) as z:
        if "files" in z.files:
            files = [Path(str(f)).name for f in z["files"]]
        else:
            files = [f"{s}_GTready_{t}.npz" for s, t in zip(z["subjects"], z["topologies"])]
        Z = np.asarray(z["Z"], np.float64)
        ckpt = Path(str(z["checkpoint"]))
    pos = {f: k for k, f in enumerate(files)}
    if names is None:
        names = sorted(f for f in files if not f.endswith("_GTready_crop.npz"))
    miss = [n for n in names if n not in pos]
    if miss:
        raise SystemExit(f"{path}: mancano {len(miss)} mesh (es. {miss[:2]})")
    return names, Z[[pos[n] for n in names]], ckpt


def gt_matrix(path: Path, kind: str = "sr") -> tuple[np.ndarray, np.ndarray]:
    """(names, D in unita' fisiche): ``D_orig`` x unita' del json, o ``D_<kind>`` gia' fisica (GT di D1 e di FaMoS)."""
    with np.load(path, allow_pickle=True) as z:
        names = np.asarray([str(s) for s in z["names"]])
        if f"D_{kind}" in z.files:
            return names, np.asarray(z[f"D_{kind}"], np.float64)
        D = np.asarray(z["D_orig"], np.float64)
    key = "dP_per_unit" if kind == "sr" else "mm_per_unit"
    return names, D * float(json.loads(path.with_suffix(".json").read_text())[key])


def make_block(names: list[str], Z: np.ndarray, gt_names, G: np.ndarray, dpu: float, name: str) -> H.Block:
    """Un blocco: soggetti nell'ordine di ``diag.subject_index``, G ristretta e riordinata, x = u x dp_per_unit."""
    subj, subjects = diag.subject_index(names)
    pos = {str(s): k for k, s in enumerate(gt_names)}
    ii = np.asarray([pos[s] for s in subjects])
    lab = np.asarray([diag.LABELS.index(diag.split_name(n)[1]) for n in names])
    Gs = np.asarray(G, np.float64)[np.ix_(ii, ii)]
    if not (np.allclose(Gs, Gs.T) and np.all(np.diag(Gs) == 0) and np.all(Gs[~np.eye(len(ii), dtype=bool)] > 0)):
        raise SystemExit(f"{name}: GT non simmetrica o con zeri fuori diagonale")
    return H.Block(Z[:, 1:] * dpu, subj, lab, Gs, name)


def pool_info(pool: str) -> dict:
    return json.loads((D3 / "pools" / pool / "subjects.json").read_text())


def source_data(arm: str, src: str) -> tuple[list[str], np.ndarray, Path, np.ndarray, np.ndarray]:
    """(nomi, Z, checkpoint, nomi della GT, GT d_P) di una sorgente."""
    if src in ("bfm", "ict", "gnm"):
        names, Z, c = store_by_names(diag.CALIB_EMB / arm / "embeddings.npz", diag.set_names(src))
        return (names, Z, c) + gt_matrix(diag.GT_SR)
    if src == "flame2023_s1":
        names, Z, c = store_by_names(diag.D1 / "emb" / arm / "embeddings.npz", diag.set_names(src))
        return (names, Z, c) + gt_matrix(diag.D1 / "gt_flame2023_s1.npz")
    if src == "famos":
        names, Z, c = store_by_names(D3 / "famos" / "emb" / arm / "embeddings.npz")
        return (names, Z, c) + gt_matrix(D3 / "famos" / "gt_famos.npz")
    info = pool_info(src)                                     # pool non valutato: solo le 200 sorgenti
    names = sorted(f"{s}_GTready_{lab}.npz" for s in info["source"] for lab in diag.LABELS)
    names, Z, c = store_by_names(D3 / "pools" / src / "emb" / arm / "embeddings.npz", names)
    return (names, Z, c) + gt_matrix(GT_EVAL / f"{info['gt']}_sr.npz")


EXPECT = {"flame2023_s1": 200, "famos": 80, "facescape": 200, "hifi3d": 200, "faceverse": 200}


def build_sources(arm: str, info: dict) -> None:
    for src in SOURCES:
        names, Z, c, gn, G = source_data(arm, src)
        if c.resolve() != info["ckpt"][arm].resolve():               # K4: stesso checkpoint della calibrazione
            raise SystemExit(f"{arm} {src}: checkpoint {c}, atteso {info['ckpt'][arm]}")
        if not np.isfinite(Z).all():
            raise SystemExit(f"{arm} {src}: embedding non finiti")
        b = make_block(names, Z, gn, G, info["dpu"][arm], src)
        n_subj = EXPECT.get(src, 100)
        if len(b.G) != n_subj or len(names) != n_subj * len(diag.LABELS):
            raise SystemExit(f"{arm} {src}: {len(b.G)} soggetti e {len(names)} mesh, attesi {n_subj} x 5")
        _B[(arm, src)] = b
        info.setdefault("source_sizes", {})[src] = [int(len(b.G)), int(len(names))]


# ------------------------------------------------------------------------------------------ insiemi di test

def b_columns(view: str, df) -> dict:
    """{B-<modello>|vb_<gt>: colonna sulle righe} dai fit_e1.npz della variante B."""
    out = {}
    for m in B_MODELS:
        with np.load(BP_OUT / view / m / "fit_e1.npz") as z:
            pos = {k: i for i, k in enumerate(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))}
            pa = np.asarray([pos[k] for k in zip(df["subject_a"], df["topology_a"])])
            pb = np.asarray([pos[k] for k in zip(df["subject_b"], df["topology_b"])])
            for g in ("sr", "fr"):
                out[f"B-{m}|vb_{g}"] = np.asarray(z[f"D_vb_{g}"], np.float64)[pa, pb]
    return out


def view_data(view: str) -> dict:
    """Un insieme di test di ``fact_paired``: righe, GT, seme, conteggi, mesh, embedding dei bracci e B."""
    import fact_paired as fp
    if view == "faceverse_neutral":                       # come bp_paired.main: in memoria, fact_paired non si tocca
        fp.VIEWS["faceverse_neutral"] = ("fvn", "mesh_pair_nocrop")
        fp.FORM_DIR["fvn"] = os.path.relpath(NEUTRAL_EMB, fp.EVAL)
    src = "faceverse" if view == "faceverse_neutral" else view
    df, idx, seed = fp.rows_for(src)
    Mc = fp.model_columns(view, idx)
    df = fp.be.add_columns(df, Mc, idx)
    bl = ["oracle_size"] if view == "faceverse_neutral" else [b for b in fp.BASELINES if b in df]
    cols = {m: df[m].to_numpy(np.float64) for m in list(Mc) + bl}
    cols.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in fp.GTS})
    Bc = b_columns(view, df)
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))     # prima della maschera (fact_paired)
    s2i = {s: i for i, s in enumerate(subjects)}
    sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
    mask = (sa != sb) & np.all([np.isfinite(v) for v in list(cols.values()) + list(Bc.values())], axis=0)
    meshes = [(s, t) for s in subjects for t in diag.LABELS]
    mpos = {m: k for k, m in enumerate(meshes)}
    d = df[mask]
    out = {"name": view, "seed": int(seed), "subjects": [str(s) for s in subjects], "meshes": meshes,
           "n_rows_all": int(len(df)), "sa": sa[mask], "sb": sb[mask],
           "ma": np.asarray([mpos[k] for k in zip(d["subject_a"], d["topology_a"])]),
           "mb": np.asarray([mpos[k] for k in zip(d["subject_b"], d["topology_b"])]),
           "gt": {g: cols[f"gt_{g}"][mask] for g in fp.GTS}, "B": {k: v[mask] for k, v in Bc.items()}, "Z": {},
           "ckpt": {}, "missing": [], "stores": {}}
    out["Gsub"] = {}
    for g in ("sr", "fr"):                                   # GT per soggetto (righe original e mediate)
        gn, G = gt_matrix(GT_EVAL / f"{GT_SET[view]}_{g}.npz", g)
        pos = {s: k for k, s in enumerate(gn)}
        ii = np.asarray([pos[s] for s in out["subjects"]])
        out["Gsub"][g] = G[np.ix_(ii, ii)]
    dom = fp.VIEWS[view][0]
    for arm in diag.ARMS:
        p = arm_store(arm, dom)
        if p is None and view == "faceverse_neutral":         # C3M e123 neutro: d3_embed.sbatch (passo fvn)
            hits = sorted((D3 / "fvn" / "embed").glob(f"data_*/scale_v3{arm_tag(arm)}*/zs_zeroshot/embeddings.npz"))
            p = hits[0] if hits else None
        if p is None:
            out["missing"].append(arm)
            continue
        with np.load(p, allow_pickle=True) as z:
            keys = list(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))
            kpos = {k: i for i, k in enumerate(keys)}
            out["Z"][arm] = np.asarray(z["Z"], np.float64)[[kpos[m] for m in meshes]]
            out["ckpt"][arm] = Path(str(z["checkpoint"]))
        out["stores"][arm] = str(p.relative_to(diag.REPO))
    out["counts"] = fp_counts(len(subjects), seed)
    return out


def fp_counts(n: int, seed: int, n_boot: int = diag.N_BOOT) -> np.ndarray:
    """(1 + n_boot, n): i conteggi di ``fact_paired.main`` (riga 0 = tutti 1)."""
    rng = np.random.default_rng(seed)
    return np.stack([np.ones(n, dtype=np.int64)] +
                    [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)])


def set_data(name: str, info: dict) -> dict:
    """FLAME (righe e conteggi di D1) o FaMoS (righe di ``diag.rows``, conteggi per persona) come insieme di test."""
    gpath = diag.D1 / "gt_flame2023_s1.npz" if name == "flame2023_s1" else D3 / "famos" / "gt_famos.npz"
    out = {"name": name, "Z": {}, "ckpt": {}, "missing": [], "stores": {}, "B": {}}
    for arm in diag.ARMS:
        if name == "flame2023_s1":
            names, Z, c = store_by_names(diag.D1 / "emb" / arm / "embeddings.npz", diag.set_names(name))
        else:
            names, Z, c = store_by_names(D3 / "famos" / "emb" / arm / "embeddings.npz")
        out["Z"][arm], out["ckpt"][arm] = Z, c
        out["stores"][arm] = str((diag.D1 / "emb" / arm if name == "flame2023_s1" else D3 / "famos" / "emb" / arm)
                                 .relative_to(diag.REPO))
    subj, subjects = diag.subject_index(names)
    i, j = diag.rows(names)
    out.update(subjects=subjects, meshes=[diag.split_name(n) for n in names], sa=subj[i], sb=subj[j], ma=i, mb=j,
               n_rows_all=int(len(i)), Gsub={})
    out["gt"] = {}
    for g in ("sr", "fr"):
        gn, G = gt_matrix(gpath, g)
        pos = {s: k for k, s in enumerate(gn)}
        ii = np.asarray([pos[s] for s in subjects])
        out["Gsub"][g] = G[np.ix_(ii, ii)]
        out["gt"][g] = out["Gsub"][g][out["sa"], out["sb"]]
    if name == "flame2023_s1":
        out["counts"], out["seed"] = diag.boot_counts(len(subjects), diag.SETS[name]["group"]), \
            f"D1: SeedSequence([{diag.SEED_BOOT}, {diag.SETS[name]['group']}])"
    else:
        out["counts"], out["seed"] = diag.boot_counts(len(subjects), 0, seed=SEED_FAMOS_BOOT), \
            f"SeedSequence([{SEED_FAMOS_BOOT}, 0])"
    return out


def mesh_index(T: dict) -> tuple[np.ndarray, np.ndarray]:
    """(indice della mesh original per soggetto (n,), indici delle 5 mesh per soggetto (n, 5))."""
    pos = {m: k for k, m in enumerate(T["meshes"])}
    orig = np.asarray([pos[(s, "original")] for s in T["subjects"]])
    grp = np.asarray([[pos[(s, lab)] for lab in diag.LABELS] for s in T["subjects"]])
    return orig, grp


def rowsets(T: dict) -> dict:
    """{nome: (sa, sb, gt, Z per braccio, indici di mesh a, b)}: righe incrociate, original-original, mediate."""
    out = {"cross": (T["sa"], T["sb"], T["gt"], T["Z"], T["ma"], T["mb"])}
    if T["name"] == "faceverse":                                # con espressioni: solo le righe incrociate
        return out
    orig, grp = mesh_index(T)
    i, j = np.triu_indices(len(T["subjects"]), 1)
    gt = {g: T["Gsub"][g][i, j] for g in ("sr", "fr")}
    out["orig"] = (i, j, gt, {a: Z[orig] for a, Z in T["Z"].items()}, i, j)
    out["lavg"] = (i, j, gt, {a: Z[grp].mean(1) for a, Z in T["Z"].items()}, i, j)
    return out


# ------------------------------------------------------------------------------------------ CV e teste

def _cv(task):
    arm, key, srcs, k, q, f = task
    blocks = [_B[(arm, s)] for s in srcs]
    fo = H.folds([len(b.G) for b in blocks], q, [SOURCES.index(s) for s in srcs])
    fit_b, val_b = H.split(blocks, fo, f)
    r, lam = H.CONFIGS[k]
    t0 = time.time()
    head = H.fit(fit_b, r, lam)
    out = {"task": (arm, key, k, q, f), "score": H.score(head, val_b), "nit": head.nit, "success": head.success,
           "seconds": time.time() - t0}
    if r == 0:                                                # K5: r = 0 ha i ranghi di d_P
        out["k5"] = abs(out["score"] - H.score(H.Head(1.0, np.zeros((0, head.W.shape[1])), 0, 0.0, 1.0), val_b))
    return out


def _refit(task):
    """Testa (i) con la configurazione data su tutte le sorgenti (q = -1) o su un fold, piu' c dalle coppie di fit."""
    arm, key, srcs, r, lam, q, f = task
    blocks = [_B[(arm, s)] for s in srcs]
    if q >= 0:
        blocks, _ = H.split(blocks, H.folds([len(b.G) for b in blocks], q, [SOURCES.index(s) for s in srcs]), f)
    head = H.fit(blocks, r, lam)
    return task, head, H.calib_c(lambda b, i, j: head.dist(b.X[i], b.X[j]), blocks)


def within_head(arm: str, srcs) -> tuple[H.Linear, float]:
    """Testa (ii): Mahalanobis intra-soggetto dalle sorgenti, c dalle loro coppie."""
    blocks = [_B[(arm, s)] for s in srcs]
    m = H.Linear(H.sym_pow(H.within_cov(blocks), -0.5))
    return m, H.calib_c(lambda b, i, j: m.dist(b.X[i], b.X[j]), blocks)


def coral_heads(arm: str) -> dict:
    """CORAL esplorativa: {bersaglio: (Linear, c)} con riferimento bfm, ict, gnm e la covarianza del bersaglio dalle
    u del suo pool NON valutato (le 200 sorgenti), c dalla sorgente con la sua A per generatore."""
    ref = [_B[(arm, s)] for s in ("bfm", "ict", "gnm")]
    cov = {b.name: H.ledoit_wolf(b.X)[0] for b in ref}
    S_ref = np.mean(list(cov.values()), axis=0)
    A_src = {b.name: H.coral(S_ref, cov[b.name]) for b in ref}
    c = H.calib_c(lambda b, i, j: np.linalg.norm((b.X[i] - b.X[j]) @ A_src[b.name].T, axis=1), ref)
    return {p: (H.Linear(H.coral(S_ref, H.ledoit_wolf(_B[(arm, p)].X)[0])), c) for p in POOLS}


def _curve(task):
    """Un sottoinsieme di sorgenti: (i) con la configurazione data e (ii); delta SR puntuale sulle righe incrociate."""
    arm, target, test, subset, r, lam = task
    from scipy.stats import spearmanr
    T = _TS[test]
    X = T["Z"][arm][:, 1:] * _TS["_dpu"][arm]
    Xa, Xb = X[T["ma"]], X[T["mb"]]
    g = T["gt"]["sr"]
    ref = spearmanr(np.linalg.norm(Xa - Xb, axis=1), g).correlation
    blocks = [_B[(arm, s)] for s in subset]
    head = H.fit(blocks, r, lam)
    m = H.Linear(H.sym_pow(H.within_cov(blocks), -0.5))
    return {"arm": arm, "target": target, "test": test, "k": len(subset), "sources": "+".join(subset),
            "delta_i": float(spearmanr(head.dist(Xa, Xb), g).correlation - ref),
            "delta_ii": float(spearmanr(m.dist(Xa, Xb), g).correlation - ref), "rho_dP": float(ref)}


# ------------------------------------------------------------------------------------------ colonne e repliche

def columns(T: dict, rs: tuple, heads: dict, cal: dict, info: dict, fold: dict | None = None) -> dict:
    """{colonna: valori sulle righe del rowset}: d_P e d_F calibrata, teste (i), (ii), CORAL e B (righe incrociate di
    fact_paired). ``fold``: solo le teste di fold {(braccio, chiave): {tag: (testa, c)}}."""
    sa, sb, gt, Zs, ia, ib = rs
    target = TESTS[T["name"]]
    cols = {} if fold is not None or ia is not T["ma"] else dict(T["B"])
    for arm, Z in Zs.items():
        S = np.exp(Z[:, 0])
        X = Z[:, 1:] * info["dpu"][arm]
        Sa, Sb, Xa, Xb = S[ia], S[ib], X[ia], X[ib]

        def form(d, c):
            return np.sqrt((Sa - Sb) ** 2 + Sa * Sb * (c * d) ** 2)

        if fold is not None:
            for var in ("all", "nofamos"):
                for tag, (head, c) in fold.get((arm, f"{target}|{var}"), {}).items():
                    m = f"{'i' if var == 'all' else 'i_nf'}{tag}"
                    d = head.dist(Xa, Xb)
                    cols[f"{arm}|{m}|shape"], cols[f"{arm}|{m}|form"] = d, form(d, c)
            continue
        dP = np.linalg.norm(Xa - Xb, axis=1)
        cols[f"{arm}|dP|shape"], cols[f"{arm}|dP|form"] = dP, form(dP, cal[arm])
        for m, key in (("i", f"{target}|all"), ("i_nf", f"{target}|nofamos"), ("ii", f"{target}|all"),
                       ("ii_nf", f"{target}|nofamos"), ("coral", target)):
            h = heads.get((arm, m.split("_")[0], key))
            if h is None:
                continue
            d = h[0].dist(Xa, Xb)
            cols[f"{arm}|{m}|shape"], cols[f"{arm}|{m}|form"] = d, form(d, h[1])
    bad = [k for k, v in cols.items() if not np.isfinite(v).all()]
    if bad:
        raise SystemExit(f"{T['name']}: colonne non finite {bad[:4]}")
    return cols


def _rep(task):
    """Una replica: ranghi pesati di ogni colonna, Pearson con i ranghi delle sue GT (= ``diag.wspearman``)."""
    key, b = task
    J = _J[key]
    c = J["counts"][b]
    w = c[J["sa"]] * c[J["sb"]]
    k = w > 0
    w = w[k]
    rg = {g: diag.wranks(x[k], w) for g, x in J["gt"].items()}
    out = {}
    for m, x in J["cols"].items():
        r = diag.wranks(x[k], w)
        for g in gt_of(m):
            out[(m, g)] = diag.wpearson(r, rg[g], w)
    return out


# ------------------------------------------------------------------------------------------ controlli e letture

def k1(V: dict, tests: dict) -> dict:
    """Riferimenti dei bracci e B contro factorized_paired.csv, paired_e1.csv (FaceVerse neutra) e d1_spearman.csv
    (FLAME): punto, IC, righe."""
    pub = {}
    for path, doms in ((diag.TV3 / "factorized_paired.csv", ("hifi3d", "facescape", "faceverse")),
                       (BP_OUT / "paired_e1.csv", ("faceverse_neutral", "hifi3d", "facescape", "faceverse"))):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r["domain"] in doms and r["kind"] == "rho" and r["baseline"] == "-":
                    pub.setdefault((r["domain"], r["gt"], r["arm"], r["distance"]),
                                   (float(r["arm_point"]), float(r["arm_ci_low"]), float(r["arm_ci_high"]),
                                    int(r["n_rows"]), path.name))
    with open(diag.EV / "d1_spearman.csv") as fh:
        for r in csv.DictReader(fh):
            if r["set"] == "flame2023_s1":
                arm, _, dist = r["method"].partition("|")
                pub[("flame2023_s1", r["gt"], arm, dist)] = (float(r["rho"]), float(r["ci_low"]), float(r["ci_high"]),
                                                             int(r["n_rows"]), "d1_spearman.csv")
    out, worst = {}, 0.0
    for t, Tn in tests.items():
        if t == "famos":
            continue
        Vt = V[(t, "cross")]
        cells = [(f"{a}|dP|shape", "sr", a, "shape") for a in Tn["Z"]]
        cells += [(f"{a}|dP|form", "fr", a, "form_cal") for a in Tn["Z"]]
        if Tn["B"]:
            cells += [(f"B-{m}|vb_{g}", g, "baseline", B_LABEL[(m, g)]) for m in B_MODELS for g in ("sr", "fr")]
        for col, g, arm, dist in cells:
            hit = pub.get((t, g, arm, dist))
            if hit is None:
                out[f"{t}|{col}|{g}"] = "nessun riferimento pubblicato"
                continue
            p, lo, hi, _ = diag.ci(Vt[(col, g)])
            diff = max(abs(p - hit[0]), abs(lo - hit[1]), abs(hi - hit[2]))
            worst = max(worst, diff)
            out[f"{t}|{col}|{g}"] = {"source": hit[4], "max_abs_diff": diff, "rows": int(len(Tn["sa"])),
                                     "rows_published": hit[3], "rows_equal": hit[3] == len(Tn["sa"])}
    rows_ok = all(x["rows_equal"] for x in out.values() if isinstance(x, dict))
    return {"cells": out, "max_abs_diff": worst, "rows_equal": rows_ok, "pass": bool(worst <= 1e-9 and rows_ok)}


def k7(info: dict) -> dict:
    """Riproduzione: embedding dei 5 soggetti di controllo per pool contro lo store ufficiale, per etichetta."""
    import fact_paired as fp
    fp.VIEWS["faceverse_neutral"] = ("fvn", "mesh_pair_nocrop")
    fp.FORM_DIR["fvn"] = os.path.relpath(NEUTRAL_EMB, fp.EVAL)
    dom = {"facescape": "devfs", "hifi3d": "hifi", "faceverse": "fvn"}
    out, worst = {}, 0.0
    for p in POOLS:
        chk = pool_info(p)["check"]
        for arm in diag.ARMS:
            off = arm_store(arm, dom[p])
            if off is None:                                     # C3M e123 neutro: lo store di d3/fvn (pipeline ufficiale)
                off = sorted((D3 / "fvn" / "embed").glob(f"data_*/scale_v3{arm_tag(arm)}*/zs_zeroshot/embeddings.npz"))[0]
            names = [f"{s}_GTready_{lab}.npz" for s in chk for lab in diag.LABELS]
            _, Zm, cm = store_by_names(D3 / "pools" / p / "emb" / arm / "embeddings.npz", names)
            _, Zo, co = store_by_names(off, names)
            if cm.resolve() != co.resolve():
                raise SystemExit(f"K7 {p} {arm}: checkpoint diversi")
            lab = np.asarray([diag.split_name(n)[1] for n in names])
            per = {lb: float(np.abs(Zm[lab == lb] - Zo[lab == lb]).max()) for lb in diag.LABELS}
            worst = max(worst, max(per.values()))
            out[f"{p}|{arm}"] = {"max_abs": per, "scale": float(np.abs(Zo).max())}
    return {"cells": out, "max_abs": worst, "pass": bool(worst <= 1e-2)}


def verdict(dd: dict, method: str) -> dict:
    """R_LOGO (e secondarie): per braccio n_SR >= 3 bersagli con delta SR >= +0.05 e IC > 0, FR >= -0.03 su tutti."""
    per = {}
    for arm in diag.RULE_ARMS:
        cells = {t: (dd[(t, arm, method, "sr")], dd[(t, arm, method, "fr")]) for t in PRIMARY}
        n_sr = sum(s["delta"] >= SR_MIN and s["ci_low"] > 0 for s, _ in cells.values())
        fr_ok = all(f["delta"] >= FR_MIN for _, f in cells.values())
        v = ("PASSA" if fr_ok else "SR SI, FR NO") if n_sr >= N_SR else "NO"
        per[arm] = {"n_sr": int(n_sr), "fr_ok": bool(fr_ok), "verdict": v,
                    "targets": {t: {"sr": s, "fr": f} for t, (s, f) in cells.items()}}
    vs = {per[a]["verdict"] for a in diag.RULE_ARMS}
    return {"verdict": vs.pop() if len(vs) == 1 else "NON RISOLTO", "arms": per}


def curve_verdict(recs: list) -> dict:
    """La media di delta SR sui sottoinsiemi non decresce da k = 1 a 2, 4, 7 su almeno 3 bersagli su 4, entrambi i semi."""
    out = {}
    for meth in ("i", "ii"):
        per = {}
        for arm in diag.RULE_ARMS:
            ok = 0
            means = {}
            for t in TARGETS:
                m = [float(np.mean([r[f"delta_{meth}"] for r in recs if r["arm"] == arm and r["target"] == t
                                    and r["k"] == k])) for k in CURVE_K + (7,)]
                means[t] = m
                ok += all(b >= a for a, b in zip(m, m[1:]))
            per[arm] = {"targets_non_decreasing": ok, "means_k1_2_4_7": means}
        grows = [per[a]["targets_non_decreasing"] >= 3 for a in diag.RULE_ARMS]
        out[meth] = {"grows": "si" if all(grows) else ("no" if not any(grows) else "non risolto"), "arms": per}
    return out


# ------------------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=diag.N_BOOT)
    a = ap.parse_args()
    t0 = time.time()
    D3.mkdir(parents=True, exist_ok=True)
    cal = diag.calibration()
    info = {"ckpt": {}, "dpu": {}}
    for arm in diag.ARMS:                                    # checkpoint e unita' dagli embedding della calibrazione
        _, _, ck = store_by_names(diag.CALIB_EMB / arm / "embeddings.npz", diag.set_names("bfm"))
        info["ckpt"][arm], info["dpu"][arm] = ck, diag.dp_per_unit(ck)
        build_sources(arm, info)
    for p in POOLS:                                          # K6: sorgenti dei pool disgiunte dai soggetti di controllo
        pi = pool_info(p)                                    # (e da quelli di test: sotto, insieme per insieme)
        if set(pi["source"]) & set(pi["check"]):
            raise SystemExit(f"{p}: sorgenti fra i soggetti valutati")
    print(f"[d3] sorgenti {info['source_sizes']} ({time.time() - t0:.0f}s)", flush=True)
    k2 = {}
    for arm in diag.ARMS:                                    # K2: c dei bracci dagli held-out con la regola di fact_calib
        c = H.calib_c(lambda b, i, j: np.linalg.norm(b.X[i] - b.X[j], axis=1), [_B[(arm, s)] for s in ("bfm", "ict", "gnm")])
        k2[arm] = {"c": c, "c_median": cal[arm], "abs_diff": abs(c - cal[arm])}
    with mp.get_context("fork").Pool(len(FP_VIEWS)) as pool:
        tests = dict(zip(FP_VIEWS, pool.map(view_data, FP_VIEWS)))
    for name in ("flame2023_s1", "famos"):
        tests[name] = set_data(name, info)
    for t, Tn in tests.items():
        for arm, ck in Tn["ckpt"].items():
            if ck.resolve() != info["ckpt"][arm].resolve():
                raise SystemExit(f"{t} {arm}: checkpoint {ck}, atteso {info['ckpt'][arm]}")
        if t in POOLS or t == "faceverse_neutral":           # K6: soggetti di test mai fra le sorgenti dei pool
            pi = pool_info(TESTS[t])
            if set(pi["source"]) & set(Tn["subjects"]):
                raise SystemExit(f"{t}: soggetti di test fra le sorgenti del pool")
        print(f"[d3] test {t}: {len(Tn['sa'])} righe su {Tn['n_rows_all']}, {len(Tn['subjects'])} soggetti, seme "
              f"{Tn['seed']}, bracci {sorted(Tn['Z'])}, mancano {Tn['missing']} ({time.time() - t0:.0f}s)", flush=True)
    kk7 = k7(info)
    print(f"[d3] K7 riproduzione: max |dz| {kk7['max_abs']:.1e}, passa {kk7['pass']}", flush=True)
    if not kk7["pass"]:
        diag.atomic_json(D3 / "controls.json", {"K7": kk7, "stop": "K7 non passa: ci si ferma (emendamento 1, sez. 7)"})
        raise SystemExit("K7 non passa")
    # ------------------------------------------------------------------ CV della testa (i)
    sets = settings()
    tasks = [(arm, key, srcs, k, q, f) for arm in diag.ARMS for key, srcs in sets.items()
             for k in range(len(H.CONFIGS)) for q in range(H.N_REPEAT) for f in range(H.N_FOLD)]
    tasks.sort(key=lambda t: -H.CONFIGS[t[3]][0])
    with mp.get_context("fork").Pool(a.workers) as pool:
        cv = list(pool.imap_unordered(_cv, tasks, chunksize=1))
    t_cv = time.time() - t0
    S, recs_cv, chosen, k5 = {}, [], {}, []
    for r in cv:
        arm, key, k, q, f = r["task"]
        S.setdefault((arm, key, k), []).append(r)
        if "k5" in r:
            k5.append(r["k5"])
    for arm in diag.ARMS:
        for key in sets:
            means = {}
            for k, cfg in enumerate(H.CONFIGS):
                rs = S[(arm, key, k)]
                sc = np.asarray([x["score"] for x in rs])
                means[cfg] = float(sc.mean())
                recs_cv.append({"arm": arm, "setting": key, "r": cfg[0], "lambda": cfg[1], "cv_score": means[cfg],
                                "cv_sd": float(sc.std(ddof=1)), "n_fits": len(rs),
                                "nit_median": float(np.median([x["nit"] for x in rs])),
                                "success_frac": float(np.mean([x["success"] for x in rs])),
                                "seconds_median": float(np.median([x["seconds"] for x in rs]))})
            chosen[(arm, key)] = H.choose(means)
            print(f"[d3] {arm} {key}: scelta r={chosen[(arm, key)][0]} lambda={chosen[(arm, key)][1]:g}", flush=True)
    # ------------------------------------------------------------------ teste finali e di fold
    jobs = [(arm, key, sets[key], *chosen[(arm, key)], -1, -1) for arm in diag.ARMS for key in sets]
    jobs += [(arm, key, sets[key], *chosen[(arm, key)], q, f) for arm in diag.ARMS for key in sets
             for q in range(H.N_REPEAT) for f in range(H.N_FOLD)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        fits = pool.map(_refit, jobs, chunksize=1)
    heads, fheads = {}, {}
    for (arm, key, srcs, r, lam, q, f), head, c in fits:
        if q < 0:
            heads[(arm, "i", key)] = (head, c)
        else:
            fheads.setdefault((arm, key), {})[f"@{q}{f}"] = (head, c)
    for arm in diag.ARMS:
        for key, srcs in sets.items():
            heads[(arm, "ii", key)] = within_head(arm, srcs)
        for p, h in coral_heads(arm).items():
            heads[(arm, "coral", p)] = h
    print(f"[d3] teste pronte ({time.time() - t0:.0f}s, CV {t_cv:.0f}s)", flush=True)
    # ------------------------------------------------------------------ curva sul numero di sorgenti
    _TS.update(tests)
    _TS["_dpu"] = info["dpu"]
    ctasks = []
    for arm in diag.ARMS:
        for t, test in zip(TARGETS, PRIMARY):
            avail = sets[f"{t}|all"]
            for k in CURVE_K:
                for sub in itertools.combinations(avail, k):
                    ctasks.append((arm, t, test, sub, *chosen[(arm, f"{t}|all")]))
    with mp.get_context("fork").Pool(a.workers) as pool:
        recs_curve = pool.map(_curve, ctasks, chunksize=1)
    # ------------------------------------------------------------------ colonne e repliche
    for t, Tn in tests.items():
        for rsn, rs in rowsets(Tn).items():
            _J[(t, rsn)] = {"counts": Tn["counts"][: a.n_boot + 1], "sa": rs[0], "sb": rs[1], "gt": rs[2],
                            "cols": columns(Tn, rs, heads, cal, info)}
    keys = [(key, b) for key in _J for b in range(a.n_boot + 1)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        reps = pool.map(_rep, keys, chunksize=8)
    V = {key: {} for key in _J}
    for (key, b), r in zip(keys, reps):
        for m, x in r.items():
            V[key].setdefault(m, np.empty(a.n_boot + 1))[b] = x
    from scipy.stats import spearmanr                          # teste di fold: stima puntuale sulle righe incrociate
    fold_rho = {}
    for t, Tn in tests.items():
        rs = rowsets(Tn)["cross"]
        for m, x in columns(Tn, rs, heads, cal, info, fold=fheads).items():
            for g in gt_of(m):
                fold_rho[(t, m, g)] = float(spearmanr(x, rs[2][g]).correlation)
    for arm in diag.ARMS:                                            # k = 7: la testa di R_LOGO (stima puntuale)
        for t, test in zip(TARGETS, PRIMARY):
            Vt = V[(test, "cross")]
            recs_curve.append({"arm": arm, "target": t, "test": test, "k": 7, "sources": "+".join(sets[f"{t}|all"]),
                               "delta_i": float(Vt[(f"{arm}|i|shape", "sr")][0] - Vt[(f"{arm}|dP|shape", "sr")][0]),
                               "delta_ii": float(Vt[(f"{arm}|ii|shape", "sr")][0] - Vt[(f"{arm}|dP|shape", "sr")][0]),
                               "rho_dP": float(Vt[(f"{arm}|dP|shape", "sr")][0])})
    # ------------------------------------------------------------------ uscite
    recs_rho, recs_d, dd = [], [], {}
    for (t, rsn), Vt in V.items():
        for (m, g), x in Vt.items():
            p, lo, hi, _ = diag.ci(x)
            arm, _, rest = m.partition("|")
            recs_rho.append({"test": t, "rows": rsn, "column": m, "arm": "B" if arm.startswith("B-") else arm,
                             "method": rest.split("|")[0] if not arm.startswith("B-") else arm, "gt": g, "rho": p,
                             "ci_low": lo, "ci_high": hi, "n_rows": int(len(_J[(t, rsn)]["sa"])),
                             "seed": tests[t]["seed"]})

    def add(kind, t, rsn, arm, meth, g, what, x, ref, extra=None):
        p, lo, hi, ple = diag.ci(x - ref)
        rec = {"kind": kind, "test": t, "rows": rsn, "arm": arm, "method": meth, "gt": g, "what": what, "delta": p,
               "ci_low": lo, "ci_high": hi, "p_le0": ple, "rho_method": float(x[0]), "rho_ref": float(ref[0]),
               **(extra or {})}
        recs_d.append(rec)
        return rec

    for (t, rsn), Vt in V.items():
        for arm in tests[t]["Z"]:
            for meth in METHODS[1:]:
                if (f"{arm}|{meth}|shape", "sr") not in Vt:
                    continue
                for g, col, ref, tag in (("sr", f"{arm}|{meth}|shape", f"{arm}|dP|shape", "sr"),
                                         ("fr", f"{arm}|{meth}|form", f"{arm}|dP|form", "fr"),
                                         ("fr", f"{arm}|{meth}|shape", f"{arm}|dP|shape", "fr_nosize")):
                    extra = {}
                    if rsn == "cross" and meth in ("i", "i_nf"):
                        var = "all" if meth == "i" else "nofamos"
                        fd = [fold_rho[(t, f"{arm}|{meth}{tg}|{col.rsplit('|', 1)[1]}", g)] - Vt[(ref, g)][0]
                              for tg in fheads.get((arm, f"{TESTS[t]}|{var}"), {})]
                        if fd:
                            extra = {"fold_min": float(np.min(fd)), "fold_median": float(np.median(fd)),
                                     "fold_max": float(np.max(fd))}
                    rec = add("ref", t, rsn, arm, meth, tag, f"{col} - {ref} ({g})", Vt[(col, g)], Vt[(ref, g)], extra)
                    if rsn == "cross":
                        dd[(t, arm, meth, tag)] = rec
                        for m in B_MODELS:
                            if g == "sr" and tag == "sr" and (f"B-{m}|vb_sr", "sr") in Vt:
                                add("B", t, rsn, arm, meth, "sr", f"{col} - B-{m}|vb_sr", Vt[(col, g)],
                                    Vt[(f"B-{m}|vb_sr", "sr")])
                            if tag == "fr" and (f"B-{m}|vb_fr", "fr") in Vt:
                                add("B", t, rsn, arm, meth, "fr", f"{col} - B-{m}|vb_fr", Vt[(col, g)],
                                    Vt[(f"B-{m}|vb_fr", "fr")])
            for base, nf in (("i", "i_nf"), ("ii", "ii_nf")):          # contributo del reale
                if (f"{arm}|{nf}|shape", "sr") in Vt:
                    for g, dist, tag in (("sr", "shape", "sr"), ("fr", "form", "fr")):
                        add("famos", t, rsn, arm, base, tag, f"{arm}|{base}|{dist} - {arm}|{nf}|{dist} ({g})",
                            Vt[(f"{arm}|{base}|{dist}", g)], Vt[(f"{arm}|{nf}|{dist}", g)])
    for arm in diag.ARMS:                                            # media sui 4 bersagli primari, replica per replica
        if not all(arm in tests[t]["Z"] for t in PRIMARY):
            continue
        for meth in ("i", "i_nf", "ii", "ii_nf"):
            for g, dist, tag in (("sr", "shape", "sr"), ("fr", "form", "fr")):
                x = np.mean([V[(t, "cross")][(f"{arm}|{meth}|{dist}", g)] for t in PRIMARY], axis=0)
                ref = np.mean([V[(t, "cross")][(f"{arm}|dP|{dist}", g)] for t in PRIMARY], axis=0)
                dd[("mean", arm, meth, tag)] = add("mean", "media dei 4 bersagli", "cross", arm, meth, tag,
                                                   f"media: {meth} - dP ({g})", x, ref)
    for name, recs in (("d3_cv.csv", recs_cv), ("d3_spearman.csv", recs_rho), ("d3_delta.csv", recs_d),
                       ("d3_curve.csv", recs_curve)):
        with open(diag.EV / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in recs for k in r)))
            w.writeheader()
            w.writerows(recs)
    recs_h = []
    for arm in diag.ARMS:
        for key in sets:
            head, c = heads[(arm, "i", key)]
            cvs = {(x["r"], x["lambda"]): x["cv_score"] for x in recs_cv if x["arm"] == arm and x["setting"] == key}
            recs_h.append({"arm": arm, "setting": key, "sources": "+".join(sets[key]), "method": "i", "r": head.r,
                           "lambda": head.lam, "cv_score": cvs[(head.r, head.lam)], "cv_score_dP": cvs[(0, 0.0)],
                           "edge": bool(head.r == H.R_GRID[-1] or (head.r and head.lam in (H.LAMBDAS[0], H.LAMBDAS[-1]))),
                           "alpha_over_alpha0": head.alpha / head.alpha0, "c": c, "nit": head.nit,
                           "success": head.success, "W_fro_over_alpha": float(np.linalg.norm(head.W) / head.alpha)
                           if head.r else 0.0})
            recs_h.append({"arm": arm, "setting": key, "sources": "+".join(sets[key]), "method": "ii",
                           "c": heads[(arm, "ii", key)][1]})
        for p in POOLS:
            recs_h.append({"arm": arm, "setting": p, "sources": "bfm+ict+gnm", "method": "coral",
                           "c": heads[(arm, "coral", p)][1]})
    with open(diag.EV / "d3_heads.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in recs_h for k in r)))
        w.writeheader()
        w.writerows(recs_h)
    readings = {"R_LOGO": verdict(dd, "i"), "secondary": {"ii": verdict(dd, "ii"), "i_nofamos": verdict(dd, "i_nf")},
                "mean_delta": {f"{arm}|{meth}|{tag}": dd[("mean", arm, meth, tag)] for (k0, arm, meth, tag) in dd
                               if k0 == "mean"},
                "curve": curve_verdict(recs_curve), "rule_arms": list(diag.RULE_ARMS),
                "thresholds": {"sr": SR_MIN, "fr": FR_MIN, "n_targets": N_SR}}
    controls = {"K1": k1(V, tests), "K2": {"arms": k2, "pass": bool(max(x["abs_diff"] for x in k2.values()) <= 1e-9)},
                "K3": json.loads((D3 / "famos" / "gen.json").read_text())["K3"], "K7": kk7,
                "K4": {"ckpt": {k: str(v) for k, v in info["ckpt"].items()}, "dpu": info["dpu"],
                       "source_sizes": info["source_sizes"], "stores": {t: Tn["stores"] for t, Tn in tests.items()},
                       "missing": {t: Tn["missing"] for t, Tn in tests.items()}},
                "K5_r0_max_abs_diff": float(max(k5)),
                "K6": "bersaglio fuori dalle sorgenti della sua testa (settings), sorgenti dei pool disgiunte dai soggetti "
                      "valutati e di test, soggetti di fit e di validazione disgiunti (d3_head.split) verificati nel codice",
                "settings": {k: list(v) for k, v in sets.items()},
                "tests": {t: {"rows": int(len(Tn["sa"])), "rows_all": Tn["n_rows_all"],
                              "subjects": int(len(Tn["subjects"])), "seed": Tn["seed"],
                              "rowsets": {rsn: int(len(_J[(t, rsn)]["sa"])) for rsn in rowsets(Tn)}}
                          for t, Tn in tests.items()},
                "n_boot": a.n_boot, "seconds_cv": t_cv, "seconds": time.time() - t0, "workers": a.workers}
    diag.atomic_json(D3 / "readings.json", readings)
    diag.atomic_json(D3 / "controls.json", controls)
    diag.atomic_savez(D3 / "reps.npz", **{f"{t}|{rsn}|{m}|{g}": x for (t, rsn), Vt in V.items() for (m, g), x in Vt.items()})
    diag.atomic_savez(D3 / "heads.npz", **{f"{arm}|{key}|{k}": x for (arm, meth, key), (h, c) in heads.items()
                                           if meth == "i" for k, x in (("alpha", h.alpha), ("W", h.W), ("c", c))})
    print(f"[d3] R_LOGO {readings['R_LOGO']['verdict']}; (ii) {readings['secondary']['ii']['verdict']}; senza FaMoS "
          f"{readings['secondary']['i_nofamos']['verdict']}; K1 {controls['K1']['pass']} "
          f"({controls['K1']['max_abs_diff']:.1e}), K2 {controls['K2']['pass']}, K7 {kk7['pass']}, K5 "
          f"{controls['K5_r0_max_abs_diff']:.1e}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
