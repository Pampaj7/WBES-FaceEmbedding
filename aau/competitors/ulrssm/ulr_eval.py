#!/usr/bin/env python3
"""Distanze di ULRSSM sulle viste di valutazione: per coppia (variante a) e per mesh via template (variante b).

    aau/run.sh aau/competitors/ulrssm/ulr_eval.py smoke --ckpt <pth> --cache /tmp/<job>/ulr          (GPU, controlli)
    aau/run.sh aau/competitors/ulrssm/ulr_eval.py run --view hifi3d --ckpt <pth> --tag faces_xyz \
        --cache /tmp/<job>/ulr --workers 32 [--shard 0/1] [--spec <json>]                             (GPU)
    v3_work/unified_gt/run.sh aau/competitors/ulrssm/ulr_eval.py merge --view hifi3d --tag faces_xyz  (CPU)
    aau/run.sh aau/competitors/ulrssm/ulr_eval.py select                                              (CPU)
    (ulr.sbatch, passi smoke, eval, merge, select)

``run``: le mesh della vista (``ulr_common.view``: 100 soggetti x 6 topologie col crop, FaMoS le 15 scansioni, o una
``--spec``) preparate come in training (``ulr_common.prep``: facce orientate, area 1, operatori nella cache del job),
feature di DiffusionNet del checkpoint, poi
  (a) per coppia, sulle coppie ``ulr_common.pairs`` della parte ``--shard k/n``: ``ulr_common.pair_rms`` (mappa nei
      due versi, rigida pesata senza scala, RMS pesato per area in mm; SR con ogni mesh x CS_ref / CS_i);
  (b) per mesh (solo la parte 0): mappa template -> mesh (``p2p`` con y = template: un vertice della mesh per ogni
      vertice del template) dal template fisso ``ulr_common.template_gnm`` (media di GNM, generatore di training), i
      vertici della mesh in mm portati sui 1478 punti della regione unificata con la mappa baricentrica del template.
Uscita ``<tag>/parts/<vista>_<k>of<n>.npz`` (fuori da git): distanze delle coppie della parte, configurazioni trasferite,
tempi per mesh e per coppia.

``merge``: unisce le parti; variante (b) con le funzioni della GT di valutazione importate e NON modificate
(``train_fr_sr.factor``, ``v3_work/canonical_gt/train_fr_sr.py``, che usa ``cgt.Canon``): a_i = ``rigid_robust``
(IRLS Tukey verso mu, una rigida per mesh), z_i = (a_i - c_i) / S_i (taglia tolta: centroide e centroid size pesati
della mesh); FR = RMS pesato per area fra le a (mm), SR = lo stesso fra le z (d_P, adimensionale), con la formula di
``train_fr_sr.save_eval`` (``cgt._euclid``). Il dominio passato a ``factor`` e' ``famos``, per cui ``Canon.to_F`` e'
l'identita': i punti trasferiti sono gia' in mm nel frame canonico. Controllo (``gt_check``, in ogni ``<vista>.json``):
la stessa funzione sui punti nativi delle 15 persone FaMoS TEST ridà ``famos_test_{fr,sr}.npz``. Uscite in ``aau/runs/evidence/competitors_new/ulrssm/<tag>/``: ``<vista>.npz`` (``subjects``,
``topologies``, ``paths``, matrici simmetriche ``D_pair_fr``, ``D_pair_sr``, ``D_tpl_fr``, ``D_tpl_sr``, NaN dove
non calcolate, configurazioni ``U_tpl``), ``pairs_<vista>.csv`` (righe di ``pairs``, fuori da git) e
``<vista>.json`` (tempi, controlli, mesh fallite).

``smoke``: (1) la mappa a blocchi di ``ulr_common.p2p`` contro il percorso di ULRSSM (``FMNetModel.validate_single``:
``compute_permutation_matrix`` + ``fmap2pointmap`` con ``torch.cdist``) su coppie piccole di HIFI3D: quota di vertici
con la stessa immagine; (2) tempi di preparazione, feature, mappa per una coppia di ogni vista (anche up60k).

``select`` (regola fissata PRIMA di vedere le viste di valutazione): fra ``faces_xyz`` e ``faces_wks`` il checkpoint
finale con l'errore di validazione medio (``train.json``, identita' di validazione dei generatori) minore e' il
principale; l'altro resta come variante dichiarata. Uscita ``selection.json``.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import ulr_common as U  # noqa: E402

import blmm  # noqa: E402

_P: dict = {}


# ------------------------------------------------------------------------------------------- preparazione

def _prep(k: int):
    """Mesh k della vista: (k, dict con verts, faces, V mm, aree mm^2, CS, ...) o (k, errore)."""
    import torch
    torch.set_num_threads(1)                         # worker del Pool: niente thread annidati
    v = _P["view"]
    s, t, path = v["items"][k]
    try:
        V, F = blmm.load_raw(path)
        V, F = U.referenced(V, F)
        Vmm = U.to_mm(v, V, path)
        r = U.prep(Vmm, F, _P["cache"])
        cs, src = U.size_of(v, path, Vmm, F)
        return k, {"verts": r["verts"], "faces": r["faces"], "Vmm": Vmm.astype(np.float32),
                   "w": blmm.vertex_areas(Vmm, F).astype(np.float32), "cs": cs, "cs_src": src, "flipped": r["flipped"],
                   "prep_s": r["seconds"], "n": int(len(V))}
    except Exception as exc:  # noqa: BLE001  (la mesh resta NaN in tutte le sue coppie, contata)
        return k, f"{type(exc).__name__}: {exc}"


def prepare(v: dict, cache: str, workers: int) -> tuple[dict, dict]:
    _P.update(view=v, cache=cache)
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_prep, range(len(v["items"])), chunksize=1)
    ok = {k: r for k, r in res if isinstance(r, dict)}
    failed = {k: r for k, r in res if not isinstance(r, dict)}
    return ok, failed


def on_gpu(net, it: dict, cache: str, cs_ref: float, device) -> dict:
    """Feature (fp16: la vista FaceVerse ha 18.7 M vertici), autovettori, massa, mm, aree, scala SR sulla GPU."""
    import torch
    get = U.ulrssm()["get_operators"]
    v, f = torch.from_numpy(it["verts"]), torch.from_numpy(it["faces"]).long()
    _, mass, _, _, evecs, _, _ = get(v, f, k=U.K_FMAP, cache_dir=cache)
    t0 = time.perf_counter()
    fe = U.features(net, it["verts"], it["faces"], device)
    torch.cuda.synchronize()
    return {"f": fe.half(), "e": evecs[:, :U.K_FMAP].to(device), "m": mass.to(device),
            "V": torch.from_numpy(it["Vmm"]).to(device), "w": torch.from_numpy(it["w"]).to(device),
            "s": cs_ref / it["cs"], "feat_s": time.perf_counter() - t0}


def shard_of(txt: str) -> tuple[int, int]:
    k, n = (int(x) for x in txt.split("/"))
    if not 0 <= k < n:
        raise SystemExit(f"--shard {txt}")
    return k, n


def part_path(tag: str, vname: str, k: int, n: int) -> Path:
    return U.OUT / tag / "parts" / f"{vname}_{k}of{n}.npz"


# ------------------------------------------------------------------------------------------------- run

def run(a) -> None:
    import torch
    v = U.view(a.view, a.spec)
    k_sh, n_sh = shard_of(a.shard)
    out = part_path(a.tag, v["name"], k_sh, n_sh)
    if out.exists():
        print(f"[ulr-eval] {out}: gia' presente, salto", flush=True)
        return
    os.makedirs(a.cache, exist_ok=True)
    t0 = time.time()
    ok, failed = prepare(v, a.cache, a.workers)
    t_prep = time.time() - t0
    print(f"[ulr-eval] {v['name']}: {len(ok)} mesh preparate in {t_prep:.0f}s, fallite {len(failed)} "
          f"{list(failed.values())[:3]}; facce invertite {sum(r['flipped'] for r in ok.values())}", flush=True)
    device = torch.device("cuda")
    net, input_type = U.load_net(Path(a.ckpt), device, a.cache)
    G = {k: on_gpu(net, r, a.cache, v["cs_ref"], device) for k, r in ok.items()}
    t_feat = time.time() - t0 - t_prep
    n = len(v["items"])
    res = {"subjects": np.asarray([s for s, _, _ in v["items"]]), "topologies": np.asarray([t for _, t, _ in v["items"]]),
           "paths": np.asarray([blmm.rel(p) if Path(p).is_absolute() else str(p) for _, _, p in v["items"]]),
           "n_vertices": np.asarray([ok[k]["n"] if k in ok else -1 for k in range(n)]),
           "cs": np.asarray([ok[k]["cs"] if k in ok else np.nan for k in range(n)]),
           "flipped": np.asarray([bool(ok[k]["flipped"]) if k in ok else False for k in range(n)]),
           "prep_s": np.asarray([ok[k]["prep_s"] if k in ok else np.nan for k in range(n)]),
           "feat_s": np.asarray([G[k]["feat_s"] if k in G else np.nan for k in range(n)]),
           "failed": np.asarray([f"{k}|{e}" for k, e in failed.items()], dtype="U300")}
    if k_sh == 0:                                          # (b): template -> mesh, una volta per vista
        T = U.template_gnm()
        rT = U.prep(T["V"], T["F"], a.cache)
        GT = on_gpu(net, {**rT, "Vmm": T["V"].astype(np.float32), "w": np.ones(len(T["V"]), np.float32), "cs": 1.0},
                    a.cache, 1.0, device)
        Ut = np.full((n, T["n_unified"], 3), np.nan)
        tpl_s = np.full(n, np.nan)
        for k, g in G.items():
            t1 = time.perf_counter()
            idx = U.p2p(g["f"], GT["f"], g["e"], GT["e"], GT["m"])       # un vertice della mesh per vertice del template
            Y = g["V"][idx].double().cpu().numpy()
            torch.cuda.synchronize()
            tpl_s[k] = time.perf_counter() - t1
            Ut[k] = np.einsum("nkd,nk->nd", Y[T["pos"]], T["bary"])
        res.update(U_tpl=Ut, tpl_s=tpl_s, template_n=len(T["V"]), template_flipped=bool(rT["flipped"]),
                   template_check=T["check_unified"])
        print(f"[ulr-eval] {v['name']}: template GNM ({len(T['V'])} vertici) mappato su {len(G)} mesh, "
              f"{np.nanmedian(tpl_s) * 1e3:.0f} ms/mesh", flush=True)
    I, J = U.pairs(v["items"])
    sel = np.arange(len(I))[k_sh::n_sh]
    I, J = I[sel], J[sel]
    D = np.full((len(I), 4), np.nan)
    pair_s = np.full(len(I), np.nan)
    t2 = time.time()
    for q, (i, j) in enumerate(zip(I, J)):
        if i not in G or j not in G:
            continue
        t1 = time.perf_counter()
        D[q] = U.pair_rms(G[i], G[j])
        pair_s[q] = time.perf_counter() - t1
        if q % 5000 == 0:
            el = time.time() - t2
            print(f"[ulr-eval] {v['name']} parte {k_sh}/{n_sh}: coppia {q}/{len(I)}, {el / 60:.1f} min, "
                  f"stima totale {el / (q + 1) * len(I) / 3600:.2f} h", flush=True)
    res.update(pair_i=I, pair_j=J, pair_d=D, pair_cols=np.asarray(["fr", "sr", "fr_b_to_a", "fr_a_to_b"]),
               pair_s=pair_s, shard=np.asarray([k_sh, n_sh]), input_type=input_type, ckpt=str(a.ckpt),
               t_prep_s=t_prep, t_feat_s=t_feat, t_pairs_s=time.time() - t2, gpu=torch.cuda.get_device_name(0),
               cs_ref=v["cs_ref"], job=os.environ.get("SLURM_JOB_ID", "none"))
    blmm.atomic_savez(out, **res)
    print(f"[ulr-eval] {v['name']} parte {k_sh}/{n_sh}: {len(I)} coppie in {(time.time() - t2) / 60:.1f} min "
          f"({np.nanmedian(pair_s) * 1e3:.0f} ms/coppia mediana), NaN {int(np.isnan(D[:, 0]).sum())} -> {out}", flush=True)


# ----------------------------------------------------------------------------------------------- merge

def tpl_distances(Ut: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """(FR mm, SR d_P, diagnostica) della variante (b) dalle configurazioni (n, 1478, 3) in mm, frame canonico:
    ``train_fr_sr.factor`` e la formula di ``save_eval``, importate e non modificate."""
    for _p in (U.REPO / "v3_work" / "canonical_gt", U.REPO / "v3_work" / "unified_gt"):
        sys.path.insert(0, str(_p))
    import cgt
    import train_fr_sr as tfs
    cn = cgt.Canon()
    if Ut.shape[1] != len(cn.mu):
        raise SystemExit(f"punti trasferiti {Ut.shape[1]} contro i {len(cn.mu)} della regione unificata")
    ok = np.isfinite(Ut).all(axis=(1, 2))
    FR, SR = np.full((len(Ut), len(Ut)), np.nan), np.full((len(Ut), len(Ut)), np.nan)
    f = tfs.factor(cn, "famos", np.asarray(Ut[ok], dtype=np.float64))     # to_F("famos") = identita' (gia' mm)
    FR[np.ix_(ok, ok)] = cgt._euclid(f["A"].astype(np.float64), f["A"].astype(np.float64), cn.sp.A, True)
    SR[np.ix_(ok, ok)] = cgt._euclid(f["Z"].astype(np.float64), f["Z"].astype(np.float64), cn.sp.A, True)
    a = f["A"].astype(np.float64).reshape(int(ok.sum()), -1, 3) / np.sqrt(cn.sp.w)[None, :, None]
    diag = {"rigid_robust_converged": int(f["conv"].sum()), "n": int(ok.sum()), "rigid_robust_iter_max": int(f["it"].max()),
            "centroid_size_mm_median": float(np.median(f["S"])),
            "rms_to_mu_mm_median": float(np.median(np.sqrt(np.einsum("v,nv->n", cn.W, ((a - cn.mu) ** 2).sum(-1)))))}
    return FR, SR, diag


def gt_check() -> dict:
    """``tpl_distances`` sui punti nativi delle persone FaMoS TEST (``cgt.native_points``, ordine di
    ``train_fr_sr.stage_eval``) contro ``famos_test_{fr,sr}.npz`` x l'unita' del json: scarto massimo."""
    sys.path.insert(0, str(U.REPO / "v3_work" / "canonical_gt"))
    import cgt
    import famos_common as fc
    cn = cgt.Canon()
    nat = cgt.native_points("famos", cn)
    te = np.flatnonzero(nat["split"] == "test")
    te = te[np.argsort([fc.view_id(nat["ids"][i]) for i in te])]
    ids = [fc.view_id(nat["ids"][i]) for i in te]
    FR, SR, _ = tpl_distances(nat["P"][te])
    out = {}
    for g, M in (("fr", FR), ("sr", SR)):
        p = cgt.OUT_DIR / "eval" / f"famos_test_{g}.npz"
        unit = json.loads(p.with_suffix(".json").read_text())["mm_per_unit" if g == "fr" else "dP_per_unit"]
        with np.load(p) as z:
            pos = {str(s): k for k, s in enumerate(z["names"])}
            ref = np.asarray(z["D_orig"], np.float64)[np.ix_([pos[s] for s in ids], [pos[s] for s in ids])] * unit
        out[f"{g}_max_abs"] = float(np.abs(M - ref).max())
        out[f"{g}_max_rel"] = float(np.abs(M - ref).max() / np.median(ref[np.triu_indices(len(ids), 1)]))
    return out


def merge(a) -> None:
    v = U.view(a.view, a.spec)
    root = U.OUT / a.tag
    parts = sorted((root / "parts").glob(f"{v['name']}_*of*.npz"))
    if not parts:
        raise SystemExit(f"{v['name']}: nessuna parte in {root / 'parts'}")
    ns = {int(p.stem.rsplit("of", 1)[1]) for p in parts}
    if len(ns) != 1 or len(parts) != next(iter(ns)):
        raise SystemExit(f"{v['name']}: parti incomplete o miste {[p.name for p in parts]}")
    n = len(v["items"])
    D = {f"D_pair_{m}": np.full((n, n), np.nan) for m in U.MODES}
    pair_s, info, Ut = [], {"parts": []}, None
    for p in parts:
        with np.load(p) as z:
            if [str(x) for x in z["subjects"]] != [s for s, _, _ in v["items"]]:
                raise SystemExit(f"{p}: mesh diverse dalla vista")
            I, J, Dp = z["pair_i"], z["pair_j"], z["pair_d"]
            for c, m in enumerate(U.MODES):
                D[f"D_pair_{m}"][I, J] = D[f"D_pair_{m}"][J, I] = Dp[:, c]
            pair_s.append(z["pair_s"])
            if "U_tpl" in z.files:
                Ut, tpl_s = np.asarray(z["U_tpl"]), np.asarray(z["tpl_s"])
                base = {k: np.asarray(z[k]) for k in ("subjects", "topologies", "paths", "n_vertices", "cs", "flipped",
                                                      "prep_s", "feat_s", "failed")}
                info.update(input_type=str(z["input_type"]), ckpt=str(z["ckpt"]), gpu=str(z["gpu"]),
                            template_n=int(z["template_n"]), template_flipped=bool(z["template_flipped"]),
                            cs_ref=float(z["cs_ref"]))
            info["parts"].append({"part": p.name, "job": str(z["job"]), "n_pairs": int(len(I)),
                                  "t_prep_s": float(z["t_prep_s"]), "t_feat_s": float(z["t_feat_s"]),
                                  "t_pairs_s": float(z["t_pairs_s"]), "nan_pairs": int(np.isnan(Dp[:, 0]).sum())})
    if Ut is None:
        raise SystemExit(f"{v['name']}: manca la parte 0 (template)")
    FR, SR, diag = tpl_distances(Ut)
    D.update(D_tpl_fr=FR, D_tpl_sr=SR)
    for M in D.values():
        np.fill_diagonal(M, 0.0)
    pair_s = np.concatenate(pair_s)
    I, J = U.pairs(v["items"])
    info.update(view=v["name"], n_meshes=n, n_pairs=int(len(I)), template=diag, gt_check=gt_check(),
                nan={k: int(np.isnan(M[I, J]).sum()) for k, M in D.items()}, failed=[str(x) for x in base["failed"]],
                mesh_flipped=int(base["flipped"].sum()),
                seconds={"prep_per_mesh_median": float(np.nanmedian(base["prep_s"])),
                         "feat_per_mesh_median": float(np.nanmedian(base["feat_s"])),
                         "template_map_per_mesh_median": float(np.nanmedian(tpl_s)),
                         "pair_median": float(np.nanmedian(pair_s)), "pair_mean": float(np.nanmean(pair_s)),
                         "pairs_total_h": float(np.nansum(pair_s) / 3600)},
                vertices={"median": float(np.median(base["n_vertices"])), "max": int(base["n_vertices"].max())},
                **U.stamp())
    blmm.atomic_savez(root / f"{v['name']}.npz", **{k: base[k] for k in ("subjects", "topologies", "paths")}, **D,
                      U_tpl=Ut.astype(np.float32), cs=base["cs"], n_vertices=base["n_vertices"])
    U.atomic_json(root / f"{v['name']}.json", info)
    subj, topo = base["subjects"], base["topologies"]
    csv = root / f"pairs_{v['name']}.csv"
    tmp = csv.with_name(f".{csv.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        fh.write("subject_a,topology_a,subject_b,topology_b,d_pair_fr,d_pair_sr,d_tpl_fr,d_tpl_sr\n")
        for i, j in zip(I, J):
            fh.write(f"{subj[i]},{topo[i]},{subj[j]},{topo[j]}," +
                     ",".join(f"{D[k][i, j]:.9g}" for k in ("D_pair_fr", "D_pair_sr", "D_tpl_fr", "D_tpl_sr")) + "\n")
    os.replace(tmp, csv)
    print(f"[ulr-eval] merge {v['name']}: {len(I)} coppie, NaN {info['nan']}, template {diag}", flush=True)


# ----------------------------------------------------------------------------------------------- smoke

def smoke(a) -> None:
    """Controlli 1 e 2 (docstring del modulo)."""
    import torch
    sys.path.insert(0, str(U.ULRSSM_DIR))
    from utils.fmap_util import fmap2pointmap
    os.makedirs(a.cache, exist_ok=True)
    device = torch.device("cuda")
    net, input_type = U.load_net(Path(a.ckpt), device, a.cache)
    perm = U.ulrssm()["Similarity"](tau=U.CFG["tau"], hard=True)
    out = {"ckpt": str(a.ckpt), "input_type": input_type, "equal_fraction": [], "timing": {}, **U.stamp()}
    v = U.view("hifi3d")
    _P.update(view=v, cache=a.cache)
    small = [k for k, (_, t, _) in enumerate(v["items"]) if t in ("down8k", "remesh")][:6]
    prepd = dict(_prep(k) for k in small)
    G = {k: on_gpu(net, r, a.cache, v["cs_ref"], device) for k, r in prepd.items()}
    for i, j in zip(small[0::2], small[1::2]):
        x, y = G[i], G[j]
        fx, fy = x["f"].float(), y["f"].float()
        Pyx = perm(torch.bmm(fy[None], fx[None].transpose(1, 2)))[0]       # validate_single, ramo isometrico
        et_y = (y["e"].T * y["m"][None])
        Cxy = et_y @ (Pyx @ x["e"])
        ref = fmap2pointmap(Cxy, x["e"], y["e"])
        got = U.p2p(x["f"], y["f"], x["e"], y["e"], y["m"])
        # dove nasce lo scarto: Pyx (argmax delle similarita') e, a parita' di Pyx, il passo spettrale (nn_query)
        pyx_ref, pyx_got = Pyx.argmax(1), U._argmax_rows(fy, fx)
        same_c = fmap2pointmap(et_y @ x["e"][pyx_got], x["e"], y["e"])
        diff = ref != got
        mm = torch.linalg.norm(x["V"][ref] - x["V"][got], dim=1)
        rec = {"pair": [v["items"][i][:2], v["items"][j][:2]], "equal": float((~diff).float().mean()),
               "pyx_equal": float((pyx_ref == pyx_got).float().mean()),
               "equal_with_same_pyx": float((same_c == got).float().mean()),
               "mm_between_images_where_different_mean": float(mm[diff].mean()) if diff.any() else 0.0,
               "mm_between_images_where_different_max": float(mm[diff].max()) if diff.any() else 0.0,
               "fr_ref_vs_got": [U.rigid_rms(x["V"][ref], y["V"], y["w"]), U.rigid_rms(x["V"][got], y["V"], y["w"])]}
        out["equal_fraction"].append(rec)
        print(f"[ulr-smoke] coppia {i}-{j}: {rec}", flush=True)
    for vn in U.VIEWS:
        v = U.view(vn)
        _P.update(view=v)
        ks = [k for k, (_, t, _) in enumerate(v["items"]) if t in ("original", "up60k", "scan_gallery")][:2]
        t0 = time.perf_counter()
        prepd = dict(_prep(k) for k in ks)
        t1 = time.perf_counter()
        G = {k: on_gpu(net, r, a.cache, v["cs_ref"], device) for k, r in prepd.items()}
        t2 = time.perf_counter()
        d = U.pair_rms(G[ks[0]], G[ks[1]])
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        out["timing"][vn] = {"meshes": [v["items"][k][:2] for k in ks], "n": [prepd[k]["n"] for k in ks],
                             "prep_s": t1 - t0, "feat_s": t2 - t1, "pair_s": t3 - t2, "d": list(d),
                             "flipped": [bool(prepd[k]["flipped"]) for k in ks]}
        print(f"[ulr-smoke] {vn}: {out['timing'][vn]}", flush=True)
    U.atomic_json(U.OUT / "smoke" / f"smoke_{os.environ.get('SLURM_JOB_ID', 'manual')}.json", out)


# ---------------------------------------------------------------------------------------------- select

def select(a) -> None:
    runs = {t: json.loads((U.RUNS / t / "train.json").read_text()) for t in ("faces_xyz", "faces_wks")
            if (U.RUNS / t / "train.json").exists() and (U.RUNS / t / "final.pth").exists()}
    if not runs:
        raise SystemExit("nessun training completato")
    err = {t: float(r["last_val"]["all"]["err"]) for t, r in runs.items()}
    best = min(err, key=err.get)
    sel = {"rule": "errore di validazione medio minore (identita' di validazione dei generatori, original -> original e "
                   "-> noisy), checkpoint finale; fissata prima di valutare le viste",
           "val_err": err, "val_flip": {t: float(r["last_val"]["all"]["flip"]) for t, r in runs.items()},
           "main": best, "ckpt": str((U.RUNS / best / "final.pth").relative_to(U.REPO)), **U.stamp()}
    U.atomic_json(U.OUT / "selection.json", sel)
    print(f"[ulr-select] {json.dumps(sel)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("smoke", "run", "merge", "select"))
    p.add_argument("--view", default="hifi3d")
    p.add_argument("--spec", type=Path, default=None, help="vista nuova (formato in ulr_common.py), es. Ava-256")
    p.add_argument("--ckpt", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--cache", default="")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    {"smoke": smoke, "run": run, "merge": merge, "select": select}[a.step](a)


if __name__ == "__main__":
    main()
