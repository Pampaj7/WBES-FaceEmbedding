#!/usr/bin/env python3
"""Ava-256, prova confermativa: i descrittivi della sez. 9 fuori da ``conf_eval.py`` su un dominio da spec, calcolo e
lettura (PROTOCOL_CONFERMATIVO, sez. 5, 6, 9). Collaudati su HIFI3D con lo STESSO codice (spec
``aau/ava256/spec/hifi3d_full.json``, ``conf_collaudo.py desc`` e ``desc-eval``).

    aau/run.sh aau/ava256/conf_desc.py spectral --spec <json> --workers 64                    (repo)
    AAU_NV=--nv aau/outlineB/run_o3d.sh aau/ava256/conf_desc.py varifold --spec <json> [--shard k/n]   (GPU)
    aau/outlineB/run_o3d.sh aau/ava256/conf_desc.py icp-same --spec <json> --workers 64
    aau/baselines/run_bl.sh aau/ava256/conf_desc.py arcface --spec <json> --stage /tmp/<job> --workers 32
    aau/competitors/gw_percettivi/gp_run.sh aau/ava256/conf_desc.py gw-prep|gw-run|gw-merge|perc --spec <json>
    AAU_NV=--nv aau/run.sh aau/ava256/conf_desc.py ulr-run --spec <json> --stage /tmp/<job> [--shard k/n]  (GPU, v6)
    v3_work/unified_gt/run.sh aau/ava256/conf_desc.py ulr-merge --spec <json>                       (v6)
    v3_work/unified_gt/run.sh aau/ava256/conf_desc.py eval --spec <json> --workers 64     (LEGGE LE GT del dominio)
    (conf.sbatch, passi desc-*)

Calcolo (nessuna GT; uscite in ``<work>/desc``, riassunti in ``<evidence>/desc``). Mesh = soggetti dello spec x 6
topologie (``blmm.register_spec``), crop compresa: nella matrice le righe crop venivano da script a parte
(``fm_varifold.py``, ``fm_compute.py``), qui un solo calcolo con le stesse funzioni.
  spectral   ShapeDNA, HKS, WKS (``comp_spectral.spectral`` via ``fm_compute._spec``, parametri di
             competitors_hifi3d) -> ``spectral.npz`` (formato di ``fm_compute.py spectral``).
  varifold   prodotti interni varifold in mm fra tutte le mesh (``bp.varifold_measure``, ``geometric_kernel.
             inner_products`` kind ``varifold``, sigma ``bp.VARIFOLD_SIGMAS_MM``, blocco 8192 come ``fm_varifold.py``) su
             GPU; righe i con i mod n = k (``--shard k/n``) -> ``varifold_<k>of<n>.npz`` (formato di ``bp_varifold.py``).
  icp-same   ICP + Chamfer per coppia fra soggetti DIVERSI con la STESSA topologia (la regola di ``fm_compute.icp_same``:
             ``blmm_pairs.run_items`` passo ``fast``, modi mm e cs, seme = indice della coppia i < j) ->
             ``icp_same_<modo>.npz``. Il NICP per coppia su queste righe non si calcola (come nella matrice).
  arcface    l'insegnante ArcFace-render di G0 (``g0_teacher.frame`` e ``run``, importati, come ``fm_arcface.py``) su un
             insieme in memoria = le mesh della vista; rotazione attesa ``desc.arcface.rotation`` (Ava-256 e HIFI3D:
             ``x180``: R del frame ~ identita'). Se il frame check (detector sui render, nessuna GT) sceglie un'altra
             rotazione unica si usa quella e si dichiara in ``frame_note.json`` (la regola di fm_arcface.py); senza una
             scelta unica ArcFace non si calcola (``na.json``, riportato come non applicabile) ->
             ``arcface/teacher/<vista>/teacher.npz``.
  gw-prep, gw-run[:k/n], gw-merge   Gromov-Wasserstein (``gp_gw.run_prep``, ``run_view``, ``run_merge``) nella
             configurazione dichiarata ``desc.gw.config`` sulle coppie di ``gp_common.pairs`` (all_cross).
  perc       render e feature percettive SOLO delle varianti dichiarate ``desc.perc.variants`` (sez. 9: le altre non si
             calcolano): ``gp_percettivi._render`` sulle sole colorazioni che servono, ``extractor``, LPIPS in forma
             chiusa (``lpips_matrix``, ``lpips_check``), uscite con ``gp_common.write_view``.
  ulr-run[:k/n], ulr-merge (v6)   ULRSSM, variante dichiarata ``desc.ulrssm.variant`` (``faces_xyz``): ``ulr_eval.run``
             (mesh preparate come in training, feature del checkpoint della selezione, per coppia sulle coppie di soggetti
             diversi di ``ulr_common.pairs``, stessa topologia compresa, e template -> mesh) e ``ulr_eval.merge``
             (variante via template con ``train_fr_sr.factor``), funzioni importate invariate; prima: sha256 di
             checkpoint e ``selection.json`` contro lo spec, variante = quella della selezione, clone di ULRSSM
             (``ulr_common.ULRSSM_DIR``) al commit dello spec e senza modifiche, GPU della classe ``desc.ulrssm.gpu``
             (L40S, quella della valutazione di sviluppo). Vista da spec generato come per GW, uscite di ulr_common
             ridirette in ``<work>/desc/gp/ulrssm`` (plug-in ``faces_xyz/<vista>.npz``: ``D_pair_fr``, ``D_pair_sr``,
             ``D_tpl_fr``, ``D_tpl_sr``); operatori di ULRSSM nella cache ``--stage`` del job.
  GW e percettivi: vista di ``gp_common.view`` da uno spec generato (``<work>/desc/gp_spec.json``: dir, soggetti,
  topologie, frame, L_d e CS_ref di ``blmm.params`` della vista dello spec), cartelle di gp_common ridirette in
  ``<work>/desc/gp`` (plug-in ``<metodo>/<variante>/<vista>.npz``) e ``<work>/desc/gp_work``; le scelte del pilota
  (``desc.{gw,perc}.pilot_choice``, sha256 nello spec) copiate accanto, controllate contro le varianti dello spec. Con
  ``desc.subsample_subjects`` (SOLO collaudo) GW e percettivi sulle sole mesh di quei soggetti.

Lettura (``eval``; con ``freeze`` PRIMA ``conf_common.verify``):
  righe = ``conf_eval.build_rows`` (soggetti diversi a < b, topologie diverse: 597.000 su Ava-256) + stessa topologia
  (soggetti diversi a < b, t_a = t_b, per le 6 topologie: 19.900 x 6 = 119.400, sez. 5); gruppi ``GROUPS`` dalle
  definizioni di ``fm_eval.groups_of``: ``nocrop``, ``crop``, ``all_cross``, per perturbazione P senza crop
  (``con_perturbazione``), stessa topologia (``tutte`` e per topologia).
  colonne: quelle di ``conf_eval.model_columns`` e ``geo_and_protocols`` (famiglia, d_F non cal., testa, NICP su
  template cs, protocolli) + i descrittivi: C3M e123 (``columns.desc_models``, ``fact_paired.distances`` col c dello
  spec), fit B (``<modello>_vb_fr``, ``_vb_sr``, ``_vb_comp``: ``vb_beta`` di ``fit_e1.npz`` e ``fit_e2_crop.npz``,
  ``bp.mesh_distances``, composizione con il k di ``bp_paired_e2.k_values``), varifold
  (``bp_paired.varifold_distances``), spettrali (``fm_eval.spectral_columns``), taglia stimata (``est_cs``, cs di
  ``blmm.scalars_of``), ArcFace (``arcface_normals``, 1 - coseno), GW e percettivi (``fm_eval.plugin_columns`` sulle
  cartelle ``columns.desc_plugins``: ``plugin:<metodo>/<variante>|fr`` e ``|sr``; v6: ULRSSM ``|pair_fr``, ``|pair_sr``,
  ``|tpl_fr``, ``|tpl_sr``, ognuna con la sua GT); sulle righe di stessa topologia ICP +
  Chamfer da ``icp_same``, il NICP per coppia e i protocolli non esistono (non applicabili su quei gruppi).
  letture (sez. 6), con le repliche di ``conf_eval`` (``default_rng(boot_seed)``, 1000, Spearman pesato c_a c_b di
  ``fm_eval.replicas``): ``confermativa (+inf)`` = righe con GT e colonne PER MESH della famiglia finite, metodi per
  coppia della famiglia non finiti = +inf; ``mascherata (descrittiva)`` = in piu' tutte le colonne per coppia della
  famiglia finite. Un descrittivo con buchi su quelle righe va in un passo a parte sulle sue righe valide, con i
  riferimenti sulle stesse righe (regola dei plug-in di fm_eval, passo ``righe proprie di <colonna>``). Record di
  ``fm_eval.records``: Spearman con IC, delta ``vs_nostro`` (d_F cal., d_F non cal. per FR, d_P per SR, per seme) e
  ``scala_allineamento``. ``non_risolto`` per gruppo (sez. 8, 9.1, ``unresolved``): mesh fallite > 2%, righe non finite
  > 5%, ripieghi del detector > 2% (protocolli) sulle topologie del gruppo.
  Uscite in ``<evidence>/desc``: ``spearman.csv``, ``deltas.csv``, ``controls.json``, ``results.md``.
"""

from __future__ import annotations

import os

# un thread BLAS per processo (la parallelizzazione e' per processo), come fm.sbatch (spettrali) e g0_teacher.py
# (ArcFace), PRIMA di numpy; i runner che fissano i thread (gp_run.sh: GP_THREADS) non si toccano
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
for _p in (AAU / "baselines_mm", AAU / "baselines_param", AAU / "full_matrix", THIS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import conf_common as cc  # noqa: E402

TOPOS = ("original", "remesh", "down8k", "up60k", "noisy", "crop")       # fm_eval.TOPOS
NOCROP = TOPOS[:-1]
GROUPS = [("nocrop", "-"), ("crop", "crop"), ("all_cross", "-")] + [("con_perturbazione", t) for t in NOCROP] + \
    [("stessa_topologia", "tutte")] + [("stessa_topologia", t) for t in TOPOS]
VARIFOLD_BLOCK = 8192
CONF, MASKED = "confermativa (+inf)", "mascherata (descrittiva)"


def setup(spec: Path) -> tuple[dict, str]:
    """(spec, vista di blmm / bp registrata dallo spec)."""
    import bp
    s = cc.load(spec)
    if s.get("freeze"):
        cc.verify(s)
    if s.get("desc", {}).get("subsample_subjects") and not s.get("collaudo"):
        raise SystemExit("desc.subsample_subjects solo nei collaudi")
    return s, bp.register_spec(s["_path"])


def desc_dir(s: dict) -> Path:
    return cc.work(s) / "desc"


def ev_dir(s: dict) -> Path:
    return cc.evidence(s) / "desc"


def items(name: str) -> list[tuple[str, str, Path]]:
    import blmm
    return [(x, t, blmm.mesh_path(name, x, t)) for x in blmm.subjects(name) for t in blmm.TOPOLOGIES]


def pool(workers: int):
    return mp.get_context("fork").Pool(workers)


def na_mark(s: dict, what: str, why: str) -> None:
    p = desc_dir(s) / "na.json"
    na = json.loads(p.read_text()) if p.exists() else {}
    na[what] = {"motivo": why, **cc.stamp()}
    cc.atomic_json(p, na)
    cc.atomic_json(ev_dir(s) / "na.json", na)


# ------------------------------------------------------------------------------------------ spettrali

def step_spectral(spec: Path, workers: int) -> None:
    import blmm
    sys.path.insert(0, str(AAU / "competitors"))
    import fm_compute as fc
    s, name = setup(spec)
    out = desc_dir(s) / "spectral.npz"
    if out.exists():
        print(f"[conf-desc] {out}: gia' presente, salto", flush=True)
        return
    t0 = time.time()
    it = items(name)
    with pool(workers) as pl:
        res = dict(pl.map(fc._spec, [((x, t), p) for x, t, p in it], chunksize=2))
    rows = [res[(x, t)] for x, t, _ in it]
    arr = {"subjects": np.asarray([x for x, _, _ in it]), "topologies": np.asarray([t for _, t, _ in it]),
           "paths": np.asarray([blmm.rel(p) for _, _, p in it])}
    for f in ("evals", "hks", "wks"):
        arr[f] = np.stack([r[f] for r in rows])
    for f in ("n_components", "n_dropped", "area_raw", "n_verts", "identity_rel_err"):
        arr[f] = np.asarray([r[f] for r in rows])
    blmm.atomic_savez(out, **arr)
    rep = {"meshes": len(it), "components_gt1": int((arr["n_components"] > 1).sum()),
           "identity_rel_err_max": float(arr["identity_rel_err"].max()), "seconds": time.time() - t0, **cc.stamp()}
    cc.atomic_json(ev_dir(s) / "spectral.json", rep)
    print(f"[conf-desc] spettrali {name}: {json.dumps(rep)}", flush=True)


# ------------------------------------------------------------------------------------------ varifold

def _measure(task):
    import bp
    name, path = task
    return bp.varifold_measure(name, path)


def step_varifold(spec: Path, shard: str, device: str, workers: int) -> None:
    import torch
    import blmm
    import bp
    sys.path.insert(0, str(AAU / "baselines"))
    import geometric_kernel as gk
    from geometric_matrix import to_device
    s, name = setup(spec)
    k, nsh = (int(x) for x in shard.split("/"))
    out = desc_dir(s) / f"varifold_{k}of{nsh}.npz"
    if out.exists():
        print(f"[conf-desc] {out}: gia' presente, salto", flush=True)
        return
    it = items(name)
    t0 = time.time()
    with pool(workers) as pl:
        cpu = pl.map(_measure, [(name, p) for _, _, p in it], chunksize=4)
    M = [to_device(m, device) for m in cpu]
    dev = torch.cuda.get_device_name() if device == "cuda" else "cpu"
    print(f"[conf-desc] varifold {name}: {len(M)} misure in {time.time() - t0:.0f}s, device {device} ({dev})", flush=True)
    sig = bp.VARIFOLD_SIGMAS_MM
    n = len(M)
    G = np.full((len(sig), n, n), np.nan)
    rows = [i for i in range(n) if i % nsh == k]
    t1 = time.time()
    for r, i in enumerate(rows):
        for j in range(i, n):
            G[:, i, j] = G[:, j, i] = gk.inner_products(M[i], M[j], sig, kinds=("varifold",),
                                                        block=VARIFOLD_BLOCK)["varifold"].cpu().numpy()
        if r % 50 == 0:
            print(f"[conf-desc] varifold riga {r + 1}/{len(rows)}, {time.time() - t1:.0f}s", flush=True)
    blmm.atomic_savez(out, G=G, subjects=np.asarray([x for x, _, _ in it]), topologies=np.asarray([t for _, t, _ in it]),
                      atoms=np.asarray([len(m["areas"]) for m in cpu]), sigmas_mm=np.asarray(sig),
                      cell_mm=bp.VARIFOLD_CELL_MM, rows=np.asarray(rows), block=VARIFOLD_BLOCK, device=dev)
    cc.atomic_json(ev_dir(s) / f"varifold_{k}of{nsh}.json", {"meshes": n, "rows": len(rows), "device": dev,
                                                             "seconds": time.time() - t0, **cc.stamp()})
    print(f"[conf-desc] varifold {name} shard {shard}: {len(rows)} righe in {time.time() - t1:.0f}s -> {out}", flush=True)


def varifold_gram(s: dict) -> tuple[np.ndarray, list]:
    """(G (S, n, n) dagli shard, chiavi); errore se manca una cella (``bp_paired.varifold_gram`` sulla cartella dello spec)."""
    G, keys = None, None
    parts = sorted(desc_dir(s).glob("varifold_*of*.npz"))
    if not parts:
        raise FileNotFoundError(f"{desc_dir(s)}: varifold assente")
    for p in parts:
        with np.load(p) as z:
            kz = list(zip([str(x) for x in z["subjects"]], [str(t) for t in z["topologies"]]))
            if keys is not None and kz != keys:
                raise SystemExit(f"{p}: mesh diverse dagli altri shard")
            keys = kz
            G = np.asarray(z["G"]) if G is None else np.where(np.isfinite(G), G, z["G"])
    if not np.isfinite(G).all():
        raise FileNotFoundError(f"varifold incompleto ({int((~np.isfinite(G)).sum())} celle)")
    return G, keys


# ------------------------------------------------------------------------------------------ ICP, stessa topologia

def step_icp_same(spec: Path, workers: int) -> None:
    import blmm
    import blmm_pairs as bpr
    s, name = setup(spec)
    prm = blmm.params()
    subjects = blmm.subjects(name)
    iu, ju = np.triu_indices(len(subjects), 1)
    for mode in ("mm", "cs"):
        out = desc_dir(s) / f"icp_same_{mode}.npz"
        if out.exists():
            print(f"[conf-desc] {out}: gia' presente, salto", flush=True)
            continue
        t0 = time.time()
        bpr._C.clear()
        for x in subjects:
            for t in blmm.TOPOLOGIES:
                bpr._C[(x, t)] = blmm.work_coords(name, mode, blmm.mesh_path(name, x, t), prm)
        unit = blmm.unit(name, mode, prm)
        its = [((subjects[i], t), (subjects[j], t), q) for t in blmm.TOPOLOGIES for q, (i, j) in enumerate(zip(iu, ju))]
        with pool(workers) as pl:
            vals, sec, errors = bpr.run_items(pl, its, mode, "fast", 64)
        arr = {f"values_{m}": (v * unit).reshape(len(blmm.TOPOLOGIES), len(iu)) for m, v in vals.items()}
        blmm.atomic_savez(out, **arr, topologies=np.asarray(blmm.TOPOLOGIES), subjects=np.asarray(subjects), iu=iu,
                          ju=ju, unit_mm=unit, mode=mode, n_failed=len(errors), errors=np.asarray(errors[:50], dtype="U300"),
                          seconds=sec)
        rep = {"mode": mode, "pairs": len(its), "failed": len(errors), "s_per_pair_mean": float(np.mean(sec)),
               "seconds": time.time() - t0, **cc.stamp()}
        cc.atomic_json(ev_dir(s) / f"icp_same_{mode}.json", rep)
        print(f"[conf-desc] icp-same {name} {mode}: {json.dumps(rep)}", flush=True)


# ------------------------------------------------------------------------------------------ ArcFace

def arcface_path(s: dict, name: str) -> Path:
    return desc_dir(s) / "arcface" / "teacher" / name / "teacher.npz"


def step_arcface(spec: Path, stage: Path, workers: int) -> None:
    import blmm
    sys.path.insert(0, str(AAU / "diagnostics"))
    import g0_teacher as gt
    s, name = setup(spec)
    cc.require_host(s["desc"]["arcface"]["node"], "ArcFace")          # v7: nodo di G0 e del collaudo
    out = arcface_path(s, name)
    if out.exists():
        print(f"[conf-desc] {out}: gia' presente, salto", flush=True)
        return
    mesh_root = stage / "meshes"
    mesh_root.mkdir(parents=True, exist_ok=True)
    link = mesh_root / name
    if not link.exists():
        link.symlink_to(cc.rp(s["dir"]).resolve())
    subjects = blmm.subjects(name)
    want = s["desc"]["arcface"]["rotation"]
    gt.G0 = desc_dir(s) / "arcface"
    gt.SETS[name] = ("gen", name, gt.VIEW6, want)
    base = gt.subjects_of
    gt.subjects_of = lambda n: subjects if n == name else base(n)  # noqa: E731
    fc = gt.frame(name, mesh_root)
    note = {"expected": want, "chosen": fc["chosen"], "score": fc["score"]}
    if not fc["agrees"]:
        if fc["chosen"] is None:
            na_mark(s, "arcface_normals", f"frame check senza una scelta unica ({fc['score']}): ArcFace non calcolato")
            raise SystemExit(f"{name}: frame check senza una scelta unica ({fc['score']})")
        gt.SETS[name] = ("gen", name, gt.VIEW6, fc["chosen"])
        fc = gt.frame(name, mesh_root)
    note["used"] = gt.SETS[name][3]
    cc.atomic_json(out.parent / "frame_note.json", note)
    gt.run(name, stage / "render", mesh_root, workers, 512)
    (ev_dir(s) / "arcface").mkdir(parents=True, exist_ok=True)
    for f in ("frame_note.json", "frame_check.json", "timing.json", "camera.json", "arcface_align.json"):
        if (out.parent / f).exists():
            shutil.copyfile(out.parent / f, ev_dir(s) / "arcface" / f)


# ------------------------------------------------------------------------------------------ GW e percettivi

def domain_spec(s: dict, name: str, path: Path) -> Path:
    """Spec di dominio di ``gp_common.view`` / ``ulr_common.view`` (dir, soggetti, topologie, frame, L_d e CS_ref di
    ``blmm.params`` della vista dello spec; ``desc.subsample_subjects`` solo nei collaudi)."""
    import blmm
    prm = blmm.params()["domains"][name]
    sub = s["desc"].get("subsample_subjects")
    gs = {"name": name, "dir": s["dir"], "subjects": sorted(sub or s["subjects"]), "topologies": s["topologies"],
          "frame": s["frame"], "L": float(prm["L"]), "cs_ref": float(prm["cs_ref"]),
          "note": "generato da conf_desc.py" + (": SOTTOCAMPIONE di collaudo" if sub else "")}
    cc.atomic_json(path, gs)
    return path


def gp_setup(spec: Path):
    """(spec, vista, spec di gp_common, moduli gc, gp_gw, gp_percettivi) con le cartelle ridirette."""
    import blmm
    sys.path.insert(0, str(AAU / "competitors" / "gw_percettivi"))
    import gp_common as gc
    import gp_gw
    import gp_percettivi as gpp
    s, name = setup(spec)
    D = desc_dir(s)
    gc.OUT, gc.WORK = D / "gp", D / "gp_work"
    gp_gw.GW, gp_gw.WORK, gpp.PERC = gc.OUT / "gw", gc.WORK, gc.OUT / "percettivi"
    for kind, key in (("gw", "gw"), ("percettivi", "perc")):
        src = cc.rp(s["desc"][key]["pilot_choice"])
        got = cc.sha256(src)
        if got != s["desc"][key]["pilot_choice_sha256"]:
            raise SystemExit(f"{src}: sha256 {got} diverso dallo spec")
        (gc.OUT / kind).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, gc.OUT / kind / "pilot_choice.json")
    if gp_gw.choice()["config"] != s["desc"]["gw"]["config"]:
        raise SystemExit("configurazione di GW dello spec diversa dalla scelta del pilota")
    ch = json.loads((gc.OUT / "percettivi" / "pilot_choice.json").read_text())["choice"]
    if ch != s["desc"]["perc"]["variants"]:
        raise SystemExit(f"varianti percettive dello spec {s['desc']['perc']['variants']} diverse dal pilota {ch}")
    return s, name, domain_spec(s, name, D / "gp_spec.json"), gc, gp_gw, gpp


def step_gw(spec: Path, what: str, shard: str, workers: int) -> None:
    s, name, gps, gc, gp_gw, _ = gp_setup(spec)
    cfg = s["desc"]["gw"]["config"]
    if what == "prep":
        gp_gw.run_prep(name, gps, workers, cfg.startswith("geo"))
    elif what == "run":
        k, n = (int(x) for x in shard.split("/"))
        for c in gp_gw.components(cfg):
            gp_gw.run_view(name, c, k, n, workers)
    else:
        for c in dict.fromkeys(gp_gw.components(cfg) + [cfg]):
            gp_gw.run_merge(name, gps, c)
        info = json.loads((gc.col_dir("gw", gp_gw.tag(cfg), name) / f"{name}.json").read_text())
        cc.atomic_json(ev_dir(s) / "gw.json", info)


def step_perc(spec: Path, workers: int, device: str) -> None:
    """``gp_percettivi.run_view`` ristretto alle varianti dichiarate (render, embedding, LPIPS: funzioni importate)."""
    import torch
    s, name, gps, gc, _, gpp = gp_setup(spec)
    want = s["desc"]["perc"]["variants"]                       # {metodo: variante}
    todo = {m: v for m, v in want.items() if not gc.out_npz(m, v, name).exists()}
    if not todo:
        print(f"[conf-desc] percettivi {name}: uscite gia' presenti, salto", flush=True)
        return
    gpp.VARIANTS = tuple(sorted(set(want.values())))          # solo le colorazioni delle varianti dichiarate
    v = gc.view(name, gps)
    gpp._C.update(view=v, H=gpp.camera(v))
    n = len(v["items"])
    topo = np.asarray([t for _, t, _ in v["items"]])
    t0 = time.time()
    with pool(workers) as pl:
        res = pl.map(gpp._render, list(enumerate(p for _, _, p in v["items"])), chunksize=1)
    t_render = time.time() - t0
    ok = np.asarray([not r["fail"] for r in res])
    fails = [f"{v['items'][r['k']][0]}|{v['items'][r['k']][1]}|{r['fail']}" for r in res if r["fail"]]
    img = np.zeros((n, len(gpp.VARIANTS), 2, len(gpp.YAWS), gpp.SIZE, gpp.SIZE), np.uint8)
    for r in res:
        if not r["fail"]:
            img[r["k"]] = r["img"]
    clipped = np.stack([r["clip"] for r in res])
    cs = np.asarray([r["cs"] for r in res])
    sec_render = np.asarray([r["t"] for r in res])
    deg = np.asarray([r["deg"] / max(r["n_tri"], 1) for r in res])
    del res
    torch.set_num_threads(int(os.environ.get("GP_THREADS", "1")))
    base = {"view": name, "n_meshes": n, "failed": fails, "yaws": list(gpp.YAWS), "size": gpp.SIZE,
            "camera_half_mm": gpp._C["H"], "cam_k": gpp.CAM_K, "L_mm": v["L"], "cs_ref_mm": v["cs_ref"],
            "device": device, "threads": torch.get_num_threads(), "workers_render": workers,
            "rendered_variants": list(gpp.VARIANTS),
            "clipped_meshes": {m: int(clipped[:, k].any(1).sum()) for k, m in enumerate(gc.MODES)},
            "degenerate_triangle_fraction_by_topology": {
                t: {"median": float(np.nanmedian(deg[topo == t])), "max": float(np.nanmax(deg[topo == t]))}
                for t in sorted(set(topo))},
            "render_s_per_mesh_median": float(np.nanmedian(sec_render)), "render_wall_s": t_render,
            "spec": str(gps), **cc.stamp()}
    rep = {}
    for m, var in todo.items():
        q = gpp.VARIANTS.index(var)
        t1 = time.time()
        if m == "lpips":
            model = gpp.lpips_model(device)
            D, check = {}, {}
            for k, mode in enumerate(gc.MODES):
                per = []
                for y in range(len(gpp.YAWS)):
                    M = gpp.lpips_matrix(model, img[:, q, k, y], device)
                    check[f"{mode}|{gpp.YAWS[y]:+.0f}"] = gpp.lpips_check(model, img[:, q, k, y], M, device)
                    per.append(M)
                M = np.mean(per, axis=0)
                M[~ok] = M[:, ~ok] = np.nan
                np.fill_diagonal(M, 0.0)
                D[mode] = M
            info = {**base, "method": m, "variant": var, "wall_s": time.time() - t1, "check_gram_vs_forward_max_abs":
                    check, "distance": "media sulle 3 viste della LPIPS 0.1 alex fra i render della stessa vista (Gram)"}
            gc.write_view(m, var, v, D, {"cs": cs}, info)
        else:
            f = gpp.extractor(m, device)
            E, D = {}, {}
            for k, mode in enumerate(gc.MODES):
                E[mode] = f(img[:, q, k].reshape(-1, gpp.SIZE, gpp.SIZE)).reshape(n, len(gpp.YAWS), -1)
                Z = E[mode].astype(np.float64).mean(1)
                Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
                M = np.clip(1.0 - Z @ Z.T, 0.0, None)
                M[~ok] = M[:, ~ok] = np.nan
                np.fill_diagonal(M, 0.0)
                D[mode] = 0.5 * (M + M.T)
            info = {**base, "method": m, "variant": var, "embed_wall_s": time.time() - t1,
                    "distance": "1 - coseno fra embedding mediati sulle viste e rinormalizzati"}
            gc.write_view(m, var, v, D, {f"E_{x}": E[x].astype(np.float32) for x in gc.MODES} | {"cs": cs}, info)
        rep[f"{m}/{var}"] = {k: info[k] for k in info if k not in base or k in ("failed", "clipped_meshes")}
        print(f"[conf-desc] percettivi {name} {m} {var}: {n} mesh in {time.time() - t1:.0f}s", flush=True)
    cc.atomic_json(ev_dir(s) / "perc.json", {"base": base, "variants": rep})


# ------------------------------------------------------------------------------------------ ULRSSM (v6)

def ulr_setup(spec: Path):
    """(spec, vista, spec di ulr_common, modulo ulr_eval, checkpoint) con i controlli di ``desc.ulrssm`` e le uscite di
    ulr_common ridirette in ``<work>/desc/gp/ulrssm``."""
    import subprocess
    sys.path.insert(0, str(AAU / "competitors" / "ulrssm"))
    import ulr_common as U
    import ulr_eval as ue
    s, name = setup(spec)
    u = s["desc"]["ulrssm"]
    for key in ("checkpoint", "selection"):
        got = cc.sha256(cc.rp(u[key]))
        if got != u[f"{key}_sha256"]:
            raise SystemExit(f"ULRSSM {key}: sha256 {got} diverso dallo spec")
    sel = json.loads(cc.rp(u["selection"]).read_text())
    if sel["main"] != u["variant"] or cc.rp(sel["ckpt"]).resolve() != cc.rp(u["checkpoint"]).resolve():
        raise SystemExit(f"ULRSSM: selezione {sel['main']} / {sel['ckpt']} diversa dallo spec")
    git = ["git", "-C", str(U.ULRSSM_DIR)]
    head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(git + ["status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if head != u["commit"] or head != U.ULRSSM_COMMIT or dirty:
        raise SystemExit(f"clone di ULRSSM {U.ULRSSM_DIR}: commit {head}, atteso {u['commit']}, modifiche {dirty[:200]!r}")
    U.OUT = desc_dir(s) / "gp" / "ulrssm"
    U.OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cc.rp(u["selection"]), U.OUT / "selection.json")
    return s, name, domain_spec(s, name, desc_dir(s) / "ulr_spec.json"), ue, cc.rp(u["checkpoint"])


def step_ulr(spec: Path, what: str, shard: str, workers: int, stage: Path | None) -> None:
    s, name, us, ue, ck = ulr_setup(spec)
    u = s["desc"]["ulrssm"]
    a = argparse.Namespace(view=name, spec=us, shard=shard, tag=u["variant"], ckpt=str(ck), cache=str(stage or ""),
                           workers=workers, pairs="subjects")
    if what == "run":
        import torch
        if not torch.cuda.is_available() or u["gpu"] not in torch.cuda.get_device_name():
            raise SystemExit(f"ULRSSM: GPU richiesta {u['gpu']}")
        os.makedirs(a.cache, exist_ok=True)
        ue.run(a)
    else:
        ue.merge(a)
        cc.atomic_json(ev_dir(s) / "ulrssm.json",
                       json.loads((ue.U.OUT / u["variant"] / f"{name}.json").read_text()))
        # v7: letture dichiarate su Ava-256 = desc.ulrssm.readings_declared (pair_fr, tpl_sr); le altre declared false
        pj = ue.U.OUT / u["variant"] / "fm_plugin.json"
        cfg = json.loads(pj.read_text())
        for r, info in cfg["readings"].items():
            info["declared"] = r in u["readings_declared"]
        cc.atomic_json(pj, cfg)


# ------------------------------------------------------------------------------------------ lettura: colonne

def same_rows(R: dict, zes) -> dict:
    """Le righe di stessa topologia (soggetti diversi a < b, t_a = t_b) in coda a quelle di ``conf_eval.build_rows``."""
    idx, n = R["idx"], len(R["subjects"])
    i, j = np.triu_indices(n, 1)
    add = {k: [R[k]] for k in ("sa", "sb", "ia", "ib", "ta", "tb")}
    for t in zes.TOPOLOGIES:
        r = idx.rows(t)
        for k, v in (("sa", i), ("sb", j), ("ia", r[i]), ("ib", r[j]), ("ta", np.full(len(i), t)),
                     ("tb", np.full(len(i), t))):
            add[k].append(v)
    return {**R, **{k: np.concatenate(v) for k, v in add.items()}, "n_cross": len(R["sa"])}


def fill_icp_same(s: dict, R: dict, X: dict) -> dict:
    """ICP + Chamfer sulle righe di stessa topologia da ``icp_same_<modo>.npz``; NICP per coppia NaN su quelle righe."""
    same = R["ta"] == R["tb"]
    sub = R["subjects"]
    info = {}
    for mode in ("mm", "cs"):
        p = desc_dir(s) / f"icp_same_{mode}.npz"
        c = f"{mode}_rigid_icp_chamfer"
        if f"{mode}_nicp_p2tri" in X:
            X[f"{mode}_nicp_p2tri"][same] = np.nan
        if c not in X:
            continue
        X[c][same] = np.nan
        if not p.exists():
            info[c] = "icp_same assente"
            continue
        with np.load(p) as z:
            if [str(x) for x in z["subjects"]] != sub:
                raise SystemExit(f"{p}: soggetti diversi dallo spec")
            q = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(z["iu"], z["ju"]))}
            tk = {str(t): k for k, t in enumerate(z["topologies"])}
            vals = np.asarray(z["values_rigid_icp_chamfer"], np.float64)
        rows = np.flatnonzero(same)
        X[c][rows] = vals[[tk[t] for t in R["ta"][rows]], [q[(a, b)] for a, b in zip(R["sa"][rows], R["sb"][rows])]]
        info[c] = {"rows": int(len(rows)), "nonfinite": int((~np.isfinite(X[c][rows])).sum())}
    return info


def need(s: dict, path: Path, what: str, absent: dict) -> bool:
    """La sorgente c'e'? Nei collaudi un'assenza si registra; su Ava-256 e' un errore, salvo i non applicabili di
    ``na.json`` (ArcFace senza scelta del frame)."""
    if path.exists():
        return True
    na = desc_dir(s) / "na.json"
    if na.exists() and what in json.loads(na.read_text()):
        absent[what] = json.loads(na.read_text())[what]["motivo"]
        return False
    if s.get("collaudo"):
        absent[what] = f"{path} assente (collaudo)"
        return False
    raise SystemExit(f"{what}: {path} assente")


def fit_hosts(s: dict, root: Path) -> dict:
    """{file del fit B: nodo} da ``fit.npz``, ``fit_e1.npz``, ``fit_e2_crop.npz``: deve essere ``nodes.bp`` (come la testa
    di R6). Un'uscita senza ``host`` (fit di sviluppo precedenti alla v8) e' ammessa solo nei collaudi."""
    out = {}
    for f in ("fit.npz", "fit_e1.npz", "fit_e2_crop.npz"):
        with np.load(root / f) as z:
            h = str(z["host"]) if "host" in z.files else None
        out[f] = h
        if h is None and s.get("collaudo"):
            continue
        if h is None or h.split(".")[0] != s["nodes"]["bp"].split(".")[0]:
            raise SystemExit(f"{root / f}: fit B dal nodo {h}, obbligato {s['nodes']['bp']}")
    return out


def desc_columns(s: dict, name: str, R: dict, fm) -> tuple[dict, dict, dict, dict]:
    """({colonna: vettore sulle righe}, {colonna: mesh fallite}, meta dei plug-in, controlli) dei descrittivi."""
    import conf_eval as ce
    import bp
    import bp_paired as bpp
    import bp_paired_e2 as bpe2
    blmm, fp = fm.blmm, fm.fp
    idx, ia, ib = R["idx"], R["ia"], R["ib"]
    D, cols, failed, ctrl, absent = {}, {}, {}, {}, {}
    for arm, m in (s["columns"].get("desc_models") or {}).items():          # C3M e123
        if not need(s, cc.rp(m["embeddings"]), arm, absent):
            continue
        Z, ck = ce.embeddings(cc.rp(m["embeddings"]), idx)
        if ck.resolve() != cc.rp(m["checkpoint"]).resolve():
            raise SystemExit(f"{arm}: embedding col checkpoint {ck}, atteso {m['checkpoint']}")
        for k, d in fp.distances(Z, ia, ib, ck, (float(m["c"]), float(m["c"]))).items():
            if k in ("form_cal", "form", "shape"):
                cols[f"{arm}|{k}"] = d
                failed[f"{arm}|{k}"] = ~np.isfinite(Z).all(1)
    K = bpe2.k_values()
    for m in bp.MODELS:                                                     # fit B
        root = bp.out_dir(name, m)
        if not need(s, root / "fit_e1.npz", f"{m}_vb", absent):
            continue
        hosts = fit_hosts(s, root)                                         # v8: nodo obbligato del fit B
        ctrl[f"{m}_hosts"] = hosts
        ctx = bpe2.fit_context(name, m)
        beta = np.full((len(idx.keys), ctx["k_id"]), np.nan)
        seen = np.zeros(len(idx.keys), bool)
        ref = None
        for f in ("fit_e1.npz", "fit_e2_crop.npz"):
            with np.load(root / f) as z:
                pos = np.asarray([idx.pos[k] for k in fm.keys_of(z)])
                beta[pos], seen[pos] = np.asarray(z["vb_beta"]), True
                if f == "fit_e1.npz":
                    ref = (pos, {d: np.asarray(z[f"D_vb_{d}"]) for d in ("fr", "sr")})
        if not seen.all():
            raise SystemExit(f"{m}: fit B assente per {int((~seen).sum())} mesh")
        Dm = bp.mesh_distances(beta, ctx)
        ctrl[f"{m}_vb_nocrop_vs_saved_max_abs"] = {d: float(np.nanmax(np.abs(Dm[d][np.ix_(ref[0], ref[0])] - ref[1][d])))
                                                   for d in ("fr", "sr")}
        D[f"{m}_vb_fr"], D[f"{m}_vb_sr"] = Dm["fr"], Dm["sr"]
        D[f"{m}_vb_comp"] = bp.composition(bp.identity_sizes(beta, ctx), Dm["sr"], ctx, K[m])
        for d in ("fr", "sr", "comp"):
            failed[f"{m}_vb_{d}"] = ~np.isfinite(beta).all(1)
    try:                                                                    # varifold
        G, kv = varifold_gram(s)
        o = fm.ordered(kv, idx)
        D["varifold"] = bpp.varifold_distances(G[:, o][:, :, o])
    except FileNotFoundError as e:
        if not s.get("collaudo"):
            raise SystemExit(f"varifold: {e}")
        absent["varifold"] = str(e)
    sp = desc_dir(s) / "spectral.npz"                                       # spettrali
    if need(s, sp, "spectral", absent):
        with np.load(sp) as z:
            o = fm.ordered(fm.keys_of(z), idx)
        D.update(fm.spectral_columns(sp, o))
    sc = blmm.scalars_of(name)                                              # taglia stimata
    D["est_cs"] = fm.log_abs([sc[blmm.rel(blmm.mesh_path(name, x, t))]["cs"] for x, t in idx.keys])
    ap = arcface_path(s, name)                                              # ArcFace
    if need(s, ap, "arcface_normals", absent):
        with np.load(ap) as z:
            D["arcface_normals"] = fm.cosine(np.asarray(z["Z"], np.float64)[fm.ordered(fm.keys_of(z), idx)])
    gp = s["columns"]["desc_plugins"]                                       # GW e percettivi
    dirs = [(f"{m}/{v}", cc.rp(gp["root"]) / m / v) for m, v in gp["variants"].items()]
    P, pmeta, missing = fm.plugin_columns(gp.get("view", name), idx.keys, dirs)
    ud = (s["desc"].get("ulrssm") or {}).get("readings_declared")      # v7: letture non dichiarate di ULRSSM
    for c in P:
        if c.startswith("plugin:ulrssm/") and ud is not None and c.split("|", 1)[1] not in ud:
            lab, fam, _, ours = pmeta[c]
            pmeta[c] = (f"{lab} (lettura non dichiarata)", fam, (), ours)   # dichiarata = False per entrambe le GT
    for m in missing:
        if not s.get("collaudo"):
            raise SystemExit(f"plug-in {m}: nessun file per la vista")
        absent[m] = "plug-in senza file per la vista (collaudo)"
    for c, M in D.items():
        cols[c] = M[ia, ib]
    for c, M in P.items():
        cols[c] = M[ia, ib]
    ctrl["absent"] = absent
    ctrl["plugin_dirs"] = [str(d) for _, d in dirs]
    return cols, failed, pmeta, ctrl


# ------------------------------------------------------------------------------------------ lettura: repliche

def topos_of(g: tuple) -> list:
    ins, p = g
    if ins in ("crop", "all_cross") or (ins == "stessa_topologia" and p == "tutte"):
        return list(TOPOS)
    if ins == "stessa_topologia":
        return [p]
    return list(NOCROP)


def unresolved(col: str, g: tuple, rows: np.ndarray, X: dict, failed: dict, fb, R: dict) -> str:
    """Motivi della sez. 8 / 9.1 della colonna nel gruppo (righe del gruppo prima di ogni maschera)."""
    from conf_eval import MESH_FAIL_MAX, PROTOCOLS, ROW_MISS_MAX
    from conf_protocols import FALLBACK_MAX
    why = []
    if len(rows):
        miss = float((~np.isfinite(X[col][rows])).mean())
        if miss > ROW_MISS_MAX:
            why.append(f"righe non finite {miss:.2%} > {ROW_MISS_MAX:.0%}")
    tp = topos_of(g)
    if col in failed:
        k = np.asarray([t in tp for _, t in R["idx"].keys])
        fr = float(failed[col][k].mean())
        if fr > MESH_FAIL_MAX:
            why.append(f"mesh fallite {fr:.2%} > {MESH_FAIL_MAX:.0%}")
    if col in PROTOCOLS and fb is not None:
        f = sum(fb.get(t, 0) for t in tp) / (len(tp) * len(R["subjects"]))
        if f > FALLBACK_MAX:
            why.append(f"ripieghi del detector {f:.2%} > {FALLBACK_MAX:.0%}")
    return "; ".join(why)


def meta_of(fm, X: dict, pmeta: dict) -> dict:
    """{colonna: (etichetta, famiglia, GT dichiarate, nostro)} per ``fm_eval.records``."""
    meta = {**fm.METHODS, **pmeta}
    for c in X:
        if c in meta:
            continue
        if c in ("now_original", "now_metrical", "realy", "stirling"):
            meta[c] = (f"{c} (landmark del rilevatore, lmk_det)", "protocolli", ("fr", "sr"), False)
        elif c.startswith("head_"):
            sd, k = c[5:].split("|")
            meta[c] = (f"testa adattata {sd}, {'d_P' if k == 'shape' else 'd_F'}", "nostri",
                       ("sr",) if k == "shape" else ("fr",), True)
        else:
            meta[c] = (c, "altro", ("fr", "sr"), False)
    return meta


def reading(name: str, X: dict, gt: dict, R: dict, mask: np.ndarray, counts, seed: int, meta: dict, tag: str,
            workers: int, fm) -> tuple[list, list, dict]:
    """Spearman e delta (``fm_eval.records``) dei metodi validi su tutte le righe di ogni parte (topologie diverse,
    stessa topologia) dentro la maschera; i metodi con buchi in una parte, in un passo sulle loro righe valide."""
    same = R["ta"] == R["tb"]
    G = {g: r for g, r in fm.groups_of(R["ta"], R["tb"], mask).items() if g in GROUPS}
    parts = {"cross": [g for g in G if g[0] != "stessa_topologia"], "same": [g for g in G if g[0] == "stessa_topologia"]}
    sel = {"cross": mask & ~same, "same": mask & same}
    S, Dl, info = [], [], {"rows": {f"{g[0]}|{g[1]}": int(len(r)) for g, r in G.items()}, "partial": {}}
    n = len(R["subjects"])
    for part, gk in parts.items():
        if not sel[part].any():
            continue
        valid = {c: ~np.isnan(v[sel[part]]) for c, v in X.items()}
        full = [c for c in X if valid[c].all()]
        groups = {g: G[g] for g in gk}
        V = fm.replicas({c: X[c] for c in full}, gt, R["sa"], R["sb"], counts, groups, full, workers)
        s_, d_ = fm.records(name, V, gk, groups, full, meta, seed, n, tag=tag)
        S += s_
        Dl += d_
        refs = [r for r in sorted({r for g in ("fr", "sr") for r in fm.REF[g]}) if r in full]
        for c in X:
            if c in full or not valid[c].any():
                continue
            m2 = sel[part] & ~np.isnan(X[c])
            g2 = {g: r for g, r in fm.groups_of(R["ta"], R["tb"], m2).items() if g in gk}
            V2 = fm.replicas({m: X[m] for m in refs + [c]}, gt, R["sa"], R["sb"], counts, g2, refs + [c], workers)
            s2, d2 = fm.records(name, V2, list(g2), g2, refs + [c], meta, seed, n, tag=f"righe proprie di {c}, {tag}")
            S += [r for r in s2 if r["colonna"] == c]
            Dl += [r for r in d2 if r["tipo"] == "vs_nostro"]
            info["partial"][f"{part}|{c}"] = int(m2.sum())
    return S, Dl, info


def step_eval(spec: Path, workers: int, n_boot: int) -> None:
    import pandas as pd
    import conf_eval as ce
    t0 = time.time()
    s, name = setup(spec)                                         # verify (freeze) PRIMA di leggere i dati
    if s.get("pairs"):
        raise SystemExit("conf_desc eval: spec col sottocampione di coppie (solo i collaudi dei protocolli)")
    import fm_eval as fm
    fp, be, blmm = fm.fp, fm.be, fm.blmm
    R = same_rows(ce.build_rows(s, be.zes), be.zes)
    gt = ce.gt_columns(s, R, be.zsum)
    X, failed, info_m = ce.model_columns(s, R, fp)
    # i descrittivi PRIMA di geo_and_protocols: con ``columns.blmm_root`` (collaudo) quella ridirige la cartella di blmm
    # della vista sulle matrici di sviluppo; fit B e taglia stimata leggono params e scalari della vista dello spec
    Xd, fd, pmeta, info_d = desc_columns(s, name, R, fm)
    Xg, fg, info_g = ce.geo_and_protocols(s, R, fm, blmm)
    X.update(Xg)
    failed.update(fg)
    info_same = fill_icp_same(s, R, X)
    same = R["ta"] == R["tb"]
    for c in ("now_original", "now_metrical", "realy", "stirling"):          # protocolli: solo topologie diverse
        if c in X:
            X[c][same] = np.nan
    X.update(Xd)
    failed.update(fd)
    fb = ce.fallback_of(s)
    n = len(R["subjects"])
    rng = np.random.default_rng(int(s["boot_seed"]))
    counts = [np.ones(n, dtype=np.int64)] + [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)]
    mesh_cols, pair_cols = ce.family()
    mesh_cols = [c for c in mesh_cols if c in X]
    pair_cols = [c for c in pair_cols if c in X]
    defined = {c: ~same | np.isfinite(X[c][same]).any() for c in pair_cols}     # NICP e protocolli: non su stessa
    gok = np.isfinite(gt["fr"]) & np.isfinite(gt["sr"])
    mask_c = gok & np.all([np.isfinite(X[c]) for c in mesh_cols], axis=0)
    mask_m = mask_c & np.all([np.isfinite(X[c]) | ~defined[c] for c in pair_cols], axis=0)
    Xc = {c: (np.where(np.isfinite(v), v, np.where(defined[c], np.inf, np.nan)) if c in pair_cols else v)
          for c, v in X.items()}
    meta = meta_of(fm, X, pmeta)
    seed = int(s["boot_seed"])
    S1, D1, i1 = reading(name, Xc, gt, R, mask_c, counts, seed, meta, CONF, workers, fm)
    S2, D2, i2 = reading(name, X, gt, R, mask_m, counts, seed, meta, MASKED, workers, fm)
    allg = fm.groups_of(R["ta"], R["tb"], np.ones(len(same), bool))
    for r in S1 + S2:
        g = (r["insieme_righe"], r["perturbazione"])
        r["non_risolto"] = unresolved(r["colonna"], g, allg[g], X, failed, fb, R) if r["colonna"] in X else ""
    out = ev_dir(s)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(S1 + S2).to_csv(out / "spearman.csv", index=False)
    pd.DataFrame(D1 + D2).to_csv(out / "deltas.csv", index=False)
    ctrl = {"spec": s["_path"], "boot_seed": seed, "n_boot": n_boot, "subjects": n, "rows": int(len(same)),
            "rows_cross": int((~same).sum()), "rows_same": int(same.sum()),
            "rows_mask_confirmatory": int(mask_c.sum()), "rows_mask_masked": int(mask_m.sum()),
            "reading_confirmatory": i1, "reading_masked": i2, "columns": sorted(X),
            "rows_nonfinite_by_column": {c: int((~np.isfinite(v)).sum()) for c, v in X.items()},
            "meshes_failed_by_column": {c: int(v.sum()) for c, v in failed.items()}, "fallback": fb,
            "icp_same": info_same, "desc": info_d, "models": info_m, "geo_protocols": info_g,
            "seconds": time.time() - t0, "modules_sha256": cc.loaded_modules(), **cc.stamp()}
    cc.atomic_json(out / "controls.json", ctrl)
    write_results(out, pd.DataFrame(S1 + S2), ctrl)
    print(f"[conf-desc] lettura {name}: {len(S1) + len(S2)} Spearman, {len(D1) + len(D2)} delta, "
          f"{time.time() - t0:.0f}s", flush=True)


def write_results(out: Path, S, ctrl: dict) -> None:
    f3 = lambda x, lo, hi: f"{x:.3f} [{lo:.3f}, {hi:.3f}]"      # noqa: E731
    L = ["# Prova confermativa: descrittivi (sez. 9) e gruppi descrittivi", "",
         f"Generato da `aau/ava256/conf_desc.py eval` (spec `{ctrl['spec']}`). Seme delle repliche {ctrl['boot_seed']}, "
         f"{ctrl['n_boot']} repliche, {ctrl['subjects']} soggetti; righe {ctrl['rows']} ({ctrl['rows_cross']} a topologie "
         f"diverse, {ctrl['rows_same']} di stessa topologia), maschera confermativa {ctrl['rows_mask_confirmatory']}, "
         f"mascherata {ctrl['rows_mask_masked']}. Nessuna lettura confermativa: tutto descrittivo. Sorgenti assenti: "
         f"{ctrl['desc']['absent'] or 'nessuna'}.", "",
         "Spearman, lettura confermativa (+inf), gruppi principali:", "",
         "| colonna | GT | nocrop | crop | all_cross | non risolto |", "|---|---|---|---|---|---|"]
    x = S[S["passo"].str.endswith(CONF)]
    for (c, g), y in x.groupby(["colonna", "gt"], sort=True):
        cell = []
        for ins in ("nocrop", "crop", "all_cross"):
            z = y[y["insieme_righe"] == ins]
            cell.append(f3(z["rho"].iloc[0], z["ic_lo"].iloc[0], z["ic_hi"].iloc[0]) if len(z) else "-")
        nr = "; ".join(sorted({v for v in y["non_risolto"] if v}))
        L.append(f"| {c} | {g} | {' | '.join(cell)} | {nr or '-'} |")
    L += ["", "Gruppi per perturbazione e stessa topologia, delta `vs_nostro` e la lettura mascherata: `spearman.csv`, "
          "`deltas.csv`.", ""]
    (out / "results.md").write_text("\n".join(L) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("spectral", "varifold", "icp-same", "arcface", "gw-prep", "gw-run", "gw-merge", "perc",
                                    "ulr-run", "ulr-merge", "eval"))
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--stage", type=Path, default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.step == "spectral":
        step_spectral(a.spec, a.workers)
    elif a.step == "varifold":
        step_varifold(a.spec, a.shard, a.device, a.workers)
    elif a.step == "icp-same":
        step_icp_same(a.spec, a.workers)
    elif a.step == "arcface":
        step_arcface(a.spec, a.stage, a.workers)
    elif a.step.startswith("gw-"):
        step_gw(a.spec, a.step[3:], a.shard, a.workers)
    elif a.step == "perc":
        step_perc(a.spec, a.workers, a.device)
    elif a.step.startswith("ulr-"):
        step_ulr(a.spec, a.step[4:], a.shard, a.workers, a.stage)
    else:
        step_eval(a.spec, a.workers, a.n_boot)


if __name__ == "__main__":
    main()
