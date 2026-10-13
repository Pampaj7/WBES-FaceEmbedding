#!/usr/bin/env python3
"""Sensibilita' dei protocolli della comunita' (NoW originale, NoW metrical, REALY, Stirling) ai landmark STIMATI
invece che esatti: rumore sintetico per mesh e un rilevatore reale su un render frontale. Analisi di sensibilita',
NON un metodo dichiarato (plug-in con ``declared: false``).

    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_noise.py prep --view hifi3d --cond s2 --workers 64
    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_noise.py render --view hifi3d --workers 64
    aau/recon/run_recon.sh mica aau/competitors/protocolli/pr_noise_detect.py --view hifi3d     (venv con insightface)
    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_noise.py run --view hifi3d --cond s2 --workers 64
    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_noise.py merge --view hifi3d --cond s2
    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_noise.py check --view hifi3d --workers 32
    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_noise.py eval --workers 64        (LEGGE LE GT di test)
    (pr_noise.sbatch)

Viste: HIFI3D e FaceScape dev (``VIEWS``), crop compreso. **Coppie**: sottocampione uniforme senza reinserimento di
``FRAC`` delle coppie non ordinate di ``pr_common.pairs`` (148.500 per vista -> 11.880), seme ``PAIR_SEED`` e nome della
vista; lo STESSO per tutte le condizioni e tutti i metodi, e i landmark esatti (``competitors_new/<metodo>/<vista>.npz``,
gia' calcolati su tutte le coppie) si leggono sulle stesse righe. Distanza = media dei due versi, come ``pr_run``.

**Condizioni** (``--cond``): ``exact`` = i landmark di ``pr_landmarks prep`` (solo per il controllo di
riproduzione, ``check``); ``s1``, ``s2``, ``s3``, ``s5`` = rumore gaussiano isotropo con sigma 1, 2, 3, 5 mm PER ASSE,
indipendente per mesh (non per coppia): per ogni mesh si estrae una volta Z ~ N(0, I) (85 x 3, seme ``NOISE_SEED`` e
percorso della mesh: stessa estrazione per tutte le sigma, scalata), si sposta ogni landmark di sigma Z e lo si
riproietta sul punto piu' vicino della superficie della mesh. I 68 iBUG e i primi 68 degli 85 di REALY (stessi indici
iBUG, punti diversi) ricevono lo STESSO vettore (un rilevatore, un errore per landmark); i 17 di REALY oltre il 68 il
proprio. ``det`` = rilevatore reale: render frontale (``render``), 68 landmark di insightface ``1k3d68`` (buffalo_l, il
pacchetto di cui MICA usa il rilevatore; ``pr_noise_detect.py``), ogni punto 2D retroproiettato con il raggio della
camera sul primo punto colpito della mesh (raggio che non colpisce: il pixel colpito piu' vicino). I 17 keypoint di
REALY oltre il 68 non hanno un rilevatore: restano ESATTI nella condizione ``det`` (dichiarato; REALY usa 70, 71, 76,
77). Dai landmark nuovi si rifanno i ritagli di NoW e Stirling (``pr_run.load``) e il riferimento di REALY
(``pr_landmarks.realy_reference``: frame, disco, keypoint); le etichette di regione per vertice restano quelle della
prep (dalla corrispondenza, non da landmark: dichiarato).

**Render** (``render``): mesh in mm nel frame canonico (posa gia' frontale, naso verso +z), camera prospettica sull'asse
+z centrata sul riquadro xy della mesh, a ``CAM_DIST`` volte l'estensione, focale tale che l'estensione occupi ``FILL``
dell'immagine (``IMG`` x ``IMG``); raycasting di open3d (Embree), normali per vertice interpolate, Lambert con luce
``LIGHT`` + ambiente ``AMB``, colore uniforme ``ALBEDO``, sfondo nero. Nessuna GT e nessun landmark esatto usati.
Riquadro di ripiego per il rilevatore (se il detector non trova un volto): i pixel colpiti.

``run``: coppie ORDINATE del sottocampione per mesh A (``pr_run.one_pair``, il port invariato, ``--now-opt`` 0 come la
matrice); ogni A si salva (``parts/<vista>/<cond>.d/A<k>.npz``), una ripartenza salta le A salvate. ``merge``: plug-in
``competitors_new/<metodo>/lmk_<cond>/<vista>.npz`` (``D_mm`` simmetrica, NaN fuori dal sottocampione), json,
``fm_plugin.json`` con ``declared: false``, ``pairs_<vista>.csv``. ``check``: ``CHECK_N`` coppie del sottocampione con i
landmark esatti, i due versi, contro ``D_mm`` dei plug-in esistenti (riproduzione della pipeline).

``eval``: righe di ``bp_paired_e2.all_cross_rows`` ristrette al sottocampione (maschera comune = righe con tutte le
colonne finite), insiemi ``nocrop``, ``crop``, ``all_cross`` (``pr_eval.groups``), 1000 repliche per soggetto col seme
della T1 (``pr_eval.t1_seed``, come ``fm_eval``), Spearman pesato c_a c_b (``bp_paired_e3.replicas``). Colonne: ogni
metodo in ogni condizione (``exact`` dai plug-in esistenti), ICP + Chamfer in mm, il nostro factorized s1234 (d_F non
calibrata con FR, d_P con SR). Delta appaiati (IC 95% percentile, ``p_le0``): protocollo - nostro, protocollo - ICP mm,
protocollo con landmark stimati - lo stesso con landmark esatti. Uscite in ``aau/runs/evidence/protocolli/rumore_landmark/``:
``spearman.csv``, ``deltas.csv``, ``controls.json``, ``prep/``, ``checks/``, ``results.md`` (generato; il testo dopo
``MARK`` si conserva).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as pc  # noqa: E402
import pr_metrics as pm  # noqa: E402
import pr_run as prr  # noqa: E402

blmm = pc.blmm
VIEWS = ("hifi3d", "facescape")
FRAC, PAIR_SEED, NOISE_SEED = 0.08, 20261013, 20261013
SIGMAS = {"s1": 1.0, "s2": 2.0, "s3": 3.0, "s5": 5.0}
CONDS = tuple(SIGMAS) + ("det",)
CLABEL = {"exact": "landmark esatti", "s1": "sigma 1 mm", "s2": "sigma 2 mm", "s3": "sigma 3 mm", "s5": "sigma 5 mm",
          "det": "rilevatore (insightface 1k3d68 su render)"}
IMG, FILL, CAM_DIST = 640, 0.45, 4.0     # FILL: a 0.75 il detector non trova volti (punteggio 0.07-0.31 su 8
                                         # render), a 0.45 ~0.88; scelto sul solo punteggio, senza landmark ne' GT
LIGHT, AMB, ALBEDO = (0.0, 0.4, 1.0), 0.25, (0.88, 0.71, 0.61)
CHECK_N = 24
NWORK = pc.WORK / "noise"                            # landmark, render, parti (fuori da git)
NEVID = pc.EVID / "rumore_landmark"
MARK = "<!-- da qui in giu': scritto a mano, conservato da pr_noise.py results -->"
_ST: dict = {}


# ------------------------------------------------------------------------------------------ coppie

def subsample(vname: str, items: list) -> tuple[np.ndarray, np.ndarray]:
    """Coppie non ordinate (i < j) del sottocampione: ``FRAC`` di ``pr_common.pairs``, seme fisso per vista."""
    i, j = pc.pairs(items)
    rng = np.random.default_rng([PAIR_SEED, zlib.crc32(vname.encode())])
    k = np.sort(rng.choice(len(i), int(round(FRAC * len(i))), replace=False))
    return i[k], j[k]


def partners_of(i: np.ndarray, j: np.ndarray, n: int) -> list:
    """Per ogni mesh A le B del sottocampione (i due versi di ogni coppia)."""
    out = [[] for _ in range(n)]
    for a, b in zip(i, j):
        out[a].append(b)
        out[b].append(a)
    return [np.asarray(sorted(x), np.int64) for x in out]


# ------------------------------------------------------------------------------------------ render

def ray_dirs(cam: dict, px: np.ndarray) -> np.ndarray:
    """Direzioni (n, 3) dei raggi per le coordinate pixel ``px`` (n, 2) (centro del pixel = coordinata intera, come
    cv2 / insightface)."""
    d = np.stack([(px[:, 0] + 0.5 - IMG / 2) / cam["f"], -(px[:, 1] + 0.5 - IMG / 2) / cam["f"],
                  -np.ones(len(px))], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def cast(scene, cam: dict, px: np.ndarray) -> dict:
    import open3d as o3d
    d = ray_dirs(cam, px)
    R = np.hstack([np.broadcast_to(cam["o"], d.shape), d]).astype(np.float32)
    r = scene.cast_rays(o3d.core.Tensor(R))
    t = r["t_hit"].numpy().astype(np.float64)
    return {"d": d, "t": t, "hit": np.isfinite(t), "tri": r["primitive_ids"].numpy().astype(np.int64),
            "uv": r["primitive_uvs"].numpy().astype(np.float64)}


def scene_of(V: np.ndarray, F: np.ndarray):
    import open3d as o3d
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    return s


def camera(V: np.ndarray) -> dict:
    lo, hi = V.min(0), V.max(0)
    c = (lo + hi) / 2
    ext = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    dist = CAM_DIST * ext
    return {"o": np.array([c[0], c[1], hi[2] + dist]), "f": float(FILL * IMG * (dist + hi[2] - c[2]) / ext)}


def render_mesh(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
    """(immagine RGB uint8, camera, riquadro dei pixel colpiti [x0, y0, x1, y1])."""
    cam = camera(V)
    scene = scene_of(V, F)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    r = cast(scene, cam, np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64))
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    h = r["hit"]
    tri, uv = r["tri"][h], r["uv"][h]
    n = (1 - uv[:, :1] - uv[:, 1:]) * vn[F[tri, 0]] + uv[:, :1] * vn[F[tri, 1]] + uv[:, 1:] * vn[F[tri, 2]]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    n *= np.where((n * r["d"][h]).sum(1, keepdims=True) > 0, -1.0, 1.0)            # verso la camera
    L = np.asarray(LIGHT) / np.linalg.norm(LIGHT)
    shade = AMB + (1 - AMB) * np.clip(n @ L, 0.0, 1.0)
    img = np.zeros((IMG * IMG, 3))
    img[h] = shade[:, None] * np.asarray(ALBEDO)[None]
    hm = h.reshape(IMG, IMG)
    ys, xs = np.nonzero(hm)
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8).reshape(IMG, IMG, 3), cam, \
        np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float64)


def _render_task(k: int) -> tuple:
    from PIL import Image
    v, items, out = _ST["view"], _ST["items"], _ST["out"]
    V, F, _ = pc.mm_mesh(v, items[k][2])
    img, cam, box = render_mesh(V, F)
    Image.fromarray(img).save(out / f"{k:04d}.png")
    return k, cam["o"], cam["f"], box


def render(vname: str, workers: int) -> None:
    t0 = time.time()
    v = pc.view(vname)
    out = NWORK / "det" / vname
    out.mkdir(parents=True, exist_ok=True)
    _ST.update(view=v, items=v["items"], out=out)
    with mp.get_context("fork").Pool(workers) as pool:
        res = sorted(pool.map(_render_task, range(len(v["items"])), chunksize=4), key=lambda x: x[0])
    blmm.atomic_savez(out / "cams.npz", o=np.stack([r[1] for r in res]), f=np.asarray([r[2] for r in res]),
                      box=np.stack([r[3] for r in res]), paths=np.asarray([blmm.rel(p) for _, _, p in v["items"]]),
                      img=IMG)
    print(f"[pr-noise] render {vname}: {len(res)} immagini in {out}, {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------- prep

def _prep_task(k: int) -> dict:
    v, pr, cond = _ST["view"], _ST["prep"], _ST["cond"]
    import pr_landmarks as pl
    p = v["items"][k][2]
    V, F, _ = pc.mm_mesh(v, p)
    surf = pm.Surf(V, F)
    L68, K85 = pr["L68"][k], pr["K85"][k]
    info = {}
    if cond in SIGMAS:
        Z = np.random.default_rng([NOISE_SEED, zlib.crc32(blmm.rel(p).encode())]).standard_normal((85, 3))
        E = SIGMAS[cond] * Z
        L68n, K85n = surf.closest(L68 + E[:68]), surf.closest(K85 + E)
    else:                                                       # det: retroproiezione dei 68 punti 2D
        dt = _ST["det"]
        cam = {"o": dt["o"][k], "f": float(dt["f"][k])}
        scene = scene_of(V, F)
        r = cast(scene, cam, dt["P"][k][:, :2])
        miss = ~r["hit"]
        L68n = cam["o"] + r["d"] * np.where(miss, 0.0, r["t"])[:, None]
        if miss.any():                                          # pixel colpito piu' vicino al punto 2D
            yy, xx = np.mgrid[0:IMG, 0:IMG]
            full = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
            rf = cast(scene, cam, full)
            hp = full[rf["hit"]]
            j = np.linalg.norm(hp[None] - dt["P"][k][miss, None, :2], axis=2).argmin(1)
            jj = np.flatnonzero(rf["hit"])[j]
            L68n[miss] = cam["o"] + rf["d"][jj] * rf["t"][jj][:, None]
        K85n = K85.copy()
        K85n[:68] = L68n                                        # i 17 di REALY oltre il 68 restano esatti
        info = {"miss": miss, "fallback": bool(dt["fallback"][k]), "det_score": float(dt["det_score"][k])}
    ref = pl.realy_reference(V, F, pr["labels"][k], K85n, pl.REALY_CAP)
    return {"k": k, "L68": L68n, "K85": K85n, "realy": ref, "d68": np.linalg.norm(L68n - L68, axis=1),
            "d85": np.linalg.norm(K85n - K85, axis=1), **info}


def prep_path(vname: str, cond: str) -> Path:
    return NWORK / "prep" / f"{vname}_{cond}.npz"


def prep(vname: str, cond: str, workers: int) -> None:
    """Landmark stimati per ogni mesh della vista e riferimenti di REALY rifatti (formato di ``pr_common.read_prep``)."""
    t0 = time.time()
    v = pc.view(vname)
    pr = pc.read_prep(vname)
    _ST.update(view=v, prep=pr, cond=cond)
    if cond == "det":
        dd = NWORK / "det" / vname
        with np.load(dd / "cams.npz") as z, np.load(dd / "lmk2d.npz") as y:
            if list(z["paths"].astype(str)) != [blmm.rel(p) for _, _, p in v["items"]]:
                raise SystemExit(f"{vname}: render con mesh diverse dalla vista")
            _ST["det"] = {"o": z["o"], "f": z["f"], "P": y["P"], "fallback": y["fallback"], "det_score": y["det_score"]}
    with mp.get_context("fork").Pool(workers) as pool:
        res = sorted(pool.map(_prep_task, range(len(v["items"])), chunksize=2), key=lambda r: r["k"])
    obj = lambda xs: np.asarray(xs + [None], dtype=object)[:-1]       # noqa: E731
    out = {k: pr[k] for k in ("subjects", "topologies", "paths", "labels", "n_vertices")}
    out.update(L68=np.stack([r["L68"] for r in res]), K85=np.stack([r["K85"] for r in res]),
               realy=obj([r["realy"] for r in res]), d68=np.stack([r["d68"] for r in res]),
               d85=np.stack([r["d85"] for r in res]))
    prep_path(vname, cond).parent.mkdir(parents=True, exist_ok=True)
    np.savez(prep_path(vname, cond), **out)
    topo = pr["topologies"].astype(str)
    d7 = out["d68"][:, list(pc.LMK7)]

    def st(x):
        return {"mean": float(x.mean()), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
                "max": float(x.max())}

    summ = {"view": vname, "cond": cond, "sigma_per_axis_mm": SIGMAS.get(cond), "meshes": len(res),
            "shift_lmk7_mm": st(d7), "shift_lmk7_mm_by_landmark": {n: st(d7[:, i]) for i, n in enumerate(pc.LMK7_NAMES)},
            "shift_lmk7_mm_by_topology": {t: st(d7[topo == t]) for t in sorted(set(topo))},
            "shift_ibug68_mm": st(out["d68"]), "shift_realy85_mm": st(out["d85"]),
            "note": "spostamento dal landmark esatto dopo la riproiezione sulla superficie (det: retroproiezione)",
            "seconds": time.time() - t0, **pc.stamp()}
    if cond == "det":
        fb = np.asarray([r["fallback"] for r in res])
        summ.update(det_fallback={t: int(fb[topo == t].sum()) for t in sorted(set(topo))},
                    det_fallback_total=int(fb.sum()), rays_missed_ibug68=int(sum(r["miss"].sum() for r in res)),
                    rays_missed_lmk7={n: int(sum(r["miss"][i] for r in res)) for i, n in zip(pc.LMK7, pc.LMK7_NAMES)},
                    render={"img": IMG, "fill": FILL, "cam_dist_x_extent": CAM_DIST, "light": LIGHT, "ambient": AMB,
                            "albedo": ALBEDO},
                    realy_extra_17="esatti (nessun rilevatore per i keypoint di REALY oltre il 68)")
    pc.atomic_json(NEVID / "prep" / f"{vname}_{cond}.json", summ)
    print(f"[pr-noise] prep {vname} {cond}: spostamento lmk7 {summ['shift_lmk7_mm']}, {time.time() - t0:.0f}s",
          flush=True)


def load_cond(vname: str, cond: str) -> dict:
    pr = None
    if cond != "exact":
        with np.load(prep_path(vname, cond), allow_pickle=True) as z:
            pr = {k: z[k] for k in z.files}
    return prr.load(vname, None, 0, prep=pr)


# -------------------------------------------------------------------------------------------- run

def _task(a: int) -> int:
    ck = _ST["ckdir"] / f"A{a}.npz"
    if ck.exists():
        return a
    t0 = time.perf_counter()
    D, bs = _ST["D"], _ST["partners"][a]
    V, F = D["mesh"][a]
    surf = pm.Surf(V, F)
    rows = [prr.one_pair(surf, a, int(b), pc.METHODS, D) for b in bs]
    blmm.atomic_savez(ck, task_s=time.perf_counter() - t0, job=pc.stamp()["job"],
                      **prr.pack([{"a": a, "b": bs, "rows": rows}]))
    return a


def ckdir_of(vname: str, cond: str) -> Path:
    return NWORK / "parts" / vname / f"{cond}.d"


def run(vname: str, cond: str, workers: int, reverse: bool) -> None:
    t0 = time.time()
    D = load_cond(vname, cond)
    i, j = subsample(vname, D["items"])
    P = partners_of(i, j, len(D["items"]))
    ck = ckdir_of(vname, cond)
    ck.mkdir(parents=True, exist_ok=True)
    _ST.update(D=D, partners=P, ckdir=ck)
    todo = [a for a in range(len(P)) if len(P[a]) and not (ck / f"A{a}.npz").exists()]
    if reverse:
        todo = todo[::-1]
    print(f"[pr-noise] run {vname} {cond}: {len(i)} coppie, {len(todo)} mesh A da fare, {time.time() - t0:.0f}s di "
          "caricamento", flush=True)
    tl = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_task, todo, chunksize=1)):
            if (k + 1) % max(1, len(todo) // 10) == 0:
                print(f"[pr-noise] run {vname} {cond}: {k + 1}/{len(todo)} mesh A, {time.time() - tl:.0f}s", flush=True)
    print(f"[pr-noise] run {vname} {cond}: fatto in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------ merge

def plugin_dir(method: str, cond: str) -> Path:
    return pc.PLUGIN_ROOT / method / f"lmk_{cond}"


def merge(vname: str, cond: str) -> None:
    v = pc.view(vname)
    items, n = v["items"], len(v["items"])
    i, j = subsample(vname, items)
    P = partners_of(i, j, n)
    ck = ckdir_of(vname, cond)
    need = [a for a in range(n) if len(P[a])]
    miss = [a for a in need if not (ck / f"A{a}.npz").exists()]
    if miss:
        raise SystemExit(f"{vname} {cond}: {len(miss)} mesh A mancanti (es. {miss[:5]})")
    E = {m: np.full((n, n), np.nan) for m in pc.METHODS}
    errs = {m: [] for m in pc.METHODS}
    secs = {m: [] for m in pc.METHODS}
    core_s, jobs = 0.0, set()
    for a in need:
        with np.load(ck / f"A{a}.npz") as z:
            if not np.array_equal(np.sort(z["b"]), P[a]):
                raise SystemExit(f"{vname} {cond} A{a}: partner diversi dal sottocampione")
            for m in pc.METHODS:
                E[m][z["a"], z["b"]] = z[f"val:{m}"]
                secs[m].append(z[f"val:{m}|seconds"])
                e = z[f"err:{m}"]
                errs[m] += [str(x) for x in e[e != ""]]
            core_s += float(z["task_s"])
            jobs.add(str(z["job"]))
    for m in pc.METHODS:
        Dm = np.full((n, n), np.nan)
        Dm[i, j] = Dm[j, i] = (E[m][i, j] + E[m][j, i]) / 2
        np.fill_diagonal(Dm, 0.0)
        s = np.concatenate(secs[m])
        info = {"method": m, "cond": cond, "label": f"{pc.LABEL[m]}, {CLABEL[cond]}", "view": vname,
                "pairs_unordered": int(len(i)), "subsample": {"frac": FRAC, "seed": PAIR_SEED},
                "sigma_per_axis_mm": SIGMAS.get(cond), "noise_seed": NOISE_SEED if cond in SIGMAS else None,
                "ordered_missing": int(np.isnan(E[m][i, j]).sum() + np.isnan(E[m][j, i]).sum()),
                "errors": len(errs[m]), "error_examples": sorted(set(errs[m]))[:10],
                "seconds_per_ordered_pair": {"mean": float(s.mean()), "median": float(np.median(s))},
                "core_hours_all_methods": core_s / 3600, "jobs": sorted(jobs), **pc.stamp()}
        d = plugin_dir(m, cond)
        subj = np.asarray([x for x, _, _ in items])
        topo = np.asarray([t for _, t, _ in items])
        blmm.atomic_savez(d / f"{vname}.npz", subjects=subj, topologies=topo,
                          paths=np.asarray([blmm.rel(p) for _, _, p in items]), D_mm=Dm)
        pc.atomic_json(d / f"{vname}.json", info)
        lab = f"{pc.LABEL[m]}, {CLABEL[cond]} (non dichiarata, sensibilita' ai landmark)"
        pc.atomic_json(d / "fm_plugin.json", {
            "label": lab, "family": f"{pc.FAMILY} (sensibilita' ai landmark)", "declared": False, "method": m,
            "variant": f"lmk_{cond}", "readings": {"mm": {"label": f"{lab}, mm (media dei due versi)", "gt": ["fr", "sr"]}}})
        with open(d / f"pairs_{vname}.csv", "w") as fh:
            fh.write("subject_a,topology_a,subject_b,topology_b,mm\n")
            for a, b in zip(i, j):
                fh.write(f"{subj[a]},{topo[a]},{subj[b]},{topo[b]},{Dm[a, b]:.9g}\n")
        print(f"[pr-noise] merge {vname} {cond} {m}: {len(i)} coppie, versi mancanti {info['ordered_missing']}, "
              f"errori {len(errs[m])}", flush=True)
    print(f"[pr-noise] merge {vname} {cond}: {core_s / 3600:.1f} ore-core di calcolo, job {sorted(jobs)}", flush=True)


# ------------------------------------------------------------------------------------------ check

def _check_task(ab: tuple) -> dict:
    a, b = ab
    D = _ST["D"]
    out = {}
    for x, y in ((a, b), (b, a)):
        V, F = D["mesh"][x]
        r = prr.one_pair(pm.Surf(V, F), x, y, pc.METHODS, D)
        for m in pc.METHODS:
            out.setdefault(m, []).append(r[m])
            out.setdefault(f"{m}|seconds", []).append(r[f"{m}|seconds"])
    return {"a": a, "b": b, **{k: float(np.mean(x)) if "|" not in k else float(np.sum(x)) for k, x in out.items()}}


def check(vname: str, workers: int) -> None:
    """Riproduzione: ``CHECK_N`` coppie del sottocampione con i landmark esatti (questa pipeline) contro ``D_mm`` dei
    plug-in esistenti."""
    D = load_cond(vname, "exact")
    i, j = subsample(vname, D["items"])
    sel = np.random.default_rng(1234).choice(len(i), CHECK_N, replace=False)
    _ST["D"] = D
    with mp.get_context("fork").Pool(workers) as pool:
        rows = pool.map(_check_task, [(int(i[k]), int(j[k])) for k in sel], chunksize=1)
    out = {"view": vname, "n_pairs": len(rows), "methods": {}}
    for m in pc.METHODS:
        with np.load(pc.PLUGIN_ROOT / m / f"{vname}.npz") as z:
            Dm = np.asarray(z["D_mm"], np.float64)
        x = np.asarray([r[m] for r in rows])
        y = np.asarray([Dm[r["a"], r["b"]] for r in rows])
        out["methods"][m] = {"max_abs_diff_mm": float(np.max(np.abs(x - y))), "median_value_mm": float(np.median(y)),
                             "seconds_per_unordered_pair": float(np.median([r[f"{m}|seconds"] for r in rows]))}
    out.update(pc.stamp())
    pc.atomic_json(NEVID / "checks" / f"exact_{vname}.json", out)
    print(f"[pr-noise] check {vname}: {out['methods']}", flush=True)


# ------------------------------------------------------------------------------------------- eval

def col(m: str, c: str) -> str:
    return f"{m}@{c}"


def run_view(vname: str, n_boot: int, workers: int) -> tuple[list, list, dict]:
    import bp_paired_e2 as bpe2
    import bp_paired_e3 as bpe3
    import pr_eval as pe
    df, idx, _ = bpe2.all_cross_rows(vname)
    seed = pe.t1_seed(vname)
    keys = list(idx.keys)
    conds = ["exact"] + [c for c in CONDS if (plugin_dir(pc.METHODS[0], c) / f"{vname}.npz").exists()]
    D = {}
    for m in pc.METHODS:
        for c in conds:
            D[col(m, c)] = pe.plugin_matrix(m if c == "exact" else f"{m}/lmk_{c}", vname, keys)
    ref = pe.ref_columns(vname, idx)
    OURS = {"fr": pe.OURS["fr"][0], "sr": pe.OURS["sr"][0]}
    D.update({pe.ICP: ref[pe.ICP], OURS["fr"]: ref[OURS["fr"]], OURS["sr"]: ref[OURS["sr"]]})
    # sottocampione esplicito (oltre alla maschera comune)
    i, j = subsample(vname, pc.view(vname)["items"])
    S = np.zeros((len(keys), len(keys)), bool)
    S[i, j] = S[j, i] = True
    D["_sub"] = S.astype(np.float64)
    df = bpe3.be.add_columns(df, D, idx)
    cols = [c for c in D if c != "_sub"]
    X = {m: df[m].to_numpy(np.float64) for m in cols + ["oracle_size"]}
    X.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in ("fr", "sr")})
    sub = df["_sub"].to_numpy() > 0
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
    s2i = {s: k for k, s in enumerate(subjects)}
    sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
    rng = np.random.default_rng(seed)
    counts = [np.ones(len(subjects), dtype=np.int64)] + \
        [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(n_boot)]
    mask = sub & (sa != sb) & np.all([np.isfinite(v) for v in X.values()], axis=0)
    ta, tb = df["topology_a"].to_numpy(str), df["topology_b"].to_numpy(str)
    Sr, Dl = [], []
    for g, gm in pe.groups(ta, tb, mask).items():
        if g[0] not in ("nocrop", "crop", "all_cross"):
            continue
        V = bpe3.replicas(X, sa, sb, gm, cols, counts, workers)
        base = {"dominio": vname, "insieme_righe": g[0], "n_righe": int(gm.sum()), "seme": seed}

        def ci(x):
            b = x[1:][np.isfinite(x[1:])]
            return (tuple(np.percentile(b, [2.5, 97.5])) if len(b) else (np.nan, np.nan)), b

        for (gt, kind, c), a in V.items():
            if kind != "rho":
                continue
            (lo, hi), b = ci(a)
            m, cond = c.split("@") if "@" in c else (c, "-")
            Sr.append({**base, "gt": gt, "colonna": c, "metodo": m, "cond": cond, "rho": a[0], "ic_lo": lo,
                       "ic_hi": hi, "n_boot": len(b)})
        for gt in ("fr", "sr"):
            for m in pc.METHODS:
                for c in conds:
                    comp = [(OURS[gt], "protocollo - nostro"), (pe.ICP, "protocollo - ICP mm")]
                    if c != "exact":
                        comp.append((col(m, "exact"), "stimati - esatti"))
                    for b_, tipo in comp:
                        d = V[(gt, "rho", col(m, c))] - V[(gt, "rho", b_)]
                        (lo, hi), b = ci(d)
                        Dl.append({**base, "gt": gt, "tipo": tipo, "metodo": m, "cond": c, "colonna_b": b_,
                                   "rho_a": V[(gt, "rho", col(m, c))][0], "rho_b": V[(gt, "rho", b_)][0],
                                   "delta": d[0], "ic_lo": lo, "ic_hi": hi,
                                   "p_le0": float((b <= 0).mean()) if len(b) else np.nan})
    info = {"seed": seed, "rows": int(len(df)), "rows_subsample": int(sub.sum()), "rows_mask": int(mask.sum()),
            "pairs_subsample": int(len(i)), "conds": conds,
            "rows_nan_in_subsample_by_column": {m: int((~np.isfinite(X[m]) & sub).sum()) for m in cols}}
    print(f"[pr-noise] eval {vname}: {int(mask.sum())}/{int(sub.sum())} righe del sottocampione, condizioni {conds}",
          flush=True)
    return Sr, Dl, info


def evaluate(n_boot: int, workers: int) -> None:
    import pandas as pd
    sys.path.insert(0, str(pc.AAU / "famos"))
    t0 = time.time()
    S, Dl, info = [], [], {}
    for v in VIEWS:
        s, d, inf = run_view(v, n_boot, workers)
        S += s
        Dl += d
        info[v] = inf
    NEVID.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(S).to_csv(NEVID / "spearman.csv", index=False)
    pd.DataFrame(Dl).to_csv(NEVID / "deltas.csv", index=False)
    pc.atomic_json(NEVID / "controls.json", {**info, "n_boot": n_boot, "wall_s": time.time() - t0,
                                             "subsample": {"frac": FRAC, "seed": PAIR_SEED}, **pc.stamp()})
    results()


# ---------------------------------------------------------------------------------------- results

def f3(r, lo, hi, sign=False) -> str:
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"{f.format(r)} [{f.format(lo)}, {f.format(hi)}]"


def results() -> None:
    import pandas as pd
    import pr_eval as pe
    S = pd.read_csv(NEVID / "spearman.csv")
    Dl = pd.read_csv(NEVID / "deltas.csv")
    ctrl = json.loads((NEVID / "controls.json").read_text())
    conds = [c for c in ("exact",) + CONDS if c in set(S["cond"])]
    ours = {"fr": pe.OURS["fr"][0], "sr": pe.OURS["sr"][0]}
    L = ["# Protocolli della comunita' con landmark stimati invece che esatti", "",
         "Generato da `aau/competitors/protocolli/pr_noise.py` (definizioni nel docstring). Sottocampione uniforme "
         f"{FRAC:.0%} delle coppie di all_cross (seme {PAIR_SEED}), uguale per tutte le colonne; 1000 repliche per "
         "soggetto col seme della T1, IC 95% percentile. Distanza = media dei due versi. sigma = deviazione standard "
         "PER ASSE del rumore 3D prima della riproiezione sulla superficie, indipendente per mesh.", ""]
    L += ["## Errore effettivo dei landmark (7 di NoW, mm dal punto esatto)", "",
          "| vista | condizione | media | mediana | p95 |", "|---|---|---|---|---|"]
    for v in VIEWS:
        for c in CONDS:
            p = NEVID / "prep" / f"{v}_{c}.json"
            if p.exists():
                x = json.loads(p.read_text())["shift_lmk7_mm"]
                L.append(f"| {v} | {CLABEL[c]} | {x['mean']:.2f} | {x['median']:.2f} | {x['p95']:.2f} |")
    L.append("")
    for gt in ("fr", "sr"):
        L += [f"## Spearman con GT {gt.upper()}", "", "| vista | righe | metodo | " +
              " | ".join(CLABEL[c] for c in conds) + " |", "|---|---|---|" + "---|" * len(conds)]
        for v in VIEWS:
            for s in ("nocrop", "crop"):
                X = S[(S["gt"] == gt) & (S["dominio"] == v) & (S["insieme_righe"] == s)]
                if X.empty:
                    continue
                for m in pc.METHODS:
                    cells = []
                    for c in conds:
                        r = X[X["colonna"] == col(m, c)]
                        cells.append(f3(r.iloc[0].rho, r.iloc[0].ic_lo, r.iloc[0].ic_hi) if len(r) else "-")
                    L.append(f"| {v} | {s} ({int(X.iloc[0].n_righe)}) | {m} | " + " | ".join(cells) + " |")
                for c_ in (pe.ICP, ours[gt]):
                    r = X[X["colonna"] == c_]
                    if len(r):
                        L.append(f"| {v} | {s} | {pe.CLABEL[c_]} (non dipende dai landmark) | " +
                                 f3(r.iloc[0].rho, r.iloc[0].ic_lo, r.iloc[0].ic_hi) + " |" + " |" * (len(conds) - 1))
        L.append("")
    for tipo, titolo in (("stimati - esatti", "stesso protocollo, landmark stimati - esatti"),
                         ("protocollo - nostro", "protocollo - nostro (FR: d_F non cal. s1234; SR: d_P s1234; "
                                                 "negativo = il nostro ordina meglio)"),
                         ("protocollo - ICP mm", "protocollo - ICP + Chamfer in mm")):
        L += [f"## Delta appaiati: {titolo}", "", "| GT | vista | righe | metodo | " +
              " | ".join(CLABEL[c] for c in conds) + " |", "|---|---|---|---|" + "---|" * len(conds)]
        for gt in ("fr", "sr"):
            for v in VIEWS:
                for s in ("nocrop", "crop"):
                    X = Dl[(Dl["gt"] == gt) & (Dl["dominio"] == v) & (Dl["insieme_righe"] == s) & (Dl["tipo"] == tipo)]
                    for m in pc.METHODS:
                        cells = []
                        for c in conds:
                            r = X[(X["metodo"] == m) & (X["cond"] == c)]
                            cells.append(f3(r.iloc[0].delta, r.iloc[0].ic_lo, r.iloc[0].ic_hi, True) if len(r) else "-")
                        if any(x != "-" for x in cells):
                            L.append(f"| {gt.upper()} | {v} | {s} | {m} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["Righe: " + "; ".join(f"{v} {ctrl[v]['rows_mask']} / {ctrl[v]['rows_subsample']} del sottocampione "
                                f"({ctrl[v]['pairs_subsample']} coppie)" for v in VIEWS if v in ctrl), ""]
    p = NEVID / "results.md"
    tail = ""
    if p.exists() and MARK in p.read_text():
        tail = p.read_text().split(MARK, 1)[1]
    p.write_text("\n".join(L) + "\n" + MARK + tail)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("prep", "render", "run", "merge", "check", "eval", "results"))
    p.add_argument("--view", default="")
    p.add_argument("--cond", default="", choices=("",) + CONDS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--reverse", action="store_true", help="run: dalle ultime mesh A (job di riserva)")
    a = p.parse_args()
    if a.step == "prep":
        prep(a.view, a.cond, a.workers)
    elif a.step == "render":
        render(a.view, a.workers)
    elif a.step == "run":
        run(a.view, a.cond, a.workers, a.reverse)
    elif a.step == "merge":
        merge(a.view, a.cond)
    elif a.step == "check":
        check(a.view, a.workers)
    elif a.step == "eval":
        evaluate(a.n_boot, a.workers)
    else:
        results()


if __name__ == "__main__":
    main()
