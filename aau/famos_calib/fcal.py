#!/usr/bin/env python3
"""FaMoS, calibrazione LINEARE post hoc per l'invarianza all'espressione del nostro (WCCN, LDA, unione con la taglia)
sul riconoscimento dell'identita' di ``esterna_famos_id``. Regole in ``aau/runs/evidence/famos_calib/REGOLE.md``
(committate prima dei numeri). POST HOC e DESCRITTIVO: FaMoS TEST e' gia' stato visto.

    v3_work/unified_gt/run.sh aau/famos_calib/fcal.py select --workers 16
    v3_work/unified_gt/run.sh aau/famos_calib/fcal.py render-raw --workers 16
    aau/recon/run_recon.sh mica aau/competitors/protocolli/pr_noise_detect.py --dir <CAL>/raw_det
    v3_work/unified_gt/run.sh aau/famos_calib/fcal.py patch --workers 16
    v3_work/unified_gt/run.sh aau/famos_calib/fcal.py render-patch --workers 16
    aau/recon/run_recon.sh mica aau/competitors/protocolli/pr_noise_detect.py --dir <CAL>/patch_det
    v3_work/unified_gt/run.sh aau/famos_calib/fcal.py lmk --workers 16
    v3_work/unified_gt/run.sh aau/famos_calib/fcal.py folds                       (fold e resoconto della prep)
    aau/run.sh v3_work/trainer/eval_v3.py -- aau/famos_calib/fcal.py embed --model <nome> --checkpoint <pth>  (GPU)
    aau/outlineB/run_o3d.sh   aau/famos_calib/fcal.py grid --fold k --method icp --workers 32
    v3_work/unified_gt/run.sh aau/famos_calib/fcal.py grid --fold k --method now --workers 32
    aau/run.sh aau/famos_calib/fcal.py eval
    (fcal.sbatch: un passo per job; ``grid`` come job array, un fold e un metodo per elemento)

**Calibrazione** (``select``): soggetti FaMoS TRAIN (``split.json``), MAI i 15 di TEST. Su disco ci sono solo le
registrazioni sottocampionate (``datasets/FAMOS/train``, 1 fotogramma ogni 10, mesh in topologia FLAME, mm): le
scansioni grezze di TRAIN non esistono piu'. Sequenze = le 10 di ``esterna_famos_id``; pool per (soggetto, sequenza)
con la regola di ``famos_test_view.select`` (galleria = fotogramma con ``expr_mm`` minima del soggetto, peak e
nearneutral per sequenza; ``expr_mm`` contro ``V_neutral``), fotogramma = ``ei.pick_frame`` (stesso seme). Soggetti
senza una delle 10 sequenze: esclusi. **Preparazione** identica a ``ei.py``: render frontale, insightface, retroproiezione
``front_only``, ``famos_test_view.canonical_patch``, mm; landmark NoW rilevati di nuovo sulla patch. Una mesh fallita
toglie il soggetto (``folds``). **Fold**: ``rng(SEED_CAL).permutation`` dei soggetti rimasti, ``N_FOLD`` blocchi.

**Nostro** (``embed``): latenti [s, u] dei checkpoint di ``ei.CKPT`` sulle patch di calibrazione (operatori areanorm k_eig
128 su CPU, tabella di scala in mm come ``ei.sbatch``); controllo: le 150 patch di test rifatte contro i latenti di
``esterna_famos_id`` (che si RIUSANO), tolleranza ``ei.TOL['Z']``. Varianti su u (256): (a) WCCN, u -> (S_w + lam t I)^(-1/2) u
con t = tr(S_w) / 256; (b) LDA, i k autovettori principali di S_b v = mu (S_w + lam t I) v (normalizzati su
S_w + lam t I); (c) d_F ricomposta, sqrt((S_i - S_j)^2 + S_i S_j (g c0 d')^2) con d' di (a) o (b) alla loro scelta e
c0 = mediana di d_P / mediana di d' sulle coppie di persone diverse dei soggetti di stima; controllo: d_F con u non
calibrata e c = g scelto allo stesso modo. Scelta: validazione incrociata sui fold (stima su N_FOLD - 1 fold, AUC sulle
coppie interne al fold tenuto fuori, media sui fold), massimo, a pari merito il primo della griglia; poi stima su tutti
i soggetti di calibrazione e applicazione agli embedding di test.

**Concorrenti** (``grid``): ICP + Chamfer in mm e NoW metrical sulla griglia di ``robustezza_iperparametri`` (``ri.GRID``,
le funzioni di ``ri.run`` / ``ri.merge`` importate) sulle coppie interne a ciascun fold; configurazione = la migliore AUC
media sui fold (pari merito: la prima, cioe' il default). Le distanze di test di ogni configurazione esistono gia'
(``aau/runs/robustezza_iperparametri/dist/famos``).

**Valutazione** (``eval``): le 150 scansioni e le 11.175 coppie di ``esterna_famos_id``, metriche di ``ei.evaluate``
(rank-1, mAP, AUC, EER; IC con ``famos_eval.bootstrap_counts(15, 1000, 1234)``), strati d'espressione di
``famos_dominio`` (terzili di ``expr_mm``, AUC_k e rank-1_k), delta appaiati sulle stesse repliche.

Dati derivati da FaMoS (licenza MPI) solo in ``datasets/FAMOS/calib`` (fuori da git); in git nomi e aggregati.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
REPO = AAU.parent
for _p in (AAU / "esterna_famos_id", AAU / "famos", AAU / "famos_dominio", AAU / "robustezza_iperparametri",
           REPO / "v3_work" / "unified_gt"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ei  # noqa: E402  (aggiunge famos, recon, protocolli, baselines_mm al path)
import famos_common as fc  # noqa: E402

EVID = AAU / "runs" / "evidence" / "famos_calib"
CAL = REPO / "datasets" / "FAMOS" / "calib"
SEED_CAL, N_FOLD = 20261015, 5
D_U = 256
LAM = (1.0, 1e-1, 1e-2, 1e-3, 1e-4)                      # regolarizzazione relativa: lam x tr(S_w) / 256
K_LDA = (2, 4, 8, 16, 32, 48, 63)                        # dimensioni LDA (<= soggetti di stima - 1)
GAMMA = tuple(float(2.0 ** (k / 2)) for k in range(-8, 9))  # fattore della d' ricomposta con la taglia
COMP_METHODS = {"icp": ("icp", "mm"), "now": ("now_metrical", "mm")}   # griglia ri -> colonna di esterna_famos_id
MARK = "<!-- da qui in giu': scritto a mano, conservato da fcal.py eval -->"
_ST: dict = {}


# ------------------------------------------------------------------------------------------ comuni

def sequences() -> list[str]:
    return list(json.loads((ei.EVID / "selection.json").read_text())["eval"]["sequences"])


def selection() -> dict:
    return json.loads((EVID / "selection.json").read_text())


def all_items() -> list[tuple[str, str, Path]]:
    """(view_id, sequenza, patch) di tutti i soggetti selezionati: soggetti ordinati x sequenze."""
    sel = selection()
    return [(v, t, CAL / "npz" / f"{v}_GTready_{t}.npz") for v in sorted(sel["frames"]) for t in sel["sequences"]]


def spec_path(k: int) -> Path:
    return EVID / f"spec_f{k}.json"


def folds() -> dict:
    return json.loads((EVID / "folds.json").read_text())


def items() -> list[tuple[str, str, Path]]:
    """Elementi dei soggetti tenuti (``folds``), nell'ordine di ``all_items``."""
    keep = {v for f in folds()["folds"] for v in f}
    return [x for x in all_items() if x[0] in keep]


# ------------------------------------------------------------------------------------------ selezione

def _expr_task(subject: str) -> dict:
    """``famos_test_view.select`` sulle registrazioni sottocampionate del soggetto TRAIN, per le sequenze valutate."""
    from shapes import unified_distances
    sp = _ST["sp"]
    with np.load(fc.TRAIN_DIR / f"{subject}.npz") as z:
        seq, fr = z["seq"].astype(str), np.asarray(z["frame"], np.int64)
        V, Vn = np.asarray(z["V"], np.float64), np.asarray(z["V_neutral"], np.float64)
    a, _, _ = sp.align(sp.map("flame", np.concatenate([Vn[None], V])))
    S = sp.svec(a)
    e = unified_distances(S[1:], S[:1], sp.A)[:, 0]
    g = int(np.argmin(e))
    seqs = _ST["seqs"]
    missing = [q for q in seqs if not (seq == q).any()]
    if missing:
        return {"subject": subject, "missing": missing}
    out = {"subject": subject, "missing": [], "pool": {}, "frame": {}, "role": {}, "expr_mm": {}, "V": {}}
    for q in seqs:
        idx = np.flatnonzero(seq == q).tolist()
        pool = {}
        if seq[g] == q:
            pool[int(fr[g])] = ("gallery", g)
        pk = max(idx, key=lambda k: e[k])
        if pk != g:
            pool[int(fr[pk])] = ("peak", pk)
        rest = [k for k in idx if k not in (g, pk)]
        if rest:
            nn = min(rest, key=lambda k: e[k])
            pool[int(fr[nn])] = ("nearneutral", nn)
        f = ei.pick_frame(subject, q, list(pool))
        role, k = pool[f]
        out["pool"][q] = sorted(pool)
        out["frame"][q], out["role"][q], out["expr_mm"][q] = f, role, float(e[k])
        out["V"][q] = V[k].astype(np.float32)
    return out


def select(workers: int) -> None:
    from shapes import Space
    t0 = time.time()
    split = fc.load_split()
    train = sorted(split["train"])
    if set(train) & set(split["test"]):
        raise SystemExit("soggetti di TEST nella calibrazione")
    seqs = sequences()
    _ST.update(sp=Space(), seqs=seqs)
    with mp.get_context("fork").Pool(workers) as pool:
        res = sorted(pool.map(_expr_task, train, chunksize=1), key=lambda r: r["subject"])
    excluded = {r["subject"]: r["missing"] for r in res if r["missing"]}
    ok = [r for r in res if not r["missing"]]
    frames = {fc.view_id(r["subject"]): {q: r["frame"][q] for q in seqs} for r in ok}
    names, V, ex = [], [], []
    for r in ok:
        for q in seqs:
            names.append(f"{fc.view_id(r['subject'])}_GTready_{q}")
            V.append(r["V"][q])
            ex.append(r["expr_mm"][q])
    CAL.mkdir(parents=True, exist_ok=True)
    ei.atomic_savez(CAL / "reg_selected.npz", V=np.stack(V), names=np.asarray(names), expr_mm=np.asarray(ex))
    roles = [r["role"][q] for r in ok for q in seqs]
    ex = np.asarray(ex)
    sel = {"rule": "soggetti FaMoS TRAIN (split.json) con tutte le 10 sequenze di esterna_famos_id fra le registrazioni "
                   "sottocampionate (datasets/FAMOS/train, 1 ogni 10); pool per (soggetto, sequenza) = galleria (se nella "
                   "sequenza), peak, nearneutral (famos_test_view.select, expr_mm contro V_neutral); fotogramma = "
                   "ei.pick_frame (rng([20261013, numero del soggetto, crc32(sequenza)]))",
           "source": "registrazioni FaMoS TRAIN (mesh in topologia FLAME, mm); nessuna scansione grezza TRAIN su disco",
           "subjects_train": len(train), "subjects": [r["subject"] for r in ok], "subjects_excluded": excluded,
           "sequences": seqs, "frames": frames, "n_meshes": len(names),
           "roles": {k: roles.count(k) for k in ("gallery", "peak", "nearneutral")},
           "expr_mm": {"median": float(np.median(ex)), "p5": float(np.percentile(ex, 5)),
                       "p95": float(np.percentile(ex, 95))}, **ei.stamp()}
    ei.atomic_json(EVID / "selection.json", sel)
    print(f"[fcal] select: {len(ok)}/{len(train)} soggetti TRAIN, {len(names)} mesh, esclusi {excluded}, "
          f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------ preparazione

def regs() -> tuple[np.ndarray, np.ndarray]:
    """(V (n, 5023, 3) mm, F FLAME) nell'ordine di ``all_items``."""
    with np.load(CAL / "reg_selected.npz") as z:
        if list(z["names"].astype(str)) != [p.stem for _, _, p in all_items()]:
            raise SystemExit("reg_selected.npz con mesh diverse dalla selezione")
        V = np.asarray(z["V"], np.float64)
    return V, np.load(fc.FLAME_FACES).astype(np.int64)


def _render_reg(k: int) -> tuple:
    import pr_noise as prn
    from PIL import Image
    img, cam, box = prn.render_mesh(_ST["V"][k], _ST["F"])
    Image.fromarray(img).save(_ST["out"] / f"{k:04d}.png")
    return k, cam["o"], cam["f"], box


def render(kind: str, workers: int) -> None:
    """Render frontali nel formato di ``pr_noise_detect.py --dir`` (registrazione intera o patch)."""
    import pr_noise as prn
    t0 = time.time()
    it = all_items()
    out = CAL / f"{kind}_det"
    out.mkdir(parents=True, exist_ok=True)
    if kind == "raw":
        V, F = regs()
        _ST.update(V=V, F=F, out=out)                    # prima del fork del Pool
    with mp.get_context("fork").Pool(workers) as pool:
        if kind == "raw":
            res = pool.map(_render_reg, range(len(it)), chunksize=1)
        else:
            ok = {r["name"] for r in csv.DictReader(open(CAL / "patch_rows.csv"))}
            res = pool.map(ei._render_task, [(k, p, "patch", out) for k, (_, _, p) in enumerate(it) if p.stem in ok],
                           chunksize=1)
    res = sorted(res, key=lambda r: r[0])
    atomic = {"o": np.stack([r[1] for r in res]), "f": np.asarray([r[2] for r in res]), "box": np.stack([r[3] for r in res]),
              "k": np.asarray([r[0] for r in res]), "paths": np.asarray([it[r[0]][2].stem for r in res]), "img": prn.IMG}
    if kind == "patch":                                   # pr_noise_detect legge i png 0000.png ... in ordine
        for new, r in enumerate(res):
            if new != r[0]:
                (out / f"{r[0]:04d}.png").rename(out / f"{new:04d}.png")
    ei.atomic_savez(out / "cams.npz", **atomic)
    print(f"[fcal] render {kind}: {len(res)} immagini in {out}, {time.time() - t0:.0f}s", flush=True)


def det_split(kind: str, shards: int) -> None:
    """Il rilevatore (``pr_noise_detect.py --dir``) e' seriale: ``shards`` cartelle ``s<q>`` con i png a blocchi
    contigui (link) e il loro ``cams.npz``, da rilevare in parallelo e ricucire con ``det_merge``."""
    d = CAL / f"{kind}_det"
    with np.load(d / "cams.npz") as z:
        box, img = z["box"], z["img"]
    for q, part in enumerate(np.array_split(np.arange(len(box)), shards)):
        s = d / f"s{q}"
        s.mkdir(exist_ok=True)
        for f in s.glob("*.png"):
            f.unlink()
        for new, k in enumerate(part):
            (s / f"{new:04d}.png").symlink_to(d / f"{k:04d}.png")
        ei.atomic_savez(s / "cams.npz", box=box[part], img=img, k=part)


def det_merge(kind: str, shards: int) -> None:
    d = CAL / f"{kind}_det"
    out = {}
    for q in range(shards):
        with np.load(d / f"s{q}" / "lmk2d.npz") as y, np.load(d / f"s{q}" / "cams.npz") as c:
            for key in ("P", "det_score", "fallback", "n_faces"):
                out.setdefault(key, []).append(y[key])
            out.setdefault("k", []).append(c["k"])
    out = {key: np.concatenate(v) for key, v in out.items()}
    with np.load(d / "cams.npz") as z:
        if not np.array_equal(out.pop("k"), np.arange(len(z["box"]))):
            raise SystemExit(f"{kind}: shard del rilevatore incompleti")
    np.savez(d / "lmk2d.npz", **out)
    print(f"[fcal] det-merge {kind}: {len(out['P'])} render, ripieghi {int(out['fallback'].sum())}", flush=True)


def det_of(kind: str) -> dict:
    d = CAL / f"{kind}_det"
    with np.load(d / "cams.npz") as z, np.load(d / "lmk2d.npz") as y:
        out = {"o": z["o"], "f": z["f"], "paths": z["paths"].astype(str), "k": z["k"], "P": y["P"],
               "fallback": y["fallback"], "det_score": y["det_score"], "n_faces": y["n_faces"]}
    if len(out["P"]) != len(out["paths"]):
        raise SystemExit(f"{kind}: {len(out['P'])} rilevazioni per {len(out['paths'])} render")
    return out


def _patch_task(k: int) -> dict:
    try:
        return _patch(k)
    except Exception as exc:  # noqa: BLE001  (il soggetto esce dalla calibrazione, dichiarato)
        return {"k": k, "error": f"{type(exc).__name__}: {exc}"}


def _patch(k: int) -> dict:
    """``ei._patch`` riga per riga, con la registrazione al posto della scansione grezza."""
    import famos_test_view as ftv
    import now_common
    import now_prepare_meshes as nowprep
    dt, it = _ST["det"], _ST["items"][k]
    V, F = _ST["V"][k], _ST["F"]
    L68, miss = ei.backproject(V, F, {"o": dt["o"][k], "f": float(dt["f"][k])}, dt["P"][k], front_only=True)
    L7 = L68[list(now_common.LMK7_IBUG)]
    Vd, Fd, row, n_flip = ftv.canonical_patch(V, F, L7, _ST["template"], now_common.TARGET_FACES)
    Vmm = Vd / row["scale_to_mm"]                         # la similarita' sul template resa rigida: mm
    nowprep.mo.save_variant(Vmm, Fd, it[2])
    return {"k": k, "name": it[2].stem, "view_id": it[0], "seq": it[1], "n_vertices_raw": int(len(V)),
            "n_faces_reoriented": n_flip, "raw_det_score": float(dt["det_score"][k]),
            "raw_det_fallback": bool(dt["fallback"][k]), "raw_n_faces": int(dt["n_faces"][k]),
            "raw_rays_missed": int(miss.sum()), "crop_radius_mm": row["crop_radius_mm"] / row["scale_to_mm"], **row}


def patch(workers: int) -> None:
    import now_common
    t0 = time.time()
    it = all_items()
    (CAL / "npz").mkdir(parents=True, exist_ok=True)
    for stale in (CAL / "npz").glob("*.npz"):
        stale.unlink()
    dt = det_of("raw")
    if list(dt["paths"]) != [p.stem for _, _, p in it]:
        raise SystemExit("render della registrazione con mesh diverse dalla selezione")
    V, F = regs()
    template = np.asarray(json.loads(now_common.template_path().read_text())["landmarks_mm"], dtype=np.float64)
    _ST.update(det=dt, items=it, V=V, F=F, template=template)
    with mp.get_context("fork").Pool(workers, initializer=ei._cap_init) as pool:
        res = sorted(pool.map(_patch_task, range(len(it)), chunksize=1), key=lambda r: r["k"])
    errors = {it[r["k"]][2].stem: r["error"] for r in res if "error" in r}
    rows = [r for r in res if "error" not in r]
    with open(CAL / "patch_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ei.atomic_json(CAL / "patch_errors.json", {"errors": errors, **ei.stamp()})
    print(f"[fcal] patch: {len(rows)}/{len(it)} patch, errori {len(errors)} {list(errors.items())[:3]}, "
          f"{time.time() - t0:.0f}s", flush=True)


def lmk(workers: int) -> None:
    """68 landmark rilevati sulla patch (``ei._lmk_task``): quelli di NoW metrical e di ICP init_lmk."""
    t0 = time.time()
    it = all_items()
    dt = det_of("patch")
    rows = [it[k] for k in dt["k"]]
    if list(dt["paths"]) != [p.stem for _, _, p in rows]:
        raise SystemExit("render delle patch con mesh diverse")
    ei._ST.update(det=dt, items=rows)
    with mp.get_context("fork").Pool(workers) as pool:
        res = sorted(pool.map(ei._lmk_task, range(len(rows)), chunksize=1), key=lambda r: r[0])
    ei.atomic_savez(CAL / "lmk.npz", L68=np.stack([r[1] for r in res]), miss=np.stack([r[2] for r in res]),
                    fallback=dt["fallback"], det_score=dt["det_score"], n_faces=dt["n_faces"],
                    names=np.asarray([p.stem for _, _, p in rows]))
    print(f"[fcal] lmk: {len(res)} patch, ripieghi del rilevatore {int(dt['fallback'].sum())}, raggi a vuoto "
          f"{int(sum(r[2].sum() for r in res))}, {time.time() - t0:.0f}s", flush=True)


def make_folds() -> None:
    """Soggetti con tutte le patch -> N_FOLD fold (``rng(SEED_CAL).permutation``), spec per fold (formato di
    ``pr_common.view``) e resoconto della preparazione (solo controlli di qualita', nessuna distanza)."""
    import blmm
    sel = selection()
    err = json.loads((CAL / "patch_errors.json").read_text())["errors"]
    bad = sorted({n.split("_GTready_")[0] for n in err})
    keep = sorted(v for v in sel["frames"] if v not in bad)
    perm = np.random.default_rng(SEED_CAL).permutation(len(keep))
    fl = [sorted(keep[i] for i in part) for part in np.array_split(perm, N_FOLD)]
    prm = blmm.params()["domains"]["famos"]
    for k, subj in enumerate(fl):
        ei.atomic_json(spec_path(k), {
            "name": f"famos_calib_f{k}", "set": f"calib_f{k}", "dir": str((CAL / "npz").relative_to(REPO)),
            "subjects": subj, "topologies": sel["sequences"], "frames": {v: sel["frames"][v] for v in subj},
            "frame": {"u": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}, "L": float(prm["L"]),
            "cs_ref": float(prm["cs_ref"]),
            "note": "patch NoW in mm dalle registrazioni FaMoS TRAIN (aau/famos_calib/fcal.py); L e cs_ref = dominio "
                    "famos della matrice"})
    rows = list(csv.DictReader(open(CAL / "patch_rows.csv")))

    def st(key):
        x = np.asarray([float(r[key]) for r in rows])
        return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "min": float(x.min()),
                "max": float(x.max())}

    with np.load(CAL / "lmk.npz") as z:
        pdet = {"fallback": int(z["fallback"].sum()), "n_faces_not_1": int((z["n_faces"] != 1).sum()),
                "det_score_median": float(np.median(z["det_score"])), "rays_missed_total": int(z["miss"].sum())}
    prep = {"n_meshes": len(all_items()), "n_patches": len(rows), "errors": err, "subjects_dropped": bad,
            "raw_detector": {"fallback": sum(r["raw_det_fallback"] == "True" for r in rows),
                             "n_faces_not_1": sum(int(r["raw_n_faces"]) != 1 for r in rows),
                             "det_score": st("raw_det_score"),
                             "rays_missed_total": sum(int(r["raw_rays_missed"]) for r in rows)},
            "lmk7_residual_to_template_mm": st("lmk_residual_mm"), "scale_to_mm": st("scale_to_mm"),
            "crop_radius_mm": st("crop_radius_mm"), "n_vertices": st("n_vertices"), "n_faces": st("n_faces"),
            "patch_detector": pdet}
    ei.atomic_json(EVID / "prep_calib.json", {**prep, **ei.stamp()})
    ei.atomic_json(EVID / "folds.json", {"rule": f"rng({SEED_CAL}).permutation dei soggetti con tutte le patch, "
                                                 f"np.array_split in {N_FOLD}", "seed": SEED_CAL, "folds": fl,
                                         "subjects": len(keep), "subjects_dropped": bad, **ei.stamp()})
    print(f"[fcal] folds: {len(keep)} soggetti in {[len(f) for f in fl]}, tolti {bad}; prep {json.dumps(prep)[:600]}",
          flush=True)


# ------------------------------------------------------------------------------------------ nostro (GPU)

def embed(a) -> None:
    """Latenti [s, u] sulle patch di calibrazione; controllo sulle 150 di test contro i latenti di esterna_famos_id."""
    import torch
    from types import SimpleNamespace
    sys.path.insert(0, str(AAU / "recon"))
    from ws3b_latent import build_v1_model, encode
    if not torch.cuda.is_available():
        raise SystemExit("embed: serve la GPU")
    if Path(a.checkpoint).resolve() != ei.CKPT[a.model].resolve():
        raise SystemExit(f"embed: checkpoint diverso da quello di {a.model}")
    device = torch.device("cuda")
    model = build_v1_model(Path(a.checkpoint), "", device)
    out_root = CAL / "emb"
    names = [p.stem for _, _, p in items()]
    z = encode(model, "famos_calib", CAL / "ops", names, device,
               SimpleNamespace(out_root=out_root, metric=f"latent_{a.model}", overwrite=True))
    Z = np.stack([z[n] for n in names]).astype(np.float64)
    if Z.shape[1] != D_U + 1:
        raise SystemExit("servono i latenti [s, u]: WBES_V3_FACTORIZED_OUT=full")
    tn = [p.stem for _, _, p in ei.items("eval")]
    zt = encode(model, "esterna_id_check", ei.DATA / "eval" / "ops", tn, device,
                SimpleNamespace(out_root=out_root, metric=f"latent_{a.model}", overwrite=True))
    with np.load(ei.DATA / "eval" / "emb" / f"{a.model}.npz") as r:
        if list(r["names"].astype(str)) != tn:
            raise SystemExit("latenti di test con mesh diverse")
        ck = {"Z_test_vs_esterna_famos_id": ei.maxdiff(np.stack([zt[n] for n in tn]), r["Z"])}
    ck.update(gpu=torch.cuda.get_device_name(0), Z_finite=bool(np.isfinite(Z).all()), tol_Z=ei.TOL["Z"])
    ck["ok"] = bool(ck["Z_test_vs_esterna_famos_id"]["max_abs"] <= ei.TOL["Z"] and ck["Z_finite"])
    ei.atomic_savez(out_root / f"{a.model}.npz", Z=Z, names=np.asarray(names))
    ei.atomic_json(EVID / "controls" / f"embed_{a.model}.json", {**ck, "checkpoint": str(ei.CKPT[a.model].relative_to(REPO)),
                                                                 **ei.stamp()})
    print(f"[fcal] embed {a.model}: {Z.shape}, controllo {json.dumps(ck)}", flush=True)


# ------------------------------------------------------------------------------------------ concorrenti (griglia)

def fold_view(vname: str) -> dict:
    """``ri.view_data`` per un fold di calibrazione (la regola di FaMoS: X = indice minore, seme ``blmm.mesh_seed`` dei
    nomi; landmark rilevati sulla patch; voce di ``blmm.VIEWS`` come ``ei.inject_view``, dominio famos)."""
    import pr_common as pc
    blmm = pc.blmm
    k = int(vname.rsplit("_f", 1)[1])
    sp = json.loads(spec_path(k).read_text())
    if sp["name"] != vname:
        raise SystemExit(f"{vname}: spec di un altro fold")
    blmm.VIEWS[vname] = {"dir": REPO / sp["dir"], "domain": "famos", "gt": None, "template_dir": None, "fb_root": None,
                         "template_ref": None, "out": CAL / "grid" / vname,
                         "spec": {"path": str(spec_path(k)), "subjects": sp["subjects"], "template_subjects": [],
                                  "frame": sp["frame"], "ref": "evaluated", "neutral": False}}
    pv = pc.view(vname, spec_path(k))
    it = pv["items"]
    names = [p.stem for _, _, p in it]
    i, j = np.triu_indices(len(it), 1)
    with np.load(CAL / "lmk.npz") as z:
        pos = {n: q for q, n in enumerate(z["names"].astype(str))}
        L68 = np.asarray(z["L68"], np.float64)[[pos[n] for n in names]]
    return {"items": it, "i": i, "j": j, "seed": np.asarray([blmm.mesh_seed(names[a], names[b]) for a, b in zip(i, j)]),
            "L68": L68, "bview": vname, "pview": pv}


def grid(fold: int, method: str, workers: int) -> None:
    """``ri.run`` + ``ri.merge`` sul fold: tutte le configurazioni della griglia, parti in ``<CAL>/grid``."""
    import ri
    ri.view_data = fold_view
    ri.WORK = CAL / "grid"
    ri.EVID = EVID
    vname = f"famos_calib_f{fold}"
    ri.run(vname, method, workers, 0)
    ri.merge(vname, method)


# ------------------------------------------------------------------------------------------ calibrazione lineare

def scatter(U: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S_w intra-identita' con N - C gradi di liberta', S_b fra identita' pesata per numerosita')."""
    mu = U.mean(0)
    Sw, Sb = np.zeros((U.shape[1],) * 2), np.zeros((U.shape[1],) * 2)
    cls = np.unique(y)
    for c in cls:
        X = U[y == c]
        m = X.mean(0)
        Sw += (X - m).T @ (X - m)
        Sb += len(X) * np.outer(m - mu, m - mu)
    return Sw / (len(U) - len(cls)), Sb / len(U)


def fit(U: np.ndarray, y: np.ndarray, fam: str, lam: float, k: int = 0) -> np.ndarray:
    """Matrice P (256 x m): d' = ||P^T (u_i - u_j)||. WCCN: (S_w + lam t I)^(-1/2); LDA: i k autovettori principali."""
    from scipy.linalg import eigh
    Sw, Sb = scatter(U, y)
    R = Sw + lam * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    if fam == "wccn":
        e, Q = np.linalg.eigh(R)
        return Q @ np.diag(e ** -0.5) @ Q.T
    e, Vv = eigh(Sb, R)                                   # Vv^T R Vv = I, autovalori crescenti
    return Vv[:, ::-1][:, :k]


def pdist_u(U: np.ndarray, P: np.ndarray | None, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    X = U if P is None else U @ P
    return np.linalg.norm(X[i] - X[j], axis=1)


def form(S: np.ndarray, d: np.ndarray, i: np.ndarray, j: np.ndarray, c: float) -> np.ndarray:
    return np.sqrt((S[i] - S[j]) ** 2 + S[i] * S[j] * (c * d) ** 2)


def c0_of(U: np.ndarray, y: np.ndarray, P: np.ndarray, dpu: float) -> float:
    """mediana di d_P / mediana di d' sulle coppie di persone diverse dei soggetti di stima."""
    i, j = np.triu_indices(len(U), 1)
    imp = y[i] != y[j]
    return float(np.median(pdist_u(U, None, i[imp], j[imp]) * dpu) / np.median(pdist_u(U, P, i[imp], j[imp])))


def configs() -> list[tuple[str, dict]]:
    out = [("shape", {}), ("form", {"c": 1.0})]
    out += [("wccn", {"lam": lam}) for lam in LAM]
    out += [("lda", {"lam": lam, "k": k}) for lam in LAM for k in K_LDA]
    return out


def dists(fam: str, p: dict, Ufit, yfit, U, S, i, j, dpu, base=None) -> np.ndarray:
    """Distanze della variante ``fam`` (stima su ``Ufit``, ``yfit``) sulle coppie (i, j) di U."""
    if fam == "shape":
        return pdist_u(U, None, i, j) * dpu
    if fam in ("form", "form_c"):
        return form(S, pdist_u(U, None, i, j) * dpu, i, j, p["c"])
    if fam in ("wccn", "lda"):
        return pdist_u(U, fit(Ufit, yfit, fam, p["lam"], p.get("k", 0)), i, j)
    P = fit(Ufit, yfit, base[0], base[1]["lam"], base[1].get("k", 0))   # wccn_taglia, lda_taglia
    return form(S, pdist_u(U, P, i, j), i, j, p["g"] * c0_of(Ufit, yfit, P, dpu))


def cv(Z: np.ndarray, y: np.ndarray, fold: np.ndarray, dpu: float) -> tuple[list, dict]:
    """AUC per fold tenuto fuori (coppie interne al fold) di ogni configurazione; scelte per famiglia."""
    import famos_eval as fe
    U, S = Z[:, 1:], np.exp(Z[:, 0])
    rows, choice = [], {}

    def run(fam, p, base=None):
        auc = []
        for k in range(N_FOLD):
            te, tr = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
            i, j = np.triu_indices(len(te), 1)
            gen = y[te][i] == y[te][j]
            d = dists(fam, p, U[tr], y[tr], U[te], S[te], i, j, dpu, base)
            auc.append(fe.weighted_auc(-d, gen, np.ones(len(d))))
        rows.append({"famiglia": fam, "config": json.dumps(p), "auc_cv": float(np.mean(auc)),
                     **{f"auc_f{k}": float(a) for k, a in enumerate(auc)}})
        return float(np.mean(auc))

    for fam in ("shape", "form", "wccn", "lda"):
        cand = [(p, run(fam, p)) for f, p in configs() if f == fam
                and p.get("k", 0) <= min(len(np.unique(y[fold != k])) for k in range(N_FOLD)) - 1]
        best = int(np.argmax([a for _, a in cand]))      # pari merito: il primo della griglia
        choice[fam] = {"params": cand[best][0], "auc_cv": cand[best][1]}
    for fam, base in (("wccn_taglia", "wccn"), ("lda_taglia", "lda"), ("form_c", None)):
        b = (base, choice[base]["params"]) if base else None
        cand = [({"g": g} if base else {"c": g}, run(fam, {"g": g} if base else {"c": g}, b)) for g in GAMMA]
        best = int(np.argmax([a for _, a in cand]))
        choice[fam] = {"params": cand[best][0], "auc_cv": cand[best][1], **({"base": b[1]} if b else {})}
    return rows, choice


# ------------------------------------------------------------------------------------------ valutazione

FAMS = {"wccn": "WCCN (u sbiancata), d'", "lda": "LDA, d'", "wccn_taglia": "d_F ricomposta: taglia + u WCCN",
        "lda_taglia": "d_F ricomposta: taglia + u LDA", "form_c": "controllo: d_F con u non cal., c scelto su FaMoS TRAIN"}
BASE = {"form": "d_F non cal.", "form_cal": "d_F cal.", "shape": "d_P"}


def dpu_of(m: str) -> float:
    """dp_per_unit del checkpoint (``fact_paired.distances``: d_P = ||u_i - u_j|| x dpu)."""
    sys.path.insert(0, str(REPO / "v3_work" / "trainer" / "tools"))
    import fact_paired as fp
    Z = np.zeros((2, D_U + 1))
    Z[1, 1] = 1.0
    return float(fp.distances(Z, np.asarray([0]), np.asarray([1]), ei.CKPT[m])["shape"][0])


def comp_choice() -> tuple[list, dict]:
    """AUC media sui fold delle configurazioni di ICP mm e NoW metrical (coppie interne ai fold)."""
    import famos_eval as fe
    import ri
    rows, choice = [], {}
    for m in COMP_METHODS:
        auc = {c: [] for c in ri.GRID[m]}
        for k in range(N_FOLD):
            vname = f"famos_calib_f{k}"
            sp = json.loads(spec_path(k).read_text())
            with np.load(CAL / "grid" / "dist" / vname / f"{m}.npz") as z:
                i, j = z["i"], z["j"]
                subj = np.asarray([n.split("|")[0] for n in z["names"].astype(str)])
                if sorted(set(subj)) != sp["subjects"]:
                    raise SystemExit(f"{vname} {m}: soggetti diversi dallo spec")
                gen = subj[i] == subj[j]
                for c in ri.GRID[m]:
                    x = np.asarray(z[f"D_{c}"], np.float64)
                    auc[c].append(fe.weighted_auc(-np.where(np.isfinite(x), x, np.inf), gen, np.ones(len(x))))
        for c, a in auc.items():
            rows.append({"famiglia": m, "config": c, "auc_cv": float(np.mean(a)),
                         **{f"auc_f{k}": float(x) for k, x in enumerate(a)}})
        best = max(auc, key=lambda c: (np.mean(auc[c]), -list(ri.GRID[m]).index(c)))  # pari merito: il primo
        choice[m] = {"config": best, "auc_cv": float(np.mean(auc[best])), "auc_cv_default": float(np.mean(auc["default"]))}
    return rows, choice


def evaluate() -> None:
    import famos_eval as fe
    import fd
    import ri
    t0 = time.time()
    it_t = ei.items("eval")
    n = len(it_t)
    iu, ju = ei.pairs(n)
    names_t = [p.stem for _, _, p in it_t]
    cols, ctrl = {}, {}
    base_cols, _, failed = ei.load_columns("eval")
    # concorrenti: default (esterna_famos_id) e configurazione scelta su calibrazione (distanze della griglia di ri)
    crows, cchoice = comp_choice()
    Vf = {"i": iu, "j": ju}
    for m, (src, r) in COMP_METHODS.items():
        cols[f"{src}|{r}"] = base_cols[f"{src}|{r}"]
        g = ri.load_dist("famos", m, Vf)
        ctrl[f"griglia_{m}_default_vs_esterna"] = ei.maxdiff(g["default"], base_cols[f"{src}|{r}"])
        x = g[cchoice[m]["config"]]
        cols[f"{m}@cal"] = np.where(np.isfinite(x), x, np.inf)
    sel_all = folds()
    cv_rows, choices, Zt = [], {}, {}
    for m in ei.CKPT:
        for r in BASE:
            cols[f"{m}|{r}"] = base_cols[f"{m}|{r}"]
        with np.load(CAL / "emb" / f"{m}.npz") as z:
            names = z["names"].astype(str)
            Z = np.asarray(z["Z"], np.float64)
        if list(names) != [p.stem for _, _, p in items()]:
            raise SystemExit(f"{m}: latenti di calibrazione con mesh diverse")
        vid = np.asarray([x.split("_GTready_")[0] for x in names])
        fold = np.asarray([next(k for k, f in enumerate(sel_all["folds"]) if v in f) for v in vid])
        y = np.unique(vid, return_inverse=True)[1]
        dpu = dpu_of(m)
        rows, ch = cv(Z, y, fold, dpu)
        cv_rows += [{"modello": m, **r} for r in rows]
        choices[m] = ch
        with np.load(ei.DATA / "eval" / "emb" / f"{m}.npz") as z:
            if list(z["names"].astype(str)) != names_t:
                raise SystemExit(f"{m}: latenti di test con mesh diverse")
            Zt[m] = np.asarray(z["Z"], np.float64)
        Ut, St = Zt[m][:, 1:], np.exp(Zt[m][:, 0])
        ctrl[f"{m}_form_ricalcolata_vs_esterna"] = ei.maxdiff(form(St, pdist_u(Ut, None, iu, ju) * dpu, iu, ju, 1.0),
                                                              base_cols[f"{m}|form"])
        for fam in FAMS:
            p = ch[fam]["params"]
            b = None
            if fam.endswith("_taglia"):
                b = (fam.split("_")[0], ch[fam]["base"])
            cols[f"{m}|{fam}"] = dists(fam, p, Z[:, 1:], y, Ut, St, iu, ju, dpu, b)
    # metriche sulle 150 di test (regole di ei.evaluate) e strati d'espressione (regole di famos_dominio)
    subj_names = sorted({x for x, _, _ in it_t})
    subj = np.asarray([subj_names.index(x) for x, _, _ in it_t])
    ns = len(subj_names)
    gen = subj[iu] == subj[ju]
    counts = np.concatenate([np.ones((1, ns), np.int64), fe.bootstrap_counts(ns, ei.N_BOOT, ei.BOOT_SEED)])
    W = np.where(gen[None], counts[:, subj[iu]], counts[:, subj[iu]] * counts[:, subj[ju]]).astype(np.float64)
    wq = counts[:, subj].astype(np.float64)
    s_ex, cuts = fd.tertiles(fd.expr_labels())
    ps = np.maximum(s_ex[iu], s_ex[ju])
    R = {}
    for key, x in cols.items():
        D = np.zeros((n, n))
        D[iu, ju] = D[ju, iu] = x
        r1, ap = ei.retrieval(D, subj)
        R[key, "all", "rank1"] = fd.wmean(r1, wq)
        R[key, "all", "mAP"] = fd.wmean(ap, wq)
        R[key, "all", "AUC"] = np.asarray([fe.weighted_auc(-x, gen, w) for w in W])
        R[key, "all", "EER"] = np.asarray([ei.eer(x, gen, w) for w in W])
        for k, st in enumerate(fd.STRATA):
            R[key, f"expr:{st}", "rank1"] = fd.wmean(r1 * (s_ex == k), wq * (s_ex == k))
            R[key, f"expr:{st}", "AUC"] = np.asarray([fd.auc_masked(-x, gen, w, ~gen | (ps == k)) for w in W])
        print(f"[fcal] eval {key}: rank-1 {R[key, 'all', 'rank1'][0]:.3f}, AUC {R[key, 'all', 'AUC'][0]:.3f}, "
              f"AUC_H {R[key, 'expr:H', 'AUC'][0]:.3f}", flush=True)
    with open(ei.EVID / "metrics.csv", newline="") as fh:
        ref = {(r["colonna"], r["metrica"]): float(r["valore"]) for r in csv.DictReader(fh)}
    ctrl["metriche_di_base_vs_esterna_max_abs"] = max(abs(R[k, "all", met][0] - v) for (k, met), v in ref.items()
                                                      if (k, "all", met) in R)
    if ctrl["metriche_di_base_vs_esterna_max_abs"] > 1e-12:
        raise SystemExit(f"metriche di base diverse da esterna_famos_id: {ctrl['metriche_di_base_vs_esterna_max_abs']}")

    label = {f"{src}|{r}": f"{ri.LABEL[m]}, default (non cal.)" for m, (src, r) in COMP_METHODS.items()}
    label.update({f"{m}@cal": f"{ri.LABEL[m]}, configurazione scelta su FaMoS TRAIN ({cchoice[m]['config']})"
                  for m in COMP_METHODS})
    for m in ei.CKPT:
        s = m.split("_")[1]
        label.update({f"{m}|{r}": f"nostro {s}, {lab} (non cal.)" for r, lab in BASE.items()})
        label.update({f"{m}|{f}": f"nostro {s}, {lab}" for f, lab in FAMS.items()})
    mrows = []
    for key in cols:
        for (k2, part, met), x in R.items():
            if k2 == key:
                lo, hi, nx = fd.ci(x)
                mrows.append({"colonna": key, "metodo": label[key], "insieme": part, "metrica": met,
                              "valore": float(x[0]), "ic_lo": lo, "ic_hi": hi, "repliche_escluse": nx,
                              "coppie_non_finite": int((~np.isfinite(cols[key])).sum())})
    drows = []
    refs = lambda m: [f"{m}|form"] + [f"{src}|{r}" for src, r in COMP_METHODS.values()] + \
        [f"{c}@cal" for c in COMP_METHODS]                # noqa: E731
    for m in ei.CKPT:
        for f in list(FAMS) + ["form"]:
            ko = f"{m}|{f}"
            for kc in refs(m):
                if kc == ko:
                    continue
                for (k2, part, met), x in R.items():
                    if k2 != ko:
                        continue
                    sgn = -1.0 if met == "EER" else 1.0
                    d = sgn * (x - R[kc, part, met])
                    lo, hi, nx = fd.ci(d)
                    drows.append({"nostro": ko, "nostro_etichetta": label[ko], "riferimento": kc,
                                  "riferimento_etichetta": label[kc], "insieme": part, "metrica": met,
                                  "delta": float(d[0]), "ic_lo": lo, "ic_hi": hi, "repliche_escluse": nx,
                                  "verso": "positivo = il nostro (variante) meglio"})
    EVID.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("cv.csv", cv_rows + [{"modello": "concorrente", **r} for r in crows]),
                        ("metrics.csv", mrows), ("deltas.csv", drows)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(EVID / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    ctl = {p.stem: json.loads(p.read_text()) for p in sorted((EVID / "controls").glob("*.json"))}
    summary = {"calibrazione": {"soggetti": sel_all["subjects"], "fold": [len(f) for f in sel_all["folds"]],
                                "mesh": len(items()), "soggetti_tolti": sel_all["subjects_dropped"]},
               "scelte_nostro": choices, "scelte_concorrenti": cchoice, "terzili_expr_mm": cuts,
               "controlli": ctrl, "controlli_ok": {k: c.get("ok") for k, c in ctl.items() if "ok" in c},
               "test": {"scansioni": n, "soggetti": ns, "coppie": len(iu), "stessa_persona": int(gen.sum())},
               "repliche": ei.N_BOOT, "seme": ei.BOOT_SEED, "seconds": time.time() - t0, **ei.stamp()}
    ei.atomic_json(EVID / "summary.json", summary)
    write_results(mrows, drows, summary, label)
    print(f"[fcal] eval: {len(cols)} colonne, {time.time() - t0:.0f}s", flush=True)


def write_results(mrows: list, drows: list, summary: dict, label: dict) -> None:
    sel = selection()
    prep = json.loads((EVID / "prep_calib.json").read_text())

    def cell(key, part, met):
        r = next((x for x in mrows if x["colonna"] == key and x["insieme"] == part and x["metrica"] == met), None)
        return f"{r['valore']:.3f} [{r['ic_lo']:.3f}, {r['ic_hi']:.3f}]" if r else "-"

    def dcell(ko, kc, part, met):
        r = next((x for x in drows if x["nostro"] == ko and x["riferimento"] == kc and x["insieme"] == part
                  and x["metrica"] == met), None)
        return f"{r['delta']:+.3f} [{r['ic_lo']:+.3f}, {r['ic_hi']:+.3f}]" if r else "-"

    cal = summary["calibrazione"]
    L = ["# FaMoS: calibrazione lineare post hoc per l'invarianza all'espressione", "",
         "Generato da `aau/famos_calib/fcal.py eval` (regole in `REGOLE.md`, committate prima dei numeri). **POST HOC e "
         "DESCRITTIVO**: FaMoS TEST e' gia' stato visto (`esterna_famos_id`, `famos_dominio`); nessun test confermativo.",
         "", "## Dati di calibrazione", "",
         f"- {cal['soggetti']} soggetti FaMoS TRAIN su {sel['subjects_train']} (esclusi per sequenze mancanti: "
         f"{len(sel['subjects_excluded'])}; tolti per patch fallite: {len(cal['soggetti_tolti'])}), {cal['mesh']} mesh, "
         f"le 10 sequenze di `esterna_famos_id`; fold {cal['fold']}. Ruoli dei fotogrammi: {sel['roles']}; expr_mm "
         f"mediana {sel['expr_mm']['median']:.2f} mm (p5 {sel['expr_mm']['p5']:.2f}, p95 {sel['expr_mm']['p95']:.2f}).",
         "- **Dominio diverso dal test**: registrazioni FaMoS (mesh in topologia FLAME, superficie pulita e senza buchi), "
         "non scansioni grezze (quelle di TRAIN non sono su disco). Stessa preparazione di `ei.py`.",
         f"- Preparazione: {prep['n_patches']}/{prep['n_meshes']} patch, ripieghi del rilevatore sulla registrazione "
         f"{prep['raw_detector']['fallback']}, sulla patch {prep['patch_detector']['fallback']}; residuo dei 7 landmark "
         f"mediana {prep['lmk7_residual_to_template_mm']['median']:.2f} mm (`prep_calib.json`).", "",
         "## Scelte sulla calibrazione (validazione incrociata sui fold, AUC media)", "",
         "| modello | variante | parametri scelti | AUC cv |", "|---|---|---|---|"]
    for m, ch in summary["scelte_nostro"].items():
        for fam, c in ch.items():
            L.append(f"| {m} | {fam} | `{json.dumps({**c['params'], **({'base': c['base']} if 'base' in c else {})})}` | "
                     f"{c['auc_cv']:.3f} |")
    for m, c in summary["scelte_concorrenti"].items():
        L.append(f"| concorrente | {m} | `{c['config']}` (default: AUC cv {c['auc_cv_default']:.3f}) | {c['auc_cv']:.3f} |")
    parts = (("all", "rank1"), ("all", "mAP"), ("all", "EER"), ("all", "AUC"), ("expr:H", "AUC"), ("expr:H", "rank1"),
             ("expr:L", "AUC"))
    L += ["", "## Test: 15 soggetti FaMoS TEST, 150 scansioni grezze (protocollo di `esterna_famos_id`)", "",
          f"valore [IC 95%], {summary['repliche']} repliche per soggetto, seme {summary['seme']}. Strati d'espressione "
          f"di `famos_dominio` (terzili di expr_mm {summary['terzili_expr_mm'][0]:.3f} / "
          f"{summary['terzili_expr_mm'][1]:.3f} mm): AUC_k = stessa persona dello strato k contro tutte le coppie di "
          "persone diverse; rank-1_k = query dello strato.", "",
          "| metodo | rank-1 | mAP | EER | AUC | AUC H | rank-1 H | AUC L |", "|---|---|---|---|---|---|---|---|"]
    for key in dict.fromkeys(r["colonna"] for r in mrows):
        L.append(f"| {label[key]} | " + " | ".join(cell(key, p, met) for p, met in parts) + " |")
    for m in ei.CKPT:
        for kc in [f"{m}|form", "icp|mm", "icp@cal", "now_metrical|mm", "now@cal"]:
            L += ["", f"## Delta: varianti {m.split('_')[1]} contro {label[kc]}", "",
                  "positivo = la variante meglio (EER: riferimento - variante); stesse repliche.", "",
                  "| variante | rank-1 | mAP | EER | AUC | AUC H | rank-1 H | AUC L |", "|---|---|---|---|---|---|---|---|"]
            for f in list(FAMS) + ["form"]:
                ko = f"{m}|{f}"
                if ko != kc:
                    L.append(f"| {label[ko]} | " + " | ".join(dcell(ko, kc, p, met) for p, met in parts) + " |")
    L += ["", "## Controlli", "", f"- `{json.dumps(summary['controlli'])}`",
          f"- controlli_ok: `{json.dumps(summary['controlli_ok'])}`", "",
          "File: `cv.csv` (tutte le configurazioni, AUC per fold), `metrics.csv`, `deltas.csv`, `summary.json`, "
          "`selection.json`, `folds.json`, `prep_calib.json`, `controls/`.", ""]
    out = EVID / "results.md"
    tail = out.read_text().split(MARK, 1)[1] if out.exists() and MARK in out.read_text() else "\n"
    out.write_text("\n".join(L) + "\n" + MARK + tail)


# ------------------------------------------------------------------------------------------ main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("select", "render-raw", "patch", "render-patch", "lmk", "folds", "embed", "grid",
                                    "eval", "det-split", "det-merge"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--kind", default="raw", choices=("raw", "patch"))
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--model", default="", choices=("",) + tuple(ei.CKPT))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--method", default="icp", choices=tuple(COMP_METHODS))
    a = p.parse_args()
    if a.step == "select":
        select(a.workers)
    elif a.step in ("render-raw", "render-patch"):
        render(a.step.split("-")[1], a.workers)
    elif a.step == "det-split":
        det_split(a.kind, a.shards)
    elif a.step == "det-merge":
        det_merge(a.kind, a.shards)
    elif a.step == "patch":
        patch(a.workers)
    elif a.step == "lmk":
        lmk(a.workers)
    elif a.step == "folds":
        make_folds()
    elif a.step == "embed":
        embed(a)
    elif a.step == "grid":
        grid(a.fold, a.method, a.workers)
    else:
        evaluate()


if __name__ == "__main__":
    main()
