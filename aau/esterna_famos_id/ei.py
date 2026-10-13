#!/usr/bin/env python3
"""Validazione esterna su FaMoS TEST: riconoscimento dell'identita' su scansioni GREZZE, etichetta = stessa persona,
nessuna GT geometrica. Regole in ``aau/runs/evidence/esterna_famos_id/PROTOCOLLO.md`` (scritte prima dei numeri).
Le implementazioni dei metodi sono quelle della matrice, importate: qui solo la selezione, la preparazione comune,
le coppie e le metriche di riconoscimento.

    v3_work/unified_gt/run.sh aau/esterna_famos_id/ei.py select
    v3_work/unified_gt/run.sh aau/esterna_famos_id/ei.py render-raw --set eval --workers 16
    aau/recon/run_recon.sh mica aau/competitors/protocolli/pr_noise_detect.py --dir <DATA>/eval/raw_det
    v3_work/unified_gt/run.sh aau/esterna_famos_id/ei.py patch --set eval --workers 16
    v3_work/unified_gt/run.sh aau/esterna_famos_id/ei.py render-patch --set eval --workers 16
    aau/recon/run_recon.sh mica aau/competitors/protocolli/pr_noise_detect.py --dir <DATA>/eval/patch_det
    v3_work/unified_gt/run.sh aau/esterna_famos_id/ei.py lmk --set eval --workers 16
    v3_work/unified_gt/run.sh aau/esterna_famos_id/ei.py prep-report --set pilot|eval
    aau/run.sh v3_work/trainer/eval_v3.py -- aau/esterna_famos_id/ei.py embed --checkpoint <pth> --model <nome>  (GPU)
    AAU_NV=--nv aau/run.sh aau/esterna_famos_id/ei.py ulrssm                                                   (GPU)
    aau/outlineB/run_o3d.sh aau/esterna_famos_id/ei.py icp --workers 64
    v3_work/unified_gt/run.sh aau/esterna_famos_id/ei.py now --workers 64
    aau/competitors/gw_percettivi/gp_run.sh aau/esterna_famos_id/ei.py gw --workers 64
    v3_work/unified_gt/run.sh aau/esterna_famos_id/ei.py eval
    (ei.sbatch: un passo per job; operatori e tabella di scala del nostro sono passi di ei.sbatch)

**Selezione** (``select``, deterministica, nessuna distanza): soggetti = i 15 di FaMoS TEST (``split.json``); sequenze =
``K_SEQ`` estratte con ``SEED`` fra quelle con almeno una scansione per TUTTI i soggetti, le stesse per tutti; per ogni
(soggetto, sequenza) un fotogramma fra le scansioni su disco, estratto con ``[SEED, numero del soggetto, crc32 della
sequenza]``. Le scansioni su disco sono il pool di ``famos_test_view.py`` (fino a 3 per sequenza, scelte allora con
``expr_mm`` della registrazione: nessuna distanza d'identita'). Il pilota della preparazione (``--set pilot``) usa
``N_PILOT_SEQ`` sequenze NON selezionate, stessa regola: mai le scansioni valutate. Uscite: ``selection.json`` e
``spec_<set>.json`` (formato di ``pr_common`` / ``ulr_common`` / ``gp_common``: soggetto = view_id, topologia = sequenza,
frame identita', L e CS_ref del dominio ``famos`` della matrice) nella cartella dell'evidenza.

**Preparazione** (identica per tutti i metodi; nessuna registrazione FLAME): (1) render frontale della scansione grezza
in mm nel frame di acquisizione (``pr_noise.render_mesh``, invariata: camera sull'asse +z, il frame di FaMoS ha +y in alto
e il volto verso +z), 68 landmark di insightface ``1k3d68`` (``pr_noise_detect.py --dir``), retroproiezione col raggio
della camera (le righe della condizione ``det`` di ``pr_noise._prep_task``, in ``backproject``); (2) patch di NoW coi 7
landmark rilevati (``famos_test_view.canonical_patch``: similarita' sul template T7, ritaglio ``compute_mask``,
pulizia, ``bfs_orient``, 5215 triangoli), riportata in mm dividendo per la scala della similarita' (come
``blmm.to_mm`` per FaMoS): ``<DATA>/<set>/npz/<view_id>_GTready_<sequenza>.npz`` (V mm, F); (3) i landmark dei
protocolli NoW si rilevano di nuovo sulla patch con la stessa catena (la ``lmk_det`` della matrice).

**Metodi** (distanze su tutte le coppie i < j delle N scansioni, stesse coppie per tutti): nostro (factorized s1234 e
s2345, epoch072 EMA: ``ws3b_latent.encode`` dentro ``eval_v3.py`` con la tabella di scala della patch in mm, poi
``fact_paired.distances``: d_F non cal., d_F cal. con c di ``factorized_calibration.csv``, d_P); ICP + Chamfer in mm e
cs (``blmm_pairs.run_items`` / ``blmm.pair_metrics``, X = indice minore, seme ``blmm.mesh_seed`` dei nomi, la regola di
FaMoS in blmm_pairs); NoW originale e metrical (``pr_run.one_pair``, media dei due versi, ritaglio ``pm.crop`` coi
landmark rilevati, ``--now-opt`` 0); ULRSSM ``faces_xyz`` per coppia (``ulr_eval._prep``, ``on_gpu``,
``ulr_common.pair_rms``: letture mm e cs); GW ``euclid|128|cgw2`` (``gp_gw.run_prep`` e ``solve_pairs``: mm e cs). Ogni
passo ha un controllo di riproduzione contro i valori della matrice sulle 15 scansioni di galleria di FaMoS TEST
(``controls``).

**Metriche** (``eval``): rank-1 e mAP (ogni scansione e' la query, la galleria tutte le altre), AUC ed EER di verifica
(coppie stessa persona contro persone diverse); distanza non finita = +inf. IC 95% percentile con 1000 repliche per
soggetto (``famos_eval.bootstrap_counts(15, 1000, BOOT_SEED)``, le repliche di FaMoS TEST della matrice); delta appaiati
nostro - concorrente sulle stesse repliche (EER: concorrente - nostro; positivo = il nostro meglio). Descrittivo.

Dati derivati da FaMoS (licenza MPI) solo in ``datasets/FAMOS/esterna_id`` (fuori da git); in git solo nomi, numeri
aggregati e controlli.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
REPO = AAU.parent
PROT = AAU / "competitors" / "protocolli"
ULR = AAU / "competitors" / "ulrssm"
GP = AAU / "competitors" / "gw_percettivi"
for _p in (AAU / "famos", AAU / "recon", PROT, AAU / "baselines_mm", AAU / "baselines_param"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import famos_common as fc  # noqa: E402

EVID = AAU / "runs" / "evidence" / "esterna_famos_id"
DATA = REPO / "datasets" / "FAMOS" / "esterna_id"
SETS = ("eval", "pilot")
VNAME = {"eval": "esterna_id", "pilot": "esterna_id_pilot"}
SEED, K_SEQ, N_PILOT_SEQ = 20261013, 10, 2
BOOT_SEED, N_BOOT = 1234, 1000
GW_CFG = "euclid|128|cgw2"
MEM_CAP_GB = 12
_CK = "aau/runs/evidence/trainer_v3/ablations/c3f_runs"
_RUN = ("v3_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_poolmeanmax_noise60_sampler-balanced_pooling-area_"
        "meanmax_inputnorm-global_areaweights-smooth_emadecay-0.999_head-factorized_seed")
CKPT = {"factorized_s1234": REPO / _CK / "factorized" / f"{_RUN}1234__1f5d4a90" / "checkpoints" / "epoch072_ema.pth",
        "factorized_s2345": REPO / _CK / "factorized_s2345" / f"{_RUN}2345__4de9be29" / "checkpoints" / "epoch072_ema.pth"}
# latenti della matrice per FaMoS TEST (eval_famos_v3.py): controllo dell'embedding
EMB_REF = {"factorized_s1234": "v3factorizede072full", "factorized_s2345": "v3factorizeds2345e072full"}
ULR_CKPT = AAU / "runs" / "competitors_ulrssm" / "faces_xyz" / "final.pth"
# tolleranze dei controlli di riproduzione (fissate prima di eseguirli)
TOL = {"backproject_mm": 1e-9, "render_px": 0, "icp_mm": 1e-6, "now_mm": 1e-4, "gw_mm": 1e-5, "ulrssm_mm": 1e-2,
       "Z": 5e-3}

# colonne: (chiave, etichetta, famiglia, nostro)
OURS = [(f"{m}|{r}", f"nostro {m.split('_')[1]}, {lab}", "nostro", True)
        for m in CKPT for r, lab in (("form", "d_F non cal."), ("form_cal", "d_F cal."), ("shape", "d_P"))]
COMPETITORS = [("icp|mm", "ICP + Chamfer in mm", "ICP", False), ("icp|cs", "ICP + Chamfer cs", "ICP", False),
               ("now_metrical|mm", "NoW metrical (rilevatore)", "NoW", False),
               ("now_original|mm", "NoW originale (rilevatore)", "NoW", False),
               ("ulrssm|pair_fr", "ULRSSM faces_xyz per coppia, mm", "ULRSSM", False),
               ("ulrssm|pair_sr", "ULRSSM faces_xyz per coppia, cs", "ULRSSM", False),
               ("gw|fr", f"GW {GW_CFG.replace('|', '_')}, mm", "GW", False),
               ("gw|sr", f"GW {GW_CFG.replace('|', '_')}, cs", "GW", False)]
COLUMNS = OURS + COMPETITORS
METRICS = ("rank1", "mAP", "AUC", "EER")
MARK = "<!-- da qui in giu': scritto a mano, conservato da ei.py eval -->"
_ST: dict = {}


# ------------------------------------------------------------------------------------------ comuni

def stamp() -> dict:
    return {"host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID", "none"),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default) + "\n")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def spec_path(s: str) -> Path:
    return EVID / f"spec_{s}.json"


def items(s: str) -> list[tuple[str, str, Path]]:
    """(view_id, sequenza, patch) nell'ordine delle viste da spec: soggetti ordinati x sequenze dello spec."""
    sp = json.loads(spec_path(s).read_text())
    d = REPO / sp["dir"]
    return [(x, t, d / f"{x}_GTready_{t}.npz") for x in sorted(sp["subjects"]) for t in sp["topologies"]]


def raw_of(s: str) -> list[Path]:
    """Scansione grezza di ogni elemento di ``items``."""
    sp = json.loads(spec_path(s).read_text())
    inv = {fc.view_id(x): x for x in sp["famos_subjects"]}
    return [fc.scan_path(inv[v], t, int(sp["frames"][v][t])) for v, t, _ in items(s)]


def pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Tutte le coppie i < j (stessa persona compresa): le stesse per tutti i metodi."""
    return np.triu_indices(n, 1)


def dist_path(s: str, method: str) -> Path:
    return DATA / s / "dist" / f"{method}.npz"


def save_dist(s: str, method: str, readings: dict, extra: dict) -> None:
    it = items(s)
    iu, ju = pairs(len(it))
    for k, v in readings.items():
        if len(v) != len(iu):
            raise SystemExit(f"{method}|{k}: {len(v)} valori per {len(iu)} coppie")
    atomic_savez(dist_path(s, method), names=np.asarray([p.stem for _, _, p in it]), i=iu, j=ju,
                 **{f"D_{k}": np.asarray(v, np.float64) for k, v in readings.items()})
    atomic_json(EVID / "controls" / f"{method}.json", {"method": method, "set": s, "n_meshes": len(it),
                                                       "n_pairs": len(iu), **extra, **stamp()})
    print(f"[ei] {method}: {len(iu)} coppie, letture {list(readings)} -> {dist_path(s, method)}", flush=True)


def maxdiff(x, y) -> dict:
    x, y = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    d = np.abs(x[ok] - y[ok])
    return {"n": int(ok.sum()), "nan": int((~ok).sum()), "max_abs": float(d.max()) if len(d) else None,
            "median_abs": float(np.median(d)) if len(d) else None,
            "p95_abs": float(np.percentile(d, 95)) if len(d) else None, "equal": int((d == 0).sum())}


def scalars(s: str) -> dict:
    """{percorso relativo -> {cs, ...}} di ``blmm.mesh_scalars`` sulle patch in mm (cache ``<DATA>/<set>/scalars.npz``,
    formato di blmm_scalars.py): la stessa centroid size per ICP cs, ULRSSM cs e GW cs."""
    import blmm
    p = DATA / s / "scalars.npz"
    if not p.exists():
        it = items(s)
        res = []
        for _, _, q in it:
            V, F = blmm.load_raw(q)
            res.append(blmm.mesh_scalars(V, F))
        arr = {k: np.asarray([r[k] for r in res]) for k in res[0]}
        atomic_savez(p, paths=np.asarray([blmm.rel(q) for _, _, q in it]), **arr)
    with np.load(p) as z:
        cols = [c for c in z.files if c != "paths"]
        return {str(q): {c: float(z[c][k]) for c in cols} for k, q in enumerate(z["paths"])}


# ------------------------------------------------------------------------------------------ selezione

def scan_pool() -> dict:
    """{soggetto TEST: {sequenza: [fotogrammi con una scansione su disco]}}."""
    split = fc.load_split()
    out = {}
    for s in split["test"]:
        out[s] = {sd.name: [f for f, _ in fc.frames_of(sd, "obj")]
                  for sd in sorted(p for p in (fc.SCAN_DIR / s).iterdir() if p.is_dir())}
        out[s] = {q: f for q, f in out[s].items() if f}
    return out


def pick_frame(subject: str, seq: str, frames: list[int]) -> int:
    rng = np.random.default_rng([SEED, fc.subject_num(subject), zlib.crc32(seq.encode())])
    return int(sorted(frames)[int(rng.integers(len(frames)))])


def select(_a) -> None:
    import blmm
    split = fc.load_split()
    pool = scan_pool()
    common = sorted(set.intersection(*(set(v) for v in pool.values())))
    seqs = sorted(np.random.default_rng(SEED).choice(common, K_SEQ, replace=False).tolist())
    rest = sorted(set(common) - set(seqs))
    pilot = sorted(np.random.default_rng([SEED, 1]).choice(rest, N_PILOT_SEQ, replace=False).tolist())
    prm = blmm.params()["domains"]["famos"]
    sel = {"rule": "sequenze: rng(SEED).choice(sequenze comuni a tutti i 15 soggetti, K_SEQ, senza reinserimento), le "
                   "stesse per tutti; fotogramma: rng([SEED, numero del soggetto, crc32(sequenza)]) fra le scansioni su "
                   "disco della sequenza (pool di famos_test_view.py); pilota: rng([SEED, 1]).choice(sequenze comuni "
                   "non selezionate, N_PILOT_SEQ)",
           "seed": SEED, "k_seq": K_SEQ, "n_pilot_seq": N_PILOT_SEQ, "subjects": split["test"],
           "subjects_excluded": [], "common_sequences": common,
           "sequences_not_common": {s: sorted(set(pool[s]) ^ set(common)) for s in split["test"]
                                    if set(pool[s]) != set(common)},
           "pool_frames_per_sequence": {str(k): int(v) for k, v in zip(*np.unique(
               [len(f) for s in pool for f in pool[s].values()], return_counts=True))}}
    for st, qq in (("eval", seqs), ("pilot", pilot)):
        frames = {fc.view_id(s): {q: pick_frame(s, q, pool[s][q]) for q in qq} for s in split["test"]}
        sel[st] = {"sequences": qq, "frames": frames, "n_scans": len(qq) * len(split["test"])}
        spec = {"name": VNAME[st], "set": st, "dir": str((DATA / st / "npz").relative_to(REPO)),
                "subjects": sorted(frames), "topologies": qq, "famos_subjects": split["test"], "frames": frames,
                "frame": {"u": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]},
                "L": float(prm["L"]), "cs_ref": float(prm["cs_ref"]),
                "note": "patch NoW in mm dalle scansioni grezze FaMoS TEST (aau/esterna_famos_id/ei.py); L e cs_ref = "
                        "dominio famos della matrice (baselines_mm/params.json)"}
        atomic_json(spec_path(st), spec)
    if set(seqs) & set(pilot):
        raise SystemExit("pilota e valutazione con sequenze in comune")
    atomic_json(EVID / "selection.json", {**sel, **stamp()})
    print(f"[ei] select: {len(common)} sequenze comuni, valutate {seqs}, pilota {pilot}", flush=True)


# ------------------------------------------------------------------------------------------ preparazione

def backproject(V: np.ndarray, F: np.ndarray, cam: dict, P2: np.ndarray,
                front_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """68 punti 2D -> 3D col raggio della camera sul primo punto colpito; raggio a vuoto: il pixel colpito piu' vicino.
    Le righe della condizione ``det`` di ``pr_noise._prep_task`` (controllo ``backproject_vs_pr_noise``).

    ``front_only`` (solo le scansioni GREZZE, passo 1; regola fissata sul pilota): un colpo su un triangolo visto dal
    retro conta come raggio a vuoto. Le grezze hanno buchi (canti, narici): il raggio entra e colpisce l'interno della
    nuca, 190 mm dietro (pilota: 089/cheeks_in.000152, en_l a z = -136 mm, scala della similarita' 0,35). Il verso del
    retro si fissa per scansione dalla maggioranza dei pixel colpiti (il fronte e' la maggioranza), quindi non dipende
    dall'orientamento delle facce del file; il ripiego e' il pixel piu' vicino con un colpo visto di fronte."""
    import pr_noise as prn
    scene = prn.scene_of(V, F)
    r = prn.cast(scene, cam, P2[:, :2])
    miss = ~r["hit"]
    rf = full = None
    if front_only:
        yy, xx = np.mgrid[0:prn.IMG, 0:prn.IMG]
        full = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
        rf = prn.cast(scene, cam, full)
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        dot = lambda c: (fn[np.where(c["hit"], c["tri"], 0)] * c["d"]).sum(1)  # noqa: E731
        df = dot(rf)[rf["hit"]]
        sgn = 1.0 if (df < 0).sum() >= (df > 0).sum() else -1.0
        rf["hit"] = rf["hit"] & (sgn * dot(rf) < 0)
        miss = miss | (sgn * dot(r) >= 0)
    L = cam["o"] + r["d"] * np.where(miss, 0.0, r["t"])[:, None]
    if miss.any():
        if rf is None:
            yy, xx = np.mgrid[0:prn.IMG, 0:prn.IMG]
            full = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
            rf = prn.cast(scene, cam, full)
        hp = full[rf["hit"]]
        j = np.linalg.norm(hp[None] - P2[miss, None, :2], axis=2).argmin(1)
        jj = np.flatnonzero(rf["hit"])[j]
        L[miss] = cam["o"] + rf["d"][jj] * rf["t"][jj][:, None]
    return L, miss


def _render_task(task) -> tuple:
    import pr_noise as prn
    from PIL import Image
    k, path, kind, out = task
    if kind == "raw":
        V, F = fc.read_obj(path)
    else:
        with np.load(path) as z:
            V, F = np.asarray(z["V"], np.float64), np.asarray(z["F"], np.int64)
    img, cam, box = prn.render_mesh(V, F)
    Image.fromarray(img).save(out / f"{k:04d}.png")
    return k, cam["o"], cam["f"], box


def render(s: str, kind: str, workers: int) -> None:
    """Render frontali (``pr_noise.render_mesh``) nel formato di ``pr_noise_detect.py --dir``."""
    import pr_noise as prn
    t0 = time.time()
    paths = raw_of(s) if kind == "raw" else [p for _, _, p in items(s)]
    out = DATA / s / f"{kind}_det"
    out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(workers) as pool:
        res = sorted(pool.map(_render_task, [(k, p, kind, out) for k, p in enumerate(paths)], chunksize=1),
                     key=lambda r: r[0])
    atomic_savez(out / "cams.npz", o=np.stack([r[1] for r in res]), f=np.asarray([r[2] for r in res]),
                 box=np.stack([r[3] for r in res]), paths=np.asarray([str(p) for p in paths]), img=prn.IMG)
    print(f"[ei] render {kind} {s}: {len(res)} immagini in {out}, {time.time() - t0:.0f}s", flush=True)


def det_of(s: str, kind: str) -> dict:
    d = DATA / s / f"{kind}_det"
    with np.load(d / "cams.npz") as z, np.load(d / "lmk2d.npz") as y:
        out = {"o": z["o"], "f": z["f"], "paths": z["paths"].astype(str), "P": y["P"], "fallback": y["fallback"],
               "det_score": y["det_score"], "n_faces": y["n_faces"]}
    want = [str(p) for p in (raw_of(s) if kind == "raw" else [p for _, _, p in items(s)])]
    if list(out["paths"]) != want:
        raise SystemExit(f"{s} {kind}: render con mesh diverse dalla selezione")
    return out


def _cap_init() -> None:
    import resource
    cap = MEM_CAP_GB * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def _patch_task(k: int) -> dict:
    try:
        return _patch(k)
    except Exception as exc:  # noqa: BLE001  (la scansione resta fuori per TUTTI i metodi, dichiarata)
        return {"k": k, "error": f"{type(exc).__name__}: {exc}"}


def _patch(k: int) -> dict:
    import famos_test_view as ftv
    import now_common
    import now_prepare_meshes as nowprep
    dt, it, raw = _ST["det"], _ST["items"][k], _ST["raw"][k]
    V, F = fc.read_obj(raw)
    L68, miss = backproject(V, F, {"o": dt["o"][k], "f": float(dt["f"][k])}, dt["P"][k], front_only=True)
    L7 = L68[list(now_common.LMK7_IBUG)]
    Vd, Fd, row, n_flip = ftv.canonical_patch(V, F, L7, _ST["template"], now_common.TARGET_FACES)
    Vmm = Vd / row["scale_to_mm"]                         # la similarita' sul template resa rigida: mm
    nowprep.mo.save_variant(Vmm, Fd, it[2])
    return {"k": k, "name": it[2].stem, "view_id": it[0], "seq": it[1], "frame": int(raw.stem.split(".")[-1]),
            "n_vertices_raw": int(len(V)), "n_faces_reoriented": n_flip, "raw_det_score": float(dt["det_score"][k]),
            "raw_det_fallback": bool(dt["fallback"][k]), "raw_n_faces": int(dt["n_faces"][k]),
            "raw_rays_missed": int(miss.sum()), "crop_radius_mm": row["crop_radius_mm"] / row["scale_to_mm"], **row}


def patch(s: str, workers: int) -> None:
    import now_common
    t0 = time.time()
    it = items(s)
    (DATA / s / "npz").mkdir(parents=True, exist_ok=True)
    for stale in (DATA / s / "npz").glob("*.npz"):
        stale.unlink()
    template = np.asarray(json.loads(now_common.template_path().read_text())["landmarks_mm"], dtype=np.float64)
    _ST.update(det=det_of(s, "raw"), items=it, raw=raw_of(s), template=template)
    with mp.get_context("fork").Pool(workers, initializer=_cap_init) as pool:
        res = sorted(pool.map(_patch_task, range(len(it)), chunksize=1), key=lambda r: r["k"])
    errors = [f"{it[r['k']][0]}/{it[r['k']][1]}: {r['error']}" for r in res if "error" in r]
    rows = [r for r in res if "error" not in r]
    fields = list(rows[0]) if rows else ["k"]
    with open(DATA / s / "patch_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    atomic_json(DATA / s / "patch_errors.json", {"errors": errors, **stamp()})
    print(f"[ei] patch {s}: {len(rows)}/{len(it)} patch, errori {len(errors)} {errors[:3]}, {time.time() - t0:.0f}s",
          flush=True)


def _lmk_task(k: int) -> tuple:
    dt = _ST["det"]
    with np.load(_ST["items"][k][2]) as z:
        V, F = np.asarray(z["V"], np.float64), np.asarray(z["F"], np.int64)
    L68, miss = backproject(V, F, {"o": dt["o"][k], "f": float(dt["f"][k])}, dt["P"][k])
    return k, L68, miss


def lmk(s: str, workers: int) -> None:
    """68 landmark rilevati sulla patch (i landmark dei protocolli NoW) e controllo della retroproiezione."""
    t0 = time.time()
    it = items(s)
    dt = det_of(s, "patch")
    _ST.update(det=dt, items=it)
    with mp.get_context("fork").Pool(workers) as pool:
        res = sorted(pool.map(_lmk_task, range(len(it)), chunksize=1), key=lambda r: r[0])
    atomic_savez(DATA / s / "lmk.npz", L68=np.stack([r[1] for r in res]), miss=np.stack([r[2] for r in res]),
                 fallback=dt["fallback"], det_score=dt["det_score"], n_faces=dt["n_faces"],
                 names=np.asarray([p.stem for _, _, p in it]))
    print(f"[ei] lmk {s}: {len(res)} patch, ripieghi del rilevatore {int(dt['fallback'].sum())}, raggi a vuoto "
          f"{int(sum(r[2].sum() for r in res))}, {time.time() - t0:.0f}s", flush=True)
    scalars(s)                                            # centroid size delle patch: ICP cs, ULRSSM cs, GW cs
    if s == "eval":
        check_backproject()


def check_backproject(n: int = 8) -> None:
    """Retroproiezione e render contro quelli della matrice: le prime n mesh di HIFI3D della condizione det."""
    import pr_common as pc
    import pr_noise as prn
    from PIL import Image
    v = pc.view("hifi3d")
    d = prn.NWORK / "det" / "hifi3d"
    with np.load(d / "cams.npz") as z, np.load(d / "lmk2d.npz") as y, \
            np.load(prn.prep_path("hifi3d", "det"), allow_pickle=True) as q:
        o, f, P, ref = z["o"], z["f"], y["P"], q["L68"]
    got, px = [], 0
    for k in range(n):
        V, F, _ = pc.mm_mesh(v, v["items"][k][2])
        got.append(backproject(V, F, {"o": o[k], "f": float(f[k])}, P[k])[0])
        img = prn.render_mesh(V, F)[0]
        px = max(px, int((img != np.asarray(Image.open(d / f"{k:04d}.png"))).any(-1).sum()))
    ck = {"backproject_vs_pr_noise_hifi3d_mm": maxdiff(np.stack(got), ref[:n]), "render_px_diversi": px,
          "tol": {k: TOL[k] for k in ("backproject_mm", "render_px")}}
    ck["ok"] = bool(ck["backproject_vs_pr_noise_hifi3d_mm"]["max_abs"] <= TOL["backproject_mm"]
                    and px <= TOL["render_px"])
    atomic_json(EVID / "controls" / "backproject.json", {**ck, **stamp()})
    print(f"[ei] controllo retroproiezione e render: {json.dumps(ck)}", flush=True)


def prep_report(s: str) -> None:
    """Riassunto della preparazione (rilevatore, ritaglio): solo controlli di qualita', nessuna distanza."""
    rows = list(csv.DictReader(open(DATA / s / "patch_rows.csv")))
    err = json.loads((DATA / s / "patch_errors.json").read_text())["errors"]
    n = len(items(s))

    def st(key, sel=lambda r: True):
        x = np.asarray([float(r[key]) for r in rows if sel(r)], float)
        if not len(x):
            return None
        return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "min": float(x.min()),
                "max": float(x.max())}

    out = {"set": s, "n_scans": n, "n_patches": len(rows), "errors": err,
           "raw_detector": {"fallback": sum(r["raw_det_fallback"] == "True" for r in rows),
                            "n_faces_not_1": sum(int(r["raw_n_faces"]) != 1 for r in rows),
                            "det_score": st("raw_det_score"), "rays_missed_total": sum(int(r["raw_rays_missed"])
                                                                                       for r in rows)},
           "lmk7_residual_to_template_mm": st("lmk_residual_mm"), "scale_to_mm": st("scale_to_mm"),
           "crop_radius_mm": st("crop_radius_mm"), "n_vertices": st("n_vertices"), "n_faces": st("n_faces"),
           "n_folds": st("n_folds"), "outward_area": st("outward_area"),
           "meshes_reoriented_before_decimation": sum(int(r["n_faces_reoriented"]) > 0 for r in rows)}
    if (DATA / s / "lmk.npz").exists():
        with np.load(DATA / s / "lmk.npz") as z:
            out["patch_detector"] = {"fallback": int(z["fallback"].sum()), "n_faces_not_1": int((z["n_faces"] != 1).sum()),
                                     "det_score_median": float(np.median(z["det_score"])),
                                     "det_score_min": float(z["det_score"].min()), "rays_missed_total": int(z["miss"].sum())}
    atomic_json(EVID / f"prep_{s}.json", {**out, **stamp()})
    print(json.dumps(out, indent=1), flush=True)


# ------------------------------------------------------------------------------------------ nostro (GPU)

def embed(a) -> None:
    """Latenti [s, u] di un checkpoint (dentro ``eval_v3.py``: agganci del modello e della tabella di scala) sulle patch
    del set e, come controllo, sulle 15 scansioni di galleria di FaMoS TEST contro i latenti della matrice."""
    import torch
    from types import SimpleNamespace
    sys.path.insert(0, str(AAU / "recon"))
    from ws3b_latent import build_v1_model, encode
    if not torch.cuda.is_available():
        raise SystemExit("embed: serve la GPU")
    if Path(a.checkpoint).resolve() != CKPT[a.model].resolve():
        raise SystemExit(f"embed: checkpoint diverso da quello di {a.model}")
    device = torch.device("cuda")
    model = build_v1_model(Path(a.checkpoint), "", device)
    out_root = DATA / a.set / "emb"
    names = [p.stem for _, _, p in items(a.set)]
    z = encode(model, a.set, DATA / a.set / "ops", names,
               device, SimpleNamespace(out_root=out_root, metric=f"latent_{a.model}", overwrite=True))
    Z = np.stack([z[n] for n in names]).astype(np.float64)
    if Z.shape[1] != 257:
        raise SystemExit("servono i latenti [s, u]: WBES_V3_FACTORIZED_OUT=full")
    ck = {"gpu": torch.cuda.get_device_name(0), "Z_finite": bool(np.isfinite(Z).all())}
    gal = [r["name"] for r in fc_manifest() if r["kind"] == "scan" and r["role"] == "gallery"]
    zc = encode(model, "check", DATA / "check" / "ops", gal, device,
                SimpleNamespace(out_root=out_root, metric=f"latent_{a.model}", overwrite=True))
    with np.load(fc.OUT_ROOT / "eval" / "embeddings" / f"latent_{EMB_REF[a.model]}_test_view.npz") as r:
        ck["Z_galleria_famos_vs_matrice"] = maxdiff(np.stack([zc[n] for n in gal]), np.stack([r[n] for n in gal]))
    ck["tol_Z"] = TOL["Z"]
    ck["ok"] = bool(ck["Z_galleria_famos_vs_matrice"]["max_abs"] <= TOL["Z"])
    atomic_savez(out_root / f"{a.model}.npz", Z=Z, names=np.asarray(names))
    atomic_json(EVID / "controls" / f"embed_{a.model}.json", {**ck, "checkpoint": str(CKPT[a.model].relative_to(REPO)),
                                                              **stamp()})
    print(f"[ei] embed {a.model}: {Z.shape}, controllo {json.dumps(ck)}", flush=True)


def fc_manifest() -> list[dict]:
    with open(fc.VIEW_DIR / "manifest.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def check_links() -> None:
    """``<DATA>/check/npz``: link alle 15 scansioni di galleria di FaMoS TEST (operatori del controllo dell'embedding)."""
    d = DATA / "check" / "npz"
    d.mkdir(parents=True, exist_ok=True)
    for r in fc_manifest():
        if r["kind"] == "scan" and r["role"] == "gallery":
            p = d / f"{r['name']}.npz"
            if not p.exists():
                p.symlink_to(fc.VIEW_DIR / "npz" / f"{r['name']}.npz")
    print(f"[ei] check-links: {len(list(d.glob('*.npz')))} link in {d}", flush=True)


def ours(s: str) -> None:
    """d_F non cal., d_F cal. e d_P (``fact_paired.distances``) dai latenti di ``embed``."""
    sys.path.insert(0, str(REPO / "v3_work" / "trainer" / "tools"))
    import fact_paired as fp
    it = items(s)
    iu, ju = pairs(len(it))
    cal = fp.calibration()
    for m, ck in CKPT.items():
        with np.load(DATA / s / "emb" / f"{m}.npz") as z:
            if list(z["names"].astype(str)) != [p.stem for _, _, p in it]:
                raise SystemExit(f"{m}: latenti con mesh diverse")
            Z = np.asarray(z["Z"], np.float64)
        d = fp.distances(Z, iu, ju, ck, cal[m])
        save_dist(s, m, {k: d[k] for k in ("form", "form_cal", "shape")},
                  {"c": cal[m][0], "checkpoint": str(ck.relative_to(REPO)), "calibration": str(fp.CALIB.relative_to(REPO))})


# ------------------------------------------------------------------------------------------ ICP (blmm)

def inject_view(s: str) -> str:
    """Voce di ``blmm.VIEWS`` per il set (dominio ``famos``: L e CS_ref della matrice; frame dello spec, identita';
    scalari in ``<DATA>/<set>/scalars.npz``)."""
    import blmm
    sp = json.loads(spec_path(s).read_text())
    name = sp["name"]
    blmm.VIEWS[name] = {"dir": REPO / sp["dir"], "domain": "famos", "gt": None, "template_dir": None, "fb_root": None,
                        "template_ref": None, "out": DATA / s,
                        "spec": {"path": str(spec_path(s)), "subjects": sp["subjects"], "template_subjects": [],
                                 "frame": sp["frame"], "ref": "evaluated", "neutral": False}}
    scalars(s)
    return name


def icp(s: str, workers: int, chunk: int = 16) -> None:
    import blmm
    import blmm_pairs as bpp
    vname = inject_view(s)
    prm = blmm.params()
    it = items(s)
    names = [p.stem for _, _, p in it]
    iu, ju = pairs(len(it))
    out, ck, nerr = {}, {}, {}
    for mode in ("mm", "cs"):
        t0 = time.time()
        bpp._C.clear()
        for _, _, p in it:
            bpp._C[p.stem] = blmm.work_coords(vname, mode, p, prm)
        task = [(names[i], names[j], blmm.mesh_seed(names[i], names[j])) for i, j in zip(iu, ju)]
        with mp.get_context("fork").Pool(workers) as pool:
            vals, sec, errors = bpp.run_items(pool, task, mode, "fast", chunk)
        out[mode] = vals["rigid_icp_chamfer"] * blmm.unit(vname, mode, prm)
        nerr[mode] = {"n": len(errors), "first": errors[:5], "s_per_pair_median": float(np.median(sec))}
        ck[f"famos_galleria_{mode}"] = check_icp(mode, prm)
        print(f"[ei] icp {mode}: {len(task)} coppie in {time.time() - t0:.0f}s, fallite {len(errors)}", flush=True)
    ok = all(c["max_abs"] <= TOL["icp_mm"] for c in ck.values())
    save_dist(s, "icp", out, {"errors": nerr, "checks": ck, "tol_mm": TOL["icp_mm"], "ok": ok,
                              "L_mm": prm["domains"]["famos"]["L"], "cs_ref_mm": prm["domains"]["famos"]["cs_ref"]})


def check_icp(mode: str, prm: dict, n: int = 12) -> dict:
    """``rigid_icp_chamfer`` delle prime n righe scan peak di FaMoS TEST contro le 15 scansioni di galleria, rifatte
    con la regola di blmm_pairs.famos, contro ``baselines_mm/famos/<modo>/fast_0of1.npz``."""
    import blmm
    import blmm_pairs as bpp
    rows, gal = bpp.famos_gallery()
    names = [r["name"] for r in rows]
    q = [k for k, r in enumerate(rows) if r["kind"] == "scan" and r["role"] == "peak"][:n]
    gs = [g for g in gal if rows[g]["kind"] == "scan"]
    C = {names[k]: blmm.work_coords("famos", mode, blmm.VIEWS["famos"]["dir"] / f"{names[k]}.npz", prm)
         for k in set(q) | set(gs)}
    got = [blmm.pair_metrics(C[names[a]], C[names[g]], blmm.mesh_seed(names[a], names[g]), mode, "fast")
           ["rigid_icp_chamfer"] * blmm.unit("famos", mode, prm) for a in q for g in gs]
    with np.load(blmm.OUT_ROOT / "famos" / mode / "fast_0of1.npz") as z:
        col = {str(x): c for c, x in enumerate(z["gallery"])}
        pos = {int(r): k for k, r in enumerate(z["rows"])}
        ref = [z["rigid_icp_chamfer"][pos[a], col[names[g]]] for a in q for g in gs]
    return maxdiff(got, ref)


# ------------------------------------------------------------------------------------------ NoW (protocolli)

def now_data(v: dict, L68: np.ndarray) -> dict:
    """Mesh in mm e ritagli di NoW dai landmark dati (i passi di ``pr_run.load`` per i due NoW, ``--now-opt`` 0)."""
    import pr_common as pc
    import pr_metrics as pm
    import pr_run as prr
    mesh, crop = [], []
    for k, (_, _, p) in enumerate(v["items"]):
        V, F, _ = pc.mm_mesh(v, p)
        mesh.append((V, F))
        crop.append(pm.crop(V, L68[k][list(pc.LMK7)], prr.NOW_R))
    return {"mesh": mesh, "L7": L68[:, list(pc.LMK7)], "now_crop": crop, "now_opt": [None] * len(mesh)}


def _now_task(a: int) -> tuple:
    import pr_metrics as pm
    import pr_run as prr
    D, bs = _ST["D"], _ST["partners"][a]
    V, F = D["mesh"][a]
    surf = pm.Surf(V, F)
    return a, bs, [prr.one_pair(surf, a, int(b), ("now_original", "now_metrical"), D) for b in bs]


def now_ordered(D: dict, partners: list, workers: int) -> dict:
    """{(a, b): riga di ``pr_run.one_pair``} per tutte le coppie ordinate richieste."""
    _ST.update(D=D, partners=partners)
    out = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for a, bs, rows in pool.imap_unordered(_now_task, [a for a in range(len(partners)) if len(partners[a])]):
            for b, r in zip(bs, rows):
                out[(a, int(b))] = r
    return out


def now(s: str, workers: int) -> None:
    import pr_common as pc
    t0 = time.time()
    v = pc.view(VNAME[s], spec_path(s))
    if [(x, t) for x, t, _ in v["items"]] != [(x, t) for x, t, _ in items(s)]:
        raise SystemExit("now: vista diversa dalla selezione")
    with np.load(DATA / s / "lmk.npz") as z:
        L68 = np.asarray(z["L68"], np.float64)
    D = now_data(v, L68)
    n = len(v["items"])
    iu, ju = pairs(n)
    E = now_ordered(D, [np.asarray([b for b in range(n) if b != a]) for a in range(n)], workers)
    out, err = {}, {}
    for m in ("now_original", "now_metrical"):
        e1 = np.asarray([E[(a, b)][m] for a, b in zip(iu, ju)])
        e2 = np.asarray([E[(b, a)][m] for a, b in zip(iu, ju)])
        out[f"{m}_mm"] = (e1 + e2) / 2
        msgs = [r[f"{m}|error"] for r in E.values() if r[f"{m}|error"]]
        err[m] = {"ordered_failed": len(msgs), "first": msgs[:5],
                  "reflections": int(sum(r.get(f"{m}|reflection", 0) for r in E.values()))}
    ck = check_now(workers)
    ok = all(c["max_abs"] <= TOL["now_mm"] for c in ck.values())
    for m in ("now_original", "now_metrical"):
        save_dist(s, m, {"mm": out[f"{m}_mm"]}, {"errors": err[m], "checks": ck, "tol_mm": TOL["now_mm"], "ok": ok,
                                                 "landmarks": "rilevatore sulla patch (lmk.npz)",
                                                 "seconds_total": time.time() - t0})


def check_now(workers: int, n_pairs: int = 12) -> dict:
    """Le prime coppie delle 15 scansioni di galleria di FaMoS TEST coi landmark esatti della prep della matrice,
    ricostruite con ``now_data`` (media dei due versi), contro ``competitors_new/<metodo>/famos.npz``."""
    import pr_common as pc
    v = pc.view("famos")
    pr = pc.read_prep("famos")
    if list(zip(pr["subjects"].astype(str), pr["topologies"].astype(str))) != [(x, t) for x, t, _ in v["items"]]:
        raise SystemExit("check_now: prep della matrice con mesh diverse dalla vista famos")
    D = now_data(v, np.asarray(pr["L68"], np.float64))
    i, j = pc.pairs(v["items"])
    i, j = i[:n_pairs], j[:n_pairs]
    partners = [[] for _ in v["items"]]
    for a, b in zip(i, j):
        partners[a].append(b)
        partners[b].append(a)
    E = now_ordered(D, [np.asarray(sorted(p), np.int64) for p in partners], workers)
    ck = {}
    for m in ("now_original", "now_metrical"):
        got = [(E[(a, b)][m] + E[(b, a)][m]) / 2 for a, b in zip(i, j)]
        with np.load(pc.PLUGIN_ROOT / m / "famos.npz") as z:
            ref = [z["D_mm"][a, b] for a, b in zip(i, j)]
        ck[f"{m}_famos_galleria"] = maxdiff(got, ref)
    return ck


# ------------------------------------------------------------------------------------------ ULRSSM (GPU)

def ulr_feats(v: dict, cache: Path, net, device, workers: int) -> tuple[dict, dict]:
    import ulr_eval as UE
    cache.mkdir(parents=True, exist_ok=True)              # get_operators vuole la cartella esistente
    UE._P.update(view=v, cache=str(cache))
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(UE._prep, range(len(v["items"])), chunksize=1)
    fail = {k: r for k, r in res if not isinstance(r, dict)}
    G = {k: UE.on_gpu(net, r, str(cache), v["cs_ref"], device) for k, r in res if isinstance(r, dict)}
    return G, fail


def ulrssm(s: str, workers: int) -> None:
    import torch
    sys.path.insert(0, str(ULR))
    import ulr_eval as UE
    U = UE.U
    t0 = time.time()
    device = torch.device("cuda")
    cache = DATA / s / "ulr_cache"
    cache.mkdir(parents=True, exist_ok=True)
    net, input_type = U.load_net(ULR_CKPT, device, str(cache))
    v = U.view(VNAME[s], spec_path(s))
    v["scalars"] = scalars(s)
    if [(x, t) for x, t, _ in v["items"]] != [(x, t) for x, t, _ in items(s)]:
        raise SystemExit("ulrssm: vista diversa dalla selezione")
    G, fail = ulr_feats(v, cache, net, device, workers)
    iu, ju = pairs(len(v["items"]))
    fr, sr = np.full(len(iu), np.nan), np.full(len(iu), np.nan)
    for q, (a, b) in enumerate(zip(iu, ju)):
        if a in G and b in G:
            fr[q], sr[q] = U.pair_rms(G[a], G[b])[:2]
    # controllo: le 105 coppie delle 15 scansioni di galleria di FaMoS TEST contro il plug-in faces_xyz
    vf = U.view("famos")
    Gf, ff = ulr_feats(vf, DATA / "check" / "ulr_cache", net, device, workers)
    if ff:
        raise SystemExit(f"ulrssm: controllo, mesh di FaMoS TEST fallite: {ff}")
    i, j = U.pairs(vf["items"])
    got = np.asarray([U.pair_rms(Gf[a], Gf[b])[:2] for a, b in zip(i, j)])
    with np.load(U.OUT / "faces_xyz" / "famos.npz") as z:
        ck = {"pair_fr_famos_galleria": maxdiff(got[:, 0], z["D_pair_fr"][i, j]),
              "pair_sr_famos_galleria": maxdiff(got[:, 1], z["D_pair_sr"][i, j])}
    ok = all(c["max_abs"] <= TOL["ulrssm_mm"] for c in ck.values())
    save_dist(s, "ulrssm", {"pair_fr": fr, "pair_sr": sr},
              {"failed_meshes": {str(k): x for k, x in fail.items()}, "input_type": input_type,
               "checkpoint": str(ULR_CKPT.relative_to(REPO)), "gpu": torch.cuda.get_device_name(0),
               "checks": ck, "tol_mm": TOL["ulrssm_mm"], "ok": ok, "seconds": time.time() - t0})


# ------------------------------------------------------------------------------------------ GW

def gw(s: str, workers: int) -> None:
    sys.path.insert(0, str(GP))
    import blmm
    import gp_gw as G
    t0 = time.time()
    choice = json.loads((G.GW / "pilot_choice.json").read_text())["config"]
    if choice != GW_CFG:
        raise SystemExit(f"gw: configurazione dichiarata {choice}, attesa {GW_CFG}")
    G.WORK = DATA / s / "gw"                              # misure del set fuori dalle cartelle della matrice
    vname = VNAME[s]
    G.run_prep(vname, spec_path(s), workers, False)
    P = G.load_prep(vname)
    if [(str(x), str(t)) for x, t in zip(P["subjects"], P["topologies"])] != [(x, t) for x, t, _ in items(s)]:
        raise SystemExit("gw: misure con mesh diverse dalla selezione")
    sc = scalars(s)
    cs_mesh = np.asarray([sc[blmm.rel(p)]["cs"] for _, _, p in items(s)])
    iu, ju = pairs(len(P["subjects"]))
    Dv, sec = G.solve_pairs(P, iu, ju, [GW_CFG], workers)
    # controllo: coppie delle 15 scansioni di galleria di FaMoS TEST con le misure della matrice
    G.WORK = G.gc.WORK
    Pf = G.load_prep("famos")
    i, j = G.gc.pairs(G.items_of(Pf))
    Df, _ = G.solve_pairs(Pf, i[:12], j[:12], [GW_CFG], workers)
    with np.load(G.gc.out_npz("gw", G.tag(GW_CFG), "famos")) as z:
        ck = {f"{m}_famos_galleria": maxdiff(Df[:, 0, k], z[f"D_{m}"][i[:12], j[:12]]) for k, m in enumerate(G.gc.MODES)}
    ok = all(c["max_abs"] <= TOL["gw_mm"] for c in ck.values())
    save_dist(s, "gw", {m: Dv[:, 0, k] for k, m in enumerate(G.gc.MODES)},
              {"config": GW_CFG, "nan": int(np.isnan(Dv).sum()), "s_per_solve_median": float(np.nanmedian(sec)),
               "cs_vs_scalars_max_abs": float(np.nanmax(np.abs(P["cs"] - cs_mesh))), "checks": ck,
               "tol_mm": TOL["gw_mm"], "ok": ok, "seconds": time.time() - t0})


# ------------------------------------------------------------------------------------------ metriche

def retrieval(D: np.ndarray, subj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per query: rank-1 (pari al minimo: quota di stessa persona fra i pari) e AP (galleria = tutte le altre
    scansioni, ordine per distanza, pari per indice di galleria)."""
    n = len(D)
    r1, ap = np.empty(n), np.empty(n)
    for q in range(n):
        g = np.delete(np.arange(n), q)
        d, gen = D[q, g], subj[g] == subj[q]
        r1[q] = gen[d == d.min()].mean()
        rel = gen[np.lexsort((g, d))]
        prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
        ap[q] = prec[rel].mean()
    return r1, ap


def eer(d: np.ndarray, gen: np.ndarray, w: np.ndarray) -> float:
    """EER pesato: accettata se d <= t; FAR(t) e FRR(t) sulle soglie distinte (piu' il punto t = -inf), interpolazione
    lineare fra i due punti operativi dove FAR - FRR cambia segno."""
    g, i = gen & (w > 0), ~gen & (w > 0)
    t = np.unique(d[g | i])
    rates = []
    for m in (i, g):
        o = np.argsort(d[m], kind="stable")
        cw = np.concatenate([[0.0], np.cumsum(w[m][o])])
        rates.append(cw[np.searchsorted(d[m][o], t, side="right")] / cw[-1])
    far = np.concatenate([[0.0], rates[0]])
    frr = np.concatenate([[1.0], 1.0 - rates[1]])
    diff = far - frr
    k = int(np.argmax(diff >= 0))
    a = -diff[k - 1] / (diff[k] - diff[k - 1])
    return float(far[k - 1] + a * (far[k] - far[k - 1]))


def load_columns(s: str) -> tuple[dict, list, dict]:
    """{colonna: distanze sulle coppie} (non finite -> +inf), elementi, fallimenti per colonna."""
    it = items(s)
    names = [p.stem for _, _, p in it]
    cols, failed = {}, {}
    src = {**{f"{m}|{r}": (m, r) for m in CKPT for r in ("form", "form_cal", "shape")},
           "icp|mm": ("icp", "mm"), "icp|cs": ("icp", "cs"), "now_metrical|mm": ("now_metrical", "mm"),
           "now_original|mm": ("now_original", "mm"), "ulrssm|pair_fr": ("ulrssm", "pair_fr"),
           "ulrssm|pair_sr": ("ulrssm", "pair_sr"), "gw|fr": ("gw", "fr"), "gw|sr": ("gw", "sr")}
    iu, ju = pairs(len(it))
    for key, _, _, _ in COLUMNS:
        m, r = src[key]
        with np.load(dist_path(s, m)) as z:
            if list(z["names"].astype(str)) != names or not (np.array_equal(z["i"], iu) and np.array_equal(z["j"], ju)):
                raise SystemExit(f"{key}: coppie diverse")
            x = np.asarray(z[f"D_{r}"], np.float64)
        failed[key] = int((~np.isfinite(x)).sum())
        cols[key] = np.where(np.isfinite(x), x, np.inf)
    return cols, it, failed


def evaluate(s: str) -> None:
    import famos_eval as fe
    t0 = time.time()
    cols, it, failed = load_columns(s)
    subj_names = sorted({x for x, _, _ in it})
    subj = np.asarray([subj_names.index(x) for x, _, _ in it])
    n, ns = len(it), len(subj_names)
    iu, ju = pairs(n)
    gen = subj[iu] == subj[ju]
    counts = np.concatenate([np.ones((1, ns), np.int64), fe.bootstrap_counts(ns, N_BOOT, BOOT_SEED)])
    R = {}                                                 # colonna -> metrica -> (1 + N_BOOT,)
    for key, *_ in COLUMNS:
        D = np.zeros((n, n))
        D[iu, ju] = D[ju, iu] = cols[key]
        r1, ap = retrieval(D, subj)
        wq = counts[:, subj].astype(np.float64)            # query ricampionate per soggetto, galleria fissa
        R[key] = {"rank1": (wq @ r1) / wq.sum(1), "mAP": (wq @ ap) / wq.sum(1),
                  "AUC": np.empty(len(counts)), "EER": np.empty(len(counts))}
        for b, c in enumerate(counts):
            w = np.where(gen, c[subj[iu]], c[subj[iu]] * c[subj[ju]]).astype(np.float64)
            R[key]["AUC"][b] = fe.weighted_auc(-cols[key], gen, w)
            R[key]["EER"][b] = eer(cols[key], gen, w)
        print(f"[ei] eval {key}: rank-1 {R[key]['rank1'][0]:.3f}, mAP {R[key]['mAP'][0]:.3f}, "
              f"EER {R[key]['EER'][0]:.3f}, AUC {R[key]['AUC'][0]:.3f}", flush=True)

    def ci(x):
        return tuple(float(v) for v in np.percentile(x[1:], [2.5, 97.5]))

    mrows = []
    for key, label, fam, is_ours in COLUMNS:
        for met in METRICS:
            lo, hi = ci(R[key][met])
            mrows.append({"colonna": key, "metodo": label, "famiglia": fam, "nostro": is_ours, "metrica": met,
                          "valore": float(R[key][met][0]), "ic_lo": lo, "ic_hi": hi, "n_scansioni": n,
                          "n_soggetti": ns, "n_coppie": len(iu), "n_coppie_stessa_persona": int(gen.sum()),
                          "coppie_non_finite": failed[key], "repliche": N_BOOT, "seme": BOOT_SEED})
    drows = []
    for ko, lo_, _, _ in OURS:
        for kc, lc, _, _ in COMPETITORS:
            for met in METRICS:
                sgn = -1.0 if met == "EER" else 1.0
                d = sgn * (R[ko][met] - R[kc][met])
                lo, hi = ci(d)
                drows.append({"nostro": ko, "nostro_etichetta": lo_, "concorrente": kc, "concorrente_etichetta": lc,
                              "metrica": met, "delta": float(d[0]), "ic_lo": lo, "ic_hi": hi,
                              "verso": "positivo = il nostro meglio" + (" (EER: concorrente - nostro)" if sgn < 0 else
                                                                        " (nostro - concorrente)"),
                              "repliche": N_BOOT, "seme": BOOT_SEED})
    EVID.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("metrics.csv", mrows), ("deltas.csv", drows)):
        with open(EVID / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    ctrl = {p.stem: json.loads(p.read_text()) for p in sorted((EVID / "controls").glob("*.json"))}
    summary = {"set": s, "n_scans": n, "n_subjects": ns, "n_pairs": len(iu), "n_genuine": int(gen.sum()),
               "n_impostor": int((~gen).sum()), "coppie_non_finite": failed,
               "controls_ok": {k: c.get("ok") for k, c in ctrl.items()}, "seconds": time.time() - t0, **stamp()}
    atomic_json(EVID / "summary.json", summary)
    write_results(mrows, drows, summary, ctrl)
    print(f"[ei] eval: {n} scansioni, {ns} soggetti, {len(iu)} coppie ({int(gen.sum())} stessa persona), "
          f"{time.time() - t0:.0f}s", flush=True)


def write_results(mrows: list, drows: list, summary: dict, ctrl: dict) -> None:
    sel = json.loads((EVID / "selection.json").read_text())
    prep = json.loads((EVID / "prep_eval.json").read_text())

    def cell(key, met):
        r = next(x for x in mrows if x["colonna"] == key and x["metrica"] == met)
        return f"{r['valore']:.3f} [{r['ic_lo']:.3f}, {r['ic_hi']:.3f}]"

    def dcell(ko, kc, met):
        r = next(x for x in drows if x["nostro"] == ko and x["concorrente"] == kc and x["metrica"] == met)
        return f"{r['delta']:+.3f} [{r['ic_lo']:+.3f}, {r['ic_hi']:+.3f}]"

    L = ["# Validazione esterna su FaMoS: riconoscimento dell'identita' su scansioni grezze", "",
         "Generato da `aau/esterna_famos_id/ei.py eval` (regole in `PROTOCOLLO.md`, scritte e committate prima dei "
         "numeri). Lettura DESCRITTIVA: nessun test confermativo; i risultati sfavorevoli sono riportati come gli altri.",
         "", "## Dati", "",
         f"- {summary['n_subjects']} soggetti di FaMoS TEST, {summary['n_scans']} scansioni grezze "
         f"({sel['k_seq']} sequenze per soggetto, le stesse per tutti: {', '.join(sel['eval']['sequences'])}), "
         f"{summary['n_pairs']} coppie ({summary['n_genuine']} stessa persona, {summary['n_impostor']} persone "
         "diverse). Nessuna registrazione FLAME: topologie tutte diverse.",
         f"- Preparazione comune (rilevatore su render frontale, patch NoW in mm): {prep['n_patches']}/{prep['n_scans']} "
         f"patch, ripieghi del rilevatore sulla grezza {prep['raw_detector']['fallback']}, sulla patch "
         f"{prep.get('patch_detector', {}).get('fallback', '-')}; dettagli in `prep_eval.json`.",
         f"- IC 95% percentile, {N_BOOT} repliche per soggetto (seme {BOOT_SEED}, `famos_eval.bootstrap_counts`); "
         "retrieval: query ricampionate per soggetto, galleria fissa; verifica: pesi c_s per le coppie della stessa "
         "persona, c_a c_b per le altre.", "",
         "## Metriche per metodo", "", "valore [IC 95%]. rank-1, mAP, AUC: piu' alto = meglio; EER: piu' basso = meglio.",
         "", "| metodo | rank-1 | mAP | EER | AUC | coppie non finite |", "|---|---|---|---|---|---|"]
    for key, label, _, _ in COLUMNS:
        L.append(f"| {label} | {cell(key, 'rank1')} | {cell(key, 'mAP')} | {cell(key, 'EER')} | {cell(key, 'AUC')} | "
                 f"{summary['coppie_non_finite'][key]} |")
    for ko, lo_, _, _ in OURS:
        L += ["", f"## Delta appaiati: {lo_} contro ciascun concorrente", "",
              "positivo = il nostro meglio (rank-1, mAP, AUC: nostro - concorrente; EER: concorrente - nostro); stesse "
              "repliche.", "", "| concorrente | rank-1 | mAP | EER | AUC |", "|---|---|---|---|---|"]
        for kc, lc, _, _ in COMPETITORS:
            L.append(f"| {lc} | {dcell(ko, kc, 'rank1')} | {dcell(ko, kc, 'mAP')} | {dcell(ko, kc, 'EER')} | "
                     f"{dcell(ko, kc, 'AUC')} |")
    L += ["", "## Controlli di riproduzione (contro la matrice, FaMoS TEST galleria)", ""]
    for k, c in ctrl.items():
        keep = {x: c[x] for x in ("ok", "checks", "Z_galleria_famos_vs_matrice", "backproject_vs_pr_noise_hifi3d_mm",
                                  "render_px_diversi") if x in c}
        L.append(f"- `{k}`: `{json.dumps(keep)}`")
    L += ["", "File: `metrics.csv` (una riga per metodo e metrica), `deltas.csv` (tutti i delta appaiati), "
          "`summary.json`, `selection.json`, `prep_eval.json`, `prep_pilot.json`, `controls/`.", ""]
    out = EVID / "results.md"
    tail = out.read_text().split(MARK, 1)[1] if out.exists() and MARK in out.read_text() else "\n"
    out.write_text("\n".join(L) + "\n" + MARK + tail)


# ------------------------------------------------------------------------------------------ main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("select", "render-raw", "patch", "render-patch", "lmk", "prep-report", "check-links",
                                    "embed", "ours", "icp", "now", "ulrssm", "gw", "eval"))
    p.add_argument("--set", default="eval", choices=SETS)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--model", default="", choices=("",) + tuple(CKPT))
    a = p.parse_args()
    if a.step == "select":
        select(a)
    elif a.step in ("render-raw", "render-patch"):
        render(a.set, a.step.split("-")[1], a.workers)
    elif a.step == "patch":
        patch(a.set, a.workers)
    elif a.step == "lmk":
        lmk(a.set, a.workers)
    elif a.step == "prep-report":
        prep_report(a.set)
    elif a.step == "check-links":
        check_links()
    elif a.step == "embed":
        embed(a)
    elif a.step == "ours":
        ours(a.set)
    elif a.step == "icp":
        icp(a.set, a.workers)
    elif a.step == "now":
        now(a.set, a.workers)
    elif a.step == "ulrssm":
        ulrssm(a.set, a.workers)
    elif a.step == "gw":
        gw(a.set, a.workers)
    elif a.step == "eval":
        evaluate(a.set)


if __name__ == "__main__":
    main()
