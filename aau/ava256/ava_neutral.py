#!/usr/bin/env python3
"""Template di Ava-256 per la corrispondenza, neutre per cattura e controlli dei frame.

    v3_work/unified_gt/run.sh aau/ava256/ava_neutral.py template --workers 32     (prima di ava_corr.py)
    v3_work/unified_gt/run.sh aau/ava256/ava_neutral.py neutral --workers 32      (dopo ava_corr.py)

CONFERMATIVO: non valutare prima del protocollo confermativo (README). Solo dati e GT.

``template``: per cattura la media dei frame di EXP_neutral_peak, ciascuno allineato rigidamente (Umeyama senza
scala sui vertici usati dalla topologia) al primo; poi Procrustes generalizzato di similarita' fra le catture nel
frame della prima, come ``domains.multiface``. Serve SOLO alla corrispondenza (ava_corr.py):
``datasets/AVA256/template.npz``. I 1565 vertici orfani (fuori dalla superficie) seguono le trasformazioni ma non
entrano nei fit.

``neutral``: per cattura (README, scelte 1-2) la neutra dei frame di EXP_neutral_peak e il neutro ripetuto dei 16
frame di EXP_eye_neutral con la regola di FaMoS, ``famos_subsample.neutral_shape`` importata e non riscritta: le si
passa uno spazio unificato la cui ``map`` usa la mappa di Ava-256 (``AvaSpace``). Controlli per frame: vertici
finiti, triangoli degeneri (area < 1e-6 x la mediana del frame), triangoli girati rispetto al template dopo una
rigida, su tutta la superficie e sulla patch delle viste (``patch_tri`` di ava_corr.py). Una cattura e' valida con
almeno 3 frame finiti di EXP_neutral_peak (README, scelta 6; la convergenza della rigida robusta la controlla
ava_gt.py).

Uscite: ``datasets/AVA256/neutral/<ava_id>.npz`` (unita' native, float32), ``neutral/manifest.json`` e il riassunto
aggregato ``aau/ava256/neutral_summary.json`` (nessuna geometria).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
for _p in (THIS_DIR, THIS_DIR.parents[1] / "v3_work" / "unified_gt", THIS_DIR.parent / "famos"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ava_common as ac  # noqa: E402
import famos_subsample as fs  # noqa: E402  (neutral_shape: la regola della neutra di FaMoS)
import ugt as C  # noqa: E402
from shapes import Space  # noqa: E402

TEMPLATE = ac.DATA_ROOT / "template.npz"
CORR_NPZ = ac.DATA_ROOT / "corr" / "ava256.npz"
SUMMARY = ac.SUMMARY_DIR / "neutral_summary.json"
DEGENERATE_REL = 1e-6
MIN_FRAMES = 3
GPA_ITERS, GPA_TOL = 10, 1e-6           # domains.multiface

_ST: dict = {}


class AvaSpace:
    """``shapes.Space`` con ``map`` sulla mappa di Ava-256 per qualunque dominio chiesto: cosi'
    ``famos_subsample.neutral_shape``, che chiama ``sp.map("flame", V)``, gira invariata sulle mesh Ava-256."""

    def __init__(self, sp: Space, vidx: np.ndarray, bary: np.ndarray):
        self.sp, self.vidx, self.bary = sp, vidx, bary

    def map(self, d: str, V: np.ndarray) -> np.ndarray:
        return C.bary_interp(V, self.vidx, self.bary)

    def __getattr__(self, k):
        return getattr(self.sp, k)


def ava_space() -> AvaSpace:
    with np.load(CORR_NPZ) as z:
        return AvaSpace(Space(), z["vidx_unified"], z["bary_unified"])


# ------------------------------------------------------------------------------ template

def capture_mean(cap: str) -> np.ndarray:
    """Media dei frame di EXP_neutral_peak allineati rigidamente (vertici usati) al primo, (7306, 3)."""
    used = _ST["used"]
    Vs = []
    for _, p in ac.frames(cap, ac.NEUTRAL_SEG):
        V = ac.read_ply(p)
        if Vs:
            V = C.apply_sim(V, *C.umeyama(V[used], Vs[0][used], scale=False))
        Vs.append(V)
    return np.mean(Vs, axis=0) if Vs else np.full((ac.N_VERTS, 3), np.nan)


def stage_template(workers: int) -> None:
    t0 = time.time()
    F = ac.topology()
    used = np.unique(F)
    _ST["used"] = used
    caps = [c["capture"] for c in ac.captures()]
    with mp.get_context("fork").Pool(workers) as pool:
        per = np.stack(pool.map(capture_mean, caps, chunksize=4))
    ok = np.isfinite(per[:, used]).all((1, 2))
    per_ok = per[ok]
    mean = per_ok[0]
    for it in range(GPA_ITERS):
        aligned = [C.apply_sim(X, *C.umeyama(X[used], mean[used])) for X in per_ok]
        new = np.mean(aligned, axis=0)
        new = C.apply_sim(new, *C.umeyama(new[used], per_ok[0][used]))
        if np.abs(new - mean)[used].max() < GPA_TOL:
            break
        mean = new
    C.save_npz(TEMPLATE, V=mean, F=F, used=used, captures=np.asarray(caps)[ok], gpa_iterations=it + 1)
    ext = mean[used].max(0) - mean[used].min(0)
    print(f"[ava-tpl] template da {int(ok.sum())}/{len(caps)} catture, GPA {it + 1} giri, estensioni "
          f"{np.round(ext, 1).tolist()} (unita' native), {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------- neutre

def _init_neutral() -> None:
    _ST["asp"] = ava_space()
    _ST["F"] = ac.topology()
    _ST["used"] = np.unique(_ST["F"])
    with np.load(TEMPLATE) as z:
        _ST["tpl"] = z["V"]
    with np.load(CORR_NPZ) as z:
        _ST["patch_tri"] = z["patch_tri"]
    _ST["tpl_n"] = C.face_normals(_ST["tpl"], _ST["F"])


def frame_qc(V: np.ndarray) -> dict:
    """Per frame: triangoli degeneri e girati (tutta la superficie e patch delle viste) rispetto al template."""
    F, used, tpl = _ST["F"], _ST["used"], _ST["tpl"]
    n_deg, flip_all, flip_patch = [], [], []
    for X in V:
        a = C.face_areas(X, F)
        n_deg.append(int((a < DEGENERATE_REL * np.median(a)).sum()))
        Xa = C.apply_sim(X, *C.umeyama(X[used], tpl[used], scale=False))
        flipped = np.einsum("nd,nd->n", C.face_normals(Xa, F), _ST["tpl_n"]) < 0
        flip_all.append(int(flipped.sum()))
        flip_patch.append(int(flipped[_ST["patch_tri"]].sum()))
    return {"n_degenerate": np.asarray(n_deg), "n_flipped": np.asarray(flip_all),
            "n_flipped_patch": np.asarray(flip_patch)}


def process(task: dict) -> dict:
    cap, ava_id = task["capture"], task["ava_id"]
    out = {"capture": cap, "ava_id": ava_id, "view_id": task["view_id"]}
    arrays = {"capture": np.asarray(cap), "ava_id": np.asarray(ava_id)}
    for tag, seg in (("neutral", ac.NEUTRAL_SEG), ("repeat", ac.REPEAT_SEG)):
        fr = ac.frames(cap, seg)
        frames = np.asarray([f for f, _ in fr], dtype=np.int64)
        V = np.stack([ac.read_ply(p) for _, p in fr]) if fr else np.zeros((0, ac.N_VERTS, 3))
        finite = np.isfinite(V).all((1, 2))
        qc = frame_qc(np.where(finite[:, None, None], V, 0.0))
        rec = {"n_frames": int(len(frames)), "n_finite": int(finite.sum()),
               "max_degenerate": int(qc["n_degenerate"].max(initial=0)),
               "frames_with_flipped_patch": int((qc["n_flipped_patch"] > 0).sum()),
               "max_flipped_patch": int(qc["n_flipped_patch"].max(initial=0)),
               "max_flipped": int(qc["n_flipped"].max(initial=0))}
        arrays.update({f"{tag}_frames": frames, f"{tag}_finite": finite,
                       **{f"{tag}_{k}": v for k, v in qc.items()}})
        if finite.sum() >= MIN_FRAMES:
            nt = fs.neutral_shape(_ST["asp"], V[finite])
            fin = frames[finite]
            arrays.update({f"V_{tag}": nt["V"].astype(np.float32), f"{tag}_kept": fin[nt["keep"]],
                           f"{tag}_medoid": np.int64(fin[nt["medoid"]]),
                           f"{tag}_dist_to_medoid_mm": nt["dist"].astype(np.float32),
                           f"{tag}_threshold_mm": np.float32(nt["threshold"]),
                           f"{tag}_halves_mm": np.float32(nt["halves"])})
            rec.update({"n_kept": int(len(nt["keep"])), "threshold_mm": float(nt["threshold"]),
                        "halves_mm": float(nt["halves"]), "ok": bool(np.isfinite(nt["V"]).all())})
        else:
            rec.update({"n_kept": 0, "threshold_mm": float("nan"), "halves_mm": float("nan"), "ok": False})
        out[tag] = rec
    out["valid"] = bool(out["neutral"]["ok"])
    C.save_npz(ac.NEUTRAL_DIR / f"{ava_id}.npz", valid=np.bool_(out["valid"]), **arrays)
    return out


def q(x) -> dict:
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "min": float(x.min()), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def stage_neutral(workers: int) -> None:
    t0 = time.time()
    ac.NEUTRAL_DIR.mkdir(parents=True, exist_ok=True)
    tasks = ac.captures()
    rows = []
    with mp.get_context("fork").Pool(workers, initializer=_init_neutral) as pool:
        for r in pool.imap_unordered(process, tasks, chunksize=2):
            rows.append(r)
            if len(rows) % 32 == 0 or len(rows) == len(tasks):
                print(f"[ava-neutral] {len(rows)}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["ava_id"])
    C.save_json(ac.NEUTRAL_DIR / "manifest.json", {"rule": __doc__.split("``neutral``:")[1].split("Uscite")[0].strip(),
                                                   "captures": rows})
    summ = {"n_captures": len(rows), "n_valid": int(sum(r["valid"] for r in rows)),
            "invalid": [r["ava_id"] for r in rows if not r["valid"]],
            "repeat_missing": [r["ava_id"] for r in rows if not r["repeat"]["ok"]]}
    for tag in ("neutral", "repeat"):
        R = [r[tag] for r in rows]
        summ[tag] = {"frames": q([x["n_frames"] for x in R]), "kept": q([x["n_kept"] for x in R]),
                     "kept_fraction": q([x["n_kept"] / x["n_frames"] for x in R if x["n_frames"]]),
                     "threshold_mm": q([x["threshold_mm"] for x in R]), "halves_mm": q([x["halves_mm"] for x in R]),
                     "frames_total": int(sum(x["n_frames"] for x in R)),
                     "frames_not_finite": int(sum(x["n_frames"] - x["n_finite"] for x in R)),
                     "captures_with_degenerate": int(sum(x["max_degenerate"] > 0 for x in R)),
                     "captures_with_flipped_patch": int(sum(x["max_flipped_patch"] > 0 for x in R)),
                     "frames_with_flipped_patch": int(sum(x["frames_with_flipped_patch"] for x in R)),
                     "max_flipped_patch": int(max(x["max_flipped_patch"] for x in R)),
                     "max_flipped_surface": int(max(x["max_flipped"] for x in R))}
    summ["note"] = ("distanze g = GT unificata (Procrustes di similarita' verso mu, mm alla scala di mu) fra frame "
                    "della regione unificata; halves = distanza g fra le medie dei frame tenuti pari e dispari; "
                    "'flipped' = triangoli con la normale a piu' di 90 gradi da quella del template dopo una rigida: "
                    "variazione fra persone in zone molto curve (palpebre, bordo basso della patch), NON pieghe della "
                    "mesh; le pieghe (patch_quality) sono in views_summary.json")
    C.save_json(SUMMARY, summ)
    print(json.dumps({k: v for k, v in summ.items() if k != "note"}, indent=1)[:3000], flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=("template", "neutral"))
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    if a.stage == "template":
        stage_template(a.workers)
    else:
        stage_neutral(a.workers)


if __name__ == "__main__":
    main()
