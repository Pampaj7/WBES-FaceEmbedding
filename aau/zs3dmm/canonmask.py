#!/usr/bin/env python3
"""Maschera canonica: regione per mesh attorno alla punta del naso, senza GT ne' corrispondenza.

    aau/run.sh aau/zs3dmm/canonmask.py calibrate --view-dir <npz> --crop-ref-dir <vista eqsupport> --out <csv>
    aau/run.sh aau/zs3dmm/canonmask.py build --view-dir <npz> --out-dir <ZS_ROOT>/canonmask_view \\
        --mode canonical|offcenter|identity --rho <rho> --n-cores 32
    (canonmask_view.sbatch; protocollo in aau/runs/canonmask/PROTOCOL.md)

La regola guarda UNA mesh alla volta (vertici e facce, nient'altro), piu' una costante di dominio:

  1. piano del viso: PCA dei baricentri dei triangoli pesati per area; ``S0`` = radice dell'area
     della proiezione sul piano, presa con segno (area racchiusa dal bordo: il rumore all'interno
     non la cambia, a differenza dell'area della superficie). Serve solo a dimensionare la ricerca;
  2. punta del naso: fra i vertici piu' alti sul piano (entrambi i versi della normale) e lontani
     dal bordo esterno (> 1.5 anelli), quello di massima prominenza LOCALE, cioe' altezza sul piano
     dell'anello a ``RING_FRAC x S0`` attorno a lui (mento e fronte sono alti sul piano globale ma
     non sporgono localmente); poi la punta e' il baricentro pesato per area della calotta
     (altezza sul piano dell'anello >= massimo - ``CAP_FRAC`` x prominenza), due iterazioni;
  3. regione: palla euclidea di raggio ``R = rho x S_dom`` attorno alla punta; facce col baricentro
     nella palla e i loro vertici (nessuna pulizia: vedi ``restrict``). ``S_dom`` e' una costante del dominio
     (normalizzazione dichiarata: mediana di S0 su tutte le mesh di tutte le condizioni dei soggetti
     di eval), cioe' lo stesso raggio, nelle unita' del dataset, per ogni mesh del dominio: e' la
     sfera attorno al naso di NoW (95 mm), con l'unita' sostituita dalla scala del dataset.

Perche' non una scala per mesh (``calibrate``, aau/runs/canonmask/calibration_*.csv): ``S0`` per
mesh e' 4-9% piu' piccola sul crop che sulla original (il crop toglie area di bordo), quindi la
maschera del crop sarebbe sistematicamente piu' piccola -- lo stesso difetto di supporto che si
vuole togliere; la scala "del naso" (r / altezza della punta sull'anello) non e' definita sul
30-90% delle mesh e varia di +-30% su noisy. ``calibrate`` misura anche quanto puo' essere grande
``rho`` restando dentro il crop (usa la regione del crop della vista eqsupport SOLO per questa
misura di progetto: la regola non la legge).

Modi di ``build``:
  canonical  la regola;
  offcenter  controllo negativo: stessa area della maschera canonica della stessa mesh, ma centro
             spostato di un raggio nel piano del viso, in direzione casuale per mesh (seme = crc32
             del nome del file); raggio ricalcolato per bisezione fino alla stessa area;
  identity   controllo d'identita': regione = mesh intera, stesso percorso di scrittura.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import zlib
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))
sys.path.insert(0, str(THIS_DIR))

TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")
RING_FRAC = 0.15      # anello della prominenza locale, in frazione di S0
CAND_TOP = 0.25       # candidati: il 25% dei vertici piu' alti sul piano, per verso
N_CAND = 400          # al piu' tanti candidati per verso (sottocampione col seme 0)
CAP_FRAC = 0.3       # calotta della punta: altezza >= massimo - CAP_FRAC x prominenza


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as d:
        V, F = (d["V"], d["F"]) if "V" in d else (d["verts"], d["faces"])
        return np.asarray(V), np.asarray(F)


def tri_area(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    t = np.asarray(V, np.float64)[F]
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def face_frame(V: np.ndarray, F: np.ndarray) -> dict:
    """Piano PCA pesato per area, S0 dall'area proiettata con segno, aree per vertice."""
    V = np.asarray(V, np.float64)
    a = tri_area(V, F)
    c = V[F].mean(1)
    mu = (a[:, None] * c).sum(0) / a.sum()
    X = c - mu
    cov = (a[:, None] * X).T @ X / a.sum()
    w, E = np.linalg.eigh(cov)                  # crescente: E[:, 0] e' la normale
    e1, e2, n = E[:, 2], E[:, 1], E[:, 0]
    P = np.stack([(V - mu) @ e1, (V - mu) @ e2], 1)
    t = P[F]
    signed = 0.5 * ((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                    - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0]))
    va = np.zeros(len(V))
    np.add.at(va, F.ravel(), np.repeat(a / 3.0, 3))
    return {"V": V, "mu": mu, "e1": e1, "e2": e2, "n": n, "S0": float(np.sqrt(abs(signed.sum()))),
            "va": va, "area": float(a.sum())}


def ring_plane(V: np.ndarray, va: np.ndarray, tree, p: np.ndarray, r0: float, up: np.ndarray):
    """Piano (pesato per area) dei vertici a distanza r0 +- 30% da p, normale verso ``up``; None se l'anello e' vuoto."""
    nb = np.asarray(tree.query_ball_point(p, 1.3 * r0))
    if len(nb) == 0:
        return None
    ring = nb[np.abs(np.linalg.norm(V[nb] - p, axis=1) - r0) < 0.3 * r0]
    if len(ring) < 12:
        return None
    w = va[ring]
    c = (w[:, None] * V[ring]).sum(0) / w.sum()
    X = V[ring] - c
    m = np.linalg.eigh((w[:, None] * X).T @ X)[1][:, 0]
    return c, (m if m @ up > 0 else -m)


def nose_tip(fr: dict, F: np.ndarray) -> tuple[np.ndarray, dict]:
    from scipy.spatial import cKDTree
    V, va, n, mu, S0 = fr["V"], fr["va"], fr["n"], fr["mu"], fr["S0"]
    tree = cKDTree(V)
    h = (V - mu) @ n
    r0 = RING_FRAC * S0
    # Candidati lontani dal bordo ESTERNO (l'anello di bordo piu' lungo): li' l'anello e' monco e la
    # prominenza e' spuria. I buchi interni (occhi e bocca di ICT) restano ammessi.
    import igl
    bnd = np.asarray(igl.boundary_loop(np.asarray(F, np.int64)), np.int64)
    d_bnd = cKDTree(V[bnd]).query(V)[0] if len(bnd) else np.full(len(V), np.inf)
    best = None
    for s in (1.0, -1.0):
        ok = (s * h >= np.quantile(s * h, 1.0 - CAND_TOP)) & (d_bnd > 1.5 * r0)
        top = np.flatnonzero(ok)
        cand = top[np.random.default_rng(0).permutation(len(top))[:N_CAND]] if len(top) > N_CAND else top
        for i in cand:
            pl = ring_plane(V, va, tree, V[i], r0, s * n)
            if pl is None:
                continue
            prom = (V[i] - pl[0]) @ pl[1]
            if best is None or prom > best[0]:
                best = (prom, V[i].copy(), s)
    if best is None:
        raise RuntimeError("punta del naso non trovata")
    _, tip, s = best
    up = s * n
    # Rifinitura: baricentro della calotta (altezza >= massimo - CAP_FRAC x prominenza sul piano
    # dell'anello centrato sulla stima corrente), due volte. Un integrale, non un argmax: stabile
    # su punte piatte e sul rumore.
    for _ in range(2):
        c, m = ring_plane(V, va, tree, tip, r0, up)
        near = np.asarray(tree.query_ball_point(tip, 0.1 * S0))
        hm = (V[near] - c) @ m
        prom = float(hm.max())
        cap = near[hm >= prom - CAP_FRAC * prom]
        tip = (va[cap, None] * V[cap]).sum(0) / va[cap].sum()
    d_center = float(np.linalg.norm([(tip - mu) @ fr["e1"], (tip - mu) @ fr["e2"]]))
    return tip, {"prominence_rel": prom / S0, "d_center_rel": d_center / S0, "d_tip_boundary_rel":
                 float(cKDTree(V[bnd]).query(tip)[0] / S0) if len(bnd) else np.inf}


def ball_faces(V: np.ndarray, F: np.ndarray, center: np.ndarray, R: float) -> np.ndarray:
    c = np.asarray(V, np.float64)[F].mean(1)
    return np.linalg.norm(c - center, axis=1) <= R


def restrict(V: np.ndarray, F: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Facce tenute e vertici che usano, nient'altro: con ``keep`` tutto vero e' l'identita' (niente
    pulizia di degeneri o frammenti, che cambierebbe up60k e romperebbe il controllo d'identita')."""
    import mesh_ops as mo

    Vr, Fr = mo.remove_unreferenced(np.asarray(V), np.asarray(F)[keep])
    return Vr.astype(V.dtype), Fr.astype(np.int32)


# ------------------------------------------------------------------------- calibrate

def calibrate_subject(task) -> list[dict]:
    from scipy.spatial import cKDTree
    from eqsupport_view import region_mask

    sid, view_dir, tol = task
    view_dir = Path(view_dir)
    meshes = {t: load(view_dir / f"{sid}_GTready_{t}.npz") for t in TOPOLOGIES}
    Vo, Fo = meshes["original"]
    Vc, Fc = meshes["crop"]
    band = ~region_mask(Vo, Fo, Vc, Fc, tol)   # facce della original tolte dal crop
    band_c = np.asarray(Vo, np.float64)[Fo[band]].mean(1) if band.any() else np.zeros((0, 3))
    band_tree = cKDTree(band_c) if len(band_c) else None
    rows = []
    for t in TOPOLOGIES:
        V, F = meshes[t]
        fr = face_frame(V, F)
        tip, info = nose_tip(fr, F)
        d_band = float(band_tree.query(tip)[0]) if band_tree is not None else np.inf
        rows.append({"subject": sid, "topology": t, "S0": fr["S0"], "area": fr["area"],
                     "sqrt_area": np.sqrt(fr["area"]), **info,
                     "tip_x": tip[0], "tip_y": tip[1], "tip_z": tip[2], "d_tip_band": d_band,
                     "diag": float(np.linalg.norm(fr["V"].max(0) - fr["V"].min(0)))})
    return rows


def cmd_calibrate(a) -> None:
    import pandas as pd
    from zs_stage import select_subjects

    subjects = select_subjects(a.view_dir, a.seed)
    tasks = [(s, str(a.view_dir), 1e-6) for s in subjects]
    rows = []
    with mp.get_context("spawn").Pool(a.n_cores) as pool:
        for r in pool.imap_unordered(calibrate_subject, tasks):
            rows.extend(r)
    df = pd.DataFrame(rows).sort_values(["subject", "topology"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[canon-cal] {len(subjects)} soggetti -> {a.out}")


# ----------------------------------------------------------------------------- build

def mask_mesh(V, F, mode: str, s_dom: float, rho: float, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    if mode == "identity":
        Ve, Fe = restrict(V, F, np.ones(len(F), bool))
        return Ve, Fe, {"identical_to_input": bool(np.array_equal(Ve, V) and np.array_equal(Fe, F))}
    fr = face_frame(V, F)
    tip, info = nose_tip(fr, F)
    scale = s_dom
    R = rho * scale
    keep = ball_faces(V, F, tip, R)
    Ve, Fe = restrict(V, F, keep)
    stats = {"scale": scale, "R": R, "S0": fr["S0"], "tip_x": tip[0], "tip_y": tip[1], "tip_z": tip[2], **info}
    if mode == "canonical":
        return Ve, Fe, stats
    # offcenter: stessa area, centro a un raggio dalla punta in direzione casuale nel piano del viso
    target = tri_area(Ve, Fe).sum()
    th = np.random.default_rng(zlib.crc32(name.encode())).uniform(0, 2 * np.pi)
    p = tip + R * (np.cos(th) * fr["e1"] + np.sin(th) * fr["e2"])
    Vd = np.asarray(V, np.float64)
    center = Vd[int(np.argmin(np.linalg.norm(Vd - p, axis=1)))]
    lo, hi = 0.2 * R, 4.0 * R
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        Vm, Fm = restrict(V, F, ball_faces(V, F, center, mid))
        if tri_area(Vm, Fm).sum() < target:
            lo = mid
        else:
            hi = mid
    Vm, Fm = restrict(V, F, ball_faces(V, F, center, hi))
    stats.update({"off_theta": th, "off_R": hi, "off_center_dist": float(np.linalg.norm(center - tip)),
                  "off_area_ratio": float(tri_area(Vm, Fm).sum() / target)})
    return Vm, Fm, stats


def build_subject(task) -> tuple[str, list[dict]]:
    sid, view_dir, out_npz, mode, s_dom, rho, overwrite = task
    view_dir, out_npz = Path(view_dir), Path(out_npz)
    outs = {t: out_npz / f"{sid}_GTready_{t}.npz" for t in TOPOLOGIES}
    if not overwrite and all(p.exists() for p in outs.values()):
        return "[skip]", []
    rows = []
    for t in TOPOLOGIES:
        V, F = load(view_dir / f"{sid}_GTready_{t}.npz")
        Ve, Fe, st = mask_mesh(V, F, mode, s_dom, rho, f"{sid}_GTready_{t}")
        np.savez_compressed(outs[t], V=Ve, F=Fe)
        rows.append({"subject": sid, "topology": t, "n_faces": len(Fe), "n_faces_full": len(F),
                     "area_frac": float(tri_area(Ve, Fe).sum() / tri_area(V, F).sum()), **st})
    return "[ok]", rows


def cmd_build(a) -> None:
    import pandas as pd

    subjects = sorted({q.name.split("_GTready_")[0] for q in a.view_dir.glob("id*_GTready_*.npz")})
    if a.max_subjects:
        subjects = subjects[: a.max_subjects]
    out_npz = a.out_dir / "npz"
    out_npz.mkdir(parents=True, exist_ok=True)
    tasks = [(s, str(a.view_dir), str(out_npz), a.mode, a.s_dom, a.rho, a.overwrite) for s in subjects]
    print(f"[canon] {len(tasks)} soggetti modo={a.mode} S_dom={a.s_dom} rho={a.rho} R={a.rho * a.s_dom} -> {out_npz}", flush=True)
    rows, tally = [], {"[ok]": 0, "[skip]": 0}
    with mp.get_context("spawn").Pool(a.n_cores) as pool:
        for i, (status, r) in enumerate(pool.imap_unordered(build_subject, tasks, chunksize=2), 1):
            tally[status] += 1
            rows.extend(r)
            if i % 100 == 0 or i == len(tasks):
                print(f"[canon] {i}/{len(tasks)} {tally}", flush=True)
    stats_path = a.out_dir / "mask_stats.csv"
    new = pd.DataFrame(rows)
    if stats_path.exists() and not a.overwrite and len(new):
        old = pd.read_csv(stats_path)
        new = pd.concat([old[~old["subject"].isin(set(new["subject"]))], new], ignore_index=True)
    elif stats_path.exists() and not len(new):
        new = pd.read_csv(stats_path)
    new.sort_values(["subject", "topology"]).to_csv(stats_path, index=False)
    n_files = sum(1 for _ in out_npz.glob("*.npz"))
    if n_files != len(subjects) * len(TOPOLOGIES):
        raise SystemExit(f"vista incompleta: {n_files} file")
    manifest = {"mode": a.mode, "s_dom": a.s_dom, "rho": a.rho, "R": a.rho * a.s_dom,
                "ring_frac": RING_FRAC, "cand_top": CAND_TOP, "n_cand": N_CAND, "view_dir": str(a.view_dir.resolve()),
                "n_subjects": len(subjects), "cap_frac": CAP_FRAC,
                "script_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest()}
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[canon] fatto: {n_files} file, manifest in {a.out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--view-dir", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--seed", type=int, default=1234)
    c.add_argument("--n-cores", type=int, default=8)
    b = sub.add_parser("build")
    b.add_argument("--view-dir", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--mode", choices=("canonical", "offcenter", "identity"), required=True)
    b.add_argument("--s-dom", type=float, default=0.0, help="scala di dominio (PROTOCOL.md)")
    b.add_argument("--rho", type=float, default=0.0)
    b.add_argument("--n-cores", type=int, default=8)
    b.add_argument("--max-subjects", type=int, default=0)
    b.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.cmd == "build" and a.mode != "identity" and (a.rho <= 0 or a.s_dom <= 0):
        raise SystemExit("--rho e --s-dom obbligatori per canonical/offcenter")
    cmd_calibrate(a) if a.cmd == "calibrate" else cmd_build(a)


if __name__ == "__main__":
    main()
