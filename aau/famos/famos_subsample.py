#!/usr/bin/env python3
"""Registrazioni FaMoS sottocampionate, una npz per soggetto, piu' la forma neutra di riferimento.

    aau/run.sh aau/famos/famos_subsample.py --stride 10 --workers 24

Per ogni soggetto dello split (``aau/famos/split.json``), in ``datasets/FAMOS/train/`` o
``datasets/FAMOS/test/`` secondo lo split -- le persone di TEST non finiscono MAI in ``train/``,
controllato sugli id alla fine:

Fotogrammi: a 60 fps i vicini sono quasi uguali, quindi per ogni sequenza uno ogni ``--stride``
nella lista ordinata dei fotogrammi registrati (posizioni 0, stride, 2 stride, ...: per POSIZIONE, non
per numero di fotogramma, cosi' un buco nella registrazione non toglie campioni). Tutte le sequenze,
comprese le rotazioni della testa e ``sentence``: la sequenza resta nel campo ``seq``.

Forma neutra di riferimento: FaMoS non ha una sequenza neutra, ma ogni sequenza parte dal volto
neutro. Si prendono i PRIMI fotogrammi registrati di tutte le sequenze del soggetto e:
  1. si portano nella regione FLAME unificata (``v3_work/unified_gt``, mappa ``flame``: FaMoS e' gia'
     in topologia FLAME) con Procrustes di similarita' verso mu, distanze g in mm come la GT;
  2. medoide = il primo fotogramma con la somma minima delle distanze dagli altri;
  3. si tengono i primi fotogrammi entro 2 x la mediana delle distanze dal medoide (via quelli partiti
     gia' in espressione o con la registrazione sbagliata);
  4. neutra = media dei tenuti, ciascuno allineato RIGIDAMENTE (senza scala) al medoide sui punti della
     regione unificata pesati per area, applicando la stessa trasformazione a tutta la mesh FLAME
     (come la media dei neutri di Multiface in ``shapes.py``).
Controllo di riproducibilita': la stessa media sulle meta' pari e dispari dei tenuti, distanza g fra le
due (il rumore della stima, da confrontare con le distanze fra persone).

Uscita ``<soggetto>.npz`` (float32, mm = registrazioni x 1000): ``V`` (n, 5023, 3), ``seq``, ``frame``,
``V_neutral`` (5023, 3), ``neutral_from``, ``first_frames``, ``first_dist_to_medoid_mm``,
``neutral_halves_mm``; ``datasets/FAMOS/flame_faces.npy`` (le facce FLAME, uguali per tutti, controllate
su ogni primo fotogramma); ``datasets/FAMOS/manifest.json`` coi conteggi e il controllo dello split.
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
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parents[1] / "v3_work" / "unified_gt"))

import famos_common as fc  # noqa: E402
import ugt as C  # noqa: E402
from shapes import Space, unified_distances  # noqa: E402

NEUTRAL_KEEP = 2.0      # x mediana delle distanze dal medoide

_SP: Space | None = None
_F_REF: np.ndarray | None = None


def _init(F_ref: np.ndarray) -> None:
    global _SP, _F_REF
    _SP = Space()
    _F_REF = F_ref


def unified_s(sp: Space, V: np.ndarray) -> np.ndarray:
    """s_i (k, 3n) di mesh FLAME (k, 5023, 3) in mm, come shapes.py."""
    a, _, _ = sp.align(sp.map("flame", V))
    return sp.svec(a)


def neutral_shape(sp: Space, Vf: np.ndarray) -> dict:
    """Passi 1-4 della docstring su (k, 5023, 3) primi fotogrammi."""
    S = unified_s(sp, Vf)
    D = unified_distances(S, S, sp.A)
    med = int(np.argmin(D.sum(1)))
    d = D[:, med]
    thr = NEUTRAL_KEEP * float(np.median(np.delete(d, med)))
    keep = np.flatnonzero(d <= thr)
    P = sp.map("flame", Vf)
    aligned = []
    for k in keep:
        s, R, t = C.umeyama(P[k], P[med], sp.w, scale=False)
        aligned.append(C.apply_sim(Vf[k].astype(np.float64), s, R, t))
    aligned = np.stack(aligned)
    Vn = aligned.mean(0)
    halves = float("nan")
    if len(keep) >= 4:
        Sh = unified_s(sp, np.stack([aligned[0::2].mean(0), aligned[1::2].mean(0)]))
        halves = float(unified_distances(Sh[:1], Sh[1:], sp.A)[0, 0])
    return {"V": Vn, "keep": keep, "medoid": med, "dist": d, "threshold": thr, "halves": halves}


def process(task: tuple) -> dict:
    subject, split, stride, out_path, overwrite = task
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        with np.load(out_path) as z:
            return {"subject": subject, "status": "skip", "n_frames": int(len(z["frame"])),
                    "bytes": out_path.stat().st_size, "neutral_halves_mm": float(z["neutral_halves_mm"])}
    t0 = time.time()
    seq_dirs = sorted(p for p in (fc.REG_DIR / subject).iterdir() if p.is_dir())
    V, seq, frame, firsts, Vfirst = [], [], [], [], []
    for sd in seq_dirs:
        frames = fc.frames_of(sd, "ply")
        if not frames:
            continue
        for pos, (f, path) in enumerate(frames):
            if pos % stride:
                continue
            if pos == 0:
                v, F = fc.read_reg(path, with_faces=True)
                if not np.array_equal(F, _F_REF):
                    raise RuntimeError(f"{path}: facce diverse dalla topologia FLAME di riferimento")
                firsts.append(f"{sd.name}.{f:06d}")
                Vfirst.append(v)
            else:
                v = fc.read_reg(path)
            V.append(v)
            seq.append(sd.name)
            frame.append(f)
    V = np.stack(V).astype(np.float32) * np.float32(1000.0)          # m -> mm
    Vfirst = np.stack(Vfirst).astype(np.float64) * 1000.0
    if not np.isfinite(V).all():
        raise RuntimeError(f"{subject}: vertici non finiti")
    nt = neutral_shape(_SP, Vfirst)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(tmp, V=V, seq=np.asarray(seq), frame=np.asarray(frame, dtype=np.int32),
             V_neutral=nt["V"].astype(np.float32), neutral_from=np.asarray(firsts)[nt["keep"]],
             neutral_medoid=np.asarray(firsts[nt["medoid"]]), first_frames=np.asarray(firsts),
             first_dist_to_medoid_mm=nt["dist"].astype(np.float32), neutral_threshold_mm=np.float32(nt["threshold"]),
             neutral_halves_mm=np.float32(nt["halves"]), subject=np.asarray(subject), split=np.asarray(split),
             stride=np.int32(stride), units=np.asarray("mm"))
    tmp.replace(out_path)
    return {"subject": subject, "status": "ok", "n_frames": len(frame), "n_sequences": len(firsts),
            "n_neutral_kept": int(len(nt["keep"])), "neutral_threshold_mm": nt["threshold"],
            "neutral_halves_mm": nt["halves"], "bytes": out_path.stat().st_size, "seconds": time.time() - t0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--subjects", default="", help="sottoinsieme separato da virgole (prova); vuoto = tutti")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    split = fc.load_split()
    test, train = set(split["test"]), set(split["train"])
    todo = sorted(test | train)
    if a.subjects:
        todo = [s for s in todo if s in set(a.subjects.split(","))]
    fc.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    fc.TEST_DIR.mkdir(parents=True, exist_ok=True)
    _, F_ref = fc.read_reg(fc.frames_of(sorted((fc.REG_DIR / todo[0]).iterdir())[0], "ply")[0][1], with_faces=True)
    np.save(fc.FLAME_FACES, F_ref)

    tasks = []
    for s in todo:
        sp = "test" if s in test else "train"
        out = (fc.TEST_DIR if sp == "test" else fc.TRAIN_DIR) / f"{s}.npz"
        tasks.append((s, sp, a.stride, str(out), a.overwrite))
    print(f"[famos-sub] {len(tasks)} soggetti, stride {a.stride}, {a.workers} processi", flush=True)
    rows = []
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(F_ref,)) as pool:
        for r in pool.imap_unordered(process, tasks):
            rows.append(r)
            print(f"[famos-sub] {len(rows)}/{len(tasks)} {r['subject']} {r['status']} fotogrammi={r['n_frames']} "
                  f"neutra: tenuti {r.get('n_neutral_kept', '-')}, meta' {r['neutral_halves_mm']:.3f} mm "
                  f"({r['bytes'] / 1e6:.0f} MB)", flush=True)
    rows.sort(key=lambda r: r["subject"])

    # controllo esplicito dello split sugli id, su quello che c'e' DAVVERO su disco
    on_train = sorted(q.stem for q in fc.TRAIN_DIR.glob("FaMoS_subject_*.npz"))
    on_test = sorted(q.stem for q in fc.TEST_DIR.glob("FaMoS_subject_*.npz"))
    leaked = sorted(set(on_train) & test)
    for q in fc.TRAIN_DIR.glob("FaMoS_subject_*.npz"):
        with np.load(q) as z:
            if str(z["subject"]) != q.stem or str(z["split"]) != "train" or str(z["subject"]) in test:
                leaked.append(q.name)
    split_check = {"train_files": len(on_train), "test_files": len(on_test),
                   "test_subjects_in_train_dir": leaked,
                   "train_dir_subjects_not_in_split_train": sorted(set(on_train) - train),
                   "test_dir_equals_split_test": set(on_test) == test if not a.subjects else None}
    halves = np.array([r["neutral_halves_mm"] for r in rows], dtype=float)
    man = {"stride": a.stride, "units": "mm (registrazioni FaMoS x 1000)", "neutral_rule": __doc__.split("Forma neutra")[1]
           .split("Uscita")[0].strip(), "n_subjects": len(rows), "n_frames": int(sum(r["n_frames"] for r in rows)),
           "bytes": int(sum(r["bytes"] for r in rows)),
           "neutral_halves_mm": {"median": float(np.nanmedian(halves)), "max": float(np.nanmax(halves))},
           "split_check": split_check, "subjects": rows}
    (fc.OUT_ROOT / "manifest.json").write_text(json.dumps(man, indent=1, default=float) + "\n")
    print(f"[famos-sub] {man['n_frames']} fotogrammi, {man['bytes'] / 1e9:.2f} GB; neutra, distanza fra le meta' "
          f"mediana {man['neutral_halves_mm']['median']:.3f} mm; controllo split {split_check}", flush=True)
    if leaked or split_check["train_dir_subjects_not_in_split_train"] or split_check["test_dir_equals_split_test"] is False:
        raise SystemExit("ERRORE: split violato su disco")


if __name__ == "__main__":
    main()
