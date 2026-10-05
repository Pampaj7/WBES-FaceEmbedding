#!/usr/bin/env python3
"""Variante F (BOARD_DIARY, ipotesi H6): crop casuali della mesh `original` per i soggetti di training.

    aau/run.sh aau/models/random_crops_F.py gen   --out-dir datasets/REMESH/npz_data_topo_500_cropF
    aau/run.sh aau/models/random_crops_F.py check --ops-dir datasets/REMESH/npz_data_topo_500_cropF_withops \
        --view-dir datasets/REMESH/view_F

gen. Per ognuno dei 400 soggetti di training dello split del trainer (rebuild_subject_split,
eval_fraction 0.2, --seed 1234, sui 500 di REMESH con riga nella matrice GT; nessun crop per i
100 held-out) scrive 5 crop `id<NNNN>_GTready_crop_r<K>.npz` (chiavi V/F come gli npz mesh-only).
Per crop:
  - frazione di vertici tenuti f ~ U[0.6, 0.9], estratta UNA volta (i tentativi rifiutati cambiano
    solo la direzione, quindi f resta uniforme);
  - primo taglio: normale nel semispazio inferiore/laterale del volto, angolo phi dal "giu'" nel
    piano frontale in [-110, 110] gradi (0 = mento, +-90 = lato), inclinazione psi in [-25, 25]
    gradi verso l'avanti/indietro;
  - con probabilita' 0.5 un secondo taglio laterale (|phi| in [60, 120]); la rimozione 1-f si
    divide fra i due tagli (quota del primo w ~ U[0.4, 0.8], riestratta a ogni tentativo);
  - bordo irregolare: sulla distanza dal piano si somma un campo liscio (3 sinusoidi nel piano,
    lunghezza d'onda 0.25-0.8 ex-ex, ampiezza 1.5-4% di ex-ex);
  - occhi e naso protetti: un taglio e' rifiutato se toglie un vertice entro 0.15 ex-ex dalla punta
    del naso o da uno dei 4 angoli degli occhi (~13 mm; gli angoli dello stesso occhio distano ~0.27 ex-ex, quindi l'occhio e' coperto
    per intero; con 0.25 i tagli laterali erano infattibili). Landmark iBUG 31, 37, 40, 43, 46 dalla tabella BFM
    p23470 di WBES/utils, la stessa topologia di `original`; datasets/crop.py non ha landmark);
  - si tiene la componente connessa per lati piu' grande, solo i vertici referenziati (nessun
    vertice isolato); rifiutato se la frazione finale si scosta da f di piu' di 0.02.
Seme per (soggetto, K): default_rng([--crop-seed, NNNN, K]), riproducibile e indipendente
dall'ordine. Statistiche in <stats-dir>/crops_meta.csv e crops_stats.md.

check. Dopo precompute_operators_npz.py: ogni crop ha gli operatori, tutti finiti, stesse chiavi
del crop canonico, k_eig 128, vertici identici al mesh-only. Poi costruisce la vista (symlink a
tutti gli npz di --std-dir piu' i crop con operatori) e verifica come la vede il trainer:
etichette da infer_topology_label_from_name, soggetti di training con 6 file `crop` (canonico +
5 casuali), held-out con le sole 6 topologie standard.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

from size_token import bfm_train_subjects  # noqa: E402

LMK_JSON = REPO_ROOT / "WBES/utils/BFM-p23470.json"
# indici nella lista dei 51 landmark (iBUG 18..68): punta del naso, angoli degli occhi, angoli della bocca
LMK_NOSE = 13
LMK_EYES = (19, 22, 25, 28)
LMK_MOUTH = (31, 37)
STD_TOPOLOGIES = ("original", "remesh", "crop", "noisy", "down8k", "up60k")
NAME_RE = re.compile(r"^(id\d+)_GTready_(.+)\.npz$")

F_RANGE = (0.6, 0.9)
P_SECOND_CUT = 0.5
PHI1_MAX = 110.0
PHI2_RANGE = (60.0, 120.0)
PSI_MAX = 25.0
SPLIT_RANGE = (0.4, 0.8)
NOISE_AMP = (0.015, 0.04)
NOISE_WAVELENGTH = (0.25, 0.8)
PROTECT_RADIUS = 0.15
F_TOL = 0.02
MAX_TRIES = 400


def face_frame(V: np.ndarray, lmk: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """(naso, su, avanti, lato, ex-ex) dai landmark, per soggetto: nessuna ipotesi sugli assi."""
    nose = V[lmk[LMK_NOSE]]
    eyes = V[[lmk[i] for i in LMK_EYES]]
    mouth = V[[lmk[i] for i in LMK_MOUTH]].mean(0)
    up = eyes.mean(0) - mouth
    up /= np.linalg.norm(up)
    fwd = nose - 0.5 * (eyes.mean(0) + mouth)
    fwd -= (fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    lat = np.cross(up, fwd)
    exex = float(np.linalg.norm(eyes[0] - eyes[3]))
    return nose, up, fwd, lat, exex


def cut_normal(up, fwd, lat, phi_deg: float, psi_deg: float) -> np.ndarray:
    phi, psi = math.radians(phi_deg), math.radians(psi_deg)
    inplane = -math.cos(phi) * up + math.sin(phi) * lat
    n = math.cos(psi) * inplane + math.sin(psi) * fwd
    return n / np.linalg.norm(n)


def border_noise(X: np.ndarray, n: np.ndarray, exex: float, rng) -> tuple[np.ndarray, float]:
    """Campo liscio nel piano di taglio: sposta il bordo di qualche mm in modo irregolare."""
    t1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    a, b = X @ t1, X @ t2
    amp = rng.uniform(*NOISE_AMP) * exex
    g = np.zeros(len(X))
    for _ in range(3):
        th = rng.uniform(0.0, 2 * math.pi)
        lam = rng.uniform(*NOISE_WAVELENGTH) * exex
        g += np.sin(2 * math.pi * (math.cos(th) * a + math.sin(th) * b) / lam + rng.uniform(0.0, 2 * math.pi))
    return amp * g / math.sqrt(3.0), amp / exex


def largest_edge_component(F: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Indici (ordinati) dei vertici della componente connessa per lati piu' grande delle facce tenute."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    Fk = F[keep[F].all(1)]
    if len(Fk) == 0:
        return np.zeros(0, dtype=np.int64)
    e = np.sort(np.concatenate([Fk[:, [0, 1]], Fk[:, [1, 2]], Fk[:, [2, 0]]]), axis=1)
    _, eid = np.unique(e, axis=0, return_inverse=True)
    eid = eid.reshape(-1)
    nf, ne = len(Fk), int(eid.max()) + 1
    rows = np.tile(np.arange(nf), 3)
    # grafo bipartito facce + lati: due facce sono connesse se condividono un lato
    g = coo_matrix((np.ones(len(rows)), (rows, nf + eid)), shape=(nf + ne, nf + ne))
    _, lab = connected_components(g, directed=False)
    flab = lab[:nf]
    big = np.bincount(flab).argmax()
    return np.unique(Fk[flab == big])


def submesh(V: np.ndarray, F: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    remap = -np.ones(len(V), dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    Fs = remap[F]
    Fs = Fs[(Fs >= 0).all(1)]
    return V[idx], Fs.astype(F.dtype)


def sample_crop(V: np.ndarray, F: np.ndarray, lmk: list[int], rng) -> tuple[np.ndarray, dict]:
    nose, up, fwd, lat, exex = face_frame(V, lmk)
    centers = V[[lmk[LMK_NOSE]] + [lmk[i] for i in LMK_EYES]]
    prot = (np.linalg.norm(V[:, None, :] - centers[None], axis=2) < PROTECT_RADIUS * exex).any(1)
    X = V - nose
    N = len(V)
    f = rng.uniform(*F_RANGE)
    two = bool(rng.uniform() < P_SECOND_CUT)
    for attempt in range(1, MAX_TRIES + 1):
        # w si riestrae a ogni tentativo: con w piccolo il taglio laterale dovrebbe togliere piu' di
        # quanto gli occhi protetti permettono (~25% a 60 gradi, ~12% a 90)
        w = rng.uniform(*SPLIT_RANGE) if two else 1.0
        phi1 = rng.uniform(-PHI1_MAX, PHI1_MAX)
        psi1 = rng.uniform(-PSI_MAX, PSI_MAX)
        n1 = cut_normal(up, fwd, lat, phi1, psi1)
        g1, amp1 = border_noise(X, n1, exex, rng)
        p1 = X @ n1 + g1
        d1 = np.quantile(p1, 1.0 - (1.0 - f) * w)
        if d1 < p1[prot].max():
            continue
        keep = p1 <= d1
        phi2 = psi2 = amp2 = float("nan")
        if two:
            phi2 = float(rng.choice([-1.0, 1.0]) * rng.uniform(*PHI2_RANGE))
            psi2 = rng.uniform(-PSI_MAX, PSI_MAX)
            n2 = cut_normal(up, fwd, lat, phi2, psi2)
            g2, amp2 = border_noise(X, n2, exex, rng)
            p2 = X @ n2 + g2
            d2 = np.quantile(p2[keep], min(1.0, f * N / keep.sum()))
            if d2 < p2[prot].max():
                continue
            keep &= p2 <= d2
        idx = largest_edge_component(F, keep)
        if not np.isin(np.flatnonzero(prot), idx).all():
            continue
        f_act = len(idx) / N
        if abs(f_act - f) > F_TOL:
            continue
        return idx, {"f_target": f, "f_kept": f_act, "n_cuts": 2 if two else 1, "w_first": w,
                     "phi1": phi1, "psi1": psi1, "phi2": phi2, "psi2": psi2,
                     "noise_amp1": amp1, "noise_amp2": amp2, "tries": attempt,
                     "n_protected": int(prot.sum()), "exex": exex}
    raise RuntimeError(f"nessun taglio valido in {MAX_TRIES} tentativi (f={f:.3f}, tagli={1 + two})")


def gen_one(task: tuple) -> list[dict]:
    raw_dir, out_dir, sid, n_crops, crop_seed, lmk, overwrite = task
    with np.load(Path(raw_dir) / f"{sid}_GTready_original.npz", allow_pickle=False) as z:
        V, F = np.asarray(z["V"], dtype=np.float64), np.asarray(z["F"])
    if len(V) != 23470:
        raise ValueError(f"{sid}: original con {len(V)} vertici, i landmark sono per BFM p23470")
    rows = []
    for k in range(1, n_crops + 1):
        rng = np.random.default_rng([crop_seed, int(sid[2:]), k])
        idx, meta = sample_crop(V, F, lmk, rng)
        Vc, Fc = submesh(V, F, idx)
        out = Path(out_dir) / f"{sid}_GTready_crop_r{k}.npz"
        if overwrite or not out.exists():
            np.savez(out, V=Vc, F=Fc)
        # invarianti chiesti: non vuoto, nessun vertice isolato
        assert len(Fc) > 0 and np.unique(Fc).size == len(Vc), out.name
        rows.append({"name": out.stem, "subject": sid, "k": k, "n_verts": len(Vc), "n_faces": len(Fc), **meta})
    return rows


def describe(x: np.ndarray) -> str:
    q = np.quantile(x, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return (f"media {x.mean():.3f}, min {q[0]:.3f}, p5 {q[1]:.3f}, p25 {q[2]:.3f}, mediana {q[3]:.3f}, "
            f"p75 {q[4]:.3f}, p95 {q[5]:.3f}, max {q[6]:.3f}")


def write_stats(rows: list[dict], stats_dir: Path, header: list[str]) -> str:
    stats_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_dir / "crops_meta.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    f = np.array([r["f_kept"] for r in rows])
    ft = np.array([r["f_target"] for r in rows])
    nv = np.array([r["n_verts"] for r in rows])
    tries = np.array([r["tries"] for r in rows])
    two = np.array([r["n_cuts"] == 2 for r in rows])
    aphi = np.abs(np.array([r["phi1"] for r in rows]))
    hist, edges = np.histogram(f, bins=6, range=F_RANGE)
    lines = header + [
        "",
        f"Crop: {len(rows)} ({len({r['subject'] for r in rows})} soggetti), vuoti: {(nv == 0).sum()}.",
        f"Frazione di vertici tenuti (finale, dopo la componente piu' grande): {describe(f)}.",
        f"Frazione obiettivo: {describe(ft)}; |finale - obiettivo| max {np.abs(f - ft).max():.4f}.",
        f"Vertici: min {nv.min()}, mediana {int(np.median(nv))}, max {nv.max()} (original 23470; crop canonico ~0.90).",
        f"Tagli: uno {(~two).sum()}, due {two.sum()} ({two.mean():.1%}).",
        f"Primo taglio, |phi| dal mento: <30 gradi {(aphi < 30).sum()}, 30-60 {((aphi >= 30) & (aphi < 60)).sum()}, "
        f"60-90 {((aphi >= 60) & (aphi < 90)).sum()}, 90-110 {(aphi >= 90).sum()}.",
        f"Tentativi per crop: media {tries.mean():.1f}, max {tries.max()}.",
        "",
        "| frazione tenuta | crop |",
        "|---|---|",
    ] + [f"| {edges[i]:.2f}-{edges[i + 1]:.2f} | {hist[i]} |" for i in range(len(hist))]
    for lo, hi in ((0.6, 0.7), (0.7, 0.8), (0.8, 0.9)):
        sel = (ft >= lo) & (ft < hi)
        if sel.any():
            a = aphi[sel]
            lines.append(f"\nObiettivo {lo:.1f}-{hi:.1f}: |phi| mediano {np.median(a):.0f} gradi, "
                         f"due tagli {two[sel].mean():.0%}, tentativi medi {tries[sel].mean():.1f}.")
    text = "\n".join(lines) + "\n"
    (stats_dir / "crops_stats.md").write_text(text)
    return text


def cmd_gen(args) -> None:
    names = sorted(p.name[:-len(".npz")] for p in args.raw_dir.iterdir() if p.name.endswith(".npz"))
    train = bfm_train_subjects(names, args.dist_npz, args.seed)
    all_subjects = sorted({n.split("_")[0] for n in names})
    held = sorted(set(all_subjects) - set(train))
    print(f"[cropF] soggetti {len(all_subjects)}: training {len(train)}, held-out {len(held)} "
          f"(seed split {args.seed}); primi held-out {held[:5]}", flush=True)
    if args.control_runs_root is not None:
        # i soggetti dell'eval online del controllo sono held-out per costruzione: devono stare fuori
        js = list(Path(args.control_runs_root).glob("*/online_eval_summary.json"))
        sel = json.loads(js[0].read_text())["selected_subjects"]
        bad = sorted(set(sel) & set(train))
        if bad:
            raise SystemExit(f"split diverso dal controllo: soggetti dell'eval online nel training {bad}")
        print(f"[cropF] {len(sel)} soggetti dell'eval online del controllo tutti held-out: ok", flush=True)
    if args.limit:
        train = train[: args.limit]
    lmk = json.loads(LMK_JSON.read_text())["lmk_indices"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(args.raw_dir), str(args.out_dir), sid, args.n_crops, args.crop_seed, lmk, args.overwrite)
             for sid in train]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(gen_one, tasks, chunksize=4), start=1):
            rows += r
            if i % 50 == 0:
                print(f"[cropF] {i}/{len(tasks)} soggetti", flush=True)
    rows.sort(key=lambda r: r["name"])
    header = ["# Variante F: crop casuali per i soggetti di training", "",
              f"Sorgente `{args.raw_dir}` (original), split rebuild_subject_split(0.2, seed {args.seed}): "
              f"{len(train)} soggetti di training, {args.n_crops} crop ciascuno, seme {args.crop_seed}. "
              f"Held-out esclusi: {len(held)}. Regola in aau/models/random_crops_F.py."]
    print(write_stats(rows, args.stats_dir, header), flush=True)


def finite_report(path: Path, ref_keys: set[str], raw_dir: Path, k_eig: int) -> str | None:
    """None se l'npz con operatori e' sano, altrimenti il motivo."""
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        if keys != ref_keys:
            return f"chiavi {sorted(keys ^ ref_keys)}"
        n = z["verts"].shape[0]
        for k in ("verts", "mass", "evals", "evecs", "L_values", "gradX_values", "gradY_values"):
            if not np.isfinite(z[k]).all():
                return f"{k} non finito"
        if z["evals"].shape != (k_eig,) or z["evecs"].shape != (n, k_eig):
            return f"evals {z['evals'].shape} evecs {z['evecs'].shape}"
        if not (z["mass"] > 0).all():
            return "massa non positiva"
        if tuple(z["L_shape"]) != (n, n):
            return f"L_shape {tuple(z['L_shape'])}"
        with np.load(raw_dir / path.name, allow_pickle=False) as r:
            if not np.array_equal(r["V"].astype(np.float32), z["verts"]) or not np.array_equal(r["F"], z["faces"]):
                return "vertici/facce diversi dal mesh-only"
    return None


def cmd_check(args) -> None:
    raw = sorted(p.name for p in args.raw_crop_dir.iterdir() if NAME_RE.match(p.name))
    ops = {p.name for p in args.ops_dir.iterdir() if p.name.endswith(".npz")}
    missing = [n for n in raw if n not in ops]
    if missing:
        raise SystemExit(f"{len(missing)}/{len(raw)} crop senza operatori (primo {missing[0]})")
    with np.load(args.std_dir / "id0000_GTready_crop.npz", allow_pickle=False) as z:
        ref_keys = set(z.files)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        bad = [(n, r) for n, r in zip(raw, ex.map(finite_report, [args.ops_dir / n for n in raw],
                                                    [ref_keys] * len(raw), [args.raw_crop_dir] * len(raw),
                                                    [args.k_eig] * len(raw), chunksize=8)) if r]
    if bad:
        raise SystemExit(f"{len(bad)} crop con operatori non validi: {bad[:5]}")
    print(f"[cropF] operatori: {len(raw)} crop, tutti finiti, k_eig {args.k_eig}, chiavi come il crop canonico", flush=True)

    # vista: symlink assoluti, idempotente
    args.view_dir.mkdir(parents=True, exist_ok=True)
    std = sorted(p.name for p in args.std_dir.iterdir() if p.name.endswith(".npz"))
    links = [(args.std_dir / n, n) for n in std] + [(args.ops_dir / n, n) for n in raw]
    for target, name in links:
        dst = args.view_dir / name
        if dst.is_symlink() or dst.exists():
            if dst.resolve() == target.resolve():
                continue
            raise SystemExit(f"{dst} esiste e punta altrove ({dst.resolve()})")
        os.symlink(target.resolve(), dst)
    extra = sorted(set(os.listdir(args.view_dir)) - {n for _, n in links})
    if extra:
        raise SystemExit(f"{args.view_dir} contiene file estranei: {extra[:5]}")

    # come la vede il trainer (train_runner.py:1552-1563); bfm_train_subjects mette il trainer sul path
    train = set(bfm_train_subjects([f[:-4] for f in std], args.dist_npz, args.seed))
    from intrinsic_utils import SUBJECT_RE_ANY, build_subject_map  # noqa: E402
    from robustness.data_utils import infer_topology_label_from_name  # noqa: E402
    files = sorted(f for f in os.listdir(args.view_dir) if f.endswith(".npz"))
    subj_map = build_subject_map(files, subject_re=SUBJECT_RE_ANY)
    counts = {"train": {}, "held": {}}
    for sid, idxs in subj_map.items():
        labels = [infer_topology_label_from_name(files[i], sid) for i in idxs]
        if sorted(set(labels)) != sorted(STD_TOPOLOGIES):
            raise SystemExit(f"{sid}: etichette {sorted(set(labels))}")
        n_crop = labels.count("crop")
        want = 1 + args.n_crops if sid in train else 1
        if n_crop != want:
            raise SystemExit(f"{sid} ({'train' if sid in train else 'held-out'}): {n_crop} file crop, attesi {want}")
        key = "train" if sid in train else "held"
        counts[key][n_crop] = counts[key].get(n_crop, 0) + 1
    msg = (f"[cropF] vista {args.view_dir}: {len(files)} npz ({len(std)} standard + {len(raw)} crop casuali); "
           f"etichette del trainer = le 6 standard per tutti i {len(subj_map)} soggetti; file 'crop' per "
           f"soggetto: training {counts['train']}, held-out {counts['held']}")
    print(msg, flush=True)
    if args.stats_dir is not None:
        with open(args.stats_dir / "crops_stats.md", "a") as fh:
            fh.write(f"\nOperatori (k_eig {args.k_eig}): {len(raw)} crop, tutti finiti, massa positiva, "
                     f"vertici identici al mesh-only.\n\n{msg[8:]}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "check"):
        p = sub.add_parser(name)
        p.add_argument("--dist-npz", type=Path, default=Path(os.environ.get("WBES_DIST_NPZ", "")))
        p.add_argument("--seed", type=int, default=1234, help="seed dello split del training")
        p.add_argument("--n-crops", type=int, default=5)
        p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
        p.add_argument("--stats-dir", type=Path, default=REPO_ROOT / "aau/runs/ablation_F")
    g = sub.choices["gen"]
    g.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "datasets/REMESH/npz_data_topo_500")
    g.add_argument("--out-dir", type=Path, default=REPO_ROOT / "datasets/REMESH/npz_data_topo_500_cropF")
    g.add_argument("--crop-seed", type=int, default=6)
    g.add_argument("--limit", type=int, default=0, help="solo i primi N soggetti di training (prova)")
    g.add_argument("--control-runs-root", type=Path, default=None,
                   help="runs_root del controllo: verifica che il suo split coincida")
    g.add_argument("--overwrite", action="store_true")
    c = sub.choices["check"]
    c.add_argument("--raw-crop-dir", type=Path, default=REPO_ROOT / "datasets/REMESH/npz_data_topo_500_cropF")
    c.add_argument("--ops-dir", type=Path, default=REPO_ROOT / "datasets/REMESH/npz_data_topo_500_cropF_withops")
    c.add_argument("--std-dir", type=Path, default=REPO_ROOT / "datasets/REMESH/npz_data_topo_500_withops")
    c.add_argument("--view-dir", type=Path, default=REPO_ROOT / "datasets/REMESH/view_F")
    c.add_argument("--k-eig", type=int, default=128)
    args = ap.parse_args()
    (cmd_gen if args.cmd == "gen" else cmd_check)(args)


if __name__ == "__main__":
    main()
