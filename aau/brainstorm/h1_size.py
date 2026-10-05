#!/usr/bin/env python3
"""H1 (BOARD_DIARY, brainstorm sugli operatori): la taglia della testa e' invisibile al modello?

La D_GT BFM e' in coordinate grezze e contiene la taglia; il modello riceve vertici
normalizzati maxabs per mesh.  Sui 100 soggetti held-out:

  1. tre misure di taglia per ogni ``original``: divisore maxabs (``common.maxabs_normalize``,
     la stessa di dataset_gtready), radice dell'area totale, raggio rms dal baricentro;
     Spearman(D_GT, |Δlog taglia|) sulle 4950 coppie;
  2. distanza latent v1, per ogni coppia di topologie: Spearman(D_GT, latent), Spearman
     parziale fra le due controllando per |Δlog taglia| (Pearson parziale sui ranghi), e
     Spearman(rango D_GT − rango latent, |Δlog taglia|).  Positivo = il latent mette vicine
     coppie che la GT allontana proprio quando le taglie differiscono;
  3. rapporto fra il divisore maxabs di ogni topologia e quello dell'original dello stesso
     soggetto (media e dispersione), confrontato con la dispersione della taglia fra soggetti.

original→original usa la matrice gia' pubblicata (``aau/runs/baselines/matrices/latent_v1``);
le altre topologie i latenti di ``latent_matrix.py --topology ... --out-root
aau/runs/brainstorm/latent_v1`` (job 1054992), da cui le matrici cross si ottengono come in
``latent_matrix.py``: norma L2 fra latenti grezzi.  Fra due topologie diverse si usano tutte
le coppie i≠j (Z_A[i] contro Z_B[j] non e' simmetrico); dentro la stessa topologia i<j.
La taglia e' sempre quella dell'original del soggetto: e' la taglia vera, che la GT contiene.

    AAU_NV="" srun -p cpu -c 2 --mem=8G aau/run.sh aau/brainstorm/h1_size.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "baselines"))

import common  # noqa: E402

RUN_DIR = common.AAU_DIR / "runs" / "brainstorm"
LATENT_ORIG = common.OUT_ROOT / "matrices" / "latent_v1" / "original__to__original.npz"
LATENT_EMB = RUN_DIR / "latent_v1" / "embeddings"
SIZES = ("maxabs", "sqrt_area", "rms")


def size_measures(V: np.ndarray, F: np.ndarray) -> dict[str, float]:
    Vc = V - V.mean(axis=0, keepdims=True)
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    return {"maxabs": float(np.max(np.abs(Vc))), "sqrt_area": float(np.sqrt(area)),
            "rms": float(np.sqrt((Vc ** 2).sum(1).mean()))}


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    c = np.corrcoef(np.stack([rx, ry, rz]))
    return float((c[0, 1] - c[0, 2] * c[1, 2]) / np.sqrt((1 - c[0, 2] ** 2) * (1 - c[1, 2] ** 2)))


def rho(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).correlation)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=RUN_DIR / "h1")
    args = ap.parse_args()

    subjects = common.subject_set("heldout")
    n = len(subjects)
    T = common.TOPOLOGIES
    size = {t: {m: np.empty(n) for m in SIZES} for t in T}
    for k, s in enumerate(subjects):
        for t in T:
            for m, v in size_measures(*common.load_verts_faces(s, t)).items():
                size[t][m][k] = v
    D_gt = common.load_gt_submatrix(subjects)
    iu, ju = common.subject_pair_indices(n)
    logsize = {m: np.log(size["original"][m]) for m in SIZES}
    dlog = {m: np.abs(logsize[m][:, None] - logsize[m][None, :]) for m in SIZES}
    out: dict = {"n_subjects": n}

    # 1. la taglia da sola
    out["rho_gt_size"] = {m: rho(D_gt[iu, ju], dlog[m][iu, ju]) for m in SIZES}
    out["rho_between_sizes"] = {f"{a}~{b}": rho(logsize[a], logsize[b])
                                for a, b in itertools.combinations(SIZES, 2)}

    # 2. latent
    D0, subj0, *_ = common.load_matrix(LATENT_ORIG)
    if subj0 != subjects:
        raise ValueError("ordine dei soggetti della matrice latent diverso da subject_set('heldout')")
    Z = {}
    for t in T:
        with np.load(LATENT_EMB / f"latent_v1_{t}.npz") as z:
            Z[t] = np.stack([z[common.mesh_name(s, t)] for s in subjects]).astype(np.float64)
    D_orig_re = np.linalg.norm(Z["original"][:, None] - Z["original"][None], axis=2)
    out["latent_original_reproduction_maxabs_diff"] = float(np.max(np.abs(D_orig_re[iu, ju] - D0[iu, ju])))

    pairs = [(t, t) for t in T] + list(itertools.combinations(T, 2))
    rows = []
    off = ~np.eye(n, dtype=bool)
    for tA, tB in pairs:
        if tA == tB:
            sel = (iu, ju)
            D_lat = D0 if tA == "original" else np.linalg.norm(Z[tA][:, None] - Z[tA][None], axis=2)
        else:
            sel = np.nonzero(off)
            D_lat = np.linalg.norm(Z[tA][:, None] - Z[tB][None], axis=2)
        gt, lat = D_gt[sel], D_lat[sel]
        resid = rankdata(gt) - rankdata(lat)
        row = {"pair": f"{tA}__{tB}", "n_pairs": int(gt.size), "rho_gt_latent": rho(gt, lat)}
        for m in SIZES:
            row[f"partial_{m}"] = partial_spearman(gt, lat, dlog[m][sel])
            row[f"resid_{m}"] = rho(resid, dlog[m][sel])
            row[f"rho_latent_{m}"] = rho(lat, dlog[m][sel])
        rows.append(row)
        print(f"[h1] {tA}->{tB}: " + " ".join(f"{k}={v:.3f}" for k, v in row.items()
                                               if isinstance(v, float)), flush=True)

    # 3. il divisore maxabs fuori dall'original
    ratio = {}
    for t in T:
        if t == "original":
            continue
        ratio[t] = {}
        for m in SIZES:
            r = size[t][m] / size["original"][m]
            ratio[t][m] = {"mean": float(r.mean()), "std": float(r.std()), "min": float(r.min()),
                           "max": float(r.max()), "std_log": float(np.log(r).std())}
    out["ratio_to_original"] = ratio
    out["std_log_size_between_subjects"] = {m: float(logsize[m].std()) for m in SIZES}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out["pairs"] = rows
    args.out.with_suffix(".json").write_text(json.dumps(out, indent=1))

    # --- report
    get = {r["pair"]: r for r in rows}
    same = [r for r in rows if r["pair"].split("__")[0] == r["pair"].split("__")[1]]
    cross = [r for r in rows if r not in same]
    crop = [r for r in cross if "crop" in r["pair"]]
    nocrop = [r for r in cross if "crop" not in r["pair"]]

    def mean(rs, k):
        return float(np.mean([r[k] for r in rs]))

    L = [(RUN_DIR / "PREDICTIONS.md").read_text(), "",
         f"# H1: taglia invisibile ({n} soggetti held-out BFM, D_GT grezza)", "",
         "## Taglia da sola: Spearman(D_GT, |Δlog taglia|), 4950 coppie", "",
         "| maxabs | √area | rms |", "|---:|---:|---:|",
         "| " + " | ".join(f"{out['rho_gt_size'][m]:.3f}" for m in SIZES) + " |", "",
         "Concordanza fra le misure (Spearman fra log-taglie dei soggetti): " +
         ", ".join(f"{k} {v:.3f}" for k, v in out["rho_between_sizes"].items()), "",
         "## Latent v1 contro D_GT, controllando per la taglia", "",
         "| coppie | ρ(GT,latent) | parziale maxabs | parziale √area | parziale rms "
         "| ρ(resid,Δmaxabs) | ρ(resid,Δ√area) | ρ(resid,Δrms) |",
         "|---|" + "---:|" * 7]
    groups = [("original→original", [get["original__original"]]),
              ("same, media 6", same), ("cross, media 15", cross),
              ("cross senza crop, media 10", nocrop), ("cross con crop, media 5", crop)]
    keys = ["rho_gt_latent"] + [f"partial_{m}" for m in SIZES] + [f"resid_{m}" for m in SIZES]
    for name, rs in groups:
        L.append(f"| {name} | " + " | ".join(f"{mean(rs, k):.3f}" for k in keys) + " |")
    L += ["", f"Riproduzione della matrice latent original pubblicata dai latenti ricalcolati: "
          f"max |Δ| = {out['latent_original_reproduction_maxabs_diff']:.2e}.", "",
          "## Divisore della topologia / divisore dell'original, stesso soggetto", "",
          "| topologia | maxabs media | std | min–max | std log | √area media | std log |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for t, r in ratio.items():
        a, b = r["maxabs"], r["sqrt_area"]
        L.append(f"| {t} | {a['mean']:.4f} | {a['std']:.4f} | {a['min']:.3f}–{a['max']:.3f} "
                 f"| {a['std_log']:.4f} | {b['mean']:.4f} | {b['std_log']:.4f} |")
    oo = get["original__original"]
    L += ["", "Dispersione della taglia fra soggetti (std del log sugli original): " +
          ", ".join(f"{m} {v:.4f}" for m, v in out["std_log_size_between_subjects"].items()), ""]

    L += ["Per leggere il segno del residuo, Spearman(latent, |Δlog taglia|) su original→original: "
          + ", ".join(f"{m} {oo[f'rho_latent_{m}']:.3f}" for m in SIZES), ""]
    passing = [m for m in SIZES if out["rho_gt_size"][m] > 0.3 and oo[f"resid_{m}"] > 0.1]
    L += ["## Verdetto H1", "",
          "Per misura: " + ", ".join(f"{m} ρ(GT,Δ)={out['rho_gt_size'][m]:.3f} "
                                     f"ρ(resid,Δ)={oo[f'resid_{m}']:.3f}" for m in SIZES), "",
          f"**H1 {'CONFERMATA' if passing else 'NON confermata'}**"
          + (f" (misure che passano: {', '.join(passing)})" if passing else "")
          + " — soglie 0.3 e 0.1 su original→original.", "",
          f"Tutte le 21 coppie di topologie in `{args.out.with_suffix('.csv').name}`."]
    args.out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L[2:]))


if __name__ == "__main__":
    main()
