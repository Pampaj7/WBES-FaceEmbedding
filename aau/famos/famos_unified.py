#!/usr/bin/env python3
"""GT unificata delle identita' FaMoS e loro posizione rispetto agli altri domini, nello spazio unificato.

    aau/run.sh aau/famos/famos_unified.py          (dopo famos_subsample.py)

s_i come ``v3_work/unified_gt/shapes.py``: la forma neutra di riferimento di ogni soggetto
(``V_neutral`` di famos_subsample.py, topologia FLAME, mm) portata nella regione unificata con la
mappa ``flame`` (diretta: FaMoS e' gia' FLAME), Procrustes di similarita' pesato per area verso mu,
s_i = sqrt(w) a_i; g_ij = ||s_i - s_j|| / sqrt(A), mm.

Uscite (dati, fuori dal repo):
  - ``datasets/FAMOS/unified/famos_shapes.npz``: ``s``, ``ids``, ``split``, ``scale``, ``rms_to_mu``;
  - ``datasets/FAMOS/unified/gt_all_mm.npz``: D (95, 95) in mm, nomi ``FaMoS_subject_NNN``;
  - ``datasets/FAMOS/test_view/gt_matrix.npz`` (+ .json): la GT dei soggetti di TEST nel formato delle
    viste zero-shot (``D_orig`` diviso per il massimo, ``names`` id9400NN, ``mm_per_unit`` nel json),
    come ``v3_work/unified_gt/make_eval_gt.py``.
Evidenze (numeri, nel repo): ``aau/runs/evidence/famos/unified_gt.json``, ``famos_to_domain_means.csv``
(distanza di ogni soggetto dalle medie degli 8 domini) e ``unified_gt.md``. Accanto, per confronto, la
dispersione degli altri domini da ``aau/runs/evidence/e8/evidence.json`` e il controllo dello split nello
spazio: distanza di ogni persona di TEST dalla piu' vicina di TRAIN contro il rumore della stima della
neutra (distanza fra le due meta' dei primi fotogrammi).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parents[1] / "v3_work" / "unified_gt"))

import famos_common as fc  # noqa: E402
import ugt as C  # noqa: E402
from shapes import Space, unified_distances  # noqa: E402

UNI_DIR = fc.OUT_ROOT / "unified"
E8 = fc.AAU_DIR / "runs" / "evidence" / "e8" / "evidence.json"


def q(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {"median": float(np.median(x)), "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75)),
            "min": float(x.min()), "max": float(x.max())}


def main() -> None:
    split = fc.load_split()
    sp = Space()
    ids, splits, Vn, halves = [], [], [], []
    for sub, d in (("train", fc.TRAIN_DIR), ("test", fc.TEST_DIR)):
        for s in split[sub]:
            with np.load(d / f"{s}.npz") as z:
                assert str(z["subject"]) == s and str(z["split"]) == sub, f"{d / s}.npz: soggetto o split inattesi"
                Vn.append(z["V_neutral"].astype(np.float64))
                halves.append(float(z["neutral_halves_mm"]))
            ids.append(s)
            splits.append(sub)
    ids, splits, halves = np.asarray(ids), np.asarray(splits), np.asarray(halves)
    a, scale, rms = sp.align(sp.map("flame", np.stack(Vn)))
    S = sp.svec(a)
    D = unified_distances(S, S, sp.A)
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    C.save_npz(UNI_DIR / "famos_shapes.npz", s=S, ids=ids, split=splits, scale=scale, rms_to_mu=rms)
    C.save_npz(UNI_DIR / "gt_all_mm.npz", D_orig=D.astype(np.float32), names=ids)

    # GT dei soggetti di TEST nel formato delle viste (id9400NN, ordinati)
    te = np.flatnonzero(splits == "test")
    te = te[np.argsort([fc.view_id(ids[i]) for i in te])]
    Dt = D[np.ix_(te, te)]
    gmax = float(Dt.max())
    names = np.asarray([fc.view_id(ids[i]) for i in te])
    fc.VIEW_DIR.mkdir(parents=True, exist_ok=True)
    C.save_npz(fc.VIEW_DIR / "gt_matrix.npz", D_orig=(Dt / gmax).astype(np.float32), names=names)
    iu = np.triu_indices(len(te), 1)
    gt_man = {"domain": "famos_test", "n": int(len(te)), "mm_per_unit": gmax,
              "subjects": {str(n): str(ids[i]) for n, i in zip(names, te)},
              "definition": "GT unificata (v3_work/unified_gt): RMS pesato per area in mm sulla regione FLAME comune "
                            "dopo Procrustes di similarita' verso mu, dalla forma neutra di riferimento registrata "
                            "(famos_subsample.py); D_orig * mm_per_unit = mm",
              "median_mm": float(np.median(Dt[iu])), "min_offdiag_mm": float(Dt[iu].min())}
    (fc.VIEW_DIR / "gt_matrix.json").write_text(json.dumps(gt_man, indent=1) + "\n")

    # distanze dalle medie dei domini (come evidence.py part_c)
    doms = [str(x) for x in sp.z["domains"]]
    M = np.stack([sp.svec(sp.z[f"mean_{d}"][None])[0] for d in doms])
    Dm = unified_distances(S, M, sp.A)
    famos_mean = S.astype(np.float64).mean(0)[None].astype(np.float32)
    mean_to_means = unified_distances(famos_mean, M, sp.A)[0]
    fc.EVID_DIR.mkdir(parents=True, exist_ok=True)
    with open(fc.EVID_DIR / "famos_to_domain_means.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "split"] + doms)
        for k in range(len(ids)):
            w.writerow([ids[k], splits[k]] + [f"{x:.3f}" for x in Dm[k]])
        w.writerow(["MEDIA_FAMOS", "all"] + [f"{x:.3f}" for x in mean_to_means])

    # dispersione FaMoS e controllo dello split nello spazio
    Dn = D + np.diag(np.full(len(D), np.inf))
    tr = np.flatnonzero(splits == "train")
    te_all = np.flatnonzero(splits == "test")
    iu_all = np.triu_indices(len(D), 1)
    nn_test_train = Dn[np.ix_(te_all, tr)].min(1)
    spread = {"n": int(len(ids)), "pairwise_median": float(np.median(D[iu_all])), "nn_median": float(np.median(Dn.min(1))),
              "to_famos_mean_median": float(np.median(unified_distances(S, famos_mean, sp.A)[:, 0])),
              "to_flame_mean_median": float(np.median(Dm[:, doms.index("flame")]))}
    others = {}
    if E8.exists():
        e8 = json.loads(E8.read_text())["c"]
        others = {"spread": e8["spread"], "domain_means_mm": dict(zip(e8["domains"], e8["means_mm"]))}
    out = {
        "n_subjects": int(len(ids)), "n_train": int(len(tr)), "n_test": int(len(te_all)),
        "space": {"n_vertices": int(len(sp.mu)), "area_mm2": float(sp.A)},
        "procrustes_scale": q(scale), "rms_to_mu_mm": q(rms),
        "famos_spread_mm": spread,
        "to_domain_means_mm": {d: {"all": q(Dm[:, i]), "train": q(Dm[tr, i]), "test": q(Dm[te_all, i])}
                               for i, d in enumerate(doms)},
        "famos_mean_to_domain_means_mm": dict(zip(doms, map(float, mean_to_means))),
        "split_check_in_space": {
            "neutral_estimate_noise_mm": q(halves),
            "test_to_nearest_train_mm": q(nn_test_train),
            "test_to_nearest_train_per_subject": {str(ids[i]): float(v) for i, v in zip(te_all, nn_test_train)},
            "nearest_train_over_noise_min": float(nn_test_train.min() / np.median(halves)),
        },
        "test_gt": gt_man,
        "other_domains_e8": others,
    }
    (fc.EVID_DIR / "unified_gt.json").write_text(json.dumps(out, indent=1) + "\n")

    md = ["# FaMoS nello spazio della GT unificata", "",
          f"{len(ids)} soggetti (TRAIN {len(tr)}, TEST {len(te_all)}), forma neutra di riferimento registrata "
          f"(famos_subsample.py), regione unificata di {len(sp.mu)} vertici, mm.", "",
          "## Distanza dalle medie dei domini (mm, mediana [p25, p75] sui soggetti)", "",
          "| dominio | FaMoS tutti | FaMoS TRAIN | FaMoS TEST | media FaMoS -> media del dominio |",
          "| --- | --- | --- | --- | --- |"]
    for i, d in enumerate(doms):
        r = out["to_domain_means_mm"][d]
        md.append(f"| {d} | {r['all']['median']:.2f} [{r['all']['p25']:.2f}, {r['all']['p75']:.2f}] | "
                  f"{r['train']['median']:.2f} | {r['test']['median']:.2f} | {mean_to_means[i]:.2f} |")
    md += ["", "## Dispersione (mm)", "", "| dominio | n | mediana a coppie | mediana del vicino piu' prossimo |",
           "| --- | --- | --- | --- |",
           f"| **famos** | {spread['n']} | {spread['pairwise_median']:.2f} | {spread['nn_median']:.2f} |"]
    for d, r in others.get("spread", {}).items():
        md.append(f"| {d} | {r['n']} | {r['pairwise_median']:.2f} | {r['nn_median']:.2f} |")
    sc = out["split_check_in_space"]
    md += ["", "## Split nello spazio", "",
           f"- Rumore della stima della neutra (meta' pari contro dispari dei primi fotogrammi): mediana "
           f"{sc['neutral_estimate_noise_mm']['median']:.3f} mm, massimo {sc['neutral_estimate_noise_mm']['max']:.3f} mm.",
           f"- Persona di TEST -> TRAIN piu' vicina: mediana {sc['test_to_nearest_train_mm']['median']:.2f} mm, minimo "
           f"{sc['test_to_nearest_train_mm']['min']:.2f} mm ({sc['nearest_train_over_noise_min']:.1f} volte il rumore "
           f"mediano): nessun duplicato geometrico.",
           f"- GT dei {len(te)} soggetti di TEST: mediana {gt_man['median_mm']:.2f} mm, minimo fuori diagonale "
           f"{gt_man['min_offdiag_mm']:.2f} mm, mm_per_unit {gmax:.3f}."]
    (fc.EVID_DIR / "unified_gt.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
