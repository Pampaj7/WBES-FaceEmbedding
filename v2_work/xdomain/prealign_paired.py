#!/usr/bin/env python
"""Quanto del danno attribuito all'ICP rigido viene dal prealign, e non dall'ICP?

Il paper sostiene che la registrazione degrada la fedelta' del ranking d'identita', e il numero
principale e' l'ICP rigido che porta lo Spearman da 0,7295 a 0,5995 (original->original). Ma la
pipeline faceBench, prima di ogni ICP, chiama `prealign_by_bbox`
(faceBench/facebench/rigid_aligners/icp.py:109-116):

    source_scaled = source_centered / source_scale * target_scale

che riscala la sorgente al raggio del BERSAGLIO. Per la coppia ordinata (A,B) la trasformazione
dipende da B: cambia a ogni coppia. Non e' applicata a raw_chamfer, che quindi non e' nelle
stesse unita' delle colonne registrate.

Un confronto contro una pipeline diversa (`registration_utils.py`, che non fa prealign)
suggeriva che senza prealign il danno scende da 0,130 a 0,037. Ma quel confronto era su un set
di coppie diverso e con un'aggregazione diversa, quindi il fattore non era difendibile.

Qui si misura APPAIATO: le stesse coppie, lo stesso ICP, la stessa normalizzazione per-mesh
della pipeline, e come unica differenza il prealign acceso o spento. La differenza appaiata
rimuove per costruzione il confronto fra set diversi che rendeva il numero precedente inutile.

Il bersaglio resta `raw` (cioe' D_GT come effettivamente salvato), perche' l'oggetto di studio
qui e' il prealign, non il frame: cambiare due cose insieme non permetterebbe di attribuire
nulla.
"""
from __future__ import annotations

import argparse, csv, itertools, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "faceBench"))
sys.path.insert(0, str(REPO / "faceBench" / "latentVSpipeline"))

TOPOS = ["crop", "down8k", "noisy", "original", "remesh", "up60k"]


def _work(job):
    """Un lavoro = una coppia (mesh_a, mesh_b). Importato nei worker, non nel padre."""
    import facebench as fb
    from mesh_npz_utils import load_normalized_vertices_npz
    from run_facebench_remesh import sample_pts, symmetric_chamfer

    pa, pb, npts, seed = job
    try:
        X = load_normalized_vertices_npz(pa)
        Y = load_normalized_vertices_npz(pb)
        Xs, Ys = sample_pts(X, npts, seed), sample_pts(Y, npts, seed + 1)

        raw = symmetric_chamfer(Xs, Ys)
        out = {"raw_chamfer": raw}
        for tag, pre in (("rigid_bbox", "bbox"), ("rigid_noprealign", None)):
            Xa, _ = fb.icp_align(Xs, Ys, prealign=pre)
            corr = fb.chamfer_correspondence(Xa, Ys)
            out[tag] = float(np.mean(fb.p2p_distance(Xa, Ys, corr)))
        return out
    except Exception as e:                       # una coppia rotta non deve fermare la corsa
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-dir", type=Path,
                    default=REPO / "datasets" / "REMESH" / "npz_data_topo_500_withops")
    ap.add_argument("--n-subjects", type=int, default=40)
    ap.add_argument("--max-points", type=int, default=4096)
    ap.add_argument("--procs", type=int, default=32, help="meta' dei core del nodo al massimo")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "gt_matrices" / "prealign_paired.csv")
    args = ap.parse_args()

    # le stesse 100 identita' di valutazione del paper, poi le prime n
    rows = [r for d in sorted((REPO / "v2_work/phase0/extended_pair_metrics").iterdir())
              for r in csv.DictReader(open(d / "pair_metrics.csv"))]
    evalsub = sorted({r["subject_a"] for r in rows} | {r["subject_b"] for r in rows})
    sids = evalsub[:args.n_subjects]

    z = np.load(REPO / "face_embedding/gt_encdec/autoencoder/latent_analysis"
                     / "gt_distance_matrix/normalized_matrix_distances.npz")
    D, names = z["D_orig"], list(z["names"])
    gi = {n: i for i, n in enumerate(names)}
    missing = [s for s in sids if f"{s}_GTready" not in gi]
    if missing:
        raise SystemExit(f"identita' assenti dalla matrice GT: {missing[:5]}")

    subj_pairs = list(itertools.combinations(sids, 2))
    topo_pairs = [(a, b) for a in TOPOS for b in TOPOS if a != b]
    print(f"{len(sids)} identita', {len(subj_pairs)} coppie di soggetti, "
          f"{len(topo_pairs)} coppie di topologie -> {len(subj_pairs)*len(topo_pairs)} lavori "
          f"su {args.procs} processi", flush=True)

    jobs, meta = [], []
    for ta, tb in topo_pairs:
        for sa, sb in subj_pairs:
            jobs.append((str(args.mesh_dir / f"{sa}_GTready_{ta}.npz"),
                         str(args.mesh_dir / f"{sb}_GTready_{tb}.npz"), args.max_points, 0))
            meta.append((f"{ta}__to__{tb}", sa, sb))

    from multiprocessing import Pool
    with Pool(args.procs) as pool:
        res = pool.map(_work, jobs, chunksize=8)

    bad = [r for r in res if "error" in r]
    if bad:
        print(f"ATTENZIONE: {len(bad)} coppie fallite, es. {bad[0]['error']}", flush=True)
    if len(bad) > 0.02 * len(res):
        raise SystemExit(f"troppe coppie fallite ({len(bad)}/{len(res)}): risultato non usabile")

    METRICS = ["raw_chamfer", "rigid_bbox", "rigid_noprealign"]
    out = []
    for label in sorted({m[0] for m in meta}):
        idx = [i for i, m in enumerate(meta) if m[0] == label and "error" not in res[i]]
        gt = np.array([D[gi[f"{meta[i][1]}_GTready"], gi[f"{meta[i][2]}_GTready"]] for i in idx])
        rec = {"pair_label": label, "n_pairs": len(idx)}
        for k in METRICS:
            rec[k] = float(spearmanr(gt, np.array([res[i][k] for i in idx])).statistic)
        out.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair_label", "n_pairs"] + METRICS)
        w.writeheader(); w.writerows(out)

    a = np.array([r["raw_chamfer"] for r in out])
    b = np.array([r["rigid_bbox"] for r in out])
    c = np.array([r["rigid_noprealign"] for r in out])
    print(f"\nmedia su {len(out)} coppie di topologie:")
    print(f"  raw chamfer               {a.mean():.4f}")
    print(f"  rigid ICP + prealign bbox {b.mean():.4f}   danno {(b-a).mean():+.4f}"
          f"   peggiora in {int((b<a).sum())}/{len(out)}")
    print(f"  rigid ICP senza prealign  {c.mean():.4f}   danno {(c-a).mean():+.4f}"
          f"   peggiora in {int((c<a).sum())}/{len(out)}")
    d_pre, d_no = (a - b).mean(), (a - c).mean()
    print(f"\ndifferenza appaiata (stesse coppie, unica variabile il prealign): "
          f"{(c-b).mean():+.4f}")
    if d_no > 1e-6:
        print(f"quota del danno attribuibile al prealign: {1 - d_no/d_pre:.1%}" if d_pre > 0 else "")
    print(f"\nscritto in {args.out}")


if __name__ == "__main__":
    main()
