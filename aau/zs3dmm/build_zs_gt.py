#!/usr/bin/env python3
"""Matrici GT delle identita' di un dominio zero-shot: quella del protocollo ICT e quella nei coefficienti.

    aau/run.sh aau/zs3dmm/build_zs_gt.py --prefix hifi --topo-dir datasets/HIFI3D/topo \
        --identities-dir datasets/HIFI3D/identities --out-dir datasets/HIFI3D/gt

1. ``<prefix>_matrix_distances_{raw,maxabs}.npz``: gemello di
   ``v2_work/genict/build_ict_gt_matrix.py`` (stesse funzioni, importate). Media per vertice
   della distanza L2 fra le ``original`` di due identita', in corrispondenza densa per
   costruzione; ``maxabs`` dopo la normalizzazione per mesh di ``GTReadyDatasetNPZ`` (centro
   sulla media, divisione per il max |coordinata|). La GT degli eval ICT e' la ``maxabs``
   (``datasets/ICT/train_ready/gt_matrix.npz``): e' la GT primaria anche qui, altrimenti i
   numeri del dominio nuovo non si confrontano con la tabella WS2.
2. ``<prefix>_coef_distances.npz``: distanza L2 fra i vettori dei coefficienti di identita'
   standardizzati ``z`` (quelli estratti da N(0, 1), prima della scala per la deviazione
   standard del modello), cioe' la distanza di Mahalanobis nel prior del 3DMM. Seconda GT.
   Non coincide con la prima: la patch e' solo una parte della testa, e la vertex-mean-L2 e'
   una media di norme dopo maxabs. Il manifest dice di quanto (Spearman fra le due).

Chiavi come la matrice BFM: ``D_orig`` float32 simmetrica, diagonale zero, divisa per il suo
massimo, e ``names`` (``<prefix>NNNN``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))

from build_ict_gt_matrix import normalize_maxabs  # noqa: E402
from pairdist import offdiag_stats, vertex_mean_l2_matrix  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--topo-dir", type=Path, required=True)
    ap.add_argument("--identities-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--variant", default="original")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    files = sorted(args.topo_dir.glob(f"{args.prefix}[0-9]*_GTready_{args.variant}.npz"))
    if not files:
        raise SystemExit(f"nessuna mesh {args.variant} in {args.topo_dir}")

    names, raw, nrm = [], [], []
    for p in files:
        with np.load(p) as d:
            V = (d["verts"] if "verts" in d else d["V"]).astype(np.float64)
        names.append(p.name.split("_GTready")[0])
        raw.append(V)
        nrm.append(normalize_maxabs(V))
    n = len(names)
    print(f"{n} identita', {raw[0].shape[0]} vertici ciascuna", flush=True)

    out = {}
    for tag, verts in (("raw", raw), ("maxabs", nrm)):
        D = vertex_mean_l2_matrix(np.stack(verts, axis=0), device=args.device)
        scale = float(D[D > 0].max())
        out[tag] = (D / scale, scale)
        s = offdiag_stats(D)
        print(f"[{tag}] max={s['max']:.6g} mean={np.mean(D[np.triu_indices(n, 1)]):.6g} "
              f"min={s['min']:.6g} p1={s['p1']:.6g}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag, (Dn, scale) in out.items():
        np.savez(args.out_dir / f"{args.prefix}_matrix_distances_{tag}.npz",
                 D_orig=Dn.astype(np.float32), names=np.array(names))

    # Distanza nei coefficienti standardizzati, nello stesso ordine di `names`.
    coefs = json.loads((args.identities_dir / "identity_weights.json").read_text())
    missing = [s for s in names if s not in coefs]
    if missing:
        raise SystemExit(f"{len(missing)} identita' senza coefficienti, p.es. {missing[:3]}")
    Z = np.asarray([coefs[s] for s in names], dtype=np.float64)
    Dz = squareform(pdist(Z))
    np.savez(args.out_dir / f"{args.prefix}_coef_distances.npz",
             D_orig=(Dz / Dz.max()).astype(np.float32), names=np.array(names))

    iu = np.triu_indices(n, 1)
    meta = {
        "n_identities": n,
        "variant": args.variant,
        "normalization_scale": {k: v[1] for k, v in out.items()},
        "spearman_raw_vs_maxabs": float(spearmanr(out["raw"][0][iu], out["maxabs"][0][iu]).statistic),
        "spearman_coef_vs_raw": float(spearmanr(Dz[iu], out["raw"][0][iu]).statistic),
        "spearman_coef_vs_maxabs": float(spearmanr(Dz[iu], out["maxabs"][0][iu]).statistic),
        "closest_pair_frac_of_median": float(
            out["raw"][0][iu].min() / np.median(out["raw"][0][iu])
        ),
        "offdiag_stats_unnormalized": {k: offdiag_stats(v[0] * v[1]) for k, v in out.items()},
        "offdiag_stats_coef": offdiag_stats(Dz),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
