#!/usr/bin/env python3
"""Identita' sintetiche (solo forma, neutra) di un 3DMM mai visto, per il test zero-shot.

    aau/run.sh aau/zs3dmm/zs_identities.py --domain hifi --model-file <AI-NEXT-Shape.mat> \
        --n-identities 500 --out-dir datasets/HIFI3D/identities
    aau/run.sh aau/zs3dmm/zs_identities.py --domain fv --model-file <faceverse_simple_v2.npy> ...

Gemello di ``v2_work/genict/generate_identities.py``: scrive ``<prefix>0000.npz`` ... (chiavi
``V`` float32 della patch del volto, ``F`` int32, ``weights``), un ``identity_weights.json``
con tutti i vettori e un ``manifest.json`` con parametri, impronta del file del modello,
descrizione del modello (layout, scala, regione: ``hifi_model`` / ``fv_model``) e controllo Z1mX.
Prefisso ``hifi`` o ``fv``.

Coefficienti
------------
Come ICT: tutti i modi del modello, N(0, 1) senza troncamento (``--trunc 0``). Per HIFI3D e'
anche il campionatore ufficiale (``test_basis_io.py``: ``np.random.normal(size=[1, n_basis])``).
Le basi di entrambi i modelli sono gia' scalate per la deviazione standard dei modi (verificato
sulle norme in ``hifi_model`` / ``fv_model``): ``weights`` sono i coefficienti STANDARDIZZATI z,
ed e' su questi che ``build_zs_gt.py`` calcola la GT nei coefficienti.

Frame e unita'
--------------
Il frame del modello resta com'e', come ICT tiene il suo. Le UNITA' non entrano nei numeri
(maxabs per GT e Chamfer, area unitaria per gli operatori). L'ORIENTAMENTO SI': il modello
(xyz_dn) riceve le coordinate xyz come feature d'ingresso, e in training ha visto solo il frame
dei suoi dati (BFM: y in basso, naso verso -z; ICT: y in alto, naso verso +z) con rotazioni di
pochi gradi. HIFI3D e' nel frame ICT, FaceVerse in quello BFM (misurato, aau/scratch/hifi3d/
frames.py). Operatori intrinseci, GT per vertice e Chamfer invece non dipendono dal frame.
Il test del frame e' WBES_ZS_FRAME in zs_zeroshot.sbatch.

Pool
----
500 identita' = il pool held-out ICT (``eval_view_heldout``). Nessun modello ha visto questi
3DMM: lo zero-shot estrae dal pool 100 soggetti con lo stesso ``rebuild_subject_split`` e gli
stessi argomenti dello zero-shot ICT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))

import fv_model  # noqa: E402
import hifi_model  # noqa: E402
from hifi_model import face_patch, sha256  # noqa: E402
from generate_identities import min_triangle_area, sample_weights  # noqa: E402  (genict)
from pairdist import offdiag_stats, vertex_mean_l2_matrix  # noqa: E402

DOMAINS = {
    "hifi": ("hifi", hifi_model.load_hifi, hifi_model.shape_mesh),
    "fv": ("fv", fv_model.load_fv, fv_model.shape_mesh),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    p.add_argument("--model-file", type=Path, required=True)
    p.add_argument("--n-identities", type=int, default=500)
    p.add_argument("--n-shape", type=int, default=0, help="0 = tutti i modi del .mat")
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--trunc", type=float, default=0.0,
                   help="0 = N(0,1) senza troncamento, come ICT e come il campionatore HIFI3D")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="auto", help="device torch per il controllo in spazio forma")
    p.add_argument("--skip-shape-check", action="store_true")
    a = p.parse_args()

    prefix, load, shape_mesh = DOMAINS[a.domain]
    model = load(a.model_file, n_shape=a.n_shape)  # fallisce prima di scrivere qualunque cosa
    info = model["info"]
    k = int(model["shapedirs"].shape[2])
    print(f"[zs-id] {a.domain} {a.model_file}: v={info['n_verts_head']} f={info['n_faces_head']}, "
          f"modi {k}/{info['n_basis_in_file']}, layout {info['layout']}, scala: {info['scaling']}; "
          f"regione del volto: {info['face_vertices']} vertici, {info['face_faces_largest_component']} "
          f"triangoli", flush=True)

    rng = np.random.default_rng(a.seed)
    weights = sample_weights(rng, a.n_identities, k, a.sigma, a.trunc)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    verts = np.empty((a.n_identities, len(model["face_vertices"]), 3), dtype=np.float32)
    worst_area = np.inf
    F_out = None
    for i, w in enumerate(weights):
        V, F = face_patch(shape_mesh(w, model), model)
        F_out = F
        verts[i] = V
        worst_area = min(worst_area, min_triangle_area(V, F))
        np.savez_compressed(a.out_dir / f"{prefix}{i:04d}.npz", V=V.astype(np.float32),
                            F=F.astype(np.int32), weights=w)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{a.n_identities}", flush=True)

    (a.out_dir / "identity_weights.json").write_text(json.dumps(
        {f"{prefix}{i:04d}": [float(x) for x in w] for i, w in enumerate(weights)}) + "\n")
    np.save(a.out_dir / "face_vertex_indices.npy", model["face_vertices"])

    wd = pdist(weights)
    manifest = {
        "seed": a.seed,
        "sigma": a.sigma,
        "trunc": a.trunc,
        "n_shape": k,
        "n_identities": a.n_identities,
        "domain": a.domain,
        "prefix": prefix,
        "model_file": str(a.model_file),
        "model_sha256": sha256(a.model_file),
        "model": info,
        "n_verts": int(verts.shape[1]),
        "n_faces": int(len(F_out)),
        "frame": a.domain,
        "min_triangle_area": worst_area,
        "weight_l2": {"min": float(wd.min()), "p1": float(np.percentile(wd, 1)),
                      "median": float(np.median(wd)), "max": float(wd.max())},
    }
    print("weight-space L2: " + json.dumps(manifest["weight_l2"]), flush=True)
    if not worst_area > 0:
        raise SystemExit(f"triangolo degenere in almeno un'identita' (area minima {worst_area})")

    if not a.skip_shape_check:
        print("shape-space vertex-mean-L2 su tutte le coppie...", flush=True)
        D = vertex_mean_l2_matrix(verts, device=a.device)
        manifest["vertex_mean_l2"] = offdiag_stats(D)
        print("shape-space vertex-mean-L2: " + json.dumps(manifest["vertex_mean_l2"]), flush=True)

    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"scritte {a.n_identities} identita' in {a.out_dir}")


if __name__ == "__main__":
    main()
