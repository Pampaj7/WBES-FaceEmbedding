#!/usr/bin/env python3
"""Identita' FaceScape bilineari (solo forma, neutre) del set di SVILUPPO, nel formato di ``zs_identities.py``.

    aau/run.sh aau/zs3dmm/dev_fs_identities.py --n-identities 500 --seed 1234 \\
        --out-dir datasets/DEV_FACESCAPE/identities
    (dev_fs_build.sbatch)

Gemello di ``zs_identities.py`` per un modello che quel file non conosce, sul caricatore della
libreria ``v3_work/mm`` (``load_model("facescape")``: file v1.6 300_52_id, regione
``fv_indices_front``, 12.596 vertici / 24.765 triangoli, mm, +y alto e +z naso come ICT e HIFI3D).
Stesse uscite: ``fsNNNN.npz`` (``V`` float32 della patch, ``F`` int32, ``weights``),
``identity_weights.json``, ``face_vertex_indices.npy``, ``manifest.json`` con impronta del file,
descrizione del modello e controllo Z1mX.

Coefficienti: z ~ N(0, 1) SENZA troncamento su tutti i 300 modi, come i set zero-shot
(``sample_identity(tails=False, trunc=0)``: stesso flusso di ``rng.normal``), dove z e' il
fattore d'identita' standardizzato (w = id_mean + sqrt(id_var) z: il campionatore del toolkit
FaceScape). ``weights`` sono gli z: su questi ``build_zs_gt.py`` calcola la GT nei coefficienti.

FaceScape e' ``role="dev"`` (PLAN_MASSIVE sez. 14.5): qui si campiona con ``purpose="eval"``; un
generatore di training che lo chiedesse riceverebbe ``RoleError``.

Pool di 500, come gli altri set zero-shot: ``zs_stage.py`` ne estrae 100 (rebuild_subject_split,
seed 1234), e sono quelli valutati.
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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))

from v3_work.mm import load_model  # noqa: E402
from hifi_model import sha256  # noqa: E402
from generate_identities import min_triangle_area  # noqa: E402  (genict)
from pairdist import offdiag_stats, vertex_mean_l2_matrix  # noqa: E402

PREFIX = "fs"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="facescape", help="facescape (300 modi) o facescape50")
    p.add_argument("--n-identities", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-shape-check", action="store_true")
    a = p.parse_args()

    m = load_model(a.model)  # fallisce prima di scrivere qualunque cosa
    print(f"[dev-fs-id] {m.name} ({m.role}): {m.n_verts} vertici, {len(m.faces)} triangoli, {m.n_id} modi, "
          f"regione {m.region}", flush=True)
    rng = np.random.default_rng(a.seed)
    weights = np.stack([m.sample_identity(rng, tails=False, trunc=0.0, purpose="eval")
                        for _ in range(a.n_identities)])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    verts = np.empty((a.n_identities, m.n_verts, 3), dtype=np.float32)
    worst_area = np.inf
    for i, w in enumerate(weights):
        V = m.mesh(w)
        verts[i] = V
        worst_area = min(worst_area, min_triangle_area(V, m.faces))
        np.savez_compressed(a.out_dir / f"{PREFIX}{i:04d}.npz", V=V.astype(np.float32),
                            F=m.faces.astype(np.int32), weights=w)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{a.n_identities}", flush=True)

    (a.out_dir / "identity_weights.json").write_text(json.dumps(
        {f"{PREFIX}{i:04d}": [float(x) for x in w] for i, w in enumerate(weights)}) + "\n")
    np.save(a.out_dir / "face_vertex_indices.npy", m.region_vertices.astype(np.int32))

    model_file = Path(m.info["file"])
    wd = pdist(weights)
    manifest = {
        "seed": a.seed,
        "sigma": 1.0,
        "trunc": 0.0,
        "n_shape": m.n_id,
        "n_identities": a.n_identities,
        "domain": "facescape",
        "prefix": PREFIX,
        "role": m.role,
        "sampler": "v3_work.mm sample_identity(tails=False, trunc=0, purpose='eval')",
        "model_file": str(model_file),
        "model_sha256": sha256(model_file),
        "model": m.describe(),
        "n_verts": m.n_verts,
        "n_faces": int(len(m.faces)),
        "frame": "facescape: +x sinistra del soggetto, +y alto, +z naso, mm (convenzione ICT)",
        "min_triangle_area": worst_area,
        "weight_l2": {"min": float(wd.min()), "p1": float(np.percentile(wd, 1)),
                      "median": float(np.median(wd)), "max": float(wd.max())},
    }
    print("weight-space L2: " + json.dumps(manifest["weight_l2"]), flush=True)
    if not worst_area > 0:
        raise SystemExit(f"triangolo degenere in almeno un'identita' (area minima {worst_area})")
    if not a.skip_shape_check:
        D = vertex_mean_l2_matrix(verts, device=a.device)
        manifest["vertex_mean_l2"] = offdiag_stats(D)
        print("shape-space vertex-mean-L2 (mm): " + json.dumps(manifest["vertex_mean_l2"]), flush=True)
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"scritte {a.n_identities} identita' in {a.out_dir}")


if __name__ == "__main__":
    main()
