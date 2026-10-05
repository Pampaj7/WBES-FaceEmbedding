#!/usr/bin/env python3
"""Identita' gia' esistenti come ingresso di zs3dmm (dominio ``ict``, il controllo della pipeline).

    aau/run.sh aau/zs3dmm/zs_link_identities.py --source-dir datasets/ICT/identities \
        --first 4500 --n 500 --out-dir datasets/ICT_ZS/identities

Al posto di ``zs_identities.py``: nessun campionamento, symlink a ``ict4500.npz`` ...
``ict4999.npz`` (le 500 held-out di ICT-5000, chiavi ``V``/``F``/``weights``), il loro
``identity_weights.json`` filtrato e un ``manifest.json`` che dice da dove vengono (entra
nell'impronta dei dati di zs_env.sh).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in a.out_dir.glob("ict[0-9]*.npz"):
        stale.unlink()
    names = [f"ict{i:04d}" for i in range(a.first, a.first + a.n)]
    for n in names:
        src = a.source_dir / f"{n}.npz"
        if not src.is_file():
            raise SystemExit(f"identita' mancante: {src}")
        (a.out_dir / src.name).symlink_to(src.resolve())
    weights = json.loads((a.source_dir / "identity_weights.json").read_text())
    (a.out_dir / "identity_weights.json").write_text(json.dumps({n: weights[n] for n in names}) + "\n")
    src_manifest = json.loads((a.source_dir / "manifest.json").read_text())
    manifest = {"domain": "ict", "prefix": "ict", "linked_from": str(a.source_dir),
                "range": [names[0], names[-1]], "n_identities": len(names),
                "source_manifest": {k: src_manifest[k] for k in ("seed", "sigma", "trunc", "n_shape",
                                                                 "n_identities", "n_verts", "n_faces")},
                "n_shape": src_manifest["n_shape"], "seed": src_manifest["seed"],
                "model_file": src_manifest["model_dir"], "model_sha256": "n/a (ICT-FaceKit, OBJ)",
                "n_verts": src_manifest["n_verts"], "n_faces": src_manifest["n_faces"],
                "model": {"scaling": "come ICT-5000 (N(0,1) sui 100 modi)", "n_verts_head": 26719,
                          "region": "geometria #0 Face del README ICT (come ICT-5000)"}}
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
