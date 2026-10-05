#!/usr/bin/env python3
"""Patch del volto delle identita' FLAME di ``v2_work/genflame/generate_identities.py``.

    aau/run.sh aau/flame/flame_crop.py --in-dir datasets/FLAME/identities_full \
        --out-dir datasets/FLAME/identities

Il generatore storico (importato com'e', con ``--frame flame``) scrive teste intere da 5023
vertici; la pipeline storica faceva il crop dentro ``make_flame_topologies.py`` con la
corrispondenza BFM->FLAME, che qui non si puo' usare (vedi ``flame_mask.py``). Questo passo
applica il crop della maschera ufficiale e scrive ``flameNNNN.npz`` nel formato di
``v2_work/genict/generate_identities.py`` (``V`` float32 della patch, ``F`` int32, ``betas``),
cioe' l'ingresso di ``make_flame_topologies.py``, che da li' in poi e' la catena ICT.

Il manifest e' quello del generatore piu' il crop e le impronte sha256 di modello e
maschera, cosi' si sa su quali file sono stati generati i dati (FLAME 2020 ufficiale:
``FLAME2020_SHA256`` qui sotto, da ``v2_work/genflame/flame_model.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from flame_mask import CROPS, compact, crop_faces, masks_path  # noqa: E402

FLAME2020_SHA256 = "efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--in-dir", type=Path, required=True, help="flameNNNN.npz a testa intera")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--crop", default="mask", choices=CROPS,
                   help="mask = regione della maschera ufficiale (protocollo); head = testa intera")
    p.add_argument("--mask-region", default="face")
    a = p.parse_args()

    src_manifest = json.loads((a.in_dir / "manifest.json").read_text())
    # Frame canonico FLAME, come ICT tiene il suo: il modello ha l'xyz fra le feature, e il
    # frame "render" del generatore (y e z negate) non e' quello di nessun dato di training.
    if src_manifest.get("frame") != "flame":
        raise SystemExit(f"identita' in frame '{src_manifest.get('frame')}': serve --frame flame")

    files = sorted(a.in_dir.glob("flame[0-9]*.npz"))
    if not files:
        raise SystemExit(f"nessun flame*.npz in {a.in_dir}")
    with np.load(files[0]) as d:
        F_full, n_verts = np.asarray(d["F"], dtype=np.int32), len(d["V"])
    F_crop = crop_faces(F_full, n_verts, a.crop, a.mask_region)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    betas = {}
    for q in files:
        with np.load(q) as d:
            if not np.array_equal(d["F"], F_full):
                raise SystemExit(f"{q.name}: topologia diversa dalle altre identita'")
            V, F = compact(np.asarray(d["V"], dtype=np.float64), F_crop)
            b = np.asarray(d["betas"], dtype=np.float64)
        np.savez_compressed(a.out_dir / q.name, V=V.astype(np.float32), F=F.astype(np.int32), betas=b)
        betas[q.stem] = [float(x) for x in b]
    (a.out_dir / "identity_betas.json").write_text(json.dumps(betas) + "\n")
    np.save(a.out_dir / "crop_vertex_indices.npy", np.unique(F_crop).astype(np.int32))

    mfile = Path(src_manifest["model_file"])
    manifest = dict(src_manifest)
    manifest.update({
        "source_identities": str(a.in_dir),
        "model_sha256": sha256(mfile),
        "crop": a.crop,
        "mask_file": str(masks_path()) if a.crop == "mask" else None,
        "mask_sha256": sha256(masks_path()) if a.crop == "mask" else None,
        "mask_region": a.mask_region if a.crop == "mask" else None,
        "n_verts": int(len(V)),
        "n_faces": int(len(F)),
    })
    manifest["model_is_official_flame2020"] = manifest["model_sha256"] == FLAME2020_SHA256
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[flame-crop] {len(files)} identita', crop {a.crop}: {len(V)} vertici, {len(F)} triangoli "
          f"-> {a.out_dir}", flush=True)
    if not manifest["model_is_official_flame2020"]:
        print(f"[flame-crop] NOTA: sha256 del modello {manifest['model_sha256']} diverso da FLAME 2020 "
              "ufficiale (atteso solo con FLAME 2023 o col modello finto di prova)", flush=True)


if __name__ == "__main__":
    main()
