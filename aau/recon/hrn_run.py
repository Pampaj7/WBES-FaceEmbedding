#!/usr/bin/env python3
"""HRN: da una directory di immagini alla mesh finale (dettagli medi e alti) per immagine (NoW ranking).

  aau/recon/run_recon.sh hrn aau/recon/hrn_run.py --images <dir> --out <dir>

``Reconstructor.predict`` di ``models/hrn.py`` come ``run_hrn`` di demo.py (vista singola, hrn_v1.1 epoca 10,
``visualize=True``: solo con quello ``forward_hrn`` riempie ``extra_results``), senza scrivere file. La mesh e' quella che ``save_results`` scrive come ``*_hrn_high_mesh.obj``:
``extra_results['dense_mesh']`` (mesh posata BFM + deformazione media + spostamento alto lungo le normali, sulla
griglia UV), tolti i vertici a zero fuori dalla mappa UV e le facce che li toccano. I 7 landmark vengono dai 68
vertici chiave (``facemodel_front.keypoints``) della mesh media (``pred_vertex``, stesso frame). Unita' e assi come
3DDFA-V3: BFM di Deep3DFaceRecon, 1 = 100 mm; destrorsi, x a destra, y in su, z verso la camera (``pred_vertex`` con
10 - z, come in ``save_results``).

Nessuna riga del clone cambiata. Interventi da fuori, nessuno sui numeri:
  * ``nvdiffrast.torch.RasterizeGLContext`` -> ``RasterizeCudaContext``: il container non ha OpenGL; stessa
    rasterizzazione (README di nvdiffrast), e i render servono solo alle tessiture e alle immagini del demo;
  * le facce della mesh alta si ricompongono qui: ``generate_triangles`` (DECA) le da' 0-based, ``crop_mesh`` del clone
    le tratta come 1-based (``faces -= 1``), e la mesh scritta dal demo ha la triangolazione spostata di un vertice;
  * ``face_alignment.LandmarksType._3D`` (API vecchia usata da hrn.py) -> ``THREE_D`` se il pacchetto installato non
    ha piu' il nome vecchio: stesso rilevatore 3D a 68 punti;
  * numpy >= 1.24 (quello del container): ``util.preprocess.POS`` restituisce t di forma (2, 1) e ``align_img`` ne fa
    un array con ``np.array([w0, h0, s, t[0], t[1]])``, che numpy nuovo rifiuta (array disomogeneo; il numpy 1.18
    dei requisiti lo convertiva). Qui POS restituisce t appiattito a (2,): stessi valori; piu'
    ``common.patch_numpy_aliases()`` come per 3DDFA_V2;
  * ``predict`` fallisce con KeyError se non trova il volto (``predict_base`` restituisce {}): immagine mancante.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

METHOD = "hrn"
MM_PER_UNIT = 100.0
COORD_SYSTEM = "BFM model units (1 = 100 mm), right-handed: x right, y up, z towards the camera (posed, demo obj)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", type=Path, required=True, help="directory di jpg/png")
    p.add_argument("--out", type=Path, default=common.OUT_ROOT / METHOD)
    return p.parse_args()


def high_mesh(dense_V: np.ndarray, dense_F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertici non nulli e facce coi tre vertici tenuti, reindicizzate (intento di save_results, 0-based)."""
    keep = ~(dense_V == 0.0).all(axis=1)
    new = -np.ones(len(dense_V), dtype=np.int64)
    new[keep] = np.arange(int(keep.sum()))
    F = dense_F[keep[dense_F].all(axis=1)]
    return dense_V[keep], new[F]


def main() -> None:
    args = parse_args()
    images = common.list_images(args.images.resolve())
    out = args.out.resolve()

    repo = common.add_repo_to_path(METHOD)
    # Asset e checkpoint ("assets/pretrained_models/...") relativi alla radice del clone.
    os.chdir(repo)
    os.environ.setdefault("MPLBACKEND", "Agg")

    import cv2  # noqa: E402
    import nvdiffrast.torch as dr  # noqa: E402
    import torch  # noqa: E402

    dr.RasterizeGLContext = lambda *a, **k: dr.RasterizeCudaContext(device=k.get("device"))
    import face_alignment  # noqa: E402

    if not hasattr(face_alignment.LandmarksType, "_3D"):
        from types import SimpleNamespace
        face_alignment.LandmarksType = SimpleNamespace(_3D=face_alignment.LandmarksType.THREE_D)
    common.patch_numpy_aliases()
    import util.preprocess as pre  # noqa: E402

    pos = pre.POS

    def pos_flat(xp, x):
        t, s = pos(xp, x)
        return np.asarray(t).reshape(-1), s

    pre.POS = pos_flat
    from models.hrn import Reconstructor  # noqa: E402

    rec = Reconstructor(["--checkpoints_dir", "assets/pretrained_models", "--name", "hrn_v1.1", "--epoch", "10"])
    fm = rec.model.facemodel_front
    kp = np.asarray(fm.keypoints)
    print(f"[{METHOD}] hrn_v1.1 epoca 10; {len(kp)} vertici chiave", flush=True)

    n_ok = 0
    times = []
    for img_fp in images:
        t0 = time.time()
        img = cv2.imread(str(img_fp))
        if img is None:
            print(f"[{METHOD}] ILLEGGIBILE {img_fp}", flush=True)
            continue
        try:
            rec.predict(img, visualize=True)
        except KeyError:
            print(f"[{METHOD}] NESSUN VOLTO {img_fp.name}", flush=True)
            continue
        with torch.no_grad():
            mid = rec.model.pred_vertex.detach()[0].cpu().numpy().astype(np.float64)
            mid[:, 2] = fm.camera_distance - mid[:, 2]
            dense = rec.model.extra_results["dense_mesh"]
            dV = dense["vertices"][0].detach().cpu().numpy().astype(np.float64)
            dF = dense["faces"][0].detach().cpu().numpy().astype(np.int64)
        V, F = high_mesh(dV, dF)
        elapsed = time.time() - t0

        common.save_mesh(out, img_fp.stem, V, F)
        common.save_meta(out, img_fp.stem, {
            "method": METHOD,
            "image": str(img_fp),
            "image_shape": [int(img.shape[0]), int(img.shape[1])],
            "mesh_kind": "hrn_high_mesh (dense_mesh di save_results, vertici a zero tolti)",
            "coordinate_system": COORD_SYSTEM,
            "units": "BFM (Deep3DFaceRecon), 1 = 100 mm",
            "mm_per_unit": MM_PER_UNIT,
            "landmarks_68": common.to_list(mid[kp]),
            "landmarks_source": "vertici chiave della mesh media (pred_vertex)",
            "mesh": common.mesh_stats(V, F),
            "seconds": elapsed,
        })
        times.append(elapsed)
        n_ok += 1
        print(f"[{METHOD}] {img_fp.name}: {V.shape[0]} vertici, {F.shape[0]} facce, {elapsed:.3f}s", flush=True)

    if not times:
        raise SystemExit(f"[{METHOD}] nessuna immagine ricostruita")
    print(f"[{METHOD}] {n_ok}/{len(images)} immagini, "
          f"{np.mean(times):.3f}s/immagine (mediana {np.median(times):.3f}s), out={out}",
          flush=True)


if __name__ == "__main__":
    main()
