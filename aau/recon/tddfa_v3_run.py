#!/usr/bin/env python3
"""3DDFA-V3: da una directory di immagini a una mesh densa BFM per immagine (NoW ranking).

  aau/recon/run_recon.sh ddfa3 aau/recon/tddfa_v3_run.py --images <dir> --out <dir> --device cuda

Topologia: BFM "front" di Deep3DFaceRecon, 35709 vertici e 70789 triangoli, fissa per tutte le immagini.
Unita': quelle del modello, 1 = 100 mm (Deep3DFaceRecon ``util/load_mats.py`` divide le basi BFM09 per 1e5, "unify
the scale to decimeter"; controllo ex-ex della forma media qui sotto, ``MM_PER_UNIT``). Assi: destrorsi, x a destra,
y in su, z verso la camera: e' la mesh posata (rotazione e traslazione stimate) PRIMA di ``to_camera``, cioe' le
coordinate dell'obj che scrive ``util/io.py`` del demo (che rifa' 10 - z).

Si usa il codice del clone senza modificarlo: il detector del demo (``face_box``, retinaface + 106 landmark e
allineamento a 224x224) e di ``face_model.forward()`` la sola parte geometrica, nella stessa sequenza:
``net_recon -> split_alpha -> compute_shape -> compute_rotation -> transform``. Il resto di forward() (render con
nvdiffrast, tessiture, segmentazione) serve alle immagini del demo e non tocca la geometria. Due stub, che sollevano
se usati: ``util.nv_diffrast`` (``face_model.__init__`` costruisce il renderer su cuda, qui mai chiamato) e ``mtcnn``
(``face_box/__init__.py`` lo importa in testa per il detector alternativo, che non si usa).
I 68 landmark sono vertici del modello (``model['ldm68']``), come per 3DDFA_V2.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

METHOD = "3ddfa_v3"
MM_PER_UNIT = 100.0
EX_EX_MM = (80.0, 110.0)   # REGOLE.md sez. 2: intervallo atteso della distanza ex-ex della forma media
COORD_SYSTEM = "BFM model units (1 = 100 mm), right-handed: x right, y up, z towards the camera (posed, demo obj)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", type=Path, required=True, help="directory di jpg/png")
    p.add_argument("--out", type=Path, default=common.OUT_ROOT / METHOD)
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    return p.parse_args()


def install_stubs() -> None:
    def refuse(*a, **k):
        raise RuntimeError("stub di tddfa_v3_run.py: render e detector mtcnn non disponibili")

    class MeshRenderer:  # costruito da face_model.__init__ su cuda, mai chiamato qui
        def __init__(self, *a, **k):
            pass

        __call__ = refuse

    nv = types.ModuleType("util.nv_diffrast")
    nv.MeshRenderer = nv.MeshRenderer_UV = MeshRenderer
    sys.modules["util.nv_diffrast"] = nv
    mt = types.ModuleType("mtcnn")
    mt.MTCNN = refuse
    sys.modules["mtcnn"] = mt


def main() -> None:
    args = parse_args()
    images = common.list_images(args.images.resolve())
    out = args.out.resolve()

    repo = common.add_repo_to_path(METHOD)
    # I path degli asset ("./assets/face_model.npy", "assets/net_recon.pth") sono relativi alla radice del clone.
    os.chdir(repo)
    install_stubs()

    import torch  # noqa: E402
    from PIL import Image  # noqa: E402
    from face_box import face_box  # noqa: E402
    from model.recon import face_model  # noqa: E402

    opts = SimpleNamespace(device=args.device, iscrop=True, detector="retinaface", backbone="resnet50",
                           ldm68=True, ldm106=False, ldm106_2d=False, ldm134=False, seg=False, seg_visible=False,
                           useTex=False, extractTex=False)
    recon = face_model(opts)
    detector = face_box(opts).detector

    F = recon.tri.cpu().numpy().astype(np.int32)
    lmk_idx = recon.ldm68.cpu().numpy()
    mean = recon.u.cpu().numpy().reshape(-1, 3)
    ex_ex = float(np.linalg.norm(mean[lmk_idx[36]] - mean[lmk_idx[45]]) * MM_PER_UNIT)
    print(f"[{METHOD}] device={args.device} topologia: {mean.shape[0]} vertici, {F.shape[0]} facce; "
          f"ex-ex della forma media {ex_ex:.1f} mm con {MM_PER_UNIT:g} mm/unita'", flush=True)
    if not EX_EX_MM[0] <= ex_ex <= EX_EX_MM[1]:
        raise SystemExit(f"[{METHOD}] ex-ex {ex_ex:.1f} mm fuori da {EX_EX_MM}: unita' del modello da rivedere")

    n_ok = 0
    times = []
    for img_fp in images:
        t0 = time.time()
        im = Image.open(img_fp).convert("RGB")
        try:
            trans_params, im_tensor = detector(im)
        except SystemExit:  # il demo esce se non trova il volto e l'immagine non e' gia' 224x224
            print(f"[{METHOD}] NESSUN VOLTO {img_fp.name}", flush=True)
            continue

        with torch.no_grad():
            alpha = recon.net_recon(im_tensor.to(args.device))
            ad = recon.split_alpha(alpha)
            shape = recon.compute_shape(ad["id"], ad["exp"])
            rotation = recon.compute_rotation(ad["angle"])
            posed = recon.transform(shape, rotation, ad["trans"])
        elapsed = time.time() - t0

        V = posed[0].cpu().numpy().astype(np.float64)
        common.save_mesh(out, img_fp.stem, V, F)
        common.save_meta(out, img_fp.stem, {
            "method": METHOD,
            "image": str(img_fp),
            "image_shape": [int(im.size[1]), int(im.size[0])],
            "device": args.device,
            "detector": "retinaface + 106 landmark (face_box del demo, primo volto)",
            "trans_params": common.to_list(trans_params),
            "coordinate_system": COORD_SYSTEM,
            "units": "BFM (Deep3DFaceRecon), 1 = 100 mm",
            "mm_per_unit": MM_PER_UNIT,
            "alpha": {k: common.to_list(ad[k][0].cpu().numpy()) for k in ("id", "exp", "angle", "trans")},
            "landmarks_68": common.to_list(V[lmk_idx]),
            "mesh": common.mesh_stats(V, F),
            "seconds": elapsed,
        })
        times.append(elapsed)
        n_ok += 1
        print(f"[{METHOD}] {img_fp.name}: {V.shape[0]} vertici, {elapsed:.3f}s", flush=True)

    if not times:
        raise SystemExit(f"[{METHOD}] nessuna immagine ricostruita")
    print(f"[{METHOD}] {n_ok}/{len(images)} immagini, "
          f"{np.mean(times):.3f}s/immagine (mediana {np.median(times):.3f}s), out={out}",
          flush=True)


if __name__ == "__main__":
    main()
