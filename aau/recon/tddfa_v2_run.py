#!/usr/bin/env python3
"""3DDFA_V2: da una directory di immagini a una mesh densa BFM per immagine (WS3b).

  aau/recon/run_recon.sh ddfa aau/recon/tddfa_v2_run.py \
      --images aau/runs/recon/test_images --out aau/runs/recon/3ddfa_v2 --device cuda

Topologia: BFM "noneck v3" ridotto dagli autori, 38365 vertici e 76073 triangoli, fissa
per tutte le immagini (i vertici sono in corrispondenza tra immagini diverse).
Unita': pixel dell'immagine di input (vedi common.COORD_SYSTEM).

Si usa il codice del clone senza modificarlo: FaceBoxes per il box, ``TDDFA.__call__``
per i 62 parametri e ``recon_vers(dense_flag=True)`` per la mesh, che e' la stessa
sequenza di ``demo.py --opt 3d``.  L'unica toppa e' ``np.long``, che ``bfm/bfm.py`` usa
ancora (e la NMS di FaceBoxes np.int) e che numpy >= 1.24, quello del container, ha
rimosso: li rimette common.patch_numpy_aliases().
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

METHOD = "3ddfa_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", type=Path, required=True, help="directory di jpg/png")
    p.add_argument("--out", type=Path, default=common.OUT_ROOT / METHOD)
    p.add_argument("--config", type=str, default="configs/mb1_120x120.yml",
                   help="relativo alla radice del clone; mb05 e' la variante piccola")
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images = common.list_images(args.images.resolve())
    out = args.out.resolve()

    repo = common.add_repo_to_path(METHOD)
    # I path dentro i .yml (checkpoint_fp, bfm_fp) sono relativi alla radice del clone.
    os.chdir(repo)
    # utils/functions.py importa matplotlib.pyplot: nodo di calcolo senza display.
    os.environ.setdefault("MPLBACKEND", "Agg")

    common.patch_numpy_aliases()

    import cv2  # noqa: E402
    import yaml  # noqa: E402
    from FaceBoxes import FaceBoxes  # noqa: E402
    from TDDFA import TDDFA  # noqa: E402
    from utils.pose import P2sRt, calc_pose  # noqa: E402

    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    tddfa = TDDFA(gpu_mode=args.device == "cuda", **cfg)
    face_boxes = FaceBoxes()

    F = np.asarray(tddfa.tri)  # (76073, 3), gia' int32 e 0-based
    print(f"[{METHOD}] config={args.config} device={args.device} "
          f"topologia: {tddfa.bfm.u.shape[0] // 3} vertici, {F.shape[0]} facce", flush=True)

    n_ok = 0
    times = []
    for img_fp in images:
        t0 = time.time()
        img = cv2.imread(str(img_fp))
        if img is None:
            print(f"[{METHOD}] ILLEGGIBILE {img_fp}", flush=True)
            continue

        boxes = face_boxes(img)
        box, n_faces = common.pick_box(boxes)
        if n_faces == 0:
            print(f"[{METHOD}] NESSUN VOLTO {img_fp.name}", flush=True)
            continue

        param_lst, roi_box_lst = tddfa(img, [box])
        param, roi_box = param_lst[0], roi_box_lst[0]
        ver_dense = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]  # (3, n)
        ver_lmk = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]  # (3, 68)
        elapsed = time.time() - t0

        V = np.asarray(ver_dense).T
        _, pose = calc_pose(param)  # yaw, pitch, roll in gradi
        s, R, t3d = P2sRt(param[:12].reshape(3, -1))

        common.save_mesh(out, img_fp.stem, V, F)
        common.save_meta(out, img_fp.stem, {
            "method": METHOD,
            "image": str(img_fp),
            "image_shape": [int(img.shape[0]), int(img.shape[1])],
            "config": args.config,
            "device": args.device,
            "n_faces_detected": n_faces,
            "face_box": common.to_list(box),
            "roi_box": common.to_list(roi_box),
            "coordinate_system": common.COORD_SYSTEM,
            "units": "pixel",
            "pose": {
                "yaw_deg": float(pose[0]), "pitch_deg": float(pose[1]), "roll_deg": float(pose[2]),
                "scale": float(s), "R": common.to_list(R), "t3d": common.to_list(t3d),
                "note": "R, s, t3d sono la camera affine del crop 120x120, prima di "
                        "similar_transform; yaw/pitch/roll da utils.pose.calc_pose",
            },
            "param_62": common.to_list(param),
            "landmarks_68": common.to_list(np.asarray(ver_lmk).T),
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
