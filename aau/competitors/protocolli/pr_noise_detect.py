#!/usr/bin/env python3
"""Rilevatore reale per ``pr_noise.py`` (condizione ``det``): 68 landmark iBUG di insightface ``1k3d68`` (pacchetto
buffalo_l, ``~/.insightface/models``) sui render frontali di ``pr_noise.py render``. Gira nel venv di MICA (insightface
0.7.3, onnxruntime su CPU), che non ha open3d: qui solo immagini -> punti 2D; la retroproiezione e' in ``pr_noise.py
prep --cond det``.

    aau/recon/run_recon.sh mica aau/competitors/protocolli/pr_noise_detect.py --view hifi3d

Per ogni render: detector ``det_10g`` (SCRFD) a 640 x 640, volto col riquadro piu' grande; ``1k3d68`` sul suo riquadro
(come ``FaceAnalysis.get``). Se il detector non trova volti: ``1k3d68`` sul riquadro dei pixel colpiti dal render
(``cams.npz``, ``box``; contato in ``fallback``). Uscita ``aau/runs/protocolli/noise/det/<vista>/lmk2d.npz``: ``P``
(n, 68, 3; x, y in pixel, centro del pixel = coordinata intera; z del modello, non usata), ``det_score`` (0 nel
ripiego), ``fallback``, ``n_faces``. ``--dir <cartella>`` al posto di ``--view``: render e uscita in quella cartella
(dominio da spec, ``aau/ava256/conf_protocols.py``); il calcolo non cambia.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "runs" / "protocolli" / "noise" / "det"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--view", default="")
    p.add_argument("--dir", type=Path, default=None, help="cartella con cams.npz e i png (al posto di --view)")
    a = p.parse_args()
    if not a.view and a.dir is None:
        p.error("serve --view o --dir")
    import cv2
    from insightface.app import FaceAnalysis
    from insightface.app.common import Face
    t0 = time.time()
    d = a.dir if a.dir is not None else ROOT / a.view
    tag = a.view or d.name
    with np.load(d / "cams.npz") as z:
        box, n, img_size = z["box"], len(z["box"]), int(z["img"])
    app = FaceAnalysis(name="buffalo_l", root=str(Path.home() / ".insightface"),
                       allowed_modules=["detection", "landmark_3d_68"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(img_size, img_size))
    lm = app.models["landmark_3d_68"]
    P = np.zeros((n, 68, 3))
    score, fb, nf = np.zeros(n), np.zeros(n, bool), np.zeros(n, np.int64)
    for k in range(n):
        img = cv2.imread(str(d / f"{k:04d}.png"))                       # BGR, come insightface
        faces = app.get(img)
        nf[k] = len(faces)
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            score[k] = float(f.det_score)
        else:
            f = Face(bbox=box[k].astype(np.float32))
            lm.get(img, f)
            fb[k] = True
        P[k] = np.asarray(f.landmark_3d_68, np.float64)
        if (k + 1) % 100 == 0:
            print(f"[pr-noise-det] {tag}: {k + 1}/{n}, ripieghi {int(fb.sum())}, {time.time() - t0:.0f}s", flush=True)
    tmp = d / ".lmk2d.tmp.npz"
    np.savez(tmp, P=P, det_score=score, fallback=fb, n_faces=nf)
    tmp.replace(d / "lmk2d.npz")
    print(f"[pr-noise-det] {tag}: {n} render, detector senza volto {int(fb.sum())}, {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
