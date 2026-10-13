#!/usr/bin/env python3
"""Tempi del rilevatore di NoW col rilevatore (``cc.py``): 68 landmark iBUG di insightface ``1k3d68`` sui render di
``cc.py det-render``, con la stessa catena di ``aau/competitors/protocolli/pr_noise_detect.py`` (detector ``det_10g`` a
640 x 640, volto col riquadro piu' grande, ``1k3d68`` sul suo riquadro; ripiego sul riquadro dei pixel colpiti se il
detector non trova volti). Gira nel venv di MICA (insightface 0.7.3, onnxruntime su CPU).

    aau/recon/run_recon.sh mica aau/costo_confronto/cc_detect.py --work <cartella del job>

Unica differenza da ``pr_noise_detect.py``: le sessioni onnxruntime si ricreano con ``intra_op_num_threads`` = 1 (un
core, come tutti i tempi CPU di ``cc.py``); il calcolo non cambia (``cc.sbatch`` esegue anche ``pr_noise_detect.py
--dir`` sulla stessa cartella e ``now`` confronta i landmark). Giro 0 su tutte le mesh (escluso), giri 1..reps sulle
mesh cronometrate del campione. Uscite: ``<work>/det/lmk2d_cc.npz`` (``P``, ``det_score``, ``fallback``, ``n_faces``,
formato di ``pr_noise_detect.py``), ``<out>/parts/detect.json`` e ``<work>/det_check`` (i primi ``CHECK`` render con le
loro camere, per il controllo).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cc import OUT, item_of, load_sample, mesh_rounds, rec, write_part  # noqa: E402  (solo numpy)

CHECK = 24


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    import cv2
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    from insightface.app.common import Face
    S = load_sample(a.out)
    d = a.work / "det"
    loc = {k: n for n, k in enumerate(S["meshes"])}
    with np.load(d / "cams.npz") as z:
        box, img_size = z["box"], int(z["img"])
    app = FaceAnalysis(name="buffalo_l", root=str(Path.home() / ".insightface"),
                       allowed_modules=["detection", "landmark_3d_68"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(img_size, img_size))
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = a.threads, 1
    for m in app.models.values():                                      # stesse reti, un thread
        m.session = ort.InferenceSession(m.model_file, sess_options=so, providers=["CPUExecutionProvider"])
    lm = app.models["landmark_3d_68"]
    n = len(S["meshes"])
    P = np.zeros((n, 68, 3))
    score, fb, nf = np.zeros(n), np.zeros(n, bool), np.zeros(n, np.int64)
    rows = []
    for r, k in mesh_rounds(S, a.reps):
        q = loc[k]
        t0 = time.perf_counter()
        img = cv2.imread(str(d / f"{q:04d}.png"))                     # BGR, come insightface
        t1 = time.perf_counter()
        faces = app.get(img)
        t2 = time.perf_counter()
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            score[q] = float(f.det_score)
        else:
            f = Face(bbox=box[q].astype(np.float32))
            lm.get(img, f)
            fb[q] = True
        t3 = time.perf_counter()
        nf[q] = len(faces)
        P[q] = np.asarray(f.landmark_3d_68, np.float64)
        rec(rows, "now_det", "mesh", S["topo"][k], item_of(S, k), r, "cpu", lettura_png=t1 - t0, rilevamento=t2 - t1,
            ripiego=t3 - t2)
    np.savez(d / "lmk2d_cc.npz", P=P, det_score=score, fallback=fb, n_faces=nf)
    # i primi CHECK render per il controllo con pr_noise_detect.py (cc.sbatch, poi cc.py detcheck)
    c = a.work / "det_check"
    c.mkdir(exist_ok=True)
    m = min(CHECK, n)
    with np.load(d / "cams.npz") as z:
        np.savez(c / "cams.npz", **{k: (z[k][:m] if z[k].ndim else z[k]) for k in z.files})
    for q in range(m):
        shutil.copyfile(d / f"{q:04d}.png", c / f"{q:04d}.png")
    ck = {"ripieghi": int(fb.sum()), "onnxruntime": ort.__version__, "intra_op_num_threads": a.threads}
    write_part(a.out, "detect", rows, ck)


if __name__ == "__main__":
    main()
