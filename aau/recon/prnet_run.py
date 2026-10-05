#!/usr/bin/env python3
"""PRNet: da una directory di immagini a una mesh per immagine (WS3b).

  aau/recon/run_recon.sh prnet aau/recon/prnet_run.py \
      --images aau/runs/recon/test_images --out aau/runs/recon/prnet

Topologia: la UV position map 256x256 campionata su ``Data/uv-data/face_ind.txt``, cioe'
43867 vertici e 86906 triangoli (``triangles.txt``), fissa per tutte le immagini: la
mappa di posizione ha una parametrizzazione UV fissa, quindi il vertice i-esimo e' lo
stesso punto del volto in tutte le ricostruzioni.
Unita': pixel dell'immagine di input (vedi common.COORD_SYSTEM).

Tre toppe, nessuna delle quali tocca il clone:

* **TensorFlow.**  ``predictor.py`` e' TF1 con ``tf.contrib``, che non esiste piu' e non
  e' installabile su python 3.10.  Qui si registrano in ``sys.modules`` gli alias
  ``tensorflow`` -> ``tf.compat.v1`` e ``tensorflow.contrib.{layers,framework}`` ->
  ``tf_slim`` (che e' la stessa contrib.slim estratta dagli autori di TF), poi si importa
  il grafo originale e si ripristina ``sys.modules``.  I pesi restano il checkpoint TF1
  pubblicato dall'autore, letto senza conversioni.
* **BatchNorm legacy.**  ``tf_slim.batch_norm`` chiede
  ``tensorflow.python.layers.normalization.BatchNormalization``, che in TF 2.15 non e'
  piu' vendorizzato e viene lazy-caricato da ``tf_keras.legacy_tf_layers.normalization``:
  un percorso che esiste solo da tf_keras 2.16 in su, cioe' mai insieme a TF 2.15.  Lo
  stesso identico modulo sta gia' in ``keras.src.legacy_tf_layers.normalization`` (keras
  2.15 e' una dipendenza di tensorflow-cpu), e lo si registra sotto il nome che il lazy
  loader si aspetta.  Nessun pacchetto in piu' da installare.
* **Detector.**  ``api.PRN`` userebbe dlib (non installabile qui senza cmake).  Si passa
  a ``process()`` il box di FaceBoxes, che e' la via documentata dal repo per un detector
  esterno; ``--boxes full`` tratta invece l'immagine come gia' ritagliata.  Attenzione:
  PRNet e' stato messo a punto sui box di dlib, un po' piu' larghi.

TF gira su CPU: il pacchetto tensorflow-cpu non ha kernel CUDA, e le versioni con CUDA
vogliono cuDNN 8, mentre il container NGC 24.10 ha cuDNN 9.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

METHOD = "prnet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", type=Path, required=True, help="directory di jpg/png")
    p.add_argument("--out", type=Path, default=common.OUT_ROOT / METHOD)
    p.add_argument("--boxes", type=str, default="faceboxes",
                   help="faceboxes | full | percorso di un json {stem: [x1, y1, x2, y2]}")
    return p.parse_args()


def shim_legacy_batchnorm() -> None:
    """Rende risolvibile il lazy import di ``tf_keras.legacy_tf_layers.normalization``.

    Va lasciato in piedi: il lazy loader di TF scatta alla PRIMA ``tcl.conv2d`` con
    ``normalizer_fn``, cioe' molto dopo l'import del grafo (job 1019440).
    ``importlib.import_module`` guarda ``sys.modules`` prima di cercare i pacchetti
    genitori, quindi basta la chiave completa.
    """
    import importlib

    name = "tf_keras.legacy_tf_layers.normalization"
    if name not in sys.modules:
        sys.modules[name] = importlib.import_module("keras.src.legacy_tf_layers.normalization")


def import_prn(repo: Path):
    """Importa ``api.PRN`` mettendo tf_slim al posto di tf.contrib, e ripulisce dopo."""
    import tensorflow as tf
    import tf_slim

    shim_legacy_batchnorm()

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    contrib = types.ModuleType("tensorflow.contrib")
    contrib.layers = tf_slim
    contrib.framework = tf_slim
    saved = {k: sys.modules.get(k) for k in
             ("tensorflow", "tensorflow.contrib", "tensorflow.contrib.layers",
              "tensorflow.contrib.framework")}
    sys.modules["tensorflow"] = tf.compat.v1
    sys.modules["tensorflow.contrib"] = contrib
    sys.modules["tensorflow.contrib.layers"] = tf_slim
    sys.modules["tensorflow.contrib.framework"] = tf_slim
    # sys.modules da solo non basta per `import tensorflow.contrib.layers as tcl`: il
    # bytecode di `import a.b.c as x` fa getattr(a, "b"), e il ripiego introdotto da
    # bpo-30024 cerca sys.modules[a.__name__ + ".b"].  Qui `a` e' tf.compat.v1, il cui
    # __name__ e' "tensorflow._api.v2.compat.v1", quindi il ripiego cerca una chiave che
    # non esiste e l'import muore (job 1019347).  Serve l'attributo vero sul modulo.
    had_contrib = "contrib" in vars(tf.compat.v1)
    tf.compat.v1.contrib = contrib
    try:
        from api import PRN  # noqa: E402
    finally:
        if not had_contrib:
            del tf.compat.v1.contrib
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return PRN


def make_box_source(mode: str):
    """Restituisce una funzione immagine -> (bbox [left, right, top, bottom], n_volti).

    L'ordine degli estremi e' quello che si aspetta ``PRN.process``, che non e' lo stesso
    del detector (x1, y1, x2, y2).
    """
    if mode == "full":
        def from_full(img_fp, image):
            h, w = image.shape[:2]
            return [0.0, float(w - 1), 0.0, float(h - 1)], 1
        return from_full

    if mode != "faceboxes":
        with open(mode) as f:
            table = json.load(f)
        def from_json(img_fp, image):
            b = table.get(img_fp.stem)
            if b is None:
                return None, 0
            return [float(b[0]), float(b[2]), float(b[1]), float(b[3])], 1
        return from_json

    detector = common.faceboxes_detector()

    def from_faceboxes(img_fp, image):
        # FaceBoxes lavora in BGR come cv2; skimage legge in RGB.
        box, n_faces = common.pick_box(detector(image[:, :, ::-1]))
        if n_faces == 0:
            return None, 0
        return [float(box[0]), float(box[2]), float(box[1]), float(box[3])], n_faces

    return from_faceboxes


def main() -> None:
    args = parse_args()
    images = common.list_images(args.images.resolve())
    out = args.out.resolve()

    repo = common.add_repo_to_path(METHOD)
    # utils/estimate_pose.py legge 'Data/uv-data/canonical_vertices.npy' dalla cwd.
    os.chdir(repo)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # tensorflow-cpu: nessuna GPU
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    common.patch_numpy_aliases()  # la NMS di FaceBoxes usa np.int

    from skimage.io import imread  # noqa: E402
    from utils.estimate_pose import P2sRt, estimate_pose  # noqa: E402

    PRN = import_prn(repo)
    prn = PRN(is_dlib=False, prefix=str(repo))
    F = np.asarray(prn.triangles, dtype=np.int32)
    box_of = make_box_source(args.boxes)
    print(f"[{METHOD}] boxes={args.boxes} topologia: {prn.face_ind.shape[0]} vertici, "
          f"{F.shape[0]} facce", flush=True)

    n_ok = 0
    times = []
    for img_fp in images:
        t0 = time.time()
        image = imread(str(img_fp))
        if image.ndim < 3:
            image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        image = image[:, :, :3]

        bbox, n_faces = box_of(img_fp, image)
        if n_faces == 0:
            print(f"[{METHOD}] NESSUN VOLTO {img_fp.name}", flush=True)
            continue

        pos = prn.process(image, np.array(bbox, dtype=np.float64))
        if pos is None:
            print(f"[{METHOD}] POSITION MAP VUOTA {img_fp.name}", flush=True)
            continue
        V = prn.get_vertices(pos)
        kpt = prn.get_landmarks(pos)
        elapsed = time.time() - t0

        P, pose = estimate_pose(V)  # radianti, rispetto ai canonical_vertices di PRNet
        s, R, t2d = P2sRt(P)

        common.save_mesh(out, img_fp.stem, V, F)
        common.save_meta(out, img_fp.stem, {
            "method": METHOD,
            "image": str(img_fp),
            "image_shape": [int(image.shape[0]), int(image.shape[1])],
            "weights": str(repo / "Data" / "net-data" / "256_256_resfcn256_weight"),
            "device": "cpu",
            "n_faces_detected": n_faces,
            "face_box_lrtb": common.to_list(bbox),
            "boxes_mode": args.boxes,
            "coordinate_system": common.COORD_SYSTEM,
            "units": "pixel",
            "pose": {
                "yaw_deg": float(np.degrees(pose[0])), "pitch_deg": float(np.degrees(pose[1])),
                "roll_deg": float(np.degrees(pose[2])),
                "scale": float(s), "R": common.to_list(R), "t2d": common.to_list(t2d),
                "note": "da utils.estimate_pose: similarita' fra la mesh predetta e i "
                        "canonical_vertices di PRNet, quindi in una convenzione diversa "
                        "da quella dei due metodi 3DMM",
            },
            "landmarks_68": common.to_list(kpt),
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
