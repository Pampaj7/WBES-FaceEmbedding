#!/usr/bin/env python3
"""MICA: da una directory di immagini alla forma FLAME neutra, una mesh per immagine.

  aau/recon/run_recon.sh mica aau/recon/mica_run.py --images <dir> --out <dir>

Si usa il codice del clone (external/MICA) senza modificarlo, nella sequenza di ``demo.py``:
``process`` (detector RetinaFace di insightface antelopev2, volto piu' vicino al centro,
crop ArcFace 112 + crop 224) e ``to_batch`` importati da li', poi ``mica.encode`` /
``mica.decode`` e ``flame.compute_landmarks``.  Cambia solo dove stanno gli asset:

- ``mica.tar`` e antelopev2 scaricati dagli id Google Drive di ``install.sh`` in
  ``<asset-dir>`` e ``~/.insightface/models/antelopev2``;
- FLAME2020 e' la copia licenziata in ``v2_work/genflame/official/FLAME2020``, convertita una
  volta in un pickle SENZA chumpy (``<asset-dir>/generic_model_nochumpy.pkl``, fuori dal
  repo) col caricatore di ``v2_work/genflame/flame_model.py``: il pickle ufficiale vuole
  chumpy, che non si importa con la numpy del container.  Le matrici sparse restano sparse
  (``to_np`` di MICA le densifica da se').

Uscita nel formato di ``common.py``: ``<stem>.npz`` (V, F della topologia FLAME, 5023
vertici) e ``<stem>.json`` con i 68 landmark.  Differenza dagli altri tre metodi: la mesh e'
la forma CANONICA (niente posa, niente espressione) in **millimetri**, gia' destrorsa (x a
destra, y in su, z in avanti), e il json lo dice in ``coordinate_system`` -- e' da li' che
``now_common.load_recon`` decide se negare la y.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

METHOD = "mica"
REPO = common.EXTERNAL / "MICA"
COORD_SYSTEM_FLAME = "FLAME canonical: x right, y up, z forward, millimetres (no pose)"
DEFAULT_ASSETS = Path(os.environ.get("HOME", "")) / "data" / "now_eval_work" / "mica_assets"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", type=Path, required=True, help="directory di jpg/png")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSETS)
    p.add_argument("--flame", type=Path,
                   default=common.REPO_ROOT / "v2_work" / "genflame" / "official" / "FLAME2020" / "generic_model.pkl")
    return p.parse_args()


def flame_without_chumpy(src: Path, dst: Path) -> Path:
    """Il pickle FLAME con gli array chumpy sostituiti da numpy; il resto tale e quale."""
    if dst.is_file():
        return dst
    sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "genflame"))
    from flame_model import _Ch, _Unpickler

    with open(src, "rb") as fh:
        raw = _Unpickler(fh, encoding="latin1").load()
    clean = {k: (np.asarray(v.x) if isinstance(v, _Ch) else v) for k, v in raw.items()}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as fh:
        pickle.dump(clean, fh, protocol=4)
    print(f"[{METHOD}] FLAME senza chumpy -> {dst} (chiavi {sorted(clean)})", flush=True)
    return dst


def main() -> None:
    args = parse_args()
    images = common.list_images(args.images.resolve())
    out = args.out.resolve()
    flame_pkl = flame_without_chumpy(args.flame.resolve(), args.asset_dir.resolve() / "generic_model_nochumpy.pkl")
    mica_tar = args.asset_dir.resolve() / "mica.tar"
    # utils/masking.py legge FLAME da un percorso fisso dentro il clone: symlink al pickle
    # convertito (external/ e' ignorato da git, quindi FLAME non entra nel repo).
    fixed = REPO / "data" / "FLAME2020" / "generic_model.pkl"
    if not fixed.exists():
        fixed.symlink_to(flame_pkl)

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import torch  # noqa: E402
    from configs.config import get_cfg_defaults  # noqa: E402
    from utils import util  # noqa: E402
    from utils.landmark_detector import LandmarksDetector, detectors  # noqa: E402
    import demo  # noqa: E402  (process, to_batch, deterministic, load_checkpoint)

    cfg = get_cfg_defaults()
    cfg.pretrained_model_path = str(mica_tar)
    cfg.model.flame_model_path = str(flame_pkl)
    cfg.model.testing = True
    demo.deterministic(42)
    mica = util.find_model_using_name(model_dir="micalib.models", model_name=cfg.model.name)(cfg, "cuda:0")
    demo.load_checkpoint(SimpleNamespace(m=str(mica_tar)), mica)
    mica.eval()
    F = mica.flameModel.generator.faces_tensor.cpu().numpy().astype(np.int32)
    app = LandmarksDetector(model=detectors.RETINAFACE)
    print(f"[{METHOD}] topologia FLAME: {mica.flameModel.generator.v_template.shape[0]} vertici, {len(F)} facce",
          flush=True)

    n_ok, times = 0, []
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        t0 = time.time()
        # process() di demo.py: detector, crop ArcFace (.npy) e crop 224 (.jpg) per immagine.
        paths = demo.process(SimpleNamespace(a=tmp, i=str(args.images.resolve())), app)
        t_prep = (time.time() - t0) / max(len(images), 1)
        found = {Path(p).stem for p in paths}
        for img_fp in images:
            if img_fp.stem not in found:
                print(f"[{METHOD}] NESSUN VOLTO {img_fp.name}", flush=True)
        for path in paths:
            t1 = time.time()
            name = Path(path).stem
            image, arcface = demo.to_batch(path)
            opdict = mica.decode(mica.encode(image, arcface))
            mesh = opdict["pred_canonical_shape_vertices"]
            lmk = mica.flame.compute_landmarks(mesh)
            V = mesh[0].cpu().numpy() * 1000.0
            L = lmk[0].cpu().numpy() * 1000.0
            elapsed = time.time() - t1 + t_prep
            common.save_mesh(out, name, V, F)
            common.save_meta(out, name, {
                "method": METHOD, "image": str(args.images / f"{name}.jpg"),
                "coordinate_system": COORD_SYSTEM_FLAME, "units": "mm",
                "landmarks_68": common.to_list(L),
                "shape_code": common.to_list(opdict["pred_shape_code"][0].cpu().numpy()),
                "mesh": common.mesh_stats(V, F), "seconds": elapsed,
            })
            times.append(elapsed)
            n_ok += 1
            print(f"[{METHOD}] {name}: {V.shape[0]} vertici, {elapsed:.3f}s", flush=True)

    if not times:
        raise SystemExit(f"[{METHOD}] nessuna immagine ricostruita")
    print(f"[{METHOD}] {n_ok}/{len(images)} immagini, {np.mean(times):.3f}s/immagine, out={out}", flush=True)


if __name__ == "__main__":
    main()
