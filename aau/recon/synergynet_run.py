#!/usr/bin/env python3
"""SynergyNet: da una directory di immagini a una mesh densa BFM per immagine (WS3b).

  aau/recon/run_recon.sh ddfa aau/recon/synergynet_run.py \
      --images aau/runs/recon/test_images --out aau/runs/recon/synergynet --device cuda

Topologia: BFM di 3DDFA v1, 53215 vertici e 105840 triangoli (piu' densa e con piu'
collo/orecchie di quella di 3DDFA_V2), fissa per tutte le immagini.
Unita': pixel dell'immagine di input (vedi common.COORD_SYSTEM).

Segue passo per passo ``singleImage.py`` del clone (FaceBoxes, box allargato del 20% e
reso quadrato, crop a 120x120, ``forward_test``, ``predict_denseVert``), ma prende la
classe da ``synergy3DMM.py``, che e' la stessa rete senza ``.cuda()`` cablato dentro e
con i percorsi assoluti.  I pesi vengono ricaricati qui esplicitamente: il costruttore
del clone li carica dentro un ``try/except: pass``, quindi un checkpoint mancante
lascerebbe la rete inizializzata a caso senza dire niente.

Attenzione al whitening.  ``3dmm_data`` e' stato ricostruito dal repo 3DDFA v1 (il link
Drive di SynergyNet e' morto, vedi aau/recon/README.md), ma il ``param_whitening.pkl``
di 3DDFA v1 NON e' quello con cui SynergyNet e' stato addestrato: il blocco di posa
(primi 12) coincide esatto, mentre le medie di forma ed espressione differiscono fino a
0.38 deviazioni standard e le std fino al 20%.  Usarlo darebbe forme sistematicamente
sbagliate.  Le statistiche giuste sono dentro il checkpoint (``param_mean``/``param_std``,
102 valori = 62 + 40 di texture) e sono quelle che si usano qui.  Le altre tabelle di
3dmm_data (basi PCA, media, triangoli) restano quelle di 3DDFA v1: che siano la stessa
base lo dicono le std, che coincidono a meno del 20% invece che di ordini di grandezza,
e lo conferma il confronto dei 68 landmark con 3DDFA_V2 (check_meshes.py --landmarks-ref).
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

METHOD = "synergynet"
IMG_SIZE = 120  # come 3DDFA_V2, cfr. singleImage.py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", type=Path, required=True, help="directory di jpg/png")
    p.add_argument("--out", type=Path, default=common.OUT_ROOT / METHOD)
    p.add_argument("--weights", type=str, default="pretrained/best.pth.tar",
                   help="relativo alla radice del clone; best_pose.pth.tar e' la variante "
                        "ottimizzata sulla posa")
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    return p.parse_args()


# Buffer che il modello si costruisce da 3dmm_data: non sono pesi appresi, e nel
# checkpoint hanno un'altra lunghezza (102 invece di 62), quindi non vanno ricopiati.
DATA_BUFFERS = ("param_mean", "param_std", "w_shp", "u", "w_exp",
                "u_base", "w_shp_base", "w_exp_base")


def load_checkpoint(weights_fp: Path, torch) -> dict:
    """Lo state_dict del checkpoint, senza il prefisso 'module.' del training multi-gpu."""
    return {k.replace("module.", ""): v for k, v in
            torch.load(weights_fp, map_location="cpu")["state_dict"].items()}


def load_weights(model, ckpt: dict, weights_fp: Path) -> None:
    """Come ``SynergyNet.load_weights`` del clone, ma fallisce se il checkpoint non entra.

    Il clone carica dentro un ``try/except: pass`` e con ``strict=False``: con un
    checkpoint sbagliato la rete resterebbe inizializzata a caso, in silenzio.
    """
    model_dict = model.state_dict()
    missing = [k for k in model_dict
               if k not in ckpt and not k.startswith(DATA_BUFFERS)]
    if missing:
        raise SystemExit(f"ERRORE: {len(missing)} pesi non coperti dal checkpoint "
                         f"{weights_fp}, ad esempio {missing[:5]}")
    ignored = [k for k in ckpt if k not in model_dict]
    loaded = [k for k in model_dict if k in ckpt and not k.startswith(DATA_BUFFERS)]
    for k in loaded:
        model_dict[k] = ckpt[k]
    model.load_state_dict(model_dict, strict=False)
    print(f"[{METHOD}] caricati {len(loaded)} tensori da {weights_fp} "
          f"({len(ignored)} chiavi del checkpoint ignorate, es. {ignored[:3]})", flush=True)


def use_checkpoint_whitening(param_pack, ckpt: dict) -> None:
    """Sostituisce param_mean/param_std di 3dmm_data con quelli del checkpoint.

    E' la coppia con cui la rete e' stata addestrata, quindi l'unica corretta per
    de-whitenare la sua uscita.  Il blocco di posa (primi 12) deve coincidere con quello
    di 3DDFA v1: e' la prova che la convenzione dei 62 parametri e' la stessa e che le
    basi PCA prese da 3DDFA v1 sono quelle giuste.
    """
    ck_mean = ckpt["param_mean"].detach().cpu().numpy().ravel()
    ck_std = ckpt["param_std"].detach().cpu().numpy().ravel()
    old_mean = np.asarray(param_pack.param_mean).ravel()
    old_std = np.asarray(param_pack.param_std).ravel()

    pose_gap = float(np.abs(ck_mean[:12] - old_mean[:12]).max()
                     + np.abs(ck_std[:12] - old_std[:12]).max())
    if pose_gap > 1e-3:
        raise SystemExit(f"ERRORE: il blocco di posa del whitening non coincide "
                         f"(scarto {pose_gap:.3e}): 3dmm_data non e' compatibile")
    shape_gap = float(np.abs((ck_mean[12:62] - old_mean[12:62]) / ck_std[12:62]).max())
    std_ratio = ck_std[12:62] / old_std[12:62]
    print(f"[{METHOD}] whitening dal checkpoint: posa identica a 3DDFA v1, "
          f"forma/espressione diversa fino a {shape_gap:.2f} std "
          f"(rapporto std {std_ratio.min():.2f}-{std_ratio.max():.2f})", flush=True)
    param_pack.param_mean = ck_mean
    param_pack.param_std = ck_std


def main() -> None:
    args = parse_args()
    images = common.list_images(args.images.resolve())
    out = args.out.resolve()

    repo = common.add_repo_to_path(METHOD)
    # utils/render.py e model_building.py leggono './3dmm_data/...' relativo alla cwd.
    os.chdir(repo)
    # utils/inference.py importa matplotlib.pyplot: nodo di calcolo senza display.
    os.environ.setdefault("MPLBACKEND", "Agg")
    common.patch_numpy_aliases()  # la NMS di FaceBoxes usa np.int

    import cv2  # noqa: E402
    import scipy.io as sio  # noqa: E402
    import torch  # noqa: E402
    import torchvision.transforms as transforms  # noqa: E402
    from FaceBoxes import FaceBoxes  # noqa: E402
    from synergy3DMM import SynergyNet  # noqa: E402
    from utils.ddfa import Normalize, ToTensor  # noqa: E402
    from utils.inference import (  # noqa: E402
        P2sRt, crop_img, param_pack, predict_denseVert, predict_pose, predict_sparseVert,
    )

    weights_fp = (repo / args.weights).resolve()
    if not weights_fp.is_file():
        raise SystemExit(f"ERRORE: pesi mancanti: {weights_fp}\n"
                         f"  Scaricali con aau/recon/fetch_assets.sh")

    ckpt = load_checkpoint(weights_fp, torch)
    model = SynergyNet()
    load_weights(model, ckpt, weights_fp)
    use_checkpoint_whitening(param_pack, ckpt)
    model = model.to(args.device)
    model.eval()

    # tri.mat e' 1-based e memorizzato come (3, ntri), come in model_building.py.
    tri = sio.loadmat(str(repo / "3dmm_data" / "tri.mat"))["tri"] - 1
    F = np.asarray(tri.T if tri.shape[0] == 3 else tri, dtype=np.int32)

    transform = transforms.Compose([ToTensor(), Normalize(mean=127.5, std=128)])
    face_boxes = FaceBoxes()
    print(f"[{METHOD}] device={args.device} topologia: {param_pack.dim} vertici, "
          f"{F.shape[0]} facce", flush=True)

    n_ok = 0
    times = []
    for img_fp in images:
        t0 = time.time()
        img_ori = cv2.imread(str(img_fp))
        if img_ori is None:
            print(f"[{METHOD}] ILLEGGIBILE {img_fp}", flush=True)
            continue

        rects = face_boxes(img_ori)
        roi_box, n_faces = common.pick_box(rects)
        if n_faces == 0:
            print(f"[{METHOD}] NESSUN VOLTO {img_fp.name}", flush=True)
            continue

        # Allarga il box del 20% e lo rende quadrato: identico a singleImage.py.
        face_box = list(roi_box)
        HCenter = (roi_box[1] + roi_box[3]) / 2
        WCenter = (roi_box[0] + roi_box[2]) / 2
        side_len = roi_box[3] - roi_box[1]
        margin = side_len * 1.2 // 2
        roi_box[0], roi_box[1] = WCenter - margin, HCenter - margin
        roi_box[2], roi_box[3] = WCenter + margin, HCenter + margin

        img = crop_img(img_ori, roi_box)
        img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        inp = transform(img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            param = model.forward_test(inp)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        # param e' whitened: predict_* lo de-whitena da solo con param_pack.
        ver_dense = predict_denseVert(param, roi_box, transform=True)  # (3, 53215)
        ver_lmk = predict_sparseVert(param, roi_box, transform=True)  # (3, 68)
        angles, translation = predict_pose(param, roi_box)
        elapsed = time.time() - t0

        V = np.asarray(ver_dense).T
        param_raw = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
        s, R, t3d = P2sRt(param_raw[:12].reshape(3, -1))

        common.save_mesh(out, img_fp.stem, V, F)
        common.save_meta(out, img_fp.stem, {
            "method": METHOD,
            "image": str(img_fp),
            "image_shape": [int(img_ori.shape[0]), int(img_ori.shape[1])],
            "weights": str(weights_fp),
            "device": args.device,
            "n_faces_detected": n_faces,
            "face_box": common.to_list(face_box),
            "roi_box": common.to_list(roi_box),
            "coordinate_system": common.COORD_SYSTEM,
            "units": "pixel",
            "pose": {
                "yaw_deg": float(angles[0]), "pitch_deg": float(angles[1]),
                "roll_deg": float(angles[2]),
                "scale": float(s), "R": common.to_list(R),
                "t3d": common.to_list(translation),
                "note": "angoli da utils.inference.matrix2angle_corr (convenzione del clone, "
                        "diversa da quella di 3DDFA_V2); t3d gia' riportato nel piano "
                        "immagine, scale riferita al crop 120x120",
            },
            "param_62": common.to_list(param_raw),
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
