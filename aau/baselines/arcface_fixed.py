"""ArcFace sui render con un ritaglio geometrico FISSO, senza detector.

Il problema che risolve
-----------------------
``v2_work/phase0/perceptual_embed.ArcFaceExtractor`` prova il detector di insightface e,
quando non scatta, ripiega su un center-crop quadrato dell'immagine intera portato a
112x112 (``perceptual_embed.py:45-46``).  Sui render sintetici il detector scatta su certe
topologie e non su altre -- misurato su ``aau/runs/baselines_fb100/renders``, vedi il
contatore stampato da ``perceptual_matrix.py`` -- e il ripiego non e' innocuo: un ritaglio
allineato sui 5 landmark e un center-crop dell'intera immagine sono due inquadrature
diverse, quindi due spazi di embedding diversi.  Mettere le due cose nella stessa matrice
di distanze vuol dire misurare, su una parte delle coppie, la differenza fra due
inquadrature invece che fra due identita'.

La soluzione
------------
La camera e' fissa (``render_cache.py``) e ogni mesh e' normalizzata maxabs prima del
render, quindi **il volto cade sempre nello stesso riquadro dell'immagine**, a meno della
forma della singola faccia.  Basta percio' UNA trasformazione di allineamento per vista,
uguale per tutti i soggetti e tutte le topologie:

1. ``calibrate`` fa girare il detector su tutti i render una volta sola e raccoglie i 5
   landmark (kps) dove scatta.  Da li' prende la **mediana per yaw** dei 5 punti;
2. da quei 5 punti mediani ``insightface.utils.face_align.estimate_norm`` ricava la
   similarita' 2x3 verso il template ``arcface_dst``: e' la stessa trasformazione che
   insightface applicherebbe a un volto rilevato, ma congelata;
3. ``ArcFaceFixedCrop`` la applica con ``cv2.warpAffine`` a TUTTI i render della stessa
   vista e passa il 112x112 direttamente al modello di riconoscimento.  Il detector non
   viene piu' interrogato: **non esiste un ramo di ripiego**, e percio' non esiste un
   contatore di ripieghi.  La prima versione ne teneva uno, inizializzato a 0 e mai
   incrementato, che ``perceptual_matrix`` stampava come se fosse una misura: era una
   costante travestita da conteggio, ed e' stata tolta.  Quello che c'e' da contare -- i
   fallimenti del detector -- si conta una volta sola in calibrazione e si scrive in
   ``renders/arcface_align.json``.

Cosa questo NON risolve
-----------------------
La trasformazione congelata e' una per vista e viene dalla MEDIANA dei landmark sui soli
render su cui il detector e' scattato: 838 su 1500 sull'held-out (299 / 283 / 256 per yaw
-30 / 0 / +30), 749 su 1500 su facebench_first100.  Quei 838 non sono un campione uniforme
delle topologie -- 296 vengono da ``down8k``, 242 da ``original``, 223 da ``up60k``, 77 da
``remesh`` e **zero da** ``noisy`` -- quindi il riquadro e' calibrato su quattro topologie e
applicato a cinque.  E' comunque una scelta preferibile al ripiego (un solo spazio di
embedding invece di due), ma non e' neutra e va detta: un riquadro leggermente diverso
sposterebbe tutta la colonna ArcFace nello stesso verso, non una topologia sola.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CALIBRATION_NAME = "arcface_align.json"


def cpu_session_options():
    """Un thread per sessione onnxruntime: il parallelismo e' sui processi.

    OMP_NUM_THREADS non basta: onnxruntime dimensiona il suo pool intra-op sui core del NODO,
    prova a pinnarli fuori dal cgroup Slurm (``pthread_setaffinity_np failed``) e con 32
    processi x 64 thread la memoria non sta nei 48G del job (OOM, job 1057301 e 1057302).
    """
    import onnxruntime

    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return so


def calibration_path(out_root: Path) -> Path:
    return out_root / "renders" / CALIBRATION_NAME


class ArcFaceDetectorProbe:
    """Detector + recognition di insightface, usato solo per la calibrazione."""

    def __init__(self, det_size: int = 512):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection"],
            sess_options=cpu_session_options(),
        )
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def kps(self, img: np.ndarray):
        """(5, 2) landmark del volto piu' grande, o None se il detector non scatta."""
        faces = self.app.get(img[:, :, ::-1])  # insightface vuole BGR
        if not faces:
            return None
        best = max(faces, key=lambda f: float(np.prod(f.bbox[2:4] - f.bbox[0:2])))
        return np.asarray(best.kps, dtype=np.float64)


class ArcFaceFixedCrop:
    """Embedding ArcFace da una trasformazione di allineamento congelata per vista."""

    needs_yaw = True

    def __init__(self, transforms: dict[float, np.ndarray], image_size: int = 112,
                 model_root: str = "~/.insightface"):
        import os

        from insightface.model_zoo import model_zoo

        # Il modello di riconoscimento si carica diretto, non via FaceAnalysis: quella
        # pretende comunque un detector fra i moduli (`assert "detection" in self.models`),
        # e qui il detector non serve piu' a niente.
        path = os.path.join(os.path.expanduser(model_root), "models", "buffalo_l",
                            "w600k_r50.onnx")
        if not os.path.exists(path):
            raise SystemExit(f"pesi ArcFace assenti: {path}\n"
                             f"  Scaldali con: aau/submit.sh baselines/setup_env.sbatch")
        self.rec_model = model_zoo.get_model(path, providers=["CPUExecutionProvider"],
                                             sess_options=cpu_session_options())
        self.rec_model.prepare(ctx_id=-1)
        self.transforms = {float(y): np.asarray(M, dtype=np.float64)
                           for y, M in transforms.items()}
        self.image_size = int(image_size)
        # Nessun attributo `n_fallback`: qui un ramo di ripiego non c'e'. Scriverne uno a
        # 0 vorrebbe dire far stampare a perceptual_matrix un conteggio che non ha mai
        # contato niente. `n_calls` invece si incrementa davvero.
        self.n_calls = 0

    def __call__(self, img: np.ndarray, yaw: float = 0.0) -> np.ndarray:
        import cv2

        M = self.transforms.get(float(yaw))
        if M is None:
            raise KeyError(f"nessuna trasformazione calibrata per yaw={yaw} "
                           f"(disponibili {sorted(self.transforms)})")
        bgr = np.ascontiguousarray(img[:, :, ::-1])
        crop = cv2.warpAffine(bgr, M, (self.image_size, self.image_size), borderValue=0.0)
        self.n_calls += 1
        emb = self.rec_model.get_feat(crop).flatten().astype(np.float32)
        return emb / max(float(np.linalg.norm(emb)), 1e-9)


def transform_from_kps(kps: np.ndarray, image_size: int = 112) -> np.ndarray:
    """La 2x3 di insightface verso il template ``arcface_dst``, dai 5 landmark mediani."""
    from insightface.utils import face_align

    return np.asarray(
        face_align.estimate_norm(np.asarray(kps, dtype=np.float32), image_size=image_size),
        dtype=np.float64)


def build_calibration(kps_by_yaw: dict[float, list], detector_stats: dict,
                      image_size: int = 112) -> dict:
    """Mediana dei kps per yaw, sua trasformazione, e le statistiche del detector."""
    views = {}
    for yaw, samples in sorted(kps_by_yaw.items()):
        if not samples:
            raise SystemExit(
                f"il detector non e' scattato su nessun render con yaw={yaw}: senza almeno "
                f"un volto rilevato non c'e' modo di calibrare il ritaglio fisso")
        median = np.median(np.stack(samples), axis=0)
        views[f"{float(yaw):+07.2f}"] = {
            "yaw": float(yaw),
            "n_detected": len(samples),
            "kps_median": median.tolist(),
            "kps_iqr_px": (np.percentile(np.stack(samples), 75, axis=0)
                           - np.percentile(np.stack(samples), 25, axis=0)).tolist(),
            "transform": transform_from_kps(median, image_size).tolist(),
        }
    return {"image_size": int(image_size), "views": views, "detector": detector_stats}


def load_transforms(path: Path) -> dict[float, np.ndarray]:
    data = json.loads(Path(path).read_text())
    return {float(v["yaw"]): np.asarray(v["transform"], dtype=np.float64)
            for v in data["views"].values()}
