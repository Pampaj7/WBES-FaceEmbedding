#!/usr/bin/env python3
"""Matrici 100x100 delle metriche percettive sui render (ArcFace, CLIP, DINOv2, LPIPS).

CLIP e DINOv2 sono quelli di ``v2_work/phase0/perceptual_embed.py``; i render vengono da
``render_cache.py`` (3 viste, camera unica, mesh normalizzata maxabs come per Chamfer).
Protocollo:

* ArcFace / CLIP / DINOv2: embedding per vista, media sulle 3 viste, rinormalizzazione
  L2, distanza = 1 - coseno.
* LPIPS: distanza per vista fra le due immagini, poi media sulle 3 viste (LPIPS e'
  pairwise sulle immagini, non ha un embedding da mediare).

ArcFace **non** usa piu' ``perceptual_embed.ArcFaceExtractor``.  Quella classe interroga il
detector e, quando non scatta, ripiega su un center-crop dell'immagine intera: due
inquadrature diverse, quindi due spazi di embedding mescolati nella stessa matrice.  Qui si
usa ``arcface_fixed.ArcFaceFixedCrop``, che applica a tutti i render della stessa vista una
sola trasformazione di allineamento, calibrata una volta sulla mediana dei landmark del
detector (stage ``calibrate_arcface``).  Lo stesso stage e' anche il posto in cui si
CONTA quante volte il detector fallisce, per topologia: e' il numero che prima non veniva
mai stampato davvero.

  aau/run_baselines.sh aau/baselines/perceptual_matrix.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arcface_fixed  # noqa: E402
import common  # noqa: E402
import render_cache  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "phase0"))

EMBEDDING_EXTRACTORS = ("arcface", "clip", "dinov2")

# Stato per-worker quando gli embedding vengono calcolati su piu' processi.
_EXTRACTOR = None
_RENDER_ROOT: Path | None = None
_PROBE = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--extractors", type=str, default="arcface,clip,dinov2,lpips")
    p.add_argument("--settings", type=str, default=",".join(common.SETTINGS))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--yaws", type=str, default=",".join(str(y) for y in common.VIEW_YAWS))
    p.add_argument("--lpips-net", type=str, default="alex")
    p.add_argument("--lpips-size", type=int, default=256,
                   help="lato a cui i render vengono ridotti per LPIPS")
    p.add_argument("--lpips-batch", type=int, default=64)
    p.add_argument("--embed-workers", type=int, default=1,
                   help="processi per gli estrattori su CPU (ArcFace); 1 = in linea")
    p.add_argument("--subject-set", type=str, default="heldout", choices=common.SUBJECT_SETS,
                   help="heldout = split del repo; facebench_first100 = i soggetti della Tabella 2")
    p.add_argument("--max-subjects", type=int, default=0, help="0 = tutti; >0 per un test rapido")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_render(out_root: Path, subject: str, topology: str, yaw: float) -> np.ndarray:
    from PIL import Image

    path = out_root / "renders" / f"{render_cache.render_name(subject, topology, yaw)}.png"
    if not path.exists():
        raise FileNotFoundError(f"render mancante: {path} (lancia prima render_cache.py)")
    return np.asarray(Image.open(path).convert("RGB"))


def build_extractor(name: str, device: str, out_root: Path):
    from perceptual_embed import EXTRACTORS

    if name == "arcface":
        # Niente detector: la trasformazione di allineamento e' quella congelata dalla
        # calibrazione, una per vista. Gira su onnxruntime CPU e non vuole un device torch.
        return arcface_fixed.ArcFaceFixedCrop(
            arcface_fixed.load_transforms(arcface_fixed.calibration_path(out_root)))
    return EXTRACTORS[name](device=device)


def embed_render(extractor, img: np.ndarray, yaw: float) -> np.ndarray:
    """CLIP e DINOv2 non sanno niente della vista; ArcFace a crop fisso si'.

    Il riquadro del volto dipende dallo yaw (la camera e' fissa, ma la faccia ruota), quindi
    la trasformazione calibrata e' una per vista e va scelta qui.
    """
    return extractor(img, yaw) if getattr(extractor, "needs_yaw", False) else extractor(img)


def _init_embed_worker(name: str, device: str, out_root: Path) -> None:
    import os

    # onnxruntime prova a pinnare un thread per core del nodo e fallisce dentro il cgroup
    # Slurm; con piu' processi conviene comunque un thread per processo.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    global _EXTRACTOR, _RENDER_ROOT
    _EXTRACTOR = build_extractor(name, device, out_root)
    _RENDER_ROOT = out_root


def _embed_one(key):
    embedding = embed_render(_EXTRACTOR, load_render(_RENDER_ROOT, *key), key[2])
    # n_fallback e' cumulativo dentro il worker: il padre somma i massimi per worker.
    # None vuol dire che l'estrattore NON HA un ramo di ripiego (e' il caso di
    # ArcFaceFixedCrop), che non e' la stessa cosa di averlo e non averlo mai preso: nel
    # secondo caso il numero si stampa, nel primo non c'e' niente da stampare.
    return (render_cache.render_name(*key), embedding, os.getpid(),
            getattr(_EXTRACTOR, "n_fallback", None))


# ------------------------------------------------------------ calibrazione ArcFace

def _init_probe_worker(out_root: Path) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    global _PROBE, _RENDER_ROOT
    _PROBE = arcface_fixed.ArcFaceDetectorProbe()
    _RENDER_ROOT = out_root


def _probe_one(key):
    kps = _PROBE.kps(load_render(_RENDER_ROOT, *key))
    return key, (None if kps is None else kps.tolist())


def calibrate_arcface(subjects, topologies, yaws, args) -> dict:
    """Passata di detector su TUTTI i render: statistiche di fallimento e crop fisso.

    Non e' un passo di misura, e' la taratura del ritaglio: dopo di questa il detector non
    viene piu' interrogato.  Il conteggio per topologia che stampa qui e' il contatore di
    fallback vero -- prima non veniva mai calcolato.
    """
    import multiprocessing as mp

    path = arcface_fixed.calibration_path(args.out_root)
    if path.exists() and not args.overwrite:
        data = json.loads(path.read_text())
        print(f"[arcface] calibrazione dalla cache: {path}", flush=True)
        report_detector(data)
        return data

    keys = [(s, t, y) for t in topologies for s in subjects for y in yaws]
    workers = max(1, args.embed_workers)
    print(f"[arcface] calibrazione: detector su {len(keys)} render con {workers} processo/i",
          flush=True)
    t0 = time.time()
    if workers == 1:
        _init_probe_worker(args.out_root)
        results = (_probe_one(k) for k in keys)
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=workers, initializer=_init_probe_worker,
                        initargs=(args.out_root,))
        results = pool.imap_unordered(_probe_one, keys, chunksize=8)

    kps_by_yaw: dict[float, list] = {float(y): [] for y in yaws}
    stats = {t: {"n": 0, "n_failed": 0} for t in topologies}
    for i, ((subject, topology, yaw), kps) in enumerate(results, start=1):
        stats[topology]["n"] += 1
        if kps is None:
            stats[topology]["n_failed"] += 1
        else:
            kps_by_yaw[float(yaw)].append(np.asarray(kps, dtype=np.float64))
        if i % 200 == 0 or i == len(keys):
            print(f"[arcface] calibrazione {i}/{len(keys)} "
                  f"({i / max(time.time() - t0, 1e-9):.1f}/s)", flush=True)
    if workers > 1:
        pool.close()
        pool.join()

    stats["total"] = {"n": sum(v["n"] for v in stats.values()),
                      "n_failed": sum(v["n_failed"] for v in stats.values())}
    data = arcface_fixed.build_calibration(kps_by_yaw, stats)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[arcface] calibrazione in {time.time() - t0:.0f}s -> {path}", flush=True)
    report_detector(data)
    return data


def report_detector(data: dict) -> None:
    """Il contatore vero, per topologia, piu' la base su cui il crop e' stato calibrato.

    Il conteggio dei fallimenti prima era hardcodato a 0 e non stampato mai.  La seconda
    riga e' l'altra meta' della verita': la trasformazione congelata viene dalla mediana
    delle sole detection RIUSCITE, che non sono distribuite in modo uniforme sulle
    topologie, e il riquadro finisce calibrato su meno topologie di quelle su cui viene
    applicato.  Non e' un ripiego -- lo spazio di embedding resta uno solo -- ma e' un
    limite, e va letto insieme al numero sopra.
    """
    stats = data["detector"]
    for topology, row in sorted(stats.items()):
        share = row["n_failed"] / max(row["n"], 1)
        print(f"[arcface] detector fallito su {row['n_failed']:5d}/{row['n']:5d} render "
              f"({share:6.1%}) — {topology}", flush=True)
    detected = {k: v["n_detected"] for k, v in data["views"].items()}
    print(f"[arcface] nessuno di questi e' un ripiego: il crop e' fisso per tutti i render. "
          f"La mediana dei landmark viene pero' dalle sole detection riuscite, "
          f"{sum(detected.values())}/{stats['total']['n']} "
          f"({', '.join(f'{k}={v}' for k, v in sorted(detected.items()))}): il riquadro e' "
          f"calibrato su meno topologie di quelle a cui si applica.", flush=True)


def stage_embed(name: str, subjects, topologies, yaws, args) -> dict[str, np.ndarray]:
    """Embedding per vista, con cache su disco riutilizzabile fra run."""
    cache_path = args.out_root / "embeddings" / f"{name}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, np.ndarray] = {}
    if cache_path.exists() and not args.overwrite:
        with np.load(cache_path) as z:
            done = {k: z[k] for k in z.files}

    keys = [(s, t, y) for t in topologies for s in subjects for y in yaws]
    todo = [k for k in keys if render_cache.render_name(*k) not in done]
    if not todo:
        print(f"[embed:{name}] {len(keys)} embedding gia' in cache", flush=True)
        return done

    workers = max(1, args.embed_workers if name == "arcface" else 1)
    print(f"[embed:{name}] {len(todo)}/{len(keys)} da calcolare con {workers} processo/i", flush=True)
    t0 = time.time()

    extractor = None
    if workers == 1:
        extractor = build_extractor(name, args.device, args.out_root)
        results = ((render_cache.render_name(*key),
                    embed_render(extractor, load_render(args.out_root, *key), key[2]),
                    0, getattr(extractor, "n_fallback", None))
                   for key in todo)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=workers, initializer=_init_embed_worker,
                        initargs=(name, args.device, args.out_root))
        results = pool.imap_unordered(_embed_one, todo, chunksize=8)

    per_worker: dict[int, int] = {}
    for i, (key, embedding, pid, n_fallback) in enumerate(results, start=1):
        done[key] = embedding
        if n_fallback is not None:
            per_worker[pid] = max(per_worker.get(pid, 0), int(n_fallback))
        if i % 200 == 0 or i == len(todo):
            print(f"[embed:{name}] {i}/{len(todo)} ({i / max(time.time() - t0, 1e-9):.1f}/s)", flush=True)
            np.savez(cache_path, **done)
    if workers > 1:
        pool.close()
        pool.join()
    np.savez(cache_path, **done)
    # Il conteggio e' quello vero: la somma dei massimi per worker se i processi sono piu'
    # di uno, l'attributo dell'estrattore se il processo e' uno solo. Se l'estrattore non
    # ha l'attributo la riga non si stampa affatto: ArcFaceFixedCrop non ha un ramo di
    # ripiego, e stampare `0/1500` sarebbe far passare una costante per una misura.
    n_fallback = (sum(per_worker.values()) if per_worker else None) if workers > 1 \
        else getattr(extractor, "n_fallback", None)
    extra = "" if n_fallback is None else f" | ripieghi dell'estrattore: {n_fallback}/{len(todo)}"
    print(f"[embed:{name}] fine in {time.time() - t0:.0f}s{extra}", flush=True)
    return done


def view_averaged(embeddings: dict[str, np.ndarray], subjects, topology, yaws) -> np.ndarray:
    """(n_subjects, d) con la media sulle viste, rinormalizzata L2."""
    rows = []
    for subject in subjects:
        stack = np.stack([embeddings[render_cache.render_name(subject, topology, y)] for y in yaws])
        mean = stack.mean(axis=0).astype(np.float64)
        rows.append(mean / max(np.linalg.norm(mean), 1e-9))
    return np.stack(rows)


def run_embedding_metric(name, subjects, topologies, topology_pairs, yaws, args) -> None:
    embeddings = stage_embed(name, subjects, topologies, yaws, args)
    per_topology = {t: view_averaged(embeddings, subjects, t, yaws) for t in topologies}
    pair_i, pair_j = common.subject_pair_indices(len(subjects))

    for topology_a, topology_b in topology_pairs:
        out_path = common.matrix_path(name, topology_a, topology_b, args.out_root)
        if out_path.exists() and not args.overwrite:
            print(f"[{name}] {topology_a}->{topology_b}: gia' presente, salto", flush=True)
            continue
        A, B = per_topology[topology_a], per_topology[topology_b]
        full = 1.0 - A @ B.T
        D = common.empty_matrix(len(subjects))
        D[pair_i, pair_j] = full[pair_i, pair_j]
        common.save_matrix(out_path, D, subjects, name, topology_a, topology_b,
                           n_views=len(yaws), embedding_dim=A.shape[1])
        print(f"[{name}] {topology_a}->{topology_b} -> {out_path}", flush=True)


def run_lpips(subjects, topologies, topology_pairs, yaws, args) -> None:
    import lpips as lpips_mod
    import torch
    import torch.nn.functional as F

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = lpips_mod.LPIPS(net=args.lpips_net).to(device).eval()

    todo = [
        (ta, tb) for ta, tb in topology_pairs
        if args.overwrite or not common.matrix_path("lpips", ta, tb, args.out_root).exists()
    ]
    if not todo:
        print("[lpips] tutte le matrici gia' presenti, salto", flush=True)
        return

    # Tutti i render in memoria GPU come uint8: 1500 immagini a 256px sono ~300 MB.
    t0 = time.time()
    images: dict[tuple[str, float], torch.Tensor] = {}
    for topology in topologies:
        for yaw in yaws:
            stack = np.stack([load_render(args.out_root, s, topology, yaw) for s in subjects])
            tensor = torch.from_numpy(stack).permute(0, 3, 1, 2).contiguous()
            if args.lpips_size and tensor.shape[-1] != args.lpips_size:
                tensor = F.interpolate(tensor.float(), size=(args.lpips_size, args.lpips_size),
                                       mode="bilinear", align_corners=False).round().clamp(0, 255).to(torch.uint8)
            images[(topology, yaw)] = tensor.to(device)
    print(f"[lpips] {len(images) * len(subjects)} render in memoria in {time.time() - t0:.0f}s "
          f"({args.lpips_size}px, device={device})", flush=True)

    pair_i, pair_j = common.subject_pair_indices(len(subjects))
    idx_i = torch.as_tensor(pair_i, device=device)
    idx_j = torch.as_tensor(pair_j, device=device)

    for topology_a, topology_b in todo:
        t1 = time.time()
        acc = torch.zeros(len(pair_i), dtype=torch.float64, device=device)
        for yaw in yaws:
            A, B = images[(topology_a, yaw)], images[(topology_b, yaw)]
            for start in range(0, len(pair_i), args.lpips_batch):
                sl = slice(start, start + args.lpips_batch)
                x = A[idx_i[sl]].float() / 127.5 - 1.0
                y = B[idx_j[sl]].float() / 127.5 - 1.0
                with torch.no_grad():
                    acc[sl] += model(x, y).flatten().double()
        values = (acc / len(yaws)).cpu().numpy()
        D = common.empty_matrix(len(subjects))
        D[pair_i, pair_j] = values
        common.save_matrix(
            common.matrix_path("lpips", topology_a, topology_b, args.out_root),
            D, subjects, "lpips", topology_a, topology_b,
            n_views=len(yaws), lpips_net=args.lpips_net, lpips_size=args.lpips_size,
        )
        rate = len(pair_i) * len(yaws) / max(time.time() - t1, 1e-9)
        print(f"[lpips] {topology_a}->{topology_b}: {len(values)} coppie in {time.time() - t1:.0f}s "
              f"({rate:.0f} forward/s)", flush=True)


def main() -> None:
    args = parse_args()
    extractors = [e.strip() for e in args.extractors.split(",") if e.strip()]
    yaws = [float(y) for y in args.yaws.split(",") if y.strip()]

    subjects = common.subject_set(args.subject_set)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    topology_pairs = common.all_topology_pairs(settings)
    topologies = sorted({t for pair in topology_pairs for t in pair})
    print(f"[perc] estrattori={extractors} soggetti={len(subjects)} topologie={topologies} "
          f"viste={yaws} device={args.device}", flush=True)

    for name in extractors:
        if name == "arcface":
            calibrate_arcface(subjects, topologies, yaws, args)
        if name in EMBEDDING_EXTRACTORS:
            run_embedding_metric(name, subjects, topologies, topology_pairs, yaws, args)
        elif name == "lpips":
            run_lpips(subjects, topologies, topology_pairs, yaws, args)
        else:
            raise SystemExit(f"estrattore sconosciuto: {name!r}")


if __name__ == "__main__":
    main()
