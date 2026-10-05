#!/usr/bin/env python3
"""ArcFace e LPIPS sui render Multiface, sulle sole coppie del protocollo WS3a.

Gli estrattori sono quelli di ``v2_work/phase0/perceptual_embed.py``, i render vengono da
``ws3a_render.py``: e' lo stesso protocollo di ``aau/baselines/perceptual_matrix.py``,
riscritto per lavorare su una lista di coppie invece che su una matrice.

* ArcFace: embedding per vista, media sulle 3 viste, rinormalizzazione L2, distanza =
  1 - coseno.  Costa per MESH (8520 forward su onnxruntime CPU con tre topologie), non per
  coppia, quindi gli embedding vanno in cache su disco e le coppie di topologie li riusano
  — anche fra il giro ``clean`` e il giro ``hard``, che condividono ``tracked``, ``remesh``
  e ``down``.  **Non** passa da ``perceptual_embed.ArcFaceExtractor``: vedi sotto.

ArcFace: crop fisso, niente detector
------------------------------------
``perceptual_embed.ArcFaceExtractor`` interroga il detector di insightface e, quando non
scatta, ripiega su un center-crop dell'immagine intera.  Sui render Multiface il detector
fallisce **7768 volte su 47742** (misurato sul giro ``hard``), e i fallimenti non sono
sparsi: stanno tutti su ``noisy``.  Quelle 7768 distanze sono quindi calcolate in
un'inquadratura diversa dalle altre, cioe' in un altro spazio di embedding, dentro la
stessa tabella di AUC — ed e' esattamente la cella ``tracked->noisy`` che il giro hard
vuole misurare.

Qui si usa ``aau/baselines/arcface_fixed.ArcFaceFixedCrop``, la stessa classe di WS1 e con
la stessa logica: il detector gira UNA volta in calibrazione (``calibrate_arcface``), da
li' si prende la mediana per vista dei suoi 5 landmark, ``face_align.estimate_norm`` ne
ricava la similarita' 2x3 verso il template ``arcface_dst`` e quella trasformazione viene
congelata e applicata a tutti i render di quella vista, di qualunque topologia.  Un solo
spazio di embedding, nessun ramo di ripiego.  La calibrazione scrive in
``renders/arcface_align.json`` i landmark mediani, il loro IQR e il conteggio dei
fallimenti del detector per topologia, che e' il numero riportato qui sopra.
* LPIPS: e' pairwise sulle immagini e non ha un embedding da mediare, quindi costa per
  COPPIA (8000 coppie x 3 viste x configurazione).  I render della coppia di topologie in
  corso stanno in RAM come uint8 e vanno sulla GPU a batch.

CLIP e DINOv2 sono nelle baseline REMESH e passano dalla stessa strada di ArcFace
(embedding per vista, media, coseno): si accendono mettendoli in ``--extractors``.  Girano
sulla GPU e in linea, un'immagine per volta, quindi non usano ``--embed-workers``.

  aau/submit.sh multiface/ws3a_perceptual.sbatch
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ws3a_common as common  # noqa: E402
import ws3a_render as render  # noqa: E402

sys.path.insert(0, str(common.AAU_DIR / "baselines"))
sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "phase0"))

import arcface_fixed  # noqa: E402

EMBEDDING_EXTRACTORS = ("arcface", "clip", "dinov2")

# Stato per-worker quando gli embedding vengono calcolati su piu' processi.
_EXTRACTOR = None
_RENDER_ROOT: Path | None = None
_PROBE = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--extractors", type=str, default="arcface,lpips")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--yaws", type=str, default=",".join(str(y) for y in render.VIEW_YAWS))
    p.add_argument("--lpips-net", type=str, default="alex")
    p.add_argument("--lpips-size", type=int, default=256,
                   help="lato a cui i render vengono ridotti per LPIPS")
    p.add_argument("--lpips-batch", type=int, default=64)
    p.add_argument("--embed-workers", type=int, default=1,
                   help="processi per gli estrattori su CPU (ArcFace); 1 = in linea")
    p.add_argument("--max-pairs-per-class", type=int, default=0,
                   help="0 = protocollo intero; 50 = le 200 coppie del test di velocita'")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_render(out_root: Path, topology: str, name: str, yaw: float) -> np.ndarray:
    from PIL import Image

    path = render.render_path(out_root, topology, name, yaw)
    if not path.exists():
        raise FileNotFoundError(f"render mancante: {path} (lancia prima ws3a_render.py)")
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
    # onnxruntime prova a pinnare un thread per core del nodo e fallisce dentro il cgroup
    # Slurm; con piu' processi conviene comunque un thread per processo.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    global _EXTRACTOR, _RENDER_ROOT
    _EXTRACTOR = build_extractor(name, device, out_root)
    _RENDER_ROOT = out_root


def _embed_one(key):
    embedding = embed_render(_EXTRACTOR, load_render(_RENDER_ROOT, *key), key[2])
    # n_fallback e' cumulativo dentro il worker: il padre somma i massimi per worker.
    # None = l'estrattore non ha un ramo di ripiego (ArcFaceFixedCrop), che non e' la
    # stessa cosa di averlo e non averlo mai preso.
    return (render.render_name(*key), embedding, os.getpid(),
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


def calibrate_arcface(topologies: dict[str, list[str]], yaws, args) -> dict:
    """Passata di detector su TUTTI i render: statistiche di fallimento e crop fisso.

    Non e' un passo di misura, e' la taratura del ritaglio: dopo di questa il detector non
    viene piu' interrogato.  Gemella di ``aau/baselines/perceptual_matrix.calibrate_arcface``
    -- stessa classe, stesso json -- ma sulle chiavi (topologia, mesh, vista) di WS3a.
    """
    import json
    import multiprocessing as mp

    path = arcface_fixed.calibration_path(args.out_root)
    if path.exists() and not args.overwrite:
        data = json.loads(path.read_text())
        print(f"[arcface] calibrazione dalla cache: {path}", flush=True)
        report_detector(data)
        return data

    keys = [(topology, mesh, yaw)
            for topology, names in topologies.items() for mesh in names for yaw in yaws]
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
    for i, ((topology, mesh, yaw), kps) in enumerate(results, start=1):
        stats[topology]["n"] += 1
        if kps is None:
            stats[topology]["n_failed"] += 1
        else:
            kps_by_yaw[float(yaw)].append(np.asarray(kps, dtype=np.float64))
        if i % 2000 == 0 or i == len(keys):
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
    """Quante volte il detector fallisce per topologia, e su cosa e' calibrato il crop."""
    stats = data["detector"]
    for topology, row in sorted(stats.items()):
        share = row["n_failed"] / max(row["n"], 1)
        print(f"[arcface] detector fallito su {row['n_failed']:6d}/{row['n']:6d} render "
              f"({share:6.1%}) — {topology}", flush=True)
    detected = {k: v["n_detected"] for k, v in data["views"].items()}
    print(f"[arcface] nessuno di questi e' un ripiego: il crop e' fisso per tutti i render. "
          f"La mediana dei landmark viene pero' dalle sole detection riuscite, "
          f"{sum(detected.values())}/{stats['total']['n']} "
          f"({', '.join(f'{k}={v}' for k, v in sorted(detected.items()))}): il riquadro e' "
          f"calibrato su meno topologie di quelle a cui si applica.", flush=True)


def stage_embed(name: str, topologies: dict[str, list[str]], yaws, args) -> dict[str, np.ndarray]:
    """Embedding per (topologia, mesh, vista), con cache su disco riutilizzabile fra run."""
    cache_path = args.out_root / "embeddings" / f"{name}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, np.ndarray] = {}
    if cache_path.exists() and not args.overwrite:
        with np.load(cache_path) as z:
            done = {k: z[k] for k in z.files}

    keys = [(topology, mesh, yaw)
            for topology, names in topologies.items() for mesh in names for yaw in yaws]
    todo = [k for k in keys if render.render_name(*k) not in done]
    if not todo:
        print(f"[embed:{name}] {len(keys)} embedding gia' in cache", flush=True)
        return done

    workers = max(1, args.embed_workers if name == "arcface" else 1)
    print(f"[embed:{name}] {len(todo)}/{len(keys)} da calcolare con {workers} processo/i", flush=True)
    t0 = time.time()

    if workers == 1:
        extractor = build_extractor(name, args.device, args.out_root)
        results = ((render.render_name(*key),
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
        if i % 2000 == 0 or i == len(todo):
            print(f"[embed:{name}] {i}/{len(todo)} ({i / max(time.time() - t0, 1e-9):.1f}/s)", flush=True)
            np.savez(cache_path, **done)
    if workers > 1:
        pool.close()
        pool.join()
    np.savez(cache_path, **done)
    # Se l'estrattore non ha l'attributo la riga non si stampa affatto: ArcFaceFixedCrop
    # non ha un ramo di ripiego, e stampare uno zero sarebbe far passare una costante per
    # una misura. I fallimenti del detector li conta la calibrazione, una volta sola.
    n_fallback = (sum(per_worker.values()) if per_worker else None) if workers > 1 \
        else getattr(extractor, "n_fallback", None)
    extra = "" if n_fallback is None else f" | ripieghi dell'estrattore: {n_fallback}/{len(todo)}"
    print(f"[embed:{name}] fine in {time.time() - t0:.0f}s{extra}", flush=True)
    return done


def view_averaged(embeddings: dict[str, np.ndarray], topology: str, names, yaws) -> np.ndarray:
    """(n_mesh, d) con la media sulle viste, rinormalizzata L2."""
    rows = []
    for name in names:
        stack = np.stack([embeddings[render.render_name(topology, name, y)] for y in yaws])
        mean = stack.mean(axis=0).astype(np.float64)
        rows.append(mean / max(np.linalg.norm(mean), 1e-9))
    return np.stack(rows)


def run_embedding_metric(name, records, topologies, yaws, args) -> None:
    embeddings = stage_embed(name, topologies, yaws, args)
    index = {topology: {mesh: i for i, mesh in enumerate(names)}
             for topology, names in topologies.items()}
    per_topology = {topology: view_averaged(embeddings, topology, names, yaws)
                    for topology, names in topologies.items()}

    for topology_a, topology_b in common.TOPOLOGY_PAIRS:
        out_path = common.csv_path(name, topology_a, topology_b, args.out_root)
        if out_path.exists() and not args.overwrite:
            print(f"[{name}] {topology_a}->{topology_b}: gia' presente, salto", flush=True)
            continue
        t1 = time.time()
        A, B = per_topology[topology_a], per_topology[topology_b]
        idx_a = [index[topology_a][rec.name_a] for rec in records]
        idx_b = [index[topology_b][rec.name_b] for rec in records]
        values = 1.0 - np.einsum("ij,ij->i", A[idx_a], B[idx_b])
        common.write_distances(out_path, records, values, name, topology_a, topology_b,
                               seconds=time.time() - t1, n_views=len(yaws),
                               embedding_dim=int(A.shape[1]))
        print(f"[{name}] {topology_a}->{topology_b}: {len(values)} coppie -> {out_path}", flush=True)


def run_lpips(records, topologies, yaws, args) -> None:
    import lpips as lpips_mod
    import torch
    import torch.nn.functional as F

    todo = [(ta, tb) for ta, tb in common.TOPOLOGY_PAIRS
            if args.overwrite or not common.csv_path("lpips", ta, tb, args.out_root).exists()]
    if not todo:
        print("[lpips] tutti i csv gia' presenti, salto", flush=True)
        return

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = lpips_mod.LPIPS(net=args.lpips_net).to(device).eval()

    index = {topology: {mesh: i for i, mesh in enumerate(names)}
             for topology, names in topologies.items()}

    # I render restano in RAM di sistema come uint8 e salgono sulla GPU a batch: a 256px
    # sono ~0.2 MB l'uno, cioe' ~1.7 GB per topologia, che su una T4 da 16 GB non ci
    # starebbero insieme al modello (e con WBES_WS3A_PAIRS=hard le topologie sono sei).
    t0 = time.time()
    images: dict[tuple[str, float], "torch.Tensor"] = {}
    for topology in sorted({t for pair in todo for t in pair}):
        for yaw in yaws:
            stack = np.stack([load_render(args.out_root, topology, name, yaw)
                              for name in topologies[topology]])
            tensor = torch.from_numpy(stack).permute(0, 3, 1, 2).contiguous()
            if args.lpips_size and tensor.shape[-1] != args.lpips_size:
                tensor = F.interpolate(tensor.float(), size=(args.lpips_size, args.lpips_size),
                                       mode="bilinear", align_corners=False).round().clamp(0, 255).to(torch.uint8)
            images[(topology, yaw)] = tensor
    n_images = sum(t.shape[0] for t in images.values())
    print(f"[lpips] {n_images} render in RAM in {time.time() - t0:.0f}s "
          f"({args.lpips_size}px, device={device})", flush=True)

    for topology_a, topology_b in todo:
        t1 = time.time()
        idx_a = torch.as_tensor([index[topology_a][rec.name_a] for rec in records])
        idx_b = torch.as_tensor([index[topology_b][rec.name_b] for rec in records])
        acc = torch.zeros(len(records), dtype=torch.float64)
        for yaw in yaws:
            A, B = images[(topology_a, yaw)], images[(topology_b, yaw)]
            for start in range(0, len(records), args.lpips_batch):
                sl = slice(start, start + args.lpips_batch)
                x = A[idx_a[sl]].to(device).float() / 127.5 - 1.0
                y = B[idx_b[sl]].to(device).float() / 127.5 - 1.0
                with torch.no_grad():
                    acc[sl] += model(x, y).flatten().double().cpu()
        values = (acc / len(yaws)).numpy()
        seconds = time.time() - t1
        out_path = common.csv_path("lpips", topology_a, topology_b, args.out_root)
        common.write_distances(out_path, records, values, "lpips", topology_a, topology_b,
                               seconds=seconds, n_views=len(yaws), lpips_net=args.lpips_net,
                               lpips_size=args.lpips_size)
        rate = len(records) * len(yaws) / max(seconds, 1e-9)
        print(f"[lpips] {topology_a}->{topology_b}: {len(values)} coppie in {seconds:.0f}s "
              f"({rate:.0f} forward/s) -> {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    extractors = [e.strip() for e in args.extractors.split(",") if e.strip()]
    yaws = [float(y) for y in args.yaws.split(",") if y.strip()]

    records = common.load_pairs(max_pairs_per_class=args.max_pairs_per_class)
    topologies = common.topology_names(records)
    print(f"[ws3a-perc] estrattori={extractors} coppie={len(records)} viste={yaws} "
          f"device={args.device} mesh=" + ", ".join(f"{t}={len(n)}" for t, n in topologies.items()),
          flush=True)

    for name in extractors:
        if name == "arcface":
            calibrate_arcface(topologies, yaws, args)
        if name in EMBEDDING_EXTRACTORS:
            run_embedding_metric(name, records, topologies, yaws, args)
        elif name == "lpips":
            run_lpips(records, topologies, yaws, args)
        else:
            raise SystemExit(f"estrattore sconosciuto: {name!r}")


if __name__ == "__main__":
    main()
