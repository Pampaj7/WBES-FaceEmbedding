#!/usr/bin/env python3
"""Scarica il dataset Multiface (Meta, CC-BY-NC 4.0) dal bucket S3 pubblico.

Rimpiazza `external/multiface/download_dataset.py`, che importa `requests` e
`bs4`: sul frontend AAU non c'e' pip, quindi qui si usa solo la stdlib piu'
`curl` e `tar`. Il formato del file di config e' lo stesso di upstream
(entity/image/mesh/texture/metadata/audio/expression), con tre chiavi in piu':

  "cameras"          lista di camera id, oppure dict entity -> lista. Tiene solo
                     i tar `images--<SEG>_cam<ID>.tar` delle camere elencate; per
                     i segmenti impacchettati in un tar unico (senza `_cam`) il
                     filtro si applica dopo l'estrazione.
  "meshes_per_segment"  N: dopo l'estrazione tiene N frame equispaziati per
                     segmento e cancella gli altri (mesh e immagini).
  "expression"       puo' essere una lista (vale per tutte le entity) oppure un
                     dict entity -> lista: i 3 soggetti Mugsy v2 hanno segmenti
                     con nomi diversi dai 10 soggetti v1.

  scripts/multiface_download.py --config <cfg.json> --dest <dir> --dry-run
  scripts/multiface_download.py --config <cfg.json> --dest <dir>

Il --dry-run stampa l'elenco dei tar e la dimensione totale letta dall'indice
del bucket, senza scaricare nulla. Ogni tar estratto lascia un file `.done`
accanto alla destinazione, cosi' una seconda esecuzione riprende da dove si era
fermata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT_URL = (
    "https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com"
    "/MugsyDataRelease/v0.0/identities"
)
MAX_TRY = 5

# <td>data</td><td>ora</td><td>byte</td><td><a href="url">nome</a></td>
INDEX_RE = re.compile(
    r"<td>[\d-]+</td><td>[\d:]+</td><td>(\d+)</td><td><a href=\"([^\"]+)\">([^<]+)</a></td>"
)
FRAME_RE = re.compile(r"^(\d{6})")


def fetch(url: str) -> bytes:
    """GET con qualche tentativo: il bucket ogni tanto chiude la connessione."""
    for attempt in range(MAX_TRY):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            if attempt == MAX_TRY - 1:
                raise
            print(f"  ! {url} fallita ({exc}), ritento", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def index_files(entity: str) -> list[tuple[str, int, str]]:
    """Elenco (nome, byte, url) dei file di una entity, letto dal suo index.html."""
    html = fetch(f"{ROOT_URL}/{entity}/index.html").decode("utf-8", "replace")
    return [(name, int(size), url) for size, url, name in INDEX_RE.findall(html)]


def checksums(entity: str) -> dict[str, str]:
    text = fetch(f"{ROOT_URL}/{entity}/CHECKSUM").decode("utf-8", "replace")
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def per_entity(value, entity: str, default):
    """Le chiavi expression/cameras accettano sia una lista sia un dict per entity."""
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(entity, default)
    return value


def select(files, cfg, entity):
    """Filtra l'elenco dell'indice con le stesse regole di upstream, piu' le camere."""
    expressions = per_entity(cfg.get("expression"), entity, [])
    cameras = per_entity(cfg.get("cameras"), entity, None)
    keep = []
    for name, size, url in files:
        if name in ("CHECKSUM", "index.html"):
            continue
        if "unwrapped_uv" in name and not cfg.get("texture"):
            continue
        if "tracked_mesh" in name and not cfg.get("mesh"):
            continue
        if "images" in name and not cfg.get("image"):
            continue
        if "audio" in name and not cfg.get("audio"):
            continue
        if "metadata" in name and not cfg.get("metadata"):
            continue
        if "metadata" in name or "audio" in name:
            keep.append((name, size, url))
            continue
        if not any(exp in name for exp in expressions):
            continue
        if cameras is not None and "_cam" in name:
            if name.rsplit("_cam", 1)[1][:-4] not in cameras:
                continue
        keep.append((name, size, url))
    # le mesh vanno estratte prima delle immagini: definiscono i frame da tenere
    order = {"tracked_mesh": 0, "metadata": 1, "audio": 2, "unwrapped_uv_1024": 3, "images": 4}
    keep.sort(key=lambda t: (order.get(t[0].split("--")[0].split(".")[0], 9), t[0]))
    return keep


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_range(url: str, fd: int, start: int, end: int) -> None:
    """Scarica [start, end] e lo scrive in posizione con pwrite (thread-safe)."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        offset = start
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            offset += os.pwrite(fd, buf, offset)
    if offset != end + 1:
        raise IOError(f"range {start}-{end}: ricevuti {offset - start} byte")


def download(url: str, dest: Path, expected: str | None, size: int, conns: int) -> bool:
    """Scarica in `conns` range paralleli e verifica l'md5 contro il CHECKSUM.

    Una sola connessione al bucket fa circa 7 MB/s: la banda si ottiene solo
    aprendo piu' range in parallelo (misurato: 8 stream ~27 MB/s, 24 ~66 MB/s).
    """
    n = max(1, min(conns, size // (16 << 20) + 1))
    step = -(-size // n)
    for attempt in range(MAX_TRY):
        fd = os.open(str(dest), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        errors: list[BaseException] = []

        def work(i: int, fd: int = fd) -> None:
            try:
                fetch_range(url, fd, i * step, min(size, (i + 1) * step) - 1)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        os.close(fd)
        if not errors:
            if expected is None or md5(dest) == expected:
                return True
            print(f"  ! {dest.name} non passa il checksum, riscarico", flush=True)
        else:
            print(f"  ! {dest.name}: {errors[0]}, ritento", flush=True)
        dest.unlink(missing_ok=True)
        time.sleep(5 * (attempt + 1))
    return False


def capture_dir(dest: Path, tar_path: Path) -> Path | None:
    """La directory radice dentro il tar, es. m--20180227--0000--6795937--GHS."""
    with tarfile.open(tar_path) as tf:
        for member in tf:
            top = member.name.split("/")[0]
            if top not in (".", ""):
                return dest / top
    return None


def pick_frames(frames: list[str], n: int) -> set[str]:
    """N frame equispaziati sul segmento, estremi inclusi."""
    if n <= 0 or len(frames) <= n:
        return set(frames)
    step = (len(frames) - 1) / (n - 1) if n > 1 else 1
    return {frames[round(i * step)] for i in range(n)}


def segment_frames(seg_dir: Path) -> list[str]:
    """Indici dei frame presenti in una dir di segmento, ordinati e senza doppioni."""
    frames = set()
    for entry in seg_dir.iterdir():
        m = FRAME_RE.match(entry.name)
        if m:
            frames.add(m.group(1))
    return sorted(frames)


def prune_segment(seg_dir: Path, keep: set[str]) -> None:
    """Cancella da una dir di segmento i file dei frame non selezionati."""
    for entry in sorted(seg_dir.iterdir()):
        if entry.is_dir():  # images/<SEG>/<CAM>/<FRAME>.png
            prune_segment(entry, keep)
            continue
        m = FRAME_RE.match(entry.name)
        if m and m.group(1) not in keep:
            entry.unlink()


def prune(root: Path, name: str, cfg, entity, kept: dict) -> None:
    """Applica meshes_per_segment e il filtro camere ai file appena estratti."""
    n = cfg.get("meshes_per_segment", 0)
    cameras = per_entity(cfg.get("cameras"), entity, None)
    asset, sep, rest = name[:-4].partition("--")
    if not sep or asset not in ("tracked_mesh", "images", "unwrapped_uv_1024"):
        return  # metadata.tar, audio.tar: niente da potare
    segment = rest.rsplit("_cam", 1)[0]

    if asset == "tracked_mesh":
        seg_dir = root / "tracked_mesh" / segment
        if not seg_dir.is_dir():
            return
        kept[segment] = pick_frames(segment_frames(seg_dir), n)
        prune_segment(seg_dir, kept[segment])
        return

    seg_dir = root / asset / segment
    if not seg_dir.is_dir():
        return
    if cameras is not None:
        for cam_dir in sorted(seg_dir.iterdir()):
            if cam_dir.is_dir() and cam_dir.name not in cameras:
                shutil.rmtree(cam_dir)
    if segment not in kept:
        # ripresa a meta': i frame tenuti si rileggono dalle mesh gia' potate
        mesh_dir = root / "tracked_mesh" / segment
        if not mesh_dir.is_dir():
            return
        kept[segment] = set(segment_frames(mesh_dir))
    prune_segment(seg_dir, kept[segment])


def run_entity(entity: str, cfg, dest: Path, dry_run: bool, conns: int) -> int:
    files = select(index_files(entity), cfg, entity)
    total = sum(size for _, size, _ in files)
    print(f"[{entity}] {len(files)} tar, {total / 1e9:.1f} GB", flush=True)
    if dry_run:
        for name, size, _ in files:
            print(f"    {size / 1e6:9.1f} MB  {name}")
        return total

    sums = checksums(entity)
    kept: dict = {}
    existing = sorted(dest.glob(f"m--*--{entity}--GHS"))
    root = existing[0] if existing else None
    for name, size, url in files:
        tar_path = dest / (entity + name)
        done = dest / (entity + name + ".done")
        if done.exists():
            print(f"  = {name} gia' fatto", flush=True)
            continue
        print(f"  > {name} ({size / 1e6:.1f} MB)", flush=True)
        if not download(url, tar_path, sums.get(name), size, conns):
            print(f"  ! {name} SALTATO dopo {MAX_TRY} tentativi", flush=True)
            continue
        if root is None:
            root = capture_dir(dest, tar_path)
        subprocess.check_call(["tar", "-xf", str(tar_path), "-C", str(dest)])
        tar_path.unlink()
        if root is not None:
            prune(root, name, cfg, entity, kept)
        done.touch()
    return total


def main(args) -> None:
    cfg = json.loads(Path(args.config).read_text())
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for entity in cfg["entity"]:
        total += run_entity(entity, cfg, dest, args.dry_run, args.conns)
    print(f"TOTALE {total / 1e9:.1f} GB su {len(cfg['entity'])} entity", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="file json di configurazione")
    parser.add_argument("--dest", required=True, help="directory di destinazione")
    parser.add_argument("--dry-run", action="store_true", help="stampa solo elenco e dimensione")
    parser.add_argument("--conns", type=int, default=16, help="range paralleli per tar (default 16)")
    main(parser.parse_args())
