#!/usr/bin/env python3
"""Download minimo di Ava-256 (Meta, CC BY-NC 4.0): le mesh registrate dei soli frame neutri, con Range request.

    python3 aau/ava256/ava_download.py --workers 8          (sul frontend: serve la rete, basta la libreria standard)

CONFERMATIVO: non valutare prima del protocollo confermativo (``aau/ava256/README.md``).

Sorgente: il bucket S3 pubblico di ``download.py`` di Meta (copia in ``datasets/AVA256/meta``, repo
facebookresearch/ava-256), release ``4TB``: ``<BASE>/<mcd>--<mct>--<sid>/decoder/``. Per ogni cattura di
``256_ids.csv``:
  1. ``frame_list.csv`` (segmento e numero di ogni frame), intero (~130 KB);
  2. di ``kinematic_tracking/registration_vertices.zip`` (~441 MB, un PLY per frame, membri NON compressi) si
     leggono solo la coda (fine della directory centrale), la directory centrale e i membri scelti, raggruppando
     quelli contigui nello zip in una sola Range request;
  3. ogni membro e' verificato col CRC32 della directory centrale (il checksum di Meta) e salvato coi byte originali
     in ``raw/<cattura>/registration_vertices/<frame:06d>.ply``; lo sha256 va nell'indice della cattura.
Frame scelti (regola fissata dopo l'esplorazione di due soggetti e PRIMA del download degli altri, vedi README):
  - ``EXP_neutral_peak``: tutti (la neutra; il loader di Meta usa questo segmento come neutro);
  - ``EXP_eye_neutral``: ``--eye-neutral`` frame a posizioni equispaziate nella lista ordinata del segmento
    (neutri ripetuti in un altro segmento della stessa sessione: servono al rumore della GT).
Un frame elencato in ``frame_list.csv`` ma assente dallo zip (frame "dropped", datasheet) si registra come mancante.
Riprendibile: una cattura con ``index.json`` gia' scritto e tutti i file presenti si salta.

Uscite: ``datasets/AVA256/raw/<cattura>/{frame_list.csv, registration_vertices/*.ply, index.json}``,
``datasets/AVA256/manifest_files.json`` (nome, byte, CRC32, sha256 di ogni file scaricato: nessuna geometria) e
``aau/ava256/download_manifest.json`` (per cattura: id, conteggi, impronta degli sha256, ETag e taglia dello zip).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DATA_ROOT = REPO_ROOT / "datasets" / "AVA256"
META_DIR = DATA_ROOT / "meta"
RAW_DIR = DATA_ROOT / "raw"
IDS_CSV = META_DIR / "256_ids.csv"
BASE = "https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/ava-256/"
ZIP_REL = "kinematic_tracking/registration_vertices.zip"
NEUTRAL_SEG = "EXP_neutral_peak"
REPEAT_SEG = "EXP_eye_neutral"
TAIL_BYTES = 1 << 16
MAX_RUN_BYTES = 16 << 20          # una Range request al massimo di 16 MB di membri contigui
RETRIES = 5


# --------------------------------------------------------------------------------- rete

def _open(url: str, rng: tuple[int, int] | None = None, method: str = "GET"):
    req = urllib.request.Request(url, method=method)
    if rng is not None:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1] - 1}")
    return urllib.request.urlopen(req, timeout=120)


def fetch(url: str, rng: tuple[int, int] | None = None) -> bytes:
    """GET (intero o [a, b)) con ripetizioni; controlla la lunghezza ricevuta di una Range request."""
    for k in range(RETRIES):
        try:
            with _open(url, rng) as r:
                data = r.read()
                if rng is not None and (r.status != 206 or len(data) != rng[1] - rng[0]):
                    raise IOError(f"Range {rng}: stato {r.status}, {len(data)} byte")
                return data
        except (OSError, http.client.HTTPException) as exc:     # URLError, timeout, risposta corta
            if k == RETRIES - 1:
                raise
            print(f"[ava-dl] riprovo {url} {rng}: {exc}", flush=True)
            time.sleep(2.0 * (k + 1))
    raise AssertionError


def head(url: str) -> dict:
    for k in range(RETRIES):
        try:
            with _open(url, method="HEAD") as r:
                return {"bytes": int(r.headers["Content-Length"]), "etag": r.headers.get("ETag", "").strip('"'),
                        "last_modified": r.headers.get("Last-Modified", "")}
        except (OSError, http.client.HTTPException) as exc:
            if k == RETRIES - 1:
                raise
            print(f"[ava-dl] riprovo HEAD {url}: {exc}", flush=True)
            time.sleep(2.0 * (k + 1))
    raise AssertionError


# ---------------------------------------------------------------------------------- zip

def central_directory(url: str, size: int) -> tuple[dict, int]:
    """{nome: (offset dell'header locale, byte compressi, byte, CRC32, metodo)} e byte letti per trovarla."""
    tail = fetch(url, (max(0, size - TAIL_BYTES), size))
    i = tail.rfind(b"PK\x05\x06")
    if i < 0 or tail.rfind(b"PK\x06\x06") >= 0:
        raise ValueError("fine della directory centrale assente o zip64 (non previsto)")
    n_tot, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
    start = size - len(tail)
    cd = tail[cd_off - start:cd_off - start + cd_size] if cd_off >= start else fetch(url, (cd_off, cd_off + cd_size))
    read = len(tail) + (0 if cd_off >= start else cd_size)
    ents, p = {}, 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        f = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        nl, el, cl = f[10], f[11], f[12]
        ents[cd[p + 46:p + 46 + nl].decode()] = (f[16], f[8], f[9], f[7], f[4])
        p += 46 + nl + el + cl
    if len(ents) != n_tot:
        raise ValueError(f"directory centrale: {len(ents)} voci su {n_tot}")
    return ents, read


def runs(members: list[tuple[str, int, int]]) -> list[list[tuple[str, int, int]]]:
    """Membri (nome, offset, byte compressi) raggruppati in tratti contigui dello zip (al massimo MAX_RUN_BYTES)."""
    out: list[list[tuple[str, int, int]]] = []
    for m in sorted(members, key=lambda x: x[1]):
        if out and m[1] - out[-1][-1][1] <= out[-1][-1][2] + 30 + 64 + 64 and \
                m[1] + m[2] - out[-1][0][1] <= MAX_RUN_BYTES:
            out[-1].append(m)
        else:
            out.append([m])
    return out


# ------------------------------------------------------------------------------ selezione

def segment_frames(frame_list: bytes) -> dict[str, list[int]]:
    seg: dict[str, list[int]] = {}
    for r in csv.DictReader(frame_list.decode().splitlines()):
        seg.setdefault(r["seg_id"], []).append(int(r["frame_id"]))
    return {k: sorted(v) for k, v in seg.items()}


def equispaced(frames: list[int], k: int) -> list[int]:
    """k frame a posizioni equispaziate (round((n - 1) j / (k - 1))) nella lista ordinata; tutti se n <= k."""
    n = len(frames)
    if k <= 0:
        return []
    if n <= k:
        return list(frames)
    if k == 1:
        return [frames[(n - 1) // 2]]
    return [frames[round((n - 1) * j / (k - 1))] for j in range(k)]


def select(seg: dict[str, list[int]], n_repeat: int, all_segments: tuple[str, ...]) -> dict[str, list[int]]:
    """Frame da scaricare per segmento (``all_segments``: tutti i frame, solo per l'esplorazione)."""
    if all_segments:
        return {s: list(seg.get(s, [])) for s in all_segments}
    return {NEUTRAL_SEG: list(seg.get(NEUTRAL_SEG, [])), REPEAT_SEG: equispaced(seg.get(REPEAT_SEG, []), n_repeat)}


# ------------------------------------------------------------------------------- cattura

def capture_name(row: dict) -> str:
    return f"{row['mcd']}--{row['mct']}--{row['sid']}"


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def process(task: dict) -> dict:
    cap, out_root, size_tag = task["capture"], Path(task["out"]), task["size"]
    cdir = out_root / cap
    index_path = cdir / "index.json"
    if index_path.exists() and not task["overwrite"]:
        idx = json.loads(index_path.read_text())
        if all((cdir / f["file"]).exists() for f in idx["files"]):
            idx["status"] = "skip"
            return idx
    t0 = time.time()
    base = f"{BASE}{size_tag}/{cap}/decoder/"
    (cdir / "registration_vertices").mkdir(parents=True, exist_ok=True)
    fl = fetch(base + "frame_list.csv")
    _write(cdir / "frame_list.csv", fl)
    files = [{"file": "frame_list.csv", "bytes": len(fl), "sha256": hashlib.sha256(fl).hexdigest(), "crc32": None}]
    seg = segment_frames(fl)
    sel = select(seg, task["eye_neutral"], task["all_segments"])
    url = base + ZIP_REL
    hz = head(url)
    ents, nread = central_directory(url, hz["bytes"])
    want, missing = [], {}
    for s, frames in sel.items():
        for f in frames:
            name = f"{f:06d}.ply"
            if name in ents:
                want.append((name, ents[name][0], ents[name][1], s))
            else:
                missing.setdefault(s, []).append(f)
    seg_of = {w[0]: w[3] for w in want}
    for run in runs([(n, o, c) for n, o, c, _ in want]):
        lo = run[0][1]
        hi = run[-1][1] + 30 + 64 + 64 + run[-1][2]
        data = fetch(url, (lo, min(hi, hz["bytes"])))
        nread += len(data)
        for name, off, csz in run:
            q = off - lo
            sig, _, _, meth, _, _, _, _, _, nl, el = struct.unpack("<IHHHHHIIIHH", data[q:q + 30])
            if sig != 0x04034B50 or data[q + 30:q + 30 + nl].decode() != name or meth != 0:
                raise ValueError(f"{cap}/{name}: header locale inatteso")
            body = data[q + 30 + nl + el:q + 30 + nl + el + csz]
            _, _, usz, crc, _ = ents[name]
            if len(body) != usz or zlib.crc32(body) != crc:
                raise ValueError(f"{cap}/{name}: CRC32 o lunghezza diversi dalla directory dello zip")
            rel = f"registration_vertices/{name}"
            _write(cdir / rel, body)
            files.append({"file": rel, "segment": seg_of[name], "bytes": usz, "crc32": f"{crc:08x}",
                          "sha256": hashlib.sha256(body).hexdigest()})
    idx = {"capture": cap, "sid": cap.split("--")[2], "release": size_tag, "zip_url": url, "zip": hz,
           "zip_entries": len(ents), "frame_list_rows": int(sum(len(v) for v in seg.values())),
           "listed": {s: len(seg.get(s, [])) for s in sel}, "selected": sel, "missing_in_zip": missing,
           "files": files, "bytes_read": nread + len(fl), "seconds": round(time.time() - t0, 1)}
    _write(index_path, (json.dumps(idx, indent=1) + "\n").encode())
    idx["status"] = "ok"
    return idx


def dump_rows(obj: dict, key: str) -> str:
    """json con una riga per elemento di ``obj[key]``: manifest compatti e leggibili nei diff."""
    head = json.dumps({k: v for k, v in obj.items() if k != key}, indent=1)[:-2]
    rows = ",\n".join("  " + json.dumps(r, separators=(",", ":")) for r in obj[key])
    return f'{head},\n "{key}": [\n{rows}\n ]\n}}\n'


def digest(files: list[dict]) -> str:
    """sha256 delle righe ``<file> <sha256>`` ordinate: un'impronta per cattura da tenere in git."""
    lines = "\n".join(sorted(f"{f['file']} {f['sha256']}" for f in files)) + "\n"
    return hashlib.sha256(lines.encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ids", type=Path, default=IDS_CSV)
    p.add_argument("--out", type=Path, default=RAW_DIR)
    p.add_argument("--size", default="4TB", choices=("4TB", "8TB", "16TB", "32TB"))
    p.add_argument("--eye-neutral", type=int, default=16, help=f"frame equispaziati di {REPEAT_SEG}")
    p.add_argument("--subjects", default="", help="sid separati da virgole (prova); vuoto = tutti")
    p.add_argument("--all-segments", default="", help="esplorazione: segmenti scaricati per intero (virgole)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-manifest", action="store_true", help="esplorazione: non scrivere i manifest")
    a = p.parse_args()

    rows = list(csv.DictReader(a.ids.read_text().splitlines()))
    order = {capture_name(r): k for k, r in enumerate(rows)}
    if len(order) != len(rows) or len({r["sid"] for r in rows}) != len(rows):
        raise SystemExit(f"{a.ids}: catture o sid ripetuti")
    if a.subjects:
        keep = set(a.subjects.split(","))
        rows = [r for r in rows if r["sid"] in keep]
    a.out.mkdir(parents=True, exist_ok=True)
    tasks = [{"capture": capture_name(r), "out": str(a.out), "size": a.size, "eye_neutral": a.eye_neutral,
              "all_segments": tuple(s for s in a.all_segments.split(",") if s), "overwrite": a.overwrite}
             for r in rows]
    print(f"[ava-dl] {len(tasks)} catture -> {a.out} (release {a.size}, {a.workers} thread)", flush=True)
    t0 = time.time()
    done, errors = [], []
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(process, t): t["capture"] for t in tasks}
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except Exception as exc:                       # noqa: BLE001  (una cattura non ferma le altre)
                errors.append(f"{futs[fu]}: {type(exc).__name__}: {exc}")
                print(f"[ava-dl] ERRORE {errors[-1]}", flush=True)
                continue
            done.append(r)
            nf = {s: len(v) for s, v in r["selected"].items()}
            print(f"[ava-dl] {len(done)}/{len(tasks)} {r['capture']} {r['status']} frame {nf} mancanti "
                  f"{ {s: len(v) for s, v in r['missing_in_zip'].items()} } letti {r['bytes_read'] / 1e6:.1f} MB "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if errors:
        raise SystemExit(f"ERRORE: {len(errors)} catture non scaricate: {errors[:5]}")
    if a.no_manifest:
        return
    done.sort(key=lambda r: order[r["capture"]])
    files = {r["capture"]: r["files"] for r in done}
    (a.out.parent / "manifest_files.json").write_text(json.dumps(
        {"source": BASE + a.size + "/", "release": a.size, "license": "CC BY-NC 4.0 (Meta, ava-256)",
         "note": "nomi, byte, CRC32 (dalla directory dello zip di Meta) e sha256 dei file scaricati; nessuna geometria",
         "captures": files}, indent=1) + "\n")
    man = {"source": BASE + a.size + "/<cattura>/decoder/", "release": a.size,
           "ids_csv_sha256": hashlib.sha256(a.ids.read_bytes()).hexdigest(),
           "rule": {NEUTRAL_SEG: "tutti i frame", REPEAT_SEG: f"{a.eye_neutral} frame equispaziati"},
           "digest": "sha256 delle righe '<file> <sha256>' ordinate dei file della cattura (manifest_files.json)",
           "n_captures": len(done),
           "n_files": int(sum(len(r["files"]) for r in done)),
           "bytes_files": int(sum(f["bytes"] for r in done for f in r["files"])),
           "bytes_read": int(sum(r["bytes_read"] for r in done)),
           "captures": [{"ava_id": f"ava{order[r['capture']]:04d}", "capture": r["capture"],
                         "n_frames": {s: len(v) for s, v in r["selected"].items()},
                         "listed": r["listed"],
                         "missing_in_zip": {s: v for s, v in r["missing_in_zip"].items()},
                         "zip_bytes": r["zip"]["bytes"], "zip_etag": r["zip"]["etag"],
                         "digest": digest(r["files"])} for r in done]}
    (THIS_DIR / "download_manifest.json").write_text(dump_rows(man, "captures"))
    print(f"[ava-dl] {man['n_files']} file, {man['bytes_files'] / 1e6:.0f} MB su disco "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
