#!/usr/bin/env python3
"""Genera la versione ospitata dello studio umano: un solo file con tutto dentro.

Prende ``hosted/_template.html`` e ci incolla le triplette e le 100 immagini come data
URI JPEG (default 384x384, qualita' 85), producendo ``hosted/index.html``.  Il template
resta leggibile e modificabile a mano; il file generato no.

PIL non c'e' sul frontend, quindi questo script gira in un job cpu:

  AAU_NV="" srun --partition=cpu --cpus-per-task=2 --mem=8G --time=00:10:00 \
      aau/run.sh aau/human_study/build_hosted.py

Le immagini sono indicizzate per soggetto (``id0027``) e non per tripletta: i 100 render
sono condivisi fra le 330 triplette, e ripeterli triplicherebbe il file.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

THIS_DIR = Path(__file__).resolve().parent

# Campi di una tripletta che servono alla pagina: `metrics` pesa 250 KB e lo usa solo
# analyze.py, che legge triplets.json e non la pagina.
TRIPLET_FIELDS = ("id", "kind", "disagreement_type", "a", "b", "c")

MARKERS = {"/*__TRIPLETS__*/ null": "triplets", "/*__IMAGES__*/ null": "images"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--triplets", type=Path, default=THIS_DIR / "triplets.json")
    p.add_argument("--template", type=Path, default=THIS_DIR / "hosted" / "_template.html")
    p.add_argument("--out", type=Path, default=THIS_DIR / "hosted" / "index.html")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--quality", type=int, default=85)
    p.add_argument("--max-bytes", type=int, default=9_000_000,
                   help="il file generato deve stare sotto questa soglia")
    return p.parse_args()


def collect_images(payload: dict, base_dir: Path) -> dict[str, Path]:
    """Soggetto -> render, con il controllo che lo stesso soggetto non abbia due file."""
    out: dict[str, Path] = {}
    for triplet in payload["triplets"]:
        for slot in ("a", "b", "c"):
            subject = triplet[slot]
            path = base_dir / triplet["images"][slot]
            if out.setdefault(subject, path) != path:
                raise SystemExit(f"[hosted] {subject}: due render diversi "
                                 f"({out[subject]} e {path})")
    missing = sorted(str(p) for p in out.values() if not p.is_file())
    if missing:
        raise SystemExit("[hosted] render mancanti:\n  " + "\n  ".join(missing))
    return out


def encode(path: Path, size: int, quality: int) -> str:
    with Image.open(path) as src:
        img = src.convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def as_js(value) -> str:
    """JSON incollabile dentro <script>: solo `</` e i separatori di riga sono un problema."""
    text = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return text.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def main() -> int:
    args = parse_args()
    with open(args.triplets, encoding="utf-8") as fh:
        payload = json.load(fh)

    renders = collect_images(payload, args.triplets.resolve().parent)
    images = {}
    raw_bytes = 0
    for subject in sorted(renders):
        raw_bytes += renders[subject].stat().st_size
        images[subject] = encode(renders[subject], args.size, args.quality)
    jpeg_bytes = sum(len(d) for d in images.values()) * 3 // 4

    triplets = {"triplets": [{k: t[k] for k in TRIPLET_FIELDS if k in t}
                             for t in payload["triplets"]]}

    template = args.template.read_text(encoding="utf-8")
    html = template
    for marker, what in MARKERS.items():
        if html.count(marker) != 1:
            raise SystemExit(f"[hosted] marcatore {what} assente o ripetuto nel template")
        html = html.replace(marker, as_js(triplets if what == "triplets" else images))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    total = args.out.stat().st_size

    print(f"[hosted] {len(images)} immagini {args.size}x{args.size} q{args.quality}: "
          f"{raw_bytes / 1e6:.1f} MB di png -> {jpeg_bytes / 1e6:.1f} MB di jpeg "
          f"(media {jpeg_bytes / len(images) / 1e3:.0f} kB)")
    print(f"[hosted] {len(triplets['triplets'])} triplette, template {len(template)} B")
    print(f"[hosted] {args.out}: {total} B = {total / 1e6:.2f} MB "
          f"({total / 1024 / 1024:.2f} MiB)")
    if total > args.max_bytes:
        print(f"[hosted] FALLITO: oltre il tetto di {args.max_bytes} B", file=sys.stderr)
        return 1
    print(f"[hosted] OK: sotto il tetto di {args.max_bytes / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
