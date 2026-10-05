#!/usr/bin/env python3
"""Sceglie le N camere piu' frontali di ogni soggetto Multiface.

Le mesh tracciate sono in un frame canonico della testa (i .bin hanno centroide
in zero; i .obj sono gli stessi vertici portati nel frame del rig dall'headpose
`<frame>_transform.txt`). La topologia e' la stessa per tutti i soggetti, quindi
la direzione in cui guarda il viso e' una costante di quel frame: si porta ogni
camera nel frame della testa,
  C_locale = R_f^T (C_mondo - t_f)   con   C_mondo = -R_C^T t_C,
e si ordina per coseno con quella direzione. L'headpose usato e' quello del
frame mediano del segmento indicato.

FRONT e' misurata, non assunta: e' la direzione locale della camera 400015 di
6795937 al frame E057/021897, la cui immagine e' un frontale perfetto a livello
degli occhi. E' circa 10 gradi sotto +Z: prendere +Z secco sceglie camere che
guardano il viso dall'alto (verificato guardando le immagini di 400030/400041).

  scripts/multiface_frontal_cameras.py <dir_dataset> [--segment SEG] [--top N]

Stampa la tabella dei punteggi e, in coda, il blocco json "cameras" da
incollare nella config di scripts/multiface_download.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

FRONT = (0.103, -0.170, 0.980)


def load_krt(path: Path) -> dict:
    """Stesso formato di external/multiface/dataset.py: nome, K 3x3, dist, [R|t]."""
    cameras = {}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        name = lines[i].strip()
        extrin = [[float(x) for x in lines[i + 5 + k].split()] for k in range(3)]
        cameras[name] = extrin
        i += 8
    return cameras


def cam_center(extrin) -> list[float]:
    """C = -R^T t, come `campos` in dataset.py."""
    return [-sum(extrin[k][j] * extrin[k][3] for k in range(3)) for j in range(3)]


def to_head_frame(point, transf) -> list[float]:
    """R_f^T (p - t_f): dal frame del rig a quello canonico della testa."""
    diff = [point[k] - transf[k][3] for k in range(3)]
    return [sum(transf[k][j] * diff[k] for k in range(3)) for j in range(3)]


def median_transform(mesh_dir: Path):
    files = sorted(mesh_dir.glob("*_transform.txt"))
    if not files:
        return None
    text = files[len(files) // 2].read_text().split()
    return [[float(text[4 * r + c]) for c in range(4)] for r in range(3)]


def main(args) -> None:
    front = [float(x) for x in args.front.split(",")] if args.front else list(FRONT)
    front_norm = math.sqrt(sum(x * x for x in front))
    cameras = {}
    for capture in sorted(Path(args.dataset).glob("m--*--GHS")):
        entity = capture.name.split("--")[3]
        mesh_root = capture / "tracked_mesh"
        candidates = [d for d in sorted(mesh_root.iterdir()) if args.segment.lower() in d.name.lower()]
        if not candidates:
            print(f"{entity}: nessun segmento '{args.segment}', salto")
            continue
        transf = median_transform(candidates[0])
        krt = load_krt(capture / "KRT")
        rank = []
        for name, extrin in krt.items():
            local = to_head_frame(cam_center(extrin), transf)
            dist = math.sqrt(sum(x * x for x in local))
            cos = sum(local[k] * front[k] for k in range(3)) / (dist * front_norm)
            rank.append((cos, dist, name))
        rank.sort(reverse=True)
        cameras[entity] = [name for _c, _d, name in rank[: args.top]]
        head = "  ".join(
            f"{name}({math.degrees(math.acos(min(1.0, c))):.0f}deg,{d/10:.0f}cm)"
            for c, d, name in rank[: args.top + 2]
        )
        print(f"{entity:>10} [{candidates[0].name}] {head}")
    print()
    print(json.dumps({"cameras": cameras}, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="directory con le capture m--...--GHS")
    parser.add_argument("--segment", default="neutral", help="sottostringa del segmento da cui leggere l'headpose")
    parser.add_argument("--top", type=int, default=3, help="quante camere tenere (default 3)")
    parser.add_argument("--front", default=None, help="direzione frontale locale x,y,z (default: FRONT)")
    main(parser.parse_args())
