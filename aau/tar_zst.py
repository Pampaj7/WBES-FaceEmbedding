#!/usr/bin/env python3
"""Lista o estrae un .tar.zst.

Serve perche' ne il frontend ne il container NGC hanno il binario `zstd`: l'unica via
e' il modulo python `zstandard`, installato nel venv da aau/setup_env.sbatch.
Va eseguito dentro il container, cioe' via aau/run.sh.

  aau/run.sh aau/tar_zst.py list <archivio.tar.zst> [--limit N]
  aau/run.sh aau/tar_zst.py extract <archivio.tar.zst> <dest> [--strip N] [--include GLOB]

--strip N toglie i primi N componenti dal percorso di ogni membro, come
`tar --strip-components`: l'archivio ha una directory radice che qui non serve.
--include tiene solo i membri il cui nome fa match col glob: l'export di Hugging Face
si porta dietro una .cache/huggingface/upload/ con migliaia di file di metadati.
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
import tarfile
import time
from pathlib import Path

import zstandard


def open_stream(archive: Path):
    fh = open(archive, "rb")
    reader = zstandard.ZstdDecompressor().stream_reader(fh)
    # mode 'r|' = stream sequenziale, non serve seek: l'archivio non viene mai
    # decompresso per intero su disco.
    return tarfile.open(fileobj=reader, mode="r|")


def cmd_list(args: argparse.Namespace) -> int:
    counts: dict[str, int] = {}
    total = 0
    with open_stream(Path(args.archive)) as tar:
        for member in tar:
            total += 1
            if total <= args.limit:
                print(f"{member.type.decode() if isinstance(member.type, bytes) else member.type} "
                      f"{member.size:>10} {member.name}")
            top = member.name.split("/")[0]
            counts[top] = counts.get(top, 0) + 1
            if args.stop_after and total >= args.stop_after:
                print(f"... interrotto dopo {total} membri")
                break
    print(f"\nmembri totali (visti): {total}")
    print("radici di primo livello:")
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    strip = int(args.strip)
    written = 0
    skipped = 0
    started = time.time()

    with open_stream(Path(args.archive)) as tar:
        for member in tar:
            if not member.isfile():
                continue
            if args.include and not fnmatch.fnmatch(member.name, args.include):
                skipped += 1
                continue
            parts = Path(member.name).parts[strip:]
            if not parts:
                continue
            target = dest.joinpath(*parts)
            # nessun membro deve poter scrivere fuori da dest
            if not str(target.resolve()).startswith(str(dest.resolve())):
                raise RuntimeError(f"percorso sospetto nell'archivio: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(target, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            written += 1
            if written % 500 == 0:
                print(f"  {written} file, {time.time() - started:.0f}s", flush=True)

    print(f"estratti {written} file in {dest} ({time.time() - started:.0f}s), scartati {skipped}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list")
    pl.add_argument("archive")
    pl.add_argument("--limit", type=int, default=20, help="quanti membri stampare")
    pl.add_argument("--stop-after", type=int, default=0, help="0 = scorre tutto l'archivio")
    pl.set_defaults(func=cmd_list)

    pe = sub.add_parser("extract")
    pe.add_argument("archive")
    pe.add_argument("dest")
    pe.add_argument("--strip", type=int, default=0)
    pe.add_argument("--include", default=None, help="glob sul nome del membro nell'archivio")
    pe.set_defaults(func=cmd_extract)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
