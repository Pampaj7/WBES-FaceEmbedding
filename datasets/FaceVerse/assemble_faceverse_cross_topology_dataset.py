#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ORIGINAL_DIR = THIS_DIR / "downsampled_with_ops"
DEFAULT_REMESH_DIR = THIS_DIR / "remesh10k_with_ops"
DEFAULT_OUTPUT_DIR = THIS_DIR / "cross_topology_10k_with_ops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a FaceVerse cross-topology dataset by exposing the original "
            "and remeshed operator files under topology-explicit filenames."
        )
    )
    parser.add_argument("--original_dir", type=Path, default=DEFAULT_ORIGINAL_DIR)
    parser.add_argument("--remesh_dir", type=Path, default=DEFAULT_REMESH_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", type=str, default="*_01.npz")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _symlink(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    target = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(target)


def main() -> None:
    args = parse_args()
    original_dir = args.original_dir.expanduser().resolve()
    remesh_dir = args.remesh_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_paths = [path for path in sorted(original_dir.glob("*.npz")) if fnmatch.fnmatch(path.name, args.pattern)]
    if not original_paths:
        raise RuntimeError(f"No original FaceVerse npz files found in {original_dir}")

    ok = 0
    missing = []
    for original_path in original_paths:
        remesh_name = f"{original_path.stem}_remesh_10k.npz"
        remesh_path = remesh_dir / remesh_name
        if not remesh_path.exists():
            missing.append(remesh_name)
            continue

        original_link = output_dir / f"{original_path.stem}_original.npz"
        remesh_link = output_dir / remesh_name
        _symlink(original_path, original_link, overwrite=bool(args.overwrite))
        _symlink(remesh_path, remesh_link, overwrite=bool(args.overwrite))
        ok += 1

    print(f"Subjects linked: {ok}")
    print(f"Output dir: {output_dir}")
    if missing:
        print(f"Missing remesh files: {len(missing)}")
        for name in missing[:20]:
            print(f"  - {name}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


if __name__ == "__main__":
    main()
