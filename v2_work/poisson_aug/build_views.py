#!/usr/bin/env python
"""Symlink training views: existing withops meshes + the new Poisson variants.

BFM view  = datasets/REMESH/npz_data_topo_500_withops/*  + bfm_poisson_withops/*
FLAME view = v2_work/genflame/flame_train_ready/npz_withops/* (already id-offset)
           + flame_poisson_withops/* renamed idNNNN (offset +1000), same trick as
             v2_work/genflame/make_train_ready.py so the joint id0000-id1599 range
             never collides.

Symlinks only (withops npz are large); safe to re-run (skips existing links).

    .conda_env/bin/python v2_work/poisson_aug/build_views.py
    .conda_env/bin/python v2_work/poisson_aug/build_views.py --demo
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

BFM_BASE = REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500_withops"
BFM_POISSON = THIS_DIR / "bfm_poisson_withops"
BFM_VIEW = THIS_DIR / "bfm_view"

FLAME_BASE = REPO_ROOT / "v2_work" / "genflame" / "flame_train_ready" / "npz_withops"
FLAME_POISSON = THIS_DIR / "flame_poisson_withops"
FLAME_VIEW = THIS_DIR / "flame_view"

FLAME_ID_OFFSET = 1000  # matches v2_work/genflame/make_train_ready.py
FLAME_NAME_RE = re.compile(r"^flame(?P<num>\d+)_GTready_(?P<variant>.+)\.npz$")


def link_all(src_dir: Path, dst_dir: Path, rename=None) -> int:
    """Symlink every *.npz in src_dir into dst_dir, optionally renamed. Returns count linked."""
    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src_dir.glob("*.npz")):
        name = rename(p.name) if rename else p.name
        if name is None:
            continue
        link = dst_dir / name
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(p.resolve())
        n += 1
    return n


def flame_offset_name(name: str):
    m = FLAME_NAME_RE.match(name)
    if not m:
        return None
    return f"id{FLAME_ID_OFFSET + int(m['num']):04d}_GTready_{m['variant']}.npz"


def main() -> None:
    n1 = link_all(BFM_BASE, BFM_VIEW)
    n2 = link_all(BFM_POISSON, BFM_VIEW)
    print(f"bfm_view: {n1} base + {n2} poisson symlinks -> {BFM_VIEW}")

    n3 = link_all(FLAME_BASE, FLAME_VIEW)
    n4 = link_all(FLAME_POISSON, FLAME_VIEW, rename=flame_offset_name)
    print(f"flame_view: {n3} base + {n4} poisson symlinks (offset +{FLAME_ID_OFFSET}) -> {FLAME_VIEW}")


def demo() -> None:
    """Self-check: every link resolves, poisson members are present with offset ids, no copies."""
    for view, poisson_dir, tag in ((BFM_VIEW, BFM_POISSON, "id"), (FLAME_VIEW, FLAME_POISSON, "id")):
        links = sorted(view.glob("*.npz"))
        assert links, f"{view}: empty, run build_views.py first"
        broken = [p for p in links if not p.resolve().exists()]
        assert not broken, f"{view}: {len(broken)} broken symlinks, e.g. {broken[:3]}"
        not_links = [p for p in links if not p.is_symlink()]
        assert not not_links, f"{view}: {len(not_links)} real files instead of symlinks: {not_links[:3]}"
        pois = [p for p in links if "_pois" in p.name]
        assert pois, f"{view}: no *_pois*.npz members"
        assert all(p.name.startswith(tag) for p in pois), f"{view}: poisson members not renamed to '{tag}' convention"
    print(f"demo OK: bfm_view {len(list(BFM_VIEW.glob('*.npz')))} files, "
          f"flame_view {len(list(FLAME_VIEW.glob('*.npz')))} files, all symlinks resolve")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
