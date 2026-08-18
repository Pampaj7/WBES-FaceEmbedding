#!/usr/bin/env python
"""Unified multi-domain training view: symlinks + a joint GT with NaN off-blocks.

Same pattern as `v2_work/genflame/make_train_ready.py` (nothing is copied) and the
same id-offset convention: BFM 0, FLAME 1000, FaceScape 3000. The support bank
(`make_support_bank.py`) is symlinked in as extra meshes of subjects that already
have GT rows, so it adds no GT entries.

The joint GT is **block-diagonal with NaN off-block**: there is no defined
identity distance between a BFM and a FLAME subject, and NaN (not 0) is the only
encoding of that which cannot be silently consumed as data. `train_v2.py` keeps
batches domain-homogeneous so the NaN block is never read, and wraps the loaded
matrix in a guard that raises if it ever is.

The BFM GT ships 4999 subjects (id0000-id4998) while the BFM training data has
500 (id0000-id0499); GT rows are therefore restricted to the subjects actually in
the view, without which BFM's id1000+ rows would collide with FLAME's offset ids.

    .conda_env/bin/python v2_work/train_v2/make_joint_view.py            # full view
    .conda_env/bin/python v2_work/train_v2/make_joint_view.py --n-bfm 8 --n-flame 8 \
        --variants original,supp0,supp1 --out-dir /tmp/small_view
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import train_v2  # noqa: E402  (single source of the id-range -> domain convention)

DEFAULTS = {
    "bfm": (
        REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500_withops",
        REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder" / "latent_analysis"
        / "gt_distance_matrix" / "normalized_matrix_distances.npz",
    ),
    "flame": (
        REPO_ROOT / "v2_work" / "genflame" / "flame_train_ready" / "npz_withops",
        REPO_ROOT / "v2_work" / "genflame" / "flame_train_ready" / "gt_matrix.npz",
    ),
}
DEFAULT_SUPPORT_BANK = THIS_DIR / "support_bank" / "npz_withops"
NAME_RE = re.compile(r"^(?P<sid>id\d+)_GTready_(?P<variant>.+)\.npz$", re.IGNORECASE)
SUBJECT_RE = re.compile(r"(id\d+)", re.IGNORECASE)


def subject_files(data_dir: Path, variants: set[str] | None) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for p in sorted(data_dir.glob("id*_GTready_*.npz")):
        m = NAME_RE.match(p.name)
        if m is None or (variants is not None and m["variant"].lower() not in variants):
            continue
        out.setdefault(m["sid"].lower(), []).append(p)
    return out


def gt_rows(gt_npz: Path, subjects: list[str]) -> np.ndarray:
    """Sub-matrix of `gt_npz` for `subjects`, in that order."""
    with np.load(gt_npz, allow_pickle=True) as z:
        D = np.asarray(z["D_orig"], dtype=np.float64)
        idx_of = {}
        for i, name in enumerate(z["names"]):
            m = SUBJECT_RE.search(str(name))
            if m is not None:
                idx_of.setdefault(m.group(1).lower(), i)
    missing = [s for s in subjects if s not in idx_of]
    if missing:
        raise SystemExit(f"{len(missing)} subjects absent from {gt_npz}, e.g. {missing[:3]}")
    take = np.asarray([idx_of[s] for s in subjects], dtype=int)
    return D[np.ix_(take, take)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=THIS_DIR / "joint_view")
    ap.add_argument("--domains", type=str, default="bfm,flame", help=f"comma list from {sorted(DEFAULTS)}")
    ap.add_argument("--n-bfm", type=int, default=0, help="0 = all")
    ap.add_argument("--n-flame", type=int, default=0, help="0 = all")
    ap.add_argument("--variants", type=str, default="", help="comma list of _GTready_<variant> tokens to keep")
    ap.add_argument("--support-bank", type=Path, default=DEFAULT_SUPPORT_BANK, help="'' to exclude")
    a = ap.parse_args()

    variants = {v.strip().lower() for v in a.variants.split(",") if v.strip()} or None
    data_out = a.out_dir / "npz_withops"
    data_out.mkdir(parents=True, exist_ok=True)
    for stale in data_out.glob("*.npz"):
        stale.unlink()

    # "" (-> Path('.')) or a bank that was never generated both mean "no support bank"
    bank_dir = a.support_bank if (a.support_bank.name and a.support_bank.is_dir()) else None
    bank = subject_files(bank_dir, variants) if bank_dir else {}
    manifest: dict[str, object] = {"domains": {}, "support_bank": str(bank_dir or "")}
    names: list[str] = []
    blocks: list[np.ndarray] = []
    n_links = n_bank_links = 0

    wanted = [d.strip().lower() for d in a.domains.split(",") if d.strip()]
    unknown = [d for d in wanted if d not in DEFAULTS]
    if unknown:
        raise SystemExit(f"unknown domains {unknown}; known={sorted(DEFAULTS)}")

    for domain, cap in [(d, {"bfm": a.n_bfm, "flame": a.n_flame}[d]) for d in wanted]:
        data_dir, gt_npz = DEFAULTS[domain]
        by_subject = subject_files(data_dir, variants)
        subjects = sorted(by_subject)[: cap or None]
        wrong = [s for s in subjects if train_v2.domain_of(s) != domain]
        if wrong:
            raise SystemExit(
                f"{len(wrong)} {domain} ids fall outside the {domain} id range, e.g. {wrong[:3]} "
                "-- train_v2.domain_of would batch them as another domain"
            )
        for sid in subjects:
            for src in by_subject[sid] + bank.get(sid, []):
                (data_out / src.name).symlink_to(src.resolve())
                n_links += 1
            n_bank_links += len(bank.get(sid, []))
        names.extend(subjects)
        blocks.append(gt_rows(gt_npz, subjects))
        manifest["domains"][domain] = {
            "data_dir": str(data_dir),
            "gt_npz": str(gt_npz),
            "n_subjects": len(subjects),
            "id_range": [subjects[0], subjects[-1]] if subjects else [],
            "n_meshes": sum(len(by_subject[s]) + len(bank.get(s, [])) for s in subjects),
        }

    if len(names) != len(set(names)):
        raise SystemExit("subject ids collide across domains -- check the id offsets")

    n = len(names)
    D = np.full((n, n), np.nan, dtype=np.float32)
    at = 0
    for block in blocks:
        k = len(block)
        D[at : at + k, at : at + k] = block
        at += k
    assert at == n
    assert np.isfinite(np.diagonal(D)).all() and not np.any(np.diagonal(D) > 0)
    n_nan = int(np.count_nonzero(~np.isfinite(D)))
    assert n_nan == n * n - sum(len(b) ** 2 for b in blocks), "unexpected NaN pattern"

    np.savez(a.out_dir / "gt_matrix.npz", D_orig=D, names=np.array(names))
    manifest.update(
        n_subjects=n,
        n_symlinks=n_links,
        n_support_bank_links=n_bank_links,
        n_undefined_gt_entries=n_nan,
        undefined_gt_fraction=round(n_nan / (n * n), 4),
        variants=sorted(variants) if variants else "all",
    )
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
