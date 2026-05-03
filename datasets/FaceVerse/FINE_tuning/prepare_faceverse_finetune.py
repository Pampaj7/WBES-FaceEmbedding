#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
FACEVERSE_DIR = THIS_DIR.parent
DEFAULT_SOURCE_DIR = FACEVERSE_DIR / "cross_topology_10k_with_ops"
DEFAULT_GT_DIST_NPZ = (
    FACEVERSE_DIR
    / "gt_distance_matrix"
    / "faceverse_detail_pose01_vertex_mean_l2_normalized.npz"
)
FACEVERSE_NPZ_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<pose>\d+)_(?P<variant>original|remesh_10k)\.npz$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build few-shot FaceVerse fine-tuning folders compatible with the "
            "REMESH intrinsic trainer by exposing FaceVerse subjects as idXXXX."
        )
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--gt_dist_npz", type=Path, default=DEFAULT_GT_DIST_NPZ)
    parser.add_argument("--output_root", type=Path, default=THIS_DIR)
    parser.add_argument(
        "--shots",
        type=str,
        default="6,10,20,50,100",
        help="Comma-separated number of FaceVerse subjects to expose for fine-tuning.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_shots(text: str) -> list[int]:
    shots: list[int] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value <= 0:
            raise ValueError("--shots values must be positive")
        shots.append(value)
    if not shots:
        raise ValueError("No valid --shots values provided")
    return shots


def _faceverse_subject_to_id(subject: str) -> str:
    return f"id{int(subject):04d}"


def _normalize_faceverse_subject(value: object) -> str:
    text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value)
    stem = Path(text).stem
    match = re.search(r"(\d+)", stem)
    if match is None:
        raise ValueError(f"Could not parse FaceVerse subject id from {text!r}")
    return f"{int(match.group(1)):03d}"


def _relative_symlink(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(target)


def collect_variant_paths(source_dir: Path) -> dict[str, dict[str, Path]]:
    subject_variants: dict[str, dict[str, Path]] = {}
    for path in sorted(source_dir.glob("*.npz")):
        match = FACEVERSE_NPZ_RE.fullmatch(path.name)
        if match is None:
            continue
        subject = f"{int(match.group('subject')):03d}"
        variant = match.group("variant")
        subject_variants.setdefault(subject, {})[variant] = path

    complete = {
        subject: variants
        for subject, variants in subject_variants.items()
        if {"original", "remesh_10k"}.issubset(variants.keys())
    }
    if not complete:
        raise RuntimeError(f"No complete original/remesh_10k pairs found in {source_dir}")
    return dict(sorted(complete.items()))


def write_converted_distance_matrix(
    gt_dist_npz: Path,
    output_root: Path,
) -> Path:
    pack = np.load(gt_dist_npz, allow_pickle=True)
    if "D_orig" not in pack or "names" not in pack:
        raise KeyError(f"{gt_dist_npz} must contain D_orig and names")

    names = [_faceverse_subject_to_id(_normalize_faceverse_subject(name)) for name in pack["names"]]
    D_orig = np.asarray(pack["D_orig"], dtype=np.float32)
    out_path = output_root / "faceverse_id_distance_matrix.npz"
    np.savez_compressed(out_path, D_orig=D_orig, names=np.asarray(names, dtype=str))
    return out_path


def build_shot_dataset(
    source_subjects: dict[str, dict[str, Path]],
    selected_subjects: list[str],
    heldout_subjects: list[str],
    output_dir: Path,
    overwrite: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    linked: list[dict[str, str]] = []
    for subject in selected_subjects:
        id_subject = _faceverse_subject_to_id(subject)
        for variant, topo_name in (("original", "original"), ("remesh_10k", "remesh10k")):
            src = source_subjects[subject][variant]
            dst = output_dir / f"{id_subject}_GTready_faceverse_{topo_name}.npz"
            _relative_symlink(src=src, dst=dst, overwrite=overwrite)
            linked.append({"subject": id_subject, "variant": variant, "path": str(dst)})

    train_ids_path = output_dir.parent / f"{output_dir.name}_train_subject_ids.txt"
    heldout_ids_path = output_dir.parent / f"{output_dir.name}_heldout_subject_ids.txt"
    train_ids_path.write_text("\n".join(selected_subjects) + "\n", encoding="utf-8")
    heldout_ids_path.write_text("\n".join(heldout_subjects) + "\n", encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "n_subjects": int(len(selected_subjects)),
        "n_files": int(len(linked)),
        "subjects": [_faceverse_subject_to_id(subject) for subject in selected_subjects],
        "faceverse_subject_ids": list(selected_subjects),
        "heldout_faceverse_subject_ids": list(heldout_subjects),
        "train_subject_ids_txt": str(train_ids_path),
        "heldout_subject_ids_txt": str(heldout_ids_path),
        "linked_files": linked,
    }


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_subjects = collect_variant_paths(source_dir)
    all_subjects = sorted(source_subjects.keys())
    shots = _parse_shots(args.shots)
    if max(shots) > len(all_subjects):
        raise ValueError(f"Requested {max(shots)} shots, but only {len(all_subjects)} subjects are available")

    dist_path = write_converted_distance_matrix(
        gt_dist_npz=args.gt_dist_npz.expanduser().resolve(),
        output_root=output_root,
    )

    rng = np.random.default_rng(int(args.seed))
    permuted = np.asarray(all_subjects, dtype=object)
    rng.shuffle(permuted)
    ordered_subjects = [str(subject) for subject in permuted.tolist()]

    manifest = {
        "source_dir": str(source_dir),
        "gt_dist_npz": str(args.gt_dist_npz.expanduser().resolve()),
        "converted_dist_npz": str(dist_path),
        "seed": int(args.seed),
        "n_available_subjects": int(len(all_subjects)),
        "shots": {},
    }

    for shot in shots:
        selected = sorted(ordered_subjects[: int(shot)])
        heldout = sorted(subject for subject in all_subjects if subject not in set(selected))
        out_dir = output_root / f"shot{int(shot):02d}_cross_topology_with_ops"
        manifest["shots"][str(int(shot))] = build_shot_dataset(
            source_subjects=source_subjects,
            selected_subjects=selected,
            heldout_subjects=heldout,
            output_dir=out_dir,
            overwrite=bool(args.overwrite),
        )

    manifest_path = output_root / "faceverse_finetune_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote converted distance matrix: {dist_path}")
    print(f"Wrote manifest: {manifest_path}")
    for shot, info in manifest["shots"].items():
        print(f"shot{int(shot):02d}: {info['n_subjects']} subjects, {info['n_files']} files -> {info['output_dir']}")


if __name__ == "__main__":
    main()
