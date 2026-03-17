from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

from .paths import ensure_autoencoder_dir_on_syspath


ensure_autoencoder_dir_on_syspath()

from dataset_gtready import GTReadyDatasetNPZ as GTReadyDataset  # noqa: E402
from intrinsic_utils import sample_mesh_indices  # noqa: E402


GENERIC_TOPOLOGY_TOKENS = {
    "gtready",
    "gt",
    "ready",
    "mesh",
    "scan",
    "sample",
    "subject",
    "variant",
    "topology",
    "topo",
}
KNOWN_TOPOLOGY_LABEL_TOKENS = {
    "original",
    "remesh",
    "crop",
    "noisy",
    "bfm",
    "flame",
}


@dataclass(frozen=True)
class MeshTopologySignature:
    n_verts: int
    n_faces: int
    faces_sha1_16: str


@dataclass(frozen=True)
class EvalSampleRecord:
    dataset_idx: int
    sample_name: str
    subject_id: str
    topology_label: str
    topology_source: str
    topology_signature: MeshTopologySignature


def sample_to_device(sample: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "verts": sample["verts"].to(device),
        "mass": sample["mass"].to(device),
        "L": sample["L"].to(device),
        "evals": sample["evals"].to(device),
        "evecs": sample["evecs"].to(device),
        "faces": sample["faces"].to(device),
        "gradX": sample["gradX"].to(device),
        "gradY": sample["gradY"].to(device),
    }


def rebuild_subject_split(
    subjects: Sequence[str],
    eval_fraction: float,
    seed: int,
    max_subjects: int,
) -> tuple[list[str], list[str]]:
    arr = np.array(sorted(subjects), dtype=object)
    rng = np.random.default_rng(seed)

    if max_subjects > 0 and len(arr) > max_subjects:
        pick = rng.choice(len(arr), size=max_subjects, replace=False)
        arr = arr[np.sort(pick)]

    if len(arr) < 6:
        raise ValueError(f"Need at least 6 overlapping subjects, found {len(arr)}")

    n_eval = max(3, int(float(eval_fraction) * len(arr)))
    n_eval = min(n_eval, len(arr) - 3)
    eval_subjects = sorted(rng.choice(arr, size=n_eval, replace=False).tolist())
    train_subjects = sorted([sid for sid in arr.tolist() if sid not in set(eval_subjects)])
    return train_subjects, eval_subjects


def build_eval_plan(
    subj_map: Dict[str, List[int]],
    eval_subjects: Sequence[str],
    max_meshes_per_subject_eval: int,
    seed: int,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    plan: Dict[str, List[int]] = {}
    for sid in eval_subjects:
        if sid not in subj_map:
            continue
        idxs = sample_mesh_indices(
            subj_map[sid],
            max_meshes=max_meshes_per_subject_eval,
            seed=int(rng.integers(0, 2_000_000_000)),
        )
        if idxs:
            plan[sid] = [int(idx) for idx in idxs]
    return plan


def _resolve_preload_workers(workers: int) -> int:
    if workers > 0:
        return workers
    return max(1, min(8, os.cpu_count() or 1))


def preload_eval_samples(
    dataset: GTReadyDataset,
    eval_plan: Dict[str, List[int]],
    workers: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    unique_idxs = sorted({int(idx) for idxs in eval_plan.values() for idx in idxs})
    if not unique_idxs:
        return {}

    resolved_workers = _resolve_preload_workers(workers)
    loaded: Dict[int, Dict[str, torch.Tensor]] = {}

    if resolved_workers <= 1:
        for idx in tqdm(unique_idxs, desc="Preload samples", dynamic_ncols=True):
            loaded[int(idx)] = dataset[int(idx)]
        return loaded

    with ThreadPoolExecutor(max_workers=resolved_workers) as executor:
        future_to_idx = {executor.submit(dataset.__getitem__, int(idx)): int(idx) for idx in unique_idxs}
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Preload samples",
            dynamic_ncols=True,
        ):
            idx = future_to_idx[future]
            loaded[idx] = future.result()
    return loaded


def estimate_sample_cache_vertex_bytes(sample_cache: Dict[int, Dict[str, torch.Tensor]] | None) -> int:
    if not sample_cache:
        return 0

    total_bytes = 0
    for sample in sample_cache.values():
        verts = sample.get("verts")
        if torch.is_tensor(verts):
            total_bytes += int(verts.numel()) * int(verts.element_size())
    return int(total_bytes)


def maybe_cache_chamfer_vertices_on_device(
    sample_cache: Dict[int, Dict[str, torch.Tensor]] | None,
    device: torch.device,
    cache_mode: str,
    max_mb: float,
) -> tuple[Dict[int, Dict[str, torch.Tensor]] | None, Dict[str, object]]:
    total_bytes = estimate_sample_cache_vertex_bytes(sample_cache)
    stats: Dict[str, object] = {
        "mode": str(cache_mode),
        "enabled": False,
        "device": "",
        "vertex_bytes": int(total_bytes),
        "vertex_mb": float(total_bytes / (1024.0**2)),
        "reason": "",
    }

    if not sample_cache:
        stats["reason"] = "no_preloaded_samples"
        return sample_cache, stats
    if device.type != "cuda":
        stats["reason"] = "device_not_cuda"
        return sample_cache, stats
    if cache_mode == "off":
        stats["reason"] = "disabled"
        return sample_cache, stats

    limit_bytes = max(0, int(float(max_mb) * (1024.0**2)))
    if cache_mode == "auto" and total_bytes > limit_bytes:
        stats["reason"] = f"over_auto_limit_{float(max_mb):.1f}mb"
        return sample_cache, stats

    for idx in tqdm(
        sorted(sample_cache.keys()),
        desc=f"Cache Chamfer verts to {device.type}",
        dynamic_ncols=True,
        leave=False,
    ):
        sample = sample_cache[int(idx)]
        verts = sample.get("verts")
        if torch.is_tensor(verts) and verts.device != device:
            sample["verts"] = verts.contiguous().to(device, non_blocking=True)

    stats["enabled"] = True
    stats["device"] = str(device)
    stats["reason"] = "cached"
    return sample_cache, stats


def topology_signature_from_sample(sample: Dict[str, torch.Tensor]) -> MeshTopologySignature:
    faces = sample["faces"].detach().cpu().contiguous()
    faces_sha1_16 = hashlib.sha1(faces.numpy().tobytes()).hexdigest()[:16]
    return MeshTopologySignature(
        n_verts=int(sample["verts"].shape[0]),
        n_faces=int(faces.shape[0]),
        faces_sha1_16=faces_sha1_16,
    )


def topology_signature_key(signature: MeshTopologySignature) -> str:
    return f"sig_{signature.n_verts}v_{signature.n_faces}f_{signature.faces_sha1_16}"


def serialize_topology_signature(signature: MeshTopologySignature) -> Dict[str, object]:
    return {
        "n_verts": int(signature.n_verts),
        "n_faces": int(signature.n_faces),
        "faces_sha1_16": str(signature.faces_sha1_16),
        "signature_key": topology_signature_key(signature),
    }


def infer_topology_label_from_name(name: str, subject_id: str) -> str:
    stem = Path(str(name)).stem.lower()
    subject_prefix = re.escape(subject_id.lower())
    stem = re.sub(rf"^{subject_prefix}(?:[_-]+)?", "", stem)
    tokens = [tok for tok in re.split(r"[_-]+", stem) if tok]
    tokens = [tok for tok in tokens if tok not in GENERIC_TOPOLOGY_TOKENS]
    if not tokens:
        return ""

    for tok in tokens:
        if tok in KNOWN_TOPOLOGY_LABEL_TOKENS:
            return tok

    if len(tokens) == 1 and not re.fullmatch(r"\d+", tokens[0]):
        return tokens[0]
    return "_".join(tokens)


def serialize_sample_record(record: EvalSampleRecord) -> Dict[str, object]:
    return {
        "dataset_idx": int(record.dataset_idx),
        "sample_name": str(record.sample_name),
        "subject_id": str(record.subject_id),
        "topology_label": str(record.topology_label),
        "topology_source": str(record.topology_source),
        "topology_signature": serialize_topology_signature(record.topology_signature),
    }


def build_sample_eval_records(
    dataset: GTReadyDataset,
    eval_plan: Dict[str, List[int]],
    eval_subjects: Sequence[str],
    sample_cache: Dict[int, Dict[str, torch.Tensor]] | None,
) -> List[EvalSampleRecord]:
    records: List[EvalSampleRecord] = []
    for sid in eval_subjects:
        for idx in eval_plan.get(sid, []):
            sample = sample_cache[int(idx)] if sample_cache is not None else dataset[int(idx)]
            sample_name = str(sample.get("name", dataset.files[int(idx)]))
            signature = topology_signature_from_sample(sample)
            label = infer_topology_label_from_name(sample_name, sid)
            if label:
                topology_label = label
                topology_source = "filename"
            else:
                topology_label = topology_signature_key(signature)
                topology_source = "topology_signature"

            records.append(
                EvalSampleRecord(
                    dataset_idx=int(idx),
                    sample_name=sample_name,
                    subject_id=str(sid),
                    topology_label=str(topology_label),
                    topology_source=str(topology_source),
                    topology_signature=signature,
                )
            )
    records.sort(key=lambda rec: (rec.subject_id, rec.topology_label, rec.sample_name, rec.dataset_idx))
    return records


def summarize_sample_records(records: Sequence[EvalSampleRecord]) -> Dict[str, object]:
    label_counts: Dict[str, int] = {}
    label_subjects: Dict[str, set[str]] = {}
    label_signatures: Dict[str, set[str]] = {}
    source_counts: Dict[str, int] = {}
    signature_counts: Dict[str, int] = {}

    for rec in records:
        label_counts[rec.topology_label] = label_counts.get(rec.topology_label, 0) + 1
        label_subjects.setdefault(rec.topology_label, set()).add(rec.subject_id)
        label_signatures.setdefault(rec.topology_label, set()).add(topology_signature_key(rec.topology_signature))
        source_counts[rec.topology_source] = source_counts.get(rec.topology_source, 0) + 1
        sig_key = topology_signature_key(rec.topology_signature)
        signature_counts[sig_key] = signature_counts.get(sig_key, 0) + 1

    label_summary = [
        {
            "topology_label": label,
            "n_samples": int(label_counts[label]),
            "n_subjects": int(len(label_subjects.get(label, set()))),
            "n_signatures": int(len(label_signatures.get(label, set()))),
        }
        for label in sorted(label_counts.keys())
    ]

    return {
        "n_samples": int(len(records)),
        "n_subjects": int(len({rec.subject_id for rec in records})),
        "n_topology_labels": int(len(label_counts)),
        "topology_label_counts": {label: int(label_counts[label]) for label in sorted(label_counts.keys())},
        "topology_source_counts": {key: int(source_counts[key]) for key in sorted(source_counts.keys())},
        "topology_signature_counts": {key: int(signature_counts[key]) for key in sorted(signature_counts.keys())},
        "topology_label_summary": label_summary,
        "sample_preview": [serialize_sample_record(rec) for rec in list(records)[:12]],
    }
