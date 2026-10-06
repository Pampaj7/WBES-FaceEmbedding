#!/usr/bin/env python3
"""Embedding di OGNI mesh valutata, per le misure di riconoscimento d'identita' (retrieval, verifica).

    aau/run.sh aau/zs3dmm/zs_embed.py "${common_args[@]}" --embeddings_out <stage>/embeddings.npz
    (zs_zeroshot.sbatch con WBES_ZS_EMBED=1)

Gli script di eval scrivono solo distanze fra soggetti DIVERSI (le coppie con la GT), mentre
retrieval e verifica servono anche le coppie stesso soggetto / topologia diversa. Qui si
calcola l'embedding ``z`` di ogni mesh con gli stessi argomenti e la stessa catena dello script
di breakdown (``compare_model_vs_chamfer_topology_breakdown.py``, importato: argomenti, scelta
dei soggetti, piano di eval, modello, checkpoint, ``forward_model`` senza rumore), cosi' che
``||z_i - z_j||`` coincida con ``latent_distance`` delle pair_metrics (lo controlla il
summarizer, zs_expr_summarize.py).

Scrive ``embeddings.npz``: ``Z`` (n, d) float32, ``subjects``, ``topologies``, ``files`` nello
stesso ordine, ``checkpoint``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PERT = REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic" / "perturbated"
sys.path.insert(0, str(PERT))
sys.path.insert(0, str(PERT.parent))

import compare_model_vs_chamfer_topology_breakdown as bd  # noqa: E402

base = bd.base


def main() -> None:
    argv = sys.argv[1:]
    if "--embeddings_out" not in argv:
        raise SystemExit("uso: zs_embed.py <argomenti degli script di eval> --embeddings_out <file.npz>")
    k = argv.index("--embeddings_out")
    out_path = Path(argv[k + 1])
    sys.argv = [sys.argv[0]] + argv[:k] + argv[k + 2:]
    cli_args = bd.parse_args()

    # Le righe di bd.main() fino agli embedding, senza le coppie e la Chamfer.
    run_dir, checkpoint_path = base._resolve_run_dir_and_checkpoint(
        cli_args.model_path, selector=str(cli_args.checkpoint_selector))
    base_args = base.merge_run_args(checkpoint_path, explicit_config_json=cli_args.config_json)
    model_args = base._resolve_runtime_args(cli_args, base_args)
    base.seed_everything(int(model_args.seed))
    device = base._resolve_device(cli_args.device)

    dataset = base.GTReadyDataset(model_args.data_dir)
    _, gt_name_to_idx = base.load_gt_distance_matrix(model_args.dist_npz, subject_re=base.SUBJECT_RE_ANY,
                                                     dtype=base.np.float64)
    subject_map = base.build_subject_map(dataset.files, subject_re=base.SUBJECT_RE_ANY)
    subjects = sorted([sid for sid in subject_map.keys() if sid in gt_name_to_idx])
    _, _, target_subjects = base._select_subject_subset(
        subjects=subjects, subject_split=str(cli_args.subject_split),
        eval_fraction=float(cli_args.eval_fraction), seed=int(model_args.seed),
        max_subjects=int(cli_args.max_subjects))
    eval_plan = base.build_eval_plan(
        subj_map=subject_map, eval_subjects=target_subjects,
        max_meshes_per_subject_eval=int(model_args.max_meshes_per_subject_eval), seed=int(model_args.seed))
    records = base.build_sample_eval_records(dataset=dataset, eval_plan=eval_plan,
                                             eval_subjects=target_subjects, sample_cache=None)
    print(f"[zs-embed] soggetti={len(target_subjects)} mesh={len(records)} ckpt={checkpoint_path}", flush=True)

    model = base.build_model(args=model_args, device=device)
    model.load_state_dict(base.load_checkpoint_bundle(checkpoint_path)["state_dict"], strict=True)
    model.eval()
    Z, _ = bd._collect_clean_embeddings_and_vertices(model=model, dataset=dataset, sample_records=records,
                                                     sample_cache=None, device=device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(tmp, Z=Z.detach().cpu().numpy().astype(np.float32),
             subjects=np.asarray([r.subject_id for r in records], dtype="U16"),
             topologies=np.asarray([r.topology_label for r in records], dtype="U16"),
             files=np.asarray([str(dataset.files[int(r.dataset_idx)]) for r in records]),
             checkpoint=str(checkpoint_path))
    tmp.replace(out_path)
    print(f"[zs-embed] scritto {out_path}: Z {tuple(Z.shape)}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
