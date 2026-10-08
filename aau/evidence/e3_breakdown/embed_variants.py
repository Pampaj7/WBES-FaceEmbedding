#!/usr/bin/env python3
"""E3, diagnosi di down8k: embedding di e108 con UN passo della catena cambiato al test (nessun training).

    aau/run.sh aau/evidence/e3_breakdown/embed_variants.py --variants base,areacenter,areapool,areaboth \\
        <argomenti degli script di eval> --embeddings_out <dir>/embeddings.npz
    (e3b_eval.sbatch, WBES_E3_DOMAIN=hifi)

Stessa catena di ``aau/zs3dmm/zs_embed.py`` (argomenti, soggetti, piano, modello e checkpoint dello script di
breakdown, importati), con l'ingresso o il pooling cambiati in uno di questi modi:
  - ``base``        nessun cambiamento: deve riprodurre gli embedding esistenti (controllo);
  - ``areacenter``  il loader centra sulla MEDIA DEI VERTICI e divide per max|V|; qui si centra sul
                    baricentro per AREA (indipendente dalla densita' dei vertici), poi max|V|;
  - ``areapool``    il pooling ``mean`` di DiffusionEncoderOnly e' la media sui vertici; qui la media e' pesata
                    con la massa dei vertici (area di Voronoi lumped degli operatori). Il ``max`` resta;
  - ``areaboth``    i due insieme.
Scrive ``embeddings_<variante>.npz`` accanto a ``--embeddings_out`` (stesse chiavi di zs_embed.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PERT = REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic" / "perturbated"
sys.path.insert(0, str(PERT))
sys.path.insert(0, str(PERT.parent))

import compare_model_vs_chamfer_topology_breakdown as bd  # noqa: E402

base = bd.base
VARIANTS = ("base", "areacenter", "areapool", "areaboth")


def area_centered(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    tri = V[F]
    a = 0.5 * torch.linalg.vector_norm(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1), dim=1)
    c = (tri.mean(1) * a[:, None]).sum(0) / a.sum()
    X = V - c
    return X / X.abs().max()


def encode(model, d: dict, variant: str) -> torch.Tensor:
    V = d["verts"]
    if variant in ("areacenter", "areaboth"):
        V = area_centered(V, d["faces"])
    if variant in ("base", "areacenter"):
        z, _ = base.forward_model(model=model, sample_dict=d, V_in=V, return_gate_info=False, add_noise=False)
        return z.squeeze(0)
    Zv, _ = model(V, d["mass"], d["L"], d["evals"], d["evecs"], d["faces"], d["gradX"], d["gradY"],
                  return_per_vertex=True, add_noise=False)
    m = d["mass"].reshape(-1, 1).to(Zv.dtype)
    z_mean = (Zv * m).sum(0, keepdim=True) / m.sum()
    z_max = Zv.max(dim=0, keepdim=True).values
    return model.pool_proj(torch.cat([z_mean, z_max], dim=1)).squeeze(0)


def main() -> None:
    argv = sys.argv[1:]
    k = argv.index("--embeddings_out")
    out_path = Path(argv[k + 1])
    argv = argv[:k] + argv[k + 2:]
    variants = list(VARIANTS)
    if "--variants" in argv:
        j = argv.index("--variants")
        variants = argv[j + 1].split(",")
        argv = argv[:j] + argv[j + 2:]
    sys.argv = [sys.argv[0]] + argv
    cli_args = bd.parse_args()

    # Le righe di zs_embed.py fino al modello.
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
    model = base.build_model(args=model_args, device=device)
    model.load_state_dict(base.load_checkpoint_bundle(checkpoint_path)["state_dict"], strict=True)
    model.eval()
    if model.pool_mode != "meanmax":
        raise SystemExit(f"pool_mode {model.pool_mode}: atteso meanmax")
    print(f"[embed-var] soggetti={len(target_subjects)} mesh={len(records)} varianti={variants} ckpt={checkpoint_path}",
          flush=True)

    Z = {v: [] for v in variants}
    with torch.no_grad():
        for n, r in enumerate(records, 1):
            d = base.sample_to_device(dataset[int(r.dataset_idx)], device=device)
            for v in variants:
                Z[v].append(encode(model, d, v).float().cpu().numpy())
            if n % 100 == 0:
                print(f"[embed-var] {n}/{len(records)}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for v in variants:
        np.savez(out_path.with_name(f"embeddings_{v}.npz"), Z=np.stack(Z[v]).astype(np.float32),
                 subjects=np.asarray([r.subject_id for r in records], dtype="U16"),
                 topologies=np.asarray([r.topology_label for r in records], dtype="U16"),
                 files=np.asarray([str(dataset.files[int(r.dataset_idx)]) for r in records]),
                 checkpoint=str(checkpoint_path), variant=v)
    print(f"[embed-var] scritto {[str(out_path.with_name(f'embeddings_{v}.npz')) for v in variants]}", flush=True)


if __name__ == "__main__":
    main()
