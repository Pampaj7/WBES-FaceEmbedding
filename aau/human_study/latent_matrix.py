#!/usr/bin/env python3
"""Matrice 100x100 della distanza latente del modello v1 sui soggetti held-out (WS4).

Serve a ``select_triplets.py --latent-matrix``: senza, lo studio umano confronta solo
gt/chamfer/lpips e la metrica che il paper propone non entra nel disaccordo.

L'encoder non viene reimplementato: si usa la stessa catena dell'eval del repo, la stessa
di ``aau/multiface/ws3a_latent.py``, cioe'

    merge_run_args(ckpt) -> build_model -> load_state_dict -> GTReadyDatasetNPZ
    -> sample_to_device -> forward_model(V_in=sample["verts"], add_noise=False)

e la distanza fra due mesh e' la norma L2 fra i due latenti, come in
``robustness/eval_utils.py`` (``torch.linalg.vector_norm(Z[i] - Z[j])`` per la metrica
``latent``, e ``torch.cdist(Z, Z, p=2)`` nell'eval online): euclidea, non coseno, sui
latenti grezzi senza rinormalizzazione.

Il costo e' per MESH (100 forward), non per coppia: gli embedding vanno in cache in
``embeddings/<metrica>_<topologia>.npz`` accanto a quelli delle baseline percettive.
L'output e' un npz nel formato di ``common.save_matrix``, stesse chiavi e stesso ordine
dei soggetti delle matrici in ``aau/runs/baselines/matrices/``: riempita solo la parte
i<j, il resto NaN.

  aau/submit.sh human_study/latent_matrix.sbatch

Con ``--topology t1,t2,...`` produce una matrice same-topology per topologia riusando lo
stesso modello: e' cosi' che si ottengono le 6 celle diagonali della Tabella 1 del paper
(vedi ``aau/baselines/table1_diagonal.py``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "baselines"))

import common  # noqa: E402

INTRINSIC_DIR = common.REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"
sys.path.insert(0, str(INTRINSIC_DIR))

# Gli npz CON operatori DiffusionNet, layout piatto (tutte le topologie in una cartella):
# e' la stessa WBES_DATA_DIR che usano training ed eval.
OPS_DIR = Path(
    os.environ.get("WBES_DATA_DIR", common.REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500_withops")
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--ops-dir", type=Path, default=OPS_DIR,
                   help="default: WBES_DATA_DIR, gli npz con operatori")
    p.add_argument("--metric", type=str, default="latent_v1", help="nome della sottocartella")
    p.add_argument("--checkpoint", type=str, default=os.environ.get("WBES_CKPT", ""),
                   help="default: WBES_CKPT, cioe' il checkpoint v1 del repo")
    p.add_argument("--config-json", type=str, default="", help="config.json esplicito")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--subject-set", type=str, default="heldout", choices=common.SUBJECT_SETS)
    p.add_argument("--topology", type=str, default="original",
                   help="una o piu' topologie separate da virgola: una matrice "
                        "same-topology per ognuna (default: original)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    args.topologies = [t.strip() for t in args.topology.split(",") if t.strip()]
    unknown = [t for t in args.topologies if t not in common.TOPOLOGIES]
    if unknown:
        raise SystemExit(f"topologie sconosciute: {unknown} (attese {common.TOPOLOGIES})")
    return args


def build_v1_model(checkpoint_path: Path, config_json: str, device):
    """Modello e argomenti della run, ricostruiti come fa l'eval del repo."""
    from robustness.model_helpers import build_model
    from robustness.posthoc_runner import load_checkpoint_bundle, merge_run_args

    base_args = merge_run_args(checkpoint_path, explicit_config_json=config_json)
    model_args = SimpleNamespace(**base_args)
    model = build_model(args=model_args, device=device)
    bundle = load_checkpoint_bundle(checkpoint_path)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    print(f"[latent-matrix] modello={getattr(model_args, 'model', '?')} "
          f"latent_dim={getattr(model_args, 'latent_dim', '?')} "
          f"parametri={sum(p.numel() for p in model.parameters())}", flush=True)
    return model, model_args


def encode_subjects(model, subjects: list[str], topology: str, device, args) -> dict[str, np.ndarray]:
    """Latente per ogni soggetto nella topologia data, con cache su disco."""
    import torch

    from robustness.data_utils import GTReadyDataset, sample_to_device
    from robustness.model_helpers import forward_model

    cache_path = args.out_root / "embeddings" / f"{args.metric}_{topology}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, np.ndarray] = {}
    if cache_path.exists() and not args.overwrite:
        with np.load(cache_path) as z:
            done = {k: z[k] for k in z.files}
    names = [common.mesh_name(subject, topology) for subject in subjects]
    todo = [name for name in names if name not in done]
    if not todo:
        print(f"[latent-matrix] {len(names)} latenti gia' in cache ({cache_path})", flush=True)
        return done

    dataset = GTReadyDataset(str(args.ops_dir))
    name_to_idx = {Path(f).stem: i for i, f in enumerate(dataset.files)}
    missing = [name for name in todo if name not in name_to_idx]
    if missing:
        raise SystemExit(f"{len(missing)} mesh senza operatori in {args.ops_dir}: {missing[:3]}\n"
                         "  rigenerali con aau/submit.sh precompute_ops.sbatch")

    print(f"[latent-matrix] {len(todo)}/{len(names)} latenti da calcolare "
          f"(dataset: {len(dataset.files)} npz in {args.ops_dir})", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for i, name in enumerate(todo, start=1):
            sample_d = sample_to_device(dataset[name_to_idx[name]], device=device)
            z, _ = forward_model(model=model, sample_dict=sample_d, V_in=sample_d["verts"],
                                 return_gate_info=False, add_noise=False)
            done[name] = z.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if i % 25 == 0 or i == len(todo):
                print(f"[latent-matrix] {i}/{len(todo)} "
                      f"({i / max(time.time() - t0, 1e-9):.1f}/s)", flush=True)
                np.savez(cache_path, **done)
    np.savez(cache_path, **done)
    print(f"[latent-matrix] encoding finito in {time.time() - t0:.0f}s", flush=True)
    return done


def build_matrix(model, subjects: list[str], topology: str, device, args,
                 checkpoint_path: Path) -> None:
    """Matrice same-topology per una topologia, nel formato di ``common.save_matrix``."""
    out_path = common.matrix_path(args.metric, topology, topology, args.out_root)
    latents = encode_subjects(model, subjects, topology, device, args)

    Z = np.stack([latents[common.mesh_name(s, topology)] for s in subjects]).astype(np.float64)
    pair_i, pair_j = common.subject_pair_indices(len(subjects))
    D = common.empty_matrix(len(subjects))
    D[pair_i, pair_j] = np.linalg.norm(Z[pair_i] - Z[pair_j], axis=1)
    common.save_matrix(out_path, D, subjects, args.metric, topology, topology,
                       checkpoint=str(checkpoint_path), ops_dir=str(args.ops_dir),
                       distance="euclidean", latent_dim=int(Z.shape[1]))
    finite = D[pair_i, pair_j]
    print(f"[latent-matrix] {topology}: {finite.size} coppie: min {finite.min():.4f} mediana "
          f"{np.median(finite):.4f} max {finite.max():.4f}", flush=True)
    print(f"[latent-matrix] matrice {out_path}")


def main() -> None:
    import torch

    args = parse_args()
    todo = [t for t in args.topologies
            if args.overwrite
            or not common.matrix_path(args.metric, t, t, args.out_root).exists()]
    for topology in args.topologies:
        if topology not in todo:
            print(f"[latent-matrix] {common.matrix_path(args.metric, topology, topology, args.out_root)}"
                  " gia' presente, salto (--overwrite per rifarla)")
    if not todo:
        return
    if not args.checkpoint:
        raise SystemExit("serve --checkpoint (o WBES_CKPT): senza non c'e' nessun modello v1")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint non trovato: {checkpoint_path}")

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    subjects = common.subject_set(args.subject_set)
    print(f"[latent-matrix] soggetti={len(subjects)} ({args.subject_set}, topologie "
          f"{todo}) device={device} ckpt={checkpoint_path}", flush=True)

    model, _ = build_v1_model(checkpoint_path, args.config_json, device)
    for topology in todo:
        build_matrix(model, subjects, topology, device, args, checkpoint_path)


if __name__ == "__main__":
    main()
