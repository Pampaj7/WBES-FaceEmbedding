#!/usr/bin/env python3
"""Distanza latente del modello v1 sulle coppie del protocollo WS3a (Multiface).

L'encoder non viene reimplementato: si usa la stessa catena dell'eval del repo, quella di
``face_embedding/.../perturbated/compare_model_vs_chamfer_rankings.py``, cioe'

    merge_run_args(ckpt) -> build_model -> load_state_dict -> GTReadyDatasetNPZ
    -> sample_to_device -> forward_model(V_in=sample["verts"], add_noise=False)

e la distanza fra due mesh e' la norma L2 fra i due latenti, come in ``_evaluate_scenario``
(``torch.linalg.vector_norm(Z[i] - Z[j])``).  Nessun file di ricerca e' stato toccato.

Il costo e' per MESH (2840 x 3 topologie = 8520 forward), non per coppia: gli embedding
vanno in cache in ``embeddings/latent_v1_<topologia>.npz`` e le 4 coppie di topologie li
riusano.  Il grosso del tempo e' l'I/O: gli npz con gli operatori sono 21 GB in tutto e
CephFS a processo singolo fa 69 MB/s.

Operatori: di default ``_withops`` (convenzione standard), perche' e' quella con cui il
checkpoint v1 e' stato addestrato.  Per un modello areanorm si esporta
``WBES_MF_OPS_SUFFIX=_withops_areanorm``.

  aau/submit.sh multiface/ws3a_latent.sbatch
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ws3a_common as common  # noqa: E402

INTRINSIC_DIR = common.REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"
sys.path.insert(0, str(INTRINSIC_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--metric", type=str, default="latent_v1", help="nome usato nei csv")
    p.add_argument("--checkpoint", type=str, default=os.environ.get("WBES_CKPT", ""),
                   help="default: WBES_CKPT, cioe' il checkpoint v1 del repo")
    p.add_argument("--config-json", type=str, default="", help="config.json esplicito")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-pairs-per-class", type=int, default=0,
                   help="0 = protocollo intero; 50 = le 200 coppie del test di velocita'")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def build_v1_model(checkpoint_path: Path, config_json: str, device):
    """Modello e argomenti della run, ricostruiti come fa l'eval del repo."""
    import torch

    from robustness.model_helpers import build_model
    from robustness.posthoc_runner import load_checkpoint_bundle, merge_run_args

    base_args = merge_run_args(checkpoint_path, explicit_config_json=config_json)
    model_args = SimpleNamespace(**base_args)
    model = build_model(args=model_args, device=device)
    bundle = load_checkpoint_bundle(checkpoint_path)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    print(f"[ws3a-latent] modello={getattr(model_args, 'model', '?')} "
          f"latent_dim={getattr(model_args, 'latent_dim', '?')} "
          f"parametri={sum(p.numel() for p in model.parameters())}", flush=True)
    return model, model_args


def encode_topology(model, topology: str, names: list[str], device, args) -> dict[str, np.ndarray]:
    """Latente per ogni mesh della topologia, con cache su disco."""
    import torch

    from robustness.data_utils import GTReadyDataset, sample_to_device
    from robustness.model_helpers import forward_model

    cache_path = args.out_root / "embeddings" / f"{args.metric}_{topology}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, np.ndarray] = {}
    if cache_path.exists() and not args.overwrite:
        with np.load(cache_path) as z:
            done = {k: z[k] for k in z.files}
    todo = [name for name in names if name not in done]
    if not todo:
        print(f"[ws3a-latent] {topology}: {len(names)} latenti gia' in cache", flush=True)
        return done

    dataset = GTReadyDataset(str(common.ops_dir(topology)))
    name_to_idx = {Path(f).stem: i for i, f in enumerate(dataset.files)}
    missing = [name for name in todo if name not in name_to_idx]
    if missing:
        raise SystemExit(f"{len(missing)} mesh senza operatori in {common.ops_dir(topology)}: "
                         f"{missing[:3]}")

    print(f"[ws3a-latent] {topology}: {len(todo)}/{len(names)} latenti da calcolare", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for i, name in enumerate(todo, start=1):
            sample_d = sample_to_device(dataset[name_to_idx[name]], device=device)
            z, _ = forward_model(model=model, sample_dict=sample_d, V_in=sample_d["verts"],
                                 return_gate_info=False, add_noise=False)
            done[name] = z.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if i % 500 == 0 or i == len(todo):
                print(f"[ws3a-latent] {topology}: {i}/{len(todo)} "
                      f"({i / max(time.time() - t0, 1e-9):.1f}/s)", flush=True)
                np.savez(cache_path, **done)
    np.savez(cache_path, **done)
    print(f"[ws3a-latent] {topology}: fine in {time.time() - t0:.0f}s", flush=True)
    return done


def main() -> None:
    import torch

    args = parse_args()
    if not args.checkpoint:
        raise SystemExit("serve --checkpoint (o WBES_CKPT): senza non c'e' nessun modello v1")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint non trovato: {checkpoint_path}")

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    records = common.load_pairs(max_pairs_per_class=args.max_pairs_per_class)
    topologies = common.topology_names(records)
    todo = [(ta, tb) for ta, tb in common.TOPOLOGY_PAIRS
            if args.overwrite or not common.csv_path(args.metric, ta, tb, args.out_root).exists()]
    print(f"[ws3a-latent] coppie={len(records)} device={device} ops={common.OPS_SUFFIX} "
          f"ckpt={checkpoint_path} da calcolare={len(todo)}/{len(common.TOPOLOGY_PAIRS)}", flush=True)
    if not todo:
        print("[ws3a-latent] niente da fare")
        return

    model, _ = build_v1_model(checkpoint_path, args.config_json, device)
    latents = {topology: encode_topology(model, topology, topologies[topology], device, args)
               for topology in sorted({t for pair in todo for t in pair})}

    for topology_a, topology_b in todo:
        t1 = time.time()
        A = np.stack([latents[topology_a][rec.name_a] for rec in records]).astype(np.float64)
        B = np.stack([latents[topology_b][rec.name_b] for rec in records]).astype(np.float64)
        values = np.linalg.norm(A - B, axis=1)
        out_path = common.csv_path(args.metric, topology_a, topology_b, args.out_root)
        common.write_distances(out_path, records, values, args.metric, topology_a, topology_b,
                               seconds=time.time() - t1, checkpoint=str(checkpoint_path),
                               ops_suffix=common.OPS_SUFFIX, latent_dim=int(A.shape[1]))
        print(f"[ws3a-latent] {topology_a}->{topology_b}: {len(values)} coppie -> {out_path}",
              flush=True)


if __name__ == "__main__":
    main()
