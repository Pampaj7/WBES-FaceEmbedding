#!/usr/bin/env python3
"""Il terzo criterio di WS3b: distanza latente del modello v1 fra ricostruzione e GT.

    aau/submit.sh recon/ws3b_latent.sbatch

Stessa catena di ``aau/multiface/ws3a_latent.py``, cioe' quella dell'eval del repo

    merge_run_args(ckpt) -> build_model -> load_state_dict -> GTReadyDataset
    -> sample_to_device -> forward_model(V_in=sample["verts"], add_noise=False)

e distanza = norma L2 fra i due latenti, come in ``_evaluate_scenario``.  Nessun file di
ricerca e' stato toccato.

Cosa entra nell'encoder
-----------------------
Da una parte ``gt_face`` (la mesh tracciata ritagliata a 95 mm dalla punta del naso), che
e' anche il lato GT dei due criteri geometrici; dall'altra ``recon_face``, la ricostruzione
ritagliata alla stessa regione e decimata allo stesso numero di triangoli.  Le due
topologie restano diverse -- ed e' il punto: la metrica appresa e' l'unica dei tre criteri
che non chiede ne' corrispondenza ne' allineamento.  Gli operatori sono k_eig 128 con la
convenzione STANDARD, quella con cui il checkpoint v1 e' stato addestrato.

Il costo e' per MESH e non per coppia: 2848 GT + 3 x 3084 ricostruzioni, con cache su
disco in ``embeddings/``, cosi' il modo ``pairs`` non ricalcola niente.

``--recon-variant gtclip``
    Dall'altra parte entra ``recon_face_gtclip`` invece di ``recon_face``: la stessa
    ricostruzione ristretta alla patch GT di ``chamfer_gtclip_mm`` (``ws3b_gtclip_meshes.py``),
    con operatori ricalcolati.  Il lato GT e' lo stesso, e il suo latente si riprende dalla
    cache.  Colonna ``latent_v1_gtclip`` in ``gt_latent_gtclip_<metodo>.csv``, cache
    ``embeddings/latent_v1_gtclip_<metodo>.npz``: niente della corsa originale viene
    riscritto.  Solo ``--mode gt``: la patch GT e' definita contro la GT dello stesso frame,
    fra due ricostruzioni non esiste.

``--metric`` e ``--ops-suffix``
    Un altro checkpoint, con la convenzione di operatori con cui e' stato addestrato.  Il
    congiunto BFM+ICT (WS2) e' addestrato ad AREA UNITARIA, quindi

        --metric latent_joint --ops-suffix _withops_areanorm

    con gli operatori di ``ws3b_ops_areanorm.sbatch`` (``gt_face_withops_areanorm``,
    ``recon_face_withops_areanorm/<metodo>``, ``recon_face_gtclip_withops_areanorm/<metodo>``).
    Colonne ``latent_joint`` / ``latent_joint_gtclip`` in ``gt_latent_joint_<metodo>.csv`` /
    ``gt_latent_joint_gtclip_<metodo>.csv``, cache ``embeddings/latent_joint_*.npz``: i
    file del v1 hanno il nome storico e non vengono toccati.
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

import ws3b_common as common  # noqa: E402

INTRINSIC_DIR = common.REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"
sys.path.insert(0, str(INTRINSIC_DIR))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--mode", type=str, default="all", choices=("all", "gt", "pairs"))
    p.add_argument("--recon-variant", type=str, default="face", choices=("face", "gtclip"),
                   help="face = recon_face (corsa originale); gtclip = ritagliata alla patch GT")
    p.add_argument("--methods", type=str, default=",".join(common.METHODS))
    p.add_argument("--metric", type=str, default="latent_v1",
                   help="nome della colonna, dei csv e della cache (latent_v1 = nomi storici)")
    p.add_argument("--ops-suffix", type=str, default=os.environ.get("WBES_WS3B_OPS_SUFFIX", "_withops"),
                   help="_withops = convenzione standard (v1); _withops_areanorm = area unitaria")
    p.add_argument("--checkpoint", type=str, default=os.environ.get("WBES_CKPT", ""),
                   help="default: WBES_CKPT, cioe' il checkpoint v1 del repo")
    p.add_argument("--config-json", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def build_v1_model(checkpoint_path: Path, config_json: str, device):
    """Modello e argomenti della run, ricostruiti come fa l'eval del repo."""
    from robustness.model_helpers import build_model
    from robustness.posthoc_runner import load_checkpoint_bundle, merge_run_args

    model_args = SimpleNamespace(**merge_run_args(checkpoint_path, explicit_config_json=config_json))
    model = build_model(args=model_args, device=device)
    model.load_state_dict(load_checkpoint_bundle(checkpoint_path)["state_dict"], strict=True)
    model.eval()
    print(f"[ws3b-latent] modello={getattr(model_args, 'model', '?')} "
          f"latent_dim={getattr(model_args, 'latent_dim', '?')} "
          f"parametri={sum(p.numel() for p in model.parameters())}", flush=True)
    return model


def encode(model, tag: str, ops_dir: Path, names: list[str], device, args) -> dict[str, np.ndarray]:
    """Latente per ogni mesh, con cache su disco in embeddings/<tag>.npz."""
    import torch

    from robustness.data_utils import GTReadyDataset, sample_to_device
    from robustness.model_helpers import forward_model

    cache_path = args.out_root / "embeddings" / f"{args.metric}_{tag}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, np.ndarray] = {}
    if cache_path.exists() and not args.overwrite:
        with np.load(cache_path) as z:
            done = {k: z[k] for k in z.files}
    todo = [name for name in names if name not in done]
    if not todo:
        print(f"[ws3b-latent] {tag}: {len(names)} latenti gia' in cache", flush=True)
        return done

    dataset = GTReadyDataset(str(ops_dir))
    name_to_idx = {Path(f).stem: i for i, f in enumerate(dataset.files)}
    missing = [name for name in todo if name not in name_to_idx]
    if missing:
        raise SystemExit(f"{len(missing)} mesh senza operatori in {ops_dir}: {missing[:3]}")

    print(f"[ws3b-latent] {tag}: {len(todo)}/{len(names)} latenti da calcolare", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for i, name in enumerate(todo, start=1):
            sample_d = sample_to_device(dataset[name_to_idx[name]], device=device)
            z, _ = forward_model(model=model, sample_dict=sample_d, V_in=sample_d["verts"],
                                 return_gate_info=False, add_noise=False)
            done[name] = z.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if i % 500 == 0 or i == len(todo):
                print(f"[ws3b-latent] {tag}: {i}/{len(todo)} "
                      f"({i / max(time.time() - t0, 1e-9):.1f}/s)", flush=True)
                np.savez(cache_path, **done)
    np.savez(cache_path, **done)
    print(f"[ws3b-latent] {tag}: fine in {time.time() - t0:.0f}s", flush=True)
    return done


def main() -> None:
    import torch

    args = parse_args()
    args.out_root = args.out_root.resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not args.checkpoint:
        raise SystemExit("serve --checkpoint (o WBES_CKPT): senza non c'e' nessun modello v1")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint non trovato: {checkpoint_path}")

    if args.recon_variant == "gtclip" and args.mode != "gt":
        raise SystemExit("--recon-variant gtclip vale solo con --mode gt")
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available())
                          else "cpu")
    items = common.load_manifest(common.manifest_path(args.out_root))
    records = common.load_pairs(path=common.protocol_path(args.out_root)) \
        if args.mode in ("all", "pairs") else []
    print(f"[ws3b-latent] device={device} ckpt={checkpoint_path} metric={args.metric} "
          f"ops={args.ops_suffix} {len(items)} ricostruzioni, {len(records)} coppie", flush=True)

    model = build_v1_model(checkpoint_path, args.config_json, device)
    gt = encode(model, "gt_face", common.gt_face_ops_dir(args.out_root, args.ops_suffix),
                sorted({it.gt_name for it in items}), device, args)

    gtclip = args.recon_variant == "gtclip"
    metric = f"{args.metric}_gtclip" if gtclip else args.metric
    ops_label = "k_eig 128, convenzione " + ("ad area unitaria" if args.ops_suffix.endswith("_areanorm")
                                             else "standard")
    for method in methods:
        ops_dir = (common.recon_face_gtclip_ops_dir if gtclip else common.recon_face_ops_dir)(
            method, args.out_root, args.ops_suffix)
        present = {p.stem for p in ops_dir.glob("*.npz")}
        mine = [it for it in items if it.name in present]
        if len(mine) != len(items):
            print(f"[ws3b-latent] ATTENZIONE {method}: {len(mine)}/{len(items)} con operatori",
                  flush=True)
        latents = encode(model, f"gtclip_{method}" if gtclip else method, ops_dir,
                         [it.name for it in mine], device, args)

        if args.mode in ("all", "gt"):
            path = (common.gt_latent_gtclip_csv_path if gtclip else common.gt_latent_csv_path)(
                method, args.out_root, args.metric)
            if path.exists() and not args.overwrite:
                print(f"[ws3b-latent] {method}: {path.name} c'e' gia', salto", flush=True)
            else:
                A = np.stack([latents[it.name] for it in mine]).astype(np.float64)
                B = np.stack([gt[it.gt_name] for it in mine]).astype(np.float64)
                values = np.linalg.norm(A - B, axis=1)
                rows = [{**{k: getattr(it, k) for k in common.ITEM_FIELDS}, metric: float(v)}
                        for it, v in zip(mine, values)]
                common.write_rows(path, common.ITEM_FIELDS + (metric,), rows, method=method,
                                  checkpoint=str(checkpoint_path), latent_dim=int(A.shape[1]),
                                  ops=ops_label, recon_dir=str(ops_dir))
                print(f"[ws3b-latent] {method} gt: {len(rows)} righe (mediana "
                      f"{np.median(values):.4f}) -> {path.name}", flush=True)

        if args.mode in ("all", "pairs"):
            path = common.pair_csv_path(args.metric, method, args.out_root)
            keep = [r for r in records if r.name_a in latents and r.name_b in latents]
            if path.exists() and not args.overwrite:
                print(f"[ws3b-latent] {method}: {path.name} c'e' gia', salto", flush=True)
                continue
            A = np.stack([latents[r.name_a] for r in keep]).astype(np.float64)
            B = np.stack([latents[r.name_b] for r in keep]).astype(np.float64)
            values = np.linalg.norm(A - B, axis=1)
            rows = [{**{k: getattr(r, k) for k in common.PAIR_FIELDS if k != "distance"},
                     "distance": float(v)} for r, v in zip(keep, values)]
            common.write_rows(path, common.PAIR_FIELDS, rows, metric=args.metric, method=method,
                              checkpoint=str(checkpoint_path), latent_dim=int(A.shape[1]),
                              ops=ops_label)
            print(f"[ws3b-latent] {method} pairs: {len(rows)} coppie -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
