#!/usr/bin/env python3
"""Embedding globali di Uni3D e OpenShape (zero-shot, pesi pubblici) delle 600 mesh valutate di HIFI3D.

    singularity exec --nv $CONTAINER bash -c 'source $COMP_VENV/bin/activate; \\
        python3 aau/competitors/comp_embed.py --model uni3d --view-dir datasets/HIFI3D/eval_view/npz \\
        --out aau/runs/competitors_hifi3d/emb_uni3d.npz'
    (comp_embed.sbatch)

Protocollo in ``aau/runs/competitors_hifi3d/protocol.md``. In breve: 10.000 punti uniformi per
area sulla superficie (seme per mesh), rotazione opzionale (``--rotation rx90``, ablazione),
normalizzazione del loro codice (centro = media, scala = raggio massimo), RGB 0.4, encoder di
punti del modello, embedding prima della normalizzazione L2 (la distanza 1 - coseno la fa il
summary).

Il codice dei modelli e' quello dei loro repo, importato dai cloni in external_models/ senza
modificarlo. Le sole dipendenze sostituite sono le FPS da estensioni CUDA (``pointnet2_ops`` per
Uni3D, ``dgl.geometry`` per OpenShape): moduli finti in ``sys.modules`` con una FPS in torch puro
che parte dall'indice 0. I package ``models`` dei due repo si caricano come package sintetici
(``__path__`` sulla cartella, senza eseguire il loro ``__init__``, che per OpenShape importa
MinkowskiEngine). Il checkpoint si carica con ``strict=True``: chiavi mancanti o in piu' = errore.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "zs3dmm"))
sys.path.insert(0, str(THIS_DIR.parent / "ict"))

from zs_stage import TOPOLOGIES, select_subjects  # noqa: E402
from ict_summarize import stable_seed  # noqa: E402

N_POINTS = 10_000
RGB = 0.4
SAMPLE_SEED = 1234
# Rx(+90): (x, y, z) -> (x, -z, y), l'alto +y di HIFI3D va in +z (asse verticale dei dati di training)
ROTATIONS = {"none": np.eye(3), "rx90": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("uni3d", "openshape"), required=True)
    p.add_argument("--view-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rotation", choices=sorted(ROTATIONS), default="none")
    p.add_argument("--eval-seed", type=int, default=1234, help="WBES_EVAL_SEED: scelta dei soggetti")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                   help="cpu: stesso calcolo in fp32, quando nessuna L40S e' schedulabile")
    p.add_argument("--limit", type=int, default=0, help="solo le prime N mesh (prova)")
    return p.parse_args()


# ------------------------------------------------------------------------------ punti

def sample_surface(V: np.ndarray, F: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` punti uniformi per area: triangolo ~ area, baricentriche uniformi (riflessione sqrt)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tri = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w = np.stack([1 - s, s * (1 - r2), s * r2], axis=1)
    return w[:, :1] * a[tri] + w[:, 1:2] * b[tri] + w[:, 2:] * c[tri]


def normalize_pc(pc: np.ndarray) -> np.ndarray:
    """Uni3D ``pc_norm`` = OpenShape ``normalize_pc``: centro = media, scala = raggio massimo."""
    pc = pc - pc.mean(axis=0)
    return pc / np.max(np.linalg.norm(pc, axis=1))


def mesh_points(path: Path, subject: str, topology: str, rot: np.ndarray) -> np.ndarray:
    with np.load(path) as z:
        V, F = z["V"].astype(np.float64), z["F"].astype(np.int64)
    pts = sample_surface(V, F, N_POINTS, np.random.default_rng(stable_seed(SAMPLE_SEED, subject, topology)))
    return normalize_pc(pts @ rot.T).astype(np.float32)


# ------------------------------------------------------------------------------ FPS

def fps_index(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """FPS su (B, N, 3) a partire dall'indice 0, come ``pointnet2_ops.furthest_point_sample``."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device)
    far = torch.zeros(B, dtype=torch.long, device=xyz.device)
    rows = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        idx[:, i] = far
        d = ((xyz - xyz[rows, far][:, None, :]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return idx


def install_fps_stubs() -> None:
    p2 = types.ModuleType("pointnet2_ops.pointnet2_utils")
    p2.furthest_point_sample = lambda data, number: fps_index(data, number).int()
    p2.gather_operation = lambda feats, idx: torch.gather(
        feats, 2, idx.long()[:, None, :].expand(-1, feats.shape[1], -1))
    pkg = types.ModuleType("pointnet2_ops")
    pkg.pointnet2_utils = p2
    geo = types.ModuleType("dgl.geometry")
    geo.farthest_point_sampler = lambda pos, npoints, start_idx=None: fps_index(pos, npoints)
    dgl = types.ModuleType("dgl")
    dgl.geometry = geo
    sys.modules.update({"pointnet2_ops": pkg, "pointnet2_ops.pointnet2_utils": p2, "dgl": dgl, "dgl.geometry": geo})


def synthetic_package(name: str, path: Path) -> None:
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg


# ------------------------------------------------------------------------------ modelli

def load_checkpoint(path: Path) -> dict:
    """Solo tensori (``weights_only``), mappati dal file: il nodo L40S libero ha pochi GB di RAM.

    Il checkpoint di Uni3D contiene un ``set`` (fuori dal dizionario dei pesi): si ammette solo quel tipo
    builtin, il resto dell'unpickler resta quello ristretto di ``weights_only``.
    """
    torch.serialization.add_safe_globals([set])
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def build_uni3d(src: Path, ckpt: Path) -> tuple[torch.nn.Module, callable]:
    import timm

    synthetic_package("uni3d_models", src / "models")
    pe = importlib.import_module("uni3d_models.point_encoder")
    # Parametri di scripts/inference.sh, variante giant
    args = SimpleNamespace(pc_model="eva_giant_patch14_560", pc_feat_dim=1408, embed_dim=1024, group_size=64,
                           num_group=512, pc_encoder_dim=512, patch_dropout=0.0, drop_path_rate=0.0)
    transformer = timm.create_model(args.pc_model, checkpoint_path="", drop_path_rate=args.drop_path_rate)

    class Uni3D(torch.nn.Module):  # stessi nomi di models/uni3d.py (che importa losses -> utils del training)
        def __init__(self, point_encoder):
            super().__init__()
            self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.point_encoder = point_encoder

        def encode_pc(self, pc):
            return self.point_encoder(pc[:, :, :3].contiguous(), pc[:, :, 3:].contiguous())

    model = Uni3D(pe.PointcloudEncoder(transformer, args))
    sd = load_checkpoint(ckpt)["module"]
    if next(iter(sd)).startswith("module"):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model, lambda xyz, rgb: model.encode_pc(torch.cat((xyz, rgb), dim=-1))


def build_openshape(src: Path, ckpt: Path) -> tuple[torch.nn.Module, callable]:
    synthetic_package("openshape_models", src / "src" / "models")
    ppat = importlib.import_module("openshape_models.ppat")
    # README: model.name=PointBERT model.scaling=4 model.use_dense=True; in 6 (xyz + rgb), out 1280 (bigG)
    model = ppat.make(SimpleNamespace(model=SimpleNamespace(scaling=4, in_channel=6, out_channel=1280)))
    # Come load_model di src/example.py: solo le chiavi con "module", re.sub del loro pattern
    pattern = re.compile("module.")
    sd = {re.sub(pattern, "", k): v for k, v in load_checkpoint(ckpt)["state_dict"].items() if re.search("module", k)}
    model.load_state_dict(sd, strict=True)
    return model, lambda xyz, rgb: model(xyz, torch.cat((xyz, rgb), dim=-1))


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if args.device == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    install_fps_stubs()
    # Parametri creati direttamente sulla GPU: in RAM non passa mai il modello intero (Uni3D-g, 1B in fp32)
    with torch.device(device):
        if args.model == "uni3d":
            model, encode = build_uni3d(Path(os.environ["COMP_UNI3D_SRC"]), Path(os.environ["COMP_UNI3D_CKPT"]))
        else:
            model, encode = build_openshape(Path(os.environ["COMP_OPENSHAPE_SRC"]),
                                            Path(os.environ["COMP_OPENSHAPE_CKPT"]))
    model = model.to(device).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[comp-emb] {args.model}: {n_par / 1e6:.1f} M parametri, strict load ok, rotazione {args.rotation}, "
          f"device {args.device}", flush=True)

    subjects = select_subjects(args.view_dir, args.eval_seed)
    keys = [(s, t) for s in subjects for t in TOPOLOGIES]
    if args.limit:
        keys = keys[: args.limit]
    rot = ROTATIONS[args.rotation]
    E = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(keys), args.batch):
            chunk = keys[i: i + args.batch]
            pts = np.stack([mesh_points(args.view_dir / f"{s}_GTready_{t}.npz", s, t, rot) for s, t in chunk])
            xyz = torch.from_numpy(pts).to(device)
            rgb = torch.full_like(xyz, RGB)
            E.append(encode(xyz, rgb).float().cpu().numpy())
            if (i // args.batch) % 15 == 0:
                print(f"[comp-emb] {i + len(chunk)}/{len(keys)} ({time.time() - t0:.0f}s)", flush=True)
    E = np.concatenate(E)
    if not np.isfinite(E).all():
        raise SystemExit("embedding non finiti")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, E=E, subjects=np.asarray([k[0] for k in keys]), topologies=np.asarray([k[1] for k in keys]),
             model=args.model, rotation=args.rotation, n_points=N_POINTS)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    C = En @ En.T
    off = C[np.triu_indices(len(C), 1)]
    print(f"[comp-emb] E {E.shape}, norma mediana {np.median(np.linalg.norm(E, axis=1)):.3f}, coseno fuori "
          f"diagonale min/mediano/max {off.min():.4f}/{np.median(off):.4f}/{off.max():.4f}", flush=True)
    print(f"[comp-emb] scritto {args.out}", flush=True)


if __name__ == "__main__":
    main()
