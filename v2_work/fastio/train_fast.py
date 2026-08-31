#!/usr/bin/env python
"""Run the v1 trainer with the whole training set resident in memory.

Same CLI as the v1 trainer (every unknown flag is forwarded verbatim), plus:
    --cache-residency ram|device   where the prepared samples live (default ram)
    --cache-workers N              threads used for the one-time preload (default 16)
    --cache-max-gb G               refuse to start above this projected footprint
    --no-cache                     bypass entirely, for an apples-to-apples v1 run

The v1 package under face_embedding/ is not modified: the cache is injected by
rebinding the `GTReadyDataset` name inside the robustness modules before main()
runs, so a run without --cache is byte-identical to v1.

Example (GPU node):
    ESUB_BYPASS=1 bsub -I -q p1i -app h100app -n 4 \
        -R "span[hosts=1] rusage[mem=200GB]" -gpu "num=1:mode=shared" -W 720 \
        .conda_env/bin/python v2_work/fastio/train_fast.py --data_dir ... --dist_npz ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"))
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/autoencoder"))
sys.path.insert(0, str(REPO_ROOT / "diffusion-net/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_data import CachedDataset, install  # noqa: E402


def install_masked_pooling(roi_threshold: float) -> None:
    """Swap in the masked encoder and route roi_mask to it, without touching frozen v1 code.

    Two rebinds are needed, because the frozen path splits the two things apart:
    `build_model` decides WHICH model is constructed, and `forward_model` is the only place
    that still holds the sample dict when the model is called -- the model itself is invoked
    positionally and never sees roi_mask. Patching both keeps the trainer, its CLI and its
    model signatures unchanged.
    """
    import robustness.model_helpers as mh

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "masked"))
    from model_masked import DiffusionEncoderOnlyMasked

    orig_build, orig_forward = mh.build_model, mh.forward_model

    def build_model(args, device):
        if getattr(args, "model", None) != "xyz_dn":
            raise ValueError("--masked-pooling is only wired for --model xyz_dn")
        m = DiffusionEncoderOnlyMasked(
            latent_dim=args.latent_dim, width=args.width, n_blocks=args.n_blocks,
            dropout=args.dropout, pool_mode=args.pool_mode, roi_threshold=roi_threshold,
        ).to(device)
        return m

    def forward_model(model, sample_dict, V_in, return_gate_info, add_noise):
        if isinstance(model, DiffusionEncoderOnlyMasked):
            model.set_roi_mask(sample_dict.get("roi_mask"))
        return orig_forward(model, sample_dict, V_in, return_gate_info, add_noise)

    mh.build_model, mh.forward_model = build_model, forward_model
    # train_runner and the eval helpers imported these names directly, so rebinding the
    # module attribute alone would leave their copies pointing at the originals.
    for mod_name in ("robustness.train_runner", "robustness.eval_utils"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        if hasattr(mod, "build_model"):
            mod.build_model = build_model
        if hasattr(mod, "forward_model"):
            mod.forward_model = forward_model
    print(f"[masked] pooling ristretto alla ROI (soglia {roi_threshold})", flush=True)
    _ = orig_build  # kept for symmetry/debugging


def install_point_backbone(n_samples: int, knn: int) -> None:
    """Swap in the operator-free point encoder, leaving trainer, losses and CLI untouched.

    Same two rebinds as install_masked_pooling and for the same reason: `build_model` chooses
    the model, `forward_model` is the only place that still holds the sample dict when the
    model is called. Nothing else in the frozen path needs to know which backbone is running,
    which is the point -- the comparison against xyz_dn then differs in the encoder alone.
    """
    import robustness.model_helpers as mh

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pointnet"))
    from model_point import PointEncoder

    orig_forward = mh.forward_model

    def build_model(args, device):
        m = PointEncoder(
            latent_dim=args.latent_dim, width=args.width, dropout=args.dropout,
            pool_mode=args.pool_mode, n_samples=n_samples, k=knn,
        ).to(device)
        n_par = sum(p.numel() for p in m.parameters())
        print(f"[point] PointEncoder attivo: {n_par} parametri, M={n_samples}, k={knn}", flush=True)
        return m

    def forward_model(model, sample_dict, V_in, return_gate_info, add_noise):
        if isinstance(model, PointEncoder):
            z = model(V_in, sample_dict.get("mass"), add_noise=add_noise)
            return z, mh._default_gate_info(z)
        return orig_forward(model, sample_dict, V_in, return_gate_info, add_noise)

    mh.build_model, mh.forward_model = build_model, forward_model
    for mod_name in ("robustness.train_runner", "robustness.eval_utils"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        if hasattr(mod, "build_model"):
            mod.build_model = build_model
        if hasattr(mod, "forward_model"):
            mod.forward_model = forward_model
    print("[point] backbone senza operatori: nessun autovalore letto", flush=True)


def install_frame(frame: str) -> None:
    """Re-frame every cached sample's vertices, leaving the frozen v1 loader untouched.

    Applied to the cache rather than to the loader because the replacement frames are, like the
    loader's own, a translation plus a uniform scale, and the composition cancels exactly (see
    v2_work/pointnet/frames.py). A run with --frame current is therefore byte-identical to a run
    without the flag, which is what makes it usable as the control arm of the 2x2.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pointnet"))
    from frames import reframe, FRAMES

    if frame not in FRAMES:
        raise SystemExit(f"--frame must be one of {FRAMES}, got {frame!r}")
    if frame == "current":
        print("[frame] current: nessuna modifica (braccio di controllo)", flush=True)
        return

    import fast_data as fd

    orig_getitem = fd.CachedDataset.__getitem__

    def __getitem__(self, idx):
        sample = orig_getitem(self, idx)
        v = sample.get("verts")
        if v is None:
            return sample
        out = dict(sample)
        out["verts"] = reframe(v, sample["mass"], sample["faces"], frame)
        return out

    fd.CachedDataset.__getitem__ = __getitem__
    print(f"[frame] {frame}: vertici ri-inquadrati (centroide pesato per massa)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--cache-residency", default="ram", choices=["ram", "device"])
    p.add_argument("--cache-workers", type=int, default=16)
    p.add_argument("--cache-max-gb", type=float, default=900.0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--masked-pooling", action="store_true",
                   help="restrict pooling to the potential well's region of interest "
                        "(requires operators carrying roi_mask)")
    p.add_argument("--roi-threshold", type=float, default=0.5)
    p.add_argument("--point-backbone", action="store_true",
                   help="use the operator-free point encoder instead of DiffusionNet")
    p.add_argument("--point-samples", type=int, default=2048)
    p.add_argument("--point-knn", type=int, default=20)
    p.add_argument("--frame", default="current",
                   choices=["current", "rms", "area", "global"],
                   help="canonical frame for the vertex coordinates; 'current' is the "
                        "frozen loader's vertex-mean + maxabs and is a no-op. 'global' is "
                        "the only one that is NOT a per-mesh map: it inverts the loader's "
                        "normalisation and applies one dataset-wide similarity instead, which "
                        "preserves the ground truth's ranks exactly")
    p.add_argument("--global-frame-json", type=Path,
                   default=Path(__file__).resolve().parents[1] / "xdomain" / "gt_matrices"
                           / "global_frame.json",
                   help="c0/s0 fitted on the TRAINING identities only "
                        "(see v2_work/xdomain/global_frame.py)")
    known, rest = p.parse_known_args()

    # the trainer parses its own argv; hand it everything we did not consume
    sys.argv = [sys.argv[0]] + rest

    import robustness.train_runner as tr

    args = tr.parse_args()

    if known.masked_pooling and known.point_backbone:
        raise SystemExit("--masked-pooling and --point-backbone are mutually exclusive")
    if known.masked_pooling:
        install_masked_pooling(float(known.roi_threshold))
    if known.point_backbone:
        install_point_backbone(int(known.point_samples), int(known.point_knn))
    if known.frame == "global":
        # Not a cache re-frame: it needs the raw coordinates back, so it is installed from its
        # own module and keyed on the sample name. See global_frame_loader's docstring for why
        # the cancellation argument that licenses `rms`/`area` does not apply here.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pointnet"))
        from global_frame_loader import install as install_global_frame
        install_global_frame([args.data_dir], known.global_frame_json)
    else:
        install_frame(str(known.frame))

    if not known.no_cache:
        device = args.device if known.cache_residency == "device" else None
        t0 = time.time()
        ds = CachedDataset(
            args.data_dir,
            workers=known.cache_workers,
            residency=known.cache_residency,
            device=device,
            max_gb=known.cache_max_gb,
        )
        install(ds)
        print(f"[fastio] cache installed in {time.time()-t0:.0f}s; "
              f"training reads 0 bytes from NFS from here on", flush=True)
    else:
        print("[fastio] cache disabled (--no-cache): v1-identical data path", flush=True)

    tr.run_training(args)


if __name__ == "__main__":
    main()
