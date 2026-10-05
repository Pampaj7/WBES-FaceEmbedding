#!/usr/bin/env python3
"""Controlli sul token di taglia, su un checkpoint addestrato col token (smoke delle ablazioni v3):
  1. sample_to_device agganciato porta size_token fino al forward;
  2. il token entra nell'embedding: +1 std di token cambia z;
  3. un campione senza token ferma il forward (RuntimeError), invece di proseguire.

    aau/run.sh aau/models/check_size_token.py --checkpoint <pth> --data-dir <vista> --size-token-json <bfm.json>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2_work/fastio"))
sys.path.insert(0, str(REPO_ROOT / "v2_work/transfer"))
sys.path.insert(0, str(THIS_DIR))

import train_fast  # noqa: E402,F401  (path del pacchetto robustness)
import robustness.data_utils as du  # noqa: E402
import robustness.model_helpers as mh  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--size-token-json", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from ablation_hooks import SIZE_KEY, DiffusionEncoderSizeToken, install

    install(size_token_json=args.size_token_json, eval_frame="rms")
    from eval_transfer import resolve_checkpoint

    ckpt, cfg = resolve_checkpoint(args.checkpoint)
    dev = torch.device(args.device)
    m = mh.build_model(SimpleNamespace(**cfg), dev)
    pack = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(pack["state_dict"] if "state_dict" in pack else pack, strict=True)
    m.eval()
    assert isinstance(m, DiffusionEncoderSizeToken), type(m)

    ds = du.GTReadyDataset(str(args.data_dir))
    with torch.inference_mode():
        # campione costruito DENTRO inference_mode, come negli script di eval (fast_data._rebuild_sparse)
        s = du.sample_to_device(ds[0], dev)
        assert SIZE_KEY in s, "sample_to_device ha perso il token"
        z0, _ = mh.forward_model(m, s, s["verts"], False, False)
        s2 = dict(s)
        s2[SIZE_KEY] = s[SIZE_KEY] + 1.0
        z1, _ = mh.forward_model(m, s2, s2["verts"], False, False)
        dz = float((z1 - z0).norm() / z0.norm())
        print(f"[token] {ds.files[0]}: token {float(s[SIZE_KEY]):+.3f}; +1 std cambia z del {100 * dz:.2f}%")
        assert dz > 1e-4, "il token non entra nell'embedding"
        s3 = {k: v for k, v in s.items() if k != SIZE_KEY}
        try:
            mh.forward_model(m, s3, s3["verts"], False, False)
        except RuntimeError as e:
            print(f"[token] OK, senza token il forward si ferma: {e}")
        else:
            raise SystemExit("ERRORE: forward senza token NON fallito")
    print("[token] controlli OK")


if __name__ == "__main__":
    main()
