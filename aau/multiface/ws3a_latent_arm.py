#!/usr/bin/env python3
"""ws3a_latent.py per i bracci con frame rms e token di taglia (ablazioni B, E e E congiunto).

Stesso schema di aau/models/eval_ablation.py: installa gli hook di aau/models/ablation_hooks.py
(frame dell'input nel forward, modello e dataset col token) e poi lancia ws3a_latent.main()
senza toccarlo. In piu' sceglie la tabella dei token PER TOPOLOGIA: le cartelle Multiface hanno
gli stessi nomi di file in tutte le topologie, quindi una tabella per nome non basterebbe
(aau/multiface/mf_size_token.py ne scrive una per topologia, <token-dir>/<topologia>.json).

Il frame rms ri-inquadra V_in nel forward: valido perche' WS3a usa solo mesh pulite (nessuna
perturbazione dei vertici). Gli operatori li sceglie WBES_MF_OPS_SUFFIX, come per ws3a_latent.py.

    aau/run.sh aau/multiface/ws3a_latent_arm.py --frame rms --token-dir <dir> -- \
        --checkpoint <pth> --metric latent_abl_B_tokA --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2_work/fastio"))
sys.path.insert(0, str(THIS_DIR.parent / "models"))
sys.path.insert(0, str(THIS_DIR))


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("uso: ws3a_latent_arm.py --frame {current,rms} [--token-dir D] -- <argomenti di ws3a_latent.py>")
    cut = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--frame", required=True, choices=["current", "rms"])
    p.add_argument("--token-dir", type=Path, default=None,
                   help="una tabella SizeTokenTable per topologia: <dir>/<topologia>.json")
    known = p.parse_args(argv[:cut])

    import train_fast  # noqa: F401  (path del pacchetto robustness)
    import robustness.model_helpers  # noqa: F401
    import robustness.data_utils  # noqa: F401
    import ablation_hooks
    from ablation_hooks import SizeTokenTable, install

    import ws3a_latent

    first = None
    if known.token_dir is not None:
        first = known.token_dir / "tracked.json"
        if not first.is_file():
            raise SystemExit(f"tabella assente: {first}")
    install(size_token_json=first, eval_frame=known.frame)

    if known.token_dir is not None:
        orig_encode = ws3a_latent.encode_topology

        def encode_topology(model, topology, names, device, args):
            path = known.token_dir / f"{topology}.json"
            if not path.is_file():
                raise SystemExit(f"tabella dei token assente per {topology}: {path}")
            table = SizeTokenTable(path)
            ablation_hooks._STATE["table"] = table
            ablation_hooks._STATE["logged"].discard("token")
            print(f"[ws3a-arm] {topology}: token da {path} ({table.collection})", flush=True)
            return orig_encode(model, topology, names, device, args)

        ws3a_latent.encode_topology = encode_topology

    sys.argv = [str(Path(ws3a_latent.__file__))] + argv[cut + 1:]
    ws3a_latent.main()


if __name__ == "__main__":
    main()
