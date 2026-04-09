from __future__ import annotations

import os
import sys
from pathlib import Path


AUTOENCODER_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUTOENCODER_DIR.parents[2]


def _diffusion_net_candidates() -> list[Path]:
    env_candidates = [
        os.environ.get("WBES_DIFFUSION_NET_SRC", "").strip(),
        os.environ.get("DIFFUSION_NET_SRC", "").strip(),
    ]

    repo_candidates = [
        REPO_ROOT / "diffusion-net" / "src",
        REPO_ROOT.parent / "diffusion-net" / "src",
    ]

    legacy_candidates = [
        Path("/equilibrium/lpampaloni/diffusion-net/src"),
        Path("/home/pampaj/diffusion-net/src"),
        Path("/seidenas/users/lpampaloni/diffusion-net/src"),
    ]

    ordered: list[Path] = []
    seen: set[str] = set()

    for candidate in env_candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        key = str(path)
        if key not in seen:
            seen.add(key)
            ordered.append(path)

    for candidate in [*repo_candidates, *legacy_candidates]:
        path = candidate.expanduser().resolve()
        key = str(path)
        if key not in seen:
            seen.add(key)
            ordered.append(path)

    return ordered


def resolve_diffusion_net_src() -> Path:
    for candidate in _diffusion_net_candidates():
        if (candidate / "diffusion_net").is_dir():
            return candidate
    searched = "\n".join(f"  - {path}" for path in _diffusion_net_candidates())
    raise ImportError(
        "Could not locate diffusion-net/src. "
        "Set WBES_DIFFUSION_NET_SRC or DIFFUSION_NET_SRC to the directory containing the diffusion_net package.\n"
        f"Searched:\n{searched}"
    )


def ensure_diffusion_net_on_syspath() -> Path:
    diffusion_net_src = resolve_diffusion_net_src()
    diffusion_net_src_str = str(diffusion_net_src)
    if diffusion_net_src_str not in sys.path:
        sys.path.append(diffusion_net_src_str)
    return diffusion_net_src
