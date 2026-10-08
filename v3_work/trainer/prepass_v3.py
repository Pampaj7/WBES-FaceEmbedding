#!/usr/bin/env python3
"""aau/data_scale/prepass_ops.py con il build_grad vettorizzato di E9 (grad_vec.py), stessa CLI e stesse uscite.

Con il metodo spawn del Pool i worker reimportano questo modulo come __mp_main__: l'install() qui sotto, a
livello di modulo, vale quindi anche per loro. Il trainer v3 lo usa con --prepass-grad vec.
"""
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (THIS_DIR, REPO_ROOT / "aau/data_scale", REPO_ROOT / "diffusion-net/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import grad_vec  # noqa: E402

grad_vec.install()

import prepass_ops  # noqa: E402

if __name__ == "__main__":
    prepass_ops.main()
