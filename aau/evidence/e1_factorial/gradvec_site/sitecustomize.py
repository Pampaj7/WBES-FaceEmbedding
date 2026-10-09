"""E1: con WBES_E1_GRADVEC=1, ``diffusion_net.geometry.build_grad`` = la versione vettorizzata di E9 in OGNI processo
Python del job, compresi i lavoratori ``spawn`` del pre-pass (aau/data_scale/prepass_ops.py), senza toccare
diffusion-net, v2_work o aau/data_scale. Lo mette e1_train_body.sh in testa a PYTHONPATH.

L'aggancio e' pigro: un finder sostituisce ``build_grad`` subito dopo l'esecuzione del modulo
``diffusion_net.geometry`` (``compute_operators`` la risolve fra i globali del modulo a ogni chiamata), quindi i
processi che non lo importano non pagano niente. Il sitecustomize del container (Ubuntu) viene eseguito comunque.
"""
import os
import sys

_UBUNTU = "/usr/lib/python3.10/sitecustomize.py"
if os.path.exists(_UBUNTU):
    try:
        with open(_UBUNTU) as _fh:
            exec(compile(_fh.read(), _UBUNTU, "exec"), {"__name__": "sitecustomize_container"})
    except Exception:
        pass

if os.environ.get("WBES_E1_GRADVEC") == "1":
    import importlib.abc
    import importlib.machinery

    class _GradVecFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "diffusion_net.geometry":
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return None
            orig = spec.loader.exec_module

            def exec_module(module, _orig=orig):
                _orig(module)
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from e1_grad_vec import build_grad_vec
                module.build_grad = build_grad_vec
                module.E1_GRADVEC = True
                print(f"[e1-gradvec] build_grad vettorizzato (pid {os.getpid()})", file=sys.stderr, flush=True)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _GradVecFinder())
