#!/usr/bin/env python
"""Lancia uno script di eval del repo su un checkpoint v3 (come aau/models/eval_ablation.py per le ablazioni).

    aau/run.sh v3_work/trainer/eval_v3.py -- <script.py> <argomenti dello script, con --model_path>
    LAUNCH=(v3_work/trainer/eval_v3.py --)    (prefisso degli script di aau/zs3dmm, vedi slurm/eval_zs_v3.sbatch)

Legge gli ``args`` del checkpoint indicato da ``--model_path`` (file .pth, o run dir con
``--checkpoint_selector``) e installa due agganci prima di eseguire lo script:
  * build_model: ``--pooling`` diverso da meanmax -> EncoderV3 (model_v3.build_model_v3); altrimenti la build
    v1, invariata. Il forward congelato (forward_model) usa EncoderV3 senza modifiche (e' una sottoclasse);
  * forward_model con ``--input-norm sqrt_area``: ri-inquadra V_in (centroide pesato per massa, sqrt(area))
    prima del modello. Gli script danno al modello i vertici del loader (maxabs); il ri-inquadramento non
    dipende dal frame d'ingresso, quindi sugli input PULITI e' identico al training. Su input perturbati
    annullerebbe traslazione e scala della perturbazione: gli scenari diversi da clean sono rifiutati.
Con un checkpoint meanmax + maxabs non cambia nulla (stesso percorso v1).
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import common  # noqa: E402,F401


def _ckpt_args(script_args: list[str]) -> tuple[Path, dict]:
    import torch
    if "--model_path" not in script_args:
        raise SystemExit("eval_v3: lo script non ha --model_path")
    path = Path(script_args[script_args.index("--model_path") + 1]).expanduser().resolve()
    if path.is_dir():
        sel = script_args[script_args.index("--checkpoint_selector") + 1] if "--checkpoint_selector" in script_args \
            else "best_by_auc"
        cand = path / "checkpoints" / f"{sel}.pth"
        path = cand if cand.exists() else path
    pack = torch.load(path, map_location="cpu", weights_only=False)
    return path, dict(pack.get("args", {}))


def _rebind(name: str, obj) -> None:
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if mod_name.startswith(("robustness", "compare_model_vs", "eval_", "zs_")) or mod_name == "__main__":
            if hasattr(mod, name):
                setattr(mod, name, obj)


def install(ckpt_args: dict) -> dict:
    import robustness.model_helpers as mh
    from data_v3 import reframe_sqrt_area
    from model_v3 import build_model_v3

    pooling = str(ckpt_args.get("pooling", "meanmax"))
    norm = str(ckpt_args.get("input_norm", "maxabs"))
    orig_build, orig_forward = mh.build_model, mh.forward_model

    def build_model(args, device):
        if str(getattr(args, "pooling", pooling)) == "meanmax":
            return orig_build(args, device)
        a = SimpleNamespace(**{**vars(args), "pooling": getattr(args, "pooling", pooling),
                               "attn_heads": getattr(args, "attn_heads", ckpt_args.get("attn_heads", 1)),
                               "area_weights": getattr(args, "area_weights", ckpt_args.get("area_weights", "mass"))})
        return build_model_v3(a, device)

    import area_v3
    aw = str(ckpt_args.get("area_weights", "mass"))
    area_v3.CFG.update(k=int(ckpt_args.get("area_smooth_k", 64)),
                       lo=float(str(ckpt_args.get("winsor_pct", "1,99")).split(",")[0]),
                       hi=float(str(ckpt_args.get("winsor_pct", "1,99")).split(",")[1]))

    def forward_model(model, sample_dict, V_in, return_gate_info, add_noise):
        if norm == "sqrt_area" and aw == "mass":
            V_in = reframe_sqrt_area(V_in, sample_dict["mass"], sample_dict["faces"])
        elif norm == "sqrt_area":
            V_in = area_v3.area_frame(aw, V_in, sample_dict["faces"], sample_dict["mass"], sample_dict["evecs"])
        return orig_forward(model, sample_dict, V_in, return_gate_info, add_noise)

    mh.build_model = build_model
    _rebind("build_model", build_model)
    if norm != "maxabs":
        mh.forward_model = forward_model
        _rebind("forward_model", forward_model)
    return {"pooling": pooling, "input_norm": norm, "area_weights": aw}


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("uso: eval_v3.py -- script.py [args]")
    cut = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--allow-perturbed", action="store_true", help="solo per prove: non rifiuta scenari perturbati")
    known = p.parse_args(argv[:cut])
    script, script_args = Path(argv[cut + 1]).resolve(), argv[cut + 2:]
    path, cargs = _ckpt_args(script_args)
    if cargs.get("input_norm", "maxabs") != "maxabs" and not known.allow_perturbed \
            and script.name == "compare_model_vs_chamfer_rankings.py":
        sc = script_args[script_args.index("--scenarios") + 1] if "--scenarios" in script_args else None
        if sc != "clean":
            raise SystemExit(f"eval_v3: {path.name} ha input_norm {cargs['input_norm']}: solo --scenarios clean "
                             f"(ora: {sc}); con zs_zeroshot: WBES_EVAL_SCENARIOS=clean")
    import robustness.model_helpers  # noqa: F401
    import robustness.data_utils  # noqa: F401
    info = install(cargs)
    print(f"[eval_v3] {path}: pooling={info['pooling']} input_norm={info['input_norm']} area={info['area_weights']} "
          f"trainer={cargs.get('trainer', 'v3' if 'pooling' in cargs else 'v1')}", flush=True)
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)] + script_args
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
