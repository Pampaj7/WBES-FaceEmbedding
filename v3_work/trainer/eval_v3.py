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

Modalita' fattorizzata (factorized_v3.py, global_v3.py):
  * ``--head factorized``: il modello restituisce u (forma) agli script, cosi' le pipeline esistenti misurano la
    forma contro le loro GT; ``WBES_V3_FACTORIZED_OUT=full`` restituisce [s, u] (per tools/eval_factorized.py);
  * ``--input-norm global``: la scala in mm non e' nei vertici del loader, quindi il dataset congelato
    (GTReadyDatasetNPZ) aggiunge a ogni campione l'area in mm^2 e la rotazione del dominio, lette per NOME di file
    dalle tabelle di ``WBES_V3_SCALE_TABLES`` (percorsi separati da ':', tools/build_scale_table.py sulle viste di
    eval); sample_to_device le porta al forward, che ri-inquadra V_in come al training. Una mesh assente dalle
    tabelle e' un errore. Solo scenario clean, come sqrt_area.
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
    """Checkpoint dello script: --model_path (script zs/perturbated), --checkpoint o $WBES_CKPT (NoW)."""
    import os
    import torch
    if "--model_path" in script_args:
        raw = script_args[script_args.index("--model_path") + 1]
    elif "--checkpoint" in script_args:
        raw = script_args[script_args.index("--checkpoint") + 1]
    elif os.environ.get("WBES_CKPT"):
        raw = os.environ["WBES_CKPT"]
    else:
        raise SystemExit("eval_v3: checkpoint non trovato (--model_path, --checkpoint o WBES_CKPT)")
    path = Path(raw).expanduser().resolve()
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
    import os
    import robustness.model_helpers as mh
    import factorized_v3 as fz
    import global_v3
    from data_v3 import reframe_sqrt_area
    from model_v3 import build_model_v3

    pooling = str(ckpt_args.get("pooling", "meanmax"))
    norm = str(ckpt_args.get("input_norm", "maxabs"))
    head = str(ckpt_args.get("head", "embed"))
    out_mode = os.environ.get("WBES_V3_FACTORIZED_OUT", "u")
    orig_build, orig_forward = mh.build_model, mh.forward_model

    def build_model(args, device):
        if str(getattr(args, "pooling", pooling)) == "meanmax":
            return orig_build(args, device)
        a = SimpleNamespace(**{**vars(args), "pooling": getattr(args, "pooling", pooling),
                               "attn_heads": getattr(args, "attn_heads", ckpt_args.get("attn_heads", 1)),
                               "area_weights": getattr(args, "area_weights", ckpt_args.get("area_weights", "mass")),
                               "head": getattr(args, "head", head),
                               "size_hidden": getattr(args, "size_hidden", ckpt_args.get("size_hidden", 64))})
        m = build_model_v3(a, device)
        fz.set_output(m, out_mode)
        return m

    import area_v3
    aw = str(ckpt_args.get("area_weights", "mass"))
    area_v3.CFG.update(k=int(ckpt_args.get("area_smooth_k", 64)),
                       lo=float(str(ckpt_args.get("winsor_pct", "1,99")).split(",")[0]),
                       hi=float(str(ckpt_args.get("winsor_pct", "1,99")).split(",")[1]))

    unit = float(ckpt_args.get("global_unit_mm", 100.0))
    gops = str(ckpt_args.get("global_ops", "areanorm"))

    def forward_model(model, sample_dict, V_in, return_gate_info, add_noise):
        if norm == "sqrt_area" and aw == "mass":
            V_in = reframe_sqrt_area(V_in, sample_dict["mass"], sample_dict["faces"])
        elif norm == "sqrt_area":
            V_in = area_v3.area_frame(aw, V_in, sample_dict["faces"], sample_dict["mass"], sample_dict["evecs"])
        elif norm == "global":
            if "global_area_mm2" not in sample_dict:
                raise RuntimeError("eval_v3: campione senza scala globale (sample_to_device non agganciato?)")
            area = float(sample_dict["global_area_mm2"])
            c, f = global_v3.frame_params(V_in, sample_dict["faces"], sample_dict["mass"], sample_dict["evecs"], area, aw)
            V_in = global_v3.apply(V_in, c, f, sample_dict["global_R"].cpu().numpy(), unit)
            if gops == "mm":
                sample_dict = dict(sample_dict)
                sample_dict["mass"], sample_dict["evecs"] = global_v3.ops_to_mm(sample_dict["mass"], sample_dict["evecs"],
                                                                                area, unit)
        return orig_forward(model, sample_dict, V_in, return_gate_info, add_noise)

    mh.build_model = build_model
    _rebind("build_model", build_model)
    if norm == "global":
        install_global_samples(os.environ.get("WBES_V3_SCALE_TABLES", ""))
    if norm != "maxabs":
        mh.forward_model = forward_model
        _rebind("forward_model", forward_model)
    return {"pooling": pooling, "input_norm": norm, "area_weights": aw, "head": head,
            "out": out_mode if head == "factorized" else "z"}


GLOBAL_KEYS = ("global_area_mm2", "global_R")
ORIG = {}


def install_global_samples(tables: str) -> None:
    """Il loader congelato aggiunge area (mm^2) e rotazione del dominio per nome di file; sample_to_device le porta."""
    import torch
    import global_v3
    import robustness.data_utils as du
    from dataset_gtready import GTReadyDatasetNPZ
    if not tables:
        raise SystemExit("eval_v3: checkpoint con --input-norm global: serve WBES_V3_SCALE_TABLES (tabelle delle viste "
                         "di eval, tools/build_scale_table.py)")
    table = global_v3.ScaleTable(tables.split(":"))
    R = global_v3.rotations()
    ORIG.setdefault("getitem", GTReadyDatasetNPZ.__getitem__)
    ORIG.setdefault("to_device", du.sample_to_device)
    orig_get, orig_dev = ORIG["getitem"], ORIG["to_device"]

    def getitem(self, idx):
        out = orig_get(self, idx)
        area, dom = table.lookup(self.files[int(idx)])
        out["global_area_mm2"] = torch.tensor(area, dtype=torch.float64)
        out["global_R"] = torch.as_tensor(R[dom])
        return out

    def sample_to_device(sample, device):
        out = orig_dev(sample, device)
        for k in GLOBAL_KEYS:
            if k in sample:
                out[k] = sample[k]
        return out

    GTReadyDatasetNPZ.__getitem__ = getitem
    du.sample_to_device = sample_to_device
    _rebind("sample_to_device", sample_to_device)


def uninstall_global_samples() -> None:
    """Solo per i test: rimette loader e sample_to_device originali."""
    import robustness.data_utils as du
    from dataset_gtready import GTReadyDatasetNPZ
    if "getitem" in ORIG:
        GTReadyDatasetNPZ.__getitem__ = ORIG.pop("getitem")
    if "to_device" in ORIG:
        du.sample_to_device = ORIG.pop("to_device")
        _rebind("sample_to_device", du.sample_to_device)


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
          f"head={info['head']} uscita={info['out']} "
          f"trainer={cargs.get('trainer', 'v3' if 'pooling' in cargs else 'v1')}", flush=True)
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)] + script_args
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
