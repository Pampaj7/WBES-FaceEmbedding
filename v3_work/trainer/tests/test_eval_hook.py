#!/usr/bin/env python
"""L'eval di un checkpoint v3 attraverso eval_v3 (build_model/forward_model agganciati, vertici del loader
maxabs) deve dare lo STESSO embedding del training (campione ri-inquadrato al servizio, forward del modello),
su input puliti. Per ogni combinazione pooling x input_norm, su mesh BFM e ICT vere (viste con operatori).

    aau/run.sh v3_work/trainer/tests/test_eval_hook.py --out aau/runs/evidence/trainer_v3/eval_hook.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
sys.path.insert(0, str(TRAINER))

import common  # noqa: E402,F401
import data_v3 as dv  # noqa: E402
import eval_v3  # noqa: E402
from model_v3 import build_model_v3  # noqa: E402
import robustness.model_helpers as mh  # noqa: E402
from robustness.data_utils import sample_to_device  # noqa: E402

FILES = ["datasets/REMESH/npz_data_topo_500_withops_areanorm/id0001_GTready_original.npz",
         "datasets/REMESH/npz_data_topo_500_withops_areanorm/id0001_GTready_down8k.npz",
         "datasets/ICT/train_ready/npz_withops/id10000_GTready_crop.npz"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    for f in FILES:
        (tmp / Path(f).name).symlink_to((REPO / f).resolve())
    base = dv.GTReadyDatasetNPZ(str(tmp))
    rows, ok = [], True
    orig_build, orig_forward = mh.build_model, mh.forward_model
    combos = [(p, n, "mass") for p in ("meanmax", "area_meanmax", "area_attn") for n in ("maxabs", "sqrt_area")]
    combos += [("area_meanmax", "sqrt_area", "smooth"), ("area_meanmax", "sqrt_area", "winsor"),
               ("area_attn", "sqrt_area", "smooth")]
    for pooling, norm, aw in combos:
        if True:
            args = SimpleNamespace(model="xyz_dn", latent_dim=256, width=128, n_blocks=4, dropout=0.1,
                                   pool_mode="meanmax", pooling=pooling, input_norm=norm, attn_heads=1,
                                   area_weights=aw, area_smooth_k=64, winsor_pct="1,99")
            torch.manual_seed(0)
            model = build_model_v3(args, torch.device("cpu"))
            if pooling == "area_attn":   # punteggio non nullo, altrimenti l'attenzione e' la media pesata
                torch.nn.init.normal_(model.attn[-1].weight, std=0.5)
            model.eval()
            tr = dv.ServeTransform(norm, area_weights=aw)
            ref = []
            with torch.no_grad():
                for i in range(len(base)):
                    s = sample_to_device(tr(base.files[i], base[i]), torch.device("cpu"))
                    ref.append(orig_forward(model, s, s["verts"], False, False)[0])
            # percorso di eval: modello ricostruito dagli args, vertici del loader (maxabs), agganci di eval_v3
            mh.build_model, mh.forward_model = orig_build, orig_forward
            eval_v3.install(vars(args))
            m2 = mh.build_model(args, torch.device("cpu"))
            m2.load_state_dict(model.state_dict())
            m2.eval()
            with torch.no_grad():
                got = []
                for i in range(len(base)):
                    s = sample_to_device(base[i], torch.device("cpu"))
                    got.append(mh.forward_model(m2, s, s["verts"], False, False)[0])
            mh.build_model, mh.forward_model = orig_build, orig_forward
            d = max(float((x - y).abs().max()) for x, y in zip(ref, got))
            same_class = type(m2).__name__ == type(model).__name__
            row = {"pooling": pooling, "input_norm": norm, "area_weights": aw, "max_abs_diff": d,
                   "class": type(m2).__name__,
                   "pass": bool(d <= 1e-5 and same_class)}
            ok &= row["pass"]
            rows.append(row)
            print(row, flush=True)
    a.out.write_text(json.dumps({"rows": rows, "all_pass": bool(ok)}, indent=1))
    print("AGGANCIO DI EVAL:", "PASSATO" if ok else "FALLITO")


if __name__ == "__main__":
    main()
