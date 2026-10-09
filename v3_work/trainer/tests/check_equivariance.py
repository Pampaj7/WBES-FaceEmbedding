#!/usr/bin/env python
"""Equivarianza della testa fattorizzata dopo il training: s(a X) = s(X) + log a, u(a X) = u(X)?

Su mesh di soggetti NON di training (viste con operatori su disco: BFM dell'eval online, ICT-5000 held-out), per
a in {0.8, 0.9, 1.1, 1.25}: X = ingresso globale della mesh pulita (global_v3, come al servizio), forward in eval
senza rumore su X e su a X. Riporta:
  * e_s = s(a X) - s(X) - log a (media, mediana |.|, max |.|) per a, in unita' di log (0.01 = 1% di taglia);
  * r_u = ||u(a X) - u(X)|| / mediana delle ||u_i - u_j|| fra original di soggetti diversi (0 = invariante);
  * accuratezza di s su X: s - log S vero (dove il soggetto e' nella --size-table), e Spearman fra s e log S.
La scala delle mesh si legge dalla geometria grezza (tools/build_scale_table.py, stessa verifica).

    aau/run.sh v3_work/trainer/tests/check_equivariance.py --ckpt <epochNNN_ema.pth> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
sys.path.insert(0, str(TRAINER))
sys.path.insert(0, str(TRAINER / "tools"))

import common  # noqa: E402,F401
import area_v3  # noqa: E402
import data_v3 as dv  # noqa: E402
import factorized_v3 as fz  # noqa: E402
import global_v3  # noqa: E402
from model_v3 import build_model_v3  # noqa: E402

SCALES = (0.8, 0.9, 1.1, 1.25)
VIEWS = {"bfm": REPO / "datasets/REMESH/npz_data_topo_500_withops_areanorm",
         "ict": REPO / "datasets/ICT/train_ready/npz_withops"}
TOPOS = ("original", "remesh", "down8k", "up60k", "noisy", "crop")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--subjects", default="", help="id separati da virgola; vuoto = online_eval BFM dello split del "
                                                   "checkpoint + id14500-id14515 (ICT-5000 held-out)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from build_scale_table import _work
    pack = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**pack["args"])
    if getattr(args, "head", "embed") not in ("factorized", "factorized2") or args.input_norm != "global":
        raise SystemExit("checkpoint non fattorizzato o senza ingresso globale")
    area_v3.CFG.update(k=int(args.area_smooth_k))
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = build_model_v3(args, device)
    model.load_state_dict(pack["state_dict"])
    model.eval()
    fz.set_output(model, "full")
    split = json.loads(Path(args.split_json).read_text())
    subj = a.subjects.split(",") if a.subjects else \
        [s for s in split["online_eval"] if int(s[2:]) < 1000] + [f"id{14500 + k}" for k in range(16)]
    train = set(split["train"])
    if set(subj) & train:
        raise SystemExit(f"soggetti di training fra quelli del controllo: {sorted(set(subj) & train)[:3]}")
    tmp = Path(tempfile.mkdtemp())
    items = []
    for s in subj:
        view = VIEWS["bfm" if int(s[2:]) < 1000 else "ict"]
        for t in TOPOS:
            n = f"{s}_GTready_{t}.npz"
            if (view / n).exists():
                (tmp / n).symlink_to((view / n).resolve())
                items.append((n, "view", str(view / n)))
    rows = _work((items, None))
    tf = json.loads(global_v3.FRAMES.read_text())["domains"]
    names = [r[0] for r in rows]
    dom = {n: ("bfm" if int(n[2:].split("_")[0]) < 1000 else "ict") for n in names}
    tab = Path(tempfile.mkdtemp()) / "scale_table.npz"     # fuori dalla cartella delle mesh (il loader la leggerebbe)
    np.savez(tab, names=np.asarray(names), domain=np.asarray([dom[n] for n in names]),
             area_mm2=np.asarray([float(tf[dom[n]]["u"]) ** 2 * r[1] for n, r in zip(names, rows)]))
    gf = global_v3.GlobalFrame(global_v3.ScaleTable([tab]), args.global_unit_mm, args.area_weights,
                               getattr(args, "global_ops", "areanorm"))
    ds = dv.GTReadyDatasetNPZ(str(tmp))
    from robustness.data_utils import sample_to_device
    from robustness.model_helpers import forward_model
    Z = {1.0: []}
    Z.update({sc: [] for sc in SCALES})
    with torch.no_grad():
        for i, n in enumerate(ds.files):
            s = sample_to_device(gf(n, dict(ds[i])), device)
            for sc in Z:
                Z[sc].append(forward_model(model, s, s["verts"] * sc, False, False)[0].squeeze(0).double().cpu().numpy())
    Z = {sc: np.stack(v) for sc, v in Z.items()}
    s0, u0 = fz.split(Z[1.0])
    sids = [n.split("_GTready_")[0] for n in ds.files]
    topo = [n[:-4].split("_GTready_")[1] for n in ds.files]
    orig = [k for k, t in enumerate(topo) if t == "original"]
    du_ref = np.median([np.linalg.norm(u0[i] - u0[j]) for x, i in enumerate(orig) for j in orig[x + 1:]])
    out = {"ckpt": str(a.ckpt), "n_meshes": len(ds.files), "n_subjects": len(set(sids)),
           "median_u_between_subjects": float(du_ref), "by_scale": {}}
    for sc in SCALES:
        s1, u1 = fz.split(Z[sc])
        e = s1 - s0 - np.log(sc)
        ru = np.linalg.norm(u1 - u0, axis=1) / du_ref
        out["by_scale"][str(sc)] = {"e_s_mean": float(e.mean()), "e_s_median_abs": float(np.median(np.abs(e))),
                                    "e_s_max_abs": float(np.abs(e).max()), "r_u_median": float(np.median(ru)),
                                    "r_u_max": float(ru.max()),
                                    "slope_ds_dloga": float(np.mean((s1 - s0) / np.log(sc)))}
    size_table = getattr(args, "size_table", "")
    if size_table:
        lcs = fz.load_log_cs(size_table)
        have = [k for k, sid in enumerate(sids) if sid in lcs]
        if have:
            from scipy.stats import spearmanr
            err = np.asarray([s0[k] - lcs[sids[k]] for k in have])
            sub = sorted({sids[k] for k in have})
            s_mean = [np.mean([s0[k] for k in have if sids[k] == x]) for x in sub]
            out["size_accuracy"] = {"n_meshes": len(have), "n_subjects": len(sub), "err_mean": float(err.mean()),
                                    "err_median_abs": float(np.median(np.abs(err))), "err_max_abs": float(np.abs(err).max()),
                                    "spearman_subject_mean_s_vs_log_cs": float(spearmanr(s_mean, [lcs[x] for x in sub]).correlation)
                                    if len(sub) > 2 else None,
                                    "by_topology_median_abs": {t: float(np.median([abs(err[q]) for q, k in enumerate(have)
                                                                                   if topo[k] == t]))
                                                               for t in TOPOS if any(topo[k] == t for k in have)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
