#!/usr/bin/env python
"""Test CPU della modalita' fattorizzata (global_v3.py, factorized_v3.py e i loro agganci).

  1. run dir: con i flag nuovi al default il nome (hash compreso) dei bracci C3F gia' addestrati non cambia
     (righe di launch.txt -> make_run_dir == directory esistente);
  2. frame globale su mesh vere (dati di prova, tabella di scala): area(X) * L0^2 = area_mm2, e X coincide (a meno
     della traslazione) con u_d (V_grezza) R_d^T / L0 calcolato dalla geometria grezza; X non dipende dal frame
     d'ingresso (V e 3.7 V + 1.3 danno lo stesso X);
  3. ``--global-ops mm`` contro ``areanorm``: stesso embedding (invarianza documentata in global_v3.py);
  4. eval: il percorso di eval_v3 (loader agganciato, sample_to_device, forward_model) da' lo stesso [s, u] / u del
     training, su input puliti;
  5. testa: uscita [s, u] e u, s iniziale = size_init, loss finita con gradiente su encoder e testa;
  6. augmentation di scala: estrazioni riproducibili per (seme, rank, passo), dentro [lo, hi];
  7. distanza form: (S_i - S_j)^2 + S_i S_j dP^2 = S_i^2 + S_j^2 - 2 S_i S_j cos(rho) = distanza F_centered
     di E12 (RMS pesata dei punti centrati), su configurazioni casuali.

    aau/run.sh v3_work/trainer/tests/test_factorized.py --out aau/runs/evidence/trainer_v3/factorized/units.json
"""
from __future__ import annotations

import argparse
import copy
import json
import shlex
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

import common  # noqa: E402,F401
import data_v3 as dv  # noqa: E402
import factorized_v3 as fz  # noqa: E402
import global_v3  # noqa: E402
import train_v3 as T3  # noqa: E402
from losses_v3 import StepBatch  # noqa: E402
from model_v3 import build_model_v3  # noqa: E402

TD = REPO / "aau/runs/evidence/trainer_v3/testdata"
RUNS = REPO / "aau/runs/evidence/trainer_v3/ablations/c3f_runs"
FILES = {"datasets/REMESH/npz_data_topo_500_withops_areanorm": ["id0001_GTready_original.npz",
                                                                 "id0001_GTready_crop.npz", "id0002_GTready_noisy.npz"],
         "datasets/ICT/train_ready/npz_withops": ["id10000_GTready_down8k.npz", "id10001_GTready_original.npz"]}
RAW = {"id0001_GTready_original.npz": "datasets/REMESH/npz_data_topo_500/id0001_GTready_original.npz",
       "id10001_GTready_original.npz": "datasets/ICT/topo/ict0001_GTready_original.npz"}


def _args(**kw):
    base = dict(model="xyz_dn", latent_dim=256, width=128, n_blocks=4, dropout=0.1, pool_mode="meanmax",
                pooling="area_meanmax", attn_heads=1, area_weights="smooth", area_smooth_k=64, winsor_pct="1,99",
                head="factorized", size_hidden=64, size_init=4.1, input_norm="global", global_unit_mm=100.0,
                global_ops="areanorm", lambda_size=1.0, loss="v2", lambda_rank=0.5, rank_pairs=1024, rank_margin=0.05,
                rank_tau=0.02, rank_hard_frac=0.7, train_pair_mode="cross_topology", use_id_loss=True, lambda_id=0.25,
                lambda_subject=1.0, lambda_mesh=1.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _view_dataset():
    tmp = Path(tempfile.mkdtemp())
    for d, names in FILES.items():
        for n in names:
            (tmp / n).symlink_to((REPO / d / n).resolve())
    return dv.GTReadyDatasetNPZ(str(tmp))


def t_run_dir() -> dict:
    rows, ok = {}, True
    for arm in ("ctrl", "area", "arearobust", "bal", "ugtmix", "loginv"):
        argv = shlex.split((RUNS / arm / "launch.txt").read_text())[1:]
        a = T3.build_parser().parse_args(argv)
        T3.check_args(a)
        got = T3.make_run_dir(a).name
        want = [p.name for p in (RUNS / arm).glob("v3_*")]
        rows[arm] = {"got": got, "existing": want, "same": want == [got]}
        ok &= want == [got]
    return {"pass": bool(ok), "arms": rows}


def t_global_frame(table: global_v3.ScaleTable) -> dict:
    ds = _view_dataset()
    out, ok = {}, True
    for weights in ("mass", "smooth"):
        gf = global_v3.GlobalFrame(table, 100.0, weights)
        for i, name in enumerate(ds.files):
            s = ds[i]
            X = gf(name, dict(s))["verts"].double()
            area_mm2, dom = table.lookup(name)
            a_x = float(dv._total_area(X, s["faces"])) * 100.0 ** 2
            r = {"area_rel_err": abs(a_x / area_mm2 - 1.0)}
            s2 = dict(s)
            s2["verts"] = s["verts"] * 3.7 + 1.3
            X2 = global_v3.GlobalFrame(table, 100.0, weights)(name, s2)["verts"].double()
            r["frame_invariance_max_abs"] = float((X - X2).abs().max())
            if name in RAW and weights == "smooth":
                with np.load(REPO / RAW[name]) as z:
                    V = np.asarray(z["V"], np.float64)
                tf = json.loads(global_v3.FRAMES.read_text())["domains"][dom]
                Xr = float(tf["u"]) * V @ np.asarray(tf["R"]).T / 100.0
                Xr = Xr - Xr.mean(0)
                Xn = X.numpy() - X.numpy().mean(0)
                r["vs_raw_canonical_max_abs_mm"] = float(np.abs(Xn - Xr).max() * 100.0)
                ok &= r["vs_raw_canonical_max_abs_mm"] < 0.01
            ok &= r["area_rel_err"] < 1e-5 and r["frame_invariance_max_abs"] < 1e-5
            out[f"{weights}|{name}"] = r
    return {"pass": bool(ok), "rows": out}


def t_ops_mm(table) -> dict:
    ds = _view_dataset()
    torch.manual_seed(0)
    m = build_model_v3(_args(), torch.device("cpu")).eval()
    from robustness.data_utils import sample_to_device
    from robustness.model_helpers import forward_model
    worst, scale = 0.0, 0.0
    for i, name in enumerate(ds.files):
        z = []
        for ops in ("areanorm", "mm"):
            s = sample_to_device(global_v3.GlobalFrame(table, 100.0, "smooth", ops)(name, dict(ds[i])), torch.device("cpu"))
            with torch.no_grad():
                z.append(forward_model(m, s, s["verts"], False, False)[0])
        worst = max(worst, float((z[0] - z[1]).abs().max()))
        scale = max(scale, float(z[0].abs().max()))
    return {"pass": worst / scale < 1e-4, "max_abs_diff": worst, "z_scale": scale}


def t_eval_hook(table_path: Path) -> dict:
    import eval_v3
    import robustness.model_helpers as mh
    from robustness.data_utils import sample_to_device
    ds = _view_dataset()
    rows, ok = [], True
    orig_build, orig_forward = mh.build_model, mh.forward_model
    for out_mode in ("u", "full"):
        args = _args()
        torch.manual_seed(0)
        model = build_model_v3(args, torch.device("cpu"))
        torch.nn.init.normal_(model.size_head[-1].weight, std=0.1)
        model.eval()
        fz.set_output(model, out_mode)
        tr = dv.ServeTransform("global", area_weights="smooth",
                               global_frame=global_v3.GlobalFrame(global_v3.ScaleTable([table_path]), 100.0, "smooth"))
        with torch.no_grad():
            ref = [orig_forward(model, s, s["verts"], False, False)[0]
                   for s in (sample_to_device(tr(ds.files[i], ds[i]), torch.device("cpu")) for i in range(len(ds)))]
        import os
        os.environ["WBES_V3_SCALE_TABLES"] = str(table_path)
        os.environ["WBES_V3_FACTORIZED_OUT"] = out_mode
        eval_v3.install(vars(args))
        try:
            import robustness.data_utils as du
            m2 = mh.build_model(args, torch.device("cpu"))
            m2.load_state_dict(model.state_dict())
            m2.eval()
            with torch.no_grad():
                got = [mh.forward_model(m2, s, s["verts"], False, False)[0]
                       for s in (du.sample_to_device(ds[i], torch.device("cpu")) for i in range(len(ds)))]
        finally:
            eval_v3.uninstall_global_samples()
            mh.build_model, mh.forward_model = orig_build, orig_forward
        d = max(float((x - y).abs().max()) for x, y in zip(ref, got))
        dim = int(got[0].shape[-1])
        row = {"out": out_mode, "max_abs_diff": d, "dim": dim,
               "pass": bool(d <= 1e-5 and dim == (257 if out_mode == "full" else 256))}
        ok &= row["pass"]
        rows.append(row)
    return {"pass": bool(ok), "rows": rows}


def t_head_loss() -> dict:
    torch.manual_seed(0)
    args = _args()
    m = build_model_v3(args, torch.device("cpu"))
    ds = _view_dataset()
    from robustness.data_utils import sample_to_device
    from robustness.model_helpers import forward_model
    table = global_v3.ScaleTable([TD / "scale_table.npz"])
    gf = global_v3.GlobalFrame(table, 100.0, "smooth")
    zs = []
    for i in range(len(ds)):
        s = sample_to_device(gf(ds.files[i], dict(ds[i])), torch.device("cpu"))
        zs.append(forward_model(m, s, s["verts"], False, True)[0].squeeze(0))
    Z = torch.stack(zs)
    s0 = float(Z[:, 0].detach().abs().sub(args.size_init).abs().max())
    subj = [n.split("_")[0] for n in ds.files]
    topo = [n[:-4].split("_GTready_")[1] for n in ds.files]
    ids = sorted(set(subj))
    D = np.array([[0.0, 0.3, 0.4, 0.5], [0.3, 0.0, 0.2, 0.35], [0.4, 0.2, 0.0, 0.25], [0.5, 0.35, 0.25, 0.0]],
                 dtype=np.float32)
    n2i = {s_: i for i, s_ in enumerate(ids)}
    log_cs = {"id0001": 4.10, "id0002": 4.15, "id10000": 4.30, "id10001": 4.25}
    log_a = fz.draw_log_scales("0.8,1.25", len(subj), 1234, 0, 7)
    loss, terms = fz.factorized_loss(args, StepBatch(Z, subj, topo, ids, D.view(dv.NanGuardedMatrix), n2i), log_cs, log_a)
    loss.backward()
    g_head = float(sum(p.grad.abs().sum() for p in m.size_head.parameters()))
    g_enc = float(sum(p.grad.abs().sum() for p in m.encoder.parameters() if p.grad is not None))
    fz.set_output(m, "u")
    with torch.no_grad():
        s = sample_to_device(gf(ds.files[0], dict(ds[0])), torch.device("cpu"))
        du = forward_model(m, s, s["verts"], False, False)[0].shape[-1]
    ok = torch.isfinite(loss) and g_head > 0 and g_enc > 0 and s0 < 1e-5 and Z.shape[1] == 257 and du == 256 \
        and "size_mse" in terms
    return {"pass": bool(ok), "loss": float(loss), "terms": terms, "s_init_max_abs_err": s0, "grad_head": g_head,
            "grad_encoder": g_enc, "dims": [int(Z.shape[1]), int(du)]}


def t_scale_draws() -> dict:
    a = fz.draw_log_scales("0.8,1.25", 1000, 1234, 0, 55)
    b = fz.draw_log_scales("0.8,1.25", 1000, 1234, 0, 55)
    c = fz.draw_log_scales("0.8,1.25", 1000, 1234, 1, 55)
    ok = np.array_equal(a, b) and not np.array_equal(a, c) and a.min() >= np.log(0.8) and a.max() <= np.log(1.25) \
        and fz.draw_log_scales("", 3, 1, 0, 1) is None
    return {"pass": bool(ok), "range": [float(np.exp(a.min())), float(np.exp(a.max()))], "mean_log": float(a.mean())}


def t_form_identity() -> dict:
    rng = np.random.default_rng(0)
    n_pts, n = 500, 40
    w = rng.uniform(0.5, 1.5, n_pts)
    W = w / w.sum()
    base = rng.normal(size=(n_pts, 3)) * 50.0
    C = base[None] * rng.uniform(0.85, 1.15, (n, 1, 1)) + rng.normal(size=(n, n_pts, 3)) * 3.0 + rng.normal(size=(n, 1, 3)) * 5
    m = np.einsum("v,nvd->nd", W, C)
    Cc = C - m[:, None]
    S = np.sqrt(np.einsum("v,nv->n", W, (Cc ** 2).sum(-1)))
    Zp = Cc / S[:, None, None]
    i, j = np.triu_indices(n, 1)
    d_cc = np.sqrt(np.einsum("v,pv->p", W, ((Cc[i] - Cc[j]) ** 2).sum(-1)))
    dP = np.sqrt(np.einsum("v,pv->p", W, ((Zp[i] - Zp[j]) ** 2).sum(-1)))
    rho = 2 * np.arcsin(dP / 2)
    dF = fz.form_distance(S[i], S[j], dP)
    alt = np.sqrt(S[i] ** 2 + S[j] ** 2 - 2 * S[i] * S[j] * np.cos(rho))
    # pair_distances su z = (log S, u = dP / dp_per_unit con u ricostruito: qui u = Zp appiattito pesato)
    U = (np.sqrt(W)[None, :, None] * Zp).reshape(n, -1) * 4.0
    Z = np.concatenate([np.log(S)[:, None], U], axis=1)
    pdist = fz.pair_distances(Z, i, j, dp_per_unit=0.25)
    e1, e2, e3 = float(np.abs(dF - d_cc).max()), float(np.abs(dF - alt).max()), float(np.abs(pdist["form_mm"] - d_cc).max())
    return {"pass": e1 < 1e-9 and e2 < 1e-6 and e3 < 1e-9, "form_vs_centered_max_abs": e1, "form_vs_cos_max_abs": e2,
            "pair_distances_vs_centered_max_abs": e3, "median_d_cc": float(np.median(d_cc))}


def t_factorized2_exact() -> dict:
    """--head factorized2, senza training (pesi casuali, testa della dimensione non nulla): s(aX + t) - s(X) = log a e
    u(aX + t) = u(X), a in {0.8, 1, 1.25}, su mesh vere nell'ingresso globale; float32, criterio 1e-5."""
    from robustness.data_utils import sample_to_device
    from robustness.model_helpers import forward_model
    torch.manual_seed(0)
    m = build_model_v3(_args(head="factorized2", dropout=0.0), torch.device("cpu"))
    torch.nn.init.normal_(m.size_head[-1].weight, std=0.5)
    m.eval()
    ds = _view_dataset()
    gf = global_v3.GlobalFrame(global_v3.ScaleTable([TD / "scale_table.npz"]), 100.0, "smooth")
    worst_s, worst_u, rows = 0.0, 0.0, {}
    for i, name in enumerate(ds.files):
        s = sample_to_device(gf(name, dict(ds[i])), torch.device("cpu"))
        with torch.no_grad():
            z0 = forward_model(m, s, s["verts"], False, False)[0][0]
            for a in (0.8, 1.0, 1.25):
                z = forward_model(m, s, s["verts"] * a + torch.tensor([0.3, -0.2, 0.1]), False, False)[0][0]
                es = abs(float(z[0] - z0[0]) - float(np.log(a)))
                eu = float((z[1:] - z0[1:]).abs().max())
                worst_s, worst_u = max(worst_s, es), max(worst_u, eu)
                rows[f"{name}|{a}"] = {"e_s": es, "e_u_max": eu, "s": float(z[0])}
    return {"pass": worst_s < 1e-5 and worst_u < 1e-5, "max_e_s": worst_s, "max_e_u": worst_u,
            "u_scale": float(z0[1:].abs().max()), "rows": rows}


def t_dual() -> dict:
    """--head dual: uscite [z_F, u] (512), z_F e u (256); encoder e pool_proj identici a EncoderV3 con lo stesso seme
    (pool_proj_u nasce in fork_rng); loss doppia finita, gradiente su pool_proj e pool_proj_u."""
    from robustness.data_utils import sample_to_device
    from robustness.model_helpers import forward_model
    torch.manual_seed(0)
    m = build_model_v3(_args(head="dual", dropout=0.0), torch.device("cpu"))
    torch.manual_seed(0)
    ref = build_model_v3(_args(head="embed", dropout=0.0), torch.device("cpu"))
    same = all(torch.equal(a, ref.state_dict()[k]) for k, a in m.state_dict().items() if k in ref.state_dict())
    ds = _view_dataset()
    gf = global_v3.GlobalFrame(global_v3.ScaleTable([TD / "scale_table.npz"]), 100.0, "smooth")
    dims = {}
    for mode in ("full", "zf", "u"):
        fz.set_output(m, mode)
        s = sample_to_device(gf(ds.files[0], dict(ds[0])), torch.device("cpu"))
        dims[mode] = int(forward_model(m, s, s["verts"], False, False)[0].shape[-1])
    fz.set_output(m, "full")
    Z = torch.stack([forward_model(m, s, s["verts"], False, False)[0][0] for s in
                     (sample_to_device(gf(n, dict(ds[i])), torch.device("cpu")) for i, n in enumerate(ds.files))])
    subj = [n.split("_")[0] for n in ds.files]
    topo = [n[:-4].split("_GTready_")[1] for n in ds.files]
    ids = sorted(set(subj))
    D1 = np.array([[0, .3, .4, .5], [.3, 0, .2, .35], [.4, .2, 0, .25], [.5, .35, .25, 0]], dtype=np.float32)
    D2 = (D1[::-1, ::-1] * 0.5).copy()
    n2i = {s_: i for i, s_ in enumerate(ids)}
    args = _args(head="dual", lambda_form=1.0, lambda_shape=0.7)
    loss, terms = fz.dual_loss(args, StepBatch(Z, subj, topo, ids, D1.view(dv.NanGuardedMatrix), n2i),
                               D2.view(dv.NanGuardedMatrix), n2i)
    loss.backward()
    g1 = float(m.pool_proj.weight.grad.abs().sum())
    g2 = float(m.pool_proj_u.weight.grad.abs().sum())
    ok = same and dims == {"full": 512, "zf": 256, "u": 256} and bool(torch.isfinite(loss)) and g1 > 0 and g2 > 0
    return {"pass": bool(ok), "encoder_identical_to_embed": same, "dims": dims, "loss": float(loss),
            "terms": terms, "grad_pool_proj": g1, "grad_pool_proj_u": g2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    table_path = TD / "scale_table.npz"
    table = global_v3.ScaleTable([table_path])
    res = {}
    for name, fn in (("run_dir_defaults", t_run_dir), ("global_frame", lambda: t_global_frame(table)),
                     ("global_ops_mm", lambda: t_ops_mm(table)), ("eval_hook", lambda: t_eval_hook(table_path)),
                     ("head_loss", t_head_loss), ("scale_draws", t_scale_draws), ("form_identity", t_form_identity), ("factorized2_exact", t_factorized2_exact), ("dual", t_dual)):
        try:
            res[name] = fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            res[name] = {"pass": False, "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()}
        print(name, json.dumps(res[name], default=str)[:900], flush=True)
    res["all_pass"] = all(r.get("pass") for r in res.values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str))
    print("TEST FATTORIZZATA:", "PASSATI" if res["all_pass"] else "FALLITI")


if __name__ == "__main__":
    main()
