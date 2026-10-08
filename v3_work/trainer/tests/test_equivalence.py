#!/usr/bin/env python
"""Test di equivalenza, livello 1: loss e gradienti di v3 (flag di default) contro v2 sullo STESSO batch.

v2 e' la funzione vera: robustness.train_runner._train_epoch_mixed, chiamata come la chiama la catena del run
su scala, cioe' dentro train_v2._patched_v1 (batch a dominio singolo, GT con guardia NaN), con i gruppi di
etichette di train_steps.install_label_groups e la GT letta da train_steps.install_gt_keep_scale. Le
riceve UN batch di soggetti (train_subjects = 5 soggetti), quindi fa esattamente un passo. v3 e'
train_v3.train_epoch sul piano di sampler_v3.plan_epoch_v2.

Per ogni caso (dominio, epoca, modello) a parita' di seme torch:
  * v2 due volte (pavimento di non determinismo della GPU), v3 una;
  * confronto di loss, termini, gradienti PRIMA del clip e dopo, e della sequenza di mesh lette dal dataset.
Criterio: |delta loss| <= 1e-5 e max |delta grad| <= 1e-5 (assoluti; si riporta anche il relativo).

Poi il forward packed contro il sequenziale (stesso v3) in modo deterministico (dropout 0, niente rumore
latente, niente perturbazioni), in fp32 puro e con TF32 (il default del training).

    aau/run.sh v3_work/trainer/tests/test_equivalence.py --testdata aau/runs/evidence/trainer_v3/testdata \
        --stage-root /tmp/$SLURM_JOB_ID/eq --out aau/runs/evidence/trainer_v3/equivalence_l1.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

if "--fp32" in sys.argv:
    # il container NGC esporta TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, che tiene il TF32 acceso qualunque sia
    # torch.backends.cuda.matmul.allow_tf32: per un confronto in fp32 vero va spento PRIMA di torch
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import numpy as np
import torch

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
sys.path.insert(0, str(TRAINER))
sys.path.insert(0, str(REPO / "v2_work/fastio"))
sys.path.insert(0, str(REPO / "v2_work/train_v2"))

import train_v3 as T3  # noqa: E402
import sampler_v3 as sv  # noqa: E402
from model_v3 import StepEmbedder, build_model_v3  # noqa: E402

import robustness.train_runner as tr  # noqa: E402
from intrinsic_utils import seed_everything  # noqa: E402
from robustness.noise import PerturbationParams, parse_noise_mode_weights, parse_noise_modes  # noqa: E402

TOL = 1e-5
RECIPE = ("--device cuda --model xyz_dn --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1 --pool_mode meanmax "
          "--k_spec 0 --no-log_spec --eig_k 300 --batch_subjects 5 --lr 1e-4 --weight_decay 1e-6 --grad_clip 1.0 "
          "--train_level mixed --train_pair_mode cross_topology --lambda_subject 1.0 --lambda_mesh 1.0 "
          "--lambda_rank 0.5 --use_id_loss --lambda_id 0.25 --rank_margin 0.05 --rank_pairs 1024 --rank_tau 0.02 "
          "--rank_hard_frac 0.7 --max_meshes_per_subject_train 6 --max_meshes_per_subject_eval 6 "
          "--max_subjects_eval_train 16 --preload_eval_workers_train 2 --p_noise 0.6 --sigma_min 5e-4 "
          "--sigma_max 2e-2 --noise_modes translation,rotation,jitter "
          "--noise_mode_weights translation=4,rotation=2,jitter=1 --rigid_rot_deg 12.0 --rigid_rot_deg_min 0.5 "
          "--rigid_trans_scale 0.03 --rigid_trans_scale_min 0.001 --sigma_min_eval 1e-3 --sigma_max_eval 0.1 "
          "--n_sigma_eval 6 --eval_mode average").split()


class Probe(torch.optim.SGD):
    """Ottimizzatore che non muove i pesi: registra i gradienti (dopo il clip) a ogni step."""

    def __init__(self, params):
        super().__init__(params, lr=0.0)
        self.records = []

    def step(self, closure=None):
        self.records.append([p.grad.detach().clone() if p.grad is not None else None
                             for g in self.param_groups for p in g["params"]])


class Recorder:
    """Proxy del dataset che registra gli indici letti, in ordine."""

    def __init__(self, ds):
        self._ds, self.files, self.seen = ds, ds.files, []

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        self.seen.append(int(i))
        return self._ds[i]


PRECLIP: list = []
_orig_clip = torch.nn.utils.clip_grad_norm_


def _clip_spy(params, max_norm, *a, **kw):
    params = list(params)
    PRECLIP.append([p.grad.detach().clone() if p.grad is not None else None for p in params])
    return _orig_clip(params, max_norm, *a, **kw)


def seed_all(s: int) -> None:
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def grad_diff(ga, gb) -> tuple[float, float]:
    mx, scale = 0.0, 0.0
    for a, b in zip(ga, gb):
        if a is None or b is None:
            assert a is None and b is None
            continue
        mx = max(mx, float((a - b).abs().max()))
        scale = max(scale, float(a.abs().max()))
    return mx, mx / max(scale, 1e-30)


def run_v2(model, dataset, subjects, epoch, ctx, seed):
    import train_v2
    probe = Probe(model.parameters())
    PRECLIP.clear()
    rec = Recorder(dataset)
    seed_all(seed)
    with train_v2._patched_v1(5, "bfm"):
        stats = tr._train_epoch_mixed(model=model, dataset=rec, subj_map=ctx["subj_map"],
                                      subject_topology_map=ctx["topo_v2"], train_subjects=list(subjects),
                                      name_to_idx=ctx["n2i_v2"], gt_matrix=ctx["gt_v2"], device=ctx["device"],
                                      optimizer=probe, teacher_model=None, epoch=epoch, args=ctx["args1"],
                                      noise_modes=ctx["modes"], noise_mode_probs=ctx["probs"],
                                      perturbation=ctx["pert"])
    return stats, PRECLIP[0], probe.records[0], rec.seen


def run_v3(model, data, subjects, epoch, ctx, seed, forward="sequential"):
    probe = Probe(model.parameters())
    PRECLIP.clear()
    rec = Recorder(data.dataset)
    emb = StepEmbedder(model, forward, 1.5)
    emb.bind(rec, ctx["pert"])
    seed_all(seed)
    plans = sv.plan_epoch_v2(list(subjects), 5, ctx["args3"].seed, epoch, True, data.topo_map, data.subj_map,
                             ctx["drawcfg"])
    T3.STATE["steps"] = 0
    stats = T3.train_epoch(ctx["args3"], plans, emb, emb, model, probe, None, data, [], epoch, False)
    return stats, PRECLIP[0], probe.records[0], rec.seen, plans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdata", type=Path, required=True)
    ap.add_argument("--stage-root", type=Path, required=True)
    ap.add_argument("--ckpt", default="", help="checkpoint addestrato per i casi 'trained' (oltre all'init)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only-packed", action="store_true", help="solo packed contro sequenziale")
    ap.add_argument("--skip-packed", action="store_true", help="solo v2 contro v3")
    ap.add_argument("--meanpool-control", action="store_true",
                    help="controllo: modello con sola media (niente argmax del max-pooling)")
    ap.add_argument("--fp32", action="store_true", help="TF32 spento davvero (override del container compreso)")
    a = ap.parse_args()
    torch.nn.utils.clip_grad_norm_ = _clip_spy

    spec = json.loads((a.testdata / "spec.json").read_text())
    spec["n_blocks"] = 1
    a.stage_root.mkdir(parents=True, exist_ok=True)
    spec_path = a.stage_root / "spec_1block.json"
    spec_path.write_text(json.dumps(spec))
    common = RECIPE + ["--seed", "1234", "--dist_npz", str(a.testdata / "gt.npz")]
    sys.argv = ["train_runner"] + common + ["--data_dir", str(a.stage_root), "--runs_root", str(a.stage_root / "r1")]
    args1 = tr.parse_args()
    args3 = T3.build_parser().parse_args(common + [
        "--runs_root", str(a.stage_root / "r3"), "--data-spec", str(spec_path), "--split-json",
        str(a.testdata / "split.json"), "--stage-root", str(a.stage_root / "stage"), "--total-steps", str(10 ** 9),
        "--steps-per-epoch", "10", "--gt-keep-scale", "--eval_domain", "bfm", "--prepass-proc", "7"])
    T3.check_args(args3)
    device = T3.configure_device(args3, 0, False)
    if a.fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tf32_err = None
    if device.type == "cuda":
        x = torch.randn(512, 512, device=device, dtype=torch.float64)
        y = (x.float() @ x.float()).double()
        tf32_err = float(((y - x @ x).abs().max() / (x @ x).abs().max()).item())
    print(f"TF32: flag={torch.backends.cuda.matmul.allow_tf32} override={os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')} "
          f"errore relativo di una matmul fp32 contro fp64 = {tf32_err} (TF32 ~1e-3, fp32 ~1e-6)", flush=True)

    # dati v3 (gli stessi oggetti servono anche v2: il percorso dei dati e' nel test di livello 2)
    data = T3.Data(args3, 0, 1)
    data.setup(epochs=10)
    data.switch_block(0)
    if data.pending:
        T3.dv.kill_proc(data.pending[1])

    # v2: gruppi di etichette e GT come nella catena (train_steps + train_v2)
    import train_steps
    import train_v2
    train_steps.install_label_groups(spec["label_groups"])
    topo_v2 = tr._build_subject_topology_map(dataset=data.dataset, subj_map=data.subj_map)
    train_steps.install_gt_keep_scale()
    with train_v2._patched_v1(5, "bfm"):
        gt_v2, n2i_v2 = tr.load_gt_distance_matrix(str(a.testdata / "gt.npz"), dtype=np.float64)
    checks = {
        "topology_map_equal": topo_v2 == data.topo_map,
        "name_to_idx_equal": n2i_v2 == data.name_to_idx,
        "gt_equal": bool(np.array_equal(np.asarray(gt_v2).astype(np.float32), np.asarray(data.gt), equal_nan=True)),
    }
    print("controlli preliminari:", checks, flush=True)

    modes = parse_noise_modes(args3.noise_modes)
    probs = parse_noise_mode_weights(args3.noise_mode_weights, modes)
    ctx = {"subj_map": data.subj_map, "topo_v2": topo_v2, "n2i_v2": n2i_v2, "gt_v2": gt_v2, "device": device,
           "args1": args1, "args3": args3, "modes": modes, "probs": probs,
           "pert": PerturbationParams.from_namespace(args3),
           "drawcfg": sv.DrawCfg(args3.p_noise, args3.sigma_min, args3.sigma_max, modes, probs,
                                 args3.max_meshes_per_subject_train)}

    seed_everything(1234)
    models = {"init": build_model_v3(args3, device)}
    if a.ckpt:
        m = copy.deepcopy(models["init"])
        pack = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(pack["state_dict"])
        models["trained"] = m

    if a.meanpool_control:
        from diffusion_autoencoder import DiffusionEncoderOnly
        torch.manual_seed(1234)
        models = {"init_meanpool": DiffusionEncoderOnly(latent_dim=256, width=128, n_blocks=4, dropout=0.1,
                                                        pool_mode="mean").to(device)}
        if a.ckpt:
            mp = DiffusionEncoderOnly(latent_dim=256, width=128, n_blocks=4, dropout=0.1, pool_mode="mean").to(device)
            sd = {k: v for k, v in pack["state_dict"].items() if not k.startswith("pool_proj")}
            mp.load_state_dict(sd, strict=False)
            models["trained_meanpool"] = mp
    by_dom = {}
    for s in data.train:
        by_dom.setdefault(sv.domain_of(s), []).append(s)
    cases = []
    for dom in sorted(by_dom):
        for k, epoch in enumerate((3, 8, 21)):
            subj = sorted(by_dom[dom])[5 * k: 5 * k + 5]
            if len(subj) == 5:
                cases.append((dom, epoch, subj))

    results, ok = [], all(checks.values())
    for mname, model in (models.items() if not a.only_packed else []):
        for dom, epoch, subj in cases:
            seed = 4242 + epoch
            s2, pre2, post2, seen2 = run_v2(model, data.dataset, subj, epoch, ctx, seed)
            s2b, pre2b, post2b, _ = run_v2(model, data.dataset, subj, epoch, ctx, seed)
            s3, pre3, post3, seen3, plans = run_v3(model, data, subj, epoch, ctx, seed)
            dl = abs(s2["loss"] - s3["loss"])
            g_pre, g_pre_rel = grad_diff(pre2, pre3)
            g_post, g_post_rel = grad_diff(post2, post3)
            floor_l = abs(s2["loss"] - s2b["loss"])
            floor_g, _ = grad_diff(pre2, pre2b)
            terms = {k: abs(s2[k] - s3[k]) for k in ("subject_stress", "subject_rank", "mesh_stress", "mesh_rank", "id")}
            row = {"model": mname, "domain": dom, "epoch": epoch, "subjects": subj, "sigma": plans[0].sigma,
                   "n_meshes": len(plans[0].entries), "loss_v2": s2["loss"], "loss_v3": s3["loss"], "d_loss": dl,
                   "d_terms_max": max(terms.values()), "d_grad_preclip": g_pre, "d_grad_preclip_rel": g_pre_rel,
                   "d_grad_postclip": g_post, "d_grad_postclip_rel": g_post_rel, "same_meshes": seen2 == seen3,
                   "floor_v2_v2_loss": floor_l, "floor_v2_v2_grad": floor_g}
            row["pass"] = bool(dl <= TOL and g_pre <= TOL and g_post <= TOL and row["same_meshes"])
            ok &= row["pass"]
            results.append(row)
            print(f"{mname:7s} {dom:4s} ep{epoch:2d} sigma={row['sigma']:.2e} M={row['n_meshes']:2d} "
                  f"loss v2={s2['loss']:.6f} v3={s3['loss']:.6f} dL={dl:.1e} dG(pre)={g_pre:.1e} "
                  f"dG(post)={g_post:.1e} mesh={'=' if row['same_meshes'] else 'DIVERSE'} "
                  f"[pavimento v2-v2: dL={floor_l:.1e} dG={floor_g:.1e}] {'OK' if row['pass'] else 'FALLITO'}",
                  flush=True)

    # packed / groups contro sequenziale, deterministico (dropout -> Identity, niente rumore latente, sigma 0).
    # Due livelli: (i) la loss v2 intera, informativa: torch.cdist con piu' di 25 righe usa la formula a prodotti
    # scalari, mal condizionata quando gli embedding sono vicini (modello all'init), quindi scarti di 1e-7 su Z
    # possono diventare grandi nel gradiente; (ii) il CRITERIO: Z e gradienti dei parametri di (Z * G).sum() con
    # G fisso, cioe' la stessa forward e la stessa backward a parita' di cotangente.
    packed = []
    det = copy.deepcopy(ctx)
    det["args3"] = copy.deepcopy(args3)
    det["args3"].p_noise = 0.0
    det["args3"].train_latent_noise = False
    det["drawcfg"] = sv.DrawCfg(0.0, args3.sigma_min, args3.sigma_max, modes, probs, args3.max_meshes_per_subject_train)

    def vjp(m, plan, mode):
        emb = StepEmbedder(m, mode, 1.5, 0.05)
        emb.bind(data.dataset, ctx["pert"])
        emb.train()
        m.zero_grad(set_to_none=True)
        Z = emb(plan.entries, 0.0, False)
        G = torch.randn(Z.shape, generator=torch.Generator().manual_seed(5), dtype=Z.dtype).to(Z.device)
        (Z * G).sum().backward()
        return Z.detach(), [p.grad.detach().clone() for p in m.parameters()]

    for tf32 in ([] if a.skip_packed else [not a.fp32]):
        for mname, model in models.items():
            m = copy.deepcopy(model)
            for mod in list(m.modules()):
                for name, child in list(mod.named_children()):
                    if isinstance(child, torch.nn.Dropout):     # Identity: nessuna estrazione casuale
                        setattr(mod, name, torch.nn.Identity())
            for dom, epoch, subj in cases[::3]:
                plan = sv.plan_epoch_v2(list(subj), 5, args3.seed, epoch, True, data.topo_map, data.subj_map,
                                        det["drawcfg"])[0]
                z0, g0 = vjp(m, plan, "sequential")
                sa, pa, _, _, _ = run_v3(m, data, subj, epoch, det, 777)
                for mode in ("groups", "packed"):
                    z1, g1 = vjp(m, plan, mode)
                    sb, pb, _, _, _ = run_v3(m, data, subj, epoch, det, 777, forward=mode)
                    gv, gv_rel = grad_diff(g0, g1)
                    gl, gl_rel = grad_diff(pa, pb)
                    row = {"tf32": tf32, "matmul_rel_err_vs_fp64": tf32_err, "model": mname, "domain": dom,
                           "epoch": epoch, "forward": mode,
                           "z_max_abs": float((z0 - z1).abs().max()), "z_scale": float(z0.abs().max()),
                           "vjp_grad_max_abs": gv, "vjp_grad_rel": gv_rel,
                           "loss_seq": sa["loss"], "loss_other": sb["loss"], "d_loss": abs(sa["loss"] - sb["loss"]),
                           "loss_grad_max_abs": gl, "loss_grad_rel": gl_rel}
                    row["z_rel"] = row["z_max_abs"] / max(row["z_scale"], 1e-30)
                    packed.append(row)
                    print(f"{mode:7s} tf32={tf32} {mname:7s} {dom:4s} Z rel={row['z_rel']:.1e} VJP rel={gv_rel:.1e} | "
                          f"loss v2: seq={sa['loss']:.6f} {mode}={sb['loss']:.6f} dL={row['d_loss']:.1e} "
                          f"dG rel={gl_rel:.1e}", flush=True)
    if packed:
        ok_packed = all(r["z_rel"] <= TOL and r["vjp_grad_rel"] <= TOL for r in packed) if a.fp32 else None
    else:
        ok_packed = None
    out = {"packed_fp32_pass": ok_packed, "criterion": f"|dL| <= {TOL} e max|dG| <= {TOL} (pre e post clip), stesse mesh lette",
           "checks": checks, "v2_vs_v3": results, "packed_vs_sequential": packed, "all_pass": bool(ok)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=str))
    if not a.only_packed:
        print(f"\nEQUIVALENZA v2 = v3: {'PASSATA' if ok else 'FALLITA'} ({sum(r['pass'] for r in results)}/{len(results)} casi)")
    if a.fp32:
        print(f"GROUPS/PACKED = SEQUENZIALE in fp32 (Z e VJP, rel <= {TOL}): {'PASSATO' if ok_packed else 'FALLITO'}")


if __name__ == "__main__":
    main()
