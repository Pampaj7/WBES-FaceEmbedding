#!/usr/bin/env python3
"""Emendamento 2, sez. 3 (POST HOC): costi MISURATI dei bracci e del confronto di una coppia (PROTOCOL_emendamento_2.md).

    aau/run.sh aau/baselines_param/bp_cost.py ops --workers 32                    (CPU, preprocessing dei bracci)
    aau/run.sh v3_work/trainer/eval_v3.py -- aau/baselines_param/bp_cost.py forward --key factorized_s1234 \\
        <argomenti di zs_embed.py senza --embeddings_out>                         (GPU, forward dei bracci)
    aau/outlineB/run_o3d.sh aau/baselines_param/bp_cost.py pairs                   (CPU, un thread: confronto di una coppia)
    (cost.sbatch; uscite in <OUT_ROOT>/cost_e2/)

``ops``: le 600 mesh di HIFI3D (100 soggetti valutati x 6 topologie), per mesh il corpo di
``v2_work/potential/areanorm_operators.py`` (copiato, cronometrato per passo): lettura (``load_mesh``), area e
normalizzazione, ``compute_operators`` k_eig 128, scrittura dell'npz (``save_npz``) su /tmp del nodo; un processo per
mesh con un thread (torch e BLAS), ``--workers`` processi. Uscita ``ops.npz`` (tempi per passo, vertici, topologia).

``forward``: la catena di ``aau/zs3dmm/zs_embed.py`` (argomenti, soggetti, piano di eval, modello, checkpoint, agganci di
``eval_v3``) e poi, per ogni mesh, (a) batch 1: lettura dell'npz degli operatori (``dataset[i]``), copia sul device
(``sample_to_device``), ``forward_model`` senza rumore, separati, ``torch.cuda.synchronize`` attorno a ogni misura,
``WARMUP`` forward esclusi; (b) batch tipico del training dei bracci (``--forward sequential``, 5 soggetti x 6 mesh):
gruppi di ``GROUP`` mesh gia' sul device, un forward per mesh, tempo del gruppo / ``GROUP``. Controllo: gli embedding
coincidono con quelli di fact_paired (``fact_paired.embeddings``, scarto massimo). Uscita ``forward_<chiave>.json`` e
``.npz``.

``pairs``: tempo per coppia di ||beta_i - beta_j|| e delle distanze sulle mesh d'identita' di B (FR, SR, composizione;
rigida verso mu una volta per mesh), con le regioni e i beta di B di HIFI3D (``fit_e1.npz``), e dei bracci (||z_i - z_j||,
d_F calibrata) dagli embedding di HIFI3D; numpy a un thread. Uscita ``pairs.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
OUT = REPO / "aau" / "runs" / "evidence" / "baselines_param" / "cost_e2"
HIFI_NPZ = REPO / "datasets" / "HIFI3D" / "eval_view" / "npz"
K_EIG = 128
WARMUP, GROUP = 10, 30
ARMS = {"factorized_s1234": ("hifi", "factorized", "072"), "ctrlfr_s1234": ("hifi", "ctrlfr", "072")}


def host_info() -> dict:
    cpu = ""
    try:
        cpu = next(ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID", "none"), "cpu": cpu,
            "cpus_allocated": os.environ.get("SLURM_CPUS_PER_TASK"), "python": platform.python_version()}


def stats(x) -> dict:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return {"n": int(len(x)), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "mean": float(x.mean()), "max": float(x.max())} if len(x) else {"n": 0}


# ------------------------------------------------------------------------------------------- operatori

def _ops(task):
    import torch
    torch.set_num_threads(1)
    from diffusion_net.geometry import compute_operators
    from areanorm_operators import total_area
    from potential_operators import load_mesh, save_npz
    path, tmp = task
    t0 = time.perf_counter()
    V, F = load_mesh(path)
    t1 = time.perf_counter()
    A = total_area(V, F)
    V = (V - V.mean(0)) / np.sqrt(A)
    Vt, Ft = torch.tensor(V, dtype=torch.float32), torch.tensor(F, dtype=torch.int32)
    t2 = time.perf_counter()
    _, mass, L, evals, evecs, gX, gY = compute_operators(Vt, Ft, k_eig=K_EIG)
    t3 = time.perf_counter()
    save_npz(Path(tmp) / Path(path).name, V, F, mass, L, evals, evecs, gX, gY)
    t4 = time.perf_counter()
    return [t1 - t0, t2 - t1, t3 - t2, t4 - t3], len(V), len(F)


def run_ops(workers: int, n_subjects: int = 0) -> None:
    """``n_subjects`` > 0: solo i primi soggetti, uscite ``ops_<n>s_<workers>w.*`` (es. un processo solo, senza
    contesa sul nodo)."""
    import multiprocessing as mp
    for p in (REPO / "v2_work" / "potential", REPO / "diffusion-net" / "src"):
        sys.path.insert(0, str(p))
    sys.path.insert(0, str(REPO / "aau" / "baselines_mm"))
    import blmm
    subj = blmm.subjects("hifi3d")[:n_subjects] if n_subjects > 0 else blmm.subjects("hifi3d")
    files = [blmm.mesh_path("hifi3d", s, t) for s in subj for t in blmm.TOPOLOGIES]
    tag = "ops" if n_subjects <= 0 else f"ops_{n_subjects}s_{workers}w"
    tmp = Path(f"/tmp/wbes_bp_cost_{os.environ.get('SLURM_JOB_ID', 'manual')}")
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_ops, [(str(f), str(tmp)) for f in files], chunksize=1)
    wall = time.time() - t0
    T = np.asarray([r[0] for r in res])
    topo = np.asarray([f.stem.split("_GTready_")[1] for f in files])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{tag}.npz", t_stage=T, stages=np.asarray(["load", "area_norm", "compute_operators", "save"]),
             n_vertices=np.asarray([r[1] for r in res]), n_faces=np.asarray([r[2] for r in res]), topology=topo,
             files=np.asarray([str(f.relative_to(REPO)) for f in files]))
    tot = T.sum(1)
    out = {"k_eig": K_EIG, "workers": workers, "threads_per_process": 1, "n_meshes": len(files), "wall_s": wall,
           "meshes_per_s_node": len(files) / wall, "total_s_per_mesh": stats(tot),
           "by_stage": {s: stats(T[:, k]) for k, s in enumerate(["load", "area_norm", "compute_operators", "save"])},
           "by_topology": {t: {"total": stats(tot[topo == t]),
                               "n_vertices_median": float(np.median([r[1] for r, x in zip(res, topo) if x == t]))}
                           for t in sorted(set(topo))}, **host_info()}
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"[bp-cost] ops: {len(files)} mesh in {wall:.0f}s con {workers} processi; s/mesh mediana "
          f"{np.median(tot):.2f}, p95 {np.percentile(tot, 95):.2f}; per topologia " +
          ", ".join(f"{t} {np.median(tot[topo == t]):.2f}" for t in sorted(set(topo))), flush=True)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------------------------- forward

def run_forward(key: str, rest: list) -> None:
    """Dentro ``eval_v3.py --``: la catena di zs_embed.main fino al modello, poi i tempi."""
    import torch
    pert = REPO / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic" / "perturbated"
    sys.path.insert(0, str(pert))
    sys.path.insert(0, str(pert.parent))
    import compare_model_vs_chamfer_topology_breakdown as bd
    base = bd.base
    sys.argv = [sys.argv[0]] + rest
    cli_args = bd.parse_args()
    # zs_embed.main (copiata) fino al modello
    run_dir, checkpoint_path = base._resolve_run_dir_and_checkpoint(
        cli_args.model_path, selector=str(cli_args.checkpoint_selector))
    base_args = base.merge_run_args(checkpoint_path, explicit_config_json=cli_args.config_json)
    model_args = base._resolve_runtime_args(cli_args, base_args)
    base.seed_everything(int(model_args.seed))
    device = base._resolve_device(cli_args.device)
    dataset = base.GTReadyDataset(model_args.data_dir)
    _, gt_name_to_idx = base.load_gt_distance_matrix(model_args.dist_npz, subject_re=base.SUBJECT_RE_ANY,
                                                     dtype=base.np.float64)
    subject_map = base.build_subject_map(dataset.files, subject_re=base.SUBJECT_RE_ANY)
    subjects = sorted([sid for sid in subject_map.keys() if sid in gt_name_to_idx])
    _, _, target_subjects = base._select_subject_subset(
        subjects=subjects, subject_split=str(cli_args.subject_split),
        eval_fraction=float(cli_args.eval_fraction), seed=int(model_args.seed),
        max_subjects=int(cli_args.max_subjects))
    eval_plan = base.build_eval_plan(
        subj_map=subject_map, eval_subjects=target_subjects,
        max_meshes_per_subject_eval=int(model_args.max_meshes_per_subject_eval), seed=int(model_args.seed))
    records = base.build_sample_eval_records(dataset=dataset, eval_plan=eval_plan,
                                             eval_subjects=target_subjects, sample_cache=None)
    model = base.build_model(args=model_args, device=device)
    model.load_state_dict(base.load_checkpoint_bundle(checkpoint_path)["state_dict"], strict=True)
    model.eval()
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    def fwd(sd):
        z, _ = base.forward_model(model=model, sample_dict=sd, V_in=sd["verts"], return_gate_info=False,
                                  add_noise=False)
        return z.squeeze(0)

    n = len(records)
    t_load, t_dev, t_fwd, Z = np.zeros(n), np.zeros(n), np.zeros(n), []
    nv = np.zeros(n, dtype=np.int64)
    with torch.no_grad():
        for r in records[:WARMUP]:
            fwd(base.sample_to_device(dataset[int(r.dataset_idx)], device=device))
        sync()
        for k, r in enumerate(records):                         # (a) batch 1
            t0 = time.perf_counter()
            s = dataset[int(r.dataset_idx)]
            t1 = time.perf_counter()
            sd = base.sample_to_device(s, device=device)
            sync()
            t2 = time.perf_counter()
            z = fwd(sd)
            sync()
            t3 = time.perf_counter()
            t_load[k], t_dev[k], t_fwd[k] = t1 - t0, t2 - t1, t3 - t2
            nv[k] = int(sd["verts"].shape[0])
            Z.append(z.float().cpu().numpy())
        t_grp = []                                               # (b) gruppi di GROUP mesh gia' sul device
        order = np.random.default_rng(1234).permutation(n)
        for g in range(0, n - GROUP + 1, GROUP):
            sds = [base.sample_to_device(dataset[int(records[i].dataset_idx)], device=device) for i in order[g:g + GROUP]]
            sync()
            t0 = time.perf_counter()
            for sd in sds:
                fwd(sd)
            sync()
            t_grp.append((time.perf_counter() - t0) / GROUP)
            del sds
    Z = np.stack(Z)
    sys.path.insert(0, str(REPO / "v3_work" / "trainer" / "tools"))
    dom, v, e = ARMS[key]
    ref = sorted((REPO / "aau/runs/evidence/trainer_v3/ablations/c3f_eval" / f"form_{dom}").glob(
        f"data_*/scale_v3{v}fulle{e}*/zs_zeroshot/embeddings.npz"))[0]
    with np.load(ref, allow_pickle=True) as z:
        keys = list(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))
        Zr, ck_ref = np.asarray(z["Z"], np.float64), str(z["checkpoint"])
    pos = {kk: i for i, kk in enumerate(keys)}
    mine = [(r.subject_id, r.topology_label) for r in records]
    diff = float(np.abs(Z - Zr[[pos[kk] for kk in mine]]).max())
    topo = np.asarray([t for _, t in mine])
    tot = t_load + t_dev + t_fwd
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    out = {"key": key, "checkpoint": str(checkpoint_path), "reference_embeddings": str(ref.relative_to(REPO)),
           "checkpoint_matches_reference": str(Path(ck_ref).resolve()) == str(Path(checkpoint_path).resolve()),
           "max_abs_diff_vs_reference": diff, "device": gpu, "n_meshes": n, "warmup": WARMUP, "group": GROUP,
           "batch1": {"load_ops_npz": stats(t_load), "to_device": stats(t_dev), "forward": stats(t_fwd),
                      "total": stats(tot)},
           "batch1_forward_by_topology": {t: stats(t_fwd[topo == t]) for t in sorted(set(topo))},
           "group_forward_per_mesh": stats(t_grp), "n_vertices_median": float(np.median(nv)), **host_info()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"forward_{key}.json").write_text(json.dumps(out, indent=1) + "\n")
    np.savez(OUT / f"forward_{key}.npz", t_load=t_load, t_dev=t_dev, t_fwd=t_fwd, t_group=np.asarray(t_grp),
             n_vertices=nv, topology=topo)
    print(f"[bp-cost] forward {key} su {gpu}: batch 1 mediana lettura {np.median(t_load) * 1e3:.1f} ms, copia "
          f"{np.median(t_dev) * 1e3:.1f} ms, forward {np.median(t_fwd) * 1e3:.1f} ms; gruppi di {GROUP}: "
          f"{np.median(t_grp) * 1e3:.1f} ms/mesh; scarto dagli embedding di fact_paired {diff:.1e}", flush=True)


# ---------------------------------------------------------------------------------------------- coppie

def timed(f, reps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t0) / reps


def run_pairs() -> None:
    sys.path.insert(0, str(THIS))
    import bp
    rng = np.random.default_rng(0)
    out = {"threads": os.environ.get("OPENBLAS_NUM_THREADS"), **host_info(), "B": {}, "arms": {}}
    for name in bp.MODELS:
        with np.load(bp.out_dir("hifi3d", name) / "fit.npz") as z:
            ctx = bp.context("hifi3d", name, np.asarray(z["region_vertices"]))
        with np.load(bp.out_dir("hifi3d", name) / "fit_e1.npz") as z:
            beta = np.asarray(z["vb_beta"])
        w = ctx["w_r"]
        wn = w / w.sum()
        m = wn @ ctx["mu_r"]
        cs_mu = bp.identity_sizes(np.zeros((1, ctx["k_id"])), ctx)[0]

        def enroll(b):                                     # per mesh: mesh d'identita', rigida verso mu, scala SR
            a = bp.weighted_rigid_to(bp.identity_meshes(b[None], ctx), ctx["mu_r"], w)[0]
            S = np.sqrt(wn @ ((a - m) ** 2).sum(-1))
            return a, m + (a - m) * (cs_mu / S), S

        A = [enroll(b) for b in beta[:50]]
        t_enroll = timed(lambda: enroll(beta[0]), 50)
        ii, jj = rng.integers(0, 50, 2000), rng.integers(0, 50, 2000)
        sw = np.sqrt(w / w.sum())

        def d_mesh(x, y):
            return float(np.sqrt(((sw[:, None] * (x - y)) ** 2).sum()))

        t_fr = timed(lambda: [d_mesh(A[i][0], A[j][0]) for i, j in zip(ii, jj)], 3) / len(ii)
        t_comp = timed(lambda: [np.sqrt((A[i][2] - A[j][2]) ** 2 + A[i][2] * A[j][2] * (d_mesh(A[i][1], A[j][1]) / cs_mu) ** 2)
                                for i, j in zip(ii, jj)], 3) / len(ii)
        t_coef = timed(lambda: [float(np.linalg.norm(beta[i] - beta[j])) for i, j in zip(ii, jj)], 3) / len(ii)
        t_all = timed(lambda: bp.mesh_distances(beta, ctx), 1)
        n = len(beta)
        out["B"][name] = {"n_region_vertices": int(len(ctx["used"])), "k_id": int(ctx["k_id"]),
                          "enroll_identity_mesh_s": t_enroll, "pair_coef_s": t_coef, "pair_mesh_fr_or_sr_s": t_fr,
                          "pair_comp_s": t_comp, "mesh_distances_all_s": t_all, "mesh_distances_n": n,
                          "mesh_distances_per_pair_s": t_all / (n * (n - 1) / 2)}
    sys.path.insert(0, str(REPO / "v3_work" / "trainer" / "tools"))
    for key, (dom, v, e) in ARMS.items():
        ref = sorted((REPO / "aau/runs/evidence/trainer_v3/ablations/c3f_eval" / f"form_{dom}").glob(
            f"data_*/scale_v3{v}fulle{e}*/zs_zeroshot/embeddings.npz"))[0]
        with np.load(ref, allow_pickle=True) as z:
            Z = np.asarray(z["Z"], np.float64)
        ii, jj = rng.integers(0, len(Z), 20000), rng.integers(0, len(Z), 20000)
        if v == "factorized":
            S, U = np.exp(Z[:, 0]), Z[:, 1:]

            def one(i, j, c=0.5):                          # c qualunque: conta solo il tempo
                dP = float(np.linalg.norm(U[i] - U[j]))
                return np.sqrt((S[i] - S[j]) ** 2 + S[i] * S[j] * (c * dP) ** 2)
        else:
            def one(i, j):
                return float(np.linalg.norm(Z[i] - Z[j]))
        t_one = timed(lambda: [one(i, j) for i, j in zip(ii, jj)], 3) / len(ii)
        t0 = time.perf_counter()
        sq = (Z ** 2).sum(1)
        np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0, None))
        t_all = time.perf_counter() - t0
        out["arms"][key] = {"dim": int(Z.shape[1]), "pair_s": t_one, "all_pairs_s": t_all, "n": len(Z),
                            "per_pair_amortized_s": t_all / (len(Z) * (len(Z) - 1) / 2)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pairs.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("ops", "forward", "pairs"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--key", choices=tuple(ARMS))
    p.add_argument("--subjects", type=int, default=0, help="ops: solo i primi N soggetti (0 = tutti)")
    a, rest = p.parse_known_args()
    if a.step == "ops":
        run_ops(a.workers, a.subjects)
    elif a.step == "forward":
        run_forward(a.key, rest)
    else:
        run_pairs()


if __name__ == "__main__":
    main()
