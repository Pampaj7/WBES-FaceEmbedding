#!/usr/bin/env python3
"""Tempi del riconoscimento in dominio (protocollo: aau/runs/indomain_recog/protocol.md, "Tempi").

    aau/run.sh             aau/indomain/ir_timing.py model     --stage-dir /tmp/x --out <json> -- <argomenti di eval>
    aau/outlineB/run_o3d.sh aau/indomain/ir_timing.py facebench --stage-dir /tmp/x --out <json>
    aau/baselines/run_bl.sh aau/indomain/ir_timing.py arcface   --stage-dir /tmp/x --out <json>
    (ir_timing.sbatch: le tre parti in fila, nello stesso job, quindi sullo stesso nodo)

Tre ambienti diversi (container del modello, venv open3d, venv con insightface), quindi tre
invocazioni dello stesso file. Il campione e' lo stesso per tutte: 5 soggetti BFM e 5 ICT
(``rng(1234)`` sugli insiemi di ``sets.json``) x 6 topologie = 60 mesh, copiate in ``--stage-dir``
(/tmp del nodo) dalla prima parte che gira; 60 coppie fra loro, topologie diverse, 30 stesso
soggetto e 30 soggetti diversi dello stesso dominio. Le prime ``WARMUP`` misure di ogni voce si
buttano (import pigri, allocazioni CUDA, cache).

Ogni parte scrive in ``--out`` i tempi grezzi in secondi per voce, piu' host, CPU e GPU; mediana e
IQR li calcola il summarizer. Funzioni di calcolo importate, non riscritte:
  - model: ``compute_operators`` di DiffusionNet sulla mesh ad area unitaria (come
    ``areanorm_operators.py``); embedding = ``GTReadyDataset.__getitem__`` + ``sample_to_device`` +
    ``forward_model`` (la catena di ``zs_embed.py``), su GPU con sincronizzazione e su CPU;
    confronto ``||z_a - z_b||`` e retrieval 1:N (distanze + argsort) su embedding veri di
    ``embeddings/joint__{bfm,ict}.npz``, gallerie ricampionate con rimpiazzo fino a N;
  - facebench: ``run_geometry_pipeline`` col solo stadio ``rigid`` (ICP + Chamfer) e col solo
    ``nicp`` (che rifa' l'ICP rigido prima della NICP), lettura delle mesh compresa;
  - arcface: lettura + 3 render a normal map (``zs_arcface_render.render_normals``) + 3 embedding
    con ``ws3a_perceptual.embed_render`` sul crop calibrato dell'insieme + media e L2.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
RUNS = AAU_DIR / "runs" / "indomain_recog"
TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")
WARMUP = 3
N_GALLERY = (100, 1000, 10000)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        k = argv.index("--")
        argv, rest = argv[:k], argv[k + 1:]
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("part", choices=("model", "facebench", "arcface"))
    p.add_argument("--stage-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-queries", type=int, default=100)
    p.add_argument("--n-compare", type=int, default=1000)
    a = p.parse_args(argv)
    a.eval_args = rest
    return a


def sample_meshes() -> list[tuple[str, str, str]]:
    """(dominio, soggetto, topologia) delle 60 mesh, sempre le stesse."""
    sets = json.loads((RUNS / "sets.json").read_text())["sets"]
    rng = np.random.default_rng(1234)
    out = []
    for dom in ("bfm", "ict"):
        for s in sorted(rng.choice(sets[dom]["subjects"], size=5, replace=False).tolist()):
            out += [(dom, s, t) for t in TOPOLOGIES]
    return out


def sample_pairs(meshes) -> list[tuple[int, int]]:
    """30 stesso soggetto + 30 soggetti diversi (stesso dominio), sempre topologie diverse."""
    rng = np.random.default_rng(1234)
    same, diff = [], []
    for i, (da, sa, ta) in enumerate(meshes):
        for j, (db, sb, tb) in enumerate(meshes):
            if i < j and da == db and ta != tb:
                (same if sa == sb else diff).append((i, j))
    pick = lambda xs: [xs[k] for k in sorted(rng.choice(len(xs), size=30, replace=False))]  # noqa: E731
    return pick(same) + pick(diff)


def stage(meshes, dest: Path) -> list[Path]:
    """Copia gli npz (con gli operatori) delle 60 mesh in ``dest``; idempotente fra le parti."""
    sets = json.loads((RUNS / "sets.json").read_text())["sets"]
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for dom, s, t in meshes:
        out = dest / f"{s}_GTready_{t}.npz"
        if not out.exists():
            shutil.copyfile(Path(sets[dom]["view_dir"]) / out.name, out)
        paths.append(out)
    return paths


def hardware() -> dict:
    cpu = ""
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"host": platform.node(), "cpu": cpu, "job": os.environ.get("SLURM_JOB_ID", ""),
            "omp_threads": os.environ.get("OMP_NUM_THREADS", "")}


def timed(fn, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


# ------------------------------------------------------------------------------- modello

def part_model(a, meshes, paths) -> dict:
    import torch

    torch.set_num_threads(1)
    PERT = REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic" / "perturbated"
    sys.path.insert(0, str(PERT))
    sys.path.insert(0, str(PERT.parent))
    import compare_model_vs_chamfer_topology_breakdown as bd
    from diffusion_net.geometry import compute_operators

    base = bd.base
    res: dict = {}

    # (a) operatori, CPU a un thread, sulla mesh ad area unitaria come areanorm_operators.py.
    ops = []
    for k, p in enumerate([paths[0]] * WARMUP + paths):
        with np.load(p) as d:
            V, F = np.asarray(d["verts"], np.float64), np.asarray(d["faces"], np.int64)
        t0 = time.perf_counter()
        tri = V[F]
        A = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
        Vn = (V - V.mean(0)) / np.sqrt(A)
        compute_operators(torch.tensor(Vn, dtype=torch.float32), torch.tensor(F, dtype=torch.int32), k_eig=128)
        if k >= WARMUP:
            ops.append(time.perf_counter() - t0)
            print(f"[ir-timing] operatori {p.name}: {ops[-1]:.2f}s ({len(V)} vertici)", flush=True)
    res["operators_cpu"] = ops
    res["n_vertices"] = [int(np.load(p)["verts"].shape[0]) for p in paths]

    # (b) embedding: la catena di zs_embed.py, sulla stage dir.
    sys.argv = [sys.argv[0]] + a.eval_args + ["--data_dir", str(a.stage_dir)]
    cli_args = bd.parse_args()
    run_dir, checkpoint_path = base._resolve_run_dir_and_checkpoint(
        cli_args.model_path, selector=str(cli_args.checkpoint_selector))
    model_args = base._resolve_runtime_args(cli_args, base.merge_run_args(checkpoint_path,
                                                                          explicit_config_json=cli_args.config_json))
    base.seed_everything(int(model_args.seed))
    dataset = base.GTReadyDataset(str(a.stage_dir))
    by_name = {Path(f).name: k for k, f in enumerate(dataset.files)}
    order = [by_name[p.name] for p in paths]
    state = base.load_checkpoint_bundle(checkpoint_path)["state_dict"]
    res["checkpoint"] = str(checkpoint_path)

    for dev_name in ("cuda", "cpu"):
        if dev_name == "cuda" and not torch.cuda.is_available():
            continue
        device = torch.device(dev_name)
        model = base.build_model(args=model_args, device=device)
        model.load_state_dict(state, strict=True)
        model.eval()
        sync = torch.cuda.synchronize if dev_name == "cuda" else (lambda: None)
        times, Z = [], []
        with torch.no_grad():
            for k, idx in enumerate([order[0]] * WARMUP + order):
                t0 = time.perf_counter()
                sample_d = base.sample_to_device(dataset[idx], device=device)
                z, _ = base.forward_model(model=model, sample_dict=sample_d, V_in=sample_d["verts"],
                                          return_gate_info=False, add_noise=False)
                sync()
                if k >= WARMUP:
                    times.append(time.perf_counter() - t0)
                    Z.append(z.squeeze(0).float().cpu().numpy())
        res[f"embed_{dev_name}"] = times
        print(f"[ir-timing] embedding {dev_name}: mediana {np.median(times) * 1e3:.1f} ms", flush=True)
        del model
    if torch.cuda.is_available():
        res["gpu"] = torch.cuda.get_device_name(0)

    # (c) confronto e (d) retrieval su embedding veri precalcolati.
    E = np.concatenate([np.load(RUNS / "embeddings" / f"joint__{s}.npz")["Z"] for s in ("bfm", "ict")]).astype(np.float32)
    rng = np.random.default_rng(1234)
    ia, ib = rng.integers(0, len(E), (2, a.n_compare + WARMUP))
    cmp = []
    for k in range(len(ia)):
        za, zb = E[ia[k]], E[ib[k]]
        t0 = time.perf_counter()
        float(np.linalg.norm(za - zb))
        if k >= WARMUP:
            cmp.append(time.perf_counter() - t0)
    res["compare_cpu"] = cmp
    for n in N_GALLERY:
        G = E[rng.integers(0, len(E), n)]
        Q = E[rng.integers(0, len(E), a.n_queries + WARMUP)]
        t_cpu = []
        for k, q in enumerate(Q):
            t0 = time.perf_counter()
            np.argsort(np.sqrt(((G - q) ** 2).sum(1)))
            if k >= WARMUP:
                t_cpu.append(time.perf_counter() - t0)
        res[f"retrieval_cpu_N{n}"] = t_cpu
        if torch.cuda.is_available():
            Gt, Qt = torch.tensor(G, device="cuda"), torch.tensor(Q, device="cuda")
            t_gpu = []
            for k in range(len(Qt)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                torch.argsort(torch.linalg.vector_norm(Gt - Qt[k], dim=1))
                torch.cuda.synchronize()
                if k >= WARMUP:
                    t_gpu.append(time.perf_counter() - t0)
            res[f"retrieval_gpu_N{n}"] = t_gpu
        print(f"[ir-timing] retrieval N={n}: CPU mediana {np.median(t_cpu) * 1e3:.3f} ms", flush=True)
    return res


# ------------------------------------------------------------------------------ faceBench

def part_facebench(a, meshes, paths) -> dict:
    sys.path.insert(0, str(REPO_ROOT / "faceBench" / "latentVSpipeline"))
    import run_facebench_remesh as rfr

    pairs = sample_pairs(meshes)
    res: dict = {"pairs": [[int(i), int(j)] for i, j in pairs]}
    for key, stages in (("icp_chamfer", ["rigid"]), ("nicp_p2tri", ["nicp"])):
        times, vals = [], []
        for k, (i, j) in enumerate(pairs[:WARMUP] + pairs):
            t0 = time.perf_counter()
            m = rfr.run_geometry_pipeline(str(paths[i]), str(paths[j]), stages, 4096, k)
            if k >= WARMUP:
                times.append(time.perf_counter() - t0)
                vals.append(float(m.get("rigid_p2p" if key == "icp_chamfer" else "nicp_p2tri", np.nan)))
        res[key] = times
        res[f"{key}_values"] = vals
        print(f"[ir-timing] {key}: mediana {np.median(times):.2f}s su {len(times)} coppie, "
              f"NaN {int(np.isnan(vals).sum())}", flush=True)
    return res


# -------------------------------------------------------------------------------- ArcFace

def part_arcface(a, meshes, paths) -> dict:
    sys.path.insert(0, str(AAU_DIR / "zs3dmm"))
    import zs_arcface_render as zar

    perc, render = zar.perc, zar.render
    rot = {"bfm": "none", "ict": "x180"}
    setup = {}
    for dom in ("bfm", "ict"):
        root = RUNS / "arcface" / dom / "normals"
        camera = json.loads((root / "renders" / "camera.json").read_text())
        if camera["base_rotation"] != rot[dom] or camera["mode"] != "normals":
            raise SystemExit(f"{root}: camera {camera['base_rotation']}/{camera['mode']}, attesi {rot[dom]}/normals")
        setup[dom] = (camera, perc.build_extractor("arcface", "cpu", root))
    res: dict = {"render": [], "embed": [], "total": []}
    for k, ((dom, s, t), p) in enumerate(list(zip(meshes, paths))[:WARMUP] + list(zip(meshes, paths))):
        camera, extractor = setup[dom]
        center = np.asarray(camera["center"], np.float64)
        t0 = time.perf_counter()
        V, F = zar.load_verts_faces(p.parent, s, t, zar.BASE_ROTATIONS[rot[dom]])
        imgs = [zar.render_normals(zar._yaw_rotate(V, y, center) if y else V, F, 512, camera["scale"], center)
                for y in camera["yaws"]]
        t1 = time.perf_counter()
        E = np.stack([perc.embed_render(extractor, img, y) for img, y in zip(imgs, camera["yaws"])]).mean(0)
        E /= np.linalg.norm(E)
        t2 = time.perf_counter()
        if k >= WARMUP:
            res["render"].append(t1 - t0)
            res["embed"].append(t2 - t1)
            res["total"].append(t2 - t0)
    print(f"[ir-timing] ArcFace normal map: mediana {np.median(res['total']):.2f}s per mesh "
          f"(render {np.median(res['render']):.2f}s, embedding {np.median(res['embed']):.2f}s)", flush=True)
    return res


def main() -> None:
    a = parse_args()
    meshes = sample_meshes()
    paths = stage(meshes, a.stage_dir)
    print(f"[ir-timing] {a.part}: {len(meshes)} mesh in {a.stage_dir}, host {platform.node()}", flush=True)
    res = {"part_model": part_model, "part_facebench": part_facebench, "part_arcface": part_arcface}[f"part_{a.part}"](
        a, meshes, paths)
    res.update(hardware(), meshes=[list(m) for m in meshes], warmup=WARMUP)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res) + "\n")
    print(f"[ir-timing] scritto {a.out}", flush=True)


if __name__ == "__main__":
    main()
