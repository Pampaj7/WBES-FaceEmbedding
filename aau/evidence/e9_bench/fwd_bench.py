#!/usr/bin/env python3
"""E9 (b): forward + backward a gruppi su una L40S, per taglia del modello e numero di vertici.

    aau/run.sh aau/evidence/e9_bench/fwd_bench.py --work-dir /tmp/$SLURM_JOB_ID/fwd \
        --out-dir aau/runs/evidence/e9/fwd

Modello: ``DiffusionEncoderOnly`` (latent 256, meanmax, dropout 0.1, in train, rumore latente 0.01
come il run su scala) nelle taglie S = 128x4, M = 256x6, L = 384x8. Un passo = zero_grad, forward
di B mesh, loss surrogata (media dei quadrati degli embedding: il costo della loss vera e' trascurabile),
backward, AdamW. Tensori fp32 come nel training; i valori sparsi restano fp32 (fp16 va in overflow,
aau/data_scale/PLAN.md, variante a16).

Percorsi:
  seq     un forward per mesh (``robustness.model_helpers.forward_model``, il percorso del trainer v1),
          poi un solo backward;
  bat     ``v2_work/fastio/batched.py::embed_samples`` cosi' com'e' (gruppi di taglia simile entro
          ``pad_slack``, padding, gradienti con un ``torch.mm`` sparso per mesh);
  pk      "packed" (questo file): vertici di tutte le mesh concatenati [T, C]; gradX/gradY come UNA
          matrice a blocchi diagonali [T, T] (un solo mm sparso per blocco della rete); MLP sui soli
          vertici veri; la diffusione spettrale per gruppi di taglia simile (``spec_slack``) con bmm
          su evecs con padding. Stessi pesi, stessa matematica: lo verifica ``check_equivalence``;
  pk_ck   pk con checkpoint di attivazione per blocco (ricalcolo nel backward): meno memoria;
  pkc:n   pk a pacchetti di n mesh consecutive in ordine di taglia (un forward packed per pacchetto):
          i gruppi omogenei rendono di piu' a B 2-8 che a B grandi (job 1061636), qui sul passo misto.

Dati: operatori ad area unitaria letti dal loader congelato (``GTReadyDatasetNPZ``), gli stessi file
del run su scala: ICT original (9409 vertici), BFM original (23470), BFM up60k (~60.4k); FLAME up60k
(4523, il punto "~5k") calcolato qui con ``prepass_ops.ops_areanorm``. 12 mesh distinte per taglia,
ripetute a giro per i gruppi grandi. Scenario misto: 64 mesh per passo (16 identita' x 4 viste, il
passo per GPU di PLAN_MASSIVE.md sez. 7) con V continuo: ogni mesh e' una sorgente reale troncata ai
primi n vertici, n ~ U(0.7, 1.0) x V (supporto variabile); le righe di gradX/gradY con un estremo
tagliato si scartano. Va bene per i tempi, non per i numeri del modello.

Memoria: ``max_memory_allocated`` (picco dell'allocatore) e la memoria totale della GPU; il gruppo
massimo si cerca raddoppiando B fino all'OOM, poi per bisezione con un passo ciascuno.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
for _p in (REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic",
           REPO_ROOT / "face_embedding/gt_encdec/autoencoder",
           REPO_ROOT / "diffusion-net/src", REPO_ROOT / "v2_work/fastio",
           REPO_ROOT / "aau/data_scale", THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from batched import embed_samples, size_groups  # noqa: E402

DS = REPO_ROOT / "datasets"
ICT_OPS = DS / "ICT/topo_withops"
BFM_OPS = DS / "REMESH/npz_data_topo_500_withops_areanorm"
SIZES = {"S": (128, 4), "M": (256, 6), "L": (384, 8)}
KEEP = ("verts", "mass", "evals", "evecs", "gradX", "gradY", "L", "faces")


# --- dati -----------------------------------------------------------------------------------

def flame_ops(dest: Path, n: int) -> Path:
    """FLAME up60k (4523 vertici) con ops_areanorm, k 128. Nome id1NNN: dominio flame per prepass_ops."""
    import prepass_ops
    from grad_vec import install
    install()   # identico all'originale (grad_check.json), 400x piu' veloce
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        out = dest / f"id{1000 + i}_GTready_up60k.npz"
        if not out.exists():
            src = DS / f"FLAME/topo/flame{i:04d}_GTready_up60k.npz"
            tmp = dest.parent / f"id{1000 + i}_GTready_up60k.npz"
            tmp.write_bytes(src.read_bytes())
            prepass_ops.ops_areanorm(tmp, out)
            tmp.unlink()
    return dest


def link_dir(dest: Path, files: list[Path]) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        q = dest / f.name
        if not q.exists():
            q.symlink_to(f)
    return dest


def load_dir(d: Path, dev) -> list[dict]:
    from dataset_gtready import GTReadyDatasetNPZ
    ds = GTReadyDatasetNPZ(str(d))
    out = []
    for i in range(len(ds)):
        s = ds[i]
        out.append({k: (s[k].to(dev) if k in KEEP else s[k]) for k in KEEP} | {"name": ds.files[i]})
    return out


def truncate(s: dict, n: int) -> dict:
    """Primi n vertici: sotto-matrici di gradX/gradY/L, facce interne. Solo per i tempi."""
    def sub(m):
        idx, val = m.indices(), m.values()
        keep = (idx[0] < n) & (idx[1] < n)
        return torch.sparse_coo_tensor(idx[:, keep], val[keep], (n, n)).coalesce()
    f = s["faces"]
    return {"verts": s["verts"][:n].contiguous(), "mass": s["mass"][:n].contiguous(),
            "evals": s["evals"], "evecs": s["evecs"][:n].contiguous(), "gradX": sub(s["gradX"]),
            "gradY": sub(s["gradY"]), "L": sub(s["L"]), "faces": f[(f < n).all(dim=1)],
            "name": f"{s['name']}[:{n}]"}


# --- percorso packed (blocchi diagonali) ----------------------------------------------------

def collate_packed(samples, dev, spec_slack: float) -> dict:
    groups = size_groups(samples, spec_slack)          # indici ordinati per taglia
    order = [i for g in groups for i in g]
    ns = [int(samples[i]["verts"].shape[0]) for i in order]
    offs = np.concatenate([[0], np.cumsum(ns)])
    T, K = int(offs[-1]), int(samples[0]["evals"].numel())
    pk = {"order": torch.tensor(order, device=dev), "B": len(order), "T": T,
          "x": torch.cat([samples[i]["verts"] for i in order]),
          "mass": torch.cat([samples[i]["mass"] for i in order]),
          "seg": torch.repeat_interleave(torch.arange(len(order), device=dev),
                                         torch.tensor(ns, device=dev)),
          "count": torch.tensor(ns, device=dev, dtype=torch.float32)}
    for g in ("gradX", "gradY"):
        idx = torch.cat([samples[i][g].indices() + int(o) for i, o in zip(order, offs[:-1])], dim=1)
        val = torch.cat([samples[i][g].values() for i in order])
        pk[g] = torch.sparse_coo_tensor(idx, val, (T, T)).coalesce()
    buckets, pos = [], 0
    for g in groups:
        Bg, Ng = len(g), max(ns[pos:pos + len(g)])
        evecs = torch.zeros(Bg, Ng, K, device=dev)
        mass = torch.zeros(Bg, Ng, device=dev)
        pad = []
        for j in range(Bg):
            s, n = samples[order[pos + j]], ns[pos + j]
            evecs[j, :n] = s["evecs"]
            mass[j, :n] = s["mass"]
            pad.append(torch.arange(n, device=dev) + j * Ng)
        buckets.append({"a": int(offs[pos]), "b": int(offs[pos + Bg]), "B": Bg, "N": Ng,
                        "pad": torch.cat(pad), "evecs": evecs, "mass": mass,
                        "evals": torch.stack([samples[order[pos + j]]["evals"].flatten() for j in range(Bg)])})
        pos += Bg
    pk["buckets"] = buckets
    return pk


def _diffuse(blk, x, pk):
    """LearnedTimeDiffusion spettrale, per gruppo di taglia simile (bmm con padding)."""
    from diffusion_net.geometry import from_basis, to_basis
    d = blk.diffusion
    with torch.no_grad():
        d.diffusion_time.data = torch.clamp(d.diffusion_time, min=1e-8)
    outs = []
    for bk in pk["buckets"]:
        C = x.shape[-1]
        xb = x.new_zeros(bk["B"] * bk["N"], C).index_copy(0, bk["pad"], x[bk["a"]:bk["b"]])
        spec = to_basis(xb.view(bk["B"], bk["N"], C), bk["evecs"], bk["mass"])
        spec = torch.exp(-bk["evals"].unsqueeze(-1) * d.diffusion_time.unsqueeze(0)) * spec
        outs.append(from_basis(spec, bk["evecs"]).reshape(-1, C).index_select(0, bk["pad"]))
    return torch.cat(outs)


def _block(blk, x, pk):
    """DiffusionNetBlock.forward (layers.py) sul layout packed."""
    xd = _diffuse(blk, x, pk)
    gx = torch.sparse.mm(pk["gradX"], xd)
    gy = torch.sparse.mm(pk["gradY"], xd)
    gf = blk.gradient_features(torch.stack((gx, gy), dim=-1))
    return blk.mlp(torch.cat((x, xd, gf), dim=-1)) + x


def embed_packed(model, samples, dev, add_noise: bool, spec_slack: float = 0.25,
                 checkpoint: bool = False, pk: dict | None = None) -> torch.Tensor:
    from torch.utils.checkpoint import checkpoint as ckpt
    pk = pk or collate_packed(samples, dev, spec_slack)
    enc = model.encoder
    x = enc.first_lin(pk["x"])
    for blk in enc.blocks:
        x = ckpt(_block, blk, x, pk, use_reentrant=False) if checkpoint else _block(blk, x, pk)
    x = model.vertex_bottleneck(enc.last_lin(x))
    if add_noise:
        x = x + 0.01 * torch.randn_like(x)
    D = x.shape[-1]
    z_mean = x.new_zeros(pk["B"], D).index_add(0, pk["seg"], x) / pk["count"].unsqueeze(-1)
    z_max = torch.cat([x.new_full((bk["B"] * bk["N"], D), float("-inf"))
                       .index_copy(0, bk["pad"], x[bk["a"]:bk["b"]]).view(bk["B"], bk["N"], D).amax(1)
                       for bk in pk["buckets"]])
    z = model.pool_proj(torch.cat([z_mean, z_max], dim=-1)) if model.pool_mode == "meanmax" \
        else model.pool_proj(z_mean)
    return z.new_empty(z.shape).index_copy(0, pk["order"], z)


# --- passi ----------------------------------------------------------------------------------

def embed(path: str, model, samples, dev, add_noise: bool) -> torch.Tensor:
    from robustness.model_helpers import forward_model
    kind, _, arg = path.partition(":")
    if kind == "seq":
        return torch.stack([forward_model(model, s, s["verts"], return_gate_info=False,
                                          add_noise=add_noise)[0].reshape(-1) for s in samples])
    if kind == "bat":
        return embed_samples(model, samples, dev, add_noise=add_noise, pad_slack=float(arg))
    if kind in ("pk", "pk_ck"):
        return embed_packed(model, samples, dev, add_noise, spec_slack=float(arg), checkpoint=kind == "pk_ck")
    if kind == "pkc":
        n = int(arg)
        order = sorted(range(len(samples)), key=lambda i: int(samples[i]["verts"].shape[0]))
        z = torch.cat([embed_packed(model, [samples[i] for i in order[j:j + n]], dev, add_noise)
                       for j in range(0, len(order), n)])
        return z.new_empty(z.shape).index_copy(0, torch.tensor(order, device=dev), z)
    raise ValueError(path)


def make_step(path, model, opt, samples, dev):
    def step():
        opt.zero_grad(set_to_none=True)
        z = embed(path, model, samples, dev, add_noise=True)
        z.pow(2).mean().backward()
        opt.step()
    return step


def _free(opt):
    opt.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()


def measure(step, opt, n_warm: int = 1, n_iter: int = 3) -> dict:
    _free(opt)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(n_warm):
            step()
        ts = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        # cuSPARSE/cuBLAS a corto di memoria alzano RuntimeError, non OutOfMemoryError
        if not isinstance(exc, torch.cuda.OutOfMemoryError) and not any(
                m in str(exc) for m in ("out of memory", "ALLOC_FAILED", "CUBLAS_STATUS")):
            raise
        _free(opt)
        return {"oom": True}
    return {"oom": False, "t_step_s": statistics.median(ts) if ts else None, "t_steps_s": ts,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9, "base_gb": base / 1e9}


def fits(step, opt) -> bool:
    return not measure(step, opt, n_warm=1, n_iter=0)["oom"]


def sweep(path, model, opt, pool, dev, cap: int, deadline: float) -> dict:
    """B = 1, 2, 4, ... fino all'OOM (o cap), poi bisezione del B massimo."""
    rows, B, lo, hi = [], 1, 0, None
    while B <= cap and time.time() < deadline:
        r = measure(make_step(path, model, opt, [pool[i % len(pool)] for i in range(B)], dev), opt)
        if not r["oom"]:
            r["mesh_per_s"] = B / r["t_step_s"]
        rows.append({"B": B, **r})
        if r["oom"]:
            hi = B
            break
        print(f"    {path:<9} B={B:<4} {r['t_step_s'] * 1e3:8.1f} ms/passo {r['mesh_per_s']:7.1f} mesh/s "
              f"picco {r['peak_gb']:.1f} GB", flush=True)
        lo, B = B, B * 2
    while hi is not None and hi - lo > max(1, lo // 16) and time.time() < deadline:
        mid = (lo + hi) // 2
        if fits(make_step(path, model, opt, [pool[i % len(pool)] for i in range(mid)], dev), opt):
            lo = mid
        else:
            hi = mid
    ok = [r for r in rows if not r["oom"]]
    best = max(ok, key=lambda r: r["mesh_per_s"]) if ok else None
    return {"path": path, "rows": rows, "B_max": lo, "B_oom": hi, "capped": hi is None,
            "best_mesh_per_s": best["mesh_per_s"] if best else None, "best_B": best["B"] if best else None}


# --- equivalenza ----------------------------------------------------------------------------

def check_equivalence(model, samples, dev) -> dict:
    """eval (niente dropout ne' rumore): embedding e gradienti dei pesi, seq contro bat e pk."""
    model.eval()
    out, grads = {}, {}
    for path in ("seq", "bat:inf", "pk:0.25", "pk:0", "pk_ck:0.25", "pkc:2"):
        model.zero_grad(set_to_none=True)
        z = embed(path, model, samples, dev, add_noise=False)
        z.pow(2).sum().backward()
        out[path] = z.detach()
        grads[path] = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    model.zero_grad(set_to_none=True)
    model.train()
    ref, gref = out["seq"], grads["seq"]
    return {p: {"max_abs_z": float((out[p] - ref).abs().max()), "z_scale": float(ref.abs().max()),
                "max_abs_grad": float((grads[p] - gref).abs().max()), "grad_scale": float(gref.abs().max())}
            for p in out if p != "seq"}


# --- main -----------------------------------------------------------------------------------

def mixed(sources: dict, n: int, rng: random.Random) -> list[dict]:
    cats = sorted(sources)
    out = []
    for _ in range(n):
        s = rng.choice(sources[rng.choice(cats)])
        out.append(truncate(s, int(round(rng.uniform(0.7, 1.0) * s["verts"].shape[0]))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sizes", default="S,M,L")
    ap.add_argument("--n-distinct", type=int, default=12)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--deadline-min", type=float, default=52.0)
    ap.add_argument("--mix-paths", default="seq,bat:0.05,bat:0.25,bat:inf,pk:0.25,pk_ck:0.25",
                    help="percorsi del passo misto, separati da virgola")
    ap.add_argument("--skip-homo", action="store_true", help="solo equivalenza e passo misto")
    ap.add_argument("--tag", default="", help="suffisso dei JSON")
    a = ap.parse_args()
    deadline = time.time() + 60 * a.deadline_min
    a.out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    from diffusion_autoencoder import DiffusionEncoderOnly

    env = {"gpu": torch.cuda.get_device_name(0), "mem_total_gb": torch.cuda.mem_get_info()[1] / 1e9,
           "torch": torch.__version__, "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
           "float32_matmul_precision": torch.get_float32_matmul_precision(),
           "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
           "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID")}
    print(env, flush=True)

    t0 = time.time()
    nd = a.n_distinct
    W = a.work_dir
    ict = lambda lab, n: [ICT_OPS / f"ict{50 * j:04d}_GTready_{lab}.npz" for j in range(n)]  # noqa: E731
    bfm = lambda lab, n: [BFM_OPS / f"id{5 * j:04d}_GTready_{lab}.npz" for j in range(n)]  # noqa: E731
    pools = {"4.5k_flame_up60k": load_dir(flame_ops(W / "flame", nd), dev),
             "9.4k_ict_original": load_dir(link_dir(W / "ict_original", ict("original", nd)), dev),
             "23.5k_bfm_original": load_dir(link_dir(W / "bfm_original", bfm("original", nd)), dev),
             "60.4k_bfm_up60k": load_dir(link_dir(W / "bfm_up60k", bfm("up60k", nd)), dev)}
    extra = {"ict_remesh": ict("remesh", 8), "ict_down8k": ict("down8k", 8), "bfm_down8k": bfm("down8k", 8),
             "bfm_remesh": bfm("remesh", 8), "ict_up60k": ict("up60k", 8)}
    src = {k: load_dir(link_dir(W / k, v), dev) for k, v in extra.items()}
    src.update({"flame_up60k": pools["4.5k_flame_up60k"][:8], "ict_original": pools["9.4k_ict_original"][:8],
                "bfm_original": pools["23.5k_bfm_original"][:8], "bfm_up60k": pools["60.4k_bfm_up60k"][:8]})
    mixes = {"le10k": {k: src[k] for k in ("flame_up60k", "ict_down8k", "ict_remesh", "bfm_down8k", "ict_original")},
             "full": {k: src[k] for k in ("ict_down8k", "ict_remesh", "ict_original", "ict_up60k",
                                          "bfm_down8k", "bfm_remesh", "bfm_original", "bfm_up60k")}}
    env["load_s"] = time.time() - t0
    env["pools"] = {k: [int(s["verts"].shape[0]) for s in v] for k, v in pools.items()}
    (a.out_dir / "env.json").write_text(json.dumps(env, indent=1))
    print(f"[fwd] dati pronti in {env['load_s']:.0f}s: { {k: v[0] for k, v in env['pools'].items()} }", flush=True)

    # equivalenza dei percorsi, modello S, 6 mesh di taglie diverse (anche padding e piu' gruppi)
    rng = random.Random(0)
    eq_samples = mixed(mixes["le10k"], 6, rng)
    model = DiffusionEncoderOnly(latent_dim=256, width=128, n_blocks=4, dropout=0.1, pool_mode="meanmax").to(dev)
    eq = check_equivalence(model, eq_samples, dev)
    print(f"[fwd] equivalenza: {json.dumps(eq)}", flush=True)
    (a.out_dir / f"equivalence{a.tag}.json").write_text(json.dumps(
        {"n": [int(s["verts"].shape[0]) for s in eq_samples], "vs_seq": eq}, indent=1))
    del model

    for size in a.sizes.split(","):
        width, nb = SIZES[size]
        torch.manual_seed(0)
        model = DiffusionEncoderOnly(latent_dim=256, width=width, n_blocks=nb, dropout=0.1,
                                     pool_mode="meanmax").to(dev).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"[fwd] taglia {size}: width {width}, blocchi {nb}, {n_par / 1e6:.2f}M parametri", flush=True)
        # misto: 64 mesh per passo, tutti i percorsi (se 64 non entra: 32, 16)
        for mix, srcs in mixes.items():
            rng = random.Random(1)
            step_samples = mixed(srcs, 64, rng)
            res = {"size": size, "mix": mix, "params": n_par,
                   "V": [int(s["verts"].shape[0]) for s in step_samples], "paths": {}}
            for path in a.mix_paths.split(","):
                if time.time() > deadline:
                    break
                for B in (64, 32, 16):
                    r = measure(make_step(path, model, opt, step_samples[:B], dev), opt)
                    if not r["oom"]:
                        break
                r["B"] = B
                if not r["oom"]:
                    r["mesh_per_s"] = B / r["t_step_s"]
                    if path.startswith("bat"):
                        g = size_groups(step_samples[:B], float(path.split(":")[1]))
                        n = [int(s["verts"].shape[0]) for s in step_samples[:B]]
                        r["n_groups"] = len(g)
                        r["pad_rows_ratio"] = sum(len(x) * max(n[i] for i in x) for x in g) / sum(n)
                res["paths"][path] = r
                print(f"  [{size} {mix}] {path:<10} B={B:<3} "
                      + ("OOM" if r["oom"] else f"{r['mesh_per_s']:7.1f} mesh/s picco {r['peak_gb']:.1f} GB"
                         + (f" gruppi {r.get('n_groups')} padding x{r.get('pad_rows_ratio', 1):.2f}" if 'n_groups' in r else "")),
                      flush=True)
            (a.out_dir / f"mix_{size}_{mix}{a.tag}.json").write_text(json.dumps(res))
        # omogeneo: per taglia di mesh, B fino all'OOM per bat, pk, pk_ck; seq a B = min(30, B_max)
        for name, pool in ({} if a.skip_homo else pools).items():
            if time.time() > deadline:
                print(f"[fwd] scadenza: salto {size} {name}", flush=True)
                continue
            res = {"size": size, "pool": name, "V": int(pool[0]["verts"].shape[0]), "params": n_par, "sweeps": {}}
            print(f"  [{size} {name}]", flush=True)
            for path in ("bat:0.05", "pk:0.25", "pk_ck:0.25"):
                res["sweeps"][path] = sweep(path, model, opt, pool, dev, a.cap, deadline)
            bmax = res["sweeps"]["bat:0.05"]["B_max"]
            if bmax and time.time() < deadline:
                B = min(30, bmax)
                r = measure(make_step("seq", model, opt, [pool[i % len(pool)] for i in range(B)], dev), opt)
                r["B"] = B
                if not r["oom"]:
                    r["mesh_per_s"] = B / r["t_step_s"]
                    print(f"    seq       B={B:<4} {r['t_step_s'] * 1e3:8.1f} ms/passo {r['mesh_per_s']:7.1f} mesh/s",
                          flush=True)
                res["seq"] = r
            (a.out_dir / f"homo_{size}_{name}.json").write_text(json.dumps(res))
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
    print("[fwd] OK", flush=True)


if __name__ == "__main__":
    main()
