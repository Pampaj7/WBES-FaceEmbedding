#!/usr/bin/env python3
"""Studente distillato v2: lo stesso trainer del pilota con ingressi molto piu' vari.

    aau/run.sh aau/distill/train_distill_v2.py --sources bfm,ict5k,ictscale,gnm --stage /tmp/$SLURM_JOB_ID \\
        --run-dir <run dir> --device cuda --n-proc 60
    (train_distill_v2.sbatch; protocollo in aau/runs/distill_v2/protocol.md)

Uguale al pilota (``train_distill.py``, importato: ``distill_losses``, ``l2``, ``NAME_RE``): modello
``xyz_dn`` a 512 dimensioni con normalizzazione L2, loss (1 - coseno) + relazionale, Adam, batch di
``--batch-subjects`` soggetti a caso (domini mescolati) x ``--meshes-per-subject`` mesh, rumore xyz e
latente del congiunto. Cambia:

1. **Dati** (``distill_data.py``): BFM e ICT-5000 del pilota piu' 10.000 ICT nuovi e 10.000 GNM Head,
   ciascuno con le sue etichette dell'insegnante (``--teacher-root``: ``bfm``/``ict`` del pilota,
   ``ictscale``/``gnm`` di v2).
2. **Frame canonico ICT** (+Y alto, naso +Z): BFM con Rx(180) e facce invertite PRIMA degli operatori
   (``prepass_ops.apply_frame``), che quindi si ricalcolano; ICT, ICT nuovi e GNM sono gia' li'.
3. **Operatori nel job**: per BFM, ICT nuovi e GNM si calcolano qui (``prepass_ops.ops_areanorm``:
   area unitaria, k_eig 128, la convenzione delle viste in uso) da ``--n-proc`` processi; per ICT-5000
   si leggono le viste esistenti. Ogni mesh passa dal loader congelato (``GTReadyDatasetNPZ``,
   centratura + maxabs, scala spettrale) e va in una cache RAM COMPATTA: ``evecs`` in float16 (la
   variante "e16" di aau/data_scale/PLAN.md, misurata innocua per dati nuovi di training), facce e
   indici in int32, gli indici di L/gradX/gradY tenuti una volta sola quando coincidono. Si ricostruisce
   il campione del loader sulla GPU a ogni lettura. In validazione, eval comprese, nessuna compressione:
   ``embed_student.py`` usa il loader congelato.
4. **Augmentation** dello studente (l'insegnante vede sempre la mesh NON deformata, vedi il protocollo):
   - deformazione liscia a bassa frequenza con probabilita' ``--p-deform``: campo di
     ``--deform-modes`` onde piane ``a_k sin(2 pi f_k <u_k, x> + phi_k)`` (direzioni e ampiezze
     vettoriali a caso, ``f_k`` in [0.5, 1.5] cicli per unita' maxabs, cioe' 1-3 oscillazioni sul
     volto), riscalato a uno spostamento RMS per vertice ``U(0, --deform-rms)``;
   - il rumore xyz del congiunto, invariato;
   - rotazione rigida attorno al centro, asse uniforme e angolo U(0, ``--rot-deg``) (30 gradi).
   Tutte toccano solo il canale xyz: gli operatori restano quelli della mesh (come il rumore della ricetta).
5. **Passi fissi** (``--steps``, ``--steps-per-epoch`` per log e validazione): i soggetti scorrono per
   permutazioni successive, cosi' bracci con piu' o meno dati fanno gli stessi passi.

Validazione (solo curve; checkpoint valutato = l'ULTIMO): soggetti ``val`` di ogni sorgente, coseno con
l'insegnante e rank-1 cross-topologia, complessivo e per sorgente.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(AAU_DIR / "data_scale"))

import distill_data as dd  # noqa: E402
import train_distill as td  # noqa: E402  (path di robustness, loader, fast_data)

from robustness.model_helpers import build_model, forward_model  # noqa: E402
from robustness.noise import (  # noqa: E402
    PerturbationParams,
    apply_xyz_perturbation_with_params,
    parse_noise_mode_weights,
    parse_noise_modes,
    sample_log_uniform_sigma,
)

SPARSE = ("L", "gradX", "gradY")
TEACHER_DOMAIN = {"bfm": "bfm", "ict5k": "ict", "ictscale": "ictscale", "gnm": "gnm"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default="bfm,ict5k,ictscale,gnm")
    p.add_argument("--teacher-root", type=Path, default=AAU_DIR / "runs" / "distill_v2" / "teacher")
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--limit-subjects", type=int, default=0, help=">0: N soggetti di training per sorgente (smoke)")
    p.add_argument("--n-proc", type=int, default=32, help="processi per gli operatori e la cache")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--batch-subjects", type=int, default=16)
    p.add_argument("--meshes-per-subject", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-halve-at", type=float, default=0.75)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-rel", type=float, default=1.0)
    p.add_argument("--latent-dim", type=int, default=512)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pool-mode", default="meanmax")
    p.add_argument("--no-latent-noise", action="store_true")
    # augmentation del congiunto
    p.add_argument("--p-noise", type=float, default=0.6)
    p.add_argument("--sigma-min", type=float, default=5e-4)
    p.add_argument("--sigma-max", type=float, default=2e-2)
    p.add_argument("--noise-modes", default="translation,rotation,jitter")
    p.add_argument("--noise-mode-weights", default="translation=4,rotation=2,jitter=1")
    p.add_argument("--rigid-rot-deg", type=float, default=12.0)
    p.add_argument("--rigid-rot-deg-min", type=float, default=0.5)
    p.add_argument("--rigid-trans-scale", type=float, default=0.03)
    p.add_argument("--rigid-trans-scale-min", type=float, default=0.001)
    p.add_argument("--outlier-frac", type=float, default=0.02)
    p.add_argument("--outlier-scale", type=float, default=6.0)
    # augmentation v2
    p.add_argument("--rot-deg", type=float, default=30.0)
    p.add_argument("--p-deform", type=float, default=0.5)
    p.add_argument("--deform-rms", type=float, default=0.005,
                   help="massimo dello spostamento RMS (unita' maxabs); il vicino d'identita' piu' prossimo "
                        "di ICT sta a ~0.017 (mediana, gen_ict_shard)")
    p.add_argument("--deform-modes", type=int, default=4)
    p.add_argument("--val-every", type=int, default=2, help="in epoche")
    p.add_argument("--save-every", type=int, default=10, help="in epoche")
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args(argv)


# ------------------------------------------------------------------ cache compatta

def load_frozen(path: Path) -> dict:
    """Un campione dal loader congelato, senza listare la directory."""
    from dataset_gtready import GTReadyDatasetNPZ

    ds = GTReadyDatasetNPZ.__new__(GTReadyDatasetNPZ)
    ds.data_dir, ds.files, ds.verbose = str(path.parent), [path.name], False
    return ds[0]


def compact(s: dict) -> dict:
    out = {"verts": s["verts"].numpy().astype(np.float32), "faces": s["faces"].numpy().astype(np.int32),
           "mass": s["mass"].numpy().astype(np.float32), "evals": s["evals"].numpy().astype(np.float32),
           "evecs": s["evecs"].numpy().astype(np.float16)}
    idx = []
    for k in SPARSE:
        c = s[k].coalesce()
        ind = c.indices().numpy().astype(np.int32)
        ref = next((j for j, x in enumerate(idx) if x.shape == ind.shape and np.array_equal(x, ind)), None)
        if ref is None:
            idx.append(ind)
            ref = len(idx) - 1
        out[f"{k}_ref"] = ref
        out[f"{k}_values"] = c.values().numpy().astype(np.float32)
        out[f"{k}_shape"] = tuple(int(x) for x in c.shape)
    out["indices"] = idx
    return out


def compact_bytes(c: dict) -> int:
    return sum(v.nbytes for v in c.values() if isinstance(v, np.ndarray)) + sum(x.nbytes for x in c["indices"])


def _prepare(task):
    """Worker: (i, tipo, path, tmp_dir) -> (i, campione compatto, secondi, errore)."""
    i, kind, path, tmp_dir = task
    t0 = time.time()
    try:
        path = Path(path)
        if kind == "ops":
            return i, compact(load_frozen(path)), time.time() - t0, ""
        from prepass_ops import ops_areanorm

        out = Path(tmp_dir) / path.name
        ops_areanorm(path, out, dd.BFM_CANON)   # trasformazione solo per gli id BFM (domain_of_name)
        try:
            return i, compact(load_frozen(out)), time.time() - t0, ""
        finally:
            out.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        return i, None, time.time() - t0, f"{type(exc).__name__}: {exc}"


def to_device(c: dict, device) -> dict:
    """Il campione del loader congelato, sulla GPU, dalla forma compatta."""
    out = {"verts": torch.from_numpy(c["verts"]).to(device),
           "faces": torch.from_numpy(c["faces"]).to(device).long(),
           "mass": torch.from_numpy(c["mass"]).to(device),
           "evals": torch.from_numpy(c["evals"]).to(device),
           "evecs": torch.from_numpy(c["evecs"]).to(device).float()}
    idx = [torch.from_numpy(x).to(device).long() for x in c["indices"]]
    for k in SPARSE:
        out[k] = torch.sparse_coo_tensor(idx[c[f"{k}_ref"]], torch.from_numpy(c[f"{k}_values"]).to(device),
                                         c[f"{k}_shape"], is_coalesced=True)
    return out


# ------------------------------------------------------------------ augmentation

def random_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    th = math.radians(rng.uniform(0.0, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def smooth_deform(V: torch.Tensor, rng: np.random.Generator, n_modes: int, rms: float) -> torch.Tensor:
    """Somma di onde piane a bassa frequenza, riscalata a spostamento RMS ``rms``."""
    dirs = rng.normal(size=(n_modes, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    freq = rng.uniform(0.5, 1.5, size=n_modes)
    phase = rng.uniform(0, 2 * np.pi, size=n_modes)
    amp = rng.normal(size=(n_modes, 3))
    W = torch.as_tensor(dirs * freq[:, None] * 2 * np.pi, dtype=V.dtype, device=V.device)
    D = torch.sin(V @ W.T + torch.as_tensor(phase, dtype=V.dtype, device=V.device)) @ \
        torch.as_tensor(amp, dtype=V.dtype, device=V.device)
    D = D - D.mean(0, keepdim=True)
    return V + D * (rms / D.pow(2).sum(1).mean().sqrt().clamp_min(1e-12))


# ------------------------------------------------------------------ validazione

@torch.no_grad()
def validate(model, cache, val_rows, targets, device) -> dict:
    model.eval()
    Z = []
    for i, _ in val_rows:
        sd = to_device(cache[i], device)
        z, _ = forward_model(model=model, sample_dict=sd, V_in=sd["verts"], return_gate_info=False, add_noise=False)
        Z.append(td.l2(z.squeeze(0).float()))
    Z = torch.stack(Z)
    T = torch.stack([targets[(r["sid"], r["key"])] for _, r in val_rows]).to(device)
    cos = (Z * T).sum(1)
    pos = {(r["sid"], r["label"]): n for n, (_, r) in enumerate(val_rows)}
    out = {"val_cos": float(cos.mean())}
    by_src: dict[str, list[str]] = {}
    for _, r in val_rows:
        by_src.setdefault(r["source"], [])
        if r["sid"] not in by_src[r["source"]]:
            by_src[r["source"]].append(r["sid"])
    tot_h, tot_q = 0, 0
    for src, subjects in sorted(by_src.items()):
        hits, n_q = 0, 0
        for ta in dd.NOCROP:
            for tb in dd.NOCROP:
                if ta == tb:
                    continue
                ok = [s for s in subjects if (s, ta) in pos and (s, tb) in pos]
                if len(ok) < 2:
                    continue
                pred = (Z[[pos[(s, ta)] for s in ok]] @ Z[[pos[(s, tb)] for s in ok]].T).argmax(1).cpu().numpy()
                hits += int((pred == np.arange(len(ok))).sum())
                n_q += len(ok)
        out[f"val_rank1_{src}"] = hits / max(n_q, 1)
        tot_h, tot_q = tot_h + hits, tot_q + n_q
    out["val_rank1"] = tot_h / max(tot_q, 1)
    model.train()
    return out


# ------------------------------------------------------------------ main

def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sources = [s for s in args.sources.split(",") if s]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    rows = dd.plan(sources)
    if args.limit_subjects > 0:
        keep = set()
        for src in sources:
            tr = sorted({r["sid"] for r in rows if r["source"] == src and r["split"] == "train"})
            va = sorted({r["sid"] for r in rows if r["source"] == src and r["split"] == "val"})
            keep |= set(tr[: args.limit_subjects]) | set(va[: max(4, args.limit_subjects // 4)])
        rows = [r for r in rows if r["sid"] in keep]

    targets = {}
    for src in sources:
        z = np.load(args.teacher_root / TEACHER_DOMAIN[src] / "teacher.npz")
        for s, t, row in zip(z["subjects"], z["topologies"], z["Z"]):
            targets[(str(s), str(t))] = torch.from_numpy(row.astype(np.float32))
    missing = [(r["sid"], r["key"]) for r in rows if (r["sid"], r["key"]) not in targets]
    if missing:
        raise SystemExit(f"{len(missing)} bersagli dell'insegnante mancanti, es. {missing[:3]}")

    # Geometria degli shard su /tmp, poi operatori e cache.
    t0 = time.time()
    geom = {}
    for src in ("ictscale", "gnm"):
        want = [r["origin"][1] for r in rows if r["source"] == src]
        if want:
            geom.update(dd.extract(src, want, args.stage / "geom" / src))
    tmp_ops = args.stage / "ops_tmp"
    tmp_ops.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i, r in enumerate(rows):
        kind, ref = r["origin"]
        path = geom[ref] if kind == "tar" else Path(ref)
        tasks.append((i, "ops" if kind == "ops" else "geom", str(path), str(tmp_ops)))
    tasks.sort(key=lambda t: -Path(t[2]).stat().st_size)
    counts = {s: sum(r["source"] == s for r in rows) for s in sources}
    print(f"[v2] {len(rows)} mesh {counts}, "
          f"geometria estratta in {time.time() - t0:.0f}s; operatori/cache con {args.n_proc} processi", flush=True)
    cache: list = [None] * len(rows)
    nbytes, cpu = 0, {"ops": 0.0, "geom": 0.0}
    t1 = time.time()
    fails = []
    with mp.get_context("spawn").Pool(args.n_proc) as pool:
        for n, (i, c, sec, err) in enumerate(pool.imap_unordered(_prepare, tasks, chunksize=4), start=1):
            if err:
                fails.append(f"{rows[i]['sid']}/{rows[i]['label']}: {err}")
                continue
            cache[i] = c
            nbytes += compact_bytes(c)
            cpu["ops" if rows[i]["origin"][0] == "ops" else "geom"] += sec
            if n % 2000 == 0 or n == len(tasks):
                rate = n / (time.time() - t1)
                print(f"[v2] cache {n}/{len(tasks)} ({rate:.1f} mesh/s, {nbytes / 1024 ** 3:.1f} GiB, "
                      f"eta {(len(tasks) - n) / max(rate, 1e-9) / 60:.0f} min)", flush=True)
    if fails:
        raise SystemExit(f"{len(fails)} mesh fallite: {fails[:5]}")
    import shutil
    shutil.rmtree(args.stage / "geom", ignore_errors=True)
    n_geom = sum(r["origin"][0] != "ops" for r in rows)
    prep = {"n_meshes": len(rows), "n_ops_computed": n_geom, "wall_s": time.time() - t1,
            "cpu_s_per_computed_mesh": cpu["geom"] / max(n_geom, 1),
            "cpu_s_per_loaded_mesh": cpu["ops"] / max(len(rows) - n_geom, 1),
            "cache_gib": nbytes / 1024 ** 3,
            "cache_mib_per_mesh": {s: float(np.mean([compact_bytes(cache[i]) for i, r in enumerate(rows)
                                                     if r["source"] == s])) / 1024 ** 2 for s in sources}}
    (args.run_dir / "prep.json").write_text(json.dumps(prep, indent=2) + "\n")
    print(f"[v2] cache pronta: {json.dumps(prep)}", flush=True)

    by_subject: dict[str, list[int]] = {}
    val_rows = []
    for i, r in enumerate(rows):
        if r["split"] == "val":
            val_rows.append((i, r))
        else:
            by_subject.setdefault(r["sid"], []).append(i)
    subjects = np.array(sorted(by_subject), dtype=object)
    print(f"[v2] training: {len(subjects)} soggetti, {sum(map(len, by_subject.values()))} mesh; "
          f"validazione: {len({r['sid'] for _, r in val_rows})} soggetti, {len(val_rows)} mesh", flush=True)

    margs = argparse.Namespace(model="xyz_dn", latent_dim=args.latent_dim, width=args.width,
                               n_blocks=args.n_blocks, dropout=args.dropout, pool_mode=args.pool_mode)
    model = build_model(margs, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    noise_modes = parse_noise_modes(args.noise_modes)
    noise_probs = np.asarray(parse_noise_mode_weights(args.noise_mode_weights, noise_modes), dtype=np.float64)
    perturbation = PerturbationParams.from_namespace(args)
    halve_step = int(round(args.lr_halve_at * args.steps))
    n_epochs = math.ceil(args.steps / args.steps_per_epoch)

    ckpt_dir = args.run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    train_log = open(args.run_dir / "train_log.csv", "w", newline="")
    tw = csv.writer(train_log)
    tw.writerow(["epoch", "step", "lr", "loss", "point", "rel", "train_cos", "s_per_step", "elapsed_s"])
    val_log = open(args.run_dir / "val_log.csv", "w", newline="")
    vw = None
    rng = np.random.default_rng(args.seed + 977)
    queue: list = []
    step = 0
    t_start = time.time()
    model.train()
    for epoch in range(1, n_epochs + 1):
        sums, n_steps, t_ep = np.zeros(4), 0, time.time()
        while n_steps < args.steps_per_epoch and step < args.steps:
            if step == halve_step:
                for g in optimizer.param_groups:
                    g["lr"] = args.lr / 2
            while len(queue) < args.batch_subjects:
                queue.extend(rng.permutation(subjects).tolist())
            batch, queue = queue[: args.batch_subjects], queue[args.batch_subjects:]
            do_noise = bool(rng.uniform() < args.p_noise)
            sigma = sample_log_uniform_sigma(args.sigma_min, args.sigma_max, rng) if do_noise else 0.0
            optimizer.zero_grad(set_to_none=True)
            zs, keys = [], []
            for s in batch:
                cand = by_subject[s]
                for j in rng.choice(len(cand), size=min(args.meshes_per_subject, len(cand)), replace=False):
                    i = cand[int(j)]
                    sd = to_device(cache[i], device)
                    V_in = sd["verts"]
                    if rng.uniform() < args.p_deform:
                        V_in = smooth_deform(V_in, rng, args.deform_modes, rng.uniform(0.0, args.deform_rms))
                    if sigma > 0.0:
                        mode = noise_modes[int(rng.choice(len(noise_modes), p=noise_probs))]
                        V_in = apply_xyz_perturbation_with_params(V=V_in, mode=mode, sigma=sigma, params=perturbation)
                    if args.rot_deg > 0:
                        R = torch.as_tensor(random_rotation(rng, args.rot_deg), dtype=V_in.dtype, device=device)
                        V_in = V_in @ R.T
                    z, _ = forward_model(model=model, sample_dict=sd, V_in=V_in, return_gate_info=False,
                                         add_noise=not args.no_latent_noise)
                    zs.append(z.squeeze(0))
                    keys.append((rows[i]["sid"], rows[i]["key"]))
            S = td.l2(torch.stack(zs).float())
            T = torch.stack([targets[k] for k in keys]).to(device)
            point, rel = td.distill_losses(S, T)
            loss = point + args.lambda_rel * rel
            if not torch.isfinite(loss):
                raise SystemExit(f"loss non finita al passo {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            n_steps += 1
            sums += [float(loss), float(point), float(rel), float((S * T).sum(1).mean())]
            if n_steps % 200 == 0:
                print(f"[v2] passo {step}: loss {sums[0] / n_steps:.4f} ({(time.time() - t_ep) / n_steps:.2f} s/passo)",
                      flush=True)
        m = sums / max(n_steps, 1)
        sps = (time.time() - t_ep) / max(n_steps, 1)
        lr = optimizer.param_groups[0]["lr"]
        tw.writerow([epoch, step, lr, *[f"{x:.6f}" for x in m], f"{sps:.3f}", f"{time.time() - t_start:.0f}"])
        train_log.flush()
        line = (f"[v2] epoca {epoch}/{n_epochs} passo {step} lr {lr:.1e}: loss {m[0]:.4f} punt {m[1]:.4f} "
                f"rel {m[2]:.5f} cos {m[3]:.4f} | {sps:.2f} s/passo, {time.time() - t_start:.0f}s")
        if epoch % args.val_every == 0 or epoch == n_epochs:
            v = validate(model, cache, val_rows, targets, device)
            if vw is None:
                vw = csv.DictWriter(val_log, fieldnames=["epoch", "step", *v])
                vw.writeheader()
            vw.writerow({"epoch": epoch, "step": step, **v})
            val_log.flush()
            line += " | val " + " ".join(f"{k[4:]} {x:.3f}" for k, x in v.items())
        print(line, flush=True)
        bundle = {"state_dict": model.state_dict(), "args": vars(margs), "epoch": epoch, "step": step}
        torch.save(bundle, ckpt_dir / "last.pth")
        if epoch % args.save_every == 0:
            torch.save(bundle, ckpt_dir / f"epoch{epoch:03d}.pth")
    train_log.close()
    val_log.close()
    if torch.cuda.is_available():
        print(f"[v2] picco memoria GPU {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GiB", flush=True)
    print(f"[v2] fine in {time.time() - t_start:.0f}s -> {ckpt_dir / 'last.pth'}", flush=True)


if __name__ == "__main__":
    main()
