#!/usr/bin/env python3
"""Studente DiffusionNet distillato dall'insegnante ArcFace-render: trainer snello del pilota.

    aau/run.sh aau/distill/train_distill.py --teacher-dirs aau/runs/distill_pilot/teacher/bfm \\
        aau/runs/distill_pilot/teacher/ict --stage /tmp/$SLURM_JOB_ID --run-dir <run dir> --device cuda
    (train_distill.sbatch; piano in paper/PLAN_DISTILL.md, protocollo in aau/runs/distill_pilot/protocol.md)

Accanto a ``train_runner`` / ``train_fast``, che non tocca: ne importa i pezzi e riscrive solo il
ciclo, perche' la loss del v1 (stress e ranking su una GT di distanze 3DMM a dominio singolo) qui
non c'e'. Importati, non riscritti:

- dati: ``v2_work/fastio/fast_data.CachedDataset`` (loader congelato ``GTReadyDatasetNPZ``,
  vertici centrati e maxabs, operatori ad area unitaria k_eig 128 delle viste in uso, tutto in RAM,
  non pinned come in train_steps.py);
- modello: ``robustness.model_helpers.build_model`` / ``forward_model``, ``xyz_dn`` con la ricetta
  del congiunto (width 128, 4 blocchi, dropout 0.1, pool meanmax, rumore latente in training) ma
  ``latent_dim`` 512; l'uscita e' normalizzata L2 QUI, fuori dal modello;
- augmentation: ``robustness.noise`` con i parametri del congiunto 1019532 (p 0.6, sigma
  log-uniforme in [5e-4, 2e-2], traslazione/rotazione/jitter 4:2:1, rotazione rigida <= 12 gradi).

Dati: la vista di symlink in ``--stage`` contiene, per ogni soggetto di ``teacher.npz`` (training e
validazione), le topologie ``--topologies`` (default le 5 senza crop) dalle viste con operatori del
congiunto (BFM ``REMESH/npz_data_topo_500_withops_areanorm``, ICT ``ICT/train_ready/npz_withops``)
e le espressioni ``rexpr<k>`` (``ICT/expressions_random_withops``) dove l'insegnante le ha. Frame
nativi di ogni dominio, come il congiunto.

Bersaglio di una mesh: l'embedding dell'insegnante della ``original`` dello stesso soggetto (della
``rexpr<k>`` stessa per un'espressione, che ha la topologia della original). Loss per passo, su n
mesh (``--batch-subjects`` soggetti a caso, domini mescolati, ``--meshes-per-subject`` mesh distinte
ciascuno):

    puntuale    mean_i (1 - <s_i, t_i>)
    relazionale mean_{i != j} (<s_i, s_j> - <t_i, t_j>)^2,  peso --lambda-rel

Validazione a ogni ``--val-every`` epoche sui soggetti ``split == val`` (held-out del congiunto, mai
nel training): coseno medio con l'insegnante e rank-1 cross-topologia (query in una topologia,
galleria dei soggetti di validazione in un'altra, coppie ordinate delle topologie di
``--topologies``). Solo curve: il checkpoint valutato e' l'ULTIMO, fissato prima.

Scrive in ``--run-dir``: ``args.json``, ``train_log.csv`` (per epoca), ``val_log.csv``,
``checkpoints/last.pth`` (``state_dict`` + ``args``) e ``epochNNN.pth`` ogni ``--save-every``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"))
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "fastio"))

import fast_data as fd  # noqa: E402
from robustness.data_utils import sample_to_device  # noqa: E402
from robustness.model_helpers import build_model, forward_model  # noqa: E402
from robustness.noise import (  # noqa: E402
    PerturbationParams,
    apply_xyz_perturbation_with_params,
    parse_noise_mode_weights,
    parse_noise_modes,
    sample_log_uniform_sigma,
)

DS = REPO_ROOT / "datasets"
NAME_RE = re.compile(r"^(id\d+)_GTready_(\w+)$")
NOCROP = ("original", "remesh", "down8k", "up60k", "noisy")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher-dirs", type=Path, nargs="+", required=True, help="dir con teacher.npz")
    p.add_argument("--stage", type=Path, required=True, help="su /tmp: vista di symlink")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--topologies", default=",".join(NOCROP))
    p.add_argument("--no-expr", action="store_true", help="esclude le rexpr<k> dall'ingresso")
    p.add_argument("--limit-subjects", type=int, default=0, help=">0: primi N soggetti di training (smoke)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-subjects", type=int, default=16)
    p.add_argument("--meshes-per-subject", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-halve-at", type=float, default=0.75, help="frazione delle epoche dopo cui lr / 2")
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-rel", type=float, default=1.0)
    # modello (ricetta del congiunto, tranne latent_dim)
    p.add_argument("--model", default="xyz_dn")
    p.add_argument("--latent-dim", type=int, default=512)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pool-mode", default="meanmax")
    p.add_argument("--no-latent-noise", action="store_true")
    # augmentation xyz (ricetta del congiunto 1019532)
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
    # esecuzione
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--cache-workers", type=int, default=16)
    p.add_argument("--cache-max-gb", type=float, default=600.0)
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args(argv)


def operator_file(subject: str, topology: str) -> Path:
    """File con operatori del congiunto per (soggetto, topologia)."""
    if topology.startswith("rexpr"):
        return DS / "ICT" / "expressions_random_withops" / f"{subject}_rexpr_{topology[5:]}.npz"
    if int(subject[2:]) < 1000:
        return DS / "REMESH" / "npz_data_topo_500_withops_areanorm" / f"{subject}_GTready_{topology}.npz"
    return DS / "ICT" / "train_ready" / "npz_withops" / f"{subject}_GTready_{topology}.npz"


def load_teacher(dirs) -> tuple[dict, dict]:
    """(soggetto, topologia dell'insegnante) -> Z; soggetto -> split."""
    targets, split = {}, {}
    for d in dirs:
        z = np.load(Path(d) / "teacher.npz")
        for s, t, sp, row in zip(z["subjects"], z["topologies"], z["split"], z["Z"]):
            targets[(str(s), str(t))] = row.astype(np.float32)
            if split.setdefault(str(s), str(sp)) != str(sp):
                raise SystemExit(f"{s}: split incoerente fra le etichette")
    return targets, split


def build_view(view: Path, entries) -> None:
    view.mkdir(parents=True, exist_ok=True)
    for s, t in entries:
        src = operator_file(s, t)
        if not src.exists():
            raise SystemExit(f"operatori assenti: {src}")
        dst = view / f"{s}_GTready_{t}.npz"
        if not dst.exists():
            dst.symlink_to(src)


def l2(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z, dim=-1)


def distill_losses(S: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Puntuale (1 - coseno) e relazionale (MSE fra matrici di coseni fuori diagonale)."""
    point = (1.0 - (S * T).sum(dim=1)).mean()
    off = ~torch.eye(S.shape[0], dtype=torch.bool, device=S.device)
    rel = ((S @ S.T - T @ T.T)[off] ** 2).mean()
    return point, rel


@torch.no_grad()
def embed(model, dataset, indices, device) -> torch.Tensor:
    model.eval()
    out = []
    for i in indices:
        sd = sample_to_device(dataset[int(i)], device)
        z, _ = forward_model(model=model, sample_dict=sd, V_in=sd["verts"], return_gate_info=False, add_noise=False)
        out.append(l2(z.squeeze(0).float()))
    return torch.stack(out)


def validate(model, dataset, val_items, targets_t, device, topologies) -> dict:
    """Coseno con l'insegnante e rank-1 cross-topologia sui soggetti di validazione."""
    idx = [i for i, _, _ in val_items]
    Z = embed(model, dataset, idx, device)
    T = torch.stack([targets_t[k] for _, _, k in val_items]).to(device)
    cos = (Z * T).sum(1)
    pos = {(s, t): n for n, (_, (s, t), _) in enumerate(val_items)}
    subjects = sorted({s for _, (s, _), _ in val_items})
    hits, n_q = 0, 0
    for ta in topologies:
        for tb in topologies:
            if ta == tb:
                continue
            ok = [s for s in subjects if (s, ta) in pos and (s, tb) in pos]
            if len(ok) < 2:
                continue
            qa = Z[[pos[(s, ta)] for s in ok]]
            gb = Z[[pos[(s, tb)] for s in ok]]
            pred = (qa @ gb.T).argmax(1).cpu().numpy()
            hits += int((pred == np.arange(len(ok))).sum())
            n_q += len(ok)
    model.train()
    return {"val_cos": float(cos.mean()), "val_cos_p10": float(torch.quantile(cos, 0.1)),
            "val_rank1": hits / max(n_q, 1), "val_n_queries": n_q, "val_n_meshes": len(val_items)}


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    topologies = [t for t in args.topologies.split(",") if t]

    targets, split = load_teacher(args.teacher_dirs)
    train_subj = sorted(s for s, sp in split.items() if sp == "train")
    val_subj = sorted(s for s, sp in split.items() if sp == "val")
    if args.limit_subjects > 0:
        rng0 = np.random.default_rng(args.seed)
        train_subj = sorted(rng0.choice(train_subj, size=args.limit_subjects, replace=False).tolist())
        val_subj = val_subj[: max(4, args.limit_subjects // 4)]
    keep = set(train_subj) | set(val_subj)
    # (soggetto, topologia d'ingresso) -> chiave dell'insegnante
    entries = {}
    for (s, t) in targets:
        if s not in keep:
            continue
        if t == "original":
            for topo in topologies:
                entries[(s, topo)] = (s, "original")
        elif not args.no_expr:
            entries[(s, t)] = (s, t)
    view = args.stage / "view"
    build_view(view, sorted(entries))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    # Cache RAM non pinned (come train_steps.py: la memoria pinned non compare nel MaxRSS ma il
    # cgroup la conta).
    orig_res = fd._to_residency
    fd._to_residency = lambda sample, dev, pin: orig_res(sample, dev, False)
    t0 = time.time()
    dataset = fd.CachedDataset(view, workers=args.cache_workers, residency="ram", max_gb=args.cache_max_gb)
    print(f"[distill] cache: {len(dataset)} mesh in {time.time() - t0:.0f}s", flush=True)

    by_subject: dict[str, list[tuple[int, tuple[str, str]]]] = {}
    val_items = []
    val_set = set(val_subj)
    for i, f in enumerate(dataset.files):
        m = NAME_RE.match(Path(f).stem)
        s, t = m.group(1), m.group(2)
        key = entries[(s, t)]
        if s in val_set:
            val_items.append((i, (s, t), key))
        else:
            by_subject.setdefault(s, []).append((i, key))
    targets_t = {k: torch.from_numpy(v) for k, v in targets.items()}
    n_expr = sum(t.startswith("rexpr") for (_, t) in entries)
    print(f"[distill] training: {len(by_subject)} soggetti, {sum(map(len, by_subject.values()))} mesh "
          f"({n_expr} rexpr in tutto); validazione: {len(val_subj)} soggetti, {len(val_items)} mesh; "
          f"topologie {topologies}", flush=True)

    margs = argparse.Namespace(model=args.model, latent_dim=args.latent_dim, width=args.width,
                               n_blocks=args.n_blocks, dropout=args.dropout, pool_mode=args.pool_mode)
    model = build_model(margs, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    noise_modes = parse_noise_modes(args.noise_modes)
    noise_probs = np.asarray(parse_noise_mode_weights(args.noise_mode_weights, noise_modes), dtype=np.float64)
    perturbation = PerturbationParams.from_namespace(args)
    halve_epoch = int(round(args.lr_halve_at * args.epochs))

    ckpt_dir = args.run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    train_log = open(args.run_dir / "train_log.csv", "w", newline="")
    tw = csv.writer(train_log)
    tw.writerow(["epoch", "lr", "steps", "loss", "point", "rel", "train_cos", "s_per_step", "elapsed_s"])
    val_log = open(args.run_dir / "val_log.csv", "w", newline="")
    vw = None
    subjects = np.array(sorted(by_subject), dtype=object)
    t_start = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        if epoch == halve_epoch + 1:
            for g in optimizer.param_groups:
                g["lr"] = args.lr / 2
        rng = np.random.default_rng(args.seed + 977 + epoch)
        perm = rng.permutation(subjects)
        sums = np.zeros(4)
        n_steps = 0
        t_ep = time.time()
        for start in range(0, len(perm) - args.batch_subjects + 1, args.batch_subjects):
            batch = perm[start:start + args.batch_subjects]
            do_noise = bool(rng.uniform() < args.p_noise)
            sigma = sample_log_uniform_sigma(args.sigma_min, args.sigma_max, rng) if do_noise else 0.0
            optimizer.zero_grad(set_to_none=True)
            zs, keys = [], []
            for s in batch:
                cand = by_subject[s]
                pick = rng.choice(len(cand), size=min(args.meshes_per_subject, len(cand)), replace=False)
                for j in pick:
                    i, key = cand[int(j)]
                    sd = sample_to_device(dataset[i], device)
                    V_in = sd["verts"]
                    if sigma > 0.0:
                        mode = noise_modes[int(rng.choice(len(noise_modes), p=noise_probs))]
                        V_in = apply_xyz_perturbation_with_params(V=V_in, mode=mode, sigma=sigma, params=perturbation)
                    z, _ = forward_model(model=model, sample_dict=sd, V_in=V_in, return_gate_info=False,
                                         add_noise=not args.no_latent_noise)
                    zs.append(z.squeeze(0))
                    keys.append(key)
            S = l2(torch.stack(zs).float())
            T = torch.stack([targets_t[k] for k in keys]).to(device)
            point, rel = distill_losses(S, T)
            loss = point + args.lambda_rel * rel
            if not torch.isfinite(loss):
                raise SystemExit(f"loss non finita all'epoca {epoch}, passo {n_steps}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n_steps += 1
            sums += [float(loss), float(point), float(rel), float((S * T).sum(1).mean())]
            if n_steps % 50 == 0:
                print(f"[distill] epoca {epoch} passo {n_steps}: loss {sums[0] / n_steps:.4f} "
                      f"({(time.time() - t_ep) / n_steps:.2f} s/passo)", flush=True)
        m = sums / max(n_steps, 1)
        sps = (time.time() - t_ep) / max(n_steps, 1)
        lr = optimizer.param_groups[0]["lr"]
        tw.writerow([epoch, lr, n_steps, *[f"{x:.6f}" for x in m], f"{sps:.3f}", f"{time.time() - t_start:.0f}"])
        train_log.flush()
        line = (f"[distill] epoca {epoch}/{args.epochs} lr {lr:.1e}: loss {m[0]:.4f} punt {m[1]:.4f} "
                f"rel {m[2]:.5f} cos {m[3]:.4f} | {sps:.2f} s/passo, {time.time() - t_start:.0f}s")
        if epoch % args.val_every == 0 or epoch == args.epochs:
            v = validate(model, dataset, val_items, targets_t, device, topologies)
            if vw is None:
                vw = csv.DictWriter(val_log, fieldnames=["epoch", *v])
                vw.writeheader()
            vw.writerow({"epoch": epoch, **v})
            val_log.flush()
            line += f" | val cos {v['val_cos']:.4f} rank-1 {v['val_rank1']:.3f}"
        print(line, flush=True)
        bundle = {"state_dict": model.state_dict(), "args": vars(margs), "epoch": epoch}
        torch.save(bundle, ckpt_dir / "last.pth")
        if epoch % args.save_every == 0:
            torch.save(bundle, ckpt_dir / f"epoch{epoch:03d}.pth")
    train_log.close()
    val_log.close()
    if torch.cuda.is_available():
        print(f"[distill] picco memoria GPU {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GiB", flush=True)
    print(f"[distill] fine in {time.time() - t_start:.0f}s -> {ckpt_dir / 'last.pth'}", flush=True)


if __name__ == "__main__":
    main()
