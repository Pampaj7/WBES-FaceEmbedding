#!/usr/bin/env python3
"""Confronto delle corse corte di trainer_repro.sbatch: train_steps.py riproduce v1?

    aau/run.sh aau/data_scale/trainer_repro_compare.py --runs-root <dir> --out-json <json>

Corse attese sotto ``--runs-root`` (una run dir di train_runner ciascuna):
  v1     train_fast.py, ricetta v1, seme 1234                            (riferimento)
  v1b    la stessa corsa ripetuta: misura il non-determinismo della GPU
  id     train_steps.py, stesso numero di passi, nessun blocco          (deve essere v1)
  seed   train_steps.py, split esplicito = quello di v1, seme 2345      (rumore del seme)
  blk    train_steps.py, 2 blocchi, operatori dal PRE-PASS sulla geometria, S dimezzato

Due misure:
  * la loss di training allineata per PASSO (le epoche di ``blk`` sono lunghe la meta');
  * una valutazione FISSA, uguale per tutte, indipendente dal seme dell'eval online: ultimo
    checkpoint, i 100 held-out BFM dello split v1 x 6 topologie (operatori in uso), Spearman
    latente-GT su tutte le coppie di mesh di soggetti diversi e sulle sole coppie
    cross-topologia.
Esito: ``id`` dista da ``v1`` non piu' di ``v1b`` (cioe' coincide a meno del
non-determinismo della GPU); ``|blk - v1|`` non supera ``|seed - v1|`` (piu' una
tolleranza di 0.01 sullo Spearman, perche' con un solo seme di confronto il rumore e' stimato
da una sola coppia).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"))
from robustness.data_utils import GTReadyDataset, sample_to_device  # noqa: E402

VIEW = REPO_ROOT / "datasets/REMESH/npz_data_topo_500_withops_areanorm"
GT = (REPO_ROOT / "face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix"
      / "normalized_matrix_distances.npz")
SPLIT = REPO_ROOT / "aau/data_scale/repro_split_bfm_s1234.json"
RUNS = ("v1", "v1b", "id", "seed", "blk")


def run_dir(root: Path, tag: str) -> Path:
    dirs = [p.parent for p in (root / tag).glob("*/train_log.csv")]
    if len(dirs) != 1:
        raise SystemExit(f"{root / tag}: attesa una run dir, trovate {len(dirs)}")
    return dirs[0]


def loss_by_step(rd: Path, steps_per_epoch: int) -> np.ndarray:
    """Loss media per epoca, espansa a una per passo (per confrontare epoche di lunghezza diversa)."""
    with open(rd / "train_log.csv") as f:
        rows = list(csv.reader(f))
    vals = [float(r[1]) for r in rows[1:]]
    return np.repeat(np.array(vals), steps_per_epoch)


def heldout_eval(rd: Path, ds: GTReadyDataset, idx: list[int], sids: list[str], D_gt: np.ndarray,
                 dev) -> dict:
    from types import SimpleNamespace
    from scipy.stats import spearmanr
    from robustness.model_helpers import build_model, forward_model
    from robustness.posthoc_runner import load_checkpoint_bundle, merge_run_args

    ck = sorted((rd / "checkpoints").glob("epoch*.pth"))[-1]
    model = build_model(args=SimpleNamespace(**merge_run_args(ck, "")), device=dev)
    model.load_state_dict(load_checkpoint_bundle(ck)["state_dict"], strict=True)
    model.eval()
    Z = []
    with torch.no_grad():
        for i in idx:
            s = sample_to_device(ds[i], dev)
            z, _ = forward_model(model, s, s["verts"], return_gate_info=False, add_noise=False)
            Z.append(z.reshape(-1).cpu().numpy())
    Z = np.stack(Z)
    D = np.linalg.norm(Z[:, None] - Z[None], axis=-1)
    topo = [ds.files[i].split("_GTready_")[1][:-4] for i in idx]
    iu = np.triu_indices(len(idx), 1)
    diff = np.array([sids[i] != sids[j] for i, j in zip(*iu)])
    xt = np.array([topo[i] != topo[j] for i, j in zip(*iu)]) & diff
    lat, gt = D[iu], D_gt[iu]
    return {"checkpoint": ck.name,
            "spearman_all": float(spearmanr(lat[diff], gt[diff]).statistic),
            "spearman_xtopo": float(spearmanr(lat[xt], gt[xt]).statistic)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--steps-per-epoch", type=json.loads, default='{"v1":80,"v1b":80,"id":80,"seed":80,"blk":40}')
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    held = json.loads(SPLIT.read_text())["heldout"]
    ds = GTReadyDataset(str(VIEW))
    idx = [i for i, n in enumerate(ds.files) if n.split("_GTready_")[0] in set(held)]
    sids = [ds.files[i].split("_GTready_")[0] for i in idx]
    with np.load(GT) as z:
        names = [str(n).split("_GTready")[0] for n in z["names"]]
        G = z["D_orig"]
    pos = {n: k for k, n in enumerate(names)}
    gi = [pos[s] for s in sids]
    D_gt = G[np.ix_(gi, gi)].astype(np.float64)

    out: dict = {"n_meshes_eval": len(idx)}
    for tag in RUNS:
        rd = run_dir(a.runs_root, tag)
        out[tag] = {"run_dir": str(rd), **heldout_eval(rd, ds, idx, sids, D_gt, dev)}
        out[tag]["loss_by_step"] = loss_by_step(rd, a.steps_per_epoch[tag])
    T = min(len(out[t]["loss_by_step"]) for t in RUNS)
    for tag in RUNS:
        L = out[tag].pop("loss_by_step")
        out[tag]["n_steps_logged"] = int(len(L))
        out[tag]["loss_last_quarter_mean"] = float(L[3 * T // 4:T].mean())
        if tag != "v1":
            Lr = loss_by_step(Path(out["v1"]["run_dir"]), a.steps_per_epoch["v1"])
            out[tag]["loss_max_abs_diff_vs_v1"] = float(np.abs(L[:T] - Lr[:T]).max())
    for m in ("spearman_all", "spearman_xtopo"):
        for tag in ("v1b", "id", "seed", "blk"):
            out[tag][f"{m}_minus_v1"] = out[tag][m] - out["v1"][m]
    noise = {m: abs(out["seed"][f"{m}_minus_v1"]) for m in ("spearman_all", "spearman_xtopo")}
    out["verdict"] = {
        "id_equals_v1_up_to_gpu_nondeterminism": bool(
            out["id"]["loss_max_abs_diff_vs_v1"] <= out["v1b"]["loss_max_abs_diff_vs_v1"] + 1e-6
            and abs(out["id"]["spearman_all_minus_v1"]) <= abs(out["v1b"]["spearman_all_minus_v1"]) + 1e-6),
        "blk_within_seed_noise": bool(all(abs(out["blk"][f"{m}_minus_v1"]) <= noise[m] + 0.01
                                          for m in noise)),
        "seed_noise": noise,
    }
    text = json.dumps(out, indent=1)
    print(text)
    a.out_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
