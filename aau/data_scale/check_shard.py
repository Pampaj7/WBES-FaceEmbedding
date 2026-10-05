#!/usr/bin/env python3
"""Verifica di uno shard di gen_ict_shard.py (mini-lotto), prima di lanciare l'array.

    aau/run.sh aau/data_scale/check_shard.py --shard-tar <shard.tar> --ckpt <ckpt.pth> \
        --work-dir /tmp/$SLURM_JOB_ID/check --out-json <report.json> --n-proc 16

Blocchi, ognuno con esito nel report; l'uscita e' 1 se uno fallisce.

1. ``meshes``: ricontrollo indipendente dalla guardia del generatore (vertici finiti, nessun
   vertice non referenziato, original/noisy/rexpr sulla tabella di facce di ICT-5000
   ``ict0000``, triangoli entro il 2% dei target, crop piu' piccolo di original).
2. ``operators``: gli operatori del pre-pass (``prepass_ops.py``) contro quelli dello script
   che ha prodotto gli operatori in uso, ``v2_work/potential/areanorm_operators.py``, eseguito
   COSI' COM'E' sulla stessa geometria. Entrambi letti dal loader CONGELATO; si confrontano i
   tensori (gli autovettori a meno del segno: eigsh parte da un vettore casuale) e
   l'embedding del modello congiunto.
3. ``gt``: la vertex-mean-L2 fra original normalizzate maxabs (la GT di
   ``build_ict_gt_matrix.py``) fra identita' nuove ha la distribuzione di ICT-5000; nessuna
   identita' nuova e' un quasi-duplicato di una delle 5000 (held-out compresi); la stessa
   funzione, applicata a coppie di ICT-5000, ridà la matrice GT in uso.
4. ``model``: per ogni mesh nuova la distanza latente media verso le altre mesh della stessa
   identita' sta sotto quella verso le altre identita' (sanita' del dato).
5. ``seeds``: i pesi del manifest sono quelli che i semi rigenerano.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"))
sys.path.insert(0, str(REPO_ROOT / "v2_work/genict"))
sys.path.insert(0, str(THIS_DIR))

from make_ict_topologies import REMESH_DECIMATION, triangle_targets  # noqa: E402
from robustness.data_utils import GTReadyDataset, sample_to_device  # noqa: E402

ICT = REPO_ROOT / "datasets/ICT"
NAME_RE = re.compile(r"^(id\d+)_GTready_([a-z0-9]+)\.npz$")
SPARSE = ("L", "gradX", "gradY")


def maxabs(V: np.ndarray) -> np.ndarray:
    Vc = V - V.mean(0, keepdims=True)
    return Vc / max(float(np.abs(Vc).max()), 1e-9)


def vml2_rows(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.stack([(B - a).norm(dim=-1).mean(-1) for a in A])


def check_meshes(geom: Path, report: dict) -> dict:
    F_ref = np.load(ICT / "topo/ict0000_GTready_original.npz")["F"]
    n_orig = len(F_ref)
    down_t, up_t = triangle_targets(n_orig)
    expect = {"remesh": int(n_orig * REMESH_DECIMATION), "down8k": down_t, "up60k": up_t}
    bad, by_label, originals = [], {}, {}
    for p in sorted(geom.glob("*.npz")):
        sid, label = NAME_RE.match(p.name).groups()
        with np.load(p) as z:
            V, F = z["V"].astype(np.float64), z["F"]
        key = "rexpr" if label.startswith("rexpr") else label
        by_label[key] = by_label.get(key, 0) + 1
        if not np.isfinite(V).all():
            bad.append(f"{p.name}: vertici non finiti")
        if F.max() != len(V) - 1 or len(np.unique(F)) != len(V):
            bad.append(f"{p.name}: vertici non referenziati")
        if key in ("original", "noisy", "rexpr") and not np.array_equal(F, F_ref):
            bad.append(f"{p.name}: facce diverse da ICT-5000 original")
        if label in expect and abs(len(F) - expect[label]) > 0.02 * expect[label]:
            bad.append(f"{p.name}: {len(F)} triangoli, attesi ~{expect[label]}")
        if label == "crop" and len(F) >= n_orig:
            bad.append(f"{p.name}: crop non ha tolto niente")
        if label == "original":
            originals[sid] = V
    report["meshes"] = {"ok": not bad, "n_files": sum(by_label.values()), "by_label": by_label,
                        "targets": expect, "errors": bad[:20]}
    return originals


def compare_ops(view_a: Path, view_b: Path, report: dict) -> tuple[GTReadyDataset, GTReadyDataset]:
    da, db = GTReadyDataset(str(view_a)), GTReadyDataset(str(view_b))
    if da.files != db.files:
        raise SystemExit("le due viste di operatori non hanno gli stessi file")
    worst = {k: 0.0 for k in ("verts", "mass", "evals", "evecs_signfree", *SPARSE)}
    rejected = []
    for i, name in enumerate(da.files):
        a, b = da[i], db[i]
        if a is None or b is None:
            rejected.append(name)
            continue
        for k in ("verts", "mass", "evals"):
            worst[k] = max(worst[k], float((a[k] - b[k]).abs().max()))
        sign = torch.sign((a["evecs"] * b["evecs"]).sum(0))
        sign[sign == 0] = 1
        rel = float((a["evecs"] - b["evecs"] * sign).abs().max() / b["evecs"].abs().max())
        worst["evecs_signfree"] = max(worst["evecs_signfree"], rel)
        for k in SPARSE:
            ca, cb = a[k].coalesce(), b[k].coalesce()
            if not torch.equal(ca.indices(), cb.indices()):
                worst[k] = float("inf")
            else:
                worst[k] = max(worst[k], float((ca.values() - cb.values()).abs().max()
                                               / cb.values().abs().max()))
    report["operators"] = {"n_files": len(da.files), "rejected_by_frozen_loader": rejected,
                           "max_abs_diff_prepass_vs_reference": worst,
                           "note": "evecs e sparse in relativo al massimo |valore|"}
    return da, db


def embed(ds: GTReadyDataset, model, dev) -> np.ndarray:
    from robustness.model_helpers import forward_model
    Z = []
    with torch.no_grad():
        for i in range(len(ds.files)):
            s = sample_to_device(ds[i], dev)
            z, _ = forward_model(model, s, s["verts"], return_gate_info=False, add_noise=False)
            Z.append(z.reshape(-1).cpu().numpy())
    return np.stack(Z)


def check_model(da: GTReadyDataset, db: GTReadyDataset, ckpt: Path, report: dict) -> None:
    from types import SimpleNamespace
    from robustness.model_helpers import build_model
    from robustness.posthoc_runner import load_checkpoint_bundle, merge_run_args

    margs = SimpleNamespace(**merge_run_args(ckpt, ""))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args=margs, device=dev)
    model.load_state_dict(load_checkpoint_bundle(ckpt)["state_dict"], strict=True)
    model.eval()
    Za, Zb = embed(da, model, dev), embed(db, model, dev)
    D = np.linalg.norm(Zb[:, None] - Zb[None], axis=-1)
    med = float(np.median(D[np.triu_indices(len(D), 1)]))
    dz = np.linalg.norm(Za - Zb, axis=1) / med
    report["operators"]["embedding_dz_rel_max"] = float(dz.max())
    report["operators"]["ok"] = bool(not report["operators"]["rejected_by_frozen_loader"]
                                     and dz.max() < 1e-4)

    sid = np.array([NAME_RE.match(n).group(1) for n in db.files])
    same = sid[:, None] == sid[None, :]
    np.fill_diagonal(same, False)
    diff = ~(sid[:, None] == sid[None, :])
    within = np.array([D[i][same[i]].mean() for i in range(len(sid))])
    across = np.array([D[i][diff[i]].mean() for i in range(len(sid))])
    frac = float(np.mean(within < across))
    report["model"] = {"ckpt": str(ckpt), "device": str(dev),
                       "mean_within_identity": float(within.mean()),
                       "mean_across_identity": float(across.mean()),
                       "frac_meshes_within_lt_across": frac, "ok": frac >= 0.9}


def check_gt(originals: dict, report: dict) -> None:
    man = json.loads((ICT / "gt/manifest.json").read_text())
    scale = float(man["normalization_scale"]["maxabs"])
    ref = man["offdiag_stats_unnormalized"]["maxabs"]
    with np.load(ICT / "gt/ict_matrix_distances_maxabs.npz") as z:
        D_old = z["D_orig"].astype(np.float64) * scale
        old_names = [str(n) for n in z["names"]]
    D_loo = D_old.copy()
    np.fill_diagonal(D_loo, np.inf)
    nn_old = D_loo.min(1)

    olds = []
    for n in old_names:
        with np.load(ICT / f"topo/{n}_GTready_original.npz") as z:
            olds.append(maxabs(z["V"].astype(np.float64)))
    O = torch.tensor(np.stack(olds), dtype=torch.float32)
    sids = sorted(originals)
    N = torch.tensor(np.stack([maxabs(originals[s]) for s in sids]), dtype=torch.float32)
    D_nn = vml2_rows(N, N).double().numpy()
    D_no = vml2_rows(N, O).double().numpy()
    # la stessa funzione sulle prime 20 di ICT-5000 deve ridare la GT in uso
    D_re = vml2_rows(O[:20], O[:20]).double().numpy()
    iu20 = np.triu_indices(20, 1)
    gt_reprod = float(np.abs(D_re[iu20] - D_old[:20, :20][iu20]).max() / D_old[:20, :20][iu20].max())

    iu = np.triu_indices(len(sids), 1)
    new_pairs = D_nn[iu]
    nn_new = D_no.min(1)
    out = {
        "new_vs_new": {"median": float(np.median(new_pairs)), "p1": float(np.percentile(new_pairs, 1)),
                       "min": float(new_pairs.min())},
        "ict5000_ref": {"median": ref["median"], "p1": ref["p1"], "min": ref["min"]},
        "nn_new_to_ict5000": {"min": float(nn_new.min()), "median": float(np.median(nn_new))},
        "nn_ict5000_loo": {"min": float(nn_old.min()), "p1": float(np.percentile(nn_old, 1)),
                           "median": float(np.median(nn_old))},
        "gt_function_vs_stored_matrix_rel_max": gt_reprod,
    }
    med_ok = abs(out["new_vs_new"]["median"] - ref["median"]) < 0.15 * ref["median"]
    dup_ok = out["nn_new_to_ict5000"]["min"] > 0.5 * float(nn_old.min())
    out["ok"] = bool(med_ok and dup_ok and gt_reprod < 1e-4)
    report["gt"] = out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-tar", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--n-proc", type=int, default=8)
    a = ap.parse_args()

    import prepass_ops
    from gen_ict_shard import identity_weights

    a.work_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"shard_tar": str(a.shard_tar)}
    geom = a.work_dir / "geom"
    with tarfile.open(a.shard_tar) as tar:
        tar.extractall(geom)
    man = json.loads((geom / "manifest.json").read_text())
    (geom / "manifest.json").unlink()

    originals = check_meshes(geom, report)

    # operatori: pre-pass contro lo script di riferimento eseguito cosi' com'e'
    rep_pp = prepass_ops.run(a.work_dir / "ops_prepass", [a.shard_tar], [], a.n_proc,
                             geom_stage=a.work_dir / "geom_prepass")
    ref_dir = a.work_dir / "ops_reference"
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, str(REPO_ROOT / "v2_work/potential/areanorm_operators.py"),
                               "--input-dir", str(geom), "--output-dir", str(ref_dir),
                               "--k-eig", "128", "--shard", f"{i}/{a.n_proc}"],
                              stdout=subprocess.DEVNULL) for i in range(a.n_proc)]
    rcs = [p.wait() for p in procs]
    report["prepass"] = {**rep_pp, "reference_wall_seconds": time.time() - t0, "reference_rc": rcs}
    n_meshes = rep_pp["n_meshes"]
    report["prepass"]["cpu_seconds_per_mesh"] = rep_pp["cpu_seconds"] / max(n_meshes, 1)
    da, db = compare_ops(a.work_dir / "ops_prepass", ref_dir, report)
    torch.set_num_threads(a.n_proc)   # il pre-pass era a thread singolo, il forward no
    check_model(da, db, a.ckpt, report)

    check_gt(originals, report)

    rec = man["identities"][0]
    report["seeds"] = {"ok": bool(np.allclose(identity_weights(rec["g"]), rec["weights"]))}

    side = json.loads(a.shard_tar.with_suffix(".json").read_text())
    report["shard"] = {k: side[k] for k in ("n_identities", "n_expr", "n_meshes", "tar_bytes",
                                            "bytes_per_label_mean", "seconds", "n_cores", "guard")}
    report["shard"]["tar_bytes_per_identity"] = side["tar_bytes"] / side["n_identities"]

    blocks = [report[k]["ok"] for k in ("meshes", "operators", "gt", "model", "seeds")]
    report["ok"] = bool(all(blocks) and not rep_pp["n_failed"] and not any(rcs))
    text = json.dumps(report, indent=1)
    print(text)
    a.out_json.write_text(text + "\n")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
