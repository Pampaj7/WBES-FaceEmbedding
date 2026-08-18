#!/usr/bin/env python
"""The two runnable checks for the v2 training package (assert-based, no framework).

    .conda_env/bin/python v2_work/train_v2/check_train_v2.py --task support
    .conda_env/bin/python v2_work/train_v2/check_train_v2.py --task joint

`support`: builds a BFM-only view that includes the support bank, then runs 2 CPU
epochs and asserts the losses are finite (bank *structure* is checked by
`make_support_bank.py --demo`).
`joint`: unit-asserts the three v2 safety pieces (domain-pure batches, the NaN
guard, single-domain eval), then runs 2 CPU epochs on BFM(8)+FLAME(8).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

import train_v2  # noqa: E402


def build_view(out_dir: Path, extra: list[str]) -> Path:
    cmd = [sys.executable, str(THIS_DIR / "make_joint_view.py"), "--out-dir", str(out_dir), *extra]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, stdout=subprocess.DEVNULL)
    n = len(list((out_dir / "npz_withops").glob("*.npz")))
    print(f"[check] view {out_dir} : {n} meshes")
    assert n > 0
    return out_dir


def _argv(view: Path, runs_root: Path, batch_subjects: int) -> list[str]:
    return [
        "check_train_v2",
        "--data_dir", str(view / "npz_withops"),
        "--dist_npz", str(view / "gt_matrix.npz"),
        "--runs_root", str(runs_root),
        "--device", "cpu",
        "--epochs", "2",
        "--batch_subjects", str(batch_subjects),
        "--model", "xyz_dn", "--k_spec", "0", "--no-log_spec",
        "--latent_dim", "64", "--width", "32", "--n_blocks", "2", "--eig_k", "64",
        "--max_meshes_per_subject_train", "2",
        "--eval_every", "0", "--no-preload_eval_samples_train",
        "--seed", "0",
    ]


def train_two_epochs(view: Path, runs_root: Path, batch_subjects: int) -> Path:
    sys.argv = _argv(view, runs_root, batch_subjects)
    before = train_v2.STATS["blocked_permutations"]
    train_v2.run_training_v2(train_v2.parse_args_v2())
    n_perm = train_v2.STATS["blocked_permutations"] - before

    run_dirs = sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    run_dir = run_dirs[-1]
    assert (run_dir / "config.json").is_file(), "config.json not written"
    assert (run_dir / "launch_command.txt").is_file(), "launch_command.txt not written"

    rows = [r.split(",") for r in (run_dir / "train_log.csv").read_text().strip().split("\n")[1:]]
    assert len(rows) == 2, f"expected 2 epoch rows, got {len(rows)}"
    for row in rows:
        loss, stress = float(row[1]), float(row[2])
        assert np.isfinite(loss) and np.isfinite(stress), f"non-finite loss/stress: {row}"
        assert loss > 0.0, f"suspicious zero loss: {row}"
    assert n_perm == 2, f"expected one domain-blocked permutation per epoch, got {n_perm}"
    print(f"[check] 2 epochs OK, losses={[float(r[1]) for r in rows]}, run_dir={run_dir}")
    return run_dir


def check_support(work: Path) -> None:
    view = build_view(
        work / "view_support",
        ["--domains", "bfm", "--n-bfm", "12", "--variants", "original,supp0,supp1,supp2"],
    )
    per_subject: dict[str, list[str]] = {}
    for p in (view / "npz_withops").glob("*.npz"):
        per_subject.setdefault(p.name.split("_")[0], []).append(p.name)
    assert len(per_subject) == 12, per_subject.keys()
    for sid, files in per_subject.items():
        supp = [f for f in files if "_supp" in f]
        assert len(supp) >= 2, f"{sid}: support bank missing from the view ({files})"
    # the augmentation is the sampler: support variants are ordinary meshes of the subject.
    # batch_subjects >= 3: stress_loss normalizes by the off-diagonal mean, so a 2-subject
    # batch is identically 0 with zero gradient (v1 only guards against < 2).
    train_two_epochs(view, work / "runs_support", batch_subjects=3)


def check_joint(work: Path) -> None:
    # 1. domain-pure batches
    subjects = [f"id{i:04d}" for i in range(6)] + [f"id{1000 + i}" for i in range(5)]
    order = train_v2.domain_blocked_order(subjects, 2, np.random.default_rng(0)).tolist()
    assert len(order) == 10, order  # 6 bfm + 4 of 5 flame (short tail dropped)
    for start in range(0, len(order), 2):
        chunk = order[start : start + 2]
        assert len({train_v2.domain_of(s) for s in chunk}) == 1, f"batch spans domains: {chunk}"
    assert set(order) <= set(subjects) and len(set(order)) == len(order)
    try:
        train_v2.domain_blocked_order(subjects, 6, np.random.default_rng(0))
    except ValueError as exc:
        assert "flame" in str(exc), exc
    else:
        raise AssertionError("a domain smaller than batch_subjects must raise, not be dropped")

    # 2. NaN guard: an undefined inter-domain entry cannot be read silently
    D = np.array([[0.0, 0.5, np.nan], [0.5, 0.0, np.nan], [np.nan, np.nan, 0.0]])
    guarded = D.view(train_v2.NanGuardedMatrix)
    assert np.allclose(guarded[np.ix_([0, 1], [0, 1])], D[:2, :2])  # intra-domain: fine
    for key in (np.ix_([0, 2], [0, 2]), (0, 2)):
        try:
            guarded[key]
        except AssertionError:
            pass
        else:
            raise AssertionError(f"NaN read not caught for {key!r}")

    # 3. online eval is restricted to one domain
    seen = {}
    kept = train_v2._single_domain_eval(lambda **kw: seen.update(kw) or kw["eval_subjects"], "")(
        eval_subjects=["id0000", "id0001", "id1000"], max_subjects_eval_train=0, seed=0
    )
    assert kept == ["id0000", "id0001"], kept

    # 4. 2 CPU epochs on BFM(8) + FLAME(8)
    view = build_view(
        work / "view_joint",
        ["--domains", "bfm,flame", "--n-bfm", "8", "--n-flame", "8",
         "--variants", "original,down8k", "--support-bank", ""],
    )
    with np.load(view / "gt_matrix.npz", allow_pickle=True) as z:
        D = z["D_orig"]
        names = [str(s) for s in z["names"]]
    assert len(names) == 16 and len(set(names)) == 16, names
    assert int(np.count_nonzero(~np.isfinite(D))) == 2 * 8 * 8, "off-block must be undefined (NaN)"
    train_two_epochs(view, work / "runs_joint", batch_subjects=4)

    # 5. negative control: with v1's own batching the guard must fire on a mixed batch
    saved = train_v2.v1.load_gt_distance_matrix
    train_v2.v1.load_gt_distance_matrix = train_v2._nan_guarded_loader(saved)
    sys.argv = _argv(view, work / "runs_negative_control", 4)
    try:
        train_v2.v1.run_training(train_v2.parse_args_v2())
    except AssertionError as exc:
        assert "undefined (NaN) GT entry" in str(exc), exc
        print("[check] negative control OK: unblocked batching reads the undefined GT block")
    else:
        raise AssertionError("a mixed-domain batch trained without touching the NaN block")
    finally:
        train_v2.v1.load_gt_distance_matrix = saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("support", "joint"), required=True)
    ap.add_argument("--work-dir", type=Path, default=THIS_DIR / "checks")
    a = ap.parse_args()
    a.work_dir.mkdir(parents=True, exist_ok=True)
    {"support": check_support, "joint": check_joint}[a.task](a.work_dir)
    print(f"CHECK OK: {a.task}")
