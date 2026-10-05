#!/usr/bin/env python
"""WS5: expression variants of the ICT held-out identities.

    aau/run.sh v2_work/genict/make_expressions.py --n-cores 8

For each held-out identity, 5 expressions x 3 intensities on the `original`
topology, written as `datasets/ICT/expressions/idNNNN_expr_<name>_<intensity>.npz`
with keys `V`/`F`.  Same vertex set and same triangles as
`idNNNN_GTready_original.npz`, so an expression mesh is directly comparable,
vertex by vertex, to the neutral one it came from -- which is what the WS5
question needs ("does the identity ranking survive an expression?",
paper/PLAN_CVPR2027.md:120).

Model: `V = neutral + sum_i w_i * id_i + t * sum_{s in shapes} expr_s`, with the
identity weights `w` read back from `identity_weights.json` so the expression
meshes are the *same* identities as the neutral ones, not a fresh sample.

The five expressions are picked to span the face: two are jaw/lip shapes driven
by a single ICT blendshape, three are symmetric pairs that ICT splits per side
(`mouthSmile_L`/`_R`) and that are driven here with one shared weight, because a
one-sided smile is not the perturbation we mean.

Intensities 0.33 / 0.66 / 1.00 are ICT's own blendshape range (1.0 = the shipped
morph target at full strength), so the table can be read as a dose-response
curve; the file name carries two decimals for all three so the names sort.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ict_model import MODEL_DIR, N_SHAPE, ict_expression_mesh, load_ict  # noqa: E402
from mesh_ops import save_variant  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPRESSIONS: dict[str, tuple[str, ...]] = {
    "browDown": ("browDown_L", "browDown_R"),
    "eyeBlink": ("eyeBlink_L", "eyeBlink_R"),
    "jawOpen": ("jawOpen",),
    "mouthPucker": ("mouthPucker",),
    "mouthSmile": ("mouthSmile_L", "mouthSmile_R"),
}
INTENSITIES = (0.33, 0.66, 1.0)

_MODEL: dict | None = None


def _init(model_dir: str, n_shape: int) -> None:
    global _MODEL
    shapes = tuple(s for names in EXPRESSIONS.values() for s in names)
    _MODEL = load_ict(model_dir, n_shape=n_shape, expressions=shapes)


def process_subject(task: tuple[str, list[float], str, bool]) -> tuple[str, str]:
    sid, weights, out_dir_str, overwrite = task
    out_dir = Path(out_dir_str)
    try:
        w = np.asarray(weights, dtype=np.float64)
        n_written = 0
        for name, shapes in EXPRESSIONS.items():
            for t in INTENSITIES:
                out = out_dir / f"{sid}_expr_{name}_{t:.2f}.npz"
                if out.exists() and not overwrite:
                    continue
                V, F = ict_expression_mesh(w, _MODEL, shapes, t)
                save_variant(V, F, out)
                n_written += 1
        return "[ok]", f"{sid} wrote={n_written}"
    except Exception as exc:  # one bad identity must not kill the batch
        return "[fail]", f"{sid}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights-json", type=Path,
                    default=REPO_ROOT / "datasets/ICT/identities/identity_weights.json")
    ap.add_argument("--heldout-file", type=Path,
                    default=REPO_ROOT / "datasets/ICT/train_ready/split_heldout.txt")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "datasets/ICT/expressions")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--n-shape", type=int, default=N_SHAPE)
    ap.add_argument("--id-offset", type=int, default=10000)
    ap.add_argument("--n-cores", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    weights = json.loads(a.weights_json.read_text())
    heldout = [ln.strip() for ln in a.heldout_file.read_text().splitlines() if ln.strip()]
    if not heldout:
        raise SystemExit(f"no subject ids in {a.heldout_file}")

    tasks = []
    for sid in heldout:
        src = f"ict{int(sid[2:]) - a.id_offset:04d}"
        if src not in weights:
            raise SystemExit(f"{sid} -> {src} has no entry in {a.weights_json}")
        tasks.append((sid, weights[src], str(a.out_dir), a.overwrite))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    n_files = len(tasks) * len(EXPRESSIONS) * len(INTENSITIES)
    print(f"{len(tasks)} held-out identities x {len(EXPRESSIONS)} expressions x "
          f"{len(INTENSITIES)} intensities = {n_files} meshes -> {a.out_dir}", flush=True)

    if a.n_cores > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        pool = mp.Pool(processes=a.n_cores, initializer=_init,
                       initargs=(str(a.model_dir), a.n_shape))
        results = pool.imap_unordered(process_subject, tasks)
    else:
        _init(str(a.model_dir), a.n_shape)
        pool, results = None, map(process_subject, tasks)

    tally = {"[ok]": 0, "[fail]": 0}
    failures = []
    try:
        for i, (status, msg) in enumerate(results, start=1):
            tally[status] += 1
            if status == "[fail]":
                failures.append(msg)
            if status != "[ok]" or i % 50 == 0 or i <= 3:
                print(f"[{i}/{len(tasks)}] {status} {msg}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    manifest = {
        "expressions": {k: list(v) for k, v in EXPRESSIONS.items()},
        "intensities": list(INTENSITIES),
        "n_subjects": len(tasks),
        "n_meshes": n_files,
        "topology": "original",
        "id_offset": a.id_offset,
        "source_weights": str(a.weights_json),
        "name_pattern": "idNNNN_expr_<expression>_<intensity:.2f>.npz",
    }
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nDone. ok={tally['[ok]']} fail={tally['[fail]']}")
    for msg in failures[:20]:
        print(f"  - {msg}")
    if tally["[fail]"]:
        raise SystemExit(1)


def demo() -> None:
    """Self-check: every expression moves the face region, and more at higher intensity."""
    _init(str(MODEL_DIR), 8)
    w = np.zeros(8)
    V0, F = ict_expression_mesh(w, _MODEL, (), 0.0)
    for name, shapes in EXPRESSIONS.items():
        prev = 0.0
        for t in INTENSITIES:
            V, _ = ict_expression_mesh(w, _MODEL, shapes, t)
            d = np.linalg.norm(V - V0, axis=1)
            assert d.max() > 1e-3, f"{name}@{t} does not move the face region"
            assert d.max() > prev, f"{name}: intensity {t} is not stronger than the previous"
            prev = float(d.max())
            print(f"  {name}@{t:.2f}: mean {d.mean() * 10:.3f} mm, max {d.max() * 10:.2f} mm, "
                  f"moved verts {int((d > 1e-4).sum())}/{len(V)}")
    print("demo OK: all 5 expressions are non-degenerate and monotone in intensity")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
