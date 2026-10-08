#!/usr/bin/env python3
"""Le 6 topologie con un'espressione CASUALE PER MESH sulle identita' FaceScape del set di sviluppo.

    aau/run.sh aau/zs3dmm/dev_fs_expr_topologies.py --identities-dir datasets/DEV_FACESCAPE/identities \\
        --out-dir datasets/DEV_FACESCAPE/expr_topo --gt-dir datasets/DEV_FACESCAPE/gt --n-cores 32
    (dev_fs_build.sbatch)

Gemello di ``make_zs_expr_topologies.py`` (set FaceVerse con espressioni), che conosce solo
FaceVerse e GNM; regole importate da li' e da ``make_ict_topologies.py``, non riscritte:
  - per ogni (identita', topologia) un vettore d'espressione indipendente, seme
    ``[--seed, numero dell'identita', indice della topologia]``, con la ricetta WS5 di FaceVerse
    (3-8 blendshape attivi, U(0.3, 1.0)) sul pool dei 51 blendshape FaceScape: il file non ha i
    nomi, quindi nessuna esclusione dello sguardo (``v3_work/mm/loaders.load_facescape``);
  - identita' = gli z di ``identities/identity_weights.json``, mesh = il forward bilineare della
    libreria (``BilinearModel``: core . id . exp, pesi residui contro la neutra);
  - topologia ricavata dalla patch con espressione con ``make_remesh`` / ``make_crop`` /
    ``make_noisy`` / ``make_down8k`` / ``make_up60k``, stessi target;
  - solo per il ``crop``: se con l'espressione il crop non toglie niente si riestrae, fino a 20
    volte (stessa regola, contata nel manifest).

Il core FaceScape (2.4 GB) resta nel processo principale: per ogni identita' quello calcola la
matrice (3n, 52) dell'identita' sulle 52 espressioni grezze e la passa al worker, che ne ricava
le patch per qualunque peso d'espressione. Spostamento e manifest come ``make_zs_expr_topologies``
(convenzione maxabs, sd fra soggetti, con ``--gt-dir`` il rapporto con la distanza GT dal soggetto
piu' vicino), piu' il riferimento del set FaceVerse con espressioni se e' su disco.

Output: ``fsNNNN_GTready_<topologia>.npz`` (V/F), ``expression_vectors.json`` (vettori sparsi
``{bsNN: peso}`` per identita' e topologia), ``manifest.json``. Riprendibile per identita'.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
# A livello di modulo: con lo start method spawn i worker rieseguono questo file.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))

import mesh_ops as mo  # noqa: E402
from make_ict_topologies import VARIANTS, make_down8k, make_noisy, make_remesh, make_up60k, triangle_targets  # noqa: E402
from make_zs_expr_topologies import (ICT_REFERENCE, MAX_CROP_DRAWS, maxabs_normalize,  # noqa: E402
                                     nearest_distances, sha256)
from v3_work.mm.model import COEF_HIGH, COEF_LOW, N_ACTIVE_MAX, N_ACTIVE_MIN, sample_expression_coefficients  # noqa: E402

PREFIX = "fs"
FV_EXPR_MANIFEST = REPO_ROOT / "datasets" / "FACEVERSE_ZS" / "expr_topo" / "manifest.json"

_F: np.ndarray | None = None
_EXPR = None


def _init(faces: np.ndarray, expr) -> None:
    global _F, _EXPR
    _F, _EXPR = faces, expr


def patch(M: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Patch (n, 3) da M (3n, 52, espressioni grezze) e pesi residui w (51)."""
    a = np.concatenate([[1.0 - w.sum()], w])
    return (M @ a).reshape(-1, 3)


def process_subject(task: tuple[str, np.ndarray, str, int, bool]) -> tuple[str, str, dict]:
    subject, M, out_dir_str, seed, overwrite = task
    out_dir = Path(out_dir_str)
    out = {v: out_dir / f"{subject}_GTready_{v}.npz" for v in VARIANTS}
    num = int(subject[len(PREFIX):])
    try:
        M = M.astype(np.float64)
        n_expr = len(_EXPR.names)
        V0 = patch(M, np.zeros(n_expr))
        N0 = maxabs_normalize(V0)
        diameter = float(np.linalg.norm(N0.max(axis=0) - N0.min(axis=0)))
        vectors, means, maxima, crop_draws, n_written, counts = {}, [], [], 0, 0, []
        for t_idx, variant in enumerate(VARIANTS):
            rng = np.random.default_rng([seed, num, t_idx])
            built = None
            for draw in range(1, MAX_CROP_DRAWS + 1):
                w = sample_expression_coefficients(rng, _EXPR)
                V = patch(M, w)
                base = mo.prepare_open_surface(*mo.as_arrays(V, _F))
                if len(base[1]) != len(_F):
                    return "[fail]", f"{subject}/{variant}: la patch non e' una superficie pulita", {"subject": subject}
                if variant != "crop":
                    break
                built = mo.make_crop(*base)   # il crop di mesh_ops torna la patch intatta se toglierebbe >15%
                if len(built[0]) != len(base[0]):
                    crop_draws = draw
                    break
            else:
                return "[fail]", f"{subject}/crop: crop identico alla patch per {MAX_CROP_DRAWS} espressioni", \
                    {"subject": subject}
            vectors[variant] = {_EXPR.names[i]: float(w[i]) for i in np.flatnonzero(w)}
            d = np.linalg.norm(maxabs_normalize(V) - N0, axis=1)
            means.append(float(d.mean()))
            maxima.append(float(d.max()))
            if overwrite or not out[variant].exists():
                down_target, up_target = triangle_targets(len(base[1]))
                builders = {
                    "original": lambda: base,
                    "remesh": lambda: make_remesh(*base),
                    "crop": lambda: built,
                    "noisy": lambda: make_noisy(*base, seed=int(subject[-4:])),
                    "down8k": lambda: make_down8k(*base, target=down_target),
                    "up60k": lambda: make_up60k(*base, target=up_target),
                }
                mo.save_variant(*builders[variant](), path=out[variant])
                n_written += 1
            with np.load(out[variant]) as z:
                counts.append(f"{variant}={len(z['V'])}/{len(z['F'])}")
        stats = {"subject": subject, "diameter": diameter, "vectors": vectors, "mean_shift": means,
                 "max_shift": maxima, "crop_draws": crop_draws}
        return "[ok]", f"{subject} wrote={n_written} shift_mean={np.mean(means):.5f} " + " ".join(counts), stats
    except Exception as exc:  # una identita' rotta non deve uccidere il batch
        return "[fail]", f"{subject}: {exc}", {"subject": subject}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="facescape")
    p.add_argument("--identities-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-subjects", type=int, default=0, help="0 = tutte")
    p.add_argument("--n-cores", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--gt-dir", type=Path, default=None,
                   help="GT neutra: aggiunge al manifest lo spostamento relativo al soggetto piu' vicino")
    a = p.parse_args()

    from v3_work.mm import load_model
    m = load_model(a.model)
    m._guard("eval")
    weights = json.loads((a.identities_dir / "identity_weights.json").read_text())
    subjects = sorted(s for s in weights if s.startswith(PREFIX))
    if a.n_subjects:
        subjects = subjects[: a.n_subjects]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(subjects)} identita' x {len(VARIANTS)} topologie, un'espressione per mesh -> {a.out_dir}", flush=True)
    print(f"pool {len(m.expr.pool)}/{m.n_expr} blendshape, {N_ACTIVE_MIN}-{N_ACTIVE_MAX} attivi, coefficienti "
          f"U({COEF_LOW}, {COEF_HIGH}), seme [{a.seed}, identita', topologia]", flush=True)

    # (3n, 52) per identita', float32 (7.9 MB): solo questo passa ai worker
    tasks = ((s, m._identity_core(np.asarray(weights[s])).astype(np.float32), str(a.out_dir), int(a.seed),
              a.overwrite) for s in subjects)
    if a.n_cores > 1:
        pool_p = mp.get_context("spawn").Pool(processes=a.n_cores, initializer=_init, initargs=(m.faces, m.expr))
        results = pool_p.imap_unordered(process_subject, tasks, chunksize=1)
    else:
        _init(m.faces, m.expr)
        pool_p, results = None, map(process_subject, tasks)
    tally, failures, collected = {"[ok]": 0, "[fail]": 0}, [], []
    try:
        for i, (status, msg, stats) in enumerate(results, start=1):
            tally[status] += 1
            if status == "[fail]":
                failures.append(msg)
            else:
                collected.append(stats)
            if status != "[ok]" or i % 100 == 0 or i <= 3:
                print(f"[{i}/{len(subjects)}] {status} {msg}", flush=True)
    finally:
        if pool_p is not None:
            pool_p.close()
            pool_p.join()
    print(f"\nFatto. ok={tally['[ok]']} fail={tally['[fail]']}")
    for msg in failures[:20]:
        print(f"  - {msg}")
    if tally["[fail]"]:
        raise SystemExit(1)

    # Controllo: a espressione zero si deve riottenere la patch neutra di dev_fs_identities.py.
    with np.load(a.identities_dir / f"{subjects[0]}.npz") as z:
        V_id = z["V"].astype(np.float64)
    V_zero = patch(m._identity_core(np.asarray(weights[subjects[0]])), np.zeros(m.n_expr))
    neutral_check = float(np.abs(V_zero - V_id).max())
    if not neutral_check < 1e-4 * float(np.abs(V_id).max()):
        raise SystemExit(f"{subjects[0]}: patch a espressione zero diversa dall'identita' neutra ({neutral_check})")
    collected.sort(key=lambda s: s["subject"])
    (a.out_dir / "expression_vectors.json").write_text(json.dumps({
        "blendshapes": m.expr.names, "pool": [m.expr.names[i] for i in m.expr.pool], "excluded": [],
        "note": "vettori sparsi per (identita', topologia): pesi RESIDUI contro la neutra (toolkit FaceScape); "
                "i blendshape non elencati hanno peso 0",
        "vectors": {s["subject"]: s["vectors"] for s in collected},
    }, indent=1) + "\n")

    mean_shift = np.asarray([s["mean_shift"] for s in collected], dtype=np.float64)
    max_shift = np.asarray([s["max_shift"] for s in collected], dtype=np.float64)
    per_subject = mean_shift.mean(axis=1)
    n_active = [len(v) for s in collected for v in s["vectors"].values()]
    # riferimento "bocca aperta": il blendshape che sposta di piu' la patch media (bs20, 4.4 mm)
    V0m = patch(m._identity_core(np.zeros(m.n_id)), np.zeros(m.n_expr))
    jaw = int(np.argmax([np.linalg.norm(m.expr_basis_at(np.zeros(m.n_id))[:, :, k], axis=1).mean()
                         for k in range(m.n_expr)]))
    w_ref = np.zeros(m.n_expr)
    w_ref[jaw] = 1.0
    d_ref = np.linalg.norm(maxabs_normalize(patch(m._identity_core(np.zeros(m.n_id)), w_ref)) - maxabs_normalize(V0m), axis=1)
    shift = {
        "mean": float(mean_shift.mean()), "sd": float(mean_shift.std()),
        "min": float(mean_shift.min()), "max": float(mean_shift.max()),
        "max_shift_mean": float(max_shift.mean()), "max_shift_max": float(max_shift.max()),
        "between_subject_sd": float(per_subject.std()),
        "between_subject_min": float(per_subject.min()),
        "between_subject_max": float(per_subject.max()),
        "diameter_mean": float(np.mean([s["diameter"] for s in collected])),
        "crop_draws": {"n_redrawn": int(sum(s["crop_draws"] > 1 for s in collected)),
                       "max": int(max(s["crop_draws"] for s in collected))},
        f"{m.expr.names[jaw]}_1.00_mean_identity": {"mean": float(d_ref.mean()), "max": float(d_ref.max())},
        "ict_reference": ICT_REFERENCE,
    }
    if FV_EXPR_MANIFEST.is_file():
        fv = json.loads(FV_EXPR_MANIFEST.read_text())["shift_maxabs"]
        shift["faceverse_reference"] = {k: fv[k] for k in ("mean", "between_subject_sd", "diameter_mean") if k in fv}
    if a.gt_dir is not None:
        nn = nearest_distances(a.gt_dir, PREFIX)
        ratio = mean_shift / np.asarray([nn[s["subject"]] for s in collected])[:, None]
        shift["shift_over_nn"] = {"median": float(np.median(ratio)), "p25": float(np.percentile(ratio, 25)),
                                  "p75": float(np.percentile(ratio, 75)), "mean": float(ratio.mean()),
                                  "nn_median": float(np.median(list(nn.values())))}
    model_file = Path(m.info["file"])
    manifest = {
        "domain": "facescape", "prefix": PREFIX, "role": m.role,
        "model_file": str(model_file), "model_sha256": sha256(model_file),
        "identities_manifest_sha256": sha256(a.identities_dir / "manifest.json"),
        "n_subjects": len(collected), "topologies": list(VARIANTS),
        "expression": {"pool_size": int(len(m.expr.pool)), "excluded": [], "n_active": [N_ACTIVE_MIN, N_ACTIVE_MAX],
                       "coef_uniform": [COEF_LOW, COEF_HIGH], "seed": a.seed,
                       "seed_rule": "default_rng([seed, numero identita', indice topologia in VARIANTS])",
                       "n_active_mean": float(np.mean(n_active)),
                       "weights": "residui contro la neutra (toolkit FaceScape), 51 blendshape senza nomi"},
        "shift_maxabs": shift,
        "neutral_check_max_abs": neutral_check,
        "gt": "neutra: la GT del set neutro (build_zs_gt.py sulle original NEUTRE), non ricalcolata",
    }
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("spostamento maxabs: " + json.dumps(shift), flush=True)


if __name__ == "__main__":
    main()
