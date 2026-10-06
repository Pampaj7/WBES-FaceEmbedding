#!/usr/bin/env python3
"""Le 6 topologie con un'espressione CASUALE PER MESH, sulle identita' di un dominio zero-shot (FaceVerse).

    aau/run.sh aau/zs3dmm/make_zs_expr_topologies.py --domain fv --model-file <faceverse_simple_v2.npy> \\
        --identities-dir datasets/FACEVERSE_ZS/identities --out-dir datasets/FACEVERSE_ZS/expr_topo --n-cores 32
    (zs_build_expr.sbatch, WBES_ZS_EXPR=1)

Test della GT d'identita': la GT resta quella delle forme NEUTRE (``build_zs_gt.py`` sulle
``original`` neutre, riusata com'e'), mentre ogni mesh in ingresso ai confronti porta
un'espressione propria. Per ogni (identita', topologia) si estrae un vettore d'espressione
indipendente, si genera la testa ``V = forma(w) + exBase @ c`` con gli STESSI pesi d'identita'
``w`` di ``identities/identity_weights.json``, si prende la patch del volto e se ne ricava SOLO
quella topologia, con le regole di ``v2_work/genict/make_ict_topologies.py`` (importate, non
riscritte: ``make_remesh``, ``make_crop``, ``make_noisy``, ``make_down8k``, ``make_up60k``, stessi
target di triangoli). Due mesh confrontate hanno quindi sempre espressioni diverse, anche fra
topologie dello stesso soggetto.

Campionamento (la ricetta di ``aau/ict/ict_expressions_random.py``, WS5 su ICT, trasportata):
  - pool: i 52 blendshape ARKit di FaceVerse (``exp_name_list``) meno gli 8 ``eyeLook*``
    (direzione dello sguardo, esclusione semantica come su ICT): restano 44;
  - 3..8 blendshape attivi, senza rimpiazzo; coefficiente U(0.3, 1.0) su ognuno. I coefficienti
    ARKit hanno [0, 1] come intervallo semantico (0 = neutro, 1 = espressione piena): l'intervallo
    e' dentro quello del modello, e sotto 0.3 il blendshape quasi non si vede;
  - seme ``[--seed, numero dell'identita', indice della topologia]``: dipende solo da
    (identita', topologia), non dall'ordine dei worker;
  - solo per il ``crop``: se con l'espressione estratta il crop di ``mesh_ops`` non toglie niente
    (succede con la bocca molto aperta: la banda supererebbe il 15% dei vertici) l'espressione si
    riestrae dallo stesso generatore, fino a 20 volte. Sposta la distribuzione delle espressioni
    del solo crop, che e' declassato e riportato a parte; il manifest conta le riestrazioni.

Spostamento: con la convenzione del benchmark (ognuna delle due mesh normalizzata maxabs per
conto suo), patch con espressione contro patch neutra della stessa identita', sulla topologia
d'origine (corrispondenza densa). Il manifest riporta media, massimo, deviazione standard TRA
SOGGETTI (se fosse ~0 il disturbo sarebbe di modo comune, il difetto del primo WS5) e il
riferimento jawOpen a 1.0, accanto ai numeri di ICT (0.017 medio con questa ricetta, 0.042
jawOpen a 1.0, diametro 2.18).

Output: ``<prefix>NNNN_GTready_<topologia>.npz`` (V/F, come ``make_zs_topologies.py``),
``expression_vectors.json`` (vettori sparsi per identita' e topologia), ``manifest.json``.
Riprendibile: salta le identita' con tutte e 6 le topologie, a meno di ``--overwrite``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
# A livello di modulo: con lo start method spawn i worker rieseguono questo file.
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))

import mesh_ops as mo  # noqa: E402
from make_ict_topologies import (  # noqa: E402
    VARIANTS,
    make_down8k,
    make_noisy,
    make_remesh,
    make_up60k,
    triangle_targets,
)

GAZE_PREFIX = "eyeLook"   # eyeLook{Down,In,Out,Up}{Left,Right}: 8 blendshape, esclusi dal pool
N_ACTIVE_MIN = 3          # estremi inclusi
N_ACTIVE_MAX = 8
COEF_LOW = 0.3
COEF_HIGH = 1.0
MAX_CROP_DRAWS = 20       # riestrazioni dell'espressione del crop quando il crop non toglie niente
ICT_REFERENCE = {"recipe_mean_shift": 0.017, "jawOpen_1.00_mean": 0.042, "diameter": 2.18}

_MODEL: dict | None = None
_EXPR: dict | None = None
_POOL: tuple[int, ...] = ()


def maxabs_normalize(V: np.ndarray) -> np.ndarray:
    """Centro sulla media, divisione per max|coordinata| (``GTReadyDatasetNPZ``, ``ict_expressions_random``)."""
    Vc = np.asarray(V, dtype=np.float64)
    Vc = Vc - Vc.mean(axis=0, keepdims=True)
    scale = float(np.abs(Vc).max())
    return Vc / scale if scale > 1e-6 else Vc * 0.0


def sample_expression(rng: np.random.Generator, n_pool: int) -> tuple[np.ndarray, np.ndarray]:
    """Indici (nel pool) dei blendshape attivi e loro coefficienti, per una sola espressione."""
    n_active = int(rng.integers(N_ACTIVE_MIN, N_ACTIVE_MAX + 1))
    indices = rng.choice(n_pool, size=n_active, replace=False)
    coefficients = rng.uniform(COEF_LOW, COEF_HIGH, size=n_active)
    return indices, coefficients


def _load(domain: str, model_file: str) -> tuple[dict, dict]:
    if domain != "fv":
        raise SystemExit(f"--domain {domain}: base d'espressione disponibile solo per fv (FaceVerse v2)")
    import fv_model
    return fv_model.load_fv(model_file), fv_model.load_fv_expressions(model_file)


def _init(domain: str, model_file: str, pool: tuple[int, ...]) -> None:
    global _MODEL, _EXPR, _POOL
    _MODEL, _EXPR = _load(domain, model_file)
    _POOL = pool


def expressed_patch(w: np.ndarray, coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Patch del volto (V, F) per identita' ``w`` ed espressione densa ``coef`` (52,)."""
    from hifi_model import face_patch, hifi_shape_mesh
    V_head = hifi_shape_mesh(w, _MODEL) + _EXPR["exprdirs"] @ coef
    return face_patch(V_head, _MODEL)


def process_subject(task: tuple[str, list[float], str, int, bool]) -> tuple[str, str, dict]:
    subject, weights, out_dir_str, seed, overwrite = task
    out_dir = Path(out_dir_str)
    out = {v: out_dir / f"{subject}_GTready_{v}.npz" for v in VARIANTS}
    num = int("".join(ch for ch in subject if ch.isdigit()))
    try:
        w = np.asarray(weights, dtype=np.float64)
        n_expr = _EXPR["exprdirs"].shape[2]
        V0, F = expressed_patch(w, np.zeros(n_expr))
        N0 = maxabs_normalize(V0)
        diameter = float(np.linalg.norm(N0.max(axis=0) - N0.min(axis=0)))
        vectors, means, maxima, counts, crop_draws = {}, [], [], [], 0
        n_written = 0
        for t_idx, variant in enumerate(VARIANTS):
            rng = np.random.default_rng([seed, num, t_idx])
            built = None
            for draw in range(1, MAX_CROP_DRAWS + 1):
                indices, coefficients = sample_expression(rng, len(_POOL))
                coef = np.zeros(n_expr)
                coef[[_POOL[int(i)] for i in indices]] = coefficients
                V, Fp = expressed_patch(w, coef)
                base = mo.prepare_open_surface(*mo.as_arrays(V, Fp))
                if len(base[1]) != len(Fp):
                    return "[fail]", f"{subject}/{variant}: la patch non e' una superficie pulita", {"subject": subject}
                if variant != "crop":
                    break
                # Il crop di mesh_ops torna la patch INTATTA se toglierebbe oltre il 15% dei
                # vertici (zs_check_crop.py), e con alcune espressioni (bocca molto aperta) lo
                # fa. Qui non lo si vede confrontando con la original, che ha un'altra
                # espressione: si controlla il conteggio e si riestrae l'espressione del solo
                # crop, dallo stesso generatore (deterministico). Il crop e' riportato a parte.
                built = mo.make_crop(*base)
                if len(built[0]) != len(base[0]):
                    crop_draws = draw
                    break
            else:
                return "[fail]", f"{subject}/crop: crop identico alla patch intera per {MAX_CROP_DRAWS} espressioni", \
                    {"subject": subject}
            vectors[variant] = {_EXPR["names"][_POOL[int(i)]]: float(c) for i, c in zip(indices, coefficients)}
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
        stats = {"subject": subject, "diameter": diameter, "vectors": vectors,
                 "mean_shift": means, "max_shift": maxima, "crop_draws": crop_draws}
        return "[ok]", f"{subject} wrote={n_written} shift_mean={np.mean(means):.5f} " + " ".join(counts), stats
    except Exception as exc:  # una identita' rotta non deve uccidere il batch
        return "[fail]", f"{subject}: {exc}", {"subject": subject}


def jaw_reference(weights: list[float]) -> dict:
    """jawOpen a 1.0 sulla prima identita', stessa misura dello spostamento (``--demo`` di ICT)."""
    w = np.asarray(weights, dtype=np.float64)
    n_expr = _EXPR["exprdirs"].shape[2]
    V0, _ = expressed_patch(w, np.zeros(n_expr))
    coef = np.zeros(n_expr)
    coef[_EXPR["names"].index("jawOpen")] = 1.0
    V1, _ = expressed_patch(w, coef)
    d = np.linalg.norm(maxabs_normalize(V1) - maxabs_normalize(V0), axis=1)
    return {"mean": float(d.mean()), "max": float(d.max())}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--domain", required=True)
    p.add_argument("--model-file", type=Path, required=True)
    p.add_argument("--identities-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="fv")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-subjects", type=int, default=0, help="0 = tutte")
    p.add_argument("--n-cores", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    _, expr = _load(a.domain, str(a.model_file))  # fallisce prima di scrivere qualunque cosa
    names = expr["names"]
    excluded = [n for n in names if n.startswith(GAZE_PREFIX)]
    if len(excluded) != 8:
        raise SystemExit(f"attesi 8 blendshape {GAZE_PREFIX}*, trovati {excluded}")
    pool = tuple(i for i, n in enumerate(names) if not n.startswith(GAZE_PREFIX))

    weights = json.loads((a.identities_dir / "identity_weights.json").read_text())
    subjects = sorted(s for s in weights if s.startswith(a.prefix))
    if a.n_subjects:
        subjects = subjects[: a.n_subjects]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(s, weights[s], str(a.out_dir), int(a.seed), a.overwrite) for s in subjects]
    print(f"{len(tasks)} identita' x {len(VARIANTS)} topologie, un'espressione per mesh -> {a.out_dir}", flush=True)
    print(f"pool {len(pool)}/{len(names)} blendshape (esclusi {len(excluded)} {GAZE_PREFIX}*), "
          f"{N_ACTIVE_MIN}-{N_ACTIVE_MAX} attivi, coefficienti U({COEF_LOW}, {COEF_HIGH}), "
          f"seme [{a.seed}, identita', topologia]", flush=True)

    if a.n_cores > 1:
        pool_p = mp.get_context("spawn").Pool(processes=a.n_cores, initializer=_init,
                                              initargs=(a.domain, str(a.model_file), pool))
        results = pool_p.imap_unordered(process_subject, tasks)
    else:
        _init(a.domain, str(a.model_file), pool)
        pool_p, results = None, map(process_subject, tasks)

    tally = {"[ok]": 0, "[fail]": 0}
    failures, collected = [], []
    try:
        for i, (status, msg, stats) in enumerate(results, start=1):
            tally[status] += 1
            if status == "[fail]":
                failures.append(msg)
            else:
                collected.append(stats)
            if status != "[ok]" or i % 100 == 0 or i <= 3:
                print(f"[{i}/{len(tasks)}] {status} {msg}", flush=True)
    finally:
        if pool_p is not None:
            pool_p.close()
            pool_p.join()
    print(f"\nFatto. ok={tally['[ok]']} fail={tally['[fail]']}")
    for msg in failures[:20]:
        print(f"  - {msg}")
    if tally["[fail]"]:
        raise SystemExit(1)

    _init(a.domain, str(a.model_file), pool)
    # Controllo: a espressione zero si deve riottenere la patch neutra di zs_identities.py.
    with np.load(a.identities_dir / f"{subjects[0]}.npz") as z:
        V_id = z["V"].astype(np.float64)
    V_zero, _ = expressed_patch(np.asarray(weights[subjects[0]]), np.zeros(_EXPR["exprdirs"].shape[2]))
    neutral_check = float(np.abs(V_zero - V_id).max())
    if not neutral_check < 1e-4 * float(np.abs(V_id).max()):
        raise SystemExit(f"{subjects[0]}: patch a espressione zero diversa dall'identita' neutra ({neutral_check})")
    collected.sort(key=lambda s: s["subject"])
    (a.out_dir / "expression_vectors.json").write_text(json.dumps({
        "blendshapes": names,
        "pool": [names[i] for i in pool],
        "excluded": excluded,
        "note": "vettori sparsi per (identita', topologia): i blendshape non elencati hanno coefficiente 0",
        "vectors": {s["subject"]: s["vectors"] for s in collected},
    }, indent=1) + "\n")

    mean_shift = np.asarray([s["mean_shift"] for s in collected], dtype=np.float64)
    max_shift = np.asarray([s["max_shift"] for s in collected], dtype=np.float64)
    per_subject = mean_shift.mean(axis=1)
    n_active = [len(v) for s in collected for v in s["vectors"].values()]
    shift = {
        "mean": float(mean_shift.mean()), "sd": float(mean_shift.std()),
        "min": float(mean_shift.min()), "max": float(mean_shift.max()),
        "max_shift_mean": float(max_shift.mean()), "max_shift_max": float(max_shift.max()),
        # 0 = tutti i soggetti si muovono uguale: il disturbo non toccherebbe il ranking
        "between_subject_sd": float(per_subject.std()),
        "between_subject_min": float(per_subject.min()),
        "between_subject_max": float(per_subject.max()),
        "diameter_mean": float(np.mean([s["diameter"] for s in collected])),
        "crop_draws": {"n_redrawn": int(sum(s["crop_draws"] > 1 for s in collected)),
                       "max": int(max(s["crop_draws"] for s in collected))},
        "jawOpen_1.00": jaw_reference(weights[subjects[0]]),
        "ict_reference": ICT_REFERENCE,
    }
    manifest = {
        "domain": a.domain, "prefix": a.prefix,
        "model_file": str(a.model_file), "model_sha256": sha256(a.model_file),
        "identities_manifest_sha256": sha256(a.identities_dir / "manifest.json"),
        "n_subjects": len(collected), "topologies": list(VARIANTS),
        "expression": {"pool_size": len(pool), "excluded": excluded, "n_active": [N_ACTIVE_MIN, N_ACTIVE_MAX],
                       "coef_uniform": [COEF_LOW, COEF_HIGH], "seed": a.seed,
                       "seed_rule": "default_rng([seed, numero identita', indice topologia in VARIANTS])",
                       "n_active_mean": float(np.mean(n_active))},
        "shift_maxabs": shift,
        "neutral_check_max_abs": neutral_check,
        "gt": "neutra: la GT dello zero-shot (build_zs_gt.py sulle original NEUTRE), non ricalcolata",
    }
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("spostamento maxabs: " + json.dumps(shift), flush=True)


if __name__ == "__main__":
    main()
