#!/usr/bin/env python3
"""Le 6 topologie con un'espressione CASUALE PER MESH, sulle identita' di un dominio zero-shot (FaceVerse, GNM).

    aau/run.sh aau/zs3dmm/make_zs_expr_topologies.py --domain fv --model-file <faceverse_simple_v2.npy> \\
        --identities-dir datasets/FACEVERSE_ZS/identities --out-dir datasets/FACEVERSE_ZS/expr_topo --n-cores 32
    aau/run.sh aau/zs3dmm/make_zs_expr_topologies.py --domain gnm --prefix gnm --model-file <gnm_head.npz> \\
        --identities-dir datasets/GNM_ZS/identities --out-dir datasets/GNM_ZS/expr_topo --gt-dir datasets/GNM_ZS/gt ...
    (zs_build_expr.sbatch, WBES_ZS_EXPR=1)

Test della GT d'identita': la GT resta quella delle forme NEUTRE (``build_zs_gt.py`` sulle
``original`` neutre, riusata com'e'), mentre ogni mesh in ingresso ai confronti porta
un'espressione propria. Per ogni (identita', topologia) si estrae un vettore d'espressione
indipendente, si genera la testa ``V = forma(w) + exBase @ c`` (GNM: ``expression_basis``, in posa
neutra) con gli STESSI pesi d'identita'
``w`` di ``identities/identity_weights.json``, si prende la patch del volto e se ne ricava SOLO
quella topologia, con le regole di ``v2_work/genict/make_ict_topologies.py`` (importate, non
riscritte: ``make_remesh``, ``make_crop``, ``make_noisy``, ``make_down8k``, ``make_up60k``, stessi
target di triangoli). Due mesh confrontate hanno quindi sempre espressioni diverse, anche fra
topologie dello stesso soggetto.

Campionamento FaceVerse (la ricetta di ``aau/ict/ict_expressions_random.py``, WS5 su ICT, trasportata):
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

Campionamento GNM (base d'espressione PCA per regioni, non blendshape ARKit: niente sottoinsieme
attivo ne' intervallo [0, 1]):
  - pool: le 350 componenti ``lower_face_region`` (150), ``left_eye_region`` e
    ``right_eye_region`` (100 + 100); a zero ``tongue`` (32) e ``pupils`` (1), fuori dalla
    regione del volto;
  - tutte attive, coefficienti i.i.d. N(0, ``GNM_EXPR_SIGMA``^2): il prior della base (README:
    intervallo tipico -3..3) a un'ampiezza fissata. ``GNM_EXPR_SIGMA`` e' tarato UNA volta, prima
    di qualunque eval, perche' l'ampiezza relativa sia quella del benchmark FaceVerse: mediana per
    mesh di (spostamento medio / distanza GT dal soggetto piu' vicino) = ``FV_REFERENCE`` (vedi la
    costante); stesso seme, stessa regola di riestrazione del crop.

Spostamento: con la convenzione del benchmark (ognuna delle due mesh normalizzata maxabs per
conto suo), patch con espressione contro patch neutra della stessa identita', sulla topologia
d'origine (corrispondenza densa). Il manifest riporta media, massimo, deviazione standard TRA
SOGGETTI (se fosse ~0 il disturbo sarebbe di modo comune, il difetto del primo WS5) e il
riferimento jawOpen a 1.0 (GNM: ``lower_face_region_000`` a 3), accanto ai numeri di ICT (0.017
medio con questa ricetta, 0.042 jawOpen a 1.0, diametro 2.18). Con ``--gt-dir`` anche il rapporto
fra lo spostamento di ogni mesh e la distanza GT (maxabs, non normalizzata) del suo soggetto dal
soggetto piu' vicino del pool, la misura con cui si confrontano le ampiezze fra domini.

Output: ``<prefix>NNNN_GTready_<topologia>.npz`` (V/F, come ``make_zs_topologies.py``),
``expression_vectors.json`` (vettori sparsi per identita' e topologia; GNM: vettori densi del pool
in ``expression_vectors.npz``), ``manifest.json``.
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

GNM_POOL_GROUPS = ("lower_face_region", "left_eye_region", "right_eye_region")
GNM_EXPR_SIGMA = None      # tarato su FV_REFERENCE, vedi sotto
# Riferimento FaceVerse (expr_topo della build 1056xxx, ricalcolato da aau/scratch/gnm/calib.py):
# mediana per mesh di spostamento / distanza GT dal soggetto piu' vicino.
FV_REFERENCE = {"shift_over_nn_median": None}
# Espressione di riferimento per dominio (il "jawOpen a 1.0" del manifest): nome, coefficiente.
REFERENCE_EXPRESSION = {"fv": ("jawOpen", 1.0), "gnm": ("lower_face_region_000", 3.0)}

_MODEL: dict | None = None
_EXPR: dict | None = None
_POOL: tuple[int, ...] = ()
_DOMAIN = ""


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


def sample_expression_gnm(rng: np.random.Generator, n_pool: int) -> tuple[np.ndarray, np.ndarray]:
    """GNM: tutto il pool attivo, coefficienti N(0, GNM_EXPR_SIGMA^2)."""
    return np.arange(n_pool), rng.normal(0.0, GNM_EXPR_SIGMA, size=n_pool)


def _load(domain: str, model_file: str) -> tuple[dict, dict]:
    if domain == "fv":
        import fv_model
        return fv_model.load_fv(model_file), fv_model.load_fv_expressions(model_file)
    if domain == "gnm":
        import gnm_model
        return gnm_model.load_gnm(model_file), gnm_model.load_gnm_expressions(model_file)
    raise SystemExit(f"--domain {domain}: base d'espressione disponibile solo per fv (FaceVerse v2) e gnm (GNM Head)")


def expression_pool(domain: str, expr: dict) -> tuple[tuple[int, ...], list[str]]:
    """Indici del pool nella base d'espressione e nomi esclusi."""
    names = expr["names"]
    if domain == "gnm":
        pool = tuple(i for g in GNM_POOL_GROUPS for i in expr["groups"][g])
        if len(pool) != 350:
            raise SystemExit(f"attese 350 componenti {GNM_POOL_GROUPS}, trovate {len(pool)}")
        return pool, [n for i, n in enumerate(names) if i not in set(pool)]
    excluded = [n for n in names if n.startswith(GAZE_PREFIX)]
    if len(excluded) != 8:
        raise SystemExit(f"attesi 8 blendshape {GAZE_PREFIX}*, trovati {excluded}")
    return tuple(i for i, n in enumerate(names) if not n.startswith(GAZE_PREFIX)), excluded


def _init(domain: str, model_file: str, pool: tuple[int, ...]) -> None:
    global _MODEL, _EXPR, _POOL, _DOMAIN
    _MODEL, _EXPR = _load(domain, model_file)
    _POOL = pool
    _DOMAIN = domain


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
            sample = sample_expression_gnm if _DOMAIN == "gnm" else sample_expression
            for draw in range(1, MAX_CROP_DRAWS + 1):
                indices, coefficients = sample(rng, len(_POOL))
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
            if _DOMAIN == "gnm":  # denso, nell'ordine del pool
                vectors[variant] = [float(c) for c in coefficients]
            else:
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
    """jawOpen a 1.0 (GNM: ``REFERENCE_EXPRESSION``) sulla prima identita', stessa misura dello
    spostamento (``--demo`` di ICT)."""
    w = np.asarray(weights, dtype=np.float64)
    n_expr = _EXPR["exprdirs"].shape[2]
    V0, _ = expressed_patch(w, np.zeros(n_expr))
    coef = np.zeros(n_expr)
    name, value = REFERENCE_EXPRESSION[_DOMAIN]
    coef[_EXPR["names"].index(name)] = value
    V1, _ = expressed_patch(w, coef)
    d = np.linalg.norm(maxabs_normalize(V1) - maxabs_normalize(V0), axis=1)
    return {"mean": float(d.mean()), "max": float(d.max())}


def nearest_distances(gt_dir: Path, prefix: str) -> dict[str, float]:
    """Per soggetto, la distanza GT (vertex-mean-L2 maxabs, NON divisa per il massimo) dal piu' vicino."""
    with np.load(gt_dir / f"{prefix}_matrix_distances_maxabs.npz") as z:
        D = z["D_orig"].astype(np.float64)
        names = [str(n) for n in z["names"]]
    D = D * json.loads((gt_dir / "manifest.json").read_text())["normalization_scale"]["maxabs"]
    np.fill_diagonal(D, np.inf)
    return dict(zip(names, D.min(axis=1)))


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
    p.add_argument("--gt-dir", type=Path, default=None,
                   help="GT neutra del dominio: aggiunge al manifest lo spostamento relativo al soggetto piu' vicino")
    a = p.parse_args()

    if a.domain == "gnm" and GNM_EXPR_SIGMA is None:
        raise SystemExit("GNM_EXPR_SIGMA non ancora tarato")
    _, expr = _load(a.domain, str(a.model_file))  # fallisce prima di scrivere qualunque cosa
    names = expr["names"]
    pool, excluded = expression_pool(a.domain, expr)

    weights = json.loads((a.identities_dir / "identity_weights.json").read_text())
    subjects = sorted(s for s in weights if s.startswith(a.prefix))
    if a.n_subjects:
        subjects = subjects[: a.n_subjects]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(s, weights[s], str(a.out_dir), int(a.seed), a.overwrite) for s in subjects]
    print(f"{len(tasks)} identita' x {len(VARIANTS)} topologie, un'espressione per mesh -> {a.out_dir}", flush=True)
    if a.domain == "gnm":
        print(f"pool {len(pool)}/{len(names)} componenti {GNM_POOL_GROUPS}, tutte attive, coefficienti "
              f"N(0, {GNM_EXPR_SIGMA}^2), seme [{a.seed}, identita', topologia]", flush=True)
    else:
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
    if a.domain == "gnm":
        # 500 x 6 x 350 coefficienti: in json sarebbero decine di MB.
        np.savez_compressed(a.out_dir / "expression_vectors.npz",
                            subjects=np.array([s["subject"] for s in collected]), topologies=np.array(VARIANTS),
                            pool=np.array([names[i] for i in pool]),
                            coef=np.asarray([[s["vectors"][v] for v in VARIANTS] for s in collected]))
        vectors_note = ("vettori densi del pool in expression_vectors.npz: coef[soggetto, topologia, componente], "
                        "le componenti fuori dal pool hanno coefficiente 0")
        vectors_json = None
    else:
        vectors_note = "vettori sparsi per (identita', topologia): i blendshape non elencati hanno coefficiente 0"
        vectors_json = {s["subject"]: s["vectors"] for s in collected}
    (a.out_dir / "expression_vectors.json").write_text(json.dumps({
        "blendshapes": names,
        "pool": [names[i] for i in pool],
        "excluded": excluded,
        "note": vectors_note,
        "vectors": vectors_json,
    }, indent=1) + "\n")

    mean_shift = np.asarray([s["mean_shift"] for s in collected], dtype=np.float64)
    max_shift = np.asarray([s["max_shift"] for s in collected], dtype=np.float64)
    per_subject = mean_shift.mean(axis=1)
    n_active = [len(v) for s in collected for v in s["vectors"].values()]
    ref_name, ref_value = REFERENCE_EXPRESSION[a.domain]
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
        f"{ref_name}_{ref_value:.2f}": jaw_reference(weights[subjects[0]]),
        "ict_reference": ICT_REFERENCE,
    }
    seed_rule = "default_rng([seed, numero identita', indice topologia in VARIANTS])"
    if a.domain == "gnm":
        recipe = {"pool_size": len(pool), "pool_groups": list(GNM_POOL_GROUPS), "excluded": excluded,
                  "coef_normal_sigma": GNM_EXPR_SIGMA, "all_active": True, "seed": a.seed, "seed_rule": seed_rule,
                  "sigma_calibration": f"mediana spostamento / distanza dal piu' vicino = FaceVerse {FV_REFERENCE}"}
    else:
        recipe = {"pool_size": len(pool), "excluded": excluded, "n_active": [N_ACTIVE_MIN, N_ACTIVE_MAX],
                  "coef_uniform": [COEF_LOW, COEF_HIGH], "seed": a.seed, "seed_rule": seed_rule,
                  "n_active_mean": float(np.mean(n_active))}
    if a.gt_dir is not None:
        nn = nearest_distances(a.gt_dir, a.prefix)
        ratio = mean_shift / np.asarray([nn[s["subject"]] for s in collected])[:, None]
        shift["shift_over_nn"] = {"median": float(np.median(ratio)), "p25": float(np.percentile(ratio, 25)),
                                  "p75": float(np.percentile(ratio, 75)), "mean": float(ratio.mean()),
                                  "nn_median": float(np.median(list(nn.values()))),
                                  "fv_reference": FV_REFERENCE}
    manifest = {
        "domain": a.domain, "prefix": a.prefix,
        "model_file": str(a.model_file), "model_sha256": sha256(a.model_file),
        "identities_manifest_sha256": sha256(a.identities_dir / "manifest.json"),
        "n_subjects": len(collected), "topologies": list(VARIANTS),
        "expression": recipe,
        "shift_maxabs": shift,
        "neutral_check_max_abs": neutral_check,
        "gt": "neutra: la GT dello zero-shot (build_zs_gt.py sulle original NEUTRE), non ricalcolata",
    }
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("spostamento maxabs: " + json.dumps(shift), flush=True)


if __name__ == "__main__":
    main()
