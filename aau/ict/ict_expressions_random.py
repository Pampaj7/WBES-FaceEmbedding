#!/usr/bin/env python3
"""WS5 rifatto: un'espressione CASUALE E DIVERSA per ogni identita' held-out ICT.

Perche' esiste, accanto a `v2_work/genict/make_expressions.py`: quello applica lo STESSO
blendshape con lo STESSO coefficiente a tutte le 500 identita', quindi sposta i 100
soggetti valutati tutti insieme e il ranking di identita' non puo' cambiare (misurato:
0.8406 contro 0.8405 della baseline neutra, job 1019698/1019710). Una deformazione di
modo comune non e' il disturbo che WS5 vuole misurare. Qui ogni soggetto riceve K=5
vettori di espressione diversi dai suoi vicini, cioe' un disturbo PER SOGGETTO.

    aau/submit.sh ict/ict_expressions_random.sbatch

Campionamento, per ogni (soggetto, k):
  - pool: i 53 blendshape ufficiali di ICT (`FaceXModel/vertex_indices.json`, chiave
    "expressions"), meno gli 8 `eyeLook*`, che sono la direzione dello SGUARDO, cioe' la
    rotazione di un bulbo oculare che nella regione volto (i primi 9409 vertici) non c'e':
    sulle mesh usate qui muovono 0.0009-0.0017 in media contro 0.0417 di jawOpen. E' una
    esclusione semantica, non per magnitudine -- `eyeWide` resta nel pool e muove ancora
    meno (0.0007) -- ma deforma una palpebra, che al volto appartiene. Restano 45.
  - 3..8 blendshape attivi, estratti senza rimpiazzo
  - coefficiente U(0.3, 1.0) su ognuno (sotto 0.3 il blendshape non si vede)
  - seme `--seed + int(sid[2:])`, cioe' 1234 + 14500..14999: dipende solo dal soggetto,
    non dall'ordine in cui i worker lo pescano, quindi il dataset e' lo stesso a
    qualunque `--n-cores`.

Modello: `V = neutro + sum_i w_i * id_i + sum_s c_s * expr_s`, con i pesi di identita' `w`
riletti da `identity_weights.json` esattamente come fa make_expressions.py: le mesh
espressive sono le STESSE identita' delle neutre, non un campione nuovo.

Uscita in `datasets/ICT/expressions_random/`:
  idNNNNN_rexpr_k.npz      V/F (regione volto, topologia `original`), k = 1..5
  expression_vectors.json  i vettori, in forma sparsa {blendshape: coefficiente}
  manifest.json            configurazione + le statistiche di spostamento qui sotto

Spostamento: misurato con la convenzione del benchmark, cioe' con OGNUNA delle due mesh
normalizzata maxabs per conto suo (centro sulla media, divisione per max|coordinata|) --
e' quello che fa `GTReadyDatasetNPZ` al caricamento, quindi e' la deformazione che il
modello vede davvero, riscalamento globale incluso. Su quella scala il diametro della
mesh e' 2.18 e jawOpen a intensita' 1.00 sposta i vertici di 0.042 in media
(`--demo` lo ricontrolla). Oltre a media e massimo il manifest riporta la DEVIAZIONE
STANDARD TRA SOGGETTI dello spostamento medio: se fosse ~0 il test sarebbe di nuovo un
no-op, perche' vorrebbe dire che tutti i soggetti si muovono della stessa quantita'.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))
from ict_model import MODEL_DIR, N_SHAPE, ict_shape_mesh, load_ict  # noqa: E402
from mesh_ops import save_variant  # noqa: E402

# Blendshape dello sguardo: escluse dal pool, vedi il docstring.
GAZE_SHAPES = tuple(
    f"eyeLook{direction}_{side}"
    for direction in ("Down", "In", "Out", "Up")
    for side in ("L", "R")
)
N_EXPR = 5              # K, quante espressioni per soggetto
N_ACTIVE_MIN = 3        # estremi inclusi
N_ACTIVE_MAX = 8
COEF_LOW = 0.3
COEF_HIGH = 1.0
SEED = 1234

_MODEL: dict | None = None
_POOL: tuple[str, ...] = ()


def maxabs_normalize(V: np.ndarray) -> np.ndarray:
    """Centro sulla media dei vertici, scala sul massimo valore assoluto.

    Terza copia della stessa funzione (`aau/baselines/common.py:146`,
    `aau/recon/ws3b_geometric.py:99`, `v2_work/genict/build_ict_gt_matrix.py:40`): sono
    quattro righe e importarne una vorrebbe dire tirarsi dietro il modulo che la
    contiene, che qui non serve a nient'altro.
    """
    Vc = np.asarray(V, dtype=np.float64)
    Vc = Vc - Vc.mean(axis=0, keepdims=True)
    scale = float(np.abs(Vc).max())
    return Vc / scale if scale > 1e-6 else Vc * 0.0


def blendshape_names(model_dir: Path) -> list[str]:
    """I 53 nomi ufficiali, letti dal modello invece che ricopiati a mano."""
    names = json.loads((Path(model_dir) / "vertex_indices.json").read_text())["expressions"]
    if len(names) != 53:
        raise SystemExit(f"attesi 53 blendshape in vertex_indices.json, trovati {len(names)}")
    return list(names)


def sample_expression(rng: np.random.Generator, n_pool: int) -> tuple[np.ndarray, np.ndarray]:
    """Indici dei blendshape attivi e loro coefficienti, per una sola espressione."""
    n_active = int(rng.integers(N_ACTIVE_MIN, N_ACTIVE_MAX + 1))
    indices = rng.choice(n_pool, size=n_active, replace=False)
    coefficients = rng.uniform(COEF_LOW, COEF_HIGH, size=n_active)
    return indices, coefficients


def _init(model_dir: str, n_shape: int, pool: tuple[str, ...]) -> None:
    global _MODEL, _POOL
    _MODEL = load_ict(model_dir, n_shape=n_shape, expressions=pool)
    _POOL = pool


def process_subject(task: tuple[str, list[float], str, int, bool]) -> tuple[str, str, dict]:
    sid, weights, out_dir_str, seed, overwrite = task
    out_dir = Path(out_dir_str)
    try:
        w = np.asarray(weights, dtype=np.float64)
        V0, F = ict_shape_mesh(w, _MODEL)
        N0 = maxabs_normalize(V0)
        diameter = float(np.linalg.norm(N0.max(axis=0) - N0.min(axis=0)))

        rng = np.random.default_rng(seed + int(sid[2:]))
        vectors: list[dict[str, float]] = []
        means: list[float] = []
        maxima: list[float] = []
        n_written = 0
        for k in range(1, N_EXPR + 1):
            indices, coefficients = sample_expression(rng, len(_POOL))
            vector = {_POOL[int(i)]: float(c) for i, c in zip(indices, coefficients)}
            vectors.append(vector)

            V = V0.copy()
            for name, coefficient in vector.items():
                V += coefficient * _MODEL["exprdirs"][name]
            d = np.linalg.norm(maxabs_normalize(V) - N0, axis=1)
            means.append(float(d.mean()))
            maxima.append(float(d.max()))

            out = out_dir / f"{sid}_rexpr_{k}.npz"
            if not out.exists() or overwrite:
                save_variant(V, F, out)
                n_written += 1
        stats = {"subject": sid, "diameter": diameter, "vectors": vectors,
                 "mean_shift": means, "max_shift": maxima}
        return "[ok]", f"{sid} wrote={n_written} shift_mean={np.mean(means):.5f}", stats
    except Exception as exc:  # una identita' rotta non deve uccidere il batch
        return "[fail]", f"{sid}: {exc}", {"subject": sid}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights-json", type=Path,
                    default=REPO_ROOT / "datasets/ICT/identities/identity_weights.json")
    ap.add_argument("--manifest-json", type=Path,
                    default=REPO_ROOT / "datasets/ICT/train_ready/manifest.json")
    ap.add_argument("--heldout-file", type=Path,
                    default=REPO_ROOT / "datasets/ICT/train_ready/split_heldout.txt")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "datasets/ICT/expressions_random")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--n-shape", type=int, default=N_SHAPE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-cores", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    a = parse_args()

    # L'offset id<->ict sta nel manifest train-ready, non e' una costante di questo script:
    # id14500 e' la mesh di ict4500, ed e' quello il nome che la matrice GT usa.
    id_offset = int(json.loads(a.manifest_json.read_text())["id_offset"])
    weights = json.loads(a.weights_json.read_text())
    heldout = [ln.strip() for ln in a.heldout_file.read_text().splitlines() if ln.strip()]
    if not heldout:
        raise SystemExit(f"no subject ids in {a.heldout_file}")

    names = blendshape_names(a.model_dir)
    pool = tuple(name for name in names if name not in GAZE_SHAPES)
    if len(pool) != len(names) - len(GAZE_SHAPES):
        raise SystemExit(f"nomi dello sguardo non trovati fra i {len(names)} blendshape")

    tasks = []
    for sid in heldout:
        src = f"ict{int(sid[2:]) - id_offset:04d}"
        if src not in weights:
            raise SystemExit(f"{sid} -> {src} has no entry in {a.weights_json}")
        tasks.append((sid, weights[src], str(a.out_dir), int(a.seed), a.overwrite))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    n_files = len(tasks) * N_EXPR
    print(f"{len(tasks)} held-out identities x {N_EXPR} random expressions = {n_files} meshes "
          f"-> {a.out_dir}", flush=True)
    print(f"pool {len(pool)}/{len(names)} blendshape (esclusi {len(GAZE_SHAPES)} eyeLook*), "
          f"{N_ACTIVE_MIN}-{N_ACTIVE_MAX} attivi, coefficienti U({COEF_LOW}, {COEF_HIGH}), "
          f"seme {a.seed}+id", flush=True)

    if a.n_cores > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        p = mp.Pool(processes=a.n_cores, initializer=_init,
                    initargs=(str(a.model_dir), a.n_shape, pool))
        results = p.imap_unordered(process_subject, tasks)
    else:
        _init(str(a.model_dir), a.n_shape, pool)
        p, results = None, map(process_subject, tasks)

    tally = {"[ok]": 0, "[fail]": 0}
    failures = []
    collected: list[dict] = []
    try:
        for i, (status, msg, stats) in enumerate(results, start=1):
            tally[status] += 1
            if status == "[fail]":
                failures.append(msg)
            else:
                collected.append(stats)
            if status != "[ok]" or i % 50 == 0 or i <= 3:
                print(f"[{i}/{len(tasks)}] {status} {msg}", flush=True)
    finally:
        if p is not None:
            p.close()
            p.join()

    collected.sort(key=lambda s: s["subject"])
    (a.out_dir / "expression_vectors.json").write_text(json.dumps({
        "blendshapes": names,
        "pool": list(pool),
        "excluded": list(GAZE_SHAPES),
        "note": "vettori sparsi: i blendshape non elencati hanno coefficiente 0",
        "vectors": {s["subject"]: s["vectors"] for s in collected},
    }, indent=2) + "\n")

    mean_shift = np.asarray([s["mean_shift"] for s in collected], dtype=np.float64)
    max_shift = np.asarray([s["max_shift"] for s in collected], dtype=np.float64)
    per_subject = mean_shift.mean(axis=1)
    n_active = [len(v) for s in collected for v in s["vectors"]]
    shift = {
        # su tutte le n_soggetti x K mesh
        "mean": float(mean_shift.mean()),
        "sd": float(mean_shift.std()),
        "min": float(mean_shift.min()),
        "max": float(mean_shift.max()),
        "max_shift_mean": float(max_shift.mean()),
        "max_shift_max": float(max_shift.max()),
        # la riga che dice se il test e' un no-op: 0 = tutti i soggetti si muovono uguale
        "between_subject_sd": float(per_subject.std()),
        "between_subject_min": float(per_subject.min()),
        "between_subject_max": float(per_subject.max()),
        "diameter_mean": float(np.mean([s["diameter"] for s in collected])),
        "reference_jawOpen_1.00_mean": 0.042,
        "reference_jawOpen_1.00_max": 0.215,
    }
    (a.out_dir / "manifest.json").write_text(json.dumps({
        "n_subjects": len(collected),
        "n_expressions_per_subject": N_EXPR,
        "n_meshes": n_files,
        "topology": "original",
        "id_offset": id_offset,
        "seed": int(a.seed),
        "seed_rule": "np.random.default_rng(seed + int(sid[2:]))",
        "n_active_range": [N_ACTIVE_MIN, N_ACTIVE_MAX],
        "coefficient_range": [COEF_LOW, COEF_HIGH],
        "n_blendshapes": len(names),
        "n_pool": len(pool),
        "excluded_blendshapes": list(GAZE_SHAPES),
        "source_weights": str(a.weights_json),
        "name_pattern": "idNNNNN_rexpr_<k>.npz",
        "shift_normalization": "maxabs per mesh (GTReadyDatasetNPZ), diametro ~2.18",
        "n_active_mean": float(np.mean(n_active)),
        "vertex_shift": shift,
    }, indent=2) + "\n")

    print(f"\nDone. ok={tally['[ok]']} fail={tally['[fail]']}")
    for msg in failures[:20]:
        print(f"  - {msg}")
    print(f"blendshape attivi per espressione: media {np.mean(n_active):.2f}")
    print(f"spostamento medio  {shift['mean']:.5f} (sd {shift['sd']:.5f}, "
          f"min {shift['min']:.5f}, max {shift['max']:.5f})")
    print(f"spostamento massimo {shift['max_shift_mean']:.5f} in media, "
          f"{shift['max_shift_max']:.5f} il peggiore")
    print(f"riferimento jawOpen@1.00: medio 0.042, massimo 0.215, diametro 2.18 "
          f"(qui {shift['diameter_mean']:.3f})")
    print(f"TRA SOGGETTI dello spostamento medio: sd {shift['between_subject_sd']:.5f} "
          f"({100.0 * shift['between_subject_sd'] / max(shift['mean'], 1e-12):.1f}% della media), "
          f"da {shift['between_subject_min']:.5f} a {shift['between_subject_max']:.5f}")
    if shift["between_subject_sd"] < 0.01 * shift["mean"]:
        raise SystemExit("VARIANZA TRA SOGGETTI NULLA: il disturbo e' di modo comune, come prima")
    if tally["[fail]"]:
        raise SystemExit(1)


def demo() -> None:
    """Ricontrolla la scala: jawOpen@1.00 su ict4500 deve dare 0.042 / 0.215, diametro 2.18."""
    names = blendshape_names(MODEL_DIR)
    pool = tuple(name for name in names if name not in GAZE_SHAPES)
    _init(str(MODEL_DIR), N_SHAPE, tuple(names))
    weights = json.loads((REPO_ROOT / "datasets/ICT/identities/identity_weights.json").read_text())
    V0, _ = ict_shape_mesh(np.asarray(weights["ict4500"]), _MODEL)
    N0 = maxabs_normalize(V0)
    diameter = float(np.linalg.norm(N0.max(axis=0) - N0.min(axis=0)))
    print(f"ict4500 (= id14500) diametro maxabs-normalizzato = {diameter:.3f}")
    assert abs(diameter - 2.180) < 5e-3, diameter

    for name, shapes in (("jawOpen", ("jawOpen",)), ("mouthSmile", ("mouthSmile_L", "mouthSmile_R"))):
        V = V0 + sum(_MODEL["exprdirs"][s] for s in shapes)
        d = np.linalg.norm(maxabs_normalize(V) - N0, axis=1)
        print(f"  {name}@1.00: medio {d.mean():.5f} massimo {d.max():.5f}")
    V = V0 + _MODEL["exprdirs"]["jawOpen"]
    d = np.linalg.norm(maxabs_normalize(V) - N0, axis=1)
    assert abs(float(d.mean()) - 0.042) < 1e-3, d.mean()

    # Gli 8 eyeLook* devono restare trascurabili sulla regione volto: sotto il 5% di jawOpen.
    worst_gaze = max(
        float(np.linalg.norm(maxabs_normalize(V0 + _MODEL["exprdirs"][s]) - N0, axis=1).mean())
        for s in GAZE_SHAPES
    )
    weakest_pool = min(
        float(np.linalg.norm(maxabs_normalize(V0 + _MODEL["exprdirs"][s]) - N0, axis=1).mean())
        for s in pool
    )
    print(f"  eyeLook* peggiore {worst_gaze:.5f}, blendshape del pool piu' debole {weakest_pool:.5f}")
    assert worst_gaze < 0.05 * 0.042, worst_gaze

    # Due soggetti vicini devono ricevere espressioni diverse: e' tutto il punto del file.
    a = np.random.default_rng(SEED + 14500)
    b = np.random.default_rng(SEED + 14501)
    ia, ca = sample_expression(a, len(pool))
    ib, cb = sample_expression(b, len(pool))
    print(f"  id14500 k=1: {[pool[int(i)] for i in ia]}")
    print(f"  id14501 k=1: {[pool[int(i)] for i in ib]}")
    assert set(ia.tolist()) != set(ib.tolist()) or not np.allclose(ca, cb)
    print("demo OK: scala, esclusioni e indipendenza fra soggetti verificate")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
