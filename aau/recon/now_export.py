#!/usr/bin/env python3
"""Esporta le ricostruzioni nel layout del codice NoW ufficiale.

    aau/run.sh aau/recon/now_export.py [--methods 3ddfa_v2,synergynet,prnet]

Per ogni immagine e metodo scrive, sotto ``now_common.pred_dir(metodo)``,

    <soggetto>/<sfida>/<IMG>.obj    la mesh INTERA del metodo, y negata (destrorsa)
    <soggetto>/<sfida>/<IMG>.npy    (7, 3) i landmark NoW sulla stessa mesh

cioe' quello che ``compute_error.py`` cerca (README di NoW, "Output structure").  Le unita'
restano i pixel del metodo: il protocollo stima la scala con la Procrustes dai landmark.
Si nega la y anche se la Procrustes ufficiale ammette la riflessione (``reflection='best'``):
il raffinamento successivo e' una rotazione pura, e la mesh deve arrivarci gia' destrorsa,
come nel resto del cantiere.  Nessuna pulizia della mesh: e' quella del metodo.

Scrive anche ``scan_selfcheck/``: la scansione di un soggetto presentata come predizione di
se stessa, spostata con una similarita' nota (scala 1/1000, cioe' in metri, 20 gradi di
rotazione, traslazione) e coi landmark perturbati da 2 mm di rumore gaussiano.  E' il
controllo di sanita' del codice ufficiale -- errore atteso ~0, perche' il raffinamento
scan-to-mesh deve recuperare l'errore dei landmark -- e gira con lo stesso
``compute_error.py``.  La scansione identica coi landmark esatti non si puo' usare: dopo la
Procrustes il residuo e' zero e il dogleg di chumpy produce NaN (assert in
``sbody/mesh_distance.py``, visto nel job 1060237).

Gli export gia' presenti non si riscrivono (``--overwrite`` per forzare).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import now_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", type=str, default=",".join(common.METHODS))
    p.add_argument("--selfcheck-subjects", type=int, default=2,
                   help="soggetti per il controllo scansione-su-se-stessa (0 = niente)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def write_obj(path: Path, V: np.ndarray, F: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in V))
        fh.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in np.asarray(F, dtype=np.int64)))


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    items = common.load_items()
    for method in methods:
        n_ok, missing = 0, []
        for it in items:
            if not (common.recon_dir(method) / f"{it.name}.npz").is_file():
                missing.append(it.name)
                continue
            stem = common.pred_dir(method) / it.subject / it.challenge / Path(it.image).stem
            if not args.overwrite and stem.with_suffix(".obj").is_file() and stem.with_suffix(".npy").is_file():
                n_ok += 1
                continue
            V, F, lmk = common.load_recon(method, it.name)
            write_obj(stem.with_suffix(".obj"), V, F)
            np.save(stem.with_suffix(".npy"), lmk.astype(np.float64))
            n_ok += 1
        print(f"[now-export] {method}: {n_ok}/{len(items)} predizioni -> {common.pred_dir(method)}"
              + (f"; mancanti {len(missing)} (es. {missing[:3]})" if missing else ""), flush=True)

    # Controllo: la scansione come predizione di se stessa, sulla prima immagine dei primi
    # soggetti, spostata e coi landmark rumorosi. L'errore deve venire ~0.
    subjects = common.subjects_of(items)[: args.selfcheck_subjects]
    root = common.WORK_ROOT / "pred" / "scan_selfcheck"
    rng = np.random.default_rng(1234)
    a = np.deg2rad(20.0)
    R = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    lines = []
    for subject in subjects:
        V, F, lmk = common.load_scan(subject)
        lmk_noisy = lmk + rng.normal(scale=2.0, size=lmk.shape)
        move = lambda X: 1e-3 * X @ R.T + np.array([0.1, -0.2, 0.3])  # noqa: E731
        it = next(x for x in items if x.subject == subject)
        stem = root / it.subject / it.challenge / Path(it.image).stem
        write_obj(stem.with_suffix(".obj"), move(V), F)
        np.save(stem.with_suffix(".npy"), move(lmk_noisy))
        lines.append(it.image)
    if lines:
        (root / "imagepaths_selfcheck.txt").write_text("\n".join(lines) + "\n")
        print(f"[now-export] scan_selfcheck: {len(lines)} immagini di {subjects} -> {root}", flush=True)


if __name__ == "__main__":
    main()
