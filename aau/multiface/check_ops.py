"""Verifica di un file di operatori per topologia: chiavi, k_eig, area totale, conteggi.

Lo chiamano `multiface_ops.sbatch` e `multiface_ops_areanorm.sbatch` alla fine del job,
tramite tre variabili d'ambiente (le stesse tre righe di controllo dei gemelli REMESH/ICT,
qui in un file invece che in un heredoc perche' vanno guardate due convenzioni diverse):

    WBES_CHECK_PREP    root di `datasets/Multiface/prep`
    WBES_CHECK_TOPOS   "tracked remesh down"
    WBES_CHECK_SUFFIX  "_withops" oppure "_withops_areanorm"

Con `_withops_areanorm` l'area totale deve valere 1 (e' la definizione della convenzione);
con `_withops` deve invece valere l'area della mesh grezza, e viene solo stampata.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

NEEDED = {"verts", "faces", "mass", "evals", "evecs",
          "L_indices", "L_values", "L_shape",
          "gradX_indices", "gradX_values", "gradX_shape",
          "gradY_indices", "gradY_values", "gradY_shape"}
K_EIG = int(os.environ.get("WBES_K_EIG", "128"))


def total_area(V: np.ndarray, F: np.ndarray) -> float:
    t = V[F]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def main() -> None:
    prep = Path(os.environ["WBES_CHECK_PREP"])
    topologies = os.environ.get("WBES_CHECK_TOPOS", "tracked remesh down").split()
    suffix = os.environ["WBES_CHECK_SUFFIX"]
    areanorm = suffix.endswith("_areanorm")

    bad = 0
    for topology in topologies:
        ops_dir = prep / f"{topology}{suffix}"
        mesh_dir = prep / topology
        files = sorted(ops_dir.glob("*.npz"))
        raw = sorted(mesh_dir.glob("*.npz"))
        if len(files) != len(raw):
            print(f"  {ops_dir.name}: {len(files)} operatori contro {len(raw)} mesh")
            bad += 1
            continue

        pick = files[len(files) // 2]
        with np.load(pick, allow_pickle=False) as z:
            keys = set(z.files)
            V, F = z["verts"].astype(np.float64), z["faces"]
            k = int(z["evals"].shape[0])
            mass_min = float(np.asarray(z["mass"]).min())
        with np.load(mesh_dir / pick.name, allow_pickle=False) as z:
            A_raw = total_area(z["V"].astype(np.float64), z["F"])
        A = total_area(V, F)

        missing = NEEDED - keys
        print(f"  {ops_dir.name}: {len(files)} file, {pick.name} verts={len(V)} faces={len(F)} "
              f"k_eig={k} area={A:.6g} area_mesh={A_raw:.6g} mass_min={mass_min:.3g} "
              f"chiavi_mancanti={sorted(missing) if missing else 'nessuna'}")
        if missing or k != K_EIG or mass_min <= 0.0:
            bad += 1
        if areanorm and abs(A - 1.0) > 1e-3:
            print(f"    area totale {A} invece di 1: la convenzione non e' stata applicata")
            bad += 1
        if not areanorm and abs(A - A_raw) > 1e-3 * max(A_raw, 1.0):
            print(f"    area totale {A} diversa da quella della mesh {A_raw}: mesh riscalata")
            bad += 1

    print("  verifica OK" if bad == 0 else f"  VERIFICA FALLITA su {bad} controlli")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
