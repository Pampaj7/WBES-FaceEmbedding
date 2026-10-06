#!/usr/bin/env python3
"""Baseline geometriche zero-shot: alignment_matrix.py e rank_from_matrix.py col set ``<dominio>_heldout``.

    aau/outlineB/run_o3d.sh aau/zs3dmm/zs_bl.py hifi 900000 align --out-root <dir> --workers 30
    aau/run.sh              aau/zs3dmm/zs_bl.py hifi 900000 rank  --out-root <dir> --settings ...

I primi due argomenti sono il dominio (``hifi``/``fv``/``ict``/``bfm``) e il suo offset degli id.
Per ``bfm`` niente registrazione: il set e' ``heldout`` di ``common`` (i 100 soggetti standard,
letti dalle pair table del paper), che coincide con ``select_subjects`` della vista BFM col seed
1234 (verificato, aau/scratch/eqsupport/probe.py) e quindi coi soggetti di ``zs_stage.py``.

Perche' un wrapper invece di una riga in ``aau/baselines/common.py``: ``common.py`` e'
condiviso e lo sta modificando il lavoro FLAME in parallelo. Qui il set ``<dominio>_heldout``
viene AGGIUNTO a runtime (``SUBJECT_SETS`` e ``subject_set``) e poi si chiama il ``main()``
dei due script, importati e non riscritti: stessa pipeline faceBench (chamfer, rigid ICP,
NICP P2P/P2Tri), stesso bootstrap per soggetto.

``<dominio>_heldout`` e' la stessa catena di ``common.ict_heldout_subjects``: pool = tutte le
identita' della vista del dominio (``WBES_REMESH_NOOPS_DIR``), poi ``rebuild_subject_split``
con ``max_subjects=500``, ``eval_fraction=0.2`` e il seed di ``WBES_EVAL_SEED`` -- la stessa
chiamata di ``zs_stage.py``, quindi gli stessi 100 soggetti dello zero-shot dei modelli. Mesh e
D_GT vanno indicati con ``WBES_REMESH_NOOPS_DIR`` e ``WBES_DIST_NPZ`` (lo fa
``zs_baselines.sbatch``), che ``common`` legge all'import. La vista ha solo la geometria
(chiavi ``V``/``F``): e' tutto quello che la pipeline faceBench legge.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
sys.path.insert(0, str(AAU_DIR / "baselines"))
sys.path.insert(0, str(AAU_DIR / "outlineB"))
sys.path.insert(0, str(THIS_DIR))

import common  # noqa: E402
from zs_stage import select_subjects  # noqa: E402


def register(set_name: str, id_offset: int) -> None:
    def domain_subjects() -> list[str]:
        pool = sorted({p.name.split("_GTready_")[0] for p in common.MESH_ROOT.glob("*_GTready_*.npz")})
        ok = [s for s in pool if re.fullmatch(r"id\d{6}", s) and 0 <= int(s[2:]) - id_offset < 100_000]
        if not pool or len(ok) != len(pool):
            raise FileNotFoundError(
                f"{common.MESH_ROOT} non e' la vista del dominio (attesi id{id_offset}..): esporta "
                "WBES_REMESH_NOOPS_DIR e WBES_DIST_NPZ della vista (zs_baselines.sbatch)")
        return select_subjects(common.MESH_ROOT, int(os.environ.get("WBES_EVAL_SEED", "1234")))

    if set_name not in common.SUBJECT_SETS:
        common.SUBJECT_SETS = tuple(common.SUBJECT_SETS) + (set_name,)
    original = common.subject_set

    def subject_set(name: str = "heldout") -> list[str]:
        return domain_subjects() if name == set_name else original(name)

    common.subject_set = subject_set


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[3] not in ("align", "rank"):
        raise SystemExit("uso: zs_bl.py <dominio> <id_offset> align|rank [argomenti dello script]")
    domain, id_offset, step = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    if domain == "bfm":
        set_name = "heldout"
    else:
        set_name = f"{domain}_heldout"
        register(set_name, id_offset)
    if step == "align":
        import alignment_matrix as target  # noqa: E402
    else:
        import rank_from_matrix as target  # noqa: E402
    sys.argv = [target.__file__, "--subject-set", set_name] + sys.argv[4:]
    target.main()


if __name__ == "__main__":
    main()
