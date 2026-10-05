"""Percorsi, split held-out e I/O delle matrici di distanza per le baseline estese (WS1).

Ogni baseline produce, per ogni coppia ORDINATA di topologie (tA, tB), una matrice
100x100 ``D`` con ``D[i, j] = distanza(soggetto i in topologia tA, soggetto j in tB)``,
salvata in un npz insieme ai nomi dei soggetti.  ``rank_from_matrix.py`` rilegge quelle
matrici e calcola Spearman/Pearson vs D_GT con i CI bootstrap subject-level.

Tutte le metriche qui sono simmetriche nei due argomenti, quindi viene riempita e usata
solo la parte i<j: il resto della matrice resta NaN.  L'unione su tutte le coppie
ordinate di topologie copre comunque entrambi i versi, esattamente come i pair table del
paper.

I 100 soggetti held-out NON sono ricalcolati con lo split: sono letti dai pair table gia'
usati per la Tabella 2 del paper
(``paper_artifacts/bootstrap_ci/table1_pairlevel_exact/<tA>__to__<tB>/pair_metrics.csv``),
cosi' il protocollo e' identico per costruzione e non dipende dal seme.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent

PAIR_TABLE_ROOT = REPO_ROOT / "paper_artifacts" / "bootstrap_ci" / "table1_pairlevel_exact"
MESH_ROOT = Path(
    os.environ.get("WBES_REMESH_NOOPS_DIR", REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500")
)
GT_MATRIX = Path(
    os.environ.get(
        "WBES_DIST_NPZ",
        REPO_ROOT
        / "face_embedding" / "gt_encdec" / "autoencoder" / "latent_analysis"
        / "gt_distance_matrix" / "normalized_matrix_distances.npz",
    )
)
OUT_ROOT = Path(os.environ.get("WBES_BASELINES_OUT", AAU_DIR / "runs" / "baselines"))

TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")
NOCROP_TOPOLOGIES = tuple(t for t in TOPOLOGIES if t != "crop")

# Le quattro topologie che differiscono SOLO per come la superficie e' tassellata: stessa
# faccia, stessa copertura, nessun rumore aggiunto.  Le altre due cambiano la superficie:
# ``noisy`` la sporca, ``crop`` ne toglie un pezzo.
TESSELLATION_TOPOLOGIES = ("original", "remesh", "down8k", "up60k")
PERTURBATION_TOPOLOGIES = ("noisy", "crop")

# Le colonne della Tabella 2 estesa.  Le prime due sono quelle del paper
# (tab:alignment_effect); le altre due spezzano la seconda, perche' "cross-topology" ci
# mette dentro due cose diverse: cambiare la tassellazione della stessa superficie e
# cambiare la superficie.  La decomposizione e' ESATTA -- 12 + 8 = 20 coppie ordinate --
# perche' vive nello stesso universo no-crop della colonna che spezza: ``crop`` non entra
# in nessuna delle quattro, esattamente come non entra in ``nocrop_cross_topology``, e
# nessuna matrice con crop esiste nei due run (vedi il README).  "perturbation" qui vuol
# dire percio' "una delle due mesh e' noisy".
SETTINGS = ("original_to_original", "nocrop_cross_topology",
            "tessellation_cross_topology", "perturbation_cross_topology")

# Le 6 celle diagonali (same-topology) della Tabella 1 del paper
# (tab:clean_xtopo_chamfer_latent_matrices): un setting per topologia, perche'
# rank_from_matrix aggrega per setting e ogni cella della diagonale e' una riga a se'.
# crop compreso, a differenza di nocrop_cross_topology.
SAME_TOPOLOGY_SETTINGS = tuple(f"same_{t}" for t in TOPOLOGIES)

# Viste per i render percettivi: frontale e +/-30 gradi di yaw.
VIEW_YAWS = (0.0, -30.0, 30.0)


def setting_topology_pairs(setting: str) -> list[tuple[str, str]]:
    """Coppie ordinate di topologie che compongono una colonna della Tabella 2."""
    if setting == "original_to_original":
        return [("original", "original")]
    if setting == "nocrop_cross_topology":
        return [(a, b) for a in NOCROP_TOPOLOGIES for b in NOCROP_TOPOLOGIES if a != b]
    if setting == "tessellation_cross_topology":
        return [(a, b) for a in TESSELLATION_TOPOLOGIES for b in TESSELLATION_TOPOLOGIES
                if a != b]
    if setting == "perturbation_cross_topology":
        return [(a, b) for a in NOCROP_TOPOLOGIES for b in NOCROP_TOPOLOGIES
                if a != b and (a in PERTURBATION_TOPOLOGIES or b in PERTURBATION_TOPOLOGIES)]
    if setting in SAME_TOPOLOGY_SETTINGS:
        topology = setting[len("same_"):]
        return [(topology, topology)]


    if setting == "crop_cross_topology":
        return [(a, b) for a in TOPOLOGIES for b in TOPOLOGIES
                if a != b and "crop" in (a, b)]
    if setting == "all_cross_topology":
        return [(a, b) for a in TOPOLOGIES for b in TOPOLOGIES if a != b]
    raise ValueError(f"setting sconosciuto: {setting!r} "
                     f"(attesi {SETTINGS}, {EXTRA_SETTINGS} o {SAME_TOPOLOGY_SETTINGS})")


# Due setting fuori da SETTINGS, che restano i default di tutti gli script: le matrici con
# crop esistono solo nei run di aau/outlineB/, e un default che le chiedesse farebbe
# fallire (o avvisare) tutti gli altri.  ``crop_cross_topology`` sono le 10 coppie
# ordinate con crop da un lato; ``all_cross_topology`` le 30 coppie ordinate cross, cioe'
# l'insieme su cui e' calcolata la tabella di compressione del paper (148.500 righe).
EXTRA_SETTINGS = ("crop_cross_topology", "all_cross_topology")


def all_topology_pairs(settings=SETTINGS) -> list[tuple[str, str]]:
    """Unione, senza duplicati e in ordine stabile, delle coppie di tutti i setting."""
    out: list[tuple[str, str]] = []
    for setting in settings:
        for pair in setting_topology_pairs(setting):
            if pair not in out:
                out.append(pair)
    return out


SUBJECT_SETS = ("heldout", "facebench_first100", "ict_heldout", "flame_heldout")


def heldout_subjects() -> list[str]:
    """I 100 soggetti held-out, letti dai pair table gia' usati per il paper."""
    if not PAIR_TABLE_ROOT.is_dir():
        raise FileNotFoundError(
            f"pair table del paper non trovate in {PAIR_TABLE_ROOT}: senza quelle non e' "
            "possibile riprodurre lo split held-out usato dalla Tabella 1."
        )
    subjects: set[str] = set()
    for pair_dir in sorted(PAIR_TABLE_ROOT.iterdir()):
        csv_path = pair_dir / "pair_metrics.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                subjects.add(row["subject_a"])
                subjects.add(row["subject_b"])
    if not subjects:
        raise RuntimeError(f"nessun pair_metrics.csv leggibile sotto {PAIR_TABLE_ROOT}")
    return sorted(subjects)


def facebench_first100_subjects(n: int = 100) -> list[str]:
    """I primi n soggetti in ordine alfabetico, cioe' quelli usati dalla Tabella 2.

    ``faceBench/latentVSpipeline/run_facebench_remesh.py`` costruisce la lista con
    ``sorted(...)[:max_subjects]``, senza passare dallo split: sono quindi id0000..id0099,
    che con lo split held-out condividono solo 21 soggetti.  Serve per riprodurre lo
    0.729 pubblicato; per i numeri nuovi si usa ``heldout``.
    """
    subjects = sorted(p.stem.split("_GTready_")[0] for p in MESH_ROOT.glob("*_GTready_original.npz"))
    if not subjects:
        raise FileNotFoundError(f"nessuna mesh *_GTready_original.npz in {MESH_ROOT}")
    return subjects[:n] if n > 0 else subjects


def ict_heldout_subjects() -> list[str]:
    """I 100 soggetti ICT che valutano gli eval cross-3DMM (``aau/ict/``).

    Stessa catena di ``ict_zeroshot_rank.sbatch``: pool = le 500 identita' held-out ICT
    della vista ``datasets/ICT/eval_view_heldout``, poi ``rebuild_subject_split``
    (importata, non riscritta) con ``--max_subjects 500 --eval_fraction 0.2`` e il seed di
    ``WBES_EVAL_SEED`` (default 1234).  Mesh e D_GT non sono quelli BFM: vanno indicati con
    ``WBES_REMESH_NOOPS_DIR`` e ``WBES_DIST_NPZ``, e se ``MESH_ROOT`` non contiene mesh ICT
    si esce qui invece di valutare soggetti ICT contro la GT sbagliata.
    """
    sys.path.insert(0, str(REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"))
    from robustness.data_utils import rebuild_subject_split  # noqa: E402

    pool = sorted({p.name.split("_GTready_")[0] for p in MESH_ROOT.glob("*_GTready_*.npz")})
    if not pool or not all(len(s) == 7 and s.startswith("id1") for s in pool):
        raise FileNotFoundError(
            f"{MESH_ROOT} non e' la vista ICT (attesi id1NNNN_GTready_*.npz): esporta "
            "WBES_REMESH_NOOPS_DIR=datasets/ICT/eval_view_heldout e "
            "WBES_DIST_NPZ=datasets/ICT/train_ready/gt_matrix.npz"
        )
    seed = int(os.environ.get("WBES_EVAL_SEED", "1234"))
    _, eval_subjects = rebuild_subject_split(pool, eval_fraction=0.2, seed=seed, max_subjects=500)
    return eval_subjects


def flame_heldout_subjects() -> list[str]:
    """I soggetti FLAME dello zero-shot WS-FLAME (``aau/flame/flame_zeroshot.sbatch``).

    Stessa catena di ``ict_heldout_subjects``: pool = tutte le identita' della vista FLAME
    (``WBES_FLAME_MESH_DIR``, sola geometria, id1000.. con l'offset di ``make_train_ready.py``), poi
    ``rebuild_subject_split`` con ``max_subjects=500``, ``eval_fraction=0.2`` e il seed di
    ``WBES_EVAL_SEED``.  Nessun modello ha visto FLAME, quindi lo split serve solo a dare
    agli eval del modello e alle baseline la stessa selezione.  Mesh e D_GT vanno indicati
    con ``WBES_REMESH_NOOPS_DIR`` e ``WBES_DIST_NPZ`` (lo fa ``aau/flame/flame_baselines.sbatch``).
    """
    sys.path.insert(0, str(REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"))
    from robustness.data_utils import rebuild_subject_split  # noqa: E402

    pool = sorted({p.name.split("_GTready_")[0] for p in MESH_ROOT.glob("*_GTready_*.npz")})
    if not pool or not all(len(s) == 6 and s.startswith("id") and 1000 <= int(s[2:]) < 10000
                           for s in pool):
        raise FileNotFoundError(
            f"{MESH_ROOT} non e' la vista FLAME (attesi id1000..id9999_GTready_*.npz): esporta "
            "WBES_REMESH_NOOPS_DIR=$WBES_FLAME_MESH_DIR e WBES_DIST_NPZ=$WBES_FLAME_DIST_NPZ"
        )
    seed = int(os.environ.get("WBES_EVAL_SEED", "1234"))
    _, eval_subjects = rebuild_subject_split(pool, eval_fraction=0.2, seed=seed, max_subjects=500)
    return eval_subjects


def subject_set(name: str = "heldout") -> list[str]:
    if name == "heldout":
        return heldout_subjects()
    if name == "facebench_first100":
        return facebench_first100_subjects()
    if name == "ict_heldout":
        return ict_heldout_subjects()
    if name == "flame_heldout":
        return flame_heldout_subjects()
    raise ValueError(f"subject set sconosciuto: {name!r} (attesi {SUBJECT_SETS})")


def mesh_name(subject: str, topology: str) -> str:
    return f"{subject}_GTready_{topology}"


def mesh_path(subject: str, topology: str) -> Path:
    return MESH_ROOT / f"{mesh_name(subject, topology)}.npz"


def load_verts_faces(subject: str, topology: str):
    """Vertici/facce grezzi dall'npz mesh-only (chiavi V/F o verts/faces)."""
    with np.load(mesh_path(subject, topology)) as data:
        keys = ("V", "F") if "V" in data else ("verts", "faces")
        return np.asarray(data[keys[0]], np.float64), np.asarray(data[keys[1]], np.int64)


def maxabs_normalize(V: np.ndarray) -> np.ndarray:
    """Centro sulla media dei vertici, scala sul massimo valore assoluto.

    E' la normalizzazione geometrica di ``dataset_gtready``, cioe' quella con cui il repo
    misura Chamfer.  Sta qui e non dentro un solo script perche' la usano sia
    ``chamfer_matrix.py`` sia ``render_cache.py``: se le due divergessero, le distanze
    percettive e quelle geometriche non guarderebbero piu' la stessa mesh.
    """
    Vc = np.asarray(V, dtype=np.float64)
    Vc = Vc - Vc.mean(axis=0, keepdims=True)
    scale = float(np.max(np.abs(Vc)))
    return Vc / scale if scale > 1e-6 else Vc * 0.0


def subject_pair_indices(n_subjects: int) -> tuple[np.ndarray, np.ndarray]:
    """Indici (i, j) con i<j: le stesse 4950 coppie non ordinate dei pair table."""
    i, j = np.triu_indices(n_subjects, k=1)
    return i.astype(np.int64), j.astype(np.int64)


def load_gt_submatrix(subjects: list[str]) -> np.ndarray:
    """Sotto-matrice D_GT (len(subjects) x len(subjects)) con la normalizzazione del repo."""
    sys.path.insert(0, str(REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"))
    from intrinsic_utils import SUBJECT_RE_ANY, load_gt_distance_matrix  # noqa: E402

    # SUBJECT_RE_ANY come posthoc_runner: il default a 4 cifre tronca gli id ICT a 5 cifre
    # (id14501 -> id1450) e 10 soggetti finirebbero sulla stessa riga.  Sugli id BFM
    # (id0000..id4998) le due regex danno gli stessi nomi.
    D, name_to_idx = load_gt_distance_matrix(str(GT_MATRIX), subject_re=SUBJECT_RE_ANY)
    missing = [s for s in subjects if s not in name_to_idx]
    if missing:
        raise KeyError(f"soggetti assenti dalla matrice GT: {missing[:5]} ({len(missing)} in tutto)")
    idx = np.asarray([name_to_idx[s] for s in subjects], dtype=np.int64)
    return np.asarray(D, dtype=np.float64)[np.ix_(idx, idx)]


# ------------------------------------------------------------------ matrici su disco

def matrix_path(metric: str, topology_a: str, topology_b: str, out_root: Path = None) -> Path:
    root = OUT_ROOT if out_root is None else out_root
    return root / "matrices" / metric / f"{topology_a}__to__{topology_b}.npz"


def save_matrix(path: Path, D: np.ndarray, subjects: list[str], metric: str,
                topology_a: str, topology_b: str, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        D=np.asarray(D, dtype=np.float64),
        subjects=np.asarray(subjects, dtype="U16"),
        metric=metric,
        topology_a=topology_a,
        topology_b=topology_b,
        **extra,
    )


def load_matrix(path: Path) -> tuple[np.ndarray, list[str], str, str, str]:
    with np.load(path, allow_pickle=False) as z:
        D = np.asarray(z["D"], dtype=np.float64)
        subjects = [str(s) for s in z["subjects"]]
        metric = str(z["metric"])
        topology_a = str(z["topology_a"])
        topology_b = str(z["topology_b"])
    if D.shape != (len(subjects), len(subjects)):
        raise ValueError(f"{path}: D {D.shape} incoerente con {len(subjects)} soggetti")
    return D, subjects, metric, topology_a, topology_b


def empty_matrix(n: int) -> np.ndarray:
    """Matrice NaN da riempire solo nella parte i<j."""
    return np.full((n, n), np.nan, dtype=np.float64)
