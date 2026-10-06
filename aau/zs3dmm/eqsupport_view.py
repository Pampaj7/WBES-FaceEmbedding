#!/usr/bin/env python3
"""Vista a supporto equalizzato: ogni topologia di un soggetto ritagliata sulla regione del suo crop.

    aau/run.sh aau/zs3dmm/eqsupport_view.py --view-dir datasets/HIFI3D/eval_view/npz \\
        --out-dir datasets/HIFI3D/eqsupport_view --n-cores 32
    (eqsupport_view.sbatch, WBES_ZS_DOMAIN=bfm|hifi|ict)

Il rimedio di protocollo "equalize support" (checklist di paper/OUTLINE_B.md): in una coppia
con un ``crop``, l'altra mesh si riporta allo STESSO supporto prima della metrica. Il crop di
un soggetto e' la banda di bordo di ``datasets/remesh.py::make_crop`` (porting igl in
``v2_work/genict/mesh_ops.py``) applicata alla SUA ``original``; quindi la regione del viso e'
una proprieta' del soggetto, non della coppia, e la vista si costruisce una volta per soggetto:

  1. regione = le facce della ``original`` che stanno sul crop su disco (baricentro a distanza
     <= ``--tol`` x diagonale dal crop). Su HIFI3D e ICT il crop e' un sottoinsieme esatto
     dell'original (vertici e facce, misurato), e la regione e' esattamente quella del taglio.
     Su BFM no: il crop l'ha fatto ``prepare_open_surface`` di open3d
     (``datasets/expand_remesh_topologies``, assente dal repo), che sposta 2-4 vertici di bordo
     su ~21k (fino a 0.2% della diagonale); il porting igl rifatto sull'original da' 60-160
     facce diverse. Per questo la regione si LEGGE dal crop su disco e non si ricalcola: rifare
     il taglio darebbe un supporto vicino ma non quello del crop con cui si confronta;
  2. ``original``, ``noisy``: stessa connettivita' (``F`` identica, misurato), si tengono le
     facce della regione per indice. Su HIFI3D/ICT ``original`` equalizzata == ``crop`` bit per
     bit (controllato qui, riga ``original_is_crop`` del csv);
  3. ``remesh``, ``down8k``, ``up60k``: tassellazione diversa, si tiene una faccia se il punto
     della ``original`` piu' vicino al suo baricentro cade in una faccia della regione. Il bordo
     esce frastagliato alla scala del triangolo di quella topologia (``down8k`` e' la piu' grossa);
  4. ``prepare_open_surface`` (componente piu' grande, niente orfani), come dopo il taglio vero;
  5. ``crop``: symlink al file della vista, invariato.

NON si riapplica il taglio a ciascuna topologia: la banda e' una distanza geodetica dal bordo
in frazione della diagonale, e su ``noisy`` (archi piu' lunghi), ``remesh`` (bordo ritirato
dallo smoothing) o ``down8k`` taglierebbe una regione diversa da quella del crop. Qui la
regione e' identica come regione del viso, non solo per numero di vertici.

Stessi nomi e stesso pool della vista di partenza (tutte le identita', non solo le 100
valutate): ``zs_stage.py`` e ``zs_bl.py`` estraggono i soggetti dai nomi, quindi sulla vista
equalizzata scelgono gli stessi 100 senza toccare la selezione. GT invariata (e' per identita').

Scrive ``npz/<sid>_GTready_<topologia>.npz`` (chiavi ``V``/``F``, dtype dei vertici della vista),
``region_stats.csv`` (per soggetto e topologia: facce, area relativa al crop, distanze
campionate fra la mesh equalizzata e il crop, in frazione della diagonale della original) e
``manifest.json``. Riprendibile: salta i soggetti gia' completi.
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
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))

TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")
SAME_CONNECTIVITY = ("original", "noisy")
RETESSELLATED = ("remesh", "down8k", "up60k")
N_SAMPLES = 20_000


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as d:
        V, F = (d["V"], d["F"]) if "V" in d else (d["verts"], d["faces"])
        return np.asarray(V), np.asarray(F)


def area(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    tri = np.asarray(V, np.float64)[F]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def sample(V: np.ndarray, F: np.ndarray, n: int, rng) -> np.ndarray:
    """Punti uniformi sulla superficie (per area)."""
    a = area(V, F)
    f = rng.choice(len(F), size=n, p=a / a.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    tri = np.asarray(V, np.float64)[F[f]]
    return tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])


def dist_to(P: np.ndarray, V: np.ndarray, F: np.ndarray) -> np.ndarray:
    import igl

    d2, _, _ = igl.point_mesh_squared_distance(np.ascontiguousarray(P, np.float64),
                                               np.ascontiguousarray(V, np.float64),
                                               np.ascontiguousarray(F, np.int64))
    return np.sqrt(np.maximum(d2, 0.0))


def closest_face(P: np.ndarray, V: np.ndarray, F: np.ndarray) -> np.ndarray:
    import igl

    _, idx, _ = igl.point_mesh_squared_distance(np.ascontiguousarray(P, np.float64),
                                                np.ascontiguousarray(V, np.float64),
                                                np.ascontiguousarray(F, np.int64))
    return np.asarray(idx, np.int64)


def region_mask(Vo, Fo, Vc, Fc, tol: float) -> np.ndarray:
    """Facce della original che stanno sul crop: baricentro a <= tol x diagonale dal crop."""
    diag = float(np.linalg.norm(Vo.max(0) - Vo.min(0)))
    centroids = np.asarray(Vo, np.float64)[Fo].mean(1)
    return dist_to(centroids, Vc, Fc) <= tol * diag


def restrict(V: np.ndarray, F: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Facce tenute, poi la stessa pulizia di make_crop (componente piu' grande, niente orfani)."""
    import mesh_ops as mo

    dtype = V.dtype
    Vr, Fr = mo.prepare_open_surface(V, F[keep])
    return Vr.astype(dtype), Fr.astype(np.int32)


def process_subject(task: tuple) -> tuple[str, list[dict]]:
    sid, view_dir, out_npz, tol, overwrite = task
    view_dir, out_npz = Path(view_dir), Path(out_npz)
    outs = {t: out_npz / f"{sid}_GTready_{t}.npz" for t in TOPOLOGIES}
    if not overwrite and all(p.exists() or p.is_symlink() for p in outs.values()):
        return "[skip]", []

    meshes = {t: load(view_dir / f"{sid}_GTready_{t}.npz") for t in TOPOLOGIES}
    Vo, Fo = meshes["original"]
    Vc, Fc = meshes["crop"]
    diag = float(np.linalg.norm(Vo.max(0) - Vo.min(0)))
    mask = region_mask(Vo, Fo, Vc, Fc, tol)
    rng = np.random.default_rng(int(sid[2:]))
    crop_area = float(area(Vc, Fc).sum())
    P_crop = sample(Vc, Fc, N_SAMPLES, rng)

    rows = []
    for t in TOPOLOGIES:
        if t == "crop":
            if outs[t].is_symlink() or outs[t].exists():
                outs[t].unlink()
            outs[t].symlink_to((view_dir / f"{sid}_GTready_crop.npz").resolve())
            continue
        V, F = meshes[t]
        if t in SAME_CONNECTIVITY:
            if not np.array_equal(F, Fo):
                raise SystemExit(f"{sid} {t}: connettivita' diversa dalla original, atteso F identica")
            keep = mask
        else:
            keep = mask[closest_face(np.asarray(V, np.float64)[F].mean(1), Vo, Fo)]
        Ve, Fe = restrict(V, F, keep)
        np.savez_compressed(outs[t], V=Ve, F=Fe)
        P_eq = sample(Ve, Fe, N_SAMPLES, rng)
        d_eq_to_crop = dist_to(P_eq, Vc, Fc) / diag     # eccesso: superficie fuori dal crop
        d_crop_to_eq = dist_to(P_crop, Ve, Fe) / diag   # mancanza: crop non coperto
        # Pavimento: la stessa distanza fra la topologia intera e la original (rumore, smoothing).
        d_floor = dist_to(sample(V, F, N_SAMPLES, rng), Vo, Fo) / diag
        rows.append({
            "subject": sid, "topology": t, "n_verts": len(Ve), "n_faces": len(Fe),
            "n_faces_full": len(F), "kept_face_frac": float(len(Fe) / len(F)),
            "region_faces_of_original": int(mask.sum()), "crop_faces": len(Fc),
            "area_ratio_to_crop": float(area(Ve, Fe).sum() / crop_area),
            "eq_to_crop_p99": float(np.percentile(d_eq_to_crop, 99)), "eq_to_crop_max": float(d_eq_to_crop.max()),
            "crop_to_eq_p99": float(np.percentile(d_crop_to_eq, 99)), "crop_to_eq_max": float(d_crop_to_eq.max()),
            "full_to_original_p99": float(np.percentile(d_floor, 99)),
            "original_is_crop": (bool(np.array_equal(Ve, Vc) and np.array_equal(Fe, Fc))
                                 if t == "original" else ""),
        })
    return "[ok]", rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--view-dir", type=Path, required=True, help="vista di partenza (npz/ con le 6 topologie)")
    p.add_argument("--out-dir", type=Path, required=True, help="scrive out-dir/npz, region_stats.csv, manifest.json")
    p.add_argument("--tol", type=float, default=1e-6, help="baricentro sul crop entro tol x diagonale")
    p.add_argument("--n-cores", type=int, default=1)
    p.add_argument("--max-subjects", type=int, default=0, help="0 = tutti; >0 per una prova")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    import pandas as pd

    subjects = sorted({q.name.split("_GTready_")[0] for q in a.view_dir.glob("id*_GTready_*.npz")})
    if not subjects:
        raise SystemExit(f"nessuna mesh id*_GTready_*.npz in {a.view_dir}")
    if a.max_subjects:
        subjects = subjects[: a.max_subjects]
    out_npz = a.out_dir / "npz"
    out_npz.mkdir(parents=True, exist_ok=True)
    tasks = [(s, str(a.view_dir), str(out_npz), a.tol, a.overwrite) for s in subjects]
    print(f"[eq-view] {len(tasks)} soggetti {a.view_dir} -> {out_npz} tol={a.tol} workers={a.n_cores}", flush=True)

    stats_path = a.out_dir / "region_stats.csv"
    old = pd.read_csv(stats_path) if stats_path.exists() and not a.overwrite else pd.DataFrame()
    rows, tally = [], {"[ok]": 0, "[skip]": 0}
    with mp.get_context("spawn").Pool(max(1, a.n_cores)) as pool:
        for i, (status, r) in enumerate(pool.imap_unordered(process_subject, tasks, chunksize=2), start=1):
            tally[status] += 1
            rows.extend(r)
            if i % 50 == 0 or i == len(tasks):
                print(f"[eq-view] {i}/{len(tasks)} ok={tally['[ok]']} skip={tally['[skip]']}", flush=True)
    new = pd.DataFrame(rows)
    if len(old) and len(new):
        old = old[~old["subject"].isin(set(new["subject"]))]
    stats = pd.concat([old, new], ignore_index=True).sort_values(["subject", "topology"], ignore_index=True)
    stats.to_csv(stats_path, index=False)

    n_files = sum(1 for _ in out_npz.glob("*.npz"))
    if n_files != len(subjects) * len(TOPOLOGIES):
        raise SystemExit(f"vista incompleta: {n_files} file, attesi {len(subjects) * len(TOPOLOGIES)}")
    if set(stats["subject"]) != set(subjects):
        raise SystemExit("region_stats.csv non copre tutti i soggetti: rilancia con --overwrite")

    summary = {}
    for t, g in stats.groupby("topology"):
        summary[t] = {k: float(g[k].median()) for k in ("kept_face_frac", "area_ratio_to_crop", "eq_to_crop_p99",
                                                        "crop_to_eq_p99", "full_to_original_p99")}
        summary[t].update({"area_ratio_min": float(g["area_ratio_to_crop"].min()),
                           "area_ratio_max": float(g["area_ratio_to_crop"].max()),
                           "crop_to_eq_max": float(g["crop_to_eq_max"].max())})
    orig = stats[stats["topology"] == "original"]
    summary["original"]["n_bit_identical_to_crop"] = int((orig["original_is_crop"].astype(str) == "True").sum())
    summary["n_subjects_crop_is_whole_original"] = int(
        (orig["region_faces_of_original"] == orig["n_faces_full"]).sum())
    manifest = {
        "rule": "regione = facce della original col baricentro sul crop (<= tol x diag); original/noisy per "
                "indice di faccia, remesh/down8k/up60k per faccia della original piu' vicina al baricentro; "
                "poi prepare_open_surface; crop invariato (symlink)",
        "view_dir": str(a.view_dir.resolve()), "tol": a.tol, "n_subjects": len(subjects),
        "script_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest(),
        "mesh_ops_sha1": hashlib.sha1((REPO_ROOT / "v2_work/genict/mesh_ops.py").read_bytes()).hexdigest(),
        "median_by_topology": summary,
    }
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"[eq-view] fatto: {n_files} file, ok={tally['[ok]']} skip={tally['[skip]']}, manifest in {a.out_dir}")


if __name__ == "__main__":
    main()
