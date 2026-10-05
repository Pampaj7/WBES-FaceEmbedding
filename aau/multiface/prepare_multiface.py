"""Prepara le 2848 mesh tracciate di Multiface in formato REMESH, in sei topologie.

    aau/run.sh aau/multiface/prepare_multiface.py \
        --full-dir  datasets/Multiface/full \
        --out-dir   datasets/Multiface/prep --n-cores 8

Input:  `full/m--<data>--<ora>--<soggetto>--GHS/tracked_mesh/<segmento>/<frame>.bin`,
        cioe' 7306x3 float32 nel frame TESTA (`external/multiface/dataset.py:191` legge
        esattamente questo file con `np.fromfile(..., np.float32)`), piu' l'`.obj` accanto
        da cui si prende la topologia. Gli `.obj` portano gli stessi vertici nel frame del
        rig: qui si usano solo per le facce, la geometria arriva sempre dal `.bin`.
Output: `<out-dir>/<topologia>/<soggetto>__<segmento>__<frame>.npz` con chiavi `V`
        float32 / `F` int32, le stesse di `datasets/REMESH/npz_data_topo_500` e di
        `datasets/ICT/topo`, piu' `manifest.csv` e `prepare_summary.json`.

Vertici isolati
---------------
La topologia tracciata ha 7306 vertici ma solo 5471 sono referenziati da una faccia: i
1835 restanti non appartengono alla superficie e vanno tolti, altrimenti gli operatori
DiffusionNet trovano righe nulle nella matrice di massa. La pulizia e' `prepare_open_surface`
di `v2_work/genict/mesh_ops.py`, la stessa usata dalla meta' ICT: toglie triangoli
degeneri, schegge staccate e vertici orfani, e non fa nient'altro (su una patch gia'
pulita e' l'identita').

Le sei topologie
----------------
`tracked` e' la topologia comune a tutti i soggetti e tutti i frame, quindi ha
corrispondenza vertice-a-vertice: e' il caso facile, e serve da riferimento.  Le altre
sono **senza corrispondenza** (tranne `noisy`, che tiene le facce), che e' la condizione
in cui WS3a deve misurare, e sono costruite con le primitive di `mesh_ops` gia' usate per
le topologie REMESH:

  `remesh` 2 iterazioni di umbrella smoothing (regola `remesh` di `datasets/remesh.py`),
           poi una suddivisione midpoint 1-a-4 e una decimazione quadrica al bersaglio.
           La suddivisione c'e' per lo stesso motivo per cui c'e' in `up60k`: decimare
           10936 triangoli a ~10000 collasserebbe l'8% degli spigoli e lascerebbe il 92%
           dei vertici dove stavano, cioe' una corrispondenza quasi intatta. Partendo da
           43744 triangoli il collasso e' del 77% e la ritessellatura e' vera.
  `down`   decimazione quadrica diretta al bersaglio, come `down8k`.
  `up`     suddivisione midpoint 1-a-4 seguita da decimazione al bersaglio, come `up60k`:
           una suddivisione pura potrebbe atterrare solo su 4x.
  `crop`   taglio canonico: via la fascia bassa (mento e collo) e le due fasce laterali.
  `noisy`  rumore gaussiano sui vertici, facce invariate, come il `noisy` di REMESH.

Le prime tre varianti sono "gentili" — le tre scansioni restano pulite, complete e a
risoluzione confrontabile — e su di esse il protocollo WS3a e' risultato saturo (AUC ~ 1
per tutte le metriche tranne il varifold, `aau/runs/multiface_ws3a/summary.md`).  `crop`,
`noisy` e `up` sono la versione dura, speculare alle perturbazioni REMESH: sovrapposizione
parziale, geometria sporca, risoluzione molto diversa.

I bersagli sono dati in VERTICI (~5000, ~2500 e ~13700, cioe' ~1x, ~0.5x e ~2.5x la
risoluzione della patch tracciata) e convertiti in triangoli con il rapporto V/F della
patch stessa: e' il numero di vertici che conta per il costo degli operatori e per il
confronto fra topologie.  Il fattore 0.7x del `remesh` di REMESH non e' stato riusato
perche' su questa patch darebbe ~3.8k vertici, fuori dalla specifica del cantiere.

`crop` non e' il `crop` di REMESH (`mesh_ops.make_crop`, la fascia geodetica di 0.06 x
diagonale sul bordo esterno): quello toglie al piu' il 15% dei vertici e qui servirebbe a
poco.  La regola e' invece un taglio anatomico, la stessa per ogni mesh: si buttano i
vertici sotto il percentile `CROP_BOTTOM_PERCENTILE` di y (in alto e' +y, verificato dai
render: `ws3a_render.py` porta le Multiface nel frame del renderer con Rx(180)) e quelli
fuori dalla banda centrale di x fra i percentili `CROP_LATERAL_PERCENTILE` e il suo
complemento; poi `prepare_open_surface` ripulisce il bordo del taglio.  Le soglie sono
percentili della mesh stessa, come il `BOUNDARY_RADIUS_PERCENTILE` di `datasets/remesh.py`,
cosi' la frazione tenuta non dipende dalla scala ne' dal soggetto: ~70% dei vertici.

`noisy` usa la sigma relativa di REMESH (0.003 x diagonale del bbox) e un seme per frame
preso dal crc32 del nome: `hash()` di una stringa e' randomizzato per processo e con
`spawn` darebbe rumore diverso a ogni riavvio.

Unita' di misura
----------------
Non sono documentate a monte: lo script le misura (bounding box di ogni mesh) e le riporta
in `prepare_summary.json`, con un verdetto mm/cm/m dedotto dalla diagonale mediana.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import re
import sys
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))
import mesh_ops as mo  # noqa: E402

# Topologia tracciata di Multiface, come scaricata (verificato dal download).
FULL_VERTICES = 7306
FULL_TRIANGLES = 10936
SURFACE_VERTICES = 5471  # 7306 - 1835 orfani

REMESH_SMOOTH_ITERS = 2       # b8adab3 datasets/remesh.py::make_remesh
REMESH_TARGET_VERTICES = 5000
DOWN_TARGET_VERTICES = 2500
UP_TARGET_VERTICES = 13700    # ~2.5x SURFACE_VERTICES

NOISE_STD_BBOX_FRACTION = 0.003   # b8adab3 datasets/remesh.py::make_noisy
CROP_BOTTOM_PERCENTILE = 16.0     # via la fascia bassa (y crescente = verso l'alto)
CROP_LATERAL_PERCENTILE = 7.5     # via le due fasce laterali, una per lato

TOPOLOGIES = ("tracked", "remesh", "down", "crop", "noisy", "up")

# m--20180105--0000--002539136--GHS  ->  002539136
CAPTURE_RE = re.compile(r"^m--\d+--\d+--(?P<subject>[^-]+)--GHS$")


def subject_of(capture_dir: Path) -> str:
    m = CAPTURE_RE.match(capture_dir.name)
    if m is None:
        raise ValueError(f"nome di capture inatteso: {capture_dir.name}")
    return m.group("subject")


def version_of(segments: list[str]) -> str:
    """v1 = i 10 soggetti Mugsy con segmenti `E0xx_*`, v2 = i 3 con `EXP_*`."""
    if all(s.startswith("EXP_") for s in segments):
        return "v2"
    if all(re.match(r"^E\d{3}_", s) for s in segments):
        return "v1"
    raise ValueError(f"segmenti ne' v1 ne' v2: {sorted(segments)[:3]}")


def read_obj_faces(path: Path) -> np.ndarray:
    """Facce triangolari di un `.obj` Multiface (`f v/vt v/vt v/vt`, indici da 1)."""
    faces = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("f "):
                continue
            corners = line.split()[1:]
            if len(corners) != 3:
                raise ValueError(f"{path.name}: faccia non triangolare ({len(corners)} lati)")
            faces.append([int(c.split("/")[0]) - 1 for c in corners])
    return np.asarray(faces, dtype=np.int32)


def triangle_target(target_vertices: int, n_verts: int, n_faces: int) -> int:
    """Triangoli che danno ~target_vertices su una patch con questo rapporto V/F."""
    return max(4, int(round(target_vertices * n_faces / n_verts)))


def make_remesh(V: np.ndarray, F: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Smoothing leggero + risuddivisione + decimazione quadrica al bersaglio."""
    Vs = mo.smooth_simple(V, F, REMESH_SMOOTH_ITERS)
    Vu, Fu = mo.subdivide_midpoint(Vs, F, 1)
    return mo.decimate_to(Vu, Fu, target)


def make_down(V: np.ndarray, F: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    return mo.decimate_to(V, F, target)


def make_up(V: np.ndarray, F: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Risuddivisione 1-a-4 e decimazione al bersaglio, come `up60k` di REMESH."""
    Vu, Fu = mo.subdivide_midpoint(V, F, 1)
    return mo.decimate_to(Vu, Fu, target)


def make_crop(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Taglio canonico della fascia bassa e delle due laterali: ~70% dei vertici.

    Non e' `mo.make_crop` (la fascia geodetica sul bordo di `datasets/remesh.py`, che
    toglie al piu' il 15%): vedi il docstring del modulo.  Le soglie sono percentili
    della mesh stessa, quindi la regola e' la stessa per ogni frame e ogni soggetto.
    """
    y_low = np.percentile(V[:, 1], CROP_BOTTOM_PERCENTILE)
    x_low, x_high = np.percentile(V[:, 0], [CROP_LATERAL_PERCENTILE,
                                            100.0 - CROP_LATERAL_PERCENTILE])
    keep = (V[:, 1] >= y_low) & (V[:, 0] >= x_low) & (V[:, 0] <= x_high)
    # Un triangolo sopravvive solo se tutti e tre i vertici sopravvivono; poi la pulizia
    # toglie gli orfani del bordo del taglio e le schegge staccate dal ritaglio.
    return mo.prepare_open_surface(V, F[np.all(keep[F], axis=1)])


def make_noisy(V: np.ndarray, F: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Rumore gaussiano a 0.003 x diagonale del bbox, topologia invariata (`noisy` REMESH)."""
    scale = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    noise = np.random.default_rng(seed).normal(0.0, NOISE_STD_BBOX_FRACTION * scale, V.shape)
    return V + noise, F


def noise_seed(name: str) -> int:
    """Seme stabile per frame: `hash()` di una stringa e' randomizzato per processo."""
    return zlib.crc32(name.encode("utf-8"))


def bbox_stats(V: np.ndarray) -> tuple[float, float, float, float]:
    """(extent x, y, z, diagonale) del bounding box, nelle unita' del file."""
    ext = V.max(axis=0) - V.min(axis=0)
    return float(ext[0]), float(ext[1]), float(ext[2]), float(np.linalg.norm(ext))


# ------------------------------------------------------------------ un frame per volta

def process_frame(task: tuple) -> tuple[str, str, list[dict]]:
    bin_str, out_dir_str, subject, segment, frame, version, faces_str, overwrite = task
    bin_path, out_dir = Path(bin_str), Path(out_dir_str)
    name = f"{subject}__{segment}__{frame}"
    out = {t: out_dir / t / f"{name}.npz" for t in TOPOLOGIES}

    try:
        rows: list[dict] = []
        # Una variante per volta e solo quelle mancanti: cosi' aggiungere `crop`, `noisy`
        # e `up` a una prep gia' fatta non ricalcola (ne' riscrive) le altre tre.
        todo = [t for t in TOPOLOGIES if overwrite or not out[t].exists()]
        if todo:
            F_full = np.load(faces_str)["F"]
            V_full = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3).astype(np.float64)
            if len(V_full) != FULL_VERTICES:
                raise ValueError(f"{len(V_full)} vertici invece di {FULL_VERTICES}")

            V, F = mo.prepare_open_surface(V_full, F_full)
            builders = {
                "tracked": lambda: (V, F),
                "remesh": lambda: make_remesh(V, F, triangle_target(REMESH_TARGET_VERTICES, len(V), len(F))),
                "down": lambda: make_down(V, F, triangle_target(DOWN_TARGET_VERTICES, len(V), len(F))),
                "up": lambda: make_up(V, F, triangle_target(UP_TARGET_VERTICES, len(V), len(F))),
                "crop": lambda: make_crop(V, F),
                "noisy": lambda: make_noisy(V, F, seed=noise_seed(name)),
            }
            for topology in todo:
                Vt, Ft = builders[topology]()
                if not np.isfinite(Vt).all():
                    raise ValueError(f"{topology}: vertici non finiti")
                mo.save_variant(Vt, Ft, out[topology])

        for topology in TOPOLOGIES:
            with np.load(out[topology]) as d:
                Vt, Ft = d["V"].astype(np.float64), d["F"]
            ex, ey, ez, diag = bbox_stats(Vt)
            rows.append({
                "name": name, "subject": subject, "segment": segment, "frame": frame,
                "version": version, "topology": topology,
                "n_vertices": len(Vt), "n_faces": len(Ft),
                "bbox_x": round(ex, 4), "bbox_y": round(ey, 4), "bbox_z": round(ez, 4),
                "bbox_diag": round(diag, 4),
                "npz": str(out[topology].relative_to(REPO_ROOT)),
            })
        return "[ok]", name, rows
    except Exception as exc:  # una mesh rotta non deve fermare il batch
        return "[fail]", f"{name}: {type(exc).__name__}: {exc}", []


# ------------------------------------------------------------------ raccolta degli input

def collect_frames(full_dir: Path) -> tuple[list[tuple[str, str, str, str, Path]], dict]:
    """(soggetto, segmento, frame, versione, path del .bin) per ogni frame, piu' un indice."""
    frames = []
    per_subject: dict[str, list[str]] = {}
    for capture in sorted(full_dir.iterdir()):
        mesh_root = capture / "tracked_mesh"
        if not mesh_root.is_dir():
            continue
        subject = subject_of(capture)
        segments = sorted(d.name for d in mesh_root.iterdir() if d.is_dir())
        version = version_of(segments)
        per_subject[subject] = segments
        for segment in segments:
            for bin_path in sorted((mesh_root / segment).glob("*.bin")):
                frames.append((subject, segment, bin_path.stem, version, bin_path))
    if not frames:
        raise FileNotFoundError(f"nessun tracked_mesh/*/*.bin sotto {full_dir}")
    return frames, per_subject


def reference_faces(full_dir: Path, frames: list) -> np.ndarray:
    """Facce della topologia tracciata, verificate identiche su un `.obj` per soggetto."""
    by_subject: dict[str, Path] = {}
    for subject, _segment, _frame, _version, bin_path in frames:
        by_subject.setdefault(subject, bin_path.with_suffix(".obj"))

    ref_subject, ref_path = sorted(by_subject.items())[0]
    F = read_obj_faces(ref_path)
    if F.shape != (FULL_TRIANGLES, 3) or int(F.max()) != FULL_VERTICES - 1:
        raise ValueError(f"{ref_path}: {F.shape} facce, indice max {int(F.max())}")
    for subject, obj_path in sorted(by_subject.items())[1:]:
        if not np.array_equal(read_obj_faces(obj_path), F):
            raise ValueError(f"la topologia di {subject} non coincide con quella di {ref_subject}")
    return F


def unit_verdict(diag_median: float) -> str:
    """Una testa umana e' ~25 cm: la diagonale del bbox dice in che unita' e' scritta."""
    if 100.0 <= diag_median <= 1000.0:
        return "mm"
    if 10.0 <= diag_median < 100.0:
        return "cm"
    if 0.1 <= diag_median < 10.0:
        return "m"
    return "ignote"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--full-dir", type=Path, default=REPO_ROOT / "datasets/Multiface/full")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "datasets/Multiface/prep")
    p.add_argument("--n-frames", type=int, default=0, help="0 = tutti, >0 per un test rapido")
    p.add_argument("--n-cores", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    frames, per_subject = collect_frames(a.full_dir)
    if a.n_frames:
        frames = frames[: a.n_frames]
    print(f"{len(frames)} frame, {len(per_subject)} soggetti  {a.full_dir} -> {a.out_dir}", flush=True)

    F_full = reference_faces(a.full_dir, frames)
    print(f"topologia tracciata: {FULL_VERTICES} vertici / {len(F_full)} facce, "
          f"identica su tutti i {len(per_subject)} soggetti", flush=True)

    # La topologia e' fissa, quindi la pulizia si puo' fare una volta sola qui e il
    # risultato vale per ogni frame: serve a stampare i bersagli prima di partire.
    V_probe = np.fromfile(frames[0][4], dtype=np.float32).reshape(-1, 3).astype(np.float64)
    V_ref, F_ref = mo.prepare_open_surface(V_probe, F_full)
    n_comp, _ = __import__("igl").facet_components(np.asarray(F_ref, dtype=np.int64))
    print(f"dopo la pulizia: {len(V_ref)} vertici / {len(F_ref)} facce, "
          f"{FULL_VERTICES - len(V_ref)} orfani rimossi, {n_comp} componente/i", flush=True)
    t_remesh = triangle_target(REMESH_TARGET_VERTICES, len(V_ref), len(F_ref))
    t_down = triangle_target(DOWN_TARGET_VERTICES, len(V_ref), len(F_ref))
    t_up = triangle_target(UP_TARGET_VERTICES, len(V_ref), len(F_ref))
    print(f"bersagli: remesh {t_remesh} triangoli (~{REMESH_TARGET_VERTICES} vertici), "
          f"down {t_down} triangoli (~{DOWN_TARGET_VERTICES} vertici), "
          f"up {t_up} triangoli (~{UP_TARGET_VERTICES} vertici)", flush=True)
    V_crop, F_crop = make_crop(V_ref, F_ref)
    print(f"crop: {len(V_crop)} vertici / {len(F_crop)} facce sul primo frame, cioe' il "
          f"{len(V_crop) / len(V_ref):.1%} dei vertici (soglie: y sotto il "
          f"{CROP_BOTTOM_PERCENTILE}o percentile, x fuori dal {CROP_LATERAL_PERCENTILE}-"
          f"{100.0 - CROP_LATERAL_PERCENTILE})", flush=True)

    for topology in TOPOLOGIES:
        (a.out_dir / topology).mkdir(parents=True, exist_ok=True)

    # Le facce viaggiano ai worker come npz su disco e non come array nel task: con `spawn`
    # ogni task viene serializzato una volta per frame, e 10936x3 int32 x 2848 task sono
    # 370 MB di pickle inutile.
    faces_cache = a.out_dir / ".tracked_faces.npz"
    np.savez(faces_cache, F=F_full)

    tasks = [(str(bin_path), str(a.out_dir), subject, segment, frame, version,
              str(faces_cache), a.overwrite)
             for subject, segment, frame, version, bin_path in frames]

    if a.n_cores > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        pool = mp.Pool(processes=a.n_cores)
        results = pool.imap_unordered(process_frame, tasks)
    else:
        pool, results = None, map(process_frame, tasks)

    rows: list[dict] = []
    failures: list[str] = []
    ok = 0
    try:
        for i, (status, msg, frame_rows) in enumerate(results, start=1):
            if status == "[ok]":
                ok += 1
                rows.extend(frame_rows)
            else:
                failures.append(msg)
            if status != "[ok]" or i % 200 == 0 or i <= 3:
                print(f"[{i}/{len(tasks)}] {status} {msg}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"\nFatto. ok={ok} fail={len(failures)}")
    for msg in failures[:20]:
        print(f"  - {msg}")

    fields = ["name", "subject", "segment", "frame", "version", "topology",
              "n_vertices", "n_faces", "bbox_x", "bbox_y", "bbox_z", "bbox_diag", "npz"]
    rows.sort(key=lambda r: (r["topology"], r["name"]))
    manifest = a.out_dir / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"manifest: {manifest} ({len(rows)} righe)")

    # --- riepilogo: unita', scala, conteggi per topologia -----------------------------
    summary = {
        "full_dir": str(a.full_dir),
        "out_dir": str(a.out_dir),
        "n_frames": ok,
        "n_subjects": len(per_subject),
        "n_failures": len(failures),
        "tracked_full_vertices": FULL_VERTICES,
        "tracked_surface_vertices": len(V_ref),
        "tracked_faces": len(F_ref),
        "orphan_vertices": FULL_VERTICES - len(V_ref),
        "triangle_targets": {"remesh": t_remesh, "down": t_down, "up": t_up},
        "crop_rule": {"bottom_percentile": CROP_BOTTOM_PERCENTILE,
                      "lateral_percentile": CROP_LATERAL_PERCENTILE,
                      "kept_vertices_first_frame": len(V_crop),
                      "kept_fraction_first_frame": round(len(V_crop) / len(V_ref), 4)},
        "noise_std_bbox_fraction": NOISE_STD_BBOX_FRACTION,
        "topologies": {},
    }
    for topology in TOPOLOGIES:
        sel = [r for r in rows if r["topology"] == topology]
        if not sel:
            continue
        nv = np.asarray([r["n_vertices"] for r in sel])
        nf = np.asarray([r["n_faces"] for r in sel])
        diag = np.asarray([r["bbox_diag"] for r in sel])
        ext = np.asarray([[r["bbox_x"], r["bbox_y"], r["bbox_z"]] for r in sel])
        summary["topologies"][topology] = {
            "n_meshes": len(sel),
            "n_vertices": {"min": int(nv.min()), "median": float(np.median(nv)), "max": int(nv.max())},
            "n_faces": {"min": int(nf.min()), "median": float(np.median(nf)), "max": int(nf.max())},
            "bbox_diag": {"min": round(float(diag.min()), 3),
                          "median": round(float(np.median(diag)), 3),
                          "max": round(float(diag.max()), 3)},
            "bbox_extent_median": [round(float(x), 3) for x in np.median(ext, axis=0)],
            "unit": unit_verdict(float(np.median(diag))),
        }
        s = summary["topologies"][topology]
        print(f"  {topology}: {s['n_meshes']} mesh, vertici {s['n_vertices']}, "
              f"facce {s['n_faces']}, bbox diag {s['bbox_diag']} -> unita' {s['unit']}")

    summary_path = a.out_dir / "prepare_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"riepilogo: {summary_path}")

    faces_cache.unlink(missing_ok=True)
    if failures:
        raise SystemExit(1)


def demo(argv: list[str]) -> None:
    """Self-check: chiavi, tipi, orfani, e la proprieta' che conta per WS3a.

    Quella proprieta' e' che `remesh`, `down`, `up` e `crop` **non** siano un insieme di
    indici fisso: se due mesh diverse finissero con le stesse facce, fra loro ci sarebbe
    di nuovo una corrispondenza vertice-a-vertice e la coppia non misurerebbe piu' il caso
    senza corrispondenza. `noisy` invece le facce le tiene per definizione, quindi sta con
    `tracked` fra le topologie a indici fissi. Che una singola variante condivida dei
    vertici con la SUA tracked e' invece normale e atteso per una decimazione a collasso di
    spigoli (`down8k` di REMESH si comporta allo stesso modo): viene stampato come
    diagnostica, non verificato.

    I frame vengono presi a passo costante sulla lista ordinata, cosi' cadono su soggetti
    diversi invece che tutti dentro il primo segmento.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "datasets/Multiface/prep")
    p.add_argument("--n-frames", type=int, default=6)
    a, _ = p.parse_known_args(argv)

    everything = sorted(q.stem for q in (a.out_dir / "tracked").glob("*.npz"))
    assert everything, f"servono npz in {a.out_dir}/tracked"
    n = min(a.n_frames, len(everything))
    step = max(1, len(everything) // n)
    names = everything[::step][:n]

    faces: dict[str, list[np.ndarray]] = {t: [] for t in TOPOLOGIES}
    for name in names:
        loaded = {}
        for topology in TOPOLOGIES:
            with np.load(a.out_dir / topology / f"{name}.npz") as d:
                V, F = d["V"], d["F"]
            assert V.dtype == np.float32 and F.dtype == np.int32, (V.dtype, F.dtype)
            assert int(F.max()) == len(V) - 1, f"{name}/{topology}: vertici non referenziati"
            assert np.isfinite(V).all(), f"{name}/{topology}: vertici non finiti"
            loaded[topology] = (V.astype(np.float64), F)
            faces[topology].append(F)
            print(f"  {name}/{topology}: V={len(V)} F={len(F)}")
        Vt = loaded["tracked"][0]
        for topology in ("remesh", "down", "up", "crop"):
            Vx = loaded[topology][0]
            _, counts = np.unique(np.vstack([Vt, Vx]), axis=0, return_counts=True)
            shared = int((counts > 1).sum())
            print(f"  {name}/{topology}: vertici coincidenti con la sua tracked "
                  f"{shared}/{len(Vx)} ({shared / len(Vx):.1%})")
        # Le tre varianti dure, misurate: il crop deve tenerne ~70%, l'up alzarli a ~2.5x
        # e il noisy spostarli di ~0.003 x diagonale senza toccare le facce.
        Vn = loaded["noisy"][0]
        diag = float(np.linalg.norm(Vt.max(axis=0) - Vt.min(axis=0)))
        print(f"  {name}: crop {len(loaded['crop'][0]) / len(Vt):.1%} dei vertici, "
              f"up {len(loaded['up'][0]) / len(Vt):.2f}x, noisy scarto rms "
              f"{float(np.sqrt(((Vn - Vt) ** 2).sum(axis=1).mean())) / diag:.4f} x diagonale")

    def all_equal(arrays: list[np.ndarray]) -> bool:
        return all(a_.shape == arrays[0].shape and np.array_equal(a_, arrays[0]) for a_ in arrays)

    for topology in ("tracked", "noisy"):
        assert all_equal(faces[topology]), f"{topology} non e' una topologia a indici fissi"
    for topology in ("remesh", "down", "up", "crop"):
        assert not all_equal(faces[topology]), \
            f"{topology}: {len(names)} mesh con le stesse facce, la corrispondenza non e' rotta"
    print(f"demo OK su {len(names)} frame: tracked/noisy a indici fissi, "
          "remesh/down/up/crop con facce diverse da mesh a mesh")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo(sys.argv[1:])
    else:
        main()
