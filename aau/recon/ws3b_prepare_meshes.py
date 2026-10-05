#!/usr/bin/env python3
"""Ritaglia la regione volto su GT e ricostruzioni, e le porta alla stessa risoluzione (WS3b).

    aau/run.sh aau/recon/ws3b_prepare_meshes.py --stage gt
    aau/run.sh aau/recon/ws3b_prepare_meshes.py --stage recon

Produce due famiglie di mesh, entrambe a **topologia fissa**:

``gt_face/<sogg>__<segmento>__<frame>.npz``
    la mesh tracciata ridotta ai vertici entro 95 mm dalla punta del naso, che e' il
    ritaglio del protocollo NoW.  La punta del naso e' il vertice di z massima sulla mesh
    MEDIA delle 2848 tracciate; siccome la topologia tracciata e' in corrispondenza densa,
    quell'indice e la lista dei vertici tenuti valgono per tutti i soggetti e tutti i
    frame.  Serve perche' la patch di Multiface arriva alla nuca e alla base del collo
    (bbox mediana 190x341x227 mm) e nessun metodo da singola immagine la ricostruisce:
    misurare li' vorrebbe dire misurare quanto collo c'e' nella topologia del metodo.

``recon_face/<metodo>/<nome>.npz``
    la ricostruzione ridotta alla stessa regione e decimata al numero di triangoli di
    ``gt_face``.  La regione e' definita sulla topologia (fissa) del metodo, una volta per
    metodo, sulla mesh media in coordinate canoniche -- origine nel landmark 30 (punta del
    naso) e unita' la distanza fra i landmark 36 e 45 (gli exocanthion).  Il raggio e'
    ``95 mm / ex-ex`` aperture oculari, dove l'ex-ex e' quello della ricostruzione riportato
    in millimetri veri dalla scala dell'ICP (86.10 mm mediani, misurati da
    ``ws3b_landmarks.py``): da una foto sola la scala metrica non e' osservabile, e quella
    scala e' la sola grandezza che lega i pixel del metodo ai millimetri della GT.  Con quel
    raggio il ritaglio della ricostruzione vale 95 mm veri, come quello della GT.

    La prima versione metteva al denominatore 90 mm, la media adulta di Farkas invece del
    valore misurato su QUESTI soggetti, e il ritaglio veniva piu' stretto: con la scala
    pixel->mm stimata dall'ICP il raggio effettivo era 90.7 mm per 3DDFA_V2, 91.6 per
    SynergyNet, 91.0 per PRNet, contro i 95 della GT.  Su una calotta di raggio ~95 mm un
    4% di raggio in meno e' circa un 8% di area in meno: la Chamfer grezza e il latent, che
    guardano il supporto, pagavano quella corona.  ``ws3b_geometric.py`` riporta ora anche
    l'area delle due patch in mm^2 e il loro rapporto, cosi' la coincidenza dei supporti si
    controlla sui numeri e non sulle intenzioni.

Decimazione
-----------
Il RITAGLIO e' a indici fissi per metodo (la topologia dei tre metodi e' fissa, quindi la
regione volto e' la stessa lista di vertici per tutte le immagini), la DECIMAZIONE no: e'
``mesh_ops.decimate_to``, cioe' lo stesso collasso quadrico di ``igl.qslim`` con cui sono
state costruite le topologie ``remesh`` e ``down`` di Multiface e di REMESH, applicato a
ogni mesh.  Si era provato a decimare una volta sola sulla media e a riusare gli indici di
nascita di qslim come sottocampionamento fisso: i vertici di nascita distano fino al 3%
della diagonale da quelli che qslim calcola (che sono medie pesate dei vertici collassati,
non vertici di ingresso), troppo per spacciarli per la stessa decimazione.  Meglio 9225
decimazioni vere, che costano pochi minuti, che una decimazione finta.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ws3b_common as common  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "genict"))
import mesh_ops as mo  # noqa: E402

# Stato condiviso coi worker via fork.
_TEMPLATE: dict = {}


def gt_outer_canthal_mm(out_root: Path, override: float = 0.0, source: str = "icp") -> float:
    """Il denominatore che converte i 95 mm del ritaglio in aperture oculari.

    Non e' un valore di letteratura ma una misura, e ce ne sono due, che
    ``ws3b_landmarks.py`` scrive entrambe:

    ``icp`` (default)
        ``ex_ex_icp_mm``: l'ex-ex della RICOSTRUZIONE espresso in millimetri della GT
        attraverso la scala dell'ICP.  Misurato 86.10 mm.  E' questo che serve: il raggio
        del ritaglio della ricostruzione, riportato in millimetri veri, deve valere 95 come
        quello della GT, e ``r = 95 / ex_ex_icp`` e' esattamente la condizione perche' ci
        valga.  E' anche la sola grandezza osservabile che lega i pixel del metodo ai
        millimetri della GT.

    ``landmarks``
        ``ex_ex_mm``: l'ex-ex della GT misurato sui vertici stimati, 91.72 mm.  Misura una
        cosa diversa -- l'anatomia della GT, non la scala della ricostruzione -- e i due
        numeri differiscono del 6.5%, perche' i landmark trasferiti hanno 5-7 mm di
        dispersione.  Usarlo qui darebbe un ritaglio di 89.2 mm veri, cioe' peggio di prima.
    """
    if override > 0:
        return float(override)
    path = common.landmarks_path(out_root)
    if not path.is_file():
        raise SystemExit(
            f"manca {path}: il raggio del ritaglio dipende dall'ex-ex misurato.\n"
            f"  Crealo con: aau/run.sh aau/recon/ws3b_landmarks.py")
    data = json.loads(path.read_text())
    key = {"icp": "ex_ex_icp_mm", "landmarks": "ex_ex_mm"}[source]
    return float(data[key])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--stage", type=str, default="all", choices=("all", "gt", "recon"))
    p.add_argument("--methods", type=str, default=",".join(common.METHODS))
    p.add_argument("--radius-mm", type=float, default=common.FACE_RADIUS_MM)
    p.add_argument("--ex-ex-mm", type=float, default=0.0,
                   help="0 = il valore misurato da ws3b_landmarks.py (vedi --ex-ex-source)")
    p.add_argument("--ex-ex-source", type=str, default="icp", choices=("icp", "landmarks"),
                   help="icp = ex-ex della ricostruzione in mm veri; landmarks = ex-ex della GT")
    p.add_argument("--mean-sample", type=int, default=200,
                   help="ricostruzioni usate per la mesh media canonica di un metodo")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def flip_to_right_handed(V: np.ndarray) -> np.ndarray:
    """Da spazio immagine (y in giu') al frame testa di Multiface (y in su).

    Non e' un allineamento: e' la riflessione che rende le due mani concordi, senza la
    quale una similarita' non puo' sovrapporre le due mesh (vedi ws3b_common).
    """
    out = np.array(V, dtype=np.float64, copy=True)
    out[:, 1] *= -1.0
    return out


def crop_region(V: np.ndarray, F: np.ndarray, center: np.ndarray, radius: float):
    """Vertici entro ``radius`` dal centro, facce coi tre spigoli dentro, patch ripulita.

    Restituisce (indici dei vertici tenuti nella numerazione di ingresso, facce rimappate).
    """
    inside = np.linalg.norm(V - center[None, :], axis=1) <= radius
    F_in = np.asarray(F, dtype=np.int64)
    F_in = F_in[inside[F_in].all(axis=1)]
    if len(F_in) == 0:
        raise SystemExit(f"il ritaglio di raggio {radius} non contiene nessun triangolo")
    kept = np.unique(F_in)
    remap = np.zeros(len(V), dtype=np.int64)
    remap[kept] = np.arange(len(kept))
    Vc, Fc = mo.prepare_open_surface(V[kept], remap[F_in])
    # prepare_open_surface puo' togliere schegge: si riporta la selezione sugli indici veri.
    kept2 = np.unique(np.asarray(Fc, dtype=np.int64))
    remap2 = np.zeros(len(Vc), dtype=np.int64)
    remap2[kept2] = np.arange(len(kept2))
    return kept[kept2], np.ascontiguousarray(remap2[np.asarray(Fc, dtype=np.int64)], dtype=np.int32)


# --------------------------------------------------------------------------- GT

def build_gt_template(args) -> dict:
    """Mesh media delle tracciate, punta del naso, ritaglio a 95 mm."""
    names = sorted(p.stem for p in (common.MF_PREP / "tracked").glob("*.npz"))
    if not names:
        raise SystemExit(f"nessuna mesh tracciata in {common.MF_PREP / 'tracked'}")
    t0 = time.time()
    acc = None
    for i, name in enumerate(names):
        with np.load(common.gt_mesh_path(name)) as z:
            V = np.asarray(z["V"], dtype=np.float64)
            if acc is None:
                acc, F = np.zeros_like(V), np.asarray(z["F"], dtype=np.int32)
        acc += V
    V_mean = acc / len(names)
    nose = int(np.argmax(V_mean[:, 2]))  # +z = in avanti nel frame testa di Multiface
    idx, F_face = crop_region(V_mean, F, V_mean[nose], args.radius_mm)
    ext = V_mean[idx].max(axis=0) - V_mean[idx].min(axis=0)
    print(f"[ws3b-prep] mesh media su {len(names)} tracciate in {time.time() - t0:.0f}s; "
          f"punta del naso = vertice {nose} a {np.round(V_mean[nose], 1)} mm", flush=True)
    print(f"[ws3b-prep] gt_face: {len(idx)}/{len(V_mean)} vertici, {len(F_face)} triangoli, "
          f"extent {np.round(ext, 1)} mm", flush=True)
    return {"names": names, "vertex_indices": idx, "faces": F_face,
            "nose_vertex": nose, "nose_xyz": V_mean[nose], "extent_mm": ext}


def _write_gt(name: str) -> str:
    idx, F = _TEMPLATE["vertex_indices"], _TEMPLATE["faces"]
    with np.load(common.gt_mesh_path(name)) as z:
        V = np.asarray(z["V"], dtype=np.float64)[idx]
    mo.save_variant(V, F, _TEMPLATE["out_dir"] / f"{name}.npz")
    return name


# ------------------------------------------------------------------------ recon

def load_recon(method: str, name: str, out_root: Path):
    """(vertici destrorsi, facce, landmark destrorsi) di una ricostruzione."""
    d = common.recon_dir(method, out_root)
    with np.load(d / f"{name}.npz") as z:
        V, F = flip_to_right_handed(z["V"]), np.asarray(z["F"], dtype=np.int32)
    with open(d / f"{name}.json", encoding="utf-8") as fh:
        lmk = flip_to_right_handed(np.asarray(json.load(fh)["landmarks_68"], dtype=np.float64))
    return V, F, lmk


def canonical(V: np.ndarray, lmk: np.ndarray) -> np.ndarray:
    """Origine nella punta del naso, unita' la distanza fra gli angoli esterni degli occhi."""
    eye_span = float(np.linalg.norm(lmk[common.LMK_EYE_OUTER[0]] - lmk[common.LMK_EYE_OUTER[1]]))
    return (V - lmk[common.LMK_NOSE_TIP][None, :]) / max(eye_span, 1e-9)


def build_recon_template(method: str, names: list[str], target_faces: int, ex_ex_mm: float,
                         args) -> dict:
    """Indici della regione volto sulla topologia (fissa) del metodo, dalla mesh media."""
    sample = names[:: max(1, len(names) // args.mean_sample)][:args.mean_sample]
    acc, F = None, None
    for name in sample:
        V, F, lmk = load_recon(method, name, args.out_root)
        Vc = canonical(V, lmk)
        acc = Vc if acc is None else acc + Vc
    V_mean = acc / len(sample)
    radius = args.radius_mm / ex_ex_mm
    idx, F_face = crop_region(V_mean, F, np.zeros(3), radius)

    ext = V_mean[idx].max(axis=0) - V_mean[idx].min(axis=0)
    print(f"[ws3b-prep] {method}: media su {len(sample)} mesh, ritaglio {len(idx)}/{len(V_mean)} "
          f"vertici e {len(F_face)} triangoli, raggio {radius:.4f} aperture oculari "
          f"({args.radius_mm:.1f} mm / ex-ex GT {ex_ex_mm:.2f} mm), "
          f"extent {np.round(ext, 3)} aperture oculari; decimazione a {target_faces} triangoli",
          flush=True)
    return {"vertex_indices": idx, "faces": np.asarray(F_face, dtype=np.int32),
            "target_faces": int(target_faces), "radius_eye_spans": radius,
            "ex_ex_mm": float(ex_ex_mm)}


def _write_recon(name: str) -> str:
    idx, F = _TEMPLATE["vertex_indices"], _TEMPLATE["faces"]
    V, _, _ = load_recon(_TEMPLATE["method"], name, _TEMPLATE["out_root"])
    Vd, Fd = mo.decimate_to(V[idx], F, _TEMPLATE["target_faces"])
    mo.save_variant(Vd, Fd, _TEMPLATE["out_dir"] / f"{name}.npz")
    return name


# ----------------------------------------------------------------------------

def run_pool(fn, names: list[str], workers: int, tag: str) -> None:
    t0 = time.time()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(fn, names, chunksize=32), start=1):
            if i % 1000 == 0:
                print(f"[ws3b-prep] {tag}: {i}/{len(names)} "
                      f"({i / max(time.time() - t0, 1e-9):.0f}/s)", flush=True)
    print(f"[ws3b-prep] {tag}: {len(names)} mesh in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    args = parse_args()
    args.out_root = args.out_root.resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    region_npz = args.out_root / "face_region.npz"

    if args.stage in ("all", "gt"):
        template = build_gt_template(args)
        out_dir = common.gt_face_dir(args.out_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(region_npz, vertex_indices=template["vertex_indices"], faces=template["faces"])
        common.face_region_path(args.out_root).write_text(json.dumps({
            "radius_mm": args.radius_mm,
            "source_topology": "datasets/Multiface/prep/tracked",
            "n_tracked_meshes": len(template["names"]),
            "nose_tip_vertex": template["nose_vertex"],
            "nose_tip_xyz_mm": [float(x) for x in template["nose_xyz"]],
            "n_vertices": int(len(template["vertex_indices"])),
            "n_faces": int(len(template["faces"])),
            "extent_mm": [float(x) for x in template["extent_mm"]],
        }, indent=2), encoding="utf-8")
        todo = [n for n in template["names"]
                if args.overwrite or not (out_dir / f"{n}.npz").is_file()]
        _TEMPLATE.clear()
        _TEMPLATE.update(template, out_dir=out_dir)
        print(f"[ws3b-prep] gt_face: {len(todo)}/{len(template['names'])} da scrivere", flush=True)
        if todo:
            run_pool(_write_gt, todo, args.workers, "gt_face")

    if args.stage in ("all", "recon"):
        if not region_npz.is_file():
            raise SystemExit(f"manca {region_npz}: gira prima --stage gt")
        with np.load(region_npz) as z:
            target_faces = int(len(z["faces"]))
        ex_ex_mm = gt_outer_canthal_mm(args.out_root, args.ex_ex_mm, args.ex_ex_source)
        items = common.load_manifest(common.manifest_path(args.out_root))
        for method in methods:
            src = common.recon_dir(method, args.out_root)
            names = sorted(it.name for it in items if (src / f"{it.name}.npz").is_file())
            if len(names) != len(items):
                print(f"[ws3b-prep] ATTENZIONE {method}: {len(names)}/{len(items)} mesh presenti",
                      flush=True)
            if not names:
                raise SystemExit(f"nessuna ricostruzione in {src}")
            template = build_recon_template(method, names, target_faces, ex_ex_mm, args)
            out_dir = common.recon_face_dir(method, args.out_root)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(out_dir.parent / f"template_{method}.npz",
                     vertex_indices=template["vertex_indices"], faces=template["faces"],
                     target_faces=template["target_faces"],
                     radius_eye_spans=template["radius_eye_spans"],
                     ex_ex_mm=template["ex_ex_mm"], radius_mm=args.radius_mm)
            todo = [n for n in names if args.overwrite or not (out_dir / f"{n}.npz").is_file()]
            _TEMPLATE.clear()
            _TEMPLATE.update(template, method=method, out_dir=out_dir, out_root=args.out_root)
            print(f"[ws3b-prep] {method}: {len(todo)}/{len(names)} da scrivere", flush=True)
            if todo:
                run_pool(_write_recon, todo, args.workers, f"recon_face/{method}")


if __name__ == "__main__":
    main()
