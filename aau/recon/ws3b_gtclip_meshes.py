#!/usr/bin/env python3
"""Le ricostruzioni ritagliate alla patch GT di ``chamfer_gtclip_mm``, come mesh (WS3b).

    aau/submit.sh recon/ws3b_gtclip_prep.sbatch
    aau/run.sh aau/recon/ws3b_gtclip_meshes.py --max-items 20     # controllo, niente file

Serve alla domanda "la discordanza della metrica appresa scompare a parita' di patch?".
Oggi il latent legge ``recon_face`` (ritaglio a raggio scalato in aperture oculari, poi
decimazione), mentre ``chamfer_gtclip_mm`` usa la stessa ``recon_face`` allineata con
l'ICP e ristretta alla sfera di 95 mm attorno al pronasale della GT, nel frame della GT.
Qui si produce quella stessa patch come MESH, perche' l'encoder vuole facce e operatori:

  1. similarita' dall'ICP rifatta esattamente come in ``ws3b_geometric._gt_chunk`` (stesso
     indice globale dell'elemento come seme, stessi punti e iterazioni), quindi lo stesso
     ``inside`` vertice per vertice;
  2. facce di ``recon_face`` con i tre vertici dentro, poi ``mesh_ops.prepare_open_surface``
     (degeneri, schegge, orfani), la stessa pulizia di ``ws3b_prepare_meshes.crop_region``;
  3. la mesh si salva nelle coordinate di ``recon_face`` (pixel, frame destrorso), cioe' nel
     frame in cui il latent la leggeva gia': cambia SOLO il supporto, non la posa.  La
     normalizzazione maxabs di ``GTReadyDatasetNPZ`` toglie comunque scala e traslazione.

Il controllo che la patch sia davvero quella: ``chamfer_gtclip_mm`` viene ricalcolata
sugli stessi vertici e confrontata con la colonna di ``gt_metrics_<metodo>.csv``; lo
scarto massimo si stampa, e oltre 1e-6 mm lo script si ferma senza scrivere il csv.
Nel csv ``gtclip_meshes_<metodo>.csv`` ci sono anche la frazione dei vertici dentro la
sfera che sopravvive al filtro sulle facce e il rapporto d'area patch/GT dopo il ritaglio.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ws3b_common as common  # noqa: E402
from ws3b_geometric import gt_face_landmarks, load_npz, surface_area  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT / "faceBench" / "latentVSpipeline"))
sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "genict"))
import mesh_ops as mo  # noqa: E402

# Stato condiviso coi worker via fork.
_STATE: dict = {}

# Scarto massimo ammesso fra la chamfer_gtclip_mm ricalcolata e quella del csv: il csv la
# scrive con 10 cifre significative, su valori di 2-3 mm.
CHECK_TOL_MM = 1e-6

MEASURES = ("chamfer_gtclip_mm_check", "n_recon_face_verts", "n_inside", "n_kept_verts",
            "n_kept_faces", "kept_over_inside", "gtclip_area_mm2", "gtclip_area_ratio")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--methods", type=str, default=",".join(common.METHODS))
    # Questi tre DEVONO restare quelli della corsa di ws3b_geometric che ha scritto il csv:
    # altrimenti la similarita' cambia e la patch non e' piu' quella di chamfer_gtclip_mm.
    p.add_argument("--sample-points", type=int, default=4096)
    p.add_argument("--icp-points", type=int, default=4096)
    p.add_argument("--icp-iter", type=int, default=30)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--max-items", type=int, default=0,
                   help=">0: gira solo sui primi N elementi e NON scrive niente (controllo)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _chunk(items):
    """Ritaglio alla patch GT per un blocco di ricostruzioni."""
    import fg_metrics as fg

    method, out_root = _STATE["method"], _STATE["out_root"]
    lmk_gt, write = _STATE["lmk_gt"], _STATE["write"]
    out_dir = common.recon_face_gtclip_dir(method, out_root)
    out = []
    for index, name, gt_name in items:
        Vg, Fg = load_npz(common.gt_face_dir(out_root) / f"{gt_name}.npz")
        Vr, _ = load_npz(common.recon_dir(method, out_root) / f"{name}.npz")
        Vr[:, 1] *= -1.0  # spazio immagine -> frame testa, vedi ws3b_common
        Vf, Ff = load_npz(common.recon_face_dir(method, out_root) / f"{name}.npz")

        # Le righe (ii), (iv) e (v) di ws3b_geometric._gt_chunk, con gli stessi semi.
        Y = fg.sample_vertices(Vr, _STATE["icp_points"], seed=index + 1)
        Vg_aligned = fg.rigid_icp_align(Vg, Y, max_points=_STATE["icp_points"],
                                        max_iter=_STATE["icp_iter"], seed=index)
        scale, R, trans = fg.procrustes_transform(Vg, Vg_aligned, scaling=True)
        Vf_mm = (Vf - trans) @ R.T / max(scale, 1e-12)
        Zmm = fg.sample_vertices(Vg, _STATE["sample_points"], seed=index + 1)
        center = Vg[lmk_gt[common.LMK7_NAMES.index("prn")]]
        inside = np.linalg.norm(Vf_mm - center[None, :], axis=1) <= common.FACE_RADIUS_MM
        n_inside = int(inside.sum())
        check = float(fg.symmetric_chamfer(
            fg.sample_vertices(Vf_mm[inside], _STATE["sample_points"], seed=index), Zmm)) \
            if n_inside >= 3 else float("nan")

        # La stessa patch come mesh: facce coi tre vertici dentro, poi la pulizia di
        # crop_region.  Coordinate di recon_face, non quelle allineate (vedi docstring).
        F_in = Ff[inside[Ff].all(axis=1)]
        if len(F_in) == 0:
            out.append({"chamfer_gtclip_mm_check": check, "n_recon_face_verts": len(Vf),
                        "n_inside": n_inside, "n_kept_verts": 0, "n_kept_faces": 0,
                        "kept_over_inside": 0.0, "gtclip_area_mm2": float("nan"),
                        "gtclip_area_ratio": float("nan")})
            continue
        Vc, Fc = mo.prepare_open_surface(Vf, F_in)
        area = surface_area(Vc, Fc) / max(scale, 1e-12) ** 2
        if write:
            mo.save_variant(Vc, Fc, out_dir / f"{name}.npz")
        out.append({
            "chamfer_gtclip_mm_check": check,
            "n_recon_face_verts": int(len(Vf)), "n_inside": n_inside,
            "n_kept_verts": int(len(Vc)), "n_kept_faces": int(len(Fc)),
            "kept_over_inside": float(len(Vc) / max(n_inside, 1)),
            "gtclip_area_mm2": float(area),
            "gtclip_area_ratio": float(area / max(surface_area(Vg, Fg), 1e-12)),
        })
    return out


def run_method(method: str, items, args) -> None:
    csv_path = common.gtclip_mesh_csv_path(method, args.out_root)
    if csv_path.exists() and not args.overwrite and args.max_items <= 0:
        print(f"[ws3b-gtclip] {method}: {csv_path.name} c'e' gia', salto", flush=True)
        return
    reference = {r["name"]: float(r["chamfer_gtclip_mm"])
                 for r in common.read_rows(common.gt_csv_path(method, args.out_root))}
    write = args.max_items <= 0
    if write:
        common.recon_face_gtclip_dir(method, args.out_root).mkdir(parents=True, exist_ok=True)
    # Indice globale dell'elemento = seme, come in ws3b_geometric.run_gt.
    tasks = [[(start + k, it.name, it.gt_name)
              for k, it in enumerate(items[start:start + args.chunk])]
             for start in range(0, len(items), args.chunk)]

    t0 = time.time()
    _STATE.clear()
    _STATE.update(method=method, out_root=args.out_root, write=write,
                  lmk_gt=gt_face_landmarks(args.out_root), icp_points=args.icp_points,
                  icp_iter=args.icp_iter, sample_points=args.sample_points)
    with mp.get_context("fork").Pool(processes=args.workers) as pool:
        results = [r for block in pool.map(_chunk, tasks) for r in block]
    seconds = time.time() - t0

    diffs = np.asarray([abs(res["chamfer_gtclip_mm_check"] - reference[it.name])
                        for it, res in zip(items, results)
                        if np.isfinite(res["chamfer_gtclip_mm_check"])])
    worst = float(diffs.max()) if diffs.size else float("nan")
    n_empty = sum(1 for res in results if res["n_kept_faces"] == 0)
    med = {k: float(np.nanmedian([res[k] for res in results])) for k in MEASURES}
    print(f"[ws3b-gtclip] {method}: {len(results)} ricostruzioni in {seconds:.0f}s; "
          f"chamfer_gtclip_mm ricalcolata vs csv: scarto max {worst:.3g} mm su {diffs.size}\n"
          f"    mediane: vertici {med['n_recon_face_verts']:.0f} -> dentro {med['n_inside']:.0f} "
          f"-> tenuti {med['n_kept_verts']:.0f} ({med['kept_over_inside']:.1%} dei dentro), "
          f"{med['n_kept_faces']:.0f} triangoli, area patch/GT {med['gtclip_area_ratio']:.4f}; "
          f"patch vuote: {n_empty}", flush=True)
    if not worst <= CHECK_TOL_MM:
        raise SystemExit(f"{method}: la patch NON e' quella di chamfer_gtclip_mm "
                         f"(scarto {worst:.3g} mm > {CHECK_TOL_MM}): csv non scritto")
    if not write:
        print(f"[ws3b-gtclip] {method}: CONTROLLO su {len(results)} elementi, niente scritto",
              flush=True)
        return
    rows = [{**{k: getattr(it, k) for k in common.ITEM_FIELDS}, **res}
            for it, res in zip(items, results)]
    common.write_rows(csv_path, common.ITEM_FIELDS + MEASURES, rows, method=method,
                      seconds=seconds, icp_points=args.icp_points, icp_iter=args.icp_iter,
                      sample_points=args.sample_points, radius_mm=common.FACE_RADIUS_MM,
                      check_max_abs_diff_mm=worst,
                      frame="coordinate di recon_face (pixel, destrorso), solo il supporto cambia")


def main() -> None:
    args = parse_args()
    args.out_root = args.out_root.resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    items = common.load_manifest(common.manifest_path(args.out_root))
    for method in methods:
        # Stesso filtro di ws3b_geometric.main: l'indice (= seme) e' la posizione in `mine`.
        present = {p.stem for p in common.recon_face_dir(method, args.out_root).glob("*.npz")}
        mine = [it for it in items if it.name in present]
        if args.max_items > 0:
            mine = mine[: args.max_items]
        run_method(method, mine, args)


if __name__ == "__main__":
    main()
