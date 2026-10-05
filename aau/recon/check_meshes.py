#!/usr/bin/env python3
"""Controlla le mesh prodotte dai runner e stampa la tabella di riepilogo (WS3b).

  aau/recon/run_recon.sh ddfa aau/recon/check_meshes.py aau/runs/recon/3ddfa_v2 ...

Per ogni directory verifica che: ci sia un npz per immagine con le chiavi V/F e i dtype
giusti, nessun NaN o infinito, gli indici delle facce dentro il range dei vertici, la
topologia sia la STESSA per tutte le immagini (e' il presupposto di WS3b: i metodi hanno
topologia fissa, quindi le mesh sono confrontabili vertice per vertice all'interno di uno
stesso metodo) e il bounding box sia plausibile, cioe' dell'ordine di grandezza del box
del volto salvato nel json.  Esce con codice 1 se un controllo fallisce.

Con ``--landmarks-ref <dir>`` confronta anche i 68 landmark con quelli di un altro
metodo, nello stesso spazio immagine: e' il controllo incrociato che serve a SynergyNet,
la cui tabella 3dmm_data e' stata ricostruita da un altro repo (vedi synergynet_run.py).
Due metodi diversi non devono dare gli stessi landmark al pixel, ma se una base PCA fosse
sbagliata lo scarto salterebbe a una frazione grossa del volto.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dirs", nargs="+", type=Path, help="directory di output di un metodo")
    p.add_argument("--expect", type=int, default=0,
                   help="numero di mesh attese per directory (0 = non controllare)")
    p.add_argument("--landmarks-ref", type=Path, default=None,
                   help="directory di un altro metodo con cui confrontare i 68 landmark")
    p.add_argument("--landmarks-tol", type=float, default=0.15,
                   help="scarto medio massimo ammesso, in frazione del lato del box")
    return p.parse_args()


def landmark_gap(meta: dict, ref_meta: dict) -> tuple[float, float]:
    """Scarto medio e massimo fra i 68 landmark di due metodi, in pixel (solo x, y).

    La z non entra: i metodi la riportano con origini diverse (3DDFA_V2 sottrae il minimo),
    quindi sarebbe un confronto fra convenzioni, non fra ricostruzioni.
    """
    a = np.asarray(meta["landmarks_68"], dtype=np.float64)[:, :2]
    b = np.asarray(ref_meta["landmarks_68"], dtype=np.float64)[:, :2]
    d = np.linalg.norm(a - b, axis=1)
    return float(d.mean()), float(d.max())


def check_dir(d: Path, expect: int, ref: Path | None, tol: float) -> tuple[bool, list[str]]:
    lines = []
    ok = True
    meshes = sorted(d.glob("*.npz"))
    if not meshes:
        return False, [f"{d}: NESSUN npz"]
    if expect and len(meshes) != expect:
        ok = False
        lines.append(f"{d.name}: attese {expect} mesh, trovate {len(meshes)}")

    topo = set()
    times = []
    for fp in meshes:
        z = np.load(fp)
        if "V" not in z or "F" not in z:
            ok = False
            lines.append(f"  {fp.name}: chiavi {list(z.keys())}, attese V e F")
            continue
        V, F = z["V"], z["F"]
        st = common.mesh_stats(V, F)
        topo.add((st["n_vertices"], st["n_faces"]))

        problems = []
        if V.dtype != np.float32:
            problems.append(f"V dtype {V.dtype}")
        if F.dtype != np.int32:
            problems.append(f"F dtype {F.dtype}")
        if not st["finite"]:
            problems.append("NaN/inf in V")
        if not st["faces_in_range"]:
            problems.append("indici di F fuori range")
        if st["finite"] and min(st["bbox_extent"]) <= 0:
            problems.append(f"bbox degenere {st['bbox_extent']}")

        meta_fp = fp.with_suffix(".json")
        box_side = float("nan")
        if meta_fp.is_file():
            meta = json.loads(meta_fp.read_text())
            times.append(meta.get("seconds", float("nan")))
            box = meta.get("face_box") or meta.get("face_box_lrtb")
            if box:
                # face_box e' (x1, y1, x2, y2), face_box_lrtb e' (left, right, top, bottom)
                box_side = (abs(box[2] - box[0]) if "face_box" in meta
                            else abs(box[1] - box[0]))
            if ref is not None and ref.resolve() != d:
                ref_fp = ref / meta_fp.name
                if not ref_fp.is_file():
                    problems.append(f"manca il json di riferimento {ref_fp}")
                else:
                    gap, gap_max = landmark_gap(meta, json.loads(ref_fp.read_text()))
                    lines.append(f"    landmark vs {ref.name}: medio {gap:.1f} px, "
                                 f"max {gap_max:.1f} px (lato box {box_side:.0f} px)")
                    if box_side > 0 and gap > tol * box_side:
                        problems.append(f"landmark a {gap / box_side:.0%} del lato del box")
            # Il volto ricostruito deve stare nell'ordine di grandezza del box del volto:
            # un fattore 10 in piu' o in meno vuol dire crop o scala sbagliati.
            if st["finite"] and box_side == box_side and box_side > 0:
                ratio = max(st["bbox_extent"][:2]) / box_side
                if not 0.2 < ratio < 5.0:
                    problems.append(f"bbox/box del volto = {ratio:.2f}, fuori [0.2, 5]")
        else:
            problems.append("json della posa mancante")

        ext = ", ".join(f"{e:.1f}" for e in st["bbox_extent"])
        lines.append(f"  {fp.name}: {st['n_vertices']} v, {st['n_faces']} f, "
                     f"bbox ({ext}) px" + ("  -> " + "; ".join(problems) if problems else ""))
        if problems:
            ok = False

    if len(topo) > 1:
        ok = False
        lines.append(f"{d.name}: topologia NON fissa fra le immagini: {sorted(topo)}")
    if times:
        lines.append(f"{d.name}: {np.nanmean(times):.3f} s/immagine "
                     f"(mediana {np.nanmedian(times):.3f} s, n={len(times)})")
    lines.insert(0, f"{d}: {len(meshes)} mesh, topologia {sorted(topo)}")
    return ok, lines


def main() -> None:
    args = parse_args()
    all_ok = True
    for d in args.dirs:
        ok, lines = check_dir(d.resolve(), args.expect, args.landmarks_ref, args.landmarks_tol)
        all_ok &= ok
        print("\n".join(lines), flush=True)
        print(f"  --> {'OK' if ok else 'FALLITO'}\n", flush=True)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
