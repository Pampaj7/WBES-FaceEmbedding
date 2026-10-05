#!/usr/bin/env python3
"""I criteri geometrici di WS3b: NoW vero, NoW con ICP, e Chamfer grezza.

    aau/submit.sh recon/ws3b_geometric.sbatch
    aau/run.sh aau/recon/ws3b_geometric.py --mode gt --workers 16

Due modi.

``--mode gt``  una riga per ricostruzione, contro la GT dello stesso frame.

  ``now_median`` / ``now_mean``
      Il protocollo NoW, quello vero.  La similarita' (rotazione + traslazione + scala) e'
      stimata con Procrustes sui **7 landmark** -- exocanthion e endocanthion destro e
      sinistro, pronasale, cheilion destro e sinistro -- e non c'e' nessun ICP.  I 7 punti
      sulla ricostruzione sono sette dei 68 iBUG che ogni metodo restituisce; sulla GT sono
      sette indici di vertice fissi sulla topologia tracciata, stimati da
      ``ws3b_landmarks.py``.  Poi distanza **punto-superficie** dai vertici della GT
      ritagliata alla superficie ricostruita INTERA, in millimetri.
      La differenza con il criterio qui sotto non e' cosmetica: l'ICP ha 7 gradi di
      liberta' liberi di inseguire la superficie e puo' compensare un errore di forma
      ruotando o riscalando, i 7 landmark no.

  ``sim_icp_p2s_median`` / ``sim_icp_p2s_mean``
      Lo stesso punto-superficie, ma con la similarita' stimata con l'ICP del repo
      (``faceBench/latentVSpipeline/fg_metrics.rigid_icp_align``, 30 iterazioni, Procrustes
      con scala a ogni passo) su tutta la superficie.  **Non e' il protocollo NoW**, e per
      questo non si chiama piu' ``now_median`` come nella prima versione: e' un criterio
      allineato in senso forte, che qui serve come termine di paragone del NoW vero.
      L'ICP va dalla GT alla ricostruzione e non viceversa: cosi' la sorgente e' la stessa
      per tutti e tre i metodi, e i vicini piu' prossimi si cercano nella nuvola piu' fitta.
      La similarita' viene poi riletta con ``procrustes_transform`` fra la GT di partenza e
      quella allineata (la composizione di similarita' e' una similarita', quindi il
      recupero e' esatto) e invertita, cosi' la misura si fa in millimetri veri e non in
      pixel: e' l'unica scala in cui i tre metodi sono confrontabili, visto che nessuno dei
      tre osserva la scala metrica.

  ``chamfer_raw``
      Il criterio SENZA allineamento.  Chamfer simmetrica fra GT ritagliata e ricostruzione
      ritagliata, ognuna centrata sulla media e divisa per il proprio maxabs -- la
      normalizzazione di ``dataset_gtready`` e di ``aau/baselines/chamfer_matrix.py`` -- su
      4096 punti per lato, seme = indice per un lato e indice+1 per l'altro, cioe' il
      protocollo ``--variant facebench`` gia' usato per la Tabella 2 estesa.  Qui la
      ricostruzione entra ritagliata: la maxabs e' l'estensione della mesh, e senza
      ritaglio il criterio misurerebbe soprattutto quanto collo e orecchie ci sono nella
      topologia del metodo.

  ``chamfer_icp_mm``
      La stessa Chamfer, ma **in millimetri**: la scala non e' la maxabs per mesh, che e'
      una statistica di un vertice solo e quindi diversa per le due mesh e per i tre metodi,
      e' quella dell'ICP -- la sola grandezza osservabile che lega i pixel del metodo ai
      millimetri della GT.  La ricostruzione ritagliata viene portata nel frame della GT con
      ``(V - trans) @ R.T / scale`` (la stessa similarita' di ``sim_icp_p2s_*``, che percio'
      non costa niente in piu') e poi la Chamfer si fa in quello spazio, senza nessun'altra
      normalizzazione.  Serve a leggere ``chamfer_raw``: se le due righe non danno la stessa
      classifica, la differenza fra i metodi che ``chamfer_raw`` vede e' in parte la
      differenza fra i loro maxabs.

  ``chamfer_gtclip_mm``
      Come sopra, ma con la ricostruzione allineata **ritagliata alla patch della GT**: la
      stessa sfera di 95 mm, attorno allo stesso pronasale, nello stesso frame.  Il supporto
      delle due nuvole e' quindi uguale per costruzione, e non per taratura di un raggio.
      E' il controllo che manca a tutte le altre righe: il rapporto di area recon/GT vale
      1.148 / 1.063 / 1.087 per 3DDFA_V2 / SynergyNet / PRNet, cioe' i supporti NON
      coincidono, e una Chamfer fra supporti diversi paga anche la corona di superficie che
      c'e' da una parte e non dall'altra.  La colonna ``recon_inside_gt_patch`` dice che
      frazione dei vertici della ricostruzione allineata sopravvive al ritaglio.

  ``scale_px_per_mm`` / ``eye_span_mm`` / ``crop_radius_mm`` / ``area_*`` / ``area_ratio``
      Controlli, non risultati.  La scala dell'ICP converte i pixel in millimetri; da li'
      si leggono la distanza ex-ex della ricostruzione in millimetri veri, il raggio del
      ritaglio in millimetri veri (deve cadere sui 95 della GT) e l'area delle due patch,
      il cui rapporto dice se i due supporti coincidono davvero.  Con il vecchio raggio
      (95/90 aperture oculari) il ritaglio della ricostruzione veniva 90.7-91.6 mm invece
      di 95, cioe' ~8% di area in meno della GT.

``--mode pairs``  una riga per coppia di ricostruzioni del protocollo di identita', con la
  stessa ``chamfer_raw`` del modo gt.  Serve all'AUC stesso/diverso soggetto: se la
  ricostruzione conserva l'identita', due ricostruzioni dello stesso soggetto devono
  risultare piu' vicine di due ricostruzioni di soggetti diversi, senza mai passare da una
  verita' a terra.
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

sys.path.insert(0, str(common.REPO_ROOT / "faceBench" / "latentVSpipeline"))

# Stato condiviso coi worker via fork.
_STATE: dict = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--mode", type=str, default="all", choices=("all", "gt", "pairs"))
    p.add_argument("--methods", type=str, default=",".join(common.METHODS))
    p.add_argument("--sample-points", type=int, default=4096,
                   help="punti per lato della chamfer, default del protocollo facebench")
    p.add_argument("--icp-points", type=int, default=4096)
    p.add_argument("--icp-iter", type=int, default=30)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--max-items", type=int, default=0,
                   help=">0: gira solo sui primi N elementi e NON scrive il csv (controllo)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def maxabs_normalize(V: np.ndarray) -> np.ndarray:
    """Centro sulla media, scala sul massimo valore assoluto: come dataset_gtready."""
    Vc = V - V.mean(axis=0, keepdims=True)
    scale = float(np.max(np.abs(Vc)))
    return Vc / scale if scale > 1e-6 else Vc * 0.0


def load_npz(path: Path):
    with np.load(path) as z:
        return np.asarray(z["V"], dtype=np.float64), np.asarray(z["F"], dtype=np.int64)


def _init_worker(icp_points: int, icp_iter: int, sample_points: int) -> None:
    """Solo i parametri scalari.

    Tutto il resto (metodo, percorsi, e soprattutto le mesh gia' normalizzate del modo
    pairs, ~400 MB) sta in ``_STATE`` prima che la pool nasca, e arriva ai worker con il
    fork in copy-on-write: passarlo per ``initargs`` vorrebbe dire serializzarlo una volta
    per worker, come in aau/multiface/ws3a_geometric.py.
    """
    _STATE.update(icp_points=icp_points, icp_iter=icp_iter, sample_points=sample_points)


def surface_area(V: np.ndarray, F: np.ndarray) -> float:
    tri = V[np.asarray(F, dtype=np.int64)]
    return float(0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def point_to_surface(Vg: np.ndarray, Vr: np.ndarray, Fr: np.ndarray) -> np.ndarray:
    """Distanza da ogni vertice della GT alla superficie ricostruita, in millimetri."""
    import igl

    sqrD, _, _ = igl.point_mesh_squared_distance(
        np.ascontiguousarray(Vg), np.ascontiguousarray(Vr), np.ascontiguousarray(Fr))
    return np.sqrt(np.maximum(np.asarray(sqrD, dtype=np.float64), 0.0))


def _gt_chunk(items):
    """I criteri geometrici per un blocco di ricostruzioni."""
    import json

    import fg_metrics as fg

    out = []
    method, out_root = _STATE["method"], _STATE["out_root"]
    lmk_gt, radius_eye_spans = _STATE["lmk_gt"], _STATE["radius_eye_spans"]
    for index, name, gt_name, eye_span_px in items:
        Vg, Fg = load_npz(common.gt_face_dir(out_root) / f"{gt_name}.npz")
        Vr, Fr = load_npz(common.recon_dir(method, out_root) / f"{name}.npz")
        Vr[:, 1] *= -1.0  # spazio immagine -> frame testa, vedi ws3b_common
        Vf, Ff = load_npz(common.recon_face_dir(method, out_root) / f"{name}.npz")
        with open(common.recon_dir(method, out_root) / f"{name}.json", encoding="utf-8") as fh:
            lmk = np.asarray(json.load(fh)["landmarks_68"], dtype=np.float64)
        lmk[:, 1] *= -1.0

        # (i) NoW vero: similarita' dai soli 7 landmark, nessun ICP.
        s_now, R_now, t_now = fg.procrustes_transform(
            lmk[list(common.LMK7_IBUG)], Vg[lmk_gt], scaling=True)
        d_now = point_to_surface(Vg, s_now * (Vr @ R_now) + t_now, Fr)

        # (ii) stessa distanza, ma similarita' dall'ICP su tutta la superficie: NON e' NoW.
        Y = fg.sample_vertices(Vr, _STATE["icp_points"], seed=index + 1)
        Vg_aligned = fg.rigid_icp_align(Vg, Y, max_points=_STATE["icp_points"],
                                        max_iter=_STATE["icp_iter"], seed=index)
        scale, R, trans = fg.procrustes_transform(Vg, Vg_aligned, scaling=True)
        Vr_mm = (Vr - trans) @ R.T / max(scale, 1e-12)
        d = point_to_surface(Vg, Vr_mm, Fr)

        # (iii) Chamfer grezza: sola normalizzazione maxabs, nessun allineamento.
        X = fg.sample_vertices(maxabs_normalize(Vf), _STATE["sample_points"], seed=index)
        Z = fg.sample_vertices(maxabs_normalize(Vg), _STATE["sample_points"], seed=index + 1)

        # (iv) la stessa Chamfer in millimetri, con la scala dell'ICP al posto della maxabs
        # per mesh. La similarita' e' gia' stata stimata al punto (ii): qui si riusa.
        Vf_mm = (Vf - trans) @ R.T / max(scale, 1e-12)
        Zmm = fg.sample_vertices(Vg, _STATE["sample_points"], seed=index + 1)
        chamfer_icp_mm = float(fg.symmetric_chamfer(
            fg.sample_vertices(Vf_mm, _STATE["sample_points"], seed=index), Zmm))

        # (v) e con la ricostruzione ritagliata alla patch della GT: stessa sfera, stesso
        # centro, stesso frame, quindi supporto uguale per costruzione.
        center = Vg[lmk_gt[common.LMK7_NAMES.index("prn")]]
        inside = np.linalg.norm(Vf_mm - center[None, :], axis=1) <= common.FACE_RADIUS_MM
        n_inside = int(inside.sum())
        chamfer_gtclip_mm = float(fg.symmetric_chamfer(
            fg.sample_vertices(Vf_mm[inside], _STATE["sample_points"], seed=index), Zmm)) \
            if n_inside >= 3 else float("nan")

        # Controlli sul supporto: la scala dell'ICP porta le aree della patch ricostruita
        # dai pixel ai millimetri quadrati, dove sono confrontabili con quelle della GT.
        area_gt = surface_area(Vg, Fg)
        area_recon = surface_area(Vf, Ff) / max(scale, 1e-12) ** 2
        out.append({
            "now_median": float(np.median(d_now)), "now_mean": float(d_now.mean()),
            "sim_icp_p2s_median": float(np.median(d)), "sim_icp_p2s_mean": float(d.mean()),
            "chamfer_raw": float(fg.symmetric_chamfer(X, Z)),
            "chamfer_icp_mm": chamfer_icp_mm,
            "chamfer_gtclip_mm": chamfer_gtclip_mm,
            "recon_inside_gt_patch": float(n_inside / max(len(inside), 1)),
            "scale_px_per_mm": float(scale),
            "eye_span_mm": float(eye_span_px / max(scale, 1e-12)),
            "crop_radius_mm": float(radius_eye_spans * eye_span_px / max(scale, 1e-12)),
            "gt_face_area_mm2": area_gt,
            "recon_face_area_mm2": area_recon,
            "area_ratio": float(area_recon / max(area_gt, 1e-12)),
        })
    return out


def _pairs_chunk(items):
    """chamfer_raw fra due ricostruzioni gia' normalizzate e in memoria condivisa."""
    import fg_metrics as fg

    verts = _STATE["verts"]
    out = []
    for index, name_a, name_b in items:
        X = fg.sample_vertices(verts[name_a], _STATE["sample_points"], seed=index)
        Y = fg.sample_vertices(verts[name_b], _STATE["sample_points"], seed=index + 1)
        out.append(float(fg.symmetric_chamfer(X, Y)))
    return out


# ----------------------------------------------------------------------------

def eye_spans(method: str, names: list[str], out_root: Path) -> dict[str, float]:
    """Distanza fra i landmark 36 e 45, in pixel, letta dai json dei runner."""
    import json

    d = common.recon_dir(method, out_root)
    out = {}
    for name in names:
        with open(d / f"{name}.json", encoding="utf-8") as fh:
            lmk = np.asarray(json.load(fh)["landmarks_68"], dtype=np.float64)
        out[name] = float(np.linalg.norm(lmk[common.LMK_EYE_OUTER[0]] - lmk[common.LMK_EYE_OUTER[1]]))
    return out


def gt_face_landmarks(out_root: Path) -> np.ndarray:
    """I 7 landmark NoW rinumerati sui vertici di ``gt_face``.

    ``ws3b_landmarks.py`` li scrive nella numerazione della topologia tracciata; qui serve
    la loro posizione dentro il ritaglio, che e' una sottolista ORDINATA di quei vertici.
    """
    import json

    path = common.landmarks_path(out_root)
    if not path.is_file():
        raise SystemExit(f"manca {path}: il criterio NoW ha bisogno dei 7 landmark sulla GT.\n"
                         f"  Crealo con: aau/run.sh aau/recon/ws3b_landmarks.py")
    vertices = json.loads(path.read_text())["vertex_indices"]
    with np.load(out_root / "face_region.npz") as z:
        kept = np.asarray(z["vertex_indices"], dtype=np.int64)
    wanted = np.asarray([vertices[n] for n in common.LMK7_NAMES], dtype=np.int64)
    pos = np.searchsorted(kept, wanted)
    missing = [n for n, p, w in zip(common.LMK7_NAMES, pos, wanted)
               if p >= len(kept) or kept[p] != w]
    if missing:
        raise SystemExit(f"landmark fuori dal ritaglio gt_face: {missing}. "
                         f"Il raggio di {common.FACE_RADIUS_MM} mm non li contiene tutti.")
    return pos


def recon_template_params(method: str, out_root: Path) -> dict:
    """Raggio del ritaglio e ex-ex usati da ws3b_prepare_meshes per questo metodo."""
    path = common.recon_face_dir(method, out_root).parent / f"template_{method}.npz"
    if not path.is_file():
        raise SystemExit(f"manca {path}: gira prima ws3b_prepare_meshes.py --stage recon")
    with np.load(path) as z:
        if "radius_eye_spans" not in z.files:
            raise SystemExit(f"{path} viene da una prep vecchia, senza il raggio: "
                             f"rifai ws3b_prepare_meshes.py --stage recon --overwrite")
        return {"radius_eye_spans": float(z["radius_eye_spans"]),
                "ex_ex_mm": float(z["ex_ex_mm"])}


def run_gt(method: str, items, args) -> None:
    path = common.gt_csv_path(method, args.out_root)
    if path.exists() and not args.overwrite and args.max_items <= 0:
        print(f"[ws3b-geom] {method}: {path.name} c'e' gia', salto", flush=True)
        return
    spans = eye_spans(method, [it.name for it in items], args.out_root)
    template = recon_template_params(method, args.out_root)
    # L'indice e' quello globale dell'elemento: e' il seme del sottocampionamento, e deve
    # restare lo stesso al variare del numero di worker e della dimensione del blocco.
    tasks = [[(start + k, it.name, it.gt_name, spans[it.name])
              for k, it in enumerate(items[start:start + args.chunk])]
             for start in range(0, len(items), args.chunk)]

    t0 = time.time()
    _STATE.clear()
    _STATE.update(method=method, out_root=args.out_root,
                  lmk_gt=gt_face_landmarks(args.out_root),
                  radius_eye_spans=template["radius_eye_spans"])
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers, initializer=_init_worker,
                  initargs=(args.icp_points, args.icp_iter, args.sample_points)) as pool:
        results = [r for block in pool.map(_gt_chunk, tasks) for r in block]
    seconds = time.time() - t0

    rows = [{**{k: getattr(it, k) for k in common.ITEM_FIELDS}, **res}
            for it, res in zip(items, results)]
    measures = ("now_median", "now_mean", "sim_icp_p2s_median", "sim_icp_p2s_mean",
                "chamfer_raw", "chamfer_icp_mm", "chamfer_gtclip_mm",
                "recon_inside_gt_patch", "scale_px_per_mm", "eye_span_mm", "crop_radius_mm",
                "gt_face_area_mm2", "recon_face_area_mm2", "area_ratio")
    if args.max_items > 0:
        print(f"[ws3b-geom] {method}: CONTROLLO su {len(rows)} elementi, csv non scritto",
              flush=True)
    else:
        common.write_rows(path, common.ITEM_FIELDS + measures, rows, method=method,
                          seconds=seconds, icp_points=args.icp_points, icp_iter=args.icp_iter,
                          sample_points=args.sample_points, variant="facebench",
                          ex_ex_mm=template["ex_ex_mm"],
                          radius_eye_spans=template["radius_eye_spans"],
                          units="mm per now_*/sim_icp_*, adimensionale per chamfer_raw")
    med = {k: float(np.median([r[k] for r in rows])) for k in measures}
    print(f"[ws3b-geom] {method} gt: {len(rows)} ricostruzioni in {seconds:.0f}s -> {path.name}\n"
          f"    mediane: now_median {med['now_median']:.2f} mm, "
          f"sim_icp_p2s_median {med['sim_icp_p2s_median']:.2f} mm, "
          f"chamfer_raw {med['chamfer_raw']:.4f}, "
          f"chamfer_icp_mm {med['chamfer_icp_mm']:.3f} mm, "
          f"chamfer_gtclip_mm {med['chamfer_gtclip_mm']:.3f} mm "
          f"({med['recon_inside_gt_patch']:.1%} dei vertici dentro la patch GT), "
          f"scala {med['scale_px_per_mm']:.2f} px/mm\n"
          f"    supporto: ex-ex {med['eye_span_mm']:.2f} mm (GT {template['ex_ex_mm']:.2f}), "
          f"raggio ritaglio {med['crop_radius_mm']:.2f} mm (GT {common.FACE_RADIUS_MM:.1f}), "
          f"area recon/GT {med['area_ratio']:.4f} "
          f"({med['recon_face_area_mm2']:.0f}/{med['gt_face_area_mm2']:.0f} mm^2)", flush=True)


def run_pairs(method: str, records, args) -> None:
    path = common.pair_csv_path("chamfer_raw", method, args.out_root)
    if path.exists() and not args.overwrite:
        print(f"[ws3b-geom] {method}: {path.name} c'e' gia', salto", flush=True)
        return
    names = sorted({r.name_a for r in records} | {r.name_b for r in records})
    t0 = time.time()
    verts = {name: maxabs_normalize(load_npz(common.recon_face_dir(method, args.out_root)
                                             / f"{name}.npz")[0]) for name in names}
    mb = sum(v.nbytes for v in verts.values()) / 1024.0 ** 2
    print(f"[ws3b-geom] {method} pairs: {len(names)} mesh caricate in {time.time() - t0:.0f}s "
          f"(~{mb:.0f} MB, condivise coi worker via fork)", flush=True)

    tasks = [[(r.pair_index, r.name_a, r.name_b) for r in records[start:start + args.chunk]]
             for start in range(0, len(records), args.chunk)]
    t0 = time.time()
    _STATE.clear()
    _STATE.update(verts=verts)
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers, initializer=_init_worker,
                  initargs=(args.icp_points, args.icp_iter, args.sample_points)) as pool:
        values = [v for block in pool.map(_pairs_chunk, tasks) for v in block]
    seconds = time.time() - t0

    rows = [{**{k: getattr(r, k) for k in common.PAIR_FIELDS if k != "distance"},
             "distance": value} for r, value in zip(records, values)]
    common.write_rows(path, common.PAIR_FIELDS, rows, metric="chamfer_raw", method=method,
                      seconds=seconds, sample_points=args.sample_points, variant="facebench")
    print(f"[ws3b-geom] {method} pairs: {len(rows)} coppie in {seconds:.0f}s -> {path.name}",
          flush=True)


def main() -> None:
    args = parse_args()
    args.out_root = args.out_root.resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    items = common.load_manifest(common.manifest_path(args.out_root))
    records = common.load_pairs(path=common.protocol_path(args.out_root)) \
        if args.mode in ("all", "pairs") else []
    print(f"[ws3b-geom] {len(items)} ricostruzioni x {len(methods)} metodi, "
          f"{len(records)} coppie, workers={args.workers}", flush=True)

    for method in methods:
        present = {p.stem for p in common.recon_face_dir(method, args.out_root).glob("*.npz")}
        mine = [it for it in items if it.name in present]
        if len(mine) != len(items):
            print(f"[ws3b-geom] ATTENZIONE {method}: {len(mine)}/{len(items)} mesh ritagliate",
                  flush=True)
        if args.max_items > 0:
            mine = mine[: args.max_items]
        if args.mode in ("all", "gt"):
            run_gt(method, mine, args)
        if args.mode in ("all", "pairs"):
            keep = [r for r in records if r.name_a in present and r.name_b in present]
            if len(keep) != len(records):
                print(f"[ws3b-geom] {method}: {len(keep)}/{len(records)} coppie complete",
                      flush=True)
            run_pairs(method, keep, args)


if __name__ == "__main__":
    main()
