#!/usr/bin/env python3
"""I 7 landmark del protocollo NoW sulla topologia tracciata di Multiface (WS3b).

    aau/run.sh aau/recon/ws3b_landmarks.py --per-method 60

Il protocollo NoW stima la similarita' fra predizione e scansione da SETTE punti:
exocanthion e endocanthion destro e sinistro, pronasale, cheilion destro e sinistro.  Sui
tre metodi ci sono gia': sono sette dei 68 landmark iBUG che ogni runner scrive nel json.
Sulla GT no, perche' Multiface non distribuisce landmark.

Come si trovano sulla GT
------------------------
La topologia tracciata e' in corrispondenza densa e le mesh vivono tutte nello stesso frame
testa, quindi **basta un indice di vertice per punto**, valido per tutti i soggetti e tutti
i frame.  Gli indici si trasferiscono dai metodi:

1. per un campione di ricostruzioni si stima la similarita' GT -> ricostruzione con lo
   stesso ICP di ``ws3b_geometric.py`` (``fg.rigid_icp_align``, 30 iterazioni), la si
   rilegge con ``procrustes_transform`` e la si INVERTE, cosi' i 68 landmark del metodo
   finiscono nel frame testa della GT, in millimetri;
2. dei 7 punti si prende la **mediana delle posizioni** su tutto il campione, e solo dopo si
   cerca il vertice piu' vicino sulla mesh MEDIA delle 2848 tracciate.  Mediare prima e
   agganciare dopo, e non viceversa: il singolo allineamento sbaglia di 3-4 mm, e il vertice
   piu' vicino a un punto sbagliato di 3 mm su una mesh fitta e' un vertice qualunque
   (misurato: prendendo la moda dei vertici per frame, i tre metodi votavano vertici diversi
   nel 5-36% dei casi e l'ex-ex che ne usciva era 95.6 mm, ~10% sopra sia Farkas sia la
   stima indipendente dall'ICP);
3. il **pronasale non si trasferisce**: e' il vertice di z massima sulla mesh media, cioe' lo
   stesso indice che ``ws3b_prepare_meshes.py`` usa per centrare il ritaglio.  Un massimo
   geometrico e' piu' affidabile di qualunque trasferimento.

L'ICP entra solo qui, una volta sola, per battezzare sette vertici.  Il criterio NoW che ne
esce non lo usa: la sua similarita' viene dai 7 punti e basta.

Controlli che lo script stampa e salva
--------------------------------------
* dispersione fra campioni della posizione trasferita, e distanza fra la posizione mediana e
  il vertice scelto: se il primo numero e' grande il landmark non e' identificato;
* lo scarto fra i tre metodi, calcolato sulle loro mediane separate;
* le distanze antropometriche sulle 2848 mesh tracciate -- ex-ex, en-en, ch-ch -- da
  confrontare con i valori adulti di Farkas (ex-ex ~87-92 mm, en-en ~31-34 mm, ch-ch
  ~50-55 mm).  Se cadono fuori, i vertici sono sbagliati e si vede subito;
* ``ex_ex_icp_mm``, la stessa distanza stimata SENZA landmark sulla GT: la scala dell'ICP
  porta l'ex-ex della ricostruzione in millimetri veri.  Misurato 86.10 mm contro i
  91.72 mm dai landmark, cioe' 6.5% di scarto -- i landmark trasferiti sono un po' troppo
  laterali.  E' ``ex_ex_icp_mm`` che ``ws3b_prepare_meshes.py`` usa per convertire i 95 mm
  del ritaglio in aperture oculari, perche' quel che deve tornare e' il raggio in
  millimetri VERI, e la scala dell'ICP e' la sola grandezza che lega i pixel del metodo ai
  millimetri della GT (vedi ``gt_outer_canthal_mm``).

Esito, e perche' il criterio NoW resta fuori scope
--------------------------------------------------
Misurato con ``--per-method 40`` (120 campioni): dispersione fra campioni 4.6-6.9 mm,
aggancio al vertice 1.7-4.4 mm, scarto fra le mediane dei tre metodi 6.8-10.6 mm.
L'antropometria che ne esce e' plausibile ma tirata (ex-ex 91.7, en-en 37.0, ch-ch 53.0 mm
contro Farkas 87-92 / 31-34 / 50-55).  Con un errore di annotazione di 5-7 mm contro un
errore di ricostruzione di 1.2-1.6 mm, la similarita' da 7 punti misura l'annotazione: su
20 elementi ``now_median`` viene 8.77 mm per 3DDFA_V2 e 2.4-2.6 mm per gli altri due.  La
colonna resta nei csv come diagnostico, fuori dalla classifica.
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
import ws3b_geometric as geom  # noqa: E402
import ws3b_prepare_meshes as prep  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT / "faceBench" / "latentVSpipeline"))

_STATE: dict = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--methods", type=str, default=",".join(common.METHODS))
    p.add_argument("--per-method", type=int, default=60,
                   help="ricostruzioni campionate per metodo, sparse su tutti i soggetti")
    p.add_argument("--icp-points", type=int, default=4096)
    p.add_argument("--icp-iter", type=int, default=30)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _transport_one(task):
    """I 7 landmark del metodo portati nel frame testa della GT, piu' l'ex-ex in mm."""
    import fg_metrics as fg

    index, method, name, gt_name = task
    out_root = _STATE["out_root"]
    Vg, _ = geom.load_npz(common.gt_face_dir(out_root) / f"{gt_name}.npz")
    Vr, _, lmk = prep.load_recon(method, name, out_root)

    # Stessa similarita' di ws3b_geometric: GT -> ricostruzione, poi la sua inversa.
    Y = fg.sample_vertices(Vr, _STATE["icp_points"], seed=index + 1)
    Vg_aligned = fg.rigid_icp_align(Vg, Y, max_points=_STATE["icp_points"],
                                    max_iter=_STATE["icp_iter"], seed=index)
    scale, R, trans = fg.procrustes_transform(Vg, Vg_aligned, scaling=True)
    lmk_mm = (lmk - trans) @ R.T / max(scale, 1e-12)
    eye_span_mm = float(np.linalg.norm(lmk_mm[common.LMK_EYE_OUTER[0]]
                                       - lmk_mm[common.LMK_EYE_OUTER[1]]))
    return lmk_mm[list(common.LMK7_IBUG)], eye_span_mm


def sample_items(items, method: str, out_root: Path, n: int) -> list:
    """Campione sparso su tutti i soggetti, deterministico (passo costante sull'ordine)."""
    present = [it for it in items
               if (common.recon_dir(method, out_root) / f"{it.name}.npz").is_file()]
    if not present:
        raise SystemExit(f"nessuna ricostruzione per {method} sotto {out_root}")
    step = max(1, len(present) // max(n, 1))
    return present[::step][:n]


def mean_tracked_mesh() -> np.ndarray:
    """Mesh media delle tracciate, come in ``ws3b_prepare_meshes.build_gt_template``."""
    names = sorted(p.stem for p in (common.MF_PREP / "tracked").glob("*.npz"))
    if not names:
        raise SystemExit(f"nessuna mesh tracciata in {common.MF_PREP / 'tracked'}")
    acc = None
    for name in names:
        with np.load(common.gt_mesh_path(name)) as z:
            V = np.asarray(z["V"], dtype=np.float64)
        acc = V.copy() if acc is None else acc + V
    return acc / len(names), names


def anthropometry(indices: dict[str, int], names: list[str]) -> dict:
    """ex-ex, en-en e ch-ch su tutte le mesh tracciate, in millimetri."""
    pairs = {"ex_ex": ("ex_r", "ex_l"), "en_en": ("en_r", "en_l"), "ch_ch": ("ch_r", "ch_l")}
    values = {k: [] for k in pairs}
    for name in names:
        with np.load(common.gt_mesh_path(name)) as z:
            V = np.asarray(z["V"], dtype=np.float64)
        for key, (a, b) in pairs.items():
            values[key].append(float(np.linalg.norm(V[indices[a]] - V[indices[b]])))
    return {key: {"median_mm": float(np.median(v)), "p5_mm": float(np.percentile(v, 5)),
                  "p95_mm": float(np.percentile(v, 95)), "n": len(v)}
            for key, v in values.items()}


def main() -> None:
    args = parse_args()
    args.out_root = args.out_root.resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    path = common.landmarks_path(args.out_root)
    if path.exists() and not args.overwrite:
        print(f"[ws3b-lmk] {path.name} c'e' gia', salto (usa --overwrite)")
        return

    items = common.load_manifest(common.manifest_path(args.out_root))
    tasks = []
    for method in methods:
        for k, it in enumerate(sample_items(items, method, args.out_root, args.per_method)):
            tasks.append((k, method, it.name, it.gt_name))
    print(f"[ws3b-lmk] {len(tasks)} ricostruzioni campionate ({len(methods)} metodi), "
          f"workers={args.workers}", flush=True)

    t0 = time.time()
    _STATE.clear()
    _STATE.update(out_root=args.out_root, icp_points=args.icp_points, icp_iter=args.icp_iter)
    with mp.get_context("fork").Pool(processes=args.workers) as pool:
        results = pool.map(_transport_one, tasks)
    print(f"[ws3b-lmk] landmark trasportati in {time.time() - t0:.0f}s", flush=True)

    coords = np.stack([r[0] for r in results])                    # (n_campioni, 7, 3)
    spans = np.asarray([r[1] for r in results], dtype=np.float64)
    method_of = np.asarray([t[1] for t in tasks])

    V_mean, names = mean_tracked_mesh()
    nose = int(np.argmax(V_mean[:, 2]))  # +z = in avanti nel frame testa di Multiface
    target = np.median(coords, axis=0)
    target[list(common.LMK7_NAMES).index("prn")] = V_mean[nose]

    from scipy.spatial import cKDTree

    tree = cKDTree(V_mean)
    snap_dist, snap_idx = tree.query(target, k=1)

    indices, report = {}, {}
    for k, name in enumerate(common.LMK7_NAMES):
        indices[name] = int(snap_idx[k])
        per_method = {m: np.median(coords[method_of == m, k, :], axis=0) for m in methods}
        pairwise = [float(np.linalg.norm(per_method[a] - per_method[b]))
                    for i, a in enumerate(methods) for b in methods[i + 1:]]
        scatter = float(np.median(np.linalg.norm(coords[:, k, :] - target[k], axis=1)))
        report[name] = {
            "vertex": int(snap_idx[k]),
            "from": "argmax z sulla mesh media" if name == "prn" else "mediana dei trasferimenti",
            "sample_scatter_median_mm": scatter,
            "snap_to_vertex_mm": float(snap_dist[k]),
            "method_disagreement_max_mm": float(max(pairwise)) if pairwise else 0.0,
        }
        print(f"[ws3b-lmk] {name:5s} vertice {snap_idx[k]:6d} ({report[name]['from']}), "
              f"dispersione fra campioni {scatter:.2f} mm, aggancio al vertice "
              f"{snap_dist[k]:.2f} mm, scarto fra metodi {report[name]['method_disagreement_max_mm']:.2f} mm",
              flush=True)

    anthro = anthropometry(indices, names)
    for key, row in anthro.items():
        print(f"[ws3b-lmk] {key}: mediana {row['median_mm']:.2f} mm "
              f"[p5 {row['p5_mm']:.2f}, p95 {row['p95_mm']:.2f}] su {row['n']} mesh", flush=True)
    ex_ex_icp = float(np.median(spans))
    ex_ex_icp_by_method = {m: float(np.median(spans[method_of == m])) for m in methods}
    print(f"[ws3b-lmk] ex-ex dai landmark {anthro['ex_ex']['median_mm']:.2f} mm contro "
          f"{ex_ex_icp:.2f} mm dalla scala dell'ICP (stima indipendente): scarto "
          f"{abs(anthro['ex_ex']['median_mm'] - ex_ex_icp) / ex_ex_icp:.1%}", flush=True)
    # Per metodo i tre valori stanno entro l'1% del pooled, quindi il raggio del ritaglio
    # usa il pooled: distinguerli sposterebbe il raggio di meno dello 0.2%.
    print(f"[ws3b-lmk] ex-ex dall'ICP per metodo: "
          + ", ".join(f"{m} {v:.2f}" for m, v in ex_ex_icp_by_method.items()), flush=True)

    payload = {
        "source": "mediana dei 68 landmark iBUG dei tre metodi trasportati con l'ICP inverso; "
                  "pronasale dal massimo di z sulla mesh media",
        "n_samples": len(tasks),
        "per_method": args.per_method,
        "names": list(common.LMK7_NAMES),
        "ibug_indices": list(common.LMK7_IBUG),
        "vertex_indices": indices,
        "nose_tip_vertex": nose,
        "quality": report,
        "anthropometry_mm": anthro,
        "ex_ex_mm": anthro["ex_ex"]["median_mm"],
        "ex_ex_icp_mm": ex_ex_icp,
        "ex_ex_icp_mm_by_method": ex_ex_icp_by_method,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ws3b-lmk] -> {path}")


if __name__ == "__main__":
    main()
