#!/usr/bin/env python3
"""Studio umano v2, passo 2: render a scala ASSOLUTA, 3 viste per volto, una sola camera in mm.

    v3_work/unified_gt/run.sh aau/human_study_v2/render_v2.py      (run.sbatch, passo ``render``)

Differenze dalla v1 (``aau/baselines/render_cache.py``), che normalizzava ogni mesh con maxabs prima di una camera
fissa e cosi' mostrava volti "alti uguali":
  - **nessuna normalizzazione di scala per mesh**: ogni testa GNM va nel frame di GT-F in mm con la trasformazione
    del DOMINIO (``hs2.to_mm``: f = u R V + t), poi con la sua rigida robusta verso mu (``hs2.robust_rigid``, la
    stessa della GT F): rotazione e traslazione, mai scala. Si renderizzano i triangoli della maschera del volto
    (``hs2.RENDER_GROUPS``), che contiene tutta la regione su cui si misurano le GT;
  - **una camera**: ortografica, ``px_per_mm`` unico, stesso centro e stessa finestra per ogni volto e ogni vista.
    La finestra si calcola UNA volta sull'unione di tutti i soggetti e di tutte le viste (``camera.json``), poi
    ogni render legge solo quella: nessuna inquadratura sul singolo volto, quindi una testa piu' grande occupa piu'
    pixel;
  - **3 viste**: la testa ruota attorno all'asse verticale (+y) per il punto fisso ``pivot`` (centroide pesato
    della regione unificata sulla media FLAME, uguale per tutti) di 0 (frontale), 45 (3/4) e 90 gradi (profilo,
    naso verso destra), cosi' si vede la profondita';
  - **visibilita' esatta**: ray casting (open3d/Embree, ``ugt.Surface``, la classe ``View`` di
    ``v3_work/unified_gt/render.py``) al posto del painter's algorithm, che sul profilo sbaglierebbe le occlusioni;
  - **materiale grigio uniforme**, normali per vertice interpolate (niente sfaccettature), luce direzionale fissa
    nel frame della camera + ambiente, sfondo nero; supersampling SSAA x SSAA.
Per ogni soggetto: ``renders/<soggetto>.png`` (striscia verticale: frontale, 3/4, profilo, separatori di
``SEP`` px) e ``renders/<soggetto>.jpg`` (la stessa, q90, quella che va nella pagina).

Controllo (``render_check.json``, ``size_check.csv``): per ogni volto e vista l'estensione della silhouette in
pixel contro quella della mesh proiettata in mm per ``px_per_mm``; correlazione fra altezza in pixel e altezza in
mm sui 100 volti; il volto piu' piccolo e il piu' grande (centroid size) affiancati in ``checks/size_extremes.png``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import hs2

YAWS = (0.0, 45.0, 90.0)
VIEW_NAMES = ("front", "three_quarter", "profile")
TILE = 384                      # px per vista nella striscia
SSAA = 3
SEP = 4                         # separatore fra le viste (px), grigio scuro
SEP_GRAY = 48
MARGIN = 1.04                   # margine della finestra comune
ALBEDO = 0.80                   # grigio uniforme
AMBIENT = 0.22
LIGHT = np.array([-0.35, 0.45, 1.0]) / np.linalg.norm([-0.35, 0.45, 1.0])   # frame camera: alto a sinistra, davanti
BACKGROUND = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tile", type=int, default=TILE)
    p.add_argument("--max-subjects", type=int, default=0, help="0 = tutti; >0 per una prova rapida")
    return p.parse_args()


def yaw_matrix(deg: float) -> np.ndarray:
    """Rotazione della testa attorno a +y: il naso (+z) va verso +x (destra dell'immagine) a 90 gradi."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def view_coords(X: np.ndarray, pivot: np.ndarray, yaw: float) -> np.ndarray:
    """Coordinate di vista (x destra, y alto, z verso la camera) della testa ruotata attorno al pivot."""
    return (X - pivot) @ yaw_matrix(yaw).T


def compute_camera(meshes_mm: dict, used: np.ndarray, pivot: np.ndarray, tile: int) -> dict:
    """UNA finestra per tutti: centro e semilato dall'unione dei bbox di tutti i volti (vertici renderizzati
    ``used``) in tutte le viste."""
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for X in meshes_mm.values():
        for yaw in YAWS:
            Y = view_coords(X[used], pivot, yaw)[:, :2]
            lo, hi = np.minimum(lo, Y.min(0)), np.maximum(hi, Y.max(0))
    offset = 0.5 * (lo + hi)
    half = float(np.ceil(0.5 * (hi - lo).max() * MARGIN))      # mm, intero
    return {"projection": "ortografica", "pivot_mm": pivot.tolist(), "view_offset_mm": offset.tolist(),
            "half_mm": half, "tile_px": tile, "px_per_mm": tile / (2.0 * half), "ssaa": SSAA,
            "yaws_deg": list(YAWS), "views": list(VIEW_NAMES), "light_camera_frame": LIGHT.tolist(),
            "ambient": AMBIENT, "albedo": ALBEDO, "background": BACKGROUND,
            "union_bbox_view_mm": {"lo": lo.tolist(), "hi": hi.tolist()}, "n_meshes": len(meshes_mm)}


def make_view(cam: dict, yaw: float):
    """``View`` di render.py: R = rotazione di vista, centro nel frame del dato = pivot + offset di vista."""
    from render import View
    R = yaw_matrix(yaw)
    off = np.array([*cam["view_offset_mm"], 0.0])
    return View(R, np.asarray(cam["pivot_mm"]) + off @ R, cam["half_mm"], cam["tile_px"] * cam["ssaa"])


def render_view(surf, Vn: np.ndarray, view) -> tuple[np.ndarray, np.ndarray]:
    """(immagine grigia float (res, res) a risoluzione SSAA, maschera di colpito)."""
    res = view.res
    jj, ii = np.meshgrid(np.arange(res) + 0.5, np.arange(res) + 0.5)
    z0 = view.to_view(surf.V)[:, 2].max() + 10.0 * view.px
    o, d = view.rays(jj.ravel(), ii.ravel(), z0)
    hit = surf.cast(view.from_view(o), d @ view.R)
    tri = np.maximum(hit["tri"], 0)
    n = np.einsum("nkd,nk->nd", Vn[surf.F[tri]], hit["bary"]) @ view.R.T
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n[n[:, 2] < 0] *= -1.0                                   # due facce: il verso dei triangoli non conta
    shade = ALBEDO * (AMBIENT + (1.0 - AMBIENT) * np.clip(n @ LIGHT, 0.0, 1.0))
    img = np.where(hit["hit"], shade, BACKGROUND)
    return img.reshape(res, res), hit["hit"].reshape(res, res)


def downsample(img: np.ndarray, ssaa: int) -> np.ndarray:
    """Media dei blocchi ssaa x ssaa (box filter): niente ringing attorno alla silhouette."""
    r = img.shape[0] // ssaa
    return img.reshape(r, ssaa, r, ssaa).mean(axis=(1, 3))


def strip(tiles: list[np.ndarray]) -> np.ndarray:
    """Viste impilate dall'alto (frontale, 3/4, profilo) con separatori, uint8 RGB."""
    w = tiles[0].shape[1]
    parts = []
    for k, t in enumerate(tiles):
        if k:
            parts.append(np.full((SEP, w), SEP_GRAY / 255.0))
        parts.append(t)
    g = np.round(255.0 * np.clip(np.concatenate(parts, axis=0), 0.0, 1.0)).astype(np.uint8)
    return np.repeat(g[:, :, None], 3, axis=2)


def silhouette_extent(mask: np.ndarray, ssaa: int) -> tuple[float, float]:
    """(larghezza, altezza) della silhouette in pixel della striscia finale."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if not len(rows):
        return 0.0, 0.0
    return (cols[-1] - cols[0] + 1) / ssaa, (rows[-1] - rows[0] + 1) / ssaa


def main() -> None:
    from PIL import Image
    from ugt import Surface, vertex_normals

    a = parse_args()
    t0 = time.time()
    subj = hs2.subjects()
    if a.max_subjects > 0:
        subj = subj[: a.max_subjects]
    cn, frame = hs2.canon()
    pivot = cn.W @ cn.mu                                     # punto fisso, mm, uguale per tutti
    rob = hs2.robust_rigid(cn, subj)
    meshes, faces = {}, None
    for k, s in enumerate(subj):
        V, faces = hs2.load_mesh(s)
        meshes[s] = hs2.apply_rigid(hs2.to_mm(cn, V), rob["R"][k], rob["t"][k])
    cam = compute_camera(meshes, np.unique(faces), pivot, a.tile)
    cam["frame"] = frame
    hs2.RENDER_DIR.mkdir(parents=True, exist_ok=True)
    hs2.write_json(hs2.RENDER_DIR / "camera.json", cam)
    print(f"[hs2-render] camera: semilato {cam['half_mm']:.0f} mm, {cam['px_per_mm']:.3f} px/mm, "
          f"{len(subj)} soggetti, frame {frame['source']}", flush=True)

    views = [make_view(cam, y) for y in YAWS]
    used = np.unique(faces)
    rows = []
    for k, s in enumerate(subj):
        X = meshes[s]
        surf = Surface(X, faces)          # i vertici fuori dalla maschera non hanno triangoli: non si vedono
        Vn = vertex_normals(X, faces)
        tiles = []
        for name, yaw, view in zip(VIEW_NAMES, YAWS, views):
            img, mask = render_view(surf, Vn, view)
            tiles.append(downsample(img, cam["ssaa"]))
            w_px, h_px = silhouette_extent(mask, cam["ssaa"])
            Y = view_coords(X[used], pivot, yaw)
            w_mm, h_mm = np.ptp(Y[:, 0]), np.ptp(Y[:, 1])
            rows.append({"subject": s, "view": name, "sil_w_px": w_px, "sil_h_px": h_px, "mesh_w_mm": w_mm,
                         "mesh_h_mm": h_mm, "err_w_px": w_px - w_mm * cam["px_per_mm"],
                         "err_h_px": h_px - h_mm * cam["px_per_mm"],
                         "touches_border": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any()
                                                or mask[:, -1].any())})
        rgb = strip(tiles)
        Image.fromarray(rgb).save(hs2.RENDER_DIR / f"{s}.png")
        Image.fromarray(rgb).save(hs2.RENDER_DIR / f"{s}.jpg", quality=90)
        if (k + 1) % 20 == 0 or k + 1 == len(subj):
            print(f"[hs2-render] {k + 1}/{len(subj)} ({time.time() - t0:.0f}s)", flush=True)

    sc = pd.DataFrame(rows)
    sc.to_csv(hs2.RENDER_DIR / "size_check.csv", index=False)
    cs = None
    size_csv = hs2.GT_DIR / "size.csv"
    if size_csv.exists():
        cs = pd.read_csv(size_csv).set_index("subject")["centroid_size_mm"]
    check = {"px_per_mm": cam["px_per_mm"], "any_touches_border": bool(sc["touches_border"].any()),
             "max_abs_err_px": float(np.abs(sc[["err_w_px", "err_h_px"]].to_numpy()).max()), "views": {}}
    for name in VIEW_NAMES:
        v = sc[sc["view"] == name]
        check["views"][name] = {
            "pearson_h_px_vs_h_mm": float(np.corrcoef(v["sil_h_px"], v["mesh_h_mm"])[0, 1]),
            "pearson_w_px_vs_w_mm": float(np.corrcoef(v["sil_w_px"], v["mesh_w_mm"])[0, 1]),
            "h_px_min": float(v["sil_h_px"].min()), "h_px_max": float(v["sil_h_px"].max()),
            "w_px_min": float(v["sil_w_px"].min()), "w_px_max": float(v["sil_w_px"].max()),
            "h_mm_min": float(v["mesh_h_mm"].min()), "h_mm_max": float(v["mesh_h_mm"].max())}
    if cs is not None and set(subj) <= set(cs.index):
        c = cs.loc[subj]
        lo, hi = str(c.idxmin()), str(c.idxmax())
        check["extremes"] = {"smallest": lo, "largest": hi, "cs_mm": [float(c.min()), float(c.max())],
                             "cs_ratio": float(c.max() / c.min())}
        im = [np.asarray(Image.open(hs2.RENDER_DIR / f"{s}.png")) for s in (lo, hi)]
        gap = np.full((im[0].shape[0], 8, 3), SEP_GRAY, dtype=np.uint8)
        (hs2.DATA_DIR / "checks").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate([im[0], gap, im[1]], axis=1)).save(hs2.DATA_DIR / "checks" / "size_extremes.png")
    hs2.write_json(hs2.RENDER_DIR / "render_check.json", check)
    print(f"[hs2-render] controllo: {check}", flush=True)
    print(f"[hs2-render] fatto in {time.time() - t0:.0f}s -> {hs2.RENDER_DIR}", flush=True)


if __name__ == "__main__":
    main()
