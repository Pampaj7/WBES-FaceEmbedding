#!/usr/bin/env python3
"""Render delle mesh Multiface in 3 viste, con UNA sola camera per tutte.

Gemello di ``aau/baselines/render_cache.py`` per il protocollo a coppie: stesso renderer
(``v2_work/phase0/render_mesh.py``), stessa camera condivisa, stessi tre yaw.  La camera
unica non e' un dettaglio: il default del renderer normalizza sul bbox della singola mesh,
e cosi' una topologia decimata verrebbe disegnata alla stessa dimensione della tracked
anche quando la testa e' diversa, mettendo nelle distanze percettive un segnale di
topologia invece che di identita'.

Ogni mesh viene normalizzata per conto suo (centro sulla media dei vertici, divisione per
il massimo scarto assoluto) PRIMA del render: vedi ``normalize_maxabs``.  La camera
condivisa resta, ma dopo la normalizzazione non porta piu' informazione, perche' tutte le
mesh stanno in [-1, 1]^3.

Rispetto al gemello REMESH cambiano due cose.

1. La scala: 2840 mesh x 3 viste per topologia (25560 png sulle tre topologie di
   ``WBES_WS3A_PAIRS=clean``, altrettanti sulle tre di ``hard``), in una sottocartella per
   topologia, perche' 25k file in una sola directory su CephFS sono scomodi da elencare
   (e ``ls`` e' l'unica diagnostica che si ha a job finito).  Le due passate convivono
   nella stessa out-root e condividono la camera: vedi ``camera_fits``.
2. L'orientamento.  Il renderer guarda lungo -z e mette in alto -y; le mesh REMESH sono
   nel frame BFM e ci cascano dentro cosi' come sono, le Multiface no: sono nel frame
   TESTA e col default si renderizza la nuca.  Il primo test lo ha reso evidente in due
   modi: i png mostravano il retro del cranio e il detector di ArcFace non e' scattato su
   nessuno dei 1926 render.  La rotazione giusta e' Rx(180), cioe' ``(x, y, z) ->
   (x, -y, -z)``, scelta guardando i quattro candidati Rx/Ry/Rz/identita': con
   l'identita' si vede la nuca, con Ry(180) la faccia arriva davanti ma capovolta, con
   Rx(180) e' frontale e dritta.  E' la stessa matrice per tutte le mesh e tutte le
   topologie, applicata attorno all'origine prima di qualunque altra cosa, quindi non
   introduce nessun segnale di identita' ne' di topologia.

  aau/submit.sh multiface/ws3a_perceptual.sbatch     # render + metriche, un job solo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ws3a_common as common  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "phase0"))

# _to_screen / _yaw_rotate / FACE_SPAN sono interni al renderer: si riusano di proposito,
# perche' la camera condivisa deve seguire esattamente la stessa convenzione di assi e di
# inquadratura di render_mesh(), non una sua copia.
from render_mesh import FACE_SPAN, _to_screen, _yaw_rotate, render_mesh  # noqa: E402

# Le stesse tre viste delle baseline REMESH: frontale e +/-30 gradi di yaw.
VIEW_YAWS = (0.0, -30.0, 30.0)

# Frame testa Multiface -> frame del renderer.  Vedi il docstring: e' una rotazione, non
# una riflessione (det = +1), quindi non specchia la faccia.
BASE_ROTATIONS = {"x180": np.diag([1.0, -1.0, -1.0]), "none": np.eye(3)}
_BASE_ROTATION = BASE_ROTATIONS["x180"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--yaws", type=str, default=",".join(str(y) for y in VIEW_YAWS))
    p.add_argument("--base-rotation", type=str, default="x180", choices=sorted(BASE_ROTATIONS),
                   help="rotazione fissa dal frame testa Multiface a quello del renderer")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--camera-sample", type=int, default=0,
                   help="0 = camera su tutte le mesh; >0 = su un sottoinsieme regolare, per i test")
    p.add_argument("--max-pairs-per-class", type=int, default=0,
                   help="0 = protocollo intero; 50 = le 200 coppie del test di velocita'")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def render_name(topology: str, name: str, yaw: float) -> str:
    return f"{name}__{topology}__yaw{yaw:+07.2f}"


def render_path(out_root: Path, topology: str, name: str, yaw: float) -> Path:
    return out_root / "renders" / topology / f"{render_name(topology, name, yaw)}.png"


def normalize_maxabs(V: np.ndarray) -> np.ndarray:
    """Centro sulla media dei vertici e divisione per il massimo scarto assoluto.

    E' la stessa normalizzazione che ``ws3a_geometric.load_shared`` applica prima della
    Chamfer, e la stessa di ``dataset_gtready``.  Serve perche' senza di essa il render
    porta dentro la POSIZIONE e la DIMENSIONE assoluta della testa nel frame tracciato, che
    su Multiface sono quasi un'etichetta di identita': un proxy di soli quattro numeri
    (centro del bbox e diagonale) separa i soggetti con AUC 0.996, e le metriche percettive
    su render non normalizzati non stavano misurando la forma.

    La regola e' la stessa per tutte le topologie e per tutte le mesh, ma NON e' vero che
    per questo non introduca nessun segnale di topologia: il divisore e' il massimo scarto
    assoluto, cioe' una statistica di un vertice solo, e cambia se la topologia cambia
    quel vertice.  Misurato su 72 mesh, rapporto fra la diagonale del bbox della mesh
    normalizzata e quella della sua ``tracked`` (mediana [5%, 95%], scarto massimo da 1):

        remesh   1.0010 [0.9909, 1.0101]   1.35%
        noisy    1.0029 [0.9916, 1.0106]   1.64%
        up       1.0140 [0.9838, 1.0212]   2.71%
        down     1.0080 [0.9858, 1.0252]   2.96%
        crop     0.8332 [0.7361, 0.9310]  27.28%

    Sulle quattro varianti che conservano la superficie la scala residua sta entro il 3%,
    che e' meno di un pixel di spostamento sul bordo di un render da 512 px: li' la frase
    regge.  Su ``crop`` no, e non e' un difetto della normalizzazione ma la conseguenza del
    ritaglio: togliendo la parte bassa del volto il massimo scarto assoluto cala, quindi la
    mesh ritagliata viene disegnata **~20% piu' grande** della sua ``tracked``.  Le celle
    ``tracked->crop``, ``remesh->crop`` e ``crop->crop`` del giro ``hard`` contengono percio'
    anche un salto di scala sistematico, e le loro AUC percettive vanno lette sapendolo.
    L'alternativa -- normalizzare il crop con il divisore della sua tracked -- non e'
    praticabile in generale, perche' in un confronto vero fra due scansioni la mesh
    "intera" corrispondente non c'e'.
    """
    Vc = V - V.mean(axis=0, keepdims=True)
    scale = float(np.max(np.abs(Vc)))
    return Vc / scale if scale > 1e-6 else Vc


def load_verts_faces(topology: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Mesh gia' portata nel frame del renderer: da qui in giu' vale la convenzione REMESH.

    La rotazione e' attorno all'ORIGINE, non attorno al centro della mesh: cosi' la camera
    condivisa si calcola sulle mesh gia' ruotate e non c'e' nessun ordine da rispettare.
    La normalizzazione arriva dopo, e siccome e' per mesh la camera condivisa smette di
    essere un veicolo di informazione: tutte le mesh finiscono in [-1, 1]^3.
    """
    with np.load(common.mesh_path(topology, name), allow_pickle=False) as data:
        V = np.asarray(data["V"], np.float64) @ _BASE_ROTATION.T
        return normalize_maxabs(V), np.asarray(data["F"], np.int64)


def _required_extent(V: np.ndarray, center: np.ndarray, yaw: float) -> float:
    """Estensione minima che render_mesh deve ricevere per non tagliare la mesh."""
    S = _to_screen(_yaw_rotate(V, yaw, center) if yaw else np.asarray(V, dtype=np.float64))
    centre_screen = _to_screen(np.asarray(center, dtype=np.float64)[None])[0]
    return float(2.0 * FACE_SPAN * np.abs(S[:, :2] - centre_screen[:2]).max())


def compute_camera(meshes, yaws, margin: float = 1.02) -> dict:
    """Camera unica: centro medio dei bbox, scala che inquadra tutto in ogni vista."""
    centres = []
    for topology, name in meshes:
        V, _ = load_verts_faces(topology, name)
        centres.append((V.min(axis=0) + V.max(axis=0)) / 2.0)
    center = np.mean(np.stack(centres), axis=0)

    extent = 0.0
    for topology, name in meshes:
        V, _ = load_verts_faces(topology, name)
        for yaw in yaws:
            extent = max(extent, _required_extent(V, center, yaw))
    return {"center": center.tolist(), "scale": float(extent * margin),
            "yaws": [float(y) for y in yaws], "n_meshes": len(meshes)}


def camera_fits(meshes, camera, yaws) -> tuple[float, str]:
    """(estensione massima richiesta, mesh che la richiede) con questa camera.

    Serve quando si aggiungono topologie a una out-root gia' renderizzata: ricalcolare la
    camera cambierebbe tutti i png gia' fatti (e gli embedding in cache, che sono
    indicizzati per nome del render e non ne accorgerebbero).  Se la camera vecchia
    inquadra anche le mesh nuove non c'e' niente da rifare, ed e' il caso di `crop`,
    `noisy` e `up`, che stanno tutte dentro il bbox della loro `tracked` a meno del
    rumore.  Costa una lettura per mesh, meno delle due di `compute_camera`.
    """
    center = np.asarray(camera["center"], dtype=np.float64)
    worst, worst_name = 0.0, ""
    for topology, name in meshes:
        V, _ = load_verts_faces(topology, name)
        for yaw in yaws:
            need = _required_extent(V, center, yaw)
            if need > worst:
                worst, worst_name = need, f"{topology}/{name}"
    return worst, worst_name


def _render_one(task) -> str:
    topology, name, yaw, center, scale, size, out_path = task
    from PIL import Image

    V, F = load_verts_faces(topology, name)
    center = np.asarray(center, dtype=np.float64)
    if yaw:
        V = _yaw_rotate(V, yaw, center)
    Image.fromarray(render_mesh(V, F, size=size, scale=scale, center=center)).save(out_path)
    return out_path


def main() -> None:
    import multiprocessing as mp

    args = parse_args()
    global _BASE_ROTATION
    _BASE_ROTATION = BASE_ROTATIONS[args.base_rotation]
    yaws = [float(y) for y in args.yaws.split(",") if y.strip()]
    records = common.load_pairs(max_pairs_per_class=args.max_pairs_per_class)
    topologies = common.topology_names(records)
    meshes = [(topology, name) for topology, names in topologies.items() for name in names]

    out_dir = args.out_root / "renders"
    for topology in topologies:
        (out_dir / topology).mkdir(parents=True, exist_ok=True)
    camera_path = out_dir / "camera.json"

    t0 = time.time()
    if camera_path.exists() and not args.overwrite:
        camera = json.loads(camera_path.read_text())
        if camera.get("yaws") != yaws or camera.get("base_rotation") != args.base_rotation:
            raise SystemExit(
                f"{camera_path}: yaw {camera.get('yaws')}, base_rotation "
                f"{camera.get('base_rotation')}; ora servono {yaws}, {args.base_rotation}: "
                "rilancia con --overwrite o cambia --out-root"
            )
        # Una camera calcolata su meno mesh di quelle che si stanno per disegnare puo'
        # tagliarne qualcuna: succede se si riusa la out-root di un test troncato, o se si
        # aggiungono topologie.  Invece di contare le mesh si verifica quello che conta,
        # cioe' che inquadri davvero anche le nuove: cosi' i png e gli embedding gia'
        # fatti restano validi.
        if int(camera.get("n_meshes", 0)) < len(meshes):
            t_check = time.time()
            worst, worst_name = camera_fits(meshes, camera, yaws)
            if worst > camera["scale"]:
                raise SystemExit(
                    f"{camera_path}: calcolata su {camera.get('n_meshes')} mesh e scala "
                    f"{camera['scale']:.4f}, ma {worst_name} ne chiede {worst:.4f} fra le "
                    f"{len(meshes)} di adesso: rilancia con --overwrite (rifa' TUTTI i "
                    "render) o cambia --out-root"
                )
            print(f"[ws3a-render] camera dalla cache ({camera.get('n_meshes')} mesh) "
                  f"verificata su {len(meshes)} mesh in {time.time() - t_check:.0f}s: "
                  f"serve {worst:.4f} <= {camera['scale']:.4f}", flush=True)
        print(f"[ws3a-render] camera dalla cache: {camera_path}", flush=True)
    else:
        # La camera e' un max su tutte le mesh: con --camera-sample si accetta un margine
        # meno stretto per non pagare 8520 letture in un test da 200 coppie.
        picked = meshes if args.camera_sample <= 0 else meshes[:: max(1, len(meshes) // args.camera_sample)]
        camera = compute_camera(picked, yaws)
        camera["base_rotation"] = args.base_rotation
        camera_path.write_text(json.dumps(camera, indent=2), encoding="utf-8")
        print(f"[ws3a-render] camera calcolata su {len(picked)}/{len(meshes)} mesh in "
              f"{time.time() - t0:.0f}s", flush=True)
    print(f"[ws3a-render] center={np.round(camera['center'], 4).tolist()} "
          f"scale={camera['scale']:.4f} yaws={yaws} base_rotation={args.base_rotation} "
          f"size={args.size}", flush=True)

    tasks = []
    for topology, name in meshes:
        for yaw in yaws:
            out_path = render_path(args.out_root, topology, name, yaw)
            if out_path.exists() and not args.overwrite:
                continue
            tasks.append((topology, name, yaw, camera["center"], camera["scale"],
                          args.size, str(out_path)))
    total = len(meshes) * len(yaws)
    print(f"[ws3a-render] {len(tasks)}/{total} render da fare con {args.workers} worker", flush=True)
    if not tasks:
        return

    t1 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers) as pool:
        for done, _ in enumerate(pool.imap_unordered(_render_one, tasks, chunksize=4), start=1):
            if done % 2000 == 0 or done == len(tasks):
                rate = done / max(time.time() - t1, 1e-9)
                print(f"[ws3a-render] {done}/{len(tasks)} ({rate:.1f}/s)", flush=True)
    print(f"[ws3a-render] fine in {time.time() - t1:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
