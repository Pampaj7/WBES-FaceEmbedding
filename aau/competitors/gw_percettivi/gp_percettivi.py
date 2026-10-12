#!/usr/bin/env python3
"""Feature percettive su render (DINOv2, CLIP, LPIPS, DreamSim): FR col render in mm, SR a centroid size.

    gp_run.sh aau/competitors/gw_percettivi/gp_percettivi.py warm                            (pesi in $GP_WEIGHTS)
    gp_run.sh aau/competitors/gw_percettivi/gp_percettivi.py posecheck --view hifi3d --workers 32
    GP_THREADS=32 gp_run.sh aau/competitors/gw_percettivi/gp_percettivi.py run --view hifi3d --workers 32
    (gp.sbatch; un dominio nuovo: ``--view <nome> --spec <json>``, vedi gp_common.py)

Render (``render``): il renderer del repo (``v2_work/phase0/render_mesh.py``: numpy + PIL, ortografico, luce fissa,
ombreggiatura di Lambert uniforme a due facce, nessuna texture, sfondo nero; lo stesso delle baseline percettive di
``aau/baselines/perceptual_matrix.py``), 512 x 512, viste a yaw 0 e +-30 gradi (``common.VIEW_YAWS``) attorno al
baricentro. La mesh entra in mm nel frame canonico (``gp_common.mm_mesh``: UNA trasformazione per dominio, centrata
sul baricentro pesato per area, nessuna rigida per mesh e nessuna GT), portata nella convenzione del renderer con
(x, -y, -z). Camera UNICA per dominio: semilato del quadro H = ``CAM_K`` x L_d mm (``blmm.params``), cioe' una scala in
pixel per mm fissa. Modo ``fr``: la mesh in mm come e' (la taglia resta visibile nel render); modo ``sr``: la mesh
x CS_ref / CS_i (centroid size robusta) con la stessa camera. Senza la camera metrica queste feature sarebbero cieche
alla taglia per costruzione (``perceptual_matrix.py`` normalizza maxabs ogni mesh): qui lo sono solo in ``sr``.
Le mesh che escono dal quadro si contano (``clipped``).

Modelli (estrattori importati da ``v2_work/phase0/perceptual_embed.py``, gli stessi delle baseline percettive):
DINOv2 ViT-S/14 (token CLS, 224), CLIP ViT-B/32 laion2b (open_clip), DreamSim ``ensemble`` 0.2.1 (DINO ViT-B/16 + CLIP
ViT-B/16 + OpenCLIP ViT-B/32 con LoRA), LPIPS 0.1 AlexNet a 256. Distanze: DINOv2 / CLIP / DreamSim = 1 - coseno fra
gli embedding mediati sulle 3 viste e rinormalizzati L2 (la regola di ``perceptual_matrix.py``); LPIPS = media sulle 3
viste della LPIPS fra i render della stessa vista. LPIPS su tutte le coppie in forma chiusa: con a = sqrt(w) f / sqrt(HW)
(f feature normalizzate, w pesi lineari) LPIPS(x, y) = sum_l ||a_l(x) - a_l(y)||^2, quindi dalle Gram; controllo
contro ``lpips.LPIPS.forward`` su ``N_LPIPS_CHECK`` coppie. Nessun iperparametro scelto sui dati: viste, risoluzione,
modelli e regole sono quelli delle baseline percettive gia' nel repo; ``CAM_K`` e' fissato prima dei numeri e
controllato solo sul taglio del quadro.

``posecheck`` (la posa e' gia' canonica?): per ogni mesh della vista, ICP rigido (punto-punto, 30 giri, 1024 campioni
FPS) sulla original del primo soggetto NON valutato (``blmm.template_subjects``; FaMoS: la prima scansione di
galleria) e l'angolo della rotazione trovata; piu' la punta del naso (vertice con z massima) rispetto al centro del
quadro. Nessuna GT. Uscita ``percettivi/posecheck_<vista>.json`` e un png di controllo (fuori da git).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

import gp_common as gc

PERC = gc.OUT / "percettivi"
YAWS = (0.0, -30.0, 30.0)                 # aau/baselines/common.VIEW_YAWS
SIZE = 512
LPIPS_SIZE = 256                          # perceptual_matrix.py --lpips-size
CAM_K = 1.3                               # semilato del quadro = CAM_K x L_d (mm)
FLIP = np.array([1.0, -1.0, -1.0])        # frame canonico (+y alto, volto verso +z) -> AXES di render_mesh
EMBED = ("dinov2", "clip", "dreamsim")
N_LPIPS_CHECK = 20
BATCH = 32
_C: dict = {}


# ------------------------------------------------------------------------------------------- render

def render(Vc: np.ndarray, F: np.ndarray, yaw: float, H: float) -> np.ndarray:
    """(SIZE, SIZE) uint8 della mesh centrata ``Vc`` (mm, frame canonico) a ``yaw``, quadro di semilato H mm."""
    from render_mesh import FACE_SPAN, _yaw_rotate, render_mesh
    V = Vc * FLIP
    if yaw:
        V = _yaw_rotate(V, yaw, np.zeros(3))
    return render_mesh(V, F, size=SIZE, scale=2.0 * FACE_SPAN * H, center=np.zeros(3))[:, :, 0]


def _render(task):
    """Render fr e sr delle 3 viste di una mesh: (immagini (2, 3, S, S), taglia, fonte, taglio, secondi)."""
    from render_mesh import _yaw_rotate
    k, path = task
    v, H = _C["view"], _C["H"]
    out = {"k": k, "fail": "", "img": None, "cs": np.nan, "src": "", "clip": np.zeros((2, len(YAWS)), bool),
           "t": np.nan}
    try:
        t0 = time.perf_counter()
        V, F, _ = gc.mm_mesh(v, path)
        cs, src = gc.size_of(v, path, V, F)
        img = np.zeros((2, len(YAWS), SIZE, SIZE), np.uint8)
        for m, X in enumerate((V, V * (v["cs_ref"] / cs))):
            for y, yaw in enumerate(YAWS):
                img[m, y] = render(X, F, yaw, H)
                R = _yaw_rotate(X * FLIP, yaw, np.zeros(3)) if yaw else X * FLIP
                out["clip"][m, y] = bool(np.abs(R[:, :2]).max() > H)
        out.update(img=img, cs=cs, src=src, t=time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001  (la mesh resta NaN, contata)
        out["fail"] = f"{type(exc).__name__}: {exc}"
    return out


# ------------------------------------------------------------------------------------------ modelli

def extractor(name: str, device: str):
    """Funzione lista di immagini (S, S) uint8 -> embedding (n, d) L2-normalizzati."""
    import torch
    from PIL import Image
    if name in ("dinov2", "clip"):
        import perceptual_embed as pe
        ex = (pe.Dinov2Extractor if name == "dinov2" else pe.ClipExtractor)(device=device)
        return lambda imgs: np.stack([ex(np.repeat(g[:, :, None], 3, axis=2)) for g in imgs])
    from dreamsim import dreamsim
    model, preprocess = dreamsim(pretrained=True, device=device, cache_dir=os.environ["GP_DREAMSIM_CACHE"])
    model.eval()

    def run(imgs):
        out = []
        for s in range(0, len(imgs), BATCH):
            x = torch.cat([preprocess(Image.fromarray(g).convert("RGB")) for g in imgs[s:s + BATCH]]).to(device)
            with torch.no_grad():
                e = model.embed(x).float().cpu().numpy()
            out.append(e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9))
        return np.concatenate(out)
    return run


def lpips_model(device: str):
    import lpips
    return lpips.LPIPS(net="alex", version="0.1", verbose=False).to(device).eval()


def lpips_tensor(imgs: np.ndarray, device: str):
    """(n, S, S) uint8 -> (n, 3, LPIPS_SIZE, LPIPS_SIZE) in [-1, 1] (ridimensionamento LANCZOS di PIL)."""
    import torch
    from PIL import Image
    x = np.stack([np.asarray(Image.fromarray(g).resize((LPIPS_SIZE, LPIPS_SIZE), Image.LANCZOS)) for g in imgs])
    t = torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)[:, None].repeat(1, 3, 1, 1)
    return t.to(device)


def lpips_matrix(model, imgs: np.ndarray, device: str) -> np.ndarray:
    """LPIPS (n, n) fra tutte le immagini ``imgs`` (n, S, S) dalle Gram delle feature pesate (forma chiusa)."""
    import lpips
    import torch
    n = len(imgs)
    G = None
    with torch.no_grad():
        feats = []
        for s in range(0, n, BATCH):
            f = model.net.forward(model.scaling_layer(lpips_tensor(imgs[s:s + BATCH], device)))
            feats.append([lpips.normalize_tensor(x) for x in f])
        D = torch.zeros((n, n), dtype=torch.float64, device=device)
        for k in range(len(feats[0])):
            A = torch.cat([b[k] for b in feats])                         # (n, C, h, w)
            w = model.lins[k].model[-1].weight.view(1, -1, 1, 1)
            hw = A.shape[2] * A.shape[3]
            Gk = ((A * w).flatten(1).double() @ A.flatten(1).double().T) / hw
            d = torch.diag(Gk)
            D += d[:, None] + d[None, :] - 2.0 * Gk
        G = D.clamp_min(0.0).cpu().numpy()
    G = 0.5 * (G + G.T)
    np.fill_diagonal(G, 0.0)
    return G


def lpips_check(model, imgs: np.ndarray, D: np.ndarray, device: str) -> float:
    """Massimo |Gram - forward| su ``N_LPIPS_CHECK`` coppie a caso (seme 0)."""
    import torch
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(imgs), N_LPIPS_CHECK), rng.integers(0, len(imgs), N_LPIPS_CHECK)
    with torch.no_grad():
        ref = model(lpips_tensor(imgs[i], device), lpips_tensor(imgs[j], device)).flatten().double().cpu().numpy()
    return float(np.abs(ref - D[i, j]).max())


# ------------------------------------------------------------------------------------------ passi

def warm(device: str) -> None:
    """Scarica e carica i pesi di tutti i modelli (cache in $GP_WEIGHTS) su un'immagine nera."""
    img = np.zeros((2, SIZE, SIZE), np.uint8)
    for name in EMBED:
        e = extractor(name, device)(img)
        print(f"[gp-perc] {name}: embedding {e.shape}", flush=True)
    m = lpips_model(device)
    print(f"[gp-perc] lpips: {lpips_matrix(m, img, device)[0, 1]:.3e} su due immagini uguali", flush=True)


def camera(v: dict) -> float:
    return CAM_K * v["L"]


def run_view(vname: str, spec: Path | None, workers: int, device: str) -> None:
    import torch
    outs = {m: gc.out_npz(m, vname) for m in EMBED + ("lpips",)}
    if all(p.exists() for p in outs.values()):
        print(f"[gp-perc] {vname}: uscite gia' presenti, salto", flush=True)
        return
    v = gc.view(vname, spec)
    _C.update(view=v, H=camera(v))
    n = len(v["items"])
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_render, list(enumerate(p for _, _, p in v["items"])), chunksize=1)
    t_render = time.time() - t0
    ok = np.asarray([not r["fail"] for r in res])
    fails = [f"{v['items'][r['k']][0]}|{v['items'][r['k']][1]}|{r['fail']}" for r in res if r["fail"]]
    img = np.zeros((n, 2, len(YAWS), SIZE, SIZE), np.uint8)
    for r in res:
        if not r["fail"]:
            img[r["k"]] = r["img"]
    clipped = np.stack([r["clip"] for r in res])
    cs = np.asarray([r["cs"] for r in res])
    sec_render = np.asarray([r["t"] for r in res])
    del res
    control_png(vname, img[0])
    torch.set_num_threads(int(os.environ.get("GP_THREADS", "1")))
    base = {"view": vname, "n_meshes": n, "failed": fails, "yaws": list(YAWS), "size": SIZE, "camera_half_mm": _C["H"],
            "cam_k": CAM_K, "L_mm": v["L"], "cs_ref_mm": v["cs_ref"], "device": device,
            "threads": torch.get_num_threads(), "workers_render": workers,
            "clipped_meshes": {m: int(clipped[:, k].any(1).sum()) for k, m in enumerate(gc.MODES)},
            "render_s_per_mesh_median": float(np.nanmedian(sec_render)), "render_wall_s": t_render,
            "render_note": "s per mesh = 6 render (fr, sr x 3 viste) + lettura + taglia, un processo", **gc.stamp()}
    for name in EMBED:
        if outs[name].exists():
            continue
        t1 = time.time()
        f = extractor(name, device)
        t_load = time.time() - t1
        E = {}
        t1 = time.time()
        for k, mode in enumerate(gc.MODES):
            flat = img[:, k].reshape(-1, SIZE, SIZE)
            E[mode] = f(flat).reshape(n, len(YAWS), -1)
        t_emb = time.time() - t1
        D = {}
        for mode in gc.MODES:
            Z = E[mode].astype(np.float64).mean(1)
            Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
            M = np.clip(1.0 - Z @ Z.T, 0.0, None)
            M[~ok] = M[:, ~ok] = np.nan
            np.fill_diagonal(M, 0.0)
            D[mode] = 0.5 * (M + M.T)
        info = {**base, "method": name, "load_s": t_load, "embed_wall_s": t_emb,
                "s_per_image": t_emb / (2 * n * len(YAWS)),
                "s_per_mesh_total": t_emb / n + float(np.nanmedian(sec_render)),
                "distance": "1 - coseno fra embedding mediati sulle viste e rinormalizzati"}
        gc.write_view(name, v, D, {f"E_{m}": E[m].astype(np.float32) for m in gc.MODES} | {"cs": cs}, info)
        print(f"[gp-perc] {vname} {name}: {n} mesh, {t_emb:.0f}s di embedding ({info['s_per_image'] * 1e3:.0f} ms per "
              f"immagine), taglio fr/sr {base['clipped_meshes']}", flush=True)
    if not outs["lpips"].exists():
        model = lpips_model(device)
        t1 = time.time()
        D, check = {}, {}
        for k, mode in enumerate(gc.MODES):
            per = []
            for y in range(len(YAWS)):
                M = lpips_matrix(model, img[:, k, y], device)
                check[f"{mode}|{YAWS[y]:+.0f}"] = lpips_check(model, img[:, k, y], M, device)
                per.append(M)
            M = np.mean(per, axis=0)
            M[~ok] = M[:, ~ok] = np.nan
            np.fill_diagonal(M, 0.0)
            D[mode] = M
        t_l = time.time() - t1
        info = {**base, "method": "lpips", "wall_s": t_l, "s_per_pair": t_l / (n * (n - 1) / 2),
                "check_gram_vs_forward_max_abs": check,
                "distance": "media sulle 3 viste della LPIPS 0.1 alex fra i render della stessa vista (Gram)"}
        gc.write_view("lpips", v, D, {"cs": cs}, info)
        print(f"[gp-perc] {vname} lpips: {t_l:.0f}s per tutte le coppie, controllo Gram-forward max "
              f"{max(check.values()):.2e}", flush=True)


def control_png(vname: str, img: np.ndarray) -> None:
    """Le 6 immagini della prima mesh (fr in alto, sr in basso) in ``percettivi/control_<vista>.png`` (fuori da git)."""
    from PIL import Image
    rows = [np.concatenate(list(img[m]), axis=1) for m in range(2)]
    PERC.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).resize((3 * SIZE // 2, SIZE)).save(PERC / f"control_{vname}.png")


# ----------------------------------------------------------------------------------------- posa

def _pose(task):
    from gp_gw import fps
    k, path = task
    try:
        V, F, w = gc.mm_mesh(_C["view"], path)
        X = V[fps(V, 1024, int((V ** 2).sum(1).argmin()))]
        T, tree = _C["ref"], _C["tree"]
        R, t = np.eye(3), np.zeros(3)
        for _ in range(30):
            Y = X @ R.T + t
            _, nn = tree.query(Y)
            A, B = X - X.mean(0), T[nn] - T[nn].mean(0)
            U, _, Vt = np.linalg.svd(B.T @ A)
            R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            t = T[nn].mean(0) - X.mean(0) @ R.T
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
        nose = V[int(V[:, 2].argmax())]
        h = np.ptp(V[:, 1])
        return k, ang, float(nose[0]), float((nose[1] - V[:, 1].min()) / h), ""
    except Exception as exc:  # noqa: BLE001
        return k, np.nan, np.nan, np.nan, f"{type(exc).__name__}: {exc}"


def posecheck(vname: str, spec: Path | None, workers: int) -> None:
    from scipy.spatial import cKDTree
    from gp_gw import fps
    v = gc.view(vname, spec)
    if vname == "famos" or spec is not None:
        ref_path = v["items"][0][2]
        ref_name = f"{v['items'][0][0]} (prima mesh della vista)"
    else:
        s0 = gc.blmm.template_subjects(vname)[0]
        ref_path, ref_name = gc.blmm.mesh_path(vname, s0, "original"), f"{s0} original (NON valutato)"
    V, _, _ = gc.mm_mesh(v, ref_path)
    T = V[fps(V, 4096, int((V ** 2).sum(1).argmin()))]
    _C.update(view=v, ref=T, tree=cKDTree(T))
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_pose, list(enumerate(p for _, _, p in v["items"])), chunksize=1)
    ang = np.asarray([r[1] for r in res])
    nx = np.asarray([r[2] for r in res])
    ny = np.asarray([r[3] for r in res])
    topo = np.asarray([t for _, t, _ in v["items"]])

    def q(x):
        return {"median": float(np.nanmedian(x)), "p95": float(np.nanpercentile(x, 95)), "max": float(np.nanmax(x))}

    out = {"view": vname, "reference": ref_name, "n_meshes": len(res), "failed": [r[4] for r in res if r[4]],
           "icp_angle_deg": q(ang), "icp_angle_deg_by_topology": {t: q(ang[topo == t]) for t in sorted(set(topo))},
           "nose_x_mm_abs": q(np.abs(nx)), "nose_height_fraction": q(ny),
           "definition": "angolo della rotazione dell'ICP rigido (30 giri, 1024 campioni FPS -> 4096 del riferimento) "
                         "dalla posa del frame canonico; naso = vertice con z massima (x in mm dal baricentro, "
                         "altezza come frazione dell'estensione in y dal basso)", **gc.stamp()}
    gc.atomic_json(PERC / f"posecheck_{vname}.json", out)
    print(f"[gp-perc] posa {vname}: angolo ICP mediana {out['icp_angle_deg']['median']:.2f} p95 "
          f"{out['icp_angle_deg']['p95']:.2f} max {out['icp_angle_deg']['max']:.2f} gradi; naso |x| mediana "
          f"{out['nose_x_mm_abs']['median']:.1f} mm, altezza {out['nose_height_fraction']['median']:.2f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("warm", "posecheck", "run"))
    p.add_argument("--view")
    p.add_argument("--spec", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.step == "warm":
        warm(a.device)
    elif a.step == "posecheck":
        posecheck(a.view, a.spec, a.workers)
    else:
        run_view(a.view, a.spec, a.workers, a.device)


if __name__ == "__main__":
    main()
