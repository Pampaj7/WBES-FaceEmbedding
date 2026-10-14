#!/usr/bin/env python3
"""Costo per confronto, misurato: il nostro encoder (embedding per mesh, poi d_F e d_P fra vettori) contro i protocolli
per coppia della matrice (ICP + Chamfer in mm, NICP per coppia in mm, NoW metrical col rilevatore, ULRSSM per coppia).
Le implementazioni sono quelle della matrice, importate: qui solo il cronometro fra i loro passi e i controlli di
riproduzione contro i valori gia' calcolati.

    v3_work/unified_gt/run.sh aau/costo_confronto/cc.py sample  --work W            (coppie e mesh del campione)
    aau/outlineB/run_o3d.sh   aau/costo_confronto/cc.py geo --metric icp|nicp --work W
    v3_work/unified_gt/run.sh aau/costo_confronto/cc.py det-render --work W
    aau/recon/run_recon.sh mica aau/costo_confronto/cc_detect.py --work W
    v3_work/unified_gt/run.sh aau/costo_confronto/cc.py now --work W
    aau/run.sh                aau/costo_confronto/cc.py ops --work W
    AAU_NV=--nv aau/run.sh    aau/costo_confronto/cc.py ours --work W               (GPU)
    v3_work/unified_gt/run.sh aau/costo_confronto/cc.py cmp --work W
    AAU_NV=--nv aau/run.sh    aau/costo_confronto/cc.py ulrssm --work W             (GPU)
    v3_work/unified_gt/run.sh aau/costo_confronto/cc.py report
    (cc.sbatch: tutti i passi in un job, un nodo L40S)

**Campione** (``sample``): vista HIFI3D della matrice (``pr_common.view``), topologie original, down8k, up60k. Coppie:
``N_PAIRS`` per tipo (original-down8k, original-up60k, down8k-up60k), estratte con ``SEED`` dal sottocampione di
``pr_noise.subsample`` (soggetti diversi, topologie diverse), cosi' che per ogni coppia esistano i valori gia'
calcolati di tutti i metodi (NoW col rilevatore e' calcolato solo li'). Mesh: quelle delle coppie; i tempi per mesh si
misurano su ``N_MESH`` mesh per topologia estratte fra queste (le altre si preparano una volta, non cronometrate).

**Ripetizioni**: un giro di riscaldamento (``rep`` 0, escluso: import, cache dei file, compilazione) e ``--reps`` giri
cronometrati; per elemento (mesh o coppia) la mediana dei giri, poi mediana e IQR fra gli elementi (``report``).
Tutto il calcolo su CPU a UN thread (``OMP_NUM_THREADS`` = 1 in cc.sbatch; onnxruntime con ``intra_op_num_threads`` 1,
torch con ``set_num_threads(1)``): i tempi CPU sono secondi-core. GPU: ``torch.cuda.synchronize`` prima di ogni stop.

**Passi per metodo** (fasi = chiavi ``phase`` delle righe; ``dev`` cpu o gpu):
  * nostro (``ops``, ``ours``, ``cmp``): per mesh lettura della mesh, operatori areanorm k_eig 128 (le righe di
    ``v2_work/potential/areanorm_operators.py``, ``compute_operators`` di DiffusionNet), scrittura e lettura dell'npz
    degli operatori (la pipeline passa dal disco), area in mm^2 della tabella di scala (``build_scale_table.total_area``
    x u^2), trasferimento e forward su GPU con gli agganci di ``eval_v3.install`` e gli argomenti di ``d3_embed.sbatch``
    (come ``zs_embed.py`` / ``conf_embed.py``), checkpoint factorized s1234; per coppia ``fact_paired.distances`` (d_F e
    d_P dai vettori [s, u]) su TUTTE le coppie di N mesh, N = ``N_CMP``, a blocchi di ``CMP_CHUNK`` coppie (ogni chiamata
    rilegge il checkpoint: costo fisso per blocco, misurato a parte con una chiamata da una coppia);
  * ICP + Chamfer in mm e NICP per coppia in mm (``geo``): ``blmm.pair_metrics`` (modo ``mm``) spezzata nei suoi passi:
    per mesh ``work_coords``; per coppia campionamento dei 4096 punti coi semi della matrice, ICP rigido
    (``blmm.icp``), NICP (``nonrigid_icp_align``), corrispondenza e distanza (p2p media, p2tri media). Un verso, come
    la matrice (X = il soggetto con indice minore);
  * NoW metrical col rilevatore (``det-render``, ``cc_detect.py``, ``now``): per mesh render frontale
    (``pr_noise.render_mesh``), png, 68 landmark di insightface 1k3d68 (``pr_noise_detect.py``), retroproiezione (le
    righe della condizione ``det`` di ``pr_noise._prep_task``), ritaglio di NoW (``pm.crop``, raggio 1.4) e superficie
    di A (``pm.Surf``, costruita una volta per mesh A come in ``pr_run._task``); per coppia i DUE versi (la distanza e'
    la loro media): Procrustes, ICP rigido di NoW (``pm.now_icp``) e distanza scan-to-mesh (``pm.scan_to_mesh``), cioe'
    ``pm.now_pair`` con ``metrical`` vero spezzata nei suoi passi, tutti i punti (``--now-opt`` 0, come la matrice);
  * ULRSSM per coppia (``ulrssm``, checkpoint ``faces_xyz`` dichiarato): per mesh ``ulr_eval._prep`` (operatori K 200
    su CPU, cache nuova a ogni giro) e ``ulr_eval.on_gpu`` (feature di DiffusionNet); per coppia ``ulr_common.pair_rms``
    (mappe nei due versi e rigide, GPU).

**Controlli** (``checks`` di ogni ``parts/<passo>.json``): valori ricalcolati contro quelli della matrice (``blmm``
mm, ``competitors_new/now_metrical/lmk_det``, ``competitors_new/ulrssm/faces_xyz``), landmark e render contro quelli
di ``pr_noise``, embedding contro quelli del collaudo L40S (``conf_embed``) e di sviluppo, NoW spezzata contro
``pm.now_pair``.

**Uscite** in ``aau/runs/evidence/costo_confronto/`` (``--out``): ``parts/<passo>.json`` (righe grezze, controlli,
ambiente), e da ``report``: ``tempi.csv`` (righe grezze di tutti i passi), ``sintesi.csv`` (mediana e IQR per metodo,
livello, fase, gruppo), ``scala.csv`` (estrapolazione a N mesh), ``results.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
REPO = AAU.parent
PROT = AAU / "competitors" / "protocolli"
ULR = AAU / "competitors" / "ulrssm"
TRAINER = REPO / "v3_work" / "trainer"
for _p in (PROT, AAU / "baselines_mm"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

VIEW = "hifi3d"
TOPOS = ("original", "down8k", "up60k")
TYPES = (("original", "down8k"), ("original", "up60k"), ("down8k", "up60k"))
N_PAIRS, N_MESH, SEED = 50, 20, 20261013
N_CMP, CMP_CHUNK = (100, 1000, 10000), 500_000
CORES_REF = 16                                      # core del job di misura (cc.sbatch): stima del tempo a muro
CKPT = REPO / ("aau/runs/evidence/trainer_v3/ablations/c3f_runs/factorized/v3_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_"
               "w128_b4_bs5_poolmeanmax_noise60_sampler-balanced_pooling-area_meanmax_inputnorm-global_areaweights-smooth_"
               "emadecay-0.999_head-factorized_seed1234__1f5d4a90/checkpoints/epoch072_ema.pth")
SCALE_TABLE = REPO / "aau/runs/evidence/trainer_v3/factorized/scale_tables/hifi3d_eval.npz"
EMB_L40S = REPO / "aau/runs/ava256_collaudo/hifi3d_full/embed/eval_view/factorized_s1234/embeddings.npz"
EMB_DEV = REPO / ("aau/runs/evidence/trainer_v3/ablations/c3f_eval/form_hifi/data_328f2bfc1a/scale_v3factorizedfulle072_"
                  "embed/zs_zeroshot/embeddings.npz")
ULR_CKPT = AAU / "runs/competitors_ulrssm/faces_xyz/final.pth"
E10 = AAU / "evidence" / "e10_gpu_ops"
# Operatori su GPU (``ours --ops-src gpu``): tolleranza fissata PRIMA della misura sugli embedding [s, u] ottenuti dagli
# operatori GPU contro quelli dagli operatori CPU, stesso nodo, stesso job, stesso forward: max |dZ| <= TOL_Z_GPU. Base:
# il pavimento del percorso CPU stesso sulla L40S (operatori CPU ricalcolati contro gli embedding del collaudo: 5,6e-4,
# job 1072554; il TF32 del forward amplifica le differenze all'ultimo bit degli operatori) e E10 (operatori ``bk`` a
# 1,4e-6 dalla CPU sugli embedding, forward su CPU); 1e-3 resta sotto 1/90 della distanza minima fra identita' di E10.
TOL_Z_GPU = 1e-3
OUT = AAU / "runs" / "evidence" / "costo_confronto"
METHODS = {"nostro": "nostro (factorized s1234: embedding per mesh, d_F e d_P fra vettori)",
           "icp": "ICP + Chamfer in mm (per coppia)", "nicp": "NICP per coppia in mm",
           "now_det": "NoW metrical col rilevatore (insightface 1k3d68 su render)",
           "ulrssm": "ULRSSM per coppia (faces_xyz)"}


# ------------------------------------------------------------------------------------------ comuni

def env_info() -> dict:
    cpu = ""
    try:
        cpu = next(x.split(":", 1)[1].strip() for x in open("/proc/cpuinfo") if x.startswith("model name"))
    except (OSError, StopIteration):
        pass
    out = {"host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID", "none"), "cpu": cpu,
           "cores_job": len(os.sched_getaffinity(0)), "python": platform.python_version(),
           "threads": {k: os.environ.get(k, "") for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
           "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    if "torch" in sys.modules:
        import torch
        out.update(torch=torch.__version__, torch_threads=torch.get_num_threads())
        if torch.cuda.is_available():
            out.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
    return out


def rec(rows: list, method: str, level: str, group: str, item: str, rep: int, dev: str, **phases) -> None:
    """Una riga per fase: (metodo, livello mesh|coppia|blocco, gruppo = topologia o tipo, elemento, giro, dispositivo)."""
    for ph, s in phases.items():
        rows.append({"method": method, "level": level, "group": group, "item": item, "rep": rep, "dev": dev,
                     "phase": ph, "s": float(s)})


def write_part(out: Path, step: str, rows: list, checks: dict, extra: dict | None = None) -> None:
    p = out / "parts" / f"{step}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"step": step, "env": env_info(), "checks": checks, **(extra or {}), "rows": rows},
                              separators=(",", ":")) + "\n")
    os.replace(tmp, p)
    print(f"[cc] {step}: {len(rows)} righe -> {p}; controlli {json.dumps(checks)[:600]}", flush=True)


def load_sample(out: Path) -> dict:
    s = json.loads((out / "parts" / "sample.json").read_text())
    s["path"] = {int(k): REPO / p for k, p in s["path"].items()}
    s["topo"] = {int(k): t for k, t in s["topo"].items()}
    s["subj"] = {int(k): x for k, x in s["subj"].items()}
    return s


def mesh_rounds(S: dict, reps: int):
    """(giro, mesh): giro 0 su tutte le mesh del campione, giri 1..reps sulle mesh cronometrate."""
    for r in range(reps + 1):
        for k in (S["meshes"] if r == 0 else S["timed"]):
            yield r, k


def item_of(S: dict, k: int) -> str:
    return f"{S['subj'][k]}/{S['topo'][k]}"


def pair_item(S: dict, a: int, b: int) -> str:
    return f"{item_of(S, a)}|{item_of(S, b)}"


def maxdiff(x, y) -> dict:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    d = np.abs(x[ok] - y[ok])
    rel = d / np.maximum(np.abs(y[ok]), 1e-12)
    return {"n": int(ok.sum()), "nan": int((~ok).sum()), "max_abs": float(d.max()) if len(d) else None,
            "max_rel": float(rel.max()) if len(d) else None, "equal": int((d == 0).sum())}


# ------------------------------------------------------------------------------------------ campione

def sample(a) -> None:
    import pr_common as pc
    import pr_noise as prn
    v = pc.view(VIEW)
    items = v["items"]
    i, j = prn.subsample(VIEW, items)
    topo = np.asarray([t for _, t, _ in items])
    rng = np.random.default_rng(SEED)
    pairs = []
    for ta, tb in TYPES:
        k = np.flatnonzero(((topo[i] == ta) & (topo[j] == tb)) | ((topo[i] == tb) & (topo[j] == ta)))
        sel = np.sort(rng.choice(k, a.n_pairs, replace=False))
        pairs += [[int(i[q]), int(j[q]), f"{ta}-{tb}"] for q in sel]
    meshes = sorted({x for p in pairs for x in p[:2]})
    timed = []
    for t in TOPOS:
        cand = [k for k in meshes if topo[k] == t]
        timed += sorted(int(x) for x in rng.choice(cand, min(a.n_mesh, len(cand)), replace=False))
    obj = {"view": VIEW, "seed": SEED, "n_pairs_per_type": a.n_pairs, "n_mesh_per_topology": a.n_mesh,
           "source": "pr_noise.subsample (sottocampione della condizione det)", "pairs": pairs, "meshes": meshes,
           "timed": timed, "subj": {k: items[k][0] for k in meshes}, "topo": {k: items[k][1] for k in meshes},
           "path": {k: str(Path(items[k][2]).relative_to(REPO)) for k in meshes}, "env": env_info()}
    p = a.out / "parts" / "sample.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=0) + "\n")
    print(f"[cc] sample: {len(pairs)} coppie, {len(meshes)} mesh, {len(timed)} cronometrate per mesh -> {p}", flush=True)


# ------------------------------------------------------------------------------- ICP e NICP (blmm)

def geo(a) -> None:
    import blmm
    import facebench as fb
    import run_facebench_remesh as rfr
    sys.path.insert(0, str(blmm.AAU_DIR / "baselines"))
    import common as bc
    S = load_sample(a.out)
    metric = {"icp": "rigid_icp_chamfer", "nicp": "nicp_p2tri"}[a.metric]
    prm = blmm.params()
    unit = blmm.unit(VIEW, "mm", prm)
    subjects = blmm.subjects(VIEW)
    s2i = {s: n for n, s in enumerate(subjects)}
    iu, ju = np.triu_indices(len(subjects), 1)
    q_of = {(int(x), int(y)): q for q, (x, y) in enumerate(zip(iu, ju))}
    rows, C = [], {}
    for r, k in mesh_rounds(S, a.reps):
        t0 = time.perf_counter()
        C[k] = blmm.work_coords(VIEW, "mm", S["path"][k], prm)
        rec(rows, a.metric, "mesh", S["topo"][k], item_of(S, k), r, "cpu", caricamento=time.perf_counter() - t0)
    got, ref, failed = [], [], set()
    for r in range(a.reps + 1):
        for pa, pb, typ in S["pairs"]:
            x, y = (pa, pb) if s2i[S["subj"][pa]] < s2i[S["subj"][pb]] else (pb, pa)
            seed = q_of[(s2i[S["subj"][x]], s2i[S["subj"][y]])]
            X, Y = C[x], C[y]
            t0 = time.perf_counter()
            Xs = rfr.sample_pts(X[0], blmm.N_POINTS, seed)
            Ys = rfr.sample_pts(Y[0], blmm.N_POINTS, seed + 1)
            t1 = time.perf_counter()
            X_rig = blmm.icp(Xs, Ys, "mm")
            t2 = time.perf_counter()
            if a.metric == "icp":
                corr = fb.chamfer_correspondence(X_rig, Ys)
                d = float(np.mean(fb.p2p_distance(X_rig, Ys, corr)))
                t3 = time.perf_counter()
                rec(rows, "icp", "coppia", typ, pair_item(S, x, y), r, "cpu", campionamento=t1 - t0,
                    allineamento_rigido=t2 - t1, distanza=t3 - t2)
            else:
                try:                                                   # come pair_metrics: fallita = NaN
                    X_n = fb.nonrigid_icp_align(X_rig, Ys)
                    t3 = time.perf_counter()
                    corr = fb.chamfer_correspondence(X_n, Ys)
                    d = float(np.mean(fb.p2tri_distance(X_n, Ys, corr)))
                except Exception as exc:  # noqa: BLE001  (il tempo speso resta nella riga)
                    t3, d = time.perf_counter(), np.nan
                    failed.add(f"{pair_item(S, x, y)}: {type(exc).__name__}: {exc}")
                t4 = time.perf_counter()
                rec(rows, "nicp", "coppia", typ, pair_item(S, x, y), r, "cpu", campionamento=t1 - t0,
                    allineamento_rigido=t2 - t1, allineamento_nonrigido=t3 - t2, distanza=t4 - t3)
            if r == 0:
                D, subj, _, _, _ = bc.load_matrix(blmm.view_root(VIEW) / "mm" / "matrices" / metric /
                                                  f"{S['topo'][x]}__to__{S['topo'][y]}.npz")
                if subj != subjects:
                    raise SystemExit("soggetti della matrice diversi da blmm.subjects")
                got.append(d * unit)
                ref.append(D[s2i[S["subj"][x]], s2i[S["subj"][y]]])
    g, f = np.asarray(got), np.asarray(ref)
    checks = {f"{metric}_mm_vs_matrice_blmm": maxdiff(got, ref), "unit_mm": unit,
              "nan_qui_e_nella_matrice": int((np.isnan(g) & np.isnan(f)).sum()),
              "nan_solo_qui": int((np.isnan(g) & ~np.isnan(f)).sum()),
              "nan_solo_nella_matrice": int((~np.isnan(g) & np.isnan(f)).sum()), "fallite": sorted(failed)}
    write_part(a.out, a.metric, rows, checks)


# --------------------------------------------------------------------------------- NoW col rilevatore

def det_render(a) -> None:
    import pr_common as pc
    import pr_noise as prn
    import blmm
    from PIL import Image
    S = load_sample(a.out)
    v = pc.view(VIEW)
    out = a.work / "det"
    out.mkdir(parents=True, exist_ok=True)
    loc = {k: n for n, k in enumerate(S["meshes"])}
    cams, rows, same = {}, [], []
    for r, k in mesh_rounds(S, a.reps):
        t0 = time.perf_counter()
        V, F, _ = pc.mm_mesh(v, S["path"][k])
        t1 = time.perf_counter()
        img, cam, box = prn.render_mesh(V, F)
        t2 = time.perf_counter()
        Image.fromarray(img).save(out / f"{loc[k]:04d}.png")
        t3 = time.perf_counter()
        rec(rows, "now_det", "mesh", S["topo"][k], item_of(S, k), r, "cpu", caricamento=t1 - t0, render=t2 - t1,
            scrittura_png=t3 - t2)
        if r == 0:
            cams[k] = (cam["o"], cam["f"], box)
            old = np.asarray(Image.open(prn.NWORK / "det" / VIEW / f"{k:04d}.png"))
            same.append(bool(np.array_equal(old, img)))
    ks = S["meshes"]
    blmm.atomic_savez(out / "cams.npz", o=np.stack([cams[k][0] for k in ks]), f=np.asarray([cams[k][1] for k in ks]),
                      box=np.stack([cams[k][2] for k in ks]), paths=np.asarray([blmm.rel(S["path"][k]) for k in ks]),
                      img=prn.IMG)
    write_part(a.out, "det_render", rows, {"render_uguali_a_pr_noise": int(sum(same)), "render": len(same)})


def now(a) -> None:
    import pr_common as pc
    import pr_metrics as pm
    import pr_noise as prn
    import pr_run as prr
    S = load_sample(a.out)
    v = pc.view(VIEW)
    loc = {k: n for n, k in enumerate(S["meshes"])}
    dd = a.work / "det"
    with np.load(dd / "cams.npz") as z, np.load(dd / "lmk2d_cc.npz") as y:
        O, Fc, P = z["o"], z["f"], y["P"]
    IMG = prn.IMG
    rows, M = [], {}
    for r, k in mesh_rounds(S, a.reps):
        V, F, _ = pc.mm_mesh(v, S["path"][k])                         # gia' cronometrata in det-render
        t0 = time.perf_counter()
        # retroproiezione: le righe della condizione det di pr_noise._prep_task
        cam = {"o": O[loc[k]], "f": float(Fc[loc[k]])}
        scene = prn.scene_of(V, F)
        rr = prn.cast(scene, cam, P[loc[k]][:, :2])
        miss = ~rr["hit"]
        L68 = cam["o"] + rr["d"] * np.where(miss, 0.0, rr["t"])[:, None]
        if miss.any():
            yy, xx = np.mgrid[0:IMG, 0:IMG]
            full = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
            rf = prn.cast(scene, cam, full)
            hp = full[rf["hit"]]
            jn = np.linalg.norm(hp[None] - P[loc[k]][miss, None, :2], axis=2).argmin(1)
            jj = np.flatnonzero(rf["hit"])[jn]
            L68[miss] = cam["o"] + rf["d"][jj] * rf["t"][jj][:, None]
        t1 = time.perf_counter()
        L7 = L68[list(pc.LMK7)]
        crop = pm.crop(V, L7, prr.NOW_R)
        t2 = time.perf_counter()
        surf = pm.Surf(V, F)
        t3 = time.perf_counter()
        rec(rows, "now_det", "mesh", S["topo"][k], item_of(S, k), r, "cpu", retroproiezione=t1 - t0, ritaglio=t2 - t1,
            superficie_A=t3 - t2)
        M[k] = (L68, L7, crop, surf)
    with np.load(prn.prep_path(VIEW, "det"), allow_pickle=True) as z:
        L_old = z["L68"]
    ck = {"L68_vs_prep_det_mm": maxdiff(np.stack([M[k][0] for k in S["meshes"]]), L_old[S["meshes"]])}
    with np.load(prn.NWORK / "det" / VIEW / "lmk2d.npz") as z:          # landmark 2D della matrice
        ck["lmk2d_vs_pr_noise_px"] = maxdiff(P, z["P"][S["meshes"]])
    got, ref, split, whole = [], [], [], []
    for r in range(a.reps + 1):
        for pa, pb, typ in S["pairs"]:
            ph = {"procrustes": 0.0, "icp_now": 0.0, "distanza": 0.0}
            e = []
            for x, y in ((pa, pb), (pb, pa)):                          # A = x (predetta), B = y (riferimento)
                LA, LB, Bc, surf = M[x][1], M[y][1], M[y][2], M[x][3]
                t0 = time.perf_counter()
                tf = pm.procrustes(LB, LA, scaling=True)
                t1 = time.perf_counter()
                (sig, Q, tau), _ = pm.now_icp(surf, Bc, 1.0, tf["rotation"], tf["translation"], with_scale=False)
                t2 = time.perf_counter()
                e.append(float(pm.scan_to_mesh(surf, Bc, sig, Q, tau).mean()))
                t3 = time.perf_counter()
                ph["procrustes"] += t1 - t0
                ph["icp_now"] += t2 - t1
                ph["distanza"] += t3 - t2
                if r == 0:
                    split.append(e[-1])
                    whole.append(pm.now_pair(surf, LA, Bc, LB, metrical=True)["mean"])
            rec(rows, "now_det", "coppia", typ, pair_item(S, pa, pb), r, "cpu", **ph)
            if r == 0:
                got.append(sum(e) / 2)
    with np.load(pc.PLUGIN_ROOT / "now_metrical" / "lmk_det" / f"{VIEW}.npz") as z:
        Dm = np.asarray(z["D_mm"], np.float64)
    ref = [Dm[pa, pb] for pa, pb, _ in S["pairs"]]
    ck.update(now_det_mm_vs_plugin_lmk_det=maxdiff(got, ref), now_spezzata_vs_now_pair=maxdiff(split, whole),
              versi_per_coppia=2)
    write_part(a.out, "now", rows, ck)


def detcheck(a) -> None:
    """Landmark 2D di ``cc_detect.py`` (un thread) contro ``pr_noise_detect.py --dir`` sugli stessi render (thread di
    default, eseguito da cc.sbatch dopo le misure sui primi render, ``det_check`` di ``cc_detect.py``)."""
    with np.load(a.work / "det" / "lmk2d_cc.npz") as y, np.load(a.work / "det_check" / "lmk2d.npz") as z:
        n = len(z["P"])
        ck = {"render": n, "lmk2d_cc_vs_pr_noise_detect_stessi_render_px": maxdiff(y["P"][:n], z["P"]),
              "fallback_uguali": bool(np.array_equal(y["fallback"][:n], z["fallback"]))}
    write_part(a.out, "detcheck", [], ck)


# ------------------------------------------------------------------------------------------- nostro

def ops(a) -> None:
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(REPO / "v2_work" / "potential"))
    sys.path.insert(0, str(TRAINER / "tools"))
    import areanorm_operators as ao
    import build_scale_table as bst
    S = load_sample(a.out)
    u = float(json.loads(bst.FRAMES.read_text())["domains"][VIEW]["u"])
    out = a.work / "ops"
    out.mkdir(parents=True, exist_ok=True)
    rows, area = [], {}
    for r in range(a.reps + 1):
        for k in S["timed"]:                                           # solo le cronometrate entrano nel forward
            p = S["path"][k]
            t0 = time.perf_counter()
            V, F = ao.load_mesh(p)
            t1 = time.perf_counter()
            A = ao.total_area(V, F)                                    # righe di areanorm_operators.main
            Vn = (V - V.mean(0)) / np.sqrt(A)
            _, mass, L, evals, evecs, gX, gY = ao.compute_operators(torch.tensor(Vn, dtype=torch.float32),
                                                                    torch.tensor(F, dtype=torch.int32), k_eig=128)
            t2 = time.perf_counter()
            ao.save_npz(out / p.name, Vn, F, mass, L, evals, evecs, gX, gY)
            t3 = time.perf_counter()
            area[k] = u ** 2 * bst.total_area(V, F)
            t4 = time.perf_counter()
            rec(rows, "nostro", "mesh", S["topo"][k], item_of(S, k), r, "cpu", caricamento=t1 - t0,
                operatori=t2 - t1, scrittura_operatori=t3 - t2, tabella_scala=t4 - t3)
    with np.load(SCALE_TABLE) as z:
        tab = dict(zip([str(x) for x in z["names"]], z["area_mm2"].astype(np.float64)))
    ck = {"area_mm2_vs_tabella": maxdiff([area[k] for k in S["timed"]], [tab[S["path"][k].name] for k in S["timed"]])}
    write_part(a.out, "ops", rows, ck)


def ops_gpu(a, S: dict, out: Path) -> tuple[list, dict]:
    """Operatori su GPU, percorso di E10: ``gpu_ops.prepare`` (le righe di ``ops_areanorm``: centro e area 1) e
    ``gpu_ops.compute_batch(method="bk")`` una mesh alla volta (B = 1, latenza), k 128, opzioni di default (bs 16,
    m = max(4k, k + 256), tol 1e-5, 8 thread host per il riordino di cuDSS), poi l'npz come la pipeline CPU."""
    import torch
    sys.path.insert(0, str(E10))
    sys.path.insert(0, str(TRAINER / "tools"))
    import build_scale_table as bst
    import gpu_ops
    from potential_operators import load_mesh
    u = float(json.loads(bst.FRAMES.read_text())["domains"][VIEW]["u"])
    dev = torch.device("cuda")
    out.mkdir(parents=True, exist_ok=True)
    rows, conv, res = [], [], []
    for r in range(a.reps + 1):
        for k in S["timed"]:
            p = S["path"][k]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            m = gpu_ops.prepare(p)
            t1 = time.perf_counter()
            st = {}
            o = gpu_ops.compute_batch([m], 128, "bk", dev, eig_opts={"stats": st})[0]
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            gpu_ops.save_npz(o, out / p.name)
            t3 = time.perf_counter()
            V, F = load_mesh(p)                                          # gia' letta in prepare: non cronometrata
            t4 = time.perf_counter()
            bst.total_area(V, F) * u ** 2
            t5 = time.perf_counter()
            rec(rows, "nostro_gpu", "mesh", S["topo"][k], item_of(S, k), r, "cpu", caricamento=t1 - t0,
                scrittura_operatori=t3 - t2, tabella_scala=t5 - t4)
            rec(rows, "nostro_gpu", "mesh", S["topo"][k], item_of(S, k), r, "gpu", operatori_gpu=t2 - t1)
            conv.append(bool(st.get("converged", False)))
            res.append(float(st.get("max_res_rel", np.nan)))
    return rows, {"bk_convergenti": int(sum(conv)), "bk_chiamate": len(conv), "bk_max_res_rel": float(np.nanmax(res)),
                  "cudss": "nvidia-cudss-cu12 0.8.0.10 (.venv_e10)"}


def ours(a) -> None:
    import torch
    torch.set_num_threads(1)
    gpu = a.ops_src == "gpu"
    meth, step = ("nostro_gpu", "ours_gpu") if gpu else ("nostro", "ours")
    S = load_sample(a.out)
    ops_dir = a.work / ("ops_gpu" if gpu else "ops")
    rows, ck = ops_gpu(a, S, ops_dir) if gpu else ([], {})
    os.environ.update(WBES_V3_FACTORIZED_OUT="full", WBES_EVAL_SCENARIOS="clean", WBES_V3_SCALE_TABLES=str(SCALE_TABLE))
    sys.path.insert(0, str(TRAINER))
    sys.path.insert(0, str(AAU / "ava256"))
    import eval_v3
    from conf_embed import ARGS                                         # gli argomenti di d3_embed.sbatch
    _, cargs = eval_v3._ckpt_args(["--model_path", str(CKPT)])
    import robustness.model_helpers  # noqa: F401
    import robustness.data_utils  # noqa: F401
    eval_v3.install(cargs)
    pert = REPO / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic" / "perturbated"
    sys.path.insert(0, str(pert))
    sys.path.insert(0, str(pert.parent))
    import compare_model_vs_chamfer_topology_breakdown as bd
    base = bd.base
    names = a.work / "names.npz"                                        # come conf_embed stage: solo i nomi
    subj = sorted({S["subj"][k] for k in S["timed"]})
    np.savez(names, names=np.asarray(subj), D_orig=np.zeros((len(subj), len(subj))))
    sys.argv = ["cc", "--model_path", str(CKPT), "--data_dir", str(ops_dir), "--dist_npz", str(names), *ARGS]
    cli = bd.parse_args()
    _, ckpt = base._resolve_run_dir_and_checkpoint(cli.model_path, selector=str(cli.checkpoint_selector))
    margs = base._resolve_runtime_args(cli, base.merge_run_args(ckpt, explicit_config_json=cli.config_json))
    base.seed_everything(int(margs.seed))
    device = base._resolve_device(cli.device)
    if device.type != "cuda":
        raise SystemExit("ours: serve la GPU")
    dataset = base.GTReadyDataset(margs.data_dir)
    model = base.build_model(args=margs, device=device)
    model.load_state_dict(base.load_checkpoint_bundle(ckpt)["state_dict"], strict=True)
    model.eval()
    by_name = {S["path"][k].name: k for k in S["timed"]}
    files = [Path(f).name for f in dataset.files]
    if sorted(files) != sorted(by_name):
        raise SystemExit("ours: operatori diversi dalle mesh cronometrate")
    Z = {}
    with torch.no_grad():
        for r in range(a.reps + 1):
            for idx, fn in enumerate(files):
                k = by_name[fn]
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                smp = dataset[idx]
                t1 = time.perf_counter()
                sd = base.sample_to_device(smp, device=device)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                z, _ = base.forward_model(model=model, sample_dict=sd, V_in=sd["verts"], return_gate_info=False,
                                          add_noise=False)
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                rec(rows, meth, "mesh", S["topo"][k], item_of(S, k), r, "cpu", lettura_operatori=t1 - t0)
                rec(rows, meth, "mesh", S["topo"][k], item_of(S, k), r, "gpu", trasferimento=t2 - t1, forward=t3 - t2)
                if r == 0:
                    Z[k] = z.squeeze(0).double().cpu().numpy()
    Zs = np.stack([Z[k] for k in S["timed"]])
    np.savez(a.work / f"Z_{step}.npz", Z=Zs, timed=np.asarray(S["timed"]))
    if gpu:                                                             # operatori GPU contro CPU, stesso job
        with np.load(a.work / "Z_ours.npz") as z:
            Zc = z["Z"]
        iu, ju = np.triu_indices(len(Zs), 1)
        du = lambda X: np.linalg.norm(X[iu, 1:] - X[ju, 1:], axis=1)   # noqa: E731  (d_P a meno di dp_per_unit)
        dz = maxdiff(Zs, Zc)
        ck.update(Z_gpu_vs_cpu=dz, tolleranza_Z=TOL_Z_GPU, entro_tolleranza=bool(dz["max_abs"] <= TOL_Z_GPU),
                  distanze_u_gpu_vs_cpu=maxdiff(du(Zs), du(Zc)), log_s_gpu_vs_cpu=maxdiff(Zs[:, 0], Zc[:, 0]),
                  distanza_u_minima_cpu=float(du(Zc).min()))
    for tag, p in (("Z_vs_collaudo_L40S", EMB_L40S), ("Z_vs_sviluppo_A100", EMB_DEV)):
        with np.load(p, allow_pickle=True) as e:
            pos = {(str(s), str(t)): n for n, (s, t) in enumerate(zip(e["subjects"], e["topologies"]))}
            ref = np.stack([np.asarray(e["Z"][pos[(S["subj"][k], S["topo"][k])]], np.float64) for k in S["timed"]])
        ck[tag] = maxdiff(np.stack([Z[k] for k in S["timed"]]), ref)
    write_part(a.out, step, rows, ck, {"checkpoint": str(ckpt.relative_to(REPO)), "args": list(ARGS)})


def cmp(a) -> None:
    """d_F e d_P di ``fact_paired.distances`` su tutte le coppie di N mesh (vettori presi dagli embedding del collaudo,
    estratti con reinserimento: il costo non dipende dai valori)."""
    sys.path.insert(0, str(TRAINER / "tools"))
    import fact_paired as fp
    with np.load(EMB_L40S, allow_pickle=True) as e:
        Z0 = np.asarray(e["Z"], np.float64)
    rows = []
    for r in range(a.reps + 1):                                         # costo fisso di una chiamata (checkpoint)
        t0 = time.perf_counter()
        fp.distances(Z0, np.asarray([0]), np.asarray([1]), CKPT)
        rec(rows, "nostro", "blocco", "chiamata", "1 coppia", r, "cpu", chiamata_1_coppia=time.perf_counter() - t0)
    one = fp.distances(Z0, *np.triu_indices(60, 1), CKPT)
    rng = np.random.default_rng(SEED)
    for N in a.n_cmp:
        Zn = Z0[rng.integers(len(Z0), size=N)]
        for r in range(a.reps + 1):
            t0 = time.perf_counter()
            iu, ju = np.triu_indices(N, 1)
            t1 = time.perf_counter()
            for c in range(0, len(iu), CMP_CHUNK):
                d = fp.distances(Zn, iu[c:c + CMP_CHUNK], ju[c:c + CMP_CHUNK], CKPT)
            t2 = time.perf_counter()
            rec(rows, "nostro", "blocco", f"N={N}", f"{len(iu)} coppie", r, "cpu", indici=t1 - t0, confronti=t2 - t1)
            del iu, ju, d
            print(f"[cc] cmp N={N} giro {r}: {t2 - t1:.2f} s", flush=True)
    # controllo: a blocchi = in una chiamata (sulle prime 60 mesh)
    iu, ju = np.triu_indices(60, 1)
    blk = {k: np.concatenate([fp.distances(Z0, iu[c:c + 500], ju[c:c + 500], CKPT)[k] for c in range(0, len(iu), 500)])
           for k in ("form", "shape")}
    ck = {f"{k}_a_blocchi_vs_intera": maxdiff(blk[k], one[k]) for k in ("form", "shape")}
    write_part(a.out, "cmp", rows, ck, {"chunk": CMP_CHUNK, "n_cmp": list(a.n_cmp), "Z_dim": int(Z0.shape[1])})


# ------------------------------------------------------------------------------------------- ULRSSM

def ulrssm(a) -> None:
    import multiprocessing as mp
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ULR))
    import ulr_eval as UE
    U = UE.U
    S = load_sample(a.out)
    v = U.view(VIEW)
    for k in S["meshes"]:
        if Path(v["items"][k][2]).resolve() != S["path"][k].resolve():
            raise SystemExit("ulrssm: vista con mesh diverse dal campione")
    cache = a.work / "ulr"
    for r in range(a.reps + 1):                                        # get_operators vuole la cartella esistente
        (cache / f"r{r}").mkdir(parents=True, exist_ok=True)
    rows, R = [], {}
    UE._P.update(view=v, cache=str(cache / "r0"))                      # giro 0 (escluso): tutte le mesh, in parallelo
    with mp.get_context("fork").Pool(a.pool) as pool:
        for k, res in pool.imap(UE._prep, S["meshes"], chunksize=1):
            if not isinstance(res, dict):
                raise SystemExit(f"ulrssm: mesh {k} fallita: {res}")
            R[(0, k)] = res
    for r in range(1, a.reps + 1):                                     # giri cronometrati: cache nuova a ogni giro
        UE._P["cache"] = str(cache / f"r{r}")
        for k in S["timed"]:
            t0 = time.perf_counter()
            _, res = UE._prep(k)
            w = time.perf_counter() - t0
            R[(r, k)] = res
            rec(rows, "ulrssm", "mesh", S["topo"][k], item_of(S, k), r, "cpu", operatori=res["prep_s"],
                caricamento_e_scalari=w - res["prep_s"])
    device = torch.device("cuda")
    net, input_type = UE.U.load_net(ULR_CKPT, device, str(cache / "r0"))
    G = {}
    for r in range(a.reps + 1):
        for k in (S["meshes"] if r == 0 else S["timed"]):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g = UE.on_gpu(net, R[(r, k)], str(cache / f"r{r}"), v["cs_ref"], device)
            torch.cuda.synchronize()
            w = time.perf_counter() - t0
            rec(rows, "ulrssm", "mesh", S["topo"][k], item_of(S, k), r, "gpu", feature=g["feat_s"])
            rec(rows, "ulrssm", "mesh", S["topo"][k], item_of(S, k), r, "cpu", operatori_da_cache=w - g["feat_s"])
            if r == 0:
                G[k] = g
    got = []
    for r in range(a.reps + 1):
        for pa, pb, typ in S["pairs"]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            d = U.pair_rms(G[pa], G[pb])
            torch.cuda.synchronize()
            rec(rows, "ulrssm", "coppia", typ, pair_item(S, pa, pb), r, "gpu", mappe_e_distanza=time.perf_counter() - t0)
            if r == 0:
                got.append(d[0])
    with np.load(U.OUT / "faces_xyz" / f"{VIEW}.npz") as z:
        Dm = np.asarray(z["D_pair_fr"], np.float64)
    ck = {"d_pair_fr_vs_plugin_faces_xyz": maxdiff(got, [Dm[pa, pb] for pa, pb, _ in S["pairs"]]),
          "input_type": input_type}
    write_part(a.out, "ulrssm", rows, ck, {"checkpoint": str(ULR_CKPT.relative_to(REPO)), "pool_giro0": a.pool})


# ------------------------------------------------------------------------------------------- report

def per_item(df):
    """Per elemento e fase: mediana dei giri cronometrati (giro 0 escluso)."""
    t = df[df["rep"] > 0]
    return t.groupby(["method", "level", "group", "item", "dev", "phase"], as_index=False)["s"].median()


def stats(x) -> dict:
    x = np.asarray(x, np.float64)
    return {"n": len(x), "median": float(np.median(x)), "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)), "mean": float(x.mean())}


def fmt(s: float) -> str:
    """Secondi in unita' leggibili."""
    if s >= 1:
        return f"{s:.2f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.1f} ms"
    if s >= 1e-6:
        return f"{s * 1e6:.2f} us"
    return f"{s * 1e9:.1f} ns"


def fmt_h(s: float) -> str:
    if s >= 86400 * 2:
        return f"{s / 86400:,.1f} giorni"
    if s >= 3600:
        return f"{s / 3600:,.1f} h"
    if s >= 60:
        return f"{s / 60:.1f} min"
    return f"{s:.1f} s"


def report(a) -> None:
    import pandas as pd
    parts = {p.stem: json.loads(p.read_text()) for p in sorted((a.out / "parts").glob("*.json"))}
    S = parts.pop("sample")
    rows = [r for p in parts.values() for r in p["rows"]]
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "tempi.csv", index=False)
    it = per_item(df)
    # sintesi per fase e totali per elemento (somma delle fasi, per dispositivo e complessiva)
    tot = it.groupby(["method", "level", "group", "item", "dev"], as_index=False)["s"].sum()
    tot["phase"] = "totale_" + tot["dev"]
    tot_all = it.groupby(["method", "level", "group", "item"], as_index=False)["s"].sum()
    tot_all["phase"], tot_all["dev"] = "totale", "cpu+gpu"
    allp = pd.concat([it, tot, tot_all], ignore_index=True)
    syn = []
    for (m, lv, ph, dev), g in allp.groupby(["method", "level", "phase", "dev"]):
        for grp, gg in list(g.groupby("group")) + [("tutti", g)]:
            if lv == "blocco" and grp == "tutti":
                continue
            syn.append({"method": m, "level": lv, "phase": ph, "dev": dev, "group": grp, **stats(gg["s"])})
    syn = pd.DataFrame(syn)
    syn.to_csv(a.out / "sintesi.csv", index=False)

    def cell(m, lv, ph, grp="tutti"):
        r = syn[(syn.method == m) & (syn.level == lv) & (syn.phase == ph) & (syn.group == grp)]
        return r.iloc[0] if len(r) else None

    def mean_of(m, lv, ph):
        r = cell(m, lv, ph)
        return float(r["mean"]) if r is not None else 0.0

    # estrapolazione: N mesh, N(N-1)/2 coppie; per mesh e per coppia la MEDIA fra gli elementi (stima di una somma)
    cmp_med = {}                                    # indici delle coppie + confronti, mediana dei giri
    b = it[(it.method == "nostro") & (it.level == "blocco") & it.phase.isin(["indici", "confronti"])]
    for grp, g in b.groupby("group"):
        cmp_med[int(grp.split("=")[1])] = float(g["s"].sum())
    sc = []
    for N in N_CMP:
        P = N * (N - 1) / 2
        for m in METHODS:
            if m not in set(it.method):
                continue
            cpu = N * mean_of(m, "mesh", "totale_cpu") + P * mean_of(m, "coppia", "totale_cpu")
            gpu = N * mean_of(m, "mesh", "totale_gpu") + P * mean_of(m, "coppia", "totale_gpu")
            if m == "nostro":
                cpu += cmp_med.get(N, np.nan)
            sc.append({"N": N, "pairs": int(P), "method": m, "cpu_core_s": cpu, "gpu_s": gpu,
                       "wall_s_16core_1gpu": cpu / CORES_REF + gpu})
    sc = pd.DataFrame(sc)
    ours_wall = sc[sc.method == "nostro"].set_index("N")["wall_s_16core_1gpu"]
    sc["ratio_vs_nostro"] = [r.wall_s_16core_1gpu / ours_wall.get(r.N, np.nan) for r in sc.itertuples()]
    sc.to_csv(a.out / "scala.csv", index=False)

    env = {k: p["env"] for k, p in parts.items()}
    L = ["# Costo per confronto: il nostro encoder contro i protocolli per coppia", "",
         "Generato da `aau/costo_confronto/cc.py report` (definizioni nel docstring di `cc.py`). Tempi MISURATI in "
         "questa sessione; l'ultima tabella e' un'estrapolazione lineare dai tempi misurati.", "",
         "## Protocollo", "",
         f"- Mesh: HIFI3D (vista della matrice), topologie {', '.join(TOPOS)}. Coppie: {S['n_pairs_per_type']} per tipo "
         f"({', '.join(f'{x}-{y}' for x, y in TYPES)}), {len(S['pairs'])} in tutto, soggetti diversi, estratte con seme "
         f"{S['seed']} dal sottocampione di `pr_noise.subsample` (dove esistono i valori di tutti i metodi). Tempi per "
         f"mesh su {S['n_mesh_per_topology']} mesh per topologia fra quelle delle coppie.",
         f"- Ripetizioni: 1 giro di riscaldamento escluso + {int(df['rep'].max())} cronometrati; per elemento la "
         "mediana dei giri, poi mediana [IQR p25-p75] fra gli elementi.",
         "- CPU a UN thread per processo (secondi-core); GPU con `torch.cuda.synchronize`. Durante la misura girano in "
         "parallelo fino a ~7 processi a un thread sui core del job (stesse condizioni per tutti i metodi).",
         "- Una coppia dei protocolli NoW = i due versi (A predetta, B riferimento), come la matrice; ICP e NICP = un "
         "verso, come la matrice; ULRSSM = mappe nei due versi (`pair_rms`).", ""]
    e0 = next(iter(env.values()))
    gpu = next((e.get("gpu") for e in env.values() if e.get("gpu")), "-")
    L += ["## Hardware", "", f"- Nodo `{e0['host']}`, job {e0['job']}; CPU: {e0['cpu']}; core del job: "
          f"{e0['cores_job']}; GPU: {gpu}; thread: OMP/MKL/OpenBLAS = {e0['threads']['OMP_NUM_THREADS']}, torch 1, "
          "onnxruntime 1.", ""]
    phase_order = {"nostro": ["caricamento", "operatori", "scrittura_operatori", "tabella_scala", "lettura_operatori",
                              "trasferimento", "forward"],
                   "icp": ["caricamento"], "nicp": ["caricamento"],
                   "now_det": ["caricamento", "render", "scrittura_png", "lettura_png", "rilevamento", "ripiego",
                               "retroproiezione", "ritaglio", "superficie_A"],
                   "ulrssm": ["caricamento_e_scalari", "operatori", "operatori_da_cache", "feature"]}
    pair_order = {"icp": ["campionamento", "allineamento_rigido", "distanza"],
                  "nicp": ["campionamento", "allineamento_rigido", "allineamento_nonrigido", "distanza"],
                  "now_det": ["procrustes", "icp_now", "distanza"], "ulrssm": ["mappe_e_distanza"]}

    def row(m, lv, ph, groups):
        cells = []
        for g in groups:
            c = cell(m, lv, ph, g)
            cells.append(f"{fmt(c['median'])} [{fmt(c['p25'])}, {fmt(c['p75'])}]" if c is not None else "-")
        dev = syn[(syn.method == m) & (syn.level == lv) & (syn.phase == ph)]["dev"]
        return f"| {m} | {ph} | {dev.iloc[0] if len(dev) else '-'} | " + " | ".join(cells) + " |"

    L += ["## Per mesh (una volta per mesh: landmark, operatori, embedding)", "",
          "mediana [IQR] per mesh. `totale` = somma delle fasi.", "",
          "| metodo | fase | disp. | original | down8k | up60k | tutte |", "|---|---|---|---|---|---|---|"]
    groups = list(TOPOS) + ["tutti"]
    for m in METHODS:
        for ph in phase_order.get(m, []) + ["totale_cpu", "totale_gpu", "totale"]:
            if cell(m, "mesh", ph) is not None:
                L.append(row(m, "mesh", ph, groups))
    L += ["", "## Per coppia", "", "mediana [IQR] per coppia non ordinata.", "",
          "| metodo | fase | disp. | " + " | ".join(f"{x}-{y}" for x, y in TYPES) + " | tutte |",
          "|---|---|---|---|---|---|---|"]
    tgroups = [f"{x}-{y}" for x, y in TYPES] + ["tutti"]
    for m in METHODS:
        for ph in pair_order.get(m, []) + ["totale_cpu", "totale_gpu", "totale"]:
            if cell(m, "coppia", ph) is not None:
                L.append(row(m, "coppia", ph, tgroups))
    L += ["", "### Nostro: confronti fra vettori (`fact_paired.distances`, d_F e d_P), tutte le coppie di N mesh, CPU a "
          "un thread", "", "| N | coppie | tempo (mediana dei giri) | per coppia |", "|---|---|---|---|"]
    for N in sorted(cmp_med):
        P = N * (N - 1) // 2
        L.append(f"| {N:,} | {P:,} | {fmt(cmp_med[N])} | {fmt(cmp_med[N] / P)} |")
    c1 = cell("nostro", "blocco", "chiamata_1_coppia", "chiamata")
    if c1 is not None:
        L += ["", f"Costo fisso di una chiamata (rilegge il checkpoint): {fmt(c1['median'])}; i blocchi sono di "
              f"{CMP_CHUNK:,} coppie, quindi incluso una volta per blocco."]
    L += ["", "## Per coppia isolata (le due mesh da zero + la coppia)", "",
          "media per mesh x 2 + media per coppia (per il nostro: 2 embedding + 1 confronto dal blocco N = 10.000).", "",
          "| metodo | CPU (s-core) | GPU (s) |", "|---|---|---|"]
    for m in METHODS:
        if m not in set(it.method):
            continue
        cpu = 2 * mean_of(m, "mesh", "totale_cpu") + mean_of(m, "coppia", "totale_cpu")
        gp = 2 * mean_of(m, "mesh", "totale_gpu") + mean_of(m, "coppia", "totale_gpu")
        if m == "nostro" and 10000 in cmp_med:
            cpu += cmp_med[10000] / (10000 * 9999 / 2)
        L.append(f"| {m} | {fmt(cpu)} | {fmt(gp)} |")
    L += ["", "## Scalabilita': tutte le coppie di N mesh (ESTRAPOLAZIONE lineare dai tempi misurati)", "",
          "costo = N x (media per mesh) + N(N-1)/2 x (media per coppia); per il nostro i confronti sono MISURATI su N "
          "mesh (tabella sopra), gli embedding estrapolati. Media e non mediana: stima di una somma. Tempo a muro "
          f"stimato = s-core CPU / {CORES_REF} (parallelismo ideale sui {CORES_REF} core del nodo di misura) + s GPU "
          "(GPU in serie, nessuna sovrapposizione).", "",
          "| N | coppie | metodo | CPU (ore-core) | GPU (ore) | a muro, 16 core + 1 GPU | x nostro |",
          "|---|---|---|---|---|---|---|"]
    for r in sc.itertuples():
        x = f"{r.ratio_vs_nostro:,.2f}" if r.ratio_vs_nostro < 10 else f"{r.ratio_vs_nostro:,.0f}"
        L.append(f"| {r.N:,} | {r.pairs:,} | {r.method} | {r.cpu_core_s / 3600:,.3g} | {r.gpu_s / 3600:,.3g} | "
                 f"{fmt_h(r.wall_s_16core_1gpu)} | {x} |")
    L += ["", "## Controlli di riproduzione", ""]
    for k, p in parts.items():
        L.append(f"- `{k}`: `{json.dumps(p['checks'])}`")
    L += ["", "## Ambiente per passo", ""]
    for k, e in env.items():
        L.append(f"- `{k}`: {e['host']}, job {e['job']}, {e.get('gpu', 'senza GPU')}, core {e['cores_job']}, "
                 f"torch {e.get('torch', '-')} ({e.get('torch_threads', '-')} thread), {e['date']}")
    if (a.out / "gpu" / "parts" / "ours_gpu.json").exists():
        means = {m: {k: mean_of(m, lv, f"totale_{d}") for k, lv, d in (("mc", "mesh", "cpu"), ("mg", "mesh", "gpu"),
                                                                        ("pc", "coppia", "cpu"), ("pg", "coppia", "gpu"))}
                 for m in METHODS if m in set(it.method)}
        L += gpu_section(a.out / "gpu", means, cmp_med, sc)
    (a.out / "results.md").write_text("\n".join(L) + "\n")
    print(f"[cc] report: {a.out / 'results.md'}", flush=True)


def gpu_section(gout: Path, means: dict, cmp_med: dict, sc) -> list:
    """Sezione "Operatori su GPU": il nostro con gli operatori di E10 (``ours --ops-src gpu``, parti in ``gpu/parts``),
    i concorrenti e i confronti fra vettori RIUSATI dalla misura principale (``means``, ``cmp_med``, ``sc``)."""
    import pandas as pd
    parts = {p.stem: json.loads(p.read_text()) for p in sorted((gout / "parts").glob("*.json")) if p.stem != "sample"}
    df = pd.DataFrame([r for p in parts.values() for r in p["rows"]])
    df.to_csv(gout / "tempi.csv", index=False)
    it = per_item(df)
    tot = it.groupby(["method", "level", "group", "item", "dev"], as_index=False)["s"].sum()
    tot["phase"] = "totale_" + tot["dev"]
    tot_all = it.groupby(["method", "level", "group", "item"], as_index=False)["s"].sum()
    tot_all["phase"], tot_all["dev"] = "totale", "cpu+gpu"
    allp = pd.concat([it, tot, tot_all], ignore_index=True)
    syn = []
    for (m, ph, dev), g in allp.groupby(["method", "phase", "dev"]):
        for grp, gg in list(g.groupby("group")) + [("tutti", g)]:
            syn.append({"method": m, "level": "mesh", "phase": ph, "dev": dev, "group": grp, **stats(gg["s"])})
    syn = pd.DataFrame(syn)
    syn.to_csv(gout / "sintesi.csv", index=False)

    def cell(m, ph, grp="tutti"):
        r = syn[(syn.method == m) & (syn.phase == ph) & (syn.group == grp)]
        return r.iloc[0] if len(r) else None

    ours = {"mc": cell("nostro_gpu", "totale_cpu")["mean"], "mg": cell("nostro_gpu", "totale_gpu")["mean"]}
    P10 = 10000 * 9999 / 2
    cmp_pair = cmp_med[10000] / P10                                     # s-core per confronto (misura principale)
    e = parts["ours_gpu"]["env"]
    ck = parts["ours_gpu"]["checks"]
    L = ["", "## Operatori su GPU", "",
         "Il nostro con gli operatori calcolati su GPU invece che su CPU: percorso di E10 "
         "(`aau/evidence/e10_gpu_ops/gpu_ops.py`, `compute_batch(method=\"bk\")`: shift-invert con cuDSS + Krylov a "
         "blocchi, k 128, opzioni di default, una mesh alla volta), poi lo stesso npz, loader, forward e confronti. "
         f"Stesso nodo (`{e['host']}`), stesso campione (60 mesh cronometrate, 20 per topologia), job {e['job']}, "
         "passi in sequenza senza altri processi. I tempi dei concorrenti e dei confronti fra vettori NON sono "
         "rimisurati: sono quelli delle sezioni sopra. Il riordino di cuDSS usa fino a 8 thread host: il tempo "
         "`operatori_gpu` e' contato come GPU (a muro, in serie).", "",
         f"Tolleranza fissata prima della misura (`cc.TOL_Z_GPU`): max |Z_GPU - Z_CPU| <= {ck['tolleranza_Z']:g} sugli "
         f"embedding [s, u] delle 60 mesh, operatori CPU ricalcolati nello stesso job. Esito: max {ck['Z_gpu_vs_cpu']['max_abs']:.2e} "
         f"-> {'ENTRO' if ck['entro_tolleranza'] else 'FUORI'} tolleranza; distanze fra le u (1.770 coppie) max "
         f"{ck['distanze_u_gpu_vs_cpu']['max_abs']:.2e} (rel. {ck['distanze_u_gpu_vs_cpu']['max_rel']:.1e}), distanza "
         f"minima {ck['distanza_u_minima_cpu']:.3f}; log s max {ck['log_s_gpu_vs_cpu']['max_abs']:.1e}. Pavimento CPU nello "
         f"stesso job (operatori CPU ricalcolati contro il collaudo): {parts['ours']['checks']['Z_vs_collaudo_L40S']['max_abs']:.1e}; "
         f"GPU contro il collaudo: {ck['Z_vs_collaudo_L40S']['max_abs']:.1e}. `bk` convergente in "
         f"{ck['bk_convergenti']}/{ck['bk_chiamate']} chiamate (residuo relativo max {ck['bk_max_res_rel']:.1e}).", "",
         "### Per mesh, mediana [IQR]", "", "| variante | fase | disp. | original | down8k | up60k | tutte |",
         "|---|---|---|---|---|---|---|"]
    for m, phs in (("nostro_gpu", ["caricamento", "operatori_gpu", "scrittura_operatori", "tabella_scala",
                                   "lettura_operatori", "trasferimento", "forward", "totale_cpu", "totale_gpu", "totale"]),
                   ("nostro", ["totale_cpu", "totale_gpu", "totale"])):
        for ph in phs:
            if cell(m, ph) is None:
                continue
            cells = [f"{fmt(c['median'])} [{fmt(c['p25'])}, {fmt(c['p75'])}]" if (c := cell(m, ph, g)) is not None
                     else "-" for g in list(TOPOS) + ["tutti"]]
            lab = "operatori GPU" if m == "nostro_gpu" else "operatori CPU (stesso job, riferimento)"
            L.append(f"| {lab} | {ph} | {cell(m, ph)['dev']} | " + " | ".join(cells) + " |")
    L += ["", "### Per coppia isolata (le due mesh da zero + la coppia)", "", "| metodo | CPU (s-core) | GPU (s) |",
          "|---|---|---|", f"| nostro, operatori GPU | {fmt(2 * ours['mc'] + cmp_pair)} | {fmt(2 * ours['mg'])} |"]
    for m, x in means.items():
        if m != "nostro":
            L.append(f"| {m} (misura principale) | {fmt(2 * x['mc'] + x['pc'])} | {fmt(2 * x['mg'] + x['pg'])} |")

    def wall(c, g):
        return c / CORES_REF + g

    rows = []
    for N in N_CMP:
        Pn = N * (N - 1) / 2
        cpu, gp = N * ours["mc"] + cmp_med[N], N * ours["mg"]
        rows.append((N, int(Pn), "nostro, operatori GPU", cpu, gp, wall(cpu, gp)))
    L += ["", "### Scalabilita' (ESTRAPOLAZIONE lineare, stesso modello della tabella sopra)", "",
          "| N | coppie | metodo | CPU (ore-core) | GPU (ore) | a muro, 16 core + 1 GPU | x nostro GPU |",
          "|---|---|---|---|---|---|---|"]
    for N, Pn, m, cpu, gp, w in rows:
        L.append(f"| {N:,} | {Pn:,} | {m} | {cpu / 3600:,.3g} | {gp / 3600:,.3g} | {fmt_h(w)} | 1.00 |")
        for r in sc[sc.N == N].itertuples():
            x = r.wall_s_16core_1gpu / w
            xs = f"{x:,.2f}" if x < 10 else f"{x:,.0f}"
            lab = "nostro, operatori CPU" if r.method == "nostro" else r.method
            L.append(f"| {N:,} | {Pn:,} | {lab} | {r.cpu_core_s / 3600:,.3g} | {r.gpu_s / 3600:,.3g} | "
                     f"{fmt_h(r.wall_s_16core_1gpu)} | {xs} |")
    # pareggio con ICP: N m_o + P c_o = N m_i + P c_i  ->  N* = 1 + 2 (m_o - m_i) / (c_i - c_o)
    icp = means["icp"]
    L += ["", "### Punto di pareggio contro ICP + Chamfer mm", "",
          "N* = 1 + 2 (costo per mesh nostro - ICP) / (costo per coppia ICP - nostro): da N* mesh in su (tutte le "
          "coppie) il nostro costa meno. Due modelli di costo: a muro (s-core / 16 + s GPU, come la tabella) e "
          "risorse sommate (s-core + s GPU, un solo core).", "", "| variante del nostro | N*, a muro | N*, risorse sommate |",
          "|---|---|---|"]
    for lab, mc, mg in (("operatori GPU", ours["mc"], ours["mg"]), ("operatori CPU", means["nostro"]["mc"],
                                                                     means["nostro"]["mg"])):
        n_w = 1 + 2 * (wall(mc, mg) - wall(icp["mc"], icp["mg"])) / (wall(icp["pc"], icp["pg"]) - wall(cmp_pair, 0))
        n_r = 1 + 2 * ((mc + mg) - (icp["mc"] + icp["mg"])) / ((icp["pc"] + icp["pg"]) - cmp_pair)
        L.append(f"| {lab} | {n_w:,.1f} | {n_r:,.1f} |")
    L += ["", f"Controlli del passo: `{json.dumps(ck)}`"]
    return L


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("sample", "geo", "det-render", "now", "detcheck", "ops", "ours", "cmp", "ulrssm",
                                     "report"))
    p.add_argument("--metric", choices=("icp", "nicp"), default="icp")
    p.add_argument("--work", type=Path, default=None, help="cartella di lavoro del job (render, operatori, cache)")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--n-pairs", type=int, default=N_PAIRS)
    p.add_argument("--n-mesh", type=int, default=N_MESH)
    p.add_argument("--n-cmp", type=lambda s: tuple(int(x) for x in s.split(",")), default=N_CMP)
    p.add_argument("--pool", type=int, default=4, help="ulrssm: processi del giro 0 (escluso)")
    p.add_argument("--ops-src", choices=("cpu", "gpu"), default="cpu", help="ours: operatori CPU o GPU (E10)")
    a = p.parse_args()
    if a.step not in ("sample", "report", "cmp", "geo") and a.work is None:
        p.error("serve --work")
    {"sample": sample, "geo": geo, "det-render": det_render, "now": now, "detcheck": detcheck, "ops": ops, "ours": ours,
     "cmp": cmp, "ulrssm": ulrssm, "report": report}[a.step](a)


if __name__ == "__main__":
    main()
