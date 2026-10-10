#!/usr/bin/env python3
"""D1 (PROTOCOL_D.md sez. 3.2-3.3): mesh grezze, tabella di scala e GT degli insiemi nuovi.

    AAU_NV="" aau/run.sh aau/diagnostics/d1_gen.py --workers 32          (diag.sbatch, passo gen)

Insiemi (``diag.NEW_SETS``): ``regen_ict`` e ``regen_gnm`` (i 100 + 100 held-out della calibrazione, dai loro pesi e coi
semi di ``noisy`` statici: controllo C1), ``flame2023_s1`` e ``flame2023`` (200 identita' FLAME 2023 Open N(0, 1) non
troncate, ``SeedSequence([SEED_ID, k])``; patch suddivisa 1-a-4 una volta o nativa). Per identita': mesh di lavoro
= media + base x z della sorgente dello stream (``sources.MMSource``, unita' e frame NATIVI, facce della sorgente),
le 5 discretizzazioni con ``views.discretize`` (funzioni di make_ict_topologies), salvate {V float32, F int32} in
``datasets/DIAG_D1/in/<sid>_GTready_<etichetta>.npz``. GT al volo dello stream (``targets.CanonTargets``) dai punti
neutri della regione unificata: vettori fr, sr, S e matrici d_FR (mm), d_P.

Uscite in ``aau/runs/evidence/diagnostics/d1``: ``scale_table.npz`` (formato di fact_calib.stage: area_mm2 = u_d^2 x area
grezza, dominio = chiave del frame di E12), ``gt_<insieme>.npz``, ``gt_all_sr.npz`` (D_orig a blocchi + names: solo per
la scelta dei soggetti di zs_embed), ``gen.json`` (semi, conteggi, controlli C1-GT, tempi).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import diag  # noqa: E402

sys.path.insert(0, str(diag.REPO / "v3_work/stream"))
import sources as S  # noqa: E402  (mette v3_work/unified_gt su sys.path)
import views as VW  # noqa: E402

_C: dict = {}           # sorgenti e identita', ereditate dai worker (fork)


def identities(name: str, src) -> list[tuple[str, np.ndarray, int]]:
    """(sid, z, seme di noisy) dell'insieme."""
    cfg = diag.SETS[name]
    if cfg["id_base"] is None:
        import domains as DM
        sids = diag.heldout_subjects()[cfg["frame"]]
        W = (DM.ict_weights if cfg["source"] == "ict" else DM.gnm_weights)(sids)
        if cfg["source"] == "ict":              # ICT-5000: make_ict_topologies, seed=int(subject[-4:])
            if not all(10000 <= int(s[2:]) < 15000 for s in sids):
                raise SystemExit("held-out ICT fuori da ICT-5000: regola del seme di noisy non nota")
            seeds = [int(s[-4:]) for s in sids]
        else:                                   # GNM_DISTILL: SeedSequence([20261011, j]), j = id - 100000
            seeds = [int(np.random.SeedSequence([diag.GNM_NOISE_SEED, int(s[2:]) - 100000]).generate_state(1)[0])
                     for s in sids]
        return [(s, np.asarray(w[: src.n_id], np.float64), sd) for s, w, sd in zip(sids, W, seeds)]
    out = []
    for k in range(diag.N_FLAME):
        rng = np.random.default_rng(np.random.SeedSequence([diag.SEED_ID, k]))
        z = src.model.sample_identity(rng, tails=False, trunc=0.0, purpose="eval")
        seed = int(np.random.SeedSequence([diag.SEED_NOISE, k]).generate_state(1)[0])
        out.append((f"id{cfg['id_base'] + k}", z, seed))
    return out


def work_mesh(src, z: np.ndarray, subdiv: int) -> tuple[np.ndarray, np.ndarray]:
    """Mesh di lavoro nelle unita' native: patch della sorgente, eventualmente suddivisa 1-a-4 (punto medio)."""
    V = src.mean_w + src.id_w @ np.asarray(z, dtype=np.float64)
    F = np.asarray(src.faces, dtype=np.int64)
    if subdiv:
        import igl
        V, F = igl.upsample(V, F, int(subdiv))
    return V, F


def _gen(task):
    name, k = task
    src, (sid, z, seed) = _C["src"][name], _C["ident"][name][k]
    V, F = work_mesh(src, z, diag.SETS[name]["subdiv"])
    u = _C["u"][diag.SETS[name]["frame"]]
    rows = []
    for lab in diag.LABELS:
        Vd, Fd = VW.discretize(V, F, lab, seed)
        fn = f"{sid}_GTready_{lab}.npz"
        np.savez(diag.RAW_IN / fn, V=Vd.astype(np.float32), F=Fd.astype(np.int32))
        Vr = Vd.astype(np.float32).astype(np.float64)       # come la rilegge fact_calib.stage
        a = float(0.5 * np.linalg.norm(np.cross(Vr[Fd[:, 1]] - Vr[Fd[:, 0]], Vr[Fd[:, 2]] - Vr[Fd[:, 0]]),
                                       axis=1).sum())
        rows.append((fn, diag.SETS[name]["frame"], u ** 2 * a, a, float(np.abs(Vr - Vr.mean(0)).max()),
                     int(len(Vd)), int(len(Fd))))
    return rows


def gt_of(name: str, src, tg) -> dict:
    """Vettori fr, sr, S per identita' e matrici d_FR (mm), d_P (``targets.CanonTargets`` sui punti neutri)."""
    P = np.stack([src.neutral_points(z) for _, z, _ in _C["ident"][name]])
    t = tg(diag.SETS[name]["source"], P)
    out = {"names": np.asarray([s for s, _, _ in _C["ident"][name]]), "S": np.asarray(t["S"], np.float64),
           "converged": np.asarray(t["converged"])}
    for kind in ("fr", "sr"):
        X = np.asarray(t[kind], np.float64)
        G = (X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2.0 * X @ X.T
        D = np.sqrt(np.clip(G, 0.0, None) / tg.A)
        np.fill_diagonal(D, 0.0)
        out[kind] = np.asarray(t[kind], np.float32)
        out[f"D_{kind}"] = 0.5 * (D + D.T)
    return out


def c1_gt(name: str, g: dict) -> dict:
    """C1-GT: d_P contro gt_sr.npz x dP_per_unit (scarto relativo), d_FR contro gt_frcal.npz (rapporto costante)."""
    out = {}
    iu = np.triu_indices(len(g["names"]), 1)
    for kind, path in (("sr", diag.GT_SR), ("fr", diag.GT_FR)):
        with np.load(path, allow_pickle=True) as z:
            pos = {str(n): k for k, n in enumerate(z["names"])}
            ii = np.asarray([pos[str(s)] for s in g["names"]])
            ref = np.asarray(z["D_orig"], np.float64)[np.ix_(ii, ii)][iu]
        x = g[f"D_{kind}"][iu]
        if kind == "sr":
            ref = ref * float(json.loads(path.with_suffix(".json").read_text())["dP_per_unit"])
            out["sr_max_rel"] = float(np.max(np.abs(x - ref) / np.maximum(ref, 1e-12)))
        else:
            r = x / np.maximum(ref, 1e-12)
            out["fr_ratio_median"] = float(np.median(r))
            out["fr_ratio_rel_spread"] = float((r.max() - r.min()) / np.median(r))
    out["pass"] = bool(out["sr_max_rel"] <= 1e-4 and out["fr_ratio_rel_spread"] <= 1e-4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--sets", default=",".join(diag.NEW_SETS))
    a = ap.parse_args()
    t0 = time.time()
    names = a.sets.split(",")
    frames = json.loads(diag.FRAMES.read_text())["domains"]
    uni = S.Unified()
    srcs = S.build_sources(sorted({diag.SETS[n]["source"] for n in names}), uni)
    for d, s in srcs.items():
        fr = frames[{"flame2023": "flame"}.get(d, d)]
        if abs(float(s.model.frame.mm_per_unit) - float(fr["u"])) > 1e-9 or s.canon["flip_faces"]:
            raise SystemExit(f"{d}: unita' {s.model.frame.mm_per_unit} contro u di E12 {fr['u']}, o facce capovolte")
    from targets import CanonTargets
    tg = CanonTargets(srcs)
    _C.update(src={n: srcs[diag.SETS[n]["source"]] for n in names},
              ident={n: identities(n, srcs[diag.SETS[n]["source"]]) for n in names},
              u={k: float(v["u"]) for k, v in frames.items()})
    diag.RAW_IN.mkdir(parents=True, exist_ok=True)
    tasks = [(n, k) for n in names for k in range(len(_C["ident"][n]))]
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(_gen, tasks, chunksize=2)
    rows = [r for block in res for r in block]
    t_gen = time.time() - t0
    rep = {"definition": "PROTOCOL_D.md sez. 3.2-3.3", "seeds": {"identity": diag.SEED_ID, "noise": diag.SEED_NOISE,
                                                                 "gnm_noise": diag.GNM_NOISE_SEED},
           "sets": {}, "n_meshes": len(rows), "seconds_gen": t_gen, "raw_dir": str(diag.RAW_IN)}
    gts = {}
    for n in names:
        g = gt_of(n, _C["src"][n], tg)
        gts[n] = g
        diag.atomic_savez(diag.D1 / f"gt_{n}.npz", **g)
        mine = [r for r in rows if r[0].split("_GTready_")[0] in set(g["names"].tolist())]
        info = {"n_identities": len(g["names"]), "n_meshes": len(mine), "frame": diag.SETS[n]["frame"],
                "source": diag.SETS[n]["source"], "subdiv": diag.SETS[n]["subdiv"],
                "gt_converged": float(g["converged"].mean()), "S_mean_mm": float(g["S"].mean()),
                "dP_gt_median": float(np.median(g["D_sr"][np.triu_indices(len(g["names"]), 1)])),
                "verts_by_label": {lab: [int(np.min([r[5] for r in mine if r[0].endswith(f"_{lab}.npz")])),
                                         int(np.max([r[5] for r in mine if r[0].endswith(f"_{lab}.npz")]))]
                                   for lab in diag.LABELS}}
        if diag.SETS[n]["id_base"] is None:
            info["c1_gt"] = c1_gt(n, g)
        rep["sets"][n] = info
        print(f"[d1-gen] {n}: {info}", flush=True)
    order = np.argsort([r[0] for r in rows])
    rows = [rows[k] for k in order]
    diag.atomic_savez(diag.D1 / "scale_table.npz", names=np.asarray([r[0] for r in rows]),
                      domain=np.asarray([r[1] for r in rows]), area_mm2=np.asarray([r[2] for r in rows]),
                      area_raw=np.asarray([r[3] for r in rows]), maxabs_raw=np.asarray([r[4] for r in rows]),
                      check=np.full(len(rows), np.nan))
    allnames = np.concatenate([gts[n]["names"] for n in names])
    if len(set(allnames.tolist())) != len(allnames):
        raise SystemExit("id ripetuti fra gli insiemi nuovi")
    D = np.zeros((len(allnames), len(allnames)), np.float32)
    o = 0
    for n in names:
        k = len(gts[n]["names"])
        D[o:o + k, o:o + k] = gts[n]["D_sr"]
        o += k
    diag.atomic_savez(diag.D1 / "gt_all_sr.npz", D_orig=D, names=allnames)
    rep["seconds"] = time.time() - t0
    diag.atomic_json(diag.D1 / "gen.json", rep)
    print(f"[d1-gen] {len(rows)} mesh in {diag.RAW_IN}, {t_gen:.0f}s di generazione, {rep['seconds']:.0f}s in tutto",
          flush=True)


if __name__ == "__main__":
    main()
