#!/usr/bin/env python3
"""Fit dei 3DMM (GNM Head, FLAME 2023 Open) sulle mesh valutate: regione, iscrizione NICP, regressione MAP.

    aau/outlineB/run_o3d.sh aau/baselines_param/bp_fit.py --views hifi3d,famos --models gnm,flame2023 --workers 32
    (bp.sbatch, passo fit; definizioni: bp.py)

Per (vista, modello): ``bp.region`` (scritta in ``<OUT_ROOT>/<vista>/<modello>/region.npz`` e ``region.json``),
poi per ogni mesh di ``bp.meshes`` (100 soggetti x 5 topologie senza crop; FaMoS: 15 scansioni di galleria)
``blmm.work_coords`` nel modo ``mm``, ``bp.enroll_model`` col seme ``blmm.mesh_seed(soggetto, topologia)`` (quello
del NICP su template di blmm) e ``bp.fit_points``. Fallimento (eccezione, valori non finiti, RMS > ``FAIL_RMS_MM``):
coefficienti NaN, motivo in ``failed``. Uscita ``fit.npz``: ``subjects``, ``topologies``, ``beta`` (n, k_id), ``psi``,
``theta``, ``R``, ``t``, ``rms_mm``, ``nicp_resid_mm``, ``seconds``, ``failed``, ``region_vertices``,
``model_file``, ``model_sha256`` e le matrici (n, n) di ``bp.mesh_distances`` nell'ordine delle mesh: ``D_coef``,
``D_fr``, ``D_sr`` (NaN sulle righe dei fit falliti). Ripartibile: una (vista, modello) gia' su disco si salta.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

import bp
import blmm  # noqa: E402  (aau/baselines_mm, messo su sys.path da bp)

_C: dict = {}           # contesto della (vista, modello), ereditato dai worker (fork)


def _one(task):
    subject, topology, path = task
    t0 = time.perf_counter()
    nan = {"beta": np.full(_C["ctx"]["k_id"], np.nan), "psi": np.full(_C["n_psi"], np.nan), "theta": np.nan,
           "R": np.full((3, 3), np.nan), "t": np.full(3, np.nan), "rms": np.nan, "nicp": np.nan}
    try:
        X, _ = blmm.work_coords(_C["view"], "mm", path, _C["prm"])
        Y, nd = bp.enroll_model(X, blmm.mesh_seed(subject, topology), _C["ctx"])
        f = bp.fit_points(Y, _C["ctx"])
        f["nicp"] = nd
        ok = all(np.isfinite(np.asarray(f[k], dtype=np.float64)).all() for k in ("beta", "psi", "theta", "R", "t"))
        if not ok:
            return {**nan, "rms": f["rms"], "nicp": nd}, time.perf_counter() - t0, "valori non finiti"
        if not f["rms"] <= bp.FAIL_RMS_MM:
            return {**nan, "rms": f["rms"], "nicp": nd}, time.perf_counter() - t0, f"RMS {f['rms']:.2f} mm"
        return f, time.perf_counter() - t0, ""
    except Exception as exc:  # noqa: BLE001  (la mesh resta NaN, contata)
        return nan, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--views", default=",".join(bp.VIEWS))
    p.add_argument("--models", default=",".join(bp.MODELS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    prm = blmm.params()
    for view in a.views.split(","):
        items = bp.meshes(view)
        for name in a.models.split(","):
            root = bp.out_dir(view, name)
            out = root / "fit.npz"
            if out.exists() and not a.overwrite:
                print(f"[bp-fit] {out}: gia' presente, salto", flush=True)
                continue
            t0 = time.time()
            m = bp.load_model(name)
            reg = bp.region(view, m)
            blmm.atomic_savez(root / "region.npz", vertices=reg["vertices"], votes=reg["votes"],
                              n_refs=reg["n_refs"], n_model=len(m["mu"]))
            bp.atomic_json(root / "region.json", {"view": view, "model": name, "n_vertices": int(len(reg["vertices"])),
                                                  "n_model_vertices": int(len(m["mu"])), "references": reg["diag"]})
            ctx = bp.context(view, name, reg["vertices"], prm)
            _C.update(view=view, prm=prm, ctx=ctx, n_psi=ctx["C_s"].shape[2] - ctx["k_id"])
            print(f"[bp-fit] {view} {name}: regione {len(reg['vertices'])}/{len(m['mu'])} vertici "
                  f"({reg['n_refs']} riferimenti), template {len(ctx['sub'])} punti, {len(items)} mesh", flush=True)
            with mp.get_context("fork").Pool(a.workers) as pool:
                res = pool.map(_one, items, chunksize=1)
            failed = [f"{s}|{t}|{r[2]}" for (s, t, _), r in zip(items, res) if r[2]]
            D = bp.mesh_distances(np.stack([r[0]["beta"] for r in res]), ctx)
            blmm.atomic_savez(
                out, subjects=np.asarray([s for s, _, _ in items]), topologies=np.asarray([t for _, t, _ in items]),
                paths=np.asarray([blmm.rel(q) for _, _, q in items]),
                **{k: np.stack([np.asarray(r[0][k], dtype=np.float64) for r in res]) for k in ("beta", "psi", "theta", "R", "t")},
                rms_mm=np.asarray([r[0]["rms"] for r in res]), nicp_resid_mm=np.asarray([r[0]["nicp"] for r in res]),
                seconds=np.asarray([r[1] for r in res]), failed=np.asarray(failed, dtype="U300"),
                region_vertices=reg["vertices"], template_sub=ctx["sub"], model_file=ctx["file"],
                model_sha256=bp.sha256(ctx["file"]), sigma_noise_mm=bp.SIGMA_NOISE_MM, fail_rms_mm=bp.FAIL_RMS_MM,
                **{f"D_{k}": M for k, M in D.items()})
            rms = np.asarray([r[0]["rms"] for r in res])
            print(f"[bp-fit] {view} {name}: {len(items)} mesh in {time.time() - t0:.0f}s, mediana "
                  f"{np.median([r[1] for r in res]):.1f}s/mesh, RMS mediana {np.nanmedian(rms):.2f} mm (max "
                  f"{np.nanmax(rms):.2f}), fallite {len(failed)}" + (f" ({failed[:3]})" if failed else ""), flush=True)


if __name__ == "__main__":
    main()
