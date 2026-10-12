#!/usr/bin/env python3
"""Controlli della logica portata (``pr_metrics``), senza GT. Ogni controllo stampa OK / FALLITO e il numero.

    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_selftest.py [--view hifi3d]

1. Procrustes: ritrova una similarita' nota (scala, rotazione, traslazione) da 7 punti.
2. Rodrigues: jacobiano analitico contro differenze finite.
3. NoW: B = A trasformata con una similarita' nota (scala 1.08, 7 gradi, 15 mm) e landmark perturbati di 2 mm ->
   NoW originale ~0 (l'ICP con scala recupera la similarita'); NoW metrical > 0 (la scala resta) ma uguale a quello che
   da' l'ICP rigido sulla mesh riscalata della scala vera (controllo con la scala 1).
4. Dogleg contro ``scipy.optimize.least_squares`` sullo stesso obiettivo NoW (stesso minimo).
5. nICP: una soluzione di ``_lsq`` contro ``scipy.sparse.linalg.lsqr`` sullo stesso sistema.
6. REALY: B = A trasformata (similarita') -> errore ~0 su tutte le regioni.
7. Stirling: B = A trasformata con landmark esatti -> RMSE ~0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as pc  # noqa: E402
import pr_landmarks as pl  # noqa: E402
import pr_metrics as pm  # noqa: E402

RES = []


def check(name: str, ok: bool, value) -> None:
    RES.append(ok)
    print(f"[selftest] {'OK' if ok else 'FALLITO'} {name}: {value}", flush=True)


def rot(axis, deg):
    R, _ = pm.rodrigues(np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(deg))
    return R


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--view", default="hifi3d")
    a = p.parse_args()
    rng = np.random.default_rng(0)
    # 1
    X = rng.normal(size=(7, 3)) * 30
    R0 = rot([0.3, 1, 0.2], 12)
    Y = (X - 5) @ R0.T / 1.1
    tf = pm.procrustes(X, Y)
    Z = tf["scale"] * Y @ tf["rotation"] + tf["translation"]
    check("Procrustes ritrova la similarita'", np.abs(Z - X).max() < 1e-9 and abs(tf["scale"] - 1.1) < 1e-9,
          f"scarto {np.abs(Z - X).max():.1e}, scala {tf['scale']:.6f}")
    # 2
    r = np.array([0.2, -0.4, 0.7])
    _, dR = pm.rodrigues(r)
    num = np.stack([(pm.rodrigues(r + 1e-6 * e)[0] - pm.rodrigues(r - 1e-6 * e)[0]) / 2e-6 for e in np.eye(3)])
    check("jacobiano di Rodrigues", np.abs(num - dR).max() < 1e-7, f"scarto {np.abs(num - dR).max():.1e}")
    # mesh vere
    v = pc.view(a.view)
    pr = pc.read_prep(v["name"])
    k = 0
    V, F, _ = pc.mm_mesh(v, v["items"][k][2])
    L7 = pr["L68"][k][list(pc.LMK7)]
    K85 = pr["K85"][k]
    Rt = rot([0.2, 1.0, 0.1], 7)
    s_true, t_true = 1.08, np.array([15.0, -4.0, 6.0])
    VB = s_true * V @ Rt.T + t_true
    LB = s_true * L7 @ Rt.T + t_true
    LBn = LB + rng.normal(size=LB.shape) * 2.0
    surf = pm.Surf(V, F)
    Bc = pm.crop(VB, LBn, 1.4)
    r_o = pm.now_pair(surf, L7, Bc, LBn, metrical=False)
    check("NoW originale su B = similarita' di A", r_o["mean"] < 0.02,
          f"media {r_o['mean']:.4f} mm, scala {r_o['scale']:.5f} (vera {s_true}), {r_o['iterations']} iterazioni, "
          f"{r_o['fevals']} valutazioni, arresto {r_o['stop']}")
    r_m = pm.now_pair(surf, L7, Bc, LBn, metrical=True)
    VB1 = V @ Rt.T + t_true                                 # stessa B senza la scala: il metrical deve dare ~0
    r_1 = pm.now_pair(surf, L7, pm.crop(VB1, L7 @ Rt.T + t_true, 1.4), L7 @ Rt.T + t_true, metrical=True)
    check("NoW metrical vede la scala (B scalata 1.08) e non la rigida", r_m["mean"] > 1.0 and r_1["mean"] < 0.02,
          f"scalata {r_m['mean']:.3f} mm, rigida {r_1['mean']:.4f} mm")
    # 4: dogleg contro least_squares sullo stesso obiettivo, B = original di un ALTRO soggetto (minimo non nullo)
    from scipy.optimize import least_squares
    k2 = next(i for i, (s, t, _) in enumerate(v["items"]) if t == "original" and s != v["items"][k][0])
    V2, _, _ = pc.mm_mesh(v, v["items"][k2][2])
    L2 = pr["L68"][k2][list(pc.LMK7)]
    tfp = pm.procrustes(L2, L7)
    P = pm.crop(V2, L2, 1.4)
    P = P[:: max(1, len(P) // 3000)]

    def resid(x):
        s, R, t = x[0], pm.rodrigues(x[1:4])[0], x[4:7]
        sig, Q, tau = s * tfp["scale"], tfp["rotation"] @ R.T, s * tfp["translation"] @ R.T + t
        return pm.scan_to_mesh(surf, P, sig, Q, tau)

    ls = least_squares(resid, np.r_[1.0, np.zeros(6)], method="lm", xtol=1e-12, ftol=1e-12)
    (sig, Q, tau), _ = pm.now_icp(surf, P, tfp["scale"], tfp["rotation"], tfp["translation"], with_scale=True)
    e_dl = float((pm.scan_to_mesh(surf, P, sig, Q, tau) ** 2).sum())
    e_ls = float((ls.fun ** 2).sum())
    check("dogleg = least_squares (somma dei quadrati, B di un altro soggetto)", e_dl <= e_ls * 1.001,
          f"dogleg {e_dl:.3f}, LM {e_ls:.3f} mm^2")
    # 5
    import scipy.sparse.linalg as spl
    ref = pr["realy"][k]
    Pn, Fn = ref["regions"]["nose"]
    M = pm.edge_incidence(Fn.astype(np.int64), len(Pn))
    import scipy.sparse as sp
    A = sp.vstack([sp.kron(M, sp.diags([1.0, 1, 1, 1])) * 50, pm._D(Pn)]).tocsc()
    B = np.vstack([np.zeros((A.shape[0] - len(Pn), 3)), Pn + 0.1])
    X1 = pm._lsq(A, B)
    X2 = np.stack([spl.lsqr(A, B[:, c], atol=1e-14, btol=1e-14, iter_lim=200000)[0] for c in range(3)], axis=1)
    check("soluzione del nICP (splu delle equazioni normali) = lsqr", np.abs(X1 - X2).max() < 1e-5,
          f"scarto {np.abs(X1 - X2).max():.1e}")
    # 6
    KB = s_true * K85 @ Rt.T + t_true
    refB = pl.realy_reference(VB, F, pr["labels"][k], KB, 1500)
    rr = pm.realy_pair(surf, K85, refB)
    # ~0 ma non 0: il keypoint del riferimento e' il suo vertice piu' vicino nella regione ([Re] :51), non il punto
    # esatto, e il nICP tira quel vertice sul keypoint di A
    check("REALY su B = similarita' di A", rr["total"] < 0.15,
          f"totale {rr['total']:.4f} mm, regioni " + ", ".join(f"{x} {rr['err_' + x]:.4f}" for x in pc.REGIONS))
    # 7
    rs = pm.stirling_pair(surf, L7, pm.crop(VB, LB, 1.2), LB)
    check("Stirling su B = similarita' di A con landmark esatti", rs["rmse"] < 1e-3, f"RMSE {rs['rmse']:.2e} mm")
    print(f"[selftest] {sum(RES)}/{len(RES)} OK", flush=True)
    if not all(RES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
