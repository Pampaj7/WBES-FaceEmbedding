#!/usr/bin/env python3
"""Controllo: da dove viene la differenza fra GT maxabs e GT unificata? GT intermedie sulla patch NATIVA.

    v3_work/unified_gt/run.sh v3_work/unified_gt/native_gt.py      (dopo shapes.py)

Sui pool di 500 di HIFI3D e FaceVerse, sulla topologia ``original`` delle viste (la forma salvata
delle identita', niente mappa FLAME), una catena di GT che cambia un ingrediente alla volta:
  G0 ``maxabs``            la GT pubblicata (centro sulla media dei vertici, divisione per il max
                           |coordinata|, media delle norme per vertice);
  G1 ``native_maxabs_rms`` stessa normalizzazione, RMS per vertice invece della media delle norme;
  G2 ``native_sim``        Procrustes di similarita' (vertici a peso uguale) verso la media
                           generalizzata del pool, RMS per vertice: l'effetto dell'allineamento;
  G3 ``native_sim_area``   come G2 ma pesi d'area (allineamento e RMS): la stessa definizione della
                           GT unificata, sulla patch nativa;
  G3r ``native_sim_area_region``  come G3 ma solo sui vertici nativi dentro l'impronta della regione
                           unificata (punto piu' vicino sulla regione FLAME deformata, non sul suo bordo,
                           a meno di 2 mm): separa l'effetto della REGIONE da quello del ricampionamento;
  G4 ``unified``           la GT unificata (regione FLAME comune, via mappa baricentrica).
Se G3 e G4 si somigliano, la mappa FLAME non cambia il quadro e la differenza con maxabs viene
dall'allineamento; i salti G0->G1->G2->G3 dicono quanto pesa ogni ingrediente.

Scrive ``datasets/UNIFIED_GT/gt/<dominio>_<gt>.npz`` (formato di ``load_gt_distance_matrix``).
"""

from __future__ import annotations

import numpy as np

import ugt as C
import domains
from shapes import GT_DIR, ids_range


def procrustes_batch(P: np.ndarray, ref: np.ndarray, W: np.ndarray, scale: bool = True) -> np.ndarray:
    mx = np.einsum("v,nvd->nd", W, P)
    my = W @ ref
    Xc, Yc = P - mx[:, None], ref - my
    Cm = np.einsum("v,va,nvb->nab", W, Yc, Xc)
    U, S, Vt = np.linalg.svd(Cm)
    D = np.ones((len(P), 3))
    D[:, 2] = np.sign(np.linalg.det(U @ Vt))
    R = np.einsum("nab,nb,nbc->nac", U, D, Vt)
    s = (S * D).sum(1) / np.einsum("v,nvd->n", W, Xc ** 2) if scale else np.ones(len(P))
    return s[:, None, None] * np.einsum("nvb,nab->nva", Xc, R) + my


def gpa(P: np.ndarray, W: np.ndarray, n_iter: int = 5) -> np.ndarray:
    mu = P[0]
    for _ in range(n_iter):
        A = procrustes_batch(P, mu, W)
        new = A.mean(0)
        new = procrustes_batch(new[None], P[0], W)[0]     # frame e scala della prima forma
        mu = new
    return procrustes_batch(P, mu, W)


def rms_dist(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    X = A.reshape(len(A), -1) * np.repeat(np.sqrt(W), 3)[None]
    sq = (X ** 2).sum(1)
    D = np.sqrt(np.clip(sq[:, None] + sq[None] - 2 * X @ X.T, 0, None))
    np.fill_diagonal(D, 0)
    return 0.5 * (D + D.T)


def region_footprint(dom: str, V_native_mean: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Vertici della patch nativa dentro l'impronta della regione unificata (bool, n)."""
    import nicp as NI
    with np.load(C.DATA_ROOT / "unified_space.npz") as z:
        ridx, Fu = z["ridx"], z["F"]
    with np.load(C.CORR_DIR / f"{dom}.npz") as z:
        Y = z["Y_mm"][ridx]
        T = (float(z["T_landmark_s"]), z["T_landmark_R"], z["T_landmark_t"])
    # la patch nativa nel frame FLAME con la similarita' dei landmark (la stessa del NICP)
    X = C.apply_sim(V_native_mean, *T)
    surf = C.Surface(Y, Fu)
    c = surf.closest(X)
    bnd = NI.on_boundary(Fu, c["tri"], c["bary"], C.boundary_edge_mask(Fu), C.boundary_vertices(Fu, len(Y)))
    return ~bnd & (c["dist"] < 2.0)


def main() -> None:
    for dom, root, prefix, off in (("hifi3d", "HIFI3D", "hifi", 900000), ("faceverse", "FACEVERSE_ZS", "fv", 910000)):
        sids = ids_range(off, off + 500)
        Vs, F = [], None
        for s in sids:
            with np.load(domains.DATASETS / root / "identities" / f"{prefix}{int(s[2:]) - off:04d}.npz") as z:
                Vs.append(np.asarray(z["V"], np.float64))
                F = np.asarray(z["F"], np.int64) if F is None else F
        P = np.stack(Vs)
        n = P.shape[1]
        Wu = np.full(n, 1.0 / n)
        # G1: normalizzazione maxabs (normalize_vertices di faceBench), RMS
        Pm = P - P.mean(1, keepdims=True)
        Pm = Pm / np.abs(Pm).reshape(len(P), -1).max(1)[:, None, None]
        D1 = rms_dist(Pm, Wu)
        # G2: similarita' a peso uniforme verso la media generalizzata
        D2 = rms_dist(gpa(P, Wu), Wu)
        # G3: similarita' e RMS pesati per area (aree sulla media grezza del pool)
        wa = C.vertex_areas(P.mean(0), F)
        Wa = wa / wa.sum()
        D3 = rms_dist(gpa(P, Wa), Wa)
        # G3r: stessi pesi d'area, solo dentro l'impronta della regione unificata
        tpl = domains.template(dom)
        if dom == "hifi3d":
            import hifi_model
            Vmean, _ = hifi_model.face_patch(tpl["V"], hifi_model.load_hifi(domains.HIFI_MAT))
        else:
            Vmean = tpl["V"]
        if np.abs(Vmean - P.mean(0)).max() > 0.05 * np.abs(Vmean).max():
            print(f"[native] ATTENZIONE {dom}: media del modello lontana dalla media del pool", flush=True)
        foot = region_footprint(dom, Vmean, F)
        Wr = np.where(foot, wa, 0.0)
        Wr = Wr / Wr.sum()
        D3r = rms_dist(gpa(P, Wr), Wr)
        print(f"[native] {dom}: impronta della regione unificata {int(foot.sum())}/{n} vertici, "
              f"area {wa[foot].sum() / wa.sum():.2f} della patch", flush=True)
        for tag, D in (("native_maxabs_rms", D1), ("native_sim", D2), ("native_sim_area", D3),
                       ("native_sim_area_region", D3r)):
            C.save_npz(GT_DIR / f"{dom}_{tag}.npz", D_orig=D, names=np.array(sids))
        unused = n - len(np.unique(F))
        print(f"[native] {dom}: {len(sids)} identita', {n} vertici ({unused} non referenziati), "
              f"mediane D1 {np.median(D1[np.triu_indices(500, 1)]):.4g} D2 {np.median(D2[np.triu_indices(500, 1)]):.4g} "
              f"D3 {np.median(D3[np.triu_indices(500, 1)]):.4g}", flush=True)


if __name__ == "__main__":
    main()
