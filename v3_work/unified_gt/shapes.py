#!/usr/bin/env python3
"""Passo 4b: s_i per ogni identita' neutra, e le GT unificate dei set di valutazione.

    v3_work/unified_gt/run.sh v3_work/unified_gt/shapes.py --sets hifi3d,faceverse,bfm,ict,gnm,flame,multiface

Per un'identita' i del dominio d:
  1. p_i = mappa baricentrica di d applicata alla forma neutra (n, 3), nella regione unificata. Per i
     domini definiti dai pesi la mappa e' lineare: p_i = mappa(V_medio) + mappa(base) @ w_i, senza
     generare la mesh (controllo: coincide con la mappa della mesh generata, e la mesh generata con la
     patch salvata su disco);
  2. Procrustes di similarita' pesato per area verso la media globale mu: a_i = s R p_i + t;
  3. s_i = sqrt(w) * a_i, appiattito (3n,), float32.
GT: g_ij = ||s_i - s_j|| / sqrt(A), A = sum(w): RMS pesato per area, in mm (scala di mu = FLAME).

Set (``datasets/UNIFIED_GT/shapes/<set>.npz``: ``s``, ``ids``, ``scale``, ``rms_to_mu``):
  - hifi3d, faceverse: i pool di 500 delle viste zero-shot (id900000-, id910000-), pesi delle identita';
  - bfm: le 500 original REMESH (mesh: niente coefficienti);
  - ict: ICT-5000 (id10000-14999) + ICT nuove (id20000-69999, pesi dai manifest degli shard);
  - gnm: GNM_DISTILL (id100000-110099);
  - flame: 1000 identita' N(0, 1) sui 300 modi (seme 1234), solo per la dispersione del dominio;
  - multiface: la media dei frame neutri di ognuno dei 13 soggetti.
GT dei set di valutazione (formato di ``load_gt_distance_matrix``: ``D_orig``, ``names``):
``datasets/UNIFIED_GT/gt/<set>_unified.npz`` e ``<set>_unified_pairwise.npz`` (variante con
Procrustes per coppia, ``pairwise_distances``).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import ugt as C
import domains

SHAPES_DIR = C.DATA_ROOT / "shapes"
GT_DIR = C.DATA_ROOT / "gt"
CHUNK = 4000


class Space:
    def __init__(self):
        with np.load(C.DATA_ROOT / "unified_space.npz") as z:
            self.z = {k: z[k] for k in z.files}
        self.mu, self.w, self.A = self.z["mu"], self.z["w"], float(self.z["area_total"])
        self.W = self.w / self.w.sum()
        self.sqw = np.sqrt(self.w)

    def map(self, d: str, V: np.ndarray) -> np.ndarray:
        return C.bary_interp(V, self.z[f"vidx_{d}"], self.z[f"bary_{d}"])

    def linear(self, d: str, tpl: dict) -> tuple[np.ndarray, np.ndarray]:
        """(M0 (n, 3), B (k, n, 3)): punti mappati = M0 + w @ B."""
        vidx, bary = self.z[f"vidx_{d}"], self.z[f"bary_{d}"]
        M0 = C.bary_interp(tpl["V"], vidx, bary)
        B = np.einsum("nkdj,nk->jnd", tpl["basis"][vidx], bary)
        return M0, B

    def align(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Procrustes di similarita' pesato di ogni forma (N, n, 3) verso mu: (a, scala, rms)."""
        W, mu = self.W, self.mu
        mx = np.einsum("v,nvd->nd", W, P)
        my = W @ mu
        Xc = P - mx[:, None]
        Yc = mu - my
        Cm = np.einsum("v,va,nvb->nab", W, Yc, Xc)
        U, S, Vt = np.linalg.svd(Cm)
        d = np.sign(np.linalg.det(U @ Vt))
        D = np.ones((len(P), 3))
        D[:, 2] = d
        R = np.einsum("nab,nb,nbc->nac", U, D, Vt)
        s = (S * D).sum(1) / np.einsum("v,nvd->n", W, Xc ** 2)
        a = s[:, None, None] * np.einsum("nvb,nab->nva", Xc, R) + my
        rms = np.sqrt(np.einsum("v,nv->n", W, ((a - mu) ** 2).sum(-1)))
        return a, s, rms

    def svec(self, a: np.ndarray) -> np.ndarray:
        return (self.sqw[None, :, None] * a).reshape(len(a), -1).astype(np.float32)


def unified_distances(S1: np.ndarray, S2: np.ndarray, A: float) -> np.ndarray:
    """g_ij = ||s_i - s_j|| / sqrt(A), (n1, n2)."""
    S1, S2 = S1.astype(np.float64), S2.astype(np.float64)
    sq = (S1 ** 2).sum(1)[:, None] + (S2 ** 2).sum(1)[None, :] - 2.0 * S1 @ S2.T
    return np.sqrt(np.clip(sq, 0.0, None) / A)


def pairwise_distances(S: np.ndarray, sp: Space) -> np.ndarray:
    """Variante: Procrustes di similarita' (pesato) per ogni coppia, simmetrizzata.

    Per forme centrate x, y (pesi W normalizzati) il minimo di sum W ||s R x + t - y||^2 e'
    sigma_y^2 - T^2 / sigma_x^2, con T = somma dei valori singolari di sum W y x^T (col segno del
    determinante). d(i, j) = media delle radici nei due versi.
    """
    n = len(S)
    a = S.astype(np.float64).reshape(n, -1, 3) / sp.sqw[None, :, None]
    a = a - np.einsum("v,nvd->nd", sp.W, a)[:, None]
    sig = np.einsum("v,nvd->n", sp.W, a ** 2)
    aw = a * np.sqrt(sp.W)[None, :, None]
    out = np.zeros((n, n))
    for i in range(n):
        Cm = np.einsum("jva,vb->jab", aw, aw[i])          # sum W y_j x_i^T
        sv = np.linalg.svd(Cm, compute_uv=False)
        det = np.sign(np.linalg.det(Cm))
        T = sv[:, 0] + sv[:, 1] + det * sv[:, 2]
        dij = np.sqrt(np.clip(sig - T ** 2 / sig[i], 0, None))
        dji = np.sqrt(np.clip(sig[i] - T ** 2 / sig, 0, None))
        out[i] = 0.5 * (dij + dji)
    np.fill_diagonal(out, 0.0)
    return 0.5 * (out + out.T)


# ------------------------------------------------------------------------------ set

def ids_range(lo: int, hi: int) -> list[str]:
    return [f"id{g}" if g >= 10000 else f"id{g:04d}" for g in range(lo, hi)]


def from_weights(sp: Space, d: str, W: np.ndarray) -> tuple:
    tpl = domains.template(d)
    M0, B = sp.linear(d, tpl)
    out_s, out_scale, out_rms = [], [], []
    for k in range(0, len(W), CHUNK):
        P = M0[None] + np.einsum("nk,kvd->nvd", W[k:k + CHUNK, : B.shape[0]], B)
        a, s, rms = sp.align(P)
        out_s.append(sp.svec(a))
        out_scale.append(s)
        out_rms.append(rms)
    # controllo: mappa lineare == mappa della mesh generata (prime 3 identita')
    lin_check = 0.0
    for j in range(min(3, len(W))):
        Vfull = tpl["V"] + tpl["basis"][:, :, : W.shape[1]] @ W[j]
        P1 = sp.map(d, Vfull)
        P2 = M0 + np.einsum("k,kvd->vd", W[j, : B.shape[0]], B)
        lin_check = max(lin_check, float(np.abs(P1 - P2).max()))
    return np.concatenate(out_s), np.concatenate(out_scale), np.concatenate(out_rms), {"linear_vs_mesh_max_abs": lin_check}


def check_stored(d: str, sids: list[str], W: np.ndarray) -> float:
    """Mesh generata dai pesi contro la forma salvata su disco (prime 3 identita'), max |diff|."""
    tpl = domains.template(d)
    diff = 0.0
    for j in range(min(3, len(sids))):
        V = tpl["V"] + tpl["basis"][:, :, : W.shape[1]] @ W[j]
        if d in ("hifi3d", "faceverse"):
            from hifi_model import face_patch
            import hifi_model, fv_model  # noqa: F401
            m = hifi_model.load_hifi(domains.HIFI_MAT) if d == "hifi3d" else fv_model.load_fv(domains.FV_NPY)
            Vp, _ = face_patch(V, m)
            ref = domains.zs_stored_vertices(d, sids[j])
        elif d == "ict":
            Vp = V
            g = int(sids[j][2:])
            if g < 15000:
                with np.load(domains.DATASETS / "ICT" / "identities" / f"ict{g - 10000:04d}.npz") as z:
                    ref = z["V"]
            else:
                ref = read_shard_member(domains.DATASETS / "ICT_SCALE" / "shards", sids[j])
        elif d == "gnm":
            import gnm_model
            m = gnm_model.load_gnm(domains.GNM_NPZ)
            Vh = m["v_template"] + m["shapedirs"] @ W[j]
            Vp, _ = gnm_model.face_patch(Vh, m)
            ref = read_shard_member(domains.DATASETS / "GNM_DISTILL" / "shards", sids[j])
        else:
            return float("nan")
        diff = max(diff, float(np.abs(np.asarray(Vp) - np.asarray(ref, dtype=np.float64)).max()))
    return diff


def read_shard_member(shard_dir, sid: str) -> np.ndarray:
    import io
    import tarfile
    for tar in sorted(shard_dir.glob("*shard_*.tar")):
        with tarfile.open(tar) as t:
            try:
                m = t.getmember(f"{sid}_GTready_original.npz")
            except KeyError:
                continue
            with np.load(io.BytesIO(t.extractfile(m).read())) as z:
                return z["V"]
    raise KeyError(sid)


def build_set(sp: Space, name: str) -> dict:
    t0 = time.time()
    checks = {}
    if name in ("hifi3d", "faceverse"):
        off = 900000 if name == "hifi3d" else 910000
        sids = ids_range(off, off + 500)
        W = domains.zs_weights(name, sids)
        s, scale, rms, ch = from_weights(sp, name, W)
        checks.update(ch)
        checks["weights_vs_stored_max_abs"] = check_stored(name, sids, W)
    elif name == "ict":
        sids = ids_range(10000, 15000) + ids_range(20000, 70000)
        W = domains.ict_weights(sids)
        s, scale, rms, ch = from_weights(sp, "ict", W)
        checks.update(ch)
        checks["weights_vs_stored_max_abs"] = check_stored("ict", [sids[0], sids[5000]], W[[0, 5000]])
    elif name == "gnm":
        sids = ids_range(100000, 110100)
        W = domains.gnm_weights(sids)
        s, scale, rms, ch = from_weights(sp, "gnm", W)
        checks.update(ch)
        checks["weights_vs_stored_max_abs"] = check_stored("gnm", sids, W)
    elif name == "flame":
        sids = [f"flame{k:04d}" for k in range(1000)]
        W = np.random.default_rng(1234).normal(size=(1000, 300))
        s, scale, rms, ch = from_weights(sp, "flame", W)
        checks.update(ch)
    elif name == "bfm":
        paths = domains.bfm_original_paths()
        sids = [p.name.split("_GTready_")[0] for p in paths]
        P = np.stack([sp.map("bfm", domains.load_bfm_original(p)[0]) for p in paths])
        a, scale, rms = sp.align(P)
        s = sp.svec(a)
    elif name == "multiface":
        groups = domains.multiface_neutral_paths()
        sids, P = [], []
        for subj, paths in sorted(groups.items()):
            Vs = []
            for p in paths:
                with np.load(p) as z:
                    V = np.asarray(z["V"], np.float64)
                if Vs:
                    V = C.apply_sim(V, *C.umeyama(V, Vs[0], scale=False))
                Vs.append(V)
            sids.append(subj)
            P.append(sp.map("multiface", np.mean(Vs, axis=0)))
        a, scale, rms = sp.align(np.stack(P))
        s = sp.svec(a)
    else:
        raise SystemExit(f"set sconosciuto: {name}")
    C.save_npz(SHAPES_DIR / f"{name}.npz", s=s, ids=np.array(sids), scale=scale, rms_to_mu=rms)
    info = {"set": name, "n": len(sids), "seconds": time.time() - t0,
            "rms_to_mu_mm": {"median": float(np.median(rms)), "p95": float(np.percentile(rms, 95))}, **checks}
    print(f"[shapes] {info}", flush=True)
    if name in ("hifi3d", "faceverse"):
        D = unified_distances(s, s, sp.A)
        np.fill_diagonal(D, 0.0)
        D = 0.5 * (D + D.T)
        C.save_npz(GT_DIR / f"{name}_unified.npz", D_orig=D, names=np.array(sids))
        Dp = pairwise_distances(s, sp)
        C.save_npz(GT_DIR / f"{name}_unified_pairwise.npz", D_orig=Dp, names=np.array(sids))
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sets", default="hifi3d,faceverse,bfm,gnm,flame,multiface,ict")
    a = p.parse_args()
    sp = Space()
    infos = {}
    for name in a.sets.split(","):
        infos[name] = build_set(sp, name)
    prev = {}
    if (SHAPES_DIR / "manifest.json").exists():
        import json
        prev = json.loads((SHAPES_DIR / "manifest.json").read_text())
    prev.update(infos)
    C.save_json(SHAPES_DIR / "manifest.json", prev)


if __name__ == "__main__":
    main()
