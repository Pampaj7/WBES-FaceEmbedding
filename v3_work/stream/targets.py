"""Bersagli della GT di riferimento dopo E12 (PLAN_MASSIVE sez. 20), per identita', al volo nei produttori.

Lo stesso codice di ``v3_work/canonical_gt/train_fr_sr.py::_factor_chunk`` (importato da ``cgt``, non copiato):
  - f_i = u_d R_d p_i + t_d, il frame di GT-F (``cgt.Canon.to_F``), p_i = punti della regione unificata della
    forma NEUTRA dell'identita' (``sources.*.neutral_points``), unita' native;
  - a_i = rigida robusta verso mu (``Canon.rigid_robust``, IRLS Tukey);
  - S_i = centroid size pesata di a_i (mm), c_i = centroide pesato, z_i = (a_i - c_i) / S_i.
Vettori come in ``fr_train.npz`` / ``sr_train.npz``: ``fr`` = sqrt(w) a_i e ``sr`` = sqrt(w) z_i appiattiti, float32,
quindi d_FR = ||fr_i - fr_j|| / sqrt(A) (mm) e d_P = ||sr_i - sr_j|| / sqrt(A): la stessa forma della GT
unificata (``consumer.StreamGT``), cambia solo il vettore.

Frame per dominio dello stream (``u_d`` e' l'unica parte che conta: la rigida robusta toglie R_d e t_d):
  ict, gnm         ``aau/runs/evidence/e12/frames.json`` (u dichiarata: cm, m);
  flame2020, flame2023  la voce ``flame`` (m -> mm; FLAME 2023 Open ha la stessa topologia e le stesse unita');
  bfm2019          assente da E12: u = ``frame.mm_per_unit`` della libreria (mm, dichiarata, la regola di E12),
                   R_d e t_d = rigida della media mappata verso la media FLAME (come ``gt.py::frames``);
  famos            catture reali gia' in mm: nessuna trasformazione (``Canon.to_F``), poi la rigida robusta;
  bfm              BFM REMESH di E12 (solo per i controlli contro i file: non e' un dominio dello stream).

Fattore delle aree (``mm_factor``): le viste dello stream sono nel frame canonico della libreria, la similarita'
del json (scala s_d = u_d k_d, la media del dominio portata sulla taglia della media FLAME); la taglia vera e'
u_d / s_d volte quella della vista. Le aree della vista vanno moltiplicate per mm_factor^2 (``area_mm2``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (REPO_ROOT / "v3_work" / "unified_gt", REPO_ROOT / "v3_work" / "canonical_gt"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

E12_KEY = {"ict": "ict", "gnm": "gnm", "flame2020": "flame", "flame2023": "flame", "bfm": "bfm"}   # bfm2019, famos: sotto


class CanonTargets:
    """FR, SR e S_i di un'identita' dai suoi punti nativi della regione unificata."""

    def __init__(self, sources: dict | None = None) -> None:
        import cgt
        self.cn = cgt.Canon()
        if not self.cn.frames:
            raise SystemExit(f"{cgt.FRAMES_JSON} assente: i frame di GT-F sono di E12 (canonical_gt/gt.py)")
        for d, k in E12_KEY.items():
            self.cn.frames[d] = self.cn.frames[k]
        self.sq = np.sqrt(self.cn.sp.w)[:, None]
        self.A = float(self.cn.sp.A)
        self.n = int(len(self.cn.mu))
        if sources and "bfm2019" in sources:
            self.cn.frames["bfm2019"] = self.bfm2019_frame(sources["bfm2019"])

    def bfm2019_frame(self, src) -> dict:
        """u = unita' dichiarata (mm); R, t = rigida della media mappata verso la media FLAME (gt.py::frames)."""
        import domains
        from cgt import C
        u = float(src.model.frame.mm_per_unit)
        P_flame = domains.flame()["V"][self.cn.sp.z["flame_vidx"]] * 1000.0
        w_flame = C.vertex_areas(P_flame, self.cn.sp.z["F"])
        _, R, t = C.umeyama(u * src.mean_u, P_flame, w_flame, scale=False)
        return {"u": u, "R": R.tolist(), "t": t.tolist(), "unit_source": "dichiarata (v3_work/mm, frame.mm_per_unit)"}

    def mm_factor(self, domain: str, src) -> float:
        """mm veri per mm della vista dello stream (u_d / s_d); 1 per FaMoS (gia' metrico, rigida senza scala)."""
        if domain == "famos":
            return 1.0
        return float(self.cn.frames[domain]["u"]) / float(src.canon["scale"])

    def __call__(self, domain: str, P: np.ndarray) -> dict:
        """``fr``, ``sr`` (3n,) float32; ``S`` (mm), ``c`` (3,), ``converged``, ``iterations``. ``P``: (n, 3)
        o (b, n, 3) punti nativi; con (b, n, 3) ogni voce ha la dimensione b davanti."""
        cn = self.cn
        P = np.asarray(P, dtype=np.float64)
        one = P.ndim == 2
        X = cn.to_F(domain, P[None] if one else P)
        rb = cn.rigid_robust(X)
        a = rb["a"]
        S = cn.centroid_size(a)
        c = cn.centroid(a)
        z = (a - c[:, None]) / S[:, None, None]
        out = {"fr": (self.sq[None] * a).reshape(len(a), -1).astype(np.float32),
               "sr": (self.sq[None] * z).reshape(len(a), -1).astype(np.float32),
               "S": S, "c": c, "converged": rb["converged"], "iterations": rb["iterations"]}
        return {k: v[0] for k, v in out.items()} if one else out

    def describe(self) -> dict:
        return {d: {"u": float(f["u"]), "unit_source": f.get("unit_source", "")} for d, f in self.cn.frames.items()
                if d in ("ict", "gnm", "flame2020", "flame2023", "bfm2019", "bfm")}


# --- scala della GT nel trainer ------------------------------------------------------------------------------

CGT_TRAIN = REPO_ROOT / "datasets" / "CANONICAL_GT" / "train"
UNIT_KEY = {"fr": ("gt_fr_bfm_ict_gnm_calib.json", "mm_per_unit_by_domain"),
            "sr": ("gt_sr_bfm_ict_gnm_calib.json", "dP_per_unit_by_domain")}


def gt_unit(kind: str) -> float:
    """Unita' fisiche (mm per fr, d_P per sr) per unita' della GT dello stream: la media geometrica delle unita'
    per dominio della versione TARATA di E12 (BFM, ICT, GNM). Per sr e' d_P_per_unit / kappa, la scala dei bracci
    factorized (GT-SR grezza x kappa, kappa = media geometrica dei fattori); per fr la stessa regola (i bracci
    ctrlfr usano un fattore per dominio, che con batch misti fra domini non e' definito)."""
    import json
    name, key = UNIT_KEY[kind]
    u = json.loads((CGT_TRAIN / name).read_text())[key]
    return float(np.exp(np.mean([np.log(float(u[d])) for d in ("bfm", "ict", "gnm")])))
