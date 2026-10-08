#!/usr/bin/env python3
"""Prove della libreria 3DMM: controlli per modello, render, statistiche di campionamento, determinismo.

    srun -p prioritized --gres=NONE -c 8 --mem=64G -t 00:40:00 env AAU_NV= \\
        aau/run.sh v3_work/mm/selftest.py --out-dir aau/runs/evidence/dev_facescape/mm_selftest
    ... v3_work/mm/selftest.py --calibrate          # solo la taratura di EXPR_SCALE (modelli pca)

Per ogni modello (``--models``, default tutti):
  1. controlli: patch pulita (``prepare_open_surface`` non toglie niente), ``mesh`` a coefficienti
     nulli = media; frame canonico di ripiego: assi dai landmark (angoli esterni degli occhi ->
     +x, bocca -> occhi -> +y, punta del naso davanti agli occhi in +z), normali uscenti
     (frazione di triangoli uscenti dal baricentro > 0.7), distanza fra gli angoli esterni degli occhi in mm; per i
     bilineari, pesi d'espressione nulli = neutra e ``mesh(z, e) = mesh(z) + expr_basis_at(z) @ e``;
  2. render (``v2_work/phase0/render_mesh.py``, frame canonico): riga 1 media e 3 identita'
     (code pesanti), riga 2 le stesse con un'espressione, riga 3 le neutre a 60 gradi;
  3. statistiche: coefficienti (code pesanti contro N(0,1)), distanze fra identita' in mm e dopo
     maxabs (la convenzione della GT), spostamento delle espressioni (maxabs, contro lo 0.017 della
     ricetta ICT), ampiezza dei bump RBF, coppie di perturbazione rispetto alla distanza mediana;
  4. determinismo: due esecuzioni con lo stesso seme danno array identici, un altro seme no;
  5. ruoli: ``purpose="train"`` su un modello dev/test solleva ``RoleError``;
  6. hook del JSON dei frame: una voce in un JSON temporaneo si applica, un modello assente ripiega.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "phase0"))

from v3_work.mm import MODELS, RoleError, load_model  # noqa: E402
from v3_work.mm.loaders import EXPR_SCALE  # noqa: E402
import mesh_ops as mo  # noqa: E402  (sul path grazie a loaders)

GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)          # aau/distill/gen_gnm_shard.py
TARGET_SHIFT = 0.017                              # ricetta ICT, make_zs_expr_topologies.ICT_REFERENCE
CALIB_SEED = 20261013
N_CALIB = 50


def maxabs(V: np.ndarray) -> np.ndarray:
    Vc = np.asarray(V, dtype=np.float64) - np.mean(V, axis=0)
    return Vc / np.abs(Vc).max()


def shift(V1: np.ndarray, V0: np.ndarray) -> float:
    return float(np.linalg.norm(maxabs(V1) - maxabs(V0), axis=1).mean())


def purpose_of(m) -> str:
    return "train" if m.role == "train" else "eval"


# ----------------------------------------------------------------------------- controlli

def frame_checks(m) -> dict:
    Vc, Fc = m.canonical_transform(m.mean, m.faces)
    tri = Vc[Fc]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    # verso: frazione di triangoli con la normale lontana dal baricentro (vale anche per la testa
    # intera, dove la somma delle normali la decide il buco del collo e non il volto)
    out = {"canonical_source": m.canonical_params()["source"],
           "bbox_mm": (Vc.max(0) - Vc.min(0)).round(1).tolist(),
           "normal_sum_dot_z": float(n.sum(0)[2] / np.linalg.norm(n.sum(0))),
           "frac_outward": float(((tri.mean(1) - Vc.mean(0)) * n).sum(1).__gt__(0).mean())}
    ids = None if m.landmark_ids is None else {int(k): i for i, k in enumerate(m.landmark_ids)}
    if m.landmark_scheme in ("ibug68", "ibug51", "faceverse66", "bfm2019") and all(k in ids for k in (30, 36, 45)):
        L = lambda k: Vc[m.landmarks[ids[k]]]  # noqa: E731
        eyes = (L(36) + L(45)) / 2
        mouth = (L(48) + L(54)) / 2 if m.landmark_scheme != "faceverse66" else L(60) if 60 in ids else L(57)
        left, up = L(45) - L(36), eyes - mouth
        out.update({"eye_outer_mm": float(np.linalg.norm(left)),
                    "cos_left_x": float(left[0] / np.linalg.norm(left)),
                    "cos_up_y": float(up[1] / np.linalg.norm(up)),
                    "nose_ahead_mm": float(L(30)[2] - eyes[2])})
        out["frame_ok"] = bool(out["cos_left_x"] > 0.95 and out["cos_up_y"] > 0.8 and out["nose_ahead_mm"] > 5
                               and out["frac_outward"] > 0.7)
    else:
        # niente landmark iBUG: punta del naso = vertice piu' avanti, deve stare sulla mezzeria
        tip = Vc[np.argmax(Vc[:, 2])]
        c = (Vc.max(0) + Vc.min(0)) / 2
        out.update({"nose_tip_offset_x_mm": float(tip[0] - c[0]), "nose_tip_rel_height": float(
            (tip[1] - Vc[:, 1].min()) / (Vc[:, 1].max() - Vc[:, 1].min()))})
        out["frame_ok"] = bool(abs(out["nose_tip_offset_x_mm"]) < 10 and 0.3 < out["nose_tip_rel_height"] < 0.7
                               and out["frac_outward"] > 0.7)
    return out


def json_vs_fallback(m) -> dict | None:
    """Voce del JSON reale contro il frame di ripiego: angolo fra le rotazioni, rapporto di scala."""
    p = m.canonical_params()
    if not p["source"].startswith("json:"):
        return None
    R0, s0 = m.frame.rotation(), m.frame.mm_per_unit
    cosang = (np.trace(p["R"] @ R0.T) - 1.0) / 2.0
    return {"angle_deg": float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))),
            "scale_ratio": float(p["scale"] / s0),
            "flip_faces_agree": bool(p["flip_faces"] == (m.frame.normals == "inward"))}


def model_checks(m) -> dict:
    Vp, Fp = mo.prepare_open_surface(m.mean, m.faces)
    out = {"patch_clean": bool(len(Vp) == m.n_verts and len(Fp) == len(m.faces)),
           "mesh_zero_is_mean": bool(np.allclose(m.mesh(np.zeros(m.n_id)), m.mean))}
    if m.expr.kind != "none":
        out["mesh_zero_expr_is_mean"] = bool(np.allclose(m.mesh(np.zeros(m.n_id), np.zeros(m.n_expr)), m.mean,
                                                         atol=1e-6 * np.abs(m.mean).max()))
    rng = np.random.default_rng(0)
    z = m.sample_identity(rng, purpose=purpose_of(m))
    e = m.sample_expression(rng, purpose=purpose_of(m))
    lin = m.mesh(z) + m.expr_basis_at(z) @ e
    out["expr_additive_max_abs"] = float(np.abs(m.mesh(z, e) - lin).max())
    out["expr_additive_ok"] = bool(out["expr_additive_max_abs"] < 1e-5 * np.abs(m.mean).max())
    out.update(frame_checks(m))
    out["json_vs_fallback"] = json_vs_fallback(m)
    return out


# ----------------------------------------------------------------------------- render

def render_grid(m, path: Path, size: int = 256) -> None:
    from PIL import Image
    from render_mesh import _yaw_rotate, render_mesh
    rng = np.random.default_rng(2026)
    p = purpose_of(m)
    ids = [None] + [m.sample_identity(rng, purpose=p) for _ in range(3)]
    exprs = [m.sample_expression(rng, purpose=p) for _ in range(4)]
    rows = []
    for row in range(3):
        tiles = []
        for k, z in enumerate(ids):
            V = m.mesh(z, exprs[k] if row == 1 else None)
            Vc, Fc = m.canonical_transform(V, m.faces)
            Vr = Vc * np.array([1.0, -1.0, -1.0])         # frame del renderer: alto -y, naso -z
            if row == 2:
                Vr = _yaw_rotate(Vr, 60.0, (Vr.max(0) + Vr.min(0)) / 2)
            tiles.append(render_mesh(Vr, Fc, size=size))
        rows.append(np.concatenate(tiles, axis=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


# ----------------------------------------------------------------------------- statistiche

def coef_stats(z: np.ndarray) -> dict:
    x = z.ravel()
    return {"std": float(x.std()), "excess_kurtosis": float(((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3),
            "max_abs": float(np.abs(x).max()), "frac_abs_gt2": float((np.abs(x) > 2).mean()),
            "frac_abs_gt3": float((np.abs(x) > 3).mean())}


def pairwise_vertex_l2(X: np.ndarray) -> np.ndarray:
    """(N, N) media per vertice della distanza L2 (la grandezza della GT)."""
    N = len(X)
    D = np.zeros((N, N))
    for i in range(N):
        D[i, i + 1:] = np.linalg.norm(X[i + 1:] - X[i], axis=2).mean(axis=1)
    return D + D.T


def sampling_stats(m, n_id: int = 100) -> dict:
    p = purpose_of(m)
    rng = np.random.default_rng(11)
    out = {"coef_tails": coef_stats(np.stack([m.sample_identity(rng, purpose=p) for _ in range(2000)])),
           "coef_normal_trunc": coef_stats(np.stack([m.sample_identity(rng, tails=False, purpose=p)
                                                     for _ in range(2000)]))}
    s = m.frame.mm_per_unit
    Z = np.stack([m.sample_identity(rng, purpose=p) for _ in range(n_id)])
    V = np.stack([m.mesh(z) for z in Z])
    D_mm = pairwise_vertex_l2(V * s)
    D_mx = pairwise_vertex_l2(np.stack([maxabs(v) for v in V]))
    iu = np.triu_indices(n_id, 1)
    nn = np.where(np.eye(n_id, dtype=bool), np.inf, D_mx).min(1)
    out["identity"] = {"n": n_id, "dist_from_mean_mm_median": float(np.median(np.linalg.norm(V - m.mean, axis=2).mean(1) * s)),
                       "pair_mm_median": float(np.median(D_mm[iu])), "pair_mm_min": float(D_mm[iu].min()),
                       "pair_maxabs_median": float(np.median(D_mx[iu])),
                       "nn_over_median_maxabs": float(np.median(nn) / np.median(D_mx[iu]))}
    med = float(np.median(D_mx[iu]))
    sh, sh_mm = [], []
    for z in Z[:50]:
        e = m.sample_expression(rng, purpose=p)
        V0, V1 = m.mesh(z), m.mesh(z, e)
        sh.append(shift(V1, V0))
        sh_mm.append(float(np.linalg.norm(V1 - V0, axis=1).mean() * s))
    out["expression"] = {"shift_maxabs_median": float(np.median(sh)), "shift_maxabs_mean": float(np.mean(sh)),
                         "shift_mm_median": float(np.median(sh_mm)), "ict_reference": TARGET_SHIFT,
                         "kind": m.expr.kind, "scale": m.expr.scale}
    amp, rad, frac = [], [], []
    for z in Z[:50]:
        V0 = m.mesh(z)
        V1, prm = m.local_rbf_deform(V0, rng, purpose=p, return_params=True)
        d = np.linalg.norm(V1 - V0, axis=1) * s
        amp.append(float(d.max()))
        rad.append(prm[0]["radius_mm"])
        frac.append(float((d > 0.01).mean()))
    out["rbf"] = {"max_disp_mm_min": float(min(amp)), "max_disp_mm_max": float(max(amp)),
                  "radius_mm_range": [float(min(rad)), float(max(rad))],
                  "frac_vertices_moved_median": float(np.median(frac)),
                  "in_spec": bool(0.99 <= min(amp) and max(amp) <= 3.01)}
    pairs = {}
    for ds in (0.2, 0.35, 0.5):
        r = []
        for z in Z[:50]:
            Va, Vb, _, _ = m.perturbation_pair(z, ds, rng, purpose=p)
            r.append(float(np.linalg.norm(maxabs(Va) - maxabs(Vb), axis=1).mean()) / med)
        pairs[str(ds)] = {"gt_over_median_pair_median": float(np.median(r)), "max": float(max(r))}
    out["perturbation_pairs"] = pairs
    return out


# ----------------------------------------------------------------------------- determinismo, ruoli, JSON

def draw(m, seed: int) -> np.ndarray:
    p = purpose_of(m)
    rng = np.random.default_rng(seed)
    z = m.sample_identity(rng, purpose=p)
    e = m.sample_expression(rng, purpose=p)
    V = m.mesh(z, e)
    W = m.local_rbf_deform(m.mesh(z), rng, purpose=p)
    Va, Vb, zb, ds = m.perturbation_pair(z, None, rng, purpose=p)
    return np.concatenate([z, e, V.ravel(), W.ravel(), Va.ravel(), Vb.ravel(), zb, [ds]])


def determinism(m) -> dict:
    a, b, c = draw(m, 7), draw(m, 7), draw(m, 8)
    return {"same_seed_identical": bool(np.array_equal(a, b)), "other_seed_differs": bool(not np.array_equal(a, c))}


def role_guard(m) -> dict:
    rng = np.random.default_rng(0)
    try:
        m.sample_identity(rng)               # purpose di default: "train"
        raised = False
    except RoleError:
        raised = True
    ok = raised == (m.role != "train")
    try:
        m.sample_expression(rng, purpose="eval")
        m.perturbation_pair(np.zeros(m.n_id), 0.3, rng, purpose="eval")
    except RoleError:
        ok = False
    return {"train_sampler_raises": raised, "guard_ok": ok}


def json_hook(m) -> dict:
    th = np.deg2rad(30.0)
    R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "canonical_transforms.json"
        path.write_text(json.dumps({"models": {m.name: {"R": R.tolist(), "t": [1.0, 2.0, 3.0], "s": 2.5}}}))
        got = m.canonical_transform(m.mean, json_path=path)
        want = 2.5 * m.mean @ R.T + np.array([1.0, 2.0, 3.0])
        src = m.canonical_params(path)["source"]
        other = Path(d) / "altro.json"     # un altro file: la cache del JSON va per (mtime, dimensione)
        other.write_text(json.dumps({"domains": {"altro_modello": {"s": 1.0}}}))
        fb = m.canonical_params(other)["source"]
    return {"json_applied": bool(np.allclose(got, want) and src.startswith("json:")),
            "missing_entry_falls_back": fb.startswith("fallback")}


# ----------------------------------------------------------------------------- taratura

def calibrate(m) -> dict:
    """La regola di gen_gnm_shard.py --calibrate sulla base pca del modello."""
    k, pool = m.n_id, m.expr.pool
    grid = {}
    for s in GRID:
        vals = []
        for j in range(N_CALIB):
            rng = np.random.default_rng([CALIB_SEED, j])
            z = rng.normal(size=k)
            e = np.zeros(m.n_expr)
            e[pool] = rng.normal(0.0, s, size=len(pool))
            for i, v in m.expr.fixed.items():
                e[i] = s * v
            vals.append(shift(m.mesh(z, e), m.mesh(z)))
        grid[s] = float(np.median(vals))
    best = min(grid, key=lambda s: abs(grid[s] - TARGET_SHIFT))
    return {"grid": grid, "target": TARGET_SHIFT, "sigma": best, "n_identities": N_CALIB, "current": EXPR_SCALE.get(m.name)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    names = [n for n in a.models.split(",") if n]
    report = {}
    for name in names:
        t0 = time.time()
        m = load_model(name)
        t_load = time.time() - t0
        if a.calibrate:
            if m.expr.kind == "pca":
                report[name] = calibrate(m)
                print(name, json.dumps(report[name]), flush=True)
            continue
        r = {"describe": m.describe(), "load_seconds": round(t_load, 1)}
        r["checks"] = model_checks(m)
        r["stats"] = sampling_stats(m)
        r["determinism"] = determinism(m)
        r["roles"] = role_guard(m)
        r["json_hook"] = json_hook(m)
        if a.out_dir is not None:
            render_grid(m, a.out_dir / "renders" / f"{name}.png")
        r["seconds"] = round(time.time() - t0, 1)
        report[name] = r
        print(f"[mm-selftest] {name}: {json.dumps({k: r[k] for k in ('checks', 'determinism', 'roles', 'json_hook')})}",
              flush=True)
        print(f"[mm-selftest] {name} stats: {json.dumps(r['stats'])}", flush=True)
    if not a.calibrate and {"facescape", "facescape50"} <= set(names):
        # stessa popolazione, due troncamenti: le medie neutre devono quasi coincidere
        d = np.linalg.norm(load_model("facescape").mean - load_model("facescape50").mean, axis=1)
        report["facescape50_vs_300_mean_mm"] = {"mean": float(d.mean()), "max": float(d.max())}
        print(f"[mm-selftest] media neutra facescape50 contro facescape: {report['facescape50_vs_300_mean_mm']}")
    if a.out_dir is not None:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        fn = "calibration.json" if a.calibrate else "selftest.json"
        (a.out_dir / fn).write_text(json.dumps(report, indent=1, default=str) + "\n")
        print(f"[mm-selftest] scritto {a.out_dir / fn}")
    if not a.calibrate:
        bad = [n for n, r in report.items() if n in names
               and not (r["checks"]["patch_clean"] and r["checks"]["mesh_zero_is_mean"] and r["checks"]["frame_ok"]
                       and r["checks"]["expr_additive_ok"] and r["determinism"]["same_seed_identical"]
                       and r["determinism"]["other_seed_differs"] and r["roles"]["guard_ok"]
                       and r["json_hook"]["json_applied"] and r["json_hook"]["missing_entry_falls_back"]
                       and r["stats"]["rbf"]["in_spec"])]
        print("[mm-selftest] " + ("OK" if not bad else f"FALLITI: {bad}"))
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
