#!/usr/bin/env python3
"""Passi 2-3: corrispondenza FLAME -> template di ogni dominio e trasformazioni canoniche.

    v3_work/unified_gt/run.sh v3_work/unified_gt/correspond.py --domains bfm,ict,...
    (correspond.sbatch; prima flame_region.py)

Per ogni dominio, una volta sola:
  1. landmark (render.py): FLAME e template con la stessa procedura -- orientamento assiale
     cercato col detector, render frontale, FaceMesh, retroproiezione; poi il template portato nel
     frame FLAME con la similarita' dei landmark e RIRENDERIZZATO con la stessa camera di FLAME
     (stessa posa, stessa scala nell'immagine), e i landmark si rifanno li'. Validi: punti non sul
     contorno (FACE_OVAL), raggio non radente (|cos| > 0.3), su triangoli della regione FLAME / della
     pelle del template; si usano quelli validi in entrambi;
  2. similarita' robusta sui landmark (Umeyama, 3 giri scartando i residui > 3 volte la mediana);
  3. NICP della regione FLAME sul template (nicp.py), guidato dai landmark;
  4. mappa baricentrica: punto piu' vicino sul template di ogni vertice FLAME deformato. Coperto =
     non sul bordo del template e a meno di ``COVER_MM`` dalla superficie;
  5. trasformazione canonica: similarita' (pesata per area FLAME) dai punti mappati del template,
     nel frame dei dati, ai vertici della media FLAME in mm, sui vertici coperti.

Controlli scritti in ``corr/<dominio>.json`` e nelle figure ``aau/runs/evidence/e8/corr_<dominio>.png``:
residuo sui landmark in mm (prima e dopo NICP; e su un 20% di landmark TENUTI FUORI da un secondo
NICP: non e' un residuo d'adattamento), Chamfer fra regione mappata e template, distorsione dei
triangoli, copertura.

Le unita' in mm sono quelle della media FLAME: il template e' portato alla scala FLAME dalla
similarita', quindi un residuo di 1 mm e' 1 mm su un volto di dimensioni FLAME.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import ugt as C
import domains
import nicp as NI
import render as rd

COVER_MM = 2.0
COS_MIN = 0.3
HOLDOUT_FRACTION = 0.2
RES = 1024
PASSES = 4


def tri_lookup(F_all: np.ndarray, F_sub: np.ndarray) -> np.ndarray:
    """Per ogni triangolo di F_all, l'indice in F_sub dello stesso triangolo (a meno dell'ordine), o -1."""
    key = {tuple(sorted(t)): i for i, t in enumerate(F_sub.tolist())}
    return np.array([key.get(tuple(sorted(t)), -1) for t in F_all.tolist()], dtype=np.int64)


def flame_frame():
    """FLAME in mm, regione, vista frontale di riferimento e landmark FLAME."""
    tpl = domains.flame()
    V = tpl["V"] * 1000.0
    reg = np.load(C.DATA_ROOT / "flame_region.npz")
    vidx, Fr = reg["vidx"], reg["F"]
    Fg = vidx[Fr]                                       # triangoli della regione, indici FLAME
    surf = C.Surface(V, tpl["F_render"])
    view = rd.framing(V, np.eye(3), RES)
    det = rd.detect(surf, view)
    look = tri_lookup(tpl["F_render"], Fg)
    reg_tri = np.where(det["hit"], look[np.maximum(det["tri"], 0)], -1)
    # baricentriche riportate all'ordine dei vertici del triangolo di regione
    bary = np.zeros_like(det["bary"])
    for l in np.flatnonzero(reg_tri >= 0):
        src = tpl["F_render"][det["tri"][l]]
        dst = Fg[reg_tri[l]]
        for k in range(3):
            bary[l, k] = det["bary"][l][list(src).index(dst[k])]
    valid = (reg_tri >= 0) & (det["cos"] > COS_MIN)
    return {"V": V, "F_render": tpl["F_render"], "vidx": vidx, "Fr": Fr, "view": view, "det": det,
            "lm_tri": reg_tri, "lm_bary": bary, "lm_valid": valid, "surf": surf}


def template_landmarks(tpl: dict, fl: dict, oval: np.ndarray) -> dict:
    """Landmark del template con la procedura comune; ritorna anche la similarita' verso FLAME."""
    surf = C.Surface(tpl["V"], tpl["F_render"])
    in_target = tri_lookup(tpl["F_render"], tpl["F_target"]) >= 0
    R0, search_log = rd.search_orientation(surf)
    det1 = rd.detect(surf, rd.framing(tpl["V"], R0, RES))
    fl_pts = C.bary_interp(fl["V"][fl["vidx"]], fl["Fr"][np.maximum(fl["lm_tri"], 0)], fl["lm_bary"])

    def usable(det):
        ok = det["hit"] & (det["cos"] > COS_MIN)
        ok &= in_target[np.maximum(det["tri"], 0)]
        ok &= fl["lm_valid"]
        ok[oval] = False
        return ok

    def robust_sim(P, Q):
        keep = np.ones(len(P), dtype=bool)
        for _ in range(3):
            T = C.umeyama(P[keep], Q[keep])
            r = np.linalg.norm(C.apply_sim(P, *T) - Q, axis=1)
            keep = r < 3.0 * np.median(r[keep])
        return T, keep, r

    ok1 = usable(det1)
    T1, _, _ = robust_sim(det1["point"][ok1], fl_pts[ok1])
    # passaggi successivi: template nel frame FLAME, stessa camera di FLAME, finche' la correzione
    # della similarita' e' sotto 1 grado e 1% di scala (al massimo PASSES)
    passes = []
    for k in range(PASSES):
        V1 = C.apply_sim(tpl["V"], *T1)
        det2 = rd.detect(C.Surface(V1, tpl["F_render"]), fl["view"])
        ok2 = usable(det2)
        T2, keep, r = robust_sim(det2["point"][ok2], fl_pts[ok2])
        ang = float(np.degrees(np.arccos(np.clip((np.trace(T2[1]) - 1) / 2, -1, 1))))
        passes.append({"rotation_deg": ang, "scale": float(T2[0]), "n_valid": int(ok2.sum())})
        if (ang < 1.0 and abs(T2[0] - 1.0) < 0.01) or k == PASSES - 1:
            break                                   # T1 = similarita' del render di det2, T2 = correzione
        T1 = C.compose_sim(T1, T2)
    use = np.flatnonzero(ok2)[keep]
    return {"R0": R0, "search_log": search_log, "T1": T1, "T2": T2, "det1": det1, "det2": det2,
            "use": use, "n_detected_valid": int(ok2.sum()), "n_outliers": int((~keep).sum()),
            "points_flame_T1": det2["point"], "fl_pts": fl_pts, "passes": passes,
            "sim_residual_mm": r}


def exterior_target(V: np.ndarray, F_target: np.ndarray, F_render: np.ndarray) -> tuple[np.ndarray, int]:
    """Triangoli del bersaglio con tre vertici esterni (stessa regola della regione FLAME, nel frame
    FLAME): via cavita' della bocca, orbite, narici profonde, che altrimenti attirerebbero i punti
    piu' vicini. Normali orientate come la maggioranza verso +z; occlusori = tutta la mesh."""
    used = np.unique(F_target)
    n = C.vertex_normals(V, F_target)
    if (C.face_normals(V, F_target, unit=False)[:, 2]).sum() < 0:
        n = -n
    vis = np.zeros(len(V), dtype=bool)
    vis[used] = C.exterior(V, F_render, used, normals=n)
    F = C.largest_component(F_target[vis[F_target].all(1)])
    return F, int(len(used) - len(np.unique(F)))


def run_nicp(fl: dict, V_t: np.ndarray, F_t: np.ndarray, lm_idx: np.ndarray, lm_tgt: np.ndarray) -> dict:
    """NICP in coordinate normalizzate; ritorna Y (mm) e il bersaglio normalizzato."""
    Vs = fl["V"][fl["vidx"]]
    c = Vs.mean(0)
    S = float(np.abs(Vs - c).max())
    norm = lambda X: (X - c) / S  # noqa: E731
    target = NI.Target(norm(V_t), F_t)
    flipped = target.orient_like(norm(Vs), C.vertex_normals(Vs, fl["Fr"]))
    out = NI.nicp(norm(Vs), fl["Fr"], target, fl["lm_tri"][lm_idx], fl["lm_bary"][lm_idx], norm(lm_tgt))
    out["Y"] = out["Y"] * S + c
    out["target_normals_flipped"] = flipped
    out["S"], out["c"] = S, c
    out["target"] = target
    return out


def correspond(name: str, fl: dict, oval: np.ndarray) -> dict:
    t0 = time.time()
    tpl = domains.template(name)
    lm = template_landmarks(tpl, fl, oval)
    # template nel frame FLAME: T2 dopo T1 (entrambe similarita')
    T = C.compose_sim(lm["T1"], lm["T2"])
    V_t = C.apply_sim(tpl["V"], *T)
    F_t, n_interior = exterior_target(V_t, tpl["F_target"], tpl["F_render"])
    use = lm["use"]
    lm_tgt = C.apply_sim(lm["points_flame_T1"], *lm["T2"])          # landmark del template, frame FLAME
    Vs = fl["V"][fl["vidx"]]
    lm_src0 = lm["fl_pts"]                                          # landmark FLAME (media, mm)

    out = run_nicp(fl, V_t, F_t, use, lm_tgt[use])
    Y = out["Y"]
    lm_def = C.bary_interp(Y, fl["Fr"][np.maximum(fl["lm_tri"], 0)], fl["lm_bary"])
    res_before = np.linalg.norm(lm_src0[use] - lm_tgt[use], axis=1)
    res_after = np.linalg.norm(lm_def[use] - lm_tgt[use], axis=1)

    # landmark tenuti fuori: secondo NICP senza il 20% dei landmark (seme fisso), residuo su quelli
    rng = np.random.default_rng(1234)
    held = np.zeros(len(use), dtype=bool)
    held[rng.choice(len(use), int(round(HOLDOUT_FRACTION * len(use))), replace=False)] = True
    out_h = run_nicp(fl, V_t, F_t, use[~held], lm_tgt[use[~held]])
    lm_def_h = C.bary_interp(out_h["Y"], fl["Fr"][np.maximum(fl["lm_tri"], 0)], fl["lm_bary"])
    res_held = np.linalg.norm(lm_def_h[use[held]] - lm_tgt[use[held]], axis=1)
    res_held_before = res_before[held]

    # mappa baricentrica e copertura
    surf_t = C.Surface(V_t, F_t)
    cl = surf_t.closest(Y)
    bnd = NI.on_boundary(F_t, cl["tri"], cl["bary"], C.boundary_edge_mask(F_t), C.boundary_vertices(F_t, len(V_t)))
    covered = ~bnd & (cl["dist"] < COVER_MM)
    vidx_map = F_t[cl["tri"]]
    bary_map = cl["bary"]

    # Chamfer: regione deformata -> template (vertici coperti) e template -> regione deformata
    # (vertici del template nell'impronta della regione: punto piu' vicino non sul bordo della regione)
    surf_y = C.Surface(Y, fl["Fr"])
    used_t = np.unique(F_t)
    ct = surf_y.closest(V_t[used_t])
    bnd_y = NI.on_boundary(fl["Fr"], ct["tri"], ct["bary"], C.boundary_edge_mask(fl["Fr"]),
                           C.boundary_vertices(fl["Fr"], len(Y)))
    foot = ~bnd_y
    d_st = cl["dist"][covered]
    d_ts = ct["dist"][foot]

    # distorsione dei triangoli FLAME
    a0 = C.face_areas(Vs, fl["Fr"])
    a1 = C.face_areas(Y, fl["Fr"])
    n0 = C.face_normals(Vs, fl["Fr"])
    n1 = C.face_normals(Y, fl["Fr"])
    log_area = np.log(np.maximum(a1, 1e-12) / a0)
    flips = int((np.einsum("nd,nd->n", n0, n1) < 0).sum())

    # trasformazione canonica: punti mappati del template nel frame dei DATI -> media FLAME in mm
    w = C.vertex_areas(Vs, fl["Fr"])
    Q_data = C.bary_interp(tpl["V"], vidx_map, bary_map)
    canon = C.umeyama(Q_data[covered], Vs[covered], w[covered])
    canon_rms = float(np.sqrt(np.average(((C.apply_sim(Q_data[covered], *canon) - Vs[covered]) ** 2).sum(1),
                                         weights=w[covered])))
    # verso dei triangoli dopo la trasformazione: normale media (pesata) dei triangoli colpiti
    Vc = C.apply_sim(tpl["V"], *canon)
    nt = C.face_normals(Vc, F_t, unit=False)[cl["tri"][covered]]
    outward = float(nt.sum(0)[2] / np.linalg.norm(nt.sum(0)))
    flip_faces = outward < 0

    stats = lambda x: {"mean": float(np.mean(x)), "median": float(np.median(x)),  # noqa: E731
                       "p95": float(np.percentile(x, 95)), "max": float(np.max(x)), "n": int(len(x))}
    info = {
        "domain": name, "units": tpl["units"], "frame": tpl["frame"], "note": tpl["note"],
        "n_template_vertices": int(len(tpl["V"])), "n_target_faces": int(len(F_t)),
        "n_target_vertices_dropped_interior": n_interior,
        "orientation_R0": lm["R0"].tolist(), "frontal_passes": lm["passes"],
        "landmarks": {"n_valid_both": lm["n_detected_valid"], "n_outliers_similarity": lm["n_outliers"],
                      "n_used": int(len(use)), "n_held_out": int(held.sum()),
                      "residual_after_similarity_mm": stats(res_before),
                      "residual_after_nicp_mm": stats(res_after),
                      "held_out_before_mm": stats(res_held_before),
                      "held_out_after_nicp_mm": stats(res_held)},
        "nicp_log": out["log"], "target_normals_flipped": bool(out["target_normals_flipped"]),
        "coverage": {"n_region": int(len(Y)), "n_covered": int(covered.sum()),
                     "n_on_target_boundary": int(bnd.sum()), "n_far": int((~bnd & ~covered).sum())},
        "chamfer_mm": {"region_to_template": stats(d_st), "template_to_region": stats(d_ts),
                       "symmetric_mean": float(0.5 * (d_st.mean() + d_ts.mean()))},
        "distortion": {"log_area_ratio_abs_median": float(np.median(np.abs(log_area))),
                       "log_area_ratio_p95": float(np.percentile(np.abs(log_area), 95)),
                       "flipped_triangles": flips},
        "canonical": {"s": canon[0], "R": canon[1].tolist(), "t": canon[2].tolist(),
                      "rms_mm": canon_rms, "mean_normal_z": outward, "flip_faces": bool(flip_faces)},
        "seconds": time.time() - t0,
    }
    arrays = {"vidx": vidx_map.astype(np.int64), "bary": bary_map, "covered": covered,
              "dist_mm": cl["dist"], "on_boundary": bnd, "Y_mm": Y,
              "lm_use": use, "lm_target_flame_mm": lm_tgt, "lm_template_data": C.apply_sim(
                  lm_tgt, *invert_sim(T)), "canon_s": canon[0], "canon_R": canon[1], "canon_t": canon[2],
              "T_landmark_s": T[0], "T_landmark_R": T[1], "T_landmark_t": T[2]}
    figure(name, tpl, fl, lm, V_t, F_t, Y, cl, covered, use, lm_tgt, lm_def, info)
    return {"info": info, "arrays": arrays}


def invert_sim(T: tuple) -> tuple:
    s, R, t = T
    return 1.0 / s, R.T, -(R.T @ t) / s


def flame_identity_map(fl: dict) -> dict:
    """Il dominio FLAME: mappa identita' (ogni vertice di regione su se stesso), trasformazione m -> mm."""
    n = len(fl["vidx"])
    vidx = np.repeat(fl["vidx"][:, None], 3, axis=1)
    bary = np.zeros((n, 3))
    bary[:, 0] = 1.0
    info = {"domain": "flame", "units": "metri", "frame": "FLAME: +y alto, +z naso",
            "coverage": {"n_region": n, "n_covered": n},
            "landmarks": {"n_valid_flame": int(fl["lm_valid"].sum())},
            "canonical": {"s": 1000.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0], "rms_mm": 0.0,
                          "flip_faces": False}}
    arrays = {"vidx": vidx, "bary": bary, "covered": np.ones(n, dtype=bool), "dist_mm": np.zeros(n),
              "canon_s": 1000.0, "canon_R": np.eye(3), "canon_t": np.zeros(3)}
    return {"info": info, "arrays": arrays}


def figure(name, tpl, fl, lm, V_t, F_t, Y, cl, covered, use, lm_tgt, lm_def, info) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    surf_r = C.Surface(V_t, tpl["F_render"])
    fig, axs = plt.subplots(1, 4, figsize=(24, 6.6))
    # 1: template, landmark del secondo passaggio (verdi usati, rossi scartati)
    det2 = lm["det2"]
    axs[0].imshow(det2["img"])
    allv = np.arange(len(det2["uv"]))
    rej = np.setdiff1d(allv, use)
    axs[0].scatter(det2["uv"][rej, 0], det2["uv"][rej, 1], s=3, c="red")
    axs[0].scatter(det2["uv"][use, 0], det2["uv"][use, 1], s=3, c="lime")
    axs[0].set_title(f"{name}: FaceMesh sul render frontale\nverdi = usati ({len(use)}), rossi = scartati")
    # 2: FLAME coi suoi landmark
    d = fl["det"]
    axs[1].imshow(d["img"])
    axs[1].scatter(d["uv"][fl["lm_valid"], 0], d["uv"][fl["lm_valid"], 1], s=3, c="lime")
    axs[1].set_title("FLAME (media): FaceMesh, punti validi sulla regione")
    # 3-4: template con la regione deformata in wireframe, colorata per distanza; frontale e 3/4
    E = np.unique(np.sort(np.concatenate([fl["Fr"][:, [0, 1]], fl["Fr"][:, [1, 2]], fl["Fr"][:, [2, 0]]]), axis=1), axis=0)
    for ax, (title, R) in zip(axs[2:], (("frontale", np.eye(3)), ("3/4 (yaw 50)", C.rot_y(-50)))):
        view = rd.framing(V_t, R, 900) if title != "frontale" else rd.View(np.eye(3), fl["view"].center, fl["view"].half, 900)
        out = rd.render(surf_r, view)
        ax.imshow(out["img"])
        P = view.project(Y)
        # visibilita' del punto medio di ogni spigolo: confronto con la profondita' del template
        mid = 0.5 * (Y[E[:, 0]] + Y[E[:, 1]])
        pm = view.project(mid)
        bp = rd.backproject(surf_r, view, pm)
        dm = view.to_view(mid)[:, 2]
        hit_depth = view.to_view(bp["point"])[:, 2]
        vis = ~bp["hit"] | (dm > hit_depth - 1.5)
        dist = cl["dist"]
        col = plt.cm.viridis(np.clip(0.5 * (dist[E[:, 0]] + dist[E[:, 1]]) / COVER_MM, 0, 1))
        uncovered = ~(covered[E[:, 0]] & covered[E[:, 1]])
        col[uncovered] = (1.0, 0.0, 0.0, 1.0)
        segs = np.stack([P[E[:, 0]], P[E[:, 1]]], axis=1)[vis]
        ax.add_collection(LineCollection(segs, colors=col[vis], linewidths=0.4))
        if title == "frontale":
            a, b = view.project(lm_def[use]), view.project(lm_tgt[use])
            ax.add_collection(LineCollection(np.stack([a, b], axis=1), colors="magenta", linewidths=1.0))
            ax.scatter(b[:, 0], b[:, 1], s=2, c="magenta")
        ax.set_xlim(0, 900)
        ax.set_ylim(900, 0)
        ax.set_title(f"regione FLAME deformata sul template ({title})\ncolore = distanza 0-{COVER_MM:g} mm, "
                     f"rosso = non coperto; magenta = landmark (residuo)")
    for ax in axs:
        ax.axis("off")
    L = info["landmarks"]
    fig.suptitle(f"{name}: landmark {L['n_used']} usati, residuo mediano {L['residual_after_similarity_mm']['median']:.2f}"
                 f" -> {L['residual_after_nicp_mm']['median']:.2f} mm (tenuti fuori: "
                 f"{L['held_out_after_nicp_mm']['median']:.2f} mm); Chamfer {info['chamfer_mm']['symmetric_mean']:.3f} mm; "
                 f"coperti {info['coverage']['n_covered']}/{info['coverage']['n_region']}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    C.EVID_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.EVID_DIR / f"corr_{name}.png", dpi=80)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--domains", default=",".join(domains.DOMAINS))
    a = p.parse_args()
    fl = flame_frame()
    oval = rd.face_oval_indices()
    print(f"[corr] FLAME: {int(fl['lm_valid'].sum())} landmark validi sulla regione", flush=True)
    for name in a.domains.split(","):
        if name == "flame":
            r = flame_identity_map(fl)
        else:
            r = correspond(name, fl, oval)
        C.save_npz(C.CORR_DIR / f"{name}.npz", **r["arrays"])
        C.save_json(C.CORR_DIR / f"{name}.json", r["info"])
        i = r["info"]
        if name != "flame":
            L = i["landmarks"]
            print(f"[corr] {name}: landmark {L['n_used']} (outlier {L['n_outliers_similarity']}), residuo mediano "
                  f"{L['residual_after_similarity_mm']['median']:.2f} -> {L['residual_after_nicp_mm']['median']:.2f} mm, "
                  f"tenuti fuori {L['held_out_before_mm']['median']:.2f} -> {L['held_out_after_nicp_mm']['median']:.2f} mm; "
                  f"Chamfer {i['chamfer_mm']['symmetric_mean']:.3f} mm; coperti {i['coverage']['n_covered']}/"
                  f"{i['coverage']['n_region']} (bordo {i['coverage']['n_on_target_boundary']}); "
                  f"canonica s={i['canonical']['s']:.5g} rms {i['canonical']['rms_mm']:.2f} mm, "
                  f"flip_faces={i['canonical']['flip_faces']}; {i['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
