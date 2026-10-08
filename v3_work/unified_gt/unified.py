#!/usr/bin/env python3
"""Passi 3-4a: regione comune, trasformazioni canoniche, media globale.

    v3_work/unified_gt/run.sh v3_work/unified_gt/unified.py      (dopo correspond.py)

Regione unificata: i vertici della regione FLAME (flame_region.py) coperti dalla mappa di TUTTI gli
8 domini (correspond.py: non sul bordo del template, a meno di 2 mm), poi la sola componente
connessa piu' grande. Una sola regione per tutti: una distanza fra identita' di domini diversi ha
senso solo sullo stesso supporto. Il dominio che la limita di piu' e' BFM (il crop p23470 non
arriva alle tempie e al bordo alto della fronte): misurato e scritto nel json.

Trasformazioni canoniche (``datasets/UNIFIED_GT/canonical_transforms.json``, la legge un altro
agente): per ogni dominio la similarita' ai minimi quadrati, pesata per l'area FLAME dei vertici,
dai punti della regione unificata sul template medio, nel frame e nelle unita' dei DATI del
dominio, ai vertici della media FLAME in mm. ``x_canon = s * R @ x + t``: +y in alto, +z fuori dal
volto, mm, origine di FLAME. ``flip_faces``: dopo la trasformazione (rotazione propria, scala
positiva) le normali dei triangoli del dominio puntano dentro, quindi vanno invertiti (F[:, ::-1])
per averle uscenti.

Media globale mu: Procrustes generalizzato (similarita', pesi d'area) delle medie dei domini di
TRAINING (flame, bfm, ict, gnm, facescape; non i domini di test, che restano fuori da ogni scelta)
portate sulla regione, con pesi uguali per dominio; a ogni giro mu e' riportata sulla media FLAME
con una similarita' (frame e scala FLAME, mm). I pesi d'area w sono le aree baricentriche dei
vertici su mu, ricalcolate a ogni giro.

Uscite: ``datasets/UNIFIED_GT/unified_space.npz`` (tutto quello che serve per calcolare s_i: indici
e baricentriche per dominio, mu, w, triangoli) e ``aau/runs/evidence/e8/unified_region.png``.
"""

from __future__ import annotations

import json

import numpy as np

import ugt as C
import domains
import render as rd

APPLIES_TO = {
    "flame": "mesh FLAME 2020 (v_template + shapedirs @ betas, metri), p.es. v2_work/genflame e datasets/FLAME",
    "bfm": "V delle npz BFM REMESH (datasets/REMESH/npz_data_topo_500*/id0000-0499_GTready_*.npz), micrometri; "
           "tutte le topologie della stessa identita' stanno nello stesso frame",
    "ict": "V delle npz ICT (datasets/ICT/*, datasets/ICT_SCALE/shards, ICT_ZS), unita' ICT (~cm)",
    "gnm": "V delle npz GNM (datasets/GNM_DISTILL/shards, GNM_ZS), metri; anche la testa intera di gnm_head.npz",
    "facescape": "uscita del bilineare FaceScape v1.6 (gen_full: core . id . exp, topologia a 26.278 vertici)",
    "hifi3d": "V delle npz HIFI3D (datasets/HIFI3D/identities, topo, eval_view), unita' del .mat",
    "faceverse": "V delle npz FaceVerse (datasets/FACEVERSE_ZS/*), frame nativo di faceverse_simple_v2.npy",
    "multiface": "SOLO il template (media dei neutri, frame del soggetto 002421669): le mesh tracked di "
                 "datasets/Multiface/prep hanno la posa della testa di ogni frame, serve un allineamento rigido per mesh",
}


def load_corr(name: str) -> dict:
    with np.load(C.CORR_DIR / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def gpa(means: dict, w0: np.ndarray, F: np.ndarray, ref: np.ndarray, n_iter: int = 50) -> tuple:
    mu, w = ref.copy(), w0.copy()
    for it in range(n_iter):
        aligned = {d: C.apply_sim(M, *C.umeyama(M, mu, w)) for d, M in means.items()}
        new = np.mean(list(aligned.values()), axis=0)
        new = C.apply_sim(new, *C.umeyama(new, ref, w))
        w = C.vertex_areas(new, F)
        delta = float(np.abs(new - mu).max())
        mu = new
        if delta < 1e-6:
            break
    return mu, w, it + 1


def main() -> None:
    reg = np.load(C.DATA_ROOT / "flame_region.npz")
    vidx0, Fr0 = reg["vidx"], reg["F"]
    flame = domains.flame()
    V_flame = flame["V"] * 1000.0
    corr = {d: load_corr(d) for d in domains.DOMAINS}
    cov = np.stack([corr[d]["covered"] for d in domains.DOMAINS])
    keep = cov.all(0)
    Fk = C.largest_component(Fr0[keep[Fr0].all(1)])
    ridx = np.unique(Fk)                                  # indici nella regione FLAME (1671)
    remap = -np.ones(len(vidx0), dtype=np.int64)
    remap[ridx] = np.arange(len(ridx))
    F = remap[Fk]
    P_flame = V_flame[vidx0[ridx]]
    w_flame = C.vertex_areas(P_flame, F)
    A0 = C.face_areas(V_flame[vidx0], Fr0).sum()
    # variante senza il vincolo di BFM (il dominio che limita di piu'): solo misurata e salvata, per
    # quando BFM 3DDFA sara' sostituito (PLAN_MASSIVE, sezione 2); la GT usa la regione di tutti
    keep_nb = np.stack([corr[d]["covered"] for d in domains.DOMAINS if d != "bfm"]).all(0)
    F_nb = C.largest_component(Fr0[keep_nb[Fr0].all(1)])
    ridx_nobfm = np.unique(F_nb)
    excluded_by = {d: int((~corr[d]["covered"]).sum()) for d in domains.DOMAINS}
    only_by = {d: int(((~corr[d]["covered"]) & (cov.sum(0) == len(domains.DOMAINS) - 1)).sum())
               for d in domains.DOMAINS}

    # medie dei domini sulla regione (frame dei dati) e trasformazioni canoniche
    means, canon = {}, {}
    for d in domains.DOMAINS:
        tpl = domains.template(d)
        vidx, bary = corr[d]["vidx"][ridx], corr[d]["bary"][ridx]
        M = C.bary_interp(tpl["V"], vidx, bary)
        means[d] = M
        s, R, t = C.umeyama(M, P_flame, w_flame)
        res = np.sqrt(((C.apply_sim(M, s, R, t) - P_flame) ** 2).sum(1))
        info = json.loads((C.CORR_DIR / f"{d}.json").read_text())
        canon[d] = {
            "s": s, "R": R.tolist(), "t": t.tolist(), "det_R": float(np.linalg.det(R)),
            "flip_faces": bool(info["canonical"]["flip_faces"]),
            "units_in": info.get("units", ""), "frame_in": info.get("frame", ""),
            "applies_to": APPLIES_TO[d],
            "residual_to_flame_mean_mm": {"rms_area_weighted": float(np.sqrt(np.average(res ** 2, weights=w_flame))),
                                          "max": float(res.max())},
        }
        if d != "flame":
            L = info["landmarks"]
            canon[d]["correspondence_checks"] = {
                "landmark_residual_after_nicp_median_mm": L["residual_after_nicp_mm"]["median"],
                "landmark_held_out_median_mm": L["held_out_after_nicp_mm"]["median"],
                "chamfer_symmetric_mean_mm": info["chamfer_mm"]["symmetric_mean"],
            }
    # FLAME: esattamente m -> mm (la media FLAME e' il riferimento)
    canon["flame"].update({"s": 1000.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]})

    mu, w, n_it = gpa({d: means[d] for d in domains.TRAIN_DOMAINS}, w_flame, F, P_flame)
    A = float(w.sum())
    dom_aligned = {}
    for d in domains.DOMAINS:
        T = C.umeyama(means[d], mu, w)
        dom_aligned[d] = C.apply_sim(means[d], *T)

    arrays = {"ridx": ridx, "flame_vidx": vidx0[ridx], "F": F, "mu": mu, "w": w, "area_total": A,
              "ridx_nobfm": ridx_nobfm,
              "domains": np.array(domains.DOMAINS), "train_domains": np.array(domains.TRAIN_DOMAINS)}
    for d in domains.DOMAINS:
        arrays[f"vidx_{d}"] = corr[d]["vidx"][ridx]
        arrays[f"bary_{d}"] = corr[d]["bary"][ridx]
        arrays[f"mean_{d}"] = dom_aligned[d]
    C.save_npz(C.DATA_ROOT / "unified_space.npz", **arrays)

    region_info = {
        "n_region_flame": int(len(vidx0)), "n_unified": int(len(ridx)), "n_faces": int(len(F)),
        "area_flame_region_mm2": float(A0), "area_unified_on_flame_mm2": float(w_flame.sum()),
        "area_unified_on_mu_mm2": A, "uncovered_by_domain": excluded_by,
        "uncovered_only_by_domain": only_by, "gpa_iterations": n_it,
        "dropped_by_component": int(keep.sum() - len(ridx)),
        "variant_without_bfm": {"n_vertices": int(len(ridx_nobfm)),
                                "area_on_flame_mm2": float(C.face_areas(V_flame[vidx0], F_nb).sum())},
    }
    out = {
        "description": "x_canon = s * R @ x + t (R 3x3 righe per riga, x vettore colonna nel frame e nelle "
                       "unita' dei dati del dominio). Frame canonico = FLAME 2020: +y in alto, +z fuori dal "
                       "volto, millimetri, origine di FLAME. Similarita' ai minimi quadrati (pesi = area FLAME "
                       "per vertice) dalla regione unificata del template medio del dominio alla media FLAME. "
                       "flip_faces: invertire il verso dei triangoli (F[:, ::-1]) per avere normali uscenti.",
        "reference": f"media FLAME 2020 generic_model.pkl x 1000, regione unificata di {len(ridx)} vertici "
                     f"({w_flame.sum():.0f} mm^2 sulla media FLAME)",
        "region": region_info,
        "domains": canon,
        "notes": [
            "Le similarita' portano ogni template medio sulla media FLAME, scala compresa: un dato trasformato "
            "ha la scala relativa al SUO template (un volto BFM medio diventa grande quanto il volto FLAME medio).",
            "La scala include il cambio di unita' (BFM micrometri -> mm: s ~ 1e-3; GNM metri -> mm: s ~ 1e3).",
            "Dati normalizzati per mesh (maxabs, area unitaria) NON sono nel frame dei dati: la trasformazione "
            "vale per le coordinate grezze delle npz.",
            "Multiface: vale solo per il template; ogni mesh tracked ha la sua posa.",
        ],
        "source": "v3_work/unified_gt/unified.py; corrispondenze in datasets/UNIFIED_GT/corr/<dominio>.json",
    }
    C.save_json(C.DATA_ROOT / "canonical_transforms.json", out)
    figure(V_flame, flame["F_render"], vidx0, ridx, corr)
    print(region_info)
    for d in domains.DOMAINS:
        c = canon[d]
        print(f"[unified] {d}: s={c['s']:.6g} flip={c['flip_faces']} residuo verso la media FLAME "
              f"{c['residual_to_flame_mean_mm']['rms_area_weighted']:.2f} mm (rms)")


def figure(V, F, vidx0, ridx, corr) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surf = C.Surface(V, F)
    in_unified = np.zeros(len(V), dtype=bool)
    in_unified[vidx0[ridx]] = True
    in_r0 = np.zeros(len(V), dtype=bool)
    in_r0[vidx0] = True
    lost = {d: np.zeros(len(V), dtype=bool) for d in domains.DOMAINS}
    for d in domains.DOMAINS:
        lost[d][vidx0[~corr[d]["covered"]]] = True
    colors = {"bfm": (1.0, 0.2, 0.1), "faceverse": (1.0, 0.8, 0.0), "gnm": (0.6, 0.2, 0.9),
              "facescape": (0.1, 0.8, 0.3), "ict": (0.0, 0.8, 0.8), "hifi3d": (1.0, 0.4, 0.8),
              "multiface": (0.5, 0.5, 0.5), "flame": (0, 0, 0)}
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, (title, R) in zip(axs, (("frontale", np.eye(3)), ("3/4", C.rot_y(-50)))):
        view = rd.framing(V, R, 700)
        o = rd.render(surf, view)
        img = o["img"].astype(float) / 255
        tri = np.maximum(o["tri"], 0)
        hit = o["tri"] >= 0
        m = hit & in_unified[F[tri]].all(-1)
        img[m] = 0.55 * img[m] + 0.45 * np.array([0.2, 0.5, 1.0])
        for d, col in colors.items():
            md = hit & in_r0[F[tri]].all(-1) & lost[d][F[tri]].any(-1)
            img[md] = 0.3 * img[md] + 0.7 * np.array(col)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=f"non coperto da {d}")
               for d, c in colors.items() if d != "flame"]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.suptitle(f"Regione unificata: {len(ridx)} di {len(vidx0)} vertici della regione FLAME (blu)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(C.EVID_DIR / "unified_region.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
