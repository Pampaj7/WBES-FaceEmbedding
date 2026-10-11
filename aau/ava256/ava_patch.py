#!/usr/bin/env python3
"""Patch delle viste di Ava-256 (scelta 7 del README, modificata l'11 ottobre su decisione del PI PRIMA di qualsiasi
valutazione: README, sezione "Modifiche").

    v3_work/unified_gt/run.sh aau/ava256/ava_patch.py          (dopo ava_corr.py)

CONFERMATIVO: non valutare prima del protocollo confermativo (README). Solo dati e GT.

**Impronta in stile NoW, quella delle viste FaMoS** (``aau/famos/famos_test_view.py`` -> ``now_common.now_mask`` e
``now_prepare_meshes.crop_patch``): sfera di centro subnasale + 0.3 (ponte - subnasale) e raggio 1.4 (ex-ex + naso) / 2
dai landmark iBUG 36, 39, 42, 45, 33 del template; restano i triangoli coi tre vertici dentro. La sfera e'
equivariante per similarita': calcolata nel frame FLAME della corrispondenza da' gli stessi triangoli che nel frame
canonico NoW di FaMoS. Landmark: l'embedding iBUG di FLAME 2020 delle viste FaMoS (``famos_common.flame_lmk68``) portato
sul template con la mappa di ava_corr.py (baricentrico se il triangolo FLAME sta nella regione FLAME, altrimenti il
vertice di regione piu' vicino). Il contorno NON e' la maschera ``face`` di FLAME (la regione del fit B-FLAME). Controllo:
gli stessi landmark da FaceMesh sul render frontale del template (indici 33, 133, 362, 263, 2) e la sfera che ne esce.

**Buchi: bordo esterno, bocca e aperture palpebrali, niente bulbo oculare** (come le original di HIFI3D e FaceScape dev;
FaceVerse copre gli occhi). La topologia di Ava e' chiusa su occhi e bocca: l'apertura palpebrale e' chiusa da pochi
triangoli lunghi da margine a margine (la calotta, senza vertici interni), la bocca da triangoli lungo la rima. Per ogni
buco si tolgono i triangoli entro 20 mm (bocca 35 mm) dal centro dell'anello della regione FLAME deformata dal NICP attorno
a quel buco (la regione da cui viene la GT) col baricentro dentro l'anello proiettato sul piano frontale (x, y del frame
FLAME); si tiene la componente connessa piu' grande fra quelle che arrivano entro 10 mm dal centro dei landmark del buco
(iBUG 36-41, 42-47, 60-67). Provati e scartati prima di questa regola (anelli spuri, calotta non tolta): il punto piu'
vicino della regione sull'anello (il NICP stende le palpebre FLAME vicino alla calotta: 2-5 triangoli per occhio), i tre
vertici dentro l'anello (calotta quasi intatta), la regola di esternita' per la bocca (non isola la tasca), il taglio lungo
l'anello agganciato ai vertici di Ava (zigzag fra le labbra chiuse). I punti della GT che cadono su un triangolo tolto si
contano come residuo. Ogni altro buco (narici, frammenti) si chiude coi triangoli del template; i vertici a farfalla si
chiudono aggiungendo i triangoli dei buchi che li toccano. La patch deve finire con 4 anelli di bordo, altrimenti lo script
si ferma.

Uscite: ``datasets/AVA256/corr/patch.npz`` (``patch_tri`` sulle righe della topologia, ``patch_F``, centro e raggio, i
dischi tolti) e ``aau/ava256/patch_summary.json`` (conteggi, nessuna geometria); figura di controllo interna in
``datasets/AVA256/evidence/patch.png`` (render del volto medio: fuori da git).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from scipy.sparse.csgraph import connected_components

THIS_DIR = Path(__file__).resolve().parent
for _p in (THIS_DIR, THIS_DIR.parents[1] / "v3_work" / "unified_gt", THIS_DIR.parent / "famos", THIS_DIR.parent / "recon"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ava_common as ac  # noqa: E402
import correspond as CO  # noqa: E402
import domains  # noqa: E402
import famos_common as fc  # noqa: E402
import now_common  # noqa: E402
import render as rd  # noqa: E402
import ugt as C  # noqa: E402

TEMPLATE = ac.DATA_ROOT / "template.npz"
CORR_NPZ = ac.DATA_ROOT / "corr" / "ava256.npz"
SUMMARY = ac.SUMMARY_DIR / "patch_summary.json"
REGION = domains.DATASETS / "UNIFIED_GT" / "flame_region.npz"
HOLE_LMK = {"eye_r": tuple(range(36, 42)), "eye_l": tuple(range(42, 48)), "mouth": tuple(range(60, 68))}
HOLE_MAX_MM = 10.0                     # distanza massima fra il centro di un anello e il suo gruppo di landmark
HOLE_RADIUS_MM = {"eye_r": 20.0, "eye_l": 20.0, "mouth": 35.0}   # candidati: entro questa distanza dal centro dell'anello
FACEMESH_LMK7 = (33, 133, 362, 263, 2)  # ex_r, en_r, en_l, ex_l, subnasale (solo controllo della sfera)
EXPECTED_LOOPS = 4


# ---------------------------------------------------------------------------- topologia

def boundary_loops(F: np.ndarray) -> list[list[int]]:
    """Anelli di bordo (vertici in ordine) di una mesh orientata."""
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    _, inv, cnt = np.unique(np.sort(E, 1), axis=0, return_inverse=True, return_counts=True)
    nxt = {}
    for a, b in E[cnt[inv.ravel()] == 1]:
        nxt.setdefault(int(a), []).append(int(b))
    out, seen = [], set()
    for s0 in nxt:
        if s0 in seen:
            continue
        lp, v = [], s0
        while v not in seen:
            seen.add(v)
            lp.append(v)
            v = nxt[v][0]
        out.append(lp)
    return sorted(out, key=len, reverse=True)


def face_graph(F: np.ndarray) -> sps.coo_matrix:
    """Adiacenza fra triangoli che condividono uno spigolo."""
    m = len(F)
    E = np.sort(np.concatenate([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]]), 1)
    key = E[:, 0].astype(np.int64) * ac.N_VERTS + E[:, 1]
    tri = np.tile(np.arange(m), 3)
    o = np.argsort(key, kind="stable")
    same = key[o][1:] == key[o][:-1]
    a, b = tri[o][1:][same], tri[o][:-1][same]
    return sps.coo_matrix((np.ones(len(a)), (a, b)), shape=(m, m))


def components(F: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Etichetta della componente connessa (per spigolo) dei triangoli selezionati, -1 fuori."""
    idx = np.flatnonzero(sel)
    lab = -np.ones(len(F), dtype=np.int64)
    if len(idx):
        lab[idx] = connected_components(face_graph(F[idx]), directed=False)[1]
    return lab


def pick(F: np.ndarray, Tt: np.ndarray, sel: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """La componente connessa di ``sel`` piu' grande fra quelle che arrivano entro HOLE_MAX_MM da ``centre``."""
    lab = components(F, sel)
    best, size, dmin = -1, 0, np.inf
    for k in np.unique(lab[lab >= 0]):
        m = lab == k
        d = float(np.linalg.norm(Tt[np.unique(F[m])] - centre, axis=1).min())
        dmin = min(dmin, d)
        if d <= HOLE_MAX_MM and m.sum() > size:
            best, size = k, int(m.sum())
    if best < 0:
        raise SystemExit(f"nessun candidato entro {HOLE_MAX_MM} mm dal centro del buco ({dmin:.1f})")
    return lab == best


def hole_ring(F: np.ndarray, Tt: np.ndarray, cand_tri: np.ndarray, ring: np.ndarray, centre: np.ndarray,
              radius: float) -> tuple[np.ndarray, dict]:
    """Buco dentro un anello della regione FLAME deformata (punti in ordine, frame FLAME): triangoli entro ``radius``
    dal centro dell'anello col baricentro dentro l'anello proiettato sul piano frontale (x, y); la componente piu'
    grande vicino a ``centre`` (``pick``). Sono i triangoli che chiudono l'apertura da margine a margine (calotta
    oculare, chiusura delle labbra)."""
    from matplotlib.path import Path as Poly
    c = ring.mean(0)
    cent = Tt[F].mean(1)
    near = cand_tri & (np.linalg.norm(cent - c, axis=1) <= radius)
    inside = np.zeros(len(F), dtype=bool)
    idx = np.flatnonzero(near)
    inside[idx[Poly(ring[:, :2]).contains_points(cent[idx, :2])]] = True
    return pick(F, Tt, inside, centre), {"candidates": int(inside.sum())}


def pinch_vertices(Fp: np.ndarray) -> np.ndarray:
    """Vertici con piu' di due spigoli di bordo (bordo non manifold, "a farfalla")."""
    E = np.concatenate([Fp[:, [0, 1]], Fp[:, [1, 2]], Fp[:, [2, 0]]])
    _, inv, cnt = np.unique(np.sort(E, 1), axis=0, return_inverse=True, return_counts=True)
    b = E[cnt[inv.ravel()] == 1]
    deg = np.bincount(b.ravel(), minlength=ac.N_VERTS)
    return np.flatnonzero(deg > 2)


def fix_pinches(F: np.ndarray, patch: np.ndarray, holes: np.ndarray, rounds: int = 10) -> tuple[np.ndarray, np.ndarray, int]:
    """Aggiunge alla patch i triangoli dei buchi che toccano un vertice a farfalla, finche' non ce ne sono piu'."""
    added = 0
    for _ in range(rounds):
        pv = pinch_vertices(F[patch])
        if not len(pv):
            break
        add = holes & np.isin(F, pv).any(1)
        if not add.any():
            break
        patch, holes, added = patch | add, holes & ~add, added + int(add.sum())
    return patch, holes, added


def complement_fill(F: np.ndarray, patch: np.ndarray, holes: np.ndarray, neck_tri: np.ndarray) -> tuple[np.ndarray, int]:
    """Chiude i buchi spuri: le componenti dei triangoli FUORI dalla patch che non toccano il collo ne' i dischi tolti."""
    out = ~patch
    sub = np.flatnonzero(out)
    g = face_graph(F[sub])
    _, lab = connected_components(g, directed=False)
    pos = -np.ones(len(F), dtype=np.int64)
    pos[sub] = np.arange(len(sub))
    keep_out = set(lab[pos[neck_tri[out[neck_tri]]]].tolist()) | set(lab[pos[np.flatnonzero(holes)]].tolist())
    fill = np.zeros(len(F), dtype=bool)
    fill[sub[~np.isin(lab, list(keep_out))]] = True
    return patch | fill, int(fill.sum())


# ---------------------------------------------------------------------------- landmark

def transferred_landmarks(Tt: np.ndarray, z: dict) -> tuple[np.ndarray, list[str]]:
    """I 68 iBUG di FLAME 2020 sul template (frame FLAME, mm): baricentrici sulle immagini dei vertici della regione FLAME
    se il triangolo sta nella regione, altrimenti l'immagine del vertice di regione piu' vicino."""
    flame = domains.flame()
    Vf = flame["V"] * 1000.0
    Ff = flame["F_render"]
    fi, bc = fc.flame_lmk68()
    vidx_r = np.load(REGION)["vidx"]
    pos = -np.ones(len(Vf), dtype=np.int64)
    pos[vidx_r] = np.arange(len(vidx_r))
    img = C.bary_interp(Tt, z["vidx"], z["bary"])
    out, how = [], []
    for i in range(68):
        tri = Ff[fi[i]]
        if (pos[tri] >= 0).all():
            out.append((bc[i][:, None] * img[pos[tri]]).sum(0))
            how.append("baricentrico")
        else:
            pf = (bc[i][:, None] * Vf[tri]).sum(0)
            j = int(np.linalg.norm(Vf[vidx_r] - pf, axis=1).argmin())
            out.append(img[j])
            how.append(f"vicino {np.linalg.norm(Vf[vidx_r][j] - pf):.2f} mm")
    return np.asarray(out), how


def facemesh_check(Tt: np.ndarray, F: np.ndarray, lm7: np.ndarray, centre: np.ndarray, radius: float) -> dict:
    """Stessi 5 landmark della sfera da FaceMesh (render frontale del template) e la sfera che ne esce."""
    Vc, Fc, _ = C.compact(Tt, F)
    det = rd.detect(C.Surface(Vc, Fc), CO.flame_frame()["view"])
    if not det["hit"][list(FACEMESH_LMK7)].all():
        return {"ok": False}
    P = det["point"][list(FACEMESH_LMK7)]
    c2, r2 = now_common.now_mask(np.concatenate([P, lm7[5:]]))
    return {"ok": True, "landmark_distance_mm": [round(float(x), 2) for x in np.linalg.norm(P - lm7[:5], axis=1)],
            "centre_shift_mm": float(np.linalg.norm(c2 - centre)), "radius_ratio": float(r2 / radius)}


# ---------------------------------------------------------------------------- patch

def build() -> tuple[dict, dict]:
    with np.load(TEMPLATE) as t, np.load(CORR_NPZ) as zz:
        T, F = t["V"], t["F"].astype(np.int64)
        z = {k: zz[k] for k in ("vidx", "bary", "Y_mm", "T_landmark_s", "T_landmark_R", "T_landmark_t",
                                "vidx_unified", "bary_unified")}
    Tt = C.apply_sim(T, float(z["T_landmark_s"]), z["T_landmark_R"], z["T_landmark_t"])   # frame FLAME, mm
    n = len(T)
    lm, how = transferred_landmarks(Tt, z)
    lm7 = lm[list(now_common.LMK7_IBUG)]
    centre, radius = now_common.now_mask(lm7)
    inside = np.linalg.norm(Tt - centre, axis=1) <= radius
    sphere = np.zeros(len(F), dtype=bool)
    sphere[inside[F].all(1)] = True
    neck_tri = np.flatnonzero(C.boundary_edge_mask(F).any(1))

    # buchi: dentro gli anelli della regione FLAME deformata attorno a occhi e bocca
    Fr = np.load(REGION)["F"]
    Y = z["Y_mm"]
    loops = boundary_loops(Fr)
    tri_index = {tuple(sorted(t)): i for i, t in enumerate(F.tolist())}
    gt_tri = np.array([tri_index.get(tuple(sorted(t)), -1) for t in z["vidx_unified"].tolist()])
    is_gt = np.zeros(len(F), dtype=bool)
    is_gt[gt_tri[gt_tri >= 0]] = True
    holes, hole_info = np.zeros(len(F), dtype=bool), {}
    for name, ids in HOLE_LMK.items():
        target = lm[list(ids)].mean(0)
        d = [float(np.linalg.norm(Y[lp].mean(0) - target)) for lp in loops]
        k = int(np.argmin(d))
        if d[k] > HOLE_MAX_MM:
            raise SystemExit(f"{name}: nessun anello della regione FLAME entro {HOLE_MAX_MM} mm ({d[k]:.1f})")
        disc, extra = hole_ring(F, Tt, sphere, Y[loops[k]], target, HOLE_RADIUS_MM[name])
        extra.update(flame_loop_vertices=len(loops[k]), loop_centre_to_landmarks_mm=round(d[k], 2))
        n_gt = int((disc & is_gt).sum())
        if (disc & holes).any():
            raise SystemExit(f"{name}: buco sovrapposto a un altro")
        holes |= disc
        dv = np.unique(F[disc])
        hole_info[name] = {**extra, "triangles": int(disc.sum()), "gt_triangles_in_hole": n_gt,
                           "extent_mm": [round(float(x), 1) for x in Tt[dv].max(0) - Tt[dv].min(0)]}
    patch = C_mask(F, sphere & ~holes)
    patch, n_filled = complement_fill(F, patch, holes, neck_tri)
    patch, holes, n_pinch = fix_pinches(F, patch, holes)
    patch = C_mask(F, patch)
    lps = boundary_loops(F[patch])

    # copertura della GT dentro la vista: triangolo della mappa nella patch, o i suoi tre vertici
    pv = np.zeros(n, dtype=bool)
    pv[np.unique(F[patch])] = True
    in_tri = (gt_tri >= 0) & patch[np.maximum(gt_tri, 0)]
    in_vert = pv[z["vidx_unified"]].all(1)
    W = np.load(domains.DATASETS / "UNIFIED_GT" / "unified_space.npz")["w"]
    W = W / W.sum()
    fm = facemesh_check(Tt, F, lm7, centre, radius)
    pF = F[patch]
    info = {
        "rule": "sfera NoW (now_common.now_mask) dai landmark iBUG del template; tolti i triangoli col baricentro dentro "
                "gli anelli della regione FLAME deformata attorno a occhi e bocca (proiezione frontale); altri buchi chiusi "
                "coi triangoli del template; 4 anelli di bordo",
        "landmarks_lmk7": dict(zip(map(str, now_common.LMK7_IBUG), [how[i] for i in now_common.LMK7_IBUG])),
        "sphere_radius_mm_flame_frame": float(radius),
        "gt_points_inside_sphere": int((np.linalg.norm(C.bary_interp(Tt, z["vidx_unified"], z["bary_unified"]) - centre,
                                                       axis=1) <= radius).sum()),
        "facemesh_check": fm, "holes": hole_info, "spurious_holes_filled_triangles": n_filled,
        "pinch_triangles_added": n_pinch, "pinch_vertices_left": int(len(pinch_vertices(pF))),
        "patch": {"n_vertices": int(pv.sum()), "n_faces": int(patch.sum()),
                  "area_mm2_flame_frame": float(C.face_areas(Tt, pF).sum()),
                  "boundary_loops": [len(lp) for lp in lps], "n_boundary_loops": len(lps)},
        "gt_coverage": {"n": int(len(gt_tri)), "triangle_in_view": int(in_tri.sum()), "vertices_in_view": int(in_vert.sum()),
                        "area_weight_outside": float(W[~in_tri].sum())},
    }
    arrays = {"patch_tri": patch, "patch_F": pF, "centre_flame_mm": centre, "radius_flame_mm": radius,
              "holes_tri": holes, "T_landmark_s": z["T_landmark_s"], "T_landmark_R": z["T_landmark_R"],
              "T_landmark_t": z["T_landmark_t"]}
    return arrays, {"info": info, "Tt": Tt, "F": F, "patch": patch, "holes": holes, "centre": centre, "radius": radius}


def C_mask(F: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Componente connessa piu' grande (per spigolo) dei triangoli selezionati, come maschera sulle righe di F."""
    idx = np.flatnonzero(sel)
    _, lab = connected_components(face_graph(F[idx]), directed=False)
    out = np.zeros(len(F), dtype=bool)
    out[idx[lab == np.bincount(lab).argmax()]] = True
    return out


def figure(ctx: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Tt, F = ctx["Tt"], ctx["F"]
    surf = C.Surface(Tt, F)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, (title, R) in zip(axs, (("frontale", np.eye(3)), ("3/4", C.rot_y(-45)), ("profilo", C.rot_y(-90)))):
        view = rd.framing(Tt, R, 600)
        o = rd.render(surf, view)
        img = o["img"].astype(float) / 255
        t = np.maximum(o["tri"], 0)
        hit = o["tri"] >= 0
        mp = hit & ctx["patch"][t]
        mh = hit & ctx["holes"][t]
        img[mp] = 0.6 * img[mp] + 0.4 * np.array([0.2, 0.5, 1.0])
        img[mh] = 0.4 * img[mh] + 0.6 * np.array([1.0, 0.2, 0.1])
        ax.imshow(img)
        ax.set_title(f"{title}: blu = patch, rosso = dischi tolti")
        ax.axis("off")
    i = ctx["info"]
    fig.suptitle(f"patch {i['patch']['n_vertices']} v / {i['patch']['n_faces']} t, anelli {i['patch']['boundary_loops']}, "
                 f"GT nella vista {i['gt_coverage']['triangle_in_view']}/{i['gt_coverage']['n']}")
    fig.tight_layout()
    ac.EVID_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ac.EVID_DIR / "patch.png", dpi=80)
    plt.close(fig)


def main() -> None:
    ac.refuse_if_frozen("la patch")
    arrays, ctx = build()
    C.save_npz(ac.PATCH_NPZ, **arrays)
    C.save_json(SUMMARY, ctx["info"])
    figure(ctx)
    i = ctx["info"]
    print(f"[ava-patch] {i['patch']['n_vertices']} vertici / {i['patch']['n_faces']} triangoli, anelli "
          f"{i['patch']['boundary_loops']}, buchi chiusi {i['spurious_holes_filled_triangles']} triangoli; GT nella vista "
          f"{i['gt_coverage']['triangle_in_view']}/{i['gt_coverage']['n']} (vertici {i['gt_coverage']['vertices_in_view']}); "
          f"sfera {i['sphere_radius_mm_flame_frame']:.1f} mm; FaceMesh {i['facemesh_check']}; buchi "
          f"{ {k: (v['triangles'], v['gt_triangles_in_hole']) for k, v in i['holes'].items()} }; farfalle "
          f"aggiunte {i['pinch_triangles_added']}, rimaste {i['pinch_vertices_left']}", flush=True)
    if i["patch"]["n_boundary_loops"] != EXPECTED_LOOPS:
        raise SystemExit(f"la patch ha {i['patch']['n_boundary_loops']} anelli di bordo, attesi {EXPECTED_LOOPS}")


if __name__ == "__main__":
    main()
