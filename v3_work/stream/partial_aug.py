"""Parzialita' variabile delle viste: augment OPT-IN del run secondario (``producer.py --partial-p``), spento di default.

Perche': i modelli appresi crollano sulle coppie con il bordo ritagliato (Spearman da ~0.67-0.76 a ~0.24-0.38), pur
avendo nel training la vista ``crop`` (mesh_ops.make_crop: UNA banda geodesica fissa, 0.06 x diagonale dai punti di
bordo oltre il 70esimo percentile del raggio, ~4-5% dei vertici, deterministica per identita'). Qui la parte tolta
cambia a ogni vista, sulla scia di v2_work/train_v2/make_support_bank.py (CROP_QUANTILE 0.5-0.9, bande e buchi).

Con probabilita' ``p`` per vista, DOPO la discretizzazione (views.discretize: la parzialita' si somma a original,
remesh, down8k, noisy, crop, up60k), si toglie una frazione d'area estratta (``MODES``):
  band   banda geodesica dal bordo (Dijkstra sugli spigoli) di area r ~ U(lo, hi); sorgenti: con prob. 1/2 i punti
         di bordo oltre il percentile P ~ U(0, 90) del raggio dalla mediana (P = 70 sono quelle del crop di
         valutazione, P = 0 tutto il bordo), altrimenti un arco del bordo di ampiezza U(60, 240) gradi nel piano del
         volto (i due assi principali);
  plane  taglio planare: si tolgono le facce oltre una retta del piano del volto, direzione uniforme, area r ~ U(lo, hi);
  holes  solo buchi interni (1 o 2), ognuno di area U(HOLE_ONLY);
piu', dopo band e plane, 0-2 buchi interni piccoli (``EXTRA_HOLES``, ognuno U(HOLE_EXTRA)). Un buco e' una palla
geodesica attorno a un vertice lontano dal bordo (1.5 x il raggio atteso del buco), quindi un anello di bordo interno.
L'area si toglie per FACCE in ordine di chiave (distanza o proiezione al baricentro), fino alla frazione voluta: la
perdita realizzata e' quella estratta a meno di una faccia, piu' le schegge staccate (mesh_ops.prepare_open_surface:
niente degeneri, una sola componente, nessun vertice isolato). Con lo, hi = 0.03, 0.40 il log del rapporto sqrt(area)
va da -0.015 a -0.26 circa: copre e supera il crop di valutazione (-0.05 / -0.13).

Validita': perdita realizzata <= ``max_loss`` e almeno ``min_verts`` vertici, altrimenti si riestrae (``tries``) e
alla fine la vista resta intera (``fallback``); gli operatori (views.operators) li calcola make_view come sempre.

Bersagli: invariati, come per ``crop``: s_i, FR/SR, S_i del gruppo vengono dalla forma neutra dell'identita'
(producer.make_group, targets.py); ``area_mm2`` della vista e' l'area della mesh parziale (views.make_view), quindi
l'ingresso globale (global_v3.frame_params: f = sqrt(area_mm2 / area servita)) resta in mm veri.

Seme: ``view_seed(noise_seed)`` = SeedSequence([noise_seed, SALT]), dal seme del rumore della vista, gia'
deterministico e registrato nella provenienza (producer.view_recipe, ``noise_seed`` nei metadati): la parzialita'
NON estrae nulla dal generatore del gruppo, quindi a parita' di semi un anello con e senza ``--partial-p`` ha le
stesse identita', espressioni, discretizzazioni e rumore, e differisce solo per le parti tolte. Parametri realizzati
nei metadati della vista (``partial``), la configurazione nella ricetta dello shard: regen.py la rigenera.
"""
from __future__ import annotations

import numpy as np

SALT = 0x50415254                       # "PART": SeedSequence([noise_seed, SALT])
VERSION = 1
DEFAULTS = {
    "v": VERSION, "p": 0.0, "lo": 0.03, "hi": 0.40,
    "modes": {"band": 0.5, "plane": 0.3, "holes": 0.2},
    "band_pct": [0.0, 90.0],            # percentile del raggio delle sorgenti (famiglia "radius")
    "band_arc_deg": [60.0, 240.0],      # ampiezza dell'arco di bordo (famiglia "arc")
    "extra_holes_p": [0.6, 0.3, 0.1],   # 0, 1, 2 buchi dopo band e plane
    "hole_extra": [0.005, 0.02],        # area di ognuno di quei buchi
    "hole_only_p": [0.5, 0.5],          # 1, 2 buchi nel modo holes
    "hole_only": [0.01, 0.05],
    "max_loss": 0.5, "min_verts": 300, "tries": 3,
}


def config(p: float, area: str = "") -> dict | None:
    """Configurazione (nella ricetta dello shard) da ``--partial-p`` e ``--partial-area lo,hi``; None se p <= 0."""
    if not float(p) > 0:
        return None
    if float(p) > 1:
        raise SystemExit(f"--partial-p {p}: probabilita' per vista in (0, 1]")
    cfg = {**DEFAULTS, "p": float(p)}
    if area:
        lo, hi = (float(x) for x in area.split(","))
        if not 0 < lo <= hi < cfg["max_loss"]:
            raise SystemExit(f"--partial-area {area}: serve 0 < lo <= hi < {cfg['max_loss']}")
        cfg.update(lo=lo, hi=hi)
    return cfg


def view_seed(noise_seed: int) -> list[int]:
    return [int(noise_seed), SALT]


# --- geometria ----------------------------------------------------------------------------------------

def face_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    t = V[F]
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def edge_graph(V: np.ndarray, F: np.ndarray):
    """Grafo simmetrico degli spigoli (csr, pesi = lunghezze) per la Dijkstra di scipy."""
    from scipy.sparse import coo_matrix
    import mesh_ops as mo
    E = mo.extract_unique_edges(F)
    w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    w = np.maximum(w, 1e-12)                            # uno spigolo nullo sarebbe "assente" per csgraph
    n = len(V)
    return coo_matrix((np.concatenate([w, w]), (np.concatenate([E[:, 0], E[:, 1]]),
                                                 np.concatenate([E[:, 1], E[:, 0]]))), shape=(n, n)).tocsr()


def geodesic_from(G, sources: np.ndarray) -> np.ndarray:
    from scipy.sparse.csgraph import dijkstra
    return dijkstra(G, directed=False, indices=np.asarray(sources, dtype=np.int64), min_only=True)


def frame(V: np.ndarray, Af: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(centro, assi principali in colonna, varianza decrescente) pesati per area: e1, e2 = piano del volto."""
    w = np.zeros(len(V))
    np.add.at(w, F.reshape(-1), np.repeat(Af / 3.0, 3))
    c = (w[:, None] * V).sum(0) / w.sum()
    X = V - c
    C = (w[:, None] * X).T @ X / w.sum()
    ev, U = np.linalg.eigh(C)
    return c, U[:, np.argsort(ev)[::-1]]


def cut_faces(F: np.ndarray, Af: np.ndarray, key: np.ndarray, frac: float) -> np.ndarray:
    """Maschera delle facce da TENERE: via quelle di chiave piu' bassa fino a ``frac`` dell'area (ordine stabile)."""
    order = np.argsort(key, kind="stable")
    cum = np.cumsum(Af[order])
    k = int(np.searchsorted(cum, frac * cum[-1], side="left")) + 1
    keep = np.ones(len(F), dtype=bool)
    keep[order[:min(k, len(F))]] = False
    return keep


def clean(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import mesh_ops as mo
    return mo.prepare_open_surface(V, F)


def band(V, F, rng, cfg, frac) -> tuple[np.ndarray, np.ndarray, dict]:
    import mesh_ops as mo
    Af = face_areas(V, F)
    bnd = mo.extract_boundary_vertices(F)
    info: dict = {}
    if rng.random() < 0.5:
        pct = float(rng.uniform(*cfg["band_pct"]))
        r = np.linalg.norm(V[bnd] - np.median(V, axis=0), axis=1)
        src = bnd[r >= np.percentile(r, pct)]
        info.update(family="radius", pct=round(pct, 3))
    else:
        c, U = frame(V, Af, F)
        ang = np.degrees(np.arctan2((V[bnd] - c) @ U[:, 1], (V[bnd] - c) @ U[:, 0]))
        a0, width = float(rng.uniform(-180.0, 180.0)), float(rng.uniform(*cfg["band_arc_deg"]))
        src = bnd[np.mod(ang - a0, 360.0) <= width]
        info.update(family="arc", a0=round(a0, 3), width=round(width, 3))
    if src.size == 0:
        src = bnd
    d = geodesic_from(edge_graph(V, F), src)
    d[~np.isfinite(d)] = np.inf
    keep = cut_faces(F, Af, d[F].mean(1), frac)
    Vo, Fo = clean(V, F[keep])
    return Vo, Fo, info


def plane(V, F, rng, cfg, frac) -> tuple[np.ndarray, np.ndarray, dict]:
    Af = face_areas(V, F)
    c, U = frame(V, Af, F)
    th = float(rng.uniform(0.0, 2 * np.pi))
    u = np.cos(th) * U[:, 0] + np.sin(th) * U[:, 1]
    keep = cut_faces(F, Af, -(V[F].mean(1) - c) @ u, frac)      # via prima le facce piu' avanti lungo u
    Vo, Fo = clean(V, F[keep])
    return Vo, Fo, {"theta": round(th, 6)}


def hole(V, F, rng, frac) -> tuple[np.ndarray, np.ndarray, bool]:
    """Palla geodesica di area ``frac`` attorno a un vertice lontano dal bordo; (V, F, fatto)."""
    import mesh_ops as mo
    Af = face_areas(V, F)
    G = edge_graph(V, F)
    db = geodesic_from(G, mo.extract_boundary_vertices(F))
    rad = np.sqrt(frac * Af.sum() / np.pi)
    cand = np.flatnonzero(np.isfinite(db) & (db > 1.5 * rad))
    if cand.size == 0:
        return V, F, False
    d = geodesic_from(G, [int(cand[int(rng.integers(cand.size))])])
    d[~np.isfinite(d)] = np.inf
    keep = cut_faces(F, Af, d[F].mean(1), frac)
    Vo, Fo = clean(V, F[keep])
    return Vo, Fo, True


def _draw(V, F, rng, cfg) -> tuple[np.ndarray, np.ndarray, dict]:
    modes = list(cfg["modes"])
    mode = modes[int(rng.choice(len(modes), p=np.asarray([cfg["modes"][m] for m in modes], dtype=np.float64)))]
    info: dict = {"mode": mode}
    if mode in ("band", "plane"):
        r = float(rng.uniform(cfg["lo"], cfg["hi"]))
        V, F, extra = (band if mode == "band" else plane)(V, F, rng, cfg, r)
        info.update(target=round(r, 6), **extra)
        n_h = int(rng.choice(len(cfg["extra_holes_p"]), p=cfg["extra_holes_p"]))
        hr = cfg["hole_extra"]
    else:
        n_h = 1 + int(rng.choice(len(cfg["hole_only_p"]), p=cfg["hole_only_p"]))
        hr = cfg["hole_only"]
    sizes, made = [], 0
    for _ in range(n_h):
        h = float(rng.uniform(*hr))
        sizes.append(round(h, 6))
        V, F, ok = hole(V, F, rng, h)
        made += int(ok)
    info.update(holes=made, hole_sizes=sizes)
    return V, F, info


def apply(V: np.ndarray, F: np.ndarray, cfg: dict, noise_seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """(V, F, info) della vista: parziale con probabilita' cfg['p'], dal seme ``view_seed(noise_seed)``.
    ``info`` va nei metadati della vista: on, seme, modo e parametri estratti, perdita d'area realizzata."""
    rng = np.random.default_rng(np.random.SeedSequence(view_seed(noise_seed)))
    info: dict = {"on": False, "seed": view_seed(noise_seed)}
    if not rng.random() < float(cfg["p"]):
        return V, F, info
    A0, n0 = float(face_areas(V, F).sum()), len(V)
    for t in range(int(cfg["tries"])):
        Vo, Fo, d = _draw(V, F, rng, cfg)
        loss = 1.0 - float(face_areas(Vo, Fo).sum()) / A0
        if 0.0 < loss <= float(cfg["max_loss"]) and len(Vo) >= int(cfg["min_verts"]):
            Vo = np.asarray(Vo, dtype=np.float64)
            return Vo, np.ascontiguousarray(Fo, dtype=np.int32), {
                **info, "on": True, "try": t, **d, "loss": round(loss, 6), "n_in": n0, "n_out": int(len(Vo))}
    return V, F, {**info, "fallback": True}
