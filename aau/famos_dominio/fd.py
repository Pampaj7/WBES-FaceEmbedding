#!/usr/bin/env python3
"""FaMoS come confine di dominio: perche' sul riconoscimento dell'identita' (``aau/esterna_famos_id``) il nostro e' sotto
ICP mm e NoW metrical. Regole in ``aau/runs/evidence/famos_dominio/REGOLE.md`` (scritte prima dei numeri).

    aau/run.sh aau/famos_dominio/fd.py stats --workers 16     (statistiche geometriche delle grezze e delle patch)
    aau/run.sh aau/famos_dominio/fd.py analyze                 (H1, H2, H3: csv, summary.json, results.md)
    (fd.sbatch: i due passi e la figura, solo CPU)

Nessuna distanza nuova: le 14 colonne di ``ei.load_columns`` sulle 11.175 coppie. **H1** (espressione): etichetta
``expr_mm`` di ``famos_test_view`` (registrazione contro neutra del soggetto, Procrustes), terzili delle 150 scansioni,
coppia nello strato della piu' espressiva; AUC_k = stessa persona dello strato k contro tutte le coppie di persone
diverse; rank-1_k per query dello strato; interazione Delta_H - Delta_L; sottoinsieme quasi neutro (solo L). **H2**
(qualita'): stesse letture sui terzili dell'irregolarita' del bordo della patch (coppia nello strato della peggiore),
piu' Spearman fra statistiche per scansione ed eccesso d'errore. **H3** (taglia): |CS_i - CS_j| come unico predittore,
e la quota del vantaggio di ICP mm (NoW metrical) che sparisce togliendo la taglia.

Valori per scansione (etichette, statistiche) solo in ``datasets/FAMOS/esterna_id/eval/dominio`` (fuori da git, licenza
MPI); in git solo aggregati.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
for _p in (AAU / "esterna_famos_id", AAU / "famos"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ei  # noqa: E402  (aggiunge famos, recon, protocolli, baselines_mm al path)
import famos_common as fc  # noqa: E402

EVID = AAU / "runs" / "evidence" / "famos_dominio"
OUT = ei.DATA / "eval" / "dominio"
MANIFEST = fc.VIEW_DIR / "manifest.csv"
# letture del verdetto (REGOLE.md): d_F non cal. dei due semi contro le due migliori
MAIN = ("factorized_s1234|form", "factorized_s2345|form")
COMP = ("icp|mm", "now_metrical|mm")
STRATA = ("L", "M", "H")
FACTORS = {"expr": "H1 espressione (expr_mm, mm)", "quality": "H2 irregolarita' del bordo della patch (q)"}
RAW_STATS = ("raw_n_vertices", "raw_n_faces", "raw_area_mm2", "raw_density_per_cm2", "raw_components",
             "raw_boundary_mm", "raw_boundary_loops")
PATCH_STATS = ("patch_area_mm2", "patch_components", "patch_boundary_loops", "patch_boundary_mm", "patch_q")
MARK = "<!-- da qui in giu': scritto a mano, conservato da fd.py analyze -->"
LABEL = {k: lab for k, lab, _, _ in ei.COLUMNS}


# ------------------------------------------------------------------------------------------ etichette e statistiche

def expr_labels() -> np.ndarray:
    """e_i = ``expr_mm`` della riga ``scan`` di test_view con lo stesso soggetto, sequenza e fotogramma (ordine di
    ``ei.items``). Solo etichetta."""
    sp = json.loads(ei.spec_path("eval").read_text())
    with open(MANIFEST, newline="") as fh:
        idx = {(r["view_id"], r["seq"], int(r["frame"])): float(r["expr_mm"])
               for r in csv.DictReader(fh) if r["kind"] == "scan"}
    return np.asarray([idx[(v, t, int(sp["frames"][v][t]))] for v, t, _ in ei.items("eval")])


def mesh_stats(V: np.ndarray, F: np.ndarray) -> dict:
    """Area, vertici usati, componenti connesse (per vertici condivisi), bordo (spigoli di un solo triangolo):
    lunghezza e anelli (componenti del grafo degli spigoli di bordo)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()
    used = np.unique(F)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    Eu, cnt = np.unique(E, axis=0, return_counts=True)
    B = Eu[cnt == 1]

    def n_comp(edges: np.ndarray) -> int:
        if not len(edges):
            return 0
        a = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(V), len(V)))
        _, lab = connected_components(a, directed=False)
        return int(len(np.unique(lab[np.unique(edges)])))

    return {"n_vertices": int(len(used)), "n_faces": int(len(F)), "area_mm2": float(area),
            "components": n_comp(Eu), "boundary_mm": float(np.linalg.norm(V[B[:, 0]] - V[B[:, 1]], axis=1).sum()),
            "boundary_loops": n_comp(B)}


def _stats_task(k: int) -> dict:
    raw, (_, _, patch) = ei.raw_of("eval")[k], ei.items("eval")[k]
    r = mesh_stats(*fc.read_obj(raw))
    with np.load(patch) as z:
        p = mesh_stats(z["V"].astype(np.float64), z["F"].astype(np.int64))
    return {"raw_n_vertices": r["n_vertices"], "raw_n_faces": r["n_faces"], "raw_area_mm2": r["area_mm2"],
            "raw_density_per_cm2": r["n_vertices"] / (r["area_mm2"] / 100.0), "raw_components": r["components"],
            "raw_boundary_mm": r["boundary_mm"], "raw_boundary_loops": r["boundary_loops"],
            "patch_area_mm2": p["area_mm2"], "patch_components": p["components"],
            "patch_boundary_loops": p["boundary_loops"], "patch_boundary_mm": p["boundary_mm"],
            "patch_q": p["boundary_mm"] / (2.0 * np.sqrt(np.pi * p["area_mm2"]))}


def stats(workers: int) -> None:
    t0 = time.time()
    it = ei.items("eval")
    with mp.Pool(workers) as pool:
        rows = pool.map(_stats_task, range(len(it)))
    ei.atomic_savez(OUT / "quality.npz", names=np.asarray([p.stem for _, _, p in it]), expr_mm=expr_labels(),
                    **{c: np.asarray([r[c] for r in rows], np.float64) for c in RAW_STATS + PATCH_STATS})
    print(f"[fd] stats: {len(rows)} scansioni -> {OUT / 'quality.npz'}, {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------ analisi

def tertiles(x: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Strato 0/1/2 (L, M, H) coi terzili ``statistics.quantiles(x, n=3)``: L se x <= t1, H se x > t2."""
    t1, t2 = statistics.quantiles(x.tolist(), n=3)
    return np.where(x <= t1, 0, np.where(x <= t2, 1, 2)), (float(t1), float(t2))


def ci(x: np.ndarray) -> tuple[float, float, int]:
    """IC percentile sulle repliche finite (la 0 e' il punto) e il numero di repliche escluse."""
    r = x[1:][np.isfinite(x[1:])]
    lo, hi = np.percentile(r, [2.5, 97.5]) if len(r) else (np.nan, np.nan)
    return float(lo), float(hi), int(len(x) - 1 - len(r))


def wmean(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Media pesata per replica (w: repliche x elementi); nan dove il peso e' nullo."""
    s = w.sum(1)
    return np.where(s > 0, (w @ v) / np.where(s > 0, s, 1.0), np.nan)


def auc_masked(score: np.ndarray, gen: np.ndarray, w: np.ndarray, keep: np.ndarray) -> float:
    import famos_eval as fe
    g, i = keep & gen & (w > 0), keep & ~gen & (w > 0)
    if not g.any() or not i.any():
        return np.nan
    return fe.weighted_auc(score[keep], gen[keep], w[keep])


def pair_far(d: np.ndarray, gen: np.ndarray) -> np.ndarray:
    """Per coppia della stessa persona: quota (non pesata) di coppie di persone diverse piu' vicine, pari = 1/2."""
    di = np.sort(d[~gen])
    lo, hi = np.searchsorted(di, d[gen], "left"), np.searchsorted(di, d[gen], "right")
    return (lo + 0.5 * (hi - lo)) / len(di)


def spearman_boot(x: np.ndarray, y: np.ndarray, subj: np.ndarray, counts: np.ndarray) -> np.ndarray:
    from scipy.stats import spearmanr
    out = np.empty(len(counts))
    for b, c in enumerate(counts):
        k = np.repeat(np.arange(len(x)), c[subj])
        out[b] = spearmanr(x[k], y[k]).correlation if np.ptp(x[k]) > 0 and np.ptp(y[k]) > 0 else np.nan
    return out


def analyze() -> None:
    import famos_eval as fe
    t0 = time.time()
    cols, it, _ = ei.load_columns("eval")
    names = [p.stem for _, _, p in it]
    with np.load(OUT / "quality.npz") as z:
        if list(z["names"].astype(str)) != names:
            raise SystemExit("quality.npz: scansioni diverse da ei.items")
        Q = {k: np.asarray(z[k], np.float64) for k in z.files if k != "names"}
    if not np.allclose(Q["expr_mm"], expr_labels()):
        raise SystemExit("expr_mm diverso dal manifest")
    # H3: taglia oracolo come colonna in piu' (CS della patch, la stessa di ICP cs)
    import blmm
    sc = ei.scalars("eval")
    cs = np.asarray([sc[blmm.rel(p)]["cs"] for _, _, p in it])
    subj_names = sorted({x for x, _, _ in it})
    subj = np.asarray([subj_names.index(x) for x, _, _ in it])
    n, ns = len(it), len(subj_names)
    iu, ju = ei.pairs(n)
    gen = subj[iu] == subj[ju]
    cols["oracle|cs"] = np.abs(cs[iu] - cs[ju])
    keys = [k for k, *_ in ei.COLUMNS] + ["oracle|cs"]
    LABEL["oracle|cs"] = "taglia oracolo, abs(CS_i - CS_j)"
    counts = np.concatenate([np.ones((1, ns), np.int64), fe.bootstrap_counts(ns, ei.N_BOOT, ei.BOOT_SEED)])
    W = np.where(gen[None], counts[:, subj[iu]], counts[:, subj[iu]] * counts[:, subj[ju]]).astype(np.float64)
    wq = counts[:, subj].astype(np.float64)
    strata = {"expr": tertiles(Q["expr_mm"]), "quality": tertiles(Q["patch_q"])}
    low = strata["expr"][0] == 0                                  # quasi neutro: solo L di H1
    li = np.flatnonzero(low)
    sub_pairs = low[iu] & low[ju]
    R = {}                                                         # (colonna, chiave) -> repliche
    for key in keys:
        D = np.zeros((n, n))
        D[iu, ju] = D[ju, iu] = cols[key]
        r1, ap = ei.retrieval(D, subj)
        R[key, "all", "rank1"] = wmean(r1, wq)
        R[key, "all", "mAP"] = wmean(ap, wq)
        R[key, "all", "AUC"] = np.asarray([fe.weighted_auc(-cols[key], gen, w) for w in W])
        R[key, "all", "EER"] = np.asarray([ei.eer(cols[key], gen, w) for w in W])
        for fac, (s, _) in strata.items():
            ps = np.maximum(s[iu], s[ju])
            for k, st in enumerate(STRATA):
                R[key, f"{fac}:{st}", "rank1"] = wmean(r1 * (s == k), wq * (s == k))
                R[key, f"{fac}:{st}", "AUC"] = np.asarray(
                    [auc_masked(-cols[key], gen, w, ~gen | (ps == k)) for w in W])
                R[key, f"{fac}:{st}", "AUC_imp_strato"] = np.asarray(
                    [auc_masked(-cols[key], gen, w, ps == k) for w in W])
        # quasi neutro: query e galleria in L; query senza stessa persona in L escluse
        r1s, _ = ei.retrieval(D[np.ix_(li, li)], subj[li])
        has = np.asarray([(subj[li] == subj[q]).sum() > 1 for q in li])
        R[key, "quasi_neutro", "rank1"] = wmean(r1s * has, wq[:, li] * has)
        R[key, "quasi_neutro", "AUC"] = np.asarray([auc_masked(-cols[key], gen, w, sub_pairs) for w in W])
        print(f"[fd] {key}: AUC {R[key, 'all', 'AUC'][0]:.3f} (controllo contro ei.py)", flush=True)
    # controllo: le metriche globali devono coincidere con metrics.csv di esterna_famos_id
    with open(ei.EVID / "metrics.csv", newline="") as fh:
        ref = {(r["colonna"], r["metrica"]): float(r["valore"]) for r in csv.DictReader(fh)}
    check = max(abs(R[k, "all", m][0] - v) for (k, m), v in ref.items())
    if check > 1e-12:
        raise SystemExit(f"metriche globali diverse da esterna_famos_id/metrics.csv: {check}")

    srows, verdict = [], {}
    for key in keys:
        for (k2, part, met), x in R.items():
            if k2 != key:
                continue
            lo, hi, nx = ci(x)
            srows.append({"colonna": key, "metodo": LABEL[key], "insieme": part, "metrica": met, "valore": float(x[0]),
                          "ic_lo": lo, "ic_hi": hi, "repliche_escluse": nx, "riga": "metrica"})
    for ko in MAIN:
        for kc in [k for k, *_ in ei.COMPETITORS] + ["oracle|cs"]:
            for (k2, part, met), x in R.items():
                if k2 != ko or (kc, part, met) not in R:
                    continue
                d = R[ko, part, met] - R[kc, part, met]
                lo, hi, nx = ci(d)
                srows.append({"colonna": f"{ko} - {kc}", "metodo": f"{LABEL[ko]} - {LABEL[kc]}", "insieme": part,
                              "metrica": met, "valore": float(d[0]), "ic_lo": lo, "ic_hi": hi,
                              "repliche_escluse": nx, "riga": "delta"})
            for fac in strata:
                for met in ("AUC", "rank1"):
                    dl = R[ko, f"{fac}:L", met] - R[kc, f"{fac}:L", met]
                    dh = R[ko, f"{fac}:H", met] - R[kc, f"{fac}:H", met]
                    lo, hi, nx = ci(dh - dl)
                    srows.append({"colonna": f"{ko} - {kc}", "metodo": f"{LABEL[ko]} - {LABEL[kc]}",
                                  "insieme": f"{fac}:H-L", "metrica": met, "valore": float(dh[0] - dl[0]),
                                  "ic_lo": lo, "ic_hi": hi, "repliche_escluse": nx, "riga": "interazione"})
    # quota del divario globale di AUC per strato (1 - AUC = media pesata della FAR delle coppie della stessa persona)
    for ko in MAIN:
        for kc in COMP:
            fo, fcmp = pair_far(cols[ko], gen), pair_far(cols[kc], gen)
            for fac, (s, _) in strata.items():
                ps = np.maximum(s[iu], s[ju])[gen]
                tot = (fcmp - fo).sum()
                for k, st in enumerate(STRATA):
                    srows.append({"colonna": f"{ko} - {kc}", "metodo": f"{LABEL[ko]} - {LABEL[kc]}",
                                  "insieme": f"{fac}:{st}", "metrica": "quota_divario_AUC",
                                  "valore": float((fcmp - fo)[ps == k].sum() / tot), "ic_lo": np.nan,
                                  "ic_hi": np.nan, "repliche_escluse": 0, "riga": "scomposizione"})

    def row(col, part, met, kind):
        return next(r for r in srows if r["colonna"] == col and r["insieme"] == part and r["metrica"] == met
                    and r["riga"] == kind)

    for fac in strata:
        inter = [row(f"{ko} - {kc}", f"{fac}:H-L", "AUC", "interazione") for ko in MAIN for kc in COMP]
        below = [r["ic_hi"] < 0 for r in inter]
        v = "SUPPORTATA" if all(below) else "NON SUPPORTATA" if not any(below) else "PARZIALE"
        gap = [row(f"{ko} - {kc}", p, "AUC", "delta") for ko in MAIN for kc in COMP
               for p in ((f"{fac}:L", "quasi_neutro") if fac == "expr" else (f"{fac}:L",))]
        explains = v == "SUPPORTATA" and all(r["ic_hi"] >= 0 for r in gap)
        verdict[fac] = {"verdetto": v, "spiega_il_divario": explains,
                        "interazioni_AUC": {r["colonna"]: [r["valore"], r["ic_lo"], r["ic_hi"]] for r in inter},
                        "delta_AUC_basso": {f"{r['colonna']} @ {r['insieme']}": [r["valore"], r["ic_lo"], r["ic_hi"]]
                                            for r in gap}}

    # H3: quota del vantaggio dovuta alla taglia
    zrows = []
    for ko in MAIN:
        for tag, km, kn in (("ICP", "icp|mm", "icp|cs"), ("NoW", "now_metrical|mm", "now_original|mm")):
            for met in ("AUC", "rank1"):
                num = R[km, "all", met] - R[kn, "all", met]
                den = R[km, "all", met] - R[ko, "all", met]
                f = np.where(den != 0, num / np.where(den != 0, den, 1.0), np.nan)
                for q, x in (("numeratore (con taglia - senza)", num), ("denominatore (concorrente - nostro)", den),
                             ("f = quota della taglia", f)):
                    lo, hi, nx = ci(x)
                    zrows.append({"nostro": ko, "famiglia": tag, "metrica": met, "grandezza": q, "valore": float(x[0]),
                                  "ic_lo": lo, "ic_hi": hi, "repliche_escluse": nx})
    num_ok = next(r for r in zrows if r["famiglia"] == "ICP" and r["metrica"] == "AUC"
                  and r["grandezza"].startswith("numeratore"))["ic_lo"] > 0
    f_icp = [next(r for r in zrows if r["nostro"] == ko and r["famiglia"] == "ICP" and r["metrica"] == "AUC"
                  and r["grandezza"].startswith("f"))["valore"] for ko in MAIN]
    verdict["size"] = {"verdetto": ("SUPPORTATA" if num_ok and min(f_icp) >= 0.5 else "PARZIALE" if num_ok
                                    else "NON SUPPORTATA"), "f_ICP_AUC": dict(zip(MAIN, f_icp)),
                       "numeratore_ICP_AUC_sopra_0": bool(num_ok)}

    # H2 (e H1 continua): Spearman fra statistica per scansione ed eccesso d'errore x_i
    qrows = []
    gi, gj = iu[gen], ju[gen]
    for ko in MAIN:
        for kc in COMP:
            x = pair_far(cols[ko], gen) - pair_far(cols[kc], gen)
            xi = (np.bincount(gi, x, n) + np.bincount(gj, x, n)) / (np.bincount(gi, None, n) + np.bincount(gj, None, n))
            for st in ("expr_mm",) + RAW_STATS + PATCH_STATS:
                rho = spearman_boot(Q[st], xi, subj, counts)
                lo, hi, nx = ci(rho)
                qrows.append({"nostro": ko, "concorrente": kc, "statistica": st, "spearman": float(rho[0]),
                              "ic_lo": lo, "ic_hi": hi, "repliche_escluse": nx})
    desc = {st: {"mediana": float(np.median(Q[st])), "p5": float(np.percentile(Q[st], 5)),
                 "p95": float(np.percentile(Q[st], 95))} for st in ("expr_mm",) + RAW_STATS + PATCH_STATS}

    EVID.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("strata.csv", srows), ("size.csv", zrows), ("quality.csv", qrows)):
        with open(EVID / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    sizes = {fac: {"tagli": cut, "scansioni": [int((s == k).sum()) for k in range(3)],
                   "coppie_stessa_persona": [int((np.maximum(s[iu], s[ju])[gen] == k).sum()) for k in range(3)]}
             for fac, (s, cut) in strata.items()}
    summary = {"n_scans": n, "n_subjects": ns, "n_pairs": len(iu), "n_genuine": int(gen.sum()), "strati": sizes,
               "quasi_neutro": {"scansioni": int(low.sum()), "soggetti_con_coppie": int(len(
                   [s for s in np.unique(subj[li]) if (subj[li] == s).sum() > 1])),
                   "coppie_stessa_persona": int((sub_pairs & gen).sum())},
               "statistiche": desc, "verdetti": verdict, "controllo_metriche_globali_max_abs": check,
               "repliche": ei.N_BOOT, "seme": ei.BOOT_SEED, "seconds": time.time() - t0, **ei.stamp()}
    ei.atomic_json(EVID / "summary.json", summary)
    write_results(srows, zrows, qrows, summary)
    print(f"[fd] analyze: {json.dumps({k: v['verdetto'] for k, v in verdict.items()})}, {time.time() - t0:.0f}s",
          flush=True)


def write_results(srows: list, zrows: list, qrows: list, summary: dict) -> None:
    def get(col, part, met, kind="metrica"):
        return next((r for r in srows if r["colonna"] == col and r["insieme"] == part and r["metrica"] == met
                     and r["riga"] == kind), None)

    def cell(r, sign=False):
        if r is None:
            return "-"
        f = "{:+.3f}" if sign else "{:.3f}"
        return f"{f.format(r['valore'])} [{f.format(r['ic_lo'])}, {f.format(r['ic_hi'])}]"

    keys = [k for k, *_ in ei.COLUMNS] + ["oracle|cs"]
    st = summary["strati"]
    L = ["# FaMoS come confine di dominio: espressione, qualita' della scansione, taglia", "",
         "Generato da `aau/famos_dominio/fd.py analyze` (regole in `REGOLE.md`, scritte e committate prima dei numeri). "
         "Distanze di `esterna_famos_id` (protocollo 29baaa7), nessuna nuova. Lettura DESCRITTIVA; IC 95% percentile, "
         f"{summary['repliche']} repliche per soggetto (seme {summary['seme']}). Controllo: metriche globali uguali a "
         f"`esterna_famos_id/metrics.csv` (scarto massimo {summary['controllo_metriche_globali_max_abs']:.1e}).", "",
         "## Verdetti", ""]
    for fac, lab in (("expr", "H1 espressione"), ("quality", "H2 qualita' della scansione"), ("size", "H3 taglia")):
        v = summary["verdetti"][fac]
        extra = (f"; spiega il divario: {'si' if v['spiega_il_divario'] else 'no'}" if "spiega_il_divario" in v else
                 f"; f_ICP(AUC) = {', '.join(f'{x:.2f}' for x in v['f_ICP_AUC'].values())} (s1234, s2345)")
        L.append(f"- **{lab}**: {v['verdetto']}{extra}.")
    for fac, title in FACTORS.items():
        s = st[fac]
        L += ["", f"## {title}", "",
              f"Terzili t1 = {s['tagli'][0]:.3f}, t2 = {s['tagli'][1]:.3f}; scansioni L/M/H "
              f"{'/'.join(map(str, s['scansioni']))}; coppie della stessa persona per strato (della scansione "
              f"{'piu espressiva' if fac == 'expr' else 'peggiore'}) {'/'.join(map(str, s['coppie_stessa_persona']))}.",
              "", "AUC_k: stessa persona dello strato k contro tutte le coppie di persone diverse. rank-1_k: query dello "
              "strato, galleria completa.", "",
              "| metodo | AUC L | AUC M | AUC H | rank-1 L | rank-1 M | rank-1 H |", "|---|---|---|---|---|---|---|"]
        for k in keys:
            L.append(f"| {LABEL[k]} | " + " | ".join(cell(get(k, f"{fac}:{x}", m)) for m in ("AUC", "rank1")
                                                     for x in STRATA) + " |")
        L += ["", "Delta nostro - concorrente per strato (positivo = il nostro meglio) e interazione Delta_H - Delta_L.",
              "", "| nostro - concorrente | dAUC L | dAUC M | dAUC H | interazione AUC | interazione rank-1 | "
              "quota del divario AUC L/M/H |", "|---|---|---|---|---|---|---|"]
        for ko in MAIN:
            for kc in [k for k, *_ in ei.COMPETITORS]:
                c = f"{ko} - {kc}"
                share = [get(c, f"{fac}:{x}", "quota_divario_AUC", "scomposizione") for x in STRATA]
                L.append(f"| {LABEL[ko]} - {LABEL[kc]} | "
                         + " | ".join(cell(get(c, f"{fac}:{x}", "AUC", "delta"), True) for x in STRATA)
                         + f" | {cell(get(c, f'{fac}:H-L', 'AUC', 'interazione'), True)}"
                         + f" | {cell(get(c, f'{fac}:H-L', 'rank1', 'interazione'), True)} | "
                         + ("/".join(f"{r['valore']:.2f}" for r in share) if share[0] else "-") + " |")
        if fac == "expr":
            qn = summary["quasi_neutro"]
            L += ["", f"### Sottoinsieme quasi neutro (solo L: {qn['scansioni']} scansioni, {qn['soggetti_con_coppie']} "
                  f"soggetti con almeno una coppia, {qn['coppie_stessa_persona']} coppie della stessa persona)", "",
                  "| metodo | rank-1 | AUC | d rank-1 (s1234 - metodo) | d AUC (s1234 - metodo) |", "|---|---|---|---|---|"]
            for k in keys:
                L.append(f"| {LABEL[k]} | {cell(get(k, 'quasi_neutro', 'rank1'))} | {cell(get(k, 'quasi_neutro', 'AUC'))}"
                         f" | {cell(get(f'{MAIN[0]} - {k}', 'quasi_neutro', 'rank1', 'delta'), True)}"
                         f" | {cell(get(f'{MAIN[0]} - {k}', 'quasi_neutro', 'AUC', 'delta'), True)} |")
        L += ["", "Secondaria (AUC_k con le sole coppie di persone diverse dello stesso strato), per le letture del "
              "verdetto:", "", "| metodo | L | M | H |", "|---|---|---|---|"]
        for k in MAIN + COMP:
            L.append(f"| {LABEL[k]} | " + " | ".join(cell(get(k, f"{fac}:{x}", "AUC_imp_strato")) for x in STRATA)
                     + " |")
    L += ["", "## H3 taglia", "", "Predittore a sola taglia (|CS_i - CS_j|, CS della patch in mm):", "",
          "| rank-1 | mAP | AUC | EER |", "|---|---|---|---|",
          "| " + " | ".join(cell(get("oracle|cs", "all", m)) for m in ("rank1", "mAP", "AUC", "EER")) + " |", "",
          "Quota del vantaggio dovuta alla taglia, f = (con taglia - senza) / (concorrente con taglia - nostro):", "",
          "| nostro | famiglia | metrica | grandezza | valore [IC] |", "|---|---|---|---|---|"]
    for r in zrows:
        L.append(f"| {LABEL[r['nostro']]} | {r['famiglia']} | {r['metrica']} | {r['grandezza']} | {cell(r, True)} |")
    L += ["", "## H2 (e H1 continua): Spearman fra statistica per scansione ed eccesso d'errore", "",
          "x_i = c_i(nostro) - c_i(concorrente), c_i = quota media di coppie di persone diverse piu' vicine delle coppie "
          "della stessa persona che contengono i; rho > 0 = la statistica cresce dove il nostro sbaglia di piu'.", "",
          "| statistica | mediana [p5, p95] | " + " | ".join(f"{LABEL[ko].split(',')[0]} - {LABEL[kc]}" for ko in MAIN
                                                    for kc in COMP) + " |", "|---|---|" + "---|" * 4]
    for s in ("expr_mm",) + RAW_STATS + PATCH_STATS:
        d = summary["statistiche"][s]
        cells = [next(r for r in qrows if r["nostro"] == ko and r["concorrente"] == kc and r["statistica"] == s)
                 for ko in MAIN for kc in COMP]
        L.append(f"| {s} | {d['mediana']:.4g} [{d['p5']:.4g}, {d['p95']:.4g}] | "
                 + " | ".join(f"{r['spearman']:+.2f} [{r['ic_lo']:+.2f}, {r['ic_hi']:+.2f}]" for r in cells) + " |")
    L += ["", "File: `strata.csv` (metriche, delta, interazioni e scomposizione per strato), `size.csv`, `quality.csv`, "
          "`summary.json`, figura `fig_famos_expr.pdf` (`fig_famos_expr.py`).", ""]
    out = EVID / "results.md"
    tail = out.read_text().split(MARK, 1)[1] if out.exists() and MARK in out.read_text() else "\n"
    out.write_text("\n".join(L) + "\n" + MARK + tail)


# ------------------------------------------------------------------------------------------ main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("stats", "analyze"))
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    if a.step == "stats":
        stats(a.workers)
    else:
        analyze()


if __name__ == "__main__":
    main()
