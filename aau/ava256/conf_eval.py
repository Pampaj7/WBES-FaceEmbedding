#!/usr/bin/env python3
"""Ava-256, valutazione confermativa (PROTOCOL_CONFERMATIVO, sez. 5-8, 9.2): R1-R7 con le righe fallite dei metodi per
coppia a +inf, Holm con m = 10 per seme del modello, regola "non risolto", lettura dei due semi; accanto, la versione
mascherata (righe fallite tolte), NoW metrical contro d_F e gli Spearman dei protocolli (descrittivi). Collaudato su
HIFI3D con lo STESSO codice (spec ``aau/ava256/spec/hifi3d_full.json`` e ``hifi3d_det.json``, ``conf_collaudo.py``).

    v3_work/unified_gt/run.sh aau/ava256/conf_eval.py --spec <json> --workers 64      (LEGGE LE GT del dominio)

Con ``freeze`` nello spec, ``conf_common.verify`` (sez. 11) PRIMA di leggere qualunque file del dominio.

Righe (sez. 5): coppie di soggetti valutati diversi a < b (soggetti ordinati) x coppie ORDINATE di topologie diverse
(``zs_expr_summarize.TOPOLOGIES``); 200 soggetti: 19.900 x 30 = 597.000. Gruppi come ``fm_eval.groups_of``: ``nocrop``
(nessuna crop: ``nocrop_cross`` del protocollo), ``crop`` (una crop), ``all_cross``. Con ``pairs`` nello spec (solo
collaudo) le righe del sottocampione di ``conf_protocols.pairs_of``.

Colonne (``columns`` dello spec), come quelle della matrice unificata: per seme d_F calibrata
(``factorized_<seme>|form_cal``, c dello spec), d_F non calibrata (``|form``), d_P (``|shape``) da
``fact_paired.distances`` sugli embedding; ICP + Chamfer e NICP per coppia in mm e cs, NICP su template in mm (e cs,
descrittivo) da ``fm_eval.geo_columns`` sulle matrici di blmm della vista (``blmm.register_spec``); NoW originale,
NoW metrical, REALY, Stirling dai plug-in (``fm_eval.plugin_columns``, ``D_mm``); testa di R6 (``head_<seme>|shape``,
d_F della testa ``|form`` descrittiva) da ``conf_head.py``. GT: ``zs_summarize.load_gt`` dei file dello spec.

Repliche (sez. 6): 1000 per soggetto, ``default_rng(boot_seed)`` sui soggetti ordinati, replica 0 = il punto; le stesse
in ogni gruppo e per ogni colonna (``fm_eval.replicas``: Spearman pesato c_a c_b, ranghi delle righe ripetute).
Lettura confermativa: righe del gruppo con TUTTE le colonne PER MESH della famiglia finite (d_F cal., d_P e testa dei
due semi, NICP su template in mm); una distanza non finita di un metodo PER COPPIA della famiglia (ICP e NICP per coppia
in mm e cs, NoW originale e metrical) vale +inf (rango massimo, ranghi medi fra le righe fallite). Delta = a - b
replica per replica; IC 95% percentile; k = repliche con delta <= 0, p = k / 1000. Holm per seme sulle 10 regole
(``RULES``): p ordinati, si rifiuta H(i) finche' 40 k(i) (m - i + 1) <= n (n = 1000: k (m - i + 1) <= 25); R3a, R3b, R7
(senza seme) con lo stesso p in entrambe le famiglie. Una regola non risolta o la R6 con fit non convergente resta nella
famiglia col suo p. Esiti (sez. 6, 8): confermata = Holm rifiuta e delta > 0 in entrambi i semi (R3a, R3b, R7: in
entrambe le famiglie); non risolta = regola della sez. 8 (``unresolved``), prevale; non confermata = il resto.

Non risolto (sez. 8, 9.1), per gruppo e metodo della regola: mesh fallite > 2% delle mesh delle topologie del gruppo
(metodi per mesh), righe con distanza non finita > 5% delle righe del gruppo (contate prima della maschera e del
+inf), ripieghi del detector > ``conf_protocols.FALLBACK_MAX`` = 2% delle mesh del gruppo (protocolli, R7 e
descrittivi). Uscite in ``<evidence>/conf``: ``rules.csv``, ``deltas.csv``, ``spearman.csv``, ``controls.json``,
``results.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(AAU / "full_matrix"))
import conf_common as cc  # noqa: E402

SEEDS = ("s1234", "s2345")
GROUPS = ("nocrop", "crop", "all_cross")
# (regola, a, b, GT, gruppo, per seme); delta = a - b, positivo = a davanti
RULES = (("R1", "factorized_{s}|form_cal", "mm_rigid_icp_chamfer", "fr", "nocrop", True),
         ("R2a", "factorized_{s}|form_cal", "mm_nicp_template", "fr", "nocrop", True),
         ("R2b", "factorized_{s}|form_cal", "mm_nicp_p2tri", "fr", "nocrop", True),
         ("R3a", "mm_rigid_icp_chamfer", "cs_rigid_icp_chamfer", "fr", "nocrop", False),
         ("R3b", "cs_rigid_icp_chamfer", "cs_nicp_p2tri", "fr", "nocrop", False),
         ("R4", "factorized_{s}|form_cal", "mm_rigid_icp_chamfer", "fr", "crop", True),
         ("R5a", "factorized_{s}|shape", "cs_rigid_icp_chamfer", "sr", "nocrop", True),
         ("R5b", "factorized_{s}|shape", "cs_nicp_p2tri", "sr", "nocrop", True),
         ("R6", "head_{s}|shape", "factorized_{s}|shape", "sr", "nocrop", True),
         ("R7", "now_metrical", "now_original", "fr", "nocrop", False))
M_HOLM = len(RULES)
PER_PAIR = ("mm_rigid_icp_chamfer", "cs_rigid_icp_chamfer", "mm_nicp_p2tri", "cs_nicp_p2tri", "now_original",
            "now_metrical")
PROTOCOLS = ("now_original", "now_metrical", "realy", "stirling")
MESH_FAIL_MAX, ROW_MISS_MAX = 0.02, 0.05


def family() -> tuple[list, list]:
    """(colonne per mesh, colonne per coppia) della famiglia confermativa."""
    mesh = [f"factorized_{s}|{d}" for s in SEEDS for d in ("form_cal", "shape")] + \
        [f"head_{s}|shape" for s in SEEDS] + ["mm_nicp_template"]
    return mesh, list(PER_PAIR)


# ------------------------------------------------------------------------------------------ righe e colonne

def build_rows(s: dict, zes) -> dict:
    """Righe (sez. 5) sugli indici di ``zes.Index`` dei soggetti dello spec."""
    subj = sorted(s["subjects"])
    if sorted(s["topologies"]) != sorted(zes.TOPOLOGIES):
        raise SystemExit(f"topologie {s['topologies']} diverse da {zes.TOPOLOGIES}")
    idx = zes.Index(subj)
    n = len(subj)
    i, j = np.triu_indices(n, 1)
    sa, sb, ia, ib, ta, tb = [], [], [], [], [], []
    for x in zes.TOPOLOGIES:
        for y in zes.TOPOLOGIES:
            if x == y:
                continue
            sa.append(i), sb.append(j), ia.append(idx.rows(x)[i]), ib.append(idx.rows(y)[j])
            ta.append(np.full(len(i), x)), tb.append(np.full(len(i), y))
    R = {"idx": idx, "subjects": subj, **{k: np.concatenate(v) for k, v in
                                          (("sa", sa), ("sb", sb), ("ia", ia), ("ib", ib), ("ta", ta), ("tb", tb))}}
    if s.get("pairs"):                                   # sottocampione (collaudo): le coppie di conf_protocols
        import conf_protocols as cp
        import pr_common as pc
        items = pc.view(s["name"], Path(s["_path"]))["items"]
        pi, pj = cp.pairs_of(s, items)
        pos = np.asarray([idx.pos[(x, t)] for x, t, _ in items])
        sub = np.zeros((len(idx.keys), len(idx.keys)), bool)
        sub[pos[pi], pos[pj]] = sub[pos[pj], pos[pi]] = True
        keep = sub[R["ia"], R["ib"]]
        R.update({k: R[k][keep] for k in ("sa", "sb", "ia", "ib", "ta", "tb")})
        R["subsample_pairs"] = int(len(pi))
    return R


def gt_columns(s: dict, R: dict, zsum) -> dict:
    out = {}
    for g in ("fr", "sr"):
        D, pos = zsum.load_gt(cc.rp(s["gt"][g]))
        k = np.asarray([pos[x] for x in R["subjects"]])
        out[g] = D[k[R["sa"]], k[R["sb"]]]
    return out


def embeddings(path: Path, idx) -> tuple[np.ndarray, Path]:
    """Z sulle chiavi di idx (store con ``subjects``/``topologies`` o ``files``), checkpoint."""
    with np.load(path, allow_pickle=True) as z:
        if "files" in z.files:
            keys = [tuple(Path(str(f)).name[:-4].split("_GTready_", 1)) for f in z["files"]]
        else:
            keys = list(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))
        Z, ck = np.asarray(z["Z"], np.float64), Path(str(z["checkpoint"]))
    pos = {k: r for r, k in enumerate(keys)}
    miss = [k for k in idx.keys if k not in pos]
    if miss:
        raise SystemExit(f"{path}: mancano {len(miss)} mesh (es. {miss[:2]})")
    return Z[[pos[k] for k in idx.keys]], ck


def model_columns(s: dict, R: dict, fp) -> tuple[dict, dict, dict]:
    """d_F cal., d_F non cal., d_P per seme (``fact_paired.distances`` sulle righe); testa di R6; mesh fallite."""
    import conf_head as ch
    cols, failed, info = {}, {}, {}
    for sd in SEEDS:
        m = s["columns"]["models"][f"factorized_{sd}"]
        Z, ck = embeddings(cc.rp(m["embeddings"]), R["idx"])
        if ck.resolve() != cc.rp(m["checkpoint"]).resolve():
            raise SystemExit(f"{sd}: embedding col checkpoint {ck}, atteso {m['checkpoint']}")
        bad = ~np.isfinite(Z).all(1)
        d = fp.distances(Z, R["ia"], R["ib"], ck, (float(m["c"]), float(m["c"])))
        for k in ("form_cal", "form", "shape"):
            cols[f"factorized_{sd}|{k}"] = d[k]
            failed[f"factorized_{sd}|{k}"] = bad
        import diag
        dpu = diag.dp_per_unit(ck)
        info[sd] = {"embeddings": m["embeddings"], "c": float(m["c"]), "dp_per_unit": dpu, "meshes_failed": int(bad.sum())}
        if abs(dpu - float(s["columns"]["dp_per_unit"])) > 0:
            raise SystemExit(f"{sd}: dp_per_unit {dpu} diverso da quello dello spec")
        hp = (s["columns"].get("heads") or {}).get(f"factorized_{sd}")
        if hp:
            h, c, meta = ch.load_head(cc.rp(hp))
            if Path(meta["checkpoint"]).resolve() != ck.resolve() or meta["dpu"] != dpu:
                raise SystemExit(f"{sd}: testa con checkpoint o dp_per_unit diversi dagli embedding")
            X = Z[:, 1:] * dpu
            S_ = np.exp(Z[:, 0])
            dh = h.dist(X[R["ia"]], X[R["ib"]])
            Sa, Sb = S_[R["ia"]], S_[R["ib"]]
            cols[f"head_{sd}|shape"] = dh
            cols[f"head_{sd}|form"] = np.sqrt((Sa - Sb) ** 2 + Sa * Sb * (c * dh) ** 2)
            failed[f"head_{sd}|shape"] = failed[f"head_{sd}|form"] = bad
            info[sd]["head"] = {"path": hp, **meta}
    return cols, failed, info


def geo_and_protocols(s: dict, R: dict, fm, blmm) -> tuple[dict, dict, dict]:
    """Colonne di blmm (``fm_eval.geo_columns``) e dei protocolli (``fm_eval.plugin_columns``); mesh fallite del NICP
    su template."""
    name = blmm.register_spec(s["_path"])
    if s["columns"].get("blmm_root"):
        blmm.VIEWS[name]["out"] = cc.rp(s["columns"]["blmm_root"])
    G = fm.geo_columns(name, R["idx"])
    cols = {c: M[R["ia"], R["ib"]] for c, M in G.items()}
    failed = {c: ~np.isfinite(M).any(1) for c, M in G.items() if "template" in c}
    pr = s["columns"]["protocols"]
    dirs = [(m, cc.rp(pr["root"]) / m / pr.get("variant", "")) for m in PROTOCOLS]
    P, _, missing = fm.plugin_columns(pr.get("view", s["name"]), R["idx"].keys, dirs)
    for m in PROTOCOLS:
        if f"plugin:{m}|mm" in P:
            cols[m] = P[f"plugin:{m}|mm"][R["ia"], R["ib"]]
    info = {"blmm_root": str(blmm.view_root(name)), "geo": sorted(G), "protocols": {"dirs": [str(d) for _, d in dirs],
                                                                                    "missing": missing}}
    return cols, failed, info


def fallback_of(s: dict) -> dict | None:
    """{topologia: mesh col ripiego del detector} da ``conf_protocols.py detprep`` (None: landmark esatti, collaudo)."""
    p = s["columns"]["protocols"].get("fallback")
    if not p:
        return None
    return json.loads(cc.rp(p).read_text())["det_fallback"]


# ------------------------------------------------------------------------------------------ non risolto

def unresolved(col: str, gname: str, R: dict, X: dict, failed: dict, fb: dict | None, topos) -> list:
    """Motivi della sez. 8 / 9.1 per la colonna nel gruppo (lista vuota = risolta)."""
    from conf_protocols import FALLBACK_MAX
    gt_ = [t for t in topos if gname != "nocrop" or t != "crop"]
    rows = group_rows(R, gname, np.ones(len(R["sa"]), bool))
    why = []
    miss = float((~np.isfinite(X[col][rows])).mean())
    if miss > ROW_MISS_MAX:
        why.append(f"righe non finite {miss:.2%} > {ROW_MISS_MAX:.0%}")
    if col in failed:
        k = np.asarray([t in gt_ for _, t in R["idx"].keys])
        fr = float(failed[col][k].mean())
        if fr > MESH_FAIL_MAX:
            why.append(f"mesh fallite {fr:.2%} > {MESH_FAIL_MAX:.0%}")
    if col in PROTOCOLS and fb is not None:
        f = sum(fb.get(t, 0) for t in gt_) / (len(gt_) * len(R["subjects"]))
        if f > FALLBACK_MAX:
            why.append(f"ripieghi del detector {f:.2%} > {FALLBACK_MAX:.0%}")
    return why


def group_rows(R: dict, gname: str, mask: np.ndarray) -> np.ndarray:
    crop = (R["ta"] == "crop") | (R["tb"] == "crop")
    cross = (R["ta"] != R["tb"]) & mask
    return np.flatnonzero({"nocrop": cross & ~crop, "crop": cross & crop, "all_cross": cross}[gname])


# ------------------------------------------------------------------------------------------ letture

def reading(X: dict, gt: dict, R: dict, mask: np.ndarray, methods: list, counts, workers: int, fm) -> tuple[dict, dict]:
    groups = {(g, "-"): group_rows(R, g, mask) for g in GROUPS}
    V = fm.replicas({m: X[m] for m in methods}, gt, R["sa"], R["sb"], counts, groups, methods, workers)
    return {g: V[:, k] for k, g in enumerate(GROUPS)}, {g: len(r) for g, r in zip(GROUPS, groups.values())}


def delta(V: dict, mi: dict, g: str, gt: str, a: str, b: str) -> dict:
    ti = ("fr", "sr").index(gt)
    d = V[g][:, ti, mi[a]] - V[g][:, ti, mi[b]]
    bb = d[1:][np.isfinite(d[1:])]
    lo, hi = np.percentile(bb, [2.5, 97.5]) if len(bb) else (np.nan, np.nan)
    return {"rho_a": float(V[g][0, ti, mi[a]]), "rho_b": float(V[g][0, ti, mi[b]]), "delta": float(d[0]),
            "ic_lo": float(lo), "ic_hi": float(hi), "k_le0": int((bb <= 0).sum()), "n_boot": int(len(bb)),
            "p_le0": float((bb <= 0).mean()) if len(bb) else np.nan}


def holm(ps: list) -> dict:
    """{regola: (posizione i, soglia, rifiutata)} sulle (regola, k, n) di una famiglia; regola intera 40 k (m-i+1) <= n."""
    order = sorted(range(len(ps)), key=lambda q: ps[q][1] / ps[q][2])        # stabile: a parita' l'ordine delle regole
    out, going = {}, True
    for i, q in enumerate(order, 1):
        rule, k, n = ps[q]
        ok = going and 40 * k * (M_HOLM - i + 1) <= n
        going = ok
        out[rule] = (i, 0.025 / (M_HOLM - i + 1), ok)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    t0 = time.time()
    s = cc.load(a.spec)
    if not s.get("collaudo") and not s.get("freeze"):
        raise SystemExit("spec senza freeze e non di collaudo")
    ver = cc.verify(s)                                              # PRIMA di leggere qualunque file del dominio
    import fm_eval as fm                                            # fact_paired, blmm_eval, blmm in sola lettura
    fp, be, blmm = fm.fp, fm.be, fm.blmm
    R = build_rows(s, be.zes)
    gt = gt_columns(s, R, be.zsum)
    X, failed, info_m = model_columns(s, R, fp)
    Xg, fg, info_g = geo_and_protocols(s, R, fm, blmm)
    X.update(Xg)
    failed.update(fg)
    fb = fallback_of(s)
    n = len(R["subjects"])
    rng = np.random.default_rng(int(s["boot_seed"]))
    counts = [np.ones(n, dtype=np.int64)] + [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(a.n_boot)]
    mesh_cols, pair_cols = family()
    absent = [c for c in mesh_cols + pair_cols if c not in X]
    if absent and not s.get("collaudo"):
        raise SystemExit(f"colonne della famiglia assenti: {absent}")
    mesh_cols = [c for c in mesh_cols if c in X]
    pair_cols = [c for c in pair_cols if c in X]
    gok = np.isfinite(gt["fr"]) & np.isfinite(gt["sr"])
    mask_c = gok & np.all([np.isfinite(X[c]) for c in mesh_cols], axis=0)
    mask_m = mask_c & np.all([np.isfinite(X[c]) for c in pair_cols], axis=0)
    Xc = {c: (np.where(np.isfinite(v), v, np.inf) if c in pair_cols else v) for c, v in X.items()}
    extra = [c for c in X if c not in mesh_cols + pair_cols and np.isfinite(X[c][mask_c]).all()]   # descrittivi pieni
    meth = mesh_cols + pair_cols + extra
    mi = {m: k for k, m in enumerate(meth)}
    Vc, nc = reading(Xc, gt, R, mask_c, meth, counts, a.workers, fm)
    Vm, nm = reading(X, gt, R, mask_m, mesh_cols + pair_cols, counts, a.workers, fm)
    holes = {}                                                       # descrittivi con buchi: passo sulle loro righe
    for c in X:
        if c not in meth and np.isfinite(X[c][mask_c]).any():
            V2, n2 = reading(X, gt, R, mask_c & np.isfinite(X[c]), [c] + mesh_cols[:4], counts, a.workers, fm)
            holes[c] = (V2, n2)
    topos = list(be.zes.TOPOLOGIES)
    # ---------------------------------------------------------------- regole
    recs, dl = [], []
    for rule, ra, rb, g_, grp, per_seed in RULES:
        for sd in (SEEDS if per_seed else ("-",)):
            ca, cb = ra.format(s=sd), rb.format(s=sd)
            base = {"regola": rule, "seme": sd, "gt": g_, "gruppo": grp, "colonna_a": ca, "colonna_b": cb}
            if ca not in mi or cb not in mi:
                recs.append({**base, "assente": True, "k_le0": a.n_boot, "n_boot": a.n_boot, "p_le0": 1.0,
                             "non_risolto": "", "nota": "colonna assente dallo spec (solo collaudo): p = 1"})
                continue
            d = delta(Vc, mi, grp, g_, ca, cb)
            why = unresolved(ca, grp, R, X, failed, fb, topos) + unresolved(cb, grp, R, X, failed, fb, topos)
            nota = ""
            if ca.startswith("head_") and not info_m[sd]["head"]["success"]:
                nota = "fit della testa non convergente (success = False): resta nella famiglia col suo p"
            recs.append({**base, "assente": False, **d, "n_righe": nc[grp], "non_risolto": "; ".join(why), "nota": nota})
            dm = delta(Vm, {m: k for k, m in enumerate(mesh_cols + pair_cols)}, grp, g_, ca, cb)
            dl += [{"lettura": "confermativa (+inf)", **base, **d, "n_righe": nc[grp], "non_risolto": "; ".join(why)},
                   {"lettura": "mascherata (descrittiva)", **base, **dm, "n_righe": nm[grp]}]
    rules = pd.DataFrame(recs)
    for sd in SEEDS:                                                 # Holm per seme
        fam = rules[(rules["seme"] == sd) | (rules["seme"] == "-")]
        if len(fam) != M_HOLM:
            raise SystemExit(f"famiglia {sd}: {len(fam)} ipotesi invece di {M_HOLM}")
        h = holm([(r.regola, int(r.k_le0), int(r.n_boot)) for r in fam.itertuples()])
        for rule, (i, thr, rej) in h.items():
            for q in rules.index[(rules["regola"] == rule) & rules["seme"].isin([sd, "-"])]:
                rules.loc[q, f"holm_i_{sd}"], rules.loc[q, f"holm_soglia_{sd}"] = i, thr
                rules.loc[q, f"holm_rifiuta_{sd}"] = rej
    esiti = {}
    for rule, *_rest in RULES:
        x = rules[rules["regola"] == rule]
        if x["assente"].any():
            esiti[rule] = "non calcolata (collaudo)"
            continue
        if (x["non_risolto"] != "").any():
            esiti[rule] = "non risolta"
            continue
        if (x["seme"] == "-").all():
            ok = bool(x["holm_rifiuta_s1234"].iloc[0]) and bool(x["holm_rifiuta_s2345"].iloc[0]) and x["delta"].iloc[0] > 0
        else:
            ok = all(bool(x[x["seme"] == sd][f"holm_rifiuta_{sd}"].iloc[0]) and x[x["seme"] == sd]["delta"].iloc[0] > 0
                     for sd in SEEDS)
        esiti[rule] = "confermata" if ok else "non confermata"
    rules["esito"] = rules["regola"].map(esiti)
    # ---------------------------------------------------------------- descrittivi
    sp = []
    for lab, V, ns, cols_ in (("confermativa (+inf)", Vc, nc, meth), ("mascherata (descrittiva)", Vm, nm,
                                                                       mesh_cols + pair_cols)):
        for g in GROUPS:
            for ti, gt_ in enumerate(("fr", "sr")):
                for k, c in enumerate(cols_):
                    x = V[g][:, ti, k]
                    b = x[1:][np.isfinite(x[1:])]
                    sp.append({"lettura": lab, "gruppo": g, "gt": gt_, "colonna": c, "rho": float(x[0]),
                               "ic_lo": float(np.percentile(b, 2.5)), "ic_hi": float(np.percentile(b, 97.5)),
                               "n_righe": ns[g], "non_risolto": "; ".join(unresolved(c, g, R, X, failed, fb, topos))})
    for c, (V2, n2) in holes.items():
        for g in GROUPS:
            for ti, gt_ in enumerate(("fr", "sr")):
                x = V2[g][:, ti, 0]
                b = x[1:][np.isfinite(x[1:])]
                sp.append({"lettura": "righe proprie (descrittiva)", "gruppo": g, "gt": gt_, "colonna": c,
                           "rho": float(x[0]), "ic_lo": float(np.percentile(b, 2.5)) if len(b) else np.nan,
                           "ic_hi": float(np.percentile(b, 97.5)) if len(b) else np.nan, "n_righe": n2[g],
                           "non_risolto": "; ".join(unresolved(c, g, R, X, failed, fb, topos))})
    for sd in SEEDS:                                                 # sez. 9.2: NoW metrical - d_F, IC bilaterale
        for ref in (f"factorized_{sd}|form_cal", f"factorized_{sd}|form"):
            if "now_metrical" in mi and ref in mi:
                for g in GROUPS:
                    why = unresolved("now_metrical", g, R, X, failed, fb, topos) + unresolved(ref, g, R, X, failed, fb,
                                                                                            topos)
                    dl.append({"lettura": "sez. 9.2 (descrittiva)", "regola": "NoW metrical - d_F", "seme": sd,
                               "gt": "fr", "gruppo": g, "colonna_a": "now_metrical", "colonna_b": ref,
                               **delta(Vc, mi, g, "fr", "now_metrical", ref), "n_righe": nc[g],
                               "non_risolto": "; ".join(why)})
    out = cc.evidence(s) / "conf"
    out.mkdir(parents=True, exist_ok=True)
    rules.to_csv(out / "rules.csv", index=False)
    pd.DataFrame(dl).to_csv(out / "deltas.csv", index=False)
    pd.DataFrame(sp).to_csv(out / "spearman.csv", index=False)
    ctrl = {"spec": s["_path"], "verify": ver, "boot_seed": int(s["boot_seed"]), "n_boot": a.n_boot,
            "subjects": n, "rows": int(len(R["sa"])), "rows_mask_confirmatory": int(mask_c.sum()),
            "rows_mask_masked": int(mask_m.sum()), "rows_by_group_confirmatory": nc, "rows_by_group_masked": nm,
            "rows_nonfinite_by_column": {c: int((~np.isfinite(v)).sum()) for c, v in X.items()},
            "meshes_failed_by_column": {c: int(v.sum()) for c, v in failed.items()}, "fallback": fb,
            "subsample_pairs": R.get("subsample_pairs"), "models": info_m, "geo_protocols": info_g,
            "absent_family_columns": absent, "esiti": esiti, "seconds": time.time() - t0, **cc.stamp()}
    cc.atomic_json(out / "controls.json", ctrl)
    write_results(out, rules, ctrl)
    print(f"[conf-eval] {s['name']}: {json.dumps(esiti)} ({time.time() - t0:.0f}s)", flush=True)


def write_results(out: Path, rules: pd.DataFrame, ctrl: dict) -> None:
    f3 = lambda x, lo, hi: f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]"      # noqa: E731
    L = ["# Prova confermativa: regole R1-R7", "", f"Generato da `aau/ava256/conf_eval.py` (spec `{ctrl['spec']}`). "
         f"Seme delle repliche {ctrl['boot_seed']}, {ctrl['n_boot']} repliche, {ctrl['subjects']} soggetti, righe "
         f"{ctrl['rows']} (lettura confermativa: {ctrl['rows_by_group_confirmatory']}). Lettura confermativa: righe "
         "fallite dei metodi per coppia = +inf; Holm per seme con m = 10.", "",
         "| regola | seme | delta [IC 95%] | k (p) | Holm i / soglia / rifiuta | non risolto | esito |",
         "|---|---|---|---|---|---|---|"]
    for r in rules.itertuples():
        if r.assente:
            L.append(f"| {r.regola} | {r.seme} | - | - | - | - | {r.esito} |")
            continue
        hs = [f"{int(getattr(r, f'holm_i_{sd}'))} / {getattr(r, f'holm_soglia_{sd}'):.4f} / "
              f"{'si' if getattr(r, f'holm_rifiuta_{sd}') else 'no'}" for sd in SEEDS
              if r.seme in (sd, "-") and not pd.isna(getattr(r, f"holm_i_{sd}"))]
        L.append(f"| {r.regola} | {r.seme} | {f3(r.delta, r.ic_lo, r.ic_hi)} | {int(r.k_le0)} ({r.p_le0:.3f}) | "
                 f"{'; '.join(hs)} | {r.non_risolto or '-'} | {r.esito} |")
    L += ["", "Delta della versione mascherata, NoW metrical contro d_F (sez. 9.2) e Spearman dei protocolli: "
          "`deltas.csv`, `spearman.csv` (descrittivi, non si leggono).", ""]
    (out / "results.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
