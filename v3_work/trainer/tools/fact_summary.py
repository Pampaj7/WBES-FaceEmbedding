#!/usr/bin/env python3
"""Riepilogo dei run di factorized_protocol.md (factorized, factorized2, ctrlfr; s1234, s2345; passi 10.548 e 21.096)
accanto alle baseline in mm (aau/runs/evidence/baselines_mm) e a e108, con le stesse colonne.

    aau/run.sh v3_work/trainer/tools/fact_summary.py

Dagli embedding dei bracci (passo ``form`` di ablations/c3f/eval_body.sh, ``c3f_eval/form_*``):
  * graduata, livello coppia di mesh senza crop e cross-topologia (``nocrop_cross`` di E12), GT FR, SR, maxabs, IC
    bootstrap per soggetto (tools/eval_factorized.boot_rows, 1000 repliche, seme 1234): HIFI3D (primaria, FR),
    dev FaceScape neutra, FaceVerse con espressioni. Distanza: d_F per i fattorizzati (anche d_P), ||z|| per ctrlfr;
  * rank-1 senza crop (zs_expr_summarize: retrieval fra topologie diverse, funzioni importate): HIFI3D, dev FaceScape
    con espressioni, FaceVerse con espressioni; la distanza e' la stessa della graduata;
  * punteggio dev (come il protocollo delle ablazioni, qui dagli embedding): media di graduata maxabs della neutra e
    rank-1 con espressioni, dev FaceScape;
  * FaMoS TEST (``c3f_eval/famos/<tag>``) e NoW tau (``c3f_eval/now/<tag>``) se presenti.
Emendamento 4: d_F calibrata (``form_cal``, c di factorized_calibration.csv, tools/fact_calib.py; ``form_cal_ls`` con
c dei minimi quadrati) per i fattorizzati all'ultimo checkpoint e per C3M (e123, e205, descrittivo); etichette di
verdetto dai delta appaiati contro le baseline in mm (factorized_paired.csv, tools/fact_paired.py); analisi della
taglia (a)-(c), regola "forma oltre la taglia" e regola dual dalle stesse righe. Emendamento 5: composizioni con k
degli held-out (factorized_calibration_bl.csv), SR contro NICP e ICP cs, sezione esplorativa NON preregistrata dopo
quelle preregistrate (factorized_explore.csv, factorized_calibration_test.csv di fact_paired.py). Scrive
aau/runs/evidence/trainer_v3/factorized_results.md e factorized_results.csv.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
for _p in (TRAINER, THIS, REPO / "aau/zs3dmm"):
    sys.path.insert(0, str(_p))

import eval_factorized as ef  # noqa: E402
import fact_calib  # noqa: E402
import factorized_v3 as fz  # noqa: E402

EV = REPO / "aau/runs/evidence/trainer_v3"
EVAL = EV / "ablations/c3f_eval"
RUNS = EV / "ablations/c3f_runs"
BL = REPO / "aau/runs/evidence/baselines_mm"
G = REPO / "datasets/CANONICAL_GT/eval"
ARMS = ("factorized", "factorized2", "ctrlfr", "dual")
SEEDS = (1234, 2345)
EPOCHS = ("036", "072")
STEPS = {"036": 10548, "072": 21096}
DOMS = {"hifi": ("form_hifi", "hifi3d", "HIFI3D/eval_view"), "devfs": ("form_devfs", "facescape", "DEV_FACESCAPE/eval_view"),
        "devfs_expr": ("form_devfs_expr", "facescape", "DEV_FACESCAPE/expr_view"),
        "fv": ("form_fv_expr", "faceverse", "FACEVERSE_ZS/expr_view")}
# C3M (descrittivo): (epoca, passi); 293 passi per epoca, 60.000 in tutto
C3M = (("123", 36039), ("205", 60000))
NB, SEED = 1000, 1234
MARGIN = 0.03                       # emendamento 4: non inferiorita' sull'estremo inferiore dell'IC del delta
COMP_EST = "taglia stimata + ICP cs cal."      # emendamento 5: concorrente della condizione (c)
VERDICT_BL = ("ICP + Chamfer in mm", "NICP su template in mm")
BL_ROWS = (("scale_e108", "e108 (cieco alla taglia)"), ("mm_rigid_icp_chamfer", "ICP + Chamfer in mm"),
           ("mm_nicp_p2tri", "NICP per coppia in mm"), ("mm_nicp_template", "NICP su template in mm"),
           ("cs_nicp_p2tri", "NICP per coppia, modo cs"), ("est_cs", "taglia stimata (non oracolo)"),
           ("oracle_size", "taglia oracolo"), ("chamfer_eval", "Chamfer eval"))
BL_GROUP = {"hifi": "nocrop_cross", "devfs": "nocrop_cross", "fv": "mesh_pair_nocrop"}
BL_DOMAIN = {"hifi": "hifi3d", "devfs": "facescape", "fv": "faceverse"}


def vname(arm: str, seed: int) -> str:
    return arm + ("" if seed == 1234 else f"s{seed}")


def calib_key(arm: str, seed: int, e: str) -> str:
    """Chiave di fact_calib.CKPTS (e prefisso delle colonne di fact_paired)."""
    return f"factorizedc3m_e{e}" if arm == "factorizedc3m" else f"{arm}_s{seed}"


def embeddings(dom: str, arm: str, seed: int, e: str) -> Path | None:
    root = EVAL / DOMS[dom][0]
    hits = sorted(root.glob(f"data_*/scale_v3{vname(arm, seed)}fulle{e}*/zs_zeroshot/embeddings.npz"))
    return hits[0] if hits else None


def load_emb(path: Path):
    with np.load(path, allow_pickle=True) as z:
        return (np.asarray(z["Z"], np.float64), [str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]],
                Path(str(z["checkpoint"])))


def model_D(Z: np.ndarray, i: np.ndarray, j: np.ndarray, head: str, dpu: float, cal: dict | None = None) -> dict:
    """fz.model_distances; con ``cal`` (riga di factorized_calibration.csv) anche d_F calibrata, c primaria e LS."""
    out = fz.model_distances(Z, i, j, head, dpu)
    if cal is not None and head in fz.FACTORIZED:
        S = np.exp(np.asarray(Z, np.float64)[:, 0])
        for k, c in (("form_cal", cal["c_median"]), ("form_cal_ls", cal["c_ls"])):
            out[k] = fz.form_distance(S[i], S[j], c * out["shape"])
    return out


def head_of(ckpt: Path):
    import torch
    head = torch.load(ckpt, map_location="cpu", weights_only=False)["args"].get("head", "embed")
    return head, (ef.dp_from_ckpt(ckpt) if head in fz.FACTORIZED else float("nan"))


FORM_DIST = {"factorized": "form", "factorized2": "form", "ctrlfr": "z", "dual": "zf", "factorizedc3m": "form"}
SHAPE_DIST = {"factorized": "shape", "factorized2": "shape", "ctrlfr": "z", "dual": "u", "factorizedc3m": "shape"}
NOW_DIST = {"factorized": (("", "u"),), "factorized2": (("", "u"),), "dual": (("", "u"), ("_zf", "zf")),
            "factorizedc3m": (("", "u"),)}
# distanza form di riferimento dopo l'emendamento 4: d_F calibrata per i fattorizzati
REF_DIST = {"factorized": "form_cal", "factorized2": "form_cal", "ctrlfr": "z", "dual": "zf", "factorizedc3m": "form_cal"}


def graded(dom: str, path: Path, cal: dict | None = None) -> list[dict]:
    Z, subj, topo, ckpt = load_emb(path)
    head, dpu = head_of(ckpt)
    subjects = sorted(set(subj))
    pos = {s: k for k, s in enumerate(subjects)}
    i, j = np.triu_indices(len(Z), 1)
    si, sj = np.asarray([pos[s] for s in subj])[i], np.asarray([pos[s] for s in subj])[j]
    ti, tj = np.asarray(topo)[i], np.asarray(topo)[j]
    keep = (si != sj) & (ti != tj) & (ti != "crop") & (tj != "crop")
    i, j, si, sj = i[keep], j[keep], si[keep], sj[keep]
    _, set_name, view = DOMS[dom]
    gts = {"fr": ef.load_gt(G / f"{set_name}_fr.npz", subjects), "sr": ef.load_gt(G / f"{set_name}_sr.npz", subjects),
           "maxabs": ef.load_gt(REPO / "datasets" / view / "gt_matrix.npz", subjects)}
    return ef.boot_rows(model_D(Z, i, j, head, dpu, cal), gts, si, sj, len(subjects), NB, SEED)


def rank1(dom: str, path: Path, cal: dict | None = None) -> dict:
    import zs_expr_summarize as zes
    from zs_stage import TOPOLOGIES
    Z, subj, topo, ckpt = load_emb(path)
    head, dpu = head_of(ckpt)
    idx = zes.Index(sorted(set(subj)))
    order = np.asarray([list(zip(subj, topo)).index(k) for k in idx.keys])
    Z = Z[order]
    n = len(Z)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    Ds = model_D(Z, i.ravel(), j.ravel(), head, dpu, cal)
    nocrop = [t for t in TOPOLOGIES if t != "crop"]
    pairs = [(a, b) for a in nocrop for b in nocrop if a != b]
    counts = zes.bootstrap_counts(len(idx.subjects), NB, SEED)
    out = {}
    for k, d in Ds.items():
        D = d.reshape(n, n)
        v = zes.recognition_values(zes.retrieval_queries(D, idx, pairs), zes.verification_pairs(D, idx, pairs), idx, counts)
        out[k] = {"rank1": v["rank1"][0], "rank1_ci": [float(x) for x in np.percentile(v["rank1"][1:], [2.5, 97.5])],
                  "auc": v["auc"][0]}
    return out


def famos(arm: str, seed: int, e: str) -> list[dict]:
    p = EVAL / "famos" / f"v3{vname(arm, seed)}e{e}" / "graded.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def now_tau(arm: str, seed: int, e: str, sfx: str = ""):
    p = EVAL / "now" / f"v3{vname(arm, seed)}e{e}{sfx}" / "concordance.csv"
    if not p.exists():
        return None
    for r in csv.DictReader(open(p)):
        if r["metric"] == "latent_joint" and r["methods"] == "3ddfa_v2+synergynet+prnet":
            return float(r["tau_image"]), float(r["tau_image_ci_low"]), float(r["tau_image_ci_high"])
    return None


def baselines() -> list[dict]:
    rows = list(csv.DictReader(open(BL / "spearman.csv")))
    out = []
    for dom in ("hifi", "devfs", "fv"):
        for m, lab in BL_ROWS:
            for gt in ("fr", "sr", "maxabs"):
                hit = [r for r in rows if r["domain"] == BL_DOMAIN[dom] and r["group"] == BL_GROUP[dom]
                       and r["method"] == m and r["gt"] == gt]
                if hit:
                    r = hit[0]
                    out.append({"dom": dom, "method": lab, "gt": gt, "point": float(r["point"]),
                                "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"])})
    return out


def fmt(p, lo, hi):
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def runs():
    """(braccio, seme, epoca, passi): i bracci C3F a 10.548 e 21.096 passi, poi C3M (descrittivo)."""
    for arm in ARMS:
        for seed in SEEDS:
            for e in EPOCHS:
                yield arm, seed, e, STEPS[e]
    for e, steps in C3M:
        yield "factorizedc3m", 1234, e, steps


def famos_dist(method: str, arm: str, seed: int, e: str) -> str:
    """Nome della distanza di una riga di graded.csv di eval_famos_v3 (``v3<tag>full[_<d>]`` -> d)."""
    pre = f"v3{vname(arm, seed)}e{e}full"
    if method == pre:
        return "z"
    return method[len(pre) + 1:] if method.startswith(pre + "_") else method


def dlab(r: dict) -> str:
    return f"{float(r['delta']):+.3f} [{float(r['ci_low']):+.3f}, {float(r['ci_high']):+.3f}]"


def verdict(P: list, key: str, dist: str) -> str:
    """Emendamento 4, sez. 4: sopra / pari / sotto le baseline in mm (delta appaiato su HIFI3D FR)."""
    out = []
    for b in VERDICT_BL:
        hit = [r for r in P if r["domain"] == "hifi3d" and r["gt"] == "fr" and r["kind"] == "rho" and r["arm"] == key
               and r["distance"] == dist and r["baseline"] == b]
        if hit:
            lo, hi = float(hit[0]["ci_low"]), float(hit[0]["ci_high"])
            out.append(f"{'sopra' if lo > 0 else 'sotto' if hi < 0 else 'pari a'} {b.replace('NICP su template', 'NICP tpl')} "
                       f"({dlab(hit[0])})")
    return "; ".join(out) or "-"


def explore_section(P: list, cal: dict, kbl: dict) -> list[str]:
    """Emendamento 5, sez. 3: esplorativa NON preregistrata, dopo le sezioni preregistrate. (i) parziale (a) della
    sola d_P (factorized_paired.csv); (ii) d_F(c) sulla griglia di c (factorized_explore.csv); (iii) c e k "ideali"
    sul test (factorized_calibration_test.csv)."""
    md = ["", "## Esplorativa, non preregistrata (motivata dalla dipendenza di (a) da c)", "",
          "Analisi post hoc (emendamento 5, sez. 3), dopo i numeri dell'emendamento 4: non cambia alcun verdetto. "
          "HIFI3D `nocrop_cross`, GT FR, righe e repliche di `fact_paired.py`.", ""]
    f = lambda r: fmt(float(r["arm_point"]), float(r["arm_ci_low"]), float(r["arm_ci_high"])) if r else "-"  # noqa: E731
    g = lambda r: dlab(r) if r else "-"  # noqa: E731

    def get(rows, key, dist, kind, b):
        hit = [r for r in rows if r["domain"] == "hifi3d" and r["gt"] == "fr" and r["kind"] == kind and r["arm"] == key
               and r["distance"] == dist and r["baseline"] == b]
        return hit[0] if hit else None

    # (i)
    md += ["### (i) Parziale (a) della sola d_P", "",
           "| braccio | distanza | (a) parziale | - ICP mm | - NICP cs | - ICP cs |", "| --- | --- | --- | --- | --- | --- |"]
    for key, d in [(f"{a}_s{s}", "shape") for a in ("factorized", "factorized2") for s in SEEDS] + \
            [(f"factorizedc3m_e{e}", "shape") for e, _ in C3M] + [(f"dual_s{s}", "u") for s in SEEDS]:
        v = get(P, key, d, "partial", "-")
        if v:
            md.append(f"| {key} | {d} | {f(v)} | " + " | ".join(
                g(get(P, key, d, "partial", b)) for b in ("ICP + Chamfer in mm", "NICP per coppia (cs)", "ICP + Chamfer (cs)"))
                + " |")
    for lab in ("ICP + Chamfer in mm", "ICP + Chamfer (cs)", "NICP per coppia (cs)"):
        v = get(P, "baseline", lab, "partial", "-")
        if v:
            md.append(f"| {lab} (baseline) | - | {f(v)} | - | - | - |")
    # (ii)
    xp = EV / "factorized_explore.csv"
    X = list(csv.DictReader(open(xp))) if xp.exists() else []
    md += ["", "### (ii) Sensibilita' a c di d_F(c) = sqrt((S_i - S_j)^2 + S_i S_j (c d_P)^2)", "",
           f"Criterio a c: le condizioni della regola \"forma oltre la taglia\" calcolate con d_F(c) (descrittivo). "
           "`*` = c del checkpoint (held-out, arrotondata alla griglia piu' vicina). `factorized_explore.csv`.", "",
           "| braccio | c | (a) parziale | (a) - ICP mm | (a) - NICP cs | Spearman FR | - stimata + ICP cs cal. | "
           "criterio a c |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    xkeys = list(dict.fromkeys((r["arm"], r["distance"]) for r in X if r["domain"] == "hifi3d"))
    crit_c = {}
    for key, d in xkeys:
        c = float(d.split("form_at_c")[1])
        grid = sorted({float(dd.split("form_at_c")[1]) for kk, dd in xkeys if kk == key})
        own = cal.get(key, {}).get("c_median")
        star = "*" if own is not None and c == min(grid, key=lambda x: abs(x - own)) else ""
        a, b1, b2 = (get(X, key, d, "partial", b) for b in ("-", "ICP + Chamfer in mm", "NICP per coppia (cs)"))
        r0, r1 = get(X, key, d, "rho", "-"), get(X, key, d, "rho", COMP_EST)
        ok = None if not (b1 and b2 and r1) else (float(b1["ci_low"]) > 0 and float(b2["ci_low"]) > 0
                                                   and float(r1["ci_low"]) > -MARGIN)
        crit_c[(key, c)] = ok
        md.append(f"| {key} | {c:g}{star} | {f(a)} | {g(b1)} | {g(b2)} | {f(r0)} | {g(r1)} | "
                  f"{'-' if ok is None else ('soddisfatto' if ok else 'non soddisfatto')} |")
    if not X:
        md.append("| in attesa | - | - | - | - | - | - | - |")
    # (iii)
    tp = EV / "factorized_calibration_test.csv"
    T = list(csv.DictReader(open(tp))) if tp.exists() else []
    md += ["", "### (iii) Errore di dominio della calibrazione (informazione, non parametro)", "",
           "Scala \"ideale\" sul test = mediana(d_P GT) / mediana(d_P del metodo) sulle righe della maschera (d_P GT = "
           "GT-SR x dP_per_unit), contro c (modelli) e k (baseline cs) degli held-out sintetici. Non si usa per alcuna "
           "distanza.", "", "| metodo | held-out | HIFI3D | rapporto | dev FaceScape | rapporto |",
           "| --- | --- | --- | --- | --- | --- |"]
    for m in dict.fromkeys(r["method"] for r in T):
        key = m.split(" ")[0]
        ho = {"cs_rigid_icp_chamfer": kbl.get("icp_cs", {}).get("k_median"),
              "cs_nicp_p2tri": kbl.get("nicp_cs", {}).get("k_median")}.get(m, cal.get(key, {}).get("c_median"))
        ho = float(ho) if ho not in (None, "") else None
        cells = []
        for dom in ("hifi3d", "facescape"):
            hit = [r for r in T if r["method"] == m and r["domain"] == dom]
            v = float(hit[0]["scale_test"]) if hit else None
            cells += [f"{v:.3f}" if v is not None else "-", f"{v / ho:.2f}" if v is not None and ho else "-"]
        md.append(f"| {m} | {f'{ho:.3f}' if ho else '-'} | " + " | ".join(cells) + " |")
    if not T:
        md.append("| in attesa | - | - | - | - | - |")
    for key in dict.fromkeys(k for k, _ in xkeys):
        hit = [r for r in T if r["method"] == f"{key} shape" and r["domain"] == "hifi3d"]
        if hit:
            ct = float(hit[0]["scale_test"])
            c = min((cc for kk, cc in crit_c if kk == key), key=lambda x: abs(x - ct))
            ok = crit_c[(key, c)]
            md.append(f"\n{key}: col c della griglia piu' vicino al c sul test di HIFI3D ({ct:.3f} -> c = {c:g}) il "
                      f"criterio e' {'-' if ok is None else ('soddisfatto' if ok else 'non soddisfatto')} (tabella (ii)).")
    return md


def main() -> None:
    rows, missing, famos_ch = [], [], {}
    cal = fact_calib.load()
    for arm, seed, e, steps in runs():
        tag = f"{arm} s{seed} {steps}"
        c = cal.get(calib_key(arm, seed, e)) if e == "072" or arm == "factorizedc3m" else None
        for dom in ("hifi", "devfs", "fv"):
            p = embeddings(dom, arm, seed, e)
            if p is None:
                missing.append(f"{tag} {dom}")
                continue
            for r in graded(dom, p, c):
                if r["level"] == "mesh_pair":
                    rows.append({"arm": arm, "seed": seed, "steps": steps, "dom": dom, "kind": "graded", **r})
        for dom in ("hifi", "devfs_expr", "fv"):
            p = embeddings(dom, arm, seed, e)
            if p is not None:
                for k, v in rank1(dom, p, c).items():
                    rows.append({"arm": arm, "seed": seed, "steps": steps, "dom": dom, "kind": "rank1",
                                 "distance": k, "gt": "-", "point": v["rank1"], "ci_low": v["rank1_ci"][0],
                                 "ci_high": v["rank1_ci"][1]})
        for r in famos(arm, seed, e):
            # chamfer_full / chamfer_stable: Chamfer di aau/famos/famos_eval.py sulla patch, NON mm_chamfer delle
            # baseline (emendamento 4, sez. 3): fuori dalle righe del braccio, una sola volta piu' sotto
            if r["method"] in ("chamfer_full", "chamfer_stable"):
                famos_ch.setdefault((r["method"], r["gt"], r["block"]), r)
                continue
            rows.append({"arm": arm, "seed": seed, "steps": steps, "dom": "famos", "kind": f"graded {r['block']}",
                         "distance": famos_dist(r["method"], arm, seed, e), "gt": r["gt"],
                         "point": float(r["spearman"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"])})
        # NoW: u per i fattorizzati, z per ctrlfr; dual su u (now/<tag>) e su z_F (now/<tag>_zf)
        for sfx, dist in NOW_DIST.get(arm, (("", "z"),)):
            t = now_tau(arm, seed, e, sfx)
            if t:
                rows.append({"arm": arm, "seed": seed, "steps": steps, "dom": "now", "kind": "tau",
                             "distance": dist, "gt": "-", "point": t[0], "ci_low": t[1], "ci_high": t[2]})
    keys = ["arm", "seed", "steps", "dom", "kind", "distance", "gt", "point", "ci_low", "ci_high"]
    with open(EV / "factorized_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    bl = baselines()
    pp = EV / "factorized_paired.csv"
    P = list(csv.DictReader(open(pp))) if pp.exists() else []
    md = ["# Risultati di factorized_protocol.md (emendamenti 1-5)", "",
          "Generato da `v3_work/trainer/tools/fact_summary.py`. Pesi EMA. IC 95% bootstrap per soggetto (1000 repliche, "
          "seme 1234). Graduata: livello coppia di mesh, senza crop, topologie diverse (il `nocrop_cross` di E12), "
          "calcolata dagli embedding. Distanze del modello: d_F grezza (`form`), d_F calibrata (`form_cal`, c primaria "
          "dell'emendamento 4; `form_cal_ls` con c dei minimi quadrati) e d_P (`shape`) per factorized/factorized2/C3M, "
          "||z|| per ctrlfr, ||z_F|| (FR) e ||u|| (SR) per dual. C3M (factorized sul run su scala, un seme) e' "
          "descrittivo, fuori dalle regole.", ""]
    last = [r for r in rows if r["steps"] == 21096]
    c3m = [r for r in rows if r["arm"] == "factorizedc3m"]

    def cell(arm, seed, dom, kind, dist, gt, src=None, steps=None):
        src = (c3m if arm == "factorizedc3m" else last) if src is None else src
        hit = [r for r in src if r["arm"] == arm and r["seed"] == seed and r["dom"] == dom and r["kind"] == kind
               and r["distance"] == dist and r["gt"] == gt and steps in (None, r["steps"])]
        return fmt(hit[0]["point"], hit[0]["ci_low"], hit[0]["ci_high"]) if hit else "-"

    # ---- calibrazione
    md += ["## Calibrazione della scala di d_P (emendamento 4, sez. 1)", "",
           "c = mediana(d_P GT) / mediana(d_P modello) su coppie di held-out SINTETICI del training (100 soggetti per "
           "dominio, stesso dominio, soggetti ed etichette diversi; `tools/fact_calib.py`, "
           "`factorized_calibration.csv`); c_LS = minimi quadrati senza intercetta (sensibilita'); c per dominio "
           "descrittivo. Mai i domini di test.", "",
           "| checkpoint | c | c_LS | c bfm | c ict | c gnm | mediana d_P GT | mediana d_P modello | coppie |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k in fact_calib.CKPTS:
        r = cal.get(k)
        md.append(f"| {k} | " + (" | ".join(f"{r[c]:.3f}" for c in ("c_median", "c_ls", "c_median_bfm", "c_median_ict",
                                                                        "c_median_gnm", "median_dP_gt", "median_dP_model"))
                                + f" | {int(r['n_pairs'])} |" if r else "in attesa | - | - | - | - | - | - | - |"))
    # ---- k delle composizioni (emendamento 5)
    md += ["", "k delle composizioni (emendamento 5, sez. 1): stesse mesh e coppie held-out, ICP + Chamfer e NICP per "
           "coppia in modo cs (`fact_calib.py bl`), k = mediana(d_P GT) / mediana(d_P baseline), d_P baseline = "
           "distanza / CS_ref a centroid size 1; NICP su 6000 coppie fisse (`icp_cs_sub`: ICP sulle stesse). "
           "Composizioni con k primaria; k_LS solo riportata.", "",
           "| baseline | k | k_LS | k bfm | k ict | k gnm | mediana d_P GT | mediana d_P baseline | coppie | fallite |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    kb = EV / "factorized_calibration_bl.csv"
    kbl = {r["key"]: r for r in csv.DictReader(open(kb))} if kb.exists() else {}
    for k in ("icp_cs", "nicp_cs", "icp_cs_sub"):
        r = kbl.get(k)
        md.append(f"| {k} | " + (" | ".join(f"{float(r[c]):.3f}" for c in ("k_median", "k_ls", "k_median_bfm", "k_median_ict",
                                                                            "k_median_gnm", "median_dP_gt", "median_dP_bl"))
                                 + f" | {r['n_pairs']} | {r['n_failed']} |" if r else
                                 "in attesa | - | - | - | - | - | - | - | - |"))
    # ---- HIFI3D primaria
    md += ["", "## HIFI3D, GT FR (primaria), ultimo checkpoint", "",
           "Verdetto (emendamento 4, sez. 4): delta appaiato contro ICP + Chamfer in mm e NICP su template in mm (stesse "
           "righe e repliche, `factorized_paired.csv`): sopra se l'IC sta sopra 0, sotto se sta sotto, pari altrimenti.",
           "", "| braccio | seme | distanza | Spearman con FR | verdetto | SR (d_P, u, z) | maxabs |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm, seed, e, steps in runs():
        if e == "036":
            continue
        dists = ("form", "form_cal", "form_cal_ls") if arm in fz.FACTORIZED + ("factorizedc3m",) else (FORM_DIST[arm],)
        key = calib_key(arm, seed, e)
        lab = f"{arm} e{e}" if arm == "factorizedc3m" else arm
        for d in dists:
            md.append(f"| {lab} | {seed} | {d} | {cell(arm, seed, 'hifi', 'graded', d, 'fr', steps=steps)} | {verdict(P, key, d)} | "
                      f"{cell(arm, seed, 'hifi', 'graded', SHAPE_DIST[arm], 'sr', steps=steps)} | "
                      f"{cell(arm, seed, 'hifi', 'graded', d, 'maxabs', steps=steps)} |")
    md += ["", "Baseline (aau/runs/evidence/baselines_mm, stesse colonne):", "",
           "| metodo | FR | SR | maxabs |", "| --- | --- | --- | --- |"]
    for _, lab in BL_ROWS:
        c = {r["gt"]: fmt(r["point"], r["ci_low"], r["ci_high"]) for r in bl if r["dom"] == "hifi" and r["method"] == lab}
        if c:
            md.append(f"| {lab} | {c.get('fr', '-')} | {c.get('sr', '-')} | {c.get('maxabs', '-')} |")
    # ---- tabella calibrata con i delta appaiati
    # emendamento 5, sez. 2: per SR prima i concorrenti invarianti alla taglia (NICP e ICP cs)
    cmp_fr = ("ICP + Chamfer in mm", "NICP su template in mm", "taglia oracolo", "oracolo taglia + ICP cs cal.",
              COMP_EST, "taglia stimata + NICP cs cal.")
    cmp_by_gt = {"fr": cmp_fr, "sr": ("NICP per coppia (cs)", "ICP + Chamfer (cs)") + cmp_fr}

    def prow(dom, gt, kind, key, dist, cmp_bl):
        rr = {r["baseline"]: r for r in P if r["domain"] == dom and r["gt"] == gt and r["kind"] == kind
              and r["arm"] == key and r["distance"] == dist}
        if "-" not in rr:
            return None
        v = rr["-"]
        return (fmt(float(v["arm_point"]), float(v["arm_ci_low"]), float(v["arm_ci_high"])),
                [dlab(rr[b]) if b in rr else "-" for b in cmp_bl])

    keys_tab = [(f"{a}_s{s}", d) for a in ("factorized", "factorized2") for s in SEEDS for d in ("form_cal", "form")] + \
        [(f"factorizedc3m_e{e}", d) for e, _ in C3M for d in ("form_cal", "form")] + \
        [(f"ctrlfr_s{s}", "z") for s in SEEDS] + [(f"dual_s{s}", "zf") for s in SEEDS]
    md += ["", "## HIFI3D FR e SR con d_F calibrata: delta appaiati (emendamenti 4 e 5)", "",
           "Righe e repliche di `fact_paired.py` (maschera comune). Composizioni (emendamento 5): d_F = sqrt((S_i - "
           "S_j)^2 + S_i S_j d_P^2) con d_P = k_ICP x ICP + Chamfer cs / CS_ref (S oracolo di FR o stimata dalla mesh) "
           "o k_NICP x NICP per coppia cs / CS_ref (S stimata), k degli held-out sintetici. Per SR le prime colonne "
           "sono i concorrenti invarianti alla taglia. delta [IC 95%].", ""]
    for gt in ("fr", "sr"):
        cmp_bl = cmp_by_gt[gt]
        md += [f"### GT {gt.upper()}", "", "| braccio | distanza | Spearman | " + " | ".join(cmp_bl) + " |",
               "| --- | --- | --- | " + " | ".join("---" for _ in cmp_bl) + " |"]
        for key, d in keys_tab:
            d = d if gt == "fr" else {"form_cal": "shape", "form": None, "z": "z", "zf": "u"}[d]
            x = prow("hifi3d", gt, "rho", key, d, cmp_bl) if d else None
            if x:
                md.append(f"| {key} | {d} | {x[0]} | " + " | ".join(x[1]) + " |")
        md.append("")
    # ---- analisi della taglia
    md += ["## Analisi della taglia (emendamento 4, sez. 2), GT FR", "",
           "(a) Spearman parziale dato l'oracolo (ranghi regrediti su rango(|delta log S|) e rango^2, Pearson dei "
           "residui); (b) Spearman nel quintile basso di |delta log S|; (c) Spearman grezzo contro le composizioni "
           "calibrate (emendamento 5). "
           f"Criterio \"forma oltre la taglia\" (solo HIFI3D, per seme): delta di (a) contro ICP mm e contro NICP cs con "
           f"IC sopra 0, e delta grezzo contro {COMP_EST} con estremo inferiore > -{MARGIN}; la composizione con NICP "
           "e' descrittiva. dev FaceScape e FaMoS TEST: secondarie.", ""]
    sz_keys = [(f"ctrlfr_s{s}", "z") for s in SEEDS] + \
        [(f"{a}_s{s}", "form_cal") for a in ("factorized", "factorized2") for s in SEEDS] + \
        [(f"dual_s{s}", "zf") for s in SEEDS] + [(f"factorizedc3m_e{e}", "form_cal") for e, _ in C3M]

    def get(dom, kind, key, dist, b):
        hit = [r for r in P if r["domain"] == dom and r["gt"] == "fr" and r["kind"] == kind and r["arm"] == key
               and r["distance"] == dist and r["baseline"] == b]
        return hit[0] if hit else None

    crit = {}
    for dom, title in (("hifi3d", "HIFI3D"), ("facescape", "dev FaceScape"), ("famos", "FaMoS TEST, scan gallery -> scan")):
        if not any(r["domain"] == dom for r in P):
            continue
        md += [f"### {title}", "",
               "| braccio | distanza | (a) parziale | (a) - ICP mm | (a) - NICP cs | (b) quintile | (b) - ICP mm | "
               "(c) Spearman | - oracolo + ICP cs cal. | - stimata + ICP cs cal. | - stimata + NICP cs cal. | criterio |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key, d in sz_keys:
            pts = {k: get(dom, k, key, d, "-") for k in ("partial", "q20", "rho")}
            if not all(pts.values()):
                continue
            ref = {(k, b): get(dom, k, key, d, b) for k in ("partial", "q20", "rho") for b in
                   ("ICP + Chamfer in mm", "NICP per coppia (cs)", "oracolo taglia + ICP cs cal.", COMP_EST,
                    "taglia stimata + NICP cs cal.")}
            ok = None
            if dom == "hifi3d" and ref[("partial", "ICP + Chamfer in mm")] and ref[("partial", "NICP per coppia (cs)")] \
                    and ref[("rho", COMP_EST)]:
                ok = (float(ref[("partial", "ICP + Chamfer in mm")]["ci_low"]) > 0
                      and float(ref[("partial", "NICP per coppia (cs)")]["ci_low"]) > 0
                      and float(ref[("rho", COMP_EST)]["ci_low"]) > -MARGIN)
                crit[key] = ok
            f = lambda r: fmt(float(r["arm_point"]), float(r["arm_ci_low"]), float(r["arm_ci_high"]))  # noqa: E731
            g = lambda r: dlab(r) if r else "-"  # noqa: E731
            md.append(f"| {key} | {d} | {f(pts['partial'])} | {g(ref[('partial', 'ICP + Chamfer in mm')])} | "
                      f"{g(ref[('partial', 'NICP per coppia (cs)')])} | {f(pts['q20'])} | "
                      f"{g(ref[('q20', 'ICP + Chamfer in mm')])} | {f(pts['rho'])} | "
                      f"{g(ref[('rho', 'oracolo taglia + ICP cs cal.')])} | {g(ref[('rho', COMP_EST)])} | "
                      f"{g(ref[('rho', 'taglia stimata + NICP cs cal.')])} | "
                      f"{'-' if ok is None else ('soddisfatto' if ok else 'non soddisfatto')} |")
        md += ["", "Baseline sulle stesse righe:", "", "| metodo | (a) parziale | (b) quintile | Spearman |",
               "| --- | --- | --- | --- |"]
        for lab in dict.fromkeys(r["distance"] for r in P if r["domain"] == dom and r["arm"] == "baseline"):
            c = {}
            for k in ("partial", "q20", "rho"):
                hit = [r for r in P if r["domain"] == dom and r["gt"] == "fr" and r["kind"] == k and r["arm"] == "baseline"
                       and r["distance"] == lab]
                c[k] = fmt(float(hit[0]["arm_point"]), float(hit[0]["arm_ci_low"]), float(hit[0]["arm_ci_high"])) \
                    if hit else "- (nullo per costruzione)" if k == "partial" and lab == "taglia oracolo" else "-"
            md.append(f"| {lab} | {c['partial']} | {c['q20']} | {c['rho']} |")
        md.append("")
    md += ["Verdetto \"forma oltre la taglia\" su HIFI3D (entrambi i semi):", ""]
    for arm, d in (("ctrlfr", "z"), ("factorized", "form_cal"), ("factorized2", "form_cal"), ("dual", "zf")):
        v = [crit.get(f"{arm}_s{s}") for s in SEEDS]
        out = "in attesa" if None in v else "si'" if all(v) else "no" if not any(v) else "discordante"
        md.append(f"- {arm} ({d}): {out} (" + ", ".join(f"s{s} {'-' if x is None else ('si' if x else 'no')}"
                                                         for s, x in zip(SEEDS, v)) + ")")
    # ---- regola dual
    md += ["", "## Regola dual (emendamento 4, sez. 5)", "",
           f"dual si adotta solo se in entrambi i semi l'estremo inferiore di ogni delta qui sotto e' > -{MARGIN}; "
           "altrimenti, o se non e' risolto, si sceglie factorized con d_F calibrata.", ""]
    rule = [(dom, gt, kind, a, ref.format(s=s), s) for s in SEEDS for dom, gt, kind, a, ref in
            (("hifi3d", "fr", "rho", "zf", "ctrlfr_s{s}|z"), ("hifi3d", "sr", "rho", "u", "factorized_s{s}|shape"),
             ("hifi3d", "fr", "partial", "zf", "factorized_s{s}|form_cal"),
             ("facescape", "fr", "rho", "zf", "factorized_s{s}|form_cal"),
             ("facescape", "sr", "rho", "u", "factorized_s{s}|shape"))]
    have_dual = any(r["arm"].startswith("dual_") for r in P)
    if not have_dual:
        md.append("**in attesa** delle righe dual (z_F e u) in `factorized_paired.csv`.")
    else:
        md += ["| seme | dominio | GT | misura | dual | riferimento | delta [IC] | > -0.03 |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        res = []
        for dom, gt, kind, a, ref, s in rule:
            hit = [r for r in P if r["domain"] == dom and r["gt"] == gt and r["kind"] == kind and r["arm"] == f"dual_s{s}"
                   and r["distance"] == a and r["baseline"] == f"braccio {ref}"]
            ok = float(hit[0]["ci_low"]) > -MARGIN if hit else None
            res.append(ok)
            md.append(f"| {s} | {dom} | {gt.upper()} | {kind} | {a} | {ref} | {dlab(hit[0]) if hit else 'mancante'} | "
                      f"{'-' if ok is None else ('si' if ok else 'no')} |")
        # emendamento 5, sez. 4: un esito falso decide ("no") anche con righe mancanti
        dec = ("si adotta dual" if all(x is True for x in res) else
               "no, si sceglie factorized calibrato" + (" (anche con righe mancanti)" if None in res else "")
               if False in res else "non risolto (righe mancanti): si sceglie factorized calibrato")
        md += ["", f"**Esito: {dec}.**"]
    # ---- domini secondari
    for dom, title in (("devfs", "dev FaceScape (neutra)"), ("fv", "FaceVerse con espressioni")):
        md += ["", f"## {title}, graduata, ultimo checkpoint", "",
               "| braccio | seme | FR | FR d_F calibrata | SR | maxabs |", "| --- | --- | --- | --- | --- | --- |"]
        for arm, seed, e, steps in runs():
            if e == "036":
                continue
            d = FORM_DIST[arm]
            lab = f"{arm} e{e}" if arm == "factorizedc3m" else arm
            md.append(f"| {lab} | {seed} | {cell(arm, seed, dom, 'graded', d, 'fr', steps=steps)} | "
                      f"{cell(arm, seed, dom, 'graded', 'form_cal', 'fr', steps=steps) if d == 'form' else '-'} | "
                      f"{cell(arm, seed, dom, 'graded', SHAPE_DIST[arm], 'sr', steps=steps)} | "
                      f"{cell(arm, seed, dom, 'graded', d, 'maxabs', steps=steps)} |")
        for _, lab in BL_ROWS:
            c = {r["gt"]: fmt(r["point"], r["ci_low"], r["ci_high"]) for r in bl if r["dom"] == dom and r["method"] == lab}
            if c:
                md.append(f"| {lab} (baseline) | - | {c.get('fr', '-')} | - | {c.get('sr', '-')} | {c.get('maxabs', '-')} |")
    md += ["", "## Riconoscimento rank-1 senza crop e punteggio dev, ultimo checkpoint", "",
           "| braccio | seme | distanza | HIFI3D | dev FaceScape espr. | FaceVerse espr. | punteggio dev |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm, seed, e, steps in runs():
        if e == "036":
            continue
        src = c3m if arm == "factorizedc3m" else last
        for d in dict.fromkeys((FORM_DIST[arm], REF_DIST[arm], SHAPE_DIST[arm])):
            hit = lambda dom, k, g, dd=d: [r for r in src if r["arm"] == arm and r["seed"] == seed and r["steps"] == steps  # noqa: E731
                                           and r["dom"] == dom and r["kind"] == k and r["distance"] == dd and r["gt"] == g]
            a, b = hit("devfs", "graded", "maxabs"), hit("devfs_expr", "rank1", "-")
            dev = f"{(a[0]['point'] + b[0]['point']) / 2:.3f}" if a and b else "-"
            lab = f"{arm} e{e}" if arm == "factorizedc3m" else arm
            md.append(f"| {lab} | {seed} | {d} | {cell(arm, seed, 'hifi', 'rank1', d, '-', src, steps=steps)} | "
                      f"{cell(arm, seed, 'devfs_expr', 'rank1', d, '-', src, steps=steps)} | {cell(arm, seed, 'fv', 'rank1', d, '-', src, steps=steps)} | "
                      f"{dev} |")
    md += ["", "## FaMoS TEST (scan gallery -> scan) e NoW, ultimo checkpoint", "",
           "Valori di `eval_famos_v3.py` (d_F grezza; la calibrata e i delta appaiati nelle sezioni dell'emendamento 4 e "
           "in fondo). Le Chamfer di `aau/famos/famos_eval.py` sulla patch (intera, regione stabile) sono un'ALTRA "
           "Chamfer, non la `mm_chamfer` delle baseline in mm: riportate una volta, con il loro nome.", "",
           "| braccio | seme | distanza | FaMoS FR | FaMoS SR | NoW tau |", "| --- | --- | --- | --- | --- | --- |"]
    for arm, seed, e, steps in runs():
        if e == "036":
            continue
        src = c3m if arm == "factorizedc3m" else last
        mine = [r for r in src if r["arm"] == arm and r["seed"] == seed and r["steps"] == steps]
        dists = sorted({r["distance"] for r in mine if r["dom"] == "famos"})
        tau = [r for r in mine if r["dom"] == "now"]
        lab = f"{arm} e{e}" if arm == "factorizedc3m" else arm
        for d in dists or ["-"]:
            tt = [r for r in tau if r["distance"] == d] or [r for r in tau if r["distance"] in ("u", "z")]
            t = fmt(tt[0]["point"], tt[0]["ci_low"], tt[0]["ci_high"]) if tt else "-"
            md.append(f"| {lab} | {seed} | {d} | {cell(arm, seed, 'famos', 'graded scan gallery -> scan', d, 'fr', src, steps=steps)} | "
                      f"{cell(arm, seed, 'famos', 'graded scan gallery -> scan', d, 'sr', src, steps=steps)} | {t} |")
    for m, lab in (("chamfer_full", "Chamfer di famos_eval.py, patch intera (non mm_chamfer)"),
                   ("chamfer_stable", "Chamfer di famos_eval.py, regione stabile (non mm_chamfer)")):
        c = {g: famos_ch[(m, g, "scan gallery -> scan")] for g in ("fr", "sr") if (m, g, "scan gallery -> scan") in famos_ch}
        if c:
            md.append(f"| {lab} | - | - | " + " | ".join(
                fmt(float(c[g]["spearman"]), float(c[g]["ci_low"]), float(c[g]["ci_high"])) if g in c else "-"
                for g in ("fr", "sr")) + " | - |")
    fb = [r for r in csv.DictReader(open(BL / "spearman.csv")) if r["domain"] == "famos" and r["group"] == "scan gallery -> scan"]
    for m, lab in BL_ROWS:
        c = {r["gt"]: fmt(float(r["point"]), float(r["ci_low"]), float(r["ci_high"])) for r in fb if r["method"] == m}
        if c:
            md.append(f"| {lab} (baseline) | - | - | {c.get('fr', '-')} | {c.get('sr', '-')} | - |")
    if P:
        md += ["", "## Delta appaiati braccio - baseline, Spearman grezzo (stesse righe e repliche di baselines_mm)", "",
               "Generati da `tools/fact_paired.py`: righe e seme di `aau/baselines_mm/blmm_eval.py` (FaMoS: blocco scan "
               "gallery -> scan di `blmm_eval.famos`), maschera comune (righe con tutte le distanze finite). "
               "delta [IC 95%] (P(delta <= 0)).", ""]
        D = [r for r in P if r["kind"] == "rho" and r["baseline"] != "-" and not r["baseline"].startswith("braccio")]
        bls = list(dict.fromkeys(r["baseline"] for r in D))
        for dom in dict.fromkeys(r["domain"] for r in D):
            for gt in ("fr", "sr"):
                bd = [b for b in bls if any(r["domain"] == dom and r["baseline"] == b for r in D)]
                md += [f"### {dom}, GT {gt.upper()}", "", "| braccio | distanza | punto | " + " | ".join(bd) + " |",
                       "| --- | --- | --- | " + " | ".join("---" for _ in bd) + " |"]
                for key in dict.fromkeys((r["arm"], r["distance"]) for r in D if r["domain"] == dom and r["gt"] == gt):
                    rr = {r["baseline"]: r for r in D if r["domain"] == dom and r["gt"] == gt and (r["arm"], r["distance"]) == key}
                    first = next(iter(rr.values()))
                    md.append(f"| {key[0]} | {key[1]} | {float(first['arm_point']):.3f} | " + " | ".join(
                        f"{dlab(rr[b])} ({float(rr[b]['p_le0']):.2f})" if b in rr else "-" for b in bd) + " |")
                md.append("")
        ctl = EV / "factorized_paired_controls.json"
        if ctl.exists():
            md += ["Controlli di fact_paired.py: " + "; ".join(json.loads(ctl.read_text())), ""]
    md += explore_section(P, cal, kbl)
    md += ["", "## Mancanti", "", ", ".join(missing) if missing else "nessuno", "",
           "Valori completi (anche 10.548 passi, d_P, FaMoS per blocco): `factorized_results.csv`; delta e analisi della "
           "taglia: `factorized_paired.csv`."]
    (EV / "factorized_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
