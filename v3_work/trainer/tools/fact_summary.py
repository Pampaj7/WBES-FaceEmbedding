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
Regola del protocollo su HIFI3D FR, ultimo checkpoint, per seme. Scrive aau/runs/evidence/trainer_v3/
factorized_results.md e factorized_results.csv.
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
NB, SEED = 1000, 1234
NICP_E12, E108_E12, NICP_LO = 0.367, 0.194, 0.289
BL_ROWS = (("scale_e108", "e108 (cieco alla taglia)"), ("mm_rigid_icp_chamfer", "ICP + Chamfer in mm"),
           ("mm_nicp_p2tri", "NICP per coppia in mm"), ("mm_nicp_template", "NICP su template in mm"),
           ("cs_nicp_p2tri", "NICP per coppia, modo cs"), ("est_cs", "taglia stimata (non oracolo)"),
           ("oracle_size", "taglia oracolo"), ("chamfer_eval", "Chamfer eval"))
BL_GROUP = {"hifi": "nocrop_cross", "devfs": "nocrop_cross", "fv": "mesh_pair_nocrop"}
BL_DOMAIN = {"hifi": "hifi3d", "devfs": "facescape", "fv": "faceverse"}


def vname(arm: str, seed: int) -> str:
    return arm + ("" if seed == 1234 else f"s{seed}")


def embeddings(dom: str, arm: str, seed: int, e: str) -> Path | None:
    root = EVAL / DOMS[dom][0]
    hits = sorted(root.glob(f"data_*/scale_v3{vname(arm, seed)}fulle{e}*/zs_zeroshot/embeddings.npz"))
    return hits[0] if hits else None


def load_emb(path: Path):
    with np.load(path, allow_pickle=True) as z:
        return (np.asarray(z["Z"], np.float64), [str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]],
                Path(str(z["checkpoint"])))


def model_D(Z: np.ndarray, i: np.ndarray, j: np.ndarray, head: str, dpu: float) -> dict:
    return fz.model_distances(Z, i, j, head, dpu)


def head_of(ckpt: Path):
    import torch
    head = torch.load(ckpt, map_location="cpu", weights_only=False)["args"].get("head", "embed")
    return head, (ef.dp_from_ckpt(ckpt) if head in fz.FACTORIZED else float("nan"))


FORM_DIST = {"factorized": "form", "factorized2": "form", "ctrlfr": "z", "dual": "zf"}
SHAPE_DIST = {"factorized": "shape", "factorized2": "shape", "ctrlfr": "z", "dual": "u"}
NOW_DIST = {"factorized": (("", "u"),), "factorized2": (("", "u"),), "dual": (("", "u"), ("_zf", "zf"))}


def graded(dom: str, path: Path) -> list[dict]:
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
    return ef.boot_rows(model_D(Z, i, j, head, dpu), gts, si, sj, len(subjects), NB, SEED)


def rank1(dom: str, path: Path) -> dict:
    import zs_expr_summarize as zes
    from zs_stage import TOPOLOGIES
    Z, subj, topo, ckpt = load_emb(path)
    head, dpu = head_of(ckpt)
    idx = zes.Index(sorted(set(subj)))
    order = np.asarray([list(zip(subj, topo)).index(k) for k in idx.keys])
    Z = Z[order]
    n = len(Z)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    Ds = model_D(Z, i.ravel(), j.ravel(), head, dpu)
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


def verdict(p: float, lo: float, hi: float) -> str:
    g = (p - E108_E12) / (NICP_E12 - E108_E12)
    if p >= NICP_LO and hi >= NICP_E12:
        return f"raggiunge NICP (g {g:.2f})"
    if g >= 0.5 and lo > E108_E12:
        return f"si avvicina a NICP (g {g:.2f})"
    return f"non si avvicina (g {g:.2f})"


def main() -> None:
    rows, missing = [], []
    for arm in ARMS:
        for seed in SEEDS:
            for e in EPOCHS:
                tag = f"{arm} s{seed} {STEPS[e]}"
                for dom in ("hifi", "devfs", "fv"):
                    p = embeddings(dom, arm, seed, e)
                    if p is None:
                        missing.append(f"{tag} {dom}")
                        continue
                    for r in graded(dom, p):
                        if r["level"] == "mesh_pair":
                            rows.append({"arm": arm, "seed": seed, "steps": STEPS[e], "dom": dom, "kind": "graded", **r})
                for dom in ("hifi", "devfs_expr", "fv"):
                    p = embeddings(dom, arm, seed, e)
                    if p is not None:
                        for k, v in rank1(dom, p).items():
                            rows.append({"arm": arm, "seed": seed, "steps": STEPS[e], "dom": dom, "kind": "rank1",
                                         "distance": k, "gt": "-", "point": v["rank1"], "ci_low": v["rank1_ci"][0],
                                         "ci_high": v["rank1_ci"][1]})
                for r in famos(arm, seed, e):
                    rows.append({"arm": arm, "seed": seed, "steps": STEPS[e], "dom": "famos", "kind": f"graded {r['block']}",
                                 "distance": r["method"], "gt": r["gt"], "point": float(r["spearman"]),
                                 "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"])})
                # NoW: u per i fattorizzati, z per ctrlfr; dual su u (now/<tag>) e su z_F (now/<tag>_zf)
                for sfx, dist in NOW_DIST.get(arm, (("", "z"),)):
                    t = now_tau(arm, seed, e, sfx)
                    if t:
                        rows.append({"arm": arm, "seed": seed, "steps": STEPS[e], "dom": "now", "kind": "tau",
                                     "distance": dist, "gt": "-", "point": t[0], "ci_low": t[1], "ci_high": t[2]})
    keys = ["arm", "seed", "steps", "dom", "kind", "distance", "gt", "point", "ci_low", "ci_high"]
    with open(EV / "factorized_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    bl = baselines()
    md = ["# Risultati di factorized_protocol.md (emendamenti 1-3)", "",
          "Generato da `v3_work/trainer/tools/fact_summary.py`. Pesi EMA. IC 95% bootstrap per soggetto (1000 repliche, "
          "seme 1234). Graduata: livello coppia di mesh, senza crop, topologie diverse (il `nocrop_cross` di E12), "
          "calcolata dagli embedding. Distanza del modello: d_F (form) per factorized/factorized2, ||z|| per ctrlfr, "
          "||z_F|| (FR) e ||u|| (SR) per dual.", ""]
    last = [r for r in rows if r["steps"] == 21096]

    def cell(arm, seed, dom, kind, dist, gt, src=last):
        hit = [r for r in src if r["arm"] == arm and r["seed"] == seed and r["dom"] == dom and r["kind"] == kind
               and r["distance"] == dist and r["gt"] == gt]
        return fmt(hit[0]["point"], hit[0]["ci_low"], hit[0]["ci_high"]) if hit else "-"

    md += ["## HIFI3D, GT FR (primaria) e regola, 21.096 passi", "",
           "| braccio | seme | Spearman con FR | verdetto | SR | maxabs |", "| --- | --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        d = FORM_DIST[arm]
        for seed in SEEDS:
            hit = [r for r in last if r["arm"] == arm and r["seed"] == seed and r["dom"] == "hifi" and r["kind"] == "graded"
                   and r["distance"] == d and r["gt"] == "fr"]
            v = verdict(hit[0]["point"], hit[0]["ci_low"], hit[0]["ci_high"]) if hit else "-"
            ds = SHAPE_DIST[arm]
            md.append(f"| {arm} | {seed} | {cell(arm, seed, 'hifi', 'graded', d, 'fr')} | {v} | "
                      f"{cell(arm, seed, 'hifi', 'graded', ds, 'sr')} | {cell(arm, seed, 'hifi', 'graded', d, 'maxabs')} |")
    md += ["", "Baseline (aau/runs/evidence/baselines_mm, stesse colonne):", "",
           "| metodo | FR | SR | maxabs |", "| --- | --- | --- | --- |"]
    for _, lab in BL_ROWS:
        c = {r["gt"]: fmt(r["point"], r["ci_low"], r["ci_high"]) for r in bl if r["dom"] == "hifi" and r["method"] == lab}
        if c:
            md.append(f"| {lab} | {c.get('fr', '-')} | {c.get('sr', '-')} | {c.get('maxabs', '-')} |")
    for dom, title in (("devfs", "dev FaceScape (neutra)"), ("fv", "FaceVerse con espressioni")):
        md += ["", f"## {title}, graduata, 21.096 passi", "", "| braccio | seme | FR | SR | maxabs |", "| --- | --- | --- | --- | --- |"]
        for arm in ARMS:
            d = FORM_DIST[arm]
            for seed in SEEDS:
                md.append(f"| {arm} | {seed} | {cell(arm, seed, dom, 'graded', d, 'fr')} | "
                          f"{cell(arm, seed, dom, 'graded', SHAPE_DIST[arm], 'sr')} | "
                          f"{cell(arm, seed, dom, 'graded', d, 'maxabs')} |")
        for _, lab in BL_ROWS:
            c = {r["gt"]: fmt(r["point"], r["ci_low"], r["ci_high"]) for r in bl if r["dom"] == dom and r["method"] == lab}
            if c:
                md.append(f"| {lab} (baseline) | - | {c.get('fr', '-')} | {c.get('sr', '-')} | {c.get('maxabs', '-')} |")
    md += ["", "## Riconoscimento rank-1 senza crop e punteggio dev, 21.096 passi", "",
           "| braccio | seme | distanza | HIFI3D | dev FaceScape espr. | FaceVerse espr. | punteggio dev |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        for d in dict.fromkeys((FORM_DIST[arm], SHAPE_DIST[arm])):
            for seed in SEEDS:
                hit = lambda dom, k, g, dd=d: [r for r in last if r["arm"] == arm and r["seed"] == seed  # noqa: E731
                                               and r["dom"] == dom and r["kind"] == k and r["distance"] == dd and r["gt"] == g]
                a, b = hit("devfs", "graded", "maxabs"), hit("devfs_expr", "rank1", "-")
                dev = f"{(a[0]['point'] + b[0]['point']) / 2:.3f}" if a and b else "-"
                md.append(f"| {arm} | {seed} | {d} | {cell(arm, seed, 'hifi', 'rank1', d, '-')} | "
                          f"{cell(arm, seed, 'devfs_expr', 'rank1', d, '-')} | {cell(arm, seed, 'fv', 'rank1', d, '-')} | {dev} |")
    md += ["", "## FaMoS TEST (scan gallery -> scan) e NoW, 21.096 passi", "",
           "| braccio | seme | distanza | FaMoS FR | FaMoS SR | NoW tau |", "| --- | --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        for seed in SEEDS:
            dists = sorted({r["distance"] for r in last if r["arm"] == arm and r["seed"] == seed and r["dom"] == "famos"})
            tau = [r for r in last if r["arm"] == arm and r["seed"] == seed and r["dom"] == "now"]
            for d in dists or ["-"]:
                tt = [r for r in tau if d.endswith("_" + r["distance"])] or tau   # dual: tau della stessa uscita
                t = fmt(tt[0]["point"], tt[0]["ci_low"], tt[0]["ci_high"]) if tt else "-"
                md.append(f"| {arm} | {seed} | {d} | {cell(arm, seed, 'famos', 'graded scan gallery -> scan', d, 'fr')} | "
                          f"{cell(arm, seed, 'famos', 'graded scan gallery -> scan', d, 'sr')} | {t} |")
    fb = [r for r in csv.DictReader(open(BL / "spearman.csv")) if r["domain"] == "famos" and r["group"] == "scan gallery -> scan"]
    for m, lab in BL_ROWS:
        c = {r["gt"]: fmt(float(r["point"]), float(r["ci_low"]), float(r["ci_high"])) for r in fb if r["method"] == m}
        if c:
            md.append(f"| {lab} (baseline) | - | - | {c.get('fr', '-')} | {c.get('sr', '-')} | - |")
    pp = EV / "factorized_paired.csv"
    if pp.exists():
        P = list(csv.DictReader(open(pp)))
        md += ["", "## Delta appaiati braccio - baseline (stesse righe e repliche di baselines_mm, 21.096 passi)", "",
               "Generati da `tools/fact_paired.py`: righe e seme di `aau/baselines_mm/blmm_eval.py`, maschera comune (righe "
               "con tutte le distanze finite). delta [IC 95%] (P(delta <= 0)).", ""]
        bls = list(dict.fromkeys(r["baseline"] for r in P))
        for dom in dict.fromkeys(r["domain"] for r in P):
            for gt in ("fr", "sr"):
                md += [f"### {dom}, GT {gt.upper()}", "", "| braccio | distanza | punto | " + " | ".join(bls) + " |",
                       "| --- | --- | --- | " + " | ".join("---" for _ in bls) + " |"]
                for key in dict.fromkeys((r["arm"], r["distance"]) for r in P if r["domain"] == dom and r["gt"] == gt):
                    rr = {r["baseline"]: r for r in P if r["domain"] == dom and r["gt"] == gt and (r["arm"], r["distance"]) == key}
                    first = next(iter(rr.values()))
                    md.append(f"| {key[0]} | {key[1]} | {float(first['arm_point']):.3f} | " + " | ".join(
                        f"{float(rr[b]['delta']):+.3f} [{float(rr[b]['ci_low']):+.3f}, {float(rr[b]['ci_high']):+.3f}] "
                        f"({float(rr[b]['p_le0']):.2f})" if b in rr else "-" for b in bls) + " |")
                md.append("")
    md += ["", "## Mancanti", "", ", ".join(missing) if missing else "nessuno", "",
           "Valori completi (anche 10.548 passi, d_P, FaMoS per blocco): `factorized_results.csv`."]
    (EV / "factorized_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
