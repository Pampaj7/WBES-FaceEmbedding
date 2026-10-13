#!/usr/bin/env python3
"""Quota della taglia nella GT FR per dominio: perche' i risultati cambiano da un generatore all'altro (sola GT).

    v3_work/unified_gt/run.sh aau/size_share/size_share.py          (size_share.sbatch; LEGGE LE GT di test)

Domini: HIFI3D, FaceScape (dev), FaceVerse con espressioni, FaceVerse neutra (stesse chiavi e GT di FaceVerse: i
numeri della GT coincidono, cambiano solo i rho dei metodi), FaMoS TEST (15 soggetti, descrittivo). Niente Ava-256.

Coppie: le coppie di identita' diverse della matrice unificata, a livello identita' (ogni coppia di soggetti una
volta). Soggetti come ``fm_eval.run_view``: per le viste sintetiche l'unione ordinata dei soggetti delle righe di
``fm_eval.rows_view`` (righe e GT di ``fact_paired.rows_for`` / ``bp_paired_e2.all_cross_rows``), tutte le i < j;
FaMoS: i 15 ``view_id`` di ``gt_matrix.npz``, le 105 coppie di ``fact_paired.famos_frame``. GT in unita' fisiche:
FR (mm) e SR (d_P, pre-forme a centroid size 1) di ``CANONICAL_GT/eval`` per l'unita' del json, S = centroid size di
FR (``fact_paired.oracle_S``). Controllo: la GT delle righe della matrice = queste GT normalizzate (scarto massimo).

Grandezze (ogni replica pesa la coppia c_i c_j, come ``fm_eval._rep``):
  - Kendall: FR^2 ~ (S_i - S_j)^2 + S_i S_j d_SR^2; errore relativo |approssimazione - FR^2| / FR^2 (mediana, max;
    solo sul punto). Il resto e' il termine dei centroidi della rigida robusta (``train_fr_sr`` docstring);
  - quota della taglia = sum w (S_i - S_j)^2 / sum w FR^2, e la quota della forma = sum w S_i S_j d_SR^2 / sum w FR^2;
  - Spearman di FR col solo termine taglia (S_i - S_j)^2 e col solo termine forma S_i S_j d_SR^2 (ranghi delle righe
    ripetute, Pearson: ``fact_paired._pearson``);
  - CV di S fra le identita' (std / media, ddof 0 come ``bp_paired_e2.gt_controls``; pesi c_i);
  - SR: mediana e IQR sulle coppie.
Repliche: 1000 per soggetto, quelle della matrice: ``default_rng(seme)`` sui soggetti ordinati, seme della T1 del
dominio (HIFI3D 990708, FaceScape 796786, FaceVerse e neutra 271049, letti da ``full_matrix/spearman.csv`` e
controllati); FaMoS ``famos_eval.bootstrap_counts(15, 1000, 1234)``. Replica 0 = punto; IC 95% percentile.

Collegamento ai risultati (FR, senza crop, letti dai CSV con file:riga): oracolo taglia, ICP mm, ICP cs, d_F non
cal. e cal. (s1234, s2345) da ``full_matrix/spearman.csv`` (passo ``comune``); NoW originale e metrical da
``protocolli/spearman.csv`` (i plug-in di ``competitors_new``, coi landmark esatti); ULRSSM per coppia FR da
``competitors_new/ulrssm/faces_xyz/spearman.csv`` (la variante di ``selection.json``). Delta: ICP mm - ICP cs
(``full_matrix/deltas.csv``, scala_allineamento passo 1) e NoW metrical - NoW originale (``protocolli/deltas.csv``,
segno e IC invertiti). NoW e ULRSSM sono sulle 99.000 righe senza crop (stesso seme), non sulla maschera comune
della matrice (HIFI3D 98.224, FaceScape 88.724 righe).

Uscite in ``aau/runs/evidence/size_share/``: ``gt_size.csv`` (dominio x grandezza, punto e IC), ``methods.csv``
(dominio x metodo / delta, punto, IC, fonte), ``controls.json``, ``results.md``.
"""

from __future__ import annotations

import csv
import functools
import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
sys.path.insert(0, str(AAU / "full_matrix"))

import fm_eval as fm  # noqa: E402  (sola lettura: righe, GT, viste)

fp, be, blmm = fm.fp, fm.be, fm.blmm
REPO = fm.REPO
EV = fm.EV
OUT = EV / "size_share"
N_BOOT = 1000
FAMOS_SEED = 1234
VIEWS = fm.VIEWS                                       # hifi3d, facescape, faceverse, faceverse_neutral, famos
T1_SEED = {"hifi3d": 990708, "facescape": 796786, "faceverse": 271049, "faceverse_neutral": 271049, "famos": 1234}
FM_SP, FM_DL = EV / "full_matrix" / "spearman.csv", EV / "full_matrix" / "deltas.csv"
PR_SP, PR_DL = EV / "protocolli" / "spearman.csv", EV / "protocolli" / "deltas.csv"
ULR_SP = EV / "competitors_new" / "ulrssm" / "faces_xyz" / "spearman.csv"
# (chiave, etichetta, fonte, filtro); le colonne del valore dipendono dalla fonte
FM_COLS = (("oracle_size", "taglia oracolo"), ("mm_rigid_icp_chamfer", "ICP mm"), ("cs_rigid_icp_chamfer", "ICP cs"),
           ("factorized_s1234|form", "d_F non cal. s1234"), ("factorized_s1234|form_cal", "d_F cal. s1234"),
           ("factorized_s2345|form", "d_F non cal. s2345"), ("factorized_s2345|form_cal", "d_F cal. s2345"))
GT_Q = ("quota_taglia", "quota_forma", "rho_fr_taglia", "rho_fr_forma", "cv_S", "sr_mediana", "sr_iqr")


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO))


# ------------------------------------------------------------------------------------------------ GT

def gt_view(view: str) -> tuple[dict, np.ndarray, int, dict]:
    """({fr, sr, S_i, S_j} sulle coppie i < j, counts (1 + N_BOOT, n), seme, controlli) del dominio."""
    src = "faceverse" if view == "faceverse_neutral" else view
    gset = blmm.VIEWS[src]["gt"]
    unit = {"fr": "mm_per_unit", "sr": "dP_per_unit"}
    G = {g: be.zsum.load_gt(be.EVAL_GT / f"{gset}_{g}.npz") for g in fp.GTS}
    U = {g: float(json.loads((be.EVAL_GT / f"{gset}_{g}.json").read_text())[unit[g]]) for g in fp.GTS}
    orc = fp.oracle_S(gset)
    ctrl = {"gt_set": gset, "units": U}
    if view == "famos":
        with np.load(blmm.DATASETS / "FAMOS" / "test_view" / "gt_matrix.npz") as z:
            subj = [str(x) for x in z["names"]]
        sys.path.insert(0, str(AAU / "famos"))
        import famos_eval as fe
        counts = np.concatenate([np.ones((1, len(subj)), np.int64), fe.bootstrap_counts(len(subj), N_BOOT, FAMOS_SEED)])
        seed = FAMOS_SEED
    else:
        rows, _, spread = fm.rows_view(view)
        subj = sorted(set(rows["subject_a"]) | set(rows["subject_b"]))
        seed = T1_SEED[view]
        rng = np.random.default_rng(seed)
        counts = np.stack([np.ones(len(subj), dtype=np.int64)] +
                          [np.bincount(rng.integers(0, len(subj), len(subj)), minlength=len(subj))
                           for _ in range(N_BOOT)])
        # controllo: la GT delle righe (normalizzata) e' la GT per coppia di soggetti di qui
        s2i = {s: k for k, s in enumerate(subj)}
        ra, rb = rows["subject_a"].map(s2i).to_numpy(), rows["subject_b"].map(s2i).to_numpy()
        diff = {}
        for g in fp.GTS:
            Dg, pos = G[g]
            p = np.asarray([pos[s] for s in subj])
            diff[g] = float(np.nanmax(np.abs(Dg[p[ra], p[rb]] - rows[f"gt_{g}"].to_numpy(np.float64))))
        pairs = {(min(a, b), max(a, b)) for a, b in zip(ra, rb) if a != b}
        ctrl.update(rows=int(len(rows)), gt_rows_vs_identity_max_abs=diff, gt_spread_within_subject_pair=spread,
                    subject_pairs_in_rows=len(pairs), subjects_equal_blmm=subj == sorted(blmm.subjects(src)))
    i, j = np.triu_indices(len(subj), 1)
    out = {}
    for g in fp.GTS:
        Dg, pos = G[g]
        p = np.asarray([pos[s] for s in subj])
        out[g] = Dg[np.ix_(p, p)][i, j] * U[g]
    S = np.asarray([orc[s] for s in subj], np.float64)
    out.update(S=S, i=i, j=j)
    ctrl.update(n_subjects=len(subj), n_pairs=int(len(i)), seed=seed)
    return out, counts, seed, ctrl


def _wq(x: np.ndarray, w: np.ndarray, q) -> np.ndarray:
    """Quantili delle righe ripetute w volte (come i ranghi di ``fm_eval._rep``)."""
    return np.percentile(np.repeat(x, w), q)


def replicas(P: dict, counts: np.ndarray) -> dict:
    """{grandezza: (1 + N_BOOT,)} con i pesi c_i c_j sulle coppie (c_i sulle identita' per il CV)."""
    from scipy.stats import rankdata
    S, i, j = P["S"], P["i"], P["j"]
    size = (S[i] - S[j]) ** 2
    shape = S[i] * S[j] * P["sr"] ** 2
    fr2 = P["fr"] ** 2
    V = {q: np.full(len(counts), np.nan) for q in GT_Q}
    for k, c in enumerate(counts):
        w = c[i].astype(np.int64) * c[j]
        keep = w > 0
        wk = w[keep]
        tot = (wk * fr2[keep]).sum()
        V["quota_taglia"][k] = (wk * size[keep]).sum() / tot
        V["quota_forma"][k] = (wk * shape[keep]).sum() / tot
        rf = rankdata(np.repeat(P["fr"][keep], wk))
        V["rho_fr_taglia"][k] = fp._pearson(rankdata(np.repeat(size[keep], wk)), rf)
        V["rho_fr_forma"][k] = fp._pearson(rankdata(np.repeat(shape[keep], wk)), rf)
        m = (c * S).sum() / c.sum()
        V["cv_S"][k] = np.sqrt((c * (S - m) ** 2).sum() / c.sum()) / m
        q25, q50, q75 = _wq(P["sr"][keep], wk, [25, 50, 75])
        V["sr_mediana"][k], V["sr_iqr"][k] = q50, q75 - q25
    return V


def kendall_error(P: dict) -> dict:
    S, i, j = P["S"], P["i"], P["j"]
    fr2 = P["fr"] ** 2
    err = np.abs((S[i] - S[j]) ** 2 + S[i] * S[j] * P["sr"] ** 2 - fr2) / fr2
    return {"kendall_err_rel_mediana": float(np.median(err)), "kendall_err_rel_max": float(err.max()),
            "kendall_err_rel_p95": float(np.percentile(err, 95))}


# ----------------------------------------------------------------------------------- metodi (dai CSV)

@functools.lru_cache(maxsize=None)
def _rows(p: Path) -> list[tuple[int, dict]]:
    with open(p, newline="") as f:
        return [(n, r) for n, r in enumerate(csv.DictReader(f), 2)]


def _one(rows, p: Path, **kw) -> tuple[int, dict] | None:
    hits = [(n, r) for n, r in rows if all(r.get(k) == v for k, v in kw.items())]
    if len(hits) > 1:
        raise SystemExit(f"{rel(p)}: piu' righe per {kw}: {[n for n, _ in hits]}")
    return hits[0] if hits else None


def method_records(view: str, seed: int) -> list[dict]:
    """Spearman con FR (senza crop) e i due delta 'tiene la taglia - toglie la taglia', con la fonte file:riga."""
    out = []

    def add(key, label, kind, hit, p, vcol, lcol, hcol, sign=1.0):
        if hit is None:
            out.append({"dominio": view, "chiave": key, "metodo": label, "tipo": kind, "punto": np.nan,
                        "ic_lo": np.nan, "ic_hi": np.nan, "n_righe": "", "seme": "", "fonte": "assente"})
            return
        n, r = hit
        v, lo, hi = (sign * float(r[c]) for c in (vcol, lcol, hcol))
        out.append({"dominio": view, "chiave": key, "metodo": label, "tipo": kind, "punto": v,
                    "ic_lo": min(lo, hi), "ic_hi": max(lo, hi), "n_righe": r.get("n_righe", r.get("n_rows", "")),
                    "seme": r.get("seme", r.get("seed", "")), "fonte": f"{rel(p)}:{n}"})

    sp = _rows(FM_SP)
    for c, lab in FM_COLS:
        add(c, lab, "rho", _one(sp, FM_SP, dominio=view, gt="fr", insieme_righe="nocrop", perturbazione="-",
                                passo="comune", colonna=c), FM_SP, "rho", "ic_lo", "ic_hi")
    pr = _rows(PR_SP)
    for c, lab in (("now_original", "NoW originale"), ("now_metrical", "NoW metrical")):
        add(c, lab, "rho", _one(pr, PR_SP, dominio=view, insieme_righe="nocrop", perturbazione="-", gt="fr",
                                kind="rho", colonna=c), PR_SP, "rho", "ic_lo", "ic_hi")
    ul = _rows(ULR_SP)
    add("ulrssm_pair_fr", "ULRSSM per coppia", "rho",
        _one(ul, ULR_SP, domain=view, group="nocrop", gt="fr", kind="rho", arm="baseline",
             distance="ULRSSM per coppia, FR (mm)"), ULR_SP, "arm_point", "arm_ci_low", "arm_ci_high")
    dl = _rows(FM_DL)
    add("d_icp", "ICP mm - ICP cs", "delta",
        _one(dl, FM_DL, dominio=view, gt="fr", insieme_righe="nocrop", perturbazione="-", passo="comune",
             tipo="scala_allineamento", colonna_a="mm_rigid_icp_chamfer", colonna_b="cs_rigid_icp_chamfer"),
        FM_DL, "delta", "ic_lo", "ic_hi")
    pd_ = _rows(PR_DL)
    add("d_now", "NoW metrical - NoW originale", "delta",
        _one(pd_, PR_DL, dominio=view, insieme_righe="nocrop", perturbazione="-", gt="fr",
             tipo="NoW originale - NoW metrical"), PR_DL, "delta", "ic_lo", "ic_hi", sign=-1.0)
    bad = [r["fonte"] for r in out if r["seme"] not in ("", str(seed))]
    if bad:
        raise SystemExit(f"{view}: seme diverso da {seed} in {bad}")
    return out


# ------------------------------------------------------------------------------------------- uscite

def ci(x: np.ndarray) -> tuple[float, float]:
    return fm.ci(x)


def f3(x: float, pct: bool = False) -> str:
    return "-" if not np.isfinite(x) else (f"{100 * x:.1f}" if pct else f"{x:.3f}")


def cell(r: dict, pct: bool = False) -> str:
    return f"{f3(r['punto'], pct)} [{f3(r['ic_lo'], pct)}, {f3(r['ic_hi'], pct)}]"


def write_md(G: list[dict], M: list[dict], K: dict, C: dict) -> None:
    g = {(r["dominio"], r["grandezza"]): r for r in G}
    m = {(r["dominio"], r["chiave"]): r for r in M}
    md = ["# Quota della taglia nella GT FR, per dominio (sola GT)", "",
          "Generato da `aau/size_share/size_share.py` (definizioni nel docstring). Coppie di identita' diverse a "
          "livello identita', GT FR (mm) e SR (d_P) di `datasets/CANONICAL_GT/eval`, S = centroid size di FR. "
          f"IC 95% percentile su {N_BOOT} repliche per soggetto, le stesse della matrice unificata (seme della T1 "
          "del dominio; FaMoS `bootstrap_counts(15, 1000, 1234)`). FaceVerse neutra ha le stesse chiavi e GT di "
          "FaceVerse con espressioni: i numeri della GT coincidono. FaMoS TEST: 15 soggetti, descrittivo. "
          "Niente Ava-256.", "",
          "## Tabella", "",
          "Quota della taglia = mean[(S_i - S_j)^2] / mean[FR^2] (%). rho = Spearman con la GT FR, righe senza crop. "
          "Delta = 'tiene la taglia' - 'la toglie'.", "",
          "| dominio | quota taglia % | CV di S % | rho(FR, taglia) | rho(FR, forma) | oracolo taglia | ICP mm | "
          "ICP cs | NoW orig. | NoW metr. | ULRSSM coppia | d_F non cal. | d_F cal. | ICP mm - cs | NoW metr. - orig. |",
          "|---" * 15 + "|"]
    for v in VIEWS:
        md.append(f"| {fm.DOMAIN[v]} | {cell(g[(v, 'quota_taglia')], True)} | {cell(g[(v, 'cv_S')], True)} | "
                  f"{cell(g[(v, 'rho_fr_taglia')])} | {cell(g[(v, 'rho_fr_forma')])} | " +
                  " | ".join(cell(m[(v, k)]) for k in ("oracle_size", "mm_rigid_icp_chamfer", "cs_rigid_icp_chamfer",
                                                         "now_original", "now_metrical", "ulrssm_pair_fr",
                                                         "factorized_s1234|form", "factorized_s1234|form_cal",
                                                         "d_icp", "d_now")) + " |")
    md += ["", "d_F: factorized s1234 (s2345 in `methods.csv`). ICP, oracolo e d_F sulla maschera comune della matrice "
           "(passo `comune`); NoW e ULRSSM sulle 99.000 righe senza crop dei loro CSV (FaMoS 105), stesso seme.", "",
           "## GT per dominio", "",
           "| dominio | soggetti | coppie | quota taglia % | quota forma % | SR mediana | SR IQR | Kendall: err. rel. "
           "mediana | p95 | max |", "|---" * 10 + "|"]
    for v in VIEWS:
        k = K[v]
        md.append(f"| {fm.DOMAIN[v]} | {C[v]['n_subjects']} | {C[v]['n_pairs']} | "
                  f"{cell(g[(v, 'quota_taglia')], True)} | {cell(g[(v, 'quota_forma')], True)} | "
                  f"{cell(g[(v, 'sr_mediana')])} | {cell(g[(v, 'sr_iqr')])} | {k['kendall_err_rel_mediana']:.2e} | "
                  f"{k['kendall_err_rel_p95']:.2e} | {k['kendall_err_rel_max']:.2e} |")
    md += ["", "Errore di Kendall = |(S_i - S_j)^2 + S_i S_j d_SR^2 - FR^2| / FR^2 sul punto: il resto e' il termine "
           "dei centroidi della rigida robusta (quota taglia + quota forma + quota centroidi = 1).", "",
           "## Controlli", ""]
    for v in VIEWS:
        c = C[v]
        extra = "" if v == "famos" else (
            f", GT delle righe della matrice contro quella per coppia di soggetti: FR {c['gt_rows_vs_identity_max_abs']['fr']:.1e}, "
            f"SR {c['gt_rows_vs_identity_max_abs']['sr']:.1e} (normalizzate); coppie di soggetti nelle righe "
            f"{c['subject_pairs_in_rows']}; soggetti = blmm.subjects: {c['subjects_equal_blmm']}")
        md.append(f"- {fm.DOMAIN[v]}: seme {c['seed']}, CV di S contro `baselines_param/controls_e2.json` "
                  f"{c['cv_vs_controls_e2']:.1e}{extra}.")
    md += ["", "## Fonti dei rho", "", "`methods.csv`, colonna `fonte` (file:riga, riga 1 = intestazione).", ""]
    (OUT / "results.md").write_text("\n".join(md) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ctrl_e2 = json.loads((EV / "baselines_param" / "controls_e2.json").read_text())
    G, M, K, C = [], [], {}, {}
    for v in VIEWS:
        P, counts, seed, ctrl = gt_view(v)
        V = replicas(P, counts)
        for q in GT_Q:
            lo, hi = ci(V[q])
            G.append({"dominio": v, "grandezza": q, "punto": float(V[q][0]), "ic_lo": float(lo), "ic_hi": float(hi),
                      "n_soggetti": ctrl["n_subjects"], "n_coppie": ctrl["n_pairs"], "seme": seed,
                      "n_boot": N_BOOT})
        K[v] = kendall_error(P)
        for k, x in K[v].items():
            G.append({"dominio": v, "grandezza": k, "punto": x, "ic_lo": np.nan, "ic_hi": np.nan,
                      "n_soggetti": ctrl["n_subjects"], "n_coppie": ctrl["n_pairs"], "seme": seed, "n_boot": 0})
        ctrl["cv_vs_controls_e2"] = abs(float(V["cv_S"][0]) - ctrl_e2["gt"][v]["S_cv"])
        C[v] = ctrl
        M += method_records(v, seed)
        print(f"[size-share] {v}: quota taglia {100 * V['quota_taglia'][0]:.1f}% CV {100 * V['cv_S'][0]:.2f}% "
              f"Kendall max {K[v]['kendall_err_rel_max']:.1e}", flush=True)
    for name, R in (("gt_size.csv", G), ("methods.csv", M)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(R[0]))
            w.writeheader()
            w.writerows(R)
    (OUT / "controls.json").write_text(json.dumps(C, indent=1, default=float) + "\n")
    write_md(G, M, K, C)
    print(f"[size-share] scritto {rel(OUT)}", flush=True)


if __name__ == "__main__":
    main()
