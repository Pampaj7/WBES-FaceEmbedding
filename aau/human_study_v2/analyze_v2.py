#!/usr/bin/env python3
"""Studio umano v2: accordo di ogni GT con le scelte umane, IC bootstrap per partecipante, test appaiati.

Regole fissate in ``PROTOCOL.md`` prima dei dati. In breve:
  - **filtro**: il Google Form e' condiviso con la v1, contano solo i payload con ``study_version == "v2"`` e con
    l'impronta ``triplets_hash`` delle triplette analizzate;
  - **esclusione** come la v1: fuori chi sbaglia piu' di ``--max-control-errors`` (1) controlli sui 4; le prove
    della sessione di prova non contano; un codice partecipante ripetuto vale una volta (la sessione piu' lunga);
  - **accordo** di una GT = quota delle RISPOSTE di test (tutte le triplette di test, gli stessi denominatori
    per tutte le GT) in cui la scelta umana coincide con la risposta attesa dalla GT; IC 95% bootstrap sui
    PARTECIPANTI (l'unita' campionaria: le risposte della stessa persona sono correlate), le stesse repliche per
    tutte le GT, quindi le differenze fra GT sono appaiate;
  - **test primari**: per ogni strato ``X_vs_Y`` (F_vs_S principale, S_vs_maxabs) la quota delle risposte che sta
    con X dentro lo strato (X e Y vi danno risposte opposte, quindi accordo(X) - accordo(Y) = 2 quota - 1), H0
    quota = 0.5, p a due code dal test di permutazione a segni ribaltati per partecipante (``--n-perm``),
    correzione di Holm sugli strati, alfa 0.05;
  - **secondari**: accordo complessivo di ogni GT e differenze appaiate fra tutte le coppie di GT (IC bootstrap,
    P(diff <= 0), p a segni ribaltati); sensibilita' col bootstrap incrociato partecipanti x triplette;
    maggioranza per tripletta e kappa di Fleiss come la v1.

    aau/human_study_v2/analyze_v2.py --responses-dir <dir con i JSON della pagina>
    aau/human_study_v2/analyze_v2.py --form-csv <export CSV del foglio del Google Form>
    aau/human_study_v2/analyze_v2.py --self-test          # partecipanti simulati, nessun dato reale
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

THIS_DIR = Path(__file__).resolve().parent
CHOICES = ("b", "c")
STUDY_VERSION = "v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--triplets", type=Path, default=THIS_DIR / "triplets.json")
    p.add_argument("--responses-dir", type=Path, default=None, help="un JSON per partecipante (payload della pagina)")
    p.add_argument("--form-csv", type=Path, default=None, help="export CSV del foglio delle risposte del Google Form")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <dir dei dati>/analysis_v2")
    p.add_argument("--gts", default="", help="GT da valutare (default: tutte quelle delle triplette)")
    p.add_argument("--max-control-errors", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--min-votes", type=int, default=3, help="solo per la maggioranza (secondaria, come la v1)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--sim-participants", type=int, default=40)
    p.add_argument("--sim-follow", default="all", help="GT seguita dai partecipanti simulati ('all' = SCENARIOS)")
    p.add_argument("--sim-accuracy", type=float, default=0.75)
    return p.parse_args()


# ----------------------------------------------------------------------------- lettura

def load_triplets(path: Path) -> tuple[dict, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["meta"], {t["id"]: t for t in payload["triplets"]}


def _payloads_in(obj) -> list[dict]:
    if isinstance(obj, list):
        return [d for x in obj for d in _payloads_in(x)]
    if isinstance(obj, dict) and isinstance(obj.get("trials"), list):
        return [obj]
    return []


def load_responses_dir(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.glob("*.json")):
        for doc in _payloads_in(json.loads(p.read_text(encoding="utf-8"))):
            doc["_source"] = p.name
            out.append(doc)
    if not out:
        raise SystemExit(f"[analyze-v2] nessun payload con 'trials' in {d}")
    return out


def load_form_csv(path: Path) -> list[dict]:
    """Ogni riga del foglio: la cella che e' un JSON con ``trials`` e' il payload (la domanda "Paragrafo")."""
    csv.field_size_limit(sys.maxsize)
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for k, row in enumerate(csv.reader(fh)):
            for cell in row:
                cell = cell.strip()
                if not cell.startswith("{"):
                    continue
                try:
                    doc = json.loads(cell)
                except json.JSONDecodeError:
                    continue
                for d in _payloads_in(doc):
                    d["_source"] = f"{path.name}:riga{k + 1}"
                    out.append(d)
    if not out:
        raise SystemExit(f"[analyze-v2] nessun payload JSON nel CSV {path}")
    return out


def only_v2(docs: list[dict], meta: dict) -> tuple[list[dict], int, int]:
    """Il Google Form e' condiviso con la v1: valgono solo i payload con ``study_version == "v2"`` e fatti sulle
    triplette di ``triplets.json`` (``triplets_hash``): una sessione su un insieme rigenerato con contenuto diverso
    attribuirebbe le risposte alle triplette sbagliate. (kept, n altre versioni, n altre triplette)"""
    v2 = [d for d in docs if d.get("study_version") == STUDY_VERSION]
    keep = [d for d in v2 if d.get("triplets_hash") == meta.get("triplets_hash")]
    return keep, len(docs) - len(v2), len(v2) - len(keep)


def dedupe(docs: list[dict]) -> tuple[list[dict], int]:
    """Un codice partecipante vale una volta: la sessione con piu' prove (poi la prima)."""
    best: dict[str, dict] = {}
    for d in docs:
        code = str(d.get("participant") or d["_source"])
        if code not in best or len(d["trials"]) > len(best[code]["trials"]):
            best[code] = d
    return list(best.values()), len(docs) - len(best)


# ---------------------------------------------------------------------------- esclusione

def screen(docs: list[dict], triplets: dict, max_errors: int) -> tuple[list[dict], list[dict]]:
    rows, kept = [], []
    for d in docs:
        n_c = ok_c = n_t = 0
        unknown = 0
        for tr in d["trials"]:
            t = triplets.get(tr.get("triplet_id"))
            if t is None:
                unknown += 1
                continue
            if t["kind"] == "control":
                n_c += 1
                ok_c += int(tr["choice"] == t["metrics"]["F"]["expected"])
            elif t["kind"] == "test":
                n_t += 1
        rts = [tr["rt_ms"] for tr in d["trials"] if "rt_ms" in tr]
        keep = n_c > 0 and (n_c - ok_c) <= max_errors and unknown == 0
        rows.append({"participant": d.get("participant", d["_source"]), "source": d["_source"], "n_test": n_t,
                     "n_control": n_c, "control_correct": ok_c, "unknown_triplets": unknown,
                     "median_rt_ms": float(np.median(rts)) if rts else math.nan, "kept": bool(keep)})
        if keep:
            kept.append(d)
    return kept, rows


# ------------------------------------------------------------------------------- analisi

class Votes:
    """Matrici partecipanti x triplette di test: ``seen``, ``chose_c``; ``hit(g)`` = scelta = attesa della GT."""

    def __init__(self, docs: list[dict], triplets: dict):
        self.tids = sorted(t for t, v in triplets.items() if v["kind"] == "test")
        pos = {t: k for k, t in enumerate(self.tids)}
        self.seen = np.zeros((len(docs), len(self.tids)), dtype=bool)
        self.chose_c = np.zeros_like(self.seen)
        for i, d in enumerate(docs):
            for tr in d["trials"]:
                k = pos.get(tr["triplet_id"])
                if k is not None:
                    self.seen[i, k] = True
                    self.chose_c[i, k] = tr["choice"] == "c"
        self.triplets = triplets
        self.type = np.array([triplets[t]["disagreement_type"] for t in self.tids])

    def expected_c(self, g: str) -> np.ndarray:
        return np.array([self.triplets[t]["metrics"][g]["expected"] == "c" for t in self.tids])

    def hit(self, g: str) -> np.ndarray:
        return (self.seen & (self.chose_c == self.expected_c(g)[None])).astype(np.float64)


def weighted_rate(H: np.ndarray, S: np.ndarray, CP: np.ndarray, CT: np.ndarray | None) -> np.ndarray:
    """Quota pesata per repliche: sum_pt cp ct H / sum_pt cp ct S; CP (R, P), CT (R, T) o None (pesi 1)."""
    if CT is None:
        return (CP @ H.sum(1)) / (CP @ S.sum(1))
    return np.einsum("rp,pt,rt->r", CP, H, CT) / np.einsum("rp,pt,rt->r", CP, S, CT)


def boot_counts(n: int, reps: int, rng) -> np.ndarray:
    """(reps + 1, n) moltiplicita' di ricampionamento, riga 0 = il punto (tutti 1)."""
    return np.vstack([np.ones(n)] + [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(reps)])


def crossed_se(H: np.ndarray, S: np.ndarray, reps: int, rng) -> dict:
    """Errore standard della quota sum H / sum S per un disegno incrociato partecipanti x triplette (H, S: (P, T)
    conteggi). V = V_P + V_T - V_0 (Owen 2007; Owen e Eckles 2012): V_P e V_T dai bootstrap sui soli
    partecipanti e sulle sole triplette, V_0 = q (1 - q) / n la varianza binomiale delle risposte indipendenti,
    che ciascuno dei due contiene una volta e la somma conterebbe due (il bootstrap "pigeonhole" che ricampiona
    insieme righe e colonne la conta due volte ed e' conservativo: alfa empirico 0.006-0.033 in power_v2.md).
    Mai sotto il piu' grande dei due bootstrap separati."""
    P, T = H.shape
    q = H.sum() / S.sum()
    vp = float(np.var(weighted_rate(H, S, boot_counts(P, reps, rng)[1:], None), ddof=1))
    vt = float(np.var(weighted_rate(H.T, S.T, boot_counts(T, reps, rng)[1:], None), ddof=1))
    v0 = float(q * (1.0 - q) / S.sum())
    v = max(vp + vt - v0, vp, vt)
    df = satterthwaite(v, vp, vt, int((S.sum(1) > 0).sum()), int((S.sum(0) > 0).sum()))
    return {"se": math.sqrt(v), "df": df, "v_participants": vp, "v_triplets": vt, "v_iid": v0}


def satterthwaite(v: float, vp: float, vt: float, n_p: int, n_t: int) -> float:
    """Gradi di liberta' di Satterthwaite della varianza V_P + V_T (- V_0): il p si legge su una t, non sulla
    normale, che con pochi partecipanti e' liberale (power_v2.md)."""
    den = vp ** 2 / max(n_p - 1, 1) + vt ** 2 / max(n_t - 1, 1)
    return float(max(v ** 2 / den, 2.0)) if den > 0 else float(max(n_p - 1, 2))


def sign_flip_p(d: np.ndarray, n: np.ndarray, reps: int, rng) -> float:
    """p a due code di sum_p d_p / sum_p n_p = 0 col ribaltamento dei segni per partecipante."""
    obs = abs(d.sum())
    flips = rng.choice((-1.0, 1.0), size=(reps, len(d)))
    return float((1 + (np.abs(flips @ d) >= obs - 1e-12).sum()) / (1 + reps))


def holm(p: dict, alpha: float) -> dict:
    order = sorted(p, key=p.get)
    out, m, stop = {}, len(p), False
    for k, key in enumerate(order):
        thr = alpha / (m - k)
        rej = (not stop) and p[key] <= thr
        stop = stop or not rej
        out[key] = {"p": p[key], "holm_threshold": thr, "reject": rej,
                    "p_holm": min(1.0, max((m - j) * p[order[j]] for j in range(k + 1)))}
    return out


def ci(v: np.ndarray) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (math.nan, math.nan)


def fleiss_kappa(cb: np.ndarray, cc: np.ndarray) -> float:
    n = cb + cc
    k = n >= 2
    if k.sum() < 2:
        return math.nan
    b, c, n = cb[k].astype(float), cc[k].astype(float), n[k].astype(float)
    p_bar = float(((b ** 2 + c ** 2 - n) / (n * (n - 1.0))).mean())
    pb = float(b.sum() / n.sum())
    pe = pb ** 2 + (1.0 - pb) ** 2
    return math.nan if math.isclose(pe, 1.0) else (p_bar - pe) / (1.0 - pe)


def analyse(docs: list[dict], triplets: dict, meta: dict, gts: list[str], args) -> dict:
    v = Votes(docs, triplets)
    rng = np.random.default_rng(args.seed)
    P, T = v.seen.shape
    CP = boot_counts(P, args.n_bootstrap, rng)
    CTx = boot_counts(T, args.n_bootstrap, rng)          # bootstrap incrociato (sensibilita')
    S = v.seen.astype(np.float64)
    H = {g: v.hit(g) for g in gts}
    n_p = S.sum(1)
    res = {"n_participants": P, "n_test_triplets": T, "n_responses": int(S.sum()),
           "n_triplets_seen": int((S.sum(0) > 0).sum()), "responses_per_participant": float(n_p.mean()),
           "overall": [], "paired": [], "strata": []}
    rep = {g: weighted_rate(H[g], S, CP, None) for g in gts}
    repx = {g: weighted_rate(H[g], S, CP, CTx) for g in gts}
    for g in gts:
        lo, hi = ci(rep[g][1:])
        xlo, xhi = ci(repx[g][1:])
        res["overall"].append({"gt": g, "agreement": float(rep[g][0]), "ci_low": lo, "ci_high": hi,
                               "ci_low_crossed": xlo, "ci_high_crossed": xhi})
    for i, a in enumerate(gts):
        for b in gts[i + 1:]:
            dv = rep[a] - rep[b]
            lo, hi = ci(dv[1:])
            d = (H[a] - H[b]).sum(1)
            res["paired"].append({"a": a, "b": b, "diff": float(dv[0]), "ci_low": lo, "ci_high": hi,
                                  "p_boot_le0": float((dv[1:] <= 0).mean()),
                                  "p_signflip": sign_flip_p(d, n_p, args.n_perm, rng),
                                  "n_disagreeing_triplets": int((v.expected_c(a) != v.expected_c(b)).sum())})
    p_primary = {}
    for st in meta["strata"]:
        label, x, y = st["label"], st["x"], st["y"]
        m = v.type == label
        Sm = S[:, m]
        Hx, Hy = H[x][:, m] if x in H else v.hit(x)[:, m], H[y][:, m] if y in H else v.hit(y)[:, m]
        if not np.allclose(Hx + Hy, Sm):
            raise AssertionError(f"{label}: {x} e {y} non sono opposte su tutte le triplette dello strato")
        # PRIMARIO: varianza per disegno incrociato partecipanti x triplette (crossed_se): le triplette sono un
        # campione anche loro, e ricampionare solo i partecipanti sottostima la varianza (alfa reale fino a 0.20
        # con eterogeneita' fra triplette, power_v2.md).
        r = weighted_rate(Hx, Sm, CP, None)
        cx = crossed_se(Hx, Sm, args.n_bootstrap, rng)
        se, df = cx["se"], cx["df"]
        z = (float(r[0]) - 0.5) / se if se > 0 else math.inf
        p = float(2.0 * student_t.sf(abs(z), df))
        p_primary[label] = p
        d = (Hx - Hy).sum(1)
        lo, hi = ci(r[1:])
        tq = float(student_t.ppf(0.975, df))
        xlo, xhi = float(r[0]) - tq * se, float(r[0]) + tq * se
        rec = {"type": label, "x": x, "y": y, "share_with_x": float(r[0]), "ci_low_crossed": xlo,
               "ci_high_crossed": xhi, "se_crossed": se, "df_crossed": df, "z_crossed": z, "p_crossed": p,
               "var_components": {k: cx[k] for k in ("v_participants", "v_triplets", "v_iid")}, "ci_low": lo,
               "ci_high": hi, "diff_x_minus_y": float(2 * r[0] - 1),
               "p_signflip": sign_flip_p(d, Sm.sum(1), args.n_perm, rng), "n_responses": int(Sm.sum()),
               "n_participants": int((Sm.sum(1) > 0).sum()), "n_triplets_seen": int((Sm.sum(0) > 0).sum())}
        rec["others_share"] = {g: float((H[g][:, m].sum()) / max(Sm.sum(), 1)) for g in gts}
        res["strata"].append(rec)
    res["holm"] = holm(p_primary, args.alpha)
    cb, cc = (S * (1 - v.chose_c)).sum(0), (S * v.chose_c).sum(0)
    usable = (cb + cc >= args.min_votes) & (cb != cc)
    maj_c = cc > cb
    res["majority"] = {"n_usable": int(usable.sum()), "min_votes": args.min_votes,
                       "agreement": {g: (float((v.expected_c(g)[usable] == maj_c[usable]).mean())
                                         if usable.any() else math.nan) for g in gts}}
    res["fleiss_kappa"] = fleiss_kappa(cb, cc)
    return res


# -------------------------------------------------------------------------------- uscite

def write_report(out_dir: Path, args, meta, screening, res, n_dup) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    hs = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "triplets": str(args.triplets),
          "triplets_meta": {k: meta.get(k) for k in ("generated_at", "seed", "gt_status", "types", "n_test")},
          "rules": {"max_control_errors": args.max_control_errors, "alpha": args.alpha,
                    "n_bootstrap": args.n_bootstrap, "n_perm": args.n_perm, "seed": args.seed},
          "duplicates_dropped": n_dup, "screening": screening, "result": res}
    (out_dir / "analysis_v2.json").write_text(json.dumps(hs, indent=1), encoding="utf-8")
    with open(out_dir / "agreement_v2.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res["overall"][0]))
        w.writeheader()
        w.writerows(res["overall"])
    kept = sum(r["kept"] for r in screening)
    L = ["# Studio umano v2: accordo delle GT con le scelte umane", "",
         f"Triplette: {meta.get('generated_at')}, stato delle GT: **{meta.get('gt_status')}**.",
         f"{kept} partecipanti tenuti su {len(screening)} (fuori: piu' di {args.max_control_errors} errori sui "
         f"controlli; {n_dup} sessioni duplicate scartate). {res['n_responses']} risposte di test, "
         f"{res['n_triplets_seen']} triplette viste su {res['n_test_triplets']}.", "",
         "## Test primari: dentro ogni strato, quota delle risposte che sta con X", "",
         "PRIMARIO: errore standard per disegno incrociato partecipanti x triplette (V_P + V_T - V_0, "
         "`crossed_se`), IC 95% e p a due code su una t con gradi di liberta' di Satterthwaite, Holm sugli "
         "strati. Secondari: IC bootstrap sui soli partecipanti e p a segni ribaltati per partecipante (non "
         "tengono conto della variabilita' fra triplette).", "",
         "| strato | quota con X | IC 95% incrociato | p incrociato | p Holm | rifiuta H0 | accordo X - Y | "
         "IC partecipanti | p segni | risposte |",
         "|---|---:|---|---:|---:|---|---:|---|---:|---:|"]
    for r in res["strata"]:
        h = res["holm"][r["type"]]
        L.append(f"| `{r['type']}` | {r['share_with_x']:.3f} | "
                 f"[{r['ci_low_crossed']:.3f}, {r['ci_high_crossed']:.3f}] | "
                 f"{r['p_crossed']:.4f} | {h['p_holm']:.4f} | {'si' if h['reject'] else 'no'} | "
                 f"{r['diff_x_minus_y']:+.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['p_signflip']:.4f} | "
                 f"{r['n_responses']} |")
    L += ["", "## Accordo complessivo (tutte le triplette di test)", "",
          "| GT | accordo | IC 95% partecipanti | IC incrociato (pigeonhole, conservativo) |", "|---|---:|---|---|"]
    for r in sorted(res["overall"], key=lambda r: -r["agreement"]):
        L.append(f"| {r['gt']} | {r['agreement']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | "
                 f"[{r['ci_low_crossed']:.3f}, {r['ci_high_crossed']:.3f}] |")
    show = set(meta.get("principal_gts", [])) | {"EDM_s", "unified"}
    L += ["", "## Differenze appaiate (secondarie)", "",
          f"Fra {', '.join(sorted(show))}; tutte le coppie in `analysis_v2.json`.", "",
          "| A | B | accordo A - B | IC 95% | P(diff <= 0) | p segni | triplette in disaccordo |",
          "|---|---|---:|---|---:|---:|---:|"]
    for r in [r for r in res["paired"] if r["a"] in show and r["b"] in show]:
        L.append(f"| {r['a']} | {r['b']} | {r['diff']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                 f"{r['p_boot_le0']:.3f} | {r['p_signflip']:.4f} | {r['n_disagreeing_triplets']} |")
    maj = res["majority"]
    L += ["", f"Maggioranza per tripletta (come la v1, >= {maj['min_votes']} voti, niente parita'): "
          f"{maj['n_usable']} triplette; " + ", ".join(f"{g} {a:.3f}" for g, a in maj["agreement"].items()) + ".",
          f"Kappa di Fleiss: {res['fleiss_kappa']:.3f}.", "", "## Partecipanti", "",
          "| partecipante | test | controlli giusti | RT mediano (ms) | tenuto |", "|---|---:|---:|---:|---|"]
    for r in screening:
        L.append(f"| {r['participant']} | {r['n_test']} | {r['control_correct']}/{r['n_control']} | "
                 f"{r['median_rt_ms']:.0f} | {'si' if r['kept'] else 'NO'} |")
    md = out_dir / "analysis_v2.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    return md


# ------------------------------------------------------------------- partecipanti simulati

def simulate(triplets: dict, meta: dict, out_dir: Path, args) -> None:
    """Partecipanti che seguono ``--sim-follow`` con probabilita' ``--sim-accuracy``; l'ultimo risponde a caso e
    sbaglia 2 controlli (deve essere escluso); il primo compare due volte (duplicato); un payload della v1 e uno
    su triplette con un'altra impronta devono essere ignorati. Stessa composizione della
    sessione della pagina: le quote per strato del meta, 4 controlli, 3 prove di prova."""
    rng = np.random.default_rng(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_type = {}
    for t in triplets.values():
        by_type.setdefault((t["kind"], t["disagreement_type"]), []).append(t["id"])
    controls = by_type[("control", "unanime")]
    practice = by_type[("practice", "unanime")]
    for p in range(args.sim_participants):
        sloppy = p == args.sim_participants - 1
        trials = []
        for tid in rng.choice(practice, 3, replace=False):
            trials.append({"triplet_id": str(tid), "kind": "practice", "shown_left": "b",
                           "choice": triplets[tid]["metrics"]["F"]["expected"], "rt_ms": 5000})
        ids = [str(x) for ty, q in meta["session_quota"].items()
               for x in rng.choice(by_type[("test", ty)], q, replace=False)]
        ids += [str(x) for x in rng.choice(controls, 4, replace=False)]
        n_ctrl = 0
        for tid in rng.permutation(ids):
            t = triplets[tid]
            if t["kind"] == "control":
                n_ctrl += 1
                truth = t["metrics"]["F"]["expected"]
                ch = ("c" if truth == "b" else "b") if (sloppy and n_ctrl <= 2) else truth
            elif sloppy:
                ch = CHOICES[int(rng.integers(0, 2))]
            else:
                truth = t["metrics"][args.sim_follow]["expected"]
                ch = truth if rng.random() < args.sim_accuracy else ("c" if truth == "b" else "b")
            trials.append({"triplet_id": tid, "kind": t["kind"], "shown_left": "b", "choice": ch,
                           "rt_ms": int(rng.normal(9000, 2000))})
        doc = {"study_version": STUDY_VERSION, "triplets_hash": meta["triplets_hash"], "participant": f"PSIM{p:03d}",
               "trials": trials,
               "user_agent": "simulated"}
        (out_dir / f"PSIM{p:03d}.json").write_text(json.dumps(doc), encoding="utf-8")
        if p == 0:
            short = dict(doc, trials=trials[:10])
            (out_dir / "PSIM000_partial.json").write_text(json.dumps(short), encoding="utf-8")
            v1 = {"participant": "PV1", "trials": [{"triplet_id": "t0001", "choice": "b"}]}   # payload della v1
            (out_dir / "PV1.json").write_text(json.dumps(v1), encoding="utf-8")
            stale = dict(doc, participant="PSTALE", triplets_hash="0" * 16)          # triplette rigenerate
            (out_dir / "PSTALE.json").write_text(json.dumps(stale), encoding="utf-8")


# Comportamenti simulati e attese per strato: +1 = vince X con H0 rifiutata, -1 = vince Y, 0 = H0 NON rifiutata
# (lo strato non deve confondere quel comportamento con il proprio contrasto).
SCENARIOS = {
    "S": {"F_vs_S": -1, "S_vs_maxabs": +1, "F_vs_size": +1},
    "F": {"F_vs_S": +1, "S_vs_maxabs": 0, "F_vs_size": +1},
    "size_only": {"F_vs_S": +1, "S_vs_maxabs": 0, "F_vs_size": -1},
}


def run_scenario(args, meta, triplets, gts, follow: str, tmp: Path) -> tuple[list[str], dict]:
    d = tmp / f"responses_{follow}"
    args.sim_follow = follow
    simulate(triplets, meta, d, args)
    docs, n_other, n_stale = only_v2(load_responses_dir(d), meta)
    docs, n_dup = dedupe(docs)
    kept, screening = screen(docs, triplets, args.max_control_errors)
    res = analyse(kept, triplets, meta, gts, args)
    md = write_report(tmp / f"analysis_{follow}", args, meta, screening, res, n_dup)
    fail = []
    if (n_other, n_stale, n_dup) != (1, 1, 1):
        fail.append(f"scartati attesi 1 v1, 1 su triplette rigenerate, 1 duplicato: {n_other}, {n_stale}, {n_dup}")
    if len(kept) != args.sim_participants - 1:
        fail.append(f"attesi {args.sim_participants - 1} tenuti, trovati {len(kept)}")
    for r in res["strata"]:
        want = SCENARIOS.get(follow, {}).get(r["type"])
        rej = res["holm"][r["type"]]["reject"]
        got = 0 if not rej else (1 if r["share_with_x"] > 0.5 else -1)
        if want is not None and got != want:
            fail.append(f"{r['type']}: atteso {want:+d}, ottenuto {got:+d} (quota {r['share_with_x']:.3f}, "
                        f"p {r['p_crossed']:.4f})")
        if not (r["ci_low_crossed"] <= r["share_with_x"] <= r["ci_high_crossed"]):
            fail.append(f"{r['type']}: stima fuori dal proprio IC")
    return fail, {"md": md, "res": res}


def run_self_test(args, meta, triplets, gts) -> int:
    """Per ogni comportamento di SCENARIOS (``--sim-follow`` ne sceglie uno solo; ``all`` = tutti) i partecipanti
    simulati devono produrre esattamente gli esiti attesi: in particolare chi segue F o la sola taglia NON deve
    produrre un effetto in ``S_vs_maxabs``."""
    follows = list(SCENARIOS) if args.sim_follow == "all" else [args.sim_follow]
    fails = 0
    with tempfile.TemporaryDirectory(prefix="hs2_selftest_") as tmp:
        for f in follows:
            fail, out = run_scenario(args, meta, triplets, gts, f, Path(tmp))
            rows = ", ".join(f"{r['type']} {r['share_with_x']:.3f} (p {r['p_crossed']:.4f})"
                             for r in out["res"]["strata"])
            print(f"[self-test] segue {f}: {rows}")
            for line in fail:
                print(f"[self-test] FALLITO ({f}): {line}", file=sys.stderr)
            fails += bool(fail)
            if f == follows[-1]:
                print(out["md"].read_text(encoding="utf-8")[:3000])
    print(f"[self-test] {'OK' if not fails else 'FALLITO'}: {len(follows)} comportamenti, {args.sim_participants} "
          f"simulati ciascuno")
    return 1 if fails else 0


def main() -> int:
    args = parse_args()
    meta, triplets = load_triplets(args.triplets)
    gts = [g for g in args.gts.split(",") if g.strip()] or list(meta["gts"])
    if args.self_test:
        return run_self_test(args, meta, triplets, gts)
    if (args.responses_dir is None) == (args.form_csv is None):
        raise SystemExit("[analyze-v2] serve esattamente uno fra --responses-dir e --form-csv")
    docs = load_form_csv(args.form_csv) if args.form_csv else load_responses_dir(args.responses_dir)
    docs, n_other, n_stale = only_v2(docs, meta)
    docs, n_dup = dedupe(docs)
    kept, screening = screen(docs, triplets, args.max_control_errors)
    print(f"[analyze-v2] ignorati: {n_other} payload di altre versioni, {n_stale} su triplette diverse da "
          f"{args.triplets.name}; sessioni v2 {len(docs)} (+{n_dup} duplicate), tenute {len(kept)}", flush=True)
    if not kept:
        raise SystemExit("[analyze-v2] nessun partecipante ha superato i controlli")
    res = analyse(kept, triplets, meta, gts, args)
    src = args.form_csv.parent if args.form_csv else args.responses_dir
    md = write_report(args.out_dir or (src / "analysis_v2"), args, meta, screening, res, n_dup)
    print(md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
