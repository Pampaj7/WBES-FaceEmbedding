#!/usr/bin/env python3
"""AUC same/different con CI bootstrap per soggetto sui csv di WS3a (Multiface).

Legge tutti i ``aau/runs/multiface_ws3a/<metrica>_<topoA>_<topoB>.csv`` e per ognuno
calcola tre AUC, che rispondono a tre domande diverse:

``auc_ab_vs_cd``  (a+b) contro (c+d) — la domanda di WS3a: la metrica distingue "stesso
                  soggetto" da "soggetto diverso" su scansioni reali, senza nessuna D_GT.
``auc_b_vs_c``    (b) contro (c) — il caso difficile, ed e' quello che conta.  In (b)
                  cambia l'espressione e l'identita' no; in (c) l'espressione e' la stessa
                  e cambia l'identita'.  Una metrica che misura la forma della faccia
                  invece dell'identita' finisce sotto 0.5 proprio qui.
``auc_a_vs_c``    (a) contro (c) — il caso facile: stessa espressione da entrambe le
                  parti, cambia solo l'identita'.  Serve da tetto: se anche questo e'
                  basso, il problema non e' l'espressione.

Due cose sulla lettura della tabella.  La riga ``bbox_proxy`` non e' una metrica ma un
controllo (vedi ``ws3a_geometric.py``): dice quanta della separazione si ottiene con quattro
numeri banali, e va guardata prima delle altre.  E i CI che vengono fuori tutti uguali non
si stampano come ``[1.000, 1.000]``: sono degeneri, non stretti (vedi ``format_auc``).

Convenzione: la distanza piccola deve voler dire "stesso soggetto", quindi
AUC = P(d_positivo < d_negativo) + 0.5 P(uguali), con i positivi che sono le coppie
"same".  0.5 e' il caso, sotto 0.5 la metrica e' invertita.

Bootstrap
---------
1000 repliche ricampionando i 13 SOGGETTI con reinserimento, non le coppie: le coppie non
sono indipendenti, ogni soggetto compare in migliaia di righe e un bootstrap per coppia
darebbe CI ottimisti di un fattore grosso.  Lo schema di peso e' quello gia' usato dal
repo in ``scripts/compute_bootstrap_ci.weighted_bootstrap_spearman``: pescati i soggetti,
ogni riga entra con peso ``count[subject_a] * count[subject_b]``.  Le AUC sono statistiche
di rango, quindi il peso si applica direttamente senza materializzare le ripetizioni.

Con 13 soggetti i CI sono larghi per costruzione: e' il numero di soggetti di Multiface,
non un difetto del calcolo, e va detto nel paper.

Delta vs clean
--------------
Con ``--baseline-pair tracked,tracked`` ogni riga porta anche l'AUC della stessa metrica e
dello stesso confronto su ``tracked->tracked`` e la differenza fra le due.  Serve al giro
``WBES_WS3A_PAIRS=hard``: da sola l'AUC di ``tracked->crop`` non dice se a farla scendere
e' la perturbazione o la metrica, il salto rispetto al caso pulito si'.  La coppia di
riferimento viene letta dal suo csv nella stessa out-root, quindi il giro clean va fatto
prima.

    aau/run.sh aau/multiface/ws3a_analysis.py            # su un nodo di calcolo
    python3 aau/multiface/ws3a_analysis.py --n-bootstrap 1000
    WBES_WS3A_PAIRS=hard aau/run.sh aau/multiface/ws3a_analysis.py \
        --summary-name summary_hard --baseline-pair tracked,tracked
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ws3a_common as common  # noqa: E402

# I tre confronti, come insiemi di classi positive (same) e negative (different).
COMPARISONS = {
    "auc_ab_vs_cd": (("a_same_subject_same_expression", "b_same_subject_diff_expression"),
                     ("c_diff_subject_same_expression", "d_diff_subject_diff_expression")),
    "auc_b_vs_c": (("b_same_subject_diff_expression",), ("c_diff_subject_same_expression",)),
    "auc_a_vs_c": (("a_same_subject_same_expression",), ("c_diff_subject_same_expression",)),
}

SUMMARY_FIELDS = (
    "metric", "topology_a", "topology_b", "comparison",
    "auc", "ci_low", "ci_high", "n_positive", "n_negative", "n_subjects", "n_bootstrap",
)

# Le due colonne in piu' quando c'e' una coppia di riferimento.  Restano fuori dal caso
# normale perche' senza riferimento sarebbero 72 celle vuote.
BASELINE_FIELDS = ("auc_clean", "delta_vs_clean")

# La colonna di delta mostrata nella tabella markdown: e' il confronto difficile, l'unico
# su cui il set pulito non era gia' saturo.
DELTA_COMPARISON = "auc_b_vs_c"

# La riga di controllo, che non e' una metrica di forma: vedi ws3a_geometric.py.
PROXY_METRIC = "bbox_proxy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--metrics", type=str, default=",".join(common.METRICS))
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ci", type=float, default=95.0, help="ampiezza del CI in percento")
    p.add_argument("--summary-name", type=str, default="summary",
                   help="nome senza estensione di summary.csv/.md nella out-root")
    p.add_argument("--baseline-pair", type=str, default="",
                   help="coppia di topologie di riferimento per il delta, es. tracked,tracked")
    return p.parse_args()


def weighted_auc(inv: np.ndarray, n_bins: int, is_positive: np.ndarray,
                 weights: np.ndarray) -> float:
    """P(d_pos < d_neg) + 0.5 P(d_pos == d_neg), con un peso per riga.

    Implementata sui ranghi invece che sulle coppie: le coppie di coppie sarebbero
    4000x4000 per confronto e per replica bootstrap.  ``inv`` e' l'indice del valore nella
    lista ordinata dei distinti (``np.unique(..., return_inverse=True)``), calcolato una
    volta sola fuori dal ciclo bootstrap perche' i valori non cambiano fra repliche: a
    cambiare sono solo i pesi.
    """
    w_pos = np.bincount(inv, weights=weights * is_positive, minlength=n_bins)
    w_neg = np.bincount(inv, weights=weights * ~is_positive, minlength=n_bins)
    total_pos, total_neg = w_pos.sum(), w_neg.sum()
    if total_pos <= 0 or total_neg <= 0:
        return math.nan
    # Peso dei positivi strettamente piu' piccoli di ogni valore, piu' meta' dei pari.
    w_pos_before = np.concatenate([[0.0], np.cumsum(w_pos)[:-1]])
    return float(np.sum(w_neg * (w_pos_before + 0.5 * w_pos)) / (total_pos * total_neg))


def bootstrap_auc(values, is_positive, subj_a, subj_b, n_subjects, n_bootstrap, ci, rng):
    """AUC puntuale e CI percentile, ricampionando i soggetti con reinserimento."""
    uniq, inv = np.unique(values, return_inverse=True)
    inv = np.asarray(inv, dtype=np.int64).ravel()
    point = weighted_auc(inv, len(uniq), is_positive, np.ones(len(values), dtype=np.float64))
    if n_bootstrap <= 0:
        return point, math.nan, math.nan

    replicas: list[float] = []
    for _ in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, n_subjects, size=n_subjects), minlength=n_subjects)
        # Peso della riga = molteplicita' del soggetto A per quella del soggetto B, come
        # in scripts/compute_bootstrap_ci.py.  Le righe a peso zero non contribuiscono.
        weights = (counts[subj_a] * counts[subj_b]).astype(np.float64)
        auc = weighted_auc(inv, len(uniq), is_positive, weights)
        if math.isfinite(auc):
            replicas.append(auc)
    if not replicas:
        return point, math.nan, math.nan
    tail = (100.0 - ci) / 2.0
    low, high = np.percentile(np.asarray(replicas, dtype=np.float64), [tail, 100.0 - tail])
    return point, float(low), float(high)


def load_table(path: Path):
    """(distanze, classi, soggetto_a, soggetto_b) da un csv di distanze."""
    rows = common.read_distances(path)
    values = np.asarray([float(r["distance"]) for r in rows], dtype=np.float64)
    classes = np.asarray([r["pair_class"] for r in rows])
    subject_a = np.asarray([r["subject_a"] for r in rows])
    subject_b = np.asarray([r["subject_b"] for r in rows])
    return values, classes, subject_a, subject_b


def analyze_file(path: Path, metric: str, topology_a: str, topology_b: str, args) -> list[dict]:
    values, classes, subject_a, subject_b = load_table(path)
    finite = np.isfinite(values)
    if not finite.all():
        print(f"[ws3a-an] {path.name}: {int((~finite).sum())}/{len(values)} distanze non finite, "
              f"escluse", flush=True)

    subjects = sorted(set(subject_a) | set(subject_b))
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    subj_a = np.asarray([subject_to_idx[s] for s in subject_a], dtype=np.int64)
    subj_b = np.asarray([subject_to_idx[s] for s in subject_b], dtype=np.int64)

    out: list[dict] = []
    for offset, (comparison, (positive, negative)) in enumerate(COMPARISONS.items()):
        mask = finite & np.isin(classes, positive + negative)
        is_positive = np.isin(classes[mask], positive)
        # Seme che dipende solo dal confronto, non dalla metrica: cosi' i soggetti pescati
        # sono gli stessi in tutte le righe della tabella e i CI sono appaiati.  Niente
        # hash() di stringhe, che e' randomizzato per processo.
        rng = np.random.default_rng(args.seed + 10_007 * offset)
        auc, ci_low, ci_high = bootstrap_auc(
            values[mask], is_positive, subj_a[mask], subj_b[mask],
            n_subjects=len(subjects), n_bootstrap=args.n_bootstrap, ci=args.ci, rng=rng,
        )
        out.append({
            "metric": metric, "topology_a": topology_a, "topology_b": topology_b,
            "comparison": comparison, "auc": auc, "ci_low": ci_low, "ci_high": ci_high,
            "n_positive": int(is_positive.sum()), "n_negative": int((~is_positive).sum()),
            "n_subjects": len(subjects), "n_bootstrap": args.n_bootstrap,
        })
    return out


# Sotto questa ampiezza il CI bootstrap e' degenere: tutte le repliche danno lo stesso
# valore, cioe' la separazione e' perfetta comunque si ripeschino i 13 soggetti.
DEGENERATE_CI_WIDTH = 1e-9


def format_auc(row: dict) -> str:
    """AUC con CI, ma senza stampare un intervallo degenere.

    Un `1.000 [1.000, 1.000]` promette una precisione che 13 soggetti non possono dare: il
    bootstrap non ha trovato variabilita' perche' la separazione e' perfetta in ogni
    replica, non perche' l'incertezza sia nulla.  Meglio dirlo.
    """
    auc, low, high = row["auc"], row["ci_low"], row["ci_high"]
    if not math.isfinite(low) or not math.isfinite(high):
        return f"{auc:.3f} [CI n.d.]"
    if high - low < DEGENERATE_CI_WIDTH:
        return f"{auc:.3f} [CI degenere]"
    return f"{auc:.3f} [{low:.3f}, {high:.3f}]"


def write_markdown(path: Path, rows: list[dict], metrics: list[str], baseline=None) -> None:
    """Una riga per metrica x coppia di topologie, tre colonne di AUC con CI.

    Con una coppia di riferimento si aggiunge in coda la colonna del delta sul confronto
    difficile: e' un numero solo, non tre, perche' e' quello che si guarda.  Gli altri due
    delta stanno comunque nel csv.
    """
    by_key = {(r["metric"], r["topology_a"], r["topology_b"], r["comparison"]): r for r in rows}
    columns = list(COMPARISONS)
    headers = [c.replace("auc_", "") for c in columns]
    if baseline is not None:
        headers.append(f"delta {DELTA_COMPARISON.replace('auc_', '')} vs clean")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# WS3a Multiface: AUC same-vs-different per metrica e topologia\n\n")
        fh.write("AUC = P(distanza same < distanza different), 0.5 = caso. CI 95% bootstrap "
                 "su 1000 repliche, ricampionando i 13 soggetti. \"CI degenere\" = tutte le "
                 "repliche danno lo stesso valore (separazione perfetta in ognuna): "
                 "l'intervallo non e' stretto, e' non informativo.\n\n")
        fh.write("La riga `bbox_proxy` non e' una metrica di forma ma un CONTROLLO: distanza "
                 "fra due vettori di 4 numeri (centro e diagonale del bounding box) presi "
                 "sulla mesh **gia' normalizzata maxabs**, cioe' la stessa mesh che vedono "
                 "chamfer, il latent e i render. Va letta per prima: dove il proxy e' alto "
                 "quanto una metrica, quella cella e' risolta da informazione banale.\n\n")
        fh.write(proxy_paragraph(rows, metrics))
        fh.write("Classi: (a) stesso soggetto stessa espressione, (b) stesso soggetto "
                 "espressione diversa, (c) soggetti diversi stessa espressione, (d) tutto "
                 "diverso.\n\n")
        if baseline is not None:
            fh.write(f"Coppie di topologie: {common.PAIR_SET}. L'ultima colonna e' "
                     f"AUC({DELTA_COMPARISON.replace('auc_', '')}) meno la stessa AUC della "
                     f"stessa metrica su {baseline[0]}->{baseline[1]} (il caso pulito): "
                     "negativa = la perturbazione fa perdere separazione.\n\n")
        fh.write("| metrica | topologie | " + " | ".join(headers) + " |\n")
        fh.write("|---|---|" + "---|" * len(headers) + "\n")
        for metric in metrics:
            for topology_a, topology_b in common.TOPOLOGY_PAIRS:
                cells = []
                for comparison in columns:
                    row = by_key.get((metric, topology_a, topology_b, comparison))
                    cells.append("--" if row is None else format_auc(row))
                if all(cell == "--" for cell in cells):
                    continue
                if baseline is not None:
                    row = by_key.get((metric, topology_a, topology_b, DELTA_COMPARISON))
                    delta = None if row is None else row.get("delta_vs_clean")
                    cells.append("--" if delta is None or not math.isfinite(delta)
                                 else f"{delta:+.3f}")
                fh.write(f"| {metric} | {topology_a}->{topology_b} | " + " | ".join(cells) + " |\n")


def proxy_paragraph(rows: list[dict], metrics: list[str]) -> str:
    """Il commento alla riga di controllo, scritto sui numeri di QUESTA corsa.

    La versione precedente descriveva il proxy a memoria e con un AUC preso da un'altra
    corsa (e da un'altra definizione del proxy, sulle unita' grezze).  Qui il testo si
    genera dalle righe appena calcolate, cosi' o e' vero o non c'e'.
    """
    by_key = {(r["metric"], r["topology_a"], r["topology_b"], r["comparison"]): r for r in rows}
    if not any(k[0] == PROXY_METRIC for k in by_key):
        return (f"> La riga `{PROXY_METRIC}` NON e' stata calcolata in questa corsa: senza di "
                f"lei la tabella non ha il suo metro, e le AUC qui sotto vanno lette come "
                f"valori assoluti e basta. Rilancia `ws3a_geometric.sbatch` con "
                f"`{PROXY_METRIC}` fra le metriche.\n\n")

    others = [m for m in metrics if m != PROXY_METRIC]
    lines = []
    for topology_a, topology_b in common.TOPOLOGY_PAIRS:
        proxy = by_key.get((PROXY_METRIC, topology_a, topology_b, DELTA_COMPARISON))
        if proxy is None or not math.isfinite(proxy["auc"]):
            continue
        beaten = [m for m in others
                  if (r := by_key.get((m, topology_a, topology_b, DELTA_COMPARISON))) is not None
                  and math.isfinite(r["auc"]) and r["auc"] <= proxy["auc"]]
        present = [m for m in others
                   if (m, topology_a, topology_b, DELTA_COMPARISON) in by_key]
        lines.append(f"> - `{topology_a}->{topology_b}`: proxy "
                     f"{proxy['auc']:.3f} sul confronto difficile "
                     f"({DELTA_COMPARISON.replace('auc_', '')}); "
                     f"{len(beaten)}/{len(present)} metriche non lo battono"
                     + (f" ({', '.join(sorted(beaten))})" if beaten else "") + "\n")
    header = (f"> **Quanto vale il metro.** Quattro numeri sulla mesh normalizzata, per "
              f"coppia di topologie:\n")
    return header + "".join(lines) + "\n"


def baseline_aucs(baseline, metrics: list[str], args) -> dict[tuple[str, str], float]:
    """AUC della coppia di riferimento per (metrica, confronto), dai suoi csv."""
    out: dict[tuple[str, str], float] = {}
    for metric in metrics:
        path = common.csv_path(metric, baseline[0], baseline[1], args.out_root)
        if not path.exists():
            print(f"[ws3a-an] riferimento assente: {path.name}, niente delta per {metric}",
                  flush=True)
            continue
        for row in analyze_file(path, metric, baseline[0], baseline[1], args):
            out[(metric, row["comparison"])] = row["auc"]
        print(f"[ws3a-an] riferimento {path.name}: fatto", flush=True)
    return out


def main() -> None:
    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    baseline = None
    if args.baseline_pair:
        parts = [p.strip() for p in args.baseline_pair.split(",") if p.strip()]
        if len(parts) != 2:
            raise SystemExit(f"--baseline-pair vuole 'topoA,topoB', ricevuto {args.baseline_pair!r}")
        baseline = (parts[0], parts[1])

    rows: list[dict] = []
    missing: list[str] = []
    for metric in metrics:
        for topology_a, topology_b in common.TOPOLOGY_PAIRS:
            path = common.csv_path(metric, topology_a, topology_b, args.out_root)
            if not path.exists():
                missing.append(path.name)
                continue
            rows.extend(analyze_file(path, metric, topology_a, topology_b, args))
            print(f"[ws3a-an] {path.name}: fatto", flush=True)
    if missing:
        print(f"[ws3a-an] {len(missing)} csv assenti, saltati: {missing}", flush=True)
    if not rows:
        raise SystemExit(f"nessun csv di distanze sotto {args.out_root}")

    fields = SUMMARY_FIELDS
    if baseline is not None:
        reference = baseline_aucs(baseline, metrics, args)
        for row in rows:
            clean = reference.get((row["metric"], row["comparison"]), math.nan)
            row["auc_clean"] = clean
            row["delta_vs_clean"] = row["auc"] - clean
        fields = SUMMARY_FIELDS + BASELINE_FIELDS

    csv_out = args.out_root / f"{args.summary_name}.csv"
    md_out = args.out_root / f"{args.summary_name}.md"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k])
                             for k in fields})
    write_markdown(md_out, rows, metrics, baseline=baseline)
    print(f"[ws3a-an] {len(rows)} righe -> {csv_out} e {md_out}")


if __name__ == "__main__":
    main()
