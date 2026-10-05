#!/usr/bin/env python3
"""Da matrici di distanza 100x100 a Spearman/Pearson vs D_GT con CI bootstrap.

Unico punto in cui si calcolano i numeri della Tabella 2 estesa, per tutte le baseline.
Il bootstrap NON e' riscritto: ``weighted_bootstrap_spearman`` viene importata da
``scripts/compute_bootstrap_ci.py``, cioe' dalla stessa funzione che ha prodotto i CI
della Tabella 2 del paper.  Per Pearson quella funzione non esiste: invece di duplicarla
si scambia temporaneamente ``finite_spearman`` dentro il modulo (vedi ``_corr_backend``),
cosi' il ricampionamento dei soggetti resta byte per byte lo stesso e cambia solo la
correlazione calcolata.

Per ogni setting della Tabella 2 le coppie sono l'unione, su tutte le coppie ordinate di
topologie del setting, delle 4950 coppie di soggetti i<j: 4950 righe per
original->original e 99000 per il no-crop cross-topology, gli stessi conteggi di
``paper_artifacts/bootstrap_ci/bootstrap_ci.csv``.  Le altre due colonne spezzano la
seconda (59400 righe la tassellazione, 39600 la perturbazione); ``n_topology_pairs`` dice
su quante coppie ordinate di topologie ogni cella e' stata davvero calcolata, ed e' scritto
sia nel csv sia in testa al .tex, perche' una metrica che copre meno coppie delle altre non
ha una cella confrontabile con le loro (e' il caso di ``chamfer_sq``, vedi
``PARTIAL_COVERAGE_NOTE``).

  aau/run_baselines.sh aau/baselines/rank_from_matrix.py --metrics chamfer,varifold
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

BOOTSTRAP_SRC = common.REPO_ROOT / "scripts" / "compute_bootstrap_ci.py"

# Riferimenti pubblicati (paper/main_short.tex, tab:alignment_effect) per il gate di
# validazione: Chamfer raw deve ricadere dentro il suo CI.
PAPER_REFERENCE = {
    ("chamfer", "original_to_original"): (0.729, 0.667, 0.788),
    ("chamfer", "nocrop_cross_topology"): (0.552, 0.488, 0.606),
}

# Intestazioni delle colonne, nell'ordine di common.SETTINGS.
SETTING_LABELS = {
    "original_to_original": "Original-to-original",
    "nocrop_cross_topology": "No-crop cross-topology",
    "tessellation_cross_topology": "Tessellation only",
    "perturbation_cross_topology": "Perturbation (noisy)",
}

PARTIAL_COVERAGE_NOTE = (
    "riga a copertura PARZIALE: la cella e' una media su meno coppie ordinate di "
    "topologie delle altre righe, quindi NON e' confrontabile con loro."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--metrics", type=str, default="", help="vuoto = tutte quelle trovate in matrices/")
    p.add_argument("--settings", type=str, default=",".join(common.SETTINGS))
    p.add_argument("--subject-set", type=str, default="heldout", choices=common.SUBJECT_SETS,
                   help="heldout = split del repo; facebench_first100 = i soggetti della Tabella 2")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--strict", action="store_true",
                   help="esce 1 se una coppia di topologie richiesta manca")
    return p.parse_args()


def load_bootstrap_module():
    """Carica scripts/compute_bootstrap_ci.py come modulo (la dir non e' un package)."""
    if not BOOTSTRAP_SRC.exists():
        raise FileNotFoundError(f"bootstrap del repo non trovato: {BOOTSTRAP_SRC}")
    spec = importlib.util.spec_from_file_location("wbes_compute_bootstrap_ci", BOOTSTRAP_SRC)
    module = importlib.util.module_from_spec(spec)
    # Va registrato PRIMA di eseguirlo: @dataclass risale a sys.modules[cls.__module__],
    # e senza questa riga BootstrapTask esplode con AttributeError su NoneType.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_pearson(gt: np.ndarray, values: np.ndarray) -> float:
    """Gemello di ``finite_spearman`` del repo, con Pearson al posto di Spearman."""
    from scipy.stats import pearsonr

    mask = np.isfinite(gt) & np.isfinite(values)
    if int(mask.sum()) < 3:
        return math.nan
    x = gt[mask]
    y = values[mask]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan
    return float(pearsonr(x, y).statistic)


@contextmanager
def _corr_backend(module, corr_fn):
    """``weighted_bootstrap_spearman`` risolve ``finite_spearman`` come globale del modulo.

    Scambiandola si ottiene un'altra correlazione con esattamente lo stesso codice di
    ricampionamento, invece di duplicare la funzione.
    """
    original = module.finite_spearman
    module.finite_spearman = corr_fn
    try:
        yield
    finally:
        module.finite_spearman = original


def stable_seed(base: int, *tokens: str) -> int:
    """Seme riproducibile per (metrica, setting, correlazione), indipendente dall'ordine."""
    return int(base + zlib.crc32("|".join(tokens).encode()) % 1_000_000)


def build_pair_frame(metric: str, setting: str, args) -> pd.DataFrame | None:
    """Righe (subject_a, subject_b, gt_distance, <metric>) per un setting della Tabella 2."""
    out_root, strict = args.out_root, args.strict
    subjects = common.subject_set(args.subject_set)
    G = common.load_gt_submatrix(subjects)
    pair_i, pair_j = common.subject_pair_indices(len(subjects))
    names = np.asarray(subjects)

    frames = []
    missing = []
    for topology_a, topology_b in common.setting_topology_pairs(setting):
        path = common.matrix_path(metric, topology_a, topology_b, out_root)
        if not path.exists():
            missing.append(f"{topology_a}->{topology_b}")
            continue
        D, matrix_subjects, _, _, _ = common.load_matrix(path)
        if matrix_subjects != subjects:
            raise ValueError(f"{path}: soggetti diversi dal set richiesto ({args.subject_set})")
        frames.append(pd.DataFrame({
            "subject_a": names[pair_i],
            "subject_b": names[pair_j],
            "topology_a": topology_a,
            "topology_b": topology_b,
            "gt_distance": G[pair_i, pair_j],
            metric: D[pair_i, pair_j],
        }))

    if missing:
        message = f"[rank] {metric}/{setting}: mancano {len(missing)} coppie di topologie: {missing}"
        if strict:
            raise SystemExit(message)
        print(f"WARNING: {message}", file=sys.stderr)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def compute_rows(metric: str, setting: str, args, bootstrap_module) -> list[dict]:
    df = build_pair_frame(metric, setting, args)
    if df is None:
        return []

    n_finite = int(np.isfinite(df[metric].to_numpy(dtype=np.float64)).sum())
    rows = []
    for corr_name, corr_fn in (("spearman", bootstrap_module.finite_spearman),
                               ("pearson", finite_pearson)):
        rng = np.random.default_rng(stable_seed(args.seed, metric, setting, corr_name, args.subject_set))
        with _corr_backend(bootstrap_module, corr_fn):
            point, ci_low, ci_high, n_subjects, n_pairs = bootstrap_module.weighted_bootstrap_spearman(
                df, value_col=metric, n_bootstrap=args.n_bootstrap, rng=rng,
            )
        rows.append({
            "metric": metric,
            "setting": setting,
            "subject_set": args.subject_set,
            "correlation": corr_name,
            "value": point,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_subjects": n_subjects,
            "n_pairs": n_pairs,
            "n_finite": n_finite,
            "n_topology_pairs": int(df.groupby(["topology_a", "topology_b"]).ngroups),
            "n_bootstrap": args.n_bootstrap,
        })
    return rows


def discover_metrics(out_root: Path) -> list[str]:
    root = out_root / "matrices"
    if not root.is_dir():
        raise FileNotFoundError(f"nessuna matrice sotto {root}: lancia prima gli script *_matrix.py")
    return sorted(d.name for d in root.iterdir() if d.is_dir() and any(d.glob("*.npz")))


def check_paper_reference(results: pd.DataFrame) -> list[str]:
    """Gate di validazione: Chamfer raw deve ricadere nel CI pubblicato."""
    lines = []
    for (metric, setting), (ref, ref_low, ref_high) in PAPER_REFERENCE.items():
        row = results[
            results["metric"].eq(metric)
            & results["setting"].eq(setting)
            & results["correlation"].eq("spearman")
        ]
        if row.empty:
            lines.append(f"  [assente] {metric}/{setting}: paper {ref:.3f} [{ref_low:.3f}, {ref_high:.3f}]")
            continue
        value = float(row.iloc[0]["value"])
        inside = ref_low <= value <= ref_high
        lines.append(
            f"  [{'OK' if inside else 'FUORI CI'}] {metric}/{setting}: qui {value:.4f} "
            f"[{float(row.iloc[0]['ci_low']):.3f}, {float(row.iloc[0]['ci_high']):.3f}] "
            f"vs paper {ref:.3f} [{ref_low:.3f}, {ref_high:.3f}] (delta {value - ref:+.4f})"
        )
    return lines


def fmt_interval(row) -> str:
    values = [row["value"], row["ci_low"], row["ci_high"]]
    if any(not math.isfinite(float(v)) for v in values):
        return "--"
    return f"{float(values[0]):.3f} [{float(values[1]):.3f}, {float(values[2]):.3f}]"


def coverage(results: pd.DataFrame, metric: str, setting: str) -> int:
    """Coppie ordinate di topologie su cui la cella (metrica, setting) e' stata calcolata."""
    row = results[
        results["metric"].eq(metric)
        & results["setting"].eq(setting)
        & results["correlation"].eq("spearman")
    ]
    return int(row.iloc[0]["n_topology_pairs"]) if len(row) else 0


def is_partial(results: pd.DataFrame, metric: str, settings: list[str]) -> bool:
    """True se in almeno un setting la metrica copre meno coppie di topologie del dovuto."""
    return any(
        0 < coverage(results, metric, setting) < len(common.setting_topology_pairs(setting))
        for setting in settings
    )


def make_latex_table(results: pd.DataFrame, metrics: list[str], subject_set: str = "") -> str:
    primary = subject_set == "heldout"
    settings = list(common.SETTINGS)
    full = [m for m in metrics if not is_partial(results, m, settings)]
    partial = [m for m in metrics if is_partial(results, m, settings)]

    lines = [
        r"% tabella " + ("PRIMARIA: split held-out, nessun soggetto di training."
                         if primary else
                         f"su {subject_set}: ~79 soggetti su 100 sono di TRAINING, "
                         "e' un gate di riproduzione, non una tabella di risultati."),
        r"% Le ultime due colonne spezzano la seconda: 12 + 8 = 20 coppie ordinate di",
        r"% topologie. crop non entra in nessuna colonna (nessuna matrice con crop esiste).",
        r"% coppie ordinate di topologie attese per colonna:",
    ]
    lines += [f"%   {setting}: {len(common.setting_topology_pairs(setting))}"
              for setting in settings]
    for metric in partial:
        counts = ", ".join(f"{setting}={coverage(results, metric, setting)}"
                           for setting in settings)
        lines.append(f"% {metric}: {PARTIAL_COVERAGE_NOTE} ({counts})")

    lines += [
        r"\begin{tabular}{l" + "c" * len(settings) + "}",
        r"\toprule",
        "Method & " + " & ".join(SETTING_LABELS[s] for s in settings) + r" \\",
        r"\midrule",
    ]

    def body(metric: str, suffix: str = "") -> str:
        cells = []
        for setting in settings:
            row = results[
                results["metric"].eq(metric)
                & results["setting"].eq(setting)
                & results["correlation"].eq("spearman")
            ]
            cells.append(fmt_interval(row.iloc[0]) if len(row) else "--")
        return f"{metric}{suffix} & " + " & ".join(cells) + r" \\"

    lines.extend(body(metric) for metric in full)
    if partial:
        # Sotto la riga, e con il pugnale: la loro cella non e' una media sulle stesse
        # coppie di topologie delle righe sopra, quindi metterla in colonna con loro
        # inviterebbe a un confronto che non regge.
        lines.append(r"\midrule")
        lines.extend(body(metric, r"$^{\dagger}$") for metric in partial)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if partial:
        lines.append(r"% $^{\dagger}$ " + PARTIAL_COVERAGE_NOTE)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    bootstrap_module = load_bootstrap_module()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] or discover_metrics(args.out_root)
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    print(f"[rank] metriche={metrics} setting={settings} soggetti={args.subject_set} "
          f"bootstrap={args.n_bootstrap} seed={args.seed}", flush=True)

    rows = []
    for metric in metrics:
        for setting in settings:
            new_rows = compute_rows(metric, setting, args, bootstrap_module)
            for row in new_rows:
                if row["correlation"] == "spearman":
                    print(f"[rank] {metric:<12} {setting:<24} spearman={row['value']:.4f} "
                          f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
                          f"n_pairs={row['n_pairs']}", flush=True)
            rows.extend(new_rows)

    if not rows:
        raise SystemExit("[rank] nessun risultato: mancano le matrici")

    results = pd.DataFrame(rows)
    out_dir = args.out_root / "ranking"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"table2_extended_{args.subject_set}"
    csv_path = out_dir / f"{stem}.csv"
    tex_path = out_dir / f"{stem}.tex"
    json_path = out_dir / f"{stem}.json"
    results.to_csv(csv_path, index=False)
    tex_path.write_text(make_latex_table(results, metrics, args.subject_set) + "\n",
                        encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if args.subject_set == "heldout":
        print("\n[rank] questa e' la TABELLA PRIMARIA: split held-out, nessun soggetto di "
              "training.")
    else:
        print(f"\n[rank] ATTENZIONE: {args.subject_set} non e' una tabella di risultati. "
              "Condivide 21 soggetti su 100 con lo split held-out, quindi ~79 dei suoi "
              "soggetti sono soggetti di TRAINING: serve solo come gate di riproduzione "
              "dello 0.729 pubblicato. La tabella primaria e' --subject-set heldout.")

    print("\n[rank] validazione contro i numeri pubblicati:")
    if args.subject_set != "facebench_first100":
        print("  (i numeri della Tabella 2 sono su facebench_first100, non su "
              f"{args.subject_set}: il confronto qui sotto e' solo indicativo)")
    for line in check_paper_reference(results):
        print(line)
    print(f"\n[rank] CSV   {csv_path}")
    print(f"[rank] LaTeX {tex_path}")


if __name__ == "__main__":
    main()
