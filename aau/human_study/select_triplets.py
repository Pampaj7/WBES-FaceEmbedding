#!/usr/bin/env python3
"""Triplette "quale tra B e C somiglia di piu' ad A" per lo studio umano (WS4).

Si parte dal pool COMPLETO delle triplette sui 100 soggetti held-out in topologia
``original`` (100 x C(99,2) = 485100, A e' il riferimento, {B, C} non ordinata) e si
tengono solo quelle in cui le metriche litigano: almeno due fra {GT, chamfer, lpips,
latent} devono ordinare d(A,B) e d(A,C) al contrario, ciascuna con un margine relativo
>= ``--margin``.  Se le metriche sono d'accordo la tripletta non porta informazione: la
risposta umana e' gia' prevista da tutte e non separa nessuna ipotesi.

Il margine relativo e' ``|d(A,B) - d(A,C)| / media(d(A,B), d(A,C))``: simmetrico nei due
argomenti e senza unita', cosi' la stessa soglia vale per metriche con scale diverse
(chamfer sta su 0.02-0.07, LPIPS su 0.05-0.20, GT su 0-0.69).

Il "tipo di disaccordo" e' la partizione delle metriche decisive nei due schieramenti,
canonicalizzata rispetto allo scambio B<->C (che e' arbitrario): ``gt_vs_lpips``,
``chamfer+gt_vs_lpips``, ...  Le 300 triplette dello studio sono bilanciate su questi
tipi (riempimento dal tipo piu' raro al piu' comune), a cui si aggiungono 30 triplette di
controllo dove TUTTE le metriche sono d'accordo con un margine grande: sono gli attention
check, chi le sbaglia non stava guardando lo schermo.

Serve un nodo di calcolo (il frontend non ha numpy):

  srun -p cpu --mem=16G --time=00:20:00 \
      env AAU_NV= aau/run.sh aau/human_study/select_triplets.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "baselines"))

import common  # noqa: E402

# La convenzione dei nomi dei render non viene riscritta: e' quella con cui la cache e'
# stata prodotta, e se cambia deve cambiare in un posto solo.
from render_cache import render_name  # noqa: E402

FRONTAL_YAW = 0.0
MIN_RENDER_PX = 512
# Ordine fisso delle metriche: entra nella codifica intera dei tipi di disaccordo.
METRIC_ORDER = ("gt", "chamfer", "lpips", "latent")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=THIS_DIR)
    p.add_argument("--matrix-root", type=Path, default=common.AAU_DIR / "runs" / "baselines",
                   help="run delle baseline da cui leggere matrices/ e renders/")
    p.add_argument("--renders", type=Path, default=None, help="default: <matrix-root>/renders")
    p.add_argument("--subject-set", type=str, default="heldout", choices=common.SUBJECT_SETS)
    p.add_argument("--topology", type=str, default="original", choices=common.TOPOLOGIES)
    p.add_argument("--latent-matrix", type=Path, default=None,
                   help="npz nel formato di common.save_matrix con le distanze latent v1; "
                        "se assente lo studio usa solo gt+chamfer+lpips")
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--n-control", type=int, default=30)
    p.add_argument("--margin", type=float, default=0.15,
                   help="margine relativo minimo perche' una metrica conti come decisiva")
    p.add_argument("--control-margin", type=float, default=0.50,
                   help="margine relativo minimo, su OGNI metrica, per un attention check")
    p.add_argument("--control-pool-factor", type=int, default=10,
                   help="i controlli si estraggono dai primi n_control*factor per margine minimo")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-copy-images", action="store_true",
                   help="non copiare i render frontali in <out-dir>/img")
    return p.parse_args()


# ------------------------------------------------------------------ matrici di distanza

def symmetrize(D: np.ndarray) -> np.ndarray:
    """Matrice piena a partire da una riempita solo per i<j (la convenzione di common)."""
    lower = np.tril(np.ones_like(D, dtype=bool), -1)
    if np.isfinite(D[lower]).all():
        if not np.allclose(D[lower], D.T[lower], equal_nan=True):
            raise ValueError("matrice piena ma non simmetrica: distanze incoerenti")
        S = np.array(D, dtype=np.float64)
    else:
        upper = np.triu(np.ones_like(D, dtype=bool), 1)
        S = np.zeros_like(D, dtype=np.float64)
        S[upper] = D[upper]
        S = S + S.T
    np.fill_diagonal(S, 0.0)
    return S


def load_distances(args, subjects: list[str]) -> dict[str, np.ndarray]:
    """Una matrice piena 100x100 per metrica; ``latent`` solo se il job di eval ha finito."""
    D = {"gt": symmetrize(common.load_gt_submatrix(subjects))}
    for metric in ("chamfer", "lpips"):
        path = common.matrix_path(metric, args.topology, args.topology, args.matrix_root)
        if not path.exists():
            raise FileNotFoundError(f"matrice {metric} mancante: {path}")
        M, matrix_subjects, _, _, _ = common.load_matrix(path)
        if matrix_subjects != subjects:
            raise ValueError(f"{path}: soggetti diversi dal set richiesto ({args.subject_set})")
        D[metric] = symmetrize(M)

    if args.latent_matrix is not None:
        M, matrix_subjects, _, _, _ = common.load_matrix(args.latent_matrix)
        if matrix_subjects != subjects:
            raise ValueError(f"{args.latent_matrix}: soggetti diversi da {args.subject_set}")
        D["latent"] = symmetrize(M)
    else:
        print("[triplets] nessuna matrice latent: lo studio confronta solo gt/chamfer/lpips",
              file=sys.stderr)
    return {m: D[m] for m in METRIC_ORDER if m in D}


# ------------------------------------------------------------------ pool e disaccordi

def triplet_pool(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tutte le (A, B, C) con A fuori da {B, C} e B < C, in ordine deterministico."""
    ii, jj = np.triu_indices(n - 1, k=1)
    a_list, b_list, c_list = [], [], []
    for a in range(n):
        others = np.delete(np.arange(n, dtype=np.int32), a)
        a_list.append(np.full(ii.size, a, dtype=np.int32))
        b_list.append(others[ii])
        c_list.append(others[jj])
    return np.concatenate(a_list), np.concatenate(b_list), np.concatenate(c_list)


def metric_view(S: np.ndarray, a, b, c) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """d(A,B), d(A,C) e margine relativo per ogni tripletta del pool."""
    d_ab = S[a, b]
    d_ac = S[a, c]
    mean = 0.5 * (d_ab + d_ac)
    with np.errstate(invalid="ignore", divide="ignore"):
        margin = np.where(mean > 0, np.abs(d_ab - d_ac) / mean, 0.0)
    return d_ab, d_ac, margin


def type_label(code: int, metrics: list[str]) -> str:
    """Da codice ternario (0 non decisiva, 1 vota B, 2 vota C) all'etichetta del tipo.

    Le due squadre sono ordinate fra loro perche' quale delle due sta su B e quale su C
    dipende solo da come e' stata scritta la tripletta, non dal disaccordo.
    """
    side_b, side_c = [], []
    for k, metric in enumerate(metrics):
        vote = (code // 3 ** k) % 3
        if vote == 1:
            side_b.append(metric)
        elif vote == 2:
            side_c.append(metric)
    left, right = sorted(("+".join(side_b), "+".join(side_c)))
    return f"{left}_vs_{right}"


def balanced_take(pools: dict[str, np.ndarray], total: int, rng: np.random.Generator) -> dict:
    """Quote il piu' uguali possibile fra i tipi: i pool piccoli si esauriscono e il
    residuo si ridistribuisce sui piu' capienti."""
    taken: dict[str, np.ndarray] = {}
    remaining = total
    order = sorted(pools, key=lambda t: (len(pools[t]), t))
    for k, label in enumerate(order):
        left = len(order) - k
        quota = min(len(pools[label]), -(-remaining // left))
        picked = rng.permutation(pools[label])[:quota]
        taken[label] = np.sort(picked)
        remaining -= quota
    if remaining > 0:
        raise SystemExit(
            f"[triplets] pool insufficiente: mancano {remaining} triplette su {total}. "
            "Abbassa --margin o allarga il set di soggetti.")
    return taken


# ------------------------------------------------------------------ scrittura

def triplet_record(idx: int, kind: str, tid: str, label: str, subjects, a, b, c,
                   per_metric: dict, topology: str, margin_thr: float,
                   swap: bool) -> dict:
    """Un record di tripletta, con la risposta attesa da ogni metrica."""
    s_a, s_b, s_c = subjects[a[idx]], subjects[b[idx]], subjects[c[idx]]
    if swap:
        s_b, s_c = s_c, s_b
    entry = {
        "id": tid,
        "kind": kind,
        "disagreement_type": label,
        "a": s_a,
        "b": s_b,
        "c": s_c,
        "images": {
            key: f"img/{render_name(subject, topology, FRONTAL_YAW)}.png"
            for key, subject in (("a", s_a), ("b", s_b), ("c", s_c))
        },
        "metrics": {},
    }
    for metric, (d_ab, d_ac, margin) in per_metric.items():
        ab, ac = float(d_ab[idx]), float(d_ac[idx])
        if swap:
            ab, ac = ac, ab
        entry["metrics"][metric] = {
            "d_ab": round(ab, 6),
            "d_ac": round(ac, 6),
            "margin": round(float(margin[idx]), 6),
            "expected": "b" if ab < ac else "c",
            "decisive": bool(margin[idx] >= margin_thr),
        }
    return entry


def copy_renders(entries: list[dict], renders: Path, out_dir: Path, topology: str) -> tuple[int, int]:
    """Copia in <out-dir>/img i soli render frontali usati; controlla che siano >= 512 px."""
    from PIL import Image

    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    used = sorted({e[key] for e in entries for key in ("a", "b", "c")})
    total = 0
    for subject in used:
        name = f"{render_name(subject, topology, FRONTAL_YAW)}.png"
        src = renders / name
        if not src.exists():
            raise FileNotFoundError(
                f"render frontale mancante: {src}\n"
                "  rigeneralo con aau/baselines/render_cache.py (v2_work/phase0/render_mesh.py, "
                "luce fissa, sfondo neutro, camera condivisa)")
        with Image.open(src) as im:
            if min(im.size) < MIN_RENDER_PX:
                raise ValueError(f"{src}: {im.size}, servono almeno {MIN_RENDER_PX} px")
        shutil.copyfile(src, img_dir / name)
        total += (img_dir / name).stat().st_size
    return len(used), total


def write_stats(path: Path, args, subjects, metrics, pool_size, type_pools, chosen_types,
                entries, control_stats, image_stats) -> None:
    test_entries = [e for e in entries if e["kind"] == "test"]
    control_entries = [e for e in entries if e["kind"] == "control"]
    usage = Counter(s for e in entries for s in (e["a"], e["b"], e["c"]))

    lines = [
        "# Triplette per lo studio umano (WS4)",
        "",
        f"Generato il {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} da "
        f"`aau/human_study/select_triplets.py`, seed {args.seed}.",
        "",
        "| parametro | valore |",
        "|---|---|",
        f"| soggetti | {len(subjects)} ({args.subject_set}, topologia `{args.topology}`) |",
        f"| metriche | {', '.join(metrics)} |",
        f"| pool completo | {pool_size} triplette (A, {{B, C}}) |",
        f"| margine relativo minimo | {args.margin:.2f} (decisiva) / "
        f"{args.control_margin:.2f} (controllo) |",
        f"| triplette in disaccordo | {sum(len(v) for v in type_pools.values())} |",
        f"| selezionate | {len(test_entries)} test + {len(control_entries)} controllo |",
        "",
        "## Tipi di disaccordo",
        "",
        "Squadre di metriche che si contraddicono, con il margine relativo richiesto su "
        "ognuna. L'etichetta e' invariante allo scambio B<->C.",
        "",
        "| tipo | disponibili | scelte |",
        "|---|---:|---:|",
    ]
    for label in sorted(type_pools, key=lambda t: (-len(type_pools[t]), t)):
        lines.append(f"| `{label}` | {len(type_pools[label])} | {len(chosen_types.get(label, []))} |")

    lines += [
        "",
        "## Margini nelle triplette scelte",
        "",
        "| metrica | mediana test | min test | decisiva in test | mediana controllo |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        m_test = np.array([e["metrics"][metric]["margin"] for e in test_entries])
        m_ctrl = np.array([e["metrics"][metric]["margin"] for e in control_entries])
        n_dec = sum(e["metrics"][metric]["decisive"] for e in test_entries)
        lines.append(
            f"| {metric} | {np.median(m_test):.3f} | {m_test.min():.3f} | "
            f"{n_dec}/{len(test_entries)} | {np.median(m_ctrl):.3f} |")

    lines += [
        "",
        "## Controlli (attention check)",
        "",
        f"- {len(control_entries)} triplette con accordo unanime di tutte le metriche e "
        f"margine relativo >= {args.control_margin:.2f} su ognuna.",
        f"- candidate: {control_stats['pool']}; estratte dalle prime "
        f"{control_stats['top']} per margine minimo.",
        f"- margine minimo (sulla metrica peggiore) nelle scelte: mediana "
        f"{control_stats['median_min_margin']:.3f}, minimo {control_stats['min_min_margin']:.3f}.",
        "",
        "## Copertura dei soggetti",
        "",
        f"- {len(usage)} soggetti distinti sui {len(subjects)} disponibili.",
        f"- comparse per soggetto: min {min(usage.values())}, mediana "
        f"{int(np.median(list(usage.values())))}, max {max(usage.values())}.",
        "",
        "## Pacchetto",
        "",
        f"- {image_stats['n_images']} render frontali PNG {MIN_RENDER_PX}x{MIN_RENDER_PX} in "
        f"`img/` ({image_stats['bytes'] / 1e6:.1f} MB).",
        "- `triplets.json` (dati + risposte attese) e `triplets.js` (stesso contenuto, "
        "caricato da `index.html` con un tag `<script>` perche' `fetch()` su `file://` e' "
        "bloccato dal browser).",
        "",
        "## Riproduzione",
        "",
        "```bash",
        "srun -p cpu --mem=16G --time=00:20:00 \\",
        "    env AAU_NV= aau/run.sh aau/human_study/select_triplets.py \\",
        f"    --seed {args.seed} --n-test {args.n_test} --n-control {args.n_control} \\",
        f"    --margin {args.margin} --control-margin {args.control_margin}",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    renders = args.renders or (args.matrix_root / "renders")
    rng = np.random.default_rng(args.seed)

    subjects = common.subject_set(args.subject_set)
    D = load_distances(args, subjects)
    metrics = list(D)
    print(f"[triplets] soggetti={len(subjects)} metriche={metrics} margine={args.margin} "
          f"seed={args.seed}", flush=True)

    a, b, c = triplet_pool(len(subjects))
    per_metric = {m: metric_view(D[m], a, b, c) for m in metrics}
    print(f"[triplets] pool completo: {a.size} triplette", flush=True)

    # Voto di ogni metrica: 1 = "B piu' vicino ad A", 2 = "C piu' vicino", 0 = non decisiva.
    votes = np.zeros((len(metrics), a.size), dtype=np.int8)
    code = np.zeros(a.size, dtype=np.int64)
    for k, metric in enumerate(metrics):
        d_ab, d_ac, margin = per_metric[metric]
        decisive = np.isfinite(d_ab) & np.isfinite(d_ac) & (margin >= args.margin)
        votes[k] = np.where(decisive, np.where(d_ab < d_ac, 1, 2), 0)
        code += votes[k].astype(np.int64) * 3 ** k

    disagree = np.any(votes == 1, axis=0) & np.any(votes == 2, axis=0)
    idx_disagree = np.flatnonzero(disagree)
    type_pools: dict[str, np.ndarray] = {}
    for value in np.unique(code[idx_disagree]):
        label = type_label(int(value), metrics)
        pool = idx_disagree[code[idx_disagree] == value]
        type_pools.setdefault(label, []).append(pool)
    type_pools = {k: np.concatenate(v) for k, v in type_pools.items()}
    for label, pool in sorted(type_pools.items()):
        print(f"[triplets] {label:<28} {pool.size:>7}", flush=True)

    chosen_types = balanced_take(type_pools, args.n_test, rng)

    # Controlli: tutte le metriche finite, d'accordo, e con un margine grande su OGNUNA.
    finite = np.ones(a.size, dtype=bool)
    margins = np.empty((len(metrics), a.size), dtype=np.float64)
    closer_b = np.empty((len(metrics), a.size), dtype=bool)
    for k, metric in enumerate(metrics):
        d_ab, d_ac, margin = per_metric[metric]
        finite &= np.isfinite(d_ab) & np.isfinite(d_ac)
        margins[k] = margin
        closer_b[k] = d_ab < d_ac
    unanimous = finite & (closer_b.all(axis=0) | (~closer_b).all(axis=0))
    min_margin = margins.min(axis=0)
    control_pool = np.flatnonzero(unanimous & (min_margin >= args.control_margin))
    if control_pool.size < args.n_control:
        raise SystemExit(f"[triplets] solo {control_pool.size} controlli con margine "
                         f">= {args.control_margin}: abbassa --control-margin")
    top = control_pool[np.argsort(-min_margin[control_pool])][: args.n_control * args.control_pool_factor]
    chosen_control = np.sort(rng.permutation(top)[: args.n_control])
    print(f"[triplets] controlli: pool {control_pool.size}, scelti {chosen_control.size}", flush=True)

    # B e C si scambiano a caso: nel JSON non deve restare traccia di quale squadra di
    # metriche ha "vinto" la posizione b (la pagina randomizza comunque destra/sinistra).
    entries: list[dict] = []
    n = 0
    for label in sorted(chosen_types):
        for idx in chosen_types[label]:
            n += 1
            entries.append(triplet_record(int(idx), "test", f"t{n:04d}", label, subjects,
                                          a, b, c, per_metric, args.topology, args.margin,
                                          bool(rng.random() < 0.5)))
    for n, idx in enumerate(chosen_control, start=1):
        entries.append(triplet_record(int(idx), "control", f"c{n:04d}", "unanime", subjects,
                                      a, b, c, per_metric, args.topology, args.margin,
                                      bool(rng.random() < 0.5)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_stats = {"n_images": 0, "bytes": 0}
    if not args.no_copy_images:
        n_images, total_bytes = copy_renders(entries, renders, args.out_dir, args.topology)
        image_stats = {"n_images": n_images, "bytes": total_bytes}
        print(f"[triplets] copiati {n_images} render ({total_bytes / 1e6:.1f} MB) in "
              f"{args.out_dir / 'img'}", flush=True)

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": args.seed,
            "subject_set": args.subject_set,
            "topology": args.topology,
            "n_subjects": len(subjects),
            "metrics": metrics,
            "margin": args.margin,
            "control_margin": args.control_margin,
            "n_test": sum(e["kind"] == "test" for e in entries),
            "n_control": sum(e["kind"] == "control" for e in entries),
            "pool_size": int(a.size),
            "n_disagree": int(idx_disagree.size),
            "matrix_root": str(args.matrix_root),
            "renders": str(renders),
        },
        "triplets": entries,
    }
    json_path = args.out_dir / "triplets.json"
    json_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    (args.out_dir / "triplets.js").write_text(
        "// Generato da select_triplets.py: index.html lo carica con un tag <script>,\n"
        "// perche' fetch('triplets.json') su file:// viene bloccato dal browser.\n"
        "window.WBES_TRIPLETS = " + json.dumps(payload) + ";\n", encoding="utf-8")

    control_min = min_margin[chosen_control]
    write_stats(args.out_dir / "triplets_stats.md", args, subjects, metrics, int(a.size),
                type_pools, chosen_types, entries,
                {"pool": int(control_pool.size), "top": int(top.size),
                 "median_min_margin": float(np.median(control_min)),
                 "min_min_margin": float(control_min.min())},
                image_stats)

    print(f"[triplets] JSON  {json_path}")
    print(f"[triplets] stats {args.out_dir / 'triplets_stats.md'}")


if __name__ == "__main__":
    main()
