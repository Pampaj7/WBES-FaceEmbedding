"""Protocollo di coppie di WS3a su Multiface: quattro classi, campionate in modo bilanciato.

    python3 aau/multiface/make_pairs_protocol.py            # gira anche sul frontend
    python3 aau/multiface/make_pairs_protocol.py --max-pairs 2000 --seed 1234

Legge `datasets/Multiface/prep/manifest.csv` e scrive `aau/multiface/pairs_protocol.json`.
Solo stdlib: niente numpy, quindi non serve un job Slurm.

Le quattro classi
-----------------
  a  stesso soggetto, stessa espressione, frame diversi      -> same
  b  stesso soggetto, espressione diversa                    -> same
  c  soggetti diversi, stessa espressione                    -> different
  d  soggetti diversi, espressione diversa                   -> different

"Espressione" e' il nome del segmento. I 10 soggetti Mugsy v1 hanno segmenti `E0xx_*` e i
3 v2 hanno `EXP_*`: le due liste sono disgiunte, quindi la classe (c) esiste solo dentro
v1 o dentro v2, mai a cavallo. Per lo stesso motivo ogni coppia v1-v2 finisce in (d).

I due test che questo protocollo serve
--------------------------------------
  `same_vs_different`  (a)+(b) contro (c)+(d): AUC della metrica come discriminante di
                       identita' su scansioni reali, senza nessuna D_GT.
  `hard_b_vs_c`        (b) contro (c): il caso difficile. In (b) cambia l'espressione ma
                       l'identita' no; in (c) l'espressione e' la stessa e cambia
                       l'identita'. Una metrica che misura la forma della faccia invece
                       dell'identita' finisce sotto 0.5 proprio qui.

Campionamento
-------------
Tetto di `--max-pairs` coppie per classe, e dentro la classe round-robin sui gruppi
naturali (soggetto x segmento per (a), soggetto x coppia di segmenti per (b), coppia di
soggetti x segmento per (c), coppia di soggetti per (d)): senza questo la classe (c) si
riempirebbe quasi solo di coppie v1, che sono 45 coppie di soggetti contro 3, e la (d)
sarebbe dominata dai soggetti con piu' frame.

Le classi (a), (b) e (c) vengono enumerate per intero (27k / 286k / 211k coppie, il conto
esatto e' nel json). La (d) no: sono ~3.4 milioni di coppie e materializzarle costerebbe
mezzo giga per campionarne 2000. Li' si estrae per costruzione, con rifiuto delle coppie a
segmento uguale e delle ripetizioni; la numerosita' della popolazione e' calcolata a mano.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CLASSES = {
    "a_same_subject_same_expression": "same",
    "b_same_subject_diff_expression": "same",
    "c_diff_subject_same_expression": "different",
    "d_diff_subject_diff_expression": "different",
}

TESTS = {
    "same_vs_different": {
        "positive": ["a_same_subject_same_expression", "b_same_subject_diff_expression"],
        "negative": ["c_diff_subject_same_expression", "d_diff_subject_diff_expression"],
    },
    "hard_b_vs_c": {
        "positive": ["b_same_subject_diff_expression"],
        "negative": ["c_diff_subject_same_expression"],
    },
}

# Numero di tentativi per coppia estratta nella classe (d) prima di rinunciare al giro:
# le collisioni sono rarissime (2000 estrazioni su 3.4e6 coppie), 50 e' solo un paletto.
MAX_DRAWS = 50


def read_manifest(path: Path, topology: str = "tracked") -> tuple[dict, dict, dict]:
    """(mesh per (soggetto, segmento), versione per soggetto, segmenti per soggetto)."""
    by_cell: dict[tuple[str, str], list[str]] = {}
    version: dict[str, str] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["topology"] != topology:
                continue
            key = (row["subject"], row["segment"])
            by_cell.setdefault(key, []).append(row["name"])
            version[row["subject"]] = row["version"]
    if not by_cell:
        raise RuntimeError(f"nessuna riga con topology={topology!r} in {path}")
    for names in by_cell.values():
        names.sort()
    segments: dict[str, list[str]] = {}
    for subject, segment in by_cell:
        segments.setdefault(subject, []).append(segment)
    for seg in segments.values():
        seg.sort()
    return by_cell, version, segments


def repo_relative(path: Path) -> str:
    """Path relativo alla root del repo quando ci sta dentro, assoluto altrimenti."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def ordered(a: str, b: str) -> tuple[str, str]:
    """Le metriche sono simmetriche: una coppia e' identificata dai due nomi ordinati."""
    return (a, b) if a <= b else (b, a)


def balanced_sample(groups: list[list], quota: int, rng: random.Random) -> list:
    """Round-robin sui gruppi non vuoti finche' la quota e' piena o i gruppi si esauriscono.

    Ogni giro pesca al piu' un elemento per gruppo, quindi i gruppi piccoli si svuotano e
    quelli grandi continuano: e' la ripartizione piu' uniforme possibile a quota fissa.
    L'ordine dei gruppi viene rimescolato a ogni giro, altrimenti l'ultimo giro (quello
    troncato dalla quota) favorirebbe sempre i primi gruppi.
    """
    pools = [list(g) for g in groups if g]
    for pool in pools:
        rng.shuffle(pool)
    out: list = []
    while pools and len(out) < quota:
        rng.shuffle(pools)
        survivors = []
        for pool in pools:
            if len(out) < quota:
                out.append(pool.pop())
            if pool:
                survivors.append(pool)
        pools = survivors
    return out


# ------------------------------------------------------------------ le quattro classi

def groups_a(by_cell: dict) -> list[list[tuple[str, str]]]:
    """Stesso soggetto, stesso segmento, frame diversi. Gruppo = (soggetto, segmento)."""
    return [[ordered(x, y) for x, y in itertools.combinations(names, 2)]
            for _, names in sorted(by_cell.items())]


def groups_b(by_cell: dict, segments: dict) -> list[list[tuple[str, str]]]:
    """Stesso soggetto, segmenti diversi. Gruppo = (soggetto, coppia di segmenti)."""
    groups = []
    for subject in sorted(segments):
        for seg_a, seg_b in itertools.combinations(segments[subject], 2):
            groups.append([ordered(x, y)
                           for x in by_cell[(subject, seg_a)]
                           for y in by_cell[(subject, seg_b)]])
    return groups


def groups_c(by_cell: dict, version: dict, segments: dict) -> list[list[tuple[str, str]]]:
    """Soggetti diversi, stesso nome di segmento. Gruppo = (coppia di soggetti, segmento).

    Il nome del segmento coincide solo dentro v1 o dentro v2, quindi le coppie a cavallo
    delle due versioni qui non ci sono per costruzione, non per una scelta.
    """
    groups = []
    for subject_a, subject_b in itertools.combinations(sorted(segments), 2):
        if version[subject_a] != version[subject_b]:
            continue
        for segment in sorted(set(segments[subject_a]) & set(segments[subject_b])):
            groups.append([ordered(x, y)
                           for x in by_cell[(subject_a, segment)]
                           for y in by_cell[(subject_b, segment)]])
    return groups


def sample_d(by_cell: dict, segments: dict, quota: int, rng: random.Random) -> list[tuple[str, str]]:
    """Soggetti diversi, segmenti diversi, per costruzione. Gruppo = coppia di soggetti."""
    by_subject = {s: [(seg, name) for seg in segments[s] for name in by_cell[(s, seg)]]
                  for s in sorted(segments)}
    subject_pairs = list(itertools.combinations(sorted(segments), 2))
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    while len(out) < quota:
        rng.shuffle(subject_pairs)
        added = 0
        for subject_a, subject_b in subject_pairs:
            if len(out) >= quota:
                break
            for _ in range(MAX_DRAWS):
                seg_a, name_a = rng.choice(by_subject[subject_a])
                seg_b, name_b = rng.choice(by_subject[subject_b])
                if seg_a == seg_b:
                    continue
                pair = ordered(name_a, name_b)
                if pair in seen:
                    continue
                seen.add(pair)
                out.append(pair)
                added += 1
                break
        if added == 0:  # nessun giro produttivo: la popolazione e' esaurita
            break
    return out


def population_d(by_cell: dict, version: dict, segments: dict) -> int:
    """|d| = (coppie di mesh di soggetti diversi) - (quelle a segmento uguale), senza enumerare."""
    per_subject = {s: sum(len(by_cell[(s, seg)]) for seg in segments[s]) for s in segments}
    total = same_segment = 0
    for subject_a, subject_b in itertools.combinations(sorted(segments), 2):
        total += per_subject[subject_a] * per_subject[subject_b]
        if version[subject_a] != version[subject_b]:
            continue
        for segment in set(segments[subject_a]) & set(segments[subject_b]):
            same_segment += len(by_cell[(subject_a, segment)]) * len(by_cell[(subject_b, segment)])
    return total - same_segment


def version_breakdown(pairs: list[tuple[str, str]], version: dict) -> dict[str, int]:
    """Quante coppie campionate stanno in v1, in v2, o a cavallo."""
    counts = {"v1": 0, "v2": 0, "v1_v2": 0}
    for name_a, name_b in pairs:
        va = version[name_a.split("__")[0]]
        vb = version[name_b.split("__")[0]]
        counts["v1_v2" if va != vb else va] += 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", type=Path,
                   default=REPO_ROOT / "datasets/Multiface/prep/manifest.csv")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "pairs_protocol.json")
    p.add_argument("--topology", default="tracked",
                   help="quale topologia leggere dal manifest per elencare le mesh (i nomi "
                        "sono gli stessi in tutte e tre, cambia solo la cartella)")
    p.add_argument("--max-pairs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()

    by_cell, version, segments = read_manifest(a.manifest, a.topology)
    n_meshes = sum(len(v) for v in by_cell.values())
    print(f"{n_meshes} mesh, {len(segments)} soggetti, {len(by_cell)} celle soggetto x segmento")

    rng = random.Random(a.seed)
    enumerated = {
        "a_same_subject_same_expression": groups_a(by_cell),
        "b_same_subject_diff_expression": groups_b(by_cell, segments),
        "c_diff_subject_same_expression": groups_c(by_cell, version, segments),
    }

    classes = {}
    for name, groups in enumerated.items():
        pairs = sorted(set(balanced_sample(groups, a.max_pairs, rng)))
        classes[name] = {
            "label": CLASSES[name],
            "n_groups": len([g for g in groups if g]),
            "n_population": sum(len(g) for g in groups),
            "n_sampled": len(pairs),
            "by_version": version_breakdown(pairs, version),
            "pairs": [list(q) for q in pairs],
        }

    name = "d_diff_subject_diff_expression"
    pairs_d = sorted(set(sample_d(by_cell, segments, a.max_pairs, rng)))
    classes[name] = {
        "label": CLASSES[name],
        "n_groups": len(list(itertools.combinations(sorted(segments), 2))),
        "n_population": population_d(by_cell, version, segments),
        "n_sampled": len(pairs_d),
        "by_version": version_breakdown(pairs_d, version),
        "pairs": [list(q) for q in pairs_d],
    }

    protocol = {
        "protocol": "ws3a_multiface_pairs",
        "seed": a.seed,
        "max_pairs_per_class": a.max_pairs,
        "manifest": repo_relative(a.manifest),
        "topologies": ["tracked", "remesh", "down", "crop", "noisy", "up"],
        "npz_pattern": "datasets/Multiface/prep/{topology}/{name}.npz",
        "operators_pattern": "datasets/Multiface/prep/{topology}_withops[_areanorm]/{name}.npz",
        "n_meshes": n_meshes,
        "n_subjects": len(segments),
        "subjects": {s: {"version": version[s], "n_segments": len(segments[s]),
                         "n_frames": sum(len(by_cell[(s, seg)]) for seg in segments[s])}
                     for s in sorted(segments)},
        "classes": classes,
        "tests": TESTS,
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(protocol, fh, indent=1)

    print(f"\nclasse                                 gruppi   popolazione  campionate  v1/v2/misto")
    for name, info in classes.items():
        v = info["by_version"]
        print(f"  {name:<36} {info['n_groups']:>6} {info['n_population']:>13} "
              f"{info['n_sampled']:>11}  {v['v1']}/{v['v2']}/{v['v1_v2']}")
    for test, spec in TESTS.items():
        pos = sum(classes[c]["n_sampled"] for c in spec["positive"])
        neg = sum(classes[c]["n_sampled"] for c in spec["negative"])
        print(f"  test {test}: {pos} positive vs {neg} negative")
    print(f"\nscritto {a.out}")


if __name__ == "__main__":
    main()
