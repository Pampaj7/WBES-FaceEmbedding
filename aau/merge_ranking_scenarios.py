#!/usr/bin/env python3
"""Fonde i ranking_summary di piu' out dir di eval_ranking.sbatch in una sola.

Serve perche' compare_model_vs_chamfer_rankings.py tiene le righe in memoria e scrive
ranking_summary.json solo dopo l'ultimo scenario: con 2.3h per scenario i cinque scenari
non stanno in un walltime ragionevole, quindi si sottomette un job per gruppo di scenari
(WBES_EVAL_SCENARIOS, vedi aau/eval_ranking.sbatch) e si rimette insieme la tabella qui.

  aau/merge_ranking_scenarios.py --out_dir RUN/ranking_merged \\
      RUN/ranking_clean_jitter_translation RUN/ranking_rotation_mixed

Solo stdlib, cosi' gira sul frontend senza container: non ricalcola niente, riordina le
righe gia' scritte dai job. json, csv e md hanno lo stesso formato dei file di partenza.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Ordine canonico della tabella del paper: le out dir arrivano nell'ordine in cui i job
# sono finiti, che non e' quello in cui si legge il risultato.
SCENARIO_ORDER = ("clean", "jitter", "translation", "rotation", "mixed")

# Chiavi del payload che devono coincidere in tutti gli input: se cambia una di queste i
# numeri non sono confrontabili fra loro e la tabella fusa sarebbe una bugia. Stessa idea
# di eval_key.txt in aau/eval_common.sh, applicata dopo il fatto.
SHARED_KEYS = (
    "checkpoint",
    "checkpoint_selector",
    "data_dir",
    "dist_npz",
    "subject_split",
    "eval_fraction",
    "seed",
    "max_subjects_requested",
    "max_meshes_per_subject_eval",
    "pair_mode",
    "aggregation_level",
    "selected_subjects",
    "pair_context",
    "perturbation_params",
)

MD_HEADER = (
    "| Scenario | Jitter sigma | Rotation sigma | Translation sigma | Rot max deg | "
    "Trans axis std | Lat Sp | Chamfer Sp | Delta Sp | Lat Pe | Chamfer Pe | Delta Pe | "
    "Model > Chamfer |\n"
)
MD_COLUMNS = (
    "scenario",
    "jitter_sigma",
    "rotation_sigma",
    "translation_sigma",
    "rotation_angle_max_deg",
    "translation_axis_std",
    "latent_spearman",
    "chamfer_spearman",
    "delta_spearman",
    "latent_pearson",
    "chamfer_pearson",
    "delta_pearson",
)


def scenario_sort_key(name: str) -> tuple[int, str]:
    if name in SCENARIO_ORDER:
        return (SCENARIO_ORDER.index(name), "")
    # Uno scenario non previsto non viene scartato: finisce in fondo, in ordine alfabetico.
    return (len(SCENARIO_ORDER), name)


def load_inputs(in_dirs: list[Path]) -> list[tuple[Path, dict, list[dict], list[str]]]:
    loaded = []
    for d in in_dirs:
        json_path = d / "ranking_summary.json"
        csv_path = d / "ranking_summary.csv"
        if not json_path.is_file():
            # Il caso tipico: il job e' stato ucciso dal walltime e non ha scritto niente.
            raise SystemExit(f"ERRORE: {json_path} non esiste (job non arrivato in fondo?)")
        if not csv_path.is_file():
            raise SystemExit(f"ERRORE: {csv_path} non esiste")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        loaded.append((d, payload, csv_rows, fieldnames))
    return loaded


def check_shared_keys(loaded: list[tuple[Path, dict, list[dict], list[str]]]) -> None:
    ref_dir, ref_payload, _ref_rows, ref_fields = loaded[0]
    for d, payload, _rows, fields in loaded[1:]:
        if fields != ref_fields:
            raise SystemExit(
                f"ERRORE: colonne del csv diverse fra {ref_dir} e {d}: gli input vengono da "
                f"due versioni diverse di compare_model_vs_chamfer_rankings.py."
            )
        for key in SHARED_KEYS:
            if payload.get(key) != ref_payload.get(key):
                raise SystemExit(
                    f"ERRORE: '{key}' diverso fra {ref_dir} e {d}: i due job non hanno "
                    f"valutato la stessa cosa, la tabella fusa non avrebbe senso.\n"
                    f"  {ref_dir}: {ref_payload.get(key)!r}\n"
                    f"  {d}: {payload.get(key)!r}"
                )


def merge_rows(
    loaded: list[tuple[Path, dict, list[dict], list[str]]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Ritorna (rows json, spec scenari, rows csv), deduplicate per nome di scenario."""
    json_rows: dict[str, dict] = {}
    specs: dict[str, dict] = {}
    csv_rows: dict[str, dict] = {}
    origin: dict[str, Path] = {}

    for d, payload, rows, _fields in loaded:
        spec_by_name = {spec["name"]: spec for spec in payload.get("scenarios", [])}
        csv_by_name = {row["scenario"]: row for row in rows}
        for row in payload.get("rows", []):
            name = row["scenario"]
            if name in json_rows:
                # Due job che rifanno lo stesso scenario sono uno spreco, non un errore:
                # si accetta solo se i numeri coincidono, altrimenti c'e' da capire perche'.
                if json_rows[name] != row:
                    raise SystemExit(
                        f"ERRORE: scenario '{name}' presente sia in {origin[name]} sia in "
                        f"{d} con risultati diversi. Tieni solo la out dir buona."
                    )
                continue
            json_rows[name] = row
            origin[name] = d
            if name in spec_by_name:
                specs[name] = spec_by_name[name]
            if name in csv_by_name:
                csv_rows[name] = csv_by_name[name]

    missing_csv = sorted(set(json_rows) - set(csv_rows))
    if missing_csv:
        raise SystemExit(f"ERRORE: scenari senza riga nel csv: {missing_csv}")
    missing_spec = sorted(set(json_rows) - set(specs))
    if missing_spec:
        # Senza lo spec il csv non e' ricostruibile: e' il record dei sigma usati.
        raise SystemExit(f"ERRORE: scenari senza spec in 'scenarios': {missing_spec}")

    order = sorted(json_rows, key=scenario_sort_key)
    return (
        [json_rows[name] for name in order],
        [specs[name] for name in order],
        [csv_rows[name] for name in order],
    )


def write_md(path: Path, csv_rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Model vs Chamfer Ranking Summary\n\n")
        f.write(MD_HEADER)
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in csv_rows:
            beats = "yes" if int(float(row["model_beats_chamfer"])) else "no"
            f.write("| " + " | ".join([row[col] for col in MD_COLUMNS] + [beats]) + " |\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("in_dirs", nargs="+", help="out dir degli stage ranking da fondere")
    p.add_argument("--out_dir", required=True, help="dove scrivere ranking_summary.{json,csv,md}")
    args = p.parse_args()

    in_dirs = [Path(d).expanduser().resolve() for d in args.in_dirs]
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir in in_dirs:
        raise SystemExit(f"ERRORE: --out_dir {out_dir} e' anche un input, riscriverebbe i dati di partenza")

    loaded = load_inputs(in_dirs)
    check_shared_keys(loaded)
    json_rows, specs, csv_rows = merge_rows(loaded)

    # Il resto del payload viene dal primo input: dopo check_shared_keys tutto cio' che
    # conta e' identico. Le chiavi che invece dipendono dal singolo job (device, statistiche
    # della cache Chamfer) vengono tenute per out dir dentro merged_from.
    payload = dict(loaded[0][1])
    payload["scenarios"] = specs
    payload["rows"] = json_rows
    payload["merged_from"] = [
        {
            "out_dir": str(d),
            "scenarios": [row["scenario"] for row in pl.get("rows", [])],
            "device": pl.get("device"),
            "chamfer_cache_verts": pl.get("chamfer_cache_verts"),
        }
        for d, pl, _rows, _fields in loaded
    ]
    for key in ("device", "chamfer_cache_verts"):
        payload.pop(key, None)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ranking_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
        f.write("\n")

    fieldnames = loaded[0][3]
    with open(out_dir / "ranking_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    write_md(out_dir / "ranking_summary.md", csv_rows)

    print(f"Fusi {len(loaded)} out dir -> {out_dir}")
    print(f"Scenari: {', '.join(row['scenario'] for row in json_rows)}")
    missing = [name for name in SCENARIO_ORDER if name not in {row["scenario"] for row in json_rows}]
    if missing:
        print(f"ATTENZIONE: mancano ancora {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
