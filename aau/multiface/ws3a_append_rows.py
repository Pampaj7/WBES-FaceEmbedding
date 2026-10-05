#!/usr/bin/env python3
"""Aggiunge in CODA a summary_hard.{csv,md} le righe di metriche nuove prese da un altro summary di
ws3a_analysis.py, senza ricalcolare ne' toccare le righe esistenti.

ws3a_analysis.py riscrive il summary intero, compreso il paragrafo del proxy che conta le metriche
("x/10 metriche non lo battono"): rilanciarlo con le metriche nuove cambierebbe il testo delle righe
vecchie. Qui le metriche nuove si analizzano in un summary a parte (stessa out-root, stessa coppia di
riferimento, stesso seme del bootstrap per confronto, quindi numeri identici a una corsa unica) e si
accodano. Rifiuta metriche gia' presenti. Solo stdlib: gira sul frontend.

    python3 aau/multiface/ws3a_append_rows.py --src aau/runs/multiface_ws3a_hard/summary_hard_abl \
        --dst aau/runs/multiface_ws3a_hard/summary_hard --note "..."
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="summary senza estensione con le righe nuove")
    ap.add_argument("--dst", type=Path, required=True, help="summary senza estensione da estendere")
    ap.add_argument("--note", type=str, default="", help="riga di nota dopo la tabella")
    args = ap.parse_args()

    src_csv, dst_csv = args.src.with_suffix(".csv"), args.dst.with_suffix(".csv")
    with open(dst_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        have = {r["metric"] for r in reader}
    with open(src_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != fields:
            raise SystemExit(f"colonne diverse: {src_csv} {reader.fieldnames} contro {dst_csv} {fields}")
        rows = list(reader)
    clash = sorted({r["metric"] for r in rows} & have)
    if clash:
        raise SystemExit(f"metriche gia' presenti in {dst_csv}: {clash}")
    with open(dst_csv, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writerows(rows)

    new = sorted({r["metric"] for r in rows})
    md_rows = [l for l in args.src.with_suffix(".md").read_text().splitlines()
               if l.startswith("| ") and l.split("|")[1].strip() in new]
    dst_md = args.dst.with_suffix(".md")
    lines = dst_md.read_text().rstrip("\n").splitlines()
    last = max(i for i, l in enumerate(lines) if l.startswith("| "))   # fine della tabella
    lines[last + 1:last + 1] = md_rows
    if args.note:
        lines += ["", args.note]
    dst_md.write_text("\n".join(lines) + "\n")
    print(f"accodate {len(rows)} righe csv e {len(md_rows)} righe md per {new}")


if __name__ == "__main__":
    main()
