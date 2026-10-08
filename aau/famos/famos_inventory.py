#!/usr/bin/env python3
"""Verifica dell'estrazione e inventario di FaMoS: soggetti, sequenze, fotogrammi, copertura del test.

    aau/run.sh aau/famos/famos_inventory.py --listing datasets/FAMOS/reg_archive_listing.tsv \\
        --log external_data/famos/extracted/x_reg_1062020.log --workers 16

Verifica (prima di qualunque cancellazione): ogni file elencato nell'archivio delle registrazioni
(``7z l -slt registrations.zip.001``, tsv ``dimensione<TAB>percorso``) esiste estratto con la stessa
dimensione in byte, nessun file in piu', e il log di 7z chiude con "Everything is Ok" (7z controlla
il CRC di ogni file). Le scansioni di test sono gia' state verificate allo stesso modo (8351 file,
65.769.349.665 byte, log OK) e il loro archivio cancellato: qui solo l'inventario.

Inventario (``aau/runs/evidence/famos/inventory.json`` e ``.md``): per soggetto le sequenze e i
fotogrammi registrati, le scansioni di test, la copertura (ogni scansione ha la registrazione dello
stesso fotogramma), lo split di ``aau/famos/split.json``. Solo conteggi, nessun dato.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import famos_common as fc  # noqa: E402


def scan_tree(root: Path, ext: str, workers: int) -> dict:
    """{soggetto: {sequenza: {fotogramma: dimensione}}} dei file ``*.ext`` (stat in parallelo: CephFS)."""
    subjects = sorted(d.name for d in root.iterdir() if d.is_dir() and fc.SUBJ_RE.match(d.name))

    def one(subject: str) -> tuple[str, dict]:
        seqs = {}
        for sd in sorted(p for p in (root / subject).iterdir() if p.is_dir()):
            fr = {}
            with os.scandir(sd) as it:
                for e in it:
                    m = fc.FRAME_RE.match(e.name)
                    if m and e.name.endswith(f".{ext}") and m["seq"] == sd.name:
                        fr[int(m["frame"])] = e.stat().st_size
            seqs[sd.name] = fr
        return subject, seqs

    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, subjects))


def verify(listing: Path, tree: dict, log: Path | None) -> dict:
    expected = {}
    for line in listing.read_text().splitlines():
        size, path = line.split("\t")
        parts = path.split("/")
        if len(parts) == 4 and path.endswith(".ply"):
            m = fc.FRAME_RE.match(parts[3])
            expected[(parts[1], parts[2], int(m["frame"]))] = int(size)
    found = {(s, q, f): sz for s, seqs in tree.items() for q, fr in seqs.items() for f, sz in fr.items()}
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    wrong = sorted(k for k in set(expected) & set(found) if expected[k] != found[k])
    log_ok = None
    if log is not None:
        log_ok = "Everything is Ok" in log.read_text(errors="replace")
    out = {"listed_ply": len(expected), "extracted_ply": len(found), "missing": len(missing), "extra": len(extra),
           "size_mismatch": len(wrong), "bytes_listed": sum(expected.values()), "bytes_extracted": sum(found.values()),
           "log_everything_ok": log_ok, "examples": {"missing": [list(map(str, k)) for k in missing[:5]],
                                                     "size_mismatch": [list(map(str, k)) for k in wrong[:5]]}}
    out["complete"] = (not missing and not extra and not wrong and log_ok is not False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--listing", type=Path, required=True)
    p.add_argument("--log", type=Path, default=None, help="log di 7z dell'estrazione delle registrazioni")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()

    reg = scan_tree(fc.REG_DIR, "ply", a.workers)
    scans = scan_tree(fc.SCAN_DIR, "obj", a.workers)
    check = verify(a.listing, reg, a.log)
    print(f"[famos-inv] verifica registrazioni: {json.dumps({k: v for k, v in check.items() if k != 'examples'})}",
          flush=True)

    split = fc.load_split()
    test, train = set(split["test"]), set(split["train"])
    per_subject = {}
    uncovered = []
    for s in sorted(reg):
        seqs = reg[s]
        row = {"split": "test" if s in test else ("train" if s in train else "NON NELLO SPLIT"),
               "n_sequences": len(seqs), "n_frames": sum(len(v) for v in seqs.values()),
               "frames_per_sequence": {q: len(v) for q, v in seqs.items()}}
        if s in scans:
            row["n_scan_sequences"] = len(scans[s])
            row["n_scans"] = sum(len(v) for v in scans[s].values())
            for q, fr in scans[s].items():
                for f in fr:
                    if f not in seqs.get(q, {}):
                        uncovered.append(f"{s}/{q}.{f:06d}")
        per_subject[s] = row
    scan_only = sorted(set(scans) - set(reg))
    all_seqs = sorted({q for s in reg.values() for q in s})
    seq_count = {q: sum(q in reg[s] for s in reg) for q in all_seqs}
    nf = [r["n_frames"] for r in per_subject.values()]
    ns = [r["n_sequences"] for r in per_subject.values()]
    inv = {
        "extraction_check": check,
        "n_subjects_registered": len(reg), "n_subjects_test_scans": len(scans),
        "test_scan_subjects": sorted(scans), "test_scan_subjects_without_registrations": scan_only,
        "test_scans_without_registration_of_same_frame": len(uncovered), "uncovered_examples": uncovered[:10],
        "n_frames_registered": int(sum(nf)), "n_scans": int(sum(r.get("n_scans", 0) for r in per_subject.values())),
        "frames_per_subject": {"min": min(nf), "median": sorted(nf)[len(nf) // 2], "max": max(nf)},
        "sequences_per_subject": {"min": min(ns), "median": sorted(ns)[len(ns) // 2], "max": max(ns)},
        "sequences": seq_count,
        "subjects_not_in_split": sorted(s for s in reg if s not in test | train),
        "split_subjects_missing": sorted((test | train) - set(reg)),
        "per_subject": per_subject,
    }
    fc.EVID_DIR.mkdir(parents=True, exist_ok=True)
    (fc.EVID_DIR / "inventory.json").write_text(json.dumps(inv, indent=1) + "\n")

    md = ["# FaMoS: inventario e verifica dell'estrazione", "",
          f"- Registrazioni: {len(reg)} soggetti, {inv['n_frames_registered']} fotogrammi (60 fps), sequenze per "
          f"soggetto {inv['sequences_per_subject']}, fotogrammi per soggetto {inv['frames_per_subject']}.",
          f"- Scansioni di test (TEMPEH): {len(scans)} soggetti ({min(scans)}..{max(scans)}), {inv['n_scans']} scansioni; "
          f"soggetti senza registrazioni: {scan_only or 'nessuno'}; scansioni senza la registrazione dello stesso "
          f"fotogramma: {len(uncovered)}.",
          f"- Split (aau/famos/split.json): TEST {len(test)}, TRAIN {len(train)}; soggetti registrati fuori dallo "
          f"split: {inv['subjects_not_in_split'] or 'nessuno'}; soggetti dello split senza registrazioni: "
          f"{inv['split_subjects_missing'] or 'nessuno'}.",
          f"- Verifica estrazione registrazioni: {check['extracted_ply']}/{check['listed_ply']} ply, mancanti "
          f"{check['missing']}, in piu' {check['extra']}, dimensione diversa {check['size_mismatch']}, byte "
          f"{check['bytes_extracted']}/{check['bytes_listed']}, log 7z OK: {check['log_everything_ok']} -> "
          f"**{'COMPLETA' if check['complete'] else 'INCOMPLETA'}**.", "",
          "| sequenza | soggetti |", "| --- | --- |"]
    md += [f"| {q} | {n} |" for q, n in seq_count.items()]
    md += ["", "| soggetto | split | sequenze | fotogrammi | scansioni di test |", "| --- | --- | --- | --- | --- |"]
    md += [f"| {s} | {r['split']} | {r['n_sequences']} | {r['n_frames']} | {r.get('n_scans', '')} |"
           for s, r in per_subject.items()]
    (fc.EVID_DIR / "inventory.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:6]), flush=True)
    if not check["complete"] or inv["subjects_not_in_split"] or inv["split_subjects_missing"] or scan_only:
        raise SystemExit("ERRORE: estrazione incompleta o split incoerente (vedi inventory.json)")


if __name__ == "__main__":
    main()
