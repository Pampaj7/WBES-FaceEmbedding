#!/usr/bin/env python3
"""Dai JSON dei partecipanti all'accordo di ogni metrica con la maggioranza umana (WS4).

Legge i file esportati da ``index.html`` (uno per partecipante), butta via chi ha
sbagliato piu' di ``--max-control-errors`` attention check su quattro, e per ogni metrica
misura quante volte la sua risposta attesa coincide con la maggioranza degli umani sulla
stessa tripletta.  Le triplette senza maggioranza netta (parita', o meno di
``--min-votes`` risposte) non entrano nel conto.

Il CI e' bootstrap sui PARTECIPANTI, non sulle triplette: l'unita' campionaria dello
studio e' la persona, e ricampionare le triplette darebbe intervalli troppo stretti
perche' le risposte dello stesso partecipante sono correlate.  Ogni replica rifa' anche
le maggioranze, che dipendono da chi e' stato estratto.

L'accordo fra annotatori e' il kappa di Fleiss nella forma a numero variabile di
giudici per item (ogni partecipante vede 40 triplette su 330, quindi ogni tripletta
raccoglie un numero diverso di voti).

  aau/human_study/analyze.py --responses-dir aau/human_study/responses
  aau/human_study/analyze.py --from-dir aau/human_study/responses_db   # export della hosted
  aau/human_study/analyze.py --self-test        # 5 partecipanti simulati, nessun dato reale
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent

CHOICES = ("b", "c")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--triplets", type=Path, default=THIS_DIR / "triplets.json")
    p.add_argument("--responses-dir", type=Path, default=THIS_DIR / "responses")
    p.add_argument("--from-dir", type=Path, default=None,
                   help="documenti della collection `responses` esportati dalla versione "
                        "ospitata (hosted/index.html), uno per file; ha la precedenza su "
                        "--responses-dir")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <dir dei dati>/analysis")
    p.add_argument("--max-control-errors", type=int, default=1,
                   help="partecipanti con piu' errori di cosi' sugli attention check vengono scartati")
    p.add_argument("--min-votes", type=int, default=3,
                   help="voti minimi perche' una tripletta abbia una maggioranza usabile")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--self-test", action="store_true",
                   help="genera 5 partecipanti simulati in una dir temporanea e analizza quelli")
    p.add_argument("--sim-participants", type=int, default=5)
    p.add_argument("--sim-pool", type=int, default=60,
                   help="test simulati estratti da un pool ridotto, per avere gli stessi "
                        "~3 voti a tripletta di uno studio vero da 25-30 persone su 300")
    p.add_argument("--sim-trials", type=int, default=36)
    p.add_argument("--sim-accuracy", type=float, default=0.75,
                   help="probabilita' che un partecipante simulato risponda come la GT")
    return p.parse_args()


# ------------------------------------------------------------------ lettura

def load_triplets(path: Path) -> tuple[dict, dict]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["meta"], {t["id"]: t for t in payload["triplets"]}


def load_responses(responses_dir: Path) -> list[dict]:
    files = sorted(p for p in responses_dir.glob("*.json") if p.is_file())
    if not files:
        raise SystemExit(f"[analyze] nessun JSON di partecipante in {responses_dir}")
    out = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
        record["_file"] = path.name
        out.append(record)
    return out


# Chiavi sotto cui un export puo' aver messo il corpo del documento invece di scriverlo
# nudo: quale sia dipende da chi ha fatto l'export, non dal db.
DOC_ENVELOPES = ("data", "document", "body")


def _documents(payload, where: str) -> list[dict]:
    """Corpi di documento contenuti in un file di export: nudo, imbustato o in lista."""
    if isinstance(payload, list):
        return [doc for item in payload for doc in _documents(item, where)]
    if not isinstance(payload, dict):
        raise SystemExit(f"[analyze] {where}: atteso un oggetto JSON, "
                         f"trovato {type(payload).__name__}")
    for key in DOC_ENVELOPES:
        inner = payload.get(key)
        if isinstance(inner, dict) and "trials" in inner:
            return [inner]
    return [payload]


def _from_hosted(doc: dict, where: str) -> dict:
    """Un documento `responses/<codice>` nel formato che il resto dello script si aspetta.

    La pagina ospitata scrive il minimo indispensabile (``trials`` con ``choice``,
    ``control``, ``shown_left``); ``kind``, ``chosen`` e il lato destro si ricavano qui,
    una volta sola, cosi' screening, maggioranze e bootstrap restano quelli.
    """
    if "trials" not in doc:
        raise SystemExit(f"[analyze] {where}: manca il campo 'trials' "
                         f"(chiavi trovate: {sorted(doc)})")
    responses = []
    for position, trial in enumerate(doc["trials"], start=1):
        try:
            left = trial["shown_left"]
            responses.append({
                "position": position,
                "triplet_id": trial["triplet_id"],
                "kind": "control" if trial["control"] else "test",
                "left": left,
                "right": "c" if left == "b" else "b",
                "chosen": trial["choice"],
                "rt_ms": trial["rt_ms"],
            })
        except KeyError as exc:
            raise SystemExit(f"[analyze] {where}, prova {position}: manca il campo {exc}")
    return {
        "app_version": "hosted",
        "participant": doc.get("participant", where),
        "started_at": doc.get("started"),
        "finished_at": doc.get("finished"),
        "n_trials": len(responses),
        "user_agent": doc.get("user_agent"),
        "responses": responses,
    }


def load_hosted_responses(from_dir: Path) -> list[dict]:
    """I documenti della collection `responses` esportati dalla versione ospitata."""
    files = sorted(p for p in from_dir.glob("*.json") if p.is_file())
    if not files:
        raise SystemExit(f"[analyze] nessun documento JSON in {from_dir}")
    out = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        for doc in _documents(payload, path.name):
            record = _from_hosted(doc, path.name)
            record["_file"] = path.name
            out.append(record)
    return out


def screen(participants: list[dict], triplets: dict, max_errors: int) -> tuple[list[dict], list[dict]]:
    """Controllo degli attention check: la risposta attesa e' quella (unanime) delle metriche."""
    rows = []
    for record in participants:
        n_control = n_correct = 0
        for answer in record["responses"]:
            triplet = triplets.get(answer["triplet_id"])
            if triplet is None or triplet["kind"] != "control":
                continue
            expected = {m["expected"] for m in triplet["metrics"].values()}
            if len(expected) != 1:
                raise ValueError(f"{triplet['id']}: controllo con metriche in disaccordo")
            n_control += 1
            n_correct += int(answer["chosen"] == expected.pop())
        rts = [a["rt_ms"] for a in record["responses"]]
        rows.append({
            "participant": record.get("participant", record["_file"]),
            "file": record["_file"],
            "n_responses": len(record["responses"]),
            "n_control": n_control,
            "n_control_correct": n_correct,
            "control_errors": n_control - n_correct,
            "median_rt_ms": float(np.median(rts)) if rts else math.nan,
            "kept": bool(n_control > 0 and (n_control - n_correct) <= max_errors),
        })
    kept = [record for record, row in zip(participants, rows) if row["kept"]]
    return kept, rows


# ------------------------------------------------------------------ voti e accordo

def vote_matrix(participants: list[dict], triplet_ids: list[str]) -> np.ndarray:
    """(n_partecipanti x n_triplette) con 0 = ha scelto B, 1 = ha scelto C, -1 = non vista."""
    position = {tid: k for k, tid in enumerate(triplet_ids)}
    V = np.full((len(participants), len(triplet_ids)), -1, dtype=np.int8)
    for i, record in enumerate(participants):
        for answer in record["responses"]:
            k = position.get(answer["triplet_id"])
            if k is not None:
                V[i, k] = CHOICES.index(answer["chosen"])
    return V


def majority(counts_b: np.ndarray, counts_c: np.ndarray, min_votes: int):
    """Maggioranza netta per tripletta: niente parita', niente triplette poco viste."""
    total = counts_b + counts_c
    usable = (total >= min_votes) & (counts_b != counts_c)
    return usable, np.where(counts_c > counts_b, 1, 0)


def _votes_per_seen(total: np.ndarray) -> float:
    seen = total[total > 0]
    return float(seen.mean()) if seen.size else 0.0


def agreement(expected: np.ndarray, usable: np.ndarray, choice: np.ndarray) -> float:
    if not usable.any():
        return math.nan
    return float((expected[usable] == choice[usable]).mean())


def fleiss_kappa(counts_b: np.ndarray, counts_c: np.ndarray) -> tuple[float, int]:
    """Kappa di Fleiss con numero di giudici variabile per item (item con >= 2 voti)."""
    n_i = counts_b + counts_c
    keep = n_i >= 2
    if keep.sum() < 2:
        return math.nan, int(keep.sum())
    b, c, n = counts_b[keep].astype(float), counts_c[keep].astype(float), n_i[keep].astype(float)
    p_i = (b ** 2 + c ** 2 - n) / (n * (n - 1.0))
    p_bar = float(p_i.mean())
    p_b = float(b.sum() / n.sum())
    p_e = p_b ** 2 + (1.0 - p_b) ** 2
    if math.isclose(p_e, 1.0):
        return math.nan, int(keep.sum())
    return (p_bar - p_e) / (1.0 - p_e), int(keep.sum())


def analyse(kept: list[dict], triplets: dict, metrics: list[str], args) -> dict:
    test_ids = sorted(tid for tid, t in triplets.items() if t["kind"] == "test")
    V = vote_matrix(kept, test_ids)
    is_b = (V == 0).astype(np.int32)
    is_c = (V == 1).astype(np.int32)

    counts_b, counts_c = is_b.sum(axis=0), is_c.sum(axis=0)
    usable, choice = majority(counts_b, counts_c, args.min_votes)
    expected = {m: np.array([CHOICES.index(triplets[t]["metrics"][m]["expected"]) for t in test_ids])
                for m in metrics}

    rng = np.random.default_rng(args.seed)
    n_participants = len(kept)
    draws = {m: np.empty(args.n_bootstrap) for m in metrics}
    for r in range(args.n_bootstrap):
        idx = rng.integers(0, n_participants, n_participants)
        cb, cc = is_b[idx].sum(axis=0), is_c[idx].sum(axis=0)
        u, ch = majority(cb, cc, args.min_votes)
        for m in metrics:
            draws[m][r] = agreement(expected[m], u, ch)

    kappa, n_kappa_items = fleiss_kappa(counts_b, counts_c)
    rows = []
    for m in metrics:
        finite = draws[m][np.isfinite(draws[m])]
        low, high = (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))) \
            if finite.size else (math.nan, math.nan)
        rows.append({
            "metric": m,
            "agreement": agreement(expected[m], usable, choice),
            "ci_low": low,
            "ci_high": high,
            "n_triplets": int(usable.sum()),
            "n_participants": n_participants,
            "n_bootstrap": args.n_bootstrap,
        })
    return {
        "rows": rows,
        "kappa": kappa,
        "n_kappa_items": n_kappa_items,
        "counts": {"n_test_triplets": len(test_ids),
                   "n_seen": int((counts_b + counts_c > 0).sum()),
                   "n_usable": int(usable.sum()),
                   "n_votes": int((counts_b + counts_c).sum()),
                   "votes_per_seen_triplet": _votes_per_seen(counts_b + counts_c)},
    }


# ------------------------------------------------------------------ output

def write_report(out_dir: Path, args, meta, screening, result) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv

    csv_path = out_dir / "human_agreement.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])

    (out_dir / "human_agreement.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "triplets": str(args.triplets),
        "responses_dir": str(args.from_dir or args.responses_dir),
        "triplets_meta": meta,
        "screening": screening,
        "result": result,
    }, indent=1), encoding="utf-8")

    kept = [r for r in screening if r["kept"]]
    lines = [
        "# Accordo con il giudizio umano (WS4)",
        "",
        f"{len(kept)} partecipanti tenuti su {len(screening)} "
        f"(scartati: piu' di {args.max_control_errors} errori sui controlli).",
        f"{result['counts']['n_usable']} triplette con maggioranza netta sulle "
        f"{result['counts']['n_seen']} viste da qualcuno (su {result['counts']['n_test_triplets']} "
        f"nel pool), {result['counts']['votes_per_seen_triplet']:.1f} voti per tripletta vista, "
        f"minimo richiesto {args.min_votes}.",
        f"Kappa di Fleiss fra annotatori: {result['kappa']:.3f} "
        f"su {result['n_kappa_items']} triplette con almeno 2 voti.",
        "",
        "| metrica | accordo con la maggioranza | CI 95% (bootstrap sui partecipanti) | triplette |",
        "|---|---:|---|---:|",
    ]
    for row in sorted(result["rows"], key=lambda r: -r["agreement"]):
        lines.append(f"| {row['metric']} | {row['agreement']:.3f} | "
                     f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {row['n_triplets']} |")
    lines += ["", "## Partecipanti", "",
              "| partecipante | risposte | controlli | RT mediano (ms) | tenuto |",
              "|---|---:|---:|---:|---|"]
    for row in screening:
        lines.append(f"| {row['participant']} | {row['n_responses']} | "
                     f"{row['n_control_correct']}/{row['n_control']} | "
                     f"{row['median_rt_ms']:.0f} | {'si' if row['kept'] else 'NO'} |")
    md_path = out_dir / "human_agreement.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


# ------------------------------------------------------------------ partecipanti simulati

def simulate(triplets: dict, meta: dict, out_dir: Path, args) -> None:
    """Partecipanti finti per collaudare la pipeline: uno di loro sbaglia i controlli.

    Il modello e' volutamente grezzo: l'umano simulato segue la GT con probabilita'
    ``--sim-accuracy``.  Serve solo a verificare che lo screening, le maggioranze, il
    bootstrap e il kappa girino su file nel formato vero, non a stimare niente.
    """
    rng = np.random.default_rng(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_ids = sorted(tid for tid, t in triplets.items() if t["kind"] == "test")
    control_ids = sorted(tid for tid, t in triplets.items() if t["kind"] == "control")
    pool = list(rng.choice(test_ids, size=min(args.sim_pool, len(test_ids)), replace=False))

    for p in range(args.sim_participants):
        sloppy = (p == args.sim_participants - 1)   # l'ultimo non guarda lo schermo
        seen = list(rng.choice(pool, size=min(args.sim_trials, len(pool)), replace=False))
        controls = list(rng.choice(control_ids, size=4, replace=False))
        responses = []
        n_control_seen = 0
        for position, tid in enumerate(rng.permutation(seen + controls), start=1):
            triplet = triplets[tid]
            truth = triplet["metrics"]["gt"]["expected"]
            other = "c" if truth == "b" else "b"
            if triplet["kind"] == "control":
                # Lo sciatto sbaglia i primi 2 controlli su 4, quindi deve essere scartato.
                n_control_seen += 1
                chosen = other if (sloppy and n_control_seen <= 2) else truth
            elif sloppy:
                chosen = CHOICES[int(rng.integers(0, 2))]
            else:
                chosen = truth if rng.random() < args.sim_accuracy else other
            responses.append({
                "position": position,
                "triplet_id": tid,
                "kind": triplet["kind"],
                "a": triplet["a"], "b": triplet["b"], "c": triplet["c"],
                "left": "b", "right": "c",
                "chosen": chosen,
                "chosen_subject": triplet[chosen],
                "rt_ms": int(rng.normal(4200, 900)),
                "input": "click",
            })
        code = f"PSIM{p:03d}"
        (out_dir / f"wbes_human_study_{code}.json").write_text(json.dumps({
            "app_version": "ws4-1-simulated",
            "participant": code,
            "seed": int(args.seed + p),
            "started_at": "2026-01-01T09:00:00.000Z",
            "finished_at": "2026-01-01T09:10:00.000Z",
            "n_trials": len(responses),
            "user_agent": "simulated",
            "screen": {"w": 1920, "h": 1080, "dpr": 1},
            "triplets_meta": {k: meta.get(k) for k in
                              ("generated_at", "seed", "metrics", "n_test", "n_control")},
            "responses": responses,
        }, indent=1), encoding="utf-8")


def run_self_test(args, meta, triplets, metrics) -> int:
    with tempfile.TemporaryDirectory(prefix="wbes_selftest_") as tmp:
        tmp_dir = Path(tmp)
        args.responses_dir = tmp_dir / "responses"
        simulate(triplets, meta, args.responses_dir, args)
        participants = load_responses(args.responses_dir)
        kept, screening = screen(participants, triplets, args.max_control_errors)
        result = analyse(kept, triplets, metrics, args)
        report = write_report(tmp_dir / "analysis", args, meta, screening, result)
        print(report.read_text(encoding="utf-8"))

        failures = []
        if len(kept) != args.sim_participants - 1:
            failures.append(f"attesi {args.sim_participants - 1} partecipanti tenuti, "
                            f"trovati {len(kept)}")
        if result["counts"]["n_usable"] < 10:
            failures.append(f"solo {result['counts']['n_usable']} triplette con maggioranza")
        best = max(result["rows"], key=lambda r: r["agreement"])["metric"]
        if best != "gt":
            failures.append(f"la metrica piu' in accordo doveva essere gt (oracolo), e' {best}")
        if not math.isfinite(result["kappa"]):
            failures.append("kappa di Fleiss non finito")
        for row in result["rows"]:
            if not (row["ci_low"] <= row["agreement"] <= row["ci_high"]):
                failures.append(f"{row['metric']}: stima fuori dal proprio CI")
        for line in failures:
            print(f"[self-test] FALLITO: {line}", file=sys.stderr)
        print(f"[self-test] {'OK' if not failures else 'FALLITO'}: "
              f"{args.sim_participants} partecipanti simulati, {len(kept)} tenuti")
        return 1 if failures else 0


def main() -> int:
    args = parse_args()
    meta, triplets = load_triplets(args.triplets)
    metrics = list(meta["metrics"])

    if args.self_test:
        return run_self_test(args, meta, triplets, metrics)

    source = args.from_dir or args.responses_dir
    participants = (load_hosted_responses(args.from_dir) if args.from_dir
                    else load_responses(args.responses_dir))
    kept, screening = screen(participants, triplets, args.max_control_errors)
    print(f"[analyze] partecipanti {len(participants)}, tenuti {len(kept)}, "
          f"scartati {len(participants) - len(kept)}", flush=True)
    if not kept:
        raise SystemExit("[analyze] nessun partecipante ha superato gli attention check")

    result = analyse(kept, triplets, metrics, args)
    out_dir = args.out_dir or (source / "analysis")
    report = write_report(out_dir, args, meta, screening, result)
    for row in sorted(result["rows"], key=lambda r: -r["agreement"]):
        print(f"[analyze] {row['metric']:<10} accordo={row['agreement']:.3f} "
              f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] su {row['n_triplets']} triplette")
    print(f"[analyze] kappa di Fleiss={result['kappa']:.3f} "
          f"({result['n_kappa_items']} triplette con >= 2 voti)")
    print(f"[analyze] report {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
