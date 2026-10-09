#!/usr/bin/env python3
"""Studio umano v2, passo 3: triplette INFORMATIVE fra le GT di E12, bilanciate per tipo di disaccordo.

    v3_work/unified_gt/run.sh aau/human_study_v2/select_triplets_v2.py      (run.sbatch, passo ``select``)

Pool completo sui 100 soggetti (100 x C(99,2) = 485.100 triplette, A riferimento, {B, C} non ordinata), come la
v1 (``aau/human_study/select_triplets.py``). Una tripletta e' del tipo ``X_vs_Y`` (``--pairs``) se le GT X e Y
ordinano d(A,B) e d(A,C) al contrario, CIASCUNA con margine relativo ``|d(A,B) - d(A,C)| / media >= --margin``.
Tipi di default: F contro S, F contro maxabs, S contro maxabs, EDM contro F. I tipi si riempiono dal pool piu'
piccolo al piu' grande senza riusare una tripletta; dentro un tipo si estrae a caso fra le ``--pool-factor x
--n-per-pair`` con il margine piu' grande (il minimo dei due margini), cosi' il disaccordo e' netto ma non
ristretto ai pochi casi estremi, con un tetto di ``--max-uses`` comparse per soggetto nei test.

Controlli (attention check) e prove della sessione di prova: triplette in cui TUTTE le ``--control-gts`` sono
d'accordo con margine >= ``--control-margin`` su ognuna; i due insiemi sono disgiunti.

Le GT vengono da ``gt_v2.py`` (``datasets/HUMAN_STUDY_V2/gt``): provvisorie finche' E12 non ha finito
(``manifest.json`` -> ``frame.e12_complete``); il parametro di stato finisce nel meta delle triplette. Ogni
tripletta porta d(A,B), d(A,C), margine e risposta attesa di TUTTE le GT su disco (anche quelle fuori dai tipi:
unified, EDM_s, varianti rigide di F, baseline "solo taglia" e "solo altezza"), per l'analisi.

Uscite: ``aau/human_study_v2/triplets.json`` (con le metriche), ``triplets_stats.md``,
``docs/human_study_v2/triplets.js`` (senza metriche) e ``docs/human_study_v2/img/<soggetto>.jpg`` (le strisce di
``render_v2.py`` dei soli soggetti usati).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import hs2

PRINCIPAL = ("F", "S", "EDM", "maxabs")
# Lo stesso Google Form riceve v1 e v2: gli id della v2 hanno un prefisso che non si confonde con la v1
ID_PREFIX = "v2_"
PAIRS = "F:S,F:maxabs,S:maxabs,EDM:F"
CONTROL_GTS = "F,S,EDM,EDM_s,unified,maxabs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gt-dir", type=Path, default=hs2.GT_DIR)
    p.add_argument("--renders", type=Path, default=hs2.RENDER_DIR)
    p.add_argument("--out-dir", type=Path, default=hs2.THIS_DIR)
    p.add_argument("--docs-dir", type=Path, default=hs2.DOCS_DIR)
    p.add_argument("--pairs", default=PAIRS, help="tipi di disaccordo X:Y separati da virgole")
    p.add_argument("--control-gts", default=CONTROL_GTS)
    p.add_argument("--n-per-pair", type=int, default=60)
    p.add_argument("--n-control", type=int, default=30)
    p.add_argument("--n-practice", type=int, default=6)
    p.add_argument("--margin", type=float, default=0.10,
                   help="margine relativo minimo per OGNUNA delle due GT in disaccordo")
    p.add_argument("--pool-factor", type=int, default=5,
                   help="si estrae fra le n-per-pair x pool-factor col margine minimo piu' grande")
    p.add_argument("--max-uses", type=int, default=15, help="comparse massime di un soggetto nei test")
    p.add_argument("--control-margin", type=float, default=0.40)
    p.add_argument("--control-pool-factor", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-copy-images", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------------------ dati

def load_gts(gt_dir: Path) -> tuple[list[str], dict, dict]:
    man = hs2.read_json(gt_dir / "manifest.json")
    D, subjects = {}, None
    for g in man["gts"]:
        with np.load(gt_dir / f"{g}.npz") as z:
            names = [str(s) for s in z["names"]]
            if subjects is None:
                subjects = names
            elif names != subjects:
                raise SystemExit(f"{g}: soggetti diversi dalle altre GT")
            D[g] = np.asarray(z["D"], dtype=np.float64)
    return subjects, D, man


def triplet_pool(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tutte le (A, B, C) con A fuori da {B, C} e B < C, in ordine deterministico (come la v1)."""
    ii, jj = np.triu_indices(n - 1, k=1)
    a_list, b_list, c_list = [], [], []
    for a in range(n):
        others = np.delete(np.arange(n, dtype=np.int32), a)
        a_list.append(np.full(ii.size, a, dtype=np.int32))
        b_list.append(others[ii])
        c_list.append(others[jj])
    return np.concatenate(a_list), np.concatenate(b_list), np.concatenate(c_list)


def metric_view(S: np.ndarray, a, b, c) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """d(A,B), d(A,C) e margine relativo |d(A,B) - d(A,C)| / media, come la v1."""
    d_ab, d_ac = S[a, b], S[a, c]
    mean = 0.5 * (d_ab + d_ac)
    with np.errstate(invalid="ignore", divide="ignore"):
        margin = np.where(mean > 0, np.abs(d_ab - d_ac) / mean, 0.0)
    return d_ab, d_ac, margin


# ------------------------------------------------------------------------ selezione

def pair_pools(pairs, view, margin_thr) -> dict:
    """Indici del pool in cui X e Y sono decisive e opposte, e il margine minimo dei due."""
    out = {}
    for x, y in pairs:
        dxb, dxc, mx = view[x]
        dyb, dyc, my = view[y]
        ok = (mx >= margin_thr) & (my >= margin_thr) & ((dxb < dxc) != (dyb < dyc))
        idx = np.flatnonzero(ok)
        out[f"{x}_vs_{y}"] = (idx, np.minimum(mx, my)[idx])
    return out


def take_pairs(pools: dict, n: int, factor: int, max_uses: int, abc, rng) -> dict:
    """Dal tipo piu' raro al piu' comune: le n*factor col margine piu' grande fra quelle non ancora usate, in
    ordine casuale, poi il resto del pool per margine decrescente; si accetta una tripletta solo se nessuno dei
    suoi tre soggetti e' gia' comparso ``max_uses`` volte nei test (i disaccordi si concentrano sui volti
    estremi: senza tetto lo stesso volto tornerebbe in una prova su cinque)."""
    a, b, c = abc
    used: set[int] = set()
    uses = Counter()
    taken = {}
    for label in sorted(pools, key=lambda t: (len(pools[t][0]), t)):
        idx, m = pools[label]
        free = np.array([i not in used for i in idx], dtype=bool)
        idx, m = idx[free], m[free]
        order = idx[np.argsort(-m, kind="stable")]
        cand = np.concatenate([rng.permutation(order[: n * factor]), order[n * factor:]])
        pick = []
        for i in cand:
            s3 = (int(a[i]), int(b[i]), int(c[i]))
            if all(uses[s] < max_uses for s in s3):
                pick.append(int(i))
                uses.update(s3)
                if len(pick) == n:
                    break
        if len(pick) < n:
            raise SystemExit(f"[hs2-select] {label}: solo {len(pick)} triplette libere con margine sufficiente e "
                             f"soggetti sotto il tetto, ne servono {n}. Abbassa --margin o alza --max-uses.")
        taken[label] = np.sort(np.array(pick))
        used.update(pick)
    return taken


def unanimous(view, gts, thr) -> tuple[np.ndarray, np.ndarray]:
    closer_b = np.stack([view[g][0] < view[g][1] for g in gts])
    margins = np.stack([view[g][2] for g in gts])
    ok = (closer_b.all(0) | (~closer_b).all(0)) & (margins.min(0) >= thr)
    return np.flatnonzero(ok), margins.min(0)


def record(idx: int, kind: str, tid: str, label: str, pair, subjects, a, b, c, view, margin_thr, swap) -> dict:
    s_a, s_b, s_c = subjects[a[idx]], subjects[b[idx]], subjects[c[idx]]
    if swap:
        s_b, s_c = s_c, s_b
    entry = {"id": tid, "kind": kind, "disagreement_type": label, "pair": list(pair) if pair else None,
             "a": s_a, "b": s_b, "c": s_c, "metrics": {}}
    for g, (d_ab, d_ac, margin) in view.items():
        ab, ac = float(d_ab[idx]), float(d_ac[idx])
        if swap:
            ab, ac = ac, ab
        entry["metrics"][g] = {"d_ab": round(ab, 6), "d_ac": round(ac, 6), "margin": round(float(margin[idx]), 6),
                               "expected": "b" if ab < ac else "c", "decisive": bool(margin[idx] >= margin_thr)}
    return entry


# ----------------------------------------------------------------------------- uscite

def copy_images(entries, renders: Path, docs_dir: Path) -> tuple[int, int]:
    img_dir = docs_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.jpg"):
        old.unlink()
    used = sorted({e[k] for e in entries for k in ("a", "b", "c")})
    total = 0
    for s in used:
        src = renders / f"{s}.jpg"
        if not src.exists():
            raise FileNotFoundError(f"render mancante: {src} (rigeneralo con render_v2.py)")
        shutil.copyfile(src, img_dir / f"{s}.jpg")
        total += (img_dir / f"{s}.jpg").stat().st_size
    return len(used), total


def write_stats(path: Path, args, meta, entries, pools, view_names) -> None:
    tests = [e for e in entries if e["kind"] == "test"]
    ctrl = [e for e in entries if e["kind"] == "control"]
    prac = [e for e in entries if e["kind"] == "practice"]
    usage = Counter(s for e in entries for s in (e["a"], e["b"], e["c"]))
    fr = meta["gt_frame"]
    lines = [
        "# Triplette dello studio umano v2", "",
        f"Generato il {meta['generated_at']} da `aau/human_study_v2/select_triplets_v2.py`, seed {args.seed}.", "",
        f"**Stato delle GT: {meta['gt_status']}** (frame di GT-F: {fr['source']}, E12 concluso: "
        f"{fr.get('e12_complete')}, impronta di cgt.py+gt.py `{fr['code_sha256'][:12]}`).", "",
        "| parametro | valore |", "|---|---|",
        f"| soggetti | {meta['n_subjects']} (BFM REMESH `{meta['topology']}`, held-out della v1) |",
        f"| GT su disco | {', '.join(meta['gts'])} |",
        f"| tipi | {', '.join(meta['types'])} |",
        f"| pool completo | {meta['pool_size']} triplette (A, {{B, C}}) |",
        f"| margine relativo | >= {args.margin:.2f} su entrambe le GT in disaccordo; estrazione fra le "
        f"{args.pool_factor} x {args.n_per_pair} col margine minimo piu' grande, <= {args.max_uses} comparse per "
        f"soggetto nei test |",
        f"| controlli | {len(ctrl)}, unanimi su {args.control_gts} con margine >= {args.control_margin:.2f} |",
        f"| prova | {len(prac)} triplette unanimi (stessa regola, disgiunte dai controlli), escluse dall'analisi |",
        f"| test | {len(tests)} |", "",
        "## Tipi di disaccordo", "",
        "| tipo | disponibili (margine ok) | scelte | margine minimo: mediana / min nelle scelte |",
        "|---|---:|---:|---|",
    ]
    for label, (idx, m) in sorted(pools.items(), key=lambda kv: len(kv[1][0])):
        chosen = [e for e in tests if e["disagreement_type"] == label]
        mm = np.array([min(e["metrics"][x]["margin"] for x in e["pair"]) for e in chosen])
        lines.append(f"| `{label}` | {len(idx)} | {len(chosen)} | {np.median(mm):.3f} / {mm.min():.3f} |")
    lines += ["", "## Come votano le altre GT dentro ogni tipo", "",
              "Quota delle triplette del tipo in cui la GT da' la stessa risposta della PRIMA GT del tipo (X in "
              "`X_vs_Y`); 1 = sta con X, 0 = sta con Y. Dice che cosa si confronta davvero in ogni strato.", "",
              "| GT | " + " | ".join(f"`{t}`" for t in meta["types"]) + " |",
              "|---|" + "---:|" * len(meta["types"])]
    for g in view_names:
        cells = []
        for t in meta["types"]:
            chosen = [e for e in tests if e["disagreement_type"] == t]
            x = chosen[0]["pair"][0]
            cells.append(f"{np.mean([e['metrics'][g]['expected'] == e['metrics'][x]['expected'] for e in chosen]):.2f}")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    lines += ["", "## Margini relativi nelle triplette scelte (mediana)", "",
              "| GT | test | controlli |", "|---|---:|---:|"]
    for g in view_names:
        lines.append(f"| {g} | {np.median([e['metrics'][g]['margin'] for e in tests]):.3f} | "
                     f"{np.median([e['metrics'][g]['margin'] for e in ctrl]):.3f} |")
    lines += ["", "## Copertura dei soggetti", "",
              f"- {len(usage)} soggetti distinti sui {meta['n_subjects']}; comparse per soggetto: min "
              f"{min(usage.values())}, mediana {int(np.median(list(usage.values())))}, max {max(usage.values())}.",
              "", "## Pacchetto", "",
              f"- {meta['n_images']} strisce JPEG (frontale, 3/4, profilo) in `docs/human_study_v2/img/` "
              f"({meta['image_bytes'] / 1e6:.1f} MB), camera unica {meta['camera']['px_per_mm']:.3f} px/mm.",
              "- `triplets.json` (con metriche, per l'analisi) e `docs/human_study_v2/triplets.js` (senza).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    subjects, D, man = load_gts(args.gt_dir)
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p.strip()]
    control_gts = [g for g in args.control_gts.split(",") if g.strip()]
    for g in {x for p in pairs for x in p} | set(control_gts):
        if g not in D:
            raise SystemExit(f"GT {g} assente da {args.gt_dir}")
    a, b, c = triplet_pool(len(subjects))
    view = {g: metric_view(D[g], a, b, c) for g in D}
    print(f"[hs2-select] {len(subjects)} soggetti, pool {a.size}, GT {list(D)}, stato "
          f"{'definitivo' if man['frame'].get('e12_complete') else 'provvisorio'}", flush=True)

    pools = pair_pools(pairs, view, args.margin)
    for label, (idx, m) in pools.items():
        print(f"[hs2-select] {label:<18} {idx.size:>7} disponibili, margine minimo mediano "
              f"{np.median(m) if idx.size else float('nan'):.3f}", flush=True)
    taken = take_pairs(pools, args.n_per_pair, args.pool_factor, args.max_uses, (a, b, c), rng)
    used = {int(i) for v in taken.values() for i in v}

    upool, umin = unanimous(view, control_gts, args.control_margin)
    upool = np.array([i for i in upool if int(i) not in used], dtype=np.int64)
    need = args.n_control + args.n_practice
    if upool.size < need:
        raise SystemExit(f"[hs2-select] solo {upool.size} triplette unanimi con margine >= {args.control_margin}")
    top = upool[np.argsort(-umin[upool], kind="stable")][: max(need, args.n_control * args.control_pool_factor)]
    pick = rng.permutation(top)[:need]
    chosen_control, chosen_practice = np.sort(pick[: args.n_control]), np.sort(pick[args.n_control:])
    print(f"[hs2-select] unanimi: {upool.size}, controlli {chosen_control.size}, prova {chosen_practice.size}",
          flush=True)

    # B e C si scambiano a caso: nel JSON non resta traccia di quale GT sta su b (la pagina sorteggia il lato).
    entries, n = [], 0
    for label in [f"{x}_vs_{y}" for x, y in pairs]:
        for idx in taken[label]:
            n += 1
            entries.append(record(int(idx), "test", f"{ID_PREFIX}t{n:04d}", label, label.split("_vs_"), subjects,
                                  a, b, c, view, args.margin, bool(rng.random() < 0.5)))
    for k, idx in enumerate(chosen_control, start=1):
        entries.append(record(int(idx), "control", f"{ID_PREFIX}c{k:04d}", "unanime", None, subjects, a, b, c, view,
                              args.margin, bool(rng.random() < 0.5)))
    for k, idx in enumerate(chosen_practice, start=1):
        entries.append(record(int(idx), "practice", f"{ID_PREFIX}p{k:04d}", "unanime", None, subjects, a, b, c, view,
                              args.margin, bool(rng.random() < 0.5)))
    for e in entries:
        if e["kind"] != "test":
            exp = {e["metrics"][g]["expected"] for g in control_gts}
            if len(exp) != 1:
                raise AssertionError(f"{e['id']}: controllo non unanime")

    camera = hs2.read_json(args.renders / "camera.json") if (args.renders / "camera.json").exists() else {}
    if not args.no_copy_images and camera.get("frame", {}).get("frame") != man["frame"]["frame"]:
        raise SystemExit("[hs2-select] render assenti o con un frame diverso dalle GT: rifai render_v2.py")
    n_img, n_bytes = (0, 0) if args.no_copy_images else copy_images(entries, args.renders, args.docs_dir)
    status = "definitivo (E12 concluso)" if man["frame"].get("e12_complete") else "PROVVISORIO (E12 non concluso)"
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": args.seed,
            "study": "v2", "study_version": "v2", "id_prefix": ID_PREFIX, "domain": hs2.DOMAIN,
            "topology": hs2.TOPOLOGY, "subject_set": hs2.SUBJECT_SET,
            "n_subjects": len(subjects), "gts": list(D), "principal_gts": list(PRINCIPAL),
            "types": [f"{x}_vs_{y}" for x, y in pairs], "control_gts": control_gts, "margin": args.margin,
            "pool_factor": args.pool_factor, "max_uses": args.max_uses, "control_margin": args.control_margin,
            "n_test": sum(e["kind"] == "test" for e in entries),
            "n_control": sum(e["kind"] == "control" for e in entries),
            "n_practice": sum(e["kind"] == "practice" for e in entries), "pool_size": int(a.size),
            "pool_per_type": {k: int(v[0].size) for k, v in pools.items()}, "gt_status": status,
            "gt_frame": man["frame"], "gt_units": {g: v["units"] for g, v in man["gts"].items()},
            "camera": {k: camera.get(k) for k in ("px_per_mm", "half_mm", "tile_px", "yaws_deg", "views")},
            "n_images": n_img, "image_bytes": n_bytes}
    shown = [{k: e[k] for k in ("id", "kind", "disagreement_type", "a", "b", "c")} for e in entries]
    # Impronta di cio' che la pagina mostra: la pagina la rimanda nel payload e analyze_v2.py tiene solo le
    # sessioni fatte su QUESTE triplette (una rigenerazione con contenuto diverso cambia l'impronta).
    meta["triplets_hash"] = hashlib.sha256(json.dumps(shown, sort_keys=True).encode()).hexdigest()[:16]
    public_meta = {k: meta[k] for k in ("generated_at", "seed", "study", "study_version", "n_test", "n_control",
                                        "n_practice", "types", "gt_status", "triplets_hash")}
    public = {"meta": public_meta, "triplets": shown}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "triplets.json").write_text(json.dumps({"meta": meta, "triplets": entries}, indent=1),
                                                encoding="utf-8")
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    (args.docs_dir / "triplets.js").write_text(
        "// Triplette dello studio umano v2 (aau/human_study_v2/triplets.json senza `metrics`, che serve solo\n"
        "// all'analisi). Immagini: img/<soggetto>.jpg, striscia verticale frontale / 3/4 / profilo.\n"
        "window.WBES_TRIPLETS = " + json.dumps(public) + ";\n", encoding="utf-8")
    write_stats(args.out_dir / "triplets_stats.md", args, meta, entries, pools, list(D))
    print(f"[hs2-select] {meta['n_test']} test, {meta['n_control']} controlli, {meta['n_practice']} prova, "
          f"{n_img} immagini ({n_bytes / 1e6:.1f} MB); stato GT: {status}", flush=True)


if __name__ == "__main__":
    main()
