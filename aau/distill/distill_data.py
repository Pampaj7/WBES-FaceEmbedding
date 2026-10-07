#!/usr/bin/env python3
"""Dati della distillazione v2: sorgenti, soggetti, mesh d'ingresso dello studente e bersagli dell'insegnante.

Unico posto in cui si decide QUALI mesh entrano (lo leggono ``teacher_labels.py`` e
``train_distill_v2.py``, cosi' le etichette dell'insegnante coprono esattamente gli ingressi dello
studente). Protocollo in ``aau/runs/distill_v2/protocol.md``.

Sorgenti (frame d'ingresso dello studente: canonico ICT, +Y alto e naso +Z):
  ``bfm``       i 392 BFM di training del congiunto + i 16 dell'eval online (validazione), le 5
                topologie senza crop; geometria ``REMESH/npz_data_topo_500``, operatori ricalcolati
                nel job con Rx(180) e facce invertite (``prepass_ops.apply_frame``, come ``zs_stage``);
  ``ict5k``     ICT-5000 del pilota (training del congiunto + 84 held-out di validazione), 5 topologie e
                ``rexpr``; operatori delle viste in uso (gia' nel frame ICT);
  ``ictscale``  i primi ``N_ICTSCALE`` ICT nuovi (``datasets/ICT_SCALE``, shard 0-39, tutti nel training
                di ``split_scale.json`` e controllati contro ``heldout_frozen``), operatori nel job;
  ``gnm``       GNM Head (``datasets/GNM_DISTILL``, ``gen_gnm_shard.py``): 10.000 di training e 100 di
                validazione, operatori nel job.
Per ``ictscale`` e ``gnm`` (training) ogni identita' entra con ``K_NEUTRAL`` topologie neutre a caso fra
le 5 senza crop e UNA espressione a caso fra le sue (``SeedSequence([SEED_SELECT, numero])``): la RAM
non contiene tutte le topologie di 20.000 identita', e la varieta' si sposta sulle identita'. I
soggetti di validazione GNM entrano con tutte le topologie e tutte le espressioni.

Bersaglio dell'insegnante: (soggetto, ``original``) per una topologia neutra, (soggetto, ``rexpr<k>``)
per un'espressione. Dominio dell'insegnante (camera e crop): ``bfm``, ``ict`` (= ``ict5k``),
``ictscale``, ``gnm``.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
DS = REPO_ROOT / "datasets"
SPLIT_JSON = AAU_DIR / "data_scale" / "split_scale.json"
FROZEN_JSON = AAU_DIR / "data_scale" / "heldout_frozen.json"
NOCROP = ("original", "remesh", "down8k", "up60k", "noisy")
ICTSCALE_SHARDS = DS / "ICT_SCALE" / "shards"
GNM_SHARDS = DS / "GNM_DISTILL" / "shards"
N_ICTSCALE = 10000
ICTSCALE_FIRST = 20000
GNM_BASE, GNM_TOTAL, GNM_VAL = 100000, 10100, 100
K_NEUTRAL = 1
SEED_SELECT = 20261013
BFM_CANON = {"bfm": {"R": [[1, 0, 0], [0, -1, 0], [0, 0, -1]], "flip_faces": True}}
PILOT_TEACHER = AAU_DIR / "runs" / "distill_pilot" / "teacher"


def num(sid: str) -> int:
    return int(sid[2:])


def source_of(sid: str) -> str:
    n = num(sid)
    if n < 1000:
        return "bfm"
    if 10000 <= n < 15000:
        return "ict5k"
    if 20000 <= n < 70000:
        return "ictscale"
    if GNM_BASE <= n < GNM_BASE + GNM_TOTAL:
        return "gnm"
    raise ValueError(f"{sid}: nessuna sorgente")


def subjects(source: str) -> tuple[list[str], list[str]]:
    """(training, validazione) di una sorgente."""
    if source in ("bfm", "ict5k"):
        dom = "bfm" if source == "bfm" else "ict"
        z = np.load(PILOT_TEACHER / dom / "teacher.npz")
        sp = dict(zip([str(s) for s in z["subjects"]], [str(x) for x in z["split"]]))
        return sorted(s for s, x in sp.items() if x == "train"), sorted(s for s, x in sp.items() if x == "val")
    if source == "ictscale":
        split = json.loads(SPLIT_JSON.read_text())
        frozen = json.loads(FROZEN_JSON.read_text())
        train = [f"id{ICTSCALE_FIRST + j}" for j in range(N_ICTSCALE)]
        bad = set(train) & (set(frozen["bfm"]) | set(frozen["ict_view"]) | set(split["heldout"]))
        if bad or not set(train) <= set(split["train"]):
            raise SystemExit(f"ictscale: soggetti fuori dal training dello split: {sorted(bad)[:5]}")
        return train, []
    if source == "gnm":
        ids = [f"id{GNM_BASE + j}" for j in range(GNM_TOTAL)]
        return ids[: GNM_TOTAL - GNM_VAL], ids[GNM_TOTAL - GNM_VAL:]
    raise ValueError(source)


def _rexpr_by_subject(members) -> dict[str, list[str]]:
    """Soggetto -> etichette ``rexpr<k>`` presenti fra i membri degli shard (una sola passata)."""
    out: dict[str, list[str]] = {}
    for m in members:
        sid, _, rest = m.partition("_GTready_")
        if rest.startswith("rexpr"):
            out.setdefault(sid, []).append(rest[:-4])
    return {s: sorted(v) for s, v in out.items()}


def tar_members(source: str) -> dict[str, list[tuple[str, int]]]:
    """Nome del membro -> lista di (tar, indice); per ``ictscale`` dall'indice dei tar."""
    out: dict[str, list] = {}
    if source == "ictscale":
        with np.load(ICTSCALE_SHARDS / "index.npz") as z:
            for i, n in enumerate(z["names"]):
                out.setdefault(str(n), []).append(("index", i))
        return out
    for t in sorted(GNM_SHARDS.glob("gnm_shard_*.tar")):
        with tarfile.open(t) as tar:
            for m in tar:
                if m.isfile() and m.name.endswith(".npz"):
                    out.setdefault(m.name, []).append((str(t), 0))
    return out


def select(source: str, sid: str, available_rexpr: list[str], val: bool) -> list[tuple[str, str]]:
    """(etichetta d'ingresso, chiave dell'insegnante) per un soggetto."""
    if source == "bfm":
        return [(t, "original") for t in NOCROP]
    if val or source == "ict5k":
        return [(t, "original") for t in NOCROP] + [(r, r) for r in available_rexpr]
    rng = np.random.default_rng(np.random.SeedSequence([SEED_SELECT, num(sid)]))
    topo = rng.choice(len(NOCROP), size=K_NEUTRAL, replace=False)
    out = [(NOCROP[int(i)], "original") for i in sorted(topo)]
    if available_rexpr:
        r = available_rexpr[int(rng.integers(len(available_rexpr)))]
        out.append((r, r))
    return out


def plan(sources) -> list[dict]:
    """Una riga per mesh d'ingresso: sid, label, key (bersaglio), split, source, origin.

    ``origin``: ("ops", path) operatori gia' calcolati; ("geom", path) geometria su disco;
    ("tar", member) geometria in uno shard.
    """
    rows = []
    for src in sources:
        train, val = subjects(src)
        members = tar_members(src) if src in ("ictscale", "gnm") else {}
        names = set(members)
        rexpr_of = _rexpr_by_subject(names)
        ict5k_rexpr = {}
        if src == "ict5k":
            z = np.load(PILOT_TEACHER / "ict" / "teacher.npz")
            for s, t in zip(z["subjects"], z["topologies"]):
                if str(t).startswith("rexpr"):
                    ict5k_rexpr.setdefault(str(s), []).append(str(t))
        for split, ids in (("train", train), ("val", val)):
            for sid in ids:
                if src in ("ictscale", "gnm"):
                    avail = rexpr_of.get(sid, [])
                else:
                    avail = sorted(ict5k_rexpr.get(sid, []))
                for label, key in select(src, sid, avail, split == "val"):
                    name = f"{sid}_GTready_{label}.npz"
                    if src == "bfm":
                        origin = ("geom", str(DS / "REMESH" / "npz_data_topo_500" / name))
                    elif src == "ict5k":
                        origin = ("ops", str(DS / "ICT" / "expressions_random_withops" / f"{sid}_rexpr_{label[5:]}.npz")
                                  if label.startswith("rexpr") else str(DS / "ICT" / "train_ready" / "npz_withops" / name))
                    else:
                        if name not in names:
                            raise SystemExit(f"{src}: membro assente {name}")
                        origin = ("tar", name)
                    rows.append({"sid": sid, "label": label, "key": key, "split": split, "source": src,
                                 "origin": origin})
    return rows


def teacher_meshes(source: str) -> list[tuple[str, str, str]]:
    """(sid, etichetta da etichettare, split) per ``ictscale`` / ``gnm``: le chiavi dei bersagli."""
    seen = {}
    for r in plan([source]):
        seen[(r["sid"], r["key"])] = r["split"]
    return sorted((s, k, sp) for (s, k), sp in seen.items())


def extract(source: str, wanted: list[str], dest: Path) -> dict[str, Path]:
    """Membri ``wanted`` degli shard di ``source`` scritti in ``dest`` (gia' presenti saltati)."""
    dest.mkdir(parents=True, exist_ok=True)
    want = set(wanted)
    out = {}
    if source == "ictscale":
        import sys
        sys.path.insert(0, str(AAU_DIR / "data_scale"))
        from cache_budget import load_index, read_member
        idx = load_index(ICTSCALE_SHARDS / "index.npz")
        handles: dict = {}
        try:
            for i, n in enumerate(idx["names"]):
                n = str(n)
                if n in want:
                    p = dest / n
                    if not p.exists():
                        p.write_bytes(read_member(idx, i, handles))
                    out[n] = p
        finally:
            for fh in handles.values():
                fh.close()
    else:
        for t in sorted(GNM_SHARDS.glob("gnm_shard_*.tar")):
            with tarfile.open(t) as tar:
                for m in tar:
                    if m.name in want:
                        p = dest / m.name
                        if not p.exists():
                            with tar.extractfile(m) as fh:
                                p.write_bytes(fh.read())
                        out[m.name] = p
    missing = want - set(out)
    if missing:
        raise SystemExit(f"{source}: {len(missing)} membri non trovati, es. {sorted(missing)[:3]}")
    return out

