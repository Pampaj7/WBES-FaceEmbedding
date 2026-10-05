#!/usr/bin/env python3
"""Token di taglia (ablazioni B ed E): log del raggio rms della mesh GREZZA, per file.

Il raggio e' quello del frame rms (v2_work/pointnet/frames.py), ma calcolato sulle coordinate
grezze, prima di qualunque normalizzazione: centroide e raggio pesati con le aree vertice
(un terzo delle aree delle facce incidenti, come pp3d.vertex_areas). Il frame rms dell'input
cancella proprio questa scala; il token la restituisce al modello come un solo numero,
senza il rumore del divisore maxabs, che sul crop vale quanto la variabilita' di taglia fra
soggetti (BOARD_DIARY, diagnostica H1).

Il token che il modello vede e' standardizzato, (log r - media) / std, con media e std sulle
mesh (tutte e 6 le topologie) dei soggetti di TRAINING:
  - BFM/REMESH: lo split del trainer (rebuild_subject_split, eval_fraction 0.2, seed del run),
    sui soggetti che hanno anche una riga nella matrice GT;
  - ICT (solo zero-shot): i 4500 soggetti di train_ready/split_train.txt. Le coordinate grezze
    ICT non sono nelle unita' BFM (BFM ~1e5), quindi la standardizzazione e' per collezione,
    come la scala del pozzo nel pilota pot (calib_ict.json). Nessun held-out entra nelle stats.

    aau/run.sh aau/models/size_token.py bfm --raw-dir datasets/REMESH/npz_data_topo_500 \
        --dist-npz <gt> --seed 1234 --out aau/runs/ablations_v3/size_token_bfm_s1234.json
    aau/run.sh aau/models/size_token.py ict --raw-dir datasets/ICT/topo --id-offset 10000 \
        --train-list datasets/ICT/train_ready/split_train.txt --out .../size_token_ict.json

(per ICT la cartella mesh-only datasets/ICT/topo, 5.6 GB, invece della vista train_ready: i suoi
npz con gli operatori pesano ~300 GB e leggerne solo i vertici su CephFS costa decine di minuti)

Scrive anche <out>.check.md: per topologia, la differenza di token rispetto all'original dello
stesso soggetto (deve essere ~0 a meno della differenza reale di raggio; sul crop no).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

NAME_RE = re.compile(r"^(id\d+)_GTready_([a-z0-9]+)$")
TOPOLOGIES = ("original", "remesh", "crop", "noisy", "down8k", "up60k")


def vertex_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    tri = V[F]
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return np.bincount(F.reshape(-1), weights=np.repeat(a / 3.0, 3), minlength=len(V))


def log_rms_radius(V: np.ndarray, F: np.ndarray) -> float:
    """log del raggio rms pesato per area attorno al centroide pesato per area."""
    V = np.asarray(V, dtype=np.float64)
    w = vertex_areas(V, np.asarray(F, dtype=np.int64))
    w = w / w.sum()
    c = (w[:, None] * V).sum(0)
    return 0.5 * math.log(float((w * ((V - c) ** 2).sum(1)).sum()))


ICT_RAW_RE = re.compile(r"^ict(\d+)_GTready_([a-z0-9]+)$")


def read_one(path: Path, id_offset: int | None = None) -> tuple[str, float]:
    """(nome, log r). Con id_offset i nomi grezzi ICT ict<NNNN> diventano id<NNNN+offset>, quelli
    della vista train_ready / eval_view_heldout (manifest.json, id_offset 10000)."""
    name = path.name[:-len(".npz")]
    if id_offset is not None:
        m = ICT_RAW_RE.match(name)
        if m is None:
            raise ValueError(f"nome grezzo ICT inatteso: {name}")
        name = f"id{int(m.group(1)) + id_offset}_GTready_{m.group(2)}"
    with np.load(path, allow_pickle=False) as z:
        V = z["verts"] if "verts" in z.files else z["V"]
        F = z["faces"] if "faces" in z.files else z["F"]
        return name, log_rms_radius(V, F)


def split_name(name: str) -> tuple[str, str]:
    m = NAME_RE.match(name)
    if m is None:
        raise ValueError(f"nome inatteso: {name}")
    return m.group(1), m.group(2)


def bfm_train_subjects(names: list[str], dist_npz: Path, seed: int) -> list[str]:
    """Esattamente lo split del trainer (train_runner.py:1557)."""
    for p in ("face_embedding/gt_encdec/remeshing/intrinsic", "face_embedding/gt_encdec/autoencoder",
              "diffusion-net/src"):
        sys.path.insert(0, str(REPO_ROOT / p))
    from intrinsic_utils import SUBJECT_RE_ANY, build_subject_map, load_gt_distance_matrix  # noqa: E402
    from robustness.data_utils import rebuild_subject_split  # noqa: E402
    subj_map = build_subject_map([n + ".npz" for n in names], subject_re=SUBJECT_RE_ANY)
    _, name_to_idx = load_gt_distance_matrix(str(dist_npz), dtype=np.float64)
    subjects = sorted(s for s in subj_map if s in name_to_idx)
    train, _ = rebuild_subject_split(subjects=subjects, eval_fraction=0.2, seed=seed, max_subjects=0)
    return train


def consistency(log_r: dict[str, float], std: float, subjects: list[str] | None = None) -> list[dict]:
    """Per topologia: log r(topo) - log r(original) dello stesso soggetto."""
    by = {}
    for name, v in log_r.items():
        s, t = split_name(name)
        by.setdefault(s, {})[t] = v
    subs = sorted(by) if subjects is None else [s for s in subjects if s in by]
    orig_between = float(np.std([by[s]["original"] for s in subs if "original" in by[s]]))
    rows = []
    for t in TOPOLOGIES:
        if t == "original":
            continue
        d = np.array([by[s][t] - by[s]["original"] for s in subs if t in by[s] and "original" in by[s]])
        if d.size == 0:
            continue
        rows.append({"topology": t, "n": int(d.size), "mean": float(d.mean()), "std": float(d.std()),
                     "max_abs": float(np.abs(d).max()), "ratio_mean": float(np.exp(d.mean())),
                     "mean_in_token_std": float(d.mean() / std),
                     "max_abs_in_token_std": float(np.abs(d).max() / std),
                     "between_subject_std_original": orig_between})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("collection", choices=["bfm", "ict"])
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dist-npz", type=Path, default=None, help="bfm: matrice GT dello split")
    ap.add_argument("--seed", type=int, default=1234, help="bfm: seed del training")
    ap.add_argument("--train-list", type=Path, default=None, help="ict: un id per riga")
    ap.add_argument("--id-offset", type=int, default=None,
                    help="ict: la raw dir ha nomi ict<NNNN> (datasets/ICT/topo), mappati a id<NNNN+offset>")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    files = sorted(p for p in args.raw_dir.iterdir() if p.suffix == ".npz" and not p.name.startswith("."))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        log_r = dict(ex.map(lambda p: read_one(p, args.id_offset), files))
    names = sorted(log_r)
    print(f"[token] {len(names)} mesh da {args.raw_dir}", flush=True)

    if args.collection == "bfm":
        if args.dist_npz is None:
            raise SystemExit("bfm vuole --dist-npz")
        train = bfm_train_subjects(names, args.dist_npz, args.seed)
        rule = (f"rebuild_subject_split(eval_fraction=0.2, seed={args.seed}, max_subjects=0) sui "
                f"soggetti di {args.raw_dir.name} con riga in {args.dist_npz.name}: training")
    else:
        if args.train_list is None:
            raise SystemExit("ict vuole --train-list")
        train = sorted(l.strip() for l in args.train_list.read_text().splitlines() if l.strip())
        rule = f"soggetti di {args.train_list} (training ICT, nessun held-out)"
    tset = set(train)
    vals = np.array([v for n, v in log_r.items() if split_name(n)[0] in tset])
    if vals.size == 0:
        raise SystemExit("nessuna mesh dei soggetti di training")
    mean, std = float(vals.mean()), float(vals.std())
    payload = {"collection": args.collection, "source_dir": str(args.raw_dir.resolve()),
               "definition": "log rms radius, area-weighted, raw coordinates",
               "train": {"rule": rule, "seed": args.seed if args.collection == "bfm" else None,
                         "n_subjects": len(tset), "n_meshes": int(vals.size), "mean": mean, "std": std},
               "log_r": log_r}
    rows = consistency(log_r, std)
    payload["consistency_vs_original"] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    lines = [f"# Token di taglia, {args.collection}: consistenza fra topologie dello stesso soggetto\n",
             f"Sorgente `{args.raw_dir}`, {len(names)} mesh. Standardizzazione: {rule}; "
             f"{int(vals.size)} mesh, media {mean:.5f}, std {std:.5f}.\n",
             f"Std fra soggetti del log r sull'original: {rows[0]['between_subject_std_original']:.5f}.\n",
             "| topologia | n | media Δlog r | std Δlog r | max abs | rapporto raggi | media in std token | max abs in std token |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['topology']} | {r['n']} | {r['mean']:+.5f} | {r['std']:.5f} | {r['max_abs']:.5f} | "
                     f"{r['ratio_mean']:.4f} | {r['mean_in_token_std']:+.3f} | {r['max_abs_in_token_std']:.3f} |")
    md = "\n".join(lines) + "\n"
    Path(str(args.out) + ".check.md").write_text(md)
    print(md)
    print(f"[token] scritto {args.out}")


if __name__ == "__main__":
    main()
