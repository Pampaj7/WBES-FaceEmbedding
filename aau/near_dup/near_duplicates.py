#!/usr/bin/env python3
"""Quasi-duplicati fra test e training: per ogni soggetto di test, la distanza GT dal soggetto di training piu' vicino.

    srun -p cpu -c 8 --mem=32G -t 00:30:00 env AAU_NV= aau/run.sh aau/near_dup/near_duplicates.py
    (near_dup.sbatch)

Punto 9 del rebuttal (``paper/REBUTTAL_PLAN.md``): le identita' di test sono davvero distinte da
quelle di training? Due split:

- ``neurips``: lo split del paper, BFM 400 di training e 100 di test (``bfm_only`` di
  ``aau/runs/ws2_cross3dmm/splits.json``, = ``rebuild_subject_split`` seme 1234 sui 500 soggetti,
  verificato da ``aau/data_scale/freeze_heldout.py``; il test coincide con gli held-out di WS1);
- ``joint``: il congiunto ``x3dmm_joint_bfm_ict_s1234_1019532`` (``joint`` dello stesso file; i suoi
  held-out sono quelli congelati in ``aau/data_scale/heldout_frozen.json``, controllato qui).
  BFM e ICT hanno template diversi: il vicino si cerca solo dentro lo stesso 3DMM.

Metrica PRIMARIA: la stessa della guardia della valanga (``gen_ict_shard.guard_duplicates``),
vertex-mean-L2 fra le ``original`` normalizzate maxabs (centro sulla media, divisione per il max
|coordinata|), cosi' la soglia 0.0055 (meta' del vicino piu' prossimo minimo di ICT-5000) si applica
nelle sue unita'. BFM: calcolata qui dalle mesh di ``datasets/REMESH/npz_data_topo_500``. ICT:
``D_orig * normalization_scale.maxabs`` di ``datasets/ICT/gt/ict_matrix_distances_maxabs.npz``, come
``freeze_heldout.py``; verificata qui contro il calcolo diretto su alcune coppie.

Metrica SECONDARIA: la GT con cui il run e' stato addestrato (``normalized_matrix_distances.npz`` per
``neurips``, ``datasets/JOINT_BFM_ICT/gt_matrix.npz`` per ``joint``), letta con
``load_gt_distance_matrix`` (normalizzata al massimo). La soglia 0.0055 non e' nelle sue unita': si
riporta la regola che l'ha generata (meta' del vicino piu' prossimo minimo dentro il pool del 3DMM).

Confronti per ogni (split, 3DMM, metrica):
- NN test -> training (il numero chiesto);
- NN training -> training (lascia-uno-fuori): il vicino piu' prossimo fra identita' indipendenti con
  un pool della stessa taglia, il termine di paragone giusto per la prima riga;
- NN test -> test e la distribuzione di tutte le distanze test-test.

Scrive in ``aau/runs/near_duplicates/`` ``near_duplicates.csv`` (una riga per soggetto di test e metrica),
``stats.csv`` e ``results.md``; ``summary.md`` (protocollo e lettura, scritti prima) si compone a mano.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"))
from intrinsic_utils import SUBJECT_RE_ANY, load_gt_distance_matrix  # noqa: E402

DS = REPO_ROOT / "datasets"
RUNS = AAU_DIR / "runs"
ICT_OFFSET = 10000
AVALANCHE_THRESHOLD = 0.005529044838528674   # aau/data_scale/heldout_ict_originals.npz, dup_threshold
GT_NEURIPS = (REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder" / "latent_analysis"
              / "gt_distance_matrix" / "normalized_matrix_distances.npz")
GT_JOINT = DS / "JOINT_BFM_ICT" / "gt_matrix.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=RUNS / "near_duplicates")
    p.add_argument("--splits", type=Path, default=RUNS / "ws2_cross3dmm" / "splits.json")
    p.add_argument("--frozen", type=Path, default=AAU_DIR / "data_scale" / "heldout_frozen.json")
    return p.parse_args()


def is_bfm(s: str) -> bool:
    return int(s[2:]) < 1000


def ict_raw(s: str) -> str:
    return f"ict{int(s[2:]) - ICT_OFFSET:04d}"


def maxabs(V: np.ndarray) -> np.ndarray:
    """Come ``gen_ict_shard.maxabs``."""
    Vc = V - V.mean(0, keepdims=True)
    return Vc / max(float(np.abs(Vc).max()), 1e-9)


def vertex_mean_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(len(A), len(B)) di mean_v ||a_v - b_v||, come la guardia (blocchi per riga)."""
    out = np.empty((len(A), len(B)))
    for i, a in enumerate(A):
        out[i] = np.linalg.norm(B - a[None], axis=-1).mean(-1)
    return out


def bfm_maxabs_matrix(names: list[str]) -> np.ndarray:
    V = np.stack([maxabs(np.load(DS / "REMESH" / "npz_data_topo_500" / f"{s}_GTready_original.npz")["V"]
                         .astype(np.float64)) for s in names]).astype(np.float32)
    return vertex_mean_l2(V, V)


def ict_maxabs_matrix(names: list[str]) -> np.ndarray:
    """Dalla matrice precalcolata, con un controllo diretto su 3 soggetti."""
    with np.load(DS / "ICT" / "gt" / "ict_matrix_distances_maxabs.npz") as z:
        pos = {str(n): i for i, n in enumerate(z["names"])}
        idx = np.asarray([pos[ict_raw(s)] for s in names])
        scale = float(json.loads((DS / "ICT" / "gt" / "manifest.json").read_text())["normalization_scale"]["maxabs"])
        D = z["D_orig"].astype(np.float64)[np.ix_(idx, idx)] * scale
    probe = names[:3]
    V = np.stack([maxabs(np.load(DS / "ICT" / "topo" / f"{ict_raw(s)}_GTready_original.npz")["V"].astype(np.float64))
                  for s in probe])
    direct = vertex_mean_l2(V, V)
    err = float(np.abs(direct - D[:3, :3]).max())
    print(f"[near-dup] ICT: precalcolata vs diretta su {probe}: max |diff| {err:.2e} "
          f"(valori {np.round(D[0, 1:3], 5).tolist()})", flush=True)
    if err > 1e-4:
        raise SystemExit("la matrice ICT precalcolata non e' la vertex-mean-L2 maxabs")
    return D


def gt_matrix(path: Path, names: list[str]) -> np.ndarray:
    D, pos = load_gt_distance_matrix(str(path), subject_re=SUBJECT_RE_ANY, dtype=np.float64)
    missing = [s for s in names if s not in pos]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} soggetti assenti ({missing[:3]})")
    idx = np.asarray([pos[s] for s in names])
    return np.asarray(D)[np.ix_(idx, idx)]


def describe(x: np.ndarray) -> dict:
    return {"n": len(x), "min": float(x.min()), "p1": float(np.percentile(x, 1)), "p5": float(np.percentile(x, 5)),
            "median": float(np.median(x)), "max": float(x.max())}


def analyse(split: str, family: str, metric: str, D: np.ndarray, names: list[str], train: set, test: set,
            threshold: float | None) -> tuple[list[dict], dict]:
    names = np.asarray(names)
    tr = np.flatnonzero(np.isin(names, sorted(train)))
    te = np.flatnonzero(np.isin(names, sorted(test)))
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    if not np.isfinite(D[np.ix_(te, tr)]).all():
        raise SystemExit(f"{split}/{family}/{metric}: distanze non finite")
    nn_test_train = D[np.ix_(te, tr)].min(1)
    arg = tr[D[np.ix_(te, tr)].argmin(1)]
    nn_train_train = D[np.ix_(tr, tr)].min(1)
    nn_test_test = D[np.ix_(te, te)].min(1)
    iu = np.triu_indices(len(te), 1)
    test_pairs = D[np.ix_(te, te)][iu]
    pool_rule = 0.5 * float(min(nn_train_train.min(), nn_test_test.min(), nn_test_train.min()))
    thr = threshold if threshold is not None else pool_rule
    rows = [{"split": split, "family": family, "metric": metric, "subject": names[i], "nn_train": float(d),
             "nn_train_subject": names[j], "nn_test": float(t), "below_threshold": bool(d <= thr)}
            for i, d, j, t in zip(te, nn_test_train, arg, nn_test_test)]
    stat = {"split": split, "family": family, "metric": metric, "n_train": len(tr), "n_test": len(te),
            "threshold": thr, "threshold_rule": "valanga 0.0055" if threshold is not None else "meta' NN minimo del pool",
            "pool_rule_threshold": pool_rule,
            "n_test_below": int((nn_test_train <= thr).sum()),
            "ratio_min_nn_to_threshold": float(nn_test_train.min() / thr),
            "n_test_nn_train_below_p1_test_pairs": int((nn_test_train < np.percentile(test_pairs, 1)).sum()),
            "median_ratio_nn_train_over_nn_test": float(np.median(nn_test_train / nn_test_test))}
    for tag, x in (("nn_test_train", nn_test_train), ("nn_train_train", nn_train_train),
                   ("nn_test_test", nn_test_test), ("test_pairs", test_pairs)):
        stat.update({f"{tag}_{k}": v for k, v in describe(x).items()})
    return rows, stat


def main() -> None:
    args = parse_args()
    splits = json.loads(args.splits.read_text())["models"]
    frozen = json.loads(args.frozen.read_text())
    joint_test = set(splits["joint"]["heldout"])
    if sorted(s for s in joint_test if is_bfm(s)) != sorted(frozen["bfm"]) or \
            sorted(s for s in joint_test if not is_bfm(s)) != sorted(frozen["ict_view"]):
        raise SystemExit("held-out del congiunto diversi da heldout_frozen.json")
    plan = {
        "neurips": (set(splits["bfm_only"]["train"]), set(splits["bfm_only"]["heldout"]), GT_NEURIPS),
        "joint": (set(splits["joint"]["train"]), joint_test, GT_JOINT),
    }
    for split, (train, test, _) in plan.items():
        if train & test:
            raise SystemExit(f"{split}: {len(train & test)} soggetti sia in training sia in test")
        print(f"[near-dup] {split}: training {len(train)} (BFM {sum(map(is_bfm, train))}), "
              f"test {len(test)} (BFM {sum(map(is_bfm, test))})", flush=True)

    # Matrici primarie, una per 3DMM, sull'unione dei soggetti che servono.
    bfm_names = sorted({s for tr, te, _ in plan.values() for s in tr | te if is_bfm(s)})
    ict_names = sorted({s for tr, te, _ in plan.values() for s in tr | te if not is_bfm(s)}, key=lambda s: int(s[2:]))
    print(f"[near-dup] BFM: {len(bfm_names)} soggetti, ICT: {len(ict_names)}", flush=True)
    prim = {"bfm": (bfm_maxabs_matrix(bfm_names), bfm_names), "ict": (ict_maxabs_matrix(ict_names), ict_names)}

    rows, stats, checks = [], [], []
    for split, (train, test, gt_path) in plan.items():
        for family, (D, names) in prim.items():
            fam_train = {s for s in train if is_bfm(s) == (family == "bfm")}
            fam_test = {s for s in test if is_bfm(s) == (family == "bfm")}
            if not fam_test:
                continue
            r, s = analyse(split, family, "vertex_mean_l2_maxabs", D, names, fam_train, fam_test, AVALANCHE_THRESHOLD)
            rows += r
            stats.append(s)
            sub = sorted(fam_train | fam_test, key=lambda x: int(x[2:]))
            G = gt_matrix(gt_path, sub)
            r, s = analyse(split, family, f"gt_training ({gt_path.name})", G, sub, fam_train, fam_test, None)
            rows += r
            stats.append(s)
            # Le due metriche ordinano allo stesso modo le coppie test-training?
            pos = {n: i for i, n in enumerate(names)}
            Dp = D[np.ix_([pos[x] for x in sub], [pos[x] for x in sub])]
            iu = np.triu_indices(len(sub), 1)
            rho = pd.Series(Dp[iu]).corr(pd.Series(G[iu]), method="spearman")
            checks.append(f"- {split} / {family}: Spearman fra vertex-mean-L2 maxabs e GT di training su "
                          f"{len(iu[0])} coppie = {rho:.3f}")
            print(checks[-1], flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "near_duplicates.csv", index=False)
    st = pd.DataFrame(stats)
    st.to_csv(args.out_dir / "stats.csv", index=False)

    def f(x: float) -> str:
        return f"{x:.4f}" if abs(x) >= 1e-3 else f"{x:.2e}"

    lines = ["| split | 3DMM | metrica | train / test | soglia (regola) | test sotto soglia | NN test->train: min / p5 / mediana "
             "| NN train->train (lascia-uno-fuori): min / p5 / mediana | NN test->test: min / mediana "
             "| distanze test-test: p1 / mediana | min NN test->train / soglia |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, s in st.iterrows():
        lines.append(f"| {s['split']} | {s['family'].upper()} | {s['metric']} | {s['n_train']} / {s['n_test']} | "
                     f"{f(s['threshold'])} ({s['threshold_rule']}) | {s['n_test_below']} | "
                     f"{f(s['nn_test_train_min'])} / {f(s['nn_test_train_p5'])} / {f(s['nn_test_train_median'])} | "
                     f"{f(s['nn_train_train_min'])} / {f(s['nn_train_train_p5'])} / {f(s['nn_train_train_median'])} | "
                     f"{f(s['nn_test_test_min'])} / {f(s['nn_test_test_median'])} | "
                     f"{f(s['test_pairs_p1'])} / {f(s['test_pairs_median'])} | {s['ratio_min_nn_to_threshold']:.1f}x |")
    (args.out_dir / "results.md").write_text("\n".join(lines + ["", "Controlli:", *checks]) + "\n", encoding="utf-8")
    print("\n".join(lines + checks), flush=True)


if __name__ == "__main__":
    main()
