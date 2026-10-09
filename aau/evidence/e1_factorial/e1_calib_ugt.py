#!/usr/bin/env python3
"""E1, C3F-UGT: GT unificata di training TARATA sulla scala della GT maxabs del run (richiesta del critic, 9 ottobre).

    aau/run.sh aau/evidence/e1_factorial/e1_calib_ugt.py build     (e1_calib_ugt.sbatch)
    aau/run.sh aau/evidence/e1_factorial/e1_calib_ugt.py check

Perche': i margini della loss (``--rank_margin 0.05`` ecc.) sono in unita' GT. La GT unificata
(datasets/UNIFIED_GT/train/gt_unified_bfm_ict_gnm.npz) ha un'altra scala, quindi C3F-UGT contro C3F mescolerebbe
il CONTENUTO della GT con la sua SCALA.

build: per ogni dominio d (BFM, ICT, GNM, dai nomi come train_steps.domain_of_name), f_d = mediana del blocco
intra-dominio della GT maxabs del run (datasets/SCALE_ALL/gt_joint_bfm_ict_gnm.npz) / mediana dello stesso
blocco della GT unificata. Mediane ESATTE sulle coppie i < j del blocco, tutti i soggetti del dominio (training
e held-out, gli stessi nei due file). Poi:
  - blocco (d, d) x f_d;
  - blocco (d1, d2) x sqrt(f_d1 * f_d2), la media geometrica: il trainer a batch monodominio non lo legge mai.
Stessi ``names`` nello stesso ordine, ``D_orig`` float32, np.savez non compresso come l'originale; json accanto
con ``global_max`` (la guardia check_gt_scale), fattori e mediane.

check: la catena di lettura del trainer, importata e non modificata (check_gt_scale, install_gt_keep_scale,
train_v2._nan_guarded_loader, dtype float64 come train_runner), come v3_work/unified_gt/check_train_gt.py:
``name_to_idx`` identico a quello della GT del run, held-out ed eval online sugli stessi indici, nessun valore
non finito, diagonale 0, simmetria e letture monodominio su campioni; in piu' il rapporto tarata/unificata = f
sui campioni di ogni blocco e le mediane per dominio di nuovo uguali a quelle della maxabs. Scrive
``check_calib.json`` con ``ok``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
D_DIR = REPO / "datasets/UNIFIED_GT/train"
UGT = D_DIR / "gt_unified_bfm_ict_gnm.npz"
OUT = D_DIR / "gt_unified_bfm_ict_gnm_calib.npz"
RUN_GT = REPO / "datasets/SCALE_ALL/gt_joint_bfm_ict_gnm.npz"
SPLIT = REPO / "aau/data_scale/split_scale_all.json"
CHUNK = 2048


def domain(name: str) -> str:
    num = int(str(name)[2:])
    if 100000 <= num < 200000:
        return "gnm"
    return "bfm" if num < 1000 else ("ict" if num >= 10000 else "flame")


def block_median(D: np.ndarray, idx: np.ndarray) -> float:
    """Mediana esatta delle coppie i < j del blocco (idx, idx), senza copiare il blocco intero."""
    n = len(idx)
    buf = np.empty(n * (n - 1) // 2, dtype=np.float32)
    pos = 0
    for s in range(0, n, CHUNK):
        rows = D[idx[s:s + CHUNK]][:, idx]
        for r in range(rows.shape[0]):
            i = s + r
            seg = rows[r, i + 1:]
            buf[pos:pos + seg.size] = seg
            pos += seg.size
    assert pos == buf.size
    if not np.isfinite(buf).all():
        raise SystemExit("valori non finiti nel blocco intra-dominio")
    k = buf.size // 2
    if buf.size % 2:
        return float(np.partition(buf, k)[k])
    part = np.partition(buf, [k - 1, k])
    return float((np.float64(part[k - 1]) + np.float64(part[k])) / 2)


def load(path: Path):
    with np.load(path) as z:
        return z["D_orig"], np.array([str(n) for n in z["names"]])


def build() -> None:
    t0 = time.time()
    U, names = load(UGT)
    M, names_m = load(RUN_GT)
    if not np.array_equal(names, names_m):
        raise SystemExit("names diversi fra GT unificata e GT del run")
    doms = np.array([domain(n) for n in names])
    idx = {d: np.flatnonzero(doms == d) for d in ("bfm", "ict", "gnm")}
    if sum(len(v) for v in idx.values()) != len(names):
        raise SystemExit("soggetti fuori da BFM/ICT/GNM")
    med_m = {d: block_median(M, ix) for d, ix in idx.items()}
    del M
    med_u = {d: block_median(U, ix) for d, ix in idx.items()}
    f = {d: med_m[d] / med_u[d] for d in idx}
    print(f"[calib] mediane maxabs {med_m}, unificata {med_u}, fattori {f} ({time.time() - t0:.0f}s)", flush=True)
    for d1, ix1 in idx.items():
        for d2, ix2 in idx.items():
            fac = np.float32(f[d1] if d1 == d2 else np.sqrt(f[d1] * f[d2]))
            for s in range(0, len(ix1), CHUNK):
                r = ix1[s:s + CHUNK]
                U[np.ix_(r, ix2)] *= fac
    gmax = float(np.nanmax(U))
    tmp = OUT.with_name(".calib.tmp.npz")
    np.savez(tmp, D_orig=U, names=names)
    tmp.replace(OUT)
    src = json.loads(UGT.with_suffix(".json").read_text())
    man = {"scale": "unified_calibrated_to_maxabs_medians", "global_max": gmax, "n_total": int(len(names)),
           "n_by_domain": {d: int(len(ix)) for d, ix in idx.items()},
           "source_unified": str(UGT), "source_maxabs": str(RUN_GT),
           "median_maxabs": med_m, "median_unified": med_u, "factor": f,
           "cross_domain_factor": "sqrt(f_d1 * f_d2) (media geometrica; il trainer a batch monodominio non le legge)",
           "mm_per_unit_by_domain": {d: src["mm_per_unit"] / f[d] for d in idx},
           "definition": "blocco (d, d) della GT unificata x f_d, f_d = mediana(maxabs, d) / mediana(unificata, d), "
                         "mediane esatte sulle coppie i < j; names e ordine identici",
           "built_by": "aau/evidence/e1_factorial/e1_calib_ugt.py", "seconds": time.time() - t0}
    OUT.with_suffix(".json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man, indent=1), flush=True)


def check() -> None:
    for p in (REPO / "v2_work/fastio", REPO / "v2_work/train_v2", REPO / "face_embedding/gt_encdec/remeshing/intrinsic"):
        sys.path.insert(0, str(p))
    import train_steps
    import robustness.train_runner as tr
    import train_v2
    from intrinsic_utils import SUBJECT_RE_ANY, extract_subject_id

    out = {"file": str(OUT)}
    train_steps.check_gt_scale(["--dist_npz", str(OUT)], True)
    train_steps.install_gt_keep_scale()
    loader = train_v2._nan_guarded_loader(tr.load_gt_distance_matrix)
    D, name_to_idx = loader(str(OUT), dtype=np.float64)
    with np.load(RUN_GT) as z:
        ref = {}
        for i, n in enumerate(z["names"]):
            sid = extract_subject_id(str(n), subject_re=SUBJECT_RE_ANY)
            if sid is not None:
                ref[sid] = i
    out["name_to_idx_identical_to_run_gt"] = name_to_idx == ref
    split = json.loads(SPLIT.read_text())
    groups = {"train": split["train"], "heldout": split["heldout"], "online_eval": split["online_eval"],
              "online_eval_extra_gnm": split["online_eval_extra"]["gnm"]}
    out["index_match"] = {g: {"n": len(v), "all_same_index": all(name_to_idx.get(s) == ref.get(s) for s in v)}
                          for g, v in groups.items()}
    raw = np.asarray(D)
    out["n_nonfinite"] = int((~np.isfinite(raw)).sum())
    out["diag_max_abs"] = float(np.abs(np.diag(raw)).max())
    rng = np.random.default_rng(1234)
    i, j = rng.integers(0, len(raw), 200000), rng.integers(0, len(raw), 200000)
    out["symmetry_max_abs_sample"] = float(np.abs(raw[i, j] - raw[j, i]).max())
    man = json.loads(OUT.with_suffix(".json").read_text())
    with np.load(UGT) as z:
        U = z["D_orig"]
    names = np.array(sorted(ref, key=ref.get))
    doms = np.array([domain(n) for n in names])
    ratio, meds = {}, {}
    for d1 in ("bfm", "ict", "gnm"):
        a = np.flatnonzero(doms == d1)
        meds[d1] = {"calib_sample": float(np.median(_offdiag_sample(raw, a, rng))),
                    "maxabs_exact": man["median_maxabs"][d1]}
        for d2 in ("bfm", "ict", "gnm"):
            b = np.flatnonzero(doms == d2)
            r, c = rng.choice(a, 5000), rng.choice(b, 5000)
            keep = r != c
            want = man["factor"][d1] if d1 == d2 else np.sqrt(man["factor"][d1] * man["factor"][d2])
            got = raw[r[keep], c[keep]] / U[r[keep], c[keep]].astype(np.float64)
            ratio[f"{d1}-{d2}"] = float(np.abs(got / want - 1).max())
    out["ratio_calib_over_unified_vs_factor_max_rel"] = ratio
    out["median_by_domain"] = meds
    for d1, d2 in (("bfm", "bfm"), ("ict", "ict"), ("gnm", "gnm"), ("bfm", "gnm")):
        a, b = np.flatnonzero(doms == d1)[:50], np.flatnonzero(doms == d2)[:50]
        _ = D[np.ix_(a, b)]
    out["guarded_reads"] = "blocchi monodominio e fra domini letti senza errori"
    out["ok"] = bool(out["name_to_idx_identical_to_run_gt"] and out["n_nonfinite"] == 0 and out["diag_max_abs"] == 0
                     and out["symmetry_max_abs_sample"] == 0
                     and all(v["all_same_index"] for v in out["index_match"].values())
                     and max(ratio.values()) < 1e-6
                     and all(abs(m["calib_sample"] / m["maxabs_exact"] - 1) < 0.02 for m in meds.values()))
    (D_DIR / "check_calib.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)
    if not out["ok"]:
        raise SystemExit("[calib-check] FALLITO")


def _offdiag_sample(raw: np.ndarray, a: np.ndarray, rng) -> np.ndarray:
    r, c = rng.choice(a, 2_000_000), rng.choice(a, 2_000_000)
    keep = r != c
    return raw[r[keep], c[keep]]


if __name__ == "__main__":
    {"build": build, "check": check}[sys.argv[1]]()
