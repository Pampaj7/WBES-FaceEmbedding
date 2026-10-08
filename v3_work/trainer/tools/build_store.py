#!/usr/bin/env python3
"""Store in mmap degli operatori di uno split: calcolati UNA volta, letti da tutti i run (data_v3.MmapStore).

    aau/run.sh v3_work/trainer/tools/build_store.py --spec <spec.json> --split <split.json> --out <dir> \
        --prepass-proc 120 --workers 32 --shards 64

1. soggetti: train + online_eval + online_eval_extra dello split; mesh: tutte quelle della spec (viste e tar);
2. pre-pass delle mesh dei tar (v3_work/trainer/prepass_v3.py: aau/data_scale/prepass_ops.py con il build_grad
   vettorizzato di E9) in <out>/ops_tmp; le viste si leggono dove stanno;
3. ogni mesh passa dal loader congelato (GTReadyDatasetNPZ: centro e maxabs, autovalori normalizzati,
   gradienti scalati) e si scrive nel formato della cache compatta (facce e indici COO int32, niente L) in
   <out>/shards/shard_NNN.bin, array allineati a 64 byte; <out>/index.npz con nome, shard, offset e forme;
4. verifica: ``--verify`` mesh a caso lette dallo store contro il loader, tensore per tensore (devono coincidere);
5. <out>/ops_tmp si cancella (salvo --keep-ops). Ripartibile: pre-pass e shard gia' scritti si saltano.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
sys.path.insert(0, str(TRAINER))

ALIGN = 64


def load_one(path: Path):
    """GTReadyDatasetNPZ.__getitem__ su un solo file (stesso codice, nessuna copia del loader)."""
    import common  # noqa: F401
    from dataset_gtready import GTReadyDatasetNPZ
    ds = GTReadyDatasetNPZ.__new__(GTReadyDatasetNPZ)
    ds.data_dir, ds.files, ds.verbose = str(path.parent), [path.name], False
    return ds[0]


def compact_arrays(s) -> dict:
    """Array da scrivere. evecs nel layout del loader: in ordine colonna (stride (1, n), il caso di viste e
    pre-pass) si scrive evecs.T, (k, n) contiguo, con ``evecs_f`` = True (data_v3.store_shapes)."""
    gx, gy = s["gradX"].coalesce(), s["gradY"].coalesce()
    ev = s["evecs"]
    ef = ev.dim() == 2 and ev.shape[1] > 1 and ev.stride() == (1, ev.shape[0])
    return {"verts": s["verts"].numpy().astype(np.float32, copy=False),
            "faces": s["faces"].numpy().astype(np.int32),
            "mass": s["mass"].numpy().astype(np.float32, copy=False),
            "evals": s["evals"].numpy().astype(np.float32, copy=False),
            "evecs": (ev.t() if ef else ev).numpy().astype(np.float32, copy=False), "evecs_f": bool(ef),
            "gxi": gx.indices().numpy().astype(np.int32), "gxv": gx.values().numpy().astype(np.float32, copy=False),
            "gyi": gy.indices().numpy().astype(np.int32), "gyv": gy.values().numpy().astype(np.float32, copy=False)}


def build_shard(task):
    k, items, out = task
    import torch
    torch.set_num_threads(1)      # un thread per worker: con i thread di default i worker si contendono i core
    import data_v3 as dv
    path = out / "shards" / f"shard_{k:03d}.bin"
    meta_path = out / "shards" / f"shard_{k:03d}.json"
    if path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())
    tmp = path.with_suffix(".bin.tmp")
    rows = []
    with open(tmp, "wb") as fh:
        pos = 0
        for name, src in items:
            arr = compact_arrays(load_one(Path(src)))
            row = {"name": name, "n": arr["verts"].shape[0], "m": arr["faces"].shape[0], "k": arr["evals"].shape[0],
                   "nx": arr["gxv"].shape[0], "ny": arr["gyv"].shape[0], "evecs_f": arr["evecs_f"]}
            for f, dt in dv.STORE_FIELDS:
                pad = (-pos) % ALIGN
                if pad:
                    fh.write(b"\0" * pad)
                    pos += pad
                b = np.ascontiguousarray(arr[f], dtype=dt).tobytes()
                row[f"off_{f}"] = pos
                fh.write(b)
                pos += len(b)
            rows.append(row)
    os.replace(tmp, path)
    meta = {"shard": k, "bytes": pos, "rows": rows}
    meta_path.write_text(json.dumps(meta))
    return meta


def verify(out: Path, srcs: dict, n: int, seed: int = 0) -> dict:
    import torch
    import data_v3 as dv
    st = dv.MmapStore(out)
    rng = np.random.default_rng(seed)
    avail = [i for i, nm in enumerate(st.names) if Path(srcs.get(nm, "")).exists()]
    if not avail:
        return {"n_checked": 0, "note": "sorgenti non piu' presenti (ops_tmp cancellato): verifica gia' fatta"}
    pick = rng.choice(np.asarray(avail), size=min(n, len(avail)), replace=False)
    worst = 0.0
    for i in pick:
        name = st.names[int(i)]
        ref = load_one(Path(srcs[name]))
        got = dv._serve(st.compact_sample(int(i)))
        for key in ("verts", "mass", "evals", "evecs"):
            worst = max(worst, float((ref[key] - got[key]).abs().max()))
            # anche il layout: sotto TF32 lo stride cambia il kernel e quindi gli arrotondamenti
            assert ref[key].dtype == got[key].dtype and ref[key].stride() == got[key].stride(), \
                (name, key, ref[key].dtype, got[key].dtype, ref[key].stride(), got[key].stride())
        assert torch.equal(ref["faces"], got["faces"]), name
        for key in ("gradX", "gradY"):
            a, b = ref[key].coalesce(), got[key]
            assert torch.equal(a.indices(), b.indices()) and torch.equal(a.values(), b.values()), (name, key)
    return {"n_checked": int(len(pick)), "max_abs_dense": worst, "dense_strides_identical": True,
            "sparse_identical": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prepass-proc", type=int, default=64)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--verify", type=int, default=100)
    ap.add_argument("--keep-ops", action="store_true")
    ap.add_argument("--tolerate", type=int, default=0, help="mesh senza operatori ammesse (escluse dallo store)")
    a = ap.parse_args()
    import data_v3 as dv
    from common import split_name
    t0 = time.time()
    spec = json.loads(a.spec.read_text())
    split = json.loads(a.split.read_text())
    subjects = set(split["train"]) | set(split.get("online_eval", [])) | \
        {s for ids in (split.get("online_eval_extra") or {}).values() for s in ids}
    sources = dv.collect_sources(spec)
    names = sorted(n for n in sources if split_name(n)[0] in subjects)
    tar_subj = sorted({split_name(n)[0] for n in names if sources[n][0] == "tar"})
    print(f"[store] {len(subjects)} soggetti, {len(names)} mesh ({sum(sources[n][0] == 'tar' for n in names)} dai tar, "
          f"{len(tar_subj)} soggetti da pre-passare) -> {a.out}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "shards").mkdir(exist_ok=True)
    ops = a.out / "ops_tmp"
    if tar_subj and not (a.out / "index.npz").exists():
        subj_file = a.out / "prepass_subjects.txt"
        subj_file.write_text("\n".join(tar_subj) + "\n")
        tars = sorted({sources[n][1] for n in names if sources[n][0] == "tar"})
        cmd = [sys.executable, str(TRAINER / "prepass_v3.py"), "--out-dir", str(ops), "--subjects", str(subj_file),
               "--convention", spec.get("convention", "areanorm"), "--n-proc", str(a.prepass_proc), "--tars", *tars]
        if spec.get("tar_index"):
            cmd += ["--tar-index", spec["tar_index"]]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        t1 = time.time()
        with open(a.out / "prepass.log", "a") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        print(f"[store] pre-pass rc={rc} in {time.time() - t1:.0f}s", flush=True)
        if rc not in (0, 1):     # 1 = qualche mesh fallita: si contano sotto, contro --tolerate
            raise SystemExit(f"pre-pass fallito (rc={rc}), log in {a.out / 'prepass.log'}")
    srcs = {n: (sources[n][1] if sources[n][0] == "view" else str(ops / n)) for n in names}
    missing = [n for n in names if not Path(srcs[n]).exists()]
    if missing and not (a.out / "index.npz").exists():
        if len(missing) > a.tolerate:
            raise SystemExit(f"{len(missing)} mesh senza operatori > --tolerate {a.tolerate} (prima {missing[0]}), "
                             f"log in {a.out / 'prepass.log'}")
        # come --prepass-tolerate del trainer: fuori dallo store, quindi mai campionate
        print(f"[store] AVVISO: {len(missing)} mesh senza operatori, escluse: {missing[:10]}", flush=True)
        (a.out / "missing.json").write_text(json.dumps(missing, indent=1))
        names = [n for n in names if n not in set(missing)]
    if not (a.out / "index.npz").exists():
        S = int(a.shards)
        tasks = [(k, [(n, srcs[n]) for n in names[k::S]], a.out) for k in range(S)]
        t1 = time.time()
        os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            metas = list(ex.map(build_shard, tasks))
        rows = sorted((dict(r, shard=m["shard"]) for m in metas for r in m["rows"]), key=lambda r: r["name"])
        idx = {"names": np.asarray([r["name"] for r in rows]), "shard": np.asarray([r["shard"] for r in rows], np.int16)}
        for key in ("n", "m", "k", "nx", "ny"):
            idx[key] = np.asarray([r[key] for r in rows], np.int64)
        idx["evecs_f"] = np.asarray([r["evecs_f"] for r in rows], np.int8)
        for f, _ in dv.STORE_FIELDS:
            idx[f"off_{f}"] = np.asarray([r[f"off_{f}"] for r in rows], np.int64)
        np.savez(a.out / "index.tmp.npz", **idx)
        os.replace(a.out / "index.tmp.npz", a.out / "index.npz")
        gib = sum(m["bytes"] for m in metas) / 2 ** 30
        print(f"[store] {len(rows)} mesh in {S} shard, {gib:.1f} GiB, in {time.time() - t1:.0f}s", flush=True)
    check = verify(a.out, srcs, a.verify)
    print(f"[store] verifica: {check}", flush=True)
    miss_path = a.out / "missing.json"
    manifest = {"spec": str(a.spec), "split": str(a.split), "n_subjects": len(subjects),
                "n_meshes": len(np.load(a.out / "index.npz")["names"]),
                "missing": json.loads(miss_path.read_text()) if miss_path.exists() else [],
                "prepass": "prepass_v3.py (build_grad vettorizzato di E9), --convention " + spec.get("convention", "areanorm"),
                "format": "cache compatta di data_v3 (int32, niente L), loader GTReadyDatasetNPZ", "verify": check,
                "seconds": time.time() - t0, "host": os.uname().nodename}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if not a.keep_ops and ops.exists():
        import shutil
        shutil.rmtree(ops, ignore_errors=True)
    print("[store] OK", flush=True)


if __name__ == "__main__":
    main()
