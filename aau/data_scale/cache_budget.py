#!/usr/bin/env python3
"""Indice dei tar degli shard e memoria ESATTA della cache di un blocco (revisione del critic, 6 ottobre).

    aau/run.sh aau/data_scale/cache_budget.py index  --shards-dir datasets/ICT_SCALE/shards --n-proc 16
    aau/run.sh aau/data_scale/cache_budget.py blocks --spec <spec.json> --split aau/data_scale/split_scale.json

``index``: una sola scansione dei 200 tar (a freddo su CephFS ~34 min se la si rifa' a ogni blocco,
perche' tarfile legge 3.500 header sparsi per tar). Per ogni membro: tar, offset dei dati, byte, e i
conteggi della mesh (vertici n, facce m, spigoli unici E). Scritto in ``<shards-dir>/index.npz``.
Prepass e trainer leggono i membri per offset invece di riscorrere i tar.

Byte di un campione nella cache (``fast_data._sample_bytes`` sul campione del loader congelato):
    verts float32 12n + faces int64 24m + mass float32 4n + evals float32 4k + evecs float32 4nk
    + per L, gradX, gradY: indici int64 (2 x nnz x 8) + valori float32 (nnz x 4) = 20 nnz ciascuno.
Per un file di operatori che esiste, n, m, k e nnz si leggono dagli header npy (``npz_sample_bytes``,
esatto). Per una mesh nuova, prima del pre-pass: k = 128 e nnz(L) = nnz(gradX) = nnz(gradY) = n + 2E
(Laplaciano cotangente: diagonale piu' le due direzioni di ogni spigolo; gradX/gradY hanno la stessa
sparsita', misurato su 24 mesh in measure_npz.json). La formula e' confrontata con gli header reali
nello smoke (``[steps] ... previsione dall'indice``).

``blocks``: la stessa partizione in blocchi di train_steps.py (``partition_blocks``) e la memoria
esatta della cache di ognuno: viste dagli header dei file, mesh nuove dall'indice.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

K_EIG = 128
SPARSE = ("L", "gradX", "gradY")


def bytes_from_counts(n: int, m: int, k: int, nnz: tuple[int, int, int]) -> int:
    return 12 * n + 24 * m + 4 * n + 4 * k + 4 * n * k + sum(20 * z for z in nnz)


def npz_shapes(path: Path) -> dict[str, tuple]:
    """Forme degli array di un npz dagli header, senza leggere i dati."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            with zf.open(name) as fh:
                version = np.lib.format.read_magic(fh)
                read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                        else np.lib.format.read_array_header_2_0)
                shape, _, _ = read(fh)
            out[name[:-4] if name.endswith(".npy") else name] = shape
    return out


def npz_sample_bytes(path: Path) -> int:
    s = npz_shapes(path)
    n, m, k = s["verts"][0], s["faces"][0], s["evals"][0]
    return bytes_from_counts(n, m, k, tuple(s[f"{b}_values"][0] for b in SPARSE))


def predicted_bytes(n: int, m: int, E: int) -> int:
    nnz = n + 2 * E
    return bytes_from_counts(n, m, min(K_EIG, n - 2), (nnz, nnz, nnz))


# --- indice dei tar ----------------------------------------------------------------------------

def _index_one(tar_path: str) -> dict:
    names, offs, sizes, ns, ms, es = [], [], [], [], [], []
    with tarfile.open(tar_path) as tar:
        for mem in tar:
            if not mem.isfile() or not mem.name.endswith(".npz"):
                continue
            with np.load(io.BytesIO(tar.extractfile(mem).read())) as z:
                V, F = z["V"], z["F"]
            e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
            E = len(np.unique(e[:, 0].astype(np.int64) * (int(F.max()) + 1) + e[:, 1]))
            names.append(mem.name); offs.append(mem.offset_data); sizes.append(mem.size)
            ns.append(len(V)); ms.append(len(F)); es.append(E)
    return {"tar": tar_path, "names": names, "offs": offs, "sizes": sizes, "n": ns, "m": ms, "E": es}


def build_index(shards_dir: Path, n_proc: int) -> Path:
    tars = sorted(str(p) for p in shards_dir.glob("shard_*.tar"))
    with ProcessPoolExecutor(n_proc) as ex:
        parts = list(ex.map(_index_one, tars))
    tar_id = np.concatenate([np.full(len(p["names"]), i, dtype=np.int16) for i, p in enumerate(parts)])
    cat = lambda k, dt: np.concatenate([np.asarray(p[k], dtype=dt) for p in parts])  # noqa: E731
    out = shards_dir / "index.npz"
    tmp = out.with_name(".index.tmp.npz")
    np.savez(tmp, tars=np.array([Path(p["tar"]).name for p in parts]), tar_id=tar_id,
             names=np.array([n for p in parts for n in p["names"]]),
             offset=cat("offs", np.int64), size=cat("sizes", np.int64),
             n=cat("n", np.int32), m=cat("m", np.int32), E=cat("E", np.int32))
    tmp.replace(out)
    return out


def load_index(path: Path) -> dict:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d["dir"] = path.parent
    return d


def read_member(index: dict, i: int, handles: dict) -> bytes:
    tar = str(index["dir"] / str(index["tars"][int(index["tar_id"][i])]))
    fh = handles.get(tar)
    if fh is None:
        fh = handles[tar] = open(tar, "rb")
    fh.seek(int(index["offset"][i]))
    return fh.read(int(index["size"][i]))


# --- memoria dei blocchi ----------------------------------------------------------------------

def blocks_report(spec_path: Path, split_path: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v2_work/fastio"))
    from train_steps import _split_name, collect_sources, partition_blocks

    spec = json.loads(spec_path.read_text())
    split = json.loads(split_path.read_text())
    sources = collect_sources(spec)
    have = {_split_name(n)[0] for n in sources}
    train = sorted(s for s in split["train"] if s in have)
    blocks = partition_blocks(train, spec)
    index = load_index(Path(spec["tar_index"])) if spec.get("tar_index") else None
    pos = {str(n): i for i, n in enumerate(index["names"])} if index is not None else {}
    by_sid: dict[str, list[str]] = {}
    for n in sources:
        by_sid.setdefault(_split_name(n)[0], []).append(n)
    cache: dict[str, int] = {}

    def sample_bytes(name: str) -> int:
        if name not in cache:
            kind, src = sources[name]
            if kind == "view":
                cache[name] = npz_sample_bytes(Path(src))
            else:
                i = pos[name]
                cache[name] = predicted_bytes(int(index["n"][i]), int(index["m"][i]), int(index["E"][i]))
        return cache[name]

    # header delle viste (operatori gia' calcolati, su CephFS) in parallelo: CephFS scala coi lettori
    from concurrent.futures import ThreadPoolExecutor
    views = sorted({n for b in blocks for s in b for n in by_sid[s] if sources[n][0] == "view"})
    with ThreadPoolExecutor(16) as ex:
        for n, v in zip(views, ex.map(lambda n: npz_sample_bytes(Path(sources[n][1])), views)):
            cache[n] = v
    per_block = []
    for k, b in enumerate(blocks):
        names = [n for s in b for n in by_sid[s]]
        tot = sum(sample_bytes(n) for n in names)
        per_block.append({"block": k, "subjects": len(b), "samples": len(names), "gib": tot / 2 ** 30})
        print(f"blocco {k:02d}: {len(b)} soggetti, {len(names)} campioni, {tot / 2 ** 30:.1f} GiB", flush=True)
    g = [p["gib"] for p in per_block]
    return {"n_blocks": len(blocks), "max_gib": max(g), "min_gib": min(g), "mean_gib": float(np.mean(g)),
            "argmax": int(np.argmax(g)), "blocks": per_block}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("index")
    a1.add_argument("--shards-dir", type=Path, required=True)
    a1.add_argument("--n-proc", type=int, default=16)
    a2 = sub.add_parser("blocks")
    a2.add_argument("--spec", type=Path, required=True)
    a2.add_argument("--split", type=Path, required=True)
    a2.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "index":
        out = build_index(a.shards_dir, a.n_proc)
        with np.load(out) as z:
            print(f"indice {out}: {len(z['names'])} membri da {len(z['tars'])} tar")
    else:
        rep = blocks_report(a.spec, a.split)
        print(json.dumps({k: v for k, v in rep.items() if k != "blocks"}, indent=1))
        if a.out_json:
            a.out_json.write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
