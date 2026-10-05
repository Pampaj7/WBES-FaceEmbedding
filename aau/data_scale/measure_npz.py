#!/usr/bin/env python3
"""Quanto occupa una mesh con operatori, chiave per chiave, e quanto si risparmia senza
toccare la ricetta.

    srun -p cpu -c 4 --mem=32G -t 00:30:00 env AAU_NV= aau/run.sh aau/data_scale/measure_npz.py \
        --ckpt <checkpoint.pth> FILE.npz [FILE.npz ...]

Per ogni file:
  * byte per chiave (in memoria, cioe' non compressi) e byte su disco;
  * due ricodifiche, scritte in una dir temporanea:
      ``i32``  senza perdita: indici COO e facce in int32 invece di int64, npz compresso;
      ``c16``  come ``i32`` piu' ``evecs`` in float16 (l'unica chiave con perdita).
    Entrambe si leggono col loader congelato (``GTReadyDatasetNPZ`` fa ``.long()`` sugli
    indici e ``.float()`` su ``evecs``): e' proprio questo che si verifica qui, caricando
    ogni file nelle tre codifiche con il loader e confrontando i tensori;
  * se ``--ckpt`` e' dato, l'embedding del modello sulle tre codifiche: lo scarto
    ``||z_c16 - z_orig||`` va letto contro la distanza mediana fra embedding di file
    diversi dello stesso lotto, che e' la scala a cui lavora il ranking.

Riporta anche se L, gradX e gradY condividono la stessa sparsita' (se si', gli indici si
potrebbero salvare una volta sola; serve pero' un loader che lo sappia, quindi non e' fra
le ricodifiche provate).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"))

from robustness.data_utils import GTReadyDataset, sample_to_device  # noqa: E402

SPARSE = ("L", "gradX", "gradY")


def encode(src: Path, dst: Path, evecs_fp16: bool) -> None:
    with np.load(src, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["faces"] = d["faces"].astype(np.int32)
    for b in SPARSE:
        d[f"{b}_indices"] = d[f"{b}_indices"].astype(np.int32)
    if evecs_fp16:
        d["evecs"] = d["evecs"].astype(np.float16)
    np.savez_compressed(dst, **d)


def sparse_dense_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.coalesce(), b.coalesce()
    if not torch.equal(a.indices(), b.indices()):
        return float("inf")
    return float((a.values() - b.values()).abs().max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="wbes_measure_"))
    dirs = {k: tmp / k for k in ("orig", "i32", "c16")}
    for d in dirs.values():
        d.mkdir()

    rows = []
    for p in a.files:
        with np.load(p, allow_pickle=False) as z:
            keys = {k: (str(z[k].dtype), list(z[k].shape), int(z[k].nbytes)) for k in z.files}
            idx = {b: z[f"{b}_indices"] for b in SPARSE}
        # nome unico per file: le tre cartelle d'origine hanno gli stessi nomi di file
        tag = f"{p.parent.name}__{p.name}"
        shutil.copy(p, dirs["orig"] / tag)
        encode(p, dirs["i32"] / tag, evecs_fp16=False)
        encode(p, dirs["c16"] / tag, evecs_fp16=True)
        same = {f"L==grad{c}": bool(idx["L"].shape == idx[f"grad{c}"].shape
                                    and np.array_equal(idx["L"], idx[f"grad{c}"])) for c in "XY"}
        rows.append({
            "file": str(p), "tag": tag,
            "n_verts": keys["verts"][1][0], "n_faces": keys["faces"][1][0],
            "bytes_disk": {k: (dirs[k] / tag).stat().st_size for k in dirs},
            "bytes_in_memory_by_key": {k: v[2] for k, v in keys.items()},
            "dtypes": {k: v[0] for k, v in keys.items()},
            "sparsity_shared": same,
        })

    # --- round trip col loader congelato ---------------------------------------------------
    ds = {k: GTReadyDataset(str(d)) for k, d in dirs.items()}
    for k in ds:
        assert ds[k].files == ds["orig"].files, f"liste diverse in {k}"
    # il loader ordina i file per nome: riallinea i campioni all'ordine di ``rows``
    pos = {name: i for i, name in enumerate(ds["orig"].files)}
    samples = {k: [ds[k][pos[r["tag"]]] for r in rows] for k in ds}
    for i, r in enumerate(rows):
        o = samples["orig"][i]
        assert o is not None, f"il loader rifiuta l'originale {r['tag']}"
        r["loader"] = {}
        for k in ("i32", "c16"):
            s = samples[k][i]
            if s is None:
                r["loader"][k] = "RIFIUTATO dal loader"
                continue
            ev_scale = float(o["evecs"].abs().max())
            r["loader"][k] = {
                "verts_maxdiff": float((s["verts"] - o["verts"]).abs().max()),
                "faces_equal": bool(torch.equal(s["faces"], o["faces"])),
                "mass_maxdiff": float((s["mass"] - o["mass"]).abs().max()),
                "evals_maxdiff": float((s["evals"] - o["evals"]).abs().max()),
                "evecs_maxdiff_rel": float((s["evecs"] - o["evecs"]).abs().max()) / ev_scale,
                **{f"{b}_maxdiff": sparse_dense_diff(s[b], o[b]) for b in SPARSE},
            }

    # --- embedding --------------------------------------------------------------------------
    summary = {}
    if a.ckpt is not None:
        from robustness.model_helpers import build_model, forward_model
        from robustness.posthoc_runner import load_checkpoint_bundle, merge_run_args
        from types import SimpleNamespace

        margs = SimpleNamespace(**merge_run_args(a.ckpt, ""))
        dev = torch.device("cpu")
        model = build_model(args=margs, device=dev)
        model.load_state_dict(load_checkpoint_bundle(a.ckpt)["state_dict"], strict=True)
        model.eval()
        Z = {}
        with torch.no_grad():
            for k in samples:
                zs = []
                for s in samples[k]:
                    sd = sample_to_device(s, dev)
                    z, _ = forward_model(model, sd, sd["verts"], return_gate_info=False, add_noise=False)
                    zs.append(z.reshape(-1))
                Z[k] = torch.stack(zs)
        D = torch.cdist(Z["orig"], Z["orig"])
        med = float(D[torch.triu(torch.ones_like(D), 1) > 0].median())
        for k in ("i32", "c16"):
            dz = (Z[k] - Z["orig"]).norm(dim=1)
            summary[f"embedding_{k}"] = {
                "max_dz": float(dz.max()),
                "median_pairwise_dist_orig": med,
                "max_dz_over_median_pair": float(dz.max()) / med,
            }
            for i, r in enumerate(rows):
                r.setdefault("embedding_dz", {})[k] = float(dz[i])
        summary["ckpt"] = str(a.ckpt)

    for r in rows:
        bd = r["bytes_disk"]
        print(f"{r['tag']}: V={r['n_verts']} F={r['n_faces']}  disco orig={bd['orig']/1e6:.2f} MB "
              f"i32={bd['i32']/1e6:.2f} MB c16={bd['c16']/1e6:.2f} MB  "
              f"evecs_rel={r['loader']['c16']['evecs_maxdiff_rel']:.2e}  "
              f"sparsita' condivisa={r['sparsity_shared']}", flush=True)
    print(json.dumps(summary, indent=2))
    if a.out_json:
        a.out_json.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2) + "\n")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
