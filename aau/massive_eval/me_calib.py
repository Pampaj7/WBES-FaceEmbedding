#!/usr/bin/env python3
"""Modelli massivi (``aau/runs/massive_release``): calibrazione c di d_P PRIMA di qualunque valutazione.

    AAU_NV=--nv aau/run.sh aau/massive_eval/me_calib.py --model main|open        (me.sbatch, passo calib; GPU)

Metodo del paper (``fact_calib.calib_one``): coppie dello STESSO dominio, identita' diverse, etichette (topologie)
diverse, nessuna crop; d_P,modello = ||u_i - u_j|| x dp_per_unit (``fact_calib.dp_of``: per lo stream deve coincidere
con ``targets.gt_unit("sr")``); c = mediana(d_P,GT) / mediana(d_P,modello), sensibilita' c_LS = sum(g m) / sum(m^2),
c per dominio. Held-out: le identita' sintetiche dei GENERATORI DI TRAINING del modello (fonti di ``--stream-sources``
del checkpoint meno FaMoS, che non e' un generatore) nell'held-out del monitoraggio ``datasets/MASSIVE_HELDOUT``
(``aau/massive/heldout.json``: semi riservati, mai nel training). Viste: ``heldout.load_heldout``; embedding
``heldout.embed`` (lo stesso servizio dei campioni del training, senza rotazione ne' rumore, pesi EMA); d_P,GT =
||sr_i - sr_j|| / sqrt(A) (``heldout.metrics``). PRIMARIA: solo viste neutre (le viste di calibrazione del paper sono
mesh d'identita'); sensibilita' ``con_espressioni``: anche le viste con espressione (stessa GT d'identita').
Uscite: ``aau/runs/evidence/massive_eval/calib/<modello>.json`` (c, sha256 di checkpoint, held-out ed embedding) e gli
embedding in ``aau/runs/massive_eval/calib/<modello>.npz`` (fuori da git). Nessun dominio di test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
for _p in (REPO / "aau/massive", REPO / "v3_work/trainer/tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import heldout as H  # noqa: E402  (sola lettura; mette trainer e stream su sys.path)

RUNS = REPO / "aau/runs/massive_release"
HELDOUT = REPO / "datasets/MASSIVE_HELDOUT"
EV = REPO / "aau/runs/evidence/massive_eval"
WORK = REPO / "aau/runs/massive_eval"
# modello -> (cartella del run, epoca del checkpoint finale = ultimo EMA, PLAN_MASSIVE sez. 9 / 22.6)
MODELS = {"main": ("main_a100", 330), "open": ("release_open", 230)}


def checkpoint(model: str) -> Path:
    run, ep = MODELS[model]
    hits = sorted((RUNS / run / "runs").glob(f"v3_*/checkpoints/epoch{ep:03d}_ema.pth"))
    if len(hits) != 1:
        raise SystemExit(f"{model}: {len(hits)} epoch{ep:03d}_ema.pth in {RUNS / run}")
    return hits[0]


def generators(ckpt: Path) -> list[str]:
    """Fonti di training del checkpoint (``--stream-sources``) meno FaMoS: i generatori sintetici."""
    import torch
    import sources as S
    args = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
    return [d for d in S.parse_sources(str(args["stream_sources"])) if d != "famos"]


def calib(Z: np.ndarray, rows: list, gt: dict, dpu: float, sqA: float, doms: list, expr: bool) -> dict:
    """La regola di ``fact_calib.calib_one`` sulle viste dell'held-out."""
    key = np.asarray([r["key"] for r in rows])
    dom = np.asarray([r["domain"] for r in rows])
    lab = np.asarray([r["label"] for r in rows])
    neutral = np.asarray([r["expr"] == "neutral" for r in rows])          # tag della vista: "neutral" | "expr"
    ok = (lab != "crop") & (np.ones(len(rows), bool) if expr else neutral)
    sr = np.stack([gt[k]["sr"] for k in key])
    u = Z[:, 1:]
    i, j = np.triu_indices(len(rows), 1)
    keep = ok[i] & ok[j] & (dom[i] == dom[j]) & (key[i] != key[j]) & (lab[i] != lab[j])
    i, j = i[keep], j[keep]
    m = np.linalg.norm(u[i] - u[j], axis=1) * dpu
    t = np.linalg.norm(sr[i] - sr[j], axis=1) / sqA
    out = {"n_views": int(ok.sum()), "n_pairs": int(len(i)), "median_dP_gt": float(np.median(t)),
           "median_dP_model": float(np.median(m)), "c_median": float(np.median(t) / np.median(m)),
           "c_ls": float((t * m).sum() / (m * m).sum())}
    for d in doms:
        k = dom[i] == d
        out[f"c_median_{d}"] = float(np.median(t[k]) / np.median(m[k])) if k.any() else float("nan")
        out[f"n_pairs_{d}"] = int(k.sum())
    return out


def main() -> None:
    import fact_calib as fc
    import sources as S
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    t0 = time.time()
    ck = checkpoint(a.model)
    doms = generators(ck)
    out = EV / "calib" / f"{a.model}.json"
    if out.exists():
        raise SystemExit(f"{out} esiste gia': la c si fissa una volta, prima della valutazione")
    Hd = H.load_heldout(HELDOUT, doms)
    Hd["root"] = HELDOUT
    if sorted({r["domain"] for r in Hd["rows"]}) != sorted(doms):
        raise SystemExit(f"held-out: domini {sorted({r['domain'] for r in Hd['rows']})}, attesi {sorted(doms)}")
    Z, info = H.embed(ck, Hd, a.device)
    if not np.isfinite(Z).all():
        raise SystemExit("embedding non finiti")
    emb = WORK / "calib" / f"{a.model}.npz"
    emb.parent.mkdir(parents=True, exist_ok=True)
    np.savez(emb, Z=Z, key=np.asarray([r["key"] for r in Hd["rows"]]), domain=np.asarray([r["domain"] for r in Hd["rows"]]),
             label=np.asarray([r["label"] for r in Hd["rows"]]), expr=np.asarray([r["expr"] for r in Hd["rows"]]),
             checkpoint=str(ck))
    dpu = fc.dp_of(ck)
    sqA = float(np.sqrt(S.Unified().A))
    prim = calib(Z, Hd["rows"], Hd["gt"], dpu, sqA, doms, expr=False)
    sens = calib(Z, Hd["rows"], Hd["gt"], dpu, sqA, doms, expr=True)
    import torch
    rep = {"model": a.model, "checkpoint": str(ck.relative_to(REPO)), "sha256_checkpoint": fc.sha256(ck),
           "epoch": info["epoch"], "steps": info["epoch"] * info["steps_per_epoch"], "head": info["head"],
           "generators": doms, "dp_per_unit": dpu, **prim, "con_espressioni": sens,
           "heldout": str(HELDOUT.relative_to(REPO)), "sha256_heldout_json": fc.sha256(HELDOUT / "heldout.json"),
           "embeddings": str(emb.relative_to(REPO)), "sha256_embeddings": fc.sha256(emb),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "definition": "fact_calib.calib_one sull'held-out del run massivo: coppie stesso dominio, identita' ed "
                         "etichette diverse, senza crop, solo viste neutre (sensibilita': con le espressioni); c = "
                         "mediana(d_P GT) / mediana(||u_i - u_j|| x dp_per_unit), c_ls = minimi quadrati senza intercetta",
           "seconds": time.time() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"[me-calib] {a.model}: c {prim['c_median']:.4f} (LS {prim['c_ls']:.4f}; con espressioni "
          f"{sens['c_median']:.4f}), {prim['n_pairs']} coppie su {prim['n_views']} viste, "
          + ", ".join(f"{d} {prim[f'c_median_{d}']:.3f}" for d in doms) + f" -> {out}", flush=True)


if __name__ == "__main__":
    main()
