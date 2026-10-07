#!/usr/bin/env python3
"""Congela in un file i soggetti di TEST attuali di BFM e ICT, prima di generare dati nuovi.

    srun -p cpu -c 4 --mem=16G -t 00:20:00 env AAU_NV= aau/run.sh aau/data_scale/freeze_heldout.py

Perche' serve: ``rebuild_subject_split`` ricalcola lo split sull'UNIONE dei soggetti della
vista, quindi aggiungere identita' sposterebbe nel training soggetti che oggi sono di test.
Da qui in avanti lo split dei soggetti vecchi e' questo file, non il seme.

Un soggetto e' "di test" se e' stato held-out in ALMENO UNO dei training i cui numeri
esistono, cioe' l'unione di (tutti letti dagli split veri, nessuno ricostruito a mano tranne
i semi BFM 2345/3456, controllati contro il 1234 di splits.json):
  BFM  ``rebuild_subject_split`` sui 500 soggetti, eval_fraction 0.2, semi 1234/2345/3456
       (ricetta v1: remesh_v1recipe_*, areanorm, current, rms, ablazioni);
       held-out del congiunto (aau/runs/ws2_cross3dmm/splits.json) e di joint_E
       (aau/runs/joint_E/splits.json); celle di valutazione di ws2.
  ICT  held-out di ICT-5000 (id14500-id14999, datasets/ICT/train_ready/splits.json);
       held-out di ICT-only (1000) e del congiunto in ws2; held-out ed eval di joint_E.
HIFI3D e FLAME sono DOMINI di test: nessun loro soggetto puo' entrare nel training, e gli id
nuovi (``new_id_range``) non si sovrappongono ai loro range (FLAME id1000-id5999).

Due insiemi (decisione del PI, 6 ottobre):
  heldout_frozen.json          POLITICA IN USO ("joint_exact", 6 ottobre, dopo la revisione del
                               critic): ESATTAMENTE gli held-out del congiunto
                               x3dmm_joint_bfm_ict_s1234_1019532 (BFM 108 + ICT 992, ws2 splits.json
                               models.joint), ne' piu' ne' meno, cosi' il training BFM coincide con
                               il suo (392 soggetti). I 100 BFM del protocollo standard NON sono
                               congelati: 81 di loro sono nel training del congiunto stesso, quindi
                               il confronto sul protocollo standard ha gia' quel limite per il
                               congiunto. E' quello che legge la guardia del trainer.
  heldout_frozen_union15.json  l'unione di sopra, per riferimento: e' quella con cui e' stata fatta
                               la guardia di generazione degli shard (piu' severa, quindi valida
                               anche per la politica in uso).
  heldout_ict_originals.npz    le ``original`` degli held-out ICT dell'UNIONE, normalizzate maxabs
                               (float32), piu' la soglia di quasi-duplicato: il controllo di
                               gen_ict_shard.py confronta OGNI identita' nuova con queste.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"))
from robustness.data_utils import rebuild_subject_split  # noqa: E402

DS = REPO_ROOT / "datasets"
RUNS = REPO_ROOT / "aau/runs"
ICT_OFFSET = 10000
NEW_ID_RANGE = (20000, 69999)
BFM_SEEDS = (1234, 2345, 3456)


def ict_view_to_raw(v: str) -> str:
    return f"ict{int(v[2:]) - ICT_OFFSET:04d}"


def main() -> None:
    ws2 = json.loads((RUNS / "ws2_cross3dmm/splits.json").read_text())
    je = json.loads((RUNS / "joint_E/splits.json").read_text())
    ict5000 = json.loads((DS / "ICT/train_ready/splits.json").read_text())

    bfm_all = sorted(p.name.split("_GTready")[0]
                     for p in (DS / "REMESH/npz_data_topo_500").glob("id*_GTready_original.npz"))
    assert len(bfm_all) == 500, len(bfm_all)
    sources: dict[str, list[str]] = {}
    for seed in BFM_SEEDS:
        _, held = rebuild_subject_split(bfm_all, eval_fraction=0.2, seed=seed, max_subjects=0)
        sources[f"bfm_v1_seed{seed}"] = held
    if sources["bfm_v1_seed1234"] != sorted(ws2["models"]["bfm_only"]["heldout"]):
        raise SystemExit("lo split BFM ricostruito col seme 1234 non coincide con ws2 splits.json")

    def is_bfm(s: str) -> bool:
        return int(s[2:]) < 1000

    sources["joint_ws2_bfm"] = [s for s in ws2["models"]["joint"]["heldout"] if is_bfm(s)]
    sources["joint_ws2_ict"] = [s for s in ws2["models"]["joint"]["heldout"] if not is_bfm(s)]
    sources["ict_only_ws2"] = list(ws2["models"]["ict_only"]["heldout"])
    sources["ict5000_train_ready"] = list(ict5000["heldout"])
    sources["joint_E_bfm"] = [s for s in je["heldout"] if is_bfm(s)] + list(je["eval_subjects"]["bfm"])
    sources["joint_E_ict"] = [s for s in je["heldout"] if not is_bfm(s)] + list(je["eval_subjects"]["ict"])
    for cell, c in ws2["cells"].items():
        sources[f"ws2_cell_{cell}"] = list(c["subjects"])

    bfm, ict = set(), set()
    for name, lst in sources.items():
        for s in lst:
            (bfm if is_bfm(s) else ict).add(s)
    bad = [s for s in ict if not ICT_OFFSET <= int(s[2:]) < ICT_OFFSET + 5000]
    if bad:
        raise SystemExit(f"id ICT fuori range: {bad[:5]}")

    out = {
        "rule": "unione degli held-out di ogni training i cui numeri esistono; vedi docstring",
        "bfm": sorted(bfm),
        "ict_view": sorted(ict, key=lambda s: int(s[2:])),
        "ict_raw": sorted(ict_view_to_raw(s) for s in ict),
        "counts": {"bfm": len(bfm), "ict": len(ict)},
        "sources": {k: len(v) for k, v in sources.items()},
        "test_domains": ["hifi3d", "flame"],
        "reserved_id_ranges": {"bfm": [0, 499], "flame": [1000, 5999], "ict5000": [10000, 14999]},
        "new_id_range": list(NEW_ID_RANGE),
    }

    # riferimento geometrico: original degli held-out ICT, normalizzate maxabs
    raw = out["ict_raw"]
    V = []
    for r in raw:
        with np.load(DS / f"ICT/topo/{r}_GTready_original.npz") as z:
            v = z["V"].astype(np.float64)
        v = v - v.mean(0, keepdims=True)
        V.append((v / np.abs(v).max()).astype(np.float32))
    man = json.loads((DS / "ICT/gt/manifest.json").read_text())
    scale = float(man["normalization_scale"]["maxabs"])
    with np.load(DS / "ICT/gt/ict_matrix_distances_maxabs.npz") as z:
        D = z["D_orig"].astype(np.float64) * scale
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    # soglia: meta' del vicino piu' prossimo PIU' VICINO fra le 5000 di ICT-5000. Nessuna
    # coppia di identita' indipendenti di ICT-5000 ci scende sotto; una copia ci starebbe a 0.
    thr = 0.5 * float(nn.min())
    np.savez_compressed(THIS_DIR / "heldout_ict_originals.npz", names=np.array(raw),
                        V=np.stack(V), dup_threshold=thr, gt_scale=scale)
    out["ict_duplicate_check"] = {"metric": "vertex-mean-L2 fra original normalizzate maxabs",
                                  "threshold": thr, "ict5000_nn_min": float(nn.min()),
                                  "ict5000_nn_p1": float(np.percentile(nn, 1)),
                                  "ict5000_nn_median": float(np.median(nn))}
    (THIS_DIR / "heldout_frozen_union15.json").write_text(json.dumps(out, indent=1) + "\n")

    pol_bfm = sorted(set(sources["joint_ws2_bfm"]))
    pol_ict = sorted(set(sources["joint_ws2_ict"]), key=lambda s: int(s[2:]))
    std = set(sources["bfm_v1_seed1234"])
    joint_train = set(ws2["models"]["joint"]["train"])
    policy = {
        "policy": "joint_exact+gnm_val100",
        "reason": ("congela ESATTAMENTE gli held-out del congiunto x3dmm_joint_bfm_ict_s1234_1019532, "
                   "ne' piu' ne' meno, perche' il run grande deve essere confrontabile con lui: stesso "
                   "training BFM (392 soggetti) e stessi held-out. Decisione del PI, 6 ottobre (prima: "
                   "joint_compare = congiunto + 100 BFM standard, 189 BFM; prima ancora: unione dei 15 run). "
                   "7 ottobre: piu' i 100 GNM di validazione id110000-110099 (dominio di training GNM); "
                   "HIFI3D, FaceVerse e FLAME restano domini di test, fuori dal training per costruzione."),
        "bfm_standard_100_not_frozen": {
            "n": len(std - set(pol_bfm)),
            "of_which_in_joint_1019532_training": len((std - set(pol_bfm)) & joint_train),
            "note": "sul protocollo BFM standard il run grande ha lo stesso limite del congiunto: "
                    "quei soggetti sono di training per entrambi"},
        "compared_model": {"x3dmm_joint_bfm_ict_s1234_1019532": ws2["models"]["joint"]["run_dir"]},
        "online_eval_joint_1019532": list(ws2["models"]["joint"]["online_eval"]),
        "bfm": pol_bfm,
        "ict_view": pol_ict,
        # 7 ottobre: i 100 GNM tenuti come validazione dalla distillazione v2 (datasets/GNM_DISTILL)
        "gnm": [f"id{i}" for i in range(110000, 110100)],
        "ict_raw": sorted(ict_view_to_raw(s) for s in pol_ict),
        "counts": {"bfm": len(pol_bfm), "ict": len(pol_ict), "gnm": 100},
        "test_domains": ["hifi3d", "faceverse", "flame"],
        "reserved_id_ranges": out["reserved_id_ranges"],
        "new_id_range": out["new_id_range"],
        "generation_guard": "gli shard sono stati controllati contro heldout_frozen_union15.json, che contiene questo insieme",
    }
    (THIS_DIR / "heldout_frozen.json").write_text(json.dumps(policy, indent=1) + "\n")
    print(json.dumps({"union15": out["counts"], "policy": policy["counts"],
                      "ict_duplicate_check": out["ict_duplicate_check"]}, indent=1))


if __name__ == "__main__":
    main()
