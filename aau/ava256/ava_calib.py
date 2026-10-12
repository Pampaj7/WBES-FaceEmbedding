#!/usr/bin/env python3
"""GT dei 56 soggetti di calibrazione di Ava-256 e controllo delle viste di calibrazione (12 ottobre 2026, PRIMA di
qualsiasi valutazione).

    v3_work/unified_gt/run.sh aau/ava256/ava_calib.py gt [--dry-run]   (una volta sola: scrive e congela la GT)
    v3_work/unified_gt/run.sh aau/ava256/ava_calib.py views     (viste di calibrazione contro quelle valutate)
    v3_work/unified_gt/run.sh aau/ava256/ava_calib.py verify    (verify_frozen() + verify_calib())

CONFERMATIVO: non valutare prima del protocollo confermativo (README). Nessun metodo, nessuna distanza di un metodo,
nessuno Spearman. Della GT dei 200 valutati si calcolano SOLO le impronte; della GT di calibrazione n coppie e range.

``gt``: la STESSA procedura di ``ava_gt.main`` per la GT della prova, con le stesse funzioni importate e non modificate:
``ava_gt.load`` (neutre sulla regione unificata), frame di dominio u = 1, R = I, t = 0, ``train_fr_sr.factor`` su TUTTI i
validi (la rigida robusta e' per identita' verso mu, quindi una riga non dipende dalle altre), poi ``train_fr_sr.save_eval``
sul sottoinsieme: li' le righe di valutazione, qui quelle di calibrazione (ordine di ``split.json``). Controlli prima di
scrivere:
  - ``verify_frozen()`` prima e dopo (GT dei 200, viste e tabella di scala invariate);
  - le GT ``ava256_eval_*`` e ``ava256_all_*`` ricalcolate in una cartella temporanea (poi cancellata, i numeri non si
    leggono) devono avere le impronte congelate: stessa procedura, bit per bit;
  - la GT di calibrazione in mm coincide col blocco di calibrazione di ``ava256_all`` (entro l'arrotondamento float32).
Uscite: ``datasets/AVA256/calib_gt/ava256_calib_{fr,sr}.npz`` + json (formato di ``save_eval``),
``ava256_calib_centroid_size.npz``, ``ids.json``; poi sola lettura (file e cartella). Impronte (``ava_gt.content_sha256``)
in ``aau/ava256/calib_manifest.json`` (in git) e in ``datasets/AVA256/calib_gt/FROZEN_CALIB.json``. Con la cartella
presente ``gt`` si rifiuta di riscrivere: rigenerare richiede una decisione del PI.

``verify_calib()``: la funzione che gli strumenti di calibrazione chiamano (insieme a ``ava_freeze.verify_frozen()``)
prima di leggere la GT di calibrazione: ricalcola le impronte e le confronta con quelle in git; un solo scarto ->
RuntimeError.

``views``: la vista ``calib_view`` contro la ``eval_view``, una mesh (o una coppia di topologie della stessa persona) per
volta, nessuna distanza fra soggetti:
  - split: ``ac.split()`` ricalcolato = ``split.json`` in git = ``datasets/AVA256/gt/split.json``; 56 + 200 disgiunti;
  - completezza: 56 x 6 mesh (le 6 topologie del protocollo), nomi = id di calibrazione, nessun id valutato e viceversa;
    symlink verso ``topo/ava<riga>_GTready_<topologia>.npz`` della persona giusta; ``content_sha256.txt`` e
    ``labels.json`` coerenti; nessuna GT nella vista; tabella di scala con gli stessi nomi e aree positive;
  - coerenza con le viste valutate, per topologia: vertici e triangoli, lato medio, anelli di bordo, triangoli degeneri,
    vertici non usati; original e noisy con gli stessi triangoli in tutte le 256 persone; noisy = original spostata
    (RMS mm); crop diverso dalla original (frazione di vertici tenuti).
Uscita: ``aau/ava256/calib_view_check.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import ava_common as ac  # noqa: E402
import ava_freeze as af  # noqa: E402

CALIB_DIR = ac.DATA_ROOT / "calib_gt"
CALIB_FROZEN = CALIB_DIR / "FROZEN_CALIB.json"
MANIFEST = ac.SUMMARY_DIR / "calib_manifest.json"
VIEW_CHECK = ac.SUMMARY_DIR / "calib_view_check.json"
NAME = "ava256_calib"
TOPOLOGIES = ("original", "remesh", "crop", "noisy", "down8k", "up60k")
SHARED_F = ("original", "noisy")                 # stessi triangoli per tutte le persone (patch suddivisa comune)
BLOCK_TOL = 1e-5                                 # tolleranza relativa (float32) col blocco di ava256_all


def calib_sha256(path: Path) -> str:
    """Come ``ava_gt.content_sha256`` (= ``ava_freeze.gt_sha256``): ``D_orig`` float32 e nomi."""
    return af.gt_sha256(path)


def size_sha256(path: Path) -> str:
    with np.load(path) as z:
        h = hashlib.sha256(np.ascontiguousarray(z["S"], dtype="<f8").tobytes())
        h.update("\n".join(str(s) for s in z["names"]).encode())
    return h.hexdigest()


def fingerprints(root: Path = CALIB_DIR) -> dict:
    return {f"{NAME}_fr": calib_sha256(root / f"{NAME}_fr.npz"), f"{NAME}_sr": calib_sha256(root / f"{NAME}_sr.npz"),
            f"{NAME}_centroid_size": size_sha256(root / f"{NAME}_centroid_size.npz")}


def verify_calib(root: Path = CALIB_DIR, manifest: Path = MANIFEST) -> dict:
    """Controllo dell'impronta della GT di calibrazione, da chiamare (con ``ava_freeze.verify_frozen()``) prima di
    leggerla. Ritorna le impronte; se una sola differisce da quelle in git solleva RuntimeError."""
    ref = json.loads(Path(manifest).read_text())["fingerprints"]
    try:
        cur = fingerprints(root)
    except FileNotFoundError as e:
        raise RuntimeError(f"Ava-256: GT di calibrazione mancante ({e})") from e
    diff = [k for k, v in ref.items() if cur.get(k) != v]
    if diff:
        raise RuntimeError(f"Ava-256: GT di calibrazione diversa da quella congelata ({manifest}): {diff}")
    return cur


def q3(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"min": float(x.min()), "median": float(np.median(x)), "max": float(x.max())}


# ------------------------------------------------------------------------------------------------ GT di calibrazione

def stage_gt(dry_run: bool = False) -> None:
    """``dry_run``: tutti i controlli, GT di calibrazione in una cartella temporanea (impronte stampate), nulla di scritto
    nei dati ne' in git."""
    if CALIB_DIR.exists():
        raise SystemExit(f"{CALIB_DIR} esiste gia': GT di calibrazione congelata (rigenerare: decisione del PI)")
    t0 = time.time()
    frozen = af.verify_frozen()                       # prima: GT dei 200, viste, scala come in git
    import ava_gt as ag                              # noqa: E402  (stesse funzioni della GT della prova)
    tfs, Canon, C = ag.tfs, ag.Canon, ag.C

    # come ava_gt.main, righe 1-6 della GT
    cn = Canon()
    data = ag.load()
    caps, valid = data["caps"], data["valid"]
    iv = np.flatnonzero(valid)
    cn.frames[ag.NAME] = {"u": ag.U_MM, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}
    sp = ac.split()
    pos = {caps[i]["ava_id"]: k for k, i in enumerate(iv)}
    ie = np.array([pos[a] for a in sp["evaluation"] if a in pos])
    ic = np.array([pos[a] for a in sp["calibration"] if a in pos])
    ids = [caps[i]["view_id"] for i in iv]
    F = tfs.factor(cn, ag.NAME, data["P"][iv])

    # la procedura riproduce le GT congelate (impronte soltanto: i numeri dei valutati non si leggono)
    with tempfile.TemporaryDirectory(prefix="ava_calib_") as tmp:
        tfs.EVAL_DIR = Path(tmp)
        tfs.save_eval(f"{ag.NAME}_all", ids, F, cn.sp.A)
        tfs.save_eval(f"{ag.NAME}_eval", [ids[k] for k in ie], ag.subset(F, ie), cn.sp.A)
        repro = {f"ava256_{s}_{k}": af.gt_sha256(Path(tmp) / f"{ag.NAME}_{s}_{k}.npz")
                 for s in ("eval", "all") for k in ("fr", "sr")}
    bad = [k for k, v in repro.items() if v != frozen["gt"][k]]
    if bad:
        raise SystemExit(f"la procedura non riproduce le GT congelate: {bad} (non scrivo nulla)")

    # GT di calibrazione, nella cartella nuova (scritta a parte, poi rinominata)
    names = [ids[k] for k in ic]
    with open(ac.SUMMARY_DIR / "split.json") as f:
        if names != json.load(f)["calibration_view_ids"]:
            raise SystemExit("righe di calibrazione diverse da split.json")
    work = CALIB_DIR.with_name(CALIB_DIR.name + ".tmp")
    if dry_run:
        work = Path(tempfile.mkdtemp(prefix="ava_calib_dry_")) / CALIB_DIR.name
    elif work.exists():
        raise SystemExit(f"{work} esiste: resto di un'esecuzione interrotta, da togliere a mano")
    tfs.EVAL_DIR = work
    work.mkdir(parents=True)
    D = tfs.save_eval(NAME, names, ag.subset(F, ic), cn.sp.A)
    rows = [{"view_id": caps[iv[k]]["view_id"], "ava_id": caps[iv[k]]["ava_id"], "capture": caps[iv[k]]["capture"]}
            for k in ic]
    C.save_json(work / "ids.json", {"note": f"righe di {NAME}_* (i {len(ic)} soggetti di calibrazione, ordine di "
                                            "split.json); MAI valutati, da nessun metodo", "calib": rows})

    # controllo indipendente: blocco di calibrazione di ava256_all (GT congelata), in mm
    block = {}
    for k in ("fr", "sr"):
        with np.load(ac.GT_DIR / f"ava256_all_{k}.npz") as z:
            all_names = [str(s) for s in z["names"]]
            sel = np.array([all_names.index(n) for n in names])
            unit = json.loads((ac.GT_DIR / f"ava256_all_{k}.json").read_text())[tfs.UNIT[k][0]]
            B = z["D_orig"].astype(np.float64)[np.ix_(sel, sel)] * unit
        with np.load(work / f"{NAME}_{k}.npz") as z:
            G = z["D_orig"].astype(np.float64) * json.loads((work / f"{NAME}_{k}.json").read_text())[tfs.UNIT[k][0]]
        err = float(np.abs(G - B).max() / np.abs(B).max())
        block[k] = {"max_rel_diff": err, "inside_tol": bool(err <= BLOCK_TOL)}
        del B, G
    if not all(b["inside_tol"] for b in block.values()):
        raise SystemExit(f"GT di calibrazione diversa dal blocco di ava256_all: {block} (non congelo)")

    if dry_run:
        print(json.dumps({"dry_run": True, "frozen_gt_reproduced": {k: v == frozen["gt"][k] for k, v in repro.items()},
                          "block_of_ava256_all": block, "fingerprints": fingerprints(work)}, indent=1), flush=True)
        shutil.rmtree(work.parent)
        return
    work.rename(CALIB_DIR)
    fp = fingerprints()
    iu = np.triu_indices(len(ic), 1)
    man = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "rule": __doc__.split("``verify_calib()``")[0].split("``gt``:")[1].strip(),
           "files": f"datasets/AVA256/calib_gt/{NAME}_{{fr,sr}}.npz (D_orig / massimo, json con l'unita'), "
                    f"{NAME}_centroid_size.npz, ids.json",
           "split": {"n_calibration": int(len(ic)), "n_evaluation": int(len(ie)), "salt": ac.SPLIT_SALT},
           "procedure_check": {"frozen_gt_reproduced": {k: v == frozen["gt"][k] for k, v in repro.items()},
                               "block_of_ava256_all": block, "tolerance": BLOCK_TOL},
           "descriptive": {"n_subjects": int(len(ic)), "n_pairs": int(len(iu[0])),
                           "fr_mm": q3(D["fr"][iu]), "sr_dP": q3(D["sr"][iu]),
                           "note": "solo la GT dei 56 di calibrazione; nessun metodo"},
           "fingerprints": fp}
    C.save_json(MANIFEST, man)
    C.save_json(CALIB_FROZEN, man)
    n = af._readonly(CALIB_DIR)
    verify_calib()
    af.verify_frozen()                                # dopo: niente e' cambiato nei dati congelati
    print(f"[ava-calib] GT di calibrazione: {len(ic)} soggetti, {n} voci in sola lettura ({time.time() - t0:.0f}s)",
          flush=True)
    print(json.dumps({k: man[k] for k in ("procedure_check", "descriptive", "fingerprints")}, indent=1), flush=True)


# ------------------------------------------------------------------------------------------ viste di calibrazione

def mesh_facts(path: Path) -> dict:
    """Proprieta' di una mesh sola: dimensioni, lato medio, anelli di bordo, degeneri, vertici non usati, impronte."""
    import ava_view_qc as qc                          # noqa: E402  (mean_edge, n_loops: definizioni di view_qc.json)
    with np.load(path) as z:
        V, F = z["V"], z["F"]
    V64 = V.astype(np.float64)
    area = 0.5 * np.linalg.norm(np.cross(V64[F[:, 1]] - V64[F[:, 0]], V64[F[:, 2]] - V64[F[:, 0]]), axis=1)
    rep = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
    ok_idx = bool(F.min() >= 0 and F.max() < len(V))
    return {"nv": int(len(V)), "nf": int(len(F)), "finite": bool(np.isfinite(V).all()), "index_ok": ok_idx,
            "degenerate": int(((area < 1e-12) | rep).sum()),
            "unused": int(len(V) - len(np.unique(F))) if ok_idx else -1,
            "edge": qc.mean_edge(V64, F) if ok_idx else float("nan"), "loops": qc.n_loops(F) if ok_idx else -1,
            "sha": af.mesh_sha256(path),
            "F_sha": hashlib.sha256(np.ascontiguousarray(F, dtype="<i4").tobytes()).hexdigest()}


def view_facts(kind: str, members: list[str]) -> tuple[dict, list[str]]:
    """Fatti per mesh di una vista e lista dei problemi (completezza, symlink, content_sha256.txt, labels.json)."""
    root = ac.DATA_ROOT / kind
    probs = []
    want = {f"{v}_GTready_{t}" for v in members for t in TOPOLOGIES}
    have = {p.stem for p in (root / "npz").glob("*.npz")}
    if have != want:
        probs.append(f"{kind}: {len(want - have)} mesh mancanti, {len(have - want)} in piu'")
    listed = dict(ln.split(" ") for ln in (root / "content_sha256.txt").read_text().split("\n") if ln)
    labels = json.loads((root / "labels.json").read_text())
    if set(labels["labels"]) != want or set(labels["subjects"]) != set(members):
        probs.append(f"{kind}: labels.json diverso dai membri")
    facts = {}
    for name in sorted(have & want):
        p = root / "npz" / f"{name}.npz"
        vid, topo = name.split("_GTready_")
        tgt = ac.DATA_ROOT / "topo" / f"ava{int(vid[2:]) - ac.ID_OFFSET:04d}_GTready_{topo}.npz"
        if not p.is_symlink() or p.resolve() != tgt.resolve():
            probs.append(f"{kind}: {name} non punta a {tgt.name}")
        f = mesh_facts(p)
        if listed.get(name) != f["sha"]:
            probs.append(f"{kind}: {name} diverso da content_sha256.txt")
        if labels["labels"].get(name) != vid:
            probs.append(f"{kind}: etichetta di {name}")
        f["vid"], f["topo"] = vid, topo
        facts[name] = f
    return facts, probs


def pair_facts(root: Path, vid: str) -> dict:
    """Noisy e crop contro la original della stessa persona (una persona sola, nessuna distanza fra soggetti)."""
    with np.load(root / "npz" / f"{vid}_GTready_original.npz") as z:
        Vo, Fo = z["V"].astype(np.float64), z["F"]
    with np.load(root / "npz" / f"{vid}_GTready_noisy.npz") as z:
        Vn, Fn = z["V"].astype(np.float64), z["F"]
    with np.load(root / "npz" / f"{vid}_GTready_crop.npz") as z:
        Vc, Fc = z["V"], z["F"]
    same_F = bool(Fn.shape == Fo.shape and (Fn == Fo).all())
    rms = float(np.sqrt(((Vn - Vo) ** 2).sum(1).mean())) if same_F else float("nan")
    crop_same = bool(Vc.shape == Vo.shape and Fc.shape == Fo.shape and (Fc == Fo).all())
    return {"noisy_same_F": same_F, "noisy_rms_mm": rms, "crop_kept_vertex_fraction": len(Vc) / len(Vo),
            "crop_identical_to_original": crop_same}


def summarize(facts: dict, pairs: dict) -> dict:
    out = {}
    for t in TOPOLOGIES:
        fs = [f for f in facts.values() if f["topo"] == t]
        out[t] = {"n": len(fs), "vertices": q3([f["nv"] for f in fs]), "faces": q3([f["nf"] for f in fs]),
                  "mean_edge_mm": q3([f["edge"] for f in fs]),
                  "boundary_loops": sorted({f["loops"] for f in fs}),
                  "non_finite": sum(not f["finite"] for f in fs), "bad_index": sum(not f["index_ok"] for f in fs),
                  "degenerate_faces": q3([f["degenerate"] for f in fs]), "unused_vertices": q3([f["unused"] for f in fs]),
                  "distinct_F": len({f["F_sha"] for f in fs})}
    out["noisy_vs_original"] = {"same_F": sum(p["noisy_same_F"] for p in pairs.values()),
                                "rms_mm": q3([p["noisy_rms_mm"] for p in pairs.values()])}
    out["crop_vs_original"] = {"identical": sum(p["crop_identical_to_original"] for p in pairs.values()),
                               "kept_vertex_fraction": q3([p["crop_kept_vertex_fraction"] for p in pairs.values()])}
    return out


def stage_views() -> None:
    t0 = time.time()
    af.verify_frozen()
    probs = []
    sp = ac.split()
    git_split = json.loads((ac.SUMMARY_DIR / "split.json").read_text())
    data_split = json.loads((ac.GT_DIR / "split.json").read_text())
    split_ok = all(s["calibration"] == sp["calibration"] and s["evaluation"] == sp["evaluation"]
                   for s in (git_split, data_split))
    if not split_ok:
        probs.append("split.json diverso da ac.split()")
    calib, evalv = git_split["calibration_view_ids"], git_split["evaluation_view_ids"]
    if len(calib) != ac.N_CALIB or len(evalv) != 256 - ac.N_CALIB or set(calib) & set(evalv) or \
            len(set(calib) | set(evalv)) != 256:
        probs.append("split: non 56 + 200 disgiunti su 256")

    facts, pairs, summ = {}, {}, {}
    for kind, members in (("calib_view", calib), ("eval_view", evalv)):
        facts[kind], pr = view_facts(kind, members)
        probs += pr
        pairs[kind] = {v: pair_facts(ac.DATA_ROOT / kind, v) for v in members}
        summ[kind] = summarize(facts[kind], pairs[kind])
    root = ac.DATA_ROOT / "calib_view"
    if list(root.glob("gt_*")):
        probs.append("calib_view contiene una GT")
    if not (root / "CALIBRAZIONE_NON_VALUTARE.txt").exists():
        probs.append("calib_view senza CALIBRAZIONE_NON_VALUTARE.txt")
    with np.load(ac.DATA_ROOT / "scale" / "calib_view.npz") as z:
        if sorted(str(s) for s in z["names"]) != sorted(f"{n}.npz" for n in facts["calib_view"]) or \
                not (np.isfinite(z["area_mm2"]).all() and (z["area_mm2"] > 0).all()):
            probs.append("tabella di scala di calib_view incoerente con la vista")

    # coerenza con le viste valutate, per topologia
    cmp = {}
    for t in TOPOLOGIES:
        c, e = summ["calib_view"][t], summ["eval_view"][t]
        cmp[t] = {"loops_equal": c["boundary_loops"] == e["boundary_loops"],
                  **{f"{k}_median_rel_diff": c[k]["median"] / e[k]["median"] - 1.0
                     for k in ("vertices", "faces", "mean_edge_mm")}}
        if c["n"] != ac.N_CALIB or c["non_finite"] or c["bad_index"]:
            probs.append(f"calib_view {t}: {c['n']} mesh, {c['non_finite']} non finite, {c['bad_index']} indici errati")
        if not cmp[t]["loops_equal"]:
            probs.append(f"calib_view {t}: anelli di bordo {c['boundary_loops']} contro {e['boundary_loops']}")
    for t in SHARED_F:
        shared = {f["F_sha"] for k in facts for f in facts[k].values() if f["topo"] == t}
        cmp[t]["same_F_all_256"] = len(shared) == 1
        if len(shared) != 1:
            probs.append(f"{t}: {len(shared)} triangolazioni diverse fra le 256 persone")
    for kind in pairs:
        if summ[kind]["noisy_vs_original"]["same_F"] != len(pairs[kind]) or summ[kind]["crop_vs_original"]["identical"]:
            probs.append(f"{kind}: noisy con triangoli diversi dalla original o crop identico alla original")

    out = {"definition": __doc__.split("``views``:")[1].split("Uscita:")[0].strip(),
           "verified_frozen": True, "split": {"n_calibration": len(calib), "n_evaluation": len(evalv),
                                              "split_json_equals_rule": split_ok},
           "views": summ, "calib_vs_eval": cmp, "problems": probs, "ok": not probs,
           "seconds": round(time.time() - t0)}
    VIEW_CHECK.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({"ok": out["ok"], "problems": probs, "calib_vs_eval": cmp}, indent=1), flush=True)
    if probs:
        raise SystemExit(f"[ava-calib] viste di calibrazione: {len(probs)} problemi")
    print(f"[ava-calib] viste di calibrazione complete e coerenti ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=("gt", "views", "verify"))
    p.add_argument("--dry-run", action="store_true", help="gt: controlli e impronte senza scrivere nulla")
    a = p.parse_args()
    if a.stage == "gt":
        stage_gt(a.dry_run)
    elif a.stage == "views":
        stage_views()
    else:
        af.verify_frozen()
        print(json.dumps(verify_calib(), indent=1))
        print("[ava-calib] impronte uguali a quelle congelate (dati del freeze e GT di calibrazione)")


if __name__ == "__main__":
    main()
