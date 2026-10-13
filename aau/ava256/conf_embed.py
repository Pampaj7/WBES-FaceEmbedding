#!/usr/bin/env python3
"""Ava-256, prova confermativa: embedding [s, u] dei modelli sulle viste di un dominio da spec (PROTOCOL_CONFERMATIVO,
sez. 7.1 e 10), con la pipeline delle viste dev della matrice. Collaudato su HIFI3D con lo STESSO codice (spec
``aau/ava256/spec/hifi3d_full.json``: le 600 mesh di ``form_hifi`` e il pool NON valutato di D3, ``conf_collaudo.py
embed``). Nessun passo legge GT.

    aau/run.sh aau/ava256/conf_embed.py stage --spec <json>                       (conf.sbatch, passo embed-stage)
    AAU_NV=--nv aau/run.sh aau/ava256/conf_embed.py run --spec <json>             (conf.sbatch, passo embed; A100)

Insiemi (``embed.sets`` dello spec): {nome: {dir, subjects ("subjects" = i valutati dello spec, "head" = il blocco di
R6, ``head.subjects``), labels, scale_table, arms, role}}. Ava-256: ``eval_view`` (200 x 6 topologie: i due semi e C3M
e123, sez. 9) e ``calib_view`` (56 x 5 senza crop, ``diag.LABELS``: i due semi, blocco di R6). Bracci = chiavi di
``columns.models`` (i semi) o ``columns.desc_models`` (C3M): checkpoint e sha256.

``stage``: per insieme, un link per mesh in ``<embed.ops_root>/<insieme>/in`` (il file della vista, invariato: come
``zs_stage.py`` e ``d3_pools.py``) e ``names.npz`` con gli id dei soggetti in ``names`` e ``D_orig`` di zeri.
``zs_embed.py`` legge dal ``--dist_npz`` SOLO i nomi (``load_gt_distance_matrix``: i soggetti da tenere); qui nessuna GT
entra nell'embedding (``d3_embed.sbatch`` gli passava la gt_matrix della vista, che su Ava-256 non si legge). Gli
operatori (``v2_work/potential/areanorm_operators.py --k-eig 128``, come ``d3.sbatch``) li calcola ``conf.sbatch`` in
``<embed.ops_root>/<insieme>/ops``, a shard, col controllo del conteggio.

``run``: per insieme e braccio un processo nuovo (``one``: gli agganci di ``eval_v3.install`` si installano una volta
per processo), che controlla lo sha256 del checkpoint e che gli operatori siano esattamente le mesh messe in scena, poi
esegue ``eval_v3.main`` su ``aau/zs3dmm/zs_embed.py`` con gli argomenti di ``d3_embed.sbatch`` (``ARGS``),
``WBES_V3_FACTORIZED_OUT=full``, ``WBES_EVAL_SCENARIOS=clean``, ``WBES_V3_SCALE_TABLES`` = la tabella dell'insieme.
Rotazione del dominio: ``global_v3.rotations`` legge R_d da ``e12/frames.json`` per NOME del dominio della tabella; un
dominio assente (Ava-256: ``ava256``, ``ava_scale_table.py``) entra in ``global_v3.EXTRA_FRAMES`` con u e R del frame
dello spec (= ``frame.json``, sez. 11.8); un dominio presente deve avere esattamente la R dello spec (HIFI3D: uguali).
GPU obbligatoria (``zs_embed`` ripiegherebbe in silenzio sulla CPU), della classe ``embed.gpu`` dello spec (v5: ``A100``,
quella degli embedding di sviluppo; il TF32 del container resta attivo come in sviluppo). Uscita ``<work>/embed/<insieme>/<braccio>/``
``embeddings.npz`` (formato di zs_embed: ``Z`` = [s, u], ``subjects``, ``topologies``, ``files``, ``checkpoint``) e
``<evidence>/embed/<insieme>/<braccio>.json`` (mesh, checkpoint e sha256, tabella e sha256, dominio, origine di R, GPU,
nodo, job). Spec non di collaudo: le uscite devono essere i percorsi che leggono ``conf_eval.py``, ``conf_head.py`` e
``conf_desc.py`` (``columns.models``, ``columns.desc_models``, ``head.embeddings``); nei collaudi quei percorsi sono i
riferimenti di sviluppo (confronto in ``conf_collaudo.py embed``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
import conf_common as cc  # noqa: E402

TRAINER = cc.REPO / "v3_work" / "trainer"
ZS_EMBED = cc.AAU / "zs3dmm" / "zs_embed.py"
# argomenti di d3_embed.sbatch (embed_one), dopo --model_path / --data_dir / --dist_npz
ARGS = ("--checkpoint_selector", "best_by_clean", "--subject_split", "all", "--eval_fraction", "0.2", "--max_subjects", "0",
        "--max_meshes_per_subject_eval", "10", "--preload_workers", "8", "--pair_mode", "cross_topology", "--seed", "1234")
ENV = {"WBES_V3_FACTORIZED_OUT": "full", "WBES_EVAL_SCENARIOS": "clean"}


def arms_of(s: dict) -> dict:
    """{braccio: voce di ``columns.models`` o ``columns.desc_models``} (checkpoint, checkpoint_sha256, c, embeddings)."""
    return {**s["columns"]["models"], **(s["columns"].get("desc_models") or {})}


def subjects_of(s: dict, st: dict) -> list[str]:
    return sorted(s["subjects"] if st["subjects"] == "subjects" else s["head"]["subjects"])


def items_of(s: dict, name: str) -> list[tuple[str, str, Path]]:
    """(soggetto, etichetta, percorso) delle mesh dell'insieme."""
    st = s["embed"]["sets"][name]
    d = cc.rp(st["dir"])
    return [(x, t, d / f"{x}_GTready_{t}.npz") for x in subjects_of(s, st) for t in st["labels"]]


def ops_root(s: dict, name: str) -> Path:
    return cc.rp(s["embed"]["ops_root"]) / name


def out_path(s: dict, name: str, arm: str) -> Path:
    return cc.work(s) / "embed" / name / arm / "embeddings.npz"


def check_spec(s: dict) -> None:
    """Insiemi e bracci dello spec; spec non di collaudo: uscite = percorsi letti dalla catena."""
    import diag
    E, arms = s["embed"], arms_of(s)
    for name, st in E["sets"].items():
        if st["role"] == "head" and (st["subjects"] != "head" or list(st["labels"]) != list(diag.LABELS)):
            raise SystemExit(f"{name}: il blocco di R6 vuole i soggetti di head e le etichette {diag.LABELS}")
        if st["role"] == "eval" and (st["subjects"] != "subjects" or sorted(st["labels"]) != sorted(s["topologies"])):
            raise SystemExit(f"{name}: l'insieme valutato vuole i soggetti e le topologie dello spec")
        miss = [a for a in st["arms"] if a not in arms]
        if miss:
            raise SystemExit(f"{name}: bracci {miss} senza checkpoint nello spec")
        if s.get("collaudo"):
            continue
        for a in st["arms"]:
            want = arms[a]["embeddings"] if st["role"] == "eval" else s["head"]["embeddings"][a]
            if cc.rp(want).resolve() != out_path(s, name, a).resolve():
                raise SystemExit(f"{name} {a}: uscita {out_path(s, name, a)} diversa dal percorso letto ({want})")


# ------------------------------------------------------------------------------------------ messa in scena

def step_stage(spec: Path) -> None:
    s = cc.load(spec)
    if s.get("freeze"):
        cc.verify(s)
    check_spec(s)
    rep = {}
    for name, st in s["embed"]["sets"].items():
        items = items_of(s, name)
        miss = [str(p) for _, _, p in items if not p.exists()]
        if miss:
            raise SystemExit(f"{name}: {len(miss)} mesh assenti (es. {miss[:2]})")
        root = ops_root(s, name)
        d = root / "in"
        d.mkdir(parents=True, exist_ok=True)
        for stale in d.glob("*.npz"):
            stale.unlink()
        for _, _, p in items:
            (d / p.name).symlink_to(p.resolve())
        subj = subjects_of(s, st)
        np.savez(root / "names.npz", D_orig=np.zeros((len(subj), len(subj)), np.float32), names=np.asarray(subj))
        rep[name] = {"meshes": len(items), "subjects": len(subj), "labels": list(st["labels"]), "in": str(d),
                     "ops": str(root / "ops")}
        print(f"[conf-embed] {name}: {len(items)} mesh in scena in {d}", flush=True)
    cc.atomic_json(cc.evidence(s) / "embed" / "stage.json", {"spec": s["_path"], "sets": rep, **cc.stamp()})


# ------------------------------------------------------------------------------------------ embedding

def step_run(spec: Path, only_set: str | None, only_arm: str | None) -> None:
    s = cc.load(spec)
    if s.get("freeze"):
        cc.verify(s)
    check_spec(s)
    for name, st in s["embed"]["sets"].items():
        if only_set and name != only_set:
            continue
        for arm in st["arms"]:
            if only_arm and arm != only_arm:
                continue
            if out_path(s, name, arm).exists():
                print(f"[conf-embed] {out_path(s, name, arm)}: gia' presente, salto", flush=True)
                continue
            r = subprocess.run([sys.executable, __file__, "one", "--spec", str(spec), "--set", name, "--arm", arm])
            if r.returncode:
                raise SystemExit(f"{name} {arm}: embedding fallito ({r.returncode})")


def domain_rotation(s: dict, table: Path) -> tuple[str, str]:
    """(dominio della tabella, origine di R_d): aggiunge il dominio a ``global_v3.EXTRA_FRAMES`` se manca."""
    import global_v3
    with np.load(table) as z:
        doms = sorted({str(d) for d in z["domain"]})
    if len(doms) != 1:
        raise SystemExit(f"{table}: domini {doms}, atteso uno solo")
    dom, f = doms[0], s["frame"]
    known = global_v3.rotations()
    if dom in known:
        if not np.array_equal(known[dom], np.asarray(f["R"], dtype=np.float64)):
            raise SystemExit(f"{dom}: R di frames.json diversa da quella dello spec")
        return dom, "e12/frames.json (uguale allo spec)"
    global_v3.EXTRA_FRAMES[dom] = {"u": float(f["u"]), "R": f["R"], "unit_source": f"spec {s['_path']}"}
    return dom, f"spec {s['_path']} (frame.json)"


def step_one(spec: Path, name: str, arm: str) -> None:
    import torch
    s = cc.load(spec)
    st, a = s["embed"]["sets"][name], arms_of(s)[arm]
    if not torch.cuda.is_available():
        raise SystemExit("embedding senza GPU: zs_embed ripiegherebbe sulla CPU")
    if s["embed"]["gpu"] not in torch.cuda.get_device_name():               # v5: classe di GPU dello sviluppo
        raise SystemExit(f"GPU {torch.cuda.get_device_name()}, richiesta {s['embed']['gpu']} (embed.gpu dello spec)")
    ck = cc.rp(a["checkpoint"])
    got = cc.sha256(ck)
    if got != a["checkpoint_sha256"]:
        raise SystemExit(f"{arm}: sha256 del checkpoint {got}, atteso {a['checkpoint_sha256']}")
    root = ops_root(s, name)
    want = sorted(p.name for _, _, p in items_of(s, name))
    have = sorted(p.name for p in (root / "ops").glob("*.npz") if not p.name.startswith("."))
    if have != want:
        raise SystemExit(f"{name}: operatori {len(have)} contro {len(want)} mesh in scena")
    table = cc.rp(st["scale_table"])
    sys.path.insert(0, str(TRAINER))
    dom, rsrc = domain_rotation(s, table)
    out = out_path(s, name, arm)
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(ENV, WBES_V3_SCALE_TABLES=str(table))
    import eval_v3
    t0 = time.time()
    sys.argv = [eval_v3.__file__, "--", str(ZS_EMBED), "--model_path", str(ck), "--data_dir", str(root / "ops"),
                "--dist_npz", str(root / "names.npz"), *ARGS, "--embeddings_out", str(out)]
    with torch.no_grad():
        eval_v3.main()
    with np.load(out, allow_pickle=True) as z:
        keys = sorted(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))
        Z = np.asarray(z["Z"])
    if keys != sorted((x, t) for x, t, _ in items_of(s, name)) or not np.isfinite(Z).all():
        raise SystemExit(f"{out}: mesh diverse da quelle in scena o embedding non finiti")
    rep = {"set": name, "arm": arm, "meshes": len(keys), "Z_shape": list(Z.shape), "checkpoint": a["checkpoint"],
           "checkpoint_sha256": got, "scale_table": st["scale_table"], "scale_table_sha256": cc.sha256(table),
           "domain": dom, "rotation_source": rsrc, "ops": str(root / "ops"), "args": list(ARGS), "env": ENV,
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "seconds": time.time() - t0,
           "npz": str(out), **cc.stamp()}
    cc.atomic_json(cc.evidence(s) / "embed" / name / f"{arm}.json", rep)
    print(f"[conf-embed] {name} {arm}: Z {tuple(Z.shape)} su {rep['gpu']} in {rep['seconds']:.0f}s -> {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("stage", "run", "one"))
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--set", default=None)
    p.add_argument("--arm", default=None)
    a = p.parse_args()
    sys.path.insert(0, str(cc.AAU / "diagnostics"))
    if a.step == "stage":
        step_stage(a.spec)
    elif a.step == "run":
        step_run(a.spec, a.set, a.arm)
    else:
        step_one(a.spec, a.set, a.arm)


if __name__ == "__main__":
    main()
