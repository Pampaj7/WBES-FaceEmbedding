#!/usr/bin/env python3
"""E congiunto BFM+ICT contro controllo congiunto: soggetti, viste, split, token e controllo leak.

Il congiunto attuale (job 1019532, 5500 soggetti) non entra nella cache in RAM di train_fast.py con
le modifiche di E, quindi si addestrano DUE modelli nuovi sullo stesso sottoinsieme ridotto:
  - BFM: tutti i 500 soggetti;
  - ICT: 1500 dei 5000, estratti con np.random.default_rng(1234) dalla lista ordinata.
Due bracci, stessa ricetta v1 (aau/cross3dmm/train_joint_E.sbatch):
  - ctrl: frame current, operatori del congiunto attuale (BFM cotangente area 1 =
    npz_data_topo_500_withops_areanorm, ICT train_ready/npz_withops), niente token;
  - e:    frame rms, operatori robusti ad area 1 per entrambi i domini, token di taglia
    standardizzato PER DOMINIO con media e std dei soggetti di training di ciascun dominio.
GT: quella del congiunto attuale (datasets/JOINT_BFM_ICT/gt_matrix.npz, NaN fuori blocco), intera: il
trainer interseca da solo i soggetti con la data dir, e la matrice intera lascia invariato il massimo
con cui load_gt_distance_matrix normalizza (stessa scala di loss del congiunto attuale).

Split: quello del trainer, ricostruito con le sue funzioni (rebuild_subject_split su TUTTI i soggetti
della vista, eval_fraction 0.2, seed 1234), come train_v2 e aau/cross3dmm/ws2_views.py: lo split e'
sull'unione e i conteggi held-out per dominio vengono da li'. Dipende solo dai nomi dei file, che sono
gli stessi nei due bracci, quindi e' lo stesso split per entrambi.

Gli operatori robusti ICT (1500 soggetti, ~53 GB compressi) NON stanno nella quota CephFS da 1 TB, piena:
si riusano quelli di datasets/ICT/eval_view_heldout_robust_area1 se c'e' ancora (cancellata il 5 ottobre
per liberare la quota), gli altri si calcolano dentro i job, su /tmp del nodo (stage-input ->
robust_area1_operators.py -> stage-view).

Sottocomandi:
  build        soggetti, vista ctrl (CephFS), split, tabelle dei token per dominio, viste di eval
               held-out (BFM: tutti gli held-out BFM; ICT: 100 held-out ICT estratti con seed 1234) per
               ctrl e, per e, solo BFM; aau/runs/joint_E/splits.json.
  stage-input  --dest D --what train|eval_ict: in D/ict_in i symlink alle mesh ICT che non hanno
               ancora operatori robusti, in D/ict_ra1 i symlink a quelli gia' calcolati.
  stage-view   --dest D --what train|eval_ict: vista e in D/view (BFM robusti da CephFS, ICT da D/ict_ra1).
  verify       --run-root R: seed e i 16 soggetti dell'eval online scritti dal run coincidono con lo
               split ricostruito; nessun soggetto di eval e' di training.

    aau/run.sh aau/cross3dmm/joint_E_prep.py build
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"))
sys.path.insert(0, str(REPO_ROOT / "v2_work/train_v2"))
sys.path.insert(0, str(AAU_DIR / "ict"))

from make_ict_eval_views import link  # noqa: E402
from robustness import train_runner as v1  # noqa: E402  (stesse funzioni del training)
from robustness.data_utils import GTReadyDataset, rebuild_subject_split  # noqa: E402
from train_v2 import domain_of  # noqa: E402

DATASETS = REPO_ROOT / "datasets"
OUT = DATASETS / "JOINT_E"
RUNS = AAU_DIR / "runs" / "joint_E"
SEED = 1234
N_ICT = 1500
N_EVAL_ICT = 100
TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")

JOINT_GT = DATASETS / "JOINT_BFM_ICT" / "gt_matrix.npz"
BFM_GT = (REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder" / "latent_analysis"
          / "gt_distance_matrix" / "normalized_matrix_distances.npz")
ICT_GT = DATASETS / "ICT" / "train_ready" / "gt_matrix.npz"
CTRL_OPS = {"bfm": DATASETS / "REMESH" / "npz_data_topo_500_withops_areanorm",
            "ict": DATASETS / "ICT" / "train_ready" / "npz_withops"}
BFM_RA1 = DATASETS / "REMESH" / "npz_data_topo_500_withops_robust_area1"
ICT_RA1_DONE = DATASETS / "ICT" / "eval_view_heldout_robust_area1"
TOKEN_SRC = {"bfm": AAU_DIR / "runs" / "ablations_v3" / "size_token_bfm_s1234.json",
             "ict": AAU_DIR / "runs" / "ablations_v3" / "size_token_ict.json"}
NAME_RE = re.compile(r"^(id\d+)_GTready_([a-z0-9]+)$")


def select_subjects() -> dict[str, list[str]]:
    """BFM: tutti i soggetti del congiunto attuale; ICT: 1500 estratti con seed 1234."""
    _, name_to_idx = v1.load_gt_distance_matrix(str(JOINT_GT), subject_re=v1.SUBJECT_RE_ANY, dtype=np.float32)
    subs = sorted(name_to_idx)
    bfm = [s for s in subs if domain_of(s) == "bfm"]
    ict = [s for s in subs if domain_of(s) == "ict"]
    if len(bfm) != 500 or len(ict) != 5000:
        raise SystemExit(f"GT congiunta: {len(bfm)} BFM e {len(ict)} ICT, attesi 500 e 5000")
    pick = np.random.default_rng(SEED).choice(len(ict), size=N_ICT, replace=False)
    return {"bfm": bfm, "ict": sorted(ict[int(i)] for i in pick)}


def files_of(sid: str) -> list[str]:
    return [f"{sid}_GTready_{t}.npz" for t in TOPOLOGIES]


def build_view(out: Path, subjects: dict[str, list[str]], ops: dict[str, Path]) -> int:
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.npz"):
        stale.unlink()
    n = 0
    for dom, subs in subjects.items():
        for sid in subs:
            for name in files_of(sid):
                link(out / name, ops[dom] / name)
                n += 1
    return n


def trainer_split(view: Path) -> tuple[list[str], list[str]]:
    """Le righe di run_training che decidono lo split, con le funzioni del training (train_v2: GT ANY)."""
    dataset = GTReadyDataset(str(view))
    subj_map = v1.build_subject_map(dataset.files, subject_re=v1.SUBJECT_RE_ANY)
    _, name_to_idx = v1.load_gt_distance_matrix(str(JOINT_GT), subject_re=v1.SUBJECT_RE_ANY, dtype=np.float32)
    subjects = sorted(sid for sid in subj_map if sid in name_to_idx)
    return rebuild_subject_split(subjects=subjects, eval_fraction=0.2, seed=SEED, max_subjects=0)


def token_table(dom: str, subjects: list[str], train: list[str], out: Path) -> dict:
    """Tabella SizeTokenTable del dominio: log r grezzi di size_token.py, stats sul training di QUESTO split."""
    src = json.loads(TOKEN_SRC[dom].read_text())
    keep = set(subjects)
    log_r = {n: v for n, v in src["log_r"].items() if NAME_RE.match(n).group(1) in keep}
    if len(log_r) != 6 * len(subjects):
        raise SystemExit(f"{dom}: {len(log_r)} log r per {len(subjects)} soggetti in {TOKEN_SRC[dom]}")
    tset = set(train)
    vals = np.array([v for n, v in log_r.items() if NAME_RE.match(n).group(1) in tset])
    payload = {"collection": dom, "source_dir": src["source_dir"], "definition": src["definition"],
               "train": {"rule": (f"soggetti {dom} di training dello split del trainer sulla vista congiunta "
                                  f"ridotta (rebuild_subject_split, eval_fraction 0.2, seed {SEED})"),
                         "seed": SEED, "n_subjects": len(tset & keep), "n_meshes": int(vals.size),
                         "mean": float(vals.mean()), "std": float(vals.std())},
               "log_r": log_r}
    out.write_text(json.dumps(payload, indent=1))
    return payload["train"]


def load_splits() -> dict:
    return json.loads((RUNS / "splits.json").read_text())


def cmd_build(_args) -> None:
    subjects = select_subjects()
    view = OUT / "ctrl" / "npz_withops"
    n = build_view(view, subjects, CTRL_OPS)
    print(f"[joint-E] BFM {len(subjects['bfm'])}, ICT {len(subjects['ict'])} "
          f"({subjects['ict'][0]}..{subjects['ict'][-1]}); vista ctrl {n} file in {view}")
    train, heldout = trainer_split(view)
    if set(train) & set(heldout):
        raise SystemExit("train e held-out si intersecano")
    by_dom = {d: {"train": [s for s in train if domain_of(s) == d],
                  "heldout": [s for s in heldout if domain_of(s) == d]} for d in ("bfm", "ict")}

    RUNS.mkdir(parents=True, exist_ok=True)
    tokens = {}
    for dom in ("bfm", "ict"):
        path = RUNS / f"size_token_joint_{dom}_s{SEED}.json"
        tokens[dom] = {"path": str(path), **token_table(dom, subjects[dom], by_dom[dom]["train"], path)}

    ict_held = by_dom["ict"]["heldout"]
    pick = np.sort(np.random.default_rng(SEED).choice(len(ict_held), size=min(N_EVAL_ICT, len(ict_held)),
                                                     replace=False))
    evals = {"bfm": list(by_dom["bfm"]["heldout"]), "ict": [ict_held[int(i)] for i in pick]}
    leak = {d: len(set(s) & set(train)) for d, s in evals.items()}
    if any(leak.values()):
        raise SystemExit(f"leak: {leak}")
    eval_views = {}
    for key, (subs, ops) in {"ctrl__bfm": (evals["bfm"], CTRL_OPS["bfm"]),
                             "ctrl__ict": (evals["ict"], CTRL_OPS["ict"]),
                             "e__bfm": (evals["bfm"], BFM_RA1)}.items():
        dom = key.split("__")[1]
        v = OUT / "eval" / key
        eval_views[key] = {"view": str(v), "n_files": build_view(v, {dom: subs}, {dom: ops})}
    online = v1._select_online_eval_subjects(eval_subjects=by_dom["bfm"]["heldout"],
                                             max_subjects_eval_train=16, seed=SEED)
    out = {"seed": SEED, "subjects": subjects, "ctrl_view": str(view), "gt": str(JOINT_GT),
           "eval_gt": {"bfm": str(BFM_GT), "ict": str(ICT_GT)},
           "n_train": len(train), "n_heldout": len(heldout),
           "counts": {d: {k: len(v) for k, v in by_dom[d].items()} for d in by_dom},
           "train": train, "heldout": heldout, "eval_subjects": evals, "eval_views": eval_views,
           "online_eval_expected": online, "size_tokens": tokens, "leak_check": leak}
    (RUNS / "splits.json").write_text(json.dumps(out, indent=1))
    print(f"[joint-E] split: train {len(train)}, held-out {len(heldout)}, per dominio {out['counts']}")
    print(f"[joint-E] eval: BFM {len(evals['bfm'])}, ICT {len(evals['ict'])}; soggetti di training "
          f"fra quelli di eval {leak}")
    for dom, t in tokens.items():
        print(f"[joint-E] token {dom}: media {t['mean']:.5f} std {t['std']:.5f} su {t['n_subjects']} soggetti "
              f"({t['n_meshes']} mesh) -> {t['path']}")


def staged_subjects(what: str, splits: dict) -> dict[str, list[str]]:
    if what == "train":
        return splits["subjects"]
    if what == "eval_ict":
        return {"ict": splits["eval_subjects"]["ict"]}
    raise SystemExit(f"--what {what}")


def cmd_stage_input(args) -> None:
    subs = staged_subjects(args.what, load_splits())
    inp, ra1 = args.dest / "ict_in", args.dest / "ict_ra1"
    inp.mkdir(parents=True, exist_ok=True)
    ra1.mkdir(parents=True, exist_ok=True)
    reused = todo = 0
    for sid in subs["ict"]:
        for name in files_of(sid):
            if (ICT_RA1_DONE / name).is_file():
                link(ra1 / name, ICT_RA1_DONE / name)
                reused += 1
            else:
                link(inp / name, CTRL_OPS["ict"] / name)
                todo += 1
    print(f"[joint-E] stage {args.what}: operatori robusti ICT riusati {reused}, da calcolare {todo} in {ra1}")


def cmd_stage_view(args) -> None:
    subs = staged_subjects(args.what, load_splits())
    ops = {"bfm": BFM_RA1, "ict": args.dest / "ict_ra1"}
    n = build_view(args.dest / "view", subs, ops)
    print(f"[joint-E] vista e ({args.what}): {n} file in {args.dest / 'view'}")


def cmd_verify(args) -> None:
    splits = load_splits()
    runs = [p.parent for p in args.run_root.glob("*/config.json")]
    if len(runs) != 1:
        raise SystemExit(f"attesa una run dir in {args.run_root}, trovate {runs}")
    run = runs[0]
    cfg = json.loads((run / "config.json").read_text())["args"]
    if int(cfg["seed"]) != splits["seed"]:
        raise SystemExit(f"seed del run {cfg['seed']} != {splits['seed']}")
    written = json.loads((run / "online_eval_summary.json").read_text())["selected_subjects"]
    if list(written) != list(splits["online_eval_expected"]):
        raise SystemExit(f"eval online del run {written[:4]}.. diversa dallo split ricostruito "
                         f"{splits['online_eval_expected'][:4]}..: lo split NON e' quello del training")
    train = set(splits["train"])
    leak = {d: len(set(s) & train) for d, s in splits["eval_subjects"].items()}
    if any(leak.values()):
        raise SystemExit(f"leak: {leak}")
    print(f"[joint-E] verify {args.run_root.name}: seed {cfg['seed']}, eval online 16/16 come lo split "
          f"ricostruito, soggetti di eval nel training {leak}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    for name in ("stage-input", "stage-view"):
        s = sub.add_parser(name)
        s.add_argument("--dest", type=Path, required=True)
        s.add_argument("--what", required=True, choices=["train", "eval_ict"])
    v = sub.add_parser("verify")
    v.add_argument("--run-root", type=Path, required=True)
    args = ap.parse_args()
    {"build": cmd_build, "stage-input": cmd_stage_input, "stage-view": cmd_stage_view,
     "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
