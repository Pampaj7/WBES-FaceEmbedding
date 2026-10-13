#!/usr/bin/env python3
"""Ava-256, prova confermativa: parte comune della catena da spec (``conf_protocols.py``, ``conf_head.py``,
``conf_eval.py``) e verifiche della sez. 11 del protocollo.

    v3_work/unified_gt/run.sh aau/ava256/conf_common.py verify --spec aau/ava256/spec/ava256.json
    aau/recon/run_recon.sh mica aau/ava256/conf_common.py versions --spec aau/ava256/spec/ava256.json
    v3_work/unified_gt/run.sh aau/ava256/conf_common.py sha        (sha256 degli script della catena, per la sez. 11)
    v3_work/unified_gt/run.sh aau/ava256/conf_common.py host --spec <json> --role bp     (v7: nodo obbligato)
    v3_work/unified_gt/run.sh aau/ava256/conf_common.py modules --spec <json>  (v7: moduli importati da un collaudo)

Spec = un json per dominio (``aau/ava256/spec/``): lo stesso formato per Ava-256 e per i collaudi sui domini di
sviluppo, che passano dallo STESSO codice. Chiavi (oltre a quelle di ``pr_common.view`` e ``blmm.register_spec``):
``collaudo`` (true solo sui domini di sviluppo), ``freeze`` (true: verifiche della sez. 11 prima di leggere i dati),
``gt`` {fr, sr}, ``boot_seed``, ``pairs`` (sottocampione, solo collaudo), ``columns`` (sorgenti delle colonne del
valutatore), ``head`` (blocco di R6), ``detector`` (pesi e versioni attese), ``out`` {work, evidence}; v4: ``embed``
(insiemi e operatori dell'embedding, ``conf_embed.py``) e ``desc`` (descrittivi, ``conf_desc.py``).

v7, codice congelato a runtime: con ``CONF_FROZEN_COMMIT`` nell'ambiente (obbligatorio con ``freeze``) ``verify``
controlla che HEAD del repo da cui gira sia quel commit e che nessun file tracciato sia modificato (``git status
--porcelain --untracked-files=no``): su Ava-256 si gira da un worktree al commit congelato (``conf_worktree.sh``).
``loaded_modules``: sha256 dei moduli del repo importati dal processo e dai suoi figli (log di ``conf_run.py``,
``CONF_IMPORT_LOG``; senza log, ``sys.modules``), scritto in ``verify.json`` e nelle uscite. ``require_host``: nodo
obbligato di un passo (``host --role <ruolo>`` per i passi che non passano da qui: fit B).

``verify`` (sez. 11; un solo scarto -> RuntimeError): sha256 dei byte dei manifest in git, ``ava_freeze.verify_frozen()``,
``verify_frozen_supp()``, ``ava_calib.verify_calib()``, sha256 dei checkpoint (semi e, v4, ``columns.desc_models``:
C3M e123), dei pesi del rilevatore, di
``template.npz`` e ``corr/ava256.npz``, frame e soggetti dello spec contro ``frame.json`` e ``split.json``. ``versions``
(nel venv di MICA): versioni di insightface e onnxruntime contro lo spec. Senza ``freeze`` (collaudi) solo i controlli
che non toccano Ava-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
REPO = AAU.parent
# script della catena (sez. 11 del protocollo): sha256 dei byte, nell'ordine
SCRIPTS = ("aau/ava256/conf_common.py", "aau/ava256/conf_protocols.py", "aau/ava256/conf_head.py",
           "aau/ava256/conf_eval.py", "aau/ava256/conf_collaudo.py", "aau/ava256/conf.sbatch",
           "aau/ava256/spec/ava256.json", "aau/ava256/ava_freeze.py", "aau/ava256/ava_calib.py",
           "aau/competitors/protocolli/pr_common.py", "aau/competitors/protocolli/pr_landmarks.py",
           "aau/competitors/protocolli/pr_metrics.py", "aau/competitors/protocolli/pr_run.py",
           "aau/competitors/protocolli/pr_noise.py", "aau/competitors/protocolli/pr_noise_detect.py",
           "aau/baselines_mm/blmm.py", "aau/baselines_mm/blmm_scalars.py", "aau/baselines_mm/blmm_pairs.py",
           "aau/baselines_mm/blmm_template.py", "aau/baselines_param/bp.py", "aau/baselines_param/bp_fit.py",
           "aau/baselines_param/bp_fit_e1.py", "aau/baselines_param/bp_e2.py", "aau/full_matrix/fm_eval.py",
           "aau/diagnostics/d3_head.py", "aau/diagnostics/d3_e2.py", "aau/diagnostics/d3_stats.py",
           "aau/diagnostics/diag.py", "v3_work/trainer/tools/fact_paired.py",
           # v4: embedding e descrittivi fuori da conf_eval (conf_embed.py, conf_desc.py) e cio' che importano
           "aau/ava256/conf_embed.py", "aau/ava256/conf_desc.py", "aau/zs3dmm/zs_embed.py", "v3_work/trainer/eval_v3.py",
           "v3_work/trainer/global_v3.py", "v2_work/potential/areanorm_operators.py", "aau/full_matrix/fm_compute.py",
           "aau/competitors/comp_spectral.py", "aau/baselines_param/bp_paired.py", "aau/baselines_param/bp_paired_e2.py",
           "aau/baselines_mm/blmm_eval.py", "aau/diagnostics/g0_teacher.py", "aau/competitors/gw_percettivi/gp_common.py",
           "aau/competitors/gw_percettivi/gp_gw.py", "aau/competitors/gw_percettivi/gp_percettivi.py",
           "aau/ava256/spec/hifi3d_full.json", "aau/ava256/spec/hifi3d_det.json",
           # v6: ULRSSM (descrittivo); codice di terzi fuori dal repo al commit ulr_common.ULRSSM_COMMIT
           "aau/competitors/ulrssm/ulr_common.py", "aau/competitors/ulrssm/ulr_eval.py",
           # v7: lancio, worktree e spec del tracciamento (gli altri moduli: conf_common.py modules su quel collaudo)
           "aau/ava256/conf_run.py", "aau/ava256/conf_worktree.sh", "aau/ava256/spec/hifi3d_trace.json")
# v8: cloni esterni importati dalla catena attraverso un percorso del repo (link nel worktree): commit atteso, albero
# pulito (``verify``), sha256 dei file importati registrati come ``<clone>/<percorso>`` (``loaded_modules``)
EXTERNAL_CLONES = {"diffusion-net": "b1019b049597711d10259ce28250f7dfc4335a2b"}
MANIFESTS = {"aau/ava256/freeze_manifest.json": "e485fb7c44f0638ef3111343ba8741a3febae80be35f8fcdae14ae0839d5f365",
             "aau/ava256/freeze_manifest_supp.json": "c6d2157a145fa29e7a8415dc60aa1bd98e9edfdf2bc38563356cc15f16f3b02e",
             "aau/ava256/calib_manifest.json": "d3657a186525dd32c7e152356047204a2bbfad86fd65a098d9895db91b1a3af7",
             "aau/ava256/split.json": "08234346a85c84188bcba3ed54debd38b79836e77fd968e591175e5b5457b7ec"}


def rp(p) -> Path:
    """Percorso dello spec: assoluto o relativo alla radice del repo."""
    return Path(p) if Path(p).is_absolute() else REPO / p


def load(spec: Path) -> dict:
    s = json.loads(Path(spec).read_text())
    s["_path"] = str(spec)
    return s


def work(s: dict) -> Path:
    return rp(s["out"]["work"])


def evidence(s: dict) -> Path:
    return rp(s["out"]["evidence"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp() -> dict:
    return {"host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID", "none"),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_default) + "\n")
    os.replace(tmp, path)


def _default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------------ verifiche (sez. 11)

# ------------------------------------------------------------------------------------------ v7: codice e nodi

def git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def frozen_code(s: dict) -> dict:
    """HEAD = ``CONF_FROZEN_COMMIT`` e nessun file tracciato modificato; obbligatorio con ``freeze``."""
    want = os.environ.get("CONF_FROZEN_COMMIT", "")
    if not want:
        if s.get("freeze"):
            raise RuntimeError("CONF_FROZEN_COMMIT assente: su Ava-256 si gira dal worktree del commit congelato")
        return {"checked": False}
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    rep = {"checked": True, "repo": str(REPO), "head": head, "frozen_commit": want, "dirty": dirty.splitlines()[:50]}
    if not head.startswith(want) or len(want) < 7:
        raise RuntimeError(f"HEAD {head} diverso dal commit congelato {want}")
    if dirty:
        raise RuntimeError(f"file tracciati modificati nel repo da cui si gira: {dirty.splitlines()[:5]}")
    return rep


def external_clones() -> dict:
    """Commit e albero pulito di ogni clone di ``EXTERNAL_CLONES``; uno scarto -> RuntimeError."""
    out = {}
    for name, want in EXTERNAL_CLONES.items():
        root = (REPO / name).resolve()
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        d = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
        head, dirty = r.stdout.strip(), d.stdout.strip()
        out[name] = {"path": str(root), "head": head, "dirty": dirty.splitlines()[:20]}
        if r.returncode or head != want:
            raise RuntimeError(f"clone {name} ({root}): commit {head or r.stderr.strip()}, atteso {want}")
        if d.returncode or dirty:
            raise RuntimeError(f"clone {name} ({root}): file modificati {dirty.splitlines()[:5]}")
    return out


def short_host() -> str:
    return socket.gethostname().split(".")[0]


def require_host(node: str, what: str) -> None:
    """Il passo gira solo sul nodo ``node`` (sez. 7, 9, 10: fit B, ArcFace, testa di R6)."""
    if short_host() != node.split(".")[0]:
        raise SystemExit(f"{what}: nodo {short_host()}, obbligato {node}")


def loaded_modules() -> dict:
    """{percorso: sha256} dei moduli importati sotto la radice del repo (o del worktree) da questo processo e dai figli:
    il log di ``conf_run.py`` se c'e' (``CONF_IMPORT_LOG``), piu' ``sys.modules``. v8: un modulo il cui percorso sta
    sotto la radice ma che RISOLVE fuori (link del worktree) e' un ERRORE, salvo i cloni di ``EXTERNAL_CLONES``
    (registrati come ``<clone>/<percorso nel clone>``, commit controllato da ``external_clones``) e le librerie
    installate dei venv (``site-packages``: non codice del progetto, come i pacchetti del container)."""
    log = os.environ.get("CONF_IMPORT_LOG")
    paths = set()
    if log and Path(log).exists():
        paths |= {x for x in Path(log).read_text().split("\n") if x}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f:
            paths.add(str(Path(f).absolute()))
    root, rroot = str(REPO), REPO.resolve()
    clones = {n: (REPO / n).resolve() for n in EXTERNAL_CLONES}
    out, bad = {}, []
    for f in sorted(paths):
        if not f.endswith(".py") or not (f == root or f.startswith(root + os.sep)) or not Path(f).exists():
            continue
        if "site-packages" in Path(f).parts or "dist-packages" in Path(f).parts:
            continue
        r = Path(f).resolve()
        if r.is_relative_to(rroot):
            out[str(r.relative_to(rroot))] = sha256(r)
            continue
        name = next((n for n, c in clones.items() if r.is_relative_to(c)), None)
        if name is None:
            bad.append(f"{f} -> {r}")
            continue
        out[f"{name}/{r.relative_to(clones[name])}"] = sha256(r)
    if bad:
        raise RuntimeError(f"moduli sotto il repo che risolvono fuori, non in EXTERNAL_CLONES: {bad[:5]}")
    return out


def check(cond: bool, what: str, errors: list) -> None:
    if not cond:
        errors.append(what)


def verify(s: dict) -> dict:
    """Le verifiche della sez. 11 che lo spec richiede; un solo scarto -> RuntimeError. Ritorna il resoconto."""
    err, rep = [], {"spec": s["_path"], "freeze": bool(s.get("freeze"))}
    rep["code"] = frozen_code(s)                                           # v7: commit congelato, albero pulito
    rep["external_clones"] = external_clones()                             # v8: diffusion-net al suo commit
    det = s.get("detector", {})
    for f, h in det.get("weights_sha256", {}).items():                     # pesi del rilevatore (anche nei collaudi)
        p = Path.home() / ".insightface" / "models" / "buffalo_l" / f
        got = sha256(p) if p.exists() else "assente"
        rep.setdefault("detector_weights", {})[f] = got
        check(got == h, f"pesi del rilevatore {f}: {got}", err)
    arms = {**s.get("columns", {}).get("models", {}), **(s.get("columns", {}).get("desc_models") or {})}
    for arm, m in arms.items():                                            # checkpoint dei semi e di C3M (sez. 9)
        if m.get("checkpoint_sha256"):
            p = rp(m["checkpoint"])
            got = sha256(p) if p.exists() else "assente"
            rep.setdefault("checkpoints", {})[arm] = got
            check(got == m["checkpoint_sha256"], f"checkpoint {arm}: {got}", err)
    if s.get("freeze"):
        sys.path.insert(0, str(THIS))
        import ava_calib
        import ava_freeze
        for f, h in MANIFESTS.items():
            got = sha256(REPO / f)
            rep.setdefault("manifests", {})[f] = got
            check(got == h, f"manifest {f}: {got}", err)
        if not err:                                                        # i manifest prima delle impronte
            ava_freeze.verify_frozen()
            ava_freeze.verify_frozen_supp()
            ava_calib.verify_calib()
            rep["verify_frozen"] = rep["verify_frozen_supp"] = rep["verify_calib"] = "uguali"
        for key, h in s.get("data_sha256", {}).items():                   # template.npz, corr/ava256.npz
            got = sha256(rp(key))
            rep.setdefault("data", {})[key] = got
            check(got == h, f"{key}: {got}", err)
        fr = json.loads(rp(s["frame_file"]).read_text())
        check(all(fr[k] == s["frame"][k] for k in ("u", "R", "t")), "frame dello spec diverso da frame.json", err)
        sp = json.loads((REPO / "aau/ava256/split.json").read_text())
        check(sorted(s["subjects"]) == sorted(sp["evaluation_view_ids"]), "soggetti dello spec != valutati", err)
        check(sorted(s["template"]["subjects"]) == sorted(sp["calibration_view_ids"]),
              "template dello spec != calibrazione", err)
        check(sorted(s["head"]["subjects"]) == sorted(sp["calibration_view_ids"]), "blocco di R6 != calibrazione", err)
    rep["errors"] = err
    if err:
        raise RuntimeError(f"verifiche della sez. 11 fallite: {err}")
    return {**rep, "modules_sha256": loaded_modules(), **stamp()}


def versions(s: dict) -> dict:
    """Versioni del rilevatore (venv di MICA) contro lo spec; uno scarto -> RuntimeError."""
    import importlib.metadata as md
    want = s.get("detector", {}).get("versions", {})
    got = {k: md.version(k) for k in want}
    if got != want:
        raise RuntimeError(f"versioni del rilevatore {got}, attese {want}")
    return {"versions": got, **stamp()}


def script_sha() -> dict:
    return {f: sha256(REPO / f) for f in SCRIPTS}


def collected_modules(s: dict) -> dict:
    """Unione dei moduli registrati da ``conf_run.py`` in ``<evidenza>/modules/*.json`` (un collaudo intero): {percorso:
    sha256}; uno stesso file con due sha diversi e' un errore."""
    out = {}
    for p in sorted((evidence(s) / "modules").glob("*.json")):
        for f, h in json.loads(p.read_text())["modules_sha256"].items():
            if out.get(f, h) != h:
                raise RuntimeError(f"{f}: due sha256 diversi fra i passi ({p.name})")
            out[f] = h
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("verify", "versions", "sha", "host", "modules"))
    p.add_argument("--spec", type=Path, default=None)
    p.add_argument("--role", default="")
    a = p.parse_args()
    if a.step == "sha":
        print(json.dumps(script_sha(), indent=1))
        return
    s = load(a.spec)
    if a.step == "host":
        require_host(s["nodes"][a.role], a.role)
        print(f"[conf] nodo {short_host()} = {s['nodes'][a.role]} ({a.role})", flush=True)
        return
    if a.step == "modules":
        m = collected_modules(s)
        atomic_json(evidence(s) / "modules_union.json", {"modules_sha256": m, "n": len(m), **stamp()})
        print(json.dumps(m, indent=1))
        return
    rep = verify(s) if a.step == "verify" else versions(s)
    atomic_json(evidence(s) / f"{a.step}.json", rep)
    print(f"[conf] {a.step} {s['name']}: OK {json.dumps(rep)[:400]}", flush=True)


if __name__ == "__main__":
    main()
