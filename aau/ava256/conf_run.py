#!/usr/bin/env python3
"""Ava-256, prova confermativa (v7): lancia uno script della catena e registra OGNI modulo del repo che importa, anche
nei processi figli (worker ``fork`` dei Pool), poi ne scrive gli sha256 e, con ``CONF_FROZEN_COMMIT``, li confronta
coi file del commit congelato.

    <runner> aau/ava256/conf_run.py <script.py> [argomenti dello script]          (conf.sbatch, ogni passo Python)

Registrazione: un finder in testa a ``sys.meta_path`` annota il percorso di ogni modulo trovato sotto la radice del
repo nel file ``CONF_IMPORT_LOG`` (uno per job, in append: i figli ``fork`` ereditano il finder e l'ambiente; i figli
lanciati con ``subprocess`` passano di nuovo da qui, ``conf_embed.py``). Alla fine, anche se lo script esce con errore:
``<evidenza>/modules/<script>-<job>-<pid>.json`` (spec da ``CONF_SPEC``) con gli sha256 dei moduli del log
(``conf_common.loaded_modules``). Con ``CONF_FROZEN_COMMIT``: ogni modulo deve essere tracciato in git e uguale al
suo blob nel commit congelato (``git rev-parse <commit>:<percorso>`` contro ``git hash-object``); uno scarto ->
uscita 3 (il job si ferma) e l'elenco degli scarti nel json. L'elenco della sez. 11.1 si ricava da questi json di un
collaudo (``conf_common.py modules``).
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import runpy
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
# un log per passo (pid del primo conf_run.py); i figli lanciati con subprocess lo ereditano dall'ambiente
LOG = os.environ.setdefault("CONF_IMPORT_LOG", f"/tmp/wbes_conf_imports_{os.environ.get('SLURM_JOB_ID', 'x')}_{os.getpid()}.txt")


class Tracer(importlib.abc.MetaPathFinder):
    """Annota i moduli del repo; non carica niente (ritorna None, decide il resto di ``sys.meta_path``)."""

    def find_spec(self, name, path, target=None):
        try:
            spec = importlib.machinery.PathFinder.find_spec(name, path)
        except Exception:  # noqa: BLE001  (il finder non deve mai rompere un import)
            return None
        o = getattr(spec, "origin", None) if spec else None
        if o and o.startswith(str(REPO)) and o.endswith(".py"):
            with open(LOG, "a") as fh:
                fh.write(str(Path(o).resolve()) + "\n")
        return None


def record(script: Path) -> list:
    """Scrive il json dei moduli; ritorna gli scarti dal commit congelato (vuoto se non controllato)."""
    sys.path.insert(0, str(THIS))
    import conf_common as cc
    mods = cc.loaded_modules()
    bad = []
    want = os.environ.get("CONF_FROZEN_COMMIT", "")
    if want:
        for f in mods:
            try:
                ref = cc.git("rev-parse", f"{want}:{f}")
            except RuntimeError:
                bad.append(f"{f}: non tracciato nel commit {want}")
                continue
            if cc.git("hash-object", f) != ref:
                bad.append(f"{f}: diverso dal commit {want}")
    spec = os.environ.get("CONF_SPEC")
    if spec:
        s = cc.load(cc.rp(spec))
        cc.atomic_json(cc.evidence(s) / "modules" / f"{script.stem}-{os.environ.get('SLURM_JOB_ID', 'x')}-{os.getpid()}"
                       f".json", {"script": str(script), "argv": sys.argv[1:], "modules_sha256": mods,
                                  "frozen_commit": want or None, "mismatch": bad, **cc.stamp()})
    return bad


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("uso: conf_run.py <script.py> [argomenti]")
    sys.meta_path.insert(0, Tracer())
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script)] + sys.argv[2:]
    sys.path.insert(0, str(script.parent))
    code = 0
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if not isinstance(e.code, int) and e.code is not None:
            print(e.code, file=sys.stderr)
    finally:
        bad = record(script)
    if bad:
        print(f"[conf-run] moduli diversi dal commit congelato: {bad[:5]}", file=sys.stderr, flush=True)
        raise SystemExit(3)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
