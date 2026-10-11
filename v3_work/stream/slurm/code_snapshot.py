#!/usr/bin/env python3
"""Copia congelata del codice di un run a streaming: riavvii e continuazioni eseguono i file della prima partenza.

    python3 v3_work/stream/slurm/code_snapshot.py make <repo> <dest>       (massive.sbatch, prima partenza)
    python3 <dest>/v3_work/stream/slurm/code_snapshot.py verify <dest>     (massive.sbatch, riavvio o continuazione)
    python3 <dest>/v3_work/stream/slurm/code_snapshot.py version <dest>    (code_version del manifesto)

<dest> ha la forma del repo: i file di codice (*.py, *.sh, *.sbatch tracciati o non ignorati da git) copiati come sono
nell'albero di lavoro, ogni altra voce delle loro directory un link simbolico al repo vivo (dati, venv, diffusion-net,
aau/runs ...): i percorsi calcolati da __file__ (REPO_ROOT di common.py, sources.py, views.py ...) trovano il codice
nella copia e i dati di sempre. Niente .git: la versione del codice e' ``code_version`` del manifesto (commit, piu'
"+modifiche" se un file di v3_work differisce dal commit, la regola di producer.py), che massive.sbatch passa ai
produttori (WBES_CODE_VERSION, nella ricetta di ogni shard). ``MANIFEST.json`` (commit, file diversi dal commit,
sha256 per file) si scrive per ultimo e la copia si costruisce a parte e si rinomina: c'e' solo se e' completa.
``.gitignore`` con '*' (sotto aau/runs git mostrerebbe i .json). Solo stdlib: gira fuori dal container.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

PATTERNS = ("*.py", "*.sh", "*.sbatch")
OWN = (".gitignore", "MANIFEST.json")      # scritti qui nella radice: mai link (si scriverebbe nel repo vivo)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True, timeout=300).stdout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make(repo: Path, dest: Path) -> dict:
    repo, dest = repo.resolve(), dest.parent.resolve() / dest.name
    files = sorted({f for f in git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
                                   *PATTERNS).split("\0") if f and (repo / f).is_file()})
    head = git(repo, "rev-parse", "HEAD").strip()
    st = git(repo, "status", "--porcelain", "-z", "--untracked-files=all", "--", *PATTERNS)
    dirty = sorted({e[3:] for e in st.split("\0") if len(e) > 3} & set(files))
    dirs = {""}
    for f in files:
        p = os.path.dirname(f)
        while p and p not in dirs:
            dirs.add(p)
            p = os.path.dirname(p)
    tmp = dest.with_name(f".{dest.name}.tmp{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    fs, links = set(files), 0
    for d in sorted(dirs, key=lambda x: (x.count("/"), x)):
        (tmp / d).mkdir(parents=True, exist_ok=True)
        for e in sorted(os.listdir(repo / d)):
            rel = f"{d}/{e}" if d else e
            # niente .git, ne' __pycache__ (python scrive i .pyc nella copia), ne' la copia stessa
            if rel in dirs or rel in fs or rel in (".git",) + OWN or e == "__pycache__" \
                    or str(repo / rel) in (str(dest), str(tmp)):
                continue
            os.symlink(repo / rel, tmp / rel)
            links += 1
    sums = {}
    for f in files:
        if (tmp / f).is_symlink():
            raise SystemExit(f"[code] ERRORE: {tmp / f} e' un link, la copia scriverebbe nel repo vivo")
        shutil.copy2(repo / f, tmp / f)
        sums[f] = sha256(tmp / f)
    man = {"repo": str(repo), "head": head,
           "code_version": head + ("+modifiche" if any(f.startswith("v3_work/") for f in dirty) else ""),
           "dirty": dirty, "patterns": list(PATTERNS), "files": len(files), "links": links,
           "bytes": sum((tmp / f).stat().st_size for f in files), "time": time.strftime("%F %T"),
           "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID", ""), "sha256": sums}
    for name in OWN:
        if (tmp / name).is_symlink() or (tmp / name).exists():
            raise SystemExit(f"[code] ERRORE: {tmp / name} esiste gia'")
    (tmp / ".gitignore").write_text("*\n")
    (tmp / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    if dest.exists():                  # copia incompleta di una prima partenza interrotta (senza manifesto)
        shutil.rmtree(dest)
    os.rename(tmp, dest)
    return man


def verify(dest: Path) -> tuple[dict, list[str]]:
    """File della copia diversi dal manifesto (mancanti, link al posto della copia, sha256 diverso)."""
    man = json.loads((dest / "MANIFEST.json").read_text())
    bad = [f for f, h in man["sha256"].items()
           if (dest / f).is_symlink() or not (dest / f).is_file() or sha256(dest / f) != h]
    return man, bad


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("make", "verify", "version") or (sys.argv[1] == "make") != (len(sys.argv) == 4):
        raise SystemExit(__doc__.split("\n\n")[1])
    if sys.argv[1] == "make":
        man = make(Path(sys.argv[2]), Path(sys.argv[3]))
        print(f"[code] copia di {man['files']} file ({man['bytes'] / 2 ** 20:.1f} MiB) e {man['links']} link al repo, "
              f"commit {man['code_version']}" + (f" ({len(man['dirty'])} file diversi dal commit)" if man["dirty"] else "")
              + f" -> {sys.argv[3]}", flush=True)
    elif sys.argv[1] == "verify":
        man, bad = verify(Path(sys.argv[2]))
        if bad:
            raise SystemExit(f"[code] ERRORE: {len(bad)} file di {sys.argv[2]} diversi dal manifesto: {bad[:10]}")
        print(f"[code] {sys.argv[2]}: {len(man['sha256'])} file identici al manifesto (commit {man['code_version']}, "
              f"copia del {man['time']}, job {man['job']})", flush=True)
    else:
        print(json.loads((Path(sys.argv[2]) / "MANIFEST.json").read_text())["code_version"])


if __name__ == "__main__":
    main()
