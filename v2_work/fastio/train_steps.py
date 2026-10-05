#!/usr/bin/env python
"""Il trainer v1 a NUMERO DI PASSI FISSO, con staging a blocchi e operatori dal pre-pass.

Accanto a train_fast.py, che resta invariato: senza ``--total-steps`` questo file chiama
``train_fast.main()`` e basta. Piano e motivazioni in aau/data_scale/PLAN.md.

Flag propri (tutto il resto va a train_fast.py e da li' al trainer v1, ricetta invariata):
    --total-steps T        passi di ottimizzazione totali (0 = comportamento di sempre)
    --steps-per-epoch S    passi per "epoca" (granularita' di log, eval online, scheduler e
                           checkpoint). 0 = la lunghezza naturale dell'epoca v1 sui soggetti di
                           training. Le epoche diventano ceil(T/S); l'ultima e' troncata a T.
    --split-json J         split ESPLICITO {"train": [...], "heldout": [...]} al posto di
                           ``rebuild_subject_split``, che ricalcolerebbe lo split sull'unione
                           dei soggetti della vista (aggiungere identita' sposterebbe nel
                           training soggetti oggi di test). Obbligatorio con --data-spec.
    --frozen-heldout F     aau/data_scale/heldout_frozen.json: il run si ferma se un soggetto
                           di training e' in quella lista
    --data-spec D          dati a blocchi, JSON:
                             {"views": [dir con operatori gia' calcolati],
                              "tars": [shard di gen_ict_shard.py], "geom_dirs": [dir V/F],
                              "labels": [etichette tenute] | null,
                              "convention": "areanorm" | "robust_area1",
                              "n_blocks": K, "block_seed": 0}
                           I soggetti di training sono divisi in K blocchi (permutazione con
                           block_seed); ogni blocco e' preparato su --stage-root (symlink per le
                           viste, ``aau/data_scale/prepass_ops.py`` per tar e geometria), messo
                           nella cache RAM di train_fast (``CachedDataset``) e addestrato per
                           ceil(epoche/K) epoche, mentre il blocco successivo si prepara in un
                           processo a parte. I soggetti dell'eval online sono preparati una
                           volta sola e restano residenti.
    --stage-root R         radice su /tmp (RAM: conta contro --mem)
    --prepass-proc P       processi del pre-pass (default 16)
    --domain-blocked       entra nelle patch di v2_work/train_v2 (batch a dominio singolo,
                           GT con NaN fuori blocco); --eval_domain come li'

Cosa resta identico per costruzione: modello, loss, ottimizzatore, scheduler, augmentation,
campionamento delle mesh per soggetto, eval online, checkpoint. Cambia solo QUALI soggetti
riceve ogni epoca (``_train_epoch`` e' chiamata con un sottoinsieme di S*batch soggetti del
blocco residente) e da dove arrivano i campioni. Con un solo blocco, S naturale e T multiplo
di S, la funzione d'epoca riceve la stessa lista di soggetti di v1: la corsa e' quella di v1.

    aau/run.sh v2_work/fastio/train_steps.py --total-steps 105600 --split-json split.json \
        --frozen-heldout aau/data_scale/heldout_frozen.json --data-spec spec.json \
        --stage-root /tmp/$SLURM_JOB_ID/stage --prepass-proc 24 \
        --cache-residency ram --frame current --data_dir /tmp/$SLURM_JOB_ID/stage ... (ricetta v1)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "v2_work/train_v2"))

import train_fast  # noqa: E402  (mette sul path robustness e fast_data)

PREPASS = REPO_ROOT / "aau/data_scale/prepass_ops.py"
STATE: dict = {"steps": 0}


# --- dati a blocchi -----------------------------------------------------------------------------

def _split_name(name: str) -> tuple[str, str]:
    sid, label = name[:-4].split("_GTready_", 1)
    return sid, label


def collect_sources(spec: dict) -> dict[str, tuple[str, str]]:
    """nome file -> (tipo, sorgente) per ogni mesh della spec, filtrata per etichetta."""
    labels = set(spec["labels"]) if spec.get("labels") else None
    out: dict[str, tuple[str, str]] = {}

    def add(name: str, kind: str, src: str) -> None:
        if not name.endswith(".npz") or "_GTready_" not in name:
            return
        if labels is not None and _split_name(name)[1] not in labels:
            return
        if name in out:
            raise SystemExit(f"{name} compare in due sorgenti: {out[name][1]} e {src}")
        out[name] = (kind, src)

    for d in spec.get("views", []):
        for n in sorted(os.listdir(d)):
            add(n, "view", str(Path(d) / n))
    for t in spec.get("tars", []):
        with tarfile.open(t) as tar:
            for n in tar.getnames():
                add(n, "tar", str(t))
    for d in spec.get("geom_dirs", []):
        for n in sorted(os.listdir(d)):
            add(n, "geom", str(Path(d) / n))
    return out


class BlockedDataset:
    """Superficie di GTReadyDatasetNPZ (``files``, ``__len__``, ``__getitem__``) su blocchi.

    ``files`` e' l'elenco COMPLETO e ordinato di tutte le mesh (serve a mappa dei soggetti,
    topologie e GT, che il trainer costruisce una volta). In memoria c'e' solo il blocco
    residente piu' l'insieme dell'eval online; chiedere un campione fuori da entrambi e' un
    errore, non un caricamento silenzioso da disco.
    """

    def __init__(self, sources: dict[str, tuple[str, str]], cfg: dict) -> None:
        self.files = sorted(sources)
        self.sources = sources
        self.cfg = cfg
        self._pos = {n: i for i, n in enumerate(self.files)}
        self._parts: list = []        # [(CachedDataset, {idx_globale: idx_locale}), ...]
        self.resident_block = -1

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        for ds, local in self._parts:
            j = local.get(int(idx))
            if j is not None:
                return ds[j]
        raise KeyError(f"{self.files[int(idx)]}: non e' nel blocco residente "
                       f"({self.resident_block}) ne' nell'eval online")

    # staging ------------------------------------------------------------------------------
    def stage(self, names: list[str], dest: Path, wait: bool = True):
        """Vista piatta in ``dest``: symlink per le viste, pre-pass per tar e geometria."""
        dest.mkdir(parents=True, exist_ok=True)
        need_tars, need_geom, subjects = set(), set(), set()
        for n in names:
            kind, src = self.sources[n]
            if kind == "view":
                p = dest / n
                if not p.exists():
                    p.symlink_to(Path(src).resolve())
            else:
                (need_tars if kind == "tar" else need_geom).add(src if kind == "tar" else str(Path(src).parent))
                subjects.add(_split_name(n)[0])
        if not subjects:
            return None
        subj_file = dest.parent / f"{dest.name}.subjects.txt"
        subj_file.write_text("\n".join(sorted(subjects)) + "\n")
        labels = ",".join(self.cfg["labels"]) if self.cfg.get("labels") else ""
        cmd = [sys.executable, str(PREPASS), "--out-dir", str(dest), "--subjects", str(subj_file),
               "--convention", self.cfg.get("convention", "areanorm"),
               "--n-proc", str(self.cfg["prepass_proc"])]
        if need_tars:
            cmd += ["--tars", *sorted(need_tars)]
        if need_geom:
            cmd += ["--geom-dirs", *sorted(need_geom)]
        if labels:
            cmd += ["--labels", labels]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        log = open(dest.parent / f"{dest.name}.prepass.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        if wait:
            self.finish(proc, dest)
            return None
        return proc

    def finish(self, proc, dest: Path) -> None:
        if proc is None:
            return
        t0 = time.time()
        rc = proc.wait()
        if rc != 0:
            raise SystemExit(f"pre-pass di {dest} fallito (rc={rc}), log in {dest.parent / (dest.name + '.prepass.log')}")
        print(f"[steps] pre-pass {dest.name} pronto (atteso {time.time() - t0:.0f}s)", flush=True)

    def load(self, dest: Path, names: list[str]):
        from fast_data import CachedDataset
        c = self.cfg
        missing = [n for n in names if not (dest / n).exists()]
        if missing:
            raise SystemExit(f"{dest}: {len(missing)} mesh mancanti dopo il pre-pass (prima {missing[0]})")
        ds = CachedDataset(dest, workers=c["cache_workers"], residency=c["cache_residency"],
                           device=c["device"], max_gb=c["cache_max_gb"])
        local = {self._pos[n]: j for j, n in enumerate(ds.files) if n in self._pos}
        return ds, local


# --- patch del trainer --------------------------------------------------------------------------

class _Fwd:
    """Inoltra ogni attributo a ``target`` tranne quelli sovrascritti (come train_v2._Fwd)."""

    def __init__(self, target, **overrides):
        self._target, self._overrides = target, overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._target, name)


def epoch_subset(pool: list[str], steps: int, batch: int, seed: int, domain_blocked: bool) -> list[str]:
    """``steps * batch`` soggetti del blocco, che v1 trasformera' in esattamente ``steps`` batch.

    Permutazione seminata per epoca, ciclica se il blocco ha meno soggetti di quelli chiesti
    (un soggetto non compare due volte nella stessa epoca finche' il blocco basta). Con il
    batching a dominio singolo ogni dominio riceve un multiplo di ``batch`` soggetti, con
    passi ripartiti in proporzione ai soggetti del dominio (resto maggiore): train_v2
    tronca ogni dominio al multiplo di ``batch``, e senza questo l'epoca farebbe meno passi.
    """
    rng = np.random.default_rng(seed)
    if not domain_blocked:
        groups = {"": list(pool)}
        alloc = {"": steps}
    else:
        from train_v2 import domain_of
        groups = {}
        for s in pool:
            groups.setdefault(domain_of(s), []).append(s)
        quota = {d: steps * len(g) / len(pool) for d, g in groups.items()}
        alloc = {d: int(q) for d, q in quota.items()}
        for d in sorted(quota, key=lambda d: quota[d] - alloc[d], reverse=True)[: steps - sum(alloc.values())]:
            alloc[d] += 1
    out = []
    for d in sorted(groups):
        n_need = alloc[d] * batch
        if n_need == 0:
            continue
        perm = rng.permutation(np.array(sorted(groups[d]), dtype=object)).tolist()
        out += (perm * math.ceil(n_need / len(perm)))[:n_need]
    if len(set(out)) != len(out):
        # un soggetto ripetuto nella stessa epoca: v1 lo tratterebbe come due soggetti distinti
        # nella stessa batch solo se cadessero insieme, ma la GT avrebbe diagonale 0 fuori posto
        raise SystemExit(f"blocco troppo piccolo: {len(set(out))} soggetti per {len(out)} posti; "
                         "riduci --steps-per-epoch o i blocchi")
    return sorted(out)


def natural_steps(train_subjects: list[str], batch: int, domain_blocked: bool) -> int:
    """Passi di un'epoca v1 sugli stessi soggetti (chunk con almeno 2 soggetti)."""
    if domain_blocked:
        from train_v2 import domain_of
        counts: dict[str, int] = {}
        for s in train_subjects:
            counts[domain_of(s)] = counts.get(domain_of(s), 0) + 1
        return sum(n // batch for n in counts.values())
    n = len(train_subjects)
    return n // batch + (1 if n % batch >= 2 else 0)


def install(known: argparse.Namespace) -> None:
    import robustness.data_utils as du
    import robustness.train_runner as tr

    cfg: dict = {}
    orig_parse, orig_split, orig_epoch = tr.parse_args, tr.rebuild_subject_split, tr._train_epoch

    def parse_args():
        args = orig_parse()
        cfg["args"] = args
        cfg["epochs_cli"] = int(args.epochs)
        return args
    tr.parse_args = parse_args

    frozen = set()
    if known.frozen_heldout:
        fz = json.loads(Path(known.frozen_heldout).read_text())
        frozen = set(fz["bfm"]) | set(fz["ict_view"])
    split = json.loads(Path(known.split_json).read_text()) if known.split_json else None

    def rebuild_subject_split(subjects, eval_fraction, seed, max_subjects):
        if split is None:
            train, held = orig_split(subjects=subjects, eval_fraction=eval_fraction,
                                     seed=seed, max_subjects=max_subjects)
        else:
            have = set(subjects)
            train = sorted(s for s in split["train"] if s in have)
            held = sorted(s for s in split["heldout"] if s in have)
            if set(train) & set(held):
                raise SystemExit("split-json: soggetti sia in train sia in heldout")
            print(f"[steps] split esplicito {known.split_json}: train={len(train)} heldout={len(held)} "
                  f"(fuori dallo split: {len(have) - len(train) - len(held)})", flush=True)
        leak = sorted(set(train) & frozen)
        if leak:
            raise SystemExit(f"GUARDIA HELD-OUT: {len(leak)} soggetti di test congelati nel training "
                             f"(primo {leak[0]}), vedi {known.frozen_heldout}")
        if frozen:
            print(f"[steps] guardia held-out OK: nessuno dei {len(frozen)} soggetti congelati e' di training",
                  flush=True)
        args = cfg["args"]
        B = int(args.batch_subjects)
        S = int(known.steps_per_epoch) or natural_steps(train, B, known.domain_blocked)
        epochs = math.ceil(int(known.total_steps) / S)
        if int(args.save_every) >= cfg["epochs_cli"]:
            args.save_every = epochs          # la ricetta salva solo l'ultima epoca
        args.epochs = epochs
        cfg.update(S=S, B=B, train=train, eval=held, epochs=epochs)
        print(f"[steps] T={known.total_steps} passi, S={S} per epoca, epoche={epochs}", flush=True)
        if cfg.get("blocked") is not None:
            _stage_initial(held, train)
        return train, held
    tr.rebuild_subject_split = rebuild_subject_split

    def _stage_initial(held, train):
        ds: BlockedDataset = cfg["blocked"]
        args = cfg["args"]
        root = Path(known.stage_root)
        # stessi soggetti che v1 scegliera' per l'eval online (stessa funzione, stesso seme)
        online = tr._select_online_eval_subjects(eval_subjects=held,
                                                 max_subjects_eval_train=args.max_subjects_eval_train,
                                                 seed=args.seed)
        eval_names = [n for n in ds.files if _split_name(n)[0] in set(online)]
        rng = np.random.default_rng(int(cfg["spec"].get("block_seed", 0)))
        perm = rng.permutation(np.array(train, dtype=object)).tolist()
        K = int(cfg["spec"].get("n_blocks", 1))
        cfg["blocks"] = [sorted(perm[k::K]) for k in range(K)]
        cfg["epochs_per_block"] = math.ceil(cfg["epochs"] / K)
        print(f"[steps] {K} blocchi da {[len(b) for b in cfg['blocks']]} soggetti, "
              f"{cfg['epochs_per_block']} epoche ciascuno; eval online {len(online)} soggetti", flush=True)
        ds.stage(eval_names, root / "eval")
        ds._parts.append(ds.load(root / "eval", eval_names))
        shutil.rmtree(root / "eval", ignore_errors=True)
        _switch_block(0)

    def _names_of(k: int) -> list[str]:
        subj = set(cfg["blocks"][k])
        return [n for n in cfg["blocked"].files if _split_name(n)[0] in subj]

    def _switch_block(k: int):
        ds: BlockedDataset = cfg["blocked"]
        root = Path(known.stage_root)
        dest = root / f"block{k:03d}"
        if cfg.get("pending") and cfg["pending"][0] == k:
            ds.finish(cfg["pending"][1], dest)
        else:
            ds.stage(_names_of(k), dest)
        if len(ds._parts) > 1:                    # libera il blocco precedente prima di caricare
            ds._parts.pop()
            shutil.rmtree(root / f"block{ds.resident_block:03d}", ignore_errors=True)
        t0 = time.time()
        ds._parts.append(ds.load(dest, _names_of(k)))
        # tutti i campioni del blocco sono ora decodificati in RAM: la copia su /tmp (che conta
        # contro --mem) non serve piu'. Senza questo il picco e' cache + operatori su disco
        # (OOM a 120G sui 400 soggetti BFM, job 1055893/1055894)
        shutil.rmtree(dest, ignore_errors=True)
        ds.resident_block = k
        print(f"[steps] blocco {k} residente ({len(cfg['blocks'][k])} soggetti) in {time.time() - t0:.0f}s",
              flush=True)
        nxt = k + 1
        if nxt < len(cfg["blocks"]):
            cfg["pending"] = (nxt, ds.stage(_names_of(nxt), root / f"block{nxt:03d}", wait=False))

    def _train_epoch(*a, **kw):
        epoch = int(kw["epoch"])
        S, B = cfg["S"], cfg["B"]
        remaining = int(known.total_steps) - STATE["steps"]
        steps = min(S, remaining)
        if cfg.get("blocked") is not None:
            k = min((epoch - 1) // cfg["epochs_per_block"], len(cfg["blocks"]) - 1)
            if k != cfg["blocked"].resident_block:
                _switch_block(k)
            pool = cfg["blocks"][k]
        else:
            pool = list(kw["train_subjects"])
        if len(cfg["blocks"] if cfg.get("blocked") is not None else [0]) == 1 and steps == S \
                and S == natural_steps(list(kw["train_subjects"]), B, known.domain_blocked):
            subset = kw["train_subjects"]          # epoca v1 identica: stessa lista
        else:
            subset = epoch_subset(pool, steps, B, int(cfg["args"].seed) + 31 + epoch,
                                  known.domain_blocked)
        kw["train_subjects"] = subset
        before = STATE["steps"]
        stats = orig_epoch(*a, **kw)
        if STATE["steps"] - before != steps:
            # v1 salta i batch senza coppie valide: lo si dice, e l'epoca successiva recupera
            print(f"[steps] AVVISO epoca {epoch}: {STATE['steps'] - before} passi eseguiti, attesi {steps}",
                  flush=True)
        return stats
    tr._train_epoch = _train_epoch

    # conta i passi VERI: ogni optimizer.step() dell'ottimizzatore che costruisce run_training
    class CountingAdam(tr.optim.Adam):
        def step(self, *a, **kw):
            STATE["steps"] += 1
            return super().step(*a, **kw)
    tr.optim = _Fwd(tr.optim, Adam=CountingAdam)

    if known.data_spec:
        if not known.split_json:
            raise SystemExit("--data-spec richiede --split-json")
        spec = json.loads(Path(known.data_spec).read_text())
        cfg["spec"] = spec
        sources = collect_sources(spec)
        cfg["blocked"] = BlockedDataset(sources, {
            "labels": spec.get("labels"), "convention": spec.get("convention", "areanorm"),
            "prepass_proc": int(known.prepass_proc), "cache_workers": int(known.cache_workers),
            "cache_residency": known.cache_residency, "cache_max_gb": float(known.cache_max_gb),
            "device": None})
        factory = lambda *_a, **_k: cfg["blocked"]  # noqa: E731
        du.GTReadyDataset = factory
        tr.GTReadyDataset = factory
        print(f"[steps] data-spec {known.data_spec}: {len(sources)} mesh, staging in {known.stage_root}",
              flush=True)


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--total-steps", type=int, default=0)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--split-json", default="")
    p.add_argument("--frozen-heldout", default="")
    p.add_argument("--data-spec", default="")
    p.add_argument("--stage-root", default="")
    p.add_argument("--prepass-proc", type=int, default=16)
    p.add_argument("--domain-blocked", action="store_true")
    p.add_argument("--eval_domain", default="")
    known, rest = p.parse_known_args()

    if known.total_steps <= 0:
        if known.data_spec or known.split_json or known.domain_blocked:
            raise SystemExit("--data-spec/--split-json/--domain-blocked richiedono --total-steps")
        sys.argv = [sys.argv[0]] + rest
        train_fast.main()                         # comportamento di sempre
        return

    # flag di train_fast che servono anche qui; restano in `rest` per train_fast
    q = argparse.ArgumentParser(add_help=False)
    q.add_argument("--cache-residency", default="ram")
    q.add_argument("--cache-workers", type=int, default=16)
    q.add_argument("--cache-max-gb", type=float, default=900.0)
    q.add_argument("--no-cache", action="store_true")
    fast, _ = q.parse_known_args(rest)
    for k, v in vars(fast).items():
        setattr(known, k, v)
    if known.data_spec:
        if not known.stage_root:
            raise SystemExit("--data-spec richiede --stage-root")
        if "--no-cache" not in rest:
            rest = rest + ["--no-cache"]          # la cache la costruiscono i blocchi

    import robustness.train_runner  # noqa: F401
    install(known)
    sys.argv = [sys.argv[0]] + rest
    if known.domain_blocked:
        import train_v2
        batch = int(rest[rest.index("--batch_subjects") + 1])
        with train_v2._patched_v1(batch, str(known.eval_domain)):
            train_fast.main()
    else:
        train_fast.main()
    print(f"[steps] passi eseguiti: {STATE['steps']} (richiesti {known.total_steps})", flush=True)


if __name__ == "__main__":
    main()
