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
                           Chiavi opzionali, tutte SPENTE di default:
                             "label_groups": {"rexprA": ["rexpr1", ...], ...} etichette di
                               topologia fuse in gruppi (il campionatore v1 pesca un'etichetta
                               alla volta, quindi la frazione di espressioni per soggetto nuovo e'
                               n_gruppi_expr / n_etichette; con 6 topologie e 2 gruppi = 25%);
                             "canon": {"bfm": [[3x3]]} canonicalizzazione del frame per dominio,
                               applicata alla geometria PRIMA degli operatori (pre-pass): richiede
                               che quel dominio arrivi da geometria o tar, non da una vista;
                             "aug": {"rot_deg": a, "reflect_p": p} rotazione casuale (asse
                               uniforme, angolo <= a) e riflessione x->-x con probabilita' p dei
                               vertici dei soli soggetti di training. Come le perturbazioni della
                               ricetta, tocca solo il canale xyz: gli operatori non sono
                               ricalcolati (con la riflessione la chiralita' dei gradienti non
                               segue: va dichiarato se la si accende).
    --gt-keep-scale        legge D_orig COSI' COM'E' nel file: sostituisce ``load_gt_distance_matrix``,
                           che divide ogni voce per il massimo globale della matrice. Necessario con
                           datasets/ICT_SCALE/gt_joint_bfm_ict.npz, che e' gia' alla scala della GT in uso
                           (blocco BFM come nel congiunto, massimo 0.834; blocco ICT diviso per il massimo
                           di ICT-5000, quindi massimo 1.186): senza, OGNI distanza, BFM compresa, verrebbe
                           divisa per 1.186. Coincide con la lettura v1 solo se il massimo del file e'
                           esattamente 1 (vero per JOINT_BFM_ICT/gt_matrix.npz). Una guardia ferma il run se
                           il manifest della GT dichiara massimo > 1 e il flag manca.
    --lr-steps "N:lr,..."   lr a passi fissi (dal passo N+1), al posto di ReduceLROnPlateau. Il run grande usa
                           81747:5e-5, il dimezzamento del congiunto 1019532 misurato sul suo log.
    --train-threads K      thread di torch del training: col pre-pass in parallelo sugli stessi core va
                           limitato (smoke S1: 35 s/passo invece di 0.75 senza limite)
    --plateau-patience P   patience di ReduceLROnPlateau in epoche (alternativo a --lr-steps).
                           Nella data-spec anche: "tar_index": <shards>/index.npz (lettura dei tar per
                           offset, cache_budget.py index); "pin_domains": ["bfm"] (soggetti di quei domini in
                           ogni blocco) e "domain_step_share": {"bfm": 0.09} (frazione dei passi).
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
import gc
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
    if spec.get("tars") and spec.get("tar_index"):
        # una lettura dell'indice al posto di 200 scansioni di tar (~34 min a freddo su CephFS)
        with np.load(spec["tar_index"]) as z:
            tar_names, tar_id, names = [str(x) for x in z["tars"]], z["tar_id"], z["names"]
        wanted_tars = {Path(t).name: str(t) for t in spec["tars"]}
        missing = sorted(set(wanted_tars) - set(tar_names))
        if missing:
            raise SystemExit(f"tar_index {spec['tar_index']}: mancano {missing[:3]}")
        for n, i in zip(names, tar_id):
            t = tar_names[int(i)]
            if t in wanted_tars:
                add(str(n), "tar", wanted_tars[t])
    else:
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
                sample = ds[j]
                aug = self.cfg.get("aug")
                if aug and _split_name(self.files[int(idx)])[0] in self.cfg["train_set"]:
                    sample = dict(sample)
                    sample["verts"] = augment_verts(sample["verts"], aug, self.cfg["aug_rng"])
                return sample
        raise KeyError(f"{self.files[int(idx)]}: non e' nel blocco residente "
                       f"({self.resident_block}) ne' nell'eval online")

    # staging ------------------------------------------------------------------------------
    def stage(self, names: list[str], dest: Path, wait: bool = True):
        """Vista piatta in ``dest``: symlink per le viste, pre-pass per tar e geometria."""
        dest.mkdir(parents=True, exist_ok=True)
        need_tars, need_geom, subjects = set(), set(), set()
        canon = self.cfg.get("canon") or {}
        for n in names:
            kind, src = self.sources[n]
            if kind == "view" and domain_of_name(n) in canon:
                raise SystemExit(f"canon per {domain_of_name(n)}: {n} arriva da una vista con operatori "
                                 "gia' calcolati; servono geometria o tar (geom_dirs/tars)")
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
        cmd = [sys.executable, str(PREPASS), "--compress", "--out-dir", str(dest), "--subjects", str(subj_file),
               "--convention", self.cfg.get("convention", "areanorm"),
               "--n-proc", str(self.cfg["prepass_proc"])]
        if need_tars:
            cmd += ["--tars", *sorted(need_tars)]
        if need_geom:
            cmd += ["--geom-dirs", *sorted(need_geom)]
        if labels:
            cmd += ["--labels", labels]
        if canon:
            cmd += ["--transform", json.dumps(canon)]
        if self.cfg.get("tar_index") and need_tars:
            cmd += ["--tar-index", str(self.cfg["tar_index"])]
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

    def block_gib(self, dest: Path, names: list[str]) -> tuple[float, str]:
        """GiB esatti della cache del blocco, in secondi invece di ~25 minuti.

        Leggere gli header di tutti i 20.000 file (run 1056832) teneva la GPU ferma 22-26 min a ogni
        cambio di blocco. Ora: mesh dei tar dall'indice (n, m, spigoli), con la formula che e' risultata
        IDENTICA agli header in ogni misura (smoke S4 31.172/31.930, run 156.870/157.248 GiB); viste
        dagli header, una volta sola (le viste BFM sono le stesse in ogni blocco). Controllo a campione:
        200 file dei tar contro i loro header, ogni blocco; uno scarto ferma il run.
        """
        sys.path.insert(0, str(REPO_ROOT / "aau/data_scale"))
        from cache_budget import load_index, npz_sample_bytes, predicted_bytes
        if "view_bytes" not in self.cfg and self.cfg.get("view_bytes_json"):
            self.cfg["view_bytes"] = {k: int(v) for k, v in json.loads(Path(self.cfg["view_bytes_json"]).read_text()).items()}
            print(f"[steps] byte delle viste da {self.cfg['view_bytes_json']} ({len(self.cfg['view_bytes'])} file)", flush=True)
        vb = self.cfg.setdefault("view_bytes", {})
        tar_names = [n for n in names if self.sources[n][0] == "tar"]
        if tar_names and "index" not in self.cfg:
            idx = load_index(Path(self.cfg["tar_index"]))
            self.cfg["index"] = (idx, {str(n): i for i, n in enumerate(idx["names"])})
        total, check = 0, "nessun file dai tar"
        for n in names:
            if self.sources[n][0] == "tar":
                idx, pos = self.cfg["index"]
                i = pos[n]
                total += predicted_bytes(int(idx["n"][i]), int(idx["m"][i]), int(idx["E"][i]))
            else:
                if n not in vb:
                    vb[n] = npz_sample_bytes(dest / n)
                total += vb[n]
        if tar_names:
            idx, pos = self.cfg["index"]
            rng = np.random.default_rng(len(names))
            probe = [tar_names[int(j)] for j in rng.choice(len(tar_names), min(200, len(tar_names)), replace=False)]
            bad = [n for n in probe if npz_sample_bytes(dest / n)
                   != predicted_bytes(int(idx["n"][pos[n]]), int(idx["m"][pos[n]]), int(idx["E"][pos[n]]))]
            if bad:
                raise SystemExit(f"{dest.name}: la previsione dall'indice non torna con gli header per {bad[:3]}")
            check = f"indice = header su {len(probe)} file a campione"
        return total / 2 ** 30, check

    def predicted_gib(self, names: list[str]):
        """(previsione dall'indice, esatto dagli header) per le mesh che vengono dai tar: verifica la
        formula che cache_budget.py usa per i 40 blocchi prima che esistano."""
        if not self.cfg.get("tar_index"):
            return None
        sys.path.insert(0, str(REPO_ROOT / "aau/data_scale"))
        from cache_budget import load_index, npz_sample_bytes, predicted_bytes
        if "index" not in self.cfg:
            idx = load_index(Path(self.cfg["tar_index"]))
            self.cfg["index"] = (idx, {str(n): i for i, n in enumerate(idx["names"])})
        idx, pos = self.cfg["index"]
        tar_names = [n for n in names if self.sources[n][0] == "tar"]
        if not tar_names:
            return None
        pred = sum(predicted_bytes(int(idx["n"][pos[n]]), int(idx["m"][pos[n]]), int(idx["E"][pos[n]]))
                   for n in tar_names)
        dest = self.cfg["current_dest"]
        exact = sum(npz_sample_bytes(dest / n) for n in tar_names)
        return pred / 2 ** 30, exact / 2 ** 30

    def load(self, dest: Path, names: list[str]):
        self.cfg["current_dest"] = dest
        from fast_data import CachedDataset
        c = self.cfg
        missing = [n for n in names if not (dest / n).exists()]
        if missing:
            raise SystemExit(f"{dest}: {len(missing)} mesh mancanti dopo il pre-pass (prima {missing[0]})")
        t0 = time.time()
        gib, check = self.block_gib(dest, names)
        print(f"[steps] {dest.name}: {len(names)} campioni, cache {gib:.1f} GiB (indice per i tar, header "
              f"per le viste; {check}; tetto {c['cache_max_gb']:.0f} GiB, {time.time() - t0:.0f}s)", flush=True)
        if gib > c["cache_max_gb"]:
            raise SystemExit(f"{dest.name}: cache {gib:.1f} GiB > --cache-max-gb {c['cache_max_gb']:.0f}: "
                             "piu' blocchi o tetto piu' alto")
        # il controllo di fast_data proietta dal PRIMO campione soltanto: lo sostituisce quello sopra
        ds = CachedDataset(dest, workers=c["cache_workers"], residency=c["cache_residency"],
                           device=c["device"], max_gb=float("inf"))
        local = {self._pos[n]: j for j, n in enumerate(ds.files) if n in self._pos}
        return ds, local


def exact_cache_gib(dest: Path, names: list[str]) -> float:
    """GiB della cache del blocco, ESATTI: forme dagli header npy di TUTTI i file (cache_budget.py).

    La versione precedente campionava con linspace sulla lista ordinata: il passo si allineava alle
    topologie e pescava quasi sempre up60k (fino a 420 GB proiettati contro ~219 GiB reali).
    """
    sys.path.insert(0, str(REPO_ROOT / "aau/data_scale"))
    from cache_budget import npz_sample_bytes
    return sum(npz_sample_bytes(dest / n) for n in names) / 2 ** 30


def _cgroup_dir() -> Path | None:
    """Cgroup di memoria del JOB (quello su cui Slurm applica --mem), v1 o v2."""
    try:
        for line in open("/proc/self/cgroup"):
            _, ctrl, path = line.strip().split(":", 2)
            if ctrl in ("memory", ""):
                for base in (Path("/sys/fs/cgroup/memory"), Path("/sys/fs/cgroup")):
                    p = base / path.lstrip("/")
                    # risali fino a job_<id>: il limite --mem sta li', non sullo step o sul task
                    while p != base and not p.name.startswith("job_"):
                        p = p.parent
                    if p.name.startswith("job_") and p.exists():
                        return p
    except OSError:
        pass
    return None


def mem_log(what: str) -> None:
    """RSS del processo e memoria del cgroup del job (attuale e picco) in un riga del log."""
    rss = float("nan")
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) / 2 ** 20
    cg, cur, peak = _cgroup_dir(), float("nan"), float("nan")
    if cg is not None:
        for f, var in (("memory.usage_in_bytes", "cur"), ("memory.current", "cur"),
                       ("memory.max_usage_in_bytes", "peak"), ("memory.peak", "peak")):
            q = cg / f
            if q.exists():
                v = int(q.read_text().split()[0]) / 2 ** 30
                if var == "cur":
                    cur = v
                else:
                    peak = v
    print(f"[mem] {what}: RSS processo {rss:.1f} GiB, cgroup job {cur:.1f} GiB (picco {peak:.1f})", flush=True)


def malloc_trim() -> None:
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def partition_blocks(train: list[str], spec: dict) -> list[list[str]]:
    """Blocchi di soggetti di training. Usata dal trainer e da cache_budget.py (stessa partizione).

    I domini ``pin_domains`` (es. BFM, pochi soggetti) stanno in OGNI blocco: senza, 392 soggetti
    BFM su 54.000 avrebbero lo 0.7% dei passi e solo nel loro blocco. Gli altri sono permutati con
    ``block_seed`` e distribuiti a strisce.
    """
    rng = np.random.default_rng(int(spec.get("block_seed", 0)))
    pinned_dom = set(spec.get("pin_domains") or [])
    pinned = sorted(s for s in train if domain_of_name(s + "_GTready_x.npz") in pinned_dom)
    rest = [s for s in train if s not in set(pinned)]
    K = int(spec.get("n_blocks", 1))
    if not spec.get("stratify_blocks"):
        perm = rng.permutation(np.array(rest, dtype=object)).tolist()
        return [sorted(perm[k::K] + pinned) for k in range(K)]
    # stratificata per dominio (run con GNM): ogni dominio permutato e distribuito a strisce per conto
    # suo, cosi' ogni blocco ha lo stesso numero di soggetti per dominio (+-1). Con la striscia sulla
    # permutazione mista i GNM per blocco varierebbero di ~+-15 attorno a ~217, sotto i 210 che servono
    # a un'epoca (nessun soggetto due volte nella stessa epoca).
    blocks: list[list[str]] = [list(pinned) for _ in range(K)]
    for dom in sorted({domain_of_name(s + "_GTready_x.npz") for s in rest}):
        pool = sorted(s for s in rest if domain_of_name(s + "_GTready_x.npz") == dom)
        perm = rng.permutation(np.array(pool, dtype=object)).tolist()
        for k in range(K):
            blocks[k] += perm[k::K]
    return [sorted(b) for b in blocks]


def domain_of_name(name: str) -> str:
    num = int(name.split("_GTready_")[0][2:])
    if 100000 <= num < 200000:          # GNM Head, come train_v2.GNM_RANGE
        return "gnm"
    return "bfm" if num < 1000 else ("ict" if num >= 10000 else "flame")


def augment_verts(V, aug: dict, rng: np.random.Generator):
    """Rotazione casuale (asse uniforme, angolo <= rot_deg) e riflessione x->-x, sui soli vertici."""
    import torch
    R = np.eye(3)
    a = float(aug.get("rot_deg", 0.0))
    if a > 0:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        th = np.deg2rad(rng.uniform(-a, a))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    if rng.uniform() < float(aug.get("reflect_p", 0.0)):
        R = R @ np.diag([-1.0, 1.0, 1.0])
    return V @ torch.as_tensor(R.T, dtype=V.dtype, device=V.device)


def install_label_groups(groups: dict) -> dict:
    """Fonde etichette di topologia (es. rexpr1..4 -> rexprA) nel nome che il trainer legge."""
    import robustness.data_utils as du
    import robustness.train_runner as tr
    # due forme: {gruppo: [etichette]} per tutti i domini, oppure {"by_domain": {dominio: {...}}}
    if "by_domain" in groups:
        inv_dom = {d: {lab: g for g, labs in gr.items() for lab in labs} for d, gr in groups["by_domain"].items()}
    else:
        inv_dom = {None: {lab: g for g, labs in groups.items() for lab in labs}}
    inv = inv_dom.get(None, {})
    orig = du.infer_topology_label_from_name

    def infer(name, subject_id):
        lab = orig(name, subject_id)
        table = inv_dom.get(domain_of_name(str(subject_id) + "_GTready_x.npz"), inv)
        return table.get(lab, lab)
    du.infer_topology_label_from_name = infer
    tr.infer_topology_label_from_name = infer
    print(f"[steps] gruppi di etichette: {groups}", flush=True)
    return inv


def install_gt_keep_scale() -> None:
    """``load_gt_distance_matrix`` senza la divisione per il massimo (stessa lettura dei nomi)."""
    import robustness.train_runner as tr
    from intrinsic_utils import SUBJECT_RE_4DIGIT, extract_subject_id

    def load(path, subject_re=SUBJECT_RE_4DIGIT, dtype=np.float32):
        pack = np.load(path, allow_pickle=True)
        D = pack["D_orig"].astype(dtype)
        name_to_idx = {}
        for i, n in enumerate(pack["names"]):
            n = n.decode("utf-8", errors="ignore") if isinstance(n, bytes) else n
            sid = extract_subject_id(str(n), subject_re=subject_re)
            if sid is not None:
                name_to_idx[sid] = i
        if not name_to_idx:
            raise RuntimeError(f"Could not parse subject ids from names in {path}")
        print(f"[steps] GT a scala invariata {Path(path).name}: {D.shape}, "
              f"max={float(np.nanmax(D)):.4f} (nessuna divisione)", flush=True)
        return D, name_to_idx
    tr.load_gt_distance_matrix = load


# --- patch del trainer --------------------------------------------------------------------------

class _Fwd:
    """Inoltra ogni attributo a ``target`` tranne quelli sovrascritti (come train_v2._Fwd)."""

    def __init__(self, target, **overrides):
        self._target, self._overrides = target, overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._target, name)


def epoch_subset(pool: list[str], steps: int, batch: int, seed: int, domain_blocked: bool,
                 share: dict | None = None) -> list[str]:
    """``steps * batch`` soggetti del blocco, che v1 trasformera' in esattamente ``steps`` batch.

    Permutazione seminata per epoca, ciclica se il blocco ha meno soggetti di quelli chiesti
    (un soggetto non compare due volte nella stessa epoca finche' il blocco basta). Con il
    batching a dominio singolo ogni dominio riceve un multiplo di ``batch`` soggetti, con
    passi ripartiti in proporzione ai soggetti del dominio, oppure secondo ``share``
    ({dominio: frazione dei passi}, i domini non elencati si dividono il resto in proporzione),
    con il metodo del resto maggiore: train_v2
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
        share = {d: float(f) for d, f in (share or {}).items() if d in groups}
        rest = [d for d in groups if d not in share]
        n_rest = sum(len(groups[d]) for d in rest)
        quota = {d: steps * f for d, f in share.items()}
        quota.update({d: steps * (1 - sum(share.values())) * len(groups[d]) / max(n_rest, 1) for d in rest})
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
        frozen = set(fz["bfm"]) | set(fz["ict_view"]) | set(fz.get("gnm", []))
    split = json.loads(Path(known.split_json).read_text()) if known.split_json else None

    if split is not None and split.get("online_eval"):
        fixed_online = list(split["online_eval"])

        def select_online(eval_subjects, max_subjects_eval_train, seed):
            missing = sorted(set(fixed_online) - {str(s) for s in eval_subjects})
            if missing:
                raise SystemExit(f"eval online esplicito: {missing[:3]} non sono held-out di questo run")
            print(f"[steps] eval online esplicito ({len(fixed_online)} soggetti, dallo split): "
                  f"{fixed_online[:4]}...", flush=True)
            return list(fixed_online)
        tr._select_online_eval_subjects = select_online

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
        if cfg.get("blocked") is not None:
            cfg["blocked"].cfg["train_set"] = set(train)
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
        extra_ids = sorted({s_ for ids in (split or {}).get("online_eval_extra", {}).values() for s_ in ids})
        bad = sorted(set(extra_ids) - set(held))
        if bad:
            raise SystemExit(f"online_eval_extra: {bad[:3]} non sono held-out")
        eval_names = [n for n in ds.files if _split_name(n)[0] in set(online) | set(extra_ids)]
        K = int(cfg["spec"].get("n_blocks", 1))
        if K > cfg["epochs"]:
            raise SystemExit(f"n_blocks={K} > epoche={cfg['epochs']}: qualche blocco non verrebbe mai addestrato")
        cfg["blocks"] = partition_blocks(train, cfg["spec"])
        pinned_dom = set(cfg["spec"].get("pin_domains") or [])
        pinned = [s for s in cfg["blocks"][0] if domain_of_name(s + "_GTready_x.npz") in pinned_dom]
        print(f"[steps] {K} blocchi da {[len(b) for b in cfg['blocks']]} soggetti "
              f"({len(pinned)} residenti in tutti: {sorted(pinned_dom) or '-'}), "
              f"epoche {cfg['epochs']} ripartite in proporzione; eval online {len(online)} soggetti", flush=True)
        _report_expression_fraction(ds, train)
        ds.stage(eval_names, root / "eval")
        ds._parts.append(ds.load(root / "eval", eval_names))
        shutil.rmtree(root / "eval", ignore_errors=True)
        shutil.rmtree(root / "eval_geom", ignore_errors=True)
        _switch_block(0)

    def _report_expression_fraction(ds: BlockedDataset, train: list[str]) -> None:
        """Frazione di mesh d'espressione che il campionatore v1 sceglie per un soggetto che ne ha."""
        by_sid: dict[str, list[int]] = {}
        for i, n in enumerate(ds.files):
            by_sid.setdefault(_split_name(n)[0], []).append(i)
        # un soggetto per (dominio, numero di espressioni): GNM ne ha 1 o 2, ICT nuovi 8
        picks = {}
        for s_ in train:
            n_ex = sum("rexpr" in ds.files[i] for i in by_sid.get(s_, []))
            if n_ex:
                picks.setdefault((domain_of_name(s_ + "_GTready_x.npz"), n_ex), s_)
        for (dom, n_ex), sid in sorted(picks.items()):
            _expr_fraction_one(ds, by_sid, sid, f"{dom}, {n_ex} espressioni")

    def _expr_fraction_one(ds, by_sid, sid, tag):
        topo: dict[str, list[int]] = {}
        for i in by_sid[sid]:
            topo.setdefault(tr.infer_topology_label_from_name(ds.files[i], sid), []).append(i)
        rng = np.random.default_rng(0)
        n_expr = n_tot = 0
        for _ in range(4000):
            for idx, _lab in tr._sample_subject_mesh_entries(sid, {sid: topo},
                                                            int(cfg["args"].max_meshes_per_subject_train), rng):
                n_tot += 1
                n_expr += "rexpr" in ds.files[idx]
        frac = n_expr / max(n_tot, 1)
        cfg["expr_fraction"] = frac
        print(f"[steps] {sid} ({tag}): etichette {sorted(topo)} -> mesh d'espressione scelte {frac:.3f} "
              f"(simulazione del campionatore v1, 4000 estrazioni)", flush=True)

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
        if len(ds._parts) > 1:
            # libera il blocco precedente PRIMA di caricare il nuovo, esplicitamente: la tupla esce dalla
            # lista, poi gc e malloc_trim, che restituisce al sistema la memoria rimasta nelle arene di
            # glibc (run 1056832: OOM al primo cambio, vedi aau/data_scale/PLAN.md)
            mem_log(f"cambio {ds.resident_block}->{k}: prima di liberare")
            old = ds._parts.pop()
            del old
            gc.collect()
            malloc_trim()
            mem_log(f"cambio {ds.resident_block}->{k}: blocco {ds.resident_block} liberato (gc + malloc_trim)")
            shutil.rmtree(root / f"block{ds.resident_block:03d}", ignore_errors=True)
        # la geometria estratta dai tar la cancella gia' prepass_ops; qui per sicurezza, perche'
        # lasciata su /tmp si accumulerebbe (~3 GB a blocco, ~130 GB su 40 blocchi)
        shutil.rmtree(root / f"block{k:03d}_geom", ignore_errors=True)
        t0 = time.time()
        ds._parts.append(ds.load(dest, _names_of(k)))
        # tutti i campioni del blocco sono ora decodificati in RAM: la copia su /tmp (che conta
        # contro --mem) non serve piu'. Senza questo il picco e' cache + operatori su disco
        # (OOM a 120G sui 400 soggetti BFM, job 1055893/1055894)
        mem_log(f"blocco {k} caricato, operatori ancora su /tmp")
        shutil.rmtree(dest, ignore_errors=True)
        mem_log(f"blocco {k} caricato, /tmp liberato")
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
            k = (epoch - 1) * len(cfg["blocks"]) // cfg["epochs"]   # ogni blocco riceve la sua quota
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
                                  known.domain_blocked, (cfg.get("spec") or {}).get("domain_step_share"))
        kw["train_subjects"] = subset
        before = STATE["steps"]
        pend = cfg.get("pending")
        busy0 = bool(pend and pend[1] is not None and pend[1].poll() is None)
        t0 = time.time()
        stats = orig_epoch(*a, **kw)
        dt = time.time() - t0
        busy1 = bool(pend and pend[1] is not None and pend[1].poll() is None)
        print(f"\n[steps] epoca {epoch}: {STATE['steps'] - before} passi in {dt:.0f}s "
              f"({dt / max(STATE['steps'] - before, 1):.2f} s/passo), pre-pass in corso "
              f"all'inizio={'si' if busy0 else 'no'} alla fine={'si' if busy1 else 'no'}", flush=True)
        if cfg.get("blocked") is not None:
            mem_log(f"fine epoca {epoch}")
        if STATE["steps"] - before != steps:
            # v1 salta i batch senza coppie valide: lo si dice, e l'epoca successiva recupera
            print(f"[steps] AVVISO epoca {epoch}: {STATE['steps'] - before} passi eseguiti, attesi {steps}",
                  flush=True)
        return stats
    tr._train_epoch = _train_epoch

    # conta i passi VERI: ogni optimizer.step() dell'ottimizzatore che costruisce run_training
    lr_steps = sorted((int(b), float(v)) for b, v in
                      (x.split(":") for x in known.lr_steps.split(",") if x)) if known.lr_steps else []

    class CountingAdam(tr.optim.Adam):
        def step(self, *a, **kw):
            n = STATE["steps"] + 1                     # numero (1-based) del passo che sta per farsi
            for boundary, lr in lr_steps:
                if n > boundary and self.param_groups[0]["lr"] != lr:
                    for g in self.param_groups:
                        g["lr"] = lr
                    print(f"\n[steps] passo {n}: lr -> {lr:g} (oltre il passo {boundary})", flush=True)
            STATE["steps"] = n
            return super().step(*a, **kw)
    tr.optim = _Fwd(tr.optim, Adam=CountingAdam)

    # run dir, per scriverci extra_eval.csv
    orig_make_run_dir = tr.make_run_dir

    def make_run_dir(*a, **kw):
        cfg["run_dir"] = orig_make_run_dir(*a, **kw)
        return cfg["run_dir"]
    tr.make_run_dir = make_run_dir

    extra = (split or {}).get("online_eval_extra") or {}
    if extra:
        import dataclasses
        orig_eval = tr.evaluate_subject_robustness_grid

        def evaluate(*a, **kw):
            """Eval online di sempre (sceglie i best_by_*), poi la stessa funzione su ogni insieme
            ``online_eval_extra`` (p.es. 16 GNM di validazione): solo registrata, non sceglie niente."""
            res = orig_eval(*a, **kw)
            main_ctx = kw["eval_ctx"]
            args = cfg["args"]
            for dom, ids in extra.items():
                key = f"extra_ctx_{dom}"
                if key not in cfg:
                    plan = tr.build_eval_plan(subj_map=main_ctx.subj_map, eval_subjects=ids,
                                              max_meshes_per_subject_eval=int(args.max_meshes_per_subject_eval),
                                              seed=int(args.seed) + 91_000)
                    cache = tr.preload_eval_samples(dataset=main_ctx.dataset, eval_plan=plan, workers=2)
                    cfg[key] = dataclasses.replace(main_ctx, eval_subjects=list(ids), eval_plan=plan,
                                                   sample_cache=cache)
                r = orig_eval(*a, **{**kw, "eval_ctx": cfg[key]})
                vals = {k: float(r[k]) for k in ("spearman_clean", "pearson_clean", "auc_r")}
                print(f"[steps] eval extra {dom} ({len(ids)} soggetti, passo {STATE['steps']}): "
                      + " ".join(f"{k}={v:.4f}" for k, v in vals.items()), flush=True)
                csv = Path(cfg["run_dir"]) / "extra_eval.csv"
                new_file = not csv.exists()
                with open(csv, "a") as fh:
                    if new_file:
                        fh.write("step,domain,n_subjects,spearman_clean,pearson_clean,auc_r\n")
                    fh.write(f"{STATE['steps']},{dom},{len(ids)},{vals['spearman_clean']:.6f},"
                             f"{vals['pearson_clean']:.6f},{vals['auc_r']:.6f}\n")
            return res
        tr.evaluate_subject_robustness_grid = evaluate

    if lr_steps:
        class FixedSchedule:
            """Al posto di ReduceLROnPlateau: il lr lo decide CountingAdam per numero di passo."""

            def __init__(self, *a, **kw):
                pass

            def step(self, *a, **kw):
                pass
        tr.ReduceLROnPlateau = FixedSchedule
        print(f"[steps] lr a passi fissi {lr_steps} (ReduceLROnPlateau disattivato)", flush=True)

    if known.plateau_patience is not None:
        # la ricetta usa ReduceLROnPlateau(patience=8) per EPOCA: con epoche piu' corte la si
        # riporta agli stessi PASSI (es. 8 epoche da 880 = 64 epoche da 110)
        orig_plateau = tr.ReduceLROnPlateau

        def plateau(opt, **kw):
            kw["patience"] = int(known.plateau_patience)
            print(f"[steps] ReduceLROnPlateau patience={kw['patience']} epoche", flush=True)
            return orig_plateau(opt, **kw)
        tr.ReduceLROnPlateau = plateau

    if known.data_spec:
        if not known.split_json:
            raise SystemExit("--data-spec richiede --split-json")
        spec = json.loads(Path(known.data_spec).read_text())
        cfg["spec"] = spec
        sources = collect_sources(spec)
        if spec.get("label_groups"):
            install_label_groups(spec["label_groups"])
        aug = spec.get("aug") or None
        if aug and not (float(aug.get("rot_deg", 0)) > 0 or float(aug.get("reflect_p", 0)) > 0):
            aug = None
        cfg["blocked"] = BlockedDataset(sources, {
            "labels": spec.get("labels"), "convention": spec.get("convention", "areanorm"),
            "prepass_proc": int(known.prepass_proc), "cache_workers": int(known.cache_workers),
            "cache_residency": known.cache_residency, "cache_max_gb": float(known.cache_max_gb),
            "device": None, "canon": spec.get("canon") or None, "aug": aug, "train_set": set(),
            "tar_index": spec.get("tar_index"), "view_bytes_json": spec.get("view_bytes_json"),
            "aug_rng": np.random.default_rng(int(spec.get("aug_seed", 0)))})
        print(f"[steps] frame: canon={spec.get('canon') or 'spento'} aug={aug or 'spento'}", flush=True)
        if not known.pin_cache:
            # memoria pinned: non compare nel MaxRSS ma il cgroup la conta, e il CachingHostAllocator
            # la arrotonda alla potenza di 2 (S3: cgroup 117.8 GiB contro 61 di MaxRSS). Spenta nella
            # cache dei blocchi; la copia verso la GPU diventa pageable (costo misurato nello smoke).
            import fast_data as fd
            orig_res = fd._to_residency
            fd._to_residency = lambda sample, device, pin: orig_res(sample, device, False)
            print("[steps] cache dei blocchi NON pinned", flush=True)
        factory = lambda *_a, **_k: cfg["blocked"]  # noqa: E731
        du.GTReadyDataset = factory
        tr.GTReadyDataset = factory
        print(f"[steps] data-spec {known.data_spec}: {len(sources)} mesh, staging in {known.stage_root}",
              flush=True)


def check_gt_scale(rest: list[str], keep_scale: bool) -> None:
    """Fallisce se la GT ha massimo > 1 (GT estesa a scala di ICT-5000) e manca --gt-keep-scale:
    ``load_gt_distance_matrix`` dividerebbe TUTTO, BFM compreso, per quel massimo."""
    if "--dist_npz" not in rest:
        return
    side = Path(rest[rest.index("--dist_npz") + 1]).with_suffix(".json")
    if not side.exists():
        return
    man = json.loads(side.read_text())
    gmax = float(man.get("global_max", 1.0))
    if gmax > 1.0 + 1e-6 and not keep_scale:
        raise SystemExit(f"GT {side.with_suffix('.npz').name}: massimo {gmax:.4f} > 1 (scala {man.get('scale')}); "
                         "serve --gt-keep-scale, altrimenti ogni distanza verrebbe divisa per il massimo")
    if keep_scale:
        print(f"[steps] GT {side.with_suffix('.npz').name}: massimo {gmax:.4f}, letta a scala invariata", flush=True)


def main() -> None:
    # allow_abbrev=False: con le abbreviazioni il flag di ricetta `--lr 1e-4` veniva preso come
    # `--lr-steps 1e-4` e tolto al trainer v1 (smoke 1056267)
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--total-steps", type=int, default=0)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--split-json", default="")
    p.add_argument("--frozen-heldout", default="")
    p.add_argument("--data-spec", default="")
    p.add_argument("--stage-root", default="")
    p.add_argument("--prepass-proc", type=int, default=16)
    p.add_argument("--domain-blocked", action="store_true")
    p.add_argument("--eval_domain", default="")
    p.add_argument("--gt-keep-scale", action="store_true")
    p.add_argument("--plateau-patience", type=int, default=None)
    p.add_argument("--lr-steps", default="",
                   help="schedule a passi fissi 'passo:lr,...': lr dal passo successivo (es. 81747:5e-5 "
                        "= il dimezzamento del congiunto 1019532); disattiva ReduceLROnPlateau")
    p.add_argument("--pin-cache", action="store_true",
                   help="tiene la memoria pinned nella cache dei blocchi (default: spenta, vedi install)")
    p.add_argument("--train-threads", type=int, default=0,
                   help="thread di torch del processo di training (0 = default): va limitato perche' "
                        "il pre-pass del blocco successivo gira in parallelo sugli stessi core")
    known, rest = p.parse_known_args()

    if known.total_steps <= 0:
        if known.data_spec or known.split_json or known.domain_blocked:
            raise SystemExit("--data-spec/--split-json/--domain-blocked richiedono --total-steps")
        sys.argv = [sys.argv[0]] + rest
        train_fast.main()                         # comportamento di sempre
        return

    # flag di train_fast che servono anche qui; restano in `rest` per train_fast
    q = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
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

    if known.lr_steps and known.plateau_patience is not None:
        raise SystemExit("--lr-steps e --plateau-patience sono alternativi")
    if known.train_threads > 0:
        import torch
        torch.set_num_threads(int(known.train_threads))
        print(f"[steps] thread di torch del training: {torch.get_num_threads()}", flush=True)
    check_gt_scale(rest, known.gt_keep_scale)

    import robustness.train_runner  # noqa: F401
    install(known)
    if known.gt_keep_scale:
        install_gt_keep_scale()        # prima di train_v2: il suo guardiano NaN avvolge questa
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
