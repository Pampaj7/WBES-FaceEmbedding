"""Protocolli di valutazione della comunita' (NoW originale, NoW metrical, REALY, Stirling/ESRC) come distanze fra
due mesh: parte comune.

Viste, mesh in mm, landmark, coppie, percorsi e uscite di ``pr_landmarks.py``, ``pr_run.py`` e ``pr_eval.py``.
Infrastruttura di ``aau/baselines_mm`` importata, non copiata: viste e soggetti (``blmm.VIEWS``, ``blmm.subjects``),
mm nel frame canonico (``blmm.to_mm``: UNA trasformazione per dominio), mesh di FaMoS TEST (``bp.meshes``).

Vista = elenco di mesh (soggetto, topologia, percorso) + frame in mm + il dominio dei landmark. Le viste note sono
quelle della matrice unificata (HIFI3D, FaceScape dev, FaceVerse con espressioni e neutra: 100 soggetti valutati x 6
topologie, crop compreso; FaMoS TEST: le 15 scansioni di galleria). Un dominio nuovo (es. Ava-256) entra SENZA
modificare il codice con ``--spec <json>``:

    {"name": "ava256", "dir": "datasets/AVA256/eval_view/npz", "subjects": [...] | "subjects_file": "...",
     "topologies": ["crop", "down8k", "noisy", "original", "remesh", "up60k"],
     "frame": {"u": 1.0, "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,0]},
     "landmarks": {"template": "<npz con V, F: template nel frame dei dati>", "corr": "<npz con vidx, bary: mappa
                   regione FLAME -> template, come datasets/UNIFIED_GT/corr/>", "snap_dir": "<cartella delle original
                   di soggetti NON valutati>", "snap_subjects": [...]}}

(mesh ``<dir>/<soggetto>_GTready_<topologia>.npz`` con ``V``, ``F``; ``frame``: V_mm = u R V + t; ``landmarks``: gli
ingressi di ``pr_landmarks.py define``, che scrive la definizione dei landmark del dominio).

Mesh in mm (``mm_mesh``): solo i vertici referenziati, NON centrate (i protocolli stimano la traslazione; l'origine e'
quella del frame canonico, la media FLAME). Le coordinate native restano disponibili per gli indici dei landmark.

Coppie (``pairs``): tutte le coppie non ordinate i < j di soggetti diversi e topologie diverse (le 148.500 di
all_cross per vista, crop compreso); con una sola topologia (FaMoS) tutte le coppie di soggetti diversi. Ogni
protocollo e' asimmetrico (A predetta, B riferimento): la distanza della coppia e' la MEDIA DEI DUE VERSI,
d(a, b) = (e(A = a, B = b) + e(A = b, B = a)) / 2; ``pr_run.py`` calcola le coppie ordinate.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
AAU = THIS.parents[1]
REPO = AAU.parent
for _p in (AAU / "baselines_mm", AAU / "baselines_param", REPO / "v2_work" / "phase0", AAU / "famos",
           AAU / "recon", REPO / "v3_work" / "unified_gt"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import blmm  # noqa: E402

EXT = Path(os.environ.get("PR_EXT", Path.home() / "ext" / "protocolli"))
CLONES = EXT / "cloni"                     # cloni ufficiali (fuori dal repo), commit in CLONE_COMMITS
REALY_DATA = CLONES / "REALY" / "data"
CLONE_COMMITS = {"now_evaluation": "7bd149850a5cad431a488857f1dc774e1eb533c9",
                 "REALY": "3f13e9adcac8cbcfd4ac7db23e2c7c4f63174e8a",
                 "fg2018-competition": "3bbbdefb53001cd3736137af7c4f39686fb691b5"}

WORK = AAU / "runs" / "protocolli"                   # intermedi (prep, parti): fuori dalle cartelle dei plug-in
PLUGIN_ROOT = AAU / "runs" / "evidence" / "competitors_new"
EVID = AAU / "runs" / "evidence" / "protocolli"      # Spearman, delta, controlli, results.md
DEFS = THIS / "landmarks"                            # definizione dei landmark per dominio (indici, in git)
DEFS_NPZ = WORK / "defs"                             # 85 punti e regioni per vertice (fuori da git)

VIEWS = ("hifi3d", "facescape", "faceverse", "faceverse_neutral", "famos")
SYNTH = VIEWS[:4]
METHODS = ("now_original", "now_metrical", "realy", "stirling")
LABEL = {"now_original": "NoW originale (Procrustes + ICP con scala)",
         "now_metrical": "NoW metrical / MICA (Procrustes senza scala + ICP rigido)",
         "realy": "REALY (gICP con scala + rICP per regione + nICP)",
         "stirling": "Stirling/ESRC (Procrustes con scala, RMSE)"}
FAMILY = "protocolli della comunita'"
# 7 landmark NoW / Stirling (scans_lmks_onlypp): ex_r, en_r, en_l, ex_l, subnasale, ch_r, ch_l = iBUG 36, 39, 42, 45,
# 33, 48, 54 (now_common.LMK7_IBUG; REALY utils/util.py:51 "seven_keypoints")
LMK7 = (36, 39, 42, 45, 33, 48, 54)
LMK7_NAMES = ("ex_r", "en_r", "en_l", "ex_l", "subnasale", "ch_r", "ch_l")
# regioni REALY (main.py:177): etichette per vertice 1..4, 0 = fuori
REGIONS = ("nose", "mouth", "forehead", "cheek")


# ------------------------------------------------------------------------------------------- viste

def view(name: str, spec: Path | None = None) -> dict:
    """{name, lm_domain, items [(soggetto, topologia, percorso)], frame | None, spec | None}."""
    if spec is not None:
        s = json.loads(Path(spec).read_text())
        d = Path(s["dir"]) if Path(s["dir"]).is_absolute() else REPO / s["dir"]
        subj = s.get("subjects") or json.loads(Path(s["subjects_file"]).read_text())
        items = [(str(x), t, d / f"{x}_GTready_{t}.npz") for x in sorted(subj) for t in s["topologies"]]
        return {"name": s["name"], "lm_domain": s["name"], "items": items, "frame": s["frame"], "spec": s}
    import bp
    if name == "famos":
        items = bp.meshes("famos")
    else:
        items = [(x, t, blmm.mesh_path(name, x, t)) for x in blmm.subjects(name) for t in blmm.TOPOLOGIES]
    return {"name": name, "lm_domain": blmm.VIEWS[name]["domain"], "items": items, "frame": None, "spec": None}


def unit_of(v: dict) -> float:
    """mm per unita' nativa (``u`` del frame del dominio)."""
    if v["frame"] is not None:
        return float(v["frame"]["u"])
    if v["name"] == "famos":
        return 1.0
    return float(json.loads(blmm.FRAMES_JSON.read_text())["domains"][v["lm_domain"]]["u"])


def to_mm(v: dict, V: np.ndarray, path: Path) -> np.ndarray:
    if v["frame"] is None:
        return blmm.to_mm(v["name"], V, Path(path).stem)
    f = v["frame"]
    return float(f["u"]) * V @ np.asarray(f["R"], dtype=np.float64).T + np.asarray(f["t"], dtype=np.float64)


def mm_mesh(v: dict, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(V mm sui vertici referenziati, F rinumerata, indici dei vertici tenuti nella mesh su disco)."""
    V, F = blmm.load_raw(path)
    used = np.unique(F)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return to_mm(v, V[used], path), remap[F], used


def pairs(items: list) -> tuple[np.ndarray, np.ndarray]:
    """Coppie i < j di soggetti diversi e topologie diverse (una sola topologia: soggetti diversi)."""
    s = np.asarray([x for x, _, _ in items])
    t = np.asarray([y for _, y, _ in items])
    i, j = np.triu_indices(len(items), 1)
    keep = s[i] != s[j]
    if len(set(t)) > 1:
        keep &= t[i] != t[j]
    return i[keep], j[keep]


# ------------------------------------------------------------------------------------------- landmark

def lm_def(domain: str) -> dict:
    """Definizione dei landmark del dominio (``pr_landmarks.py define``): indici iBUG 68 (embedding FLAME 2020 di
    DECA/MICA) e 85 di REALY sulla topologia original, piu' le etichette di regione per vertice (npz fuori da git)."""
    d = json.loads((DEFS / f"{domain}.json").read_text())
    with np.load(DEFS_NPZ / f"{domain}.npz") as z:
        d["labels"] = np.asarray(z["labels"], dtype=np.int8)
    return d


def prep_path(vname: str) -> Path:
    return WORK / "prep" / f"{vname}.npz"


def read_prep(vname: str) -> dict:
    with np.load(prep_path(vname), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------- uscite

def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default) + "\n")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def plugin_json(method: str) -> None:
    """``<metodo>/fm_plugin.json`` per ``aau/full_matrix/fm_eval.py``: una lettura ``mm`` (la distanza del protocollo
    sulle mesh in mm, come la calcola la comunita'), confrontata con ENTRAMBE le GT."""
    atomic_json(PLUGIN_ROOT / method / "fm_plugin.json", {
        "label": LABEL[method], "family": FAMILY,
        "readings": {"mm": {"label": f"{LABEL[method]}, mm (media dei due versi)", "gt": ["fr", "sr"]}}})


def write_view(method: str, vname: str, items: list, D: np.ndarray, extra: dict, info: dict) -> None:
    """``<metodo>/<vista>.npz`` (``subjects``, ``topologies``, ``paths``, ``D_mm`` simmetrica, NaN dove non calcolata,
    + ``extra`` con chiavi che NON iniziano per ``D``: il loader leggerebbe ogni ``D_*`` come lettura),
    ``<metodo>/<vista>.json``, ``<metodo>/fm_plugin.json``, ``<metodo>/pairs_<vista>.csv`` (fuori da git)."""
    if any(k == "D" or k.startswith("D_") for k in extra):
        raise ValueError("chiavi extra che il loader leggerebbe come letture")
    subj = np.asarray([x for x, _, _ in items])
    topo = np.asarray([t for _, t, _ in items])
    blmm.atomic_savez(PLUGIN_ROOT / method / f"{vname}.npz", subjects=subj, topologies=topo,
                      paths=np.asarray([blmm.rel(p) for _, _, p in items]), D_mm=D, **extra)
    atomic_json(PLUGIN_ROOT / method / f"{vname}.json", info)
    plugin_json(method)
    i, j = pairs(items)
    csv = PLUGIN_ROOT / method / f"pairs_{vname}.csv"
    tmp = csv.with_name(f".{csv.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        fh.write("subject_a,topology_a,subject_b,topology_b,mm\n")
        for a, b, x in zip(i, j, D[i, j]):
            fh.write(f"{subj[a]},{topo[a]},{subj[b]},{topo[b]},{x:.9g}\n")
    os.replace(tmp, csv)


def stamp() -> dict:
    return {"host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID", "none"),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}
