"""Gromov-Wasserstein (POT) e feature percettive su render (DINOv2, CLIP, LPIPS, DreamSim): parte comune.

Viste, mesh in mm, taglia, coppie e uscite di ``gp_gw.py``, ``gp_percettivi.py`` e ``gp_eval.py``. Infrastruttura di
``aau/baselines_mm`` importata, non copiata: viste e soggetti (``blmm.VIEWS``, ``blmm.subjects``), mm nel frame
canonico (``blmm.to_mm``: UNA trasformazione per dominio, nessuna rigida della GT), L_d e CS_ref (``blmm.params``),
centroid size robusta (``blmm.mesh_scalars``, cache ``blmm.scalars_of``), mesh del pilota (``blmm.template_subjects``).

Vista = elenco di mesh (soggetto, topologia, percorso) + frame in mm + (L_d, CS_ref). Le viste note sono quelle di
``bp.VIEWS`` (HIFI3D, FaceScape dev, FaceVerse con espressioni e neutra: 100 soggetti valutati x 6 topologie, crop
compreso; FaMoS TEST: le 15 scansioni di galleria, ``bp.meshes("famos")``). Un dominio nuovo (es. Ava-256) entra
SENZA modificare il codice con ``--spec <json>``:

    {"name": "ava256", "dir": "datasets/AVA256/eval_view/npz", "subjects": [...] | "subjects_file": "...",
     "topologies": ["crop", "down8k", "noisy", "original", "remesh", "up60k"],
     "frame": {"u": 1.0, "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,0]}, "L": <mm>, "cs_ref": <mm>}

(mesh ``<dir>/<soggetto>_GTready_<topologia>.npz`` con ``V``, ``F``; ``frame``: V_mm = u R V + t verso il frame
canonico, +y in alto, volto verso +z; ``L`` e ``cs_ref`` dai soggetti di calibrazione, come ``blmm.params``).

Mesh in mm (``mm_mesh``): vertici referenziati, centrati sul baricentro pesato per area (aree baricentriche della mesh
com'e'). Taglia (``size_of``): la centroid size ROBUSTA di ``blmm.mesh_scalars`` (aree della geometria passa-basso), dalla
cache della vista se c'e', altrimenti calcolata. Modo ``fr``: mm come sono; modo ``sr``: x CS_ref / CS_i (la
normalizzazione del modo ``cs`` di blmm, "mm alla taglia di riferimento").

Coppie (``pairs``): tutte le coppie non ordinate i < j di soggetti diversi e topologie diverse (le 148.500 di all_cross
per vista, crop compreso); con una sola topologia (FaMoS) tutte le coppie di soggetti diversi. Le distanze sono
simmetriche: la riga (a, b) e la (b, a) di all_cross leggono la stessa cella.
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
for _p in (AAU / "baselines_mm", AAU / "baselines_param", REPO / "v2_work" / "phase0"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import blmm  # noqa: E402

OUT = AAU / "runs" / "evidence" / "competitors_new"
VIEWS = ("hifi3d", "facescape", "faceverse", "faceverse_neutral", "famos")          # bp.VIEWS
PILOT_VIEWS = ("hifi3d", "facescape", "faceverse", "faceverse_neutral")            # bp.PILOT_VIEWS
PILOT_SUBJECTS = 12                       # primi soggetti NON valutati di blmm.template_subjects per vista
MODES = ("fr", "sr")
METHODS = ("gw", "dinov2", "clip", "lpips", "dreamsim")
LABEL = {"gw": "Gromov-Wasserstein (POT)", "dinov2": "DINOv2 ViT-S/14 su render", "clip": "CLIP ViT-B/32 su render",
         "lpips": "LPIPS alex su render", "dreamsim": "DreamSim ensemble su render"}
MODE_LABEL = {"fr": "mm (FR)", "sr": "a CS_ref (SR)"}
FAMILY = {"gw": "geometrico senza corrispondenza", "dinov2": "percettivo su render", "clip": "percettivo su render",
          "lpips": "percettivo su render", "dreamsim": "percettivo su render"}


# ------------------------------------------------------------------------------------------- viste

def view(name: str, spec: Path | None = None) -> dict:
    """{name, domain, items [(soggetto, topologia, percorso)], L, cs_ref, frame | None, scalars (cache o None)}."""
    if spec is not None:
        s = json.loads(Path(spec).read_text())
        d = Path(s["dir"]) if Path(s["dir"]).is_absolute() else REPO / s["dir"]
        subj = s.get("subjects") or json.loads(Path(s["subjects_file"]).read_text())
        items = [(str(x), t, d / f"{x}_GTready_{t}.npz") for x in sorted(subj) for t in s["topologies"]]
        return {"name": s["name"], "domain": s["name"], "items": items, "L": float(s["L"]),
                "cs_ref": float(s["cs_ref"]), "frame": s["frame"], "scalars": None}
    import bp
    prm = blmm.params()["domains"][blmm.VIEWS[name]["domain"]]
    if name == "famos":
        items = bp.meshes("famos")
    else:
        items = [(x, t, blmm.mesh_path(name, x, t)) for x in blmm.subjects(name) for t in blmm.TOPOLOGIES]
    try:
        sc = blmm.scalars_of(name)
    except FileNotFoundError:
        sc = None
    return {"name": name, "domain": blmm.VIEWS[name]["domain"], "items": items, "L": float(prm["L"]),
            "cs_ref": float(prm["cs_ref"]), "frame": None, "scalars": sc}


def pilot_view(name: str) -> dict:
    """La vista coi primi ``PILOT_SUBJECTS`` soggetti NON valutati (``blmm.template_subjects``) x 6 topologie, nella
    cartella della vista valutata (come il pilota di ``bp_fit_e1``)."""
    v = view(name)
    ss = blmm.template_subjects(name)[:PILOT_SUBJECTS]
    if set(ss) & set(blmm.subjects(name)):
        raise SystemExit(f"{name}: soggetti del pilota fra i valutati")
    v["items"] = [(x, t, blmm.mesh_path(name, x, t)) for x in ss for t in blmm.TOPOLOGIES]
    v["name"] = f"pilot_{name}"
    return v


def to_mm(v: dict, V: np.ndarray, path: Path) -> np.ndarray:
    if v["frame"] is None:
        return blmm.to_mm(v["name"].removeprefix("pilot_"), V, Path(path).stem)
    f = v["frame"]
    return float(f["u"]) * V @ np.asarray(f["R"], dtype=np.float64).T + np.asarray(f["t"], dtype=np.float64)


def mm_mesh(v: dict, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(V mm centrata sul baricentro pesato per area, F, aree baricentriche dei vertici) sui vertici referenziati."""
    V, F = blmm.load_raw(path)
    used = np.unique(F)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    V, F = to_mm(v, V[used], path), remap[F]
    w = blmm.vertex_areas(V, F)
    return V - (w @ V) / w.sum(), F, w


def size_of(v: dict, path: Path, V: np.ndarray, F: np.ndarray) -> tuple[float, str]:
    """Centroid size robusta (mm) e la sua fonte (cache della vista o calcolata con ``blmm.mesh_scalars``)."""
    sc = v["scalars"]
    if sc is not None and blmm.rel(path) in sc:
        return float(sc[blmm.rel(path)]["cs"]), "cache"
    return float(blmm.mesh_scalars(V, F)["cs"]), "calcolata"


def pairs(items: list) -> tuple[np.ndarray, np.ndarray]:
    """Coppie i < j di soggetti diversi e topologie diverse (una sola topologia: soggetti diversi)."""
    s = np.asarray([x for x, _, _ in items])
    t = np.asarray([y for _, y, _ in items])
    i, j = np.triu_indices(len(items), 1)
    keep = s[i] != s[j]
    if len(set(t)) > 1:
        keep &= t[i] != t[j]
    return i[keep], j[keep]


# ------------------------------------------------------------------------------------------- uscite

def out_npz(method: str, vname: str) -> Path:
    return OUT / method / f"{vname}.npz"


def write_view(method: str, v: dict, D: dict, extra: dict, info: dict) -> None:
    """``<metodo>/<vista>.npz`` (``subjects``, ``topologies``, ``D_fr``, ``D_sr`` simmetriche, NaN dove non
    calcolate, + ``extra``), ``<metodo>/<vista>.json`` (``info``), ``<metodo>/fm_plugin.json`` e ``<metodo>/pairs_<vista>.csv`` (le coppie di
    ``pairs``: soggetto e topologia di a e b, d_fr, d_sr; fuori da git per la regola ``pairs_*.csv``)."""
    items = v["items"]
    subj = np.asarray([x for x, _, _ in items])
    topo = np.asarray([t for _, t, _ in items])
    blmm.atomic_savez(out_npz(method, v["name"]), subjects=subj, topologies=topo,
                      paths=np.asarray([blmm.rel(p) for _, _, p in items]),
                      **{f"D_{m}": D[m] for m in MODES}, **extra)
    atomic_json(OUT / method / f"{v['name']}.json", info)
    plugin_json(method)
    i, j = pairs(items)
    csv = OUT / method / f"pairs_{v['name']}.csv"
    tmp = csv.with_name(f".{csv.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        fh.write("subject_a,topology_a,subject_b,topology_b,d_fr,d_sr\n")
        for a, b, x, y in zip(i, j, D["fr"][i, j], D["sr"][i, j]):
            fh.write(f"{subj[a]},{topo[a]},{subj[b]},{topo[b]},{x:.9g},{y:.9g}\n")
    os.replace(tmp, csv)


def plugin_json(method: str) -> None:
    """``<metodo>/fm_plugin.json`` per ``aau/full_matrix/fm_eval.py``: etichetta, famiglia, letture ``fr`` -> GT FR e
    ``sr`` -> GT SR."""
    atomic_json(OUT / method / "fm_plugin.json", {
        "label": LABEL[method], "family": FAMILY[method],
        "readings": {m: {"label": f"{LABEL[method]}, {MODE_LABEL[m]}", "gt": [m]} for m in MODES}})


def read_view(method: str, vname: str) -> dict:
    with np.load(out_npz(method, vname)) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def stamp() -> dict:
    return {"host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID", "none"),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}
