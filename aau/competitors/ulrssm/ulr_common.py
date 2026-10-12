"""ULRSSM (Cao, Roetzer, Bernard, "Unsupervised Learning of Robust Spectral Shape Matching", ACM TOG / SIGGRAPH 2023)
come concorrente "corrispondenza densa imparata + distanza": parte comune di ``ulr_data.py``, ``ulr_train.py``,
``ulr_eval.py``, ``ulr_paired.py``.

Codice di terzi NON copiato nel repo: il repo ufficiale (licenza MIT, Copyright (c) 2021 Dongliang Cao,
https://github.com/dongliangcao/Unsupervised-Learning-of-Robust-Spectral-Shape-Matching, commit ``ULRSSM_COMMIT``) e'
clonato fuori dal repo in ``ULRSSM_DIR`` (default ``~/ext/ulrssm``) e IMPORTATO: DiffusionNet, RegularizedFMNet,
Similarity, SURFMNetLoss, SquaredFrobeniusLoss, ``get_operators`` (operatori e cache su disco). Qui solo la colla:
viste, preparazione delle mesh, mappa punto-a-punto, Procrustes, uscite.

Iperparametri (``CFG``): quelli di ``options/train/faust.yaml`` (la configurazione di default del lavoro per forme
complete), tranne ``input_type`` che e' un argomento (``xyz`` o ``wks``, vedi ``ulr_train.py``).

Preparazione di una mesh (``prep``), la stessa per training, template e valutazione:
  1. vertici referenziati, coordinate nel frame canonico (+y in alto, volto verso +z; mm per le viste di valutazione);
  2. verso delle facce (``orient``): invertite se la normale media pesata per area ha z < 0, cosi' il volto guarda
     verso +z come nelle viste dei generatori di training (FaceVerse e' salvato con le facce invertite, cfr.
     ``zs_stage.py --flip-faces``). Le feature di DiffusionNet dipendono dall'orientazione dei riferimenti tangenti;
  3. normalizzazione di ``preprocess.py`` di ULRSSM: centro dei vertici, divisione per la radice dell'area totale;
  4. ``get_operators`` con ``K_FMAP`` = 200 autovettori nella cache del job (DiffusionNet li ritrova per hash).

Mappa punto-a-punto (``p2p``) = ``FMNetModel.validate_single`` di ULRSSM nel caso isometrico (quello dei test di
default): Pyx = argmax sulle x della similarita' coseno delle feature (``Similarity`` con ``hard``), Cxy =
evecs_trans_y Pyx evecs_x, p2p = nearest neighbour di evecs_y fra le righe di evecs_x Cxy^T (``fmap2pointmap``). Qui a
blocchi sulla GPU (``torch.cdist`` 73k x 73k non sta in memoria): argmin di |g_x|^2 - 2 e_y . g_x, la stessa
quantita' di cdist a meno dei pareggi numerici (controllo in ``ulr_eval.py smoke``).

Distanza per coppia (``pair_rms``): mappa B -> A, rigida pesata SENZA scala (Umeyama) dei punti di A sui vertici di B,
RMS pesato con le aree baricentriche di B; poi lo stesso nel verso A -> B; d = media dei due versi. FR: coordinate in
mm; SR: ogni mesh x CS_ref / CS_i (CS_i = centroid size robusta, ``blmm.mesh_scalars``; "mm alla taglia di
riferimento", come il modo ``cs`` di blmm e ``gw_percettivi``).

Viste (``view``): quelle di ``bp.VIEWS`` (100 soggetti valutati x 6 topologie, crop compreso; FaMoS TEST le 15
scansioni di galleria) o un dominio nuovo SENZA modificare il codice con ``--spec <json>``, nello stesso formato di
``aau/competitors/gw_percettivi/gp_common.py``:

    {"name": "ava256", "dir": "datasets/AVA256/eval_view/npz", "subjects": [...] | "subjects_file": "...",
     "topologies": ["crop", "down8k", "noisy", "original", "remesh", "up60k"],
     "frame": {"u": 1.0, "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,0]}, "L": <mm>, "cs_ref": <mm>}

(mesh ``<dir>/<soggetto>_GTready_<topologia>.npz`` con ``V``, ``F``; ``frame``: V_mm = u R V + t verso il frame
canonico, +y in alto, volto verso +z: l'ingresso ``xyz`` di DiffusionNet NON e' invariante alla rotazione).

Coppie (``pairs``): come ``gp_common.pairs``, tutte le i < j di soggetti diversi e topologie diverse (le 148.500 di
all_cross per vista, crop compreso); con una sola topologia (FaMoS) tutte le coppie di soggetti diversi.
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
for _p in (AAU / "baselines_mm", AAU / "baselines_param"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import blmm  # noqa: E402

ULRSSM_DIR = Path(os.environ.get("ULRSSM_DIR", Path.home() / "ext" / "ulrssm"))
ULRSSM_COMMIT = "0625cbcbb7cd007893dd665b21ef55b275563472"
OUT = AAU / "runs" / "evidence" / "competitors_new" / "ulrssm"
RUNS = AAU / "runs" / "competitors_ulrssm"                 # checkpoint (fuori da git: *.pth)
DATA = REPO / "datasets" / "ULRSSM_FACES"                  # mesh di training generate (fuori da git)
VIEWS = ("hifi3d", "facescape", "faceverse", "faceverse_neutral", "famos")          # bp.VIEWS
MODES = ("fr", "sr")
K_FMAP = 200                                               # num_evecs (dataset) = n_eig di preprocess.py
# options/train/faust.yaml: rete, fmap, permutazione, ottimizzazione, perdite (in_channels dipende dall'ingresso)
CFG = {"out_channels": 256, "wks_channels": 128, "fmap_bidirectional": True, "tau": 0.07, "lr": 1.0e-3,
       "eta_min": 1.0e-4, "epochs": 15, "w_bij": 1.0, "w_orth": 1.0, "w_lap": 0.0, "w_align": 1.0, "clip": 1.0,
       "val_freq": 4000, "seed": 1234}
CHUNK = 4096                                               # righe per blocco della mappa sulla GPU


# ------------------------------------------------------------------------------------------- ULRSSM

def ulrssm():
    """I moduli di ULRSSM (importati dal clone, mai copiati): dict nome -> oggetto."""
    if not (ULRSSM_DIR / "networks" / "diffusion_network.py").exists():
        raise SystemExit(f"ULRSSM assente in {ULRSSM_DIR}: git clone https://github.com/dongliangcao/"
                         f"Unsupervised-Learning-of-Robust-Spectral-Shape-Matching.git {ULRSSM_DIR}")
    if str(ULRSSM_DIR) not in sys.path:
        sys.path.insert(0, str(ULRSSM_DIR))
    from networks.diffusion_network import DiffusionNet
    from networks.fmap_network import RegularizedFMNet
    from networks.permutation_network import Similarity
    from losses.fmap_loss import SURFMNetLoss, SquaredFrobeniusLoss
    from utils.geometry_util import get_operators
    return {"DiffusionNet": DiffusionNet, "RegularizedFMNet": RegularizedFMNet, "Similarity": Similarity,
            "SURFMNetLoss": SURFMNetLoss, "SquaredFrobeniusLoss": SquaredFrobeniusLoss, "get_operators": get_operators}


def feature_net(input_type: str, cache_dir: str | None):
    """DiffusionNet di ``faust.yaml`` (in 128 WKS o 3 xyz, out 256, il resto di default)."""
    m = ulrssm()
    cin = CFG["wks_channels"] if input_type == "wks" else 3
    return m["DiffusionNet"](in_channels=cin, out_channels=CFG["out_channels"], cache_dir=cache_dir,
                             input_type=input_type)


def load_net(ckpt: Path, device, cache_dir: str):
    """(rete in eval, input_type) da un checkpoint di ``ulr_train.py`` o da uno pubblicato (``checkpoints/*.pth``:
    ``networks.feature_extractor``; ingresso dedotto dalla prima lineare)."""
    import torch
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    fe = {k.replace("module.", ""): v for k, v in sd["networks"]["feature_extractor"].items()}   # resume_model
    input_type = sd.get("input_type") or ("wks" if fe["first_linear.weight"].shape[1] == CFG["wks_channels"] else "xyz")
    net = feature_net(input_type, cache_dir)
    net.load_state_dict(fe)
    return net.to(device).eval(), input_type


# --------------------------------------------------------------------------------------- preparazione

def orient(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, bool]:
    """(F, invertite?): facce invertite se la normale media pesata per area ha componente z negativa."""
    t = V[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]).sum(0)
    if n[2] < 0:
        return np.ascontiguousarray(F[:, ::-1]), True
    return F, False


def referenced(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solo i vertici referenziati dalle facce (le eigsh di ULRSSM non reggono vertici isolati: massa 0)."""
    used = np.unique(F)
    if len(used) == len(V):
        return V, F
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def normalize(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """``preprocess.py`` di ULRSSM: centro dei vertici, / sqrt(area totale) (somma della massa di potpourri3d)."""
    import potpourri3d as pp3d
    X = V - V.mean(0)
    return (X / np.sqrt(pp3d.vertex_areas(X, F).sum())).astype(np.float32)


def prep(V: np.ndarray, F: np.ndarray, cache_dir: str) -> dict:
    """Mesh pronta per ULRSSM: vertici normalizzati (float32), facce orientate (int64), operatori in cache;
    ``evecs`` (n, K_FMAP), ``evals`` (K_FMAP,), ``mass`` (n,) float32, ``flipped``, secondi."""
    import torch
    t0 = time.perf_counter()
    F, flipped = orient(V, np.asarray(F, dtype=np.int64))
    Xn = normalize(V, F)
    out = ulrssm()["get_operators"](torch.from_numpy(Xn), torch.from_numpy(F).long(), k=K_FMAP, cache_dir=cache_dir)
    return {"verts": Xn, "faces": F, "mass": out[1].numpy(), "evals": out[3].numpy(), "evecs": out[4].numpy(),
            "flipped": flipped, "seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------------------- mappa, distanze

def features(net, verts: np.ndarray, faces: np.ndarray, device):
    """Feature (n, 256) normalizzate L2 (``compute_permutation_matrix``), float32 sulla GPU."""
    import torch
    import torch.nn.functional as Fn
    with torch.no_grad():
        f = net(torch.from_numpy(verts)[None].to(device), torch.from_numpy(faces).long()[None].to(device))[0]
    return Fn.normalize(f.float(), dim=-1, p=2)


def _argmax_rows(Q, K, chunk: int = CHUNK):
    """argmax_k (Q_i . K_k) per ogni riga i di Q, a blocchi."""
    import torch
    out = torch.empty(len(Q), dtype=torch.long, device=Q.device)
    for a in range(0, len(Q), chunk):
        out[a:a + chunk] = (Q[a:a + chunk] @ K.T).argmax(1)
    return out


def p2p(fx, fy, ex, ey, my, chunk: int = CHUNK):
    """Mappa y -> x di ULRSSM (indice di un vertice di x per ogni vertice di y): feature normalizzate ``fx``, ``fy``,
    autovettori ``ex`` (nx, K), ``ey`` (ny, K), massa ``my`` (ny,)."""
    import torch
    pyx = _argmax_rows(fy.float(), fx.float(), chunk)              # Pyx hard (Similarity, argmax sulle x)
    Cxy = (ey.T * my[None]) @ ex[pyx]                               # evecs_trans_y @ Pyx @ evecs_x, (K, K)
    G = ex @ Cxy.T                                                  # (nx, K)
    g2 = (G * G).sum(1)
    out = torch.empty(len(ey), dtype=torch.long, device=ey.device)
    for a in range(0, len(ey), chunk):                              # nn_query(G, ey): argmin |g|^2 - 2 e.g
        out[a:a + chunk] = (g2[None] - 2.0 * ey[a:a + chunk] @ G.T).argmin(1)
    return out


def rigid_rms(X, Y, w):
    """RMS pesato (w) dopo la rigida pesata SENZA scala di X su Y (Umeyama, riflessione esclusa), float64."""
    import torch
    X, Y, w = X.double(), Y.double(), w.double()
    wn = w / w.sum()
    mx, my = wn @ X, wn @ Y
    C = ((Y - my) * wn[:, None]).T @ (X - mx)
    U, _, Vt = torch.linalg.svd(C)
    d = torch.sign(torch.linalg.det(U @ Vt))
    D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    R = U @ D @ Vt
    r = (X - mx) @ R.T + my - Y
    return float(torch.sqrt(wn @ (r * r).sum(1)))


def pair_rms(A: dict, B: dict, chunk: int = CHUNK) -> tuple[float, float, float, float]:
    """(FR, SR, FR B->A, FR A->B) di una coppia; ``A``, ``B`` = dict sulla GPU con ``f`` (feature), ``e`` (evecs),
    ``m`` (massa normalizzata), ``V`` (mm), ``w`` (aree mm^2), ``s`` (CS_ref / CS_i)."""
    ba = p2p(A["f"], B["f"], A["e"], B["e"], B["m"], chunk)        # ogni vertice di B -> un vertice di A
    ab = p2p(B["f"], A["f"], B["e"], A["e"], A["m"], chunk)
    fr1 = rigid_rms(A["V"][ba], B["V"], B["w"])
    fr2 = rigid_rms(B["V"][ab], A["V"], A["w"])
    sr1 = rigid_rms(A["V"][ba] * A["s"], B["V"] * B["s"], B["w"])
    sr2 = rigid_rms(B["V"][ab] * B["s"], A["V"] * A["s"], A["w"])
    return 0.5 * (fr1 + fr2), 0.5 * (sr1 + sr2), fr1, fr2


# ------------------------------------------------------------------------------------------- viste

def view(name: str, spec: Path | None = None) -> dict:
    """{name, domain, items [(soggetto, topologia, percorso)], cs_ref, frame | None, scalars (cache o None)}
    (``gp_common.view``, stesso formato della spec)."""
    if spec is not None:
        s = json.loads(Path(spec).read_text())
        d = Path(s["dir"]) if Path(s["dir"]).is_absolute() else REPO / s["dir"]
        subj = s.get("subjects") or json.loads(Path(s["subjects_file"]).read_text())
        items = [(str(x), t, d / f"{x}_GTready_{t}.npz") for x in sorted(subj) for t in s["topologies"]]
        return {"name": s["name"], "domain": s["name"], "items": items, "cs_ref": float(s["cs_ref"]),
                "frame": s["frame"], "scalars": None}
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
    return {"name": name, "domain": blmm.VIEWS[name]["domain"], "items": items, "cs_ref": float(prm["cs_ref"]),
            "frame": None, "scalars": sc}


def to_mm(v: dict, V: np.ndarray, path: Path) -> np.ndarray:
    if v["frame"] is None:
        return blmm.to_mm(v["name"], V, Path(path).stem)
    f = v["frame"]
    return float(f["u"]) * V @ np.asarray(f["R"], dtype=np.float64).T + np.asarray(f["t"], dtype=np.float64)


def size_of(v: dict, path: Path, V: np.ndarray, F: np.ndarray) -> tuple[float, str]:
    """Centroid size robusta (mm) e la sua fonte (cache della vista o ``blmm.mesh_scalars``)."""
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


# ------------------------------------------------------------------------------------------- template

def template_gnm() -> dict:
    """Template fisso della variante per mesh: la media di GNM Head (hockey_mask, un generatore di training) nel
    frame canonico dei generatori (``sources.MMSource.view_mesh`` con z = 0, senza espressione), e la sua mappa
    baricentrica sui 1478 punti della regione unificata (``unified_space.npz``, posizioni nella patch come in
    ``MMSource.__init__``). Controllo: i punti mappati della media = ``mean_u`` della sorgente nel frame canonico."""
    sys.path.insert(0, str(REPO / "v3_work" / "stream"))
    import sources as S
    uni = S.Unified()
    src = S.MMSource("gnm", uni)
    V, F, tag = src.view_mesh(np.zeros(src.n_id), np.random.default_rng(0), False)
    if tag != "neutral" or src.subdiv or src.work is not None:
        raise SystemExit("template GNM: vista inattesa")
    vidx, bary = uni.map_of("gnm")
    rv = np.asarray(src.model.region_vertices if src.model.region_vertices is not None else np.arange(src.model.n_verts))
    pos = np.searchsorted(rv, vidx)
    c = src.canon
    ref = c["scale"] * (src.mean_u @ np.asarray(c["R"]).T) + np.asarray(c["t"])
    err = float(np.abs(np.einsum("nkd,nk->nd", V[pos], bary) - ref).max())
    if err > 1e-6:
        raise SystemExit(f"template GNM: punti unificati diversi da mean_u ({err:.2e})")
    return {"V": np.asarray(V, dtype=np.float64), "F": np.asarray(F, dtype=np.int64), "pos": pos, "bary": bary,
            "n_unified": int(len(pos)), "check_unified": err}


# ------------------------------------------------------------------------------------------- uscite

def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def stamp() -> dict:
    return {"host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID", "none"),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}
