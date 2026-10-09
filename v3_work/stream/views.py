"""Una vista: discretizzazione, operatori DiffusionNet, campione del loader, formato compatto dello shard.

Discretizzazioni (``LABELS``), gli STESSI generatori dei dati attuali, importati senza modifiche
(``v2_work/genict/make_ict_topologies.py`` + ``mesh_ops.py``, come aau/data_scale/gen_ict_shard.py):
  original  la mesh di lavoro (prepare_open_surface)
  remesh    2 iterazioni di smoothing + decimazione 0.7x
  down8k    decimazione a 0.345x i triangoli della original (il rapporto BFM 16k / 46.440)
  noisy     rumore gaussiano 0.003 x diagonale del bbox, stessa topologia
  crop      taglio della banda di bordo (datasets/remesh.py)
  up60k     suddivisione 1-a-4 + decimazione a 2.584x (V fino a ~2.6x la original: a bassa frequenza)
La geometria passa per float32 dopo il generatore, come nelle npz di gen_ict_shard: gli operatori vedono gli
stessi numeri del pre-pass sulla stessa mesh.

Operatori: ``operators`` = il corpo di aau/data_scale/prepass_ops.py::ops_areanorm (mesh centrata, area 1,
``compute_operators`` con ``k_eig`` a scelta), senza la scrittura dell'npz; col ``build_grad`` vettorizzato di
E9 (``install_grad_vec``, identico bit per bit dopo il cast a fp32). ``serve_like_loader`` = le righe di
``GTReadyDatasetNPZ.__getitem__`` (centro e maxabs dei vertici, autovalori / lambda_max, gradienti /
sqrt(lambda_max)) sugli stessi array che il loader leggerebbe dall'npz; L non serve (la diffusione spettrale non
lo legge: e' la cache compatta del trainer). tests/test_ops_equiv.py li confronta col percorso attuale.

Compatto (``compact``): vertici, massa, autovalori e valori dei gradienti fp32; facce e indici COO int32;
autovettori fp16 (default) o fp32, scritti come (k, n) contiguo quando il loader li da' in ordine colonna
(stride (1, n), ``evecs_f``), cosi' il consumatore li serve con ``.t()`` nello stesso layout del loader.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (REPO_ROOT / "v2_work" / "genict", REPO_ROOT / "v2_work" / "potential", REPO_ROOT / "v3_work" / "trainer",
           REPO_ROOT / "diffusion-net" / "src", REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mesh_ops as mo  # noqa: E402
from make_ict_topologies import make_down8k, make_noisy, make_remesh, make_up60k, triangle_targets  # noqa: E402

LABELS = ("original", "remesh", "down8k", "noisy", "crop", "up60k")
LABEL_WEIGHTS = {"original": 1.0, "remesh": 1.0, "down8k": 1.0, "noisy": 1.0, "crop": 1.0, "up60k": 0.25}
SPARSE = ("L", "gradX", "gradY")
EVECS_DTYPES = {"fp16": np.float16, "fp32": np.float32}
FIELDS = ("verts", "faces", "mass", "evals", "evecs", "gxi", "gxv", "gyi", "gyv")


def install_grad_vec() -> None:
    import grad_vec   # v3_work/trainer/grad_vec.py: sostituisce geometry.build_grad nel solo processo corrente
    grad_vec.install()


def discretize(V: np.ndarray, F: np.ndarray, label: str, noise_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(V float64 arrotondata a float32, F int32) della discretizzazione ``label`` della mesh di lavoro."""
    base = mo.prepare_open_surface(V, F)
    down_t, up_t = triangle_targets(len(base[1]))
    if label == "original":
        out = base
    elif label == "remesh":
        out = make_remesh(*base)
    elif label == "crop":
        out = mo.make_crop(*base)
    elif label == "noisy":
        out = make_noisy(*base, seed=int(noise_seed))
    elif label == "down8k":
        out = make_down8k(*base, target=down_t)
    elif label == "up60k":
        out = make_up60k(*base, target=up_t)
    else:
        raise ValueError(f"discretizzazione {label!r} non in {LABELS}")
    Vo, Fo = out
    return np.asarray(Vo, dtype=np.float32).astype(np.float64), np.ascontiguousarray(Fo, dtype=np.int32)


def operators(V: np.ndarray, F: np.ndarray, k_eig: int) -> dict:
    """== prepass_ops.ops_areanorm (convenzione ``areanorm``) senza il file: le chiavi dell'npz che scriverebbe."""
    import torch
    from areanorm_operators import total_area
    from diffusion_net.geometry import compute_operators

    A = total_area(V, F)
    if not np.isfinite(A) or A <= 0:
        raise ValueError(f"area non valida: {A}")
    V = (V - V.mean(0)) / np.sqrt(A)
    Vt = torch.tensor(V, dtype=torch.float32)
    Ft = torch.tensor(F, dtype=torch.int32)
    _, mass, L, evals, evecs, gX, gY = compute_operators(Vt, Ft, k_eig=int(k_eig))
    data = {"verts": V.astype(np.float32), "faces": F.astype(np.int32), "mass": mass.numpy(),
            "evals": evals.numpy(), "evecs": evecs.numpy()}
    for name, t in zip(SPARSE, (L, gX, gY)):
        c = t.coalesce()
        data[f"{name}_indices"] = c.indices().numpy().astype(np.int32)
        data[f"{name}_values"] = c.values().numpy()
        data[f"{name}_shape"] = np.array(c.shape)
    return data


def serve_like_loader(data: dict) -> dict:
    """Le righe di GTReadyDatasetNPZ.__getitem__ (face_embedding/gt_encdec/autoencoder/dataset_gtready.py:294-381)
    sugli array dell'npz, nello stesso ordine e negli stessi tipi; senza L e con le facce int32 (il loader le
    da' int64: valori identici, il consumatore le riporta a int64)."""
    import torch
    from dataset_gtready import GTReadyDatasetNPZ, scale_sparse_tensor

    V = data["verts"].astype(np.float64)
    V = V - V.mean(axis=0, keepdims=True)
    scale = np.max(np.abs(V))
    V = V / scale if scale > 1e-6 else V * 0.0
    V_torch = torch.tensor(V, dtype=torch.float32).contiguous()
    mass = torch.from_numpy(np.asarray(data["mass"])).float().flatten()
    evals = torch.from_numpy(np.asarray(data["evals"])).float()
    evecs = torch.from_numpy(np.asarray(data["evecs"])).float()
    gradX = GTReadyDatasetNPZ.coo_dict_to_sparse_tensor(None, data, "gradX")
    gradY = GTReadyDatasetNPZ.coo_dict_to_sparse_tensor(None, data, "gradY")
    if gradX is None or gradY is None:
        raise RuntimeError("gradienti non validi")
    mass = torch.nan_to_num(mass, nan=1e-6, posinf=1e6, neginf=-1e6)
    evals = torch.nan_to_num(evals, nan=0.0, posinf=1e6, neginf=-1e6)
    evecs = torch.nan_to_num(evecs, nan=0.0, posinf=1e6, neginf=-1e6)
    evals = torch.clamp(evals, min=0.0)
    lambda_max = float(evals.max().item()) + 1e-9
    evals = evals / lambda_max
    lambda_max_sqrt = np.sqrt(max(lambda_max, 1e-9))
    gradX = scale_sparse_tensor(gradX, lambda_max_sqrt)
    gradY = scale_sparse_tensor(gradY, lambda_max_sqrt)
    for k, t in {"V": V_torch, "mass": mass, "evals": evals, "evecs": evecs}.items():
        if not bool(torch.isfinite(t).all()):
            raise RuntimeError(f"valori non finiti in {k}")
    return {"verts": V_torch, "faces": torch.from_numpy(np.ascontiguousarray(data["faces"], dtype=np.int32)),
            "mass": mass, "evals": evals, "evecs": evecs, "gradX": gradX, "gradY": gradY}


def compact(s: dict, evecs_dtype: str = "fp16") -> tuple[dict, bool]:
    """Array dello shard (``FIELDS``) e ``evecs_f`` (build_store.compact_arrays, con gli autovettori in fp16)."""
    gx, gy = s["gradX"].coalesce(), s["gradY"].coalesce()
    ev = s["evecs"]
    ef = ev.dim() == 2 and ev.shape[1] > 1 and ev.stride() == (1, ev.shape[0])
    dt = EVECS_DTYPES[evecs_dtype]
    arr = {"verts": s["verts"].numpy().astype(np.float32, copy=False),
           "faces": s["faces"].numpy().astype(np.int32, copy=False),
           "mass": s["mass"].numpy().astype(np.float32, copy=False),
           "evals": s["evals"].numpy().astype(np.float32, copy=False),
           "evecs": np.ascontiguousarray((ev.t() if ef else ev).numpy()).astype(dt),
           "gxi": gx.indices().numpy().astype(np.int32), "gxv": gx.values().numpy().astype(np.float32, copy=False),
           "gyi": gy.indices().numpy().astype(np.int32), "gyv": gy.values().numpy().astype(np.float32, copy=False)}
    return arr, bool(ef)


def make_view(V: np.ndarray, F: np.ndarray, label: str, k_eig: int, evecs_dtype: str,
              noise_seed: int) -> tuple[dict, dict]:
    """(array dello shard, metadati + tempi in s per fase) di una vista della mesh di lavoro (V, F)."""
    t0 = time.perf_counter()
    Vd, Fd = discretize(V, F, label, noise_seed)
    t1 = time.perf_counter()
    data = operators(Vd, Fd, k_eig)
    t2 = time.perf_counter()
    arr, ef = compact(serve_like_loader(data), evecs_dtype)
    t3 = time.perf_counter()
    meta = {"label": label, "n": int(len(arr["verts"])), "m": int(len(arr["faces"])), "k": int(len(arr["evals"])),
            "nx": int(len(arr["gxv"])), "ny": int(len(arr["gyv"])), "evecs_f": ef,
            "evecs_dtype": evecs_dtype}
    return arr, {**meta, "t_gen": t1 - t0, "t_ops": t2 - t1, "t_pack": t3 - t2}
