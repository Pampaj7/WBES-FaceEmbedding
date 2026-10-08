"""Pesi per area dei vertici, normali o robusti al rumore, per il centro dell'input, la scala e il pooling.

Perche' (E3, aau/runs/evidence/e3/summary.md): il 74% degli errori di riconoscimento su HIFI3D cade su coppie
con down8k, perche' sia il centro dell'input del loader sia il pooling medio sono medie PER VERTICE e dipendono
dalla densita'. Pesare per area entrambi (solo al test) porta original<->down8k a 1.00 ma noisy scende a ~0.7:
il rumore gonfia le aree (su una mesh BFM l'area totale della noisy e' 2.3 volte l'original,
v2_work/fastio/area_norm_data.py), e con esse i pesi e la scala sqrt(area).

Modi (``--area-weights``):
  * ``mass``    aree baricentriche dei vertici (1/3 delle facce incidenti) della mesh com'e';
  * ``smooth``  le stesse aree calcolate sulla geometria PASSA-BASSO: V proiettata sui primi ``k`` autovettori del
                Laplaciano del campione (V_s = Phi_k Phi_k^T M V, Phi M-ortonormali come in diffusion-net). Le
                autofunzioni non dipendono dalla discretizzazione, quindi il filtro toglie il rumore ad alta
                frequenza allo stesso modo su mesh dense e rade (un lisciamento laplaciano uniforme a iterazioni no:
                il suo passo e' la lunghezza degli spigoli);
  * ``winsor``  aree ``mass`` limitate al percentile ``lo``-``hi`` della mesh (default 1-99).
Tutte le funzioni sono torch e girano uguali su CPU (servizio dei campioni) e GPU (pooling, eval).
"""
from __future__ import annotations

import torch

AREA_WEIGHTS = ("mass", "smooth", "winsor")
CFG = {"k": 64, "lo": 1.0, "hi": 99.0}   # impostati dal trainer (--area-smooth-k, --winsor-pct)


def lumped_areas(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Area baricentrica per vertice: 1/3 dell'area di ogni faccia incidente. V [n,3], F [m,3] -> [n]."""
    F = F.long()
    tri = V[F]
    a = 0.5 * torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
    w = V.new_zeros(V.shape[0])
    for j in range(3):
        w = w.index_add(0, F[:, j], a / 3.0)
    return w


def lowpass(V: torch.Tensor, mass: torch.Tensor, evecs: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """Proiezione M-ortogonale di V sulle prime k autofunzioni."""
    phi = evecs[:, : (k or CFG["k"])].to(V.dtype)
    return phi @ (phi.transpose(0, 1) @ (mass.reshape(-1, 1).to(V.dtype) * V))


def area_weights(mode: str, V: torch.Tensor, F: torch.Tensor, mass: torch.Tensor,
                 evecs: torch.Tensor) -> torch.Tensor:
    """Pesi per vertice (non normalizzati) secondo ``mode``."""
    if mode == "mass":
        return lumped_areas(V, F)
    if mode == "smooth":
        return lumped_areas(lowpass(V, mass, evecs), F)
    if mode == "winsor":
        w = lumped_areas(V, F)
        q = torch.quantile(w, torch.tensor([CFG["lo"] / 100.0, CFG["hi"] / 100.0], dtype=w.dtype, device=w.device))
        return w.clamp(q[0], q[1])
    raise ValueError(f"area-weights {mode!r}")


def pool_weights(mode: str, V: torch.Tensor, F: torch.Tensor | None, mass: torch.Tensor,
                 evecs: torch.Tensor) -> torch.Tensor:
    """Pesi del pooling: ``mass`` = la massa degli operatori (come E3); ``winsor`` la limita ai percentili;
    ``smooth`` = aree della geometria passa-basso (serve F)."""
    m = mass.reshape(-1)
    if mode == "mass":
        return m
    if mode == "winsor":
        q = torch.quantile(m, torch.tensor([CFG["lo"] / 100.0, CFG["hi"] / 100.0], dtype=m.dtype, device=m.device))
        return m.clamp(q[0], q[1])
    if F is None:
        raise ValueError("pooling con aree 'smooth': servono le facce (non usare --fast-data)")
    return area_weights("smooth", V, F, mass, evecs)


def area_frame(mode: str, V: torch.Tensor, F: torch.Tensor, mass: torch.Tensor, evecs: torch.Tensor) -> torch.Tensor:
    """Centro pesato e scala sqrt(somma dei pesi) con i pesi di ``mode``: (V - c) / sqrt(A).

    Con ``mass`` coincide con data_v3.reframe_sqrt_area. Come quella, non dipende dal frame d'ingresso (le aree
    scalano col quadrato, il passa-basso commuta con traslazione e scala uniforme)."""
    w = area_weights(mode, V, F, mass, evecs)
    tot = w.sum()
    if not torch.isfinite(tot) or float(tot) <= 0:
        raise ValueError("aree degeneri: impossibile il frame per area")
    c = (w.unsqueeze(1) * V).sum(0, keepdim=True) / tot
    return (V - c) / torch.sqrt(tot)
