"""Prodotti interni varifold/currents su un device qualunque, a blocchi.

Trascrizione di ``v2_work/phase0/measure_distances._inner`` con due sole differenze, e
nessuna delle due tocca la matematica:

1. gli accumulatori restano sul device fino alla fine.  L'originale somma dentro un
   tensore CPU float64 (``out[k] += torch.dot(...).double()``), e con tensori CUDA quella
   riga solleva un errore di device: e' l'unico motivo per cui il kernel di phase0 non
   gira su GPU cosi' com'e';
2. varifold e currents si calcolano in **una passata sola**.  Il costo e' l'esponenziale
   della gaussiana, che e' identico per i due kind: cambia solo il peso del blocco,
   ``(n_i . n_j)^2 a_i a_j`` contro ``(n_i . n_j) a_i a_j``.  Chiederli insieme dimezza il
   conto rispetto a due chiamate.

Il modulo di phase0 NON e' modificato: ``--self-test`` di ``geometric_matrix.py``
confronta le due implementazioni sulle stesse mesh e stampa lo scarto relativo, che deve
restare nel rumore del float32.
"""

from __future__ import annotations

from typing import Sequence

import torch

KINDS = ("varifold", "currents")


def inner_products(mA: dict, mB: dict, sigmas: Sequence[float], kinds=KINDS,
                   block: int = 2048) -> dict[str, torch.Tensor]:
    """``{kind: tensore (len(sigmas),) di <A, B>}``, in tile (block x block)."""
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise ValueError(f"kind sconosciuto: {unknown} (attesi {list(KINDS)})")

    cA, nA, aA = mA["centroids"], mA["normals"], mA["areas"]
    cB, nB, aB = mB["centroids"], mB["normals"], mB["areas"]
    device = cA.device
    inv_s2 = torch.tensor([1.0 / (float(s) ** 2) for s in sigmas],
                          dtype=torch.float32, device=device)
    out = {k: torch.zeros(len(sigmas), dtype=torch.float64, device=device) for k in kinds}

    for i in range(0, len(aA), block):
        ci, ni, ai = cA[i:i + block], nA[i:i + block], aA[i:i + block]
        for j in range(0, len(aB), block):
            cj, nj, aj = cB[j:j + block], nB[j:j + block], aB[j:j + block]
            nd2 = torch.cdist(ci, cj)
            nd2 = nd2.mul_(nd2).neg_()                     # -||c_i - c_j||^2, (bi, bj)
            dot = ni @ nj.T                                # tensore nuovo, in-place sicuro
            areas = ai[:, None] * aj[None, :]
            # Stesso ordine delle moltiplicazioni di phase0: (n.n)^2 prima, aree poi.
            weights = {}
            if "varifold" in kinds:
                weights["varifold"] = (dot * dot).mul_(areas).reshape(-1)
            if "currents" in kinds:
                weights["currents"] = (dot * areas).reshape(-1)
            for k in range(len(sigmas)):
                # clamp_min_ come in phase0: sotto -87 exp() esce dai normali float32 e il
                # percorso di underflow costa ~8x, mentre il valore e' 1e-38, cioe' niente.
                e = torch.exp((nd2 * inv_s2[k]).clamp_min_(-87.0)).reshape(-1)
                for kind in kinds:
                    out[kind][k] += torch.dot(e, weights[kind]).double()
    return out


def self_inner(m: dict, sigmas, kinds=KINDS, block: int = 2048) -> dict[str, torch.Tensor]:
    """``<X, X>`` per kind, memorizzato dentro la misura come fa ``_self_inner``."""
    key = (tuple(sorted(kinds)), tuple(float(s) for s in sigmas), block)
    cache = m.setdefault("_cache_multi", {})
    if key not in cache:
        cache[key] = inner_products(m, m, sigmas, kinds, block)
    return cache[key]


def distances(mA: dict, mB: dict, sigmas, kinds=KINDS, block: int = 2048) -> dict[str, float]:
    """``{kind: d}`` con la stessa formula di ``measure_distances._distance``."""
    xx = self_inner(mA, sigmas, kinds, block)
    yy = self_inner(mB, sigmas, kinds, block)
    xy = inner_products(mA, mB, sigmas, kinds, block)
    return {k: float(torch.sqrt((xx[k] + yy[k] - 2.0 * xy[k]).clamp_min(0.0).sum()))
            for k in kinds}
