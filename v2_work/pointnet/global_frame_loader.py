#!/usr/bin/env python
"""Il frame globale, applicato ai sample della cache invertendo la normalizzazione per-mesh.

PERCHE' NON PUO' STARE IN `frames.py`. Quel modulo si applica alle coordinate gia' normalizzate
e si giustifica con una cancellazione: `rms` e `area` sono, come la mappa del loader, una
traslazione piu' una scala uniforme, quindi comporle con la mappa precedente da' lo stesso
risultato che applicarle alla mesh grezza. **Il frame globale non gode di questa proprieta'.**
La sua traslazione e la sua scala sono costanti dell'insieme, non funzioni della singola mesh;
comporle sopra una normalizzazione per-mesh lascerebbe dentro il `c_i` e l's_i' della mesh, cioe'
proprio l'informazione che vogliamo togliere. Il frame globale va applicato alle coordinate
grezze, e qui lo si fa invertendo prima la mappa del loader.

L'INVERSIONE E' ESATTA. Il loader (`dataset_gtready.py:299-306`) calcola
    V_c = (V_r - c_i) / s_i,   c_i = media dei vertici,   s_i = max|V_r - c_i|
quindi V_r = V_c * s_i + c_i. Bastano `c_i` e `s_i`, che si ricavano leggendo solo `verts`
dall'npz -- niente operatori, niente autovettori.

    V_globale = (V_c * s_i + c_i - c0) / s0

COSA CAMBIA, CONCRETAMENTE. Sotto `maxabs` una mesh `crop` viene gonfiata del 14% rispetto alla
stessa faccia in `original`, perche' il crop toglie il vertice che fissa il divisore. Sotto il
frame globale la mesh `crop` resta piu' piccola, che e' vero: e' una forma parziale. La rete
riceve la differenza invece di vedersela normalizzare via.

c0 e s0 vanno stimati SOLO sulle identita' di training (v2_work/xdomain/global_frame.py), e
sono UNA coppia per tutto l'insieme, non una per topologia: una costante per topologia
reintrodurrebbe esattamente il riscalamento dipendente dalla topologia che stiamo togliendo.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def build_scale_table(npz_dirs, verbose: bool = True) -> dict[str, tuple[np.ndarray, float]]:
    """{nome mesh -> (c_i, s_i)} leggendo solo `verts`. Una passata, una volta sola."""
    table: dict[str, tuple[np.ndarray, float]] = {}
    for d in npz_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.npz")):
            with np.load(p, allow_pickle=False, mmap_mode="r") as z:
                key = "verts" if "verts" in z else "V"
                V = np.asarray(z[key], dtype=np.float64)
            c = V.mean(axis=0)
            s = float(np.abs(V - c).max())
            table[p.name[:-4]] = (c, s if s > 1e-6 else 1.0)
    if verbose:
        print(f"[globalframe] tabella di scala: {len(table)} mesh", flush=True)
    if not table:
        raise SystemExit(f"nessun npz trovato in {list(npz_dirs)}")
    return table


def install(npz_dirs, frame_json: Path, verbose: bool = True) -> None:
    """Rimpiazza la normalizzazione per-mesh col frame globale, dentro la cache."""
    fr = json.loads(Path(frame_json).read_text())
    c0 = np.asarray(fr["c0"], dtype=np.float64)
    s0 = float(fr["s0"])
    if not np.isfinite(c0).all() or not np.isfinite(s0) or s0 <= 0:
        raise SystemExit(f"frame globale non valido in {frame_json}: c0={c0}, s0={s0}")
    table = build_scale_table(npz_dirs, verbose)

    import fast_data as fd
    orig = fd.CachedDataset.__getitem__
    seen = {"hit": 0, "miss": 0}

    def __getitem__(self, idx):
        sample = orig(self, idx)
        v = sample.get("verts")
        name = sample.get("name")
        if v is None or name not in table:
            seen["miss"] += 1
            return sample
        seen["hit"] += 1
        c_i, s_i = table[name]
        dev, dt = v.device, v.dtype
        c_i_t = torch.as_tensor(c_i, dtype=torch.float64, device=dev)
        raw = v.to(torch.float64) * s_i + c_i_t
        out = dict(sample)
        out["verts"] = ((raw - torch.as_tensor(c0, dtype=torch.float64, device=dev)) / s0).to(dt)
        return out

    fd.CachedDataset.__getitem__ = __getitem__
    fd._globalframe_counters = seen        # ispezionabile a fine run
    if verbose:
        print(f"[globalframe] c0={np.round(c0,1)} s0={s0:.1f}  "
              f"(stimati su {fr.get('n_train_subjects','?')} identita' di training)", flush=True)


def _demo() -> None:
    """Il controllo che conta: l'inversione deve restituire le coordinate grezze esatte."""
    rng = np.random.default_rng(0)
    V_raw = rng.normal(size=(400, 3)) * 5000 + np.array([200.0, -9000.0, -33000.0])

    # esattamente cio' che fa il loader
    c_i = V_raw.mean(axis=0)
    s_i = float(np.abs(V_raw - c_i).max())
    V_c = (V_raw - c_i) / s_i

    back = V_c * s_i + c_i
    err = np.abs(back - V_raw).max() / np.abs(V_raw).max()
    assert err < 1e-12, f"l'inversione non e' esatta: errore relativo {err:.3e}"

    c0, s0 = np.array([250.0, -9970.0, -33800.0]), 56600.0
    got = (back - c0) / s0
    want = (V_raw - c0) / s0
    assert np.abs(got - want).max() < 1e-9

    # e la proprieta' per cui esiste: una similarita' globale non cambia i ranghi delle distanze
    A = rng.normal(size=(30, 60, 3)) * 5000
    d_raw = np.array([np.linalg.norm(A[i] - A[j], axis=1).mean()
                      for i in range(30) for j in range(i + 1, 30)])
    B = (A - c0) / s0
    d_glb = np.array([np.linalg.norm(B[i] - B[j], axis=1).mean()
                      for i in range(30) for j in range(i + 1, 30)])
    ratio = d_glb / d_raw
    assert ratio.max() / ratio.min() - 1 < 1e-12, "non e' una similarita' pura"

    # per contrasto: la normalizzazione per-mesh li cambia eccome
    C = np.stack([(a - a.mean(0)) / np.abs(a - a.mean(0)).max() for a in A])
    d_max = np.array([np.linalg.norm(C[i] - C[j], axis=1).mean()
                      for i in range(30) for j in range(i + 1, 30)])
    from scipy.stats import spearmanr
    rho = float(spearmanr(d_raw, d_max).statistic)
    assert rho < 0.999, "atteso che maxabs per-mesh perturbi i ranghi"
    print(f"OK  inversione esatta; similarita' globale rho=1, maxabs per-mesh rho={rho:.4f}")


if __name__ == "__main__":
    _demo()
