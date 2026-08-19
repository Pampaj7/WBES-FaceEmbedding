#!/usr/bin/env python
"""Il ground truth dipende dal frame in cui lo misuri. Quanto, e quale frame lo salva?

Il bersaglio su cui alleniamo e' d(i,j) = media sui vertici di ||V_i[v] - V_j[v]||, calcolato
dopo la normalizzazione per-mesh del loader (centro sulla media dei vertici, dividi per
maxabs). Misurato sulle STESSE 500 identita' in due topologie diverse -- BFM a 23.470 vertici e
FLAME a 1.930 -- l'ordinamento indotto concorda con se stesso a:

    raw     rho = 0.8653
    maxabs  rho = 0.5734      <- la convenzione effettivamente usata

Ventinove punti persi nella normalizzazione, sul BERSAGLIO e non sul modello. Nessuna rete puo'
superare il proprio target, quindi questo e' un tetto su tutto cio' che misuriamo.

Questo script aggiunge due frame che sui nostri assi battevano maxabs ovunque, e chiede quale
rende il ground truth piu' indipendente dalla topologia:

    raw     nessuna normalizzazione
    maxabs  media dei vertici, diviso il vertice piu' lontano        (attuale)
    rms     centroide pesato per area, diviso raggio quadratico medio pesato per area
    area    centroide pesato per area, diviso sqrt(area totale)
    global_rms  centroide per-mesh, ma diviso una COSTANTE unica per tutto l'insieme

L'ultimo frame esiste per separare due spiegazioni del vantaggio di `raw`. Se raw vince perche'
la TAGLIA ASSOLUTA del volto e' informazione d'identita', allora global_rms -- che centra ogni
mesh ma preserva le differenze di taglia fra identita' -- deve valere quanto raw. Se invece raw
vince solo perche' evita la normalizzazione per-mesh, global_rms deve valere quanto rms.
La distinzione conta: un bersaglio che dipende dalla taglia assoluta e' inservibile se
l'ingresso del modello la taglia via, ed e' esattamente cio' che il loader fa.

Nota sul confronto: lo Spearman e' basato sui ranghi, quindi la scala assoluta e' irrilevante e
frame con unita' incomparabili si confrontano lecitamente. Cio' che si misura e' se l'ORDINE
delle identita' sopravvive al cambio di topologia.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

THIS = Path(__file__).resolve().parent
FRAMES = ("raw", "maxabs", "rms", "area", "global_rms")


def vertex_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area baricentrica per vertice: un terzo di ogni triangolo incidente.

    Ricostruita dalle facce e non letta da `mass`, perche' il set in topologia FLAME non porta
    gli operatori. Deve coincidere con la massa di DiffusionNet, ed e' stato verificato altrove
    che coincide a 1e-8 sulla somma e a 4e-10 sul raggio rms.
    """
    tri = V[F]
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    out = np.zeros(len(V))
    for c in range(3):
        np.add.at(out, F[:, c], a / 3.0)
    return out


def reframe(V: np.ndarray, F: np.ndarray, frame: str, gscale: float | None = None) -> np.ndarray:
    if frame == "raw":
        return V
    if frame == "maxabs":
        Vn = V - V.mean(0, keepdims=True)
        return Vn / max(float(np.abs(Vn).max()), 1e-12)
    w = vertex_areas(V, F)
    tot = float(w.sum())
    if not np.isfinite(tot) or tot <= 0:
        raise ValueError("aree per vertice non valide")
    w = w / tot
    X = V - (w[:, None] * V).sum(0, keepdims=True)
    if frame == "global_rms":
        return X / max(float(gscale), 1e-12)
    if frame == "rms":
        s = float(np.sqrt((w * (X * X).sum(1)).sum()))
    elif frame == "area":
        s = float(np.sqrt(tot))
    else:
        raise ValueError(frame)
    return X / max(s, 1e-12)


def load_set(d: Path, n_max: int) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    """Le due sorgenti usano chiavi diverse: verts/faces da un lato, V/F dall'altro."""
    files = sorted(d.glob("*_GTready_original.npz"))[:n_max]
    if not files:
        raise SystemExit(f"nessuna mesh original in {d}")
    names, verts, faces = [], [], None
    for p in files:
        with np.load(p) as z:
            V = (z["verts"] if "verts" in z else z["V"]).astype(np.float64)
            F = (z["faces"] if "faces" in z else z["F"]).astype(np.int64)
        names.append(p.name.split("_GTready")[0])
        verts.append(V)
        faces = F if faces is None else faces
    return names, verts, faces


def pairwise(verts: list[np.ndarray], device: str) -> np.ndarray:
    """D[i,j] = media sui vertici della distanza L2. La corrispondenza densa vale per
    costruzione dentro un 3DMM, che e' l'assunzione su cui l'intero ground truth poggia."""
    V = torch.tensor(np.stack(verts), dtype=torch.float32, device=device)   # (n, nv, 3)
    n = V.shape[0]
    D = torch.zeros(n, n, dtype=torch.float64, device=device)
    for i in range(n):
        D[i] = (V[i].unsqueeze(0) - V).pow(2).sum(-1).sqrt().mean(-1).double()
    return D.cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfm-dir", type=Path, default=Path("datasets/REMESH/npz_data_topo_500_withops"))
    ap.add_argument("--flame-dir", type=Path, default=THIS / "bfm_in_flame")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", type=Path, default=THIS / "gt_matrices" / "frame_agreement.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nb, vb, fb = load_set(args.bfm_dir, args.n)
    nf, vf, ff = load_set(args.flame_dir, args.n)
    assert nb == nf, "ordine delle identita' diverso fra le due topologie"
    print(f"{len(nb)} identita'   BFM {vb[0].shape[0]} vert   FLAME {vf[0].shape[0]} vert   [{device}]")

    iu = np.triu_indices(len(nb), 1)
    # una sola costante per topologia: la mediana dei raggi rms. Preserva le differenze di
    # taglia FRA identita' e rimuove solo la differenza di unita' fra le due topologie.
    def gs(verts, F):
        return float(np.median([
            np.sqrt((lambda w, X: (w * (X * X).sum(1)).sum())(
                (a := vertex_areas(V, F)) / a.sum(),
                V - ((a / a.sum())[:, None] * V).sum(0, keepdims=True))) for V in verts]))
    gb, gf = gs(vb, fb), gs(vf, ff)
    print(f"scala globale: BFM {gb:.6g}, FLAME {gf:.6g}")

    res = {}
    for frame in FRAMES:
        Db = pairwise([reframe(V, fb, frame, gb) for V in vb], device)
        Df = pairwise([reframe(V, ff, frame, gf) for V in vf], device)
        rho = float(spearmanr(Db[iu], Df[iu]).statistic)
        res[frame] = {"spearman_cross_topology": rho}
        print(f"  {frame:7s} rho(BFM-topo, FLAME-topo) = {rho:.4f}")

    best = max(res, key=lambda k: res[k]["spearman_cross_topology"])
    cur = res["maxabs"]["spearman_cross_topology"]
    print(f"\nmigliore: {best} ({res[best]['spearman_cross_topology']:.4f}), "
          f"attuale maxabs {cur:.4f}, guadagno {res[best]['spearman_cross_topology'] - cur:+.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_identities": len(nb), "frames": res, "best": best}, indent=2))
    print(f"scritto in {args.out}")


if __name__ == "__main__":
    main()
