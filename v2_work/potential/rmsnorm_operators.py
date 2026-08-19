#!/usr/bin/env python
"""Standard DiffusionNet operators computed on RMS-NORMALISED meshes.

Nasce dall'interazione misurata nel 2x2 del 19 agosto:

    frame\\operatori   attuali   area unitaria
    current             0.7072      0.7292
    rms                 0.7637      0.7377

Da soli i due rimedi guadagnano; insieme perdono. Il frame rms e' invariante a traslazione e
scala uniforme, quindi l'xyz di pot_rms_area e' bit-identico a quello di pot_rms e l'unica
differenza fra i due bracci sono gli operatori --- gli stessi operatori che, a frame invariato,
avevano fatto guadagnare 0.022. Lo stesso cambiamento cambia dunque segno a seconda del frame.

IPOTESI, registrata prima di costruire questo braccio: con il frame rms la rete riceve xyz in
unita' di raggio quadratico medio e un Laplaciano in unita' di area, cioe' due canali
normalizzati da grandezze diverse. I tempi di diffusione appresi agiscono su lambda, che non e'
piu' coerente con la scala delle coordinate.

PREVISIONE: operatori calcolati sulla stessa normalizzazione che la rete vede sull'xyz ---
centroide pesato per massa, raggio quadratico medio unitario --- devono fare almeno quanto
pot_rms. Se non lo fanno, l'ipotesi della coerenza di unita' e' sbagliata e va detto.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parents[1] / "diffusion-net/src"))
from diffusion_net.geometry import compute_operators          # noqa: E402
from potential_operators import load_mesh, save_npz           # noqa: E402



def vertex_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area baricentrica per vertice: un terzo dell'area di ogni triangolo incidente.

    Deve coincidere con il peso usato da v2_work/pointnet/frames.py, che sul dato in cache
    prende la `mass` salvata negli operatori. Qui gli operatori non esistono ancora --- sono
    esattamente cio' che stiamo per calcolare --- quindi la stessa quantita' va ricostruita
    dalle facce. Se le due definizioni divergessero, xyz e operatori tornerebbero a vivere in
    scale diverse, che e' il difetto che questo braccio esiste per rimuovere.
    """
    tri = V[F]
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    out = np.zeros(len(V), dtype=np.float64)
    np.add.at(out, F[:, 0], a / 3.0)
    np.add.at(out, F[:, 1], a / 3.0)
    np.add.at(out, F[:, 2], a / 3.0)
    return out


def total_area(V: np.ndarray, F: np.ndarray) -> float:
    tri = V[F]
    return float(0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--k-eig", type=int, default=128)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.npz"))
    si, sn = (int(x) for x in args.shard.split("/"))
    # shard the full sorted list FIRST, then skip what exists: filtering first makes each
    # worker's stripe depend on when it started, which silently leaves files unassigned.
    todo = [p for p in files[si::sn]
            if args.overwrite or not (args.output_dir / p.name).exists()]
    print(f"{len(files)} inputs, {len(todo)} da calcolare (shard {si}/{sn})", flush=True)

    t0 = time.time(); ok = fail = 0
    for i, p in enumerate(todo):
        try:
            V, F = load_mesh(p)
            w = vertex_areas(V, F)
            tot = float(w.sum())
            if not np.isfinite(tot) or tot <= 0:
                raise ValueError(f"aree per vertice non valide: {tot}")
            w = w / tot
            c = (w[:, None] * V).sum(0)
            r = float(np.sqrt((w * ((V - c) ** 2).sum(1)).sum()))
            if not np.isfinite(r) or r <= 0:
                raise ValueError(f"raggio rms non valido: {r}")
            V = (V - c) / r                            # stesso frame che la rete vede sull'xyz
            Vt = torch.tensor(V, dtype=torch.float32)
            Ft = torch.tensor(F, dtype=torch.int32)
            _, mass, L, evals, evecs, gX, gY = compute_operators(Vt, Ft, k_eig=args.k_eig)
            save_npz(args.output_dir / p.name, V, F, mass, L, evals, evecs, gX, gY)
            ok += 1
        except Exception as exc:                       # noqa: BLE001
            fail += 1
            print(f"  FAIL {p.name}: {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 25 == 0:
            r = (i + 1) / max(time.time() - t0, 1e-9)
            print(f"  {i+1}/{len(todo)} ok={ok} fail={fail} ({r:.2f}/s)", flush=True)
    print(f"done ok={ok} fail={fail} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
