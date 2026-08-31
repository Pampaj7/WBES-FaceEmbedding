#!/usr/bin/env python
"""Il frame del ground truth decide quale metrica vince?

`gate1_summary.csv` classifica otto metriche per quanto bene ordinano le identita' come le
ordina D_GT. Ma D_GT non e' una quantita' data: dipende dal frame in cui si misurano le mesh, e
quel frame non compare in nessun paper del campo -- lo scelgono i loader.

QUALE frame sia il nostro e' stato accertato qui e non assunto. Il costruttore della matrice
(`compute_gt_distance_matrix_normalized.py`) dichiara nel docstring di applicare "the SAME
normalisation applied in GTReadyDatasetNPZ: subtract mean, divide by global max |coord|". **Il
codice non lo fa**: `load_mesh` restituisce `verts` grezzi e l'unica normalizzazione e' un
singolo scalare globale sulla matrice delle distanze, che essendo una costante non cambia
nessun rango. Verificato per confronto diretto sulle 100 identita' del test:

    raw         rho(stored, ricalcolato) = 1.000000     <- il bersaglio e' questo
    global_rms                             0.816329
    area                                   0.795105
    rms                                    0.790902
    maxabs                                 0.785416

Il bersaglio e' dunque `raw`. Il loader del modello (`dataset_gtready.py:167-171`) invece centra
sulla media dei vertici e divide per maxabs. Ingresso e bersaglio vivono in frame diversi, e
`gt_frames.py` ha isolato per via algebrica che il vantaggio di `raw` sta nella CENTRATURA --
esattamente cio' che il loader rimuove. La rete deve predire una distanza che dipende dalla
posizione assoluta partendo da un ingresso da cui la posizione assoluta e' stata tolta.

Questo script chiede la domanda che conta per il campo, non per noi: **se D_GT fosse stato
costruito in un frame diverso, le otto metriche cambierebbero ordine?** Se lo cambiano, una
classifica di metriche pubblicata senza dichiarare la normalizzazione non e' riproducibile.

Le colonne delle metriche sono congelate in `extended_pair_metrics/<pair>/pair_metrics.csv` e non
dipendono dal frame: chamfer, arcface, clip, dinov2, varifold, currents, lpips sono calcolate
sulle mesh, non sul bersaglio. L'unica cosa che si ricalcola e' `gt_distance`.

Avvertenza registrata prima di guardare i numeri: `latent_distance` e' la NOSTRA metrica, ed e'
stata addestrata contro D_GT in `raw`. Un suo vantaggio sotto `raw` che sparisce sotto gli altri
frame non e' evidenza che il metodo funzioni: e' evidenza che valutiamo nel frame in cui abbiamo
allenato. Le altre sette metriche non hanno mai visto D_GT e non hanno questo vantaggio.

Controllo di validita': il frame `raw` ricalcolato qui deve riprodurre la colonna `gt_distance`
gia' salvata. Se non lo fa, la ricostruzione e' sbagliata e il resto non vale.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
sys.path.insert(0, str(THIS))
from gt_frames import FRAMES, reframe, vertex_areas, pairwise  # noqa: E402

PAIR_ROOT = REPO / "v2_work" / "phase0" / "extended_pair_metrics"
MESH_DIR = REPO / "datasets" / "REMESH" / "npz_data_topo_500_withops"

# Il frame in cui il ground truth salvato e' effettivamente costruito (accertato, non assunto).
REF = "raw"

METRICS = ["latent_distance", "raw_chamfer", "arcface_dist", "clip_dist",
           "dinov2_dist", "varifold_dist", "currents_dist", "lpips_dist"]


def _f(x: str) -> float:
    """Celle vuote -> nan, cosi' una metrica parziale non fa cadere l'intera tabella."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_pair_tables() -> dict[str, list[dict]]:
    tables = {}
    for d in sorted(PAIR_ROOT.iterdir()):
        f = d / "pair_metrics.csv"
        if f.exists():
            with open(f, newline="") as fh:
                tables[d.name] = list(csv.DictReader(fh))
    if not tables:
        raise SystemExit(f"nessuna tabella sotto {PAIR_ROOT}")
    return tables


def subjects_of(tables) -> list[str]:
    s = set()
    for rows in tables.values():
        for r in rows:
            s.add(r["subject_a"]); s.add(r["subject_b"])
    return sorted(s)


def load_meshes(sids: list[str]):
    verts, faces = [], None
    for sid in sids:
        p = MESH_DIR / f"{sid}_GTready_original.npz"
        with np.load(p) as z:
            V = (z["verts"] if "verts" in z else z["V"]).astype(np.float64)
            F = (z["faces"] if "faces" in z else z["F"]).astype(np.int64)
        verts.append(V)
        faces = F if faces is None else faces
    return verts, faces


def global_scale(verts, F) -> float:
    """Mediana dei raggi rms pesati per area: una costante per l'insieme, come in gt_frames."""
    rs = []
    for V in verts:
        a = vertex_areas(V, F); w = a / a.sum()
        X = V - (w[:, None] * V).sum(0, keepdims=True)
        rs.append(np.sqrt((w * (X * X).sum(1)).sum()))
    return float(np.median(rs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=THIS / "gt_matrices")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tables = load_pair_tables()
    sids = subjects_of(tables)
    idx = {s: i for i, s in enumerate(sids)}
    verts, F = load_meshes(sids)
    print(f"{len(sids)} identita', {verts[0].shape[0]} vertici, "
          f"{len(tables)} coppie di topologie [{device}]")

    gs = global_scale(verts, F)
    D = {f: pairwise([reframe(V, F, f, gs) for V in verts], device) for f in FRAMES}

    # --- controllo di validita': maxabs deve riprodurre la colonna salvata ---
    rows_all = [r for rows in tables.values() for r in rows]
    stored = np.array([float(r["gt_distance"]) for r in rows_all])
    recomp = np.array([D[REF][idx[r["subject_a"]], idx[r["subject_b"]]] for r in rows_all])
    rho_check = float(spearmanr(stored, recomp).statistic)
    print(f"validita': rho({REF} ricalcolato, gt_distance salvato) = {rho_check:.6f}")
    assert rho_check > 0.999, (
        f"la ricostruzione di {REF} non riproduce il ground truth salvato (rho={rho_check:.4f}): "
        "il resto della tabella non e' interpretabile")

    # --- Spearman per metrica, per frame, complessivo e per coppia di topologie ---
    out_rows = []
    for frame in FRAMES:
        Dm = D[frame]
        for label, rows in [("OVERALL", rows_all)] + sorted(tables.items()):
            gt = np.array([Dm[idx[r["subject_a"]], idx[r["subject_b"]]] for r in rows])
            rec = {"frame": frame, "pair_label": label, "n_pairs": len(rows)}
            for m in METRICS:
                # lpips e' stata calcolata solo su un sottoinsieme di soggetti: celle vuote.
                v = np.array([_f(r.get(m, "")) for r in rows])
                ok = np.isfinite(v) & np.isfinite(gt)
                rec[m] = float(spearmanr(gt[ok], v[ok]).statistic) if ok.sum() > 2 else float("nan")
                rec[m + "__n"] = int(ok.sum())
            out_rows.append(rec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "frame_metric_ranking.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "pair_label", "n_pairs"] + METRICS
                           + [m + "__n" for m in METRICS])
        w.writeheader(); w.writerows(out_rows)

    # --- la domanda: l'ordine delle metriche cambia col frame? ---
    overall = {r["frame"]: r for r in out_rows if r["pair_label"] == "OVERALL"}
    orders = {}
    print(f"\n{'frame':11s} " + " ".join(f"{m.replace('_dist','').replace('_distance',''):>9s}" for m in METRICS))
    for frame in FRAMES:
        r = overall[frame]
        print(f"{frame:11s} " + " ".join(f"{r[m]:9.4f}" for m in METRICS))
        orders[frame] = sorted(METRICS, key=lambda m: -r[m])

    print("\nclassifica per frame (dalla migliore):")
    for frame in FRAMES:
        print(f"  {frame:11s} " + " > ".join(m.replace("_dist", "").replace("_distance", "")
                                             for m in orders[frame]))
    ref = orders[REF]
    flips = {f: sum(a != b for a, b in zip(ref, orders[f])) for f in FRAMES if f != REF}
    print(f"\nposizioni che cambiano rispetto a {REF} (il frame effettivo del ground truth): {flips}")

    (args.out_dir / "frame_metric_ranking.json").write_text(json.dumps(
        {"n_identities": len(sids), "overall": {f: {m: overall[f][m] for m in METRICS} for f in FRAMES},
         "orders": orders, "reference_frame": REF, f"rank_changes_vs_{REF}": flips,
         "validity_rho_maxabs_vs_stored": rho_check}, indent=2))
    print(f"\nscritto in {csv_path} e .json")


if __name__ == "__main__":
    main()
