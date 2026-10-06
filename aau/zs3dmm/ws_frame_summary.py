#!/usr/bin/env python3
"""Summary unico dei test di frame e normalizzazione (WS-frame) sui domini zero-shot e su ICT.

    aau/run.sh aau/zs3dmm/ws_frame_summary.py        # -> aau/runs/ws_frame/summary.md

Non ricalcola niente: mette insieme i csv che ``zs_summarize.py`` ha scritto per ogni dominio e
variante (``table_cells<v>.csv``, ``paired<v>.csv``, ``chamfer_vs_base<v>.csv``), con i CI
bootstrap per soggetto e le differenze appaiate calcolate li'. Una variante senza csv e' segnata
"mancante" invece di far fallire tutto.

Convenzioni di frame, misurate sulle mesh dei dati (aau/scratch/hifi3d/frames.py, winding2.py):

    dominio      alto   naso   normali
    BFM (train)  -y     -z     verso l'interno (7% verso l'esterno)
    ICT (train)  +y     +z     verso l'esterno (83%)
    HIFI3D       +y     +z     verso l'esterno (83%)  = convenzione ICT
    FaceVerse    -y     -z     verso l'esterno (94%)  = rotazione BFM, verso ICT

Varianti (suffissi delle out dir di zs_zeroshot.sbatch):
    ""                     dati come sono
    _frame-xmymz           Rx(180) = diag(1,-1,-1): rotazione propria, verso INVARIATO
    _frame-xmymz_flip      Rx(180) + facce invertite: convenzione BFM completa (HIFI3D, ICT)
    _flip                  facce invertite (FaceVerse: + rotazione nativa = convenzione BFM completa)
    _frame-mxymz           Ry(180) = diag(-1,1,-1) (la rotazione proposta dal critic)
    _evalframe-rms         V_in ri-inquadrato nel frame rms nel forward (ablation_hooks.py)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
RUNS = THIS_DIR.parent / "runs"
ARMS = ("joint", "bfm_only", "ict_only")
LABEL = {"joint": "BFM+ICT", "bfm_only": "BFM-only", "ict_only": "ICT-only"}
PROTOS = (("mesh_pair_nocrop_cross", "nocrop_cross", "senza crop"),
          ("mesh_pair_all_cross", "all_cross", "tutte le topologie"),
          ("subject_pair_mean", "subject_pair_mean", "media per coppia di soggetti"))

# dominio -> (dir, [(variante, descrizione, convenzione d'ingresso)])
DOMAINS = {
    "HIFI3D": ("ws_hifi3d", [
        ("", "nativo", "ICT completa"),
        ("_frame-xmymz", "Rx(180)", "rotazione BFM, verso ICT"),
        ("_frame-xmymz_flip", "Rx(180) + facce invertite", "BFM completa"),
        ("_frame-mxymz", "Ry(180) (critic)", "naso -z ma alto +y: nessuna delle due"),
        ("_evalframe-rms", "nativo, eval_frame rms", "ICT completa, ri-inquadrato rms"),
    ]),
    "FaceVerse": ("ws_faceverse", [
        ("", "nativo", "rotazione BFM, verso ICT"),
        ("_frame-xmymz", "Rx(180)", "ICT completa"),
        ("_flip", "facce invertite", "BFM completa"),
    ]),
    "ICT held-out (controllo e2e)": ("ws_ictzs", [
        ("", "nativo (e2e)", "ICT completa"),
        ("_frame-xmymz", "Rx(180)", "rotazione BFM, verso ICT"),
        ("_frame-xmymz_flip", "Rx(180) + facce invertite", "BFM completa"),
    ]),
}


def read(d: Path, name: str, v: str) -> pd.DataFrame | None:
    p = d / f"{name}{v}.csv"
    return pd.read_csv(p) if p.exists() else None


def fmt(r, pre="latent") -> str:
    lo, hi = r[f"{pre}_ci_low"], r[f"{pre}_ci_high"]
    return f"{r[f'{pre}_spearman']:.3f}" + ("" if not np.isfinite(lo) else f" [{lo:.2f}, {hi:.2f}]")


def fmt_d(r) -> str:
    return f"{r['diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"


def cell(t: pd.DataFrame, arm: str, proto: str, pre="latent") -> str:
    s = t[(t["model"] == arm) & (t["gt"] == "maxabs") & (t["protocol"] == proto) & (t["scenario"] == "clean")]
    return "-" if s.empty else fmt(s.iloc[0], pre)


def pcell(p: pd.DataFrame, comp: str, scen: str) -> str:
    s = p[(p["comparison"] == comp) & (p["scenario"] == scen) & (p["gt"] == "maxabs")]
    return "-" if s.empty else fmt_d(s.iloc[0])


def domain_section(name: str, sub: str, variants) -> list[str]:
    d = RUNS / sub
    out = [f"\n## {name}\n"]
    have = [(v, desc, conv) for v, desc, conv in variants if read(d, "table_cells", v) is not None]
    miss = [f"`{v}` ({desc})" for v, desc, conv in variants if read(d, "table_cells", v) is None]
    if miss:
        out.append(f"Varianti mancanti (non ancora calcolate): {', '.join(miss)}.\n")
    for proto, scen, plabel in PROTOS:
        out.append(f"\n### Spearman latent con la GT maxabs, {plabel} (CI 95% per soggetto)\n")
        head = "| variante | ingresso | " + " | ".join(LABEL[a] for a in ARMS) + " | Chamfer eval |"
        out += [head, "|" + " --- |" * (len(ARMS) + 3)]
        for v, desc, conv in have:
            t = read(d, "table_cells", v)
            out.append(f"| {desc} | {conv} | " + " | ".join(cell(t, a, proto) for a in ARMS)
                       + f" | {cell(t, 'joint', proto, 'chamfer')} |")
        out.append(f"\n#### Differenze appaiate, {plabel}: variante - nativo (stesso braccio) e modello - Chamfer\n")
        out += ["| variante | " + " | ".join(f"{LABEL[a]}: var - nativo" for a in ARMS)
                + " | " + " | ".join(f"{LABEL[a]} - Chamfer" for a in ARMS) + " |",
                "|" + " --- |" * (2 * len(ARMS) + 1)]
        for v, desc, conv in have:
            p = read(d, "paired", v)
            if p is None:
                continue
            eff = [pcell(p, f"{LABEL[a]}: variante - base", scen) if v else "0" for a in ARMS]
            ch = [pcell(p, f"{LABEL[a]} - Chamfer eval", scen) for a in ARMS]
            out.append(f"| {desc} | " + " | ".join(eff) + " | " + " | ".join(ch) + " |")
        out.append(f"\n#### Differenze appaiate fra bracci, {plabel}\n")
        out += ["| variante | BFM+ICT - ICT-only | BFM+ICT - BFM-only |", "| --- | --- | --- |"]
        for v, desc, conv in have:
            p = read(d, "paired", v)
            if p is not None:
                out.append(f"| {desc} | {pcell(p, 'BFM+ICT - ICT-only', scen)} | {pcell(p, 'BFM+ICT - BFM-only', scen)} |")
    out.append("\n### Controllo di sanita': Chamfer eval della variante contro il nativo, riga per riga\n")
    out += ["| variante | max diff relativa | righe con diff relativa > 1e-4 (su 148.500) | GPU variante / nativo |",
            "| --- | --- | --- | --- |"]
    for v, desc, conv in have:
        c = read(d, "chamfer_vs_base", v)
        if c is not None:
            r = c.iloc[0]
            out.append(f"| {desc} | {r['max_rel_diff']:.2e} | {int(r['n_rows_rel_gt_1e-4'])} | vedi nota |")
    return out


def crop_rows() -> list[str]:
    """Riga crop -> altre topologie, latent, HIFI3D nativo contro rms (dal csv per coppia di topologie)."""
    d = RUNS / "ws_hifi3d"
    out = ["\n## Crop e normalizzazione (HIFI3D): riga crop -> B, Spearman latent, GT maxabs\n",
           "| braccio | variante | crop->down8k | crop->noisy | crop->original | crop->remesh | crop->up60k |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for v, desc in (("", "maxabs (nativo)"), ("_evalframe-rms", "rms")):
        t = read(d, "topology_pairs", v)
        if t is None:
            continue
        for a in ARMS:
            s = t[(t["model"] == a) & (t["metric"] == "latent") & (t["gt"] == "maxabs")].set_index("topology_pair")
            vals = [f"{s.loc[f'crop__to__{b}', 'spearman']:.3f}" for b in ("down8k", "noisy", "original", "remesh", "up60k")]
            out.append(f"| {LABEL[a]} | {desc} | " + " | ".join(vals) + " |")
    t = read(d, "topology_pairs", "")
    if t is not None:
        s = t[(t["model"] == "joint") & (t["metric"] == "chamfer") & (t["gt"] == "maxabs")].set_index("topology_pair")
        vals = [f"{s.loc[f'crop__to__{b}', 'spearman']:.3f}" for b in ("down8k", "noisy", "original", "remesh", "up60k")]
        out.append("| Chamfer eval | - | " + " | ".join(vals) + " |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=RUNS / "ws_frame" / "summary.md")
    a = ap.parse_args()
    parts = ["# WS-frame: frame, verso delle facce e normalizzazione nei test zero-shot\n",
             __doc__.split("Convenzioni di frame", 1)[1].join(["Convenzioni di frame", ""]).strip() + "\n",
             "Tutti i numeri: 100 soggetti per dominio, gli stessi in ogni variante e braccio; GT maxabs "
             "(protocollo ICT); CI 95% bootstrap per soggetto, 1000 repliche; differenze appaiate sulle "
             "stesse repliche (zs_summarize.py). Tabelle complete per variante in "
             "`aau/runs/<ws_hifi3d|ws_faceverse|ws_ictzs>/summary<variante>.md`.\n",
             "Nota sul controllo Chamfer: dove nativo e variante sono girati sullo stesso tipo di GPU "
             "(ICT, entrambi A10) la Chamfer e' identica bit per bit; su HIFI3D il nativo e' su L40S e le "
             "varianti su A10, e 165 righe su 148.500 (tutte con `down8k`) differiscono fino al 3% "
             "relativo, con Spearman di Chamfer identico a 2e-6.\n"]
    for name, (sub, variants) in DOMAINS.items():
        parts += domain_section(name, sub, variants)
    parts += crop_rows()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"scritto {a.out}")


if __name__ == "__main__":
    main()
