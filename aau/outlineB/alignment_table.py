#!/usr/bin/env python3
"""tab:alignment_effect su held-out BFM e held-out ICT, con il gate di riproduzione su fb100.

Le matrici sono quelle di ``alignment_matrix.py`` (pipeline faceBench importata); i numeri
con CI escono da ``rank_from_matrix.compute_rows``, cioe' dal bootstrap per soggetto di
``scripts/compute_bootstrap_ci.py``, come per tutte le altre tabelle di aau/baselines.

Tre blocchi, ognuno con il suo D_GT:

    bfm_fb100     facebench_first100, D_GT BFM del repo   -> SOLO gate: 79 soggetti su 100
                  sono di training del v1, ma per le pipeline geometriche (che non
                  addestrano) il gate e' lecito
    bfm_heldout   i 100 held-out BFM, stessa D_GT
    ict_heldout   i 100 held-out ICT (seed 1234), D_GT ICT di train_ready (frame maxabs,
                  quella di tutte le tabelle ICT); in piu' la stessa colonna contro la D_GT
                  ICT in coordinate grezze, perche' su ICT le due matrici concordano solo a
                  0.351 (datasets/ICT/gt/manifest.json) e la BFM del repo e' grezza

Ogni blocco esiste in due campionamenti: 4096 punti (default di run_facebench_remesh, quello
della colonna original->original del paper e di tutta la Tabella 2 estesa) e 2048 punti
(suffisso ``_s2048``), quello con cui e' stata prodotta la colonna cross no-crop del paper:
l'artefatto ``facebench_remesh_100subj_norm`` si riproduce esattamente solo a 2048 (job
1054872).  Il gate confronta quindi original->original su ``bfm_fb100`` e la colonna cross
su ``bfm_fb100_s2048``; gli altri due incroci sono stampati come informazione.

Il cambio di dataset fra un blocco e l'altro e' fatto riassegnando ``common.MESH_ROOT`` e
``common.GT_MATRIX``, che le funzioni di common leggono al momento della chiamata.

  aau/run.sh aau/outlineB/alignment_table.py --runs aau/runs/outlineB
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "baselines"))

import common  # noqa: E402
import rank_from_matrix as rank  # noqa: E402

# m3dfb_rlr_chamfer (m3dfb_matrix.py) e' opzionale: un blocco senza le sue matrici lo salta.
METRICS = ("chamfer", "rigid_icp_chamfer", "nicp_p2p", "nicp_p2tri", "m3dfb_rlr_chamfer")
OPTIONAL = ("m3dfb_rlr_chamfer",)
LABEL = {"chamfer": "Chamfer", "rigid_icp_chamfer": "Rigid ICP + Chamfer",
         "nicp_p2p": "Rigid ICP + NICP + P2P", "nicp_p2tri": "Rigid ICP + NICP + P2Tri",
         "m3dfb_rlr_chamfer": "M3DFB RLR + Chamfer"}
SETTINGS = ("original_to_original", "nocrop_cross_topology",
            "tessellation_cross_topology", "perturbation_cross_topology")
TABLE_SETTINGS = ("original_to_original", "nocrop_cross_topology")

BFM_MESH = common.REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500"
BFM_GT = common.GT_MATRIX
ICT_MESH = common.REPO_ROOT / "datasets" / "ICT" / "eval_view_heldout"
ICT_GT = common.REPO_ROOT / "datasets" / "ICT" / "train_ready" / "gt_matrix.npz"
ICT_GT_RAW_SRC = common.REPO_ROOT / "datasets" / "ICT" / "gt" / "ict_matrix_distances_raw.npz"
ICT_GT_MAXABS_SRC = common.REPO_ROOT / "datasets" / "ICT" / "gt" / "ict_matrix_distances_maxabs.npz"
ICT_ID_OFFSET = 10000   # datasets/ICT/train_ready/manifest.json

# tab:alignment_effect pubblicata, letta dal csv che l'ha prodotta (stime a piena precisione,
# cosi' il delta non e' l'arrotondamento a tre cifre del .tex), fb100.
PAPER_CSV = common.REPO_ROOT / "paper_artifacts" / "bootstrap_ci" / "bootstrap_ci.csv"
PAPER_METHOD = {"Chamfer": "chamfer", "ICP + Chamfer": "rigid_icp_chamfer",
                "ICP + NICP + P2P": "nicp_p2p", "ICP + NICP + P2Tri": "nicp_p2tri"}
PAPER_SETTING = {"Original-to-original": "original_to_original",
                 "No-crop cross-topology": "nocrop_cross_topology"}


def paper_reference() -> dict[tuple[str, str], tuple[float, float, float]]:
    ref = {}
    with open(PAPER_CSV, newline="") as fh:
        for row in csv.reader(fh):
            if (len(row) >= 6 and row[0] == "REMESH" and row[1] in PAPER_SETTING
                    and row[2] in PAPER_METHOD):
                ref[(PAPER_METHOD[row[2]], PAPER_SETTING[row[1]])] = tuple(float(x) for x in row[3:6])
    if len(ref) != 8:
        raise SystemExit(f"{PAPER_CSV}: attese 8 celle di tab:alignment_effect, trovate {len(ref)}")
    return ref


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=Path, default=common.AAU_DIR / "runs" / "outlineB")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def ict_raw_gt(out_dir: Path) -> Path:
    """La D_GT ICT grezza con i nomi id1NNNN di train_ready (la sorgente ha ict0NNNN).

    L'ordine delle righe e' lo stesso della maxabs, di cui train_ready e' una copia
    rinominata: lo si verifica invece di assumerlo.
    """
    path = out_dir / "gt_ict_raw_idnames.npz"
    if path.exists():
        return path
    with np.load(ICT_GT_RAW_SRC, allow_pickle=True) as raw, \
            np.load(ICT_GT_MAXABS_SRC, allow_pickle=True) as maxabs, \
            np.load(ICT_GT, allow_pickle=True) as ready:
        if not np.array_equal(maxabs["D_orig"], ready["D_orig"]):
            raise SystemExit("train_ready/gt_matrix.npz non e' la maxabs di datasets/ICT/gt")
        if [str(n) for n in raw["names"]] != [str(n) for n in maxabs["names"]]:
            raise SystemExit("raw e maxabs ICT hanno nomi in ordine diverso")
        names = np.asarray([f"id{ICT_ID_OFFSET + int(str(n)[3:])}" for n in raw["names"]])
        if [str(n) for n in names] != [str(n) for n in ready["names"]]:
            raise SystemExit("la rinomina ict -> id+10000 non riproduce i nomi di train_ready")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(path, D_orig=raw["D_orig"], names=names)
    return path


# (blocco, setting) che il paper ha pubblicato con quel campionamento: sono il gate.
GATE = {("bfm_fb100", "original_to_original"), ("bfm_fb100_s2048", "nocrop_cross_topology")}


def blocks(runs: Path) -> list[dict]:
    out = []
    for suffix in ("", "_s2048"):
        out += [
            {"block": f"bfm_fb100{suffix}", "subject_set": "facebench_first100",
             "dir": runs / f"bfm_fb100{suffix}", "mesh": BFM_MESH, "gt": BFM_GT,
             "gt_label": "BFM raw (repo)"},
            {"block": f"bfm_heldout{suffix}", "subject_set": "heldout",
             "dir": runs / f"bfm_heldout{suffix}", "mesh": BFM_MESH, "gt": BFM_GT,
             "gt_label": "BFM raw (repo)"},
            {"block": f"ict_heldout{suffix}", "subject_set": "ict_heldout",
             "dir": runs / f"ict_heldout{suffix}", "mesh": ICT_MESH, "gt": ICT_GT,
             "gt_label": "ICT maxabs (train_ready)"},
            {"block": f"ict_heldout{suffix}_gtraw", "subject_set": "ict_heldout",
             "dir": runs / f"ict_heldout{suffix}", "mesh": ICT_MESH, "gt": None,
             "gt_label": "ICT raw"},
        ]
    return out


def main() -> None:
    args = parse_args()
    bootstrap_module = rank.load_bootstrap_module()
    rows, gate = [], []
    paper = paper_reference()
    for block in blocks(args.runs):
        common.MESH_ROOT = block["mesh"]
        common.GT_MATRIX = block["gt"] if block["gt"] is not None else ict_raw_gt(args.runs / "ict_heldout")
        ns = argparse.Namespace(out_root=block["dir"], subject_set=block["subject_set"],
                                n_bootstrap=args.n_bootstrap, seed=args.seed, strict=True)
        for metric in METRICS:
            if metric in OPTIONAL and not common.matrix_path(
                    metric, "original", "original", block["dir"]).exists():
                print(f"[align-tab] {block['block']}: {metric} assente, salto", flush=True)
                continue
            for setting in SETTINGS:
                for row in rank.compute_rows(metric, setting, ns, bootstrap_module):
                    row.update(block=block["block"], gt=block["gt_label"],
                               gt_path=str(common.GT_MATRIX))
                    rows.append(row)
                    if row["correlation"] != "spearman":
                        continue
                    print(f"[align-tab] {block['block']:<18} {metric:<18} {setting:<28} "
                          f"{row['value']:.4f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
                          f"n_pairs={row['n_pairs']}", flush=True)
                    ref = paper.get((metric, setting))
                    if block["block"].startswith("bfm_fb100") and ref is not None:
                        delta = row["value"] - ref[0]
                        verdict = (("OK" if abs(delta) < 1e-3 else "DIVERSO")
                                   if (block["block"], setting) in GATE else "info")
                        gate.append(f"  [{verdict}] {block['block']} "
                                    f"{metric}/{setting}: qui {row['value']:.4f} "
                                    f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] vs paper "
                                    f"{ref[0]:.4f} [{ref[1]:.3f}, {ref[2]:.3f}] (delta {delta:+.5f})")

    results = pd.DataFrame(rows)
    out = args.runs
    results.to_csv(out / "alignment_effect.csv", index=False)
    (out / "alignment_effect.json").write_text(json.dumps({"rows": rows, "gate": gate}, indent=2))

    def cell(block: str, metric: str, setting: str) -> str:
        r = results[results["block"].eq(block) & results["metric"].eq(metric)
                    & results["setting"].eq(setting) & results["correlation"].eq("spearman")]
        return rank.fmt_interval(r.iloc[0]) if len(r) else "--"

    shown = ("bfm_heldout", "bfm_heldout_s2048", "ict_heldout", "ict_heldout_s2048",
             "ict_heldout_gtraw", "ict_heldout_s2048_gtraw", "bfm_fb100", "bfm_fb100_s2048")
    head = {"bfm_heldout": "BFM held-out", "ict_heldout": "ICT held-out (GT maxabs)",
            "ict_heldout_gtraw": "ICT held-out (GT raw)", "bfm_fb100": "fb100 (gate)"}
    head.update({f"{b.replace('_gtraw', '')}_s2048{'_gtraw' if b.endswith('_gtraw') else ''}":
                 f"{h}, 2048 pt" for b, h in list(head.items())})
    md = ["# tab:alignment_effect su held-out (OUTLINE_B §4, esperimento 1)", "",
          "Spearman vs D_GT, CI 95% bootstrap per soggetto (1000 repliche). Colonne: "
          "same = original->original (4950 coppie), cross = no-crop cross-topology (99000). "
          "Senza suffisso: 4096 punti campionati per mesh; '2048 pt': il campionamento della "
          "colonna cross del paper.", ""]
    for setting in TABLE_SETTINGS + ("tessellation_cross_topology", "perturbation_cross_topology"):
        md += [f"## {setting}", "", "| metodo | " + " | ".join(head[b] for b in shown) + " |",
               "|---|" + "---|" * len(shown)]
        md += [f"| {LABEL[m]} | " + " | ".join(cell(b, m, setting) for b in shown) + " |"
               for m in METRICS]
        md.append("")
    md += ["## Gate su facebench_first100 contro tab:alignment_effect pubblicata", "", "```", *gate, "```"]
    (out / "alignment_effect.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [r"% tab:alignment_effect su held-out. Spearman vs D_GT, CI 95% bootstrap per soggetto.",
           r"% 4096 punti per mesh in tutte le celle; le varianti a 2048 (protocollo della colonna",
           r"% cross del paper) sono in alignment_effect.md.",
           r"% BFM: D_GT del repo (grezza). ICT: D_GT di train_ready (maxabs); la riga con GT",
           r"% grezza ICT e' in alignment_effect.md. Generato da aau/outlineB/alignment_table.py.",
           r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{BFM held-out} & \multicolumn{2}{c}{ICT held-out} \\",
           r"Method & Same topology & No-crop cross & Same topology & No-crop cross \\", r"\midrule"]
    tex += [f"{LABEL[m]} & " + " & ".join(cell(b, m, s) for b in ("bfm_heldout", "ict_heldout")
                                          for s in TABLE_SETTINGS) + r" \\" for m in METRICS]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "alignment_effect.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    print("\n[align-tab] gate su fb100 (info = campionamento diverso da quello pubblicato):")
    print("\n".join(gate))
    print(f"\n[align-tab] {out / 'alignment_effect.md'}")


if __name__ == "__main__":
    main()
