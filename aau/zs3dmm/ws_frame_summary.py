#!/usr/bin/env python3
"""Summary unico dei test di frame e normalizzazione (WS-frame), con contrasti fra varianti appaiati.

    aau/run.sh aau/zs3dmm/ws_frame_summary.py        # -> aau/runs/ws_frame/summary.md
    (ws_frame_summary.sbatch: e' un job, il bootstrap vuole i core)

Protocolli, in quest'ordine (DICHIARATO DOPO AVER VISTO QUESTI NUMERI, 6 ottobre 2026: vale come
regola per i test futuri, non come scelta a priori di questi):
  1. PRIMARIO   coppie di mesh cross-topologia SENZA crop: 20 coppie ordinate x 4950 coppie di
                soggetti, una osservazione per (coppia di soggetti, coppia di topologie);
  2. SECONDARIO media per coppia di soggetti (lo scenario clean dello script di ranking), sulle 30
                coppie ordinate cross, crop compreso;
  3. A PARTE    le 10 coppie ordinate con crop da un lato.

Tutti gli Spearman, i CI e i contrasti vengono dalle STESSE repliche bootstrap per soggetto (la
ricampionatura di ``weighted_bootstrap_spearman``: soggetti con reinserimento, peso di una coppia
= prodotto dei conteggi dei suoi due soggetti), applicate a una tabella in cui ogni variante e
ogni braccio e' una colonna allineata sulle stesse righe (stessi soggetti, stesse topologie,
stessa GT; ``merge`` one-to-one, altrimenti esce). Cosi' anche i contrasti FRA VARIANTI -- p.es.
congiunto nativo contro BFM-only nel frame dei suoi dati -- hanno un CI appaiato.

Convenzioni di frame, misurate sulle mesh dei dati (aau/scratch/hifi3d/frames.py, winding2.py):

    dominio      alto   naso   normali
    BFM (train)  -y     -z     verso l'interno (7% verso l'esterno)
    ICT (train)  +y     +z     verso l'esterno (83%)
    HIFI3D       +y     +z     verso l'esterno (83%)  = convenzione ICT
    FaceVerse    -y     -z     verso l'esterno (94%)  = rotazione BFM, verso ICT

"Frame dei dati di training" di un braccio: BFM-only -> convenzione BFM completa (rotazione +
facce invertite); ICT-only -> convenzione ICT completa; congiunto -> ambiguo (ha visto entrambe),
riportato nel nativo.

Leakage ICT: il dominio ``ict`` e' il pool id14500-id14999 di ICT-5000, che NON e' held-out per
congiunto e ICT-only (il loro split e' casuale sull'intera data dir: aau/cross3dmm/ws2_views.py).
Le celle di quei due bracci su ICT, varianti e contrasti compresi, sono marcate "inquinate"; il
numero di soggetti in training e' contato qui da ``aau/runs/ws2_cross3dmm/splits.json``.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import zs_summarize as zs  # noqa: E402  (arm_sources, pair_metrics, chiavi di riga)

RUNS = THIS_DIR.parent / "runs"
ARMS = ("joint", "bfm_only", "ict_only")
LABEL = {"joint": "BFM+ICT", "bfm_only": "BFM-only", "ict_only": "ICT-only"}
PROTOCOLS = (
    ("nocrop", "PRIMARIO: coppie di mesh cross-topologia senza crop (20 coppie ordinate)"),
    ("spm", "SECONDARIO: media per coppia di soggetti (30 coppie ordinate, crop compreso)"),
    ("crop", "A PARTE: coppie con crop da un lato (10 coppie ordinate)"),
)
DOMAINS = {
    "hifi": {
        "label": "HIFI3D", "dir": "ws_hifi3d", "polluted": (),
        "variants": [("", "nativo", "ICT completa"),
                     ("_frame-xmymz", "Rx(180)", "rotazione BFM, verso ICT"),
                     ("_frame-xmymz_flip", "Rx(180) + facce invertite", "BFM completa"),
                     ("_frame-mxymz", "Ry(180) (critic)", "naso -z, alto +y: nessuna delle due"),
                     ("_evalframe-rms", "nativo, eval_frame rms", "ICT completa, ri-inquadrato rms")],
        "own": {"joint": "", "bfm_only": "_frame-xmymz_flip", "ict_only": ""},
    },
    "fv": {
        "label": "FaceVerse v2", "dir": "ws_faceverse", "polluted": (),
        "variants": [("", "nativo", "rotazione BFM, verso ICT"),
                     ("_frame-xmymz", "Rx(180)", "ICT completa"),
                     ("_flip", "facce invertite", "BFM completa")],
        "own": {"joint": "", "bfm_only": "_flip", "ict_only": "_frame-xmymz"},
    },
    "ict": {
        "label": "ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only)",
        "dir": "ws_ictzs", "polluted": ("joint", "ict_only"),
        "variants": [("", "nativo (e2e)", "ICT completa"),
                     ("_frame-xmymz", "Rx(180)", "rotazione BFM, verso ICT"),
                     ("_frame-xmymz_flip", "Rx(180) + facce invertite", "BFM completa")],
        "own": {"joint": "", "bfm_only": "_frame-xmymz_flip", "ict_only": ""},
    },
}
POLLUTED = "inquinata (soggetti in training)"


# ------------------------------------------------------------------------------ dati

def data_dir(dom: dict) -> Path:
    found = sorted((RUNS / dom["dir"]).glob("data_*"))
    if len(found) != 1:
        raise SystemExit(f"{dom['dir']}: attesa una sola data_<fp>, trovate {found}")
    return found[0]


def load_table(dom: dict) -> tuple[pd.DataFrame, list[str], dict]:
    """Una colonna latent per (braccio, variante) disponibile, piu' la Chamfer eval del nativo."""
    root = data_dir(dom)
    table, cols, cham_check = None, [], {}
    for v, _, _ in dom["variants"]:
        for arm in ARMS:
            try:
                src = zs.arm_sources(root, arm + v)
            except SystemExit:
                continue
            pm = zs.base.read_pair_metrics(src["topology"])
            col = f"{arm}{v}"
            keep = pm[zs.PAIR_KEYS + ["gt_distance", "latent_distance", "raw_chamfer"]].rename(
                columns={"latent_distance": col, "raw_chamfer": f"cham{col}", "gt_distance": f"gt{col}"})
            if table is None:
                table = keep
            else:
                n = len(table)
                table = table.merge(keep, on=zs.PAIR_KEYS, validate="one_to_one")
                if len(table) != n:
                    raise SystemExit(f"{dom['label']} {col}: righe non allineate ({n} -> {len(table)})")
            cols.append(col)
    if table is None:
        raise SystemExit(f"{dom['label']}: nessun braccio")
    ref = cols[0]
    for c in cols:
        if not np.allclose(table[f"gt{c}"], table[f"gt{ref}"]):
            raise SystemExit(f"{dom['label']} {c}: GT diversa sulle stesse righe")
        d = (table[f"cham{c}"] - table[f"cham{ref}"]).abs() / table[f"cham{ref}"].abs().clip(lower=1e-12)
        cham_check[c] = (float(d.max()), int((d > 1e-4).sum()))
    table = table.rename(columns={f"gt{ref}": "gt_distance", f"cham{ref}": "chamfer"})
    table = table[zs.PAIR_KEYS + ["gt_distance", "chamfer"] + cols]
    return table, cols, cham_check


def protocol_frame(t: pd.DataFrame, proto: str, cols: list[str]) -> pd.DataFrame:
    crop = t["topology_a"].eq("crop") | t["topology_b"].eq("crop")
    if proto == "nocrop":
        return t[~crop]
    if proto == "crop":
        return t[crop]
    return t.groupby(["subject_a", "subject_b"], as_index=False)[["gt_distance", "chamfer"] + cols].mean()


# ------------------------------------------------------------------------- bootstrap

def spearman_many(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman di x con ogni colonna di Y (ranghi medi sui pari merito, come scipy.spearmanr)."""
    rx = rankdata(x)
    rx = (rx - rx.mean()) / rx.std()
    out = np.empty(Y.shape[1])
    for k in range(Y.shape[1]):
        ry = rankdata(Y[:, k])
        ry = (ry - ry.mean()) / ry.std()
        out[k] = float(np.mean(rx * ry))
    return out


def multi_bootstrap(df: pd.DataFrame, cols: list[str], n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Punto e repliche (n_boot, k) degli Spearman con la GT di tutte le colonne, stesse repliche.

    Ricampionatura di ``weighted_bootstrap_spearman`` (scripts/compute_bootstrap_ci.py).
    """
    w = df[df["subject_a"].astype(str) != df["subject_b"].astype(str)]
    subjects = np.array(sorted(set(w["subject_a"].astype(str)) | set(w["subject_b"].astype(str))))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa = w["subject_a"].astype(str).map(s2i).to_numpy(np.int64)
    sb = w["subject_b"].astype(str).map(s2i).to_numpy(np.int64)
    gt = w["gt_distance"].to_numpy(np.float64)
    Y = w[cols].to_numpy(np.float64)
    if not (np.isfinite(gt).all() and np.isfinite(Y).all()):
        raise SystemExit("valori non finiti nelle colonne del bootstrap")
    point = spearman_many(gt, Y)
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, len(cols)))
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, len(subjects), size=len(subjects)), minlength=len(subjects))
        wt = counts[sa] * counts[sb]
        keep = wt > 0
        reps[b] = spearman_many(np.repeat(gt[keep], wt[keep]), np.repeat(Y[keep], wt[keep], axis=0))
    return point, reps


def _task(args):
    key, df, cols, n_boot, seed = args
    point, reps = multi_bootstrap(df, cols, n_boot, seed)
    return key, cols, point, reps, len(df), df["subject_a"].nunique()


# ------------------------------------------------------------------------- markdown

def ci(point: float, reps: np.ndarray) -> str:
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return f"{point:.3f} [{lo:.2f}, {hi:.2f}]"


def ci_d(point: float, reps: np.ndarray) -> str:
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def contrasts(dom: dict, idx: dict, point: np.ndarray, reps: np.ndarray) -> list[tuple[str, str, str, bool]]:
    """(nome, stima [CI], P(boot <= 0), inquinata) dei contrasti fra varianti e bracci."""
    def s(c):
        return (point[idx[c]], reps[:, idx[c]]) if c in idx else None

    rows = []

    def add(name, a, b, pol):
        if a is None or b is None:
            rows.append((name, "mancante", "-", pol))
            return
        d, dr = a[0] - b[0], a[1] - b[1]
        rows.append((name, ci_d(d, dr), f"{np.mean(dr <= 0):.3f}", pol))

    pol = lambda *arms: any(a in dom["polluted"] for a in arms)  # noqa: E731
    own = {a: f"{a}{dom['own'][a]}" for a in ARMS}
    ch = (point[idx["chamfer"]], reps[:, idx["chamfer"]])
    for a in ARMS:
        if dom["own"][a]:
            add(f"{LABEL[a]}: frame dei suoi dati ({dom['own'][a]}) - nativo", s(own[a]), s(a), pol(a))
    add("BFM+ICT nativo - BFM-only nativo (gap a parita' di ingresso)", s("joint"), s("bfm_only"), pol("joint"))
    add("BFM+ICT nativo - BFM-only nel frame dei suoi dati", s("joint"), s(own["bfm_only"]), pol("joint"))
    add("BFM+ICT nativo - ICT-only nel frame dei suoi dati", s("joint"), s(own["ict_only"]), pol("joint", "ict_only"))
    # frazione del gap congiunto - BFM-only spiegata dal frame: (BFM-only proprio - nativo) / gap
    a, bn, bo = s("joint"), s("bfm_only"), s(own["bfm_only"])
    if None in (a, bn, bo):
        rows.append(("frazione del gap BFM+ICT - BFM-only spiegata dal frame", "mancante", "-", pol("joint")))
    else:
        gap, gapr = a[0] - bn[0], a[1] - bn[1]
        fe, fer = bo[0] - bn[0], bo[1] - bn[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            fr, frr = fe / gap, fer / gapr
        frr = frr[np.isfinite(frr)]
        lo, hi = np.percentile(frr, [2.5, 97.5])
        note = "" if np.percentile(gapr, 2.5) > 0 else " (gap con CI che tocca 0: rapporto instabile)"
        rows.append(("frazione del gap BFM+ICT - BFM-only spiegata dal frame",
                     f"{fr:+.2f} [{lo:+.2f}, {hi:+.2f}]{note}", "-", pol("joint")))
    for a_ in ARMS:
        add(f"{LABEL[a_]} nel frame dei suoi dati - Chamfer eval", s(own[a_]), ch, pol(a_))
    return rows


def leak_report() -> str:
    splits = json.loads((RUNS / "ws2_cross3dmm" / "splits.json").read_text())
    subj_file = next(iter(sorted(data_dir(DOMAINS["ict"]).glob("*/subjects.json"))))
    subjects = json.loads(subj_file.read_text())["subjects"]
    # id924501 (vista zs, offset 920000) <-> id14501 (ICT-5000, offset 10000)
    orig = {f"id{int(s[2:]) - 920000 + 10000}" for s in subjects}
    parts = []
    for arm in ARMS:
        train = set(splits["models"][arm]["train"])
        parts.append(f"{LABEL[arm]} {len(orig & train)}/{len(orig)}")
    return ("Soggetti ICT valutati che sono nel training di ciascun modello (splits.json di WS2): "
            + ", ".join(parts) + ".")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=RUNS / "ws_frame" / "summary.md")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    tables, tasks = {}, []
    for dk, dom in DOMAINS.items():
        t, cols, chk = load_table(dom)
        tables[dk] = (t, cols, chk)
        print(f"[ws-frame] {dom['label']}: {len(t)} righe, colonne {cols}", flush=True)
        for pk, _ in PROTOCOLS:
            f = protocol_frame(t, pk, cols)
            seed = a.seed + zlib.crc32(f"{dk}|{pk}".encode()) % 1_000_000
            tasks.append(((dk, pk), f, ["chamfer"] + cols, a.n_bootstrap, seed))
    n_workers = min(len(tasks), int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    res = {}
    with mp.get_context("fork").Pool(n_workers) as pool:
        for key, cols, point, reps, n_rows, _ in pool.imap_unordered(_task, tasks):
            res[key] = (cols, point, reps, n_rows)
            print(f"[ws-frame] {key} fatto ({n_rows} righe)", flush=True)

    out = ["# WS-frame: frame, verso delle facce e normalizzazione nei test zero-shot\n",
           __doc__[__doc__.index("Protocolli,"):].strip() + "\n",
           f"Tutti i numeri: 100 soggetti per dominio, gli stessi in ogni variante e braccio; GT maxabs "
           f"(protocollo ICT); CI 95% percentili su {a.n_bootstrap} repliche bootstrap per soggetto, le "
           "STESSE per tutte le colonne di un dominio e protocollo. P(boot <= 0) = frazione di repliche "
           "con contrasto <= 0. Tabelle per variante (anche con la GT nei coefficienti) in "
           "`aau/runs/<ws_hifi3d|ws_faceverse|ws_ictzs>/summary<variante>.md`.\n",
           leak_report() + "\n"]
    for pk, plabel in PROTOCOLS:
        out.append(f"\n## {plabel}\n")
        for dk, dom in DOMAINS.items():
            cols, point, reps, n_rows = res[(dk, pk)]
            idx = {c: i for i, c in enumerate(cols)}
            out.append(f"\n### {dom['label']} ({n_rows} righe)\n")
            out += ["| variante | ingresso | " + " | ".join(LABEL[x] for x in ARMS) + " | Chamfer eval |",
                    "|" + " --- |" * (len(ARMS) + 3)]
            for v, desc, conv in dom["variants"]:
                cells = []
                for arm in ARMS:
                    c = f"{arm}{v}"
                    txt = ci(point[idx[c]], reps[:, idx[c]]) if c in idx else "mancante"
                    if c in idx and arm in dom["polluted"]:
                        txt += f" ({POLLUTED})"
                    cells.append(txt)
                out.append(f"| {desc} | {conv} | " + " | ".join(cells)
                           + f" | {ci(point[idx['chamfer']], reps[:, idx['chamfer']])} |")
            out += ["\n| contrasto (appaiato, stesse repliche) | stima [CI 95%] | P(boot <= 0) |",
                    "| --- | --- | --- |"]
            for name, est, p, pol in contrasts(dom, idx, point, reps):
                out.append(f"| {name}{' -- ' + POLLUTED if pol else ''} | {est} | {p} |")
    out.append("\n## Coerenza con zs_summarize.py (stessi dati, stessa definizione di Spearman)\n")
    out += ["| dominio | braccio nativo | protocollo | qui | zs_summarize | ", "| --- | --- | --- | --- | --- |"]
    for dk, dom in DOMAINS.items():
        tc = pd.read_csv(RUNS / dom["dir"] / "table_cells.csv")
        for pk, proto in (("nocrop", "mesh_pair_nocrop_cross"), ("spm", "subject_pair_mean")):
            cols, point, _, _ = res[(dk, pk)]
            for arm in ARMS:
                r = tc[(tc["model"] == arm) & (tc["gt"] == "maxabs") & (tc["protocol"] == proto) & (tc["scenario"] == "clean")]
                ref = float(r.iloc[0]["latent_point_check" if pk == "spm" else "latent_spearman"]) if not r.empty else np.nan
                out.append(f"| {dom['label']} | {LABEL[arm]} | {pk} | {point[cols.index(arm)]:.6f} | {ref:.6f} |")
    out.append("\n## Crop per singola coppia di topologie (HIFI3D): riga crop -> B, Spearman latent, GT maxabs\n")
    out.append("SOLO PUNTO, senza CI: in zs_summarize.py le 30 celle per coppia di topologie hanno "
               "n_bootstrap=0. Il CI del crop aggregato e' nella sezione A PARTE qui sopra.\n")
    out += ["| braccio | variante | crop->down8k | crop->noisy | crop->original | crop->remesh | crop->up60k |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    hdir = RUNS / "ws_hifi3d"
    for v, desc in (("", "maxabs (nativo)"), ("_evalframe-rms", "rms")):
        p = hdir / f"topology_pairs{v}.csv"
        if not p.exists():
            continue
        tp = pd.read_csv(p)
        for metric, arms in (("latent", ARMS), ("chamfer", ("joint",))):
            for arm in arms:
                s = tp[(tp["model"] == arm) & (tp["metric"] == metric) & (tp["gt"] == "maxabs")].set_index("topology_pair")
                vals = [f"{s.loc[f'crop__to__{b}', 'spearman']:.3f}" for b in ("down8k", "noisy", "original", "remesh", "up60k")]
                name = LABEL[arm] if metric == "latent" else "Chamfer eval"
                out.append(f"| {name} | {desc} | " + " | ".join(vals) + " |")
    out.append("\n## Controllo di sanita': Chamfer eval di ogni colonna contro la prima del dominio\n")
    out += ["| dominio | colonna | max diff relativa | righe con diff relativa > 1e-4 |", "| --- | --- | --- | --- |"]
    for dk, (t, cols, chk) in tables.items():
        for c, (mx, nbad) in chk.items():
            out.append(f"| {DOMAINS[dk]['label']} | {c} | {mx:.2e} | {nbad} |")
    out.append("\nSu HIFI3D il nativo e' girato su L40S e le varianti su A10: 165 righe su 148.500 (tutte con "
               "`down8k`) differiscono fino al 3% relativo, con Spearman di Chamfer identico a 2e-6; dove "
               "nativo e variante sono sulla stessa GPU (ICT, entrambi A10) la Chamfer e' identica bit per bit.\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[ws-frame] scritto {a.out}", flush=True)


if __name__ == "__main__":
    main()
