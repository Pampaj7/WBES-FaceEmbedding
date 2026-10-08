#!/usr/bin/env python3
"""Tabelle di E10 dai JSON dei job: aau/runs/evidence/e10/tables.md (gira sul frontend, solo stdlib).

    python3 aau/evidence/e10_gpu_ops/summarize.py

Il confronto con la CPU usa i JSON di E9 (aau/runs/evidence/e9/ops, nodo L40S con 100 CPU logiche,
build_grad vettorizzato): per i campioni interi il mesh/s misurato, per le classi omogenee la stima
"P / mediana del tempo per mesh della categoria" sotto carico (la stessa colonna delle tabelle E9).
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
E9 = REPO_ROOT / "aau/runs/evidence/e9/ops"
E10 = REPO_ROOT / "aau/runs/evidence/e10"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def e9_runs() -> dict:
    return {p.stem: load(p) for p in sorted(E9.glob("d_vec_*.json"))}


def e9_cat_rate(runs: dict, cat: str, k: int) -> tuple[float, str] | None:
    """Stima E9 per una sola categoria: P / media(t), come la colonna "(P / t)" delle tabelle E9, dalla
    configurazione a P=100 che la contiene."""
    for name in (f"d_vec_k{k}_full_P100", f"d_vec_k{k}_le10k_P100"):
        r = runs.get(name)
        if r is None:
            continue
        ts = [m["t"] for m in r["per_mesh"] if m["cat"] == cat]
        if ts:
            return r["P"] / st.mean(ts), name
    return None


def e9_sample_best(runs: dict, sample: str, k: int) -> tuple[float, str] | None:
    tag = "full" if sample == "full200" else "le10k"
    cands = [(r["mesh_per_s"], n) for n, r in runs.items() if r["k"] == k and f"_{tag}_" in n]
    return max(cands) if cands else None


def check_tables(out: list[str]) -> None:
    files = sorted(E10.glob("check_k*.json"))
    if not files:
        return
    out += ["## Correttezza: operatori GPU contro CPU (ops_areanorm) su 60 mesh", "",
            "Campione: BFM e ICT x {original, remesh, down8k, up60k, noisy, crop} x 5 identita'. "
            "`cpu_v0` = stessa CPU con un altro vettore iniziale di eigsh (il rumore del solo risolutore CPU). "
            "Angoli fra sottospazi M-ortonormali; 'primi k-8' = angolo massimo dei primi k-8 vettori GPU "
            "dentro lo span dei k CPU (insensibile al mescolamento al bordo dello spettro). Embedding: "
            "checkpoint e108, loader congelato, CPU; dz = |z_var - z_cpu|.", ""]
    for f in files:
        d = load(f)
        out += [f"### k = {d['k']}, GPU {d.get('gpu', '?')} ({f.name})", "",
                f"Distanza mediana fra coppie di embedding CPU {d['embedding']['median_pairwise_dist']:.3f}; "
                f"distanza dal vicino piu' prossimo: minima {d['embedding']['nn_dist_min']:.3f}, "
                f"mediana {d['embedding']['nn_dist_median']:.3f}.", "",
                "| variante | autovalori i>=1: err. rel. max (mediana) | lambda_0: err. ass. max | tutti identici in fp32 | angolo max gradi | primi k-8 max gradi | struttura L/grad identica | L, massa max rel | grad max rel | dz max | dz mediana | dz max / dist. mediana | dz max / dist. min vicino | cos min |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v, ops in d["ops"].items():
            o = list(ops.values())
            e = d["embedding"][v]
            out.append(
                f"| {v} | {max(x['evals_relerr_max'] for x in o):.1e} ({st.median(x['evals_relerr_median'] for x in o):.1e}) | "
                f"{max(x['eval0_abs'] for x in o):.1e} | {sum(x['evals_identical_fp32'] for x in o)}/{len(o)} | {max(x['angle_max_deg'] for x in o):.2e} | "
                f"{max(x['angle_first_k-8_into_k_max_deg'] for x in o):.2e} | "
                f"{all(x['L_same_structure'] and x['gradX_same_structure'] and x['gradY_same_structure'] for x in o)} | "
                f"{max(x['L_maxrel'] for x in o):.1e}, {max(x['mass_maxrel'] for x in o):.1e} | "
                f"{max(max(x['gradX_maxrel'], x['gradY_maxrel']) for x in o):.1e} | **{e['dz_max']:.2e}** | "
                f"{e['dz_median']:.2e} | {e['dz_max_rel_median_pair']:.1e} | {e['dz_max'] / d['embedding']['nn_dist_min']:.1e} | "
                f"{e['cos_min']:.8f} |")
        out.append("")
        # per categoria, variante principale
        main = "gpu_bk" if "gpu_bk" in d["ops"] else next(iter(d["ops"]))
        files_e = d["embedding"]["files"]
        dz = dict(zip(files_e, d["embedding"][main]["dz"]))
        dz0 = dict(zip(files_e, d["embedding"]["cpu_v0"]["dz"])) if "cpu_v0" in d["embedding"] else {}
        cats: dict = {}
        for n, m in d["meshes"].items():
            cats.setdefault(m["cat"], []).append(n)
        out += [f"Per categoria ({main}):", "",
                "| categoria | V mediano | grad max rel | angolo max gradi | dz max | dz max cpu_v0 | iterazioni (solve, colonne) | batch s (B=5; il primo include il riscaldamento) |",
                "|---|---|---|---|---|---|---|---|"]
        for c in sorted(cats):
            ns = cats[c]
            o = [d["ops"][main][n] for n in ns]
            run = d["runs"][main][ns[0]]
            out.append(f"| {c} | {st.median(x['V'] for x in o):.0f} | "
                       f"{max(max(x['gradX_maxrel'], x['gradY_maxrel']) for x in o):.1e} | "
                       f"{max(x['angle_max_deg'] for x in o):.2e} | {max(dz[n] for n in ns):.2e} | "
                       f"{max(dz0.get(n, float('nan')) for n in ns):.2e} | "
                       f"{run.get('n_solve', '')}, {run.get('rhs_cols', '')} | {run.get('batch_s', 0):.2f} |")
        out.append("")


def classes_tables(out: list[str], e9: dict) -> None:
    files = sorted((E10 / "bench").glob("classes_*.json"))
    for f in files:
        d = load(f)
        rows = [r for r in d["rows"] if not r.get("oom")]
        ooms = {(r["class"], r["k"]): r["B"] for r in d["rows"] if r.get("oom")}
        info = d["info"]
        out += [f"## Classi omogenee: {info['gpu']} (job {info['job']}, variante {info['variant']}, {f.name})", "",
                "Mesh gia' lette e normalizzate in RAM; tempo = geometria + autovettori + copia su host, "
                "media di 3 batch dopo uno di riscaldamento; un solo stream. Picco = memoria usata sul "
                "dispositivo (torch + cuDSS + contesto). CPU = stima E9 per la sola categoria (P / t medio, "
                "nodo L40S, 100 CPU logiche).", "",
                "| classe | V | k | latenza B=1 ms | picco B=1 GB | migliore mesh/s (B) | picco GB al migliore | B max (OOM a) | CPU nodo E9 mesh/s | GPU / CPU |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        vof = {r["class"]: r["V"] for r in rows}
        keys = sorted({(r["class"], r["k"]) for r in rows}, key=lambda t: (vof[t[0]], t[1]))
        for cname, k in keys:
            rr = [r for r in rows if r["class"] == cname and r["k"] == k]
            b1 = next((r for r in rr if r["B"] == 1), None)
            best = max(rr, key=lambda r: r["mesh_per_s"])
            cpu = e9_cat_rate(e9, cname, k)
            cpu_cols = f"{cpu[0]:.1f} | {best['mesh_per_s'] / cpu[0]:.2f}x |" if cpu else " | |"
            out.append(f"| {cname} | {best['V']} | {k} | {b1['s_per_mesh'] * 1e3:.0f} | {b1['dev_peak_gb']:.2f} | "
                       f"**{best['mesh_per_s']:.1f}** ({best['B']}) | {best['dev_peak_gb']:.1f} | "
                       f"{max(r['B'] for r in rr)} ({ooms.get((cname, k), '-')}) | " + cpu_cols)
        out += ["", "Curve (B: mesh/s / picco GB; fasi al B migliore in ms per mesh: analisi+fattorizzazione cuDSS, "
                "iterazioni, geometria, copia su host):", ""]
        for cname, k in keys:
            rr = sorted([r for r in rows if r["class"] == cname and r["k"] == k], key=lambda r: r["B"])
            best = max(rr, key=lambda r: r["mesh_per_s"])
            ph = best["phases"]
            B = best["B"]
            out.append(f"- {cname} k={k}: " + ", ".join(f"{r['B']}: {r['mesh_per_s']:.1f}/{r['dev_peak_gb']:.1f}"
                                                         for r in rr)
                       + f" | B={B}: fattorizzazione {ph.get('factor_s', 0) / B * 1e3:.0f}, iterazioni "
                       f"{ph.get('iter_s', 0) / B * 1e3:.0f}, geometria {ph.get('geom_s', 0) / B * 1e3:.1f}, "
                       f"copia {ph.get('d2h_s', 0) / B * 1e3:.1f}; convergenti {all(r['all_converged'] for r in rr)}")
        out.append("")


def methods_tables(out: list[str]) -> None:
    for f in sorted((E10 / "bench").glob("methods_*.json")):
        d = load(f)
        out += [f"## Vie alternative per gli autovettori, una mesh: {d['info']['gpu']} ({f.name})", "",
                "Errore rispetto all'eigh denso fp64 della stessa mesh (stessa A). Tempo = geometria + autovettori, "
                "seconda chiamata.", "",
                "| classe | V | k | via | s per mesh | autovalori err. rel. max (mediana) | note |", "|---|---|---|---|---|---|---|"]
        for r in d["rows"]:
            if r.get("spectrum"):
                continue
            if "error" in r:
                out.append(f"| {r['class']} | | {r['k']} | {r['method']} | | | {r['error'][:80]} |")
                continue
            note = []
            for key in ("outer", "n_solve", "rhs_cols", "converged"):
                if key in r:
                    note.append(f"{key} {r[key]}")
            out.append(f"| {r['class']} | {r['V']} | {r['k']} | {r['method']} | {r['s']:.3f} | "
                       f"{r['evals_relerr_max']:.1e} ({r['evals_relerr_median']:.1e}) | {', '.join(note)} |")
        out += ["", "Spettro di A = M^-1/2 L M^-1/2 (area 1):", "",
                "| classe | lambda_1 | lambda_64 | lambda_128 | lambda_max | lambda_max / lambda_128 |", "|---|---|---|---|---|---|"]
        for r in d["rows"]:
            if r.get("spectrum"):
                out.append(f"| {r['class']} | {r['lambda_1']:.1f} | {r['lambda_64']:.0f} | {r['lambda_128']:.0f} | "
                           f"{r['lambda_max']:.2e} | {r['lambda_max'] / r['lambda_128']:.1e} |")
        out.append("")


def e2e_tables(out: list[str], e9: dict) -> None:
    files = sorted((E10 / "bench").glob("e2e_*.json"))
    if not files:
        return
    out += ["## Pipeline completa sugli stessi campioni di E9", "",
            "Lettura e normalizzazione (thread), batch per categoria sulla GPU (piu' lavoratori, uno stream "
            "ciascuno), npz scritti su /tmp (thread) e cancellati. CPU = il migliore di E9 sullo stesso "
            "campione e lo stesso k (nodo L40S, 100 CPU logiche, build_grad vettorizzato).", "",
            "| GPU | file | campione | k | lavoratori GPU | CPU nel job | mesh | wall s | GPU mesh/s | convergenti | picco GB | CPU nodo E9 mesh/s (config.) | GPU / nodo CPU |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in files:
        d = load(f)
        for r in d["rows"]:
            cpu = e9_sample_best(e9, r["sample"], r["k"])
            out.append(f"| {d['info']['gpu']} | {f.name} | {r['sample']} | {r['k']} | {r['gpu_workers']} | {r['cpus_in_job']} | "
                       f"{r['n_meshes']} | {r['wall_s']:.1f} | **{r['mesh_per_s']:.1f}** | {r['n_converged']}/{r['n_meshes']} | "
                       f"{r['dev_peak_gb']:.1f} | {cpu[0]:.1f} ({cpu[1]}) | {r['mesh_per_s'] / cpu[0]:.2f}x |")
    out.append("")


def main() -> None:
    e9 = e9_runs()
    out = ["# E10: tabelle generate da aau/evidence/e10_gpu_ops/summarize.py", ""]
    check_tables(out)
    e2e_tables(out, e9)
    classes_tables(out, e9)
    methods_tables(out)
    (E10 / "tables.md").write_text("\n".join(out) + "\n")
    print(f"scritto {E10 / 'tables.md'} ({len(out)} righe)")


if __name__ == "__main__":
    main()
