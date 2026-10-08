#!/usr/bin/env python3
"""Tabelle di E9 dai JSON dei job: aau/runs/evidence/e9/tables.md (gira anche sul frontend, solo stdlib).

    python3 aau/evidence/e9_bench/summarize.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
E9 = REPO_ROOT / "aau/runs/evidence/e9"
CATS = ("ict_down8k", "ict_remesh", "bfm_down8k", "ict_original", "bfm_remesh", "bfm_original",
        "ict_up60k", "bfm_up60k")


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def ops_tables(out: list[str]) -> None:
    runs = {p.stem: load(p) for p in sorted((E9 / "ops").glob("*.json"))}
    if not runs:
        return
    out += ["## (a) e (d): operatori su CPU", "",
            "| configurazione | build_grad | k | campione | P | compiti | wall s | mesh/s | regime P/t medio | s/mesh per processo | RSS max worker MB |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(runs, key=lambda n: (n[0], runs[n]["grad"], -runs[n]["k"], n.split("_")[3], -runs[n]["P"]))
    for n in order:
        r = runs[n]
        samp = "200 (3.3k-60k)" if "full" in n else ("40 (5 per cat.)" if "p1sub" in n else "400 (V<=10k)")
        out.append(f"| {n} | {r['grad']} | {r['k']} | {samp} | {r['P']} | {r['n_tasks']} | {r['wall_s']:.1f} | "
                   f"**{r['mesh_per_s']:.2f}** | {r['steady_mesh_per_s']:.2f} | {r['cpu_s_per_mesh']:.2f} | "
                   f"{r['max_worker_rss_mb']:.0f} |")
    out.append("")
    # tempo per mesh per categoria: P=1 (senza contesa) e P=64/P=100 (sotto carico), profilo per fase
    for n in ("a_orig_k128_p1sub_P1", "a_orig_k64_p1sub_P1", "a_orig_k128_full_P64", "a_orig_k64_full_P64",
              "d_vec_k64_full_P100", "d_vec_k128_full_P100", "d_vec_k64_le10k_P100", "d_vec_k128_le10k_P100"):
        r = runs.get(n)
        if not r:
            continue
        out += [f"### Tempo per mesh, {n} (mediane per categoria, secondi)", "",
                "| categoria | V | n | totale | Laplaciano+massa | eigsh | build_grad | save | resto | mesh/s stimate solo questa categoria (P / t) |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for c in CATS:
            m = [x for x in r["per_mesh"] if x["cat"] == c]
            if not m:
                continue
            med = lambda k: st.median(x[k] for x in m)  # noqa: E731
            out.append(f"| {c} | {med('V'):.0f} | {len(m)} | {med('t'):.2f} | {med('lap'):.3f} | {med('eigsh'):.2f} | "
                       f"{med('grad'):.3f} | {med('save'):.3f} | {med('rest'):.2f} | "
                       f"{r['P'] / st.mean(x['t'] for x in m):.1f} |")
        out.append("")
    for n in ("a_orig_k128_full_P64", "a_orig_k64_full_P64"):
        r = runs.get(n)
        if not r or "sizes" not in r:
            continue
        out += [f"### Dimensioni per mesh, k {r['k']} (MB = 10^6 byte, mediane per categoria)", "",
                "| categoria | V | nnz | fp32 + int64 (legacy) | fp32 + int32 npz (file del pre-pass) | evecs fp16 + int32 | idem + zlib | solo forward, evecs fp16 |",
                "|---|---|---|---|---|---|---|---|"]
        by = {}
        for s in r["sizes"]:
            for c in CATS:
                if s["name"].startswith(("id1" if c.startswith("ict") else "id0")) and f"_{c.split('_')[1]}_r0" in s["name"]:
                    by.setdefault(c, []).append(s)
        for c in CATS:
            m = by.get(c, [])
            if not m:
                continue
            med = lambda k: st.median(x[k] for x in m) / 1e6  # noqa: E731
            out.append(f"| {c} | {st.median(x['V'] for x in m):.0f} | {st.median(x['nnz'] for x in m):.0f} | "
                       f"{med('fp32_i64'):.1f} | {med('file_fp32_i32_npz'):.1f} | {med('e16_i32'):.1f} | "
                       f"{med('e16_i32_zlib'):.1f} | {med('forward_only_e16_i32'):.1f} |")
        tot = {k: sum(x[k] for x in r["sizes"]) / len(r["sizes"]) / 1e6
               for k in ("fp32_i64", "file_fp32_i32_npz", "e16_i32", "e16_i32_zlib", "forward_only_e16_i32")}
        same = all(x["same_sparsity_L_gradX_gradY"] for x in r["sizes"])
        out += [f"| media sulle 200 | | | {tot['fp32_i64']:.1f} | {tot['file_fp32_i32_npz']:.1f} | {tot['e16_i32']:.1f} | "
                f"{tot['e16_i32_zlib']:.1f} | {tot['forward_only_e16_i32']:.1f} |", "",
                f"L, gradX e gradY hanno la stessa sparsita' su tutte le 200 mesh: {same}.", ""]


def grad_table(out: list[str]) -> None:
    g = load(E9 / "grad_check.json")
    if not g:
        return
    out += ["## (d) build_grad vettorizzato contro l'originale", "",
            f"20 mesh, stessi archi e vettori tangenti. Struttura identica: {g['all_same_structure']}. "
            f"Errore in complex128: max assoluto {g['worst_complex128']['max_abs']:.2e}, relativo al massimo "
            f"{g['worst_complex128']['max_abs_over_max_ref']:.2e}. Dopo il cast float32 di compute_operators: "
            f"max assoluto {g['worst']['max_abs']:.2e}, relativo elemento per elemento {g['worst']['max_rel_elementwise']:.2e}. "
            f"Embedding e108 (loader congelato, CPU): max |dz| {g['embedding']['max_dz']:.2e} su distanza mediana "
            f"{g['embedding']['median_pairwise_dist']:.3f}. Speedup del solo build_grad sulle 20 mesh: "
            f"{g['speedup_total']:.0f}x.", "",
            "| mesh | V | build_grad originale s | vettorizzato ms | speedup |", "|---|---|---|---|---|"]
    for m in sorted(g["meshes"], key=lambda m: m["V"]):
        out.append(f"| {m['name']} | {m['V']} | {m['t_orig_s']:.2f} | {m['t_vec_s'] * 1e3:.1f} | "
                   f"{m['t_orig_s'] / m['t_vec_s']:.0f}x |")
    out.append("")


def fwd_tables(out: list[str]) -> None:
    d = E9 / "fwd"
    env = load(d / "env.json")
    if not env:
        return
    eq = load(d / "equivalence.json")
    out += ["## (b) forward + backward su 1 L40S", "",
            f"GPU {env['gpu']}, memoria totale {env['mem_total_gb']:.1f} GB, torch {env['torch']}, "
            f"TF32 matmul {env['allow_tf32_matmul']} (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE={env['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']}), "
            f"nodo {env['host']}, job {env['job']}.", ""]
    if eq:
        out += [f"Equivalenza con il percorso un-forward-per-mesh (eval, modello S, V {eq['n']}): " + "; ".join(
            f"{p}: max |dz| {v['max_abs_z']:.1e} (scala {v['z_scale']:.2f}), max |dgrad| {v['max_abs_grad']:.1e} "
            f"(scala {v['grad_scale']:.2f})" for p, v in eq["vs_seq"].items()), ""]
    homo = [load(p) for p in sorted(d.glob("homo_*.json"))]
    if homo:
        out += ["### Gruppi omogenei (stessa topologia, B mesh per passo)", "",
                "| taglia | V | seq mesh/s (B) | bat mesh/s migliore (B) | bat B max | pk mesh/s migliore (B) | pk B max | pk_ck mesh/s migliore (B) | pk_ck B max | picco GB a B max bat |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for h in sorted(homo, key=lambda h: ("SML".index(h["size"]), h["V"])):
            s = h["sweeps"]

            def cell(p):
                x = s.get(p)
                if not x or x["best_mesh_per_s"] is None:
                    return "n/d | n/d"
                bm = f"{x['B_max']}" + ("+ (tetto)" if x["capped"] else "")
                return f"{x['best_mesh_per_s']:.1f} ({x['best_B']}) | {bm}"
            seq = h.get("seq")
            seqc = f"{seq['mesh_per_s']:.1f} ({seq['B']})" if seq and not seq.get("oom") else "n/d"
            bat = s.get("bat:0.05", {})
            pk_rows = [r for r in bat.get("rows", []) if not r["oom"]]
            peak = f"{pk_rows[-1]['peak_gb']:.1f} (B={pk_rows[-1]['B']})" if pk_rows else "n/d"
            out.append(f"| {h['size']} | {h['V']} | {seqc} | {cell('bat:0.05')} | {cell('pk:0.25')} | {cell('pk_ck:0.25')} | {peak} |")
        out.append("")
        out += ["Curve complete (mesh/s e picco GB per B):", ""]
        for h in sorted(homo, key=lambda h: ("SML".index(h["size"]), h["V"])):
            for p, x in h["sweeps"].items():
                pts = ", ".join(f"{r['B']}: {r['mesh_per_s']:.0f}/{r['peak_gb']:.1f}" if not r["oom"] else f"{r['B']}: OOM"
                                for r in x["rows"])
                out.append(f"- {h['size']} V={h['V']} {p}: {pts}")
        out.append("")
    mixes = [load(p) for p in sorted(d.glob("mix_*.json"))]
    if mixes:
        out += ["### Passo misto: 64 mesh con V continuo (sorgenti reali troncate al 70-100%)", "",
                "| taglia | mix | V min-mediana-max | percorso | B | mesh/s | picco GB | gruppi | righe con padding / righe vere |",
                "|---|---|---|---|---|---|---|---|---|"]
        for m in sorted(mixes, key=lambda m: ("SML".index(m["size"]), m["mix"])):
            v = sorted(m["V"])
            for p, r in m["paths"].items():
                out.append(f"| {m['size']} | {m['mix']} | {v[0]}-{v[len(v) // 2]}-{v[-1]} | {p} | {r['B']} | "
                           + ("OOM | | |" if r["oom"] else f"**{r['mesh_per_s']:.1f}** | {r['peak_gb']:.1f} | "
                              f"{r.get('n_groups', '')} | {r.get('pad_rows_ratio', 1.0):.2f} |"))
        out.append("")


def nccl_tables(out: list[str]) -> None:
    rs = [load(p) for p in sorted((E9 / "nccl").glob("*.json"))]
    if not rs:
        return
    out += ["## (c) all_reduce fra due GPU", "",
            "| backend | rank | nodi | 8 B ms | 10 MB ms (GB/s) | 50 MB ms (GB/s) | 100 MB ms (GB/s) |",
            "|---|---|---|---|---|---|---|"]
    for r in rs:
        x = r["results"]
        f = lambda t: f"{x[t]['time_ms']:.2f} ({x[t]['algbw_GBps']:.2f})"  # noqa: E731
        out.append(f"| {r['backend'] if 'backend' in r else 'nccl'} {r.get('nccl') or ''} | {r['world']} | "
                   f"{', '.join(sorted(set(r['hosts'])))} | {x['8B']['time_ms']:.3f} | {f('10MB')} | {f('50MB')} | {f('100MB')} |")
    out.append("")


def main() -> None:
    out = ["# E9: tabelle generate da aau/evidence/e9_bench/summarize.py", ""]
    ops_tables(out)
    grad_table(out)
    fwd_tables(out)
    nccl_tables(out)
    (E9 / "tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
