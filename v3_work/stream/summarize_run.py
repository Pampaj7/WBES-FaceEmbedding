#!/usr/bin/env python3
"""Riepilogo di una directory di misure di slurm/train_stream.sbatch (aau/runs/evidence/stream/train_<job>).

    python3 v3_work/stream/summarize_run.py aau/runs/evidence/stream/train_<job> [...]   (solo stdlib)

Scrive ``summary.json`` nella directory e stampa una riga per run:
  * passo: s/passo per epoca (train.log), mesh/s consumate = identita' x viste massime per passo / s per passo
    (tetto) e usi del rank 0 / secondi dell'epoca (misurate); lr per epoca (deve restare costante con --lr-constant);
  * stream (stream_stats_rank0.jsonl): usi, viste distinte, riuso per epoca e cumulato (a regime: le ultime
    ``REUSE_WINDOW`` epoche; il tetto e' morbido), shard usati (seq), viste uscite dall'anello senza uso, gruppi
    oltre il tetto di riuso, attese e servizio per passo, batch misti per dominio e per fonti;
  * produttori (producers.json): viste fresche/s a regime;
  * GPU (gpu.csv): utilizzo medio mentre il trainer tiene la GPU (memoria usata > 1 GiB);
  * memoria (mem.csv): picco del cgroup del job e dell'anello su /tmp.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REUSE_WINDOW = 3          # epoche finali per il riuso a regime


def summarize(d: Path) -> dict:
    out: dict = {"dir": str(d)}
    log = (d / "train.log").read_text() if (d / "train.log").exists() else ""
    ep = [(int(a), int(b), float(c)) for a, b, c in re.findall(r"\[v3\] epoca (\d+): (\d+) passi .*? \(([\d.]+) s/passo\)", log)]
    losses = [float(x) for x in re.findall(r"^Epoch \d+ \| loss=([\d.naninf]+)", log, flags=re.M)]
    mesh_per_step = None
    lt = d / "launch.txt"
    if lt.exists():
        txt = lt.read_text()
        b = int(re.findall(r"--batch_subjects (\d+)", txt)[-1])          # l'ultimo vince (argparse)
        v = int(re.findall(r"--max_meshes_per_subject_train (\d+)", txt)[-1])
        mesh_per_step = b * v
    out["steps"] = sum(n for _, n, _ in ep)
    out["s_per_step_by_epoch"] = [s for _, _, s in ep]
    out["loss_by_epoch"] = losses
    out["lr_by_epoch"] = [float(x) for x in re.findall(r"^Epoch \d+ \| .*? lr=([\d.e+-]+)", log, flags=re.M)]
    out["loss_finite"] = all(x == x and abs(x) != float("inf") for x in losses)
    if mesh_per_step and ep:
        out["meshes_per_step"] = mesh_per_step
        out["meshes_per_s_consumed"] = [mesh_per_step / s for _, _, s in ep]
    st = sorted((d / "runs").glob("*/stream_stats_rank0.jsonl"))
    if st:
        rows = [json.loads(x) for x in st[0].read_text().splitlines() if x.strip()]
        out["stream_by_epoch"] = [{
            "epoch": r["epoch"], "seq_used": [r["seq_used_min"], r["seq_used_max"]], "uses": r["d_uses"],
            "unique_views": r["d_unique_views_used"], "reuse_epoch": r["d_uses"] / max(r["d_unique_views_used"], 1),
            "reuse_cumulative": r["reuse_factor"], "views_evicted_unused": r["d_views_evicted_unused"],
            "over_reuse_groups": r["d_over_reuse_groups"], "wait_ms_per_step": 1e3 * r["d_wait_s"] / max(r["plans_yielded"], 1),
            "serve_ms_per_step": 1e3 * r["d_serve_s"] / max(r["plans_yielded"], 1),
            "plan_ms_per_step": 1e3 * r["d_plan_s"] / max(r["plans_yielded"], 1), "mean_age_s": r["mean_age_s"],
            "pool_groups": r["pool_groups"], "meshes_per_s_rank0": r["d_uses"] / max(r["seconds"], 1e-9),
            "batches_mixed": r.get("d_batches_mixed"), "batches_mixed_sources": r.get("d_batches_mixed_sources")}
            for r in rows]
        w = rows[-REUSE_WINDOW:]
        out["reuse"] = {"cap_soft": rows[-1].get("reuse_cap"), "cumulative": rows[-1]["reuse_factor"],
                        "steady_last_epochs": sum(r["d_uses"] for r in w) / max(sum(r["d_unique_views_used"] for r in w), 1),
                        "window_epochs": len(w), "over_reuse_groups": sum(r["d_over_reuse_groups"] for r in rows)}
        out["by_domain_uses"] = rows[-1]["by_domain"]
        out["batches_by_domain"] = rows[-1].get("batches_by_domain")
        out["batches_by_sources"] = rows[-1].get("batches_by_sources")
        out["by_label_uses"] = rows[-1]["by_label"]
    if (d / "producers.json").exists():
        p = json.loads((d / "producers.json").read_text())
        out["producers"] = {k: p[k] for k in ("n_proc", "k_eig", "evecs_dtype", "views_per_s_steady", "views_per_s_wall",
                                              "views", "bytes_per_view", "mean_verts_per_view", "failures", "views_by_domain",
                                              "views_by_label")}
        out["producers"].update({k: p.get(k) for k in ("expr_frac", "expr_fraction", "expr_fraction_by_domain",
                                                       "label_draw", "mm_aug", "views_by_origin")})
    if (d / "consumer.json").exists():
        out["consumer_bench"] = json.loads((d / "consumer.json").read_text())
    if (d / "gpu.csv").exists():
        util = []
        for line in (d / "gpu.csv").read_text().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 3 and f[1].endswith("%") and f[2].endswith("MiB") and float(f[2][:-4]) > 1024:
                util.append(float(f[1][:-1]))
        if util:
            out["gpu_util_mean_while_training"] = sum(util) / len(util)
            out["gpu_samples"] = len(util)
    if (d / "mem.csv").exists():
        cg, ring = [], []
        for line in (d / "mem.csv").read_text().splitlines():
            f = line.split(",")
            if len(f) >= 3 and f[1].isdigit():
                cg.append(int(f[1]))
            if len(f) >= 3 and f[2].strip().isdigit():
                ring.append(int(f[2]))
        out["mem_peak_gib"] = {"cgroup_job": max(cg) / 2 ** 30 if cg else None, "ring_tmp": max(ring) / 2 ** 30 if ring else None}
    (d / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    for a in sys.argv[1:]:
        s = summarize(Path(a))
        sp = s.get("s_per_step_by_epoch") or [float("nan")]
        print(f"{a}: {s['steps']} passi, s/passo {sp[0]:.3f} -> {sp[-1]:.3f}, GPU {s.get('gpu_util_mean_while_training', float('nan')):.0f}%, "
              f"fresche/s {s.get('producers', {}).get('views_per_s_steady', float('nan')):.1f}, "
              f"riuso a regime {s.get('reuse', {}).get('steady_last_epochs', float('nan')):.2f} (ultime "
              f"{s.get('reuse', {}).get('window_epochs')} epoche), cumulato {s.get('reuse', {}).get('cumulative', float('nan')):.2f}, "
              f"tetto morbido {s.get('reuse', {}).get('cap_soft')} (gruppi oltre {s.get('reuse', {}).get('over_reuse_groups')}), "
              f"lr {sorted(set(s.get('lr_by_epoch') or []))}, batch per dominio {s.get('batches_by_domain')}, "
              f"per fonti {s.get('batches_by_sources')}, "
              f"memoria {s.get('mem_peak_gib')}")


if __name__ == "__main__":
    main()
