#!/usr/bin/env python3
"""Stato e allarmi del run del modello da rilasciare (aau/massive/MONITOR.md). Solo stdlib: gira sul frontend.

    python3 aau/massive/status.py [--out aau/runs/massive_release/main_a100] [--json]

Legge ``<out>/steps.jsonl`` (train_v3.step_jsonl), ``<out>/monitor/heldout.jsonl`` (heldout.py watch),
``<out>/segments.log``, ``<out>/node*/producers.log``, la coda (squeue) e il disco (getfattr ceph.dir.rbytes su ~).
Allarmi (uscita 2 se almeno uno):
  NAN      loss non finite nel jsonl (qualunque), o embedding dell'held-out non finiti;
  LOSS     loss media dell'ultima finestra > 3 x la mediana delle ultime 200 righe (dopo le prime 50);
  FERMO    job RUNNING ma nessuna riga di passo da piu' di ``--stall-min`` minuti (20);
  HELDOUT  media fra domini di sp_fr o rank1_fr scesa di piu' di 0.10 sotto il massimo visto finora; errori del
           sorvegliante;
  NODO     nodo del job down/drain/fail, o job uscito dalla coda prima della fine senza COMPLETED;
  DISCO    meno di 300 GB liberi sulla quota da 2 TiB (``--quota-gb`` 2199).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def rows(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for ln in path.read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REPO / "aau/runs/massive_release/main_a100"))
    ap.add_argument("--stall-min", type=float, default=20.0)
    ap.add_argument("--quota-gb", type=float, default=2199.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    alarms, info = [], {"out": str(out), "now": time.strftime("%F %T")}

    # --- passi
    st = rows(out / "steps.jsonl")
    steps = [r for r in st if "step" in r and r.get("event") != "start"]
    starts = [r for r in st if r.get("event") == "start"]
    cfgT = None
    launch = sorted(out.glob("node0/launch.txt"))
    if launch:
        toks = launch[0].read_text().split()
        if "--total-steps" in toks:
            cfgT = int(toks[toks.index("--total-steps") + 1])
    info["total_steps"] = cfgT
    info["starts"] = [{"step": r["step"], "job": r.get("job"), "t": time.strftime("%F %T", time.localtime(r["t"]))}
                      for r in starts]
    if steps:
        last = steps[-1]
        bad = sum(int(r.get("nonfinite", 0)) for r in steps)
        lm = [num(r["loss_mean"]) for r in steps[-200:]]
        recent = [r for r in steps if r["t"] > last["t"] - 3600]
        sph = (recent[-1]["step"] - recent[0]["step"]) / max(recent[-1]["t"] - recent[0]["t"], 1) * 3600 \
            if len(recent) > 1 else float("nan")
        info["step"] = last["step"]
        info["loss_mean"] = last["loss_mean"]
        info["lr"] = last["lr"]
        info["s_per_step"] = last["s_per_step"]
        info["steps_per_hour_last_hour"] = round(sph, 1)
        info["last_step_age_min"] = round((time.time() - last["t"]) / 60, 1)
        if cfgT and sph == sph and sph > 0:
            info["eta_hours"] = round((cfgT - last["step"]) / sph, 1)
        info["nonfinite_total"] = bad
        if bad:
            alarms.append(f"NAN: {bad} loss non finite nel jsonl (ultima riga con nonfinite: "
                          f"{next(r['step'] for r in reversed(steps) if r.get('nonfinite'))})")
        if len(steps) > 50 and lm:
            med = statistics.median(lm)
            if num(last["loss_mean"]) > 3 * med:
                alarms.append(f"LOSS: loss media {last['loss_mean']:.4g} > 3 x mediana recente {med:.4g}")

    # --- coda e nodi
    q = sh(["squeue", "-h", "-u", sh(["whoami"]), "-o", "%i|%j|%T|%M|%L|%N|%R"])
    mine = {str(r.get("job")) for r in starts if r.get("job")}      # i job di QUESTO run (righe di partenza)
    jobs = [ln.split("|") for ln in q.splitlines() if "wbes-massive-rel" in ln]
    if mine:
        jobs = [j for j in jobs if j[0] in mine] or [j for j in jobs if j[2] != "RUNNING"]
    info["jobs"] = [{"id": j[0], "name": j[1], "state": j[2], "elapsed": j[3], "left": j[4], "nodes": j[5],
                     "reason": j[6]} for j in jobs]
    running = [j for j in jobs if j[2] == "RUNNING"]
    for j in running:
        for node in sh(["scontrol", "show", "hostnames", j[5]]).split():
            stt = sh(["sinfo", "-h", "-n", node, "-o", "%T"])
            if any(x in stt for x in ("down", "drain", "fail")):
                alarms.append(f"NODO: {node} in stato {stt} (job {j[0]})")
        if steps and info["last_step_age_min"] > a.stall_min:
            alarms.append(f"FERMO: job {j[0]} RUNNING, ultima riga di passo {info['last_step_age_min']} min fa")
    if not jobs and steps and cfgT and steps[-1]["step"] < cfgT:
        ids = sorted({r.get("job") for r in starts if r.get("job")})
        acct = sh(["sacct", "-n", "-X", "-P", "-o", "JobID,State,End", "-j", ",".join(ids)]) if ids else ""
        alarms.append(f"NODO: nessun job in coda e run fermo al passo {steps[-1]['step']}/{cfgT}; sacct: "
                      f"{acct.splitlines()[-1] if acct else '?'}")

    # --- produttori
    prod = {}
    for f in sorted(out.glob("node*/producers.log")):
        ln = [x for x in f.read_text(errors="replace").splitlines() if "viste/s" in x or "/s nell'intervallo" in x]
        prod[f.parent.name] = ln[-1][:200] if ln else ""
    info["producers"] = prod

    # --- held-out
    ho = rows(out / "monitor" / "heldout.jsonl")
    ok = [r for r in ho if "mean" in r]
    err = [r for r in ho if "error" in r]
    if err:
        alarms.append(f"HELDOUT: {len(err)} checkpoint con errore (ultimo {Path(err[-1]['ckpt']).name}: "
                      f"{err[-1]['error'][:120]})")
    if any(r.get("finite") is False for r in ho):
        alarms.append("NAN: embedding non finiti sull'held-out")
    if ok:
        lr_ = ok[-1]
        info["heldout_last"] = {"ckpt": Path(lr_["ckpt"]).name, "mean": lr_["mean"], "c": lr_.get("c"),
                                "domains": {d: {k: round(v[k], 3) for k in ("sp_fr", "sp_sr", "rank1_fr")}
                                            for d, v in lr_["domains"].items()}}
        for k in ("sp_fr", "rank1_fr"):
            best = max(num(r["mean"][k]) for r in ok)
            info[f"heldout_best_{k}"] = best
            if num(lr_["mean"][k]) < best - 0.10:
                alarms.append(f"HELDOUT: {k} medio {lr_['mean'][k]:.3f} sotto il massimo {best:.3f} di piu' di 0.10")

    # --- disco
    used = sh(["getfattr", "--only-values", "-n", "ceph.dir.rbytes", str(Path.home())])
    try:
        free_gb = a.quota_gb - int(used) / 1e9
        info["disk_free_gb"] = round(free_gb, 1)
        info["run_dir_gb"] = round(int(sh(["getfattr", "--only-values", "-n", "ceph.dir.rbytes", str(out)]) or 0) / 1e9, 2)
        if free_gb < 300:
            alarms.append(f"DISCO: {free_gb:.0f} GB liberi (< 300)")
    except ValueError:
        info["disk_free_gb"] = None

    info["alarms"] = alarms
    if a.json:
        print(json.dumps(info, indent=1))
    else:
        for k, v in info.items():
            if k != "alarms":
                print(f"{k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
        print("ALLARMI: " + ("; ".join(alarms) if alarms else "nessuno"))
    raise SystemExit(2 if alarms else 0)


if __name__ == "__main__":
    main()
