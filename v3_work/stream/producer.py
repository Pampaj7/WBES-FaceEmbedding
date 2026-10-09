#!/usr/bin/env python3
"""Produttori CPU della pipeline P1: viste fresche, operatori e shard nel buffer ad anello.

    aau/run.sh v3_work/stream/producer.py --ring /tmp/$SLURM_JOB_ID/stream --ring-gb 40 --n-proc 96 \
        [--k-eig 64|128] [--alpha 0] [--domains bfm2019,ict,gnm,flame2020,famos] [--duration 0]

Ogni processo (fork del padre, che carica modelli e basi una volta: le pagine restano condivise) ripete:
  1. dominio d ~ p_d proporzionale a n_d^alpha (``sources.domain_probs``; alpha 0 = uniforme);
  2. un'identita' fresca (3DMM: coefficienti a code larghe; FaMoS: una persona TRAIN) e il suo s_i dalla
     forma neutra (GT unificata);
  3. ``--views`` viste: espressione con probabilita' ``--p-expr`` per vista, o una quota esplicita per gruppo
     (``--expr-frac``) (prior del modello; FaMoS: un fotogramma registrato), discretizzazione estratta coi pesi
     ``--label-weights`` (up60k a bassa frequenza), operatori con ``--k-eig`` autovettori (views.make_view);
     con ``--canonical-gt`` anche i bersagli della GT di E12 dalla forma neutra (targets.py: FR, SR, S_i) e
     l'area in mm^2 veri di ogni vista (ingresso globale);
  4. ``--groups-per-shard`` identita' in uno shard, scritto nell'anello (ring.Ring, FIFO a ``--ring-gb``).
Tutti i numeri casuali di un processo vengono da SeedSequence([seme, processo]).

Run massivo (slurm/massive.sbatch, slurm/extra_producers.sbatch):
  * ``--sources massive|open_core|<domini>``: preset delle fonti (sources.SOURCE_PRESETS; FLAME 2023 Open, non 2020);
  * ``--provenance``: un generatore per gruppo, SeedSequence([seme, processo, gruppo]), e nello shard seme, fonti,
    licenza ereditata, coefficienti, ricetta: regen.py rigenera ogni vista;
  * ``--mm-aug hybrid=..,expr_transfer=..,rbf=..``: gruppi dei moltiplicatori di v3_work/mm_aug (aug_group_spec:
    UNA identita' ibrida / con bump / pura col trasferimento d'espressione e le sue viste), GT dalla neutra esatta;
  * ``--stats-dir`` per un anello condiviso da piu' job, ``--evict-every`` per un anello su CephFS.

Statistiche: ``<ring>/stats/w<NNN>.json`` per processo (viste, gruppi, byte, secondi per fase, fallimenti) e,
dal padre, una riga ogni ``--stats-every`` s con viste/s nell'intervallo, shard e GiB nell'anello;
``--summary`` scrive il riepilogo finale in JSON (regime = dopo il primo shard di ogni processo).

Thread BLAS a 1 per processo (imposti qui, prima di numpy): la parallelizzazione e' per processo.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import sources as S  # noqa: E402
from ring import Ring  # noqa: E402


def parse_weights(txt: str, keys) -> dict:
    out = {}
    for item in txt.split(","):
        k, v = item.split("=")
        if k not in keys:
            raise SystemExit(f"{k!r} non in {keys}")
        out[k] = float(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    import views as VW
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ring", required=True, help="directory dell'anello (su /tmp: conta contro --mem)")
    p.add_argument("--ring-gb", type=float, default=32.0, help="budget dell'anello in GiB (FIFO)")
    p.add_argument("--evict-every", type=int, default=1,
                   help="budget applicato ogni N scritture del processo (anello condiviso su CephFS: es. 8)")
    p.add_argument("--n-proc", type=int, default=8)
    p.add_argument("--k-eig", type=int, default=128, choices=[64, 128])
    p.add_argument("--evecs-dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--domains", default=",".join(S.DOMAINS))
    p.add_argument("--sources", default="", help="fonti, sostituisce --domains: preset (massive = bfm2019, ict, gnm, "
                                                 "flame2023, famos; open_core = gnm, ict, flame2023) o domini")
    p.add_argument("--provenance", action="store_true",
                   help="un seme per gruppo, SeedSequence([seme, processo, gruppo]): ogni vista e' rigenerabile "
                        "(regen.py); il gruppo porta seme, licenza ereditata, coefficienti e la ricetta")
    p.add_argument("--alpha", type=float, default=0.0, help="p_d ~ n_d^alpha (0 = uniforme fra domini)")
    p.add_argument("--domain-sizes", default="", help="n_d, es. 'famos=80,ict=100' (default: modi / persone)")
    p.add_argument("--views", type=int, default=4, help="viste per identita'")
    p.add_argument("--groups-per-shard", type=int, default=2)
    p.add_argument("--p-expr", type=float, default=0.6)
    p.add_argument("--expr-frac", type=float, default=-1.0,
                   help="quota esplicita: in ogni gruppo esattamente round(f * viste) viste con espressione (arrotondamento "
                        "stocastico), le altre neutre; < 0 = --p-expr per vista")
    p.add_argument("--canonical-gt", action="store_true",
                   help="bersagli della GT di E12 per gruppo (fr, sr, S: targets.py) e area_mm2 per vista (ingresso globale)")
    p.add_argument("--mm-aug", default="",
                   help="moltiplicatori di v3_work/mm_aug, probabilita' per gruppo dei 3DMM, es. "
                        "'hybrid=0.3,expr_transfer=0.15,rbf=0.15' (il resto: identita' pure; FaMoS sempre pura)")
    p.add_argument("--mm-aug-validate", default="cheap", choices=["none", "cheap", "full"],
                   help="controlli di validita' di mm_aug (cheap: triangoli capovolti e degeneri, 0.05 s per vista)")
    p.add_argument("--v-max", type=int, default=S.V_MAX, help="patch native oltre V_MAX vertici -> topologia decimata")
    p.add_argument("--v-work", type=int, default=S.V_WORK, help="vertici della topologia di lavoro decimata")
    p.add_argument("--label-weights", default=",".join(f"{k}={v:g}" for k, v in VW.LABEL_WEIGHTS.items()))
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--duration", type=float, default=0.0, help="secondi (0 = finche' non viene fermato)")
    p.add_argument("--stats-every", type=float, default=30.0)
    p.add_argument("--summary", default="", help="JSON del riepilogo finale")
    p.add_argument("--stats-dir", default="", help="statistiche per processo (default <anello>/stats; un anello condiviso "
                                                   "da piu' job: una directory per job)")
    p.add_argument("--cpus", default="", help="CPU logiche dei produttori, es. '8-31,40-63' (vuoto = tutte): le altre "
                                              "restano al trainer (la contesa rallenta il passo, vedi README)")
    return p


def cpu_list(txt: str) -> set:
    out: set = set()
    for item in txt.split(","):
        a, _, b = item.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


# --- moltiplicatori (v3_work/mm_aug): ibridi, trasferimento d'espressioni, bump RBF ---------------------------

MM_AUG_KINDS = ("hybrid", "expr_transfer", "rbf")


def check_mm_aug(txt: str) -> dict:
    """Probabilita' per gruppo dei tipi di --mm-aug (il resto: identita' pure)."""
    probs = parse_weights(txt, MM_AUG_KINDS)
    if sum(probs.values()) > 1 + 1e-9 or min(probs.values()) < 0:
        raise SystemExit(f"--mm-aug {txt}: probabilita' fuori da [0, 1]")
    return probs


def mm_aug_setup(cfg, domains) -> tuple:
    """(AugLibrary, AugConfig) di v3_work/mm_aug sulle fonti del produttore: template = i 3DMM fra ``domains``, fonti
    degli ibridi = gli stessi, fonti delle espressioni = gli stessi piu' FaMoS se c'e'. Nessuna fonte fuori da
    ``--sources``: col preset open_core gli ibridi restano nel nucleo aperto."""
    from v3_work.mm_aug import aug as AG
    tpls = tuple(d for d in domains if d in AG.ALL_TEMPLATES)
    if len(tpls) < 2:
        raise SystemExit(f"--mm-aug: servono almeno due 3DMM fra le fonti ({domains})")
    exprs = tpls + (("famos",) if "famos" in domains else ())
    acfg = AG.AugConfig(p_kind=(("pure", 0.0),) + tuple((k, float(v)) for k, v in cfg.mm_aug_probs.items()),
                        templates=tpls, hybrid_sources=tpls, expr_sources=exprs, p_expr=cfg.p_expr,
                        validate=cfg.mm_aug_validate)
    return AG.AugLibrary(acfg), acfg


def group_kind(rg: np.random.Generator, cfg, domain: str) -> str:
    """Tipo del gruppo: puro senza --mm-aug (nessuna estrazione: la sequenza del generatore resta quella di prima) e
    per FaMoS; altrimenti un'estrazione con le probabilita' di --mm-aug."""
    probs = getattr(cfg, "mm_aug_probs", None)
    if not probs or domain == "famos":
        return "pure"
    u, acc = rg.random(), 0.0
    for k in MM_AUG_KINDS:
        acc += probs.get(k, 0.0)
        if u < acc:
            return k
    return "pure"


def aug_group_spec(lib, acfg, A: str, kind: str, rg: np.random.Generator, cfg) -> dict:
    """Un gruppo dei moltiplicatori sul template A: UNA identita' (ibrida, con bump o pura per il trasferimento) e le
    sue ``cfg.views`` viste (espressione nativa di A, o trasferita da un'altra fonte per expr_transfer, con la quota
    di --expr-frac/--p-expr), con i controlli di validita' di mm_aug (``--mm-aug-validate``): un'identita' scartata
    si riestrae fino a max_tries volte e poi diventa pura, un'espressione scartata si riestrae e poi la vista resta
    neutra. Tutto dal generatore del gruppo: la usano make_aug_group e regen.py."""
    from v3_work.mm_aug import aug as AG
    check = acfg.validate
    for attempt in range(acfg.max_tries + 1):
        k = kind if attempt < acfg.max_tries else "pure"
        ident = AG.draw_identity(rg, "pure" if k == "expr_transfer" else k, A, acfg, lib)
        spec0 = AG.assemble({"identity": ident, "expression": None, "kind": k}, lib, check=check)
        if check == "none" or AG.valid(spec0, lib) or attempt == acfg.max_tries:
            break
    quota = expr_quota(rg, cfg)
    views, sources = [], [A] + ([ident["B"]] if ident.get("B") else [])
    for i in range(cfg.views):
        want = quota[i] if quota is not None else bool(rg.random() < cfg.p_expr)
        spec, expr = spec0, None
        if want:
            mode = "transfer" if k == "expr_transfer" else "native"
            for _ in range(acfg.max_tries):
                e = AG.draw_expression(rg, A, mode, acfg, lib)
                sp = AG.assemble({"identity": ident, "expression": e, "kind": k}, lib, check=check)
                if check == "none" or AG.valid(sp, lib):
                    spec, expr = sp, e
                    break
        if expr is not None and expr["source"] not in sources:
            sources.append(expr["source"])
        label = cfg.labels[int(rg.choice(len(cfg.labels), p=cfg.label_p))]
        views.append({"i": i, "V": spec["V"], "F": spec["F"], "tag": "expr" if expr else "neutral", "label": label,
                      "noise_seed": int(rg.integers(2 ** 31)), "expression": expr})
    return {"kind": k, "kind_drawn": kind, "attempt": attempt, "identity": ident, "P": spec0["neutral_points"],
            "views": views, "sources": sources}


def make_aug_group(lib, acfg, domain: str, kind: str, rg: np.random.Generator, cfg, uni: S.Unified, key: str,
                   st: dict, tg=None, seed=None) -> dict:
    """make_group per i gruppi dei moltiplicatori: GT (s_i e, con --canonical-gt, FR/SR/S_i) dalla neutra ESATTA
    del campione sulla regione unificata (mm_aug), viste e operatori come i puri."""
    import views as VW
    t0 = time.perf_counter()
    G = aug_group_spec(lib, acfg, domain, kind, rg, cfg)
    s = uni.svec(G["P"])
    extra, area_factor = {}, None
    if tg is not None:
        t = tg(domain, G["P"])
        extra = {"fr": t["fr"], "sr": t["sr"], "S": float(t["S"])}
        area_factor = cfg.mm_factor[domain]
    st["t_ident"] += time.perf_counter() - t0
    out = []
    for v in G["views"]:
        try:
            arr, meta = VW.make_view(v["V"], v["F"], v["label"], cfg.k_eig, cfg.evecs_dtype, v["noise_seed"],
                                     area_factor=area_factor)
        except Exception as exc:  # noqa: BLE001
            st["failures"] += 1
            st["last_failure"] = f"{domain}/{G['kind']}/{v['label']}: {type(exc).__name__}: {exc}"
            continue
        meta["expr"] = v["tag"]
        if seed is not None:
            meta.update(vi=v["i"], noise_seed=v["noise_seed"], frame=int((v["expression"] or {}).get("frame", -1)))
        for ph in ("gen", "ops", "pack"):
            st[f"t_{ph}"] += meta.pop(f"t_{ph}")
        st["views"] += 1
        st["by_label"][v["label"]] = st["by_label"].get(v["label"], 0) + 1
        st["verts"] += meta["n"]
        out.append((arr, meta))
    st["by_domain"][domain] = st["by_domain"].get(domain, 0) + len(out)
    st["by_origin"] = st.get("by_origin", {})
    st["by_origin"][G["kind"]] = st["by_origin"].get(G["kind"], 0) + len(out)
    st["expr_views"] = st.get("expr_views", 0) + sum(m["expr"] == "expr" for _, m in out)
    g = {"key": key, "domain": domain, "s": s, "views": out, "origin": G["kind"], **extra}
    if seed is not None:
        lic, rank = S.inherited_license(G["sources"])
        g["prov"] = {"seed": [int(x) for x in seed], "sources": G["sources"], "license": lic, "license_rank": rank,
                     "person": None, "kind_drawn": G["kind_drawn"], "attempt": G["attempt"],
                     "mm_aug": {"identity": G["identity"], "expressions": [v["expression"] for v in G["views"]]}}
        g["zid"] = np.asarray(G["identity"]["z_A"], dtype=np.float64)
    return g


# --- un gruppo = un'identita' con le sue viste -----------------------------------------------------------

def expr_quota(rng: np.random.Generator, cfg) -> list[bool] | None:
    """Quota esplicita (--expr-frac): quali viste del gruppo hanno l'espressione; None = Bernoulli(--p-expr) per
    vista, estratta nel ciclo delle viste come prima (stessa sequenza del generatore)."""
    if cfg.expr_frac < 0:
        return None
    x = cfg.expr_frac * cfg.views
    n = int(np.floor(x)) + int(rng.random() < x - np.floor(x))
    on = np.zeros(cfg.views, dtype=bool)
    on[rng.permutation(cfg.views)[:n]] = True
    return on.tolist()


def view_recipe(src, ident, rng: np.random.Generator, cfg):
    """Le viste di un gruppo prima della discretizzazione, nell'ordine in cui consumano il generatore:
    (indice, V, F, etichetta d'espressione, discretizzazione, seme del rumore, fotogramma FaMoS o -1).
    La usano make_group e regen.py: stessa sequenza, stessa mesh."""
    quota = expr_quota(rng, cfg)
    for i in range(cfg.views):
        V, F, tag = src.view_mesh(ident, rng, quota[i] if quota is not None else bool(rng.random() < cfg.p_expr))
        label = cfg.labels[int(rng.choice(len(cfg.labels), p=cfg.label_p))]
        yield i, V, F, tag, label, int(rng.integers(2 ** 31)), int(getattr(src, "last_frame", -1))


def recipe(cfg, acfg=None) -> dict:
    """I parametri che servono a rigenerare un gruppo dal suo seme (regen.py), scritti in ogni shard."""
    return {"views": cfg.views, "p_expr": cfg.p_expr, "expr_frac": cfg.expr_frac, "labels": list(cfg.labels),
            "label_p": [float(x) for x in cfg.label_p], "v_max": cfg.v_max, "v_work": cfg.v_work,
            "code": getattr(cfg, "code_version", ""), "mm_aug_probs": dict(getattr(cfg, "mm_aug_probs", {}) or {}),
            "mm_aug_config": acfg.to_dict() if acfg is not None else None}


def make_group(src, domain: str, rng: np.random.Generator, cfg, uni: S.Unified, key: str, st: dict,
               tg=None, seed=None) -> dict:
    import views as VW
    t0 = time.perf_counter()
    ident = src.identity(rng)
    P = src.neutral_points(ident)
    s = uni.svec(P)
    if src.kind == "famos":
        key = f"famos/{ident}"
    extra, area_factor = {}, None
    if tg is not None:                 # GT di E12: FR, SR, S_i dalla stessa forma neutra (targets.py)
        t = tg(domain, P)
        extra = {"fr": t["fr"], "sr": t["sr"], "S": float(t["S"])}
        area_factor = cfg.mm_factor[domain]
    st["t_ident"] += time.perf_counter() - t0
    out = []
    t1 = time.perf_counter()
    for i, V, F, tag, label, noise_seed, frame in view_recipe(src, ident, rng, cfg):
        st["t_mesh"] += time.perf_counter() - t1
        try:
            arr, meta = VW.make_view(V, F, label, cfg.k_eig, cfg.evecs_dtype, noise_seed, area_factor=area_factor)
        except Exception as exc:  # noqa: BLE001  (una vista rotta non ferma il produttore)
            st["failures"] += 1
            st["last_failure"] = f"{domain}/{label}: {type(exc).__name__}: {exc}"
            t1 = time.perf_counter()
            continue
        meta["expr"] = tag
        if seed is not None:          # provenienza della vista: indice nella ricetta, seme del rumore, fotogramma
            meta.update(vi=i, noise_seed=noise_seed, frame=frame)
        for ph in ("gen", "ops", "pack"):
            st[f"t_{ph}"] += meta.pop(f"t_{ph}")
        st["views"] += 1
        st["by_label"][label] = st["by_label"].get(label, 0) + 1
        st["verts"] += meta["n"]
        out.append((arr, meta))
        t1 = time.perf_counter()
    st["by_domain"][domain] = st["by_domain"].get(domain, 0) + len(out)
    st["by_origin"] = st.get("by_origin", {})
    st["by_origin"]["pure"] = st["by_origin"].get("pure", 0) + len(out)
    st["expr_views"] = st.get("expr_views", 0) + sum(m["expr"] == "expr" for _, m in out)
    g = {"key": key, "domain": domain, "s": s, "views": out, "origin": "pure", **extra}
    if seed is not None:
        lic, rank = S.inherited_license([domain])
        g["prov"] = {"seed": [int(x) for x in seed], "sources": [domain], "license": lic, "license_rank": rank,
                     "person": ident if src.kind == "famos" else None}
        if src.kind == "mm":
            g["zid"] = np.asarray(ident, dtype=np.float64)
    return g


def worker(wid: int, cfg, srcs: dict, uni: S.Unified, stop_at: float, tg=None, aug=None) -> None:
    import torch
    import views as VW
    torch.set_num_threads(1)
    if cfg.cpus:
        os.sched_setaffinity(0, cpu_list(cfg.cpus))
    VW.install_grad_vec()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, wid]))
    ring = Ring(cfg.ring, int(cfg.ring_gb * 2 ** 30), cfg.evict_every)
    doms = list(cfg.probs)
    p = np.asarray([cfg.probs[d] for d in doms])
    st = {"wid": wid, "pid": os.getpid(), "t_start": time.time(), "views": 0, "groups": 0, "shards": 0, "bytes": 0,
          "verts": 0, "failures": 0, "evicted": 0, "by_domain": {}, "by_label": {}, "t_first_shard": None,
          "views_at_first_shard": 0, **{f"t_{k}": 0.0 for k in ("ident", "mesh", "gen", "ops", "pack", "write")}}
    stats_path = stats_dir(cfg, ring) / f"w{wid:03d}.json"
    last_dump, n = 0.0, 0
    try:
        while not stop_at or time.time() < stop_at:
            groups = []
            for _ in range(cfg.groups_per_shard):
                d = doms[int(rng.choice(len(doms), p=p))]
                seed = (cfg.seed, wid, n) if cfg.provenance else None
                rg = np.random.default_rng(np.random.SeedSequence(list(seed))) if seed else rng
                kind = group_kind(rg, cfg, d)
                if kind == "pure":
                    g = make_group(srcs[d], d, rg, cfg, uni, f"{d}/{cfg.seed}-{wid}-{n}", st, tg, seed)
                else:
                    g = make_aug_group(*aug, d, kind, rg, cfg, uni, f"{d}/{cfg.seed}-{wid}-{n}", st, tg, seed)
                n += 1
                if g["views"]:
                    groups.append(g)
            if not groups:
                continue
            t0 = time.perf_counter()
            _, nb = ring.write(groups, wid, cfg.recipe if cfg.provenance else None)
            st["t_write"] += time.perf_counter() - t0
            st["shards"] += 1
            st["groups"] += len(groups)
            st["bytes"] += nb
            if st["t_first_shard"] is None:
                st["t_first_shard"], st["views_at_first_shard"] = time.time(), st["views"]
            if time.time() - last_dump > cfg.stats_every / 2:
                _dump(stats_path, st)
                last_dump = time.time()
    finally:
        st["t_end"] = time.time()
        _dump(stats_path, st)


def stats_dir(cfg, ring: Ring) -> Path:
    return Path(cfg.stats_dir) if getattr(cfg, "stats_dir", "") else ring.root / "stats"


def _dump(path: Path, st: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    os.replace(tmp, path)


# --- padre ---------------------------------------------------------------------------------------------

def read_stats(ring: Ring, cfg=None) -> list[dict]:
    out = []
    for f in sorted(stats_dir(cfg, ring).glob("w*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    return out


def summarize(ring: Ring, t0: float, cfg) -> dict:
    st = read_stats(ring, cfg)
    now = time.time()
    views = sum(s["views"] for s in st)
    steady = [s for s in st if s.get("t_first_shard")]
    # regime: viste dopo il primo shard di ogni processo, sul tempo dal primo shard alla fine (o ad adesso)
    v_ss = sum(s["views"] - s["views_at_first_shard"] for s in steady)
    t_ss = [(s.get("t_end") or now) - s["t_first_shard"] for s in steady]
    rate_ss = sum((s["views"] - s["views_at_first_shard"]) / max(t, 1e-9) for s, t in zip(steady, t_ss))
    lst = ring.listing()
    phases = {k: sum(s[f"t_{k}"] for s in st) for k in ("ident", "mesh", "gen", "ops", "pack", "write")}
    by_dom, by_lab = {}, {}
    for s in st:
        for k, v in s["by_domain"].items():
            by_dom[k] = by_dom.get(k, 0) + v
        for k, v in s["by_label"].items():
            by_lab[k] = by_lab.get(k, 0) + v
    return {"n_proc": cfg.n_proc, "k_eig": cfg.k_eig, "evecs_dtype": cfg.evecs_dtype, "alpha": cfg.alpha,
            "probs": cfg.probs, "views_per_group": cfg.views, "elapsed_s": now - t0, "views": views,
            "views_per_s_wall": views / max(now - t0, 1e-9), "views_steady": v_ss,
            "views_per_s_steady": rate_ss, "groups": sum(s["groups"] for s in st),
            "shards_written": sum(s["shards"] for s in st), "bytes_written": sum(s["bytes"] for s in st),
            "mean_verts_per_view": sum(s["verts"] for s in st) / max(views, 1),
            "bytes_per_view": sum(s["bytes"] for s in st) / max(views, 1),
            "failures": sum(s["failures"] for s in st),
            "expr_fraction": sum(s.get("expr_views", 0) for s in st) / max(views, 1),
            "views_by_origin": {k: sum(s.get("by_origin", {}).get(k, 0) for s in st)
                                for k in sorted({k for s in st for k in s.get("by_origin", {})})},
            "mm_aug": cfg.mm_aug_probs,
            "canonical_gt": bool(cfg.canonical_gt), "expr_frac": cfg.expr_frac, "p_expr": cfg.p_expr,
            "last_failure": next((s["last_failure"] for s in st if s.get("last_failure")), ""),
            "cpu_s_per_view": {k: v / max(views, 1) for k, v in phases.items()},
            "views_by_domain": by_dom, "views_by_label": by_lab,
            "ring": {"shards": len(lst), "gib": sum(b for _, b, _ in lst) / 2 ** 30,
                     "seq_min": lst[0][0] if lst else None, "seq_max": lst[-1][0] if lst else None}}


def main() -> None:
    import views as VW
    cfg = build_parser().parse_args()
    cfg.mm_aug_probs = check_mm_aug(cfg.mm_aug) if cfg.mm_aug else {}
    cfg.labels = list(VW.LABELS)
    lw = parse_weights(cfg.label_weights, VW.LABELS)
    w = np.asarray([lw.get(k, 0.0) for k in cfg.labels])
    cfg.label_p = w / w.sum()
    domains = S.parse_sources(cfg.sources) if cfg.sources else [d for d in cfg.domains.split(",") if d]
    if cfg.provenance:
        import subprocess
        try:
            git = ["git", "-C", str(S.REPO_ROOT)]
            cfg.code_version = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True,
                                              timeout=30).stdout.strip()
            if subprocess.run(git + ["status", "--porcelain", "--", "v3_work"], capture_output=True, text=True,
                              timeout=60).stdout.strip():
                cfg.code_version += "+modifiche"     # codice non committato: la ricetta vale con QUESTI file
        except (OSError, subprocess.SubprocessError):
            cfg.code_version = ""
    t_load = time.time()
    uni = S.Unified()
    srcs = S.build_sources(domains, uni, v_max=cfg.v_max, v_work=cfg.v_work)
    tg = None
    if cfg.canonical_gt:
        from targets import CanonTargets
        tg = CanonTargets(srcs)
        cfg.mm_factor = {d: tg.mm_factor(d, srcs[d]) for d in domains}
        print(f"[producer] GT di E12 al volo (fr, sr, S): frame {tg.describe()}; mm veri per mm della vista "
              f"{cfg.mm_factor}", flush=True)
    aug = None
    if cfg.mm_aug_probs:
        aug = mm_aug_setup(cfg, domains)
        print(f"[producer] moltiplicatori {cfg.mm_aug_probs} (resto puro) su template {aug[1].templates}, "
              f"espressioni da {aug[1].expr_sources}, validita' {cfg.mm_aug_validate}", flush=True)
    if cfg.provenance:
        cfg.recipe = recipe(cfg, aug[1] if aug else None)
    sizes = {d: srcs[d].n_id for d in domains}
    if cfg.domain_sizes:
        sizes.update({k: int(v) for k, v in parse_weights(cfg.domain_sizes, domains).items()})
    cfg.probs = S.domain_probs(sizes, cfg.alpha)
    print(f"[producer] sorgenti in {time.time() - t_load:.0f}s: " + "; ".join(
        f"{d} n_d={sizes[d]} p={cfg.probs[d]:.3f} {srcs[d].describe()}" for d in domains), flush=True)
    ring = Ring(cfg.ring, int(cfg.ring_gb * 2 ** 30))
    stats_dir(cfg, ring).mkdir(parents=True, exist_ok=True)
    for f in stats_dir(cfg, ring).glob("w*.json"):
        f.unlink()
    t0 = time.time()
    stop_at = t0 + cfg.duration if cfg.duration > 0 else 0.0
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=worker, args=(i, cfg, srcs, uni, stop_at, tg, aug), daemon=True)
             for i in range(cfg.n_proc)]
    for pr in procs:
        pr.start()

    def stop(*_):
        for pr in procs:
            if pr.is_alive():
                pr.terminate()
    signal.signal(signal.SIGTERM, lambda *_: (stop(), sys.exit(0)))
    print(f"[producer] {cfg.n_proc} processi, k_eig={cfg.k_eig} evecs={cfg.evecs_dtype}, anello {ring.root} "
          f"({cfg.ring_gb:g} GiB), alpha={cfg.alpha} p={cfg.probs}, CPU {cfg.cpus or 'tutte'}", flush=True)
    last_v, last_t = 0, t0
    try:
        while any(pr.is_alive() for pr in procs):
            for pr in procs:
                pr.join(timeout=cfg.stats_every / len(procs))
            if time.time() - last_t >= cfg.stats_every:
                s = summarize(ring, t0, cfg)
                now = time.time()
                print(f"[producer] {now - t0:6.0f}s viste {s['views']} ({(s['views'] - last_v) / (now - last_t):.1f}/s "
                      f"nell'intervallo, regime {s['views_per_s_steady']:.1f}/s) gruppi {s['groups']} fallite "
                      f"{s['failures']} | anello {s['ring']['shards']} shard {s['ring']['gib']:.2f} GiB "
                      f"seq {s['ring']['seq_min']}-{s['ring']['seq_max']}", flush=True)
                last_v, last_t = s["views"], now
    finally:
        stop()
        s = summarize(ring, t0, cfg)
        print("[producer] riepilogo " + json.dumps(s), flush=True)
        if cfg.summary:
            Path(cfg.summary).parent.mkdir(parents=True, exist_ok=True)
            Path(cfg.summary).write_text(json.dumps(s, indent=1) + "\n")


if __name__ == "__main__":
    main()
