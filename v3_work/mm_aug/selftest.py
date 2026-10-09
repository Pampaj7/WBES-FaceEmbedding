#!/usr/bin/env python3
"""Prove dei moltiplicatori di dati (``aug.py``, ``transfer.py``).

    srun -p prioritized --gres=NONE -c 8 --mem=48G -t 00:30:00 env AAU_NV= aau/run.sh v3_work/mm_aug/selftest.py

  1. trasporto: lineare; interpolante sui punti della regione (template FLAME, dove i punti sono vertici); il taglio
     netto senza raccordo e' discontinuo, il raccordo no (strain sul bordo <= strain nella regione);
  2. determinismo: stesso seme -> stessi array; seme diverso -> diversi; ``sample_view_spec`` con due generatori
     uguali -> stessa sequenza; ricostruzione dalla provenienza DOPO un giro in JSON -> stessi array, bit per bit;
  3. provenienza: fonti, alpha e beta, coefficienti, licenze; ibrido ICT+GNM ridistribuibile, con FLAME no;
  4. GT: nei trasferimenti d'espressione la neutra (e FR) e' quella dell'identita' pura di A; negli ibridi la neutra
     esatta coincide con quella letta dalla mesh entro 1 mm (topologia decimata di BFM 2019 compresa);
  5. ruoli: un modello dev/test come fonte solleva un errore; nessun modello dev/test caricato;
  6. tipi: con le probabilita' di default escono tutti e quattro, con viste valide secondo le soglie.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from v3_work.mm_aug import aug as AG  # noqa: E402
from v3_work.mm_aug import transfer as TR  # noqa: E402

RESULTS: dict = {}


def check(name: str, ok: bool, info=None) -> None:
    RESULTS[name] = {"ok": bool(ok), **({"info": info} if info is not None else {})}
    print(f"[mm_aug-selftest] {'OK ' if ok else 'NO '} {name}" + (f": {info}" if info is not None else ""), flush=True)


def main() -> None:
    t0 = time.time()
    cfg = AG.AugConfig(templates=AG.TEMPLATES + ("famos",))
    L = AG.get_library(cfg)
    rng = np.random.default_rng(5)

    # 1. trasporto
    t = L.tpl["flame2020"]
    g1, g2 = rng.normal(size=(2, L.uni.n, 3))
    lin = np.abs(t.rt.apply(2.0 * g1 - 0.5 * g2) - (2.0 * t.rt.apply(g1) - 0.5 * t.rt.apply(g2))).max()
    check("trasporto lineare", lin < 1e-9, float(lin))
    # sui vertici FLAME che sono punti della regione il trasporto restituisce g
    vidx, _ = L.uni.map_of("flame2020")
    rv = np.asarray(t.model.region_vertices)
    pos = np.searchsorted(rv, vidx[:, 0])
    on = t.rt.inside[pos]
    err = np.abs(t.rt.apply(g1)[pos[on]] - g1[on]).max()
    check("interpolante sui punti (FLAME)", err < 1e-6 and on.mean() > 0.9, {"max_abs": float(err),
                                                                             "frac_inside": float(on.mean())})
    for A in AG.TEMPLATES:
        tA = L.tpl[A]
        tb = L.tpl["ict" if A != "ict" else "gnm"]
        g = tb.to_mm(tb.src.id_u @ tb.model.sample_identity(rng))
        s, h = TR.edge_strain(tA.rt.apply(g), tA.rt), TR.edge_strain(tA.rt.hard_cut(g), tA.rt)
        check(f"raccordo continuo su {A}", s["seam"]["max"] <= s["region"]["max"] and h["seam"]["max"] > s["seam"]["max"],
              {"seam_max": s["seam"]["max"], "region_max": s["region"]["max"], "hard_cut_seam_max": h["seam"]["max"]})

    # 2. determinismo e ricostruzione
    specs_a = [AG.sample_view_spec(np.random.default_rng(11), cfg, L) for _ in range(2)]
    check("stesso generatore -> stesso campione", all(np.array_equal(specs_a[0][k], specs_a[1][k])
                                                     for k in ("V", "F", "neutral_points")))
    ga, gb = np.random.default_rng(3), np.random.default_rng(3)
    seq_a = [AG.sample_view_spec(ga, cfg, L) for _ in range(30)]
    seq_b = [AG.sample_view_spec(gb, cfg, L) for _ in range(30)]
    check("sequenza deterministica", all(np.array_equal(a["V"], b["V"]) and a["provenance"] == b["provenance"]
                                         for a, b in zip(seq_a, seq_b)))
    check("semi diversi -> campioni diversi", len({a["provenance"]["seed"] for a in seq_a}) == 30
          and not np.array_equal(seq_a[0]["V"], AG.view_spec_from_seed(seq_a[0]["provenance"]["seed"] + 1, cfg, L)["V"]))
    bad = []
    for a in seq_a:
        prov = json.loads(AG.provenance_json(a["provenance"]))
        b = AG.rebuild_view_spec(prov, L)
        if not all(np.array_equal(a[k], b[k]) for k in ("V", "F", "neutral_points")):
            bad.append(a["provenance"]["seed"])
    check("ricostruzione dalla provenienza (JSON), bit per bit", not bad, {"n": len(seq_a), "diversi": bad})

    # 3. provenienza e licenze
    kinds = {a["kind"] for a in seq_a}
    for a in seq_a:
        p = a["provenance"]
        need = {"seed", "kind", "template", "identity", "expression", "sources_used", "redistributable", "license",
                "config", "attempt", "version"}
        if not need <= set(p):
            check("provenienza completa", False, sorted(need - set(p)))
            break
        if p["kind"] == "hybrid" and not {"B", "z_B", "z_A", "alpha", "beta"} <= set(p["identity"]):
            check("provenienza completa", False, "ibrido senza B / z_B / alpha / beta")
            break
    else:
        check("provenienza completa", True, sorted(kinds))
    check("licenza: ICT+GNM ridistribuibile", AG.license_of(["ict", "gnm"])["redistributable"])
    check("licenza: ICT+FLAME non ridistribuibile", not AG.license_of(["ict", "flame2020"])["redistributable"])
    check("licenza: FaMoS come espressione non ridistribuibile",
          not AG.license_of(AG._sources_of({"A": "gnm"}, {"source": "famos"}))["redistributable"])

    # 4. GT
    for A in AG.TEMPLATES:
        p = AG.draw_view(np.random.default_rng([9, AG.TEMPLATES.index(A)]), "expr_transfer", A, cfg, L)
        prov = AG._provenance(0, "expr_transfer", "expr_transfer", 0, p, cfg)
        s = AG.assemble(prov, L)
        pure = AG.assemble(AG._provenance(0, "pure", "pure", 0, {"identity": p["identity"], "expression": None}, cfg), L)
        same = np.array_equal(s["neutral_points"], pure["neutral_points"])
        fr = np.array_equal(AG.gt_targets(s, L)["fr"], AG.gt_targets(pure, L)["fr"])
        moved = float(np.abs(s["V"] - pure["V"]).max())
        check(f"trasferimento d'espressione su {A}: neutra e FR invariate", same and fr and moved > 0,
              {"fonte": p["expression"]["source"], "spostamento_max_vista": moved})
        ph = AG.draw_view(np.random.default_rng([10, AG.TEMPLATES.index(A)]), "hybrid", A, AG.AugConfig(p_expr=0.0), L)
        Vw, P = AG.build_identity(ph["identity"], L)
        t = L.tpl[A]
        d = np.linalg.norm(t.to_mm(P - t.u2w(Vw)), axis=1)
        check(f"ibrido su {A}: GT esatta contro mesh", d.max() < 1.0, {"max_mm": float(d.max()),
                                                                       "median_mm": float(np.median(d))})

    # 5. ruoli
    try:
        AG.AugLibrary(AG.AugConfig(hybrid_sources=("ict", "hifi3d")))
        check("fonte di test rifiutata", False)
    except ValueError as e:
        check("fonte di test rifiutata", True, str(e)[:80])
    loaded = set(L.srcs)
    check("nessun modello dev/test caricato", loaded <= set(AG.EXPR_SOURCES), sorted(loaded))

    # 6. tipi e validita'
    g = np.random.default_rng(21)
    t1 = time.time()
    sp = [AG.sample_view_spec(g, cfg, L) for _ in range(120)]
    from collections import Counter
    cnt = Counter(s["kind"] for s in sp)
    vt = [s["provenance"]["valid"] for s in sp]
    att = [s["provenance"]["attempt"] for s in sp]
    check("tutti i tipi, viste valide", set(cnt) == set(AG.KINDS) and np.mean(vt) > 0.97,
          {"tipi": dict(cnt), "frac_valide": float(np.mean(vt)), "tentativi_medi": float(np.mean(att)),
           "size_mask_tutte": bool(all(s["size_mask"] for s in sp)), "s_per_vista": (time.time() - t1) / len(sp)})
    ok = all(r["ok"] for r in RESULTS.values())
    print(f"[mm_aug-selftest] {'OK' if ok else 'FALLITO'} ({time.time() - t0:.0f}s)")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(RESULTS, indent=1, default=str) + "\n")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
