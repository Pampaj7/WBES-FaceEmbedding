#!/usr/bin/env python3
"""Controlli del critic sulla calibrazione LDA di ``fcal.py`` (configurazione scelta: lam 0,01, k 16). POST HOC e
DESCRITTIVI, dopo i risultati; nessuna scelta cambia.

    v3_work/unified_gt/run.sh aau/famos_calib/fcal_checks.py        (fcal.sbatch, FC_STEP=checks)

1. **Sequenze disgiunte** (leave-sequences-out): per ognuna delle C(10, 5) = 252 divisioni delle 10 sequenze in A e B,
   LDA stimata sulle sole sequenze A della calibrazione, valutata sulle scansioni di test delle sole sequenze B
   (75 scansioni); a parita' di campioni, la stessa LDA stimata sulle sequenze B (sequenze in comune). Accanto d_F non
   cal., ICP mm e NoW metrical (default) sulle stesse 75 scansioni. Premio delle sequenze in comune = comune - disgiunte.
2. **Bootstrap dei soggetti di calibrazione**: 200 repliche dei 73 soggetti con reinserimento (copie = identita'
   distinte), seme 1; AUC e rank-1 di test.
3. **Etichette permutate**: LDA stimata con le identita' di calibrazione permutate (3 permutazioni).
4. **Quasi duplicati**: nello spazio LDA, distanza fra le medie per soggetto; vicino piu' prossimo calibrazione ->
   calibrazione contro test -> calibrazione (un soggetto di test quasi uguale a uno di calibrazione starebbe molto
   sotto la distanza tipica fra persone di calibrazione).

Metriche su un sottoinsieme di scansioni: AUC con pesi 1 (``famos_eval.weighted_auc``) e rank-1 medio
(``ei.retrieval``) con galleria = le altre scansioni del sottoinsieme; nessun bootstrap dei soggetti di test.
Uscite: ``aau/runs/evidence/famos_calib/controlli_critic.json`` e ``controlli_critic_lso.csv``.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fcal  # noqa: E402
import ei  # noqa: E402

LAM, K = 0.01, 16
N_BOOT, BOOT_SEED, N_PERM = 200, 1, 3


def main() -> None:
    import famos_eval as fe
    t0 = time.time()
    it = ei.items("eval")
    n = len(it)
    iu, ju = ei.pairs(n)
    tsub = np.asarray([x for x, _, _ in it])
    tseq = np.asarray([t for _, t, _ in it])
    subj = np.unique(tsub, return_inverse=True)[1]
    cols, _, _ = ei.load_columns("eval")
    ci = fcal.items()
    vc = np.asarray([v for v, _, _ in ci])
    sc = np.asarray([t for _, t, _ in ci])
    y = np.unique(vc, return_inverse=True)[1]
    if set(tsub) & set(vc):
        raise SystemExit("soggetti di test nella calibrazione")
    seqs = sorted(set(sc))
    if seqs != sorted(set(tseq)):
        raise SystemExit("sequenze di calibrazione diverse da quelle di test")

    def met(x, m):
        """(AUC, rank-1) sulle scansioni della maschera m; x = distanze su tutte le coppie."""
        idx = np.flatnonzero(m)
        D = np.zeros((n, n))
        D[iu, ju] = D[ju, iu] = x
        Ds, s = D[np.ix_(idx, idx)], subj[idx]
        a, b = np.triu_indices(len(idx), 1)
        auc = fe.weighted_auc(-Ds[a, b], s[a] == s[b], np.ones(len(a)))
        return float(auc), float(ei.retrieval(Ds, s)[0].mean())

    ALL = np.ones(n, bool)
    out, rows = {}, []
    for m in ei.CKPT:
        with np.load(fcal.CAL / "emb" / f"{m}.npz") as z:
            if list(z["names"].astype(str)) != [p.stem for _, _, p in ci]:
                raise SystemExit(f"{m}: latenti di calibrazione con mesh diverse")
            Uc = np.asarray(z["Z"], np.float64)[:, 1:]
        with np.load(ei.DATA / "eval" / "emb" / f"{m}.npz") as z:
            Ut = np.asarray(z["Z"], np.float64)[:, 1:]

        def lda(U, yy):
            return fcal.pdist_u(Ut, fcal.fit(U, yy, "lda", LAM, K), iu, ju)

        ref = {"LDA": met(lda(Uc, y), ALL), "dF": met(cols[f"{m}|form"], ALL), "ICP": met(cols["icp|mm"], ALL),
               "NoW": met(cols["now_metrical|mm"], ALL)}
        # 1. sequenze disgiunte contro sequenze in comune, a parita' di campioni
        o = []
        for A in itertools.combinations(seqs, 5):
            B = [q for q in seqs if q not in A]
            mA, mB, tB = np.isin(sc, A), np.isin(sc, B), np.isin(tseq, B)
            r = {"modello": m, "A": "+".join(A), "B": "+".join(B)}
            for lab, x in (("LDA_disgiunte", lda(Uc[mA], y[mA])), ("LDA_comuni", lda(Uc[mB], y[mB])),
                           ("dF", cols[f"{m}|form"]), ("ICP", cols["icp|mm"]), ("NoW", cols["now_metrical|mm"])):
                r[f"{lab}_AUC"], r[f"{lab}_rank1"] = met(x, tB)
            o.append(r)
        rows += o
        g = {k: np.asarray([r[k] for r in o]) for k in o[0] if k.endswith(("_AUC", "_rank1"))}
        prem = g["LDA_comuni_AUC"] - g["LDA_disgiunte_AUC"]
        vs_now = g["LDA_disgiunte_AUC"] - g["NoW_AUC"]
        vs_icp = g["LDA_disgiunte_AUC"] - g["ICP_AUC"]
        q = lambda x: {"mediana": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}  # noqa: E731
        lso = {k: q(v) for k, v in g.items()}
        lso.update({"premio_sequenze_comuni_AUC": q(prem),
                    "disgiunte_meno_NoW_AUC": {**q(vs_now), "quota_sopra_0": float((vs_now > 0).mean())},
                    "disgiunte_meno_ICP_AUC": {**q(vs_icp), "quota_sopra_0": float((vs_icp > 0).mean())},
                    "divisioni": len(o)})
        # 2. bootstrap dei soggetti di calibrazione
        rng = np.random.default_rng(BOOT_SEED)
        C = int(y.max()) + 1
        bs = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, C, C)
            sel = np.concatenate([np.flatnonzero(y == c) for c in pick])
            yy = np.concatenate([np.full(int((y == c).sum()), k) for k, c in enumerate(pick)])
            bs.append(met(lda(Uc[sel], yy), ALL))
        bs = np.asarray(bs)
        boot = {"AUC_2.5_50_97.5": np.percentile(bs[:, 0], [2.5, 50, 97.5]).tolist(),
                "rank1_2.5_50_97.5": np.percentile(bs[:, 1], [2.5, 50, 97.5]).tolist(),
                "quota_AUC_<=_NoW": float((bs[:, 0] <= ref["NoW"][0]).mean()), "repliche": N_BOOT, "seme": BOOT_SEED}
        # 3. etichette permutate (dopo il bootstrap, stesso generatore)
        perm = [met(lda(Uc, rng.permutation(y)), ALL) for _ in range(N_PERM)]
        # 4. quasi duplicati nello spazio LDA
        P = fcal.fit(Uc, y, "lda", LAM, K)
        Mc = np.stack([(Uc @ P)[y == c].mean(0) for c in range(C)])
        Mt = np.stack([(Ut @ P)[subj == s].mean(0) for s in range(subj.max() + 1)])
        dcc = np.linalg.norm(Mc[:, None] - Mc[None], axis=2)
        np.fill_diagonal(dcc, np.inf)
        dtc = np.linalg.norm(Mt[:, None] - Mc[None], axis=2)
        nn = {"calib_calib_min": float(dcc.min(1).min()), "calib_calib_mediana": float(np.median(dcc.min(1))),
              "test_calib_min": float(dtc.min(1).min()), "test_calib_mediana": float(np.median(dtc.min(1)))}
        out[m] = {"riferimento_tutte_le_150": ref, "sequenze_disgiunte": lso, "bootstrap_calibrazione": boot,
                  "etichette_permutate": perm, "vicino_piu_prossimo": nn}
        print(f"[fcal-checks] {m}: {json.dumps(out[m])}", flush=True)
    with open(fcal.EVID / "controlli_critic_lso.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ei.atomic_json(fcal.EVID / "controlli_critic.json", {**out, "lam": LAM, "k": K, "seconds": time.time() - t0,
                                                         **ei.stamp()})
    print(f"[fcal-checks] fatto in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
