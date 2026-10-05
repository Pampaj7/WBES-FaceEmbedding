#!/usr/bin/env python3
"""La domanda di WS3b: la classifica dei metodi cambia a seconda che si allinei o no?

    aau/run.sh aau/recon/ws3b_analysis.py            # su un nodo di calcolo
    python3 aau/recon/ws3b_analysis.py --n-bootstrap 1000

Legge i csv di ``ws3b_geometric.py`` e ``ws3b_latent.py`` e scrive
``aau/runs/multiface_ws3b/summary.md`` piu' le tre tabelle in csv.  Tre blocchi.

(a) Classifica dei tre metodi per ciascun criterio
--------------------------------------------------
Ai criteri storici se ne aggiungono due, che non rifanno la classifica ma la mettono alla
prova: ``chamfer_icp_mm`` e' la stessa Chamfer in millimetri veri (scala dall'ICP invece
che maxabs per mesh) e ``chamfer_gtclip_mm`` la calcola sulla ricostruzione allineata e
ritagliata alla patch della GT, cioe' su un supporto uguale per costruzione.  Entrambi
stanno in ``RANKED``, in coda, cosi' il Kendall tau e lo split-half li coprono: servono a
vedere se l'ordine dei tre metodi dipende dalla normalizzazione e dal supporto, non a
sostituire i criteri esistenti.  Per la stessa ragione, sempre in coda, c'e'
``latent_v1_gtclip``: il latent v1 con la ricostruzione ristretta alla STESSA patch di
``chamfer_gtclip_mm`` (operatori ricalcolati), cioe' la metrica appresa a parita' di supporto.
Dopo di lui, ancora in coda, ``latent_joint`` e ``latent_joint_gtclip``: le stesse due
distanze col modello congiunto BFM+ICT di WS2 (operatori ad area unitaria, quelli del suo
training), per vedere se il disaccordo del v1 coi criteri allineati e' del modello o della
metrica appresa in se'.
Il punteggio di un soggetto e' la MEDIANA delle sue ricostruzioni (13 soggetti x 4
segmenti x 20 frame x 3 camere), il punteggio globale e' la MEDIA sui 13 soggetti: cosi'
ogni soggetto pesa uguale, e i due soggetti con qualche frame in meno non contano meno
degli altri.  Il CI e' bootstrap sui SOGGETTI (1000 repliche): le ricostruzioni non sono
indipendenti, 240 vengono dalla stessa faccia, e un bootstrap per ricostruzione darebbe
intervalli ottimisti di un fattore grosso.  Accanto al CI c'e' ``p_first``, la frazione di
repliche in cui il metodo e' primo: e' la stabilita' della classifica, che e' esattamente
cio' che il cantiere vuole misurare.

(b) Kendall tau fra le classifiche per soggetto
-----------------------------------------------
Per ogni soggetto ciascun criterio ordina i tre metodi.  Il tau-b fra due criteri, su tre
elementi, vale 1, 1/3, -1/3 o -1: si riporta la media sui 13 soggetti con CI bootstrap
sugli stessi soggetti, il tau fra le due classifiche GLOBALI, e la frazione di soggetti in
cui i due criteri danno lo stesso identico ordine.  Un tau vicino a 1 vuol dire che
allineare o no non cambia chi vince; e' l'ipotesi nulla del cantiere.

Un tau da solo non si legge: servono i due metri accanto.

* **Sotto l'ipotesi nulla** (due classifiche indipendenti) il tau medio ha valore atteso 0
  e deviazione standard ``sqrt(Var[tau]/n_soggetti)``, con ``Var[tau] = 0.40741`` esatta
  sulle 6 permutazioni di tre elementi: con 13 soggetti fa **0.177**.  La frazione di
  soggetti con ordine identico vale invece **1/6 = 16.7%**.  Un tau di 0.33 e' quindi meno
  di due deviazioni dal caso.
* **Il tetto**: nessun tau fra criteri puo' superare l'affidabilita' interna dei criteri
  stessi.  ``split_half_rows`` la misura come fa il critic: si spezzano a caso in due meta'
  le immagini di ogni soggetto, si ricalcola la classifica dei tre metodi su ciascuna meta'
  e si prende il tau fra le due, mediato sui soggetti e su 200 ripetizioni.  Se un criterio
  ha split-half 0.95 e il tau con un altro criterio e' 0.33, quel 0.33 e' un disaccordo
  vero, non rumore di misura.

(c) AUC stesso/diverso soggetto sulle sole ricostruzioni
---------------------------------------------------------
Nessuna verita' a terra: si confrontano ricostruzioni fra loro, sulle stesse quattro classi
di coppie di WS3a (a stesso soggetto stessa espressione, b stesso soggetto espressione
diversa, c soggetti diversi stessa espressione, d tutto diverso).  Se la ricostruzione
conserva l'identita', due ricostruzioni dello stesso soggetto devono risultare piu' vicine.
L'AUC pesata e il suo bootstrap sono importati da ``aau/multiface/ws3a_analysis.py``, non
riscritti: le due tabelle devono essere leggibili una accanto all'altra.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ws3b_common as common  # noqa: E402

# Le funzioni di AUC pesata e di bootstrap per soggetto sono quelle di WS3a.
sys.path.insert(0, str(common.AAU_DIR / "multiface"))
from ws3a_analysis import COMPARISONS, bootstrap_auc  # noqa: E402

# I criteri, in ordine, con l'etichetta che finisce nel summary.  Per tutti "piccolo =
# meglio", quindi la classifica e' sempre in ordine crescente.
CRITERIA = (
    ("sim_icp_p2s_median", "similarita' da ICP su tutta la superficie, punto-superficie, mm"),
    ("sim_icp_p2s_mean", "come sopra, media invece che mediana, mm"),
    ("chamfer_raw", "Chamfer grezza (sola normalizzazione maxabs)"),
    ("chamfer_icp_mm", "Chamfer in mm, scala dalla similarita' ICP invece che maxabs per mesh"),
    ("chamfer_gtclip_mm", "Chamfer in mm sulla recon allineata e RITAGLIATA alla patch GT "
                          "(supporto uguale per costruzione)"),
    ("latent_v1", "distanza latente, checkpoint v1"),
    ("latent_v1_gtclip", "distanza latente v1 con la recon ritagliata alla patch GT di "
                         "chamfer_gtclip_mm (operatori ricalcolati)"),
    ("latent_joint", "distanza latente, modello congiunto BFM+ICT (WS2, operatori ad area "
                     "unitaria)"),
    ("latent_joint_gtclip", "distanza latente del congiunto con la recon ritagliata alla "
                            "patch GT di chamfer_gtclip_mm"),
    ("now_median", "DIAGNOSTICO, fuori scope: NoW da 7 landmark (vedi nota)"),
    ("now_mean", "DIAGNOSTICO, fuori scope: NoW da 7 landmark, media"),
)

# Il seme del bootstrap di un criterio e' il suo posto in QUESTA tupla, non in CRITERIA.
# Le due cose coincidevano finche' i criteri non sono cambiati; da quando ce ne sono di
# nuovi non possono piu', perche' inserirli in mezzo a CRITERIA sposterebbe il seme di
# tutti quelli dopo e cambierebbe i CI di righe che nessuno ha ricalcolato.  I criteri
# nuovi si aggiungono in CODA qui, e in CRITERIA vanno dove si leggono meglio.
SEED_ORDER = ("sim_icp_p2s_median", "sim_icp_p2s_mean", "chamfer_raw", "latent_v1",
              "now_median", "now_mean", "chamfer_icp_mm", "chamfer_gtclip_mm",
              "latent_v1_gtclip", "latent_joint", "latent_joint_gtclip")

# Quali criteri entrano nel confronto fra classifiche: uno per famiglia di allineamento,
# senza le varianti *_mean, che sono lo stesso allineamento letto con un'altra statistica.
#
# ``now_median`` NON e' fra questi.  Il criterio e' implementato e la sua colonna sta nei
# csv, ma il protocollo NoW ha bisogno di 7 landmark sulla scansione, e Multiface non li
# distribuisce: quelli stimati da ``ws3b_landmarks.py`` hanno 5-7 mm di dispersione fra
# campioni e 8-10 mm di scarto fra i tre metodi che li trasferiscono, contro un segnale
# (l'errore di ricostruzione) di 1.2-1.6 mm.  Misurato: su 20 elementi now_median vale
# 8.77 mm per 3DDFA_V2 e 2.4-2.6 mm per gli altri due, cioe' l'errore dei landmark domina e
# per un metodo fa saltare del tutto la stima della similarita'.  Metterlo in classifica
# vorrebbe dire pubblicare il rumore dell'annotazione: resta come diagnostico.
#
# I due controlli della Chamfer, e dopo di loro il latent a patch uguale, stanno in CODA per
# la stessa ragione di SEED_ORDER: il seme dello split-half e' il posto del criterio in
# questa tupla, e inserirli in mezzo cambierebbe le righe dei criteri dopo.
RANKED = ("sim_icp_p2s_median", "chamfer_raw", "latent_v1",
          "chamfer_icp_mm", "chamfer_gtclip_mm", "latent_v1_gtclip",
          "latent_joint", "latent_joint_gtclip")
DIAGNOSTIC = ("now_median", "now_mean")

PAIR_METRICS = ("chamfer_raw", "latent_v1", "latent_joint")

# I modelli di cui ws3b_latent.py ha scritto i csv (``--metric``): latent_v1 coi nomi
# storici, gli altri in gt_<metrica>_<metodo>.csv.
LATENT_METRICS = ("latent_v1", "latent_joint")

# Varianza esatta del tau-b fra due classifiche indipendenti di TRE elementi: sulle 6
# permutazioni tau vale 1, 1/3, 1/3, -1/3, -1/3, -1, quindi E[tau] = 0 e E[tau^2] = 11/27.
TAU_NULL_VARIANCE = 11.0 / 27.0
# Probabilita' che due classifiche indipendenti di tre elementi coincidano: 1 permutazione
# su 6.
IDENTICAL_ORDER_NULL = 1.0 / 6.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--methods", type=str, default=",".join(common.METHODS))
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-split-half", type=int, default=200,
                   help="ripetizioni dello split-half sulle immagini di ogni soggetto")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ci", type=float, default=95.0, help="ampiezza del CI in percento")
    return p.parse_args()


# --------------------------------------------------------------------- lettura

def load_scores(methods: list[str], out_root: Path) -> dict[str, dict[str, np.ndarray]]:
    """{criterio: {metodo: {soggetto: {nome: valore}}}} piu' il conteggio delle righe.

    Il valore resta indicizzato per NOME della ricostruzione e non solo per soggetto:
    ``split_half_rows`` ha bisogno di spezzare in due le immagini dello stesso soggetto.

    I csv (geometrico, latente, latente sulla patch GT) si uniscono sul nome della
    ricostruzione; se un latente non c'e' ancora, il criterio resta assente e il resto del summary si fa lo
    stesso.
    """
    per_subject: dict[str, dict[str, dict[str, list[float]]]] = {}
    n_rows: dict[str, int] = {}
    for method in methods:
        merged: dict[str, dict] = {}
        paths = [common.gt_csv_path(method, out_root)]
        for metric in LATENT_METRICS:
            paths += [common.gt_latent_csv_path(method, out_root, metric),
                      common.gt_latent_gtclip_csv_path(method, out_root, metric)]
        for path in paths:
            if not path.exists():
                print(f"[ws3b-an] assente, salto: {path.name}")
                continue
            for row in common.read_rows(path):
                merged.setdefault(row["name"], {"subject": row["subject"]}).update(row)
        n_rows[method] = len(merged)
        for row in merged.values():
            for criterion, _ in CRITERIA:
                if criterion not in row or row[criterion] == "":
                    continue
                value = float(row[criterion])
                if math.isfinite(value):
                    per_subject.setdefault(criterion, {}).setdefault(method, {}) \
                        .setdefault(row["subject"], {})[row["name"]] = value
    return per_subject, n_rows


def subject_matrix(block: dict[str, dict[str, dict[str, float]]], methods: list[str]):
    """(soggetti, matrice metodi x soggetti dei punteggi mediani)."""
    subjects = sorted(set.intersection(*(set(block[m]) for m in methods if m in block))) \
        if all(m in block for m in methods) else []
    if not subjects:
        return [], np.zeros((len(methods), 0))
    M = np.asarray([[float(np.median(list(block[m][s].values()))) for s in subjects]
                    for m in methods])
    return subjects, M


def support_stats(methods: list[str], out_root: Path) -> dict[str, dict[str, float]]:
    """Mediane delle colonne di CONTROLLO dei csv geometrici, per metodo.

    Non sono criteri e non entrano in classifica: servono al paragrafo dei limiti, che
    altrimenti citerebbe numeri a memoria.
    """
    columns = ("area_ratio", "crop_radius_mm", "eye_span_mm", "recon_inside_gt_patch")
    out: dict[str, dict[str, float]] = {}
    for method in methods:
        path = common.gt_csv_path(method, out_root)
        if not path.exists():
            continue
        rows = common.read_rows(path)
        block = {}
        for column in columns:
            values = [float(r[column]) for r in rows
                      if r.get(column, "") != "" and math.isfinite(float(r[column]))]
            if values:
                block[column] = float(np.median(values))
        if block:
            out[method] = block
    return out


def landmark_disagreement_mm(out_root: Path) -> tuple[float, float]:
    """Disaccordo fra i tre metodi sui 7 landmark della GT: (mediana, peggiore), in mm."""
    import json

    path = common.landmarks_path(out_root)
    if not path.is_file():
        return math.nan
    quality = json.loads(path.read_text()).get("quality", {})
    values = [v["method_disagreement_max_mm"] for v in quality.values()
              if "method_disagreement_max_mm" in v]
    if not values:
        return math.nan, math.nan
    return float(np.median(values)), float(np.max(values))


# ------------------------------------------------------- (a) classifica e CI

def ranking_rows(criterion: str, label: str, methods: list[str], subjects: list[str],
                 M: np.ndarray, offset: int, args) -> list[dict]:
    """Punteggio globale, CI bootstrap sui soggetti e probabilita' di essere primo.

    Il seme dipende dall'indice del criterio e non dal suo nome: ``hash()`` di una stringa
    e' randomizzato per processo, e i CI non sarebbero riproducibili.
    """
    rng = np.random.default_rng(args.seed + 10_007 * offset)
    n = len(subjects)
    point = M.mean(axis=1)
    replicas = np.empty((args.n_bootstrap, len(methods)))
    for r in range(args.n_bootstrap):
        pick = rng.integers(0, n, size=n)
        replicas[r] = M[:, pick].mean(axis=1)
    tail = (100.0 - args.ci) / 2.0
    low, high = np.percentile(replicas, [tail, 100.0 - tail], axis=0)
    p_first = (np.argmin(replicas, axis=1)[:, None] == np.arange(len(methods))[None, :]).mean(axis=0)
    order = np.argsort(point)
    rank = np.empty(len(methods), dtype=int)
    rank[order] = np.arange(1, len(methods) + 1)
    return [{
        "criterion": criterion, "label": label, "method": methods[i], "rank": int(rank[i]),
        "score": float(point[i]), "ci_low": float(low[i]), "ci_high": float(high[i]),
        "p_first": float(p_first[i]), "n_subjects": n, "n_bootstrap": args.n_bootstrap,
    } for i in range(len(methods))]


# --------------------------------------------------------- (b) Kendall tau

def kendall_rows(scores: dict, methods: list[str], args) -> list[dict]:
    """tau-b fra le classifiche per soggetto di due criteri, piu' il tau fra le globali."""
    out = []
    available = [c for c in RANKED if c in scores]
    for i, crit_a in enumerate(available):
        for crit_b in available[i + 1:]:
            subjects_a, Ma = subject_matrix(scores[crit_a], methods)
            subjects_b, Mb = subject_matrix(scores[crit_b], methods)
            subjects = sorted(set(subjects_a) & set(subjects_b))
            if not subjects:
                continue
            ia = [subjects_a.index(s) for s in subjects]
            ib = [subjects_b.index(s) for s in subjects]
            taus = np.asarray([kendalltau(Ma[:, a], Mb[:, b]).statistic
                               for a, b in zip(ia, ib)], dtype=np.float64)
            same = float(np.mean([np.array_equal(np.argsort(Ma[:, a]), np.argsort(Mb[:, b]))
                                  for a, b in zip(ia, ib)]))
            rng = np.random.default_rng(args.seed + 7)
            reps = np.asarray([taus[rng.integers(0, len(taus), size=len(taus))].mean()
                               for _ in range(args.n_bootstrap)])
            tail = (100.0 - args.ci) / 2.0
            low, high = np.percentile(reps, [tail, 100.0 - tail])
            global_tau = kendalltau(Ma[:, ia].mean(axis=1), Mb[:, ib].mean(axis=1)).statistic
            out.append({
                "criterion_a": crit_a, "criterion_b": crit_b,
                "mean_tau_per_subject": float(taus.mean()),
                "ci_low": float(low), "ci_high": float(high),
                "tau_null_mean": 0.0,
                "tau_null_sd": float(np.sqrt(TAU_NULL_VARIANCE / len(subjects))),
                "tau_z_vs_null": float(taus.mean()
                                       / np.sqrt(TAU_NULL_VARIANCE / len(subjects))),
                "tau_global_ranking": float(global_tau),
                "share_identical_ranking": same,
                "share_identical_null": IDENTICAL_ORDER_NULL,
                "n_subjects": len(subjects), "n_bootstrap": args.n_bootstrap,
            })
    return out


# ------------------------------------------- (b bis) affidabilita' interna split-half

def split_half_rows(scores: dict, methods: list[str], args) -> list[dict]:
    """Tetto di affidabilita' di ogni criterio, con lo stesso split-half del critic.

    Le immagini di un soggetto si spezzano a caso in due meta', si ricalcola la classifica
    dei tre metodi su ciascuna e si prende il tau fra le due, mediato sui soggetti; il tutto
    ripetuto ``--n-split-half`` volte.  Nessun tau FRA criteri puo' onestamente superare
    questo numero: se lo supera, o le due meta' non sono indipendenti o il conto e' sbagliato.
    """
    out = []
    for offset, criterion in enumerate(RANKED):
        if criterion not in scores:
            continue
        block = scores[criterion]
        if not all(m in block for m in methods):
            continue
        subjects = sorted(set.intersection(*(set(block[m]) for m in methods)))
        rng = np.random.default_rng(args.seed + 101 * offset)
        replicas = []
        for _ in range(args.n_split_half):
            taus = []
            for subject in subjects:
                names = sorted(set.intersection(*(set(block[m][subject]) for m in methods)))
                if len(names) < 2:
                    continue
                order = rng.permutation(len(names))
                halves = [[names[i] for i in order[: len(names) // 2]],
                          [names[i] for i in order[len(names) // 2:]]]
                scores_half = [[float(np.median([block[m][subject][n] for n in half]))
                                for m in methods] for half in halves]
                taus.append(kendalltau(scores_half[0], scores_half[1]).statistic)
            if taus:
                replicas.append(float(np.mean(taus)))
        if not replicas:
            continue
        replicas = np.asarray(replicas)
        out.append({
            "criterion": criterion,
            "split_half_tau": float(replicas.mean()),
            "ci_low": float(np.percentile(replicas, 2.5)),
            "ci_high": float(np.percentile(replicas, 97.5)),
            "n_subjects": len(subjects), "n_repeats": len(replicas),
        })
    return out


# ------------------------------------------------------------------ (c) AUC

def auc_rows(methods: list[str], args) -> list[dict]:
    """AUC stesso/diverso soggetto sulle coppie di ricostruzioni, per metrica e metodo."""
    out = []
    for metric in PAIR_METRICS:
        for method in methods:
            path = common.pair_csv_path(metric, method, args.out_root)
            if not path.exists():
                print(f"[ws3b-an] assente, salto: {path.name}")
                continue
            rows = common.read_rows(path)
            values = np.asarray([float(r["distance"]) for r in rows])
            classes = np.asarray([r["pair_class"] for r in rows])
            subject_a = np.asarray([r["subject_a"] for r in rows])
            subject_b = np.asarray([r["subject_b"] for r in rows])
            finite = np.isfinite(values)
            subjects = sorted(set(subject_a) | set(subject_b))
            to_idx = {s: i for i, s in enumerate(subjects)}
            sa = np.asarray([to_idx[s] for s in subject_a], dtype=np.int64)
            sb = np.asarray([to_idx[s] for s in subject_b], dtype=np.int64)
            for offset, (comparison, (positive, negative)) in enumerate(COMPARISONS.items()):
                mask = finite & np.isin(classes, positive + negative)
                is_positive = np.isin(classes[mask], positive)
                # Stesso seme per confronto di WS3a: i soggetti pescati sono gli stessi in
                # tutte le righe, quindi i CI sono appaiati fra metriche e metodi.
                rng = np.random.default_rng(args.seed + 10_007 * offset)
                auc, low, high = bootstrap_auc(
                    values[mask], is_positive, sa[mask], sb[mask], n_subjects=len(subjects),
                    n_bootstrap=args.n_bootstrap, ci=args.ci, rng=rng)
                out.append({
                    "metric": metric, "method": method, "comparison": comparison,
                    "auc": auc, "ci_low": low, "ci_high": high,
                    "n_positive": int(is_positive.sum()), "n_negative": int((~is_positive).sum()),
                    "n_subjects": len(subjects), "n_bootstrap": args.n_bootstrap,
                })
            print(f"[ws3b-an] {path.name}: fatto", flush=True)
    return out


# ---------------------------------------------------------------- scrittura

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})


def write_limits(fh, methods, ranking, support, disagreement_mm) -> None:
    """I limiti del cantiere, con i numeri della corsa accanto.

    Sta in fondo e non in nota perche' ognuna di queste quattro righe puo' cambiare la
    lettura della classifica, e nessuna delle quattro e' risolta.
    """
    fh.write("## Limiti di questo confronto\n\n")

    ratios = {m: support.get(m, {}).get("area_ratio", math.nan) for m in methods}
    fh.write("**1. I supporti non coincidono.** Il rapporto fra l'area della patch "
             "ricostruita e quella della patch GT vale "
             + ", ".join(f"{m} **{ratios[m]:.3f}**" for m in methods if math.isfinite(ratios[m]))
             + ". Il raggio del ritaglio e' ormai giusto (mediana "
             + ", ".join(f"{support[m]['crop_radius_mm']:.2f} mm" for m in methods
                         if "crop_radius_mm" in support.get(m, {}))
             + " contro i 95.0 della GT), quindi la differenza residua e' forma, non "
               "supporto: un metodo mette piu' superficie dentro la stessa sfera. "
               "`chamfer_raw`, che guarda il supporto attraverso la maxabs, paga quella "
               "differenza; la riga `chamfer_gtclip_mm` e' il controllo che la toglie, "
               "perche' ritaglia la ricostruzione allineata con la stessa sfera della GT.\n\n")

    fh.write("**2. Il criterio allineato e' unidirezionale e su ricostruzione NON "
             "ritagliata.** `sim_icp_p2s_*` misura la distanza dai vertici della GT "
             "ritagliata alla superficie ricostruita INTERA, in un verso solo. E' la "
             "convenzione di NoW (superficie in piu' non disturba), ma vuol dire che un "
             "metodo che ricostruisce piu' faccia del necessario non viene mai penalizzato, "
             "e che meta' dell'informazione -- quanto della ricostruzione non ha un "
             "corrispondente nella GT -- non entra nel numero.\n\n")

    median_mm, worst_mm = disagreement_mm
    fh.write(f"**3. I 7 landmark della GT sono stimati, non misurati.** Multiface non "
             f"distribuisce landmark: quelli usati vengono trasferiti dai 68 iBUG dei tre "
             f"metodi con l'ICP inverso, e il disaccordo fra i tre metodi sullo stesso punto "
             f"e' di **{median_mm:.1f} mm** in mediana e **{worst_mm:.1f} mm** sul punto "
             f"peggiore, contro un errore di ricostruzione di 1.2-1.6 mm. Da li' viene anche "
             f"l'ex-ex che fissa il raggio del ritaglio. E' il motivo per cui `now_*` resta "
             f"diagnostico, e una ragione in piu' per non fidarsi del terzo decimale di "
             f"nessuna riga.\n\n")

    block = sorted((r for r in ranking if r["criterion"] == "sim_icp_p2s_median"),
                   key=lambda r: r["rank"])
    if len(block) >= 2:
        fh.write(f"**4. Primo e secondo non sono distinguibili sul criterio allineato.** "
                 f"Su `sim_icp_p2s_median` "
                 + " e ".join(f"{r['method']} ha p_first {r['p_first']:.2f}" for r in block[:2])
                 + f": il bootstrap sui {block[0]['n_subjects']} soggetti mette "
                   f"{block[0]['method']} primo in poco piu' della meta' delle repliche. "
                   f"La distanza fra il primo e il secondo ("
                 + f"{block[0]['score']:.3f} contro {block[1]['score']:.3f}"
                 + ") e' dentro i CI di tutti e due. La differenza che si puo' sostenere e' "
                   "fra i primi due e il terzo, non fra il primo e il secondo.\n\n")


def write_markdown(path: Path, methods, n_rows, ranking, kendall, split_half, auc,
                   per_subject, scores, args, support=None,
                   disagreement_mm=(math.nan, math.nan)):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# WS3b Multiface: la classifica dei metodi di ricostruzione cambia se si allinea?\n\n")
        fh.write(f"Tre metodi a pesi pubblici (3DDFA_V2, SynergyNet, PRNet) su tutte le immagini "
                 f"frontali di Multiface con mesh tracciata: "
                 f"{', '.join(f'{m} {n_rows.get(m, 0)}' for m in methods)} ricostruzioni.\n")
        fh.write(f"CI 95% bootstrap su {args.n_bootstrap} repliche, ricampionando i "
                 f"{max((r['n_subjects'] for r in ranking), default=0)} soggetti.\n\n")

        fh.write("## (a) Classifica per criterio\n\n")
        fh.write("Punteggio = media sui soggetti della mediana delle loro ricostruzioni; "
                 "piccolo = meglio. `p_first` = frazione di repliche bootstrap in cui il "
                 "metodo e' primo.\n\n")
        for criterion, label in CRITERIA:
            block = [r for r in ranking if r["criterion"] == criterion]
            if not block:
                continue
            fh.write(f"**{criterion}** — {label}\n\n")
            fh.write("| rango | metodo | punteggio [CI 95%] | p_first |\n|---|---|---|---|\n")
            for r in sorted(block, key=lambda r: r["rank"]):
                fh.write(f"| {r['rank']} | {r['method']} | {r['score']:.4g} "
                         f"[{r['ci_low']:.4g}, {r['ci_high']:.4g}] | {r['p_first']:.2f} |\n")
            fh.write("\n")

        fh.write("> **`now_median` / `now_mean` sono diagnostici, non risultati.** Il protocollo "
                 "NoW vero (similarita' da 7 landmark, nessun ICP) e' implementato in "
                 "`ws3b_geometric.py`, ma ha bisogno di 7 landmark sulla scansione e Multiface "
                 "non li distribuisce. Quelli stimati da `ws3b_landmarks.py` trasferendoli dai "
                 "68 iBUG dei tre metodi hanno 5-7 mm di dispersione fra campioni e 8-10 mm di "
                 "scarto fra metodi, contro un errore di ricostruzione di 1.2-1.6 mm: l'errore "
                 "dell'annotazione e' 4 volte il segnale. Per questo i due criteri non entrano "
                 "ne' nel Kendall tau ne' nella classifica primaria, e la colonna resta nei csv "
                 "solo come traccia. Il criterio allineato di riferimento e' "
                 "`sim_icp_p2s_median`.\n\n")

        fh.write("### Classifica per soggetto\n\n")
        fh.write("Ordine dei tre metodi per ciascun soggetto e criterio (dal migliore).\n\n")
        header = " | ".join(c for c in RANKED if c in scores)
        fh.write(f"| soggetto | {header} |\n|---|" + "---|" * len(header.split(" | ")) + "\n")
        for subject in per_subject["subjects"]:
            cells = []
            for criterion in RANKED:
                order = per_subject.get((criterion, subject))
                cells.append(" > ".join(order) if order else "--")
            fh.write(f"| {subject} | " + " | ".join(cells) + " |\n")
        fh.write("\n")

        fh.write("## (b) Accordo fra le classifiche: Kendall tau\n\n")
        fh.write("tau-b fra le classifiche dei tre metodi, calcolato dentro ogni soggetto e "
                 "mediato sui soggetti. Su tre elementi tau vale 1, 1/3, -1/3 o -1.\n\n")
        null_sd = kendall[0]["tau_null_sd"] if kendall else float("nan")
        fh.write(f"Le due colonne \"caso\" sono il metro: sotto l'ipotesi nulla di classifiche "
                 f"indipendenti il tau medio ha valore atteso 0 e deviazione standard "
                 f"**{null_sd:.3f}** su {kendall[0]['n_subjects'] if kendall else 0} soggetti "
                 f"(varianza esatta 11/27 sulle 6 permutazioni di tre elementi), e la frazione "
                 f"di soggetti con ordine identico vale **{IDENTICAL_ORDER_NULL:.1%}** "
                 f"(1 permutazione su 6).\n\n")
        fh.write("| criteri | tau medio per soggetto [CI 95%] | caso: 0 +- sd | tau/sd | "
                 "tau fra le classifiche globali | soggetti con ordine identico "
                 "| caso |\n|---|---|---|---|---|---|---|\n")
        for r in kendall:
            fh.write(f"| {r['criterion_a']} vs {r['criterion_b']} | "
                     f"{r['mean_tau_per_subject']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | "
                     f"0 +- {r['tau_null_sd']:.3f} | {r['tau_z_vs_null']:.2f} | "
                     f"{r['tau_global_ranking']:.3f} | {r['share_identical_ranking']:.0%} | "
                     f"{r['share_identical_null']:.1%} |\n")
        fh.write("\n")

        if split_half:
            fh.write("### Tetto di affidabilita': split-half dentro ogni criterio\n\n")
            fh.write(f"Le immagini di ogni soggetto spezzate a caso in due meta', classifica "
                     f"ricalcolata su ciascuna, tau fra le due, mediato sui soggetti e su "
                     f"{args.n_split_half} ripetizioni. Nessun tau FRA criteri della tabella "
                     f"qui sopra puo' superare questi valori.\n\n")
            fh.write("| criterio | tau split-half [2.5%, 97.5%] |\n|---|---|\n")
            for r in split_half:
                fh.write(f"| {r['criterion']} | {r['split_half_tau']:.3f} "
                         f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] |\n")
            fh.write("\n")

        fh.write("## (c) La ricostruzione conserva l'identita'?\n\n")
        fh.write("AUC = P(distanza fra ricostruzioni dello stesso soggetto < distanza fra "
                 "ricostruzioni di soggetti diversi), 0.5 = caso. Nessuna verita' a terra "
                 "entra nel conto. Classi: (a) stesso soggetto stessa espressione, (b) stesso "
                 "soggetto espressione diversa, (c) soggetti diversi stessa espressione, "
                 "(d) tutto diverso.\n\n")
        columns = list(COMPARISONS)
        fh.write("| metrica | metodo | " + " | ".join(c.replace("auc_", "") for c in columns)
                 + " |\n|---|---|" + "---|" * len(columns) + "\n")
        by_key = {(r["metric"], r["method"], r["comparison"]): r for r in auc}
        for metric in PAIR_METRICS:
            for method in methods:
                cells = []
                for comparison in columns:
                    r = by_key.get((metric, method, comparison))
                    cells.append("--" if r is None
                                 else f"{r['auc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
                if all(c == "--" for c in cells):
                    continue
                fh.write(f"| {metric} | {method} | " + " | ".join(cells) + " |\n")
        fh.write("\n")

        if support is not None:
            write_limits(fh, methods, ranking, support, disagreement_mm)


def main() -> None:
    args = parse_args()
    args.out_root = args.out_root.resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    scores, n_rows = load_scores(methods, args.out_root)
    if not scores:
        raise SystemExit(f"nessun csv di criteri sotto {args.out_root}")

    ranking: list[dict] = []
    per_subject: dict = {"subjects": []}
    for criterion, label in CRITERIA:
        if criterion not in scores:
            continue
        subjects, M = subject_matrix(scores[criterion], methods)
        if not subjects:
            print(f"[ws3b-an] {criterion}: non tutti i metodi hanno righe, salto")
            continue
        ranking.extend(ranking_rows(criterion, label, methods, subjects, M,
                                    SEED_ORDER.index(criterion), args))
        per_subject["subjects"] = subjects
        for j, subject in enumerate(subjects):
            per_subject[(criterion, subject)] = [methods[i] for i in np.argsort(M[:, j])]

    kendall = kendall_rows(scores, methods, args)
    split_half = split_half_rows(scores, methods, args)
    auc = auc_rows(methods, args)

    write_csv(args.out_root / "ranking.csv", ranking)
    write_csv(args.out_root / "kendall.csv", kendall)
    write_csv(args.out_root / "split_half.csv", split_half)
    write_csv(args.out_root / "auc.csv", auc)
    write_markdown(args.out_root / "summary.md", methods, n_rows, ranking, kendall, split_half,
                   auc, per_subject, scores, args,
                   support=support_stats(methods, args.out_root),
                   disagreement_mm=landmark_disagreement_mm(args.out_root))
    print(f"[ws3b-an] {len(ranking)} righe di classifica, {len(kendall)} di Kendall, "
          f"{len(split_half)} di split-half, {len(auc)} di AUC "
          f"-> {args.out_root / 'summary.md'}")


if __name__ == "__main__":
    main()
