#!/usr/bin/env python3
"""Matrici 100x100 varifold e currents sui soggetti held-out.

Le distanze le calcola il kernel di ``v2_work/phase0/measure_distances.py``: qui c'e' lo
scheduling, la costruzione delle misure e il salvataggio nel formato di ``common.py``.
Le sigma restano quelle di phase0, ``(0.5, 0.2, 0.1, 0.05)``.

Normalizzazione: maxabs, non area
---------------------------------
Il default di phase0 e' ``normalize="area"``: centro sul baricentro pesato sulle aree e
divisione per ``sqrt(area totale)``, cosi' la massa della misura vale 1 per ogni mesh.  Qui
il default e' ``maxabs``, cioe' ``common.maxabs_normalize``, la stessa normalizzazione di
Chamfer e dei render.  Il motivo e' che dividere per l'area rende la misura cieca proprio
alla perturbazione che dovrebbe vedere: su REMESH ``noisy`` ha 2.28x l'area della sua
``original`` (il rumore increspa i triangoli senza toccare l'ingombro), quindi la
normalizzazione area la rimpicciolisce di ``1/sqrt(2.28)`` e accorcia le distanze del 34%.
Misurato sulle matrici area (held-out, varifold): la mediana cross-soggetto vale 0.0846 fra
topologie di sola tassellazione e 0.3227 appena entra ``noisy``, cioe' d(original_i,
noisy_i) dello STESSO soggetto finisce dentro la nuvola dei soggetti diversi, e lo Spearman
vs D_GT delle celle con noisy scende da ~0.57 a ~0.15.  Con maxabs le due mesh restano
nello stesso spazio in cui le guarda Chamfer.

Esito misurato (job 1019726/1019727, Spearman vs D_GT mediato sulle coppie di topologie,
held-out, area -> maxabs): varifold same 0.577 -> 0.656, sola tassellazione 0.569 -> 0.503,
con noisy 0.162 -> 0.145; currents same 0.614 -> 0.661, tassellazione 0.337 -> 0.328, con
noisy 0.145 -> 0.323.  Cioe' maxabs NON ripara il varifold sulle celle con noisy: il
problema cambia segno invece di sparire.  Con area la misura di noisy era rimpicciolita
(mediana cross 0.32 contro 0.085); con maxabs ha 2.28x la massa della sua original, e la
distanza e' dominata dalla differenza di massa (mediana cross 1.36 contro 0.34).  Currents
ne beneficia perche' le normali increspate di noisy si cancellano col segno; il varifold,
che le eleva al quadrato, no.  Nessuna normalizzazione per similarita' toglie una
differenza di area vera: servirebbe normalizzare la massa (o confrontare misure a massa
unitaria) senza rimpicciolire la geometria.

``--normalize maxabs_unitmass`` fa esattamente questo, come opzione (i default non
cambiano): posizione e scala come Chamfer (centro sulla media dei vertici, divisione per il
maxabs), poi i pesi dei triangoli divisi per l'area totale, cosi' la massa vale 1 per ogni
mesh senza toccare la geometria.  Le sigma di phase0 erano tarate nel frame area, dove una
lunghezza vale ``L / sqrt(area)``; nel frame maxabs vale ``L / maxabs``.  Le sigma si
riportano quindi con ``SIGMA_AREA_TO_MAXABS``, la media di ``sqrt(area) / maxabs`` sulle
100 mesh original held-out (misurata, vedi la costante), e con loro la cella della griglia.
Le matrici si salvano sotto ``<kind>_maxabs_unitmass``, cioe' accanto a quelle maxabs e non
al loro posto.

Esito misurato (job 1054478 self-test, 1054479 matrici, 1054486 rank; held-out, Spearman
delle colonne same / no-crop / tassellazione / noisy, maxabs -> maxabs_unitmass): varifold
0.655/0.192/0.472/0.136 -> 0.646/0.258/0.496/0.307, currents 0.661/0.156/0.292/0.266 ->
0.663/0.210/0.287/0.289.  d(original_i, noisy_i) sta sotto la mediana fra soggetti diversi
per 100/100 soggetti (varifold 0.147 contro 0.166, currents 0.123 contro 0.143).

Un effetto collaterale da dichiarare: le sigma sono in unita' della normalizzazione, e la
mesh maxabs e' circa 1.6x piu' grande di quella area-normalizzata (misurato: ~21000 contro
~8000 atomi alla stessa cella).  A parita' di sigma il kernel guarda quindi un dettaglio
fisico piu' fine, e la misura costa 890 ms/coppia su T4 invece di 122.  Le due varianti non
differiscono percio' SOLO per la normalizzazione; chi vuole la vecchia passa
``--normalize area``, e le matrici di quel giro sono conservate in
``aau/runs/ws1_old_geom/areanorm_*``.

Risoluzione della misura: niente sottocampionamento casuale
-----------------------------------------------------------
``mesh_measure`` di phase0 sottocampiona a caso ``max_tris`` triangoli con un seme, e la
prima versione di questo script passava ``max_tris=4000`` con ``seed=indice del soggetto``.
Misurato: la stessa identica mesh con due semi diversi dista **0.054** (original), 0.044
(remesh), 0.058 (down8k), 0.050 (noisy), 0.044 (up60k) da se' stessa, contro distanze fra
soggetti diversi dello stesso ordine.  Il rumore di sottocampionamento era il segnale.

Qui il sottocampionamento casuale e' **spento** (``max_tris`` di phase0 messo oltre il
numero di triangoli) e sostituito da una regola deterministica per mesh:

la misura (non la mesh) viene **quantizzata su una griglia cubica**: i triangoli che
cadono nella stessa cella diventano un atomo solo, con area la somma delle aree, centroide
la media pesata sulle aree e normale la media pesata rinormalizzata.  La massa totale resta
identica, e la stessa mesh da sempre lo stesso risultato, quindi la self-distance e'
esattamente 0 (verificato su tutte e cinque le topologie).

La griglia e' la STESSA per tutte le topologie, anche per quelle che stanno gia' sotto il
tetto di atomi: quantizzarne solo alcune rimetterebbe dentro la misura proprio
l'asimmetria fra topologie che si vuole togliere.

La decimazione della MESH con ``mesh_ops.decimate_to`` sarebbe stata l'idioma del repo, ma
non e' applicabile qui: le mesh REMESH non sono edge-manifold (verificato con
``igl.is_edge_manifold`` su original, noisy e remesh) e sia ``igl.qslim`` sia
``igl.decimate`` restituiscono una mesh **vuota** su tutte e tre.  La quantizzazione della
misura non ha quel vincolo perche' non deve produrre una superficie: varifold e currents
vedono solo la terna (centroide, normale, area).

Il lato della cella e' scelto UNA VOLTA per run come il massimo fra due vincoli (vedi
``calibrate_cell``): un quarto della sigma piu' piccola, cioe' quattro volte piu' fine
della scala piu' fine a cui il kernel guarda, e il lato che tiene la mesh piu' fitta sotto
``--max-atoms``.  La griglia e' ancorata all'origine, che dopo la normalizzazione e' il
centro della mesh -- la media dei vertici con ``maxabs``, il baricentro pesato sulle aree
con ``area`` -- cioe' in tutti e due i casi un punto che non si sposta al cambiare della
topologia.  Cosi' il numero di atomi diventa una proprieta' della SUPERFICIE e non della
tassellazione, che e' esattamente la nuisance che il varifold dovrebbe ignorare.

Conteggi misurati sui triangoli di REMESH (8 soggetti): down8k 16000, remesh 32507,
crop 41637, original e noisy 46440, up60k 120129.  Coi default (cella 0.0125 = sigma_min/4)
gli atomi sono, con ``--normalize maxabs``, 12848 / 19575 / 20987 / 22884 / 25498 per
down8k / remesh / original / noisy / up60k (mediane su 12 soggetti), e con
``--normalize area`` 6832 / 7859 / 8008 / 4159 / 8744.  In tutti e due i casi quasi
indipendenti dalla tassellazione; con area ``noisy`` e' l'unica fuori riga, ed e' il
sintomo del problema descritto sopra.

Che la quantizzazione non sposti le distanze e' misurato, non assunto (6 soggetti, coppie
original->remesh e original->original, contro le stesse misure a triangoli interi):

    cella      atomi orig   scarto max same-subj   scarto max diff-subj   spearman diff-subj
    0.0070     20324        5.3%                   0.8%                   1.000
    0.0125     7992         8.1%                   0.7%                   1.000
    0.0250     2229         9.5%                   2.3%                   0.943

Le distanze fra soggetti diversi -- quelle su cui si calcola lo Spearman contro D_GT --
cambiano meno dell'1% e non cambiano di ordine per nulla; a 0.0250 (cella = sigma_min/2) si
comincia a vedere, ed e' per questo che il default e' un quarto e non la meta'.

Device
------
A 20000 atomi per lato il prodotto interno e' una somma su 4e8 coppie per sigma: misurato,
16.5 s per coppia su un core e 0.89 s su GPU (T4, blocco 4096, ``--normalize maxabs``),
cioe' su CPU non e' praticabile.  ``--device cuda`` costruisce le misure sulla GPU e usa ``geometric_kernel.py``,
che e' la stessa formula a blocchi di phase0 resa device-agnostica e con i due kind
calcolati in una passata sola.  Su GPU il processo e' uno solo (CUDA non sopravvive a
``fork``); su CPU resta la pool.

  aau/submit.sh baselines/geometric.sbatch
  aau/run.sh aau/baselines/geometric_matrix.py --device cuda --time-only 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import geometric_kernel  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "phase0"))

from measure_distances import (  # noqa: E402
    BLOCK,
    DEFAULT_SIGMAS,
    mesh_measure,
)

KINDS = geometric_kernel.KINDS
NORMALIZATIONS = ("area", "maxabs", "maxabs_unitmass")

# sqrt(area totale) / maxabs sulle 100 mesh original held-out: media 1.8482 (mediana 1.8477,
# std 0.0495, da 1.731 a 1.986; 1.8612 su facebench_first100), aau/scratch/
# ws1_unitmass_factor.py.  Una lunghezza che nel frame area vale s nel frame maxabs vale
# s * SIGMA_AREA_TO_MAXABS.  E' una costante e non si ricalcola per set, perche' altrimenti
# lo stesso nome di metrica avrebbe sigma diverse su set diversi.
SIGMA_AREA_TO_MAXABS = 1.8482

# Stato condiviso coi worker CPU via fork: le misure si costruiscono una volta sola nel
# padre (quantizzare 500 mesh una volta per worker sarebbe lavoro buttato).
_MEASURES: dict[tuple[str, str], dict] = {}
_KINDS: tuple[str, ...] = KINDS
_SIGMAS: tuple[float, ...] = DEFAULT_SIGMAS
_BLOCK: int = BLOCK


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--kinds", type=str, default=",".join(KINDS))
    p.add_argument("--settings", type=str, default=",".join(common.SETTINGS))
    p.add_argument("--max-atoms", type=int, default=25000,
                   help="tetto al numero di atomi della misura quantizzata")
    p.add_argument("--cell-sigma-frac", type=float, default=0.25,
                   help="lato di cella come frazione della sigma piu' piccola")
    p.add_argument("--normalize", type=str, default="maxabs", choices=NORMALIZATIONS,
                   help="maxabs = la normalizzazione di Chamfer, il default qui; "
                        "area = il default di phase0; maxabs_unitmass = maxabs con massa 1, "
                        "vedi il docstring")
    p.add_argument("--sigmas", type=str, default="",
                   help="vuoto = quelle di phase0, riportate nel frame maxabs con "
                        "--normalize maxabs_unitmass")
    p.add_argument("--block", type=int, default=BLOCK)
    p.add_argument("--device", type=str, default="cpu", help="cpu oppure cuda")
    p.add_argument("--workers", type=int, default=8, help="solo con --device cpu")
    p.add_argument("--chunk", type=int, default=128, help="coppie per task inviato ai worker")
    p.add_argument("--subject-set", type=str, default="heldout", choices=common.SUBJECT_SETS,
                   help="heldout = split del repo; facebench_first100 = i soggetti della Tabella 2")
    p.add_argument("--max-subjects", type=int, default=0, help="0 = tutti; >0 per un test rapido")
    p.add_argument("--topology-pairs", type=str, default="",
                   help="sottoinsieme 'tA:tB,tA:tB' delle coppie del setting, per spezzare il job")
    p.add_argument("--self-test", action="store_true",
                   help="self-distance e confronto col kernel di phase0, poi esce")
    p.add_argument("--time-only", type=int, default=0,
                   help=">0: misura il costo su questo numero di coppie e esce")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _cell_counts(centroids: np.ndarray, cell: float) -> np.ndarray:
    """Etichetta di cella di ogni triangolo, su una griglia ancorata all'origine."""
    idx = np.floor(np.asarray(centroids, dtype=np.float64) / cell).astype(np.int64)
    idx -= idx.min(axis=0)
    span = idx.max(axis=0) + 1
    return idx[:, 0] * (span[1] * span[2]) + idx[:, 1] * span[2] + idx[:, 2]


def choose_cell_size(centroids: np.ndarray, target: int, steps: int = 48) -> float:
    """Lato di cella che porta il numero di celle occupate vicino a ``target``.

    Le celle occupate calano al crescere del lato, quindi la bisezione e' ben definita.
    """
    lo, hi = 1e-5, 1.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        occupied = len(np.unique(_cell_counts(centroids, mid)))
        if occupied > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def quantize_measure(measure: dict, cell: float) -> dict:
    """Un atomo per cella occupata: massa totale invariata, risultato deterministico."""
    c = measure["centroids"].numpy().astype(np.float64)
    n = measure["normals"].numpy().astype(np.float64)
    a = measure["areas"].numpy().astype(np.float64)

    _, inverse = np.unique(_cell_counts(c, cell), return_inverse=True)
    k = int(inverse.max()) + 1
    weights = np.bincount(inverse, weights=a, minlength=k)
    safe = np.maximum(weights, 1e-30)[:, None]
    centroids = np.stack([np.bincount(inverse, weights=a * c[:, d], minlength=k)
                          for d in range(3)], axis=1) / safe
    normals = np.stack([np.bincount(inverse, weights=a * n[:, d], minlength=k)
                        for d in range(3)], axis=1)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    # Una cella che contiene una piega ha normale media ~0: li' si tiene la normale del
    # triangolo di area maggiore, invece di amplificare il rumore con la rinormalizzazione.
    degenerate = (norm[:, 0] < 1e-9 * weights)
    if degenerate.any():
        order = np.lexsort((a, inverse))
        largest = np.zeros(k, dtype=np.int64)
        largest[inverse[order]] = order
        normals[degenerate] = n[largest[degenerate]]
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-30)

    import torch

    t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    return {"centroids": t(centroids), "normals": t(normals), "areas": t(weights), "_cache": {}}


def to_device(measure: dict, device: str) -> dict:
    import torch

    dev = torch.device(device)
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in measure.items()} | {"_cache": {}}


def metric_name(kind: str, normalize: str) -> str:
    """Nome della matrice su disco: area e maxabs restano ``kind`` come prima."""
    return kind if normalize in ("area", "maxabs") else f"{kind}_{normalize}"


def default_sigmas(normalize: str) -> tuple[float, ...]:
    """Le sigma di phase0, riportate nel frame maxabs per ``maxabs_unitmass``."""
    if normalize == "maxabs_unitmass":
        return tuple(s * SIGMA_AREA_TO_MAXABS for s in DEFAULT_SIGMAS)
    return tuple(DEFAULT_SIGMAS)


def phase0_measure(src, max_tris: int, seed: int, normalize: str) -> dict:
    """``mesh_measure`` di phase0, piu' la massa unitaria per ``maxabs_unitmass``.

    Posizione e scala sono quelle di ``maxabs``; poi le aree si dividono per la loro somma.
    Dividere le aree non tocca centroidi ne' normali, quindi la geometria resta quella che
    vede Chamfer.
    """
    base = "maxabs" if normalize == "maxabs_unitmass" else normalize
    measure = mesh_measure(src, max_tris=max_tris, seed=seed, normalize=base)
    if normalize == "maxabs_unitmass":
        areas = measure["areas"].double()
        measure["areas"] = (areas / areas.sum()).float()
    return measure


def build_measure(subject: str, topology: str, normalize: str, device: str,
                  cell: float | None = None) -> dict:
    """Misura geometrica di una mesh, senza nessun sottocampionamento casuale.

    A ``mesh_measure`` si passa un ``max_tris`` piu' grande del numero di triangoli, cosi'
    il ramo ``np.random.default_rng(seed).choice`` di phase0 non viene mai preso; la
    riduzione, se serve, la fa ``quantize_measure`` sulla misura gia' normalizzata.
    """
    V, F = common.load_verts_faces(subject, topology)
    measure = phase0_measure((V, F), max_tris=len(F) + 1, seed=0, normalize=normalize)
    if cell is not None:
        measure = quantize_measure(measure, cell)
    return to_device(measure, device)


def calibrate_cell(subjects, topologies, sigmas, args) -> float:
    """Lato di cella comune a TUTTE le mesh: il massimo fra due vincoli.

    1. **Risoluzione**: la cella deve stare ben sotto la sigma piu' piccola, altrimenti la
       quantizzazione si vede nella distanza.  ``--cell-sigma-frac 0.25`` mette la cella a
       un quarto di ``min(sigmas)``, cioe' quattro volte piu' fine della scala piu' fine a
       cui il kernel guarda: sotto quel livello la gaussiana non distingue piu' niente.
    2. **Costo**: il prodotto interno e' O(N*M), quindi si tiene comunque un tetto
       ``--max-atoms`` sul numero di atomi della mesh piu' fitta.

    Si prende il lato piu' GRANDE fra i due, cioe' il piu' economico che soddisfa entrambi.
    La stessa cella vale poi per ogni topologia: e' quello che rende il numero di atomi una
    proprieta' della superficie e non della tassellazione.
    """
    counts = {t: len(common.load_verts_faces(subjects[0], t)[1]) for t in topologies}
    densest = max(counts, key=counts.get)
    measure = phase0_measure(common.mesh_path(subjects[0], densest),
                             max_tris=counts[densest] + 1, seed=0, normalize=args.normalize)
    centroids = measure["centroids"].numpy().astype(np.float64)
    by_cost = choose_cell_size(centroids, args.max_atoms)
    by_sigma = args.cell_sigma_frac * min(sigmas)
    cell = max(by_cost, by_sigma)
    print(f"[geom] lato di cella {cell:.6f} = max(costo {by_cost:.6f} per <= {args.max_atoms} "
          f"atomi su {subjects[0]}/{densest}, risoluzione {by_sigma:.6f} = "
          f"{args.cell_sigma_frac} x sigma_min {min(sigmas)})", flush=True)
    return cell


def build_all_measures(subjects, topologies, args, cell) -> None:
    """Riempie ``_MEASURES`` nel padre: i worker CPU se lo ereditano col fork."""
    t0 = time.time()
    for topology in topologies:
        for subject in subjects:
            _MEASURES[(subject, topology)] = build_measure(
                subject, topology, args.normalize, args.device, cell)
        sizes = [len(_MEASURES[(s, topology)]["areas"]) for s in subjects]
        print(f"[geom] misure {topology}: {len(subjects)} mesh, atomi "
              f"{min(sizes)}/{int(np.median(sizes))}/{max(sizes)} (min/med/max) "
              f"({time.time() - t0:.0f}s)", flush=True)


def _init_worker(kinds, sigmas, block) -> None:
    import torch

    torch.set_num_threads(1)  # il parallelismo e' sui processi, non dentro BLAS
    global _KINDS, _SIGMAS, _BLOCK
    _KINDS, _SIGMAS, _BLOCK = kinds, sigmas, block


def _run_chunk(task):
    """Le distanze di tutti i kind per un blocco di coppie: una passata di esponenziali."""
    topology_a, topology_b, subjects_a, subjects_b = task
    return [geometric_kernel.distances(_MEASURES[(sa, topology_a)], _MEASURES[(sb, topology_b)],
                                       sigmas=_SIGMAS, kinds=_KINDS, block=_BLOCK)
            for sa, sb in zip(subjects_a, subjects_b)]


# --------------------------------------------------------------------- controlli

def run_self_test(subjects, topologies, args, kinds, sigmas, cell) -> None:
    """Self-distance della regola deterministica e scarto dal kernel di phase0."""
    from measure_distances import currents_distance, varifold_distance

    print("[self-test] self-distance con la regola deterministica (deve essere 0):", flush=True)
    for topology in topologies:
        m1 = build_measure(subjects[0], topology, args.normalize, args.device, cell)
        m2 = build_measure(subjects[0], topology, args.normalize, args.device, cell)
        d = geometric_kernel.distances(m1, m2, sigmas=sigmas, kinds=kinds, block=args.block)
        print(f"  {topology:10s} atomi={len(m1['areas']):6d} "
              f"massa={float(m1['areas'].double().sum()):.6f} " +
              " ".join(f"{k}={d[k]:.3e}" for k in kinds), flush=True)

    print("[self-test] self-distance del PROTOCOLLO VECCHIO (max_tris=4000, seme per mesh):",
          flush=True)
    for topology in topologies:
        path = common.mesh_path(subjects[0], topology)
        m1 = phase0_measure(path, max_tris=4000, seed=0, normalize=args.normalize)
        m2 = phase0_measure(path, max_tris=4000, seed=1, normalize=args.normalize)
        print(f"  {topology:10s} varifold={varifold_distance(m1, m2, sigmas=sigmas):.6f} "
              f"currents={currents_distance(m1, m2, sigmas=sigmas):.6f}", flush=True)

    print("[self-test] geometric_kernel contro measure_distances (stesse misure, CPU):", flush=True)
    ref = {"varifold": varifold_distance, "currents": currents_distance}
    for topology in topologies[:2]:
        mA = build_measure(subjects[0], topology, args.normalize, "cpu", cell)
        mB = build_measure(subjects[1], topology, args.normalize, "cpu", cell)
        mine = geometric_kernel.distances(mA, mB, sigmas=sigmas, kinds=kinds, block=args.block)
        for kind in kinds:
            theirs = ref[kind](mA, mB, sigmas=sigmas, block=args.block)
            rel = abs(mine[kind] - theirs) / max(abs(theirs), 1e-30)
            print(f"  {topology:10s} {kind:9s} qui {mine[kind]:.8f} phase0 {theirs:.8f} "
                  f"rel {rel:.2e}", flush=True)

    if {"original", "noisy"} <= set(topologies):
        run_same_vs_cross(subjects, args, kinds, sigmas, cell)


def run_same_vs_cross(subjects, args, kinds, sigmas, cell) -> None:
    """d(original_i, noisy_i) dello stesso soggetto contro la mediana fra soggetti diversi.

    E' il controllo che la normalizzazione area falliva: se la stessa faccia con il rumore
    sta piu' lontana della mediana di due facce diverse, la colonna con noisy non puo'
    rankare niente.  Le coppie diverse sono le i<j, le stesse delle matrici.
    """
    orig = [build_measure(s, "original", args.normalize, args.device, cell) for s in subjects]
    noisy = [build_measure(s, "noisy", args.normalize, args.device, cell) for s in subjects]
    pair_i, pair_j = common.subject_pair_indices(len(subjects))
    same = {k: [] for k in kinds}
    cross = {k: [] for k in kinds}
    for i in range(len(subjects)):
        d = geometric_kernel.distances(orig[i], noisy[i], sigmas=sigmas, kinds=kinds,
                                       block=args.block)
        for k in kinds:
            same[k].append(d[k])
    for i, j in zip(pair_i, pair_j):
        d = geometric_kernel.distances(orig[i], noisy[j], sigmas=sigmas, kinds=kinds,
                                       block=args.block)
        for k in kinds:
            cross[k].append(d[k])
    print(f"[self-test] original->noisy, {len(subjects)} soggetti uguali contro {len(pair_i)} "
          f"coppie di soggetti diversi (normalize={args.normalize}):", flush=True)
    for k in kinds:
        s, c = np.asarray(same[k]), np.asarray(cross[k])
        median_cross = float(np.median(c))
        print(f"  {k:9s} stesso soggetto mediana {np.median(s):.6f} (max {s.max():.6f})  "
              f"soggetti diversi mediana {median_cross:.6f}  "
              f"stesso < mediana diversi: {int((s < median_cross).sum())}/{len(s)}", flush=True)


def run_time_only(subjects, topology_pairs, args, kinds, sigmas, n_pairs: int) -> None:
    """Costo per coppia sulle prime ``n_pairs`` coppie della prima coppia di topologie."""
    pair_i, pair_j = common.subject_pair_indices(len(subjects))
    names = np.asarray(subjects)
    for topology_a, topology_b in topology_pairs[:2]:
        t0 = time.time()
        for k in range(n_pairs):
            geometric_kernel.distances(
                _MEASURES[(names[pair_i[k]], topology_a)],
                _MEASURES[(names[pair_j[k]], topology_b)],
                sigmas=sigmas, kinds=kinds, block=args.block)
        if args.device != "cpu":
            import torch
            torch.cuda.synchronize()
        seconds = time.time() - t0
        total = len(pair_i) * len(topology_pairs)
        print(f"[time] {topology_a}->{topology_b}: {n_pairs} coppie in {seconds:.1f}s "
              f"({seconds / n_pairs * 1000:.0f} ms/coppia, self-inner gia' in cache dopo la prima) "
              f"-> {total} coppie x {seconds / n_pairs / 3600:.4f} h = "
              f"{total * seconds / n_pairs / 3600:.1f} h per set", flush=True)


# ----------------------------------------------------------------------------

def selected_topology_pairs(args) -> list[tuple[str, str]]:
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    pairs = common.all_topology_pairs(settings)
    if not args.topology_pairs.strip():
        return pairs
    wanted = [tuple(p.split(":")) for p in args.topology_pairs.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in pairs]
    if unknown:
        raise SystemExit(f"coppie di topologie fuori dai setting richiesti: {unknown}")
    return wanted


def main() -> None:
    import multiprocessing as mp

    args = parse_args()
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise SystemExit(f"kind sconosciuto: {unknown} (attesi {list(KINDS)})")
    sigmas = (tuple(float(s) for s in args.sigmas.split(",") if s.strip())
              or default_sigmas(args.normalize))
    metrics = {kind: metric_name(kind, args.normalize) for kind in kinds}

    subjects = common.subject_set(args.subject_set)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]
    n = len(subjects)
    pair_i, pair_j = common.subject_pair_indices(n)
    names = np.asarray(subjects)

    topology_pairs = selected_topology_pairs(args)
    topologies = sorted({t for pair in topology_pairs for t in pair})

    global _KINDS, _SIGMAS, _BLOCK
    _KINDS, _SIGMAS, _BLOCK = kinds, sigmas, args.block

    if args.self_test:
        run_self_test(subjects, topologies, args, kinds, sigmas,
                      calibrate_cell(subjects, topologies, sigmas, args))
        return

    todo = [
        (ta, tb) for ta, tb in topology_pairs
        if args.overwrite or not all(
            common.matrix_path(metrics[kind], ta, tb, args.out_root).exists() for kind in kinds)
    ]
    print(f"[geom] soggetti={n} coppie/topologia={len(pair_i)} topologie={topologies} "
          f"metriche={list(metrics.values())} da calcolare={len(todo)}/{len(topology_pairs)} "
          f"max_atoms={args.max_atoms} normalize={args.normalize} sigmas={sigmas} "
          f"device={args.device} workers={args.workers}", flush=True)
    if not todo and not args.time_only:
        print("[geom] niente da fare")
        return

    t0 = time.time()
    cell = calibrate_cell(subjects, topologies, sigmas, args)
    build_all_measures(subjects, topologies, args, cell)
    print(f"[geom] {len(_MEASURES)} misure pronte in {time.time() - t0:.0f}s", flush=True)

    if args.time_only:
        run_time_only(subjects, topology_pairs, args, kinds, sigmas, args.time_only)
        return

    extra = ({"sigma_area_to_maxabs": SIGMA_AREA_TO_MAXABS}
             if args.normalize == "maxabs_unitmass" else {})

    def save(kind, topology_a, topology_b, values) -> None:
        D = common.empty_matrix(n)
        D[pair_i, pair_j] = values
        common.save_matrix(
            common.matrix_path(metrics[kind], topology_a, topology_b, args.out_root),
            D, subjects, metrics[kind], topology_a, topology_b,
            max_atoms=args.max_atoms, normalize=args.normalize, sigmas=np.asarray(sigmas),
            cell_size=float(cell),
            reduction="quantizzazione su griglia, nessun sottocampionamento casuale",
            **extra,
        )

    if args.device == "cpu" and args.workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.workers, initializer=_init_worker,
                      initargs=(kinds, sigmas, args.block)) as pool:
            for topology_a, topology_b in todo:
                t1 = time.time()
                chunks = [
                    (topology_a, topology_b,
                     names[pair_i[start:start + args.chunk]].tolist(),
                     names[pair_j[start:start + args.chunk]].tolist())
                    for start in range(0, len(pair_i), args.chunk)
                ]
                blocks = [r for block in pool.map(_run_chunk, chunks) for r in block]
                for kind in kinds:
                    save(kind, topology_a, topology_b,
                         np.asarray([b[kind] for b in blocks], dtype=np.float64))
                print(f"[geom] {topology_a}->{topology_b}: {len(blocks)} coppie x {len(kinds)} "
                      f"kind in {time.time() - t1:.0f}s", flush=True)
    else:
        for topology_a, topology_b in todo:
            t1 = time.time()
            values = {kind: np.empty(len(pair_i), dtype=np.float64) for kind in kinds}
            for k, (i, j) in enumerate(zip(pair_i, pair_j)):
                d = geometric_kernel.distances(
                    _MEASURES[(names[i], topology_a)], _MEASURES[(names[j], topology_b)],
                    sigmas=sigmas, kinds=kinds, block=args.block)
                for kind in kinds:
                    values[kind][k] = d[kind]
                if (k + 1) % 500 == 0:
                    rate = (k + 1) / max(time.time() - t1, 1e-9)
                    print(f"[geom] {topology_a}->{topology_b}: {k + 1}/{len(pair_i)} "
                          f"({rate:.1f} coppie/s)", flush=True)
            for kind in kinds:
                save(kind, topology_a, topology_b, values[kind])
            print(f"[geom] {topology_a}->{topology_b}: {len(pair_i)} coppie x {len(kinds)} "
                  f"kind in {time.time() - t1:.0f}s", flush=True)

    print(f"[geom] fine in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
