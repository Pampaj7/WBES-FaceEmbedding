# P1, produttori e consumatori: note e misure (9 ottobre 2026)

Codice in `v3_work/stream/` (docstring di ogni modulo). Piano: `paper/PLAN_MASSIVE.md` sez. 8, 14.3, 15 (E9).

## Come si usa

    # produttori sul nodo del training (anello su /tmp: conta contro --mem)
    aau/run.sh v3_work/stream/producer.py --ring /tmp/$SLURM_JOB_ID/stream --ring-gb 24 --n-proc 48 \
        --k-eig 128 [--alpha 0] [--evecs-dtype fp16|fp32] [--cpus 8-31,40-63]
    # trainer v3 con --stream (stessi flag di train_v3.py)
    aau/run.sh v3_work/stream/train_stream.py --stream /tmp/$SLURM_JOB_ID/stream --stream-reuse 4 \
        --total-steps T --steps-per-epoch S --batch_subjects 16 --max_meshes_per_subject_train 4 ...
    # tutto insieme, con le misure: v3_work/stream/slurm/train_stream.sbatch (variabili STREAM_*)

## Scelte e assunzioni

1. **`--stream` e' un wrapper, non un flag di `train_v3.py`.** La run dir di train_v3 e' l'hash di TUTTI gli
   argomenti: un flag nuovo nel parser cambierebbe la directory delle ablazioni c3f riprese dopo una prelazione
   (A100 con `--requeue`) e le farebbe ripartire da zero. `v3_work/trainer/` non e' toccato.
   `train_stream.py` sostituisce solo i batch di training (piani, embedder, GT del batch); tutto il resto e'
   `train_v3.run`. L'eval online resta sul dataset di `--data_dir`/`--store`.
2. **Domini:** bfm2019, ict, gnm, flame2020, famos (80 persone TRAIN, controllo contro lo split). Bilanciamento
   p_d ~ n_d^alpha con n_d = modi d'identita' (199/100/170/300) e persone FaMoS (80). alpha 0 = uniforme.
3. **BFM 2019 non aveva una mappa unificata.** `bfm2019_map.py` la costruisce con la procedura degli altri
   domini (`unified_gt/correspond.py`, chiamata com'e'). Esito: 1478/1478 punti coperti; landmark mediani
   1.78 -> 0.61 mm, tenuti fuori 0.89 mm (ICT 0.85, BFM 3DDFA 0.98); Chamfer 0.20 mm. Regione e mu invariati.
   Topologia di lavoro: media decimata a 10.101 v. (patch nativa 47.439), vertici incollati con baricentriche.
4. **FaMoS:** viste = neutra di riferimento o un fotogramma registrato (espressione reale), sulla maschera "face"
   FLAME, allineate rigidamente alla media FLAME.
5. **Etichette di topologia:** la discretizzazione, con `+x` se la vista ha un'espressione.
6. **GT al volo:** g = ||s_i - s_j|| / sqrt(A) / 14.1639 mm, la scala di `--gt unified`.
7. **Fuori:** deformazioni RBF, coppie di perturbazione, trasferimento d'espressioni fra modelli, FLAME 2023 e le
   varianti face12/fullHead di BFM 2019, quota fissa per i dati reali.

## Correttezza

**(a) Operatori** (`tests/test_ops_equiv.py`, `ops_equiv.json`): 30 viste (5 domini x 6 discretizzazioni).
Si confrontano la pipeline attuale (`prepass_v3` + loader congelato) e lo stream (anello + consumatore), con
l'embedding del checkpoint e108 su CPU.

| | vertici, massa, autovalori, facce, gradienti | max abs dz | dz relativo |
|---|---|---|---|
| attuale contro se stessa (pavimento: eigsh di ARPACK) | identici | 2.98e-7 | |
| stream, autovettori fp32 | identici bit per bit, stesso layout | **2.98e-7** | 2.5e-7 |
| stream, autovettori fp16 (default) | identici | **2.47e-4** | 2.5e-4 |

Con fp32 la soglia di 1e-5 e' rispettata (pari al pavimento). **Con fp16 no: 25 volte la soglia.** Gli
autovettori differiscono a meno del segno anche fra due giri della pipeline attuale; l'embedding no.

**(b) GT** (`tests/test_gt.py`, `gt_check.json`):
- s_i identico (scarto 0.0) a `UNIFIED_GT/shapes` per 600 ICT (ICT-5000 e due shard ICT_SCALE), 500 GNM e
  1000 FLAME;
- contro `s_train.npz`: al massimo 7e-6 mm;
- 604.450 coppie, di cui 300.000 fra domini, contro `D_orig` di `gt_unified_bfm_ict_gnm.npz`: al massimo
  1.27e-6 mm (8.9e-8 unita').

**DDP** (`slurm/ddp_cpu_check.sbatch`, `ddp_cpu_1062092/`): 2 rank, gloo. Il rank 0 usa solo gli shard pari, il
rank 1 solo i dispari; `check_sync` passa.

## Misure

**Produttori, solo CPU** (`producers/`, nodo a768-l40s-03, EPYC 9354, 64 CPU logiche allocate, 240 s per
configurazione). Viste di 5.7k vertici in media (5% up60k); tutti i domini; nessuna vista fallita.

| k | P=16 | P=32 | P=64 | MB per vista (fp16) |
|---|---|---|---|---|
| 64 | 49.5 | 82.2 | **89.9** | 1.9 |
| 128 | 27.4 | **38.0** | 33.4 | 2.7 |

Viste fresche/s a regime. Il gate di E9 (>= 20/s per nodo) passa con entrambi i k. A k 128 la curva si
satura, e cala, oltre 32 processi. Interpretazione, non misurata: le 64 CPU logiche sono 32 core con SMT,
ed eigsh e' limitata dalla banda di memoria.

**Run con 1 GPU** (nodi A10, Xeon Gold 6326, 64 CPU logiche; nessuna L40S con memoria libera nella sessione).
Produttori P=48, k 128, fp16, anello da 24 GiB, 16 identita' x 4 viste per passo, forward a gruppi, loss v2.

| run | passi | s/passo | mesh/s consumate | fresche/s | riuso | GPU (media) | attesa dati per passo |
|---|---|---|---|---|---|---|---|
| `train_1062091`, CPU condivise | 2000 | 1.81 -> 1.74 | 37 | 15.0 | 2.65 | 44% | 2 ms |
| `train_1062093`, anello congelato, senza produttori | 400 | 1.10 | 58 | - | - | 70% | <1 ms |
| `train_1062108`, 8 core al trainer, 24 ai produttori | 600 | 1.25 -> 1.20 | 53 | 13.1 | 3.6 | 66% | 1 ms |
| `train_1062115`, come 1062108 col codice finale | 1000 | 1.24 -> 1.17 | 53 | 13.3 | 3.77 | 64% | 1 ms |

- **Stabilita' (2000 passi):** loss per epoca 0.103 -> 0.050, sempre finita. Eval finale sul BFM REMESH (solo
  per far girare il trainer intero): sp_clean 0.518, aucR 0.994.
- **Rotazione dell'anello:** gli shard usati scorrono da 0-877 (epoca 1) a 4917-6774 (epoca 10). Eta' media
  della vista all'uso: 287 s. Dall'epoca 3, 130-430 viste per epoca (circa 8% delle ~5.200 prodotte) escono
  dall'anello senza essere usate. Nessun gruppo oltre il tetto di riuso.
- **Domini e discretizzazioni:** uniformi (25.2k-26.1k usi per dominio); up60k al 5%.
- **RAM:** anello a regime 24.0 GiB (il budget), 2.6 MB per vista con k 128 e fp16. Picco del cgroup del job:
  55 GiB, di cui 24 di anello e il resto produttori e trainer. Con autovettori fp32: +1.5 MB per vista a 5.7k
  vertici (+55%, stima dal formato, non misurata).
- **Collo di bottiglia:** non e' lo stream. Il trainer aspetta i dati 1-2 ms per passo, grazie al prefetch
  del batch successivo. Il passo rallenta del 60% (1.10 -> 1.74 s) per la contesa delle CPU fra produttori e
  la parte CPU del passo stesso. Dare 8 core al trainer riporta il passo a 1.20 s (GPU dal 44% al 66%), con il
  13% di viste fresche in meno.
- **Tetto del consumatore per rank, senza GPU** (`bench_consumer.py`):
  - nei run A10, con le CPU contese dai produttori e senza `--fast-data`: 146-252 mesh/s;
  - su un nodo L40S (a768-l40s-06, EPYC 9454), anello congelato, `--fast-data`, consumatore ristretto a 2/4/8
    CPU logiche (`consumer/`): 1.4k-1.6k / 1.2k-1.7k / 1.8k-2.1k mesh/s. Il valore piu' basso in ogni coppia e'
    con 8 thread di prefetch, quello piu' alto con 4.
  
  E' 8-10 volte le ~170 mesh/s per GPU di E9.
- **Costo del piano:** 20-37 ms per passo (2-3%), cresce col pool del rank (fino a 2.500 gruppi con un solo rank).

## Aperto

- **fp16 contro 1e-5:** decidere fra fp16 (2.5e-4 sull'embedding, -36% di RAM per vista) e fp32 (passa).
- **Passo su L40S non misurato.** Nessuna L40S con memoria libera nella sessione. Il lato dati per rank regge
  (punto sopra). Il limite vero di un nodo da 8 GPU sono le viste fresche, ed e' una stima, non una misura:
  - E9 misura ~170 mesh/s per GPU, quindi un consumo di ~1.300 mesh/s;
  - qui misurate 38/s (k 128) e 90/s (k 64) su 64 CPU logiche;
  - ne segue un riuso di ~15-35 volte, contro il tetto di 4 provato qui.

  Leve: k 64, piu' CPU ai produttori, R piu' alto o batch piu' piccoli. Le CPU vanno partizionate fra
  produttori e rank (punto "Collo di bottiglia").
