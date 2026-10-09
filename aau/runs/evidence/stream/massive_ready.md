# P1 pronta per il run massivo con la GT di E12 (9 ottobre 2026, sera)

Codice in `v3_work/stream/` (docstring di ogni modulo). Il run vero NON e' lanciato. In `v3_work/trainer/` nulla e'
cambiato. In `v3_work/mm_aug/aug.py` c'e' un'aggiunta: FLAME 2023 ammesso come template, con le soglie di FLAME 2020.
I default dello stream sono invariati: ogni novita' passa da un flag.

## Come si lancia

    # run principale: 12 L40S = 6 + 6 (STESSO numero di GPU per nodo: vedi "NCCL")
    STREAM_ARM=factorized|factorized2|ctrlfr STREAM_STEPS=T sbatch --nodes=2 --gpus-per-node=l40s:6 \
        --time=2-00:00:00 v3_work/stream/slurm/massive.sbatch
    # produttori extra solo CPU, anello condiviso su CephFS (<= 150 GiB; controlla la quota prima di partire)
    sbatch -p aicentre-a100 --qos=unprivileged --nodelist=nv-ai-04 --cpus-per-task=128 --mem=200G \
        --time=2-00:00:00 v3_work/stream/slurm/extra_producers.sbatch      # e STREAM_EXTRA=<anello> al run
    # secondari sulle A100 (secondo seme, LODO, nucleo aperto): stesso sbatch, -p aicentre-a100 --qos=unprivileged
    #   STREAM_SEED=2345 | STREAM_SOURCES=gnm,ict,flame2023,... | STREAM_SOURCES=open_core

Per nodo: produttori, anello in RAM e rank DDP (`massive_node.sh`). Il rendezvous e' c10d. Dopo un crash si
lancia un torchrun nuovo, fino a 3 tentativi, che riparte da `last.pth`. Un requeue riparte da `--resume auto`.

## 1. Domini (verificato)

- **Preset `massive`:** bfm2019, ict, gnm, flame2023 (FLAME 2023 Open, non 2020), famos.
  - FaMoS: le 80 persone TRAIN di `split.json`; c'e' una guardia contro quelle di TEST.
- **Preset `open_core`:** gnm, ict, flame2023.
- **ICT:** e' la versione Light pubblica, MIT (`external/ICT-FaceKit`, commit da5f95a, README "ICT Face Model Light").
- **Domini vietati:** FaceScape, FaceScape50, HIFI3D e FaceVerse sollevano `RoleError` in `v3_work/mm` e
  `ValueError` nello stream (`targets_check.json`, "roles").

## 2. GT di E12 al volo (`targets.py`, `producer.py --canonical-gt`)

Il codice e' quello di `train_fr_sr._factor_chunk`, importato da `cgt`. Per identita', dalla forma NEUTRA:
- **fr:** sqrt(w) a_i, dove a_i e' la rigida robusta in mm;
- **sr:** sqrt(w) z_i;
- **S_i:** la centroid size.

Unita' u_d:
- quelle di E12 per ict, gnm e flame;
- BFM 2019: mm dichiarati; R e t con la stessa rigida di `gt.py`;
- FaMoS: gia' in mm.

Mappe sulla regione:
- BFM 2019: `datasets/STREAM/maps/bfm2019.npz`, 1478/1478 punti;
- FLAME 2020 e 2023: la regione stessa.

**Controllo contro i file** (`tests/test_targets.py`, `targets_check.json`, PASSA):

| | identita' | fr (mm) | sr | S (mm) |
|---|---|---|---|---|
| ICT (ICT-5000 + 2 shard) | 600 | 1.7e-6 | 1.3e-8 | 9e-14 |
| GNM | 250 | 1.3e-6 | 1.0e-8 | 6e-14 |
| BFM REMESH (solo la funzione: non e' un dominio dello stream) | 200 | 1.6e-6 | 1.8e-8 | 6e-14 |

Coppie, 550.725, contro le GT grezze `gt_{fr,sr}`:
- FR: scarto massimo 2.1e-6 mm (relativo 2e-7);
- SR: 2.8e-8.

Nell'anello col consumatore la GT del batch coincide coi bersagli (scarto 0) e log S anche. L'area di X vale
area_mm2 / L0^2 entro 1e-8.

**Scala nel trainer** (`targets.gt_unit`):
- sr: 0.24412 d_P per unita' (= d_P_per_unit / kappa, la scala dei bracci factorized);
- fr: 16.974 mm per unita'. E' la media geometrica delle unita' tarate per dominio, perche' con batch misti un
  fattore per dominio non e' definito. **Assunzione mia.**

## 3. Taglia (`--size-mask-domains`)

Funziona coi domini dello stream (`tests/test_size_mask.py`, PASSA). BFM REMESH non e' nello stream: `bfm`
non maschera nulla.

CV di S_i (500 identita' per 3DMM, 80 per FaMoS):

| dominio | CV |
|---|---|
| **BFM 2019** | **0.049** (S 56.7 mm, p5-p95 52.0-61.0): valida |
| ICT | 0.053 |
| GNM | 0.063 |
| FLAME 2020 | 0.047 |
| FaMoS | 0.038 |
| BFM REMESH (E12) | 0.019 |

Nella prova (sez. 11, ibridi compresi) il CV vale 0.052 per BFM 2019 (3089 identita'), 0.051 per FLAME 2023, 0.058
per GNM e 0.054 per ICT.

## 4. Espressioni

`--expr-frac f`: esattamente round(f x viste) viste con espressione per gruppo, con arrotondamento stocastico.
Fonti delle espressioni:
- il prior di ogni 3DMM;
- i fotogrammi reali di FaMoS;
- con `--mm-aug`, anche trasferite fra modelli.

Il default `--p-expr` per vista e' invariato. Misurato: quota 0.500 (`provenance_check.json`). Spostamento
massimo espressione contro neutra: 2.5-31 mm su tutti i domini.

## 5. Ingresso globale, operatori

- Ogni vista porta `area_mm2`, l'area in mm veri: area della vista x (u_d / s_d)^2.
- Il consumatore fa X = ((V - c) f) / L0 come `global_v3`, senza R_d perche' la vista e' gia' canonica.
- Con `--input-norm global` serve `--stream-scale 0`; la scala con bersaglio e' `--scale-aug`.
- `grad_vec` nei produttori.
- Flag: `--k-eig`, `--v-max`, `--v-work`.

## 6. Trainer (`train_stream.py --stream`)

- **Flag:** `--stream-gt unified|fr|sr`.
- **Teste provate su L40S:**
  - factorized (100 passi): 1.12-1.19 s/passo;
  - factorized2 (60): 1.18-1.21;
  - ctrlfr (60): 1.14-1.18.
- **log S:** viene dalle viste (`StreamLogCS`).
- **Bias iniziale di s:** media sull'anello, mediata fra i rank.
- **DDP:**
  - 2 GPU su un nodo: 0.97-1.07 s/passo;
  - **2 nodi x 1 GPU con crash al passo 25:** ripresa da last.pth al passo 20, 60/60 passi, pesi uguali (check_sync).
  - Fra nodi gli shard si dividono per rank locale.

**NCCL (scoperta, misurata):**
- con un numero di GPU DIVERSO fra i nodi (2 + 1) la prima all_reduce non finisce mai;
- l'ho provato con RoCE, socket, Ring/Tree, PXN/P2P/SHM spenti, NCCL 2.22 e quello di pytorch_25.09;
- 1 + 1, 1 + 1 + 1 e 2 + 2 (2 A100 + 2 L40S) funzionano;
- **quindi niente 8 + 4: 6 + 6.** massive.sbatch rifiuta i job eterogenei senza `STREAM_ALLOW_HETERO=1`;
- anche il riavvio elastico dentro lo stesso torchrun si blocca fra nodi; per questo si usa il ciclo di torchrun nuovi;
- banda RoCE: busbw 1.7-2.7 GB/s (64 MB). Il gradiente pesa ~3 MB per passo.

## 7. Rilascio: provenienza, rigenerazione, registro, fonti

- **`--provenance`:** un seme per gruppo. Ogni gruppo porta fonti, licenza ereditata (la piu' restrittiva), tipo,
  coefficienti e ricetta.
  - `regen.py` rigenera la mesh: 120 gruppi, 480 viste, di tutti e 4 i tipi; facce identiche, vertici entro 1.3e-7.
  - Gli operatori NON si rigenerano bit per bit (eigsh).
- **`--stream-log-views`:** una riga per vista usata in `views_used/*.npz`, ~64 B per riga prima della
  compressione. Il riepilogo e' `views_log.py`.
- **Filtro delle fonti:** `--sources` / `--stream-sources`, preset `open_core`. Un gruppo entra solo se TUTTE le
  sue fonti sono ammesse; misurato: solo gnm, ict, flame2023, 100% ridistribuibile.

## 8. Moltiplicatori (`--mm-aug hybrid=..,expr_transfer=..,rbf=..`)

- Un gruppo = UNA identita' di mm_aug e le sue viste (`aug_group_spec`). La GT viene dalla neutra esatta.
- Con validita' `cheap`: il pezzo riscartato si riestrae.
- FaMoS resta puro, perche' manca come fonte d'identita' degli ibridi.
- Default nello sbatch: 0.3 / 0.15 / 0.15 (quelli di mm_aug).
- **Misurato:** viste pure / ibride / trasferite / rbf = 1412 / 616 / 336 / 308.
- `selftest` di mm_aug dopo la mia modifica: 24/24.

## 9. Anello condiviso su CephFS

- **Banda CephFS:** scrittura 907 MB/s (16 processi). Lettura da un altro nodo: 60 / 376 / 602 MB/s con 1 / 8 / 16
  processi.
- **Letto direttamente dal trainer:** 3.4 s/passo invece di 1.05. Il readahead al primo accesso costa ~2 s per piano.
- **Rimedio, `--stream-extra-mirror`:** un thread per rank copia gli shard su /tmp. Copia misurata 37.6 MB/s per
  rank; il passo torna a 1.02-1.10 s.
- **Tetto stimato:** 12 rank x 37.6 MB/s = ~450 MB/s, cioe' ~110 viste/s a k128.
- **Budget:** 150 GiB, imposto, e controllo che restino >= 300 GB liberi.
- **Produttori extra misurati:** 7.8 viste/s su 32 CPU Xeon (nodi A10), k128.

## 10. k64 contro k128

**Produttori** (EPYC 9354, CPU di un nodo L40S, 64 CPU logiche, massive, fp32), viste/s:

| | P=16 | P=32 | P=64 | MiB per vista |
|---|---|---|---|---|
| k64 | 47.7 | 76.1 | **89.9** | 2.5 |
| k128 | 26.9 | 39.4 | **40.5** | 3.9 |

Rapporto **2.2x** a P=64, non 3x.

**Ablazione C3F** (robal, 21.096 passi, A100):
- k64 per troncatura degli operatori dello store (`ktrunc.py`). E' equivalente al calcolo con k=64: embedding
  relativo 2.9e-7, pavimento 2.4e-7 (`ktrunc_check.json`);
- **LANCIATA alle 22:19, dopo la fine dei 6 bracci:** 1065293 (k128, in corso) e 1065295 (k64, in coda), eval
  1065294 e 1065296.

## 11. Prova: 2 L40S, 2000 passi (`trial_2gpu/`)

Configurazione: factorized, massive, mm_aug, k128, anello locale 20 GiB + CephFS con mirror.

- **Passo:** 1.02-1.10 s per 8 epoche, 1.14-1.27 alla fine (l'anello condiviso non si riempiva piu'). Cioe'
  ~61 mesh/s per GPU.
- **GPU al 35%:** il passo e' limitato dalla CPU (forward sequenziale, obbligato per factorized; 4 CPU per rank).
- **Viste fresche:** 16.3/s dai 20 produttori locali, piu' ~12/s dal CephFS (41% degli usi).
- **Riuso:** 3.4 -> 4.2; 1124 gruppi oltre il tetto all'ultima epoca.
- **Loss:** 0.123 -> 0.050; MSE di s 0.018 -> 0.0018.
- **RAM:** picco 61.5 GiB.
- **Registro:** 60.984 viste uniche e 12.257 identita'. Il 47% delle viste e' ridistribuibile: chi ha BFM o FaMoS
  fra le fonti eredita "solo ricetta". Il registro e' nel formato vecchio, con stringhe e coefficienti: 10 MB.

## 12. Stima per 12 L40S (6 + 6) in 48 h

Ipotesi:
- 14 CPU per GPU: 4 al rank e 60 produttori per nodo;
- passo 1.05-1.09 s, uguale su uno o due nodi (sez. 13). Il caso peggiore a 1.9 s di prima era un artefatto;
- 768 mesh per passo.

**Passi e consumo:** ~160k passi; ~710 mesh/s. Con 8 CPU per rank e groups (solo ctrlfr): fino a ~1270 mesh/s.

**Viste fresche/s**, da bench e prova:

| | nodi del training | extra (CephFS) | totale |
|---|---|---|---|
| k128 | ~80 | ~50, con tetto ~110 | ~130 |
| k64 | ~180 | ~100 | ~280 |

**In 48 h:**

| | viste uniche | identita' | riuso |
|---|---|---|---|
| k128 | ~22M | ~5.6M | 3-6, oltre il tetto di 4 |
| k64 | ~48M | ~12M | 1.5-2.6 |

**Ripartizione delle identita'** (alpha 0, mm_aug di default):
- 20% per dominio. FaMoS sono 80 persone: i suoi gruppi sono fotogrammi diversi;
- per il 80% di gruppi 3DMM: puri 40%, ibridi 30%, trasferimenti 15%, rbf 15%;
- a k128 questo fa circa 1.8M puri, 1.35M ibridi e 0.7M + 0.7M.

**RAM per nodo da 6 GPU:** ~225 GB su 480. Sono anello 60, mirror 75, produttori ~30 e rank ~60.

Il produttore extra di nv-ai-04 non e' misurato: il bench 1065297 e' in coda. **Gli extra sono una stima.**

## 13. Profilo del passo e passo fra nodi (`profile/`, WBES_STREAM_PROFILE=1)

Mediane per passo (64 mesh per rank), 150 passi:

| configurazione | dati | forward | backward | totale | mesh/s per GPU | GPU |
|---|---|---|---|---|---|---|
| factorized, sequenziale, 4 CPU, 1 nodo | 0.004 | 0.63 | 0.42 | 1.08 | 59 | 36% |
| factorized, sequenziale, 8 CPU | 0.003 | 0.52 | 0.38 | 0.93 | 69 | 38% |
| factorized, **2 nodi x 1 GPU**, 4 CPU | 0.005 | 0.56-0.63 | 0.43-0.47 | **1.09** | 59 | 35% |
| idem, checkpoint ogni minuto | | | | 1.08 | 59 | 40% |
| ctrlfr, sequenziale, 4 CPU | 0.004 | 0.65 | 0.38 | 1.07 | 60 | 38% |
| ctrlfr, `--forward groups`, 4 CPU | 0.004 | 0.36 | 0.30 | 0.68 | 95 | 48% |
| ctrlfr, `--forward groups`, 8 CPU | 0.003 | 0.27 | 0.31 | 0.60 | 106 | 62% |

**Fra nodi:**
- NCCL usa RoCE (`NET/IBext_v8`, `mlx5_bond_0`), non il TCP;
- il passo e' identico a quello su un nodo, e la all_reduce costa <= 0.05 s (differenza dei backward);
- l'1.75-1.9 s di `multinode_test` non si riproduce. Veniva dalla configurazione del test: epoche da 20 passi
  (il riscaldamento si spalma su pochi passi), checkpoint ogni 6 s, crash e ripresa;
- nessuna variabile d'ambiente necessaria.

**GPU al 35%:**
- non sono i dati: piano 4 ms, attesa dei campioni 1 ms;
- e' il costo CPU e di lancio dei kernel di 64 forward e backward per mesh.

**Correzioni, in ordine di costo:**
1. **8 CPU al rank invece di 4:** -14% sul passo. Costa poco ai produttori: a k128 la curva e' piatta oltre 32
   processi, 39.4 contro 40.5 viste/s.
2. **`--forward groups`:** -37% sul passo, -44% con 8 CPU. C'e' gia' per la testa standard (ctrlfr). NON per
   factorized/factorized2: il trainer lo vieta, perche' `embed_groups` chiama solo `pool_proj`. Aggiungere la testa
   della taglia a `_pool_masked` vale per factorized (~10 righe in `model_v3`); per factorized2 serve una
   normalizzazione per mesh nel batch. Da notare: groups e sequenziale non coincidono in fp32 (evidenza dell'8
   ottobre).

pin_memory e piu' consumer non servono: l'attesa dei dati e' gia' ~0.

## Aperto

- **Passo fra nodi:** risolto, nessun rallentamento (sez. 13).
- **forward a gruppi per factorized:** modifica del trainer da decidere.
- **Riuso:** con k128 servono `--stream-reuse 6` o k64. Decide l'ablazione k.
- **Pesi di dominio:** FaMoS al 20% con sole 80 persone.
- **Pulizia:** l'anello `datasets/STREAM/shared_ring` va cancellato a mano a fine run. Quello delle prove e'
  gia' cancellato.

## 14. Massima velocita' (10 ottobre, decisione dell'utente: scarti numerici di groups accettati)

### Forward a gruppi per factorized e factorized2

`model_v3._pool_masked` calcola anche la testa della taglia; `factorized_v3.group_normalize` fa la normalizzazione
per mesh di factorized2; `train_v3.check_args` ammette groups (packed resta vietato). I default sono invariati.

**Equivalenza** (`v3_work/trainer/tests/test_groups_factorized.py`, `trainer_v3/groups_factorized.json`, 64 viste vere):

| | Z relativo | s (max) | gradienti relativi (max) |
|---|---|---|---|
| fp32 | 6-7e-8 | 0 e 5e-7 | 1.4-3.1e-4 |
| TF32 | 8-9e-5 | | |

**Mesh/s per GPU**, solo forward + backward: sequenziale 84-86, groups 126-144.

### CPU per rank

Nel trainer completo, factorized con groups, mediane. L'ottimo e' 8 CPU per rank:

| CPU per rank | mesh/s per GPU |
|---|---|
| 8 | 108 (GPU 68%) |
| 12 | 112 |
| 16 | 110 |

- factorized2 con groups e 8 CPU: 106 mesh/s;
- nuovi default di massive.sbatch: `--cpus-per-gpu=20`, 8 CPU per rank, `--forward groups`, 2 copie del mirror per
  rank. Su un nodo da 6 GPU restano 72 CPU ai produttori.

### bf16 sulle MLP

Misurato con MiniMLP e SpatialGradientFeatures in autocast: groups 144 -> 147 mesh/s (+2%), sequenziale 86 -> 77.
L'autocast di tutto il forward fallisce (spmm sparso). NON adottato.

### Flotta di produttori (solo CPU, `--gres=NONE`: non conta nel tetto delle 12 GPU, QoS normal = solo gres/gpu)

Viste/s per nodo con tutte le CPU del job:

| nodo (CPU) | k64 | k128 |
|---|---|---|
| a768-l40s (EPYC 9354, 64) | 93.5 | 43.6 |
| a512-l4-06 (EPYC 7543, 96) | 69.9 | 27.8 |
| a256-a40 (EPYC 7302, 44) | 39.8 | 14.6 |
| i256-a40 (Xeon 5317, 48; partizione aicentre, solo `--qos=unprivileged`) | 33.1 | 13.9 |

Rapporto k64/k128 2.1-2.7. Partizione `cpu`: nodo in fail. Nodi A10 e T4: memoria piena (229-243 GB su 244).
nv-ai-04: 158 CPU libere ma 965 GB di memoria su 980 occupati dai job A100.

**Prova della flotta** (k64, tre job, 204 CPU, 15 min):
- **produzione:** 206 viste/s (38.7 + 76.0 + 91.3), nessun fallimento, ~515 MB/s scritti su CephFS;
- **lettura:** un nodo simulato (6 rank su 12, 2 copie per rank) legge 263 MB/s, cioe' 100 viste/s, la sua meta';
  persi 27 shard su 5080.
- Il budget dell'anello sforava di ~7 GiB su 40 con `--evict-every 8`. Ora c'e' `--evict-by-count`, il budget
  contato in numero di shard, attivo negli extra.

### Configurazione consigliata

- **Training:** 12 L40S = 6 + 6 nodi, `--cpus-per-gpu=20`, groups, 8 CPU per rank, mirror a 2-4 copie, k da
  decidere.
- **Flotta:** a256-a40-04..07, a512-l4-06, a768-l40s-01/03 (le CPU libere), i256-a40-01/02 (unprivileged), e
  nv-ai-04 se la memoria si libera. Tutti con `extra_producers.sbatch`, anello <= 150 GiB.

**Stima per 48 h:**

| | k64 | k128 |
|---|---|---|
| consumo | ~1300 mesh/s (12 x 108), ~293k passi | stesso |
| viste fresche locali | ~160/s | ~86/s |
| flotta (misurata 206 su 204 CPU, ~450 con tutte le CPU libere di oggi) | ~450/s | ~190/s |
| tetto di lettura CephFS (stima: ~600 MB/s per nodo con 16 flussi) | ~480 viste/s | |
| **totale** | **~610 viste/s** | **~280 viste/s** |
| riuso | ~2.1 | ~4.7 |
| viste uniche | ~105M | ~48M |
| identita' | ~26M | ~12M |

La flotta dipende da cosa e' libero, e le CPU unprivileged sono prelazionabili: e' una stima.

### k64 contro k128

Ablazione C3F ancora in corso:
- k128 (1065293): epoca ~30 di 72, ~3.5 h al termine;
- k64 (1065295): in coda sulle A100 (Priority).

Regola: k64 se le metriche del protocollo non peggiorano oltre il rumore.
