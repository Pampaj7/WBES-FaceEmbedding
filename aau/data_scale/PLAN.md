# Scala dei dati di training: studio misurato delle opzioni (5 ottobre 2026)

Obiettivo: almeno 10 volte i dati di training (oggi 500 identità BFM e 5000 ICT, × 6 topologie
= 33.000 mesh), mantenendo il rigore. **In questa fase non si generano dati.** Lo studio usa
90 mesh già esistenti, tutte scritte su `/tmp` del nodo. Sulla home sono finiti solo i JSON.

Legenda: **[M]** misurato in questa sessione, **[E]** estrapolato da misure, **[D]** da
documenti o log di sessioni precedenti, non rimisurato.

## 0. Vincolo: spazio

- Quota CephFS 1 TiB = 1099.5 GB (`ceph.quota.max_bytes`) [M]. Usati 1087.3 GB alle 11:20,
  967.7-968.0 GB alle 12:48, cioè 131.5 GB liberi, condivisi con gli altri agenti [M].
- Dove stanno oggi gli operatori (`ceph.dir.rbytes`) [M]: `ICT/topo_withops` 304.8 GB (30.000 mesh),
  `REMESH` 269.9 GB (7 varianti delle stesse 3000 mesh: `withops` 45.7, `areanorm` 76.0, `pot055`
  76.3, `robust_area1` 45.6, `cropF_withops` 20.9, più la geometria), `ICT/expressions_withops`
  69.9, `Multiface` 196.5, `aau/runs` 117.6 (di cui `multiface_ws3b` 104.2). Le viste
  `JOINT_*` e `train_ready` sono symlink e pesano 0.
- Composizione di una mesh `original` [M] (`measure_npz.json`): `evecs` 51.6%; indici COO di L,
  gradX e gradY in **int64** 33.6% (11.2% ciascuno, e i tre hanno **la stessa sparsità** su
  tutte le 24 mesh controllate); facce int64 4.8%; valori di L e dei gradienti 8.4%; vertici,
  massa ed evals 1.6%. Gli npz ad area unitaria (`areanorm`, `ICT/topo_withops`) sono scritti
  con `np.savez` **non compresso**.

## 1. Tabella misurata

Campione [M]: 10 identità ICT held-out (ict4500-4509) e 5 BFM (id0000-0004), × 6 topologie =
90 mesh, operatori ad area unitaria k_eig 128. Modello: il congiunto BFM+ICT
`x3dmm_joint_bfm_ict_s1234_1019532/epoch120`, cioè la stessa convenzione degli operatori.
Ogni variante è letta dal **loader congelato** (`GTReadyDatasetNPZ`). Script:
`study_options.py`, risultati in `study_options.json`. Il tempo del loader è in
`loader_timing.json`.

"dz" = ‖z_var − z_ref‖ / mediana delle distanze latenti fra mesh del campione. "ρ var" =
Spearman fra le distanze latenti di ref e della variante, su tutte le coppie di mesh di identità
diverse (1620 coppie ICT, 360 BFM). "ρ GT" = Spearman latente-GT a livello di coppia di mesh,
cioè il tipo di numero già riportato nel paper: ref vale 0.9698 su ICT e 0.6962 su BFM.

| variante | MB/identità ICT | MB/identità BFM | identità in 300 GB (ICT / BFM) | dz max | ρ var ICT / BFM | ρ GT ICT / BFM | note |
|---|---|---|---|---|---|---|---|
| ref (oggi) | 61.0 | 152.0 | 4.920 / 1.970 | 0 | 1 / 1 | 0.9698 / 0.6962 | |
| **i32**: facce e indici int32, npz compresso | 37.0 | 90.8 | 8.100 / 3.300 | **0** | 1 / 1 | identico | senza perdita |
| e16: i32 + `evecs` fp16 | 22.4 | 54.0 | 13.400 / 5.560 | 0.0031 | 0.999994 / 0.99994 | 0.9698 / 0.6954 | perdita minima |
| a16: e16 + valori di L e gradienti fp16 | 20.2 | 48.7 | — | 0.0043 | — | — | **overflow**: valori oltre 65504 diventano inf. Scartata |
| k64 (ricalcolati con k_eig 64) | 22.4 | 54.5 | 13.400 / 5.500 | **0.50** | 0.977 / 0.926 | **0.9539 / 0.6464** | cambia l'output del modello |
| k64 + e16 | 15.1 | 36.1 | 19.900 / 8.300 | 0.50 | 0.977 / 0.926 | 0.9539 / 0.6456 | come k64 |
| rec128: operatori **ricalcolati ora**, k_eig 128 | 37.0 | 90.8 | (come i32) | **1.4e-6** | 1 / 1 | identico | riproduce i file in uso |
| solo geometria (V fp32 + F int32, compresso) | **1.18** | **2.74** | **254.000 / 109.000** | — | — | — | operatori da ricostruire per job |

Tempo del loader congelato per campione, ICT, su `/tmp`, cache calda [M]: ref 55 ms, i32 **non**
compresso 51 ms (8.2 MB/mesh), i32 compresso 91 ms (6.2 MB/mesh). La compressione costa quindi
+36 ms per campione in lettura.

Calcolo degli operatori su CPU, k_eig 128, processo singolo [M] (nodo `cpu` a512-mi100-01,
condiviso con un altro job):

| | original | remesh | crop | noisy | down8k | up60k | **per identità** |
|---|---|---|---|---|---|---|---|
| ICT, s | 3.66 | 2.36 | 3.41 | 3.51 | 1.04 | 10.25 | **24.2** |
| BFM, s | 10.96 | 7.42 | 9.74 | 11.00 | 3.28 | 26.25 | **68.7** |

Con k_eig 64: 15.5 s per identità ICT e 43.5 s BFM [M]. Throughput ICT con N processi [M]:
N=1 0.27 mesh/s, N=4 0.91, N=8 1.33, N=16 **1.67**. Scala meno che linearmente: N=16 rende
6.2 volte N=1.

Rigenerare la geometria dai soli pesi d'identità (2 identità ICT-5000) [M]: `original`, `crop`,
`remesh` e `noisy` coincidono con il disco entro 1e-5. **`down8k` e `up60k` no**: facce diverse,
e per ict4501 `up60k` ha un numero diverso di vertici. La decimazione di igl non è
riproducibile fra esecuzioni, quindi la geometria decimata va salvata, non rigenerata.

## 2. Le tre opzioni

Riferimento per i tempi [D]: il congiunto 1019532 (4400 soggetti di training × 6 = 26.400 mesh
per epoca) è durato 47 h 20 min per 120 epoche (sacct, staging ed eval compresi). Sono al più
**23.7 min per epoca, circa 18.6 mesh/s** consumate dalla L40S. Il trainer v1 chiama
`dataset[idx]` dentro il ciclo del batch, senza worker.

### (a) Operatori in fp16 e/o k_eig ridotto

- **i32 (senza perdita)**: −39% su ICT e −40% su BFM rispetto a oggi. Output identico, dz = 0
  [M]. Se letto compresso aggiunge +36 ms per campione [M], cioè circa +16 min per epoca del
  congiunto [E], perché il loader è sul percorso critico. Decompresso su `/tmp` allo staging
  costa 0. Rischio per i risultati già ottenuti: **nullo**.
- **e16 (evecs fp16)**: −63% / −64%. Le distanze cambiano al più dello 0.3-0.5%, ρ GT si muove
  di −0.00003 (ICT) e −0.0008 (BFM) [M]. Costo per epoca: 0. Rischio: piccolo ma non nullo. Va
  bene per dati **nuovi** di training; non va usato per ricodificare insiemi di valutazione già
  pubblicati.
- **k64**: −63%, ma con il modello attuale l'output cambia davvero: dz mediano 0.42, ρ GT BFM
  0.696 → 0.646 [M]. Richiederebbe di riaddestrare e renderebbe incomparabili tutti i numeri
  già ottenuti. **Sconsigliato.**
- fp16 sui valori sparsi: overflow [M]. **Scartato.**
- Anche nel caso migliore (e16), 300 GB contengono **13.400 identità ICT**, circa 2.4 volte
  ICT-5000. Da sola, (a) **non arriva a 10x**.

### (b) Operatori calcolati al volo

- **Nel DataLoader, a ogni epoca**: per stare al passo della GPU servono 18.6 mesh/s [D], ma 16
  processi ne danno 1.67 [M]. L'epoca diventerebbe **circa 11 volte più lenta** [E]: circa 4.4 h
  invece di 24 min. **Non praticabile.**
- **Pre-pass a inizio job su `/tmp`**: 24.2 CPU-s per identità ICT e 68.7 per BFM [M]. Al ritmo
  misurato con 16 processi (1.67 mesh/s), 5000 identità ICT (30.000 mesh) richiedono circa
  **5 h di pre-pass per job** [E]. Con più core sulla stessa macchina si scende, ma la scalatura
  è sublineare [M], quindi qualunque numero oltre N=16 è [E]. Il pre-pass non rallenta le epoche
  se scrive i32 **non compresso**, che il loader legge in 51 ms contro i 55 di oggi [M].
- Rischio per la riproducibilità: **nullo in pratica**. Il ricalcolo riproduce gli operatori in
  uso con dz ≤ 1.4e-6, e ρ GT è identico [M]. La ricostruzione è deterministica a meno del
  float32, anche se eigsh parte da un vettore casuale.
- Limite vero: la RAM del nodo. `/tmp` conta contro `--mem`; L40S ha 735 GB di RAM e circa
  378 GB di `/tmp` [D]. In i32 non compresso un'identità ICT occupa 49 MB [M], quindi un job
  ospita al più circa 6.000-7.000 identità ICT con operatori [E], lasciando RAM al training.

### (c) Solo geometria nella home, operatori ricostruiti per job

- **1.18 MB per identità ICT e 2.74 BFM** [M]: 300 GB contengono 254.000 identità ICT [E]. Lo
  spazio smette di essere il vincolo: **50.000 identità ICT nuove (10x) occupano circa 59 GB**
  [E], in shard tar invece di 300.000 file piccoli. La GT non costa spazio: è la vertex-mean-L2
  fra le `original` normalizzate maxabs, calcolabile dalla geometria salvata.
- Costo e rischio sono quelli del pre-pass (b): circa 24 CPU-s per identità ICT per job [M] e
  riproducibilità a 1e-6 [M]. La geometria va **salvata**, non rigenerata dai pesi: la
  decimazione non è riproducibile [M].
- Il collo di bottiglia si sposta su RAM del nodo e CPU per job: con 10x i dati non stanno
  tutti in un job (vedi §3).

## 3. Cosa vale per tutte le opzioni: l'epoca

Il costo di un'epoca cresce con il numero di mesh per epoca, indipendentemente dal formato. 10
volte le identità vuol dire circa 4 h per epoca, e 120 epoche sarebbero circa 20 giorni su una
L40S [E, dai 23.7 min per epoca [D]]. Per scalare i dati serve quindi una decisione sul trainer,
fuori dallo scope di questo studio: **lo stesso numero di passi** su un insieme più grande
(meno epoche) e lo staging a blocchi, per esempio blocchi di circa 5000 identità ICT, ciascuno
con il proprio pre-pass su `/tmp`. Oggi già il congiunto da 5500 soggetti non entra nella cache
in RAM di `train_fast.py` (`aau/cross3dmm/joint_E_prep.py:5`) [D].

## 4. Raccomandazione

1. **Nuovi dati: opzione (c).** Nella home solo geometria compressa, in shard tar (circa 59 GB
   per 50.000 identità ICT nuove, con GT ricavabile dalla stessa geometria). Gli operatori si
   calcolano in un pre-pass CPU per blocco, scritti su `/tmp` in i32 non compresso, con k_eig
   128 e fp32: risultati identici a oggi (dz 1.4e-6) e nessun rallentamento del loader. e16 solo
   se la RAM del nodo diventa il vincolo, e solo sui dati nuovi di training.
2. **Liberare spazio senza perdita, dopo che i training in corso hanno finito lo staging:**
   ricodificare in i32 compresso gli insiemi oggi non compressi. `ICT/topo_withops` passa da
   305 a circa 185 GB, `REMESH/*_areanorm` e `*_pot055` da 76 a circa 45 GB ciascuno,
   `ICT/expressions_withops` da 70 a circa 42 GB [E, rapporto i32/ref misurato 0.61]. Sono circa
   **210 GB recuperati con output del modello identico** (dz = 0 [M]). Costa +36 ms per campione
   se letti compressi, quindi lo staging deve decomprimere. La decisione spetta al PI: gli
   insiemi sono usati da job in corso.
3. **Da non fare:** k_eig 64, perché cambia i numeri già ottenuti (ρ GT BFM −0.05); fp16 sugli
   operatori sparsi, perché va in overflow; calcolo al volo a ogni epoca, 11 volte più lento;
   rigenerare le topologie dai pesi, perché la decimazione non è riproducibile.

## 5. Composizione e held-out (dalla fase 1, ancora valide)

- **BFM: oggi non si possono generare identità nuove nello stesso dominio.** Le 500 mesh REMESH
  **non** stanno nel sottospazio 40+10 del BFM di 3DDFA, l'unico sul cluster nella topologia a
  53215 vertici: il residuo relativo mediano è 0.22-0.24, contro 0.0036 del controllo
  (`probe_bfm_basis.py`, 60 mesh) [M]. Il generatore originale (`render3d_Leonardo/...`, 4999
  mesh `GT_ready`, di cui esiste la GT 4999×4999 in `normalized_matrix_distances.npz`) e il
  modello BFM2009 completo non sono sul cluster. **Serve dall'autore**: le 4999 mesh GT_ready,
  oppure `01_MorphableModel.mat` più i parametri di campionamento. Nello snapshot HF c'è
  `model2017-1_bfm_nomouth.h5` (BFM2017), ma in un'altra topologia e senza mappa p23470: sarebbe
  di fatto un altro 3DMM. Lo snapshot contiene anche `FLAME_*.pkl`, che STATUS.md:97 dichiara
  non originali; FLAME resta comunque dominio di test.
- **ICT**: identità nuove con N(0,1) sui 100 modi, come ICT-5000, ma con semi diversi da 1234;
  id da 20000 in su, nessuna collisione con BFM 0-499, FLAME 1000-5999, ICT 10000-14999. Il
  rumore di `noisy` va seminato sull'id intero: `make_ict_topologies` usa `int(subject[-4:])`,
  che darebbe a `id20000` lo stesso campo di rumore di `ict0000`. Le espressioni usano il
  campionatore di WS5 (3-8 dei 45 blendshape non di sguardo), con la GT dell'identità.
- **Held-out, non negoziabile**: lo split dei vecchi soggetti va **congelato e passato
  esplicitamente**. `rebuild_subject_split` ricalcola lo split sull'unione dei soggetti della
  vista, quindi aggiungere identità sposterebbe soggetti oggi di test nel training. Le identità
  nuove vanno tutte in training (più eventualmente un loro held-out nuovo). Prima di ogni
  training: controllo che nessun soggetto held-out esistente sia nella vista, e controllo di
  quasi-duplicati fra le nuove identità e le 5000 di ICT-5000. HIFI3D e FLAME restano fuori.

## 6. Esecuzione (5 ottobre, pomeriggio: quota 2 TiB, opzione (c) approvata)

### Dati generati [M]

- `datasets/ICT_SCALE/shards/`: **200 shard, 50.000 identità ICT nuove (id20000-id69999), 700.000
  mesh**. Ogni identità ha le 6 topologie più **8 espressioni** (`rexpr1..8`, topologia
  `original`, campionatore di WS5). Sola geometria, **132.7 GB** (2.65 MB per identità). Ho scelto
  8 espressioni perché stanno nel budget di 150 GB con un margine del 12%: con 10 si arriverebbe
  a circa 150 GB.
- Semi `SeedSequence([seme, g])`, con semi di famiglia 20261005/6/7, diversi dal 1234 di ICT-5000 e
  di WS5. Il rumore di `noisy` è seminato sull'id intero.
- Tempo [M]: 118 s per shard da 250 identità con 12 core (mediana; minimo 114, massimo 123);
  1.68 CPU-s per identità. L'array 1055827 (10 task da 20 shard, due alla volta) ha impiegato
  circa 3 h 20 min.
- **Held-out congelato**: `heldout_frozen.json` contiene 339 soggetti BFM e 2302 ICT, cioè
  l'unione degli held-out di ogni training i cui numeri esistono (15 fonti, elencate nel file).
  `heldout_ict_originals.npz` contiene le loro `original`. La guardia sta dentro
  `gen_ict_shard.py`: se fallisce, lo shard non viene scritto. Controlla tre cose: i nomi contro
  la lista e il range riservato; i quasi-duplicati (vertex-mean-L2 verso ogni held-out ICT sopra
  0.0055, metà del vicino più prossimo minimo di ICT-5000); la validità delle mesh. Su tutti i
  200 shard nessuna identità è sotto soglia; il vicino più prossimo minimo è 0.0107.
- La stessa lista blocca il trainer (`--frozen-heldout`). Verificato: lanciato con lo split v1
  del seme 1234, il run si ferma con "239 soggetti di test congelati nel training".

### Mini-lotto verificato (job 1055815, `minilot/check.json`) [M]

24 identità (id20000-id20023) × 14 mesh = 336 mesh.
- **Mesh**: nessun errore. Topologie fisse sulla tabella di facce di ICT-5000, triangoli entro il
  2% dei target.
- **Operatori**: il pre-pass (`prepass_ops.py`) e `v2_work/potential/areanorm_operators.py`,
  eseguito così com'è sulla stessa geometria e letti entrambi dal loader congelato, coincidono.
  Scarto 0 su vertici, massa, autovalori, L e gradienti; autovettori 4.6e-8 a meno del segno;
  embedding del modello congiunto 1.6e-6 relativo.
- **GT**: mediana fra le nuove 0.0378 contro 0.0398 di ICT-5000, p1 0.0207 contro 0.0205. Il
  vicino più prossimo verso ICT-5000 è al minimo 0.0134, contro 0.0111 interno a ICT-5000. La
  funzione GT ridà la matrice in uso a 1.4e-7.
- **Modello**: per il 100% delle mesh la distanza latente media dentro l'identità (0.44) è
  minore di quella verso le altre identità (1.35).
- Pre-pass: 3.95 CPU-s per mesh ICT, espressioni comprese.

### GT estesa

`build_gt.py` (job 1056223, su GPU) scrive `datasets/ICT_SCALE/gt_joint_bfm_ict.npz`: BFM 500 + ICT
55.000, coppie fra domini a NaN, blocco BFM invariato. Il blocco ICT era diviso per il proprio massimo;
poi `rescale_gt.py` l'ha riportato alla scala della GT in uso (vedi le decisioni del PI qui sotto).
Se le coppie ricalcolate non coincidono con la GT in uso, il file non viene scritto.

### Trainer a passi fissi: `v2_work/fastio/train_steps.py`

Wrapper accanto a `train_fast.py`, che resta invariato. Nessun file di ricerca modificato. Senza
`--total-steps` chiama `train_fast.main()` e basta. Con `--total-steps T`:
- **Passi fissi**: `--steps-per-epoch S` (0 = lunghezza naturale dell'epoca v1) e ceil(T/S)
  "epoche", che restano la granularità di log, eval online, scheduler e checkpoint. I passi sono
  contati sugli `optimizer.step()` reali. Con un solo blocco e S naturale, la funzione d'epoca
  riceve la stessa lista di soggetti di v1.
- **Split esplicito** (`--split-json`) al posto di `rebuild_subject_split`, più la guardia
  `--frozen-heldout`.
- **Blocchi** (`--data-spec`): i soggetti di training sono divisi in K blocchi. Ogni blocco passa
  dal pre-pass su `/tmp` (tar, geometria o viste esistenti), finisce nella cache RAM di
  `train_fast` e poi viene cancellato da `/tmp`. Il blocco successivo si prepara in un processo a
  parte mentre si addestra quello corrente. L'eval online è preparato una volta sola.
- **Multi-dominio**: `--domain-blocked --eval_domain` entra nelle patch di `train_v2`. Con il
  batching a dominio singolo, ogni epoca riceve un multiplo di `batch` soggetti per dominio.
- Ricetta invariata: modello, loss, ottimizzatore, scheduler, augmentation, campionamento delle
  mesh, eval.
- Misurato nella prova [M]: pre-pass BFM 12.5 CPU-s per mesh (1200 mesh in 501 s con 30
  processi). Il blocco 1, preparato in background, era pronto al cambio (attesa 0 s).
- Primo OOM: il picco era cache più operatori su `/tmp`, 120G non bastavano. Corretto
  cancellando il blocco da `/tmp` dopo il caricamento in cache.
- Composizione: con 8 espressioni ogni soggetto ICT nuovo ha 14 etichette, e v1 ne pesca 6 per
  passo, quindi circa il 43% delle mesh viste sono espressioni. `labels` nella data-spec decide la
  composizione: è una scelta del PI.

### Prova di riproduzione del trainer [M]

Dati di oggi: BFM, 500 soggetti, operatori ad area unitaria (braccio ctrl). Ricetta v1 identica a
`train_joint_E.sbatch` senza le patch multi-dominio; 480 passi (6 epoche v1); split esplicito =
lo split v1 del seme 1234. Valutazione fissa per tutte le corse: ultimo checkpoint, 100 held-out
× 6 topologie, Spearman latente-GT (`trainer_repro_compare*.py`).

- Giro 1 (1055816): v1 0.518, v1b (v1 ripetuto identico) 0.511, id (train_steps, stessi passi)
  0.495, seed (seme 2345) 0.497, blk 0.471. Una sola coppia di rumore non basta: serviva il giro 2.
- Giro 2 (1055892, 1056033, 1056034; confronto 1056035, `trainer_repro2.json`): tre semi.
  - **A** = percorso v1: 0.509 / 0.480 / 0.489, media 0.493 ± 0.015 fra semi.
  - **P** = pre-pass dalla geometria, un blocco: differenza appaiata con A +0.017 / −0.035 /
    −0.017 (media −0.012).
  - **B** = pre-pass, 2 blocchi: differenza appaiata −0.041 / −0.037 / +0.017 (media −0.020).
  - Cross-topologia: stesso andamento (P −0.012, B −0.019).
- Lettura: P esegue gli stessi calcoli di A (operatori identici a 1e-6, stessa lista di soggetti
  per epoca), quindi le sue differenze, ±0.035, sono il **rumore da corsa a corsa** a parità di
  seme: non-determinismo GPU amplificato in un training a 480 passi, ancora ripido. **Il trainer
  adattato riproduce il training attuale entro questo rumore** (P e B entro 2 sd di A).
- Limite onesto: per B due semi su tre sono a −0.04. Con 3 semi il test esclude solo effetti dei
  blocchi più grandi di circa 0.04, e su 480 passi, dove l'ordine conta di più: ogni soggetto è
  visto 3 volte di fila nel suo blocco invece di 6 volte distribuite. Su un training lungo con
  molti blocchi l'effetto andrebbe rimisurato.

### Cosa manca per un training grande

1. **GT estesa: fatta** (1056223, 40 min). `datasets/ICT_SCALE/gt_joint_bfm_ict.npz` (12.3 GB,
   55.500 × 55.500 float32). Il primo tentativo (1055828) è morto per OOM sulla GPU: `pairdist`
   converte la matrice da 55.000² a float64 sulla GPU, 22.5 GiB in più. Corretto in
   `build_gt.py`: righe sull'host in float32, tar letti con 8 thread (469 s contro 2183).
   Controllo: le coppie di ICT-5000 ricalcolate coincidono con la GT in uso a 2.4e-7 [M].
   **Da decidere (PI)**: il massimo del blocco ICT sale da 0.1718 a 0.2038 [M], quindi le coppie
   di ICT-5000 valgono 0.843 volte quelle del congiunto attuale. La rank loss non ne risente; la
   stress loss sì. L'alternativa, cioè tenere la vecchia scala, porterebbe il massimo globale a
   1.19 e `load_gt_distance_matrix` riscalerebbe tutto, BFM compreso.
2. **Decisione sullo split** (PI). La lista congelata è l'unione degli held-out di tutti i run:
   lascia in training solo **161 soggetti BFM su 500** e 2698 di ICT-5000; le 50.000 nuove sono
   tutte di training. L'alternativa è congelare solo gli held-out dei modelli contro cui si
   confronterà il nuovo run.
3. **Composizione** (PI): con 8 espressioni circa il 43% delle mesh viste da un soggetto nuovo
   sono espressioni; `labels` nella data-spec lo regola.
4. **Smoke test del percorso congiunto**: `--domain-blocked` (batching a dominio singolo con i
   sottoinsiemi per dominio) e la data-spec da tar non sono stati esercitati nella prova, che era
   BFM-only da directory di geometria. Il pre-pass da tar è verificato nel mini-lotto.
5. **Dimensionamento** [E]: cache circa 8 MB di RSS per campione ICT, 14 mesh per identità, quindi
   circa 2.500 identità per blocco con 300 GB, cioè circa 22 blocchi per 55.000 identità. Pre-pass
   di un blocco: 2.500 × 14 × 3.95 CPU-s ≈ 38 CPU-h, circa 1.3 h con 32 processi. Con gli stessi
   passi del congiunto attuale (105.600) sono circa 4.800 passi per blocco, cioè un'ora e mezza
   di GPU per blocco: il pre-pass del blocco successivo sta dentro.
6. Sbatch del training grande: run dir su `/tmp` con sync verso la home, come `train_joint_E.sbatch`.

### Decisioni del PI (5 ottobre sera) e loro attuazione

1. **Split**. `heldout_frozen.json` ha ora la politica `joint_compare`: held-out del congiunto
   1019532 (BFM 108, ICT 992) più i 100 BFM del protocollo standard, cioè **189 BFM e 992 ICT**.
   Motivazione e modelli di confronto sono nel file. L'unione dei 15 run resta in
   `heldout_frozen_union15.json`: è quella con cui sono stati controllati gli shard, più severa.
   **Sostituita il 6 ottobre** dalla politica `joint_exact` (vedi sotto). La guardia del trainer
   resta attiva.
2. **Scala della GT**. `rescale_gt.py` riporta il blocco ICT alla scala di ICT-5000 (fattore
   1.18585). Le coppie del congiunto in uso tornano identiche a 3e-7 [M]; il massimo globale è
   1.186. **`--gt-keep-scale`** fa leggere al trainer la matrice così com'è nel file, al posto di
   `load_gt_distance_matrix`, che divide ogni voce per il massimo globale: senza il flag tutte le
   distanze, BFM comprese, sarebbero divise per 1.186. Coincide con la lettura v1 solo quando il
   massimo del file è esattamente 1, com'è per `JOINT_BFM_ICT/gt_matrix.npz`. Il trainer si ferma se
   il manifest della GT dichiara massimo > 1 e il flag manca.
3. **Espressioni al 25%**: `label_groups` fonde rexpr1-4 in rexprA e rexpr5-8 in rexprB. Un
   soggetto nuovo ha così 8 etichette, di cui 2 d'espressione. La frazione simulata col
   campionatore v1 è **0.250** [M, smoke].
4. **Frame**, spento di default.
   - `canon`: matrice per dominio applicata alla geometria prima degli operatori nel pre-pass, con
     l'ordine dei vertici delle facce invertito se il determinante è negativo. Verificato su BFM
     specchiato x->-x (`check_canon.py`): autovalori uguali a 1e-14, massa identica, vertici e
     normali specchiati [M]. Con `canon` BFM passa dalla geometria e non dalla vista, e la cella
     BFM della catena di eval non viene lanciata (le viste hanno il frame vecchio).
   - `aug`: rotazioni e riflessioni dei vertici di training, solo sul canale xyz come le
     perturbazioni della ricetta.
5. **Smoke** del percorso congiunto [M]:
   - **S1** (job 1056247): default del run grande. BFM e ICT-5000 dalle viste, ICT nuove lette dai
     tar dentro il trainer, `--domain-blocked --eval_domain bfm`, GT a scala invariata, guardia
     (1181 congelati, OK), BFM residente con il 9% dei passi, 2 blocchi. 60/60 passi, rc 0, 24.5
     min. Pre-pass del primo blocco 469 s (3500 mesh nuove con 30 processi, 4.0 CPU-s per mesh);
     il secondo era pronto al cambio (attesa 0 s). RSS massimo 62 GB.
   - **S2** (job 1056248): canon BFM x->-x più aug (180 gradi, riflessione 0.5), 1 blocco. 30/30
     passi, rc 0, RSS massimo 69 GB. Che l'aug abbia effetto l'ho verificato solo per via
     indiretta (loss più alta), non misurato.
6. **Run lungo**: `train_scale.sbatch`, `eval_chain_scale.sbatch`, `launch_scale.sh`. **Non
   lanciato.** 105.600 passi (come il congiunto); S=220 con eval ogni 8 epoche e patience 32,
   cioè le stesse cadenze in passi della ricetta. 40 blocchi, 360G, 72 h. La catena di eval è
   provata a vuoto (`WBES_DS_DRY=1`) sul checkpoint di S1.

### Revisione del critic (6 ottobre, BLOCCANTE) e correzioni

- **D1**: la geometria estratta per il pre-pass (`blockNNN_geom`) restava su `/tmp`. Ora la cancella
  `prepass_ops.run` a fine pre-pass, e il trainer la cancella anche al cambio di blocco.
- **D2**: il tetto della cache era proiettato da un solo campione, sempre BFM (~20 MB). Ora
  `project_cache_gb` usa 12 campioni misti per dominio, presi lungo la lista; il controllo di
  `fast_data` viene scavalcato.
- **R1, lr**: fissato in passi (`--lr-steps 81747:5e-5`), con ReduceLROnPlateau disattivato.
  Misurato sul log del congiunto 1019532: le epoche sono da **879** passi (tqdm `/879`), 120 epoche
  fanno **105.480** passi, e la riga "Epoch 093" stampa già 5e-5. Il lr però è letto **dopo**
  `scheduler.step` (`train_runner.py:1735-1736`), quindi l'epoca 93 è stata fatta a 1e-4: 1e-4 per
  i passi 1-81.747 (93 × 879), 5e-5 dal passo 81.748. Il valore 80.868 = 92 × 879 indicato dal
  critic corrisponde a leggere il lr stampato come quello usato durante l'epoca.
- **R1, epoca**: S = 293 = 879/3. Eval ogni 6 epoche = 1.758 passi, come il congiunto; 360 epoche
  esatte.
- **R2**: politica `joint_exact`, cioè esattamente gli held-out del congiunto (108 BFM, 992 ICT). In
  training 392 BFM e 4008 ICT-5000, uguali ai 4400 del congiunto (verificato con un assert in
  `make_scale_split.py`), più le 50.000 nuove. Dei 100 BFM standard, 81 sono di training anche per
  il congiunto (documentato in `heldout_frozen.json`). L'eval online usa i 16 soggetti del
  congiunto, passati espliciti (`online_eval` nello split). Quota BFM 26/293 = 78/879.
- **R3**: 8 thread di torch al training (`--train-threads`, `OMP_NUM_THREADS`) e 22 processi al
  pre-pass. Il trainer stampa s/passo per epoca, segnalando se il pre-pass era in corso. Rimisurato
  nello smoke S3.
- **R4**: `--no-requeue`. La run dir prende il suffisso `_restartN` se `SLURM_RESTART_COUNT` > 0, e
  `_rerunN` se contiene già un run. La catena di eval legge la run dir effettiva.
- **Guardia GT**: il trainer si ferma se la GT ha massimo > 1 e manca `--gt-keep-scale`.
- **Frame**: Rx(180) = diag(1,−1,−1) più `flip_faces`, indipendente dal determinante
  (`prepass_ops.apply_frame`). Verificato (`check_canon.py`):
  - BFM ha frontale −z e normali verso l'interno; ICT frontale +z e normali verso l'esterno;
  - dopo il canon BFM ha frontale +z e normali verso l'esterno, come ICT;
  - autovalori uguali a 1e-14, massa identica.
  Il canon resta spento di default.

### Smoke S3 dopo le correzioni (job 1056268) [M]

Configurazione come il run grande: 8 thread di training, 22 processi di pre-pass, lr a passi fissi
(confine di prova 300), quota BFM 78/879, eval online del congiunto, GT estesa con
`--gt-keep-scale`, guardia attiva. 600 passi su 2 blocchi da 320 soggetti.
- 600/600 passi, rc 0, 27 min. Lr da 1e-4 a 5e-5 esattamente al passo 301.
- **Tempo per passo**: 1.28-2.33 s con il pre-pass del blocco successivo in corso (epoche 1-20,
  mediana circa 1.4), 0.70-0.79 s senza (epoche 21-40). Senza limite di thread erano circa 35 s
  (smoke S1).
- Pre-pass: 3542 mesh in 514 s con 22 processi (3.2 CPU-s per mesh). Il blocco 1 era pronto al
  cambio con 8 s di attesa.
- **RAM**: MaxRSS del job 64 GB. Il cgroup arriva a 142.5 GB di picco, ma comprende la page cache,
  che è recuperabile.
- **/tmp**: picco 19.9 GB e ritorno a 0 dopo ogni caricamento in cache, quindi la geometria non si
  accumula più.
- Proiezione della cache su campione misto: ICT 9.7 MB, BFM 51.1 MB per campione; 47 GB per 3896
  campioni.
- Dimensionamento del run lungo, ricavato da qui [E]: `--cache-max-gb 330` (proiezione di un blocco
  circa 295 GB) e `--mem 420G`. Circa 61 minuti per blocco, circa 41 ore in tutto.

### Secondo giro del critic (6 ottobre, BLOCCANTE sulla memoria) e correzioni

- **B1, cache esatta**: `exact_cache_gib` (trainer) somma i byte di ogni campione del blocco dagli
  header npy di TUTTI i file, con la stessa contabilità di `fast_data._sample_bytes`. Il
  campionamento con `linspace` cadeva quasi sempre su `up60k`.
  Per i 40 blocchi del run grande, prima che esistano, `cache_budget.py blocks` usa la stessa
  partizione (`partition_blocks`): header per le viste, indice dei tar per le mesh nuove, con k = 128
  e nnz(L) = nnz(gradX) = nnz(gradY) = n + 2E. Nello smoke S4 la formula coincide con gli header reali
  (31.172 e 31.930 GiB, identici) [M].
  **Massimo sui 40 blocchi: 220.4 GiB** (blocco 30), minimo 218.0, media 218.9 [M, job 1056278].
  Tetto `--cache-max-gb 240`.
- **B2, pinning**: spento nella cache dei blocchi (`--pin-cache` per riaccenderlo).
  - S4 [M, job 1056277]: picco del cgroup 107.2 GiB (S3 con pinning: 142.5); picco anon 85.1, shmem
    19.4, page cache 21.5.
  - Costo del non pinning [M, nodi diversi, una corsa per lato]: senza pre-pass 0.73-1.11 s/passo
    (media 0.875) contro 0.70-0.79 di S3 (media circa 0.74), circa +19%; con il pre-pass in corso
    1.26-1.91 contro 1.28-2.33.
  - Bilancio del run grande dal cgroup, sola memoria non recuperabile:
    - anon = cache 220.4 + eval 2.3 + GT float64 22.9 + resto 20.8 (da S4) = 266.4 GiB;
    - shmem (`/tmp`) del blocco successivo = 5.67 MiB per mesh (S4) × 17.500 = 97 GiB;
    - totale circa 363 GiB; `--mem 430G` = +18%.
  - Verifica su blocchi di grandezza reale: smoke **S5** (job 1056296, V100, `--mem=430G`), 2 blocchi
    da 1742 soggetti (quelli dei blocchi 0 e 1 della partizione reale), 2930 passi. **In coda**: lo
    scheduler ne stima la partenza al 7 ottobre alle 00:42, perché nessun nodo ha 430G liberi
    prima. Finché non gira, i 430G restano una stima dalle misure, non una verifica.
- **R1**: i `blockNNN.prepass.log` (e le liste dei soggetti) sono copiati in
  `<run>/prepass_logs/` a ogni sync e in uscita, anche quando il job fallisce. Verificato a parte.
- **R2**: indice dei tar `datasets/ICT_SCALE/shards/index.npz`: 700.000 membri, una scansione in
  5.4 min [M]. Trainer e pre-pass leggono i nomi dall'indice e i membri per offset.

## 7. Stato degli script in `aau/data_scale/`

| file | stato |
|---|---|
| `probe_bfm_basis.py`, `measure_npz.py`, `study_options.py`, `loader_timing.py` | studio, eseguiti (1055670, 1055672, 1055753, 1055786) |
| `freeze_heldout.py` -> `heldout_frozen.json`, `heldout_ict_originals.npz` | eseguito (1055812) |
| `gen_ict_shard.py`, `gen_ict_array.sbatch` | eseguiti: mini-lotto 1055815, array 1055827 |
| `prepass_ops.py` | usato dal mini-lotto e dal trainer |
| `check_shard.py`, `minilot.sbatch` | mini-lotto verificato (`minilot/check.json`) |
| `build_gt.py`, `build_gt.sbatch` | job 1055828 |
| `trainer_repro*.sbatch`, `trainer_repro_compare*.py` | prova di riproduzione del trainer, sezione 6 |
| `v2_work/fastio/train_steps.py` | wrapper del trainer |
| `rescale_gt.py`, `make_scale_split.py`, `check_canon.py`, `recipe_v1.sh`, `smoke_scale.sbatch` | decisioni del 5-6 ottobre, eseguiti (smoke S1/S2/S3) |
| `cache_budget.py` -> `shards/index.npz`, `cache_blocks.json`, `spec_scale_default.json` | indice dei tar e memoria esatta dei 40 blocchi |
| `train_scale.sbatch`, `eval_chain_scale.sbatch`, `launch_scale.sh` | run lungo, **non lanciato** |
