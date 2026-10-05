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

`build_gt.py` (job 1055828, su GPU) scrive `datasets/ICT_SCALE/gt_joint_bfm_ict.npz`: BFM 500 + ICT
55.000, coppie fra domini a NaN, blocco BFM invariato, blocco ICT diviso per il proprio massimo
(la convenzione di ogni file GT del repo). Il fattore di riscalatura delle coppie di ICT-5000 va
nel manifest accanto al file. Se le coppie ricalcolate non coincidono con la GT in uso, il file
non viene scritto.

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
