# Ablazione 2x2 dello stream: FLAME 2023 come quarto generatore x parzialita' variabile (protocollo, PRIMA dei numeri)

Scritto l'11 ottobre 2026 dal coder, su richiesta del PI, PRIMA di lanciare i quattro training e quindi prima di
qualunque embedding, Spearman o delta di questa ablazione. L'impronta sha256 di questo file sta nel messaggio del commit
che lo introduce e in `PROTOCOL.sha256`; ogni modifica va in un emendamento datato, scritto prima dei numeri che tocca.
Codice: `v3_work/stream/slurm/ablation_2x2.sbatch` (training), `v3_work/stream/eval_ablation.py` e le sbatch della
catena di valutazione (scritti dopo il lancio, prima dei numeri); evidenze in `aau/runs/evidence/stream/ablation_2x2/`.

**Cosa esisteva gia' al momento della scrittura** (nessun numero di questa ablazione):
- la ricetta c3m dello stream (`v3_work/stream/slurm/massive_node.sh`) e il mini-smoke a 2 L40S (job 1067709,
  `smoke_c3m_recipe/node0/summary.json`): 0.435-0.447 s/passo, GPU al 76%, 6.1 viste fresche/s, riuso a regime 20.8;
- la parzialita' variabile (`v3_work/stream/partial_aug.py`, commit efffd74) con i suoi controlli
  (`partial_aug/partial_check.json`: 0 viste invalide su 432, perdita d'area mediana 0.17, 0.16 s per applicazione
  contro 3.2 s di operatori per vista);
- le letture che motivano l'ablazione: `aau/runs/evidence/diagnostics/results.md` (D1/D2), `aau/runs/evidence/e1/
  summary.md`, `aau/runs/evidence/baselines_param/conclusions.md` (emendamento 3, crop), `paper/BOARD_DIARY.md`.

## 1. Domande

Testa `factorized` (d_F^2 = (S_i - S_j)^2 + S_i S_j (c d_P)^2, d_P = ||u_i - u_j|| x dp_per_unit).
- **Q1, FLAME:** aggiungere un quarto generatore, FLAME 2023 Open (CC BY 4.0), al training dell'encoder migliora la
  generalizzazione alle famiglie mai viste? Motivo: sui generatori visti la metrica legge la forma (D1: d_P contro SR
  0.90-0.93), su FLAME 2023 mai visto cala (0.806); in E1 passare da 2 a 3 generatori aiuta, 10 volte le identita'
  (C3M) no.
- **Q2, parzialita':** la parzialita' variabile per vista riduce il crollo sulle righe col crop (factorized d_F cal.
  con FR su FaceScape dev da 0.661 / 0.669 senza crop a 0.351 / 0.278 sulle righe col crop, `paired_e2.csv`), senza
  costare senza crop?

Le famiglie di test (FaceScape dev bilineare, AI-NEXT "HIFI3D", FaceVerse) sono campioni di 3DMM, non dati reali:
"generalizzazione" qui vuol dire a famiglie di 3DMM mai viste.

## 2. Disegno

| cella | fonti dello stream (`STREAM_SOURCES`) | parzialita' (`STREAM_PARTIAL`) |
|---|---|---|
| A (controllo) | `validated` = bfm2019, ict, gnm | spenta |
| B | `validated,flame2023` = bfm2019, ict, gnm, flame2023 | spenta |
| C | `validated` | p = 0.5 per vista |
| D | `validated,flame2023` | p = 0.5 per vista |

**Identico nelle quattro celle** (ricetta c3m di `massive_node.sh`, `STREAM_RECIPE=c3m` esplicito in tutte):
- testa `factorized`, k_eig 128, ingresso globale, `--area robust --area-robust smooth`, dropout 0, `--scale-aug
  0.8,1.25`, GT-SR al volo (`--stream-gt sr`), nessuna maschera di taglia, forward `groups`, EMA 0.999;
- lr 1e-4 costante; batch 5 identita' x <= 6 viste per rank, 2 rank (2 L40S su un nodo): 10 identita' per passo;
  batch a dominio singolo, passi uguali fra i domini in ogni epoca (S = 600: 200 passi per dominio con 3 domini, 150
  con 4);
- produttori: 6 viste per identita', discretizzazioni senza reinserimento (`perm`), nessuna rotazione a ogni uso,
  nessun moltiplicatore (mm_aug), riuso con tetto morbido 4, anello 10 GiB per GPU, domini uniformi (alpha 0);
- semi: trainer 1234; produttori 20261009 (+ 1000 x nodo + 100000 x segmento, come `massive_node.sh`);
- risorse: 2 L40S, 20 CPU e 80 GB per GPU (8 CPU per rank al trainer, le altre ai produttori), partizione
  `prioritized`, QoS normal, `--requeue`, `STREAM_PREEMPTIBLE=1` (default), checkpoint di ripresa ogni 20 minuti;
- codice congelato alla prima partenza in `<cella>/code` (`code_snapshot.py`), lo stesso commit per le quattro celle.

**Quota d'espressioni** (`STREAM_EXPR_FRAC`): bfm2019 0, ict 0.2315, gnm 0.1968 (le quote del C3M) in tutte le celle;
in B e D anche **flame2023 0.2142**, la media delle quote di ICT e GNM (i 3DMM del C3M con modello d'espressione; per
FLAME non c'e' un precedente nel C3M). Scelta prima dei numeri.

**FLAME 2023 nello stream** (`sources.py`): maschera "face" di FLAME, 1.787 vertici (patch nativa, nessuna topologia
decimata), 300 modi d'identita' a code larghe, 100 d'espressione; stessa mappa della GT unificata di FLAME 2020. E' una
risoluzione piu' bassa degli altri domini (BFM 2019 10.101 v. di lavoro, ICT 9.409, GNM 9.022): caratteristica della
fonte, non cambiata. In B e D, a T uguale, bfm2019, ict e gnm ricevono 3/4 dei passi e delle identita' che ricevono in
A e C (diluizione, voluta: si confronta a budget uguale, come C3F contro C2F in E1; misurata dagli held-out, sez. 4.3).

**Parzialita'** (C, D): `--partial-p 0.5`, area tolta U(0.03, 0.40) per banda dal bordo e taglio planare, buchi
(`partial_aug.DEFAULTS`, versione 1), applicata dopo la discretizzazione e prima degli operatori; identita',
espressioni, discretizzazioni e rumore non cambiano (seme dal seme del rumore della vista); bersagli dalla forma neutra
(invariati, come per `crop`). La vista `crop` di training (`make_crop`, peso 1.0) c'e' in tutte le celle.

**FaMoS non entra in training** in nessuna cella.

**Nodi.** Le quattro celle su due nodi uguali a 192 CPU e 8 L40S, a blocchi: A e D su `a768-l40s-05`, B e C su
`a768-l40s-06` (`--nodelist`). Cosi' l'effetto del nodo (velocita', contesa, quindi viste fresche e riuso) si
cancella negli effetti principali e si confonde solo con l'interazione. Se un nodo non e' disponibile al lancio si
usa l'altro nodo L40S a 192 CPU con la stessa regola, e lo si scrive.

**Budget T = 60.000 passi** (S = 600 passi per epoca, 100 epoche), uguale per le quattro celle.
- C3F (bracci factorized): 21.096 passi x 1 rank x 5 identita' = 105.480 estrazioni d'identita' (<= 6 viste
  ciascuna) su 5.793 identita' fisse, circa 18 usi per identita'.
- C3M (factorized, 6 L40S): 60.000 passi x 6 rank x 5 = 1,8 M estrazioni su 64.400 identita'; e123 (36.039 passi,
  1,08 M estrazioni, circa 17 usi per identita') e' quasi uguale a e205 (HIFI3D FR d_F cal. 0.748 / 0.739, SR 0.595 /
  0.590; FaceScape SR 0.746 / 0.756; FaceVerse 0.289 / 0.290; eccezione FaceScape FR 0.665 / 0.711,
  `factorized_paired.csv`): il plateau e' raggiunto, in passi d'ottimizzatore, entro 36.000 passi a lr 1e-4.
- Qui: 60.000 passi d'ottimizzatore (= T del C3M, oltre il suo plateau in passi), 600.000 estrazioni d'identita'
  (5,7 x C3F, 0,56 x C3M e123); con 6.1 viste fresche/s (smoke) circa 27.000 identita' distinte, ognuna usata circa
  22 volte (C3F 18, C3M e123 17, e205 28).
- Durata attesa: 60.000 x 0.435-0.447 s = 7,3-7,5 ore piu' avvio ed eval online; `--time 10:00:00`. Se un run va in
  TIMEOUT si continua con un job nuovo sullo stesso `STREAM_OUT` (ripresa da `last.pth`) fino a 60.000 passi.
- Checkpoint `epochNNN.pth` / `epochNNN_ema.pth` ed eval online (BFM REMESH, solo per far girare il trainer) ogni 10
  epoche (10%).

## 3. Checkpoint della lettura

**L'ultimo checkpoint con i pesi EMA, `epoch100_ema.pth` (60.000 passi), di ogni cella.** Nessun altro checkpoint entra
nelle letture: non gli intermedi, non i `best_by_*` (scelti sull'eval online). Una cella che non arriva a 60.000 passi
(crash non ripartibile) manca, e le letture che la usano non si calcolano: nessuna sostituzione.

## 4. Valutazione

### 4.1 Calibrazione c (per checkpoint, sugli held-out sintetici)

`v3_work/trainer/slurm/calib_ckpt.sbatch` (`fact_calib.py calib-ckpt`) su `epoch100_ema.pth` di ogni cella: held-out
bfm (BFM REMESH), ict, gnm, 100 soggetti per dominio, etichette down8k, noisy, original, remesh, up60k; coppie stesso
dominio, soggetti ed etichette diversi (297.000); c = mediana(d_P GT-SR) / mediana(d_P modello); c_LS e c per dominio
riportati. Uscita `<cella>/eval/calib/calib.json` con gli sha256 di checkpoint ed embedding.
- Gli held-out bfm sono BFM REMESH, non BFM 2019: per lo stream la c del dominio bfm e' gia' fuori dominio
  (`PLAN_MASSIVE.md` sez. 22.5). Stesso insieme di calibrazione per le quattro celle (anche B e D: niente held-out
  FLAME nella c), quindi c confrontabile.
- Ordine: c si calcola dai soli held-out in un passo che precede ogni numero di test della stessa cella (catena con
  `afterok`, sez. 7); `eval_ablation.py` controlla che lo sha256 del checkpoint in `calib.json` sia quello del
  checkpoint valutato. Deviazione dichiarata da `PLAN_MASSIVE.md` sez. 22.5 (c nel commit prima dell'eval di test): la
  catena e' automatica, quindi `calib.json` entra nel commit insieme ai risultati; c non dipende dai test.

### 4.2 Famiglie mai viste (test)

| famiglia | vista | soggetti | GT |
|---|---|---|---|
| HIFI3D (AI-NEXT) | `datasets/HIFI3D/eval_view`, 6 topologie | 100 | `datasets/CANONICAL_GT/eval/hifi3d_{fr,sr}.npz` |
| dev FaceScape (bilineare v1.6), neutra | `datasets/DEV_FACESCAPE/eval_view`, 6 topologie | 100 | `facescape_{fr,sr}.npz` |
| FaceVerse v2 con espressioni | `datasets/FACEVERSE_ZS/expr_view`, convenzione BFM (`_flip`) | 100 | `faceverse_{fr,sr}.npz` |

Nessuna delle quattro celle le ha in training. Sono le stesse viste, soggetti (selezione `zs_stage` col seme 1234) e
tabelle di scala (`trainer_v3/factorized/scale_tables/{hifi3d_eval,devfs_eval,fv_expr}.npz`) della valutazione
ufficiale: embedding [s, u] col passo `form` di `v3_work/trainer/ablations/c3f/eval_body.sh` (copiato all'avvio, cambia
solo la directory di uscita; i suoi sottoprodotti, embedding di dev FaceScape con espressioni e `eval_factorized.py`,
non entrano nelle letture). Nessun passo `famos` ne' `now`.

**Gruppi e righe:**
- `nocrop_cross` (**primario**): le righe di `fact_paired.rows_for` (importato in sola lettura): HIFI3D e dev FaceScape
  `nocrop_cross` di E12 (5 topologie senza crop, topologie diverse, soggetti diversi), FaceVerse `mesh_pair_nocrop`;
  seme del bootstrap = quello del gruppo in E12 (lo stesso di `fact_paired`);
- `all_cross`: tutte le coppie di topologie diverse col crop, come `bp_paired_e2.all_cross_rows` (emendamento 2 dei
  concorrenti parametrici, copiata): HIFI3D e dev FaceScape `all_cross` di E12, FaceVerse le `pair_metrics` dello stage
  di `fv_frames` (`aau/runs/ws_faceverse_expr/data_736f96956a/scale_e108_flip_topology`) senza il filtro del crop;
  seme di `all_cross` (E12; FaceVerse `stable_seed(1234, "expr_sec", "scale_e108@bfm", "all_cross", "latent_distance",
  "raw_chamfer")`);
- **righe col crop**: le righe di `all_cross` con il crop su almeno un lato, con le repliche di `all_cross`.
Controllo: le righe di `all_cross` senza crop hanno le stesse chiavi e GT di `fact_paired.rows_for`, altrimenti ci si
ferma. Maschera comune per (famiglia, gruppo): righe con GT FR e SR e le distanze delle quattro celle finite, soggetti
diversi.

**GT e distanze:** GT FR (forma, mm, primaria) letta con **d_F calibrata** (c della cella, sez. 4.1); GT SR letta con
**d_P**; distanze di `fact_paired.distances` (S = exp(s), dp_per_unit dal json di `--dist_npz` / `--gt-scale` del
checkpoint, uguale all'unita' della GT dello stream: `fact_calib.dp_of` lo verifica). d_F non calibrata (c = 1) con FR:
descrittiva. Statistica: Spearman per replica di `fact_paired._rep` (righe ripetute c_a x c_b volte, ranghi, Pearson).

### 4.3 Held-out sintetici dei generatori visti (controllo in distribuzione)

Gli insiemi statici `bfm`, `ict`, `gnm` di D1 (`aau/diagnostics/diag.py`, importato in sola lettura): i 100 held-out per
dominio della calibrazione, 5 etichette senza crop, righe = coppie di soggetti ed etichette diversi (99.000 per
insieme), GT SR = `c3f/gt_sr.npz` x dP_per_unit e GT FR = `c3f/gt_frcal.npz` (`d1_stats.gt_static`), embedding = quelli
della calibrazione della stessa cella; repliche `diag.boot_counts(100, gruppo)` (SeedSequence([20261113, gruppo]),
gruppi 0, 1, 2). Avvertenze: `bfm` e' BFM REMESH (famiglia di Basilea, non il BFM 2019 dello stream: vicino alla
distribuzione, non dentro); ict e gnm sono gli stessi generatori dello stream con identita' statiche di C3F/C3M; con FR
la d_F calibrata e' circolare per c (stimata sulle stesse coppie): **in distribuzione si legge SR (d_P, invariante a
c)**, FR e' descrittiva.

### 4.4 FLAME 2023 sintetico di D1 (solo A e C)

Gli insiemi `flame2023_s1` (patch suddivisa 1-a-4, 6.986 v.; primario come in D1) e `flame2023` (nativa, 1.787 v.) di
D1: 200 identita' FLAME 2023 Open N(0, 1) non troncate (SeedSequence([20261111, k])), 5 etichette senza crop, righe =
coppie di soggetti ed etichette diversi (398.000), GT FR e SR di `aau/runs/evidence/diagnostics/d1/gt_flame2023*.npz`,
repliche `diag.boot_counts(200, 3)`. Mesh e operatori di D1 sono stati cancellati a fine lavoro: si rigenerano col
codice di D1 (`d1_gen.py`, importato, uscite reindirizzate in `ablation_2x2/d1flame` e in `datasets/`), stessi id e
semi. Controllo: la GT rigenerata coincide con quella di D1 e i vertici per etichetta con `d1/gen.json`; se no, FLAME
non si valuta e lo si scrive. **Solo A e C**, per i quali FLAME resta mai visto; B e D non si valutano su questi insiemi.
Lettura descrittiva: livelli di A e C e C - A (la parzialita' su un generatore mai visto).

### 4.5 Bootstrap e IC

Bootstrap per soggetto, 1000 repliche, conteggi uguali per le quattro celle dentro (insieme, gruppo): differenze
appaiate sulle stesse righe e repliche. IC 95% percentile; P(delta <= 0) sulle repliche. Semi: sez. 4.2-4.4 (stampati
in `controls.json`). Media su piu' famiglie (descrittiva): media replica per replica di stime indipendenti.

## 5. Letture principali (soglie scritte prima dei numeri)

Notazione: rho_X = Spearman della cella X in (famiglia, gruppo, GT); tutte le differenze sulle stesse righe e repliche.
- **Effetto FLAME** E_F = [(rho_B - rho_A) + (rho_D - rho_C)] / 2.
- **Effetto parzialita'** E_P = [(rho_C - rho_A) + (rho_D - rho_B)] / 2.
- **Interazione** I = (rho_D - rho_C) - (rho_B - rho_A).
- Effetti semplici: B - A e D - C (FLAME senza e con parzialita'), C - A e D - B (parzialita' senza e con FLAME).
Cella di lettura = (famiglia, GT): FR con d_F calibrata, SR con d_P.

**Soglie.** Un solo seme per cella: l'IC copre i soggetti di test, non la variabilita' da run a run. Il rumore fra semi
di riferimento: factorized s1234 contro s2345 (C3F), 6 celle per gruppo (3 famiglie x {FR d_F cal., SR d_P}),
`factorized_paired.csv` per il gruppo primario (differenze +0.018, +0.009, -0.008, -0.006, -0.015, -0.027),
`baselines_param/paired_e2.csv` per `all_cross` (+0.048, +0.017, +0.019, +0.026, -0.004, -0.019) e per le righe col crop
(+0.110, +0.036, +0.073, +0.042, +0.018, -0.005). sigma_run = RMS delle differenze / sqrt(2); deviazione standard da seme
attesa: sigma_run per un effetto principale (media di due contrasti), sqrt(2) sigma_run per un effetto semplice,
2 sigma_run per l'interazione. Soglia = 2.5 volte quella deviazione, al centesimo:

| gruppo | sigma_run | delta_M (effetto principale) | delta_S (effetto semplice) | delta_I (interazione) |
|---|---|---|---|---|
| `nocrop_cross` | 0.011 | 0.03 | 0.04 | 0.06 |
| `all_cross` | 0.018 | 0.05 | 0.06 | 0.09 |
| righe col crop | 0.042 | 0.10 | 0.15 | 0.21 |

Per un effetto E in una cella: **beneficio** se E >= +delta e IC_low > 0; **danno** se E <= -delta e IC_high < 0.

### 5.1 R_F, effetto FLAME (primaria): 6 celle `nocrop_cross` (3 famiglie x FR, SR)

- **SI'**: beneficio in almeno 3 celle su 6, in almeno 2 famiglie, e nessun danno;
- **NO**: nessuna cella con beneficio;
- **PARZIALE**: altrimenti (si elencano le celle).
I danni si elencano sempre. Accanto: E_F medio sulle tre famiglie per GT (descrittivo).

### 5.2 R_P, effetto parzialita'

- **R_P1, crollo sul crop**: 6 celle delle righe col crop (3 famiglie x FR, SR), con delta_M = 0.10: SI' / NO /
  PARZIALE con la regola di R_F.
- **R_P2, costo senza crop**: 6 celle `nocrop_cross`: **costa** se almeno una cella ha danno (E_P <= -0.03 e
  IC_high < 0), altrimenti **non costa**; le celle con IC_high < 0 ma E_P > -0.03 si elencano come costo sotto soglia.
- Verdetto: "riduce il crollo senza costo" solo con R_P1 SI' e R_P2 non costa; le altre combinazioni si scrivono come
  sono.

### 5.3 R_I, interazione

Per ogni cella di R_F (nocrop) e di R_P1 (crop): **interazione da leggere** se |I| >= delta_I e l'IC esclude 0. In
quella cella l'effetto principale e' una media di effetti diversi: si leggono gli effetti semplici (con delta_S) e lo
si scrive accanto al verdetto, che resta quello calcolato con le regole 5.1-5.2.

### 5.4 Letture secondarie, descrittive (nessuna regola)

- `all_cross`: E_F, E_P, I con le soglie della tabella (segnalate, non verdetti).
- Crollo per cella = rho(righe col crop) - rho(righe senza crop di `all_cross`), con le repliche di `all_cross`;
  effetto della parzialita' sul crollo = [(crollo_C - crollo_A) + (crollo_D - crollo_B)] / 2 (e lo stesso per FLAME).
- d_F non calibrata con FR (la distanza preregistrata di factorized_protocol), sui tre gruppi.
- Held-out in distribuzione (sez. 4.3): rho per cella, E_F (diluizione: B e D vedono 3/4 dei passi per dominio),
  E_P, I; si legge SR, segnalando con le soglie di `nocrop_cross`.
- FLAME 2023 di D1 (sez. 4.4): rho di A e C, C - A.
- c, c_LS e c per dominio delle quattro celle; s/passo, viste fresche/s, riuso, identita' distinte e loss
  (`summarize_run.py`, `stream_stats_rank*.jsonl`): se il riuso o le viste fresche di due celle differiscono di piu' del
  20%, lo si scrive accanto agli effetti che le confrontano.

## 6. Esclusioni

- **Mai Ava-256** (`datasets/AVA256`, `aau/ava256/`): test confermativo vergine, nessun file letto.
- **Mai FaMoS TEST**: nessun passo `famos` della valutazione, nessuna chiamata a `fact_paired.famos_frame`; FaMoS non e'
  nemmeno nel training.
- Nessun numero di questa ablazione sceglie checkpoint, iperparametri, c o soglie: tutto e' fissato qui.

## 7. Esecuzione

- Training: `CELL=<A|B|C|D> sbatch --nodelist=<nodo> v3_work/stream/slurm/ablation_2x2.sbatch` (variabili della sez. 2,
  poi `massive.sbatch`); uscite in `ablation_2x2/<cella>` (`STREAM_OUT`). Nessuna modifica agli script dello stream
  mentre i run girano.
- Valutazione per cella, dopo il training (`afterok`), su una A100 di nv-ai-04 con `--qos=unprivileged --requeue`
  (job brevi, ripartibili): calibrazione (4.1), embedding dei test (4.2), per A e C embedding di FLAME (4.4); prima, un
  job CPU rigenera mesh e operatori di FLAME (4.4). Riepilogo (`v3_work/stream/eval_ablation.py`, CPU) dopo tutti:
  `spearman.csv`, `effects.csv`, `readings.json`, `controls.json`, `results.md` in `ablation_2x2/`.
- Disco: checkpoint circa 0.2 GB per cella; operatori FLAME rigenerati circa 12 GB, da cancellare dopo i risultati
  (`datasets/` e' fuori da git). Nel repo (pubblico) solo .md, .csv, .json piccoli: niente mesh, npz, checkpoint.

## 8. Limiti dichiarati prima

- **Un seme per cella**: le soglie vengono da UNA coppia di semi di C3F (altra ricetta, altri dati) e sono grezze; un
  effetto vicino alla soglia va confermato con un secondo seme.
- Il nodo e' confuso con l'interazione (sez. 2).
- Diluizione dei domini in B e D a T uguale; la risoluzione di FLAME nello stream (1.787 v.) e' piu' bassa delle altre.
- c sugli held-out BFM REMESH / ICT / GNM per tutte le celle; FR in distribuzione circolare per c.
- Famiglie di test = campioni di 3DMM, non dati reali; HIFI3D e dev FaceScape sono gia' entrati in scelte di modello
  (regola dual, k, calibrazione), mai in training. Nessun dato reale in questa ablazione.
- Le righe col crop di FaceVerse non esistono in E12: vengono dall'emendamento 2 dei concorrenti parametrici.
- Il riuso delle viste a 2 GPU e' alto (circa 20 usi per vista fresca, smoke): uguale per costruzione fra le celle a
  meno della velocita' dei produttori, che si misura.
