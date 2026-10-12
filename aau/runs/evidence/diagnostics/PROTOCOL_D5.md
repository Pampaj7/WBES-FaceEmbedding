# Diagnostica D5: adattamento few-shot della testa metrica a un generatore nuovo (protocollo, scritto PRIMA dei numeri)

Scritto il 12 ottobre 2026 dal coder, su richiesta del PI, PRIMA di ogni testa e di ogni Spearman di D5. L'impronta
sha256 di questo file sta nel messaggio del commit che lo introduce e in `PROTOCOL_D5.sha256`. E' una DIAGNOSTICA post hoc
rispetto ai risultati gia' pubblicati (nasce dall'emendamento 2 di D3): si legge come tale, con le regole sotto fissate
prima dei suoi numeri.

Domanda: quanti soggetti etichettati di un generatore nuovo servono a una testa metrica lineare sugli embedding
congelati per correggere d_P su quel generatore? Se bastano 10-20 identita', "encoder addestrato su sintetici +
adattamento few-shot" e' una tesi difendibile.

## 0. Cosa era gia' visto prima di scrivere

- **D3, emendamento 2 (POST HOC, `results_d3.md`, `d3_e2_*.csv`; protocollo 36fe43e, numeri 3bec6b9).** Testa (i-e2) =
  testa (i) di D3 con perdita riscalata per sorgente e lambda in {1e-6, ..., 1}; CV per soggetto sulle sorgenti; tutte le
  scelte sul bordo lambda = 1e-6 (r = 64, tranne FLAME s1234 e FaceVerse s2345 con r = 32).
  - LOGO (tutte le sorgenti tranne T), delta d_P con SR sulle righe incrociate, s1234 / s2345: FaceScape +0.034 /
    +0.016, HIFI3D +0.024 / +0.027, FaceVerse neutra +0.016 / -0.029, FLAME (200 soggetti) +0.028 / +0.030; media sui 4
    +0.025 [+0.004, +0.045] / +0.011 [-0.013, +0.037]. FLAME sulle sole righe fra i soggetti 100-199 (`cross_half`):
    +0.018 [-0.008, +0.045] / +0.019 [-0.006, +0.043].
  - Testa congiunta (tutte le 8 sorgenti, compresi i 200 soggetti non valutati di ogni pool e FLAME 0-99): media sui 4
    +0.120 [+0.098, +0.142] / +0.102 [+0.076, +0.130]; per bersaglio (s1234) FaceScape +0.142, HIFI3D +0.153, FaceVerse
    +0.112, FLAME +0.075.
  - Critic: a (r = 32, lambda = 1e-5) il LOGO estrapola meglio della scelta di CV (FaceScape s1234 +0.050 contro +0.034,
    FaceVerse s2345 +0.008 contro -0.029): la CV sulle sorgenti premia l'adattamento ai generatori visti.
- Nessun numero di D5: nessuna testa con soggetti del bersaglio fuori dalla testa congiunta di D3 e' stata adattata ne'
  valutata prima di questo file.

## 1. Disegno

Per ogni bersaglio T in {FaceScape, HIFI3D, FaceVerse neutra, FLAME 2023} (`d3_stats.TARGETS` / `PRIMARY`) e per ogni
braccio (sez. 3):

- **Sorgenti**: le 7 sorgenti di "tutte tranne T" della catena e2 (`d3_stats.settings()["T|all"]`, FaMoS TRAIN
  compreso), identiche al LOGO di D3 e2, PIU' un blocco con **k soggetti etichettati di T** presi dal suo pool NON
  valutato, con tutte e 5 le etichette senza crop; k in **{0, 5, 10, 20, 50, 100, 200}**.
  - Pool di T: per FaceScape, HIFI3D e FaceVerse le 200 sorgenti non valutate di D3 (`d3/pools/<pool>/subjects.json`,
    campo `source`), con embedding e GT di D3; per FLAME i soggetti 0-99 (`id710000`-`id710099`, come la testa
    congiunta di D3).
  - **FLAME ha un pool di 100**: per FLAME k in {0, 5, 10, 20, 50, 100}; nella colonna k = 200 (e nella media sui 4
    bersagli a k = 200) FLAME entra col pool intero, k = 100. Dichiarato in ogni tabella.
  - Il blocco del bersaglio e' `d3_stats._B[(braccio, T)].take(soggetti estratti)`: stesse x = u x dp_per_unit, stessa
    GT (SR, `CANONICAL_GT/eval` per i pool, `d1/gt_flame2023_s1.npz` per FLAME), stesse coppie (soggetti diversi,
    etichette diverse). Ordine dei blocchi: quello di `d3_stats.SOURCES`.
- **Testa (i-e2) a configurazione fissa**: per (braccio, T) r e lambda sono quelli scelti dalla CV di D3 e2 per `T|all`
  (`d3_e2_heads.csv`, kind `e2`): s1234 FaceScape / HIFI3D / FaceVerse (64, 1e-6), FLAME (32, 1e-6); s2345 FaceScape /
  HIFI3D / FLAME (64, 1e-6), FaceVerse (32, 1e-6). Perdita riscalata per blocco (`d3_e2.rescale`: GT del blocco divisa per
  l'alpha0 del blocco, quindi anche del blocco con k soggetti), `d3_head.fit` (L-BFGS-B, maxiter 2000, W iniziale dalla
  base principale). **Nessuna nuova CV** al variare di k: con k = 5 una CV per soggetto dentro il bersaglio non ha
  coppie di validazione, e la CV sulle sole sorgenti e' quella di D3 e2. Perche' fissa: con k = 0 la testa e' la testa
  LOGO di D3 e2, e la curva in k misura solo l'effetto dei soggetti del bersaglio.
- **c** per d_F calibrata della testa: `d3_head.calib_c` sui blocchi di training (GT originale, non riscalata, pesi uguali
  per blocco), blocco del bersaglio compreso: la regola di D3 e2 (`d3_e2._refit`).
- **k = 0** e' il LOGO di D3 e2 (stessi blocchi, stessa configurazione): controllo K0 (sez. 5).
- **Estrazioni**: R = 10 estrazioni per k. Per (T, ripetizione q = 0..9) una permutazione del pool
  `default_rng(SeedSequence([20261125, j_T, q])).permutation(n_pool)`, j_T = indice di T in `d3_stats.TARGETS`
  (FaceScape 0, HIFI3D 1, FaceVerse 2, FLAME 3), pool nell'ordine dei soggetti del blocco (id crescente); i k soggetti
  della ripetizione q sono i primi k della permutazione. **Estrazioni annidate**: in una ripetizione i 5 soggetti di
  k = 5 stanno nei 10 di k = 10 e cosi' via. Stesse estrazioni per i due bracci e per le due varianti di peso
  (confronti appaiati). Col pool intero (k = 200 per i pool, k = 100 per FLAME) le 10 estrazioni coincidono: un solo fit,
  dispersione 0. Soggetti estratti in `d5/draws.json`.
- **Pesi (regola dichiarata)**:
  - **primaria, "peso uguale per sorgente" (EQ)**: il blocco del bersaglio pesa come ogni altra sorgente, 1/8 della
    perdita qualunque sia k (la regola di `d3_head.Problem`: ogni blocco pesa lo stesso). A k piccolo questo gia' pesa
    di piu' ogni coppia del bersaglio (k = 5: 1/8 della perdita su 10 coppie di soggetti);
  - **variante "meta' al bersaglio" (W), secondaria**: il blocco del bersaglio pesa **1/2** della perdita, le 7 sorgenti
    1/14 ciascuna (stesso k, stesse estrazioni, stessa configurazione, stessa c). Peso fisso, non scelto sui dati: una CV
    interna ai k soggetti non e' definita per k = 5 (un soggetto per fold, nessuna coppia).

## 2. Valutazione

- **Soggetti di test di T**: FaceScape, HIFI3D, FaceVerse neutra = i 100 soggetti valutati, righe incrociate di
  `fact_paired.rows_for` (`d3_stats.view_data`; le righe `cross` di D3: 88.725, 98.224, 99.000), mai fra le sorgenti
  dei pool; FLAME = le righe fra i soggetti 100-199 (`cross_half` di D3 e2, 99.000 righe), mai nel pool 0-99.
- **Bootstrap per soggetto di test**, 1000 repliche, i conteggi di D3: `fact_paired` col seme della vista (FaceScape
  796786, HIFI3D 990708, FaceVerse 271049), FLAME `diag.boot_counts(100, 0, seed=20261124)`. Spearman pesato c_a c_b
  (`diag.wranks`, `diag.wpearson`), le stesse repliche per tutte le colonne.
- **Misure**:
  - **primaria**: delta = rho(d_h, SR) - rho(d_P, SR), stesso braccio, stesse righe e repliche;
  - **vincolo**: delta FR = rho(d_F della testa, FR) - rho(d_F cal. di d_P, FR), d_F = sqrt((S_a - S_b)^2 + S_a S_b
    (c d)^2) con la taglia S = e^s del modello; c della testa (sez. 1), c ufficiale per d_P
    (`factorized_calibration.csv`), come D3.
- **Aggregazione sulle estrazioni**:
  - **delta medio sulle estrazioni**: per replica, la media dei delta delle R estrazioni; punto = replica 0; **IC 95%
    percentile** sulle repliche. L'IC e' condizionato alle estrazioni e ai fit: misura il rumore dei soggetti di test;
  - **dispersione fra estrazioni**: deviazione standard (ddof = 1), minimo e massimo dei delta puntuali delle R
    estrazioni; riportata accanto, mai fusa nell'IC;
  - **media sui 4 bersagli**: per replica la media dei 4 delta medi sulle estrazioni (FLAME col suo k, sez. 1); la sua
    dispersione fra estrazioni appaia le ripetizioni q dei 4 bersagli (media sui 4 della ripetizione q, poi SD su q).
  - **incremento** su k = 0: delta(k) - delta(0), per replica (k = 0 e' un fit solo).
- Si riportano tutti i (braccio, T, k, variante) con punto, IC, P(delta <= 0), SD / min / max fra estrazioni, e per
  estrazione i delta puntuali, r, lambda, c, alpha / alpha0, iterazioni ed esito del fit.

## 3. Bracci

factorized s1234 e factorized s2345 (le chiavi di `diag.RULE_ARMS`), embedding di D3 (calibrazione held-out, D1, pool di
D3, FaMoS TRAIN, store ufficiali delle viste). C3M non fa parte di D5. **Nessun nuovo embedding.** **Esclusi**: Ava-256
(test confermativo vergine) e FaMoS TEST.

## 4. Lettura preregistrata

Sulla variante primaria EQ, per braccio:

- **F(braccio)** vale a **k = 20** se (i) il delta medio sulle estrazioni di d_P con SR, mediato sui 4 bersagli, e'
  **>= +0.05** con **IC 95% sopra 0**, e (ii) su **ognuno** dei 4 bersagli il delta FR medio sulle estrazioni e'
  **>= -0.03** (stima puntuale).
- **FUNZIONA** se F vale in entrambi i semi; **NON CONFERMATO** se in uno solo; **NO** se in nessuno.
- Si giudica a un solo k (20, il massimo "few-shot" ammesso), non al migliore fra 5, 10 e 20: una prova sola.
- **k a meta' guadagno** per braccio: G(k) = delta SR medio sui 4 bersagli (punto, EQ); k_1/2 = il k minimo della griglia
  con G(k) >= G(200) / 2 (FLAME a k = 100 dentro G(200)); "non definito" se G(200) <= 0. Accanto, descrittiva, la
  versione sull'incremento: G(k) - G(0) >= (G(200) - G(0)) / 2.

Perche' queste soglie. +0.05 e' la soglia per bersaglio di D3 (R_LOGO), applicata qui alla media sui 4: e' circa il
doppio del LOGO di D3 e2 (+0.025 / +0.011) e meno della meta' della testa congiunta (+0.120 / +0.102), quindi separa
"quasi nulla come senza dati del bersaglio" da "una parte sostanziale di quello che si ottiene vedendo il generatore".
-0.03 su FR e' il vincolo di D3, per bersaglio. Due semi, come D3 e D4.

Descrittivo, accanto (non regole): per braccio il k minimo della griglia {5, ..., 200} a cui F vale; la frazione delle 10
estrazioni con delta medio sui 4 >= +0.05 a k = 5, 10, 20; per bersaglio il k a meta' guadagno e il numero di bersagli
con delta SR >= +0.05 e IC sopra 0; la variante W con la stessa regola (etichettata "variante W"), e W - EQ per replica;
k = 200 accanto alla testa congiunta di D3 e2 (che ha anche i pool degli altri bersagli: non e' lo stesso disegno).

## 5. Controlli

- **K0**: a k = 0, per braccio e bersaglio, delta SR e FR (punto e IC) = quelli di `d3_e2_delta.csv` (metodo `i_e2-dP`,
  righe `cross` per le viste, `cross_half` per FLAME), scarto <= 1e-9. Il calcolo gira sul nodo della valutazione di D3
  e2 (i256-a10-10, job 1068260), perche' BLAS diverse fra nodi cambiano i fit con lambda = 1e-6 (D3 e2: fino a 0.008 nel
  punteggio di CV). Se K0 non passa si riporta lo scarto e si prosegue: il riferimento di D5 e' il suo k = 0, fittato
  nello stesso job.
- **K1**: rho di d_P (SR) e di d_F cal. (FR) dei bracci = `d3_e2_spearman.csv` sulle stesse righe (punto e IC), scarto
  <= 1e-9.
- **K6**: soggetti estratti dentro il pool non valutato, disgiunti dai soggetti di test (viste: i 100 valutati; FLAME:
  100-199); annidamento in k verificato; il blocco del bersaglio a k > 0 ha k soggetti x 5 etichette.
- **Kw**: la perdita pesata con pesi uguali coincide con quella di `d3_head.Problem` (prove su dati finti in
  `test_d5.py`); esito del fit (`success`) e iterazioni riportati per ogni testa.

## 6. Risorse, uscite, codice

- CPU, partizione prioritized, un job da 64 CPU sul nodo i256-a10-10 (sez. 5); nessuna GPU. Fit ripartibili (una cache
  per testa in `d5/heads/`, fuori da git).
- Numeri: `d5_draws.csv` (una riga per estrazione), `d5_delta.csv` (aggregati), `d5/readings.json`,
  `d5/controls.json`, `d5/draws.json`; repliche in `d5/reps.npz` (fuori da git); risultati in `results_d5.md`.
- Codice nuovo in `aau/diagnostics/d5_fewshot.py`, `test_d5.py`, `d5.sbatch`; `diag.py`, `d1_stats.py`, `d3_*.py`,
  `fact_paired.py` e `v3_work/stream/` importati in sola lettura o non toccati.
