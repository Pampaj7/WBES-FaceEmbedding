# Protocollo adattabile (few-shot): regole (scritte PRIMA dei numeri)

Domanda: chi ha un dataset proprio dedica k identita' etichettate (GT FR / SR) all'adattamento; con quanti k la nostra
metrica adattata ordina le identita' RESTANTI meglio dei migliori concorrenti, se anche loro ricevono le stesse k
identita' per scegliere la configurazione? Codice `aau/few_shot/` (`fs_common.py`, `fs_now.py`, `fs_head.py`,
`fs_eval.py`, `fs.sbatch`).

Questo file e il codice sono nel commit delle regole, prima di qualunque rho, delta o metrica sulle identita' di test.
Prima di scriverle: (i) selftest (`selftest.json`: estrazioni, blocchi, checkpoint; nessuna GT letta per un metodo);
(ii) collaudo di `fs_now.py run` su una mesh A di HIFI3D (396 coppie ordinate, 0 errori, 0,34 s per verso e metodo);
(iii) render, rilevatore e retroproiezione di NoW su FaceVerse (sez. 5), nessuna GT; (iv) i run di NoW col rilevatore
sulle 4 viste sono partiti prima di questo commit (job 1074590, 1074591, 1074626, 1074627: solo distanze, nessuna GT).
Erano noti: i rho della matrice completa sui 100 soggetti (`full_matrix/spearman.csv`) e i risultati di D5
(adattamento contro d_P, `results_d5.md`). Nessun numero per split.

## 1. Perimetro

- Domini: **HIFI3D**, **FaceScape dev**, **FaceVerse con espressioni**, **FaceVerse neutra**: i 100 soggetti valutati
  della matrice x le 5 topologie senza crop. **FaMoS TEST**: solo riconoscimento dell'identita' (sez. 7).
- **Niente Ava-256**: nessun file letto; i 56 di calibrazione restano riservati alla R6.
- Modelli: factorized **s1234** e **s2345**, i checkpoint della calibrazione (quelli di D3, D5 e R6; controllato contro
  lo store di ogni dominio). Encoder congelato: si adatta solo la testa.
- Calcolo solo su CPU (partizione prioritized, al massimo 4 job contemporanei). Nessuna GPU necessaria: gli embedding
  sono quelli della matrice e di `esterna_famos_id`.

## 2. Estrazioni

- k = **5, 10, 20, 50**; **5 estrazioni** per k. Per dominio d (indice j: hifi3d 0, facescape 1, faceverse 2,
  faceverse_neutral 3, famos 4) e ripetizione q = 0..4: permutazione dei soggetti in ordine di stringa,
  `default_rng(SeedSequence([20261127, j, q])).permutation(n)`; adattamento = i primi k (annidati in k), test = gli
  altri n - k. Le permutazioni con i nomi vanno in `draws.json`.
- Le identita' di adattamento e di test sono disgiunte per costruzione (controllo `fs_common.check_draws`).

## 3. Righe

- Righe = l'insieme `nocrop` della matrice (`fm_eval.rows_view`: coppie di soggetti diversi e topologie diverse,
  nessuna crop; 4.950 coppie di soggetti x 20 = 99.000 per dominio), tenute dove TUTTE le colonne del confronto e le
  due GT sono finite. La maschera della matrice toglie in piu' le righe dove fallisce il NICP per coppia (HIFI3D 776,
  FaceScape 10.276): qui il NICP non e' fra i metodi, quindi quelle righe restano (dichiarato). Righe tolte per colonna
  in `controls.json`.
- **Adattamento** = righe con i due soggetti in A; **test** = righe con i due soggetti nel test. Le righe miste non si
  usano.

## 4. Nostro: la testa adattata di R6

- Procedura della R6 (`PROTOCOL_CONFERMATIVO_candidato.md`, sez. 10; `conf_head.py`): `d3_e2.fit_e2` (perdita
  riscalata per blocco, `d3_head.fit`, L-BFGS-B, maxiter 2000), **pesi EQ**, **r = 64, lambda = 1e-6 fissi**, nessuna
  CV, nessun rifit; il blocco del dominio in coda alle sorgenti; **c della testa con `d3_head.calib_c` sulle SOLE
  sorgenti** (la GT del dominio non entra in c).
- **Sorgenti: le 7 di D3 tranne il generatore del dominio** (`d3_stats.settings()['<T>|all']`, T = hifi3d, facescape,
  faceverse per entrambe le viste FaceVerse; famos per FaMoS). Scostamento dalla R6 (8 sorgenti: Ava non e' fra loro),
  dichiarato: con tutte e 8 il pool non valutato dello stesso generatore (200 soggetti) farebbe da dataset proprio
  gratuito e k non sarebbe piu' il numero di identita' etichettate del dominio.
- **Blocco del dominio**: i k soggetti di adattamento x le 5 etichette senza crop, x = u x dp_per_unit dagli store della
  matrice; GT SR del dominio (`datasets/CANONICAL_GT/eval/<gt>_sr.npz` x dP_per_unit, le GT delle sorgenti di D3);
  coppie di soggetti diversi ed etichette diverse.
- **Letture**: SR = d_h (GT SR); FR = d_F della testa, sqrt((S_a - S_b)^2 + S_a S_b (c d_h)^2) (GT FR). Fissate, nessuna
  scelta con le k identita'.
- Riferimenti sulle stesse righe di test (descrittivi): nostro non adattato (d_P con SR, d_F cal. con FR: le colonne
  della matrice) e **testa k = 0** (le stesse sorgenti, nessun blocco: una per braccio e generatore escluso).
- Fit sul nodo **i256-a10-10** (D3 e2, D5, R6). Controllo **K0**: la testa `joint` (8 sorgenti, FLAME 0-99, r = 64,
  lambda = 1e-6) riproduce c e alpha / alpha0 di `d3_e2_heads.csv` entro 1e-6 relativo; se no si riporta e si
  prosegue. Un fit con `success` = False si usa com'e' e si conta.

## 5. Concorrenti: le stesse k identita' per scegliere la configurazione

`aau/robustezza_iperparametri/` non esiste al momento di queste regole (13 ottobre): **griglia piccola dichiarata qui**,
fatta delle varianti della stessa famiglia gia' calcolate nella matrice (nessun parametro numerico interno, limite
dichiarato).

| famiglia | configurazioni (ordine dei pari) | letture con GT FR | letture con GT SR |
|---|---|---|---|
| ICP + Chamfer (`blmm`, matrice) | mm, cs | entrambe | entrambe |
| NoW col rilevatore | metrical (Procrustes senza scala + ICP rigido), originale (con scala) | entrambe | entrambe |
| fit 3DMM vB (matrice) | GNM (visto), FLAME 2023 Open x {coefficienti, mesh d'identita' FR, mesh d'identita' SR, composizione S_B + k d_P} | coef, mesh FR, composizione (6) | coef, mesh SR (4) |

- **Regola di scelta**: per (dominio, k, q, famiglia, GT) vince la configurazione col rho di Spearman (non pesato)
  massimo con la GT sulle righe di ADATTAMENTO; pari esatti: la prima nell'ordine della tabella. Solo letture
  dichiarate per quella GT (`fm_eval.METHODS`). Le scelte vanno in `choices.csv`.
- **NoW col rilevatore** calcolato qui su tutte le coppie senza crop delle 4 viste (`fs_now.py`; la matrice lo ha solo
  su un sottocampione dell'8% di HIFI3D e FaceScape): la catena `det` di `pr_noise.py` importata senza modifiche
  (render frontale, insightface `1k3d68`, retroproiezione, ritagli di NoW, `pr_run.one_pair` con `--now-opt` 0), media
  dei due versi. HIFI3D e FaceScape riusano render, landmark e prep del job 1071153; controllo di riproduzione sulle
  coppie senza crop del sottocampione esistente, scarto <= 1e-4 mm (`ei.TOL["now_mm"]`), altrimenti ci si ferma.
  FaceVerse: render, rilevatore e prep qui (riepiloghi in `now_det/prep/`). **Il rilevatore va peggio su FaceVerse**:
  spostamento medio dei 7 landmark di NoW 8,4 mm (con espressioni) e 7,3 mm (neutra) contro 3,5 (HIFI3D) e 2,9
  (FaceScape); 54 e 46 render senza volto (tutti `noisy`, ripiego sul riquadro). Per questo:
- **NoW coi landmark esatti** della matrice (metrical, originale; plug-in dichiarati) entra come famiglia
  **descrittiva**, con la stessa regola di scelta, fuori dal k di sorpasso e dal "migliore" (un utente non ha i
  landmark esatti).
- Il nostro non sceglie nulla: la sua configurazione e' fissa (sez. 4).

## 6. Statistiche e lettura

- **Spearman pesato c_a c_b** sulle righe di test (`diag.wranks` / `diag.wpearson`: i ranghi delle righe ripetute,
  come `fm_eval._rep`).
- **Repliche**: 1000 per soggetto coi conteggi della matrice (`default_rng(seme T1)` sui 100 soggetti ordinati: HIFI3D
  990708, FaceScape 796786, FaceVerse e neutra 271049), **ristretti alle identita' di test** (quelle di adattamento non
  hanno righe). Lo stesso ricampionamento della popolazione per tutte le estrazioni di un dominio: la media sulle
  estrazioni porta la covarianza dei soggetti comuni. Gli IC sono condizionati alle estrazioni, ai fit e alle scelte.
- Per (dominio, k, GT): rho = **media sulle 5 estrazioni**, replica per replica; IC 95% percentile; sd fra estrazioni.
  **Delta appaiati** = nostro adattato - concorrente adattato (la configurazione scelta su A in quella estrazione),
  replica per replica, mediati sulle estrazioni; anche contro il nostro non adattato e la testa k = 0.
- **Curva in k**: nostro adattato (due semi), nostro non adattato, testa k = 0, ogni famiglia adattata, la migliore
  configurazione fra le tre famiglie scelta su A (descrittiva), ogni configurazione fissa (in `spearman.csv`).
- **Miglior concorrente** (tabelle): la famiglia adattata col rho medio piu' alto sul test a quel (dominio, k, GT):
  scelta post hoc, a favore dei concorrenti.
- **k di sorpasso** per (dominio, GT, seme): il k minimo della griglia al quale l'IC 95% del delta contro **ognuna**
  delle tre famiglie adattate sta sopra lo 0 (equivale a battere il migliore, comunque lo si scelga); "nessuno" se non
  c'e' entro k = 50. **Lettura principale: entrambi i semi** (k di sorpasso = il maggiore dei due; "nessuno" se un seme
  non sorpassa). Si riportano anche i k successivi (`sopra_a_k`, `stabile_dopo`): se a un k maggiore l'IC torna a
  contenere lo 0, lo si dice.
- Tutto **descrittivo**: nessuna correzione per confronti multipli (4 domini x 2 GT x 2 semi x 4 k), dichiarato.

## 7. FaMoS TEST: riconoscimento dell'identita'

- Le 150 scansioni grezze di 15 soggetti di `esterna_famos_id` (stessa preparazione, latenti e distanze), metriche di
  `ei.py`: rank-1, mAP, AUC, EER, ristrette alle scansioni dei soggetti di test (galleria = le altre scansioni di test).
- k = **5 e 10** (test su 10 e 5 soggetti; k = 20 e 50 impossibili), 5 estrazioni, seme della sez. 2 (j = 4).
  **Limite**: con k = 10 restano 5 identita' di test; IC larghi, rank-1 vicino al soffitto.
- Testa: sorgenti `famos|all` (FaMoS TRAIN fuori); blocco = i k soggetti x le 10 sequenze (etichetta = sequenza), GT SR
  di FaMoS TEST (`CANONICAL_GT/eval/famos_test_sr.npz`) ristretta ad A; stessa configurazione; lettura principale
  **d_F della testa** (riconoscimento: la taglia conta; d_h descrittiva).
- Concorrenti: ICP + Chamfer {mm, cs}, NoW col rilevatore {metrical, originale} (le distanze di `esterna_famos_id`);
  scelta = **mAP massimo sulle scansioni di adattamento** (galleria = le altre scansioni di adattamento), pari: ordine.
  **Fit 3DMM vB: assente su FaMoS** (nessun fit sulle scansioni grezze; dichiarato).
- Repliche: `famos_eval.bootstrap_counts(15, 1000, 1234)` (quelle di `ei.py`) ristrette ai soggetti di test; una
  replica senza soggetti di test e' NaN ed esce dall'IC (contate). Delta = nostro - concorrente (EER: concorrente -
  nostro), mediati sulle estrazioni. Metrica principale **mAP**; k di sorpasso letto come nella sez. 6 contro le due
  famiglie. Descrittivo.

## 8. Controlli

- **K1**: sulle righe della maschera della matrice (le righe della sez. 3 con il NICP finito), tutti i 100 soggetti,
  rho e IC ricalcolati qui di ogni colonna della matrice usata (nostro non adattato, ICP, fit vB, NoW coi landmark
  esatti) contro `full_matrix/spearman.csv` (`nocrop`, passo `comune`): scarto atteso 0 (soglia 1e-9) e righe uguali.
- **K-emb**: d_P dai latenti degli store = colonna d_P della matrice sulle stesse righe (scarto relativo); FaMoS: contro
  `esterna_famos_id`. **K-GT**: la G del blocco proporzionale alla GT SR delle righe. **K4**: checkpoint degli store =
  quelli della calibrazione. **K0** (sez. 4). **NoW**: riproduzione (sez. 5), coppie mancanti ed errori per vista.
- Teste lette dal valutatore: soggetti dell'estrazione controllati, `success` e nodo contati.

## 9. Fuori da questo giro

- Parametri numerici interni dei concorrenti (iterazioni e campionamento di ICP, raggio del ritaglio di NoW,
  regolarizzazione dei fit 3DMM): la griglia di `aau/robustezza_iperparametri/` non era disponibile. Se arriva, si
  rifa' la sola scelta dei concorrenti con la sua griglia, in un passo separato e dichiarato.
- Scelta di r e lambda della testa con le k identita' (la configurazione resta quella di R6), variante W di D5, crop.
- Gli altri concorrenti della matrice (NICP, REALY, ULRSSM, GW, percettivi): la domanda nomina ICP mm, NoW metrical col
  rilevatore e il miglior fit 3DMM vB.

## Emendamento 1 (13 ottobre, sera; PRIMA di qualunque numero sulle identita' di test)

Nessun rho, delta o metrica sulle identita' di test e' stato calcolato prima di questo emendamento.

1. **Controllo di riproduzione di NoW col rilevatore (sez. 5): fallito alla soglia scritta (1e-4 mm), causa la CPU.**
   - HIFI3D, 7.942 coppie in comune col sottocampione: mediana dello scarto 1,8e-9 mm (metrical) e 3,6e-10 mm
     (originale); oltre 1e-4 mm 89 coppie (metrical, massimo 2,3e-3 mm = 0,1% relativo) e 38 (originale, massimo
     0,036 mm su una sola coppia, 1,6% relativo). FaceScape: massimo 2,3e-3 mm (metrical) e 7,3e-3 mm (originale).
   - Il riferimento (`competitors_new/<metodo>/lmk_det`, job 1071153) e' stato calcolato su a256-t4-04; il run completo
     su a768-l40s-06 (HIFI3D) e a768-l40s-03 (FaceScape).
   - Verifica (`fs_now.py recheck`, `now_det/<vista>_recheck_<nodo>.json`): le coppie fuori soglia ricalcolate sul nodo
     del riferimento riproducono il riferimento con scarto **0,0 mm** (HIFI3D, 125 coppie), e sul nodo del run
     riproducono il run con scarto **0,0 mm** (HIFI3D 125, FaceScape 49 coppie). La catena e' la stessa;
     l'ottimizzatore iterativo di NoW da' risultati diversi su CPU diverse, deterministici sulla stessa.
   - **Decisione**: si usano i valori del run completo. Il criterio di riproduzione diventa: scarto 0 delle coppie fuori
     soglia ricalcolate sul nodo del riferimento. `merge` registra lo scarto e non si ferma; decide `recheck`.
2. **FaceVerse costa 4-5 volte HIFI3D per coppia** (mesh di 28.632 vertici contro 9.518; mediana 1,45 s per verso
   contro 0,31 s col metrical): i job da 2 h non bastano e si continuano con job ripartibili (le mesh A gia' salvate si
   saltano). Le coppie di una vista FaceVerse possono venire da nodi diversi: il job di ogni mesh A e' registrato nelle
   parti, i job in `now_det/<vista>.json`.
3. **Nodo del fit (sez. 4)**: i256-a10-10 era occupato da un altro job. 187 teste su 188 sono state calcolate su
   **i256-a10-08** (job 1075578, 141 s di mediana); l'ultima (s1234, FaceVerse neutra, k = 50, q = 0) non era finita
   dopo 90 minuti e il job e' scaduto (in D5 il fit piu' lungo durava 211 s: processo bloccato, non convergenza
   lenta). Rifatta su **i256-a10-10** (job 1075738): 42 s, `success`. Sullo stesso nodo **K0 passa con scarto relativo
   0,0** su c e alpha / alpha0 per entrambi i semi. Controllo dei nodi: 3 teste di -08 rifatte su -10 sono identiche
   bit per bit (W, alpha, c, iterazioni; `heads_refit_check.json`).
