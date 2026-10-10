# Concorrenti parametrici, emendamento 1 (POST HOC): fit senza espressione, modello nel ciclo, errore di superficie

Scritto il 10 ottobre 2026 dal coder, su richiesta del PI, **DOPO aver visto i numeri** di `results.md` (commit
52160bc, protocollo `PROTOCOL.md` sha256 `0d7444e6...`). E' un emendamento **post hoc**: la motivazione sono i rilievi
del critic (verdetto RISERVE), riassunti qui sotto, non i numeri in se'. Le colonne, le righe, i semi, le repliche e la
regola di lettura del protocollo originale NON cambiano: le colonne originali restano l'analisi preregistrata; quelle
di questo emendamento sono analisi di sensibilita' post hoc e si leggono come tali. L'impronta sha256 di questo file
sta nel messaggio del commit che lo introduce; la sez. 8 (valori congelati dal pilota) si aggiunge con un commit
successivo, PRIMA di ogni calcolo sulle viste valutate.

Rilievi del critic che motivano l'emendamento:
1. l'espressione libera assorbe identita' (`bp.fit_points`, `bp.py:273-300`): con FLAME la sola psi spiega piu' della
   sola beta; sulle viste neutre il fit "con espressione" e' un concorrente indebolito;
2. la catena NICP -> regressione e' a due stadi: il modello non entra mai nelle corrispondenze, gli errori del NICP
   restano nei coefficienti;
3. "0 fit falliti" misura solo l'RMS sui punti registrati dal NICP, non l'aderenza alla superficie di ingresso;
4. l'avvertenza 3 della sez. 2 ("FLAME su FaMoS quasi oracolo, limite superiore") e' smentita dai numeri;
5. l'esito va formulato col suo perimetro esatto (sez. 7 qui sotto).

## 1. Variante A (`va`): niente espressione sulle viste neutre

Viste neutre: `hifi3d`, `facescape`, `faceverse_neutral` e `famos`. Per FaMoS le scansioni di galleria sono il frame
meno espressivo di ogni sequenza (`expr_mm` del manifest da 0.21 a 0.46 mm, contro 1.2-7.1 mm dei frame `peak`): si
trattano come neutre. Fit: gli STESSI punti registrati del NICP (`bp.enroll_model`, stesso seme; il NICP e'
deterministico: controllo in sez. 6), poi

    min_{R, t, beta} sum_v ||R (mu_v + B_v beta) + t - y_v||^2 + sigma^2 ||beta||^2,   psi = 0,

e per FLAME la mandibola fissa a theta = 0 (la posa del template). sigma = 1 mm, alternanza e arresto come la sez. 3.3
del protocollo (`FIT_ITERS`, `FIT_TOL_MM`).

Vista con espressioni (`faceverse`): espressione libera e, per FLAME, mandibola libera come nel protocollo, ma col
prior sull'espressione corretto (sotto).

**`EXPR_SCALE` (`v3_work/mm/loaders.py:88`) non applicato.** Il loader dichiara la distribuzione dei coefficienti
d'espressione di ogni modello come N(0, s^2) sul pool (s = `EXPR_SCALE`: GNM 0.5, FLAME 2023 1.0; e' la sigma con cui
il repo genera le espressioni). `bp.load_model` (`bp.py:88`) legge la base ma non `expr.scale`, e la regressione usa
N(0, 1) su psi. Rispetto al protocollo NON e' un bug (la sez. 3.3 dichiara N(0, 1) su identita' ed espressione), ma
e' un'incoerenza col prior dichiarato dal repo: per GNM la precisione del prior su psi e' 4 volte piu' debole del
dichiarato. Nelle varianti A e B il prior su psi e' N(0, s^2), cioe' termine sigma^2 ||psi / s||^2. Effetto: solo GNM
e solo dove l'espressione e' libera (`faceverse`); per FLAME s = 1 e la variante A su `faceverse` coincide col fit
originale (controllo in sez. 6). Le colonne originali restano come preregistrate.

## 2. Variante B (`vb`): modello nel ciclo

Parte dal fit della variante A (beta, psi, theta, R, t) e alterna, per `LOOP_ITERS` iterazioni esterne (numero fisso,
nessun arresto anticipato):

1. mesh del modello posata M = R x(beta, psi, theta) + t su TUTTI i vertici della regione (triangoli della regione);
2. corrispondenze nei due versi, punto-superficie (`open3d.t.geometry.RaycastingScene.compute_closest_points`, punto
   piu' vicino su triangolo):
   - modello -> ingresso: i vertici del template (al piu' 4096, gli stessi del NICP) sul punto piu' vicino della mesh
     di ingresso (in mm, frame di lavoro); scartata se la distanza supera `LOOP_TAU_MM` o il triangolo trovato ha un
     vertice sul bordo della mesh di ingresso;
   - ingresso -> modello: i 4096 punti della mesh di ingresso del NICP (stesso seme) sul punto piu' vicino di M,
     espresso in coordinate baricentriche del triangolo (quindi lineare nei coefficienti); scartata se la distanza
     supera `LOOP_TAU_MM` o il triangolo ha un vertice sul bordo della regione;
3. regressione MAP sulle corrispondenze con la stessa alternanza (Umeyama pesato senza scala, coefficienti in forma
   chiusa, ricerca della mandibola dove e' libera), al piu' `LOOP_INNER` = 5 giri interni per iterazione esterna,
   prior `LOOP_SIGMA_MM`^2 (||beta||^2 + ||psi / s||^2). Pesi: ogni verso ha massa totale N/2 (N = punti del template),
   cioe' peso 0.5 N / n_verso per corrispondenza, cosi' il termine dei dati ha la stessa massa del fit originale e
   sigma lo stesso significato.

Fallimento (oltre la sez. 6 del protocollo): meno del 25% di corrispondenze tenute in uno dei due versi all'ultima
iterazione, oppure RMS pesato sulle corrispondenze dell'ultima iterazione > `FAIL_RMS_MM`. Le espressioni seguono la
variante A (assenti sulle viste neutre, libere su `faceverse`). Entrambi i modelli; se il fit B di FLAME su `faceverse`
supera 2 ore in un job da 48 core, B si fa solo con FLAME 2023 Open (scelto a priori: e' il prior non visto) su tutte
le viste, e lo si dichiara.

**Scelta degli iperparametri, solo sul pilota non valutato.** Griglia: `LOOP_SIGMA_MM` in {0.5, 1, 2},
`LOOP_TAU_MM` in {2, 5, 10}, `LOOP_ITERS` in {5, 10} (lo stato a 5 e a 10 dello stesso run): 18 configurazioni per
modello. Pilota: i primi 10 soggetti, in ordine, di `blmm.template_subjects(vista)` (i 100 soggetti NON valutati del
NICP su template, lo stesso insieme da cui veniva il pilota del protocollo, i cui 3 soggetti non sono stati annotati)
x 5 topologie, sulle viste `hifi3d`, `facescape`, `faceverse` (con espressioni) e `faceverse_neutral` (gli stessi
soggetti, mesh neutre): 200 mesh per modello. FaMoS non ha soggetti non valutati: riceve i valori congelati.
Criterio, senza GT: per vista, S = media delle distanze `_fr` (mesh d'identita', sez. 4 (ii)) fra mesh dello stesso
soggetto (topologie diverse) / media fra soggetti diversi; punteggio = media di S sulle 4 viste; vince il minimo. A
parita' (punteggio entro l'1% relativo del minimo) si preferisce `LOOP_ITERS` minore, poi `LOOP_SIGMA_MM` maggiore,
poi `LOOP_TAU_MM` maggiore. Configurazioni con un fit fallito sul pilota escluse. Scelta separata per modello, uguale
per tutte le viste. Sul pilota si riportano anche S della variante A e del fit originale (informazione) e l'errore di
superficie (sez. 3).

## 3. Errore del fit rispetto alla superficie

Per ogni fit (originale, A, B) e mesh: M = mesh del modello posata come fittata (identita', espressione, mandibola,
R, t) su tutti i vertici della regione; ingresso = la mesh valutata in mm nel frame di lavoro (quello dei punti
registrati). Distanze punto-superficie, nessun rigetto per distanza:
- M -> ingresso: ogni vertice della regione sul punto piu' vicino della mesh di ingresso;
- ingresso -> M: i 4096 punti della mesh di ingresso del NICP (stesso seme) sul punto piu' vicino di M, tenuti solo
  se il triangolo trovato non ha vertici sul bordo della regione (altrimenti il punto sta fuori dalla regione).
Per mesh: mediana e p95 dell'unione dei due insiemi (mm), piu' la mediana di ciascun verso e la frazione di punti
ingresso -> M tenuti. Per (vista, modello, variante): mediana sulle mesh della mediana e del p95, massimo del p95.
Comprende il rumore della mesh di ingresso (topologia `noisy`): e' un errore di aderenza, non di identita'.

Formulazione: "0 fit falliti" si riscrive "nessun fit numericamente fallito (eccezione, valori non finiti, RMS sui
punti registrati > 5 mm)"; l'aderenza alla superficie e' l'errore di questa sezione.

## 4. Colonne nuove

`<modello>_va_{coef,fr,sr}` e `<modello>_vb_{coef,fr,sr}` (modello `gnm`, `flame2023`), definite come la sez. 4 del
protocollo sui beta delle varianti. Su `faceverse` la variante A e' "espressione libera col prior corretto"; altrove
"senza espressione".

## 5. Delta appaiati

Lo stesso `bp_paired.py` (righe, GT, repliche, semi e `fact_paired` importato in sola lettura), con le 12 colonne
nuove accanto alle 7 originali; maschera comune per dominio come nel protocollo (un fit fallito di A o B toglie le sue
righe a TUTTI i metodi, e lo si riporta). Uscite in file nuovi (`paired_e1.csv`, `spearman_e1.csv`,
`controls_e1.json`); `paired.csv` resta quello preregistrato. Controllo: sulle stesse righe le colonne originali e i
bracci devono ridare `paired.csv`. Si riportano gli stessi delta della sez. 7 del protocollo (factorized `form_cal` e
ctrlfr con FR, factorized `shape` e ctrlfr con SR) contro le colonne nuove, e si dice se qualche cella passa a "IC
sotto 0" (concorrente davanti al braccio).

## 6. Controlli

1. Determinismo del NICP: il residuo NICP ricalcolato coincide con `nicp_resid_mm` di `fit.npz` (scarto massimo).
2. FLAME su `faceverse`, variante A = fit originale (s = 1): scarto massimo dei beta.
3. Colonne originali e bracci in `paired_e1.csv` = `paired.csv` (stesse righe).

## 7. Testi corretti da questo emendamento

- **Ritirata l'avvertenza 3 della sez. 2 del protocollo** ("FLAME su FaMoS e' quasi un oracolo ... limite
  superiore"): i numeri la smentiscono (FLAME su FaMoS con FR: coefficienti 0.457, mesh d'identita' FR 0.565, sotto
  GNM 0.624 / 0.586 e sotto i bracci). Resta il fatto: la GT di FaMoS viene da registrazioni con la topologia di FLAME;
  il fit qui passa dal NICP sulle scansioni grezze e non ne trae un vantaggio osservabile.
- **Perimetro dell'esito.** "Nessun concorrente batte i bracci dichiarati" vale per: factorized con d_F calibrata
  (`form_cal`, letta con FR) o d_P (`shape`, letta con SR), e ctrlfr (||z||, senza calibrazione, con FR e SR). NON
  vale per le letture non dichiarate: con d_F NON calibrata factorized e' pari a FLAME mesh d'identita' FR su HIFI3D
  (s1234 +0.003 [-0.073, +0.081]; s2345 -0.008 [-0.088, +0.074]); C3M (`factorizedc3m`) con d_F non calibrata perde
  su HIFI3D FR contro FLAME mesh FR (e123 -0.111 [-0.218, -0.009], e205 -0.131 [-0.236, -0.026]) e, e205, contro GNM
  coefficienti (-0.111 [-0.220, -0.003]); d_P letta con FR e u di dual perdono su HIFI3D FR contro piu' concorrenti
  (letture non dichiarate: d_P e' la distanza per SR).
- **Controllo varifold GPU/CPU** (job 1067582, `bp_varifold.py --check 5` su `hifi3d`, L40S contro CPU, stesso codice
  float32): scarto relativo massimo per coppia dei prodotti interni da 3.0e-4 a 6.2e-4 sulle 5 coppie, SISTEMATICO (in tutte le 15
  celle il valore GPU e' maggiore di quello CPU). Il protocollo (sez. 4) riportava <= 1e-4 per float32 contro float64
  sul pilota: e' un confronto diverso, e questo scarto e' piu' grande. L'effetto sullo Spearman del varifold non e'
  misurato.

## 8. Valori congelati dal pilota

Aggiunta del 10 ottobre 2026, dopo il pilota e PRIMA di ogni calcolo sulle viste valutate (la sez. 1-7 e' quella del
commit 3595f0a, sha256 `63569a41...`). Job 1067652 (GNM, 4 min 37 s) e 1067653 (FLAME, 13 min 47 s), 40 core CPU,
`bp_fit_e1.py --pilot`; 200 mesh per modello, 0 fit falliti in tutte le 20 configurazioni (originale, A, 18 di B).
Punteggi completi in `pilot_e1/<modello>.json`. Applicata la regola della sez. 2:

| modello | S originale | S variante A | S minimo di B | entro l'1% | scelta (regola di parita') | S scelta |
| --- | --- | --- | --- | --- | --- | --- |
| GNM | 0.1518 | 0.1513 | 0.1559 (sigma 2, tau 5, 5 iter.) | sigma 2: tau 5 e 10, 5 iter. | **sigma 2 mm, tau 10 mm, 5 iterazioni** | 0.1566 |
| FLAME 2023 Open | 0.2304 | 0.2260 | 0.2090 (sigma 2, tau 2, 10 iter.) | sigma 2: (2, 10), (5, 5), (5, 10), (10, 10) | **sigma 2 mm, tau 5 mm, 5 iterazioni** | 0.2106 |

`bp.LOOP` = GNM {sigma 2.0, tau 10.0, iters 5}, FLAME 2023 {sigma 2.0, tau 5.0, iters 5}. Da dichiarare con i numeri:
1. entrambe le scelte stanno sul bordo superiore della griglia di sigma (2 mm): un prior piu' forte potrebbe dare S
   minore; la griglia non si allarga (sarebbe una seconda scelta dopo aver visto il pilota);
2. sul pilota, per GNM il ciclo NON migliora S rispetto all'originale (0.1566 contro 0.1518): migliora FaceScape
   (0.252 -> 0.160) e peggiora FaceVerse con espressioni (0.197 -> 0.302) e neutra (0.077 -> 0.087); per FLAME migliora
   (0.230 -> 0.211), con lo stesso peggioramento su FaceVerse con espressioni (0.288 -> 0.375);
3. l'errore di superficie sul pilota scende con B da 1.1-1.6 mm di mediana (p95 3.4-5.5 mm) a 0.3-0.6 mm (p95
   1.0-2.0 mm) in tutte le viste.
