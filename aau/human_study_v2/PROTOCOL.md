# Studio umano v2: protocollo (scritto il 9 ottobre 2026, PRIMA di ogni risposta umana)

Nessuna risposta alla v2 esiste a quest'ora: la pagina `docs/human_study_v2/index.html` non e' stata ancora
distribuita. Le sole "risposte" analizzate finora sono i partecipanti simulati del self-test. Hash del file in
`PROTOCOL.sha256`. Ogni modifica successiva va in un emendamento datato in fondo, con il motivo, e non puo'
guardare i dati.

## 0. Scopo

Arbitro (b) di `paper/PLAN_MASSIVE.md` §19: quale GT di E12 (`aau/runs/evidence/e12/protocol.md`, Emendamento 1)
somiglia di piu' al giudizio umano di somiglianza della forma del volto. GT a confronto: **GT-F** (forma metrica in
mm, frame canonico), **GT-S** (forma senza taglia, centroid size), **GT-EDM** (distanze interne, senza
allineamento), **maxabs** (legacy, quella della v1 e del paper). La v1 non puo' farlo: i suoi render normalizzano
ogni mesh con maxabs (`aau/baselines/render_cache.py`, righe 7-18), quindi mostrano volti alti uguali, e sono solo
frontali, quindi nascondono la profondita'.

## 1. Stimoli

- **Dominio:** BFM REMESH, topologia `original`, i 100 soggetti held-out della v1 (`common.subject_set("heldout")`).
  Motivi: (i) i render JPEG dei volti BFM sintetici sono gia' pubblicati con la v1, mentre la licenza di FaceScape
  vieta la distribuzione e permette i render solo dei soggetti "portrait-authorized"
  (`literature/DATA_ACCESS_2026-10-07.md`); (ii) la variabilita' di taglia non e' migliore su FaceScape dev: CV
  della centroid size della regione 1.5% (100 identita' dev) contro 1.9% (BFM, 100 held-out), misurati qui;
  (iii) continuita' con la v1, stessi volti.
- **Geometria:** mesh intera della REMESH nel frame di GT-F in mm, `f = u_d R_d V + t_d` con il frame di E12
  (`aau/runs/evidence/e12/frames.json`, dominio `bfm`: u = 0.001, unita' dichiarata). Una trasformazione per il
  dominio, nessuna per mesh.
- **Camera:** ortografica, 1.959 px/mm, finestra di 196 mm (semilato 98), uguale per ogni volto e ogni vista,
  calcolata una volta sull'unione dei 100 volti nelle 3 viste (`render_v2.compute_camera`). Nessuna inquadratura
  per volto.
- **Viste:** la testa ruota attorno a +y per un punto fisso (centroide della regione unificata sulla media FLAME):
  0 gradi (frontale), 45 (3/4), 90 (profilo, naso a destra). Ray casting (visibilita' esatta).
- **Materiale e luce:** grigio uniforme (albedo 0.8), Lambert con normali interpolate, luce direzionale fissa nel
  frame della camera (alto a sinistra, davanti) + ambiente 0.22, sfondo nero, supersampling 3x3.
- **Immagine:** una striscia verticale per volto (frontale, 3/4, profilo; 384 x 1160 px, JPEG q90).
- **Pagina:** tre colonne, candidato | riferimento | candidato. Le righe allineano la stessa vista dei tre volti.
  Il riferimento sta al centro, con una cornice colorata.
- **Controllo della scala** (`render_check.json`): l'altezza della silhouette in pixel contro l'altezza della mesh
  in mm ha r = 0.9999 in ogni vista. L'errore massimo e' di 0.6 px. Nessuna silhouette tocca il bordo. Le altezze
  vanno da 158 a 182 mm, cioe' da 309 a 357 px. La centroid size va da 51.7 a 56.5 mm (rapporto 1.094).

## 2. Triplette

GT: le matrici di E12 calcolate con lo stesso codice (`cgt.all_gts`) sui 100 soggetti (`gt_v2.py`). E12 non
produce matrici BFM. Maxabs e' la GT legacy della v1 (`common.load_gt_submatrix`). Stato: **definitivo**. E12 ha
concluso il passo delle GT (`gt.json`) e le triplette sono identiche a quelle della selezione provvisoria (stesse
distanze, stessa impronta `d46022b22bdb7bd1`). Se E12 cambiasse ancora `cgt.py` o `gt.py` prima dell'avvio della
raccolta, si rifanno `HS2_STEPS="gt select"`. Una volta avviata la raccolta, l'insieme non si cambia.

- Pool completo: 485.100 triplette (A riferimento, {B, C}).
- **Tipi** (4 x 60 = 240 test): `F_vs_S`, `F_vs_maxabs`, `S_vs_maxabs`, `EDM_vs_F`. Una tripletta e' del tipo
  `X_vs_Y` se X e Y ordinano d(A,B) e d(A,C) al contrario, ciascuna con margine relativo
  `|d(A,B) - d(A,C)| / media >= 0.10`. Estrazione casuale (seed 1234) fra le 300 col margine minimo piu' grande.
  Nessuna tripletta e' riusata fra i tipi. Un soggetto compare in al massimo 15 test.
  Margine minimo mediano nelle scelte: 0.138 (F_vs_S), 0.133 (F_vs_maxabs), 0.167 (S_vs_maxabs), 0.378 (EDM_vs_F).
- **Controlli:** 30 triplette unanimi su F, S, EDM, EDM_s, unified, maxabs, con margine >= 0.40 su ognuna.
- **Prova:** 6 triplette unanimi con la stessa regola, disgiunte dai controlli.
- Ogni tripletta porta le distanze e la risposta attesa di tutte le 12 GT su disco. Sono in `triplets.json` e
  servono solo all'analisi. La pagina non le riceve.

## 3. Sessione

3 prove di prova, poi una schermata di transizione. Seguono 36 test, 9 per tipo pescati a giro, e 4 controlli, uno
per blocco di 9 test. Ordine e lato (sinistra/destra) sono casuali e seminati dal codice partecipante. Istruzioni:
"Pick the candidate that looks more similar to the reference", "Ignore light and shadows: look at the shape",
"same camera, at the same scale". Non si chiede esplicitamente di guardare la taglia: si dice solo che la scala e'
la stessa per tutti. Invio al Google Form condiviso con la v1, con `study_version: "v2"`, id `v2_*` e
`triplets_hash`.

## 4. Analisi (`analyze_v2.py`), fissata ora

- **Inclusione:** payload con `study_version == "v2"` e `triplets_hash` uguale a quello di `triplets.json`. Un
  codice partecipante ripetuto conta una volta (la sessione piu' lunga). Le prove di prova non contano.
- **Esclusione** (regola della v1): fuori chi sbaglia piu' di 1 controllo sui 4. Chi non ha nessun controllo
  (sessione interrotta presto) e' fuori.
- **Test primari (4):** per ogni tipo `X_vs_Y`, la quota q delle risposte che stanno con X. Dentro lo strato
  accordo(X) - accordo(Y) = 2q - 1. H0: q = 0.5. Statistica: sum_p (risposte con X - risposte con Y). p a due code
  dalla permutazione a segni ribaltati per partecipante (10.000). Correzione di Holm sui 4 tipi, alfa 0.05. IC 95%
  di q: bootstrap sui partecipanti (2.000 repliche, seed 1234).
- **Lettura** (dichiarata ora):
  - `F_vs_S` con q > 0.5 significativo: la taglia assoluta conta per la somiglianza percepita (a favore di F).
    Con q < 0.5: no (a favore di S).
  - `F_vs_maxabs` e `S_vs_maxabs` con q > 0.5: la normalizzazione maxabs e' peggiore della GT principiata.
  - `EDM_vs_F`: vedi i limiti (sez. 6). Un esito a favore di EDM non distingue "senza allineamento" da
    "insensibile alla posa residua".
- **Secondarie:** accordo complessivo di ogni GT su tutte le 240 triplette (stesse risposte, stesse repliche). Le
  differenze appaiate fra GT si leggono con IC bootstrap, P(diff <= 0) e p a segni ribaltati, senza correzione,
  come descrittive. Sensibilita': bootstrap incrociato partecipanti x triplette. Maggioranza per tripletta
  (>= 3 voti) e kappa di Fleiss, come la v1.
- **Numerosita'** (sez. 5): obiettivo 100 partecipanti tenuti, minimo 60. Una sola analisi confermativa, quando si
  raggiungono 100 tenuti o alla chiusura della raccolta. Prima di allora si contano solo partecipanti ed esclusi,
  senza calcolare l'accordo.

## 5. Potenza (`power_v2.py`, `power_v2.md`)

Simulazione: 9 prove per strato e partecipante su 60 triplette, effetti casuali logit per partecipante
(sd 0.5) e per tripletta (sd 0 oppure 0.8), test come sopra (approssimazione normale della permutazione), 4.000
studi per cella. Differenziale realistico: nella v1 (7 partecipanti, triplette dove le metriche litigano)
l'accordo per risposta era 0.619 per LPIPS e 0.488 per la GT maxabs, un differenziale di 0.13. La sd fra
partecipanti oltre la binomiale era 0.21-0.38 logit, quindi 0.5 e' prudente.

| q nello strato | differenziale | N per 80%, alfa 0.05 | N per 80%, alfa 0.0125 (Holm, caso peggiore) | N per 90%, alfa 0.0125 |
|---:|---:|---:|---:|---:|
| 0.55 | 0.10 | 140-200 | 200-300 | 250->400 |
| 0.575 | 0.15 | 60-70 | 90-100 | 120-160 |
| 0.60 | 0.20 | 35 | 50 | 70 |
| 0.65 | 0.30 | 15-20 | 25 | 30 |

Gli intervalli vanno da sd fra triplette 0 a 0.8. Con 100 partecipanti tenuti si rileva un differenziale di 0.15
in uno strato all'80% anche alla soglia prudente. Un differenziale di 0.10 richiede 200-300 partecipanti: se l'esito
e' nullo con 100, si riporta l'IC, non "nessuna differenza".

## 6. Limiti dichiarati ora

- **Taglia poco variabile.** Le REMESH sono state allineate per similarita' una per una prima di esistere qui
  (`v3_work/unified_gt/domains.py`, `bfm()`). La CV della centroid size e' 1.9%, quella dell'altezza 3.5%.
  `F_vs_S` sceglie proprio le triplette dove la taglia decide: la baseline "solo taglia" sta con F nel 100% dello
  strato. Il contrasto visivo resta pero' di pochi punti percentuali (fino a 48 px di differenza d'altezza su
  ~330).
- **EDM_vs_F e posa residua.** Su BFM, F conserva il residuo rigido di quell'allineamento: Spearman fra F e
  F_rig_ls 0.81 sulle 4.950 coppie. Nello strato `EDM_vs_F`:
  - le varianti rigide di F stanno con EDM nel 65-67% delle triplette;
  - "solo taglia" sta con EDM nel 97%.
  Lo strato confronta quindi soprattutto posa residua e taglia. La posa e' visibile nei render, come la vedrebbe
  F. Le GT rigide (`F_rig_ls`, `F_rig_rob`) sono nelle secondarie.
- **Un solo dominio.** Le conclusioni valgono per volti BFM sintetici in topologia unica.
