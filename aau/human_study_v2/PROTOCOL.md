# Studio umano v2: protocollo (9 ottobre 2026, revisione 2, PRIMA di ogni risposta umana)

Nessuna risposta alla v2 esiste: la pagina `docs/human_study_v2/index.html` non e' mai stata distribuita, e le
sole risposte analizzate sono quelle simulate del self-test. Questa revisione sostituisce la prima (hash
`0c7ed04d...`, dominio BFM REMESH, 4 strati da 9 prove), scartata prima della raccolta su richiesta del PI. I motivi:
- su BFM la taglia era gia' normalizzata (CV 1.9%);
- la potenza era insufficiente con un numero di partecipanti realistico;
- E12 ha fissato F = rigida robusta.
Hash di questo file in `PROTOCOL.sha256`. Ogni modifica successiva va in un emendamento datato in fondo, senza
guardare i dati.

## 0. Scopo

Arbitro (b) di `paper/PLAN_MASSIVE.md` §19. Due domande, in ordine d'importanza:
1. **Form contro shape:** nel giudizio umano di somiglianza della forma del volto conta la taglia assoluta (GT-F)
   o no (GT-S)?
2. **Shape contro maxabs:** la normalizzazione legacy per max|coord| e' peggiore della shape di Procrustes?

La v1 non puo' rispondere. I suoi render normalizzano ogni mesh con maxabs (`aau/baselines/render_cache.py`) e
sono solo frontali.

## 1. Stimoli

- **Dominio: GNM Head v3.0** (`~/data/gnm_head/gnm_head.npz`). Licenza Apache-2.0 su codice e pesi (scheda HF:
  "suitable for both academic research and commercial applications"): render pubblicabili, con citazione
  (ploumpis2026gnmhead).
  - 100 identita' campionate come i set zero-shot: z ~ N(0, 1) sui 170 modi `head_*`, senza code ne' troncamento
    (`v3_work.mm`, `sample_identity(tails=False, trunc=0)`), seed 1234.
  - Testa nel frame metrico del modello, nessuna normalizzazione.
  - CV della centroid size della regione: 5.3% (E12, `size_cv.csv`: 5.3% su 10.100 identita' GNM; qui 5.28% sulle
    100). Gli altri domini: ICT 4.8% (MIT), BFM REMESH 1.9%, FaceScape 1.6%. GNM ha la variabilita' piu' vicina
    ai valori antropometrici attesi (4-8%) ed e' pubblicabile.
- **Geometria dei render:** la mesh va nel frame di GT-F di E12 (`frames.json`, dominio `gnm`: u = 1000, una
  trasformazione per dominio). Poi si applica la rigida robusta della sua regione verso mu, la stessa della GT F:
  rotazione e traslazione, mai scala. Rotazione mediana 0.97 gradi, massima 3.9; IRLS a convergenza in 99 casi su 100.
- **Area renderizzata:** triangoli di `hockey_mask` ∩ `skin_exterior`, piu' `eye_exteriors` (gli occhi non restano
  fori). La regione unificata su cui si misurano le GT sta tutta dentro `hockey_mask` (verificato: 1478 su 1478).
- **Camera:** ortografica, 1.745 px/mm (finestra di 220 mm), uguale per ogni volto e ogni vista. E' calcolata una
  volta sull'unione dei 100 volti nelle 3 viste: nessuna inquadratura per volto.
- **Viste:** rotazione della testa attorno a +y per un punto fisso, a 0, 45 e 90 gradi (naso a destra). Ray
  casting.
- **Materiale e luce:** grigio uniforme (albedo 0.8), Lambert, normali interpolate, luce fissa nel frame della
  camera piu' ambiente 0.22, sfondo nero, supersampling 3x3.
- **Immagine:** una striscia verticale per volto (frontale, 3/4, profilo), 384 x 1160 px, JPEG q90.
- **Pagina:** candidato | riferimento | candidato, con le righe allineate per vista.
- **Controllo della scala** (`render_check.json`): altezza della silhouette in px contro altezza della mesh in mm,
  r = 0.99998. Errore massimo 0.7 px, nessuna silhouette sul bordo. Altezza frontale da 259 a 351 px (148-201 mm).
  Centroid size della regione da 48.7 a 63.3 mm (rapporto 1.30).

## 2. GT (`gt_v2.py`, funzioni di `v3_work/canonical_gt/cgt.py`)

Punti della regione unificata (1478) nel frame di GT-F in mm (X0). Rigida robusta per identita' verso mu
(`rigid_robust`, IRLS Tukey, E12 Emendamento 1) -> a_i.

- **F** = RMS pesata fra le a_i: forma metrica con taglia. E' la GT form di riferimento di E12; la F pura misura la
  posizione del volto nel frame del modello, che non e' osservabile.
- **S** = a_i centrata sul suo centroide pesato e scalata alla centroid size di mu: Procrustes pieno con la
  rotazione robusta. La S di E12 "a punto fisso" non e' usata, perche' non toglie la traslazione.
- **maxabs** = GT legacy della pipeline zero-shot: patch `hockey_mask` di `v3_work.mm`, maxabs per mesh, media
  per vertice della distanza L2 (`gt.maxabs_matrix`).
- **Registrate per l'analisi secondaria:** EDM, EDM_s, unified, F_rig_ls, F_pure, "solo taglia", "solo altezza".
- **Spearman sulle 4.950 coppie:**
  - F con S 0.59 (su BFM era 0.96);
  - F con "solo taglia" 0.80;
  - S con maxabs 0.68;
  - S con unified 0.999.

## 3. Triplette (`select_triplets_v2.py`, seed 1234)

Pool di 485.100 triplette. Uno strato `X_vs_Y` contiene le triplette in cui X e Y ordinano d(A,B) e d(A,C) al
contrario, ciascuna con margine relativo ≥ 0.10.

**Margini piu' grandi possibile:** dentro ogni strato si estrae a caso fra le 3n triplette col margine minimo piu'
grande. Margine minimo mediano nelle scelte: 0.48 (F_vs_S) e 0.42 (S_vs_maxabs), contro 0.13-0.17 nella prima
revisione.

| strato | ruolo | triplette | prove per sessione | disponibili |
|---|---|---:|---:|---:|
| `F_vs_S` | principale | 120 | 36 | 57.506 |
| `S_vs_maxabs` | secondario confermativo | 80 | 24 | 37.523 |

`F_vs_S` contrappone "con taglia" a "senza taglia". Dalla parte di F stanno EDM e "solo taglia" (100% delle
triplette). Dalla parte di S stanno unified ed EDM_s (100%) e maxabs (97%). `EDM_vs_F` e `F_vs_maxabs` sono
eliminati. Nessuna tripletta e' riusata e ogni soggetto compare al massimo in 15 test. Controlli: 30 triplette
unanimi su F, S, EDM, unified e maxabs, con margine ≥ 0.40 su ognuna. Prova: 6 triplette con la stessa regola,
disgiunte dai controlli. Impronta delle triplette mostrate: **`triplets_hash` = `30f02a3b4150927c`**.

Stato: definitivo. E12 ha concluso il passo GT (`gt.json`); impronta di `cgt.py` + `gt.py` nel manifest. Una
volta avviata la raccolta, l'insieme non cambia.

## 4. Sessione

La sessione ha questa sequenza:
1. 3 prove di prova;
2. una schermata di transizione;
3. 60 test, 36 F_vs_S e 24 S_vs_maxabs, estratti a caso dentro lo strato;
4. 4 controlli, uno per blocco di 15.

Le 64 prove richiedono circa 10 minuti. Ordine e lato sono casuali, seminati dal codice partecipante. Istruzioni:
- "Pick the candidate that looks more similar to the reference";
- "Ignore light and shadows: look at the shape";
- "same camera, at the same scale".

Non si chiede esplicitamente di guardare la taglia. La pagina invia al Google Form condiviso con la v1, con
`study_version: "v2"`, id `v2_*` e `triplets_hash`.

## 5. Analisi (`analyze_v2.py`), fissata ora

- **Inclusione:** `study_version == "v2"` e `triplets_hash == 30f02a3b4150927c`. Un codice ripetuto conta una
  volta (la sessione piu' lunga). Le prove di prova non contano.
- **Esclusione** (regola della v1): piu' di 1 errore sui 4 controlli, oppure nessun controllo visto.
- **Test primari (2):** in ogni strato `X_vs_Y` si misura la quota q delle risposte con X; accordo(X) - accordo(Y)
  = 2q - 1, con H0 q = 0.5.
  - Il p a due code viene dalla permutazione a segni ribaltati per partecipante (10.000), con correzione di Holm
    sui 2 strati, alfa 0.05.
  - L'IC 95% di q viene dal bootstrap sui partecipanti (2.000 repliche, seed 1234).
- **Lettura:**
  - `F_vs_S` con q significativamente > 0.5: la taglia assoluta conta per la somiglianza percepita, a favore di F
    (e del ramo "form" della §19).
  - `F_vs_S` con q < 0.5: a favore di S.
  - Un esito non significativo si riporta con l'IC, non come "nessuna differenza".
  - `S_vs_maxabs` con q > 0.5: maxabs e' peggiore della shape di Procrustes.
- **Secondarie, descrittive:**
  - accordo complessivo di ogni GT sulle 200 triplette;
  - differenze appaiate (IC bootstrap, P(diff ≤ 0), p a segni ribaltati, senza correzione);
  - bootstrap incrociato partecipanti x triplette;
  - maggioranza per tripletta e kappa di Fleiss, come la v1.
- **Numerosita':** obiettivo 40 partecipanti tenuti, minimo 25 (sez. 6). Una sola analisi confermativa, a 40
  tenuti o alla chiusura. Prima di allora si contano solo partecipanti ed esclusi.

## 6. Potenza (`power_v2.py`, `power_v2.md`)

**Modello della simulazione:**
- prove per partecipante come nella sessione: 36 su 120 triplette e 24 su 80;
- effetti logit per partecipante con sd 0.5 (nella v1: 0.21-0.38 oltre la binomiale);
- effetti per tripletta con sd 0 oppure 0.8;
- 4.000 studi per cella.

La soglia prudente per uno strato con Holm su 2 e' alfa = 0.025. N tenuti:

| q | 2q - 1 | F_vs_S, 80% | F_vs_S, 90% | S_vs_maxabs, 80% | S_vs_maxabs, 90% |
|---:|---:|---:|---:|---:|---:|
| 0.55 | 0.10 | 100 | 120-150 | 100-120 | 150->200 |
| 0.575 | 0.15 | 40 | 50 | 50 | 60-70 |
| 0.60 | 0.20 | 25 | 30 | 25-30 | 35 |
| 0.65 | 0.30 | 12 | 15 | 12-15 | 15-18 |

**Differenziale realistico.** Nella v1 l'accordo per risposta su triplette con margine ≥ 0.15 era 0.619 per LPIPS
e 0.488 per maxabs, cioe' 0.13. Qui i margini sono circa 3 volte maggiori (mediana 0.42-0.48) e lo strato
principale contrappone una differenza di taglia fino al 30%, visibile. Si pianifica su **q = 0.60 (differenziale
0.20)**, con q = 0.575 come caso prudente. **40 partecipanti tenuti** danno:
- per F_vs_S, l'80% anche nel caso prudente e oltre il 90% a q = 0.60;
- per S_vs_maxabs, il 90% a q = 0.60.

## 7. Limiti dichiarati ora

- **Un solo dominio, sintetico.** Le identita' GNM sono campionate dal modello. La taglia varia con la
  distribuzione del modello (CV 5.3%), non con quella di una popolazione reale.
- **Nessun riferimento all'ambiente.** La taglia si vede solo per confronto fra i tre volti, a parita' di camera;
  manca un oggetto di riferimento nella scena. E' il caso d'uso di una ricostruzione metrica.
- **Rigida robusta anche nei render.** I render usano la rigida robusta per identita', come la GT F. La F pura
  (posizione nel frame del modello) e' nelle secondarie.
- **Un caso non a convergenza.** Su un'identita' su 100 l'IRLS della rigida robusta non converge entro 100 giri.
  Resta nello studio con la stima all'ultimo giro.
