# Studio umano v2: protocollo (9 ottobre 2026, revisione 4, PRIMA di ogni risposta umana)

Nessuna risposta alla v2 esiste: la pagina `docs/human_study_v2/index.html` non e' mai stata distribuita, e le
sole risposte analizzate sono quelle simulate. Le revisioni precedenti, tutte scartate prima della raccolta:
- 1 (`0c7ed04d...`): BFM REMESH;
- 2 (`9336b3aa...`): GNM, 2 strati;
- 3 (`99a2c1db...`): 60 prove, giudicata dal critic "RISERVE".

La revisione 4 cambia:
- la sessione, ridotta a circa 20 prove su richiesta dell'utente;
- la pagina, ridisegnata;
- l'inferenza, con test a cascata e soglia tarata;
- le letture di `F_vs_size` e `S_vs_maxabs`, una ciascuna;
- la frase di potenza;
- il consenso, che ora dichiara anche `localStorage`.

Hash di questo file in `PROTOCOL.sha256`. Ogni modifica successiva va in un emendamento datato in fondo, senza
guardare i dati.

## 0. Scopo e letture dichiarate

Arbitro (b) di `paper/PLAN_MASSIVE.md` §19: che cosa guida il giudizio umano di somiglianza fra volti 3D.

| ordine | strato | lettura se q > 0.5 | lettura se q < 0.5 |
|---:|---|---|---|
| 1 | `F_vs_S` (principale) | **la taglia conta per la somiglianza percepita** | la taglia non conta (a favore di S) |
| 2 | `F_vs_size` | a taglia quasi uguale conta la forma | a taglia quasi uguale decide la taglia |
| 3 | `S_vs_maxabs` | la shape di Procrustes e' piu' vicina al giudizio umano di maxabs | maxabs e' piu' vicina al giudizio umano della shape di Procrustes |

q e' la quota delle risposte che stanno con la prima GT dello strato. Tre avvertenze sulla lettura:
- `F_vs_S` non distingue F da "solo taglia" ("solo taglia" sta con F nel 100% dello strato): la sua lettura e'
  solo "la taglia conta".
- F si attribuisce soltanto dallo schema congiunto: `F_vs_S` > 0.5 e `F_vs_size` > 0.5.
- In `S_vs_maxabs` c'e' un'unica lettura per verso. Il meccanismo piu' probabile di un esito q < 0.5, cioe' che
  maxabs normalizzi di fatto per l'altezza, e' discusso come limite (sez. 7), non come seconda lettura.

Sotto la potenza dichiarata (sez. 6) un esito non significativo si riporta con l'IC, non come "nessuna
differenza".

## 1. Stimoli

- **Dominio: GNM Head v3.0** (Apache-2.0 su codice e pesi; citato nella pagina). 100 identita' campionate con
  z ~ N(0, 1) sui 170 modi `head_*`, seed 1234, nel frame metrico del modello. CV della centroid size della
  regione 5.3%.
- **Geometria dei render:**
  - frame di GT-F di E12, poi la rigida robusta per identita' verso mu (la stessa della GT F); mai una scala per
    mesh;
  - triangoli di `hockey_mask` ∩ `skin_exterior` piu' `eye_exteriors`;
  - camera ortografica unica, viste a 0, 45 e 90 gradi;
  - grigio uniforme e luce fissa.
- **Immagine:** striscia verticale di 3 viste, **512 x 1544 px per tutti i volti**, 2.327 px/mm. Altezza
  frontale del volto da 345 a 467 px. Altezza in px contro mm: r = 0.99998, errore massimo 0.64 px.
- **Nella pagina:** le tre strisce di una prova hanno la stessa larghezza CSS (una sola variabile, `--col`), e
  quindi lo stesso ingrandimento.
  - Al caricamento la pagina verifica che OGNI immagine misuri esattamente 512 x 1544 (`meta.image_px`): un'unica
    immagine diversa ferma lo studio (verificato in V8).
  - Sul telefono le colonne restano tre e ognuna impila le tre viste in verticale, alla stessa scala per tutti i
    volti.

## 2. GT

Invariate (`gt_v2.py`, funzioni di `v3_work/canonical_gt/cgt.py`):
- **F:** rigida robusta per identita', in mm;
- **S:** la stessa centrata e scalata alla centroid size di mu;
- **maxabs:** legacy;
- **"solo taglia":** |Δ log CS|;
- **secondarie registrate:** EDM, EDM_s, unified, F_rig_ls, F_pure, "solo altezza".

## 3. Triplette (`select_triplets_v2.py`, `STRATA`, seed 1234)

Pool di 485.100 triplette. In ogni strato le due GT sono opposte, ciascuna con margine relativo ≥ 0.10. Si estrae
a caso fra le 3n col margine minimo piu' grande. Un soggetto compare al massimo in 25 test; tutti i 100 sono
usati.

| strato | triplette nel pool | prove per sessione | vincoli | margine minimo (mediana / min) | contrasto di taglia \|Δ d_size\| |
|---|---:|---:|---|---|---|
| `F_vs_S` | 200 | 12 | nessuno | 0.46 / 0.43 | 0.124 (0.076-0.200) |
| `F_vs_size` | 80 | 3 | \|Δ d_size\| ≥ 0.02 | 0.59 / 0.55 | 0.025 (0.020-0.056) |
| `S_vs_maxabs` | 80 | 3 | \|Δ d_size\| ≤ 0.01; F e "solo taglia" al 50% | 0.26 / 0.11 | 0.007 (0.000-0.010) |

**Composizione degli strati** (quota delle triplette in cui la GT sta con la prima dello strato):
- `F_vs_S`: EDM, "solo taglia" e "solo altezza" stanno con F (99-100%); unified e maxabs con S (0-5%).
- `F_vs_size`: S, maxabs, unified ed EDM stanno con F (99-100%).
- `S_vs_maxabs`: F 0.50 e "solo taglia" 0.50, bilanciati; "solo altezza" 0.36.

**Visibilita' del contrasto in `F_vs_size`.** Il contrasto di taglia (2-5%) vale circa 10 px di altezza del volto
nell'immagine. A schermo e' circa 7 px su desktop (colonna di circa 360 px) e circa 2 px su un telefono (colonna
di circa 120 px): probabilmente sotto la soglia visiva. Per questo la lettura e' "a taglia quasi uguale".

**Controlli e prova:**
- 30 controlli unanimi su F, S, EDM, unified e maxabs, con margine ≥ 0.40 su ognuna;
- 6 triplette di prova, disgiunte dai controlli.

Impronta delle triplette mostrate: **`triplets_hash` = `1b116bc49c39c436`**.

## 4. Sessione e pagina

- **Sequenza:** 2 prove di esercizio, una schermata di transizione, poi 18 test e 2 controlli, uno per blocco di
  9. In tutto 22 schermate, circa 4 minuti.
- **Composizione dei test:** 12 `F_vs_S`, 3 `F_vs_size` e 3 `S_vs_maxabs`, estratti a caso dal pool di ogni strato
  con un seme dato dal codice partecipante. L'insieme dei partecipanti copre tutto il pool.
- **Perche' 12/3/3:** la potenza di `F_vs_S` cambia poco fra 12 e 18 prove a testa (esplorazione ad alfa 0.05, caso prudente, N per l'80%:
  50 con 12 o 14, 40-50 con 18; alla soglia tarata 0.04 lo schema scelto ne richiede 60); conta di piu' il pool, meglio 200 triplette che 100. Le 6 prove restanti tengono
  vivi i due strati secondari come stima.
- **Pagina:**
  - un solo carattere (Inter, Google Fonts), palette neutra, tema chiaro e scuro con `prefers-color-scheme`;
  - barra di avanzamento, pulsanti grandi Left / Right, tasti ←/→ e 1/2, clic o tocco sul volto;
  - pagina finale con il codice partecipante.
- **Testo:** neutro. La domanda e' "Which face is more similar to the reference face?" e l'unica indicazione e'
  "Ignore light and shadows". La parola "shape" non compare, e la resa comune ("grey material, same lighting, same
  camera") e' detta una sola volta.
- **Consenso:** elenca cio' che si registra (scelte, tempi, browser, dimensioni della finestra, DPR) e la nota in
  `localStorage` (avanzamento e flag "gia' partecipato"). Dichiara che font e invio passano da Google, e cita GNM
  Head.
- **Doppia partecipazione:** dopo un invio riuscito, lo stesso browser mostra "already taken part".
- **Payload:** `study_version: "v2"`, id `v2_*`, `triplets_hash`.

## 5. Analisi (`analyze_v2.py`), fissata ora

- **Inclusione:** `study_version == "v2"` e `triplets_hash == 1b116bc49c39c436`. Un codice ripetuto conta una volta.
  Le prove di esercizio non contano.
- **Esclusione:** almeno un errore sui 2 controlli, oppure nessun controllo visto. Con 2 controlli la regola "al
  massimo 1 errore su 4" della v1 diventa "nessun errore su 2" (`--max-control-errors 0`).
- **Statistica:** in ogni strato la quota q delle risposte con la prima GT, con H0 q = 0.5. L'errore standard e'
  quello per disegno incrociato partecipanti x triplette, V = V_P + V_T - V_0 (Owen 2007), con bootstrap da 2.000
  repliche su partecipanti e triplette VISTE dello strato, seed 1234. p e IC 95% si leggono su una t di
  Satterthwaite.
- **Test a cascata** (gatekeeping, ordine della sez. 0): ogni strato si testa solo se il precedente ha rifiutato
  H0, ciascuno alla soglia nominale tarata **0.04**. L'errore complessivo resta ≤ alfa senza dividerlo.
- **Taratura prima dei dati** (`power_v2.py`): sotto H0 si misura l'alfa empirico alle soglie 0.025-0.05 in
  84 condizioni (3 strati, sd fra partecipanti 0.5 e 1.0, sd fra triplette 0 e 0.8, N da 15 a 120, 4.000
  studi ciascuna). La soglia tarata e' la piu' grande con alfa empirico ≤ 0.05 in TUTTE le condizioni:
  **0.04** (massimo empirico 0.050; a 0.05 il massimo sarebbe 0.061).
- **Secondarie, descrittive:**
  - IC sui soli partecipanti e p a segni ribaltati (il cui alfa reale arriva a 0.19 con eterogeneita' fra
    triplette);
  - accordo complessivo per GT e differenze appaiate;
  - maggioranza per tripletta e kappa di Fleiss.
- **Numerosita':** obiettivo **60 partecipanti tenuti** (sez. 6). Una sola analisi confermativa, al
  raggiungimento o alla chiusura. Prima di allora si contano solo partecipanti ed esclusi.

## 6. Potenza (`power_v2.py`, `power_v2.md`), alla soglia tarata

Simulazione con lo stesso test:
- 12 prove su 200 triplette (`F_vs_S`), 3 su 80 e 3 su 80;
- effetti logit per partecipante con sd 0.5 o 1.0, e per tripletta con sd 0 o 0.8;
- 1.000 studi per cella.

**Caso prudente:** sd 1.0 fra partecipanti e 0.8 fra triplette. N tenuti:

| q | differenziale | F_vs_S 80% | F_vs_S 90% | F_vs_size 80% | S_vs_maxabs 80% |
|---:|---:|---:|---:|---:|---:|
| 0.575 | 0.15 | 100 | 150 | 300 | 300 |
| 0.6 | 0.20 | 60 | 70 | 120 | 120 |
| 0.65 | 0.30 | 25 | 30 | 50 | 50 |
| 0.7 | 0.40 | 15 | 20 | 25 | 25 |

**L'80% di potenza vale per differenziali ≥ 0.20 (q ≥ 0.60) su `F_vs_S`; sotto, solo intervallo di
confidenza.** Gli strati secondari, con 3 prove a testa, servono come stima con IC: hanno l'80% solo per
differenziali ≥ 0.30, e a N = 60 la loro potenza a 0.20 e' 0.57 (F_vs_size) e 0.56 (S_vs_maxabs).

**Differenziale di pianificazione: 0.20.**
- Nella v1 l'accordo per risposta era 0.619 (LPIPS) contro 0.488 (maxabs), un differenziale di 0.13, con margini
  circa 3 volte piu' piccoli di questi.
- In `F_vs_S` il contrasto di taglia e' grande: 0.124 in log della centroid size, cioe' circa 50 px di altezza
  del volto nell'immagine.

## 7. Limiti dichiarati ora

- **Un solo dominio, sintetico** (GNM).
- **Altezza in `S_vs_maxabs`.** "Solo altezza" sta con maxabs nel 64% dello strato: chi giudicasse con la sola
  altezza del volto spingerebbe q sotto 0.5. Il simulato che segue "solo altezza" da' q circa 0.46. Un esito
  q < 0.5 resta "maxabs piu' vicina al giudizio umano"; che il meccanismo sia l'altezza e' un'ipotesi, non una
  conclusione dello studio.
- **Contrasto di `F_vs_size`.** E' forse invisibile (sez. 3); per questo la lettura e' "a taglia quasi uguale".
- **Taglia giudicata per confronto.** Si vede solo confrontando i tre volti, a parita' di camera e ingrandimento.
- **Rigida robusta nei render,** come la GT F.
- **Doppia partecipazione.** Il flag in `localStorage` non impedisce di partecipare da un altro browser.
- **Pochi controlli.** Con 2 controlli l'esclusione e' piu' grossolana di quella della v1.
