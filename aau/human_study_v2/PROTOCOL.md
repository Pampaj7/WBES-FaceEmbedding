# Studio umano v2: protocollo (9 ottobre 2026, revisione 3, PRIMA di ogni risposta umana)

Nessuna risposta alla v2 esiste: la pagina `docs/human_study_v2/index.html` non e' mai stata distribuita, e le
sole risposte analizzate sono quelle simulate. Questa revisione sostituisce:
- la revisione 1 (hash `0c7ed04d...`, BFM REMESH);
- la revisione 2 (hash `9336b3aa...`, GNM con 2 strati), giudicata BLOCCANTE dal critic prima del lancio.

Le correzioni della revisione 3:
- `S_vs_maxabs` a taglia neutra e bilanciato rispetto a F;
- nuovo strato `F_vs_size`;
- istruzioni neutre;
- inferenza primaria incrociata partecipanti x triplette, con alfa verificato;
- potenza con sd 1.0 fra partecipanti;
- flag "gia' partecipato";
- consenso e citazione completi.

Hash di questo file in `PROTOCOL.sha256`. Ogni modifica successiva va in un emendamento datato in fondo, senza
guardare i dati.

## 0. Scopo e letture dichiarate

Arbitro (b) di `paper/PLAN_MASSIVE.md` §19: che cosa guida il giudizio umano di somiglianza della forma del volto.

| strato | contrasto | lettura se q > 0.5 (H0 q = 0.5 rifiutata) | lettura se q < 0.5 |
|---|---|---|---|
| `F_vs_S` (principale) | con taglia contro senza taglia | **la taglia conta per la somiglianza percepita** | la taglia non conta (a favore di S) |
| `F_vs_size` | forma (F) contro sola taglia | oltre alla taglia conta la forma, nel verso di F | la taglia domina la forma |
| `S_vs_maxabs` | shape di Procrustes contro maxabs, a taglia neutra | maxabs e' peggiore della shape | maxabs e' migliore |

`F_vs_S` da solo non distingue F da "solo taglia": "solo taglia" sta con F nel 100% dello strato. Per questo la
lettura e' "la taglia conta", non "F e' la GT giusta". `F_vs_size` scioglie il nodo. Un esito non significativo si
riporta con l'IC, non come "nessuna differenza".

## 1. Stimoli

Invariati dalla revisione 2.
- **Dominio: GNM Head v3.0** (Apache-2.0 su codice e pesi; citato nella pagina). 100 identita' campionate con
  z ~ N(0, 1) sui 170 modi `head_*`, seed 1234, nel frame metrico del modello. CV della centroid size della
  regione 5.3%.
- **Geometria dei render:**
  - frame di GT-F di E12 (`frames.json`, dominio `gnm`), poi la rigida robusta per identita' verso mu (la stessa
    della GT F); mai una scala per mesh;
  - triangoli di `hockey_mask` ∩ `skin_exterior` piu' `eye_exteriors`; la regione delle GT sta tutta nella
    maschera;
  - camera ortografica unica, 1.745 px/mm, viste a 0, 45 e 90 gradi;
  - grigio uniforme e luce fissa.
- **Controllo della scala:** altezza in px contro altezza in mm, r = 0.99998, errore massimo 0.7 px. Altezza
  frontale da 259 a 351 px.

## 2. GT

Invariate dalla revisione 2 (`gt_v2.py`, funzioni di `v3_work/canonical_gt/cgt.py`).
- **F:** rigida robusta per identita', in mm.
- **S:** la stessa centrata e scalata alla centroid size di mu.
- **maxabs:** legacy della pipeline zero-shot.
- **"solo taglia":** |Δ log CS| (CS della regione allineata).
- **Secondarie registrate:** EDM, EDM_s, unified, F_rig_ls, F_pure, "solo altezza".

## 3. Triplette (`select_triplets_v2.py`, `STRATA`, seed 1234)

Pool di 485.100 triplette. In ogni strato `X_vs_Y` le GT X e Y ordinano d(A,B) e d(A,C) al contrario, ciascuna
con margine relativo ≥ 0.10. Dentro lo strato (e dentro ogni cella) si estrae a caso fra le 3n col margine
minimo piu' grande. Un soggetto compare al massimo in 15 test.

| strato | triplette | prove per sessione | vincoli | disponibili | margine minimo (mediana / min) |
|---|---:|---:|---|---:|---|
| `F_vs_S` | 100 | 24 | nessuno | 57.506 | 0.48 / 0.45 |
| `F_vs_size` | 80 | 18 | \|d_size(A,B) - d_size(A,C)\| ≥ 0.02 | 16.853 | 0.59 / 0.55 |
| `S_vs_maxabs` | 80 | 18 | \|d_size(A,B) - d_size(A,C)\| ≤ 0.01; F e "solo taglia" con S nel 50% | 6.038 | 0.26 / 0.11 |

- **`F_vs_size`:** la differenza di taglia percepibile va dal 2.0 al 5.0% della centroid size (mediana 2.5%).
  F la scavalca per la forma: S, maxabs, unified ed EDM stanno con F nel 99-100%.
- **`S_vs_maxabs`:** la differenza di taglia sta sotto l'1% (mediana 0.66%). La cella "F contro S, taglia con S"
  ha solo 5 triplette, quindi le quote di cella (`cell_quotas`) rendono 50/50 le due marginali: F 0.50 e
  "solo taglia" 0.50 con S, verificato. "Solo altezza" sta pero' con S solo nel 36%: maxabs normalizza di fatto
  per l'altezza (sez. 7).
- **Controlli e prova:**
  - 30 controlli unanimi su F, S, EDM, unified e maxabs, con margine ≥ 0.40 su ognuna;
  - 6 triplette di prova con la stessa regola, disgiunte dai controlli.
- **Impronta delle triplette mostrate:** **`triplets_hash` = `9cea595d967d69ab`**.

## 4. Sessione

La sessione ha questa sequenza:
1. 3 prove di prova;
2. una schermata di transizione;
3. 60 test (24 F_vs_S, 18 F_vs_size, 18 S_vs_maxabs), estratti a caso dentro lo strato;
4. 4 controlli, uno per blocco di 15.

In tutto 64 prove, circa 10 minuti. Ordine e lato sono casuali, seminati dal codice partecipante.

**Istruzioni neutre.** Domanda: "Which face is more similar to the reference face? Ignore light and shadows." La
parola "shape" non compare. La resa comune ("grey material, same lighting, same camera") e' detta una sola volta,
senza enfasi.

**Consenso.** Elenca anche dimensioni della finestra e DPR. Cita GNM Head.

**Doppia partecipazione.** Dopo un invio riuscito, `localStorage` conserva il flag "gia' partecipato": lo stesso
browser mostra un messaggio invece di una nuova sessione. Un altro browser non e' bloccato.

**Payload.** Invio al Google Form condiviso con la v1, con `study_version: "v2"`, id `v2_*` e `triplets_hash`.

## 5. Analisi (`analyze_v2.py`), fissata ora

- **Inclusione:** `study_version == "v2"` e `triplets_hash == 9cea595d967d69ab`. Un codice ripetuto conta una volta.
  Le prove di prova non contano.
- **Esclusione:** piu' di 1 errore sui 4 controlli, oppure nessun controllo visto.
- **Test primari (3).** In ogni strato si misura la quota q delle risposte con X, con H0 q = 0.5.
  - Errore standard per disegno incrociato partecipanti x triplette (`crossed_se`): V = V_P + V_T - V_0
    (Owen 2007). V_P viene dal bootstrap sui soli partecipanti, V_T da quello sulle sole triplette dello strato
    (2.000 repliche ciascuno, seed 1234), V_0 = q(1 - q)/n e' la varianza binomiale.
  - p a due code e IC 95% su una t con gradi di liberta' di Satterthwaite.
  - Correzione di Holm sui 3 strati, alfa 0.05.
- **Perche' non i segni ribaltati.** Il test a segni ribaltati ricampiona solo i partecipanti: in simulazione,
  con sd fra triplette 0.8 logit, il suo alfa empirico arriva a 0.19. Il bootstrap "pigeonhole" (righe e colonne
  insieme) e' troppo conservativo (0.004-0.044). Il primario resta fra 0.043 e 0.075, mediana circa 0.056, su 48
  condizioni da 4.000 studi ciascuna (`power_v2.md`).
- **Secondarie, descrittive:**
  - IC sui soli partecipanti e p a segni ribaltati;
  - accordo complessivo di ogni GT sulle 260 triplette, con le differenze appaiate;
  - maggioranza per tripletta e kappa di Fleiss.
- **Numerosita'** (sez. 6): obiettivo 60 partecipanti tenuti. Una sola analisi confermativa, a 60 tenuti o alla
  chiusura. Prima di allora si contano solo partecipanti ed esclusi.

## 6. Potenza (`power_v2.py`, `power_v2.md`)

Simulazione con lo stesso test del primario:
- prove come nella sessione: 24 su 100 triplette, 18 su 80, 18 su 80;
- effetti logit per partecipante con sd 0.5 o 1.0 (nella v1: 0.21-0.38 oltre la binomiale);
- effetti per tripletta con sd 0 o 0.8;
- 1.000 studi per cella.

**Caso prudente:** sd 1.0 fra partecipanti e 0.8 fra triplette; alfa = 0.05/3 (Holm, caso peggiore). N tenuti:

| q | differenziale | F_vs_S 80% | F_vs_S 90% | F_vs_size 80% | S_vs_maxabs 80% |
|---:|---:|---:|---:|---:|---:|
| 0.575 | 0.15 | 200 | 250 | 200 | 250 |
| 0.60 | 0.20 | **60** | 100 | 80 | 80 |
| 0.65 | 0.30 | 25 | 30 | 30 | 30 |

Con sd 0.5 fra partecipanti, a q = 0.60, servono 40-50 partecipanti.

**Differenziale di pianificazione: 0.20 (q = 0.60).**
- Nella v1 l'accordo per risposta era 0.619 (LPIPS) contro 0.488 (maxabs), un differenziale di 0.13, su
  triplette con margini circa 3 volte piu' piccoli di questi.
- In `F_vs_S` il contrasto di taglia |d_size(A,B) - d_size(A,C)| va da 0.09 a 0.19 in log della centroid size
  (mediana 0.13), ben visibile.

**Obiettivo: 60 partecipanti tenuti.** Danno l'80% su `F_vs_S` nel caso prudente a q = 0.60.

**LIMITE dichiarato:**
- con 60 partecipanti i due strati secondari hanno il 76-77% nel caso prudente (ne servirebbero 80);
- un differenziale di 0.15 richiede 200-250 partecipanti, fuori dalla nostra portata: se l'effetto vero e' di
  quell'ordine, lo studio dara' un IC, non una decisione.

## 7. Limiti dichiarati ora

- **Un solo dominio, sintetico** (GNM). La taglia varia come nel modello (CV 5.3%).
- **Altezza in `S_vs_maxabs`.** Chi giudicasse con la sola altezza del volto preferirebbe maxabs nel 64% delle
  triplette dello strato. Con 40 simulati che seguono "solo altezza" la quota attesa e' circa 0.43 (osservata
  0.459, p 0.15). Un esito a favore di maxabs in questo strato va quindi letto come "conta l'altezza", che e'
  proprio cio' che maxabs normalizza.
- **Taglia giudicata per confronto.** La taglia si vede solo confrontando i tre volti, a parita' di camera:
  manca un riferimento nella scena.
- **Rigida robusta nei render.** I render usano la rigida robusta per identita', come la GT F.
- **Doppia partecipazione.** Il flag in `localStorage` non impedisce di partecipare da un altro browser o in
  navigazione privata.
