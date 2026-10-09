# Studio umano v2: taglia, forma e normalizzazione nel giudizio umano di somiglianza

Arbitro percettivo di `paper/PLAN_MASSIVE.md` §19 fra le GT di E12. Il protocollo, scritto prima dei dati, sta in
`PROTOCOL.md` (revisione 4, hash in `PROTOCOL.sha256`), con le letture dichiarate per ogni strato. Lo studio v1
(`aau/human_study/`, `docs/human_study/`) non e' toccato.

## Disegno in breve

| | |
|---|---|
| volti | GNM Head (Apache-2.0), 100 identita' campionate, CV della taglia 5.3% |
| geometria | mm nel frame di GT-F di E12 + rigida robusta per identita' (come la GT F); mai una scala per mesh |
| immagini | striscia frontale / 3/4 / profilo, 512 x 1544 px per TUTTI i volti, 2.327 px/mm, camera unica |
| prova | candidato \| riferimento \| candidato, righe allineate per vista; stessa larghezza CSS per i tre volti |
| sessione | 2 di esercizio + 18 test (12 `F_vs_S`, 3 `F_vs_size`, 3 `S_vs_maxabs`) + 2 controlli, circa 4 minuti |
| analisi | quota con la prima GT per strato, SE incrociato partecipanti x triplette, t di Satterthwaite |
| test | a cascata `F_vs_S` → `F_vs_size` → `S_vs_maxabs`, ciascuno alla soglia tarata 0.04 |
| obiettivo | 60 partecipanti tenuti |

## Strati e letture

| strato | cosa contrappone | lettura di q > 0.5 |
|---|---|---|
| `F_vs_S` (principale) | con taglia (F, EDM, "solo taglia") contro senza (S, unified, maxabs) | la taglia conta |
| `F_vs_size` | forma (F, S, maxabs, unified) contro "solo taglia", a contrasto di taglia del 2-5% | a taglia quasi uguale conta la forma |
| `S_vs_maxabs` | shape di Procrustes contro maxabs, a taglia neutra; F e "solo taglia" bilanciati al 50% | la shape e' piu' vicina al giudizio umano |

F si attribuisce solo dallo schema congiunto (`F_vs_S` e `F_vs_size` entrambi > 0.5). Il self-test lo verifica:
partecipanti simulati che seguono S, F o la sola taglia producono esattamente le firme attese. Chi segue F o la
sola taglia non produce effetto in `S_vs_maxabs`.

## La pagina

- **Aspetto:** un solo carattere (Inter), palette neutra, tema chiaro e scuro automatico.
- **Comandi:** barra di avanzamento, pulsanti grandi Left / Right, tasti ←/→ e 1/2, clic o tocco sul volto.
- **Telefono:** tre colonne strette, ognuna con le tre viste impilate.
- **Scala identica:** le tre strisce hanno la stessa larghezza CSS (`--col`) e le stesse dimensioni intrinseche.
  La pagina verifica le dimensioni di ogni immagine (`meta.image_px`) e non parte se una differisce.
- **Testo neutro:** "Which face is more similar to the reference face?" e "Ignore light and shadows"; mai "shape".
- **Consenso:** dichiara cosa si registra, la nota in `localStorage` (avanzamento e "gia' partecipato"), il
  passaggio da Google (font e form) e cita GNM Head.
- **Fine:** pagina di ringraziamento con il codice partecipante. Dopo un invio riuscito lo stesso browser mostra
  "already taken part".

## Esecuzione

Tutto su CPU, nel venv della GT unificata. Il nodo `cpu` ha avuto errori di prolog: in quel caso si usa
`-p prioritized --gres=NONE`.

```bash
sbatch aau/human_study_v2/run.sbatch                     # gt, render, select, selftest (~8 min)
HS2_STEPS=power sbatch aau/human_study_v2/run.sbatch      # taratura dell'alfa e potenza (~30 min con 16 CPU)
aau/human_study_v2/analyze_v2.py --form-csv risposte.csv # dopo la raccolta
```

| file | ruolo |
|---|---|
| `hs2.py` | dominio GNM, campionamento, frame di GT-F, rigida robusta, triangoli da renderizzare |
| `gt_v2.py` | GT in `datasets/HUMAN_STUDY_V2/gt/` |
| `render_v2.py` | render, `camera.json`, `render_check.json` |
| `select_triplets_v2.py` | `STRATA` (pool, quote, vincoli di taglia, bilanciamento), `triplets.json`, `docs/human_study_v2/` |
| `analyze_v2.py` | analisi: filtro v2 + `triplets_hash`, esclusione, SE incrociato, cascata; `--self-test` |
| `power_v2.py` | taratura dell'alfa e potenza: `power_v2.md`, `power_v2.json` |

## Inferenza e potenza

**Perche' non i segni ribaltati.** Il test a segni ribaltati per partecipante ignora che anche le triplette sono
un campione: in simulazione il suo alfa arriva a 0.19. Il primario usa quindi V = V_P + V_T - V_0, cioe' i
bootstrap sulle sole righe e sulle sole colonne viste meno la varianza binomiale (Owen 2007). Il p si legge su una
t di Satterthwaite.

**Taratura prima dei dati.** Sotto H0, in 84 condizioni:
- alla soglia nominale 0.05 l'alfa empirico arriva a 0.061;
- alla soglia **0.04** resta ≤ 0.050 ovunque, ed e' la soglia fissata.

**Potenza nel caso prudente** (sd 1.0 fra partecipanti, 0.8 fra triplette), N tenuti:

| differenziale | F_vs_S 80% | F_vs_S 90% | secondari 80% |
|---:|---:|---:|---:|
| 0.15 | 100 | 150 | 300 |
| 0.20 | **60** | 70 | 120 |
| 0.30 | 25 | 30 | 50 |

**L'80% vale per differenziali ≥ 0.20 su `F_vs_S`; sotto, solo intervallo di confidenza.** Con 60 partecipanti i
secondari, con 3 prove a testa, hanno il 56-57% a 0.20 e l'80% solo da 0.30 in su: servono come stima. Il
differenziale di pianificazione e' 0.20: la v1 dava 0.13 con margini circa 3 volte piu' piccoli, e in `F_vs_S` il
contrasto di taglia e' di circa 50 px nell'immagine.

## Limiti

Dettagli in `PROTOCOL.md` §7:
- un solo dominio sintetico;
- "solo altezza" in `S_vs_maxabs` favorisce maxabs (64% dello strato);
- il contrasto di `F_vs_size` e' forse invisibile;
- l'esclusione con 2 controlli (nessun errore ammesso) e' piu' grossolana della v1;
- il flag "gia' partecipato" vale per un solo browser.
