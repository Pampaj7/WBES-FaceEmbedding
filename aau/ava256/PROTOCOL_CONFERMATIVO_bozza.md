# Ava-256: protocollo della prova confermativa (BOZZA, 11 ottobre 2026)

> **BOZZA: da completare dal PI prima di qualsiasi valutazione.** Le voci marcate **[PI]** sono segnaposto.
> Il protocollo diventa vincolante quando il PI lo completa e ne committa il testo con il suo sha256; solo dopo si
> estendono gli strumenti di valutazione ad Ava-256 e si lancia una qualunque valutazione.
>
> **Stato al momento della bozza:** su Ava-256 esistono solo dati e GT. Nessun modello, baseline o metrica e' stato
> calcolato, nemmeno per prova; Ava-256 non e' in nessuno strumento di valutazione. Le sole statistiche calcolate
> riguardano la GT stessa (unita', taglia, affidabilita'): `README.md`, `gt_summary.json`.

## 1. Dominio

- **Ava-256** (Meta, Codec Avatar Studio, NeurIPS D&B 2024): 256 persone catturate in una dome a 172 camere,
  Pittsburgh, 2021-2023; 124 uomini, 124 donne, 6 non binari, 2 non dichiarati; adulti, sbilanciato verso pelle chiara
  e meno di 35 anni (DATASHEET). Licenza CC BY-NC 4.0 (README, sezione Licenza).
- **Mai calcolato dal progetto** prima di questa bozza: e' il test vergine "in senso forte" di
  `literature/TEST_SET_VERGINE_2026-10-11.md` (sez. 3).
- **Mesh:** registrazioni del tracking cinematico interno di Meta (non scansioni), topologia a 7306 vertici che
  **estende quella di Multiface** (stessi indici per i 5471 vertici di Multiface). Multiface (13 persone, stesso
  laboratorio) e' stato usato in sviluppo e fra gli 8 domini che definiscono la regione unificata: Ava-256 ha persone
  nuove, ma non e' un dominio di acquisizione indipendente da Multiface.
- **Superficie visibile:** il tracking segue barba e capelli (ava0163 ha una barba folta: distanza da mu 8,0 mm contro
  3,9 di mediana, concentrata sulla mandibola). La GT misura quella superficie, la stessa che ricevono i metodi.

## 2. Frame, neutra, soggetti

- Neutra di ogni persona: frame di `EXP_neutral_peak` (7-54 per persona, mediana 15) con la regola di FaMoS
  (`ava_neutral.py`). Una sola sessione per persona.
- **Soggetti: tutti i 256** (256 validi su 256; i 13 segnalati dalla regola MAD restano, `gt_summary.json`). Nessuna
  esclusione dopo i numeri; un metodo che fallisce su una mesh produce righe mancanti, contate e riportate (maschera
  comune, come `fact_paired.py`).
- **[PI]** Confermare: tutti i 256 per tutti i metodi, oppure un sottoinsieme dichiarato ora per i metodi costosi
  (vedi sez. 7, costi).

## 3. Regione e GT (congelate)

- Regione: i 1478 punti della regione unificata, portati su Ava-256 da `correspond.py` (copertura 1478/1478;
  landmark: residuo 0,49 mm, tenuti fuori 0,65 mm; Chamfer 0,208 mm).
- **GT primaria: FR** (mm, rigida robusta per identita' verso mu, `train_fr_sr.py`); **secondaria: SR** (pre-forme a
  centroid size 1). Unita' verificate: millimetri (IPD mediana 66,0 mm; centroid size 57,9 mm, CV 4,8% [4,4; 5,2]).
- File e impronte del contenuto (sha256 di `D_orig` float32 e dei nomi, `ava_gt.content_sha256`):
  - FR `datasets/AVA256/gt/ava256_fr.npz`: `58a3ba2c6a4217906a852bf3cf10450aa854f3557d59558f7d24ff2d013bba6e`
  - SR `datasets/AVA256/gt/ava256_sr.npz`: `5f5cb152fda3f13c9c00be3b6bf8dc3646a3cae488fd7498e4347b6a699c85cd`
  - righe e colonne: `datasets/AVA256/gt/ids.json` (id950000 + riga di `256_ids.csv`).
- **Affidabilita' della GT** (solo GT contro GT, riferimento per la lettura, non un criterio):
  - neutro ripetuto (`EXP_eye_neutral`, altro segmento della stessa sessione): Spearman FR 0,972, SR 0,966 sulle
    32.640 coppie. E' un limite inferiore dell'affidabilita': quel segmento contiene anche variazioni d'espressione
    (sopracciglia, mandibola);
  - distanza FR intra-persona mediana 0,76 mm contro 6,68 mm fra persone;
  - mappa di Multiface al posto della nuova: Spearman FR 0,996.

## 4. Viste (congelate)

- `datasets/AVA256/eval_view/npz/id<950000 + k>_GTready_<topologia>.npz`, 256 x 6 = 1536 mesh: `original` (patch di
  3127 vertici / 6021 triangoli), `remesh`, `crop`, `noisy`, `down8k`, `up60k`, prodotte dal codice di HIFI3D,
  FaceVerse e FaceScape dev (`make_ict_topologies.process_subject`). mm, frame nativo dei dati (nessuna rigida della GT
  negli ingressi). Frame di dominio per gli strumenti in mm: `datasets/AVA256/gt/frame.json`.
- Impronta del contenuto (`views_summary.json`, `content_sha256.txt`):
  `8150ba42b702b892aa27b1b4eb3da1cc061aaaa5ad11723fa345935199ec9cd6`. Rigenerandole, `up60k` cambia tassellazione
  (decimazione): si valutano QUESTE mesh.
- Nessun operatore e nessun embedding esistono a oggi.

## 5. Gruppi e righe

Riga = coppia di soggetti diversi (a < b) e coppia ordinata di topologie DIVERSE (t_a, t_b); GT della riga = GT della
coppia di soggetti.

| Gruppo | Topologie | Coppie ordinate | Righe (256 soggetti) |
|---|---|---|---|
| `nocrop_cross` (senza crop) | original, remesh, noisy, down8k, up60k | 20 | 32.640 x 20 = 652.800 |
| `all_cross` | le 6, crop compreso | 30 | 32.640 x 30 = 979.200 |

- **[PI]** Gruppo primario. Proposta: `nocrop_cross`, come HIFI3D e dev FaceScape in E12 e `fact_paired.py`;
  `all_cross` secondario, perche' il crop e' un'affermazione a parte.
- Descrittivo (non decisionale): Spearman dentro ogni coppia ordinata di topologie senza crop e dentro la stessa
  topologia (definizione della sez. 6 di `aau/runs/evidence/baselines_param/PROTOCOL_emendamento_3.md`).

## 6. Statistiche

- Per metodo e gruppo: **Spearman** fra la distanza del metodo e la GT sulle righe (coppie).
- **IC 95%:** bootstrap per soggetto, 1000 repliche: ogni replica estrae 256 soggetti con reinserimento (conteggi c),
  peso di una riga c_a c_b (come E8 sez. 4 ed E12), IC percentile. Stesse repliche per tutti i metodi e i delta.
- **Seme: [PI]** da dichiarare qui prima dei numeri (proposta: `default_rng(20261011)`, uno solo per gruppo).
- **Delta appaiati** (metodo - riferimento) sulle stesse righe e repliche: IC 95% percentile e P(delta <= 0).
- GT FR per le distanze "form", GT SR per quelle "shape" (d_P, distanze a taglia normalizzata); l'altra GT si riporta
  senza leggerla.

## 7. Confronti previsti

1. **Modello finale** (checkpoint del run principale: **[PI]** percorso, passo, sha256), distanza primaria
   preregistrata: **d_F calibrata** = sqrt((S_i - S_j)^2 + S_i S_j (c d_P)^2), con c dagli held-out sintetici
   (`factorized_protocol_emendamento_4.md` sez. 1, `tools/fact_calib.py`; **[PI]** valore di c congelato prima dei
   numeri). Con SR: d_P. Nessun dato Ava-256 entra nella taratura.
2. **Baseline ICP / NICP** (`aau/baselines_mm`): ICP + Chamfer in mm, NICP su template in mm (con FR); NICP per coppia
   cs (con SR); composizioni con k dagli held-out sintetici (`factorized_protocol_emendamento_5.md`).
3. **Fit B** dei modelli parametrici (`aau/runs/evidence/baselines_param/PROTOCOL_emendamento_1.md`, variante B, GNM
   Head e FLAME 2023 Open): `<modello>_fr` con FR, `<modello>_sr` con SR, e la composizione forma B + taglia B
   (`PROTOCOL_emendamento_2.md`).

**Da decidere prima dei numeri [PI]:**
- NICP su template e regione del fit B usano negli altri domini la media di 100 soggetti NON valutati del pool. Su
  Ava-256 tutti i 256 sono valutati: serve una fonte dichiarata del template (per esempio un'altra vista, la media
  FLAME, o una divisione dei 256 in template e valutazione decisa ora, col seme).
- Costi: il NICP per coppia su 979.200 righe e' circa 6,6 volte il carico di HIFI3D a 100 soggetti (148.500 righe con
  il crop). Se serve un sottoinsieme, va dichiarato qui (regola e seme).
- Frame di lavoro in mm per `baselines_mm`: `frame.json` (u = 1, R e t come `frames.json` di E12).

## 8. Regole di lettura (segnaposto) [PI]

- **R1 (primaria):** [PI] es. "il modello finale con d_F calibrata e' almeno pari a X su `nocrop_cross` con FR se
  l'estremo inferiore dell'IC 95% del delta contro X supera [soglia]".
- **R2:** [PI] confronto con il fit B e con NICP su template (FR).
- **R3:** [PI] SR: d_P del modello contro NICP per coppia cs e contro `<modello>_sr` del fit B.
- **R4:** [PI] `all_cross` (crop): lettura separata, secondaria.
- Riferimenti per la lettura (non criteri): affidabilita' della GT (sez. 3) e i risultati degli stessi confronti sui
  domini di sviluppo, gia' noti.
- Esito "non risolto": [PI] (righe mancanti, fallimenti di un metodo oltre una soglia).

## 9. Vincoli

- Una sola valutazione per metodo, nessuna scelta (iperparametri, checkpoint, c, k, soggetti, regione, viste) fatta su
  Ava-256. Tutto si riporta, comprese le righe mancanti e i fallimenti.
- Gli strumenti esistenti si estendono ad Ava-256 solo dopo il commit del protocollo completato; prima di allora la
  vista porta `CONFERMATIVO_NON_VALUTARE.txt`.
- Figure: solo forme medie o aggregate, salvo conferma scritta di Meta per render di singole persone (README,
  Licenza).

**DA COMPLETARE DAL PI PRIMA DI QUALSIASI VALUTAZIONE.**
