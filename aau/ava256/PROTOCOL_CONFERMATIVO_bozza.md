# Ava-256: protocollo della prova confermativa (BOZZA, 11 ottobre 2026; rivista lo stesso giorno dopo il critic)

> **BOZZA: da completare dal PI prima di qualsiasi valutazione.** Le voci **[PI]** sono segnaposto.
> Il protocollo diventa vincolante quando il PI lo completa e ne committa il testo con il suo sha256. Solo dopo:
> - si estendono gli strumenti di valutazione ad Ava-256;
> - si lancia una valutazione.
>
> **Stato al momento della bozza:** su Ava-256 esistono solo dati e GT. Nessun modello, embedding, baseline o metrica di
> prestazione e' stato calcolato, nemmeno per prova, e Ava-256 non e' in nessuno strumento di valutazione. Le sole
> statistiche calcolate riguardano la GT (unita', taglia, affidabilita') e la geometria delle viste: `README.md`,
> `gt_summary.json`, `view_qc.json`.
>
> **Revisione dell'11 ottobre.** Il critic ha dato RISERVE sui dati; le correzioni del PI sono state applicate PRIMA di
> qualsiasi valutazione (README, sezione "Modifiche"):
> - viste suddivise 1-a-4;
> - patch con 4 anelli e impronta NoW;
> - split 56 + 200;
> - u = 1 dichiarata;
> - stabilita' su tutti i frame;
> - congelamento con impronte.

## 1. Dominio

- **Ava-256** (Meta, Codec Avatar Studio, NeurIPS D&B 2024): 256 persone catturate in una dome a 172 camere,
  Pittsburgh, 2021-2023; 124 uomini, 124 donne, 6 non binari, 2 non dichiarati; adulti, sbilanciato verso pelle chiara
  e meno di 35 anni (DATASHEET). Licenza CC BY-NC 4.0 (README, sezione Licenza).
- **Mai calcolato dal progetto** prima di questa bozza: e' il test vergine "in senso forte" di
  `literature/TEST_SET_VERGINE_2026-10-11.md` (sez. 3).
- **Mesh:** registrazioni del tracking cinematico interno di Meta (non scansioni), topologia a 7306 vertici che
  **estende quella di Multiface** (stessi indici per i 5471 vertici di Multiface). Multiface (13 persone, stesso
  laboratorio) e' stato usato in sviluppo e fra gli 8 domini che definiscono la regione unificata. Le persone di Ava-256
  sono nuove, ma il dominio di acquisizione non e' indipendente da Multiface.
- **Superficie visibile:** il tracking segue barba e capelli (un soggetto con barba folta sta a 8,0 mm da mu contro 3,9
  di mediana). La GT misura quella superficie, la stessa che ricevono i metodi.

## 2. Soggetti e split (congelati)

- Neutra di ogni persona: i frame di `EXP_neutral_peak` (7-54 per persona, mediana 15), con la regola di FaMoS
  (`ava_neutral.py`). Una sola sessione per persona. 256 validi su 256.
- **Split fisso** (`ava_common.split`, `split.json`): i 256 sid in ordine di
  sha256(`ava256-confermativo-calibrazione-2026-10-11:` + sid); la regola dipende solo dagli id.
  - **i primi 56 sono di calibrazione**: template del NICP, regione del fit B, L_d e cs_ref (sez. 7). Non si valutano
    mai, con nessun metodo;
  - **gli altri 200 sono i valutati**: 19.900 coppie di soggetti.
- I 5 soggetti segnalati dai controlli di qualita' (4 fra i valutati) restano. Nessuna esclusione dopo i numeri; un
  metodo che fallisce su una mesh produce righe mancanti, contate e riportate (maschera comune, come `fact_paired.py`).

## 3. Regione e GT (congelate)

- Regione: i 1478 punti della regione unificata, portati su Ava-256 da `correspond.py` (copertura 1478/1478;
  landmark: residuo 0,49 mm, tenuti fuori 0,65 mm; Chamfer 0,208 mm).
- **GT primaria: FR** (mm, rigida robusta per identita' verso mu, `train_fr_sr.py`); **secondaria: SR** (pre-forme a
  centroid size 1). Unita': mm, dichiarata (README, modifica 5).
- GT della prova: `datasets/AVA256/gt/ava256_eval_{fr,sr}.npz`, sui 200 valutati, righe in `ids.json`.
  Impronte del contenuto (sha256 di `D_orig` float32 e dei nomi):
  - FR: `9ccc80ba339c79a3aa6133a7f2c0d03dce4a9466dce90e8cff11f67a7954ea47`
  - SR: `33da2b51ea16af303cbd5e0f3c95c420fcc6629bc4b78049f2719380492392b8`
- **Rumore della GT** (solo GT contro GT, 256 soggetti; riferimento descrittivo, non un criterio):
  - neutra contro neutro ripetuto di `EXP_eye_neutral`: Spearman FR 0,972, SR 0,966. E' una stima del rumore fra
    segmenti diversi della stessa sessione, gonfiata dall'espressione presente nel segmento ripetuto;
  - **non e' un tetto per gli Spearman dei metodi**: viste e GT vengono dalla stessa neutra, quindi il rumore fra
    segmenti non entra fra vista e GT;
  - la mappa di Multiface al posto della nuova da' Spearman FR 0,996.

## 4. Viste (congelate)

- `datasets/AVA256/eval_view/npz/id<950000 + k>_GTready_<topologia>.npz`: **200 valutati x 6 = 1.200 mesh**.
  - Topologie: `original`, `remesh`, `crop`, `noisy`, `down8k`, `up60k`.
  - La `original` e' la neutra nella patch NoW, con bordo esterno, bocca e aperture palpebrali, suddivisa 1-a-4.
  - Codice: lo stesso di HIFI3D, FaceVerse e FaceScape dev.
  - mm, frame nativo dei dati: nessuna rigida della GT negli ingressi.
- `datasets/AVA256/calib_view/npz/`: i 56 di calibrazione x 6 = 336 mesh, senza GT.
- Equivalenza con le viste dev (`view_qc.json`): lato medio della original 1,86 mm, spostamento normale del remesh
  0,134 mm, lato di down8k 3,46 mm, tutti dentro gli intervalli di HIFI3D, FaceVerse e FaceScape dev. 4 anelli di bordo
  in ogni topologia.
- Copertura della GT dentro la vista: 1434/1478 punti su triangoli della vista, 1472/1478 coi tre vertici. Il residuo
  (0,35% del peso d'area) sta sui margini palpebrali e sulla rima.
- **Impronte del contenuto** in `aau/ava256/freeze_manifest.json`: mesh, GT e tabella di scala.
  - **Lo strumento di valutazione chiama `aau/ava256/ava_freeze.verify_frozen()` prima di leggere qualsiasi file di
    Ava-256**; un solo scarto blocca la valutazione.
  - `up60k` non e' rigenerabile bit per bit: si valutano QUESTE mesh.
- Nessun operatore e nessun embedding esistono a oggi.

## 5. Gruppi e righe

Riga = coppia di soggetti valutati diversi (a < b) e coppia ordinata di topologie DIVERSE (t_a, t_b); GT della riga = GT
della coppia di soggetti.

| Gruppo | Topologie | Coppie ordinate | Righe (200 soggetti) |
|---|---|---|---|
| `nocrop_cross` (senza crop) | original, remesh, noisy, down8k, up60k | 20 | 19.900 x 20 = 398.000 |
| `all_cross` | le 6, crop compreso | 30 | 19.900 x 30 = 597.000 |

- **[PI]** Gruppo primario. Proposta: `nocrop_cross`, come HIFI3D e FaceScape dev in E12 e `fact_paired.py`;
  `all_cross` secondario, perche' il crop e' un'affermazione a parte.
- Descrittivo, non decisionale: Spearman dentro ogni coppia ordinata di topologie senza crop e dentro la stessa
  topologia (definizione della sez. 6 di `aau/runs/evidence/baselines_param/PROTOCOL_emendamento_3.md`).

## 6. Statistiche

- Per metodo e gruppo: **Spearman** fra la distanza del metodo e la GT sulle righe.
- **IC 95%:** bootstrap per soggetto, 1000 repliche: ogni replica estrae i 200 soggetti valutati con reinserimento
  (conteggi c), peso di una riga c_a c_b (come E8 sez. 4 ed E12), IC percentile. Stesse repliche per tutti i metodi e i
  delta.
- **Seme: [PI]** da dichiarare qui prima dei numeri. Proposta: `default_rng(20261011)`, uno per gruppo.
- **Delta appaiati** (metodo - riferimento) sulle stesse righe e repliche: IC 95% percentile e P(delta <= 0).
- GT FR per le distanze "form", GT SR per quelle "shape" (d_P, distanze a taglia normalizzata); l'altra GT si riporta
  senza leggerla.
- **Confronti multipli: [PI]** da fissare prima dei numeri. Per esempio: gerarchia dichiarata (R1, poi R2-R4 solo se R1
  e' confermata), oppure Holm sulla famiglia dei delta primari con livello [PI]; gli IC si riportano comunque al 95%.

## 7. Metodi e loro calibrazione (tutto fissato prima dei numeri; nessuna scelta su Ava-256)

1. **Modello finale** (checkpoint del run principale: **[PI]** percorso, passo, sha256). Distanza primaria
   preregistrata: **d_F calibrata** = sqrt((S_i - S_j)^2 + S_i S_j (c d_P)^2). c viene dagli held-out sintetici
   (`factorized_protocol_emendamento_4.md` sez. 1, `tools/fact_calib.py`, `factorized_calibration.csv`; **[PI]** valore
   congelato). Con SR: d_P.
   - Ingresso a scala globale: tabella `datasets/AVA256/scale/eval_view.npz` (`ava_scale_table.py`: stessa definizione
     di `build_scale_table.py`, area_mm2 = u^2 area; u e R di `datasets/AVA256/gt/frame.json`).
   - Nessun dato Ava-256 entra in c.
2. **Baseline ICP / NICP** (`aau/baselines_mm`): ICP + Chamfer in mm e NICP su template in mm (con FR); NICP per coppia cs
   (con SR); composizioni con k dagli held-out sintetici (`factorized_protocol_emendamento_5.md`).
   - Frame di lavoro in mm (`blmm.to_mm`): u = 1, R e t di `frame.json`.
   - **L_d** = mediana di max|V_mm - centro dei vertici| e **cs_ref** = mediana della centroid size robusta
     (`blmm.mesh_scalars`, aree passa-basso k = 64), sulle original dei **56 di calibrazione**. Sono le definizioni di
     `blmm_scalars.py`, che negli altri domini usava le original valutate.
   - **Template del NICP** = media delle original dei **56 di calibrazione, tutti**, nella normalizzazione del modo
     (`blmm.build_template`, 4096 vertici con `rng(0)`), al posto dei 100 non valutati estratti con `rng(1234)`.
3. **Fit B** dei modelli parametrici (variante B di `aau/runs/evidence/baselines_param/PROTOCOL_emendamento_1.md`; GNM
   Head e FLAME 2023 Open): `<modello>_fr` con FR, `<modello>_sr` con SR, e la composizione forma B + taglia B
   (`PROTOCOL_emendamento_2.md`, k dagli held-out sintetici).
   - **Regione del modello** (`bp.region`): voto sulle original dei **56 di calibrazione**, con le soglie di `bp.py`
     (`REGION_DIST_MM` 10, `REGION_VOTE` 0,5).
   - **Iperparametri congelati dell'emendamento 1** (`aau/baselines_param/bp.py`):
     - ciclo: `LOOP` gnm {sigma 2,0, tau 10 mm, 5 iterazioni}, flame2023 {sigma 2,0, tau 5 mm, 5 iterazioni},
       `LOOP_INNER` 5, `LOOP_MIN_KEPT` 0,25;
     - regressione: `SIGMA_NOISE_MM` 1,0, `FIT_ITERS` 30, `FIT_TOL_MM` 1e-4, `JAW_BOUNDS` (-0,1; 0,6) rad,
       `FAIL_RMS_MM` 5,0.
   - Nessuna nuova taratura, ne' la sensibilita' a sigma 4/8 dell'emendamento 2: **[PI]** confermare.

**Esecuzione** (dopo il commit del protocollo completato):
1. `verify_frozen()`;
2. estensione degli strumenti con la vista `ava256`: `eval_view` per i valutati, `calib_view` per template e regione,
   L_d e cs_ref sui 56;
3. embedding e distanze dei metodi sulle 1.200 mesh;
4. righe, repliche, delta;
5. si riporta tutto, comprese le righe mancanti.

## 8. Regole di lettura (struttura fissata; soglie da completare dal PI)

Per ogni regola: confronto, GT, gruppo, statistica, soglia, esito in {confermato, non confermato, non risolto}.

| Regola | Confronto (delta = modello - riferimento) | GT | Gruppo | Statistica | Soglia |
|---|---|---|---|---|---|
| **R1** (primaria) | modello finale d_F calibrata contro [PI] (proposta: NICP su template in mm) | FR | primario | IC 95% del delta di Spearman | [PI] |
| **R2** | modello finale d_F calibrata contro fit B (GNM, FLAME 2023 Open) e contro forma B + taglia B | FR | primario | idem | [PI] |
| **R3** | modello finale d_P contro NICP per coppia cs e contro `<modello>_sr` del fit B | SR | primario | idem | [PI] |
| **R4** | come R1-R3 | FR / SR | `all_cross` | idem | [PI] |

- Esito "non risolto": [PI] (per esempio righe mancanti oltre una soglia, o un metodo che fallisce su piu' di [PI]
  mesh).
- Confronti multipli: sez. 6, [PI].
- Si riportano accanto, senza leggerli, i risultati degli stessi confronti sui domini di sviluppo, gia' noti.

## 9. Vincoli

- Una sola valutazione per metodo. Nessuna scelta fatta su Ava-256: iperparametri, checkpoint, c, k, soggetti,
  regione, viste, split. Tutto si riporta, comprese righe mancanti e fallimenti.
- I 56 di calibrazione non entrano in nessuna riga di valutazione.
- Gli strumenti esistenti si estendono ad Ava-256 solo dopo il commit del protocollo completato. Fino ad allora la vista
  porta `CONFERMATIVO_NON_VALUTARE.txt` e i dati sono in sola lettura.
- Figure: solo forme medie o aggregate, salvo conferma scritta di Meta per render di singole persone (README,
  Licenza).

**DA COMPLETARE DAL PI PRIMA DI QUALSIASI VALUTAZIONE.**
