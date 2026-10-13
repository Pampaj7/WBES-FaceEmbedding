# Ava-256: protocollo della prova confermativa (CANDIDATO v2, 13 ottobre 2026; NON congelato)

> **CANDIDATO v2, NON CONGELATO.** E' la bozza dell'11 ottobre (`PROTOCOL_CONFERMATIVO_bozza.md`, sha256 `38ce3a5d...`,
> commit `53b03ac`) completata con le decisioni del PI del 12 ottobre (sez. 5-9, 11-13), piu' l'emendamento sull'uso
> dei 56 di calibrazione (sez. 10). Il testo delle sez. 1-4 e' quello della bozza, salvo le note della v2 nella sez. 2.
>
> **Modifiche della v2 rispetto al candidato v1** (sha256 `7df532d7...`, commit `0972896`), decisioni del PI del 13
> ottobre:
> 1. R7 confermativa: NoW metrical - NoW originale > 0, GT FR, `nocrop_cross`, una volta sola; famiglia di Holm da
>    m = 9 a **m = 10**, soglie ricalcolate (sez. 6, 8).
> 2. Landmark su Ava-256 per NoW, NoW metrical, REALY, Stirling: rilevatore insightface `1k3d68` su render frontale,
>    pipeline di `pr_noise.py` / `pr_noise_detect.py` (`f47a375`), parametri congelati (sez. 9.1).
> 3. REALY: keypoint oltre il 68 e regioni dall'oracolo FLAME (mappa di `datasets/AVA256/corr/`), con la condizione di
>    non applicabilita' (sez. 9.1); definizione dei landmark sulle original dei 56 (sez. 10).
> 4. NoW metrical contro d_F: descrittivo, delta e IC bilaterale sempre riportati; limite atteso (sez. 9.2).
> 5. Righe fallite dei metodi per coppia = +inf (rango massimo) nella lettura confermativa; maschera = descrittiva (sez. 6).
> 6. Nota su `split.json:526` (sez. 2); elenco dei descrittivi con la variante dichiarata (sez. 9); campi in attesa
>    aggiornati (sez. 14).
>
> Diventa vincolante solo quando:
> - tutti i campi **[in attesa]** e **[PI, in attesa ...]** sono fissati (elenco nella sez. 14);
> - il PI ne committa il testo con il suo sha256.
>
> Solo dopo si estendono gli strumenti ad Ava-256 e si lancia una valutazione.
>
> **Stato al 13 ottobre** (come nella bozza): su Ava-256 esistono solo dati, GT (200 valutati, 56 di calibrazione) e
> controlli delle viste. Nessun modello, embedding, baseline, rilevatore o metrica di prestazione e' stato calcolato su
> Ava-256. Per scrivere questo candidato si sono lette solo le impronte di `freeze_manifest.json`,
> `freeze_manifest_supp.json`, `calib_manifest.json`, la regola e i conteggi di `split.json` e i numeri aggregati di
> `corr_summary.json` (v2: copertura della mappa FLAME); nessun file di GT.
>
> **Revisione dell'11 ottobre** (dalla bozza). Il critic ha dato RISERVE sui dati; le correzioni del PI sono state
> applicate PRIMA di qualsiasi valutazione (README, sezione "Modifiche"):
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
    mai, con nessun metodo. **Uso esteso alla testa adattata di R6 e, per la sola geometria delle original, alla
    definizione dei landmark dei protocolli, dall'emendamento della sez. 10;**
  - **gli altri 200 sono i valutati**: 19.900 coppie di soggetti.
  - **Nota (v2).** Il campo `use` di `split.json:526` ("SOLO template del NICP, regione del fit B, L_d e cs_ref; mai
    valutati, da nessun metodo") va letto insieme all'emendamento della sez. 10: testa adattata di R6 e definizione dei
    landmark dei protocolli. `split.json` non si modifica (le sue impronte sono nella sez. 11).
- I 5 soggetti segnalati dai controlli di qualita' (4 fra i valutati) restano. Nessuna esclusione dopo i numeri; un
  metodo che fallisce su una mesh produce righe mancanti, contate e riportate (maschera comune, come `fact_paired.py`;
  v2: le righe fallite dei metodi per coppia valgono +inf nella lettura confermativa, sez. 6).

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
    Ava-256**; un solo scarto blocca la valutazione (verifiche complete nella sez. 11).
  - `up60k` non e' rigenerabile bit per bit: si valutano QUESTE mesh.
- Nessun operatore e nessun embedding esistono a oggi.

## 5. Gruppi e righe

Riga = coppia di soggetti valutati diversi (a < b) e coppia ordinata di topologie DIVERSE (t_a, t_b); GT della riga = GT
della coppia di soggetti.

| Gruppo | Ruolo | Topologie | Coppie ordinate | Righe (200 soggetti) |
|---|---|---|---|---|
| `nocrop_cross` (senza crop) | **primario** | original, remesh, noisy, down8k, up60k | 20 | 19.900 x 20 = 398.000 |
| `crop` | secondario | crop contro una delle altre 5 | 10 | 19.900 x 10 = 199.000 |
| `all_cross` | secondario | le 6, crop compreso | 30 | 19.900 x 30 = 597.000 |

- **Definizioni = quelle della matrice unificata** (`aau/full_matrix/fm_eval.py:473-477`, `groups_of`, commit
  `adc0df5`): `cross = (t_a != t_b)`, `crop = (t_a == "crop") | (t_b == "crop")`;
  - `nocrop_cross` = `cross & ~crop` (in fm_eval l'insieme si chiama `nocrop`);
  - `crop` = `cross & crop`. Le righe hanno topologie diverse, quindi "almeno una mesh crop" coincide con "esattamente
    una": crop - crop sta solo nella stessa topologia, che e' descrittiva;
  - `all_cross` = `cross`.
- **Descrittivi** (nessuna lettura confermativa), con le definizioni di fm_eval:
  - per singola perturbazione P (`con_perturbazione`: P su un lato, l'altro una topologia diversa senza crop; per P =
    crop coincide con `crop`);
  - stessa topologia (soggetti diversi, t_a = t_b, per P e tutte, crop - crop compresa: 19.900 x 6 = 119.400 righe;
    definizione della sez. 6 di `aau/runs/evidence/baselines_param/PROTOCOL_emendamento_3.md`).

## 6. Statistiche

- Per metodo e gruppo: **Spearman** fra la distanza del metodo e la GT sulle righe, pesato c_a c_b come
  `fm_eval._rep` (ranghi delle righe ripetute, Pearson).
- **Repliche** (bootstrap per soggetto, 1000 repliche): per ogni gruppo un generatore nuovo
  `np.random.default_rng(20261011)`; replica 0 = conteggi tutti 1 (il punto); repliche 1-1000 =
  `np.bincount(rng.integers(0, 200, 200), minlength=200)` sui 200 valutati ordinati come in fm_eval (`fm_eval.py:763-765`,
  con questo seme al posto di quello della T1).
  - Il seme riparte in ogni gruppo, quindi i conteggi sono gli stessi in tutti i gruppi, per tutti i metodi, per i due
    semi del modello e per tutti i delta.
- **IC 95%**: percentile sulle repliche 1-1000, riportato per ogni Spearman e ogni delta, senza correzione.
- **Delta appaiati** a - b sulle stesse righe e repliche. **p = P(delta <= 0)** = frazione delle 1000 repliche con
  delta <= 0 (`p_le0` di fm_eval).
  - Risoluzione 1/1000: il p minimo diverso da 0 e' 0,001; p = 0 si riporta come "< 0,001".
- **Confronti multipli: Holm**, per seme del modello, sulla famiglia {R1, R2a, R2b, R3a, R3b, R4, R5a, R5b, R6, R7}:
  **m = 10** ipotesi (v1: 9; v2 aggiunge R7), livello unilaterale 0,025.
  - Procedura: p ordinati p(1) <= ... <= p(m); si rifiuta H(i) finche' p(i) <= 0,025 / (m - i + 1); la prima che
    non passa ferma la procedura. A parita' di p l'ordine fra le ipotesi non cambia l'esito.
  - In interi, per evitare l'arrotondamento: con k(i) = numero di repliche su 1000 con delta <= 0, si rifiuta H(i) se
    **k(i) (m - i + 1) <= 25**. Soglie per m = 10:

    | i | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | 0,025 / (m - i + 1) | 0,0025 | 0,0028 | 0,0031 | 0,0036 | 0,0042 | 0,0050 | 0,0063 | 0,0083 | 0,0125 | 0,025 |
    | p massimo (k / 1000) | 0,002 | 0,002 | 0,003 | 0,003 | 0,004 | 0,005 | 0,006 | 0,008 | 0,012 | 0,025 |

  - La soglia piu' severa e' 0,025 / 10 = 0,0025: con la risoluzione di 1/1000 basta p <= 0,002 (al massimo 2 repliche
    su 1000 con delta <= 0), ancora sopra 0,001.
  - **R3a, R3b e R7 non dipendono dal seme** e si calcolano una volta sola: lo stesso p entra in entrambe le famiglie.
- **Esito per confronto:**
  - confermato: in **entrambi** i semi Holm rifiuta e il delta puntuale e' > 0;
  - **R3a, R3b, R7**: la regola "confermata in entrambi i semi" **non si applica** (nessun seme del modello entra nel
    confronto: un solo delta, un solo p). Sono confermate se Holm le rifiuta in **entrambe** le famiglie (le soglie
    possono differire, perche' gli altri p dipendono dal seme) e il delta puntuale e' > 0;
  - non risolto: secondo la regola della sez. 8, che prevale;
  - non confermato: tutti gli altri casi.
- GT FR per d_F e le distanze in mm; GT SR per d_P e le distanze a taglia normalizzata (cs). L'altra GT si riporta senza
  leggerla.
- **Righe fallite dei metodi per coppia (v2, decisione del PI).** Nella lettura confermativa una distanza non finita di
  un metodo per coppia della famiglia (ICP + Chamfer in mm e cs, NICP per coppia in mm e cs, NoW originale, NoW metrical)
  vale **+inf**: rango massimo, a pari merito fra le righe fallite (ranghi medi di `rankdata`). La riga resta.
  - E' la "sensibilita' NICP fallito = distanza massima" di fm_eval (`fm_eval.py:827-833`, valore
    nanmax x (1 + 1e-9)), che da' gli stessi ranghi, estesa a tutti i metodi per coppia della famiglia.
  - Le righe fallite contano comunque per la regola "non risolto" (sez. 8).
- **Maschera comune** (come fm_eval): le regole si leggono sulle righe del gruppo dove sono finite tutte le colonne
  **per mesh** della famiglia confermativa:
  - d_F cal., d_P e testa adattata dei due semi;
  - NICP su template in mm.

  Le colonne per coppia (ICP + Chamfer in mm e cs, NICP per coppia in mm e cs, NoW originale e metrical) non tolgono
  righe: +inf, sopra. I descrittivi stanno su questa maschera; se hanno buchi, in un passo a parte sulle loro righe
  finite, con i riferimenti sulle stesse righe (regola dei plug-in di fm_eval).
- **Versione mascherata (descrittiva)**: le stesse regole sulla maschera di v1, cioe' le righe dove sono finite TUTTE le
  colonne della famiglia (per mesh e per coppia). Si riporta accanto, non si legge.

## 7. Metodi e loro calibrazione (tutto fissato prima dei numeri; nessuna scelta su Ava-256)

1. **Modello: i due semi factorized gia' valutati nella matrice; nessun run nuovo.**

   | Seme | Checkpoint (sotto `aau/runs/evidence/trainer_v3/ablations/c3f_runs/`) | Passo | sha256 | c (`c_median`) |
   |---|---|---|---|---|
   | s1234 | `factorized/v3_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_poolmeanmax_noise60_sampler-balanced_pooling-area_meanmax_inputnorm-global_areaweights-smooth_emadecay-0.999_head-factorized_seed1234__1f5d4a90/checkpoints/epoch072_ema.pth` | 21.096 (epoca 72, pesi EMA) | `317532da3866e3be377bc57e26155c728db5760c61d1bfc23029b80f1a8ac84c` | 0.4052667604991996 |
   | s2345 | `factorized_s2345/v3_mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_poolmeanmax_noise60_sampler-balanced_pooling-area_meanmax_inputnorm-global_areaweights-smooth_emadecay-0.999_head-factorized_seed2345__4de9be29/checkpoints/epoch072_ema.pth` | 21.096 (epoca 72, pesi EMA) | `5489a97e93c9de25b184e69a27f68980ba61429a3ae0ff66030688d6acb690b0` | 0.40544514090039896 |

   - **Provenienza.** Chiavi `factorized_s1234` / `factorized_s2345` di `fact_calib.CKPTS`
     (`v3_work/trainer/tools/fact_calib.py:72-74`). Sono gli stessi file:
     - del campo `checkpoint` degli embedding letti dalla matrice (`fact_paired.embeddings`: `form_hifi`, `form_devfs`,
       `form_fv_expr`, passo 072);
     - della colonna `checkpoint` di `aau/runs/evidence/trainer_v3/factorized_calibration.csv:2-3`.

     Passo da `factorized_protocol_emendamento_4.md:16`. sha256 calcolato il 12 ottobre; i file hanno mtime 9 ottobre
     21:55-21:56, prima degli embedding della matrice (10 ottobre 06:28). I checkpoint sono fuori da git: lo strumento
     ne verifica lo sha256 prima dell'embedding (sez. 11).
   - Ogni regola che coinvolge il modello si valuta su **entrambi** i semi ed e' confermata solo se lo e' in entrambi
     (sez. 6).
   - **Distanza FR: d_F calibrata** = sqrt((S_i - S_j)^2 + S_i S_j (c d_P)^2), con il c congelato della tabella.
     - c e' la stima primaria dell'emendamento 4 (`factorized_protocol_emendamento_4.md:33`, mediana; colonna
       `c_median` del csv);
     - c viene SOLO dagli held-out sintetici bfm, ict, gnm (`factorized_protocol_emendamento_4.md:23-26`;
       `tools/fact_calib.py`). Nessun dato Ava-256 entra in c.
   - **SR: d_P.** **d_F non calibrata**: descrittiva (sez. 9).
   - Embedding con la pipeline delle viste dev della matrice: `eval_v3.py` + `zs_embed.py`,
     `WBES_V3_FACTORIZED_OUT=full`, scenario clean, seme 1234 (`factorized_protocol_emendamento_4.md:29-30`);
     `dp_per_unit` 0.24411710201003828 (`factorized_calibration.csv`).
   - Ingresso a scala globale: tabella `datasets/AVA256/scale/eval_view.npz` (`ava_scale_table.py`: stessa definizione
     di `build_scale_table.py`, area_mm2 = u^2 area; u e R di `datasets/AVA256/gt/frame.json`).
2. **Baseline ICP / NICP** (`aau/baselines_mm`, commit `6c8a747`), tutte nella famiglia confermativa:
   - ICP + Chamfer in mm e cs (cs = ogni mesh scalata alla cs_ref, `blmm.work_coords`);
   - NICP su template in mm;
   - NICP per coppia in mm e cs.

   Le composizioni con k dagli held-out sintetici (`factorized_protocol_emendamento_5.md`) sono descrittive.
   - Frame di lavoro in mm (`blmm.to_mm`): u = 1, R e t di `frame.json`.
   - **L_d** = mediana di max|V_mm - centro dei vertici| e **cs_ref** = mediana della centroid size robusta
     (`blmm.mesh_scalars`, aree passa-basso k = 64), sulle original dei **56 di calibrazione**. Sono le definizioni di
     `blmm_scalars.py`, che negli altri domini usava le original valutate.
   - **Template del NICP** = media delle original dei **56 di calibrazione, tutti**, nella normalizzazione del modo
     (`blmm.build_template`, 4096 vertici con `rng(0)`), al posto dei 100 non valutati estratti con `rng(1234)`.
3. **Fit B** dei modelli parametrici (GNM Head e FLAME 2023 Open): **descrittivo** (sez. 9). Variante B di
   `aau/runs/evidence/baselines_param/PROTOCOL_emendamento_1.md`, codice `aau/baselines_param/bp.py` (commit `26a6d5d`).
   Letture:
   - `<modello>_fr` con FR, `<modello>_sr` con SR;
   - la composizione forma B + taglia B (`PROTOCOL_emendamento_2.md`, k dagli held-out sintetici).

   Configurazione:
   - **Regione del modello** (`bp.region`): voto sulle original dei **56 di calibrazione**, con le soglie di `bp.py`
     (`REGION_DIST_MM` 10, `REGION_VOTE` 0,5).
   - **Iperparametri congelati dell'emendamento 1** (`aau/baselines_param/bp.py`):
     - ciclo: `LOOP` gnm {sigma 2,0, tau 10 mm, 5 iterazioni}, flame2023 {sigma 2,0, tau 5 mm, 5 iterazioni},
       `LOOP_INNER` 5, `LOOP_MIN_KEPT` 0,25;
     - regressione: `SIGMA_NOISE_MM` 1,0, `FIT_ITERS` 30, `FIT_TOL_MM` 1e-4, `JAW_BOUNDS` (-0,1; 0,6) rad,
       `FAIL_RMS_MM` 5,0.
   - **sigma = 2,0 confermato dal PI**: nessuna nuova taratura, nessuna sensibilita' a sigma 4/8 dell'emendamento 2.
4. **Testa adattata (R6)**: sez. 10.
5. **NoW originale e NoW metrical (R7, v2)**: i port di `aau/competitors/protocolli/` (`pr_metrics.py`, `pr_run.py`,
   commit `f47a375`; port invariati da `32c3839`) con i landmark del rilevatore (sez. 9.1), distanza = media dei due
   versi, ICP di NoW su tutti i punti del riferimento ritagliato (`--now-opt` 0, come nella matrice). Nessun parametro
   scelto su Ava-256.

**Esecuzione** (dopo il commit del protocollo congelato):
1. verifiche della sez. 11 (`verify_frozen()`, `verify_frozen_supp()`, `verify_calib()`, manifest, checkpoint e pesi
   del rilevatore);
2. estensione degli strumenti con la vista `ava256`: `eval_view` per i valutati, `calib_view` per template, regione,
   L_d e cs_ref sui 56, per gli ingressi della testa di R6 e per la definizione dei landmark dei protocolli (sez. 10);
3. embedding e distanze dei metodi sulle 1.200 mesh; embedding delle 280 mesh senza crop dei 56 per R6; render,
   rilevatore e landmark delle 1.200 mesh, poi le distanze dei protocolli sulle 597.000 coppie di `all_cross`;
4. fit della testa di R6 (un fit per seme, sez. 10);
5. righe, repliche, delta, Holm;
6. si riporta tutto, comprese le righe mancanti e fallite.

## 8. Regole di lettura

Delta = a - b (positivo = a davanti). Esiti: confermato, non confermato, non risolto (sez. 6). Statistica: delta appaiato
di Spearman con IC 95% e p = P(delta <= 0), Holm sulla famiglia.

| Regola | a | b | GT | Gruppo | Semi |
|---|---|---|---|---|---|
| **R1** (primaria) | d_F cal. | ICP + Chamfer in mm (rigido) | FR | `nocrop_cross` | s1234, s2345 |
| **R2a** | d_F cal. | NICP su template in mm | FR | `nocrop_cross` | s1234, s2345 |
| **R2b** | d_F cal. | NICP per coppia in mm | FR | `nocrop_cross` | s1234, s2345 |
| **R3a** (scoperta A su dati reali: togliere la taglia) | ICP + Chamfer in mm | ICP + Chamfer cs (ogni mesh alla cs_ref) | FR | `nocrop_cross` | nessuno (una volta) |
| **R3b** (scoperta A: deformazione non rigida per coppia) | ICP + Chamfer cs | NICP per coppia cs | FR | `nocrop_cross` | nessuno (una volta) |
| **R4** | d_F cal. | ICP + Chamfer in mm | FR | `crop` | s1234, s2345 |
| **R5a** | d_P | ICP + Chamfer cs | SR | `nocrop_cross` | s1234, s2345 |
| **R5b** | d_P | NICP per coppia cs | SR | `nocrop_cross` | s1234, s2345 |
| **R6** (adattamento few-shot, ex H2) | testa adattata sui 56 (sez. 10) | d_P dello stesso seme | SR | `nocrop_cross` | s1234, s2345 |
| **R7** (scala per coppia nei protocolli: NoW originale - NoW metrical < 0) | NoW metrical (Procrustes senza scala + ICP rigido), landmark del rilevatore | NoW originale (Procrustes + ICP con scala), stessi landmark | FR | `nocrop_cross` | nessuno (una volta) |

**Evidenze di sviluppo** (solo famiglie sintetiche e dev; FaMoS TEST non usato). Righe della matrice: `nocrop` di
`aau/runs/evidence/full_matrix/results.md` (commit `adc0df5`) per s1234 e per la scala, `deltas.csv` della stessa
cartella per s2345. Ordine: HIFI3D, FaceScape, FaceVerse con espressioni, FaceVerse neutra; "x / y" = s1234 / s2345.

- **R1**: +0.106 [+0.062, +0.154] / +0.088 [+0.040, +0.138]; +0.207 [+0.160, +0.257] / +0.216 [+0.166, +0.266]; -0.035 [-0.105, +0.036] / -0.019 [-0.091, +0.049]; -0.068 [-0.139, +0.011] / -0.038 [-0.105, +0.031] (results.md:469, 678, 880, 1075; deltas.csv:32, 17005, 32398, 40356). IC sopra 0 in 2 famiglie su 4.
- **R2a**: +0.134 [+0.085, +0.188] / +0.117 [+0.066, +0.171]; +0.119 [+0.030, +0.210] / +0.127 [+0.039, +0.213]; +0.091 [-0.010, +0.189] / +0.106 [+0.006, +0.201]; +0.117 [+0.013, +0.219] / +0.147 [+0.050, +0.241] (results.md:473, 682, 884, 1079; deltas.csv:36, 17009, 32402, 40360).
- **R2b**: +0.253 [+0.170, +0.342] / +0.235 [+0.149, +0.329]; +0.286 [+0.229, +0.345] / +0.295 [+0.227, +0.357]; +0.102 [+0.015, +0.190] / +0.117 [+0.037, +0.194]; +0.063 [-0.027, +0.152] / +0.093 [+0.003, +0.181] (results.md:471, 680, 882, 1077; deltas.csv:34, 17007, 32400, 40358). Il NICP per coppia fallisce su righe che dipendono dalla topologia: righe nocrop in maschera 98224 / 99000 su HIFI3D e 88724 / 99000 su FaceScape, cioe' il 10,4% perso; up60k 25,9% (results.md:309-311, 318, 334).
- **R3a**: +0.285 [+0.187, +0.383]; +0.038 [+0.018, +0.058]; +0.048 [-0.004, +0.102]; +0.035 [-0.018, +0.090] (results.md:150, 157, 164, 171); conteggio 2 / 0 / 2 su 4 (results.md:276).
- **R3b**: -0.017 [-0.046, +0.012]; +0.069 [+0.043, +0.095]; +0.078 [+0.022, +0.129]; +0.088 [+0.022, +0.148] (results.md:151, 158, 165, 172); conteggio 3 / 0 / 1 su 4 (results.md:277).
- **R4** (colonna crop): +0.245 [+0.199, +0.294] / +0.135 [+0.088, +0.184]; +0.062 [-0.002, +0.130] / -0.011 [-0.077, +0.054]; -0.019 [-0.083, +0.050] / -0.037 [-0.102, +0.028]; -0.034 [-0.098, +0.035] / -0.044 [-0.101, +0.015] (results.md:469, 678, 880, 1075; deltas.csv:296, 17245, 32646, 40586). IC sopra 0 in entrambi i semi solo su HIFI3D.
- **R5a**: +0.041 [-0.005, +0.091] / +0.033 [-0.027, +0.088]; +0.269 [+0.217, +0.317] / +0.275 [+0.222, +0.328]; -0.020 [-0.093, +0.055] / +0.007 [-0.065, +0.078]; -0.063 [-0.140, +0.017] / -0.025 [-0.096, +0.051] (results.md:187-188, 198-199, 209-210, 219-220); conteggio 1 / 0 / 3 per seme (results.md:292-293).
- **R5b**: +0.027 [-0.029, +0.083] / +0.018 [-0.046, +0.082]; +0.349 [+0.291, +0.401] / +0.355 [+0.290, +0.412]; +0.051 [-0.027, +0.124] / +0.078 [-0.005, +0.152]; +0.010 [-0.068, +0.095] / +0.049 [-0.041, +0.133] (results.md:191-192, 202-203, 213-214, 223-224); conteggio 1 / 0 / 3 per seme (results.md:296-297).
- **R6** (D5, EQ, d_P con SR, media dei 4 bersagli sintetici, k = 50): +0.090 [+0.069, +0.109] / +0.066 [+0.043, +0.091] (`aau/runs/evidence/diagnostics/results_d5.md:26, 36`). k = 50 e' il primo k della griglia con F in entrambi i semi (results_d5.md:171-174); per bersaglio non sempre: FaceVerse neutra s2345 +0.018 [-0.050, +0.088] (results_d5.md:34).
- **R7** (riportata come NoW originale - NoW metrical, cioe' -delta; landmark oracolo, matrice completa,
  `aau/runs/evidence/protocolli/results.md`, commit `c960859`): -0.281 [-0.371, -0.194]; -0.053 [-0.092, -0.018];
  -0.038 [-0.078, +0.000]; -0.021 [-0.060, +0.019] (results.md:93, 106, 119, 132). IC sotto 0 in 2 famiglie su 4, la
  neutra non risolta. Col rilevatore (sottocampione 8%, `rumore_landmark/results.md`, commit `f47a375`) gli Spearman
  restano quasi uguali: HIFI3D 0.413 contro 0.685, FaceScape 0.616 contro 0.678 (rumore_landmark/results.md:24-25,
  36-37; delta appaiato non calcolato). R7 non usa i semi del modello: sez. 6.

**Non risolto** (decisione del PI). Le regole che coinvolgono un metodo sono "non risolte" in un gruppo se, in quel
gruppo:
- il metodo ha piu' del 2% di mesh fallite (mesh delle topologie del gruppo: 1.000 per `nocrop_cross`, 1.200 per `crop`
  e `all_cross`);
- oppure piu' del 5% di righe mancanti (distanza non finita, contata prima della maschera comune).

Per i metodi per coppia (ICP, NICP per coppia, NoW originale e metrical) il fallimento e' della riga, non della mesh:
conta solo la soglia sulle righe, contate prima del +inf della sez. 6. Per i protocolli un ripiego del rilevatore (sez.
9.1) non e' un fallimento: la riga ha una distanza finita. Le regole non risolte si riportano comunque, con delta, IC e p.

Accanto a ogni regola, senza leggerli, si riportano gli stessi confronti sui domini di sviluppo, gia' noti (sopra).

## 9. Descrittivi (riportati tutti; nessuna lettura confermativa)

Ogni descrittivo si calcola con il codice al commit indicato. Il protocollo si congela solo quando non resta nessun
**[in attesa]**.

**Variante dichiarata.** Per i concorrenti nuovi e' quella con `declared: true` nel `fm_plugin.json` della cartella
`aau/runs/evidence/competitors_new/<metodo>/<variante>/`, scelta sui domini di sviluppo (piloti sui NON valutati) prima
di questo candidato. **Su Ava-256 si calcola SOLO la variante dichiarata**; le varianti con `declared: false` non si
calcolano. Per i metodi della matrice senza `fm_plugin.json` la variante e' unica ("unica" nella tabella).

| Descrittivo | Variante dichiarata (`declared`) | Letture | Codice (commit) | Stato |
|---|---|---|---|---|
| Fit B GNM Head e FLAME 2023 Open | variante B (`<modello>_vb_fr`, `_vb_sr`, `_vb_comp` di fm_eval); preregistrata e vA non si calcolano | FR (mesh FR), SR (mesh SR), composizione forma B + taglia B | `aau/baselines_param/bp.py` (`26a6d5d`), emendamenti 1-2; sez. 7.3, sigma = 2,0 | fissato |
| d_F non calibrata, dei due semi | unica | FR | quella del modello (sez. 7.1) | fissato |
| Composizioni delle baseline con k | unica | FR | `aau/baselines_mm` (`6c8a747`), `factorized_protocol_emendamento_5.md` | fissato |
| NICP su template cs | unica | SR | `aau/baselines_mm` (`6c8a747`), template dei 56 (sez. 7.2) | fissato |
| Versione mascherata delle regole (righe fallite tolte) | - | come le regole | `aau/full_matrix/fm_eval.py` (`adc0df5`), maschera comune | fissato |
| Varifold in mm (massa unitaria) | unica | FR, SR | `aau/baselines_param/bp_varifold.py` (`451b30c`), `aau/full_matrix/fm_varifold.py` (`c7e5299`) | fissato |
| ShapeDNA k = 50, HKS e WKS globali | unica (parametri di `competitors_hifi3d`) | FR, SR | `aau/full_matrix/fm_compute.py spectral` (`c7e5299`), `aau/competitors/comp_spectral.py` (`e6ea65e`) | fissato |
| Taglia stimata (`est_cs`) | unica | FR | `blmm.mesh_scalars` (`6c8a747`) | fissato |
| C3M e123 (d_F cal., d_F non cal., d_P), se disponibile | unica | FR, SR | checkpoint `aau/runs/evidence/trainer_v3/factorized/c3m/runs/v3_*_compactcache-true_head-factorized_seed1234__909688e6/checkpoints/epoch123_ema.pth`, sha256 `4ab9ca4531b06b0a199cb9c50f95d9756857663f4fa2078b6f97fee5412a2b77`, c 0.25732677076342675 (`factorized_calibration.csv:8`) | fissato |
| ArcFace-render, normal map 3 viste (G0) | unica (G0) | FR, SR | `aau/full_matrix/fm_arcface.py` (`c7e5299`) | fissato |
| Gromov-Wasserstein (POT) | `gw/euclid_128_cgw2` (`true`); `euclid_256_cg`, `euclid_256_cgw2` (`false`) | `fr` (mm), `sr` (a cs_ref) | `aau/competitors/gw_percettivi/` e `gw/pilot_choice.json` (`5c9c3e3`) | fissato |
| LPIPS alex, DINOv2 ViT-S/14, CLIP ViT-B/32, DreamSim su render | `lpips/depth`, `dinov2/depth`, `clip/smooth`, `dreamsim/smooth` (`true`); le altre 8 varianti di render (`false`) | `fr` (mm), `sr` (a cs_ref) | `aau/competitors/gw_percettivi/` e `percettivi/pilot_choice.json` (`5c9c3e3`) | fissato |
| Protocolli: NoW metrical (anche contro d_F, sez. 9.2), REALY, Stirling; NoW originale e metrical con GT SR | rilevatore (sez. 9.1); landmark oracolo e rumore sintetico (`lmk_s*`) non si calcolano | `mm` con FR e SR | `aau/competitors/protocolli/` (`32c3839`, `f47a375`) | fissato |
| ULRSSM | `ulrssm/faces_xyz` (`true` nel `fm_plugin.json`, NON ancora in git); `zeroshot_faust` (`false`) | per coppia (`pair_fr`, `pair_sr`) e via template (`tpl_fr`, `tpl_sr`) | `aau/competitors/ulrssm/` (`d727704`), codice esterno `0625cbcb` | **[in attesa]**: se variante dichiarata e pesi riaddestrati sono in git prima del congelamento entra, altrimenti **escluso** |
| Hybrid FM | nessuna | per coppia e per mesh | nessun codice nel repo | **[in attesa]**: se pronto (codice, variante dichiarata, commit) prima del congelamento entra, altrimenti **escluso** |
| Gruppi descrittivi (per perturbazione, stessa topologia) | - | come le regole | `aau/full_matrix/fm_eval.py` (`adc0df5`) | fissato |
| Testa adattata di R6 con d_F (FR) | unica | FR | sez. 10 | fissato |

Non si calcolano su Ava-256 (come nella matrice, dove esistono solo su alcuni domini): ctrlfr, C3M e205, ArcFace
ombreggiato (E12), OpenShape, Uni3D, coefficienti dei fit.

### 9.1 Landmark dei protocolli su Ava-256 (v2, decisione del PI)

Su dati reali non esistono landmark esatti. NoW originale, NoW metrical, REALY e Stirling usano i landmark del
**rilevatore**: la condizione `det` di `aau/competitors/protocolli/pr_noise.py` e `pr_noise_detect.py` (commit
`f47a375`), la stessa pipeline e gli stessi parametri, senza nessuna scelta su Ava-256.
- **Render** (`pr_noise.render_mesh`): una immagine per mesh, per tutte le 1.200 mesh di `eval_view` (6 topologie,
  crop compreso), mesh in mm nel frame di lavoro (`blmm.to_mm`: u = 1, R e t di `frame.json`, sez. 7.2).
  - Camera prospettica sull'asse +z, centrata sul riquadro xy della mesh, a `CAM_DIST` = 4,0 volte l'estensione.
  - **Frazione del volto nel render `FILL` = 0,45** (estensione xy / lato dell'immagine), immagine `IMG` = 640 x 640.
    Scelta su 8 render di sviluppo **sul solo punteggio del detector**, senza landmark ne' GT: a 0,75 il detector non
    trovava volti (punteggio 0,07-0,31), a 0,45 circa 0,88 (`pr_noise.py:80-81`). Non si ritara su Ava-256.
  - Raycasting di open3d, normali per vertice interpolate, Lambert con luce `LIGHT` (0; 0,4; 1) + ambiente `AMB` 0,25,
    colore uniforme `ALBEDO` (0,88; 0,71; 0,61), sfondo nero.
- **Rilevatore** (`pr_noise_detect.py`): insightface 0.7.3 (venv di MICA, onnxruntime su CPU), pacchetto `buffalo_l`.
  - Detector `det_10g` (SCRFD) a 640 x 640; fra i volti trovati **il riquadro di area massima** (il suo punteggio
    `det_score` si registra); `1k3d68` sul suo riquadro, come `FaceAnalysis.get`.
  - Nessun volto: `1k3d68` sul riquadro dei pixel colpiti dal render (ripiego, contato e riportato per topologia).
  - Pesi (sha256): `det_10g.onnx` `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`,
    `1k3d68.onnx` `df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc`
    (`~/.insightface/models/buffalo_l/`; sez. 11).
- **Retroproiezione** (`pr_noise._prep_task`, ramo `det`): ogni punto 2D col raggio della camera sul primo punto
  colpito della mesh; raggio che non colpisce: il pixel colpito piu' vicino (contati e riportati).
- **Uso**: i 7 landmark di NoW e Stirling = iBUG 36, 39, 42, 45, 33, 48, 54 del rilevatore; da questi i ritagli di NoW
  (cerchio 1,4) e Stirling (1,2) e il riferimento di REALY (`pr_run.load` con `prep`, `pr_landmarks.realy_reference`).
- **REALY**: dei suoi 85 keypoint, i primi 68 (stessi indici iBUG) vengono dal rilevatore. I 17 oltre il 68 (REALY ne
  usa 4: 70, 71, 76, 77) e le etichette di regione per vertice restano dall'**oracolo FLAME**, come nella condizione
  `det` di sviluppo (rumore_landmark/results.md, "Limiti dichiarati": REALY resta ottimistico), e solo dove esiste la
  corrispondenza:
  - la mappa regione FLAME -> template di Ava-256 esiste (`datasets/AVA256/corr/ava256.npz`, `ava_corr.py` con
    `correspond.py`): copertura della regione FLAME 1671/1671, residuo sui landmark tenuti fuori 0,65 mm
    (`corr_summary.json`);
  - i 17 keypoint oltre il 68 cadono su triangoli FLAME dentro la regione (trasferimento baricentrico; la condizione
    dipende solo dalla regione FLAME: vero in HIFI3D, FaceScape, FaceVerse, `landmarks/<dominio>.json`,
    `transfer_realy85_nonbary` solo sotto il 68);
  - etichetta di regione solo entro `COVER_MM` = 3 mm dalla regione FLAME deformata; oltre, nessuna regione;
  - definizione per Ava-256 con `pr_landmarks.py define --spec` (`32c3839`): template `datasets/AVA256/template.npz`,
    mappa `datasets/AVA256/corr/ava256.npz`, aggancio sulla media delle original dei **56 di calibrazione** (sola
    geometria, sez. 10). Frame della spec = u, R, t di `frame.json` (sez. 7.2), non l'identita'.
  - **Non applicabilita'**: se la mappa manca o se `define` fallisce il controllo del frame (mediana media -> template
    > `FRAME_MAX_MM` = 2 mm), la corrispondenza su Ava-256 non esiste e **REALY e' non applicabile**: si riporta come
    tale, con il motivo, senza ripieghi. Una regione vuota in un singolo riferimento resta come su sviluppo (errore NaN
    della regione, `pr_metrics.realy_pair`). NoW e Stirling non dipendono dall'oracolo: usano solo i 7 landmark del
    rilevatore.
- **Coppie**: tutte le 597.000 coppie non ordinate di `all_cross` (nessun sottocampione: `FRAC` di `pr_noise.py` non si
  usa), i due versi, distanza = media.
- **Controlli prima di Ava-256**: lo strumento esteso importa le funzioni di `pr_noise.py` / `pr_noise_detect.py`
  invariate (render, camera, raggi, retroproiezione) e le costanti di sopra; sui landmark `det` di HIFI3D gia' calcolati
  (job 1071153) riproduce i 2D del rilevatore su 24 mesh. Scarti riportati.
- **Limite dichiarato**: i landmark del rilevatore hanno su sviluppo un errore medio di 2,90 mm (FaceScape) e 3,54 mm
  (HIFI3D) sui 7 di NoW, 5,7-6,5 mm su `noisy` (rumore_landmark/results.md:13, 18, 203-204). NoW originale e metrical
  ne dipendono poco (<= 0,031 di Spearman anche a sigma 5), REALY e Stirling molto (col rilevatore REALY -0,06 / -0,20,
  Stirling -0,05 / -0,12; rumore_landmark/results.md:214-216).

### 9.2 NoW metrical contro la nostra d_F (v2, decisione del PI): solo descrittivo

- Delta = NoW metrical - d_F (positivo = NoW metrical davanti), GT FR, per d_F cal. e d_F non cal., per seme, nei
  gruppi `nocrop_cross`, `crop` e `all_cross`, sulle righe e repliche della sez. 6.
- Si riportano **sempre** il delta puntuale e l'**IC 95% bilaterale** (percentile, 2,5-97,5), **qualunque sia il segno**.
  Nessun test, nessuna famiglia, nessuna lettura confermativa.
- **Limite atteso** (sviluppo, landmark del rilevatore, d_F non cal. s1234, sottocampione 8%,
  rumore_landmark/results.md:120, 124, 132):
  - senza crop **pari**: HIFI3D +0.041 [-0.011, +0.094] (con landmark esatti sulla matrice completa +0.055 [+0.009,
    +0.105], protocolli/results.md:100);
  - col crop **NoW metrical sopra**: HIFI3D +0.133 [+0.069, +0.203], FaceScape +0.170 [+0.083, +0.260].

  Su Ava-256 ci si aspetta quindi che NoW metrical stia alla pari o sopra la nostra d_F con GT FR, soprattutto col crop.
  Si scrive qui, prima dei numeri, perche' il risultato non si legga come una sorpresa ne' si ometta.

## 10. Emendamento sull'uso dei 56 di calibrazione (12 ottobre 2026, PRIMA di qualunque valutazione)

**Cosa limitava l'uso.** La bozza limitava i 56 a template del NICP, regione del fit B, L_d e cs_ref
(`PROTOCOL_CONFERMATIVO_bozza.md:42-43, 124-128, 132-133`; vincoli 167-169). Lo stesso limite sta in `split.json:526`
(campo `use`: "SOLO template del NICP, regione del fit B, L_d e cs_ref; mai valutati, da nessun metodo").

**Cosa cambia.** L'uso si estende alla **sola R6**: la testa metrica (i-e2) si adatta con **k = 56** soggetti
etichettati di Ava-256, cioe' tutti i 56 di calibrazione.
- Restano invariati: template, regione, L_d e cs_ref (sez. 7).
- Nessun altro metodo usa la GT dei 56.
- I 56 non entrano in nessuna riga di valutazione.
- `split.json` non si modifica: il suo campo `use` (`split.json:526`) va letto con questo emendamento.

**Aggiunta della v2: landmark dei protocolli.** Le original dei 56 (`calib_view`, **sola geometria, nessuna GT**)
servono anche come superficie di aggancio di `pr_landmarks.py define --spec` (sez. 9.1): indici dei 68 iBUG e degli 85
keypoint di REALY ed etichette di regione sulla topologia original comune di Ava-256. E' l'analogo dei "100 soggetti NON
valutati" dei domini di sviluppo (`blmm.template_subjects`) e dello stesso tipo d'uso del template del NICP. Sui
valutati si usano solo i 17 keypoint di REALY oltre il 68 e le etichette di regione (sez. 9.1); i 7 landmark di NoW e
Stirling vengono dal rilevatore.

**Motivazione.** D5 (sviluppo, bersagli sintetici) da' il primo k con il criterio in entrambi i semi a k = 50
(results_d5.md:171-174, sez. 8, R6).

**c resta tarata solo sul sintetico** (sez. 7.1). La GT dei 56 non entra in c.

**GT dei 56.** Congelata in `datasets/AVA256/calib_gt/` (commit `4083d66`; impronte dei byte dei file in `29464be`).
Impronte del contenuto in `calib_manifest.json`:
- FR `70faa95e243b0941fd7f8664a66713d2c8f7000790df298e8dbd4f543feb6372`;
- SR `af7ed8769090cf08edfaa9bfa4681e155e0c6b3be43579972e587daab6c82d4e`;
- centroid size `76ac2f567d66fffeeec5f1ac1a724741bc4bc0fd7be037ebdefc95af60a3cfbe`.

Calcolata nella stessa esecuzione sui 256. Ricalcolata dai soli 56 differisce di 3,2e-8 relativo (README, "GT di
calibrazione", indipendenza dalla batch): la dipendenza dai 200 valutati e' solo il numero di iterazioni comuni
dell'IRLS.

**Procedura di R6: quella di D5 congelata**. Fonti:
- `aau/runs/evidence/diagnostics/PROTOCOL_D5.md`, sha256 `3c2f7e7f...`, commit `484d638`;
- codice `aau/diagnostics/d5_fewshot.py` (codice `5f9913b`, ultima modifica `34af962`), `d3_e2.fit_e2`, `d3_head.fit`.

I punti fissati:
- **Sorgenti**: in D5 "tutte tranne T" (PROTOCOL_D5.md:34-36). Ava-256 non e' fra le sorgenti, quindi sono tutte e 8
  (`d3_stats.SOURCES`: bfm, ict, gnm, flame2023_s1, facescape, hifi3d, faceverse, famos). E' l'impostazione `joint` di
  D3 e2 (`d3_e2.py:52-54`, FLAME 0-99), con gli stessi blocchi e gli stessi embedding.
- **Blocco Ava-256**, in coda all'ordine di `d3_stats.SOURCES`:
  - i 56 di calibrazione x 5 etichette senza crop (`calib_view`), embedding del seme con x = u x dp_per_unit;
  - GT SR di `calib_gt/ava256_calib_sr.npz`;
  - coppie di soggetti diversi ed etichette diverse: 1.540 x 20 = 30.800.
- **Configurazione**: per entrambi i semi quella scelta dalla CV di D3 e2 per `joint`, r = 64, lambda = 1e-6
  (`aau/runs/evidence/diagnostics/d3_e2_heads.csv:21, 31`). E' la regola di D5: r e lambda della CV di D3 e2 per
  l'insieme di sorgenti della testa (PROTOCOL_D5.md:45-51).
  - **Fonte e motivo di lambda**: 1e-6 e' il valore preregistrato di D5. lambda = 1e-5 e' escluso perche' scelto
    guardando i bersagli (results_d5.md:208-210).
  - Limite dichiarato: 1e-6 sta sul bordo della griglia (`edge` = True; results_d5.md:193-196).
- **Perdita e fit**:
  - perdita riscalata per blocco (`d3_e2.rescale`: GT del blocco divisa per il suo alpha0, anche per il blocco Ava);
  - `d3_head.fit` (L-BFGS-B, maxiter 2000, W iniziale dalla base principale);
  - **pesi EQ** (la primaria di D5): ogni blocco pesa uguale, 1/9 con il blocco Ava. La variante W di D5 non si
    calcola;
  - nessuna nuova CV.
- **Una sola testa per seme**: k = 56 e' il pool intero, quindi non ci sono estrazioni (come il pool intero in D5,
  PROTOCOL_D5.md:59-61).
- **Nodo**: il fit gira su i256-a10-10, il nodo di D3 e2 e D5. Con lambda = 1e-6 BLAS diverse cambiano i fit
  (PROTOCOL_D5.md:126-128). Si registrano nodo, job, esito (`success`) e iterazioni; nessun rifit.
- **Lettura**: delta = rho(d_h, SR) - rho(d_P, SR), stesso seme, `nocrop_cross` dei 200 valutati, stesse repliche
  (sez. 6).
- **d_F della testa** (FR): solo descrittiva. **Scostamento da D5 dichiarato**: D5 calcola il c della testa anche sul
  blocco del bersaglio (PROTOCOL_D5.md:52-53, `d3_e2._refit`). Qui `d3_head.calib_c` usa le sole 8 sorgenti, senza il
  blocco Ava, perche' la GT dei 56 non deve entrare in nessun c.
- **Controllo K0-Ava** (prima del blocco Ava, nessun dato Ava):
  - la testa con le sole 8 sorgenti, stessa configurazione e stesso nodo, deve riprodurre c e alpha/alpha0 di
    `d3_e2_heads.csv:21, 31` entro 1e-6 relativo;
  - se non li riproduce, si riporta lo scarto e si prosegue (come K0 di D5).

## 11. Verifiche obbligatorie nello strumento di valutazione

Si fanno prima di leggere qualunque file di Ava-256, nell'ordine sotto. Un solo scarto solleva RuntimeError e blocca la
valutazione. Codice: `aau/ava256/ava_freeze.py`, `aau/ava256/ava_calib.py` (commit `29464be`).

1. **Manifest in git**: sha256 dei byte dei file.

   | File | sha256 | Commit |
   |---|---|---|
   | `aau/ava256/freeze_manifest.json` | `e485fb7c44f0638ef3111343ba8741a3febae80be35f8fcdae14ae0839d5f365` | `53b03ac` |
   | `aau/ava256/freeze_manifest_supp.json` | `c6d2157a145fa29e7a8415dc60aa1bd98e9edfdf2bc38563356cc15f16f3b02e` | `29464be` |
   | `aau/ava256/calib_manifest.json` | `d3657a186525dd32c7e152356047204a2bbfad86fd65a098d9895db91b1a3af7` | `29464be` (prima versione `4083d66`) |
   | `aau/ava256/split.json` | `08234346a85c84188bcba3ed54debd38b79836e77fd968e591175e5b5457b7ec` | `53b03ac` |

2. **`ava_freeze.verify_frozen()`**: contenuto di viste, GT e scala contro `freeze_manifest.json`. Impronte attese:
   - viste: `eval_view` (1.200 mesh) e `calib_view` (336), una per mesh;
   - GT: `ava256_eval_fr` `9ccc80ba339c79a3aa6133a7f2c0d03dce4a9466dce90e8cff11f67a7954ea47`,
     `ava256_eval_sr` `33da2b51ea16af303cbd5e0f3c95c420fcc6629bc4b78049f2719380492392b8`,
     `ava256_all_fr` `58a3ba2c6a4217906a852bf3cf10450aa854f3557d59558f7d24ff2d013bba6e`,
     `ava256_all_sr` `5f5cb152fda3f13c9c00be3b6bf8dc3646a3cae488fd7498e4347b6a699c85cd`;
   - scala: `eval_view` `4cf3b9d30746ec2d849c67b5ff8e3c6466fec07316a5d7fb2460aad18873ee5b`,
     `calib_view` `efa76bc2068dc38adbd8d9eb0db456c254176e5dff3cd7a355fc91c3bd16338e`.
3. **`ava_freeze.verify_frozen_supp()`**: byte dei 41 file di `freeze_manifest_supp.json`:
   - tutto `gt/` (json con l'unita', `ids.json`, `split.json`, `frame.json`);
   - i file in cima a `eval_view/` e `calib_view/`;
   - `scale/`.
4. **`ava_calib.verify_calib()`**: contenuto e byte di `calib_gt/` contro `calib_manifest.json` (7 file). Impronte
   attese:
   - FR `70faa95e243b0941fd7f8664a66713d2c8f7000790df298e8dbd4f543feb6372`;
   - SR `af7ed8769090cf08edfaa9bfa4681e155e0c6b3be43579972e587daab6c82d4e`;
   - centroid size `76ac2f567d66fffeeec5f1ac1a724741bc4bc0fd7be037ebdefc95af60a3cfbe`.
5. **Checkpoint**: sha256 dei due `epoch072_ema.pth` (sez. 7.1) e, se si calcola, di C3M e123 (sez. 9).
6. **Rilevatore** (v2): sha256 di `det_10g.onnx` e `1k3d68.onnx` di `buffalo_l` (sez. 9.1) prima del primo render.

## 12. Vincoli

- Una sola valutazione per metodo. Nessuna scelta fatta su Ava-256: iperparametri, checkpoint, c, k, soggetti,
  regione, viste, split. Tutto si riporta, comprese righe mancanti e fallimenti.
- I 56 di calibrazione non entrano in nessuna riga di valutazione. La loro GT entra solo nella testa di R6, la geometria
  delle loro original anche nella definizione dei landmark dei protocolli (sez. 10).
- Nessun parametro del render o del rilevatore si ritara su Ava-256 (sez. 9.1).
- Gli strumenti esistenti si estendono ad Ava-256 solo dopo il commit del protocollo congelato. Fino ad allora la vista
  porta `CONFERMATIVO_NON_VALUTARE.txt` e i dati sono in sola lettura.
- Figure: solo forme medie o aggregate, salvo conferma scritta di Meta per render di singole persone (README,
  Licenza).

## 13. Cosa NON e' confermativo

- Tutto cio' che non e' R1-R7:
  - i descrittivi della sez. 9, compreso NoW metrical contro la d_F (sez. 9.2);
  - i gruppi descrittivi;
  - l'altra GT di ogni distanza (anche NoW originale e metrical con GT SR);
  - la d_F della testa di R6;
  - la versione mascherata delle regole (righe fallite tolte invece che +inf, sez. 6);
  - i confronti sui domini di sviluppo riportati accanto.
- Una regola non confermata in un seme non si "recupera" con l'altro seme, con un altro gruppo o con un descrittivo.
- **FaMoS TEST resta descrittivo nel paper.** Non motiva nessuna scelta di questo protocollo e non entra in nessuna
  regola.

## 14. Campi in attesa e condizione di congelamento

Il protocollo si congela solo quando sono fissati:
1. ULRSSM: variante dichiarata (`faces_xyz`, `fm_plugin.json` con `declared: true`) e pesi riaddestrati in git
   [in attesa; se non pronti al congelamento, ULRSSM e' escluso];
2. Hybrid FM: codice, variante dichiarata e commit [in attesa; se non pronto al congelamento, e' escluso].

Fissati nella v2 (non piu' in attesa): R7 e la famiglia di Holm (m = 10); Gromov-Wasserstein e percettivi (varianti
dichiarate in `5c9c3e3`); protocolli NoW, NoW metrical, REALY, Stirling (codice in `32c3839`, risultati di sviluppo in
`c960859`, pipeline del rilevatore in `f47a375`).

Poi il PI committa questo testo con il suo sha256 e la parola "congelato" nel titolo.

**CANDIDATO v2, NON CONGELATO: NESSUNA VALUTAZIONE SU AVA-256 FINO AL CONGELAMENTO.**
