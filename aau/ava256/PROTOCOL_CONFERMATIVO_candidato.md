# Ava-256: protocollo della prova confermativa (CANDIDATO v4, 13 ottobre 2026; NON congelato)

> **CANDIDATO v4, NON CONGELATO.** E' la bozza dell'11 ottobre (`PROTOCOL_CONFERMATIVO_bozza.md`, sha256 `38ce3a5d...`,
> commit `53b03ac`) completata con le decisioni del PI del 12 ottobre (sez. 5-9, 11-13), piu' l'emendamento sull'uso
> dei 56 di calibrazione (sez. 10). Il testo delle sez. 1-4 e' quello della bozza, salvo le note della v2 nella sez. 2.
>
> **Modifiche della v4 rispetto al candidato v3** (sha256 `955c13b9491af3a129e7d6a9b03e0bbf6aca06f089437eba37a7848ab7175451`,
> commit `123cb4e`): chiusi i due campi **[in attesa]** di codice della sez. 14 (punti 3 e 4) con lo standard della v3:
> codice scritto prima del congelamento con un unico percorso `--spec`, collaudo su HIFI3D con le tolleranze scritte prima
> di eseguire (sez. 11.2), sha256 di ogni file (sez. 11.1). Su Ava-256 non si e' eseguito niente.
> 1. **Embedding** (sez. 7.1, 10, 11): `aau/ava256/conf_embed.py` (passi `embed-stage`, `embed` di `conf.sbatch`): i due
>    semi (e C3M e123, descrittivo) sulle 1.200 mesh di `eval_view`, i due semi sulle 280 senza crop di `calib_view`;
>    operatori `areanorm` k_eig 128, `eval_v3.py` + `zs_embed.py` con gli argomenti di `d3_embed.sbatch`, tabelle di
>    scala di `datasets/AVA256/scale/`, R di `frame.json`, GPU L40S; nessuna GT nell'embedding (nomi dei soggetti da un
>    file di soli id). Collaudo: le 600 mesh di `form_hifi` e il pool NON valutato di D3 rifatti: **fuori tolleranza**
>    (Z <= 1,2e-3, Spearman <= 4,3e-4 contro 1e-4: L40S contro gli A100 di sviluppo, TF32 attivo nel container; registrato,
>    tolleranza non allargata, sez. 11.2).
> 2. **Descrittivi fuori da `conf_eval.py`** (sez. 5, 9): `aau/ava256/conf_desc.py` (passi `desc-*`): calcolo di
>    spettrali, varifold, ICP + Chamfer di stessa topologia, ArcFace, GW e percettivi (SOLO le varianti dichiarate) e
>    lettura di tutti i descrittivi (fit B letto con la composizione, varifold, spettrali, taglia stimata, C3M e123,
>    ArcFace, GW, percettivi) e dei gruppi descrittivi (per perturbazione, stessa topologia), lettura confermativa (+inf)
>    e mascherata. Collaudo: colonne ricalcolate contro quelle di sviluppo (GW e percettivi su un sottocampione
>    dichiarato), lettura contro `full_matrix` in tutti i gruppi (sez. 11.2).
> 3. ULRSSM e Hybrid FM: invariati (sez. 9, 14: entrano se pronti prima del congelamento, altrimenti esclusi).
>
> **Modifiche della v3 rispetto al candidato v2** (sha256 `99bfd256517d038f3e6cefdfcfbef518359db1ba82e48765fa185a0d75c2e0f0`,
> commit `014fc70`): correzioni dei sei punti del BLOCCANTE del critic, decise dal PI il 13 ottobre.
> 1. **Pipeline di Ava-256 scritta, collaudata e hashata PRIMA del congelamento**: un unico percorso `--spec`
>    (`aau/ava256/conf_*.py`, `conf.sbatch`, `blmm_*.py` e `bp_*.py` con `--spec`) per i protocolli col rilevatore, il
>    valutatore confermativo, il blocco di R6 e le baseline. Collaudo su HIFI3D dallo stesso percorso, con le tolleranze
>    scritte prima di eseguire (sez. 11.2). Commit e sha256 di ogni script (sez. 11.1). Regola sui bug scoperti dopo il
>    congelamento (sez. 12).
> 2. **Deroga esplicita per i protocolli**: su Ava-256 si calcola SOLO la variante `lmk_det` (rilevatore, sez. 9.1), anche
>    se il suo plug-in ha `declared: false` (sez. 9, 9.1).
> 3. **Holm: m = 10 fisso**. Le regole non risolte e la R6 con fit non convergente restano nella famiglia col loro p
>    (sez. 6).
> 4. **+inf**: accanto a ogni evidenza di sviluppo la colonna "fallito = distanza massima" (FaceScape R3b da +0,069 a
>    +0,155), dichiarata come la statistica letta su Ava-256, con l'effetto atteso (sez. 6, 8).
> 5. **Sez. 11**: sha256 di `datasets/AVA256/template.npz` e `corr/ava256.npz`, versioni di insightface e onnxruntime
>    accanto ai pesi.
> 6. **Rilevatore**: il controllo sui 24 render di HIFI3D ha tolleranza 0 ed e' bloccante; soglia sui ripieghi (2% delle
>    mesh del gruppo) oltre la quale R7 e i descrittivi dei protocolli sono "non risolti", fissata sui tassi di sviluppo
>    (HIFI3D e FaceScape: 1 mesh su 600) (sez. 8, 9.1).
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
> Solo dopo si lancia una valutazione su Ava-256. Gli strumenti sono gia' estesi (v3 e v4, sez. 7, 9 e 11): il codice
> della catena, embedding e descrittivi compresi (v4), e' scritto, collaudato su HIFI3D e hashato; non e' mai stato
> eseguito su Ava-256.
>
> **Stato al 13 ottobre** (come nella bozza): su Ava-256 esistono solo dati, GT (200 valutati, 56 di calibrazione) e
> controlli delle viste. Nessun modello, embedding, baseline, rilevatore o metrica di prestazione e' stato calcolato su
> Ava-256. Per scrivere questo candidato si sono lette solo le impronte di `freeze_manifest.json`,
> `freeze_manifest_supp.json`, `calib_manifest.json`, la regola e i conteggi di `split.json` e i numeri aggregati di
> `corr_summary.json` (v2: copertura della mappa FLAME); nessun file di GT. v3: in piu' `frame.json` (u, R, t nello spec),
> gli id di `split.json` (nello spec), l'elenco delle chiavi di `template.npz` e `corr/ava256.npz` (non i valori) e gli
> sha256 dei byte di questi due file; nessuna mesh, nessuna GT. v4: nessun file di Ava-256 in piu'; lo spec ha in piu' solo
> percorsi (insiemi dell'embedding, tabelle di scala, cartelle dei descrittivi), il checkpoint di C3M e123 e le varianti
> dichiarate dei descrittivi.
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
  - **m = 10 fisso (v3).** Ogni famiglia ha sempre le 10 ipotesi. Una regola "non risolta" (sez. 8) e la R6 con un fit
    della testa non convergente (`success` = False, sez. 10) **restano nella famiglia col loro p** e occupano il loro
    posto nell'ordinamento: non si tolgono, non si sostituiscono, non cambiano m. Il loro esito resta quello della sez. 8
    (non risolta) o della regola sotto; il fit non convergente si riporta accanto.
  - Codice: `conf_eval.holm` (sez. 11.1), in interi: si rifiuta H(i) finche' 40 k(i) (m - i + 1) <= n, n = repliche
    finite (1000: k(i) (m - i + 1) <= 25).
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
  - **E' la statistica letta su Ava-256 (v3).** Su sviluppo la colonna corrispondente e' il passo "NICP fallito =
    distanza massima" di `full_matrix/deltas.csv`, citato accanto a ogni evidenza della sez. 8 (la maschera resta
    riportata, descrittiva). Sui domini di sviluppo i metodi per coppia della famiglia che falliscono sono solo i due NICP
    per coppia (`full_matrix/controls.json`: `rows_dropped_by_column_cross` = solo `mm_nicp_p2tri`, `cs_nicp_p2tri`), quindi
    il passo coincide con la lettura +inf (verificato nel collaudo, sez. 11.2).
  - **Effetto atteso, asimmetrico.** Il +inf penalizza solo il metodo che fallisce. Il NICP per coppia fallisce su righe
    scelte dalla topologia (up60k: 25,9% di righe perse su FaceScape), non dalla GT: messe in coda, abbassano il suo
    Spearman. Le regole con b = NICP per coppia diventano piu' favorevoli ad a: su FaceScape R3b da +0,069 a **+0,155**,
    R2b da +0,286 / +0,295 a +0,362 / +0,370, R5b da +0,349 / +0,355 a +0,431 / +0,438; su HIFI3D (970 righe perse su
    148.500) gli spostamenti sono di 0,01 al piu'. Le altre regole cambiano solo per le righe rientrate (R1 su FaceScape
    da +0,207 a +0,192). Se un metodo per coppia fallisce molto su Ava-256, la regola e' anche "non risolta" (sez. 8):
    il +inf non decide da solo.
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
   - **Codice (v4): `aau/ava256/conf_embed.py`**, insiemi `embed.sets` dello spec:
     - `eval_view`: i 200 valutati x 6 topologie (1.200 mesh), i due semi e C3M e123 (descrittivo, sez. 9);
     - `calib_view`: i 56 di calibrazione x le 5 etichette senza crop (`diag.LABELS`; 280 mesh), i due semi (blocco di
       R6, sez. 10).
     - `stage`: un link per mesh (il file della vista, invariato) e un file di soli nomi (`names.npz`: gli id dei
       soggetti e una matrice di zeri). `zs_embed.py` legge dal `--dist_npz` solo i nomi dei soggetti da tenere: la GT di
       Ava-256 non entra nell'embedding (`d3_embed.sbatch` passava la gt_matrix della vista).
     - Operatori `areanorm` k_eig 128 (`v2_work/potential/areanorm_operators.py`, come `aau/diagnostics/d3.sbatch`) in
       `datasets/V3_OPS_CACHE/conf/ava256/<insieme>/ops`, con il controllo del conteggio.
     - `run`: per insieme e braccio un processo nuovo; sha256 del checkpoint contro lo spec; operatori = esattamente le
       mesh messe in scena; `eval_v3.py` + `zs_embed.py` con gli argomenti di `d3_embed.sbatch` (`--subject_split all
       --eval_fraction 0.2 --max_subjects 0 --max_meshes_per_subject_eval 10 --preload_workers 8 --pair_mode
       cross_topology --seed 1234`, `--checkpoint_selector best_by_clean`); tabella di scala dell'insieme
       (`datasets/AVA256/scale/eval_view.npz`, `calib_view.npz`, dominio `ava256`).
     - **Rotazione del dominio**: `global_v3` legge R_d da `e12/frames.json` per nome del dominio; `ava256` non c'e', quindi
       entra in `global_v3.EXTRA_FRAMES` con u e R dello spec (= `frame.json`, controllato dalla sez. 11.8). Un dominio
       gia' presente (HIFI3D nel collaudo) deve avere la stessa R dello spec.
     - GPU obbligatoria (`zs_embed` ripiegherebbe sulla CPU in silenzio): L40S (`--gres=gpu:l40s:1`), come il collaudo.
     - Uscite: i percorsi letti da `conf_eval.py`, `conf_head.py` e `conf_desc.py` (`columns.models`,
       `columns.desc_models`, `head.embeddings`; controllato), con un json per insieme e braccio (sha256 di checkpoint e
       tabella, dominio, origine di R, GPU, nodo, job).
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

**Esecuzione** (dopo il commit del protocollo congelato). Un solo spec, `aau/ava256/spec/ava256.json` (sha256 nella sez.
11.1): vista `eval_view` dei 200 valutati, `calib_view` dei 56 (template del NICP, regione del fit B, L_d e cs_ref,
aggancio dei landmark, blocco di R6), frame di `frame.json`, GT, seme 20261011, sorgenti delle colonne, cartelle
`aau/runs/ava256_conf` (lavoro) e `aau/runs/evidence/ava256_conf` (evidenza). Tutti i passi con `aau/ava256/conf.sbatch`
(`CONF_SPEC` = lo spec, `CONF_STEPS` = i passi), lo STESSO codice collaudato su HIFI3D (sez. 11.2):
1. verifiche della sez. 11: primo passo di ogni job (`conf_common.py verify`; un solo scarto ferma il job);
2. protocolli (`conf_protocols.py`, sez. 9.1), in un job sul nodo `a256-t4-04` (quello del rilevatore di sviluppo):
   `versions gate define prep render detect detprep`, poi `run:k/n` (le 597.000 coppie, ripartibile; circa 1.500
   ore-core, stima dalle 30,5 ore-core delle 11.880 coppie di HIFI3D) e `merge`;
3. baseline: `blmm-scalars blmm-template blmm-pairs:<modo>:<passo>` (`blmm_*.py --spec`); fit B: `bp-fit bp-fit-e1
   bp-crop` (`bp_*.py --spec`; preregistrata e vA si calcolano solo come passi intermedi di B: la regione e
   l'inizializzazione; non si leggono), su un nodo `i256-a10` (`--nodelist=i256-a10-08`, dove il collaudo riproduce lo
   sviluppo a scarto 0; sez. 11.2);
4. embedding (v4, sez. 7.1): `embed-stage` (link, nomi, operatori; CPU), poi `embed` su una L40S (`--gres=gpu:l40s:1`):
   i due semi e C3M e123 sulle 1.200 mesh di `eval_view`, i due semi sulle 280 senza crop di `calib_view`, nei percorsi
   dello spec;
5. descrittivi (v4, sez. 9, `conf_desc.py`): `desc-spectral`, `desc-icp-same`, `desc-arcface` (su un nodo `i256-a10`,
   `--nodelist=i256-a10-09`, quello di G0 e del collaudo), `desc-gw-prep`, `desc-gw-run:k/n` (le 597.000 coppie di
   all_cross x 2 modi, ripartibile; nel collaudo mediana 0,03 s per soluzione), `desc-gw-merge`, `desc-perc`;
   `desc-varifold[:k/n]` su una L40S (circa 720.000 prodotti interni; stima ~1,5 ore dai 1.367 s delle 600 x 600 del
   collaudo);
6. testa di R6: `head-fit` sul nodo `i256-a10-10` (`conf_head.py`: K0, poi un fit per seme, sez. 10);
7. `eval` (`conf_eval.py`): righe, repliche, +inf, delta, Holm, non risolto, esiti; versione mascherata, sez. 9.2 e
   Spearman dei protocolli accanto; `desc-eval` (`conf_desc.py`, v4): descrittivi e gruppi descrittivi (sez. 9);
8. si riporta tutto, comprese le righe mancanti e fallite.

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

**Evidenze di sviluppo** (solo famiglie sintetiche e dev; FaMoS TEST non usato). Per ogni regola la riga principale e'
la maschera (passo `comune`, descrittiva); sotto, **+inf** = passo "NICP fallito = distanza massima" di
`full_matrix/deltas.csv` (stessa cartella), la statistica che si legge su Ava-256 (sez. 6). Righe della matrice: `nocrop` di
`aau/runs/evidence/full_matrix/results.md` (commit `adc0df5`) per s1234 e per la scala, `deltas.csv` della stessa
cartella per s2345. Ordine: HIFI3D, FaceScape, FaceVerse con espressioni, FaceVerse neutra; "x / y" = s1234 / s2345.

- **R1**: +0.106 [+0.062, +0.154] / +0.088 [+0.040, +0.138]; +0.207 [+0.160, +0.257] / +0.216 [+0.166, +0.266]; -0.035 [-0.105, +0.036] / -0.019 [-0.091, +0.049]; -0.068 [-0.139, +0.011] / -0.038 [-0.105, +0.031] (results.md:469, 678, 880, 1075; deltas.csv:32, 17005, 32398, 40356). IC sopra 0 in 2 famiglie su 4.
  - **+inf** (letta su Ava-256): HIFI3D +0.105 [+0.062, +0.154] / +0.088 [+0.039, +0.138]; FaceScape +0.192 [+0.144, +0.241] / +0.200 [+0.149, +0.252]; FaceVerse e neutra = maschera (il NICP non vi perde righe) (deltas.csv:8490, 8520, 24674, 24701).
- **R2a**: +0.134 [+0.085, +0.188] / +0.117 [+0.066, +0.171]; +0.119 [+0.030, +0.210] / +0.127 [+0.039, +0.213]; +0.091 [-0.010, +0.189] / +0.106 [+0.006, +0.201]; +0.117 [+0.013, +0.219] / +0.147 [+0.050, +0.241] (results.md:473, 682, 884, 1079; deltas.csv:36, 17009, 32402, 40360).
  - **+inf**: HIFI3D +0.134 [+0.085, +0.188] / +0.117 [+0.066, +0.171]; FaceScape +0.116 [+0.028, +0.207] / +0.124 [+0.037, +0.212]; FaceVerse e neutra = maschera (deltas.csv:8494, 8524, 24678, 24705).
- **R2b**: +0.253 [+0.170, +0.342] / +0.235 [+0.149, +0.329]; +0.286 [+0.229, +0.345] / +0.295 [+0.227, +0.357]; +0.102 [+0.015, +0.190] / +0.117 [+0.037, +0.194]; +0.063 [-0.027, +0.152] / +0.093 [+0.003, +0.181] (results.md:471, 680, 882, 1077; deltas.csv:34, 17007, 32400, 40358). Il NICP per coppia fallisce su righe che dipendono dalla topologia: righe nocrop in maschera 98224 / 99000 su HIFI3D e 88724 / 99000 su FaceScape, cioe' il 10,4% perso; up60k 25,9% (results.md:309-311, 318, 334).
  - **+inf**: HIFI3D +0.262 [+0.179, +0.350] / +0.245 [+0.160, +0.334]; FaceScape +0.362 [+0.307, +0.416] / +0.370 [+0.307, +0.427]; FaceVerse e neutra = maschera (deltas.csv:8492, 8522, 24676, 24703).
- **R3a**: +0.285 [+0.187, +0.383]; +0.038 [+0.018, +0.058]; +0.048 [-0.004, +0.102]; +0.035 [-0.018, +0.090] (results.md:150, 157, 164, 171); conteggio 2 / 0 / 2 su 4 (results.md:276).
  - **+inf**: HIFI3D +0.284 [+0.187, +0.382]; FaceScape +0.038 [+0.018, +0.058]; FaceVerse e neutra = maschera (deltas.csv:8670, 24836).
- **R3b**: -0.017 [-0.046, +0.012]; +0.069 [+0.043, +0.095]; +0.078 [+0.022, +0.129]; +0.088 [+0.022, +0.148] (results.md:151, 158, 165, 172); conteggio 3 / 0 / 1 su 4 (results.md:277).
  - **+inf**: HIFI3D -0.009 [-0.036, +0.020]; FaceScape **+0.155 [+0.127, +0.180]** (maschera +0.069); FaceVerse e neutra = maschera (deltas.csv:8671, 24837).
- **R4** (colonna crop): +0.245 [+0.199, +0.294] / +0.135 [+0.088, +0.184]; +0.062 [-0.002, +0.130] / -0.011 [-0.077, +0.054]; -0.019 [-0.083, +0.050] / -0.037 [-0.102, +0.028]; -0.034 [-0.098, +0.035] / -0.044 [-0.101, +0.015] (results.md:469, 678, 880, 1075; deltas.csv:296, 17245, 32646, 40586). IC sopra 0 in entrambi i semi solo su HIFI3D.
  - **+inf**: HIFI3D +0.246 [+0.200, +0.295] / +0.136 [+0.088, +0.185]; FaceScape +0.070 [+0.003, +0.138] / -0.003 [-0.069, +0.061]; FaceVerse e neutra = maschera (deltas.csv:8754, 8784, 24914, 24941).
- **R5a**: +0.041 [-0.005, +0.091] / +0.033 [-0.027, +0.088]; +0.269 [+0.217, +0.317] / +0.275 [+0.222, +0.328]; -0.020 [-0.093, +0.055] / +0.007 [-0.065, +0.078]; -0.063 [-0.140, +0.017] / -0.025 [-0.096, +0.051] (results.md:187-188, 198-199, 209-210, 219-220); conteggio 1 / 0 / 3 per seme (results.md:292-293).
  - **+inf**: HIFI3D +0.041 [-0.006, +0.091] / +0.032 [-0.026, +0.087]; FaceScape +0.254 [+0.202, +0.302] / +0.261 [+0.207, +0.314]; FaceVerse e neutra = maschera (deltas.csv:8681, 8709, 24847, 24872).
- **R5b**: +0.027 [-0.029, +0.083] / +0.018 [-0.046, +0.082]; +0.349 [+0.291, +0.401] / +0.355 [+0.290, +0.412]; +0.051 [-0.027, +0.124] / +0.078 [-0.005, +0.152]; +0.010 [-0.068, +0.095] / +0.049 [-0.041, +0.133] (results.md:191-192, 202-203, 213-214, 223-224); conteggio 1 / 0 / 3 per seme (results.md:296-297).
  - **+inf**: HIFI3D +0.037 [-0.019, +0.093] / +0.028 [-0.035, +0.092]; FaceScape +0.431 [+0.376, +0.481] / +0.438 [+0.379, +0.491]; FaceVerse e neutra = maschera (deltas.csv:8683, 8711, 24849, 24874).
- **R6** (D5, EQ, d_P con SR, media dei 4 bersagli sintetici, k = 50): +0.090 [+0.069, +0.109] / +0.066 [+0.043, +0.091] (`aau/runs/evidence/diagnostics/results_d5.md:26, 36`). k = 50 e' il primo k della griglia con F in entrambi i semi (results_d5.md:171-174); per bersaglio non sempre: FaceVerse neutra s2345 +0.018 [-0.050, +0.088] (results_d5.md:34).
  - **+inf**: nessun metodo per coppia nella regola, la lettura +inf coincide con la maschera.
- **R7** (riportata come NoW originale - NoW metrical, cioe' -delta; landmark oracolo, matrice completa,
  `aau/runs/evidence/protocolli/results.md`, commit `c960859`): -0.281 [-0.371, -0.194]; -0.053 [-0.092, -0.018];
  -0.038 [-0.078, +0.000]; -0.021 [-0.060, +0.019] (results.md:93, 106, 119, 132). IC sotto 0 in 2 famiglie su 4, la
  neutra non risolta. Col rilevatore (sottocampione 8%, `rumore_landmark/results.md`, commit `f47a375`) gli Spearman
  restano quasi uguali: HIFI3D 0.413 contro 0.685, FaceScape 0.616 contro 0.678 (rumore_landmark/results.md:24-25,
  36-37; delta appaiato non calcolato). R7 non usa i semi del modello: sez. 6.
  - **+inf**: NoW originale e metrical non hanno righe fallite su sviluppo (versi mancanti 0 in tutti i plug-in,
    `competitors_new/now_*/<dominio>.json`): la lettura +inf coincide con la maschera.

**Non risolto** (decisione del PI). Le regole che coinvolgono un metodo sono "non risolte" in un gruppo se, in quel
gruppo:
- il metodo ha piu' del 2% di mesh fallite (mesh delle topologie del gruppo: 1.000 per `nocrop_cross`, 1.200 per `crop`
  e `all_cross`);
- oppure piu' del 5% di righe mancanti (distanza non finita, contata prima della maschera comune).

Per i metodi per coppia (ICP, NICP per coppia, NoW originale e metrical) il fallimento e' della riga, non della mesh:
conta solo la soglia sulle righe, contate prima del +inf della sez. 6. Per i protocolli un ripiego del rilevatore (sez.
9.1) non e' un fallimento della riga (la distanza e' finita), ma ha una soglia propria (v3):
- **ripieghi del detector** (nessun volto trovato, `1k3d68` sul riquadro dei pixel colpiti, sez. 9.1): se in un gruppo
  piu' del **2%** delle mesh delle topologie del gruppo (1.000 / 1.200, come sopra) ha avuto il ripiego, **R7 e tutti i
  descrittivi dei protocolli** (NoW originale e metrical con FR e SR, REALY, Stirling, NoW metrical contro d_F della sez.
  9.2) sono "non risolti" in quel gruppo;
- **tassi di sviluppo** (`rumore_landmark/prep/<dominio>_det.json`, `det_fallback`, job 1071153): HIFI3D 1 mesh su 600
  (0,17%; per topologia: 1 su 100 = 1% su `noisy`, 0 nelle altre), FaceScape 1 su 600 (0,17%; 1 su 100 su `noisy`). La
  soglia del 2% e' 12 volte il tasso complessivo e il doppio del peggiore per topologia osservati; e' la stessa delle
  mesh fallite dei metodi per mesh;
- codice: `conf_protocols.FALLBACK_MAX` = 0,02, applicata per gruppo da `conf_eval.unresolved`; i conteggi per topologia
  in `<evidenza>/protocolli/detprep_ava256.json`.

Le regole non risolte si riportano comunque, con delta, IC e p, e restano nella famiglia di Holm (sez. 6).

Accanto a ogni regola, senza leggerli, si riportano gli stessi confronti sui domini di sviluppo, gia' noti (sopra).

## 9. Descrittivi (riportati tutti; nessuna lettura confermativa)

Ogni descrittivo si calcola con il codice al commit indicato. Il protocollo si congela solo quando non resta nessun
**[in attesa]**.

**Variante dichiarata.** Per i concorrenti nuovi e' quella con `declared: true` nel `fm_plugin.json` della cartella
`aau/runs/evidence/competitors_new/<metodo>/<variante>/`, scelta sui domini di sviluppo (piloti sui NON valutati) prima
di questo candidato. **Su Ava-256 si calcola SOLO la variante dichiarata**; le varianti con `declared: false` non si
calcolano. Per i metodi della matrice senza `fm_plugin.json` la variante e' unica ("unica" nella tabella).

**Deroga per i protocolli (v3, univoca).** Per NoW originale, NoW metrical, REALY e Stirling la regola qui sopra NON si
applica: vale la sez. 9.1. Su Ava-256 si calcola SOLO la variante `lmk_det` (landmark del rilevatore), anche se il suo
`fm_plugin.json` ha `declared: false` (su sviluppo e' una sensibilita' e resta tale). Il campo `declared` non si cambia;
`conf_eval.py` legge la variante per nome dallo spec (`columns.protocols.variant` = `lmk_det`), non da `declared`. Le
altre varianti dei protocolli (landmark oracolo, `lmk_s1`-`lmk_s5`) non si calcolano su Ava-256. Questo vale per R7 e
per tutti i descrittivi dei protocolli.

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
| C3M e123 (d_F cal., d_F non cal., d_P) | unica | FR, SR | checkpoint `aau/runs/evidence/trainer_v3/factorized/c3m/runs/v3_*_compactcache-true_head-factorized_seed1234__909688e6/checkpoints/epoch123_ema.pth`, sha256 `4ab9ca4531b06b0a199cb9c50f95d9756857663f4fa2078b6f97fee5412a2b77`, c 0.25732677076342675 (`factorized_calibration.csv:8`) | fissato |
| ArcFace-render, normal map 3 viste (G0) | unica (G0) | FR, SR | `aau/full_matrix/fm_arcface.py` (`c7e5299`) | fissato |
| Gromov-Wasserstein (POT) | `gw/euclid_128_cgw2` (`true`); `euclid_256_cg`, `euclid_256_cgw2` (`false`) | `fr` (mm), `sr` (a cs_ref) | `aau/competitors/gw_percettivi/` e `gw/pilot_choice.json` (`5c9c3e3`) | fissato |
| LPIPS alex, DINOv2 ViT-S/14, CLIP ViT-B/32, DreamSim su render | `lpips/depth`, `dinov2/depth`, `clip/smooth`, `dreamsim/smooth` (`true`); le altre 8 varianti di render (`false`) | `fr` (mm), `sr` (a cs_ref) | `aau/competitors/gw_percettivi/` e `percettivi/pilot_choice.json` (`5c9c3e3`) | fissato |
| Protocolli: NoW metrical (anche contro d_F, sez. 9.2), REALY, Stirling; NoW originale e metrical con GT SR | `lmk_det` (rilevatore, sez. 9.1) **per deroga** (sotto): ha `declared: false`; landmark oracolo e rumore sintetico (`lmk_s*`) non si calcolano | `mm` con FR e SR | `aau/competitors/protocolli/` (`32c3839`, `f47a375`), catena `aau/ava256/conf_protocols.py` (sez. 11.1) | fissato |
| ULRSSM | `ulrssm/faces_xyz` (`true` nel `fm_plugin.json`, NON ancora in git); `zeroshot_faust` (`false`) | per coppia (`pair_fr`, `pair_sr`) e via template (`tpl_fr`, `tpl_sr`) | `aau/competitors/ulrssm/` (`d727704`), codice esterno `0625cbcb` | **[in attesa]**: se variante dichiarata e pesi riaddestrati sono in git prima del congelamento entra, altrimenti **escluso** |
| Hybrid FM | nessuna | per coppia e per mesh | nessun codice nel repo | **[in attesa]**: se pronto (codice, variante dichiarata, commit) prima del congelamento entra, altrimenti **escluso** |
| Gruppi descrittivi (per perturbazione, stessa topologia) | - | come le regole | `aau/full_matrix/fm_eval.py` (`adc0df5`) | fissato |
| Testa adattata di R6 con d_F (FR) | unica | FR | sez. 10 | fissato |

Non si calcolano su Ava-256 (come nella matrice, dove esistono solo su alcuni domini): ctrlfr, C3M e205, ArcFace
ombreggiato (E12), OpenShape, Uni3D, coefficienti dei fit.

**Calcolo e lettura su Ava-256 (v4): `aau/ava256/conf_desc.py`**, stesso spec, passi `desc-*` di `conf.sbatch` (sez. 7,
Esecuzione 5 e 7). Funzioni di sviluppo importate invariate; mesh = i 200 valutati x 6 topologie, crop compresa (nella
matrice le righe crop venivano da script a parte, qui un solo calcolo con le stesse funzioni).
- **Spettrali**: `comp_spectral.spectral` (via `fm_compute._spec`) su ogni mesh, un thread BLAS per processo.
- **Varifold**: Gram fra tutte le mesh (`bp.varifold_measure`, `geometric_kernel.inner_products`, sigma 10 / 20 / 40 mm,
  blocco 8192) su una L40S, ripartibile per righe; distanza `bp_paired.varifold_distances`.
- **ICP + Chamfer di stessa topologia** (righe dei gruppi di stessa topologia): la regola di `fm_compute.icp_same`
  (`blmm_pairs.run_items`, passo `fast`, mm e cs, seme = indice della coppia). Il NICP per coppia e i protocolli su
  queste righe non si calcolano (come nella matrice): non applicabili su quei gruppi.
- **ArcFace** (G0): `g0_teacher.frame` e `run` su un insieme in memoria = le 1.200 mesh, rotazione attesa `x180` (R di
  `frame.json` ~ identita', come HIFI3D). Regola di `fm_arcface.py`, fissata qui: se il frame check (detector sui render,
  nessuna GT) sceglie un'altra rotazione unica, si usa quella e si dichiara (`frame_note.json`); senza una scelta unica
  ArcFace non si calcola e si riporta come non applicabile (`na.json`). Nodo `i256-a10-09` (quello di G0 e del collaudo).
- **GW e percettivi**: `gp_gw` (`run_prep`, `run_view`, `run_merge`) nella configurazione dichiarata `euclid|128|cgw2`
  sulle 597.000 coppie di all_cross; per i percettivi SOLO le quattro varianti dichiarate (`lpips/depth`, `dinov2/depth`,
  `clip/smooth`, `dreamsim/smooth`): si rendono solo le colorazioni `depth` e `smooth`, le altre varianti non si
  calcolano (regola della variante dichiarata, sopra). Vista di `gp_common` da uno spec generato (L_d e cs_ref di
  `blmm_scalars.py` sui 56); le scelte del pilota (`competitors_new/{gw,percettivi}/pilot_choice.json`) si controllano
  per sha256 contro lo spec e contro le varianti dichiarate.
- **Lettura** (`desc-eval`, legge le GT; `conf_common.verify` prima): righe di `conf_eval` (topologie diverse) + stessa
  topologia (19.900 x 6 = 119.400, sez. 5); gruppi `nocrop`, `crop`, `all_cross`, per perturbazione P senza crop
  (`con_perturbazione`) e di stessa topologia (tutte e per topologia), con le definizioni di `fm_eval.groups_of`;
  colonne: tutte quelle di `conf_eval` + C3M e123 (d_F cal., d_F non cal., d_P; c della tabella), fit B (`_vb_fr`,
  `_vb_sr`, `_vb_comp`: `vb_beta` di `fit_e1.npz` e `fit_e2_crop.npz`, `bp.mesh_distances`, composizione col k di
  `calib_e2/k.json`), varifold, ShapeDNA / HKS / WKS, taglia stimata (`est_cs`), ArcFace, GW (`fr`, `sr`), i quattro
  percettivi (`fr`, `sr`). Due letture, con le repliche della sez. 6: **confermativa (+inf)** (maschera della sez. 6,
  righe fallite dei metodi per coppia della famiglia = +inf) e **mascherata** (in piu' tutte le colonne per coppia della
  famiglia finite). Un descrittivo con buchi su quelle righe va nel passo `righe proprie di <colonna>` con i
  riferimenti sulle stesse righe. Record di `fm_eval.records`: Spearman con IC, delta `vs_nostro` e
  `scala_allineamento`; `non_risolto` per gruppo con la regola della sez. 8 e 9.1 sulle topologie del gruppo. Uscite:
  `<evidenza>/desc/{spearman,deltas}.csv`, `controls.json`, `results.md`. Tutto descrittivo (sez. 13).

### 9.1 Landmark dei protocolli su Ava-256 (v2, decisione del PI; v3: deroga, controllo bloccante, ripieghi)

Su dati reali non esistono landmark esatti. NoW originale, NoW metrical, REALY e Stirling usano i landmark del
**rilevatore**: la condizione `det` di `aau/competitors/protocolli/pr_noise.py` e `pr_noise_detect.py` (commit
`f47a375`), la stessa pipeline e gli stessi parametri, senza nessuna scelta su Ava-256. E' la variante `lmk_det`, unica
calcolata su Ava-256 per la deroga della sez. 9 (il plug-in resta `declared: false`). Codice della catena (v3):
`aau/ava256/conf_protocols.py` (passi `define`, `prep`, `gate-render` / `gate-compare`, `render`, `detprep`, `run`,
`merge`), che importa INVARIATE le funzioni per mesh di `pr_noise.py` (`_render_task`, `_prep_task`, `_task`,
`subsample`), `pr_landmarks.py` (`define`, `prep`) e `pr_run.py` (`load`, `one_pair`); `pr_noise_detect.py --dir` (v3:
solo la cartella dei render come argomento, il calcolo non cambia).
- **Render** (`pr_noise.render_mesh`): una immagine per mesh, per tutte le 1.200 mesh di `eval_view` (6 topologie,
  crop compreso), mesh in mm nel frame di lavoro (`blmm.to_mm`: u = 1, R e t di `frame.json`, sez. 7.2).
  - Camera prospettica sull'asse +z, centrata sul riquadro xy della mesh, a `CAM_DIST` = 4,0 volte l'estensione.
  - **Frazione del volto nel render `FILL` = 0,45** (estensione xy / lato dell'immagine), immagine `IMG` = 640 x 640.
    Scelta su 8 render di sviluppo **sul solo punteggio del detector**, senza landmark ne' GT: a 0,75 il detector non
    trovava volti (punteggio 0,07-0,31), a 0,45 circa 0,88 (`pr_noise.py:80-81`). Non si ritara su Ava-256.
  - Raycasting di open3d, normali per vertice interpolate, Lambert con luce `LIGHT` (0; 0,4; 1) + ambiente `AMB` 0,25,
    colore uniforme `ALBEDO` (0,88; 0,71; 0,61), sfondo nero.
- **Rilevatore** (`pr_noise_detect.py`): insightface **0.7.3**, onnxruntime **1.23.2** su CPU (venv di MICA,
  `external/venvs/mica`, Python 3.10.12; versioni controllate da `conf_common.py versions`, sez. 11), pacchetto `buffalo_l`.
  - Detector `det_10g` (SCRFD) a 640 x 640; fra i volti trovati **il riquadro di area massima** (il suo punteggio
    `det_score` si registra); `1k3d68` sul suo riquadro, come `FaceAnalysis.get`.
  - Nessun volto: `1k3d68` sul riquadro dei pixel colpiti dal render (ripiego, contato e riportato per topologia).
    Soglia (v3): oltre il 2% delle mesh di un gruppo col ripiego, R7 e i descrittivi dei protocolli sono "non risolti"
    in quel gruppo (sez. 8; sviluppo: 1 mesh su 600 su HIFI3D e su FaceScape).
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
  - definizione per Ava-256 con `pr_landmarks.define` sullo spec (`conf_protocols.py define`; codice `32c3839`, v3: uscite
    nelle cartelle dello spec, `pr_common.use_roots`): template `datasets/AVA256/template.npz` (sha256 nella sez. 11),
    mappa `datasets/AVA256/corr/ava256.npz` (sha256 nella sez. 11), aggancio sulla media delle original dei **56 di
    calibrazione** (sola
    geometria, sez. 10). Frame della spec = u, R, t di `frame.json` (sez. 7.2), non l'identita'.
  - **Non applicabilita'**: se la mappa manca o se `define` fallisce il controllo del frame (mediana media -> template
    > `FRAME_MAX_MM` = 2 mm), la corrispondenza su Ava-256 non esiste e **REALY e' non applicabile**: si riporta come
    tale, con il motivo, senza ripieghi. Una regione vuota in un singolo riferimento resta come su sviluppo (errore NaN
    della regione, `pr_metrics.realy_pair`). NoW e Stirling non dipendono dall'oracolo: usano solo i 7 landmark del
    rilevatore.
- **Coppie**: tutte le 597.000 coppie non ordinate di `all_cross` (nessun sottocampione: `FRAC` di `pr_noise.py` non si
  usa), i due versi, distanza = media.
- **Controllo del rilevatore prima di Ava-256 (v3, BLOCCANTE)**: nello stesso job e sullo stesso nodo del render di
  Ava-256 (`a256-t4-04`, quello del job di sviluppo 1071153), passo `gate` di `conf.sbatch`: 24 mesh di HIFI3D (vista di
  sviluppo, `rng(1234)`, `conf_protocols.GATE_N`), render rifatto e rilevatore rifatto, contro i png e i punti 2D di
  `aau/runs/protocolli/noise/det/hifi3d` (job 1071153). **Tolleranza 0**: immagini uguali pixel per pixel, camere
  uguali, punti 2D con scarto massimo 0 px, stessi ripieghi. Uno scarto -> `SystemExit`, il job si ferma prima di ogni
  render di Ava-256 e non si prosegue senza una decisione del PI. Esito in `<evidenza>/protocolli/gate_detector.json`.
  Nel collaudo (sez. 11.2) il controllo e' passato a tolleranza 0, e lo stesso vale per tutte le 600 mesh di HIFI3D.
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
  (PROTOCOL_D5.md:126-128). Si registrano nodo, job, esito (`success`) e iterazioni; nessun rifit. Un fit non
  convergente (`success` = False) si usa com'e' e R6 resta nella famiglia di Holm col suo p (sez. 6, v3).
- **Codice (v3)**: `aau/ava256/conf_head.py` (`fit`, sez. 11.1): blocco del dominio = `head` dello spec (i 56 x 5
  etichette, embedding di `calib_view`, `calib_gt/ava256_calib_sr.npz`), `d3_e2.blocks_for(braccio, "joint")` + il
  blocco in coda, `d3_e2.fit_e2`, c con `d3_head.calib_c` sulle sole 8 sorgenti; uscita letta da `conf_eval.py`.
- **Lettura**: delta = rho(d_h, SR) - rho(d_P, SR), stesso seme, `nocrop_cross` dei 200 valutati, stesse repliche
  (sez. 6).
- **d_F della testa** (FR): solo descrittiva. **Scostamento da D5 dichiarato**: D5 calcola il c della testa anche sul
  blocco del bersaglio (PROTOCOL_D5.md:52-53, `d3_e2._refit`). Qui `d3_head.calib_c` usa le sole 8 sorgenti, senza il
  blocco Ava, perche' la GT dei 56 non deve entrare in nessun c.
- **Controllo K0-Ava** (prima del blocco Ava, nessun dato Ava):
  - la testa con le sole 8 sorgenti, stessa configurazione e stesso nodo, deve riprodurre c e alpha/alpha0 di
    `d3_e2_heads.csv:21, 31` entro 1e-6 relativo;
  - se non li riproduce, si riporta lo scarto e si prosegue (come K0 di D5);
  - collaudo (v3, sez. 11.2): sullo stesso nodo, scarto relativo 0 su c e alpha / alpha0 per entrambi i semi.

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
5. **Checkpoint**: sha256 dei due `epoch072_ema.pth` (sez. 7.1) e di C3M e123 (sez. 9; v4: `columns.desc_models` dello
   spec, controllato da `conf_common.verify` e di nuovo da `conf_embed.py` prima dell'embedding).
6. **Rilevatore** (v2; v3: versioni): sha256 dei pesi di `buffalo_l` prima del primo render, e versioni dei pacchetti
   nel venv di MICA (`conf_common.py versions`, primo passo del job del rilevatore):

   | Voce | Valore atteso |
   |---|---|
   | `~/.insightface/models/buffalo_l/det_10g.onnx` (sha256) | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
   | `~/.insightface/models/buffalo_l/1k3d68.onnx` (sha256) | `df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc` |
   | `insightface` (versione) | `0.7.3` |
   | `onnxruntime` (versione) | `1.23.2` |

7. **Dati della mappa dei landmark (v3)**: sha256 dei byte, prima di `define` (sez. 9.1):

   | File | sha256 |
   |---|---|
   | `datasets/AVA256/template.npz` | `19a532492a6fcd86981ca8e91e96a273f5b697b6f36f251f322e44eff4363c68` |
   | `datasets/AVA256/corr/ava256.npz` | `475be926a1577190d8a8787feaa2827687fcf0894e694410f3d90bd01f6ea9e2` |

8. **Spec (v3)**: frame dello spec uguale a `frame.json`; soggetti dello spec = `evaluation_view_ids`, template e blocco
   di R6 = `calibration_view_ids` di `split.json`.

Le verifiche 1-8 sono `conf_common.verify` (con `freeze` nello spec), primo passo di ogni job di `conf.sbatch` e prima
riga di `conf_eval.py` e `conf_head.py`; le versioni del punto 6 sono `conf_common.versions` (venv di MICA).

### 11.1 Codice della catena (v3)

Commit del codice e degli esiti del collaudo (sez. 11.2): **`adaac1f`** (v3) e **`fe32e8f`** (v4: embedding e
descrittivi). sha256 dei byte di ogni script e degli spec (`conf_common.py sha`, `conf_common.SCRIPTS`, v4: due spec di
collaudo compresi); lo strumento non si esegue su Ava-256 con file diversi da questi (regola sui bug, sez. 12). Nella
v4 sono cambiati `conf_common.py` (sha256 di C3M in `verify`, lista degli script), `conf_collaudo.py` (passi `embed`,
`desc`, `desc-eval`), `conf.sbatch` (passi `embed-stage`, `embed`, `desc-*`) e i due spec `ava256.json` e
`hifi3d_full.json` (chiavi `embed`, `desc`, `columns.desc_models`, `columns.desc_plugins`); nessuno script di sviluppo
e' cambiato:

| Script | sha256 | Stato |
|---|---|---|
| `aau/ava256/conf_common.py` | `7229230de2b86935165c33daf69ad2d689b82df09b4a21946363cbd4e1cc7b8b` | modificato (v4) |
| `aau/ava256/conf_protocols.py` | `7f4db409cdf1f8f7486f1ac7de12908fa24c54ab25529b93839ca91e8c257e79` | nuovo (v3) |
| `aau/ava256/conf_head.py` | `9b74992f0264292d50bcb620824d60d0ccdf1eb258e19f07101af038fe3f140d` | nuovo (v3) |
| `aau/ava256/conf_eval.py` | `8169cc46ee43604807b95ff73c37b2b1d0a1990441b451ba83fdf1cb8afa1b15` | nuovo (v3) |
| `aau/ava256/conf_collaudo.py` | `3090c3ebefc13359e42a3cccffe3ebe26d25207d00a706bd23f92f43458def46` | modificato (v4) |
| `aau/ava256/conf.sbatch` | `712f0ec9b6bbb05a7a39d989577a3f24cfc512829ddc8d1bd55b8b90fd70bced` | modificato (v4) |
| `aau/ava256/spec/ava256.json` | `1b0a7e43d3769925b6c54387c361651ec803ca96daddd23614d8346f48097a07` | modificato (v4) |
| `aau/ava256/ava_freeze.py` | `520c002858619f2be5a2bb735c33a4663555230fc87e7b1ac34bc0c9c2c590ca` | invariato, importato |
| `aau/ava256/ava_calib.py` | `a56bf34b4eaeb37f899bbbfa4e4fd2680af82be5522d92deb66da9dbfa215897` | invariato, importato |
| `aau/competitors/protocolli/pr_common.py` | `deeb13ed5e755ff3f591518f26e08577d3172b873744504711ba858120df84ab` | modificato (v3: `--spec`) |
| `aau/competitors/protocolli/pr_landmarks.py` | `da5096451e981325c9c7049d09098b5f21523e6117f33a5e46a7491d84b62301` | modificato (v3: `--spec`) |
| `aau/competitors/protocolli/pr_metrics.py` | `75293276c3a07f7b70b324859527a32f19f43623e64f7d1fd68ab6eb59981eda` | invariato, importato |
| `aau/competitors/protocolli/pr_run.py` | `dffbae45ada9f2a26628f4e0b4761ea6563fa3b26c9a3ad485993f9530377238` | modificato (v3: `--spec`) |
| `aau/competitors/protocolli/pr_noise.py` | `6f16ac05c8cc71298508988062de5ddac85d14ad2d3caad2d127247010fac4c0` | invariato, importato |
| `aau/competitors/protocolli/pr_noise_detect.py` | `063ca84b315c44925fa2357ead410a10be2e199702f44427f31968d6585bebb9` | modificato (v3: `--spec`) |
| `aau/baselines_mm/blmm.py` | `3af056eebe2addfbf3e881f1ae4406bdf67d6bb9ed4ac263ffd4937c3b83f1bc` | modificato (v3: `--spec`) |
| `aau/baselines_mm/blmm_scalars.py` | `b34b8cfbf3bdf562bf698f274d5a8ebd2ffcb786d7bc7e088fb59536f41ae37d` | modificato (v3: `--spec`) |
| `aau/baselines_mm/blmm_pairs.py` | `34bde06d2b454090891700591d0b78a0ea1960bdff67a14c423d06db90c26788` | modificato (v3: `--spec`) |
| `aau/baselines_mm/blmm_template.py` | `870a422674597ddb2a9adebceed13dd0fb791728ef94a806ba3a385aaed19496` | modificato (v3: `--spec`) |
| `aau/baselines_param/bp.py` | `1b678c51f0169c69ded1523323484058fadf0f2d2ba4ce8cc369863cc4138389` | modificato (v3: `--spec`) |
| `aau/baselines_param/bp_fit.py` | `c9ecf86135d5a3eaf6ada8222f2e561582bc06d3acbe47ab901ad2d2a1e25a39` | modificato (v3: `--spec`) |
| `aau/baselines_param/bp_fit_e1.py` | `a97d98a86d507f506999ba60aab844c03dc98c84bde8184e7afa68a04dde4fdf` | modificato (v3: `--spec`) |
| `aau/baselines_param/bp_e2.py` | `0e43db2cc3e1d50ef199b2b4d9b7ea3ae5d60f47c6dc2237f65543a116bdca98` | modificato (v3: `--spec`) |
| `aau/full_matrix/fm_eval.py` | `5e0a2f5d4502b064bd61251ef8a716e6db238b4f864244f07b914d07b2d68958` | invariato, importato |
| `aau/diagnostics/d3_head.py` | `a3161d56513fc5dba6b4cd10e8884ac1f5788f6eb6884bf020b0d7707410754d` | invariato, importato |
| `aau/diagnostics/d3_e2.py` | `42b962a432fa9f219cbe754f44c41da11a3b2a3143cb4a5262ff68a4d9c23542` | invariato, importato |
| `aau/diagnostics/d3_stats.py` | `573aef82d60feb7722dafa031c2ea268bf64f35dd20efe5b2124d8e322a6fcc9` | invariato, importato |
| `aau/diagnostics/diag.py` | `399882d0e81b4f1e7458bb0e8bd77777cf42ad8b8847b716d667a2a11f43af4d` | invariato, importato |
| `v3_work/trainer/tools/fact_paired.py` | `cb37bf32916237077069157d8c3ac79c2838b6a85fcbebdb639b8419b2fe709a` | invariato, importato |
| `aau/ava256/conf_embed.py` | `f5a6791e87ecaf9b7a72032b06360e0818173754f1d281b4e0e83a3f3a8a6817` | nuovo (v4) |
| `aau/ava256/conf_desc.py` | `7687adb0d1a8b90d2e2060799f6730c52cc495d7ff3df30c1fdfe1d57b217f79` | nuovo (v4) |
| `aau/zs3dmm/zs_embed.py` | `195b189111fe91cb6ea5b02826e5cede87815bbe7bf8e861d7d91cb5dd3d6cf8` | invariato, importato (v4) |
| `v3_work/trainer/eval_v3.py` | `cf9f9d240bc0105ac94b29209037f5a626c537ed462d0ce3a76a448ac04a6613` | invariato, importato (v4) |
| `v3_work/trainer/global_v3.py` | `408ac38d549784469c463215c27ed6253534f2b09e402fbbc50873f7306edfdb` | invariato, importato (v4) |
| `v2_work/potential/areanorm_operators.py` | `511d0a52758b91c4948cfbdb7c53ad68854b4e1cec5351f23c779ee15c96008f` | invariato, importato (v4) |
| `aau/full_matrix/fm_compute.py` | `b0dad61c47ceb9328df018a77893410b9dbc0e5f4e24e87fb167f959f441c0af` | invariato, importato (v4) |
| `aau/competitors/comp_spectral.py` | `a485eebe9081c295ce4d91cf5e2300a287c0aa99741170188d8d6aaba81a68b1` | invariato, importato (v4) |
| `aau/baselines_param/bp_paired.py` | `a0bffb67b4de77a7e17933c9ab7a1dd6ec79c3943944998a0f31b4c0fbcb4432` | invariato, importato (v4) |
| `aau/baselines_param/bp_paired_e2.py` | `be9c190fa0c22ff7aa09fab8fb5c1c1be9ba4dda17f9cc370e59742579d2b8b4` | invariato, importato (v4) |
| `aau/baselines_mm/blmm_eval.py` | `7d439122deadd5c1502492c15ca463681984a2ca24fb8e2863d372c7fd12cda8` | invariato, importato (v4) |
| `aau/diagnostics/g0_teacher.py` | `d50e76ca6452cd38b92d9a0f547da2e7edcb9a678900d02bfb42b5b029144832` | invariato, importato (v4) |
| `aau/competitors/gw_percettivi/gp_common.py` | `3f58957396ec135dce1c85aca0d5954edf8814d4e3ad5a9c2d6f5a83630f018a` | invariato, importato (v4) |
| `aau/competitors/gw_percettivi/gp_gw.py` | `d64f04401d25f176dc927894f960e81e936b523c0da581813e0fd90dce2895c4` | invariato, importato (v4) |
| `aau/competitors/gw_percettivi/gp_percettivi.py` | `dad7542ae75a3babffc6f8166ae59dd4834394123ba90f8abb90d0b3a4933b24` | invariato, importato (v4) |
| `aau/ava256/spec/hifi3d_full.json` | `57d1c260690c67e8caacd9b626dc89907b2b7fb2a042afb41ab6685266493173` | modificato (v4) |
| `aau/ava256/spec/hifi3d_det.json` | `5b12b58f255118c8748ed923edec9cb2e0c03108423be362b0deb47612c760ea` | nuovo (v3) |

### 11.2 Collaudo su HIFI3D (v3, stesso codice, nessun dato di Ava-256)

HIFI3D passa dallo STESSO percorso `--spec` di Ava-256 (`conf.sbatch`, specifiche `aau/ava256/spec/hifi3d_full.json` e
`hifi3d_det.json`, che differiscono da `ava256.json` solo nei dati: soggetti, frame, GT, seme della T1 990708, colonne di
sviluppo, sottocampione dell'8% per il rilevatore). Confronto con i risultati di sviluppo gia' in git
(`conf_collaudo.py`). **Tolleranze, scritte prima di eseguire** (`conf_collaudo.TOL`):
- 0 (uguale): render (pixel), camere, punti 2D del rilevatore, ripieghi, indici dei landmark, etichette, chiavi, blocco
  di R6, numeri di righe;
- 1e-12: Spearman, delta, IC e p del valutatore contro `full_matrix` e `protocolli` (stessi ingressi, cambia solo
  l'ordine delle righe);
- 1e-4 mm: geometria ricalcolata (landmark, `D_mm` dei protocolli, matrici e template di blmm, fit B), il limite del
  controllo di sviluppo `pr_noise.check` (REALY <= 1e-4 mm);
- 1e-4: Spearman e delta col rilevatore contro `rumore_landmark` (dipendono dalla geometria ricalcolata);
- 1e-9 relativo: scalari di blmm (cs, L_d, cs_ref); K0 della testa: 1e-6 relativo (sez. 10).

**Esiti** (13 ottobre; uscite in `aau/runs/evidence/ava256_conf/collaudo/<passo>.json`; job 1071283 su `a256-t4-04`,
1071286 su `i256-a10-10`, 1071285 su `a512-l4-06`, 1071673 su `i256-a10-08`, confronti finali col codice della sez. 11.1
nel job 1071672):

| Passo | Confronto | Scarto massimo misurato | Tolleranza | Esito |
|---|---|---|---|---|
| `gate` (24 render) | png, camere, punti 2D, ripieghi contro il job 1071153 | 0 pixel, 0 px | 0 | passa |
| `det` (600 mesh) | render, punti 2D, ripieghi (1 su 600, `noisy`), landmark retroproiettati | 0 pixel, 0 px, 0 mm | 0 / 1e-4 mm | passa |
| `det` | `D_mm` dei 4 protocolli sulle 11.880 coppie del sottocampione | 0 mm | 1e-4 mm | passa |
| `landmarks` | indici 7 / 68 / 85, etichette di regione, prep (L68, K85) | uguali, 0 mm | 0 / 1e-4 mm | passa |
| `eval-full` | 54 + 72 Spearman e 14 + 14 delta (R1-R5b, due semi) contro `full_matrix` passo `comune` / "NICP fallito = distanza massima"; 24 Spearman dei protocolli e R7 contro `protocolli` | 0 (rho, IC, delta, p, righe) | 1e-12 | passa |
| `eval-det` | 24 Spearman dei protocolli col rilevatore e 3 delta NoW metrical - d_F non cal. s1234 contro `rumore_landmark` | 0 | 1e-4 | passa |
| `blmm` | scalari di 700 mesh, L_d, cs_ref | 4,0e-15 relativo | 1e-9 rel. | passa |
| `blmm` | R del NICP su template mm e cs (600 iscrizioni) | 0 mm | 1e-4 mm | passa |
| `blmm` | ICP + Chamfer mm / cs (1.000 coppie), NICP per coppia mm / cs (100 coppie) | 0 / 4,0e-14 / 1,2e-10 / 4,1e-5 mm; stessi fallimenti | 1e-4 mm | passa |
| `bp` su `i256-a10-08` (nodo di sviluppo) | regione, fit preregistrato, variante B senza crop e col crop (beta, `D_vb_fr`, `D_vb_sr`) | 0 | 1e-4 | passa |
| `bp` su `a512-l4-06` | gli stessi | regione uguale, preregistrato 1,7e-8; variante B: beta 1,4e-2, `D_vb` 1,6e-3 mm | 1e-4 | **fuori tolleranza** |
| `head` | blocco (X, soggetti, etichette, G) contro `d3_stats.build_sources`; K0 (c, alpha / alpha0) contro `d3_e2_heads.csv:21, 31` | uguali; 0 relativo | 0 / 1e-6 rel. | passa |

- La variante B del fit dipende dalla CPU: stesso codice, stesso nodo di sviluppo = scarto 0; su un'altra CPU la ricerca
  delle corrispondenze amplifica scarti di 1e-8 del fit iniziale fino a 1,6e-3 mm (anche due esecuzioni di sviluppo
  differiscono: `time_e2.npz` contro `fit_e1.npz`, beta fino a 8,8e-3 su FLAME). La tolleranza non si allarga: su
  Ava-256 il fit B (descrittivo) gira sui nodi `i256-a10` (sez. 7, Esecuzione 3), dove il collaudo passa; l'esito fuori
  tolleranza su `a512-l4-06` resta registrato (`collaudo/bp_a512-l4-06.json`).
- Il collaudo non tocca Ava-256: spec di HIFI3D, GT e risultati di HIFI3D gia' in git. Gli esiti delle regole su HIFI3D
  (seme 990708) stanno in `collaudo/hifi3d_full/conf/rules.csv` solo come controllo del codice: non sono evidenze nuove e
  non si leggono.
- Non collaudati nella v3: `blmm_pairs.py --spec` su tutte le coppie (collaudate le funzioni per coppia sulla vista da
  spec, non il ciclo, che non cambia); il ramo "REALY non applicabile" di `conf_protocols.py` (su HIFI3D la mappa
  esiste). L'embedding e i descrittivi fuori da `conf_eval.py` sono collaudati nella v4 (sotto).

**Collaudo v4: embedding e descrittivi** (13 ottobre, spec `hifi3d_full.json`, stesso percorso di Ava-256: passi
`embed-stage`, `embed`, `desc-*`, `desc-eval` di `conf.sbatch`). Riferimenti di sviluppo gia' in git o su disco: gli
embedding di `form_hifi` (A100, `nv-ai-04`) e del pool NON valutato di D3, le colonne di `full_matrix`, `protocolli` e
`competitors_new`. **Tolleranze v4, scritte prima di eseguire** (`conf_collaudo.TOL`, commento nel codice):
- `embed_abs` 1e-4: Z = [s, u] e distanze d_F (mm), d_P ricalcolate; Spearman e IC con `spearman_geom` 1e-4;
- `desc_rel` 1e-4 (relativo): autovalori, HKS, WKS, Gram varifold ricalcolati (`fm_varifold.py` misura 1,2e-5 fra due
  device);
- `geom_mm` 1e-4 mm: ICP + Chamfer di stessa topologia e GW ricalcolati;
- `perc_abs` 1e-4: ArcFace (1 - coseno) e percettivi ricalcolati; ritaglio di ArcFace (calibrazione del detector) e
  chiavi: uguali (0);
- lettura (`desc-eval`): 1e-12 (`num`) sulle colonne con gli stessi ingressi di sviluppo; 1e-4 (`spearman_geom`) su
  quelle ricalcolate (varifold, spettrali, ArcFace, taglia stimata, ICP sulle righe di stessa topologia), 1e-3
  (`p_geom`, una replica su 1000) sui loro p; righe uguali (0).

Esiti (job 1071702 operatori su `a256-t4-04`; 1071703 embedding su L40S `a768-l40s-01`; 1071731 varifold su L40S
`a768-l40s-01`; 1071723 spettrali, ICP, ArcFace, GW, percettivi su `i256-a10-09`; 1071724 lettura e confronti su
`a256-t4-04`; uscite in `aau/runs/evidence/ava256_conf/collaudo/{embed,desc,desc-eval}.json` e
`collaudo/hifi3d_full/{embed,desc}/`):

| Passo | Confronto | Scarto massimo misurato | Tolleranza | Esito |
|---|---|---|---|---|
| `embed` | Z delle 600 mesh di `form_hifi` (s1234, s2345, C3M e123) e delle 1.000 del pool di D3 (s1234, s2345), L40S contro A100 | 6,3e-4 / 7,2e-4 / 1,2e-3 (C3M); pool 6,5e-4 / 7,7e-4 | 1e-4 | **fuori tolleranza** |
| `embed` | d_F cal., d_F non cal., d_P su tutte le coppie (179.700 e 499.500) | d_F 0,057 mm, d_P 1,1e-3 | 1e-4 | **fuori tolleranza** |
| `embed` | Spearman e IC (nocrop, crop, all_cross; GT FR e SR) delle 9 distanze, righe e repliche di `conf_eval` | 4,3e-4 (C3M, crop, SR); semi <= 2,1e-4 | 1e-4 | **fuori tolleranza** |
| `desc` spettrali | autovalori 1-100, HKS, WKS di 600 mesh contro `competitors_hifi3d/desc_spectral.npz` | 7,9e-10 rel. | 1e-4 rel. | passa |
| `desc` varifold | Gram 600 x 600 contro gli shard di `bp_varifold.py` + le righe crop di `fm_varifold.py` | 2,7e-7 rel. (identico bit per bit in due esecuzioni su L40S) | 1e-4 rel. | passa |
| `desc` ICP stessa topologia | 29.700 coppie x 2 modi contro `full_matrix/icp_same` | 0 mm (mm), 6,5e-14 mm (cs); 0 fallite | 1e-4 mm | passa |
| `desc` ArcFace | Z, E, camera, ritaglio contro `g0/teacher/hifi3d` (frame check: `x180`, come atteso) | 0 | 0 / 1e-4 | passa |
| `desc` GW (sottocampione: primi 12 soggetti per id, 72 mesh, 1.980 coppie) | D_fr, D_sr contro `competitors_new/gw/euclid_128_cgw2` | 0 / 8,8e-7 mm | 1e-4 mm | passa |
| `desc` percettivi (stesso sottocampione, 2.556 coppie) | D_fr, D_sr delle 4 varianti dichiarate contro `competitors_new` | <= 2,7e-7 (fr), <= 4,5e-5 (sr) | 1e-4 | passa |
| `desc-eval` | 782 Spearman (con IC) e 1.602 delta (`vs_nostro`, `scala_allineamento`) per lettura contro `full_matrix` (mascherata = `comune`, confermativa = "NICP fallito = distanza massima"), 15 gruppi | 0 (stessi ingressi); <= 5,7e-6 e p 0 (ricalcolati); righe 0 | 1e-12 / 1e-4 / 1e-3 | passa |
| `desc-eval` | 64 Spearman dei protocolli (8 gruppi, per perturbazione compresi) contro `protocolli/spearman.csv`; 160 rho puntuali di GW e percettivi contro `competitors_new/spearman.csv` | 0; 9,7e-17 | 1e-12 | passa |

- **Embedding fuori tolleranza: cause misurate** (diagnostica, non un criterio; job 1071727, factorized s1234, 600 mesh,
  max |dz|): L40S contro L40S con gli stessi operatori 4,8e-7 (ripetibile); L40S con gli operatori di sviluppo contro
  A100 di sviluppo 6,3e-4 (la GPU); L40S con gli operatori ricalcolati contro gli operatori di sviluppo 3,2e-4 (gli
  operatori differiscono solo all'ultimo bit float32: autovettori 2,4e-7, il resto 0); sviluppo contro sviluppo (A100,
  pool di D3 contro store, 25 mesh) 2,8e-4, e il controllo K7 di D3 aveva misurato 5,2e-4 (tolleranza di allora 1e-2,
  `results_d3.md:471`). Il container NGC ha `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` (moltiplicazioni fp32 in TF32): e' la
  causa probabile dello scarto fra le due GPU, non isolata (misurato l'effetto della GPU, non quello del TF32 da solo).
  La tolleranza di 1e-4 era quindi sotto il rumore di riproduzione del
  forward gia' noto in sviluppo. **Non si allarga**: l'esito resta "fuori tolleranza" e si registra; su Ava-256
  l'embedding gira su L40S come qui. Effetto sulle letture del collaudo: Spearman <= 4,3e-4.
- Non collaudati nella v4: GW e percettivi sulle 600 mesh (solo il sottocampione dichiarato); il varifold a piu' shard
  (`--shard k/n`, un solo shard nel collaudo); il ramo "ArcFace non applicabile" (su HIFI3D il frame check sceglie
  `x180`); i passi di `conf_desc.py eval` con descrittivi bucati (su HIFI3D nessun descrittivo ha buchi: passo
  `righe proprie` vuoto).

## 12. Vincoli

- Una sola valutazione per metodo. Nessuna scelta fatta su Ava-256: iperparametri, checkpoint, c, k, soggetti,
  regione, viste, split. Tutto si riporta, comprese righe mancanti e fallimenti.
- I 56 di calibrazione non entrano in nessuna riga di valutazione. La loro GT entra solo nella testa di R6, la geometria
  delle loro original anche nella definizione dei landmark dei protocolli (sez. 10).
- Nessun parametro del render o del rilevatore si ritara su Ava-256 (sez. 9.1).
- Gli strumenti sono estesi ad Ava-256 PRIMA del congelamento (v3 e v4, sez. 11.1) ma non si eseguono su Ava-256 prima del
  commit del protocollo congelato. Fino ad allora la vista porta `CONFERMATIVO_NON_VALUTARE.txt` e i dati sono in sola
  lettura.
- **Bug scoperti dopo il congelamento (v3).** Se dopo il congelamento si trova un errore nel codice della sez. 11.1, si
  riportano **entrambe** le versioni: i numeri del codice congelato e quelli del codice corretto, con il diff, il commit
  e il motivo. **Fa fede la versione congelata** per gli esiti di R1-R7; la corretta e' descrittiva e non cambia nessun
  esito. Lo stesso per un errore nei dati scoperto dopo (si riportano entrambi, fa fede il congelato).
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
2. Hybrid FM: codice, variante dichiarata e commit [in attesa; se non pronto al congelamento, e' escluso];
3. ~~embedding di Ava-256~~: **fissato nella v4** (`conf_embed.py`, sez. 7.1, 11.1, 11.2); collaudo **fuori
   tolleranza** e registrato (sez. 11.2), come il fit B su `a512-l4-06` nella v3;
4. ~~descrittivi della sez. 9 fuori da `conf_eval.py`~~: **fissati nella v4** (`conf_desc.py`, sez. 9, 11.1, 11.2).

Fissati nella v2 (non piu' in attesa): R7 e la famiglia di Holm (m = 10); Gromov-Wasserstein e percettivi (varianti
dichiarate in `5c9c3e3`); protocolli NoW, NoW metrical, REALY, Stirling (codice in `32c3839`, risultati di sviluppo in
`c960859`, pipeline del rilevatore in `f47a375`).

Fissati nella v3: catena dei protocolli col rilevatore, valutatore confermativo, blocco di R6, baseline e fit B con
`--spec`, collaudati su HIFI3D (sez. 11.1, 11.2); deroga dei protocolli (sez. 9); controllo bloccante del rilevatore e
soglia dei ripieghi (sez. 8, 9.1); versioni e sha256 della sez. 11.

Fissati nella v4: embedding dei due semi e di C3M e123 (`conf_embed.py`), calcolo e lettura dei descrittivi e dei
gruppi descrittivi (`conf_desc.py`), collaudati su HIFI3D con gli esiti della sez. 11.2 (descrittivi entro tolleranza,
embedding fuori), sha256 nella sez. 11.1.
Restano **[in attesa]** solo i punti 1 e 2 (ULRSSM, Hybrid FM), con la regola gia' scritta: se non sono pronti al
congelamento sono esclusi.

Poi il PI committa questo testo con il suo sha256 e la parola "congelato" nel titolo.

**CANDIDATO v4, NON CONGELATO: NESSUNA VALUTAZIONE SU AVA-256 FINO AL CONGELAMENTO.**
