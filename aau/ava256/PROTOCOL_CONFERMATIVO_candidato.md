# Ava-256: protocollo della prova confermativa (v9, CONGELATO il 13 ottobre 2026; codice congelato `8a6e184`)

> **v9, CONGELATO. Codice congelato: commit `8a6e184d2951d73df7ea55e48296c4df36b92aef`** (`8a6e184`): su Ava-256 si
> esegue SOLO da un worktree a quel commit, con `CONF_FROZEN_COMMIT=8a6e184d2951d73df7ea55e48296c4df36b92aef`
> (`aau/ava256/conf_worktree.sh`); impronte e comando in `aau/ava256/FROZEN.md`, tag git `ava256-frozen`.
>
> **Modifiche della v9 rispetto al candidato v8** (sha256 `d1364505a97e2d3127557341ecfbc173807d7d6c3ad402b4cbae13fd8564bfee`,
> commit `68199f0`): solo documento: codice congelato fissato a `8a6e184`, sha di `v3_work/mm/model.py` e
> `v3_work/stream/sources.py` nella sez. 11.1 a quel commit, tolta la condizione sul "commit da congelare".
>
> E' la bozza dell'11 ottobre (`PROTOCOL_CONFERMATIVO_bozza.md`, sha256 `38ce3a5d...`,
> commit `53b03ac`) completata con le decisioni del PI del 12 ottobre (sez. 5-9, 11-13), piu' l'emendamento sull'uso
> dei 56 di calibrazione (sez. 10). Il testo delle sez. 1-4 e' quello della bozza, salvo le note della v2 nella sez. 2.
>
> **Modifiche della v8 rispetto al candidato v7** (sha256 `8156154b5bc2d1b9847cbb5b2502ee305ac727ccd44e41d925d7196d0a2f8048`,
> commit `a4fa951`): RISERVE del critic mirato sulla v7.
> 1. **Clone diffusion-net** (sez. 11.10, 11.1): `conf_common.verify` controlla commit (`b1019b0`) e albero pulito del
>    clone, come per ULRSSM; `loaded_modules` registra i file importati del clone (`diffusion-net/<percorso>`, sha256
>    nella sez. 11.1) e da' ERRORE per ogni modulo sotto il repo o il worktree che risolve fuori, salvo i cloni elencati
>    (`conf_common.EXTERNAL_CLONES`) e le librerie installate dei venv (`site-packages`).
> 2. **Nodo del fit B nelle uscite**: `bp_fit.py`, `bp_fit_e1.py` e `bp_e2.py crop` scrivono `host`; il lettore del fit
>    B (`conf_desc.py`, `fit_hosts`; `conf_eval.py` non legge il fit B) lo confronta con `nodes.bp`, come per R6.
> 3. **Ripresa di `conf_head`** dopo un'interruzione fra i bracci (sez. 12): braccio con npz intero e sha256 registrato
>    -> si salta; altrimenti fermo.
> Collaudo (`guard` con i casi nuovi, tracciamento rifatto, collaudi numerici rifatti dal worktree): sez. 11.2.
>
> **Modifiche della v7 rispetto al candidato v6** (sha256 `8c8e88092e32d7ddbcdb5694696fe301ee8bcc3885d418e92e88a218a032e1c0`,
> commit `cc839e6`): terzo giro autorizzato dall'utente dopo il BLOCCANTE del critic finale sulla v6; corretti SOLO i
> punti A-F.
> - **A. Codice congelato garantito a runtime** (sez. 7 Esecuzione, 11.9, 11.1). Su Ava-256 si gira da un worktree di
>   git al commit congelato (`aau/ava256/conf_worktree.sh`); `conf_common.verify` controlla HEAD = `CONF_FROZEN_COMMIT`
>   (obbligatorio con `freeze`) e nessun file tracciato modificato; `aau/ava256/conf_run.py` lancia ogni passo Python e
>   registra gli sha256 di TUTTI i moduli del repo importati (anche nei processi figli), li confronta col commit e li
>   scrive in `verify.json`, `<evidenza>/modules/`, `conf/controls.json` e `conf/modules.json` (accanto a `rules.csv`),
>   `embed/*.json`, `head/*.json`. La sez. 11.1 elenca ora i 127 moduli importati da un collaudo completo della catena
>   (spec `hifi3d_trace.json`), ricavati in modo automatico, piu' i file non modulo. Una correzione resa necessaria per
>   girare da un worktree: `blmm.rel` prende i percorsi risolti relativi al repo che contiene `datasets/` (nel repo
>   principale non cambia nulla).
> - **B. R6**: `conf_head.py fit` si ferma su un nodo diverso da `head.node` e, con `freeze`, non sovrascrive
>   `head/<braccio>.npz`; `conf_eval.py` rifiuta una testa calcolata su un altro nodo.
> - **C. Sez. 12**: clausola per il codice congelato che va in crash su Ava-256 (correzione minima di una riga, col
>   diff; se tocca la catena di R1-R7, la regola e' non risolta).
> - **D. Nodi nel codice** per il fit B (`i256-a10-08`, `conf_common.py host --role bp`) e per ArcFace (`i256-a10-09`,
>   il nodo di G0 e del collaudo della v4, `desc.arcface.node`), come la classe di GPU per A100 e L40S; `--nodelist`
>   nelle righe d'uso di `conf.sbatch`.
> - **E. Citazioni di righe** di `deltas.csv`, `results.md` e `fm_eval.py` riferite al commit (`@adc0df5` per la
>   matrice, `@c960859` e `@f47a375` per i protocolli).
> - **F. ULRSSM**: letture dichiarate su Ava-256 solo `pair_fr` e `tpl_sr`; `pair_sr` e `tpl_fr` escono con
>   `dichiarata` = False (spec `desc.ulrssm.readings_declared`, `fm_plugin.json` di Ava-256, lettura di `conf_desc.py`).
> - `fm_eval.py` della matrice e' alla versione `41b7135` (sha256 `9101b3f9...`; funzioni della catena con AST identico).
> - Collaudo dei controlli e del tracciamento: sez. 11.2.
>
> **Modifiche della v6 rispetto al candidato v5** (sha256 `6b79aadee9e67c4c25d1a66d9f51388f06613f4e6d6b3b06e1d773b86d1cef91`,
> commit `0dc907d`), con la regola della sez. 14:
> 1. **ULRSSM entra come DESCRITTIVO** (sez. 9): variante dichiarata `faces_xyz` (selezione `selection.json`, fissata
>    prima delle viste di valutazione), letture per coppia con GT FR (`pair_fr`) e via template con GT SR (`tpl_sr`);
>    `pair_sr` e `tpl_fr` escono dallo stesso calcolo e si riportano come le altre letture. Codice
>    `aau/competitors/ulrssm/` (`--spec`), chiamato da `conf_desc.py` (passi `desc-ulr-run`, `desc-ulr-merge`); sha256 di
>    codice, checkpoint finale e `selection.json`, commit del clone `~/ext/ulrssm` (`0625cbc`) nella sez. 11.1.
>    Vincolo GPU: L40S, la classe della valutazione di sviluppo. Collaudo su HIFI3D su un sottocampione dichiarato
>    con la tolleranza fissata prima (sez. 11.2).
> 2. **Hybrid FM escluso definitivamente**: non pronto (nessun codice, variante o commit), per la regola della sez. 14.
> 3. Sez. 14: nessun campo **[in attesa]** resta aperto.
>
> **Modifiche della v5 rispetto al candidato v4** (sha256 `f0d60531173b3aa9a6eacf12b3357bc9b0e8421220aa7ce249c0e65a200c58d0`,
> commit `638c591`), decisioni del PI del 13 ottobre:
> 1. **Embedding su A100** (sez. 7.1, 7 Esecuzione 4): su Ava-256 l'embedding gira sulla classe di GPU degli embedding di
>    sviluppo, le A100 di `nv-ai-04` (`-p aicentre-a100 --qos=unprivileged --gres=gpu:a100:1`; job breve di
>    valutazione, ripartibile). E' un vincolo come il nodo del fit B; `conf_embed.py` lo controlla (`embed.gpu` = `A100`
>    nello spec). Il TF32 del container resta attivo, come in sviluppo.
> 2. **Tolleranza del collaudo dell'embedding** (sez. 11.2). La tolleranza della v4, 1e-4 su Z e distanze, **era sotto il
>    rumore di riproduzione gia' noto in sviluppo**: A100 contro A100 con operatori ricalcolati, max |dz| 2,8e-4 (pool di
>    D3 contro store) e 5,2e-4 (K7 di D3). Nuove tolleranze, dichiarate PRIMA del collaudo su A100: Z 1e-3; d_F (cal. e
>    non cal.) 0,1 mm; d_P 1e-3; Spearman e IC 1e-3. **L'esito fuori tolleranza su L40S della v4 resta registrato**
>    (`collaudo/embed_l40s.json`), con la sua tolleranza di allora. Collaudo rifatto su A100: sez. 11.2.
> 3. **Composizioni delle baseline con k** (sez. 9): non piu' "fissato" ma **non calcolata su Ava-256**: nessun codice
>    nella catena congelata. Nessun codice nuovo.
> 4. **Il crop esce dalla parte confermativa** (sez. 5, 6, 8, 13): R4 (d_F cal. contro ICP + Chamfer in mm sul gruppo
>    `crop`) lascia la famiglia di Holm, che passa da **m = 10 a m = 9** (soglie intere ricalcolate, sez. 6); il gruppo
>    `crop` e R4 restano calcolati e si riportano completi come descrittivi, e nel supplementary come limite.
>    **Motivazione**: e' una scelta di ambito del paper, non un esito. Il crop e' fuori dalla tesi principale per una
>    decisione precedente (11 ottobre: il crop di bordo esce dalla tesi e resta nei limiti). La scelta e' presa PRIMA di
>    qualunque lettura di Ava-256 ed e' dichiarata con le evidenze di sviluppo gia' qui: col crop NoW metrical sta sopra
>    la nostra d_F (sez. 9.2: HIFI3D +0.133 [+0.069, +0.203], FaceScape +0.170 [+0.083, +0.260]) e R4 ha l'IC sopra 0 in
>    entrambi i semi solo su HIFI3D (sez. 8). Codice: `conf_eval.py` (`RULES` senza R4, `DESC_RULES`, m = 9), sha256
>    nella sez. 11.1; collaudo rifatto (sez. 11.2).
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
| `crop` | **descrittivo** (v5; era secondario) | crop contro una delle altre 5 | 10 | 19.900 x 10 = 199.000 |
| `all_cross` | secondario | le 6, crop compreso | 30 | 19.900 x 30 = 597.000 |

- **Definizioni = quelle della matrice unificata** (`aau/full_matrix/fm_eval.py@adc0df5:473-477`, `groups_of`, commit
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
  `np.bincount(rng.integers(0, 200, 200), minlength=200)` sui 200 valutati ordinati come in fm_eval (`fm_eval.py@adc0df5:763-765`,
  con questo seme al posto di quello della T1).
  - Il seme riparte in ogni gruppo, quindi i conteggi sono gli stessi in tutti i gruppi, per tutti i metodi, per i due
    semi del modello e per tutti i delta.
- **IC 95%**: percentile sulle repliche 1-1000, riportato per ogni Spearman e ogni delta, senza correzione.
- **Delta appaiati** a - b sulle stesse righe e repliche. **p = P(delta <= 0)** = frazione delle 1000 repliche con
  delta <= 0 (`p_le0` di fm_eval).
  - Risoluzione 1/1000: il p minimo diverso da 0 e' 0,001; p = 0 si riporta come "< 0,001".
- **Confronti multipli: Holm**, per seme del modello, sulla famiglia {R1, R2a, R2b, R3a, R3b, R5a, R5b, R6, R7}:
  **m = 9** ipotesi (v1: 9; v2 aggiunge R7: 10; v5 toglie R4, gruppo crop, ora descrittiva), livello unilaterale 0,025.
  - Procedura: p ordinati p(1) <= ... <= p(m); si rifiuta H(i) finche' p(i) <= 0,025 / (m - i + 1); la prima che
    non passa ferma la procedura. A parita' di p l'ordine fra le ipotesi non cambia l'esito.
  - In interi, per evitare l'arrotondamento: con k(i) = numero di repliche su 1000 con delta <= 0, si rifiuta H(i) se
    **k(i) (m - i + 1) <= 25**. Soglie per m = 9 (v5):

    | i | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
    |---|---|---|---|---|---|---|---|---|---|
    | m - i + 1 | 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
    | 0,025 / (m - i + 1) | 0,0028 | 0,0031 | 0,0036 | 0,0042 | 0,0050 | 0,0063 | 0,0083 | 0,0125 | 0,025 |
    | k massimo (k (m - i + 1) <= 25) | 2 | 3 | 3 | 4 | 5 | 6 | 8 | 12 | 25 |
    | p massimo (k / 1000) | 0,002 | 0,003 | 0,003 | 0,004 | 0,005 | 0,006 | 0,008 | 0,012 | 0,025 |

  - La soglia piu' severa e' 0,025 / 9 = 0,0028: con la risoluzione di 1/1000 basta p <= 0,002 (al massimo 2 repliche
    su 1000 con delta <= 0), ancora sopra 0,001.
  - **R3a, R3b e R7 non dipendono dal seme** e si calcolano una volta sola: lo stesso p entra in entrambe le famiglie.
  - **m fisso (v3; v5: m = 9).** Ogni famiglia ha sempre le 9 ipotesi. Una regola "non risolta" (sez. 8) e la R6 con un fit
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
  - E' la "sensibilita' NICP fallito = distanza massima" di fm_eval (`fm_eval.py@adc0df5:827-833`, valore
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
     - GPU obbligatoria (`zs_embed` ripiegherebbe sulla CPU in silenzio). **v5: A100** (`-p aicentre-a100
       --qos=unprivileged --gres=gpu:a100:1`, nodo `nv-ai-04`), la classe di GPU degli embedding di sviluppo della
       matrice; vincolo controllato da `conf_embed.py` (`embed.gpu` dello spec); TF32 del container attivo come in
       sviluppo. La v4 indicava L40S: il collaudo su L40S e' uscito fuori tolleranza (sez. 11.2).
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
(`CONF_SPEC` = lo spec, `CONF_STEPS` = i passi), lo STESSO codice collaudato su HIFI3D (sez. 11.2).
**v7, codice congelato a runtime**: tutti i job partono da un worktree di git al commit congelato
(`aau/ava256/conf_worktree.sh <commit> <cartella>`: checkout staccato del commit, dati, venv e uscite come link),
sottomessi con `--chdir=<worktree>` e `CONF_FROZEN_COMMIT=<commit>`; `conf_common.verify` (punto 9 della sez. 11)
ferma il job se HEAD non e' quel commit o se un file tracciato e' modificato; ogni passo Python passa da
`aau/ava256/conf_run.py`, che registra gli sha256 di tutti i moduli del repo importati (anche nei processi figli) in
`<evidenza>/modules/` e ferma il job (uscita 3) se uno differisce dal commit:
1. verifiche della sez. 11: primo passo di ogni job (`conf_common.py verify`; un solo scarto ferma il job);
2. protocolli (`conf_protocols.py`, sez. 9.1), in un job sul nodo `a256-t4-04` (quello del rilevatore di sviluppo):
   `versions gate define prep render detect detprep`, poi `run:k/n` (le 597.000 coppie, ripartibile; circa 1.500
   ore-core, stima dalle 30,5 ore-core delle 11.880 coppie di HIFI3D) e `merge`;
3. baseline: `blmm-scalars blmm-template blmm-pairs:<modo>:<passo>` (`blmm_*.py --spec`); fit B: `bp-fit bp-fit-e1
   bp-crop` (`bp_*.py --spec`; preregistrata e vA si calcolano solo come passi intermedi di B: la regione e
   l'inizializzazione; non si leggono), su un nodo `i256-a10` (`--nodelist=i256-a10-08`, dove il collaudo riproduce lo
   sviluppo a scarto 0; sez. 11.2); v7: il nodo e' controllato nel codice (`conf_common.py host --role bp`, spec
   `nodes.bp` = `i256-a10-08`), un altro nodo ferma il job;
4. embedding (v4, sez. 7.1): `embed-stage` (link, nomi, operatori; CPU), poi `embed` su una A100 (v5: `-p aicentre-a100
   --qos=unprivileged --requeue --gres=gpu:a100:1`):
   i due semi e C3M e123 sulle 1.200 mesh di `eval_view`, i due semi sulle 280 senza crop di `calib_view`, nei percorsi
   dello spec;
5. descrittivi (v4, sez. 9, `conf_desc.py`): `desc-spectral`, `desc-icp-same`, `desc-arcface` (su un nodo `i256-a10`,
   `--nodelist=i256-a10-09`, quello di G0 e del collaudo), `desc-gw-prep`, `desc-gw-run:k/n` (le 597.000 coppie di
   all_cross x 2 modi, ripartibile; nel collaudo mediana 0,03 s per soluzione), `desc-gw-merge`, `desc-perc`;
   `desc-varifold[:k/n]` su una L40S (circa 720.000 prodotti interni; stima ~1,5 ore dai 1.367 s delle 600 x 600 del
   collaudo); v6: `desc-ulr-run[:k/n]` su una L40S e `desc-ulr-merge` (ULRSSM);
6. testa di R6: `head-fit` sul nodo `i256-a10-10` (`conf_head.py`: K0, poi un fit per seme, sez. 10); v7: `conf_head.py`
   si ferma su un altro nodo e, con `freeze`, non sovrascrive una testa gia' scritta; `conf_eval.py` rifiuta una testa
   calcolata su un altro nodo;
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
| R4 (**descrittiva dalla v5**: fuori dalla famiglia di Holm, nessun esito) | d_F cal. | ICP + Chamfer in mm | FR | `crop` | s1234, s2345 |
| **R5a** | d_P | ICP + Chamfer cs | SR | `nocrop_cross` | s1234, s2345 |
| **R5b** | d_P | NICP per coppia cs | SR | `nocrop_cross` | s1234, s2345 |
| **R6** (adattamento few-shot, ex H2) | testa adattata sui 56 (sez. 10) | d_P dello stesso seme | SR | `nocrop_cross` | s1234, s2345 |
| **R7** (scala per coppia nei protocolli: NoW originale - NoW metrical < 0) | NoW metrical (Procrustes senza scala + ICP rigido), landmark del rilevatore | NoW originale (Procrustes + ICP con scala), stessi landmark | FR | `nocrop_cross` | nessuno (una volta) |

**Evidenze di sviluppo** (solo famiglie sintetiche e dev; FaMoS TEST non usato). Per ogni regola la riga principale e'
la maschera (passo `comune`, descrittiva); sotto, **+inf** = passo "NICP fallito = distanza massima" di
`full_matrix/deltas.csv` (stessa cartella), la statistica che si legge su Ava-256 (sez. 6). Righe della matrice: `nocrop` di
`aau/runs/evidence/full_matrix/results.md` (commit `adc0df5`) per s1234 e per la scala, `deltas.csv` della stessa
cartella per s2345. Ordine: HIFI3D, FaceScape, FaceVerse con espressioni, FaceVerse neutra; "x / y" = s1234 / s2345.

- **R1**: +0.106 [+0.062, +0.154] / +0.088 [+0.040, +0.138]; +0.207 [+0.160, +0.257] / +0.216 [+0.166, +0.266]; -0.035 [-0.105, +0.036] / -0.019 [-0.091, +0.049]; -0.068 [-0.139, +0.011] / -0.038 [-0.105, +0.031] (results.md@adc0df5:469, 678, 880, 1075; deltas.csv@adc0df5:32, 17005, 32398, 40356). IC sopra 0 in 2 famiglie su 4.
  - **+inf** (letta su Ava-256): HIFI3D +0.105 [+0.062, +0.154] / +0.088 [+0.039, +0.138]; FaceScape +0.192 [+0.144, +0.241] / +0.200 [+0.149, +0.252]; FaceVerse e neutra = maschera (il NICP non vi perde righe) (deltas.csv@adc0df5:8490, 8520, 24674, 24701).
- **R2a**: +0.134 [+0.085, +0.188] / +0.117 [+0.066, +0.171]; +0.119 [+0.030, +0.210] / +0.127 [+0.039, +0.213]; +0.091 [-0.010, +0.189] / +0.106 [+0.006, +0.201]; +0.117 [+0.013, +0.219] / +0.147 [+0.050, +0.241] (results.md@adc0df5:473, 682, 884, 1079; deltas.csv@adc0df5:36, 17009, 32402, 40360).
  - **+inf**: HIFI3D +0.134 [+0.085, +0.188] / +0.117 [+0.066, +0.171]; FaceScape +0.116 [+0.028, +0.207] / +0.124 [+0.037, +0.212]; FaceVerse e neutra = maschera (deltas.csv@adc0df5:8494, 8524, 24678, 24705).
- **R2b**: +0.253 [+0.170, +0.342] / +0.235 [+0.149, +0.329]; +0.286 [+0.229, +0.345] / +0.295 [+0.227, +0.357]; +0.102 [+0.015, +0.190] / +0.117 [+0.037, +0.194]; +0.063 [-0.027, +0.152] / +0.093 [+0.003, +0.181] (results.md@adc0df5:471, 680, 882, 1077; deltas.csv@adc0df5:34, 17007, 32400, 40358). Il NICP per coppia fallisce su righe che dipendono dalla topologia: righe nocrop in maschera 98224 / 99000 su HIFI3D e 88724 / 99000 su FaceScape, cioe' il 10,4% perso; up60k 25,9% (results.md@adc0df5:309-311, 318, 334).
  - **+inf**: HIFI3D +0.262 [+0.179, +0.350] / +0.245 [+0.160, +0.334]; FaceScape +0.362 [+0.307, +0.416] / +0.370 [+0.307, +0.427]; FaceVerse e neutra = maschera (deltas.csv@adc0df5:8492, 8522, 24676, 24703).
- **R3a**: +0.285 [+0.187, +0.383]; +0.038 [+0.018, +0.058]; +0.048 [-0.004, +0.102]; +0.035 [-0.018, +0.090] (results.md@adc0df5:150, 157, 164, 171); conteggio 2 / 0 / 2 su 4 (results.md@adc0df5:276).
  - **+inf**: HIFI3D +0.284 [+0.187, +0.382]; FaceScape +0.038 [+0.018, +0.058]; FaceVerse e neutra = maschera (deltas.csv@adc0df5:8670, 24836).
- **R3b**: -0.017 [-0.046, +0.012]; +0.069 [+0.043, +0.095]; +0.078 [+0.022, +0.129]; +0.088 [+0.022, +0.148] (results.md@adc0df5:151, 158, 165, 172); conteggio 3 / 0 / 1 su 4 (results.md@adc0df5:277).
  - **+inf**: HIFI3D -0.009 [-0.036, +0.020]; FaceScape **+0.155 [+0.127, +0.180]** (maschera +0.069); FaceVerse e neutra = maschera (deltas.csv@adc0df5:8671, 24837).
- **R4** (colonna crop; v5: descrittiva, evidenza riportata come motivazione, sez. 6): +0.245 [+0.199, +0.294] / +0.135 [+0.088, +0.184]; +0.062 [-0.002, +0.130] / -0.011 [-0.077, +0.054]; -0.019 [-0.083, +0.050] / -0.037 [-0.102, +0.028]; -0.034 [-0.098, +0.035] / -0.044 [-0.101, +0.015] (results.md@adc0df5:469, 678, 880, 1075; deltas.csv@adc0df5:296, 17245, 32646, 40586). IC sopra 0 in entrambi i semi solo su HIFI3D.
  - **+inf**: HIFI3D +0.246 [+0.200, +0.295] / +0.136 [+0.088, +0.185]; FaceScape +0.070 [+0.003, +0.138] / -0.003 [-0.069, +0.061]; FaceVerse e neutra = maschera (deltas.csv@adc0df5:8754, 8784, 24914, 24941).
- **R5a**: +0.041 [-0.005, +0.091] / +0.033 [-0.027, +0.088]; +0.269 [+0.217, +0.317] / +0.275 [+0.222, +0.328]; -0.020 [-0.093, +0.055] / +0.007 [-0.065, +0.078]; -0.063 [-0.140, +0.017] / -0.025 [-0.096, +0.051] (results.md@adc0df5:187-188, 198-199, 209-210, 219-220); conteggio 1 / 0 / 3 per seme (results.md@adc0df5:292-293).
  - **+inf**: HIFI3D +0.041 [-0.006, +0.091] / +0.032 [-0.026, +0.087]; FaceScape +0.254 [+0.202, +0.302] / +0.261 [+0.207, +0.314]; FaceVerse e neutra = maschera (deltas.csv@adc0df5:8681, 8709, 24847, 24872).
- **R5b**: +0.027 [-0.029, +0.083] / +0.018 [-0.046, +0.082]; +0.349 [+0.291, +0.401] / +0.355 [+0.290, +0.412]; +0.051 [-0.027, +0.124] / +0.078 [-0.005, +0.152]; +0.010 [-0.068, +0.095] / +0.049 [-0.041, +0.133] (results.md@adc0df5:191-192, 202-203, 213-214, 223-224); conteggio 1 / 0 / 3 per seme (results.md@adc0df5:296-297).
  - **+inf**: HIFI3D +0.037 [-0.019, +0.093] / +0.028 [-0.035, +0.092]; FaceScape +0.431 [+0.376, +0.481] / +0.438 [+0.379, +0.491]; FaceVerse e neutra = maschera (deltas.csv@adc0df5:8683, 8711, 24849, 24874).
- **R6** (D5, EQ, d_P con SR, media dei 4 bersagli sintetici, k = 50): +0.090 [+0.069, +0.109] / +0.066 [+0.043, +0.091] (`aau/runs/evidence/diagnostics/results_d5.md:26, 36`). k = 50 e' il primo k della griglia con F in entrambi i semi (results_d5.md:171-174); per bersaglio non sempre: FaceVerse neutra s2345 +0.018 [-0.050, +0.088] (results_d5.md:34).
  - **+inf**: nessun metodo per coppia nella regola, la lettura +inf coincide con la maschera.
- **R7** (riportata come NoW originale - NoW metrical, cioe' -delta; landmark oracolo, matrice completa,
  `aau/runs/evidence/protocolli/results.md`, commit `c960859`): -0.281 [-0.371, -0.194]; -0.053 [-0.092, -0.018];
  -0.038 [-0.078, +0.000]; -0.021 [-0.060, +0.019] (results.md@c960859:93, 106, 119, 132). IC sotto 0 in 2 famiglie su 4, la
  neutra non risolta. Col rilevatore (sottocampione 8%, `rumore_landmark/results.md`, commit `f47a375`) gli Spearman
  restano quasi uguali: HIFI3D 0.413 contro 0.685, FaceScape 0.616 contro 0.678 (rumore_landmark/results.md@f47a375:24-25,
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
| Composizioni delle baseline con k | unica | FR | `aau/baselines_mm` (`6c8a747`), `factorized_protocol_emendamento_5.md` | **non calcolata su Ava-256** (v5): nessun codice nella catena congelata (`conf_desc.py` non la calcola) |
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
| ULRSSM (v6: entra) | `ulrssm/faces_xyz` (`true`, `selection.json` sha256 `94f6f065...`); `faces_wks`, `zeroshot_faust` (`false`: non si calcolano) | **dichiarate su Ava-256 (v7)**: per coppia con GT FR (`pair_fr`) e via template con GT SR (`tpl_sr`); `pair_sr`, `tpl_fr` escono dallo stesso calcolo con `dichiarata` = False (spec `desc.ulrssm.readings_declared`; `fm_plugin.json` di Ava-256 con `declared` per lettura, scritto da `conf_desc.py ulr-merge`) | `aau/competitors/ulrssm/` (`ulr_common.py`, `ulr_eval.py`; sha256 nella sez. 11.1), clone esterno `~/ext/ulrssm` al commit `0625cbcb`, checkpoint `aau/runs/competitors_ulrssm/faces_xyz/final.pth`; catena `conf_desc.py` | fissato (v6) |
| Hybrid FM | - | - | nessun codice nel repo | **escluso** (v6): non pronto al congelamento (regola della sez. 14) |
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
  ArcFace non si calcola e si riporta come non applicabile (`na.json`). Nodo `i256-a10-09` (quello di G0 e del collaudo;
  v7: controllato nel codice, `desc.arcface.node`).
- **GW e percettivi**: `gp_gw` (`run_prep`, `run_view`, `run_merge`) nella configurazione dichiarata `euclid|128|cgw2`
  sulle 597.000 coppie di all_cross; per i percettivi SOLO le quattro varianti dichiarate (`lpips/depth`, `dinov2/depth`,
  `clip/smooth`, `dreamsim/smooth`): si rendono solo le colorazioni `depth` e `smooth`, le altre varianti non si
  calcolano (regola della variante dichiarata, sopra). Vista di `gp_common` da uno spec generato (L_d e cs_ref di
  `blmm_scalars.py` sui 56); le scelte del pilota (`competitors_new/{gw,percettivi}/pilot_choice.json`) si controllano
  per sha256 contro lo spec e contro le varianti dichiarate.
- **ULRSSM** (v6): `conf_desc.py ulr-run[:k/n]` su una L40S (`--gres=gpu:l40s:1`, la classe della valutazione di
  sviluppo; controllato) e `ulr-merge`, con le funzioni di `ulr_eval.py` invariate (`run`: mesh preparate come in
  training, feature del checkpoint, mappa per coppia sulle coppie di soggetti diversi, stessa topologia compresa, e
  template GNM -> mesh; `merge`: variante via template con `train_fr_sr.factor`). Prima del calcolo: sha256 di
  checkpoint e `selection.json` contro lo spec, variante = quella della selezione, clone `~/ext/ulrssm` al commit
  `0625cbcbb7cd007893dd665b21ef55b275563472` e senza modifiche (`git status`). Coppie su Ava-256: 716.400 (1.200 mesh,
  soggetti diversi), ripartibili; nel collaudo di sviluppo (HIFI3D, 178.200 coppie) 36 minuti su una L40S.
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
  (HIFI3D) sui 7 di NoW, 5,7-6,5 mm su `noisy` (rumore_landmark/results.md@f47a375:13, 18, 203-204). NoW originale e metrical
  ne dipendono poco (<= 0,031 di Spearman anche a sigma 5), REALY e Stirling molto (col rilevatore REALY -0,06 / -0,20,
  Stirling -0,05 / -0,12; rumore_landmark/results.md@f47a375:214-216).

### 9.2 NoW metrical contro la nostra d_F (v2, decisione del PI): solo descrittivo

- Delta = NoW metrical - d_F (positivo = NoW metrical davanti), GT FR, per d_F cal. e d_F non cal., per seme, nei
  gruppi `nocrop_cross`, `crop` e `all_cross`, sulle righe e repliche della sez. 6.
- Si riportano **sempre** il delta puntuale e l'**IC 95% bilaterale** (percentile, 2,5-97,5), **qualunque sia il segno**.
  Nessun test, nessuna famiglia, nessuna lettura confermativa.
- **Limite atteso** (sviluppo, landmark del rilevatore, d_F non cal. s1234, sottocampione 8%,
  rumore_landmark/results.md@f47a375:120, 124, 132):
  - senza crop **pari**: HIFI3D +0.041 [-0.011, +0.094] (con landmark esatti sulla matrice completa +0.055 [+0.009,
    +0.105], protocolli/results.md@c960859:100);
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
- **Nodo**: il fit gira su i256-a10-10, il nodo di D3 e2 e D5 (v7: `conf_head.py fit` si ferma su un altro nodo,
  `conf_eval.py` rifiuta una testa con un altro `host`; con `freeze` una testa gia' scritta non si sovrascrive). Con lambda = 1e-6 BLAS diverse cambiano i fit
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

9. **Codice congelato (v7)**: `CONF_FROZEN_COMMIT` obbligatorio con `freeze` (controllato PRIMA di ogni altra verifica e
   di qualunque lettura); HEAD del repo da cui si gira = quel commit; `git status --porcelain --untracked-files=no`
   vuoto. In piu', per ogni passo, `conf_run.py`: ogni modulo del repo importato (log di un finder in `sys.meta_path`,
   ereditato dai figli) tracciato e uguale al suo blob nel commit; gli sha256 dei moduli in `verify.json`,
   `<evidenza>/modules/*.json`, `conf/controls.json` e `conf/modules.json` (accanto a `rules.csv`), `desc/controls.json`,
   `embed/*/*.json`, `head/*.json`.

10. **Cloni esterni (v8)**: `diffusion-net` (link nel worktree, importato da `PYTHONPATH`, `aau/env.sh:25`) al commit
    `b1019b049597711d10259ce28250f7dfc4335a2b` con albero pulito (`conf_common.external_clones`, in ogni `verify`);
    ULRSSM (`~/ext/ulrssm`, `0625cbcb`) e' controllato da `conf_desc.py` prima del calcolo (sez. 9). Ogni modulo che
    sta sotto il repo o il worktree ma risolve fuori e non e' in un clone elencato ferma il passo (`conf_run.py`).

Le verifiche 1-10 sono `conf_common.verify` (con `freeze` nello spec), primo passo di ogni job di `conf.sbatch` e prima
riga di `conf_eval.py` e `conf_head.py`; le versioni del punto 6 sono `conf_common.versions` (venv di MICA).

### 11.1 Codice della catena (v3)

Commit del codice e degli esiti del collaudo (sez. 11.2): **`adaac1f`** (v3), **`fe32e8f`** (v4: embedding e
descrittivi), **`9675df8`** (v5: embedding su A100 con le tolleranze v5; R4 fuori dalla famiglia in `conf_eval.py`) e
**`2619dff`** (v6: ULRSSM; collaudi rifatti con `fm_eval.py` della matrice rigenerata, `46fe7a4`) e **`2cea824`** (v7:
codice congelato a runtime, nodi, ULRSSM; collaudi della sez. 11.2 rifatti da un worktree a quel commit) e
**`8a6e184`** (v8: clone diffusion-net, nodo del fit B nelle uscite, ripresa della testa; tracciamento e collaudi rifatti
da un worktree a quel commit).

**v8.** Il tracciamento rifatto al commit `8a6e184` (job 1074574, 1074990, 1074991, 1074992) registra 131 moduli: i
127 della v7 piu' 4 file del clone diffusion-net (`diffusion-net/src/diffusion_net/{__init__,geometry,layers,utils}.py`,
commit `b1019b0`), tutti uguali al commit o al clone, nessuno scarto. v9: tutti gli sha della tabella sono quelli del
commit congelato `8a6e184` (i commit successivi di altri lavori, es. `771fee8` su `v3_work/mm/model.py` e
`v3_work/stream/sources.py`, non entrano: su Ava-256 si gira dal worktree a `8a6e184` e `conf_run.py` confronta ogni
modulo importato con quel commit).

**v7: elenco ricavato dalle importazioni effettive.** I moduli sono quelli registrati da `conf_run.py` in un collaudo
completo della catena (spec `aau/ava256/spec/hifi3d_trace.json`: tutti i passi di Ava-256 dallo stesso codice su 3
soggetti di HIFI3D, dal worktree al commit `2cea824`, job 1073353-1073355; unione in
`collaudo/hifi3d_trace/modules/*.json`): 127 moduli del repo, tutti uguali al commit (nessuno scarto). Si aggiungono i
file che non sono moduli importati o che il tracciamento non attraversa: `conf.sbatch`, gli spec, `conf_worktree.sh`,
`conf_collaudo.py`, `ava_freeze.py` e `ava_calib.py` (solo con `freeze`, cioe' su Ava-256) e `zs_embed.py` (eseguito con
`runpy` da `eval_v3.py`). `pr_eval.py` non e' importato dalla catena (tracciamento) e non e' in tabella. Su Ava-256
`conf_run.py` confronta di nuovo ogni modulo importato col commit congelato, quindi un modulo fuori da questa tabella
ma tracciato al commit non passerebbe inosservato: finirebbe in `<evidenza>/modules/`. sha256 dei byte di ogni script e degli spec (`conf_common.py sha`, `conf_common.SCRIPTS`, v4: due spec di
collaudo compresi); lo strumento non si esegue su Ava-256 con file diversi da questi (regola sui bug, sez. 12). Nella
v4 sono cambiati `conf_common.py` (sha256 di C3M in `verify`, lista degli script), `conf_collaudo.py` (passi `embed`,
`desc`, `desc-eval`), `conf.sbatch` (passi `embed-stage`, `embed`, `desc-*`) e i due spec `ava256.json` e
`hifi3d_full.json` (chiavi `embed`, `desc`, `columns.desc_models`, `columns.desc_plugins`). Nella v5 sono cambiati
`conf_eval.py` (R4 in `DESC_RULES`, m = 9), `conf_embed.py` (controllo `embed.gpu`), `conf_collaudo.py` (tolleranze v5),
`conf.sbatch` e i due spec (`embed.gpu` = `A100`). Nella v6: `conf_desc.py` (passi ULRSSM), `conf_common.py` (script di
ULRSSM nella lista), `conf_collaudo.py` (passo `ulr`, tolleranza `ulr_rel`), `conf.sbatch`, i due spec (`desc.ulrssm`);
`aau/full_matrix/fm_eval.py` e' stato aggiornato da un altro lavoro (matrice rigenerata con tutti i plug-in, `46fe7a4`):
le funzioni importate dalla catena (`groups_of`, `records`, `plugin_columns`, `replicas`, `spectral_columns`,
`geo_columns` e le minori) hanno AST identico a `9675df8`; cambiano costanti del modulo (es. `LADDER`, passo 1b sui
plug-in dei protocolli coi landmark esatti, che la catena non ha), e tutti i collaudi sono stati rifatti con questa
versione (sez. 11.2):

| File | sha256 | Origine; stato |
|---|---|---|
| `aau/ava256/conf_common.py` | `c2b5c2e4f58470543708f2b14fc34a65303df9b9a4ae739f77ee46a67d97e13d` | importato (tracciamento); modificato (v8) |
| `aau/ava256/conf_desc.py` | `5e310928d277cdeff484a7f7a8b704b538adb660711c89295b2bf05efc2a51ef` | importato (tracciamento); modificato (v8) |
| `aau/ava256/conf_embed.py` | `3173402793193dd244ddca0771b5baeae8788e64a15ab1095d50807898f922e2` | importato (tracciamento); invariato |
| `aau/ava256/conf_eval.py` | `fd5be2e4d6d943b51b64188391ffdb821e9571c8313b22dfcb8aee26aba4e3f1` | importato (tracciamento); invariato |
| `aau/ava256/conf_head.py` | `4937a7e3ae44d5eabf7e438d7558c22d3218f05467a0e11a4277509e667facfb` | importato (tracciamento); modificato (v8) |
| `aau/ava256/conf_protocols.py` | `7f4db409cdf1f8f7486f1ac7de12908fa24c54ab25529b93839ca91e8c257e79` | importato (tracciamento); invariato |
| `aau/ava256/conf_run.py` | `53d416b7c9a328a71a6efa25748f233f6a46fbbc8a1db4181f8176f7d707184c` | importato (tracciamento); modificato (v8) |
| `aau/baselines/arcface_fixed.py` | `834e8e21ac8e213eff70695875e7f7e12bfdb08d013a3b592a935670f23d21a3` | importato (tracciamento); invariato |
| `aau/baselines/common.py` | `725ebe31c86746ade6921547cd4536f823c3f9ecea54828f9cba94c9d4dc8c78` | importato (tracciamento); invariato |
| `aau/baselines/geometric_kernel.py` | `8b1c81fe831d1a9fca8af5f96adf5f0f2ebebca629e7c3bedc511a95104550bd` | importato (tracciamento); invariato |
| `aau/baselines/geometric_matrix.py` | `6fffb91758430e518246562ff7016746eb5a09168e43749d1047d704e1ea4955` | importato (tracciamento); invariato |
| `aau/baselines_mm/blmm.py` | `9c893b2cb3ca150d92cd21844dfeafae105c204ec559e635fb5c8e2800949446` | importato (tracciamento); invariato |
| `aau/baselines_mm/blmm_eval.py` | `7d439122deadd5c1502492c15ca463681984a2ca24fb8e2863d372c7fd12cda8` | importato (tracciamento); invariato |
| `aau/baselines_mm/blmm_pairs.py` | `34bde06d2b454090891700591d0b78a0ea1960bdff67a14c423d06db90c26788` | importato (tracciamento); invariato |
| `aau/baselines_mm/blmm_scalars.py` | `b34b8cfbf3bdf562bf698f274d5a8ebd2ffcb786d7bc7e088fb59536f41ae37d` | importato (tracciamento); invariato |
| `aau/baselines_mm/blmm_template.py` | `870a422674597ddb2a9adebceed13dd0fb791728ef94a806ba3a385aaed19496` | importato (tracciamento); invariato |
| `aau/baselines_param/bp.py` | `0f8dd0571593c512a002463736a1e6cca244395cc78d7f6694793f13d572177c` | importato (tracciamento); modificato (v8) |
| `aau/baselines_param/bp_e2.py` | `3d4ce9d209e95637e9d4c8368bc76da551fb64f81fc1a7715a7070c80a2c67e8` | importato (tracciamento); modificato (v8) |
| `aau/baselines_param/bp_fit.py` | `33f99c2582f88f8215ca9ee654109410a3cc2f91c245f72e0f5c9fb17af95588` | importato (tracciamento); modificato (v8) |
| `aau/baselines_param/bp_fit_e1.py` | `d0cf991687cafeb65e6e8fc629120608bdb97b6cd0de93529007345945db7755` | importato (tracciamento); modificato (v8) |
| `aau/baselines_param/bp_paired.py` | `a0bffb67b4de77a7e17933c9ab7a1dd6ec79c3943944998a0f31b4c0fbcb4432` | importato (tracciamento); invariato |
| `aau/baselines_param/bp_paired_e2.py` | `be9c190fa0c22ff7aa09fab8fb5c1c1be9ba4dda17f9cc370e59742579d2b8b4` | importato (tracciamento); invariato |
| `aau/competitors/comp_spectral.py` | `a485eebe9081c295ce4d91cf5e2300a287c0aa99741170188d8d6aaba81a68b1` | importato (tracciamento); invariato |
| `aau/competitors/comp_summarize.py` | `3296bfe723d697e0aa996e9e99187e461eceb741ff866aec27717f22ee5d3e4a` | importato (tracciamento); invariato |
| `aau/competitors/gw_percettivi/gp_common.py` | `3f58957396ec135dce1c85aca0d5954edf8814d4e3ad5a9c2d6f5a83630f018a` | importato (tracciamento); invariato |
| `aau/competitors/gw_percettivi/gp_gw.py` | `d64f04401d25f176dc927894f960e81e936b523c0da581813e0fd90dce2895c4` | importato (tracciamento); invariato |
| `aau/competitors/gw_percettivi/gp_percettivi.py` | `dad7542ae75a3babffc6f8166ae59dd4834394123ba90f8abb90d0b3a4933b24` | importato (tracciamento); invariato |
| `aau/competitors/protocolli/pr_common.py` | `deeb13ed5e755ff3f591518f26e08577d3172b873744504711ba858120df84ab` | importato (tracciamento); invariato |
| `aau/competitors/protocolli/pr_landmarks.py` | `da5096451e981325c9c7049d09098b5f21523e6117f33a5e46a7491d84b62301` | importato (tracciamento); invariato |
| `aau/competitors/protocolli/pr_metrics.py` | `75293276c3a07f7b70b324859527a32f19f43623e64f7d1fd68ab6eb59981eda` | importato (tracciamento); invariato |
| `aau/competitors/protocolli/pr_noise.py` | `6f16ac05c8cc71298508988062de5ddac85d14ad2d3caad2d127247010fac4c0` | importato (tracciamento); invariato |
| `aau/competitors/protocolli/pr_noise_detect.py` | `063ca84b315c44925fa2357ead410a10be2e199702f44427f31968d6585bebb9` | importato (tracciamento); invariato |
| `aau/competitors/protocolli/pr_run.py` | `dffbae45ada9f2a26628f4e0b4761ea6563fa3b26c9a3ad485993f9530377238` | importato (tracciamento); invariato |
| `aau/competitors/ulrssm/ulr_common.py` | `16ac87192ef08ad8a37939b4140acc8338db0385d9bc963a7239240cb476295b` | importato (tracciamento); invariato |
| `aau/competitors/ulrssm/ulr_eval.py` | `7544cca6ab53b8d1b967b03a110bb2813fa6352e58b6d8da470750e48b18e499` | importato (tracciamento); invariato |
| `aau/diagnostics/d3_e2.py` | `42b962a432fa9f219cbe754f44c41da11a3b2a3143cb4a5262ff68a4d9c23542` | importato (tracciamento); invariato |
| `aau/diagnostics/d3_head.py` | `a3161d56513fc5dba6b4cd10e8884ac1f5788f6eb6884bf020b0d7707410754d` | importato (tracciamento); invariato |
| `aau/diagnostics/d3_pools.py` | `9dc12cb1297e0bc2dabd5cf82fe1f9725853f41e33a8f10acbd595f55337303b` | importato (tracciamento); invariato |
| `aau/diagnostics/d3_stats.py` | `573aef82d60feb7722dafa031c2ea268bf64f35dd20efe5b2124d8e322a6fcc9` | importato (tracciamento); invariato |
| `aau/diagnostics/diag.py` | `399882d0e81b4f1e7458bb0e8bd77777cf42ad8b8847b716d667a2a11f43af4d` | importato (tracciamento); invariato |
| `aau/diagnostics/g0_teacher.py` | `d50e76ca6452cd38b92d9a0f547da2e7edcb9a678900d02bfb42b5b029144832` | importato (tracciamento); invariato |
| `aau/distill/teacher_labels.py` | `6b338d5de7d50987ddd08ab5ad3c8a91df0ba6957be64d4c3506f11a8499844d` | importato (tracciamento); invariato |
| `aau/evidence/e1_factorial/e1_calib_ugt.py` | `cc53db872a1ecc47f751e58677fd489d5684525bbe9598dbf6c19e11cd17aa72` | importato (tracciamento); invariato |
| `aau/famos/famos_common.py` | `6c7b95c2ed19e4f9259f828b120a7fadfa8017727b66d4525505531ca3c88430` | importato (tracciamento); invariato |
| `aau/full_matrix/fm_compute.py` | `b0dad61c47ceb9328df018a77893410b9dbc0e5f4e24e87fb167f959f441c0af` | importato (tracciamento); invariato |
| `aau/full_matrix/fm_eval.py` | `9101b3f9585dc337fd3d67cc5a8bb08146b4d32bd0409c50f52f2ccd21981770` | importato (tracciamento); invariato |
| `aau/ict/ict_summarize.py` | `9d1d7cf783361e9568ddc621a698b239f3ecffc093ccc52d2b9a848d50ba7776` | importato (tracciamento); invariato |
| `aau/indomain/ir_simicp.py` | `c853ffbdec5469cff5a677facce6c1a443cda00e1906bb82c556932a85197ffa` | importato (tracciamento); invariato |
| `aau/indomain/ir_template.py` | `4f9060820f9f4da48ac1bbbeacb5d3d64bd8e6ee06201fca6de561a76f110416` | importato (tracciamento); invariato |
| `aau/multiface/ws3a_common.py` | `f85f284010088abf803cf408eaee611a9a5ba3d5e1c8fb3b8958b22b4c733568` | importato (tracciamento); invariato |
| `aau/multiface/ws3a_perceptual.py` | `99632c31a6b80f654a93c5ce329da069cde6ce36c42e3eabf5904e054e53767f` | importato (tracciamento); invariato |
| `aau/multiface/ws3a_render.py` | `ea499a94184913b8d7d76648b6f9d4b0e870e56e1eff8907e520834a96b338b6` | importato (tracciamento); invariato |
| `aau/zs3dmm/dev_fs_summarize.py` | `3fbd8e9f419180ced32fcf1762a88e4d5c7829e4e71fff7910c4220bbb25fef1` | importato (tracciamento); invariato |
| `aau/zs3dmm/gnm_model.py` | `3e678f7a4c1cb80f85c76bb8d6444b4e5da10b541deeafa11f32cd7e685b187b` | importato (tracciamento); invariato |
| `aau/zs3dmm/hifi_model.py` | `a6e71c53d65eee56d5941c75b7f62a71b287fe04ac28229f9629fa38fe5a5802` | importato (tracciamento); invariato |
| `aau/zs3dmm/zs_arcface_render.py` | `9c59232044085fd4f8e2fae3ee71cdee2ab6f0b335781bd81409348bf90b9857` | importato (tracciamento); invariato |
| `aau/zs3dmm/zs_arcface_summarize.py` | `6d1b90749696c660a94306e10d31b8c4159cfd2edf1668a31e30041c5e60a191` | importato (tracciamento); invariato |
| `aau/zs3dmm/zs_arcface_vs_scale.py` | `0a9a680b6d348ef89b5cd57b64668e5316035123ced6662ec7350fa364b8bf78` | importato (tracciamento); invariato |
| `aau/zs3dmm/zs_expr_summarize.py` | `b543d01b4c8c4c720459cf94b45c65ef13f929884e167cb5974ada4a445325ac` | importato (tracciamento); invariato |
| `aau/zs3dmm/zs_stage.py` | `6cac25ee537ee9169e5b64fc03c1ee8795d616604d6226b56c8d53524ecfc21c` | importato (tracciamento); invariato |
| `aau/zs3dmm/zs_summarize.py` | `f19456ad9abea0ba2095cd43a14c341bb9db67e5fd33931ddf060bff344517cd` | importato (tracciamento); invariato |
| `diffusion-net/src/diffusion_net/__init__.py` | `5849f836a4a95db7a36d432fbea8c35203521249d77af134f8792ff674ac2be4` | clone diffusion-net al commit b1019b0 (tracciamento); nuovo nella tabella |
| `diffusion-net/src/diffusion_net/geometry.py` | `2663a9665e6d57f49fd56b6ae8075e43b7c5aa74c3b825ac7f28f114c71cf090` | clone diffusion-net al commit b1019b0 (tracciamento); nuovo nella tabella |
| `diffusion-net/src/diffusion_net/layers.py` | `e593676dc7732ec9aca5adb5a1f1fed5301a901c3ef0e498e74333118ba79f17` | clone diffusion-net al commit b1019b0 (tracciamento); nuovo nella tabella |
| `diffusion-net/src/diffusion_net/utils.py` | `4d22ce294f67f6dac71aaaa3c3a8da0093728903f71ee0bc416dfa47e43d1099` | clone diffusion-net al commit b1019b0 (tracciamento); nuovo nella tabella |
| `faceBench/facebench/__init__.py` | `cde107bb77aebf02d40f890d76d6b096e0b33392c89e8bd8ac1ff639595669e8` | importato (tracciamento); invariato |
| `faceBench/facebench/config.py` | `520fd812731a9ba49379ba5d4d67b4e369e506f35811cfb939fa6fdd1ec0f592` | importato (tracciamento); invariato |
| `faceBench/facebench/corrector/topology_consistency.py` | `d84476f2764659b5827568cc6f4e746c02b946fc8006c04799027ffeea27c976` | importato (tracciamento); invariato |
| `faceBench/facebench/corrector/topology_utils.py` | `5217023cd53f7d2b04422a5c490d2f16661ebf69b3302f755ee89c3612bf2fd7` | importato (tracciamento); invariato |
| `faceBench/facebench/correspondences/chamfer.py` | `2cbe0ab0a70e8f497941e1fd52af8ce254a655c025f37aed1f27bdc860abbea1` | importato (tracciamento); invariato |
| `faceBench/facebench/correspondences/identity.py` | `67187265334856b131a31b1555836880fef4cf56cb3dcfc0bf0970901433591c` | importato (tracciamento); invariato |
| `faceBench/facebench/distances/landmarks.py` | `d3333cb9002940d7e6377302746a71554539371fb9713b3803031c9c897e344a` | importato (tracciamento); invariato |
| `faceBench/facebench/distances/p2p.py` | `d8717476b9658d39ffd87a7b905701b754196c490a2b2de8696927517d7753da` | importato (tracciamento); invariato |
| `faceBench/facebench/distances/p2tri.py` | `9ee6da144e1b9914c6aae39410720c87c3dce2d5a448ebc62d998fec57ccb9b2` | importato (tracciamento); invariato |
| `faceBench/facebench/mesh_croppers/point_based_cropper.py` | `c0d2cfeb768e5116c7d8ac1a85adc5a2eaeb47788d5844d3de3d725305e584d0` | importato (tracciamento); invariato |
| `faceBench/facebench/nonrigid_aligners/landmark_elastic.py` | `e38207afbbba21ae9676381eccf58eaa7d3cfc46f2cd93a258ce716340a704bc` | importato (tracciamento); invariato |
| `faceBench/facebench/nonrigid_aligners/nonrigid_icp.py` | `1ee2d391a9da17970dde2d48373ebdd2d63c61454db2379ff6e33ea463351682` | importato (tracciamento); invariato |
| `faceBench/facebench/rigid_aligners/icp.py` | `1b0a429633d496a11d444a45bb243ac9e6c077d37f35694c19d37ad9bc242454` | importato (tracciamento); invariato |
| `faceBench/facebench/rigid_aligners/landmark_based.py` | `fd2a9141415489cfcd58e7cbd7dc5a7cfc129e8905324a3e482914574f81ae25` | importato (tracciamento); invariato |
| `faceBench/facebench/utils/pipeline.py` | `985212baaed7763c51fcf261b16e95d565a26e9b2eeea8e802c024e1287b4bfb` | importato (tracciamento); invariato |
| `faceBench/facebench/utils/utils.py` | `a981a9e871cd671a530abfda330888c9567535b1d3cd5f8b3c09d919b7b4c373` | importato (tracciamento); invariato |
| `faceBench/facebench/utils/warnings.py` | `6deeb01f2c9d6de66fdfc5e84e6c73a78105bba0d2bf1a22fc3911279237edd5` | importato (tracciamento); invariato |
| `faceBench/facebench/visualization/correlation.py` | `e8c24f75ef28418e6a1183f51962eda4f964b8b4632f8c5202c6610de04f4544` | importato (tracciamento); invariato |
| `faceBench/facebench/visualization/heatmap_plain.py` | `9101f316176f9a42125162a5c988941f2cc9558e902f9f9695ae2e8829a39acc` | importato (tracciamento); invariato |
| `faceBench/facebench/visualization/results_table.py` | `79a253b07e7688356c947c670dce8e21aa94074fc30ec8dc9ac1c2809fb59a0e` | importato (tracciamento); invariato |
| `faceBench/facebench/visualization/space_heatmap.py` | `f1c10d95d3d3cb33aad8ebb5d96e8f751d27c2295e9b7729645436ad3b5cc398` | importato (tracciamento); invariato |
| `faceBench/latentVSpipeline/mesh_npz_utils.py` | `abaf9cba9a129d4d4526e98c5a4b7b67dfe4479a73dd079bf07826e9cd4063d8` | importato (tracciamento); invariato |
| `faceBench/latentVSpipeline/run_facebench_remesh.py` | `c9bca1ae48703b12892064167e0103817b98edf4752bb934b46175626848eaca` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/autoencoder/dataset_gtready.py` | `fd271dcf8a649f5a2c99c88bf9da50ebab443b8028d4ebb6c7ac258622480638` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/autoencoder/diffusion_autoencoder.py` | `7dfb5dce2b75634b2b0a8c7eacfd2069b24de1d079757145f3d12269bb938ad4` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/autoencoder/latent_loss.py` | `78568fd16956449fb0839b564b2d55ab82b934de24b8689f5d336fbeed67dc71` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/autoencoder/path_setup.py` | `7c67b6b36edc3e5e3ee8d8e102a59b580e2b38d9f704aa5f397e783117677811` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/remeshing/intrinsic/intrinsic_utils.py` | `c1b9254dbd84cc40dac6f141fceb62a7a56644b85bb79d0da7c34a014c630c34` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/remeshing/intrinsic/perturbated/compare_model_vs_chamfer_rankings.py` | `51a87d6e78c3ea63eb69d35b3e547558b17d1bf87c3ec377df6bffc045a9fcc3` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/remeshing/intrinsic/perturbated/compare_model_vs_chamfer_topology_breakdown.py` | `71047c56f613bd1257b60758f0f81f32ac40b4fabd297adc97394f3a8c8fab75` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/remeshing/intrinsic/robustness/__init__.py` | `9c7ced11ac713b9a9642ae563e3ad74d4e697c82f241b7881e0fd055b393cad5` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/remeshing/intrinsic/robustness/data_utils.py` | `02fb657182568536d834ee7fef2cea862aca38582debc00d5200a14063d56b61` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/remeshing/intrinsic/robustness/eval_utils.py` | `a26133f7160934be1e8e2a65dfd9b255912632b2f9a107dd9983298e2eaeb4db` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/remeshing/intrinsic/robustness/model_helpers.py` | `497e9c6796e27259c1921ab6ed018a2dba09a4f29bddfe9ed31100db31943c53` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/remeshing/intrinsic/robustness/noise.py` | `7a788155cfd6de5855a7c65d8becec3d61b5eddc19d9872162e25e5d0b70e646` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/remeshing/intrinsic/robustness/paths.py` | `a84e2757c2e91a60cea795e70fbebe445e78accc1c13c809067bf17b0d78da7a` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/remeshing/intrinsic/robustness/posthoc_runner.py` | `31bb97966774610d79e5b829681bcc1f87aa098ae1f33a35d408a05c07a96c31` | importato (tracciamento); invariato |
| `face_embedding/gt_encdec/remeshing/intrinsic/robustness/train_runner.py` | `7623c0b35a65279e99cfcad04a2e6ffd5969a789afaeb0bd4e00364e644de6a8` | importato (tracciamento); invariato |
| `v2_work/genflame/flame_model.py` | `226e5d98a00884aa4f47ed930a76f6524d32c2b13dcaef96a1afdfe22de35486` | importato (tracciamento); invariato |
| `v2_work/genict/mesh_ops.py` | `c5efce43e2c6c2993d946ad18606ee4ecdb590697f4315bbbb91e6881d02c572` | importato (tracciamento); invariato |
| `v2_work/phase0/measure_distances.py` | `8f448fa335a15e32cd157034bb14232985d966053d8caced6c967167ee5c994f` | importato (tracciamento); invariato |
| `v2_work/phase0/perceptual_embed.py` | `d91b048631b4c77a919e1471842b0d6dc5db77471e4b74f5f5211e592b36dd4e` | importato (tracciamento); invariato |
| `v2_work/phase0/render_mesh.py` | `be1c643a67be190193d3453a610bba4375097aa86f7c68ebbc50202d34272fce` | importato (tracciamento); invariato |
| `v2_work/potential/areanorm_operators.py` | `511d0a52758b91c4948cfbdb7c53ad68854b4e1cec5351f23c779ee15c96008f` | importato (tracciamento); invariato |
| `v2_work/potential/potential_operators.py` | `caddf7bc244e3f6b2c28145017e8ed2f7e82e351e55372d62834400a92110c87` | importato (tracciamento); invariato |
| `v3_work/canonical_gt/cgt.py` | `2739f59aaf7954055efcbb39df104c9c4f99ca89af898a63026a7aeec5d3b20e` | importato (tracciamento); invariato |
| `v3_work/canonical_gt/methods.py` | `dd9fc773ed4ccd4bf1344995ebb12e3351bb3a38552f94cad7ce468692dd91ff` | importato (tracciamento); invariato |
| `v3_work/canonical_gt/train_fr_sr.py` | `7b488063e3f07bedc3678ccf86f8d36d8d186a29cffb89cd084e4e476531c811` | importato (tracciamento); invariato |
| `v3_work/mm/__init__.py` | `cd66ae62e1baeaa19a4efef44a93b4daa20301de673409562fc6edd5c05c5bd3` | importato (tracciamento); invariato |
| `v3_work/mm/loaders.py` | `2c25c0b97573bb7bb9c7773a4652cb8f0c71d793ce89d66a37c4c599a3d330cd` | importato (tracciamento); invariato |
| `v3_work/mm/model.py` | `b0bb23301914ae8d1c33126d76c801b868e4d7c831489a902175fc0e5cd0c157` | importato (tracciamento); sha al commit congelato `8a6e184` (v9) |
| `v3_work/stream/sources.py` | `9fe85112f7ded28d64d12bfe773a67943b43d4b9288bccdf2fcd955ef809290c` | importato (tracciamento); sha al commit congelato `8a6e184` (v9) |
| `v3_work/trainer/area_v3.py` | `8d50f9b98ba2013f6261a169c6e3e09d2fb399baec59380bb569b69421b9f63e` | importato (tracciamento); invariato |
| `v3_work/trainer/common.py` | `03b9e5c5dc8a1b47eb883f86b812772314d3443f6f22cc6a5eee301b91ed3d42` | importato (tracciamento); invariato |
| `v3_work/trainer/data_v3.py` | `2953a2fc34ee6299dd0b21b58ba95cd63be7535c74b174e96f939943707e7ca0` | importato (tracciamento); invariato |
| `v3_work/trainer/eval_v3.py` | `cf9f9d240bc0105ac94b29209037f5a626c537ed462d0ce3a76a448ac04a6613` | importato (tracciamento); invariato |
| `v3_work/trainer/factorized_v3.py` | `c072057606889d744e171264c6c722ba7f1fdf61bc633bc938987061201776cd` | importato (tracciamento); invariato |
| `v3_work/trainer/global_v3.py` | `408ac38d549784469c463215c27ed6253534f2b09e402fbbc50873f7306edfdb` | importato (tracciamento); invariato |
| `v3_work/trainer/losses_v3.py` | `6598e5e5098b9d1a24d1bb143e1fcaa2008e1b778b952ae60a0341470f14f387` | importato (tracciamento); invariato |
| `v3_work/trainer/model_v3.py` | `468024b48f58a8df14612d1d3a6be0871e39135ad693cc2e9c3dd53c02e30248` | importato (tracciamento); invariato |
| `v3_work/trainer/tools/fact_paired.py` | `cb37bf32916237077069157d8c3ac79c2838b6a85fcbebdb639b8419b2fe709a` | importato (tracciamento); invariato |
| `v3_work/unified_gt/domains.py` | `1447b777f48d6df428ddc031336628e7a377fe40399c571325ebb7d8b54b7383` | importato (tracciamento); invariato |
| `v3_work/unified_gt/eval_methods.py` | `9d27233ec18a0dca758429105d4092d6d543e0f3d6c7579420a9c1272214432e` | importato (tracciamento); invariato |
| `v3_work/unified_gt/shapes.py` | `3db8dca67eab92510c5a197a9db0a938b6d955effc56df87f5a4c44e11f63aa4` | importato (tracciamento); invariato |
| `v3_work/unified_gt/train_gt.py` | `c8290adcc639fdf5596f1476f1ce30c65576b807b86906c65b78a2db0d153850` | importato (tracciamento); invariato |
| `v3_work/unified_gt/ugt.py` | `2cfbad40a815c8fc81ce4b46b4a13686d0e7f0024e1a86fe335431f946bbb340` | importato (tracciamento); invariato |
| `aau/ava256/ava_calib.py` | `a56bf34b4eaeb37f899bbbfa4e4fd2680af82be5522d92deb66da9dbfa215897` | lista fissa (non modulo o fuori dal tracciamento); invariato |
| `aau/ava256/ava_freeze.py` | `520c002858619f2be5a2bb735c33a4663555230fc87e7b1ac34bc0c9c2c590ca` | lista fissa (non modulo o fuori dal tracciamento); invariato |
| `aau/ava256/conf.sbatch` | `1edfcbc8dd6d35dc2ce78687467f8930c6a563ac9ac683e4fc7e8ef4858c2607` | lista fissa (non modulo o fuori dal tracciamento); invariato |
| `aau/ava256/conf_collaudo.py` | `7200469a37ef90a67022b2a18d2840a876471a201f9974ef305e8ac4c71014fe` | lista fissa (non modulo o fuori dal tracciamento); modificato (v8) |
| `aau/ava256/conf_worktree.sh` | `e20482f996dd9d9b1c234d1b9656df9ac3bcfa085c63ef06e4c4e11e57d926e7` | lista fissa (non modulo o fuori dal tracciamento); invariato |
| `aau/ava256/spec/ava256.json` | `36100e6187c3360d86071245ac6e816905a3de479085cd4ed836f811e1809a8f` | lista fissa (non modulo o fuori dal tracciamento); invariato |
| `aau/ava256/spec/hifi3d_det.json` | `5b12b58f255118c8748ed923edec9cb2e0c03108423be362b0deb47612c760ea` | lista fissa (non modulo o fuori dal tracciamento); invariato |
| `aau/ava256/spec/hifi3d_full.json` | `d892d0dd3a491ee91a394798c2b71245a974b634865b8ec09c51ff3733c91fcc` | lista fissa (non modulo o fuori dal tracciamento); invariato |
| `aau/ava256/spec/hifi3d_trace.json` | `3d3c1a2fb662ef0b125f2157d652e565664cc60f1166f8a8d32982fb573f07b7` | lista fissa (non modulo o fuori dal tracciamento); invariato |
| `aau/zs3dmm/zs_embed.py` | `195b189111fe91cb6ea5b02826e5cede87815bbe7bf8e861d7d91cb5dd3d6cf8` | lista fissa (non modulo o fuori dal tracciamento); invariato |

**ULRSSM (v6)**, fuori dal repo o non codice, controllati da `conf_desc.py` prima del calcolo:

| Voce | Valore |
|---|---|
| clone `~/ext/ulrssm` (repo ufficiale, MIT) | commit `0625cbcbb7cd007893dd665b21ef55b275563472`, nessuna modifica (`git status`) |
| `aau/runs/competitors_ulrssm/faces_xyz/final.pth` (checkpoint finale, sha256) | `c9c6b674195e7f770c447170eadd78746b3a4efb92e746d408a36ce6320aae6f` |
| `aau/runs/evidence/competitors_new/ulrssm/selection.json` (sha256) | `94f6f065083490da4b2067fefb855036b135ea141484c7052ad25ee08d8137ae` |

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
`a256-t4-04`; uscite in `aau/runs/evidence/ava256_conf/collaudo/{embed_l40s,desc,desc-eval}.json` e
`collaudo/hifi3d_full/{embed_l40s,desc}/`; v5: gli esiti dell'embedding su L40S spostati in `embed_l40s`, invariati):

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
  forward gia' noto in sviluppo. L'esito su L40S resta "fuori tolleranza" e si registra. Effetto sulle letture del
  collaudo: Spearman <= 4,3e-4. v5: su Ava-256 l'embedding gira su A100, con il collaudo e le tolleranze sotto.
**Collaudo v5** (13 ottobre, decisioni del PI; uscite in `collaudo/embed.json`, `collaudo/hifi3d_full/embed/`,
`collaudo/eval-full.json`, `collaudo/eval-det.json`):
- **Embedding su A100.** Tolleranze v5, dichiarate PRIMA del collaudo (`conf_collaudo.TOL`, `embed_*_v5`): Z 1e-3 (circa
  3,5 volte il rumore A100 contro A100 di sviluppo, 2,8e-4); d_F (cal. e non cal.) 0,1 mm (1e-3 su s = log S, S ~ 100
  mm); d_P 1e-3; Spearman e IC 1e-3. La tolleranza della v4 (1e-4) era sotto il rumore noto; l'esito v4 su L40S resta
  registrato come fuori tolleranza (`collaudo/embed_l40s.json`). Embedding rifatto con gli stessi operatori del collaudo
  v4 su A100 (`nv-ai-04`, job 1071933; confronto rifatto col codice finale nel job 1072069):

  | Confronto | Scarto massimo misurato | Tolleranza v5 | Esito |
  |---|---|---|---|
  | Z, 600 mesh di `form_hifi` (s1234 / s2345 / C3M e123) e 1.000 del pool di D3 (s1234 / s2345) | 3,4e-4 / 3,0e-4 / 6,1e-4; pool 3,7e-4 / 4,7e-4 | 1e-3 | passa |
  | d_F cal. / d_F non cal. su tutte le coppie | 7,4e-3 / 0,026 mm | 0,1 mm | passa |
  | d_P su tutte le coppie | 4,5e-4 | 1e-3 | passa |
  | Spearman e IC (nocrop, crop, all_cross; GT FR e SR) delle 9 distanze | 4,3e-5 | 1e-3 | passa |

- **Valutatore con m = 9** (`conf_eval.py` senza R4 nella famiglia): `eval-full` (72 + 54 Spearman, 14 + 14 delta con
  R4 descrittiva, 24 Spearman dei protocolli e R7) ed `eval-det` (24 Spearman col rilevatore, 3 delta) rifatti: scarto
  0, righe uguali (job 1072069, 1072070). Gli esiti delle regole su HIFI3D (`collaudo/hifi3d_full/conf/rules.csv`, 9
  ipotesi per famiglia) restano solo un controllo del codice.
- Cancellata dopo il collaudo la cache degli operatori di HIFI3D (`datasets/V3_OPS_CACHE/conf/hifi3d_full`, 16 GB).

**Collaudo v6** (13 ottobre):
- **ULRSSM**, tolleranza fissata PRIMA del collaudo (`conf_collaudo.TOL["ulr_rel"]`): per ognuna delle 4 letture, scarto
  massimo / mediana delle distanze di sviluppo confrontate <= 1e-3 (la mappa punto-a-punto e' un argmax discreto su
  feature calcolate con operatori rifatti); stesse mesh e coppie fallite. Sottocampione dichiarato: i primi 12 soggetti
  per id (72 mesh, `desc.subsample_subjects`), stessa vista da spec di Ava-256, L40S (`a768-l40s-04`, job 1072196):

  | Lettura | Coppie | Scarto massimo | Relativo alla mediana | Esito |
  |---|---|---|---|---|
  | `pair_fr` (mm) | 2.376 | 2,1e-3 mm | 3,5e-4 | passa |
  | `pair_sr` (mm a CS_ref) | 2.376 | 2,8e-3 mm | 5,0e-4 | passa |
  | `tpl_fr` (mm) | 2.556 | 1,8e-3 mm | 2,9e-4 | passa |
  | `tpl_sr` (d_P) | 2.556 | 3,0e-5 | 3,2e-4 | passa |

  L40S riproduce entro la tolleranza: il vincolo GPU resta L40S. Lettura (`desc-eval`, plug-in di sviluppo): 24 rho
  puntuali delle 4 letture (nocrop, crop, all_cross; FR e SR) contro `ulrssm/faces_xyz/spearman.csv`, scarto 8e-17.
- **Tutti i confronti rifatti** con `fm_eval.py` della matrice rigenerata (`46fe7a4`) e la nuova `full_matrix` (job
  1072542, 1072543): `eval-full` ed `eval-det` 0; `embed` (A100) entro le tolleranze v5; `desc` invariato; `desc-eval`
  782 Spearman e 2.190 delta (ora anche contro i plug-in) a 0 sugli stessi ingressi, <= 5,7e-6 sui ricalcolati, righe
  uguali. La rilettura del job 1072197 aveva usato una copia di lavoro intermedia di `fm_eval.py` ed e' sostituita.

**Collaudo v7** (13 ottobre):
- **Controlli A-D** (`conf_collaudo.py guard`, eseguito da un worktree al commit `2cea824`, nodo `i256-a10-07`, job
  1073352; `collaudo/guard.json`): 13 casi su 13 come atteso. `verify` passa sul worktree pulito al suo commit e
  fallisce con un altro commit, con un file tracciato modificato (e torna a passare dopo il ripristino) e su Ava-256
  senza `CONF_FROZEN_COMMIT`, prima di leggere qualunque dato. `conf_run.py` esce con 0 se i moduli sono uguali al commit
  e con 3 contro un commit precedente in cui `conf_common.py` era diverso. `host --role bp`, `conf_head.py fit` e
  `conf_desc.py arcface` falliscono sul nodo sbagliato, `host` passa sul nodo giusto (spec copia). `guard_overwrite`
  (testa gia' scritta con `freeze`) e `check_host` (testa di un altro nodo) rifiutano.
- **Tracciamento** (spec `hifi3d_trace.json`, worktree al commit `2cea824`): tutti i passi della catena di Ava-256 su 3
  soggetti, con `CONF_FROZEN_COMMIT`. Passi CPU su `a512-l4-06` (job 1073353; nodi dello spec di tracciamento su quel
  nodo, `gate` escluso), embedding, varifold e ULRSSM su A100 (job 1073354), testa, valutatore e descrittivi (job
  1073355). 127 moduli, nessuno scarto dal commit. Il K0 della testa qui non si legge: la testa gira fuori da
  `i256-a10-10`, come vuole lo spec di tracciamento.
- **Collaudi numerici rifatti dal worktree** al commit `2cea824` con `CONF_FROZEN_COMMIT` (job 1074450, 1074451):
  `eval-full` ed `eval-det` 0; `embed` (A100) entro le tolleranze v5; `desc` invariato; `ulr` entro `ulr_rel`;
  `desc-eval` 782 Spearman e 2.100 delta a 0 sugli stessi ingressi e <= 5,7e-6 sui ricalcolati, righe uguali. I delta
  sono 90 meno della v6 perche' `pair_sr` e `tpl_fr` di ULRSSM sono ora non dichiarate e non entrano nei `vs_nostro`.

**Collaudo v8** (13 ottobre, worktree al commit `8a6e184`, `CONF_FROZEN_COMMIT`):
- `guard` (job 1074573, nodo `i256-a10-07`): 19 casi su 19 come atteso. Oltre ai 13 della v7: `verify` passa con una
  copia pulita del clone diffusion-net al posto del link, **fallisce con `geometry.py` modificato** nella copia e ripassa
  col link originale; `conf_head.resume` salta un braccio con npz intero e sha registrato e si ferma con npz senza
  json o con sha diverso; `conf_desc.fit_hosts` rifiuta un fit B scritto da un altro nodo.
- Tracciamento: 131 moduli, nessuno scarto (sopra). Il passo dei percettivi si e' fermato una volta con un segfault della
  libreria UCX all'avvio (job 1074574, nessuna uscita scritta) ed e' stato rilanciato identico (job 1074990, passa).
- Collaudi numerici rifatti (job 1074577, 1074578): `eval-full`, `eval-det`, `embed`, `desc`, `desc-eval`, `ulr` passano
  come nella v7 (le uscite del fit B del collaudo, precedenti alla v8, non hanno `host`: ammesso solo nei collaudi).

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
- **Ripresa della testa di R6 (v8).** Se `head-fit` si interrompe fra i due bracci, si rilancia lo stesso passo:
  `conf_head.resume` salta il braccio gia' completo solo se il suo `head/<braccio>.npz` e' intero (si legge) e il suo
  sha256 e' quello registrato nel json d'evidenza (`npz_sha256`, scritto dopo l'npz); un npz senza json, con sha diverso
  o illeggibile ferma il job e decide il PI (nessun rifit di un braccio completato).
- **Crash del codice congelato su Ava-256 (v7).** Se un passo del codice congelato si ferma con un errore su Ava-256
  (eccezione, nessun risultato), e' ammessa SOLO una correzione minima, limitata alla riga che fallisce: commit a parte
  sopra il congelato, pubblicato col diff, il motivo e il messaggio d'errore; il passo si rifa' da un worktree a quel
  commit, che deve differire dal congelato solo per quella riga (`git diff`). Nessun altro cambiamento, nessuna scelta
  guardando i numeri. Se la riga sta nella catena di una regola R1-R7 (codice delle sue colonne, dei loro ingressi o
  del valutatore), quella regola e' **non risolta** (sez. 8): si riporta con delta, IC e p e resta nella famiglia di
  Holm col suo p; un descrittivo toccato si riporta con la nota. Un errore che non si corregge in una riga lascia la
  regola o il descrittivo non calcolati, con il motivo.
- Figure: solo forme medie o aggregate, salvo conferma scritta di Meta per render di singole persone (README,
  Licenza).

## 13. Cosa NON e' confermativo

- Tutto cio' che non e' R1-R7 (R4 esclusa dalla v5):
  - R4 e l'intero gruppo `crop` (v5: scelta di ambito, il crop e' fuori dalla tesi principale; si riportano completi,
    con delta e IC di R4 nei due semi, in `conf/deltas.csv` con `famiglia` = False, e nel supplementary come limite);
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
1. ~~ULRSSM~~: **fissato nella v6**, entra come descrittivo (`faces_xyz`, sez. 9, 11.1, 11.2);
2. ~~Hybrid FM~~: **escluso nella v6** (non pronto al congelamento);
3. ~~embedding di Ava-256~~: **fissato nella v4** (`conf_embed.py`, sez. 7.1, 11.1, 11.2); collaudo su L40S **fuori
   tolleranza** e registrato (sez. 11.2); v5: su A100, collaudo rifatto con le tolleranze v5 (sez. 11.2);
4. ~~descrittivi della sez. 9 fuori da `conf_eval.py`~~: **fissati nella v4** (`conf_desc.py`, sez. 9, 11.1, 11.2).

Fissati nella v2 (non piu' in attesa): R7 e la famiglia di Holm (m = 10; v5: m = 9, senza R4); Gromov-Wasserstein e percettivi (varianti
dichiarate in `5c9c3e3`); protocolli NoW, NoW metrical, REALY, Stirling (codice in `32c3839`, risultati di sviluppo in
`c960859`, pipeline del rilevatore in `f47a375`).

Fissati nella v3: catena dei protocolli col rilevatore, valutatore confermativo, blocco di R6, baseline e fit B con
`--spec`, collaudati su HIFI3D (sez. 11.1, 11.2); deroga dei protocolli (sez. 9); controllo bloccante del rilevatore e
soglia dei ripieghi (sez. 8, 9.1); versioni e sha256 della sez. 11.

Fissati nella v4: embedding dei due semi e di C3M e123 (`conf_embed.py`), calcolo e lettura dei descrittivi e dei
gruppi descrittivi (`conf_desc.py`), collaudati su HIFI3D con gli esiti della sez. 11.2 (descrittivi entro tolleranza,
embedding fuori), sha256 nella sez. 11.1.
Fissati nella v6: ULRSSM come descrittivo (codice, pesi, selezione e commit del clone nella sez. 11.1, collaudo nella
sez. 11.2); Hybrid FM escluso. **Nessun campo resta in attesa**: il testo e' pronto per il congelamento del PI.

Poi il PI committa questo testo con il suo sha256 e la parola "congelato" nel titolo.

Al congelamento (v9): codice `8a6e184`; worktree a quel commit (`conf_worktree.sh`), tutti i job con
`CONF_FROZEN_COMMIT=8a6e184d2951d73df7ea55e48296c4df36b92aef` (comando in `aau/ava256/FROZEN.md`).

**v9, CONGELATO: UNA SOLA VALUTAZIONE SU AVA-256, DAL CODICE `8a6e184`.**
