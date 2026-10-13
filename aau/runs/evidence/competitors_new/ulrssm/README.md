# ULRSSM: distanze per coppia e via template (concorrente "corrispondenza densa imparata")

Metodo: Cao, Roetzer, Bernard, "Unsupervised Learning of Robust Spectral Shape Matching", ACM TOG (SIGGRAPH) 2023.
Codice ufficiale (MIT) clonato FUORI dal repo e importato (`ULRSSM_DIR`, default `~/ext/ulrssm`, commit
`0625cbcbb7cd007893dd665b21ef55b275563472`); la colla e' in `aau/competitors/ulrssm/` (docstring dei moduli per le
definizioni esatte). Pesi, mesh e npz NON sono in git.

## Cosa c'e' qui

| file | contenuto | in git |
| --- | --- | --- |
| `selection.json` | regola e scelta fra ingresso `xyz` e `wks` (errore di validazione sui generatori, prima della valutazione) | si' |
| `<tag>/<vista>.json` | tempi per mesh e per coppia, mesh fallite, diagnostica della rigida robusta, NaN | si' |
| `<tag>/spearman.csv`, `<tag>/paired.csv`, `<tag>/results.md`, `<tag>/paired_info.json` | Spearman FR / SR con IC, delta appaiati, tabelle | si' |
| `<tag>/<vista>.npz` | matrici simmetriche (mesh x mesh) e configurazioni trasferite | no (`*.npz`) |
| `<tag>/pairs_<vista>.csv` | le distanze per coppia in formato lungo (sotto) | no (`pairs_*.csv`) |
| `<tag>/parts/` | uscite grezze dei job GPU | no |
| `smoke/` | controlli della mappa e tempi di prova | si' |
| `<tag>/fm_plugin.json` | marcatura del plug-in per `aau/full_matrix/fm_eval.py`: letture `pair_fr`, `pair_sr`, `tpl_fr`, `tpl_sr` (matrici `D_*` dell'npz) con la loro GT; `declared` solo per la variante di `selection.json` | si' |

`<tag>`: `faces_xyz`, `faces_wks` (ULRSSM riaddestrato sui generatori di training, ingresso xyz o WKS),
`zeroshot_faust` (pesi pubblici FAUST, controllo).

## Formato delle distanze per coppia (`pairs_<vista>.csv`)

Una riga per coppia NON ordinata di mesh, chiavi uguali alle righe di E12 / delle baseline parametriche:

    subject_a,topology_a,subject_b,topology_b,d_pair_fr,d_pair_sr,d_tpl_fr,d_tpl_sr

- coppie: `ulr_common.pairs`: tutte le i < j con soggetti diversi, nell'ordine delle mesh della vista (soggetti di
  `blmm.subjects` ordinati, topologie `crop, down8k, noisy, original, remesh, up60k`): 178.200 per vista, crop
  compreso (all_cross + stessa topologia, le righe della matrice unificata). Le parti zero-shot di HIFI3D, FaceScape e
  FaceVerse con espressioni sono state calcolate solo su all_cross (`--pairs all_cross`, 148.500): li' le coppie con la
  stessa topologia sono NaN nell'npz e assenti dal csv. FaMoS TEST: le 105 coppie delle 15 scansioni di galleria
  (`topology = scan_gallery`, soggetto = `view_id`).
- la variante (b) e' per mesh: nell'npz `D_tpl_*` e' piena (tutte le coppie, anche stesso soggetto).
- le distanze sono simmetriche: la riga (a, b) di all_cross e la (b, a) leggono la stessa cella (cercare entrambi
  gli ordini, oppure usare le matrici dell'npz con `subjects`/`topologies` e l'`Index` di `zs_expr_summarize`).
- `d_pair_fr` (mm): variante (a), mappa punto-a-punto di ULRSSM nei due versi, rigida pesata SENZA scala sui punti
  corrispondenti, RMS pesato per area (aree baricentriche della mesh d'arrivo), media dei due versi.
- `d_pair_sr` (mm alla taglia di riferimento): lo stesso con ogni mesh moltiplicata per CS_ref / CS_i (centroid size
  robusta di `blmm.mesh_scalars`, CS_ref del dominio da `blmm.params`).
- `d_tpl_fr` (mm), `d_tpl_sr` (d_P, adimensionale, come la GT SR): variante (b), ogni mesh mappata sul template
  fisso (media di GNM Head, generatore di training), i suoi vertici trasferiti sui 1478 punti della regione unificata
  e passati alla funzione della GT di valutazione importata senza modifiche (`train_fr_sr.factor`:
  `cgt.Canon.rigid_robust` verso mu, poi a_i per FR e pre-forme z_i = (a_i - c_i) / S_i per SR; distanze con la
  formula di `train_fr_sr.save_eval`). Controllo `gt_check` in `<vista>.json`: la stessa funzione sui punti nativi di
  FaMoS TEST ridà `famos_test_{fr,sr}`.
- `NaN`: mesh fallita in preparazione (elencata in `<vista>.json`).

Rieseguire su un dominio nuovo (es. Ava-256, NON eseguito): `ULR_SPEC=<json>` nei passi `eval` e `merge` di
`aau/competitors/ulrssm/ulr.sbatch` (formato della spec nel docstring di `ulr_common.py`, lo stesso di
`gw_percettivi`); lo Spearman di `ulr_paired.py` usa le righe e le GT delle viste di questo repo.

## Ricetta di training (`training/`)

- Dati (`training/data_manifest.json`, `ulr_data.py`): SOLO i generatori del nostro encoder, preset `validated`
  (BFM 2019, ICT, GNM), ricetta `c3m` dello stream: 200 identita' fresche per dominio (+ 10 di validazione, identita'
  diverse), 6 viste per identita' con una discretizzazione per etichetta (original, remesh, down8k, noisy, crop,
  up60k), quota d'espressioni del C3M per dominio. 3.600 mesh di training (vertici: mediana 9.022, max 26.017), 180
  di validazione. Nessun dominio di valutazione, nessuna GT, nessun dato Ava.
- Modello e perdite: `FMNetModel` di ULRSSM con `options/train/faust.yaml` (DiffusionNet out 256, RegularizedFMNet
  bidirezionale, SURFMNet bij + orth, allineamento nei due versi, Similarity tau 0.07, Adam 1e-3, coseno a 1e-4 su 15
  epoche, clip 1). Budget: 100.000 coppie (default FAUST 80^2 x 15 = 96.000), coppie uniformi con reinserimento.
- Ingresso: due run, `xyz` (coordinate ad area 1 con l'augmentation di ULRSSM) e `wks` (il default per forme
  complete). Regola fissata prima delle valutazioni (`selection.json`): errore di validazione minore sui generatori
  (coppie con la stessa connettivita', corrispondenza nota) -> `faces_xyz` dichiarata, `faces_wks` variante.
- Tempi (L40S): xyz 12,2 h (job 1070607), wks 12,5 h (job 1070608), piu' ~16 min di operatori su CPU a inizio job.
  Errore di validazione finale (x100, area 1): xyz 1,87, wks 1,95; a 4.000 iterazioni era 1,73 / 2,00 (curva in
  `training/*_train.json` e nei `log.jsonl` locali): si usa comunque il checkpoint FINALE, come dichiarato.

## Esito in breve (numeri in `<tag>/results.md`)

- `faces_xyz` (dichiarata): concorrente forte su HIFI3D senza crop e su FaMoS TEST, debole su FaceScape e FaceVerse,
  e crolla sulle righe col crop di HIFI3D e FaceScape (vedi tabelle).
- `faces_wks`: rho fra -0,07 e 0,17 ovunque. Sulla validazione dei generatori era quasi pari a xyz (1,95 contro 1,87),
  ma sulle viste di valutazione la mappa sul template non regge: residuo verso mu 27-68 mm (xyz: 5-11 mm). La
  validazione in distribuzione non vede lo spostamento di dominio; la regola ha comunque scelto xyz.
- `zeroshot_faust` (pesi pubblici FAUST, WKS): rho fra -0,03 e 0,34 (FaMoS, IC larghi), il fallimento atteso.

## Scelte e deviazioni dichiarate

- Il ciclo di training e la mappa di test sono riscritti in poche righe (il `train.py` di ULRSSM mette la L densa in
  ogni campione); stesse operazioni nello stesso ordine. Mappa a blocchi contro il percorso di ULRSSM
  (`smoke/smoke_1070609.json`): Pyx identica, 99,6-99,75% dei vertici con la stessa immagine dopo il passo spettrale
  (pareggi numerici di cdist; immagini diverse a ~2 mm, FR della coppia uguale entro 0,04%).
- Facce orientate col volto verso +z (normale media; nessuna mesh di training o di valutazione e' stata invertita).
- Nessun test-time adaptation (il default di ULRSSM per FAUST/SCAPE non la usa).
- Crop: ULRSSM per forme complete, nessuna variante parziale (le `PartialFmapsLoss` di SHREC'16 richiedono coppie
  parziale-completo con la parte nota); le righe col crop sono valutate col modello base e riportate a parte.
- Variante (b): template = media di GNM (la sua hockey mask); le viste con piu' superficie del template (FaceScape:
  191 mm in altezza) violano l'ipotesi completo-completo, si vede nel residuo verso mu (`<vista>.json`,
  `template.rms_to_mu_mm_median`: FaceScape 11,4 mm contro 5-6,5 di HIFI3D e FaceVerse; 16 rigide robuste non
  convergono in 100 iterazioni su FaceScape).
- Hybrid Functional Maps (Bastian et al., CVPR 2024) NON eseguito: la base elastica richiede `pyshell` (estensione
  C++ di gitlab.com/numod/shell-energy, da compilare) e un'autodecomposizione 3n x 3n per mesh (fino a 220k gradi di
  liberta' sulle up60k di FaceVerse) per 3.780 mesh di training e 2.415 di valutazione, piu' un secondo training di
  ~12 h: non "quasi gratuito".

## Job (12-13 ottobre 2026)

dati 1070577; smoke 1070609; training 1070607 (xyz), 1070608 (wks); zero-shot FAUST 1070618, 1070619, 1070682,
1070802-1070805 (merge 1070747-1070749, 1070884, 1070885; Spearman 1070886); valutazione xyz 1071079, 1071080,
1071083, 1071084, 1071087, 1071088, 1071162 (merge 1071093, 1071096, 1071097, 1071288, 1071094; Spearman 1071289);
valutazione wks 1071081, 1071082, 1071085, 1071086, 1071089, 1071090, 1071092 (merge 1071099-1071103; Spearman
1071104); selezione 1071078.
