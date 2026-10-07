# Protocollo dichiarato prima delle eval: riconoscimento d'identita' IN DOMINIO, soggetti mai visti, e tempi

Dichiarato il 2026-10-07 11:50 CEST, prima di calcolare qualunque embedding, matrice faceBench, render
ArcFace o tempo su questi soggetti. Numeri gia' visti e noti in anticipo: la tabella WS2
(`aau/runs/ws2_cross3dmm/summary.md`, Spearman con la GT di distanza, congiunto BFM 0.865 / ICT 0.985
senza crop, misto espressioni 0.881) e i riconoscimenti zero-shot di `aau/runs/ws_faceverse_expr` e
`aau/runs/arcface_render_zs`. Nessun numero di riconoscimento in dominio esiste a quest'ora.

**Domanda.** Con le sole etichette d'identita' (non la GT di distanza), su soggetti che il modello non ha
mai visto ma dello stesso 3DMM del training, il congiunto (DiffusionNet, BFM+ICT, job 1019532) riconosce
le persone attraverso topologie (ed espressioni) diverse quanto le pipeline geometriche e ArcFace? E
quanto costa, in tempo, rispetto a loro?

## Soggetti e insiemi (nessun soggetto nel training del congiunto)

Viste di `aau/cross3dmm/ws2_views.py` (split del training RICOSTRUITO con le funzioni del training e
verificato contro i 16 soggetti dell'eval online; `aau/runs/ws2_cross3dmm/splits.json`). Prima delle eval
il preparatore (`aau/indomain/ir_views.py`) ricontrolla |valutati inter training del congiunto| = 0 su
`splits.json` e si ferma altrimenti.
- **BFM**: `datasets/WS2_CROSS3DMM/joint__bfm`, i 108 held-out BFM del congiunto, 6 topologie (crop,
  down8k, noisy, original, remesh, up60k).
- **ICT**: `datasets/WS2_CROSS3DMM/joint__ict`, gli 89 held-out ICT del congiunto nel pool
  id14500-id14999, 6 topologie.
- **ICT con espressioni (rexpr)**: gli stessi 89 soggetti; per ognuno la mesh neutra (`original`) e le 5
  espressioni casuali di `datasets/ICT/expressions_random_withops` (`rexpr1..5`, estrazioni indipendenti
  per soggetto: l'etichetta k non e' la stessa espressione fra soggetti). Tutte nella topologia ICT
  `original` (9409 vertici, stessa connettivita'): qui varia l'espressione, NON la topologia.
- **BFM-only**: dei 108 soggetti BFM, 89 sono nel training del BFM-only (job 1019310). Il BFM-only si
  valuta SOLO sui 19 soggetti held-out di entrambi i modelli (`BFM-19`), e li' si confronta appaiato
  col congiunto e con le baseline sugli stessi 19. Galleria di 19: rank-1 piu' facile che su 108, e CI
  larghi; non si confronta con le righe a 108.
- Nota (stessa delle eval WS2): 16 dei 108 soggetti BFM erano nell'eval online del congiunto, cioe' hanno
  pesato sulla scelta del checkpoint `best_by_xtopo_mesh_clean` (non sul gradiente). ICT: 0.

## Metodi (stessi per ogni insieme salvo dove detto)

- **Congiunto**: `||z_i - z_j||` dagli embedding di ogni mesh (`aau/zs3dmm/zs_embed.py`, stessa catena
  dello script di breakdown), frame nativo dei dati di training (BFM e ICT come nel training: in dominio
  non serve alcuna convenzione), checkpoint `best_by_xtopo_mesh_clean`.
- **BFM-only** (solo BFM-19): idem col checkpoint del job 1019310.
- **Chamfer**, **ICP rigido + Chamfer**, **NICP P2Tri**: pipeline faceBench vera
  (`run_facebench_remesh.run_geometry_pipeline`, via `alignment_matrix._run_chunk`, importata):
  normalizzazione maxabs per mesh, 4096 punti campionati, `icp_align` con prealign bbox,
  `nonrigid_icp_align`. Ogni coppia non ordinata di mesh con etichette diverse si calcola UNA volta,
  con X = la mesh dell'etichetta che viene prima nell'ordine delle etichette (NICP e' asimmetrico:
  l'orientazione e' fissata dall'ordine delle etichette, non query -> galleria, come in zs3dmm). Seme
  della coppia = indice della coppia nella sua coppia di etichette. Coppie fallite = NaN = +inf.
- **ArcFace su normal map** (riga ArcFace di riferimento): `aau/zs3dmm/zs_arcface_render.py` con
  `--mode normals`, normali in spazio camera, 3 viste (yaw 0, -30, +30), 512 px, crop fisso calibrato
  sui render ombreggiati dello stesso insieme (`--calibration-from`), `w600k_r50.onnx`, distanza 1 -
  coseno della media rinormalizzata delle 3 viste. Rotazione verso il frame del renderer UNA per dominio,
  fissata col `--frame-check` prima degli embedding (attese: BFM nessuna, ICT Rx(180)). Secondaria:
  ArcFace ombreggiato 3 viste (serve comunque alla calibrazione).

## Misure (funzioni importate da `aau/zs3dmm/zs_expr_summarize.py`, non riscritte)

`retrieval_queries`, `verification_pairs`, `recognition_values`, `weighted_auc`, `bootstrap_counts`:
- retrieval: query (A, t1), galleria = le N mesh in t2 != t1 (una per soggetto); rank-1 e mAP (un solo
  rilevante: AP = 1/rank; pari a meta' strada);
- verifica: AUC di -distanza su coppie (A, t1)-(B, t2), t1 < t2 nell'ordine delle etichette, stessa persona
  contro persone diverse;
- IC 95% bootstrap per SOGGETTO, 1000 repliche, le STESSE repliche per tutti i metodi di un insieme
  (seme `stable_seed(1234, "indomain_recognition:<insieme>")`): delta APPAIATI **congiunto - ciascun
  metodo**, con CI e P(delta <= 0).

Blocchi:
- **BFM, ICT (PRIMARIO: senza crop)**: le 5 topologie senza crop, 20 coppie ordinate per il retrieval, 10
  non ordinate per la verifica. **Crop a parte**: coppie con crop da un lato (10 ordinate per il retrieval,
  5 per la verifica), mai nel primario.
- **rexpr (PRIMARIO: espressione contro espressione)**: query in `rexpr k`, galleria in `rexpr k'`, k != k'
  (20 coppie ordinate; verifica sulle 10 non ordinate). **Secondario (dichiarato): neutra in galleria**:
  query in `rexpr k`, galleria neutra (5 coppie; verifica neutra-espressione, 5).
- **BFM-19**: gli stessi due blocchi di BFM, sui 19 soggetti.

**Lettura fissata ora** (per insieme e blocco primario, su rank-1 e AUC separatamente): "congiunto sopra X"
se il CI 95% del delta congiunto - X e' tutto sopra 0; "pari" se contiene 0; "sotto" se e' tutto sotto 0.
Nessuna correzione per confronti multipli: i delta si leggono uno a uno, e lo si dice. Il risultato su
rexpr non e' cross-topologia (stessa mesh ICT) e si legge come test d'invarianza all'espressione.

## Tempi

Un solo job su un nodo L40S (`--gres=gpu:l40s:1`), tutto sullo stesso nodo; host, CPU e GPU nel referto.
Parti CPU a UN thread (`OMP/MKL/OPENBLAS_NUM_THREADS=1`), come girano le pipeline. Campione: 60 mesh (5
soggetti BFM + 5 ICT dagli insiemi sopra, 6 topologie ciascuno) e 60 coppie fra loro (topologie diverse,
meta' stessa persona). Si riportano mediana e IQR (25-75%) per voce, sul campione intero (e per topologia
nel csv). Dati letti da `/tmp` (copiati a inizio job), quindi il tempo di lettura e' quello da RAM.
- **Congiunto, per mesh**: (a) operatori DiffusionNet (`compute_operators`, k_eig = 128, sulla mesh ad area
  unitaria come in `areanorm_operators.py`), CPU; (b) embedding: caricamento del campione come nel
  dataset + `forward_model` sulla GPU (con sincronizzazione), e a parte sulla CPU a un thread; totale = a + b.
- **Congiunto, confronto**: `||z_a - z_b||` fra due embedding gia' calcolati (CPU, numpy, d = 256).
- **Congiunto, retrieval 1:N**, N = 100, 1.000, 10.000, embedding precalcolati: distanze query-galleria +
  ordinamento (argsort), CPU a un thread e GPU. Le gallerie oltre le mesh disponibili si costruiscono
  ricampionando embedding veri (il tempo non dipende dai valori). Almeno 50 query per N.
- **ICP rigido + Chamfer** e **NICP P2Tri**, per coppia: `run_geometry_pipeline` con il solo stadio `rigid`
  e il solo stadio `nicp` (lo stadio `nicp` rifa' da se' l'ICP rigido che precede la NICP; con `raw,rigid,nicp`
  l'ICP rigido si conterebbe due volte), lettura delle due mesh compresa. [Precisazione delle 12:05, prima di
  qualunque misura: la prima stesura diceva `raw,rigid` e `raw,rigid,nicp`.]
- **ArcFace-render (normal map), per mesh**: lettura + 3 render a normal map + crop + 3 embedding ArcFace
  (onnxruntime CPU) + media; il crop usa la calibrazione gia' fatta (la calibrazione e' una tantum per
  dominio, riportata a parte se misurabile, non nel tempo per mesh).
- Lettura per l'utente: costo di un confronto 1:1 e di un 1:N di ogni metodo; per le pipeline a coppie
  un 1:N costa N volte la coppia (stima, non misurata, e dichiarata come tale).

## Revisione 1 -- 2026-10-07 16:25 CEST, dopo la revisione del critic e PRIMA di qualunque numero nuovo

Gia' visti a quest'ora: tutti i numeri della prima tornata (summary.md del 2026-10-07 ~15:00). Nessun
numero esiste per le varianti qui sotto. La prima tornata resta com'e'; le sue righe restano nelle tabelle.

**Perche'.** Le mesh sono ad area unitaria: il crop ha meno area e quindi esce ingrandito (~8%) rispetto
alle altre topologie dello stesso soggetto. faceBench normalizza maxabs per mesh, prealinea col bbox e fa
un ICP RIGIDO (senza scala): l'errore di scala del crop non viene mai corretto, e le righe faceBench con
crop della prima tornata misurano anche questo artefatto. Inoltre il congiunto ha visto la topologia crop
(e noisy, down8k, ...) dei soggetti di training: e' un'augmentation che le baseline non hanno.

**A. ICP di similarita' (baseline standard da qui in poi).** Stessa pipeline faceBench (maxabs, 4096 punti,
`prealign_by_bbox`), ma l'ICP di open3d con `TransformationEstimationPointToPoint(with_scaling=True)`
(soglia 1000 come `icp_align`). Due righe nuove: **ICP di similarita' + Chamfer** (media P2P sulle
corrispondenze Chamfer, come `rigid_p2p`) e **ICP di similarita' + NICP P2Tri** (`nonrigid_icp_align` e
`p2tri_distance` di faceBench sul risultato). Tutte le coppie, con crop e senza, su BFM (108) e ICT (89),
stesse regole della prima tornata. rexpr NON si rifa': niente crop, stessa topologia (dichiarato).

**B. NICP su template (baseline "iscrivi una volta, confronta in corrispondenza densa").** Template per
dominio = media vertice per vertice delle mesh `original` (maxabs) di 100 soggetti di TRAINING del congiunto
(`rng(1234)`), stessa topologia del 3DMM; 4096 vertici del template scelti una volta (`rng(0)`). Iscrizione
di una mesh: maxabs, 4096 punti campionati (seme fisso per mesh), ICP di similarita' template -> mesh,
`nonrigid_icp_align` del template sulla mesh, poi similarita' (Procrustes con scala) dei 4096 punti
registrati verso il template, cioe' tutto in un frame canonico. Confronto = distanza L2 media per vertice
fra due mesh iscritte. Insiemi: bfm, ict, rexpr, ict992.

**C. ArcFace configurato al meglio (righe nuove; le vecchie restano come secondarie).** Inquadratura PER
MESH invece della camera unica per dominio: centro ed estensione dai vertici dentro i percentili 0.5-99.5
per asse (un vertice isolato non rimpicciolisce il volto), stesse 3 viste; normali smussate per vertice
(normali di faccia girate verso la camera, mediate per area sui vertici, colore del triangolo = media
normalizzata delle sue 3 normali di vertice). Crop ricalibrato su questi render ombreggiati e copiato alla
normal map, come prima. Flag nuovi in `zs_arcface_render.py` (`--camera mesh`, `--normals vertex`) con i
default di prima: i risultati in `aau/runs/arcface_render_zs` non cambiano.

**D. Galleria grande (effetto soffitto).** `ict992`: i 992 soggetti ICT held-out del congiunto (0 nel suo
training, ricontrollato), 6 topologie. Query: 100 soggetti scelti con `rng(1234)` fra i 992. Due blocchi:
**noisy -> original** (senza crop, PRIMARIO del blocco) e **crop -> original** (a parte); galleria = i 992
in `original`. Metodi sugli stessi blocchi: congiunto, ICP di similarita' + NICP P2Tri (a coppie, X = query),
NICP su template, Chamfer faceBench. In piu', solo per congiunto e template (costano un'iscrizione per
mesh): tutte le 992 query su tutte le 20 coppie senza crop e le 10 con crop (secondario).
Bootstrap sulle query (galleria fissa), 1000 repliche. Verifica sulle coppie query x galleria: 100
genuine e 99.100 impostori per blocco; TAR@FAR = 1e-3 e 1e-4 si appoggia a ~99 e ~10 impostori sopra
soglia: la seconda e' rumorosa e lo si dice.

**E. Misure.** Si aggiunge TAR@FAR 1e-3 e 1e-4 (soglia = quantile pesato degli impostori; TAR = frazione
pesata dei genuini sopra soglia) a tutte le tabelle. Lettura invariata (CI dei delta appaiati).

**F. Tempi.** Si rifanno tutti su un nodo L40S in `--exclusive`. Tabella divisa in **iscrizione** (per mesh:
congiunto = operatori + embedding; template = ICP di similarita' + NICP + Procrustes; ArcFace = render +
embedding) e **ricerca su galleria gia' iscritta** (1:N, N = 100, 1.000, 10.000: congiunto L2 su 256-d,
template L2 media per vertice su 4096 x 3, ArcFace coseno su 512-d). Per i metodi a coppie (ICP di
similarita' + Chamfer, ICP di similarita' + NICP P2Tri) la ricerca 1:N costa N coppie: tempo per coppia
misurato, 1:N stimato come N x mediana (dichiarato).

*Precisazione al punto C, 2026-10-07 17:15 CEST, prima di qualunque embedding della configurazione nuova:*
l'inquadratura per mesh con margine minimo (1.02, come la camera di dominio) fa riempire al volto tutto il
fotogramma, e il detector della calibrazione non scatta su nessun render a yaw -30 (job 1060307, fallito
prima degli embedding). Il margine per mesh diventa 1.2, cioe' la dimensione media del volto della camera
di dominio (BFM: scala 1.83 contro ~1.54 di estensione minima per mesh). Nient'altro cambia.
