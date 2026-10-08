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

*Aggiunta al punto D, 2026-10-08 02:25 CEST, dopo aver visto il blocco noisy -> original e PRIMA di
calcolare il blocco nuovo:* su noisy -> original la Chamfer grezza fa rank-1 1.000 con 992 soggetti. Non e'
un risultato ma un difetto del blocco: `noisy` e' la `original` con i vertici perturbati, quindi le due
mesh campionano la stessa superficie con gli stessi vertici, e qualunque metodo a punti le appaia. Il
blocco resta riportato ma non misura il soffitto. Si aggiunge **remesh -> original** (stesse 100 query,
galleria di 992 in `original`): `remesh` e' una ritassellazione senza vertici in comune con `original`
(down8k e up60k invece li condividono in parte). Diventa il PRIMARIO senza crop della galleria grande.
Stessi metodi (congiunto, ICP di similarita' + NICP P2Tri, NICP su template, Chamfer). Anche crop ->
original condivide i vertici della `original` (e' un suo ritaglio): lo si dice nella lettura.

---

# Risultati

CI 95% bootstrap per soggetto, 1000 repliche (salvo dove detto), le stesse per tutti i metodi di un blocco. mAP = MRR (un solo rilevante). Distanze NaN (coppie fallite) = +inf.

**Nota (revisione 1):** il congiunto ha visto in training la topologia crop (e noisy, down8k, remesh, up60k) dei soggetti di training, cioe' un'augmentation che nessuna baseline ha avuto. Le righe ICP rigido (prima tornata) portano l'artefatto di scala del crop; le righe ICP di similarita' no.

## BFM, 108 held-out del congiunto

### PRIMARIO, 5 topologie senza crop

Retrieval: 2160 query (20 coppie ordinate x 108), galleria di 108. Verifica: 1080 coppie stessa persona, 115560 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | TAR@FAR 0.001 | TAR@FAR 0.0001 | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 0.998 [0.996, 1.000] | 0.999 [0.998, 1.000] | 1.000 [1.000, 1.000] | 0.969 [0.936, 0.989] | 0.863 [0.654, 0.940] | 0 |
| ICP di similarita' + NICP P2Tri | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.995, 1.000] | 0 |
| ICP di similarita' + Chamfer | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.988, 1.000] | 0 |
| NICP su template (iscrizione) | 0.856 [0.825, 0.884] | 0.915 [0.895, 0.932] | 0.991 [0.988, 0.995] | 0.613 [0.531, 0.717] | 0.367 [0.278, 0.465] | 0 |
| ICP rigido + NICP P2Tri (prima tornata) | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0.967 [0.936, 0.980] | 0.894 [0.841, 0.943] | 0 |
| ICP rigido + Chamfer (prima tornata) | 0.934 [0.920, 0.948] | 0.954 [0.944, 0.964] | 0.992 [0.990, 0.995] | 0.825 [0.794, 0.859] | 0.731 [0.612, 0.798] | 0 |
| Chamfer faceBench | 0.652 [0.624, 0.679] | 0.726 [0.702, 0.748] | 0.931 [0.920, 0.943] | 0.298 [0.269, 0.334] | 0.204 [0.169, 0.258] | 0 |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | 0.987 [0.978, 0.995] | 0.993 [0.988, 0.997] | 0.979 [0.972, 0.985] | 0.600 [0.600, 0.621] | 0.600 [0.600, 0.600] | 0 |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | 0.917 [0.893, 0.939] | 0.946 [0.928, 0.962] | 0.959 [0.950, 0.968] | 0.605 [0.600, 0.618] | 0.600 [0.600, 0.601] | 0 |
| ArcFace, normal map, camera di dominio (prima tornata) | 0.981 [0.968, 0.992] | 0.989 [0.979, 0.996] | 0.978 [0.971, 0.983] | 0.601 [0.600, 0.608] | 0.600 [0.600, 0.600] | 0 |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | 0.919 [0.896, 0.943] | 0.948 [0.931, 0.965] | 0.959 [0.950, 0.967] | 0.600 [0.600, 0.602] | 0.600 [0.600, 0.600] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | TAR@FAR 0.001: delta [CI] | TAR@FAR 0.0001: delta [CI] | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- | --- |
| ICP di similarita' + NICP P2Tri | -0.002 [-0.004, -0.000] (1.000) | -0.000 [-0.000, -0.000] (1.000) | -0.031 [-0.064, -0.011] | -0.137 [-0.345, -0.059] | sotto / sotto |
| ICP di similarita' + Chamfer | -0.002 [-0.004, -0.000] (1.000) | -0.000 [-0.000, -0.000] (1.000) | -0.031 [-0.064, -0.011] | -0.134 [-0.345, -0.056] | sotto / sotto |
| NICP su template (iscrizione) | +0.142 [+0.115, +0.174] (0.000) | +0.008 [+0.005, +0.012] (0.000) | +0.356 [+0.249, +0.437] | +0.496 [+0.293, +0.640] | sopra / sopra |
| ICP rigido + NICP P2Tri (prima tornata) | -0.001 [-0.003, +0.000] (0.945) | +0.000 [-0.000, +0.000] (0.088) | +0.002 [-0.030, +0.041] | -0.031 [-0.229, +0.069] | pari / pari |
| ICP rigido + Chamfer (prima tornata) | +0.064 [+0.051, +0.077] (0.000) | +0.008 [+0.005, +0.010] (0.000) | +0.144 [+0.103, +0.180] | +0.132 [-0.062, +0.272] | sopra / sopra |
| Chamfer faceBench | +0.346 [+0.319, +0.374] (0.000) | +0.069 [+0.057, +0.080] (0.000) | +0.670 [+0.633, +0.702] | +0.659 [+0.470, +0.738] | sopra / sopra |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | +0.012 [+0.003, +0.021] (0.001) | +0.021 [+0.015, +0.028] (0.000) | +0.369 [+0.335, +0.388] | +0.263 [+0.054, +0.340] | sopra / sopra |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | +0.081 [+0.058, +0.106] (0.000) | +0.040 [+0.031, +0.050] (0.000) | +0.364 [+0.331, +0.386] | +0.263 [+0.054, +0.340] | sopra / sopra |
| ArcFace, normal map, camera di dominio (prima tornata) | +0.017 [+0.006, +0.030] (0.000) | +0.022 [+0.016, +0.028] (0.000) | +0.368 [+0.336, +0.388] | +0.263 [+0.054, +0.340] | sopra / sopra |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | +0.079 [+0.055, +0.103] (0.000) | +0.041 [+0.033, +0.050] (0.000) | +0.369 [+0.336, +0.389] | +0.263 [+0.054, +0.340] | sopra / sopra |

### a parte: crop da un lato

Retrieval: 1080 query (10 coppie ordinate x 108), galleria di 108. Verifica: 540 coppie stessa persona, 57780 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | TAR@FAR 0.001 | TAR@FAR 0.0001 | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 0.944 [0.911, 0.970] | 0.965 [0.944, 0.982] | 0.991 [0.986, 0.995] | 0.707 [0.598, 0.789] | 0.422 [0.278, 0.650] | 0 |
| ICP di similarita' + NICP P2Tri | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.996 [0.991, 1.000] | 0 |
| ICP di similarita' + Chamfer | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.996 [0.991, 1.000] | 0.994 [0.943, 1.000] | 0 |
| NICP su template (iscrizione) | 0.010 [0.000, 0.024] | 0.057 [0.044, 0.074] | 0.576 [0.552, 0.605] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 |
| ICP rigido + NICP P2Tri (prima tornata) | 0.686 [0.637, 0.736] | 0.776 [0.736, 0.816] | 0.970 [0.959, 0.979] | 0.343 [0.233, 0.465] | 0.133 [0.078, 0.315] | 0 |
| ICP rigido + Chamfer (prima tornata) | 0.106 [0.065, 0.154] | 0.181 [0.137, 0.230] | 0.651 [0.631, 0.674] | 0.057 [0.030, 0.100] | 0.033 [0.013, 0.070] | 0 |
| Chamfer faceBench | 0.044 [0.024, 0.069] | 0.099 [0.075, 0.127] | 0.568 [0.556, 0.582] | 0.004 [0.000, 0.011] | 0.000 [0.000, 0.002] | 0 |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | 0.988 [0.981, 0.994] | 0.993 [0.989, 0.997] | 0.985 [0.980, 0.989] | 0.802 [0.800, 0.807] | 0.800 [0.800, 0.800] | 0 |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | 0.956 [0.944, 0.969] | 0.971 [0.962, 0.979] | 0.970 [0.964, 0.976] | 0.802 [0.800, 0.807] | 0.800 [0.800, 0.806] | 0 |
| ArcFace, normal map, camera di dominio (prima tornata) | 0.984 [0.975, 0.993] | 0.989 [0.982, 0.995] | 0.984 [0.980, 0.988] | 0.800 [0.800, 0.806] | 0.800 [0.800, 0.800] | 0 |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | 0.947 [0.934, 0.960] | 0.965 [0.956, 0.975] | 0.966 [0.960, 0.973] | 0.800 [0.800, 0.800] | 0.800 [0.800, 0.800] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | TAR@FAR 0.001: delta [CI] | TAR@FAR 0.0001: delta [CI] | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- | --- |
| ICP di similarita' + NICP P2Tri | -0.056 [-0.089, -0.030] (1.000) | -0.009 [-0.014, -0.005] (1.000) | -0.293 [-0.402, -0.211] | -0.574 [-0.720, -0.348] | sotto / sotto |
| ICP di similarita' + Chamfer | -0.056 [-0.089, -0.030] (1.000) | -0.009 [-0.014, -0.005] (1.000) | -0.289 [-0.398, -0.207] | -0.572 [-0.715, -0.341] | sotto / sotto |
| NICP su template (iscrizione) | +0.933 [+0.897, +0.962] (0.000) | +0.415 [+0.388, +0.439] (0.000) | +0.707 [+0.598, +0.789] | +0.422 [+0.278, +0.650] | sopra / sopra |
| ICP rigido + NICP P2Tri (prima tornata) | +0.257 [+0.193, +0.320] (0.000) | +0.021 [+0.010, +0.033] (0.000) | +0.365 [+0.194, +0.508] | +0.289 [+0.033, +0.513] | sopra / sopra |
| ICP rigido + Chamfer (prima tornata) | +0.838 [+0.781, +0.888] (0.000) | +0.339 [+0.316, +0.360] (0.000) | +0.650 [+0.520, +0.743] | +0.389 [+0.237, +0.619] | sopra / sopra |
| Chamfer faceBench | +0.899 [+0.862, +0.931] (0.000) | +0.423 [+0.409, +0.435] (0.000) | +0.704 [+0.591, +0.783] | +0.422 [+0.278, +0.650] | sopra / sopra |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | -0.044 [-0.077, -0.018] (1.000) | +0.006 [-0.000, +0.013] (0.028) | -0.094 [-0.204, -0.013] | -0.378 [-0.522, -0.150] | sotto / pari |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | -0.012 [-0.049, +0.018] (0.791) | +0.021 [+0.013, +0.028] (0.000) | -0.094 [-0.202, -0.015] | -0.378 [-0.522, -0.150] | pari / sopra |
| ArcFace, normal map, camera di dominio (prima tornata) | -0.041 [-0.073, -0.014] (0.999) | +0.007 [+0.000, +0.013] (0.024) | -0.093 [-0.202, -0.013] | -0.378 [-0.522, -0.150] | sotto / sopra |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | -0.004 [-0.038, +0.024] (0.601) | +0.025 [+0.017, +0.032] (0.000) | -0.093 [-0.202, -0.011] | -0.378 [-0.522, -0.150] | pari / sopra |

## ICT, 89 held-out del congiunto

### PRIMARIO, 5 topologie senza crop

Retrieval: 1780 query (20 coppie ordinate x 89), galleria di 89. Verifica: 890 coppie stessa persona, 78320 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | TAR@FAR 0.001 | TAR@FAR 0.0001 | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.994 [0.985, 1.000] | 0 |
| ICP di similarita' + NICP P2Tri | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.992, 1.000] | 0.963 [0.931, 0.998] | 0 |
| ICP di similarita' + Chamfer | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.962 [0.934, 0.984] | 0.903 [0.875, 0.935] | 0 |
| NICP su template (iscrizione) | 0.898 [0.874, 0.921] | 0.944 [0.931, 0.956] | 0.996 [0.993, 0.998] | 0.616 [0.502, 0.736] | 0.318 [0.167, 0.472] | 0 |
| ICP rigido + NICP P2Tri (prima tornata) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.994 [0.991, 0.996] | 0.769 [0.729, 0.809] | 0.685 [0.610, 0.738] | 0 |
| ICP rigido + Chamfer (prima tornata) | 0.670 [0.654, 0.687] | 0.726 [0.712, 0.742] | 0.816 [0.798, 0.834] | 0.584 [0.573, 0.594] | 0.557 [0.518, 0.579] | 0 |
| Chamfer faceBench | 0.381 [0.347, 0.416] | 0.472 [0.444, 0.502] | 0.689 [0.675, 0.705] | 0.198 [0.192, 0.203] | 0.189 [0.182, 0.197] | 0 |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | 0.981 [0.969, 0.992] | 0.989 [0.981, 0.995] | 0.987 [0.982, 0.992] | 0.601 [0.600, 0.624] | 0.600 [0.600, 0.600] | 0 |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | 0.947 [0.924, 0.967] | 0.966 [0.951, 0.979] | 0.970 [0.962, 0.977] | 0.600 [0.600, 0.606] | 0.600 [0.600, 0.600] | 0 |
| ArcFace, normal map, camera di dominio (prima tornata) | 0.981 [0.970, 0.991] | 0.990 [0.983, 0.995] | 0.994 [0.992, 0.997] | 0.606 [0.600, 0.666] | 0.600 [0.600, 0.604] | 0 |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | 0.971 [0.956, 0.983] | 0.982 [0.972, 0.990] | 0.978 [0.972, 0.983] | 0.600 [0.600, 0.603] | 0.600 [0.600, 0.600] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | TAR@FAR 0.001: delta [CI] | TAR@FAR 0.0001: delta [CI] | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- | --- |
| ICP di similarita' + NICP P2Tri | +0.000 [+0.000, +0.000] (1.000) | +0.000 [-0.000, +0.000] (0.026) | +0.003 [+0.000, +0.008] | +0.031 [-0.006, +0.065] | pari / pari |
| ICP di similarita' + Chamfer | +0.000 [+0.000, +0.000] (1.000) | +0.000 [+0.000, +0.000] (0.000) | +0.038 [+0.016, +0.066] | +0.091 [+0.060, +0.121] | pari / sopra |
| NICP su template (iscrizione) | +0.102 [+0.079, +0.126] (0.000) | +0.004 [+0.002, +0.007] (0.000) | +0.384 [+0.264, +0.498] | +0.676 [+0.525, +0.831] | sopra / sopra |
| ICP rigido + NICP P2Tri (prima tornata) | +0.000 [+0.000, +0.000] (1.000) | +0.006 [+0.004, +0.009] (0.000) | +0.231 [+0.191, +0.271] | +0.309 [+0.257, +0.384] | pari / sopra |
| ICP rigido + Chamfer (prima tornata) | +0.330 [+0.313, +0.346] (0.000) | +0.184 [+0.166, +0.202] (0.000) | +0.416 [+0.404, +0.427] | +0.437 [+0.415, +0.480] | sopra / sopra |
| Chamfer faceBench | +0.619 [+0.584, +0.653] (0.000) | +0.311 [+0.295, +0.325] (0.000) | +0.802 [+0.796, +0.808] | +0.806 [+0.794, +0.815] | sopra / sopra |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | +0.019 [+0.008, +0.031] (0.000) | +0.013 [+0.008, +0.018] (0.000) | +0.399 [+0.376, +0.400] | +0.394 [+0.385, +0.400] | sopra / sopra |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | +0.053 [+0.033, +0.076] (0.000) | +0.030 [+0.023, +0.038] (0.000) | +0.400 [+0.394, +0.400] | +0.394 [+0.385, +0.400] | sopra / sopra |
| ArcFace, normal map, camera di dominio (prima tornata) | +0.019 [+0.009, +0.030] (0.000) | +0.006 [+0.003, +0.008] (0.000) | +0.394 [+0.334, +0.400] | +0.394 [+0.384, +0.400] | sopra / sopra |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | +0.029 [+0.017, +0.044] (0.000) | +0.022 [+0.017, +0.028] (0.000) | +0.400 [+0.396, +0.400] | +0.394 [+0.385, +0.400] | sopra / sopra |

### a parte: crop da un lato

Retrieval: 890 query (10 coppie ordinate x 89), galleria di 89. Verifica: 445 coppie stessa persona, 39160 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | TAR@FAR 0.001 | TAR@FAR 0.0001 | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 0.993 [0.980, 1.000] | 0.996 [0.989, 1.000] | 1.000 [1.000, 1.000] | 0.978 [0.937, 1.000] | 0.933 [0.874, 0.975] | 0 |
| ICP di similarita' + NICP P2Tri | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| ICP di similarita' + Chamfer | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.993, 1.000] | 0.993 [0.987, 1.000] | 0 |
| NICP su template (iscrizione) | 0.045 [0.019, 0.079] | 0.120 [0.094, 0.151] | 0.711 [0.680, 0.743] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 |
| ICP rigido + NICP P2Tri (prima tornata) | 0.600 [0.554, 0.647] | 0.698 [0.659, 0.737] | 0.962 [0.951, 0.973] | 0.330 [0.292, 0.387] | 0.267 [0.220, 0.328] | 0 |
| ICP rigido + Chamfer (prima tornata) | 0.306 [0.273, 0.342] | 0.405 [0.372, 0.441] | 0.833 [0.809, 0.860] | 0.189 [0.169, 0.211] | 0.171 [0.153, 0.193] | 0 |
| Chamfer faceBench | 0.097 [0.067, 0.131] | 0.228 [0.199, 0.263] | 0.708 [0.693, 0.724] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | 0.980 [0.971, 0.989] | 0.988 [0.982, 0.993] | 0.988 [0.984, 0.991] | 0.800 [0.800, 0.811] | 0.800 [0.800, 0.800] | 0 |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | 0.969 [0.954, 0.981] | 0.980 [0.970, 0.988] | 0.975 [0.970, 0.980] | 0.800 [0.800, 0.800] | 0.800 [0.800, 0.800] | 0 |
| ArcFace, normal map, camera di dominio (prima tornata) | 0.993 [0.988, 0.998] | 0.995 [0.991, 0.999] | 0.995 [0.993, 0.997] | 0.800 [0.800, 0.818] | 0.800 [0.800, 0.800] | 0 |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | 0.981 [0.971, 0.990] | 0.988 [0.981, 0.994] | 0.985 [0.981, 0.989] | 0.800 [0.800, 0.800] | 0.800 [0.800, 0.800] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | TAR@FAR 0.001: delta [CI] | TAR@FAR 0.0001: delta [CI] | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- | --- |
| ICP di similarita' + NICP P2Tri | -0.007 [-0.020, +0.000] (1.000) | -0.000 [-0.000, -0.000] (1.000) | -0.022 [-0.063, +0.000] | -0.067 [-0.126, -0.025] | pari / sotto |
| ICP di similarita' + Chamfer | -0.007 [-0.020, +0.000] (1.000) | -0.000 [-0.000, -0.000] (0.999) | -0.022 [-0.061, +0.000] | -0.061 [-0.121, -0.020] | pari / sotto |
| NICP su template (iscrizione) | +0.948 [+0.913, +0.978] (0.000) | +0.288 [+0.257, +0.320] (0.000) | +0.978 [+0.937, +1.000] | +0.933 [+0.874, +0.975] | sopra / sopra |
| ICP rigido + NICP P2Tri (prima tornata) | +0.393 [+0.338, +0.442] (0.000) | +0.037 [+0.027, +0.049] (0.000) | +0.647 [+0.584, +0.690] | +0.665 [+0.582, +0.730] | sopra / sopra |
| ICP rigido + Chamfer (prima tornata) | +0.688 [+0.648, +0.721] (0.000) | +0.167 [+0.140, +0.191] (0.000) | +0.789 [+0.751, +0.816] | +0.762 [+0.701, +0.802] | sopra / sopra |
| Chamfer faceBench | +0.897 [+0.861, +0.929] (0.000) | +0.292 [+0.276, +0.306] (0.000) | +0.978 [+0.937, +1.000] | +0.933 [+0.874, +0.975] | sopra / sopra |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | +0.013 [-0.006, +0.028] (0.086) | +0.012 [+0.008, +0.016] (0.000) | +0.178 [+0.135, +0.200] | +0.133 [+0.074, +0.175] | pari / sopra |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | +0.025 [+0.003, +0.044] (0.017) | +0.025 [+0.020, +0.030] (0.000) | +0.178 [+0.137, +0.200] | +0.133 [+0.074, +0.175] | sopra / sopra |
| ArcFace, normal map, camera di dominio (prima tornata) | +0.000 [-0.015, +0.009] (0.507) | +0.005 [+0.003, +0.006] (0.000) | +0.178 [+0.130, +0.200] | +0.133 [+0.074, +0.175] | pari / sopra |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | +0.012 [-0.004, +0.026] (0.067) | +0.015 [+0.011, +0.019] (0.000) | +0.178 [+0.137, +0.200] | +0.133 [+0.074, +0.175] | pari / sopra |

## ICT con espressioni casuali, 89 held-out del congiunto

### PRIMARIO, espressione contro espressione (k != k')

Retrieval: 1780 query (20 coppie ordinate x 89), galleria di 89. Verifica: 890 coppie stessa persona, 78320 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | TAR@FAR 0.001 | TAR@FAR 0.0001 | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 0.822 [0.785, 0.860] | 0.865 [0.834, 0.895] | 0.978 [0.968, 0.985] | 0.670 [0.611, 0.730] | 0.590 [0.508, 0.669] | 0 |
| NICP su template (iscrizione) | 0.388 [0.339, 0.436] | 0.515 [0.468, 0.559] | 0.920 [0.901, 0.937] | 0.113 [0.065, 0.169] | 0.019 [0.006, 0.051] | 0 |
| ICP rigido + NICP P2Tri (prima tornata) | 0.905 [0.878, 0.929] | 0.929 [0.906, 0.949] | 0.977 [0.966, 0.986] | 0.745 [0.666, 0.806] | 0.609 [0.484, 0.730] | 0 |
| ICP rigido + Chamfer (prima tornata) | 0.896 [0.867, 0.922] | 0.923 [0.899, 0.944] | 0.981 [0.972, 0.989] | 0.717 [0.654, 0.785] | 0.630 [0.553, 0.707] | 0 |
| Chamfer faceBench | 0.867 [0.835, 0.898] | 0.901 [0.876, 0.926] | 0.979 [0.971, 0.987] | 0.696 [0.626, 0.771] | 0.601 [0.534, 0.691] | 0 |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.992, 1.000] | 0 |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.997, 1.000] | 0 |
| ArcFace, normal map, camera di dominio (prima tornata) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.998 [0.992, 1.000] | 0 |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.999 [0.994, 1.000] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | TAR@FAR 0.001: delta [CI] | TAR@FAR 0.0001: delta [CI] | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- | --- |
| NICP su template (iscrizione) | +0.434 [+0.394, +0.478] (0.000) | +0.058 [+0.042, +0.074] (0.000) | +0.556 [+0.483, +0.631] | +0.571 [+0.483, +0.647] | sopra / sopra |
| ICP rigido + NICP P2Tri (prima tornata) | -0.083 [-0.110, -0.055] (1.000) | +0.001 [-0.007, +0.011] (0.404) | -0.075 [-0.120, -0.004] | -0.019 [-0.139, +0.123] | sotto / pari |
| ICP rigido + Chamfer (prima tornata) | -0.074 [-0.098, -0.050] (1.000) | -0.003 [-0.010, +0.005] (0.812) | -0.047 [-0.099, +0.001] | -0.040 [-0.113, +0.038] | sotto / pari |
| Chamfer faceBench | -0.046 [-0.069, -0.021] (1.000) | -0.001 [-0.006, +0.004] (0.709) | -0.026 [-0.080, +0.024] | -0.011 [-0.099, +0.047] | sotto / pari |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | -0.178 [-0.215, -0.140] (1.000) | -0.022 [-0.032, -0.015] (1.000) | -0.330 [-0.389, -0.270] | -0.409 [-0.489, -0.327] | sotto / sotto |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | -0.178 [-0.215, -0.140] (1.000) | -0.022 [-0.032, -0.015] (1.000) | -0.330 [-0.389, -0.270] | -0.409 [-0.489, -0.330] | sotto / sotto |
| ArcFace, normal map, camera di dominio (prima tornata) | -0.178 [-0.215, -0.140] (1.000) | -0.022 [-0.032, -0.015] (1.000) | -0.330 [-0.389, -0.270] | -0.408 [-0.489, -0.327] | sotto / sotto |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | -0.178 [-0.215, -0.140] (1.000) | -0.022 [-0.032, -0.015] (1.000) | -0.330 [-0.389, -0.270] | -0.409 [-0.489, -0.330] | sotto / sotto |

### secondario: galleria neutra, query con espressione

Retrieval: 445 query (5 coppie ordinate x 89), galleria di 89. Verifica: 445 coppie stessa persona, 39160 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | TAR@FAR 0.001 | TAR@FAR 0.0001 | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 0.962 [0.942, 0.980] | 0.976 [0.962, 0.987] | 0.992 [0.987, 0.995] | 0.800 [0.753, 0.847] | 0.735 [0.654, 0.807] | 0 |
| NICP su template (iscrizione) | 0.571 [0.519, 0.622] | 0.676 [0.635, 0.718] | 0.954 [0.942, 0.964] | 0.265 [0.148, 0.344] | 0.067 [0.034, 0.135] | 0 |
| ICP rigido + NICP P2Tri (prima tornata) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.995, 0.999] | 0.874 [0.796, 0.917] | 0.764 [0.710, 0.874] | 0 |
| ICP rigido + Chamfer (prima tornata) | 0.998 [0.993, 1.000] | 0.999 [0.997, 1.000] | 0.997 [0.995, 0.999] | 0.849 [0.813, 0.903] | 0.778 [0.733, 0.858] | 0 |
| Chamfer faceBench | 0.998 [0.993, 1.000] | 0.999 [0.996, 1.000] | 0.994 [0.990, 0.997] | 0.834 [0.771, 0.872] | 0.766 [0.706, 0.836] | 0 |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| ArcFace, normal map, camera di dominio (prima tornata) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | TAR@FAR 0.001: delta [CI] | TAR@FAR 0.0001: delta [CI] | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- | --- |
| NICP su template (iscrizione) | +0.391 [+0.342, +0.447] (0.000) | +0.038 [+0.028, +0.049] (0.000) | +0.535 [+0.449, +0.658] | +0.667 [+0.560, +0.751] | sopra / sopra |
| ICP rigido + NICP P2Tri (prima tornata) | -0.038 [-0.058, -0.020] (1.000) | -0.005 [-0.009, -0.002] (1.000) | -0.074 [-0.117, -0.004] | -0.029 [-0.171, +0.047] | sotto / sotto |
| ICP rigido + Chamfer (prima tornata) | -0.036 [-0.056, -0.018] (1.000) | -0.005 [-0.009, -0.002] (1.000) | -0.049 [-0.103, -0.013] | -0.043 [-0.153, +0.029] | sotto / sotto |
| Chamfer faceBench | -0.036 [-0.056, -0.020] (1.000) | -0.002 [-0.005, +0.001] (0.883) | -0.034 [-0.065, +0.018] | -0.031 [-0.103, +0.034] | sotto / pari |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | -0.038 [-0.058, -0.020] (1.000) | -0.008 [-0.013, -0.005] (1.000) | -0.200 [-0.247, -0.153] | -0.265 [-0.346, -0.193] | sotto / sotto |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | -0.038 [-0.058, -0.020] (1.000) | -0.008 [-0.013, -0.005] (1.000) | -0.200 [-0.247, -0.153] | -0.265 [-0.346, -0.193] | sotto / sotto |
| ArcFace, normal map, camera di dominio (prima tornata) | -0.038 [-0.058, -0.020] (1.000) | -0.008 [-0.013, -0.005] (1.000) | -0.200 [-0.247, -0.153] | -0.265 [-0.346, -0.193] | sotto / sotto |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | -0.038 [-0.058, -0.020] (1.000) | -0.008 [-0.013, -0.005] (1.000) | -0.200 [-0.247, -0.153] | -0.265 [-0.346, -0.193] | sotto / sotto |

## BFM-19, held-out di congiunto E BFM-only

### PRIMARIO, 5 topologie senza crop

Retrieval: 380 query (20 coppie ordinate x 19), galleria di 19. Verifica: 190 coppie stessa persona, 3420 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | TAR@FAR 0.001 | TAR@FAR 0.0001 | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.968 [0.932, 1.000] | 0.953 [0.926, 1.000] | 0 |
| BFM-only (1019310) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| ICP di similarita' + NICP P2Tri | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| ICP di similarita' + Chamfer | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| NICP su template (iscrizione) | 0.955 [0.916, 0.987] | 0.977 [0.957, 0.993] | 0.994 [0.989, 0.999] | 0.700 [0.521, 0.953] | 0.542 [0.447, 0.942] | 0 |
| ICP rigido + NICP P2Tri (prima tornata) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.998, 1.000] | 0.905 [0.853, 0.995] | 0.868 [0.842, 0.989] | 0 |
| ICP rigido + Chamfer (prima tornata) | 0.945 [0.921, 0.966] | 0.965 [0.950, 0.978] | 0.985 [0.975, 0.993] | 0.774 [0.689, 0.858] | 0.732 [0.684, 0.837] | 0 |
| Chamfer faceBench | 0.750 [0.708, 0.797] | 0.832 [0.803, 0.862] | 0.917 [0.888, 0.941] | 0.263 [0.211, 0.432] | 0.226 [0.200, 0.368] | 0 |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.962 [0.932, 0.984] | 0.600 [0.600, 0.700] | 0.600 [0.600, 0.690] | 0 |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | 0.939 [0.887, 0.982] | 0.966 [0.935, 0.991] | 0.924 [0.889, 0.954] | 0.600 [0.600, 0.600] | 0.600 [0.600, 0.600] | 0 |
| ArcFace, normal map, camera di dominio (prima tornata) | 0.992 [0.976, 1.000] | 0.996 [0.988, 1.000] | 0.955 [0.929, 0.976] | 0.600 [0.600, 0.611] | 0.600 [0.600, 0.600] | 0 |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | 0.950 [0.911, 0.984] | 0.973 [0.952, 0.992] | 0.927 [0.897, 0.952] | 0.600 [0.600, 0.611] | 0.600 [0.600, 0.611] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | TAR@FAR 0.001: delta [CI] | TAR@FAR 0.0001: delta [CI] | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- | --- |
| BFM-only (1019310) | +0.000 [+0.000, +0.000] (1.000) | -0.000 [-0.000, +0.000] (1.000) | -0.032 [-0.068, +0.000] | -0.047 [-0.074, +0.000] | pari / pari |
| ICP di similarita' + NICP P2Tri | +0.000 [+0.000, +0.000] (1.000) | -0.000 [-0.000, +0.000] (1.000) | -0.032 [-0.068, +0.000] | -0.047 [-0.074, +0.000] | pari / pari |
| ICP di similarita' + Chamfer | +0.000 [+0.000, +0.000] (1.000) | -0.000 [-0.000, +0.000] (1.000) | -0.032 [-0.068, +0.000] | -0.047 [-0.074, +0.000] | pari / pari |
| NICP su template (iscrizione) | +0.045 [+0.013, +0.084] (0.003) | +0.006 [+0.001, +0.011] (0.000) | +0.268 [+0.021, +0.458] | +0.411 [+0.021, +0.521] | sopra / sopra |
| ICP rigido + NICP P2Tri (prima tornata) | +0.000 [+0.000, +0.000] (1.000) | +0.001 [-0.000, +0.002] (0.078) | +0.063 [-0.042, +0.142] | +0.084 [-0.037, +0.147] | pari / pari |
| ICP rigido + Chamfer (prima tornata) | +0.055 [+0.034, +0.079] (0.000) | +0.015 [+0.007, +0.025] (0.000) | +0.195 [+0.105, +0.300] | +0.221 [+0.116, +0.300] | sopra / sopra |
| Chamfer faceBench | +0.250 [+0.203, +0.292] (0.000) | +0.083 [+0.059, +0.112] (0.000) | +0.705 [+0.558, +0.774] | +0.726 [+0.595, +0.789] | sopra / sopra |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | +0.000 [+0.000, +0.000] (1.000) | +0.038 [+0.016, +0.068] (0.000) | +0.368 [+0.274, +0.400] | +0.353 [+0.274, +0.400] | pari / sopra |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | +0.061 [+0.018, +0.113] (0.004) | +0.075 [+0.046, +0.111] (0.000) | +0.368 [+0.332, +0.400] | +0.353 [+0.326, +0.400] | sopra / sopra |
| ArcFace, normal map, camera di dominio (prima tornata) | +0.008 [+0.000, +0.024] (0.372) | +0.045 [+0.024, +0.071] (0.000) | +0.368 [+0.326, +0.400] | +0.353 [+0.321, +0.400] | pari / sopra |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | +0.050 [+0.016, +0.089] (0.000) | +0.073 [+0.048, +0.103] (0.000) | +0.368 [+0.332, +0.400] | +0.353 [+0.321, +0.400] | sopra / sopra |

### a parte: crop da un lato

Retrieval: 190 query (10 coppie ordinate x 19), galleria di 19. Verifica: 95 coppie stessa persona, 1710 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | TAR@FAR 0.001 | TAR@FAR 0.0001 | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 0.979 [0.953, 1.000] | 0.989 [0.976, 1.000] | 0.995 [0.988, 1.000] | 0.842 [0.705, 0.968] | 0.800 [0.695, 0.958] | 0 |
| BFM-only (1019310) | 0.942 [0.863, 1.000] | 0.970 [0.930, 1.000] | 0.990 [0.977, 1.000] | 0.800 [0.653, 0.947] | 0.758 [0.632, 0.937] | 0 |
| ICP di similarita' + NICP P2Tri | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| ICP di similarita' + Chamfer | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.979, 1.000] | 0.989 [0.979, 1.000] | 0 |
| NICP su template (iscrizione) | 0.053 [0.000, 0.132] | 0.214 [0.160, 0.278] | 0.549 [0.501, 0.612] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 |
| ICP rigido + NICP P2Tri (prima tornata) | 0.911 [0.832, 0.974] | 0.951 [0.906, 0.986] | 0.981 [0.964, 0.992] | 0.411 [0.305, 0.663] | 0.411 [0.284, 0.621] | 0 |
| ICP rigido + Chamfer (prima tornata) | 0.184 [0.095, 0.268] | 0.348 [0.267, 0.431] | 0.664 [0.624, 0.704] | 0.011 [0.000, 0.084] | 0.011 [0.000, 0.074] | 0 |
| Chamfer faceBench | 0.105 [0.026, 0.190] | 0.253 [0.180, 0.326] | 0.554 [0.530, 0.594] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.970 [0.948, 0.987] | 0.800 [0.800, 0.811] | 0.800 [0.800, 0.800] | 0 |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | 0.958 [0.932, 0.984] | 0.976 [0.960, 0.991] | 0.944 [0.923, 0.965] | 0.800 [0.800, 0.800] | 0.800 [0.800, 0.800] | 0 |
| ArcFace, normal map, camera di dominio (prima tornata) | 0.989 [0.974, 1.000] | 0.994 [0.984, 1.000] | 0.968 [0.947, 0.984] | 0.800 [0.800, 0.800] | 0.800 [0.800, 0.800] | 0 |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | 0.953 [0.926, 0.979] | 0.971 [0.952, 0.987] | 0.941 [0.920, 0.962] | 0.800 [0.800, 0.800] | 0.800 [0.800, 0.800] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | TAR@FAR 0.001: delta [CI] | TAR@FAR 0.0001: delta [CI] | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- | --- |
| BFM-only (1019310) | +0.037 [-0.005, +0.095] (0.078) | +0.005 [-0.002, +0.013] (0.096) | +0.042 [-0.053, +0.168] | +0.042 [-0.042, +0.158] | pari / pari |
| ICP di similarita' + NICP P2Tri | -0.021 [-0.047, +0.000] (1.000) | -0.005 [-0.012, -0.000] (1.000) | -0.158 [-0.295, -0.032] | -0.200 [-0.305, -0.042] | pari / sotto |
| ICP di similarita' + Chamfer | -0.021 [-0.047, +0.000] (1.000) | -0.005 [-0.012, -0.000] (1.000) | -0.158 [-0.295, -0.032] | -0.189 [-0.305, -0.032] | pari / sotto |
| NICP su template (iscrizione) | +0.926 [+0.847, +0.989] (0.000) | +0.446 [+0.384, +0.494] (0.000) | +0.842 [+0.705, +0.968] | +0.800 [+0.695, +0.958] | sopra / sopra |
| ICP rigido + NICP P2Tri (prima tornata) | +0.068 [+0.000, +0.153] (0.031) | +0.014 [+0.002, +0.032] (0.009) | +0.432 [+0.158, +0.600] | +0.389 [+0.168, +0.600] | pari / sopra |
| ICP rigido + Chamfer (prima tornata) | +0.795 [+0.705, +0.884] (0.000) | +0.331 [+0.292, +0.370] (0.000) | +0.832 [+0.694, +0.947] | +0.789 [+0.684, +0.947] | sopra / sopra |
| Chamfer faceBench | +0.874 [+0.784, +0.958] (0.000) | +0.441 [+0.403, +0.466] (0.000) | +0.842 [+0.705, +0.968] | +0.800 [+0.695, +0.958] | sopra / sopra |
| ArcFace migliore: normal map smussata, inquadratura per mesh, 3 viste | -0.021 [-0.047, +0.000] (1.000) | +0.025 [+0.009, +0.046] (0.000) | +0.042 [-0.095, +0.168] | +0.000 [-0.105, +0.158] | pari / sopra |
| ArcFace, ombreggiato, inquadratura per mesh (secondaria) | +0.021 [-0.016, +0.058] (0.167) | +0.051 [+0.032, +0.071] (0.000) | +0.042 [-0.095, +0.168] | +0.000 [-0.105, +0.158] | pari / sopra |
| ArcFace, normal map, camera di dominio (prima tornata) | -0.011 [-0.037, +0.016] (0.812) | +0.027 [+0.012, +0.048] (0.000) | +0.042 [-0.095, +0.168] | +0.000 [-0.105, +0.158] | pari / sopra |
| ArcFace, ombreggiato, camera di dominio (prima tornata, secondaria) | +0.026 [-0.011, +0.063] (0.119) | +0.054 [+0.036, +0.073] (0.000) | +0.042 [-0.095, +0.168] | +0.000 [-0.105, +0.158] | pari / sopra |

## ICT, galleria grande: 992 held-out del congiunto

### PRIMARIO della galleria grande: 100 query in remesh, galleria di 992 in original

Retrieval: 100 query in `remesh`, galleria di 992 in `original`. Verifica: 100 coppie stessa persona, 99100 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | TAR@FAR 0.001 | TAR@FAR 0.0001 | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| ICP di similarita' + NICP P2Tri | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| NICP su template (iscrizione) | 0.820 [0.740, 0.900] | 0.891 [0.835, 0.936] | 0.997 [0.993, 1.000] | 0.880 [0.770, 0.940] | 0.330 [0.110, 0.550] | 0 |
| Chamfer faceBench | 0.160 [0.100, 0.240] | 0.260 [0.194, 0.332] | 0.931 [0.916, 0.945] | 0.100 [0.050, 0.150] | 0.020 [0.010, 0.080] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | TAR@FAR 0.001: delta [CI] | TAR@FAR 0.0001: delta [CI] | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- | --- |
| ICP di similarita' + NICP P2Tri | +0.000 [+0.000, +0.000] (1.000) | +0.000 [+0.000, +0.000] (1.000) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | pari / pari |
| NICP su template (iscrizione) | +0.180 [+0.100, +0.260] (0.000) | +0.003 [+0.000, +0.007] (0.000) | +0.120 [+0.060, +0.230] | +0.670 [+0.450, +0.890] | sopra / sopra |
| Chamfer faceBench | +0.840 [+0.760, +0.900] (0.000) | +0.069 [+0.055, +0.084] (0.000) | +0.900 [+0.850, +0.950] | +0.980 [+0.920, +0.990] | sopra / sopra |

### galleria grande, blocco degenere (noisy = original perturbata): 100 query in noisy, 992 in original

Retrieval: 100 query in `noisy`, galleria di 992 in `original`. Verifica: 100 coppie stessa persona, 99100 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | TAR@FAR 0.001 | TAR@FAR 0.0001 | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| ICP di similarita' + NICP P2Tri | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| NICP su template (iscrizione) | 0.340 [0.250, 0.430] | 0.528 [0.464, 0.597] | 0.992 [0.985, 0.996] | 0.330 [0.240, 0.450] | 0.110 [0.040, 0.170] | 0 |
| Chamfer faceBench | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | TAR@FAR 0.001: delta [CI] | TAR@FAR 0.0001: delta [CI] | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- | --- |
| ICP di similarita' + NICP P2Tri | +0.000 [+0.000, +0.000] (1.000) | +0.000 [+0.000, +0.000] (1.000) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | pari / pari |
| NICP su template (iscrizione) | +0.660 [+0.570, +0.750] (0.000) | +0.008 [+0.004, +0.015] (0.000) | +0.670 [+0.550, +0.760] | +0.890 [+0.830, +0.960] | sopra / sopra |
| Chamfer faceBench | +0.000 [+0.000, +0.000] (1.000) | +0.000 [+0.000, +0.000] (1.000) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | pari / pari |

### a parte: 100 query in crop, galleria di 992 in original

Retrieval: 100 query in `crop`, galleria di 992 in `original`. Verifica: 100 coppie stessa persona, 99100 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | TAR@FAR 0.001 | TAR@FAR 0.0001 | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.980 [0.940, 1.000] | 0.970 [0.920, 1.000] | 0 |
| ICP di similarita' + NICP P2Tri | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| NICP su template (iscrizione) | 0.000 [0.000, 0.000] | 0.019 [0.012, 0.029] | 0.714 [0.686, 0.743] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 |
| Chamfer faceBench | 0.030 [0.000, 0.060] | 0.073 [0.040, 0.109] | 0.807 [0.780, 0.835] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | TAR@FAR 0.001: delta [CI] | TAR@FAR 0.0001: delta [CI] | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- | --- |
| ICP di similarita' + NICP P2Tri | +0.000 [+0.000, +0.000] (1.000) | -0.000 [-0.001, -0.000] (1.000) | -0.020 [-0.060, +0.000] | -0.030 [-0.080, +0.000] | pari / sotto |
| NICP su template (iscrizione) | +1.000 [+1.000, +1.000] (0.000) | +0.286 [+0.256, +0.314] (0.000) | +0.980 [+0.940, +1.000] | +0.970 [+0.920, +1.000] | sopra / sopra |
| Chamfer faceBench | +0.970 [+0.940, +1.000] (0.000) | +0.193 [+0.165, +0.220] (0.000) | +0.980 [+0.940, +1.000] | +0.970 [+0.920, +1.000] | sopra / sopra |

### secondario: tutte le 992 query, 20 coppie di topologie senza crop (solo metodi a iscrizione)

Retrieval: 19840 query (20 coppie ordinate x 992), galleria di 992. Verifica: 9920 coppie stessa persona, 9830720 persone diverse. 200 repliche bootstrap.

| metodo | rank-1 | mAP | AUC verifica | TAR@FAR 0.001 | TAR@FAR 0.0001 | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | 0.995 [0.992, 0.997] | 0 |
| NICP su template (iscrizione) | 0.582 [0.569, 0.595] | 0.709 [0.700, 0.719] | 0.995 [0.994, 0.996] | 0.593 [0.574, 0.613] | 0.220 [0.198, 0.241] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | TAR@FAR 0.001: delta [CI] | TAR@FAR 0.0001: delta [CI] | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- | --- |
| NICP su template (iscrizione) | +0.418 [+0.405, +0.431] (0.000) | +0.005 [+0.004, +0.006] (0.000) | +0.407 [+0.387, +0.426] | +0.775 [+0.755, +0.798] | sopra / sopra |

### secondario: tutte le 992 query, coppie con crop (solo metodi a iscrizione)

Retrieval: 9920 query (10 coppie ordinate x 992), galleria di 992. Verifica: 4960 coppie stessa persona, 4915360 persone diverse. 200 repliche bootstrap.

| metodo | rank-1 | mAP | AUC verifica | TAR@FAR 0.001 | TAR@FAR 0.0001 | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 0.996 [0.994, 0.998] | 0.998 [0.996, 0.999] | 1.000 [1.000, 1.000] | 0.980 [0.973, 0.986] | 0.926 [0.909, 0.939] | 0 |
| NICP su template (iscrizione) | 0.004 [0.001, 0.007] | 0.018 [0.015, 0.021] | 0.703 [0.693, 0.713] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | TAR@FAR 0.001: delta [CI] | TAR@FAR 0.0001: delta [CI] | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- | --- |
| NICP su template (iscrizione) | +0.992 [+0.989, +0.995] (0.000) | +0.297 [+0.286, +0.306] (0.000) | +0.980 [+0.973, +0.986] | +0.926 [+0.909, +0.939] | sopra / sopra |

## Tempi

- model: host `a768-l40s-06.srv.aau.dk`, CPU AMD EPYC 9454 48-Core Processor, GPU NVIDIA L40S, job 1060419, OMP_NUM_THREADS=1
- facebench: host `a768-l40s-06.srv.aau.dk`, CPU AMD EPYC 9454 48-Core Processor, job 1060419, OMP_NUM_THREADS=1
- template: host `a768-l40s-06.srv.aau.dk`, CPU AMD EPYC 9454 48-Core Processor, job 1060419, OMP_NUM_THREADS=1
- arcface: host `a768-l40s-06.srv.aau.dk`, CPU AMD EPYC 9454 48-Core Processor, job 1060419, OMP_NUM_THREADS=1
- campione: 60 mesh, vertici da 3275 a 60435 (mediana 12955)

### iscrizione (per mesh)

| voce | n | mediana | IQR (25-75%) |
| --- | --- | --- | --- |
| congiunto: operatori DiffusionNet (k=128), CPU 1 thread | 60 | 2.03 s | 1.23 s - 4.33 s |
| congiunto: embedding (lettura + forward), GPU | 60 | 54.96 ms | 39.45 ms - 94.24 ms |
| congiunto: totale (operatori CPU + embedding GPU) | 60 | 2.08 s | 1.27 s - 4.42 s |
| congiunto: embedding (lettura + forward), CPU 1 thread | 60 | 401.24 ms | 264.78 ms - 859.73 ms |
| NICP su template: ICP di similarita' + NICP + Procrustes, CPU 1 thread | 60 | 1.15 s | 1.13 s - 1.31 s |
| ArcFace migliore: inquadratura + 3 render + 3 embedding, CPU 1 thread | 60 | 570.08 ms | 503.23 ms - 709.34 ms |
|   di cui inquadratura + 3 render | 60 | 228.71 ms | 160.85 ms - 367.66 ms |
| ArcFace prima tornata (camera di dominio), CPU 1 thread | 60 | 522.28 ms | 469.24 ms - 626.49 ms |

### ricerca su galleria iscritta (per query)

| voce | n | mediana | IQR (25-75%) |
| --- | --- | --- | --- |
| congiunto: un confronto ||z_a - z_b|| (1:1), CPU | 1000 | 2.4 us | 2.4 us - 2.5 us |
| congiunto 1:100 (256-d, distanze + argsort), CPU 1 thread | 100 | 19.8 us | 19.6 us - 19.9 us |
| congiunto 1:100 (256-d, distanze + argsort), GPU | 100 | 40.2 us | 39.9 us - 40.5 us |
| congiunto 1:1,000 (256-d, distanze + argsort), CPU 1 thread | 100 | 188.3 us | 187.1 us - 189.1 us |
| congiunto 1:1,000 (256-d, distanze + argsort), GPU | 100 | 50.0 us | 49.7 us - 50.2 us |
| congiunto 1:10,000 (256-d, distanze + argsort), CPU 1 thread | 100 | 1.96 ms | 1.95 ms - 1.96 ms |
| congiunto 1:10,000 (256-d, distanze + argsort), GPU | 100 | 68.0 us | 67.7 us - 68.4 us |
| NICP su template 1:100 (4096 x 3, L2 media per vertice), CPU 1 thread | 100 | 7.33 ms | 7.32 ms - 7.34 ms |
| NICP su template 1:1,000 (4096 x 3, L2 media per vertice), CPU 1 thread | 100 | 81.19 ms | 81.16 ms - 81.24 ms |
| NICP su template 1:10,000 (4096 x 3, L2 media per vertice), CPU 1 thread | 100 | 811.30 ms | 810.68 ms - 811.90 ms |
| ArcFace 1:100 (512-d, coseno), CPU 1 thread | 100 | 8.1 us | 8.0 us - 8.2 us |
| ArcFace 1:1,000 (512-d, coseno), CPU 1 thread | 100 | 76.3 us | 75.7 us - 77.1 us |
| ArcFace 1:10,000 (512-d, coseno), CPU 1 thread | 100 | 884.9 us | 873.7 us - 898.2 us |
| ICP di similarita' + Chamfer 1:100: STIMA = N x tempo per coppia | 60 | 4.52 s | - |
| ICP di similarita' + Chamfer 1:1,000: STIMA = N x tempo per coppia | 60 | 45.24 s | - |
| ICP di similarita' + Chamfer 1:10,000: STIMA = N x tempo per coppia | 60 | 452.35 s | - |
| ICP di similarita' + NICP P2Tri 1:100: STIMA = N x tempo per coppia | 60 | 128.61 s | - |
| ICP di similarita' + NICP P2Tri 1:1,000: STIMA = N x tempo per coppia | 60 | 1286.06 s | - |
| ICP di similarita' + NICP P2Tri 1:10,000: STIMA = N x tempo per coppia | 60 | 3.6 h | - |

### a coppie (per coppia)

| voce | n | mediana | IQR (25-75%) |
| --- | --- | --- | --- |
| ICP di similarita' + Chamfer, CPU 1 thread | 60 | 45.24 ms | 39.49 ms - 50.16 ms |
| ICP di similarita' + NICP P2Tri, CPU 1 thread | 60 | 1.29 s | 1.25 s - 1.37 s |
| ICP rigido + Chamfer (prima tornata), CPU 1 thread | 60 | 33.60 ms | 29.84 ms - 41.19 ms |
| ICP rigido + NICP P2Tri (prima tornata), CPU 1 thread | 60 | 1.35 s | 1.30 s - 1.72 s |

## Controlli

- BFM, 108 held-out del congiunto, BFM+ICT congiunto: max |diff| contro `latent_distance` WS2 = 3.14e-03 su 173340 coppie (mediana di latent_distance 0.763)
- BFM, 108 held-out del congiunto: `chamfer_sim` (variante sim) contro `chamfer` (prima tornata), max |diff| = 0.00e+00 su 349920 distanze
- ICT, 89 held-out del congiunto, BFM+ICT congiunto: max |diff| contro `latent_distance` WS2 = 2.28e-03 su 117480 coppie (mediana di latent_distance 1.173)
- ICT, 89 held-out del congiunto: `chamfer_sim` (variante sim) contro `chamfer` (prima tornata), max |diff| = 0.00e+00 su 237630 distanze
- ICT con espressioni casuali, 89 held-out del congiunto, BFM+ICT congiunto: max |diff| contro `latent_distance` WS2 = 2.11e-03 su 82236 coppie (mediana di latent_distance 1.266)
- BFM-19, held-out di congiunto E BFM-only, BFM+ICT congiunto: max |diff| contro `latent_distance` WS2 = 2.32e-03 su 5130 coppie (mediana di latent_distance 0.750)
- BFM-19, held-out di congiunto E BFM-only, BFM-only (1019310): max |diff| contro `latent_distance` WS2 = 8.62e-04 su 5130 coppie (mediana di latent_distance 0.509)
- faceBench: NICP asimmetrico, orientazione della coppia = ordine delle etichette (insiemi quadrati) o query -> galleria (ict992), non sempre query -> galleria.
- leak (`sets.json`): {"bfm": {"n": 108, "in_joint_train": 0, "in_bfm_only_train": 89, "in_joint_online_eval": 16, "in_bfm_only_online_eval": 2}, "ict": {"n": 89, "in_joint_train": 0, "in_bfm_only_train": 0, "in_joint_online_eval": 0, "in_bfm_only_online_eval": 0}, "rexpr": {"n": 89, "in_joint_train": 0, "in_bfm_only_train": 0, "in_joint_online_eval": 0, "in_bfm_only_online_eval": 0}, "bfm19": {"n": 19, "in_joint_train": 0, "in_bfm_only_train": 0, "in_joint_online_eval": 5, "in_bfm_only_online_eval": 2}, "ict992": {"n": 992, "in_joint_train": 0, "in_bfm_only_train": 0, "in_joint_online_eval": 0, "in_bfm_only_online_eval": 0}}


---

# Lettura (revisione 1; scritta dopo i risultati, con le regole fissate nel protocollo)

La lettura della prima tornata e' in `lettura_tornata1.md`. Su crop e ICP quella e' superata da questa.

**Leak.** Nessun soggetto valutato e' nel training del congiunto: 0/108 BFM, 0/89 ICT, 0/992 ICT nella
galleria grande. Restano due note.
- 16 dei 108 soggetti BFM erano nell'eval online del congiunto, cioe' hanno pesato sulla scelta del
  checkpoint.
- Il BFM-only e' confrontabile solo su 19 soggetti.

Il congiunto ha visto in training le topologie crop e noisy dei soggetti di training; nessuna baseline ha
avuto un'augmentation simile.

**1. Con l'ICP di similarita' il vantaggio del congiunto sul crop sparisce: era l'artefatto di scala.**
- **Con il crop.** Rank-1 congiunto contro ICP di similarita' + NICP P2Tri:
  - BFM: 0.944 contro 1.000, delta -0.056 [-0.089, -0.030], **sotto**;
  - ICT: 0.993 contro 1.000, delta -0.007 [-0.020, +0.000], pari.

  Anche la sola **ICP di similarita' + Chamfer**, senza NICP, fa 1.000 in tutti e due i domini, con e
  senza crop. Con l'ICP rigido, sul crop, la stessa Chamfer scendeva a 0.11 (BFM) e 0.31 (ICT).
- **Senza crop.** Il congiunto e' pari alle due varianti di similarita' sul rank-1 (BFM 0.998 contro
  1.000; ICT 1.000 contro 1.000). Sul TAR@FAR 1e-3 e' sotto in BFM (0.969 contro 1.000) e alla pari o
  sopra in ICT (1.000 contro 0.997 NICP e 0.962 Chamfer). BFM-19: congiunto, BFM-only e similarita' tutti a
  1.000.

In dominio, quindi, una pipeline geometrica configurata bene riconosce le persone attraverso topologie e
crop **almeno quanto** il congiunto. La differenza della prima tornata (congiunto 0.944 / 0.993 contro
NICP rigido 0.686 / 0.600 sul crop) veniva dalla scala non corretta.

**2. La galleria grande non rompe il soffitto.**
- **Blocchi rettangolari (100 query, galleria di 992).**
  - remesh -> original (primario): congiunto e ICP di similarita' + NICP P2Tri entrambi a 1.000 su
    rank-1, AUC e TAR@FAR 1e-3 e 1e-4. La Chamfer cade a 0.16, il template a 0.82.
  - crop -> original: rank-1 1.000 per entrambi; il TAR del congiunto e' 0.98 / 0.97, quello di NICP
    1.00 / 1.00.
- **Tutte le 992 query, solo congiunto.**
  - senza crop: rank-1 1.000, TAR@FAR 1e-4 0.995;
  - con crop: rank-1 0.996, TAR@FAR 1e-4 0.926.
- **noisy -> original e' degenere.** La noisy e' la original perturbata: anche la Chamfer grezza fa
  1.000. Lo stesso vale in parte per crop -> original, che e' un ritaglio della original. Il blocco
  remesh -> original e' stato aggiunto per questo, prima di calcolarlo.

Con 992 identita' ICT il compito resta saturo per i due metodi migliori; per separarli servono dati piu'
difficili, non piu' soggetti dello stesso 3DMM.

**3. Espressioni (rexpr), invariato.** Il congiunto resta il metodo peggiore fra i forti: rank-1 0.822,
contro 0.905 di NICP rigido e 1.000 di ArcFace. Il template fa 0.388. Le varianti di similarita' non sono
state calcolate su rexpr (niente crop, stessa topologia; dichiarato nel protocollo).

**4. Baseline "NICP su template".**
- **Senza crop.** Rank-1 0.856 (BFM) e 0.898 (ICT) sui 108/89; sulla galleria grande 0.82 (remesh) e
  0.34 (noisy).
- **Con il crop.** Crolla: 0.00-0.05. Il template copre tutta la faccia, e registrandolo su un ritaglio la
  NICP inventa la parte mancante, che poi entra nella distanza media per vertice.

E' l'implementazione semplice dichiarata, con i parametri NICP di faceBench non ritoccati: va letta come
limite inferiore della famiglia "iscrivi una volta", non come il suo valore migliore.

**5. ArcFace migliore.** Con l'inquadratura per mesh e le normali smussate:
- rank-1 BFM 0.987 senza crop (0.981 prima) e 0.988 con il crop; ICT 0.981 e 0.980;
- il TAR@FAR resta inchiodato a 0.600 senza crop e 0.800 con il crop: le coppie con la `noisy` non
  passano mai una soglia stretta, perche' la media sull'1-anello non basta a togliere quel rumore;
- su rexpr fa 1.000.

Il congiunto e' **sopra** ArcFace migliore senza crop (BFM +0.012 [+0.003, +0.021], ICT +0.019), **sotto**
con il crop in BFM (-0.044 [-0.077, -0.018]) e **pari** in ICT.

**6. Tempi** (job 1060419, `a768-l40s-06` in `--exclusive`, AMD EPYC 9454 + L40S; CPU a un thread).

| fase | metodo | tempo (mediana) |
| --- | --- | --- |
| iscrizione, per mesh | congiunto (operatori CPU 2.03 s + embedding GPU 55 ms) | 2.08 s [1.27-4.42] |
| iscrizione, per mesh | NICP su template | 1.15 s |
| iscrizione, per mesh | ArcFace migliore | 0.57 s |
| ricerca 1:10.000, per query | congiunto | 1.96 ms su CPU, 68 us su GPU |
| ricerca 1:10.000, per query | template (L2 media per vertice) | 811 ms |
| ricerca 1:10.000, per query | ArcFace | 0.88 ms |
| per coppia | ICP di similarita' + Chamfer | 45 ms |
| per coppia | ICP di similarita' + NICP P2Tri | 1.29 s |

Per i metodi a coppie, una ricerca 1:10.000 costerebbe circa 7.5 minuti (ICP + Chamfer) e 3.6 ore (NICP).
E' una STIMA (N x tempo per coppia), non una misura.

**Conclusione per il paper.** In dominio, il riconoscimento non e' piu' un argomento d'accuratezza a favore
del congiunto: ICP di similarita' + Chamfer lo eguaglia o lo supera (crop BFM) a 45 ms per coppia.
L'argomento che resta e' il costo della ricerca su gallerie grandi:
- il congiunto si iscrive una volta (circa 2 s) e poi confronta in microsecondi;
- la Chamfer va rifatta per ogni coppia, quindi il congiunto conviene da circa 50 confronti per query in
  su (stima: 2.08 s / 45 ms);
- fra i metodi che si iscrivono una volta, il congiunto domina il template (accuratezza) e ArcFace sulle
  topologie senza crop (TAR).

Restano a sfavore: espressioni e crop BFM contro ICP di similarita'.
