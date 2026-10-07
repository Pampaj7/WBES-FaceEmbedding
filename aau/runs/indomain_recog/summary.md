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

---

# Risultati

CI 95% bootstrap per soggetto, 1000 repliche, le stesse per tutti i metodi di un insieme. mAP = MRR (un solo rilevante). Distanze NaN (coppie faceBench fallite) = +inf.

## BFM, 108 held-out del congiunto

### PRIMARIO, 5 topologie senza crop

Retrieval: 2160 query (20 coppie ordinate x 108), galleria di 108. Verifica: 1080 coppie stessa persona, 115560 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 0.998 [0.996, 1.000] | 0.999 [0.998, 1.000] | 1.000 [1.000, 1.000] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.652 [0.624, 0.679] | 0.726 [0.702, 0.748] | 0.931 [0.920, 0.943] | 0 |
| Rigid ICP + Chamfer | 0.934 [0.920, 0.948] | 0.954 [0.944, 0.964] | 0.992 [0.990, 0.995] | 0 |
| Rigid ICP + NICP + P2Tri | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 0 |
| ArcFace, normal map, 3 viste | 0.981 [0.968, 0.992] | 0.989 [0.979, 0.996] | 0.978 [0.971, 0.983] | 0 |
| ArcFace, ombreggiato, 3 viste (secondaria) | 0.919 [0.896, 0.943] | 0.948 [0.931, 0.965] | 0.959 [0.950, 0.967] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | mAP: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- |
| Chamfer (faceBench, 4096 pt) | +0.346 [+0.319, +0.374] (0.000) | +0.273 [+0.250, +0.296] (0.000) | +0.069 [+0.057, +0.080] (0.000) | sopra / sopra |
| Rigid ICP + Chamfer | +0.064 [+0.051, +0.077] (0.000) | +0.045 [+0.036, +0.055] (0.000) | +0.008 [+0.005, +0.010] (0.000) | sopra / sopra |
| Rigid ICP + NICP + P2Tri | -0.001 [-0.003, +0.000] (0.945) | -0.001 [-0.002, +0.000] (0.945) | +0.000 [-0.000, +0.000] (0.088) | pari / pari |
| ArcFace, normal map, 3 viste | +0.017 [+0.006, +0.030] (0.000) | +0.010 [+0.004, +0.020] (0.000) | +0.022 [+0.016, +0.028] (0.000) | sopra / sopra |
| ArcFace, ombreggiato, 3 viste (secondaria) | +0.079 [+0.055, +0.103] (0.000) | +0.051 [+0.034, +0.068] (0.000) | +0.041 [+0.033, +0.050] (0.000) | sopra / sopra |

### a parte: crop da un lato

Retrieval: 1080 query (10 coppie ordinate x 108), galleria di 108. Verifica: 540 coppie stessa persona, 57780 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 0.944 [0.911, 0.970] | 0.965 [0.944, 0.982] | 0.991 [0.986, 0.995] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.044 [0.024, 0.069] | 0.099 [0.075, 0.127] | 0.568 [0.556, 0.582] | 0 |
| Rigid ICP + Chamfer | 0.106 [0.065, 0.154] | 0.181 [0.137, 0.230] | 0.651 [0.631, 0.674] | 0 |
| Rigid ICP + NICP + P2Tri | 0.686 [0.637, 0.736] | 0.776 [0.736, 0.816] | 0.970 [0.959, 0.979] | 0 |
| ArcFace, normal map, 3 viste | 0.984 [0.975, 0.993] | 0.989 [0.982, 0.995] | 0.984 [0.980, 0.988] | 0 |
| ArcFace, ombreggiato, 3 viste (secondaria) | 0.947 [0.934, 0.960] | 0.965 [0.956, 0.975] | 0.966 [0.960, 0.973] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | mAP: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- |
| Chamfer (faceBench, 4096 pt) | +0.899 [+0.862, +0.931] (0.000) | +0.866 [+0.833, +0.896] (0.000) | +0.423 [+0.409, +0.435] (0.000) | sopra / sopra |
| Rigid ICP + Chamfer | +0.838 [+0.781, +0.888] (0.000) | +0.784 [+0.730, +0.833] (0.000) | +0.339 [+0.316, +0.360] (0.000) | sopra / sopra |
| Rigid ICP + NICP + P2Tri | +0.257 [+0.193, +0.320] (0.000) | +0.189 [+0.140, +0.236] (0.000) | +0.021 [+0.010, +0.033] (0.000) | sopra / sopra |
| ArcFace, normal map, 3 viste | -0.041 [-0.073, -0.014] (0.999) | -0.024 [-0.045, -0.006] (0.995) | +0.007 [+0.000, +0.013] (0.024) | sotto / sopra |
| ArcFace, ombreggiato, 3 viste (secondaria) | -0.004 [-0.038, +0.024] (0.601) | -0.000 [-0.022, +0.017] (0.499) | +0.025 [+0.017, +0.032] (0.000) | pari / sopra |

## ICT, 89 held-out del congiunto

### PRIMARIO, 5 topologie senza crop

Retrieval: 1780 query (20 coppie ordinate x 89), galleria di 89. Verifica: 890 coppie stessa persona, 78320 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.381 [0.347, 0.416] | 0.472 [0.444, 0.502] | 0.689 [0.675, 0.705] | 0 |
| Rigid ICP + Chamfer | 0.670 [0.654, 0.687] | 0.726 [0.712, 0.742] | 0.816 [0.798, 0.834] | 0 |
| Rigid ICP + NICP + P2Tri | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.994 [0.991, 0.996] | 0 |
| ArcFace, normal map, 3 viste | 0.981 [0.970, 0.991] | 0.990 [0.983, 0.995] | 0.994 [0.992, 0.997] | 0 |
| ArcFace, ombreggiato, 3 viste (secondaria) | 0.971 [0.956, 0.983] | 0.982 [0.972, 0.990] | 0.978 [0.972, 0.983] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | mAP: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- |
| Chamfer (faceBench, 4096 pt) | +0.619 [+0.584, +0.653] (0.000) | +0.528 [+0.498, +0.556] (0.000) | +0.311 [+0.295, +0.325] (0.000) | sopra / sopra |
| Rigid ICP + Chamfer | +0.330 [+0.313, +0.346] (0.000) | +0.274 [+0.258, +0.288] (0.000) | +0.184 [+0.166, +0.202] (0.000) | sopra / sopra |
| Rigid ICP + NICP + P2Tri | +0.000 [+0.000, +0.000] (1.000) | +0.000 [+0.000, +0.000] (1.000) | +0.006 [+0.004, +0.009] (0.000) | pari / sopra |
| ArcFace, normal map, 3 viste | +0.019 [+0.009, +0.030] (0.000) | +0.010 [+0.005, +0.017] (0.000) | +0.006 [+0.003, +0.008] (0.000) | sopra / sopra |
| ArcFace, ombreggiato, 3 viste (secondaria) | +0.029 [+0.017, +0.044] (0.000) | +0.018 [+0.010, +0.028] (0.000) | +0.022 [+0.017, +0.028] (0.000) | sopra / sopra |

### a parte: crop da un lato

Retrieval: 890 query (10 coppie ordinate x 89), galleria di 89. Verifica: 445 coppie stessa persona, 39160 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 0.993 [0.980, 1.000] | 0.996 [0.989, 1.000] | 1.000 [1.000, 1.000] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.097 [0.067, 0.131] | 0.228 [0.199, 0.263] | 0.708 [0.693, 0.724] | 0 |
| Rigid ICP + Chamfer | 0.306 [0.273, 0.342] | 0.405 [0.372, 0.441] | 0.833 [0.809, 0.860] | 0 |
| Rigid ICP + NICP + P2Tri | 0.600 [0.554, 0.647] | 0.698 [0.659, 0.737] | 0.962 [0.951, 0.973] | 0 |
| ArcFace, normal map, 3 viste | 0.993 [0.988, 0.998] | 0.995 [0.991, 0.999] | 0.995 [0.993, 0.997] | 0 |
| ArcFace, ombreggiato, 3 viste (secondaria) | 0.981 [0.971, 0.990] | 0.988 [0.981, 0.994] | 0.985 [0.981, 0.989] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | mAP: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- |
| Chamfer (faceBench, 4096 pt) | +0.897 [+0.861, +0.929] (0.000) | +0.768 [+0.733, +0.798] (0.000) | +0.292 [+0.276, +0.306] (0.000) | sopra / sopra |
| Rigid ICP + Chamfer | +0.688 [+0.648, +0.721] (0.000) | +0.592 [+0.556, +0.625] (0.000) | +0.167 [+0.140, +0.191] (0.000) | sopra / sopra |
| Rigid ICP + NICP + P2Tri | +0.393 [+0.338, +0.442] (0.000) | +0.298 [+0.256, +0.341] (0.000) | +0.037 [+0.027, +0.049] (0.000) | sopra / sopra |
| ArcFace, normal map, 3 viste | +0.000 [-0.015, +0.009] (0.507) | +0.001 [-0.006, +0.006] (0.338) | +0.005 [+0.003, +0.006] (0.000) | pari / sopra |
| ArcFace, ombreggiato, 3 viste (secondaria) | +0.012 [-0.004, +0.026] (0.067) | +0.008 [-0.001, +0.016] (0.032) | +0.015 [+0.011, +0.019] (0.000) | pari / sopra |

## ICT con espressioni casuali, 89 held-out del congiunto

### PRIMARIO, espressione contro espressione (k != k')

Retrieval: 1780 query (20 coppie ordinate x 89), galleria di 89. Verifica: 890 coppie stessa persona, 78320 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 0.822 [0.785, 0.860] | 0.865 [0.834, 0.895] | 0.978 [0.968, 0.985] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.867 [0.835, 0.898] | 0.901 [0.876, 0.926] | 0.979 [0.971, 0.987] | 0 |
| Rigid ICP + Chamfer | 0.896 [0.867, 0.922] | 0.923 [0.899, 0.944] | 0.981 [0.972, 0.989] | 0 |
| Rigid ICP + NICP + P2Tri | 0.905 [0.878, 0.929] | 0.929 [0.906, 0.949] | 0.977 [0.966, 0.986] | 0 |
| ArcFace, normal map, 3 viste | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| ArcFace, ombreggiato, 3 viste (secondaria) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | mAP: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- |
| Chamfer (faceBench, 4096 pt) | -0.046 [-0.069, -0.021] (1.000) | -0.036 [-0.054, -0.019] (1.000) | -0.001 [-0.006, +0.004] (0.709) | sotto / pari |
| Rigid ICP + Chamfer | -0.074 [-0.098, -0.050] (1.000) | -0.058 [-0.078, -0.038] (1.000) | -0.003 [-0.010, +0.005] (0.812) | sotto / pari |
| Rigid ICP + NICP + P2Tri | -0.083 [-0.110, -0.055] (1.000) | -0.064 [-0.086, -0.042] (1.000) | +0.001 [-0.007, +0.011] (0.404) | sotto / pari |
| ArcFace, normal map, 3 viste | -0.178 [-0.215, -0.140] (1.000) | -0.135 [-0.166, -0.105] (1.000) | -0.022 [-0.032, -0.015] (1.000) | sotto / sotto |
| ArcFace, ombreggiato, 3 viste (secondaria) | -0.178 [-0.215, -0.140] (1.000) | -0.135 [-0.166, -0.105] (1.000) | -0.022 [-0.032, -0.015] (1.000) | sotto / sotto |

### secondario: galleria neutra, query con espressione

Retrieval: 445 query (5 coppie ordinate x 89), galleria di 89. Verifica: 445 coppie stessa persona, 39160 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 0.962 [0.942, 0.980] | 0.976 [0.962, 0.987] | 0.992 [0.987, 0.995] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.998 [0.993, 1.000] | 0.999 [0.996, 1.000] | 0.994 [0.990, 0.997] | 0 |
| Rigid ICP + Chamfer | 0.998 [0.993, 1.000] | 0.999 [0.997, 1.000] | 0.997 [0.995, 0.999] | 0 |
| Rigid ICP + NICP + P2Tri | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.995, 0.999] | 0 |
| ArcFace, normal map, 3 viste | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| ArcFace, ombreggiato, 3 viste (secondaria) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | mAP: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- |
| Chamfer (faceBench, 4096 pt) | -0.036 [-0.056, -0.020] (1.000) | -0.023 [-0.035, -0.012] (1.000) | -0.002 [-0.005, +0.001] (0.883) | sotto / pari |
| Rigid ICP + Chamfer | -0.036 [-0.056, -0.018] (1.000) | -0.023 [-0.037, -0.012] (1.000) | -0.005 [-0.009, -0.002] (1.000) | sotto / sotto |
| Rigid ICP + NICP + P2Tri | -0.038 [-0.058, -0.020] (1.000) | -0.024 [-0.038, -0.013] (1.000) | -0.005 [-0.009, -0.002] (1.000) | sotto / sotto |
| ArcFace, normal map, 3 viste | -0.038 [-0.058, -0.020] (1.000) | -0.024 [-0.038, -0.013] (1.000) | -0.008 [-0.013, -0.005] (1.000) | sotto / sotto |
| ArcFace, ombreggiato, 3 viste (secondaria) | -0.038 [-0.058, -0.020] (1.000) | -0.024 [-0.038, -0.013] (1.000) | -0.008 [-0.013, -0.005] (1.000) | sotto / sotto |

## BFM-19, held-out di congiunto E BFM-only

### PRIMARIO, 5 topologie senza crop

Retrieval: 380 query (20 coppie ordinate x 19), galleria di 19. Verifica: 190 coppie stessa persona, 3420 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| BFM-only (1019310) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.750 [0.708, 0.797] | 0.832 [0.803, 0.862] | 0.917 [0.888, 0.941] | 0 |
| Rigid ICP + Chamfer | 0.945 [0.921, 0.966] | 0.965 [0.950, 0.978] | 0.985 [0.975, 0.993] | 0 |
| Rigid ICP + NICP + P2Tri | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.999 [0.998, 1.000] | 0 |
| ArcFace, normal map, 3 viste | 0.992 [0.976, 1.000] | 0.996 [0.988, 1.000] | 0.955 [0.929, 0.976] | 0 |
| ArcFace, ombreggiato, 3 viste (secondaria) | 0.950 [0.911, 0.984] | 0.973 [0.952, 0.992] | 0.927 [0.897, 0.952] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | mAP: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- |
| BFM-only (1019310) | +0.000 [+0.000, +0.000] (1.000) | +0.000 [+0.000, +0.000] (1.000) | -0.000 [-0.000, +0.000] (1.000) | pari / pari |
| Chamfer (faceBench, 4096 pt) | +0.250 [+0.203, +0.292] (0.000) | +0.168 [+0.138, +0.197] (0.000) | +0.083 [+0.059, +0.112] (0.000) | sopra / sopra |
| Rigid ICP + Chamfer | +0.055 [+0.034, +0.079] (0.000) | +0.035 [+0.022, +0.050] (0.000) | +0.015 [+0.007, +0.025] (0.000) | sopra / sopra |
| Rigid ICP + NICP + P2Tri | +0.000 [+0.000, +0.000] (1.000) | +0.000 [+0.000, +0.000] (1.000) | +0.001 [-0.000, +0.002] (0.078) | pari / pari |
| ArcFace, normal map, 3 viste | +0.008 [+0.000, +0.024] (0.372) | +0.004 [+0.000, +0.012] (0.372) | +0.045 [+0.024, +0.071] (0.000) | pari / sopra |
| ArcFace, ombreggiato, 3 viste (secondaria) | +0.050 [+0.016, +0.089] (0.000) | +0.027 [+0.008, +0.048] (0.000) | +0.073 [+0.048, +0.103] (0.000) | sopra / sopra |

### a parte: crop da un lato

Retrieval: 190 query (10 coppie ordinate x 19), galleria di 19. Verifica: 95 coppie stessa persona, 1710 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| BFM+ICT congiunto | 0.979 [0.953, 1.000] | 0.989 [0.976, 1.000] | 0.995 [0.988, 1.000] | 0 |
| BFM-only (1019310) | 0.942 [0.863, 1.000] | 0.970 [0.930, 1.000] | 0.990 [0.977, 1.000] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.105 [0.026, 0.190] | 0.253 [0.180, 0.326] | 0.554 [0.530, 0.594] | 0 |
| Rigid ICP + Chamfer | 0.184 [0.095, 0.268] | 0.348 [0.267, 0.431] | 0.664 [0.624, 0.704] | 0 |
| Rigid ICP + NICP + P2Tri | 0.911 [0.832, 0.974] | 0.951 [0.906, 0.986] | 0.981 [0.964, 0.992] | 0 |
| ArcFace, normal map, 3 viste | 0.989 [0.974, 1.000] | 0.994 [0.984, 1.000] | 0.968 [0.947, 0.984] | 0 |
| ArcFace, ombreggiato, 3 viste (secondaria) | 0.953 [0.926, 0.979] | 0.971 [0.952, 0.987] | 0.941 [0.920, 0.962] | 0 |

Delta appaiati congiunto - metodo (lettura del protocollo su rank-1 e AUC):

| metodo | rank-1: delta [CI] (P<=0) | mAP: delta [CI] (P<=0) | AUC: delta [CI] (P<=0) | lettura rank-1 / AUC |
| --- | --- | --- | --- | --- |
| BFM-only (1019310) | +0.037 [-0.005, +0.095] (0.078) | +0.019 [-0.003, +0.049] (0.077) | +0.005 [-0.002, +0.013] (0.096) | pari / pari |
| Chamfer (faceBench, 4096 pt) | +0.874 [+0.784, +0.958] (0.000) | +0.737 [+0.666, +0.809] (0.000) | +0.441 [+0.403, +0.466] (0.000) | sopra / sopra |
| Rigid ICP + Chamfer | +0.795 [+0.705, +0.884] (0.000) | +0.642 [+0.558, +0.723] (0.000) | +0.331 [+0.292, +0.370] (0.000) | sopra / sopra |
| Rigid ICP + NICP + P2Tri | +0.068 [+0.000, +0.153] (0.031) | +0.039 [+0.001, +0.089] (0.021) | +0.014 [+0.002, +0.032] (0.009) | pari / sopra |
| ArcFace, normal map, 3 viste | -0.011 [-0.037, +0.016] (0.812) | -0.004 [-0.018, +0.011] (0.725) | +0.027 [+0.012, +0.048] (0.000) | pari / sopra |
| ArcFace, ombreggiato, 3 viste (secondaria) | +0.026 [-0.011, +0.063] (0.119) | +0.018 [-0.004, +0.039] (0.060) | +0.054 [+0.036, +0.073] (0.000) | pari / sopra |

## Tempi

- model: host `a768-l40s-05.srv.aau.dk`, CPU AMD EPYC 9454 48-Core Processor, GPU NVIDIA L40S, job 1058495, OMP_NUM_THREADS=1
- facebench: host `a768-l40s-05.srv.aau.dk`, CPU AMD EPYC 9454 48-Core Processor, job 1058495, OMP_NUM_THREADS=1
- arcface: host `a768-l40s-05.srv.aau.dk`, CPU AMD EPYC 9454 48-Core Processor, job 1058495, OMP_NUM_THREADS=1
- campione: 60 mesh, vertici da 3275 a 60435 (mediana 12955)

| voce | per | n | mediana | IQR (25-75%) |
| --- | --- | --- | --- | --- |
| congiunto: operatori DiffusionNet (k=128), CPU 1 thread | mesh | 60 | 2.28 s | 1.27 s - 4.79 s |
| congiunto: embedding (caricamento + forward), GPU | mesh | 60 | 56.77 ms | 40.70 ms - 98.00 ms |
| congiunto: embedding (caricamento + forward), CPU 1 thread | mesh | 60 | 442.80 ms | 271.95 ms - 936.74 ms |
| congiunto: confronto ||z_a - z_b||, CPU | coppia di embedding | 1000 | 2.3 us | 2.3 us - 2.4 us |
| congiunto: retrieval 1:100 (distanze + argsort), CPU 1 thread | query | 100 | 19.8 us | 19.6 us - 19.9 us |
| congiunto: retrieval 1:100 (distanze + argsort), GPU | query | 100 | 40.7 us | 40.4 us - 41.3 us |
| congiunto: retrieval 1:1,000 (distanze + argsort), CPU 1 thread | query | 100 | 187.9 us | 187.0 us - 188.9 us |
| congiunto: retrieval 1:1,000 (distanze + argsort), GPU | query | 100 | 50.1 us | 49.7 us - 50.8 us |
| congiunto: retrieval 1:10,000 (distanze + argsort), CPU 1 thread | query | 100 | 2.21 ms | 2.14 ms - 2.26 ms |
| congiunto: retrieval 1:10,000 (distanze + argsort), GPU | query | 100 | 68.9 us | 68.5 us - 69.5 us |
| ICP rigido + Chamfer, CPU 1 thread | coppia di mesh | 60 | 33.93 ms | 30.30 ms - 41.59 ms |
| NICP P2Tri (con l'ICP rigido che la precede), CPU 1 thread | coppia di mesh | 60 | 1.42 s | 1.37 s - 1.82 s |
| ArcFace normal map: lettura + 3 render, CPU 1 thread | mesh | 60 | 197.25 ms | 136.54 ms - 319.35 ms |
| ArcFace normal map: 3 embedding + media, CPU 1 thread | mesh | 60 | 345.17 ms | 344.78 ms - 350.74 ms |
| ArcFace normal map: totale | mesh | 60 | 543.17 ms | 483.23 ms - 664.58 ms |
| congiunto: operatori + embedding GPU (totale per mesh) | mesh | 60 | 2.34 s | 1.31 s - 4.89 s |

## Controlli

- BFM, 108 held-out del congiunto, BFM+ICT congiunto: max |diff| contro `latent_distance` WS2 = 3.14e-03 su 173340 coppie (mediana di latent_distance 0.763)
- ICT, 89 held-out del congiunto, BFM+ICT congiunto: max |diff| contro `latent_distance` WS2 = 2.28e-03 su 117480 coppie (mediana di latent_distance 1.173)
- ICT con espressioni casuali, 89 held-out del congiunto, BFM+ICT congiunto: max |diff| contro `latent_distance` WS2 = 2.11e-03 su 82236 coppie (mediana di latent_distance 1.266)
- BFM-19, held-out di congiunto E BFM-only, BFM+ICT congiunto: max |diff| contro `latent_distance` WS2 = 2.32e-03 su 5130 coppie (mediana di latent_distance 0.750)
- BFM-19, held-out di congiunto E BFM-only, BFM-only (1019310): max |diff| contro `latent_distance` WS2 = 8.62e-04 su 5130 coppie (mediana di latent_distance 0.509)
- faceBench: NICP asimmetrico, orientazione della coppia = ordine delle etichette, non query -> galleria.
- leak (`sets.json`): {"bfm": {"n": 108, "in_joint_train": 0, "in_bfm_only_train": 89, "in_joint_online_eval": 16, "in_bfm_only_online_eval": 2}, "ict": {"n": 89, "in_joint_train": 0, "in_bfm_only_train": 0, "in_joint_online_eval": 0, "in_bfm_only_online_eval": 0}, "rexpr": {"n": 89, "in_joint_train": 0, "in_bfm_only_train": 0, "in_joint_online_eval": 0, "in_bfm_only_online_eval": 0}, "bfm19": {"n": 19, "in_joint_train": 0, "in_bfm_only_train": 0, "in_joint_online_eval": 5, "in_bfm_only_online_eval": 2}}

---

# Lettura (scritta dopo i risultati, con le regole fissate nel protocollo)

**Leak.** Nessun soggetto valutato e' nel training del congiunto (0/108 BFM, 0/89 ICT, ricontrollato su
`splits.json`). Restano le due note del protocollo: 16 dei 108 soggetti BFM (5 dei 19 di BFM-19) erano
nell'eval online del congiunto, cioe' hanno pesato sulla scelta del checkpoint; il BFM-only e' confrontabile
solo su 19 soggetti, perche' 89 dei 108 erano nel suo training.

**1. In dominio, senza crop, il compito e' saturo.** Il congiunto fa rank-1 0.998 (BFM) e 1.000 (ICT),
AUC 1.000; NICP P2Tri fa lo stesso (1.000 / 1.000 su BFM, 1.000 / 0.994 su ICT). Lettura del protocollo:
congiunto **pari** a NICP sul rank-1 in entrambi i domini, pari sull'AUC in BFM e **sopra** in ICT
(+0.006 [+0.004, +0.009]); **sopra** a Chamfer, ICP + Chamfer e ArcFace su normal map su rank-1 e AUC in
entrambi i domini. Il riconoscimento in dominio fra topologie diverse non separa il congiunto da NICP: lo
separa da tutto il resto. Anche il congiunto contro il BFM-only (BFM-19) e' pari, 1.000 contro 1.000.

**2. Con il crop da un lato il congiunto stacca le pipeline geometriche, non ArcFace.** Rank-1 0.944 (BFM)
e 0.993 (ICT) contro 0.686 e 0.600 di NICP P2Tri (delta +0.257 e +0.393, **sopra**), e Chamfer / ICP sotto
0.31. Contro ArcFace su normal map: **sotto** sul rank-1 in BFM (-0.041 [-0.073, -0.014]), pari in ICT, e
**sopra** sull'AUC in entrambi. BFM-19, crop: congiunto 0.979 contro BFM-only 0.942, delta +0.037
[-0.005, +0.095], pari (19 soggetti: il test e' senza potenza).

**3. Con le espressioni il congiunto e' il metodo peggiore.** Espressione contro espressione (stessa
topologia ICT): rank-1 0.822, **sotto** Chamfer (0.867), ICP + Chamfer (0.896), NICP P2Tri (0.905) e ArcFace
(1.000); AUC 0.978, **pari** alle tre geometriche e **sotto** ArcFace (1.000). Con la galleria neutra il
congiunto sale a 0.962 ma resta **sotto** tutti (gli altri da 0.998 a 1.000). Due avvertenze che non cambiano
il segno: (a) qui non varia la topologia, cioe' manca proprio la difficolta' in cui le pipeline geometriche
crollano (punto 2), e Chamfer su mesh con la stessa connettivita' e' quasi una distanza vertice-vertice;
(b) il congiunto non ha mai visto espressioni ICT in training. ArcFace a 1.000 non e' un errore di
indicizzazione: i png di controllo (`arcface/rexpr/normals/control`) mostrano espressioni diverse per lo
stesso soggetto, e su ICT neutro fra topologie lo stesso ArcFace fa 0.981. Per il paper: il riconoscimento
d'identita' attraverso le espressioni e' un limite da dichiarare, non un risultato.

**4. Tempi (stesso nodo L40S, AMD EPYC 9454; parti CPU a un thread).** Congiunto: 2.34 s per mesh
[IQR 1.31-4.89], quasi tutto operatori DiffusionNet su CPU (2.28 s; l'embedding sulla GPU e' 57 ms, sulla CPU
443 ms); confronto fra due embedding 2.3 us; retrieval 1:100 / 1:1.000 / 1:10.000 in 0.02 / 0.19 / 2.2 ms
su CPU (0.04 / 0.05 / 0.07 ms su GPU). NICP P2Tri 1.42 s per coppia [1.37-1.82], ICP + Chamfer 34 ms per
coppia, ArcFace su normal map 0.54 s per mesh (render 0.20 s + 3 embedding 0.35 s). Quindi, a parita' di
accuratezza senza crop, un confronto 1:1 da zero costa al congiunto due iscrizioni (~4.7 s) contro 1.4 s di
NICP, ma con la galleria gia' iscritta una query 1:N costa ~2.3 s + N x 2.3 us, contro N x 1.42 s per NICP
(stima, non misurata: 1:10.000 ~ 4 ore a un core). Il punto di pareggio e' a N = 2.
Avvertenze: (i) il nodo dei tempi ospitava nello stesso momento un mio job faceBench a 48 core (core
diversi, ma memoria e frequenza condivise): tutte le voci sono state misurate in quelle condizioni, in fila
nello stesso job, e i tempi assoluti possono essere gonfiati; (ii) l'embedding comprende la lettura degli
operatori precalcolati da /tmp, che in uso reale verrebbero dalla memoria: totale per mesh leggermente
sovrastimato.

**Controlli.** Le distanze del congiunto dagli embedding coincidono con `latent_distance` delle eval WS2 a
meno di 3.1e-3 (mediana delle distanze 0.76-1.27, cioe' < 0.5%): stesso modello e stessa catena, differenze
numeriche fra esecuzioni GPU. faceBench: 0 coppie fallite su 412.590.

