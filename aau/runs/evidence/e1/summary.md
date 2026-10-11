# E1: varieta' di 3DMM contro quantita' di identita' (protocollo, scritto PRIMA dei numeri)

8 ottobre 2026, prima di qualsiasi training o valutazione delle celle nuove. Piano: `paper/PLAN_MASSIVE.md`,
sezione 13. Codice in `aau/evidence/e1_factorial/`, run e risultati in `aau/runs/evidence/e1/`.
Questo file non si modifica dopo i numeri: `e1_summarize.py` lo copia in testa a `summary.md`.

## Regola di lettura (decisa dal PI, dichiarata qui prima dei numeri)

**La varieta' e' sostenuta se C3F - C2F e C3M - C2M sono entrambe > 0 sulla distanza graduata
`nocrop_cross` di HIFI3D, con IC 95% che esclude lo 0.**

Come la applico, senza margini di interpretazione dopo:
- metrica: Spearman fra distanza latente e GT `maxabs` di HIFI3D, mesh-pair, 20 coppie ordinate di
  topologie senza crop, clean (la colonna "HIFI3D mesh-pair senza crop" di `aau/runs/data_scale_ood/curve.md`);
- differenza appaiata sulle STESSE righe e sulle STESSE 1000 repliche bootstrap per soggetto (seme 1234);
  "IC che esclude lo 0" = estremo inferiore dell'IC 95% > 0;
- servono ENTRAMBE le differenze; se ne passa una sola, la varieta' NON e' sostenuta e si dice quale;
- si applica separatamente a 10.548 e a 21.096 passi. Lettura principale a 21.096 (fine di ogni run);
  10.548 accanto. Se i due numeri di passi non concordano, lo si scrive, senza fonderli.

Letture secondarie (mie, simmetriche alla primaria, solo descrittive):
- quantita': C2M - C2F e C3M - C3F sulla stessa metrica, con la stessa regola;
- G1 contro C3M e contro C3F: se G1 - C3M ha IC che contiene lo 0 o e' > 0, GNM da solo basta a
  spiegare il livello di C3M su HIFI3D;
- le stesse differenze sugli altri domini di test (sotto), senza regole di decisione.

## Celle

| cella | training | da dove |
| --- | --- | --- |
| C3M | BFM 392 + ICT 54.008 (4.008 ICT-5000 + 50.000 nuove) + GNM 10.000 | run su scala 1060130, checkpoint epoch036 / epoch072: nessun training nuovo |
| C2M | BFM 392 + ICT 54.008 | nuovo run |
| C2F | BFM 392 + ICT 5.401 (401 ICT-5000 + 5.000 nuove) | nuovo run |
| C3F | BFM 392 + ICT 4.557 (338 + 4.219) + GNM 844: stesso totale non-BFM di C2F (5.401), rapporto ICT/GNM 5.40 come C3M | nuovo run |
| G1 | GNM 10.000 | nuovo run |

- Stesso trainer (`v2_work/fastio/train_steps.py`, invariato), stessa ricetta v1 flag per flag, stessa
  spec dei dati, stessa GT, stesso held-out (1.200) ed eval online, seme 1234 (modello, epoche, blocchi).
- Sottoinsiemi stratificati per sorgente e insieme di mesh (ICT nuove: 6 topologie + 8 espressioni;
  GNM: 1 o 2 espressioni), seme 1234; le ICT di C3F sono un sottoinsieme di quelle di C2F
  (`make_e1_splits.py`, `aau/evidence/e1_factorial/subsets.json`).
- Passi: checkpoint a 10.548 e 21.096 passi (epoche 36 e 72 da 293 passi). Quote di passi per epoca:
  BFM 26 (8.87%); celle a due domini ICT 267; celle a tre domini ICT 225 + GNM 42 (come C3M); G1 GNM 293.
- Blocchi: K = 40 (C2M), 4 (C2F, C3F), 6 (G1); C3M ne ha 46. Un blocco deve contenere i soggetti di
  un'epoca per dominio. C2M ha T nominale 91.709 (E = 313) per cambiare blocco alle stesse epoche di C3M
  ed e' fermato dopo il checkpoint dell'epoca 72: come C3M, i suoi checkpoint sono istantanee di un run
  piu' lungo.

## Attenzione prima di leggere i numeri: la quantita' VISTA

C3M a 21.096 passi ha toccato 10 blocchi su 46: le identita' viste sono circa 14.300, non 64.400
(`design.md`, calcolato con le funzioni del trainer). Per costruzione C2M ne vede quasi altrettante
(97%). Le celle F vedono 3.093 (10.548 passi) e 5.793 (21.096) identita'. Il contrasto di quantita'
effettivo e' quindi circa 2.5x in identita' viste (e 8 contro 18 epoche di esposizione per identita'),
non 10x. La regola primaria (varieta') non ne dipende; le letture di quantita' vanno lette cosi'.

## Domini di test (stesso protocollo e stesse repliche di `aau/runs/data_scale_ood/curve.md`)

- HIFI3D (100 soggetti x 6 topologie, pipeline `aau/zs3dmm`, bracci `scale_<tag>`): Spearman con la GT
  `maxabs` su `nocrop_cross`, `all_cross`, `subject_pair_mean` (clean); riconoscimento (rank-1, AUC di
  verifica) sulle 5 topologie senza crop, dagli embedding.
- FaceVerse con espressioni (100 soggetti, convenzione BFM `_flip`): riconoscimento (rank-1, AUC).
- NoW validation: tau di Kendall per immagine sui 3 metodi pre-registrati (`aau/recon`).
- FLAME zero-shot (`aau/flame`, 100 soggetti x 6 topologie): Spearman `nocrop_cross`, `all_cross`,
  `subject_pair_mean`.
- IC 95% bootstrap per soggetto, 1000 repliche, seme 1234; differenze fra celle appaiate sulle stesse righe
  e sulle stesse repliche.

## Limiti dichiarati prima

- Un solo seme per cella: l'IC copre il campionamento dei soggetti di test, non la variabilita' da run a run
  del training (stimata in `aau/data_scale/PLAN.md` intorno a +-0.035 su un training BFM corto, metrica diversa).
  Differenze di pochi centesimi con IC che esclude lo 0 vanno lette con questo in mente.
- C3M e' un run gia' esistente; le altre celle sono nuove (stesso codice, nodi e tempi diversi).

---

# E1, emendamento 1 al protocollo (scritto PRIMA di ogni numero delle celle nuove)

8 ottobre 2026, sera. Rimanda a `protocol.md` (sha256 493941994cd41864d2d2b815ca257e53bf6cc46913b075c1aec7155a3deb29f6,
registrato in `protocol.sha256` alle 15:30), che resta invariato. Scritto dopo la revisione del critic
(BLOCCANTE sul disegno) e le decisioni del PI, quando nessuna cella nuova di E1 aveva ancora un checkpoint ne' un
numero: gli unici numeri esistenti sono quelli di C3M su L40S (run su scala 1060130, gia' in
`aau/runs/data_scale_ood/curve.md`) e degli smoke di plumbing (che non producono metriche di test). Dove questo
file e la versione originale divergono, vale questo file. Lo sha256 di questo file e' in `protocol_amendment.sha256`.

## 1. Celle e hardware

- **Tutte le celle si addestrano su V100** (container PyTorch 24.10): C2M, C2F, C3F, C2F-GNM, C3F-UGT, i secondi
  semi C2F-s2 e C3F-s2, C2F40, C3F40, G1 e **C3M rifatta su V100** (stesso split del run su scala, K=46, fermata
  all'epoca 72). Le valutazioni girano tutte sulle A100.
- **La C3M della regola e' quella rifatta su V100.** La C3M su L40S (1060130) resta solo come riferimento.
- **C2F-GNM** = BFM 392 + GNM 5.401 (stratificate per 1/2 espressioni), nessuna ICT: lo stesso totale non-BFM
  di C2F e C3F. Le 844 GNM di C3F sono un suo sottoinsieme.
- **C3F-UGT** = C3F (stesso split, passi, seme, hardware) con la GT di training UNIFICATA
  (`datasets/UNIFIED_GT/train/gt_unified_bfm_ict_gnm.npz`, verificata da `check_train_gt.py`).
- **Secondi semi:** C2F-s2 e C3F-s2 = C2F e C3F con `--seed` e `block_seed` 2345, stesso split.
- **C2F40 e C3F40:** ICT a 1/40 (1.350 non-BFM, un solo blocco), annidate in C2F e C3F. Servono SOLO alla
  lettura della quantita' (circa 10x contro le celle M); non entrano nella regola sulla varieta'.

## 2. Regola sulla varieta' (sostituisce la regola primaria di protocol.md)

Metrica: Spearman con la GT `maxabs` di HIFI3D, mesh-pair `nocrop_cross` (20 coppie ordinate senza crop),
clean, seme 1234 delle celle. Effetto minimo: **+0.05**.

Delta = Sp(C3F) - max(Sp(C2F), Sp(C2F-GNM)). Il massimo si prende in OGNI replica bootstrap: 1000 repliche per
soggetto, le stesse per tutte le celle. IC 95% = percentili.

Gli esiti si dichiarano a 21.096 passi (lettura principale) e a 10.548 (accanto):

- **SOSTENUTA** se valgono TUTTE:
  - (a) Delta >= +0.05 e IC di Delta con estremo inferiore > 0;
  - (b) dev FaceScape: Delta (stessa definizione, GT `maxabs`, `nocrop_cross`) con punto > 0, cioe' stessa
    direzione;
  - (c) non inferiorita' su FaceVerse con espressioni (convenzione BFM): rank-1(C3F) - max(rank-1(C2F),
    rank-1(C2F-GNM)), con il massimo per replica, ha IC con estremo inferiore > -0.05;
  - (d) Delta maggiore del pavimento del rumore (sotto);
  - (e) secondo seme: Sp(C3F-s2) - Sp(C2F-s2) > 0 (punto).
- **SMENTITA** se:
  - Delta <= 0 con IC che esclude +0.05 (estremo superiore < +0.05);
  - e sul dev FaceScape Delta <= 0 (punto).
- **NON CONCLUDENTE** in ogni altro caso, con l'elenco delle condizioni mancate.

**Pavimento del rumore,** su HIFI3D `nocrop_cross` (GT `maxabs`) allo stesso numero di passi: il massimo fra
|Sp(C3M V100) - Sp(C3M L40S)| (stesso seme, hardware diverso), |Sp(C2F) - Sp(C2F-s2)| e |Sp(C3F) - Sp(C3F-s2)|.
Se e' grande quanto Delta o piu', l'esito e' NON CONCLUDENTE, anche se valgono (a)-(c).

C3M - C2M e C3F - C2F (il confronto originale) restano riportati, solo come descrittivi.

## 3. Quantita' e G1 (descrittive, nessuna regola di decisione)

- **Circa 2.5x:** C2M - C2F, C3M - C3F.
- **Circa 10x:** C2M - C2F40, C3M - C3F40.
- **G1** contro C3M e contro C3F.
- Stessa metrica, stessi IC.

## 4. GT unificata

Tutte le celle si valutano con ENTRAMBE le GT, `maxabs` e unificata, sulle stesse righe e repliche:
- HIFI3D: `datasets/UNIFIED_GT/gt/hifi3d_unified.npz`;
- dev FaceScape e FLAME: con la GT unificata del set di valutazione, se il suo agente la fornisce nel formato di
  `aau/zs3dmm`; altrimenti "-".

**Domanda dichiarata:** addestrando sulla GT unificata, il modello raggiunge le baseline con allineamento su
quella GT?
- Confronto: C3F-UGT - ICP+Chamfer (`rigid_icp_chamfer` di faceBench) con la GT unificata di HIFI3D,
  `nocrop_cross`, sulle righe dove la baseline e' finita.
- **RAGGIUNGE** se l'IC 95% della differenza include lo 0 o e' tutto positivo (estremo superiore >= 0);
  **NO** altrimenti.
- Accanto, descrittivi: C3F-UGT - NICP P2Tri, e C3F-UGT - C3F con entrambe le GT.

## 5. Invariato

Domini e misure di `protocol.md`, IC (1000 repliche per soggetto, seme 1234), "un seme per cella" tranne i
secondi semi qui sopra, limiti dichiarati.

---

# E1, nota tecnica: build_grad vettorizzato nel pre-pass (8 ottobre 2026, prima di ogni numero delle celle nuove)

Si aggiunge a `protocol.md` e a `protocol_amendment.md`, che restano invariati. Decisione del PI, presa dopo il
controllo di uguaglianza; nessun training di cella era ancora partito.

- **Cosa:** in TUTTE le celle (training su V100) il pre-pass degli operatori usa il `build_grad` vettorizzato di
  E9 al posto di `diffusion_net.geometry.build_grad`.
  - La copia e' congelata in `aau/evidence/e1_factorial/gradvec_site/e1_grad_vec.py`; sha256 della sorgente E9
    alla copia: ecdb6756..
  - E' agganciata da `gradvec_site/sitecustomize.py` con `WBES_E1_GRADVEC=1`; diffusion-net, v2_work e
    aau/data_scale restano invariati.
  - Le valutazioni (A100) usano tutte il `build_grad` originale, compresa C3M L40S.
- **Differenze misurate** (`gradvec_check.json`; 57 mesh, tutte le topologie e le espressioni, 3 ICT nuove e 2
  GNM, nodo V100):
  - uguali tutti gli array tranne `gradX_values` e `gradY_values`;
  - 10 file e 19 array diversi, il 6.3e-5 degli elementi;
  - |differenza| massima 1.2e-10, relativa al massimo 3.7e-13. Sono voci vicine a zero; la causa e' l'ordine
    di somma.
  - Molto sotto il rumore numerico del training.
- **Guadagno:** pre-pass da 4.61 a 2.81 CPU-s per mesh, cioe' 1.64x.
- **Confondenti:** nessuno fra le celle, perche' tutte usano la stessa versione. C3M L40S (1060130, `build_grad`
  originale) e' solo un riferimento: nel confronto col rumore (C3M V100 - C3M L40S) entra anche questa
  differenza, oltre all'hardware.
- **Lo smoke V100** (1061840) gira con il `build_grad` originale. Confronta solo la loss dell'epoca 1 con L40S:
  gli operatori coincidono a meno di 1.2e-10, quindi il confronto non ne risente.

---

# E1, emendamento 2: C3F-UGT con la GT unificata TARATA (9 ottobre 2026, prima di ogni numero delle celle nuove)

Si aggiunge a `protocol.md`, `protocol_amendment.md` e `nota_tecnica_gradvec.md`, che restano invariati.

**Motivo (critic).** I margini della loss sono in unita' della GT (`--rank_margin 0.05` ecc.), e la GT unificata
ha un'altra scala. Senza taratura, C3F-UGT contro C3F mescolerebbe il contenuto della GT con la sua scala.

**Stato al momento della scrittura:**
- C3F-UGT (1061850) non era ancora partito; e' stato tenuto fermo (`scontrol hold`) fino alla verifica.
- Nessuna cella nuova ha ancora un numero di valutazione.
- C2F, C3F e C2F-GNM sono in training dalle 19:15 dell'8 ottobre. Usano la GT maxabs e non cambiano.

**Taratura** (`aau/evidence/e1_factorial/e1_calib_ugt.py`, job 1062071):
- Per ogni dominio d, f_d = mediana della GT maxabs del run / mediana della GT unificata, sul blocco
  intra-dominio. Mediane esatte sulle coppie i < j, tutti i soggetti del dominio.
- Le coppie fra domini (che il trainer a batch monodominio non legge) sono scalate con sqrt(f_d1 f_d2).
- Fattori:

  | dominio | mediana maxabs | mediana unificata | f_d |
  | --- | --- | --- | --- |
  | BFM | 0.3452 | 0.2808 | 1.2294 |
  | ICT | 0.2311 | 0.2531 | 0.9134 |
  | GNM | 0.3001 | 0.2931 | 1.0241 |

- File: `datasets/UNIFIED_GT/train/gt_unified_bfm_ict_gnm_calib.npz` (+ `.json`, massimo 0.967).
- Controllo col loader del trainer (`check_calib.json`, ok):
  - names e indici di train, held-out ed eval online identici alla GT del run;
  - nessun valore non finito, diagonale 0, simmetria;
  - rapporto tarata / unificata = f entro 1e-7;
  - mediane per dominio di nuovo uguali alla maxabs.

**Celle:**
- **C3F-UGT** usa la GT TARATA. E' la cella della domanda dichiarata in `protocol_amendment.md`, sezione 4,
  che resta invariata (C3F-UGT - ICP+Chamfer, GT unificata di HIFI3D). La taratura cambia per dominio solo la
  scala, non l'ordine delle distanze: lo Spearman di valutazione non ne dipende.
- **C3F-UGT non tarata** (`c3fugtraw`, GT unificata originale): aggiunta in coda, a bassa priorita'.
  C3F-UGT - C3F-UGT non tarata e' l'effetto della scala; descrittivo, nessuna regola.

---

# Risultati

Generato da `aau/evidence/e1_factorial/e1_summarize.py`. Testi qui sopra invariati rispetto agli sha256 registrati prima dei numeri: protocol.md SI, protocol_amendment.md SI, nota_tecnica_gradvec.md SI, protocol_amendment_2.md SI.

## Regola sulla varieta' (emendamento, sezione 2): C3F - max(C2F, C2F-GNM), HIFI3D `nocrop_cross`, GT maxabs

| passi | C3F / C2F / C2F-GNM | Delta [IC 95%] (P<=0) | dev FaceScape Delta | FaceVerse rank-1 Delta | secondo seme C3F s2 - C2F s2 | pavimento del rumore | esito |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 21096 | 0.716 / 0.434 / 0.599 | +0.116 [+0.075, +0.158] (0.000) | +0.090 [+0.056, +0.106] (0.000) | -0.071 [-0.098, -0.046] (1.000) | +0.201 [+0.159, +0.239] (0.000) | 0.058 (C3M V100 - L40S +0.029, C2F s1 - s2 -0.023, C3F s1 - s2 +0.058) | **NON CONCLUDENTE: (c) FaceVerse rank-1 non inferiore (IC > -0.05) non vale** |
| 10548 | 0.716 / 0.464 / 0.476 | +0.240 [+0.185, +0.280] (0.000) | +0.115 [+0.090, +0.138] (0.000) | -0.056 [-0.081, -0.031] (1.000) | +0.259 [+0.221, +0.294] (0.000) | 0.042 (C3M V100 - L40S +0.009, C2F s1 - s2 +0.042, C3F s1 - s2 +0.036) | **NON CONCLUDENTE: (c) FaceVerse rank-1 non inferiore (IC > -0.05) non vale** |

## Domanda su C3F-UGT (emendamento, sezione 4): GT unificata di HIFI3D, `nocrop_cross`, righe con baseline finita

| passi | baseline | C3F-UGT / baseline | differenza [IC 95%] (P<=0) | esito |
| --- | --- | --- | --- | --- |
| 21096 | rigid_icp_chamfer | 0.575 / 0.522 | +0.053 [-0.015, +0.122] (0.057) | RAGGIUNGE |
| 21096 | nicp_p2tri | 0.575 / 0.577 | -0.002 [-0.063, +0.065] (0.515) | (descrittivo) |
| 10548 | rigid_icp_chamfer | 0.533 / 0.522 | +0.011 [-0.057, +0.080] (0.373) | RAGGIUNGE |
| 10548 | nicp_p2tri | 0.533 / 0.577 | -0.044 [-0.110, +0.023] (0.901) | (descrittivo) |

## Matrice celle x domini di test, 10548 passi

Punto [IC 95%, 1000 repliche per soggetto, le stesse per tutte le celle]. GT maxabs dove non indicato.

| cella | HIFI3D nocrop | HIFI3D nocrop, GT unif. | HIFI3D all_cross | HIFI3D subj-pair-mean | HIFI3D rank-1 | HIFI3D AUC | dev FaceScape nocrop | dev FaceScape, GT unif. | FaceVerse espr. rank-1 | FaceVerse espr. AUC | NoW tau | FLAME nocrop | FLAME nocrop, GT unif. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3M | 0.686 [0.615, 0.749] | 0.343 [0.267, 0.419] | 0.546 [0.462, 0.618] | 0.754 [0.681, 0.814] | 0.844 [0.818, 0.870] | 0.978 [0.972, 0.984] | 0.439 [0.376, 0.495] | 0.408 [0.342, 0.469] | 0.513 [0.471, 0.558] | 0.827 [0.799, 0.855] | 0.339 [0.248, 0.444] | 0.542 [0.493, 0.586] | - |
| C2M | 0.488 [0.417, 0.552] | 0.261 [0.206, 0.317] | 0.283 [0.229, 0.332] | 0.701 [0.621, 0.769] | 0.547 [0.512, 0.585] | 0.888 [0.874, 0.902] | 0.319 [0.274, 0.358] | 0.243 [0.198, 0.289] | 0.516 [0.476, 0.559] | 0.841 [0.816, 0.866] | 0.199 [0.100, 0.305] | 0.581 [0.518, 0.635] | - |
| C2F | 0.464 [0.393, 0.529] | 0.240 [0.192, 0.290] | 0.279 [0.225, 0.331] | 0.710 [0.635, 0.778] | 0.429 [0.396, 0.466] | 0.854 [0.836, 0.870] | 0.282 [0.244, 0.318] | 0.221 [0.178, 0.261] | 0.503 [0.466, 0.545] | 0.830 [0.805, 0.855] | 0.184 [0.077, 0.290] | 0.489 [0.428, 0.542] | - |
| C3F | 0.716 [0.645, 0.776] | 0.366 [0.283, 0.444] | 0.587 [0.504, 0.657] | 0.762 [0.688, 0.822] | 0.931 [0.912, 0.951] | 0.987 [0.982, 0.991] | 0.397 [0.344, 0.449] | 0.357 [0.295, 0.421] | 0.555 [0.512, 0.598] | 0.846 [0.820, 0.872] | 0.299 [0.189, 0.416] | 0.553 [0.498, 0.602] | - |
| C2F-GNM | 0.476 [0.410, 0.540] | 0.275 [0.225, 0.326] | 0.467 [0.398, 0.527] | 0.812 [0.755, 0.858] | 0.426 [0.395, 0.461] | 0.833 [0.814, 0.851] | 0.255 [0.208, 0.300] | 0.260 [0.212, 0.305] | 0.612 [0.566, 0.659] | 0.874 [0.846, 0.900] | 0.110 [-0.025, 0.246] | 0.677 [0.628, 0.719] | - |
| C3F-UGT | 0.439 [0.367, 0.507] | 0.533 [0.452, 0.608] | 0.376 [0.312, 0.436] | 0.507 [0.427, 0.582] | 0.895 [0.876, 0.916] | 0.981 [0.976, 0.986] | 0.372 [0.293, 0.449] | 0.562 [0.499, 0.625] | 0.783 [0.742, 0.827] | 0.937 [0.916, 0.956] | 0.358 [0.225, 0.480] | 0.472 [0.415, 0.526] | - |
| C3F-UGT non tarata | 0.405 [0.328, 0.478] | 0.534 [0.455, 0.605] | 0.356 [0.290, 0.419] | 0.460 [0.373, 0.536] | 0.906 [0.888, 0.926] | 0.988 [0.984, 0.992] | 0.382 [0.306, 0.457] | 0.587 [0.519, 0.650] | 0.756 [0.715, 0.798] | 0.918 [0.893, 0.941] | 0.364 [0.266, 0.463] | 0.484 [0.428, 0.540] | - |
| C2F s2 | 0.422 [0.360, 0.483] | 0.239 [0.190, 0.290] | 0.258 [0.210, 0.302] | 0.724 [0.648, 0.782] | 0.368 [0.336, 0.402] | 0.828 [0.808, 0.845] | 0.277 [0.237, 0.314] | 0.200 [0.159, 0.239] | 0.455 [0.416, 0.495] | 0.802 [0.775, 0.828] | 0.237 [0.126, 0.345] | 0.552 [0.490, 0.608] | - |
| C3F s2 | 0.680 [0.609, 0.742] | 0.330 [0.254, 0.405] | 0.530 [0.451, 0.596] | 0.764 [0.694, 0.823] | 0.837 [0.815, 0.859] | 0.970 [0.963, 0.977] | 0.423 [0.369, 0.472] | 0.341 [0.285, 0.396] | 0.566 [0.522, 0.609] | 0.852 [0.826, 0.877] | 0.311 [0.192, 0.435] | 0.580 [0.525, 0.633] | - |
| C2F40 | 0.482 [0.413, 0.545] | 0.274 [0.220, 0.328] | 0.288 [0.235, 0.335] | 0.705 [0.633, 0.765] | 0.535 [0.496, 0.575] | 0.887 [0.872, 0.901] | 0.313 [0.270, 0.353] | 0.275 [0.226, 0.322] | 0.527 [0.488, 0.568] | 0.839 [0.814, 0.864] | 0.220 [0.098, 0.348] | 0.596 [0.533, 0.649] | - |
| C3F40 | 0.687 [0.615, 0.750] | 0.363 [0.287, 0.440] | 0.547 [0.466, 0.620] | 0.746 [0.674, 0.809] | 0.908 [0.887, 0.928] | 0.980 [0.974, 0.985] | 0.369 [0.319, 0.416] | 0.302 [0.248, 0.352] | 0.533 [0.490, 0.578] | 0.855 [0.831, 0.881] | 0.373 [0.263, 0.492] | 0.542 [0.486, 0.593] | - |
| G1 | 0.661 [0.598, 0.719] | 0.333 [0.269, 0.402] | 0.621 [0.553, 0.678] | 0.859 [0.817, 0.892] | 0.570 [0.538, 0.601] | 0.912 [0.896, 0.927] | 0.198 [0.161, 0.235] | 0.193 [0.155, 0.227] | 0.431 [0.386, 0.475] | 0.813 [0.784, 0.841] | -0.045 [-0.170, 0.075] | 0.666 [0.618, 0.707] | - |
| C3M L40S (rif.) | 0.677 [0.608, 0.741] | 0.340 [0.261, 0.417] | 0.509 [0.432, 0.579] | 0.767 [0.698, 0.823] | 0.812 [0.785, 0.840] | 0.968 [0.960, 0.976] | 0.420 [0.361, 0.475] | 0.380 [0.316, 0.441] | 0.518 [0.475, 0.565] | 0.822 [0.794, 0.850] | 0.299 [0.216, 0.398] | 0.543 [0.490, 0.592] | - |

## Matrice celle x domini di test, 21096 passi

Punto [IC 95%, 1000 repliche per soggetto, le stesse per tutte le celle]. GT maxabs dove non indicato.

| cella | HIFI3D nocrop | HIFI3D nocrop, GT unif. | HIFI3D all_cross | HIFI3D subj-pair-mean | HIFI3D rank-1 | HIFI3D AUC | dev FaceScape nocrop | dev FaceScape, GT unif. | FaceVerse espr. rank-1 | FaceVerse espr. AUC | NoW tau | FLAME nocrop | FLAME nocrop, GT unif. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3M | 0.692 [0.623, 0.756] | 0.348 [0.273, 0.418] | 0.581 [0.500, 0.649] | 0.800 [0.740, 0.849] | 0.849 [0.825, 0.874] | 0.971 [0.962, 0.979] | 0.404 [0.348, 0.455] | 0.371 [0.308, 0.434] | 0.586 [0.541, 0.628] | 0.849 [0.821, 0.876] | 0.386 [0.269, 0.507] | 0.646 [0.599, 0.687] | - |
| C2M | 0.446 [0.380, 0.506] | 0.229 [0.178, 0.282] | 0.260 [0.211, 0.303] | 0.708 [0.635, 0.772] | 0.490 [0.460, 0.523] | 0.861 [0.847, 0.874] | 0.286 [0.248, 0.323] | 0.228 [0.188, 0.270] | 0.601 [0.561, 0.641] | 0.866 [0.840, 0.891] | 0.146 [0.038, 0.253] | 0.518 [0.453, 0.573] | - |
| C2F | 0.434 [0.367, 0.499] | 0.210 [0.166, 0.255] | 0.262 [0.209, 0.310] | 0.705 [0.628, 0.771] | 0.446 [0.416, 0.481] | 0.841 [0.826, 0.857] | 0.298 [0.258, 0.337] | 0.243 [0.198, 0.285] | 0.617 [0.577, 0.657] | 0.863 [0.837, 0.888] | 0.231 [0.136, 0.323] | 0.415 [0.360, 0.462] | - |
| C3F | 0.716 [0.650, 0.773] | 0.357 [0.284, 0.429] | 0.598 [0.523, 0.662] | 0.804 [0.745, 0.853] | 0.888 [0.867, 0.911] | 0.976 [0.969, 0.982] | 0.391 [0.336, 0.442] | 0.339 [0.277, 0.398] | 0.589 [0.548, 0.627] | 0.859 [0.832, 0.884] | 0.275 [0.153, 0.400] | 0.565 [0.515, 0.611] | - |
| C2F-GNM | 0.599 [0.529, 0.662] | 0.317 [0.258, 0.378] | 0.573 [0.502, 0.632] | 0.837 [0.786, 0.878] | 0.625 [0.597, 0.654] | 0.897 [0.880, 0.912] | 0.301 [0.246, 0.352] | 0.302 [0.246, 0.355] | 0.659 [0.611, 0.704] | 0.894 [0.869, 0.916] | 0.095 [-0.062, 0.236] | 0.716 [0.674, 0.753] | - |
| C3F-UGT | 0.451 [0.378, 0.522] | 0.575 [0.505, 0.635] | 0.389 [0.327, 0.452] | 0.508 [0.427, 0.583] | 0.910 [0.891, 0.929] | 0.980 [0.974, 0.986] | 0.389 [0.311, 0.467] | 0.522 [0.450, 0.589] | 0.783 [0.742, 0.824] | 0.929 [0.904, 0.952] | 0.244 [0.112, 0.367] | 0.449 [0.390, 0.503] | - |
| C3F-UGT non tarata | 0.428 [0.366, 0.487] | 0.522 [0.454, 0.583] | 0.367 [0.312, 0.417] | 0.529 [0.451, 0.597] | 0.822 [0.798, 0.847] | 0.964 [0.955, 0.972] | 0.398 [0.315, 0.481] | 0.568 [0.496, 0.632] | 0.744 [0.701, 0.787] | 0.906 [0.880, 0.931] | 0.254 [0.166, 0.359] | 0.445 [0.391, 0.497] | - |
| C2F s2 | 0.457 [0.391, 0.518] | 0.227 [0.179, 0.278] | 0.265 [0.216, 0.309] | 0.719 [0.645, 0.778] | 0.495 [0.460, 0.532] | 0.845 [0.828, 0.861] | 0.298 [0.256, 0.335] | 0.216 [0.174, 0.256] | 0.543 [0.500, 0.583] | 0.818 [0.791, 0.845] | 0.212 [0.120, 0.303] | 0.536 [0.471, 0.594] | - |
| C3F s2 | 0.658 [0.586, 0.722] | 0.304 [0.228, 0.375] | 0.526 [0.451, 0.592] | 0.762 [0.688, 0.820] | 0.810 [0.787, 0.834] | 0.962 [0.953, 0.970] | 0.431 [0.372, 0.485] | 0.345 [0.286, 0.402] | 0.658 [0.616, 0.699] | 0.876 [0.850, 0.902] | 0.222 [0.101, 0.347] | 0.576 [0.522, 0.625] | - |
| C2F40 | 0.462 [0.401, 0.520] | 0.253 [0.207, 0.302] | 0.268 [0.222, 0.309] | 0.715 [0.649, 0.772] | 0.525 [0.490, 0.559] | 0.860 [0.845, 0.874] | 0.336 [0.290, 0.376] | 0.299 [0.248, 0.347] | 0.570 [0.528, 0.611] | 0.840 [0.813, 0.867] | 0.203 [0.090, 0.316] | 0.502 [0.440, 0.557] | - |
| C3F40 | 0.641 [0.570, 0.707] | 0.301 [0.235, 0.368] | 0.537 [0.459, 0.608] | 0.794 [0.732, 0.844] | 0.777 [0.748, 0.806] | 0.940 [0.927, 0.952] | 0.369 [0.318, 0.414] | 0.349 [0.294, 0.403] | 0.590 [0.544, 0.635] | 0.872 [0.846, 0.897] | 0.348 [0.233, 0.468] | 0.615 [0.565, 0.659] | - |
| G1 | 0.561 [0.499, 0.619] | 0.361 [0.303, 0.420] | 0.513 [0.446, 0.570] | 0.842 [0.796, 0.879] | 0.534 [0.504, 0.563] | 0.874 [0.856, 0.890] | 0.240 [0.190, 0.289] | 0.245 [0.196, 0.293] | 0.491 [0.445, 0.536] | 0.840 [0.813, 0.867] | 0.119 [-0.024, 0.247] | 0.685 [0.639, 0.725] | - |
| C3M L40S (rif.) | 0.663 [0.594, 0.728] | 0.324 [0.254, 0.392] | 0.557 [0.479, 0.625] | 0.792 [0.731, 0.843] | 0.784 [0.758, 0.809] | 0.954 [0.943, 0.964] | 0.399 [0.343, 0.449] | 0.378 [0.317, 0.437] | 0.625 [0.581, 0.666] | 0.872 [0.846, 0.896] | 0.347 [0.227, 0.474] | 0.577 [0.530, 0.621] | - |

## Differenze appaiate (descrittive)

Cella: differenza [IC 95%] (P(boot <= 0)).


### 21096 passi

| contrasto | lettura | HIFI3D nocrop | HIFI3D nocrop, GT unif. | HIFI3D all_cross | HIFI3D subj-pair-mean | HIFI3D rank-1 | HIFI3D AUC | dev FaceScape nocrop | dev FaceScape, GT unif. | FaceVerse espr. rank-1 | FaceVerse espr. AUC | NoW tau | FLAME nocrop | FLAME nocrop, GT unif. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3F - C2F | varieta' (protocollo originale) | +0.282 [+0.243, +0.320] (0.00) | +0.147 [+0.101, +0.194] (0.00) | +0.337 [+0.297, +0.371] (0.00) | +0.100 [+0.057, +0.147] (0.00) | +0.442 [+0.408, +0.477] (0.00) | +0.134 [+0.122, +0.147] (0.00) | +0.093 [+0.069, +0.115] (0.00) | +0.097 [+0.071, +0.121] (0.00) | -0.028 [-0.049, -0.009] (1.00) | -0.005 [-0.015, +0.005] (0.80) | +0.044 [-0.082, +0.165] (0.26) | +0.150 [+0.109, +0.191] (0.00) | - |
| C3F - C2F-GNM | varieta' contro solo GNM | +0.116 [+0.075, +0.158] (0.00) | +0.040 [+0.002, +0.079] (0.02) | +0.025 [-0.010, +0.058] (0.09) | -0.033 [-0.073, +0.010] (0.93) | +0.263 [+0.238, +0.290] (0.00) | +0.079 [+0.069, +0.091] (0.00) | +0.090 [+0.056, +0.126] (0.00) | +0.037 [+0.004, +0.069] (0.02) | -0.071 [-0.098, -0.046] (1.00) | -0.035 [-0.046, -0.024] (1.00) | +0.180 [+0.021, +0.345] (0.02) | -0.151 [-0.177, -0.127] (1.00) | - |
| C3M - C2M | varieta', molte identita' | +0.246 [+0.194, +0.294] (0.00) | +0.119 [+0.065, +0.172] (0.00) | +0.321 [+0.272, +0.359] (0.00) | +0.092 [+0.043, +0.141] (0.00) | +0.358 [+0.323, +0.392] (0.00) | +0.110 [+0.100, +0.120] (0.00) | +0.118 [+0.080, +0.159] (0.00) | +0.143 [+0.100, +0.185] (0.00) | -0.015 [-0.039, +0.007] (0.91) | -0.017 [-0.029, -0.005] (1.00) | +0.241 [+0.101, +0.364] (0.00) | +0.127 [+0.077, +0.180] (0.00) | - |
| C3F s2 - C2F s2 | varieta', secondo seme | +0.201 [+0.159, +0.239] (0.00) | +0.076 [+0.027, +0.123] (0.00) | +0.261 [+0.220, +0.301] (0.00) | +0.044 [+0.001, +0.089] (0.02) | +0.316 [+0.284, +0.348] (0.00) | +0.116 [+0.106, +0.127] (0.00) | +0.134 [+0.103, +0.160] (0.00) | +0.129 [+0.101, +0.154] (0.00) | +0.115 [+0.090, +0.141] (0.00) | +0.058 [+0.046, +0.070] (0.00) | +0.009 [-0.075, +0.086] (0.40) | +0.040 [+0.001, +0.078] (0.02) | - |
| C2M - C2F | quantita' ~2.5x, 2 domini | +0.012 [-0.008, +0.034] (0.14) | +0.019 [-0.001, +0.039] (0.03) | -0.002 [-0.017, +0.014] (0.58) | +0.004 [-0.023, +0.030] (0.41) | +0.044 [+0.023, +0.064] (0.00) | +0.019 [+0.014, +0.024] (0.00) | -0.013 [-0.035, +0.009] (0.88) | -0.014 [-0.036, +0.007] (0.90) | -0.016 [-0.036, +0.005] (0.94) | +0.003 [-0.006, +0.012] (0.27) | -0.085 [-0.161, -0.003] (0.97) | +0.103 [+0.085, +0.123] (0.00) | - |
| C3M - C3F | quantita' ~2.5x, 3 domini | -0.024 [-0.046, -0.003] (0.99) | -0.009 [-0.034, +0.015] (0.79) | -0.018 [-0.040, +0.003] (0.96) | -0.004 [-0.026, +0.017] (0.65) | -0.040 [-0.056, -0.024] (1.00) | -0.005 [-0.009, -0.001] (1.00) | +0.013 [-0.003, +0.029] (0.07) | +0.032 [+0.014, +0.048] (0.00) | -0.003 [-0.023, +0.016] (0.61) | -0.009 [-0.017, -0.002] (0.99) | +0.112 [+0.034, +0.180] (0.01) | +0.081 [+0.061, +0.102] (0.00) | - |
| C2M - C2F40 | quantita' ~10x, 2 domini | -0.016 [-0.034, +0.001] (0.97) | -0.024 [-0.043, -0.004] (0.99) | -0.009 [-0.020, +0.003] (0.92) | -0.007 [-0.035, +0.020] (0.72) | -0.035 [-0.053, -0.012] (1.00) | +0.001 [-0.003, +0.006] (0.37) | -0.051 [-0.068, -0.033] (1.00) | -0.070 [-0.087, -0.054] (1.00) | +0.031 [+0.012, +0.050] (0.00) | +0.026 [+0.017, +0.035] (0.00) | -0.057 [-0.112, -0.002] (0.98) | +0.016 [+0.002, +0.032] (0.01) | - |
| C3M - C3F40 | quantita' ~10x, 3 domini | +0.051 [+0.033, +0.070] (0.00) | +0.047 [+0.025, +0.068] (0.00) | +0.043 [+0.027, +0.060] (0.00) | +0.007 [-0.011, +0.025] (0.24) | +0.071 [+0.046, +0.095] (0.00) | +0.031 [+0.025, +0.037] (0.00) | +0.034 [+0.014, +0.053] (0.00) | +0.022 [-0.001, +0.043] (0.03) | -0.004 [-0.025, +0.016] (0.64) | -0.022 [-0.032, -0.012] (1.00) | +0.038 [-0.005, +0.082] (0.04) | +0.030 [+0.016, +0.046] (0.00) | - |
| G1 - C3M | solo GNM contro C3M | -0.131 [-0.171, -0.093] (1.00) | +0.013 [-0.028, +0.053] (0.27) | -0.067 [-0.102, -0.029] (1.00) | +0.042 [+0.002, +0.086] (0.02) | -0.315 [-0.343, -0.285] (1.00) | -0.097 [-0.110, -0.086] (1.00) | -0.164 [-0.204, -0.126] (1.00) | -0.127 [-0.164, -0.089] (1.00) | -0.095 [-0.129, -0.059] (1.00) | -0.009 [-0.023, +0.005] (0.91) | -0.267 [-0.419, -0.120] (1.00) | +0.040 [+0.011, +0.067] (0.00) | - |
| G1 - C3F | solo GNM contro C3F | -0.155 [-0.196, -0.115] (1.00) | +0.004 [-0.035, +0.041] (0.43) | -0.085 [-0.122, -0.045] (1.00) | +0.038 [-0.003, +0.084] (0.04) | -0.354 [-0.385, -0.325] (1.00) | -0.102 [-0.116, -0.090] (1.00) | -0.151 [-0.191, -0.112] (1.00) | -0.095 [-0.134, -0.051] (1.00) | -0.098 [-0.133, -0.063] (1.00) | -0.018 [-0.034, -0.004] (0.99) | -0.155 [-0.308, +0.004] (0.97) | +0.120 [+0.096, +0.143] (0.00) | - |
| C3F-UGT - C3F | GT unificata (tarata) contro maxabs in training | -0.265 [-0.342, -0.181] (1.00) | +0.217 [+0.152, +0.285] (0.00) | -0.209 [-0.293, -0.120] (1.00) | -0.297 [-0.380, -0.218] (1.00) | +0.022 [+0.003, +0.040] (0.01) | +0.005 [+0.001, +0.009] (0.01) | -0.002 [-0.048, +0.047] (0.50) | +0.183 [+0.144, +0.219] (0.00) | +0.195 [+0.166, +0.225] (0.00) | +0.070 [+0.055, +0.085] (0.00) | -0.030 [-0.122, +0.060] (0.76) | -0.116 [-0.148, -0.079] (1.00) | - |
| C3F-UGT - C3F-UGT non tarata | scala della GT unificata: tarata contro non tarata | +0.023 [-0.008, +0.053] (0.07) | +0.053 [+0.021, +0.079] (0.00) | +0.023 [-0.005, +0.048] (0.06) | -0.021 [-0.056, +0.012] (0.89) | +0.088 [+0.071, +0.105] (0.00) | +0.016 [+0.012, +0.021] (0.00) | -0.009 [-0.034, +0.013] (0.76) | -0.046 [-0.070, -0.021] (1.00) | +0.039 [+0.022, +0.059] (0.00) | +0.023 [+0.016, +0.031] (0.00) | -0.009 [-0.094, +0.059] (0.64) | +0.004 [-0.019, +0.028] (0.36) | - |
| C3M - C3M L40S (rif.) | rumore: stesso seme, V100 contro L40S | +0.029 [+0.015, +0.044] (0.00) | +0.024 [+0.007, +0.040] (0.00) | +0.023 [+0.008, +0.038] (0.00) | +0.008 [-0.010, +0.026] (0.19) | +0.064 [+0.047, +0.081] (0.00) | +0.017 [+0.013, +0.021] (0.00) | +0.005 [-0.012, +0.020] (0.26) | -0.007 [-0.028, +0.011] (0.77) | -0.039 [-0.059, -0.020] (1.00) | -0.022 [-0.030, -0.015] (1.00) | +0.040 [+0.014, +0.069] (0.00) | +0.068 [+0.053, +0.086] (0.00) | - |
| C2F - C2F s2 | rumore: seme 1234 contro 2345 | -0.023 [-0.047, +0.002] (0.96) | -0.017 [-0.039, +0.003] (0.95) | -0.004 [-0.023, +0.012] (0.65) | -0.014 [-0.050, +0.023] (0.77) | -0.049 [-0.071, -0.028] (1.00) | -0.004 [-0.009, +0.001] (0.93) | +0.001 [-0.015, +0.017] (0.46) | +0.026 [+0.010, +0.044] (0.00) | +0.074 [+0.046, +0.104] (0.00) | +0.045 [+0.033, +0.058] (0.00) | +0.019 [-0.057, +0.095] (0.31) | -0.121 [-0.137, -0.103] (1.00) | - |
| C3F - C3F s2 | rumore: seme 1234 contro 2345 | +0.058 [+0.035, +0.084] (0.00) | +0.054 [+0.029, +0.081] (0.00) | +0.072 [+0.051, +0.094] (0.00) | +0.042 [+0.016, +0.073] (0.00) | +0.078 [+0.059, +0.099] (0.00) | +0.014 [+0.009, +0.019] (0.00) | -0.040 [-0.061, -0.020] (1.00) | -0.006 [-0.031, +0.021] (0.67) | -0.070 [-0.091, -0.048] (1.00) | -0.018 [-0.027, -0.008] (1.00) | +0.053 [-0.059, +0.150] (0.17) | -0.011 [-0.036, +0.015] (0.80) | - |

### 10548 passi

| contrasto | lettura | HIFI3D nocrop | HIFI3D nocrop, GT unif. | HIFI3D all_cross | HIFI3D subj-pair-mean | HIFI3D rank-1 | HIFI3D AUC | dev FaceScape nocrop | dev FaceScape, GT unif. | FaceVerse espr. rank-1 | FaceVerse espr. AUC | NoW tau | FLAME nocrop | FLAME nocrop, GT unif. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3F - C2F | varieta' (protocollo originale) | +0.252 [+0.209, +0.295] (0.00) | +0.126 [+0.073, +0.174] (0.00) | +0.307 [+0.265, +0.344] (0.00) | +0.052 [+0.012, +0.094] (0.01) | +0.502 [+0.461, +0.541] (0.00) | +0.133 [+0.118, +0.149] (0.00) | +0.115 [+0.091, +0.138] (0.00) | +0.137 [+0.109, +0.169] (0.00) | +0.052 [+0.025, +0.076] (0.00) | +0.016 [+0.005, +0.027] (0.00) | +0.116 [-0.015, +0.243] (0.04) | +0.064 [+0.022, +0.104] (0.00) | - |
| C3F - C2F-GNM | varieta' contro solo GNM | +0.240 [+0.186, +0.291] (0.00) | +0.090 [+0.037, +0.143] (0.00) | +0.120 [+0.074, +0.164] (0.00) | -0.050 [-0.104, +0.005] (0.95) | +0.505 [+0.473, +0.539] (0.00) | +0.153 [+0.137, +0.170] (0.00) | +0.142 [+0.097, +0.183] (0.00) | +0.098 [+0.049, +0.140] (0.00) | -0.056 [-0.081, -0.031] (1.00) | -0.028 [-0.041, -0.015] (1.00) | +0.189 [+0.042, +0.345] (0.01) | -0.124 [-0.158, -0.088] (1.00) | - |
| C3M - C2M | varieta', molte identita' | +0.198 [+0.160, +0.243] (0.00) | +0.082 [+0.033, +0.130] (0.00) | +0.263 [+0.222, +0.300] (0.00) | +0.053 [+0.003, +0.106] (0.01) | +0.296 [+0.260, +0.334] (0.00) | +0.090 [+0.079, +0.101] (0.00) | +0.119 [+0.082, +0.158] (0.00) | +0.165 [+0.131, +0.196] (0.00) | -0.003 [-0.029, +0.022] (0.63) | -0.014 [-0.027, -0.000] (0.98) | +0.140 [+0.033, +0.247] (0.01) | -0.039 [-0.082, +0.003] (0.97) | - |
| C3F s2 - C2F s2 | varieta', secondo seme | +0.259 [+0.221, +0.294] (0.00) | +0.090 [+0.045, +0.132] (0.00) | +0.272 [+0.233, +0.308] (0.00) | +0.040 [+0.001, +0.080] (0.02) | +0.468 [+0.438, +0.500] (0.00) | +0.142 [+0.130, +0.156] (0.00) | +0.146 [+0.123, +0.167] (0.00) | +0.141 [+0.119, +0.164] (0.00) | +0.110 [+0.088, +0.133] (0.00) | +0.050 [+0.035, +0.063] (0.00) | +0.074 [-0.024, +0.173] (0.07) | +0.028 [-0.019, +0.070] (0.12) | - |
| C2M - C2F | quantita' ~2.5x, 2 domini | +0.025 [+0.009, +0.041] (0.00) | +0.021 [+0.006, +0.037] (0.00) | +0.003 [-0.008, +0.015] (0.29) | -0.010 [-0.032, +0.013] (0.80) | +0.118 [+0.097, +0.141] (0.00) | +0.034 [+0.029, +0.040] (0.00) | +0.037 [+0.019, +0.055] (0.00) | +0.023 [+0.005, +0.040] (0.00) | +0.013 [-0.005, +0.031] (0.09) | +0.011 [+0.003, +0.020] (0.01) | +0.015 [-0.047, +0.067] (0.32) | +0.092 [+0.073, +0.114] (0.00) | - |
| C3M - C3F | quantita' ~2.5x, 3 domini | -0.030 [-0.054, -0.004] (0.99) | -0.023 [-0.044, +0.000] (0.97) | -0.041 [-0.067, -0.018] (1.00) | -0.008 [-0.037, +0.019] (0.73) | -0.088 [-0.107, -0.068] (1.00) | -0.009 [-0.012, -0.006] (1.00) | +0.042 [+0.015, +0.069] (0.00) | +0.051 [+0.025, +0.078] (0.00) | -0.042 [-0.061, -0.024] (1.00) | -0.019 [-0.028, -0.011] (1.00) | +0.040 [-0.011, +0.101] (0.07) | -0.012 [-0.030, +0.006] (0.90) | - |
| C2M - C2F40 | quantita' ~10x, 2 domini | +0.006 [-0.014, +0.026] (0.30) | -0.013 [-0.034, +0.008] (0.90) | -0.006 [-0.020, +0.009] (0.79) | -0.004 [-0.033, +0.026] (0.61) | +0.012 [-0.007, +0.031] (0.11) | +0.001 [-0.003, +0.006] (0.28) | +0.006 [-0.013, +0.026] (0.31) | -0.032 [-0.051, -0.012] (1.00) | -0.011 [-0.029, +0.009] (0.86) | +0.001 [-0.008, +0.011] (0.38) | -0.021 [-0.079, +0.027] (0.80) | -0.015 [-0.033, +0.001] (0.96) | - |
| C3M - C3F40 | quantita' ~10x, 3 domini | -0.000 [-0.019, +0.020] (0.51) | -0.020 [-0.039, +0.002] (0.96) | -0.002 [-0.019, +0.016] (0.58) | +0.008 [-0.015, +0.032] (0.25) | -0.065 [-0.081, -0.049] (1.00) | -0.001 [-0.003, +0.001] (0.88) | +0.070 [+0.041, +0.102] (0.00) | +0.106 [+0.082, +0.130] (0.00) | -0.020 [-0.042, +0.001] (0.97) | -0.028 [-0.039, -0.018] (1.00) | -0.034 [-0.094, +0.033] (0.86) | -0.000 [-0.015, +0.018] (0.52) | - |
| G1 - C3M | solo GNM contro C3M | -0.026 [-0.070, +0.019] (0.89) | -0.010 [-0.054, +0.036] (0.68) | +0.076 [+0.039, +0.114] (0.00) | +0.105 [+0.058, +0.161] (0.00) | -0.274 [-0.301, -0.246] (1.00) | -0.066 [-0.078, -0.054] (1.00) | -0.240 [-0.277, -0.199] (1.00) | -0.215 [-0.253, -0.171] (1.00) | -0.082 [-0.116, -0.047] (1.00) | -0.014 [-0.032, +0.004] (0.95) | -0.384 [-0.542, -0.226] (1.00) | +0.124 [+0.088, +0.162] (0.00) | - |
| G1 - C3F | solo GNM contro C3F | -0.055 [-0.101, -0.011] (0.99) | -0.033 [-0.077, +0.014] (0.91) | +0.035 [-0.001, +0.071] (0.03) | +0.097 [+0.050, +0.152] (0.00) | -0.362 [-0.396, -0.328] (1.00) | -0.075 [-0.089, -0.061] (1.00) | -0.198 [-0.240, -0.158] (1.00) | -0.164 [-0.210, -0.116] (1.00) | -0.124 [-0.157, -0.092] (1.00) | -0.033 [-0.051, -0.015] (1.00) | -0.345 [-0.519, -0.157] (1.00) | +0.112 [+0.078, +0.147] (0.00) | - |
| C3F-UGT - C3F | GT unificata (tarata) contro maxabs in training | -0.277 [-0.349, -0.198] (1.00) | +0.167 [+0.099, +0.235] (0.00) | -0.211 [-0.287, -0.132] (1.00) | -0.255 [-0.338, -0.183] (1.00) | -0.036 [-0.056, -0.016] (1.00) | -0.005 [-0.009, -0.002] (1.00) | -0.025 [-0.083, +0.033] (0.77) | +0.205 [+0.161, +0.248] (0.00) | +0.228 [+0.194, +0.261] (0.00) | +0.091 [+0.075, +0.107] (0.00) | +0.059 [-0.032, +0.139] (0.09) | -0.081 [-0.114, -0.047] (1.00) | - |
| C3F-UGT - C3F-UGT non tarata | scala della GT unificata: tarata contro non tarata | +0.034 [+0.008, +0.062] (0.01) | -0.001 [-0.033, +0.026] (0.53) | +0.019 [-0.007, +0.044] (0.07) | +0.047 [+0.015, +0.077] (0.00) | -0.011 [-0.026, +0.003] (0.96) | -0.007 [-0.009, -0.004] (1.00) | -0.010 [-0.033, +0.013] (0.80) | -0.024 [-0.046, -0.002] (0.99) | +0.027 [+0.008, +0.045] (0.00) | +0.020 [+0.013, +0.027] (0.00) | -0.006 [-0.070, +0.049] (0.59) | -0.012 [-0.035, +0.011] (0.84) | - |
| C3M - C3M L40S (rif.) | rumore: stesso seme, V100 contro L40S | +0.009 [-0.007, +0.026] (0.13) | +0.003 [-0.016, +0.024] (0.36) | +0.036 [+0.020, +0.052] (0.00) | -0.013 [-0.032, +0.005] (0.90) | +0.031 [+0.015, +0.048] (0.00) | +0.010 [+0.007, +0.013] (0.00) | +0.019 [-0.000, +0.040] (0.03) | +0.028 [+0.008, +0.050] (0.00) | -0.005 [-0.022, +0.010] (0.78) | +0.005 [-0.002, +0.011] (0.10) | +0.040 [-0.009, +0.090] (0.06) | -0.002 [-0.013, +0.011] (0.59) | - |
| C2F - C2F s2 | rumore: seme 1234 contro 2345 | +0.042 [+0.019, +0.066] (0.00) | +0.000 [-0.019, +0.019] (0.49) | +0.022 [+0.003, +0.039] (0.01) | -0.013 [-0.052, +0.020] (0.76) | +0.060 [+0.039, +0.082] (0.00) | +0.026 [+0.020, +0.033] (0.00) | +0.005 [-0.014, +0.022] (0.30) | +0.021 [+0.007, +0.034] (0.00) | +0.048 [+0.025, +0.072] (0.00) | +0.028 [+0.018, +0.038] (0.00) | -0.053 [-0.114, +0.016] (0.95) | -0.063 [-0.086, -0.042] (1.00) | - |
| C3F - C3F s2 | rumore: seme 1234 contro 2345 | +0.036 [+0.014, +0.059] (0.00) | +0.036 [+0.014, +0.057] (0.00) | +0.057 [+0.034, +0.078] (0.00) | -0.002 [-0.027, +0.021] (0.57) | +0.095 [+0.073, +0.116] (0.00) | +0.017 [+0.012, +0.022] (0.00) | -0.026 [-0.052, -0.001] (0.98) | +0.016 [-0.011, +0.042] (0.10) | -0.010 [-0.033, +0.014] (0.83) | -0.006 [-0.017, +0.006] (0.81) | -0.011 [-0.067, +0.036] (0.71) | -0.027 [-0.050, -0.005] (0.99) | - |

## Identita' viste per cella (design.json, calcolato prima dei numeri)

| cella | passi | viste per dominio | viste totali | esposizioni mediana |
| --- | --- | --- | --- | --- |
| C3M | 10548 | {'bfm': 392, 'gnm': 1090, 'ict': 5874} | 7356 | 8 |
| C3M | 21096 | {'bfm': 392, 'gnm': 2172, 'ict': 11695} | 14259 | 8 |
| C2M | 10548 | {'bfm': 392, 'ict': 6755} | 7147 | 8 |
| C2M | 21096 | {'bfm': 392, 'ict': 13493} | 13885 | 8 |
| C2F | 10548 | {'bfm': 392, 'ict': 2701} | 3093 | 18 |
| C2F | 21096 | {'bfm': 392, 'ict': 5401} | 5793 | 18 |
| C3F | 10548 | {'bfm': 392, 'gnm': 422, 'ict': 2279} | 3093 | 18 |
| C3F | 21096 | {'bfm': 392, 'gnm': 844, 'ict': 4557} | 5793 | 18 |
| C2F-GNM | 10548 | {'bfm': 392, 'gnm': 2701} | 3093 | 18 |
| C2F-GNM | 21096 | {'bfm': 392, 'gnm': 5401} | 5793 | 18 |
| C3F-UGT | 10548 | {'bfm': 392, 'gnm': 422, 'ict': 2279} | 3093 | 18 |
| C3F-UGT | 21096 | {'bfm': 392, 'gnm': 844, 'ict': 4557} | 5793 | 18 |
| C2F s2 | 10548 | {'bfm': 392, 'ict': 2701} | 3093 | 18 |
| C2F s2 | 21096 | {'bfm': 392, 'ict': 5401} | 5793 | 18 |
| C3F s2 | 10548 | {'bfm': 392, 'gnm': 422, 'ict': 2279} | 3093 | 18 |
| C3F s2 | 21096 | {'bfm': 392, 'gnm': 844, 'ict': 4557} | 5793 | 18 |
| C2F40 | 10548 | {'bfm': 392, 'ict': 1350} | 1742 | 36 |
| C2F40 | 21096 | {'bfm': 392, 'ict': 1350} | 1742 | 71 |
| C3F40 | 10548 | {'bfm': 392, 'gnm': 211, 'ict': 1139} | 1742 | 36 |
| C3F40 | 21096 | {'bfm': 392, 'gnm': 211, 'ict': 1139} | 1742 | 71 |
| G1 | 10548 | {'gnm': 5001} | 5001 | 11 |
| G1 | 21096 | {'gnm': 10000} | 10000 | 11 |
| C3M L40S (rif.) | 10548 | {'bfm': 392, 'gnm': 1090, 'ict': 5874} | 7356 | 8 |
| C3M L40S (rif.) | 21096 | {'bfm': 392, 'gnm': 2172, 'ict': 11695} | 14259 | 8 |

## Run

| cella | run dir (job, GPU) | passi eseguiti | blocchi (train.log) | picco rss+shmem (GiB) | picco cgroup con page cache (GiB) | --mem |
| --- | --- | --- | --- | --- | --- | --- |
| C3M | `aau/runs/evidence/e1/train_c3mv_s1234_1061856` (1061856, Tesla V100-SXM3-32GB) | 21096 (arresto voluto) | 46 da 1785 soggetti | 266.7 | 320.0 | 327680M |
| C2M | `aau/runs/evidence/e1/train_c2m_s1234_1061858` (1061858, Tesla V100-SXM3-32GB) | 21096 (arresto voluto) | 40 da 1743 soggetti | 273.5 | 320.0 | 327680M |
| C2F | `aau/runs/evidence/e1/train_c2f_s1234_1061843` (1061843, Tesla V100-SXM3-32GB) | 21096 | 4 da 1743 soggetti | 272.6 | 320.0 | 327680M |
| C3F | `aau/runs/evidence/e1/train_c3f_s1234_1061845` (1061845, Tesla V100-SXM3-32GB) | 21096 | 4 da 1743 soggetti | 262.1 | 320.0 | 327680M |
| C2F-GNM | `aau/runs/evidence/e1/train_c2fgnm_s1234_1061847` (1061847, Tesla V100-SXM3-32GB) | 21096 | 4 da 1743 soggetti | 196.8 | 240.0 | 245760M |
| C3F-UGT | `aau/runs/evidence/e1/train_c3fugt_s1234_1061850` (1061850, Tesla V100-SXM3-32GB) | 21096 | 4 da 1743 soggetti | 261.0 | 320.0 | 327680M |
| C3F-UGT non tarata | `aau/runs/evidence/e1/train_c3fugtraw_s1234_1062683` (1062683, Tesla V100-SXM3-32GB) | 21096 | 4 da 1743 soggetti | 261.7 | 320.0 | 327680M |
| C2F s2 | `aau/runs/evidence/e1/train_c2fs2_s2345_1062675` (1062675, Tesla V100-SXM3-32GB) | 21096 | 4 da 1743 soggetti | 273.8 | 320.0 | 327680M |
| C3F s2 | `aau/runs/evidence/e1/train_c3fs2_s2345_1062677` (1062677, Tesla V100-SXM3-32GB) | 21096 | 4 da 1743 soggetti | 260.6 | 320.0 | 327680M |
| C2F40 | `aau/runs/evidence/e1/train_c2f40_s1234_1061860` (1061860, Tesla V100-SXM3-32GB) | 21096 | 1 da 1742 soggetti | 261.1 | 320.0 | 327680M |
| C3F40 | `aau/runs/evidence/e1/train_c3f40_s1234_1062679` (1062679, Tesla V100-SXM3-32GB) | 21096 | 1 da 1742 soggetti | 249.3 | 320.0 | 327680M |
| G1 | `aau/runs/evidence/e1/train_g1_s1234_1062681` (1062681, Tesla V100-SXM3-32GB) | 21096 | 6 da 1667 soggetti | 162.9 | 240.0 | 245760M |

## Controlli

| controllo | valore | atteso |
| --- | --- | --- |
| hifi 10548: celle presenti (stesse righe, stessi soggetti) | ['c3mv', 'c2m', 'c2f', 'c3f', 'c2fgnm', 'c3fugt', 'c3fugtraw', 'c2fs2', 'c3fs2', 'c2f40', 'c3f40', 'g1', 'c3ml']; 100 soggetti, 148500 righe | - |
| hifi 21096: celle presenti (stesse righe, stessi soggetti) | ['c3mv', 'c2m', 'c2f', 'c3f', 'c2fgnm', 'c3fugt', 'c3fugtraw', 'c2fs2', 'c3fs2', 'c2f40', 'c3f40', 'g1', 'c3ml']; 100 soggetti, 148500 righe | - |
| devfs 10548: celle presenti (stesse righe, stessi soggetti) | ['c3mv', 'c2m', 'c2f', 'c3f', 'c2fgnm', 'c3fugt', 'c3fugtraw', 'c2fs2', 'c3fs2', 'c2f40', 'c3f40', 'g1', 'c3ml']; 100 soggetti, 148500 righe | - |
| devfs 21096: celle presenti (stesse righe, stessi soggetti) | ['c3mv', 'c2m', 'c2f', 'c3f', 'c2fgnm', 'c3fugt', 'c3fugtraw', 'c2fs2', 'c3fs2', 'c2f40', 'c3f40', 'g1', 'c3ml']; 100 soggetti, 148500 righe | - |
| flame 10548: celle presenti (stesse righe, stessi soggetti) | ['c3mv', 'c2m', 'c2f', 'c3f', 'c2fgnm', 'c3fugt', 'c3fugtraw', 'c2fs2', 'c3fs2', 'c2f40', 'c3f40', 'g1', 'c3ml']; 100 soggetti, 148500 righe | - |
| flame 21096: celle presenti (stesse righe, stessi soggetti) | ['c3mv', 'c2m', 'c2f', 'c3f', 'c2fgnm', 'c3fugt', 'c3fugtraw', 'c2fs2', 'c3fs2', 'c2f40', 'c3f40', 'g1', 'c3ml']; 100 soggetti, 148500 righe | - |
| hifi C3M 10548: embedding contro latent_distance, max |diff| | 6.06e-07 | < 1e-4 |
| hifi C3M 21096: embedding contro latent_distance, max |diff| | 6.02e-07 | < 1e-4 |
| hifi C2M 10548: embedding contro latent_distance, max |diff| | 7.00e-07 | < 1e-4 |
| hifi C2M 21096: embedding contro latent_distance, max |diff| | 7.40e-07 | < 1e-4 |
| hifi C2F 10548: embedding contro latent_distance, max |diff| | 6.71e-07 | < 1e-4 |
| hifi C2F 21096: embedding contro latent_distance, max |diff| | 7.02e-07 | < 1e-4 |
| hifi C3F 10548: embedding contro latent_distance, max |diff| | 6.08e-07 | < 1e-4 |
| hifi C3F 21096: embedding contro latent_distance, max |diff| | 5.99e-07 | < 1e-4 |
| hifi C2F-GNM 10548: embedding contro latent_distance, max |diff| | 5.89e-07 | < 1e-4 |
| hifi C2F-GNM 21096: embedding contro latent_distance, max |diff| | 5.97e-07 | < 1e-4 |
| hifi C3F-UGT 10548: embedding contro latent_distance, max |diff| | 5.94e-07 | < 1e-4 |
| hifi C3F-UGT 21096: embedding contro latent_distance, max |diff| | 6.04e-07 | < 1e-4 |
| hifi C3F-UGT non tarata 10548: embedding contro latent_distance, max |diff| | 5.76e-07 | < 1e-4 |
| hifi C3F-UGT non tarata 21096: embedding contro latent_distance, max |diff| | 6.07e-07 | < 1e-4 |
| hifi C2F s2 10548: embedding contro latent_distance, max |diff| | 7.08e-07 | < 1e-4 |
| hifi C2F s2 21096: embedding contro latent_distance, max |diff| | 7.03e-07 | < 1e-4 |
| hifi C3F s2 10548: embedding contro latent_distance, max |diff| | 5.88e-07 | < 1e-4 |
| hifi C3F s2 21096: embedding contro latent_distance, max |diff| | 6.47e-07 | < 1e-4 |
| hifi C2F40 10548: embedding contro latent_distance, max |diff| | 6.83e-07 | < 1e-4 |
| hifi C2F40 21096: embedding contro latent_distance, max |diff| | 7.02e-07 | < 1e-4 |
| hifi C3F40 10548: embedding contro latent_distance, max |diff| | 5.87e-07 | < 1e-4 |
| hifi C3F40 21096: embedding contro latent_distance, max |diff| | 6.01e-07 | < 1e-4 |
| hifi G1 10548: embedding contro latent_distance, max |diff| | 6.20e-07 | < 1e-4 |
| hifi G1 21096: embedding contro latent_distance, max |diff| | 6.04e-07 | < 1e-4 |
| hifi C3M L40S (rif.) 10548: embedding contro latent_distance, max |diff| | 6.06e-07 | < 1e-4 |
| hifi C3M L40S (rif.) 21096: embedding contro latent_distance, max |diff| | 6.08e-07 | < 1e-4 |
| NoW C3M 10548: tau ricalcolato contro concordance.csv della cella | 0.339015 / 0.339015 | uguali |
| NoW C3M 21096: tau ricalcolato contro concordance.csv della cella | 0.386364 / 0.386364 | uguali |
| NoW C2M 10548: tau ricalcolato contro concordance.csv della cella | 0.198864 / 0.198864 | uguali |
| NoW C2M 21096: tau ricalcolato contro concordance.csv della cella | 0.145833 / 0.145833 | uguali |
| NoW C2F 10548: tau ricalcolato contro concordance.csv della cella | 0.183712 / 0.183712 | uguali |
| NoW C2F 21096: tau ricalcolato contro concordance.csv della cella | 0.231061 / 0.231061 | uguali |
| NoW C3F 10548: tau ricalcolato contro concordance.csv della cella | 0.299242 / 0.299242 | uguali |
| NoW C3F 21096: tau ricalcolato contro concordance.csv della cella | 0.274621 / 0.274621 | uguali |
| NoW C2F-GNM 10548: tau ricalcolato contro concordance.csv della cella | 0.109848 / 0.109848 | uguali |
| NoW C2F-GNM 21096: tau ricalcolato contro concordance.csv della cella | 0.094697 / 0.094697 | uguali |
| NoW C3F-UGT 10548: tau ricalcolato contro concordance.csv della cella | 0.357955 / 0.357955 | uguali |
| NoW C3F-UGT 21096: tau ricalcolato contro concordance.csv della cella | 0.244318 / 0.244318 | uguali |
| NoW C3F-UGT non tarata 10548: tau ricalcolato contro concordance.csv della cella | 0.363636 / 0.363636 | uguali |
| NoW C3F-UGT non tarata 21096: tau ricalcolato contro concordance.csv della cella | 0.253788 / 0.253788 | uguali |
| NoW C2F s2 10548: tau ricalcolato contro concordance.csv della cella | 0.236742 / 0.236742 | uguali |
| NoW C2F s2 21096: tau ricalcolato contro concordance.csv della cella | 0.212121 / 0.212121 | uguali |
| NoW C3F s2 10548: tau ricalcolato contro concordance.csv della cella | 0.310606 / 0.310606 | uguali |
| NoW C3F s2 21096: tau ricalcolato contro concordance.csv della cella | 0.221591 / 0.221591 | uguali |
| NoW C2F40 10548: tau ricalcolato contro concordance.csv della cella | 0.219697 / 0.219697 | uguali |
| NoW C2F40 21096: tau ricalcolato contro concordance.csv della cella | 0.202652 / 0.202652 | uguali |
| NoW C3F40 10548: tau ricalcolato contro concordance.csv della cella | 0.373106 / 0.373106 | uguali |
| NoW C3F40 21096: tau ricalcolato contro concordance.csv della cella | 0.348485 / 0.348485 | uguali |
| NoW G1 10548: tau ricalcolato contro concordance.csv della cella | -0.045455 / -0.045455 | uguali |
| NoW G1 21096: tau ricalcolato contro concordance.csv della cella | 0.119318 / 0.119318 | uguali |
| NoW C3M L40S (rif.) 10548: tau ricalcolato contro concordance.csv della cella | 0.299242 / 0.299242 | uguali |
| NoW C3M L40S (rif.) 21096: tau ricalcolato contro concordance.csv della cella | 0.346591 / 0.346591 | uguali |
| HIFI3D C3M L40S 10548 nocrop_cross: rivalutato su A100 contro curve.md (L40S), |diff| del punto | 5.20e-05 | ~0 (hardware di valutazione) |
| HIFI3D C3M L40S 10548 all_cross: rivalutato su A100 contro curve.md (L40S), |diff| del punto | 4.93e-05 | ~0 (hardware di valutazione) |
| HIFI3D C3M L40S 10548 subject_pair_mean: rivalutato su A100 contro curve.md (L40S), |diff| del punto | 3.71e-05 | ~0 (hardware di valutazione) |
| HIFI3D C3M L40S 21096 nocrop_cross: rivalutato su A100 contro curve.md (L40S), |diff| del punto | 7.01e-06 | ~0 (hardware di valutazione) |
| HIFI3D C3M L40S 21096 all_cross: rivalutato su A100 contro curve.md (L40S), |diff| del punto | 3.65e-05 | ~0 (hardware di valutazione) |
| HIFI3D C3M L40S 21096 subject_pair_mean: rivalutato su A100 contro curve.md (L40S), |diff| del punto | 1.03e-05 | ~0 (hardware di valutazione) |
| HIFI3D GT unificata fb_rigid_icp_chamfer (10548 passi) contro e8/methods_spearman.csv, |diff| del punto | 7.11e-08 | ~0 |
| HIFI3D GT unificata fb_nicp_p2tri (10548 passi) contro e8/methods_spearman.csv, |diff| del punto | 4.07e-08 | ~0 |
| HIFI3D GT unificata scale_e036 (10548 passi) contro e8/methods_spearman.csv, |diff| del punto | 3.42e-05 | ~0 |
| HIFI3D GT unificata fb_rigid_icp_chamfer (21096 passi) contro e8/methods_spearman.csv, |diff| del punto | 7.11e-08 | ~0 |
| HIFI3D GT unificata fb_nicp_p2tri (21096 passi) contro e8/methods_spearman.csv, |diff| del punto | 4.07e-08 | ~0 |
| HIFI3D GT unificata scale_e072 (21096 passi) contro e8/methods_spearman.csv, |diff| del punto | 1.19e-05 | ~0 |
| hifi riconoscimento C3M L40S 10548: A100 contro data_scale_ood/arcface_vs_scale_hifi3d/recognition.csv, max |diff| | 1.00e-03 | ~0 |
| hifi riconoscimento C3M L40S 21096: A100 contro data_scale_ood/arcface_vs_scale_hifi3d/recognition.csv, max |diff| | 1.28e-04 | ~0 |
| fv riconoscimento C3M L40S 10548: A100 contro data_scale_ood/fvexpr_partial/recognition.csv, max |diff| | 8.38e-06 | ~0 |
| NoW C3M L40S 10548: A100 contro now_eval_scale_e036 | 0.00e+00 | ~0 |
| NoW C3M L40S 21096: A100 contro now_eval_scale_e072 | 1.89e-03 | ~0 |
