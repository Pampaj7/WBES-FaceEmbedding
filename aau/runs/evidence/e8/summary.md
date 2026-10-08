# E8: GT unificata e controlli fra domini (2026-10-08)

Piano: `paper/PLAN_MASSIVE.md`, sezioni 3 e 13 (E8). Generato da `v3_work/unified_gt/report.py`; le conclusioni in testa da `conclusions.md`.

## Conclusioni (8 ottobre 2026, sui numeri delle sezioni sotto)

**1. Corrispondenze: il residuo sui landmark e' basso per tutti i 7 template.**
- **Residuo dopo NICP:** mediana 0.42-0.72 mm.
- **Landmark tenuti fuori dal NICP:** 0.61-1.28 mm, con p95 al massimo 3.5 mm. Il peggiore e' FaceVerse.
- **Chamfer regione -> template:** 0.17-0.24 mm.
- **Distorsione:** al massimo 3 triangoli capovolti su 3.169.

La condizione di E8 per il run ("corrispondenze con un residuo sui landmark accettabile") per me e' soddisfatta. Pero' il piano non fissa una soglia, quindi la decisione resta del PI.

Il residuo comprende il rumore del detector: FaceMesh su render senza texture da' 1.4-2.6 mm gia' dopo la sola similarita'.

Il template BFM e' la media delle 500 original REMESH. Dista 0.10 mm (RMS) dalla media `u_shp` di 3DDFA ritagliata, e 0.28 mm da `u_shp + u_exp`: le identita' REMESH sono centrate sulla media di BFM2009, senza l'espressione media.

**2. La regione comune NON va da orecchio a orecchio.**
- Sono 1.478 vertici, cioe' il 59% dell'area della regione FLAME.
- **Il limite e' il crop p23470 di BFM:** perde le guance laterali, 104 vertici solo per colpa sua. FaceVerse toglie la parte alta della fronte.
- **Senza il vincolo di BFM** la regione sale all'88%. E' salvata come `ridx_nobfm`, ma non e' usata.

Se BFM 3DDFA viene sostituito (BFM 2017/2019), conviene rifare la regione.

**3. Le trasformazioni canoniche sono scritte e controllate da lettore.**
- Il file e' `datasets/UNIFIED_GT/canonical_transforms.json`.
- Sulle mesh di dati prese dal disco, le normali escono in tutti i domini. BFM richiede `flip_faces`.
- Senza altri allineamenti, l'RMS dalla media FLAME e' di 1.9-6.5 mm.

**4. (a) La GT unificata misura un'altra cosa rispetto alla maxabs, soprattutto su HIFI3D.**
- **Spearman unificata contro maxabs:** 0.54 [0.47, 0.61] su HIFI3D e 0.69 [0.64, 0.74] su FaceVerse.
- **Contro i coefficienti:** 0.13 e 0.50.

La catena delle GT intermedie dice da dove viene la differenza:
- **Su HIFI3D e' l'allineamento.** Il passaggio da maxabs a Procrustes di similarita', sulla stessa patch nativa, vale da solo 0.65.
- **Su FaceVerse pesa anche la regione.** Il passaggio dalla patch intera alla regione comune vale 0.80.
- **La mappa FLAME non cambia nulla.** L'unificata coincide con la GT nativa calcolata sulla stessa regione (0.999).

La variante con Procrustes per coppia coincide con quella globale (Spearman 0.99999, anche fra domini diversi): resta la globale, che costa meno.

**5. (b) Nessun quasi-duplicato di HIFI3D nel training, ma GNM e' sistematicamente il dominio piu' vicino.**
- **Nessun duplicato:** nessuna identita' di test ha un vicino di training piu' vicino della coppia piu' stretta interna al pool HIFI3D.
- **Vicino piu' prossimo, mediana:** GNM 2.73 mm, ICT 2.79, BFM 3.45, contro 2.54 dentro il pool di 500.
- **GNM a parita' di taglia:**
  - con 392 identita' e' piu' vicino del 6% rispetto a ICT (log-rapporto -0.065 [-0.078, -0.053]) e del 13% rispetto a BFM;
  - con 10.000 identita' resta piu' vicino del 6% rispetto a ICT;
  - e' piu' vicino per l'83% delle identita' di test.
- **FaceVerse** e' il dominio piu' disperso: distanza mediana fra coppie 7.0 mm, contro 3.6-4.3 degli altri. Le sue identita' sono piu' vicine ai pool di training che fra loro; per 24 su 100 il vicino di training e' sotto il minimo interno. Non sono copie: FaceVerse e' largo, non lontano. Anche qui GNM e' il piu' vicino (-4%).

**6. (c) Le medie dei domini formano due gruppi.**
- **Gruppo A:** FLAME, BFM, ICT e Multiface, a 1.4-2.2 mm fra loro.
- **Gruppo B:** FaceScape, HIFI3D e FaceVerse, a 0.8-1.7 mm fra loro. Sono i tre modelli costruiti su popolazioni est-asiatiche; e' un'ipotesi, non l'ho verificata.
- **Fra i due gruppi:** 3.2-3.8 mm, quanto la distanza tipica fra due identita' dello stesso dominio.
- **GNM sta in mezzo:** a 1.7-2.2 mm da tutti, ed e' la media di training piu' vicina a HIFI3D (2.16 mm, contro ICT 3.26 e BFM 3.54).

Il salto di HIFI3D con GNM (E1) e' quindi compatibile anche con la vicinanza di dominio. FaceScape dista 0.84 mm da HIFI3D: se entra nel training, HIFI3D smette di essere un test lontano, e va dichiarato.

**7. Richiesta del critic: tutti i metodi con la GT unificata, stessi soggetti, righe e repliche. L'ordine cambia.**

Controllo di riproduzione con la GT maxabs: i punti pubblicati tornano identici, e la differenza appaiata e108 - Chamfer eval torna identica col suo IC.

- **HIFI3D, mesh-pair senza crop (il primario):**

  | | GT maxabs | GT unificata |
  | --- | --- | --- |
  | NICP P2Tri | 0.389 | **0.577** |
  | ICP + Chamfer | 0.355 | 0.522 |
  | e036 | 0.677 | 0.340 |
  | e072 | 0.663 | 0.324 |
  | e108 | 0.630 | 0.301 |
  | congiunto 1019532 | 0.428 | 0.266 |
  | ArcFace | 0.24 | 0.24-0.25 (invariata) |
  | Chamfer eval | 0.372 | 0.230 |

  Differenze appaiate con la GT unificata:
  - e108 - Chamfer eval: +0.071 [+0.015, +0.123], contro +0.258 con la maxabs;
  - NICP - e108: +0.277 [+0.198, +0.355].

  Con la GT nativa di similarita', che non usa la mappa FLAME, il quadro e' lo stesso.
- **HIFI3D, subject-pair-mean:** con la GT unificata il modello non batte piu' Chamfer (e108 - Chamfer eval -0.045 [-0.123, +0.030]). NICP vale 0.632.
- **HIFI3D, original -> original** (la riga citata dal critic): Chamfer faceBench scende da 0.876 a 0.464, ICP sale da 0.401 a 0.596, NICP vale 0.632, e108 0.385.
- **FaceVerse con espressioni** (secondario, GT d'identita' neutra): tutti i metodi stanno fra 0.13 e 0.22, senza differenze significative fra modelli e baseline (e108 - Chamfer eval -0.015 [-0.076, +0.047]). e072 manca: job 1061482 in coda.

**Cosa vuol dire.** Il vantaggio del modello sulla distanza graduata di HIFI3D (0.63-0.68) viene in buona parte dalla parte della GT maxabs che non e' invariante alla similarita'. Sono rotazione, traslazione e scala contenute nei modi d'identita' del 3DMM: il modello le vede, perche' riceve xyz nel frame del dato, mentre ICP e NICP le tolgono.

Con una GT invariante alla similarita':
- il modello batte ancora Chamfer e ArcFace sulle coppie fra topologie diverse;
- resta pero' molto sotto ICP e NICP.

Il primario dichiarato resta la maxabs. Questi numeri pesano su E2 (canonicalizzazione) e sulla scelta della GT di training: decide il PI.

**Non fatto o fuori scope:**
- accordo con lo studio umano (E8 lo cita);
- identita' FaceScape: solo il template, il core e' di 4.9 GB e non serviva per (a)-(c);
- e072 su FaceVerse;
- Multiface: la trasformazione vale solo per il template, perche' le mesh tracked hanno la posa di ogni frame.

## 1. Regione del volto e corrispondenze

- **Maschere ufficiali trovate:** `v2_work/genflame/official/FLAME_masks.pkl`. Maschera `face`: 1787 vertici, intersezione nulla con orecchie, collo, bulbi oculari e `boundary` (23 vertici in comune con `scalp`, sul bordo della fronte).
- **Interno di occhi e bocca** tolto con una regola geometrica (visibilita' da un cono di 75 gradi attorno a +z, occlusori = testa intera con i bulbi): 116 vertici (105 nelle labbra, 7 attorno agli occhi). Regione FLAME: **1671 vertici**, 40141 mm^2. Figura: `flame_region.png`.
- **Regione unificata** (coperta da tutti gli 8 domini): **1478 vertici**, 23584 mm^2 sulla media FLAME (59% dell'area della regione FLAME). Vertici persi solo per colpa di un dominio: bfm 104, gnm 29, facescape 10, faceverse 7, multiface 1. Senza il vincolo di BFM: 1582 vertici, 35333 mm^2 (salvata come `ridx_nobfm`, non usata). Figura: `unified_region.png`.

Controlli delle corrispondenze (mm sulla scala della media FLAME). Landmark: mediana del residuo dopo la sola similarita' -> dopo NICP, sui landmark usati e su un 20% tenuto FUORI da un secondo NICP. Chamfer: regione deformata -> template (media / p95, vertici coperti) e template -> regione (mediana / p95, vertici del template nell'impronta della regione; la coda viene da narici e orbite profonde, che la regione FLAME non ha). Distorsione: mediana di |log rapporto d'area| dei triangoli FLAME / triangoli capovolti. Figure: `corr_<dominio>.png`.

| dominio | vertici template | landmark | residuo landmark | landmark tenuti fuori | Chamfer regione->template | Chamfer template->regione | coperti | distorsione / capovolti |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bfm | 23470 | 308 | 1.76 -> 0.64 (p95 1.90) | 2.07 -> 0.98 (p95 2.54, n=62) | 0.22 / 0.60 | 0.18 / 0.61 | 1536/1671 | 0.061 / 0 |
| ict | 9409 | 331 | 1.71 -> 0.60 (p95 1.70) | 1.79 -> 0.85 (p95 2.32, n=66) | 0.19 / 0.52 | 0.17 / 4.02 | 1671/1671 | 0.057 / 0 |
| gnm | 17821 | 327 | 1.40 -> 0.42 (p95 1.27) | 1.28 -> 0.61 (p95 1.93, n=65) | 0.17 / 0.47 | 0.15 / 0.50 | 1632/1671 | 0.051 / 0 |
| facescape | 26278 | 324 | 2.57 -> 0.60 (p95 1.93) | 2.77 -> 0.94 (p95 3.07, n=65) | 0.23 / 0.64 | 0.20 / 0.85 | 1653/1671 | 0.090 / 3 |
| hifi3d | 20481 | 324 | 2.58 -> 0.60 (p95 2.06) | 2.36 -> 0.80 (p95 3.14, n=65) | 0.22 / 0.67 | 0.20 / 0.88 | 1671/1671 | 0.096 / 3 |
| faceverse | 28632 | 328 | 2.53 -> 0.72 (p95 2.27) | 2.68 -> 1.28 (p95 3.50, n=66) | 0.24 / 0.66 | 0.21 / 0.84 | 1634/1671 | 0.125 / 2 |
| multiface | 5471 | 337 | 1.93 -> 0.56 (p95 1.78) | 1.75 -> 0.69 (p95 2.56, n=67) | 0.23 / 0.61 | 0.23 / 0.79 | 1669/1671 | 0.049 / 0 |

## 2. Trasformazioni canoniche

`datasets/UNIFIED_GT/canonical_transforms.json`: x_canon = s * R @ x + t (R 3x3 righe per riga, x vettore colonna nel frame e nelle unita' dei dati del dominio). Frame canonico = FLAME 2020: +y in alto, +z fuori dal volto, millimetri, origine di FLAME. Similarita' ai minimi quadrati (pesi = area FLAME per vertice) dalla regione unificata del template medio del dominio alla media FLAME. flip_faces: invertire il verso dei triangoli (F[:, ::-1]) per avere normali uscenti.

| dominio | scala s | rotazione (gradi) | flip_faces | residuo verso la media FLAME (mm, rms) | unita' dei dati |
| --- | --- | --- | --- | --- | --- |
| flame | 1000 | 0.0 | no | 0.00 | metri |
| bfm | 0.00103693 | 176.0 | si' | 1.43 | micrometri (BFM) |
| ict | 9.64608 | 3.4 | no | 1.77 | ICT (circa centimetri) |
| gnm | 985.579 | 3.1 | no | 1.72 | metri |
| facescape | 1.00296 | 5.8 | no | 3.65 | FaceScape (millimetri) |
| hifi3d | 9.30664 | 9.0 | no | 3.55 | unita' del .mat |
| faceverse | 165.816 | 175.4 | no | 3.40 | unita' del .npy |
| multiface | 0.990979 | 5.2 | no | 1.76 | millimetri |

Il residuo e' la differenza di forma fra la media del dominio e quella di FLAME dopo la similarita' (non un errore): e' la parte che la GT unificata vede come distanza fra le medie.

Controllo da lettore (`check_canonical.py`): una mesh di DATI per dominio presa dal disco, trasformata col json (e facce invertite se `flip_faces`), nessun altro allineamento. Normale media dei triangoli davanti lungo +z (1 = uscente), RMS dalla media FLAME sulla regione unificata (include la differenza d'identita'; un asse sbagliato darebbe decine di mm).

| dominio | mesh | normale z | RMS dalla media FLAME (mm) |
| --- | --- | --- | --- |
| flame | v_template di FLAME 2020 | 1.0000 | 0.00 |
| bfm | REMESH id0007 original | 1.0000 | 3.06 |
| ict | ICT-5000 ict0007 | 0.9999 | 3.57 |
| gnm | GNM_DISTILL id100007 (patch) | 0.9999 | 3.88 |
| hifi3d | HIFI3D hifi0007 (patch) | 1.0000 | 6.49 |
| faceverse | FaceVerse fv0007 | 1.0000 | 4.75 |
| facescape | template FaceScape (nessuna identita' su disco) | 0.9994 | 3.74 |
| multiface | template Multiface (le mesh tracked hanno pose per frame) | 1.0000 | 1.88 |

## 3. GT unificata

- s_i: forma neutra -> mappa baricentrica sulla regione unificata (1478 vertici) -> Procrustes di similarita' pesato per area verso la media globale mu -> pesi sqrt(area). mu = Procrustes generalizzato delle medie dei domini di training (flame, bfm, ict, gnm, facescape), a scala e frame FLAME. Area totale A = 22819 mm^2. g_ij = ||s_i - s_j|| / sqrt(A): RMS pesato per area, in mm.
- Dati: `datasets/UNIFIED_GT/` (`unified_space.npz`, `shapes/<set>.npz`, `gt/<set>_unified*.npz` nel formato di `load_gt_distance_matrix`). Codice: `v3_work/unified_gt/`.

| set | identita' | RMS da mu (mm, mediana) | mappa lineare vs mesh generata (max |diff|) | mesh dai pesi vs mesh su disco (max |diff|, unita' del dominio) |
| --- | --- | --- | --- | --- |
| hifi3d | 500 | 3.84 | 3.6e-15 | 4.7e-07 |
| faceverse | 500 | 5.48 | 1.1e-16 | 3.0e-08 |
| bfm | 500 | 3.16 | - | - |
| gnm | 10100 | 3.01 | 1.1e-16 | 1.5e-08 |
| flame | 1000 | 3.02 | 1.4e-17 | - |
| multiface | 13 | 3.60 | - | - |
| ict | 55000 | 2.83 | 5.3e-15 | 9.5e-07 |

## 4. (a) Accordo fra GT sugli stessi soggetti di valutazione

Spearman sulle 4.950 coppie dei 100 soggetti valutati (IC 95% bootstrap per soggetto, 1000 repliche, seme 1234) e, solo punto, sulle 124.750 coppie del pool di 500. Le righe 1-5 cambiano un ingrediente alla volta da maxabs all'unificata (GT intermedie di `native_gt.py`, sulla patch `original` delle viste); tutte le coppie in `evidence.json`.

| dominio | GT | Spearman (100 soggetti) | Spearman (pool 500) |
| --- | --- | --- | --- |
| hifi3d | maxabs contro coefficienti | 0.159 [0.064, 0.253] | 0.089 |
| hifi3d | unificata contro coefficienti | 0.134 [0.038, 0.231] | 0.148 |
| hifi3d | **unificata contro maxabs** | 0.543 [0.470, 0.612] | 0.535 |
| hifi3d | 1. media delle norme -> RMS (stessa normalizzazione maxabs) | 0.992 [0.990, 0.994] | 0.993 |
| hifi3d | 2. maxabs -> Procrustes di similarita' (patch nativa) | 0.655 [0.594, 0.716] | 0.651 |
| hifi3d | 3. vertici uniformi -> pesi d'area | 0.907 [0.885, 0.926] | 0.899 |
| hifi3d | 4. patch intera -> impronta della regione unificata | 0.910 [0.886, 0.930] | 0.898 |
| hifi3d | 5. vertici nativi -> mappa FLAME (stessa regione) | 0.999 [0.999, 0.999] | 0.999 |
| hifi3d | variante: Procrustes per coppia invece che verso mu | 1.000 [1.000, 1.000] | 1.000 |
| faceverse | maxabs contro coefficienti | 0.514 [0.437, 0.581] | 0.526 |
| faceverse | unificata contro coefficienti | 0.499 [0.428, 0.564] | 0.536 |
| faceverse | **unificata contro maxabs** | 0.693 [0.636, 0.744] | 0.673 |
| faceverse | 1. media delle norme -> RMS (stessa normalizzazione maxabs) | 0.979 [0.974, 0.984] | 0.978 |
| faceverse | 2. maxabs -> Procrustes di similarita' (patch nativa) | 0.836 [0.790, 0.874] | 0.829 |
| faceverse | 3. vertici uniformi -> pesi d'area | 0.977 [0.971, 0.982] | 0.971 |
| faceverse | 4. patch intera -> impronta della regione unificata | 0.801 [0.762, 0.836] | 0.772 |
| faceverse | 5. vertici nativi -> mappa FLAME (stessa regione) | 0.998 [0.998, 0.999] | 0.999 |
| faceverse | variante: Procrustes per coppia invece che verso mu | 1.000 [1.000, 1.000] | 1.000 |

- hifi3d: g unificata sui 100 soggetti mediana 4.40 mm (min 2.16, p5 3.17, max 9.37); variante per coppia / globale, rapporto mediano 0.999, Pearson 1.0000.
- faceverse: g unificata sui 100 soggetti mediana 7.05 mm (min 4.86, p5 5.92, max 10.33); variante per coppia / globale, rapporto mediano 0.998, Pearson 1.0000.
- Variante con Procrustes per coppia contro globale, campione fra domini (613 forme, 100 per set): Spearman 1.0000 su tutte le coppie, 1.0000 sulle coppie fra domini diversi; rapporto mediano 0.999.

## 5. (b) Quasi-duplicati fra domini (training del run su scala contro test)

Pool di training (split `aau/data_scale/split_scale_all.json`): bfm 392, ict 54008, gnm 10000. Distanze unificate in mm; per identita' di test il vicino piu' prossimo in ogni pool.

**hifi3d** (100 identita' di test)

| vicino piu' prossimo in | mediana | minimo | p5 |
| --- | --- | --- | --- |
| dentro il test (99 altri) | 2.830 | 2.164 | 2.448 |
| dentro il pool di 500 (499 altri) | 2.541 | 1.866 | 2.207 |
| training bfm (tutto) | 3.448 | 2.529 | 2.820 |
| training ict (tutto) | 2.791 | 2.087 | 2.239 |
| training gnm (tutto) | 2.734 | 2.050 | 2.199 |
| training bfm, 392 identita' (mediana su 50 sottocampioni) | 3.448 | 2.529 | 2.820 |
| training ict, 392 identita' (mediana su 50 sottocampioni) | 3.249 | 2.361 | 2.601 |
| training gnm, 392 identita' (mediana su 50 sottocampioni) | 2.993 | 2.301 | 2.473 |
| training ict, 10000 identita' (mediana su 50 sottocampioni) | 2.882 | 2.130 | 2.342 |
| training gnm, 10000 identita' (mediana su 50 sottocampioni) | 2.734 | 2.050 | 2.199 |

- identita' di test col vicino di training piu' vicino del vicino dentro il pool di 500: 35/100; piu' vicino del minimo di tutto il test: 0; dominio del vicino piu' prossimo (pool interi): bfm 0, ict 38, gnm 62.

A taglia uguale, per identita' di test: rapporto fra le distanze dal vicino (primo / secondo dominio), media geometrica e IC bootstrap sui soggetti di test del log-rapporto medio.

| confronto | rapporto (media geom.) | log-rapporto medio [IC 95%] | frazione col primo piu' vicino |
| --- | --- | --- | --- |
| n392_gnm_vs_ict | 0.937 | -0.065 [-0.078, -0.053] | 0.83 |
| n392_gnm_vs_bfm | 0.873 | -0.136 [-0.155, -0.118] | 0.96 |
| n392_ict_vs_bfm | 0.931 | -0.071 [-0.091, -0.051] | 0.76 |
| n10000_gnm_vs_ict | 0.943 | -0.059 [-0.072, -0.046] | 0.79 |

**faceverse** (100 identita' di test)

| vicino piu' prossimo in | mediana | minimo | p5 |
| --- | --- | --- | --- |
| dentro il test (99 altri) | 5.596 | 4.860 | 4.947 |
| dentro il pool di 500 (499 altri) | 5.333 | 4.252 | 4.859 |
| training bfm (tutto) | 5.086 | 3.969 | 4.413 |
| training ict (tutto) | 4.663 | 3.687 | 3.963 |
| training gnm (tutto) | 4.576 | 3.714 | 3.899 |
| training bfm, 392 identita' (mediana su 50 sottocampioni) | 5.086 | 3.969 | 4.413 |
| training ict, 392 identita' (mediana su 50 sottocampioni) | 5.029 | 3.914 | 4.269 |
| training gnm, 392 identita' (mediana su 50 sottocampioni) | 4.811 | 3.923 | 4.105 |
| training ict, 10000 identita' (mediana su 50 sottocampioni) | 4.757 | 3.704 | 4.035 |
| training gnm, 10000 identita' (mediana su 50 sottocampioni) | 4.576 | 3.714 | 3.899 |

- identita' di test col vicino di training piu' vicino del vicino dentro il pool di 500: 99/100; piu' vicino del minimo di tutto il test: 24; dominio del vicino piu' prossimo (pool interi): bfm 2, ict 27, gnm 71.

A taglia uguale, per identita' di test: rapporto fra le distanze dal vicino (primo / secondo dominio), media geometrica e IC bootstrap sui soggetti di test del log-rapporto medio.

| confronto | rapporto (media geom.) | log-rapporto medio [IC 95%] | frazione col primo piu' vicino |
| --- | --- | --- | --- |
| n392_gnm_vs_ict | 0.961 | -0.039 [-0.047, -0.032] | 0.84 |
| n392_gnm_vs_bfm | 0.945 | -0.056 [-0.066, -0.045] | 0.86 |
| n392_ict_vs_bfm | 0.983 | -0.017 [-0.026, -0.007] | 0.63 |
| n10000_gnm_vs_ict | 0.960 | -0.041 [-0.049, -0.033] | 0.86 |

## 6. (c) Distanze fra le medie dei domini (mm, spazio unificato)

|  | flame | bfm | ict | gnm | facescape | hifi3d | faceverse | multiface |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flame | 0.00 | 1.43 | 1.76 | 1.70 | 3.63 | 3.53 | 3.37 | 1.75 |
| bfm | 1.43 | 0.00 | 2.15 | 1.98 | 3.78 | 3.54 | 3.26 | 1.87 |
| ict | 1.76 | 2.15 | 0.00 | 1.80 | 3.18 | 3.26 | 3.24 | 1.70 |
| gnm | 1.70 | 1.98 | 1.80 | 0.00 | 2.23 | 2.16 | 2.22 | 2.05 |
| facescape | 3.63 | 3.78 | 3.18 | 2.23 | 0.00 | 0.84 | 1.74 | 3.62 |
| hifi3d | 3.53 | 3.54 | 3.26 | 2.16 | 0.84 | 0.00 | 1.38 | 3.59 |
| faceverse | 3.37 | 3.26 | 3.24 | 2.22 | 1.74 | 1.38 | 0.00 | 3.60 |
| multiface | 1.75 | 1.87 | 1.70 | 2.05 | 3.62 | 3.59 | 3.60 | 0.00 |

Dispersione interna (stessa metrica):

| set | identita' | distanza mediana fra coppie | vicino piu' prossimo (mediana) | distanza dalla media del template (mediana) | media empirica vs media del template |
| --- | --- | --- | --- | --- | --- |
| hifi3d | 500 | 4.31 | 2.51 | 3.04 | 0.28 |
| faceverse | 500 | 7.02 | 5.28 | 5.01 | 0.50 |
| bfm | 500 | 3.98 | 2.25 | 2.83 | 0.16 |
| ict | 55000 | 3.59 | 1.74 | 2.53 | 0.14 |
| gnm | 10100 | 4.16 | 2.00 | 2.94 | 0.20 |
| flame | 1000 | 3.90 | 2.09 | 2.76 | 0.21 |
| multiface | 13 | 4.29 | 3.34 | 3.16 | 0.17 |

Distanza mediana delle identita' di test dalle medie dei domini:

| test | flame | bfm | ict | gnm | facescape | hifi3d | faceverse | multiface |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | 4.68 | 4.73 | 4.45 | 3.83 | 3.19 | 3.16 | 3.45 | 4.68 |
| faceverse | 6.03 | 5.89 | 5.92 | 5.45 | 5.21 | 5.16 | 5.01 | 6.17 |

## 7. Metodi esistenti con la GT unificata (richiesta del critic, priorita' alta)

GT primaria = maxabs (dichiarata); l'unificata e' secondaria. Stessi soggetti, stesse righe e stesse repliche bootstrap dei summary esistenti (un seme per dominio e gruppo: quello della differenza pubblicata e108 - Chamfer eval); cambia solo la colonna della GT. Codice: `eval_methods.py`; csv: `methods_*.csv`. IC 95% bootstrap per soggetto, 1000 repliche.

### hifi3d, nocrop_cross

| metodo | GT maxabs | GT unificata | unificata - maxabs (appaiata) | unificata per coppia | nativa sim. | nativa sim. + area | nativa sim. + area, regione unificata | GT coef | rango maxabs -> unificata | NaN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + NICP P2Tri | 0.389 [0.303, 0.479] | 0.577 [0.513, 0.641] | +0.188 [+0.113, +0.264] | 0.577 | 0.579 | 0.543 | 0.572 | 0.139 | 5 -> 1 | 776 |
| ICP rigido + Chamfer | 0.355 [0.267, 0.438] | 0.522 [0.455, 0.586] | +0.167 [+0.093, +0.243] | 0.522 | 0.531 | 0.520 | 0.518 | 0.155 | 7 -> 2 | 0 |
| BFM+ICT+GNM e036 | 0.677 [0.604, 0.738] | 0.340 [0.262, 0.410] | -0.338 [-0.428, -0.242] | 0.340 | 0.385 | 0.451 | 0.334 | 0.110 | 1 -> 3 | 0 |
| BFM+ICT+GNM e072 | 0.663 [0.591, 0.725] | 0.324 [0.251, 0.389] | -0.339 [-0.428, -0.248] | 0.324 | 0.374 | 0.419 | 0.318 | 0.128 | 2 -> 4 | 0 |
| BFM+ICT+GNM e108 | 0.630 [0.562, 0.687] | 0.301 [0.233, 0.360] | -0.329 [-0.408, -0.242] | 0.301 | 0.355 | 0.386 | 0.295 | 0.131 | 3 -> 5 | 0 |
| BFM+ICT congiunto 1019532 | 0.428 [0.373, 0.477] | 0.266 [0.221, 0.315] | -0.162 [-0.218, -0.105] | 0.266 | 0.297 | 0.321 | 0.262 | 0.105 | 4 -> 6 | 0 |
| ArcFace ombreggiato (3 viste) | 0.245 [0.175, 0.322] | 0.250 [0.182, 0.316] | +0.005 [-0.063, +0.071] | 0.250 | 0.261 | 0.245 | 0.242 | 0.180 | 9 -> 7 | 0 |
| ArcFace normal map (3 viste) | 0.242 [0.163, 0.326] | 0.236 [0.166, 0.304] | -0.005 [-0.080, +0.072] | 0.236 | 0.251 | 0.235 | 0.230 | 0.192 | 10 -> 8 | 0 |
| Chamfer eval | 0.372 [0.323, 0.421] | 0.230 [0.185, 0.274] | -0.142 [-0.199, -0.089] | 0.230 | 0.246 | 0.264 | 0.227 | 0.080 | 6 -> 9 | 0 |
| Chamfer faceBench 4096 pt | 0.325 [0.282, 0.369] | 0.198 [0.157, 0.237] | -0.127 [-0.177, -0.080] | 0.198 | 0.215 | 0.225 | 0.196 | 0.069 | 8 -> 10 | 0 |

Delta appaiati contro Chamfer eval (P(<=0) fra parentesi):

| metodo - Chamfer eval | GT maxabs | GT unificata |
| --- | --- | --- |
| ICP + NICP P2Tri | +0.018 [-0.084, +0.126] (0.374) | +0.348 [+0.289, +0.412] (0.000) |
| ICP rigido + Chamfer | -0.017 [-0.117, +0.087] (0.607) | +0.292 [+0.228, +0.359] (0.000) |
| BFM+ICT+GNM e036 | +0.305 [+0.258, +0.352] (0.000) | +0.110 [+0.050, +0.168] (0.000) |
| BFM+ICT+GNM e072 | +0.291 [+0.237, +0.338] (0.000) | +0.095 [+0.038, +0.151] (0.000) |
| BFM+ICT+GNM e108 | +0.258 [+0.208, +0.301] (0.000) | +0.071 [+0.015, +0.123] (0.010) |
| BFM+ICT congiunto 1019532 | +0.056 [+0.017, +0.097] (0.001) | +0.036 [-0.005, +0.077] (0.035) |
| ArcFace ombreggiato (3 viste) | -0.127 [-0.214, -0.043] (0.999) | +0.020 [-0.059, +0.104] (0.265) |
| ArcFace normal map (3 viste) | -0.131 [-0.224, -0.034] (0.996) | +0.006 [-0.079, +0.098] (0.400) |
| Chamfer faceBench 4096 pt | -0.047 [-0.056, -0.037] (1.000) | -0.032 [-0.039, -0.024] (1.000) |

Delta appaiati contro BFM+ICT+GNM e108 (P(<=0) fra parentesi):

| metodo - BFM+ICT+GNM e108 | GT maxabs | GT unificata |
| --- | --- | --- |
| ICP + NICP P2Tri | -0.241 [-0.348, -0.125] (1.000) | +0.277 [+0.198, +0.355] (0.000) |
| ICP rigido + Chamfer | -0.275 [-0.383, -0.157] (1.000) | +0.221 [+0.141, +0.301] (0.000) |
| BFM+ICT+GNM e036 | +0.047 [+0.021, +0.078] (0.000) | +0.039 [+0.002, +0.073] (0.020) |
| BFM+ICT+GNM e072 | +0.033 [+0.017, +0.049] (0.000) | +0.023 [+0.004, +0.043] (0.015) |
| BFM+ICT congiunto 1019532 | -0.202 [-0.250, -0.149] (1.000) | -0.035 [-0.089, +0.019] (0.897) |
| ArcFace ombreggiato (3 viste) | -0.385 [-0.466, -0.293] (1.000) | -0.051 [-0.131, +0.040] (0.876) |
| ArcFace normal map (3 viste) | -0.388 [-0.482, -0.286] (1.000) | -0.065 [-0.151, +0.029] (0.910) |
| Chamfer eval | -0.258 [-0.301, -0.208] (1.000) | -0.071 [-0.123, -0.015] (0.990) |
| Chamfer faceBench 4096 pt | -0.305 [-0.350, -0.255] (1.000) | -0.103 [-0.155, -0.045] (1.000) |

### hifi3d, subject_pair_mean

| metodo | GT maxabs | GT unificata | unificata - maxabs (appaiata) | unificata per coppia | nativa sim. | nativa sim. + area | nativa sim. + area, regione unificata | GT coef | rango maxabs -> unificata | NaN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + NICP P2Tri | 0.420 [0.325, 0.508] | 0.632 [0.562, 0.696] | +0.212 [+0.133, +0.295] | 0.632 | 0.637 | 0.590 | 0.627 | 0.153 | 7 -> 1 | 0 |
| ICP rigido + Chamfer | 0.370 [0.270, 0.458] | 0.558 [0.478, 0.637] | +0.188 [+0.101, +0.272] | 0.558 | 0.568 | 0.547 | 0.554 | 0.167 | 8 -> 2 | 0 |
| BFM+ICT congiunto 1019532 | 0.720 [0.655, 0.779] | 0.467 [0.384, 0.548] | -0.253 [-0.345, -0.159] | 0.467 | 0.520 | 0.562 | 0.460 | 0.161 | 6 -> 3 | 0 |
| Chamfer faceBench 4096 pt | 0.751 [0.694, 0.803] | 0.458 [0.368, 0.551] | -0.293 [-0.392, -0.190] | 0.458 | 0.496 | 0.518 | 0.455 | 0.117 | 4 -> 4 | 0 |
| Chamfer eval | 0.743 [0.686, 0.796] | 0.455 [0.367, 0.542] | -0.288 [-0.384, -0.186] | 0.455 | 0.485 | 0.525 | 0.451 | 0.115 | 5 -> 5 | 0 |
| BFM+ICT+GNM e072 | 0.792 [0.731, 0.842] | 0.418 [0.326, 0.502] | -0.375 [-0.469, -0.276] | 0.418 | 0.475 | 0.528 | 0.410 | 0.144 | 2 -> 6 | 0 |
| BFM+ICT+GNM e108 | 0.795 [0.744, 0.842] | 0.410 [0.321, 0.491] | -0.385 [-0.476, -0.294] | 0.410 | 0.476 | 0.511 | 0.403 | 0.160 | 1 -> 7 | 0 |
| BFM+ICT+GNM e036 | 0.767 [0.702, 0.821] | 0.408 [0.312, 0.504] | -0.359 [-0.459, -0.255] | 0.408 | 0.459 | 0.535 | 0.402 | 0.118 | 3 -> 8 | 0 |
| ArcFace ombreggiato (3 viste) | 0.296 [0.204, 0.384] | 0.302 [0.214, 0.380] | +0.006 [-0.078, +0.087] | 0.302 | 0.315 | 0.294 | 0.294 | 0.219 | 9 -> 9 | 0 |
| ArcFace normal map (3 viste) | 0.270 [0.166, 0.366] | 0.266 [0.179, 0.343] | -0.004 [-0.085, +0.082] | 0.266 | 0.282 | 0.262 | 0.259 | 0.217 | 10 -> 10 | 0 |

Delta appaiati contro Chamfer eval (P(<=0) fra parentesi):

| metodo - Chamfer eval | GT maxabs | GT unificata |
| --- | --- | --- |
| ICP + NICP P2Tri | -0.324 [-0.439, -0.213] (1.000) | +0.177 [+0.079, +0.278] (0.000) |
| ICP rigido + Chamfer | -0.373 [-0.486, -0.262] (1.000) | +0.103 [-0.004, +0.217] (0.033) |
| BFM+ICT congiunto 1019532 | -0.023 [-0.085, +0.032] (0.777) | +0.012 [-0.056, +0.080] (0.379) |
| Chamfer faceBench 4096 pt | +0.007 [-0.005, +0.020] (0.098) | +0.003 [-0.009, +0.016] (0.285) |
| BFM+ICT+GNM e072 | +0.049 [-0.014, +0.104] (0.065) | -0.037 [-0.111, +0.036] (0.847) |
| BFM+ICT+GNM e108 | +0.052 [-0.004, +0.106] (0.039) | -0.045 [-0.123, +0.030] (0.867) |
| BFM+ICT+GNM e036 | +0.024 [-0.036, +0.079] (0.229) | -0.047 [-0.117, +0.020] (0.919) |
| ArcFace ombreggiato (3 viste) | -0.447 [-0.552, -0.350] (1.000) | -0.152 [-0.280, -0.034] (0.990) |
| ArcFace normal map (3 viste) | -0.473 [-0.589, -0.365] (1.000) | -0.189 [-0.318, -0.067] (0.999) |

Delta appaiati contro BFM+ICT+GNM e108 (P(<=0) fra parentesi):

| metodo - BFM+ICT+GNM e108 | GT maxabs | GT unificata |
| --- | --- | --- |
| ICP + NICP P2Tri | -0.376 [-0.485, -0.273] (1.000) | +0.222 [+0.126, +0.314] (0.000) |
| ICP rigido + Chamfer | -0.426 [-0.536, -0.322] (1.000) | +0.148 [+0.041, +0.251] (0.002) |
| BFM+ICT congiunto 1019532 | -0.075 [-0.130, -0.026] (0.999) | +0.057 [-0.005, +0.116] (0.037) |
| Chamfer faceBench 4096 pt | -0.045 [-0.098, +0.012] (0.941) | +0.048 [-0.030, +0.131] (0.119) |
| Chamfer eval | -0.052 [-0.106, +0.004] (0.961) | +0.045 [-0.030, +0.123] (0.133) |
| BFM+ICT+GNM e072 | -0.003 [-0.025, +0.017] (0.615) | +0.008 [-0.019, +0.031] (0.290) |
| BFM+ICT+GNM e036 | -0.028 [-0.063, +0.003] (0.962) | -0.002 [-0.049, +0.037] (0.534) |
| ArcFace ombreggiato (3 viste) | -0.499 [-0.591, -0.411] (1.000) | -0.108 [-0.214, +0.002] (0.972) |
| ArcFace normal map (3 viste) | -0.526 [-0.632, -0.432] (1.000) | -0.144 [-0.261, -0.025] (0.991) |

### hifi3d, original_to_original

| metodo | GT maxabs | GT unificata | unificata - maxabs (appaiata) | unificata per coppia | nativa sim. | nativa sim. + area | nativa sim. + area, regione unificata | GT coef | rango maxabs -> unificata | NaN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP + NICP P2Tri | 0.431 [0.343, 0.513] | 0.632 [0.564, 0.693] | +0.201 [+0.120, +0.274] | 0.632 | 0.651 | 0.591 | 0.627 | 0.137 | 6 -> 1 | 0 |
| ICP rigido + Chamfer | 0.401 [0.307, 0.490] | 0.596 [0.523, 0.667] | +0.195 [+0.114, +0.275] | 0.596 | 0.618 | 0.584 | 0.592 | 0.165 | 7 -> 2 | 0 |
| BFM+ICT congiunto 1019532 (conv. BFM) | 0.642 [0.580, 0.706] | 0.552 [0.473, 0.629] | -0.090 [-0.178, -0.012] | 0.553 | 0.586 | 0.645 | 0.544 | 0.105 | 5 -> 3 | 0 |
| Chamfer faceBench 4096 pt | 0.876 [0.844, 0.902] | 0.464 [0.376, 0.541] | -0.411 [-0.505, -0.330] | 0.464 | 0.526 | 0.524 | 0.460 | 0.162 | 1 -> 4 | 0 |
| BFM+ICT+GNM e108 | 0.818 [0.765, 0.859] | 0.385 [0.302, 0.464] | -0.433 [-0.517, -0.343] | 0.385 | 0.455 | 0.478 | 0.378 | 0.172 | 3 -> 5 | 0 |
| BFM+ICT+GNM e072 | 0.824 [0.769, 0.867] | 0.385 [0.300, 0.462] | -0.439 [-0.527, -0.352] | 0.385 | 0.447 | 0.484 | 0.378 | 0.153 | 2 -> 6 | 0 |
| BFM+ICT+GNM e036 | 0.802 [0.739, 0.851] | 0.372 [0.279, 0.453] | -0.431 [-0.524, -0.341] | 0.372 | 0.427 | 0.488 | 0.365 | 0.135 | 4 -> 7 | 0 |
| ArcFace ombreggiato (3 viste) | 0.285 [0.206, 0.364] | 0.288 [0.200, 0.375] | +0.003 [-0.076, +0.077] | 0.288 | 0.295 | 0.276 | 0.280 | 0.212 | 8 -> 8 | 0 |
| ArcFace normal map (3 viste) | 0.258 [0.175, 0.345] | 0.243 [0.167, 0.321] | -0.015 [-0.092, +0.061] | 0.243 | 0.261 | 0.238 | 0.237 | 0.200 | 9 -> 9 | 0 |

Delta appaiati contro BFM+ICT+GNM e108 (P(<=0) fra parentesi):

| metodo - BFM+ICT+GNM e108 | GT maxabs | GT unificata |
| --- | --- | --- |
| ICP + NICP P2Tri | -0.386 [-0.495, -0.284] (1.000) | +0.247 [+0.161, +0.334] (0.000) |
| ICP rigido + Chamfer | -0.417 [-0.524, -0.313] (1.000) | +0.211 [+0.116, +0.304] (0.000) |
| BFM+ICT congiunto 1019532 (conv. BFM) | -0.175 [-0.235, -0.117] (1.000) | +0.167 [+0.085, +0.254] (0.000) |
| Chamfer faceBench 4096 pt | +0.058 [+0.021, +0.103] (0.000) | +0.079 [+0.016, +0.141] (0.007) |
| BFM+ICT+GNM e072 | +0.006 [-0.013, +0.024] (0.262) | -0.000 [-0.025, +0.024] (0.520) |
| BFM+ICT+GNM e036 | -0.015 [-0.049, +0.016] (0.848) | -0.013 [-0.054, +0.023] (0.768) |
| ArcFace ombreggiato (3 viste) | -0.532 [-0.616, -0.440] (1.000) | -0.097 [-0.202, +0.013] (0.960) |
| ArcFace normal map (3 viste) | -0.560 [-0.653, -0.460] (1.000) | -0.142 [-0.242, -0.034] (0.996) |

### faceverse, mesh_pair_nocrop

| metodo | GT maxabs | GT unificata | unificata - maxabs (appaiata) | unificata per coppia | nativa sim. | nativa sim. + area | nativa sim. + area, regione unificata | GT coef | rango maxabs -> unificata | NaN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICP rigido + Chamfer | 0.259 [0.178, 0.338] | 0.218 [0.142, 0.289] | -0.041 [-0.102, +0.019] | 0.218 | 0.302 | 0.294 | 0.216 | 0.255 | 5 -> 1 | 0 |
| ICP + NICP P2Tri | 0.194 [0.116, 0.261] | 0.215 [0.145, 0.282] | +0.020 [-0.039, +0.077] | 0.215 | 0.233 | 0.216 | 0.215 | 0.281 | 8 -> 2 | 0 |
| Chamfer faceBench 4096 pt | 0.319 [0.255, 0.370] | 0.205 [0.142, 0.259] | -0.114 [-0.158, -0.072] | 0.205 | 0.260 | 0.248 | 0.203 | 0.197 | 1 -> 3 | 0 |
| Chamfer eval | 0.317 [0.255, 0.368] | 0.204 [0.143, 0.257] | -0.113 [-0.157, -0.072] | 0.204 | 0.269 | 0.255 | 0.202 | 0.200 | 2 -> 4 | 0 |
| BFM+ICT congiunto 1019532 (conv. BFM) | 0.282 [0.223, 0.341] | 0.194 [0.137, 0.252] | -0.088 [-0.132, -0.049] | 0.194 | 0.276 | 0.271 | 0.190 | 0.174 | 3 -> 5 | 0 |
| BFM+ICT+GNM e108 | 0.265 [0.201, 0.320] | 0.189 [0.135, 0.239] | -0.076 [-0.124, -0.029] | 0.189 | 0.269 | 0.264 | 0.184 | 0.161 | 4 -> 6 | 0 |
| BFM+ICT congiunto 1019532 (conv. ICT) | 0.243 [0.181, 0.302] | 0.156 [0.103, 0.214] | -0.087 [-0.130, -0.043] | 0.156 | 0.225 | 0.214 | 0.149 | 0.117 | 6 -> 7 | 0 |
| BFM+ICT+GNM e036 | 0.239 [0.183, 0.291] | 0.153 [0.105, 0.197] | -0.086 [-0.126, -0.046] | 0.153 | 0.230 | 0.229 | 0.150 | 0.132 | 7 -> 8 | 0 |
| ArcFace normal map (3 viste) | 0.100 [0.055, 0.147] | 0.137 [0.093, 0.184] | +0.037 [+0.002, +0.071] | 0.137 | 0.089 | 0.096 | 0.138 | 0.126 | 9 -> 9 | 0 |
| ArcFace ombreggiato (3 viste) | 0.082 [0.034, 0.131] | 0.132 [0.085, 0.178] | +0.050 [+0.006, +0.095] | 0.132 | 0.059 | 0.067 | 0.132 | 0.138 | 10 -> 10 | 0 |

Delta appaiati contro Chamfer eval (P(<=0) fra parentesi):

| metodo - Chamfer eval | GT maxabs | GT unificata |
| --- | --- | --- |
| ICP rigido + Chamfer | -0.058 [-0.135, +0.020] (0.916) | +0.014 [-0.063, +0.089] (0.352) |
| ICP + NICP P2Tri | -0.123 [-0.206, -0.040] (1.000) | +0.011 [-0.065, +0.084] (0.384) |
| Chamfer faceBench 4096 pt | +0.002 [-0.007, +0.010] (0.314) | +0.001 [-0.008, +0.010] (0.438) |
| BFM+ICT congiunto 1019532 (conv. BFM) | -0.035 [-0.097, +0.032] (0.864) | -0.010 [-0.076, +0.060] (0.608) |
| BFM+ICT+GNM e108 | -0.052 [-0.111, +0.006] (0.959) | -0.015 [-0.076, +0.047] (0.673) |
| BFM+ICT congiunto 1019532 (conv. ICT) | -0.074 [-0.123, -0.026] (0.999) | -0.048 [-0.103, +0.008] (0.948) |
| BFM+ICT+GNM e036 | -0.078 [-0.139, -0.019] (0.996) | -0.051 [-0.112, +0.006] (0.961) |
| ArcFace normal map (3 viste) | -0.217 [-0.286, -0.139] (1.000) | -0.067 [-0.138, +0.005] (0.959) |
| ArcFace ombreggiato (3 viste) | -0.234 [-0.306, -0.152] (1.000) | -0.072 [-0.145, +0.008] (0.959) |

Delta appaiati contro BFM+ICT+GNM e108 (P(<=0) fra parentesi):

| metodo - BFM+ICT+GNM e108 | GT maxabs | GT unificata |
| --- | --- | --- |
| ICP rigido + Chamfer | -0.005 [-0.095, +0.087] (0.534) | +0.029 [-0.057, +0.115] (0.260) |
| ICP + NICP P2Tri | -0.070 [-0.160, +0.017] (0.943) | +0.026 [-0.049, +0.107] (0.259) |
| Chamfer faceBench 4096 pt | +0.054 [-0.005, +0.116] (0.037) | +0.016 [-0.048, +0.081] (0.322) |
| Chamfer eval | +0.052 [-0.006, +0.111] (0.041) | +0.015 [-0.047, +0.076] (0.327) |
| BFM+ICT congiunto 1019532 (conv. BFM) | +0.018 [-0.025, +0.067] (0.198) | +0.005 [-0.038, +0.048] (0.390) |
| BFM+ICT congiunto 1019532 (conv. ICT) | -0.022 [-0.078, +0.041] (0.754) | -0.033 [-0.086, +0.028] (0.834) |
| BFM+ICT+GNM e036 | -0.025 [-0.057, +0.011] (0.917) | -0.036 [-0.069, -0.000] (0.976) |
| ArcFace normal map (3 viste) | -0.164 [-0.229, -0.088] (1.000) | -0.051 [-0.116, +0.017] (0.924) |
| ArcFace ombreggiato (3 viste) | -0.182 [-0.256, -0.101] (1.000) | -0.057 [-0.125, +0.019] (0.925) |

### faceverse, subject_pair_mean_nocrop

| metodo | GT maxabs | GT unificata | unificata - maxabs (appaiata) | unificata per coppia | nativa sim. | nativa sim. + area | nativa sim. + area, regione unificata | GT coef | rango maxabs -> unificata | NaN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT+GNM e108 | 0.423 [0.329, 0.516] | 0.314 [0.220, 0.407] | -0.109 [-0.182, -0.030] | 0.314 | 0.423 | 0.418 | 0.308 | 0.259 | 5 -> 1 | 0 |
| BFM+ICT congiunto 1019532 (conv. BFM) | 0.424 [0.326, 0.516] | 0.304 [0.204, 0.402] | -0.120 [-0.188, -0.053] | 0.304 | 0.412 | 0.406 | 0.300 | 0.268 | 4 -> 2 | 0 |
| BFM+ICT+GNM e036 | 0.426 [0.323, 0.520] | 0.286 [0.197, 0.373] | -0.139 [-0.216, -0.066] | 0.286 | 0.398 | 0.401 | 0.283 | 0.233 | 3 -> 3 | 0 |
| Chamfer faceBench 4096 pt | 0.462 [0.367, 0.548] | 0.282 [0.178, 0.378] | -0.180 [-0.250, -0.108] | 0.282 | 0.363 | 0.345 | 0.280 | 0.271 | 1 -> 4 | 0 |
| ICP + NICP P2Tri | 0.240 [0.152, 0.331] | 0.266 [0.180, 0.355] | +0.026 [-0.047, +0.099] | 0.266 | 0.288 | 0.266 | 0.266 | 0.349 | 8 -> 5 | 0 |
| Chamfer eval | 0.426 [0.325, 0.517] | 0.257 [0.155, 0.356] | -0.169 [-0.245, -0.100] | 0.257 | 0.346 | 0.327 | 0.255 | 0.251 | 2 -> 6 | 0 |
| ICP rigido + Chamfer | 0.300 [0.202, 0.391] | 0.251 [0.159, 0.337] | -0.049 [-0.126, +0.035] | 0.252 | 0.349 | 0.339 | 0.249 | 0.294 | 7 -> 7 | 0 |
| BFM+ICT congiunto 1019532 (conv. ICT) | 0.398 [0.296, 0.494] | 0.250 [0.150, 0.346] | -0.148 [-0.225, -0.075] | 0.250 | 0.365 | 0.346 | 0.239 | 0.190 | 6 -> 8 | 0 |
| ArcFace ombreggiato (3 viste) | 0.150 [0.066, 0.237] | 0.229 [0.146, 0.305] | +0.079 [+0.003, +0.161] | 0.229 | 0.108 | 0.121 | 0.228 | 0.239 | 10 -> 9 | 0 |
| ArcFace normal map (3 viste) | 0.165 [0.093, 0.235] | 0.222 [0.154, 0.289] | +0.057 [+0.002, +0.112] | 0.222 | 0.143 | 0.155 | 0.223 | 0.203 | 9 -> 10 | 0 |

Delta appaiati contro Chamfer eval (P(<=0) fra parentesi):

| metodo - Chamfer eval | GT maxabs | GT unificata |
| --- | --- | --- |
| BFM+ICT+GNM e108 | -0.003 [-0.106, +0.103] (0.513) | +0.057 [-0.048, +0.171] (0.147) |
| BFM+ICT congiunto 1019532 (conv. BFM) | -0.002 [-0.112, +0.112] (0.487) | +0.047 [-0.065, +0.166] (0.198) |
| BFM+ICT+GNM e036 | -0.001 [-0.104, +0.113] (0.496) | +0.029 [-0.068, +0.133] (0.283) |
| Chamfer faceBench 4096 pt | +0.035 [+0.020, +0.052] (0.000) | +0.025 [+0.012, +0.038] (0.000) |
| ICP + NICP P2Tri | -0.186 [-0.309, -0.052] (0.999) | +0.009 [-0.102, +0.125] (0.416) |
| ICP rigido + Chamfer | -0.126 [-0.237, -0.011] (0.988) | -0.005 [-0.109, +0.112] (0.530) |
| BFM+ICT congiunto 1019532 (conv. ICT) | -0.028 [-0.116, +0.062] (0.738) | -0.007 [-0.105, +0.094] (0.538) |
| ArcFace ombreggiato (3 viste) | -0.276 [-0.403, -0.138] (1.000) | -0.028 [-0.156, +0.102] (0.634) |
| ArcFace normal map (3 viste) | -0.262 [-0.375, -0.143] (1.000) | -0.035 [-0.153, +0.079] (0.703) |

Delta appaiati contro BFM+ICT+GNM e108 (P(<=0) fra parentesi):

| metodo - BFM+ICT+GNM e108 | GT maxabs | GT unificata |
| --- | --- | --- |
| BFM+ICT congiunto 1019532 (conv. BFM) | +0.000 [-0.070, +0.075] (0.472) | -0.010 [-0.083, +0.057] (0.597) |
| BFM+ICT+GNM e036 | +0.002 [-0.048, +0.050] (0.458) | -0.028 [-0.080, +0.026] (0.849) |
| Chamfer faceBench 4096 pt | +0.038 [-0.066, +0.137] (0.257) | -0.032 [-0.145, +0.073] (0.734) |
| ICP + NICP P2Tri | -0.183 [-0.309, -0.058] (0.998) | -0.048 [-0.153, +0.061] (0.795) |
| Chamfer eval | +0.003 [-0.103, +0.106] (0.487) | -0.057 [-0.171, +0.048] (0.853) |
| ICP rigido + Chamfer | -0.123 [-0.259, +0.012] (0.965) | -0.063 [-0.186, +0.064] (0.832) |
| BFM+ICT congiunto 1019532 (conv. ICT) | -0.025 [-0.116, +0.067] (0.725) | -0.064 [-0.164, +0.025] (0.915) |
| ArcFace ombreggiato (3 viste) | -0.274 [-0.389, -0.152] (1.000) | -0.085 [-0.206, +0.033] (0.924) |
| ArcFace normal map (3 viste) | -0.259 [-0.371, -0.153] (1.000) | -0.092 [-0.196, +0.007] (0.961) |

### Controlli (GT maxabs e coef: i numeri pubblicati devono tornare)

| dominio | riga | pubblicato | ricalcolato | |diff| |
| --- | --- | --- | --- | --- |
| hifi3d | scale_e072 subject_pair_mean (punto) | 0.7924 | 0.7924 | 0.0e+00 |
| hifi3d | scale_e036 subject_pair_mean (punto) | 0.7670 | 0.7670 | 0.0e+00 |
| hifi3d | scale_e108 subject_pair_mean (punto) | 0.7954 | 0.7954 | 0.0e+00 |
| hifi3d | chamfer_eval subject_pair_mean (punto) | 0.7431 | 0.7431 | 0.0e+00 |
| hifi3d | arcface_shaded_3v subject_pair_mean (punto) | 0.2965 | 0.2965 | 0.0e+00 |
| hifi3d | arcface_normals_3v subject_pair_mean (punto) | 0.2697 | 0.2697 | 0.0e+00 |
| hifi3d | chamfer_eval nocrop_cross (punto) | 0.3721 | 0.3721 | 0.0e+00 |
| hifi3d | scale_e036 nocrop_cross (punto) | 0.6773 | 0.6773 | 0.0e+00 |
| hifi3d | scale_e072 nocrop_cross (punto) | 0.6633 | 0.6633 | 0.0e+00 |
| hifi3d | arcface_normals_3v nocrop_cross (punto) | 0.2415 | 0.2415 | 2.8e-17 |
| hifi3d | scale_e108 nocrop_cross (punto) | 0.6299 | 0.6299 | 0.0e+00 |
| hifi3d | arcface_shaded_3v nocrop_cross (punto) | 0.2447 | 0.2447 | 0.0e+00 |
| hifi3d | e108 - Chamfer eval subject_pair_mean (point) | 0.0522 | 0.0522 | 0.0e+00 |
| hifi3d | e108 - Chamfer eval subject_pair_mean (ci_low) | -0.0039 | -0.0039 | 8.7e-17 |
| hifi3d | e108 - Chamfer eval subject_pair_mean (ci_high) | 0.1061 | 0.1061 | 6.9e-17 |
| hifi3d | e108 - Chamfer eval nocrop_cross (point) | 0.2578 | 0.2578 | 0.0e+00 |
| hifi3d | e108 - Chamfer eval nocrop_cross (ci_low) | 0.2076 | 0.2076 | 2.8e-17 |
| hifi3d | e108 - Chamfer eval nocrop_cross (ci_high) | 0.3010 | 0.3010 | 5.6e-17 |
| hifi3d | fb_chamfer nocrop_cross GT maxabs (punto) | 0.3253 | 0.3253 | 2.0e-07 |
| hifi3d | fb_rigid_icp_chamfer nocrop_cross GT maxabs (punto) | 0.3553 | 0.3553 | 2.9e-07 |
| hifi3d | fb_nicp_p2tri nocrop_cross GT maxabs (punto) | 0.3893 | 0.3893 | 3.3e-07 |
| hifi3d | fb_chamfer nocrop_cross GT coef (punto) | 0.0688 | 0.0688 | 2.8e-17 |
| hifi3d | fb_rigid_icp_chamfer nocrop_cross GT coef (punto) | 0.1553 | 0.1553 | 2.8e-17 |
| hifi3d | fb_nicp_p2tri nocrop_cross GT coef (punto) | 0.1391 | 0.1391 | 8.3e-17 |
| hifi3d | fb_chamfer original_to_original GT maxabs (punto) | 0.8758 | 0.8758 | 0.0e+00 |
| hifi3d | fb_rigid_icp_chamfer original_to_original GT maxabs (punto) | 0.4005 | 0.4005 | 5.6e-17 |
| hifi3d | fb_nicp_p2tri original_to_original GT maxabs (punto) | 0.4311 | 0.4311 | 0.0e+00 |
| hifi3d | fb_chamfer original_to_original GT coef (punto) | 0.1615 | 0.1615 | 2.8e-17 |
| hifi3d | fb_rigid_icp_chamfer original_to_original GT coef (punto) | 0.1654 | 0.1654 | 5.6e-17 |
| hifi3d | fb_nicp_p2tri original_to_original GT coef (punto) | 0.1371 | 0.1371 | 8.3e-17 |
| hifi3d | joint nocrop_cross GT maxabs (punto) | 0.4277 | 0.4277 | 0.0e+00 |
| hifi3d | scale_e108 nocrop_cross GT maxabs (punto) | 0.6299 | 0.6299 | 0.0e+00 |
| hifi3d | joint nocrop_cross GT coef (punto) | 0.1054 | 0.1054 | 8.3e-17 |
| hifi3d | scale_e108 nocrop_cross GT coef (punto) | 0.1313 | 0.1313 | 5.6e-17 |
| hifi3d | joint subject_pair_mean GT maxabs (punto) | 0.7201 | 0.7201 | 0.0e+00 |
| hifi3d | scale_e108 subject_pair_mean GT maxabs (punto) | 0.7954 | 0.7954 | 0.0e+00 |
| hifi3d | joint subject_pair_mean GT coef (punto) | 0.1612 | 0.1612 | 5.6e-17 |
| hifi3d | scale_e108 subject_pair_mean GT coef (punto) | 0.1599 | 0.1599 | 0.0e+00 |
| faceverse | chamfer_eval mesh_pair_nocrop (punto) | 0.3168 | 0.3168 | 0.0e+00 |
| faceverse | fb_chamfer mesh_pair_nocrop (punto) | 0.3189 | 0.3189 | 5.6e-17 |
| faceverse | fb_rigid_icp_chamfer mesh_pair_nocrop (punto) | 0.2593 | 0.2593 | 5.6e-17 |
| faceverse | fb_nicp_p2tri mesh_pair_nocrop (punto) | 0.1943 | 0.1943 | 0.0e+00 |
| faceverse | chamfer_eval subject_pair_mean_nocrop (punto) | 0.4262 | 0.4262 | 0.0e+00 |
| faceverse | fb_chamfer subject_pair_mean_nocrop (punto) | 0.4616 | 0.4616 | 0.0e+00 |
| faceverse | fb_rigid_icp_chamfer subject_pair_mean_nocrop (punto) | 0.3000 | 0.3000 | 0.0e+00 |
| faceverse | fb_nicp_p2tri subject_pair_mean_nocrop (punto) | 0.2402 | 0.2402 | 2.8e-17 |

