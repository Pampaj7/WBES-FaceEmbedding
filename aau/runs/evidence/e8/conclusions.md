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
