# Giudizio (scritto il 2026-10-07, dopo i numeri)

Job: render BFM held-out 1060587 (CPU, ombreggiato + normal map + ArcFace), embedding CLIP/DINOv2 1060588
(FaceVerse, HIFI3D) e 1060591 (BFM), 1 L40S ciascuno; summary 1060600 (FaceVerse; fallito poi su HIFI3D per un
bug di formattazione, corretto) e 1060602/1060603 (HIFI3D, BFM). Frame BFM 1060586. Controlli superati: le righe
gia' pubblicate in `aau/runs/arcface_render_zs` si riproducono esattamente (max |diff| 1.1e-16, FaceVerse 14 righe,
HIFI3D 12); le righe WS1 su BFM riproducono `table2_extended_heldout.csv` (Chamfer 0.469 / 0.644, ArcFace 0.218 / 0.293,
CLIP B/32 0.087 / 0.358, DINOv2 S/14 0.058 / 0.378); il congiunto su HIFI3D riproduce `summary_frame-xmymz_flip.md`
(0.245 GT maxabs, 0.043 GT coef). ArcFace sui nuovi render ombreggiati BFM (0.220) coincide con ArcFace WS1 (0.218).

**Lettura fissata nel protocollo, riconoscimento (primario, rank-1, senza crop).**
- FaceVerse con espressioni: CLIP 0.304 [0.274, 0.336], DINOv2 0.251 [0.221, 0.281]. **Sotto** ArcFace normal map
  (-0.563 e -0.616), sotto NICP P2Tri, sotto Chamfer faceBench e **sotto il congiunto** (-0.376 [-0.421, -0.329],
  -0.429). Stessa cosa sull'AUC (0.651 e 0.593).
- HIFI3D: CLIP 0.597 [0.582, 0.614], DINOv2 0.510 [0.485, 0.534]. **Sotto** ArcFace normal map (-0.401, -0.488) e
  NICP. **Sopra il congiunto sul rank-1** (CLIP +0.200 [+0.167, +0.235], DINOv2 +0.113 [+0.079, +0.147]) ma
  **sotto sull'AUC** (-0.047, -0.087). Contro Chamfer: CLIP sopra sul rank-1 (+0.119), DINOv2 pari (+0.033
  [-0.004, +0.068]); entrambi sotto sull'AUC.

**Ranking (Spearman con la GT, primario `nocrop_cross_topology`).**
- HIFI3D (GT maxabs): CLIP 0.195, DINOv2 0.107. Sotto Chamfer faceBench (-0.130, -0.219). Contro il congiunto (0.245):
  CLIP **pari** (-0.049 [-0.109, +0.006]), DINOv2 sotto (-0.138). Contro ArcFace normal map: CLIP pari, DINOv2 sotto.
  In `original_to_original` CLIP sta sopra ArcFace (+0.199) ma sotto congiunto e Chamfer. Con la GT coef tutti i
  metodi stanno fra 0.03 e 0.19 e quasi tutti i delta contengono 0.
- BFM held-out (in dominio, GT del paper): CLIP 0.129, DINOv2 0.114, contro **0.789 del modello NeurIPS** (delta
  -0.659 e -0.675) e 0.469 di Chamfer (-0.340, -0.355); sotto ArcFace normal map (-0.094, -0.109); pari a LPIPS.
  In `original_to_original` (0.332 e 0.370) pari ad ArcFace, sotto LPIPS, Chamfer e NeurIPS.

**Messaggio per il rebuttal (regola del protocollo).** CLIP e DINOv2 stanno sotto ArcFace normal map nel primario di
(a) e (b): **ArcFace resta la baseline percettiva di riferimento**, e CLIP/DINOv2 entrano come "encoder visivi
generici, piu' deboli". Fra i due CLIP L/14 batte DINOv2 B/14 nel riconoscimento (+0.053 su FaceVerse, +0.087 su
HIFI3D) e nel ranking su HIFI3D (+0.089); su BFM sono pari. Rispetto alle versioni piccole di WS1 (B/32 e S/14, render
ombreggiati interi) i modelli grandi su normal map salgono nel cross-topology BFM (0.129 contro 0.087, 0.114 contro
0.058; solo stime puntuali, non appaiate) ma restano lontanissimi dal modello appreso.

**Ablazione (crop calibrato contro render intero).** Effetto piccolo: FaceVerse rank-1 +0.001 (CLIP) e +0.004
(DINOv2), entrambi con CI che contiene 0; HIFI3D: il render intero toglie 0.024 [-0.046, -0.004] a CLIP e
aggiunge 0.006 a DINOv2 (CI con 0); BFM ranking entro 0.03, CI con 0. La scelta del ritaglio non spiega il
livello basso.

**Diagnosi (post hoc, non nel protocollo; job 1060601, FaceVerse).** Negli embedding pesa piu' la topologia
dell'identita': coseno medio CLIP fra soggetti diversi con la stessa topologia 0.946, fra lo stesso soggetto con
topologie diverse 0.934 (DINOv2: 0.883 contro 0.786). Il rank-1 original -> noisy crolla a 0.06 (CLIP) e 0.02
(DINOv2), contro 0.42 / 0.28 su original -> remesh. Gli encoder generici vedono la tassellazione nella normal map
(la "texture" dei triangoli), che ArcFace, addestrato sull'identita', ignora. Il caricamento dei pesi e' stato
verificato a parte: CLIP zero-shot sul render da' p = 1.0 a "a 3D rendering of a human face".

**Besnier 2023 e Ma 2021:** non integrate, motivi in `other_baselines.md` (nessun codice ne' pesi pubblici).

**Cosa resta aperto.** (1) Nessun FaceVerse neutro, quindi il calo su FaceVerse resta non separabile fra
espressioni e dominio, come per ArcFace. (2) Le NaN di NICP su HIFI3D (776 coppie cross nel ranking, 1965 distanze
nel riconoscimento) rendono quelle righe un limite inferiore. (3) Molti confronti senza correzione per la
molteplicita': le conclusioni sopra si reggono su delta con P(boot <= 0) pari a 0.000 o 1.000.
