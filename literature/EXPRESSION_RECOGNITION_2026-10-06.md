# Riconoscimento 3D invariante all'espressione: stato dell'arte e novelty (ricerca del 2026-10-06)

Metodo: WebSearch, WebFetch, alphaXiv (testo completo di FR3DNet, GFT-GCN, Data-Free Point Cloud Net, Face-GCN, 2601.06484, survey 2108.11082; PDF di Led3D e AAAI-21 estratti in locale). Nessun download di dataset. Molti editori (ACM, ScienceDirect, MDPI, ResearchGate) hanno risposto 403, quindi diversi paper sono verificati solo da abstract/snippet: marcati "[abstract]".
Legenda: **[V]** letto sulla fonte primaria; **[A]** solo abstract o snippet di ricerca; **[S]** riportato da un survey o da un altro paper, non dalla fonte primaria; **[?]** incerto.
Limite: la ricerca non e' esaustiva. "Non trovato" significa non trovato con queste query, non "non esiste".

## 1. Tabella dei lavori principali

| Lavoro | Anno, venue | Link | Cosa fa / cosa lo distingue |
|---|---|---|---|
| Bronstein, Bronstein, Kimmel, "Three-Dimensional Face Recognition" | 2005, IJCV 64:5-30 | https://link.springer.com/article/10.1007/s11263-005-1085-y | Espressioni modellate come isometrie; forme canoniche da distanze geodesiche (MDS). Distingue gemelli [A]. Dimensione del DB di test non verificata [?]. Intrinseco, ma niente deep, niente test cross-discretizzazione verificato. |
| Kim, Hernandez, Choi, Medioni, "Deep 3D Face Identification" | 2017, IJCB | https://arxiv.org/abs/1703.10714 | Fine-tuning di VGG-Face su depth map 224x224. Augmentation: BFM (shape) + FaceWarehouse (espressioni) adattato a 577 identita' di FRGCv2, 25 espressioni ciascuna [S: FR3DNet + survey 2108.11082]. Test Bosphorus 99.2, BU-3DFE 95.0, 3D-TEC 94.8 rank-1 [S]. E' il primo uso di un 3DMM per sintetizzare espressioni, ma le identita' sono reali. |
| Gilani, Mian, "Learning from Millions of 3D Scans" (FR3DNet) | 2018, CVPR, pp. 1896-1905 | https://arxiv.org/abs/1711.05942 | 3.1M scansioni, 100K identita' **sintetiche non da 3DMM**: nuove facce da coppie di facce reali (media di due facce con massima differenza non rigida), piu' 15 camere virtuali per posa/occlusione [V + S]. Critica esplicitamente i generatori 3DMM ("confinati allo spazio lineare del modello") [V]. Test LS3DFace: 1853 identita' di gallery, ~31K probe, fusione di FRGCv2, BU-3DFE, Bosphorus, GavabDB, Texas, BU-4DFE, CASIA, UMB, 3D-TEC [V]. Input: immagini 160x160 (depth, azimut, elevazione) [S]. FRGCv2 rank-1 97.06 senza fine-tuning, 99.88 con [V, tabella in 1911.04731]. Cross-dataset di fatto (training sintetico, test su reali), ma identita' sintetiche derivate dagli stessi scan reali dei test [V: "Our training data does not include the public datasets" ma i dati sorgente sono anche privati; Led3D dice "ensemble of many public databases and a private one" [V]. **Contraddizione da chiarire**]. |
| Zhang, Da, Yu, "Data-Free Point Cloud Network" / "Learning directly from synthetic point clouds for in-the-wild 3D FR" | 2019 arXiv; 2022 Pattern Recognition 123 | https://arxiv.org/abs/1911.04731 | **Il precedente piu' vicino al tuo (c).** Train solo su facce sintetiche da GPMM (Gaussian Process Morphable Model da 200 scan neutri), 10K identita' x 50 espressioni, PointNet++ con campionamento curvature-aware; test su FRGCv2 e Bosphorus reali [V]. Rank-1 FRGC 92.74 (zero-shot) / 98.73 (fine-tuning su reali); Bosphorus totale 93.38 / 97.50 [V]. Usa campionamento di punti diverso tra train e test per ridurre il gap di dominio [V]: l'unica variazione di discretizzazione trovata, ma e' una scelta di progetto, non un protocollo di valutazione. Un solo 3DMM, nessun leave-one-3DMM-out. |
| Yu, Da, Zhang, Sur3dNet-Face | 2021 arXiv, Multimed. Tools Appl. | https://arxiv.org/abs/2103.16927 | PointNet con guida GPMM; training su soli 943 scan FRGC Spring2003 (reali). 98.85 FRGC, 99.33 Bosphorus [A]. |
| Mu, Huang, Hu, Sun, Wang, **Led3D** | 2019, CVPR, pp. 5773-5782 | https://openaccess.thecvf.com/content_CVPR_2019/papers/Mu_Led3D_A_Lightweight_and_Efficient_Deep_Approach_to_Recognizing_Low-Quality_CVPR_2019_paper.pdf | CNN leggera (MSFF + SAV) su depth+normal map di bassa qualita'. Train/test su Lock3DFace (Kinect V2, 509 soggetti, 5671 video; protocollo NU/FE/PS/OC/TM) [V]. Dati sintetici: pose virtuali, jitter gaussiano, scaling, e identita' virtuali con il metodo di FR3DNet (nessun 3DMM) [V]. **Protocollo cross-quality** (sez. 5.2): train su FRGCv2 con identita' virtuali (122,150 facce, 1000 id); test su Bosphorus con gallery alta qualita' (prima neutra, 105 soggetti) e probe degradate: rumore N(0,16) sulla sola z + filtro massimo 3x3 [V]. Densita' media: FRGC 53K punti, Bosphorus 27K, Lock3DFace 9K [V]. Lock3DFace rank-1 totale 84.22 (depth+normal, protocollo di [2]) [V]. |
| Zhang, Yu, Xu, Li, "Learning Flexibly Distributional Representation for Low-quality 3D FR" | 2021, AAAI | https://cdn.aaai.org/ojs/16460/16460-13-19954-1-2-20210518.pdf | Rappresentazione distribuzionale (flussi normalizzanti) per volti rumorosi. Rumore gaussiano sintetico sulla z (varianza 4, 8, 16, 32, 64) su BU-3DFE/BU-4DFE; **cross-quality**: gallery alta qualita' (neutra originale), probe sintetici degradati; reali: Lock3DFace, IIIT-D (106 soggetti, Kinect V1, 4603 depth map) [V]. Confronto con Led3D. |
| Hu et al., "Boosting Depth-Based Face Recognition from a Quality Perspective" | 2019, Sensors 19(19):4124 | https://www.mdpi.com/1424-8220/19/19/4124 | Nuovo DB con 902 soggetti: forma 3D alta qualita' (scanner), depth bassa qualita' e colore; tre strategie per usare l'alta qualita' nel training del modello su depth bassa qualita' [A]. E' il DB "Extended-MultiDim" secondo il risultato di ricerca [A, nome non confermato dal testo]. Disponibilita' e licenza: non verificate [?]. |
| Yang, Li, Shen, **PointSurFace** | 2024, Pattern Recognition 156:110858 | https://www.sciencedirect.com/science/article/abs/pii/S0031320324006095 | Modulo SurF (coordinate, normali, angoli per patch) su point cloud; Lock3DFace rank-1 90.03, verifica 80.01 [A]. |
| Yu et al., "Adaptive representation learning and sample weighting for low-quality 3D FR" | 2025, Pattern Recognition 159:111161 | https://www.sciencedirect.com/science/article/abs/pii/S0031320324009129 | Low-quality 3D FR; dettagli non letti (403) [?]. |
| Yu, Zhang, Li, Liu, "Enhancing 3D FR via 2D-Aided Generative Augmentation" | 2025, Sensors 25(16):5049 | https://doi.org/10.3390/s25165049 | Ricostruisce 3D da immagini 2D come dati sintetici; nessun dato 3D reale in training. Rank-1 99.2 BU-3DFE, 98.4 FRGC, 99.3 Bosphorus, 96.5 BU-4DFE [A]. |
| Felouat, Wang, Echizen, **GFT-GCN** | 2025, arXiv 2511.19958 | https://arxiv.org/abs/2511.19958 | Graph Fourier Transform sul Laplaciano di grafo della mesh (primi k=10-25 coefficienti su 10 descrittori per vertice) + GCN, loss contrastiva; protezione del template [V]. BU-3DFE (100 soggetti x 25) e FaceScape (938 soggetti, ~20 scan [V come dichiarato]); split casuale 70/15/15 per soggetto; EER 0.0 prima della protezione [V]. **E' l'unico lavoro recente "spettrale su mesh" per identita', ma il Laplaciano e' L = D - A normalizzato (dipende dalla connettivita'), non cotangente; nessun test su connettivita' diversa; nessun test fuori dominio** [V]. |
| Papadopoulos, Kacem, Shabayek, Aouada, Face-GCN | 2022 ICVR; arXiv 2104.09145 | https://arxiv.org/abs/2104.09145 | ST-GCN su grafi di landmark, senza registrazione densa; BU-4DFE, protocollo cross-emozione, 88.45% [V]. |
| Kacem, Cherenkova, Aouada, "Disentangled Face Identity Representations..." | 2021, arXiv 2104.10273 | https://arxiv.org/abs/2104.10273 | Auto-encoder a grafo su mesh + GAN per neutralizzare l'espressione; tre dataset (nomi non letti) [A]. Richiede mesh con topologia comune [? dedotto dai GCA, non verificato]. |
| Guo et al., survey "3D Face Recognition: Two Decades of Progress and Prospects" | 2023, ACM CSUR 56(3) | https://doi.org/10.1145/3615863 | Survey [A]. |
| Jing, Lu, Gao, "3D Face Recognition: A Survey" | 2021 arXiv; 2023 Comput. Vis. Media | https://arxiv.org/abs/2108.11082 | Survey; contiene la tabella DCNN (Kim, FR3DNet, Led3D, point cloud) [V]. |
| Lee, Kang, Lee, "3D Facial Shape Similarity with Deep Perceptual Representations" | 2025, ACM TOMM 21(6) | https://dl.acm.org/doi/10.1145/3734874 | **Il lavoro piu' vicino al tuo ambito "metrica di somiglianza fra mesh facciali"**: similarita' da rappresentazioni percettive multi-vista di una mesh facciale, con training "view specificity" e "regional consistency"; dichiara di superare lo stato dell'arte [A]. Non ho potuto leggere dati, GT di somiglianza, ne' se testano topologie diverse [?]. Da leggere per intero. |
| Chai et al., **REALY** | 2022, ECCV | https://arxiv.org/abs/2203.09729 | Benchmark di ricostruzione: l'allineamento rigido con punti di riferimento diversi altera i risultati; 100 scan allineati, keypoint, maschere regionali, mesh a topologia consistente (HIFI3D++) [V]. |
| Sariyanidi et al., "3D Face Reconstruction Error Decomposed" | 2025, FG; arXiv 2505.18025 | https://arxiv.org/abs/2505.18025 | ICP + corrispondenza Chamfer: correlazione con l'errore vero fino a 0.41; >=10% di punti con corrispondenze duplicate errate; registrazione non rigida con landmark (ELR) >0.90 [V, via PMC12380054]. |
| Wang et al., "Learning Domain Agnostic Latent Embeddings of 3D Faces..." | 2026, arXiv 2601.06484 | https://arxiv.org/abs/2601.06484 | **DiffusionNet su HKS/WKS di volti**, addestrato su volti sintetici ICT-FaceKit, zero-shot su mesh di gatti. Include solo un'analisi t-SNE qualitativa dello spazio d'identita' su 50 identita' x 100 espressioni (raggruppa per identita' attraverso le espressioni) [V]. Nessuna metrica di riconoscimento. |
| Qin et al., Neural Face Rigging | 2023, SIGGRAPH | https://dl.acm.org/doi/abs/10.1145/3588432.3591556 | Retargeting su mesh facciali di topologia arbitraria, su feature intrinseche + Neural Jacobian Fields [A]; il legame con DiffusionNet e' citato in 2601.06484 [V]. Non e' riconoscimento. |

## 2. Cosa e' gia' fatto / cosa sembra nuovo

### (a) Riconoscimento invariante all'espressione con encoder intrinseci/spettrali

- **Classico, gia' fatto**: Bronstein et al. 2005 (isometrie, forme canoniche) [A]. Altri classici spettrali/intrinseci su volto (Laplace-Beltrami, ShapeDNA) non verificati con una fonte primaria.
- **HKS/WKS su volti**: trovato solo "3D skull and face similarity measurements based on a harmonic wave kernel signature" (Visual Computer, 2020, https://link.springer.com/article/10.1007/s00371-020-01946-x) [snippet; contenuto non letto]. Riguarda similarita' cranio-volto, non riconoscimento d'identita' con protocollo di espressione.
- **Deep spettrale su mesh facciali**: GFT-GCN (2025) [V]: base Laplaciana di grafo, non discretization-agnostic, solo dentro dataset.
- **DiffusionNet su volti per identita'**: nessun lavoro di riconoscimento trovato. Unico vicino: 2601.06484 (DiffusionNet + HKS/WKS + ICT sintetico, ma per trasferimento d'espressione e solo qualitativo sull'identita') [V].
- **Functional maps per riconoscimento facciale**: non trovato.
- Conclusione per (a): un encoder DiffusionNet con metriche di riconoscimento (rank-1, mAP, AUC) sotto espressione sembra **non pubblicato** (non trovato), ma 2601.06484 e GFT-GCN sono da citare come lavori adiacenti.

### (b) Valutazione cross-topologia / cross-discretizzazione

Priorita' aggiornata: "stessa persona, mesh con connettivita' diversa" come protocollo.

- **Non trovato**: nessun lavoro di riconoscimento facciale 3D che valuti la stessa persona su mesh con connettivita' diversa (remesh, decimazione, mesh da pipeline diverse) come protocollo. Le pipeline CNN (Kim, FR3DNet, Led3D) proiettano su griglia fissa (depth/azimut/elevazione/normali), quindi la connettivita' e' irrilevante per costruzione e non viene mai testata [V per FR3DNet e Led3D; S per Kim].
- **Cio' che esiste e' cross-qualita'/sensore**, che include variazione di densita' e rumore ma non di connettivita':

| Lavoro | Variazione testata esattamente |
|---|---|
| Led3D (CVPR 2019) | Sensore reale (Kinect V2 su Lock3DFace, ~9K punti vs ~27-53K degli scanner) [V]; cross-quality sintetico: gallery alta qualita' vs probe con rumore N(0,16) sulla z + filtro massimo 3x3 su Bosphorus [V]. Niente remesh, niente decimazione, niente connettivita'. |
| Zhang et al., AAAI 2021 | Rumore gaussiano sulla z (varianza 4-64) su BU-3DFE/BU-4DFE con gallery pulita; Lock3DFace e IIIT-D reali [V]. Il rumore e' sulla sola z di un depth map. |
| Hu et al., Sensors 2019 | Forma alta qualita' da scanner vs depth bassa qualita' (902 soggetti) [A]. Cross-sensore reale. |
| CAS-AIR-3D (IJCV 2025) | DB di 3093 soggetti, RealSense SR305, pose/espressioni/occlusioni/distanza [A]; non letto il protocollo cross-sensore. |
| Zhang et al. 2019/2022 | Campionamento di punti diverso tra train (CPS) e test; 28,588 punti per scan [V]. Scelta di progetto, non protocollo con metrica. |
| GFT-GCN (2025) | Nessuna. Dentro-dataset, mesh di un solo scanner per dataset [V]. |
| Face-GCN | Nessuna; landmark, senza registrazione [V]. |
| Lee et al., TOMM 2025 | Dichiara di gestire mesh irregolari via proiezioni multi-vista [A]; da verificare se testa remesh/decimazione [?]. |

- **Non trovato**: confronto remesh/decimazione/rumore sulle posizioni dei vertici su mesh (non depth map) per identita'; mesh da pipeline diverse (es. HIFI3D++ vs FLAME vs scansioni) come gallery/probe incrociati.
- Conclusione per (b): il "cross-risoluzione/cross-qualita'" esiste con depth map e rumore sulla z; il "cross-connettivita' su mesh con encoder intrinseco" sembra **nuovo**. Prudenza: non e' un'affermazione verificabile in modo assoluto.

### (c) Training sintetico su un 3DMM, test su altro 3DMM o su scansioni reali, leave-one-3DMM-out

- **Gia' fatto (train sintetico, test su reali)**: Zhang et al. 2019/2022 (GPMM, zero-shot FRGC 92.74, Bosphorus 93.38) [V]; Kim 2017 (BFM+FW, fine-tuning) [S]; Yu et al. 2025 (da immagini 2D) [A]; FR3DNet (non 3DMM) [V].
- **Non trovato**: test su un *altro 3DMM* e protocollo leave-one-3DMM-out per il riconoscimento. Nei lavori trovati il dominio sintetico e' un solo modello, valutato solo su scansioni reali.
- **Rischio di confronto**: Zhang et al. riportano che il fine-tuning su pochi scan reali alza FRGC da 92.74 a 98.73 [V]: un baseline zero-shot sintetico->reale e' gia' competitivo e va citato.
- Conclusione per (c): il protocollo leave-one-3DMM-out sembra **nuovo**; train-sintetico/test-reale **non** lo e'.

### (d) Critica metodologica dei protocolli di valutazione

- **Esiste** (per ricostruzione, non per similarita' d'identita'): REALY (ECCV 2022) sull'allineamento rigido/punti di riferimento [V]; Sariyanidi et al. (FG 2025) su ICP, Chamfer, cropping, correlazioni 0.41 vs >0.90 [V]. Quest'ultimo **non** discute circolarita' d'identita' [V, esplicito].
- **Non trovato**: critica esplicita della circolarita' di una GT geometrica valutata con Chamfer, ne' discussione del ruolo di frame/supporto in metriche di *somiglianza d'identita'* tra mesh. Da controllare Lee et al. 2025 (TOMM) e "3D Facial Similarity Measurement..." (TOMM 2020, https://dl.acm.org/doi/10.1145/3397765), non letti [?].
- Conclusione per (d): la critica su allineamento/Chamfer nella ricostruzione e' **precedente da citare**; la sua applicazione alla similarita' d'identita' e alla circolarita' GT-metrica sembra non fatta (non trovato).

### Lacune delle ricerche precedenti che sostengono la novelty (riassunto)
1. Encoder discretization-agnostic (DiffusionNet) per identita' con metriche di retrieval/verifica: non trovato.
2. Protocollo "stessa persona, connettivita' diversa" (remesh, decimazione, rumore sui vertici): non trovato.
3. Train su 3DMM A, test su 3DMM B e leave-one-3DMM-out: non trovato.
4. Critica GT geometrica/Chamfer per metriche d'identita': non trovato.

## 3. Dataset reali con piu' scansioni per persona

| Dataset | Contenuto | Licenza / accesso | Stato verifica |
|---|---|---|---|
| FRGC v2.0 | 4007 scan 3D, 466 soggetti, Minolta Vivid 910, espressioni | Copyright Notre Dame; richiesta scritta caso per caso dall'istituzione al PI; EULA (https://cvrl.nd.edu/projects/data/) | [A, EULA esistente] |
| Bosphorus | 4666 scan, 105 soggetti, espressioni FACS, pose, occlusioni; strutturated light | Richiesta e accordo di licenza ricerca (Bogazici). Il sito bosphorus.ee.boun.edu.tr non si risolve dal cluster; termini esatti [?] | [A] |
| BU-3DFE | 2500 modelli, 100 soggetti, 25 per soggetto (6 espressioni x 4 intensita' + neutra) | SUNY Binghamton, tramite l'ufficio trasferimento tecnologico; **gli studenti non possono essere destinatari, la richiesta va fatta dal supervisore** (http://www.cs.binghamton.edu/~lijun/Research/3DFE/) | [A] |
| BU-4DFE | 101 soggetti, sequenze 3D | Stesso gruppo di BU-3DFE; termini non verificati | [?] |
| Texas 3DFRD | 1149 coppie colore/range, 118 soggetti, stereo | Scaricabile gratuitamente per ricerca/educazione, uso non commerciale, citare paper e sito (https://live.ece.utexas.edu/research/texas3dfr/) | [A] |
| CASIA-3D FaceV1 | 4624 scan, 123 soggetti, Minolta Vivid 910 | Richiesta via biometrics.idealtest.org; vietati distribuzione/copia; obbligo di citazione | [A] |
| FaceScape | 16,940 modelli, 847 soggetti, 20 espressioni (come dichiarato da fonti di ricerca; GFT-GCN dice 938 soggetti) | Solo ricerca non commerciale; firmare il License Agreement e inviarlo a nju3dv@nju.edu.cn, poi chiave di download (https://facescape.nju.edu.cn/). Discrepanza sul numero di soggetti (847 vs 938): da chiarire [?] | [A] |
| Multiface (Meta) | **13 identita'**, ~12,200-23,000 frame per soggetto a 30 fps, mesh tracciate, texture, immagini multivista | **CC-BY-NC 4.0**; download da GitHub con script, mini-dataset 16.2 GB (https://github.com/facebookresearch/multiface) [V]. Con 13 identita' i punteggi di retrieval avranno intervalli di confidenza larghi. | [V] |
| NoW | 100 soggetti; val 20 (con GT), test 80; benchmark di ricostruzione da immagine | Registrazione + licenza sul sito MPI (https://ringnet.is.tue.mpg.de/); la GT del test non e' pubblica. **Non e' un dataset di riconoscimento multi-scansione** (scan 3D usati come GT di forma) | [A] |
| Headspace | 1518 soggetti con cuffia in lattice; 1212 usati per i modelli; un solo scan per soggetto | Uso non commerciale con modulo di accordo utente (host York, ~38 GB) (https://www-users.york.ac.uk/~np7/research/Headspace/). **Un scan per soggetto: non adatto a retrieval con piu' scan per persona** [A]; espressione non verificata [?] | [A] |
| Lock3DFace | 509 soggetti, 5671 video Kinect V2, variazioni NU/FE/PS/OC/TM | Licenza non verificata (https://irip.buaa.edu.cn/lock3dface) [?] | [A] |
| IIIT-D Kinect | 106 soggetti, 4603 depth map, Kinect V1 | Non verificato | [?] |
| CAS-AIR-3D, Extended-MultiDim | 3093 soggetti (RealSense SR305); 902 soggetti (scanner + RealSense) | Disponibilita' e licenza non verificate | [?] |

Nota pratica: per un test con piu' scan per persona e espressione variabile, i candidati reali piu' ricchi sono FaceScape, BU-3DFE, Bosphorus, FRGC; Multiface ha pochissime identita'; Headspace e NoW non hanno multi-scan per persona.

## 4. Verificato contro incerto: sintesi

**Verificato su fonte primaria (testo letto):** FR3DNet (contenuti e dimensioni, tecnica di sintesi non-3DMM, critica ai 3DMM), Led3D (protocolli, cross-quality, numeri), AAAI-21 Zhang et al. (protocolli e rumore), Data-Free Point Cloud Net (GPMM, numeri), GFT-GCN, Face-GCN, 2601.06484, Multiface (licenza), 2505.18025 (numeri via PMC), REALY (abstract), survey 2108.11082 (descrizione Kim e FR3DNet).

**Solo da abstract/snippet:** Bronstein 2005, Kim 2017 (numeri solo da survey), Sur3dNet-Face, PointSurFace, Yu 2025 (due), Hu 2019, CAS-AIR-3D, Lee et al. TOMM 2025, tutti i termini di licenza (tranne Multiface), HWKS cranio-volto.

**Da chiarire / contraddizioni:** (i) FR3DNet dichiara di non usare dataset pubblici nel training, ma Led3D lo descrive come sintetizzato da "un ensemble di molti DB pubblici e uno privato"; (ii) FaceScape 847 vs 938 soggetti; (iii) cosa testa davvero Lee et al. 2025 sulla discretizzazione.

**Non trovato:** DiffusionNet per riconoscimento d'identita' con metriche; functional maps per riconoscimento facciale; protocollo stessa persona/connettivita' diversa; leave-one-3DMM-out; critica della circolarita' GT geometrica-vs-Chamfer per similarita' d'identita'.

## 5. Prossimi passi consigliati (letture mancanti)
1. Lee et al., TOMM 2025 e TOMM 2020 (3D facial similarity): scaricare il testo per i punti (b) e (d).
2. Sezioni sperimentali di CAS-AIR-3D (IJCV 2025) e Extended-MultiDim: variazioni di sensore reali.
3. Cercare con altri motori (Google Scholar) "mesh-invariant face identity", "topology-agnostic face recognition", "heat kernel signature face recognition" per escludere falsi negativi.
