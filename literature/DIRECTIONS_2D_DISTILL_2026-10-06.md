# Distillare il riconoscimento facciale 2D in un encoder d'identità 3D agnostico alla discretizzazione

Ricerca bibliografica del 2026-10-06. Nessun download di PDF su disco: lettura via alphaXiv (`answer_pdf_queries`, `discover_papers`), WebSearch, WebFetch.

## Come leggere le etichette

- **[LETTO]**: ho letto il testo (o larga parte) del paper via alphaXiv. Cosa fa e i numeri citati sono verificati su quel testo.
- **[LETTO-WEB]**: pagina web riassunta da WebFetch (modello piccolo). Numeri da ricontrollare sul paper prima di citarli.
- **[SNIPPET]**: visto solo nei risultati di ricerca (titolo + riassunto). Non letto. Venue e dettagli non verificati.
- **venue non verificato**: l'anno viene dall'arXiv, la venue no.
- Le affermazioni mie (interpretazione, ipotesi) sono marcate **[MIA]**.

Limite della ricerca: circa 25 query tra alphaXiv e WebSearch (US-only, risultati a snippet). "Non ho trovato X" significa questo, non "X non esiste".

---

## Sintesi in cinque righe

1. Il trasferimento 2D -> 3D funziona già per l'identità, ma **sempre con topologia o rappresentazione fissa** (depth map allineata, output FLAME) e **quasi sempre in-dataset**.
2. Il pezzo più vicino alla tua ipotesi è **MICA** (ArcFace come encoder, supervisionato da scansioni): dà un risultato forte, ma l'input è una foto e l'output è nello spazio FLAME. Mostra anche che **fine-tuning parziale** di ArcFace è la scelta giusta.
3. Le feature 2D "foundation" sollevate sulla mesh (Diff3F, MeshFM, DenseMatcher) sono **semantiche a livello di categoria**: nessuna è stata testata su identità intra-classe, né su volti.
4. **Non ho trovato** nessun lavoro che distilli ArcFace/AdaFace in un encoder mesh/point cloud/campo continuo e lo valuti fuori dominio. Lo spazio sembra libero.
5. Un avvertimento sul tuo indizio (AUC 1.000 su crop reali Multiface): vedi la sezione "Attenzione sul tuo indizio".

---

## 1. Riconoscimento 3D tramite render 2D con reti 2D pre-addestrate

| Lavoro | Anno / venue | Link | Cosa fa | Indip. dalla topologia | Fuori dominio / sola geometria | Stato |
|---|---|---|---|---|---|---|
| **Deep 3D Face Identification** (Kim, Hernandez, Choi, Medioni) | 2017, arXiv (venue non verificato) | https://arxiv.org/abs/1703.10714 | Fine-tuning di VGG-Face su **depth map ortografiche** 224x224 da scansioni allineate (ICP + naso). Augmentation: espressioni sintetizzate con 3DMM multilineare (BFM + FaceWarehouse), pose, patch di occlusione. | Parziale. Entra una point cloud, ma serve allineamento rigido e proiezione ortografica. Nessun vincolo sulla connettività. | Solo geometria (depth, niente texture). Train FRGC v1/v2 + CASIA3D, test Bosphorus, BU-3DFE, 3D-TEC: **dataset diversi** tra train e test, ma le gallery dei test set sono aumentate e usate in training (nota d della tabella 1), quindi non è zero-shot puro. Rank-1: Bosphorus 99.24%, neutro vs non-neutro 99.2%; BU-3DFE 93-97%; 3D-TEC (gemelli) 79.9-94.8% secondo il caso. Con solo FRGC aumentato: 98.1% su Bosphorus. | LETTO |
| **Learning from Millions of 3D Scans (FR3DNet)** (Gilani, Mian) | 2018, CVPR | https://arxiv.org/abs/1711.05942 | Rete **addestrata da zero** su 3.1M scansioni sintetiche di 100K identità. Input a 3 canali: depth + azimut + elevazione delle normali, crop 160x160 al naso. Identità nuove generate interpolando coppie di scansioni reali in corrispondenza densa (non da 3DMM lineare). Test LS3DFace: 1853 identità, 31K probe. | No. Richiede corrispondenza densa in training, naso e crop in test. | Solo geometria. Test su 10 dataset fusi, quindi multi-sensore, espressioni, occlusioni. Rank-1 98.74% dopo fine-tuning dell'ultimo strato sulle gallery; **l'abstract dichiara >10% sopra lo stato dell'arte senza fine-tuning** (non ho letto la tabella). Gli autori sostengono che usare reti 2D sia "simplistic and suboptimal": **evidenza contraria alla tua ipotesi**, ma il loro confronto è con VGG-Face del 2015. | LETTO (tabelle dei risultati non lette) |
| **Enhancing 3D Face Recognition: 2D-Aided Generative Augmentation** | 2025-08, Sensors (MDPI) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12389849/ | Pre-addestra su VGGFace2 (3.31M immagini, 9131 id), poi fine-tuning su **normal map** da mesh ricostruite da foto 2D (ExpNet -> 3DMM; variante diffusion TEx-Face). Quindi: conoscenza 2D sia come pesi sia come sorgente di dati 3D. | Parziale: le mesh di training sono 3DMM, l'input è un'immagine di normali. | Solo geometria (normal image). Riportati rank-1: BU-3DFE 99.2, FRGC v2 98.4, Bosphorus 99.3, BU-4DFE 96.5, merge grande 98.2. Guadagni dichiarati sopra la baseline solo-2D: +22.0 / +10.9 / +15.8 punti. **Non ho verificato se il protocollo è cross-dataset.** | LETTO-WEB (numeri da ricontrollare) |
| **Unimodal Face Classification with Multimodal Training (MTUT)** (Teng, Bai) | 2021, IEEE (conferenza non identificata), arXiv 2112.04182 | https://arxiv.org/abs/2112.04182 | Encoder 2D (ResNet-18, FaceNet...) + PointNet su point cloud, **loss di divergenza degli embedding** 2D <-> 3D (L2 pesato in modo adattivo) + autoencoder cross-modali + attributi. Test con solo 3D. | Sì nel senso che usa point cloud (4000 punti campionati). Nessun vincolo di connettività. | Solo geometria (xyz). KinectFaceDB (52 id) e CASIA-3D (123 id), split casuali **in-dataset**. PointNet 79.5 -> 86.6% (Kinect), 82.5 -> 89.8% (CASIA). Piccolo, 2D = ResNet-18 non pre-addestrato su milioni di identità, classificazione chiusa. **È il precedente più vicino a "allinea embedding 3D a embedding 2D"**, ma su scala e teacher molto lontani da quanto vuoi tu. | LETTO |
| **Regressing Robust and Discriminative 3DMMs with a very deep NN** (Tran, Hassner, Masi, Medioni) | 2017, CVPR | https://arxiv.org/abs/1612.04904 | Regressione dei parametri 3DMM da foto con ResNet-101; target: parametri stabili per stesso soggetto, discriminativi. | No (output 3DMM fisso). | Solo da immagini. | SNIPPET |
| **PointMCD** (cross-modal distillation, multi-view) | 2022 (arXiv 2207.03128), venue non verificato | https://arxiv.org/abs/2207.03128 | Distilla un encoder immagini 2D in un encoder di point cloud tramite proiezione multi-vista (visibility-aware feature projection). **Oggetti generici, non volti.** | Sì (point cloud). | Non su volti. | SNIPPET |

**Risposta alla domanda 1 ("funziona sulla sola geometria senza texture?")**: sì, con **fine-tuning**: Kim 2017 (99.24% Bosphorus) e Sensors 2025 mostrano che un backbone da riconoscimento 2D, riaddestrato su depth o normal map, raggiunge ~98-99% rank-1 su benchmark 3D classici. **Non ho trovato nessun numero zero-shot** (ArcFace/AdaFace congelati su render grigi di sole mesh, con verifica fra identità) su scansioni reali. Il caso più vicino:

- **High-Fidelity Single-Image Head Modeling with Industry-Grade Topology** (Alibaba, arXiv 2605.04524, 2026-05, venue non verificato; LETTO). Misura l'identità con **ArcFace su render PyTorch3D con materiale grigio diffuso e luce fissa** (stesso modello per tutti i metodi). Confronto render-mesh vs foto d'ingresso: coseni bassi in assoluto (0.13-0.19 per FaceScape/DECA/MICA/Pixel3DMM sintetico; 0.38 per il loro; 0.41 in-the-wild). Una colonna "ArcFace*" confronta il render della ricostruzione con il render della ground truth (0.32-0.94). Due letture oneste: (a) ArcFace discrimina i metodi anche su render grigi, quindi è usabile come metrica *relativa*; (b) i valori assoluti contro una foto con texture sono bassi, quindi il gap di dominio render-grigio vs foto è reale. **Non è una valutazione di verifica (niente AUC/EER fra identità).** I dati sono 66 teste MetaHuman + 40 asset artistici, non scansioni.

## 2. Distillazione di embedding d'identità 2D in 3D

| Lavoro | Anno / venue | Link | Cosa fa | Indip. dalla topologia | Fuori dominio / sola geometria | Stato |
|---|---|---|---|---|---|---|
| **MICA: Towards Metrical Reconstruction of Human Faces** (Zielonka, Bolkart, Thies) | 2022, ECCV | https://arxiv.org/abs/2204.06607 | ArcFace (backbone ResNet100, pre-addestrato su Glint360K, sub-center) -> MLP a 3 strati -> 300 coefficienti FLAME. Supervisionato con loss L1 pesata sulla **mesh GT** di ~2300 soggetti (8 dataset di scansioni riregistrate a topologia FLAME). Si addestrano solo gli **ultimi 3 blocchi ResNet**. | **No**: decoder FLAME (5023 vertici). In supplementare un decoder model-free SIREN valutato sui vertici del template FLAME, "on par": ma comunque a topologia nota. | Input: **foto con texture**. Output: geometria. Valutato su NoW (mediana 0.90 mm non-metrico, 1.08 mm metrico) e Stirling (escluso dal training). Ablazione: ArcFace > FaceNet; **ArcFace congelato 1.52 mm, L4 allenabile 1.35 mm, tutto allenabile 1.42 mm**, DECA fine-tunato peggiora (overfitting). Messaggio chiave: con ~2k identità 3D, **non riaddestrare tutto**, il prior 2D va preservato. | LETTO |
| **DECA** (Feng, Feng, Black, Bolkart) | 2021, ACM ToG (SIGGRAPH) | https://arxiv.org/abs/2012.04012 | Loss d'identità: coseno fra embedding di una rete di riconoscimento (VGGFace2, Cao et al. 2018) sull'immagine d'ingresso e sul **render texturizzato** della ricostruzione. NoW val: 1.46 mm con la loss, 1.59 senza. | No (FLAME). | Render con albedo, **non** shape-only. Il segnale d'identità passa anche dalla texture predetta. | LETTO |
| **HRN** (Lei et al.) | 2023, CVPR | https://arxiv.org/abs/2302.14434 | Ricostruzione dettagliata da foto. Nel testo letto non compare una loss ArcFace: usa loss percettiva, landmark, priori 3D da scansioni. Citato qui perché va escluso dalla lista "usa ArcFace". | No (BFM + mappe UV). | Valutato su FaceScape, REALY, ESRC. | LETTO (parziale) |
| **EMOCA** (Daněček et al.) | 2022, CVPR | non letto direttamente | Loss percettiva di **emozione**, non d'identità. Per l'identità riusa i parametri DECA, quindi non migliora NoW (confermato indirettamente dal testo di Otto et al., sotto). | No. | n.d. | SNIPPET + citazione in LETTO |
| **Pixel3DMM** (Giebenhain et al.) | 2025 (arXiv 2505.00615), venue non verificato | https://arxiv.org/abs/2505.00615 | Fitting FLAME guidato da predizioni per-pixel (normali, UV). Dallo snippet: usa il prior d'identità di MICA come regolarizzatore. | No. | n.d. | SNIPPET |
| **A Perceptual Shape Loss for Monocular 3D Face Reconstruction** (Otto, Chandran, Zoss, Gross, Gotardo, Bradley) | 2023 (arXiv 2310.19580; formato CGF/Pacific Graphics, venue non verificata) | https://arxiv.org/abs/2310.19580 | Critic CNN che prende **foto + render grigio** della mesh e dà un punteggio di corrispondenza (identità, espressione, posa). Addestrato su 358 identità x 24 espressioni (studio) + dati sintetici in-the-wild. Usato come loss nel fitting e per fine-tuning di DECA. | **Sì, esplicitamente**: lavora solo su render grigi; testato su PCA a 5072/19577/38799 vertici, BFM (35709), FLAME (5023) **senza riaddestrare**. | **Sola geometria, render grigio**. Su NoW i guadagni sono piccoli (mediana 1.0780 vs 1.1757 mm DECA, validazione). Non è un encoder d'identità: è un discriminatore foto-vs-render. **Mostra che "render grigio come interfaccia topology-agnostic" è una pratica già accettata.** | LETTO |
| **Head Similarity** (Wang, Xiao, Liao) | 2026-05 (arXiv 2605.07766), venue non verificato | https://arxiv.org/abs/2605.07766 | Student ViT (init AdaFace-ViT) sulla **testa intera**, allineato con `1 - cos(z_teacher, z_student)` a un teacher AdaFace congelato sul crop del volto. Stessi soggetti visti in due formati (volto allineato per il teacher, testa intera per lo studente). | n.a. (2D). | **Analogia diretta del tuo problema** [MIA]: input spostato fuori dalla distribuzione del teacher. Dati: ArcFace-R50 su testa intera VR@FAR=1e-3 **0.025** (contro 0.870 su volto allineato); AdaFace-ViT 0.078 (contro 0.914); dopo distillazione **0.912**, AUC 0.993. Dimostra che la distillazione con teacher congelato e input accoppiati recupera quasi tutto il divario. Richiede **dati accoppiati** (stesso soggetto, due viste). | LETTO |
| **Learning Domain Agnostic Latent Embeddings of 3D Faces** (Wang et al.) | 2026-01 (arXiv 2601.06484), venue non verificato | https://arxiv.org/abs/2601.06484 | DiffusionNet su HKS/WKS (descrittori intrinseci, 32 dim/vertice) + attention + Neural Jacobian Fields. Spazi latenti separati per **identità** ed **espressione**. Training: 1000 triplette sintetiche ICT. Test: trasferimento di espressione **zero-shot su gatti** (topologia e specie diverse). | **Sì** (descrittori spettrali, DiffusionNet, nessuna corrispondenza). | Identità valutata solo con **t-SNE qualitativo** su 50 id x 100 espressioni, tutti sintetici ICT. Nessuna AUC/EER, nessun dato reale, nessun teacher 2D. **È l'esistente più vicino a un'identità mesh-agnostica, ma non è supervisionata da riconoscimento 2D.** | LETTO |
| **AEGIS** (Wolkiewicz et al.) | 2025-11 (arXiv 2511.17747), venue non verificato | https://arxiv.org/abs/2511.17747 | Attacco avversario sui coefficienti di colore di avatar 3DGS (FLAME-bound) contro ArcFace/AdaFace. Non è un encoder 3D. Utile come prova che ArcFace/AdaFace si applicano a render di avatar. Cosine fra pose dello stesso avatar non mascherato: 0.55-0.82. | n.a. | Con texture. 10 avatar. | LETTO |

**Risposta alla domanda 2**: la supervisione "face recognition 2D -> 3D" esiste in tre forme già pubblicate: (a) **encoder 2D + decoder 3D con topologia fissa** (MICA, Tran 2017); (b) **loss d'identità sul render texturizzato** durante il fitting (DECA, Deng et al. 2019 per FaceNet, citato da DECA); (c) **allineamento di embedding fra encoder 2D e point cloud** su scala piccola (MTUT). Il caso in cui l'encoder è 3D, agnostico alla discretizzazione, e il teacher è un moderno FR su milioni di identità **non l'ho trovato**.

## 3. Feature foundation 2D sollevate sul 3D

| Lavoro | Anno / venue | Link | Cosa fa | Indip. dalla topologia | Volti / espressioni / sola geometria | Stato |
|---|---|---|---|---|---|---|
| **Diff3F** (Dutt, Muralikrishnan, Mitra) | 2024, CVPR | https://arxiv.org/abs/2311.17024 | Rende mesh/point cloud da 100 viste (depth + normali) -> ControlNet/Stable Diffusion le "dipinge" -> estrae feature di diffusione + DINOv2 -> **proietta e media sui vertici** (ball query r = 1% della diagonale). Zero training. | **Sì, esplicitamente**: mesh 2-manifold, non-manifold, point cloud (servono ~8000 punti per una depth liscia). Invariante alla rotazione. | Nessun esperimento su volti. SHREC'19 (umani) acc. 26.41% @1% di tolleranza; FAUST intra-subject 5.29 cm di errore geodesico medio. **Gli autori dicono che le feature sono "semantic instead of geometric"**. Limiti dichiarati: parti non visibili, bias del modello di diffusione. Costo: 2-3 minuti per forma su una RTX 4090. | LETTO |
| **MeshFM** (Zhou, Liu, Lang, Hanocka) | 2026-07 (arXiv 2607.27592), venue non verificato | https://arxiv.org/abs/2607.27592 | Distillazione **a due stadi**: (1) campo di feature neurale per forma, da feature 2D (DINOv2, SAM per correggere "feature bleeding") con distillazione baricentrica; (2) rete feed-forward (PVCNN + triplane transformer) che dalla **point cloud** predice il campo. Augmentation SO(3). 126M parametri, 4 L40S. | **Sì**: input point cloud. | Segmentazione di parti, corrispondenza, deformazione. Categoria "Human-Shape" in segmentazione, nessun volto. Limite dichiarato: scambio sinistra/destra. **Ottimo schema da copiare** per "distilla un teacher 2D in un encoder 3D feed-forward senza etichette 3D". | LETTO |
| **DenseMatcher** (Zhu et al.) | 2024-12 (arXiv 2412.05268), venue non verificato | https://arxiv.org/abs/2412.05268 | Back-proiezione di feature SD-DINO sui vertici + rete 3D di raffinamento + functional map. Oggetti per manipolazione robotica. | Sì (mesh). | Oggetti, non volti. | SNIPPET |
| **Deep Feature Deformation Weights (DFD)** (Liu, Lang, Hanocka) | 2026-01 (arXiv 2601.12527) | https://arxiv.org/abs/2601.12527 | Distillazione baricentrica di feature 2D per pesi di deformazione. Da qui MeshFM prende il distillatore. | Sì. | Non volti. | SNIPPET + citazione in LETTO |
| **Surface-Aware Distilled 3D Semantic Features** (Uzolas et al.) | 2025, SIGGRAPH Asia | https://arxiv.org/abs/2503.18254 | Variante di distillazione 3D che tiene conto della superficie (da snippet). | Probabile. | n.d. | SNIPPET |
| **Stable-SCore** | 2025, CVPR (da snippet) | https://arxiv.org/abs/2503.21766 | Corrispondenza 3D robusta a topologia/forma/posa diverse, ispirata a Diff3F. | Sì (da snippet). | n.d. | SNIPPET |
| **WarpHE4D** (Yun et al.) | 2025, ICCV | https://openaccess.thecvf.com/content/ICCV2025/papers/Yun_WarpHE4D_Dense_4D_Head_Map_toward_Full_Head_Reconstruction_ICCV_2025_paper.pdf | Usa DINOv2 per corrispondenze immagine-UV di teste; da snippet "DINOv2 estrae feature di testa indipendentemente da posa/occlusione". | No (UV fisso). | Teste, ma con immagini. | SNIPPET |
| **SHELLS** (Google, arXiv 2605.31283), **UVFaceFusion** (Tsinghua, 2607.18798), **TopoRig** (Fleet et al., 2609.15746) | 2026, arXiv | vedi sotto | Ricostruzione / rigging di teste con topologie diverse. **TopoRig** (LETTO) è topology-agnostic: MLP/message passing su vertici con feature locali + distanze da landmark (ablazione: senza landmark MAE +54% su ICT, +36% su Pixal3D). Addestrato su 3496 identità **generate** da immagine, valutato su identità e topologie non viste ma rispetto a target ottenuti da trasferimento automatico, non da ground truth. Utile come precedente "landmark relativi = chiave per la generalizzazione cross-topologia" [MIA: rilevante per un encoder d'identità]. | Sì (TopoRig). | Espressione (FACS), non identità. | TopoRig LETTO; gli altri due SNIPPET |

**Risposta alla domanda 3**: Diff3F e MeshFM dimostrano che **il meccanismo (render multi-vista -> feature 2D -> vertici/campo -> encoder feed-forward) funziona e non dipende dalla topologia**. Ma:

- Le feature DINO/diffusione sono **semantiche di categoria** ("questo è un occhio"), e sono state misurate su corrispondenza fra *forme diverse*. L'identità è l'opposto: **variazione fine intra-classe**. [MIA] Un teacher FR (ArcFace/AdaFace) è addestrato proprio per quel regime, quindi è un teacher più adatto di DINO per l'identità.
- **Nessuno dei lavori letti ha usato un teacher di riconoscimento facciale come sorgente di feature da sollevare.** Nessuno ha valutato su volti reali o su identità.
- Possibile variante non fatta: sollevare non solo l'embedding globale, ma le **mappe spaziali intermedie** di ArcFace (feature locali, es. 7x7x512 prima della FC) sui vertici. [MIA, ipotesi non verificata: non so se le feature spaziali di un FR siano abbastanza regolari da essere distillabili.]

## 4. Robustezza all'espressione e al dominio

- **Kim 2017** [LETTO]: espressioni sintetiche via 3DMM migliorano di più il caso VGG-Face pre-addestrato (ablazione "expression generation" è l'augmentation col maggiore incremento su CMC per (c)); neutro vs non-neutro 99.2% su Bosphorus. Su 3D-TEC (gemelli, espressione diversa fra probe e gallery) cala a 79.9-81.3% nei casi III/IV: **espressione + identità simile resta il caso difficile**.
- **MICA** [LETTO]: robustezza all'espressione per costruzione (predice forma neutra) e "ereditata" da ArcFace, ma non misurata separatamente dall'espressione; NoW include un sotto-test con espressioni ma nel testo letto non è scorporato.
- **Head Similarity** [LETTO]: distillazione con teacher congelato corregge lo shift di input (volto -> testa intera) con dati accoppiati. Non riguarda espressione.
- **Learning Domain Agnostic Latent Embeddings** [LETTO]: spazio d'identità separato da quello d'espressione, cluster per identità visibili su 100 espressioni, ma solo qualitativo e sintetico.
- **Dominio** [MIA]: l'unico segnale quantitativo "fuori dominio" nei lavori letti è Kim (train FRGC+CASIA, test Bosphorus/BU-3DFE) e FR3DNet (train sintetico + scansioni proprietarie, test su 10 dataset fusi). **Nessun lavoro letto valuta su 3DMM mai visti o su topologie diverse** per l'identità. Quello più vicino è PSL (topologia, ma non identità).
- Il tuo indizio sull'inefficacia di metriche apprese su 3DMM sintetico è coerente con una critica nota nel testo di FR3DNet: le facce da modelli statistici stanno nello spazio lineare del modello, sono "over smooth" e prive di variazioni ad alta frequenza, e le identità generate coprono variazioni limitate [LETTO, sezione 2-3 di Gilani & Mian 2018]. Gilani e Mian usano come rimedio **interpolazione fra scansioni reali in corrispondenza densa**.

---

## Attenzione sul tuo indizio (AUC 1.000 su crop reali Multiface con ArcFace)

[MIA, da verificare con un esperimento di controllo] Se "crop reali" significa **foto con texture/colore**, l'AUC 1.000 dimostra che ArcFace discrimina i soggetti di Multiface, ma **non** dimostra che identità e forma 3D siano accessibili da una rete 2D. Il segnale può essere interamente texture (tono pelle, barba, occhi). Tre controlli che separano i due casi:

1. ArcFace su **render grigi diffusi, stessa luce, stessa camera** delle mesh Multiface (nessuna texture). È il caso Alibaba 2605.04524, ma lì ci sono solo confronti fra metodi, non verifica fra identità.
2. ArcFace sugli stessi crop con texture **sostituita/media** o con **solo normali**.
3. Stessa AUC su render grigi di mesh a **espressioni diverse** dello stesso soggetto vs identità diverse (verifica sotto espressione).

Dati dalla letteratura letta che dicono cosa aspettarsi: ArcFace-R50 congelato crolla su input non allineato (VR@1e-3 0.025 su testa intera, Head Similarity) e dà coseni bassi fra render grigi e foto (0.13-0.44, Alibaba). **Un ArcFace congelato su render grigi probabilmente non basta**; la letteratura suggerisce fine-tuning parziale (MICA) o distillazione con input accoppiati (Head Similarity).

---

## Cosa sembra già fatto

1. **Fine-tuning di reti di riconoscimento 2D su rappresentazioni 2D della geometria** (depth, normali) per identificare soggetti da scansioni: Kim 2017, Sensors 2025. Risultati ~98-99% rank-1 su benchmark classici; **assumono scansione allineata e rappresentazione fissa**.
2. **ArcFace come encoder di identità con output 3D** (MICA), con insegnamento sul fine-tuning parziale (ultimo blocco ResNet allenabile, meglio che tutto o niente).
3. **Loss d'identità 2D sul render durante il fitting** (DECA): gain modesto su NoW (1.59 -> 1.46 mm), usa il render texturizzato.
4. **Interfaccia render grigio come strumento topology-agnostic** (PSL, Alibaba): critic e metriche ArcFace su render grigi senza riaddestrare al cambio di topologia.
5. **Distillazione di feature 2D foundation in un encoder 3D feed-forward** su point cloud (MeshFM; PointMCD per oggetti), con augmentation SO(3). Tutto per **semantica di categoria**, non per identità.
6. **Distillazione di teacher FR verso input fuori distribuzione**, ma in 2D (Head Similarity).
7. **Identità e espressione disentangled in uno spazio mesh-agnostico** (Learning Domain Agnostic Latent Embeddings), valutazione solo qualitativa e sintetica.

## Cosa sembra uno spazio libero e promettente

[MIA, con la cautela che la ricerca è parziale]

**A. Encoder 3D d'identità, agnostico alla discretizzazione, con teacher FR 2D moderno e congelato.**
Nessun lavoro letto lo fa. Gli ingredienti esistono separatamente: lo schema di MeshFM (distillazione a due stadi, input point cloud, SO(3)), il recupero di Head Similarity (teacher congelato + coppie accoppiate), il fine-tuning parziale di MICA, l'architettura DiffusionNet di 2601.06484.

**B. Due design concreti da confrontare** (nessuno dei due è dimostrato):
- **B1, render-then-2D (interfaccia universale).** Rasterizza la mesh/point cloud/SDF in N viste grigie (o normal map) con luce fissa; lo studente è un backbone FR con ultimi blocchi allenabili (MICA) che aggrega le viste; loss `1 - cos` verso il teacher su una vista *testurizzata* accoppiata. Topologia-agnostico per costruzione (PSL lo mostra a livello di render). Rischio: la parte nel render che il 2D non vede (profondità fine) è sacrificata.
- **B2, encoder nativo su point cloud/campo continuo.** PVCNN+triplane (MeshFM) o DiffusionNet su descrittori intrinseci; target = embedding del teacher mediato su viste. Rischio: l'encoder nativo può imparare scorciatoie di topologia/densità; servono augmentation di remesh/subsampling.

**C. Dati accoppiati: la vera strozzatura.** Head Similarity funziona perché ha coppie (stesso soggetto, due formati). Per il 3D servono (scansione/mesh, foto con texture dello stesso soggetto): Multiface, FaceScape, NeRSemble, e simili [MIA: elenco da conoscenza generale, non verificato in questa ricerca]. Il problema che hai già visto con i 3DMM sintetici (spazio lineare, poco ricco) suggerisce due rimedi dalla letteratura: (i) **interpolare fra scansioni reali in corrispondenza densa** (Gilani e Mian) invece di campionare dal 3DMM, (ii) **mescolare più sorgenti di supervisione con topologie diverse** (TopoRig, ablazione "ICT only" vs "Pixal3D only" vs congiunto).

**D. Supervisione sull'espressione.** Il teacher FR è già invariante all'espressione sul lato 2D. Con coppie (stesso soggetto, espressioni diverse) la distillazione dovrebbe ereditare questa invarianza. Mai misurato per un encoder 3D: aperto, e con 3D-TEC (gemelli, espressioni diverse) come banco di prova dichiarato difficile da Kim 2017.

**E. Protocollo di valutazione che manca in tutti i lavori letti.** Un benchmark **cross-dominio e cross-topologia** per l'identità 3D: train su un insieme di topologie, test su 3DMM mai visti (FLAME, BFM, FaceVerse, ICT) e su scansioni reali, con verifica (AUC/EER, TAR@FAR) e **ablazioni shape-only vs con texture**. Se lo pubblichi, è un contributo in sé.

**F. Feature spaziali dal teacher FR.** Sollevare sui vertici le mappe intermedie di ArcFace per ottenere un campo d'identità per-vertice (utile anche per corrispondenza fra topologie). Ipotesi non verificata, vedi sezione 3.

### Rischi da dichiarare

- Il teacher è addestrato su facce con texture: la parte di identità che è puramente texture non è recuperabile dalla geometria. Lo studente imparerà un'**identità "geometrica vista dal 2D"**, non l'identità biometrica completa. Va dichiarato e misurato (controllo 1-3 sopra).
- Bias demografico ereditato dal teacher (Glint360K, VGGFace2).
- Se i dati accoppiati sono pochi (MICA: ~2300 identità), il rischio di overfitting è alto: congela molto del backbone.

---

## Elenco link per riferimento rapido

- Kim et al. 2017: https://arxiv.org/abs/1703.10714
- Gilani & Mian, CVPR 2018: https://arxiv.org/abs/1711.05942
- Sensors 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC12389849/
- MTUT: https://arxiv.org/abs/2112.04182
- MICA, ECCV 2022: https://arxiv.org/abs/2204.06607
- DECA, ToG 2021: https://arxiv.org/abs/2012.04012
- HRN, CVPR 2023: https://arxiv.org/abs/2302.14434
- Perceptual Shape Loss 2023: https://arxiv.org/abs/2310.19580
- Alibaba head modeling 2026: https://arxiv.org/abs/2605.04524
- Head Similarity 2026: https://arxiv.org/abs/2605.07766
- Learning Domain Agnostic Latent Embeddings 2026: https://arxiv.org/abs/2601.06484
- TopoRig 2026: https://arxiv.org/abs/2609.15746
- AEGIS 2025: https://arxiv.org/abs/2511.17747
- Diff3F, CVPR 2024: https://arxiv.org/abs/2311.17024
- MeshFM 2026: https://arxiv.org/abs/2607.27592
- DenseMatcher: https://arxiv.org/abs/2412.05268
- DFD: https://arxiv.org/abs/2601.12527
- Surface-Aware Distilled 3D Semantic Features: https://arxiv.org/abs/2503.18254
- Stable-SCore: https://arxiv.org/abs/2503.21766
- PointMCD: https://arxiv.org/abs/2207.03128
- Tran et al. CVPR 2017: https://arxiv.org/abs/1612.04904
- Pixel3DMM: https://arxiv.org/abs/2505.00615
- ArcFace (riferimento): https://arxiv.org/abs/1801.07698
