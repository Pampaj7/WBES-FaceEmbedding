# Generalizzazione fuori dominio dell'encoder DiffusionNet per l'identità 3D: letteratura e direzioni

Data: 2026-10-06. Ricerca con WebSearch/WebFetch e alphaXiv; nessun download di PDF o dataset.

## Legenda dell'affidabilità

- **[V]** verificato: ho aperto la pagina arXiv o il README in questa sessione e ho letto titolo, autori e abstract.
- **[S]** solo da risultati di ricerca: titolo, autori e venue compaiono nei risultati, ma non ho letto il contenuto.
- **[M]** da memoria, non ricontrollato in questa sessione. Va verificato prima di citarlo.

Tutto ciò che è etichettato "rilevanza" o "ipotesi" è mio giudizio, non un risultato misurato.

**Risultato negativo da riportare.** Non ho trovato nessun lavoro che studi esplicitamente la generalizzazione leave-one-3DMM-out per una metrica d'identità su mesh. Il problema è quindi poco coperto in letteratura. Le direzioni sotto sono per analogia con domini vicini (segmentazione medica, point cloud DG, shape matching), non repliche di risultati sul volto.

---

## 1. Domain generalization e randomizzazione per forme 3D

| Lavoro | Anno / venue | Link | Cosa dimostra | Rilevanza |
|---|---|---|---|---|
| SynthSeg (Billot et al.) [S] | 2021 arXiv, poi MedIA 2023 | https://arxiv.org/abs/2107.09559 | Domain randomisation: contrasto e risoluzione delle immagini sintetiche vengono randomizzati *completamente*, anche oltre il realistico. La CNN segmenta scansioni reali di qualsiasi contrasto e risoluzione senza retraining. Per l'addestramento servono solo le etichette. | Analogo diretto del nostro caso: 2 3DMM sono troppo pochi e troppo "puliti". Randomizzare oltre la varietà è l'idea trasferibile. |
| SynthMorph (Hoffmann et al.) [S] | IEEE TMI 2021 (arXiv 2004.10282) | https://arxiv.org/abs/2004.10282 | Training di registrazione solo su dati sintetici ottenuti da campi casuali. La rete diventa invariante al contrasto e alla geometria. | Fonte dell'idea "random fields" della tua richiesta. Si traduce in campi di spostamento casuali, a bassa frequenza e a scale multiple, sopra l'identità del 3DMM. |
| Domain-randomized deep learning for neuroimage analysis (Hoffmann) [V] | IEEE Signal Processing Magazine 2025 | https://arxiv.org/abs/2507.13458 | Tutorial/revisione. Il DR migliora la generalizzazione e riduce l'overfitting al dominio. Costo: più calcolo in training. | Base per motivare l'esperimento. Non contiene nulla sul volto. |
| Brain-ID (contrast-agnostic representations) [S] | ECCV 2024 | https://link.springer.com/chapter/10.1007/978-3-031-73254-6_19 | Apprendimento di rappresentazioni, non solo di segmentazioni, agnostiche al contrasto con dati sintetici randomizzati. | Vicino al nostro scopo: un embedding invariante al dominio, non un task head. |
| MetaSets (Huang et al.) [S] | CVPR 2021 (arXiv 2204.07311) | https://arxiv.org/abs/2204.07311 | Benchmark sim-to-real per point cloud 3D. Metodo di meta-learning con trasformazioni geometriche per la 3D domain generalization. | I titoli dei risultati indicano che le trasformazioni *progettate per il DG* battono l'augmentation 3D standard. Non verificato nel testo. |
| SUG: Single-dataset Unified Generalization [S] | 2023 (arXiv 2305.09160) | https://arxiv.org/abs/2305.09160 | Si addestra su un solo dataset e si generalizza a molti. | Caso analogo: pochi domini di training, molti di test. |
| PointDGMamba [S] | AAAI 2025 (arXiv 2408.13574) | https://arxiv.org/abs/2408.13574 | Introduce il benchmark PointDG-3to1. | Protocollo leave-one-domain-out di riferimento. |
| DG-MVP [S] | 2025 (arXiv 2504.12456) | https://arxiv.org/abs/2504.12456 | DG per classificazione di point cloud tramite viste multiple. | Marginale per noi. |
| MATE (Mirza et al.) [S] | ICCV 2023 (arXiv 2211.11432) | https://arxiv.org/abs/2211.11432 | Primo test-time training per 3D. L'obiettivo a test-time è un masked autoencoder. Robustezza a corruzioni per classificazione di point cloud. | Possibile TTA per mesh, ma dimostrato solo su classificazione e corruzioni, non su shift di dominio dell'identità. |
| Combining 3D Morphable Models (Ploumpis et al.) [S] | CVPR 2019 (arXiv 1903.03785) | https://arxiv.org/abs/1903.03785 | Metodi per combinare 3DMM con template, dati e capacità di rappresentazione diversi. | Strumento per fondere basi diverse. Non per l'identità, ma per costruire una varietà più ampia. |
| Verifier-Guided Synthetic Augmentation for 3D Human Shape Generation [V] | 2026 (arXiv 2610.04006) | https://arxiv.org/abs/2610.04006 | Nel corpo umano, PCA locali per modo più un verificatore di plausibilità producono forme nuove accettate. Più diversità senza forme implausibili. | Schema per generare identità fuori dalla varietà ma ancora plausibili, con un filtro. Non è sul volto. |

**Limite per la nostra analisi.** SynthSeg e SynthMorph hanno un'etichetta fissa che la randomizzazione non cambia. Qui le coppie sintetiche hanno una "verità" di distanza d'identità definita dal 3DMM. Deformazioni troppo forti possono alterare quella verità. Serve quindi una randomizzazione che non cambi la distanza di riferimento oppure la cambi in modo noto. Lo si può fare facendo deformare in modo identico entrambe le mesh di una coppia, solo se la metrica è invariante a quella deformazione, oppure usando un target ricalcolato.

---

## 2. Apprendimento d'identità da dati reali non etichettati o multi-acquisizione

| Lavoro | Anno / venue | Link | Cosa dimostra | Rilevanza |
|---|---|---|---|---|
| FML: Face Model Learning from Videos (Tewari et al.) [S] | CVPR 2019 | https://openaccess.thecvf.com/content_CVPR_2019/papers/Tewari_FML_Face_Model_Learning_From_Videos_CVPR_2019_paper.pdf | Training self-supervised multi-frame. Parametri d'identità condivisi tra i frame dello stesso video, con shape, aspetto, espressione e illuminazione separati. | Principio "stessa persona, espressioni diverse" per costruire positivi. È su immagini, non su mesh. |
| Learning Complete 3D Morphable Face Models from Images and Videos [S] | 2020 (arXiv 2010.01679) | https://arxiv.org/abs/2010.01679 | Apprende 3DMM da immagini e video. | Contesto. |
| MICA (Zielonka, Bolkart, Thies) [S] | ECCV 2022 | https://arxiv.org/abs/2204.06607 | Shape metrica da una rete di face recognition pre-addestrata su dati 2D in the wild, più dati 3D metrici. | Mostra che serve un ponte dalla supervisione d'identità robusta alla geometria. In senso inverso: un segnale d'identità 2D potrebbe supervisionare un encoder 3D. |
| Disentangled Face Identity Representations for 3D FR and Expression Neutralisation (Kacem et al.) [V] | 2021 (arXiv 2104.10273) | https://arxiv.org/abs/2104.10273 | Autoencoder a grafo per mesh facciali, GAN per neutralizzare l'espressione, sottorete di identità. Tre dataset pubblici (nomi non letti). | Precedente di identità 3D disaccoppiata dall'espressione. Non ho verificato la valutazione cross-dataset. |
| Learning from Millions of 3D Scans for Large-scale 3D FR (Gilani, Mian) [S] | CVPR 2018 (arXiv 1711.05942) | https://arxiv.org/abs/1711.05942 | CNN per FR 3D addestrata su 3.1 M di scansioni di 100K identità, ottenute con un modello generativo/augmentation. Rank-1 98.74% su 27K probe. | **Il precedente più vicino al nostro caso**: grande training sintetico, test su reale. Non ho letto cosa succede cross-dataset. Da leggere. |
| Reconstructing A Large Scale 3D Face Dataset for Deep 3D Face Identification (Yu, Zhang, Li) [V] | 2020 arXiv, **ritirato** a giugno 2022 | https://arxiv.org/abs/2010.08391 | Ricostruiscono milioni di scansioni da VGGFace2 con ExpNet. Pretraining 2D e fine-tuning 3D. Dichiarano 97.6% su FRGC v2.0, 98.4% su Bosphorus, 98.8% su BU-3DFE. | **Attenzione: il paper è stato ritirato.** Non usare i numeri come evidenza. L'idea (identità 2D come supervisione di scala) resta, ma non è validata. |
| Deep 3D face identification (Kim, Hernandez, Choi, Medioni) [M] | IJCB 2017 | arXiv 1703.10714 | Dalla ricerca emerge solo che è citato. Ricordo augmentation sintetica per FR 3D, non verificata. | Da verificare. |
| Unsupervised Contrastive Learning for Efficient and Robust Spectral Shape Matching (Luo, Chen) [V] | 2026 (arXiv 2603.18924) | https://arxiv.org/abs/2603.18924 | Perdita contrastiva non supervisionata sui descrittori in shape matching spettrale. Dichiara risultati forti anche su casi non isometrici. | Mostra che la perdita contrastiva su feature DiffusionNet-like funziona con corrispondenze note. Per il volto serve avere coppie positive. |
| Neural Descriptors: Self-Supervised Learning of Robust Local Surface Descriptors (Technion) [S] | 2025 (arXiv 2503.03907) | https://arxiv.org/abs/2503.03907 | Descrittori locali autosupervisionati, più robusti di HKS/WKS/SHOT alla sensibilità. | Collega descrittori classici e appresi. Locale, non globale d'identità. |

**Nota onesta.** Non ho trovato un metodo pubblicato di apprendimento contrastivo d'identità su mesh facciali 3D reali, senza etichette, con positivi da video 4D. Il caso è plausibile ma, per quanto emerso dalla ricerca, non coperto. Questo vale sia come opportunità sia come rischio.

---

## 3. Descrittori intrinseci invarianti a isometria ed espressione

| Lavoro | Anno / venue | Link | Cosa dimostra | Rilevanza |
|---|---|---|---|---|
| Three-Dimensional Face Recognition (Bronstein, Bronstein, Kimmel) [S] | IJCV 64(1), 2005 | https://dl.acm.org/doi/10.1007/s11263-005-1085-y | Le espressioni modellate come isometrie. Forme canoniche invarianti al piegamento tramite embedding delle distanze geodetiche. Riconoscimento robusto all'espressione. | Fondamento teorico del nostro approccio intrinseco. Non ho verificato i numeri né i limiti. **[M]**: l'apertura della bocca viola l'ipotesi isometrica. Controllare nel testo. |
| DiffusionNet (Sharp et al.) [V sul README, S sull'articolo] | ACM TOG 2022 (arXiv 2012.00888) | https://arxiv.org/abs/2012.00888 | Il README dice che xyz funziona bene, ma consiglia HKS per essere "naturalmente invariante a deformazioni isometriche". Invarianza a movimenti rigidi solo se le feature restano invariate. Rotazione casuale applicata dopo il calcolo degli operatori. | **Rilevante per il nostro sintomo**: con xyz in ingresso l'encoder non è invariante a rotazione. Con HKS lo diventa, con il costo di perdere informazione non isometrica. |
| Learning Domain Agnostic Latent Embeddings of 3D Faces (Wang et al.) [V] | WACV 2026 Workshop LENS (arXiv 2601.06484) | https://arxiv.org/abs/2601.06484 | Encoder su volti con ingresso HKS/WKS, latenti separati per identità ed espressione. Trasferimento zero-shot dell'espressione da umano ad animale. | **Il lavoro più vicino alla nostra architettura.** Mostra che HKS/WKS permettono di attraversare geometrie molto diverse. Il compito è l'espressione, non la metrica d'identità. |
| Frequency-Scale Saliency for Spectral Descriptor Analysis [S] | 2026 (arXiv 2606.07791) | https://arxiv.org/abs/2606.07791 | Analizza i modi di fallimento di HKS e WKS nel retrieval non rigido. | Utile per capire dove HKS perde identità. Non letto. |
| Auditing Training-Free 3D Shape Retrieval with Diffused Geodesic Moments [S] | 2026 (arXiv 2605.29004) | https://arxiv.org/abs/2605.29004 | Audit di descrittori senza training. Mostra che normalizzazione, aggregazione e metrica contano quanto il segnale locale. | Avvertenza: la scelta della metrica sul descrittore può spostare i risultati. |
| Intrinsic and Triangulation-Agnostic Attention [S] | 2026 (arXiv 2607.24954) | https://arxiv.org/abs/2607.24954 | Attention su mesh intrinseca e agnostica alla triangolazione. | Alternativa a DiffusionNet. Non valutata. |
| Deep Orientation-Aware Functional Maps (Donati, Corman, Ovsjanikov) [S] | CVPR 2022 (arXiv 2204.13453) | https://arxiv.org/abs/2204.13453 | Su DiffusionNet, perdita con campi vettoriali per risolvere le ambiguità di simmetria senza feature estrinseche. | Mostra che le reti puramente intrinseche hanno limiti (simmetrie) e come aggirarli. |

**Avvertenza di principio.** HKS e le forme canoniche sono invarianti alle isometrie, ma l'identità tra due volti è per definizione una differenza *non isometrica*. Un descrittore troppo invariante può schiacciare proprio l'identità. Il rischio è reale e non l'ho trovato discusso nei lavori letti.

---

## 4. Equivarianza e invarianza alla rotazione

| Lavoro | Anno / venue | Link | Cosa dimostra | Rilevanza |
|---|---|---|---|---|
| Vector Neurons (Deng et al.) [S] | ICCV 2021 (arXiv 2104.12229) | https://arxiv.org/abs/2104.12229 | Framework per reti SO(3)-equivarianti con neuroni vettoriali. | Base di RINO. |
| RINO: Rotation-Invariant Non-Rigid Correspondences (Gao, Hu-Chen, Deng, Marin, Guibas, Cremers) [V] | CVPR 2026 (arXiv 2603.27773) | https://arxiv.org/abs/2603.27773 | Rete invariante SO(3) con vector neurons e functional map complesse. Estrae feature dalla geometria grezza **senza pre-allineamento**. Basata sulle revisioni di DiffusionNet in versione SO(3)-equivariante. | Modello diretto per una versione rotation-invariant del nostro encoder. Non ho verificato se serve augmentation. |
| Frame Averaging (Puny et al.) [S] | ICLR 2022 (arXiv 2110.03336) | https://arxiv.org/abs/2110.03336 | Invarianza/equivarianza media su una cornice canonica calcolata dai dati. | Alternativa leggera: PCA della mesh come cornice. |
| Endowing Deep 3D Models with Rotation Invariance Based on PCA (Xiao et al.) [V] | 2019 (arXiv 1910.08901) | https://arxiv.org/abs/1910.08901 | Coordinate in cornice PCA. L'ambiguità di segno degli autovettori viene gestita elaborando tutte le cornici e fondendole con self-attention. | Alternativa economica. Per il volto, le cornici PCA sono probabilmente stabili (forma allungata), ma da misurare. |
| Shape-Pose Disentanglement using SE(3)-equivariant Vector Neurons [S] | ECCV 2022 | https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630461.pdf | Disaccoppia forma e posa con Vector Neurons. | Idea: separare forma e posa invece di invarianza forzata. |
| CRIN (Lin et al.) [S] | 2023 (arXiv 2303.03101) | https://arxiv.org/abs/2303.03101 | Invarianza con un sistema di riferimento centrifugo. | Marginale. |

---

## 5. Dataset reali con più acquisizioni per persona

Le licenze sono quelle che emergono dalle ricerche e non sono state lette nei testi di accordo: **verificare prima dell'uso**.

| Dataset | Contenuto | Accesso / licenza (da verificare) |
|---|---|---|
| FaceScape [S] | 16.940 volti 3D con texture, 847 soggetti, 20 espressioni ciascuno | Rilasciato per ricerca non commerciale. Via accordo. https://github.com/zhuhao-nju/facescape |
| Florence 4D Facial Expression [S] | Sequenze 4D con più espressioni | Zenodo e pagina MICC. **Le fonti riportano licenze incoerenti (CC BY 4.0 e MIT): controllare sul record.** https://zenodo.org/records/13939013 |
| 4DFAB [S] | 180 soggetti in 4 sessioni in 5 anni (2012-2017), oltre 1.8 M di mesh | Il più utile per l'identità a distanza di tempo. La licenza non è emersa. Accesso presumibilmente via richiesta a iBUG. https://ibug.doc.ic.ac.uk/resources/4dfab/ |
| Bosphorus [S] | 4652 scansioni, 105 soggetti, 31-54 per soggetto | Accordo di licenza. Non verificato. |
| BU-3DFE [S] | 2500 scansioni, 100 soggetti, 7 espressioni a 4 livelli | Accordo di licenza. Non verificato. |
| FRGC v2.0 [S] | 4950 scansioni, 466 soggetti, 1-22 per soggetto | Accordo. Non verificato. |
| FaceWarehouse [S] | 3000 sequenze Kinect RGB-D, 150 individui | Non verificato. Qualità bassa. |
| NeRSemble [S] | 222 teste, 4734 sequenze multi-vista | Rilascio dichiarato dagli autori, licenza non verificata. Non sono mesh: servirebbe ricostruire. https://tobias-kirschstein.github.io/nersemble/ |
| Headspace (LYHM) [S] | 1518 soggetti con cuffia in lattice | Gratuito per ricerca universitaria non commerciale, via modulo e dipendente universitario verificabile. Una sola scansione per soggetto: non serve per coppie intra-persona. https://www-users.york.ac.uk/~np7/research/Headspace/ |
| Multiface (Meta) [S] | Più identità con sequenze di espressioni | Repository archiviato. Licenza non emersa. |

Il survey "3D Face Recognition: Two Decades of Progress and Prospects" (Guo et al., ACM CSUR 2023, https://dl.acm.org/doi/10.1145/3615863) [S] afferma che il deep learning per FR 3D è limitato dalla mancanza di dataset 3D grandi. I benchmark restano piccoli (FRGC, Bosphorus, BU-3DFE, Gavab).

---

## Cosa dicono le evidenze sul nostro sintomo

Sono ipotesi, non risultati. Nessuna è stata testata in questa sessione.

1. **Xyz come scorciatoia.** Le coordinate assolute codificano convenzioni specifiche del 3DMM: ritaglio, estensione della testa, scala, orientamento, presenza di collo e orecchie. Questo spiegherebbe sia la sensibilità all'orientamento sia il fallimento su HIFI3D e FaceVerse. È coerente con la raccomandazione HKS nel README di DiffusionNet. Non è provato sul nostro modello.
2. **Poche varietà.** Due 3DMM danno due varietà lineari a bassa dimensione. Il fatto che il secondo aggiunga poco è coerente con l'idea che serva *diversità qualitativa* (campi casuali, basi mischiate) e non *più 3DMM simili*. È un'inferenza per analogia con SynthSeg e SynthMorph.
3. **La registrazione non rigida batte l'encoder** perché usa informazioni di corrispondenza dense che l'encoder deve inferire. Il vantaggio del nostro metodo è velocità e indipendenza dal template, non per forza accuratezza.
4. **Rischio HKS.** L'invarianza isometrica può togliere identità (vedi sezione 3).

---

## Le 3 direzioni più promettenti, con falsificazione rapida

L'ordine è il mio giudizio sulla probabilità di successo per rapporto costo/beneficio.

### Direzione 1: via xyz, tramite HKS/WKS più normalizzazione e canonicalizzazione

**Perché.** È la più economica e l'unica con un sintomo che la indica direttamente (sensibilità all'orientamento). Supporto: README DiffusionNet [V]; encoder HKS/WKS su volti 3D [V, 2601.06484]; RINO e Frame Averaging come alternative invarianti [V/S].

**Varianti da confrontare:**
- (a) HKS+WKS al posto di xyz.
- (b) xyz in cornice PCA con gestione del segno.
- (c) HKS+xyz normalizzato (centro, scala).

**Esperimento di falsificazione (ore, non giorni).** Riaddestra con (a), (b) e (c), a parità di dati BFM+ICT. Misura Spearman su HIFI3D e FaceVerse e sotto rotazioni SO(3) casuali.
- **Falsificata se** la Spearman su HIFI3D/FaceVerse non migliora oltre il rumore tra semi diversi, oppure la Spearman in dominio crolla di molto. Il secondo caso indica che HKS schiaccia l'identità.
- **Controllo diagnostico preliminare, senza riaddestrare.** Valuta l'encoder attuale su HIFI3D/FaceVerse dopo aver allineato PCA, centrato e scalato le mesh come BFM/ICT. Se la Spearman sale molto, la causa è la convenzione di coordinate.

### Direzione 2: randomizzazione fuori varietà dei dati sintetici (stile SynthSeg/SynthMorph)

**Perché.** È l'unica soluzione con evidenza forte di generalizzazione in domini vicini [S: SynthSeg, SynthMorph; V: rassegna di Hoffmann]. Affronta direttamente la scarsità di diversità.

**Cosa randomizzare, in ordine di costo:**
- campi di spostamento casuali a bassa frequenza e a scale multiple, sopra l'identità;
- basi PCA ruotate o miscelate tra BFM e ICT, con spazio dei coefficienti campionato oltre i limiti usuali;
- scala e proporzioni per regione;
- risoluzione e rimeshing casuali;
- rumore sul bordo e ritagli diversi.

**Problema da risolvere.** La distanza di riferimento dopo la deformazione. Opzioni: applicare la stessa deformazione a entrambe le mesh della coppia (se l'etichetta è invariante), oppure usare come riferimento una distanza ricalcolata dopo la registrazione.

**Esperimento di falsificazione.**
1. Training solo su BFM, con e senza campi casuali.
2. Test su ICT, HIFI3D e FaceVerse, tutti mai visti.
3. Se i campi casuali non spostano la Spearman fuori dominio oltre il rumore tra semi, la direzione è falsificata nella forma più semplice.
4. Confronto aggiuntivo: BFM+ICT con e senza randomizzazione. Se la randomizzazione conta più dell'aggiunta di un secondo 3DMM, la lettura "serve diversità, non più modelli" è confermata.

### Direzione 3: fine-tuning su scansioni reali con positivi intra-persona, con la registrazione come insegnante

**Perché.** Se il divario è sintetico-reale, solo dati reali lo chiudono. Esistono scansioni con più espressioni per persona (FaceScape 847×20, 4DFAB 180 soggetti in 4 sessioni, Bosphorus, BU-3DFE, Florence 4D) [S]. La registrazione non rigida, che oggi batte l'encoder, può dare distanze pseudo-etichetta, quindi anche negativi duri, oppure essere distillata. Precedenti: principio multi-frame di FML [S], perdita contrastiva su feature DiffusionNet in shape matching [V], FR 3D su grande scala [S].

**Esperimento di falsificazione.**
1. Fine-tuning dell'encoder su una sola fonte reale con licenza accessibile (per esempio Florence 4D o FaceScape), con perdita contrastiva: stessa persona con espressioni diverse come positivi, persone diverse come negativi.
2. Valuta su una fonte reale *diversa*, mai vista, e su HIFI3D/FaceVerse.
3. Falsificata se il fine-tuning migliora solo sul dataset di training ma non su quello diverso. In quel caso il problema è la specificità del dominio di acquisizione e non la diversità sintetica.
4. Controllo: confronta con la sola registrazione come baseline nello stesso split.

**Rischio noto.** Una parte della diversità tra le espressioni di una persona non è isometrica (apertura della bocca). Potrebbe servire mascherare la regione della bocca o ridurne il peso.

### Cosa rimane fuori da queste tre

- **Test-time adaptation (MATE e simili).** Dimostrata solo su classificazione di point cloud e corruzioni [S]. Non ho evidenza che aiuti sull'identità. Priorità bassa.
- **Rete SO(3)-equivariante completa (RINO, Vector Neurons).** Costosa. Va considerata solo se la Direzione 1 con HKS o cornice PCA non basta.
- **Identità 2D come supervisione di scala (MICA, 2010.08391).** Il paper 2010.08391 è ritirato. Un segnale d'identità da face recognition 2D resta un'idea ma richiede una pipeline di ricostruzione, e questo reintroduce una dipendenza da un 3DMM.

## Cosa non sono riuscito a verificare

- Se esistono lavori che studiano esplicitamente la generalizzazione tra 3DMM per l'identità su mesh. Non ne ho trovati.
- I numeri cross-dataset di Gilani e Mian (CVPR 2018) e di Kim et al. (IJCB 2017). Sono il precedente più vicino e meritano una lettura completa.
- Le licenze effettive di Bosphorus, BU-3DFE, FRGC, 4DFAB, Multiface e NeRSemble.
- Il comportamento di HKS/WKS sul riconoscimento facciale cross-dataset, in particolare se perdono identità.
- Se Frame Averaging o le cornici PCA sono stabili sulle teste umane.
