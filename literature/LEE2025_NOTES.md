# Lee, Kang, Lee (ACM TOMM 2025) - 3D Facial Shape Similarity with Deep Perceptual Representations

Riferimenti di pagina: numerazione articolo 183:N (= pagina N del PDF, 27 pagine).

Nota di metodo: `pdftoppm` non e' installato, quindi il Read tool non ha potuto renderizzare le pagine. Il testo e' stato estratto con Ghostscript (`txtwrite`). Tutto il testo, le tabelle e le didascalie sono stati letti. Le figure NON sono state viste, solo le loro didascalie. Il testo estratto ha lettere perse, ma i numeri delle tabelle sono leggibili.

## 1. Cosa propongono (pp. 2-12)

- **Obiettivo**: una misura di similarita' di forma fra due mesh facciali 3D con topologia e connettivita' arbitrarie, correlata alla percezione umana (Intro p. 2).
- **Rappresentazione**: render multi-vista, non mesh (Sez. 3.1, pp. 5-6, Fig. 3-4).
  - Per ogni vista I = [D; N], cioe' mappa di profondita' piu' mappa di normali a 3 canali, 160x160x4 (p. 12).
  - Proiezione ortografica con Z-buffer. Le normali di vertice sono interpolate baricentricamente.
  - Pipeline (Fig. 4, p. 6):
    1. normalizzazione per deviazione standard dei vertici;
    2. render da E viste casuali e rilevamento dei landmark 2D con un DCNN (Deng 2018, [12]);
    3. voto di maggioranza e back-projection per ottenere landmark 3D;
    4. allineamento rigido frontale;
    5. N+1 telecamere virtuali.
  - Le telecamere sono 5, a intervalli di 40 gradi in yaw, longitudine [-80, 80] (p. 12). Piu' viste o piu' latitudini non aiutano.
- **Modello percettivo**: una rete dedicata addestrata da zero sulle viste. Non sono usati ArcFace, LPIPS, CLIP o DINO.
  - Backbone VGG19 (p. 7), non e' dichiarato alcun pretraining. Embedding IPR a 512 dimensioni per vista (p. 12).
  - Training: triplet loss FaceNet-like con margine alpha = 0.4 (Eq. 1, p. 7).
  - Il triplet e' **view-specific**: ancora, positivo e negativo sono presi dalla stessa vista (Eq. 1).
  - **Regional consistency**: regolarizzatore (Eq. 2-3, p. 8, lambda = 0.1). Impone feature uguali sui pixel sovrapposti fra viste diverse, trovati tramite l'indice di triangolo dallo Z-buffer. Applicato a ancora, positivo e negativo.
  - Supervisione: **solo identita' dei soggetti** nei dataset di scansioni (positivo = stessa persona), non giudizi umani.
- **Due similarita'** (Sez. 3.2.3, pp. 8-9):
  - SS-SIM: distanza ID-MRF fra feature intermedie (conv4-2, conv3-2) della rete, "geometrica/spaziale" (Eq. 4).
  - GS-SIM: distanza L2 fra gli embedding IPR dell'ultimo layer, "identita'/generale" (Eq. 5).
- **Integrazione multi-vista** (Sez. 3.2.4, pp. 9-10): una SVR con kernel RBF per ciascuna delle due similarita'. Prende le similarita' delle 5 viste e predice il MOS (Eq. 6-7). Iperparametri: gamma = 0.2, C = 1 (p. 12).
- **Training** (p. 12): 1000 epoche, Adam, 5 GPU 2080 Ti (una per vista), batch 60, lr da 1e-3 a 1e-5. Augmentation per down-sampling dei vertici (x0.75, x0.5, x0.1).
- **Applicazioni** (Appendice A, pp. 24-27):
  - fitting di 3DMM a scansioni 3D e a landmark 2D, usando SS-SIM e GS-SIM come loss percettive (Tab. A1-A2);
  - riconoscimento 3D con rank-1 accuracy (Tab. A3).

## 2. Definizione e validazione della "somiglianza" (Sez. 4.1-4.2, pp. 11-13)

- **Dataset** (pp. 11-12): scansioni reali.
  - FRGCv2: 4007 scansioni, 466 soggetti.
  - BU-3DFE: 2500 scansioni, 100 soggetti.
  - Bosphorus: 4666 scansioni, 105 soggetti. Nuvole di punti, ricostruite con screened Poisson.
  - CASIA: 4624 scansioni, 123 soggetti.
  - TEC 3D-Twins: 428 scansioni, 214 soggetti (gemelli).
  - ICT-3DRFE: 345 scansioni, 23 soggetti. E' il dataset di scansioni Light Stage di Ma et al. 2007, NON ICT-FaceKit.
  - Dati propri: 176 mesh di 22 soggetti, da un sistema con 54 DSLR e 18 sensori RGB-D.
  - Split casuale 80/20 per soggetti, senza sovrapposizione (p. 12).
- **GT di training della rete**: l'identita' dei soggetti.
- **GT di valutazione**: il MOS di uno studio umano. Il MOS e' anche l'etichetta di training della SVR.
- **Studio umano** (Sez. 4.2, pp. 13-14):
  - 25 partecipanti non esperti, 21-36 anni, vista normale. Segue ITU-R BT.500 e BT.2021.
  - 40 sequenze x 5 coppie = 200 coppie. Le stesse 40 sequenze sono usate in tutti e 3 i sotto-test.
  - Giudizio: scala Likert a 5 punti sulla somiglianza della forma 3D. Istruzioni: concentrarsi sulla geometria, ignorare shading e luce.
  - Le coppie sono scelte con variazione sufficiente, per lo piu' entro distanza IPR 0.5.
  - MOS = media di 25 voti, normalizzata linearmente in [0, 1].
  - Tre sotto-test (Fig. 1, 5, 8):
    1. viste diverse per le due facce;
    2. stessa vista;
    3. multi-vista con rotazione libera. **Il MOS del sotto-test 3 e' la GT** della SVR.
  - Durata circa 20-25 minuti, nessun limite di tempo, breve sessione di training.
  - Qualita' dei dati: deviazione standard media per coppia < 0.15, test di Friedman con post-hoc di Wilcoxon (p. 13).
  - Metriche: PLCC e SRCC fra metrica e MOS.
- **Ipotesi centrale** (Fig. 1 p. 3, Fig. 5 p. 7): gli umani confrontano le viste corrispondenti. La correlazione col MOS e' piu' alta con viste uguali che con viste diverse (Tab. 4: PLCC 0.81 con vista uguale, 0.66 con viste diverse).

## 3. Risultati e baseline (Sez. 4.4-4.8, pp. 14-19)

- **Baseline, tutte geometriche e non percettive** (p. 14):
  - Zhao 2014 [68]: curvatura geodetica.
  - Zhao 2018 [69]: distanza di Frechet fra geodetiche.
  - Lv 2020 [41]: landmark piu' distanza geodetica. Lee e' coautore di un lavoro citato nella stessa famiglia.
  - Ma 2021 [43]: campo di deformazione.
  - **Besnier 2023 [7]**: misura generativa/appresa invariante alla mesh. E' la baseline piu' vicina a noi.
- **Nessun confronto** con ArcFace, LPIPS, CLIP, DINO o altre metriche percettive su render.
- **Tab. 1** (SS-SIM vs MOS, PLCC/SRCC, a x1.0 e x0.1 vertici):
  - x1.0: 0.894 / 0.893. La migliore baseline e' Ma, 0.871 / 0.874.
  - x0.1: 0.782 / 0.790. Le baseline geodetiche scendono a circa 0.59-0.62, Besnier 0.727.
- **Tab. 2** (GS-SIM, "personal similarity"): x1.0 0.897 / 0.883 contro Ma 0.862 / 0.864 e Besnier 0.870 / 0.870.
- **Tab. 3**: test di significativita' su 30 ripetizioni, con ANOVA p = 0.021, Tukey HSD e t-test. Le colonne p-value sono confuse nel testo estratto.
  - Il numero di campioni indipendenti per i "30 ripetuti" non e' chiaro.
  - Il modo in cui sono stati ottenuti i 30 campioni non e' spiegato.
- **Tab. 4**:
  - singola vista diversa: 0.66;
  - singola vista uguale: 0.81;
  - frontale: 0.81;
  - multi-vista uniforme: 0.846.
- **Tab. 5**: SVR 0.894 / 0.897 contro FC lineare 0.887 e 3 FC 0.890, quindi poca differenza.
- **Tab. 7**: la sola depth da' 0.86, la sola normal 0.63.
- **Tab. 8** (ablation): senza view-specificity PLCC 0.860, senza regional consistency 0.880, senza entrambe 0.826.
- **Tab. 6**: pipeline multi-vista circa 0.13 s in test (singola vista 0.126 s).
- **Tab. A3**: rank-1 face recognition con GS-SIM, 99.3-100% su 5 dataset (a 160x160).

Cautele sui numeri (valutazione critica nostra):
- Le correlazioni sono su sole 200 coppie.
- Non e' chiaro se la SVR sia addestrata e testata su coppie MOS disgiunte. Il testo dice solo 80/20 sui soggetti (p. 11).
- I MOS sono stati raccolti a risoluzione piena. Le colonne x0.1, x0.5, x0.75 delle Tab. 1-2 sembrano riusare gli stessi MOS su mesh sotto-campionate.
- Le baseline non sono riaddestrate ne' calibrate con la SVR.
- Nessun codice o dato e' dichiarato disponibile.

## 4. Topologie, espressioni, fuori dominio

- **Topologie/connettivita'**: si, per costruzione. La rappresentazione e' un render, e il paper rivendica robustezza a "mesh topology" (Intro, Sez. 2.1 e 4.4.1).
  - La prova empirica e' il solo down-sampling dei vertici (x0.75, x0.5, x0.1, Tab. 1-2) piu' mesh native di risoluzione e connettivita' diverse nei dataset (35k-100k vertici).
  - Non sono testati remeshing, quad-mesh, mesh rumorose, diversa tassellazione o topologie con buchi.
  - La robustezza dichiarata e' quindi limitata alla risoluzione.
- **Espressioni**: i dataset includono espressioni (BU-3DFE, FRGC, Bosphorus 35 espressioni). GS-SIM e' descritta come "invariante a espressioni e pose" (p. 9).
  - Non c'e' una valutazione dedicata e controllata dell'invarianza all'espressione su coppie MOS.
  - Il riconoscimento rank-1 su scansioni con espressione (Tab. A3) e' una prova indiretta.
- **Fuori dominio**: solo scansioni reali ad alta risoluzione e una applicazione di fitting 3DMM su ICT-3DRFE (Tab. A1-A2). Non c'e' 3DMM sintetico come dato di training, ne' ricostruzioni da immagine (NoW, ecc.) con ground truth.
- **Frontalizzazione**: dipende da un rilevatore di landmark 2D e da un voto di maggioranza. Pipeline fragile su facce non standard.

## 5. Cosa possiamo riusare

- **Protocollo dello studio umano** (Sez. 4.2, p. 13): e' il riferimento piu' diretto.
  - ITU-R BT.500 / BT.2021, Likert a 5 punti, istruzioni "ignora shading/luce", 25 non-esperti.
  - Sessione di training prima del test, controllo dello scarto per coppia (sd < 0.15), Friedman + Wilcoxon, MOS normalizzato in [0, 1].
  - Il modello di report e' PLCC e SRCC fra metrica e MOS.
  - Idea utile: usare rotazione libera come condizione di riferimento "multi-vista".
  - Da migliorare: piu' coppie, reliability inter-rater (ICC o split-half), controllo di coppie ripetute, IC bootstrap.
- **Dataset**: TEC 3D-Twins (gemelli, ottimo per somiglianza fine), BU-3DFE, FRGCv2, Bosphorus, CASIA, ICT-3DRFE. Il loro dataset proprio e i MOS non sono dichiarati pubblici. Va chiesto agli autori.
- **Baseline da citare e implementare**: Besnier et al. 2023 [7] (misura appresa invariante alla mesh, Computers & Graphics 115) e Ma 2021 [43].
  - Confronto utile anche con **3D-PSSIM** [36] dello stesso gruppo (Lee et al., IEEE TPAMI 46(12) 2024, 9595-9611), una metrica di qualita' mesh basata su proiezioni, robusta a irregolarita' topologiche.
  - A questo paper manca il confronto con baseline percettive 2D (ArcFace, LPIPS, CLIP, DINOv2). Possiamo includerle noi per differenziarci.
- **Metriche**: PLCC e SRCC rispetto al MOS. Rank-1 identification con scansioni.
- **Ablation di rappresentazione** (Tab. 7): depth e normal insieme battono ciascuna da sola. Utile come ipotesi per i nostri render.

## 6. Rischio di sovrapposizione e differenziazione

Sovrapposizione concettuale **alta**. Lo stesso problema: una metrica di identita'/somiglianza fra mesh facciali indipendente da topologia e risoluzione, appresa con triplet loss su identita', validata contro giudizi umani.

Differenziazione possibile (nostro vantaggio):
1. **Operano su render**, noi su mesh nativa con DiffusionNet. Nessun vincolo di frontalizzazione, nessun rilevatore di landmark, nessun render. Potenzialmente invarianza reale a connettivita' e non solo a risoluzione. Va dimostrato con remeshing diversi, non solo down-sampling.
2. **Supervisione**: loro usano identita' reali di scansioni (limitate a circa 1000 soggetti totali). Noi usiamo 3DMM sintetici (BFM, ICT) con molte identita' e controllo diretto sui parametri. Ma e' proprio il punto che i revisori NeurIPS hanno attaccato (circolarita' della GT), quindi il confronto con l'approccio di Lee, che usa GT umana, e' obbligato.
3. **Baseline**: loro non confrontano con metriche percettive su render. Noi dobbiamo includere ArcFace, LPIPS, CLIP e DINOv2 su render, cosa che chiudono le critiche dei revisori e che questo paper non fa.
4. **Valutazione su ricostruzioni reali** (NoW o simili): assente nel loro lavoro. Fuori dominio e ricostruzioni da immagine sono il nostro margine.
5. **Studio umano**: loro hanno 200 coppie, 25 partecipanti, una sola GT umana per la SVR. Possiamo proporre uno studio piu' ampio e riportare affidabilita' inter-rater, e usarlo come validazione e non come training (evita la loro circolarita' tra SVR e MOS).
6. **Calibrazione**: la loro SVR e' calibrata sul MOS, quindi il punteggio non e' una pura metrica di identita'. La nostra e' una metrica diretta senza regressore.

Raccomandazioni di scrittura:
- Citarli esplicitamente come lavoro piu' vicino. Non ignorarli. Un revisore TOMM o CVPR li conosce.
- Non rivendicare di essere i primi a fare una metrica appresa di somiglianza 3D con studio umano.
- Differenziarsi su: nessuna proiezione, invarianza a connettivita' (non solo a risoluzione), scala dei dati sintetici, GT umana usata solo per valutazione, valutazione su ricostruzioni reali.

## 7. BibTeX

```bibtex
@article{lee2025facial,
  author    = {Lee, Seongmin and Kang, Jiwoo and Lee, Sanghoon},
  title     = {{3D} Facial Shape Similarity with Deep Perceptual Representations},
  journal   = {ACM Transactions on Multimedia Computing, Communications, and Applications},
  volume    = {21},
  number    = {6},
  articleno = {183},
  numpages  = {27},
  month     = jul,
  year      = {2025},
  doi       = {10.1145/3734874},
  url       = {https://doi.org/10.1145/3734874}
}
```

Nota: il PDF indica "Seongmin Lee e Jiwoo Kang contribuiscono ugualmente". Le email e le affiliazioni sono a p. 1 (Hanbat National University, Sookmyung Women's University, Yonsei University). Ricevuto 13 dicembre 2024, rivisto 12 marzo 2025, accettato 1 maggio 2025. Licenza CC BY 4.0.
