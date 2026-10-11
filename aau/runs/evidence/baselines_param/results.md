# Concorrenti parametrici (GNM, FLAME 2023 Open) e varifold: risultati

Protocollo `PROTOCOL.md` (sha256 `0d7444e622c7295fe620467e5e0c136dd0f06d348e35de16de09a65e8566cda1`), codice `aau/baselines_param/`. Righe, GT e repliche di `v3_work/trainer/tools/fact_paired.py` (importato, non modificato). IC 95% percentile, 1000 repliche per soggetto. Tutti i numeri in `spearman.csv` (valori) e `paired.csv` (valori e delta, anche parziale e quintile basso).

## Lettura (regola della sez. 7 del protocollo, scritta dopo i numeri)

Job: fit 1067569 (GNM, 4 min 22 s) e 1067570 (FLAME, 20 min 40 s), 48 core CPU ciascuno; varifold 1067571-1067574
(una L40S per vista, da 17 min 38 s a 29 min 44 s); controllo GPU/CPU 1067582; delta 1067575 (15 min 31 s, 32 core).
FaMoS (15 mesh): fit e varifold eseguiti nella prova funzionale, prima del protocollo, senza GT (vedi PROTOCOL.md).
Le correzioni di questa lettura e le varianti A e B vengono dall'emendamento 1 (POST HOC, `PROTOCOL_emendamento_1.md`,
in fondo); crop, composizione, costi e sensibilita' a sigma dall'emendamento 2 (POST HOC, `PROTOCOL_emendamento_2.md`).

- **Fit: nessun fit numericamente fallito su 4.030** (2 modelli x (4 viste x 500 + 15); fallito = eccezione, valori
  non finiti o RMS sui punti registrati dal NICP > 5 mm); RMS sui punti registrati <= 0.84 mm. Questo NON misura
  l'aderenza alla superficie di ingresso: l'errore punto-superficie bidirezionale dei fit originali (emendamento 1,
  sez. 3) ha mediana 1.0-1.6 mm e p95 3.0-5.2 mm per vista e modello (p95 fino a 9.8 mm sulla singola mesh). Nessuna
  riga persa: la maschera comune coincide con quella di fact_paired e i bracci ridanno `factorized_paired.csv` (scarto
  massimo 5.6e-17 su 480 valori, stesse righe). Vista neutra: i punti dei bracci coincidono con
  `faceverse_neutral/graded.csv`.
- **Delta braccio - concorrente** (factorized s1234/s2345 d_F cal. e ctrlfr con FR; factorized d_P e ctrlfr con SR; 7
  concorrenti; 28 delta per dominio e GT): **nessun IC sotto 0 in nessun dominio**. A favore del braccio, su 28: HIFI3D
  FR 28, SR 24; dev FaceScape FR 28, SR 26; FaceVerse con espressioni FR 9, SR 17; FaceVerse neutra FR 16, SR 23; FaMoS
  FR 11, SR 7. I restanti sono non risolti.
- **Perimetro esatto di questo esito** (fit preregistrati): vale per factorized con d_F calibrata (letta con FR) o d_P
  (letta con SR) e per ctrlfr (||z||, senza calibrazione, con FR e SR). Non vale per le letture non dichiarate: con
  d_F NON calibrata factorized e' pari a FLAME mesh d'identita' FR su HIFI3D FR (s1234 +0.003 [-0.073, +0.081], s2345
  -0.008 [-0.088, +0.074]); C3M con d_F non calibrata perde su HIFI3D FR contro FLAME mesh FR (e123 -0.111 [-0.218,
  -0.009], e205 -0.131 [-0.236, -0.026]) e, e205, contro GNM coefficienti (-0.111 [-0.220, -0.003]); d_P letta con FR
  e u di dual perdono su HIFI3D FR contro piu' concorrenti. **Con la variante B dell'emendamento 1 (modello nel ciclo)
  l'esito non regge piu' in tutte le celle: vedi sotto.**
- I non risolti su HIFI3D e FaceScape con SR sono ctrlfr contro le mesh d'identita' SR (GNM e FLAME): con SR solo d_P di
  factorized supera tutti i concorrenti parametrici con IC sopra 0. Su FaceVerse (entrambe le viste) il varifold non e'
  risolto contro nessun braccio (FR 0.260 / 0.285, contro 0.26-0.37 dei bracci), e GNM con espressioni nemmeno contro
  la maggior parte. FaMoS ha 15 soggetti e IC larghi: ctrlfr batte quasi tutti i concorrenti con FR, factorized nessuno
  con IC sopra 0 (varifold FR 0.688, factorized 0.678 / 0.654).
- GNM (prior VISTO) e' davanti a FLAME (prior non visto) in quasi tutte le celle (es. HIFI3D FR coefficienti 0.619
  contro 0.566, FaceScape SR mesh d'identita' 0.576 contro 0.469); con FR sulle mesh d'identita' FLAME e' davanti su
  HIFI3D (0.639 contro 0.608). Su FaMoS FLAME non supera GNM (FR: coefficienti 0.457 contro 0.624, mesh d'identita' FR
  0.565 contro 0.586). L'avvertenza 3 della sez. 2 del protocollo ("FLAME su FaMoS quasi oracolo, limite superiore")
  e' ritirata dall'emendamento 1: i numeri la smentiscono.
- Il fit parametrico preregistrato non migliora sul NICP su template da cui parte (HIFI3D FR: GNM mesh d'identita' FR
  0.608, FLAME 0.639, NICP su template in mm 0.614; FaceScape FR 0.465 / 0.449 contro 0.542). La variante B si'
  (sotto).
- **Varifold, controllo GPU/CPU** (job 1067582, `bp_varifold.py --check 5` su `hifi3d`, L40S contro CPU, stesso codice
  float32): scarto relativo massimo per coppia dei prodotti interni da 3.0e-4 a 6.2e-4, **sistematico** (in tutte le
  15 celle, 5 coppie x 3 sigma, il valore GPU e' maggiore). E' piu' grande del <= 1e-4 del pilota del protocollo (che
  confrontava float32 con float64); l'effetto sullo Spearman del varifold non e' misurato.

## Emendamento 1 (POST HOC): lettura

Protocollo `PROTOCOL_emendamento_1.md` (sez. 1-7 commit 3595f0a, sha256 `63569a41...`; sez. 8 con i valori congelati
commit 6e09491, sha256 `3d06ab08...`), scritto dopo i numeri sopra, sui rilievi del critic. Job: pilota 1067652 (GNM,
4 min 37 s) e 1067653 (FLAME, 13 min 47 s); viste valutate 1067685 (GNM, 6 min 34 s) e 1067686 (FLAME, 18 min 8 s),
40 core CPU; delta 1067687 (12 min 54 s, 32 core). B con entrambi i modelli (FLAME su FaceVerse con espressioni: 82 s
per mesh, sotto la soglia delle 2 ore).

- **Controlli.** NICP ricalcolato identico (scarto 0 su tutte le 4.030 mesh); FLAME su FaceVerse con espressioni,
  variante A = fit originale (max |delta beta| 5e-12); `paired_e1.csv` ridà tutte le 4.877 righe di `paired.csv`
  (scarto <= 2.2e-16, stesse righe). Nessun fit numericamente fallito in A ne' in B (0 / 4.030 ciascuna);
  corrispondenze tenute in B >= 0.398 per verso (soglia 0.25).
- **Errore di superficie** (mediana sulle mesh della mediana per mesh; p95): originale 1.0-1.6 mm (p95 3.0-5.2 mm),
  variante A uguale entro 0.05 mm, **variante B 0.29-0.56 mm (p95 0.96-2.0 mm, massimo 3.6 mm)**. Il fit a due stadi
  del protocollo aderiva male alla superficie; B corregge. Il verso ingresso -> M e' in parte in-sample: usa gli
  stessi 4096 punti della mesh su cui B costruisce le corrispondenze (il verso M -> ingresso usa tutti i vertici della
  regione); per B l'errore di superficie e' quindi un po' ottimista (correzione dell'emendamento 2).
- **Variante A (senza espressione sulle viste neutre)**: nessuna cella cambia segno. A favore del braccio, su 24 (4
  bracci x 6 colonne): HIFI3D FR 24, SR 20; FaceScape FR 24, SR 22; FaceVerse con espressioni FR 9, SR 17; FaceVerse
  neutra FR 19, SR 22; FaMoS FR 10, SR 9; nessuna sotto 0. Con le letture corrispondenti (coefficienti, `_fr` con FR,
  `_sr` con SR) gli Spearman cambiano al piu' di 0.04 (es. HIFI3D FR GNM coefficienti 0.619 -> 0.603, FLAME mesh FR
  0.639 -> 0.641), tranne FaMoS GNM mesh SR con SR (0.547 -> 0.439, 15 soggetti); la lettura non corrispondente mesh
  SR con FR cala di piu' (HIFI3D GNM 0.482 -> 0.291). Togliere l'espressione aumenta |beta| (GNM su HIFI3D da 4.6 a 7.0 di
  mediana: l'espressione assorbiva identita', come detto dal critic) ma non cambia il ranking.
- **Variante B (modello nel ciclo): 21 celle dichiarate su 240 passano a IC sotto 0** (concorrente davanti al
  braccio). A favore / contro / non risolte, su 24 per dominio e GT: HIFI3D FR 24 / 0 / 0, SR 8 / 8 / 8; FaceScape FR
  8 / 4 / 12, SR 8 / 8 / 8; FaceVerse con espressioni FR 3 / 0 / 21, SR 4 / 0 / 20; FaceVerse neutra FR 1 / 0 / 23, SR
  2 / 1 / 21; FaMoS FR 7 / 0 / 17, SR 2 / 0 / 22. Le celle contro:
  - HIFI3D SR, ctrlfr (entrambi i semi) contro GNM e FLAME vB coefficienti e mesh SR (8): GNM vB coefficienti SR 0.674
    [0.610, 0.734] contro ctrlfr 0.339 / 0.348 (delta fino a -0.336 [-0.434, -0.241]). **Tutte e 8 sono contro ctrlfr**,
    che con SR perde gia' contro baseline geometriche (`factorized_paired.csv`, stesse righe): NICP per coppia cs 0.595
    (delta s1234 -0.256 [-0.349, -0.172], s2345 -0.247 [-0.338, -0.165]) e ICP + Chamfer cs 0.580 (-0.242 [-0.330,
    -0.162], -0.233 [-0.314, -0.155]): ctrlfr non e' un braccio per SR. Il braccio per SR, factorized d_P (0.622 /
    0.613), resta NON risolto contro B;
  - FaceScape FR, factorized d_F cal. e ctrlfr (entrambi i semi) contro GNM vB mesh SR (4): 0.767 [0.675, 0.843]
    contro 0.653-0.669 (delta da -0.098 a -0.114);
  - FaceScape SR, factorized d_P e ctrlfr contro GNM e FLAME vB mesh SR (8): GNM 0.854 [0.803, 0.894], FLAME 0.820
    [0.757, 0.870] contro d_P 0.747 / 0.753 (delta -0.107 [-0.155, -0.065] ... -0.067 [-0.120, -0.014]) e ctrlfr
    0.619 / 0.627. Celle solide: tutti i bracci, entrambi i semi, entrambi i modelli;
  - FaceVerse neutra SR, ctrlfr s2345 contro GNM vB mesh SR (1): -0.093 [-0.181, -0.011]. Cella fragile: 1 su 240,
    estremo dell'IC a -0.011, nessuna correzione per confronti multipli (con 240 delta descrittivi qualche cella
    isolata oltre lo 0 e' attesa anche senza effetto); ctrlfr s1234 contro la stessa colonna: -0.073 [-0.156, +0.007].
- **Cosa regge contro tutti i concorrenti, B compreso**: HIFI3D con FR, 24 / 24 a favore (factorized d_F cal. 0.749 /
  0.731, ctrlfr 0.757 / 0.746, contro il migliore di B, FLAME vB mesh FR, 0.686 [0.597, 0.758]). FaceVerse (entrambe le
  viste) e FaMoS: nessuna cella sotto 0 tranne quella di FaceVerse neutra, ma quasi tutte non risolte contro B (es.
  FaMoS SR: FLAME vB mesh SR 0.791, d_P 0.740 / 0.728; FaMoS FR: ctrlfr 0.831 / 0.818 contro GNM vB mesh FR 0.686).
- **Lettura.** La frase "nessun concorrente batte i bracci dichiarati" vale solo per il fit preregistrato (a due stadi,
  aderenza 1-1.6 mm). Con un fit che mette il modello nel ciclo (iperparametri scelti su soggetti non valutati, senza
  GT) un 3DMM statistico eguaglia o batte i bracci con SR su HIFI3D (contro ctrlfr) e su FaceScape (contro tutti), e
  con FR su FaceScape (contro tutti); vale per GNM (prior VISTO) e anche per FLAME 2023 Open (prior non visto) con SR
  su FaceScape. Contro letture non dichiarate B vince in piu' celle (C3M d_F cal. / d_P su FaceScape: 5 celle sotto
  0). Tutto cio' e' post hoc e va riportato come tale, accanto all'analisi preregistrata.
- **Perche' FaceScape (correzione dell'emendamento 2).** Ritirata la spiegazione "FaceScape e' generato da un modello
  bilineare: un 3DMM ben fittato vi ricostruisce la forma": non distingue FaceScape, perche' anche HIFI3D e FaceVerse
  sono campioni di 3DMM e B li ricostruisce altrettanto bene (errore di superficie mediano 0.36-0.37 mm su HIFI3D,
  0.45-0.56 mm su FaceVerse, 0.29-0.31 mm su FaceScape), eppure li' la forma di B ordina molto peggio (GNM vB mesh SR
  con SR sulle sole coppie original - original: 0.856 FaceScape, 0.611 HIFI3D, 0.402 FaceVerse neutra; i valori del
  critic, ricalcolati in `controls_e2.json`). Che B vinca con SR su FaceScape e' un dato, non spiegato qui. Che la
  vittoria passi anche a FR si legge con la taglia: su FaceScape la taglia oracolo varia poco (CV di S sui 100
  soggetti valutati 1.6%, contro 4.7% su HIFI3D) e la GT FR ordina quasi come la SR (Spearman FR-SR 0.888 sulle 4.950
  coppie di soggetti, contro 0.628 su HIFI3D; il critic riportava 4.9% e 0.912 / 0.582), quindi chi ordina bene la
  forma ordina bene anche FR. Non basta da sola: FaceVerse ha taglia altrettanto stretta (CV 1.8%, FR-SR 0.954) e li'
  B non vince. **FaceScape e' il dominio di sviluppo dei bracci** (dev, vantaggio di casa): la vittoria di B li' pesa
  di piu', non di meno.
- **Controllo del critic** (ricalcolato nell'emendamento 2, `controls_e2.json`): la formula d_F = sqrt((S_i - S_j)^2 +
  S_i S_j d_P^2) con la S oracolo e la d_P della GT ordina come la GT FR con rho 0.997-1.000 sulle coppie di soggetti
  dei cinque domini (il limite per una ricostruzione perfetta; il critic riportava 1: la GT FR usa la rigida robusta,
  da cui lo scarto).
- Limiti: le scelte di B stanno sul bordo superiore della griglia di sigma (2 mm); sul pilota B migliorava FaceScape e
  peggiorava FaceVerse con espressioni (sez. 8 dell'emendamento); nessuna correzione per confronti multipli.

## Emendamento 2 (POST HOC): lettura

Protocollo `PROTOCOL_emendamento_2.md` (sez. 1-8 commit 05eceb0, sha256 `0bd93adb...`; sez. 9 coi valori senza GT di
test commit 26a6d5d, sha256 `91a0fc53...`). Job: crop 1067773, held-out 1067774, pilota 1067775 / 1067776,
sensibilita' 1067787 / 1067788 (CPU, 48 core), tempi di B 1067777 / 1067783, operatori dei bracci 1067778 / 1067782,
coppie 1067779, forward 1067780 (una L40S), delta 1067791 (48 core, 29 min). Tutto e' post hoc rispetto all'analisi
preregistrata e all'emendamento 1, e si legge come sensibilita'.

- **Controlli.** Tutti passati: le 11.933 righe di `paired_e1.csv` ridate (scarto <= 2.2e-16, stesse righe); le
  matrici di B sulle 600 mesh coincidono con `fit_e1.npz` sulle coppie senza crop (<= 2.5e-10); tolto il crop, le
  righe di all_cross hanno le stesse chiavi e GT di `fact_paired.rows_for` (99.000 per vista, scarto 0); pilota: S
  del fit originale e di A identici a `pilot_e1`. 0 fit falliti su 800 crop, 3.000 held-out e 4.030 della
  sensibilita'. **Non bit a bit**: rifatti su nodi AMD, i beta di B differiscono da `fit_e1.npz` (nodi Intel) fino a
  8.8e-3 (lo stesso valore su due nodi AMD); gli embedding dei bracci ricalcolati per i tempi differiscono da quelli
  di fact_paired fino a 1.1e-3.
- **Crop e all_cross (sez. 1): B regge il ritaglio, i bracci no.** Le colonne di B cambiano poco passando dalle righe
  senza crop a all_cross e alle sole righe col crop (HIFI3D FLAME vB mesh FR 0.686 -> 0.670 -> 0.639; FaceScape GNM
  vB mesh SR con SR 0.854 -> 0.840 -> 0.830; errore di superficie di B sul crop come sulle altre topologie, 0.29-0.58
  mm). I bracci calano molto: FaceScape factorized d_F cal. con FR 0.661 / 0.669 -> 0.370 / 0.351, d_P con SR 0.747 /
  0.753 -> 0.489 / 0.464, ctrlfr con FR 0.661 / 0.653 -> 0.334 / 0.323; HIFI3D ctrlfr con FR 0.757 / 0.746 -> 0.516 /
  0.530 (sulle righe col crop 0.236 / 0.279), factorized d_F cal. 0.749 / 0.731 -> 0.709 / 0.661. Delta dichiarati
  (24 per dominio e GT, a favore / contro / non risolte), all_cross: HIFI3D FR 12 / 4 / 8, SR 5 / 10 / 9; FaceScape
  FR 0 / 24 / 0, SR 0 / 24 / 0; FaceVerse FR 1 / 0 / 23, SR 1 / 0 / 23; FaceVerse neutra FR 0 / 0 / 24, SR 0 / 1 / 23
  (senza crop, emendamento 1: HIFI3D FR 24 / 0 / 0, SR 8 / 8 / 8; FaceScape FR 8 / 4 / 12, SR 8 / 8 / 8). Le 4 celle
  HIFI3D FR contro sono ctrlfr contro le mesh d'identita' FR di B (fino a -0.154 [-0.197, -0.114]); su HIFI3D SR,
  oltre a ctrlfr, ora anche factorized d_P perde contro GNM vB coefficienti (s1234 -0.108 [-0.193, -0.022], s2345
  -0.124 [-0.216, -0.032]). Sulle sole righe col crop: HIFI3D FR 6 / 14 / 4 (anche factorized s2345 d_F cal. sotto 0
  contro le mesh FR di B), SR 4 / 10 / 10; FaceScape FR 0 / 24 / 0, SR 0 / 19 / 5; FaceVerse neutra FR 0 / 3 / 21, SR
  0 / 4 / 20 (7 celle contro, tutte contro GNM vB mesh SR; su all_cross 1). Su FaceVerse con espressioni quasi tutto
  resta non risolto (conteggi completati dall'emendamento 3, `controls_e3.json` `counts_e2`).
- **Perche' i bracci cadono sul crop** (diagnostica aggiunta dopo i numeri, non nel protocollo; tabella in
  `results.md`): il crop sposta in modo sistematico l'embedding dei bracci. Su FaceScape factorized stima per il crop
  una taglia piu' piccola (log S -0.061 / -0.070, cioe' circa 4 volte la deviazione standard di log S fra soggetti,
  0.016-0.017) e la sua d_P crop -> stesso soggetto (0.68 / 0.71) supera la mediana fra soggetti diversi (0.55); ctrlfr
  su FaceScape 1.43 contro 0.79, su HIFI3D 1.24 contro 1.05. Su HIFI3D lo spostamento di log S (-0.024 / -0.034) e'
  piccolo rispetto alla varianza della taglia (0.047): per questo factorized regge meglio li'. La GT e B sono
  definite sulla regione comune, che il crop non tocca; i bracci vedono l'area della mesh (normalizzazione globale).
  (Correzione dell'emendamento 3: il crop E' nella distribuzione di training, con perdita d'area comparabile; sugli
  held-out sintetici i bracci vi sono invarianti su ICT e GNM, parzialmente su BFM; il calo sui test viene dalla banda
  di bordo fuori dalla regione di B: vedi sotto.)
- **Composizione forma B + taglia B (sez. 2)**, k dagli held-out sintetici senza test (GNM 1.698, FLAME 1.610), lettura
  dichiarata con FR: HIFI3D GNM 0.698 [0.612, 0.769], FLAME 0.712 [0.627, 0.781] (meglio della migliore colonna FR di
  B, 0.686), contro factorized d_F cal. 0.749 / 0.731 e ctrlfr 0.757 / 0.746: 4 a favore dei bracci, 4 non risolte,
  nessuna contro. FaceScape GNM 0.800 [0.741, 0.847], FLAME 0.731 [0.657, 0.794]: **8 / 8 contro i bracci** (GNM da
  -0.131 a -0.147, FLAME da -0.062 [-0.123, -0.009] a -0.078). FaceVerse (0.338 / 0.302), FaceVerse neutra (0.382 /
  0.367), FaMoS (0.753 / 0.708, 15 soggetti): tutte non risolte. Controllo della formula con le GT: rho con FR
  0.997-1.000.
- **Sensibilita' a sigma (sez. 4)**: il pilota sceglie sigma 8 per entrambi i modelli (di nuovo sul bordo della
  griglia; S cala in modo monotono con sigma). Con B a sigma 8 il quadro dell'emendamento 1 non cambia nel segno:
  24 celle sotto 0 su 240 (21 con la B congelata), stesse aree: HIFI3D SR contro ctrlfr (8), FaceScape FR (4) e SR
  (10), una cella SR su ciascuna vista FaceVerse (ctrlfr s2345 contro GNM mesh SR, -0.081 [-0.158, -0.006] e -0.093
  [-0.183, -0.006]). HIFI3D FR resta dei bracci (23 / 0 / 1; migliore B, FLAME mesh FR, 0.694). Su FaceScape B
  migliora un poco (GNM mesh SR con SR 0.873 [0.829, 0.908] contro 0.854).
- **Costi misurati (sez. 3; tabelle in `results.md`).** Per iscrivere una mesh, mediana su CPU a un thread con il
  nodo pieno di processi (come in produzione): **B GNM 4.5-5.0 s** sulle topologie senza crop (NICP 3.0-3.4 s, A
  0.3-0.4 s, B 1.0-1.3 s; crop 5.4-6.2 s, FaMoS 2.6 s; con espressioni libere su FaceVerse 9.5 s), **B FLAME 1.6-2.0
  s** (crop 2.0-2.4 s; con espressioni e mandibola libera su FaceVerse
  85 s: A 48 s e B 37 s per la ricerca della mandibola); **bracci 4.8 s**, quasi tutto negli operatori DiffusionNet
  k128 su CPU (mediana 4.77 s, p95 14.7 s, up60k 14.3 s; il resto e' 42 ms su L40S: lettura dell'npz 34 ms, forward
  6.5 ms in batch 1, 5.6 ms per mesh in gruppi di 30). Con un processo solo sul nodo: operatori 2.97 s (up60k 9.0 s),
  B GNM 2.9 s, B FLAME 1.0 s (HIFI3D). Il vecchio "~24 s per mesh" per gli operatori non e' confermato: misurati 3-5 s
  di mediana. Confronto di una coppia: bracci 5-7 us (0.03 us ammortizzati su tutte le coppie), B coefficienti 4.6 us,
  mesh d'identita' e composizione 20-74 us (4-17 us ammortizzati) piu' 0.8-3 ms per mesh una volta (rigida verso mu).
  Il costo non separa i metodi: entrambi qualche secondo per mesh, dominati da un passo geometrico su CPU (NICP per B,
  autodecomposizione del Laplaciano per i bracci). Avvertenze: ops e tempi di B paralleli sullo stesso nodo nello
  stesso momento (a512-l4-06, AMD EPYC 7543), quelli a un processo su a256-t4-02 (AMD EPYC 7302); tempi di B senza
  l'errore di superficie.
- **Lettura.** Rispetto all'emendamento 1 il confronto cambia in due punti. (1) Con il crop nelle righe (all_cross)
  i bracci perdono gran parte del vantaggio su HIFI3D FR (ctrlfr scende sotto le mesh FR di B; factorized s1234 resta
  davanti a tutte le colonne di B, contro le mesh FR +0.058 [+0.019, +0.100] GNM e +0.039 [+0.004, +0.075] FLAME,
  s2345 e' pari alle mesh FR di B) e su FaceScape perdono contro B in tutte le 48 celle dichiarate: il gruppo primario senza
  crop nascondeva una fragilita' dei bracci al ritaglio, che B non ha (correzione dell'emendamento 3: fragilita' gia'
  nota da E1, dal dev FaceScape e da `paper/REPORT.md`, e in gran parte dovuta al supporto diverso: vedi sotto). (2)
  Una composizione dichiarata forma B +
  taglia B, con k fissato senza test, non batte i bracci su HIFI3D FR (4 a favore dei bracci, 4 non risolte) ma li
  batte su FaceScape. Quello che regge per i bracci contro tutto quanto provato: HIFI3D FR senza crop (contro B
  congelata, B a sigma 8 e la composizione). FaceScape e' il dominio di sviluppo dei bracci: le sconfitte li' pesano di
  piu'. Nessuna correzione per confronti multipli; la scelta di sigma resta sul bordo della griglia.

## Emendamento 3 (POST HOC): lettura

Protocollo `PROTOCOL_emendamento_3.md` (sez. 1-5 commit acaf0ab, sha256 `1c4331dc...`; sez. 6, aggiunta del critic del
paper, commit 6d23536, sha256 `c52cd1e4...`), scritto dopo i numeri dell'emendamento 2 sul verdetto RISERVE del critic.
Job: ritagli, operatori ed embedding dell'esperimento 1 1067973 (4 A100 di nv-ai-04, QoS unprivileged; cancellato a
embedding dell'esperimento 1 finiti, 46 / 46); held-out 1068029 (4 A100); delta 1068051 (64 core, 19 min); sez. 6
1067960; controllo della catena 1067965. Il primo tentativo 1067823 e' stato cancellato: con 4 processi di embedding a
thread OpenMP di default il preload non avanzava (34 min senza un embedding); poi 8 thread per processo (~1 min per 600
mesh da /tmp). Tutto e' post hoc e si legge come sensibilita' e diagnostica. **Lettura corretta dopo il critic
dell'emendamento 3** (RISERVE: numeri ricalcolati e confermati, quattro letture non reggevano: costo del ritaglio,
run massivo, recupero, esperimento 2). I numeri marcati "critic" vengono dai suoi controlli (stime puntuali, 12-100
soggetti, script non versionati); gli altri da `paired_e3.csv`, `heldout_e3.csv`, `e3/crop_stats.json`.

- **Controlli.** Passati: `bp.region` rieseguita ridà i vertici di `region.npz` (8 / 8); 0 ritagli falliti su 4.800;
  catena: factorized s1234 sugli operatori dello store di HIFI3D ridà lo store entro 4.8e-7; bracci interi e B su
  all_cross e righe col crop ridanno `paired_e2.csv` (496 valori, scarto 8e-17 su punti e IC, stesse righe); baseline
  geometriche su all_cross identiche a `baselines_mm/spearman.csv`; held-out: gli embedding delle 1.500 senza crop
  coincidono con quelli della calibrazione entro 8e-4 (factorized) e 4.6e-3 (C3M). **Deviazioni:** FaceVerse neutra
  non ha baseline geometriche ne' lo store C3M e123 (niente D2 li', C3M e123 saltato); per il resto come da protocollo.
- **Correzioni al testo dell'emendamento 2 (sez. 3).**
  1. *Il crop e' nella distribuzione di training* (al posto di "fuori distribuzione? non verificato"): C3F (REMESH
     500 / 500, ICT 5000 / 5000, `gnm|crop`) e C3M (`trainer_v3/factorized/scale_tables/c3m.json`) hanno il crop dello
     stesso generatore (`v2_work/genict/mesh_ops.py` `make_crop`, banda di bordo fissa), con perdita d'area comparabile
     a quella dei test (log del rapporto sqrt(area) crop / original, valori del critic: training BFM -0.080, ICT -0.104,
     GNM -0.117; test HIFI3D -0.108, FaceScape -0.131, FaceVerse -0.047; misurati qui sui test: -0.109, -0.132, -0.045).
     Anche il run massivo lo ha (`v3_work/stream/views.py`, `LABEL_WEIGHTS` crop 1.0). Sugli held-out sintetici
     l'invarianza al crop c'e' su ICT e GNM, parziale su BFM (esperimento 2, sotto).
  2. *Baseline geometriche su all_cross* (stesse 148.500 righe, `baselines_mm/spearman.csv`, ricalcolate identiche): FR
     FaceScape ICP + Chamfer mm 0.381 [0.326, 0.434], Chamfer pura 0.570 [0.499, 0.641]; HIFI3D 0.556 [0.488, 0.617] e
     0.645 [0.547, 0.725]. Con il crop i bracci interi arrivano al livello di ICP o sotto (FaceScape d_F cal. 0.370 /
     0.351, ctrlfr 0.334 / 0.323; HIFI3D ctrlfr 0.516 / 0.530; factorized s1234 0.709 resta sopra). Sulle righe col crop
     i bracci interi perdono contro la Chamfer pura: HIFI3D FR ctrlfr -0.408 / -0.365 (anche contro ICP mm -0.181 /
     -0.138) e factorized s2345 -0.091; FaceScape FR tutti e 4 (da -0.153 a -0.270), SR ctrlfr; FaceVerse FR e SR tutti
     e 4 (da -0.083 a -0.128).
  3. *C3M crolla anch'esso*: d_F cal. con FR, senza crop -> righe col crop, e205 HIFI3D 0.739 -> 0.472, FaceScape 0.712
     -> 0.449 (il critic: 0.739 -> 0.472, 0.711 -> 0.449), e123 0.748 -> 0.506 e 0.666 -> 0.472. Il run massivo ha lo
     stesso crop con peso 1.0, ma puo' usare piu' famiglie di generatori (e c'e' un run secondario con parzialita'
     variabile): se e quanto riducano il crollo va misurato, non si deduce da questi esperimenti.
  4. *La fragilita' era nota*: `aau/runs/evidence/e1/summary.md` (HIFI3D, GT maxabs, senza crop -> all_cross: C3F 0.716
     -> 0.587, C3M 0.677 -> 0.509), dev FaceScape (e108 0.642 -> 0.335 col crop, `dev_facescape/results.md`),
     `paper/REPORT.md` (legge di Weyl: il crop sposta gli autovalori del Laplaciano). L'emendamento 2 non l'ha scoperta:
     l'ha misurata contro B.
  5. *Asimmetria della regione*: B usa una regione per dominio stimata su 100 soggetti NON valutati con l'anello di
     bordo escluso (`bp.py` `region`) e scarta le corrispondenze sul bordo della regione; i bracci fanno pooling su tutta
     la superficie, bordo compreso. L'esperimento 1 la misura (sotto).
  6. *Pooling*: all_cross mescola coppie di topologie con scarti diversi e sovrastima un po' il calo; per il crop si
     leggono le righe col crop o la media dentro le coppie di topologie (gruppi c e d). Qui le due letture danno gli
     stessi conteggi entro 1-2 celle.
  Conteggi mancanti (`controls_e3.json` `counts_e2`): FaceScape SR sulle righe col crop 0 / 19 / 5; FaceVerse neutra
  sulle righe col crop FR 0 / 3 / 21, SR 0 / 4 / 20 (7 celle contro, tutte contro GNM vB mesh SR), su all_cross 1.
- **Ritaglio alla regione di B (diagnostica).** Copertura della regione da parte del crop 0.82-0.93 di mediana (altre
  topologie 0.95-0.996), sotto la soglia dichiarata di 0.95. Lo scarto viene quasi tutto dalla striscia di bordo di R
  (i vertici dei triangoli che toccano il bordo della regione): tolta quella, la copertura del crop sale a 0.99-1.00 con
  la regione FLAME e a 0.90-0.93 con la regione GNM (critic, 12 soggetti, HIFI3D e FaceScape). "B lavora gia' su dati
  parziali e regge" vale quindi poco con FLAME e solo per il 7-10% della regione interna con GNM. Dopo il ritaglio crop
  e original diventano quasi la stessa mesh: log sqrt(area crop / area original) da -0.009 a +0.001 (area entro
  0.1-2%), contro -0.045 / -0.132 prima. Il ritaglio ha dimensione fissa in mm: la regione e' posata con ICP rigido
  senza scala (`bp_e3.py`, `placed_region` e `_crop`), quindi comprime anche la variabilita' d'area fra soggetti (sd di
  log sqrt(area) delle original: HIFI3D 0.045 -> 0.010 GNM / 0.012 FLAME, FaceScape 0.015 -> 0.007 / 0.010; critic).
- **Esperimento 1: il recupero e' per costruzione.** A regione uguale ogni braccio vale sulle righe col crop quanto
  senza crop: nei 32 casi dichiarati con FR (4 viste x 4 bracci x 2 regioni) la differenza righe col crop - senza crop
  va da -0.070 a +0.045 (con SR da -0.050 a +0.043), e la distanza nell'embedding crop -> original dello stesso
  soggetto scende sotto quella noisy -> original (critic; HIFI3D factorized s1234: 0.42 col braccio intero, 0.15 / 0.05
  con le regioni GNM / FLAME, noisy 0.22-0.24). Il recupero F della sez. 1 misura quindi solo quanto il ritaglio costa o
  rende sulle righe senza crop, e le letture dichiarate ("dipendenza dal supporto" su FaceScape 8 / 8 con FR e con SR,
  su HIFI3D regione GNM FR 3 / 4 e SR 4 / 4, regione FLAME pari 2 / 2; FaceVerse non leggibile, calo < 0.05) sono
  meccaniche: non separano le due ipotesi del protocollo. Valori: FaceScape FR factorized s1234 sulle righe col crop
  0.351 -> 0.687 (GNM) / 0.664 (FLAME); HIFI3D ctrlfr s1234 0.236 -> 0.602 / 0.606; C3M (descrittivo) come i bracci.
  **Correzione:** non e' vero che factorized su HIFI3D "recupera solo con la regione GNM": con la regione FLAME e'
  invariante al crop (righe col crop - senza crop +0.008 / +0.009) e il suo F basso viene dal costo del ritaglio
  (-0.14 / -0.16 su tutte le righe). L'unico segnale informativo sul crop e' ctrlfr con la regione GNM su HIFI3D, che
  perde ancora -0.053 / -0.070 con circa il 2% d'area di differenza fra crop e original ritagliati.
  **Il costo del ritaglio su HIFI3D e' soprattutto di taglia.** Sulle righe senza crop il ritaglio costa a tutti i
  bracci con FR (D3 0 / 8 / 0, da -0.05 a -0.16, peggio con FLAME). La finestra fissa in mm comprime la variabilita'
  d'area fra soggetti, e i bracci ricavano la scala dall'area (`global_v3.frame_params`, fattore sqrt(area_mm2 /
  area)); B la conserva nell'identita'. Critic, HIFI3D, factorized s1234, righe senza crop: Spearman fra area della mesh
  e taglia oracolo 0.93 -> 0.68 (GNM) / 0.32 (FLAME); solo il termine di taglia di d_F contro FR 0.587 -> 0.488 /
  0.332; d_P contro FR 0.426 -> 0.528 / 0.512 (la forma non peggiora). Con la regione FLAME si perde anche forma (D3 con
  SR, factorized d_P, -0.141 / -0.142). Su FaceScape la FR dipende poco dalla taglia (solo il termine di taglia contro
  FR 0.145, d_P 0.677; critic) e il costo non c'e' (D3 senza crop FR 2 / 0 / 6, SR 3 / 0 / 5). Di conseguenza "a regione
  uguale" su HIFI3D e' sbilanciato contro i bracci: senza crop le loro celle FR contro B passano da 24 / 0 / 0 (bracci
  interi, emendamento 1) a 12 / 3 / 9.
  **Contro B a regione uguale, righe col crop** (D1, a favore / contro / non risolte): FaceScape FR 13 / 2 / 9
  (emendamento 2: 0 / 24 / 0; senza crop a regione uguale 8 / 2 / 14), SR 15 / 6 / 3 (0 / 19 / 5); HIFI3D FR 11 / 0 /
  13 (6 / 14 / 4), SR 8 / 10 / 6 (4 / 10 / 10). Restano per B: GNM vB mesh SR con FR su FaceScape contro factorized s1234
  @ GNM (-0.068 [-0.118, -0.018]) e ctrlfr s2345 @ GNM (-0.139); con SR le mesh SR di GNM e FLAME contro ctrlfr (4) e
  factorized (2, da -0.046 a -0.057) su FaceScape, e su HIFI3D contro ctrlfr (8, che non e' un braccio per SR) e
  factorized @ FLAME (-0.082 / -0.087): la stessa geografia delle celle B senza crop dell'emendamento 1. Contro le
  baseline geometriche (D2) sulle righe col crop: HIFI3D FR 16 / 0 / 8, SR 24 / 0 / 0; FaceScape 24 / 0 / 0; FaceVerse
  FR 7 / 7 / 10, SR 8 / 6 / 10.
  Nota del critic, descrittiva: con c raddoppiata (2c) factorized s1234 a ingresso intero su FaceScape sale sulle righe
  col crop da 0.351 a 0.488 (senza crop 0.659 -> 0.688); c resta quella degli held-out.
- **Esperimento 2: invarianza al crop in-distribuzione, ma "limite di generalizzazione" vale solo in parte.**
  Held-out sintetici (100 soggetti per dominio), coppie col crop contro senza crop. ICT e GNM: delta dello Spearman da
  -0.021 a +0.008 per tutti i bracci dichiarati (con FR e con SR), d log S del crop da -0.004 a +0.006 (<= 0.12 sd fra
  soggetti): invarianza presente secondo i criteri dichiarati. BFM: parziale per factorized (SR -0.019 / -0.026, IC
  sotto 0 ma |delta| < 0.05; d log S -0.023 / -0.019, cioe' 1.3 / 1.0 sd); ctrlfr presente. L'AUC di verifica e' a
  soffitto (0.999-1.000 con e senza crop) e non sostiene l'invarianza. Il confronto coi test (critic): factorized su
  HIFI3D si comporta come su BFM in-distribuzione (d log S del crop -0.024 / -0.034 contro -0.023 / -0.019; calo con FR
  -0.086 / -0.179 contro -0.094 / -0.068, BFM con FR descrittivo), e C3M crolla con FR anche su BFM in-distribuzione
  (-0.256 / -0.176). La sensibilita' al crop sui test supera quella in-distribuzione su FaceScape (calo dei bracci
  interi 0.23-0.42, d log S ~4 sd nell'emendamento 2) e per ctrlfr su HIFI3D (0.47-0.52), non per factorized su HIFI3D:
  solo li' si puo' parlare di limite di generalizzazione.
- **Sez. 6 (descrittiva, non cieca), Spearman dentro le coppie di topologie senza crop**, bracci interi, IC per
  soggetto col seme di all_cross. HIFI3D factorized s1234 d_F cal. con FR: media sulle 20 coppie ordinate 0.749 [0.675,
  0.811] (min 0.730 remesh -> up60k, max 0.760 up60k -> original), stessa topologia 0.760 [0.685, 0.821], differenza -0.011
  [-0.013, -0.009]; FaceScape: 0.667 [0.598, 0.732] (min 0.628, max 0.706), stessa topologia 0.706 [0.640, 0.769],
  differenza -0.039 [-0.047, -0.032]; le stime puntuali del critic sono ridate esattamente. Gli altri bracci: differenza
  20 - stessa topologia da -0.005 a -0.021 su HIFI3D, da -0.020 a -0.054 su FaceScape (ctrlfr s1234 il piu' alto), da
  -0.001 a -0.010 su FaceVerse (0.26-0.32 di livello). Lettura: senza crop la discretizzazione costa poco (fino a 0.02
  su HIFI3D, fino a 0.05 su FaceScape, sistematico: IC della differenza sotto 0 su HIFI3D e FaceScape); la topologia
  peggiore e' spesso noisy.
- **Lettura.** Il ribaltamento dell'emendamento 2 sulle righe col crop (FaceScape 48 / 48 celle contro i bracci) viene
  dalla banda di bordo fuori dalla regione di B, che B non guarda per costruzione. Ritagliato l'ingresso dei bracci alla
  regione di B (posta in modo rigido e senza scala), crop e original diventano quasi la stessa mesh e sulle righe col
  crop ogni braccio vale quanto senza crop: il recupero e' per costruzione, e l'esperimento 1 non separa "dipendenza dal
  supporto" da "limite del descrittore" (unico segnale: ctrlfr @ GNM su HIFI3D, -0.05 / -0.07 col 2% d'area). A regione
  uguale contro B con FR: FaceScape 13 / 2 / 9 col crop e 8 / 2 / 14 senza; HIFI3D 11 / 0 / 13 col crop e 12 / 3 / 9
  senza (bracci interi 6 / 14 / 4 e 24 / 0 / 0): su HIFI3D il confronto a regione uguale e' sbilanciato contro i bracci,
  perche' la finestra fissa toglie la variabilita' d'area da cui ricavano la scala. Held-out: invarianza su ICT e GNM,
  parziale su BFM, dello stesso ordine di factorized su HIFI3D; sui test la sensibilita' al crop supera quella
  in-distribuzione su FaceScape e per ctrlfr su HIFI3D. Se il run massivo (stesso crop, piu' famiglie di generatori) la
  riduca va misurato. Tutto post hoc; nessuna correzione per confronti multipli.

## Fit

| vista | modello | mesh | fallite (numericamente) | RMS fit mm, mediana (max) | resid. NICP mm, mediana | s/mesh, mediana | vertici della regione |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | gnm | 500 | 0 (0.0%) | 0.10 (0.22) | 2.07 | 4.3 | 7700 |
| hifi3d | flame2023 | 500 | 0 (0.0%) | 0.27 (0.54) | 2.13 | 26.0 | 1517 |
| facescape | gnm | 500 | 0 (0.0%) | 0.12 (0.26) | 2.16 | 4.7 | 8061 |
| facescape | flame2023 | 500 | 0 (0.0%) | 0.32 (0.63) | 2.21 | 27.1 | 1544 |
| faceverse | gnm | 500 | 0 (0.0%) | 0.19 (0.44) | 2.54 | 4.7 | 8654 |
| faceverse | flame2023 | 500 | 0 (0.0%) | 0.39 (0.84) | 2.56 | 28.6 | 1674 |
| faceverse_neutral | gnm | 500 | 0 (0.0%) | 0.18 (0.37) | 2.40 | 4.7 | 8654 |
| faceverse_neutral | flame2023 | 500 | 0 (0.0%) | 0.37 (0.84) | 2.44 | 29.1 | 1674 |
| famos | gnm | 15 | 0 (0.0%) | 0.09 (0.27) | 2.28 | 17.9 | 8631 |
| famos | flame2023 | 15 | 0 (0.0%) | 0.13 (0.37) | 2.21 | 18.1 | 1697 |

## Controlli

```
{
 "reference": "aau/runs/evidence/trainer_v3/factorized_paired.csv",
 "reference_mtime": 1791646483.4194698,
 "n_compared": 480,
 "max_abs_diff_arm_point": 5.551115123125783e-17,
 "rows_equal": true,
 "by_domain": {
  "facescape": {
   "n": 120,
   "max_abs_diff": 0.0,
   "n_rows": [
    88725
   ],
   "n_rows_ref": [
    88725
   ]
  },
  "faceverse": {
   "n": 120,
   "max_abs_diff": 5.551115123125783e-17,
   "n_rows": [
    99000
   ],
   "n_rows_ref": [
    99000
   ]
  },
  "famos": {
   "n": 120,
   "max_abs_diff": 5.551115123125783e-17,
   "n_rows": [
    105
   ],
   "n_rows_ref": [
    105
   ]
  },
  "hifi3d": {
   "n": 120,
   "max_abs_diff": 5.551115123125783e-17,
   "n_rows": [
    98224
   ],
   "n_rows_ref": [
    98224
   ]
  }
 }
}
{
 "hifi3d": {
  "rows_total": 99000,
  "rows_mask": 98224,
  "nan_rows_by_new_column": {
   "gnm_coef": 0,
   "gnm_fr": 0,
   "gnm_sr": 0,
   "flame2023_coef": 0,
   "flame2023_fr": 0,
   "flame2023_sr": 0,
   "varifold": 0
  },
  "n_arm_columns": 30,
  "baselines": [
   "mm_rigid_icp_chamfer",
   "mm_nicp_template",
   "est_cs",
   "oracle_size",
   "cs_nicp_p2tri",
   "cs_rigid_icp_chamfer",
   "comp_oracle",
   "comp_est",
   "comp_est_nicp",
   "comp_oracle_raw",
   "comp_est_raw",
   "scale_e108"
  ],
  "famos_controls": []
 },
 "facescape": {
  "rows_total": 99000,
  "rows_mask": 88725,
  "nan_rows_by_new_column": {
   "gnm_coef": 0,
   "gnm_fr": 0,
   "gnm_sr": 0,
   "flame2023_coef": 0,
   "flame2023_fr": 0,
   "flame2023_sr": 0,
   "varifold": 0
  },
  "n_arm_columns": 30,
  "baselines": [
   "mm_rigid_icp_chamfer",
   "mm_nicp_template",
   "est_cs",
   "oracle_size",
   "cs_nicp_p2tri",
   "cs_rigid_icp_chamfer",
   "comp_oracle",
   "comp_est",
   "comp_est_nicp",
   "comp_oracle_raw",
   "comp_est_raw",
   "scale_e108"
  ],
  "famos_controls": []
 },
 "faceverse": {
  "rows_total": 99000,
  "rows_mask": 99000,
  "nan_rows_by_new_column": {
   "gnm_coef": 0,
   "gnm_fr": 0,
   "gnm_sr": 0,
   "flame2023_coef": 0,
   "flame2023_fr": 0,
   "flame2023_sr": 0,
   "varifold": 0
  },
  "n_arm_columns": 30,
  "baselines": [
   "mm_rigid_icp_chamfer",
   "mm_nicp_template",
   "est_cs",
   "oracle_size",
   "cs_nicp_p2tri",
   "cs_rigid_icp_chamfer",
   "comp_oracle",
   "comp_est",
   "comp_est_nicp",
   "comp_oracle_raw",
   "comp_est_raw",
   "scale_e108"
  ],
  "famos_controls": []
 },
 "faceverse_neutral": {
  "rows_total": 99000,
  "rows_mask": 99000,
  "nan_rows_by_new_column": {
   "gnm_coef": 0,
   "gnm_fr": 0,
   "gnm_sr": 0,
   "flame2023_coef": 0,
   "flame2023_fr": 0,
   "flame2023_sr": 0,
   "varifold": 0
  },
  "n_arm_columns": 22,
  "baselines": [
   "oracle_size"
  ],
  "famos_controls": []
 },
 "famos": {
  "rows_total": 105,
  "rows_mask": 105,
  "nan_rows_by_new_column": {
   "gnm_coef": 0,
   "gnm_fr": 0,
   "gnm_sr": 0,
   "flame2023_coef": 0,
   "flame2023_fr": 0,
   "flame2023_sr": 0,
   "varifold": 0
  },
  "n_arm_columns": 30,
  "baselines": [
   "mm_rigid_icp_chamfer",
   "est_cs",
   "oracle_size",
   "cs_nicp_p2tri",
   "cs_rigid_icp_chamfer",
   "comp_oracle",
   "comp_est",
   "comp_est_nicp",
   "comp_oracle_raw",
   "comp_est_raw"
  ],
  "famos_controls": [
   "factorized_s1234|form: 0.0e+00",
   "factorized_s2345|form: 0.0e+00",
   "factorized2_s1234|form: 0.0e+00",
   "factorized2_s2345|form: 0.0e+00",
   "ctrlfr_s1234|z: 0.0e+00",
   "ctrlfr_s2345|z: 0.0e+00",
   "dual_s1234|zf: 0.0e+00",
   "dual_s2345|zf: 0.0e+00",
   "factorizedc3m_e123|form: 0.0e+00",
   "factorizedc3m_e205|form: 0.0e+00"
  ]
 }
}
```

## hifi3d (nocrop_cross, 98224 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto), coefficienti | 0.619 [0.503, 0.710] | 0.156 [0.047, 0.259] |
| GNM (visto), mesh d'identita' FR | 0.608 [0.500, 0.693] | 0.093 [-0.004, 0.189] |
| GNM (visto), mesh d'identita' SR | 0.482 [0.369, 0.582] | 0.325 [0.205, 0.441] |
| FLAME 2023 Open, coefficienti | 0.566 [0.444, 0.663] | 0.159 [0.044, 0.268] |
| FLAME 2023 Open, mesh d'identita' FR | 0.639 [0.532, 0.725] | 0.125 [0.025, 0.224] |
| FLAME 2023 Open, mesh d'identita' SR | 0.276 [0.178, 0.379] | 0.287 [0.177, 0.398] |
| varifold in mm (massa unitaria) | 0.494 [0.411, 0.567] | 0.195 [0.126, 0.261] |
| factorized s1234, d_F cal. | 0.749 [0.673, 0.806] | 0.318 [0.227, 0.407] |
| factorized s2345, d_F cal. | 0.731 [0.657, 0.792] | 0.312 [0.220, 0.401] |
| factorized s1234, d_P | 0.426 [0.333, 0.510] | 0.622 [0.550, 0.685] |
| factorized s2345, d_P | 0.417 [0.315, 0.512] | 0.613 [0.533, 0.688] |
| ctrlfr s1234 | 0.757 [0.685, 0.817] | 0.339 [0.242, 0.425] |
| ctrlfr s2345 | 0.746 [0.670, 0.809] | 0.348 [0.251, 0.435] |
| NICP su template in mm | 0.614 [0.506, 0.698] | 0.083 [-0.018, 0.185] |
| ICP + Chamfer in mm | 0.643 [0.572, 0.702] | 0.365 [0.280, 0.452] |
| NICP per coppia (cs) | 0.376 [0.296, 0.465] | 0.595 [0.546, 0.648] |
| taglia oracolo | 0.737 [0.657, 0.802] | 0.074 [-0.025, 0.165] |

Delta appaiati, braccio - concorrente (rho, stesse righe e repliche; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto), coefficienti | +0.130 [+0.078, +0.188], P 0.000 | +0.112 [+0.062, +0.176], P 0.000 | +0.139 [+0.083, +0.201], P 0.000 | +0.127 [+0.073, +0.186], P 0.000 |
| GNM (visto), mesh d'identita' FR | +0.141 [+0.088, +0.195], P 0.000 | +0.123 [+0.070, +0.180], P 0.000 | +0.149 [+0.096, +0.209], P 0.000 | +0.138 [+0.084, +0.197], P 0.000 |
| GNM (visto), mesh d'identita' SR | +0.266 [+0.195, +0.347], P 0.000 | +0.249 [+0.177, +0.330], P 0.000 | +0.275 [+0.208, +0.356], P 0.000 | +0.264 [+0.195, +0.345], P 0.000 |
| FLAME 2023 Open, coefficienti | +0.182 [+0.119, +0.254], P 0.000 | +0.165 [+0.102, +0.239], P 0.000 | +0.191 [+0.133, +0.256], P 0.000 | +0.179 [+0.121, +0.245], P 0.000 |
| FLAME 2023 Open, mesh d'identita' FR | +0.109 [+0.062, +0.161], P 0.000 | +0.092 [+0.044, +0.148], P 0.000 | +0.118 [+0.071, +0.174], P 0.000 | +0.107 [+0.060, +0.158], P 0.000 |
| FLAME 2023 Open, mesh d'identita' SR | +0.472 [+0.385, +0.563], P 0.000 | +0.455 [+0.369, +0.546], P 0.000 | +0.481 [+0.392, +0.568], P 0.000 | +0.469 [+0.383, +0.554], P 0.000 |
| varifold in mm (massa unitaria) | +0.254 [+0.211, +0.301], P 0.000 | +0.237 [+0.195, +0.284], P 0.000 | +0.263 [+0.224, +0.306], P 0.000 | +0.252 [+0.211, +0.293], P 0.000 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto), coefficienti | +0.465 [+0.375, +0.567], P 0.000 | +0.457 [+0.362, +0.560], P 0.000 | +0.182 [+0.123, +0.253], P 0.000 | +0.191 [+0.130, +0.259], P 0.000 |
| GNM (visto), mesh d'identita' FR | +0.529 [+0.433, +0.626], P 0.000 | +0.520 [+0.423, +0.618], P 0.000 | +0.245 [+0.183, +0.312], P 0.000 | +0.255 [+0.192, +0.322], P 0.000 |
| GNM (visto), mesh d'identita' SR | +0.297 [+0.212, +0.395], P 0.000 | +0.288 [+0.194, +0.393], P 0.000 | +0.014 [-0.075, +0.110], P 0.362 | +0.023 [-0.060, +0.113], P 0.303 |
| FLAME 2023 Open, coefficienti | +0.463 [+0.378, +0.556], P 0.000 | +0.454 [+0.356, +0.556], P 0.000 | +0.180 [+0.114, +0.253], P 0.000 | +0.189 [+0.122, +0.262], P 0.000 |
| FLAME 2023 Open, mesh d'identita' FR | +0.497 [+0.407, +0.592], P 0.000 | +0.488 [+0.394, +0.586], P 0.000 | +0.214 [+0.159, +0.273], P 0.000 | +0.223 [+0.164, +0.285], P 0.000 |
| FLAME 2023 Open, mesh d'identita' SR | +0.335 [+0.255, +0.421], P 0.000 | +0.326 [+0.241, +0.415], P 0.000 | +0.052 [-0.039, +0.141], P 0.131 | +0.061 [-0.026, +0.146], P 0.080 |
| varifold in mm (massa unitaria) | +0.427 [+0.359, +0.496], P 0.000 | +0.418 [+0.353, +0.487], P 0.000 | +0.144 [+0.090, +0.198], P 0.000 | +0.153 [+0.102, +0.205], P 0.000 |

## facescape (nocrop_cross, 88725 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto), coefficienti | 0.488 [0.399, 0.561] | 0.460 [0.376, 0.538] |
| GNM (visto), mesh d'identita' FR | 0.465 [0.386, 0.538] | 0.371 [0.305, 0.436] |
| GNM (visto), mesh d'identita' SR | 0.523 [0.424, 0.614] | 0.576 [0.483, 0.663] |
| FLAME 2023 Open, coefficienti | 0.385 [0.299, 0.467] | 0.396 [0.311, 0.476] |
| FLAME 2023 Open, mesh d'identita' FR | 0.449 [0.366, 0.519] | 0.379 [0.305, 0.453] |
| FLAME 2023 Open, mesh d'identita' SR | 0.404 [0.306, 0.501] | 0.469 [0.372, 0.561] |
| varifold in mm (massa unitaria) | 0.264 [0.223, 0.299] | 0.281 [0.246, 0.313] |
| factorized s1234, d_F cal. | 0.661 [0.586, 0.727] | 0.692 [0.623, 0.753] |
| factorized s2345, d_F cal. | 0.669 [0.597, 0.733] | 0.687 [0.620, 0.748] |
| factorized s1234, d_P | 0.677 [0.596, 0.750] | 0.747 [0.677, 0.804] |
| factorized s2345, d_P | 0.684 [0.601, 0.760] | 0.753 [0.687, 0.813] |
| ctrlfr s1234 | 0.661 [0.593, 0.721] | 0.619 [0.543, 0.688] |
| ctrlfr s2345 | 0.653 [0.585, 0.712] | 0.627 [0.554, 0.691] |
| NICP su template in mm | 0.542 [0.468, 0.605] | 0.399 [0.320, 0.473] |
| ICP + Chamfer in mm | 0.453 [0.389, 0.514] | 0.477 [0.414, 0.533] |
| NICP per coppia (cs) | 0.346 [0.276, 0.419] | 0.398 [0.329, 0.464] |
| taglia oracolo | 0.464 [0.345, 0.571] | 0.119 [0.022, 0.229] |

Delta appaiati, braccio - concorrente (rho, stesse righe e repliche; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto), coefficienti | +0.173 [+0.086, +0.264], P 0.000 | +0.181 [+0.091, +0.272], P 0.000 | +0.174 [+0.086, +0.260], P 0.000 | +0.165 [+0.073, +0.252], P 0.000 |
| GNM (visto), mesh d'identita' FR | +0.196 [+0.108, +0.288], P 0.000 | +0.204 [+0.115, +0.290], P 0.000 | +0.196 [+0.109, +0.278], P 0.000 | +0.188 [+0.088, +0.272], P 0.000 |
| GNM (visto), mesh d'identita' SR | +0.138 [+0.044, +0.236], P 0.003 | +0.146 [+0.046, +0.246], P 0.003 | +0.138 [+0.040, +0.232], P 0.001 | +0.130 [+0.031, +0.227], P 0.001 |
| FLAME 2023 Open, coefficienti | +0.276 [+0.189, +0.374], P 0.000 | +0.284 [+0.194, +0.380], P 0.000 | +0.276 [+0.186, +0.365], P 0.000 | +0.268 [+0.175, +0.360], P 0.000 |
| FLAME 2023 Open, mesh d'identita' FR | +0.212 [+0.123, +0.305], P 0.000 | +0.220 [+0.128, +0.311], P 0.000 | +0.212 [+0.121, +0.296], P 0.000 | +0.204 [+0.106, +0.295], P 0.000 |
| FLAME 2023 Open, mesh d'identita' SR | +0.257 [+0.157, +0.354], P 0.000 | +0.265 [+0.164, +0.366], P 0.000 | +0.258 [+0.150, +0.355], P 0.000 | +0.249 [+0.146, +0.349], P 0.000 |
| varifold in mm (massa unitaria) | +0.396 [+0.349, +0.437], P 0.000 | +0.404 [+0.362, +0.442], P 0.000 | +0.397 [+0.344, +0.445], P 0.000 | +0.389 [+0.342, +0.428], P 0.000 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto), coefficienti | +0.286 [+0.197, +0.376], P 0.000 | +0.293 [+0.208, +0.377], P 0.000 | +0.159 [+0.065, +0.246], P 0.000 | +0.167 [+0.073, +0.257], P 0.000 |
| GNM (visto), mesh d'identita' FR | +0.375 [+0.294, +0.453], P 0.000 | +0.382 [+0.300, +0.457], P 0.000 | +0.248 [+0.162, +0.323], P 0.000 | +0.256 [+0.168, +0.332], P 0.000 |
| GNM (visto), mesh d'identita' SR | +0.170 [+0.075, +0.264], P 0.000 | +0.177 [+0.091, +0.265], P 0.000 | +0.042 [-0.062, +0.144], P 0.217 | +0.051 [-0.055, +0.154], P 0.163 |
| FLAME 2023 Open, coefficienti | +0.350 [+0.261, +0.442], P 0.000 | +0.357 [+0.267, +0.445], P 0.000 | +0.223 [+0.122, +0.312], P 0.000 | +0.231 [+0.134, +0.327], P 0.000 |
| FLAME 2023 Open, mesh d'identita' FR | +0.367 [+0.285, +0.450], P 0.000 | +0.374 [+0.291, +0.456], P 0.000 | +0.240 [+0.150, +0.322], P 0.000 | +0.248 [+0.155, +0.330], P 0.000 |
| FLAME 2023 Open, mesh d'identita' SR | +0.277 [+0.177, +0.374], P 0.000 | +0.284 [+0.188, +0.380], P 0.000 | +0.150 [+0.039, +0.256], P 0.000 | +0.158 [+0.050, +0.269], P 0.001 |
| varifold in mm (massa unitaria) | +0.465 [+0.418, +0.503], P 0.000 | +0.472 [+0.429, +0.507], P 0.000 | +0.337 [+0.283, +0.384], P 0.000 | +0.346 [+0.293, +0.391], P 0.000 |

## faceverse (mesh_pair_nocrop, 99000 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto), coefficienti | 0.220 [0.148, 0.286] | 0.182 [0.104, 0.252] |
| GNM (visto), mesh d'identita' FR | 0.227 [0.140, 0.311] | 0.171 [0.082, 0.260] |
| GNM (visto), mesh d'identita' SR | 0.228 [0.145, 0.303] | 0.209 [0.129, 0.285] |
| FLAME 2023 Open, coefficienti | 0.174 [0.109, 0.237] | 0.152 [0.079, 0.218] |
| FLAME 2023 Open, mesh d'identita' FR | 0.211 [0.137, 0.279] | 0.170 [0.089, 0.245] |
| FLAME 2023 Open, mesh d'identita' SR | 0.166 [0.091, 0.235] | 0.155 [0.075, 0.228] |
| varifold in mm (massa unitaria) | 0.260 [0.201, 0.313] | 0.250 [0.193, 0.306] |
| factorized s1234, d_F cal. | 0.303 [0.229, 0.374] | 0.269 [0.195, 0.341] |
| factorized s2345, d_F cal. | 0.318 [0.249, 0.379] | 0.286 [0.206, 0.353] |
| factorized s1234, d_P | 0.283 [0.210, 0.346] | 0.286 [0.215, 0.352] |
| factorized s2345, d_P | 0.308 [0.234, 0.373] | 0.313 [0.237, 0.377] |
| ctrlfr s1234 | 0.282 [0.215, 0.343] | 0.273 [0.201, 0.339] |
| ctrlfr s2345 | 0.258 [0.187, 0.324] | 0.259 [0.188, 0.325] |
| NICP su template in mm | 0.212 [0.117, 0.308] | 0.152 [0.058, 0.246] |
| ICP + Chamfer in mm | 0.337 [0.262, 0.410] | 0.309 [0.234, 0.380] |
| NICP per coppia (cs) | 0.212 [0.131, 0.292] | 0.235 [0.158, 0.310] |
| taglia oracolo | 0.209 [0.114, 0.307] | 0.019 [-0.067, 0.106] |

Delta appaiati, braccio - concorrente (rho, stesse righe e repliche; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto), coefficienti | +0.083 [-0.014, +0.183], P 0.054 | +0.098 [+0.006, +0.191], P 0.018 | +0.062 [-0.024, +0.155], P 0.078 | +0.038 [-0.065, +0.147], P 0.238 |
| GNM (visto), mesh d'identita' FR | +0.076 [-0.016, +0.177], P 0.052 | +0.091 [-0.001, +0.184], P 0.026 | +0.055 [-0.047, +0.156], P 0.147 | +0.031 [-0.076, +0.140], P 0.309 |
| GNM (visto), mesh d'identita' SR | +0.075 [-0.030, +0.177], P 0.074 | +0.090 [-0.010, +0.192], P 0.044 | +0.055 [-0.042, +0.153], P 0.136 | +0.030 [-0.077, +0.141], P 0.301 |
| FLAME 2023 Open, coefficienti | +0.128 [+0.036, +0.221], P 0.003 | +0.144 [+0.062, +0.227], P 0.000 | +0.108 [+0.019, +0.192], P 0.005 | +0.083 [-0.012, +0.175], P 0.046 |
| FLAME 2023 Open, mesh d'identita' FR | +0.091 [+0.005, +0.177], P 0.017 | +0.107 [+0.029, +0.188], P 0.004 | +0.071 [-0.026, +0.161], P 0.063 | +0.047 [-0.051, +0.143], P 0.164 |
| FLAME 2023 Open, mesh d'identita' SR | +0.137 [+0.042, +0.245], P 0.004 | +0.152 [+0.059, +0.252], P 0.001 | +0.117 [+0.025, +0.208], P 0.002 | +0.092 [-0.013, +0.194], P 0.037 |
| varifold in mm (massa unitaria) | +0.043 [-0.019, +0.107], P 0.091 | +0.058 [-0.009, +0.118], P 0.045 | +0.022 [-0.040, +0.076], P 0.256 | -0.002 [-0.069, +0.061], P 0.531 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto), coefficienti | +0.104 [+0.015, +0.200], P 0.013 | +0.131 [+0.042, +0.224], P 0.003 | +0.090 [+0.006, +0.184], P 0.018 | +0.077 [-0.011, +0.173], P 0.042 |
| GNM (visto), mesh d'identita' FR | +0.115 [+0.013, +0.220], P 0.018 | +0.142 [+0.040, +0.240], P 0.005 | +0.102 [-0.004, +0.202], P 0.029 | +0.088 [-0.012, +0.192], P 0.040 |
| GNM (visto), mesh d'identita' SR | +0.077 [-0.016, +0.179], P 0.055 | +0.104 [+0.004, +0.208], P 0.021 | +0.064 [-0.032, +0.163], P 0.105 | +0.050 [-0.053, +0.153], P 0.181 |
| FLAME 2023 Open, coefficienti | +0.134 [+0.046, +0.224], P 0.002 | +0.161 [+0.079, +0.245], P 0.000 | +0.121 [+0.032, +0.207], P 0.002 | +0.107 [+0.015, +0.196], P 0.011 |
| FLAME 2023 Open, mesh d'identita' FR | +0.116 [+0.019, +0.206], P 0.008 | +0.143 [+0.055, +0.225], P 0.001 | +0.103 [+0.002, +0.196], P 0.024 | +0.089 [-0.001, +0.181], P 0.030 |
| FLAME 2023 Open, mesh d'identita' SR | +0.131 [+0.040, +0.233], P 0.004 | +0.158 [+0.063, +0.253], P 0.000 | +0.118 [+0.029, +0.218], P 0.004 | +0.104 [+0.001, +0.208], P 0.024 |
| varifold in mm (massa unitaria) | +0.036 [-0.021, +0.092], P 0.110 | +0.063 [-0.004, +0.123], P 0.032 | +0.023 [-0.043, +0.078], P 0.255 | +0.009 [-0.057, +0.074], P 0.376 |

## faceverse_neutral (mesh_pair_nocrop, 99000 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto), coefficienti | 0.240 [0.164, 0.309] | 0.207 [0.127, 0.277] |
| GNM (visto), mesh d'identita' FR | 0.232 [0.140, 0.318] | 0.174 [0.078, 0.265] |
| GNM (visto), mesh d'identita' SR | 0.240 [0.157, 0.320] | 0.228 [0.145, 0.308] |
| FLAME 2023 Open, coefficienti | 0.171 [0.102, 0.241] | 0.150 [0.073, 0.223] |
| FLAME 2023 Open, mesh d'identita' FR | 0.214 [0.134, 0.291] | 0.172 [0.086, 0.255] |
| FLAME 2023 Open, mesh d'identita' SR | 0.169 [0.089, 0.244] | 0.162 [0.078, 0.240] |
| varifold in mm (massa unitaria) | 0.285 [0.223, 0.346] | 0.276 [0.211, 0.337] |
| factorized s1234, d_F cal. | 0.341 [0.263, 0.416] | 0.308 [0.225, 0.387] |
| factorized s2345, d_F cal. | 0.370 [0.298, 0.440] | 0.336 [0.252, 0.409] |
| factorized s1234, d_P | 0.321 [0.237, 0.391] | 0.333 [0.251, 0.405] |
| factorized s2345, d_P | 0.361 [0.276, 0.439] | 0.372 [0.291, 0.444] |
| ctrlfr s1234 | 0.327 [0.248, 0.399] | 0.326 [0.243, 0.400] |
| ctrlfr s2345 | 0.299 [0.214, 0.374] | 0.305 [0.224, 0.377] |
| taglia oracolo | 0.209 [0.114, 0.307] | 0.019 [-0.067, 0.106] |

Delta appaiati, braccio - concorrente (rho, stesse righe e repliche; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto), coefficienti | +0.100 [-0.009, +0.201], P 0.031 | +0.130 [+0.039, +0.214], P 0.001 | +0.087 [-0.001, +0.178], P 0.027 | +0.059 [-0.047, +0.164], P 0.129 |
| GNM (visto), mesh d'identita' FR | +0.109 [+0.007, +0.214], P 0.016 | +0.139 [+0.046, +0.231], P 0.002 | +0.096 [-0.014, +0.199], P 0.039 | +0.068 [-0.043, +0.170], P 0.110 |
| GNM (visto), mesh d'identita' SR | +0.101 [-0.005, +0.208], P 0.032 | +0.131 [+0.031, +0.225], P 0.001 | +0.087 [-0.017, +0.185], P 0.045 | +0.060 [-0.055, +0.163], P 0.142 |
| FLAME 2023 Open, coefficienti | +0.170 [+0.067, +0.274], P 0.002 | +0.199 [+0.107, +0.291], P 0.000 | +0.156 [+0.055, +0.253], P 0.001 | +0.128 [+0.014, +0.225], P 0.012 |
| FLAME 2023 Open, mesh d'identita' FR | +0.127 [+0.017, +0.232], P 0.012 | +0.156 [+0.064, +0.247], P 0.002 | +0.113 [+0.008, +0.213], P 0.018 | +0.085 [-0.023, +0.188], P 0.070 |
| FLAME 2023 Open, mesh d'identita' SR | +0.172 [+0.061, +0.285], P 0.002 | +0.202 [+0.099, +0.306], P 0.000 | +0.159 [+0.057, +0.260], P 0.000 | +0.131 [+0.014, +0.234], P 0.012 |
| varifold in mm (massa unitaria) | +0.056 [-0.009, +0.128], P 0.045 | +0.086 [+0.020, +0.144], P 0.005 | +0.042 [-0.022, +0.099], P 0.101 | +0.014 [-0.059, +0.080], P 0.335 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto), coefficienti | +0.126 [+0.028, +0.222], P 0.007 | +0.165 [+0.075, +0.252], P 0.000 | +0.119 [+0.035, +0.208], P 0.004 | +0.098 [+0.003, +0.192], P 0.024 |
| GNM (visto), mesh d'identita' FR | +0.159 [+0.051, +0.265], P 0.004 | +0.198 [+0.094, +0.294], P 0.000 | +0.152 [+0.046, +0.252], P 0.007 | +0.131 [+0.036, +0.226], P 0.009 |
| GNM (visto), mesh d'identita' SR | +0.106 [+0.006, +0.206], P 0.022 | +0.144 [+0.042, +0.237], P 0.002 | +0.098 [-0.002, +0.195], P 0.027 | +0.078 [-0.027, +0.169], P 0.074 |
| FLAME 2023 Open, coefficienti | +0.183 [+0.077, +0.284], P 0.001 | +0.222 [+0.121, +0.317], P 0.000 | +0.176 [+0.071, +0.275], P 0.001 | +0.155 [+0.044, +0.252], P 0.001 |
| FLAME 2023 Open, mesh d'identita' FR | +0.162 [+0.046, +0.266], P 0.004 | +0.200 [+0.097, +0.296], P 0.001 | +0.154 [+0.045, +0.252], P 0.005 | +0.134 [+0.032, +0.228], P 0.007 |
| FLAME 2023 Open, mesh d'identita' SR | +0.171 [+0.071, +0.279], P 0.000 | +0.210 [+0.105, +0.314], P 0.000 | +0.164 [+0.060, +0.270], P 0.000 | +0.143 [+0.036, +0.246], P 0.005 |
| varifold in mm (massa unitaria) | +0.057 [-0.006, +0.125], P 0.034 | +0.096 [+0.027, +0.161], P 0.005 | +0.050 [-0.019, +0.111], P 0.080 | +0.029 [-0.045, +0.098], P 0.205 |

## famos (scan gallery -> scan, 105 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto), coefficienti | 0.624 [0.195, 0.883] | 0.418 [-0.012, 0.772] |
| GNM (visto), mesh d'identita' FR | 0.586 [0.145, 0.859] | 0.274 [-0.208, 0.690] |
| GNM (visto), mesh d'identita' SR | 0.460 [0.139, 0.720] | 0.547 [0.188, 0.825] |
| FLAME 2023 Open, coefficienti | 0.457 [0.064, 0.760] | 0.390 [0.027, 0.713] |
| FLAME 2023 Open, mesh d'identita' FR | 0.565 [0.134, 0.856] | 0.291 [-0.186, 0.691] |
| FLAME 2023 Open, mesh d'identita' SR | 0.217 [-0.108, 0.556] | 0.412 [0.083, 0.748] |
| varifold in mm (massa unitaria) | 0.688 [0.368, 0.902] | 0.582 [0.170, 0.842] |
| factorized s1234, d_F cal. | 0.678 [0.237, 0.897] | 0.662 [0.323, 0.858] |
| factorized s2345, d_F cal. | 0.654 [0.214, 0.896] | 0.616 [0.225, 0.839] |
| factorized s1234, d_P | 0.412 [0.088, 0.680] | 0.740 [0.548, 0.883] |
| factorized s2345, d_P | 0.437 [0.107, 0.698] | 0.728 [0.516, 0.869] |
| ctrlfr s1234 | 0.831 [0.575, 0.949] | 0.709 [0.394, 0.871] |
| ctrlfr s2345 | 0.818 [0.569, 0.944] | 0.691 [0.378, 0.869] |
| ICP + Chamfer in mm | 0.739 [0.362, 0.914] | 0.548 [0.185, 0.799] |
| NICP per coppia (cs) | 0.584 [0.243, 0.818] | 0.775 [0.540, 0.860] |
| taglia oracolo | 0.787 [0.546, 0.900] | 0.213 [-0.232, 0.573] |

Delta appaiati, braccio - concorrente (rho, stesse righe e repliche; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto), coefficienti | +0.054 [-0.237, +0.302], P 0.328 | +0.031 [-0.226, +0.245], P 0.388 | +0.207 [-0.013, +0.508], P 0.037 | +0.195 [-0.050, +0.531], P 0.068 |
| GNM (visto), mesh d'identita' FR | +0.092 [-0.170, +0.382], P 0.229 | +0.068 [-0.163, +0.314], P 0.246 | +0.245 [+0.021, +0.583], P 0.017 | +0.232 [+0.001, +0.598], P 0.025 |
| GNM (visto), mesh d'identita' SR | +0.218 [-0.114, +0.490], P 0.111 | +0.194 [-0.172, +0.490], P 0.156 | +0.371 [+0.089, +0.651], P 0.004 | +0.358 [+0.058, +0.644], P 0.008 |
| FLAME 2023 Open, coefficienti | +0.221 [-0.049, +0.461], P 0.040 | +0.197 [-0.058, +0.440], P 0.057 | +0.374 [+0.111, +0.662], P 0.004 | +0.361 [+0.084, +0.678], P 0.004 |
| FLAME 2023 Open, mesh d'identita' FR | +0.113 [-0.183, +0.385], P 0.202 | +0.089 [-0.176, +0.315], P 0.230 | +0.266 [+0.023, +0.603], P 0.016 | +0.253 [+0.006, +0.607], P 0.021 |
| FLAME 2023 Open, mesh d'identita' SR | +0.461 [-0.013, +0.801], P 0.028 | +0.437 [-0.041, +0.803], P 0.047 | +0.614 [+0.198, +0.953], P 0.001 | +0.602 [+0.185, +0.943], P 0.002 |
| varifold in mm (massa unitaria) | -0.010 [-0.263, +0.185], P 0.520 | -0.034 [-0.278, +0.155], P 0.633 | +0.143 [-0.016, +0.338], P 0.031 | +0.130 [+0.001, +0.330], P 0.024 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto), coefficienti | +0.323 [-0.136, +0.797], P 0.084 | +0.311 [-0.118, +0.785], P 0.088 | +0.292 [+0.045, +0.561], P 0.010 | +0.273 [-0.013, +0.555], P 0.028 |
| GNM (visto), mesh d'identita' FR | +0.466 [-0.029, +1.036], P 0.042 | +0.454 [-0.028, +1.010], P 0.036 | +0.435 [+0.099, +0.794], P 0.002 | +0.417 [+0.106, +0.727], P 0.001 |
| GNM (visto), mesh d'identita' SR | +0.193 [-0.141, +0.579], P 0.138 | +0.181 [-0.162, +0.572], P 0.164 | +0.162 [-0.132, +0.402], P 0.144 | +0.143 [-0.236, +0.416], P 0.208 |
| FLAME 2023 Open, coefficienti | +0.350 [-0.079, +0.749], P 0.044 | +0.338 [-0.066, +0.733], P 0.052 | +0.319 [+0.049, +0.562], P 0.013 | +0.301 [-0.002, +0.556], P 0.029 |
| FLAME 2023 Open, mesh d'identita' FR | +0.450 [-0.045, +1.011], P 0.039 | +0.438 [-0.033, +0.967], P 0.035 | +0.419 [+0.100, +0.743], P 0.003 | +0.400 [+0.090, +0.717], P 0.005 |
| FLAME 2023 Open, mesh d'identita' SR | +0.328 [+0.006, +0.650], P 0.023 | +0.316 [-0.042, +0.677], P 0.039 | +0.297 [-0.139, +0.649], P 0.101 | +0.279 [-0.225, +0.658], P 0.146 |
| varifold in mm (massa unitaria) | +0.158 [-0.208, +0.574], P 0.184 | +0.146 [-0.174, +0.515], P 0.178 | +0.127 [-0.051, +0.347], P 0.084 | +0.109 [-0.043, +0.287], P 0.091 |

# Emendamento 1 (POST HOC): varianti A e B, errore di superficie

Protocollo `PROTOCOL_emendamento_1.md` (sha256 `3d06ab086fb1844d03300d0adae1d5c00a5a70f949c9a010361d8a1a3b191fda`), scritto dopo i numeri sopra. Colonne `vA` = senza espressione sulle viste neutre (su `faceverse` espressione libera col prior dichiarato), `vB` = modello nel ciclo. Stesse righe e repliche; numeri in `spearman_e1.csv` e `paired_e1.csv`.

**Celle con IC sotto 0 (concorrente davanti al braccio) fra i delta dichiarati:** 21: hifi3d SR: ctrlfr s1234 - GNM (visto) vB, coefficienti = -0.336 [-0.434, -0.241]; hifi3d SR: ctrlfr s2345 - GNM (visto) vB, coefficienti = -0.327 [-0.428, -0.230]; hifi3d SR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' SR = -0.268 [-0.358, -0.191]; hifi3d SR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.259 [-0.344, -0.187]; hifi3d SR: ctrlfr s1234 - FLAME 2023 Open vB, coefficienti = -0.247 [-0.350, -0.149]; hifi3d SR: ctrlfr s2345 - FLAME 2023 Open vB, coefficienti = -0.238 [-0.341, -0.139]; hifi3d SR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' SR = -0.259 [-0.347, -0.184]; hifi3d SR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' SR = -0.250 [-0.330, -0.182]; facescape FR: factorized s1234, d_F cal. - GNM (visto) vB, mesh d'identita' SR = -0.106 [-0.175, -0.027]; facescape FR: factorized s2345, d_F cal. - GNM (visto) vB, mesh d'identita' SR = -0.098 [-0.167, -0.014]; facescape FR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' SR = -0.105 [-0.193, -0.021]; facescape FR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.114 [-0.194, -0.031]; facescape SR: factorized s1234, d_P - GNM (visto) vB, mesh d'identita' SR = -0.107 [-0.155, -0.065]; facescape SR: factorized s2345, d_P - GNM (visto) vB, mesh d'identita' SR = -0.101 [-0.148, -0.058]; facescape SR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' SR = -0.235 [-0.296, -0.182]; facescape SR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.226 [-0.283, -0.174]; facescape SR: factorized s1234, d_P - FLAME 2023 Open vB, mesh d'identita' SR = -0.074 [-0.129, -0.022]; facescape SR: factorized s2345, d_P - FLAME 2023 Open vB, mesh d'identita' SR = -0.067 [-0.120, -0.014]; facescape SR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' SR = -0.201 [-0.268, -0.140]; facescape SR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' SR = -0.193 [-0.256, -0.130]; faceverse_neutral SR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.093 [-0.181, -0.011].

## Pilota (soggetti non valutati, senza GT)

| fit | S gnm | S flame2023 |
| --- | --- | --- |
| orig | 0.1518 | 0.2304 |
| va | 0.1513 | 0.2260 |
| vB sigma 0.5 tau 2.0 iter 5 | 0.1716 | 0.2273 |
| vB sigma 0.5 tau 2.0 iter 10 | 0.1752 | 0.2259 |
| vB sigma 0.5 tau 5.0 iter 5 | 0.1713 | 0.2300 |
| vB sigma 0.5 tau 5.0 iter 10 | 0.1759 | 0.2299 |
| vB sigma 0.5 tau 10.0 iter 5 | 0.1728 | 0.2302 |
| vB sigma 0.5 tau 10.0 iter 10 | 0.1783 | 0.2308 |
| vB sigma 1.0 tau 2.0 iter 5 | 0.1639 | 0.2186 |
| vB sigma 1.0 tau 2.0 iter 10 | 0.1683 | 0.2158 |
| vB sigma 1.0 tau 5.0 iter 5 | 0.1626 | 0.2194 |
| vB sigma 1.0 tau 5.0 iter 10 | 0.1655 | 0.2196 |
| vB sigma 1.0 tau 10.0 iter 5 | 0.1639 | 0.2200 |
| vB sigma 1.0 tau 10.0 iter 10 | 0.1662 | 0.2196 |
| vB sigma 2.0 tau 2.0 iter 5 | 0.1594 | 0.2139 |
| vB sigma 2.0 tau 2.0 iter 10 | 0.1630 | 0.2090 |
| vB sigma 2.0 tau 5.0 iter 5 | 0.1559 | 0.2106 **scelta** |
| vB sigma 2.0 tau 5.0 iter 10 | 0.1577 | 0.2094 |
| vB sigma 2.0 tau 10.0 iter 5 | 0.1566 **scelta** | 0.2112 |
| vB sigma 2.0 tau 10.0 iter 10 | 0.1576 | 0.2103 |

- gnm: scelta {'key': 'vb|2.0|10.0|5', 'sigma': 2.0, 'tau': 10.0, 'iters': 5}, fit falliti sul pilota 0, 4 min
- flame2023: scelta {'key': 'vb|2.0|5.0|5', 'sigma': 2.0, 'tau': 5.0, 'iters': 5}, fit falliti sul pilota 0, 13 min

## Fit delle varianti ed errore di superficie

| vista | modello | fit | mesh | fallite (numericamente) | superficie mm: mediana | p95 (mediana sulle mesh) | p95 max | ingresso -> M tenuti | B: corrisp. tenute M -> ingr. / ingr. -> M | s/mesh (NICP + A + B + superficie) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | gnm | originale | 500 | vedi sopra | 1.14 | 3.34 | 5.05 | 0.78 | - | - |
| hifi3d | gnm | vA | 500 | 0 | 1.14 | 3.36 | 5.08 | 0.79 | - | - |
| hifi3d | gnm | vB | 500 | 0 | 0.37 | 1.40 | 2.75 | 0.78 | 0.99 / 0.79 | 4.8 |
| hifi3d | flame2023 | originale | 500 | vedi sopra | 1.34 | 4.00 | 5.98 | 0.69 | - | - |
| hifi3d | flame2023 | vA | 500 | 0 | 1.35 | 4.00 | 6.02 | 0.69 | - | - |
| hifi3d | flame2023 | vB | 500 | 0 | 0.36 | 1.41 | 2.68 | 0.69 | 0.99 / 0.69 | 2.0 |
| facescape | gnm | originale | 500 | vedi sopra | 1.02 | 3.03 | 5.73 | 0.72 | - | - |
| facescape | gnm | vA | 500 | 0 | 1.02 | 3.04 | 5.69 | 0.72 | - | - |
| facescape | gnm | vB | 500 | 0 | 0.29 | 0.96 | 1.69 | 0.72 | 0.99 / 0.73 | 5.1 |
| facescape | flame2023 | originale | 500 | vedi sopra | 1.25 | 3.68 | 6.58 | 0.67 | - | - |
| facescape | flame2023 | vA | 500 | 0 | 1.25 | 3.67 | 6.59 | 0.67 | - | - |
| facescape | flame2023 | vB | 500 | 0 | 0.31 | 1.09 | 2.34 | 0.68 | 0.99 / 0.68 | 2.3 |
| faceverse | gnm | originale | 500 | vedi sopra | 1.48 | 4.59 | 7.76 | 0.70 | - | - |
| faceverse | gnm | vA | 500 | 0 | 1.48 | 4.60 | 7.77 | 0.70 | - | - |
| faceverse | gnm | vB | 500 | 0 | 0.50 | 1.87 | 2.94 | 0.70 | 0.99 / 0.70 | 9.3 |
| faceverse | flame2023 | originale | 500 | vedi sopra | 1.61 | 5.15 | 9.76 | 0.55 | - | - |
| faceverse | flame2023 | vA | 500 | 0 | 1.61 | 5.15 | 9.76 | 0.55 | - | - |
| faceverse | flame2023 | vB | 500 | 0 | 0.47 | 1.76 | 3.62 | 0.56 | 0.99 / 0.56 | 82.4 |
| faceverse_neutral | gnm | originale | 500 | vedi sopra | 1.36 | 4.07 | 6.10 | 0.69 | - | - |
| faceverse_neutral | gnm | vA | 500 | 0 | 1.37 | 4.12 | 6.13 | 0.69 | - | - |
| faceverse_neutral | gnm | vB | 500 | 0 | 0.56 | 2.00 | 3.07 | 0.69 | 0.99 / 0.69 | 5.2 |
| faceverse_neutral | flame2023 | originale | 500 | vedi sopra | 1.46 | 4.66 | 7.76 | 0.55 | - | - |
| faceverse_neutral | flame2023 | vA | 500 | 0 | 1.47 | 4.66 | 7.80 | 0.55 | - | - |
| faceverse_neutral | flame2023 | vB | 500 | 0 | 0.45 | 1.64 | 2.91 | 0.55 | 0.99 / 0.55 | 2.6 |
| famos | gnm | originale | 15 | vedi sopra | 1.27 | 3.64 | 5.04 | 0.73 | - | - |
| famos | gnm | vA | 15 | 0 | 1.26 | 3.63 | 5.09 | 0.73 | - | - |
| famos | gnm | vB | 15 | 0 | 0.36 | 1.26 | 2.12 | 0.72 | 0.98 / 0.72 | 6.3 |
| famos | flame2023 | originale | 15 | vedi sopra | 1.21 | 3.82 | 5.66 | 0.58 | - | - |
| famos | flame2023 | vA | 15 | 0 | 1.20 | 3.84 | 5.64 | 0.59 | - | - |
| famos | flame2023 | vB | 15 | 0 | 0.35 | 1.27 | 2.03 | 0.58 | 0.97 / 0.58 | 3.7 |

## Controlli dell'emendamento

```
{
 "reference": "aau/runs/evidence/baselines_param/paired.csv",
 "n_reference": 4877,
 "n_matched": 4877,
 "max_abs_diff": {
  "arm_point": 9.71445146547012e-17,
  "delta": 9.84455572616838e-17,
  "ci_low": 9.974659986866641e-17,
  "ci_high": 2.220446049250313e-16,
  "p_le0": 8.326672684688674e-17
 },
 "rows_equal": true
}
{
 "hifi3d": {
  "rows_total": 99000,
  "rows_mask": 98224,
  "nan_rows_by_new_column": {
   "gnm_coef": 0,
   "gnm_fr": 0,
   "gnm_sr": 0,
   "flame2023_coef": 0,
   "flame2023_fr": 0,
   "flame2023_sr": 0,
   "varifold": 0,
   "gnm_va_coef": 0,
   "gnm_va_fr": 0,
   "gnm_va_sr": 0,
   "gnm_vb_coef": 0,
   "gnm_vb_fr": 0,
   "gnm_vb_sr": 0,
   "flame2023_va_coef": 0,
   "flame2023_va_fr": 0,
   "flame2023_va_sr": 0,
   "flame2023_vb_coef": 0,
   "flame2023_vb_fr": 0,
   "flame2023_vb_sr": 0
  }
 },
 "facescape": {
  "rows_total": 99000,
  "rows_mask": 88725,
  "nan_rows_by_new_column": {
   "gnm_coef": 0,
   "gnm_fr": 0,
   "gnm_sr": 0,
   "flame2023_coef": 0,
   "flame2023_fr": 0,
   "flame2023_sr": 0,
   "varifold": 0,
   "gnm_va_coef": 0,
   "gnm_va_fr": 0,
   "gnm_va_sr": 0,
   "gnm_vb_coef": 0,
   "gnm_vb_fr": 0,
   "gnm_vb_sr": 0,
   "flame2023_va_coef": 0,
   "flame2023_va_fr": 0,
   "flame2023_va_sr": 0,
   "flame2023_vb_coef": 0,
   "flame2023_vb_fr": 0,
   "flame2023_vb_sr": 0
  }
 },
 "faceverse": {
  "rows_total": 99000,
  "rows_mask": 99000,
  "nan_rows_by_new_column": {
   "gnm_coef": 0,
   "gnm_fr": 0,
   "gnm_sr": 0,
   "flame2023_coef": 0,
   "flame2023_fr": 0,
   "flame2023_sr": 0,
   "varifold": 0,
   "gnm_va_coef": 0,
   "gnm_va_fr": 0,
   "gnm_va_sr": 0,
   "gnm_vb_coef": 0,
   "gnm_vb_fr": 0,
   "gnm_vb_sr": 0,
   "flame2023_va_coef": 0,
   "flame2023_va_fr": 0,
   "flame2023_va_sr": 0,
   "flame2023_vb_coef": 0,
   "flame2023_vb_fr": 0,
   "flame2023_vb_sr": 0
  }
 },
 "faceverse_neutral": {
  "rows_total": 99000,
  "rows_mask": 99000,
  "nan_rows_by_new_column": {
   "gnm_coef": 0,
   "gnm_fr": 0,
   "gnm_sr": 0,
   "flame2023_coef": 0,
   "flame2023_fr": 0,
   "flame2023_sr": 0,
   "varifold": 0,
   "gnm_va_coef": 0,
   "gnm_va_fr": 0,
   "gnm_va_sr": 0,
   "gnm_vb_coef": 0,
   "gnm_vb_fr": 0,
   "gnm_vb_sr": 0,
   "flame2023_va_coef": 0,
   "flame2023_va_fr": 0,
   "flame2023_va_sr": 0,
   "flame2023_vb_coef": 0,
   "flame2023_vb_fr": 0,
   "flame2023_vb_sr": 0
  }
 },
 "famos": {
  "rows_total": 105,
  "rows_mask": 105,
  "nan_rows_by_new_column": {
   "gnm_coef": 0,
   "gnm_fr": 0,
   "gnm_sr": 0,
   "flame2023_coef": 0,
   "flame2023_fr": 0,
   "flame2023_sr": 0,
   "varifold": 0,
   "gnm_va_coef": 0,
   "gnm_va_fr": 0,
   "gnm_va_sr": 0,
   "gnm_vb_coef": 0,
   "gnm_vb_fr": 0,
   "gnm_vb_sr": 0,
   "flame2023_va_coef": 0,
   "flame2023_va_fr": 0,
   "flame2023_va_sr": 0,
   "flame2023_vb_coef": 0,
   "flame2023_vb_fr": 0,
   "flame2023_vb_sr": 0
  }
 }
}
```

## hifi3d, emendamento 1 (98224 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto), coefficienti | 0.619 [0.503, 0.710] | 0.156 [0.047, 0.259] |
| GNM (visto), mesh d'identita' FR | 0.608 [0.500, 0.693] | 0.093 [-0.004, 0.189] |
| GNM (visto), mesh d'identita' SR | 0.482 [0.369, 0.582] | 0.325 [0.205, 0.441] |
| FLAME 2023 Open, coefficienti | 0.566 [0.444, 0.663] | 0.159 [0.044, 0.268] |
| FLAME 2023 Open, mesh d'identita' FR | 0.639 [0.532, 0.725] | 0.125 [0.025, 0.224] |
| FLAME 2023 Open, mesh d'identita' SR | 0.276 [0.178, 0.379] | 0.287 [0.177, 0.398] |
| GNM (visto) vA, coefficienti | 0.603 [0.492, 0.693] | 0.194 [0.088, 0.295] |
| GNM (visto) vA, mesh d'identita' FR | 0.607 [0.500, 0.692] | 0.105 [0.006, 0.198] |
| GNM (visto) vA, mesh d'identita' SR | 0.291 [0.182, 0.389] | 0.339 [0.229, 0.443] |
| GNM (visto) vB, coefficienti | 0.550 [0.465, 0.628] | 0.674 [0.610, 0.734] |
| GNM (visto) vB, mesh d'identita' FR | 0.664 [0.570, 0.738] | 0.217 [0.127, 0.306] |
| GNM (visto) vB, mesh d'identita' SR | 0.411 [0.319, 0.499] | 0.607 [0.532, 0.679] |
| FLAME 2023 Open vA, coefficienti | 0.580 [0.460, 0.676] | 0.155 [0.043, 0.262] |
| FLAME 2023 Open vA, mesh d'identita' FR | 0.641 [0.536, 0.725] | 0.118 [0.020, 0.217] |
| FLAME 2023 Open vA, mesh d'identita' SR | 0.272 [0.173, 0.373] | 0.299 [0.193, 0.405] |
| FLAME 2023 Open vB, coefficienti | 0.585 [0.503, 0.661] | 0.585 [0.513, 0.652] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.686 [0.597, 0.758] | 0.231 [0.141, 0.320] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.458 [0.365, 0.542] | 0.597 [0.518, 0.673] |
| factorized s1234, d_F cal. | 0.749 [0.673, 0.806] | 0.318 [0.227, 0.407] |
| factorized s2345, d_F cal. | 0.731 [0.657, 0.792] | 0.312 [0.220, 0.401] |
| factorized s1234, d_P | 0.426 [0.333, 0.510] | 0.622 [0.550, 0.685] |
| factorized s2345, d_P | 0.417 [0.315, 0.512] | 0.613 [0.533, 0.688] |
| ctrlfr s1234 | 0.757 [0.685, 0.817] | 0.339 [0.242, 0.425] |
| ctrlfr s2345 | 0.746 [0.670, 0.809] | 0.348 [0.251, 0.435] |

Delta appaiati, braccio - concorrente delle varianti (IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vA, coefficienti | +0.145 [+0.091, +0.207], P 0.000 | +0.128 [+0.075, +0.191], P 0.000 | +0.154 [+0.099, +0.216], P 0.000 | +0.142 [+0.088, +0.200], P 0.000 |
| GNM (visto) vA, mesh d'identita' FR | +0.142 [+0.090, +0.195], P 0.000 | +0.124 [+0.073, +0.181], P 0.000 | +0.150 [+0.097, +0.210], P 0.000 | +0.139 [+0.087, +0.197], P 0.000 |
| GNM (visto) vA, mesh d'identita' SR | +0.458 [+0.365, +0.556], P 0.000 | +0.440 [+0.349, +0.537], P 0.000 | +0.466 [+0.376, +0.563], P 0.000 | +0.455 [+0.364, +0.553], P 0.000 |
| GNM (visto) vB, coefficienti | +0.199 [+0.095, +0.299], P 0.000 | +0.181 [+0.078, +0.286], P 0.000 | +0.208 [+0.105, +0.307], P 0.000 | +0.196 [+0.087, +0.300], P 0.000 |
| GNM (visto) vB, mesh d'identita' FR | +0.084 [+0.042, +0.128], P 0.000 | +0.066 [+0.023, +0.109], P 0.001 | +0.093 [+0.045, +0.139], P 0.001 | +0.081 [+0.037, +0.125], P 0.000 |
| GNM (visto) vB, mesh d'identita' SR | +0.338 [+0.235, +0.440], P 0.000 | +0.320 [+0.220, +0.420], P 0.000 | +0.347 [+0.249, +0.445], P 0.000 | +0.335 [+0.237, +0.434], P 0.000 |
| FLAME 2023 Open vA, coefficienti | +0.168 [+0.108, +0.238], P 0.000 | +0.150 [+0.090, +0.222], P 0.000 | +0.177 [+0.123, +0.241], P 0.000 | +0.165 [+0.110, +0.229], P 0.000 |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.107 [+0.061, +0.158], P 0.000 | +0.090 [+0.043, +0.143], P 0.000 | +0.116 [+0.069, +0.170], P 0.000 | +0.105 [+0.057, +0.156], P 0.000 |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.476 [+0.385, +0.570], P 0.000 | +0.458 [+0.374, +0.551], P 0.000 | +0.485 [+0.396, +0.577], P 0.000 | +0.473 [+0.383, +0.561], P 0.000 |
| FLAME 2023 Open vB, coefficienti | +0.163 [+0.072, +0.254], P 0.000 | +0.146 [+0.055, +0.238], P 0.000 | +0.172 [+0.087, +0.260], P 0.000 | +0.161 [+0.069, +0.253], P 0.000 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.063 [+0.023, +0.101], P 0.002 | +0.045 [+0.004, +0.085], P 0.015 | +0.071 [+0.031, +0.111], P 0.000 | +0.060 [+0.021, +0.099], P 0.001 |
| FLAME 2023 Open vB, mesh d'identita' SR | +0.290 [+0.203, +0.382], P 0.000 | +0.273 [+0.189, +0.362], P 0.000 | +0.299 [+0.209, +0.387], P 0.000 | +0.288 [+0.197, +0.375], P 0.000 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vA, coefficienti | +0.428 [+0.339, +0.528], P 0.000 | +0.419 [+0.323, +0.525], P 0.000 | +0.145 [+0.079, +0.216], P 0.000 | +0.154 [+0.091, +0.224], P 0.000 |
| GNM (visto) vA, mesh d'identita' FR | +0.517 [+0.423, +0.614], P 0.000 | +0.508 [+0.411, +0.606], P 0.000 | +0.234 [+0.173, +0.302], P 0.000 | +0.243 [+0.182, +0.310], P 0.000 |
| GNM (visto) vA, mesh d'identita' SR | +0.282 [+0.197, +0.378], P 0.000 | +0.274 [+0.185, +0.370], P 0.000 | -0.001 [-0.102, +0.097], P 0.483 | +0.008 [-0.090, +0.102], P 0.420 |
| GNM (visto) vB, coefficienti | -0.053 [-0.136, +0.032], P 0.885 | -0.062 [-0.149, +0.032], P 0.906 | -0.336 [-0.434, -0.241], P 1.000 | -0.327 [-0.428, -0.230], P 1.000 |
| GNM (visto) vB, mesh d'identita' FR | +0.405 [+0.320, +0.485], P 0.000 | +0.396 [+0.312, +0.482], P 0.000 | +0.122 [+0.074, +0.174], P 0.000 | +0.131 [+0.084, +0.182], P 0.000 |
| GNM (visto) vB, mesh d'identita' SR | +0.015 [-0.040, +0.074], P 0.309 | +0.006 [-0.051, +0.061], P 0.392 | -0.268 [-0.358, -0.191], P 1.000 | -0.259 [-0.344, -0.187], P 1.000 |
| FLAME 2023 Open vA, coefficienti | +0.466 [+0.380, +0.561], P 0.000 | +0.457 [+0.362, +0.559], P 0.000 | +0.183 [+0.121, +0.255], P 0.000 | +0.192 [+0.129, +0.262], P 0.000 |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.503 [+0.412, +0.599], P 0.000 | +0.495 [+0.399, +0.591], P 0.000 | +0.220 [+0.167, +0.279], P 0.000 | +0.229 [+0.171, +0.289], P 0.000 |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.323 [+0.244, +0.408], P 0.000 | +0.314 [+0.231, +0.401], P 0.000 | +0.040 [-0.053, +0.129], P 0.197 | +0.049 [-0.040, +0.134], P 0.131 |
| FLAME 2023 Open vB, coefficienti | +0.036 [-0.048, +0.123], P 0.210 | +0.028 [-0.061, +0.121], P 0.273 | -0.247 [-0.350, -0.149], P 1.000 | -0.238 [-0.341, -0.139], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.391 [+0.309, +0.473], P 0.000 | +0.382 [+0.298, +0.467], P 0.000 | +0.108 [+0.065, +0.153], P 0.000 | +0.117 [+0.074, +0.160], P 0.000 |
| FLAME 2023 Open vB, mesh d'identita' SR | +0.024 [-0.032, +0.082], P 0.203 | +0.015 [-0.040, +0.075], P 0.281 | -0.259 [-0.347, -0.184], P 1.000 | -0.250 [-0.330, -0.182], P 1.000 |

## facescape, emendamento 1 (88725 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto), coefficienti | 0.488 [0.399, 0.561] | 0.460 [0.376, 0.538] |
| GNM (visto), mesh d'identita' FR | 0.465 [0.386, 0.538] | 0.371 [0.305, 0.436] |
| GNM (visto), mesh d'identita' SR | 0.523 [0.424, 0.614] | 0.576 [0.483, 0.663] |
| FLAME 2023 Open, coefficienti | 0.385 [0.299, 0.467] | 0.396 [0.311, 0.476] |
| FLAME 2023 Open, mesh d'identita' FR | 0.449 [0.366, 0.519] | 0.379 [0.305, 0.453] |
| FLAME 2023 Open, mesh d'identita' SR | 0.404 [0.306, 0.501] | 0.469 [0.372, 0.561] |
| GNM (visto) vA, coefficienti | 0.478 [0.387, 0.556] | 0.477 [0.389, 0.562] |
| GNM (visto) vA, mesh d'identita' FR | 0.484 [0.410, 0.554] | 0.392 [0.327, 0.460] |
| GNM (visto) vA, mesh d'identita' SR | 0.507 [0.402, 0.609] | 0.570 [0.465, 0.663] |
| GNM (visto) vB, coefficienti | 0.565 [0.491, 0.636] | 0.618 [0.548, 0.683] |
| GNM (visto) vB, mesh d'identita' FR | 0.678 [0.615, 0.740] | 0.608 [0.536, 0.683] |
| GNM (visto) vB, mesh d'identita' SR | 0.767 [0.675, 0.843] | 0.854 [0.803, 0.894] |
| FLAME 2023 Open vA, coefficienti | 0.389 [0.304, 0.469] | 0.397 [0.312, 0.476] |
| FLAME 2023 Open vA, mesh d'identita' FR | 0.453 [0.375, 0.522] | 0.373 [0.301, 0.445] |
| FLAME 2023 Open vA, mesh d'identita' SR | 0.409 [0.309, 0.511] | 0.480 [0.380, 0.575] |
| FLAME 2023 Open vB, coefficienti | 0.549 [0.467, 0.625] | 0.592 [0.512, 0.663] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.630 [0.555, 0.695] | 0.590 [0.516, 0.662] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.711 [0.614, 0.798] | 0.820 [0.757, 0.870] |
| factorized s1234, d_F cal. | 0.661 [0.586, 0.727] | 0.692 [0.623, 0.753] |
| factorized s2345, d_F cal. | 0.669 [0.597, 0.733] | 0.687 [0.620, 0.748] |
| factorized s1234, d_P | 0.677 [0.596, 0.750] | 0.747 [0.677, 0.804] |
| factorized s2345, d_P | 0.684 [0.601, 0.760] | 0.753 [0.687, 0.813] |
| ctrlfr s1234 | 0.661 [0.593, 0.721] | 0.619 [0.543, 0.688] |
| ctrlfr s2345 | 0.653 [0.585, 0.712] | 0.627 [0.554, 0.691] |

Delta appaiati, braccio - concorrente delle varianti (IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vA, coefficienti | +0.183 [+0.092, +0.279], P 0.000 | +0.191 [+0.100, +0.285], P 0.000 | +0.183 [+0.091, +0.275], P 0.000 | +0.175 [+0.086, +0.264], P 0.000 |
| GNM (visto) vA, mesh d'identita' FR | +0.177 [+0.089, +0.266], P 0.000 | +0.185 [+0.098, +0.268], P 0.000 | +0.177 [+0.089, +0.258], P 0.000 | +0.169 [+0.074, +0.252], P 0.000 |
| GNM (visto) vA, mesh d'identita' SR | +0.153 [+0.053, +0.261], P 0.002 | +0.161 [+0.060, +0.270], P 0.001 | +0.154 [+0.047, +0.251], P 0.000 | +0.146 [+0.039, +0.252], P 0.001 |
| GNM (visto) vB, coefficienti | +0.095 [+0.016, +0.163], P 0.009 | +0.103 [+0.027, +0.171], P 0.009 | +0.096 [+0.003, +0.174], P 0.021 | +0.087 [+0.006, +0.163], P 0.018 |
| GNM (visto) vB, mesh d'identita' FR | -0.018 [-0.085, +0.043], P 0.691 | -0.009 [-0.074, +0.046], P 0.614 | -0.017 [-0.090, +0.051], P 0.681 | -0.025 [-0.099, +0.042], P 0.766 |
| GNM (visto) vB, mesh d'identita' SR | -0.106 [-0.175, -0.027], P 0.993 | -0.098 [-0.167, -0.014], P 0.987 | -0.105 [-0.193, -0.021], P 0.991 | -0.114 [-0.194, -0.031], P 0.997 |
| FLAME 2023 Open vA, coefficienti | +0.272 [+0.186, +0.368], P 0.000 | +0.280 [+0.192, +0.374], P 0.000 | +0.272 [+0.181, +0.359], P 0.000 | +0.264 [+0.173, +0.354], P 0.000 |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.207 [+0.117, +0.300], P 0.000 | +0.215 [+0.123, +0.304], P 0.000 | +0.208 [+0.117, +0.290], P 0.000 | +0.200 [+0.104, +0.285], P 0.000 |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.252 [+0.151, +0.350], P 0.000 | +0.260 [+0.157, +0.363], P 0.000 | +0.252 [+0.141, +0.350], P 0.000 | +0.244 [+0.139, +0.345], P 0.000 |
| FLAME 2023 Open vB, coefficienti | +0.111 [+0.038, +0.178], P 0.002 | +0.119 [+0.044, +0.190], P 0.001 | +0.112 [+0.030, +0.192], P 0.006 | +0.103 [+0.028, +0.182], P 0.008 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.031 [-0.036, +0.089], P 0.169 | +0.039 [-0.026, +0.094], P 0.115 | +0.031 [-0.047, +0.101], P 0.212 | +0.023 [-0.053, +0.089], P 0.275 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.050 [-0.124, +0.030], P 0.905 | -0.042 [-0.118, +0.043], P 0.864 | -0.050 [-0.146, +0.043], P 0.870 | -0.058 [-0.145, +0.029], P 0.912 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vA, coefficienti | +0.269 [+0.174, +0.360], P 0.000 | +0.276 [+0.185, +0.368], P 0.000 | +0.142 [+0.040, +0.233], P 0.003 | +0.150 [+0.052, +0.244], P 0.001 |
| GNM (visto) vA, mesh d'identita' FR | +0.354 [+0.269, +0.432], P 0.000 | +0.361 [+0.279, +0.437], P 0.000 | +0.226 [+0.141, +0.302], P 0.000 | +0.235 [+0.146, +0.310], P 0.000 |
| GNM (visto) vA, mesh d'identita' SR | +0.176 [+0.077, +0.278], P 0.000 | +0.183 [+0.087, +0.281], P 0.000 | +0.048 [-0.061, +0.161], P 0.201 | +0.057 [-0.055, +0.169], P 0.166 |
| GNM (visto) vB, coefficienti | +0.129 [+0.051, +0.198], P 0.000 | +0.135 [+0.052, +0.208], P 0.001 | +0.001 [-0.079, +0.078], P 0.484 | +0.010 [-0.067, +0.086], P 0.423 |
| GNM (visto) vB, mesh d'identita' FR | +0.139 [+0.064, +0.200], P 0.000 | +0.145 [+0.072, +0.205], P 0.000 | +0.011 [-0.065, +0.072], P 0.378 | +0.020 [-0.054, +0.081], P 0.303 |
| GNM (visto) vB, mesh d'identita' SR | -0.107 [-0.155, -0.065], P 1.000 | -0.101 [-0.148, -0.058], P 1.000 | -0.235 [-0.296, -0.182], P 1.000 | -0.226 [-0.283, -0.174], P 1.000 |
| FLAME 2023 Open vA, coefficienti | +0.349 [+0.261, +0.440], P 0.000 | +0.356 [+0.268, +0.442], P 0.000 | +0.222 [+0.120, +0.310], P 0.000 | +0.230 [+0.133, +0.326], P 0.000 |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.373 [+0.291, +0.457], P 0.000 | +0.380 [+0.298, +0.460], P 0.000 | +0.245 [+0.154, +0.326], P 0.000 | +0.254 [+0.165, +0.331], P 0.000 |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.267 [+0.166, +0.365], P 0.000 | +0.273 [+0.176, +0.372], P 0.000 | +0.139 [+0.029, +0.245], P 0.002 | +0.148 [+0.036, +0.258], P 0.002 |
| FLAME 2023 Open vB, coefficienti | +0.154 [+0.085, +0.220], P 0.000 | +0.161 [+0.087, +0.230], P 0.000 | +0.027 [-0.048, +0.097], P 0.229 | +0.035 [-0.036, +0.106], P 0.176 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.156 [+0.083, +0.221], P 0.000 | +0.163 [+0.092, +0.225], P 0.000 | +0.029 [-0.045, +0.092], P 0.225 | +0.037 [-0.038, +0.100], P 0.158 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.074 [-0.129, -0.022], P 0.999 | -0.067 [-0.120, -0.014], P 0.994 | -0.201 [-0.268, -0.140], P 1.000 | -0.193 [-0.256, -0.130], P 1.000 |

## faceverse, emendamento 1 (99000 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto), coefficienti | 0.220 [0.148, 0.286] | 0.182 [0.104, 0.252] |
| GNM (visto), mesh d'identita' FR | 0.227 [0.140, 0.311] | 0.171 [0.082, 0.260] |
| GNM (visto), mesh d'identita' SR | 0.228 [0.145, 0.303] | 0.209 [0.129, 0.285] |
| FLAME 2023 Open, coefficienti | 0.174 [0.109, 0.237] | 0.152 [0.079, 0.218] |
| FLAME 2023 Open, mesh d'identita' FR | 0.211 [0.137, 0.279] | 0.170 [0.089, 0.245] |
| FLAME 2023 Open, mesh d'identita' SR | 0.166 [0.091, 0.235] | 0.155 [0.075, 0.228] |
| GNM (visto) vA, coefficienti | 0.223 [0.145, 0.291] | 0.184 [0.103, 0.254] |
| GNM (visto) vA, mesh d'identita' FR | 0.230 [0.141, 0.316] | 0.174 [0.085, 0.263] |
| GNM (visto) vA, mesh d'identita' SR | 0.219 [0.137, 0.297] | 0.205 [0.124, 0.281] |
| GNM (visto) vB, coefficienti | 0.226 [0.134, 0.299] | 0.222 [0.124, 0.302] |
| GNM (visto) vB, mesh d'identita' FR | 0.292 [0.205, 0.363] | 0.244 [0.149, 0.326] |
| GNM (visto) vB, mesh d'identita' SR | 0.319 [0.235, 0.390] | 0.327 [0.239, 0.402] |
| FLAME 2023 Open vA, coefficienti | 0.174 [0.109, 0.237] | 0.152 [0.079, 0.218] |
| FLAME 2023 Open vA, mesh d'identita' FR | 0.211 [0.137, 0.279] | 0.170 [0.089, 0.245] |
| FLAME 2023 Open vA, mesh d'identita' SR | 0.166 [0.091, 0.235] | 0.155 [0.075, 0.228] |
| FLAME 2023 Open vB, coefficienti | 0.203 [0.121, 0.275] | 0.193 [0.108, 0.271] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.264 [0.183, 0.336] | 0.218 [0.125, 0.302] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.290 [0.216, 0.361] | 0.293 [0.209, 0.368] |
| factorized s1234, d_F cal. | 0.303 [0.229, 0.374] | 0.269 [0.195, 0.341] |
| factorized s2345, d_F cal. | 0.318 [0.249, 0.379] | 0.286 [0.206, 0.353] |
| factorized s1234, d_P | 0.283 [0.210, 0.346] | 0.286 [0.215, 0.352] |
| factorized s2345, d_P | 0.308 [0.234, 0.373] | 0.313 [0.237, 0.377] |
| ctrlfr s1234 | 0.282 [0.215, 0.343] | 0.273 [0.201, 0.339] |
| ctrlfr s2345 | 0.258 [0.187, 0.324] | 0.259 [0.188, 0.325] |

Delta appaiati, braccio - concorrente delle varianti (IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vA, coefficienti | +0.080 [-0.019, +0.186], P 0.061 | +0.095 [+0.002, +0.191], P 0.022 | +0.060 [-0.026, +0.154], P 0.088 | +0.035 [-0.069, +0.144], P 0.255 |
| GNM (visto) vA, mesh d'identita' FR | +0.073 [-0.020, +0.173], P 0.059 | +0.088 [-0.004, +0.181], P 0.034 | +0.052 [-0.051, +0.153], P 0.154 | +0.028 [-0.079, +0.137], P 0.325 |
| GNM (visto) vA, mesh d'identita' SR | +0.083 [-0.022, +0.189], P 0.060 | +0.099 [-0.003, +0.202], P 0.031 | +0.063 [-0.034, +0.165], P 0.110 | +0.038 [-0.070, +0.149], P 0.261 |
| GNM (visto) vB, coefficienti | +0.077 [-0.019, +0.178], P 0.046 | +0.092 [+0.008, +0.183], P 0.014 | +0.057 [-0.019, +0.135], P 0.075 | +0.032 [-0.046, +0.117], P 0.214 |
| GNM (visto) vB, mesh d'identita' FR | +0.010 [-0.062, +0.090], P 0.384 | +0.026 [-0.042, +0.098], P 0.228 | -0.010 [-0.096, +0.073], P 0.606 | -0.035 [-0.122, +0.051], P 0.753 |
| GNM (visto) vB, mesh d'identita' SR | -0.016 [-0.094, +0.071], P 0.624 | -0.000 [-0.075, +0.081], P 0.485 | -0.036 [-0.107, +0.035], P 0.836 | -0.061 [-0.137, +0.013], P 0.937 |
| FLAME 2023 Open vA, coefficienti | +0.128 [+0.036, +0.221], P 0.003 | +0.144 [+0.062, +0.227], P 0.000 | +0.108 [+0.019, +0.192], P 0.005 | +0.083 [-0.012, +0.175], P 0.046 |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.091 [+0.005, +0.177], P 0.017 | +0.107 [+0.029, +0.188], P 0.004 | +0.071 [-0.026, +0.161], P 0.063 | +0.047 [-0.051, +0.143], P 0.164 |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.137 [+0.042, +0.245], P 0.004 | +0.152 [+0.059, +0.252], P 0.001 | +0.117 [+0.025, +0.208], P 0.002 | +0.092 [-0.013, +0.194], P 0.037 |
| FLAME 2023 Open vB, coefficienti | +0.100 [+0.002, +0.195], P 0.020 | +0.115 [+0.035, +0.200], P 0.002 | +0.079 [-0.001, +0.161], P 0.028 | +0.055 [-0.028, +0.140], P 0.107 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.038 [-0.042, +0.129], P 0.190 | +0.054 [-0.019, +0.134], P 0.090 | +0.018 [-0.079, +0.108], P 0.356 | -0.006 [-0.096, +0.084], P 0.553 |
| FLAME 2023 Open vB, mesh d'identita' SR | +0.013 [-0.079, +0.111], P 0.384 | +0.029 [-0.058, +0.116], P 0.257 | -0.007 [-0.089, +0.075], P 0.570 | -0.032 [-0.118, +0.059], P 0.769 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vA, coefficienti | +0.102 [+0.014, +0.199], P 0.013 | +0.129 [+0.038, +0.222], P 0.003 | +0.089 [+0.008, +0.180], P 0.020 | +0.075 [-0.015, +0.175], P 0.048 |
| GNM (visto) vA, mesh d'identita' FR | +0.112 [+0.010, +0.219], P 0.018 | +0.139 [+0.036, +0.239], P 0.006 | +0.099 [-0.009, +0.199], P 0.031 | +0.085 [-0.016, +0.189], P 0.053 |
| GNM (visto) vA, mesh d'identita' SR | +0.081 [-0.010, +0.183], P 0.045 | +0.108 [+0.006, +0.209], P 0.018 | +0.068 [-0.027, +0.166], P 0.090 | +0.054 [-0.050, +0.158], P 0.155 |
| GNM (visto) vB, coefficienti | +0.064 [-0.022, +0.158], P 0.066 | +0.091 [+0.009, +0.185], P 0.016 | +0.051 [-0.028, +0.133], P 0.093 | +0.037 [-0.040, +0.125], P 0.186 |
| GNM (visto) vB, mesh d'identita' FR | +0.042 [-0.048, +0.135], P 0.175 | +0.069 [-0.015, +0.147], P 0.044 | +0.029 [-0.057, +0.117], P 0.258 | +0.015 [-0.070, +0.101], P 0.382 |
| GNM (visto) vB, mesh d'identita' SR | -0.041 [-0.112, +0.033], P 0.863 | -0.014 [-0.085, +0.058], P 0.635 | -0.055 [-0.127, +0.019], P 0.927 | -0.068 [-0.144, +0.006], P 0.964 |
| FLAME 2023 Open vA, coefficienti | +0.134 [+0.046, +0.224], P 0.002 | +0.161 [+0.079, +0.245], P 0.000 | +0.121 [+0.032, +0.207], P 0.002 | +0.107 [+0.015, +0.196], P 0.011 |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.116 [+0.019, +0.206], P 0.008 | +0.143 [+0.055, +0.225], P 0.001 | +0.103 [+0.002, +0.196], P 0.024 | +0.089 [-0.001, +0.181], P 0.030 |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.131 [+0.040, +0.233], P 0.004 | +0.158 [+0.063, +0.253], P 0.000 | +0.118 [+0.029, +0.218], P 0.004 | +0.104 [+0.001, +0.208], P 0.024 |
| FLAME 2023 Open vB, coefficienti | +0.093 [+0.008, +0.182], P 0.017 | +0.120 [+0.041, +0.209], P 0.002 | +0.080 [-0.006, +0.166], P 0.034 | +0.066 [-0.014, +0.151], P 0.059 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.068 [-0.031, +0.168], P 0.093 | +0.095 [+0.009, +0.175], P 0.018 | +0.054 [-0.045, +0.151], P 0.149 | +0.041 [-0.053, +0.134], P 0.200 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.006 [-0.090, +0.084], P 0.558 | +0.021 [-0.069, +0.107], P 0.312 | -0.020 [-0.106, +0.068], P 0.699 | -0.034 [-0.121, +0.054], P 0.789 |

## faceverse_neutral, emendamento 1 (99000 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto), coefficienti | 0.240 [0.164, 0.309] | 0.207 [0.127, 0.277] |
| GNM (visto), mesh d'identita' FR | 0.232 [0.140, 0.318] | 0.174 [0.078, 0.265] |
| GNM (visto), mesh d'identita' SR | 0.240 [0.157, 0.320] | 0.228 [0.145, 0.308] |
| FLAME 2023 Open, coefficienti | 0.171 [0.102, 0.241] | 0.150 [0.073, 0.223] |
| FLAME 2023 Open, mesh d'identita' FR | 0.214 [0.134, 0.291] | 0.172 [0.086, 0.255] |
| FLAME 2023 Open, mesh d'identita' SR | 0.169 [0.089, 0.244] | 0.162 [0.078, 0.240] |
| GNM (visto) vA, coefficienti | 0.230 [0.152, 0.304] | 0.211 [0.130, 0.284] |
| GNM (visto) vA, mesh d'identita' FR | 0.236 [0.142, 0.321] | 0.179 [0.080, 0.273] |
| GNM (visto) vA, mesh d'identita' SR | 0.222 [0.135, 0.301] | 0.222 [0.138, 0.300] |
| GNM (visto) vB, coefficienti | 0.299 [0.197, 0.384] | 0.304 [0.202, 0.390] |
| GNM (visto) vB, mesh d'identita' FR | 0.321 [0.225, 0.400] | 0.275 [0.171, 0.366] |
| GNM (visto) vB, mesh d'identita' SR | 0.373 [0.288, 0.449] | 0.399 [0.315, 0.473] |
| FLAME 2023 Open vA, coefficienti | 0.179 [0.108, 0.249] | 0.156 [0.077, 0.229] |
| FLAME 2023 Open vA, mesh d'identita' FR | 0.220 [0.134, 0.299] | 0.174 [0.081, 0.260] |
| FLAME 2023 Open vA, mesh d'identita' SR | 0.175 [0.094, 0.248] | 0.171 [0.087, 0.249] |
| FLAME 2023 Open vB, coefficienti | 0.249 [0.143, 0.341] | 0.268 [0.159, 0.360] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.315 [0.223, 0.393] | 0.274 [0.173, 0.366] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.358 [0.275, 0.438] | 0.382 [0.297, 0.459] |
| factorized s1234, d_F cal. | 0.341 [0.263, 0.416] | 0.308 [0.225, 0.387] |
| factorized s2345, d_F cal. | 0.370 [0.298, 0.440] | 0.336 [0.252, 0.409] |
| factorized s1234, d_P | 0.321 [0.237, 0.391] | 0.333 [0.251, 0.405] |
| factorized s2345, d_P | 0.361 [0.276, 0.439] | 0.372 [0.291, 0.444] |
| ctrlfr s1234 | 0.327 [0.248, 0.399] | 0.326 [0.243, 0.400] |
| ctrlfr s2345 | 0.299 [0.214, 0.374] | 0.305 [0.224, 0.377] |

Delta appaiati, braccio - concorrente delle varianti (IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vA, coefficienti | +0.110 [+0.005, +0.208], P 0.021 | +0.140 [+0.049, +0.227], P 0.001 | +0.097 [+0.010, +0.187], P 0.018 | +0.069 [-0.040, +0.173], P 0.095 |
| GNM (visto) vA, mesh d'identita' FR | +0.104 [+0.002, +0.210], P 0.024 | +0.134 [+0.041, +0.228], P 0.003 | +0.091 [-0.017, +0.195], P 0.054 | +0.063 [-0.047, +0.166], P 0.125 |
| GNM (visto) vA, mesh d'identita' SR | +0.119 [+0.011, +0.233], P 0.017 | +0.148 [+0.048, +0.252], P 0.000 | +0.105 [+0.001, +0.205], P 0.025 | +0.077 [-0.041, +0.186], P 0.095 |
| GNM (visto) vB, coefficienti | +0.041 [-0.060, +0.152], P 0.203 | +0.071 [-0.030, +0.175], P 0.082 | +0.028 [-0.059, +0.118], P 0.257 | -0.000 [-0.101, +0.103], P 0.492 |
| GNM (visto) vB, mesh d'identita' FR | +0.020 [-0.070, +0.118], P 0.329 | +0.050 [-0.033, +0.138], P 0.113 | +0.006 [-0.089, +0.099], P 0.424 | -0.022 [-0.123, +0.074], P 0.639 |
| GNM (visto) vB, mesh d'identita' SR | -0.032 [-0.121, +0.065], P 0.728 | -0.002 [-0.089, +0.092], P 0.525 | -0.046 [-0.129, +0.037], P 0.852 | -0.074 [-0.165, +0.015], P 0.941 |
| FLAME 2023 Open vA, coefficienti | +0.162 [+0.057, +0.267], P 0.003 | +0.192 [+0.097, +0.281], P 0.000 | +0.149 [+0.049, +0.244], P 0.002 | +0.121 [+0.010, +0.220], P 0.015 |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.120 [+0.012, +0.221], P 0.016 | +0.150 [+0.057, +0.240], P 0.002 | +0.107 [+0.001, +0.206], P 0.025 | +0.079 [-0.026, +0.183], P 0.085 |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.166 [+0.060, +0.278], P 0.002 | +0.196 [+0.095, +0.298], P 0.000 | +0.152 [+0.053, +0.251], P 0.000 | +0.124 [+0.009, +0.227], P 0.013 |
| FLAME 2023 Open vB, coefficienti | +0.092 [-0.014, +0.204], P 0.053 | +0.122 [+0.018, +0.230], P 0.009 | +0.079 [-0.022, +0.173], P 0.049 | +0.051 [-0.054, +0.151], P 0.163 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.026 [-0.070, +0.133], P 0.305 | +0.055 [-0.035, +0.148], P 0.114 | +0.012 [-0.086, +0.105], P 0.395 | -0.016 [-0.117, +0.082], P 0.612 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.018 [-0.120, +0.088], P 0.622 | +0.012 [-0.089, +0.115], P 0.400 | -0.031 [-0.124, +0.064], P 0.748 | -0.059 [-0.159, +0.040], P 0.878 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vA, coefficienti | +0.123 [+0.018, +0.219], P 0.010 | +0.161 [+0.068, +0.247], P 0.001 | +0.115 [+0.031, +0.204], P 0.006 | +0.094 [-0.007, +0.185], P 0.032 |
| GNM (visto) vA, mesh d'identita' FR | +0.154 [+0.043, +0.261], P 0.005 | +0.193 [+0.089, +0.292], P 0.001 | +0.147 [+0.042, +0.248], P 0.008 | +0.126 [+0.029, +0.222], P 0.010 |
| GNM (visto) vA, mesh d'identita' SR | +0.112 [+0.014, +0.217], P 0.015 | +0.150 [+0.043, +0.248], P 0.000 | +0.105 [+0.003, +0.202], P 0.024 | +0.084 [-0.018, +0.179], P 0.066 |
| GNM (visto) vB, coefficienti | +0.030 [-0.059, +0.130], P 0.252 | +0.068 [-0.036, +0.172], P 0.082 | +0.023 [-0.072, +0.117], P 0.306 | +0.002 [-0.099, +0.102], P 0.484 |
| GNM (visto) vB, mesh d'identita' FR | +0.059 [-0.040, +0.157], P 0.125 | +0.097 [+0.003, +0.190], P 0.022 | +0.051 [-0.047, +0.147], P 0.136 | +0.031 [-0.061, +0.119], P 0.252 |
| GNM (visto) vB, mesh d'identita' SR | -0.065 [-0.142, +0.012], P 0.941 | -0.027 [-0.110, +0.053], P 0.738 | -0.073 [-0.156, +0.007], P 0.958 | -0.093 [-0.181, -0.011], P 0.989 |
| FLAME 2023 Open vA, coefficienti | +0.178 [+0.071, +0.276], P 0.001 | +0.216 [+0.118, +0.312], P 0.000 | +0.171 [+0.065, +0.267], P 0.001 | +0.150 [+0.042, +0.245], P 0.001 |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.160 [+0.043, +0.263], P 0.006 | +0.198 [+0.092, +0.291], P 0.001 | +0.152 [+0.042, +0.250], P 0.004 | +0.132 [+0.034, +0.226], P 0.007 |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.162 [+0.063, +0.267], P 0.000 | +0.201 [+0.097, +0.301], P 0.000 | +0.155 [+0.053, +0.260], P 0.000 | +0.134 [+0.026, +0.236], P 0.006 |
| FLAME 2023 Open vB, coefficienti | +0.066 [-0.032, +0.167], P 0.097 | +0.104 [+0.005, +0.208], P 0.020 | +0.059 [-0.044, +0.159], P 0.127 | +0.038 [-0.061, +0.138], P 0.235 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.059 [-0.046, +0.162], P 0.137 | +0.098 [-0.003, +0.194], P 0.031 | +0.052 [-0.052, +0.154], P 0.156 | +0.031 [-0.070, +0.128], P 0.286 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.049 [-0.129, +0.037], P 0.855 | -0.010 [-0.105, +0.082], P 0.598 | -0.056 [-0.150, +0.036], P 0.879 | -0.077 [-0.175, +0.013], P 0.951 |

## famos, emendamento 1 (105 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto), coefficienti | 0.624 [0.195, 0.883] | 0.418 [-0.012, 0.772] |
| GNM (visto), mesh d'identita' FR | 0.586 [0.145, 0.859] | 0.274 [-0.208, 0.690] |
| GNM (visto), mesh d'identita' SR | 0.460 [0.139, 0.720] | 0.547 [0.188, 0.825] |
| FLAME 2023 Open, coefficienti | 0.457 [0.064, 0.760] | 0.390 [0.027, 0.713] |
| FLAME 2023 Open, mesh d'identita' FR | 0.565 [0.134, 0.856] | 0.291 [-0.186, 0.691] |
| FLAME 2023 Open, mesh d'identita' SR | 0.217 [-0.108, 0.556] | 0.412 [0.083, 0.748] |
| GNM (visto) vA, coefficienti | 0.619 [0.193, 0.878] | 0.438 [0.024, 0.780] |
| GNM (visto) vA, mesh d'identita' FR | 0.593 [0.161, 0.870] | 0.279 [-0.201, 0.701] |
| GNM (visto) vA, mesh d'identita' SR | 0.234 [-0.089, 0.614] | 0.439 [0.075, 0.803] |
| GNM (visto) vB, coefficienti | 0.606 [0.167, 0.836] | 0.745 [0.418, 0.882] |
| GNM (visto) vB, mesh d'identita' FR | 0.686 [0.307, 0.910] | 0.439 [-0.001, 0.794] |
| GNM (visto) vB, mesh d'identita' SR | 0.513 [0.194, 0.749] | 0.766 [0.592, 0.880] |
| FLAME 2023 Open vA, coefficienti | 0.480 [0.065, 0.785] | 0.383 [-0.006, 0.712] |
| FLAME 2023 Open vA, mesh d'identita' FR | 0.579 [0.153, 0.856] | 0.279 [-0.198, 0.682] |
| FLAME 2023 Open vA, mesh d'identita' SR | 0.161 [-0.175, 0.522] | 0.373 [0.039, 0.741] |
| FLAME 2023 Open vB, coefficienti | 0.666 [0.269, 0.878] | 0.714 [0.431, 0.877] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.668 [0.292, 0.908] | 0.483 [0.093, 0.819] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.481 [0.160, 0.714] | 0.791 [0.626, 0.889] |
| factorized s1234, d_F cal. | 0.678 [0.237, 0.897] | 0.662 [0.323, 0.858] |
| factorized s2345, d_F cal. | 0.654 [0.214, 0.896] | 0.616 [0.225, 0.839] |
| factorized s1234, d_P | 0.412 [0.088, 0.680] | 0.740 [0.548, 0.883] |
| factorized s2345, d_P | 0.437 [0.107, 0.698] | 0.728 [0.516, 0.869] |
| ctrlfr s1234 | 0.831 [0.575, 0.949] | 0.709 [0.394, 0.871] |
| ctrlfr s2345 | 0.818 [0.569, 0.944] | 0.691 [0.378, 0.869] |

Delta appaiati, braccio - concorrente delle varianti (IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vA, coefficienti | +0.059 [-0.201, +0.281], P 0.287 | +0.035 [-0.197, +0.227], P 0.342 | +0.212 [-0.017, +0.523], P 0.038 | +0.199 [-0.037, +0.536], P 0.053 |
| GNM (visto) vA, mesh d'identita' FR | +0.085 [-0.184, +0.378], P 0.256 | +0.062 [-0.171, +0.301], P 0.272 | +0.238 [+0.009, +0.577], P 0.024 | +0.226 [-0.009, +0.579], P 0.033 |
| GNM (visto) vA, mesh d'identita' SR | +0.444 [-0.045, +0.830], P 0.037 | +0.420 [-0.078, +0.832], P 0.075 | +0.597 [+0.161, +0.950], P 0.002 | +0.584 [+0.162, +0.946], P 0.001 |
| GNM (visto) vB, coefficienti | +0.072 [-0.183, +0.403], P 0.274 | +0.048 [-0.198, +0.368], P 0.337 | +0.225 [+0.040, +0.561], P 0.009 | +0.212 [+0.047, +0.522], P 0.010 |
| GNM (visto) vB, mesh d'identita' FR | -0.008 [-0.231, +0.199], P 0.560 | -0.031 [-0.227, +0.123], P 0.675 | +0.145 [-0.053, +0.423], P 0.077 | +0.133 [-0.077, +0.421], P 0.114 |
| GNM (visto) vB, mesh d'identita' SR | +0.165 [-0.177, +0.483], P 0.199 | +0.141 [-0.226, +0.474], P 0.242 | +0.318 [+0.035, +0.618], P 0.017 | +0.305 [+0.027, +0.582], P 0.018 |
| FLAME 2023 Open vA, coefficienti | +0.198 [-0.045, +0.438], P 0.051 | +0.175 [-0.065, +0.403], P 0.073 | +0.351 [+0.094, +0.641], P 0.005 | +0.339 [+0.073, +0.667], P 0.008 |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.099 [-0.182, +0.378], P 0.222 | +0.075 [-0.169, +0.304], P 0.248 | +0.252 [+0.015, +0.594], P 0.019 | +0.239 [+0.007, +0.590], P 0.020 |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.517 [+0.033, +0.899], P 0.022 | +0.493 [-0.018, +0.905], P 0.031 | +0.670 [+0.221, +1.036], P 0.001 | +0.657 [+0.225, +1.025], P 0.000 |
| FLAME 2023 Open vB, coefficienti | +0.012 [-0.227, +0.268], P 0.455 | -0.011 [-0.229, +0.247], P 0.564 | +0.165 [+0.006, +0.451], P 0.018 | +0.153 [-0.006, +0.408], P 0.033 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.010 [-0.213, +0.189], P 0.506 | -0.014 [-0.220, +0.139], P 0.598 | +0.163 [-0.035, +0.425], P 0.068 | +0.150 [-0.065, +0.424], P 0.096 |
| FLAME 2023 Open vB, mesh d'identita' SR | +0.197 [-0.112, +0.501], P 0.133 | +0.173 [-0.168, +0.506], P 0.184 | +0.350 [+0.071, +0.618], P 0.010 | +0.337 [+0.062, +0.597], P 0.008 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vA, coefficienti | +0.302 [-0.140, +0.769], P 0.090 | +0.290 [-0.124, +0.758], P 0.088 | +0.271 [+0.036, +0.532], P 0.008 | +0.253 [+0.011, +0.525], P 0.017 |
| GNM (visto) vA, mesh d'identita' FR | +0.462 [-0.040, +1.034], P 0.043 | +0.450 [-0.038, +1.005], P 0.036 | +0.431 [+0.096, +0.790], P 0.004 | +0.412 [+0.095, +0.728], P 0.003 |
| GNM (visto) vA, mesh d'identita' SR | +0.301 [-0.020, +0.681], P 0.031 | +0.289 [-0.066, +0.699], P 0.059 | +0.270 [-0.154, +0.637], P 0.141 | +0.252 [-0.227, +0.640], P 0.181 |
| GNM (visto) vB, coefficienti | -0.005 [-0.257, +0.316], P 0.470 | -0.017 [-0.228, +0.244], P 0.510 | -0.036 [-0.329, +0.199], P 0.618 | -0.054 [-0.289, +0.161], P 0.698 |
| GNM (visto) vB, mesh d'identita' FR | +0.301 [-0.139, +0.804], P 0.098 | +0.289 [-0.120, +0.754], P 0.099 | +0.270 [+0.004, +0.577], P 0.021 | +0.252 [+0.008, +0.531], P 0.021 |
| GNM (visto) vB, mesh d'identita' SR | -0.026 [-0.188, +0.181], P 0.577 | -0.038 [-0.232, +0.182], P 0.647 | -0.057 [-0.346, +0.169], P 0.706 | -0.075 [-0.384, +0.173], P 0.709 |
| FLAME 2023 Open vA, coefficienti | +0.357 [-0.079, +0.779], P 0.047 | +0.345 [-0.056, +0.761], P 0.053 | +0.326 [+0.051, +0.589], P 0.012 | +0.307 [+0.003, +0.582], P 0.024 |
| FLAME 2023 Open vA, mesh d'identita' FR | +0.462 [-0.028, +1.033], P 0.031 | +0.450 [-0.024, +1.003], P 0.031 | +0.431 [+0.111, +0.764], P 0.003 | +0.412 [+0.117, +0.730], P 0.003 |
| FLAME 2023 Open vA, mesh d'identita' SR | +0.368 [+0.025, +0.710], P 0.018 | +0.356 [-0.022, +0.732], P 0.033 | +0.337 [-0.113, +0.696], P 0.093 | +0.318 [-0.193, +0.707], P 0.131 |
| FLAME 2023 Open vB, coefficienti | +0.026 [-0.248, +0.336], P 0.407 | +0.014 [-0.219, +0.287], P 0.452 | -0.005 [-0.203, +0.178], P 0.579 | -0.023 [-0.206, +0.128], P 0.681 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.257 [-0.165, +0.688], P 0.123 | +0.245 [-0.138, +0.648], P 0.122 | +0.226 [-0.021, +0.485], P 0.037 | +0.207 [-0.041, +0.439], P 0.051 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.050 [-0.182, +0.096], P 0.726 | -0.062 [-0.228, +0.095], P 0.778 | -0.081 [-0.382, +0.171], P 0.783 | -0.100 [-0.418, +0.168], P 0.799 |

# Emendamento 2 (POST HOC): B su crop e all_cross, composizione, costi, sensibilita' a sigma

Protocollo `PROTOCOL_emendamento_2.md` (sha256 `91a0fc536357733933cdbe3924277caa3150a9ca86d3d3a4d3a2ce62f6d68072`), scritto dopo i numeri dell'emendamento 1. Numeri in `paired_e2.csv`, `spearman_e2.csv`, `controls_e2.json`, `calib_e2/`, `pilot_e2/`, `cost_e2/`.

**Celle con IC sotto 0 (concorrente davanti al braccio) fra i delta dichiarati dell'emendamento 2:** 169: hifi3d all_cross FR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' FR = -0.135 [-0.184, -0.089]; hifi3d all_cross FR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' FR = -0.121 [-0.167, -0.078]; hifi3d all_cross FR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' FR = -0.154 [-0.197, -0.114]; hifi3d all_cross FR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' FR = -0.140 [-0.180, -0.104]; hifi3d all_cross SR: factorized s1234, d_P - GNM (visto) vB, coefficienti = -0.108 [-0.193, -0.022]; hifi3d all_cross SR: factorized s2345, d_P - GNM (visto) vB, coefficienti = -0.124 [-0.216, -0.032]; hifi3d all_cross SR: ctrlfr s1234 - GNM (visto) vB, coefficienti = -0.420 [-0.493, -0.343]; hifi3d all_cross SR: ctrlfr s2345 - GNM (visto) vB, coefficienti = -0.406 [-0.481, -0.324]; hifi3d all_cross SR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' SR = -0.342 [-0.404, -0.281]; hifi3d all_cross SR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.328 [-0.386, -0.268]; hifi3d all_cross SR: ctrlfr s1234 - FLAME 2023 Open vB, coefficienti = -0.329 [-0.408, -0.249]; hifi3d all_cross SR: ctrlfr s2345 - FLAME 2023 Open vB, coefficienti = -0.314 [-0.398, -0.229]; hifi3d all_cross SR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' SR = -0.341 [-0.401, -0.282]; hifi3d all_cross SR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' SR = -0.326 [-0.385, -0.270]; hifi3d all_cross, righe col crop FR: ctrlfr s1234 - GNM (visto) vB, coefficienti = -0.288 [-0.365, -0.211]; hifi3d all_cross, righe col crop FR: ctrlfr s2345 - GNM (visto) vB, coefficienti = -0.245 [-0.328, -0.160]; hifi3d all_cross, righe col crop FR: factorized s2345, d_F cal. - GNM (visto) vB, mesh d'identita' FR = -0.072 [-0.114, -0.031]; hifi3d all_cross, righe col crop FR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' FR = -0.389 [-0.460, -0.318]; hifi3d all_cross, righe col crop FR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' FR = -0.346 [-0.410, -0.283]; hifi3d all_cross, righe col crop FR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' SR = -0.145 [-0.224, -0.066]; hifi3d all_cross, righe col crop FR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.102 [-0.186, -0.019]; hifi3d all_cross, righe col crop FR: ctrlfr s1234 - FLAME 2023 Open vB, coefficienti = -0.307 [-0.377, -0.234]; hifi3d all_cross, righe col crop FR: ctrlfr s2345 - FLAME 2023 Open vB, coefficienti = -0.264 [-0.334, -0.190]; hifi3d all_cross, righe col crop FR: factorized s2345, d_F cal. - FLAME 2023 Open vB, mesh d'identita' FR = -0.086 [-0.126, -0.048]; hifi3d all_cross, righe col crop FR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' FR = -0.403 [-0.469, -0.334]; hifi3d all_cross, righe col crop FR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' FR = -0.360 [-0.421, -0.301]; hifi3d all_cross, righe col crop FR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' SR = -0.197 [-0.270, -0.123]; hifi3d all_cross, righe col crop FR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' SR = -0.155 [-0.229, -0.082]; hifi3d all_cross, righe col crop SR: factorized s1234, d_P - GNM (visto) vB, coefficienti = -0.101 [-0.186, -0.010]; hifi3d all_cross, righe col crop SR: factorized s2345, d_P - GNM (visto) vB, coefficienti = -0.137 [-0.232, -0.038]; hifi3d all_cross, righe col crop SR: ctrlfr s1234 - GNM (visto) vB, coefficienti = -0.487 [-0.549, -0.418]; hifi3d all_cross, righe col crop SR: ctrlfr s2345 - GNM (visto) vB, coefficienti = -0.462 [-0.526, -0.392]; hifi3d all_cross, righe col crop SR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' SR = -0.406 [-0.467, -0.336]; hifi3d all_cross, righe col crop SR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.381 [-0.439, -0.316]; hifi3d all_cross, righe col crop SR: ctrlfr s1234 - FLAME 2023 Open vB, coefficienti = -0.388 [-0.464, -0.311]; hifi3d all_cross, righe col crop SR: ctrlfr s2345 - FLAME 2023 Open vB, coefficienti = -0.363 [-0.437, -0.287]; hifi3d all_cross, righe col crop SR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' SR = -0.403 [-0.462, -0.336]; hifi3d all_cross, righe col crop SR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' SR = -0.378 [-0.432, -0.316]; facescape all_cross FR: factorized s1234, d_F cal. - GNM (visto) vB, coefficienti = -0.189 [-0.253, -0.120]; facescape all_cross FR: factorized s2345, d_F cal. - GNM (visto) vB, coefficienti = -0.208 [-0.269, -0.139]; facescape all_cross FR: ctrlfr s1234 - GNM (visto) vB, coefficienti = -0.225 [-0.293, -0.153]; facescape all_cross FR: ctrlfr s2345 - GNM (visto) vB, coefficienti = -0.236 [-0.299, -0.165]; facescape all_cross FR: factorized s1234, d_F cal. - GNM (visto) vB, mesh d'identita' FR = -0.291 [-0.344, -0.234]; facescape all_cross FR: factorized s2345, d_F cal. - GNM (visto) vB, mesh d'identita' FR = -0.310 [-0.358, -0.257]; facescape all_cross FR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' FR = -0.327 [-0.378, -0.273]; facescape all_cross FR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' FR = -0.338 [-0.388, -0.283]; facescape all_cross FR: factorized s1234, d_F cal. - GNM (visto) vB, mesh d'identita' SR = -0.387 [-0.445, -0.321]; facescape all_cross FR: factorized s2345, d_F cal. - GNM (visto) vB, mesh d'identita' SR = -0.406 [-0.466, -0.337]; facescape all_cross FR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' SR = -0.424 [-0.491, -0.348]; facescape all_cross FR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.435 [-0.497, -0.359]; facescape all_cross FR: factorized s1234, d_F cal. - FLAME 2023 Open vB, coefficienti = -0.165 [-0.227, -0.096]; facescape all_cross FR: factorized s2345, d_F cal. - FLAME 2023 Open vB, coefficienti = -0.184 [-0.247, -0.117]; facescape all_cross FR: ctrlfr s1234 - FLAME 2023 Open vB, coefficienti = -0.201 [-0.267, -0.129]; facescape all_cross FR: ctrlfr s2345 - FLAME 2023 Open vB, coefficienti = -0.212 [-0.274, -0.141]; facescape all_cross FR: factorized s1234, d_F cal. - FLAME 2023 Open vB, mesh d'identita' FR = -0.236 [-0.289, -0.180]; facescape all_cross FR: factorized s2345, d_F cal. - FLAME 2023 Open vB, mesh d'identita' FR = -0.255 [-0.304, -0.201]; facescape all_cross FR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' FR = -0.272 [-0.327, -0.217]; facescape all_cross FR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' FR = -0.283 [-0.335, -0.227]; facescape all_cross FR: factorized s1234, d_F cal. - FLAME 2023 Open vB, mesh d'identita' SR = -0.336 [-0.404, -0.260]; facescape all_cross FR: factorized s2345, d_F cal. - FLAME 2023 Open vB, mesh d'identita' SR = -0.355 [-0.424, -0.275]; facescape all_cross FR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' SR = -0.373 [-0.454, -0.282]; facescape all_cross FR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' SR = -0.384 [-0.459, -0.295]; facescape all_cross SR: factorized s1234, d_P - GNM (visto) vB, coefficienti = -0.122 [-0.186, -0.056]; facescape all_cross SR: factorized s2345, d_P - GNM (visto) vB, coefficienti = -0.147 [-0.212, -0.081]; facescape all_cross SR: ctrlfr s1234 - GNM (visto) vB, coefficienti = -0.296 [-0.355, -0.235]; facescape all_cross SR: ctrlfr s2345 - GNM (visto) vB, coefficienti = -0.296 [-0.353, -0.237]; facescape all_cross SR: factorized s1234, d_P - GNM (visto) vB, mesh d'identita' FR = -0.104 [-0.168, -0.040]; facescape all_cross SR: factorized s2345, d_P - GNM (visto) vB, mesh d'identita' FR = -0.129 [-0.191, -0.069]; facescape all_cross SR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' FR = -0.278 [-0.338, -0.224]; facescape all_cross SR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' FR = -0.278 [-0.338, -0.222]; facescape all_cross SR: factorized s1234, d_P - GNM (visto) vB, mesh d'identita' SR = -0.351 [-0.402, -0.301]; facescape all_cross SR: factorized s2345, d_P - GNM (visto) vB, mesh d'identita' SR = -0.376 [-0.424, -0.330]; facescape all_cross SR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' SR = -0.525 [-0.562, -0.488]; facescape all_cross SR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.525 [-0.562, -0.486]; facescape all_cross SR: factorized s1234, d_P - FLAME 2023 Open vB, coefficienti = -0.087 [-0.151, -0.025]; facescape all_cross SR: factorized s2345, d_P - FLAME 2023 Open vB, coefficienti = -0.112 [-0.175, -0.048]; facescape all_cross SR: ctrlfr s1234 - FLAME 2023 Open vB, coefficienti = -0.261 [-0.321, -0.200]; facescape all_cross SR: ctrlfr s2345 - FLAME 2023 Open vB, coefficienti = -0.261 [-0.319, -0.201]; facescape all_cross SR: factorized s1234, d_P - FLAME 2023 Open vB, mesh d'identita' FR = -0.079 [-0.140, -0.015]; facescape all_cross SR: factorized s2345, d_P - FLAME 2023 Open vB, mesh d'identita' FR = -0.105 [-0.163, -0.044]; facescape all_cross SR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' FR = -0.254 [-0.309, -0.199]; facescape all_cross SR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' FR = -0.254 [-0.307, -0.200]; facescape all_cross SR: factorized s1234, d_P - FLAME 2023 Open vB, mesh d'identita' SR = -0.325 [-0.382, -0.269]; facescape all_cross SR: factorized s2345, d_P - FLAME 2023 Open vB, mesh d'identita' SR = -0.350 [-0.406, -0.298]; facescape all_cross SR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' SR = -0.499 [-0.547, -0.448]; facescape all_cross SR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' SR = -0.499 [-0.545, -0.446]; facescape all_cross, righe col crop FR: factorized s1234, d_F cal. - GNM (visto) vB, coefficienti = -0.198 [-0.285, -0.111]; facescape all_cross, righe col crop FR: factorized s2345, d_F cal. - GNM (visto) vB, coefficienti = -0.271 [-0.355, -0.184]; facescape all_cross, righe col crop FR: ctrlfr s1234 - GNM (visto) vB, coefficienti = -0.285 [-0.351, -0.215]; facescape all_cross, righe col crop FR: ctrlfr s2345 - GNM (visto) vB, coefficienti = -0.315 [-0.386, -0.240]; facescape all_cross, righe col crop FR: factorized s1234, d_F cal. - GNM (visto) vB, mesh d'identita' FR = -0.277 [-0.347, -0.204]; facescape all_cross, righe col crop FR: factorized s2345, d_F cal. - GNM (visto) vB, mesh d'identita' FR = -0.350 [-0.416, -0.281]; facescape all_cross, righe col crop FR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' FR = -0.364 [-0.421, -0.308]; facescape all_cross, righe col crop FR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' FR = -0.394 [-0.454, -0.331]; facescape all_cross, righe col crop FR: factorized s1234, d_F cal. - GNM (visto) vB, mesh d'identita' SR = -0.404 [-0.470, -0.338]; facescape all_cross, righe col crop FR: factorized s2345, d_F cal. - GNM (visto) vB, mesh d'identita' SR = -0.477 [-0.547, -0.404]; facescape all_cross, righe col crop FR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' SR = -0.491 [-0.568, -0.405]; facescape all_cross, righe col crop FR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.521 [-0.593, -0.443]; facescape all_cross, righe col crop FR: factorized s1234, d_F cal. - FLAME 2023 Open vB, coefficienti = -0.163 [-0.256, -0.076]; facescape all_cross, righe col crop FR: factorized s2345, d_F cal. - FLAME 2023 Open vB, coefficienti = -0.236 [-0.322, -0.145]; facescape all_cross, righe col crop FR: ctrlfr s1234 - FLAME 2023 Open vB, coefficienti = -0.250 [-0.315, -0.181]; facescape all_cross, righe col crop FR: ctrlfr s2345 - FLAME 2023 Open vB, coefficienti = -0.280 [-0.347, -0.209]; facescape all_cross, righe col crop FR: factorized s1234, d_F cal. - FLAME 2023 Open vB, mesh d'identita' FR = -0.210 [-0.281, -0.138]; facescape all_cross, righe col crop FR: factorized s2345, d_F cal. - FLAME 2023 Open vB, mesh d'identita' FR = -0.283 [-0.348, -0.209]; facescape all_cross, righe col crop FR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' FR = -0.297 [-0.353, -0.237]; facescape all_cross, righe col crop FR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' FR = -0.326 [-0.385, -0.265]; facescape all_cross, righe col crop FR: factorized s1234, d_F cal. - FLAME 2023 Open vB, mesh d'identita' SR = -0.352 [-0.426, -0.276]; facescape all_cross, righe col crop FR: factorized s2345, d_F cal. - FLAME 2023 Open vB, mesh d'identita' SR = -0.425 [-0.506, -0.336]; facescape all_cross, righe col crop FR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' SR = -0.439 [-0.527, -0.338]; facescape all_cross, righe col crop FR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' SR = -0.469 [-0.550, -0.372]; facescape all_cross, righe col crop SR: factorized s1234, d_P - GNM (visto) vB, coefficienti = -0.083 [-0.161, -0.001]; facescape all_cross, righe col crop SR: factorized s2345, d_P - GNM (visto) vB, coefficienti = -0.125 [-0.211, -0.036]; facescape all_cross, righe col crop SR: ctrlfr s1234 - GNM (visto) vB, coefficienti = -0.354 [-0.418, -0.286]; facescape all_cross, righe col crop SR: ctrlfr s2345 - GNM (visto) vB, coefficienti = -0.358 [-0.419, -0.285]; facescape all_cross, righe col crop SR: factorized s2345, d_P - GNM (visto) vB, mesh d'identita' FR = -0.090 [-0.168, -0.013]; facescape all_cross, righe col crop SR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' FR = -0.319 [-0.381, -0.257]; facescape all_cross, righe col crop SR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' FR = -0.323 [-0.390, -0.256]; facescape all_cross, righe col crop SR: factorized s1234, d_P - GNM (visto) vB, mesh d'identita' SR = -0.312 [-0.383, -0.250]; facescape all_cross, righe col crop SR: factorized s2345, d_P - GNM (visto) vB, mesh d'identita' SR = -0.354 [-0.428, -0.289]; facescape all_cross, righe col crop SR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' SR = -0.583 [-0.639, -0.524]; facescape all_cross, righe col crop SR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.587 [-0.643, -0.526]; facescape all_cross, righe col crop SR: ctrlfr s1234 - FLAME 2023 Open vB, coefficienti = -0.306 [-0.373, -0.243]; facescape all_cross, righe col crop SR: ctrlfr s2345 - FLAME 2023 Open vB, coefficienti = -0.310 [-0.376, -0.242]; facescape all_cross, righe col crop SR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' FR = -0.282 [-0.338, -0.224]; facescape all_cross, righe col crop SR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' FR = -0.286 [-0.343, -0.226]; facescape all_cross, righe col crop SR: factorized s1234, d_P - FLAME 2023 Open vB, mesh d'identita' SR = -0.289 [-0.364, -0.220]; facescape all_cross, righe col crop SR: factorized s2345, d_P - FLAME 2023 Open vB, mesh d'identita' SR = -0.330 [-0.410, -0.257]; facescape all_cross, righe col crop SR: ctrlfr s1234 - FLAME 2023 Open vB, mesh d'identita' SR = -0.560 [-0.622, -0.492]; facescape all_cross, righe col crop SR: ctrlfr s2345 - FLAME 2023 Open vB, mesh d'identita' SR = -0.564 [-0.624, -0.496]; faceverse_neutral all_cross SR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.105 [-0.184, -0.016]; faceverse_neutral all_cross, righe col crop FR: factorized s1234, d_F cal. - GNM (visto) vB, mesh d'identita' SR = -0.096 [-0.183, -0.006]; faceverse_neutral all_cross, righe col crop FR: factorized s2345, d_F cal. - GNM (visto) vB, mesh d'identita' SR = -0.106 [-0.193, -0.023]; faceverse_neutral all_cross, righe col crop FR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.110 [-0.198, -0.023]; faceverse_neutral all_cross, righe col crop SR: factorized s1234, d_P - GNM (visto) vB, mesh d'identita' SR = -0.098 [-0.181, -0.014]; faceverse_neutral all_cross, righe col crop SR: factorized s2345, d_P - GNM (visto) vB, mesh d'identita' SR = -0.086 [-0.170, -0.002]; faceverse_neutral all_cross, righe col crop SR: ctrlfr s1234 - GNM (visto) vB, mesh d'identita' SR = -0.111 [-0.201, -0.023]; faceverse_neutral all_cross, righe col crop SR: ctrlfr s2345 - GNM (visto) vB, mesh d'identita' SR = -0.128 [-0.209, -0.043]; hifi3d sens. SR: ctrlfr s1234 - GNM (visto) vB sigma sens., coefficienti = -0.229 [-0.315, -0.145]; hifi3d sens. SR: ctrlfr s2345 - GNM (visto) vB sigma sens., coefficienti = -0.220 [-0.303, -0.140]; hifi3d sens. SR: ctrlfr s1234 - GNM (visto) vB sigma sens., mesh d'identita' SR = -0.256 [-0.353, -0.180]; hifi3d sens. SR: ctrlfr s2345 - GNM (visto) vB sigma sens., mesh d'identita' SR = -0.247 [-0.334, -0.173]; hifi3d sens. SR: ctrlfr s1234 - FLAME 2023 Open vB sigma sens., coefficienti = -0.182 [-0.253, -0.116]; hifi3d sens. SR: ctrlfr s2345 - FLAME 2023 Open vB sigma sens., coefficienti = -0.173 [-0.242, -0.110]; hifi3d sens. SR: ctrlfr s1234 - FLAME 2023 Open vB sigma sens., mesh d'identita' SR = -0.208 [-0.288, -0.130]; hifi3d sens. SR: ctrlfr s2345 - FLAME 2023 Open vB sigma sens., mesh d'identita' SR = -0.199 [-0.274, -0.126]; facescape composizione FR: factorized s1234, d_F cal. - GNM (visto) vB, composizione S_B + k d_P = -0.139 [-0.194, -0.090]; facescape composizione FR: factorized s2345, d_F cal. - GNM (visto) vB, composizione S_B + k d_P = -0.131 [-0.184, -0.086]; facescape composizione FR: ctrlfr s1234 - GNM (visto) vB, composizione S_B + k d_P = -0.138 [-0.203, -0.082]; facescape composizione FR: ctrlfr s2345 - GNM (visto) vB, composizione S_B + k d_P = -0.147 [-0.208, -0.092]; facescape composizione FR: factorized s1234, d_F cal. - FLAME 2023 Open vB, composizione S_B + k d_P = -0.070 [-0.127, -0.017]; facescape composizione FR: factorized s2345, d_F cal. - FLAME 2023 Open vB, composizione S_B + k d_P = -0.062 [-0.123, -0.009]; facescape composizione FR: ctrlfr s1234 - FLAME 2023 Open vB, composizione S_B + k d_P = -0.070 [-0.149, -0.001]; facescape composizione FR: ctrlfr s2345 - FLAME 2023 Open vB, composizione S_B + k d_P = -0.078 [-0.150, -0.015]; facescape sens. FR: factorized s1234, d_F cal. - GNM (visto) vB sigma sens., mesh d'identita' SR = -0.122 [-0.192, -0.040]; facescape sens. FR: factorized s2345, d_F cal. - GNM (visto) vB sigma sens., mesh d'identita' SR = -0.114 [-0.187, -0.032]; facescape sens. FR: ctrlfr s1234 - GNM (visto) vB sigma sens., mesh d'identita' SR = -0.121 [-0.210, -0.036]; facescape sens. FR: ctrlfr s2345 - GNM (visto) vB sigma sens., mesh d'identita' SR = -0.130 [-0.213, -0.038]; facescape sens. SR: factorized s1234, d_P - GNM (visto) vB sigma sens., mesh d'identita' SR = -0.126 [-0.174, -0.086]; facescape sens. SR: factorized s2345, d_P - GNM (visto) vB sigma sens., mesh d'identita' SR = -0.120 [-0.168, -0.080]; facescape sens. SR: ctrlfr s1234 - GNM (visto) vB sigma sens., mesh d'identita' SR = -0.254 [-0.313, -0.203]; facescape sens. SR: ctrlfr s2345 - GNM (visto) vB sigma sens., mesh d'identita' SR = -0.245 [-0.302, -0.199]; facescape sens. SR: ctrlfr s1234 - FLAME 2023 Open vB sigma sens., coefficienti = -0.082 [-0.148, -0.019]; facescape sens. SR: ctrlfr s2345 - FLAME 2023 Open vB sigma sens., coefficienti = -0.073 [-0.144, -0.005]; facescape sens. SR: factorized s1234, d_P - FLAME 2023 Open vB sigma sens., mesh d'identita' SR = -0.097 [-0.154, -0.046]; facescape sens. SR: factorized s2345, d_P - FLAME 2023 Open vB sigma sens., mesh d'identita' SR = -0.090 [-0.146, -0.037]; facescape sens. SR: ctrlfr s1234 - FLAME 2023 Open vB sigma sens., mesh d'identita' SR = -0.224 [-0.294, -0.163]; facescape sens. SR: ctrlfr s2345 - FLAME 2023 Open vB sigma sens., mesh d'identita' SR = -0.216 [-0.283, -0.155]; faceverse sens. SR: ctrlfr s2345 - GNM (visto) vB sigma sens., mesh d'identita' SR = -0.081 [-0.158, -0.006]; faceverse_neutral sens. SR: ctrlfr s2345 - GNM (visto) vB sigma sens., mesh d'identita' SR = -0.093 [-0.183, -0.006].

## Valori senza GT di test (sez. 9)

```
{
 "k": {
  "gnm": {
   "k_median": 1.6980957571420483,
   "k_ls": 1.5767387405365794,
   "n_pairs": 297000,
   "n_failed_vb": 0,
   "n_meshes": 1500
  },
  "flame2023": {
   "k_median": 1.6095664203411926,
   "k_ls": 1.501189813289013,
   "n_pairs": 297000,
   "n_failed_vb": 0,
   "n_meshes": 1500
  }
 },
 "pilot": {
  "gnm": {
   "key": "vb|8.0|10.0|10",
   "score": 0.1389186409126781,
   "best": 0.1389186409126781,
   "tied": [
    "vb|8.0|10.0|10",
    "vb|8.0|5.0|10"
   ],
   "differs": true,
   "control_orig": 0.0,
   "control_va": 0.0
  },
  "flame2023": {
   "key": "vb|8.0|5.0|10",
   "score": 0.1766095979946639,
   "best": 0.1766095979946639,
   "tied": [
    "vb|8.0|5.0|10"
   ],
   "differs": true,
   "control_orig": 0.0,
   "control_va": 0.0
  }
 }
}
```

## Fit di B sul crop

| vista | modello | mesh | B fallite | superficie mm: mediana | p95 (mediana) | p95 max | corrisp. tenute |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | gnm | 100 | 0 | 0.37 | 1.53 | 3.07 | 0.96 / 0.80 |
| hifi3d | flame2023 | 100 | 0 | 0.36 | 1.42 | 2.53 | 0.97 / 0.71 |
| facescape | gnm | 100 | 0 | 0.29 | 0.96 | 1.85 | 0.98 / 0.72 |
| facescape | flame2023 | 100 | 0 | 0.31 | 1.08 | 2.24 | 0.97 / 0.67 |
| faceverse | gnm | 100 | 0 | 0.51 | 1.86 | 2.67 | 0.99 / 0.74 |
| faceverse | flame2023 | 100 | 0 | 0.46 | 1.71 | 2.67 | 0.99 / 0.59 |
| faceverse_neutral | gnm | 100 | 0 | 0.58 | 2.02 | 2.85 | 0.99 / 0.74 |
| faceverse_neutral | flame2023 | 100 | 0 | 0.44 | 1.61 | 2.60 | 0.99 / 0.59 |

## Diagnostica del crop sui bracci (aggiunta dopo i numeri, non nel protocollo)

Spostamento di log S dei fattorizzati (crop - media delle 5 topologie senza crop, stesso soggetto) e distanze nell'embedding del braccio (d_P = ||u|| per factorized, ||z|| per ctrlfr): crop -> senza crop dello stesso soggetto, fra senza crop dello stesso soggetto, mediana fra soggetti diversi.

| vista | braccio | d log S crop (media, sd) | sd di log S fra soggetti | stesso sogg. crop | stesso sogg. senza crop | soggetti diversi |
| --- | --- | --- | --- | --- | --- | --- |
| hifi3d | factorized_s1234 | -0.0235 (0.0053) | 0.0470 | 0.431 | 0.162 | 0.640 |
| hifi3d | factorized_s2345 | -0.0344 (0.0051) | 0.0472 | 0.441 | 0.131 | 0.643 |
| hifi3d | ctrlfr_s1234 | - | - | 1.244 | 0.284 | 1.046 |
| hifi3d | ctrlfr_s2345 | - | - | 1.232 | 0.275 | 1.107 |
| facescape | factorized_s1234 | -0.0607 (0.0109) | 0.0162 | 0.680 | 0.231 | 0.547 |
| facescape | factorized_s2345 | -0.0704 (0.0132) | 0.0173 | 0.706 | 0.172 | 0.547 |
| facescape | ctrlfr_s1234 | - | - | 1.432 | 0.362 | 0.794 |
| facescape | ctrlfr_s2345 | - | - | 1.442 | 0.355 | 0.822 |
| faceverse | factorized_s1234 | -0.0229 (0.0130) | 0.0253 | 0.591 | 0.446 | 0.958 |
| faceverse | factorized_s2345 | -0.0255 (0.0155) | 0.0254 | 0.596 | 0.435 | 0.967 |
| faceverse | ctrlfr_s1234 | - | - | 0.925 | 0.715 | 1.353 |
| faceverse | ctrlfr_s2345 | - | - | 0.998 | 0.733 | 1.473 |
| faceverse_neutral | factorized_s1234 | -0.0245 (0.0090) | 0.0245 | 0.424 | 0.193 | 0.906 |
| faceverse_neutral | factorized_s2345 | -0.0272 (0.0122) | 0.0241 | 0.427 | 0.161 | 0.907 |
| faceverse_neutral | ctrlfr_s1234 | - | - | 0.711 | 0.379 | 1.245 |
| faceverse_neutral | ctrlfr_s2345 | - | - | 0.777 | 0.371 | 1.362 |

## Costi misurati (sez. 3)

### B: tempi per mesh (s; CPU, un processo per mesh a un thread)

| vista | modello | campione | mesh | NICP mediana | A mediana | B mediana | con NICP mediana / p95 | senza NICP mediana / p95 | fit_e1 `seconds` (tutte, con 3 errori di superficie) mediana / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | gnm | 20 soggetti x 5 topologie | 100 | 3.35 | 0.39 | 1.26 | 4.97 / 5.54 | 1.64 / 1.87 | 4.84 / 21.48 |
| hifi3d | gnm | crop, 100 | 100 | 5.02 | 0.37 | 1.05 | 6.21 / 7.89 | 1.41 / 1.76 | - |
| hifi3d | gnm | un processo solo | 30 | 1.99 | 0.22 | 0.67 | 2.93 / 4.04 | 0.90 / 1.02 | 4.84 / 21.48 |
| hifi3d | flame2023 | 20 soggetti x 5 topologie | 100 | 0.75 | 0.23 | 0.54 | 1.56 / 1.87 | 0.76 / 1.00 | 2.03 / 17.62 |
| hifi3d | flame2023 | crop, 100 | 100 | 1.17 | 0.30 | 0.61 | 2.04 / 2.39 | 0.92 / 1.09 | - |
| hifi3d | flame2023 | un processo solo | 30 | 0.52 | 0.12 | 0.33 | 1.04 / 1.34 | 0.45 / 0.65 | 2.03 / 17.62 |
| facescape | gnm | 20 soggetti x 5 topologie | 100 | 3.03 | 0.32 | 0.99 | 4.51 / 5.97 | 1.32 / 1.61 | 5.06 / 10.02 |
| facescape | gnm | crop, 100 | 100 | 4.85 | 0.37 | 0.99 | 6.21 / 7.48 | 1.32 / 1.65 | - |
| facescape | flame2023 | 20 soggetti x 5 topologie | 100 | 0.81 | 0.23 | 0.60 | 1.68 / 2.15 | 0.83 / 1.11 | 2.28 / 7.06 |
| facescape | flame2023 | crop, 100 | 100 | 1.33 | 0.32 | 0.67 | 2.33 / 2.63 | 1.01 / 1.19 | - |
| faceverse | gnm | 20 soggetti x 5 topologie | 100 | 3.22 | 1.35 | 4.72 | 9.52 / 11.21 | 5.98 / 6.77 | 9.26 / 14.17 |
| faceverse | gnm | crop, 100 | 100 | 3.92 | 1.35 | 4.53 | 10.15 / 11.87 | 5.94 / 6.81 | - |
| faceverse | flame2023 | 20 soggetti x 5 topologie | 100 | 1.20 | 47.68 | 37.40 | 85.33 / 92.01 | 84.09 / 90.71 | 82.36 / 88.79 |
| faceverse | flame2023 | crop, 100 | 100 | 1.40 | 44.55 | 29.30 | 75.92 / 87.19 | 74.36 / 85.62 | - |
| faceverse_neutral | gnm | 20 soggetti x 5 topologie | 100 | 3.04 | 0.31 | 1.14 | 4.63 / 6.05 | 1.43 / 2.13 | 5.22 / 10.12 |
| faceverse_neutral | gnm | crop, 100 | 100 | 3.67 | 0.37 | 1.23 | 5.42 / 7.09 | 1.61 / 1.92 | - |
| faceverse_neutral | flame2023 | 20 soggetti x 5 topologie | 100 | 0.88 | 0.23 | 0.71 | 1.95 / 2.83 | 0.95 / 1.57 | 2.61 / 7.40 |
| faceverse_neutral | flame2023 | crop, 100 | 100 | 1.29 | 0.33 | 0.83 | 2.39 / 2.93 | 1.15 / 1.38 | - |
| famos | gnm | 15 scansioni | 15 | 1.93 | 0.14 | 0.50 | 2.60 / 3.12 | 0.64 / 0.80 | 6.32 / 6.87 |
| famos | flame2023 | 15 scansioni | 15 | 0.68 | 0.15 | 0.31 | 1.15 / 1.47 | 0.45 / 0.53 | 3.74 / 3.91 |

### Bracci: preprocessing degli operatori (CPU, k_eig 128, 32 processi a un thread, a512-l4-06.srv.aau.dk, AMD EPYC 7543 32-Core Processor, job 1067778)

| topologia | vertici (mediana) | s/mesh mediana | p95 |
| --- | --- | --- | --- |
| crop | 9062 | 4.78 | 6.30 |
| down8k | 3312 | 1.56 | 3.21 |
| noisy | 9518 | 5.15 | 6.70 |
| original | 9518 | 5.10 | 7.11 |
| remesh | 6679 | 3.40 | 4.88 |
| up60k | 24394 | 14.33 | 15.83 |
| tutte (600) | - | 4.77 | 14.72 |

Per passo (mediana s): load 0.027, area_norm 0.004, compute_operators 4.695, save 0.022; 4.96 mesh/s sul nodo (121 s per 600).

Un processo solo (60 mesh, a256-t4-02.srv.aau.dk, job 1067782): s/mesh mediana 2.97, p95 9.07; per topologia crop 2.96, down8k 0.93, noisy 3.07, original 3.08, remesh 1.99, up60k 9.02.

### Bracci: forward factorized_s1234 (NVIDIA L40S, a768-l40s-03.srv.aau.dk, job 1067780; scarto dagli embedding di fact_paired 6.3e-04)

batch 1, mediana / p95 (ms): lettura npz operatori 34.2 / 87.2, copia sul device 0.9 / 1.8, forward 6.5 / 7.7, totale 41.6 / 96.7; forward per topologia (mediana ms): crop 6.4, down8k 6.0, noisy 6.5, original 6.6, remesh 6.3, up60k 7.7. Gruppi di 30 gia' sul device (forward sequential del training): 5.6 ms/mesh (mediana sui gruppi).

### Bracci: forward ctrlfr_s1234 (NVIDIA L40S, a768-l40s-03.srv.aau.dk, job 1067780; scarto dagli embedding di fact_paired 1.1e-03)

batch 1, mediana / p95 (ms): lettura npz operatori 33.8 / 86.7, copia sul device 0.9 / 1.8, forward 6.4 / 7.7, totale 41.2 / 96.2; forward per topologia (mediana ms): crop 6.3, down8k 5.9, noisy 6.5, original 6.5, remesh 6.2, up60k 7.6. Gruppi di 30 gia' sul device (forward sequential del training): 5.6 ms/mesh (mediana sui gruppi).

### Confronto di una coppia (numpy, un thread, a256-t4-02.srv.aau.dk, job 1067779)

| metodo | per coppia (us) | ammortizzato su tutte le coppie (us) | per mesh, una volta (ms) |
| --- | --- | --- | --- |
| gnm vB coefficienti (170) | 4.6 | - | - |
| gnm vB mesh FR / SR (7700 vertici) | 71.7 | 17.43 (coef + FR + SR) | 3.02 |
| gnm vB composizione | 74.0 | - | 3.02 |
| flame2023 vB coefficienti (300) | 4.6 | - | - |
| flame2023 vB mesh FR / SR (1517 vertici) | 20.4 | 4.02 (coef + FR + SR) | 0.77 |
| flame2023 vB composizione | 22.5 | - | 0.77 |
| factorized_s1234 (dimensione 257) | 7.0 | 0.032 | - |
| ctrlfr_s1234 (dimensione 256) | 4.7 | 0.031 | - |

### Costo per iscrizione di una mesh (mediana, s)

| metodo | passi | hardware | s/mesh |
| --- | --- | --- | --- |
| B gnm | lettura + NICP + A + B (tutte le mesh cronometrate, 815) | CPU, 1 thread | 5.76 (p95 10.66) |
| B flame2023 | lettura + NICP + A + B (tutte le mesh cronometrate, 815) | CPU, 1 thread | 2.15 (p95 86.93) |
| factorized_s1234 | operatori k128 (CPU, 1 thread) + lettura + copia + forward (NVIDIA L40S) | CPU + GPU | 4.82 (operatori 4.77 + rete 0.042) |
| ctrlfr_s1234 | operatori k128 (CPU, 1 thread) + lettura + copia + forward (NVIDIA L40S) | CPU + GPU | 4.82 (operatori 4.77 + rete 0.041) |

## Controlli dell'emendamento 2

```
{
 "paired_e1_rows": {
  "reference": "paired_e1.csv",
  "n_reference": 11933,
  "n_matched": 11933,
  "max_abs_diff": {
   "arm_point": 9.71445146547012e-17,
   "delta": 9.974659986866641e-17,
   "ci_low": 9.974659986866641e-17,
   "ci_high": 2.220446049250313e-16,
   "p_le0": 1.1102230246251565e-16
  },
  "rows_equal": true
 },
 "gt": {
  "hifi3d": {
   "formula_gt_rho_fr": 0.9967355061632815,
   "n_subject_pairs": 4950,
   "S_cv": 0.04653362530135715,
   "rho_fr_sr_gt": 0.6279773855461537,
   "orig_orig_gnm_vb_coef_fr": 0.553500845424589,
   "orig_orig_gnm_vb_coef_sr": 0.6803767885647402,
   "orig_orig_gnm_vb_fr_fr": 0.6723605681645332,
   "orig_orig_gnm_vb_fr_sr": 0.22617802417554408,
   "orig_orig_gnm_vb_sr_fr": 0.4152864022464637,
   "orig_orig_gnm_vb_sr_sr": 0.6110576423671604,
   "orig_orig_flame2023_vb_coef_fr": 0.5777961569874108,
   "orig_orig_flame2023_vb_coef_sr": 0.5878868408156015,
   "orig_orig_flame2023_vb_fr_fr": 0.6950639633396848,
   "orig_orig_flame2023_vb_fr_sr": 0.23356501709718938,
   "orig_orig_flame2023_vb_sr_fr": 0.45904592155842516,
   "orig_orig_flame2023_vb_sr_sr": 0.6041663060005066
  },
  "facescape": {
   "formula_gt_rho_fr": 0.997603853163816,
   "n_subject_pairs": 4950,
   "S_cv": 0.015780193693559607,
   "rho_fr_sr_gt": 0.8880819949479375,
   "orig_orig_gnm_vb_coef_fr": 0.5675864941339415,
   "orig_orig_gnm_vb_coef_sr": 0.6207857681044865,
   "orig_orig_gnm_vb_fr_fr": 0.6920702726883178,
   "orig_orig_gnm_vb_fr_sr": 0.6226895380016368,
   "orig_orig_gnm_vb_sr_fr": 0.767249039287776,
   "orig_orig_gnm_vb_sr_sr": 0.8558059049493538,
   "orig_orig_flame2023_vb_coef_fr": 0.5594667831727529,
   "orig_orig_flame2023_vb_coef_sr": 0.5975053960846642,
   "orig_orig_flame2023_vb_fr_fr": 0.6377458608752021,
   "orig_orig_flame2023_vb_fr_sr": 0.5948619563771684,
   "orig_orig_flame2023_vb_sr_fr": 0.7154767245140117,
   "orig_orig_flame2023_vb_sr_sr": 0.8270600997047604
  },
  "faceverse": {
   "formula_gt_rho_fr": 0.9977021404297997,
   "n_subject_pairs": 4950,
   "S_cv": 0.017559536985174266,
   "rho_fr_sr_gt": 0.9539530178737777,
   "orig_orig_gnm_vb_coef_fr": 0.1930196039827324,
   "orig_orig_gnm_vb_coef_sr": 0.18504668008687322,
   "orig_orig_gnm_vb_fr_fr": 0.32863734955700624,
   "orig_orig_gnm_vb_fr_sr": 0.2644660736394093,
   "orig_orig_gnm_vb_sr_fr": 0.32513237016924496,
   "orig_orig_gnm_vb_sr_sr": 0.31909256576240774,
   "orig_orig_flame2023_vb_coef_fr": 0.11726618905859931,
   "orig_orig_flame2023_vb_coef_sr": 0.10947198286054076,
   "orig_orig_flame2023_vb_fr_fr": 0.2995338907985267,
   "orig_orig_flame2023_vb_fr_sr": 0.2435234158737724,
   "orig_orig_flame2023_vb_sr_fr": 0.2947394876567235,
   "orig_orig_flame2023_vb_sr_sr": 0.2941854659130703
  },
  "faceverse_neutral": {
   "formula_gt_rho_fr": 0.9977021404297997,
   "n_subject_pairs": 4950,
   "S_cv": 0.017559536985174266,
   "rho_fr_sr_gt": 0.9539530178737777,
   "orig_orig_gnm_vb_coef_fr": 0.2965705148932728,
   "orig_orig_gnm_vb_coef_sr": 0.2986748622436974,
   "orig_orig_gnm_vb_fr_fr": 0.3222566400427407,
   "orig_orig_gnm_vb_fr_sr": 0.2773493272173411,
   "orig_orig_gnm_vb_sr_fr": 0.37890638770820173,
   "orig_orig_gnm_vb_sr_sr": 0.4015074980745539,
   "orig_orig_flame2023_vb_coef_fr": 0.2442569537558007,
   "orig_orig_flame2023_vb_coef_sr": 0.26209801160698165,
   "orig_orig_flame2023_vb_fr_fr": 0.3119015773717933,
   "orig_orig_flame2023_vb_fr_sr": 0.27428847425397246,
   "orig_orig_flame2023_vb_sr_fr": 0.35348240472562614,
   "orig_orig_flame2023_vb_sr_sr": 0.3781553501706739
  },
  "famos": {
   "formula_gt_rho_fr": 0.9998133941530167,
   "n_subject_pairs": 105,
   "S_cv": 0.043912466392480226,
   "rho_fr_sr_gt": 0.6932303545511093
  }
 },
 "info": {
  "primary": {
   "hifi3d": {
    "rows_total": 99000,
    "rows_mask": 98224,
    "nan_rows_by_new_column": {
     "gnm_coef": 0,
     "gnm_fr": 0,
     "gnm_sr": 0,
     "flame2023_coef": 0,
     "flame2023_fr": 0,
     "flame2023_sr": 0,
     "varifold": 0,
     "gnm_va_coef": 0,
     "gnm_va_fr": 0,
     "gnm_va_sr": 0,
     "gnm_vb_coef": 0,
     "gnm_vb_fr": 0,
     "gnm_vb_sr": 0,
     "flame2023_va_coef": 0,
     "flame2023_va_fr": 0,
     "flame2023_va_sr": 0,
     "flame2023_vb_coef": 0,
     "flame2023_vb_fr": 0,
     "flame2023_vb_sr": 0,
     "gnm_vb_comp": 0,
     "flame2023_vb_comp": 0,
     "gnm_vbs_coef": 0,
     "gnm_vbs_fr": 0,
     "gnm_vbs_sr": 0,
     "flame2023_vbs_coef": 0,
     "flame2023_vbs_fr": 0,
     "flame2023_vbs_sr": 0
    },
    "n_arm_columns": 30,
    "baselines": [
     "mm_rigid_icp_chamfer",
     "mm_nicp_template",
     "est_cs",
     "oracle_size",
     "cs_nicp_p2tri",
     "cs_rigid_icp_chamfer",
     "comp_oracle",
     "comp_est",
     "comp_est_nicp",
     "comp_oracle_raw",
     "comp_est_raw",
     "scale_e108"
    ],
    "famos_controls": []
   },
   "facescape": {
    "rows_total": 99000,
    "rows_mask": 88725,
    "nan_rows_by_new_column": {
     "gnm_coef": 0,
     "gnm_fr": 0,
     "gnm_sr": 0,
     "flame2023_coef": 0,
     "flame2023_fr": 0,
     "flame2023_sr": 0,
     "varifold": 0,
     "gnm_va_coef": 0,
     "gnm_va_fr": 0,
     "gnm_va_sr": 0,
     "gnm_vb_coef": 0,
     "gnm_vb_fr": 0,
     "gnm_vb_sr": 0,
     "flame2023_va_coef": 0,
     "flame2023_va_fr": 0,
     "flame2023_va_sr": 0,
     "flame2023_vb_coef": 0,
     "flame2023_vb_fr": 0,
     "flame2023_vb_sr": 0,
     "gnm_vb_comp": 0,
     "flame2023_vb_comp": 0,
     "gnm_vbs_coef": 0,
     "gnm_vbs_fr": 0,
     "gnm_vbs_sr": 0,
     "flame2023_vbs_coef": 0,
     "flame2023_vbs_fr": 0,
     "flame2023_vbs_sr": 0
    },
    "n_arm_columns": 30,
    "baselines": [
     "mm_rigid_icp_chamfer",
     "mm_nicp_template",
     "est_cs",
     "oracle_size",
     "cs_nicp_p2tri",
     "cs_rigid_icp_chamfer",
     "comp_oracle",
     "comp_est",
     "comp_est_nicp",
     "comp_oracle_raw",
     "comp_est_raw",
     "scale_e108"
    ],
    "famos_controls": []
   },
   "faceverse": {
    "rows_total": 99000,
    "rows_mask": 99000,
    "nan_rows_by_new_column": {
     "gnm_coef": 0,
     "gnm_fr": 0,
     "gnm_sr": 0,
     "flame2023_coef": 0,
     "flame2023_fr": 0,
     "flame2023_sr": 0,
     "varifold": 0,
     "gnm_va_coef": 0,
     "gnm_va_fr": 0,
     "gnm_va_sr": 0,
     "gnm_vb_coef": 0,
     "gnm_vb_fr": 0,
     "gnm_vb_sr": 0,
     "flame2023_va_coef": 0,
     "flame2023_va_fr": 0,
     "flame2023_va_sr": 0,
     "flame2023_vb_coef": 0,
     "flame2023_vb_fr": 0,
     "flame2023_vb_sr": 0,
     "gnm_vb_comp": 0,
     "flame2023_vb_comp": 0,
     "gnm_vbs_coef": 0,
     "gnm_vbs_fr": 0,
     "gnm_vbs_sr": 0,
     "flame2023_vbs_coef": 0,
     "flame2023_vbs_fr": 0,
     "flame2023_vbs_sr": 0
    },
    "n_arm_columns": 30,
    "baselines": [
     "mm_rigid_icp_chamfer",
     "mm_nicp_template",
     "est_cs",
     "oracle_size",
     "cs_nicp_p2tri",
     "cs_rigid_icp_chamfer",
     "comp_oracle",
     "comp_est",
     "comp_est_nicp",
     "comp_oracle_raw",
     "comp_est_raw",
     "scale_e108"
    ],
    "famos_controls": []
   },
   "faceverse_neutral": {
    "rows_total": 99000,
    "rows_mask": 99000,
    "nan_rows_by_new_column": {
     "gnm_coef": 0,
     "gnm_fr": 0,
     "gnm_sr": 0,
     "flame2023_coef": 0,
     "flame2023_fr": 0,
     "flame2023_sr": 0,
     "varifold": 0,
     "gnm_va_coef": 0,
     "gnm_va_fr": 0,
     "gnm_va_sr": 0,
     "gnm_vb_coef": 0,
     "gnm_vb_fr": 0,
     "gnm_vb_sr": 0,
     "flame2023_va_coef": 0,
     "flame2023_va_fr": 0,
     "flame2023_va_sr": 0,
     "flame2023_vb_coef": 0,
     "flame2023_vb_fr": 0,
     "flame2023_vb_sr": 0,
     "gnm_vb_comp": 0,
     "flame2023_vb_comp": 0,
     "gnm_vbs_coef": 0,
     "gnm_vbs_fr": 0,
     "gnm_vbs_sr": 0,
     "flame2023_vbs_coef": 0,
     "flame2023_vbs_fr": 0,
     "flame2023_vbs_sr": 0
    },
    "n_arm_columns": 22,
    "baselines": [
     "oracle_size"
    ],
    "famos_controls": []
   },
   "famos": {
    "rows_total": 105,
    "rows_mask": 105,
    "nan_rows_by_new_column": {
     "gnm_coef": 0,
     "gnm_fr": 0,
     "gnm_sr": 0,
     "flame2023_coef": 0,
     "flame2023_fr": 0,
     "flame2023_sr": 0,
     "varifold": 0,
     "gnm_va_coef": 0,
     "gnm_va_fr": 0,
     "gnm_va_sr": 0,
     "gnm_vb_coef": 0,
     "gnm_vb_fr": 0,
     "gnm_vb_sr": 0,
     "flame2023_va_coef": 0,
     "flame2023_va_fr": 0,
     "flame2023_va_sr": 0,
     "flame2023_vb_coef": 0,
     "flame2023_vb_fr": 0,
     "flame2023_vb_sr": 0,
     "gnm_vb_comp": 0,
     "flame2023_vb_comp": 0,
     "gnm_vbs_coef": 0,
     "gnm_vbs_fr": 0,
     "gnm_vbs_sr": 0,
     "flame2023_vbs_coef": 0,
     "flame2023_vbs_fr": 0,
     "flame2023_vbs_sr": 0
    },
    "n_arm_columns": 30,
    "baselines": [
     "mm_rigid_icp_chamfer",
     "est_cs",
     "oracle_size",
     "cs_nicp_p2tri",
     "cs_rigid_icp_chamfer",
     "comp_oracle",
     "comp_est",
     "comp_est_nicp",
     "comp_oracle_raw",
     "comp_est_raw"
    ],
    "famos_controls": [
     "factorized_s1234|form: 0.0e+00",
     "factorized_s2345|form: 0.0e+00",
     "factorized2_s1234|form: 0.0e+00",
     "factorized2_s2345|form: 0.0e+00",
     "ctrlfr_s1234|z: 0.0e+00",
     "ctrlfr_s2345|z: 0.0e+00",
     "dual_s1234|zf: 0.0e+00",
     "dual_s2345|zf: 0.0e+00",
     "factorizedc3m_e123|form: 0.0e+00",
     "factorizedc3m_e205|form: 0.0e+00"
    ]
   }
  },
  "all_cross": {
   "hifi3d": {
    "seed": 757683,
    "rows": 148500,
    "check_rows": {
     "rows_nocrop": 99000,
     "rows_fact_paired": 99000,
     "keys_equal": true,
     "max_abs_diff_gt": 0.0
    },
    "vb600_vs_fit_e1": {
     "gnm_vb_coef": 5.329070518200751e-15,
     "gnm_vb_fr": 2.4555507627255224e-10,
     "gnm_vb_sr": 2.164604051557717e-10,
     "flame2023_vb_coef": 5.329070518200751e-15,
     "flame2023_vb_fr": 1.8132939594295294e-11,
     "flame2023_vb_sr": 1.181801878580302e-11
    },
    "rows_mask": 148500,
    "rows_mask_crop": 49500,
    "nan_rows_by_column": {
     "gnm_vb_coef": 0,
     "gnm_vb_fr": 0,
     "gnm_vb_sr": 0,
     "flame2023_vb_coef": 0,
     "flame2023_vb_fr": 0,
     "flame2023_vb_sr": 0
    }
   },
   "facescape": {
    "seed": 621096,
    "rows": 148500,
    "check_rows": {
     "rows_nocrop": 99000,
     "rows_fact_paired": 99000,
     "keys_equal": true,
     "max_abs_diff_gt": 0.0
    },
    "vb600_vs_fit_e1": {
     "gnm_vb_coef": 3.552713678800501e-15,
     "gnm_vb_fr": 1.8770898724262963e-10,
     "gnm_vb_sr": 2.0568172165447152e-10,
     "flame2023_vb_coef": 1.0658141036401503e-14,
     "flame2023_vb_fr": 9.568817960214915e-12,
     "flame2023_vb_sr": 1.1014272827125637e-11
    },
    "rows_mask": 148500,
    "rows_mask_crop": 49500,
    "nan_rows_by_column": {
     "gnm_vb_coef": 0,
     "gnm_vb_fr": 0,
     "gnm_vb_sr": 0,
     "flame2023_vb_coef": 0,
     "flame2023_vb_fr": 0,
     "flame2023_vb_sr": 0
    }
   },
   "faceverse": {
    "seed": 566363,
    "rows": 148500,
    "check_rows": {
     "rows_nocrop": 99000,
     "rows_fact_paired": 99000,
     "keys_equal": true,
     "max_abs_diff_gt": 0.0
    },
    "vb600_vs_fit_e1": {
     "gnm_vb_coef": 3.552713678800501e-15,
     "gnm_vb_fr": 3.684685889737693e-11,
     "gnm_vb_sr": 5.672307068493865e-11,
     "flame2023_vb_coef": 5.329070518200751e-15,
     "flame2023_vb_fr": 2.295164058807586e-12,
     "flame2023_vb_sr": 3.5534908349177385e-12
    },
    "rows_mask": 148500,
    "rows_mask_crop": 49500,
    "nan_rows_by_column": {
     "gnm_vb_coef": 0,
     "gnm_vb_fr": 0,
     "gnm_vb_sr": 0,
     "flame2023_vb_coef": 0,
     "flame2023_vb_fr": 0,
     "flame2023_vb_sr": 0
    }
   },
   "faceverse_neutral": {
    "seed": 566363,
    "rows": 148500,
    "check_rows": {
     "rows_nocrop": 99000,
     "rows_fact_paired": 99000,
     "keys_equal": true,
     "max_abs_diff_gt": 0.0
    },
    "vb600_vs_fit_e1": {
     "gnm_vb_coef": 3.552713678800501e-15,
     "gnm_vb_fr": 1.485143952262291e-10,
     "gnm_vb_sr": 1.738424404429395e-10,
     "flame2023_vb_coef": 7.105427357601002e-15,
     "flame2023_vb_fr": 1.094402346524248e-11,
     "flame2023_vb_sr": 1.020522555350567e-11
    },
    "rows_mask": 148500,
    "rows_mask_crop": 49500,
    "nan_rows_by_column": {
     "gnm_vb_coef": 0,
     "gnm_vb_fr": 0,
     "gnm_vb_sr": 0,
     "flame2023_vb_coef": 0,
     "flame2023_vb_fr": 0,
     "flame2023_vb_sr": 0
    }
   }
  }
 },
 "sens_columns": [
  "gnm_vbs_coef",
  "gnm_vbs_fr",
  "gnm_vbs_sr",
  "flame2023_vbs_coef",
  "flame2023_vbs_fr",
  "flame2023_vbs_sr"
 ]
}
```

## hifi3d, all_cross (148500 righe, seme 757683)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto) vB, coefficienti | 0.540 [0.457, 0.619] | 0.665 [0.597, 0.720] |
| GNM (visto) vB, mesh d'identita' FR | 0.651 [0.560, 0.727] | 0.213 [0.123, 0.303] |
| GNM (visto) vB, mesh d'identita' SR | 0.395 [0.295, 0.489] | 0.587 [0.504, 0.660] |
| FLAME 2023 Open vB, coefficienti | 0.570 [0.485, 0.643] | 0.573 [0.500, 0.636] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.670 [0.582, 0.748] | 0.223 [0.136, 0.316] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.448 [0.360, 0.536] | 0.586 [0.498, 0.661] |
| factorized s1234, d_F cal. | 0.709 [0.635, 0.771] | 0.304 [0.218, 0.387] |
| factorized s2345, d_F cal. | 0.661 [0.587, 0.724] | 0.286 [0.202, 0.372] |
| factorized s1234, d_P | 0.383 [0.290, 0.471] | 0.557 [0.483, 0.621] |
| factorized s2345, d_P | 0.370 [0.274, 0.463] | 0.540 [0.457, 0.615] |
| ctrlfr s1234 | 0.516 [0.444, 0.580] | 0.244 [0.180, 0.311] |
| ctrlfr s2345 | 0.530 [0.456, 0.594] | 0.259 [0.190, 0.328] |

Delta appaiati, braccio - B (IC 95%, P(delta <= 0)); a favore / contro / non risolte: FR 12 / 4 / 8, SR 5 / 10 / 9

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | +0.169 [+0.066, +0.268], P 0.000 | +0.121 [+0.017, +0.221], P 0.011 | -0.024 [-0.120, +0.068], P 0.681 | -0.010 [-0.111, +0.087], P 0.566 |
| GNM (visto) vB, mesh d'identita' FR | +0.058 [+0.019, +0.100], P 0.000 | +0.010 [-0.026, +0.047], P 0.299 | -0.135 [-0.184, -0.089], P 1.000 | -0.121 [-0.167, -0.078], P 1.000 |
| GNM (visto) vB, mesh d'identita' SR | +0.314 [+0.209, +0.411], P 0.000 | +0.266 [+0.164, +0.361], P 0.000 | +0.120 [+0.022, +0.213], P 0.007 | +0.134 [+0.034, +0.225], P 0.003 |
| FLAME 2023 Open vB, coefficienti | +0.139 [+0.053, +0.227], P 0.001 | +0.091 [+0.002, +0.181], P 0.020 | -0.054 [-0.130, +0.028], P 0.907 | -0.040 [-0.123, +0.044], P 0.824 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.039 [+0.004, +0.075], P 0.017 | -0.009 [-0.043, +0.025], P 0.686 | -0.154 [-0.197, -0.114], P 1.000 | -0.140 [-0.180, -0.104], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' SR | +0.260 [+0.172, +0.348], P 0.000 | +0.213 [+0.127, +0.296], P 0.000 | +0.067 [-0.020, +0.153], P 0.067 | +0.081 [-0.006, +0.166], P 0.035 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | -0.108 [-0.193, -0.022], P 0.993 | -0.124 [-0.216, -0.032], P 0.996 | -0.420 [-0.493, -0.343], P 1.000 | -0.406 [-0.481, -0.324], P 1.000 |
| GNM (visto) vB, mesh d'identita' FR | +0.344 [+0.259, +0.427], P 0.000 | +0.328 [+0.244, +0.413], P 0.000 | +0.032 [-0.014, +0.076], P 0.089 | +0.046 [+0.001, +0.090], P 0.022 |
| GNM (visto) vB, mesh d'identita' SR | -0.030 [-0.084, +0.022], P 0.870 | -0.046 [-0.101, +0.006], P 0.953 | -0.342 [-0.404, -0.281], P 1.000 | -0.328 [-0.386, -0.268], P 1.000 |
| FLAME 2023 Open vB, coefficienti | -0.016 [-0.104, +0.069], P 0.639 | -0.033 [-0.132, +0.060], P 0.754 | -0.329 [-0.408, -0.249], P 1.000 | -0.314 [-0.398, -0.229], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.334 [+0.253, +0.411], P 0.000 | +0.317 [+0.236, +0.399], P 0.000 | +0.021 [-0.024, +0.063], P 0.177 | +0.036 [-0.010, +0.078], P 0.052 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.029 [-0.084, +0.028], P 0.850 | -0.045 [-0.103, +0.013], P 0.944 | -0.341 [-0.401, -0.282], P 1.000 | -0.326 [-0.385, -0.270], P 1.000 |

## hifi3d, all_cross, righe col crop (49500 righe, seme 757683)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto) vB, coefficienti | 0.524 [0.441, 0.601] | 0.651 [0.580, 0.706] |
| GNM (visto) vB, mesh d'identita' FR | 0.625 [0.536, 0.705] | 0.205 [0.118, 0.294] |
| GNM (visto) vB, mesh d'identita' SR | 0.380 [0.282, 0.475] | 0.570 [0.482, 0.648] |
| FLAME 2023 Open vB, coefficienti | 0.542 [0.457, 0.617] | 0.551 [0.481, 0.614] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.639 [0.551, 0.719] | 0.209 [0.126, 0.299] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.433 [0.343, 0.518] | 0.567 [0.478, 0.643] |
| factorized s1234, d_F cal. | 0.663 [0.592, 0.722] | 0.297 [0.218, 0.376] |
| factorized s2345, d_F cal. | 0.552 [0.480, 0.615] | 0.255 [0.181, 0.328] |
| factorized s1234, d_P | 0.376 [0.281, 0.463] | 0.550 [0.470, 0.618] |
| factorized s2345, d_P | 0.347 [0.253, 0.440] | 0.514 [0.426, 0.595] |
| ctrlfr s1234 | 0.236 [0.183, 0.283] | 0.164 [0.122, 0.207] |
| ctrlfr s2345 | 0.279 [0.220, 0.331] | 0.188 [0.144, 0.234] |

Delta appaiati, braccio - B (IC 95%, P(delta <= 0)); a favore / contro / non risolte: FR 6 / 14 / 4, SR 4 / 10 / 10

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | +0.139 [+0.043, +0.235], P 0.001 | +0.028 [-0.067, +0.123], P 0.281 | -0.288 [-0.365, -0.211], P 1.000 | -0.245 [-0.328, -0.160], P 1.000 |
| GNM (visto) vB, mesh d'identita' FR | +0.038 [-0.004, +0.079], P 0.043 | -0.072 [-0.114, -0.031], P 1.000 | -0.389 [-0.460, -0.318], P 1.000 | -0.346 [-0.410, -0.283], P 1.000 |
| GNM (visto) vB, mesh d'identita' SR | +0.282 [+0.183, +0.374], P 0.000 | +0.172 [+0.079, +0.259], P 0.001 | -0.145 [-0.224, -0.066], P 1.000 | -0.102 [-0.186, -0.019], P 0.994 |
| FLAME 2023 Open vB, coefficienti | +0.120 [+0.035, +0.204], P 0.002 | +0.010 [-0.078, +0.094], P 0.413 | -0.307 [-0.377, -0.234], P 1.000 | -0.264 [-0.334, -0.190], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.024 [-0.017, +0.065], P 0.131 | -0.086 [-0.126, -0.048], P 1.000 | -0.403 [-0.469, -0.334], P 1.000 | -0.360 [-0.421, -0.301], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' SR | +0.229 [+0.143, +0.314], P 0.000 | +0.119 [+0.037, +0.197], P 0.002 | -0.197 [-0.270, -0.123], P 1.000 | -0.155 [-0.229, -0.082], P 1.000 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | -0.101 [-0.186, -0.010], P 0.984 | -0.137 [-0.232, -0.038], P 0.998 | -0.487 [-0.549, -0.418], P 1.000 | -0.462 [-0.526, -0.392], P 1.000 |
| GNM (visto) vB, mesh d'identita' FR | +0.344 [+0.258, +0.427], P 0.000 | +0.308 [+0.224, +0.391], P 0.000 | -0.042 [-0.114, +0.024], P 0.891 | -0.017 [-0.081, +0.041], P 0.693 |
| GNM (visto) vB, mesh d'identita' SR | -0.020 [-0.089, +0.045], P 0.744 | -0.056 [-0.121, +0.004], P 0.965 | -0.406 [-0.467, -0.336], P 1.000 | -0.381 [-0.439, -0.316], P 1.000 |
| FLAME 2023 Open vB, coefficienti | -0.002 [-0.095, +0.090], P 0.518 | -0.037 [-0.137, +0.058], P 0.769 | -0.388 [-0.464, -0.311], P 1.000 | -0.363 [-0.437, -0.287], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.341 [+0.261, +0.419], P 0.000 | +0.305 [+0.223, +0.383], P 0.000 | -0.045 [-0.116, +0.019], P 0.905 | -0.020 [-0.086, +0.037], P 0.726 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.017 [-0.078, +0.050], P 0.721 | -0.053 [-0.114, +0.008], P 0.950 | -0.403 [-0.462, -0.336], P 1.000 | -0.378 [-0.432, -0.316], P 1.000 |

## facescape, all_cross (148500 righe, seme 621096)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto) vB, coefficienti | 0.559 [0.485, 0.630] | 0.611 [0.543, 0.670] |
| GNM (visto) vB, mesh d'identita' FR | 0.661 [0.597, 0.715] | 0.593 [0.521, 0.660] |
| GNM (visto) vB, mesh d'identita' SR | 0.758 [0.675, 0.831] | 0.840 [0.791, 0.882] |
| FLAME 2023 Open vB, coefficienti | 0.535 [0.459, 0.608] | 0.576 [0.502, 0.647] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.606 [0.538, 0.668] | 0.568 [0.498, 0.637] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.707 [0.609, 0.788] | 0.814 [0.752, 0.864] |
| factorized s1234, d_F cal. | 0.370 [0.320, 0.418] | 0.392 [0.343, 0.439] |
| factorized s2345, d_F cal. | 0.351 [0.308, 0.393] | 0.365 [0.323, 0.404] |
| factorized s1234, d_P | 0.443 [0.374, 0.512] | 0.489 [0.424, 0.548] |
| factorized s2345, d_P | 0.422 [0.353, 0.485] | 0.464 [0.402, 0.519] |
| ctrlfr s1234 | 0.334 [0.296, 0.369] | 0.315 [0.271, 0.354] |
| ctrlfr s2345 | 0.323 [0.285, 0.359] | 0.315 [0.273, 0.353] |

Delta appaiati, braccio - B (IC 95%, P(delta <= 0)); a favore / contro / non risolte: FR 0 / 24 / 0, SR 0 / 24 / 0

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | -0.189 [-0.253, -0.120], P 1.000 | -0.208 [-0.269, -0.139], P 1.000 | -0.225 [-0.293, -0.153], P 1.000 | -0.236 [-0.299, -0.165], P 1.000 |
| GNM (visto) vB, mesh d'identita' FR | -0.291 [-0.344, -0.234], P 1.000 | -0.310 [-0.358, -0.257], P 1.000 | -0.327 [-0.378, -0.273], P 1.000 | -0.338 [-0.388, -0.283], P 1.000 |
| GNM (visto) vB, mesh d'identita' SR | -0.387 [-0.445, -0.321], P 1.000 | -0.406 [-0.466, -0.337], P 1.000 | -0.424 [-0.491, -0.348], P 1.000 | -0.435 [-0.497, -0.359], P 1.000 |
| FLAME 2023 Open vB, coefficienti | -0.165 [-0.227, -0.096], P 1.000 | -0.184 [-0.247, -0.117], P 1.000 | -0.201 [-0.267, -0.129], P 1.000 | -0.212 [-0.274, -0.141], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' FR | -0.236 [-0.289, -0.180], P 1.000 | -0.255 [-0.304, -0.201], P 1.000 | -0.272 [-0.327, -0.217], P 1.000 | -0.283 [-0.335, -0.227], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.336 [-0.404, -0.260], P 1.000 | -0.355 [-0.424, -0.275], P 1.000 | -0.373 [-0.454, -0.282], P 1.000 | -0.384 [-0.459, -0.295], P 1.000 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | -0.122 [-0.186, -0.056], P 1.000 | -0.147 [-0.212, -0.081], P 1.000 | -0.296 [-0.355, -0.235], P 1.000 | -0.296 [-0.353, -0.237], P 1.000 |
| GNM (visto) vB, mesh d'identita' FR | -0.104 [-0.168, -0.040], P 1.000 | -0.129 [-0.191, -0.069], P 1.000 | -0.278 [-0.338, -0.224], P 1.000 | -0.278 [-0.338, -0.222], P 1.000 |
| GNM (visto) vB, mesh d'identita' SR | -0.351 [-0.402, -0.301], P 1.000 | -0.376 [-0.424, -0.330], P 1.000 | -0.525 [-0.562, -0.488], P 1.000 | -0.525 [-0.562, -0.486], P 1.000 |
| FLAME 2023 Open vB, coefficienti | -0.087 [-0.151, -0.025], P 0.998 | -0.112 [-0.175, -0.048], P 1.000 | -0.261 [-0.321, -0.200], P 1.000 | -0.261 [-0.319, -0.201], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' FR | -0.079 [-0.140, -0.015], P 0.996 | -0.105 [-0.163, -0.044], P 1.000 | -0.254 [-0.309, -0.199], P 1.000 | -0.254 [-0.307, -0.200], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.325 [-0.382, -0.269], P 1.000 | -0.350 [-0.406, -0.298], P 1.000 | -0.499 [-0.547, -0.448], P 1.000 | -0.499 [-0.545, -0.446], P 1.000 |

## facescape, all_cross, righe col crop (49500 righe, seme 621096)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto) vB, coefficienti | 0.549 [0.471, 0.620] | 0.601 [0.531, 0.662] |
| GNM (visto) vB, mesh d'identita' FR | 0.628 [0.569, 0.680] | 0.566 [0.493, 0.632] |
| GNM (visto) vB, mesh d'identita' SR | 0.755 [0.682, 0.823] | 0.830 [0.783, 0.873] |
| FLAME 2023 Open vB, coefficienti | 0.514 [0.435, 0.586] | 0.553 [0.478, 0.629] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.561 [0.497, 0.621] | 0.530 [0.462, 0.595] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.703 [0.605, 0.784] | 0.807 [0.745, 0.859] |
| factorized s1234, d_F cal. | 0.351 [0.272, 0.428] | 0.383 [0.310, 0.450] |
| factorized s2345, d_F cal. | 0.278 [0.206, 0.348] | 0.307 [0.237, 0.371] |
| factorized s1234, d_P | 0.459 [0.369, 0.545] | 0.518 [0.429, 0.599] |
| factorized s2345, d_P | 0.426 [0.331, 0.507] | 0.476 [0.384, 0.557] |
| ctrlfr s1234 | 0.264 [0.196, 0.328] | 0.247 [0.179, 0.312] |
| ctrlfr s2345 | 0.234 [0.166, 0.298] | 0.243 [0.178, 0.308] |

Delta appaiati, braccio - B (IC 95%, P(delta <= 0)); a favore / contro / non risolte: FR 0 / 24 / 0, SR 0 / 19 / 5

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | -0.198 [-0.285, -0.111], P 1.000 | -0.271 [-0.355, -0.184], P 1.000 | -0.285 [-0.351, -0.215], P 1.000 | -0.315 [-0.386, -0.240], P 1.000 |
| GNM (visto) vB, mesh d'identita' FR | -0.277 [-0.347, -0.204], P 1.000 | -0.350 [-0.416, -0.281], P 1.000 | -0.364 [-0.421, -0.308], P 1.000 | -0.394 [-0.454, -0.331], P 1.000 |
| GNM (visto) vB, mesh d'identita' SR | -0.404 [-0.470, -0.338], P 1.000 | -0.477 [-0.547, -0.404], P 1.000 | -0.491 [-0.568, -0.405], P 1.000 | -0.521 [-0.593, -0.443], P 1.000 |
| FLAME 2023 Open vB, coefficienti | -0.163 [-0.256, -0.076], P 0.999 | -0.236 [-0.322, -0.145], P 1.000 | -0.250 [-0.315, -0.181], P 1.000 | -0.280 [-0.347, -0.209], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' FR | -0.210 [-0.281, -0.138], P 1.000 | -0.283 [-0.348, -0.209], P 1.000 | -0.297 [-0.353, -0.237], P 1.000 | -0.326 [-0.385, -0.265], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.352 [-0.426, -0.276], P 1.000 | -0.425 [-0.506, -0.336], P 1.000 | -0.439 [-0.527, -0.338], P 1.000 | -0.469 [-0.550, -0.372], P 1.000 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | -0.083 [-0.161, -0.001], P 0.976 | -0.125 [-0.211, -0.036], P 0.996 | -0.354 [-0.418, -0.286], P 1.000 | -0.358 [-0.419, -0.285], P 1.000 |
| GNM (visto) vB, mesh d'identita' FR | -0.048 [-0.121, +0.026], P 0.881 | -0.090 [-0.168, -0.013], P 0.987 | -0.319 [-0.381, -0.257], P 1.000 | -0.323 [-0.390, -0.256], P 1.000 |
| GNM (visto) vB, mesh d'identita' SR | -0.312 [-0.383, -0.250], P 1.000 | -0.354 [-0.428, -0.289], P 1.000 | -0.583 [-0.639, -0.524], P 1.000 | -0.587 [-0.643, -0.526], P 1.000 |
| FLAME 2023 Open vB, coefficienti | -0.035 [-0.121, +0.046], P 0.787 | -0.077 [-0.164, +0.008], P 0.961 | -0.306 [-0.373, -0.243], P 1.000 | -0.310 [-0.376, -0.242], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' FR | -0.011 [-0.082, +0.060], P 0.608 | -0.053 [-0.130, +0.018], P 0.899 | -0.282 [-0.338, -0.224], P 1.000 | -0.286 [-0.343, -0.226], P 1.000 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.289 [-0.364, -0.220], P 1.000 | -0.330 [-0.410, -0.257], P 1.000 | -0.560 [-0.622, -0.492], P 1.000 | -0.564 [-0.624, -0.496], P 1.000 |

## faceverse, all_cross (148500 righe, seme 566363)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto) vB, coefficienti | 0.224 [0.142, 0.307] | 0.223 [0.134, 0.310] |
| GNM (visto) vB, mesh d'identita' FR | 0.289 [0.202, 0.368] | 0.241 [0.141, 0.330] |
| GNM (visto) vB, mesh d'identita' SR | 0.314 [0.230, 0.393] | 0.323 [0.237, 0.402] |
| FLAME 2023 Open vB, coefficienti | 0.195 [0.113, 0.275] | 0.188 [0.104, 0.269] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.258 [0.176, 0.337] | 0.213 [0.116, 0.307] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.283 [0.198, 0.358] | 0.288 [0.195, 0.368] |
| factorized s1234, d_F cal. | 0.284 [0.211, 0.347] | 0.251 [0.177, 0.321] |
| factorized s2345, d_F cal. | 0.288 [0.230, 0.342] | 0.258 [0.187, 0.324] |
| factorized s1234, d_P | 0.271 [0.203, 0.339] | 0.273 [0.208, 0.342] |
| factorized s2345, d_P | 0.287 [0.221, 0.351] | 0.293 [0.227, 0.356] |
| ctrlfr s1234 | 0.273 [0.214, 0.335] | 0.261 [0.198, 0.326] |
| ctrlfr s2345 | 0.251 [0.187, 0.319] | 0.253 [0.187, 0.317] |

Delta appaiati, braccio - B (IC 95%, P(delta <= 0)); a favore / contro / non risolte: FR 1 / 0 / 23, SR 1 / 0 / 23

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | +0.061 [-0.034, +0.171], P 0.125 | +0.064 [-0.028, +0.159], P 0.085 | +0.049 [-0.031, +0.125], P 0.108 | +0.027 [-0.054, +0.117], P 0.260 |
| GNM (visto) vB, mesh d'identita' FR | -0.004 [-0.080, +0.077], P 0.538 | -0.001 [-0.071, +0.072], P 0.494 | -0.016 [-0.098, +0.076], P 0.643 | -0.037 [-0.122, +0.054], P 0.780 |
| GNM (visto) vB, mesh d'identita' SR | -0.029 [-0.115, +0.061], P 0.753 | -0.026 [-0.106, +0.053], P 0.744 | -0.041 [-0.117, +0.032], P 0.868 | -0.063 [-0.132, +0.014], P 0.949 |
| FLAME 2023 Open vB, coefficienti | +0.090 [-0.009, +0.187], P 0.042 | +0.093 [+0.006, +0.180], P 0.016 | +0.078 [-0.003, +0.157], P 0.031 | +0.056 [-0.028, +0.141], P 0.103 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.027 [-0.061, +0.111], P 0.267 | +0.030 [-0.049, +0.106], P 0.230 | +0.015 [-0.073, +0.110], P 0.362 | -0.007 [-0.096, +0.086], P 0.545 |
| FLAME 2023 Open vB, mesh d'identita' SR | +0.001 [-0.096, +0.101], P 0.501 | +0.005 [-0.080, +0.096], P 0.461 | -0.010 [-0.092, +0.074], P 0.611 | -0.032 [-0.110, +0.056], P 0.743 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | +0.051 [-0.037, +0.141], P 0.119 | +0.070 [-0.014, +0.156], P 0.057 | +0.038 [-0.035, +0.114], P 0.165 | +0.031 [-0.054, +0.117], P 0.229 |
| GNM (visto) vB, mesh d'identita' FR | +0.033 [-0.063, +0.133], P 0.247 | +0.052 [-0.033, +0.135], P 0.112 | +0.020 [-0.069, +0.113], P 0.327 | +0.012 [-0.072, +0.099], P 0.380 |
| GNM (visto) vB, mesh d'identita' SR | -0.050 [-0.121, +0.029], P 0.902 | -0.031 [-0.103, +0.045], P 0.783 | -0.062 [-0.134, +0.010], P 0.948 | -0.070 [-0.142, +0.002], P 0.971 |
| FLAME 2023 Open vB, coefficienti | +0.086 [-0.001, +0.171], P 0.029 | +0.105 [+0.023, +0.188], P 0.008 | +0.073 [-0.010, +0.154], P 0.046 | +0.065 [-0.015, +0.146], P 0.062 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.061 [-0.046, +0.164], P 0.121 | +0.080 [-0.010, +0.167], P 0.053 | +0.048 [-0.053, +0.146], P 0.179 | +0.040 [-0.051, +0.130], P 0.196 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.014 [-0.099, +0.078], P 0.627 | +0.005 [-0.075, +0.096], P 0.450 | -0.027 [-0.107, +0.068], P 0.735 | -0.034 [-0.115, +0.058], P 0.772 |

## faceverse, all_cross, righe col crop (49500 righe, seme 566363)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto) vB, coefficienti | 0.220 [0.130, 0.306] | 0.224 [0.130, 0.310] |
| GNM (visto) vB, mesh d'identita' FR | 0.281 [0.191, 0.359] | 0.235 [0.135, 0.327] |
| GNM (visto) vB, mesh d'identita' SR | 0.305 [0.226, 0.382] | 0.316 [0.230, 0.398] |
| FLAME 2023 Open vB, coefficienti | 0.179 [0.089, 0.272] | 0.178 [0.085, 0.271] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.245 [0.159, 0.328] | 0.202 [0.106, 0.299] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.270 [0.181, 0.347] | 0.278 [0.179, 0.361] |
| factorized s1234, d_F cal. | 0.252 [0.176, 0.317] | 0.220 [0.141, 0.291] |
| factorized s2345, d_F cal. | 0.234 [0.168, 0.290] | 0.208 [0.134, 0.271] |
| factorized s1234, d_P | 0.252 [0.174, 0.324] | 0.251 [0.175, 0.323] |
| factorized s2345, d_P | 0.251 [0.181, 0.314] | 0.257 [0.189, 0.322] |
| ctrlfr s1234 | 0.256 [0.193, 0.318] | 0.239 [0.174, 0.306] |
| ctrlfr s2345 | 0.239 [0.167, 0.305] | 0.243 [0.172, 0.305] |

Delta appaiati, braccio - B (IC 95%, P(delta <= 0)); a favore / contro / non risolte: FR 0 / 0 / 24, SR 0 / 0 / 24

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | +0.032 [-0.072, +0.139], P 0.262 | +0.014 [-0.089, +0.113], P 0.386 | +0.036 [-0.055, +0.119], P 0.213 | +0.019 [-0.071, +0.108], P 0.338 |
| GNM (visto) vB, mesh d'identita' FR | -0.028 [-0.104, +0.049], P 0.758 | -0.047 [-0.119, +0.025], P 0.876 | -0.025 [-0.105, +0.059], P 0.719 | -0.042 [-0.126, +0.041], P 0.824 |
| GNM (visto) vB, mesh d'identita' SR | -0.052 [-0.141, +0.039], P 0.889 | -0.070 [-0.153, +0.014], P 0.949 | -0.049 [-0.123, +0.029], P 0.912 | -0.066 [-0.143, +0.011], P 0.956 |
| FLAME 2023 Open vB, coefficienti | +0.074 [-0.043, +0.177], P 0.090 | +0.055 [-0.049, +0.154], P 0.140 | +0.077 [-0.019, +0.167], P 0.062 | +0.060 [-0.037, +0.152], P 0.104 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.007 [-0.081, +0.088], P 0.440 | -0.011 [-0.093, +0.064], P 0.603 | +0.011 [-0.080, +0.100], P 0.403 | -0.006 [-0.092, +0.082], P 0.553 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.018 [-0.120, +0.085], P 0.641 | -0.036 [-0.134, +0.060], P 0.774 | -0.014 [-0.106, +0.074], P 0.640 | -0.031 [-0.119, +0.056], P 0.757 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | +0.027 [-0.073, +0.130], P 0.295 | +0.032 [-0.061, +0.130], P 0.240 | +0.014 [-0.076, +0.097], P 0.397 | +0.018 [-0.078, +0.108], P 0.336 |
| GNM (visto) vB, mesh d'identita' FR | +0.016 [-0.078, +0.116], P 0.362 | +0.022 [-0.065, +0.105], P 0.305 | +0.004 [-0.080, +0.090], P 0.476 | +0.008 [-0.066, +0.087], P 0.422 |
| GNM (visto) vB, mesh d'identita' SR | -0.065 [-0.144, +0.019], P 0.934 | -0.059 [-0.138, +0.021], P 0.930 | -0.077 [-0.154, +0.002], P 0.973 | -0.073 [-0.149, +0.000], P 0.972 |
| FLAME 2023 Open vB, coefficienti | +0.073 [-0.030, +0.176], P 0.087 | +0.078 [-0.023, +0.170], P 0.059 | +0.060 [-0.041, +0.155], P 0.121 | +0.064 [-0.027, +0.155], P 0.081 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.049 [-0.059, +0.149], P 0.177 | +0.055 [-0.038, +0.142], P 0.131 | +0.036 [-0.062, +0.131], P 0.236 | +0.041 [-0.049, +0.124], P 0.184 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.027 [-0.120, +0.073], P 0.716 | -0.021 [-0.109, +0.070], P 0.687 | -0.039 [-0.131, +0.054], P 0.814 | -0.035 [-0.123, +0.057], P 0.780 |

## faceverse_neutral, all_cross (148500 righe, seme 566363)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto) vB, coefficienti | 0.297 [0.202, 0.383] | 0.301 [0.206, 0.390] |
| GNM (visto) vB, mesh d'identita' FR | 0.321 [0.229, 0.403] | 0.275 [0.170, 0.372] |
| GNM (visto) vB, mesh d'identita' SR | 0.375 [0.291, 0.454] | 0.400 [0.312, 0.475] |
| FLAME 2023 Open vB, coefficienti | 0.240 [0.145, 0.339] | 0.258 [0.160, 0.355] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.312 [0.224, 0.393] | 0.271 [0.171, 0.366] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.355 [0.267, 0.437] | 0.378 [0.283, 0.461] |
| factorized s1234, d_F cal. | 0.319 [0.243, 0.386] | 0.288 [0.211, 0.358] |
| factorized s2345, d_F cal. | 0.335 [0.274, 0.395] | 0.303 [0.228, 0.372] |
| factorized s1234, d_P | 0.310 [0.232, 0.384] | 0.323 [0.254, 0.391] |
| factorized s2345, d_P | 0.341 [0.265, 0.418] | 0.352 [0.283, 0.423] |
| ctrlfr s1234 | 0.316 [0.248, 0.387] | 0.314 [0.240, 0.382] |
| ctrlfr s2345 | 0.289 [0.215, 0.363] | 0.294 [0.221, 0.367] |

Delta appaiati, braccio - B (IC 95%, P(delta <= 0)); a favore / contro / non risolte: FR 0 / 0 / 24, SR 0 / 1 / 23

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | +0.022 [-0.085, +0.133], P 0.338 | +0.038 [-0.069, +0.142], P 0.233 | +0.019 [-0.069, +0.106], P 0.328 | -0.008 [-0.105, +0.091], P 0.538 |
| GNM (visto) vB, mesh d'identita' FR | -0.002 [-0.087, +0.084], P 0.523 | +0.013 [-0.071, +0.095], P 0.373 | -0.005 [-0.099, +0.090], P 0.550 | -0.033 [-0.127, +0.064], P 0.736 |
| GNM (visto) vB, mesh d'identita' SR | -0.056 [-0.147, +0.038], P 0.896 | -0.040 [-0.129, +0.043], P 0.834 | -0.059 [-0.149, +0.026], P 0.914 | -0.086 [-0.174, +0.009], P 0.964 |
| FLAME 2023 Open vB, coefficienti | +0.079 [-0.035, +0.190], P 0.089 | +0.095 [-0.018, +0.200], P 0.049 | +0.076 [-0.028, +0.170], P 0.083 | +0.049 [-0.055, +0.151], P 0.179 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.007 [-0.092, +0.099], P 0.453 | +0.023 [-0.071, +0.109], P 0.304 | +0.004 [-0.091, +0.102], P 0.462 | -0.023 [-0.121, +0.073], P 0.664 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.036 [-0.133, +0.072], P 0.758 | -0.020 [-0.116, +0.080], P 0.664 | -0.039 [-0.134, +0.058], P 0.792 | -0.066 [-0.161, +0.033], P 0.905 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | +0.022 [-0.073, +0.120], P 0.323 | +0.051 [-0.047, +0.150], P 0.149 | +0.013 [-0.083, +0.108], P 0.392 | -0.007 [-0.105, +0.097], P 0.536 |
| GNM (visto) vB, mesh d'identita' FR | +0.048 [-0.057, +0.148], P 0.177 | +0.077 [-0.020, +0.164], P 0.060 | +0.039 [-0.062, +0.131], P 0.228 | +0.019 [-0.074, +0.106], P 0.335 |
| GNM (visto) vB, mesh d'identita' SR | -0.077 [-0.154, +0.007], P 0.964 | -0.048 [-0.127, +0.038], P 0.871 | -0.086 [-0.171, +0.003], P 0.973 | -0.105 [-0.184, -0.016], P 0.989 |
| FLAME 2023 Open vB, coefficienti | +0.065 [-0.038, +0.165], P 0.111 | +0.094 [-0.012, +0.195], P 0.044 | +0.056 [-0.050, +0.154], P 0.144 | +0.037 [-0.069, +0.138], P 0.233 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.052 [-0.056, +0.153], P 0.172 | +0.081 [-0.018, +0.172], P 0.063 | +0.043 [-0.063, +0.143], P 0.225 | +0.023 [-0.070, +0.117], P 0.311 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.055 [-0.138, +0.042], P 0.891 | -0.026 [-0.116, +0.070], P 0.708 | -0.064 [-0.159, +0.039], P 0.908 | -0.084 [-0.178, +0.012], P 0.954 |

## faceverse_neutral, all_cross, righe col crop (49500 righe, seme 566363)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto) vB, coefficienti | 0.292 [0.198, 0.377] | 0.296 [0.199, 0.384] |
| GNM (visto) vB, mesh d'identita' FR | 0.323 [0.234, 0.403] | 0.276 [0.176, 0.371] |
| GNM (visto) vB, mesh d'identita' SR | 0.379 [0.299, 0.455] | 0.402 [0.317, 0.478] |
| FLAME 2023 Open vB, coefficienti | 0.223 [0.127, 0.325] | 0.238 [0.139, 0.338] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.306 [0.219, 0.386] | 0.265 [0.167, 0.356] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.348 [0.259, 0.433] | 0.371 [0.275, 0.457] |
| factorized s1234, d_F cal. | 0.282 [0.207, 0.348] | 0.255 [0.176, 0.326] |
| factorized s2345, d_F cal. | 0.272 [0.212, 0.329] | 0.247 [0.178, 0.306] |
| factorized s1234, d_P | 0.293 [0.209, 0.368] | 0.304 [0.231, 0.376] |
| factorized s2345, d_P | 0.306 [0.227, 0.381] | 0.316 [0.245, 0.389] |
| ctrlfr s1234 | 0.296 [0.227, 0.362] | 0.291 [0.214, 0.363] |
| ctrlfr s2345 | 0.269 [0.200, 0.341] | 0.274 [0.200, 0.346] |

Delta appaiati, braccio - B (IC 95%, P(delta <= 0)); a favore / contro / non risolte: FR 0 / 3 / 21, SR 0 / 4 / 20

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | -0.009 [-0.119, +0.096], P 0.565 | -0.019 [-0.121, +0.081], P 0.645 | +0.004 [-0.090, +0.091], P 0.467 | -0.022 [-0.117, +0.077], P 0.674 |
| GNM (visto) vB, mesh d'identita' FR | -0.040 [-0.120, +0.039], P 0.827 | -0.050 [-0.130, +0.031], P 0.879 | -0.027 [-0.120, +0.067], P 0.716 | -0.053 [-0.141, +0.038], P 0.868 |
| GNM (visto) vB, mesh d'identita' SR | -0.096 [-0.183, -0.006], P 0.984 | -0.106 [-0.193, -0.023], P 0.991 | -0.083 [-0.170, +0.001], P 0.973 | -0.110 [-0.198, -0.023], P 0.994 |
| FLAME 2023 Open vB, coefficienti | +0.060 [-0.056, +0.173], P 0.143 | +0.050 [-0.065, +0.153], P 0.190 | +0.073 [-0.033, +0.167], P 0.092 | +0.047 [-0.056, +0.147], P 0.194 |
| FLAME 2023 Open vB, mesh d'identita' FR | -0.024 [-0.115, +0.064], P 0.691 | -0.034 [-0.127, +0.051], P 0.780 | -0.010 [-0.109, +0.082], P 0.575 | -0.037 [-0.128, +0.059], P 0.766 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.066 [-0.166, +0.035], P 0.905 | -0.076 [-0.172, +0.024], P 0.933 | -0.053 [-0.152, +0.044], P 0.864 | -0.079 [-0.174, +0.018], P 0.944 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti | +0.008 [-0.090, +0.104], P 0.436 | +0.020 [-0.079, +0.121], P 0.334 | -0.005 [-0.100, +0.087], P 0.541 | -0.022 [-0.118, +0.077], P 0.654 |
| GNM (visto) vB, mesh d'identita' FR | +0.029 [-0.071, +0.123], P 0.286 | +0.040 [-0.053, +0.123], P 0.205 | +0.015 [-0.081, +0.110], P 0.385 | -0.001 [-0.089, +0.081], P 0.512 |
| GNM (visto) vB, mesh d'identita' SR | -0.098 [-0.181, -0.014], P 0.991 | -0.086 [-0.170, -0.002], P 0.976 | -0.111 [-0.201, -0.023], P 0.991 | -0.128 [-0.209, -0.043], P 1.000 |
| FLAME 2023 Open vB, coefficienti | +0.066 [-0.038, +0.164], P 0.111 | +0.078 [-0.028, +0.178], P 0.075 | +0.052 [-0.053, +0.151], P 0.171 | +0.036 [-0.067, +0.133], P 0.254 |
| FLAME 2023 Open vB, mesh d'identita' FR | +0.040 [-0.063, +0.135], P 0.216 | +0.051 [-0.041, +0.138], P 0.151 | +0.026 [-0.081, +0.124], P 0.315 | +0.010 [-0.084, +0.103], P 0.428 |
| FLAME 2023 Open vB, mesh d'identita' SR | -0.066 [-0.154, +0.035], P 0.917 | -0.055 [-0.146, +0.041], P 0.871 | -0.080 [-0.181, +0.023], P 0.940 | -0.096 [-0.189, +0.005], P 0.969 |

## hifi3d, composizione forma B + taglia B e sensibilita' a sigma (98224 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto) vB, composizione S_B + k d_P | 0.698 [0.612, 0.769] | 0.348 [0.257, 0.438] |
| FLAME 2023 Open vB, composizione S_B + k d_P | 0.712 [0.627, 0.781] | 0.346 [0.255, 0.437] |
| GNM (visto) vB, mesh d'identita' FR | 0.664 [0.570, 0.738] | 0.217 [0.127, 0.306] |
| GNM (visto) vB, mesh d'identita' SR | 0.411 [0.319, 0.499] | 0.607 [0.532, 0.679] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.686 [0.597, 0.758] | 0.231 [0.141, 0.320] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.458 [0.365, 0.542] | 0.597 [0.518, 0.673] |
| GNM (visto) vB sigma sens., coefficienti | 0.554 [0.481, 0.624] | 0.567 [0.493, 0.639] |
| GNM (visto) vB sigma sens., mesh d'identita' FR | 0.683 [0.591, 0.757] | 0.233 [0.141, 0.325] |
| GNM (visto) vB sigma sens., mesh d'identita' SR | 0.449 [0.348, 0.541] | 0.595 [0.513, 0.671] |
| FLAME 2023 Open vB sigma sens., coefficienti | 0.671 [0.597, 0.736] | 0.521 [0.439, 0.596] |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | 0.694 [0.600, 0.769] | 0.238 [0.144, 0.331] |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | 0.471 [0.374, 0.558] | 0.547 [0.453, 0.634] |
| factorized s1234, d_F cal. | 0.749 [0.673, 0.806] | 0.318 [0.227, 0.407] |
| factorized s2345, d_F cal. | 0.731 [0.657, 0.792] | 0.312 [0.220, 0.401] |
| factorized s1234, d_P | 0.426 [0.333, 0.510] | 0.622 [0.550, 0.685] |
| factorized s2345, d_P | 0.417 [0.315, 0.512] | 0.613 [0.533, 0.688] |
| ctrlfr s1234 | 0.757 [0.685, 0.817] | 0.339 [0.242, 0.425] |
| ctrlfr s2345 | 0.746 [0.670, 0.809] | 0.348 [0.251, 0.435] |

Delta appaiati, braccio - composizione (lettura dichiarata: FR); a favore / contro / non risolte: FR 4 / 0 / 4

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, composizione S_B + k d_P | +0.051 [+0.009, +0.094], P 0.011 | +0.033 [-0.009, +0.075], P 0.061 | +0.059 [+0.015, +0.101], P 0.002 | +0.048 [+0.007, +0.087], P 0.009 |
| FLAME 2023 Open vB, composizione S_B + k d_P | +0.037 [-0.002, +0.075], P 0.033 | +0.019 [-0.019, +0.057], P 0.154 | +0.046 [+0.006, +0.083], P 0.011 | +0.034 [-0.003, +0.071], P 0.044 |

Delta appaiati, braccio - B della sensibilita' (come l'emendamento 1); a favore / contro / non risolte: FR 23 / 0 / 1, SR 12 / 8 / 4

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB sigma sens., coefficienti | +0.194 [+0.110, +0.275], P 0.000 | +0.176 [+0.096, +0.256], P 0.000 | +0.203 [+0.122, +0.282], P 0.000 | +0.191 [+0.108, +0.273], P 0.000 |
| GNM (visto) vB sigma sens., mesh d'identita' FR | +0.065 [+0.022, +0.109], P 0.002 | +0.048 [+0.004, +0.090], P 0.015 | +0.074 [+0.025, +0.122], P 0.001 | +0.063 [+0.019, +0.107], P 0.003 |
| GNM (visto) vB sigma sens., mesh d'identita' SR | +0.299 [+0.203, +0.401], P 0.000 | +0.282 [+0.189, +0.379], P 0.000 | +0.308 [+0.211, +0.403], P 0.000 | +0.297 [+0.203, +0.389], P 0.000 |
| FLAME 2023 Open vB sigma sens., coefficienti | +0.078 [+0.025, +0.138], P 0.004 | +0.060 [+0.004, +0.119], P 0.018 | +0.086 [+0.036, +0.140], P 0.000 | +0.075 [+0.022, +0.131], P 0.005 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | +0.055 [+0.017, +0.095], P 0.003 | +0.037 [-0.004, +0.077], P 0.034 | +0.063 [+0.022, +0.105], P 0.000 | +0.052 [+0.012, +0.092], P 0.002 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | +0.278 [+0.191, +0.368], P 0.000 | +0.260 [+0.179, +0.347], P 0.000 | +0.286 [+0.206, +0.371], P 0.000 | +0.275 [+0.195, +0.364], P 0.000 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB sigma sens., coefficienti | +0.054 [-0.011, +0.120], P 0.062 | +0.046 [-0.025, +0.118], P 0.090 | -0.229 [-0.315, -0.145], P 1.000 | -0.220 [-0.303, -0.140], P 1.000 |
| GNM (visto) vB sigma sens., mesh d'identita' FR | +0.389 [+0.303, +0.469], P 0.000 | +0.380 [+0.294, +0.469], P 0.000 | +0.106 [+0.057, +0.155], P 0.000 | +0.115 [+0.067, +0.164], P 0.000 |
| GNM (visto) vB sigma sens., mesh d'identita' SR | +0.027 [-0.036, +0.090], P 0.194 | +0.018 [-0.047, +0.084], P 0.274 | -0.256 [-0.353, -0.180], P 1.000 | -0.247 [-0.334, -0.173], P 1.000 |
| FLAME 2023 Open vB sigma sens., coefficienti | +0.101 [+0.033, +0.175], P 0.002 | +0.092 [+0.022, +0.169], P 0.006 | -0.182 [-0.253, -0.116], P 1.000 | -0.173 [-0.242, -0.110], P 1.000 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | +0.383 [+0.303, +0.466], P 0.000 | +0.374 [+0.292, +0.461], P 0.000 | +0.100 [+0.058, +0.148], P 0.000 | +0.109 [+0.069, +0.151], P 0.000 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | +0.075 [+0.017, +0.136], P 0.006 | +0.066 [+0.002, +0.136], P 0.021 | -0.208 [-0.288, -0.130], P 1.000 | -0.199 [-0.274, -0.126], P 1.000 |

## facescape, composizione forma B + taglia B e sensibilita' a sigma (88725 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto) vB, composizione S_B + k d_P | 0.800 [0.741, 0.847] | 0.783 [0.724, 0.839] |
| FLAME 2023 Open vB, composizione S_B + k d_P | 0.731 [0.657, 0.794] | 0.742 [0.677, 0.802] |
| GNM (visto) vB, mesh d'identita' FR | 0.678 [0.615, 0.740] | 0.608 [0.536, 0.683] |
| GNM (visto) vB, mesh d'identita' SR | 0.767 [0.675, 0.843] | 0.854 [0.803, 0.894] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.630 [0.555, 0.695] | 0.590 [0.516, 0.662] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.711 [0.614, 0.798] | 0.820 [0.757, 0.870] |
| GNM (visto) vB sigma sens., coefficienti | 0.600 [0.514, 0.678] | 0.669 [0.592, 0.735] |
| GNM (visto) vB sigma sens., mesh d'identita' FR | 0.719 [0.658, 0.779] | 0.639 [0.567, 0.713] |
| GNM (visto) vB sigma sens., mesh d'identita' SR | 0.782 [0.692, 0.855] | 0.873 [0.829, 0.908] |
| FLAME 2023 Open vB sigma sens., coefficienti | 0.634 [0.537, 0.712] | 0.700 [0.629, 0.759] |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | 0.693 [0.621, 0.755] | 0.645 [0.567, 0.717] |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | 0.737 [0.638, 0.821] | 0.843 [0.791, 0.886] |
| factorized s1234, d_F cal. | 0.661 [0.586, 0.727] | 0.692 [0.623, 0.753] |
| factorized s2345, d_F cal. | 0.669 [0.597, 0.733] | 0.687 [0.620, 0.748] |
| factorized s1234, d_P | 0.677 [0.596, 0.750] | 0.747 [0.677, 0.804] |
| factorized s2345, d_P | 0.684 [0.601, 0.760] | 0.753 [0.687, 0.813] |
| ctrlfr s1234 | 0.661 [0.593, 0.721] | 0.619 [0.543, 0.688] |
| ctrlfr s2345 | 0.653 [0.585, 0.712] | 0.627 [0.554, 0.691] |

Delta appaiati, braccio - composizione (lettura dichiarata: FR); a favore / contro / non risolte: FR 0 / 8 / 0

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, composizione S_B + k d_P | -0.139 [-0.194, -0.090], P 1.000 | -0.131 [-0.184, -0.086], P 1.000 | -0.138 [-0.203, -0.082], P 1.000 | -0.147 [-0.208, -0.092], P 1.000 |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.070 [-0.127, -0.017], P 0.995 | -0.062 [-0.123, -0.009], P 0.992 | -0.070 [-0.149, -0.001], P 0.976 | -0.078 [-0.150, -0.015], P 0.989 |

Delta appaiati, braccio - B della sensibilita' (come l'emendamento 1); a favore / contro / non risolte: FR 2 / 4 / 18, SR 6 / 10 / 8

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB sigma sens., coefficienti | +0.061 [+0.004, +0.118], P 0.015 | +0.069 [+0.007, +0.131], P 0.013 | +0.061 [-0.016, +0.137], P 0.058 | +0.053 [-0.020, +0.121], P 0.073 |
| GNM (visto) vB sigma sens., mesh d'identita' FR | -0.058 [-0.126, +0.004], P 0.967 | -0.050 [-0.116, +0.003], P 0.956 | -0.057 [-0.127, +0.010], P 0.950 | -0.066 [-0.139, +0.002], P 0.970 |
| GNM (visto) vB sigma sens., mesh d'identita' SR | -0.122 [-0.192, -0.040], P 0.997 | -0.114 [-0.187, -0.032], P 0.993 | -0.121 [-0.210, -0.036], P 0.992 | -0.130 [-0.213, -0.038], P 0.999 |
| FLAME 2023 Open vB sigma sens., coefficienti | +0.027 [-0.041, +0.096], P 0.223 | +0.035 [-0.033, +0.109], P 0.186 | +0.028 [-0.062, +0.119], P 0.270 | +0.019 [-0.065, +0.104], P 0.341 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | -0.032 [-0.100, +0.029], P 0.835 | -0.024 [-0.092, +0.030], P 0.781 | -0.032 [-0.114, +0.039], P 0.794 | -0.040 [-0.117, +0.025], P 0.870 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | -0.076 [-0.154, +0.006], P 0.962 | -0.068 [-0.152, +0.020], P 0.943 | -0.076 [-0.174, +0.019], P 0.946 | -0.084 [-0.179, +0.009], P 0.963 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB sigma sens., coefficienti | +0.077 [+0.021, +0.135], P 0.006 | +0.084 [+0.023, +0.144], P 0.009 | -0.050 [-0.111, +0.007], P 0.953 | -0.042 [-0.103, +0.020], P 0.914 |
| GNM (visto) vB sigma sens., mesh d'identita' FR | +0.108 [+0.032, +0.175], P 0.001 | +0.114 [+0.040, +0.177], P 0.000 | -0.020 [-0.094, +0.044], P 0.726 | -0.011 [-0.083, +0.053], P 0.639 |
| GNM (visto) vB sigma sens., mesh d'identita' SR | -0.126 [-0.174, -0.086], P 1.000 | -0.120 [-0.168, -0.080], P 1.000 | -0.254 [-0.313, -0.203], P 1.000 | -0.245 [-0.302, -0.199], P 1.000 |
| FLAME 2023 Open vB sigma sens., coefficienti | +0.046 [-0.012, +0.106], P 0.059 | +0.053 [-0.007, +0.115], P 0.045 | -0.082 [-0.148, -0.019], P 0.992 | -0.073 [-0.144, -0.005], P 0.986 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | +0.102 [+0.025, +0.172], P 0.005 | +0.108 [+0.033, +0.174], P 0.000 | -0.026 [-0.103, +0.041], P 0.764 | -0.017 [-0.095, +0.049], P 0.684 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | -0.097 [-0.154, -0.046], P 1.000 | -0.090 [-0.146, -0.037], P 1.000 | -0.224 [-0.294, -0.163], P 1.000 | -0.216 [-0.283, -0.155], P 1.000 |

## faceverse, composizione forma B + taglia B e sensibilita' a sigma (99000 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto) vB, composizione S_B + k d_P | 0.338 [0.255, 0.407] | 0.306 [0.209, 0.381] |
| FLAME 2023 Open vB, composizione S_B + k d_P | 0.302 [0.222, 0.372] | 0.270 [0.182, 0.349] |
| GNM (visto) vB, mesh d'identita' FR | 0.292 [0.205, 0.363] | 0.244 [0.149, 0.326] |
| GNM (visto) vB, mesh d'identita' SR | 0.319 [0.235, 0.390] | 0.327 [0.239, 0.402] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.264 [0.183, 0.336] | 0.218 [0.125, 0.302] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.290 [0.216, 0.361] | 0.293 [0.209, 0.368] |
| GNM (visto) vB sigma sens., coefficienti | 0.268 [0.177, 0.346] | 0.266 [0.167, 0.347] |
| GNM (visto) vB sigma sens., mesh d'identita' FR | 0.290 [0.200, 0.367] | 0.238 [0.141, 0.326] |
| GNM (visto) vB sigma sens., mesh d'identita' SR | 0.333 [0.253, 0.403] | 0.340 [0.257, 0.410] |
| FLAME 2023 Open vB sigma sens., coefficienti | 0.306 [0.223, 0.380] | 0.289 [0.199, 0.372] |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | 0.274 [0.187, 0.350] | 0.221 [0.125, 0.311] |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | 0.295 [0.220, 0.364] | 0.296 [0.219, 0.369] |
| factorized s1234, d_F cal. | 0.303 [0.229, 0.374] | 0.269 [0.195, 0.341] |
| factorized s2345, d_F cal. | 0.318 [0.249, 0.379] | 0.286 [0.206, 0.353] |
| factorized s1234, d_P | 0.283 [0.210, 0.346] | 0.286 [0.215, 0.352] |
| factorized s2345, d_P | 0.308 [0.234, 0.373] | 0.313 [0.237, 0.377] |
| ctrlfr s1234 | 0.282 [0.215, 0.343] | 0.273 [0.201, 0.339] |
| ctrlfr s2345 | 0.258 [0.187, 0.324] | 0.259 [0.188, 0.325] |

Delta appaiati, braccio - composizione (lettura dichiarata: FR); a favore / contro / non risolte: FR 0 / 0 / 8

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, composizione S_B + k d_P | -0.036 [-0.105, +0.045], P 0.805 | -0.020 [-0.084, +0.053], P 0.716 | -0.056 [-0.134, +0.022], P 0.929 | -0.080 [-0.159, +0.001], P 0.973 |
| FLAME 2023 Open vB, composizione S_B + k d_P | +0.001 [-0.079, +0.089], P 0.497 | +0.017 [-0.061, +0.097], P 0.345 | -0.019 [-0.104, +0.066], P 0.680 | -0.044 [-0.124, +0.044], P 0.835 |

Delta appaiati, braccio - B della sensibilita' (come l'emendamento 1); a favore / contro / non risolte: FR 0 / 0 / 24, SR 0 / 1 / 23

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB sigma sens., coefficienti | +0.035 [-0.049, +0.129], P 0.185 | +0.050 [-0.027, +0.137], P 0.093 | +0.015 [-0.057, +0.090], P 0.314 | -0.010 [-0.081, +0.069], P 0.588 |
| GNM (visto) vB sigma sens., mesh d'identita' FR | +0.013 [-0.062, +0.093], P 0.373 | +0.028 [-0.047, +0.106], P 0.231 | -0.008 [-0.098, +0.083], P 0.589 | -0.032 [-0.123, +0.059], P 0.742 |
| GNM (visto) vB sigma sens., mesh d'identita' SR | -0.031 [-0.114, +0.055], P 0.762 | -0.015 [-0.091, +0.068], P 0.635 | -0.051 [-0.128, +0.027], P 0.904 | -0.075 [-0.152, +0.002], P 0.970 |
| FLAME 2023 Open vB sigma sens., coefficienti | -0.003 [-0.088, +0.085], P 0.516 | +0.012 [-0.067, +0.098], P 0.372 | -0.023 [-0.104, +0.055], P 0.721 | -0.048 [-0.128, +0.040], P 0.870 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | +0.028 [-0.051, +0.116], P 0.266 | +0.044 [-0.034, +0.124], P 0.144 | +0.008 [-0.089, +0.103], P 0.449 | -0.016 [-0.111, +0.078], P 0.625 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | +0.008 [-0.080, +0.099], P 0.434 | +0.023 [-0.063, +0.111], P 0.307 | -0.013 [-0.092, +0.072], P 0.627 | -0.037 [-0.120, +0.048], P 0.808 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB sigma sens., coefficienti | +0.020 [-0.054, +0.099], P 0.270 | +0.047 [-0.023, +0.124], P 0.090 | +0.007 [-0.070, +0.081], P 0.387 | -0.007 [-0.079, +0.070], P 0.542 |
| GNM (visto) vB sigma sens., mesh d'identita' FR | +0.048 [-0.052, +0.144], P 0.168 | +0.075 [-0.016, +0.160], P 0.052 | +0.035 [-0.062, +0.130], P 0.241 | +0.021 [-0.076, +0.113], P 0.344 |
| GNM (visto) vB sigma sens., mesh d'identita' SR | -0.054 [-0.124, +0.018], P 0.921 | -0.027 [-0.101, +0.045], P 0.754 | -0.067 [-0.145, +0.010], P 0.954 | -0.081 [-0.158, -0.006], P 0.982 |
| FLAME 2023 Open vB sigma sens., coefficienti | -0.003 [-0.082, +0.080], P 0.520 | +0.024 [-0.055, +0.106], P 0.276 | -0.017 [-0.101, +0.065], P 0.655 | -0.030 [-0.105, +0.053], P 0.774 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | +0.066 [-0.034, +0.162], P 0.103 | +0.092 [-0.001, +0.176], P 0.027 | +0.052 [-0.050, +0.154], P 0.156 | +0.038 [-0.058, +0.133], P 0.220 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | -0.010 [-0.089, +0.070], P 0.594 | +0.017 [-0.068, +0.096], P 0.337 | -0.023 [-0.111, +0.067], P 0.722 | -0.037 [-0.120, +0.045], P 0.812 |

## faceverse_neutral, composizione forma B + taglia B e sensibilita' a sigma (99000 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto) vB, composizione S_B + k d_P | 0.382 [0.293, 0.458] | 0.358 [0.259, 0.439] |
| FLAME 2023 Open vB, composizione S_B + k d_P | 0.367 [0.280, 0.444] | 0.345 [0.250, 0.430] |
| GNM (visto) vB, mesh d'identita' FR | 0.321 [0.225, 0.400] | 0.275 [0.171, 0.366] |
| GNM (visto) vB, mesh d'identita' SR | 0.373 [0.288, 0.449] | 0.399 [0.315, 0.473] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.315 [0.223, 0.393] | 0.274 [0.173, 0.366] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.358 [0.275, 0.438] | 0.382 [0.297, 0.459] |
| GNM (visto) vB sigma sens., coefficienti | 0.313 [0.218, 0.398] | 0.326 [0.225, 0.410] |
| GNM (visto) vB sigma sens., mesh d'identita' FR | 0.320 [0.221, 0.404] | 0.272 [0.167, 0.368] |
| GNM (visto) vB sigma sens., mesh d'identita' SR | 0.377 [0.292, 0.454] | 0.399 [0.314, 0.474] |
| FLAME 2023 Open vB sigma sens., coefficienti | 0.340 [0.245, 0.428] | 0.343 [0.240, 0.432] |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | 0.320 [0.224, 0.401] | 0.276 [0.170, 0.369] |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | 0.348 [0.267, 0.430] | 0.371 [0.289, 0.450] |
| factorized s1234, d_F cal. | 0.341 [0.263, 0.416] | 0.308 [0.225, 0.387] |
| factorized s2345, d_F cal. | 0.370 [0.298, 0.440] | 0.336 [0.252, 0.409] |
| factorized s1234, d_P | 0.321 [0.237, 0.391] | 0.333 [0.251, 0.405] |
| factorized s2345, d_P | 0.361 [0.276, 0.439] | 0.372 [0.291, 0.444] |
| ctrlfr s1234 | 0.327 [0.248, 0.399] | 0.326 [0.243, 0.400] |
| ctrlfr s2345 | 0.299 [0.214, 0.374] | 0.305 [0.224, 0.377] |

Delta appaiati, braccio - composizione (lettura dichiarata: FR); a favore / contro / non risolte: FR 0 / 0 / 8

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, composizione S_B + k d_P | -0.041 [-0.126, +0.052], P 0.808 | -0.012 [-0.086, +0.074], P 0.605 | -0.055 [-0.142, +0.033], P 0.893 | -0.083 [-0.175, +0.009], P 0.961 |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.026 [-0.123, +0.079], P 0.696 | +0.004 [-0.088, +0.097], P 0.473 | -0.039 [-0.135, +0.050], P 0.798 | -0.067 [-0.167, +0.028], P 0.906 |

Delta appaiati, braccio - B della sensibilita' (come l'emendamento 1); a favore / contro / non risolte: FR 0 / 0 / 24, SR 1 / 1 / 22

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB sigma sens., coefficienti | +0.027 [-0.076, +0.133], P 0.273 | +0.057 [-0.038, +0.163], P 0.118 | +0.014 [-0.073, +0.107], P 0.358 | -0.014 [-0.103, +0.084], P 0.606 |
| GNM (visto) vB sigma sens., mesh d'identita' FR | +0.021 [-0.069, +0.122], P 0.313 | +0.051 [-0.038, +0.147], P 0.120 | +0.008 [-0.091, +0.103], P 0.432 | -0.020 [-0.128, +0.082], P 0.630 |
| GNM (visto) vB sigma sens., mesh d'identita' SR | -0.037 [-0.127, +0.058], P 0.775 | -0.007 [-0.090, +0.083], P 0.571 | -0.050 [-0.137, +0.037], P 0.864 | -0.078 [-0.169, +0.017], P 0.950 |
| FLAME 2023 Open vB sigma sens., coefficienti | +0.000 [-0.097, +0.098], P 0.479 | +0.030 [-0.072, +0.133], P 0.248 | -0.013 [-0.111, +0.086], P 0.607 | -0.041 [-0.147, +0.061], P 0.778 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | +0.021 [-0.075, +0.130], P 0.345 | +0.050 [-0.041, +0.148], P 0.134 | +0.007 [-0.094, +0.106], P 0.436 | -0.021 [-0.129, +0.082], P 0.638 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | -0.008 [-0.107, +0.092], P 0.546 | +0.022 [-0.074, +0.120], P 0.338 | -0.021 [-0.116, +0.072], P 0.684 | -0.049 [-0.153, +0.054], P 0.839 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB sigma sens., coefficienti | +0.008 [-0.081, +0.104], P 0.412 | +0.046 [-0.042, +0.143], P 0.141 | +0.001 [-0.089, +0.099], P 0.476 | -0.020 [-0.115, +0.078], P 0.658 |
| GNM (visto) vB sigma sens., mesh d'identita' FR | +0.062 [-0.044, +0.166], P 0.128 | +0.100 [+0.001, +0.201], P 0.024 | +0.054 [-0.048, +0.156], P 0.146 | +0.034 [-0.064, +0.131], P 0.249 |
| GNM (visto) vB sigma sens., mesh d'identita' SR | -0.065 [-0.143, +0.016], P 0.941 | -0.027 [-0.108, +0.057], P 0.739 | -0.073 [-0.160, +0.010], P 0.949 | -0.093 [-0.183, -0.006], P 0.988 |
| FLAME 2023 Open vB sigma sens., coefficienti | -0.009 [-0.101, +0.083], P 0.569 | +0.029 [-0.065, +0.125], P 0.259 | -0.017 [-0.115, +0.086], P 0.625 | -0.037 [-0.140, +0.059], P 0.767 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | +0.058 [-0.051, +0.169], P 0.145 | +0.096 [-0.008, +0.198], P 0.036 | +0.050 [-0.059, +0.155], P 0.182 | +0.030 [-0.078, +0.131], P 0.285 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | -0.037 [-0.125, +0.051], P 0.808 | +0.001 [-0.093, +0.093], P 0.515 | -0.045 [-0.144, +0.050], P 0.827 | -0.065 [-0.164, +0.022], P 0.914 |

## famos, composizione forma B + taglia B e sensibilita' a sigma (105 righe)

Spearman con la GT (rho, IC 95%):

| metodo | FR | SR |
| --- | --- | --- |
| GNM (visto) vB, composizione S_B + k d_P | 0.753 [0.393, 0.938] | 0.623 [0.285, 0.875] |
| FLAME 2023 Open vB, composizione S_B + k d_P | 0.708 [0.372, 0.923] | 0.661 [0.333, 0.887] |
| GNM (visto) vB, mesh d'identita' FR | 0.686 [0.307, 0.910] | 0.439 [-0.001, 0.794] |
| GNM (visto) vB, mesh d'identita' SR | 0.513 [0.194, 0.749] | 0.766 [0.592, 0.880] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.668 [0.292, 0.908] | 0.483 [0.093, 0.819] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.481 [0.160, 0.714] | 0.791 [0.626, 0.889] |
| GNM (visto) vB sigma sens., coefficienti | 0.669 [0.260, 0.850] | 0.719 [0.427, 0.874] |
| GNM (visto) vB sigma sens., mesh d'identita' FR | 0.728 [0.384, 0.926] | 0.473 [0.062, 0.810] |
| GNM (visto) vB sigma sens., mesh d'identita' SR | 0.520 [0.198, 0.744] | 0.813 [0.643, 0.924] |
| FLAME 2023 Open vB sigma sens., coefficienti | 0.637 [0.242, 0.823] | 0.741 [0.475, 0.877] |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | 0.736 [0.404, 0.929] | 0.524 [0.137, 0.834] |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | 0.574 [0.229, 0.781] | 0.852 [0.701, 0.924] |
| factorized s1234, d_F cal. | 0.678 [0.237, 0.897] | 0.662 [0.323, 0.858] |
| factorized s2345, d_F cal. | 0.654 [0.214, 0.896] | 0.616 [0.225, 0.839] |
| factorized s1234, d_P | 0.412 [0.088, 0.680] | 0.740 [0.548, 0.883] |
| factorized s2345, d_P | 0.437 [0.107, 0.698] | 0.728 [0.516, 0.869] |
| ctrlfr s1234 | 0.831 [0.575, 0.949] | 0.709 [0.394, 0.871] |
| ctrlfr s2345 | 0.818 [0.569, 0.944] | 0.691 [0.378, 0.869] |

Delta appaiati, braccio - composizione (lettura dichiarata: FR); a favore / contro / non risolte: FR 0 / 0 / 8

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, composizione S_B + k d_P | -0.075 [-0.252, +0.050], P 0.903 | -0.099 [-0.260, +0.025], P 0.946 | +0.078 [-0.082, +0.312], P 0.186 | +0.066 [-0.111, +0.307], P 0.221 |
| FLAME 2023 Open vB, composizione S_B + k d_P | -0.030 [-0.218, +0.099], P 0.731 | -0.054 [-0.221, +0.061], P 0.834 | +0.123 [-0.046, +0.341], P 0.091 | +0.110 [-0.079, +0.334], P 0.128 |

Delta appaiati, braccio - B della sensibilita' (come l'emendamento 1); a favore / contro / non risolte: FR 7 / 0 / 17, SR 0 / 0 / 24

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB sigma sens., coefficienti | +0.009 [-0.240, +0.323], P 0.433 | -0.015 [-0.261, +0.283], P 0.509 | +0.162 [-0.001, +0.475], P 0.027 | +0.149 [+0.012, +0.426], P 0.019 |
| GNM (visto) vB sigma sens., mesh d'identita' FR | -0.050 [-0.270, +0.109], P 0.745 | -0.074 [-0.264, +0.068], P 0.876 | +0.103 [-0.075, +0.338], P 0.142 | +0.090 [-0.101, +0.339], P 0.173 |
| GNM (visto) vB sigma sens., mesh d'identita' SR | +0.158 [-0.194, +0.506], P 0.203 | +0.134 [-0.248, +0.486], P 0.271 | +0.311 [+0.023, +0.590], P 0.019 | +0.298 [+0.018, +0.568], P 0.020 |
| FLAME 2023 Open vB sigma sens., coefficienti | +0.041 [-0.155, +0.241], P 0.289 | +0.017 [-0.200, +0.221], P 0.379 | +0.194 [+0.063, +0.448], P 0.004 | +0.182 [+0.046, +0.406], P 0.009 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | -0.058 [-0.299, +0.091], P 0.796 | -0.082 [-0.301, +0.048], P 0.901 | +0.095 [-0.075, +0.316], P 0.168 | +0.082 [-0.102, +0.326], P 0.198 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | +0.104 [-0.156, +0.380], P 0.226 | +0.080 [-0.205, +0.375], P 0.306 | +0.257 [+0.026, +0.538], P 0.018 | +0.245 [+0.016, +0.503], P 0.017 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB sigma sens., coefficienti | +0.022 [-0.256, +0.321], P 0.405 | +0.010 [-0.230, +0.282], P 0.442 | -0.010 [-0.257, +0.186], P 0.530 | -0.028 [-0.250, +0.169], P 0.627 |
| GNM (visto) vB sigma sens., mesh d'identita' FR | +0.267 [-0.156, +0.738], P 0.105 | +0.255 [-0.131, +0.683], P 0.110 | +0.236 [-0.015, +0.526], P 0.033 | +0.218 [-0.002, +0.472], P 0.027 |
| GNM (visto) vB sigma sens., mesh d'identita' SR | -0.072 [-0.206, +0.077], P 0.832 | -0.084 [-0.260, +0.089], P 0.853 | -0.103 [-0.411, +0.142], P 0.825 | -0.122 [-0.437, +0.136], P 0.827 |
| FLAME 2023 Open vB sigma sens., coefficienti | -0.000 [-0.229, +0.272], P 0.462 | -0.012 [-0.209, +0.229], P 0.514 | -0.031 [-0.241, +0.131], P 0.661 | -0.050 [-0.261, +0.129], P 0.744 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' FR | +0.217 [-0.182, +0.642], P 0.153 | +0.205 [-0.158, +0.612], P 0.148 | +0.186 [-0.054, +0.421], P 0.054 | +0.167 [-0.061, +0.381], P 0.066 |
| FLAME 2023 Open vB sigma sens., mesh d'identita' SR | -0.111 [-0.253, +0.030], P 0.924 | -0.123 [-0.289, +0.018], P 0.952 | -0.143 [-0.425, +0.089], P 0.912 | -0.161 [-0.447, +0.080], P 0.908 |

# Emendamento 3 (POST HOC): bracci sulla regione di B, crop sugli held-out sintetici

Protocollo `PROTOCOL_emendamento_3.md` (sha256 `c52cd1e4a52cbeda95dac9a8ae56d193cf96bc60018d0f0a8d12b03bcf488c85`), scritto dopo i numeri dell'emendamento 2. Numeri in `paired_e3.csv`, `spearman_e3.csv`, `heldout_e3.csv`, `controls_e3.json`, `e3/crop_stats.json`.

## Ritaglio alla regione di B (diagnostica, senza GT)

Copertura della regione = frazione d'area della regione di B (collocata su ogni mesh) presente nella mesh con la regola di voto di `bp.region`; area tenuta = area del ritaglio / area della mesh; crop / original = log(sqrt(area crop) / sqrt(area original)) per soggetto, mediana [IQR], prima e dopo il ritaglio.

| vista | regione | fallite | copertura: crop mediana (p5) | altre topologie mediana | area tenuta crop / original | crop / original prima | dopo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | regione GNM (7700 vertici) | 0 | 0.821 (0.742) | 0.975-0.993 | 0.87 / 0.72 | -0.109 [-0.111, -0.106] | -0.009 [-0.019, -0.002] |
| hifi3d | regione FLAME (1517 vertici) | 0 | 0.866 (0.790) | 0.977-0.992 | 0.71 / 0.57 | -0.109 [-0.111, -0.106] | -0.002 [-0.003, -0.001] |
| faceverse | regione GNM (8654 vertici) | 0 | 0.908 (0.864) | 0.954-0.973 | 0.86 / 0.80 | -0.045 [-0.055, -0.035] | -0.009 [-0.020, +0.001] |
| faceverse | regione FLAME (1674 vertici) | 0 | 0.921 (0.860) | 0.949-0.972 | 0.71 / 0.65 | -0.045 [-0.055, -0.035] | +0.001 [-0.010, +0.009] |
| facescape | regione GNM (8061 vertici) | 0 | 0.856 (0.815) | 0.992-0.996 | 0.82 / 0.64 | -0.132 [-0.136, -0.127] | -0.004 [-0.006, -0.002] |
| facescape | regione FLAME (1544 vertici) | 0 | 0.849 (0.784) | 0.989-0.995 | 0.73 / 0.56 | -0.132 [-0.136, -0.127] | -0.001 [-0.001, +0.000] |
| faceverse_neutral | regione GNM (8654 vertici) | 0 | 0.912 (0.866) | 0.961-0.977 | 0.86 / 0.80 | -0.045 [-0.049, -0.042] | -0.009 [-0.014, -0.005] |
| faceverse_neutral | regione FLAME (1674 vertici) | 0 | 0.928 (0.878) | 0.957-0.977 | 0.72 / 0.66 | -0.045 [-0.049, -0.042] | -0.000 [-0.001, -0.000] |

## Spearman dentro le coppie di topologie senza crop (sez. 6, descrittiva, non cieca)

Bracci sull'ingresso intero. Media sulle 20 coppie ordinate di topologie diverse senza crop (righe di all_cross, 4.950 per coppia) e sulle 5 coppie di stessa topologia (righe costruite sulle stesse coppie di soggetti); IC 95% per soggetto col seme di all_cross; min e max = stime puntuali fra le coppie. Lettura dichiarata in grassetto (d_F cal. e ctrlfr con FR, d_P e ctrlfr con SR). Stime puntuali del critic dichiarate prima del calcolo: hifi3d factorized s1234, d_F cal. FR 0.749 [0.730, 0.760], stessa topologia 0.760; facescape factorized s1234, d_F cal. FR 0.667 [0.628, 0.706], stessa topologia 0.706.

| vista | braccio | GT | media 20 senza crop | min (coppia) | max (coppia) | media 5 stessa topologia | min | max | 20 - 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | factorized s1234, d_F cal. | FR | **0.749 [0.675, 0.811]** | 0.730 (remesh -> up60k) | 0.760 (up60k -> original) | **0.760 [0.685, 0.821]** | 0.754 (noisy -> noisy) | 0.764 (remesh -> remesh) | -0.011 [-0.013, -0.009] |
| hifi3d | factorized s1234, d_F cal. | SR | 0.319 [0.229, 0.410] | 0.304 (remesh -> noisy) | 0.333 (original -> remesh) | 0.325 [0.233, 0.417] | 0.314 (noisy -> noisy) | 0.334 (remesh -> remesh) | -0.006 [-0.008, -0.004] |
| hifi3d | factorized s1234, d_P | FR | 0.427 [0.331, 0.520] | 0.397 (noisy -> down8k) | 0.443 (remesh -> original) | 0.434 [0.338, 0.526] | 0.420 (noisy -> noisy) | 0.444 (original -> original) | -0.007 [-0.010, -0.004] |
| hifi3d | factorized s1234, d_P | SR | **0.623 [0.550, 0.688]** | 0.594 (down8k -> noisy) | 0.640 (original -> remesh) | **0.635 [0.561, 0.700]** | 0.619 (noisy -> noisy) | 0.642 (remesh -> remesh) | -0.011 [-0.014, -0.009] |
| hifi3d | factorized s2345, d_F cal. | FR | **0.731 [0.655, 0.794]** | 0.693 (remesh -> up60k) | 0.748 (down8k -> original) | **0.745 [0.669, 0.808]** | 0.735 (noisy -> noisy) | 0.748 (remesh -> remesh) | -0.014 [-0.016, -0.011] |
| hifi3d | factorized s2345, d_F cal. | SR | 0.313 [0.220, 0.408] | 0.299 (remesh -> noisy) | 0.324 (down8k -> remesh) | 0.318 [0.224, 0.414] | 0.307 (noisy -> noisy) | 0.322 (remesh -> remesh) | -0.005 [-0.007, -0.003] |
| hifi3d | factorized s2345, d_P | FR | 0.418 [0.321, 0.521] | 0.397 (noisy -> remesh) | 0.430 (down8k -> original) | 0.423 [0.326, 0.527] | 0.410 (noisy -> noisy) | 0.428 (original -> original) | -0.005 [-0.007, -0.003] |
| hifi3d | factorized s2345, d_P | SR | **0.613 [0.529, 0.688]** | 0.594 (remesh -> noisy) | 0.627 (original -> up60k) | **0.620 [0.535, 0.695]** | 0.597 (noisy -> noisy) | 0.632 (remesh -> remesh) | -0.007 [-0.009, -0.005] |
| hifi3d | ctrlfr s1234 | FR | **0.758 [0.684, 0.820]** | 0.715 (remesh -> up60k) | 0.782 (down8k -> original) | **0.779 [0.705, 0.841]** | 0.768 (noisy -> noisy) | 0.788 (remesh -> remesh) | -0.021 [-0.026, -0.017] |
| hifi3d | ctrlfr s1234 | SR | **0.340 [0.246, 0.439]** | 0.317 (remesh -> noisy) | 0.353 (down8k -> original) | **0.348 [0.253, 0.451]** | 0.336 (noisy -> noisy) | 0.356 (remesh -> remesh) | -0.008 [-0.012, -0.005] |
| hifi3d | ctrlfr s2345 | FR | **0.747 [0.670, 0.811]** | 0.686 (remesh -> noisy) | 0.770 (down8k -> original) | **0.765 [0.690, 0.829]** | 0.745 (noisy -> noisy) | 0.772 (remesh -> remesh) | -0.018 [-0.023, -0.015] |
| hifi3d | ctrlfr s2345 | SR | **0.349 [0.257, 0.448]** | 0.313 (remesh -> noisy) | 0.363 (down8k -> original) | **0.357 [0.263, 0.457]** | 0.341 (noisy -> noisy) | 0.365 (original -> original) | -0.008 [-0.011, -0.005] |
| facescape | factorized s1234, d_F cal. | FR | **0.667 [0.598, 0.732]** | 0.628 (up60k -> noisy) | 0.706 (original -> up60k) | **0.706 [0.640, 0.769]** | 0.691 (down8k -> down8k) | 0.724 (up60k -> up60k) | -0.039 [-0.047, -0.032] |
| facescape | factorized s1234, d_F cal. | SR | 0.698 [0.634, 0.758] | 0.663 (up60k -> noisy) | 0.728 (original -> up60k) | 0.737 [0.676, 0.795] | 0.726 (down8k -> down8k) | 0.747 (up60k -> up60k) | -0.039 [-0.047, -0.032] |
| facescape | factorized s1234, d_P | FR | 0.690 [0.612, 0.760] | 0.662 (noisy -> original) | 0.731 (up60k -> remesh) | 0.718 [0.643, 0.787] | 0.704 (down8k -> down8k) | 0.737 (up60k -> up60k) | -0.028 [-0.035, -0.023] |
| facescape | factorized s1234, d_P | SR | **0.762 [0.697, 0.818]** | 0.741 (down8k -> up60k) | 0.795 (up60k -> remesh) | **0.790 [0.727, 0.843]** | 0.778 (original -> original) | 0.802 (up60k -> up60k) | -0.028 [-0.035, -0.023] |
| facescape | factorized s2345, d_F cal. | FR | **0.673 [0.605, 0.735]** | 0.620 (noisy -> up60k) | 0.710 (original -> up60k) | **0.714 [0.648, 0.775]** | 0.702 (noisy -> noisy) | 0.736 (remesh -> remesh) | -0.041 [-0.050, -0.034] |
| facescape | factorized s2345, d_F cal. | SR | 0.692 [0.628, 0.752] | 0.638 (noisy -> up60k) | 0.725 (original -> down8k) | 0.731 [0.667, 0.790] | 0.716 (noisy -> noisy) | 0.745 (remesh -> remesh) | -0.039 [-0.047, -0.032] |
| facescape | factorized s2345, d_P | FR | 0.692 [0.610, 0.770] | 0.633 (noisy -> original) | 0.728 (up60k -> remesh) | 0.712 [0.630, 0.788] | 0.694 (noisy -> noisy) | 0.737 (remesh -> remesh) | -0.020 [-0.026, -0.014] |
| facescape | factorized s2345, d_P | SR | **0.763 [0.699, 0.818]** | 0.715 (noisy -> original) | 0.796 (up60k -> remesh) | **0.783 [0.720, 0.837]** | 0.767 (noisy -> noisy) | 0.803 (remesh -> remesh) | -0.020 [-0.026, -0.015] |
| facescape | ctrlfr s1234 | FR | **0.705 [0.642, 0.758]** | 0.592 (noisy -> remesh) | 0.760 (remesh -> original) | **0.758 [0.696, 0.811]** | 0.732 (noisy -> noisy) | 0.780 (remesh -> remesh) | -0.054 [-0.066, -0.042] |
| facescape | ctrlfr s1234 | SR | **0.659 [0.584, 0.721]** | 0.562 (noisy -> original) | 0.717 (original -> remesh) | **0.708 [0.632, 0.769]** | 0.672 (noisy -> noisy) | 0.728 (remesh -> remesh) | -0.049 [-0.060, -0.037] |
| facescape | ctrlfr s2345 | FR | **0.696 [0.632, 0.755]** | 0.616 (noisy -> remesh) | 0.739 (up60k -> original) | **0.733 [0.672, 0.790]** | 0.692 (noisy -> noisy) | 0.754 (remesh -> remesh) | -0.037 [-0.046, -0.028] |
| facescape | ctrlfr s2345 | SR | **0.669 [0.596, 0.734]** | 0.607 (remesh -> noisy) | 0.707 (up60k -> original) | **0.702 [0.627, 0.766]** | 0.661 (noisy -> noisy) | 0.722 (remesh -> remesh) | -0.033 [-0.042, -0.025] |
| faceverse | factorized s1234, d_F cal. | FR | **0.304 [0.232, 0.369]** | 0.278 (down8k -> original) | 0.337 (remesh -> noisy) | **0.307 [0.232, 0.376]** | 0.263 (down8k -> down8k) | 0.350 (noisy -> noisy) | -0.002 [-0.011, +0.005] |
| faceverse | factorized s1234, d_F cal. | SR | 0.270 [0.197, 0.342] | 0.234 (original -> down8k) | 0.310 (remesh -> noisy) | 0.273 [0.196, 0.347] | 0.233 (down8k -> down8k) | 0.309 (noisy -> noisy) | -0.002 [-0.011, +0.004] |
| faceverse | factorized s1234, d_P | FR | 0.284 [0.218, 0.352] | 0.256 (remesh -> down8k) | 0.321 (original -> noisy) | 0.290 [0.220, 0.361] | 0.260 (down8k -> down8k) | 0.341 (noisy -> noisy) | -0.005 [-0.015, +0.004] |
| faceverse | factorized s1234, d_P | SR | **0.287 [0.219, 0.355]** | 0.252 (noisy -> down8k) | 0.323 (remesh -> noisy) | **0.294 [0.222, 0.365]** | 0.260 (down8k -> down8k) | 0.336 (noisy -> noisy) | -0.007 [-0.016, +0.001] |
| faceverse | factorized s2345, d_F cal. | FR | **0.319 [0.259, 0.377]** | 0.279 (down8k -> up60k) | 0.350 (noisy -> remesh) | **0.321 [0.260, 0.381]** | 0.286 (down8k -> down8k) | 0.352 (noisy -> noisy) | -0.002 [-0.009, +0.004] |
| faceverse | factorized s2345, d_F cal. | SR | 0.287 [0.213, 0.357] | 0.250 (original -> down8k) | 0.317 (up60k -> noisy) | 0.288 [0.212, 0.361] | 0.257 (down8k -> down8k) | 0.315 (noisy -> noisy) | -0.001 [-0.008, +0.005] |
| faceverse | factorized s2345, d_P | FR | 0.308 [0.240, 0.375] | 0.286 (remesh -> down8k) | 0.339 (up60k -> noisy) | 0.310 [0.240, 0.378] | 0.290 (down8k -> down8k) | 0.344 (noisy -> noisy) | -0.002 [-0.010, +0.005] |
| faceverse | factorized s2345, d_P | SR | **0.314 [0.249, 0.379]** | 0.287 (noisy -> up60k) | 0.343 (up60k -> noisy) | **0.316 [0.249, 0.384]** | 0.293 (down8k -> down8k) | 0.342 (noisy -> noisy) | -0.002 [-0.010, +0.005] |
| faceverse | ctrlfr s1234 | FR | **0.291 [0.226, 0.358]** | 0.239 (remesh -> down8k) | 0.328 (up60k -> original) | **0.301 [0.235, 0.371]** | 0.238 (down8k -> down8k) | 0.355 (noisy -> noisy) | -0.010 [-0.021, +0.000] |
| faceverse | ctrlfr s1234 | SR | **0.281 [0.214, 0.353]** | 0.238 (remesh -> down8k) | 0.310 (up60k -> original) | **0.288 [0.218, 0.363]** | 0.231 (down8k -> down8k) | 0.325 (noisy -> noisy) | -0.007 [-0.018, +0.004] |
| faceverse | ctrlfr s2345 | FR | **0.262 [0.192, 0.335]** | 0.219 (down8k -> up60k) | 0.313 (noisy -> remesh) | **0.272 [0.200, 0.346]** | 0.212 (down8k -> down8k) | 0.352 (noisy -> noisy) | -0.010 [-0.019, -0.001] |
| faceverse | ctrlfr s2345 | SR | **0.263 [0.192, 0.336]** | 0.221 (original -> down8k) | 0.306 (noisy -> remesh) | **0.271 [0.199, 0.344]** | 0.210 (down8k -> down8k) | 0.341 (noisy -> noisy) | -0.007 [-0.017, +0.001] |

## Recupero dei bracci sulle righe col crop (criteri della sez. 1)

rho senza crop e righe col crop del braccio intero, righe col crop del braccio sulla regione; F = recupero della frazione del calo; D3 = braccio sulla regione - braccio intero (righe col crop; media dentro le 5 coppie col crop). Nota (critic dell'emendamento 3): a regione uguale righe col crop e senza crop coincidono (da -0.070 a +0.045 con FR), quindi F misura solo il costo o il guadagno del ritaglio sulle righe senza crop e le letture di questa tabella sono meccaniche; la lettura corretta e' in cima, nella sezione dell'emendamento 3.

| vista | GT | braccio | regione | rho senza crop | rho col crop | col crop sulla regione | F | D3 righe col crop | D3 media crop 5 | lettura |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hifi3d | FR | factorized s1234, d_F cal. | regione GNM | 0.749 | 0.663 | 0.680 | 0.20 | +0.017 [-0.031, +0.060] | +0.017 [-0.032, +0.059] | limite del descrittore |
| hifi3d | FR | factorized s1234, d_F cal. | regione FLAME | 0.749 | 0.663 | 0.617 | -0.53 | -0.046 [-0.120, +0.022] | -0.047 [-0.121, +0.021] | limite del descrittore |
| hifi3d | FR | factorized s2345, d_F cal. | regione GNM | 0.731 | 0.552 | 0.645 | 0.52 | +0.093 [+0.046, +0.140] | +0.092 [+0.045, +0.138] | dipendenza dal supporto |
| hifi3d | FR | factorized s2345, d_F cal. | regione FLAME | 0.731 | 0.552 | 0.581 | 0.16 | +0.029 [-0.044, +0.098] | +0.028 [-0.045, +0.096] | limite del descrittore |
| hifi3d | FR | ctrlfr s1234 | regione GNM | 0.757 | 0.236 | 0.602 | 0.70 | +0.367 [+0.309, +0.422] | +0.367 [+0.310, +0.422] | dipendenza dal supporto |
| hifi3d | FR | ctrlfr s1234 | regione FLAME | 0.757 | 0.236 | 0.606 | 0.71 | +0.371 [+0.308, +0.430] | +0.371 [+0.307, +0.430] | dipendenza dal supporto |
| hifi3d | FR | ctrlfr s2345 | regione GNM | 0.746 | 0.279 | 0.622 | 0.74 | +0.344 [+0.291, +0.394] | +0.344 [+0.291, +0.394] | dipendenza dal supporto |
| hifi3d | FR | ctrlfr s2345 | regione FLAME | 0.746 | 0.279 | 0.617 | 0.72 | +0.338 [+0.275, +0.401] | +0.337 [+0.273, +0.399] | dipendenza dal supporto |
| hifi3d | FR | C3M e123, d_F cal. | regione GNM | 0.748 | 0.506 | 0.663 | 0.65 | +0.157 [+0.109, +0.205] | +0.157 [+0.110, +0.206] | dipendenza dal supporto (C3M, descrittivo) |
| hifi3d | FR | C3M e123, d_F cal. | regione FLAME | 0.748 | 0.506 | 0.590 | 0.35 | +0.084 [+0.015, +0.150] | +0.084 [+0.015, +0.149] | misto (C3M, descrittivo) |
| hifi3d | FR | C3M e205, d_F cal. | regione GNM | 0.739 | 0.472 | 0.685 | 0.80 | +0.213 [+0.166, +0.262] | +0.216 [+0.169, +0.267] | dipendenza dal supporto (C3M, descrittivo) |
| hifi3d | FR | C3M e205, d_F cal. | regione FLAME | 0.739 | 0.472 | 0.621 | 0.56 | +0.149 [+0.086, +0.206] | +0.149 [+0.086, +0.208] | dipendenza dal supporto (C3M, descrittivo) |
| hifi3d | SR | factorized s1234, d_P | regione GNM | 0.622 | 0.550 | 0.594 | 0.62 | +0.045 [+0.001, +0.089] | +0.044 [-0.000, +0.089] | dipendenza dal supporto |
| hifi3d | SR | factorized s1234, d_P | regione FLAME | 0.622 | 0.550 | 0.485 | -0.90 | -0.065 [-0.160, +0.025] | -0.066 [-0.161, +0.024] | limite del descrittore |
| hifi3d | SR | factorized s2345, d_P | regione GNM | 0.613 | 0.514 | 0.591 | 0.78 | +0.077 [+0.029, +0.124] | +0.076 [+0.028, +0.122] | dipendenza dal supporto |
| hifi3d | SR | factorized s2345, d_P | regione FLAME | 0.613 | 0.514 | 0.480 | -0.34 | -0.034 [-0.128, +0.056] | -0.035 [-0.130, +0.055] | limite del descrittore |
| hifi3d | SR | ctrlfr s1234 | regione GNM | 0.339 | 0.164 | 0.453 | 1.65 | +0.290 [+0.230, +0.354] | +0.290 [+0.230, +0.354] | dipendenza dal supporto |
| hifi3d | SR | ctrlfr s1234 | regione FLAME | 0.339 | 0.164 | 0.440 | 1.58 | +0.277 [+0.185, +0.366] | +0.276 [+0.185, +0.366] | dipendenza dal supporto |
| hifi3d | SR | ctrlfr s2345 | regione GNM | 0.348 | 0.188 | 0.430 | 1.52 | +0.242 [+0.186, +0.302] | +0.242 [+0.186, +0.302] | dipendenza dal supporto |
| hifi3d | SR | ctrlfr s2345 | regione FLAME | 0.348 | 0.188 | 0.417 | 1.43 | +0.228 [+0.143, +0.311] | +0.227 [+0.142, +0.310] | dipendenza dal supporto |
| hifi3d | SR | C3M e123, d_P | regione GNM | 0.595 | 0.462 | 0.514 | 0.39 | +0.052 [-0.011, +0.111] | +0.049 [-0.016, +0.107] | limite del descrittore (C3M, descrittivo) |
| hifi3d | SR | C3M e123, d_P | regione FLAME | 0.595 | 0.462 | 0.460 | -0.02 | -0.002 [-0.093, +0.087] | -0.006 [-0.098, +0.084] | limite del descrittore (C3M, descrittivo) |
| hifi3d | SR | C3M e205, d_P | regione GNM | 0.591 | 0.481 | 0.544 | 0.57 | +0.062 [+0.005, +0.120] | +0.063 [+0.005, +0.122] | dipendenza dal supporto (C3M, descrittivo) |
| hifi3d | SR | C3M e205, d_P | regione FLAME | 0.591 | 0.481 | 0.471 | -0.10 | -0.011 [-0.107, +0.085] | -0.013 [-0.111, +0.084] | limite del descrittore (C3M, descrittivo) |
| facescape | FR | factorized s1234, d_F cal. | regione GNM | 0.659 | 0.351 | 0.687 | 1.09 | +0.336 [+0.269, +0.401] | +0.316 [+0.247, +0.382] | dipendenza dal supporto |
| facescape | FR | factorized s1234, d_F cal. | regione FLAME | 0.659 | 0.351 | 0.664 | 1.02 | +0.313 [+0.243, +0.383] | +0.293 [+0.218, +0.365] | dipendenza dal supporto |
| facescape | FR | factorized s2345, d_F cal. | regione GNM | 0.667 | 0.278 | 0.703 | 1.09 | +0.425 [+0.352, +0.497] | +0.404 [+0.327, +0.477] | dipendenza dal supporto |
| facescape | FR | factorized s2345, d_F cal. | regione FLAME | 0.667 | 0.278 | 0.697 | 1.08 | +0.419 [+0.342, +0.496] | +0.398 [+0.319, +0.478] | dipendenza dal supporto |
| facescape | FR | ctrlfr s1234 | regione GNM | 0.663 | 0.264 | 0.714 | 1.13 | +0.450 [+0.385, +0.506] | +0.451 [+0.384, +0.512] | dipendenza dal supporto |
| facescape | FR | ctrlfr s1234 | regione FLAME | 0.663 | 0.264 | 0.684 | 1.05 | +0.420 [+0.355, +0.479] | +0.434 [+0.366, +0.497] | dipendenza dal supporto |
| facescape | FR | ctrlfr s2345 | regione GNM | 0.652 | 0.234 | 0.616 | 0.91 | +0.382 [+0.329, +0.433] | +0.389 [+0.333, +0.443] | dipendenza dal supporto |
| facescape | FR | ctrlfr s2345 | regione FLAME | 0.652 | 0.234 | 0.692 | 1.10 | +0.458 [+0.396, +0.517] | +0.457 [+0.392, +0.520] | dipendenza dal supporto |
| facescape | FR | C3M e123, d_F cal. | regione GNM | 0.666 | 0.472 | 0.661 | 0.97 | +0.189 [+0.120, +0.257] | +0.193 [+0.122, +0.260] | dipendenza dal supporto (C3M, descrittivo) |
| facescape | FR | C3M e123, d_F cal. | regione FLAME | 0.666 | 0.472 | 0.612 | 0.72 | +0.139 [+0.070, +0.210] | +0.142 [+0.071, +0.212] | dipendenza dal supporto (C3M, descrittivo) |
| facescape | FR | C3M e205, d_F cal. | regione GNM | 0.712 | 0.449 | 0.674 | 0.86 | +0.226 [+0.157, +0.293] | +0.239 [+0.171, +0.304] | dipendenza dal supporto (C3M, descrittivo) |
| facescape | FR | C3M e205, d_F cal. | regione FLAME | 0.712 | 0.449 | 0.633 | 0.70 | +0.184 [+0.111, +0.255] | +0.195 [+0.120, +0.270] | dipendenza dal supporto (C3M, descrittivo) |
| facescape | SR | factorized s1234, d_P | regione GNM | 0.747 | 0.518 | 0.787 | 1.18 | +0.269 [+0.218, +0.326] | +0.260 [+0.206, +0.318] | dipendenza dal supporto |
| facescape | SR | factorized s1234, d_P | regione FLAME | 0.747 | 0.518 | 0.750 | 1.01 | +0.232 [+0.175, +0.298] | +0.222 [+0.163, +0.289] | dipendenza dal supporto |
| facescape | SR | factorized s2345, d_P | regione GNM | 0.754 | 0.476 | 0.785 | 1.11 | +0.308 [+0.253, +0.370] | +0.303 [+0.246, +0.366] | dipendenza dal supporto |
| facescape | SR | factorized s2345, d_P | regione FLAME | 0.754 | 0.476 | 0.786 | 1.12 | +0.310 [+0.250, +0.381] | +0.304 [+0.243, +0.377] | dipendenza dal supporto |
| facescape | SR | ctrlfr s1234 | regione GNM | 0.621 | 0.247 | 0.724 | 1.28 | +0.477 [+0.430, +0.523] | +0.479 [+0.430, +0.528] | dipendenza dal supporto |
| facescape | SR | ctrlfr s1234 | regione FLAME | 0.621 | 0.247 | 0.682 | 1.16 | +0.435 [+0.383, +0.486] | +0.448 [+0.393, +0.503] | dipendenza dal supporto |
| facescape | SR | ctrlfr s2345 | regione GNM | 0.627 | 0.243 | 0.644 | 1.04 | +0.401 [+0.352, +0.447] | +0.408 [+0.357, +0.459] | dipendenza dal supporto |
| facescape | SR | ctrlfr s2345 | regione FLAME | 0.627 | 0.243 | 0.690 | 1.16 | +0.446 [+0.391, +0.501] | +0.445 [+0.386, +0.502] | dipendenza dal supporto |
| facescape | SR | C3M e123, d_P | regione GNM | 0.746 | 0.578 | 0.759 | 1.08 | +0.181 [+0.135, +0.232] | +0.185 [+0.138, +0.237] | dipendenza dal supporto (C3M, descrittivo) |
| facescape | SR | C3M e123, d_P | regione FLAME | 0.746 | 0.578 | 0.709 | 0.78 | +0.131 [+0.083, +0.183] | +0.133 [+0.083, +0.186] | dipendenza dal supporto (C3M, descrittivo) |
| facescape | SR | C3M e205, d_P | regione GNM | 0.756 | 0.555 | 0.763 | 1.03 | +0.208 [+0.165, +0.255] | +0.220 [+0.175, +0.271] | dipendenza dal supporto (C3M, descrittivo) |
| facescape | SR | C3M e205, d_P | regione FLAME | 0.756 | 0.555 | 0.723 | 0.84 | +0.169 [+0.115, +0.222] | +0.173 [+0.116, +0.230] | dipendenza dal supporto (C3M, descrittivo) |
| faceverse | FR | factorized s1234, d_F cal. | regione GNM | 0.303 | 0.252 | 0.305 | 1.05 | +0.053 [+0.000, +0.103] | +0.053 [+0.000, +0.103] | dipendenza dal supporto |
| faceverse | FR | factorized s1234, d_F cal. | regione FLAME | 0.303 | 0.252 | 0.231 | -0.43 | -0.021 [-0.099, +0.051] | -0.022 [-0.100, +0.051] | limite del descrittore |
| faceverse | FR | factorized s2345, d_F cal. | regione GNM | 0.318 | 0.234 | 0.275 | 0.49 | +0.041 [-0.018, +0.097] | +0.041 [-0.018, +0.096] | limite del descrittore |
| faceverse | FR | factorized s2345, d_F cal. | regione FLAME | 0.318 | 0.234 | 0.208 | -0.31 | -0.026 [-0.104, +0.048] | -0.026 [-0.104, +0.048] | limite del descrittore |
| faceverse | FR | ctrlfr s1234 | regione GNM | 0.282 | 0.256 | 0.276 | nan | +0.020 [-0.037, +0.070] | +0.018 [-0.039, +0.070] | calo < 0.05 |
| faceverse | FR | ctrlfr s1234 | regione FLAME | 0.282 | 0.256 | 0.223 | nan | -0.033 [-0.108, +0.040] | -0.038 [-0.114, +0.035] | calo < 0.05 |
| faceverse | FR | ctrlfr s2345 | regione GNM | 0.258 | 0.239 | 0.257 | nan | +0.019 [-0.037, +0.073] | +0.020 [-0.036, +0.074] | calo < 0.05 |
| faceverse | FR | ctrlfr s2345 | regione FLAME | 0.258 | 0.239 | 0.200 | nan | -0.038 [-0.108, +0.031] | -0.040 [-0.110, +0.030] | calo < 0.05 |
| faceverse | FR | C3M e123, d_F cal. | regione GNM | 0.289 | 0.253 | 0.272 | nan | +0.019 [-0.035, +0.074] | +0.018 [-0.037, +0.073] | calo < 0.05 (C3M, descrittivo) |
| faceverse | FR | C3M e123, d_F cal. | regione FLAME | 0.289 | 0.253 | 0.219 | nan | -0.034 [-0.110, +0.042] | -0.036 [-0.112, +0.042] | calo < 0.05 (C3M, descrittivo) |
| faceverse | FR | C3M e205, d_F cal. | regione GNM | 0.290 | 0.250 | 0.269 | nan | +0.019 [-0.027, +0.067] | +0.018 [-0.031, +0.068] | calo < 0.05 (C3M, descrittivo) |
| faceverse | FR | C3M e205, d_F cal. | regione FLAME | 0.290 | 0.250 | 0.219 | nan | -0.031 [-0.102, +0.031] | -0.033 [-0.107, +0.031] | calo < 0.05 (C3M, descrittivo) |
| faceverse | SR | factorized s1234, d_P | regione GNM | 0.286 | 0.251 | 0.312 | nan | +0.060 [+0.010, +0.115] | +0.060 [+0.010, +0.115] | calo < 0.05 |
| faceverse | SR | factorized s1234, d_P | regione FLAME | 0.286 | 0.251 | 0.249 | nan | -0.002 [-0.085, +0.082] | -0.003 [-0.086, +0.081] | calo < 0.05 |
| faceverse | SR | factorized s2345, d_P | regione GNM | 0.313 | 0.257 | 0.293 | 0.64 | +0.036 [-0.025, +0.097] | +0.036 [-0.026, +0.097] | limite del descrittore |
| faceverse | SR | factorized s2345, d_P | regione FLAME | 0.313 | 0.257 | 0.243 | -0.25 | -0.014 [-0.096, +0.063] | -0.015 [-0.096, +0.063] | limite del descrittore |
| faceverse | SR | ctrlfr s1234 | regione GNM | 0.273 | 0.239 | 0.271 | nan | +0.032 [-0.030, +0.086] | +0.030 [-0.032, +0.085] | calo < 0.05 |
| faceverse | SR | ctrlfr s1234 | regione FLAME | 0.273 | 0.239 | 0.225 | nan | -0.014 [-0.090, +0.062] | -0.019 [-0.097, +0.056] | calo < 0.05 |
| faceverse | SR | ctrlfr s2345 | regione GNM | 0.259 | 0.243 | 0.264 | nan | +0.021 [-0.035, +0.075] | +0.022 [-0.034, +0.076] | calo < 0.05 |
| faceverse | SR | ctrlfr s2345 | regione FLAME | 0.259 | 0.243 | 0.205 | nan | -0.038 [-0.107, +0.031] | -0.039 [-0.109, +0.030] | calo < 0.05 |
| faceverse | SR | C3M e123, d_P | regione GNM | 0.313 | 0.294 | 0.305 | nan | +0.011 [-0.044, +0.072] | +0.011 [-0.045, +0.072] | calo < 0.05 (C3M, descrittivo) |
| faceverse | SR | C3M e123, d_P | regione FLAME | 0.313 | 0.294 | 0.254 | nan | -0.040 [-0.118, +0.031] | -0.040 [-0.118, +0.032] | calo < 0.05 (C3M, descrittivo) |
| faceverse | SR | C3M e205, d_P | regione GNM | 0.314 | 0.308 | 0.305 | nan | -0.003 [-0.046, +0.046] | -0.003 [-0.046, +0.048] | calo < 0.05 (C3M, descrittivo) |
| faceverse | SR | C3M e205, d_P | regione FLAME | 0.314 | 0.308 | 0.248 | nan | -0.060 [-0.130, +0.004] | -0.059 [-0.130, +0.005] | calo < 0.05 (C3M, descrittivo) |
| faceverse_neutral | FR | factorized s1234, d_F cal. | regione GNM | 0.341 | 0.282 | 0.359 | 1.31 | +0.076 [+0.022, +0.130] | +0.076 [+0.022, +0.130] | dipendenza dal supporto |
| faceverse_neutral | FR | factorized s1234, d_F cal. | regione FLAME | 0.341 | 0.282 | 0.317 | 0.59 | +0.034 [-0.042, +0.115] | +0.034 [-0.042, +0.115] | limite del descrittore |
| faceverse_neutral | FR | factorized s2345, d_F cal. | regione GNM | 0.370 | 0.272 | 0.334 | 0.63 | +0.061 [-0.000, +0.124] | +0.061 [-0.001, +0.124] | limite del descrittore |
| faceverse_neutral | FR | factorized s2345, d_F cal. | regione FLAME | 0.370 | 0.272 | 0.296 | 0.24 | +0.023 [-0.062, +0.103] | +0.023 [-0.062, +0.103] | limite del descrittore |
| faceverse_neutral | FR | ctrlfr s1234 | regione GNM | 0.327 | 0.296 | 0.356 | nan | +0.060 [+0.000, +0.119] | +0.057 [-0.004, +0.118] | calo < 0.05 |
| faceverse_neutral | FR | ctrlfr s1234 | regione FLAME | 0.327 | 0.296 | 0.315 | nan | +0.019 [-0.057, +0.096] | +0.012 [-0.065, +0.090] | calo < 0.05 |
| faceverse_neutral | FR | ctrlfr s2345 | regione GNM | 0.299 | 0.269 | 0.329 | nan | +0.060 [-0.011, +0.125] | +0.061 [-0.010, +0.129] | calo < 0.05 |
| faceverse_neutral | FR | ctrlfr s2345 | regione FLAME | 0.299 | 0.269 | 0.282 | nan | +0.012 [-0.066, +0.092] | +0.010 [-0.069, +0.089] | calo < 0.05 |
| faceverse_neutral | FR | C3M e205, d_F cal. | regione GNM | 0.336 | 0.299 | 0.320 | nan | +0.021 [-0.028, +0.071] | +0.019 [-0.033, +0.072] | calo < 0.05 (C3M, descrittivo) |
| faceverse_neutral | FR | C3M e205, d_F cal. | regione FLAME | 0.336 | 0.299 | 0.290 | nan | -0.009 [-0.080, +0.059] | -0.011 [-0.085, +0.060] | calo < 0.05 (C3M, descrittivo) |
| faceverse_neutral | SR | factorized s1234, d_P | regione GNM | 0.333 | 0.304 | 0.378 | nan | +0.073 [+0.012, +0.138] | +0.074 [+0.012, +0.138] | calo < 0.05 |
| faceverse_neutral | SR | factorized s1234, d_P | regione FLAME | 0.333 | 0.304 | 0.349 | nan | +0.045 [-0.036, +0.124] | +0.044 [-0.036, +0.124] | calo < 0.05 |
| faceverse_neutral | SR | factorized s2345, d_P | regione GNM | 0.372 | 0.316 | 0.360 | 0.78 | +0.044 [-0.027, +0.116] | +0.043 [-0.028, +0.116] | limite del descrittore |
| faceverse_neutral | SR | factorized s2345, d_P | regione FLAME | 0.372 | 0.316 | 0.325 | 0.17 | +0.009 [-0.085, +0.095] | +0.009 [-0.086, +0.095] | limite del descrittore |
| faceverse_neutral | SR | ctrlfr s1234 | regione GNM | 0.326 | 0.291 | 0.359 | nan | +0.068 [+0.005, +0.125] | +0.065 [+0.000, +0.123] | calo < 0.05 |
| faceverse_neutral | SR | ctrlfr s1234 | regione FLAME | 0.326 | 0.291 | 0.323 | nan | +0.032 [-0.045, +0.107] | +0.025 [-0.053, +0.101] | calo < 0.05 |
| faceverse_neutral | SR | ctrlfr s2345 | regione GNM | 0.305 | 0.274 | 0.335 | nan | +0.060 [-0.011, +0.127] | +0.062 [-0.010, +0.130] | calo < 0.05 |
| faceverse_neutral | SR | ctrlfr s2345 | regione FLAME | 0.305 | 0.274 | 0.296 | nan | +0.021 [-0.058, +0.101] | +0.019 [-0.060, +0.099] | calo < 0.05 |
| faceverse_neutral | SR | C3M e205, d_P | regione GNM | 0.382 | 0.372 | 0.369 | nan | -0.003 [-0.056, +0.053] | -0.001 [-0.054, +0.054] | calo < 0.05 (C3M, descrittivo) |
| faceverse_neutral | SR | C3M e205, d_P | regione FLAME | 0.382 | 0.372 | 0.341 | nan | -0.031 [-0.101, +0.038] | -0.027 [-0.098, +0.044] | calo < 0.05 (C3M, descrittivo) |

## Conteggi dei delta dichiarati (a favore / contro / non risolte)

D1 = braccio sulla regione di m - B di m (24 per GT); D2 = braccio sulla regione - baseline geometriche (24); D2 intero = braccio sull'ingresso intero - baseline geometriche (12); D3 = braccio sulla regione - braccio intero (8).

| vista | gruppo | GT | D1 | D2 | D2 intero | D3 |
| --- | --- | --- | --- | --- | --- | --- |
| hifi3d | senza crop | FR | 12 / 3 / 9 | 0 / 0 / 24 | 12 / 0 / 0 | 0 / 8 / 0 |
| hifi3d | senza crop | SR | 8 / 13 / 3 | 22 / 0 / 2 | 10 / 0 / 2 | 3 / 2 / 3 |
| hifi3d | all_cross | FR | 12 / 1 / 11 | 9 / 0 / 15 | 5 / 2 / 5 | 4 / 2 / 2 |
| hifi3d | all_cross | SR | 8 / 10 / 6 | 24 / 0 / 0 | 10 / 2 / 0 | 4 / 0 / 4 |
| hifi3d | righe col crop | FR | 11 / 0 / 13 | 16 / 0 / 8 | 4 / 6 / 2 | 5 / 0 / 3 |
| hifi3d | righe col crop | SR | 8 / 10 / 6 | 24 / 0 / 0 | 10 / 2 / 0 | 6 / 0 / 2 |
| hifi3d | media 15 coppie | FR | 12 / 1 / 11 | 7 / 0 / 17 | 7 / 0 / 5 | 2 / 2 / 4 |
| hifi3d | media 15 coppie | SR | 8 / 10 / 6 | 23 / 0 / 1 | 10 / 1 / 1 | 4 / 2 / 2 |
| hifi3d | media crop 5 coppie | FR | 11 / 0 / 13 | 16 / 0 / 8 | 4 / 6 / 2 | 5 / 0 / 3 |
| hifi3d | media crop 5 coppie | SR | 8 / 10 / 6 | 24 / 0 / 0 | 10 / 2 / 0 | 5 / 0 / 3 |
| facescape | senza crop | FR | 8 / 2 / 14 | 20 / 0 / 4 | 9 / 0 / 3 | 2 / 0 / 6 |
| facescape | senza crop | SR | 13 / 7 / 4 | 23 / 0 / 1 | 10 / 0 / 2 | 3 / 0 / 5 |
| facescape | all_cross | FR | 11 / 2 / 11 | 24 / 0 / 0 | 4 / 5 / 3 | 8 / 0 / 0 |
| facescape | all_cross | SR | 16 / 7 / 1 | 24 / 0 / 0 | 6 / 6 / 0 | 8 / 0 / 0 |
| facescape | righe col crop | FR | 13 / 2 / 9 | 24 / 0 / 0 | 5 / 4 / 3 | 8 / 0 / 0 |
| facescape | righe col crop | SR | 15 / 6 / 3 | 24 / 0 / 0 | 6 / 2 / 4 | 8 / 0 / 0 |
| facescape | media 15 coppie | FR | 12 / 2 / 10 | 18 / 0 / 6 | 8 / 4 / 0 | 8 / 0 / 0 |
| facescape | media 15 coppie | SR | 16 / 7 / 1 | 22 / 0 / 2 | 6 / 2 / 4 | 8 / 0 / 0 |
| facescape | media crop 5 coppie | FR | 14 / 2 / 8 | 22 / 0 / 2 | 4 / 6 / 2 | 8 / 0 / 0 |
| facescape | media crop 5 coppie | SR | 16 / 5 / 3 | 23 / 0 / 1 | 6 / 5 / 1 | 8 / 0 / 0 |
| faceverse | senza crop | FR | 0 / 1 / 23 | 0 / 13 / 11 | 2 / 2 / 8 | 0 / 1 / 7 |
| faceverse | senza crop | SR | 0 / 0 / 24 | 4 / 7 / 13 | 4 / 2 / 6 | 0 / 0 / 8 |
| faceverse | all_cross | FR | 0 / 0 / 24 | 3 / 11 / 10 | 3 / 4 / 5 | 0 / 1 / 7 |
| faceverse | all_cross | SR | 0 / 0 / 24 | 7 / 7 / 10 | 4 / 3 / 5 | 0 / 0 / 8 |
| faceverse | righe col crop | FR | 0 / 0 / 24 | 7 / 7 / 10 | 4 / 4 / 4 | 1 / 0 / 7 |
| faceverse | righe col crop | SR | 0 / 0 / 24 | 8 / 6 / 10 | 4 / 4 / 4 | 1 / 0 / 7 |
| faceverse | media 15 coppie | FR | 0 / 0 / 24 | 2 / 11 / 11 | 3 / 4 / 5 | 0 / 1 / 7 |
| faceverse | media 15 coppie | SR | 0 / 0 / 24 | 7 / 7 / 10 | 4 / 3 / 5 | 0 / 0 / 8 |
| faceverse | media crop 5 coppie | FR | 0 / 0 / 24 | 7 / 7 / 10 | 4 / 4 / 4 | 1 / 0 / 7 |
| faceverse | media crop 5 coppie | SR | 0 / 0 / 24 | 8 / 4 / 12 | 4 / 4 / 4 | 1 / 0 / 7 |
| faceverse_neutral | senza crop | FR | 0 / 0 / 24 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 8 |
| faceverse_neutral | senza crop | SR | 0 / 0 / 24 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 8 |
| faceverse_neutral | all_cross | FR | 0 / 0 / 24 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 8 |
| faceverse_neutral | all_cross | SR | 0 / 0 / 24 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 8 |
| faceverse_neutral | righe col crop | FR | 0 / 0 / 24 | 0 / 0 / 0 | 0 / 0 / 0 | 2 / 0 / 6 |
| faceverse_neutral | righe col crop | SR | 0 / 0 / 24 | 0 / 0 / 0 | 0 / 0 / 0 | 2 / 0 / 6 |
| faceverse_neutral | media 15 coppie | FR | 0 / 0 / 24 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 8 |
| faceverse_neutral | media 15 coppie | SR | 0 / 0 / 24 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 8 |
| faceverse_neutral | media crop 5 coppie | FR | 0 / 0 / 24 | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 7 |
| faceverse_neutral | media crop 5 coppie | SR | 0 / 0 / 24 | 0 / 0 / 0 | 0 / 0 / 0 | 2 / 0 / 6 |

## hifi3d: Spearman con la GT per gruppo (rho, IC 95%; 148500 righe all_cross, seme 757683)

GT FR:

| metodo | senza crop | all_cross | righe col crop | media 15 coppie | media crop 5 coppie |
| --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal. | 0.749 [0.674, 0.810] | 0.709 [0.635, 0.771] | 0.663 [0.592, 0.722] | 0.720 [0.649, 0.781] | 0.663 [0.593, 0.723] |
| factorized s1234, d_F cal. @ regione GNM | 0.688 [0.618, 0.746] | 0.684 [0.614, 0.743] | 0.680 [0.610, 0.742] | 0.686 [0.616, 0.746] | 0.680 [0.610, 0.742] |
| factorized s1234, d_F cal. @ regione FLAME | 0.609 [0.534, 0.685] | 0.611 [0.536, 0.687] | 0.617 [0.541, 0.693] | 0.612 [0.537, 0.688] | 0.617 [0.542, 0.693] |
| factorized s2345, d_F cal. | 0.731 [0.656, 0.794] | 0.661 [0.587, 0.724] | 0.552 [0.480, 0.615] | 0.672 [0.599, 0.733] | 0.554 [0.482, 0.616] |
| factorized s2345, d_F cal. @ regione GNM | 0.651 [0.587, 0.711] | 0.647 [0.583, 0.706] | 0.645 [0.581, 0.705] | 0.649 [0.585, 0.708] | 0.646 [0.581, 0.705] |
| factorized s2345, d_F cal. @ regione FLAME | 0.573 [0.492, 0.651] | 0.576 [0.495, 0.655] | 0.581 [0.500, 0.660] | 0.576 [0.495, 0.655] | 0.582 [0.501, 0.660] |
| ctrlfr s1234 | 0.757 [0.682, 0.819] | 0.516 [0.444, 0.580] | 0.236 [0.183, 0.283] | 0.585 [0.524, 0.638] | 0.238 [0.186, 0.286] |
| ctrlfr s1234 @ regione GNM | 0.655 [0.587, 0.716] | 0.633 [0.566, 0.693] | 0.602 [0.530, 0.667] | 0.641 [0.575, 0.700] | 0.606 [0.534, 0.670] |
| ctrlfr s1234 @ regione FLAME | 0.597 [0.516, 0.673] | 0.600 [0.519, 0.676] | 0.606 [0.526, 0.681] | 0.604 [0.524, 0.679] | 0.609 [0.529, 0.683] |
| ctrlfr s2345 | 0.746 [0.669, 0.811] | 0.530 [0.456, 0.594] | 0.279 [0.220, 0.331] | 0.592 [0.528, 0.644] | 0.281 [0.224, 0.334] |
| ctrlfr s2345 @ regione GNM | 0.692 [0.630, 0.749] | 0.663 [0.599, 0.720] | 0.622 [0.549, 0.692] | 0.672 [0.607, 0.731] | 0.625 [0.552, 0.695] |
| ctrlfr s2345 @ regione FLAME | 0.612 [0.527, 0.692] | 0.614 [0.528, 0.694] | 0.617 [0.532, 0.699] | 0.615 [0.531, 0.696] | 0.618 [0.533, 0.700] |
| C3M e123, d_F cal. | 0.748 [0.669, 0.811] | 0.642 [0.565, 0.704] | 0.506 [0.431, 0.572] | 0.670 [0.594, 0.731] | 0.510 [0.435, 0.575] |
| C3M e123, d_F cal. @ regione GNM | 0.679 [0.598, 0.746] | 0.672 [0.593, 0.740] | 0.663 [0.583, 0.732] | 0.679 [0.601, 0.747] | 0.667 [0.588, 0.736] |
| C3M e123, d_F cal. @ regione FLAME | 0.567 [0.485, 0.650] | 0.575 [0.493, 0.657] | 0.590 [0.507, 0.673] | 0.580 [0.498, 0.663] | 0.594 [0.511, 0.677] |
| C3M e205, d_F cal. | 0.739 [0.656, 0.805] | 0.630 [0.547, 0.693] | 0.472 [0.395, 0.537] | 0.653 [0.575, 0.716] | 0.476 [0.399, 0.540] |
| C3M e205, d_F cal. @ regione GNM | 0.693 [0.619, 0.756] | 0.689 [0.615, 0.751] | 0.685 [0.610, 0.747] | 0.700 [0.626, 0.761] | 0.692 [0.618, 0.754] |
| C3M e205, d_F cal. @ regione FLAME | 0.601 [0.522, 0.679] | 0.608 [0.529, 0.685] | 0.621 [0.541, 0.697] | 0.613 [0.535, 0.690] | 0.625 [0.546, 0.701] |
| factorized s1234, d_P | 0.426 [0.331, 0.518] | 0.383 [0.290, 0.471] | 0.376 [0.281, 0.463] | 0.410 [0.314, 0.498] | 0.377 [0.282, 0.464] |
| factorized s1234, d_P @ regione GNM | 0.528 [0.450, 0.607] | 0.515 [0.435, 0.595] | 0.489 [0.404, 0.569] | 0.517 [0.436, 0.597] | 0.490 [0.405, 0.570] |
| factorized s1234, d_P @ regione FLAME | 0.512 [0.425, 0.593] | 0.511 [0.425, 0.591] | 0.511 [0.426, 0.591] | 0.512 [0.425, 0.592] | 0.511 [0.427, 0.592] |
| factorized s2345, d_P | 0.418 [0.320, 0.521] | 0.370 [0.274, 0.463] | 0.347 [0.253, 0.440] | 0.395 [0.296, 0.496] | 0.348 [0.254, 0.441] |
| factorized s2345, d_P @ regione GNM | 0.521 [0.439, 0.602] | 0.513 [0.429, 0.594] | 0.497 [0.415, 0.577] | 0.514 [0.430, 0.595] | 0.497 [0.416, 0.578] |
| factorized s2345, d_P @ regione FLAME | 0.506 [0.410, 0.599] | 0.506 [0.411, 0.597] | 0.506 [0.412, 0.595] | 0.507 [0.412, 0.598] | 0.507 [0.413, 0.595] |
| C3M e123, d_P | 0.401 [0.305, 0.497] | 0.313 [0.228, 0.393] | 0.317 [0.227, 0.405] | 0.377 [0.281, 0.469] | 0.321 [0.230, 0.412] |
| C3M e123, d_P @ regione GNM | 0.526 [0.438, 0.609] | 0.507 [0.421, 0.590] | 0.471 [0.383, 0.556] | 0.513 [0.427, 0.596] | 0.475 [0.386, 0.560] |
| C3M e123, d_P @ regione FLAME | 0.500 [0.414, 0.584] | 0.503 [0.417, 0.586] | 0.509 [0.423, 0.592] | 0.507 [0.424, 0.591] | 0.512 [0.426, 0.595] |
| C3M e205, d_P | 0.405 [0.308, 0.500] | 0.326 [0.240, 0.408] | 0.329 [0.238, 0.420] | 0.384 [0.291, 0.474] | 0.333 [0.241, 0.426] |
| C3M e205, d_P @ regione GNM | 0.529 [0.442, 0.607] | 0.515 [0.429, 0.595] | 0.486 [0.395, 0.574] | 0.525 [0.440, 0.607] | 0.493 [0.402, 0.581] |
| C3M e205, d_P @ regione FLAME | 0.503 [0.422, 0.591] | 0.506 [0.425, 0.591] | 0.513 [0.429, 0.597] | 0.512 [0.431, 0.598] | 0.517 [0.434, 0.602] |
| GNM (visto) vB, coefficienti | 0.550 [0.465, 0.631] | 0.540 [0.457, 0.619] | 0.524 [0.441, 0.601] | 0.542 [0.459, 0.621] | 0.524 [0.441, 0.602] |
| GNM (visto) vB, mesh d'identita' FR | 0.664 [0.573, 0.740] | 0.651 [0.560, 0.727] | 0.625 [0.536, 0.705] | 0.651 [0.561, 0.728] | 0.625 [0.536, 0.705] |
| GNM (visto) vB, mesh d'identita' SR | 0.411 [0.310, 0.508] | 0.395 [0.295, 0.489] | 0.380 [0.282, 0.475] | 0.401 [0.300, 0.496] | 0.381 [0.283, 0.475] |
| FLAME 2023 Open vB, coefficienti | 0.585 [0.498, 0.659] | 0.570 [0.485, 0.643] | 0.542 [0.457, 0.617] | 0.572 [0.488, 0.645] | 0.543 [0.457, 0.618] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.686 [0.598, 0.762] | 0.670 [0.582, 0.748] | 0.639 [0.551, 0.719] | 0.670 [0.582, 0.748] | 0.639 [0.551, 0.719] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.458 [0.369, 0.548] | 0.448 [0.360, 0.536] | 0.433 [0.343, 0.518] | 0.450 [0.361, 0.538] | 0.433 [0.343, 0.518] |
| ICP + Chamfer in mm | 0.643 [0.570, 0.705] | 0.556 [0.488, 0.617] | 0.416 [0.358, 0.474] | 0.587 [0.520, 0.646] | 0.435 [0.376, 0.494] |
| Chamfer pura in mm | 0.646 [0.548, 0.726] | 0.645 [0.547, 0.725] | 0.643 [0.546, 0.724] | 0.646 [0.549, 0.726] | 0.645 [0.548, 0.725] |
| NICP su template in mm | 0.614 [0.510, 0.701] | 0.514 [0.410, 0.601] | 0.337 [0.239, 0.427] | 0.522 [0.424, 0.606] | 0.337 [0.240, 0.427] |

GT SR:

| metodo | senza crop | all_cross | righe col crop | media 15 coppie | media crop 5 coppie |
| --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal. | 0.319 [0.228, 0.410] | 0.304 [0.218, 0.387] | 0.297 [0.218, 0.376] | 0.312 [0.226, 0.397] | 0.298 [0.218, 0.377] |
| factorized s1234, d_F cal. @ regione GNM | 0.434 [0.339, 0.519] | 0.429 [0.335, 0.516] | 0.422 [0.330, 0.509] | 0.430 [0.336, 0.516] | 0.422 [0.330, 0.510] |
| factorized s1234, d_F cal. @ regione FLAME | 0.371 [0.262, 0.470] | 0.372 [0.262, 0.470] | 0.374 [0.266, 0.473] | 0.372 [0.263, 0.471] | 0.375 [0.266, 0.473] |
| factorized s2345, d_F cal. | 0.313 [0.219, 0.408] | 0.286 [0.202, 0.372] | 0.255 [0.181, 0.328] | 0.294 [0.208, 0.382] | 0.256 [0.182, 0.329] |
| factorized s2345, d_F cal. @ regione GNM | 0.452 [0.356, 0.538] | 0.444 [0.351, 0.529] | 0.430 [0.343, 0.514] | 0.445 [0.352, 0.530] | 0.431 [0.343, 0.514] |
| factorized s2345, d_F cal. @ regione FLAME | 0.395 [0.290, 0.496] | 0.398 [0.293, 0.498] | 0.403 [0.299, 0.503] | 0.398 [0.293, 0.499] | 0.403 [0.299, 0.503] |
| ctrlfr s1234 | 0.339 [0.245, 0.438] | 0.244 [0.180, 0.311] | 0.164 [0.122, 0.207] | 0.282 [0.209, 0.360] | 0.166 [0.124, 0.210] |
| ctrlfr s1234 @ regione GNM | 0.484 [0.389, 0.571] | 0.471 [0.381, 0.555] | 0.453 [0.366, 0.533] | 0.476 [0.385, 0.561] | 0.456 [0.369, 0.535] |
| ctrlfr s1234 @ regione FLAME | 0.431 [0.318, 0.534] | 0.434 [0.320, 0.537] | 0.440 [0.328, 0.543] | 0.437 [0.324, 0.540] | 0.442 [0.330, 0.546] |
| ctrlfr s2345 | 0.348 [0.257, 0.447] | 0.259 [0.190, 0.328] | 0.188 [0.144, 0.234] | 0.296 [0.221, 0.376] | 0.190 [0.146, 0.237] |
| ctrlfr s2345 @ regione GNM | 0.477 [0.382, 0.566] | 0.458 [0.368, 0.542] | 0.430 [0.341, 0.516] | 0.463 [0.372, 0.547] | 0.432 [0.343, 0.518] |
| ctrlfr s2345 @ regione FLAME | 0.414 [0.311, 0.517] | 0.415 [0.311, 0.519] | 0.417 [0.313, 0.522] | 0.417 [0.313, 0.521] | 0.418 [0.314, 0.523] |
| C3M e123, d_F cal. | 0.292 [0.202, 0.384] | 0.252 [0.173, 0.332] | 0.220 [0.151, 0.288] | 0.270 [0.189, 0.354] | 0.223 [0.153, 0.291] |
| C3M e123, d_F cal. @ regione GNM | 0.396 [0.294, 0.491] | 0.393 [0.294, 0.487] | 0.390 [0.291, 0.480] | 0.397 [0.298, 0.493] | 0.393 [0.293, 0.483] |
| C3M e123, d_F cal. @ regione FLAME | 0.354 [0.247, 0.459] | 0.358 [0.250, 0.464] | 0.366 [0.257, 0.473] | 0.363 [0.254, 0.469] | 0.369 [0.260, 0.477] |
| C3M e205, d_F cal. | 0.287 [0.195, 0.383] | 0.250 [0.167, 0.332] | 0.220 [0.151, 0.285] | 0.266 [0.181, 0.351] | 0.222 [0.152, 0.287] |
| C3M e205, d_F cal. @ regione GNM | 0.398 [0.301, 0.488] | 0.398 [0.304, 0.486] | 0.400 [0.306, 0.486] | 0.405 [0.309, 0.493] | 0.404 [0.310, 0.491] |
| C3M e205, d_F cal. @ regione FLAME | 0.364 [0.258, 0.466] | 0.366 [0.259, 0.468] | 0.369 [0.259, 0.473] | 0.370 [0.263, 0.474] | 0.372 [0.262, 0.477] |
| factorized s1234, d_P | 0.622 [0.549, 0.687] | 0.557 [0.483, 0.621] | 0.550 [0.470, 0.618] | 0.599 [0.527, 0.665] | 0.551 [0.472, 0.619] |
| factorized s1234, d_P @ regione GNM | 0.584 [0.500, 0.656] | 0.587 [0.503, 0.658] | 0.594 [0.514, 0.663] | 0.588 [0.505, 0.659] | 0.595 [0.514, 0.664] |
| factorized s1234, d_P @ regione FLAME | 0.481 [0.372, 0.578] | 0.482 [0.373, 0.579] | 0.485 [0.377, 0.582] | 0.483 [0.374, 0.580] | 0.485 [0.377, 0.582] |
| factorized s2345, d_P | 0.613 [0.528, 0.688] | 0.540 [0.457, 0.615] | 0.514 [0.426, 0.595] | 0.581 [0.495, 0.658] | 0.516 [0.428, 0.596] |
| factorized s2345, d_P @ regione GNM | 0.569 [0.479, 0.647] | 0.576 [0.486, 0.650] | 0.591 [0.507, 0.659] | 0.577 [0.488, 0.651] | 0.591 [0.508, 0.660] |
| factorized s2345, d_P @ regione FLAME | 0.471 [0.360, 0.568] | 0.474 [0.364, 0.571] | 0.480 [0.371, 0.577] | 0.475 [0.365, 0.572] | 0.481 [0.372, 0.577] |
| C3M e123, d_P | 0.595 [0.506, 0.670] | 0.461 [0.377, 0.532] | 0.462 [0.377, 0.540] | 0.557 [0.473, 0.631] | 0.469 [0.383, 0.548] |
| C3M e123, d_P @ regione GNM | 0.493 [0.393, 0.578] | 0.500 [0.401, 0.583] | 0.514 [0.416, 0.598] | 0.505 [0.406, 0.588] | 0.518 [0.418, 0.601] |
| C3M e123, d_P @ regione FLAME | 0.443 [0.338, 0.548] | 0.449 [0.341, 0.554] | 0.460 [0.349, 0.566] | 0.453 [0.345, 0.559] | 0.463 [0.352, 0.569] |
| C3M e205, d_P | 0.591 [0.499, 0.667] | 0.473 [0.389, 0.547] | 0.481 [0.393, 0.560] | 0.561 [0.473, 0.637] | 0.488 [0.398, 0.568] |
| C3M e205, d_P @ regione GNM | 0.515 [0.421, 0.596] | 0.525 [0.434, 0.605] | 0.544 [0.454, 0.622] | 0.535 [0.444, 0.615] | 0.551 [0.462, 0.629] |
| C3M e205, d_P @ regione FLAME | 0.456 [0.349, 0.557] | 0.461 [0.353, 0.562] | 0.471 [0.360, 0.572] | 0.467 [0.359, 0.568] | 0.475 [0.364, 0.577] |
| GNM (visto) vB, coefficienti | 0.675 [0.607, 0.731] | 0.665 [0.597, 0.720] | 0.651 [0.580, 0.706] | 0.668 [0.601, 0.723] | 0.651 [0.581, 0.707] |
| GNM (visto) vB, mesh d'identita' FR | 0.217 [0.126, 0.309] | 0.213 [0.123, 0.303] | 0.205 [0.118, 0.294] | 0.213 [0.124, 0.304] | 0.206 [0.118, 0.294] |
| GNM (visto) vB, mesh d'identita' SR | 0.607 [0.524, 0.679] | 0.587 [0.504, 0.660] | 0.570 [0.482, 0.648] | 0.595 [0.513, 0.669] | 0.570 [0.482, 0.648] |
| FLAME 2023 Open vB, coefficienti | 0.586 [0.512, 0.650] | 0.573 [0.500, 0.636] | 0.551 [0.481, 0.614] | 0.576 [0.503, 0.639] | 0.552 [0.481, 0.615] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.231 [0.143, 0.326] | 0.223 [0.136, 0.316] | 0.209 [0.126, 0.299] | 0.223 [0.136, 0.317] | 0.209 [0.126, 0.299] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.598 [0.513, 0.673] | 0.586 [0.498, 0.661] | 0.567 [0.478, 0.643] | 0.588 [0.500, 0.663] | 0.567 [0.478, 0.643] |
| ICP + Chamfer in mm | 0.366 [0.278, 0.456] | 0.327 [0.249, 0.404] | 0.266 [0.205, 0.328] | 0.344 [0.262, 0.425] | 0.279 [0.214, 0.344] |
| Chamfer pura in mm | 0.066 [-0.016, 0.158] | 0.065 [-0.017, 0.158] | 0.063 [-0.019, 0.153] | 0.066 [-0.017, 0.159] | 0.065 [-0.018, 0.156] |
| NICP su template in mm | 0.083 [-0.007, 0.178] | 0.066 [-0.013, 0.148] | 0.047 [-0.012, 0.111] | 0.071 [-0.009, 0.153] | 0.047 [-0.012, 0.111] |

Delta appaiati, righe col crop (braccio sulla regione di m - concorrente; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.156 [+0.068, +0.241], P 0.000 | +0.122 [+0.036, +0.206], P 0.003 | +0.078 [-0.006, +0.168], P 0.036 | +0.099 [+0.011, +0.188], P 0.017 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.055 [-0.008, +0.118], P 0.035 | +0.021 [-0.036, +0.075], P 0.242 | -0.022 [-0.096, +0.049], P 0.732 | -0.002 [-0.069, +0.064], P 0.527 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | +0.299 [+0.205, +0.392], P 0.000 | +0.265 [+0.181, +0.348], P 0.000 | +0.222 [+0.132, +0.306], P 0.000 | +0.242 [+0.147, +0.334], P 0.000 |
| ICP + Chamfer in mm (@ regione GNM) | +0.263 [+0.212, +0.316], P 0.000 | +0.229 [+0.177, +0.281], P 0.000 | +0.186 [+0.123, +0.251], P 0.000 | +0.206 [+0.146, +0.270], P 0.000 |
| Chamfer pura in mm (@ regione GNM) | +0.037 [-0.035, +0.115], P 0.175 | +0.002 [-0.074, +0.088], P 0.488 | -0.041 [-0.129, +0.049], P 0.814 | -0.021 [-0.108, +0.069], P 0.672 |
| NICP su template in mm (@ regione GNM) | +0.343 [+0.272, +0.417], P 0.000 | +0.308 [+0.239, +0.382], P 0.000 | +0.265 [+0.183, +0.349], P 0.000 | +0.285 [+0.209, +0.361], P 0.000 |
| braccio intero (@ regione GNM) | +0.017 [-0.031, +0.060], P 0.216 | +0.093 [+0.046, +0.140], P 0.000 | +0.367 [+0.309, +0.422], P 0.000 | +0.344 [+0.291, +0.394], P 0.000 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.074 [-0.011, +0.170], P 0.051 | +0.039 [-0.054, +0.136], P 0.218 | +0.064 [-0.018, +0.149], P 0.085 | +0.075 [-0.017, +0.169], P 0.059 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | -0.022 [-0.104, +0.051], P 0.706 | -0.057 [-0.144, +0.019], P 0.921 | -0.032 [-0.118, +0.044], P 0.789 | -0.022 [-0.102, +0.053], P 0.722 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | +0.183 [+0.108, +0.259], P 0.000 | +0.148 [+0.081, +0.221], P 0.000 | +0.173 [+0.105, +0.244], P 0.000 | +0.184 [+0.110, +0.260], P 0.000 |
| ICP + Chamfer in mm (@ regione FLAME) | +0.200 [+0.132, +0.265], P 0.000 | +0.165 [+0.095, +0.237], P 0.000 | +0.190 [+0.125, +0.255], P 0.000 | +0.200 [+0.133, +0.268], P 0.000 |
| Chamfer pura in mm (@ regione FLAME) | -0.027 [-0.121, +0.071], P 0.701 | -0.062 [-0.164, +0.044], P 0.868 | -0.037 [-0.136, +0.059], P 0.782 | -0.026 [-0.124, +0.075], P 0.702 |
| NICP su template in mm (@ regione FLAME) | +0.280 [+0.185, +0.371], P 0.000 | +0.244 [+0.149, +0.338], P 0.000 | +0.269 [+0.173, +0.363], P 0.000 | +0.280 [+0.186, +0.373], P 0.000 |
| braccio intero (@ regione FLAME) | -0.046 [-0.120, +0.022], P 0.898 | +0.029 [-0.044, +0.098], P 0.214 | +0.371 [+0.308, +0.430], P 0.000 | +0.338 [+0.275, +0.401], P 0.000 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | -0.056 [-0.143, +0.029], P 0.904 | -0.060 [-0.154, +0.028], P 0.910 | -0.198 [-0.285, -0.108], P 1.000 | -0.220 [-0.306, -0.136], P 1.000 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.389 [+0.307, +0.467], P 0.000 | +0.385 [+0.309, +0.461], P 0.000 | +0.248 [+0.176, +0.312], P 0.000 | +0.225 [+0.162, +0.286], P 0.000 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | +0.024 [-0.035, +0.087], P 0.231 | +0.021 [-0.035, +0.072], P 0.248 | -0.117 [-0.186, -0.051], P 1.000 | -0.140 [-0.217, -0.071], P 1.000 |
| ICP + Chamfer in mm (@ regione GNM) | +0.328 [+0.262, +0.398], P 0.000 | +0.325 [+0.253, +0.398], P 0.000 | +0.187 [+0.122, +0.257], P 0.000 | +0.164 [+0.100, +0.230], P 0.000 |
| Chamfer pura in mm (@ regione GNM) | +0.531 [+0.424, +0.627], P 0.000 | +0.528 [+0.422, +0.626], P 0.000 | +0.390 [+0.290, +0.483], P 0.000 | +0.367 [+0.268, +0.455], P 0.000 |
| NICP su template in mm (@ regione GNM) | +0.548 [+0.464, +0.622], P 0.000 | +0.544 [+0.461, +0.618], P 0.000 | +0.406 [+0.331, +0.481], P 0.000 | +0.384 [+0.306, +0.459], P 0.000 |
| braccio intero (@ regione GNM) | +0.045 [+0.001, +0.089], P 0.020 | +0.077 [+0.029, +0.124], P 0.001 | +0.290 [+0.230, +0.354], P 0.000 | +0.242 [+0.186, +0.302], P 0.000 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | -0.066 [-0.181, +0.047], P 0.880 | -0.071 [-0.191, +0.047], P 0.879 | -0.111 [-0.224, -0.006], P 0.981 | -0.134 [-0.249, -0.022], P 0.990 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.276 [+0.179, +0.368], P 0.000 | +0.271 [+0.175, +0.365], P 0.000 | +0.232 [+0.151, +0.313], P 0.000 | +0.208 [+0.128, +0.286], P 0.000 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.082 [-0.168, -0.012], P 0.990 | -0.087 [-0.171, -0.011], P 0.990 | -0.127 [-0.206, -0.052], P 1.000 | -0.150 [-0.232, -0.073], P 1.000 |
| ICP + Chamfer in mm (@ regione FLAME) | +0.219 [+0.131, +0.305], P 0.000 | +0.214 [+0.122, +0.307], P 0.000 | +0.174 [+0.094, +0.256], P 0.000 | +0.151 [+0.075, +0.233], P 0.000 |
| Chamfer pura in mm (@ regione FLAME) | +0.422 [+0.308, +0.530], P 0.000 | +0.417 [+0.297, +0.533], P 0.000 | +0.377 [+0.271, +0.485], P 0.000 | +0.354 [+0.251, +0.459], P 0.000 |
| NICP su template in mm (@ regione FLAME) | +0.438 [+0.332, +0.538], P 0.000 | +0.433 [+0.329, +0.533], P 0.000 | +0.394 [+0.298, +0.493], P 0.000 | +0.370 [+0.275, +0.468], P 0.000 |
| braccio intero (@ regione FLAME) | -0.065 [-0.160, +0.025], P 0.906 | -0.034 [-0.128, +0.056], P 0.755 | +0.277 [+0.185, +0.366], P 0.000 | +0.228 [+0.143, +0.311], P 0.000 |

Delta appaiati, media crop 5 coppie (braccio sulla regione di m - concorrente; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.156 [+0.068, +0.241], P 0.000 | +0.121 [+0.036, +0.206], P 0.003 | +0.081 [-0.002, +0.171], P 0.028 | +0.101 [+0.013, +0.189], P 0.016 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.055 [-0.007, +0.119], P 0.035 | +0.021 [-0.036, +0.076], P 0.240 | -0.019 [-0.092, +0.053], P 0.703 | +0.000 [-0.066, +0.067], P 0.493 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | +0.300 [+0.205, +0.392], P 0.000 | +0.265 [+0.181, +0.349], P 0.000 | +0.225 [+0.135, +0.309], P 0.000 | +0.244 [+0.150, +0.336], P 0.000 |
| ICP + Chamfer in mm (@ regione GNM) | +0.245 [+0.193, +0.298], P 0.000 | +0.211 [+0.159, +0.263], P 0.000 | +0.171 [+0.108, +0.238], P 0.000 | +0.190 [+0.130, +0.254], P 0.000 |
| Chamfer pura in mm (@ regione GNM) | +0.035 [-0.037, +0.113], P 0.183 | +0.000 [-0.077, +0.086], P 0.501 | -0.040 [-0.127, +0.050], P 0.803 | -0.020 [-0.108, +0.070], P 0.668 |
| NICP su template in mm (@ regione GNM) | +0.343 [+0.272, +0.417], P 0.000 | +0.309 [+0.240, +0.382], P 0.000 | +0.268 [+0.187, +0.351], P 0.000 | +0.288 [+0.211, +0.363], P 0.000 |
| braccio intero (@ regione GNM) | +0.017 [-0.032, +0.059], P 0.225 | +0.092 [+0.045, +0.138], P 0.000 | +0.367 [+0.310, +0.422], P 0.000 | +0.344 [+0.291, +0.394], P 0.000 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.074 [-0.011, +0.169], P 0.052 | +0.038 [-0.055, +0.135], P 0.223 | +0.066 [-0.016, +0.151], P 0.076 | +0.075 [-0.016, +0.169], P 0.058 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | -0.022 [-0.104, +0.051], P 0.704 | -0.057 [-0.144, +0.019], P 0.921 | -0.030 [-0.116, +0.046], P 0.771 | -0.021 [-0.101, +0.054], P 0.714 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | +0.184 [+0.108, +0.260], P 0.000 | +0.148 [+0.082, +0.221], P 0.000 | +0.176 [+0.107, +0.247], P 0.000 | +0.185 [+0.111, +0.261], P 0.000 |
| ICP + Chamfer in mm (@ regione FLAME) | +0.182 [+0.115, +0.246], P 0.000 | +0.147 [+0.077, +0.221], P 0.000 | +0.174 [+0.108, +0.239], P 0.000 | +0.183 [+0.114, +0.251], P 0.000 |
| Chamfer pura in mm (@ regione FLAME) | -0.028 [-0.122, +0.069], P 0.716 | -0.064 [-0.166, +0.042], P 0.874 | -0.036 [-0.136, +0.060], P 0.779 | -0.027 [-0.125, +0.074], P 0.707 |
| NICP su template in mm (@ regione FLAME) | +0.280 [+0.185, +0.371], P 0.000 | +0.244 [+0.149, +0.338], P 0.000 | +0.272 [+0.175, +0.365], P 0.000 | +0.281 [+0.187, +0.375], P 0.000 |
| braccio intero (@ regione FLAME) | -0.047 [-0.121, +0.021], P 0.904 | +0.028 [-0.045, +0.096], P 0.223 | +0.371 [+0.307, +0.430], P 0.000 | +0.337 [+0.273, +0.399], P 0.000 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | -0.056 [-0.143, +0.029], P 0.904 | -0.060 [-0.155, +0.028], P 0.910 | -0.195 [-0.283, -0.106], P 1.000 | -0.219 [-0.304, -0.135], P 1.000 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.390 [+0.308, +0.467], P 0.000 | +0.386 [+0.309, +0.461], P 0.000 | +0.251 [+0.179, +0.315], P 0.000 | +0.227 [+0.164, +0.288], P 0.000 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | +0.025 [-0.035, +0.088], P 0.225 | +0.021 [-0.035, +0.072], P 0.244 | -0.114 [-0.184, -0.048], P 1.000 | -0.138 [-0.216, -0.069], P 1.000 |
| ICP + Chamfer in mm (@ regione GNM) | +0.316 [+0.248, +0.390], P 0.000 | +0.312 [+0.239, +0.386], P 0.000 | +0.177 [+0.111, +0.247], P 0.000 | +0.153 [+0.087, +0.220], P 0.000 |
| Chamfer pura in mm (@ regione GNM) | +0.531 [+0.423, +0.626], P 0.000 | +0.527 [+0.421, +0.625], P 0.000 | +0.392 [+0.292, +0.485], P 0.000 | +0.368 [+0.269, +0.457], P 0.000 |
| NICP su template in mm (@ regione GNM) | +0.548 [+0.465, +0.623], P 0.000 | +0.545 [+0.461, +0.618], P 0.000 | +0.409 [+0.334, +0.484], P 0.000 | +0.385 [+0.307, +0.461], P 0.000 |
| braccio intero (@ regione GNM) | +0.044 [-0.000, +0.089], P 0.026 | +0.076 [+0.028, +0.122], P 0.001 | +0.290 [+0.230, +0.354], P 0.000 | +0.242 [+0.186, +0.302], P 0.000 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | -0.067 [-0.181, +0.047], P 0.883 | -0.071 [-0.191, +0.047], P 0.880 | -0.110 [-0.222, -0.005], P 0.981 | -0.134 [-0.248, -0.022], P 0.990 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.277 [+0.180, +0.368], P 0.000 | +0.272 [+0.176, +0.366], P 0.000 | +0.234 [+0.153, +0.315], P 0.000 | +0.209 [+0.129, +0.288], P 0.000 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.081 [-0.167, -0.011], P 0.989 | -0.086 [-0.171, -0.011], P 0.987 | -0.124 [-0.204, -0.050], P 1.000 | -0.149 [-0.231, -0.072], P 1.000 |
| ICP + Chamfer in mm (@ regione FLAME) | +0.206 [+0.118, +0.292], P 0.000 | +0.201 [+0.108, +0.293], P 0.000 | +0.163 [+0.082, +0.245], P 0.000 | +0.138 [+0.061, +0.221], P 0.000 |
| Chamfer pura in mm (@ regione FLAME) | +0.421 [+0.307, +0.530], P 0.000 | +0.416 [+0.296, +0.532], P 0.000 | +0.378 [+0.271, +0.487], P 0.000 | +0.353 [+0.251, +0.459], P 0.000 |
| NICP su template in mm (@ regione FLAME) | +0.439 [+0.333, +0.539], P 0.000 | +0.434 [+0.330, +0.534], P 0.000 | +0.396 [+0.299, +0.495], P 0.000 | +0.371 [+0.276, +0.470], P 0.000 |
| braccio intero (@ regione FLAME) | -0.066 [-0.161, +0.024], P 0.913 | -0.035 [-0.130, +0.055], P 0.764 | +0.276 [+0.185, +0.366], P 0.000 | +0.227 [+0.142, +0.310], P 0.000 |

Delta appaiati, senza crop (braccio sulla regione di m - concorrente; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.138 [+0.054, +0.225], P 0.000 | +0.101 [+0.019, +0.188], P 0.009 | +0.105 [+0.021, +0.188], P 0.011 | +0.142 [+0.059, +0.229], P 0.000 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.023 [-0.042, +0.085], P 0.234 | -0.014 [-0.074, +0.042], P 0.672 | -0.009 [-0.084, +0.059], P 0.611 | +0.028 [-0.039, +0.091], P 0.200 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | +0.277 [+0.181, +0.368], P 0.000 | +0.240 [+0.154, +0.322], P 0.000 | +0.244 [+0.155, +0.332], P 0.000 | +0.281 [+0.191, +0.370], P 0.000 |
| ICP + Chamfer in mm (@ regione GNM) | +0.045 [-0.004, +0.096], P 0.036 | +0.007 [-0.044, +0.061], P 0.385 | +0.012 [-0.046, +0.074], P 0.336 | +0.049 [-0.004, +0.105], P 0.035 |
| Chamfer pura in mm (@ regione GNM) | +0.042 [-0.030, +0.117], P 0.133 | +0.005 [-0.075, +0.089], P 0.473 | +0.009 [-0.072, +0.096], P 0.428 | +0.046 [-0.033, +0.131], P 0.125 |
| NICP su template in mm (@ regione GNM) | +0.074 [-0.004, +0.150], P 0.030 | +0.037 [-0.044, +0.114], P 0.188 | +0.041 [-0.046, +0.128], P 0.160 | +0.078 [-0.003, +0.161], P 0.031 |
| braccio intero (@ regione GNM) | -0.061 [-0.114, -0.011], P 0.996 | -0.080 [-0.133, -0.026], P 0.999 | -0.102 [-0.154, -0.054], P 1.000 | -0.053 [-0.104, -0.004], P 0.988 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.024 [-0.063, +0.120], P 0.340 | -0.012 [-0.103, +0.083], P 0.599 | +0.012 [-0.071, +0.097], P 0.413 | +0.027 [-0.061, +0.120], P 0.291 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | -0.077 [-0.157, -0.002], P 0.976 | -0.113 [-0.198, -0.035], P 0.998 | -0.088 [-0.173, -0.012], P 0.990 | -0.074 [-0.154, +0.001], P 0.973 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | +0.150 [+0.073, +0.228], P 0.000 | +0.114 [+0.045, +0.189], P 0.000 | +0.139 [+0.067, +0.209], P 0.000 | +0.154 [+0.077, +0.233], P 0.000 |
| ICP + Chamfer in mm (@ regione FLAME) | -0.034 [-0.098, +0.030], P 0.860 | -0.070 [-0.141, +0.003], P 0.970 | -0.046 [-0.121, +0.027], P 0.904 | -0.031 [-0.102, +0.045], P 0.804 |
| Chamfer pura in mm (@ regione FLAME) | -0.037 [-0.130, +0.059], P 0.775 | -0.073 [-0.176, +0.030], P 0.909 | -0.048 [-0.147, +0.049], P 0.839 | -0.034 [-0.131, +0.070], P 0.758 |
| NICP su template in mm (@ regione FLAME) | -0.005 [-0.104, +0.090], P 0.563 | -0.041 [-0.143, +0.056], P 0.790 | -0.017 [-0.118, +0.082], P 0.661 | -0.002 [-0.102, +0.094], P 0.530 |
| braccio intero (@ regione FLAME) | -0.140 [-0.215, -0.070], P 1.000 | -0.158 [-0.238, -0.081], P 1.000 | -0.160 [-0.233, -0.096], P 1.000 | -0.134 [-0.206, -0.067], P 1.000 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | -0.091 [-0.175, -0.007], P 0.983 | -0.106 [-0.205, -0.016], P 0.987 | -0.191 [-0.279, -0.100], P 1.000 | -0.198 [-0.285, -0.108], P 1.000 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.366 [+0.280, +0.447], P 0.000 | +0.351 [+0.270, +0.430], P 0.000 | +0.267 [+0.197, +0.340], P 0.000 | +0.260 [+0.187, +0.329], P 0.000 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.023 [-0.078, +0.034], P 0.795 | -0.038 [-0.093, +0.017], P 0.924 | -0.123 [-0.195, -0.055], P 1.000 | -0.130 [-0.206, -0.059], P 1.000 |
| ICP + Chamfer in mm (@ regione GNM) | +0.218 [+0.143, +0.299], P 0.000 | +0.203 [+0.124, +0.282], P 0.000 | +0.118 [+0.050, +0.193], P 0.000 | +0.111 [+0.049, +0.179], P 0.000 |
| Chamfer pura in mm (@ regione GNM) | +0.517 [+0.413, +0.615], P 0.000 | +0.502 [+0.393, +0.603], P 0.000 | +0.418 [+0.317, +0.518], P 0.000 | +0.410 [+0.311, +0.505], P 0.000 |
| NICP su template in mm (@ regione GNM) | +0.501 [+0.401, +0.600], P 0.000 | +0.485 [+0.380, +0.585], P 0.000 | +0.401 [+0.307, +0.497], P 0.000 | +0.394 [+0.300, +0.487], P 0.000 |
| braccio intero (@ regione GNM) | -0.038 [-0.084, +0.008], P 0.953 | -0.044 [-0.093, +0.000], P 0.974 | +0.145 [+0.090, +0.202], P 0.000 | +0.129 [+0.080, +0.183], P 0.000 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | -0.105 [-0.221, +0.005], P 0.968 | -0.115 [-0.231, -0.000], P 0.975 | -0.155 [-0.265, -0.051], P 0.996 | -0.172 [-0.281, -0.062], P 0.999 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.250 [+0.151, +0.341], P 0.000 | +0.240 [+0.144, +0.334], P 0.000 | +0.200 [+0.121, +0.279], P 0.000 | +0.184 [+0.106, +0.259], P 0.000 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.117 [-0.203, -0.044], P 1.000 | -0.127 [-0.212, -0.053], P 1.000 | -0.166 [-0.251, -0.094], P 1.000 | -0.183 [-0.265, -0.106], P 1.000 |
| ICP + Chamfer in mm (@ regione FLAME) | +0.115 [+0.027, +0.204], P 0.006 | +0.105 [+0.017, +0.196], P 0.007 | +0.065 [-0.013, +0.137], P 0.047 | +0.049 [-0.028, +0.130], P 0.112 |
| Chamfer pura in mm (@ regione FLAME) | +0.414 [+0.302, +0.522], P 0.000 | +0.405 [+0.288, +0.517], P 0.000 | +0.365 [+0.260, +0.473], P 0.000 | +0.348 [+0.244, +0.453], P 0.000 |
| NICP su template in mm (@ regione FLAME) | +0.398 [+0.283, +0.507], P 0.000 | +0.388 [+0.269, +0.497], P 0.000 | +0.348 [+0.245, +0.450], P 0.000 | +0.331 [+0.230, +0.429], P 0.000 |
| braccio intero (@ regione FLAME) | -0.141 [-0.231, -0.061], P 1.000 | -0.142 [-0.227, -0.059], P 1.000 | +0.092 [+0.024, +0.164], P 0.006 | +0.066 [-0.002, +0.135], P 0.028 |


## facescape: Spearman con la GT per gruppo (rho, IC 95%; 148500 righe all_cross, seme 621096)

GT FR:

| metodo | senza crop | all_cross | righe col crop | media 15 coppie | media crop 5 coppie |
| --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal. | 0.659 [0.589, 0.724] | 0.370 [0.320, 0.418] | 0.351 [0.272, 0.428] | 0.569 [0.500, 0.636] | 0.375 [0.294, 0.456] |
| factorized s1234, d_F cal. @ regione GNM | 0.684 [0.613, 0.750] | 0.685 [0.612, 0.751] | 0.687 [0.612, 0.755] | 0.692 [0.618, 0.758] | 0.691 [0.616, 0.758] |
| factorized s1234, d_F cal. @ regione FLAME | 0.640 [0.570, 0.704] | 0.648 [0.576, 0.710] | 0.664 [0.593, 0.729] | 0.653 [0.583, 0.716] | 0.667 [0.596, 0.732] |
| factorized s2345, d_F cal. | 0.667 [0.599, 0.729] | 0.351 [0.308, 0.393] | 0.278 [0.206, 0.348] | 0.548 [0.485, 0.604] | 0.300 [0.225, 0.375] |
| factorized s2345, d_F cal. @ regione GNM | 0.711 [0.639, 0.774] | 0.708 [0.635, 0.773] | 0.703 [0.625, 0.771] | 0.711 [0.638, 0.776] | 0.704 [0.625, 0.772] |
| factorized s2345, d_F cal. @ regione FLAME | 0.683 [0.615, 0.749] | 0.687 [0.619, 0.753] | 0.697 [0.628, 0.760] | 0.689 [0.621, 0.755] | 0.698 [0.630, 0.762] |
| ctrlfr s1234 | 0.663 [0.596, 0.718] | 0.334 [0.296, 0.369] | 0.264 [0.196, 0.328] | 0.561 [0.502, 0.613] | 0.274 [0.202, 0.343] |
| ctrlfr s1234 @ regione GNM | 0.725 [0.668, 0.774] | 0.721 [0.663, 0.770] | 0.714 [0.649, 0.768] | 0.737 [0.681, 0.785] | 0.726 [0.662, 0.779] |
| ctrlfr s1234 @ regione FLAME | 0.639 [0.581, 0.693] | 0.652 [0.596, 0.704] | 0.684 [0.629, 0.732] | 0.682 [0.630, 0.733] | 0.708 [0.654, 0.756] |
| ctrlfr s2345 | 0.652 [0.585, 0.710] | 0.323 [0.285, 0.359] | 0.234 [0.166, 0.298] | 0.547 [0.485, 0.602] | 0.247 [0.175, 0.317] |
| ctrlfr s2345 @ regione GNM | 0.677 [0.612, 0.736] | 0.653 [0.587, 0.713] | 0.616 [0.543, 0.685] | 0.675 [0.611, 0.735] | 0.636 [0.563, 0.704] |
| ctrlfr s2345 @ regione FLAME | 0.656 [0.598, 0.711] | 0.666 [0.609, 0.719] | 0.692 [0.634, 0.742] | 0.683 [0.626, 0.735] | 0.705 [0.647, 0.754] |
| C3M e123, d_F cal. | 0.666 [0.594, 0.727] | 0.420 [0.364, 0.473] | 0.472 [0.394, 0.543] | 0.611 [0.544, 0.671] | 0.479 [0.400, 0.550] |
| C3M e123, d_F cal. @ regione GNM | 0.694 [0.633, 0.747] | 0.682 [0.620, 0.736] | 0.661 [0.591, 0.722] | 0.695 [0.633, 0.749] | 0.672 [0.602, 0.732] |
| C3M e123, d_F cal. @ regione FLAME | 0.602 [0.533, 0.666] | 0.605 [0.535, 0.669] | 0.612 [0.542, 0.678] | 0.616 [0.547, 0.680] | 0.621 [0.552, 0.687] |
| C3M e205, d_F cal. | 0.712 [0.652, 0.763] | 0.427 [0.378, 0.472] | 0.449 [0.378, 0.511] | 0.637 [0.581, 0.685] | 0.453 [0.382, 0.516] |
| C3M e205, d_F cal. @ regione GNM | 0.694 [0.637, 0.746] | 0.687 [0.626, 0.742] | 0.674 [0.608, 0.735] | 0.707 [0.647, 0.759] | 0.692 [0.627, 0.752] |
| C3M e205, d_F cal. @ regione FLAME | 0.622 [0.550, 0.685] | 0.625 [0.555, 0.687] | 0.633 [0.564, 0.697] | 0.642 [0.573, 0.705] | 0.648 [0.578, 0.713] |
| factorized s1234, d_P | 0.677 [0.599, 0.746] | 0.443 [0.374, 0.512] | 0.459 [0.369, 0.545] | 0.617 [0.536, 0.691] | 0.472 [0.381, 0.558] |
| factorized s1234, d_P @ regione GNM | 0.686 [0.605, 0.759] | 0.685 [0.605, 0.757] | 0.682 [0.604, 0.755] | 0.693 [0.614, 0.765] | 0.687 [0.608, 0.759] |
| factorized s1234, d_P @ regione FLAME | 0.632 [0.557, 0.699] | 0.639 [0.564, 0.706] | 0.655 [0.579, 0.722] | 0.646 [0.571, 0.714] | 0.659 [0.584, 0.728] |
| factorized s2345, d_P | 0.684 [0.601, 0.762] | 0.422 [0.353, 0.485] | 0.426 [0.331, 0.507] | 0.606 [0.525, 0.682] | 0.433 [0.338, 0.515] |
| factorized s2345, d_P @ regione GNM | 0.728 [0.654, 0.793] | 0.718 [0.643, 0.785] | 0.700 [0.618, 0.776] | 0.722 [0.647, 0.790] | 0.702 [0.620, 0.777] |
| factorized s2345, d_P @ regione FLAME | 0.682 [0.611, 0.753] | 0.683 [0.611, 0.753] | 0.685 [0.612, 0.756] | 0.685 [0.614, 0.756] | 0.687 [0.614, 0.758] |
| C3M e123, d_P | 0.674 [0.600, 0.740] | 0.452 [0.384, 0.517] | 0.515 [0.431, 0.591] | 0.634 [0.559, 0.700] | 0.527 [0.443, 0.602] |
| C3M e123, d_P @ regione GNM | 0.694 [0.624, 0.756] | 0.687 [0.614, 0.750] | 0.675 [0.601, 0.740] | 0.703 [0.632, 0.767] | 0.689 [0.617, 0.754] |
| C3M e123, d_P @ regione FLAME | 0.614 [0.539, 0.682] | 0.618 [0.545, 0.686] | 0.629 [0.561, 0.697] | 0.634 [0.561, 0.701] | 0.643 [0.574, 0.709] |
| C3M e205, d_P | 0.684 [0.611, 0.749] | 0.458 [0.390, 0.522] | 0.498 [0.417, 0.570] | 0.637 [0.563, 0.703] | 0.511 [0.431, 0.584] |
| C3M e205, d_P @ regione GNM | 0.679 [0.611, 0.741] | 0.677 [0.609, 0.739] | 0.674 [0.601, 0.737] | 0.702 [0.635, 0.761] | 0.697 [0.627, 0.758] |
| C3M e205, d_P @ regione FLAME | 0.627 [0.556, 0.694] | 0.631 [0.559, 0.695] | 0.641 [0.570, 0.707] | 0.650 [0.579, 0.716] | 0.659 [0.588, 0.723] |
| GNM (visto) vB, coefficienti | 0.565 [0.492, 0.635] | 0.559 [0.485, 0.630] | 0.549 [0.471, 0.620] | 0.561 [0.487, 0.632] | 0.550 [0.473, 0.622] |
| GNM (visto) vB, mesh d'identita' FR | 0.680 [0.613, 0.739] | 0.661 [0.597, 0.715] | 0.628 [0.569, 0.680] | 0.664 [0.599, 0.718] | 0.630 [0.571, 0.682] |
| GNM (visto) vB, mesh d'identita' SR | 0.767 [0.678, 0.843] | 0.758 [0.675, 0.831] | 0.755 [0.682, 0.823] | 0.763 [0.682, 0.836] | 0.756 [0.683, 0.823] |
| FLAME 2023 Open vB, coefficienti | 0.549 [0.470, 0.626] | 0.535 [0.459, 0.608] | 0.514 [0.435, 0.586] | 0.539 [0.463, 0.613] | 0.515 [0.436, 0.588] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.631 [0.561, 0.693] | 0.606 [0.538, 0.668] | 0.561 [0.497, 0.621] | 0.608 [0.541, 0.670] | 0.562 [0.499, 0.623] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.711 [0.615, 0.792] | 0.707 [0.609, 0.788] | 0.703 [0.605, 0.784] | 0.709 [0.612, 0.789] | 0.703 [0.605, 0.785] |
| ICP + Chamfer in mm | 0.467 [0.404, 0.528] | 0.381 [0.326, 0.434] | 0.281 [0.242, 0.320] | 0.467 [0.412, 0.522] | 0.341 [0.298, 0.384] |
| Chamfer pura in mm | 0.608 [0.537, 0.679] | 0.570 [0.499, 0.641] | 0.504 [0.436, 0.573] | 0.639 [0.568, 0.711] | 0.615 [0.543, 0.688] |
| NICP su template in mm | 0.544 [0.470, 0.606] | 0.266 [0.228, 0.301] | 0.064 [0.018, 0.114] | 0.386 [0.335, 0.435] | 0.066 [0.018, 0.116] |

GT SR:

| metodo | senza crop | all_cross | righe col crop | media 15 coppie | media crop 5 coppie |
| --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal. | 0.690 [0.625, 0.751] | 0.392 [0.343, 0.439] | 0.383 [0.310, 0.450] | 0.601 [0.536, 0.661] | 0.407 [0.330, 0.477] |
| factorized s1234, d_F cal. @ regione GNM | 0.750 [0.696, 0.797] | 0.754 [0.698, 0.801] | 0.762 [0.705, 0.808] | 0.761 [0.707, 0.807] | 0.766 [0.709, 0.812] |
| factorized s1234, d_F cal. @ regione FLAME | 0.691 [0.631, 0.742] | 0.699 [0.641, 0.748] | 0.715 [0.659, 0.764] | 0.704 [0.647, 0.754] | 0.719 [0.663, 0.767] |
| factorized s2345, d_F cal. | 0.686 [0.621, 0.747] | 0.365 [0.323, 0.404] | 0.307 [0.237, 0.371] | 0.571 [0.509, 0.626] | 0.331 [0.256, 0.397] |
| factorized s2345, d_F cal. @ regione GNM | 0.751 [0.692, 0.802] | 0.750 [0.689, 0.801] | 0.749 [0.686, 0.801] | 0.753 [0.693, 0.804] | 0.750 [0.687, 0.802] |
| factorized s2345, d_F cal. @ regione FLAME | 0.726 [0.671, 0.777] | 0.732 [0.678, 0.782] | 0.743 [0.689, 0.793] | 0.734 [0.680, 0.785] | 0.745 [0.691, 0.795] |
| ctrlfr s1234 | 0.621 [0.546, 0.683] | 0.315 [0.271, 0.354] | 0.247 [0.179, 0.312] | 0.525 [0.456, 0.583] | 0.257 [0.186, 0.325] |
| ctrlfr s1234 @ regione GNM | 0.731 [0.669, 0.782] | 0.728 [0.668, 0.778] | 0.724 [0.661, 0.776] | 0.744 [0.688, 0.792] | 0.736 [0.673, 0.788] |
| ctrlfr s1234 @ regione FLAME | 0.639 [0.576, 0.692] | 0.652 [0.591, 0.703] | 0.682 [0.628, 0.731] | 0.681 [0.625, 0.731] | 0.705 [0.653, 0.753] |
| ctrlfr s2345 | 0.627 [0.553, 0.693] | 0.315 [0.273, 0.353] | 0.243 [0.178, 0.308] | 0.532 [0.463, 0.591] | 0.257 [0.187, 0.326] |
| ctrlfr s2345 @ regione GNM | 0.695 [0.630, 0.752] | 0.674 [0.607, 0.733] | 0.644 [0.570, 0.709] | 0.697 [0.631, 0.756] | 0.665 [0.593, 0.729] |
| ctrlfr s2345 @ regione FLAME | 0.657 [0.596, 0.712] | 0.666 [0.606, 0.722] | 0.690 [0.630, 0.742] | 0.683 [0.623, 0.736] | 0.702 [0.643, 0.755] |
| C3M e123, d_F cal. | 0.688 [0.621, 0.745] | 0.438 [0.383, 0.489] | 0.501 [0.422, 0.561] | 0.636 [0.570, 0.687] | 0.508 [0.429, 0.570] |
| C3M e123, d_F cal. @ regione GNM | 0.735 [0.680, 0.779] | 0.725 [0.671, 0.769] | 0.709 [0.649, 0.759] | 0.739 [0.686, 0.783] | 0.720 [0.659, 0.771] |
| C3M e123, d_F cal. @ regione FLAME | 0.660 [0.602, 0.712] | 0.663 [0.606, 0.715] | 0.671 [0.613, 0.724] | 0.675 [0.618, 0.726] | 0.681 [0.625, 0.734] |
| C3M e205, d_F cal. | 0.709 [0.646, 0.761] | 0.430 [0.381, 0.477] | 0.464 [0.396, 0.523] | 0.640 [0.584, 0.688] | 0.468 [0.399, 0.528] |
| C3M e205, d_F cal. @ regione GNM | 0.732 [0.682, 0.775] | 0.728 [0.678, 0.771] | 0.722 [0.668, 0.771] | 0.749 [0.699, 0.791] | 0.741 [0.687, 0.789] |
| C3M e205, d_F cal. @ regione FLAME | 0.678 [0.623, 0.726] | 0.682 [0.627, 0.730] | 0.694 [0.639, 0.743] | 0.700 [0.645, 0.748] | 0.709 [0.654, 0.759] |
| factorized s1234, d_P | 0.747 [0.681, 0.805] | 0.489 [0.424, 0.548] | 0.518 [0.429, 0.599] | 0.685 [0.616, 0.746] | 0.533 [0.443, 0.614] |
| factorized s1234, d_P @ regione GNM | 0.781 [0.726, 0.828] | 0.783 [0.728, 0.829] | 0.787 [0.730, 0.833] | 0.793 [0.740, 0.837] | 0.793 [0.737, 0.838] |
| factorized s1234, d_P @ regione FLAME | 0.728 [0.669, 0.776] | 0.735 [0.675, 0.783] | 0.750 [0.693, 0.798] | 0.743 [0.684, 0.790] | 0.755 [0.700, 0.802] |
| factorized s2345, d_P | 0.754 [0.688, 0.810] | 0.464 [0.402, 0.519] | 0.476 [0.384, 0.557] | 0.670 [0.603, 0.728] | 0.484 [0.393, 0.565] |
| factorized s2345, d_P @ regione GNM | 0.803 [0.752, 0.844] | 0.796 [0.744, 0.840] | 0.785 [0.724, 0.834] | 0.802 [0.750, 0.845] | 0.787 [0.727, 0.836] |
| factorized s2345, d_P @ regione FLAME | 0.777 [0.726, 0.824] | 0.780 [0.729, 0.826] | 0.786 [0.735, 0.831] | 0.783 [0.731, 0.828] | 0.789 [0.738, 0.833] |
| C3M e123, d_P | 0.746 [0.685, 0.797] | 0.500 [0.436, 0.557] | 0.578 [0.503, 0.644] | 0.704 [0.643, 0.757] | 0.591 [0.516, 0.657] |
| C3M e123, d_P @ regione GNM | 0.775 [0.723, 0.816] | 0.769 [0.716, 0.810] | 0.759 [0.704, 0.805] | 0.788 [0.737, 0.828] | 0.776 [0.722, 0.820] |
| C3M e123, d_P @ regione FLAME | 0.693 [0.636, 0.747] | 0.697 [0.641, 0.750] | 0.709 [0.654, 0.759] | 0.715 [0.659, 0.765] | 0.724 [0.672, 0.773] |
| C3M e205, d_P | 0.756 [0.698, 0.805] | 0.505 [0.444, 0.559] | 0.555 [0.481, 0.622] | 0.706 [0.646, 0.755] | 0.570 [0.495, 0.637] |
| C3M e205, d_P @ regione GNM | 0.762 [0.715, 0.803] | 0.762 [0.714, 0.803] | 0.763 [0.711, 0.806] | 0.790 [0.745, 0.829] | 0.790 [0.742, 0.830] |
| C3M e205, d_P @ regione FLAME | 0.709 [0.653, 0.759] | 0.712 [0.658, 0.761] | 0.723 [0.672, 0.770] | 0.733 [0.681, 0.780] | 0.743 [0.693, 0.788] |
| GNM (visto) vB, coefficienti | 0.617 [0.548, 0.676] | 0.611 [0.543, 0.670] | 0.601 [0.531, 0.662] | 0.613 [0.546, 0.672] | 0.602 [0.533, 0.664] |
| GNM (visto) vB, mesh d'identita' FR | 0.609 [0.533, 0.677] | 0.593 [0.521, 0.660] | 0.566 [0.493, 0.632] | 0.595 [0.522, 0.663] | 0.567 [0.495, 0.634] |
| GNM (visto) vB, mesh d'identita' SR | 0.854 [0.804, 0.894] | 0.840 [0.791, 0.882] | 0.830 [0.783, 0.873] | 0.847 [0.799, 0.888] | 0.831 [0.784, 0.874] |
| FLAME 2023 Open vB, coefficienti | 0.591 [0.517, 0.662] | 0.576 [0.502, 0.647] | 0.553 [0.478, 0.629] | 0.580 [0.506, 0.652] | 0.555 [0.480, 0.630] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.589 [0.515, 0.660] | 0.568 [0.498, 0.637] | 0.530 [0.462, 0.595] | 0.570 [0.500, 0.638] | 0.531 [0.463, 0.596] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.820 [0.759, 0.871] | 0.814 [0.752, 0.864] | 0.807 [0.745, 0.859] | 0.816 [0.755, 0.866] | 0.807 [0.745, 0.859] |
| ICP + Chamfer in mm | 0.491 [0.429, 0.549] | 0.401 [0.346, 0.454] | 0.294 [0.254, 0.333] | 0.490 [0.436, 0.542] | 0.357 [0.314, 0.400] |
| Chamfer pura in mm | 0.595 [0.518, 0.662] | 0.557 [0.481, 0.625] | 0.493 [0.416, 0.562] | 0.623 [0.545, 0.693] | 0.599 [0.517, 0.673] |
| NICP su template in mm | 0.398 [0.314, 0.471] | 0.195 [0.154, 0.231] | 0.044 [-0.003, 0.093] | 0.282 [0.223, 0.340] | 0.046 [-0.002, 0.095] |

Delta appaiati, righe col crop (braccio sulla regione di m - concorrente; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.139 [+0.061, +0.216], P 0.001 | +0.155 [+0.088, +0.227], P 0.000 | +0.165 [+0.090, +0.243], P 0.000 | +0.068 [-0.004, +0.140], P 0.033 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.059 [-0.003, +0.120], P 0.031 | +0.075 [+0.021, +0.130], P 0.009 | +0.086 [+0.032, +0.139], P 0.003 | -0.012 [-0.066, +0.046], P 0.646 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.068 [-0.118, -0.018], P 0.994 | -0.052 [-0.116, +0.013], P 0.948 | -0.041 [-0.094, +0.014], P 0.941 | -0.139 [-0.198, -0.080], P 1.000 |
| ICP + Chamfer in mm (@ regione GNM) | +0.407 [+0.349, +0.460], P 0.000 | +0.423 [+0.360, +0.481], P 0.000 | +0.433 [+0.384, +0.477], P 0.000 | +0.335 [+0.278, +0.391], P 0.000 |
| Chamfer pura in mm (@ regione GNM) | +0.183 [+0.121, +0.243], P 0.000 | +0.199 [+0.138, +0.266], P 0.000 | +0.209 [+0.154, +0.264], P 0.000 | +0.112 [+0.054, +0.174], P 0.000 |
| NICP su template in mm (@ regione GNM) | +0.623 [+0.532, +0.700], P 0.000 | +0.639 [+0.560, +0.709], P 0.000 | +0.649 [+0.570, +0.717], P 0.000 | +0.552 [+0.468, +0.628], P 0.000 |
| braccio intero (@ regione GNM) | +0.336 [+0.269, +0.401], P 0.000 | +0.425 [+0.352, +0.497], P 0.000 | +0.450 [+0.385, +0.506], P 0.000 | +0.382 [+0.329, +0.433], P 0.000 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.151 [+0.076, +0.231], P 0.000 | +0.183 [+0.108, +0.263], P 0.000 | +0.170 [+0.097, +0.246], P 0.000 | +0.178 [+0.107, +0.253], P 0.000 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.104 [+0.044, +0.162], P 0.000 | +0.136 [+0.079, +0.194], P 0.000 | +0.123 [+0.069, +0.175], P 0.000 | +0.132 [+0.079, +0.184], P 0.000 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.038 [-0.107, +0.033], P 0.862 | -0.006 [-0.073, +0.071], P 0.578 | -0.019 [-0.097, +0.057], P 0.687 | -0.011 [-0.081, +0.066], P 0.605 |
| ICP + Chamfer in mm (@ regione FLAME) | +0.383 [+0.324, +0.438], P 0.000 | +0.416 [+0.359, +0.471], P 0.000 | +0.403 [+0.354, +0.447], P 0.000 | +0.411 [+0.368, +0.453], P 0.000 |
| Chamfer pura in mm (@ regione FLAME) | +0.160 [+0.092, +0.228], P 0.000 | +0.192 [+0.121, +0.266], P 0.000 | +0.180 [+0.115, +0.244], P 0.000 | +0.188 [+0.130, +0.245], P 0.000 |
| NICP su template in mm (@ regione FLAME) | +0.600 [+0.513, +0.670], P 0.000 | +0.632 [+0.556, +0.702], P 0.000 | +0.620 [+0.546, +0.685], P 0.000 | +0.628 [+0.554, +0.691], P 0.000 |
| braccio intero (@ regione FLAME) | +0.313 [+0.243, +0.383], P 0.000 | +0.419 [+0.342, +0.496], P 0.000 | +0.420 [+0.355, +0.479], P 0.000 | +0.458 [+0.396, +0.517], P 0.000 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.186 [+0.131, +0.251], P 0.000 | +0.184 [+0.126, +0.249], P 0.000 | +0.123 [+0.064, +0.184], P 0.000 | +0.043 [-0.017, +0.109], P 0.089 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.221 [+0.161, +0.276], P 0.000 | +0.219 [+0.167, +0.267], P 0.000 | +0.158 [+0.103, +0.209], P 0.000 | +0.078 [+0.022, +0.130], P 0.002 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.043 [-0.090, +0.001], P 0.972 | -0.046 [-0.090, -0.005], P 0.982 | -0.106 [-0.157, -0.058], P 1.000 | -0.186 [-0.242, -0.134], P 1.000 |
| ICP + Chamfer in mm (@ regione GNM) | +0.493 [+0.455, +0.530], P 0.000 | +0.491 [+0.446, +0.532], P 0.000 | +0.430 [+0.384, +0.473], P 0.000 | +0.351 [+0.296, +0.401], P 0.000 |
| Chamfer pura in mm (@ regione GNM) | +0.294 [+0.237, +0.350], P 0.000 | +0.292 [+0.240, +0.349], P 0.000 | +0.232 [+0.176, +0.284], P 0.000 | +0.152 [+0.087, +0.213], P 0.000 |
| NICP su template in mm (@ regione GNM) | +0.743 [+0.678, +0.808], P 0.000 | +0.740 [+0.670, +0.806], P 0.000 | +0.680 [+0.611, +0.746], P 0.000 | +0.600 [+0.526, +0.672], P 0.000 |
| braccio intero (@ regione GNM) | +0.269 [+0.218, +0.326], P 0.000 | +0.308 [+0.253, +0.370], P 0.000 | +0.477 [+0.430, +0.523], P 0.000 | +0.401 [+0.352, +0.447], P 0.000 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.197 [+0.130, +0.267], P 0.000 | +0.233 [+0.164, +0.304], P 0.000 | +0.128 [+0.063, +0.194], P 0.000 | +0.136 [+0.067, +0.208], P 0.000 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.220 [+0.162, +0.280], P 0.000 | +0.257 [+0.203, +0.316], P 0.000 | +0.152 [+0.103, +0.203], P 0.000 | +0.160 [+0.109, +0.211], P 0.000 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.057 [-0.110, -0.004], P 0.977 | -0.020 [-0.068, +0.033], P 0.782 | -0.125 [-0.179, -0.068], P 1.000 | -0.117 [-0.168, -0.062], P 1.000 |
| ICP + Chamfer in mm (@ regione FLAME) | +0.456 [+0.412, +0.497], P 0.000 | +0.493 [+0.451, +0.533], P 0.000 | +0.388 [+0.342, +0.432], P 0.000 | +0.396 [+0.353, +0.436], P 0.000 |
| Chamfer pura in mm (@ regione FLAME) | +0.257 [+0.197, +0.321], P 0.000 | +0.294 [+0.234, +0.359], P 0.000 | +0.189 [+0.130, +0.247], P 0.000 | +0.197 [+0.140, +0.259], P 0.000 |
| NICP su template in mm (@ regione FLAME) | +0.706 [+0.632, +0.776], P 0.000 | +0.742 [+0.672, +0.806], P 0.000 | +0.638 [+0.566, +0.707], P 0.000 | +0.645 [+0.575, +0.713], P 0.000 |
| braccio intero (@ regione FLAME) | +0.232 [+0.175, +0.298], P 0.000 | +0.310 [+0.250, +0.381], P 0.000 | +0.435 [+0.383, +0.486], P 0.000 | +0.446 [+0.391, +0.501], P 0.000 |

Delta appaiati, media crop 5 coppie (braccio sulla regione di m - concorrente; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.141 [+0.063, +0.218], P 0.001 | +0.154 [+0.087, +0.226], P 0.000 | +0.176 [+0.100, +0.254], P 0.000 | +0.086 [+0.014, +0.158], P 0.009 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.061 [-0.002, +0.121], P 0.029 | +0.075 [+0.020, +0.129], P 0.009 | +0.096 [+0.042, +0.148], P 0.001 | +0.007 [-0.048, +0.065], P 0.380 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.065 [-0.115, -0.015], P 0.993 | -0.051 [-0.116, +0.014], P 0.947 | -0.030 [-0.080, +0.026], P 0.881 | -0.119 [-0.176, -0.061], P 1.000 |
| ICP + Chamfer in mm (@ regione GNM) | +0.350 [+0.290, +0.405], P 0.000 | +0.363 [+0.301, +0.421], P 0.000 | +0.384 [+0.335, +0.430], P 0.000 | +0.295 [+0.236, +0.352], P 0.000 |
| Chamfer pura in mm (@ regione GNM) | +0.076 [+0.013, +0.135], P 0.009 | +0.090 [+0.028, +0.153], P 0.002 | +0.111 [+0.054, +0.167], P 0.000 | +0.022 [-0.042, +0.083], P 0.258 |
| NICP su template in mm (@ regione GNM) | +0.625 [+0.533, +0.702], P 0.000 | +0.638 [+0.559, +0.709], P 0.000 | +0.660 [+0.580, +0.727], P 0.000 | +0.571 [+0.487, +0.645], P 0.000 |
| braccio intero (@ regione GNM) | +0.316 [+0.247, +0.382], P 0.000 | +0.404 [+0.327, +0.477], P 0.000 | +0.451 [+0.384, +0.512], P 0.000 | +0.389 [+0.333, +0.443], P 0.000 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.152 [+0.078, +0.233], P 0.000 | +0.183 [+0.109, +0.262], P 0.000 | +0.193 [+0.121, +0.270], P 0.000 | +0.190 [+0.118, +0.265], P 0.000 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.105 [+0.045, +0.164], P 0.000 | +0.136 [+0.078, +0.192], P 0.000 | +0.146 [+0.092, +0.199], P 0.000 | +0.142 [+0.090, +0.195], P 0.000 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.036 [-0.105, +0.037], P 0.848 | -0.005 [-0.073, +0.072], P 0.567 | +0.005 [-0.072, +0.082], P 0.461 | +0.002 [-0.069, +0.077], P 0.496 |
| ICP + Chamfer in mm (@ regione FLAME) | +0.326 [+0.267, +0.381], P 0.000 | +0.357 [+0.299, +0.411], P 0.000 | +0.367 [+0.316, +0.414], P 0.000 | +0.363 [+0.318, +0.408], P 0.000 |
| Chamfer pura in mm (@ regione FLAME) | +0.052 [-0.016, +0.117], P 0.074 | +0.083 [+0.012, +0.153], P 0.012 | +0.093 [+0.027, +0.160], P 0.002 | +0.090 [+0.028, +0.152], P 0.005 |
| NICP su template in mm (@ regione FLAME) | +0.601 [+0.516, +0.671], P 0.000 | +0.632 [+0.555, +0.701], P 0.000 | +0.642 [+0.571, +0.703], P 0.000 | +0.639 [+0.565, +0.704], P 0.000 |
| braccio intero (@ regione FLAME) | +0.293 [+0.218, +0.365], P 0.000 | +0.398 [+0.319, +0.478], P 0.000 | +0.434 [+0.366, +0.497], P 0.000 | +0.457 [+0.392, +0.520], P 0.000 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.191 [+0.134, +0.255], P 0.000 | +0.185 [+0.127, +0.251], P 0.000 | +0.134 [+0.075, +0.192], P 0.000 | +0.063 [+0.002, +0.129], P 0.023 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.225 [+0.166, +0.279], P 0.000 | +0.220 [+0.167, +0.269], P 0.000 | +0.169 [+0.114, +0.219], P 0.000 | +0.098 [+0.042, +0.150], P 0.000 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.038 [-0.085, +0.005], P 0.952 | -0.044 [-0.089, -0.004], P 0.980 | -0.095 [-0.145, -0.047], P 1.000 | -0.166 [-0.224, -0.115], P 1.000 |
| ICP + Chamfer in mm (@ regione GNM) | +0.436 [+0.397, +0.477], P 0.000 | +0.430 [+0.386, +0.473], P 0.000 | +0.379 [+0.332, +0.423], P 0.000 | +0.308 [+0.253, +0.362], P 0.000 |
| Chamfer pura in mm (@ regione GNM) | +0.194 [+0.133, +0.256], P 0.000 | +0.188 [+0.128, +0.251], P 0.000 | +0.137 [+0.082, +0.196], P 0.000 | +0.066 [-0.004, +0.132], P 0.031 |
| NICP su template in mm (@ regione GNM) | +0.747 [+0.683, +0.813], P 0.000 | +0.741 [+0.670, +0.807], P 0.000 | +0.690 [+0.622, +0.756], P 0.000 | +0.619 [+0.546, +0.693], P 0.000 |
| braccio intero (@ regione GNM) | +0.260 [+0.206, +0.318], P 0.000 | +0.303 [+0.246, +0.366], P 0.000 | +0.479 [+0.430, +0.528], P 0.000 | +0.408 [+0.357, +0.459], P 0.000 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.200 [+0.133, +0.270], P 0.000 | +0.234 [+0.165, +0.306], P 0.000 | +0.150 [+0.087, +0.217], P 0.000 | +0.147 [+0.079, +0.219], P 0.000 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.224 [+0.166, +0.282], P 0.000 | +0.258 [+0.203, +0.317], P 0.000 | +0.174 [+0.125, +0.224], P 0.000 | +0.171 [+0.122, +0.221], P 0.000 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.052 [-0.105, +0.002], P 0.971 | -0.019 [-0.066, +0.035], P 0.760 | -0.102 [-0.154, -0.044], P 0.999 | -0.105 [-0.155, -0.048], P 1.000 |
| ICP + Chamfer in mm (@ regione FLAME) | +0.398 [+0.351, +0.441], P 0.000 | +0.432 [+0.386, +0.475], P 0.000 | +0.348 [+0.300, +0.392], P 0.000 | +0.345 [+0.301, +0.388], P 0.000 |
| Chamfer pura in mm (@ regione FLAME) | +0.156 [+0.091, +0.223], P 0.000 | +0.190 [+0.120, +0.260], P 0.000 | +0.106 [+0.039, +0.171], P 0.001 | +0.103 [+0.040, +0.167], P 0.000 |
| NICP su template in mm (@ regione FLAME) | +0.709 [+0.635, +0.778], P 0.000 | +0.743 [+0.672, +0.807], P 0.000 | +0.659 [+0.589, +0.723], P 0.000 | +0.656 [+0.586, +0.723], P 0.000 |
| braccio intero (@ regione FLAME) | +0.222 [+0.163, +0.289], P 0.000 | +0.304 [+0.243, +0.377], P 0.000 | +0.448 [+0.393, +0.503], P 0.000 | +0.445 [+0.386, +0.502], P 0.000 |

Delta appaiati, senza crop (braccio sulla regione di m - concorrente; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.119 [+0.041, +0.195], P 0.001 | +0.146 [+0.079, +0.215], P 0.000 | +0.160 [+0.087, +0.237], P 0.000 | +0.111 [+0.040, +0.180], P 0.001 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.004 [-0.071, +0.077], P 0.440 | +0.031 [-0.032, +0.087], P 0.159 | +0.045 [-0.018, +0.106], P 0.072 | -0.003 [-0.065, +0.060], P 0.534 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.082 [-0.141, -0.023], P 0.995 | -0.056 [-0.128, +0.034], P 0.917 | -0.042 [-0.101, +0.022], P 0.910 | -0.090 [-0.152, -0.025], P 0.990 |
| ICP + Chamfer in mm (@ regione GNM) | +0.217 [+0.169, +0.269], P 0.000 | +0.244 [+0.187, +0.302], P 0.000 | +0.258 [+0.211, +0.306], P 0.000 | +0.210 [+0.166, +0.255], P 0.000 |
| Chamfer pura in mm (@ regione GNM) | +0.076 [+0.019, +0.128], P 0.007 | +0.103 [+0.045, +0.159], P 0.001 | +0.117 [+0.059, +0.175], P 0.000 | +0.068 [+0.012, +0.121], P 0.012 |
| NICP su template in mm (@ regione GNM) | +0.141 [+0.042, +0.237], P 0.002 | +0.167 [+0.079, +0.253], P 0.001 | +0.182 [+0.102, +0.261], P 0.000 | +0.133 [+0.047, +0.217], P 0.002 |
| braccio intero (@ regione GNM) | +0.025 [-0.021, +0.072], P 0.156 | +0.044 [+0.005, +0.082], P 0.011 | +0.063 [+0.021, +0.110], P 0.002 | +0.025 [-0.025, +0.076], P 0.169 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.091 [+0.016, +0.174], P 0.004 | +0.133 [+0.058, +0.212], P 0.000 | +0.090 [+0.019, +0.164], P 0.007 | +0.107 [+0.037, +0.185], P 0.002 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.010 [-0.056, +0.071], P 0.383 | +0.052 [-0.009, +0.110], P 0.042 | +0.008 [-0.048, +0.063], P 0.391 | +0.026 [-0.031, +0.082], P 0.196 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.071 [-0.144, +0.008], P 0.965 | -0.029 [-0.095, +0.048], P 0.790 | -0.072 [-0.151, +0.010], P 0.960 | -0.055 [-0.127, +0.024], P 0.930 |
| ICP + Chamfer in mm (@ regione FLAME) | +0.174 [+0.121, +0.225], P 0.000 | +0.216 [+0.164, +0.270], P 0.000 | +0.172 [+0.121, +0.223], P 0.000 | +0.189 [+0.144, +0.235], P 0.000 |
| Chamfer pura in mm (@ regione FLAME) | +0.032 [-0.028, +0.092], P 0.164 | +0.075 [+0.009, +0.137], P 0.012 | +0.031 [-0.031, +0.095], P 0.179 | +0.048 [-0.006, +0.104], P 0.052 |
| NICP su template in mm (@ regione FLAME) | +0.097 [-0.002, +0.188], P 0.027 | +0.139 [+0.047, +0.229], P 0.004 | +0.095 [+0.020, +0.172], P 0.005 | +0.113 [+0.036, +0.189], P 0.004 |
| braccio intero (@ regione FLAME) | -0.019 [-0.070, +0.029], P 0.784 | +0.016 [-0.030, +0.062], P 0.241 | -0.024 [-0.077, +0.032], P 0.823 | +0.004 [-0.048, +0.061], P 0.471 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.164 [+0.107, +0.229], P 0.000 | +0.186 [+0.126, +0.253], P 0.000 | +0.113 [+0.051, +0.176], P 0.000 | +0.078 [+0.016, +0.142], P 0.008 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.173 [+0.106, +0.234], P 0.000 | +0.195 [+0.131, +0.253], P 0.000 | +0.122 [+0.057, +0.179], P 0.000 | +0.086 [+0.028, +0.138], P 0.003 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.072 [-0.125, -0.025], P 0.993 | -0.050 [-0.096, -0.010], P 0.988 | -0.123 [-0.177, -0.076], P 1.000 | -0.159 [-0.212, -0.109], P 1.000 |
| ICP + Chamfer in mm (@ regione GNM) | +0.290 [+0.247, +0.337], P 0.000 | +0.312 [+0.267, +0.358], P 0.000 | +0.240 [+0.193, +0.289], P 0.000 | +0.204 [+0.158, +0.249], P 0.000 |
| Chamfer pura in mm (@ regione GNM) | +0.186 [+0.126, +0.249], P 0.000 | +0.208 [+0.151, +0.269], P 0.000 | +0.136 [+0.079, +0.194], P 0.000 | +0.100 [+0.037, +0.161], P 0.001 |
| NICP su template in mm (@ regione GNM) | +0.383 [+0.307, +0.466], P 0.000 | +0.405 [+0.329, +0.482], P 0.000 | +0.332 [+0.261, +0.408], P 0.000 | +0.297 [+0.222, +0.371], P 0.000 |
| braccio intero (@ regione GNM) | +0.034 [-0.006, +0.076], P 0.042 | +0.050 [+0.010, +0.091], P 0.006 | +0.110 [+0.073, +0.155], P 0.000 | +0.068 [+0.025, +0.116], P 0.002 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.137 [+0.072, +0.204], P 0.000 | +0.186 [+0.122, +0.253], P 0.000 | +0.048 [-0.013, +0.111], P 0.073 | +0.066 [-0.000, +0.135], P 0.027 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.139 [+0.076, +0.206], P 0.000 | +0.188 [+0.127, +0.250], P 0.000 | +0.050 [-0.009, +0.109], P 0.045 | +0.068 [+0.010, +0.125], P 0.008 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.092 [-0.148, -0.041], P 0.999 | -0.043 [-0.093, +0.006], P 0.956 | -0.181 [-0.238, -0.126], P 1.000 | -0.163 [-0.220, -0.106], P 1.000 |
| ICP + Chamfer in mm (@ regione FLAME) | +0.238 [+0.191, +0.287], P 0.000 | +0.286 [+0.241, +0.335], P 0.000 | +0.148 [+0.100, +0.196], P 0.000 | +0.166 [+0.125, +0.210], P 0.000 |
| Chamfer pura in mm (@ regione FLAME) | +0.134 [+0.073, +0.197], P 0.000 | +0.182 [+0.121, +0.246], P 0.000 | +0.044 [-0.018, +0.105], P 0.064 | +0.062 [+0.006, +0.122], P 0.015 |
| NICP su template in mm (@ regione FLAME) | +0.330 [+0.247, +0.414], P 0.000 | +0.379 [+0.303, +0.461], P 0.000 | +0.241 [+0.165, +0.317], P 0.000 | +0.259 [+0.187, +0.332], P 0.000 |
| braccio intero (@ regione FLAME) | -0.019 [-0.062, +0.031], P 0.788 | +0.023 [-0.017, +0.070], P 0.136 | +0.019 [-0.026, +0.067], P 0.219 | +0.030 [-0.018, +0.082], P 0.117 |


## faceverse: Spearman con la GT per gruppo (rho, IC 95%; 148500 righe all_cross, seme 566363)

GT FR:

| metodo | senza crop | all_cross | righe col crop | media 15 coppie | media crop 5 coppie |
| --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal. | 0.303 [0.231, 0.368] | 0.284 [0.211, 0.347] | 0.252 [0.176, 0.317] | 0.287 [0.213, 0.351] | 0.253 [0.176, 0.317] |
| factorized s1234, d_F cal. @ regione GNM | 0.287 [0.212, 0.356] | 0.293 [0.218, 0.358] | 0.305 [0.224, 0.373] | 0.294 [0.219, 0.360] | 0.305 [0.225, 0.373] |
| factorized s1234, d_F cal. @ regione FLAME | 0.246 [0.163, 0.322] | 0.241 [0.160, 0.318] | 0.231 [0.137, 0.316] | 0.241 [0.160, 0.319] | 0.231 [0.138, 0.316] |
| factorized s2345, d_F cal. | 0.318 [0.258, 0.376] | 0.288 [0.230, 0.342] | 0.234 [0.168, 0.290] | 0.290 [0.232, 0.345] | 0.234 [0.168, 0.291] |
| factorized s2345, d_F cal. @ regione GNM | 0.271 [0.198, 0.343] | 0.272 [0.200, 0.339] | 0.275 [0.202, 0.340] | 0.272 [0.200, 0.341] | 0.275 [0.202, 0.341] |
| factorized s2345, d_F cal. @ regione FLAME | 0.224 [0.148, 0.293] | 0.219 [0.145, 0.289] | 0.208 [0.127, 0.287] | 0.219 [0.146, 0.289] | 0.208 [0.127, 0.287] |
| ctrlfr s1234 | 0.282 [0.219, 0.348] | 0.273 [0.214, 0.335] | 0.256 [0.193, 0.318] | 0.281 [0.221, 0.344] | 0.263 [0.200, 0.327] |
| ctrlfr s1234 @ regione GNM | 0.264 [0.200, 0.322] | 0.268 [0.206, 0.324] | 0.276 [0.205, 0.335] | 0.274 [0.212, 0.332] | 0.281 [0.209, 0.341] |
| ctrlfr s1234 @ regione FLAME | 0.234 [0.160, 0.299] | 0.230 [0.156, 0.297] | 0.223 [0.139, 0.293] | 0.232 [0.157, 0.299] | 0.224 [0.140, 0.295] |
| ctrlfr s2345 | 0.258 [0.187, 0.329] | 0.251 [0.187, 0.319] | 0.239 [0.167, 0.305] | 0.255 [0.191, 0.324] | 0.241 [0.169, 0.307] |
| ctrlfr s2345 @ regione GNM | 0.241 [0.180, 0.300] | 0.246 [0.186, 0.306] | 0.257 [0.185, 0.320] | 0.250 [0.189, 0.310] | 0.261 [0.187, 0.324] |
| ctrlfr s2345 @ regione FLAME | 0.210 [0.140, 0.275] | 0.207 [0.137, 0.272] | 0.200 [0.122, 0.277] | 0.207 [0.138, 0.272] | 0.201 [0.122, 0.277] |
| C3M e123, d_F cal. | 0.289 [0.219, 0.350] | 0.277 [0.208, 0.336] | 0.253 [0.181, 0.316] | 0.283 [0.213, 0.343] | 0.257 [0.184, 0.322] |
| C3M e123, d_F cal. @ regione GNM | 0.261 [0.197, 0.326] | 0.265 [0.200, 0.325] | 0.272 [0.202, 0.334] | 0.269 [0.205, 0.330] | 0.275 [0.204, 0.338] |
| C3M e123, d_F cal. @ regione FLAME | 0.233 [0.159, 0.304] | 0.228 [0.154, 0.297] | 0.219 [0.137, 0.294] | 0.231 [0.156, 0.301] | 0.221 [0.139, 0.297] |
| C3M e205, d_F cal. | 0.290 [0.228, 0.347] | 0.276 [0.217, 0.330] | 0.250 [0.184, 0.307] | 0.288 [0.225, 0.345] | 0.259 [0.190, 0.319] |
| C3M e205, d_F cal. @ regione GNM | 0.259 [0.203, 0.315] | 0.262 [0.204, 0.320] | 0.269 [0.200, 0.329] | 0.272 [0.212, 0.333] | 0.276 [0.206, 0.338] |
| C3M e205, d_F cal. @ regione FLAME | 0.227 [0.156, 0.291] | 0.224 [0.154, 0.289] | 0.219 [0.139, 0.292] | 0.233 [0.160, 0.302] | 0.225 [0.143, 0.300] |
| factorized s1234, d_P | 0.283 [0.217, 0.351] | 0.271 [0.203, 0.339] | 0.252 [0.174, 0.324] | 0.273 [0.205, 0.341] | 0.252 [0.175, 0.324] |
| factorized s1234, d_P @ regione GNM | 0.285 [0.216, 0.351] | 0.291 [0.221, 0.355] | 0.305 [0.225, 0.374] | 0.292 [0.222, 0.356] | 0.305 [0.225, 0.374] |
| factorized s1234, d_P @ regione FLAME | 0.249 [0.165, 0.327] | 0.246 [0.166, 0.321] | 0.240 [0.151, 0.324] | 0.246 [0.166, 0.322] | 0.240 [0.151, 0.324] |
| factorized s2345, d_P | 0.308 [0.239, 0.374] | 0.287 [0.221, 0.351] | 0.251 [0.181, 0.314] | 0.289 [0.222, 0.353] | 0.252 [0.181, 0.315] |
| factorized s2345, d_P @ regione GNM | 0.278 [0.207, 0.347] | 0.278 [0.209, 0.347] | 0.280 [0.207, 0.346] | 0.279 [0.210, 0.348] | 0.280 [0.208, 0.346] |
| factorized s2345, d_P @ regione FLAME | 0.231 [0.153, 0.306] | 0.228 [0.150, 0.302] | 0.221 [0.136, 0.304] | 0.228 [0.150, 0.302] | 0.221 [0.136, 0.305] |
| C3M e123, d_P | 0.303 [0.225, 0.375] | 0.297 [0.221, 0.364] | 0.287 [0.209, 0.355] | 0.300 [0.223, 0.366] | 0.288 [0.210, 0.356] |
| C3M e123, d_P @ regione GNM | 0.283 [0.216, 0.351] | 0.286 [0.220, 0.352] | 0.292 [0.223, 0.357] | 0.288 [0.221, 0.354] | 0.293 [0.224, 0.358] |
| C3M e123, d_P @ regione FLAME | 0.248 [0.174, 0.321] | 0.246 [0.173, 0.317] | 0.242 [0.161, 0.315] | 0.247 [0.174, 0.321] | 0.243 [0.162, 0.317] |
| C3M e205, d_P | 0.310 [0.238, 0.377] | 0.305 [0.235, 0.370] | 0.297 [0.225, 0.364] | 0.309 [0.238, 0.374] | 0.298 [0.227, 0.366] |
| C3M e205, d_P @ regione GNM | 0.284 [0.218, 0.348] | 0.286 [0.223, 0.349] | 0.291 [0.221, 0.353] | 0.289 [0.226, 0.352] | 0.293 [0.223, 0.356] |
| C3M e205, d_P @ regione FLAME | 0.244 [0.170, 0.316] | 0.241 [0.169, 0.310] | 0.235 [0.155, 0.309] | 0.245 [0.171, 0.315] | 0.237 [0.156, 0.311] |
| GNM (visto) vB, coefficienti | 0.226 [0.142, 0.311] | 0.224 [0.142, 0.307] | 0.220 [0.130, 0.306] | 0.225 [0.143, 0.309] | 0.221 [0.131, 0.307] |
| GNM (visto) vB, mesh d'identita' FR | 0.292 [0.205, 0.371] | 0.289 [0.202, 0.368] | 0.281 [0.191, 0.359] | 0.289 [0.202, 0.368] | 0.281 [0.191, 0.359] |
| GNM (visto) vB, mesh d'identita' SR | 0.319 [0.234, 0.398] | 0.314 [0.230, 0.393] | 0.305 [0.226, 0.382] | 0.314 [0.231, 0.394] | 0.305 [0.226, 0.382] |
| FLAME 2023 Open vB, coefficienti | 0.203 [0.126, 0.282] | 0.195 [0.113, 0.275] | 0.179 [0.089, 0.272] | 0.196 [0.114, 0.277] | 0.179 [0.089, 0.273] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.264 [0.183, 0.342] | 0.258 [0.176, 0.337] | 0.245 [0.159, 0.328] | 0.258 [0.176, 0.338] | 0.245 [0.160, 0.329] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.290 [0.204, 0.364] | 0.283 [0.198, 0.358] | 0.270 [0.181, 0.347] | 0.283 [0.198, 0.359] | 0.270 [0.181, 0.348] |
| ICP + Chamfer in mm | 0.337 [0.256, 0.409] | 0.312 [0.237, 0.378] | 0.271 [0.204, 0.331] | 0.317 [0.241, 0.384] | 0.273 [0.206, 0.333] |
| Chamfer pura in mm | 0.366 [0.298, 0.432] | 0.363 [0.295, 0.428] | 0.362 [0.294, 0.427] | 0.367 [0.299, 0.432] | 0.364 [0.297, 0.430] |
| NICP su template in mm | 0.212 [0.101, 0.309] | 0.177 [0.087, 0.261] | 0.124 [0.051, 0.194] | 0.183 [0.089, 0.270] | 0.124 [0.051, 0.194] |

GT SR:

| metodo | senza crop | all_cross | righe col crop | media 15 coppie | media crop 5 coppie |
| --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal. | 0.269 [0.195, 0.340] | 0.251 [0.177, 0.321] | 0.220 [0.141, 0.291] | 0.253 [0.179, 0.324] | 0.220 [0.141, 0.291] |
| factorized s1234, d_F cal. @ regione GNM | 0.269 [0.188, 0.344] | 0.275 [0.191, 0.348] | 0.286 [0.191, 0.360] | 0.275 [0.192, 0.349] | 0.286 [0.191, 0.360] |
| factorized s1234, d_F cal. @ regione FLAME | 0.230 [0.139, 0.314] | 0.225 [0.133, 0.310] | 0.215 [0.111, 0.308] | 0.225 [0.133, 0.311] | 0.215 [0.111, 0.309] |
| factorized s2345, d_F cal. | 0.286 [0.212, 0.356] | 0.258 [0.187, 0.324] | 0.208 [0.134, 0.271] | 0.261 [0.189, 0.327] | 0.208 [0.135, 0.271] |
| factorized s2345, d_F cal. @ regione GNM | 0.267 [0.186, 0.341] | 0.267 [0.186, 0.339] | 0.267 [0.183, 0.338] | 0.267 [0.187, 0.340] | 0.267 [0.184, 0.338] |
| factorized s2345, d_F cal. @ regione FLAME | 0.221 [0.137, 0.297] | 0.216 [0.137, 0.291] | 0.208 [0.118, 0.285] | 0.217 [0.137, 0.291] | 0.208 [0.119, 0.285] |
| ctrlfr s1234 | 0.273 [0.205, 0.342] | 0.261 [0.198, 0.326] | 0.239 [0.174, 0.306] | 0.269 [0.206, 0.335] | 0.245 [0.180, 0.314] |
| ctrlfr s1234 @ regione GNM | 0.264 [0.198, 0.326] | 0.266 [0.196, 0.327] | 0.271 [0.195, 0.337] | 0.272 [0.202, 0.333] | 0.276 [0.198, 0.343] |
| ctrlfr s1234 @ regione FLAME | 0.237 [0.158, 0.308] | 0.233 [0.155, 0.303] | 0.225 [0.136, 0.300] | 0.235 [0.156, 0.306] | 0.226 [0.136, 0.302] |
| ctrlfr s2345 | 0.259 [0.187, 0.331] | 0.253 [0.187, 0.317] | 0.243 [0.172, 0.305] | 0.257 [0.191, 0.321] | 0.245 [0.174, 0.307] |
| ctrlfr s2345 @ regione GNM | 0.251 [0.187, 0.307] | 0.256 [0.191, 0.312] | 0.264 [0.186, 0.327] | 0.259 [0.194, 0.315] | 0.267 [0.189, 0.331] |
| ctrlfr s2345 @ regione FLAME | 0.214 [0.139, 0.283] | 0.211 [0.137, 0.281] | 0.205 [0.116, 0.281] | 0.212 [0.137, 0.282] | 0.205 [0.115, 0.282] |
| C3M e123, d_F cal. | 0.270 [0.192, 0.336] | 0.258 [0.180, 0.321] | 0.232 [0.156, 0.299] | 0.263 [0.184, 0.328] | 0.236 [0.158, 0.303] |
| C3M e123, d_F cal. @ regione GNM | 0.258 [0.189, 0.322] | 0.262 [0.193, 0.325] | 0.272 [0.193, 0.337] | 0.266 [0.195, 0.329] | 0.274 [0.195, 0.341] |
| C3M e123, d_F cal. @ regione FLAME | 0.233 [0.149, 0.312] | 0.227 [0.142, 0.304] | 0.215 [0.116, 0.302] | 0.230 [0.143, 0.309] | 0.217 [0.119, 0.304] |
| C3M e205, d_F cal. | 0.275 [0.207, 0.334] | 0.263 [0.197, 0.319] | 0.239 [0.170, 0.299] | 0.274 [0.205, 0.331] | 0.247 [0.176, 0.311] |
| C3M e205, d_F cal. @ regione GNM | 0.258 [0.199, 0.317] | 0.263 [0.202, 0.322] | 0.273 [0.198, 0.336] | 0.272 [0.206, 0.333] | 0.280 [0.204, 0.346] |
| C3M e205, d_F cal. @ regione FLAME | 0.228 [0.148, 0.301] | 0.224 [0.141, 0.297] | 0.217 [0.118, 0.300] | 0.233 [0.147, 0.310] | 0.223 [0.122, 0.308] |
| factorized s1234, d_P | 0.286 [0.217, 0.354] | 0.273 [0.208, 0.342] | 0.251 [0.175, 0.323] | 0.275 [0.210, 0.344] | 0.252 [0.176, 0.323] |
| factorized s1234, d_P @ regione GNM | 0.292 [0.219, 0.361] | 0.299 [0.223, 0.363] | 0.312 [0.231, 0.379] | 0.299 [0.223, 0.365] | 0.312 [0.231, 0.380] |
| factorized s1234, d_P @ regione FLAME | 0.263 [0.179, 0.341] | 0.259 [0.173, 0.337] | 0.249 [0.161, 0.335] | 0.258 [0.173, 0.337] | 0.249 [0.160, 0.336] |
| factorized s2345, d_P | 0.313 [0.248, 0.378] | 0.293 [0.227, 0.356] | 0.257 [0.189, 0.322] | 0.295 [0.229, 0.358] | 0.257 [0.189, 0.323] |
| factorized s2345, d_P @ regione GNM | 0.293 [0.216, 0.362] | 0.292 [0.219, 0.361] | 0.293 [0.221, 0.359] | 0.293 [0.220, 0.362] | 0.293 [0.221, 0.360] |
| factorized s2345, d_P @ regione FLAME | 0.252 [0.171, 0.325] | 0.249 [0.167, 0.321] | 0.243 [0.157, 0.319] | 0.249 [0.168, 0.322] | 0.243 [0.158, 0.319] |
| C3M e123, d_P | 0.313 [0.238, 0.386] | 0.306 [0.232, 0.375] | 0.294 [0.219, 0.364] | 0.309 [0.235, 0.378] | 0.295 [0.220, 0.366] |
| C3M e123, d_P @ regione GNM | 0.292 [0.221, 0.364] | 0.296 [0.225, 0.364] | 0.305 [0.233, 0.375] | 0.298 [0.227, 0.368] | 0.306 [0.234, 0.376] |
| C3M e123, d_P @ regione FLAME | 0.266 [0.186, 0.346] | 0.262 [0.181, 0.335] | 0.254 [0.170, 0.333] | 0.263 [0.183, 0.339] | 0.254 [0.171, 0.335] |
| C3M e205, d_P | 0.314 [0.243, 0.377] | 0.312 [0.244, 0.374] | 0.308 [0.239, 0.372] | 0.315 [0.247, 0.377] | 0.310 [0.239, 0.373] |
| C3M e205, d_P @ regione GNM | 0.292 [0.224, 0.355] | 0.296 [0.229, 0.358] | 0.305 [0.232, 0.371] | 0.299 [0.232, 0.362] | 0.307 [0.234, 0.374] |
| C3M e205, d_P @ regione FLAME | 0.261 [0.184, 0.337] | 0.257 [0.177, 0.330] | 0.248 [0.160, 0.330] | 0.260 [0.180, 0.336] | 0.250 [0.162, 0.333] |
| GNM (visto) vB, coefficienti | 0.222 [0.133, 0.311] | 0.223 [0.134, 0.310] | 0.224 [0.130, 0.310] | 0.224 [0.135, 0.312] | 0.225 [0.130, 0.311] |
| GNM (visto) vB, mesh d'identita' FR | 0.244 [0.144, 0.333] | 0.241 [0.141, 0.330] | 0.235 [0.135, 0.327] | 0.241 [0.141, 0.330] | 0.235 [0.135, 0.328] |
| GNM (visto) vB, mesh d'identita' SR | 0.327 [0.239, 0.407] | 0.323 [0.237, 0.402] | 0.316 [0.230, 0.398] | 0.324 [0.238, 0.404] | 0.316 [0.230, 0.399] |
| FLAME 2023 Open vB, coefficienti | 0.193 [0.109, 0.272] | 0.188 [0.104, 0.269] | 0.178 [0.085, 0.271] | 0.189 [0.105, 0.272] | 0.179 [0.085, 0.272] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.218 [0.120, 0.312] | 0.213 [0.116, 0.307] | 0.202 [0.106, 0.299] | 0.213 [0.116, 0.307] | 0.202 [0.106, 0.299] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.293 [0.199, 0.371] | 0.288 [0.195, 0.368] | 0.278 [0.179, 0.361] | 0.288 [0.195, 0.369] | 0.278 [0.180, 0.362] |
| ICP + Chamfer in mm | 0.309 [0.229, 0.380] | 0.287 [0.213, 0.355] | 0.251 [0.183, 0.312] | 0.291 [0.216, 0.359] | 0.253 [0.184, 0.315] |
| Chamfer pura in mm | 0.342 [0.276, 0.406] | 0.339 [0.277, 0.404] | 0.340 [0.272, 0.404] | 0.343 [0.282, 0.408] | 0.342 [0.275, 0.407] |
| NICP su template in mm | 0.152 [0.042, 0.250] | 0.125 [0.032, 0.205] | 0.085 [0.009, 0.147] | 0.130 [0.035, 0.212] | 0.085 [0.009, 0.147] |

Delta appaiati, righe col crop (braccio sulla regione di m - concorrente; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.085 [-0.022, +0.190], P 0.056 | +0.055 [-0.045, +0.159], P 0.150 | +0.055 [-0.040, +0.152], P 0.148 | +0.037 [-0.060, +0.137], P 0.233 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.025 [-0.048, +0.094], P 0.246 | -0.006 [-0.072, +0.066], P 0.568 | -0.005 [-0.091, +0.073], P 0.561 | -0.023 [-0.105, +0.062], P 0.698 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | +0.001 [-0.091, +0.086], P 0.495 | -0.030 [-0.113, +0.053], P 0.774 | -0.029 [-0.113, +0.051], P 0.772 | -0.047 [-0.127, +0.044], P 0.869 |
| ICP + Chamfer in mm (@ regione GNM) | +0.034 [-0.039, +0.105], P 0.153 | +0.003 [-0.058, +0.072], P 0.469 | +0.004 [-0.066, +0.075], P 0.472 | -0.014 [-0.084, +0.065], P 0.617 |
| Chamfer pura in mm (@ regione GNM) | -0.057 [-0.131, +0.012], P 0.940 | -0.087 [-0.152, -0.011], P 0.986 | -0.087 [-0.158, -0.019], P 0.996 | -0.105 [-0.183, -0.029], P 0.996 |
| NICP su template in mm (@ regione GNM) | +0.181 [+0.106, +0.256], P 0.000 | +0.151 [+0.080, +0.224], P 0.000 | +0.151 [+0.076, +0.232], P 0.000 | +0.133 [+0.052, +0.217], P 0.001 |
| braccio intero (@ regione GNM) | +0.053 [+0.000, +0.103], P 0.025 | +0.041 [-0.018, +0.097], P 0.085 | +0.020 [-0.037, +0.070], P 0.242 | +0.019 [-0.037, +0.073], P 0.277 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.052 [-0.060, +0.159], P 0.189 | +0.030 [-0.074, +0.137], P 0.319 | +0.044 [-0.064, +0.150], P 0.229 | +0.022 [-0.080, +0.123], P 0.356 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | -0.014 [-0.103, +0.074], P 0.651 | -0.037 [-0.122, +0.047], P 0.812 | -0.022 [-0.125, +0.070], P 0.690 | -0.045 [-0.130, +0.044], P 0.846 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.039 [-0.145, +0.061], P 0.785 | -0.062 [-0.151, +0.037], P 0.908 | -0.047 [-0.142, +0.045], P 0.846 | -0.070 [-0.153, +0.027], P 0.936 |
| ICP + Chamfer in mm (@ regione FLAME) | -0.040 [-0.115, +0.039], P 0.845 | -0.063 [-0.134, +0.016], P 0.947 | -0.048 [-0.129, +0.028], P 0.885 | -0.071 [-0.145, +0.006], P 0.967 |
| Chamfer pura in mm (@ regione FLAME) | -0.131 [-0.214, -0.053], P 0.999 | -0.154 [-0.237, -0.064], P 0.999 | -0.139 [-0.230, -0.056], P 0.999 | -0.162 [-0.246, -0.079], P 1.000 |
| NICP su template in mm (@ regione FLAME) | +0.107 [+0.017, +0.194], P 0.006 | +0.084 [+0.002, +0.173], P 0.022 | +0.099 [+0.006, +0.189], P 0.017 | +0.076 [-0.008, +0.171], P 0.043 |
| braccio intero (@ regione FLAME) | -0.021 [-0.099, +0.051], P 0.721 | -0.026 [-0.104, +0.048], P 0.761 | -0.033 [-0.108, +0.040], P 0.809 | -0.038 [-0.108, +0.031], P 0.871 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.087 [-0.019, +0.188], P 0.052 | +0.069 [-0.032, +0.167], P 0.097 | +0.047 [-0.054, +0.143], P 0.190 | +0.040 [-0.066, +0.144], P 0.225 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.077 [-0.021, +0.171], P 0.057 | +0.058 [-0.033, +0.155], P 0.104 | +0.036 [-0.048, +0.118], P 0.215 | +0.029 [-0.048, +0.108], P 0.250 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.004 [-0.089, +0.075], P 0.539 | -0.023 [-0.097, +0.053], P 0.721 | -0.045 [-0.135, +0.042], P 0.848 | -0.052 [-0.142, +0.043], P 0.883 |
| ICP + Chamfer in mm (@ regione GNM) | +0.061 [-0.021, +0.138], P 0.057 | +0.042 [-0.027, +0.118], P 0.124 | +0.020 [-0.048, +0.090], P 0.293 | +0.013 [-0.059, +0.087], P 0.374 |
| Chamfer pura in mm (@ regione GNM) | -0.028 [-0.106, +0.044], P 0.768 | -0.047 [-0.115, +0.025], P 0.888 | -0.069 [-0.142, -0.003], P 0.980 | -0.076 [-0.153, -0.000], P 0.975 |
| NICP su template in mm (@ regione GNM) | +0.227 [+0.142, +0.313], P 0.000 | +0.208 [+0.129, +0.291], P 0.000 | +0.186 [+0.114, +0.256], P 0.000 | +0.179 [+0.110, +0.253], P 0.000 |
| braccio intero (@ regione GNM) | +0.060 [+0.010, +0.115], P 0.013 | +0.036 [-0.025, +0.097], P 0.131 | +0.032 [-0.030, +0.086], P 0.149 | +0.021 [-0.035, +0.075], P 0.243 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.071 [-0.043, +0.176], P 0.122 | +0.065 [-0.046, +0.167], P 0.134 | +0.047 [-0.058, +0.147], P 0.210 | +0.027 [-0.080, +0.122], P 0.323 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.047 [-0.057, +0.139], P 0.186 | +0.041 [-0.068, +0.139], P 0.212 | +0.023 [-0.082, +0.114], P 0.313 | +0.003 [-0.084, +0.089], P 0.494 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.029 [-0.116, +0.060], P 0.750 | -0.035 [-0.118, +0.053], P 0.804 | -0.053 [-0.146, +0.037], P 0.873 | -0.073 [-0.167, +0.019], P 0.941 |
| ICP + Chamfer in mm (@ regione FLAME) | -0.001 [-0.085, +0.082], P 0.515 | -0.008 [-0.091, +0.074], P 0.579 | -0.026 [-0.110, +0.051], P 0.749 | -0.046 [-0.124, +0.030], P 0.885 |
| Chamfer pura in mm (@ regione FLAME) | -0.090 [-0.182, -0.010], P 0.985 | -0.097 [-0.187, -0.011], P 0.986 | -0.115 [-0.198, -0.036], P 0.997 | -0.135 [-0.216, -0.052], P 1.000 |
| NICP su template in mm (@ regione FLAME) | +0.164 [+0.068, +0.260], P 0.000 | +0.158 [+0.061, +0.248], P 0.001 | +0.140 [+0.048, +0.223], P 0.001 | +0.120 [+0.034, +0.206], P 0.005 |
| braccio intero (@ regione FLAME) | -0.002 [-0.085, +0.082], P 0.541 | -0.014 [-0.096, +0.063], P 0.640 | -0.014 [-0.090, +0.062], P 0.647 | -0.038 [-0.107, +0.031], P 0.870 |

Delta appaiati, media crop 5 coppie (braccio sulla regione di m - concorrente; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.085 [-0.023, +0.190], P 0.057 | +0.054 [-0.046, +0.159], P 0.155 | +0.060 [-0.037, +0.157], P 0.132 | +0.040 [-0.059, +0.140], P 0.217 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.025 [-0.048, +0.095], P 0.246 | -0.006 [-0.072, +0.066], P 0.568 | -0.000 [-0.088, +0.080], P 0.511 | -0.020 [-0.102, +0.065], P 0.666 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | +0.001 [-0.091, +0.086], P 0.497 | -0.030 [-0.114, +0.053], P 0.775 | -0.024 [-0.109, +0.057], P 0.731 | -0.044 [-0.125, +0.048], P 0.855 |
| ICP + Chamfer in mm (@ regione GNM) | +0.032 [-0.041, +0.103], P 0.165 | +0.002 [-0.060, +0.071], P 0.489 | +0.007 [-0.063, +0.080], P 0.439 | -0.013 [-0.082, +0.067], P 0.604 |
| Chamfer pura in mm (@ regione GNM) | -0.059 [-0.134, +0.010], P 0.948 | -0.089 [-0.155, -0.014], P 0.987 | -0.084 [-0.156, -0.015], P 0.994 | -0.104 [-0.184, -0.027], P 0.996 |
| NICP su template in mm (@ regione GNM) | +0.181 [+0.107, +0.256], P 0.000 | +0.151 [+0.080, +0.224], P 0.000 | +0.156 [+0.081, +0.237], P 0.000 | +0.136 [+0.054, +0.220], P 0.001 |
| braccio intero (@ regione GNM) | +0.053 [+0.000, +0.103], P 0.025 | +0.041 [-0.018, +0.096], P 0.086 | +0.018 [-0.039, +0.070], P 0.269 | +0.020 [-0.036, +0.074], P 0.260 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.052 [-0.061, +0.158], P 0.193 | +0.029 [-0.076, +0.137], P 0.326 | +0.045 [-0.064, +0.152], P 0.226 | +0.021 [-0.080, +0.122], P 0.362 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | -0.014 [-0.103, +0.074], P 0.653 | -0.037 [-0.122, +0.047], P 0.814 | -0.021 [-0.124, +0.072], P 0.680 | -0.044 [-0.130, +0.045], P 0.840 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.039 [-0.145, +0.061], P 0.786 | -0.062 [-0.151, +0.037], P 0.909 | -0.046 [-0.141, +0.046], P 0.844 | -0.069 [-0.153, +0.027], P 0.936 |
| ICP + Chamfer in mm (@ regione FLAME) | -0.042 [-0.117, +0.038], P 0.855 | -0.065 [-0.136, +0.015], P 0.949 | -0.049 [-0.130, +0.027], P 0.885 | -0.073 [-0.146, +0.005], P 0.967 |
| Chamfer pura in mm (@ regione FLAME) | -0.133 [-0.217, -0.055], P 0.999 | -0.156 [-0.239, -0.066], P 0.999 | -0.140 [-0.231, -0.056], P 0.999 | -0.164 [-0.250, -0.081], P 1.000 |
| NICP su template in mm (@ regione FLAME) | +0.107 [+0.017, +0.194], P 0.006 | +0.084 [+0.001, +0.173], P 0.023 | +0.100 [+0.007, +0.191], P 0.016 | +0.076 [-0.008, +0.171], P 0.042 |
| braccio intero (@ regione FLAME) | -0.022 [-0.100, +0.051], P 0.721 | -0.026 [-0.104, +0.048], P 0.762 | -0.038 [-0.114, +0.035], P 0.851 | -0.040 [-0.110, +0.030], P 0.880 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.087 [-0.021, +0.188], P 0.056 | +0.068 [-0.033, +0.167], P 0.098 | +0.051 [-0.052, +0.148], P 0.173 | +0.042 [-0.064, +0.147], P 0.213 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.077 [-0.021, +0.171], P 0.057 | +0.058 [-0.033, +0.155], P 0.100 | +0.041 [-0.042, +0.122], P 0.183 | +0.032 [-0.046, +0.111], P 0.224 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.004 [-0.090, +0.075], P 0.542 | -0.023 [-0.098, +0.053], P 0.727 | -0.040 [-0.131, +0.046], P 0.825 | -0.049 [-0.139, +0.047], P 0.867 |
| ICP + Chamfer in mm (@ regione GNM) | +0.059 [-0.022, +0.137], P 0.063 | +0.041 [-0.029, +0.117], P 0.129 | +0.023 [-0.047, +0.094], P 0.262 | +0.014 [-0.058, +0.089], P 0.361 |
| Chamfer pura in mm (@ regione GNM) | -0.030 [-0.109, +0.042], P 0.789 | -0.049 [-0.117, +0.024], P 0.901 | -0.066 [-0.140, +0.001], P 0.973 | -0.075 [-0.154, +0.002], P 0.972 |
| NICP su template in mm (@ regione GNM) | +0.227 [+0.141, +0.313], P 0.000 | +0.208 [+0.128, +0.291], P 0.000 | +0.191 [+0.119, +0.261], P 0.000 | +0.182 [+0.113, +0.256], P 0.000 |
| braccio intero (@ regione GNM) | +0.060 [+0.010, +0.115], P 0.013 | +0.036 [-0.026, +0.097], P 0.132 | +0.030 [-0.032, +0.085], P 0.170 | +0.022 [-0.034, +0.076], P 0.228 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.070 [-0.044, +0.175], P 0.126 | +0.064 [-0.047, +0.167], P 0.138 | +0.047 [-0.058, +0.148], P 0.209 | +0.026 [-0.080, +0.122], P 0.327 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.047 [-0.058, +0.139], P 0.187 | +0.041 [-0.068, +0.139], P 0.214 | +0.024 [-0.082, +0.115], P 0.305 | +0.003 [-0.083, +0.089], P 0.494 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.029 [-0.117, +0.060], P 0.750 | -0.036 [-0.119, +0.053], P 0.806 | -0.052 [-0.145, +0.039], P 0.870 | -0.073 [-0.167, +0.019], P 0.941 |
| ICP + Chamfer in mm (@ regione FLAME) | -0.003 [-0.088, +0.080], P 0.531 | -0.010 [-0.094, +0.073], P 0.598 | -0.026 [-0.111, +0.051], P 0.752 | -0.047 [-0.126, +0.029], P 0.896 |
| Chamfer pura in mm (@ regione FLAME) | -0.093 [-0.184, -0.011], P 0.989 | -0.099 [-0.190, -0.013], P 0.987 | -0.116 [-0.200, -0.036], P 0.997 | -0.137 [-0.218, -0.054], P 1.000 |
| NICP su template in mm (@ regione FLAME) | +0.164 [+0.068, +0.260], P 0.000 | +0.158 [+0.061, +0.248], P 0.001 | +0.141 [+0.048, +0.225], P 0.001 | +0.121 [+0.033, +0.207], P 0.003 |
| braccio intero (@ regione FLAME) | -0.003 [-0.086, +0.081], P 0.551 | -0.015 [-0.096, +0.063], P 0.647 | -0.019 [-0.097, +0.056], P 0.697 | -0.039 [-0.109, +0.030], P 0.879 |

Delta appaiati, senza crop (braccio sulla regione di m - concorrente; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.062 [-0.043, +0.166], P 0.127 | +0.045 [-0.056, +0.149], P 0.190 | +0.038 [-0.050, +0.129], P 0.203 | +0.015 [-0.077, +0.112], P 0.365 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | -0.005 [-0.075, +0.061], P 0.559 | -0.022 [-0.091, +0.052], P 0.721 | -0.028 [-0.107, +0.054], P 0.749 | -0.052 [-0.133, +0.036], P 0.873 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.031 [-0.114, +0.049], P 0.773 | -0.048 [-0.130, +0.034], P 0.870 | -0.054 [-0.125, +0.021], P 0.917 | -0.078 [-0.148, +0.003], P 0.971 |
| ICP + Chamfer in mm (@ regione GNM) | -0.050 [-0.120, +0.018], P 0.915 | -0.066 [-0.137, +0.007], P 0.957 | -0.073 [-0.147, +0.006], P 0.967 | -0.096 [-0.174, -0.011], P 0.988 |
| Chamfer pura in mm (@ regione GNM) | -0.079 [-0.151, -0.005], P 0.985 | -0.096 [-0.167, -0.017], P 0.991 | -0.102 [-0.171, -0.035], P 0.999 | -0.126 [-0.197, -0.051], P 1.000 |
| NICP su template in mm (@ regione GNM) | +0.076 [-0.020, +0.170], P 0.055 | +0.059 [-0.039, +0.166], P 0.101 | +0.052 [-0.052, +0.170], P 0.157 | +0.029 [-0.075, +0.144], P 0.273 |
| braccio intero (@ regione GNM) | -0.015 [-0.072, +0.043], P 0.711 | -0.047 [-0.120, +0.024], P 0.919 | -0.018 [-0.080, +0.042], P 0.710 | -0.017 [-0.075, +0.038], P 0.723 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.043 [-0.060, +0.135], P 0.210 | +0.021 [-0.076, +0.112], P 0.343 | +0.031 [-0.065, +0.116], P 0.258 | +0.007 [-0.077, +0.089], P 0.447 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | -0.018 [-0.095, +0.055], P 0.695 | -0.040 [-0.112, +0.030], P 0.870 | -0.031 [-0.123, +0.050], P 0.764 | -0.054 [-0.136, +0.026], P 0.900 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.043 [-0.136, +0.048], P 0.830 | -0.065 [-0.146, +0.020], P 0.934 | -0.056 [-0.132, +0.016], P 0.928 | -0.080 [-0.155, -0.002], P 0.978 |
| ICP + Chamfer in mm (@ regione FLAME) | -0.091 [-0.161, -0.017], P 0.994 | -0.113 [-0.185, -0.034], P 0.998 | -0.104 [-0.193, -0.022], P 0.992 | -0.127 [-0.214, -0.045], P 0.998 |
| Chamfer pura in mm (@ regione FLAME) | -0.120 [-0.199, -0.043], P 0.999 | -0.142 [-0.220, -0.059], P 1.000 | -0.133 [-0.210, -0.061], P 1.000 | -0.156 [-0.230, -0.085], P 1.000 |
| NICP su template in mm (@ regione FLAME) | +0.034 [-0.059, +0.132], P 0.235 | +0.013 [-0.082, +0.116], P 0.381 | +0.022 [-0.093, +0.143], P 0.350 | -0.002 [-0.116, +0.111], P 0.497 |
| braccio intero (@ regione FLAME) | -0.057 [-0.131, +0.016], P 0.943 | -0.094 [-0.168, -0.021], P 0.993 | -0.049 [-0.117, +0.018], P 0.924 | -0.048 [-0.114, +0.016], P 0.927 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.071 [-0.032, +0.172], P 0.092 | +0.071 [-0.028, +0.171], P 0.076 | +0.043 [-0.048, +0.132], P 0.192 | +0.030 [-0.063, +0.126], P 0.259 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.049 [-0.046, +0.151], P 0.165 | +0.049 [-0.038, +0.146], P 0.150 | +0.021 [-0.060, +0.106], P 0.322 | +0.008 [-0.073, +0.088], P 0.423 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.035 [-0.109, +0.036], P 0.818 | -0.034 [-0.107, +0.038], P 0.815 | -0.063 [-0.137, +0.016], P 0.941 | -0.076 [-0.151, +0.004], P 0.969 |
| ICP + Chamfer in mm (@ regione GNM) | -0.017 [-0.099, +0.061], P 0.663 | -0.016 [-0.095, +0.063], P 0.638 | -0.045 [-0.118, +0.032], P 0.887 | -0.058 [-0.127, +0.020], P 0.931 |
| Chamfer pura in mm (@ regione GNM) | -0.049 [-0.122, +0.017], P 0.918 | -0.049 [-0.115, +0.020], P 0.913 | -0.077 [-0.143, -0.015], P 0.987 | -0.090 [-0.157, -0.023], P 0.994 |
| NICP su template in mm (@ regione GNM) | +0.141 [+0.022, +0.262], P 0.009 | +0.141 [+0.024, +0.259], P 0.006 | +0.113 [+0.018, +0.213], P 0.011 | +0.100 [+0.004, +0.202], P 0.021 |
| braccio intero (@ regione GNM) | +0.006 [-0.050, +0.061], P 0.433 | -0.020 [-0.092, +0.047], P 0.719 | -0.008 [-0.073, +0.052], P 0.614 | -0.008 [-0.066, +0.049], P 0.626 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.071 [-0.034, +0.168], P 0.094 | +0.060 [-0.044, +0.148], P 0.123 | +0.045 [-0.049, +0.128], P 0.168 | +0.022 [-0.060, +0.099], P 0.312 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.045 [-0.060, +0.144], P 0.195 | +0.034 [-0.066, +0.126], P 0.249 | +0.019 [-0.081, +0.104], P 0.342 | -0.004 [-0.093, +0.077], P 0.556 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.029 [-0.105, +0.049], P 0.777 | -0.040 [-0.116, +0.036], P 0.853 | -0.055 [-0.133, +0.025], P 0.918 | -0.078 [-0.155, +0.000], P 0.974 |
| ICP + Chamfer in mm (@ regione FLAME) | -0.046 [-0.138, +0.036], P 0.877 | -0.057 [-0.147, +0.028], P 0.911 | -0.072 [-0.159, +0.008], P 0.964 | -0.095 [-0.176, -0.019], P 0.989 |
| Chamfer pura in mm (@ regione FLAME) | -0.079 [-0.157, -0.004], P 0.980 | -0.090 [-0.172, -0.008], P 0.984 | -0.104 [-0.178, -0.034], P 0.996 | -0.127 [-0.201, -0.053], P 1.000 |
| NICP su template in mm (@ regione FLAME) | +0.111 [-0.011, +0.228], P 0.039 | +0.100 [-0.019, +0.216], P 0.044 | +0.086 [-0.021, +0.198], P 0.073 | +0.063 [-0.043, +0.168], P 0.125 |
| braccio intero (@ regione FLAME) | -0.023 [-0.094, +0.047], P 0.733 | -0.061 [-0.139, +0.013], P 0.942 | -0.035 [-0.105, +0.034], P 0.827 | -0.045 [-0.116, +0.022], P 0.908 |


## faceverse_neutral: Spearman con la GT per gruppo (rho, IC 95%; 148500 righe all_cross, seme 566363)

GT FR:

| metodo | senza crop | all_cross | righe col crop | media 15 coppie | media crop 5 coppie |
| --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal. | 0.341 [0.267, 0.410] | 0.319 [0.243, 0.386] | 0.282 [0.207, 0.348] | 0.322 [0.245, 0.389] | 0.283 [0.207, 0.348] |
| factorized s1234, d_F cal. @ regione GNM | 0.334 [0.255, 0.406] | 0.342 [0.261, 0.413] | 0.359 [0.275, 0.429] | 0.343 [0.262, 0.414] | 0.359 [0.275, 0.429] |
| factorized s1234, d_F cal. @ regione FLAME | 0.314 [0.219, 0.400] | 0.315 [0.220, 0.403] | 0.317 [0.222, 0.404] | 0.315 [0.220, 0.403] | 0.317 [0.222, 0.404] |
| factorized s2345, d_F cal. | 0.370 [0.304, 0.435] | 0.335 [0.274, 0.395] | 0.272 [0.212, 0.329] | 0.338 [0.276, 0.400] | 0.273 [0.213, 0.329] |
| factorized s2345, d_F cal. @ regione GNM | 0.321 [0.235, 0.405] | 0.325 [0.242, 0.404] | 0.334 [0.253, 0.406] | 0.325 [0.242, 0.405] | 0.334 [0.254, 0.406] |
| factorized s2345, d_F cal. @ regione FLAME | 0.289 [0.198, 0.382] | 0.291 [0.201, 0.384] | 0.296 [0.207, 0.386] | 0.291 [0.201, 0.384] | 0.296 [0.207, 0.386] |
| ctrlfr s1234 | 0.327 [0.255, 0.398] | 0.316 [0.248, 0.387] | 0.296 [0.227, 0.362] | 0.329 [0.259, 0.400] | 0.306 [0.234, 0.374] |
| ctrlfr s1234 @ regione GNM | 0.336 [0.261, 0.406] | 0.343 [0.269, 0.409] | 0.356 [0.285, 0.417] | 0.351 [0.274, 0.417] | 0.363 [0.292, 0.425] |
| ctrlfr s1234 @ regione FLAME | 0.305 [0.217, 0.388] | 0.308 [0.221, 0.390] | 0.315 [0.227, 0.398] | 0.311 [0.223, 0.394] | 0.318 [0.229, 0.401] |
| ctrlfr s2345 | 0.299 [0.222, 0.379] | 0.289 [0.215, 0.363] | 0.269 [0.200, 0.341] | 0.295 [0.221, 0.371] | 0.273 [0.202, 0.345] |
| ctrlfr s2345 @ regione GNM | 0.309 [0.233, 0.383] | 0.316 [0.241, 0.386] | 0.329 [0.251, 0.398] | 0.322 [0.245, 0.393] | 0.334 [0.255, 0.404] |
| ctrlfr s2345 @ regione FLAME | 0.278 [0.187, 0.364] | 0.279 [0.188, 0.366] | 0.282 [0.188, 0.368] | 0.280 [0.188, 0.368] | 0.283 [0.189, 0.369] |
| C3M e205, d_F cal. | 0.336 [0.270, 0.399] | 0.324 [0.257, 0.385] | 0.299 [0.226, 0.360] | 0.346 [0.274, 0.411] | 0.313 [0.235, 0.377] |
| C3M e205, d_F cal. @ regione GNM | 0.294 [0.232, 0.355] | 0.302 [0.241, 0.364] | 0.320 [0.256, 0.384] | 0.317 [0.255, 0.379] | 0.332 [0.267, 0.397] |
| C3M e205, d_F cal. @ regione FLAME | 0.268 [0.185, 0.344] | 0.274 [0.193, 0.351] | 0.290 [0.204, 0.371] | 0.291 [0.204, 0.373] | 0.302 [0.213, 0.388] |
| factorized s1234, d_P | 0.321 [0.241, 0.394] | 0.310 [0.232, 0.384] | 0.293 [0.209, 0.368] | 0.312 [0.233, 0.386] | 0.293 [0.209, 0.368] |
| factorized s1234, d_P @ regione GNM | 0.337 [0.251, 0.411] | 0.345 [0.259, 0.420] | 0.361 [0.273, 0.440] | 0.346 [0.260, 0.421] | 0.361 [0.273, 0.441] |
| factorized s1234, d_P @ regione FLAME | 0.324 [0.232, 0.412] | 0.325 [0.233, 0.414] | 0.328 [0.236, 0.417] | 0.325 [0.233, 0.414] | 0.328 [0.236, 0.417] |
| factorized s2345, d_P | 0.361 [0.283, 0.443] | 0.341 [0.265, 0.418] | 0.306 [0.227, 0.381] | 0.343 [0.266, 0.419] | 0.306 [0.228, 0.382] |
| factorized s2345, d_P @ regione GNM | 0.336 [0.253, 0.417] | 0.339 [0.260, 0.415] | 0.346 [0.268, 0.421] | 0.340 [0.261, 0.416] | 0.346 [0.268, 0.421] |
| factorized s2345, d_P @ regione FLAME | 0.297 [0.198, 0.391] | 0.300 [0.202, 0.393] | 0.305 [0.207, 0.397] | 0.300 [0.202, 0.393] | 0.305 [0.207, 0.397] |
| C3M e205, d_P | 0.363 [0.282, 0.439] | 0.357 [0.277, 0.433] | 0.345 [0.265, 0.423] | 0.362 [0.282, 0.439] | 0.346 [0.266, 0.424] |
| C3M e205, d_P @ regione GNM | 0.330 [0.261, 0.393] | 0.334 [0.265, 0.397] | 0.342 [0.273, 0.408] | 0.339 [0.269, 0.403] | 0.345 [0.275, 0.411] |
| C3M e205, d_P @ regione FLAME | 0.301 [0.218, 0.386] | 0.306 [0.222, 0.390] | 0.315 [0.232, 0.402] | 0.313 [0.227, 0.400] | 0.321 [0.236, 0.408] |
| GNM (visto) vB, coefficienti | 0.299 [0.206, 0.386] | 0.297 [0.202, 0.383] | 0.292 [0.198, 0.377] | 0.298 [0.203, 0.385] | 0.293 [0.199, 0.378] |
| GNM (visto) vB, mesh d'identita' FR | 0.321 [0.228, 0.404] | 0.321 [0.229, 0.403] | 0.323 [0.234, 0.403] | 0.322 [0.229, 0.404] | 0.323 [0.234, 0.403] |
| GNM (visto) vB, mesh d'identita' SR | 0.373 [0.289, 0.451] | 0.375 [0.291, 0.454] | 0.379 [0.299, 0.455] | 0.375 [0.291, 0.455] | 0.379 [0.299, 0.456] |
| FLAME 2023 Open vB, coefficienti | 0.249 [0.154, 0.346] | 0.240 [0.145, 0.339] | 0.223 [0.127, 0.325] | 0.241 [0.147, 0.342] | 0.223 [0.128, 0.326] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.315 [0.226, 0.398] | 0.312 [0.224, 0.393] | 0.306 [0.219, 0.386] | 0.312 [0.224, 0.393] | 0.306 [0.219, 0.386] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.358 [0.269, 0.440] | 0.355 [0.267, 0.437] | 0.348 [0.259, 0.433] | 0.355 [0.267, 0.437] | 0.348 [0.259, 0.433] |

GT SR:

| metodo | senza crop | all_cross | righe col crop | media 15 coppie | media crop 5 coppie |
| --- | --- | --- | --- | --- | --- |
| factorized s1234, d_F cal. | 0.308 [0.226, 0.379] | 0.288 [0.211, 0.358] | 0.255 [0.176, 0.326] | 0.291 [0.213, 0.360] | 0.255 [0.176, 0.326] |
| factorized s1234, d_F cal. @ regione GNM | 0.318 [0.232, 0.397] | 0.326 [0.238, 0.403] | 0.341 [0.251, 0.416] | 0.326 [0.239, 0.404] | 0.341 [0.251, 0.416] |
| factorized s1234, d_F cal. @ regione FLAME | 0.300 [0.195, 0.398] | 0.301 [0.195, 0.397] | 0.302 [0.198, 0.398] | 0.301 [0.195, 0.397] | 0.303 [0.198, 0.398] |
| factorized s2345, d_F cal. | 0.336 [0.254, 0.411] | 0.303 [0.228, 0.372] | 0.247 [0.178, 0.306] | 0.306 [0.231, 0.375] | 0.247 [0.178, 0.306] |
| factorized s2345, d_F cal. @ regione GNM | 0.315 [0.222, 0.399] | 0.317 [0.226, 0.398] | 0.322 [0.234, 0.398] | 0.318 [0.226, 0.398] | 0.322 [0.234, 0.398] |
| factorized s2345, d_F cal. @ regione FLAME | 0.284 [0.188, 0.378] | 0.286 [0.191, 0.379] | 0.290 [0.196, 0.380] | 0.286 [0.191, 0.379] | 0.290 [0.196, 0.380] |
| ctrlfr s1234 | 0.326 [0.251, 0.398] | 0.314 [0.240, 0.382] | 0.291 [0.214, 0.363] | 0.326 [0.250, 0.398] | 0.301 [0.222, 0.374] |
| ctrlfr s1234 @ regione GNM | 0.339 [0.262, 0.410] | 0.346 [0.269, 0.414] | 0.359 [0.282, 0.424] | 0.353 [0.274, 0.422] | 0.365 [0.287, 0.432] |
| ctrlfr s1234 @ regione FLAME | 0.314 [0.224, 0.398] | 0.317 [0.227, 0.401] | 0.323 [0.232, 0.407] | 0.320 [0.230, 0.405] | 0.325 [0.234, 0.410] |
| ctrlfr s2345 | 0.305 [0.229, 0.381] | 0.294 [0.221, 0.367] | 0.274 [0.200, 0.346] | 0.301 [0.227, 0.375] | 0.278 [0.203, 0.350] |
| ctrlfr s2345 @ regione GNM | 0.318 [0.243, 0.389] | 0.324 [0.248, 0.393] | 0.335 [0.257, 0.404] | 0.330 [0.252, 0.401] | 0.340 [0.262, 0.410] |
| ctrlfr s2345 @ regione FLAME | 0.291 [0.196, 0.379] | 0.293 [0.198, 0.381] | 0.296 [0.202, 0.382] | 0.294 [0.199, 0.382] | 0.297 [0.202, 0.383] |
| C3M e205, d_F cal. | 0.329 [0.260, 0.389] | 0.318 [0.249, 0.376] | 0.297 [0.228, 0.356] | 0.339 [0.266, 0.401] | 0.310 [0.238, 0.373] |
| C3M e205, d_F cal. @ regione GNM | 0.301 [0.242, 0.363] | 0.311 [0.252, 0.372] | 0.331 [0.268, 0.392] | 0.326 [0.265, 0.388] | 0.342 [0.276, 0.406] |
| C3M e205, d_F cal. @ regione FLAME | 0.270 [0.180, 0.354] | 0.277 [0.187, 0.361] | 0.293 [0.198, 0.379] | 0.293 [0.198, 0.382] | 0.305 [0.208, 0.394] |
| factorized s1234, d_P | 0.333 [0.262, 0.404] | 0.323 [0.254, 0.391] | 0.304 [0.231, 0.376] | 0.324 [0.255, 0.392] | 0.305 [0.231, 0.376] |
| factorized s1234, d_P @ regione GNM | 0.351 [0.266, 0.424] | 0.360 [0.275, 0.433] | 0.378 [0.293, 0.449] | 0.361 [0.276, 0.434] | 0.378 [0.293, 0.450] |
| factorized s1234, d_P @ regione FLAME | 0.344 [0.252, 0.435] | 0.346 [0.254, 0.436] | 0.349 [0.258, 0.439] | 0.346 [0.255, 0.437] | 0.349 [0.258, 0.440] |
| factorized s2345, d_P | 0.372 [0.303, 0.443] | 0.352 [0.283, 0.423] | 0.316 [0.245, 0.389] | 0.353 [0.284, 0.425] | 0.317 [0.245, 0.390] |
| factorized s2345, d_P @ regione GNM | 0.352 [0.261, 0.433] | 0.354 [0.268, 0.432] | 0.360 [0.276, 0.433] | 0.355 [0.268, 0.433] | 0.360 [0.276, 0.433] |
| factorized s2345, d_P @ regione FLAME | 0.319 [0.218, 0.411] | 0.321 [0.223, 0.412] | 0.325 [0.229, 0.415] | 0.321 [0.223, 0.412] | 0.325 [0.229, 0.415] |
| C3M e205, d_P | 0.382 [0.306, 0.448] | 0.378 [0.304, 0.446] | 0.372 [0.301, 0.439] | 0.384 [0.309, 0.452] | 0.373 [0.302, 0.439] |
| C3M e205, d_P @ regione GNM | 0.350 [0.282, 0.416] | 0.356 [0.289, 0.422] | 0.369 [0.299, 0.434] | 0.360 [0.293, 0.426] | 0.371 [0.302, 0.437] |
| C3M e205, d_P @ regione FLAME | 0.324 [0.236, 0.415] | 0.330 [0.241, 0.421] | 0.341 [0.254, 0.432] | 0.337 [0.245, 0.429] | 0.346 [0.258, 0.438] |
| GNM (visto) vB, coefficienti | 0.304 [0.207, 0.391] | 0.301 [0.206, 0.390] | 0.296 [0.199, 0.384] | 0.303 [0.207, 0.392] | 0.297 [0.200, 0.385] |
| GNM (visto) vB, mesh d'identita' FR | 0.275 [0.168, 0.373] | 0.275 [0.170, 0.372] | 0.276 [0.176, 0.371] | 0.275 [0.170, 0.373] | 0.276 [0.176, 0.372] |
| GNM (visto) vB, mesh d'identita' SR | 0.399 [0.310, 0.474] | 0.400 [0.312, 0.475] | 0.402 [0.317, 0.478] | 0.400 [0.312, 0.475] | 0.402 [0.317, 0.478] |
| FLAME 2023 Open vB, coefficienti | 0.268 [0.169, 0.364] | 0.258 [0.160, 0.355] | 0.238 [0.139, 0.338] | 0.259 [0.161, 0.358] | 0.239 [0.140, 0.339] |
| FLAME 2023 Open vB, mesh d'identita' FR | 0.274 [0.170, 0.370] | 0.271 [0.171, 0.366] | 0.265 [0.167, 0.356] | 0.271 [0.171, 0.366] | 0.265 [0.167, 0.356] |
| FLAME 2023 Open vB, mesh d'identita' SR | 0.382 [0.288, 0.464] | 0.378 [0.283, 0.461] | 0.371 [0.275, 0.457] | 0.378 [0.284, 0.462] | 0.371 [0.276, 0.457] |

Delta appaiati, righe col crop (braccio sulla regione di m - concorrente; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.067 [-0.037, +0.168], P 0.104 | +0.042 [-0.064, +0.148], P 0.209 | +0.064 [-0.043, +0.165], P 0.123 | +0.037 [-0.063, +0.145], P 0.248 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.036 [-0.049, +0.117], P 0.210 | +0.011 [-0.071, +0.094], P 0.403 | +0.033 [-0.057, +0.127], P 0.253 | +0.006 [-0.089, +0.106], P 0.450 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.020 [-0.112, +0.062], P 0.689 | -0.045 [-0.138, +0.041], P 0.852 | -0.023 [-0.111, +0.060], P 0.719 | -0.050 [-0.143, +0.053], P 0.848 |
| braccio intero (@ regione GNM) | +0.076 [+0.022, +0.130], P 0.003 | +0.061 [-0.000, +0.124], P 0.027 | +0.060 [+0.000, +0.119], P 0.023 | +0.060 [-0.011, +0.125], P 0.042 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.094 [-0.028, +0.205], P 0.064 | +0.073 [-0.046, +0.183], P 0.117 | +0.092 [-0.027, +0.202], P 0.077 | +0.059 [-0.058, +0.170], P 0.181 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.011 [-0.087, +0.100], P 0.404 | -0.010 [-0.104, +0.075], P 0.590 | +0.009 [-0.101, +0.105], P 0.427 | -0.024 [-0.133, +0.080], P 0.669 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.031 [-0.137, +0.074], P 0.738 | -0.053 [-0.155, +0.050], P 0.859 | -0.033 [-0.139, +0.070], P 0.744 | -0.067 [-0.168, +0.035], P 0.893 |
| braccio intero (@ regione FLAME) | +0.034 [-0.042, +0.115], P 0.205 | +0.023 [-0.062, +0.103], P 0.290 | +0.019 [-0.057, +0.096], P 0.317 | +0.012 [-0.066, +0.092], P 0.381 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.082 [-0.024, +0.185], P 0.052 | +0.064 [-0.042, +0.167], P 0.122 | +0.063 [-0.041, +0.164], P 0.141 | +0.039 [-0.061, +0.145], P 0.242 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.102 [-0.011, +0.219], P 0.039 | +0.084 [-0.018, +0.193], P 0.061 | +0.083 [-0.009, +0.174], P 0.044 | +0.059 [-0.034, +0.156], P 0.114 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.024 [-0.116, +0.064], P 0.724 | -0.042 [-0.126, +0.042], P 0.830 | -0.043 [-0.130, +0.046], P 0.824 | -0.067 [-0.161, +0.031], P 0.917 |
| braccio intero (@ regione GNM) | +0.073 [+0.012, +0.138], P 0.011 | +0.044 [-0.027, +0.116], P 0.134 | +0.068 [+0.005, +0.125], P 0.015 | +0.060 [-0.011, +0.127], P 0.044 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.111 [-0.013, +0.227], P 0.041 | +0.087 [-0.041, +0.199], P 0.087 | +0.085 [-0.038, +0.192], P 0.095 | +0.057 [-0.059, +0.166], P 0.177 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.084 [-0.031, +0.191], P 0.090 | +0.061 [-0.051, +0.171], P 0.158 | +0.058 [-0.051, +0.158], P 0.159 | +0.031 [-0.069, +0.134], P 0.282 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.022 [-0.115, +0.078], P 0.670 | -0.045 [-0.142, +0.056], P 0.818 | -0.048 [-0.145, +0.055], P 0.840 | -0.075 [-0.172, +0.028], P 0.930 |
| braccio intero (@ regione FLAME) | +0.045 [-0.036, +0.124], P 0.158 | +0.009 [-0.085, +0.095], P 0.430 | +0.032 [-0.045, +0.107], P 0.231 | +0.021 [-0.058, +0.101], P 0.298 |

Delta appaiati, media crop 5 coppie (braccio sulla regione di m - concorrente; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.066 [-0.038, +0.167], P 0.108 | +0.041 [-0.066, +0.147], P 0.218 | +0.070 [-0.036, +0.171], P 0.106 | +0.041 [-0.059, +0.150], P 0.231 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.036 [-0.049, +0.118], P 0.209 | +0.011 [-0.071, +0.094], P 0.403 | +0.040 [-0.050, +0.134], P 0.190 | +0.011 [-0.084, +0.112], P 0.412 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.020 [-0.112, +0.062], P 0.689 | -0.045 [-0.138, +0.041], P 0.853 | -0.016 [-0.105, +0.067], P 0.666 | -0.045 [-0.139, +0.059], P 0.820 |
| braccio intero (@ regione GNM) | +0.076 [+0.022, +0.130], P 0.003 | +0.061 [-0.001, +0.124], P 0.027 | +0.057 [-0.004, +0.118], P 0.035 | +0.061 [-0.010, +0.129], P 0.041 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.094 [-0.029, +0.205], P 0.065 | +0.072 [-0.048, +0.182], P 0.118 | +0.094 [-0.027, +0.204], P 0.075 | +0.059 [-0.059, +0.171], P 0.181 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.011 [-0.087, +0.100], P 0.404 | -0.010 [-0.104, +0.075], P 0.591 | +0.011 [-0.099, +0.109], P 0.413 | -0.023 [-0.132, +0.081], P 0.659 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.031 [-0.137, +0.074], P 0.738 | -0.053 [-0.155, +0.050], P 0.859 | -0.031 [-0.137, +0.072], P 0.730 | -0.066 [-0.168, +0.036], P 0.888 |
| braccio intero (@ regione FLAME) | +0.034 [-0.042, +0.115], P 0.205 | +0.023 [-0.062, +0.103], P 0.296 | +0.012 [-0.065, +0.090], P 0.397 | +0.010 [-0.069, +0.089], P 0.404 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.081 [-0.025, +0.185], P 0.055 | +0.063 [-0.043, +0.167], P 0.127 | +0.068 [-0.037, +0.171], P 0.116 | +0.043 [-0.058, +0.151], P 0.224 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.102 [-0.011, +0.219], P 0.039 | +0.084 [-0.018, +0.193], P 0.061 | +0.090 [-0.003, +0.181], P 0.031 | +0.064 [-0.028, +0.163], P 0.103 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.024 [-0.116, +0.064], P 0.723 | -0.042 [-0.127, +0.042], P 0.832 | -0.037 [-0.124, +0.052], P 0.797 | -0.062 [-0.158, +0.036], P 0.894 |
| braccio intero (@ regione GNM) | +0.074 [+0.012, +0.138], P 0.011 | +0.043 [-0.028, +0.116], P 0.136 | +0.065 [+0.000, +0.123], P 0.025 | +0.062 [-0.010, +0.130], P 0.042 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.110 [-0.014, +0.226], P 0.042 | +0.086 [-0.042, +0.199], P 0.092 | +0.086 [-0.037, +0.194], P 0.093 | +0.057 [-0.060, +0.166], P 0.178 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.084 [-0.031, +0.191], P 0.088 | +0.061 [-0.051, +0.171], P 0.158 | +0.061 [-0.050, +0.162], P 0.146 | +0.032 [-0.068, +0.135], P 0.277 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.022 [-0.115, +0.078], P 0.670 | -0.045 [-0.142, +0.056], P 0.819 | -0.045 [-0.143, +0.059], P 0.824 | -0.074 [-0.171, +0.029], P 0.930 |
| braccio intero (@ regione FLAME) | +0.044 [-0.036, +0.124], P 0.158 | +0.009 [-0.086, +0.095], P 0.442 | +0.025 [-0.053, +0.101], P 0.289 | +0.019 [-0.060, +0.099], P 0.327 |

Delta appaiati, senza crop (braccio sulla regione di m - concorrente; IC 95%, P(delta <= 0)):

GT FR:

| concorrente | factorized s1234, d_F cal. | factorized s2345, d_F cal. | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.035 [-0.068, +0.135], P 0.249 | +0.021 [-0.083, +0.124], P 0.337 | +0.037 [-0.073, +0.140], P 0.238 | +0.010 [-0.091, +0.113], P 0.412 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.014 [-0.068, +0.093], P 0.368 | -0.000 [-0.082, +0.087], P 0.508 | +0.015 [-0.078, +0.107], P 0.388 | -0.011 [-0.107, +0.089], P 0.579 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.038 [-0.130, +0.044], P 0.818 | -0.052 [-0.151, +0.036], P 0.877 | -0.037 [-0.121, +0.052], P 0.801 | -0.064 [-0.155, +0.040], P 0.907 |
| braccio intero (@ regione GNM) | -0.006 [-0.072, +0.061], P 0.597 | -0.050 [-0.136, +0.032], P 0.898 | +0.009 [-0.058, +0.078], P 0.401 | +0.010 [-0.062, +0.079], P 0.407 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.065 [-0.049, +0.171], P 0.133 | +0.040 [-0.080, +0.149], P 0.248 | +0.056 [-0.064, +0.160], P 0.192 | +0.029 [-0.086, +0.135], P 0.311 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | -0.001 [-0.097, +0.086], P 0.505 | -0.026 [-0.117, +0.062], P 0.709 | -0.010 [-0.121, +0.087], P 0.564 | -0.037 [-0.138, +0.064], P 0.747 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.045 [-0.140, +0.054], P 0.816 | -0.070 [-0.166, +0.031], P 0.925 | -0.054 [-0.150, +0.045], P 0.877 | -0.081 [-0.177, +0.020], P 0.941 |
| braccio intero (@ regione FLAME) | -0.027 [-0.108, +0.057], P 0.743 | -0.082 [-0.180, +0.007], P 0.968 | -0.023 [-0.103, +0.056], P 0.706 | -0.021 [-0.104, +0.060], P 0.704 |

GT SR:

| concorrente | factorized s1234, d_P | factorized s2345, d_P | ctrlfr s1234 | ctrlfr s2345 |
| --- | --- | --- | --- | --- |
| GNM (visto) vB, coefficienti (@ regione GNM) | +0.048 [-0.054, +0.144], P 0.174 | +0.048 [-0.058, +0.148], P 0.184 | +0.036 [-0.071, +0.140], P 0.256 | +0.015 [-0.083, +0.115], P 0.374 |
| GNM (visto) vB, mesh d'identita' FR (@ regione GNM) | +0.077 [-0.041, +0.198], P 0.084 | +0.077 [-0.029, +0.195], P 0.091 | +0.065 [-0.032, +0.156], P 0.086 | +0.044 [-0.048, +0.142], P 0.187 |
| GNM (visto) vB, mesh d'identita' SR (@ regione GNM) | -0.047 [-0.138, +0.035], P 0.869 | -0.047 [-0.132, +0.038], P 0.845 | -0.059 [-0.146, +0.035], P 0.896 | -0.080 [-0.172, +0.020], P 0.952 |
| braccio intero (@ regione GNM) | +0.018 [-0.047, +0.084], P 0.312 | -0.020 [-0.104, +0.059], P 0.688 | +0.013 [-0.053, +0.084], P 0.356 | +0.013 [-0.058, +0.082], P 0.375 |
| FLAME 2023 Open vB, coefficienti (@ regione FLAME) | +0.077 [-0.045, +0.189], P 0.105 | +0.051 [-0.072, +0.166], P 0.207 | +0.047 [-0.069, +0.153], P 0.233 | +0.024 [-0.090, +0.128], P 0.342 |
| FLAME 2023 Open vB, mesh d'identita' FR (@ regione FLAME) | +0.070 [-0.043, +0.180], P 0.123 | +0.044 [-0.067, +0.155], P 0.227 | +0.040 [-0.071, +0.136], P 0.238 | +0.017 [-0.084, +0.114], P 0.373 |
| FLAME 2023 Open vB, mesh d'identita' SR (@ regione FLAME) | -0.038 [-0.127, +0.056], P 0.811 | -0.063 [-0.155, +0.032], P 0.913 | -0.068 [-0.157, +0.029], P 0.923 | -0.091 [-0.181, +0.011], P 0.963 |
| braccio intero (@ regione FLAME) | +0.011 [-0.067, +0.087], P 0.419 | -0.053 [-0.156, +0.041], P 0.867 | -0.012 [-0.090, +0.071], P 0.616 | -0.014 [-0.100, +0.065], P 0.647 |


## Esperimento 2: crop sugli held-out sintetici del training

Per dominio, coppie di soggetti diversi: senza crop (etichette diverse, entrambe senza crop) e col crop (un lato crop); Spearman con GT-FR calibrata (`gt_frcal`) e GT-SR (`gt_sr`); AUC di verifica (genuine = stesso soggetto). BFM: FR non si legge (taglia delle original REMESH allineate per similarita').

| dominio | metodo | misura | senza crop | col crop | col crop - senza crop |
| --- | --- | --- | --- | --- | --- |
| bfm | ctrlfr s1234 | rho_fr | 0.899 [0.877, 0.918] | 0.890 [0.865, 0.909] | -0.009 [-0.022, +0.003] |
| bfm | ctrlfr s1234 | rho_sr | 0.880 [0.853, 0.903] | 0.862 [0.833, 0.885] | -0.017 [-0.028, -0.007] |
| bfm | ctrlfr s1234 | auc | 1.000 [1.000, 1.000] | 0.999 [0.997, 1.000] | -0.001 [-0.003, -0.000] |
| bfm | ctrlfr s2345 | rho_fr | 0.898 [0.875, 0.915] | 0.879 [0.852, 0.900] | -0.019 [-0.034, -0.004] |
| bfm | ctrlfr s2345 | rho_sr | 0.870 [0.835, 0.896] | 0.845 [0.809, 0.872] | -0.024 [-0.040, -0.010] |
| bfm | ctrlfr s2345 | auc | 1.000 [1.000, 1.000] | 0.997 [0.992, 1.000] | -0.003 [-0.008, -0.000] |
| bfm | factorized s1234, d_F cal. | rho_fr | 0.825 [0.781, 0.866] | 0.731 [0.677, 0.775] | -0.094 [-0.116, -0.075] |
| bfm | factorized s1234, d_F cal. | rho_sr | 0.859 [0.822, 0.890] | 0.763 [0.714, 0.800] | -0.096 [-0.117, -0.078] |
| bfm | factorized s1234, d_F cal. | auc | 1.000 [1.000, 1.000] | 0.992 [0.982, 0.998] | -0.008 [-0.018, -0.002] |
| bfm | factorized s1234, d_P | rho_fr | 0.843 [0.801, 0.878] | 0.832 [0.794, 0.863] | -0.012 [-0.024, +0.001] |
| bfm | factorized s1234, d_P | rho_sr | 0.902 [0.878, 0.922] | 0.883 [0.857, 0.903] | -0.019 [-0.031, -0.008] |
| bfm | factorized s1234, d_P | auc | 1.000 [1.000, 1.000] | 0.998 [0.995, 1.000] | -0.002 [-0.005, -0.000] |
| bfm | factorized s2345, d_F cal. | rho_fr | 0.818 [0.771, 0.860] | 0.749 [0.692, 0.799] | -0.068 [-0.089, -0.050] |
| bfm | factorized s2345, d_F cal. | rho_sr | 0.862 [0.823, 0.894] | 0.787 [0.735, 0.828] | -0.075 [-0.095, -0.058] |
| bfm | factorized s2345, d_F cal. | auc | 1.000 [1.000, 1.000] | 0.993 [0.981, 0.999] | -0.007 [-0.019, -0.001] |
| bfm | factorized s2345, d_P | rho_fr | 0.843 [0.801, 0.879] | 0.826 [0.784, 0.862] | -0.018 [-0.035, -0.003] |
| bfm | factorized s2345, d_P | rho_sr | 0.905 [0.881, 0.927] | 0.879 [0.849, 0.902] | -0.026 [-0.041, -0.015] |
| bfm | factorized s2345, d_P | auc | 1.000 [1.000, 1.000] | 0.998 [0.995, 1.000] | -0.002 [-0.005, -0.000] |
| bfm | C3M e123, d_F cal. | rho_fr | 0.828 [0.786, 0.862] | 0.572 [0.510, 0.627] | -0.256 [-0.294, -0.219] |
| bfm | C3M e123, d_F cal. | rho_sr | 0.845 [0.807, 0.876] | 0.606 [0.546, 0.658] | -0.239 [-0.278, -0.206] |
| bfm | C3M e123, d_F cal. | auc | 1.000 [1.000, 1.000] | 0.940 [0.922, 0.957] | -0.060 [-0.078, -0.043] |
| bfm | C3M e123, d_P | rho_fr | 0.887 [0.854, 0.916] | 0.879 [0.847, 0.906] | -0.008 [-0.017, -0.000] |
| bfm | C3M e123, d_P | rho_sr | 0.949 [0.935, 0.959] | 0.938 [0.923, 0.949] | -0.010 [-0.017, -0.005] |
| bfm | C3M e123, d_P | auc | 1.000 [1.000, 1.000] | 0.998 [0.994, 1.000] | -0.002 [-0.006, -0.000] |
| bfm | C3M e205, d_F cal. | rho_fr | 0.847 [0.810, 0.878] | 0.671 [0.615, 0.719] | -0.176 [-0.208, -0.146] |
| bfm | C3M e205, d_F cal. | rho_sr | 0.872 [0.839, 0.897] | 0.706 [0.653, 0.751] | -0.167 [-0.199, -0.138] |
| bfm | C3M e205, d_F cal. | auc | 1.000 [1.000, 1.000] | 0.970 [0.958, 0.979] | -0.030 [-0.042, -0.021] |
| bfm | C3M e205, d_P | rho_fr | 0.893 [0.861, 0.921] | 0.886 [0.856, 0.912] | -0.006 [-0.015, +0.002] |
| bfm | C3M e205, d_P | rho_sr | 0.955 [0.942, 0.965] | 0.948 [0.935, 0.957] | -0.007 [-0.014, -0.002] |
| bfm | C3M e205, d_P | auc | 1.000 [1.000, 1.000] | 0.999 [0.998, 1.000] | -0.001 [-0.002, -0.000] |
| ict | ctrlfr s1234 | rho_fr | 0.965 [0.950, 0.976] | 0.960 [0.941, 0.972] | -0.006 [-0.011, -0.002] |
| ict | ctrlfr s1234 | rho_sr | 0.568 [0.463, 0.654] | 0.573 [0.466, 0.658] | +0.005 [-0.005, +0.015] |
| ict | ctrlfr s1234 | auc | 1.000 [1.000, 1.000] | 0.999 [0.999, 1.000] | -0.001 [-0.001, -0.000] |
| ict | ctrlfr s2345 | rho_fr | 0.966 [0.951, 0.976] | 0.958 [0.941, 0.971] | -0.007 [-0.012, -0.004] |
| ict | ctrlfr s2345 | rho_sr | 0.578 [0.470, 0.664] | 0.569 [0.462, 0.657] | -0.009 [-0.020, +0.002] |
| ict | ctrlfr s2345 | auc | 1.000 [1.000, 1.000] | 0.999 [0.998, 1.000] | -0.001 [-0.002, -0.000] |
| ict | factorized s1234, d_F cal. | rho_fr | 0.895 [0.859, 0.920] | 0.887 [0.849, 0.915] | -0.008 [-0.018, +0.002] |
| ict | factorized s1234, d_F cal. | rho_sr | 0.610 [0.507, 0.698] | 0.605 [0.498, 0.691] | -0.005 [-0.017, +0.007] |
| ict | factorized s1234, d_F cal. | auc | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | -0.000 [-0.000, -0.000] |
| ict | factorized s1234, d_P | rho_fr | 0.578 [0.474, 0.663] | 0.570 [0.464, 0.658] | -0.008 [-0.021, +0.004] |
| ict | factorized s1234, d_P | rho_sr | 0.930 [0.905, 0.947] | 0.920 [0.894, 0.940] | -0.010 [-0.018, -0.002] |
| ict | factorized s1234, d_P | auc | 1.000 [1.000, 1.000] | 0.999 [0.998, 1.000] | -0.001 [-0.002, -0.000] |
| ict | factorized s2345, d_F cal. | rho_fr | 0.904 [0.874, 0.925] | 0.883 [0.845, 0.911] | -0.021 [-0.033, -0.011] |
| ict | factorized s2345, d_F cal. | rho_sr | 0.603 [0.499, 0.692] | 0.586 [0.481, 0.675] | -0.017 [-0.032, -0.003] |
| ict | factorized s2345, d_F cal. | auc | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | -0.000 [-0.000, -0.000] |
| ict | factorized s2345, d_P | rho_fr | 0.582 [0.475, 0.671] | 0.576 [0.465, 0.664] | -0.006 [-0.018, +0.004] |
| ict | factorized s2345, d_P | rho_sr | 0.934 [0.912, 0.949] | 0.927 [0.902, 0.945] | -0.007 [-0.015, -0.000] |
| ict | factorized s2345, d_P | auc | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | -0.000 [-0.001, -0.000] |
| ict | C3M e123, d_F cal. | rho_fr | 0.961 [0.945, 0.972] | 0.938 [0.916, 0.955] | -0.023 [-0.035, -0.013] |
| ict | C3M e123, d_F cal. | rho_sr | 0.635 [0.532, 0.714] | 0.616 [0.511, 0.697] | -0.019 [-0.040, -0.000] |
| ict | C3M e123, d_F cal. | auc | 1.000 [1.000, 1.000] | 0.999 [0.998, 1.000] | -0.001 [-0.002, -0.000] |
| ict | C3M e123, d_P | rho_fr | 0.616 [0.517, 0.697] | 0.605 [0.505, 0.686] | -0.011 [-0.024, -0.002] |
| ict | C3M e123, d_P | rho_sr | 0.964 [0.950, 0.974] | 0.957 [0.942, 0.968] | -0.007 [-0.013, -0.003] |
| ict | C3M e123, d_P | auc | 1.000 [1.000, 1.000] | 0.999 [0.999, 1.000] | -0.001 [-0.001, -0.000] |
| ict | C3M e205, d_F cal. | rho_fr | 0.964 [0.949, 0.974] | 0.948 [0.929, 0.962] | -0.016 [-0.025, -0.008] |
| ict | C3M e205, d_F cal. | rho_sr | 0.637 [0.533, 0.718] | 0.611 [0.505, 0.694] | -0.026 [-0.047, -0.007] |
| ict | C3M e205, d_F cal. | auc | 1.000 [1.000, 1.000] | 0.999 [0.999, 1.000] | -0.001 [-0.001, -0.000] |
| ict | C3M e205, d_P | rho_fr | 0.610 [0.506, 0.693] | 0.605 [0.505, 0.689] | -0.004 [-0.015, +0.006] |
| ict | C3M e205, d_P | rho_sr | 0.969 [0.956, 0.978] | 0.965 [0.951, 0.974] | -0.005 [-0.009, -0.001] |
| ict | C3M e205, d_P | auc | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | -0.000 [-0.000, -0.000] |
| gnm | ctrlfr s1234 | rho_fr | 0.946 [0.929, 0.958] | 0.944 [0.926, 0.957] | -0.002 [-0.006, +0.002] |
| gnm | ctrlfr s1234 | rho_sr | 0.572 [0.465, 0.666] | 0.581 [0.478, 0.670] | +0.008 [-0.000, +0.017] |
| gnm | ctrlfr s1234 | auc | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | +0.000 [-0.000, +0.000] |
| gnm | ctrlfr s2345 | rho_fr | 0.938 [0.919, 0.952] | 0.940 [0.921, 0.953] | +0.001 [-0.004, +0.006] |
| gnm | ctrlfr s2345 | rho_sr | 0.562 [0.455, 0.658] | 0.557 [0.455, 0.650] | -0.005 [-0.015, +0.005] |
| gnm | ctrlfr s2345 | auc | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | -0.000 [-0.000, -0.000] |
| gnm | factorized s1234, d_F cal. | rho_fr | 0.857 [0.819, 0.890] | 0.850 [0.812, 0.883] | -0.007 [-0.016, +0.001] |
| gnm | factorized s1234, d_F cal. | rho_sr | 0.573 [0.468, 0.670] | 0.586 [0.479, 0.679] | +0.013 [+0.002, +0.023] |
| gnm | factorized s1234, d_F cal. | auc | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | -0.000 [-0.000, -0.000] |
| gnm | factorized s1234, d_P | rho_fr | 0.540 [0.445, 0.630] | 0.528 [0.429, 0.622] | -0.011 [-0.027, +0.003] |
| gnm | factorized s1234, d_P | rho_sr | 0.910 [0.886, 0.929] | 0.907 [0.883, 0.927] | -0.003 [-0.012, +0.005] |
| gnm | factorized s1234, d_P | auc | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | -0.000 [-0.001, -0.000] |
| gnm | factorized s2345, d_F cal. | rho_fr | 0.839 [0.796, 0.877] | 0.827 [0.782, 0.864] | -0.012 [-0.021, -0.005] |
| gnm | factorized s2345, d_F cal. | rho_sr | 0.552 [0.446, 0.649] | 0.553 [0.446, 0.647] | +0.002 [-0.008, +0.011] |
| gnm | factorized s2345, d_F cal. | auc | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | -0.000 [-0.001, -0.000] |
| gnm | factorized s2345, d_P | rho_fr | 0.529 [0.435, 0.621] | 0.527 [0.430, 0.622] | -0.002 [-0.016, +0.011] |
| gnm | factorized s2345, d_P | rho_sr | 0.902 [0.875, 0.923] | 0.902 [0.877, 0.921] | +0.000 [-0.007, +0.008] |
| gnm | factorized s2345, d_P | auc | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | -0.000 [-0.001, -0.000] |
| gnm | C3M e123, d_F cal. | rho_fr | 0.939 [0.919, 0.954] | 0.930 [0.908, 0.946] | -0.009 [-0.016, -0.002] |
| gnm | C3M e123, d_F cal. | rho_sr | 0.574 [0.461, 0.673] | 0.567 [0.455, 0.664] | -0.008 [-0.020, +0.003] |
| gnm | C3M e123, d_F cal. | auc | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | -0.000 [-0.001, -0.000] |
| gnm | C3M e123, d_P | rho_fr | 0.571 [0.479, 0.656] | 0.555 [0.459, 0.647] | -0.016 [-0.028, -0.005] |
| gnm | C3M e123, d_P | rho_sr | 0.953 [0.941, 0.963] | 0.950 [0.936, 0.960] | -0.003 [-0.008, +0.000] |
| gnm | C3M e123, d_P | auc | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | -0.000 [-0.000, -0.000] |
| gnm | C3M e205, d_F cal. | rho_fr | 0.953 [0.935, 0.965] | 0.950 [0.932, 0.962] | -0.003 [-0.009, +0.003] |
| gnm | C3M e205, d_F cal. | rho_sr | 0.582 [0.472, 0.680] | 0.576 [0.464, 0.672] | -0.007 [-0.017, +0.003] |
| gnm | C3M e205, d_F cal. | auc | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | -0.000 [-0.001, +0.000] |
| gnm | C3M e205, d_P | rho_fr | 0.575 [0.482, 0.661] | 0.570 [0.474, 0.661] | -0.005 [-0.014, +0.004] |
| gnm | C3M e205, d_P | rho_sr | 0.955 [0.942, 0.965] | 0.955 [0.942, 0.965] | -0.000 [-0.004, +0.004] |
| gnm | C3M e205, d_P | auc | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | -0.000 [-0.000, -0.000] |

Spostamento del crop (held-out): d log S = log S(crop) - media delle 5 senza crop (media [IC 95% per soggetto], sd), sd di log S fra soggetti; distanze nell'embedding (||u|| o ||z||).

| dominio | braccio | d log S | sd | sd fra soggetti | d log S / sd fra soggetti | stesso sogg. crop | stesso sogg. senza crop | soggetti diversi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bfm | factorized_s1234 | -0.023 [-0.024, -0.022] | 0.0073 | 0.0181 | -1.27 | 0.282 | 0.146 | 0.693 |
| bfm | factorized_s2345 | -0.019 [-0.020, -0.017] | 0.0084 | 0.0181 | -1.02 | 0.280 | 0.132 | 0.722 |
| bfm | ctrlfr_s1234 | - | - | - | - | 0.371 | 0.228 | 0.919 |
| bfm | ctrlfr_s2345 | - | - | - | - | 0.425 | 0.257 | 0.994 |
| bfm | factorizedc3m_e123 | -0.046 [-0.048, -0.044] | 0.0124 | 0.0236 | -1.95 | 0.420 | 0.284 | 1.133 |
| bfm | factorizedc3m_e205 | -0.038 [-0.040, -0.036] | 0.0112 | 0.0203 | -1.87 | 0.448 | 0.325 | 1.311 |
| ict | factorized_s1234 | +0.001 [-0.001, +0.002] | 0.0080 | 0.0495 | +0.02 | 0.280 | 0.200 | 0.688 |
| ict | factorized_s2345 | +0.006 [+0.004, +0.008] | 0.0093 | 0.0498 | +0.12 | 0.269 | 0.166 | 0.684 |
| ict | ctrlfr_s1234 | - | - | - | - | 0.447 | 0.345 | 1.250 |
| ict | ctrlfr_s2345 | - | - | - | - | 0.478 | 0.323 | 1.343 |
| ict | factorizedc3m_e123 | -0.000 [-0.003, +0.002] | 0.0122 | 0.0489 | -0.00 | 0.424 | 0.308 | 1.086 |
| ict | factorizedc3m_e205 | +0.001 [-0.001, +0.003] | 0.0116 | 0.0489 | +0.02 | 0.454 | 0.327 | 1.243 |
| gnm | factorized_s1234 | -0.002 [-0.003, -0.001] | 0.0055 | 0.0508 | -0.04 | 0.281 | 0.174 | 0.750 |
| gnm | factorized_s2345 | -0.004 [-0.005, -0.002] | 0.0066 | 0.0523 | -0.07 | 0.255 | 0.131 | 0.735 |
| gnm | ctrlfr_s1234 | - | - | - | - | 0.373 | 0.262 | 1.153 |
| gnm | ctrlfr_s2345 | - | - | - | - | 0.410 | 0.266 | 1.233 |
| gnm | factorizedc3m_e123 | -0.005 [-0.006, -0.003] | 0.0073 | 0.0502 | -0.09 | 0.402 | 0.279 | 1.144 |
| gnm | factorizedc3m_e205 | -0.002 [-0.004, -0.001] | 0.0076 | 0.0502 | -0.04 | 0.445 | 0.332 | 1.304 |

## Controlli dell'emendamento 3

```
{
 "topo": {
  "hifi3d": {
   "seed": 757683,
   "rows_cross": 99000,
   "rows_same": 24750,
   "rows_kept": 123750,
   "groups": {
    "down8k|down8k": 4950,
    "down8k|noisy": 4950,
    "down8k|original": 4950,
    "down8k|remesh": 4950,
    "down8k|up60k": 4950,
    "noisy|down8k": 4950,
    "noisy|noisy": 4950,
    "noisy|original": 4950,
    "noisy|remesh": 4950,
    "noisy|up60k": 4950,
    "original|down8k": 4950,
    "original|noisy": 4950,
    "original|original": 4950,
    "original|remesh": 4950,
    "original|up60k": 4950,
    "remesh|down8k": 4950,
    "remesh|noisy": 4950,
    "remesh|original": 4950,
    "remesh|remesh": 4950,
    "remesh|up60k": 4950,
    "up60k|down8k": 4950,
    "up60k|noisy": 4950,
    "up60k|original": 4950,
    "up60k|remesh": 4950,
    "up60k|up60k": 4950
   },
   "gt_spread_within_subject_pair": 0.0
  },
  "facescape": {
   "seed": 621096,
   "rows_cross": 99000,
   "rows_same": 24750,
   "rows_kept": 123750,
   "groups": {
    "down8k|down8k": 4950,
    "down8k|noisy": 4950,
    "down8k|original": 4950,
    "down8k|remesh": 4950,
    "down8k|up60k": 4950,
    "noisy|down8k": 4950,
    "noisy|noisy": 4950,
    "noisy|original": 4950,
    "noisy|remesh": 4950,
    "noisy|up60k": 4950,
    "original|down8k": 4950,
    "original|noisy": 4950,
    "original|original": 4950,
    "original|remesh": 4950,
    "original|up60k": 4950,
    "remesh|down8k": 4950,
    "remesh|noisy": 4950,
    "remesh|original": 4950,
    "remesh|remesh": 4950,
    "remesh|up60k": 4950,
    "up60k|down8k": 4950,
    "up60k|noisy": 4950,
    "up60k|original": 4950,
    "up60k|remesh": 4950,
    "up60k|up60k": 4950
   },
   "gt_spread_within_subject_pair": 0.0
  },
  "faceverse": {
   "seed": 566363,
   "rows_cross": 99000,
   "rows_same": 24750,
   "rows_kept": 123750,
   "groups": {
    "down8k|down8k": 4950,
    "down8k|noisy": 4950,
    "down8k|original": 4950,
    "down8k|remesh": 4950,
    "down8k|up60k": 4950,
    "noisy|down8k": 4950,
    "noisy|noisy": 4950,
    "noisy|original": 4950,
    "noisy|remesh": 4950,
    "noisy|up60k": 4950,
    "original|down8k": 4950,
    "original|noisy": 4950,
    "original|original": 4950,
    "original|remesh": 4950,
    "original|up60k": 4950,
    "remesh|down8k": 4950,
    "remesh|noisy": 4950,
    "remesh|original": 4950,
    "remesh|remesh": 4950,
    "remesh|up60k": 4950,
    "up60k|down8k": 4950,
    "up60k|noisy": 4950,
    "up60k|original": 4950,
    "up60k|remesh": 4950,
    "up60k|up60k": 4950
   },
   "gt_spread_within_subject_pair": 0.0
  }
 },
 "heldout": {
  "embeddings_vs_calib_heldout": {
   "factorized_s1234": {
    "n": 1500,
    "keys_equal": true,
    "max_abs_diff": 0.0008167028427124023
   },
   "factorized_s2345": {
    "n": 1500,
    "keys_equal": true,
    "max_abs_diff": 0.0007919073104858398
   },
   "factorizedc3m_e123": {
    "n": 1500,
    "keys_equal": true,
    "max_abs_diff": 0.0027254223823547363
   },
   "factorizedc3m_e205": {
    "n": 1500,
    "keys_equal": true,
    "max_abs_diff": 0.004578590393066406
   }
  },
  "shift": {
   "bfm": {
    "factorized_s1234": {
     "same_subject_crop_vs_nocrop": 0.28228864958920835,
     "same_subject_nocrop": 0.14604167824285824,
     "different_subjects_original_median": 0.6930568055030766,
     "dlogS_crop_mean": -0.02293174791336059,
     "dlogS_crop_sd": 0.007286780831252524,
     "logS_sd_between_subjects": 0.018087617917918197
    },
    "factorized_s2345": {
     "same_subject_crop_vs_nocrop": 0.2804242539456843,
     "same_subject_nocrop": 0.1323976723753412,
     "different_subjects_original_median": 0.7220530148261051,
     "dlogS_crop_mean": -0.01854188203811648,
     "dlogS_crop_sd": 0.008404136380639103,
     "logS_sd_between_subjects": 0.018140716042163222
    },
    "ctrlfr_s1234": {
     "same_subject_crop_vs_nocrop": 0.37058721352360985,
     "same_subject_nocrop": 0.22752818920613174,
     "different_subjects_original_median": 0.9185634287512429
    },
    "ctrlfr_s2345": {
     "same_subject_crop_vs_nocrop": 0.42483261513893955,
     "same_subject_nocrop": 0.25689354126067665,
     "different_subjects_original_median": 0.9943947236575692
    },
    "factorizedc3m_e123": {
     "same_subject_crop_vs_nocrop": 0.420394670966399,
     "same_subject_nocrop": 0.28376887012434154,
     "different_subjects_original_median": 1.1333123272519443,
     "dlogS_crop_mean": -0.04603087759017946,
     "dlogS_crop_sd": 0.01240013094743287,
     "logS_sd_between_subjects": 0.02357339057866504
    },
    "factorizedc3m_e205": {
     "same_subject_crop_vs_nocrop": 0.4479255903973109,
     "same_subject_nocrop": 0.32504667188597147,
     "different_subjects_original_median": 1.311336778152762,
     "dlogS_crop_mean": -0.03792652130126953,
     "dlogS_crop_sd": 0.011200136072115932,
     "logS_sd_between_subjects": 0.020266867997979082
    }
   },
   "ict": {
    "factorized_s1234": {
     "same_subject_crop_vs_nocrop": 0.2797349190633614,
     "same_subject_nocrop": 0.19950234192499994,
     "different_subjects_original_median": 0.6878170616027741,
     "dlogS_crop_mean": 0.0008960962295532271,
     "dlogS_crop_sd": 0.00795013092010158,
     "logS_sd_between_subjects": 0.049539283303819286
    },
    "factorized_s2345": {
     "same_subject_crop_vs_nocrop": 0.26946313328957056,
     "same_subject_nocrop": 0.16647109317763312,
     "different_subjects_original_median": 0.6842622665680896,
     "dlogS_crop_mean": 0.006085906505584724,
     "dlogS_crop_sd": 0.009326343912734758,
     "logS_sd_between_subjects": 0.049816322154345255
    },
    "ctrlfr_s1234": {
     "same_subject_crop_vs_nocrop": 0.4469222738757654,
     "same_subject_nocrop": 0.34534915051111104,
     "different_subjects_original_median": 1.2500570905218418
    },
    "ctrlfr_s2345": {
     "same_subject_crop_vs_nocrop": 0.4779085627975885,
     "same_subject_nocrop": 0.3226816520877233,
     "different_subjects_original_median": 1.3432678182431341
    },
    "factorizedc3m_e123": {
     "same_subject_crop_vs_nocrop": 0.4242021010754996,
     "same_subject_nocrop": 0.3080436089673774,
     "different_subjects_original_median": 1.0859951192877246,
     "dlogS_crop_mean": -0.0002051653861999192,
     "dlogS_crop_sd": 0.012197713827851777,
     "logS_sd_between_subjects": 0.04888881391538357
    },
    "factorizedc3m_e205": {
     "same_subject_crop_vs_nocrop": 0.4537339539872053,
     "same_subject_nocrop": 0.3272687804173346,
     "different_subjects_original_median": 1.2431057272297172,
     "dlogS_crop_mean": 0.001128187656402555,
     "dlogS_crop_sd": 0.011605384330849131,
     "logS_sd_between_subjects": 0.04887814870372237
    }
   },
   "gnm": {
    "factorized_s1234": {
     "same_subject_crop_vs_nocrop": 0.28101023467256575,
     "same_subject_nocrop": 0.1738229788540134,
     "different_subjects_original_median": 0.7495749691613834,
     "dlogS_crop_mean": -0.001798263072967541,
     "dlogS_crop_sd": 0.005470164658375153,
     "logS_sd_between_subjects": 0.05079831684814294
    },
    "factorized_s2345": {
     "same_subject_crop_vs_nocrop": 0.25519972223506904,
     "same_subject_nocrop": 0.13147174604884485,
     "different_subjects_original_median": 0.7348718678525938,
     "dlogS_crop_mean": -0.003656077861785896,
     "dlogS_crop_sd": 0.0066441122707671585,
     "logS_sd_between_subjects": 0.05232139316668677
    },
    "ctrlfr_s1234": {
     "same_subject_crop_vs_nocrop": 0.37318506657034606,
     "same_subject_nocrop": 0.2624811522136583,
     "different_subjects_original_median": 1.15283188856489
    },
    "ctrlfr_s2345": {
     "same_subject_crop_vs_nocrop": 0.4098499393456873,
     "same_subject_nocrop": 0.266358534399839,
     "different_subjects_original_median": 1.232504260733069
    },
    "factorizedc3m_e123": {
     "same_subject_crop_vs_nocrop": 0.40155884046575113,
     "same_subject_nocrop": 0.2794752594915452,
     "different_subjects_original_median": 1.1437693320300135,
     "dlogS_crop_mean": -0.004553506851196261,
     "dlogS_crop_sd": 0.007251241306004904,
     "logS_sd_between_subjects": 0.050165144428094506
    },
    "factorizedc3m_e205": {
     "same_subject_crop_vs_nocrop": 0.44467035425973156,
     "same_subject_nocrop": 0.33228249017807204,
     "different_subjects_original_median": 1.3038669568717292,
     "dlogS_crop_mean": -0.0021910924911499484,
     "dlogS_crop_sd": 0.0075865330587389515,
     "logS_sd_between_subjects": 0.05018898914799079
    }
   }
  }
 },
 "paired": {
  "hifi3d": {
   "seed": 757683,
   "rows": 148500,
   "rows_mask": 148500,
   "rows_mask_crop": 49500,
   "topology_pairs": 15,
   "nan_rows_by_column": {},
   "arm_meshes": {
    "hifi3d|gnm|factorized_s1234": {
     "meshes": 600
    },
    "hifi3d|flame2023|factorized_s1234": {
     "meshes": 600
    },
    "hifi3d|gnm|factorized_s2345": {
     "meshes": 600
    },
    "hifi3d|flame2023|factorized_s2345": {
     "meshes": 600
    },
    "hifi3d|gnm|ctrlfr_s1234": {
     "meshes": 600
    },
    "hifi3d|flame2023|ctrlfr_s1234": {
     "meshes": 600
    },
    "hifi3d|gnm|ctrlfr_s2345": {
     "meshes": 600
    },
    "hifi3d|flame2023|ctrlfr_s2345": {
     "meshes": 600
    },
    "hifi3d|gnm|factorizedc3m_e123": {
     "meshes": 600
    },
    "hifi3d|flame2023|factorizedc3m_e123": {
     "meshes": 600
    },
    "hifi3d|gnm|factorizedc3m_e205": {
     "meshes": 600
    },
    "hifi3d|flame2023|factorizedc3m_e205": {
     "meshes": 600
    }
   },
   "vb600_vs_fit_e1": {
    "gnm_vb_coef": 5.329070518200751e-15,
    "gnm_vb_fr": 2.4555507627255224e-10,
    "gnm_vb_sr": 2.164604051557717e-10,
    "flame2023_vb_coef": 5.329070518200751e-15,
    "flame2023_vb_fr": 1.8132939594295294e-11,
    "flame2023_vb_sr": 1.181801878580302e-11
   },
   "geo_missing": [
    "hifi3d maxabs_chamfer (FileNotFoundError)",
    "hifi3d maxabs_rigid_icp_chamfer (FileNotFoundError)",
    "hifi3d maxabs_nicp_p2tri (FileNotFoundError)"
   ]
  },
  "facescape": {
   "seed": 621096,
   "rows": 148500,
   "rows_mask": 148500,
   "rows_mask_crop": 49500,
   "topology_pairs": 15,
   "nan_rows_by_column": {},
   "arm_meshes": {
    "facescape|gnm|factorized_s1234": {
     "meshes": 600
    },
    "facescape|flame2023|factorized_s1234": {
     "meshes": 600
    },
    "facescape|gnm|factorized_s2345": {
     "meshes": 600
    },
    "facescape|flame2023|factorized_s2345": {
     "meshes": 600
    },
    "facescape|gnm|ctrlfr_s1234": {
     "meshes": 600
    },
    "facescape|flame2023|ctrlfr_s1234": {
     "meshes": 600
    },
    "facescape|gnm|ctrlfr_s2345": {
     "meshes": 600
    },
    "facescape|flame2023|ctrlfr_s2345": {
     "meshes": 600
    },
    "facescape|gnm|factorizedc3m_e123": {
     "meshes": 600
    },
    "facescape|flame2023|factorizedc3m_e123": {
     "meshes": 600
    },
    "facescape|gnm|factorizedc3m_e205": {
     "meshes": 600
    },
    "facescape|flame2023|factorizedc3m_e205": {
     "meshes": 600
    }
   },
   "vb600_vs_fit_e1": {
    "gnm_vb_coef": 3.552713678800501e-15,
    "gnm_vb_fr": 1.8770898724262963e-10,
    "gnm_vb_sr": 2.0568172165447152e-10,
    "flame2023_vb_coef": 1.0658141036401503e-14,
    "flame2023_vb_fr": 9.568817960214915e-12,
    "flame2023_vb_sr": 1.1014272827125637e-11
   },
   "geo_missing": []
  },
  "faceverse": {
   "seed": 566363,
   "rows": 148500,
   "rows_mask": 148500,
   "rows_mask_crop": 49500,
   "topology_pairs": 15,
   "nan_rows_by_column": {},
   "arm_meshes": {
    "faceverse|gnm|factorized_s1234": {
     "meshes": 600
    },
    "faceverse|flame2023|factorized_s1234": {
     "meshes": 600
    },
    "faceverse|gnm|factorized_s2345": {
     "meshes": 600
    },
    "faceverse|flame2023|factorized_s2345": {
     "meshes": 600
    },
    "faceverse|gnm|ctrlfr_s1234": {
     "meshes": 600
    },
    "faceverse|flame2023|ctrlfr_s1234": {
     "meshes": 600
    },
    "faceverse|gnm|ctrlfr_s2345": {
     "meshes": 600
    },
    "faceverse|flame2023|ctrlfr_s2345": {
     "meshes": 600
    },
    "faceverse|gnm|factorizedc3m_e123": {
     "meshes": 600
    },
    "faceverse|flame2023|factorizedc3m_e123": {
     "meshes": 600
    },
    "faceverse|gnm|factorizedc3m_e205": {
     "meshes": 600
    },
    "faceverse|flame2023|factorizedc3m_e205": {
     "meshes": 600
    }
   },
   "vb600_vs_fit_e1": {
    "gnm_vb_coef": 3.552713678800501e-15,
    "gnm_vb_fr": 3.684685889737693e-11,
    "gnm_vb_sr": 5.672307068493865e-11,
    "flame2023_vb_coef": 5.329070518200751e-15,
    "flame2023_vb_fr": 2.295164058807586e-12,
    "flame2023_vb_sr": 3.5534908349177385e-12
   },
   "geo_missing": [
    "faceverse maxabs_chamfer (FileNotFoundError)",
    "faceverse maxabs_rigid_icp_chamfer (FileNotFoundError)",
    "faceverse maxabs_nicp_p2tri (FileNotFoundError)"
   ]
  },
  "faceverse_neutral": {
   "seed": 566363,
   "rows": 148500,
   "rows_mask": 148500,
   "rows_mask_crop": 49500,
   "topology_pairs": 15,
   "nan_rows_by_column": {},
   "arm_meshes": {
    "faceverse_neutral|gnm|factorized_s1234": {
     "meshes": 600
    },
    "faceverse_neutral|flame2023|factorized_s1234": {
     "meshes": 600
    },
    "faceverse_neutral|gnm|factorized_s2345": {
     "meshes": 600
    },
    "faceverse_neutral|flame2023|factorized_s2345": {
     "meshes": 600
    },
    "faceverse_neutral|gnm|ctrlfr_s1234": {
     "meshes": 600
    },
    "faceverse_neutral|flame2023|ctrlfr_s1234": {
     "meshes": 600
    },
    "faceverse_neutral|gnm|ctrlfr_s2345": {
     "meshes": 600
    },
    "faceverse_neutral|flame2023|ctrlfr_s2345": {
     "meshes": 600
    },
    "faceverse_neutral|factorizedc3m_e123": "store assente, braccio saltato",
    "faceverse_neutral|gnm|factorizedc3m_e205": {
     "meshes": 600
    },
    "faceverse_neutral|flame2023|factorizedc3m_e205": {
     "meshes": 600
    }
   },
   "vb600_vs_fit_e1": {
    "gnm_vb_coef": 3.552713678800501e-15,
    "gnm_vb_fr": 1.485143952262291e-10,
    "gnm_vb_sr": 1.738424404429395e-10,
    "flame2023_vb_coef": 7.105427357601002e-15,
    "flame2023_vb_fr": 1.094402346524248e-11,
    "flame2023_vb_sr": 1.020522555350567e-11
   },
   "geo_missing": [
    "faceverse_neutral maxabs_chamfer (FileNotFoundError)",
    "faceverse_neutral maxabs_rigid_icp_chamfer (FileNotFoundError)",
    "faceverse_neutral maxabs_nicp_p2tri (FileNotFoundError)",
    "faceverse_neutral maxabs_nicp_template (FileNotFoundError)",
    "faceverse_neutral mm_chamfer (FileNotFoundError)",
    "faceverse_neutral mm_chamfer_pure (FileNotFoundError)",
    "faceverse_neutral mm_rigid_icp_chamfer (FileNotFoundError)",
    "faceverse_neutral mm_nicp_p2tri (FileNotFoundError)",
    "faceverse_neutral mm_nicp_template (FileNotFoundError)",
    "faceverse_neutral cs_chamfer (FileNotFoundError)",
    "faceverse_neutral cs_rigid_icp_chamfer (FileNotFoundError)",
    "faceverse_neutral cs_nicp_p2tri (FileNotFoundError)",
    "faceverse_neutral cs_nicp_template (FileNotFoundError)",
    "faceverse_neutral mm_rigid_icp_chamfer assente",
    "faceverse_neutral mm_chamfer_pure assente",
    "faceverse_neutral mm_nicp_template assente"
   ]
  }
 },
 "paired_e2_values": {
  "n_matched": 496,
  "columns": [
   "ctrlfr_s1234|z",
   "ctrlfr_s2345|z",
   "factorized_s1234|form_cal",
   "factorized_s1234|shape",
   "factorized_s2345|form_cal",
   "factorized_s2345|shape",
   "factorizedc3m_e123|form_cal",
   "factorizedc3m_e123|shape",
   "factorizedc3m_e205|form_cal",
   "factorizedc3m_e205|shape",
   "flame2023_vb_coef",
   "flame2023_vb_fr",
   "flame2023_vb_sr",
   "gnm_vb_coef",
   "gnm_vb_fr",
   "gnm_vb_sr"
  ],
  "max_abs_diff_point": 8.326672684688674e-17,
  "max_abs_diff_ci": 8.326672684688674e-17,
  "rows_equal": true
 },
 "chain": {
  "n": 600,
  "keys_equal": true,
  "max_abs_diff": 4.76837158203125e-07
 },
 "counts_e2": {
  "hifi3d | all_cross": "FR 12 / 4 / 8, SR 5 / 10 / 9",
  "hifi3d | all_cross, righe col crop": "FR 6 / 14 / 4, SR 4 / 10 / 10",
  "facescape | all_cross": "FR 0 / 24 / 0, SR 0 / 24 / 0",
  "facescape | all_cross, righe col crop": "FR 0 / 24 / 0, SR 0 / 19 / 5",
  "faceverse | all_cross": "FR 1 / 0 / 23, SR 1 / 0 / 23",
  "faceverse | all_cross, righe col crop": "FR 0 / 0 / 24, SR 0 / 0 / 24",
  "faceverse_neutral | all_cross": "FR 0 / 0 / 24, SR 0 / 1 / 23",
  "faceverse_neutral | all_cross, righe col crop": "FR 0 / 3 / 21, SR 0 / 4 / 20"
 },
 "region": {
  "hifi3d|gnm": {
   "n_region_npz": 7700,
   "n_rerun": 7700,
   "equal": true
  },
  "hifi3d|flame2023": {
   "n_region_npz": 1517,
   "n_rerun": 1517,
   "equal": true
  },
  "facescape|gnm": {
   "n_region_npz": 8061,
   "n_rerun": 8061,
   "equal": true
  },
  "facescape|flame2023": {
   "n_region_npz": 1544,
   "n_rerun": 1544,
   "equal": true
  },
  "faceverse|gnm": {
   "n_region_npz": 8654,
   "n_rerun": 8654,
   "equal": true
  },
  "faceverse|flame2023": {
   "n_region_npz": 1674,
   "n_rerun": 1674,
   "equal": true
  },
  "faceverse_neutral|gnm": {
   "n_region_npz": 8654,
   "n_rerun": 8654,
   "equal": true
  },
  "faceverse_neutral|flame2023": {
   "n_region_npz": 1674,
   "n_rerun": 1674,
   "equal": true
  }
 }
}
```

