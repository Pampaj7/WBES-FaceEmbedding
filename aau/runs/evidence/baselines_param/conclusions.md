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
  (Correzione dell'emendamento 3: il crop E' nella distribuzione di training, con perdita d'area comparabile; i
  bracci vi sono invarianti sugli held-out sintetici ICT e GNM; il calo sui test dipende in gran parte dal supporto:
  vedi sotto.)
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
mesh da /tmp). Tutto e' post hoc e si legge come sensibilita' e diagnostica.

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
     l'invarianza al crop c'e' (esperimento 2, sotto).
  2. *Baseline geometriche su all_cross* (stesse 148.500 righe, `baselines_mm/spearman.csv`, ricalcolate identiche): FR
     FaceScape ICP + Chamfer mm 0.381 [0.326, 0.434], Chamfer pura 0.570 [0.499, 0.641]; HIFI3D 0.556 [0.488, 0.617] e
     0.645 [0.547, 0.725]. Con il crop i bracci interi arrivano al livello di ICP o sotto (FaceScape d_F cal. 0.370 /
     0.351, ctrlfr 0.334 / 0.323; HIFI3D ctrlfr 0.516 / 0.530; factorized s1234 0.709 resta sopra). Sulle righe col crop
     i bracci interi perdono contro la Chamfer pura: HIFI3D FR ctrlfr -0.408 / -0.365 (anche contro ICP mm -0.181 /
     -0.138) e factorized s2345 -0.091; FaceScape FR tutti e 4 (da -0.153 a -0.270), SR ctrlfr; FaceVerse FR e SR tutti
     e 4 (da -0.083 a -0.128).
  3. *C3M crolla anch'esso*: d_F cal. con FR, senza crop -> righe col crop, e205 HIFI3D 0.739 -> 0.472, FaceScape 0.712
     -> 0.449 (il critic: 0.739 -> 0.472, 0.711 -> 0.449), e123 0.748 -> 0.506 e 0.666 -> 0.472. Il run massivo ha lo
     stesso crop fisso con peso 1.0: per l'esperimento 2 l'invarianza in-distribuzione c'e' gia', quindi non c'e'
     ragione di attendersi che piu' training sullo stesso crop risolva da solo la fragilita' sui test.
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
  topologie 0.95-0.996): per la regola dichiarata (< 0.95) il crop toglie anche una parte della regione di B, che
  quindi lavora gia' su dati parziali e regge lo stesso. Ma dopo il ritaglio crop e original hanno quasi la stessa
  area (log sqrt(area crop / area original) da -0.009 a +0.001, contro -0.045 / -0.132 prima): la parte di regione tolta
  e' una banda sottile al suo bordo (<= ~2% d'area). Il recupero sotto e' quindi atteso in gran parte per costruzione:
  dice che il calo veniva dalla banda fuori dalla regione di B.
- **Esperimento 1: con la stessa regione i bracci recuperano dove calavano.** FaceScape: tutti i bracci dichiarati, con
  entrambe le regioni, tornano sulle righe col crop al livello senza crop (recupero F 0.91-1.28; D3 righe col crop da
  +0.23 a +0.48, IC sopra 0): **dipendenza dal supporto**, 8 / 8 con FR e con SR. HIFI3D: ctrlfr recupera (F 0.70-0.74
  con FR, > 1 con SR); factorized recupera in parte con la regione GNM (s2345 F 0.52 con FR e 0.78 con SR; s1234 0.20 e
  0.62) e non con la regione FLAME (1.517 vertici; F <= 0.16, in tre casi su quattro negativo); lettura per maggioranza: regione GNM dipendenza
  dal supporto (FR 3 / 4, SR 4 / 4), regione FLAME pari (2 / 2). FaceVerse (entrambe le viste): il calo del braccio
  intero e' piccolo (< 0.05 per ctrlfr, 0.03-0.10 per factorized), quasi nulla da leggere. C3M (descrittivo) recupera
  come i bracci (e205 HIFI3D F 0.80 / 0.56, FaceScape 0.86 / 0.70).
  **Il ritaglio non e' gratis**: sulle righe senza crop di HIFI3D restringere l'ingresso alla regione costa a tutti i
  bracci con FR (D3 0 / 8 / 0, da -0.05 a -0.16, peggio con la regione FLAME): su HIFI3D i bracci usano informazione
  fuori dalla regione di B, la stessa che il crop rovina. Su FaceScape nessun costo (D3 senza crop FR 2 / 0 / 6, SR 3 /
  0 / 5).
  **Contro B a regione uguale, righe col crop** (D1, a favore / contro / non risolte): FaceScape FR 13 / 2 / 9
  (emendamento 2: 0 / 24 / 0), SR 15 / 6 / 3 (0 / 19 / 5); HIFI3D FR 11 / 0 / 13 (6 / 14 / 4), SR 8 / 10 / 6 (4 / 10 /
  10). Restano per B: GNM vB mesh SR con FR su FaceScape contro factorized s1234 @ GNM (-0.068 [-0.118, -0.018]) e
  ctrlfr s2345 @ GNM (-0.139); con SR le mesh SR di GNM e FLAME contro ctrlfr (4) e factorized (2, da -0.046 a -0.057)
  su FaceScape, e su HIFI3D contro ctrlfr (8, che non e' un braccio per SR) e factorized @ FLAME (-0.082 / -0.087): la
  stessa geografia delle celle B senza crop dell'emendamento 1. Contro le baseline geometriche (D2) sulle righe col
  crop: HIFI3D FR 16 / 0 / 8, SR 24 / 0 / 0; FaceScape 24 / 0 / 0; FaceVerse FR 7 / 7 / 10, SR 8 / 6 / 10.
- **Esperimento 2: in-distribuzione l'invarianza al crop c'e'.** Held-out sintetici (100 soggetti per dominio), coppie
  col crop contro senza crop: ICT e GNM, delta dello Spearman da -0.021 a +0.008 per tutti i bracci dichiarati (con FR e
  con SR), AUC di verifica 0.999-1.000 in entrambi i casi, d log S del crop da -0.004 a +0.006 (<= 0.12 sd fra
  soggetti): **invarianza presente**. BFM: con SR -0.019 / -0.026 (IC sotto 0, ma |delta| < 0.05) e d log S -0.023 /
  -0.019 (1.3 / 1.0 sd): **parziale** per factorized; ctrlfr presente; con FR BFM non si legge (taglia delle original
  REMESH non e' quella del modello). Sui test, invece, il calo dei bracci interi col crop e' di 0.23-0.42 su FaceScape
  (factorized FR 0.659 -> 0.351, 0.667 -> 0.278) e d log S era ~4 sd (emendamento 2). Quindi l'invarianza appresa sul
  crop sintetico non si trasferisce ai domini di test: e' un limite di generalizzazione, non l'assenza del crop nel
  training.
- **Sez. 6 (descrittiva, non cieca), Spearman dentro le coppie di topologie senza crop**, bracci interi, IC per
  soggetto col seme di all_cross. HIFI3D factorized s1234 d_F cal. con FR: media sulle 20 coppie ordinate 0.749 [0.675,
  0.811] (min 0.730 remesh -> up60k, max 0.760 up60k -> original), stessa topologia 0.760 [0.685, 0.821], differenza -0.011
  [-0.013, -0.009]; FaceScape: 0.667 [0.598, 0.732] (min 0.628, max 0.706), stessa topologia 0.706 [0.640, 0.769],
  differenza -0.039 [-0.047, -0.032]; le stime puntuali del critic sono ridate esattamente. Gli altri bracci: differenza
  20 - stessa topologia da -0.005 a -0.021 su HIFI3D, da -0.020 a -0.054 su FaceScape (ctrlfr s1234 il piu' alto), da
  -0.001 a -0.010 su FaceVerse (0.26-0.32 di livello). Lettura: senza crop la discretizzazione costa poco (fino a 0.02
  su HIFI3D, fino a 0.05 su FaceScape, sistematico: IC della differenza sotto 0 su HIFI3D e FaceScape); la topologia
  peggiore e' spesso noisy.
- **Lettura.** Il ribaltamento dell'emendamento 2 sulle righe col crop (FaceScape 48 / 48 celle contro i bracci) e' in
  gran parte un'asimmetria di supporto: i bracci sono sensibili alla banda di bordo fuori dalla regione di B, che B non
  guarda per costruzione; a regione uguale, con FR, non perdono quasi piu' contro B (FaceScape 13 a favore e 2 contro
  su 24, HIFI3D 11 e 0) e restano dietro alle mesh SR di B su FaceScape SR, come senza crop. Non e' un esito gratis per i bracci: su HIFI3D senza crop la
  regione di B costa (fino a -0.16), e sui test il crop resta un punto debole dei bracci sull'ingresso intero, ne' il
  training sullo stesso crop lo cura (esperimento 2). Tutto post hoc; nessuna correzione per confronti multipli.
