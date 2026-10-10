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
  contro le mesh FR di B), FaceScape FR 0 / 24 / 0. Su FaceVerse (entrambe le viste) quasi tutto resta non risolto.
- **Perche' i bracci cadono sul crop** (diagnostica aggiunta dopo i numeri, non nel protocollo; tabella in
  `results.md`): il crop sposta in modo sistematico l'embedding dei bracci. Su FaceScape factorized stima per il crop
  una taglia piu' piccola (log S -0.061 / -0.070, cioe' circa 4 volte la deviazione standard di log S fra soggetti,
  0.016-0.017) e la sua d_P crop -> stesso soggetto (0.68 / 0.71) supera la mediana fra soggetti diversi (0.55); ctrlfr
  su FaceScape 1.43 contro 0.79, su HIFI3D 1.24 contro 1.05. Su HIFI3D lo spostamento di log S (-0.024 / -0.034) e'
  piccolo rispetto alla varianza della taglia (0.047): per questo factorized regge meglio li'. La GT e B sono
  definite sulla regione comune, che il crop non tocca; i bracci vedono l'area della mesh (normalizzazione globale).
  Lo scenario crop e' fuori dalla distribuzione di training dei bracci? Non verificato qui.
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
  crop nascondeva una fragilita' dei bracci al ritaglio, che B non ha. (2) Una composizione dichiarata forma B +
  taglia B, con k fissato senza test, non batte i bracci su HIFI3D FR (4 a favore dei bracci, 4 non risolte) ma li
  batte su FaceScape. Quello che regge per i bracci contro tutto quanto provato: HIFI3D FR senza crop (contro B
  congelata, B a sigma 8 e la composizione). FaceScape e' il dominio di sviluppo dei bracci: le sconfitte li' pesano di
  piu'. Nessuna correzione per confronti multipli; la scelta di sigma resta sul bordo della griglia.
