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

