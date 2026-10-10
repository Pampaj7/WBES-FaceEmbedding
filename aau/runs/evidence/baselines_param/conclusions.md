## Lettura (regola della sez. 7 del protocollo, scritta dopo i numeri)

Job: fit 1067569 (GNM, 4 min 22 s) e 1067570 (FLAME, 20 min 40 s), 48 core CPU ciascuno; varifold 1067571-1067574
(una L40S per vista, da 17 min 38 s a 29 min 44 s); controllo GPU/CPU 1067582; delta 1067575 (15 min 31 s, 32 core).
FaMoS (15 mesh): fit e varifold eseguiti nella prova funzionale, prima del protocollo, senza GT (vedi PROTOCOL.md).
Le correzioni di questa lettura e le varianti A e B vengono dall'emendamento 1 (POST HOC, `PROTOCOL_emendamento_1.md`,
in fondo).

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
  del protocollo aderiva male alla superficie; B corregge.
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
    [0.610, 0.734] contro ctrlfr 0.339 / 0.348 (delta fino a -0.336 [-0.434, -0.241]). Factorized d_P (0.622 / 0.613)
    resta NON risolto contro B;
  - FaceScape FR, factorized d_F cal. e ctrlfr (entrambi i semi) contro GNM vB mesh SR (4): 0.767 [0.675, 0.843]
    contro 0.653-0.669 (delta da -0.098 a -0.114);
  - FaceScape SR, factorized d_P e ctrlfr contro GNM e FLAME vB mesh SR (8): GNM 0.854 [0.803, 0.894], FLAME 0.820
    [0.757, 0.870] contro d_P 0.747 / 0.753 (delta -0.107 [-0.155, -0.065] ... -0.067 [-0.120, -0.014]) e ctrlfr
    0.619 / 0.627;
  - FaceVerse neutra SR, ctrlfr s2345 contro GNM vB mesh SR (1): -0.093 [-0.181, -0.011].
- **Cosa regge contro tutti i concorrenti, B compreso**: HIFI3D con FR, 24 / 24 a favore (factorized d_F cal. 0.749 /
  0.731, ctrlfr 0.757 / 0.746, contro il migliore di B, FLAME vB mesh FR, 0.686 [0.597, 0.758]). FaceVerse (entrambe le
  viste) e FaMoS: nessuna cella sotto 0 tranne quella di FaceVerse neutra, ma quasi tutte non risolte contro B (es.
  FaMoS SR: FLAME vB mesh SR 0.791, d_P 0.740 / 0.728; FaMoS FR: ctrlfr 0.831 / 0.818 contro GNM vB mesh FR 0.686).
- **Lettura.** La frase "nessun concorrente batte i bracci dichiarati" vale solo per il fit preregistrato (a due stadi,
  aderenza 1-1.6 mm). Con un fit che mette il modello nel ciclo (iperparametri scelti su soggetti non valutati, senza
  GT) un 3DMM statistico eguaglia o batte i bracci con SR su HIFI3D (contro ctrlfr) e su FaceScape (contro tutti), e
  con FR su FaceScape (contro tutti); vale per GNM (prior VISTO) e anche per FLAME 2023 Open (prior non visto) con SR
  su FaceScape. FaceScape e' il dominio dev, generato da un modello bilineare: un 3DMM ben fittato vi ricostruisce
  bene la forma. Contro letture non dichiarate B vince in piu' celle (C3M d_F cal. / d_P su FaceScape: 5 celle sotto
  0). Tutto cio' e' post hoc e va riportato come tale, accanto all'analisi preregistrata.
- Limiti: le scelte di B stanno sul bordo superiore della griglia di sigma (2 mm); sul pilota B migliorava FaceScape e
  peggiorava FaceVerse con espressioni (sez. 8 dell'emendamento); nessuna correzione per confronti multipli.
