# Concorrenti parametrici (GNM, FLAME 2023 Open) e varifold: risultati

Protocollo `PROTOCOL.md` (sha256 `0d7444e622c7295fe620467e5e0c136dd0f06d348e35de16de09a65e8566cda1`), codice `aau/baselines_param/`. Righe, GT e repliche di `v3_work/trainer/tools/fact_paired.py` (importato, non modificato). IC 95% percentile, 1000 repliche per soggetto. Tutti i numeri in `spearman.csv` (valori) e `paired.csv` (valori e delta, anche parziale e quintile basso).

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
  neutra FR 19, SR 22; FaMoS FR 10, SR 9; nessuna sotto 0. Spearman quasi invariati (es. HIFI3D FR GNM coefficienti
  0.619 -> 0.603, FLAME mesh FR 0.639 -> 0.641). Togliere l'espressione aumenta |beta| (GNM su HIFI3D da 4.6 a 7.0 di
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

