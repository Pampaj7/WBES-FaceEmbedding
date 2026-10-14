# Protocollo adattabile (few-shot): risultati

Generato da `aau/few_shot/fs_eval.py` (non modificare a mano sopra il segno). Regole in `REGOLE.md` (commit separato, prima dei numeri). Per (dominio, k): media sulle 5 estrazioni; IC 95% percentile con 1000 repliche per soggetto ristrette alle identita' di test (stesso ricampionamento per tutte le estrazioni); sd = deviazione standard fra le estrazioni. Nostro adattato = testa di R6 (r = 64, lambda = 1e-6, EQ) con le k identita' in coda alle 7 sorgenti senza il generatore del dominio; letture d_h con SR, d_F della testa con FR. Concorrente adattato = configurazione della famiglia scelta sulle stesse k identita'. Delta = nostro - concorrente, replica per replica (positivo = il nostro davanti).

**Riferimento esterno (non parte di queste regole)**: sulla prova confermativa di Ava-256 la R6 (stessa testa, adattata sui 56 di calibrazione, delta rho(d_h, SR) - rho(d_P, SR)) NON e' confermata: s1234 -0.000 [-0.024, +0.026], s2345 +0.020 [-0.004, +0.046] (`aau/runs/evidence/ava256_conf/conf/results.md:19-20`). Qui la domanda e' il confronto coi concorrenti adattati; l'analogo della R6 e' il delta `contro nostro non adattato` con GT SR (sorgenti senza il generatore del dominio, k fino a 50).

## k di sorpasso

Il k minimo della griglia al quale l'IC 95% del delta contro OGNUNA delle tre famiglie adattate sta sopra lo 0 (sez. 6 delle regole); `entrambi` = il massimo dei due semi, `nessuno` se un seme non sorpassa.

| dominio | GT | s1234 (k sopra) | s2345 (k sopra) | entrambi |
|---|---|---|---|---|
| HIFI3D | FR | 5 (5 10 20 50) | 5 (5 10 20 50) | 5 |
| HIFI3D | SR | nessuno (-) | nessuno (-) | nessuno |
| FaceScape dev | FR | nessuno (-) | nessuno (-) | nessuno |
| FaceScape dev | SR | nessuno (-) | nessuno (-) | nessuno |
| FaceVerse con espressioni | FR | nessuno (-) | nessuno (-) | nessuno |
| FaceVerse con espressioni | SR | nessuno (-) | nessuno (-) | nessuno |
| FaceVerse neutra | FR | nessuno (-) | nessuno (-) | nessuno |
| FaceVerse neutra | SR | nessuno (-) | nessuno (-) | nessuno |

## HIFI3D

**GT FR: rho sulle identita' di test**

| colonna | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|
| nostro adattato s1234 | 0.746 [0.667, 0.808] sd 0.011 | 0.749 [0.671, 0.810] sd 0.012 | 0.771 [0.702, 0.825] sd 0.024 | 0.774 [0.709, 0.820] sd 0.035 |
| nostro adattato s2345 | 0.735 [0.659, 0.795] sd 0.013 | 0.736 [0.661, 0.797] sd 0.014 | 0.761 [0.692, 0.815] sd 0.021 | 0.753 [0.684, 0.806] sd 0.042 |
| nostro non adattato s1234 | 0.747 [0.668, 0.805] sd 0.010 | 0.744 [0.667, 0.805] sd 0.014 | 0.758 [0.684, 0.816] sd 0.020 | 0.745 [0.665, 0.801] sd 0.044 |
| nostro non adattato s2345 | 0.729 [0.650, 0.792] sd 0.011 | 0.726 [0.649, 0.790] sd 0.016 | 0.740 [0.668, 0.802] sd 0.024 | 0.729 [0.655, 0.785] sd 0.050 |
| testa k = 0 s1234 | 0.745 [0.667, 0.807] sd 0.011 | 0.743 [0.664, 0.807] sd 0.015 | 0.756 [0.677, 0.816] sd 0.020 | 0.743 [0.658, 0.804] sd 0.040 |
| testa k = 0 s2345 | 0.733 [0.655, 0.794] sd 0.012 | 0.731 [0.652, 0.792] sd 0.017 | 0.745 [0.671, 0.802] sd 0.022 | 0.728 [0.646, 0.789] sd 0.040 |
| concorrente adattato ICP + Chamfer | 0.583 [0.521, 0.639] sd 0.122 | 0.640 [0.565, 0.702] sd 0.015 | 0.644 [0.568, 0.703] sd 0.014 | 0.628 [0.555, 0.689] sd 0.055 |
| concorrente adattato NoW col rilevatore | 0.633 [0.572, 0.686] sd 0.116 | 0.687 [0.618, 0.745] sd 0.014 | 0.691 [0.620, 0.749] sd 0.014 | 0.675 [0.602, 0.730] sd 0.052 |
| concorrente adattato fit 3DMM vB | 0.667 [0.591, 0.731] sd 0.044 | 0.692 [0.604, 0.765] sd 0.015 | 0.712 [0.628, 0.777] sd 0.021 | 0.695 [0.615, 0.758] sd 0.063 |
| concorrente adattato NoW coi landmark esatti (descrittivo) | 0.639 [0.580, 0.693] sd 0.120 | 0.695 [0.622, 0.752] sd 0.012 | 0.701 [0.630, 0.759] sd 0.013 | 0.685 [0.612, 0.742] sd 0.046 |
| concorrente adattato migliore fra tutte (scelto su A) | 0.657 [0.585, 0.719] sd 0.041 | 0.690 [0.605, 0.759] sd 0.012 | 0.710 [0.629, 0.772] sd 0.022 | 0.688 [0.608, 0.748] sd 0.057 |

**GT FR: delta nostro adattato - contro**

| seme | contro | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|---|
| s1234 | ICP + Chamfer | +0.162 [+0.113, +0.215] | +0.109 [+0.068, +0.157] | +0.128 [+0.090, +0.176] | +0.145 [+0.107, +0.196] |
| s1234 | NoW col rilevatore | +0.113 [+0.064, +0.161] | +0.062 [+0.024, +0.105] | +0.081 [+0.044, +0.125] | +0.099 [+0.058, +0.149] |
| s1234 | fit 3DMM vB | +0.079 [+0.031, +0.124] | +0.056 [+0.012, +0.101] | +0.060 [+0.020, +0.106] | +0.079 [+0.040, +0.128] |
| s1234 | NoW coi landmark esatti (descrittivo) | +0.106 [+0.059, +0.151] | +0.054 [+0.019, +0.093] | +0.071 [+0.039, +0.111] | +0.089 [+0.051, +0.136] |
| s1234 | miglior famiglia (post hoc) | +0.079 [+0.031, +0.124] (fit 3DMM vB) | +0.056 [+0.012, +0.101] (fit 3DMM vB) | +0.060 [+0.020, +0.106] (fit 3DMM vB) | +0.079 [+0.040, +0.128] (fit 3DMM vB) |
| s1234 | migliore fra tutte (scelto su A) | +0.089 [+0.044, +0.131] | +0.059 [+0.018, +0.101] | +0.061 [+0.025, +0.105] | +0.086 [+0.048, +0.134] |
| s1234 | nostro non adattato | -0.001 [-0.019, +0.017] | +0.005 [-0.012, +0.023] | +0.013 [-0.004, +0.032] | +0.029 [+0.005, +0.058] |
| s1234 | testa k = 0 | +0.000 [-0.002, +0.003] | +0.006 [+0.001, +0.012] | +0.015 [+0.007, +0.026] | +0.031 [+0.009, +0.063] |
| s2345 | ICP + Chamfer | +0.152 [+0.104, +0.204] | +0.097 [+0.053, +0.145] | +0.117 [+0.074, +0.166] | +0.124 [+0.081, +0.176] |
| s2345 | NoW col rilevatore | +0.102 [+0.053, +0.150] | +0.050 [+0.012, +0.090] | +0.070 [+0.033, +0.112] | +0.077 [+0.035, +0.131] |
| s2345 | fit 3DMM vB | +0.068 [+0.025, +0.109] | +0.044 [+0.004, +0.088] | +0.049 [+0.013, +0.094] | +0.058 [+0.018, +0.106] |
| s2345 | NoW coi landmark esatti (descrittivo) | +0.095 [+0.051, +0.138] | +0.041 [+0.006, +0.080] | +0.060 [+0.026, +0.098] | +0.068 [+0.028, +0.117] |
| s2345 | miglior famiglia (post hoc) | +0.068 [+0.025, +0.109] (fit 3DMM vB) | +0.044 [+0.004, +0.088] (fit 3DMM vB) | +0.049 [+0.013, +0.094] (fit 3DMM vB) | +0.058 [+0.018, +0.106] (fit 3DMM vB) |
| s2345 | migliore fra tutte (scelto su A) | +0.078 [+0.036, +0.118] | +0.046 [+0.008, +0.088] | +0.051 [+0.015, +0.094] | +0.065 [+0.025, +0.114] |
| s2345 | nostro non adattato | +0.006 [-0.012, +0.026] | +0.010 [-0.008, +0.032] | +0.021 [+0.002, +0.045] | +0.024 [+0.004, +0.048] |
| s2345 | testa k = 0 | +0.002 [-0.000, +0.005] | +0.006 [+0.002, +0.010] | +0.016 [+0.008, +0.024] | +0.024 [+0.007, +0.048] |

Configurazioni scelte sulle k identita' (GT FR, estrazioni per configurazione):

- ICP + Chamfer: k = 5: mm 4, cs 1; k = 10: mm 5; k = 20: mm 5; k = 50: mm 5
- NoW col rilevatore: k = 5: metrical 4, originale 1; k = 10: metrical 5; k = 20: metrical 5; k = 50: metrical 5
- fit 3DMM vB: k = 5: gnm|comp 2, flame2023|coef 1, gnm|fr 1, flame2023|comp 1; k = 10: gnm|comp 3, flame2023|fr 1, flame2023|comp 1; k = 20: flame2023|comp 4, gnm|comp 1; k = 50: flame2023|comp 5
- NoW coi landmark esatti (descrittivo): k = 5: metrical 4, originale 1; k = 10: metrical 5; k = 20: metrical 5; k = 50: metrical 5
- tutte le famiglie: k = 5: fit_vb:flame2023|coef 1, icp:mm 1, fit_vb:gnm|fr 1, fit_vb:gnm|comp 1, fit_vb:flame2023|comp 1; k = 10: fit_vb:gnm|comp 2, now_det:metrical 1, fit_vb:flame2023|fr 1, fit_vb:flame2023|comp 1; k = 20: fit_vb:flame2023|comp 4, now_det:metrical 1; k = 50: fit_vb:flame2023|comp 4, now_det:metrical 1

**GT SR: rho sulle identita' di test**

| colonna | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|
| nostro adattato s1234 | 0.642 [0.557, 0.714] sd 0.018 | 0.656 [0.576, 0.721] sd 0.014 | 0.689 [0.617, 0.750] sd 0.025 | 0.725 [0.668, 0.776] sd 0.025 |
| nostro adattato s2345 | 0.644 [0.564, 0.715] sd 0.017 | 0.651 [0.575, 0.719] sd 0.015 | 0.679 [0.607, 0.742] sd 0.012 | 0.697 [0.631, 0.760] sd 0.034 |
| nostro non adattato s1234 | 0.616 [0.543, 0.681] sd 0.008 | 0.612 [0.540, 0.677] sd 0.013 | 0.614 [0.542, 0.683] sd 0.012 | 0.614 [0.531, 0.689] sd 0.035 |
| nostro non adattato s2345 | 0.609 [0.528, 0.682] sd 0.011 | 0.605 [0.525, 0.678] sd 0.016 | 0.604 [0.522, 0.680] sd 0.016 | 0.609 [0.524, 0.687] sd 0.038 |
| testa k = 0 s1234 | 0.641 [0.556, 0.713] sd 0.009 | 0.637 [0.552, 0.709] sd 0.017 | 0.643 [0.555, 0.716] sd 0.013 | 0.635 [0.536, 0.713] sd 0.044 |
| testa k = 0 s2345 | 0.637 [0.556, 0.710] sd 0.010 | 0.635 [0.554, 0.705] sd 0.018 | 0.641 [0.558, 0.713] sd 0.012 | 0.625 [0.529, 0.708] sd 0.032 |
| concorrente adattato ICP + Chamfer | 0.576 [0.526, 0.627] sd 0.008 | 0.574 [0.523, 0.624] sd 0.013 | 0.576 [0.524, 0.626] sd 0.008 | 0.573 [0.510, 0.629] sd 0.026 |
| concorrente adattato NoW col rilevatore | 0.610 [0.543, 0.669] sd 0.008 | 0.607 [0.545, 0.669] sd 0.017 | 0.609 [0.543, 0.670] sd 0.013 | 0.616 [0.551, 0.676] sd 0.031 |
| concorrente adattato fit 3DMM vB | 0.615 [0.552, 0.675] sd 0.037 | 0.606 [0.545, 0.666] sd 0.038 | 0.633 [0.577, 0.681] sd 0.055 | 0.664 [0.591, 0.729] sd 0.045 |
| concorrente adattato NoW coi landmark esatti (descrittivo) | 0.615 [0.549, 0.673] sd 0.009 | 0.612 [0.551, 0.672] sd 0.016 | 0.614 [0.553, 0.674] sd 0.012 | 0.619 [0.553, 0.681] sd 0.030 |
| concorrente adattato migliore fra tutte (scelto su A) | 0.616 [0.557, 0.673] sd 0.036 | 0.608 [0.549, 0.665] sd 0.037 | 0.638 [0.584, 0.686] sd 0.048 | 0.664 [0.591, 0.729] sd 0.045 |

**GT SR: delta nostro adattato - contro**

| seme | contro | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|---|
| s1234 | ICP + Chamfer | +0.066 [+0.006, +0.120] | +0.082 [+0.029, +0.131] | +0.114 [+0.066, +0.162] | +0.152 [+0.110, +0.198] |
| s1234 | NoW col rilevatore | +0.032 [-0.029, +0.083] | +0.048 [-0.009, +0.099] | +0.081 [+0.027, +0.131] | +0.109 [+0.064, +0.160] |
| s1234 | fit 3DMM vB | +0.027 [-0.036, +0.079] | +0.049 [-0.014, +0.101] | +0.056 [-0.010, +0.110] | +0.061 [-0.010, +0.138] |
| s1234 | NoW coi landmark esatti (descrittivo) | +0.027 [-0.031, +0.077] | +0.044 [-0.008, +0.089] | +0.076 [+0.026, +0.124] | +0.106 [+0.061, +0.154] |
| s1234 | miglior famiglia (post hoc) | +0.027 [-0.036, +0.079] (fit 3DMM vB) | +0.048 [-0.009, +0.099] (NoW col rilevatore) | +0.056 [-0.010, +0.110] (fit 3DMM vB) | +0.061 [-0.010, +0.138] (fit 3DMM vB) |
| s1234 | migliore fra tutte (scelto su A) | +0.026 [-0.036, +0.074] | +0.047 [-0.010, +0.095] | +0.051 [-0.013, +0.103] | +0.061 [-0.010, +0.138] |
| s1234 | nostro non adattato | +0.026 [-0.012, +0.066] | +0.044 [+0.008, +0.082] | +0.075 [+0.035, +0.120] | +0.112 [+0.063, +0.165] |
| s1234 | testa k = 0 | +0.001 [-0.005, +0.007] | +0.018 [+0.007, +0.031] | +0.046 [+0.026, +0.068] | +0.091 [+0.046, +0.151] |
| s2345 | ICP + Chamfer | +0.067 [+0.016, +0.124] | +0.077 [+0.028, +0.131] | +0.103 [+0.056, +0.155] | +0.124 [+0.085, +0.173] |
| s2345 | NoW col rilevatore | +0.034 [-0.020, +0.086] | +0.044 [-0.007, +0.095] | +0.070 [+0.018, +0.128] | +0.081 [+0.035, +0.137] |
| s2345 | fit 3DMM vB | +0.028 [-0.030, +0.080] | +0.045 [-0.009, +0.096] | +0.046 [-0.019, +0.106] | +0.033 [-0.045, +0.123] |
| s2345 | NoW coi landmark esatti (descrittivo) | +0.029 [-0.020, +0.081] | +0.039 [-0.008, +0.091] | +0.065 [+0.018, +0.118] | +0.078 [+0.035, +0.134] |
| s2345 | miglior famiglia (post hoc) | +0.028 [-0.030, +0.080] (fit 3DMM vB) | +0.044 [-0.007, +0.095] (NoW col rilevatore) | +0.046 [-0.019, +0.106] (fit 3DMM vB) | +0.033 [-0.045, +0.123] (fit 3DMM vB) |
| s2345 | migliore fra tutte (scelto su A) | +0.027 [-0.026, +0.078] | +0.043 [-0.006, +0.093] | +0.041 [-0.023, +0.100] | +0.033 [-0.045, +0.123] |
| s2345 | nostro non adattato | +0.034 [-0.007, +0.083] | +0.046 [+0.002, +0.098] | +0.074 [+0.025, +0.134] | +0.088 [+0.041, +0.142] |
| s2345 | testa k = 0 | +0.006 [-0.001, +0.013] | +0.016 [+0.005, +0.027] | +0.038 [+0.020, +0.059] | +0.073 [+0.037, +0.119] |

Configurazioni scelte sulle k identita' (GT SR, estrazioni per configurazione):

- ICP + Chamfer: k = 5: cs 5; k = 10: cs 5; k = 20: cs 5; k = 50: cs 5
- NoW col rilevatore: k = 5: originale 5; k = 10: originale 5; k = 20: originale 5; k = 50: originale 5
- fit 3DMM vB: k = 5: gnm|sr 3, gnm|coef 1, flame2023|sr 1; k = 10: gnm|sr 3, gnm|coef 1, flame2023|sr 1; k = 20: gnm|coef 3, gnm|sr 1, flame2023|sr 1; k = 50: gnm|coef 5
- NoW coi landmark esatti (descrittivo): k = 5: originale 5; k = 10: originale 5; k = 20: originale 5; k = 50: originale 5
- tutte le famiglie: k = 5: fit_vb:gnm|sr 2, fit_vb:gnm|coef 1, now_det:originale 1, fit_vb:flame2023|sr 1; k = 10: fit_vb:gnm|sr 2, fit_vb:gnm|coef 1, now_det:originale 1, fit_vb:flame2023|sr 1; k = 20: fit_vb:gnm|coef 3, fit_vb:gnm|sr 1, now_det:originale 1; k = 50: fit_vb:gnm|coef 5

## FaceScape dev

**GT FR: rho sulle identita' di test**

| colonna | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|
| nostro adattato s1234 | 0.704 [0.634, 0.764] sd 0.008 | 0.715 [0.643, 0.774] sd 0.004 | 0.729 [0.655, 0.788] sd 0.014 | 0.744 [0.662, 0.803] sd 0.026 |
| nostro adattato s2345 | 0.695 [0.626, 0.759] sd 0.006 | 0.702 [0.633, 0.765] sd 0.009 | 0.712 [0.643, 0.772] sd 0.019 | 0.737 [0.662, 0.796] sd 0.017 |
| nostro non adattato s1234 | 0.663 [0.588, 0.731] sd 0.004 | 0.662 [0.585, 0.731] sd 0.009 | 0.663 [0.585, 0.730] sd 0.010 | 0.669 [0.582, 0.737] sd 0.022 |
| nostro non adattato s2345 | 0.670 [0.596, 0.735] sd 0.002 | 0.670 [0.593, 0.737] sd 0.012 | 0.673 [0.594, 0.737] sd 0.015 | 0.681 [0.597, 0.746] sd 0.023 |
| testa k = 0 s1234 | 0.690 [0.620, 0.750] sd 0.002 | 0.688 [0.614, 0.750] sd 0.005 | 0.685 [0.607, 0.747] sd 0.007 | 0.677 [0.592, 0.740] sd 0.022 |
| testa k = 0 s2345 | 0.687 [0.615, 0.751] sd 0.002 | 0.687 [0.614, 0.753] sd 0.007 | 0.689 [0.613, 0.752] sd 0.014 | 0.687 [0.605, 0.754] sd 0.024 |
| concorrente adattato ICP + Chamfer | 0.467 [0.400, 0.529] sd 0.002 | 0.466 [0.396, 0.530] sd 0.006 | 0.465 [0.393, 0.527] sd 0.011 | 0.468 [0.391, 0.534] sd 0.020 |
| concorrente adattato NoW col rilevatore | 0.674 [0.605, 0.732] sd 0.002 | 0.675 [0.603, 0.733] sd 0.007 | 0.672 [0.596, 0.731] sd 0.009 | 0.673 [0.594, 0.732] sd 0.020 |
| concorrente adattato fit 3DMM vB | 0.735 [0.665, 0.792] sd 0.042 | 0.760 [0.709, 0.804] sd 0.101 | 0.810 [0.756, 0.854] sd 0.014 | 0.821 [0.764, 0.867] sd 0.022 |
| concorrente adattato NoW coi landmark esatti (descrittivo) | 0.675 [0.606, 0.733] sd 0.003 | 0.676 [0.606, 0.735] sd 0.008 | 0.660 [0.585, 0.724] sd 0.031 | 0.676 [0.596, 0.735] sd 0.024 |
| concorrente adattato migliore fra tutte (scelto su A) | 0.735 [0.665, 0.792] sd 0.042 | 0.731 [0.677, 0.779] sd 0.098 | 0.810 [0.756, 0.854] sd 0.014 | 0.821 [0.764, 0.867] sd 0.022 |

**GT FR: delta nostro adattato - contro**

| seme | contro | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|---|
| s1234 | ICP + Chamfer | +0.237 [+0.191, +0.286] | +0.249 [+0.203, +0.299] | +0.264 [+0.216, +0.318] | +0.276 [+0.226, +0.333] |
| s1234 | NoW col rilevatore | +0.030 [-0.022, +0.087] | +0.040 [-0.013, +0.098] | +0.057 [-0.002, +0.118] | +0.071 [+0.013, +0.130] |
| s1234 | fit 3DMM vB | -0.031 [-0.082, +0.015] | -0.045 [-0.089, -0.004] | -0.081 [-0.127, -0.037] | -0.076 [-0.137, -0.027] |
| s1234 | NoW coi landmark esatti (descrittivo) | +0.029 [-0.022, +0.085] | +0.039 [-0.013, +0.096] | +0.068 [+0.013, +0.130] | +0.068 [+0.012, +0.128] |
| s1234 | miglior famiglia (post hoc) | -0.031 [-0.082, +0.015] (fit 3DMM vB) | -0.045 [-0.089, -0.004] (fit 3DMM vB) | -0.081 [-0.127, -0.037] (fit 3DMM vB) | -0.076 [-0.137, -0.027] (fit 3DMM vB) |
| s1234 | migliore fra tutte (scelto su A) | -0.031 [-0.082, +0.015] | -0.016 [-0.060, +0.024] | -0.081 [-0.127, -0.037] | -0.076 [-0.137, -0.027] |
| s1234 | nostro non adattato | +0.041 [+0.015, +0.069] | +0.053 [+0.027, +0.083] | +0.066 [+0.038, +0.096] | +0.075 [+0.042, +0.111] |
| s1234 | testa k = 0 | +0.014 [+0.009, +0.019] | +0.026 [+0.019, +0.035] | +0.044 [+0.033, +0.057] | +0.068 [+0.051, +0.085] |
| s2345 | ICP + Chamfer | +0.227 [+0.177, +0.284] | +0.236 [+0.186, +0.292] | +0.247 [+0.197, +0.303] | +0.269 [+0.222, +0.324] |
| s2345 | NoW col rilevatore | +0.021 [-0.042, +0.088] | +0.028 [-0.032, +0.092] | +0.040 [-0.024, +0.106] | +0.064 [+0.010, +0.122] |
| s2345 | fit 3DMM vB | -0.040 [-0.097, +0.010] | -0.058 [-0.107, -0.015] | -0.098 [-0.151, -0.052] | -0.084 [-0.136, -0.033] |
| s2345 | NoW coi landmark esatti (descrittivo) | +0.019 [-0.041, +0.083] | +0.026 [-0.033, +0.088] | +0.051 [-0.010, +0.116] | +0.061 [+0.005, +0.121] |
| s2345 | miglior famiglia (post hoc) | -0.040 [-0.097, +0.010] (fit 3DMM vB) | -0.058 [-0.107, -0.015] (fit 3DMM vB) | -0.098 [-0.151, -0.052] (fit 3DMM vB) | -0.084 [-0.136, -0.033] (fit 3DMM vB) |
| s2345 | migliore fra tutte (scelto su A) | -0.040 [-0.097, +0.010] | -0.029 [-0.079, +0.017] | -0.098 [-0.151, -0.052] | -0.084 [-0.136, -0.033] |
| s2345 | nostro non adattato | +0.025 [+0.003, +0.049] | +0.032 [+0.010, +0.056] | +0.039 [+0.017, +0.062] | +0.056 [+0.029, +0.084] |
| s2345 | testa k = 0 | +0.008 [+0.003, +0.013] | +0.015 [+0.008, +0.023] | +0.023 [+0.013, +0.035] | +0.050 [+0.032, +0.071] |

Configurazioni scelte sulle k identita' (GT FR, estrazioni per configurazione):

- ICP + Chamfer: k = 5: mm 5; k = 10: mm 5; k = 20: mm 5; k = 50: mm 5
- NoW col rilevatore: k = 5: metrical 5; k = 10: metrical 5; k = 20: metrical 5; k = 50: metrical 5
- fit 3DMM vB: k = 5: flame2023|comp 3, gnm|fr 1, gnm|comp 1; k = 10: gnm|comp 4, gnm|coef 1; k = 20: gnm|comp 5; k = 50: gnm|comp 5
- NoW coi landmark esatti (descrittivo): k = 5: metrical 5; k = 10: metrical 5; k = 20: metrical 4, originale 1; k = 50: metrical 5
- tutte le famiglie: k = 5: fit_vb:flame2023|comp 3, fit_vb:gnm|fr 1, fit_vb:gnm|comp 1; k = 10: fit_vb:gnm|comp 3, fit_vb:gnm|coef 1, now_det:metrical 1; k = 20: fit_vb:gnm|comp 5; k = 50: fit_vb:gnm|comp 5

**GT SR: rho sulle identita' di test**

| colonna | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|
| nostro adattato s1234 | 0.792 [0.736, 0.838] sd 0.008 | 0.813 [0.762, 0.856] sd 0.013 | 0.829 [0.779, 0.872] sd 0.018 | 0.860 [0.809, 0.896] sd 0.021 |
| nostro adattato s2345 | 0.780 [0.720, 0.834] sd 0.006 | 0.797 [0.737, 0.847] sd 0.009 | 0.810 [0.752, 0.858] sd 0.015 | 0.851 [0.799, 0.891] sd 0.019 |
| nostro non adattato s1234 | 0.750 [0.681, 0.808] sd 0.005 | 0.751 [0.679, 0.809] sd 0.008 | 0.750 [0.679, 0.805] sd 0.012 | 0.757 [0.677, 0.814] sd 0.022 |
| nostro non adattato s2345 | 0.755 [0.686, 0.815] sd 0.003 | 0.757 [0.687, 0.817] sd 0.008 | 0.757 [0.685, 0.817] sd 0.012 | 0.769 [0.694, 0.824] sd 0.025 |
| testa k = 0 s1234 | 0.782 [0.727, 0.829] sd 0.004 | 0.784 [0.728, 0.831] sd 0.004 | 0.780 [0.720, 0.828] sd 0.012 | 0.779 [0.712, 0.830] sd 0.028 |
| testa k = 0 s2345 | 0.772 [0.707, 0.829] sd 0.003 | 0.773 [0.706, 0.830] sd 0.006 | 0.773 [0.707, 0.827] sd 0.013 | 0.778 [0.705, 0.833] sd 0.028 |
| concorrente adattato ICP + Chamfer | 0.493 [0.426, 0.553] sd 0.002 | 0.493 [0.423, 0.554] sd 0.003 | 0.491 [0.422, 0.552] sd 0.016 | 0.496 [0.425, 0.559] sd 0.024 |
| concorrente adattato NoW col rilevatore | 0.702 [0.642, 0.755] sd 0.001 | 0.711 [0.651, 0.764] sd 0.009 | 0.712 [0.650, 0.764] sd 0.013 | 0.722 [0.659, 0.773] sd 0.020 |
| concorrente adattato fit 3DMM vB | 0.834 [0.775, 0.880] sd 0.020 | 0.803 [0.755, 0.843] sd 0.098 | 0.859 [0.805, 0.900] sd 0.008 | 0.878 [0.829, 0.912] sd 0.013 |
| concorrente adattato NoW coi landmark esatti (descrittivo) | 0.713 [0.655, 0.764] sd 0.012 | 0.717 [0.657, 0.771] sd 0.012 | 0.719 [0.659, 0.771] sd 0.016 | 0.730 [0.667, 0.782] sd 0.017 |
| concorrente adattato migliore fra tutte (scelto su A) | 0.810 [0.757, 0.854] sd 0.064 | 0.774 [0.725, 0.814] sd 0.098 | 0.859 [0.805, 0.900] sd 0.008 | 0.878 [0.829, 0.912] sd 0.013 |

**GT SR: delta nostro adattato - contro**

| seme | contro | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|---|
| s1234 | ICP + Chamfer | +0.298 [+0.250, +0.344] | +0.320 [+0.271, +0.371] | +0.338 [+0.289, +0.390] | +0.364 [+0.313, +0.420] |
| s1234 | NoW col rilevatore | +0.090 [+0.039, +0.140] | +0.102 [+0.054, +0.150] | +0.117 [+0.068, +0.167] | +0.138 [+0.092, +0.189] |
| s1234 | fit 3DMM vB | -0.042 [-0.086, +0.000] | +0.010 [-0.025, +0.044] | -0.030 [-0.066, +0.006] | -0.018 [-0.058, +0.019] |
| s1234 | NoW coi landmark esatti (descrittivo) | +0.079 [+0.031, +0.126] | +0.096 [+0.049, +0.142] | +0.110 [+0.063, +0.159] | +0.130 [+0.084, +0.180] |
| s1234 | miglior famiglia (post hoc) | -0.042 [-0.086, +0.000] (fit 3DMM vB) | +0.010 [-0.025, +0.044] (fit 3DMM vB) | -0.030 [-0.066, +0.006] (fit 3DMM vB) | -0.018 [-0.058, +0.019] (fit 3DMM vB) |
| s1234 | migliore fra tutte (scelto su A) | -0.019 [-0.057, +0.018] | +0.039 [+0.006, +0.071] | -0.030 [-0.066, +0.006] | -0.018 [-0.058, +0.019] |
| s1234 | nostro non adattato | +0.041 [+0.012, +0.074] | +0.062 [+0.034, +0.094] | +0.079 [+0.053, +0.111] | +0.103 [+0.072, +0.143] |
| s1234 | testa k = 0 | +0.009 [+0.005, +0.014] | +0.029 [+0.021, +0.039] | +0.049 [+0.037, +0.063] | +0.081 [+0.062, +0.106] |
| s2345 | ICP + Chamfer | +0.287 [+0.233, +0.341] | +0.304 [+0.255, +0.357] | +0.319 [+0.267, +0.375] | +0.355 [+0.305, +0.413] |
| s2345 | NoW col rilevatore | +0.079 [+0.026, +0.137] | +0.086 [+0.036, +0.141] | +0.098 [+0.044, +0.156] | +0.129 [+0.083, +0.183] |
| s2345 | fit 3DMM vB | -0.054 [-0.100, -0.011] | -0.006 [-0.044, +0.029] | -0.049 [-0.091, -0.011] | -0.027 [-0.065, +0.008] |
| s2345 | NoW coi landmark esatti (descrittivo) | +0.068 [+0.014, +0.124] | +0.080 [+0.032, +0.133] | +0.091 [+0.038, +0.147] | +0.121 [+0.076, +0.173] |
| s2345 | miglior famiglia (post hoc) | -0.054 [-0.100, -0.011] (fit 3DMM vB) | -0.006 [-0.044, +0.029] (fit 3DMM vB) | -0.049 [-0.091, -0.011] (fit 3DMM vB) | -0.027 [-0.065, +0.008] (fit 3DMM vB) |
| s2345 | migliore fra tutte (scelto su A) | -0.030 [-0.071, +0.009] | +0.023 [-0.012, +0.058] | -0.049 [-0.091, -0.011] | -0.027 [-0.065, +0.008] |
| s2345 | nostro non adattato | +0.025 [-0.002, +0.055] | +0.040 [+0.014, +0.068] | +0.053 [+0.028, +0.083] | +0.082 [+0.053, +0.116] |
| s2345 | testa k = 0 | +0.008 [+0.003, +0.013] | +0.024 [+0.015, +0.034] | +0.037 [+0.025, +0.051] | +0.072 [+0.052, +0.099] |

Configurazioni scelte sulle k identita' (GT SR, estrazioni per configurazione):

- ICP + Chamfer: k = 5: mm 5; k = 10: mm 3, cs 2; k = 20: cs 3, mm 2; k = 50: mm 3, cs 2
- NoW col rilevatore: k = 5: metrical 5; k = 10: originale 3, metrical 2; k = 20: originale 4, metrical 1; k = 50: originale 5
- fit 3DMM vB: k = 5: flame2023|sr 3, gnm|sr 2; k = 10: gnm|sr 3, gnm|coef 1, flame2023|sr 1; k = 20: gnm|sr 5; k = 50: gnm|sr 5
- NoW coi landmark esatti (descrittivo): k = 5: metrical 3, originale 2; k = 10: originale 3, metrical 2; k = 20: originale 4, metrical 1; k = 50: originale 5
- tutte le famiglie: k = 5: fit_vb:flame2023|sr 2, fit_vb:gnm|sr 2, now_det:metrical 1; k = 10: fit_vb:gnm|sr 2, fit_vb:gnm|coef 1, now_det:originale 1, fit_vb:flame2023|sr 1; k = 20: fit_vb:gnm|sr 5; k = 50: fit_vb:gnm|sr 5

## FaceVerse con espressioni

**GT FR: rho sulle identita' di test**

| colonna | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|
| nostro adattato s1234 | 0.321 [0.251, 0.387] sd 0.009 | 0.329 [0.262, 0.394] sd 0.014 | 0.357 [0.296, 0.419] sd 0.010 | 0.371 [0.295, 0.444] sd 0.026 |
| nostro adattato s2345 | 0.288 [0.219, 0.353] sd 0.009 | 0.296 [0.232, 0.358] sd 0.008 | 0.312 [0.251, 0.367] sd 0.030 | 0.344 [0.272, 0.410] sd 0.059 |
| nostro non adattato s1234 | 0.307 [0.232, 0.377] sd 0.008 | 0.309 [0.237, 0.379] sd 0.011 | 0.325 [0.256, 0.391] sd 0.012 | 0.306 [0.225, 0.387] sd 0.028 |
| nostro non adattato s2345 | 0.321 [0.252, 0.379] sd 0.006 | 0.321 [0.253, 0.381] sd 0.011 | 0.334 [0.269, 0.391] sd 0.010 | 0.311 [0.237, 0.385] sd 0.021 |
| testa k = 0 s1234 | 0.314 [0.239, 0.382] sd 0.005 | 0.314 [0.245, 0.381] sd 0.002 | 0.322 [0.256, 0.384] sd 0.014 | 0.302 [0.222, 0.374] sd 0.020 |
| testa k = 0 s2345 | 0.277 [0.207, 0.345] sd 0.004 | 0.277 [0.209, 0.341] sd 0.004 | 0.292 [0.229, 0.351] sd 0.015 | 0.285 [0.212, 0.355] sd 0.038 |
| concorrente adattato ICP + Chamfer | 0.304 [0.235, 0.369] sd 0.026 | 0.325 [0.258, 0.390] sd 0.031 | 0.318 [0.250, 0.381] sd 0.037 | 0.323 [0.237, 0.396] sd 0.047 |
| concorrente adattato NoW col rilevatore | 0.218 [0.160, 0.270] sd 0.008 | 0.219 [0.161, 0.270] sd 0.011 | 0.220 [0.164, 0.273] sd 0.019 | 0.205 [0.143, 0.260] sd 0.032 |
| concorrente adattato fit 3DMM vB | 0.293 [0.218, 0.355] sd 0.057 | 0.281 [0.205, 0.338] sd 0.054 | 0.262 [0.185, 0.325] sd 0.088 | 0.342 [0.254, 0.420] sd 0.033 |
| concorrente adattato NoW coi landmark esatti (descrittivo) | 0.318 [0.249, 0.381] sd 0.012 | 0.329 [0.263, 0.390] sd 0.021 | 0.342 [0.274, 0.405] sd 0.030 | 0.327 [0.248, 0.395] sd 0.042 |
| concorrente adattato migliore fra tutte (scelto su A) | 0.291 [0.224, 0.350] sd 0.056 | 0.281 [0.205, 0.338] sd 0.054 | 0.281 [0.213, 0.335] sd 0.078 | 0.313 [0.233, 0.383] sd 0.043 |

**GT FR: delta nostro adattato - contro**

| seme | contro | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|---|
| s1234 | ICP + Chamfer | +0.017 [-0.061, +0.095] | +0.004 [-0.068, +0.076] | +0.040 [-0.033, +0.114] | +0.047 [-0.027, +0.127] |
| s1234 | NoW col rilevatore | +0.104 [+0.033, +0.174] | +0.110 [+0.032, +0.184] | +0.137 [+0.064, +0.207] | +0.166 [+0.079, +0.248] |
| s1234 | fit 3DMM vB | +0.028 [-0.055, +0.110] | +0.048 [-0.030, +0.136] | +0.095 [+0.019, +0.182] | +0.028 [-0.055, +0.131] |
| s1234 | NoW coi landmark esatti (descrittivo) | +0.004 [-0.067, +0.076] | +0.001 [-0.068, +0.072] | +0.016 [-0.051, +0.092] | +0.044 [-0.029, +0.129] |
| s1234 | miglior famiglia (post hoc) | +0.017 [-0.061, +0.095] (ICP + Chamfer) | +0.004 [-0.068, +0.076] (ICP + Chamfer) | +0.040 [-0.033, +0.114] (ICP + Chamfer) | +0.028 [-0.055, +0.131] (fit 3DMM vB) |
| s1234 | migliore fra tutte (scelto su A) | +0.030 [-0.042, +0.105] | +0.048 [-0.030, +0.136] | +0.077 [+0.009, +0.157] | +0.058 [-0.022, +0.148] |
| s1234 | nostro non adattato | +0.014 [-0.034, +0.059] | +0.020 [-0.024, +0.062] | +0.032 [-0.011, +0.070] | +0.065 [+0.019, +0.106] |
| s1234 | testa k = 0 | +0.008 [-0.004, +0.019] | +0.015 [-0.005, +0.034] | +0.035 [+0.013, +0.057] | +0.069 [+0.037, +0.103] |
| s2345 | ICP + Chamfer | -0.016 [-0.095, +0.064] | -0.029 [-0.102, +0.044] | -0.006 [-0.083, +0.070] | +0.020 [-0.056, +0.098] |
| s2345 | NoW col rilevatore | +0.071 [+0.006, +0.134] | +0.077 [+0.009, +0.141] | +0.092 [+0.024, +0.158] | +0.139 [+0.058, +0.215] |
| s2345 | fit 3DMM vB | -0.005 [-0.078, +0.071] | +0.015 [-0.058, +0.095] | +0.050 [-0.029, +0.132] | +0.001 [-0.076, +0.088] |
| s2345 | NoW coi landmark esatti (descrittivo) | -0.029 [-0.102, +0.039] | -0.033 [-0.106, +0.037] | -0.029 [-0.100, +0.042] | +0.016 [-0.061, +0.095] |
| s2345 | miglior famiglia (post hoc) | -0.016 [-0.095, +0.064] (ICP + Chamfer) | -0.029 [-0.102, +0.044] (ICP + Chamfer) | -0.006 [-0.083, +0.070] (ICP + Chamfer) | +0.001 [-0.076, +0.088] (fit 3DMM vB) |
| s2345 | migliore fra tutte (scelto su A) | -0.002 [-0.075, +0.070] | +0.015 [-0.058, +0.095] | +0.032 [-0.044, +0.107] | +0.031 [-0.047, +0.111] |
| s2345 | nostro non adattato | -0.033 [-0.087, +0.021] | -0.025 [-0.080, +0.025] | -0.022 [-0.073, +0.028] | +0.033 [-0.019, +0.084] |
| s2345 | testa k = 0 | +0.011 [-0.002, +0.024] | +0.019 [+0.003, +0.034] | +0.020 [-0.003, +0.042] | +0.058 [+0.030, +0.082] |

Configurazioni scelte sulle k identita' (GT FR, estrazioni per configurazione):

- ICP + Chamfer: k = 5: cs 4, mm 1; k = 10: mm 3, cs 2; k = 20: cs 3, mm 2; k = 50: mm 4, cs 1
- NoW col rilevatore: k = 5: originale 3, metrical 2; k = 10: originale 3, metrical 2; k = 20: metrical 3, originale 2; k = 50: metrical 3, originale 2
- fit 3DMM vB: k = 5: gnm|comp 2, gnm|fr 1, flame2023|comp 1, flame2023|coef 1; k = 10: gnm|comp 2, gnm|coef 2, gnm|fr 1; k = 20: gnm|coef 2, gnm|comp 2, flame2023|coef 1; k = 50: gnm|comp 5
- NoW coi landmark esatti (descrittivo): k = 5: originale 3, metrical 2; k = 10: metrical 3, originale 2; k = 20: metrical 4, originale 1; k = 50: metrical 4, originale 1
- tutte le famiglie: k = 5: icp:cs 2, fit_vb:gnm|comp 2, fit_vb:flame2023|coef 1; k = 10: fit_vb:gnm|comp 2, fit_vb:gnm|coef 2, fit_vb:gnm|fr 1; k = 20: fit_vb:gnm|coef 2, icp:cs 1, icp:mm 1, fit_vb:gnm|comp 1; k = 50: icp:mm 2, fit_vb:gnm|comp 2, icp:cs 1

**GT SR: rho sulle identita' di test**

| colonna | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|
| nostro adattato s1234 | 0.303 [0.225, 0.371] sd 0.008 | 0.317 [0.247, 0.379] sd 0.016 | 0.340 [0.274, 0.401] sd 0.012 | 0.362 [0.284, 0.426] sd 0.032 |
| nostro adattato s2345 | 0.276 [0.206, 0.346] sd 0.010 | 0.286 [0.218, 0.346] sd 0.010 | 0.296 [0.230, 0.355] sd 0.040 | 0.330 [0.258, 0.394] sd 0.072 |
| nostro non adattato s1234 | 0.292 [0.220, 0.356] sd 0.005 | 0.296 [0.230, 0.358] sd 0.013 | 0.301 [0.233, 0.361] sd 0.010 | 0.273 [0.201, 0.347] sd 0.016 |
| nostro non adattato s2345 | 0.317 [0.240, 0.382] sd 0.008 | 0.317 [0.242, 0.381] sd 0.013 | 0.322 [0.251, 0.384] sd 0.012 | 0.285 [0.208, 0.358] sd 0.014 |
| testa k = 0 s1234 | 0.293 [0.216, 0.365] sd 0.005 | 0.296 [0.224, 0.363] sd 0.008 | 0.293 [0.222, 0.360] sd 0.019 | 0.269 [0.188, 0.334] sd 0.036 |
| testa k = 0 s2345 | 0.265 [0.193, 0.334] sd 0.007 | 0.263 [0.194, 0.330] sd 0.006 | 0.270 [0.203, 0.332] sd 0.020 | 0.261 [0.194, 0.325] sd 0.043 |
| concorrente adattato ICP + Chamfer | 0.313 [0.239, 0.378] sd 0.010 | 0.311 [0.240, 0.376] sd 0.012 | 0.316 [0.237, 0.384] sd 0.012 | 0.295 [0.210, 0.365] sd 0.035 |
| concorrente adattato NoW col rilevatore | 0.222 [0.161, 0.275] sd 0.011 | 0.221 [0.161, 0.274] sd 0.014 | 0.219 [0.162, 0.272] sd 0.019 | 0.198 [0.131, 0.253] sd 0.029 |
| concorrente adattato fit 3DMM vB | 0.287 [0.212, 0.355] sd 0.056 | 0.248 [0.171, 0.312] sd 0.057 | 0.254 [0.172, 0.320] sd 0.053 | 0.300 [0.210, 0.377] sd 0.064 |
| concorrente adattato NoW coi landmark esatti (descrittivo) | 0.312 [0.240, 0.376] sd 0.012 | 0.315 [0.247, 0.377] sd 0.013 | 0.319 [0.253, 0.383] sd 0.013 | 0.301 [0.217, 0.371] sd 0.034 |
| concorrente adattato migliore fra tutte (scelto su A) | 0.288 [0.220, 0.351] sd 0.057 | 0.244 [0.169, 0.305] sd 0.050 | 0.277 [0.203, 0.338] sd 0.056 | 0.296 [0.203, 0.370] sd 0.065 |

**GT SR: delta nostro adattato - contro**

| seme | contro | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|---|
| s1234 | ICP + Chamfer | -0.010 [-0.082, +0.065] | +0.005 [-0.067, +0.077] | +0.024 [-0.050, +0.103] | +0.067 [-0.016, +0.149] |
| s1234 | NoW col rilevatore | +0.081 [+0.008, +0.146] | +0.096 [+0.025, +0.162] | +0.120 [+0.047, +0.182] | +0.164 [+0.083, +0.239] |
| s1234 | fit 3DMM vB | +0.016 [-0.062, +0.094] | +0.068 [-0.011, +0.154] | +0.086 [+0.004, +0.172] | +0.062 [-0.025, +0.157] |
| s1234 | NoW coi landmark esatti (descrittivo) | -0.009 [-0.080, +0.062] | +0.002 [-0.067, +0.074] | +0.021 [-0.051, +0.098] | +0.062 [-0.018, +0.151] |
| s1234 | miglior famiglia (post hoc) | -0.010 [-0.082, +0.065] (ICP + Chamfer) | +0.005 [-0.067, +0.077] (ICP + Chamfer) | +0.024 [-0.050, +0.103] (ICP + Chamfer) | +0.062 [-0.025, +0.157] (fit 3DMM vB) |
| s1234 | migliore fra tutte (scelto su A) | +0.015 [-0.060, +0.091] | +0.073 [-0.007, +0.156] | +0.063 [-0.008, +0.140] | +0.066 [-0.019, +0.155] |
| s1234 | nostro non adattato | +0.011 [-0.042, +0.060] | +0.021 [-0.029, +0.067] | +0.039 [-0.011, +0.085] | +0.089 [+0.035, +0.140] |
| s1234 | testa k = 0 | +0.010 [-0.003, +0.023] | +0.021 [+0.002, +0.042] | +0.047 [+0.022, +0.074] | +0.093 [+0.055, +0.133] |
| s2345 | ICP + Chamfer | -0.036 [-0.117, +0.046] | -0.026 [-0.108, +0.053] | -0.020 [-0.101, +0.060] | +0.035 [-0.043, +0.122] |
| s2345 | NoW col rilevatore | +0.054 [-0.014, +0.121] | +0.064 [-0.006, +0.133] | +0.077 [+0.006, +0.144] | +0.132 [+0.052, +0.210] |
| s2345 | fit 3DMM vB | -0.011 [-0.087, +0.065] | +0.037 [-0.045, +0.117] | +0.042 [-0.037, +0.123] | +0.029 [-0.050, +0.114] |
| s2345 | NoW coi landmark esatti (descrittivo) | -0.035 [-0.114, +0.038] | -0.029 [-0.105, +0.044] | -0.023 [-0.097, +0.053] | +0.029 [-0.050, +0.114] |
| s2345 | miglior famiglia (post hoc) | -0.036 [-0.117, +0.046] (ICP + Chamfer) | -0.026 [-0.108, +0.053] (ICP + Chamfer) | -0.020 [-0.101, +0.060] (ICP + Chamfer) | +0.029 [-0.050, +0.114] (fit 3DMM vB) |
| s2345 | migliore fra tutte (scelto su A) | -0.012 [-0.088, +0.064] | +0.041 [-0.039, +0.124] | +0.019 [-0.056, +0.094] | +0.034 [-0.044, +0.113] |
| s2345 | nostro non adattato | -0.041 [-0.108, +0.028] | -0.031 [-0.098, +0.032] | -0.026 [-0.090, +0.034] | +0.045 [-0.023, +0.109] |
| s2345 | testa k = 0 | +0.012 [-0.004, +0.025] | +0.023 [+0.004, +0.040] | +0.026 [-0.001, +0.049] | +0.069 [+0.037, +0.095] |

Configurazioni scelte sulle k identita' (GT SR, estrazioni per configurazione):

- ICP + Chamfer: k = 5: cs 4, mm 1; k = 10: cs 4, mm 1; k = 20: cs 5; k = 50: mm 3, cs 2
- NoW col rilevatore: k = 5: originale 3, metrical 2; k = 10: originale 3, metrical 2; k = 20: originale 3, metrical 2; k = 50: originale 3, metrical 2
- fit 3DMM vB: k = 5: flame2023|sr 2, gnm|sr 2, flame2023|coef 1; k = 10: gnm|coef 2, gnm|sr 1, flame2023|sr 1, flame2023|coef 1; k = 20: flame2023|sr 3, gnm|coef 2; k = 50: gnm|sr 4, flame2023|sr 1
- NoW coi landmark esatti (descrittivo): k = 5: originale 3, metrical 2; k = 10: originale 3, metrical 2; k = 20: originale 3, metrical 2; k = 50: metrical 3, originale 2
- tutte le famiglie: k = 5: icp:cs 3, fit_vb:gnm|sr 1, fit_vb:flame2023|coef 1; k = 10: fit_vb:gnm|coef 2, icp:cs 1, fit_vb:flame2023|sr 1, fit_vb:flame2023|coef 1; k = 20: icp:cs 2, fit_vb:flame2023|sr 2, fit_vb:gnm|coef 1; k = 50: fit_vb:gnm|sr 3, icp:cs 1, fit_vb:flame2023|sr 1

## FaceVerse neutra

**GT FR: rho sulle identita' di test**

| colonna | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|
| nostro adattato s1234 | 0.370 [0.290, 0.445] sd 0.012 | 0.382 [0.303, 0.456] sd 0.014 | 0.379 [0.301, 0.448] sd 0.035 | 0.415 [0.326, 0.491] sd 0.061 |
| nostro adattato s2345 | 0.354 [0.270, 0.426] sd 0.011 | 0.361 [0.278, 0.431] sd 0.012 | 0.379 [0.299, 0.445] sd 0.018 | 0.426 [0.347, 0.493] sd 0.055 |
| nostro non adattato s1234 | 0.338 [0.258, 0.416] sd 0.009 | 0.336 [0.257, 0.413] sd 0.009 | 0.331 [0.249, 0.412] sd 0.028 | 0.338 [0.239, 0.430] sd 0.038 |
| nostro non adattato s2345 | 0.375 [0.304, 0.441] sd 0.007 | 0.375 [0.302, 0.443] sd 0.006 | 0.368 [0.299, 0.436] sd 0.029 | 0.369 [0.288, 0.444] sd 0.060 |
| testa k = 0 s1234 | 0.366 [0.280, 0.443] sd 0.008 | 0.366 [0.280, 0.446] sd 0.011 | 0.363 [0.277, 0.442] sd 0.018 | 0.358 [0.258, 0.450] sd 0.046 |
| testa k = 0 s2345 | 0.344 [0.260, 0.421] sd 0.009 | 0.347 [0.262, 0.422] sd 0.007 | 0.352 [0.266, 0.427] sd 0.028 | 0.362 [0.275, 0.441] sd 0.065 |
| concorrente adattato ICP + Chamfer | 0.403 [0.324, 0.472] sd 0.018 | 0.384 [0.303, 0.457] sd 0.011 | 0.391 [0.316, 0.457] sd 0.031 | 0.370 [0.281, 0.451] sd 0.030 |
| concorrente adattato NoW col rilevatore | 0.280 [0.210, 0.341] sd 0.009 | 0.284 [0.214, 0.346] sd 0.006 | 0.269 [0.199, 0.334] sd 0.021 | 0.260 [0.177, 0.326] sd 0.029 |
| concorrente adattato fit 3DMM vB | 0.299 [0.217, 0.367] sd 0.052 | 0.311 [0.228, 0.381] sd 0.062 | 0.308 [0.225, 0.379] sd 0.048 | 0.332 [0.229, 0.421] sd 0.038 |
| concorrente adattato NoW coi landmark esatti (descrittivo) | 0.393 [0.320, 0.463] sd 0.009 | 0.395 [0.319, 0.466] sd 0.010 | 0.379 [0.307, 0.450] sd 0.024 | 0.363 [0.268, 0.446] sd 0.040 |
| concorrente adattato migliore fra tutte (scelto su A) | 0.319 [0.249, 0.381] sd 0.071 | 0.326 [0.254, 0.394] sd 0.067 | 0.357 [0.289, 0.416] sd 0.075 | 0.370 [0.281, 0.451] sd 0.030 |

**GT FR: delta nostro adattato - contro**

| seme | contro | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|---|
| s1234 | ICP + Chamfer | -0.033 [-0.110, +0.044] | -0.002 [-0.087, +0.084] | -0.012 [-0.086, +0.062] | +0.045 [-0.044, +0.128] |
| s1234 | NoW col rilevatore | +0.090 [+0.003, +0.170] | +0.098 [+0.013, +0.176] | +0.110 [+0.024, +0.191] | +0.155 [+0.058, +0.242] |
| s1234 | fit 3DMM vB | +0.071 [-0.017, +0.166] | +0.071 [-0.018, +0.163] | +0.072 [-0.020, +0.164] | +0.082 [-0.027, +0.185] |
| s1234 | NoW coi landmark esatti (descrittivo) | -0.023 [-0.107, +0.062] | -0.013 [-0.094, +0.072] | +0.000 [-0.085, +0.088] | +0.052 [-0.044, +0.146] |
| s1234 | miglior famiglia (post hoc) | -0.033 [-0.110, +0.044] (ICP + Chamfer) | -0.002 [-0.087, +0.084] (ICP + Chamfer) | -0.012 [-0.086, +0.062] (ICP + Chamfer) | +0.045 [-0.044, +0.128] (ICP + Chamfer) |
| s1234 | migliore fra tutte (scelto su A) | +0.051 [-0.027, +0.130] | +0.056 [-0.029, +0.138] | +0.022 [-0.053, +0.096] | +0.045 [-0.044, +0.128] |
| s1234 | nostro non adattato | +0.032 [-0.023, +0.084] | +0.046 [-0.007, +0.093] | +0.048 [+0.001, +0.094] | +0.076 [+0.017, +0.127] |
| s1234 | testa k = 0 | +0.004 [-0.006, +0.015] | +0.016 [+0.000, +0.032] | +0.016 [-0.008, +0.041] | +0.057 [+0.022, +0.092] |
| s2345 | ICP + Chamfer | -0.049 [-0.134, +0.037] | -0.023 [-0.120, +0.069] | -0.012 [-0.103, +0.069] | +0.057 [-0.028, +0.145] |
| s2345 | NoW col rilevatore | +0.073 [-0.011, +0.154] | +0.077 [-0.005, +0.156] | +0.109 [+0.022, +0.196] | +0.167 [+0.088, +0.252] |
| s2345 | fit 3DMM vB | +0.055 [-0.042, +0.150] | +0.051 [-0.044, +0.147] | +0.071 [-0.026, +0.168] | +0.094 [+0.000, +0.194] |
| s2345 | NoW coi landmark esatti (descrittivo) | -0.040 [-0.126, +0.046] | -0.034 [-0.122, +0.052] | -0.000 [-0.089, +0.088] | +0.063 [-0.021, +0.150] |
| s2345 | miglior famiglia (post hoc) | -0.049 [-0.134, +0.037] (ICP + Chamfer) | -0.023 [-0.120, +0.069] (ICP + Chamfer) | -0.012 [-0.103, +0.069] (ICP + Chamfer) | +0.057 [-0.028, +0.145] (ICP + Chamfer) |
| s2345 | migliore fra tutte (scelto su A) | +0.035 [-0.054, +0.120] | +0.035 [-0.057, +0.125] | +0.021 [-0.061, +0.104] | +0.057 [-0.028, +0.145] |
| s2345 | nostro non adattato | -0.021 [-0.084, +0.041] | -0.014 [-0.078, +0.047] | +0.010 [-0.050, +0.065] | +0.058 [-0.005, +0.121] |
| s2345 | testa k = 0 | +0.010 [-0.004, +0.023] | +0.014 [-0.005, +0.032] | +0.027 [+0.005, +0.049] | +0.064 [+0.024, +0.105] |

Configurazioni scelte sulle k identita' (GT FR, estrazioni per configurazione):

- ICP + Chamfer: k = 5: mm 4, cs 1; k = 10: cs 4, mm 1; k = 20: mm 3, cs 2; k = 50: mm 4, cs 1
- NoW col rilevatore: k = 5: originale 3, metrical 2; k = 10: metrical 3, originale 2; k = 20: originale 4, metrical 1; k = 50: metrical 3, originale 2
- fit 3DMM vB: k = 5: gnm|coef 2, flame2023|coef 1, gnm|fr 1, gnm|comp 1; k = 10: gnm|comp 2, gnm|coef 2, flame2023|coef 1; k = 20: gnm|coef 3, gnm|comp 1, flame2023|comp 1; k = 50: gnm|comp 4, flame2023|comp 1
- NoW coi landmark esatti (descrittivo): k = 5: originale 3, metrical 2; k = 10: originale 4, metrical 1; k = 20: originale 4, metrical 1; k = 50: metrical 4, originale 1
- tutte le famiglie: k = 5: fit_vb:flame2023|coef 1, fit_vb:gnm|fr 1, fit_vb:gnm|coef 1, icp:mm 1, icp:cs 1; k = 10: icp:cs 2, fit_vb:flame2023|coef 1, fit_vb:gnm|comp 1, fit_vb:gnm|coef 1; k = 20: icp:mm 2, fit_vb:gnm|comp 1, fit_vb:gnm|coef 1, icp:cs 1; k = 50: icp:mm 4, icp:cs 1

**GT SR: rho sulle identita' di test**

| colonna | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|
| nostro adattato s1234 | 0.355 [0.271, 0.434] sd 0.013 | 0.372 [0.292, 0.442] sd 0.011 | 0.379 [0.305, 0.446] sd 0.033 | 0.434 [0.341, 0.507] sd 0.050 |
| nostro adattato s2345 | 0.354 [0.268, 0.429] sd 0.011 | 0.361 [0.278, 0.435] sd 0.013 | 0.385 [0.301, 0.455] sd 0.018 | 0.449 [0.361, 0.522] sd 0.036 |
| nostro non adattato s1234 | 0.331 [0.249, 0.400] sd 0.006 | 0.331 [0.252, 0.401] sd 0.007 | 0.331 [0.250, 0.399] sd 0.029 | 0.343 [0.238, 0.423] sd 0.049 |
| nostro non adattato s2345 | 0.377 [0.297, 0.449] sd 0.008 | 0.377 [0.296, 0.452] sd 0.008 | 0.376 [0.300, 0.447] sd 0.029 | 0.367 [0.275, 0.447] sd 0.048 |
| testa k = 0 s1234 | 0.347 [0.261, 0.426] sd 0.011 | 0.348 [0.260, 0.427] sd 0.009 | 0.350 [0.264, 0.428] sd 0.017 | 0.357 [0.252, 0.446] sd 0.058 |
| testa k = 0 s2345 | 0.346 [0.258, 0.424] sd 0.010 | 0.350 [0.262, 0.427] sd 0.009 | 0.360 [0.271, 0.439] sd 0.025 | 0.365 [0.273, 0.454] sd 0.047 |
| concorrente adattato ICP + Chamfer | 0.391 [0.316, 0.458] sd 0.009 | 0.399 [0.320, 0.472] sd 0.007 | 0.394 [0.313, 0.470] sd 0.012 | 0.360 [0.274, 0.435] sd 0.046 |
| concorrente adattato NoW col rilevatore | 0.290 [0.217, 0.353] sd 0.003 | 0.293 [0.224, 0.355] sd 0.004 | 0.284 [0.216, 0.347] sd 0.016 | 0.262 [0.179, 0.333] sd 0.032 |
| concorrente adattato fit 3DMM vB | 0.371 [0.292, 0.443] sd 0.046 | 0.345 [0.266, 0.415] sd 0.073 | 0.289 [0.195, 0.366] sd 0.058 | 0.369 [0.260, 0.454] sd 0.029 |
| concorrente adattato NoW coi landmark esatti (descrittivo) | 0.397 [0.318, 0.468] sd 0.011 | 0.401 [0.325, 0.471] sd 0.009 | 0.389 [0.311, 0.461] sd 0.018 | 0.350 [0.251, 0.434] sd 0.048 |
| concorrente adattato migliore fra tutte (scelto su A) | 0.371 [0.301, 0.431] sd 0.046 | 0.343 [0.267, 0.408] sd 0.071 | 0.309 [0.222, 0.382] sd 0.075 | 0.351 [0.261, 0.426] sd 0.032 |

**GT SR: delta nostro adattato - contro**

| seme | contro | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|---|
| s1234 | ICP + Chamfer | -0.036 [-0.120, +0.049] | -0.027 [-0.113, +0.059] | -0.015 [-0.096, +0.067] | +0.074 [-0.023, +0.162] |
| s1234 | NoW col rilevatore | +0.065 [-0.014, +0.143] | +0.079 [+0.003, +0.151] | +0.095 [+0.019, +0.167] | +0.172 [+0.076, +0.255] |
| s1234 | fit 3DMM vB | -0.016 [-0.102, +0.078] | +0.026 [-0.058, +0.111] | +0.090 [-0.003, +0.187] | +0.065 [-0.032, +0.158] |
| s1234 | NoW coi landmark esatti (descrittivo) | -0.041 [-0.125, +0.046] | -0.029 [-0.112, +0.056] | -0.010 [-0.093, +0.075] | +0.084 [-0.018, +0.176] |
| s1234 | miglior famiglia (post hoc) | -0.036 [-0.120, +0.049] (ICP + Chamfer) | -0.027 [-0.113, +0.059] (ICP + Chamfer) | -0.015 [-0.096, +0.067] (ICP + Chamfer) | +0.065 [-0.032, +0.158] (fit 3DMM vB) |
| s1234 | migliore fra tutte (scelto su A) | -0.016 [-0.098, +0.068] | +0.028 [-0.055, +0.107] | +0.070 [-0.015, +0.154] | +0.084 [-0.015, +0.170] |
| s1234 | nostro non adattato | +0.025 [-0.038, +0.082] | +0.041 [-0.017, +0.096] | +0.048 [-0.006, +0.098] | +0.091 [+0.023, +0.151] |
| s1234 | testa k = 0 | +0.009 [-0.003, +0.021] | +0.024 [+0.007, +0.042] | +0.029 [+0.002, +0.058] | +0.077 [+0.037, +0.119] |
| s2345 | ICP + Chamfer | -0.037 [-0.127, +0.057] | -0.038 [-0.129, +0.061] | -0.009 [-0.103, +0.086] | +0.088 [-0.008, +0.190] |
| s2345 | NoW col rilevatore | +0.064 [-0.017, +0.146] | +0.068 [-0.012, +0.145] | +0.101 [+0.022, +0.184] | +0.187 [+0.102, +0.279] |
| s2345 | fit 3DMM vB | -0.017 [-0.104, +0.071] | +0.016 [-0.075, +0.107] | +0.096 [-0.007, +0.201] | +0.079 [-0.014, +0.179] |
| s2345 | NoW coi landmark esatti (descrittivo) | -0.042 [-0.133, +0.048] | -0.040 [-0.133, +0.049] | -0.003 [-0.096, +0.085] | +0.099 [+0.003, +0.204] |
| s2345 | miglior famiglia (post hoc) | -0.037 [-0.127, +0.057] (ICP + Chamfer) | -0.038 [-0.129, +0.061] (ICP + Chamfer) | -0.009 [-0.103, +0.086] (ICP + Chamfer) | +0.079 [-0.014, +0.179] (fit 3DMM vB) |
| s2345 | migliore fra tutte (scelto su A) | -0.017 [-0.101, +0.071] | +0.018 [-0.076, +0.107] | +0.077 [-0.020, +0.170] | +0.098 [+0.004, +0.196] |
| s2345 | nostro non adattato | -0.023 [-0.094, +0.050] | -0.016 [-0.088, +0.055] | +0.009 [-0.060, +0.080] | +0.081 [+0.005, +0.164] |
| s2345 | testa k = 0 | +0.008 [-0.007, +0.024] | +0.011 [-0.010, +0.033] | +0.026 [+0.001, +0.051] | +0.083 [+0.038, +0.131] |

Configurazioni scelte sulle k identita' (GT SR, estrazioni per configurazione):

- ICP + Chamfer: k = 5: cs 3, mm 2; k = 10: cs 5; k = 20: cs 5; k = 50: cs 3, mm 2
- NoW col rilevatore: k = 5: originale 3, metrical 2; k = 10: originale 4, metrical 1; k = 20: originale 4, metrical 1; k = 50: metrical 3, originale 2
- fit 3DMM vB: k = 5: gnm|sr 3, gnm|coef 1, flame2023|sr 1; k = 10: gnm|sr 3, flame2023|coef 1, gnm|coef 1; k = 20: gnm|coef 3, flame2023|coef 1, gnm|sr 1; k = 50: gnm|sr 4, flame2023|sr 1
- NoW coi landmark esatti (descrittivo): k = 5: originale 4, metrical 1; k = 10: originale 4, metrical 1; k = 20: originale 4, metrical 1; k = 50: metrical 3, originale 2
- tutte le famiglie: k = 5: icp:cs 2, fit_vb:gnm|coef 1, fit_vb:gnm|sr 1, icp:mm 1; k = 10: icp:cs 2, fit_vb:flame2023|coef 1, fit_vb:gnm|sr 1, fit_vb:gnm|coef 1; k = 20: fit_vb:gnm|coef 2, icp:cs 1, fit_vb:flame2023|coef 1, fit_vb:gnm|sr 1; k = 50: icp:mm 2, icp:cs 1, fit_vb:gnm|sr 1, fit_vb:flame2023|sr 1

## Controlli

- K1 (rho sulle righe della matrice, tutti i 100 soggetti, contro `full_matrix/spearman.csv`): scarto massimo 8.3e-17, righe uguali True, passa True.
- Righe: HIFI3D 99000 di 99000 nocrop (K1 98224); FaceScape dev 99000 di 99000 nocrop (K1 88724); FaceVerse con espressioni 99000 di 99000 nocrop (K1 99000); FaceVerse neutra 99000 di 99000 nocrop (K1 99000).
- K-emb (d_P dai latenti contro la colonna della matrice, scarto relativo massimo): hifi3d 0.0e+00; facescape 0.0e+00; faceverse 0.0e+00; faceverse_neutral 0.0e+00; K-GT (G del blocco proporzionale alla GT delle righe): hifi3d True, facescape True, faceverse True, faceverse_neutral True.
- Teste lette: 320 / 320 con success, nodi i256-a10-08.srv.aau.dk, i256-a10-10.srv.aau.dk.

- Fit: 188 / 188 con success, iterazioni 203-309, nodi i256-a10-08.srv.aau.dk, i256-a10-10.srv.aau.dk; K0 (testa joint di R6 contro d3_e2_heads.csv): s1234 c 0.0e+00, alpha 0.0e+00, passa True; s2345 c 0.0e+00, alpha 0.0e+00, passa True.

## FaMoS TEST: riconoscimento dell'identita' (descrittivo, limiti in REGOLE.md sez. 7)

Scansioni dei soggetti di test (galleria = le altre scansioni di test); nostro = d_F della testa adattata; concorrenti = configurazione scelta col mAP sulle scansioni di adattamento. Fit 3DMM vB assente su FaMoS.

| colonna | metrica | k = 5 | k = 10 |
|---|---|---|---|
| nostro_head_form s1234 | rank1 | 0.902 [0.860, 0.947] | 0.924 [0.886, 0.958] |
| nostro_head_form s2345 | rank1 | 0.884 [0.839, 0.930] | 0.936 [0.906, 0.963] |
| nostro_base_form s1234 | rank1 | 0.870 [0.835, 0.907] | 0.888 [0.851, 0.932] |
| nostro_base_form s2345 | rank1 | 0.840 [0.793, 0.890] | 0.880 [0.849, 0.916] |
| concorrente ICP + Chamfer | rank1 | 0.960 [0.932, 0.985] | 0.956 [0.926, 0.987] |
| concorrente NoW col rilevatore | rank1 | 0.968 [0.944, 0.990] | 0.960 [0.934, 0.989] |
| nostro_head_form s1234 | mAP | 0.719 [0.643, 0.791] | 0.788 [0.735, 0.841] |
| nostro_head_form s2345 | mAP | 0.707 [0.625, 0.786] | 0.785 [0.729, 0.841] |
| nostro_base_form s1234 | mAP | 0.699 [0.637, 0.768] | 0.766 [0.721, 0.813] |
| nostro_base_form s2345 | mAP | 0.673 [0.595, 0.750] | 0.747 [0.691, 0.803] |
| concorrente ICP + Chamfer | mAP | 0.856 [0.816, 0.898] | 0.874 [0.835, 0.914] |
| concorrente NoW col rilevatore | mAP | 0.888 [0.836, 0.933] | 0.903 [0.862, 0.945] |
| nostro_head_form s1234 | AUC | 0.829 [0.765, 0.889] | 0.828 [0.768, 0.892] |
| nostro_head_form s2345 | AUC | 0.826 [0.753, 0.893] | 0.825 [0.762, 0.894] |
| nostro_base_form s1234 | AUC | 0.819 [0.776, 0.870] | 0.810 [0.758, 0.861] |
| nostro_base_form s2345 | AUC | 0.800 [0.747, 0.857] | 0.793 [0.737, 0.849] |
| concorrente ICP + Chamfer | AUC | 0.894 [0.858, 0.936] | 0.879 [0.843, 0.930] |
| concorrente NoW col rilevatore | AUC | 0.916 [0.872, 0.956] | 0.903 [0.861, 0.952] |
| nostro_head_form s1234 | EER | 0.263 [0.193, 0.320] | 0.261 [0.186, 0.320] |
| nostro_head_form s2345 | EER | 0.267 [0.191, 0.328] | 0.261 [0.181, 0.323] |
| nostro_base_form s1234 | EER | 0.249 [0.194, 0.296] | 0.249 [0.197, 0.318] |
| nostro_base_form s2345 | EER | 0.269 [0.216, 0.320] | 0.272 [0.217, 0.339] |
| concorrente ICP + Chamfer | EER | 0.175 [0.119, 0.221] | 0.202 [0.127, 0.228] |
| concorrente NoW col rilevatore | EER | 0.161 [0.101, 0.208] | 0.175 [0.103, 0.216] |

| seme | metrica | contro | k = 5 | k = 10 |
|---|---|---|---|---|
| s1234 | rank1 | ICP + Chamfer | -0.058 [-0.099, -0.018] | -0.032 [-0.081, +0.018] |
| s1234 | rank1 | NoW col rilevatore | -0.066 [-0.099, -0.032] | -0.036 [-0.082, +0.012] |
| s1234 | rank1 | migliore | -0.066 [-0.099, -0.032] | -0.036 [-0.082, +0.012] |
| s1234 | rank1 | nostro_base_form | +0.032 [-0.005, +0.065] | +0.036 [-0.015, +0.093] |
| s2345 | rank1 | ICP + Chamfer | -0.076 [-0.114, -0.035] | -0.020 [-0.064, +0.025] |
| s2345 | rank1 | NoW col rilevatore | -0.084 [-0.116, -0.050] | -0.024 [-0.063, +0.013] |
| s2345 | rank1 | migliore | -0.084 [-0.116, -0.050] | -0.024 [-0.063, +0.013] |
| s2345 | rank1 | nostro_base_form | +0.044 [+0.014, +0.076] | +0.056 [+0.028, +0.084] |
| s1234 | mAP | ICP + Chamfer | -0.137 [-0.191, -0.088] | -0.086 [-0.126, -0.045] |
| s1234 | mAP | NoW col rilevatore | -0.169 [-0.206, -0.128] | -0.115 [-0.143, -0.082] |
| s1234 | mAP | migliore | -0.169 [-0.206, -0.128] | -0.115 [-0.143, -0.082] |
| s1234 | mAP | nostro_base_form | +0.020 [-0.015, +0.058] | +0.022 [-0.014, +0.057] |
| s2345 | mAP | ICP + Chamfer | -0.149 [-0.209, -0.094] | -0.089 [-0.134, -0.040] |
| s2345 | mAP | NoW col rilevatore | -0.181 [-0.224, -0.132] | -0.118 [-0.150, -0.077] |
| s2345 | mAP | migliore | -0.181 [-0.224, -0.132] | -0.118 [-0.150, -0.077] |
| s2345 | mAP | nostro_base_form | +0.035 [-0.005, +0.078] | +0.038 [+0.006, +0.073] |
| s1234 | AUC | ICP + Chamfer | -0.065 [-0.110, -0.025] | -0.051 [-0.100, -0.011] |
| s1234 | AUC | NoW col rilevatore | -0.086 [-0.123, -0.051] | -0.076 [-0.112, -0.045] |
| s1234 | AUC | migliore | -0.086 [-0.123, -0.051] | -0.076 [-0.112, -0.045] |
| s1234 | AUC | nostro_base_form | +0.010 [-0.023, +0.040] | +0.018 [-0.017, +0.052] |
| s2345 | AUC | ICP + Chamfer | -0.069 [-0.124, -0.019] | -0.054 [-0.104, -0.014] |
| s2345 | AUC | NoW col rilevatore | -0.090 [-0.137, -0.048] | -0.079 [-0.117, -0.046] |
| s2345 | AUC | migliore | -0.090 [-0.137, -0.048] | -0.079 [-0.117, -0.046] |
| s2345 | AUC | nostro_base_form | +0.026 [-0.018, +0.065] | +0.032 [-0.006, +0.078] |
| s1234 | EER | ICP + Chamfer | -0.088 [-0.137, -0.030] | -0.059 [-0.123, -0.016] |
| s1234 | EER | NoW col rilevatore | -0.102 [-0.139, -0.056] | -0.086 [-0.136, -0.050] |
| s1234 | EER | migliore | -0.102 [-0.139, -0.056] | -0.086 [-0.136, -0.050] |
| s1234 | EER | nostro_base_form | -0.014 [-0.041, +0.023] | -0.012 [-0.037, +0.045] |
| s2345 | EER | ICP + Chamfer | -0.093 [-0.147, -0.026] | -0.059 [-0.127, -0.017] |
| s2345 | EER | NoW col rilevatore | -0.106 [-0.153, -0.052] | -0.085 [-0.140, -0.048] |
| s2345 | EER | migliore | -0.106 [-0.153, -0.052] | -0.085 [-0.140, -0.048] |
| s2345 | EER | nostro_base_form | +0.001 [-0.036, +0.055] | +0.011 [-0.024, +0.072] |

<!-- da qui in giu': scritto a mano, conservato da fs_eval.py results -->