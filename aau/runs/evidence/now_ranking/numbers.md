Metodi: 3ddfa_v2, synergynet, prnet, 3ddfa_v3, hrn; immagini con tutti i metodi e tutti i criteri: 352 / 352; mancanti per metodo: {'3ddfa_v2': 0, 'synergynet': 0, 'prnet': 0, '3ddfa_v3': 0, 'hrn': 0}. IC 95% bootstrap per soggetto (1000 repliche, seme 1234).

### Primaria: tau per immagine contro NoW originale

| criterio | tutti i metodi | tre del 7 ottobre |
|---|---|---|
| NoW metrical (mm) | +0.388 [+0.258, +0.514] | +0.665 [+0.572, +0.754] |
| Chamfer grezza (maxabs) | +0.206 [+0.069, +0.318] | +0.246 [+0.123, +0.373] |
| ICP + Chamfer (mm) | +0.595 [+0.520, +0.675] | +0.612 [+0.537, +0.686] |
| d_P s1234 | +0.430 [+0.343, +0.517] | +0.419 [+0.361, +0.479] |
| d_F s1234 (taglia del template) | +0.410 [+0.325, +0.500] | +0.402 [+0.310, +0.487] |
| d_P s2345 | +0.420 [+0.332, +0.504] | +0.350 [+0.263, +0.438] |
| d_F s2345 (taglia del template) | +0.384 [+0.288, +0.476] | +0.316 [+0.207, +0.412] |

| delta | insieme | valore | P(<= 0) |
|---|---|---|---|
| d_P s1234 - Chamfer grezza (maxabs) | tutti | +0.224 [+0.068, +0.401] | 0.002 |
| d_P s1234 - ICP + Chamfer (mm) | tutti | -0.165 [-0.231, -0.086] | 1.000 |
| d_P s2345 - Chamfer grezza (maxabs) | tutti | +0.214 [+0.053, +0.383] | 0.002 |
| d_P s2345 - ICP + Chamfer (mm) | tutti | -0.176 [-0.231, -0.109] | 1.000 |
| d_P s1234 - Chamfer grezza (maxabs) | tre_7ottobre | +0.172 [+0.063, +0.286] | 0.000 |
| d_P s2345 - Chamfer grezza (maxabs) | tre_7ottobre | +0.104 [-0.028, +0.229] | 0.056 |

| seme | esito |
|---|---|
| s1234 | INFORMATIVO POSITIVO |
| s2345 | INFORMATIVO POSITIVO |
| finale | **INFORMATIVO POSITIVO** |

### Contro NoW metrical (descrittivo)

| criterio | tau per immagine |
|---|---|
| NoW originale (mm) | +0.388 [+0.258, +0.514] |
| Chamfer grezza (maxabs) | -0.132 [-0.223, -0.024] |
| ICP + Chamfer (mm) | +0.386 [+0.255, +0.521] |
| d_P s1234 | +0.306 [+0.210, +0.411] |
| d_F s1234 (taglia del template) | +0.311 [+0.215, +0.420] |
| d_P s2345 | +0.330 [+0.246, +0.423] |
| d_F s2345 (taglia del template) | +0.312 [+0.219, +0.413] |

### Metodi con unita' note, patch metriche (descrittivo)

| criterio | contro | tau per immagine |
|---|---|---|
| dFm_s1234 (3ddfa_v3+hrn) | metrical | +0.824 [+0.638, +0.964] |
| dPm_s1234 (3ddfa_v3+hrn) | metrical | -0.250 [-0.433, -0.039] |
| dFm_s2345 (3ddfa_v3+hrn) | metrical | +0.807 [+0.623, +0.952] |
| dPm_s2345 (3ddfa_v3+hrn) | metrical | +0.017 [-0.189, +0.225] |
| dFm_s1234 (3ddfa_v3+hrn) | now | -0.080 [-0.314, +0.150] |
| dPm_s1234 (3ddfa_v3+hrn) | now | +0.392 [+0.235, +0.541] |
| dFm_s2345 (3ddfa_v3+hrn) | now | -0.051 [-0.296, +0.179] |
| dPm_s2345 (3ddfa_v3+hrn) | now | +0.432 [+0.307, +0.554] |

### Per sfida (tau per immagine contro NoW originale)

| criterio | multiview_neutral (n) | multiview_expressions (n) | multiview_occlusions (n) | selfie (n) |
|---|---|---|---|---|
| NoW metrical (mm) | +0.36 [+0.24, +0.48] (108) | +0.42 [+0.26, +0.57] (114) | +0.35 [+0.19, +0.50] (86) | +0.44 [+0.30, +0.60] (44) |
| Chamfer grezza (maxabs) | +0.24 [+0.10, +0.36] (108) | +0.22 [+0.09, +0.33] (114) | +0.16 [-0.01, +0.32] (86) | +0.18 [-0.03, +0.36] (44) |
| ICP + Chamfer (mm) | +0.60 [+0.51, +0.69] (108) | +0.59 [+0.49, +0.69] (114) | +0.61 [+0.50, +0.74] (86) | +0.55 [+0.42, +0.67] (44) |
| d_P s1234 | +0.43 [+0.34, +0.51] (108) | +0.47 [+0.38, +0.56] (114) | +0.36 [+0.20, +0.51] (86) | +0.47 [+0.30, +0.61] (44) |
| d_F s1234 (taglia del template) | +0.42 [+0.32, +0.52] (108) | +0.37 [+0.26, +0.48] (114) | +0.46 [+0.36, +0.57] (86) | +0.39 [+0.20, +0.53] (44) |
| d_P s2345 | +0.41 [+0.32, +0.50] (108) | +0.42 [+0.33, +0.51] (114) | +0.44 [+0.33, +0.55] (86) | +0.40 [+0.23, +0.54] (44) |
| d_F s2345 (taglia del template) | +0.39 [+0.30, +0.49] (108) | +0.32 [+0.22, +0.43] (114) | +0.44 [+0.33, +0.55] (86) | +0.40 [+0.22, +0.56] (44) |

### Classifiche dei metodi (descrittive)

| criterio | ordine (punteggio [IC], P(primo)) | tau globale con NoW orig. / metrical |
|---|---|---|
| now_official | synergynet 1.350 [1.178, 1.543] 0.77; 3ddfa_v2 1.390 [1.211, 1.620] 0.17; hrn 1.508 [1.356, 1.691] 0.03; 3ddfa_v3 1.521 [1.409, 1.634] 0.03; prnet 1.563 [1.403, 1.767] 0.00 | +1.00 / +0.60 |
| metrical_official | synergynet 1.550 [1.363, 1.775] 0.87; 3ddfa_v2 1.592 [1.385, 1.838] 0.13; prnet 1.803 [1.614, 2.022] 0.00; hrn 1.910 [1.674, 2.163] 0.00; 3ddfa_v3 4.145 [3.676, 4.687] 0.00 | +0.60 / +1.00 |
| NoW originale (mm) | synergynet 1.472 [1.257, 1.678] 0.70; 3ddfa_v2 1.526 [1.298, 1.771] 0.07; hrn 1.571 [1.411, 1.745] 0.10; 3ddfa_v3 1.575 [1.461, 1.689] 0.12; prnet 1.668 [1.482, 1.888] 0.00 | +1.00 / +0.60 |
| NoW metrical (mm) | synergynet 1.684 [1.445, 1.914] 0.94; 3ddfa_v2 1.740 [1.486, 1.994] 0.06; prnet 1.948 [1.716, 2.183] 0.00; hrn 2.005 [1.739, 2.279] 0.00; 3ddfa_v3 4.462 [3.965, 5.028] 0.00 | +0.60 / +1.00 |
| Chamfer grezza (maxabs) | 3ddfa_v3 0.042 [0.039, 0.044] 1.00; synergynet 0.047 [0.044, 0.050] 0.00; hrn 0.048 [0.044, 0.052] 0.00; 3ddfa_v2 0.049 [0.046, 0.053] 0.00; prnet 0.050 [0.047, 0.053] 0.00 | +0.20 / -0.20 |
| ICP + Chamfer (mm) | synergynet 2.789 [2.578, 2.994] 0.99; 3ddfa_v2 2.894 [2.669, 3.125] 0.01; 3ddfa_v3 3.159 [2.951, 3.380] 0.00; prnet 3.322 [3.027, 3.692] 0.00; hrn 3.337 [3.099, 3.591] 0.00 | +0.60 / +0.60 |
| d_P s1234 | synergynet 0.060 [0.055, 0.066] 1.00; 3ddfa_v2 0.065 [0.060, 0.071] 0.00; prnet 0.068 [0.063, 0.072] 0.00; 3ddfa_v3 0.071 [0.066, 0.076] 0.00; hrn 0.076 [0.070, 0.081] 0.00 | +0.40 / +0.80 |
| d_F s1234 (taglia del template) | synergynet 3.660 [3.276, 4.071] 1.00; 3ddfa_v2 3.891 [3.564, 4.246] 0.00; prnet 4.114 [3.802, 4.439] 0.00; 3ddfa_v3 4.346 [4.009, 4.710] 0.00; hrn 4.804 [4.427, 5.221] 0.00 | +0.40 / +0.80 |
| d_P s2345 | synergynet 0.062 [0.057, 0.068] 1.00; 3ddfa_v2 0.067 [0.062, 0.073] 0.00; prnet 0.069 [0.064, 0.073] 0.00; 3ddfa_v3 0.076 [0.071, 0.081] 0.00; hrn 0.078 [0.073, 0.084] 0.00 | +0.40 / +0.80 |
| d_F s2345 (taglia del template) | synergynet 3.789 [3.405, 4.182] 1.00; 3ddfa_v2 4.077 [3.764, 4.402] 0.00; prnet 4.213 [3.926, 4.510] 0.00; 3ddfa_v3 4.684 [4.317, 5.074] 0.00; hrn 5.094 [4.676, 5.565] 0.00 | +0.40 / +0.80 |

### Dentro il metodo: Spearman fra le immagini (descrittivo)

| metodo | criterio | contro | n | rho |
|---|---|---|---|---|
| 3ddfa_v2 | d_P s1234 | now | 352 | +0.653 [+0.512, +0.728] |
| 3ddfa_v2 | d_P s2345 | now | 352 | +0.637 [+0.487, +0.733] |
| 3ddfa_v2 | Chamfer grezza (maxabs) | now | 352 | +0.449 [+0.155, +0.707] |
| 3ddfa_v2 | ICP + Chamfer (mm) | now | 352 | +0.853 [+0.687, +0.934] |
| synergynet | d_P s1234 | now | 352 | +0.735 [+0.565, +0.826] |
| synergynet | d_P s2345 | now | 352 | +0.746 [+0.579, +0.826] |
| synergynet | Chamfer grezza (maxabs) | now | 352 | +0.487 [+0.179, +0.725] |
| synergynet | ICP + Chamfer (mm) | now | 352 | +0.837 [+0.663, +0.925] |
| prnet | d_P s1234 | now | 352 | +0.744 [+0.670, +0.793] |
| prnet | d_P s2345 | now | 352 | +0.718 [+0.626, +0.784] |
| prnet | Chamfer grezza (maxabs) | now | 352 | +0.424 [+0.271, +0.547] |
| prnet | ICP + Chamfer (mm) | now | 352 | +0.865 [+0.752, +0.931] |
| 3ddfa_v3 | d_P s1234 | now | 352 | +0.686 [+0.557, +0.786] |
| 3ddfa_v3 | d_P s2345 | now | 352 | +0.664 [+0.549, +0.764] |
| 3ddfa_v3 | Chamfer grezza (maxabs) | now | 352 | +0.574 [+0.387, +0.725] |
| 3ddfa_v3 | ICP + Chamfer (mm) | now | 352 | +0.833 [+0.737, +0.895] |
| hrn | d_P s1234 | now | 352 | +0.511 [+0.328, +0.692] |
| hrn | d_P s2345 | now | 352 | +0.574 [+0.408, +0.729] |
| hrn | Chamfer grezza (maxabs) | now | 352 | +0.585 [+0.391, +0.713] |
| hrn | ICP + Chamfer (mm) | now | 352 | +0.834 [+0.708, +0.914] |
| 3ddfa_v3 | dFm_s1234 (patch metrica) | metrical | 352 | +0.921 [+0.873, +0.954] |
| 3ddfa_v3 | dFm_s1234 (patch metrica) | now | 352 | +0.258 [+0.025, +0.463] |
| 3ddfa_v3 | dPm_s1234 (patch metrica) | metrical | 352 | +0.303 [+0.115, +0.490] |
| 3ddfa_v3 | dPm_s1234 (patch metrica) | now | 352 | +0.699 [+0.563, +0.801] |
| 3ddfa_v3 | dFm_s2345 (patch metrica) | metrical | 352 | +0.923 [+0.880, +0.955] |
| 3ddfa_v3 | dFm_s2345 (patch metrica) | now | 352 | +0.263 [+0.055, +0.462] |
| 3ddfa_v3 | dPm_s2345 (patch metrica) | metrical | 352 | +0.355 [+0.162, +0.545] |
| 3ddfa_v3 | dPm_s2345 (patch metrica) | now | 352 | +0.651 [+0.520, +0.755] |
| hrn | dFm_s1234 (patch metrica) | metrical | 352 | +0.513 [+0.307, +0.709] |
| hrn | dFm_s1234 (patch metrica) | now | 352 | +0.491 [+0.290, +0.660] |
| hrn | dPm_s1234 (patch metrica) | metrical | 352 | +0.299 [+0.043, +0.574] |
| hrn | dPm_s1234 (patch metrica) | now | 352 | +0.506 [+0.317, +0.689] |
| hrn | dFm_s2345 (patch metrica) | metrical | 352 | +0.484 [+0.284, +0.667] |
| hrn | dFm_s2345 (patch metrica) | now | 352 | +0.518 [+0.321, +0.677] |
| hrn | dPm_s2345 (patch metrica) | metrical | 352 | +0.346 [+0.093, +0.592] |
| hrn | dPm_s2345 (patch metrica) | now | 352 | +0.568 [+0.398, +0.729] |
