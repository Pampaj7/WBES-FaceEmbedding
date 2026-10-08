# E3b: rimesh uniforme al test (e diagnosi di centro e pooling), e108 senza training

Protocollo e lato L dichiarati prima in `aau/runs/evidence/e2/protocol.md`; generatore in `aau/evidence/e3_breakdown/remesh.py` (record per mesh in `aau/runs/evidence/e3/<dominio>/remesh_records.csv`). IC 95% bootstrap per soggetto, 1000 repliche, le stesse dei riferimenti; P(<=0) = frazione di repliche con differenza <= 0.

## HIFI3D: Spearman con la GT maxabs

| metodo | nocrop_cross | all_cross | subject_pair_mean |
| --- | --- | --- | --- |
| e108 (mesh come sono) | 0.630 [0.569, 0.689] | 0.541 [0.471, 0.605] | 0.795 [0.738, 0.840] |
| e108, rimesh uniforme al test | 0.456 [0.374, 0.531] | 0.342 [0.268, 0.417] | 0.580 [0.494, 0.660] |
| e108, centro per area | 0.282 [0.227, 0.337] | 0.216 [0.168, 0.262] | 0.628 [0.535, 0.702] |
| e108, pooling medio per area | 0.332 [0.274, 0.390] | 0.306 [0.243, 0.365] | 0.560 [0.473, 0.636] |
| e108, centro e pooling per area | 0.476 [0.395, 0.550] | 0.350 [0.278, 0.422] | 0.581 [0.499, 0.660] |

Differenze appaiate variante - e108:

| variante - e108 | nocrop_cross [IC] (P<=0) | all_cross [IC] (P<=0) | subject_pair_mean [IC] (P<=0) |
| --- | --- | --- | --- |
| e108, rimesh uniforme al test | -0.174 [-0.224, -0.127] (1.000) | -0.199 [-0.242, -0.154] (1.000) | -0.215 [-0.274, -0.162] (1.000) |
| e108, centro per area | -0.348 [-0.387, -0.309] (1.000) | -0.325 [-0.361, -0.279] (1.000) | -0.167 [-0.234, -0.110] (1.000) |
| e108, pooling medio per area | -0.298 [-0.337, -0.259] (1.000) | -0.235 [-0.275, -0.195] (1.000) | -0.235 [-0.302, -0.181] (1.000) |
| e108, centro e pooling per area | -0.154 [-0.204, -0.107] (1.000) | -0.191 [-0.233, -0.150] (1.000) | -0.214 [-0.272, -0.161] (1.000) |

## HIFI3D: riconoscimento, 5 topologie senza crop

| metodo | rank-1 | mAP | AUC |
| --- | --- | --- | --- |
| e108 (mesh come sono) | 0.782 [0.754, 0.807] | 0.845 [0.822, 0.863] | 0.946 [0.934, 0.957] |
| e108, rimesh uniforme al test | 0.803 [0.779, 0.829] | 0.855 [0.834, 0.876] | 0.934 [0.918, 0.949] |
| e108, centro per area | 0.231 [0.205, 0.257] | 0.333 [0.306, 0.360] | 0.730 [0.711, 0.751] |
| e108, pooling medio per area | 0.452 [0.416, 0.486] | 0.569 [0.539, 0.599] | 0.853 [0.835, 0.871] |
| e108, centro e pooling per area | 0.841 [0.817, 0.864] | 0.891 [0.873, 0.909] | 0.955 [0.942, 0.967] |

Differenze appaiate (stesse repliche):

| A - B | rank-1 [IC] (P<=0) | mAP | AUC |
| --- | --- | --- | --- |
| e108, rimesh uniforme al test - e108 (mesh come sono) | +0.021 [-0.005, +0.044] (0.049) | +0.010 [-0.010, +0.029] (0.162) | -0.012 [-0.023, -0.003] (0.996) |
| e108, centro per area - e108 (mesh come sono) | -0.552 [-0.580, -0.523] (1.000) | -0.511 [-0.536, -0.487] (1.000) | -0.215 [-0.228, -0.202] (1.000) |
| e108, pooling medio per area - e108 (mesh come sono) | -0.330 [-0.362, -0.300] (1.000) | -0.275 [-0.301, -0.252] (1.000) | -0.093 [-0.106, -0.081] (1.000) |
| e108, centro e pooling per area - e108 (mesh come sono) | +0.058 [+0.034, +0.084] (0.000) | +0.047 [+0.029, +0.065] (0.000) | +0.009 [+0.001, +0.017] (0.018) |

## HIFI3D: riconoscimento, coppie con crop (a parte)

| metodo | rank-1 | mAP | AUC |
| --- | --- | --- | --- |
| e108 (mesh come sono) | 0.348 [0.299, 0.395] | 0.475 [0.433, 0.516] | 0.843 [0.822, 0.865] |
| e108, rimesh uniforme al test | 0.182 [0.141, 0.227] | 0.326 [0.291, 0.364] | 0.862 [0.837, 0.885] |
| e108, centro per area | 0.058 [0.037, 0.082] | 0.142 [0.119, 0.167] | 0.651 [0.635, 0.667] |
| e108, pooling medio per area | 0.253 [0.214, 0.291] | 0.381 [0.342, 0.418] | 0.837 [0.815, 0.856] |
| e108, centro e pooling per area | 0.157 [0.118, 0.200] | 0.318 [0.285, 0.353] | 0.870 [0.844, 0.894] |

Differenze appaiate (stesse repliche):

| A - B | rank-1 [IC] (P<=0) | mAP | AUC |
| --- | --- | --- | --- |
| e108, rimesh uniforme al test - e108 (mesh come sono) | -0.166 [-0.216, -0.118] (1.000) | -0.149 [-0.187, -0.111] (1.000) | +0.018 [-0.003, +0.038] (0.040) |
| e108, centro per area - e108 (mesh come sono) | -0.290 [-0.334, -0.245] (1.000) | -0.333 [-0.371, -0.298] (1.000) | -0.192 [-0.208, -0.177] (1.000) |
| e108, pooling medio per area - e108 (mesh come sono) | -0.095 [-0.150, -0.040] (1.000) | -0.094 [-0.142, -0.046] (1.000) | -0.007 [-0.029, +0.013] (0.726) |
| e108, centro e pooling per area - e108 (mesh come sono) | -0.191 [-0.240, -0.147] (1.000) | -0.157 [-0.193, -0.121] (1.000) | +0.026 [+0.006, +0.046] (0.009) |

Rimesh, HIFI3D: vertici in uscita per topologia (mediana) e qualita':

| topologia | vertici in ingresso | vertici in uscita | lato medio / L | CV dei lati | area uscita / ingresso | spigoli non-manifold (totale) | s/mesh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crop | 9062 | 7672 | 1.026 | 0.169 | 0.971 | 1 | 8.5 |
| down8k | 3312 | 7915 | 1.027 | 0.162 | 0.974 | 3 | 12.8 |
| noisy | 9518 | 12521 | 0.919 | 0.270 | 0.766 | 667 | 9.6 |
| original | 9518 | 7911 | 1.027 | 0.173 | 0.974 | 1 | 9.0 |
| remesh | 6679 | 7762 | 1.024 | 0.154 | 0.975 | 1 | 26.1 |
| up60k | 24394 | 7911 | 1.021 | 0.149 | 0.974 | 0 | 24.2 |

## FaceVerse con espressioni (convenzione BFM): riconoscimento, 5 topologie senza crop

| metodo | rank-1 | mAP | AUC |
| --- | --- | --- | --- |
| e108 (mesh come sono) | 0.625 [0.584, 0.668] | 0.689 [0.650, 0.727] | 0.869 [0.842, 0.894] |
| e108, rimesh uniforme al test | 0.767 [0.723, 0.810] | 0.808 [0.766, 0.846] | 0.921 [0.897, 0.944] |
| e108 su FaceVerse NEUTRO | 0.916 [0.897, 0.936] | 0.946 [0.933, 0.960] | 0.975 [0.968, 0.982] |

Differenze appaiate (stesse repliche):

| A - B | rank-1 [IC] (P<=0) | mAP | AUC |
| --- | --- | --- | --- |
| e108, rimesh uniforme al test - e108 (mesh come sono) | +0.142 [+0.111, +0.173] (0.000) | +0.119 [+0.094, +0.144] (0.000) | +0.052 [+0.039, +0.066] (0.000) |
| e108 su FaceVerse NEUTRO - e108 (mesh come sono) | +0.291 [+0.247, +0.338] (0.000) | +0.257 [+0.219, +0.299] (0.000) | +0.107 [+0.084, +0.132] (0.000) |

## FaceVerse con espressioni (convenzione BFM): riconoscimento, coppie con crop (a parte)

| metodo | rank-1 | mAP | AUC |
| --- | --- | --- | --- |
| e108 (mesh come sono) | 0.252 [0.208, 0.297] | 0.375 [0.332, 0.418] | 0.788 [0.761, 0.816] |
| e108, rimesh uniforme al test | 0.305 [0.256, 0.357] | 0.437 [0.390, 0.484] | 0.841 [0.815, 0.870] |
| e108 su FaceVerse NEUTRO | 0.451 [0.402, 0.502] | 0.577 [0.537, 0.619] | 0.873 [0.856, 0.892] |

Differenze appaiate (stesse repliche):

| A - B | rank-1 [IC] (P<=0) | mAP | AUC |
| --- | --- | --- | --- |
| e108, rimesh uniforme al test - e108 (mesh come sono) | +0.053 [+0.017, +0.088] (0.005) | +0.062 [+0.033, +0.090] (0.000) | +0.053 [+0.038, +0.068] (0.000) |
| e108 su FaceVerse NEUTRO - e108 (mesh come sono) | +0.199 [+0.151, +0.252] (0.000) | +0.202 [+0.159, +0.247] (0.000) | +0.085 [+0.060, +0.111] (0.000) |

Rimesh, FaceVerse con espressioni (convenzione BFM): vertici in uscita per topologia (mediana) e qualita':

| topologia | vertici in ingresso | vertici in uscita | lato medio / L | CV dei lati | area uscita / ingresso | spigoli non-manifold (totale) | s/mesh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crop | 26104 | 7835 | 1.022 | 0.165 | 0.964 | 28 | 25.6 |
| down8k | 9915 | 7784 | 1.032 | 0.190 | 0.964 | 31 | 9.5 |
| noisy | 28632 | 19039 | 0.760 | 0.320 | 0.476 | 1651 | 30.5 |
| original | 28632 | 7833 | 1.021 | 0.168 | 0.962 | 28 | 28.2 |
| remesh | 20108 | 7602 | 1.026 | 0.161 | 0.976 | 9 | 19.5 |
| up60k | 73606 | 7803 | 1.017 | 0.160 | 0.963 | 13 | 18.3 |

