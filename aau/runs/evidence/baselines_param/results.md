# Concorrenti parametrici (GNM, FLAME 2023 Open) e varifold: risultati

Protocollo `PROTOCOL.md` (sha256 `0d7444e622c7295fe620467e5e0c136dd0f06d348e35de16de09a65e8566cda1`), codice `aau/baselines_param/`. Righe, GT e repliche di `v3_work/trainer/tools/fact_paired.py` (importato, non modificato). IC 95% percentile, 1000 repliche per soggetto. Tutti i numeri in `spearman.csv` (valori) e `paired.csv` (valori e delta, anche parziale e quintile basso).

## Lettura (regola della sez. 7 del protocollo, scritta dopo i numeri)

Job: fit 1067569 (GNM, 4 min 22 s) e 1067570 (FLAME, 20 min 40 s), 48 core CPU ciascuno; varifold 1067571-1067574
(una L40S per vista, da 17 min 38 s a 29 min 44 s); controllo GPU/CPU 1067582; delta 1067575 (15 min 31 s, 32 core).
FaMoS (15 mesh): fit e varifold eseguiti nella prova funzionale, prima del protocollo, senza GT (vedi PROTOCOL.md).

- **Fit: 0 falliti su 4.030** (2 modelli x (4 viste x 500 + 15)); RMS del fit sui punti registrati <= 0.84 mm. Nessuna
  riga persa: la maschera comune coincide con quella di fact_paired e i bracci ridanno `factorized_paired.csv` (scarto
  massimo 5.6e-17 su 480 valori, stesse righe). Vista neutra: i punti dei bracci coincidono con
  `faceverse_neutral/graded.csv`.
- **Delta braccio - concorrente** (factorized s1234/s2345 d_F cal. e ctrlfr con FR; factorized d_P e ctrlfr con SR; 7
  concorrenti; 28 delta per dominio e GT): **nessun IC sotto 0 in nessun dominio**. A favore del braccio, su 28: HIFI3D
  FR 28, SR 24; dev FaceScape FR 28, SR 26; FaceVerse con espressioni FR 9, SR 17; FaceVerse neutra FR 16, SR 23; FaMoS
  FR 11, SR 7. I restanti sono non risolti.
- I non risolti su HIFI3D e FaceScape con SR sono ctrlfr contro le mesh d'identita' SR (GNM e FLAME): con SR solo d_P di
  factorized supera tutti i concorrenti parametrici con IC sopra 0. Su FaceVerse (entrambe le viste) il varifold non e'
  risolto contro nessun braccio (FR 0.260 / 0.285, contro 0.26-0.37 dei bracci), e GNM con espressioni nemmeno contro
  la maggior parte. FaMoS ha 15 soggetti e IC larghi: ctrlfr batte quasi tutti i concorrenti con FR, factorized nessuno
  con IC sopra 0 (varifold FR 0.688, factorized 0.678 / 0.654).
- GNM (prior VISTO) e' davanti a FLAME (prior non visto) in quasi tutte le celle (es. HIFI3D FR coefficienti 0.619
  contro 0.566, FaceScape SR mesh d'identita' 0.576 contro 0.469); con FR sulle mesh d'identita' FLAME e' davanti su
  HIFI3D (0.639 contro 0.608). FLAME su FaMoS (quasi oracolo, sez. 2) non supera GNM.
- Il fit parametrico non migliora sul NICP su template da cui parte (HIFI3D FR: GNM mesh d'identita' FR 0.608, FLAME
  0.639, NICP su template in mm 0.614; FaceScape FR 0.465 / 0.449 contro 0.542).

## Fit

| vista | modello | mesh | fallite | RMS fit mm, mediana (max) | resid. NICP mm, mediana | s/mesh, mediana | vertici della regione |
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

