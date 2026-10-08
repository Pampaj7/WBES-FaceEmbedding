# E2: canonicalizzazione rigida al test, e108 con e senza

Protocollo dichiarato prima: `aau/runs/evidence/e2/protocol.md`; calibrazione e soglia: `calibration/calibration.md` (T = 0.0811). Script: `aau/evidence/e2_canon/`. CI 95% bootstrap per soggetto, 1000 repliche; P(<=0) = frazione di repliche con differenza <= 0.

## Canonicalizzazione: fallimenti, tempo, angoli

### HIFI3D

| topology | n | fallimenti (residuo > T) | residuo mediano | p95 | angolo dalla convenzione, mediana | p95 | > 30 gradi | s/mesh mediani | start scelti |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crop | 100 | 0 (0.0%) | 0.0256 | 0.0358 | 5.02 | 7.23 | 0 | 3.87 | I 98, Rx+90 2 |
| down8k | 100 | 0 (0.0%) | 0.0247 | 0.0329 | 5.17 | 7.75 | 0 | 4.24 | I 97, Rx+90 3 |
| noisy | 100 | 0 (0.0%) | 0.0249 | 0.0330 | 5.37 | 7.81 | 0 | 4.11 | I 99, Rx-90 1 |
| original | 100 | 0 (0.0%) | 0.0246 | 0.0333 | 5.24 | 7.69 | 0 | 4.23 | I 96, Rx+90 4 |
| remesh | 100 | 0 (0.0%) | 0.0247 | 0.0334 | 5.29 | 7.60 | 0 | 4.15 | I 96, Rx+90 4 |
| up60k | 100 | 0 (0.0%) | 0.0247 | 0.0331 | 5.21 | 7.67 | 0 | 4.26 | I 95, Rx+90 5 |

Consistenza fra topologie dello stesso soggetto:

| topologia | angolo da R(original), mediana | p95 | max |
| --- | --- | --- | --- |
| crop | 0.91 | 2.60 | 3.44 |
| down8k | 0.48 | 1.39 | 3.01 |
| noisy | 0.73 | 2.08 | 2.55 |
| remesh | 0.58 | 1.64 | 2.85 |
| up60k | 0.49 | 1.80 | 2.60 |

### FaceVerse con espressioni

| topology | n | fallimenti (residuo > T) | residuo mediano | p95 | angolo dalla convenzione, mediana | p95 | > 30 gradi | s/mesh mediani | start scelti |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crop | 100 | 0 (0.0%) | 0.0296 | 0.0368 | 3.67 | 6.43 | 0 | 4.08 | Rx180 95, Rx-90 4, Rx+90 1 |
| down8k | 100 | 0 (0.0%) | 0.0306 | 0.0367 | 3.70 | 7.56 | 0 | 4.13 | Rx180 91, Rx-90 5, Rx+90 4 |
| noisy | 100 | 0 (0.0%) | 0.0301 | 0.0354 | 3.50 | 6.44 | 0 | 4.20 | Rx180 96, Rx-90 3, Rx+90 1 |
| original | 100 | 0 (0.0%) | 0.0296 | 0.0355 | 3.53 | 6.67 | 0 | 4.12 | Rx180 92, Rx+90 6, Rx-90 2 |
| remesh | 100 | 0 (0.0%) | 0.0304 | 0.0349 | 3.19 | 6.64 | 0 | 4.06 | Rx180 96, Rx+90 3, Rx-90 1 |
| up60k | 100 | 0 (0.0%) | 0.0291 | 0.0368 | 3.49 | 6.43 | 0 | 4.15 | Rx180 98, Rx+90 2 |

Consistenza fra topologie dello stesso soggetto:

| topologia | angolo da R(original), mediana | p95 | max |
| --- | --- | --- | --- |
| crop | 2.61 | 6.03 | 7.30 |
| down8k | 2.57 | 5.80 | 16.60 |
| noisy | 2.92 | 6.05 | 7.86 |
| remesh | 2.83 | 6.03 | 8.52 |
| up60k | 2.65 | 5.52 | 6.65 |

### NoW

| group | n | fallimenti (residuo > T) | residuo mediano | p95 | angolo dalla convenzione, mediana | p95 | > 30 gradi | s/mesh mediani | start scelti |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3ddfa_v2 | 352 | 0 (0.0%) | 0.0168 | 0.0201 | 1.52 | 6.51 | 0 | 3.98 | I 352 |
| mica | 352 | 0 (0.0%) | 0.0189 | 0.0285 | 1.92 | 4.97 | 0 | 4.09 | I 347, Rx+90 5 |
| prnet | 352 | 1 (0.3%) | 0.0189 | 0.0241 | 2.59 | 7.59 | 1 | 4.00 | I 351, Ry180 1 |
| scan_face | 20 | 0 (0.0%) | 0.0196 | 0.0298 | 2.94 | 4.30 | 0 | 4.10 | I 19, Rx+90 1 |
| synergynet | 352 | 0 (0.0%) | 0.0159 | 0.0195 | 1.81 | 4.43 | 0 | 4.01 | I 352 |

Riepilogo (fallimento = residuo > T, IC di Wilson; tempo = secondi per mesh su 1 thread, 8 start):

| dominio | n | fallimenti | tasso [IC 95%] | s/mesh mediana (p95) | angolo dalla convenzione mediana (p95) | > 30 gradi | residuo mediano |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HIFI3D | 600 | 0 | 0.000 [0.000, 0.006] | 4.15 (4.71) | 5.18 (7.68) | 0 | 0.0248 |
| FaceVerse con espressioni | 600 | 0 | 0.000 [0.000, 0.006] | 4.13 (4.48) | 3.53 (6.80) | 0 | 0.0301 |
| NoW | 1428 | 1 | 0.001 [0.000, 0.004] | 4.02 (4.31) | 1.96 (6.18) | 1 | 0.0175 |

## HIFI3D: Spearman con la GT maxabs

Righe delle pair_metrics di e108 (coppie di soggetti diversi, coppie ordinate di topologie); ogni gruppo col seme della riga e108 pubblicata, per tutte le righe e le differenze del gruppo.

| metodo | nocrop_cross | all_cross | subject_pair_mean |
| --- | --- | --- | --- |
| e108, senza canonicalizzazione | 0.630 [0.569, 0.689] | 0.541 [0.471, 0.605] | 0.795 [0.738, 0.840] |
| e108, canonicalizzato | 0.555 [0.481, 0.624] | 0.498 [0.415, 0.570] | 0.716 [0.632, 0.783] |
| Chamfer faceBench, senza canonicalizzazione | 0.325 [0.283, 0.369] | 0.291 [0.250, 0.330] | 0.751 [0.691, 0.803] |
| Chamfer faceBench, canonicalizzato | 0.264 [0.223, 0.312] | 0.247 [0.207, 0.286] | 0.686 [0.611, 0.751] |
| Chamfer eval (riferimento del summary) | 0.372 [0.326, 0.420] | 0.336 [0.291, 0.380] | 0.743 [0.678, 0.798] |

Differenze appaiate:

| A - B | nocrop_cross [IC] (P<=0) | all_cross [IC] (P<=0) | subject_pair_mean [IC] (P<=0) |
| --- | --- | --- | --- |
| e108, canonicalizzato - e108, senza canonicalizzazione | -0.074 [-0.117, -0.039] (1.000) | -0.043 [-0.080, -0.011] (0.996) | -0.080 [-0.126, -0.041] (1.000) |
| Chamfer faceBench, canonicalizzato - Chamfer faceBench, senza canonicalizzazione | -0.061 [-0.086, -0.038] (1.000) | -0.044 [-0.065, -0.024] (1.000) | -0.065 [-0.117, -0.016] (0.992) |
| e108, canonicalizzato - Chamfer faceBench, canonicalizzato | +0.291 [+0.236, +0.341] (0.000) | +0.251 [+0.188, +0.302] (0.000) | +0.030 [-0.038, +0.087] (0.202) |
| e108, senza canonicalizzazione - Chamfer faceBench, senza canonicalizzazione | +0.305 [+0.255, +0.350] (0.000) | +0.251 [+0.191, +0.299] (0.000) | +0.045 [-0.015, +0.098] (0.063) |

## HIFI3D: riconoscimento, 5 topologie senza crop (primario)

| metodo | rank-1 | mAP | AUC |
| --- | --- | --- | --- |
| e108, senza canonicalizzazione | 0.782 [0.754, 0.807] | 0.845 [0.822, 0.863] | 0.946 [0.934, 0.957] |
| e108, canonicalizzato | 0.733 [0.700, 0.763] | 0.804 [0.779, 0.828] | 0.936 [0.922, 0.948] |
| Chamfer faceBench, senza canonicalizzazione | 0.477 [0.445, 0.513] | 0.564 [0.534, 0.594] | 0.782 [0.767, 0.797] |
| Chamfer faceBench, canonicalizzato | 0.403 [0.373, 0.432] | 0.487 [0.459, 0.516] | 0.745 [0.730, 0.761] |
| Rigid ICP + Chamfer (riferimento) | 0.996 [0.991, 0.999] | 0.997 [0.995, 0.999] | 0.999 [0.999, 1.000] |

Differenze appaiate (stesse repliche):

| A - B | rank-1 [IC] (P<=0) | mAP | AUC |
| --- | --- | --- | --- |
| e108, canonicalizzato - e108, senza canonicalizzazione | -0.050 [-0.071, -0.028] (1.000) | -0.041 [-0.054, -0.027] (1.000) | -0.010 [-0.015, -0.005] (1.000) |
| Chamfer faceBench, canonicalizzato - Chamfer faceBench, senza canonicalizzazione | -0.075 [-0.095, -0.055] (1.000) | -0.078 [-0.093, -0.063] (1.000) | -0.037 [-0.045, -0.029] (1.000) |
| e108, canonicalizzato - Chamfer faceBench, canonicalizzato | +0.330 [+0.299, +0.360] (0.000) | +0.317 [+0.292, +0.341] (0.000) | +0.190 [+0.181, +0.200] (0.000) |
| e108, senza canonicalizzazione - Chamfer faceBench, senza canonicalizzazione | +0.305 [+0.278, +0.336] (0.000) | +0.280 [+0.258, +0.304] (0.000) | +0.163 [+0.153, +0.174] (0.000) |

## HIFI3D: riconoscimento, coppie con crop (a parte)

| metodo | rank-1 | mAP | AUC |
| --- | --- | --- | --- |
| e108, senza canonicalizzazione | 0.348 [0.299, 0.395] | 0.475 [0.433, 0.516] | 0.843 [0.822, 0.865] |
| e108, canonicalizzato | 0.306 [0.266, 0.347] | 0.450 [0.410, 0.487] | 0.843 [0.821, 0.863] |
| Chamfer faceBench, senza canonicalizzazione | 0.472 [0.445, 0.499] | 0.536 [0.510, 0.564] | 0.699 [0.684, 0.715] |
| Chamfer faceBench, canonicalizzato | 0.427 [0.405, 0.448] | 0.503 [0.483, 0.525] | 0.690 [0.676, 0.706] |
| Rigid ICP + Chamfer (riferimento) | 0.284 [0.237, 0.338] | 0.423 [0.381, 0.470] | 0.888 [0.865, 0.911] |

Differenze appaiate (stesse repliche):

| A - B | rank-1 [IC] (P<=0) | mAP | AUC |
| --- | --- | --- | --- |
| e108, canonicalizzato - e108, senza canonicalizzazione | -0.042 [-0.078, -0.010] (0.994) | -0.025 [-0.052, -0.001] (0.982) | -0.001 [-0.010, +0.009] (0.576) |
| Chamfer faceBench, canonicalizzato - Chamfer faceBench, senza canonicalizzazione | -0.045 [-0.067, -0.023] (1.000) | -0.033 [-0.048, -0.019] (1.000) | -0.009 [-0.015, -0.003] (0.997) |
| e108, canonicalizzato - Chamfer faceBench, canonicalizzato | -0.121 [-0.158, -0.079] (1.000) | -0.053 [-0.085, -0.020] (1.000) | +0.152 [+0.139, +0.165] (0.000) |
| e108, senza canonicalizzazione - Chamfer faceBench, senza canonicalizzazione | -0.124 [-0.163, -0.086] (1.000) | -0.061 [-0.091, -0.029] (1.000) | +0.144 [+0.131, +0.158] (0.000) |

## FaceVerse con espressioni: riconoscimento, 5 topologie senza crop (primario)

| metodo | rank-1 | mAP | AUC |
| --- | --- | --- | --- |
| e108, senza canonicalizzazione | 0.625 [0.584, 0.668] | 0.689 [0.650, 0.727] | 0.869 [0.842, 0.894] |
| e108, canonicalizzato | 0.632 [0.582, 0.679] | 0.701 [0.654, 0.744] | 0.892 [0.866, 0.915] |
| e108, convenzione ICT fissa (Rx180, job di un altro agente) | 0.640 [0.591, 0.685] | 0.702 [0.658, 0.744] | 0.879 [0.854, 0.903] |
| Chamfer faceBench, senza canonicalizzazione | 0.740 [0.698, 0.783] | 0.775 [0.737, 0.814] | 0.882 [0.853, 0.910] |
| Chamfer faceBench, canonicalizzato | 0.721 [0.679, 0.763] | 0.758 [0.719, 0.797] | 0.881 [0.850, 0.909] |
| Rigid ICP + Chamfer (riferimento) | 0.918 [0.895, 0.939] | 0.935 [0.916, 0.953] | 0.986 [0.979, 0.992] |

Differenze appaiate (stesse repliche):

| A - B | rank-1 [IC] (P<=0) | mAP | AUC |
| --- | --- | --- | --- |
| e108, canonicalizzato - e108, senza canonicalizzazione | +0.007 [-0.027, +0.040] (0.358) | +0.012 [-0.018, +0.041] (0.204) | +0.023 [+0.006, +0.041] (0.003) |
| Chamfer faceBench, canonicalizzato - Chamfer faceBench, senza canonicalizzazione | -0.019 [-0.037, -0.002] (0.987) | -0.017 [-0.033, -0.003] (0.992) | -0.002 [-0.011, +0.007] (0.655) |
| e108, canonicalizzato - Chamfer faceBench, canonicalizzato | -0.089 [-0.123, -0.056] (1.000) | -0.057 [-0.087, -0.029] (1.000) | +0.011 [-0.006, +0.028] (0.093) |
| e108, senza canonicalizzazione - Chamfer faceBench, senza canonicalizzazione | -0.115 [-0.155, -0.076] (1.000) | -0.086 [-0.122, -0.052] (1.000) | -0.014 [-0.035, +0.008] (0.897) |
| e108, convenzione ICT fissa (Rx180, job di un altro agente) - e108, senza canonicalizzazione | +0.015 [-0.016, +0.045] (0.176) | +0.014 [-0.015, +0.039] (0.155) | +0.010 [-0.003, +0.025] (0.065) |
| e108, canonicalizzato - e108, convenzione ICT fissa (Rx180, job di un altro agente) | -0.008 [-0.028, +0.011] (0.832) | -0.001 [-0.017, +0.015] (0.564) | +0.013 [+0.005, +0.020] (0.000) |

## FaceVerse con espressioni: riconoscimento, coppie con crop (a parte)

| metodo | rank-1 | mAP | AUC |
| --- | --- | --- | --- |
| e108, senza canonicalizzazione | 0.252 [0.208, 0.297] | 0.375 [0.332, 0.418] | 0.788 [0.761, 0.816] |
| e108, canonicalizzato | 0.252 [0.211, 0.299] | 0.380 [0.340, 0.425] | 0.807 [0.781, 0.833] |
| e108, convenzione ICT fissa (Rx180, job di un altro agente) | 0.288 [0.238, 0.341] | 0.403 [0.358, 0.453] | 0.806 [0.780, 0.832] |
| Chamfer faceBench, senza canonicalizzazione | 0.396 [0.336, 0.457] | 0.479 [0.421, 0.537] | 0.809 [0.781, 0.837] |
| Chamfer faceBench, canonicalizzato | 0.335 [0.275, 0.393] | 0.420 [0.362, 0.473] | 0.790 [0.760, 0.819] |
| Rigid ICP + Chamfer (riferimento) | 0.513 [0.455, 0.571] | 0.590 [0.540, 0.640] | 0.777 [0.734, 0.814] |

Differenze appaiate (stesse repliche):

| A - B | rank-1 [IC] (P<=0) | mAP | AUC |
| --- | --- | --- | --- |
| e108, canonicalizzato - e108, senza canonicalizzazione | +0.000 [-0.053, +0.055] (0.491) | +0.005 [-0.043, +0.053] (0.410) | +0.018 [-0.005, +0.040] (0.065) |
| Chamfer faceBench, canonicalizzato - Chamfer faceBench, senza canonicalizzazione | -0.061 [-0.095, -0.029] (1.000) | -0.059 [-0.087, -0.031] (1.000) | -0.019 [-0.034, -0.005] (0.995) |
| e108, canonicalizzato - Chamfer faceBench, canonicalizzato | -0.083 [-0.139, -0.019] (0.995) | -0.040 [-0.089, +0.016] (0.919) | +0.017 [-0.008, +0.044] (0.094) |
| e108, senza canonicalizzazione - Chamfer faceBench, senza canonicalizzazione | -0.144 [-0.208, -0.077] (1.000) | -0.104 [-0.164, -0.043] (0.999) | -0.021 [-0.048, +0.009] (0.925) |
| e108, convenzione ICT fissa (Rx180, job di un altro agente) - e108, senza canonicalizzazione | +0.036 [-0.018, +0.089] (0.096) | +0.029 [-0.020, +0.077] (0.128) | +0.017 [-0.005, +0.038] (0.064) |
| e108, canonicalizzato - e108, convenzione ICT fissa (Rx180, job di un altro agente) | -0.036 [-0.069, -0.003] (0.983) | -0.024 [-0.050, +0.003] (0.956) | +0.001 [-0.011, +0.014] (0.456) |

## NoW: concordanza con l'errore NoW (3 metodi pre-registrati)

| metrica | tau per immagine [IC] | Spearman per ricostruzione [IC] |
| --- | --- | --- |
| e108, senza canonicalizzazione | 0.277 [0.137, 0.417] | 0.332 [0.088, 0.549] |
| e108, canonicalizzato | 0.265 [0.149, 0.395] | 0.416 [0.187, 0.601] |
| Chamfer grezza, senza canonicalizzazione | 0.246 [0.123, 0.373] | 0.466 [0.245, 0.637] |
| Chamfer grezza, canonicalizzata | 0.233 [0.116, 0.351] | 0.499 [0.303, 0.644] |
| ICP + Chamfer (mm) | 0.612 [0.537, 0.686] | 0.866 [0.758, 0.924] |
| ICP + Chamfer (mm), mesh canonicalizzate | 0.614 [0.543, 0.689] | 0.836 [0.702, 0.914] |

Differenze appaiate (stesse repliche dei 20 soggetti):

| A - B | tau per immagine [IC] (P<=0) | Spearman [IC] (P<=0) |
| --- | --- | --- |
| e108, canonicalizzato - e108, senza canonicalizzazione | -0.011 [-0.061, +0.048] (0.682) | +0.084 [-0.051, +0.217] (0.121) |
| Chamfer grezza, canonicalizzata - Chamfer grezza, senza canonicalizzazione | -0.013 [-0.073, +0.035] (0.707) | +0.033 [-0.054, +0.121] (0.256) |
| e108, canonicalizzato - Chamfer grezza, canonicalizzata | +0.032 [-0.093, +0.156] (0.294) | -0.084 [-0.289, +0.130] (0.762) |
| e108, senza canonicalizzazione - Chamfer grezza, senza canonicalizzazione | +0.030 [-0.103, +0.153] (0.318) | -0.134 [-0.316, +0.040] (0.928) |

## Controlli: le righe senza canonicalizzazione riprodotte

| riga | pubblicato | ricalcolato | max |diff| (punto, IC) |
| --- | --- | --- | --- |
| e108, senza canonicalizzazione, all_cross | 0.541 [0.471, 0.605] | 0.541 [0.471, 0.605] | 0.00e+00 |
| Chamfer eval (riferimento del summary), all_cross | 0.336 [0.290, 0.380] | 0.336 [0.290, 0.380] | 0.00e+00 |
| e108, senza canonicalizzazione, nocrop_cross | 0.630 [0.569, 0.689] | 0.630 [0.569, 0.689] | 0.00e+00 |
| Chamfer eval (riferimento del summary), nocrop_cross | 0.372 [0.324, 0.422] | 0.372 [0.324, 0.422] | 5.55e-17 |
| e108, senza canonicalizzazione, subject_pair_mean | 0.795 [0.738, 0.840] | 0.795 [0.738, 0.840] | 0.00e+00 |
| Chamfer eval (riferimento del summary), subject_pair_mean | 0.743 [0.682, 0.796] | 0.743 [0.682, 0.796] | 0.00e+00 |

| riga (riconoscimento HIFI3D) | pubblicato rank-1 | ricalcolato rank-1 | max |diff| (rank-1, mAP, AUC, IC) |
| --- | --- | --- | --- |
| e108, senza canonicalizzazione | 0.782 [0.754, 0.807] | 0.782 [0.754, 0.807] | 1.11e-16 |
| Chamfer faceBench, senza canonicalizzazione | 0.477 [0.445, 0.513] | 0.477 [0.445, 0.513] | 0.00e+00 |
| Rigid ICP + Chamfer (riferimento) | 0.996 [0.991, 0.999] | 0.996 [0.991, 0.999] | 1.11e-16 |

| riga (riconoscimento FaceVerse) | pubblicato rank-1 | ricalcolato rank-1 | max |diff| |
| --- | --- | --- | --- |
| e108, senza canonicalizzazione | 0.625 [0.584, 0.668] | 0.625 [0.584, 0.668] | 0.00e+00 |
| Chamfer faceBench, senza canonicalizzazione | 0.740 [0.698, 0.783] | 0.740 [0.698, 0.783] | 0.00e+00 |
| Rigid ICP + Chamfer (riferimento) | 0.918 [0.895, 0.939] | 0.918 [0.895, 0.939] | 1.11e-16 |

| riga (NoW, tau per immagine) | pubblicato | ricalcolato | max |diff| (tau, Spearman, IC) |
| --- | --- | --- | --- |
| e108, senza canonicalizzazione | 0.277 [0.137, 0.417] | 0.277 [0.137, 0.417] | 8.33e-17 |
| Chamfer grezza, senza canonicalizzazione | 0.246 [0.123, 0.373] | 0.246 [0.123, 0.373] | 8.33e-17 |
| ICP + Chamfer (mm) | 0.612 [0.537, 0.686] | 0.612 [0.537, 0.686] | 1.11e-16 |

- e108: distanze dagli embedding contro latent_distance delle pair_metrics, max |diff|: 5.98e-07
- Chamfer faceBench, hifi: la funzione usata sulle mesh canonicalizzate, rifatta sulla vista originale per due coppie di topologie, contro le matrici esistenti: original->down8k (x) max |diff| 0.00e+00; original->down8k (s) max |diff| 0.00e+00; noisy->remesh (x) max |diff| 0.00e+00; noisy->remesh (s) max |diff| 0.00e+00
- Chamfer faceBench, fv: la funzione usata sulle mesh canonicalizzate, rifatta sulla vista originale per due coppie di topologie, contro le matrici esistenti: original->down8k (x) max |diff| 0.00e+00; original->down8k (s) max |diff| 0.00e+00; noisy->remesh (x) max |diff| 0.00e+00; noisy->remesh (s) max |diff| 0.00e+00
