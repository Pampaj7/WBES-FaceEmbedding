# Validazione esterna su FaMoS: riconoscimento dell'identita' su scansioni grezze

Generato da `aau/esterna_famos_id/ei.py eval` (regole in `PROTOCOLLO.md`, scritte e committate prima dei numeri). Lettura DESCRITTIVA: nessun test confermativo; i risultati sfavorevoli sono riportati come gli altri.

## Dati

- 15 soggetti di FaMoS TEST, 150 scansioni grezze (10 sequenze per soggetto, le stesse per tutti: bareteeth, blow_cheeks, disgust, fear, happiness, kissing, mouth_down, rolling_lips, surprise, wrinkle_nose), 11175 coppie (675 stessa persona, 10500 persone diverse). Nessuna registrazione FLAME: topologie tutte diverse.
- Preparazione comune (rilevatore su render frontale, patch NoW in mm): 150/150 patch, ripieghi del rilevatore sulla grezza 0, sulla patch 0; dettagli in `prep_eval.json`.
- IC 95% percentile, 1000 repliche per soggetto (seme 1234, `famos_eval.bootstrap_counts`); retrieval: query ricampionate per soggetto, galleria fissa; verifica: pesi c_s per le coppie della stessa persona, c_a c_b per le altre.

## Metriche per metodo

valore [IC 95%]. rank-1, mAP, AUC: piu' alto = meglio; EER: piu' basso = meglio.

| metodo | rank-1 | mAP | EER | AUC | coppie non finite |
|---|---|---|---|---|---|
| massivo principale (330k passi), d_F non cal. | 0.893 [0.840, 0.940] | 0.720 [0.646, 0.787] | 0.216 [0.163, 0.271] | 0.855 [0.806, 0.908] | 0 |
| massivo principale (330k passi), d_F cal. | 0.880 [0.820, 0.933] | 0.722 [0.642, 0.798] | 0.217 [0.156, 0.274] | 0.866 [0.807, 0.920] | 0 |
| massivo principale (330k passi), d_P | 0.907 [0.860, 0.953] | 0.684 [0.611, 0.753] | 0.225 [0.176, 0.279] | 0.842 [0.792, 0.895] | 0 |
| massivo aperto (230k passi), d_F non cal. | 0.867 [0.800, 0.920] | 0.669 [0.585, 0.745] | 0.237 [0.182, 0.287] | 0.832 [0.782, 0.880] | 0 |
| massivo aperto (230k passi), d_F cal. | 0.887 [0.840, 0.933] | 0.692 [0.610, 0.771] | 0.222 [0.172, 0.274] | 0.860 [0.806, 0.909] | 0 |
| massivo aperto (230k passi), d_P | 0.867 [0.800, 0.920] | 0.647 [0.560, 0.726] | 0.241 [0.190, 0.292] | 0.828 [0.777, 0.878] | 0 |
| paper s1234, d_F non cal. | 0.853 [0.813, 0.900] | 0.664 [0.600, 0.739] | 0.242 [0.184, 0.294] | 0.830 [0.783, 0.883] | 0 |
| paper s1234, d_F cal. | 0.873 [0.827, 0.927] | 0.651 [0.585, 0.726] | 0.265 [0.210, 0.323] | 0.816 [0.760, 0.870] | 0 |
| paper s1234, d_P | 0.813 [0.753, 0.873] | 0.636 [0.562, 0.716] | 0.241 [0.187, 0.294] | 0.830 [0.781, 0.885] | 0 |
| ICP + Chamfer in mm | 0.947 [0.913, 0.980] | 0.822 [0.773, 0.873] | 0.173 [0.121, 0.225] | 0.895 [0.858, 0.932] | 0 |
| ICP + Chamfer cs | 0.913 [0.873, 0.947] | 0.753 [0.686, 0.818] | 0.221 [0.164, 0.261] | 0.860 [0.818, 0.907] | 0 |
| NoW metrical (rilevatore) | 0.967 [0.940, 0.993] | 0.865 [0.812, 0.914] | 0.157 [0.101, 0.206] | 0.918 [0.879, 0.955] | 0 |
| NoW originale (rilevatore) | 0.940 [0.907, 0.967] | 0.791 [0.730, 0.850] | 0.196 [0.145, 0.239] | 0.878 [0.837, 0.923] | 0 |
| ULRSSM faces_xyz per coppia, mm | 0.927 [0.867, 0.980] | 0.727 [0.656, 0.795] | 0.225 [0.166, 0.282] | 0.869 [0.809, 0.921] | 0 |
| ULRSSM faces_xyz per coppia, cs | 0.860 [0.813, 0.900] | 0.631 [0.556, 0.707] | 0.273 [0.221, 0.316] | 0.807 [0.755, 0.860] | 0 |
| GW euclid_128_cgw2, mm | 0.820 [0.753, 0.887] | 0.628 [0.554, 0.706] | 0.215 [0.163, 0.261] | 0.872 [0.824, 0.917] | 0 |
| GW euclid_128_cgw2, cs | 0.707 [0.613, 0.787] | 0.449 [0.386, 0.513] | 0.268 [0.222, 0.312] | 0.811 [0.759, 0.861] | 0 |

## Delta appaiati: massivo principale (330k passi), d_F non cal. contro ciascun concorrente

positivo = il nostro meglio (rank-1, mAP, AUC: nostro - concorrente; EER: concorrente - nostro); stesse repliche.

| concorrente | rank-1 | mAP | EER | AUC |
|---|---|---|---|---|
| paper s1234, d_F non cal. | +0.040 [-0.013, +0.080] | +0.056 [+0.010, +0.098] | +0.026 [-0.011, +0.055] | +0.025 [-0.000, +0.052] |
| paper s1234, d_F cal. | +0.020 [-0.040, +0.067] | +0.070 [+0.022, +0.117] | +0.049 [+0.009, +0.088] | +0.039 [+0.003, +0.078] |
| paper s1234, d_P | +0.080 [+0.013, +0.147] | +0.085 [+0.025, +0.136] | +0.025 [-0.008, +0.058] | +0.025 [-0.005, +0.057] |
| ICP + Chamfer in mm | -0.053 [-0.107, -0.007] | -0.102 [-0.178, -0.028] | -0.043 [-0.098, +0.016] | -0.040 [-0.088, +0.008] |
| ICP + Chamfer cs | -0.020 [-0.080, +0.033] | -0.033 [-0.110, +0.038] | +0.004 [-0.057, +0.058] | -0.005 [-0.051, +0.043] |
| NoW metrical (rilevatore) | -0.073 [-0.114, -0.033] | -0.144 [-0.201, -0.080] | -0.059 [-0.107, -0.013] | -0.063 [-0.099, -0.027] |
| NoW originale (rilevatore) | -0.047 [-0.093, -0.007] | -0.071 [-0.126, -0.018] | -0.020 [-0.064, +0.022] | -0.023 [-0.059, +0.015] |
| ULRSSM faces_xyz per coppia, mm | -0.033 [-0.080, +0.027] | -0.007 [-0.076, +0.069] | +0.009 [-0.055, +0.071] | -0.014 [-0.064, +0.041] |
| ULRSSM faces_xyz per coppia, cs | +0.033 [+0.006, +0.060] | +0.089 [+0.029, +0.151] | +0.056 [+0.006, +0.102] | +0.049 [+0.008, +0.092] |
| GW euclid_128_cgw2, mm | +0.073 [-0.013, +0.160] | +0.093 [+0.003, +0.186] | -0.002 [-0.073, +0.062] | -0.017 [-0.084, +0.046] |
| GW euclid_128_cgw2, cs | +0.187 [+0.093, +0.300] | +0.272 [+0.203, +0.338] | +0.051 [+0.005, +0.099] | +0.044 [+0.004, +0.090] |

## Delta appaiati: massivo principale (330k passi), d_F cal. contro ciascun concorrente

positivo = il nostro meglio (rank-1, mAP, AUC: nostro - concorrente; EER: concorrente - nostro); stesse repliche.

| concorrente | rank-1 | mAP | EER | AUC |
|---|---|---|---|---|
| paper s1234, d_F non cal. | +0.027 [-0.027, +0.073] | +0.058 [+0.002, +0.108] | +0.025 [-0.012, +0.062] | +0.036 [+0.002, +0.068] |
| paper s1234, d_F cal. | +0.007 [-0.053, +0.053] | +0.071 [+0.029, +0.114] | +0.048 [+0.019, +0.083] | +0.050 [+0.021, +0.081] |
| paper s1234, d_P | +0.067 [+0.000, +0.140] | +0.087 [+0.017, +0.153] | +0.024 [-0.013, +0.070] | +0.036 [-0.006, +0.080] |
| ICP + Chamfer in mm | -0.067 [-0.120, -0.013] | -0.100 [-0.170, -0.040] | -0.044 [-0.102, +0.015] | -0.029 [-0.078, +0.009] |
| ICP + Chamfer cs | -0.033 [-0.100, +0.027] | -0.031 [-0.111, +0.044] | +0.004 [-0.068, +0.062] | +0.005 [-0.050, +0.059] |
| NoW metrical (rilevatore) | -0.087 [-0.133, -0.040] | -0.143 [-0.191, -0.094] | -0.060 [-0.105, -0.018] | -0.052 [-0.087, -0.023] |
| NoW originale (rilevatore) | -0.060 [-0.107, -0.013] | -0.069 [-0.130, -0.012] | -0.021 [-0.073, +0.027] | -0.013 [-0.059, +0.030] |
| ULRSSM faces_xyz per coppia, mm | -0.047 [-0.100, +0.013] | -0.005 [-0.068, +0.058] | +0.008 [-0.047, +0.060] | -0.003 [-0.047, +0.043] |
| ULRSSM faces_xyz per coppia, cs | +0.020 [-0.014, +0.053] | +0.091 [+0.020, +0.163] | +0.055 [-0.000, +0.108] | +0.059 [+0.009, +0.108] |
| GW euclid_128_cgw2, mm | +0.060 [-0.020, +0.147] | +0.094 [+0.012, +0.169] | -0.002 [-0.070, +0.055] | -0.007 [-0.071, +0.050] |
| GW euclid_128_cgw2, cs | +0.173 [+0.087, +0.287] | +0.274 [+0.204, +0.342] | +0.050 [-0.003, +0.104] | +0.055 [+0.000, +0.107] |

## Delta appaiati: massivo principale (330k passi), d_P contro ciascun concorrente

positivo = il nostro meglio (rank-1, mAP, AUC: nostro - concorrente; EER: concorrente - nostro); stesse repliche.

| concorrente | rank-1 | mAP | EER | AUC |
|---|---|---|---|---|
| paper s1234, d_F non cal. | +0.053 [+0.013, +0.093] | +0.019 [-0.030, +0.063] | +0.017 [-0.021, +0.043] | +0.012 [-0.014, +0.038] |
| paper s1234, d_F cal. | +0.033 [-0.013, +0.080] | +0.033 [-0.029, +0.091] | +0.040 [-0.004, +0.080] | +0.026 [-0.016, +0.069] |
| paper s1234, d_P | +0.093 [+0.040, +0.147] | +0.048 [+0.000, +0.090] | +0.016 [-0.016, +0.041] | +0.012 [-0.013, +0.037] |
| ICP + Chamfer in mm | -0.040 [-0.093, +0.020] | -0.139 [-0.217, -0.059] | -0.052 [-0.113, +0.006] | -0.053 [-0.106, -0.002] |
| ICP + Chamfer cs | -0.007 [-0.060, +0.047] | -0.070 [-0.134, -0.007] | -0.004 [-0.067, +0.043] | -0.019 [-0.065, +0.027] |
| NoW metrical (rilevatore) | -0.060 [-0.107, -0.013] | -0.181 [-0.246, -0.109] | -0.068 [-0.121, -0.020] | -0.076 [-0.118, -0.036] |
| NoW originale (rilevatore) | -0.033 [-0.080, +0.013] | -0.107 [-0.156, -0.057] | -0.029 [-0.075, +0.010] | -0.036 [-0.074, -0.001] |
| ULRSSM faces_xyz per coppia, mm | -0.020 [-0.080, +0.047] | -0.044 [-0.122, +0.036] | -0.000 [-0.067, +0.058] | -0.027 [-0.086, +0.032] |
| ULRSSM faces_xyz per coppia, cs | +0.047 [+0.013, +0.087] | +0.053 [+0.004, +0.103] | +0.047 [-0.001, +0.087] | +0.035 [-0.005, +0.076] |
| GW euclid_128_cgw2, mm | +0.087 [+0.007, +0.167] | +0.056 [-0.047, +0.156] | -0.010 [-0.086, +0.055] | -0.030 [-0.101, +0.037] |
| GW euclid_128_cgw2, cs | +0.200 [+0.120, +0.307] | +0.235 [+0.171, +0.304] | +0.042 [-0.003, +0.083] | +0.031 [-0.010, +0.076] |

## Delta appaiati: massivo aperto (230k passi), d_F non cal. contro ciascun concorrente

positivo = il nostro meglio (rank-1, mAP, AUC: nostro - concorrente; EER: concorrente - nostro); stesse repliche.

| concorrente | rank-1 | mAP | EER | AUC |
|---|---|---|---|---|
| paper s1234, d_F non cal. | +0.013 [-0.053, +0.067] | +0.004 [-0.051, +0.052] | +0.005 [-0.030, +0.040] | +0.002 [-0.030, +0.032] |
| paper s1234, d_F cal. | -0.007 [-0.087, +0.060] | +0.018 [-0.045, +0.071] | +0.028 [-0.012, +0.070] | +0.016 [-0.024, +0.057] |
| paper s1234, d_P | +0.053 [-0.014, +0.113] | +0.033 [-0.032, +0.090] | +0.004 [-0.027, +0.044] | +0.002 [-0.030, +0.036] |
| ICP + Chamfer in mm | -0.080 [-0.147, -0.013] | -0.153 [-0.235, -0.077] | -0.064 [-0.109, -0.011] | -0.063 [-0.107, -0.021] |
| ICP + Chamfer cs | -0.047 [-0.113, +0.013] | -0.084 [-0.164, -0.012] | -0.016 [-0.071, +0.030] | -0.029 [-0.072, +0.015] |
| NoW metrical (rilevatore) | -0.100 [-0.160, -0.047] | -0.196 [-0.261, -0.129] | -0.080 [-0.115, -0.041] | -0.086 [-0.117, -0.053] |
| NoW originale (rilevatore) | -0.073 [-0.140, -0.013] | -0.122 [-0.175, -0.073] | -0.041 [-0.076, -0.004] | -0.047 [-0.077, -0.016] |
| ULRSSM faces_xyz per coppia, mm | -0.060 [-0.120, +0.007] | -0.058 [-0.135, +0.019] | -0.012 [-0.063, +0.041] | -0.037 [-0.082, +0.015] |
| ULRSSM faces_xyz per coppia, cs | +0.007 [-0.040, +0.053] | +0.038 [-0.015, +0.093] | +0.036 [-0.001, +0.074] | +0.025 [-0.012, +0.066] |
| GW euclid_128_cgw2, mm | +0.047 [-0.040, +0.127] | +0.041 [-0.062, +0.140] | -0.022 [-0.087, +0.038] | -0.040 [-0.103, +0.018] |
| GW euclid_128_cgw2, cs | +0.160 [+0.067, +0.253] | +0.220 [+0.140, +0.295] | +0.031 [-0.016, +0.084] | +0.021 [-0.028, +0.070] |

## Delta appaiati: massivo aperto (230k passi), d_F cal. contro ciascun concorrente

positivo = il nostro meglio (rank-1, mAP, AUC: nostro - concorrente; EER: concorrente - nostro); stesse repliche.

| concorrente | rank-1 | mAP | EER | AUC |
|---|---|---|---|---|
| paper s1234, d_F non cal. | +0.033 [-0.013, +0.073] | +0.027 [-0.039, +0.088] | +0.020 [-0.016, +0.061] | +0.031 [-0.012, +0.070] |
| paper s1234, d_F cal. | +0.013 [-0.040, +0.067] | +0.041 [-0.002, +0.082] | +0.043 [+0.018, +0.073] | +0.045 [+0.016, +0.072] |
| paper s1234, d_P | +0.073 [+0.007, +0.153] | +0.056 [-0.023, +0.140] | +0.019 [-0.021, +0.066] | +0.030 [-0.017, +0.079] |
| ICP + Chamfer in mm | -0.060 [-0.093, -0.027] | -0.130 [-0.195, -0.075] | -0.049 [-0.090, -0.004] | -0.035 [-0.076, +0.001] |
| ICP + Chamfer cs | -0.027 [-0.080, +0.033] | -0.061 [-0.141, +0.022] | -0.001 [-0.064, +0.044] | -0.000 [-0.053, +0.048] |
| NoW metrical (rilevatore) | -0.080 [-0.120, -0.040] | -0.173 [-0.218, -0.129] | -0.065 [-0.096, -0.036] | -0.058 [-0.088, -0.030] |
| NoW originale (rilevatore) | -0.053 [-0.093, -0.007] | -0.099 [-0.163, -0.029] | -0.026 [-0.067, +0.011] | -0.018 [-0.064, +0.025] |
| ULRSSM faces_xyz per coppia, mm | -0.040 [-0.087, +0.000] | -0.035 [-0.087, +0.016] | +0.003 [-0.034, +0.039] | -0.008 [-0.046, +0.030] |
| ULRSSM faces_xyz per coppia, cs | +0.027 [-0.013, +0.067] | +0.061 [-0.011, +0.140] | +0.050 [+0.004, +0.095] | +0.054 [+0.002, +0.105] |
| GW euclid_128_cgw2, mm | +0.067 [+0.007, +0.127] | +0.064 [-0.014, +0.130] | -0.007 [-0.061, +0.040] | -0.012 [-0.069, +0.035] |
| GW euclid_128_cgw2, cs | +0.180 [+0.093, +0.293] | +0.243 [+0.163, +0.329] | +0.045 [-0.005, +0.099] | +0.049 [-0.008, +0.108] |

## Delta appaiati: massivo aperto (230k passi), d_P contro ciascun concorrente

positivo = il nostro meglio (rank-1, mAP, AUC: nostro - concorrente; EER: concorrente - nostro); stesse repliche.

| concorrente | rank-1 | mAP | EER | AUC |
|---|---|---|---|---|
| paper s1234, d_F non cal. | +0.013 [-0.053, +0.073] | -0.018 [-0.078, +0.035] | +0.000 [-0.034, +0.033] | -0.002 [-0.034, +0.027] |
| paper s1234, d_F cal. | -0.007 [-0.087, +0.067] | -0.004 [-0.074, +0.058] | +0.023 [-0.016, +0.066] | +0.012 [-0.032, +0.052] |
| paper s1234, d_P | +0.053 [-0.020, +0.113] | +0.011 [-0.054, +0.066] | -0.000 [-0.031, +0.036] | -0.002 [-0.034, +0.029] |
| ICP + Chamfer in mm | -0.080 [-0.140, -0.033] | -0.176 [-0.264, -0.092] | -0.068 [-0.121, -0.014] | -0.067 [-0.115, -0.022] |
| ICP + Chamfer cs | -0.047 [-0.107, +0.000] | -0.106 [-0.182, -0.037] | -0.021 [-0.079, +0.028] | -0.032 [-0.077, +0.011] |
| NoW metrical (rilevatore) | -0.100 [-0.160, -0.053] | -0.218 [-0.291, -0.146] | -0.084 [-0.124, -0.044] | -0.090 [-0.124, -0.057] |
| NoW originale (rilevatore) | -0.073 [-0.140, -0.020] | -0.144 [-0.194, -0.095] | -0.045 [-0.083, -0.010] | -0.050 [-0.082, -0.019] |
| ULRSSM faces_xyz per coppia, mm | -0.060 [-0.120, -0.007] | -0.080 [-0.162, -0.002] | -0.016 [-0.068, +0.034] | -0.041 [-0.090, +0.012] |
| ULRSSM faces_xyz per coppia, cs | +0.007 [-0.040, +0.047] | +0.016 [-0.035, +0.068] | +0.031 [-0.009, +0.068] | +0.022 [-0.016, +0.059] |
| GW euclid_128_cgw2, mm | +0.047 [-0.047, +0.133] | +0.019 [-0.089, +0.126] | -0.027 [-0.094, +0.034] | -0.044 [-0.110, +0.015] |
| GW euclid_128_cgw2, cs | +0.160 [+0.067, +0.260] | +0.198 [+0.116, +0.279] | +0.026 [-0.022, +0.075] | +0.017 [-0.032, +0.065] |

## Controlli di riproduzione (contro la matrice, FaMoS TEST galleria)

- `embed_massive_main`: `{"ok": true, "Z_galleria_famos_vs_matrice": {"n": 3855, "nan": 0, "max_abs": 7.539987564086914e-05, "median_abs": 0.0, "p95_abs": 1.6251206398010246e-05, "equal": 2388}}`
- `embed_massive_open`: `{"ok": true, "Z_galleria_famos_vs_matrice": {"n": 3855, "nan": 0, "max_abs": 1.4901161193847656e-07, "median_abs": 0.0, "p95_abs": 5.960464477539063e-08, "equal": 2617}}`
- `massive_main`: `{}`
- `massive_open`: `{}`

File: `metrics.csv` (una riga per metodo e metrica), `deltas.csv` (tutti i delta appaiati), `summary.json`, `selection.json`, `prep_eval.json`, `prep_pilot.json`, `controls/`.

<!-- da qui in giu': scritto a mano, conservato da ei.py eval -->
