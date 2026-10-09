# Ablazioni v3 sulla cella C3F: risultati e regola di adozione

Generato da `v3_work/trainer/tools/ablation_summary.py`. Protocolli (sha256 registrati prima dei numeri): `ablation_protocol.md` invariato; `ablation_protocol_emendamento_2026-10-09.md` invariato. Un seme per braccio; IC 95% bootstrap per soggetto (1.000 repliche), differenze appaiate sulle stesse repliche. Pesi EMA.

## Verdetti (21.096 passi)

| braccio | esito | motivo |
| --- | --- | --- |
| area | **NO: resta il default v2** | dev -0.019 [-0.049, +0.016] (serve >= +0.03 con IC > 0); nessun calo oltre 0.05 |
| arearobust | **PASSA** | dev +0.122 [+0.086, +0.160]; nessun calo oltre 0.05 |
| bal | **PASSA** | dev +0.050 [+0.032, +0.070]; nessun calo oltre 0.05 |
| ugtmix | **NO: resta il default v2** | dev -0.131 [-0.161, -0.104] (serve >= +0.03 con IC > 0); cali oltre 0.05: sp|hifi|maxabs|nocrop_cross, rec|fv|rank1 |
| loginv | **NO: resta il default v2** | dev +0.116 [+0.064, +0.162]; cali oltre 0.05: sp|hifi|maxabs|nocrop_cross, rec|fv|rank1 |

## Valori, 21096 passi

| braccio | punteggio dev | dev graduata | dev rank-1 espr. | HIFI3D graduata, GT maxabs | HIFI3D graduata, GT unif. | FaceVerse espr. rank-1 | HIFI3D rank-1 | NoW tau |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ctrl (v2) | 0.321 [0.286, 0.356] | 0.372 [0.322, 0.420] | 0.271 [0.236, 0.303] | 0.659 [0.589, 0.723] | 0.321 [0.254, 0.388] | 0.597 [0.554, 0.639] | 0.773 [0.747, 0.801] | 0.263 [0.138, 0.396] |
| area | 0.303 [0.268, 0.338] | 0.272 [0.220, 0.327] | 0.334 [0.299, 0.370] | 0.668 [0.592, 0.732] | 0.317 [0.239, 0.394] | 0.697 [0.661, 0.730] | 0.966 [0.950, 0.980] | 0.299 [0.180, 0.415] |
| arearobust | 0.443 [0.398, 0.490] | 0.395 [0.316, 0.471] | 0.491 [0.445, 0.535] | 0.636 [0.555, 0.710] | 0.342 [0.266, 0.424] | 0.819 [0.778, 0.858] | 0.996 [0.992, 0.999] | 0.265 [0.155, 0.370] |
| bal | 0.371 [0.329, 0.411] | 0.447 [0.384, 0.509] | 0.295 [0.260, 0.329] | 0.622 [0.549, 0.692] | 0.304 [0.242, 0.366] | 0.658 [0.611, 0.701] | 0.704 [0.677, 0.732] | 0.244 [0.115, 0.378] |
| ugtmix | 0.190 [0.158, 0.221] | 0.248 [0.200, 0.294] | 0.133 [0.108, 0.156] | 0.349 [0.280, 0.420] | 0.466 [0.375, 0.545] | 0.368 [0.334, 0.404] | 0.650 [0.613, 0.690] | 0.222 [0.169, 0.285] |
| loginv | 0.438 [0.381, 0.487] | 0.499 [0.406, 0.577] | 0.376 [0.337, 0.414] | 0.288 [0.185, 0.403] | 0.417 [0.325, 0.511] | 0.539 [0.489, 0.590] | 0.895 [0.864, 0.923] | 0.409 [0.326, 0.500] |

Differenze braccio - ctrl, 21096 passi: delta [IC 95%] (P(boot <= 0))

| braccio | punteggio dev | dev graduata | dev rank-1 espr. | HIFI3D graduata, GT maxabs | HIFI3D graduata, GT unif. | FaceVerse espr. rank-1 | HIFI3D rank-1 | NoW tau |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| area | -0.019 [-0.049, +0.016] (0.86) | -0.100 [-0.146, -0.049] (1.00) | +0.063 [+0.030, +0.099] (0.00) | +0.009 [-0.028, +0.049] (0.30) | -0.004 [-0.043, +0.039] (0.57) | +0.100 [+0.064, +0.138] (0.00) | +0.192 [+0.162, +0.220] (0.00) | +0.036 [-0.055, +0.120] (0.22) |
| arearobust | +0.122 [+0.086, +0.160] (0.00) | +0.023 [-0.037, +0.089] (0.22) | +0.220 [+0.180, +0.262] (0.00) | -0.023 [-0.066, +0.020] (0.82) | +0.021 [-0.028, +0.070] (0.20) | +0.222 [+0.184, +0.260] (0.00) | +0.223 [+0.195, +0.249] (0.00) | +0.002 [-0.109, +0.102] (0.50) |
| bal | +0.050 [+0.032, +0.070] (0.00) | +0.075 [+0.045, +0.108] (0.00) | +0.025 [+0.004, +0.047] (0.01) | -0.037 [-0.058, -0.016] (1.00) | -0.016 [-0.040, +0.009] (0.90) | +0.061 [+0.043, +0.080] (0.00) | -0.070 [-0.090, -0.050] (1.00) | -0.019 [-0.056, +0.020] (0.85) |
| ugtmix | -0.131 [-0.161, -0.104] (1.00) | -0.124 [-0.165, -0.086] (1.00) | -0.138 [-0.173, -0.106] (1.00) | -0.310 [-0.396, -0.211] (1.00) | +0.145 [+0.064, +0.210] (0.00) | -0.228 [-0.256, -0.200] (1.00) | -0.123 [-0.154, -0.094] (1.00) | -0.042 [-0.167, +0.083] (0.74) |
| loginv | +0.116 [+0.064, +0.162] (0.00) | +0.127 [+0.042, +0.201] (0.00) | +0.105 [+0.069, +0.143] (0.00) | -0.371 [-0.483, -0.240] (1.00) | +0.097 [+0.013, +0.175] (0.01) | -0.057 [-0.102, -0.013] (0.99) | +0.122 [+0.089, +0.154] (0.00) | +0.146 [+0.034, +0.257] (0.01) |

## Valori, 10548 passi (descrittivo)

| braccio | punteggio dev | dev graduata | dev rank-1 espr. | HIFI3D graduata, GT maxabs | HIFI3D graduata, GT unif. | FaceVerse espr. rank-1 | HIFI3D rank-1 | NoW tau |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ctrl (v2) | 0.332 [0.295, 0.367] | 0.387 [0.337, 0.434] | 0.278 [0.246, 0.310] | 0.693 [0.624, 0.754] | 0.352 [0.277, 0.426] | 0.507 [0.468, 0.549] | 0.870 [0.846, 0.893] | 0.282 [0.182, 0.400] |
| area | 0.354 [0.317, 0.393] | 0.322 [0.262, 0.378] | 0.387 [0.349, 0.426] | 0.644 [0.564, 0.711] | 0.314 [0.235, 0.393] | 0.693 [0.656, 0.730] | 0.952 [0.934, 0.967] | 0.314 [0.212, 0.419] |
| arearobust | 0.506 [0.461, 0.552] | 0.447 [0.367, 0.530] | 0.565 [0.520, 0.611] | 0.612 [0.527, 0.694] | 0.411 [0.330, 0.494] | 0.843 [0.806, 0.879] | 1.000 [1.000, 1.000] | 0.312 [0.207, 0.410] |
| bal | 0.413 [0.372, 0.454] | 0.479 [0.417, 0.537] | 0.348 [0.310, 0.385] | 0.665 [0.592, 0.729] | 0.306 [0.233, 0.379] | 0.639 [0.596, 0.683] | 0.753 [0.727, 0.778] | 0.275 [0.167, 0.386] |
| ugtmix | 0.207 [0.172, 0.242] | 0.290 [0.231, 0.343] | 0.123 [0.101, 0.147] | 0.311 [0.243, 0.378] | 0.439 [0.351, 0.515] | 0.389 [0.348, 0.429] | 0.516 [0.475, 0.558] | 0.256 [0.163, 0.359] |
| loginv | 0.353 [0.302, 0.403] | 0.417 [0.330, 0.494] | 0.288 [0.251, 0.328] | 0.281 [0.183, 0.380] | 0.351 [0.267, 0.428] | 0.541 [0.493, 0.593] | 0.876 [0.847, 0.906] | 0.364 [0.253, 0.474] |

Differenze braccio - ctrl, 10548 passi: delta [IC 95%] (P(boot <= 0))

| braccio | punteggio dev | dev graduata | dev rank-1 espr. | HIFI3D graduata, GT maxabs | HIFI3D graduata, GT unif. | FaceVerse espr. rank-1 | HIFI3D rank-1 | NoW tau |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| area | +0.022 [-0.011, +0.058] (0.09) | -0.065 [-0.114, -0.011] (0.99) | +0.109 [+0.075, +0.149] (0.00) | -0.049 [-0.090, -0.010] (1.00) | -0.038 [-0.082, +0.002] (0.97) | +0.185 [+0.145, +0.225] (0.00) | +0.082 [+0.057, +0.105] (0.00) | +0.032 [-0.074, +0.140] (0.27) |
| arearobust | +0.174 [+0.135, +0.215] (0.00) | +0.061 [-0.008, +0.131] (0.05) | +0.287 [+0.247, +0.329] (0.00) | -0.081 [-0.134, -0.029] (1.00) | +0.058 [+0.001, +0.116] (0.02) | +0.336 [+0.298, +0.369] (0.00) | +0.130 [+0.107, +0.154] (0.00) | +0.030 [-0.070, +0.131] (0.30) |
| bal | +0.081 [+0.063, +0.100] (0.00) | +0.092 [+0.059, +0.122] (0.00) | +0.070 [+0.051, +0.092] (0.00) | -0.028 [-0.052, -0.005] (0.99) | -0.046 [-0.068, -0.022] (1.00) | +0.132 [+0.109, +0.157] (0.00) | -0.117 [-0.140, -0.094] (1.00) | -0.008 [-0.081, +0.060] (0.59) |
| ugtmix | -0.126 [-0.163, -0.087] (1.00) | -0.097 [-0.155, -0.040] (1.00) | -0.154 [-0.187, -0.121] (1.00) | -0.382 [-0.473, -0.285] (1.00) | +0.087 [+0.009, +0.154] (0.01) | -0.118 [-0.155, -0.083] (1.00) | -0.354 [-0.393, -0.317] (1.00) | -0.027 [-0.166, +0.099] (0.64) |
| loginv | +0.021 [-0.026, +0.067] (0.19) | +0.030 [-0.045, +0.100] (0.20) | +0.011 [-0.022, +0.046] (0.28) | -0.413 [-0.512, -0.304] (1.00) | -0.002 [-0.081, +0.078] (0.51) | +0.034 [-0.008, +0.074] (0.06) | +0.006 [-0.021, +0.035] (0.33) | +0.081 [-0.025, +0.187] (0.07) |

## Controlli

| controllo | valore | atteso |
| --- | --- | --- |
| hifi 10548: celle presenti (stesse righe, stessi soggetti) | ['ctrl', 'area', 'arearobust', 'bal', 'ugtmix', 'loginv']; 100 soggetti, 148500 righe | - |
| hifi 21096: celle presenti (stesse righe, stessi soggetti) | ['ctrl', 'area', 'arearobust', 'bal', 'ugtmix', 'loginv']; 100 soggetti, 148500 righe | - |
| dev FaceScape ctrl 10548: embedding contro latent_distance del breakdown, max |diff| | 5.70e-07 | < 1e-4 |
| dev FaceScape ctrl 21096: embedding contro latent_distance del breakdown, max |diff| | 5.91e-07 | < 1e-4 |
| dev FaceScape area 10548: embedding contro latent_distance del breakdown, max |diff| | 5.66e-07 | < 1e-4 |
| dev FaceScape area 21096: embedding contro latent_distance del breakdown, max |diff| | 5.86e-07 | < 1e-4 |
| dev FaceScape arearobust 10548: embedding contro latent_distance del breakdown, max |diff| | 7.77e-04 | < 1e-4 |
| dev FaceScape arearobust 21096: embedding contro latent_distance del breakdown, max |diff| | 1.06e-03 | < 1e-4 |
| dev FaceScape bal 10548: embedding contro latent_distance del breakdown, max |diff| | 5.56e-07 | < 1e-4 |
| dev FaceScape bal 21096: embedding contro latent_distance del breakdown, max |diff| | 5.72e-07 | < 1e-4 |
| dev FaceScape ugtmix 10548: embedding contro latent_distance del breakdown, max |diff| | 5.47e-07 | < 1e-4 |
| dev FaceScape ugtmix 21096: embedding contro latent_distance del breakdown, max |diff| | 5.61e-07 | < 1e-4 |
| dev FaceScape loginv 10548: embedding contro latent_distance del breakdown, max |diff| | 6.00e-07 | < 1e-4 |
| dev FaceScape loginv 21096: embedding contro latent_distance del breakdown, max |diff| | 6.05e-07 | < 1e-4 |
| NoW ctrl (v2) 10548: tau ricalcolato contro concordance.csv della cella | 0.282197 / 0.282197 | uguali |
| NoW ctrl (v2) 21096: tau ricalcolato contro concordance.csv della cella | 0.263258 / 0.263258 | uguali |
| NoW area 10548: tau ricalcolato contro concordance.csv della cella | 0.314394 / 0.314394 | uguali |
| NoW area 21096: tau ricalcolato contro concordance.csv della cella | 0.299242 / 0.299242 | uguali |
| NoW arearobust 10548: tau ricalcolato contro concordance.csv della cella | 0.312500 / 0.312500 | uguali |
| NoW arearobust 21096: tau ricalcolato contro concordance.csv della cella | 0.265152 / 0.265152 | uguali |
| NoW bal 10548: tau ricalcolato contro concordance.csv della cella | 0.274621 / 0.274621 | uguali |
| NoW bal 21096: tau ricalcolato contro concordance.csv della cella | 0.244318 / 0.244318 | uguali |
| NoW ugtmix 10548: tau ricalcolato contro concordance.csv della cella | 0.255682 / 0.255682 | uguali |
| NoW ugtmix 21096: tau ricalcolato contro concordance.csv della cella | 0.221591 / 0.221591 | uguali |
| NoW loginv 10548: tau ricalcolato contro concordance.csv della cella | 0.363636 / 0.363636 | uguali |
| NoW loginv 21096: tau ricalcolato contro concordance.csv della cella | 0.409091 / 0.409091 | uguali |
