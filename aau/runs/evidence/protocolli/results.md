# Protocolli della comunita' sul nostro benchmark: Spearman con GT FR / SR e confronti appaiati

Generato da `aau/competitors/protocolli/pr_eval.py` (non modificare a mano; definizioni nel docstring e nel README di questa cartella). Righe della matrice unificata, 1000 repliche per soggetto col seme della T1, IC 95% percentile, maschera comune fra le colonne di questo confronto. Distanza di un protocollo = media dei due versi (A predetta, B riferimento e viceversa).

## GT FR, nocrop

| metodo | hifi3d | facescape | faceverse | faceverse_neutral | famos |
|---|---|---|---|---|---|
| NoW originale (Procrustes + ICP con scala) | 0.416 [0.337, 0.495] | 0.624 [0.537, 0.701] | 0.300 [0.231, 0.366] | 0.386 [0.309, 0.460] | 0.609 [0.265, 0.811] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.697 [0.628, 0.753] | 0.677 [0.611, 0.733] | 0.338 [0.264, 0.408] | 0.407 [0.326, 0.481] | 0.785 [0.488, 0.925] |
| REALY (gICP con scala + rICP per regione + nICP) | 0.495 [0.402, 0.574] | 0.618 [0.533, 0.694] | 0.302 [0.235, 0.364] | 0.479 [0.393, 0.557] | 0.517 [0.161, 0.707] |
| Stirling/ESRC (Procrustes con scala, RMSE) | 0.464 [0.375, 0.548] | 0.574 [0.480, 0.666] | 0.201 [0.136, 0.269] | 0.307 [0.213, 0.402] | 0.502 [0.134, 0.757] |
| ICP + Chamfer in mm (per coppia) | 0.643 [0.572, 0.703] | 0.467 [0.401, 0.528] | 0.337 [0.262, 0.410] | 0.409 [0.326, 0.481] | 0.739 [0.362, 0.914] |
| factorized s1234, d_F non cal. | 0.643 [0.571, 0.706] | 0.687 [0.611, 0.757] | 0.297 [0.229, 0.362] | 0.337 [0.256, 0.409] | 0.616 [0.222, 0.824] |
| factorized s2345, d_F non cal. | 0.631 [0.561, 0.697] | 0.698 [0.615, 0.771] | 0.317 [0.243, 0.383] | 0.373 [0.293, 0.451] | 0.612 [0.191, 0.831] |

Righe: hifi3d 99000, facescape 99000, faceverse 99000, faceverse_neutral 99000, famos 105

## GT FR, crop

| metodo | hifi3d | facescape | faceverse | faceverse_neutral |
|---|---|---|---|---|
| NoW originale (Procrustes + ICP con scala) | 0.411 [0.333, 0.489] | 0.597 [0.511, 0.675] | 0.301 [0.233, 0.366] | 0.400 [0.324, 0.474] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.703 [0.636, 0.759] | 0.662 [0.592, 0.719] | 0.341 [0.266, 0.409] | 0.423 [0.343, 0.497] |
| REALY (gICP con scala + rICP per regione + nICP) | 0.499 [0.407, 0.580] | 0.617 [0.531, 0.692] | 0.318 [0.252, 0.376] | 0.481 [0.394, 0.559] |
| Stirling/ESRC (Procrustes con scala, RMSE) | 0.474 [0.384, 0.560] | 0.572 [0.478, 0.667] | 0.193 [0.124, 0.259] | 0.307 [0.213, 0.405] |
| ICP + Chamfer in mm (per coppia) | 0.416 [0.360, 0.473] | 0.281 [0.241, 0.321] | 0.271 [0.207, 0.332] | 0.317 [0.248, 0.379] |
| factorized s1234, d_F non cal. | 0.563 [0.485, 0.630] | 0.488 [0.403, 0.576] | 0.265 [0.185, 0.339] | 0.306 [0.221, 0.390] |
| factorized s2345, d_F non cal. | 0.516 [0.435, 0.584] | 0.468 [0.378, 0.554] | 0.256 [0.187, 0.323] | 0.310 [0.231, 0.388] |

Righe: hifi3d 49500, facescape 49500, faceverse 49500, faceverse_neutral 49500

## GT FR, all_cross

| metodo | hifi3d | facescape | faceverse | faceverse_neutral |
|---|---|---|---|---|
| NoW originale (Procrustes + ICP con scala) | 0.414 [0.335, 0.493] | 0.615 [0.529, 0.692] | 0.299 [0.235, 0.364] | 0.389 [0.313, 0.463] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.699 [0.631, 0.755] | 0.672 [0.605, 0.728] | 0.337 [0.266, 0.406] | 0.411 [0.331, 0.485] |
| REALY (gICP con scala + rICP per regione + nICP) | 0.496 [0.403, 0.575] | 0.617 [0.532, 0.693] | 0.307 [0.242, 0.365] | 0.479 [0.393, 0.557] |
| Stirling/ESRC (Procrustes con scala, RMSE) | 0.467 [0.377, 0.552] | 0.573 [0.479, 0.665] | 0.198 [0.137, 0.260] | 0.307 [0.213, 0.403] |
| ICP + Chamfer in mm (per coppia) | 0.556 [0.491, 0.614] | 0.381 [0.323, 0.435] | 0.312 [0.242, 0.379] | 0.373 [0.297, 0.442] |
| factorized s1234, d_F non cal. | 0.584 [0.513, 0.647] | 0.443 [0.377, 0.508] | 0.286 [0.217, 0.352] | 0.326 [0.249, 0.400] |
| factorized s2345, d_F non cal. | 0.562 [0.487, 0.627] | 0.420 [0.359, 0.480] | 0.296 [0.226, 0.362] | 0.352 [0.273, 0.427] |

Righe: hifi3d 148500, facescape 148500, faceverse 148500, faceverse_neutral 148500

## GT SR, nocrop

| metodo | hifi3d | facescape | faceverse | faceverse_neutral | famos |
|---|---|---|---|---|---|
| NoW originale (Procrustes + ICP con scala) | 0.620 [0.556, 0.677] | 0.726 [0.673, 0.776] | 0.307 [0.236, 0.377] | 0.401 [0.326, 0.474] | 0.703 [0.429, 0.817] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.430 [0.349, 0.511] | 0.703 [0.645, 0.754] | 0.310 [0.234, 0.385] | 0.383 [0.297, 0.463] | 0.581 [0.239, 0.820] |
| REALY (gICP con scala + rICP per regione + nICP) | 0.715 [0.646, 0.771] | 0.718 [0.652, 0.775] | 0.312 [0.250, 0.368] | 0.521 [0.447, 0.590] | 0.776 [0.587, 0.887] |
| Stirling/ESRC (Procrustes con scala, RMSE) | 0.631 [0.553, 0.696] | 0.673 [0.604, 0.736] | 0.198 [0.134, 0.263] | 0.330 [0.246, 0.417] | 0.756 [0.541, 0.876] |
| ICP + Chamfer in mm (per coppia) | 0.366 [0.281, 0.454] | 0.491 [0.427, 0.547] | 0.309 [0.234, 0.380] | 0.383 [0.298, 0.456] | 0.548 [0.185, 0.799] |
| factorized s1234, d_P | 0.622 [0.550, 0.685] | 0.747 [0.679, 0.804] | 0.286 [0.215, 0.352] | 0.333 [0.251, 0.405] | 0.740 [0.548, 0.883] |
| factorized s2345, d_P | 0.613 [0.533, 0.688] | 0.754 [0.687, 0.812] | 0.313 [0.237, 0.377] | 0.372 [0.291, 0.444] | 0.728 [0.516, 0.869] |

Righe: hifi3d 99000, facescape 99000, faceverse 99000, faceverse_neutral 99000, famos 105

## GT SR, crop

| metodo | hifi3d | facescape | faceverse | faceverse_neutral |
|---|---|---|---|---|
| NoW originale (Procrustes + ICP con scala) | 0.607 [0.541, 0.662] | 0.692 [0.630, 0.747] | 0.312 [0.241, 0.378] | 0.414 [0.338, 0.486] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.437 [0.355, 0.516] | 0.681 [0.617, 0.738] | 0.313 [0.235, 0.384] | 0.397 [0.314, 0.474] |
| REALY (gICP con scala + rICP per regione + nICP) | 0.721 [0.653, 0.776] | 0.717 [0.650, 0.773] | 0.332 [0.270, 0.390] | 0.523 [0.448, 0.591] |
| Stirling/ESRC (Procrustes con scala, RMSE) | 0.645 [0.567, 0.711] | 0.671 [0.602, 0.736] | 0.204 [0.136, 0.268] | 0.329 [0.244, 0.418] |
| ICP + Chamfer in mm (per coppia) | 0.266 [0.209, 0.328] | 0.294 [0.255, 0.332] | 0.251 [0.186, 0.311] | 0.299 [0.233, 0.363] |
| factorized s1234, d_P | 0.550 [0.472, 0.623] | 0.518 [0.434, 0.598] | 0.251 [0.174, 0.324] | 0.304 [0.224, 0.382] |
| factorized s2345, d_P | 0.514 [0.435, 0.590] | 0.476 [0.390, 0.559] | 0.257 [0.181, 0.323] | 0.316 [0.236, 0.389] |

Righe: hifi3d 49500, facescape 49500, faceverse 49500, faceverse_neutral 49500

## GT SR, all_cross

| metodo | hifi3d | facescape | faceverse | faceverse_neutral |
|---|---|---|---|---|
| NoW originale (Procrustes + ICP con scala) | 0.614 [0.550, 0.671] | 0.715 [0.659, 0.766] | 0.308 [0.238, 0.374] | 0.404 [0.329, 0.476] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.432 [0.351, 0.512] | 0.695 [0.637, 0.748] | 0.310 [0.235, 0.381] | 0.386 [0.302, 0.465] |
| REALY (gICP con scala + rICP per regione + nICP) | 0.716 [0.648, 0.771] | 0.717 [0.651, 0.773] | 0.319 [0.260, 0.370] | 0.521 [0.447, 0.589] |
| Stirling/ESRC (Procrustes con scala, RMSE) | 0.635 [0.557, 0.700] | 0.671 [0.603, 0.735] | 0.200 [0.139, 0.260] | 0.329 [0.245, 0.417] |
| ICP + Chamfer in mm (per coppia) | 0.327 [0.252, 0.402] | 0.401 [0.343, 0.453] | 0.287 [0.217, 0.352] | 0.351 [0.274, 0.420] |
| factorized s1234, d_P | 0.557 [0.485, 0.625] | 0.489 [0.423, 0.550] | 0.273 [0.201, 0.339] | 0.323 [0.246, 0.393] |
| factorized s2345, d_P | 0.540 [0.463, 0.613] | 0.464 [0.402, 0.522] | 0.293 [0.219, 0.357] | 0.352 [0.273, 0.424] |

Righe: hifi3d 148500, facescape 148500, faceverse 148500, faceverse_neutral 148500

## Delta appaiati, GT FR (a - b; negativo = b ordina meglio)

| dominio | confronto | nocrop | crop | all_cross |
|---|---|---|---|---|
| hifi3d | NoW originale (Procrustes + ICP con scala) - NoW metrical / MICA (Procrustes senza scala + ICP rigido) | -0.281 [-0.371, -0.194] | -0.292 [-0.379, -0.204] | -0.285 [-0.373, -0.197] |
| hifi3d | NoW originale (Procrustes + ICP con scala) - ICP + Chamfer in mm (per coppia) | -0.227 [-0.321, -0.137] | -0.005 [-0.092, +0.078] | -0.142 [-0.231, -0.053] |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - ICP + Chamfer in mm (per coppia) | +0.054 [+0.030, +0.078] | +0.287 [+0.255, +0.322] | +0.143 [+0.118, +0.170] |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP) - ICP + Chamfer in mm (per coppia) | -0.148 [-0.260, -0.038] | +0.083 [-0.022, +0.181] | -0.060 [-0.172, +0.046] |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE) - ICP + Chamfer in mm (per coppia) | -0.179 [-0.283, -0.074] | +0.058 [-0.040, +0.153] | -0.089 [-0.191, +0.014] |
| hifi3d | NoW originale (Procrustes + ICP con scala) - factorized s1234, d_F non cal. | -0.226 [-0.299, -0.151] | -0.152 [-0.224, -0.074] | -0.170 [-0.240, -0.094] |
| hifi3d | NoW originale (Procrustes + ICP con scala) - factorized s2345, d_F non cal. | -0.215 [-0.288, -0.141] | -0.105 [-0.184, -0.031] | -0.148 [-0.219, -0.072] |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s1234, d_F non cal. | +0.055 [+0.009, +0.105] | +0.140 [+0.084, +0.201] | +0.115 [+0.068, +0.166] |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s2345, d_F non cal. | +0.066 [+0.011, +0.121] | +0.187 [+0.127, +0.247] | +0.137 [+0.083, +0.191] |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP) - factorized s1234, d_F non cal. | -0.147 [-0.247, -0.043] | -0.064 [-0.164, +0.044] | -0.088 [-0.187, +0.020] |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP) - factorized s2345, d_F non cal. | -0.136 [-0.237, -0.030] | -0.017 [-0.126, +0.095] | -0.066 [-0.167, +0.042] |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s1234, d_F non cal. | -0.179 [-0.266, -0.088] | -0.089 [-0.178, +0.004] | -0.117 [-0.205, -0.027] |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s2345, d_F non cal. | -0.168 [-0.255, -0.077] | -0.042 [-0.133, +0.054] | -0.095 [-0.182, -0.005] |
| facescape | NoW originale (Procrustes + ICP con scala) - NoW metrical / MICA (Procrustes senza scala + ICP rigido) | -0.053 [-0.092, -0.018] | -0.065 [-0.104, -0.031] | -0.057 [-0.096, -0.023] |
| facescape | NoW originale (Procrustes + ICP con scala) - ICP + Chamfer in mm (per coppia) | +0.157 [+0.119, +0.199] | +0.316 [+0.263, +0.364] | +0.234 [+0.193, +0.276] |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - ICP + Chamfer in mm (per coppia) | +0.210 [+0.186, +0.236] | +0.381 [+0.345, +0.410] | +0.291 [+0.263, +0.317] |
| facescape | REALY (gICP con scala + rICP per regione + nICP) - ICP + Chamfer in mm (per coppia) | +0.151 [+0.096, +0.209] | +0.336 [+0.274, +0.395] | +0.235 [+0.180, +0.292] |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE) - ICP + Chamfer in mm (per coppia) | +0.107 [+0.053, +0.165] | +0.291 [+0.222, +0.356] | +0.191 [+0.133, +0.251] |
| facescape | NoW originale (Procrustes + ICP con scala) - factorized s1234, d_F non cal. | -0.063 [-0.121, +0.002] | +0.109 [+0.030, +0.183] | +0.172 [+0.116, +0.230] |
| facescape | NoW originale (Procrustes + ICP con scala) - factorized s2345, d_F non cal. | -0.073 [-0.140, +0.002] | +0.129 [+0.044, +0.215] | +0.195 [+0.133, +0.258] |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s1234, d_F non cal. | -0.010 [-0.072, +0.052] | +0.174 [+0.093, +0.250] | +0.229 [+0.167, +0.287] |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s2345, d_F non cal. | -0.020 [-0.086, +0.049] | +0.194 [+0.111, +0.274] | +0.251 [+0.192, +0.308] |
| facescape | REALY (gICP con scala + rICP per regione + nICP) - factorized s1234, d_F non cal. | -0.069 [-0.138, +0.002] | +0.129 [+0.052, +0.205] | +0.173 [+0.108, +0.239] |
| facescape | REALY (gICP con scala + rICP per regione + nICP) - factorized s2345, d_F non cal. | -0.080 [-0.151, -0.007] | +0.149 [+0.066, +0.229] | +0.196 [+0.129, +0.260] |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s1234, d_F non cal. | -0.113 [-0.187, -0.040] | +0.084 [+0.002, +0.162] | +0.129 [+0.062, +0.198] |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s2345, d_F non cal. | -0.123 [-0.202, -0.041] | +0.104 [+0.007, +0.191] | +0.152 [+0.079, +0.224] |
| faceverse | NoW originale (Procrustes + ICP con scala) - NoW metrical / MICA (Procrustes senza scala + ICP rigido) | -0.038 [-0.078, +0.000] | -0.039 [-0.082, +0.000] | -0.038 [-0.079, -0.000] |
| faceverse | NoW originale (Procrustes + ICP con scala) - ICP + Chamfer in mm (per coppia) | -0.037 [-0.093, +0.019] | +0.030 [-0.021, +0.077] | -0.013 [-0.066, +0.039] |
| faceverse | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - ICP + Chamfer in mm (per coppia) | +0.000 [-0.038, +0.041] | +0.069 [+0.034, +0.103] | +0.025 [-0.011, +0.061] |
| faceverse | REALY (gICP con scala + rICP per regione + nICP) - ICP + Chamfer in mm (per coppia) | -0.036 [-0.120, +0.050] | +0.046 [-0.028, +0.122] | -0.005 [-0.083, +0.075] |
| faceverse | Stirling/ESRC (Procrustes con scala, RMSE) - ICP + Chamfer in mm (per coppia) | -0.137 [-0.214, -0.046] | -0.079 [-0.152, -0.004] | -0.114 [-0.188, -0.030] |
| faceverse | NoW originale (Procrustes + ICP con scala) - factorized s1234, d_F non cal. | +0.003 [-0.061, +0.068] | +0.037 [-0.035, +0.109] | +0.014 [-0.050, +0.080] |
| faceverse | NoW originale (Procrustes + ICP con scala) - factorized s2345, d_F non cal. | -0.017 [-0.084, +0.050] | +0.046 [-0.023, +0.116] | +0.004 [-0.061, +0.069] |
| faceverse | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s1234, d_F non cal. | +0.041 [-0.030, +0.111] | +0.076 [+0.002, +0.145] | +0.052 [-0.017, +0.119] |
| faceverse | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s2345, d_F non cal. | +0.021 [-0.050, +0.091] | +0.085 [+0.013, +0.156] | +0.042 [-0.026, +0.109] |
| faceverse | REALY (gICP con scala + rICP per regione + nICP) - factorized s1234, d_F non cal. | +0.005 [-0.069, +0.080] | +0.053 [-0.032, +0.142] | +0.021 [-0.052, +0.097] |
| faceverse | REALY (gICP con scala + rICP per regione + nICP) - factorized s2345, d_F non cal. | -0.015 [-0.101, +0.070] | +0.062 [-0.023, +0.145] | +0.011 [-0.070, +0.096] |
| faceverse | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s1234, d_F non cal. | -0.096 [-0.178, -0.003] | -0.072 [-0.163, +0.016] | -0.088 [-0.170, -0.001] |
| faceverse | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s2345, d_F non cal. | -0.116 [-0.207, -0.021] | -0.063 [-0.147, +0.027] | -0.098 [-0.180, -0.006] |
| faceverse_neutral | NoW originale (Procrustes + ICP con scala) - NoW metrical / MICA (Procrustes senza scala + ICP rigido) | -0.021 [-0.060, +0.019] | -0.024 [-0.063, +0.015] | -0.022 [-0.061, +0.017] |
| faceverse_neutral | NoW originale (Procrustes + ICP con scala) - ICP + Chamfer in mm (per coppia) | -0.022 [-0.080, +0.038] | +0.083 [+0.034, +0.130] | +0.016 [-0.036, +0.070] |
| faceverse_neutral | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - ICP + Chamfer in mm (per coppia) | -0.001 [-0.047, +0.044] | +0.106 [+0.066, +0.147] | +0.038 [-0.005, +0.080] |
| faceverse_neutral | REALY (gICP con scala + rICP per regione + nICP) - ICP + Chamfer in mm (per coppia) | +0.070 [-0.033, +0.178] | +0.164 [+0.072, +0.261] | +0.106 [+0.008, +0.206] |
| faceverse_neutral | Stirling/ESRC (Procrustes con scala, RMSE) - ICP + Chamfer in mm (per coppia) | -0.102 [-0.206, +0.014] | -0.010 [-0.107, +0.095] | -0.067 [-0.167, +0.042] |
| faceverse_neutral | NoW originale (Procrustes + ICP con scala) - factorized s1234, d_F non cal. | +0.050 [-0.025, +0.123] | +0.094 [+0.021, +0.166] | +0.063 [-0.008, +0.135] |
| faceverse_neutral | NoW originale (Procrustes + ICP con scala) - factorized s2345, d_F non cal. | +0.013 [-0.065, +0.085] | +0.089 [+0.014, +0.166] | +0.038 [-0.036, +0.109] |
| faceverse_neutral | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s1234, d_F non cal. | +0.070 [-0.018, +0.152] | +0.117 [+0.037, +0.197] | +0.085 [+0.000, +0.163] |
| faceverse_neutral | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s2345, d_F non cal. | +0.034 [-0.048, +0.116] | +0.113 [+0.037, +0.193] | +0.060 [-0.020, +0.139] |
| faceverse_neutral | REALY (gICP con scala + rICP per regione + nICP) - factorized s1234, d_F non cal. | +0.142 [+0.037, +0.236] | +0.175 [+0.074, +0.273] | +0.153 [+0.055, +0.245] |
| faceverse_neutral | REALY (gICP con scala + rICP per regione + nICP) - factorized s2345, d_F non cal. | +0.106 [+0.005, +0.209] | +0.170 [+0.066, +0.275] | +0.128 [+0.027, +0.229] |
| faceverse_neutral | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s1234, d_F non cal. | -0.030 [-0.141, +0.087] | +0.001 [-0.106, +0.109] | -0.019 [-0.125, +0.094] |
| faceverse_neutral | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s2345, d_F non cal. | -0.066 [-0.188, +0.056] | -0.004 [-0.114, +0.113] | -0.045 [-0.162, +0.077] |
| famos | NoW originale (Procrustes + ICP con scala) - NoW metrical / MICA (Procrustes senza scala + ICP rigido) | -0.176 [-0.433, +0.055] | - | - |
| famos | NoW originale (Procrustes + ICP con scala) - ICP + Chamfer in mm (per coppia) | -0.131 [-0.392, +0.129] | - | - |
| famos | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - ICP + Chamfer in mm (per coppia) | +0.045 [-0.013, +0.169] | - | - |
| famos | REALY (gICP con scala + rICP per regione + nICP) - ICP + Chamfer in mm (per coppia) | -0.222 [-0.487, +0.052] | - | - |
| famos | Stirling/ESRC (Procrustes con scala, RMSE) - ICP + Chamfer in mm (per coppia) | -0.238 [-0.511, +0.057] | - | - |
| famos | NoW originale (Procrustes + ICP con scala) - factorized s1234, d_F non cal. | -0.007 [-0.239, +0.192] | - | - |
| famos | NoW originale (Procrustes + ICP con scala) - factorized s2345, d_F non cal. | -0.003 [-0.216, +0.222] | - | - |
| famos | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s1234, d_F non cal. | +0.168 [-0.030, +0.389] | - | - |
| famos | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s2345, d_F non cal. | +0.172 [+0.009, +0.397] | - | - |
| famos | REALY (gICP con scala + rICP per regione + nICP) - factorized s1234, d_F non cal. | -0.099 [-0.316, +0.128] | - | - |
| famos | REALY (gICP con scala + rICP per regione + nICP) - factorized s2345, d_F non cal. | -0.095 [-0.304, +0.169] | - | - |
| famos | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s1234, d_F non cal. | -0.115 [-0.334, +0.084] | - | - |
| famos | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s2345, d_F non cal. | -0.111 [-0.335, +0.136] | - | - |

## Delta appaiati, GT SR (a - b; negativo = b ordina meglio)

| dominio | confronto | nocrop | crop | all_cross |
|---|---|---|---|---|
| hifi3d | NoW originale (Procrustes + ICP con scala) - NoW metrical / MICA (Procrustes senza scala + ICP rigido) | +0.189 [+0.132, +0.251] | +0.170 [+0.115, +0.232] | +0.182 [+0.126, +0.244] |
| hifi3d | NoW originale (Procrustes + ICP con scala) - ICP + Chamfer in mm (per coppia) | +0.254 [+0.185, +0.325] | +0.341 [+0.287, +0.394] | +0.288 [+0.230, +0.352] |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - ICP + Chamfer in mm (per coppia) | +0.065 [+0.038, +0.090] | +0.171 [+0.131, +0.211] | +0.106 [+0.076, +0.135] |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP) - ICP + Chamfer in mm (per coppia) | +0.349 [+0.260, +0.443] | +0.455 [+0.387, +0.529] | +0.389 [+0.309, +0.476] |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE) - ICP + Chamfer in mm (per coppia) | +0.265 [+0.184, +0.346] | +0.379 [+0.312, +0.446] | +0.309 [+0.235, +0.381] |
| hifi3d | NoW originale (Procrustes + ICP con scala) - factorized s1234, d_P | -0.002 [-0.047, +0.046] | +0.057 [+0.005, +0.114] | +0.058 [+0.010, +0.105] |
| hifi3d | NoW originale (Procrustes + ICP con scala) - factorized s2345, d_P | +0.007 [-0.047, +0.058] | +0.093 [+0.036, +0.150] | +0.074 [+0.023, +0.126] |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s1234, d_P | -0.191 [-0.261, -0.127] | -0.113 [-0.186, -0.040] | -0.125 [-0.195, -0.057] |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s2345, d_P | -0.182 [-0.259, -0.111] | -0.077 [-0.157, +0.003] | -0.108 [-0.184, -0.035] |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP) - factorized s1234, d_P | +0.093 [+0.013, +0.182] | +0.171 [+0.089, +0.262] | +0.159 [+0.083, +0.245] |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP) - factorized s2345, d_P | +0.102 [+0.014, +0.193] | +0.207 [+0.111, +0.302] | +0.176 [+0.089, +0.266] |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s1234, d_P | +0.010 [-0.066, +0.079] | +0.096 [+0.019, +0.165] | +0.079 [+0.006, +0.148] |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s2345, d_P | +0.018 [-0.057, +0.090] | +0.131 [+0.047, +0.208] | +0.095 [+0.017, +0.166] |
| facescape | NoW originale (Procrustes + ICP con scala) - NoW metrical / MICA (Procrustes senza scala + ICP rigido) | +0.023 [+0.008, +0.039] | +0.011 [-0.002, +0.027] | +0.019 [+0.004, +0.035] |
| facescape | NoW originale (Procrustes + ICP con scala) - ICP + Chamfer in mm (per coppia) | +0.235 [+0.212, +0.260] | +0.398 [+0.369, +0.426] | +0.314 [+0.290, +0.337] |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - ICP + Chamfer in mm (per coppia) | +0.212 [+0.192, +0.234] | +0.387 [+0.355, +0.412] | +0.294 [+0.273, +0.315] |
| facescape | REALY (gICP con scala + rICP per regione + nICP) - ICP + Chamfer in mm (per coppia) | +0.227 [+0.178, +0.280] | +0.423 [+0.374, +0.467] | +0.316 [+0.267, +0.366] |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE) - ICP + Chamfer in mm (per coppia) | +0.182 [+0.142, +0.222] | +0.378 [+0.333, +0.418] | +0.270 [+0.230, +0.311] |
| facescape | NoW originale (Procrustes + ICP con scala) - factorized s1234, d_P | -0.021 [-0.071, +0.032] | +0.174 [+0.105, +0.241] | +0.226 [+0.175, +0.276] |
| facescape | NoW originale (Procrustes + ICP con scala) - factorized s2345, d_P | -0.028 [-0.080, +0.026] | +0.216 [+0.137, +0.290] | +0.251 [+0.199, +0.301] |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s1234, d_P | -0.044 [-0.099, +0.015] | +0.162 [+0.091, +0.235] | +0.206 [+0.153, +0.260] |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s2345, d_P | -0.051 [-0.105, +0.008] | +0.204 [+0.123, +0.283] | +0.232 [+0.177, +0.287] |
| facescape | REALY (gICP con scala + rICP per regione + nICP) - factorized s1234, d_P | -0.029 [-0.092, +0.035] | +0.198 [+0.130, +0.272] | +0.228 [+0.171, +0.286] |
| facescape | REALY (gICP con scala + rICP per regione + nICP) - factorized s2345, d_P | -0.036 [-0.101, +0.029] | +0.240 [+0.164, +0.316] | +0.253 [+0.194, +0.310] |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s1234, d_P | -0.074 [-0.136, -0.010] | +0.153 [+0.085, +0.220] | +0.182 [+0.127, +0.240] |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s2345, d_P | -0.081 [-0.149, -0.017] | +0.195 [+0.114, +0.275] | +0.208 [+0.147, +0.268] |
| faceverse | NoW originale (Procrustes + ICP con scala) - NoW metrical / MICA (Procrustes senza scala + ICP rigido) | -0.003 [-0.041, +0.037] | -0.001 [-0.040, +0.039] | -0.002 [-0.041, +0.036] |
| faceverse | NoW originale (Procrustes + ICP con scala) - ICP + Chamfer in mm (per coppia) | -0.002 [-0.057, +0.049] | +0.061 [+0.016, +0.105] | +0.021 [-0.030, +0.066] |
| faceverse | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - ICP + Chamfer in mm (per coppia) | +0.001 [-0.041, +0.044] | +0.063 [+0.023, +0.099] | +0.023 [-0.016, +0.062] |
| faceverse | REALY (gICP con scala + rICP per regione + nICP) - ICP + Chamfer in mm (per coppia) | +0.003 [-0.075, +0.085] | +0.081 [+0.014, +0.157] | +0.032 [-0.039, +0.107] |
| faceverse | Stirling/ESRC (Procrustes con scala, RMSE) - ICP + Chamfer in mm (per coppia) | -0.112 [-0.188, -0.030] | -0.047 [-0.117, +0.024] | -0.087 [-0.158, -0.014] |
| faceverse | NoW originale (Procrustes + ICP con scala) - factorized s1234, d_P | +0.021 [-0.045, +0.084] | +0.061 [-0.010, +0.131] | +0.034 [-0.030, +0.100] |
| faceverse | NoW originale (Procrustes + ICP con scala) - factorized s2345, d_P | -0.006 [-0.072, +0.062] | +0.055 [-0.015, +0.126] | +0.015 [-0.048, +0.080] |
| faceverse | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s1234, d_P | +0.024 [-0.050, +0.097] | +0.062 [-0.016, +0.134] | +0.037 [-0.034, +0.107] |
| faceverse | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s2345, d_P | -0.003 [-0.073, +0.070] | +0.057 [-0.014, +0.129] | +0.017 [-0.051, +0.088] |
| faceverse | REALY (gICP con scala + rICP per regione + nICP) - factorized s1234, d_P | +0.026 [-0.041, +0.094] | +0.080 [-0.000, +0.163] | +0.045 [-0.022, +0.112] |
| faceverse | REALY (gICP con scala + rICP per regione + nICP) - factorized s2345, d_P | -0.001 [-0.082, +0.084] | +0.075 [-0.003, +0.152] | +0.026 [-0.051, +0.107] |
| faceverse | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s1234, d_P | -0.088 [-0.169, -0.000] | -0.048 [-0.136, +0.040] | -0.074 [-0.154, +0.009] |
| faceverse | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s2345, d_P | -0.115 [-0.203, -0.022] | -0.053 [-0.138, +0.035] | -0.093 [-0.178, -0.005] |
| faceverse_neutral | NoW originale (Procrustes + ICP con scala) - NoW metrical / MICA (Procrustes senza scala + ICP rigido) | +0.018 [-0.017, +0.056] | +0.018 [-0.018, +0.055] | +0.018 [-0.018, +0.055] |
| faceverse_neutral | NoW originale (Procrustes + ICP con scala) - ICP + Chamfer in mm (per coppia) | +0.019 [-0.036, +0.071] | +0.115 [+0.070, +0.157] | +0.054 [+0.003, +0.099] |
| faceverse_neutral | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - ICP + Chamfer in mm (per coppia) | +0.000 [-0.050, +0.049] | +0.098 [+0.054, +0.143] | +0.036 [-0.010, +0.081] |
| faceverse_neutral | REALY (gICP con scala + rICP per regione + nICP) - ICP + Chamfer in mm (per coppia) | +0.138 [+0.042, +0.236] | +0.224 [+0.136, +0.308] | +0.171 [+0.078, +0.261] |
| faceverse_neutral | Stirling/ESRC (Procrustes con scala, RMSE) - ICP + Chamfer in mm (per coppia) | -0.053 [-0.151, +0.052] | +0.030 [-0.059, +0.124] | -0.022 [-0.115, +0.081] |
| faceverse_neutral | NoW originale (Procrustes + ICP con scala) - factorized s1234, d_P | +0.068 [-0.006, +0.140] | +0.110 [+0.038, +0.179] | +0.081 [+0.010, +0.150] |
| faceverse_neutral | NoW originale (Procrustes + ICP con scala) - factorized s2345, d_P | +0.029 [-0.044, +0.104] | +0.098 [+0.023, +0.169] | +0.052 [-0.020, +0.125] |
| faceverse_neutral | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s1234, d_P | +0.050 [-0.035, +0.133] | +0.092 [+0.009, +0.167] | +0.064 [-0.020, +0.144] |
| faceverse_neutral | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s2345, d_P | +0.011 [-0.074, +0.091] | +0.081 [+0.001, +0.160] | +0.034 [-0.045, +0.113] |
| faceverse_neutral | REALY (gICP con scala + rICP per regione + nICP) - factorized s1234, d_P | +0.188 [+0.096, +0.274] | +0.219 [+0.132, +0.306] | +0.198 [+0.109, +0.282] |
| faceverse_neutral | REALY (gICP con scala + rICP per regione + nICP) - factorized s2345, d_P | +0.149 [+0.058, +0.240] | +0.207 [+0.109, +0.301] | +0.169 [+0.078, +0.262] |
| faceverse_neutral | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s1234, d_P | -0.004 [-0.106, +0.098] | +0.025 [-0.073, +0.122] | +0.006 [-0.094, +0.108] |
| faceverse_neutral | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s2345, d_P | -0.042 [-0.158, +0.080] | +0.013 [-0.092, +0.126] | -0.023 [-0.132, +0.098] |
| famos | NoW originale (Procrustes + ICP con scala) - NoW metrical / MICA (Procrustes senza scala + ICP rigido) | +0.122 [-0.086, +0.347] | - | - |
| famos | NoW originale (Procrustes + ICP con scala) - ICP + Chamfer in mm (per coppia) | +0.155 [-0.058, +0.409] | - | - |
| famos | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - ICP + Chamfer in mm (per coppia) | +0.033 [-0.043, +0.122] | - | - |
| famos | REALY (gICP con scala + rICP per regione + nICP) - ICP + Chamfer in mm (per coppia) | +0.227 [-0.046, +0.564] | - | - |
| famos | Stirling/ESRC (Procrustes con scala, RMSE) - ICP + Chamfer in mm (per coppia) | +0.208 [-0.089, +0.522] | - | - |
| famos | NoW originale (Procrustes + ICP con scala) - factorized s1234, d_P | -0.037 [-0.295, +0.153] | - | - |
| famos | NoW originale (Procrustes + ICP con scala) - factorized s2345, d_P | -0.025 [-0.233, +0.136] | - | - |
| famos | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s1234, d_P | -0.159 [-0.548, +0.197] | - | - |
| famos | NoW metrical / MICA (Procrustes senza scala + ICP rigido) - factorized s2345, d_P | -0.147 [-0.505, +0.163] | - | - |
| famos | REALY (gICP con scala + rICP per regione + nICP) - factorized s1234, d_P | +0.035 [-0.194, +0.243] | - | - |
| famos | REALY (gICP con scala + rICP per regione + nICP) - factorized s2345, d_P | +0.047 [-0.143, +0.237] | - | - |
| famos | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s1234, d_P | +0.016 [-0.205, +0.169] | - | - |
| famos | Stirling/ESRC (Procrustes con scala, RMSE) - factorized s2345, d_P | +0.028 [-0.150, +0.171] | - | - |

## Per perturbazione (P su un lato, l'altro una topologia diversa senza crop)

GT FR:

| dominio | metodo | original | remesh | down8k | up60k | noisy |
|---|---|---|---|---|---|---|
| hifi3d | NoW originale (Procrustes + ICP con scala) | 0.420 [0.340, 0.500] | 0.416 [0.336, 0.496] | 0.423 [0.340, 0.502] | 0.417 [0.337, 0.497] | 0.432 [0.351, 0.513] |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.702 [0.634, 0.758] | 0.698 [0.628, 0.755] | 0.710 [0.640, 0.765] | 0.700 [0.630, 0.756] | 0.719 [0.651, 0.776] |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP) | 0.499 [0.406, 0.578] | 0.498 [0.406, 0.577] | 0.494 [0.401, 0.573] | 0.499 [0.406, 0.579] | 0.498 [0.405, 0.578] |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE) | 0.471 [0.381, 0.557] | 0.465 [0.377, 0.550] | 0.464 [0.374, 0.548] | 0.464 [0.375, 0.548] | 0.476 [0.385, 0.562] |
| hifi3d | ICP + Chamfer in mm (per coppia) | 0.637 [0.566, 0.695] | 0.668 [0.596, 0.729] | 0.621 [0.552, 0.681] | 0.666 [0.594, 0.728] | 0.657 [0.587, 0.715] |
| hifi3d | factorized s1234, d_F non cal. | 0.649 [0.578, 0.712] | 0.642 [0.571, 0.705] | 0.644 [0.573, 0.708] | 0.646 [0.575, 0.710] | 0.633 [0.560, 0.698] |
| facescape | NoW originale (Procrustes + ICP con scala) | 0.608 [0.520, 0.687] | 0.631 [0.541, 0.709] | 0.682 [0.598, 0.755] | 0.638 [0.549, 0.716] | 0.641 [0.554, 0.719] |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.665 [0.597, 0.723] | 0.685 [0.615, 0.743] | 0.735 [0.673, 0.792] | 0.694 [0.627, 0.752] | 0.703 [0.635, 0.758] |
| facescape | REALY (gICP con scala + rICP per regione + nICP) | 0.616 [0.531, 0.692] | 0.619 [0.536, 0.695] | 0.623 [0.540, 0.701] | 0.624 [0.540, 0.699] | 0.630 [0.544, 0.708] |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE) | 0.570 [0.477, 0.664] | 0.580 [0.485, 0.673] | 0.602 [0.504, 0.693] | 0.581 [0.485, 0.673] | 0.586 [0.492, 0.679] |
| facescape | ICP + Chamfer in mm (per coppia) | 0.449 [0.384, 0.508] | 0.556 [0.485, 0.620] | 0.400 [0.345, 0.454] | 0.552 [0.480, 0.616] | 0.489 [0.424, 0.551] |
| facescape | factorized s1234, d_F non cal. | 0.695 [0.620, 0.767] | 0.698 [0.625, 0.767] | 0.684 [0.606, 0.757] | 0.692 [0.618, 0.760] | 0.686 [0.609, 0.759] |
| faceverse | NoW originale (Procrustes + ICP con scala) | 0.313 [0.239, 0.382] | 0.302 [0.229, 0.373] | 0.304 [0.231, 0.374] | 0.297 [0.227, 0.368] | 0.320 [0.243, 0.397] |
| faceverse | NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.352 [0.273, 0.423] | 0.341 [0.268, 0.410] | 0.336 [0.258, 0.410] | 0.335 [0.260, 0.412] | 0.356 [0.275, 0.434] |
| faceverse | REALY (gICP con scala + rICP per regione + nICP) | 0.300 [0.224, 0.368] | 0.297 [0.228, 0.361] | 0.313 [0.236, 0.385] | 0.296 [0.228, 0.359] | 0.308 [0.235, 0.374] |
| faceverse | Stirling/ESRC (Procrustes con scala, RMSE) | 0.211 [0.143, 0.279] | 0.204 [0.133, 0.278] | 0.197 [0.128, 0.268] | 0.190 [0.120, 0.259] | 0.204 [0.135, 0.274] |
| faceverse | ICP + Chamfer in mm (per coppia) | 0.351 [0.275, 0.424] | 0.338 [0.257, 0.413] | 0.324 [0.247, 0.398] | 0.335 [0.255, 0.411] | 0.340 [0.261, 0.419] |
| faceverse | factorized s1234, d_F non cal. | 0.294 [0.220, 0.366] | 0.298 [0.224, 0.366] | 0.288 [0.219, 0.353] | 0.300 [0.230, 0.361] | 0.306 [0.235, 0.373] |
| faceverse_neutral | NoW originale (Procrustes + ICP con scala) | 0.393 [0.315, 0.467] | 0.394 [0.315, 0.467] | 0.389 [0.309, 0.464] | 0.394 [0.317, 0.468] | 0.416 [0.336, 0.493] |
| faceverse_neutral | NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.413 [0.331, 0.489] | 0.414 [0.332, 0.488] | 0.410 [0.324, 0.487] | 0.413 [0.331, 0.489] | 0.434 [0.349, 0.512] |
| faceverse_neutral | REALY (gICP con scala + rICP per regione + nICP) | 0.480 [0.394, 0.559] | 0.480 [0.392, 0.558] | 0.477 [0.392, 0.555] | 0.481 [0.394, 0.559] | 0.488 [0.402, 0.565] |
| faceverse_neutral | Stirling/ESRC (Procrustes con scala, RMSE) | 0.307 [0.213, 0.404] | 0.310 [0.218, 0.407] | 0.310 [0.217, 0.404] | 0.306 [0.213, 0.402] | 0.311 [0.217, 0.410] |
| faceverse_neutral | ICP + Chamfer in mm (per coppia) | 0.415 [0.332, 0.486] | 0.414 [0.331, 0.486] | 0.402 [0.314, 0.475] | 0.412 [0.328, 0.484] | 0.404 [0.325, 0.476] |
| faceverse_neutral | factorized s1234, d_F non cal. | 0.341 [0.259, 0.414] | 0.340 [0.260, 0.413] | 0.335 [0.255, 0.408] | 0.342 [0.259, 0.416] | 0.328 [0.246, 0.397] |

GT SR:

| dominio | metodo | original | remesh | down8k | up60k | noisy |
|---|---|---|---|---|---|---|
| hifi3d | NoW originale (Procrustes + ICP con scala) | 0.623 [0.559, 0.680] | 0.620 [0.555, 0.678] | 0.632 [0.568, 0.690] | 0.622 [0.558, 0.680] | 0.641 [0.577, 0.699] |
| hifi3d | NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.437 [0.356, 0.517] | 0.431 [0.350, 0.512] | 0.432 [0.348, 0.513] | 0.429 [0.347, 0.510] | 0.447 [0.363, 0.527] |
| hifi3d | REALY (gICP con scala + rICP per regione + nICP) | 0.720 [0.652, 0.775] | 0.718 [0.650, 0.775] | 0.715 [0.646, 0.772] | 0.722 [0.653, 0.779] | 0.717 [0.648, 0.774] |
| hifi3d | Stirling/ESRC (Procrustes con scala, RMSE) | 0.640 [0.562, 0.706] | 0.634 [0.556, 0.699] | 0.631 [0.552, 0.698] | 0.632 [0.555, 0.698] | 0.646 [0.568, 0.713] |
| hifi3d | ICP + Chamfer in mm (per coppia) | 0.367 [0.285, 0.454] | 0.383 [0.296, 0.473] | 0.344 [0.262, 0.429] | 0.376 [0.288, 0.466] | 0.377 [0.292, 0.466] |
| hifi3d | factorized s1234, d_P | 0.629 [0.558, 0.693] | 0.626 [0.552, 0.690] | 0.625 [0.555, 0.688] | 0.626 [0.553, 0.690] | 0.604 [0.532, 0.669] |
| facescape | NoW originale (Procrustes + ICP con scala) | 0.710 [0.654, 0.763] | 0.736 [0.683, 0.787] | 0.789 [0.740, 0.833] | 0.743 [0.690, 0.793] | 0.746 [0.693, 0.797] |
| facescape | NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.692 [0.632, 0.744] | 0.714 [0.654, 0.767] | 0.762 [0.706, 0.810] | 0.721 [0.664, 0.773] | 0.726 [0.667, 0.780] |
| facescape | REALY (gICP con scala + rICP per regione + nICP) | 0.717 [0.650, 0.773] | 0.719 [0.652, 0.776] | 0.726 [0.662, 0.781] | 0.725 [0.659, 0.781] | 0.732 [0.667, 0.788] |
| facescape | Stirling/ESRC (Procrustes con scala, RMSE) | 0.669 [0.601, 0.733] | 0.680 [0.611, 0.744] | 0.704 [0.638, 0.766] | 0.680 [0.610, 0.744] | 0.686 [0.618, 0.750] |
| facescape | ICP + Chamfer in mm (per coppia) | 0.471 [0.407, 0.528] | 0.586 [0.521, 0.646] | 0.420 [0.366, 0.470] | 0.581 [0.515, 0.641] | 0.512 [0.449, 0.570] |
| facescape | factorized s1234, d_P | 0.757 [0.688, 0.813] | 0.753 [0.687, 0.809] | 0.746 [0.679, 0.805] | 0.752 [0.685, 0.808] | 0.753 [0.683, 0.809] |
| faceverse | NoW originale (Procrustes + ICP con scala) | 0.319 [0.243, 0.389] | 0.311 [0.237, 0.382] | 0.313 [0.233, 0.384] | 0.311 [0.236, 0.384] | 0.321 [0.240, 0.397] |
| faceverse | NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.319 [0.234, 0.397] | 0.314 [0.238, 0.386] | 0.314 [0.234, 0.394] | 0.313 [0.235, 0.393] | 0.321 [0.235, 0.402] |
| faceverse | REALY (gICP con scala + rICP per regione + nICP) | 0.313 [0.246, 0.376] | 0.310 [0.245, 0.373] | 0.325 [0.253, 0.389] | 0.306 [0.246, 0.364] | 0.312 [0.241, 0.377] |
| faceverse | Stirling/ESRC (Procrustes con scala, RMSE) | 0.206 [0.140, 0.267] | 0.196 [0.126, 0.270] | 0.199 [0.131, 0.265] | 0.194 [0.127, 0.261] | 0.196 [0.126, 0.268] |
| faceverse | ICP + Chamfer in mm (per coppia) | 0.319 [0.240, 0.388] | 0.313 [0.233, 0.384] | 0.301 [0.223, 0.375] | 0.310 [0.233, 0.388] | 0.306 [0.227, 0.380] |
| faceverse | factorized s1234, d_P | 0.282 [0.207, 0.355] | 0.294 [0.217, 0.362] | 0.277 [0.202, 0.340] | 0.290 [0.218, 0.355] | 0.290 [0.214, 0.355] |
| faceverse_neutral | NoW originale (Procrustes + ICP con scala) | 0.407 [0.330, 0.481] | 0.410 [0.333, 0.483] | 0.404 [0.327, 0.479] | 0.410 [0.335, 0.484] | 0.431 [0.350, 0.508] |
| faceverse_neutral | NoW metrical / MICA (Procrustes senza scala + ICP rigido) | 0.389 [0.303, 0.468] | 0.391 [0.303, 0.471] | 0.385 [0.293, 0.466] | 0.390 [0.303, 0.470] | 0.408 [0.319, 0.490] |
| faceverse_neutral | REALY (gICP con scala + rICP per regione + nICP) | 0.523 [0.448, 0.590] | 0.523 [0.449, 0.591] | 0.520 [0.446, 0.588] | 0.523 [0.449, 0.591] | 0.530 [0.454, 0.599] |
| faceverse_neutral | Stirling/ESRC (Procrustes con scala, RMSE) | 0.329 [0.244, 0.418] | 0.334 [0.249, 0.421] | 0.332 [0.246, 0.419] | 0.330 [0.245, 0.417] | 0.333 [0.247, 0.422] |
| faceverse_neutral | ICP + Chamfer in mm (per coppia) | 0.389 [0.305, 0.461] | 0.389 [0.305, 0.463] | 0.376 [0.290, 0.452] | 0.387 [0.301, 0.460] | 0.378 [0.296, 0.450] |
| faceverse_neutral | factorized s1234, d_P | 0.338 [0.256, 0.409] | 0.337 [0.256, 0.407] | 0.332 [0.250, 0.402] | 0.339 [0.256, 0.413] | 0.323 [0.244, 0.392] |

