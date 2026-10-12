# Gromov-Wasserstein e feature percettive su render: Spearman con GT FR / SR

Generato da `aau/competitors/gw_percettivi/gp_eval.py spearman` (non modificare a mano). Righe, semi e repliche di all_cross (`bp_paired_e2.all_cross_rows`) e di FaMoS TEST (`fact_paired.famos_frame`), 1000 repliche per soggetto, IC al 95% percentile. Colonna mm (FR) contro GT FR, colonna a CS_ref (SR) contro GT SR; le altre combinazioni sono in `spearman.csv`. Dichiarata = la variante scelta dalla regola del pilota sui soggetti NON valutati (`gw/pilot_choice.json`, `percettivi/pilot_choice.json`).

GW dichiarato: `euclid|128|cgw2` (punteggio del pilota 0.2354; regola: fra le configurazioni con mediana dei secondi per soluzione <= 1.0: massimo della media sulle 4 viste di (rho(fr, GT FR) + rho(sr, GT SR)) / 2 sulle coppie del pilota; entro 0.01 dal migliore la mediana dei secondi per soluzione piu' bassa).

Percettivi dichiarati: lpips `depth`, dinov2 `depth`, clip `smooth`, dreamsim `smooth` (regola: per metodo: massimo della media sulle 4 viste di (rho(fr, GT FR) + rho(sr, GT SR)) / 2 su tutte le coppie del pilota (soggetti e topologie diversi, crop compreso); entro 0.01 dalla migliore la prima variante nell'ordine flat, smooth, depth).

## senza crop

| metodo [variante] | GT | hifi3d | facescape | faceverse | faceverse_neutral | famos |
|---|---|---|---|---|---|---|
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | FR | 0.467 [0.379, 0.546] | 0.174 [0.146, 0.199] | 0.196 [0.127, 0.258] | 0.234 [0.155, 0.307] | 0.787 [0.505, 0.920] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | SR | 0.228 [0.181, 0.271] | 0.187 [0.154, 0.216] | 0.183 [0.126, 0.244] | 0.225 [0.164, 0.280] | 0.469 [0.099, 0.729] |
| Gromov-Wasserstein (POT) [euclid_256_cg] | FR | 0.435 [0.351, 0.510] | 0.163 [0.137, 0.188] | 0.112 [0.066, 0.158] | 0.141 [0.092, 0.189] | 0.693 [0.344, 0.879] |
| Gromov-Wasserstein (POT) [euclid_256_cg] | SR | 0.225 [0.186, 0.261] | 0.196 [0.165, 0.225] | 0.091 [0.058, 0.128] | 0.121 [0.084, 0.156] | 0.337 [-0.018, 0.576] |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | FR | 0.471 [0.384, 0.548] | 0.198 [0.172, 0.224] | 0.205 [0.140, 0.268] | 0.249 [0.172, 0.321] | 0.795 [0.529, 0.913] |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | SR | 0.264 [0.220, 0.304] | 0.252 [0.219, 0.281] | 0.191 [0.136, 0.250] | 0.243 [0.181, 0.298] | 0.521 [0.220, 0.775] |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | FR | 0.119 [0.086, 0.152] | 0.096 [0.060, 0.130] | 0.065 [0.029, 0.106] | 0.082 [0.041, 0.126] | 0.247 [-0.096, 0.580] |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | SR | 0.150 [0.120, 0.175] | 0.102 [0.067, 0.135] | 0.070 [0.031, 0.107] | 0.092 [0.051, 0.135] | 0.407 [0.077, 0.667] |
| DINOv2 ViT-S/14 su render [flat] | FR | 0.064 [0.037, 0.091] | 0.049 [0.030, 0.071] | 0.026 [0.009, 0.043] | 0.036 [0.018, 0.053] | 0.329 [-0.102, 0.677] |
| DINOv2 ViT-S/14 su render [flat] | SR | 0.078 [0.050, 0.104] | 0.051 [0.031, 0.072] | 0.022 [0.004, 0.038] | 0.035 [0.019, 0.051] | 0.469 [0.054, 0.726] |
| DINOv2 ViT-S/14 su render [smooth] | FR | 0.084 [0.054, 0.112] | 0.060 [0.035, 0.087] | 0.046 [0.021, 0.071] | 0.067 [0.042, 0.091] | 0.363 [-0.076, 0.681] |
| DINOv2 ViT-S/14 su render [smooth] | SR | 0.105 [0.081, 0.129] | 0.063 [0.038, 0.090] | 0.047 [0.021, 0.071] | 0.072 [0.045, 0.096] | 0.428 [-0.038, 0.701] |
| CLIP ViT-B/32 su render [depth] | FR | 0.149 [0.105, 0.193] | 0.090 [0.056, 0.124] | 0.023 [-0.003, 0.047] | 0.035 [0.005, 0.064] | 0.005 [-0.255, 0.344] |
| CLIP ViT-B/32 su render [depth] | SR | 0.143 [0.102, 0.183] | 0.105 [0.072, 0.139] | 0.021 [-0.004, 0.047] | 0.040 [0.012, 0.068] | -0.060 [-0.335, 0.283] |
| CLIP ViT-B/32 su render [flat] | FR | 0.135 [0.087, 0.186] | 0.104 [0.063, 0.148] | 0.034 [0.007, 0.059] | 0.043 [0.016, 0.069] | 0.474 [0.064, 0.739] |
| CLIP ViT-B/32 su render [flat] | SR | 0.153 [0.115, 0.189] | 0.094 [0.057, 0.135] | 0.034 [0.010, 0.057] | 0.055 [0.027, 0.084] | 0.653 [0.306, 0.831] |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | FR | 0.152 [0.103, 0.199] | 0.122 [0.081, 0.165] | 0.049 [0.018, 0.081] | 0.067 [0.038, 0.096] | 0.447 [0.077, 0.704] |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | SR | 0.173 [0.133, 0.213] | 0.116 [0.076, 0.157] | 0.052 [0.022, 0.081] | 0.079 [0.048, 0.108] | 0.608 [0.241, 0.809] |
| LPIPS alex su render [depth] **(dichiarata)** | FR | 0.615 [0.532, 0.692] | 0.282 [0.209, 0.348] | 0.137 [0.092, 0.183] | 0.160 [0.109, 0.210] | 0.599 [0.147, 0.874] |
| LPIPS alex su render [depth] **(dichiarata)** | SR | 0.349 [0.289, 0.414] | 0.270 [0.206, 0.329] | 0.112 [0.071, 0.156] | 0.140 [0.095, 0.189] | 0.246 [-0.148, 0.573] |
| LPIPS alex su render [flat] | FR | 0.356 [0.316, 0.389] | 0.180 [0.143, 0.214] | 0.091 [0.065, 0.117] | 0.136 [0.105, 0.166] | 0.676 [0.282, 0.890] |
| LPIPS alex su render [flat] | SR | 0.260 [0.228, 0.291] | 0.180 [0.145, 0.212] | 0.087 [0.061, 0.115] | 0.130 [0.100, 0.160] | 0.482 [0.072, 0.759] |
| LPIPS alex su render [smooth] | FR | 0.416 [0.360, 0.470] | 0.224 [0.180, 0.262] | 0.112 [0.082, 0.142] | 0.166 [0.128, 0.201] | 0.658 [0.257, 0.892] |
| LPIPS alex su render [smooth] | SR | 0.285 [0.251, 0.318] | 0.228 [0.185, 0.265] | 0.111 [0.080, 0.143] | 0.157 [0.121, 0.192] | 0.442 [0.030, 0.729] |
| DreamSim ensemble su render [depth] | FR | 0.187 [0.142, 0.235] | 0.189 [0.151, 0.225] | 0.041 [0.008, 0.075] | 0.076 [0.027, 0.130] | 0.233 [-0.096, 0.589] |
| DreamSim ensemble su render [depth] | SR | 0.216 [0.173, 0.258] | 0.185 [0.145, 0.224] | 0.047 [0.014, 0.081] | 0.087 [0.038, 0.142] | 0.222 [-0.129, 0.583] |
| DreamSim ensemble su render [flat] | FR | 0.155 [0.107, 0.203] | 0.122 [0.092, 0.155] | 0.048 [0.026, 0.071] | 0.082 [0.053, 0.109] | 0.546 [0.171, 0.804] |
| DreamSim ensemble su render [flat] | SR | 0.166 [0.127, 0.202] | 0.121 [0.091, 0.154] | 0.047 [0.024, 0.069] | 0.091 [0.062, 0.118] | 0.673 [0.378, 0.826] |
| DreamSim ensemble su render [smooth] **(dichiarata)** | FR | 0.183 [0.135, 0.230] | 0.151 [0.117, 0.186] | 0.061 [0.034, 0.088] | 0.097 [0.068, 0.125] | 0.499 [0.120, 0.769] |
| DreamSim ensemble su render [smooth] **(dichiarata)** | SR | 0.207 [0.167, 0.243] | 0.157 [0.122, 0.191] | 0.060 [0.034, 0.088] | 0.107 [0.078, 0.134] | 0.639 [0.323, 0.812] |

## righe col crop

| metodo [variante] | GT | hifi3d | facescape | faceverse | faceverse_neutral |
|---|---|---|---|---|---|
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | FR | 0.159 [0.116, 0.204] | 0.063 [0.025, 0.104] | 0.086 [0.059, 0.115] | 0.089 [0.062, 0.115] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | SR | 0.255 [0.202, 0.308] | 0.251 [0.199, 0.301] | 0.189 [0.140, 0.235] | 0.203 [0.151, 0.252] |
| Gromov-Wasserstein (POT) [euclid_256_cg] | FR | 0.181 [0.138, 0.224] | 0.075 [0.034, 0.118] | 0.071 [0.045, 0.098] | 0.085 [0.059, 0.111] |
| Gromov-Wasserstein (POT) [euclid_256_cg] | SR | 0.208 [0.164, 0.253] | 0.207 [0.164, 0.250] | 0.099 [0.066, 0.132] | 0.121 [0.087, 0.152] |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | FR | 0.157 [0.114, 0.202] | 0.065 [0.029, 0.106] | 0.084 [0.057, 0.113] | 0.093 [0.067, 0.119] |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | SR | 0.260 [0.205, 0.314] | 0.277 [0.220, 0.328] | 0.194 [0.143, 0.241] | 0.212 [0.159, 0.262] |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | FR | 0.163 [0.110, 0.219] | 0.143 [0.078, 0.199] | 0.102 [0.050, 0.149] | 0.104 [0.054, 0.159] |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | SR | 0.114 [0.058, 0.166] | 0.137 [0.079, 0.197] | 0.111 [0.063, 0.159] | 0.120 [0.077, 0.170] |
| DINOv2 ViT-S/14 su render [flat] | FR | 0.144 [0.085, 0.208] | 0.116 [0.066, 0.168] | 0.065 [0.036, 0.093] | 0.082 [0.048, 0.116] |
| DINOv2 ViT-S/14 su render [flat] | SR | 0.127 [0.079, 0.177] | 0.125 [0.081, 0.174] | 0.073 [0.045, 0.099] | 0.097 [0.067, 0.125] |
| DINOv2 ViT-S/14 su render [smooth] | FR | 0.131 [0.081, 0.186] | 0.142 [0.082, 0.204] | 0.094 [0.055, 0.131] | 0.118 [0.074, 0.164] |
| DINOv2 ViT-S/14 su render [smooth] | SR | 0.113 [0.074, 0.151] | 0.146 [0.095, 0.199] | 0.109 [0.073, 0.147] | 0.139 [0.095, 0.179] |
| CLIP ViT-B/32 su render [depth] | FR | 0.209 [0.140, 0.282] | 0.076 [0.018, 0.132] | 0.053 [0.015, 0.091] | 0.051 [0.007, 0.094] |
| CLIP ViT-B/32 su render [depth] | SR | 0.193 [0.135, 0.252] | 0.137 [0.082, 0.191] | 0.056 [0.016, 0.096] | 0.073 [0.026, 0.116] |
| CLIP ViT-B/32 su render [flat] | FR | 0.211 [0.140, 0.284] | 0.177 [0.111, 0.241] | 0.075 [0.035, 0.115] | 0.090 [0.047, 0.132] |
| CLIP ViT-B/32 su render [flat] | SR | 0.236 [0.185, 0.287] | 0.170 [0.109, 0.232] | 0.091 [0.050, 0.133] | 0.111 [0.070, 0.155] |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | FR | 0.227 [0.159, 0.301] | 0.190 [0.121, 0.251] | 0.083 [0.039, 0.126] | 0.101 [0.056, 0.149] |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | SR | 0.233 [0.180, 0.285] | 0.178 [0.115, 0.239] | 0.101 [0.057, 0.142] | 0.118 [0.071, 0.165] |
| LPIPS alex su render [depth] **(dichiarata)** | FR | 0.457 [0.385, 0.518] | 0.162 [0.098, 0.227] | 0.124 [0.075, 0.176] | 0.149 [0.096, 0.206] |
| LPIPS alex su render [depth] **(dichiarata)** | SR | 0.322 [0.256, 0.391] | 0.311 [0.225, 0.394] | 0.123 [0.071, 0.175] | 0.149 [0.094, 0.203] |
| LPIPS alex su render [flat] | FR | 0.336 [0.287, 0.379] | 0.119 [0.075, 0.166] | 0.131 [0.091, 0.172] | 0.180 [0.139, 0.222] |
| LPIPS alex su render [flat] | SR | 0.250 [0.211, 0.292] | 0.249 [0.186, 0.309] | 0.133 [0.092, 0.173] | 0.186 [0.144, 0.226] |
| LPIPS alex su render [smooth] | FR | 0.384 [0.322, 0.441] | 0.120 [0.075, 0.169] | 0.147 [0.105, 0.192] | 0.193 [0.149, 0.239] |
| LPIPS alex su render [smooth] | SR | 0.253 [0.213, 0.295] | 0.264 [0.198, 0.327] | 0.160 [0.114, 0.205] | 0.202 [0.157, 0.245] |
| DreamSim ensemble su render [depth] | FR | 0.245 [0.170, 0.323] | 0.182 [0.123, 0.239] | 0.068 [0.024, 0.107] | 0.081 [0.024, 0.143] |
| DreamSim ensemble su render [depth] | SR | 0.286 [0.215, 0.351] | 0.210 [0.152, 0.270] | 0.069 [0.029, 0.107] | 0.088 [0.030, 0.150] |
| DreamSim ensemble su render [flat] | FR | 0.235 [0.162, 0.307] | 0.208 [0.150, 0.268] | 0.104 [0.066, 0.139] | 0.145 [0.098, 0.186] |
| DreamSim ensemble su render [flat] | SR | 0.266 [0.212, 0.316] | 0.244 [0.187, 0.304] | 0.109 [0.069, 0.144] | 0.168 [0.121, 0.209] |
| DreamSim ensemble su render [smooth] **(dichiarata)** | FR | 0.246 [0.173, 0.318] | 0.224 [0.163, 0.285] | 0.113 [0.074, 0.148] | 0.150 [0.100, 0.192] |
| DreamSim ensemble su render [smooth] **(dichiarata)** | SR | 0.279 [0.225, 0.331] | 0.262 [0.203, 0.323] | 0.116 [0.075, 0.153] | 0.172 [0.123, 0.214] |

## all_cross

| metodo [variante] | GT | hifi3d | facescape | faceverse | faceverse_neutral |
|---|---|---|---|---|---|
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | FR | 0.354 [0.280, 0.422] | 0.086 [0.070, 0.102] | 0.148 [0.103, 0.194] | 0.169 [0.118, 0.218] |
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | SR | 0.192 [0.148, 0.235] | 0.158 [0.126, 0.188] | 0.179 [0.131, 0.230] | 0.207 [0.153, 0.257] |
| Gromov-Wasserstein (POT) [euclid_256_cg] | FR | 0.341 [0.270, 0.408] | 0.086 [0.069, 0.103] | 0.092 [0.056, 0.129] | 0.113 [0.076, 0.149] |
| Gromov-Wasserstein (POT) [euclid_256_cg] | SR | 0.183 [0.146, 0.220] | 0.163 [0.131, 0.191] | 0.089 [0.060, 0.118] | 0.112 [0.079, 0.144] |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | FR | 0.356 [0.283, 0.423] | 0.097 [0.082, 0.113] | 0.152 [0.108, 0.198] | 0.178 [0.127, 0.226] |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | SR | 0.210 [0.167, 0.252] | 0.206 [0.170, 0.238] | 0.185 [0.138, 0.233] | 0.219 [0.165, 0.271] |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | FR | 0.126 [0.090, 0.164] | 0.101 [0.061, 0.138] | 0.077 [0.039, 0.117] | 0.088 [0.047, 0.134] |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | SR | 0.124 [0.095, 0.150] | 0.106 [0.069, 0.140] | 0.083 [0.045, 0.122] | 0.099 [0.057, 0.143] |
| DINOv2 ViT-S/14 su render [flat] | FR | 0.081 [0.048, 0.113] | 0.060 [0.035, 0.086] | 0.035 [0.019, 0.051] | 0.047 [0.028, 0.064] |
| DINOv2 ViT-S/14 su render [flat] | SR | 0.085 [0.054, 0.113] | 0.063 [0.040, 0.089] | 0.035 [0.018, 0.051] | 0.049 [0.032, 0.065] |
| DINOv2 ViT-S/14 su render [smooth] | FR | 0.096 [0.061, 0.132] | 0.083 [0.049, 0.119] | 0.059 [0.033, 0.082] | 0.080 [0.051, 0.108] |
| DINOv2 ViT-S/14 su render [smooth] | SR | 0.106 [0.079, 0.131] | 0.085 [0.055, 0.118] | 0.063 [0.039, 0.086] | 0.089 [0.060, 0.115] |
| CLIP ViT-B/32 su render [depth] | FR | 0.164 [0.113, 0.216] | 0.083 [0.046, 0.120] | 0.032 [0.004, 0.059] | 0.039 [0.006, 0.070] |
| CLIP ViT-B/32 su render [depth] | SR | 0.154 [0.109, 0.197] | 0.109 [0.072, 0.146] | 0.031 [0.005, 0.060] | 0.048 [0.017, 0.081] |
| CLIP ViT-B/32 su render [flat] | FR | 0.153 [0.100, 0.208] | 0.124 [0.078, 0.172] | 0.046 [0.019, 0.072] | 0.056 [0.028, 0.085] |
| CLIP ViT-B/32 su render [flat] | SR | 0.173 [0.134, 0.212] | 0.115 [0.072, 0.161] | 0.051 [0.024, 0.077] | 0.071 [0.040, 0.102] |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | FR | 0.170 [0.117, 0.224] | 0.142 [0.093, 0.188] | 0.058 [0.026, 0.089] | 0.075 [0.042, 0.108] |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | SR | 0.186 [0.145, 0.225] | 0.134 [0.089, 0.179] | 0.066 [0.035, 0.096] | 0.088 [0.056, 0.122] |
| LPIPS alex su render [depth] **(dichiarata)** | FR | 0.544 [0.462, 0.617] | 0.178 [0.125, 0.226] | 0.130 [0.087, 0.174] | 0.150 [0.101, 0.197] |
| LPIPS alex su render [depth] **(dichiarata)** | SR | 0.265 [0.213, 0.320] | 0.238 [0.176, 0.297] | 0.110 [0.073, 0.153] | 0.134 [0.091, 0.181] |
| LPIPS alex su render [flat] | FR | 0.318 [0.273, 0.358] | 0.106 [0.079, 0.131] | 0.101 [0.073, 0.127] | 0.144 [0.110, 0.174] |
| LPIPS alex su render [flat] | SR | 0.157 [0.133, 0.180] | 0.148 [0.113, 0.179] | 0.095 [0.069, 0.123] | 0.135 [0.103, 0.165] |
| LPIPS alex su render [smooth] | FR | 0.385 [0.321, 0.445] | 0.125 [0.095, 0.152] | 0.117 [0.086, 0.149] | 0.161 [0.124, 0.195] |
| LPIPS alex su render [smooth] | SR | 0.171 [0.147, 0.195] | 0.162 [0.125, 0.196] | 0.113 [0.084, 0.146] | 0.147 [0.113, 0.179] |
| DreamSim ensemble su render [depth] | FR | 0.189 [0.137, 0.243] | 0.158 [0.120, 0.199] | 0.050 [0.017, 0.082] | 0.077 [0.024, 0.132] |
| DreamSim ensemble su render [depth] | SR | 0.219 [0.171, 0.266] | 0.167 [0.125, 0.209] | 0.054 [0.023, 0.086] | 0.087 [0.035, 0.145] |
| DreamSim ensemble su render [flat] | FR | 0.172 [0.119, 0.224] | 0.145 [0.106, 0.184] | 0.065 [0.040, 0.089] | 0.100 [0.067, 0.129] |
| DreamSim ensemble su render [flat] | SR | 0.188 [0.147, 0.227] | 0.152 [0.116, 0.192] | 0.066 [0.039, 0.088] | 0.113 [0.080, 0.144] |
| DreamSim ensemble su render [smooth] **(dichiarata)** | FR | 0.191 [0.137, 0.243] | 0.169 [0.128, 0.211] | 0.075 [0.049, 0.102] | 0.111 [0.076, 0.142] |
| DreamSim ensemble su render [smooth] **(dichiarata)** | SR | 0.215 [0.175, 0.254] | 0.182 [0.141, 0.224] | 0.076 [0.049, 0.102] | 0.124 [0.090, 0.154] |

## Scomposizione di "senza crop" per topologia, GT FR (punto; IC in `spearman.csv`)

Colonne: dominio | gruppo (`tutte` = senza crop; `no noisy` = senza crop e senza noisy; `con t` = una delle due mesh in topologia t, l'altra in una topologia diversa senza crop).

| metodo [variante] | dominio | tutte | no noisy | con original | con remesh | con down8k | con up60k | con noisy |
|---|---|---|---|---|---|---|---|---|
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | hifi3d | 0.467 | 0.684 | 0.537 | 0.552 | 0.535 | 0.535 | 0.282 |
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | facescape | 0.174 | 0.473 | 0.275 | 0.245 | 0.276 | 0.278 | 0.024 |
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | faceverse | 0.196 | 0.227 | 0.210 | 0.199 | 0.191 | 0.214 | 0.210 |
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | faceverse_neutral | 0.234 | 0.310 | 0.258 | 0.244 | 0.256 | 0.257 | 0.266 |
| Gromov-Wasserstein (POT) [euclid_256_cg] | hifi3d | 0.435 | 0.642 | 0.500 | 0.513 | 0.499 | 0.499 | 0.280 |
| Gromov-Wasserstein (POT) [euclid_256_cg] | facescape | 0.163 | 0.434 | 0.260 | 0.218 | 0.258 | 0.256 | 0.044 |
| Gromov-Wasserstein (POT) [euclid_256_cg] | faceverse | 0.112 | 0.130 | 0.119 | 0.113 | 0.107 | 0.131 | 0.114 |
| Gromov-Wasserstein (POT) [euclid_256_cg] | faceverse_neutral | 0.141 | 0.180 | 0.161 | 0.144 | 0.151 | 0.150 | 0.149 |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | hifi3d | 0.471 | 0.698 | 0.545 | 0.558 | 0.543 | 0.544 | 0.274 |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | facescape | 0.198 | 0.534 | 0.318 | 0.263 | 0.318 | 0.317 | 0.038 |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | faceverse | 0.205 | 0.240 | 0.217 | 0.210 | 0.202 | 0.227 | 0.216 |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | faceverse_neutral | 0.249 | 0.339 | 0.276 | 0.262 | 0.276 | 0.276 | 0.275 |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | hifi3d | 0.119 | 0.258 | 0.190 | 0.162 | 0.150 | 0.153 | 0.140 |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | facescape | 0.096 | 0.206 | 0.123 | 0.149 | 0.116 | 0.134 | 0.119 |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | faceverse | 0.065 | 0.107 | 0.077 | 0.069 | 0.080 | 0.085 | 0.065 |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | faceverse_neutral | 0.082 | 0.149 | 0.099 | 0.099 | 0.115 | 0.094 | 0.100 |
| DINOv2 ViT-S/14 su render [flat] | hifi3d | 0.064 | 0.140 | 0.114 | 0.078 | 0.100 | 0.113 | 0.083 |
| DINOv2 ViT-S/14 su render [flat] | facescape | 0.049 | 0.129 | 0.099 | 0.095 | 0.097 | 0.098 | 0.018 |
| DINOv2 ViT-S/14 su render [flat] | faceverse | 0.026 | 0.064 | 0.042 | 0.055 | 0.039 | 0.042 | 0.018 |
| DINOv2 ViT-S/14 su render [flat] | faceverse_neutral | 0.036 | 0.088 | 0.061 | 0.072 | 0.059 | 0.060 | 0.027 |
| DINOv2 ViT-S/14 su render [smooth] | hifi3d | 0.084 | 0.177 | 0.148 | 0.076 | 0.123 | 0.127 | 0.126 |
| DINOv2 ViT-S/14 su render [smooth] | facescape | 0.060 | 0.154 | 0.120 | 0.082 | 0.109 | 0.084 | 0.031 |
| DINOv2 ViT-S/14 su render [smooth] | faceverse | 0.046 | 0.095 | 0.064 | 0.069 | 0.056 | 0.060 | 0.076 |
| DINOv2 ViT-S/14 su render [smooth] | faceverse_neutral | 0.067 | 0.143 | 0.102 | 0.089 | 0.095 | 0.082 | 0.097 |
| CLIP ViT-B/32 su render [depth] | hifi3d | 0.149 | 0.297 | 0.204 | 0.198 | 0.208 | 0.172 | 0.168 |
| CLIP ViT-B/32 su render [depth] | facescape | 0.090 | 0.191 | 0.135 | 0.123 | 0.121 | 0.098 | 0.119 |
| CLIP ViT-B/32 su render [depth] | faceverse | 0.023 | 0.053 | 0.039 | 0.007 | 0.033 | 0.050 | 0.012 |
| CLIP ViT-B/32 su render [depth] | faceverse_neutral | 0.035 | 0.065 | 0.044 | 0.040 | 0.041 | 0.043 | 0.070 |
| CLIP ViT-B/32 su render [flat] | hifi3d | 0.135 | 0.315 | 0.196 | 0.179 | 0.189 | 0.195 | 0.137 |
| CLIP ViT-B/32 su render [flat] | facescape | 0.104 | 0.244 | 0.154 | 0.157 | 0.152 | 0.152 | 0.101 |
| CLIP ViT-B/32 su render [flat] | faceverse | 0.034 | 0.094 | 0.059 | 0.047 | 0.055 | 0.060 | 0.001 |
| CLIP ViT-B/32 su render [flat] | faceverse_neutral | 0.043 | 0.109 | 0.070 | 0.053 | 0.069 | 0.071 | 0.023 |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | hifi3d | 0.152 | 0.354 | 0.218 | 0.198 | 0.213 | 0.211 | 0.151 |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | facescape | 0.122 | 0.294 | 0.182 | 0.167 | 0.179 | 0.173 | 0.099 |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | faceverse | 0.049 | 0.099 | 0.068 | 0.049 | 0.059 | 0.069 | 0.081 |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | faceverse_neutral | 0.067 | 0.129 | 0.088 | 0.069 | 0.087 | 0.082 | 0.126 |
| LPIPS alex su render [depth] **(dichiarata)** | hifi3d | 0.615 | 0.702 | 0.639 | 0.628 | 0.645 | 0.638 | 0.648 |
| LPIPS alex su render [depth] **(dichiarata)** | facescape | 0.282 | 0.456 | 0.335 | 0.310 | 0.336 | 0.333 | 0.363 |
| LPIPS alex su render [depth] **(dichiarata)** | faceverse | 0.137 | 0.199 | 0.163 | 0.143 | 0.142 | 0.160 | 0.180 |
| LPIPS alex su render [depth] **(dichiarata)** | faceverse_neutral | 0.160 | 0.264 | 0.185 | 0.180 | 0.190 | 0.185 | 0.230 |
| LPIPS alex su render [flat] | hifi3d | 0.356 | 0.755 | 0.461 | 0.454 | 0.460 | 0.461 | 0.523 |
| LPIPS alex su render [flat] | facescape | 0.180 | 0.409 | 0.265 | 0.312 | 0.263 | 0.264 | 0.206 |
| LPIPS alex su render [flat] | faceverse | 0.091 | 0.222 | 0.139 | 0.140 | 0.130 | 0.140 | 0.068 |
| LPIPS alex su render [flat] | faceverse_neutral | 0.136 | 0.340 | 0.208 | 0.221 | 0.205 | 0.208 | 0.089 |
| LPIPS alex su render [smooth] | hifi3d | 0.416 | 0.755 | 0.500 | 0.487 | 0.509 | 0.499 | 0.629 |
| LPIPS alex su render [smooth] | facescape | 0.224 | 0.503 | 0.318 | 0.312 | 0.317 | 0.307 | 0.268 |
| LPIPS alex su render [smooth] | faceverse | 0.112 | 0.259 | 0.167 | 0.153 | 0.153 | 0.158 | 0.116 |
| LPIPS alex su render [smooth] | faceverse_neutral | 0.166 | 0.387 | 0.239 | 0.237 | 0.237 | 0.229 | 0.163 |
| DreamSim ensemble su render [depth] | hifi3d | 0.187 | 0.384 | 0.242 | 0.233 | 0.246 | 0.240 | 0.236 |
| DreamSim ensemble su render [depth] | facescape | 0.189 | 0.424 | 0.263 | 0.238 | 0.262 | 0.264 | 0.212 |
| DreamSim ensemble su render [depth] | faceverse | 0.041 | 0.043 | 0.042 | 0.031 | 0.030 | 0.061 | 0.050 |
| DreamSim ensemble su render [depth] | faceverse_neutral | 0.076 | 0.088 | 0.077 | 0.080 | 0.080 | 0.077 | 0.094 |
| DreamSim ensemble su render [flat] | hifi3d | 0.155 | 0.364 | 0.227 | 0.215 | 0.218 | 0.227 | 0.153 |
| DreamSim ensemble su render [flat] | facescape | 0.122 | 0.304 | 0.195 | 0.216 | 0.195 | 0.195 | 0.080 |
| DreamSim ensemble su render [flat] | faceverse | 0.048 | 0.132 | 0.079 | 0.070 | 0.070 | 0.085 | 0.006 |
| DreamSim ensemble su render [flat] | faceverse_neutral | 0.082 | 0.208 | 0.126 | 0.120 | 0.125 | 0.126 | 0.043 |
| DreamSim ensemble su render [smooth] **(dichiarata)** | hifi3d | 0.183 | 0.406 | 0.255 | 0.231 | 0.247 | 0.248 | 0.228 |
| DreamSim ensemble su render [smooth] **(dichiarata)** | facescape | 0.151 | 0.366 | 0.243 | 0.195 | 0.240 | 0.211 | 0.121 |
| DreamSim ensemble su render [smooth] **(dichiarata)** | faceverse | 0.061 | 0.138 | 0.089 | 0.073 | 0.074 | 0.092 | 0.071 |
| DreamSim ensemble su render [smooth] **(dichiarata)** | faceverse_neutral | 0.097 | 0.225 | 0.141 | 0.124 | 0.140 | 0.132 | 0.103 |

## Scomposizione di "senza crop" per topologia, GT SR (punto; IC in `spearman.csv`)

Colonne: dominio | gruppo (`tutte` = senza crop; `no noisy` = senza crop e senza noisy; `con t` = una delle due mesh in topologia t, l'altra in una topologia diversa senza crop).

| metodo [variante] | dominio | tutte | no noisy | con original | con remesh | con down8k | con up60k | con noisy |
|---|---|---|---|---|---|---|---|---|
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | hifi3d | 0.228 | 0.516 | 0.309 | 0.310 | 0.306 | 0.303 | 0.267 |
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | facescape | 0.187 | 0.487 | 0.277 | 0.284 | 0.281 | 0.275 | 0.074 |
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | faceverse | 0.183 | 0.243 | 0.201 | 0.197 | 0.197 | 0.204 | 0.189 |
| Gromov-Wasserstein (POT) [euclid_128_cgw2] **(dichiarata)** | faceverse_neutral | 0.225 | 0.350 | 0.262 | 0.252 | 0.263 | 0.260 | 0.234 |
| Gromov-Wasserstein (POT) [euclid_256_cg] | hifi3d | 0.225 | 0.505 | 0.306 | 0.308 | 0.307 | 0.308 | 0.172 |
| Gromov-Wasserstein (POT) [euclid_256_cg] | facescape | 0.196 | 0.504 | 0.291 | 0.291 | 0.291 | 0.285 | 0.093 |
| Gromov-Wasserstein (POT) [euclid_256_cg] | faceverse | 0.091 | 0.123 | 0.099 | 0.097 | 0.096 | 0.108 | 0.087 |
| Gromov-Wasserstein (POT) [euclid_256_cg] | faceverse_neutral | 0.121 | 0.182 | 0.147 | 0.130 | 0.137 | 0.139 | 0.120 |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | hifi3d | 0.264 | 0.605 | 0.358 | 0.358 | 0.360 | 0.359 | 0.291 |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | facescape | 0.252 | 0.650 | 0.373 | 0.373 | 0.375 | 0.371 | 0.110 |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | faceverse | 0.191 | 0.260 | 0.209 | 0.211 | 0.206 | 0.218 | 0.191 |
| Gromov-Wasserstein (POT) [euclid_256_cgw2] | faceverse_neutral | 0.243 | 0.399 | 0.290 | 0.280 | 0.289 | 0.291 | 0.235 |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | hifi3d | 0.150 | 0.310 | 0.229 | 0.199 | 0.194 | 0.182 | 0.217 |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | facescape | 0.102 | 0.215 | 0.128 | 0.167 | 0.125 | 0.130 | 0.137 |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | faceverse | 0.070 | 0.110 | 0.083 | 0.074 | 0.080 | 0.089 | 0.075 |
| DINOv2 ViT-S/14 su render [depth] **(dichiarata)** | faceverse_neutral | 0.092 | 0.160 | 0.106 | 0.105 | 0.130 | 0.102 | 0.124 |
| DINOv2 ViT-S/14 su render [flat] | hifi3d | 0.078 | 0.196 | 0.147 | 0.119 | 0.129 | 0.147 | 0.046 |
| DINOv2 ViT-S/14 su render [flat] | facescape | 0.051 | 0.142 | 0.104 | 0.111 | 0.102 | 0.104 | -0.004 |
| DINOv2 ViT-S/14 su render [flat] | faceverse | 0.022 | 0.065 | 0.044 | 0.051 | 0.038 | 0.041 | -0.009 |
| DINOv2 ViT-S/14 su render [flat] | faceverse_neutral | 0.035 | 0.091 | 0.062 | 0.074 | 0.061 | 0.062 | 0.015 |
| DINOv2 ViT-S/14 su render [smooth] | hifi3d | 0.105 | 0.227 | 0.185 | 0.108 | 0.155 | 0.157 | 0.148 |
| DINOv2 ViT-S/14 su render [smooth] | facescape | 0.063 | 0.157 | 0.123 | 0.083 | 0.111 | 0.087 | 0.040 |
| DINOv2 ViT-S/14 su render [smooth] | faceverse | 0.047 | 0.097 | 0.069 | 0.066 | 0.057 | 0.062 | 0.074 |
| DINOv2 ViT-S/14 su render [smooth] | faceverse_neutral | 0.072 | 0.147 | 0.105 | 0.093 | 0.100 | 0.086 | 0.121 |
| CLIP ViT-B/32 su render [depth] | hifi3d | 0.143 | 0.276 | 0.194 | 0.185 | 0.191 | 0.162 | 0.208 |
| CLIP ViT-B/32 su render [depth] | facescape | 0.105 | 0.228 | 0.158 | 0.144 | 0.136 | 0.127 | 0.133 |
| CLIP ViT-B/32 su render [depth] | faceverse | 0.021 | 0.050 | 0.038 | 0.009 | 0.028 | 0.044 | 0.007 |
| CLIP ViT-B/32 su render [depth] | faceverse_neutral | 0.040 | 0.077 | 0.053 | 0.043 | 0.049 | 0.051 | 0.074 |
| CLIP ViT-B/32 su render [flat] | hifi3d | 0.153 | 0.379 | 0.234 | 0.210 | 0.222 | 0.234 | 0.104 |
| CLIP ViT-B/32 su render [flat] | facescape | 0.094 | 0.230 | 0.148 | 0.138 | 0.147 | 0.148 | 0.074 |
| CLIP ViT-B/32 su render [flat] | faceverse | 0.034 | 0.109 | 0.068 | 0.057 | 0.059 | 0.064 | -0.030 |
| CLIP ViT-B/32 su render [flat] | faceverse_neutral | 0.055 | 0.145 | 0.090 | 0.074 | 0.089 | 0.092 | 0.016 |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | hifi3d | 0.173 | 0.420 | 0.258 | 0.232 | 0.247 | 0.250 | 0.138 |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | facescape | 0.116 | 0.282 | 0.178 | 0.150 | 0.174 | 0.169 | 0.091 |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | faceverse | 0.052 | 0.110 | 0.078 | 0.058 | 0.063 | 0.072 | 0.076 |
| CLIP ViT-B/32 su render [smooth] **(dichiarata)** | faceverse_neutral | 0.079 | 0.156 | 0.104 | 0.084 | 0.103 | 0.100 | 0.140 |
| LPIPS alex su render [depth] **(dichiarata)** | hifi3d | 0.349 | 0.520 | 0.394 | 0.390 | 0.403 | 0.393 | 0.385 |
| LPIPS alex su render [depth] **(dichiarata)** | facescape | 0.270 | 0.472 | 0.325 | 0.315 | 0.326 | 0.326 | 0.372 |
| LPIPS alex su render [depth] **(dichiarata)** | faceverse | 0.112 | 0.170 | 0.133 | 0.122 | 0.121 | 0.126 | 0.151 |
| LPIPS alex su render [depth] **(dichiarata)** | faceverse_neutral | 0.140 | 0.232 | 0.161 | 0.157 | 0.166 | 0.161 | 0.214 |
| LPIPS alex su render [flat] | hifi3d | 0.260 | 0.626 | 0.370 | 0.375 | 0.366 | 0.370 | 0.215 |
| LPIPS alex su render [flat] | facescape | 0.180 | 0.412 | 0.263 | 0.335 | 0.261 | 0.263 | 0.200 |
| LPIPS alex su render [flat] | faceverse | 0.087 | 0.214 | 0.132 | 0.143 | 0.127 | 0.136 | 0.063 |
| LPIPS alex su render [flat] | faceverse_neutral | 0.130 | 0.319 | 0.197 | 0.216 | 0.194 | 0.197 | 0.098 |
| LPIPS alex su render [smooth] | hifi3d | 0.285 | 0.631 | 0.381 | 0.381 | 0.379 | 0.379 | 0.360 |
| LPIPS alex su render [smooth] | facescape | 0.228 | 0.517 | 0.324 | 0.331 | 0.322 | 0.311 | 0.259 |
| LPIPS alex su render [smooth] | faceverse | 0.111 | 0.256 | 0.164 | 0.155 | 0.156 | 0.157 | 0.117 |
| LPIPS alex su render [smooth] | faceverse_neutral | 0.157 | 0.365 | 0.227 | 0.226 | 0.225 | 0.217 | 0.159 |
| DreamSim ensemble su render [depth] | hifi3d | 0.216 | 0.450 | 0.282 | 0.270 | 0.288 | 0.280 | 0.255 |
| DreamSim ensemble su render [depth] | facescape | 0.185 | 0.419 | 0.257 | 0.238 | 0.258 | 0.260 | 0.206 |
| DreamSim ensemble su render [depth] | faceverse | 0.047 | 0.052 | 0.052 | 0.039 | 0.038 | 0.064 | 0.054 |
| DreamSim ensemble su render [depth] | faceverse_neutral | 0.087 | 0.095 | 0.088 | 0.087 | 0.089 | 0.088 | 0.113 |
| DreamSim ensemble su render [flat] | hifi3d | 0.166 | 0.412 | 0.254 | 0.240 | 0.244 | 0.255 | 0.107 |
| DreamSim ensemble su render [flat] | facescape | 0.121 | 0.315 | 0.199 | 0.223 | 0.200 | 0.200 | 0.050 |
| DreamSim ensemble su render [flat] | faceverse | 0.047 | 0.139 | 0.085 | 0.074 | 0.072 | 0.085 | -0.019 |
| DreamSim ensemble su render [flat] | faceverse_neutral | 0.091 | 0.243 | 0.146 | 0.138 | 0.145 | 0.145 | 0.019 |
| DreamSim ensemble su render [smooth] **(dichiarata)** | hifi3d | 0.207 | 0.462 | 0.291 | 0.261 | 0.283 | 0.281 | 0.252 |
| DreamSim ensemble su render [smooth] **(dichiarata)** | facescape | 0.157 | 0.388 | 0.256 | 0.206 | 0.254 | 0.222 | 0.106 |
| DreamSim ensemble su render [smooth] **(dichiarata)** | faceverse | 0.060 | 0.145 | 0.094 | 0.076 | 0.078 | 0.093 | 0.052 |
| DreamSim ensemble su render [smooth] **(dichiarata)** | faceverse_neutral | 0.107 | 0.258 | 0.161 | 0.140 | 0.160 | 0.150 | 0.087 |

## Costi

| metodo | variante | vista | costo |
|---|---|---|---|
| gw | euclid_128_cgw2 | hifi3d | 0.022 s per coppia FR, 0.021 SR (mediana, un thread; composti: somma dei componenti); 2.8 h-core per 148500 coppie; preparazione 0.15 s per mesh |
| gw | euclid_128_cgw2 | facescape | 0.021 s per coppia FR, 0.020 SR (mediana, un thread; composti: somma dei componenti); 2.7 h-core per 148500 coppie; preparazione 0.19 s per mesh |
| gw | euclid_128_cgw2 | faceverse | 0.024 s per coppia FR, 0.023 SR (mediana, un thread; composti: somma dei componenti); 2.6 h-core per 148500 coppie; preparazione 0.38 s per mesh |
| gw | euclid_128_cgw2 | faceverse_neutral | 0.028 s per coppia FR, 0.028 SR (mediana, un thread; composti: somma dei componenti); 3.6 h-core per 148500 coppie; preparazione 0.40 s per mesh |
| gw | euclid_128_cgw2 | famos | 0.015 s per coppia FR, 0.014 SR (mediana, un thread; composti: somma dei componenti); 0.1 h-core per 105 coppie; preparazione 3.05 s per mesh |
| gw | euclid_256_cg | hifi3d | 0.551 s per coppia FR, 0.529 SR (mediana, un thread; composti: somma dei componenti); 52.4 h-core per 148500 coppie; preparazione 0.15 s per mesh |
| gw | euclid_256_cg | facescape | 0.599 s per coppia FR, 0.581 SR (mediana, un thread; composti: somma dei componenti); 54.5 h-core per 148500 coppie; preparazione 0.19 s per mesh |
| gw | euclid_256_cg | faceverse | 0.558 s per coppia FR, 0.547 SR (mediana, un thread; composti: somma dei componenti); 55.1 h-core per 148500 coppie; preparazione 0.38 s per mesh |
| gw | euclid_256_cg | faceverse_neutral | 0.572 s per coppia FR, 0.563 SR (mediana, un thread; composti: somma dei componenti); 57.1 h-core per 148500 coppie; preparazione 0.40 s per mesh |
| gw | euclid_256_cg | famos | 0.371 s per coppia FR, 0.342 SR (mediana, un thread; composti: somma dei componenti); 0.1 h-core per 105 coppie; preparazione 3.05 s per mesh |
| gw | euclid_256_cgw2 | hifi3d | 0.185 s per coppia FR, 0.178 SR (mediana, un thread; composti: somma dei componenti); 18.6 h-core per 148500 coppie; preparazione 0.15 s per mesh |
| gw | euclid_256_cgw2 | facescape | 0.140 s per coppia FR, 0.136 SR (mediana, un thread; composti: somma dei componenti); 14.3 h-core per 148500 coppie; preparazione 0.19 s per mesh |
| gw | euclid_256_cgw2 | faceverse | 0.229 s per coppia FR, 0.224 SR (mediana, un thread; composti: somma dei componenti); 23.2 h-core per 148500 coppie; preparazione 0.38 s per mesh |
| gw | euclid_256_cgw2 | faceverse_neutral | 0.186 s per coppia FR, 0.182 SR (mediana, un thread; composti: somma dei componenti); 19.1 h-core per 148500 coppie; preparazione 0.40 s per mesh |
| gw | euclid_256_cgw2 | famos | 0.094 s per coppia FR, 0.087 SR (mediana, un thread; composti: somma dei componenti); 0.1 h-core per 105 coppie; preparazione 3.05 s per mesh |
| dinov2 | depth | hifi3d | 29 ms per immagine (32 thread, cpu); 3.12 s per mesh coi 6 render della variante |
| dinov2 | depth | facescape | 28 ms per immagine (32 thread, cpu); 3.63 s per mesh coi 6 render della variante |
| dinov2 | depth | faceverse | 29 ms per immagine (32 thread, cpu); 5.35 s per mesh coi 6 render della variante |
| dinov2 | depth | faceverse_neutral | 29 ms per immagine (32 thread, cpu); 5.42 s per mesh coi 6 render della variante |
| dinov2 | depth | famos | 34 ms per immagine (16 thread, cpu); 1.12 s per mesh coi 6 render della variante |
| dinov2 | flat | hifi3d | 29 ms per immagine (32 thread, cpu); 3.12 s per mesh coi 6 render della variante |
| dinov2 | flat | facescape | 29 ms per immagine (32 thread, cpu); 3.63 s per mesh coi 6 render della variante |
| dinov2 | flat | faceverse | 28 ms per immagine (32 thread, cpu); 5.35 s per mesh coi 6 render della variante |
| dinov2 | flat | faceverse_neutral | 29 ms per immagine (32 thread, cpu); 5.42 s per mesh coi 6 render della variante |
| dinov2 | flat | famos | 41 ms per immagine (16 thread, cpu); 1.16 s per mesh coi 6 render della variante |
| dinov2 | smooth | hifi3d | 29 ms per immagine (32 thread, cpu); 3.10 s per mesh coi 6 render della variante |
| dinov2 | smooth | facescape | 29 ms per immagine (32 thread, cpu); 3.63 s per mesh coi 6 render della variante |
| dinov2 | smooth | faceverse | 28 ms per immagine (32 thread, cpu); 5.35 s per mesh coi 6 render della variante |
| dinov2 | smooth | faceverse_neutral | 29 ms per immagine (32 thread, cpu); 5.42 s per mesh coi 6 render della variante |
| dinov2 | smooth | famos | 35 ms per immagine (16 thread, cpu); 1.13 s per mesh coi 6 render della variante |
| clip | depth | hifi3d | 25 ms per immagine (32 thread, cpu); 3.09 s per mesh coi 6 render della variante |
| clip | depth | facescape | 24 ms per immagine (32 thread, cpu); 3.61 s per mesh coi 6 render della variante |
| clip | depth | faceverse | 24 ms per immagine (32 thread, cpu); 5.32 s per mesh coi 6 render della variante |
| clip | depth | faceverse_neutral | 24 ms per immagine (32 thread, cpu); 5.40 s per mesh coi 6 render della variante |
| clip | depth | famos | 31 ms per immagine (16 thread, cpu); 1.10 s per mesh coi 6 render della variante |
| clip | flat | hifi3d | 24 ms per immagine (32 thread, cpu); 3.09 s per mesh coi 6 render della variante |
| clip | flat | facescape | 24 ms per immagine (32 thread, cpu); 3.61 s per mesh coi 6 render della variante |
| clip | flat | faceverse | 24 ms per immagine (32 thread, cpu); 5.32 s per mesh coi 6 render della variante |
| clip | flat | faceverse_neutral | 25 ms per immagine (32 thread, cpu); 5.40 s per mesh coi 6 render della variante |
| clip | flat | famos | 41 ms per immagine (16 thread, cpu); 1.16 s per mesh coi 6 render della variante |
| clip | smooth | hifi3d | 25 ms per immagine (32 thread, cpu); 3.09 s per mesh coi 6 render della variante |
| clip | smooth | facescape | 24 ms per immagine (32 thread, cpu); 3.61 s per mesh coi 6 render della variante |
| clip | smooth | faceverse | 24 ms per immagine (32 thread, cpu); 5.32 s per mesh coi 6 render della variante |
| clip | smooth | faceverse_neutral | 24 ms per immagine (32 thread, cpu); 5.40 s per mesh coi 6 render della variante |
| clip | smooth | famos | 31 ms per immagine (16 thread, cpu); 1.10 s per mesh coi 6 render della variante |
| lpips | depth | hifi3d | 0.13 ms per coppia (Gram, 32 thread); render 8.84 s per mesh (tutte le varianti) |
| lpips | depth | facescape | 0.13 ms per coppia (Gram, 32 thread); render 10.38 s per mesh (tutte le varianti) |
| lpips | depth | faceverse | 0.14 ms per coppia (Gram, 32 thread); render 15.52 s per mesh (tutte le varianti) |
| lpips | depth | faceverse_neutral | 0.14 ms per coppia (Gram, 32 thread); render 15.75 s per mesh (tutte le varianti) |
| lpips | depth | famos | 22.15 ms per coppia (Gram, 16 thread); render 2.76 s per mesh (tutte le varianti) |
| lpips | flat | hifi3d | 0.14 ms per coppia (Gram, 32 thread); render 8.84 s per mesh (tutte le varianti) |
| lpips | flat | facescape | 0.15 ms per coppia (Gram, 32 thread); render 10.38 s per mesh (tutte le varianti) |
| lpips | flat | faceverse | 0.14 ms per coppia (Gram, 32 thread); render 15.52 s per mesh (tutte le varianti) |
| lpips | flat | faceverse_neutral | 0.14 ms per coppia (Gram, 32 thread); render 15.75 s per mesh (tutte le varianti) |
| lpips | flat | famos | 21.97 ms per coppia (Gram, 16 thread); render 2.76 s per mesh (tutte le varianti) |
| lpips | smooth | hifi3d | 0.13 ms per coppia (Gram, 32 thread); render 8.84 s per mesh (tutte le varianti) |
| lpips | smooth | facescape | 0.15 ms per coppia (Gram, 32 thread); render 10.38 s per mesh (tutte le varianti) |
| lpips | smooth | faceverse | 0.13 ms per coppia (Gram, 32 thread); render 15.52 s per mesh (tutte le varianti) |
| lpips | smooth | faceverse_neutral | 0.13 ms per coppia (Gram, 32 thread); render 15.75 s per mesh (tutte le varianti) |
| lpips | smooth | famos | 22.03 ms per coppia (Gram, 16 thread); render 2.76 s per mesh (tutte le varianti) |
| dreamsim | depth | hifi3d | 86 ms per immagine (32 thread, cpu); 3.46 s per mesh coi 6 render della variante |
| dreamsim | depth | facescape | 77 ms per immagine (32 thread, cpu); 3.92 s per mesh coi 6 render della variante |
| dreamsim | depth | faceverse | 76 ms per immagine (32 thread, cpu); 5.63 s per mesh coi 6 render della variante |
| dreamsim | depth | faceverse_neutral | 76 ms per immagine (32 thread, cpu); 5.71 s per mesh coi 6 render della variante |
| dreamsim | depth | famos | 143 ms per immagine (16 thread, cpu); 1.78 s per mesh coi 6 render della variante |
| dreamsim | flat | hifi3d | 86 ms per immagine (32 thread, cpu); 3.46 s per mesh coi 6 render della variante |
| dreamsim | flat | facescape | 77 ms per immagine (32 thread, cpu); 3.92 s per mesh coi 6 render della variante |
| dreamsim | flat | faceverse | 76 ms per immagine (32 thread, cpu); 5.63 s per mesh coi 6 render della variante |
| dreamsim | flat | faceverse_neutral | 76 ms per immagine (32 thread, cpu); 5.71 s per mesh coi 6 render della variante |
| dreamsim | flat | famos | 150 ms per immagine (16 thread, cpu); 1.82 s per mesh coi 6 render della variante |
| dreamsim | smooth | hifi3d | 86 ms per immagine (32 thread, cpu); 3.46 s per mesh coi 6 render della variante |
| dreamsim | smooth | facescape | 77 ms per immagine (32 thread, cpu); 3.92 s per mesh coi 6 render della variante |
| dreamsim | smooth | faceverse | 76 ms per immagine (32 thread, cpu); 5.63 s per mesh coi 6 render della variante |
| dreamsim | smooth | faceverse_neutral | 76 ms per immagine (32 thread, cpu); 5.71 s per mesh coi 6 render della variante |
| dreamsim | smooth | famos | 140 ms per immagine (16 thread, cpu); 1.76 s per mesh coi 6 render della variante |

