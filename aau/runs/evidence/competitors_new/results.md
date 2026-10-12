# Gromov-Wasserstein e feature percettive su render: Spearman con GT FR / SR

Generato da `aau/competitors/gw_percettivi/gp_eval.py spearman` (non modificare a mano). Righe, semi e repliche di all_cross (`bp_paired_e2.all_cross_rows`) e di FaMoS TEST (`fact_paired.famos_frame`), 1000 repliche per soggetto, IC al 95% percentile. Colonna mm (FR) contro GT FR, colonna a CS_ref (SR) contro GT SR; le altre combinazioni sono in `spearman.csv`.

GW: configurazione `euclid|256|cg` scelta sul pilota (punteggio 0.1833; migliore `euclid|256|cg` 0.1833; regola: fra le configurazioni con mediana dei secondi per soluzione <= 1.0: massimo della media sulle 4 viste di (rho(fr, GT FR) + rho(sr, GT SR)) / 2 sulle coppie del pilota; entro 0.01 dal migliore la mediana dei secondi per soluzione piu' bassa).

## senza crop

| metodo | GT | hifi3d | facescape | faceverse | faceverse_neutral | famos |
|---|---|---|---|---|---|---|
| Gromov-Wasserstein (POT) | FR | 0.435 [0.351, 0.510] | 0.163 [0.137, 0.188] | 0.112 [0.066, 0.158] | 0.141 [0.092, 0.189] | 0.693 [0.344, 0.879] |
| Gromov-Wasserstein (POT) | SR | 0.225 [0.186, 0.261] | 0.196 [0.165, 0.225] | 0.091 [0.058, 0.128] | 0.121 [0.084, 0.156] | 0.337 [-0.018, 0.576] |
| DINOv2 ViT-S/14 su render | FR | 0.061 [0.035, 0.087] | 0.049 [0.029, 0.070] | 0.026 [0.008, 0.043] | 0.036 [0.018, 0.053] | 0.329 [-0.102, 0.677] |
| DINOv2 ViT-S/14 su render | SR | 0.074 [0.046, 0.100] | 0.049 [0.030, 0.071] | 0.022 [0.004, 0.038] | 0.035 [0.019, 0.050] | 0.469 [0.054, 0.726] |
| CLIP ViT-B/32 su render | FR | 0.135 [0.086, 0.184] | 0.103 [0.063, 0.147] | 0.034 [0.007, 0.059] | 0.043 [0.017, 0.069] | 0.474 [0.064, 0.739] |
| CLIP ViT-B/32 su render | SR | 0.152 [0.113, 0.188] | 0.094 [0.056, 0.134] | 0.034 [0.010, 0.057] | 0.055 [0.027, 0.084] | 0.653 [0.306, 0.831] |
| LPIPS alex su render | FR | 0.355 [0.315, 0.389] | 0.180 [0.142, 0.214] | 0.090 [0.065, 0.116] | 0.136 [0.105, 0.166] | 0.676 [0.282, 0.890] |
| LPIPS alex su render | SR | 0.258 [0.227, 0.290] | 0.180 [0.145, 0.211] | 0.087 [0.061, 0.114] | 0.130 [0.100, 0.160] | 0.482 [0.072, 0.759] |
| DreamSim ensemble su render | FR | 0.153 [0.105, 0.201] | 0.122 [0.092, 0.154] | 0.048 [0.026, 0.071] | 0.082 [0.053, 0.109] | 0.546 [0.171, 0.804] |
| DreamSim ensemble su render | SR | 0.163 [0.125, 0.200] | 0.121 [0.091, 0.153] | 0.047 [0.024, 0.069] | 0.091 [0.062, 0.118] | 0.673 [0.378, 0.826] |

## all_cross

| metodo | GT | hifi3d | facescape | faceverse | faceverse_neutral |
|---|---|---|---|---|---|
| Gromov-Wasserstein (POT) | FR | 0.341 [0.270, 0.408] | 0.086 [0.069, 0.103] | 0.092 [0.056, 0.129] | 0.113 [0.076, 0.149] |
| Gromov-Wasserstein (POT) | SR | 0.183 [0.146, 0.220] | 0.163 [0.131, 0.191] | 0.089 [0.060, 0.118] | 0.112 [0.079, 0.144] |
| DINOv2 ViT-S/14 su render | FR | 0.079 [0.047, 0.110] | 0.059 [0.034, 0.085] | 0.035 [0.019, 0.051] | 0.047 [0.028, 0.064] |
| DINOv2 ViT-S/14 su render | SR | 0.082 [0.051, 0.110] | 0.062 [0.039, 0.088] | 0.035 [0.018, 0.051] | 0.049 [0.032, 0.065] |
| CLIP ViT-B/32 su render | FR | 0.153 [0.099, 0.209] | 0.123 [0.078, 0.170] | 0.046 [0.019, 0.072] | 0.056 [0.028, 0.085] |
| CLIP ViT-B/32 su render | SR | 0.173 [0.133, 0.212] | 0.114 [0.071, 0.159] | 0.051 [0.024, 0.076] | 0.071 [0.041, 0.102] |
| LPIPS alex su render | FR | 0.318 [0.273, 0.357] | 0.105 [0.079, 0.130] | 0.100 [0.073, 0.127] | 0.144 [0.110, 0.174] |
| LPIPS alex su render | SR | 0.157 [0.133, 0.179] | 0.148 [0.113, 0.179] | 0.095 [0.069, 0.123] | 0.135 [0.103, 0.165] |
| DreamSim ensemble su render | FR | 0.172 [0.119, 0.224] | 0.144 [0.107, 0.184] | 0.065 [0.040, 0.089] | 0.100 [0.067, 0.129] |
| DreamSim ensemble su render | SR | 0.187 [0.147, 0.226] | 0.152 [0.116, 0.192] | 0.066 [0.039, 0.088] | 0.113 [0.080, 0.144] |

## righe col crop

| metodo | GT | hifi3d | facescape | faceverse | faceverse_neutral |
|---|---|---|---|---|---|
| Gromov-Wasserstein (POT) | FR | 0.181 [0.138, 0.224] | 0.075 [0.034, 0.118] | 0.071 [0.045, 0.098] | 0.085 [0.059, 0.111] |
| Gromov-Wasserstein (POT) | SR | 0.208 [0.164, 0.253] | 0.207 [0.164, 0.250] | 0.099 [0.066, 0.132] | 0.121 [0.087, 0.152] |
| DINOv2 ViT-S/14 su render | FR | 0.144 [0.083, 0.207] | 0.114 [0.064, 0.166] | 0.065 [0.036, 0.093] | 0.082 [0.048, 0.116] |
| DINOv2 ViT-S/14 su render | SR | 0.123 [0.074, 0.174] | 0.123 [0.078, 0.172] | 0.073 [0.045, 0.099] | 0.097 [0.067, 0.125] |
| CLIP ViT-B/32 su render | FR | 0.211 [0.140, 0.284] | 0.177 [0.111, 0.240] | 0.075 [0.035, 0.115] | 0.090 [0.047, 0.133] |
| CLIP ViT-B/32 su render | SR | 0.237 [0.186, 0.287] | 0.169 [0.109, 0.231] | 0.091 [0.050, 0.132] | 0.112 [0.070, 0.155] |
| LPIPS alex su render | FR | 0.336 [0.287, 0.379] | 0.119 [0.075, 0.166] | 0.131 [0.092, 0.172] | 0.180 [0.139, 0.222] |
| LPIPS alex su render | SR | 0.249 [0.210, 0.291] | 0.249 [0.186, 0.309] | 0.132 [0.091, 0.173] | 0.186 [0.144, 0.226] |
| DreamSim ensemble su render | FR | 0.235 [0.163, 0.307] | 0.208 [0.150, 0.268] | 0.104 [0.065, 0.139] | 0.145 [0.098, 0.186] |
| DreamSim ensemble su render | SR | 0.267 [0.214, 0.317] | 0.244 [0.187, 0.304] | 0.109 [0.069, 0.144] | 0.168 [0.121, 0.209] |

## Costi

| metodo | vista | costo |
|---|---|---|
| gw | hifi3d | 0.551 s per soluzione FR, 0.529 SR (mediana, un thread); 52.4 h-core per 148500 coppie; preparazione 0.15 s per mesh (lettura+taglia, fps+pesi) |
| gw | facescape | 0.599 s per soluzione FR, 0.581 SR (mediana, un thread); 54.5 h-core per 148500 coppie; preparazione 0.19 s per mesh (lettura+taglia, fps+pesi) |
| gw | faceverse | 0.558 s per soluzione FR, 0.547 SR (mediana, un thread); 55.1 h-core per 148500 coppie; preparazione 0.38 s per mesh (lettura+taglia, fps+pesi) |
| gw | faceverse_neutral | 0.572 s per soluzione FR, 0.563 SR (mediana, un thread); 57.1 h-core per 148500 coppie; preparazione 0.40 s per mesh (lettura+taglia, fps+pesi) |
| gw | famos | 0.371 s per soluzione FR, 0.342 SR (mediana, un thread); 0.1 h-core per 105 coppie; preparazione 3.05 s per mesh (lettura+taglia, fps+pesi) |
| dinov2 | hifi3d | 29 ms per immagine (32 thread, cpu); 3.81 s per mesh con i 6 render |
| dinov2 | facescape | 29 ms per immagine (32 thread, cpu); 4.44 s per mesh con i 6 render |
| dinov2 | faceverse | 29 ms per immagine (32 thread, cpu); 5.78 s per mesh con i 6 render |
| dinov2 | faceverse_neutral | 29 ms per immagine (32 thread, cpu); 5.92 s per mesh con i 6 render |
| dinov2 | famos | 40 ms per immagine (16 thread, cpu); 2.06 s per mesh con i 6 render |
| clip | hifi3d | 25 ms per immagine (32 thread, cpu); 3.78 s per mesh con i 6 render |
| clip | facescape | 24 ms per immagine (32 thread, cpu); 4.41 s per mesh con i 6 render |
| clip | faceverse | 25 ms per immagine (32 thread, cpu); 5.75 s per mesh con i 6 render |
| clip | faceverse_neutral | 25 ms per immagine (32 thread, cpu); 5.90 s per mesh con i 6 render |
| clip | famos | 35 ms per immagine (16 thread, cpu); 2.03 s per mesh con i 6 render |
| lpips | hifi3d | 0.13 ms per coppia (Gram, 32 thread); render 3.63 s per mesh (6 render) |
| lpips | facescape | 0.13 ms per coppia (Gram, 32 thread); render 4.26 s per mesh (6 render) |
| lpips | faceverse | 0.13 ms per coppia (Gram, 32 thread); render 5.61 s per mesh (6 render) |
| lpips | faceverse_neutral | 0.14 ms per coppia (Gram, 32 thread); render 5.75 s per mesh (6 render) |
| lpips | famos | 18.59 ms per coppia (Gram, 16 thread); render 1.82 s per mesh (6 render) |
| dreamsim | hifi3d | 112 ms per immagine (32 thread, cpu); 4.30 s per mesh con i 6 render |
| dreamsim | facescape | 117 ms per immagine (32 thread, cpu); 4.97 s per mesh con i 6 render |
| dreamsim | faceverse | 126 ms per immagine (32 thread, cpu); 6.36 s per mesh con i 6 render |
| dreamsim | faceverse_neutral | 118 ms per immagine (32 thread, cpu); 6.45 s per mesh con i 6 render |
| dreamsim | famos | 180 ms per immagine (16 thread, cpu); 2.90 s per mesh con i 6 render |

