# Risultati: HIFI3D, neutre

Soggetti: 100; embedding da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/baselines_extra/hifi3d` (vfm_embed.py), ArcFace da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/arcface_render_zs/hifi3d`.

## Riconoscimento d'identita'

Baseline da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/baselines`, congiunto da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/joint_frame-xmymz_flip_ranking/zs_zeroshot`. CI 95% bootstrap per soggetto, 1000 repliche (le stesse per tutte le righe).

### PRIMARIO: 5 topologie senza crop (2000 query; 1000 coppie stessa persona, 99000 diverse)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.597 [0.582, 0.614] | 0.642 [0.628, 0.658] | 0.708 [0.702, 0.715] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.510 [0.485, 0.534] | 0.571 [0.551, 0.592] | 0.668 [0.661, 0.675] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.573 [0.556, 0.590] | 0.630 [0.615, 0.646] | 0.716 [0.708, 0.726] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.516 [0.496, 0.537] | 0.579 [0.561, 0.598] | 0.677 [0.671, 0.684] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.998 [0.995, 1.000] | 0.999 [0.998, 1.000] | 0.997 [0.996, 0.998] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.984 [0.974, 0.992] | 0.991 [0.985, 0.996] | 0.989 [0.985, 0.993] | 0 |
| BFM+ICT congiunto, convenzione BFM | 0.397 [0.363, 0.434] | 0.517 [0.489, 0.549] | 0.755 [0.740, 0.770] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.477 [0.445, 0.513] | 0.564 [0.534, 0.594] | 0.782 [0.767, 0.797] | 0 |
| Rigid ICP + Chamfer | 0.996 [0.991, 0.999] | 0.997 [0.995, 0.999] | 0.999 [0.999, 1.000] | 0 |
| Rigid ICP + NICP + P2P | 0.988 [0.979, 0.996] | 0.989 [0.980, 0.997] | 0.999 [0.999, 1.000] | 1965 |
| Rigid ICP + NICP + P2Tri | 0.990 [0.982, 0.998] | 0.990 [0.982, 0.998] | 1.000 [1.000, 1.000] | 1965 |

#### Delta appaiati, righe di riferimento - confronti

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP (P<=0) | AUC (P<=0) |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.401 [-0.417, -0.383] (1.000) | -0.357 [-0.371, -0.341] (1.000) | -0.289 [-0.295, -0.283] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.387 [-0.403, -0.368] (1.000) | -0.349 [-0.364, -0.332] (1.000) | -0.281 [-0.288, -0.274] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.393 [-0.411, -0.373] (1.000) | -0.348 [-0.365, -0.330] (1.000) | -0.292 [-0.298, -0.285] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.119 [+0.084, +0.155] (0.000) | +0.078 [+0.044, +0.110] (0.000) | -0.074 [-0.088, -0.060] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.200 [+0.167, +0.235] (0.000) | +0.125 [+0.095, +0.156] (0.000) | -0.047 [-0.060, -0.034] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.488 [-0.513, -0.464] (1.000) | -0.428 [-0.448, -0.406] (1.000) | -0.329 [-0.336, -0.322] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.474 [-0.500, -0.448] (1.000) | -0.420 [-0.440, -0.398] (1.000) | -0.321 [-0.328, -0.313] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.480 [-0.506, -0.456] (1.000) | -0.419 [-0.441, -0.398] (1.000) | -0.332 [-0.339, -0.325] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.033 [-0.004, +0.068] (0.044) | +0.007 [-0.025, +0.038] (0.342) | -0.114 [-0.128, -0.100] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.113 [+0.079, +0.147] (0.000) | +0.054 [+0.024, +0.085] (0.001) | -0.087 [-0.101, -0.072] (1.000) |

#### Ablazioni, delta appaiati

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP (P<=0) | AUC (P<=0) |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | -0.024 [-0.046, -0.004] (0.988) | -0.012 [-0.029, +0.004] (0.937) | +0.008 [+0.004, +0.013] (0.000) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.006 [-0.011, +0.024] (0.242) | +0.008 [-0.006, +0.022] (0.130) | +0.009 [+0.005, +0.013] (0.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.087 [+0.059, +0.114] (0.000) | +0.071 [+0.048, +0.094] (0.000) | +0.040 [+0.033, +0.046] (0.000) |

### A parte: crop (coppie di topologie con crop da un lato)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.773 [0.760, 0.787] | 0.803 [0.794, 0.813] | 0.822 [0.817, 0.827] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.666 [0.645, 0.686] | 0.719 [0.703, 0.736] | 0.772 [0.765, 0.779] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.761 [0.743, 0.778] | 0.799 [0.786, 0.811] | 0.831 [0.825, 0.838] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.676 [0.653, 0.699] | 0.730 [0.714, 0.747] | 0.785 [0.779, 0.793] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.997 [0.994, 1.000] | 0.999 [0.997, 1.000] | 0.999 [0.998, 0.999] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.990 [0.984, 0.996] | 0.994 [0.991, 0.998] | 0.994 [0.992, 0.996] | 0 |
| BFM+ICT congiunto, convenzione BFM | 0.085 [0.057, 0.117] | 0.201 [0.171, 0.233] | 0.716 [0.699, 0.734] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.472 [0.445, 0.499] | 0.536 [0.510, 0.564] | 0.699 [0.684, 0.715] | 0 |
| Rigid ICP + Chamfer | 0.284 [0.237, 0.338] | 0.423 [0.381, 0.470] | 0.888 [0.865, 0.911] | 0 |
| Rigid ICP + NICP + P2P | 0.752 [0.714, 0.788] | 0.834 [0.810, 0.858] | 0.995 [0.993, 0.997] | 1965 |
| Rigid ICP + NICP + P2Tri | 0.777 [0.740, 0.810] | 0.854 [0.829, 0.877] | 0.997 [0.995, 0.998] | 1965 |

#### Delta appaiati, crop

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP (P<=0) | AUC (P<=0) |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.224 [-0.237, -0.210] (1.000) | -0.196 [-0.205, -0.185] (1.000) | -0.177 [-0.181, -0.172] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.217 [-0.231, -0.202] (1.000) | -0.192 [-0.202, -0.180] (1.000) | -0.172 [-0.177, -0.167] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.004 [-0.043, +0.036] (0.575) | -0.051 [-0.077, -0.024] (1.000) | -0.175 [-0.180, -0.170] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.301 [+0.276, +0.327] (0.000) | +0.266 [+0.241, +0.292] (0.000) | +0.123 [+0.108, +0.136] (0.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.688 [+0.659, +0.717] (0.000) | +0.601 [+0.572, +0.630] (0.000) | +0.106 [+0.089, +0.123] (0.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.331 [-0.352, -0.310] (1.000) | -0.279 [-0.295, -0.263] (1.000) | -0.227 [-0.234, -0.220] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.324 [-0.346, -0.302] (1.000) | -0.275 [-0.292, -0.258] (1.000) | -0.222 [-0.229, -0.215] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.111 [-0.150, -0.071] (1.000) | -0.134 [-0.163, -0.105] (1.000) | -0.225 [-0.232, -0.218] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.194 [+0.165, +0.223] (0.000) | +0.183 [+0.156, +0.211] (0.000) | +0.073 [+0.058, +0.086] (0.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.581 [+0.544, +0.616] (0.000) | +0.518 [+0.481, +0.552] (0.000) | +0.056 [+0.039, +0.073] (0.000) |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | -0.012 [-0.031, +0.007] (0.904) | -0.004 [-0.017, +0.009] (0.706) | +0.009 [+0.005, +0.012] (0.000) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.010 [-0.011, +0.033] (0.209) | +0.011 [-0.004, +0.028] (0.090) | +0.013 [+0.009, +0.018] (0.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.107 [+0.086, +0.130] (0.000) | +0.083 [+0.066, +0.101] (0.000) | +0.050 [+0.044, +0.057] (0.000) |

### Controllo

- riproduzione di `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/arcface_render_zs/hifi3d/recognition.csv` (stesse repliche): max |diff| su punto e CI di rank-1, mAP, AUC = 1.11e-16 su 12 righe (arcface_shaded_3v, chamfer, joint@bfm, nicp_p2p, nicp_p2tri, rigid_icp_chamfer)

## Ranking: Spearman con la GT

Coppie: `nocrop_cross_topology` = 20 coppie ordinate di topologie x 4950 = 99000 righe; `original_to_original` = 4950.

### Spearman con la GT `maxabs`, CI 95% bootstrap per soggetto (1000 repliche)

| metodo | nocrop_cross_topology | original_to_original | distanze NaN (cross) |
| --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.195 [0.143, 0.254] | 0.457 [0.362, 0.544] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.107 [0.071, 0.144] | 0.335 [0.230, 0.435] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.196 [0.136, 0.268] | 0.414 [0.305, 0.508] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.130 [0.090, 0.172] | 0.389 [0.287, 0.494] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.242 [0.156, 0.332] | 0.258 [0.172, 0.355] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.245 [0.172, 0.322] | 0.285 [0.204, 0.372] | 0 |
| BFM+ICT congiunto, convenzione BFM | 0.245 [0.197, 0.294] | 0.642 [0.580, 0.703] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.325 [0.278, 0.371] | 0.876 [0.845, 0.902] | 0 |
| Rigid ICP + Chamfer | 0.355 [0.274, 0.433] | 0.401 [0.310, 0.484] | 0 |
| Rigid ICP + NICP + P2P | 0.355 [0.266, 0.438] | 0.409 [0.315, 0.502] | 776 |
| Rigid ICP + NICP + P2Tri | 0.389 [0.305, 0.468] | 0.431 [0.340, 0.522] | 776 |

Delta appaiati A - B dello Spearman (stesse repliche), [CI 95%] (P<=0), righe finite per entrambi:

| A | B | nocrop_cross_topology | original_to_original |
| --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.130 [-0.192, -0.066] (1.000) | -0.419 [-0.513, -0.321] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.194 [-0.285, -0.104] (1.000) | +0.026 [-0.094, +0.142] (0.375) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.046 [-0.115, +0.019] (0.925) | +0.199 [+0.093, +0.292] (0.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | -0.049 [-0.109, +0.006] (0.957) | -0.185 [-0.278, -0.097] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.089 [+0.052, +0.125] (0.000) | +0.122 [+0.034, +0.213] (0.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.219 [-0.265, -0.166] (1.000) | -0.541 [-0.637, -0.437] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.283 [-0.369, -0.199] (1.000) | -0.096 [-0.218, +0.032] (0.935) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.135 [-0.209, -0.064] (1.000) | +0.077 [-0.027, +0.185] (0.073) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | -0.138 [-0.188, -0.090] (1.000) | -0.307 [-0.400, -0.209] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | -0.089 [-0.125, -0.052] (1.000) | -0.122 [-0.213, -0.034] (1.000) |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | +0.001 [-0.022, +0.023] (0.469) | -0.043 [-0.093, +0.003] (0.962) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.023 [+0.010, +0.037] (0.000) | +0.054 [+0.010, +0.092] (0.007) |

### Spearman con la GT `coef`, CI 95% bootstrap per soggetto (1000 repliche)

| metodo | nocrop_cross_topology | original_to_original | distanze NaN (cross) |
| --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.054 [0.015, 0.096] | 0.131 [0.029, 0.235] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.028 [-0.006, 0.064] | 0.116 [0.024, 0.205] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.058 [0.008, 0.111] | 0.147 [0.047, 0.252] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.039 [-0.004, 0.084] | 0.152 [0.049, 0.249] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.192 [0.123, 0.254] | 0.200 [0.123, 0.283] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.180 [0.112, 0.241] | 0.212 [0.133, 0.288] | 0 |
| BFM+ICT congiunto, convenzione BFM | 0.043 [0.009, 0.075] | 0.105 [0.027, 0.188] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.069 [0.030, 0.108] | 0.162 [0.067, 0.252] | 0 |
| Rigid ICP + Chamfer | 0.155 [0.069, 0.241] | 0.165 [0.069, 0.257] | 0 |
| Rigid ICP + NICP + P2P | 0.128 [0.046, 0.204] | 0.131 [0.045, 0.223] | 776 |
| Rigid ICP + NICP + P2Tri | 0.139 [0.051, 0.213] | 0.137 [0.052, 0.217] | 776 |

Delta appaiati A - B dello Spearman (stesse repliche), [CI 95%] (P<=0), righe finite per entrambi:

| A | B | nocrop_cross_topology | original_to_original |
| --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.015 [-0.061, +0.035] (0.718) | -0.030 [-0.129, +0.076] (0.698) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.085 [-0.156, -0.000] (0.975) | -0.006 [-0.116, +0.104] (0.545) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.138 [-0.202, -0.076] (1.000) | -0.069 [-0.173, +0.032] (0.919) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.011 [-0.034, +0.055] (0.324) | +0.026 [-0.071, +0.128] (0.303) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.025 [-0.010, +0.062] (0.099) | +0.015 [-0.074, +0.100] (0.364) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.040 [-0.085, +0.010] (0.946) | -0.046 [-0.143, +0.071] (0.788) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.111 [-0.183, -0.030] (0.998) | -0.021 [-0.111, +0.076] (0.657) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.163 [-0.229, -0.093] (1.000) | -0.084 [-0.174, +0.006] (0.965) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | -0.014 [-0.057, +0.034] (0.738) | +0.011 [-0.081, +0.119] (0.410) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | -0.025 [-0.062, +0.010] (0.901) | -0.015 [-0.100, +0.074] (0.636) |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | +0.005 [-0.015, +0.023] (0.317) | +0.016 [-0.027, +0.057] (0.245) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.011 [-0.008, +0.028] (0.109) | +0.036 [-0.010, +0.079] (0.061) |
