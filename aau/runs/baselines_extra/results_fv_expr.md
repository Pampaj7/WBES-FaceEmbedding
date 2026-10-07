# Risultati: FaceVerse v2 con espressioni casuali

Soggetti: 100; embedding da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/baselines_extra/fv_expr` (vfm_embed.py), ArcFace da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/arcface_render_zs/fv_expr`.

## Riconoscimento d'identita'

Baseline da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_faceverse_expr/data_736f96956a/baselines`, congiunto da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_faceverse_expr/data_736f96956a/joint_flip_topology/zs_zeroshot`. CI 95% bootstrap per soggetto, 1000 repliche (le stesse per tutte le righe).

### PRIMARIO: 5 topologie senza crop (2000 query; 1000 coppie stessa persona, 99000 diverse)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.304 [0.274, 0.336] | 0.380 [0.352, 0.411] | 0.651 [0.643, 0.661] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.251 [0.221, 0.281] | 0.332 [0.303, 0.361] | 0.593 [0.584, 0.602] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.304 [0.274, 0.338] | 0.385 [0.355, 0.416] | 0.656 [0.647, 0.666] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.255 [0.226, 0.281] | 0.333 [0.305, 0.360] | 0.596 [0.587, 0.606] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.867 [0.843, 0.889] | 0.911 [0.893, 0.927] | 0.934 [0.921, 0.946] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.750 [0.724, 0.775] | 0.807 [0.785, 0.829] | 0.863 [0.847, 0.879] | 0 |
| BFM+ICT congiunto, convenzione BFM | 0.680 [0.634, 0.723] | 0.731 [0.688, 0.770] | 0.875 [0.847, 0.900] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.740 [0.698, 0.783] | 0.775 [0.737, 0.814] | 0.882 [0.853, 0.910] | 0 |
| Rigid ICP + Chamfer | 0.918 [0.895, 0.939] | 0.935 [0.916, 0.953] | 0.986 [0.979, 0.992] | 0 |
| Rigid ICP + NICP + P2P | 0.954 [0.936, 0.971] | 0.964 [0.949, 0.978] | 0.994 [0.991, 0.997] | 0 |
| Rigid ICP + NICP + P2Tri | 0.959 [0.940, 0.975] | 0.968 [0.953, 0.981] | 0.995 [0.992, 0.998] | 0 |

#### Delta appaiati, righe di riferimento - confronti

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP (P<=0) | AUC (P<=0) |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.563 [-0.601, -0.523] (1.000) | -0.530 [-0.563, -0.495] (1.000) | -0.283 [-0.295, -0.270] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.446 [-0.484, -0.409] (1.000) | -0.426 [-0.461, -0.391] (1.000) | -0.212 [-0.229, -0.195] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.655 [-0.684, -0.623] (1.000) | -0.588 [-0.614, -0.558] (1.000) | -0.344 [-0.352, -0.335] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.436 [-0.479, -0.393] (1.000) | -0.395 [-0.435, -0.354] (1.000) | -0.231 [-0.258, -0.204] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | -0.376 [-0.421, -0.329] (1.000) | -0.351 [-0.392, -0.308] (1.000) | -0.224 [-0.249, -0.197] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.616 [-0.656, -0.574] (1.000) | -0.578 [-0.613, -0.543] (1.000) | -0.342 [-0.355, -0.328] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.499 [-0.535, -0.463] (1.000) | -0.475 [-0.510, -0.440] (1.000) | -0.271 [-0.288, -0.253] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.708 [-0.737, -0.677] (1.000) | -0.636 [-0.664, -0.606] (1.000) | -0.402 [-0.411, -0.392] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.489 [-0.528, -0.448] (1.000) | -0.443 [-0.476, -0.406] (1.000) | -0.290 [-0.315, -0.265] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | -0.429 [-0.467, -0.390] (1.000) | -0.399 [-0.431, -0.364] (1.000) | -0.282 [-0.306, -0.258] (1.000) |

#### Ablazioni, delta appaiati

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP (P<=0) | AUC (P<=0) |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | +0.001 [-0.018, +0.021] (0.481) | +0.004 [-0.010, +0.020] (0.266) | +0.005 [+0.000, +0.009] (0.014) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.004 [-0.011, +0.017] (0.337) | +0.001 [-0.011, +0.014] (0.468) | +0.003 [-0.000, +0.006] (0.026) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.053 [+0.027, +0.079] (0.000) | +0.048 [+0.022, +0.072] (0.000) | +0.059 [+0.050, +0.067] (0.000) |

### A parte: crop (coppie di topologie con crop da un lato)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.398 [0.355, 0.444] | 0.486 [0.451, 0.526] | 0.753 [0.738, 0.769] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.290 [0.245, 0.336] | 0.374 [0.332, 0.418] | 0.674 [0.655, 0.692] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.406 [0.366, 0.452] | 0.493 [0.458, 0.533] | 0.761 [0.745, 0.776] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.295 [0.254, 0.340] | 0.382 [0.342, 0.425] | 0.681 [0.663, 0.699] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.952 [0.937, 0.964] | 0.966 [0.955, 0.976] | 0.963 [0.954, 0.970] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.881 [0.867, 0.896] | 0.909 [0.897, 0.921] | 0.922 [0.910, 0.932] | 0 |
| BFM+ICT congiunto, convenzione BFM | 0.243 [0.205, 0.283] | 0.364 [0.325, 0.405] | 0.801 [0.771, 0.829] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.396 [0.336, 0.457] | 0.479 [0.421, 0.537] | 0.809 [0.781, 0.837] | 0 |
| Rigid ICP + Chamfer | 0.513 [0.455, 0.571] | 0.590 [0.540, 0.640] | 0.777 [0.734, 0.814] | 0 |
| Rigid ICP + NICP + P2P | 0.898 [0.864, 0.929] | 0.924 [0.897, 0.947] | 0.981 [0.971, 0.991] | 0 |
| Rigid ICP + NICP + P2Tri | 0.909 [0.876, 0.938] | 0.934 [0.910, 0.956] | 0.983 [0.974, 0.992] | 0 |

#### Delta appaiati, crop

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP (P<=0) | AUC (P<=0) |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.554 [-0.600, -0.506] (1.000) | -0.480 [-0.518, -0.437] (1.000) | -0.210 [-0.227, -0.192] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.483 [-0.527, -0.434] (1.000) | -0.423 [-0.460, -0.381] (1.000) | -0.169 [-0.187, -0.151] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.511 [-0.552, -0.467] (1.000) | -0.447 [-0.484, -0.408] (1.000) | -0.230 [-0.244, -0.216] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.002 [-0.057, +0.067] (0.492) | +0.008 [-0.049, +0.068] (0.399) | -0.056 [-0.082, -0.027] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.155 [+0.101, +0.214] (0.000) | +0.122 [+0.074, +0.177] (0.000) | -0.048 [-0.075, -0.017] (0.999) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.662 [-0.713, -0.613] (1.000) | -0.592 [-0.638, -0.548] (1.000) | -0.289 [-0.310, -0.269] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, 3 viste | -0.591 [-0.639, -0.542] (1.000) | -0.535 [-0.580, -0.491] (1.000) | -0.248 [-0.269, -0.228] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.619 [-0.668, -0.572] (1.000) | -0.559 [-0.598, -0.517] (1.000) | -0.309 [-0.325, -0.293] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | -0.106 [-0.174, -0.034] (0.999) | -0.104 [-0.164, -0.041] (0.999) | -0.135 [-0.165, -0.105] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | BFM+ICT congiunto, convenzione BFM | +0.047 [-0.009, +0.101] (0.053) | +0.010 [-0.040, +0.061] (0.356) | -0.127 [-0.155, -0.100] (1.000) |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | +0.008 [-0.020, +0.039] (0.293) | +0.007 [-0.014, +0.029] (0.245) | +0.008 [+0.002, +0.013] (0.003) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.005 [-0.027, +0.035] (0.379) | +0.007 [-0.019, +0.032] (0.267) | +0.007 [+0.001, +0.013] (0.011) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.108 [+0.062, +0.153] (0.000) | +0.112 [+0.072, +0.155] (0.000) | +0.079 [+0.063, +0.098] (0.000) |

### Controllo

- riproduzione di `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/arcface_render_zs/fv_expr/recognition.csv` (stesse repliche): max |diff| su punto e CI di rank-1, mAP, AUC = 1.11e-16 su 14 righe (arcface_normals_3v, arcface_shaded_3v, chamfer, joint@bfm, nicp_p2p, nicp_p2tri, rigid_icp_chamfer)
