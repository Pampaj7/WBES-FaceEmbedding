# Risultati: BFM held-out (in dominio, i 100 soggetti di WS1)

Soggetti: 100; embedding da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/baselines_extra/bfm_heldout` (vfm_embed.py), ArcFace da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/baselines_extra/bfm_heldout`.


## Ranking: Spearman con la GT

Coppie: `nocrop_cross_topology` = 20 coppie ordinate di topologie x 4950 = 99000 righe; `original_to_original` = 4950.

### Spearman con la GT `paper`, CI 95% bootstrap per soggetto (1000 repliche)

| metodo | nocrop_cross_topology | original_to_original | distanze NaN (cross) |
| --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | 0.129 [0.084, 0.177] | 0.332 [0.240, 0.442] | 0 |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | 0.114 [0.077, 0.156] | 0.370 [0.265, 0.473] | 0 |
| CLIP ViT-L/14, normal map, render intero, 3 viste | 0.143 [0.097, 0.187] | 0.344 [0.237, 0.443] | 0 |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | 0.110 [0.074, 0.153] | 0.397 [0.297, 0.486] | 0 |
| ArcFace, normal map, 3 viste (stessi render) | 0.223 [0.163, 0.283] | 0.280 [0.205, 0.357] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.220 [0.162, 0.277] | 0.301 [0.228, 0.377] | 0 |
| Chamfer faceBench (WS1) | 0.469 [0.400, 0.534] | 0.644 [0.571, 0.712] | 0 |
| LPIPS AlexNet, ombreggiato (WS1) | 0.127 [0.099, 0.157] | 0.590 [0.503, 0.662] | 0 |
| ArcFace, ombreggiato, crop fisso (WS1) | 0.218 [0.160, 0.272] | 0.293 [0.220, 0.364] | 0 |
| CLIP ViT-B/32 laion2b, ombreggiato intero (WS1) | 0.087 [0.057, 0.117] | 0.358 [0.261, 0.456] | 0 |
| DINOv2 ViT-S/14, ombreggiato intero (WS1) | 0.058 [0.030, 0.089] | 0.378 [0.278, 0.478] | 0 |
| Modello NeurIPS (BFM, v1) | 0.789 [0.737, 0.830] | 0.830 [0.779, 0.872] | 0 |

Delta appaiati A - B dello Spearman (stesse repliche), [CI 95%] (P<=0), righe finite per entrambi:

| A | B | nocrop_cross_topology | original_to_original |
| --- | --- | --- | --- |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Chamfer faceBench (WS1) | -0.340 [-0.413, -0.258] (1.000) | -0.312 [-0.429, -0.202] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | LPIPS AlexNet, ombreggiato (WS1) | +0.002 [-0.045, +0.051] (0.486) | -0.258 [-0.368, -0.143] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, crop fisso (WS1) | -0.089 [-0.141, -0.042] (1.000) | +0.039 [-0.054, +0.134] (0.209) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.094 [-0.152, -0.033] (0.999) | +0.052 [-0.056, +0.163] (0.159) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | Modello NeurIPS (BFM, v1) | -0.659 [-0.718, -0.588] (1.000) | -0.498 [-0.606, -0.394] (1.000) |
| CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | +0.015 [-0.017, +0.051] (0.207) | -0.038 [-0.129, +0.057] (0.775) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Chamfer faceBench (WS1) | -0.355 [-0.422, -0.280] (1.000) | -0.273 [-0.383, -0.174] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | LPIPS AlexNet, ombreggiato (WS1) | -0.013 [-0.053, +0.026] (0.746) | -0.220 [-0.329, -0.124] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, ombreggiato, crop fisso (WS1) | -0.104 [-0.157, -0.057] (1.000) | +0.077 [-0.018, +0.164] (0.057) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | ArcFace, normal map, 3 viste (stessi render) | -0.109 [-0.161, -0.055] (1.000) | +0.090 [-0.017, +0.197] (0.054) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | Modello NeurIPS (BFM, v1) | -0.675 [-0.728, -0.610] (1.000) | -0.460 [-0.567, -0.364] (1.000) |
| DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | -0.015 [-0.051, +0.017] (0.793) | +0.038 [-0.057, +0.129] (0.225) |
| CLIP ViT-L/14, normal map, render intero, 3 viste | CLIP ViT-L/14, normal map, crop, 3 viste (riferimento) | +0.014 [-0.002, +0.029] (0.048) | +0.012 [-0.036, +0.060] (0.321) |
| DINOv2 ViT-B/14, normal map, render intero, 3 viste | DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento) | -0.003 [-0.016, +0.009] (0.733) | +0.027 [-0.027, +0.078] (0.151) |
