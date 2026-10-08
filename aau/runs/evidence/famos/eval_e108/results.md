# FaMoS, persone mai viste: e108 e Chamfer

15 persone di TEST, 1284 mesh (patch NoW, 5215 triangoli). CI 95% bootstrap sui soggetti (1000 repliche). Con 15 persone: prova della pipeline, non risultato.

## Riconoscimento (galleria di 15, una mesh per persona)

| blocco | query | metodo | rank-1 | AUC verifica |
| --- | --- | --- | --- | --- |
| scan peak -> scan | 418 | e108 | 0.514 [0.412, 0.613] | 0.760 [0.708, 0.813] |
| scan peak -> scan | 418 | Chamfer intera | 0.495 [0.406, 0.579] | 0.727 [0.687, 0.771] |
| scan peak -> scan | 418 | Chamfer regione stabile | 0.596 [0.519, 0.689] | 0.785 [0.740, 0.834] |
| scan peak -> scan | | e108 - Chamfer intera | +0.019 [-0.043, +0.079] | +0.033 [+0.006, +0.057] |
| scan peak -> scan | | e108 - Chamfer regione stabile | -0.081 [-0.165, +0.002] | -0.025 [-0.080, +0.032] |
| scan nearneutral -> scan | 418 | e108 | 0.995 [0.988, 1.000] | 0.999 [0.997, 1.000] |
| scan nearneutral -> scan | 418 | Chamfer intera | 0.995 [0.988, 1.000] | 1.000 [0.999, 1.000] |
| scan nearneutral -> scan | 418 | Chamfer regione stabile | 0.993 [0.986, 1.000] | 0.999 [0.998, 1.000] |
| scan nearneutral -> scan | | e108 - Chamfer intera | +0.000 [-0.007, +0.007] | -0.000 [-0.002, +0.000] |
| scan nearneutral -> scan | | e108 - Chamfer regione stabile | +0.002 [-0.005, +0.010] | +0.000 [-0.001, +0.001] |
| reg peak -> reg | 418 | e108 | 0.488 [0.396, 0.587] | 0.768 [0.709, 0.824] |
| reg peak -> reg | 418 | Chamfer intera | 0.505 [0.423, 0.579] | 0.713 [0.667, 0.764] |
| reg peak -> reg | 418 | Chamfer regione stabile | 0.605 [0.529, 0.683] | 0.744 [0.693, 0.803] |
| reg peak -> reg | | e108 - Chamfer intera | -0.017 [-0.086, +0.057] | +0.055 [+0.022, +0.089] |
| reg peak -> reg | | e108 - Chamfer regione stabile | -0.117 [-0.194, -0.040] | +0.024 [-0.038, +0.079] |
| scan peak -> reg | 418 | e108 | 0.347 [0.217, 0.488] | 0.700 [0.645, 0.761] |
| scan peak -> reg | 418 | Chamfer intera | 0.455 [0.373, 0.540] | 0.697 [0.645, 0.752] |
| scan peak -> reg | 418 | Chamfer regione stabile | 0.376 [0.249, 0.508] | 0.702 [0.645, 0.764] |
| scan peak -> reg | | e108 - Chamfer intera | -0.108 [-0.242, +0.019] | +0.003 [-0.026, +0.036] |
| scan peak -> reg | | e108 - Chamfer regione stabile | -0.029 [-0.186, +0.105] | -0.002 [-0.051, +0.044] |
| scan nearneutral -> reg | 418 | e108 | 0.567 [0.360, 0.774] | 0.887 [0.808, 0.955] |
| scan nearneutral -> reg | 418 | Chamfer intera | 0.780 [0.638, 0.909] | 0.889 [0.826, 0.948] |
| scan nearneutral -> reg | 418 | Chamfer regione stabile | 0.770 [0.617, 0.904] | 0.894 [0.809, 0.969] |
| scan nearneutral -> reg | | e108 - Chamfer intera | -0.213 [-0.464, +0.041] | -0.002 [-0.064, +0.059] |
| scan nearneutral -> reg | | e108 - Chamfer regione stabile | -0.203 [-0.398, -0.038] | -0.007 [-0.089, +0.073] |
| reg peak -> scan | 418 | e108 | 0.321 [0.187, 0.470] | 0.715 [0.651, 0.777] |
| reg peak -> scan | 418 | Chamfer intera | 0.208 [0.093, 0.354] | 0.657 [0.619, 0.703] |
| reg peak -> scan | 418 | Chamfer regione stabile | 0.371 [0.223, 0.540] | 0.689 [0.625, 0.758] |
| reg peak -> scan | | e108 - Chamfer intera | +0.112 [-0.070, +0.261] | +0.058 [+0.025, +0.089] |
| reg peak -> scan | | e108 - Chamfer regione stabile | -0.050 [-0.212, +0.108] | +0.026 [-0.043, +0.093] |

## Distanza graduata: Spearman con la GT unificata (mm, neutre registrate)

| blocco | coppie | metodo | Spearman | delta modello - metodo |
| --- | --- | --- | --- | --- |
| scan gallery -> scan | 105 | e108 | 0.370 [-0.016, 0.644] | - |
| scan gallery -> scan | 105 | Chamfer intera | 0.498 [0.135, 0.747] | -0.128 [-0.376, +0.056] (P<=0 0.919) |
| scan gallery -> scan | 105 | Chamfer regione stabile | 0.263 [-0.023, 0.548] | +0.106 [-0.308, +0.418] (P<=0 0.284) |
| reg gallery -> reg | 105 | e108 | 0.446 [0.097, 0.695] | - |
| reg gallery -> reg | 105 | Chamfer intera | 0.490 [0.120, 0.723] | -0.044 [-0.206, +0.166] (P<=0 0.748) |
| reg gallery -> reg | 105 | Chamfer regione stabile | 0.436 [0.079, 0.744] | +0.010 [-0.396, +0.436] (P<=0 0.502) |
| scan gallery -> reg | 210 | e108 | 0.166 [-0.128, 0.427] | - |
| scan gallery -> reg | 210 | Chamfer intera | 0.321 [0.057, 0.533] | -0.155 [-0.344, +0.093] (P<=0 0.882) |
| scan gallery -> reg | 210 | Chamfer regione stabile | 0.189 [-0.040, 0.420] | -0.023 [-0.401, +0.317] (P<=0 0.535) |
| scan peak -> scan | 5852 | e108 | 0.243 [0.068, 0.344] | - |
| scan peak -> scan | 5852 | Chamfer intera | 0.241 [0.092, 0.340] | +0.002 [-0.077, +0.081] (P<=0 0.450) |
| scan peak -> scan | 5852 | Chamfer regione stabile | 0.103 [-0.023, 0.211] | +0.140 [-0.062, +0.290] (P<=0 0.091) |
| scan nearneutral -> scan | 5852 | e108 | 0.375 [0.022, 0.616] | - |
| scan nearneutral -> scan | 5852 | Chamfer intera | 0.490 [0.156, 0.706] | -0.115 [-0.300, +0.058] (P<=0 0.920) |
| scan nearneutral -> scan | 5852 | Chamfer regione stabile | 0.214 [-0.040, 0.442] | +0.161 [-0.205, +0.442] (P<=0 0.171) |

## Controlli

```
{
 "n_meshes": 1284,
 "n_subjects": 15,
 "latent_dim": 256,
 "latent_finite": true,
 "chamfer_finite": true,
 "kept_vertex_fraction_min_median": [
  0.34106412005457026,
  0.5381945306470503
 ],
 "gt_mm_median_offdiag": 4.282371564105735
}
```
