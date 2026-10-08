## Distanza graduata (vista neutra), Spearman con la GT [CI 95%]

| metodo | senza crop, GT maxabs | all_cross, GT maxabs | subject-pair-mean, GT maxabs | senza crop, GT coef |
| --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) | 0.394 [0.340, 0.442] | 0.360 [0.304, 0.411] | 0.735 [0.680, 0.780] | 0.109 [0.062, 0.155] |
| Chamfer intera (4096 pt) | 0.361 [0.303, 0.417] | 0.231 [0.188, 0.279] | 0.584 [0.514, 0.654] | 0.111 [0.060, 0.161] |
| Chamfer regione stabile | 0.277 [0.203, 0.348] | 0.245 [0.173, 0.317] | 0.453 [0.349, 0.561] | 0.120 [0.074, 0.168] |

Delta appaiati braccio - Chamfer (stesse repliche bootstrap per soggetto):

| confronto | senza crop, maxabs | all_cross, maxabs | subject-pair-mean, maxabs | senza crop, coef |
| --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) - Chamfer intera (4096 pt) | +0.032 [-0.016, +0.086] (P<=0 0.100) | +0.129 [+0.084, +0.176] (P<=0 0.000) | +0.150 [+0.088, +0.220] (P<=0 0.000) | -0.002 [-0.040, +0.037] (P<=0 0.517) |
| e108 (BFM+ICT+GNM) - Chamfer regione stabile | +0.116 [+0.050, +0.188] (P<=0 0.000) | +0.114 [+0.048, +0.188] (P<=0 0.001) | +0.282 [+0.186, +0.385] (P<=0 0.000) | -0.010 [-0.059, +0.033] (P<=0 0.662) |

## Riconoscimento con espressioni (vista con espressioni)

| metodo | blocco | rank-1 | mAP | AUC verifica | NaN |
| --- | --- | --- | --- | --- | --- |
| Chamfer intera (4096 pt) | crop | 0.040 [0.020, 0.065] | 0.103 [0.080, 0.128] | 0.610 [0.593, 0.627] | 0 |
| Chamfer regione stabile | crop | 0.110 [0.081, 0.139] | 0.229 [0.200, 0.258] | 0.737 [0.715, 0.758] | 0 |
| e108 (BFM+ICT+GNM) | crop | 0.157 [0.126, 0.189] | 0.268 [0.237, 0.302] | 0.722 [0.699, 0.743] | 0 |
| Chamfer intera (4096 pt) | nocrop | 0.424 [0.383, 0.468] | 0.523 [0.485, 0.563] | 0.781 [0.760, 0.802] | 0 |
| Chamfer regione stabile | nocrop | 0.594 [0.547, 0.637] | 0.662 [0.618, 0.703] | 0.740 [0.715, 0.765] | 0 |
| e108 (BFM+ICT+GNM) | nocrop | 0.285 [0.254, 0.319] | 0.395 [0.363, 0.427] | 0.748 [0.725, 0.769] | 0 |

| delta | blocco | rank-1 | mAP | AUC |
| --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) - Chamfer intera (4096 pt) | crop | +0.117 [+0.083, +0.150] (P<=0 0.000) | +0.165 [+0.130, +0.200] (P<=0 0.000) | +0.111 [+0.088, +0.133] (P<=0 0.000) |
| e108 (BFM+ICT+GNM) - Chamfer regione stabile | crop | +0.047 [+0.005, +0.089] (P<=0 0.011) | +0.039 [-0.002, +0.078] (P<=0 0.030) | -0.015 [-0.042, +0.012] (P<=0 0.867) |
| e108 (BFM+ICT+GNM) - Chamfer intera (4096 pt) | nocrop | -0.139 [-0.169, -0.110] (P<=0 1.000) | -0.128 [-0.156, -0.103] (P<=0 1.000) | -0.033 [-0.048, -0.018] (P<=0 0.999) |
| e108 (BFM+ICT+GNM) - Chamfer regione stabile | nocrop | -0.308 [-0.358, -0.256] (P<=0 1.000) | -0.267 [-0.316, -0.220] (P<=0 1.000) | +0.008 [-0.019, +0.037] (P<=0 0.298) |

## Riconoscimento senza espressioni (vista neutra, accessorio)

| metodo | blocco | rank-1 | mAP | AUC verifica | NaN |
| --- | --- | --- | --- | --- | --- |
| Chamfer intera (4096 pt) | crop | 0.052 [0.028, 0.081] | 0.121 [0.095, 0.150] | 0.651 [0.630, 0.671] | 0 |
| Chamfer regione stabile | crop | 0.139 [0.105, 0.172] | 0.258 [0.223, 0.289] | 0.768 [0.744, 0.791] | 0 |
| e108 (BFM+ICT+GNM) | crop | 0.336 [0.293, 0.376] | 0.482 [0.446, 0.516] | 0.799 [0.782, 0.816] | 0 |
| Chamfer intera (4096 pt) | nocrop | 0.685 [0.667, 0.705] | 0.741 [0.723, 0.757] | 0.794 [0.777, 0.811] | 0 |
| Chamfer regione stabile | nocrop | 0.677 [0.638, 0.715] | 0.741 [0.706, 0.774] | 0.782 [0.763, 0.802] | 0 |
| e108 (BFM+ICT+GNM) | nocrop | 0.642 [0.622, 0.665] | 0.708 [0.689, 0.729] | 0.800 [0.784, 0.816] | 0 |

| delta | blocco | rank-1 | mAP | AUC |
| --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) - Chamfer intera (4096 pt) | crop | +0.284 [+0.240, +0.326] (P<=0 0.000) | +0.362 [+0.323, +0.396] (P<=0 0.000) | +0.148 [+0.127, +0.169] (P<=0 0.000) |
| e108 (BFM+ICT+GNM) - Chamfer regione stabile | crop | +0.197 [+0.148, +0.244] (P<=0 0.000) | +0.225 [+0.184, +0.266] (P<=0 0.000) | +0.031 [+0.011, +0.052] (P<=0 0.002) |
| e108 (BFM+ICT+GNM) - Chamfer intera (4096 pt) | nocrop | -0.043 [-0.063, -0.023] (P<=0 1.000) | -0.033 [-0.048, -0.018] (P<=0 1.000) | +0.006 [-0.006, +0.019] (P<=0 0.148) |
| e108 (BFM+ICT+GNM) - Chamfer regione stabile | nocrop | -0.035 [-0.074, +0.003] (P<=0 0.957) | -0.032 [-0.064, -0.001] (P<=0 0.977) | +0.018 [+0.002, +0.036] (P<=0 0.019) |

## Secondario: vista con espressioni, Spearman con la GT d'identita' neutra, senza crop

| metodo | Spearman [CI] | delta braccio - metodo [CI] |
| --- | --- | --- |
| e108 (BFM+ICT+GNM) | 0.300 [0.249, 0.357] | - |
| Chamfer intera (4096 pt) | 0.340 [0.278, 0.397] | e108 (BFM+ICT+GNM): -0.040 [-0.094, +0.014] |
| Chamfer regione stabile | 0.243 [0.178, 0.311] | e108 (BFM+ICT+GNM): +0.058 [-0.002, +0.121] |

## Punteggio dev (sez. 9): media fra graduata senza crop (neutra, GT maxabs) e rank-1 con espressioni senza crop

| metodo | graduata | rank-1 | punteggio [CI 95%] |
| --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) | 0.394 | 0.285 | 0.340 [0.302, 0.376] |
| Chamfer intera (4096 pt) | 0.361 | 0.424 | 0.393 [0.353, 0.431] |
| e108 (BFM+ICT+GNM) - Chamfer intera (4096 pt) | | | -0.053 [-0.082, -0.021] (P<=0 0.999) |

## Controlli

```
{
 "region_kept_fraction_neutral_min_median": [
  0.4723731884057971,
  0.5050431667841823
 ],
 "region_kept_fraction_expr_min_median": [
  0.45497737556561085,
  0.5039695141314703
 ],
 "chamfer_eval_available": false,
 "arm_sources": {
  "neutral/scale_e108": "aau/runs/ws_dev_facescape/data_aca84a16c6/scale_e108_embed/zs_zeroshot",
  "expr/scale_e108": "aau/runs/ws_dev_facescape_expr/data_b03b1fca0e/scale_e108_embed/zs_zeroshot"
 }
}
```
