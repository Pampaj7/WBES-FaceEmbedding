# Risultati: HIFI3D, neutre (senza espressioni)

Soggetti: 100 (`select_subjects`, seed 1234), mesh da `datasets/HIFI3D/eval_view/npz`; studente da `aau/runs/distill_pilot/eval/hifi3d`, insegnante da `aau/runs/arcface_render_zs/hifi3d`, baseline da `aau/runs/ws_hifi3d/data_328f2bfc1a/baselines`, congiunto da `aau/runs/ws_hifi3d/data_328f2bfc1a/joint_frame-xmymz_flip_ranking/zs_zeroshot`. CI 95% bootstrap per soggetto, 1000 repliche (le stesse del gate).

## PRIMARIO: riconoscimento d'identita', 5 topologie senza crop

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | 0.224 [0.207, 0.244] | 0.279 [0.261, 0.299] | 0.603 [0.589, 0.617] | 0 |
| Studente distillato, convenzione ICT | 0.355 [0.322, 0.392] | 0.446 [0.413, 0.479] | 0.781 [0.761, 0.800] | 0 |
| ArcFace, normal map, 3 viste | 0.998 [0.995, 1.000] | 0.999 [0.998, 1.000] | 0.997 [0.996, 0.998] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.984 [0.974, 0.992] | 0.991 [0.985, 0.996] | 0.989 [0.985, 0.993] | 0 |
| Rigid ICP + NICP + P2Tri | 0.990 [0.982, 0.998] | 0.990 [0.982, 0.998] | 1.000 [1.000, 1.000] | 1965 |
| Rigid ICP + Chamfer | 0.996 [0.991, 0.999] | 0.997 [0.995, 0.999] | 0.999 [0.999, 1.000] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.477 [0.445, 0.513] | 0.564 [0.534, 0.594] | 0.782 [0.767, 0.797] | 0 |
| BFM+ICT, convenzione BFM | 0.397 [0.363, 0.434] | 0.517 [0.489, 0.549] | 0.755 [0.740, 0.770] | 0 |

### Delta appaiati, studente (convenzione BFM) - baseline

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, normal map, 3 viste | -0.774 [-0.791, -0.754] (1.000) | -0.720 [-0.738, -0.700] (1.000) | -0.395 [-0.409, -0.380] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, ombreggiato, 3 viste | -0.760 [-0.778, -0.738] (1.000) | -0.712 [-0.731, -0.691] (1.000) | -0.387 [-0.400, -0.372] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + NICP + P2Tri | -0.766 [-0.785, -0.744] (1.000) | -0.711 [-0.733, -0.689] (1.000) | -0.397 [-0.411, -0.383] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + Chamfer | -0.772 [-0.790, -0.750] (1.000) | -0.718 [-0.737, -0.698] (1.000) | -0.397 [-0.411, -0.382] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Chamfer (faceBench, 4096 pt) | -0.253 [-0.285, -0.222] (1.000) | -0.285 [-0.311, -0.261] (1.000) | -0.180 [-0.197, -0.163] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | BFM+ICT, convenzione BFM | -0.173 [-0.207, -0.139] (1.000) | -0.238 [-0.267, -0.211] (1.000) | -0.153 [-0.167, -0.137] (1.000) |

### Criterio: rapporto con l'insegnante a normal map

- Studente distillato, convenzione BFM (riferimento): rank-1 / insegnante normal map = 0.224 [0.208, 0.245]
- Studente distillato, convenzione ICT: rank-1 / insegnante normal map = 0.356 [0.322, 0.392]

### Secondari

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione ICT | BFM+ICT, convenzione BFM | -0.042 [-0.079, -0.003] (0.986) | -0.071 [-0.102, -0.038] (1.000) | +0.026 [+0.008, +0.043] (0.002) |
| Studente distillato, convenzione ICT | Studente distillato, convenzione BFM (riferimento) | +0.131 [+0.099, +0.162] (0.000) | +0.167 [+0.138, +0.192] (0.000) | +0.179 [+0.157, +0.200] (0.000) |
| Studente distillato, convenzione ICT | ArcFace, normal map, 3 viste | -0.643 [-0.676, -0.607] (1.000) | -0.553 [-0.585, -0.519] (1.000) | -0.216 [-0.237, -0.197] (1.000) |

## A parte: crop (coppie di topologie con crop da un lato)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | 0.038 [0.020, 0.060] | 0.110 [0.089, 0.136] | 0.579 [0.567, 0.592] | 0 |
| Studente distillato, convenzione ICT | 0.257 [0.219, 0.293] | 0.367 [0.333, 0.400] | 0.704 [0.685, 0.722] | 0 |
| ArcFace, normal map, 3 viste | 0.997 [0.994, 1.000] | 0.999 [0.997, 1.000] | 0.999 [0.998, 0.999] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.990 [0.984, 0.996] | 0.994 [0.991, 0.998] | 0.994 [0.992, 0.996] | 0 |
| Rigid ICP + NICP + P2Tri | 0.777 [0.740, 0.810] | 0.854 [0.829, 0.877] | 0.997 [0.995, 0.998] | 1965 |
| Rigid ICP + Chamfer | 0.284 [0.237, 0.338] | 0.423 [0.381, 0.470] | 0.888 [0.865, 0.911] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.472 [0.445, 0.499] | 0.536 [0.510, 0.564] | 0.699 [0.684, 0.715] | 0 |
| BFM+ICT, convenzione BFM | 0.085 [0.057, 0.117] | 0.201 [0.171, 0.233] | 0.716 [0.699, 0.734] | 0 |

### Delta appaiati, crop

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, normal map, 3 viste | -0.959 [-0.978, -0.937] (1.000) | -0.889 [-0.910, -0.863] (1.000) | -0.419 [-0.432, -0.406] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, ombreggiato, 3 viste | -0.952 [-0.972, -0.929] (1.000) | -0.885 [-0.906, -0.859] (1.000) | -0.415 [-0.428, -0.401] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + NICP + P2Tri | -0.739 [-0.777, -0.700] (1.000) | -0.744 [-0.774, -0.713] (1.000) | -0.418 [-0.430, -0.405] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + Chamfer | -0.246 [-0.299, -0.202] (1.000) | -0.313 [-0.361, -0.273] (1.000) | -0.308 [-0.331, -0.284] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Chamfer (faceBench, 4096 pt) | -0.434 [-0.461, -0.408] (1.000) | -0.427 [-0.449, -0.404] (1.000) | -0.120 [-0.133, -0.106] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | BFM+ICT, convenzione BFM | -0.047 [-0.075, -0.022] (1.000) | -0.092 [-0.120, -0.067] (1.000) | -0.136 [-0.151, -0.121] (1.000) |
| Studente distillato, convenzione ICT | BFM+ICT, convenzione BFM | +0.172 [+0.130, +0.212] (0.000) | +0.166 [+0.132, +0.201] (0.000) | -0.012 [-0.030, +0.006] (0.901) |
| Studente distillato, convenzione ICT | Studente distillato, convenzione BFM (riferimento) | +0.219 [+0.184, +0.253] (0.000) | +0.257 [+0.226, +0.287] (0.000) | +0.124 [+0.107, +0.141] (0.000) |
| Studente distillato, convenzione ICT | ArcFace, normal map, 3 viste | -0.740 [-0.778, -0.703] (1.000) | -0.631 [-0.666, -0.599] (1.000) | -0.295 [-0.314, -0.276] (1.000) |

- Studente distillato, convenzione BFM (riferimento): rank-1 / insegnante normal map = 0.038 [0.020, 0.060]
- Studente distillato, convenzione ICT: rank-1 / insegnante normal map = 0.258 [0.220, 0.294]

## Controlli

- riproduzione di `aau/runs/arcface_render_zs/hifi3d/recognition.csv` (stesse repliche): max |diff| su punto e CI = 1.11e-16 su 10 righe (arcface_shaded_3v, chamfer, joint@bfm, nicp_p2tri, rigid_icp_chamfer); righe non presenti nel csv del gate (calcolate qui): arcface_normals_3v
