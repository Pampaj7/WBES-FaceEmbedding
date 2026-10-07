# Risultati: FaceVerse v2 con espressioni casuali

Soggetti: 100 (`select_subjects`, seed 1234), mesh da `datasets/FACEVERSE_ZS/expr_view/npz`; studente da `aau/runs/distill_pilot/eval/fv_expr`, insegnante da `aau/runs/arcface_render_zs/fv_expr`, baseline da `aau/runs/ws_faceverse_expr/data_736f96956a/baselines`, congiunto da `aau/runs/ws_faceverse_expr/data_736f96956a/joint_flip_topology/zs_zeroshot`. CI 95% bootstrap per soggetto, 1000 repliche (le stesse del gate).

## PRIMARIO: riconoscimento d'identita', 5 topologie senza crop

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | 0.332 [0.299, 0.366] | 0.432 [0.400, 0.465] | 0.797 [0.773, 0.821] | 0 |
| Studente distillato, convenzione ICT | 0.570 [0.520, 0.618] | 0.655 [0.611, 0.696] | 0.899 [0.875, 0.921] | 0 |
| ArcFace, normal map, 3 viste | 0.867 [0.843, 0.889] | 0.911 [0.893, 0.927] | 0.934 [0.921, 0.946] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.750 [0.724, 0.775] | 0.807 [0.785, 0.829] | 0.863 [0.847, 0.879] | 0 |
| Rigid ICP + NICP + P2Tri | 0.959 [0.940, 0.975] | 0.968 [0.953, 0.981] | 0.995 [0.992, 0.998] | 0 |
| Rigid ICP + Chamfer | 0.918 [0.895, 0.939] | 0.935 [0.916, 0.953] | 0.986 [0.979, 0.992] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.740 [0.698, 0.783] | 0.775 [0.737, 0.814] | 0.882 [0.853, 0.910] | 0 |
| BFM+ICT, convenzione BFM | 0.680 [0.634, 0.723] | 0.731 [0.688, 0.770] | 0.875 [0.847, 0.900] | 0 |

### Delta appaiati, studente (convenzione BFM) - baseline

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, normal map, 3 viste | -0.535 [-0.575, -0.496] (1.000) | -0.479 [-0.515, -0.443] (1.000) | -0.138 [-0.163, -0.112] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, ombreggiato, 3 viste | -0.418 [-0.457, -0.379] (1.000) | -0.375 [-0.411, -0.338] (1.000) | -0.067 [-0.093, -0.039] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + NICP + P2Tri | -0.627 [-0.660, -0.592] (1.000) | -0.536 [-0.566, -0.505] (1.000) | -0.198 [-0.220, -0.175] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + Chamfer | -0.586 [-0.621, -0.548] (1.000) | -0.503 [-0.535, -0.470] (1.000) | -0.190 [-0.211, -0.168] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Chamfer (faceBench, 4096 pt) | -0.408 [-0.454, -0.365] (1.000) | -0.343 [-0.383, -0.304] (1.000) | -0.086 [-0.110, -0.061] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | BFM+ICT, convenzione BFM | -0.348 [-0.395, -0.300] (1.000) | -0.300 [-0.342, -0.258] (1.000) | -0.078 [-0.104, -0.052] (1.000) |

### Criterio: rapporto con l'insegnante a normal map

- Studente distillato, convenzione BFM (riferimento): rank-1 / insegnante normal map = 0.383 [0.344, 0.422]
- Studente distillato, convenzione ICT: rank-1 / insegnante normal map = 0.657 [0.596, 0.717]

### Secondari

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione ICT | BFM+ICT, convenzione BFM | -0.110 [-0.154, -0.062] (1.000) | -0.077 [-0.117, -0.033] (1.000) | +0.024 [-0.005, +0.051] (0.047) |
| Studente distillato, convenzione ICT | Studente distillato, convenzione BFM (riferimento) | +0.238 [+0.186, +0.290] (0.000) | +0.223 [+0.180, +0.269] (0.000) | +0.102 [+0.076, +0.125] (0.000) |
| Studente distillato, convenzione ICT | ArcFace, normal map, 3 viste | -0.297 [-0.353, -0.242] (1.000) | -0.256 [-0.303, -0.210] (1.000) | -0.036 [-0.063, -0.010] (0.998) |

## A parte: crop (coppie di topologie con crop da un lato)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | 0.031 [0.018, 0.046] | 0.106 [0.088, 0.126] | 0.651 [0.626, 0.678] | 0 |
| Studente distillato, convenzione ICT | 0.269 [0.218, 0.324] | 0.378 [0.325, 0.431] | 0.804 [0.773, 0.836] | 0 |
| ArcFace, normal map, 3 viste | 0.952 [0.937, 0.964] | 0.966 [0.955, 0.976] | 0.963 [0.954, 0.970] | 0 |
| ArcFace, ombreggiato, 3 viste | 0.881 [0.867, 0.896] | 0.909 [0.897, 0.921] | 0.922 [0.910, 0.932] | 0 |
| Rigid ICP + NICP + P2Tri | 0.909 [0.876, 0.938] | 0.934 [0.910, 0.956] | 0.983 [0.974, 0.992] | 0 |
| Rigid ICP + Chamfer | 0.513 [0.455, 0.571] | 0.590 [0.540, 0.640] | 0.777 [0.734, 0.814] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.396 [0.336, 0.457] | 0.479 [0.421, 0.537] | 0.809 [0.781, 0.837] | 0 |
| BFM+ICT, convenzione BFM | 0.243 [0.205, 0.283] | 0.364 [0.325, 0.405] | 0.801 [0.771, 0.829] | 0 |

### Delta appaiati, crop

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, normal map, 3 viste | -0.921 [-0.939, -0.902] (1.000) | -0.860 [-0.881, -0.838] (1.000) | -0.312 [-0.339, -0.284] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | ArcFace, ombreggiato, 3 viste | -0.850 [-0.870, -0.829] (1.000) | -0.803 [-0.824, -0.780] (1.000) | -0.271 [-0.298, -0.242] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + NICP + P2Tri | -0.878 [-0.911, -0.840] (1.000) | -0.828 [-0.857, -0.796] (1.000) | -0.332 [-0.358, -0.306] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Rigid ICP + Chamfer | -0.482 [-0.541, -0.422] (1.000) | -0.484 [-0.536, -0.429] (1.000) | -0.126 [-0.168, -0.083] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | Chamfer (faceBench, 4096 pt) | -0.365 [-0.425, -0.304] (1.000) | -0.373 [-0.430, -0.315] (1.000) | -0.158 [-0.194, -0.124] (1.000) |
| Studente distillato, convenzione BFM (riferimento) | BFM+ICT, convenzione BFM | -0.212 [-0.254, -0.170] (1.000) | -0.258 [-0.303, -0.213] (1.000) | -0.150 [-0.181, -0.116] (1.000) |
| Studente distillato, convenzione ICT | BFM+ICT, convenzione BFM | +0.026 [-0.034, +0.091] (0.215) | +0.014 [-0.048, +0.076] (0.327) | +0.003 [-0.033, +0.040] (0.402) |
| Studente distillato, convenzione ICT | Studente distillato, convenzione BFM (riferimento) | +0.238 [+0.186, +0.291] (0.000) | +0.272 [+0.218, +0.321] (0.000) | +0.153 [+0.112, +0.194] (0.000) |
| Studente distillato, convenzione ICT | ArcFace, normal map, 3 viste | -0.683 [-0.733, -0.629] (1.000) | -0.588 [-0.640, -0.537] (1.000) | -0.159 [-0.190, -0.124] (1.000) |

- Studente distillato, convenzione BFM (riferimento): rank-1 / insegnante normal map = 0.033 [0.019, 0.049]
- Studente distillato, convenzione ICT: rank-1 / insegnante normal map = 0.283 [0.229, 0.340]

## Controlli

- riproduzione di `aau/runs/arcface_render_zs/fv_expr/recognition.csv` (stesse repliche): max |diff| su punto e CI = 1.11e-16 su 12 righe (arcface_normals_3v, arcface_shaded_3v, chamfer, joint@bfm, nicp_p2tri, rigid_icp_chamfer)
