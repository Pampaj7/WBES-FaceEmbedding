# Frame rms contro maxabs, Spearman vs D_GT per gruppo (eval_by_topology.py)

Delta = rms - maxabs, appaiato per seed. Ultime due righe: media e dev. std campionaria (ddof=1) dei Delta su 3 seed.

| seed | budget rms | budget maxabs | crop rms | crop maxabs | crop Δ | noisy rms | noisy maxabs | noisy Δ | resample rms | resample maxabs | resample Δ | all rms | all maxabs | all Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1234 | 120/120 (best 106) | PARZIALE 94/120 (best 68) | 0.7755 | 0.7349 | +0.0406 | 0.8090 | 0.7809 | +0.0281 | 0.7918 | 0.7799 | +0.0119 | 0.7879 | 0.7588 | +0.0292 |
| 2345 | 120/120 (best 94) | 120/120 (best 116) | 0.8077 | 0.7813 | +0.0264 | 0.8520 | 0.8353 | +0.0167 | 0.8469 | 0.8574 | -0.0105 | 0.8334 | 0.8242 | +0.0091 |
| 3456 | PARZIALE 108/120 (best 106) | 120/120 (best 116) | 0.8317 | 0.8237 | +0.0080 | 0.8611 | 0.8315 | +0.0296 | 0.8538 | 0.8507 | +0.0032 | 0.8469 | 0.8337 | +0.0132 |
| media | | | 0.8050 | 0.7800 | +0.0250 | 0.8407 | 0.8159 | +0.0248 | 0.8308 | 0.8293 | +0.0015 | 0.8227 | 0.8056 | +0.0172 |
| dev.std Δ | | |  |  | 0.0163 |  |  | 0.0071 |  |  | 0.0113 |  |  | 0.0106 |

Riferimento, fuori dal Delta appaiato:

| riferimento | frame | budget | crop | noisy | resample | all |
|---|---|---|---|---|---|---|
| v1_s1234 | current | 120/120 (best 82) | 0.7093 | 0.7937 | 0.7852 | 0.7506 |

ATTENZIONE: budget ridotto (training interrotto) in seed 1234, seed 3456: il Delta di quei seed confronta budget diversi.

Bracci:
- seed 1234: rms `remesh_v1recipe_rms_s1234_1054484_rms`, maxabs `remesh_v1recipe_current_s1234_1054511_maxabs`
- seed 2345: rms `remesh_v1recipe_rms_s2345_1054482_rms`, maxabs `remesh_v1recipe_current_s2345_1054492_maxabs`
- seed 3456: rms `remesh_v1recipe_rms_s3456_1054483_rms`, maxabs `remesh_v1recipe_current_s3456_1054493_maxabs`
