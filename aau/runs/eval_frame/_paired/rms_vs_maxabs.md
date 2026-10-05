# Frame rms contro maxabs, Spearman vs D_GT per gruppo (eval_by_topology.py)

Delta = rms - maxabs, appaiato per seed. Ultime due righe: media e dev. std campionaria (ddof=1) dei Delta su 3 seed.

| seed | budget rms | budget maxabs | crop rms | crop maxabs | crop Δ | noisy rms | noisy maxabs | noisy Δ | resample rms | resample maxabs | resample Δ | all rms | all maxabs | all Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1234 | 120/120 (best 106) | 120/120 (best 114) | 0.7755 | 0.7653 | +0.0102 | 0.8090 | 0.7829 | +0.0261 | 0.7918 | 0.8048 | -0.0130 | 0.7879 | 0.7816 | +0.0063 |
| 2345 | 120/120 (best 94) | 120/120 (best 116) | 0.8077 | 0.7813 | +0.0264 | 0.8520 | 0.8353 | +0.0167 | 0.8469 | 0.8574 | -0.0105 | 0.8334 | 0.8242 | +0.0091 |
| 3456 | 120/120 (best 106) | 120/120 (best 116) | 0.8446 | 0.8237 | +0.0209 | 0.8707 | 0.8315 | +0.0392 | 0.8573 | 0.8507 | +0.0066 | 0.8549 | 0.8337 | +0.0212 |
| media | | | 0.8093 | 0.7901 | +0.0192 | 0.8439 | 0.8166 | +0.0273 | 0.8320 | 0.8376 | -0.0056 | 0.8254 | 0.8132 | +0.0122 |
| dev.std Δ | | |  |  | 0.0082 |  |  | 0.0113 |  |  | 0.0106 |  |  | 0.0079 |

Riferimento, fuori dal Delta appaiato:

| riferimento | frame | budget | crop | noisy | resample | all |
|---|---|---|---|---|---|---|
| v1_s1234 | current | 120/120 (best 82) | 0.7093 | 0.7937 | 0.7852 | 0.7506 |

Bracci:
- seed 1234: rms `remesh_v1recipe_rms_s1234_1054484_rms`, maxabs `remesh_v1recipe_current_s1234_1055026_maxabs`
- seed 2345: rms `remesh_v1recipe_rms_s2345_1054482_rms`, maxabs `remesh_v1recipe_current_s2345_1054492_maxabs`
- seed 3456: rms `remesh_v1recipe_rms_s3456_1055025_rms`, maxabs `remesh_v1recipe_current_s3456_1054493_maxabs`
