# Ablazione k_eig 64 contro 128: risultati (regola di PROTOCOL_emendamento_1.md)

Generato da `v3_work/trainer/tools/k_ablation_summary.py`; numeri in `k_ablation_paired.csv`. Un solo seme (1234): gli IC coprono i soggetti di eval, non la variabilita' fra semi.

## Spearman graduato (IC 95% bootstrap per soggetto)

| dominio | gruppo | righe | GT | epoca | k64 | k128 |
|---|---|---|---|---|---|---|
| hifi3d | nocrop_cross | 98224 | FR | e036 | 0.216 [0.109, 0.333] | 0.230 [0.125, 0.338] |
| hifi3d | nocrop_cross | 98224 | SR | e036 | 0.309 [0.217, 0.392] | 0.321 [0.228, 0.403] |
| hifi3d | nocrop_cross | 98224 | MAXABS | e036 | 0.640 [0.545, 0.718] | 0.694 [0.615, 0.759] |
| hifi3d | nocrop_cross | 98224 | FR | e072 | 0.217 [0.112, 0.330] | 0.240 [0.131, 0.347] |
| hifi3d | nocrop_cross | 98224 | SR | e072 | 0.320 [0.234, 0.398] | 0.337 [0.252, 0.421] |
| hifi3d | nocrop_cross | 98224 | MAXABS | e072 | 0.648 [0.561, 0.720] | 0.694 [0.614, 0.758] |
| facescape | nocrop_cross | 88725 | FR | e036 | 0.317 [0.234, 0.402] | 0.305 [0.223, 0.392] |
| facescape | nocrop_cross | 88725 | SR | e036 | 0.375 [0.306, 0.442] | 0.357 [0.286, 0.431] |
| facescape | nocrop_cross | 88725 | MAXABS | e036 | 0.368 [0.300, 0.431] | 0.362 [0.295, 0.432] |
| facescape | nocrop_cross | 88725 | FR | e072 | 0.287 [0.213, 0.372] | 0.270 [0.196, 0.354] |
| facescape | nocrop_cross | 88725 | SR | e072 | 0.339 [0.269, 0.409] | 0.311 [0.235, 0.385] |
| facescape | nocrop_cross | 88725 | MAXABS | e072 | 0.337 [0.266, 0.405] | 0.306 [0.228, 0.376] |
| faceverse | mesh_pair_nocrop | 99000 | FR | e036 | 0.138 [0.070, 0.201] | 0.139 [0.067, 0.206] |
| faceverse | mesh_pair_nocrop | 99000 | SR | e036 | 0.148 [0.080, 0.217] | 0.141 [0.068, 0.211] |
| faceverse | mesh_pair_nocrop | 99000 | MAXABS | e036 | 0.222 [0.148, 0.294] | 0.212 [0.135, 0.284] |
| faceverse | mesh_pair_nocrop | 99000 | FR | e072 | 0.118 [0.043, 0.190] | 0.160 [0.088, 0.228] |
| faceverse | mesh_pair_nocrop | 99000 | SR | e072 | 0.131 [0.062, 0.205] | 0.164 [0.092, 0.240] |
| faceverse | mesh_pair_nocrop | 99000 | MAXABS | e072 | 0.214 [0.140, 0.292] | 0.235 [0.153, 0.313] |

## Delta appaiati (stesse repliche)

| dominio | GT | epoca | confronto | delta | IC 95% | P(delta<=0) |
|---|---|---|---|---|---|---|
| hifi3d | FR | e036 | k64-k128 | -0.014 | [-0.044, +0.020] | 0.791 |
| hifi3d | SR | e036 | k64-k128 | -0.013 | [-0.046, +0.023] | 0.735 |
| hifi3d | MAXABS | e036 | k64-k128 | -0.054 | [-0.087, -0.023] | 1.000 |
| hifi3d | FR | e072 | k64-k128 | -0.024 | [-0.059, +0.011] | 0.900 |
| hifi3d | SR | e072 | k64-k128 | -0.018 | [-0.053, +0.016] | 0.824 |
| hifi3d | MAXABS | e072 | k64-k128 | -0.046 | [-0.077, -0.018] | 1.000 |
| facescape | FR | e036 | k64-k128 | +0.011 | [-0.014, +0.040] | 0.214 |
| facescape | SR | e036 | k64-k128 | +0.018 | [-0.007, +0.044] | 0.082 |
| facescape | MAXABS | e036 | k64-k128 | +0.006 | [-0.022, +0.030] | 0.355 |
| facescape | FR | e072 | k64-k128 | +0.017 | [-0.007, +0.042] | 0.077 |
| facescape | SR | e072 | k64-k128 | +0.029 | [+0.003, +0.056] | 0.017 |
| facescape | MAXABS | e072 | k64-k128 | +0.032 | [+0.005, +0.059] | 0.013 |
| faceverse | FR | e036 | k64-k128 | -0.001 | [-0.051, +0.056] | 0.518 |
| faceverse | SR | e036 | k64-k128 | +0.007 | [-0.042, +0.060] | 0.398 |
| faceverse | MAXABS | e036 | k64-k128 | +0.009 | [-0.045, +0.066] | 0.359 |
| faceverse | FR | e072 | k64-k128 | -0.043 | [-0.097, +0.014] | 0.922 |
| faceverse | SR | e072 | k64-k128 | -0.033 | [-0.088, +0.021] | 0.875 |
| faceverse | MAXABS | e072 | k64-k128 | -0.021 | [-0.072, +0.033] | 0.799 |

## Tempo per passo (train.log, media delle epoche 2-72)

| k | s/passo |
|---|---|
| 64 | 0.818 |
| 128 | 1.047 |

## Regola (checkpoint e072)

- celle valutate: 9 di 9; peggiore: hifi3d MAXABS -0.046 [-0.077, -0.018]
- celle sotto -0.03: hifi3d MAXABS -0.046, faceverse FR -0.043, faceverse SR -0.033
- velocita' (motivo dell'adozione, non criterio): 0.818 contro 1.047 s/passo nel train.log

k adottato = **128** (k64 perde piu' di 0.03 in almeno una cella)
