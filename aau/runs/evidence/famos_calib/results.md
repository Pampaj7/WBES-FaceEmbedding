# FaMoS: calibrazione lineare post hoc per l'invarianza all'espressione

Generato da `aau/famos_calib/fcal.py eval` (regole in `REGOLE.md`, committate prima dei numeri). **POST HOC e DESCRITTIVO**: FaMoS TEST e' gia' stato visto (`esterna_famos_id`, `famos_dominio`); nessun test confermativo.

## Dati di calibrazione

- 73 soggetti FaMoS TRAIN su 80 (esclusi per sequenze mancanti: 7; tolti per patch fallite: 0), 730 mesh, le 10 sequenze di `esterna_famos_id`; fold [15, 15, 15, 14, 14]. Ruoli dei fotogrammi: {'gallery': 15, 'peak': 352, 'nearneutral': 363}; expr_mm mediana 1.01 mm (p5 0.32, p95 5.89).
- **Dominio diverso dal test**: registrazioni FaMoS (mesh in topologia FLAME, superficie pulita e senza buchi), non scansioni grezze (quelle di TRAIN non sono su disco). Stessa preparazione di `ei.py`.
- Preparazione: 730/730 patch, ripieghi del rilevatore sulla registrazione 0, sulla patch 0; residuo dei 7 landmark mediana 2.70 mm (`prep_calib.json`).

## Scelte sulla calibrazione (validazione incrociata sui fold, AUC media)

| modello | variante | parametri scelti | AUC cv |
|---|---|---|---|
| factorized_s1234 | shape | `{}` | 0.816 |
| factorized_s1234 | form | `{"c": 1.0}` | 0.822 |
| factorized_s1234 | wccn | `{"lam": 0.1}` | 0.977 |
| factorized_s1234 | lda | `{"lam": 0.01, "k": 16}` | 0.998 |
| factorized_s1234 | wccn_taglia | `{"g": 16.0, "base": {"lam": 0.1}}` | 0.972 |
| factorized_s1234 | lda_taglia | `{"g": 16.0, "base": {"lam": 0.01, "k": 16}}` | 0.998 |
| factorized_s1234 | form_c | `{"c": 0.7071067811865476}` | 0.823 |
| factorized_s2345 | shape | `{}` | 0.800 |
| factorized_s2345 | form | `{"c": 1.0}` | 0.807 |
| factorized_s2345 | wccn | `{"lam": 0.1}` | 0.974 |
| factorized_s2345 | lda | `{"lam": 0.01, "k": 16}` | 0.995 |
| factorized_s2345 | wccn_taglia | `{"g": 16.0, "base": {"lam": 0.1}}` | 0.968 |
| factorized_s2345 | lda_taglia | `{"g": 16.0, "base": {"lam": 0.01, "k": 16}}` | 0.995 |
| factorized_s2345 | form_c | `{"c": 0.7071067811865476}` | 0.808 |
| concorrente | icp | `rej_pulli` (default: AUC cv 0.939) | 0.942 |
| concorrente | now | `lmk68` (default: AUC cv 0.926) | 0.927 |

## Test: 15 soggetti FaMoS TEST, 150 scansioni grezze (protocollo di `esterna_famos_id`)

valore [IC 95%], 1000 repliche per soggetto, seme 1234. Strati d'espressione di `famos_dominio` (terzili di expr_mm 0.862 / 3.315 mm): AUC_k = stessa persona dello strato k contro tutte le coppie di persone diverse; rank-1_k = query dello strato.

| metodo | rank-1 | mAP | EER | AUC | AUC H | rank-1 H | AUC L |
|---|---|---|---|---|---|---|---|
| ICP + Chamfer in mm, default (non cal.) | 0.947 [0.913, 0.980] | 0.822 [0.773, 0.873] | 0.173 [0.121, 0.225] | 0.895 [0.858, 0.932] | 0.820 [0.752, 0.885] | 0.860 [0.744, 0.952] | 1.000 [1.000, 1.000] |
| ICP + Chamfer in mm, configurazione scelta su FaMoS TRAIN (rej_pulli) | 0.953 [0.927, 0.980] | 0.831 [0.781, 0.882] | 0.170 [0.117, 0.218] | 0.901 [0.866, 0.938] | 0.830 [0.763, 0.895] | 0.880 [0.787, 0.953] | 1.000 [1.000, 1.000] |
| NoW metrical (rilevatore), default (non cal.) | 0.967 [0.940, 0.993] | 0.865 [0.812, 0.914] | 0.157 [0.101, 0.206] | 0.918 [0.879, 0.955] | 0.852 [0.778, 0.919] | 0.900 [0.818, 0.977] | 1.000 [1.000, 1.000] |
| NoW metrical (rilevatore), configurazione scelta su FaMoS TRAIN (lmk68) | 0.967 [0.940, 0.993] | 0.865 [0.812, 0.914] | 0.157 [0.101, 0.206] | 0.918 [0.879, 0.954] | 0.852 [0.777, 0.919] | 0.900 [0.818, 0.977] | 1.000 [1.000, 1.000] |
| nostro s1234, d_F non cal. (non cal.) | 0.853 [0.813, 0.900] | 0.664 [0.600, 0.739] | 0.242 [0.184, 0.294] | 0.830 [0.783, 0.883] | 0.713 [0.636, 0.796] | 0.640 [0.465, 0.765] | 1.000 [1.000, 1.000] |
| nostro s1234, d_F cal. (non cal.) | 0.873 [0.827, 0.927] | 0.651 [0.585, 0.726] | 0.265 [0.210, 0.323] | 0.816 [0.760, 0.870] | 0.697 [0.594, 0.788] | 0.700 [0.537, 0.814] | 1.000 [1.000, 1.000] |
| nostro s1234, d_P (non cal.) | 0.813 [0.753, 0.873] | 0.636 [0.562, 0.716] | 0.241 [0.187, 0.294] | 0.830 [0.781, 0.885] | 0.716 [0.639, 0.797] | 0.540 [0.382, 0.667] | 1.000 [1.000, 1.000] |
| nostro s1234, WCCN (u sbiancata), d' | 1.000 [1.000, 1.000] | 0.929 [0.893, 0.956] | 0.104 [0.065, 0.139] | 0.964 [0.941, 0.985] | 0.936 [0.893, 0.974] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| nostro s1234, LDA, d' | 1.000 [1.000, 1.000] | 0.982 [0.966, 0.993] | 0.044 [0.012, 0.074] | 0.993 [0.983, 0.999] | 0.988 [0.968, 0.999] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| nostro s1234, d_F ricomposta: taglia + u WCCN | 1.000 [1.000, 1.000] | 0.914 [0.873, 0.949] | 0.121 [0.079, 0.163] | 0.949 [0.920, 0.978] | 0.910 [0.858, 0.961] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| nostro s1234, d_F ricomposta: taglia + u LDA | 1.000 [1.000, 1.000] | 0.978 [0.960, 0.992] | 0.055 [0.017, 0.084] | 0.991 [0.980, 0.999] | 0.984 [0.961, 0.998] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| nostro s1234, controllo: d_F con u non cal., c scelto su FaMoS TRAIN | 0.873 [0.833, 0.913] | 0.673 [0.608, 0.746] | 0.244 [0.184, 0.293] | 0.830 [0.782, 0.882] | 0.715 [0.631, 0.797] | 0.700 [0.558, 0.805] | 1.000 [1.000, 1.000] |
| nostro s2345, d_F non cal. (non cal.) | 0.820 [0.767, 0.880] | 0.641 [0.566, 0.724] | 0.259 [0.201, 0.313] | 0.811 [0.758, 0.869] | 0.681 [0.596, 0.771] | 0.540 [0.333, 0.719] | 1.000 [1.000, 1.000] |
| nostro s2345, d_F cal. (non cal.) | 0.853 [0.807, 0.907] | 0.628 [0.549, 0.712] | 0.283 [0.218, 0.332] | 0.805 [0.743, 0.862] | 0.678 [0.563, 0.777] | 0.660 [0.488, 0.793] | 1.000 [1.000, 1.000] |
| nostro s2345, d_P (non cal.) | 0.800 [0.753, 0.847] | 0.614 [0.539, 0.696] | 0.265 [0.204, 0.320] | 0.808 [0.754, 0.869] | 0.679 [0.596, 0.768] | 0.500 [0.341, 0.625] | 1.000 [1.000, 1.000] |
| nostro s2345, WCCN (u sbiancata), d' | 1.000 [1.000, 1.000] | 0.921 [0.865, 0.960] | 0.119 [0.073, 0.157] | 0.957 [0.929, 0.981] | 0.924 [0.874, 0.966] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| nostro s2345, LDA, d' | 0.993 [0.980, 1.000] | 0.978 [0.955, 0.995] | 0.037 [0.012, 0.070] | 0.993 [0.983, 0.999] | 0.987 [0.967, 0.999] | 0.980 [0.932, 1.000] | 1.000 [1.000, 1.000] |
| nostro s2345, d_F ricomposta: taglia + u WCCN | 1.000 [1.000, 1.000] | 0.903 [0.838, 0.953] | 0.134 [0.080, 0.179] | 0.940 [0.902, 0.974] | 0.892 [0.822, 0.953] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| nostro s2345, d_F ricomposta: taglia + u LDA | 0.987 [0.967, 1.000] | 0.975 [0.949, 0.993] | 0.043 [0.015, 0.078] | 0.991 [0.979, 0.999] | 0.984 [0.961, 0.998] | 0.960 [0.894, 1.000] | 1.000 [1.000, 1.000] |
| nostro s2345, controllo: d_F con u non cal., c scelto su FaMoS TRAIN | 0.833 [0.780, 0.893] | 0.651 [0.574, 0.731] | 0.258 [0.203, 0.316] | 0.813 [0.758, 0.869] | 0.686 [0.588, 0.777] | 0.580 [0.366, 0.750] | 1.000 [1.000, 1.000] |

## Delta: varianti s1234 contro nostro s1234, d_F non cal. (non cal.)

positivo = la variante meglio (EER: riferimento - variante); stesse repliche.

| variante | rank-1 | mAP | EER | AUC | AUC H | rank-1 H | AUC L |
|---|---|---|---|---|---|---|---|
| nostro s1234, WCCN (u sbiancata), d' | +0.147 [+0.100, +0.187] | +0.264 [+0.206, +0.313] | +0.137 [+0.084, +0.194] | +0.134 [+0.086, +0.178] | +0.223 [+0.156, +0.296] | +0.360 [+0.235, +0.535] | -0.000 [-0.000, +0.000] |
| nostro s1234, LDA, d' | +0.147 [+0.100, +0.187] | +0.317 [+0.247, +0.378] | +0.198 [+0.143, +0.249] | +0.163 [+0.111, +0.210] | +0.275 [+0.199, +0.349] | +0.360 [+0.235, +0.535] | +0.000 [+0.000, +0.000] |
| nostro s1234, d_F ricomposta: taglia + u WCCN | +0.147 [+0.100, +0.187] | +0.250 [+0.185, +0.302] | +0.120 [+0.059, +0.179] | +0.120 [+0.072, +0.172] | +0.197 [+0.122, +0.281] | +0.360 [+0.235, +0.535] | -0.000 [-0.000, +0.000] |
| nostro s1234, d_F ricomposta: taglia + u LDA | +0.147 [+0.100, +0.187] | +0.314 [+0.243, +0.375] | +0.187 [+0.133, +0.245] | +0.161 [+0.108, +0.209] | +0.271 [+0.196, +0.347] | +0.360 [+0.235, +0.535] | +0.000 [+0.000, +0.000] |
| nostro s1234, controllo: d_F con u non cal., c scelto su FaMoS TRAIN | +0.020 [+0.000, +0.040] | +0.008 [-0.005, +0.023] | -0.003 [-0.013, +0.011] | +0.000 [-0.008, +0.009] | +0.002 [-0.013, +0.016] | +0.060 [+0.000, +0.140] | +0.000 [+0.000, +0.000] |

## Delta: varianti s1234 contro ICP + Chamfer in mm, default (non cal.)

positivo = la variante meglio (EER: riferimento - variante); stesse repliche.

| variante | rank-1 | mAP | EER | AUC | AUC H | rank-1 H | AUC L |
|---|---|---|---|---|---|---|---|
| nostro s1234, WCCN (u sbiancata), d' | +0.053 [+0.020, +0.087] | +0.107 [+0.057, +0.156] | +0.069 [+0.025, +0.124] | +0.069 [+0.037, +0.104] | +0.115 [+0.066, +0.178] | +0.140 [+0.048, +0.256] | -0.000 [-0.000, +0.000] |
| nostro s1234, LDA, d' | +0.053 [+0.020, +0.087] | +0.159 [+0.110, +0.208] | +0.129 [+0.075, +0.188] | +0.098 [+0.062, +0.134] | +0.167 [+0.108, +0.235] | +0.140 [+0.048, +0.256] | +0.000 [+0.000, +0.000] |
| nostro s1234, d_F ricomposta: taglia + u WCCN | +0.053 [+0.020, +0.087] | +0.092 [+0.032, +0.148] | +0.052 [+0.007, +0.105] | +0.054 [+0.018, +0.096] | +0.089 [+0.029, +0.167] | +0.140 [+0.048, +0.256] | -0.000 [-0.000, +0.000] |
| nostro s1234, d_F ricomposta: taglia + u LDA | +0.053 [+0.020, +0.087] | +0.156 [+0.107, +0.206] | +0.119 [+0.071, +0.180] | +0.096 [+0.061, +0.133] | +0.164 [+0.104, +0.233] | +0.140 [+0.048, +0.256] | +0.000 [+0.000, +0.000] |
| nostro s1234, controllo: d_F con u non cal., c scelto su FaMoS TRAIN | -0.073 [-0.113, -0.033] | -0.150 [-0.205, -0.095] | -0.071 [-0.121, -0.019] | -0.065 [-0.109, -0.025] | -0.106 [-0.168, -0.051] | -0.160 [-0.283, -0.047] | +0.000 [+0.000, +0.000] |
| nostro s1234, d_F non cal. (non cal.) | -0.093 [-0.133, -0.047] | -0.158 [-0.222, -0.098] | -0.068 [-0.120, -0.017] | -0.065 [-0.110, -0.025] | -0.108 [-0.167, -0.052] | -0.220 [-0.372, -0.091] | +0.000 [+0.000, +0.000] |

## Delta: varianti s1234 contro ICP + Chamfer in mm, configurazione scelta su FaMoS TRAIN (rej_pulli)

positivo = la variante meglio (EER: riferimento - variante); stesse repliche.

| variante | rank-1 | mAP | EER | AUC | AUC H | rank-1 H | AUC L |
|---|---|---|---|---|---|---|---|
| nostro s1234, WCCN (u sbiancata), d' | +0.047 [+0.020, +0.073] | +0.098 [+0.050, +0.147] | +0.066 [+0.021, +0.119] | +0.062 [+0.032, +0.096] | +0.106 [+0.058, +0.167] | +0.120 [+0.047, +0.213] | -0.000 [-0.000, +0.000] |
| nostro s1234, LDA, d' | +0.047 [+0.020, +0.073] | +0.151 [+0.103, +0.198] | +0.126 [+0.071, +0.183] | +0.092 [+0.056, +0.128] | +0.157 [+0.098, +0.225] | +0.120 [+0.047, +0.213] | +0.000 [+0.000, +0.000] |
| nostro s1234, d_F ricomposta: taglia + u WCCN | +0.047 [+0.020, +0.073] | +0.084 [+0.025, +0.139] | +0.049 [+0.003, +0.101] | +0.048 [+0.014, +0.088] | +0.080 [+0.022, +0.156] | +0.120 [+0.047, +0.213] | -0.000 [-0.000, +0.000] |
| nostro s1234, d_F ricomposta: taglia + u LDA | +0.047 [+0.020, +0.073] | +0.148 [+0.099, +0.196] | +0.116 [+0.064, +0.175] | +0.090 [+0.055, +0.126] | +0.154 [+0.095, +0.223] | +0.120 [+0.047, +0.213] | +0.000 [+0.000, +0.000] |
| nostro s1234, controllo: d_F con u non cal., c scelto su FaMoS TRAIN | -0.080 [-0.113, -0.047] | -0.158 [-0.212, -0.104] | -0.074 [-0.127, -0.026] | -0.071 [-0.115, -0.032] | -0.116 [-0.177, -0.059] | -0.180 [-0.286, -0.091] | +0.000 [+0.000, +0.000] |
| nostro s1234, d_F non cal. (non cal.) | -0.100 [-0.133, -0.067] | -0.166 [-0.229, -0.105] | -0.071 [-0.126, -0.022] | -0.072 [-0.115, -0.031] | -0.117 [-0.177, -0.061] | -0.240 [-0.375, -0.136] | +0.000 [+0.000, +0.000] |

## Delta: varianti s1234 contro NoW metrical (rilevatore), default (non cal.)

positivo = la variante meglio (EER: riferimento - variante); stesse repliche.

| variante | rank-1 | mAP | EER | AUC | AUC H | rank-1 H | AUC L |
|---|---|---|---|---|---|---|---|
| nostro s1234, WCCN (u sbiancata), d' | +0.033 [+0.007, +0.060] | +0.064 [+0.028, +0.110] | +0.053 [+0.012, +0.096] | +0.046 [+0.020, +0.074] | +0.084 [+0.038, +0.139] | +0.100 [+0.023, +0.182] | -0.000 [-0.000, +0.000] |
| nostro s1234, LDA, d' | +0.033 [+0.007, +0.060] | +0.117 [+0.075, +0.165] | +0.113 [+0.059, +0.163] | +0.075 [+0.040, +0.112] | +0.135 [+0.075, +0.207] | +0.100 [+0.023, +0.182] | +0.000 [+0.000, +0.000] |
| nostro s1234, d_F ricomposta: taglia + u WCCN | +0.033 [+0.007, +0.060] | +0.050 [+0.002, +0.100] | +0.036 [-0.011, +0.081] | +0.031 [-0.001, +0.067] | +0.058 [-0.001, +0.126] | +0.100 [+0.023, +0.182] | -0.000 [-0.000, +0.000] |
| nostro s1234, d_F ricomposta: taglia + u LDA | +0.033 [+0.007, +0.060] | +0.114 [+0.071, +0.162] | +0.102 [+0.056, +0.157] | +0.073 [+0.038, +0.110] | +0.132 [+0.073, +0.203] | +0.100 [+0.023, +0.182] | +0.000 [+0.000, +0.000] |
| nostro s1234, controllo: d_F con u non cal., c scelto su FaMoS TRAIN | -0.093 [-0.127, -0.060] | -0.192 [-0.237, -0.141] | -0.087 [-0.130, -0.047] | -0.088 [-0.122, -0.057] | -0.138 [-0.190, -0.093] | -0.200 [-0.311, -0.108] | +0.000 [+0.000, +0.000] |
| nostro s1234, d_F non cal. (non cal.) | -0.113 [-0.147, -0.073] | -0.200 [-0.253, -0.144] | -0.085 [-0.132, -0.044] | -0.088 [-0.123, -0.056] | -0.139 [-0.192, -0.096] | -0.260 [-0.400, -0.156] | +0.000 [+0.000, +0.000] |

## Delta: varianti s1234 contro NoW metrical (rilevatore), configurazione scelta su FaMoS TRAIN (lmk68)

positivo = la variante meglio (EER: riferimento - variante); stesse repliche.

| variante | rank-1 | mAP | EER | AUC | AUC H | rank-1 H | AUC L |
|---|---|---|---|---|---|---|---|
| nostro s1234, WCCN (u sbiancata), d' | +0.033 [+0.007, +0.060] | +0.064 [+0.028, +0.110] | +0.053 [+0.012, +0.096] | +0.046 [+0.020, +0.075] | +0.084 [+0.038, +0.140] | +0.100 [+0.023, +0.182] | -0.000 [-0.000, +0.000] |
| nostro s1234, LDA, d' | +0.033 [+0.007, +0.060] | +0.117 [+0.075, +0.165] | +0.113 [+0.059, +0.163] | +0.075 [+0.040, +0.112] | +0.136 [+0.075, +0.207] | +0.100 [+0.023, +0.182] | +0.000 [+0.000, +0.000] |
| nostro s1234, d_F ricomposta: taglia + u WCCN | +0.033 [+0.007, +0.060] | +0.050 [+0.002, +0.100] | +0.036 [-0.011, +0.081] | +0.031 [-0.001, +0.067] | +0.058 [-0.001, +0.126] | +0.100 [+0.023, +0.182] | -0.000 [-0.000, +0.000] |
| nostro s1234, d_F ricomposta: taglia + u LDA | +0.033 [+0.007, +0.060] | +0.114 [+0.071, +0.163] | +0.102 [+0.056, +0.157] | +0.073 [+0.038, +0.110] | +0.132 [+0.073, +0.203] | +0.100 [+0.023, +0.182] | +0.000 [+0.000, +0.000] |
| nostro s1234, controllo: d_F con u non cal., c scelto su FaMoS TRAIN | -0.093 [-0.127, -0.060] | -0.192 [-0.237, -0.141] | -0.087 [-0.130, -0.047] | -0.088 [-0.121, -0.057] | -0.137 [-0.190, -0.093] | -0.200 [-0.311, -0.108] | +0.000 [+0.000, +0.000] |
| nostro s1234, d_F non cal. (non cal.) | -0.113 [-0.147, -0.073] | -0.200 [-0.253, -0.143] | -0.085 [-0.132, -0.044] | -0.088 [-0.123, -0.056] | -0.139 [-0.192, -0.096] | -0.260 [-0.400, -0.156] | +0.000 [+0.000, +0.000] |

## Delta: varianti s2345 contro nostro s2345, d_F non cal. (non cal.)

positivo = la variante meglio (EER: riferimento - variante); stesse repliche.

| variante | rank-1 | mAP | EER | AUC | AUC H | rank-1 H | AUC L |
|---|---|---|---|---|---|---|---|
| nostro s2345, WCCN (u sbiancata), d' | +0.180 [+0.120, +0.233] | +0.279 [+0.209, +0.337] | +0.141 [+0.081, +0.207] | +0.146 [+0.089, +0.197] | +0.243 [+0.161, +0.328] | +0.460 [+0.281, +0.667] | +0.000 [+0.000, +0.000] |
| nostro s2345, LDA, d' | +0.173 [+0.120, +0.227] | +0.336 [+0.264, +0.401] | +0.222 [+0.162, +0.276] | +0.182 [+0.125, +0.235] | +0.306 [+0.221, +0.391] | +0.440 [+0.268, +0.643] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F ricomposta: taglia + u WCCN | +0.180 [+0.120, +0.233] | +0.261 [+0.191, +0.325] | +0.125 [+0.058, +0.194] | +0.129 [+0.065, +0.190] | +0.211 [+0.112, +0.316] | +0.460 [+0.281, +0.667] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F ricomposta: taglia + u LDA | +0.167 [+0.100, +0.227] | +0.333 [+0.262, +0.396] | +0.216 [+0.154, +0.271] | +0.180 [+0.123, +0.234] | +0.303 [+0.218, +0.389] | +0.420 [+0.241, +0.634] | +0.000 [+0.000, +0.000] |
| nostro s2345, controllo: d_F con u non cal., c scelto su FaMoS TRAIN | +0.013 [+0.000, +0.040] | +0.009 [-0.002, +0.022] | +0.001 [-0.013, +0.012] | +0.002 [-0.007, +0.012] | +0.005 [-0.011, +0.020] | +0.040 [+0.000, +0.146] | +0.000 [+0.000, +0.000] |

## Delta: varianti s2345 contro ICP + Chamfer in mm, default (non cal.)

positivo = la variante meglio (EER: riferimento - variante); stesse repliche.

| variante | rank-1 | mAP | EER | AUC | AUC H | rank-1 H | AUC L |
|---|---|---|---|---|---|---|---|
| nostro s2345, WCCN (u sbiancata), d' | +0.053 [+0.020, +0.087] | +0.098 [+0.037, +0.155] | +0.055 [+0.006, +0.117] | +0.062 [+0.024, +0.105] | +0.103 [+0.041, +0.182] | +0.140 [+0.048, +0.256] | +0.000 [+0.000, +0.000] |
| nostro s2345, LDA, d' | +0.047 [+0.020, +0.073] | +0.155 [+0.110, +0.198] | +0.136 [+0.080, +0.190] | +0.098 [+0.063, +0.134] | +0.167 [+0.107, +0.234] | +0.120 [+0.047, +0.213] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F ricomposta: taglia + u WCCN | +0.053 [+0.020, +0.087] | +0.080 [+0.010, +0.145] | +0.039 [-0.014, +0.103] | +0.045 [-0.004, +0.097] | +0.072 [-0.010, +0.168] | +0.140 [+0.048, +0.256] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F ricomposta: taglia + u LDA | +0.040 [+0.007, +0.067] | +0.152 [+0.106, +0.194] | +0.130 [+0.074, +0.185] | +0.096 [+0.061, +0.133] | +0.163 [+0.103, +0.233] | +0.100 [+0.000, +0.200] | +0.000 [+0.000, +0.000] |
| nostro s2345, controllo: d_F con u non cal., c scelto su FaMoS TRAIN | -0.113 [-0.160, -0.067] | -0.172 [-0.234, -0.114] | -0.085 [-0.138, -0.040] | -0.082 [-0.128, -0.039] | -0.134 [-0.205, -0.072] | -0.280 [-0.447, -0.157] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F non cal. (non cal.) | -0.127 [-0.173, -0.080] | -0.181 [-0.247, -0.121] | -0.086 [-0.139, -0.039] | -0.084 [-0.131, -0.040] | -0.139 [-0.208, -0.074] | -0.320 [-0.500, -0.186] | +0.000 [+0.000, +0.000] |

## Delta: varianti s2345 contro ICP + Chamfer in mm, configurazione scelta su FaMoS TRAIN (rej_pulli)

positivo = la variante meglio (EER: riferimento - variante); stesse repliche.

| variante | rank-1 | mAP | EER | AUC | AUC H | rank-1 H | AUC L |
|---|---|---|---|---|---|---|---|
| nostro s2345, WCCN (u sbiancata), d' | +0.047 [+0.020, +0.073] | +0.090 [+0.031, +0.146] | +0.052 [+0.003, +0.113] | +0.056 [+0.019, +0.098] | +0.094 [+0.032, +0.172] | +0.120 [+0.047, +0.213] | +0.000 [+0.000, +0.000] |
| nostro s2345, LDA, d' | +0.040 [+0.013, +0.067] | +0.147 [+0.103, +0.189] | +0.133 [+0.077, +0.187] | +0.091 [+0.057, +0.127] | +0.157 [+0.098, +0.223] | +0.100 [+0.034, +0.191] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F ricomposta: taglia + u WCCN | +0.047 [+0.020, +0.073] | +0.072 [+0.002, +0.135] | +0.036 [-0.018, +0.098] | +0.038 [-0.009, +0.089] | +0.062 [-0.017, +0.155] | +0.120 [+0.047, +0.213] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F ricomposta: taglia + u LDA | +0.033 [+0.000, +0.060] | +0.144 [+0.098, +0.186] | +0.127 [+0.073, +0.181] | +0.090 [+0.055, +0.126] | +0.154 [+0.095, +0.222] | +0.080 [+0.000, +0.174] | +0.000 [+0.000, +0.000] |
| nostro s2345, controllo: d_F con u non cal., c scelto su FaMoS TRAIN | -0.120 [-0.167, -0.073] | -0.180 [-0.241, -0.123] | -0.088 [-0.141, -0.044] | -0.088 [-0.134, -0.046] | -0.144 [-0.216, -0.081] | -0.300 [-0.468, -0.167] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F non cal. (non cal.) | -0.133 [-0.180, -0.087] | -0.189 [-0.255, -0.129] | -0.089 [-0.144, -0.043] | -0.090 [-0.137, -0.047] | -0.149 [-0.218, -0.085] | -0.340 [-0.513, -0.196] | +0.000 [+0.000, +0.000] |

## Delta: varianti s2345 contro NoW metrical (rilevatore), default (non cal.)

positivo = la variante meglio (EER: riferimento - variante); stesse repliche.

| variante | rank-1 | mAP | EER | AUC | AUC H | rank-1 H | AUC L |
|---|---|---|---|---|---|---|---|
| nostro s2345, WCCN (u sbiancata), d' | +0.033 [+0.007, +0.060] | +0.056 [+0.011, +0.102] | +0.039 [-0.007, +0.089] | +0.039 [+0.007, +0.074] | +0.072 [+0.016, +0.139] | +0.100 [+0.023, +0.182] | +0.000 [+0.000, +0.000] |
| nostro s2345, LDA, d' | +0.027 [+0.007, +0.047] | +0.113 [+0.076, +0.150] | +0.120 [+0.068, +0.164] | +0.075 [+0.041, +0.110] | +0.135 [+0.076, +0.204] | +0.080 [+0.018, +0.159] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F ricomposta: taglia + u WCCN | +0.033 [+0.007, +0.060] | +0.038 [-0.019, +0.091] | +0.023 [-0.029, +0.077] | +0.022 [-0.021, +0.065] | +0.040 [-0.034, +0.122] | +0.100 [+0.023, +0.182] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F ricomposta: taglia + u LDA | +0.020 [-0.007, +0.047] | +0.110 [+0.073, +0.146] | +0.114 [+0.064, +0.161] | +0.073 [+0.039, +0.109] | +0.132 [+0.073, +0.201] | +0.060 [-0.022, +0.149] | +0.000 [+0.000, +0.000] |
| nostro s2345, controllo: d_F con u non cal., c scelto su FaMoS TRAIN | -0.133 [-0.180, -0.087] | -0.214 [-0.266, -0.161] | -0.101 [-0.149, -0.067] | -0.105 [-0.139, -0.072] | -0.166 [-0.225, -0.117] | -0.320 [-0.488, -0.193] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F non cal. (non cal.) | -0.147 [-0.193, -0.100] | -0.223 [-0.281, -0.168] | -0.102 [-0.150, -0.065] | -0.107 [-0.142, -0.073] | -0.171 [-0.229, -0.124] | -0.360 [-0.535, -0.216] | +0.000 [+0.000, +0.000] |

## Delta: varianti s2345 contro NoW metrical (rilevatore), configurazione scelta su FaMoS TRAIN (lmk68)

positivo = la variante meglio (EER: riferimento - variante); stesse repliche.

| variante | rank-1 | mAP | EER | AUC | AUC H | rank-1 H | AUC L |
|---|---|---|---|---|---|---|---|
| nostro s2345, WCCN (u sbiancata), d' | +0.033 [+0.007, +0.060] | +0.056 [+0.011, +0.102] | +0.039 [-0.007, +0.089] | +0.039 [+0.007, +0.074] | +0.072 [+0.016, +0.139] | +0.100 [+0.023, +0.182] | +0.000 [+0.000, +0.000] |
| nostro s2345, LDA, d' | +0.027 [+0.007, +0.047] | +0.113 [+0.076, +0.150] | +0.120 [+0.068, +0.164] | +0.075 [+0.041, +0.110] | +0.135 [+0.076, +0.204] | +0.080 [+0.018, +0.159] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F ricomposta: taglia + u WCCN | +0.033 [+0.007, +0.060] | +0.038 [-0.019, +0.091] | +0.023 [-0.029, +0.077] | +0.022 [-0.021, +0.065] | +0.040 [-0.033, +0.122] | +0.100 [+0.023, +0.182] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F ricomposta: taglia + u LDA | +0.020 [-0.007, +0.047] | +0.110 [+0.073, +0.147] | +0.114 [+0.064, +0.160] | +0.073 [+0.039, +0.109] | +0.132 [+0.073, +0.201] | +0.060 [-0.022, +0.149] | +0.000 [+0.000, +0.000] |
| nostro s2345, controllo: d_F con u non cal., c scelto su FaMoS TRAIN | -0.133 [-0.180, -0.087] | -0.214 [-0.266, -0.161] | -0.101 [-0.149, -0.067] | -0.105 [-0.139, -0.072] | -0.166 [-0.225, -0.117] | -0.320 [-0.488, -0.193] | +0.000 [+0.000, +0.000] |
| nostro s2345, d_F non cal. (non cal.) | -0.147 [-0.193, -0.100] | -0.223 [-0.281, -0.168] | -0.102 [-0.150, -0.065] | -0.107 [-0.142, -0.073] | -0.171 [-0.229, -0.123] | -0.360 [-0.535, -0.216] | +0.000 [+0.000, +0.000] |

## Controlli

- `{"griglia_icp_default_vs_esterna": {"n": 11175, "nan": 0, "max_abs": 9.414691248821327e-14, "median_abs": 4.440892098500626e-15, "p95_abs": 2.3092638912203256e-14, "equal": 674}, "griglia_now_default_vs_esterna": {"n": 11175, "nan": 0, "max_abs": 0.0, "median_abs": 0.0, "p95_abs": 0.0, "equal": 11175}, "factorized_s1234_form_ricalcolata_vs_esterna": {"n": 11175, "nan": 0, "max_abs": 1.2434497875801753e-14, "median_abs": 0.0, "p95_abs": 3.552713678800501e-15, "equal": 8883}, "factorized_s2345_form_ricalcolata_vs_esterna": {"n": 11175, "nan": 0, "max_abs": 1.2434497875801753e-14, "median_abs": 0.0, "p95_abs": 3.552713678800501e-15, "equal": 8100}, "metriche_di_base_vs_esterna_max_abs": 2.220446049250313e-16}`
- controlli_ok: `{"embed_factorized_s1234": true, "embed_factorized_s2345": true}`

File: `cv.csv` (tutte le configurazioni, AUC per fold), `metrics.csv`, `deltas.csv`, `summary.json`, `selection.json`, `folds.json`, `prep_calib.json`, `controls/`.

<!-- da qui in giu': scritto a mano, conservato da fcal.py eval -->

## Lettura (REGOLE.md, sezione 6; POST HOC e DESCRITTIVA)

**Variante primaria** (AUC di validazione incrociata massima su FaMoS TRAIN fra le quattro): **LDA** in entrambi i semi
(lam 0,01, k 16; AUC cv 0,998 e 0,995).

- **Guadagno**: SI', entrambi i semi. AUC 0,830 -> 0,993 [0,983, 0,999] (s1234) e 0,811 -> 0,993 [0,983, 0,999] (s2345);
  delta contro la d_F non cal. +0,163 [+0,111, +0,210] e +0,182 [+0,125, +0,235]; nello strato H +0,275 e +0,306.
- **Divario chiuso**: SI', e ribaltato, entrambi i semi, contro ICP mm e NoW metrical, sia di default sia nella
  configurazione scelta su calibrazione (ICP: rej_pulli; NoW: lmk68, numeri identici al default). AUC contro NoW
  metrical +0,075 [+0,040, +0,112] (s1234) e +0,075 [+0,041, +0,110] (s2345); contro ICP mm scelto +0,092 e +0,091;
  strato H contro NoW metrical +0,135 e +0,135. Rank-1 1,000 e 0,993 (NoW 0,967), EER 0,044 e 0,037 (NoW 0,157).
- WCCN da sola: anch'essa sopra entrambi i concorrenti in AUC (0,964 / 0,957). Le varianti con la taglia (c) NON
  aiutano: la validazione incrociata sceglie g = 16, il bordo della griglia (peso della taglia quasi nullo), e
  WCCN + taglia peggiora rispetto a WCCN sola. Il controllo d_F con c riscelto su FaMoS TRAIN (c = 0,707) resta dov'era
  (delta AUC +0,000 e +0,002): il guadagno viene dalla calibrazione di u, non dal ripesare la taglia.

## Cautele

- **Non e' un confronto alla pari nel senso dei dati**: la variante calibrata usa 73 persone FaMoS TRAIN con etichetta
  d'identita' (adattamento supervisionato al dominio, stesso sistema di acquisizione); ICP e NoW non hanno nulla da
  stimare oltre alla configurazione. E' la misura di quanto una correzione lineare post hoc, stimata su persone diverse,
  basta: non la prestazione del modello cosi' com'e'.
- Calibrazione su registrazioni (superfici pulite), test su scansioni grezze: il trasferimento regge (AUC cv 0,998 ->
  test 0,993), ma la differenza di dominio resta una delle condizioni dell'esperimento.
- Gli IC ricampionano i 15 soggetti di test, non i soggetti di calibrazione; con 15 persone e rank-1 a 1,000 gli IC al
  tetto sono degeneri.

## Verifiche fuori pipeline (job interattivi su prioritized, script fuori da git)

- Nessun soggetto di test fra quelli di calibrazione (intersezione vuota dei view_id).
- LDA rifatta con `sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.01,
  n_components=16)` sulle stesse u: AUC di test 0,9929 / 0,9927, uguale a `fcal.fit` (shrinkage 0: 0,9905 / 0,9873;
  'auto': 0,989 / 0,985).
- Controllo negativo: LDA e WCCN stimate con le etichette d'identita' di calibrazione PERMUTATE (3 permutazioni): AUC di
  test 0,744-0,831 (LDA) e 0,811-0,839 (WCCN), contro 0,830 / 0,808 della u grezza: senza l'identita' il guadagno
  sparisce.

## Modifiche dopo il commit delle regole (d6d6cf3)

- `aau/famos_calib/det_threads.py` (nuovo) e `fcal.sbatch`: il rilevatore a blocchi in parallelo apriva un thread
  onnxruntime per core della macchina (carico 1311 su 32 core, job 1082383 fermato); l'involucro fissa
  `intra_op_num_threads` = 4 per blocco. Stessi modelli e immagini; 0 ripieghi. `prep` reso ripartibile. Nessuna
  modifica a `fcal.py` dopo d6d6cf3.
- Job: preparazione 1082398 (10 min), latenti 1082402 (V100, 5 min; controllo sulle 150 di test max |dZ| 1,2e-3 e
  1,1e-3 <= 5e-3), griglia 1082403 (array 0-9, 64 core, 2-8 min l'uno, 0 NaN), valutazione 1082444 (4 min).
