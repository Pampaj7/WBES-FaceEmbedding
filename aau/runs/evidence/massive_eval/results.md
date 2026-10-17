# Modelli massivi: valutazione con la pipeline della riga "10x identities"

Generato da `aau/massive_eval/me_results.py` (fonti: `metrics.csv`, colonna `fonte` = file:riga di ogni numero). Checkpoint finali EMA: principale `main_a100` epoch330 (330k passi, job 1075583 + 1078908), aperto `release_open` epoch230 (230k passi, job 1079557). Nessun numero di questi modelli e' nelle affermazioni del paper (PLAN_MASSIVE sez. 23); Ava-256 SOLO DESCRITTIVO, fuori dalla confermativa (prova congelata non toccata).

## Calibrazione c (fissata e committata prima di ogni valutazione)

`fact_calib.calib_one` sull'held-out sintetico dei generatori di training del modello (`me_calib.py`): coppie stesso dominio, identita' ed etichette diverse, senza crop, viste neutre; c = mediana d_P GT / mediana d_P modello.

| modello | c | c_LS | c con espressioni | c per dominio | catena del paper (held-out C3M bfm/ict/gnm), controllo | fonte |
|---|---|---|---|---|---|---|
| massive_main | 0.4258 | 0.4618 | 0.4203 | bfm2019 0.460, ict 0.378, gnm 0.430, flame2023 0.389, hifi3d 0.487, faceverse 0.615, facescape 0.340 | 0.4045 (bfm 0.416, ict 0.370, gnm 0.425) | `aau/runs/evidence/massive_eval/calib/main.json` |
| massive_open | 0.2686 | 0.2710 | 0.2666 | gnm 0.282, ict 0.257, flame2023 0.264 | 0.2694 (bfm 0.272, ict 0.255, gnm 0.280) | `aau/runs/evidence/massive_eval/calib/open.json` |

Riferimento: modello del paper s1234 c = 0.4053, C3M e123 c = 0.2573 (`factorized_calibration.csv`).

## Contaminazione (dalla configurazione del training)

- **massive_main** (principale (release, 8 fonti), 330k passi): fonti faceverse (106488 viste), gnm (107694 viste), ict (106524 viste), facescape (106956 viste), bfm2019 (107634 viste), flame2023 (108246 viste), hifi3d (106764 viste), famos (106716 viste) (`aau/runs/massive_release/main_a100/node0/producers.json (views_by_domain)`; ruoli, purpose e persone FaMoS escluse in `aau/runs/massive_release/main_a100/node0/producers.log:7`).
  - HIFI3D: **IN-DOMAIN**. generatore 'hifi3d' nel training (106764 viste): stesso 3DMM delle identita' di valutazione; identita' esatte non condivise per costruzione (coefficienti continui, semi del training indipendenti; non verificato coefficiente per coefficiente).
  - FaceScape: **IN-DOMAIN**. generatore 'facescape' nel training (106956 viste): stesso 3DMM delle identita' di valutazione; identita' esatte non condivise per costruzione (coefficienti continui, semi del training indipendenti; non verificato coefficiente per coefficiente).
  - FaceVerse: **IN-DOMAIN**. generatore 'faceverse' nel training (106488 viste): stesso 3DMM delle identita' di valutazione; identita' esatte non condivise per costruzione (coefficienti continui, semi del training indipendenti; non verificato coefficiente per coefficiente).
  - FV neutral: **IN-DOMAIN**. generatore 'faceverse' nel training (106488 viste): stesso 3DMM delle identita' di valutazione; identita' esatte non condivise per costruzione (coefficienti continui, semi del training indipendenti; non verificato coefficiente per coefficiente).
  - FaMoS: **IN-DOMAIN**. FaMoS TRAIN nel training (106716 viste; 74 persone, 6 held-out); stesso sistema di cattura e registrazione. Persone di TEST escluse (FamosSource; persone TRAIN e TEST in comune: 0, aau/famos/split.json): nessuna identita' condivisa.
  - Ava-256 (descr.): **zero-shot**. nessuna sorgente Ava-256 (esclusa sempre, PLAN_MASSIVE sez. 23.1).
- **massive_open** (aperto (GNM, ICT, FLAME 2023 Open), 230k passi): fonti gnm (324240 viste), flame2023 (323982 viste), ict (324288 viste) (`aau/runs/massive_release/release_open/node0/producers.json (views_by_domain)`; ruoli, purpose e persone FaMoS escluse in `aau/runs/massive_release/release_open/node0/producers.log:6`).
  - HIFI3D: **zero-shot**. generatore del dominio non nel training (fonti: flame2023, gnm, ict).
  - FaceScape: **zero-shot**. generatore del dominio non nel training (fonti: flame2023, gnm, ict).
  - FaceVerse: **zero-shot**. generatore del dominio non nel training (fonti: flame2023, gnm, ict).
  - FV neutral: **zero-shot**. generatore del dominio non nel training (fonti: flame2023, gnm, ict).
  - FaMoS: **vicino**. zero-shot per persone e dataset (nessuna vista FaMoS), ma FLAME 2023 Open nel training (323982 viste): FaMoS e' registrato su FLAME (topologia e GT).
  - Ava-256 (descr.): **zero-shot**. nessuna sorgente Ava-256 (esclusa sempre, PLAN_MASSIVE sez. 23.1).
- Per entrambi (e per i modelli del paper): HIFI3D, FaceVerse e dev FaceScape sono stati usati nella scelta della ricetta (PLAN_MASSIVE sez. 22.6).

## Matrice di sviluppo (nocrop, passo comune; IC 95%, 1000 repliche per soggetto, semi della matrice)

Celle marcate: `[IN]` = in-domain, `[~]` = dominio vicino; senza marca = zero-shot.

### massive_main

| lettura | GT | HIFI3D | FaceScape | FaceVerse | FV neutral | FaMoS | Ava-256 (descr.) |
|---|---|---|---|---|---|---|---|
| d_F cal., rho | FR | 0.902 [0.864, 0.931] [IN] | 0.870 [0.841, 0.894] [IN] | 0.180 [0.140, 0.224] [IN] | 0.192 [0.149, 0.236] [IN] | 0.703 [0.311, 0.906] [IN] | 0.735 [0.690, 0.775] |
| d_F cal., delta vs paper s1234 | FR | +0.153 [+0.112, +0.203] [IN] | +0.209 [+0.159, +0.262] [IN] | -0.123 [-0.187, -0.053] [IN] | -0.149 [-0.222, -0.074] [IN] | +0.025 [-0.131, +0.232] [IN] | +0.094 [+0.054, +0.137] |
| d_F cal., delta vs ICP mm | FR | +0.259 [+0.215, +0.310] [IN] | +0.416 [+0.366, +0.467] [IN] | -0.157 [-0.224, -0.090] [IN] | -0.217 [-0.284, -0.146] [IN] | -0.036 [-0.175, +0.102] [IN] | +0.111 [+0.063, +0.159] |
| d_F c=1, rho | FR | 0.827 [0.788, 0.862] [IN] | 0.844 [0.803, 0.879] [IN] | 0.184 [0.139, 0.229] [IN] | 0.192 [0.142, 0.242] [IN] | 0.553 [0.143, 0.821] [IN] | 0.660 [0.606, 0.706] |
| d_F c=1, delta vs paper s1234 | FR | +0.185 [+0.131, +0.246] [IN] | +0.156 [+0.107, +0.210] [IN] | -0.113 [-0.173, -0.049] [IN] | -0.145 [-0.217, -0.073] [IN] | -0.063 [-0.311, +0.130] [IN] | +0.077 [+0.041, +0.116] |
| d_F c=1, delta vs ICP mm | FR | +0.184 [+0.130, +0.246] [IN] | +0.390 [+0.346, +0.441] [IN] | -0.153 [-0.219, -0.083] [IN] | -0.217 [-0.284, -0.146] [IN] | -0.186 [-0.414, +0.038] [IN] | +0.036 [-0.011, +0.087] |
| d_P, rho | SR | 0.870 [0.836, 0.900] [IN] | 0.921 [0.899, 0.939] [IN] | 0.207 [0.161, 0.252] [IN] | 0.208 [0.156, 0.258] [IN] | 0.613 [0.333, 0.819] [IN] | 0.707 [0.653, 0.754] |
| d_P, delta vs paper s1234 | SR | +0.249 [+0.189, +0.315] [IN] | +0.174 [+0.129, +0.231] [IN] | -0.079 [-0.140, -0.014] [IN] | -0.126 [-0.197, -0.053] [IN] | -0.128 [-0.338, +0.019] [IN] | +0.044 [+0.011, +0.077] |
| d_P, delta vs ICP mm | SR | +0.505 [+0.424, +0.586] [IN] | +0.444 [+0.396, +0.496] [IN] | -0.102 [-0.163, -0.035] [IN] | -0.175 [-0.239, -0.107] [IN] | +0.064 [-0.314, +0.486] [IN] | +0.139 [+0.074, +0.199] |

### massive_open

| lettura | GT | HIFI3D | FaceScape | FaceVerse | FV neutral | FaMoS | Ava-256 (descr.) |
|---|---|---|---|---|---|---|---|
| d_F cal., rho | FR | 0.759 [0.676, 0.822] | 0.675 [0.613, 0.725] | 0.170 [0.113, 0.224] | 0.167 [0.112, 0.219] | 0.749 [0.476, 0.910] [~] | 0.735 [0.687, 0.775] |
| d_F cal., delta vs paper s1234 | FR | +0.011 [-0.024, +0.045] | +0.014 [-0.043, +0.069] | -0.133 [-0.194, -0.072] | -0.174 [-0.231, -0.115] | +0.071 [-0.099, +0.327] [~] | +0.094 [+0.055, +0.136] |
| d_F cal., delta vs ICP mm | FR | +0.116 [+0.066, +0.165] | +0.221 [+0.162, +0.282] | -0.167 [-0.227, -0.108] | -0.242 [-0.298, -0.179] | +0.010 [-0.149, +0.223] [~] | +0.111 [+0.061, +0.161] |
| d_F c=1, rho | FR | 0.546 [0.459, 0.626] | 0.663 [0.597, 0.721] | 0.172 [0.119, 0.222] | 0.177 [0.119, 0.227] | 0.563 [0.235, 0.783] [~] | 0.571 [0.505, 0.630] |
| d_F c=1, delta vs paper s1234 | FR | -0.096 [-0.149, -0.051] | -0.025 [-0.059, +0.010] | -0.124 [-0.180, -0.073] | -0.160 [-0.220, -0.104] | -0.054 [-0.221, +0.128] [~] | -0.012 [-0.045, +0.020] |
| d_F c=1, delta vs ICP mm | FR | -0.097 [-0.169, -0.027] | +0.209 [+0.164, +0.256] | -0.165 [-0.226, -0.101] | -0.232 [-0.294, -0.162] | -0.177 [-0.452, +0.088] [~] | -0.053 [-0.113, +0.004] |
| d_P, rho | SR | 0.567 [0.476, 0.648] | 0.726 [0.669, 0.775] | 0.175 [0.120, 0.226] | 0.180 [0.125, 0.231] | 0.726 [0.514, 0.879] [~] | 0.679 [0.620, 0.731] |
| d_P, delta vs paper s1234 | SR | -0.054 [-0.109, -0.005] | -0.020 [-0.051, +0.012] | -0.111 [-0.165, -0.059] | -0.153 [-0.211, -0.096] | -0.015 [-0.135, +0.083] [~] | +0.016 [-0.017, +0.046] |
| d_P, delta vs ICP mm | SR | +0.202 [+0.120, +0.284] | +0.249 [+0.202, +0.298] | -0.135 [-0.196, -0.075] | -0.203 [-0.266, -0.136] | +0.177 [-0.182, +0.561] [~] | +0.111 [+0.041, +0.172] |

## FaMoS TEST, riconoscimento (150 scansioni grezze, 15 persone; `ei.py`)

Delta: positivo = massivo meglio (EER: concorrente - massivo).

### massive_main (IN-DOMAIN)

| lettura | | rank-1 | mAP | AUC | EER |
|---|---|---|---|---|---|
| d_F cal. | valore | 0.880 [0.820, 0.933] | 0.722 [0.642, 0.798] | 0.866 [0.807, 0.920] | 0.217 [0.156, 0.274] |
| d_F cal. | delta vs paper s1234 | +0.007 [-0.053, +0.053] | +0.071 [+0.029, +0.114] | +0.050 [+0.021, +0.081] | +0.048 [+0.019, +0.083] |
| d_F cal. | delta vs ICP mm | -0.067 [-0.120, -0.013] | -0.100 [-0.170, -0.040] | -0.029 [-0.078, +0.009] | -0.044 [-0.102, +0.015] |
| d_F c=1 | valore | 0.893 [0.840, 0.940] | 0.720 [0.646, 0.787] | 0.855 [0.806, 0.908] | 0.216 [0.163, 0.271] |
| d_F c=1 | delta vs paper s1234 | +0.040 [-0.013, +0.080] | +0.056 [+0.010, +0.098] | +0.025 [-0.000, +0.052] | +0.026 [-0.011, +0.055] |
| d_F c=1 | delta vs ICP mm | -0.053 [-0.107, -0.007] | -0.102 [-0.178, -0.028] | -0.040 [-0.088, +0.008] | -0.043 [-0.098, +0.016] |
| d_P | valore | 0.907 [0.860, 0.953] | 0.684 [0.611, 0.753] | 0.842 [0.792, 0.895] | 0.225 [0.176, 0.279] |
| d_P | delta vs paper s1234 | +0.093 [+0.040, +0.147] | +0.048 [+0.000, +0.090] | +0.012 [-0.013, +0.037] | +0.016 [-0.016, +0.041] |
| d_P | delta vs ICP mm | -0.040 [-0.093, +0.020] | -0.139 [-0.217, -0.059] | -0.053 [-0.106, -0.002] | -0.052 [-0.113, +0.006] |

### massive_open (vicino)

| lettura | | rank-1 | mAP | AUC | EER |
|---|---|---|---|---|---|
| d_F cal. | valore | 0.887 [0.840, 0.933] | 0.692 [0.610, 0.771] | 0.860 [0.806, 0.909] | 0.222 [0.172, 0.274] |
| d_F cal. | delta vs paper s1234 | +0.013 [-0.040, +0.067] | +0.041 [-0.002, +0.082] | +0.045 [+0.016, +0.072] | +0.043 [+0.018, +0.073] |
| d_F cal. | delta vs ICP mm | -0.060 [-0.093, -0.027] | -0.130 [-0.195, -0.075] | -0.035 [-0.076, +0.001] | -0.049 [-0.090, -0.004] |
| d_F c=1 | valore | 0.867 [0.800, 0.920] | 0.669 [0.585, 0.745] | 0.832 [0.782, 0.880] | 0.237 [0.182, 0.287] |
| d_F c=1 | delta vs paper s1234 | +0.013 [-0.053, +0.067] | +0.004 [-0.051, +0.052] | +0.002 [-0.030, +0.032] | +0.005 [-0.030, +0.040] |
| d_F c=1 | delta vs ICP mm | -0.080 [-0.147, -0.013] | -0.153 [-0.235, -0.077] | -0.063 [-0.107, -0.021] | -0.064 [-0.109, -0.011] |
| d_P | valore | 0.867 [0.800, 0.920] | 0.647 [0.560, 0.726] | 0.828 [0.777, 0.878] | 0.241 [0.190, 0.292] |
| d_P | delta vs paper s1234 | +0.053 [-0.020, +0.113] | +0.011 [-0.054, +0.066] | -0.002 [-0.034, +0.029] | -0.000 [-0.031, +0.036] |
| d_P | delta vs ICP mm | -0.080 [-0.140, -0.033] | -0.176 [-0.264, -0.092] | -0.067 [-0.115, -0.022] | -0.068 [-0.121, -0.014] |

## Controlli di riproduzione

- Matrice: 68844 righe di `matrix/spearman.csv` (colonne non massive) con le stesse chiavi di `full_matrix/spearman.csv`; n_righe diversi 0; scarto massimo punto 0.00e+00, IC 0.00e+00; righe senza riferimento 514 (colonne nuove di `celle_mancanti`).
- Ava-256: 2684 righe contro `ava256_conf/desc/spearman.csv`; n_righe diversi 0; scarto massimo punto 0.00e+00, IC 0.00e+00. Embedding su A10 (le A100 della prova erano occupate): C3M e123 rifatto su A10 contro le sue colonne su A100, 180 righe, scarto massimo punto 2.59e-04, IC 4.19e-04.
- FaMoS riconoscimento: 44 righe (paper s1234 e concorrenti) contro `esterna_famos_id/metrics.csv`, scarto massimo 0.00e+00; embedding contro i latenti della matrice: `{"embed_massive_main": true, "embed_massive_open": true}`.

Tutto in `controls.json`.

