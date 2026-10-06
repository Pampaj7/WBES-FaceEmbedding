# WS-frame: frame, verso delle facce e normalizzazione nei test zero-shot

Convenzioni di frame, misurate sulle mesh dei dati (aau/scratch/hifi3d/frames.py, winding2.py):

    dominio      alto   naso   normali
    BFM (train)  -y     -z     verso l'interno (7% verso l'esterno)
    ICT (train)  +y     +z     verso l'esterno (83%)
    HIFI3D       +y     +z     verso l'esterno (83%)  = convenzione ICT
    FaceVerse    -y     -z     verso l'esterno (94%)  = rotazione BFM, verso ICT

Varianti (suffissi delle out dir di zs_zeroshot.sbatch):
    ""                     dati come sono
    _frame-xmymz           Rx(180) = diag(1,-1,-1): rotazione propria, verso INVARIATO
    _frame-xmymz_flip      Rx(180) + facce invertite: convenzione BFM completa (HIFI3D, ICT)
    _flip                  facce invertite (FaceVerse: + rotazione nativa = convenzione BFM completa)
    _frame-mxymz           Ry(180) = diag(-1,1,-1) (la rotazione proposta dal critic)
    _evalframe-rms         V_in ri-inquadrato nel frame rms nel forward (ablation_hooks.py)

Tutti i numeri: 100 soggetti per dominio, gli stessi in ogni variante e braccio; GT maxabs (protocollo ICT); CI 95% bootstrap per soggetto, 1000 repliche; differenze appaiate sulle stesse repliche (zs_summarize.py). Tabelle complete per variante in `aau/runs/<ws_hifi3d|ws_faceverse|ws_ictzs>/summary<variante>.md`.

Nota sul controllo Chamfer: dove nativo e variante sono girati sullo stesso tipo di GPU (ICT, entrambi A10) la Chamfer e' identica bit per bit; su HIFI3D il nativo e' su L40S e le varianti su A10, e 165 righe su 148.500 (tutte con `down8k`) differiscono fino al 3% relativo, con Spearman di Chamfer identico a 2e-6.


## HIFI3D


### Spearman latent con la GT maxabs, senza crop (CI 95% per soggetto)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo | ICT completa | 0.428 [0.38, 0.48] | 0.206 [0.17, 0.24] | 0.382 [0.33, 0.43] | 0.372 [0.32, 0.42] |
| Rx(180) | rotazione BFM, verso ICT | 0.246 [0.20, 0.30] | 0.280 [0.22, 0.34] | 0.353 [0.30, 0.40] | 0.372 [0.32, 0.42] |
| Rx(180) + facce invertite | BFM completa | 0.245 [0.20, 0.29] | 0.320 [0.25, 0.38] | 0.327 [0.28, 0.38] | 0.372 [0.32, 0.42] |
| Ry(180) (critic) | naso -z ma alto +y: nessuna delle due | 0.216 [0.18, 0.25] | 0.179 [0.14, 0.22] | 0.314 [0.26, 0.36] | 0.372 [0.32, 0.42] |
| nativo, eval_frame rms | ICT completa, ri-inquadrato rms | 0.330 [0.28, 0.38] | 0.094 [0.07, 0.12] | 0.427 [0.36, 0.49] | 0.372 [0.32, 0.42] |

#### Differenze appaiate, senza crop: variante - nativo (stesso braccio) e modello - Chamfer

| variante | BFM+ICT: var - nativo | BFM-only: var - nativo | ICT-only: var - nativo | BFM+ICT - Chamfer | BFM-only - Chamfer | ICT-only - Chamfer |
| --- | --- | --- | --- | --- | --- | --- |
| nativo | 0 | 0 | 0 | +0.056 [+0.015, +0.094] | -0.166 [-0.203, -0.124] | +0.010 [-0.019, +0.038] |
| Rx(180) | -0.182 [-0.219, -0.140] | +0.074 [+0.023, +0.126] | -0.029 [-0.060, +0.003] | -0.126 [-0.174, -0.083] | -0.092 [-0.161, -0.024] | -0.019 [-0.049, +0.008] |
| Rx(180) + facce invertite | -0.183 [-0.223, -0.139] | +0.114 [+0.067, +0.165] | -0.054 [-0.082, -0.024] | -0.128 [-0.172, -0.082] | -0.052 [-0.123, +0.020] | -0.045 [-0.075, -0.015] |
| Ry(180) (critic) | -0.212 [-0.261, -0.164] | -0.027 [-0.067, +0.010] | -0.068 [-0.098, -0.039] | -0.156 [-0.204, -0.105] | -0.193 [-0.253, -0.135] | -0.059 [-0.090, -0.029] |
| nativo, eval_frame rms | -0.098 [-0.131, -0.066] | -0.112 [-0.147, -0.077] | +0.045 [+0.010, +0.079] | -0.043 [-0.095, +0.016] | -0.278 [-0.325, -0.227] | +0.055 [+0.010, +0.100] |

#### Differenze appaiate fra bracci, senza crop

| variante | BFM+ICT - ICT-only | BFM+ICT - BFM-only |
| --- | --- | --- |
| nativo | +0.046 [+0.018, +0.075] | +0.222 [+0.182, +0.260] |
| Rx(180) | -0.108 [-0.143, -0.069] | -0.035 [-0.077, +0.009] |
| Rx(180) + facce invertite | -0.083 [-0.123, -0.044] | -0.075 [-0.107, -0.040] |
| Ry(180) (critic) | -0.097 [-0.149, -0.043] | +0.037 [-0.001, +0.075] |
| nativo, eval_frame rms | -0.097 [-0.153, -0.042] | +0.235 [+0.190, +0.280] |

### Spearman latent con la GT maxabs, tutte le topologie (CI 95% per soggetto)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo | ICT completa | 0.246 [0.21, 0.28] | 0.180 [0.14, 0.22] | 0.222 [0.18, 0.26] | 0.336 [0.29, 0.38] |
| Rx(180) | rotazione BFM, verso ICT | 0.197 [0.16, 0.24] | 0.198 [0.15, 0.25] | 0.276 [0.23, 0.32] | 0.336 [0.29, 0.38] |
| Rx(180) + facce invertite | BFM completa | 0.197 [0.16, 0.24] | 0.217 [0.17, 0.26] | 0.256 [0.21, 0.30] | 0.336 [0.29, 0.38] |
| Ry(180) (critic) | naso -z ma alto +y: nessuna delle due | 0.158 [0.12, 0.19] | 0.128 [0.10, 0.16] | 0.261 [0.21, 0.31] | 0.336 [0.29, 0.38] |
| nativo, eval_frame rms | ICT completa, ri-inquadrato rms | 0.232 [0.19, 0.28] | 0.097 [0.07, 0.12] | 0.238 [0.19, 0.28] | 0.336 [0.29, 0.38] |

#### Differenze appaiate, tutte le topologie: variante - nativo (stesso braccio) e modello - Chamfer

| variante | BFM+ICT: var - nativo | BFM-only: var - nativo | ICT-only: var - nativo | BFM+ICT - Chamfer | BFM-only - Chamfer | ICT-only - Chamfer |
| --- | --- | --- | --- | --- | --- | --- |
| nativo | 0 | 0 | 0 | -0.090 [-0.123, -0.059] | -0.156 [-0.194, -0.117] | -0.114 [-0.136, -0.091] |
| Rx(180) | -0.049 [-0.080, -0.015] | +0.018 [-0.025, +0.065] | +0.055 [+0.029, +0.080] | -0.139 [-0.178, -0.098] | -0.138 [-0.202, -0.074] | -0.060 [-0.090, -0.031] |
| Rx(180) + facce invertite | -0.049 [-0.080, -0.017] | +0.037 [-0.002, +0.074] | +0.035 [+0.011, +0.058] | -0.139 [-0.182, -0.099] | -0.119 [-0.172, -0.062] | -0.079 [-0.105, -0.055] |
| Ry(180) (critic) | -0.088 [-0.123, -0.053] | -0.052 [-0.082, -0.022] | +0.040 [+0.011, +0.069] | -0.178 [-0.225, -0.137] | -0.208 [-0.258, -0.156] | -0.075 [-0.102, -0.046] |
| nativo, eval_frame rms | -0.014 [-0.039, +0.012] | -0.083 [-0.114, -0.052] | +0.017 [-0.007, +0.039] | -0.104 [-0.146, -0.058] | -0.239 [-0.286, -0.189] | -0.098 [-0.133, -0.064] |

#### Differenze appaiate fra bracci, tutte le topologie

| variante | BFM+ICT - ICT-only | BFM+ICT - BFM-only |
| --- | --- | --- |
| nativo | +0.024 [+0.004, +0.044] | +0.066 [+0.036, +0.093] |
| Rx(180) | -0.079 [-0.111, -0.047] | -0.001 [-0.036, +0.037] |
| Rx(180) + facce invertite | -0.059 [-0.095, -0.022] | -0.020 [-0.048, +0.007] |
| Ry(180) (critic) | -0.104 [-0.148, -0.056] | +0.030 [-0.000, +0.060] |
| nativo, eval_frame rms | -0.006 [-0.047, +0.036] | +0.135 [+0.102, +0.170] |

### Spearman latent con la GT maxabs, media per coppia di soggetti (CI 95% per soggetto)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo | ICT completa | 0.720 [0.66, 0.78] | 0.607 [0.53, 0.68] | 0.713 [0.64, 0.78] | 0.743 [0.68, 0.80] |
| Rx(180) | rotazione BFM, verso ICT | 0.585 [0.51, 0.65] | 0.474 [0.39, 0.56] | 0.694 [0.62, 0.75] | 0.743 [0.68, 0.80] |
| Rx(180) + facce invertite | BFM completa | 0.571 [0.49, 0.64] | 0.510 [0.42, 0.59] | 0.692 [0.61, 0.75] | 0.743 [0.68, 0.80] |
| Ry(180) (critic) | naso -z ma alto +y: nessuna delle due | 0.523 [0.44, 0.60] | 0.498 [0.41, 0.58] | 0.689 [0.61, 0.75] | 0.743 [0.68, 0.80] |
| nativo, eval_frame rms | ICT completa, ri-inquadrato rms | 0.551 [0.47, 0.63] | 0.390 [0.30, 0.49] | 0.606 [0.52, 0.69] | 0.743 [0.68, 0.80] |

#### Differenze appaiate, media per coppia di soggetti: variante - nativo (stesso braccio) e modello - Chamfer

| variante | BFM+ICT: var - nativo | BFM-only: var - nativo | ICT-only: var - nativo | BFM+ICT - Chamfer | BFM-only - Chamfer | ICT-only - Chamfer |
| --- | --- | --- | --- | --- | --- | --- |
| nativo | 0 | 0 | 0 | -0.023 [-0.083, +0.036] | -0.136 [-0.210, -0.065] | -0.030 [-0.086, +0.023] |
| Rx(180) | -0.135 [-0.203, -0.065] | -0.133 [-0.228, -0.041] | -0.019 [-0.071, +0.037] | -0.158 [-0.227, -0.085] | -0.269 [-0.371, -0.174] | -0.049 [-0.096, -0.007] |
| Rx(180) + facce invertite | -0.149 [-0.222, -0.076] | -0.097 [-0.188, -0.012] | -0.021 [-0.074, +0.037] | -0.172 [-0.245, -0.097] | -0.234 [-0.324, -0.143] | -0.051 [-0.093, -0.008] |
| Ry(180) (critic) | -0.197 [-0.287, -0.105] | -0.108 [-0.203, -0.022] | -0.024 [-0.067, +0.020] | -0.220 [-0.316, -0.119] | -0.245 [-0.348, -0.143] | -0.054 [-0.100, -0.011] |
| nativo, eval_frame rms | -0.170 [-0.216, -0.120] | -0.217 [-0.313, -0.130] | -0.107 [-0.171, -0.050] | -0.193 [-0.266, -0.117] | -0.353 [-0.453, -0.250] | -0.137 [-0.203, -0.073] |

#### Differenze appaiate fra bracci, media per coppia di soggetti

| variante | BFM+ICT - ICT-only | BFM+ICT - BFM-only |
| --- | --- | --- |
| nativo | +0.007 [-0.040, +0.063] | +0.113 [+0.044, +0.181] |
| Rx(180) | -0.109 [-0.168, -0.048] | +0.112 [+0.047, +0.178] |
| Rx(180) + facce invertite | -0.121 [-0.188, -0.049] | +0.062 [+0.005, +0.121] |
| Ry(180) (critic) | -0.166 [-0.259, -0.066] | +0.024 [-0.060, +0.105] |
| nativo, eval_frame rms | -0.056 [-0.129, +0.020] | +0.161 [+0.089, +0.244] |

### Controllo di sanita': Chamfer eval della variante contro il nativo, riga per riga

| variante | max diff relativa | righe con diff relativa > 1e-4 (su 148.500) | GPU variante / nativo |
| --- | --- | --- | --- |
| Rx(180) | 3.27e-02 | 165 | vedi nota |
| Rx(180) + facce invertite | 3.27e-02 | 165 | vedi nota |
| Ry(180) (critic) | 3.27e-02 | 165 | vedi nota |
| nativo, eval_frame rms | 3.27e-02 | 165 | vedi nota |

## FaceVerse

Varianti mancanti (non ancora calcolate): `_flip` (facce invertite).


### Spearman latent con la GT maxabs, senza crop (CI 95% per soggetto)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo | rotazione BFM, verso ICT | 0.392 [0.33, 0.45] | 0.364 [0.28, 0.44] | 0.297 [0.24, 0.35] | 0.425 [0.37, 0.48] |
| Rx(180) | ICT completa | 0.261 [0.19, 0.32] | 0.184 [0.13, 0.24] | 0.266 [0.20, 0.34] | 0.425 [0.37, 0.48] |

#### Differenze appaiate, senza crop: variante - nativo (stesso braccio) e modello - Chamfer

| variante | BFM+ICT: var - nativo | BFM-only: var - nativo | ICT-only: var - nativo | BFM+ICT - Chamfer | BFM-only - Chamfer | ICT-only - Chamfer |
| --- | --- | --- | --- | --- | --- | --- |
| nativo | 0 | 0 | 0 | -0.033 [-0.098, +0.034] | -0.060 [-0.143, +0.028] | -0.128 [-0.176, -0.078] |
| Rx(180) | -0.130 [-0.201, -0.060] | -0.180 [-0.257, -0.087] | -0.031 [-0.092, +0.022] | -0.164 [-0.218, -0.103] | -0.241 [-0.294, -0.188] | -0.159 [-0.215, -0.105] |

#### Differenze appaiate fra bracci, senza crop

| variante | BFM+ICT - ICT-only | BFM+ICT - BFM-only |
| --- | --- | --- |
| nativo | +0.094 [+0.031, +0.157] | +0.027 [-0.033, +0.084] |
| Rx(180) | -0.005 [-0.052, +0.045] | +0.077 [+0.015, +0.143] |

### Spearman latent con la GT maxabs, tutte le topologie (CI 95% per soggetto)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo | rotazione BFM, verso ICT | 0.338 [0.28, 0.38] | 0.337 [0.27, 0.41] | 0.205 [0.17, 0.24] | 0.298 [0.25, 0.34] |
| Rx(180) | ICT completa | 0.168 [0.12, 0.21] | 0.173 [0.12, 0.22] | 0.175 [0.12, 0.22] | 0.298 [0.25, 0.34] |

#### Differenze appaiate, tutte le topologie: variante - nativo (stesso braccio) e modello - Chamfer

| variante | BFM+ICT: var - nativo | BFM-only: var - nativo | ICT-only: var - nativo | BFM+ICT - Chamfer | BFM-only - Chamfer | ICT-only - Chamfer |
| --- | --- | --- | --- | --- | --- | --- |
| nativo | 0 | 0 | 0 | +0.040 [-0.018, +0.098] | +0.039 [-0.043, +0.118] | -0.093 [-0.133, -0.048] |
| Rx(180) | -0.170 [-0.227, -0.105] | -0.164 [-0.238, -0.091] | -0.029 [-0.074, +0.010] | -0.130 [-0.179, -0.081] | -0.125 [-0.181, -0.069] | -0.123 [-0.162, -0.080] |

#### Differenze appaiate fra bracci, tutte le topologie

| variante | BFM+ICT - ICT-only | BFM+ICT - BFM-only |
| --- | --- | --- |
| nativo | +0.133 [+0.076, +0.180] | +0.001 [-0.053, +0.050] |
| Rx(180) | -0.007 [-0.044, +0.028] | -0.005 [-0.059, +0.046] |

### Spearman latent con la GT maxabs, media per coppia di soggetti (CI 95% per soggetto)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo | rotazione BFM, verso ICT | 0.503 [0.43, 0.57] | 0.452 [0.36, 0.53] | 0.517 [0.44, 0.59] | 0.577 [0.49, 0.67] |
| Rx(180) | ICT completa | 0.377 [0.28, 0.47] | 0.315 [0.20, 0.42] | 0.391 [0.29, 0.49] | 0.577 [0.49, 0.67] |

#### Differenze appaiate, media per coppia di soggetti: variante - nativo (stesso braccio) e modello - Chamfer

| variante | BFM+ICT: var - nativo | BFM-only: var - nativo | ICT-only: var - nativo | BFM+ICT - Chamfer | BFM-only - Chamfer | ICT-only - Chamfer |
| --- | --- | --- | --- | --- | --- | --- |
| nativo | 0 | 0 | 0 | -0.074 [-0.161, +0.037] | -0.125 [-0.243, +0.002] | -0.061 [-0.148, +0.029] |
| Rx(180) | -0.127 [-0.230, -0.030] | -0.138 [-0.264, -0.013] | -0.125 [-0.219, -0.045] | -0.201 [-0.307, -0.102] | -0.263 [-0.379, -0.150] | -0.186 [-0.274, -0.092] |

#### Differenze appaiate fra bracci, media per coppia di soggetti

| variante | BFM+ICT - ICT-only | BFM+ICT - BFM-only |
| --- | --- | --- |
| nativo | -0.014 [-0.090, +0.066] | +0.051 [-0.017, +0.127] |
| Rx(180) | -0.015 [-0.091, +0.061] | +0.062 [-0.063, +0.190] |

### Controllo di sanita': Chamfer eval della variante contro il nativo, riga per riga

| variante | max diff relativa | righe con diff relativa > 1e-4 (su 148.500) | GPU variante / nativo |
| --- | --- | --- | --- |
| Rx(180) | 0.00e+00 | 0 | vedi nota |

## ICT held-out (controllo e2e)


### Spearman latent con la GT maxabs, senza crop (CI 95% per soggetto)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo (e2e) | ICT completa | 0.991 [0.99, 0.99] | 0.259 [0.21, 0.31] | 0.991 [0.99, 0.99] | 0.443 [0.37, 0.51] |
| Rx(180) | rotazione BFM, verso ICT | 0.368 [0.31, 0.42] | 0.260 [0.22, 0.30] | 0.358 [0.29, 0.42] | 0.443 [0.37, 0.51] |
| Rx(180) + facce invertite | BFM completa | 0.354 [0.30, 0.40] | 0.268 [0.22, 0.31] | 0.346 [0.28, 0.41] | 0.443 [0.37, 0.51] |

#### Differenze appaiate, senza crop: variante - nativo (stesso braccio) e modello - Chamfer

| variante | BFM+ICT: var - nativo | BFM-only: var - nativo | ICT-only: var - nativo | BFM+ICT - Chamfer | BFM-only - Chamfer | ICT-only - Chamfer |
| --- | --- | --- | --- | --- | --- | --- |
| nativo (e2e) | 0 | 0 | 0 | +0.547 [+0.481, +0.614] | -0.184 [-0.214, -0.148] | +0.548 [+0.481, +0.618] |
| Rx(180) | -0.622 [-0.674, -0.578] | +0.000 [-0.025, +0.024] | -0.633 [-0.693, -0.576] | -0.075 [-0.114, -0.035] | -0.184 [-0.227, -0.140] | -0.085 [-0.107, -0.058] |
| Rx(180) + facce invertite | -0.636 [-0.683, -0.593] | +0.009 [-0.014, +0.033] | -0.645 [-0.707, -0.590] | -0.089 [-0.128, -0.052] | -0.175 [-0.213, -0.129] | -0.097 [-0.119, -0.072] |

#### Differenze appaiate fra bracci, senza crop

| variante | BFM+ICT - ICT-only | BFM+ICT - BFM-only |
| --- | --- | --- |
| nativo (e2e) | -0.001 [-0.002, -0.000] | +0.731 [+0.687, +0.779] |
| Rx(180) | +0.010 [-0.019, +0.042] | +0.109 [+0.083, +0.134] |
| Rx(180) + facce invertite | +0.008 [-0.020, +0.039] | +0.086 [+0.063, +0.108] |

### Spearman latent con la GT maxabs, tutte le topologie (CI 95% per soggetto)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo (e2e) | ICT completa | 0.989 [0.98, 0.99] | 0.294 [0.24, 0.34] | 0.991 [0.99, 0.99] | 0.442 [0.37, 0.51] |
| Rx(180) | rotazione BFM, verso ICT | 0.321 [0.27, 0.36] | 0.242 [0.20, 0.28] | 0.392 [0.33, 0.45] | 0.442 [0.37, 0.51] |
| Rx(180) + facce invertite | BFM completa | 0.310 [0.26, 0.35] | 0.247 [0.20, 0.28] | 0.382 [0.32, 0.44] | 0.442 [0.37, 0.51] |

#### Differenze appaiate, tutte le topologie: variante - nativo (stesso braccio) e modello - Chamfer

| variante | BFM+ICT: var - nativo | BFM-only: var - nativo | ICT-only: var - nativo | BFM+ICT - Chamfer | BFM-only - Chamfer | ICT-only - Chamfer |
| --- | --- | --- | --- | --- | --- | --- |
| nativo (e2e) | 0 | 0 | 0 | +0.546 [+0.480, +0.622] | -0.149 [-0.184, -0.114] | +0.548 [+0.480, +0.623] |
| Rx(180) | -0.668 [-0.714, -0.625] | -0.052 [-0.076, -0.026] | -0.598 [-0.663, -0.535] | -0.122 [-0.164, -0.077] | -0.201 [-0.241, -0.158] | -0.050 [-0.071, -0.026] |
| Rx(180) + facce invertite | -0.678 [-0.724, -0.637] | -0.047 [-0.074, -0.023] | -0.609 [-0.672, -0.552] | -0.132 [-0.173, -0.088] | -0.196 [-0.235, -0.149] | -0.061 [-0.082, -0.037] |

#### Differenze appaiate fra bracci, tutte le topologie

| variante | BFM+ICT - ICT-only | BFM+ICT - BFM-only |
| --- | --- | --- |
| nativo (e2e) | -0.002 [-0.003, -0.001] | +0.695 [+0.651, +0.741] |
| Rx(180) | -0.072 [-0.107, -0.037] | +0.079 [+0.057, +0.102] |
| Rx(180) + facce invertite | -0.071 [-0.106, -0.039] | +0.064 [+0.044, +0.083] |

### Spearman latent con la GT maxabs, media per coppia di soggetti (CI 95% per soggetto)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo (e2e) | ICT completa | 0.996 [0.99, 1.00] | 0.836 [0.78, 0.88] | 0.996 [0.99, 1.00] | 0.911 [0.87, 0.94] |
| Rx(180) | rotazione BFM, verso ICT | 0.864 [0.83, 0.90] | 0.741 [0.67, 0.80] | 0.916 [0.88, 0.94] | 0.911 [0.87, 0.94] |
| Rx(180) + facce invertite | BFM completa | 0.866 [0.83, 0.90] | 0.741 [0.67, 0.80] | 0.920 [0.89, 0.94] | 0.911 [0.87, 0.94] |

#### Differenze appaiate, media per coppia di soggetti: variante - nativo (stesso braccio) e modello - Chamfer

| variante | BFM+ICT: var - nativo | BFM-only: var - nativo | ICT-only: var - nativo | BFM+ICT - Chamfer | BFM-only - Chamfer | ICT-only - Chamfer |
| --- | --- | --- | --- | --- | --- | --- |
| nativo (e2e) | 0 | 0 | 0 | +0.085 [+0.059, +0.120] | -0.075 [-0.108, -0.045] | +0.085 [+0.059, +0.118] |
| Rx(180) | -0.132 [-0.173, -0.101] | -0.095 [-0.137, -0.056] | -0.080 [-0.110, -0.054] | -0.047 [-0.070, -0.028] | -0.170 [-0.220, -0.123] | +0.005 [-0.015, +0.026] |
| Rx(180) + facce invertite | -0.130 [-0.168, -0.099] | -0.095 [-0.134, -0.053] | -0.076 [-0.107, -0.053] | -0.045 [-0.066, -0.024] | -0.170 [-0.215, -0.121] | +0.009 [-0.010, +0.028] |

#### Differenze appaiate fra bracci, media per coppia di soggetti

| variante | BFM+ICT - ICT-only | BFM+ICT - BFM-only |
| --- | --- | --- |
| nativo (e2e) | -0.000 [-0.001, +0.000] | +0.160 [+0.119, +0.214] |
| Rx(180) | -0.052 [-0.075, -0.030] | +0.123 [+0.085, +0.167] |
| Rx(180) + facce invertite | -0.054 [-0.079, -0.030] | +0.125 [+0.089, +0.166] |

### Controllo di sanita': Chamfer eval della variante contro il nativo, riga per riga

| variante | max diff relativa | righe con diff relativa > 1e-4 (su 148.500) | GPU variante / nativo |
| --- | --- | --- | --- |
| Rx(180) | 0.00e+00 | 0 | vedi nota |
| Rx(180) + facce invertite | 0.00e+00 | 0 | vedi nota |

## Crop e normalizzazione (HIFI3D): riga crop -> B, Spearman latent, GT maxabs

| braccio | variante | crop->down8k | crop->noisy | crop->original | crop->remesh | crop->up60k |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT | maxabs (nativo) | 0.042 | 0.113 | 0.140 | 0.067 | 0.069 |
| BFM-only | maxabs (nativo) | 0.064 | 0.388 | 0.362 | 0.152 | 0.172 |
| ICT-only | maxabs (nativo) | 0.023 | 0.091 | 0.114 | 0.048 | 0.043 |
| BFM+ICT | rms | 0.072 | 0.192 | 0.232 | 0.166 | 0.136 |
| BFM-only | rms | 0.236 | 0.253 | 0.138 | 0.180 | 0.187 |
| ICT-only | rms | 0.050 | 0.140 | 0.112 | 0.094 | 0.073 |
| Chamfer eval | - | 0.099 | 0.651 | 0.646 | 0.306 | 0.326 |
