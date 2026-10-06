# WS-frame: frame, verso delle facce e normalizzazione nei test zero-shot

Protocolli, in quest'ordine (DICHIARATO DOPO AVER VISTO QUESTI NUMERI, 6 ottobre 2026: vale come
regola per i test futuri, non come scelta a priori di questi):
  1. PRIMARIO   coppie di mesh cross-topologia SENZA crop: 20 coppie ordinate x 4950 coppie di
                soggetti, una osservazione per (coppia di soggetti, coppia di topologie);
  2. SECONDARIO media per coppia di soggetti (lo scenario clean dello script di ranking), sulle 30
                coppie ordinate cross, crop compreso;
  3. A PARTE    le 10 coppie ordinate con crop da un lato.

Tutti gli Spearman, i CI e i contrasti vengono dalle STESSE repliche bootstrap per soggetto (la
ricampionatura di ``weighted_bootstrap_spearman``: soggetti con reinserimento, peso di una coppia
= prodotto dei conteggi dei suoi due soggetti), applicate a una tabella in cui ogni variante e
ogni braccio e' una colonna allineata sulle stesse righe (stessi soggetti, stesse topologie,
stessa GT; ``merge`` one-to-one, altrimenti esce). Cosi' anche i contrasti FRA VARIANTI -- p.es.
congiunto nativo contro BFM-only nel frame dei suoi dati -- hanno un CI appaiato.

Convenzioni di frame, misurate sulle mesh dei dati (aau/scratch/hifi3d/frames.py, winding2.py):

    dominio      alto   naso   normali
    BFM (train)  -y     -z     verso l'interno (7% verso l'esterno)
    ICT (train)  +y     +z     verso l'esterno (83%)
    HIFI3D       +y     +z     verso l'esterno (83%)  = convenzione ICT
    FaceVerse    -y     -z     verso l'esterno (94%)  = rotazione BFM, verso ICT

"Frame dei dati di training" di un braccio: BFM-only -> convenzione BFM completa (rotazione +
facce invertite); ICT-only -> convenzione ICT completa; congiunto -> ambiguo (ha visto entrambe),
riportato nel nativo.

Leakage ICT: il dominio ``ict`` e' il pool id14500-id14999 di ICT-5000, che NON e' held-out per
congiunto e ICT-only (il loro split e' casuale sull'intera data dir: aau/cross3dmm/ws2_views.py).
Le celle di quei due bracci su ICT, varianti e contrasti compresi, sono marcate "inquinate"; il
numero di soggetti in training e' contato qui da ``aau/runs/ws2_cross3dmm/splits.json``.

Tutti i numeri: 100 soggetti per dominio, gli stessi in ogni variante e braccio; GT maxabs (protocollo ICT); CI 95% percentili su 1000 repliche bootstrap per soggetto, le STESSE per tutte le colonne di un dominio e protocollo. P(boot <= 0) = frazione di repliche con contrasto <= 0. Tabelle per variante (anche con la GT nei coefficienti) in `aau/runs/<ws_hifi3d|ws_faceverse|ws_ictzs>/summary<variante>.md`.

Soggetti ICT valutati che sono nel training di ciascun modello (splits.json di WS2): BFM+ICT 80/100, BFM-only 0/100, ICT-only 77/100.


## PRIMARIO: coppie di mesh cross-topologia senza crop (20 coppie ordinate)


### HIFI3D (99000 righe)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo | ICT completa | 0.428 [0.37, 0.48] | 0.206 [0.17, 0.25] | 0.382 [0.32, 0.43] | 0.372 [0.32, 0.42] |
| Rx(180) | rotazione BFM, verso ICT | 0.246 [0.20, 0.29] | 0.280 [0.23, 0.34] | 0.353 [0.30, 0.41] | 0.372 [0.32, 0.42] |
| Rx(180) + facce invertite | BFM completa | 0.245 [0.19, 0.29] | 0.320 [0.26, 0.38] | 0.327 [0.27, 0.38] | 0.372 [0.32, 0.42] |
| Ry(180) (critic) | naso -z, alto +y: nessuna delle due | 0.216 [0.18, 0.26] | 0.179 [0.14, 0.22] | 0.314 [0.26, 0.37] | 0.372 [0.32, 0.42] |
| nativo, eval_frame rms | ICT completa, ri-inquadrato rms | 0.330 [0.27, 0.39] | 0.094 [0.07, 0.12] | 0.427 [0.35, 0.49] | 0.372 [0.32, 0.42] |

| contrasto (appaiato, stesse repliche) | stima [CI 95%] | P(boot <= 0) |
| --- | --- | --- |
| BFM-only: frame dei suoi dati (_frame-xmymz_flip) - nativo | +0.114 [+0.064, +0.166] | 0.000 |
| BFM+ICT nativo - BFM-only nativo (gap a parita' di ingresso) | +0.222 [+0.179, +0.259] | 0.000 |
| BFM+ICT nativo - BFM-only nel frame dei suoi dati | +0.108 [+0.047, +0.159] | 0.000 |
| BFM+ICT nativo - ICT-only nel frame dei suoi dati | +0.046 [+0.018, +0.076] | 0.000 |
| frazione del gap BFM+ICT - BFM-only spiegata dal frame | +0.51 [+0.30, +0.76] | - |
| BFM+ICT nel frame dei suoi dati - Chamfer eval | +0.056 [+0.016, +0.092] | 0.001 |
| BFM-only nel frame dei suoi dati - Chamfer eval | -0.052 [-0.115, +0.014] | 0.939 |
| ICT-only nel frame dei suoi dati - Chamfer eval | +0.010 [-0.021, +0.039] | 0.274 |

### FaceVerse v2 (99000 righe)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo | rotazione BFM, verso ICT | 0.392 [0.33, 0.45] | 0.364 [0.29, 0.44] | 0.297 [0.23, 0.35] | 0.425 [0.37, 0.47] |
| Rx(180) | ICT completa | 0.261 [0.19, 0.32] | 0.184 [0.13, 0.24] | 0.266 [0.19, 0.33] | 0.425 [0.37, 0.47] |
| facce invertite | BFM completa | mancante | mancante | mancante | 0.425 [0.37, 0.47] |

| contrasto (appaiato, stesse repliche) | stima [CI 95%] | P(boot <= 0) |
| --- | --- | --- |
| BFM-only: frame dei suoi dati (_flip) - nativo | mancante | - |
| ICT-only: frame dei suoi dati (_frame-xmymz) - nativo | -0.031 [-0.087, +0.024] | 0.857 |
| BFM+ICT nativo - BFM-only nativo (gap a parita' di ingresso) | +0.027 [-0.033, +0.086] | 0.207 |
| BFM+ICT nativo - BFM-only nel frame dei suoi dati | mancante | - |
| BFM+ICT nativo - ICT-only nel frame dei suoi dati | +0.126 [+0.044, +0.213] | 0.003 |
| frazione del gap BFM+ICT - BFM-only spiegata dal frame | mancante | - |
| BFM+ICT nel frame dei suoi dati - Chamfer eval | -0.033 [-0.100, +0.033] | 0.830 |
| BFM-only nel frame dei suoi dati - Chamfer eval | mancante | - |
| ICT-only nel frame dei suoi dati - Chamfer eval | -0.159 [-0.209, -0.103] | 1.000 |

### ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) (99000 righe)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo (e2e) | ICT completa | 0.991 [0.99, 0.99] (inquinata (soggetti in training)) | 0.259 [0.21, 0.31] | 0.991 [0.99, 0.99] (inquinata (soggetti in training)) | 0.443 [0.37, 0.51] |
| Rx(180) | rotazione BFM, verso ICT | 0.368 [0.31, 0.42] (inquinata (soggetti in training)) | 0.260 [0.22, 0.30] | 0.358 [0.29, 0.42] (inquinata (soggetti in training)) | 0.443 [0.37, 0.51] |
| Rx(180) + facce invertite | BFM completa | 0.354 [0.30, 0.40] (inquinata (soggetti in training)) | 0.268 [0.22, 0.31] | 0.346 [0.28, 0.41] (inquinata (soggetti in training)) | 0.443 [0.37, 0.51] |

| contrasto (appaiato, stesse repliche) | stima [CI 95%] | P(boot <= 0) |
| --- | --- | --- |
| BFM-only: frame dei suoi dati (_frame-xmymz_flip) - nativo | +0.009 [-0.014, +0.032] | 0.212 |
| BFM+ICT nativo - BFM-only nativo (gap a parita' di ingresso) -- inquinata (soggetti in training) | +0.731 [+0.683, +0.777] | 0.000 |
| BFM+ICT nativo - BFM-only nel frame dei suoi dati -- inquinata (soggetti in training) | +0.722 [+0.678, +0.764] | 0.000 |
| BFM+ICT nativo - ICT-only nel frame dei suoi dati -- inquinata (soggetti in training) | -0.001 [-0.002, -0.000] | 0.991 |
| frazione del gap BFM+ICT - BFM-only spiegata dal frame -- inquinata (soggetti in training) | +0.01 [-0.02, +0.04] | - |
| BFM+ICT nel frame dei suoi dati - Chamfer eval -- inquinata (soggetti in training) | +0.547 [+0.479, +0.618] | 0.000 |
| BFM-only nel frame dei suoi dati - Chamfer eval | -0.175 [-0.216, -0.137] | 1.000 |
| ICT-only nel frame dei suoi dati - Chamfer eval -- inquinata (soggetti in training) | +0.548 [+0.480, +0.619] | 0.000 |

## SECONDARIO: media per coppia di soggetti (30 coppie ordinate, crop compreso)


### HIFI3D (4950 righe)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo | ICT completa | 0.720 [0.65, 0.78] | 0.607 [0.53, 0.68] | 0.713 [0.64, 0.78] | 0.743 [0.68, 0.79] |
| Rx(180) | rotazione BFM, verso ICT | 0.585 [0.51, 0.67] | 0.474 [0.39, 0.56] | 0.694 [0.63, 0.75] | 0.743 [0.68, 0.79] |
| Rx(180) + facce invertite | BFM completa | 0.571 [0.49, 0.66] | 0.510 [0.43, 0.59] | 0.692 [0.62, 0.75] | 0.743 [0.68, 0.79] |
| Ry(180) (critic) | naso -z, alto +y: nessuna delle due | 0.523 [0.43, 0.62] | 0.498 [0.41, 0.59] | 0.689 [0.62, 0.75] | 0.743 [0.68, 0.79] |
| nativo, eval_frame rms | ICT completa, ri-inquadrato rms | 0.551 [0.47, 0.63] | 0.390 [0.30, 0.49] | 0.606 [0.52, 0.68] | 0.743 [0.68, 0.79] |

| contrasto (appaiato, stesse repliche) | stima [CI 95%] | P(boot <= 0) |
| --- | --- | --- |
| BFM-only: frame dei suoi dati (_frame-xmymz_flip) - nativo | -0.097 [-0.181, -0.014] | 0.989 |
| BFM+ICT nativo - BFM-only nativo (gap a parita' di ingresso) | +0.113 [+0.049, +0.179] | 0.001 |
| BFM+ICT nativo - BFM-only nel frame dei suoi dati | +0.210 [+0.131, +0.289] | 0.000 |
| BFM+ICT nativo - ICT-only nel frame dei suoi dati | +0.007 [-0.038, +0.058] | 0.386 |
| frazione del gap BFM+ICT - BFM-only spiegata dal frame | -0.86 [-2.88, -0.09] | - |
| BFM+ICT nel frame dei suoi dati - Chamfer eval | -0.023 [-0.080, +0.036] | 0.761 |
| BFM-only nel frame dei suoi dati - Chamfer eval | -0.234 [-0.326, -0.140] | 1.000 |
| ICT-only nel frame dei suoi dati - Chamfer eval | -0.030 [-0.088, +0.021] | 0.861 |

### FaceVerse v2 (4950 righe)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo | rotazione BFM, verso ICT | 0.503 [0.43, 0.57] | 0.452 [0.36, 0.53] | 0.517 [0.44, 0.59] | 0.577 [0.48, 0.66] |
| Rx(180) | ICT completa | 0.377 [0.28, 0.47] | 0.315 [0.20, 0.42] | 0.391 [0.28, 0.48] | 0.577 [0.48, 0.66] |
| facce invertite | BFM completa | mancante | mancante | mancante | 0.577 [0.48, 0.66] |

| contrasto (appaiato, stesse repliche) | stima [CI 95%] | P(boot <= 0) |
| --- | --- | --- |
| BFM-only: frame dei suoi dati (_flip) - nativo | mancante | - |
| ICT-only: frame dei suoi dati (_frame-xmymz) - nativo | -0.125 [-0.222, -0.033] | 0.998 |
| BFM+ICT nativo - BFM-only nativo (gap a parita' di ingresso) | +0.051 [-0.017, +0.119] | 0.077 |
| BFM+ICT nativo - BFM-only nel frame dei suoi dati | mancante | - |
| BFM+ICT nativo - ICT-only nel frame dei suoi dati | +0.112 [-0.003, +0.239] | 0.030 |
| frazione del gap BFM+ICT - BFM-only spiegata dal frame | mancante | - |
| BFM+ICT nel frame dei suoi dati - Chamfer eval | -0.074 [-0.171, +0.033] | 0.925 |
| BFM-only nel frame dei suoi dati - Chamfer eval | mancante | - |
| ICT-only nel frame dei suoi dati - Chamfer eval | -0.186 [-0.281, -0.091] | 1.000 |

### ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) (4950 righe)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo (e2e) | ICT completa | 0.996 [0.99, 1.00] (inquinata (soggetti in training)) | 0.836 [0.78, 0.88] | 0.996 [0.99, 1.00] (inquinata (soggetti in training)) | 0.911 [0.87, 0.94] |
| Rx(180) | rotazione BFM, verso ICT | 0.864 [0.82, 0.90] (inquinata (soggetti in training)) | 0.741 [0.67, 0.80] | 0.916 [0.88, 0.94] (inquinata (soggetti in training)) | 0.911 [0.87, 0.94] |
| Rx(180) + facce invertite | BFM completa | 0.866 [0.82, 0.90] (inquinata (soggetti in training)) | 0.741 [0.67, 0.80] | 0.920 [0.89, 0.94] (inquinata (soggetti in training)) | 0.911 [0.87, 0.94] |

| contrasto (appaiato, stesse repliche) | stima [CI 95%] | P(boot <= 0) |
| --- | --- | --- |
| BFM-only: frame dei suoi dati (_frame-xmymz_flip) - nativo | -0.095 [-0.138, -0.054] | 1.000 |
| BFM+ICT nativo - BFM-only nativo (gap a parita' di ingresso) -- inquinata (soggetti in training) | +0.160 [+0.118, +0.216] | 0.000 |
| BFM+ICT nativo - BFM-only nel frame dei suoi dati -- inquinata (soggetti in training) | +0.255 [+0.192, +0.324] | 0.000 |
| BFM+ICT nativo - ICT-only nel frame dei suoi dati -- inquinata (soggetti in training) | -0.000 [-0.001, +0.000] | 0.773 |
| frazione del gap BFM+ICT - BFM-only spiegata dal frame -- inquinata (soggetti in training) | -0.59 [-0.91, -0.32] | - |
| BFM+ICT nel frame dei suoi dati - Chamfer eval -- inquinata (soggetti in training) | +0.085 [+0.058, +0.119] | 0.000 |
| BFM-only nel frame dei suoi dati - Chamfer eval | -0.170 [-0.224, -0.126] | 1.000 |
| ICT-only nel frame dei suoi dati - Chamfer eval -- inquinata (soggetti in training) | +0.085 [+0.058, +0.120] | 0.000 |

## A PARTE: coppie con crop da un lato (10 coppie ordinate)


### HIFI3D (49500 righe)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo | ICT completa | 0.049 [0.02, 0.07] | 0.135 [0.10, 0.17] | 0.031 [0.00, 0.06] | 0.267 [0.22, 0.31] |
| Rx(180) | rotazione BFM, verso ICT | 0.142 [0.11, 0.18] | 0.144 [0.10, 0.19] | 0.152 [0.11, 0.20] | 0.267 [0.22, 0.31] |
| Rx(180) + facce invertite | BFM completa | 0.143 [0.11, 0.18] | 0.136 [0.10, 0.17] | 0.140 [0.10, 0.19] | 0.267 [0.22, 0.31] |
| Ry(180) (critic) | naso -z, alto +y: nessuna delle due | 0.110 [0.08, 0.15] | 0.109 [0.08, 0.14] | 0.168 [0.12, 0.22] | 0.267 [0.22, 0.31] |
| nativo, eval_frame rms | ICT completa, ri-inquadrato rms | 0.125 [0.08, 0.16] | 0.113 [0.08, 0.14] | 0.051 [0.02, 0.09] | 0.267 [0.22, 0.31] |

| contrasto (appaiato, stesse repliche) | stima [CI 95%] | P(boot <= 0) |
| --- | --- | --- |
| BFM-only: frame dei suoi dati (_frame-xmymz_flip) - nativo | +0.001 [-0.038, +0.037] | 0.468 |
| BFM+ICT nativo - BFM-only nativo (gap a parita' di ingresso) | -0.086 [-0.114, -0.053] | 1.000 |
| BFM+ICT nativo - BFM-only nel frame dei suoi dati | -0.087 [-0.121, -0.047] | 1.000 |
| BFM+ICT nativo - ICT-only nel frame dei suoi dati | +0.018 [+0.005, +0.031] | 0.007 |
| frazione del gap BFM+ICT - BFM-only spiegata dal frame | -0.01 [-0.53, +0.41] (gap con CI che tocca 0: rapporto instabile) | - |
| BFM+ICT nel frame dei suoi dati - Chamfer eval | -0.218 [-0.256, -0.180] | 1.000 |
| BFM-only nel frame dei suoi dati - Chamfer eval | -0.131 [-0.181, -0.082] | 1.000 |
| ICT-only nel frame dei suoi dati - Chamfer eval | -0.236 [-0.267, -0.204] | 1.000 |

### FaceVerse v2 (49500 righe)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo | rotazione BFM, verso ICT | 0.254 [0.20, 0.30] | 0.296 [0.23, 0.35] | 0.097 [0.06, 0.14] | 0.154 [0.09, 0.22] |
| Rx(180) | ICT completa | 0.070 [0.02, 0.11] | 0.164 [0.10, 0.22] | 0.083 [0.03, 0.14] | 0.154 [0.09, 0.22] |
| facce invertite | BFM completa | mancante | mancante | mancante | 0.154 [0.09, 0.22] |

| contrasto (appaiato, stesse repliche) | stima [CI 95%] | P(boot <= 0) |
| --- | --- | --- |
| BFM-only: frame dei suoi dati (_flip) - nativo | mancante | - |
| ICT-only: frame dei suoi dati (_frame-xmymz) - nativo | -0.014 [-0.060, +0.029] | 0.760 |
| BFM+ICT nativo - BFM-only nativo (gap a parita' di ingresso) | -0.041 [-0.091, +0.016] | 0.933 |
| BFM+ICT nativo - BFM-only nel frame dei suoi dati | mancante | - |
| BFM+ICT nativo - ICT-only nel frame dei suoi dati | +0.171 [+0.101, +0.238] | 0.000 |
| frazione del gap BFM+ICT - BFM-only spiegata dal frame | mancante | - |
| BFM+ICT nel frame dei suoi dati - Chamfer eval | +0.101 [+0.019, +0.172] | 0.002 |
| BFM-only nel frame dei suoi dati - Chamfer eval | mancante | - |
| ICT-only nel frame dei suoi dati - Chamfer eval | -0.071 [-0.129, -0.014] | 0.990 |

### ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) (49500 righe)

| variante | ingresso | BFM+ICT | BFM-only | ICT-only | Chamfer eval |
| --- | --- | --- | --- | --- | --- |
| nativo (e2e) | ICT completa | 0.986 [0.98, 0.99] (inquinata (soggetti in training)) | 0.379 [0.32, 0.44] | 0.989 [0.99, 0.99] (inquinata (soggetti in training)) | 0.472 [0.39, 0.54] |
| Rx(180) | rotazione BFM, verso ICT | 0.282 [0.24, 0.32] (inquinata (soggetti in training)) | 0.225 [0.18, 0.27] | 0.509 [0.43, 0.58] (inquinata (soggetti in training)) | 0.472 [0.39, 0.54] |
| Rx(180) + facce invertite | BFM completa | 0.279 [0.24, 0.32] (inquinata (soggetti in training)) | 0.225 [0.18, 0.27] | 0.501 [0.42, 0.57] (inquinata (soggetti in training)) | 0.472 [0.39, 0.54] |

| contrasto (appaiato, stesse repliche) | stima [CI 95%] | P(boot <= 0) |
| --- | --- | --- |
| BFM-only: frame dei suoi dati (_frame-xmymz_flip) - nativo | -0.154 [-0.186, -0.123] | 1.000 |
| BFM+ICT nativo - BFM-only nativo (gap a parita' di ingresso) -- inquinata (soggetti in training) | +0.607 [+0.555, +0.661] | 0.000 |
| BFM+ICT nativo - BFM-only nel frame dei suoi dati -- inquinata (soggetti in training) | +0.761 [+0.720, +0.800] | 0.000 |
| BFM+ICT nativo - ICT-only nel frame dei suoi dati -- inquinata (soggetti in training) | -0.003 [-0.005, -0.002] | 1.000 |
| frazione del gap BFM+ICT - BFM-only spiegata dal frame -- inquinata (soggetti in training) | -0.25 [-0.33, -0.19] | - |
| BFM+ICT nel frame dei suoi dati - Chamfer eval -- inquinata (soggetti in training) | +0.513 [+0.444, +0.589] | 0.000 |
| BFM-only nel frame dei suoi dati - Chamfer eval | -0.248 [-0.288, -0.205] | 1.000 |
| ICT-only nel frame dei suoi dati - Chamfer eval -- inquinata (soggetti in training) | +0.516 [+0.446, +0.593] | 0.000 |

## Coerenza con zs_summarize.py (stessi dati, stessa definizione di Spearman)

| dominio | braccio nativo | protocollo | qui | zs_summarize | 
| --- | --- | --- | --- | --- |
| HIFI3D | BFM+ICT | nocrop | 0.427715 | 0.427715 |
| HIFI3D | BFM-only | nocrop | 0.205971 | 0.205971 |
| HIFI3D | ICT-only | nocrop | 0.381923 | 0.381923 |
| HIFI3D | BFM+ICT | spm | 0.720054 | 0.720054 |
| HIFI3D | BFM-only | spm | 0.606695 | 0.606695 |
| HIFI3D | ICT-only | spm | 0.712774 | 0.712774 |
| FaceVerse v2 | BFM+ICT | nocrop | 0.391595 | 0.391595 |
| FaceVerse v2 | BFM-only | nocrop | 0.364371 | 0.364371 |
| FaceVerse v2 | ICT-only | nocrop | 0.297199 | 0.297199 |
| FaceVerse v2 | BFM+ICT | spm | 0.503187 | 0.503187 |
| FaceVerse v2 | BFM-only | spm | 0.452447 | 0.452447 |
| FaceVerse v2 | ICT-only | spm | 0.516694 | 0.516694 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | BFM+ICT | nocrop | 0.990547 | 0.990547 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | BFM-only | nocrop | 0.259327 | 0.259327 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | ICT-only | nocrop | 0.991426 | 0.991426 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | BFM+ICT | spm | 0.995666 | 0.995666 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | BFM-only | spm | 0.835737 | 0.835737 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | ICT-only | spm | 0.995868 | 0.995868 |

## Crop per singola coppia di topologie (HIFI3D): riga crop -> B, Spearman latent, GT maxabs

SOLO PUNTO, senza CI: in zs_summarize.py le 30 celle per coppia di topologie hanno n_bootstrap=0. Il CI del crop aggregato e' nella sezione A PARTE qui sopra.

| braccio | variante | crop->down8k | crop->noisy | crop->original | crop->remesh | crop->up60k |
| --- | --- | --- | --- | --- | --- | --- |
| BFM+ICT | maxabs (nativo) | 0.042 | 0.113 | 0.140 | 0.067 | 0.069 |
| BFM-only | maxabs (nativo) | 0.064 | 0.388 | 0.362 | 0.152 | 0.172 |
| ICT-only | maxabs (nativo) | 0.023 | 0.091 | 0.114 | 0.048 | 0.043 |
| Chamfer eval | maxabs (nativo) | 0.099 | 0.651 | 0.646 | 0.306 | 0.326 |
| BFM+ICT | rms | 0.072 | 0.192 | 0.232 | 0.166 | 0.136 |
| BFM-only | rms | 0.236 | 0.253 | 0.138 | 0.180 | 0.187 |
| ICT-only | rms | 0.050 | 0.140 | 0.112 | 0.094 | 0.073 |
| Chamfer eval | rms | 0.099 | 0.651 | 0.646 | 0.306 | 0.326 |

## Controllo di sanita': Chamfer eval di ogni colonna contro la prima del dominio

| dominio | colonna | max diff relativa | righe con diff relativa > 1e-4 |
| --- | --- | --- | --- |
| HIFI3D | joint | 0.00e+00 | 0 |
| HIFI3D | bfm_only | 0.00e+00 | 0 |
| HIFI3D | ict_only | 0.00e+00 | 0 |
| HIFI3D | joint_frame-xmymz | 3.27e-02 | 165 |
| HIFI3D | bfm_only_frame-xmymz | 3.27e-02 | 165 |
| HIFI3D | ict_only_frame-xmymz | 3.27e-02 | 165 |
| HIFI3D | joint_frame-xmymz_flip | 3.27e-02 | 165 |
| HIFI3D | bfm_only_frame-xmymz_flip | 3.27e-02 | 165 |
| HIFI3D | ict_only_frame-xmymz_flip | 3.27e-02 | 165 |
| HIFI3D | joint_frame-mxymz | 3.27e-02 | 165 |
| HIFI3D | bfm_only_frame-mxymz | 3.27e-02 | 165 |
| HIFI3D | ict_only_frame-mxymz | 3.27e-02 | 165 |
| HIFI3D | joint_evalframe-rms | 3.27e-02 | 165 |
| HIFI3D | bfm_only_evalframe-rms | 3.27e-02 | 165 |
| HIFI3D | ict_only_evalframe-rms | 3.27e-02 | 165 |
| FaceVerse v2 | joint | 0.00e+00 | 0 |
| FaceVerse v2 | bfm_only | 0.00e+00 | 0 |
| FaceVerse v2 | ict_only | 0.00e+00 | 0 |
| FaceVerse v2 | joint_frame-xmymz | 0.00e+00 | 0 |
| FaceVerse v2 | bfm_only_frame-xmymz | 0.00e+00 | 0 |
| FaceVerse v2 | ict_only_frame-xmymz | 0.00e+00 | 0 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | joint | 0.00e+00 | 0 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | bfm_only | 0.00e+00 | 0 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | ict_only | 0.00e+00 | 0 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | joint_frame-xmymz | 0.00e+00 | 0 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | bfm_only_frame-xmymz | 0.00e+00 | 0 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | ict_only_frame-xmymz | 0.00e+00 | 0 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | joint_frame-xmymz_flip | 0.00e+00 | 0 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | bfm_only_frame-xmymz_flip | 0.00e+00 | 0 |
| ICT, pool id14500-id14999 (controllo e2e; NON held-out per congiunto e ICT-only) | ict_only_frame-xmymz_flip | 0.00e+00 | 0 |

Su HIFI3D il nativo e' girato su L40S e le varianti su A10: 165 righe su 148.500 (tutte con `down8k`) differiscono fino al 3% relativo, con Spearman di Chamfer identico a 2e-6; dove nativo e variante sono sulla stessa GPU (ICT, entrambi A10) la Chamfer e' identica bit per bit.

