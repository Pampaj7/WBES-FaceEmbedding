# Quota della taglia nella GT FR, per dominio (sola GT)

Generato da `aau/size_share/size_share.py` (definizioni nel docstring). Coppie di identita' diverse a livello identita', GT FR (mm) e SR (d_P) di `datasets/CANONICAL_GT/eval`, S = centroid size di FR. IC 95% percentile su 1000 repliche per soggetto, le stesse della matrice unificata (seme della T1 del dominio; FaMoS `bootstrap_counts(15, 1000, 1234)`). FaceVerse neutra ha le stesse chiavi e GT di FaceVerse con espressioni: i numeri della GT coincidono. FaMoS TEST: 15 soggetti, descrittivo. Niente Ava-256.

## Tabella

Quota della taglia = mean[(S_i - S_j)^2] / mean[FR^2] (%). rho = Spearman con la GT FR, righe senza crop. Delta = 'tiene la taglia' - 'la toglie'.

| dominio | quota taglia % | CV di S % | rho(FR, taglia) | rho(FR, forma) | oracolo taglia | ICP mm | ICP cs | NoW orig. | NoW metr. | ULRSSM coppia | d_F non cal. | d_F cal. | ICP mm - cs | NoW metr. - orig. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HIFI3D | 36.3 [29.8, 42.5] | 4.7 [4.0, 5.3] | 0.737 [0.657, 0.803] | 0.643 [0.548, 0.732] | 0.737 [0.657, 0.802] | 0.643 [0.572, 0.702] | 0.358 [0.276, 0.438] | 0.416 [0.337, 0.495] | 0.697 [0.628, 0.753] | 0.691 [0.609, 0.757] | 0.642 [0.570, 0.706] | 0.749 [0.673, 0.806] | 0.285 [0.187, 0.383] | 0.281 [0.194, 0.371] |
| FaceScape (dev) | 15.2 [11.2, 19.6] | 1.6 [1.3, 1.8] | 0.463 [0.344, 0.573] | 0.895 [0.849, 0.937] | 0.464 [0.345, 0.571] | 0.453 [0.389, 0.514] | 0.416 [0.344, 0.485] | 0.624 [0.537, 0.701] | 0.677 [0.611, 0.733] | 0.272 [0.239, 0.302] | 0.687 [0.612, 0.758] | 0.661 [0.586, 0.727] | 0.038 [0.018, 0.058] | 0.053 [0.018, 0.092] |
| FaceVerse con espressioni | 3.6 [2.4, 5.0] | 1.8 [1.4, 2.1] | 0.209 [0.114, 0.308] | 0.955 [0.916, 0.982] | 0.209 [0.114, 0.307] | 0.337 [0.262, 0.410] | 0.290 [0.218, 0.358] | 0.300 [0.231, 0.366] | 0.338 [0.264, 0.408] | 0.159 [0.125, 0.191] | 0.297 [0.229, 0.362] | 0.303 [0.229, 0.374] | 0.048 [-0.004, 0.102] | 0.038 [-0.000, 0.078] |
| FaceVerse neutra | 3.6 [2.4, 5.0] | 1.8 [1.4, 2.1] | 0.209 [0.114, 0.308] | 0.955 [0.916, 0.982] | 0.209 [0.114, 0.307] | 0.409 [0.326, 0.481] | 0.374 [0.293, 0.449] | 0.386 [0.309, 0.460] | 0.407 [0.326, 0.481] | 0.195 [0.153, 0.233] | 0.337 [0.256, 0.409] | 0.341 [0.263, 0.416] | 0.035 [-0.018, 0.090] | 0.021 [-0.019, 0.060] |
| FaMoS TEST | 39.6 [21.4, 51.6] | 4.4 [2.6, 5.6] | 0.801 [0.550, 0.912] | 0.694 [0.266, 0.884] | 0.787 [0.546, 0.900] | 0.739 [0.362, 0.914] | 0.601 [0.254, 0.815] | 0.609 [0.265, 0.811] | 0.785 [0.488, 0.925] | 0.774 [0.532, 0.914] | 0.616 [0.222, 0.824] | 0.678 [0.237, 0.897] | 0.138 [-0.125, 0.418] | 0.176 [-0.055, 0.433] |

d_F: factorized s1234 (s2345 in `methods.csv`). ICP, oracolo e d_F sulla maschera comune della matrice (passo `comune`); NoW e ULRSSM sulle 99.000 righe senza crop dei loro CSV (FaMoS 105), stesso seme.

## GT per dominio

| dominio | soggetti | coppie | quota taglia % | quota forma % | SR mediana | SR IQR | Kendall: err. rel. mediana | p95 | max |
|---|---|---|---|---|---|---|---|---|---|
| HIFI3D | 100 | 4950 | 36.3 [29.8, 42.5] | 60.5 [54.7, 66.8] | 0.080 [0.078, 0.084] | 0.023 [0.021, 0.026] | 1.69e-02 | 1.07e-01 | 2.82e-01 |
| FaceScape (dev) | 100 | 4950 | 15.2 [11.2, 19.6] | 82.7 [78.4, 86.6] | 0.049 [0.047, 0.051] | 0.016 [0.014, 0.018] | 1.26e-02 | 8.15e-02 | 2.45e-01 |
| FaceVerse con espressioni | 100 | 4950 | 3.6 [2.4, 5.0] | 95.2 [93.7, 96.4] | 0.127 [0.124, 0.130] | 0.018 [0.016, 0.019] | 7.87e-03 | 3.87e-02 | 1.22e-01 |
| FaceVerse neutra | 100 | 4950 | 3.6 [2.4, 5.0] | 95.2 [93.7, 96.4] | 0.127 [0.124, 0.130] | 0.018 [0.016, 0.019] | 7.87e-03 | 3.87e-02 | 1.22e-01 |
| FaMoS TEST | 15 | 105 | 39.6 [21.4, 51.6] | 59.8 [47.9, 77.6] | 0.077 [0.064, 0.083] | 0.021 [0.013, 0.035] | 5.25e-03 | 2.83e-02 | 5.53e-02 |

Errore di Kendall = |(S_i - S_j)^2 + S_i S_j d_SR^2 - FR^2| / FR^2 sul punto: il resto e' il termine dei centroidi della rigida robusta (quota taglia + quota forma + quota centroidi = 1).

## Controlli

- HIFI3D: seme 990708, CV di S contro `baselines_param/controls_e2.json` 0.0e+00, GT delle righe della matrice contro quella per coppia di soggetti: FR 0.0e+00, SR 0.0e+00 (normalizzate); coppie di soggetti nelle righe 4950; soggetti = blmm.subjects: True.
- FaceScape (dev): seme 796786, CV di S contro `baselines_param/controls_e2.json` 0.0e+00, GT delle righe della matrice contro quella per coppia di soggetti: FR 0.0e+00, SR 0.0e+00 (normalizzate); coppie di soggetti nelle righe 4950; soggetti = blmm.subjects: True.
- FaceVerse con espressioni: seme 271049, CV di S contro `baselines_param/controls_e2.json` 0.0e+00, GT delle righe della matrice contro quella per coppia di soggetti: FR 0.0e+00, SR 0.0e+00 (normalizzate); coppie di soggetti nelle righe 4950; soggetti = blmm.subjects: True.
- FaceVerse neutra: seme 271049, CV di S contro `baselines_param/controls_e2.json` 0.0e+00, GT delle righe della matrice contro quella per coppia di soggetti: FR 0.0e+00, SR 0.0e+00 (normalizzate); coppie di soggetti nelle righe 4950; soggetti = blmm.subjects: True.
- FaMoS TEST: seme 1234, CV di S contro `baselines_param/controls_e2.json` 0.0e+00.

## Fonti dei rho

`methods.csv`, colonna `fonte` (file:riga, riga 1 = intestazione).

