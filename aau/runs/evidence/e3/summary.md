# E3 / E3b / E3c: da dove vengono gli errori di riconoscimento di e108

Script: `aau/evidence/e3_breakdown/breakdown.py`. IC 95% bootstrap per soggetto, 1000 repliche (seme `expr_recognition`, le repliche del riconoscimento pubblicato; E3c: seme della riga e108 `nocrop_cross` pubblicata). Distanze e embedding gia' salvati, salvo dove indicato.

## (1) HIFI3D: rank-1 per coppia ordinata di topologie

### e108

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.15 | 0.48 | 0.47 | 0.28 | 0.18 |
| down8k | 0.16 | - | 0.33 | 0.33 | 0.74 | 0.91 |
| noisy | 0.59 | 0.35 | - | 1.00 | 0.92 | 0.79 |
| original | 0.62 | 0.36 | 1.00 | - | 0.97 | 0.79 |
| remesh | 0.34 | 0.82 | 0.95 | 0.97 | - | 1.00 |
| up60k | 0.21 | 0.94 | 0.74 | 0.76 | 0.98 | - |

### Chamfer faceBench

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.03 | 0.95 | 0.91 | 0.22 | 0.16 |
| down8k | 0.02 | - | 0.05 | 0.04 | 0.24 | 0.35 |
| noisy | 0.97 | 0.03 | - | 1.00 | 0.45 | 0.31 |
| original | 0.96 | 0.03 | 1.00 | - | 0.52 | 0.42 |
| remesh | 0.28 | 0.23 | 0.73 | 0.77 | - | 1.00 |
| up60k | 0.22 | 0.30 | 0.50 | 0.58 | 1.00 | - |

### Rigid ICP + Chamfer

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.38 | 0.28 | 0.27 | 0.44 | 0.41 |
| down8k | 0.03 | - | 0.96 | 0.99 | 1.00 | 1.00 |
| noisy | 0.20 | 1.00 | - | 1.00 | 1.00 | 1.00 |
| original | 0.40 | 1.00 | 1.00 | - | 1.00 | 1.00 |
| remesh | 0.26 | 1.00 | 1.00 | 0.99 | - | 1.00 |
| up60k | 0.17 | 1.00 | 0.98 | 0.99 | 1.00 | - |

### e108, rimesh uniforme al test (E3b)

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.13 | 0.09 | 0.11 | 0.27 | 0.11 |
| down8k | 0.23 | - | 0.52 | 1.00 | 0.99 | 1.00 |
| noisy | 0.11 | 0.58 | - | 0.61 | 0.40 | 0.59 |
| original | 0.25 | 1.00 | 0.58 | - | 1.00 | 1.00 |
| remesh | 0.29 | 0.99 | 0.28 | 1.00 | - | 1.00 |
| up60k | 0.23 | 1.00 | 0.53 | 1.00 | 1.00 | - |

### e108, ricalcolato (controllo)

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.15 | 0.48 | 0.47 | 0.28 | 0.18 |
| down8k | 0.16 | - | 0.33 | 0.33 | 0.74 | 0.91 |
| noisy | 0.59 | 0.35 | - | 1.00 | 0.92 | 0.79 |
| original | 0.62 | 0.36 | 1.00 | - | 0.97 | 0.79 |
| remesh | 0.34 | 0.83 | 0.95 | 0.97 | - | 1.00 |
| up60k | 0.21 | 0.94 | 0.74 | 0.76 | 0.98 | - |

### e108, centro per area

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.02 | 0.12 | 0.04 | 0.03 | 0.06 |
| down8k | 0.01 | - | 0.10 | 0.04 | 0.10 | 0.09 |
| noisy | 0.13 | 0.06 | - | 0.08 | 0.23 | 0.38 |
| original | 0.07 | 0.04 | 0.14 | - | 0.15 | 0.16 |
| remesh | 0.05 | 0.16 | 0.24 | 0.14 | - | 0.99 |
| up60k | 0.05 | 0.11 | 0.32 | 0.13 | 0.95 | - |

### e108, pooling medio pesato per area

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.17 | 0.10 | 0.18 | 0.42 | 0.32 |
| down8k | 0.18 | - | 0.55 | 0.12 | 0.31 | 0.30 |
| noisy | 0.16 | 0.46 | - | 0.12 | 0.34 | 0.30 |
| original | 0.19 | 0.08 | 0.11 | - | 0.77 | 0.75 |
| remesh | 0.40 | 0.35 | 0.31 | 0.84 | - | 0.99 |
| up60k | 0.41 | 0.37 | 0.34 | 0.66 | 0.97 | - |

### e108, centro e pooling per area

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.12 | 0.06 | 0.12 | 0.22 | 0.12 |
| down8k | 0.22 | - | 0.64 | 1.00 | 1.00 | 1.00 |
| noisy | 0.09 | 0.72 | - | 0.72 | 0.50 | 0.70 |
| original | 0.20 | 1.00 | 0.66 | - | 1.00 | 1.00 |
| remesh | 0.23 | 1.00 | 0.26 | 1.00 | - | 1.00 |
| up60k | 0.19 | 1.00 | 0.61 | 1.00 | 1.00 | - |

AUC di verifica per coppia (e108):

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.79 | 0.97 | 0.97 | 0.91 | 0.89 |
| down8k | 0.79 | - | 0.92 | 0.92 | 0.99 | 1.00 |
| noisy | 0.97 | 0.92 | - | 1.00 | 1.00 | 0.99 |
| original | 0.97 | 0.92 | 1.00 | - | 1.00 | 0.99 |
| remesh | 0.91 | 0.99 | 1.00 | 1.00 | - | 1.00 |
| up60k | 0.89 | 1.00 | 0.99 | 0.99 | 1.00 | - |

Coppie chiave, rank-1 [IC] (e AUC):

| metodo | original -> down8k | down8k -> original | remesh -> down8k | up60k -> down8k | original -> noisy | remesh -> up60k | crop -> original |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e108 | 0.360 [0.270, 0.450] (0.924) | 0.330 [0.240, 0.420] (0.924) | 0.820 [0.750, 0.890] (0.994) | 0.940 [0.890, 0.980] (0.999) | 1.000 [1.000, 1.000] (1.000) | 1.000 [1.000, 1.000] (1.000) | 0.470 [0.370, 0.560] (0.974) |
| Chamfer faceBench | 0.030 [0.000, 0.070] (0.658) | 0.040 [0.010, 0.080] (0.657) | 0.230 [0.150, 0.320] (0.893) | 0.300 [0.220, 0.390] (0.881) | 1.000 [1.000, 1.000] (1.000) | 1.000 [1.000, 1.000] (1.000) | 0.910 [0.850, 0.960] (0.998) |
| Rigid ICP + Chamfer | 1.000 [1.000, 1.000] (1.000) | 0.990 [0.970, 1.000] (0.999) | 1.000 [1.000, 1.000] (1.000) | 1.000 [1.000, 1.000] (1.000) | 1.000 [1.000, 1.000] (1.000) | 1.000 [1.000, 1.000] (1.000) | 0.270 [0.180, 0.370] (0.857) |
| e108, rimesh uniforme al test (E3b) | 1.000 [1.000, 1.000] (1.000) | 1.000 [1.000, 1.000] (1.000) | 0.990 [0.970, 1.000] (1.000) | 1.000 [1.000, 1.000] (1.000) | 0.580 [0.480, 0.680] (0.970) | 1.000 [1.000, 1.000] (1.000) | 0.110 [0.050, 0.180] (0.886) |
| e108, ricalcolato (controllo) | 0.360 [0.270, 0.450] (0.924) | 0.330 [0.240, 0.420] (0.924) | 0.830 [0.760, 0.900] (0.994) | 0.940 [0.890, 0.980] (0.999) | 1.000 [1.000, 1.000] (1.000) | 1.000 [1.000, 1.000] (1.000) | 0.470 [0.370, 0.560] (0.974) |
| e108, centro per area | 0.040 [0.010, 0.080] (0.628) | 0.040 [0.010, 0.080] (0.628) | 0.160 [0.090, 0.230] (0.806) | 0.110 [0.060, 0.170] (0.748) | 0.140 [0.080, 0.210] (0.800) | 0.990 [0.970, 1.000] (1.000) | 0.040 [0.000, 0.080] (0.751) |
| e108, pooling medio pesato per area | 0.080 [0.030, 0.140] (0.767) | 0.120 [0.060, 0.190] (0.767) | 0.350 [0.260, 0.450] (0.895) | 0.370 [0.280, 0.470] (0.884) | 0.110 [0.050, 0.170] (0.841) | 0.990 [0.970, 1.000] (0.999) | 0.180 [0.110, 0.260] (0.897) |
| e108, centro e pooling per area | 1.000 [1.000, 1.000] (1.000) | 1.000 [1.000, 1.000] (1.000) | 1.000 [1.000, 1.000] (1.000) | 1.000 [1.000, 1.000] (1.000) | 0.660 [0.570, 0.750] (0.987) | 1.000 [1.000, 1.000] (1.000) | 0.120 [0.060, 0.190] (0.890) |

Varianti di e108 meno e108, rank-1 (stesse repliche):

| variante - e108 | original -> down8k | down8k -> original | remesh -> down8k | up60k -> down8k | original -> noisy | remesh -> up60k | crop -> original |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e108, rimesh uniforme al test (E3b) | +0.640 [+0.550, +0.730] | +0.670 [+0.580, +0.760] | +0.170 [+0.100, +0.240] | +0.060 [+0.020, +0.110] | -0.420 [-0.520, -0.320] | +0.000 [+0.000, +0.000] | -0.360 [-0.460, -0.260] |
| e108, ricalcolato (controllo) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.010 [+0.000, +0.030] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| e108, centro per area | -0.320 [-0.410, -0.230] | -0.290 [-0.390, -0.190] | -0.660 [-0.760, -0.550] | -0.830 [-0.900, -0.760] | -0.860 [-0.920, -0.790] | -0.010 [-0.030, +0.000] | -0.430 [-0.530, -0.330] |
| e108, pooling medio pesato per area | -0.280 [-0.390, -0.170] | -0.210 [-0.310, -0.110] | -0.470 [-0.590, -0.350] | -0.570 [-0.670, -0.470] | -0.890 [-0.950, -0.830] | -0.010 [-0.030, +0.000] | -0.290 [-0.390, -0.200] |
| e108, centro e pooling per area | +0.640 [+0.550, +0.730] | +0.670 [+0.580, +0.760] | +0.180 [+0.110, +0.250] | +0.060 [+0.020, +0.110] | -0.340 [-0.430, -0.250] | +0.000 [+0.000, +0.000] | -0.350 [-0.450, -0.250] |

Rapporto genuina / impostore piu' vicino della galleria, mediana per coppia (e108; > 1 = errore):

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 1.14 | 1.01 | 1.02 | 1.11 | 1.10 |
| down8k | 1.19 | - | 1.06 | 1.06 | 0.85 | 0.71 |
| noisy | 0.97 | 1.05 | - | 0.40 | 0.73 | 0.87 |
| original | 0.96 | 1.03 | 0.40 | - | 0.66 | 0.87 |
| remesh | 1.08 | 0.86 | 0.72 | 0.65 | - | 0.64 |
| up60k | 1.13 | 0.71 | 0.88 | 0.89 | 0.64 | - |

## (2) HIFI3D: dispersione intra-identita' / distanza dall'identita' piu' vicina, per topologia

Rapporto per mesh (i, t) sulle 5 topologie senza crop: media delle distanze dalle altre topologie della stessa identita' / min sulle altre identita' della stessa media.

| metodo | topologia | mediana [IC] | IQR | frazione > 1 [IC] |
| --- | --- | --- | --- | --- |
| e108 | down8k | 0.926 [0.869, 0.954] | 0.81-1.05 | 0.330 [0.240, 0.420] |
| e108 | noisy | 0.767 [0.724, 0.787] | 0.69-0.83 | 0.030 [0.000, 0.060] |
| e108 | original | 0.742 [0.717, 0.767] | 0.69-0.82 | 0.000 [0.000, 0.000] |
| e108 | remesh | 0.679 [0.654, 0.697] | 0.62-0.73 | 0.000 [0.000, 0.000] |
| e108 | up60k | 0.739 [0.709, 0.771] | 0.68-0.81 | 0.020 [0.000, 0.050] |
| Chamfer faceBench | down8k | 1.114 [1.094, 1.137] | 1.05-1.20 | 0.860 [0.790, 0.920] |
| Chamfer faceBench | noisy | 1.010 [0.992, 1.031] | 0.95-1.08 | 0.580 [0.480, 0.680] |
| Chamfer faceBench | original | 0.994 [0.969, 1.011] | 0.93-1.04 | 0.490 [0.390, 0.580] |
| Chamfer faceBench | remesh | 0.839 [0.829, 0.852] | 0.81-0.88 | 0.010 [0.000, 0.030] |
| Chamfer faceBench | up60k | 0.857 [0.841, 0.871] | 0.82-0.89 | 0.020 [0.000, 0.050] |
| Rigid ICP + Chamfer | down8k | 0.755 [0.741, 0.769] | 0.73-0.82 | 0.000 [0.000, 0.000] |
| Rigid ICP + Chamfer | noisy | 0.729 [0.713, 0.738] | 0.70-0.75 | 0.000 [0.000, 0.000] |
| Rigid ICP + Chamfer | original | 0.651 [0.636, 0.660] | 0.61-0.69 | 0.000 [0.000, 0.000] |
| Rigid ICP + Chamfer | remesh | 0.705 [0.695, 0.715] | 0.67-0.73 | 0.000 [0.000, 0.000] |
| Rigid ICP + Chamfer | up60k | 0.734 [0.717, 0.751] | 0.70-0.78 | 0.000 [0.000, 0.000] |
| e108, rimesh uniforme al test (E3b) | down8k | 0.547 [0.531, 0.560] | 0.49-0.60 | 0.000 [0.000, 0.000] |
| e108, rimesh uniforme al test (E3b) | noisy | 0.978 [0.942, 1.002] | 0.87-1.08 | 0.430 [0.340, 0.520] |
| e108, rimesh uniforme al test (E3b) | original | 0.531 [0.517, 0.546] | 0.49-0.58 | 0.000 [0.000, 0.000] |
| e108, rimesh uniforme al test (E3b) | remesh | 0.633 [0.607, 0.676] | 0.56-0.72 | 0.000 [0.000, 0.000] |
| e108, rimesh uniforme al test (E3b) | up60k | 0.536 [0.522, 0.559] | 0.49-0.60 | 0.000 [0.000, 0.000] |
| e108, ricalcolato (controllo) | down8k | 0.927 [0.868, 0.955] | 0.81-1.05 | 0.330 [0.240, 0.420] |
| e108, ricalcolato (controllo) | noisy | 0.767 [0.724, 0.787] | 0.69-0.83 | 0.030 [0.000, 0.060] |
| e108, ricalcolato (controllo) | original | 0.742 [0.718, 0.767] | 0.69-0.82 | 0.000 [0.000, 0.000] |
| e108, ricalcolato (controllo) | remesh | 0.679 [0.654, 0.696] | 0.62-0.73 | 0.000 [0.000, 0.000] |
| e108, ricalcolato (controllo) | up60k | 0.738 [0.710, 0.770] | 0.68-0.81 | 0.020 [0.000, 0.050] |
| e108, centro per area | down8k | 1.291 [1.251, 1.346] | 1.14-1.47 | 0.910 [0.850, 0.960] |
| e108, centro per area | noisy | 1.067 [1.022, 1.090] | 0.99-1.14 | 0.720 [0.630, 0.810] |
| e108, centro per area | original | 1.159 [1.109, 1.211] | 1.06-1.29 | 0.850 [0.780, 0.920] |
| e108, centro per area | remesh | 0.888 [0.866, 0.906] | 0.84-0.97 | 0.160 [0.090, 0.230] |
| e108, centro per area | up60k | 0.880 [0.859, 0.901] | 0.82-0.94 | 0.060 [0.020, 0.110] |
| e108, pooling medio pesato per area | down8k | 1.067 [1.036, 1.094] | 0.98-1.20 | 0.700 [0.610, 0.790] |
| e108, pooling medio pesato per area | noisy | 1.022 [0.980, 1.054] | 0.93-1.13 | 0.550 [0.450, 0.650] |
| e108, pooling medio pesato per area | original | 0.983 [0.971, 1.016] | 0.93-1.04 | 0.460 [0.360, 0.560] |
| e108, pooling medio pesato per area | remesh | 0.827 [0.810, 0.843] | 0.77-0.87 | 0.000 [0.000, 0.000] |
| e108, pooling medio pesato per area | up60k | 0.816 [0.797, 0.851] | 0.75-0.91 | 0.020 [0.000, 0.050] |
| e108, centro e pooling per area | down8k | 0.471 [0.452, 0.484] | 0.42-0.51 | 0.000 [0.000, 0.000] |
| e108, centro e pooling per area | noisy | 0.946 [0.915, 0.977] | 0.86-1.05 | 0.330 [0.240, 0.430] |
| e108, centro e pooling per area | original | 0.444 [0.427, 0.458] | 0.40-0.48 | 0.000 [0.000, 0.000] |
| e108, centro e pooling per area | remesh | 0.594 [0.561, 0.606] | 0.53-0.65 | 0.000 [0.000, 0.000] |
| e108, centro e pooling per area | up60k | 0.503 [0.490, 0.523] | 0.46-0.56 | 0.000 [0.000, 0.000] |

## (3) HIFI3D: gli errori cadono sulle identita' vicine in GT?

Query senza crop (2000). Spearman fra rango del match vero e distanza GT dell'identita' query dal suo impostore piu' vicino (negativo = piu' errori quando c'e' un vicino stretto); tasso d'errore di rank-1 per quartile di quella distanza (Q1 = vicino piu' stretto); per le query sbagliate, percentile GT dell'impostore messo al primo posto (0 = il vicino GT; caso: mediana 0.5, entro il 5% con probabilita' 0.05).

| metodo | Spearman(rango, d_GT vicino) [IC] | errore Q1 | Q2 | Q3 | Q4 | query sbagliate | percentile GT del primo impostore, mediana | primo impostore nel 5% GT piu' vicino | = vicino GT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e108 | -0.146 [-0.202, -0.083] | 0.304 [0.244, 0.367] | 0.224 [0.185, 0.270] | 0.194 [0.150, 0.238] | 0.148 [0.103, 0.198] | 435 | 0.173 | 0.257 [0.195, 0.323] | 0.083 [0.048, 0.118] |
| Chamfer faceBench | -0.128 [-0.186, -0.060] | 0.612 [0.543, 0.668] | 0.562 [0.491, 0.623] | 0.486 [0.430, 0.545] | 0.430 [0.373, 0.492] | 1045 | 0.378 | 0.091 [0.061, 0.121] | 0.019 [0.008, 0.031] |
| Rigid ICP + Chamfer | -0.049 [-0.101, 0.023] | 0.012 [0.000, 0.028] | 0.002 [0.000, 0.007] | 0.002 [0.000, 0.007] | 0.002 [0.000, 0.007] | 9 | 0.173 | 0.444 [0.000, 1.000] | 0.333 [0.000, 0.889] |
| e108, rimesh uniforme al test (E3b) | -0.116 [-0.168, -0.050] | 0.260 [0.205, 0.312] | 0.188 [0.143, 0.229] | 0.202 [0.150, 0.257] | 0.136 [0.098, 0.175] | 393 | 0.184 | 0.224 [0.173, 0.286] | 0.051 [0.028, 0.079] |
| e108, ricalcolato (controllo) | -0.147 [-0.203, -0.084] | 0.304 [0.244, 0.367] | 0.224 [0.185, 0.270] | 0.194 [0.150, 0.238] | 0.146 [0.103, 0.195] | 434 | 0.179 | 0.258 [0.195, 0.323] | 0.083 [0.048, 0.118] |
| e108, centro per area | -0.090 [-0.152, -0.026] | 0.788 [0.731, 0.839] | 0.818 [0.772, 0.858] | 0.744 [0.694, 0.795] | 0.728 [0.670, 0.781] | 1539 | 0.235 | 0.170 [0.143, 0.199] | 0.043 [0.029, 0.060] |
| e108, pooling medio pesato per area | -0.072 [-0.138, 0.004] | 0.596 [0.517, 0.665] | 0.548 [0.483, 0.618] | 0.542 [0.475, 0.618] | 0.506 [0.445, 0.569] | 1096 | 0.204 | 0.198 [0.166, 0.229] | 0.056 [0.036, 0.076] |
| e108, centro e pooling per area | -0.133 [-0.197, -0.061] | 0.252 [0.196, 0.302] | 0.136 [0.089, 0.184] | 0.138 [0.093, 0.185] | 0.112 [0.076, 0.154] | 319 | 0.204 | 0.257 [0.178, 0.343] | 0.041 [0.014, 0.076] |

## (1) FaceVerse con espressioni (e108 in convenzione BFM): rank-1 per coppia ordinata di topologie

### e108

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.08 | 0.24 | 0.32 | 0.23 | 0.32 |
| down8k | 0.11 | - | 0.37 | 0.45 | 0.58 | 0.47 |
| noisy | 0.31 | 0.40 | - | 0.78 | 0.65 | 0.84 |
| original | 0.38 | 0.39 | 0.79 | - | 0.69 | 0.79 |
| remesh | 0.24 | 0.53 | 0.67 | 0.71 | - | 0.68 |
| up60k | 0.29 | 0.38 | 0.83 | 0.80 | 0.71 | - |

### Chamfer faceBench

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.25 | 0.55 | 0.36 | 0.46 | 0.33 |
| down8k | 0.31 | - | 0.68 | 0.64 | 0.67 | 0.68 |
| noisy | 0.52 | 0.63 | - | 0.83 | 0.84 | 0.84 |
| original | 0.36 | 0.61 | 0.82 | - | 0.75 | 0.86 |
| remesh | 0.45 | 0.61 | 0.79 | 0.71 | - | 0.77 |
| up60k | 0.37 | 0.61 | 0.84 | 0.84 | 0.78 | - |

### Rigid ICP + Chamfer

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.38 | 0.35 | 0.35 | 0.43 | 0.29 |
| down8k | 0.59 | - | 0.93 | 0.88 | 0.90 | 0.91 |
| noisy | 0.64 | 0.86 | - | 0.93 | 0.94 | 0.96 |
| original | 0.66 | 0.82 | 0.93 | - | 0.93 | 0.95 |
| remesh | 0.74 | 0.84 | 0.94 | 0.95 | - | 0.94 |
| up60k | 0.70 | 0.85 | 0.97 | 0.96 | 0.97 | - |

### e108, rimesh uniforme al test (E3b)

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.27 | 0.20 | 0.33 | 0.33 | 0.26 |
| down8k | 0.30 | - | 0.67 | 0.80 | 0.81 | 0.81 |
| noisy | 0.25 | 0.76 | - | 0.79 | 0.74 | 0.81 |
| original | 0.35 | 0.81 | 0.66 | - | 0.80 | 0.82 |
| remesh | 0.41 | 0.79 | 0.60 | 0.74 | - | 0.78 |
| up60k | 0.35 | 0.78 | 0.73 | 0.83 | 0.82 | - |

### e108, FaceVerse NEUTRO

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.18 | 0.54 | 0.68 | 0.31 | 0.52 |
| down8k | 0.13 | - | 0.70 | 0.72 | 0.91 | 0.81 |
| noisy | 0.55 | 0.83 | - | 0.99 | 0.93 | 1.00 |
| original | 0.67 | 0.84 | 0.99 | - | 0.96 | 1.00 |
| remesh | 0.43 | 0.90 | 0.94 | 0.99 | - | 1.00 |
| up60k | 0.50 | 0.84 | 0.99 | 0.99 | 1.00 | - |

AUC di verifica per coppia (e108):

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.70 | 0.85 | 0.86 | 0.81 | 0.86 |
| down8k | 0.70 | - | 0.83 | 0.85 | 0.87 | 0.85 |
| noisy | 0.85 | 0.83 | - | 0.91 | 0.89 | 0.95 |
| original | 0.86 | 0.85 | 0.91 | - | 0.89 | 0.93 |
| remesh | 0.81 | 0.87 | 0.89 | 0.89 | - | 0.93 |
| up60k | 0.86 | 0.85 | 0.95 | 0.93 | 0.93 | - |

Coppie chiave, rank-1 [IC] (e AUC):

| metodo | original -> down8k | down8k -> original | remesh -> down8k | up60k -> down8k | original -> noisy | remesh -> up60k | crop -> original |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e108 | 0.390 [0.300, 0.490] (0.850) | 0.450 [0.360, 0.550] (0.850) | 0.530 [0.430, 0.630] (0.868) | 0.380 [0.290, 0.470] (0.849) | 0.790 [0.710, 0.870] (0.910) | 0.680 [0.590, 0.760] (0.925) | 0.320 [0.230, 0.420] (0.860) |
| Chamfer faceBench | 0.610 [0.510, 0.700] (0.851) | 0.640 [0.540, 0.730] (0.851) | 0.610 [0.510, 0.700] (0.851) | 0.610 [0.510, 0.710] (0.842) | 0.820 [0.740, 0.890] (0.920) | 0.770 [0.680, 0.850] (0.914) | 0.360 [0.270, 0.450] (0.836) |
| Rigid ICP + Chamfer | 0.820 [0.730, 0.880] (0.972) | 0.880 [0.820, 0.940] (0.984) | 0.840 [0.760, 0.910] (0.974) | 0.850 [0.780, 0.920] (0.975) | 0.930 [0.880, 0.970] (0.982) | 0.940 [0.890, 0.980] (0.987) | 0.350 [0.260, 0.450] (0.776) |
| e108, rimesh uniforme al test (E3b) | 0.810 [0.730, 0.880] (0.926) | 0.800 [0.710, 0.880] (0.926) | 0.790 [0.710, 0.870] (0.922) | 0.780 [0.690, 0.860] (0.913) | 0.660 [0.560, 0.750] (0.918) | 0.780 [0.700, 0.850] (0.934) | 0.330 [0.240, 0.420] (0.855) |
| e108, FaceVerse NEUTRO | 0.840 [0.770, 0.910] (0.982) | 0.720 [0.640, 0.820] (0.982) | 0.900 [0.840, 0.950] (0.998) | 0.840 [0.770, 0.910] (0.991) | 0.990 [0.970, 1.000] (0.998) | 1.000 [1.000, 1.000] (1.000) | 0.680 [0.590, 0.770] (0.983) |

Varianti di e108 meno e108, rank-1 (stesse repliche):

| variante - e108 | original -> down8k | down8k -> original | remesh -> down8k | up60k -> down8k | original -> noisy | remesh -> up60k | crop -> original |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e108, rimesh uniforme al test (E3b) | +0.420 [+0.320, +0.520] | +0.350 [+0.240, +0.450] | +0.260 [+0.170, +0.340] | +0.400 [+0.290, +0.510] | -0.130 [-0.200, -0.070] | +0.100 [+0.040, +0.170] | +0.010 [-0.090, +0.100] |

Rapporto genuina / impostore piu' vicino della galleria, mediana per coppia (e108; > 1 = errore):

| query \ galleria | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 1.29 | 1.11 | 1.09 | 1.17 | 1.12 |
| down8k | 1.22 | - | 1.07 | 1.03 | 0.96 | 1.03 |
| noisy | 1.10 | 1.09 | - | 0.70 | 0.89 | 0.70 |
| original | 1.08 | 1.07 | 0.70 | - | 0.80 | 0.74 |
| remesh | 1.19 | 0.98 | 0.84 | 0.80 | - | 0.76 |
| up60k | 1.11 | 1.08 | 0.72 | 0.73 | 0.77 | - |

## (2) FaceVerse con espressioni (e108 in convenzione BFM): dispersione intra-identita' / distanza dall'identita' piu' vicina, per topologia

Rapporto per mesh (i, t) sulle 5 topologie senza crop: media delle distanze dalle altre topologie della stessa identita' / min sulle altre identita' della stessa media.

| metodo | topologia | mediana [IC] | IQR | frazione > 1 [IC] |
| --- | --- | --- | --- | --- |
| e108 | down8k | 0.982 [0.931, 1.034] | 0.86-1.11 | 0.450 [0.350, 0.550] |
| e108 | noisy | 0.832 [0.785, 0.873] | 0.70-0.96 | 0.190 [0.110, 0.270] |
| e108 | original | 0.764 [0.722, 0.842] | 0.68-0.90 | 0.150 [0.080, 0.230] |
| e108 | remesh | 0.831 [0.780, 0.890] | 0.69-0.97 | 0.230 [0.150, 0.320] |
| e108 | up60k | 0.790 [0.754, 0.837] | 0.66-0.91 | 0.140 [0.070, 0.210] |
| Chamfer faceBench | down8k | 0.869 [0.827, 0.913] | 0.78-1.00 | 0.230 [0.150, 0.310] |
| Chamfer faceBench | noisy | 0.781 [0.759, 0.818] | 0.71-0.90 | 0.120 [0.060, 0.190] |
| Chamfer faceBench | original | 0.794 [0.761, 0.832] | 0.71-0.94 | 0.160 [0.090, 0.240] |
| Chamfer faceBench | remesh | 0.832 [0.783, 0.862] | 0.74-0.94 | 0.170 [0.100, 0.250] |
| Chamfer faceBench | up60k | 0.808 [0.777, 0.843] | 0.72-0.91 | 0.140 [0.070, 0.210] |
| Rigid ICP + Chamfer | down8k | 0.756 [0.724, 0.773] | 0.67-0.84 | 0.050 [0.010, 0.090] |
| Rigid ICP + Chamfer | noisy | 0.756 [0.733, 0.781] | 0.71-0.82 | 0.030 [0.000, 0.070] |
| Rigid ICP + Chamfer | original | 0.702 [0.690, 0.723] | 0.65-0.76 | 0.030 [0.000, 0.070] |
| Rigid ICP + Chamfer | remesh | 0.731 [0.709, 0.761] | 0.67-0.79 | 0.030 [0.000, 0.060] |
| Rigid ICP + Chamfer | up60k | 0.715 [0.692, 0.725] | 0.66-0.78 | 0.020 [0.000, 0.050] |
| e108, rimesh uniforme al test (E3b) | down8k | 0.663 [0.632, 0.730] | 0.56-0.87 | 0.170 [0.100, 0.250] |
| e108, rimesh uniforme al test (E3b) | noisy | 0.806 [0.770, 0.841] | 0.73-0.94 | 0.130 [0.070, 0.200] |
| e108, rimesh uniforme al test (E3b) | original | 0.692 [0.643, 0.761] | 0.56-0.88 | 0.120 [0.060, 0.190] |
| e108, rimesh uniforme al test (E3b) | remesh | 0.712 [0.666, 0.805] | 0.57-0.92 | 0.160 [0.090, 0.240] |
| e108, rimesh uniforme al test (E3b) | up60k | 0.682 [0.613, 0.724] | 0.55-0.82 | 0.100 [0.040, 0.160] |
| e108, FaceVerse NEUTRO | down8k | 0.873 [0.828, 0.900] | 0.78-0.95 | 0.150 [0.080, 0.220] |
| e108, FaceVerse NEUTRO | noisy | 0.645 [0.624, 0.663] | 0.60-0.70 | 0.010 [0.000, 0.030] |
| e108, FaceVerse NEUTRO | original | 0.591 [0.579, 0.609] | 0.55-0.65 | 0.000 [0.000, 0.000] |
| e108, FaceVerse NEUTRO | remesh | 0.696 [0.662, 0.711] | 0.64-0.75 | 0.010 [0.000, 0.030] |
| e108, FaceVerse NEUTRO | up60k | 0.644 [0.609, 0.657] | 0.58-0.70 | 0.000 [0.000, 0.000] |

## (3) FaceVerse con espressioni (e108 in convenzione BFM): gli errori cadono sulle identita' vicine in GT?

Query senza crop (2000). Spearman fra rango del match vero e distanza GT dell'identita' query dal suo impostore piu' vicino (negativo = piu' errori quando c'e' un vicino stretto); tasso d'errore di rank-1 per quartile di quella distanza (Q1 = vicino piu' stretto); per le query sbagliate, percentile GT dell'impostore messo al primo posto (0 = il vicino GT; caso: mediana 0.5, entro il 5% con probabilita' 0.05).

| metodo | Spearman(rango, d_GT vicino) [IC] | errore Q1 | Q2 | Q3 | Q4 | query sbagliate | percentile GT del primo impostore, mediana | primo impostore nel 5% GT piu' vicino | = vicino GT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e108 | -0.023 [-0.113, 0.084] | 0.367 [0.280, 0.443] | 0.420 [0.346, 0.498] | 0.410 [0.315, 0.502] | 0.300 [0.217, 0.398] | 749 | 0.235 | 0.167 [0.131, 0.205] | 0.035 [0.020, 0.052] |
| Chamfer faceBench | -0.057 [-0.142, 0.042] | 0.296 [0.208, 0.387] | 0.276 [0.200, 0.352] | 0.254 [0.167, 0.348] | 0.216 [0.144, 0.296] | 520 | 0.214 | 0.200 [0.158, 0.240] | 0.062 [0.036, 0.091] |
| Rigid ICP + Chamfer | -0.051 [-0.118, 0.028] | 0.098 [0.050, 0.145] | 0.102 [0.056, 0.156] | 0.073 [0.037, 0.113] | 0.056 [0.027, 0.098] | 164 | 0.408 | 0.079 [0.036, 0.129] | 0.030 [0.000, 0.072] |
| e108, rimesh uniforme al test (E3b) | -0.076 [-0.157, 0.022] | 0.273 [0.185, 0.358] | 0.258 [0.186, 0.333] | 0.229 [0.141, 0.325] | 0.172 [0.114, 0.238] | 465 | 0.265 | 0.159 [0.117, 0.206] | 0.037 [0.016, 0.061] |

### FaceVerse: espressione contro topologia nell'embedding di e108

Per ogni identita' i: d_vicino = distanza fra la sua `original` NEUTRA e quella neutra dell'identita' piu' vicina. Colonne: mediana di ||z_espr(i,t) - z_neutro(i,t)|| / d_vicino (solo espressione, stessa topologia; per t diversa da original anche la triangolazione e' rigenerata) e di ||z_neutro(i,t) - z_neutro(i,original)|| / d_vicino (solo topologia).

| topologia | espressione / d_vicino | topologia / d_vicino |
| --- | --- | --- |
| crop | 0.60 | 1.13 |
| down8k | 1.04 | 1.19 |
| noisy | 0.48 | 0.38 |
| original | 0.44 | - |
| remesh | 0.44 | 0.73 |
| up60k | 0.39 | 0.47 |

## E3c: struttura locale (HIFI3D, nocrop_cross, GT maxabs)

Decili delle 4950 coppie di soggetti per distanza GT (decile 1 = coppie piu' vicine; limiti 0.251, 0.276, 0.298, 0.316, 0.337, 0.360, 0.387, 0.427, 0.485); tutte le 20 coppie ordinate di topologie di una coppia di soggetti stanno nello stesso decile. Ristringere l'intervallo della GT abbassa lo Spearman di QUALUNQUE metrica (attenuazione): il confronto giusto e' fra metodi nello stesso decile, e la caduta globale - locale di e108 contro quella delle baseline.

| gruppo | e108 | Chamfer eval | Chamfer faceBench | Rigid ICP + Chamfer | e108, rimesh uniforme al test (E3b) |
| --- | --- | --- | --- | --- | --- |
| globale | 0.630 [0.569, 0.689] | 0.372 [0.326, 0.420] | 0.325 [0.283, 0.369] | 0.355 [0.273, 0.434] | 0.456 [0.374, 0.531] |
| quintile 1 | 0.186 [0.130, 0.244] | 0.114 [0.083, 0.145] | 0.101 [0.072, 0.129] | 0.236 [0.138, 0.319] | 0.121 [0.052, 0.187] |
| decile 1 | 0.188 [0.110, 0.263] | 0.108 [0.066, 0.143] | 0.094 [0.056, 0.127] | 0.276 [0.157, 0.370] | 0.137 [0.044, 0.227] |
| decile 2 | -0.006 [-0.085, 0.077] | 0.030 [-0.014, 0.071] | 0.029 [-0.010, 0.063] | 0.007 [-0.113, 0.127] | -0.033 [-0.122, 0.057] |
| decile 3 | 0.062 [-0.019, 0.144] | 0.027 [-0.020, 0.073] | 0.024 [-0.017, 0.064] | 0.023 [-0.110, 0.145] | 0.074 [-0.025, 0.166] |
| decile 4 | 0.020 [-0.064, 0.101] | 0.013 [-0.041, 0.062] | 0.007 [-0.037, 0.048] | 0.037 [-0.093, 0.157] | 0.033 [-0.069, 0.123] |
| decile 5 | 0.055 [-0.041, 0.150] | 0.048 [0.003, 0.093] | 0.039 [-0.000, 0.080] | 0.080 [-0.046, 0.199] | 0.043 [-0.057, 0.133] |
| decile 6 | 0.080 [-0.010, 0.169] | 0.047 [-0.004, 0.102] | 0.039 [-0.005, 0.087] | 0.088 [-0.053, 0.217] | 0.066 [-0.038, 0.178] |
| decile 7 | 0.049 [-0.043, 0.146] | 0.023 [-0.036, 0.075] | 0.023 [-0.026, 0.070] | 0.036 [-0.085, 0.146] | 0.046 [-0.069, 0.158] |
| decile 8 | 0.155 [0.054, 0.255] | 0.067 [0.003, 0.124] | 0.059 [0.001, 0.109] | 0.040 [-0.111, 0.174] | 0.131 [0.014, 0.239] |
| decile 9 | 0.117 [0.014, 0.223] | 0.079 [0.019, 0.139] | 0.065 [0.018, 0.116] | -0.011 [-0.141, 0.115] | 0.059 [-0.039, 0.166] |
| decile 10 | 0.434 [0.241, 0.553] | 0.200 [0.093, 0.273] | 0.174 [0.084, 0.235] | -0.045 [-0.210, 0.124] | 0.302 [0.077, 0.446] |

Differenze appaiate e108 - baseline per gruppo:

| gruppo | e108 - Chamfer eval (P<=0) | e108 - Chamfer faceBench (P<=0) | e108 - Rigid ICP + Chamfer (P<=0) | e108 - e108, rimesh uniforme al test (E3b) (P<=0) |
| --- | --- | --- | --- | --- |
| globale | +0.258 [+0.210, +0.302] (0.000) | +0.305 [+0.255, +0.350] (0.000) | +0.275 [+0.170, +0.382] (0.000) | +0.174 [+0.127, +0.224] (0.000) |
| quintile 1 | +0.072 [+0.014, +0.131] (0.006) | +0.085 [+0.027, +0.144] (0.000) | -0.049 [-0.135, +0.039] (0.874) | +0.065 [+0.016, +0.115] (0.006) |
| decile 1 | +0.079 [+0.002, +0.148] (0.021) | +0.093 [+0.015, +0.164] (0.005) | -0.088 [-0.205, +0.041] (0.911) | +0.051 [-0.019, +0.118] (0.070) |
| decile 2 | -0.036 [-0.113, +0.042] (0.800) | -0.034 [-0.110, +0.050] (0.788) | -0.013 [-0.140, +0.108] (0.567) | +0.027 [-0.049, +0.093] (0.201) |
| decile 3 | +0.034 [-0.042, +0.114] (0.173) | +0.038 [-0.039, +0.117] (0.155) | +0.038 [-0.088, +0.169] (0.277) | -0.012 [-0.085, +0.070] (0.581) |
| decile 4 | +0.007 [-0.071, +0.092] (0.428) | +0.013 [-0.065, +0.094] (0.379) | -0.017 [-0.148, +0.124] (0.596) | -0.014 [-0.077, +0.054] (0.635) |
| decile 5 | +0.007 [-0.078, +0.087] (0.439) | +0.016 [-0.069, +0.097] (0.363) | -0.025 [-0.177, +0.111] (0.640) | +0.013 [-0.052, +0.079] (0.328) |
| decile 6 | +0.034 [-0.055, +0.121] (0.249) | +0.041 [-0.051, +0.128] (0.183) | -0.008 [-0.147, +0.143] (0.551) | +0.014 [-0.066, +0.094] (0.377) |
| decile 7 | +0.026 [-0.063, +0.118] (0.291) | +0.026 [-0.064, +0.112] (0.294) | +0.013 [-0.133, +0.154] (0.423) | +0.002 [-0.081, +0.091] (0.456) |
| decile 8 | +0.088 [-0.007, +0.190] (0.040) | +0.096 [-0.001, +0.198] (0.030) | +0.115 [-0.050, +0.297] (0.074) | +0.024 [-0.054, +0.100] (0.266) |
| decile 9 | +0.038 [-0.079, +0.160] (0.255) | +0.052 [-0.064, +0.175] (0.181) | +0.128 [-0.039, +0.299] (0.069) | +0.058 [-0.028, +0.142] (0.092) |
| decile 10 | +0.234 [+0.093, +0.330] (0.002) | +0.260 [+0.113, +0.360] (0.002) | +0.479 [+0.243, +0.677] (0.000) | +0.131 [+0.022, +0.237] (0.008) |

Caduta globale - locale (stesse repliche) e differenza delle cadute e108 - baseline (> 0: e108 perde di piu' nel locale):

| gruppo | metodo / confronto | caduta [IC] | P(<=0) |
| --- | --- | --- | --- |
| decile 1 | e108 | +0.442 [+0.358, +0.539] | - |
| decile 1 | Chamfer eval | +0.264 [+0.205, +0.329] | - |
| decile 1 | Chamfer faceBench | +0.231 [+0.181, +0.291] | - |
| decile 1 | Rigid ICP + Chamfer | +0.079 [-0.038, +0.210] | - |
| decile 1 | e108, rimesh uniforme al test (E3b) | +0.320 [+0.202, +0.436] | - |
| decile 1 | caduta e108 - caduta chamfer_eval | +0.178 [+0.091, +0.264] | 0.000 |
| decile 1 | caduta e108 - caduta chamfer | +0.211 [+0.122, +0.298] | 0.000 |
| decile 1 | caduta e108 - caduta rigid_icp_chamfer | +0.363 [+0.211, +0.506] | 0.000 |
| decile 1 | caduta e108 - caduta e108_remesh | +0.123 [+0.035, +0.210] | 0.001 |
| quintile 1 | e108 | +0.444 [+0.369, +0.519] | - |
| quintile 1 | Chamfer eval | +0.258 [+0.209, +0.314] | - |
| quintile 1 | Chamfer faceBench | +0.224 [+0.182, +0.272] | - |
| quintile 1 | Rigid ICP + Chamfer | +0.120 [+0.024, +0.226] | - |
| quintile 1 | e108, rimesh uniforme al test (E3b) | +0.335 [+0.239, +0.439] | - |
| quintile 1 | caduta e108 - caduta chamfer_eval | +0.185 [+0.120, +0.255] | 0.000 |
| quintile 1 | caduta e108 - caduta chamfer | +0.219 [+0.153, +0.288] | 0.000 |
| quintile 1 | caduta e108 - caduta rigid_icp_chamfer | +0.324 [+0.205, +0.441] | 0.000 |
| quintile 1 | caduta e108 - caduta e108_remesh | +0.109 [+0.047, +0.176] | 0.000 |

Residuo relativo |d/media(d) - g/media(g)| / (g/media(g)) per gruppo: media [IC] (mediana):

| gruppo | e108 | Chamfer eval | Chamfer faceBench | Rigid ICP + Chamfer | e108, rimesh uniforme al test (E3b) |
| --- | --- | --- | --- | --- | --- |
| globale | 0.178 [0.167, 0.188] (0.15) | 0.458 [0.436, 0.481] (0.39) | 0.250 [0.238, 0.262] (0.20) | 0.198 [0.185, 0.211] (0.17) | 0.218 [0.205, 0.233] (0.18) |
| quintile 1 | 0.232 [0.216, 0.247] (0.18) | 0.496 [0.472, 0.519] (0.40) | 0.331 [0.310, 0.350] (0.21) | 0.316 [0.293, 0.340] (0.30) | 0.301 [0.275, 0.332] (0.23) |
| decile 1 | 0.256 [0.237, 0.276] (0.19) | 0.517 [0.489, 0.541] (0.41) | 0.384 [0.359, 0.408] (0.26) | 0.384 [0.359, 0.411] (0.37) | 0.339 [0.307, 0.377] (0.26) |
| decile 2 | 0.208 [0.193, 0.224] (0.16) | 0.475 [0.452, 0.498] (0.40) | 0.278 [0.264, 0.293] (0.18) | 0.247 [0.224, 0.271] (0.23) | 0.263 [0.237, 0.294] (0.20) |
| decile 3 | 0.184 [0.173, 0.197] (0.15) | 0.462 [0.438, 0.485] (0.39) | 0.246 [0.233, 0.259] (0.17) | 0.202 [0.180, 0.225] (0.18) | 0.229 [0.211, 0.251] (0.18) |
| decile 4 | 0.173 [0.162, 0.185] (0.14) | 0.447 [0.424, 0.469] (0.38) | 0.225 [0.213, 0.236] (0.17) | 0.154 [0.132, 0.177] (0.12) | 0.204 [0.189, 0.225] (0.16) |
| decile 5 | 0.171 [0.160, 0.183] (0.15) | 0.432 [0.406, 0.458] (0.37) | 0.209 [0.198, 0.221] (0.17) | 0.122 [0.107, 0.139] (0.09) | 0.196 [0.181, 0.215] (0.16) |
| decile 6 | 0.166 [0.153, 0.178] (0.14) | 0.438 [0.414, 0.468] (0.38) | 0.210 [0.199, 0.224] (0.18) | 0.117 [0.103, 0.132] (0.09) | 0.193 [0.178, 0.210] (0.16) |
| decile 7 | 0.164 [0.152, 0.174] (0.14) | 0.445 [0.414, 0.480] (0.37) | 0.213 [0.200, 0.226] (0.19) | 0.113 [0.102, 0.126] (0.09) | 0.184 [0.169, 0.198] (0.16) |
| decile 8 | 0.160 [0.149, 0.172] (0.14) | 0.451 [0.420, 0.486] (0.39) | 0.222 [0.208, 0.235] (0.21) | 0.137 [0.124, 0.153] (0.13) | 0.180 [0.167, 0.194] (0.16) |
| decile 9 | 0.155 [0.140, 0.169] (0.14) | 0.447 [0.412, 0.481] (0.39) | 0.238 [0.224, 0.252] (0.24) | 0.193 [0.171, 0.219] (0.19) | 0.188 [0.168, 0.210] (0.17) |
| decile 10 | 0.139 [0.124, 0.158] (0.13) | 0.465 [0.429, 0.503] (0.41) | 0.274 [0.257, 0.288] (0.29) | 0.309 [0.261, 0.346] (0.32) | 0.203 [0.167, 0.246] (0.19) |

## Geometria delle topologie HIFI3D (perche' down8k)

Il loader centra sulla MEDIA DEI VERTICI e divide per max|V|; il pooling medio del modello e' una media sui vertici. Colonne: mediane sui 100 soggetti.

| topologia | vertici | CV area per vertice | frazione di spigoli di bordo | |media vertici - baricentro area| / max|V| | |media vertici - quella della original| / max|V| | max|V| / quello della original |
| --- | --- | --- | --- | --- | --- | --- |
| crop | 9062 | 1.34 | 0.013 | 0.1457 | 0.0261 | 0.968 |
| down8k | 3312 | 0.75 | 0.020 | 0.1187 | 0.0822 | 0.960 |
| noisy | 9518 | 1.02 | 0.013 | 0.1305 | 0.0001 | 1.008 |
| original | 9518 | 1.38 | 0.013 | 0.1903 | 0.0000 | 1.000 |
| remesh | 6679 | 1.09 | 0.014 | 0.1649 | 0.0339 | 0.959 |
| up60k | 24394 | 0.92 | 0.007 | 0.1577 | 0.0343 | 0.986 |

Spearman, sui 100 soggetti, fra ||z(original) - z(down8k)|| di e108 e: center_shift_vs_original +0.12, maxabs_ratio_vs_original -0.19, vertex_area_cv +0.13

## Controlli

- HIFI3D, e108 ricalcolato nella catena di E3 contro gli embedding esistenti, max |diff| delle distanze: 2.52e-03
- E3c, e108 globale nocrop_cross: 0.630 [0.569, 0.689] (pubblicato 0.630 [0.569, 0.689])
