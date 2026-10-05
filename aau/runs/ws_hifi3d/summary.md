# Zero-shot su un 3DMM mai visto: HIFI3D

Modelli: BFM+ICT congiunto (job 1019532), BFM-only (1019310), ICT-only (1019531): gli stessi della tabella WS2, seed 1234, ricetta v1, operatori ad area unitaria. Nessuno ha visto HIFI3D. Dati: 500 identita' dal 3DMM `/home/create.aau.dk/ga41wf/data/hifi3d/files/AI-NEXT-Shape.mat` (sha256 53e252f41e0b..), 500 modi, z ~ N(0,1) senza troncamento, seed 1234; scala: as_is (N(0,1) su basis_shape, come test_basis_io). Regione: `mask_face` del .mat, 9518 vertici / 18684 triangoli (modello intero 20481 vertici). Stesse 6 topologie e stessi operatori di ICT (calcolati su /tmp nel job di eval, per i soli soggetti valutati). Valutati 100 soggetti estratti dal pool con `rebuild_subject_split` seed 1234, gli stessi per i tre modelli e per le baseline. CI 95% bootstrap per soggetto, 1000 ricampionamenti. Risultati da `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a`.

GT `maxabs` = protocollo ICT (vertex-mean-L2 fra le `original` dopo la normalizzazione maxabs, quella di `datasets/ICT/train_ready/gt_matrix.npz`). GT `coef` = distanza L2 fra i coefficienti di identita' standardizzati z. Spearman fra le due GT sul pool: 0.089.

## Mesh-pair cross-topology, clean (protocollo dello 0.301 zero-shot)

| modello | latent (GT maxabs) | chamfer eval (GT maxabs) | latent (GT coef) | chamfer eval (GT coef) |
| --- | --- | --- | --- | --- |
| BFM+ICT | 0.246 [0.21, 0.28] | 0.336 [0.29, 0.38] | 0.066 [0.04, 0.09] | 0.073 [0.03, 0.11] |
| BFM-only | 0.180 [0.14, 0.22] | 0.336 [0.29, 0.38] | 0.046 [0.02, 0.07] | 0.073 [0.03, 0.11] |
| ICT-only | 0.222 [0.18, 0.26] | 0.336 [0.29, 0.38] | 0.047 [0.02, 0.07] | 0.073 [0.03, 0.11] |

### Senza crop (colonna della Tabella 2 del paper)

| modello | latent (GT maxabs) | chamfer eval (GT maxabs) | latent (GT coef) | chamfer eval (GT coef) |
| --- | --- | --- | --- | --- |
| BFM+ICT | 0.428 [0.38, 0.48] | 0.372 [0.32, 0.42] | 0.105 [0.05, 0.15] | 0.080 [0.03, 0.12] |
| BFM-only | 0.206 [0.17, 0.24] | 0.372 [0.32, 0.42] | 0.049 [0.02, 0.08] | 0.080 [0.04, 0.12] |
| ICT-only | 0.382 [0.33, 0.43] | 0.372 [0.33, 0.42] | 0.078 [0.04, 0.12] | 0.080 [0.04, 0.12] |

## Subject-pair-mean (script di ranking), clean

| modello | latent (GT maxabs) | chamfer eval (GT maxabs) | latent (GT coef) | chamfer eval (GT coef) |
| --- | --- | --- | --- | --- |
| BFM+ICT | 0.720 [0.66, 0.78] | 0.743 [0.68, 0.80] | 0.161 [0.09, 0.24] | 0.115 [0.03, 0.20] |
| BFM-only | 0.607 [0.53, 0.68] | 0.743 [0.69, 0.80] | 0.130 [0.05, 0.21] | 0.115 [0.03, 0.20] |
| ICT-only | 0.713 [0.64, 0.78] | 0.743 [0.68, 0.80] | 0.127 [0.05, 0.20] | 0.115 [0.03, 0.20] |

## Subject-pair-mean, mixed (solo punto, solo GT dello script)

| modello | latent (GT maxabs) | chamfer eval (GT maxabs) | latent (GT coef) | chamfer eval (GT coef) |
| --- | --- | --- | --- | --- |
| BFM+ICT | 0.685 | 0.700 | - | - |
| BFM-only | 0.583 | 0.700 | - | - |
| ICT-only | 0.688 | 0.700 | - | - |

## Riferimento: le stesse celle su ICT (tabella WS2)

| modello | ICT mesh-pair all cross, latent | ICT nocrop, latent | ICT subject-pair-mean, latent |
| --- | --- | --- | --- |
| BFM+ICT | 0.981 [0.97, 0.99] | 0.985 [0.98, 0.99] | 0.993 [0.99, 0.99] |
| BFM-only | 0.294 [0.24, 0.34] | 0.259 [0.21, 0.31] | 0.836 [0.79, 0.88] |
| ICT-only | 0.986 [0.98, 0.99] | 0.988 [0.98, 0.99] | 0.994 [0.99, 1.00] |

## Baseline geometriche (pipeline faceBench, stessi soggetti)

| metodo | setting | GT maxabs | GT coef | NaN |
| --- | --- | --- | --- | --- |
| Chamfer (faceBench, 4096 pt) | original_to_original | 0.876 [0.84, 0.90] | 0.162 [0.07, 0.25] | 0 |
| Chamfer (faceBench, 4096 pt) | nocrop_cross_topology | 0.325 [0.28, 0.37] | 0.069 [0.03, 0.11] | 0 |
| Chamfer (faceBench, 4096 pt) | all_cross_topology | 0.291 [0.25, 0.33] | 0.062 [0.02, 0.10] | 0 |
| Rigid ICP + Chamfer | original_to_original | 0.401 [0.31, 0.49] | 0.165 [0.07, 0.25] | 0 |
| Rigid ICP + Chamfer | nocrop_cross_topology | 0.355 [0.27, 0.44] | 0.155 [0.06, 0.24] | 0 |
| Rigid ICP + Chamfer | all_cross_topology | 0.263 [0.19, 0.33] | 0.115 [0.05, 0.18] | 0 |
| Rigid ICP + NICP + P2P | original_to_original | 0.409 [0.31, 0.51] | 0.131 [0.04, 0.22] | 0 |
| Rigid ICP + NICP + P2P | nocrop_cross_topology | 0.355 [0.27, 0.44] | 0.128 [0.05, 0.21] | 776 |
| Rigid ICP + NICP + P2P | all_cross_topology | 0.320 [0.24, 0.39] | 0.121 [0.04, 0.20] | 970 |
| Rigid ICP + NICP + P2Tri | original_to_original | 0.431 [0.33, 0.52] | 0.137 [0.05, 0.22] | 0 |
| Rigid ICP + NICP + P2Tri | nocrop_cross_topology | 0.389 [0.31, 0.47] | 0.139 [0.06, 0.22] | 776 |
| Rigid ICP + NICP + P2Tri | all_cross_topology | 0.346 [0.27, 0.42] | 0.129 [0.05, 0.21] | 970 |

`chamfer eval` nelle tabelle dei modelli e' la Chamfer degli script di eval (media delle distanze al quadrato su tutti i vertici); la riga `Chamfer (faceBench)` qui sopra e' quella della Tabella 2 (4096 punti campionati). `all_cross_topology` = le 30 coppie ordinate cross.


## Controlli

| gt | n_subjects | diag_max_abs | symmetric_max_abs | offdiag_min | offdiag_median | offdiag_std | offdiag_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| maxabs | 500 | 0.0000 | 0.0000 | 0.1357 | 0.3352 | 0.0965 | 1.0000 |
| coef | 500 | 0.0000 | 0.0000 | 0.7513 | 0.8684 | 0.0281 | 1.0000 |

Soggetti: identici nei tre bracci e nelle baseline = True (100, primi id900001, id900004, id900018).


Sorgenti dei bracci: BFM+ICT: joint/ (completo); BFM-only: bfm_only/ (completo); ICT-only: ict_only/ (completo).


## Matrici per coppia di topologie (clean, mesh-pair, GT maxabs; righe A, colonne B)


### BFM+ICT, latent

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.042 | 0.113 | 0.140 | 0.067 | 0.069 |
| down8k | -0.026 | - | 0.134 | 0.131 | 0.355 | 0.382 |
| noisy | 0.050 | 0.219 | - | 0.746 | 0.605 | 0.585 |
| original | 0.087 | 0.214 | 0.751 | - | 0.593 | 0.539 |
| remesh | 0.010 | 0.436 | 0.576 | 0.549 | - | 0.742 |
| up60k | 0.006 | 0.468 | 0.519 | 0.464 | 0.714 | - |

### BFM-only, latent

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.064 | 0.388 | 0.362 | 0.152 | 0.172 |
| down8k | 0.041 | - | 0.105 | 0.117 | 0.293 | 0.243 |
| noisy | 0.396 | 0.098 | - | 0.634 | 0.258 | 0.277 |
| original | 0.379 | 0.112 | 0.663 | - | 0.306 | 0.325 |
| remesh | 0.141 | 0.280 | 0.306 | 0.355 | - | 0.642 |
| up60k | 0.175 | 0.239 | 0.328 | 0.371 | 0.619 | - |

### ICT-only, latent

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.023 | 0.091 | 0.114 | 0.048 | 0.043 |
| down8k | -0.031 | - | 0.088 | 0.076 | 0.290 | 0.334 |
| noisy | 0.056 | 0.143 | - | 0.780 | 0.566 | 0.528 |
| original | 0.080 | 0.128 | 0.777 | - | 0.503 | 0.458 |
| remesh | -0.002 | 0.329 | 0.538 | 0.479 | - | 0.749 |
| up60k | -0.006 | 0.369 | 0.487 | 0.421 | 0.735 | - |

### Chamfer eval (uguale per i tre bracci a meno del campione: si riporta il congiunto)

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.099 | 0.651 | 0.646 | 0.306 | 0.326 |
| down8k | 0.036 | - | 0.084 | 0.103 | 0.398 | 0.369 |
| noisy | 0.656 | 0.139 | - | 0.829 | 0.449 | 0.478 |
| original | 0.650 | 0.157 | 0.829 | - | 0.512 | 0.535 |
| remesh | 0.287 | 0.441 | 0.436 | 0.502 | - | 0.789 |
| up60k | 0.304 | 0.404 | 0.457 | 0.517 | 0.794 | - |
