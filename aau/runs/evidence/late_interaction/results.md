# Interazione tardiva sulle feature per vertice: risultati

Generato da `aau/late_interaction/li_eval.py` (non modificare a mano). Regole in `REGOLE.md` (commit separato, prima dei numeri). Righe, maschera comune, seme e 1000 repliche per soggetto della matrice (`full_matrix`), solo insiemi senza crop; IC 95% percentile; delta appaiati = interazione tardiva - colonna, replica per replica (positivo = interazione tardiva davanti). Lettura `fr` = ricomposta con S e c' (POST HOC), con la GT FR; lettura `sr` = d_LI, con la GT SR.

## Scelta sul pilota (soggetti NON valutati)

Variante dichiarata: **sinkhorn** (interazione tardiva, OT entropico sul sottocampione). Regola: media su semi (s1234, s2345) e piloti (HIFI3D, FaceScape) di (rho(fr, GT FR) + rho(sr, GT SR)) / 2 sulle coppie del pilota; entro 0.01 dalla migliore la mediana dei secondi per coppia isolata piu' bassa.

| variante | punteggio | ms per coppia isolata (GPU) |
|---|---|---|
| chamfer | 0.4851 | 4.69 |
| chamfer_sub | 0.4860 | 0.37 |
| sinkhorn **(dichiarata)** | 0.5466 | 38.88 |

Riferimento sulle stesse coppie del pilota (vettori della stessa forward, fuori dalla regola): rho medio d_F cal. con GT FR 0.7022, d_P con GT SR 0.7661.

## Calibrazione di c' (held-out dei generatori di training)

| seme | variante | coppie | c' (mediana) | c' LS | bfm / ict / gnm |
|---|---|---|---|---|---|
| s1234 | chamfer | 3960 | 0.10541 | 0.092576 | 0.1423 / 0.07172 / 0.1136 |
| s1234 | chamfer_sub | 3960 | 0.076989 | 0.070273 | 0.09283 / 0.053 / 0.08621 |
| s1234 | sinkhorn | 3960 | 0.027806 | 0.027527 | 0.03016 / 0.02255 / 0.03094 |
| s1234 | _vettore_d_P (c ufficiale 0.4053) | 3960 | 0.41627 | 0.40456 | 0.4371 / 0.3879 / 0.4221 |
| s2345 | chamfer | 3960 | 0.10807 | 0.093852 | 0.1428 / 0.07369 / 0.1167 |
| s2345 | chamfer_sub | 3960 | 0.077743 | 0.070648 | 0.09196 / 0.05382 / 0.08726 |
| s2345 | sinkhorn | 3960 | 0.027379 | 0.027066 | 0.02946 / 0.02223 / 0.03053 |
| s2345 | _vettore_d_P (c ufficiale 0.4054) | 3960 | 0.41819 | 0.4061 | 0.4113 / 0.3846 / 0.4379 |

## hifi3d: Spearman (seme delle repliche 990708)

GT FR (grassetto = variante dichiarata):

| colonna | nocrop | stessa topologia senza crop | con original | con remesh | con down8k | con up60k | con noisy |
|---|---|---|---|---|---|---|---|
| factorized s1234, d_F cal. | 0.749 [0.673, 0.806] | 0.760 [0.685, 0.817] | 0.754 [0.679, 0.811] | 0.741 [0.668, 0.798] | 0.752 [0.678, 0.808] | 0.751 [0.674, 0.808] | 0.747 [0.670, 0.806] |
| factorized s1234, d_F non cal. | 0.642 [0.570, 0.706] | 0.653 [0.582, 0.717] | 0.649 [0.577, 0.712] | 0.642 [0.571, 0.705] | 0.644 [0.572, 0.707] | 0.646 [0.573, 0.710] | 0.633 [0.560, 0.698] |
| factorized s1234, d_P | 0.426 [0.333, 0.510] | 0.434 [0.341, 0.518] | 0.432 [0.339, 0.517] | 0.429 [0.337, 0.514] | 0.427 [0.335, 0.510] | 0.429 [0.335, 0.513] | 0.413 [0.317, 0.501] |
| factorized s2345, d_F cal. | 0.731 [0.657, 0.792] | 0.744 [0.671, 0.805] | 0.737 [0.663, 0.799] | 0.717 [0.645, 0.778] | 0.738 [0.663, 0.798] | 0.731 [0.659, 0.793] | 0.731 [0.657, 0.793] |
| factorized s2345, d_F non cal. | 0.631 [0.560, 0.697] | 0.639 [0.570, 0.705] | 0.636 [0.565, 0.701] | 0.629 [0.559, 0.694] | 0.634 [0.564, 0.700] | 0.634 [0.562, 0.699] | 0.623 [0.551, 0.689] |
| factorized s2345, d_P | 0.417 [0.315, 0.512] | 0.423 [0.319, 0.518] | 0.421 [0.319, 0.517] | 0.419 [0.317, 0.513] | 0.420 [0.318, 0.515] | 0.418 [0.315, 0.513] | 0.409 [0.306, 0.501] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | 0.697 [0.628, 0.753] | - | 0.702 [0.634, 0.758] | 0.698 [0.628, 0.755] | 0.710 [0.640, 0.765] | 0.700 [0.631, 0.756] | 0.719 [0.651, 0.775] |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.495 [0.402, 0.575] | - | 0.498 [0.406, 0.579] | 0.498 [0.406, 0.577] | 0.493 [0.400, 0.574] | 0.498 [0.404, 0.578] | 0.498 [0.405, 0.578] |
| ICP + Chamfer in mm (per coppia) | 0.643 [0.572, 0.702] | 0.659 [0.588, 0.719] | 0.637 [0.565, 0.696] | 0.668 [0.596, 0.729] | 0.621 [0.552, 0.680] | 0.666 [0.595, 0.727] | 0.657 [0.587, 0.715] |
| FLAME 2023 Open vB, composizione S_B + k d_P | 0.712 [0.627, 0.781] | 0.715 [0.629, 0.784] | 0.715 [0.631, 0.784] | 0.708 [0.622, 0.778] | 0.713 [0.627, 0.783] | 0.715 [0.632, 0.784] | 0.706 [0.621, 0.776] |
| GNM (visto) vB, coefficienti | 0.550 [0.465, 0.628] | 0.549 [0.465, 0.627] | 0.551 [0.466, 0.630] | 0.551 [0.468, 0.629] | 0.547 [0.462, 0.626] | 0.547 [0.463, 0.626] | 0.554 [0.468, 0.632] |
| ULRSSM [faces_xyz] per coppia, mm (FR) | 0.691 [0.609, 0.757] | 0.780 [0.710, 0.837] | 0.704 [0.624, 0.769] | 0.710 [0.631, 0.775] | 0.705 [0.626, 0.770] | 0.710 [0.631, 0.775] | 0.704 [0.633, 0.763] |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.297 [0.215, 0.367] | 0.411 [0.311, 0.501] | 0.317 [0.231, 0.390] | 0.320 [0.232, 0.393] | 0.324 [0.236, 0.399] | 0.320 [0.231, 0.393] | 0.387 [0.288, 0.475] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | 0.629 [0.547, 0.697] | 0.752 [0.671, 0.818] | 0.649 [0.564, 0.721] | 0.657 [0.573, 0.729] | 0.667 [0.589, 0.730] | 0.658 [0.572, 0.731] | 0.557 [0.482, 0.619] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.248 [0.163, 0.334] | 0.331 [0.241, 0.424] | 0.245 [0.157, 0.334] | 0.232 [0.145, 0.322] | 0.287 [0.200, 0.375] | 0.246 [0.155, 0.339] | 0.302 [0.206, 0.388] |
| interazione tardiva, Chamfer su tutti i vertici [factorized s1234] (non dichiarata), ricomposta con S e c' (FR, post hoc) | 0.684 [0.599, 0.755] | 0.691 [0.605, 0.763] | 0.689 [0.604, 0.761] | 0.675 [0.589, 0.746] | 0.690 [0.605, 0.761] | 0.686 [0.601, 0.758] | 0.689 [0.604, 0.760] |
| interazione tardiva, Chamfer sul sottocampione [factorized s1234] (non dichiarata), ricomposta con S e c' (FR, post hoc) | 0.667 [0.577, 0.743] | 0.680 [0.590, 0.756] | 0.671 [0.581, 0.748] | 0.658 [0.568, 0.732] | 0.671 [0.581, 0.747] | 0.667 [0.576, 0.743] | 0.671 [0.582, 0.748] |
| **interazione tardiva, OT entropico sul sottocampione [factorized s1234], ricomposta con S e c' (FR, post hoc)** | 0.662 [0.571, 0.737] | 0.672 [0.582, 0.747] | 0.666 [0.574, 0.740] | 0.653 [0.563, 0.727] | 0.665 [0.573, 0.739] | 0.663 [0.571, 0.737] | 0.665 [0.575, 0.742] |
| interazione tardiva, Chamfer su tutti i vertici [factorized s2345] (non dichiarata), ricomposta con S e c' (FR, post hoc) | 0.670 [0.586, 0.743] | 0.679 [0.592, 0.754] | 0.676 [0.591, 0.750] | 0.659 [0.576, 0.731] | 0.678 [0.593, 0.751] | 0.671 [0.587, 0.745] | 0.674 [0.590, 0.747] |
| interazione tardiva, Chamfer sul sottocampione [factorized s2345] (non dichiarata), ricomposta con S e c' (FR, post hoc) | 0.654 [0.565, 0.731] | 0.668 [0.581, 0.746] | 0.659 [0.570, 0.736] | 0.640 [0.551, 0.717] | 0.660 [0.572, 0.737] | 0.652 [0.565, 0.730] | 0.659 [0.570, 0.736] |
| **interazione tardiva, OT entropico sul sottocampione [factorized s2345], ricomposta con S e c' (FR, post hoc)** | 0.652 [0.561, 0.730] | 0.666 [0.575, 0.743] | 0.657 [0.565, 0.735] | 0.638 [0.547, 0.716] | 0.658 [0.566, 0.735] | 0.651 [0.559, 0.729] | 0.657 [0.566, 0.735] |

GT SR (grassetto = variante dichiarata):

| colonna | nocrop | stessa topologia senza crop | con original | con remesh | con down8k | con up60k | con noisy |
|---|---|---|---|---|---|---|---|
| factorized s1234, d_F cal. | 0.318 [0.227, 0.407] | 0.325 [0.233, 0.415] | 0.322 [0.232, 0.411] | 0.320 [0.230, 0.408] | 0.320 [0.229, 0.410] | 0.319 [0.229, 0.407] | 0.310 [0.217, 0.403] |
| factorized s1234, d_F non cal. | 0.551 [0.463, 0.630] | 0.563 [0.474, 0.642] | 0.558 [0.470, 0.636] | 0.555 [0.468, 0.634] | 0.554 [0.466, 0.634] | 0.554 [0.465, 0.633] | 0.535 [0.446, 0.613] |
| factorized s1234, d_P | 0.622 [0.550, 0.685] | 0.634 [0.562, 0.699] | 0.629 [0.558, 0.692] | 0.626 [0.553, 0.690] | 0.625 [0.555, 0.688] | 0.626 [0.554, 0.690] | 0.604 [0.533, 0.668] |
| factorized s2345, d_F cal. | 0.312 [0.220, 0.401] | 0.317 [0.224, 0.409] | 0.315 [0.223, 0.405] | 0.311 [0.220, 0.398] | 0.316 [0.222, 0.406] | 0.312 [0.220, 0.401] | 0.308 [0.216, 0.397] |
| factorized s2345, d_F non cal. | 0.539 [0.445, 0.621] | 0.547 [0.453, 0.629] | 0.543 [0.450, 0.626] | 0.541 [0.448, 0.622] | 0.543 [0.449, 0.625] | 0.541 [0.446, 0.622] | 0.529 [0.433, 0.610] |
| factorized s2345, d_P | 0.613 [0.533, 0.688] | 0.620 [0.541, 0.693] | 0.617 [0.537, 0.691] | 0.616 [0.537, 0.690] | 0.616 [0.537, 0.690] | 0.616 [0.537, 0.689] | 0.600 [0.519, 0.675] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | 0.430 [0.349, 0.509] | - | 0.437 [0.356, 0.517] | 0.431 [0.349, 0.511] | 0.432 [0.347, 0.513] | 0.428 [0.345, 0.510] | 0.446 [0.363, 0.527] |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.715 [0.645, 0.771] | - | 0.720 [0.652, 0.775] | 0.718 [0.650, 0.775] | 0.715 [0.646, 0.772] | 0.722 [0.653, 0.779] | 0.717 [0.648, 0.774] |
| ICP + Chamfer in mm (per coppia) | 0.365 [0.280, 0.452] | 0.377 [0.292, 0.466] | 0.367 [0.285, 0.454] | 0.382 [0.295, 0.474] | 0.344 [0.261, 0.428] | 0.375 [0.286, 0.466] | 0.377 [0.292, 0.465] |
| FLAME 2023 Open vB, composizione S_B + k d_P | 0.346 [0.255, 0.437] | 0.348 [0.258, 0.439] | 0.348 [0.258, 0.437] | 0.344 [0.253, 0.436] | 0.349 [0.257, 0.438] | 0.347 [0.255, 0.437] | 0.341 [0.252, 0.430] |
| GNM (visto) vB, coefficienti | 0.674 [0.610, 0.734] | 0.673 [0.607, 0.733] | 0.676 [0.610, 0.735] | 0.674 [0.607, 0.733] | 0.672 [0.606, 0.732] | 0.673 [0.607, 0.733] | 0.680 [0.614, 0.738] |
| ULRSSM [faces_xyz] per coppia, mm (FR) | 0.269 [0.181, 0.356] | 0.315 [0.220, 0.412] | 0.273 [0.186, 0.362] | 0.278 [0.190, 0.366] | 0.274 [0.186, 0.362] | 0.274 [0.185, 0.362] | 0.304 [0.217, 0.391] |
| ULRSSM [faces_xyz] per coppia, a CS_ref (SR) | 0.479 [0.412, 0.541] | 0.666 [0.596, 0.730] | 0.514 [0.448, 0.576] | 0.517 [0.451, 0.579] | 0.524 [0.457, 0.588] | 0.520 [0.452, 0.582] | 0.629 [0.557, 0.696] |
| ULRSSM [faces_xyz] via template, GT FR (FR) | 0.298 [0.226, 0.369] | 0.355 [0.268, 0.441] | 0.303 [0.227, 0.377] | 0.310 [0.232, 0.385] | 0.317 [0.241, 0.392] | 0.306 [0.227, 0.383] | 0.287 [0.221, 0.351] |
| ULRSSM [faces_xyz] via template, GT SR (d_P) (SR) | 0.482 [0.410, 0.551] | 0.600 [0.526, 0.674] | 0.486 [0.410, 0.560] | 0.476 [0.400, 0.552] | 0.535 [0.463, 0.604] | 0.492 [0.416, 0.568] | 0.554 [0.484, 0.621] |
| interazione tardiva, Chamfer su tutti i vertici [factorized s1234] (non dichiarata), d_LI (SR) | 0.302 [0.213, 0.388] | 0.300 [0.210, 0.388] | 0.308 [0.218, 0.395] | 0.303 [0.213, 0.389] | 0.313 [0.222, 0.400] | 0.307 [0.215, 0.394] | 0.308 [0.221, 0.391] |
| interazione tardiva, Chamfer sul sottocampione [factorized s1234] (non dichiarata), d_LI (SR) | 0.304 [0.213, 0.390] | 0.323 [0.228, 0.410] | 0.310 [0.217, 0.398] | 0.306 [0.215, 0.394] | 0.313 [0.221, 0.400] | 0.301 [0.209, 0.387] | 0.307 [0.219, 0.392] |
| **interazione tardiva, OT entropico sul sottocampione [factorized s1234], d_LI (SR)** | 0.453 [0.363, 0.535] | 0.470 [0.377, 0.555] | 0.458 [0.367, 0.541] | 0.458 [0.367, 0.541] | 0.459 [0.369, 0.542] | 0.451 [0.362, 0.534] | 0.448 [0.362, 0.530] |
| interazione tardiva, Chamfer su tutti i vertici [factorized s2345] (non dichiarata), d_LI (SR) | 0.284 [0.192, 0.375] | 0.277 [0.185, 0.370] | 0.287 [0.194, 0.379] | 0.285 [0.193, 0.376] | 0.292 [0.198, 0.385] | 0.289 [0.197, 0.382] | 0.287 [0.197, 0.376] |
| interazione tardiva, Chamfer sul sottocampione [factorized s2345] (non dichiarata), d_LI (SR) | 0.290 [0.197, 0.384] | 0.299 [0.204, 0.396] | 0.292 [0.199, 0.387] | 0.290 [0.197, 0.384] | 0.294 [0.200, 0.388] | 0.288 [0.194, 0.382] | 0.291 [0.200, 0.383] |
| **interazione tardiva, OT entropico sul sottocampione [factorized s2345], d_LI (SR)** | 0.444 [0.351, 0.536] | 0.454 [0.360, 0.548] | 0.446 [0.353, 0.539] | 0.447 [0.354, 0.540] | 0.446 [0.353, 0.539] | 0.443 [0.350, 0.534] | 0.440 [0.347, 0.531] |

Delta appaiati della variante dichiarata (sinkhorn), lettura fr, GT FR:

| seme | contro | nocrop | stessa topologia senza crop | con original | con remesh | con down8k | con up60k | con noisy |
|---|---|---|---|---|---|---|---|---|
| s1234 | factorized s1234, d_F cal. | -0.087 [-0.121, -0.053] | -0.087 [-0.122, -0.053] | -0.088 [-0.123, -0.054] | -0.088 [-0.123, -0.054] | -0.087 [-0.122, -0.052] | -0.088 [-0.122, -0.054] | -0.081 [-0.115, -0.048] |
| s1234 | factorized s1234, d_F non cal. | +0.020 [-0.054, +0.095] | +0.019 [-0.053, +0.095] | +0.017 [-0.056, +0.092] | +0.010 [-0.062, +0.086] | +0.021 [-0.055, +0.095] | +0.016 [-0.057, +0.092] | +0.032 [-0.040, +0.109] |
| s1234 | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.035 [-0.080, +0.017] | - | -0.036 [-0.082, +0.017] | -0.045 [-0.092, +0.005] | -0.045 [-0.091, +0.006] | -0.037 [-0.083, +0.014] | -0.054 [-0.101, -0.001] |
| s1234 | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | +0.167 [+0.037, +0.306] | - | +0.167 [+0.038, +0.307] | +0.155 [+0.025, +0.294] | +0.171 [+0.039, +0.312] | +0.165 [+0.032, +0.304] | +0.168 [+0.037, +0.307] |
| s1234 | ICP + Chamfer in mm (per coppia) | +0.019 [-0.030, +0.073] | +0.013 [-0.038, +0.068] | +0.029 [-0.020, +0.083] | -0.016 [-0.064, +0.037] | +0.044 [-0.004, +0.097] | -0.004 [-0.054, +0.050] | +0.008 [-0.043, +0.064] |
| s2345 | factorized s2345, d_F cal. | -0.079 [-0.113, -0.047] | -0.079 [-0.113, -0.047] | -0.080 [-0.115, -0.048] | -0.079 [-0.114, -0.048] | -0.080 [-0.114, -0.048] | -0.080 [-0.114, -0.049] | -0.074 [-0.108, -0.044] |
| s2345 | factorized s2345, d_F non cal. | +0.021 [-0.057, +0.094] | +0.026 [-0.053, +0.101] | +0.022 [-0.058, +0.095] | +0.008 [-0.068, +0.081] | +0.023 [-0.055, +0.096] | +0.017 [-0.060, +0.091] | +0.034 [-0.045, +0.108] |
| s2345 | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.045 [-0.096, +0.008] | - | -0.045 [-0.097, +0.009] | -0.060 [-0.110, -0.007] | -0.052 [-0.100, -0.000] | -0.049 [-0.098, +0.003] | -0.062 [-0.115, -0.008] |
| s2345 | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | +0.157 [+0.028, +0.296] | - | +0.159 [+0.030, +0.299] | +0.140 [+0.010, +0.278] | +0.164 [+0.034, +0.302] | +0.153 [+0.022, +0.294] | +0.159 [+0.030, +0.298] |
| s2345 | ICP + Chamfer in mm (per coppia) | +0.009 [-0.044, +0.064] | +0.006 [-0.048, +0.064] | +0.020 [-0.032, +0.076] | -0.031 [-0.082, +0.023] | +0.037 [-0.015, +0.091] | -0.015 [-0.067, +0.039] | -0.000 [-0.056, +0.057] |
| s1234 | miglior concorrente della matrice | -0.050 [-0.098, -0.002] (FLAME 2023 Open vB, composizione S_B + k d_P) | -0.108 [-0.147, -0.073] (ULRSSM [faces_xyz] per coppia, mm (FR)) | -0.049 [-0.098, -0.003] (FLAME 2023 Open vB, composizione S_B + k d_P) | -0.058 [-0.094, -0.026] (ULRSSM [faces_xyz] per coppia, mm (FR)) | -0.048 [-0.097, +0.001] (FLAME 2023 Open vB, composizione S_B + k d_P) | -0.053 [-0.099, -0.005] (FLAME 2023 Open vB, composizione S_B + k d_P) | -0.054 [-0.101, -0.001] (NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)) |
| s2345 | miglior concorrente della matrice | -0.060 [-0.109, -0.008] (FLAME 2023 Open vB, composizione S_B + k d_P) | -0.114 [-0.155, -0.076] (ULRSSM [faces_xyz] per coppia, mm (FR)) | -0.058 [-0.108, -0.007] (FLAME 2023 Open vB, composizione S_B + k d_P) | -0.073 [-0.109, -0.037] (ULRSSM [faces_xyz] per coppia, mm (FR)) | -0.055 [-0.106, -0.003] (FLAME 2023 Open vB, composizione S_B + k d_P) | -0.064 [-0.114, -0.012] (FLAME 2023 Open vB, composizione S_B + k d_P) | -0.062 [-0.115, -0.008] (NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)) |

Delta appaiati della variante dichiarata (sinkhorn), lettura sr, GT SR:

| seme | contro | nocrop | stessa topologia senza crop | con original | con remesh | con down8k | con up60k | con noisy |
|---|---|---|---|---|---|---|---|---|
| s1234 | factorized s1234, d_P | -0.169 [-0.241, -0.104] | -0.164 [-0.239, -0.097] | -0.172 [-0.245, -0.107] | -0.168 [-0.241, -0.104] | -0.166 [-0.240, -0.101] | -0.176 [-0.248, -0.112] | -0.156 [-0.230, -0.088] |
| s1234 | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | +0.023 [-0.035, +0.086] | - | +0.021 [-0.038, +0.085] | +0.027 [-0.033, +0.091] | +0.027 [-0.031, +0.090] | +0.022 [-0.036, +0.087] | +0.002 [-0.057, +0.067] |
| s1234 | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.262 [-0.349, -0.178] | - | -0.262 [-0.350, -0.176] | -0.261 [-0.349, -0.175] | -0.256 [-0.344, -0.169] | -0.271 [-0.359, -0.186] | -0.269 [-0.353, -0.187] |
| s1234 | ICP + Chamfer in mm (per coppia) | +0.087 [+0.026, +0.153] | +0.093 [+0.030, +0.161] | +0.090 [+0.029, +0.156] | +0.075 [+0.011, +0.143] | +0.115 [+0.054, +0.181] | +0.076 [+0.012, +0.141] | +0.071 [+0.009, +0.140] |
| s2345 | factorized s2345, d_P | -0.169 [-0.243, -0.102] | -0.166 [-0.241, -0.098] | -0.171 [-0.246, -0.105] | -0.169 [-0.244, -0.104] | -0.170 [-0.245, -0.104] | -0.173 [-0.249, -0.107] | -0.159 [-0.236, -0.091] |
| s2345 | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | +0.014 [-0.048, +0.083] | - | +0.009 [-0.053, +0.079] | +0.016 [-0.047, +0.085] | +0.015 [-0.046, +0.083] | +0.014 [-0.047, +0.084] | -0.006 [-0.066, +0.064] |
| s2345 | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.271 [-0.365, -0.181] | - | -0.274 [-0.369, -0.183] | -0.272 [-0.366, -0.182] | -0.269 [-0.364, -0.178] | -0.279 [-0.373, -0.188] | -0.277 [-0.370, -0.187] |
| s2345 | ICP + Chamfer in mm (per coppia) | +0.079 [+0.014, +0.151] | +0.077 [+0.012, +0.151] | +0.079 [+0.014, +0.150] | +0.064 [-0.002, +0.139] | +0.103 [+0.039, +0.174] | +0.068 [+0.002, +0.142] | +0.063 [-0.003, +0.137] |
| s1234 | miglior concorrente della matrice | -0.262 [-0.349, -0.178] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) | -0.203 [-0.293, -0.116] (GNM (visto) vB, coefficienti) | -0.262 [-0.350, -0.176] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) | -0.261 [-0.349, -0.175] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) | -0.256 [-0.344, -0.169] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) | -0.271 [-0.359, -0.186] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) | -0.269 [-0.353, -0.187] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) |
| s2345 | miglior concorrente della matrice | -0.271 [-0.365, -0.181] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) | -0.219 [-0.322, -0.127] (GNM (visto) vB, coefficienti) | -0.274 [-0.369, -0.183] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) | -0.272 [-0.366, -0.182] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) | -0.269 [-0.364, -0.178] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) | -0.279 [-0.373, -0.188] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) | -0.277 [-0.370, -0.187] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) |

## facescape: Spearman (seme delle repliche 796786)

GT FR (grassetto = variante dichiarata):

| colonna | nocrop | stessa topologia senza crop | con original | con remesh | con down8k | con up60k | con noisy |
|---|---|---|---|---|---|---|---|
| factorized s1234, d_F cal. | 0.661 [0.586, 0.727] | 0.705 [0.636, 0.770] | 0.669 [0.596, 0.736] | 0.665 [0.592, 0.730] | 0.661 [0.585, 0.730] | 0.660 [0.588, 0.721] | 0.660 [0.585, 0.728] |
| factorized s1234, d_F non cal. | 0.687 [0.612, 0.758] | 0.724 [0.652, 0.794] | 0.695 [0.620, 0.768] | 0.697 [0.622, 0.767] | 0.684 [0.606, 0.758] | 0.695 [0.623, 0.762] | 0.687 [0.611, 0.759] |
| factorized s1234, d_P | 0.677 [0.596, 0.750] | 0.717 [0.640, 0.789] | 0.687 [0.609, 0.761] | 0.682 [0.604, 0.754] | 0.674 [0.592, 0.750] | 0.685 [0.608, 0.755] | 0.678 [0.596, 0.752] |
| factorized s2345, d_F cal. | 0.669 [0.597, 0.733] | 0.714 [0.644, 0.776] | 0.678 [0.605, 0.742] | 0.668 [0.596, 0.732] | 0.682 [0.611, 0.745] | 0.662 [0.595, 0.724] | 0.660 [0.586, 0.725] |
| factorized s2345, d_F non cal. | 0.698 [0.619, 0.770] | 0.723 [0.643, 0.797] | 0.702 [0.620, 0.776] | 0.711 [0.632, 0.783] | 0.704 [0.623, 0.779] | 0.709 [0.630, 0.779] | 0.680 [0.597, 0.753] |
| factorized s2345, d_P | 0.684 [0.601, 0.760] | 0.712 [0.627, 0.787] | 0.689 [0.607, 0.767] | 0.694 [0.612, 0.770] | 0.691 [0.607, 0.768] | 0.696 [0.611, 0.770] | 0.667 [0.579, 0.744] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | 0.675 [0.610, 0.732] | - | 0.662 [0.592, 0.719] | 0.683 [0.614, 0.741] | 0.734 [0.672, 0.792] | 0.693 [0.626, 0.754] | 0.700 [0.632, 0.757] |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.618 [0.534, 0.693] | - | 0.616 [0.532, 0.691] | 0.619 [0.536, 0.695] | 0.622 [0.539, 0.701] | 0.624 [0.543, 0.700] | 0.630 [0.544, 0.707] |
| ICP + Chamfer in mm (per coppia) | 0.453 [0.389, 0.514] | 0.611 [0.546, 0.673] | 0.451 [0.388, 0.510] | 0.548 [0.478, 0.613] | 0.395 [0.342, 0.448] | 0.524 [0.454, 0.590] | 0.486 [0.421, 0.546] |
| GNM (visto) vB, composizione S_B + k d_P | 0.800 [0.741, 0.847] | 0.807 [0.749, 0.855] | 0.800 [0.741, 0.847] | 0.801 [0.743, 0.849] | 0.794 [0.732, 0.842] | 0.803 [0.742, 0.852] | 0.801 [0.742, 0.850] |
| GNM (visto) vB, mesh d'identita' SR | 0.767 [0.675, 0.843] | 0.767 [0.677, 0.842] | 0.766 [0.675, 0.841] | 0.768 [0.677, 0.843] | 0.765 [0.674, 0.841] | 0.768 [0.678, 0.843] | 0.767 [0.674, 0.842] |
| interazione tardiva, Chamfer su tutti i vertici [factorized s1234] (non dichiarata), ricomposta con S e c' (FR, post hoc) | 0.501 [0.432, 0.565] | 0.574 [0.497, 0.644] | 0.526 [0.454, 0.593] | 0.521 [0.451, 0.587] | 0.527 [0.455, 0.594] | 0.511 [0.441, 0.576] | 0.534 [0.466, 0.594] |
| interazione tardiva, Chamfer sul sottocampione [factorized s1234] (non dichiarata), ricomposta con S e c' (FR, post hoc) | 0.475 [0.407, 0.537] | 0.581 [0.507, 0.652] | 0.502 [0.434, 0.567] | 0.483 [0.415, 0.548] | 0.504 [0.436, 0.569] | 0.466 [0.399, 0.527] | 0.498 [0.436, 0.556] |
| **interazione tardiva, OT entropico sul sottocampione [factorized s1234], ricomposta con S e c' (FR, post hoc)** | 0.495 [0.430, 0.559] | 0.569 [0.500, 0.638] | 0.512 [0.444, 0.578] | 0.489 [0.424, 0.553] | 0.516 [0.449, 0.581] | 0.477 [0.411, 0.539] | 0.515 [0.450, 0.579] |
| interazione tardiva, Chamfer su tutti i vertici [factorized s2345] (non dichiarata), ricomposta con S e c' (FR, post hoc) | 0.540 [0.471, 0.601] | 0.583 [0.512, 0.649] | 0.554 [0.483, 0.618] | 0.548 [0.481, 0.614] | 0.558 [0.488, 0.622] | 0.539 [0.473, 0.603] | 0.570 [0.503, 0.631] |
| interazione tardiva, Chamfer sul sottocampione [factorized s2345] (non dichiarata), ricomposta con S e c' (FR, post hoc) | 0.507 [0.441, 0.568] | 0.595 [0.525, 0.663] | 0.527 [0.460, 0.589] | 0.495 [0.430, 0.556] | 0.532 [0.464, 0.595] | 0.483 [0.418, 0.543] | 0.529 [0.464, 0.588] |
| **interazione tardiva, OT entropico sul sottocampione [factorized s2345], ricomposta con S e c' (FR, post hoc)** | 0.492 [0.431, 0.547] | 0.578 [0.511, 0.640] | 0.512 [0.448, 0.571] | 0.469 [0.409, 0.524] | 0.518 [0.454, 0.577] | 0.464 [0.406, 0.517] | 0.508 [0.446, 0.563] |

GT SR (grassetto = variante dichiarata):

| colonna | nocrop | stessa topologia senza crop | con original | con remesh | con down8k | con up60k | con noisy |
|---|---|---|---|---|---|---|---|
| factorized s1234, d_F cal. | 0.692 [0.623, 0.753] | 0.736 [0.672, 0.795] | 0.701 [0.632, 0.762] | 0.696 [0.626, 0.757] | 0.695 [0.625, 0.757] | 0.689 [0.621, 0.747] | 0.693 [0.623, 0.753] |
| factorized s1234, d_F non cal. | 0.745 [0.677, 0.801] | 0.784 [0.721, 0.838] | 0.753 [0.686, 0.809] | 0.753 [0.687, 0.808] | 0.743 [0.675, 0.802] | 0.750 [0.682, 0.805] | 0.747 [0.680, 0.804] |
| factorized s1234, d_P | 0.747 [0.677, 0.804] | 0.789 [0.724, 0.843] | 0.756 [0.687, 0.813] | 0.751 [0.683, 0.807] | 0.746 [0.676, 0.805] | 0.753 [0.682, 0.809] | 0.753 [0.682, 0.809] |
| factorized s2345, d_F cal. | 0.687 [0.620, 0.748] | 0.730 [0.664, 0.790] | 0.696 [0.629, 0.757] | 0.686 [0.620, 0.747] | 0.702 [0.634, 0.762] | 0.681 [0.614, 0.741] | 0.678 [0.610, 0.739] |
| factorized s2345, d_F non cal. | 0.751 [0.686, 0.809] | 0.778 [0.715, 0.833] | 0.755 [0.690, 0.813] | 0.763 [0.701, 0.819] | 0.758 [0.693, 0.816] | 0.760 [0.696, 0.819] | 0.734 [0.663, 0.793] |
| factorized s2345, d_P | 0.753 [0.687, 0.813] | 0.783 [0.720, 0.839] | 0.759 [0.693, 0.818] | 0.762 [0.698, 0.820] | 0.761 [0.694, 0.821] | 0.764 [0.700, 0.823] | 0.739 [0.668, 0.799] |
| NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | 0.701 [0.642, 0.753] | - | 0.688 [0.629, 0.741] | 0.712 [0.652, 0.764] | 0.761 [0.706, 0.809] | 0.721 [0.662, 0.774] | 0.724 [0.665, 0.777] |
| REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | 0.718 [0.654, 0.774] | - | 0.717 [0.650, 0.773] | 0.719 [0.652, 0.775] | 0.726 [0.662, 0.781] | 0.726 [0.663, 0.781] | 0.732 [0.667, 0.787] |
| ICP + Chamfer in mm (per coppia) | 0.477 [0.414, 0.533] | 0.639 [0.576, 0.698] | 0.473 [0.411, 0.529] | 0.578 [0.513, 0.638] | 0.415 [0.361, 0.464] | 0.554 [0.488, 0.614] | 0.509 [0.447, 0.566] |
| GNM (visto) vB, composizione S_B + k d_P | 0.783 [0.724, 0.839] | 0.789 [0.731, 0.845] | 0.785 [0.727, 0.839] | 0.783 [0.724, 0.838] | 0.780 [0.721, 0.836] | 0.788 [0.730, 0.843] | 0.783 [0.721, 0.840] |
| GNM (visto) vB, mesh d'identita' SR | 0.854 [0.803, 0.894] | 0.854 [0.805, 0.894] | 0.853 [0.803, 0.892] | 0.855 [0.806, 0.895] | 0.853 [0.802, 0.893] | 0.856 [0.805, 0.896] | 0.854 [0.803, 0.894] |
| interazione tardiva, Chamfer su tutti i vertici [factorized s1234] (non dichiarata), d_LI (SR) | 0.525 [0.451, 0.592] | 0.602 [0.528, 0.671] | 0.560 [0.484, 0.627] | 0.556 [0.481, 0.623] | 0.559 [0.484, 0.627] | 0.568 [0.493, 0.633] | 0.582 [0.511, 0.642] |
| interazione tardiva, Chamfer sul sottocampione [factorized s1234] (non dichiarata), d_LI (SR) | 0.518 [0.445, 0.586] | 0.661 [0.586, 0.730] | 0.560 [0.487, 0.628] | 0.557 [0.482, 0.626] | 0.560 [0.488, 0.629] | 0.538 [0.463, 0.603] | 0.565 [0.496, 0.627] |
| **interazione tardiva, OT entropico sul sottocampione [factorized s1234], d_LI (SR)** | 0.620 [0.546, 0.688] | 0.710 [0.642, 0.772] | 0.638 [0.567, 0.704] | 0.647 [0.575, 0.712] | 0.642 [0.571, 0.707] | 0.623 [0.551, 0.689] | 0.688 [0.619, 0.751] |
| interazione tardiva, Chamfer su tutti i vertici [factorized s2345] (non dichiarata), d_LI (SR) | 0.594 [0.519, 0.660] | 0.610 [0.536, 0.676] | 0.611 [0.536, 0.677] | 0.619 [0.545, 0.684] | 0.611 [0.536, 0.676] | 0.645 [0.572, 0.712] | 0.644 [0.573, 0.709] |
| interazione tardiva, Chamfer sul sottocampione [factorized s2345] (non dichiarata), d_LI (SR) | 0.610 [0.535, 0.676] | 0.692 [0.616, 0.758] | 0.628 [0.554, 0.696] | 0.638 [0.564, 0.706] | 0.630 [0.556, 0.697] | 0.619 [0.545, 0.685] | 0.650 [0.578, 0.713] |
| **interazione tardiva, OT entropico sul sottocampione [factorized s2345], d_LI (SR)** | 0.643 [0.571, 0.709] | 0.712 [0.640, 0.775] | 0.653 [0.582, 0.718] | 0.669 [0.598, 0.732] | 0.657 [0.588, 0.722] | 0.643 [0.573, 0.709] | 0.685 [0.615, 0.747] |

Delta appaiati della variante dichiarata (sinkhorn), lettura fr, GT FR:

| seme | contro | nocrop | stessa topologia senza crop | con original | con remesh | con down8k | con up60k | con noisy |
|---|---|---|---|---|---|---|---|---|
| s1234 | factorized s1234, d_F cal. | -0.165 [-0.209, -0.122] | -0.136 [-0.186, -0.091] | -0.157 [-0.202, -0.113] | -0.176 [-0.218, -0.134] | -0.145 [-0.192, -0.099] | -0.183 [-0.224, -0.143] | -0.145 [-0.192, -0.099] |
| s1234 | factorized s1234, d_F non cal. | -0.192 [-0.248, -0.138] | -0.155 [-0.220, -0.094] | -0.184 [-0.243, -0.128] | -0.208 [-0.264, -0.154] | -0.168 [-0.231, -0.110] | -0.219 [-0.276, -0.165] | -0.172 [-0.233, -0.112] |
| s1234 | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.180 [-0.247, -0.106] | - | -0.150 [-0.220, -0.074] | -0.194 [-0.263, -0.117] | -0.219 [-0.284, -0.147] | -0.216 [-0.284, -0.144] | -0.185 [-0.253, -0.111] |
| s1234 | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.122 [-0.205, -0.036] | - | -0.105 [-0.190, -0.017] | -0.130 [-0.211, -0.044] | -0.107 [-0.191, -0.019] | -0.147 [-0.229, -0.060] | -0.115 [-0.200, -0.027] |
| s1234 | ICP + Chamfer in mm (per coppia) | +0.042 [-0.017, +0.105] | -0.042 [-0.111, +0.032] | +0.061 [-0.000, +0.128] | -0.059 [-0.119, +0.008] | +0.120 [+0.064, +0.179] | -0.048 [-0.110, +0.018] | +0.029 [-0.033, +0.094] |
| s2345 | factorized s2345, d_F cal. | -0.177 [-0.218, -0.136] | -0.136 [-0.183, -0.088] | -0.165 [-0.208, -0.124] | -0.199 [-0.240, -0.161] | -0.163 [-0.207, -0.121] | -0.198 [-0.239, -0.157] | -0.152 [-0.196, -0.110] |
| s2345 | factorized s2345, d_F non cal. | -0.206 [-0.265, -0.145] | -0.145 [-0.211, -0.075] | -0.189 [-0.249, -0.127] | -0.242 [-0.298, -0.183] | -0.186 [-0.245, -0.123] | -0.244 [-0.304, -0.182] | -0.172 [-0.233, -0.107] |
| s2345 | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.183 [-0.244, -0.115] | - | -0.150 [-0.215, -0.075] | -0.213 [-0.275, -0.145] | -0.216 [-0.278, -0.150] | -0.229 [-0.292, -0.161] | -0.192 [-0.255, -0.120] |
| s2345 | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.125 [-0.209, -0.043] | - | -0.104 [-0.189, -0.021] | -0.149 [-0.232, -0.069] | -0.104 [-0.187, -0.022] | -0.160 [-0.242, -0.074] | -0.122 [-0.207, -0.037] |
| s2345 | ICP + Chamfer in mm (per coppia) | +0.039 [-0.017, +0.098] | -0.034 [-0.096, +0.040] | +0.061 [+0.001, +0.126] | -0.079 [-0.138, -0.020] | +0.123 [+0.070, +0.179] | -0.060 [-0.124, +0.003] | +0.022 [-0.036, +0.084] |
| s1234 | miglior concorrente della matrice | -0.304 [-0.361, -0.250] (GNM (visto) vB, composizione S_B + k d_P) | -0.238 [-0.299, -0.177] (GNM (visto) vB, composizione S_B + k d_P) | -0.288 [-0.347, -0.232] (GNM (visto) vB, composizione S_B + k d_P) | -0.312 [-0.370, -0.259] (GNM (visto) vB, composizione S_B + k d_P) | -0.278 [-0.338, -0.221] (GNM (visto) vB, composizione S_B + k d_P) | -0.326 [-0.384, -0.272] (GNM (visto) vB, composizione S_B + k d_P) | -0.286 [-0.344, -0.232] (GNM (visto) vB, composizione S_B + k d_P) |
| s2345 | miglior concorrente della matrice | -0.307 [-0.361, -0.259] (GNM (visto) vB, composizione S_B + k d_P) | -0.229 [-0.288, -0.171] (GNM (visto) vB, composizione S_B + k d_P) | -0.288 [-0.344, -0.235] (GNM (visto) vB, composizione S_B + k d_P) | -0.332 [-0.385, -0.285] (GNM (visto) vB, composizione S_B + k d_P) | -0.275 [-0.331, -0.222] (GNM (visto) vB, composizione S_B + k d_P) | -0.339 [-0.393, -0.291] (GNM (visto) vB, composizione S_B + k d_P) | -0.293 [-0.349, -0.242] (GNM (visto) vB, composizione S_B + k d_P) |

Delta appaiati della variante dichiarata (sinkhorn), lettura sr, GT SR:

| seme | contro | nocrop | stessa topologia senza crop | con original | con remesh | con down8k | con up60k | con noisy |
|---|---|---|---|---|---|---|---|---|
| s1234 | factorized s1234, d_P | -0.126 [-0.167, -0.086] | -0.079 [-0.123, -0.036] | -0.118 [-0.160, -0.077] | -0.103 [-0.145, -0.064] | -0.104 [-0.149, -0.063] | -0.130 [-0.170, -0.091] | -0.065 [-0.107, -0.021] |
| s1234 | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.081 [-0.142, -0.020] | - | -0.050 [-0.113, +0.015] | -0.064 [-0.128, -0.001] | -0.119 [-0.178, -0.061] | -0.099 [-0.161, -0.035] | -0.036 [-0.099, +0.025] |
| s1234 | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.098 [-0.168, -0.031] | - | -0.078 [-0.149, -0.009] | -0.072 [-0.143, -0.001] | -0.084 [-0.151, -0.016] | -0.103 [-0.172, -0.037] | -0.044 [-0.114, +0.029] |
| s1234 | ICP + Chamfer in mm (per coppia) | +0.143 [+0.087, +0.199] | +0.071 [+0.008, +0.134] | +0.165 [+0.106, +0.222] | +0.069 [+0.009, +0.127] | +0.227 [+0.176, +0.277] | +0.068 [+0.007, +0.127] | +0.179 [+0.120, +0.235] |
| s2345 | factorized s2345, d_P | -0.110 [-0.145, -0.074] | -0.071 [-0.106, -0.032] | -0.106 [-0.141, -0.069] | -0.093 [-0.127, -0.057] | -0.104 [-0.140, -0.067] | -0.121 [-0.155, -0.086] | -0.054 [-0.090, -0.016] |
| s2345 | NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi) | -0.058 [-0.119, +0.004] | - | -0.035 [-0.098, +0.029] | -0.043 [-0.105, +0.022] | -0.104 [-0.162, -0.045] | -0.078 [-0.139, -0.013] | -0.039 [-0.100, +0.024] |
| s2345 | REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi) | -0.075 [-0.144, -0.003] | - | -0.064 [-0.134, +0.008] | -0.050 [-0.120, +0.023] | -0.068 [-0.136, +0.002] | -0.083 [-0.152, -0.013] | -0.047 [-0.116, +0.026] |
| s2345 | ICP + Chamfer in mm (per coppia) | +0.166 [+0.107, +0.224] | +0.073 [+0.010, +0.136] | +0.180 [+0.118, +0.238] | +0.090 [+0.028, +0.151] | +0.243 [+0.190, +0.294] | +0.089 [+0.027, +0.151] | +0.176 [+0.115, +0.233] |
| s1234 | miglior concorrente della matrice | -0.233 [-0.290, -0.183] (GNM (visto) vB, mesh d'identita' SR) | -0.144 [-0.197, -0.094] (GNM (visto) vB, mesh d'identita' SR) | -0.214 [-0.270, -0.164] (GNM (visto) vB, mesh d'identita' SR) | -0.208 [-0.265, -0.156] (GNM (visto) vB, mesh d'identita' SR) | -0.211 [-0.266, -0.159] (GNM (visto) vB, mesh d'identita' SR) | -0.233 [-0.288, -0.185] (GNM (visto) vB, mesh d'identita' SR) | -0.166 [-0.217, -0.115] (GNM (visto) vB, mesh d'identita' SR) |
| s2345 | miglior concorrente della matrice | -0.210 [-0.265, -0.160] (GNM (visto) vB, mesh d'identita' SR) | -0.143 [-0.196, -0.091] (GNM (visto) vB, mesh d'identita' SR) | -0.200 [-0.256, -0.149] (GNM (visto) vB, mesh d'identita' SR) | -0.187 [-0.241, -0.134] (GNM (visto) vB, mesh d'identita' SR) | -0.195 [-0.249, -0.144] (GNM (visto) vB, mesh d'identita' SR) | -0.213 [-0.267, -0.164] (GNM (visto) vB, mesh d'identita' SR) | -0.169 [-0.223, -0.117] (GNM (visto) vB, mesh d'identita' SR) |

## Lettura della domanda (regola di REGOLE.md, sez. 6: variante dichiarata, nocrop, entrambi i semi)

| dominio | GT | contro il vettore (s1234; s2345) | contro il protocollo (s1234; s2345) | lettura |
|---|---|---|---|---|
| hifi3d | FR | -0.087 [-0.121, -0.053]; -0.079 [-0.113, -0.047] | -0.035 [-0.080, +0.017]; -0.045 [-0.096, +0.008] (NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)) | nessun recupero |
| hifi3d | SR | -0.169 [-0.241, -0.104]; -0.169 [-0.243, -0.102] | -0.262 [-0.349, -0.178]; -0.271 [-0.365, -0.181] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) | nessun recupero |
| facescape | FR | -0.165 [-0.209, -0.122]; -0.177 [-0.218, -0.136] | -0.180 [-0.247, -0.106]; -0.183 [-0.244, -0.115] (NoW metrical / MICA (Procrustes senza scala + ICP rigido), mm (media dei due versi)) | nessun recupero |
| facescape | SR | -0.126 [-0.167, -0.086]; -0.110 [-0.145, -0.074] | -0.098 [-0.168, -0.031]; -0.075 [-0.144, -0.003] (REALY (gICP con scala + rICP per regione + nICP), mm (media dei due versi)) | nessun recupero |

## Costo per confronto (stesso nodo; misurato)

Nodo `a768-l40s-04.srv.aau.dk`, job 1074616, GPU NVIDIA L40S; CPU AMD EPYC 9354 32-Core Processor a un thread per l'ICP e i vettori. Campione di `cc.py sample` (HIFI3D, 3 tipi di coppie, 150 coppie); mediana [p25, p75] fra le coppie della mediana di 3 giri.

| metodo | dispositivo | per coppia |
|---|---|---|
| interazione tardiva, chamfer | GPU, coppia isolata | 2.95 ms [1.49 ms, 10.80 ms] |
| interazione tardiva, chamfer_sub | GPU, coppia isolata | 239.58 us [232.34 us, 264.17 us] |
| interazione tardiva, sinkhorn **(dichiarata)** | GPU, coppia isolata | 43.58 ms [39.53 ms, 47.62 ms] |
| vettori [s, u] (d_F e d_P), a blocchi di tutte le coppie di N mesh | CPU 1 thread | N = 1000: 0.72 us, N = 10000: 0.76 us |
| vettori, una chiamata di `fact_paired.distances` per una coppia (rilegge il checkpoint) | CPU 1 thread | 1.76 ms |
| ICP + Chamfer in mm | CPU 1 thread | 41.65 ms [37.29 ms, 45.36 ms] |

Per mesh (una volta, GPU): forward coi token 44.04 ms, sottocampione FPS + Voronoi 86.62 ms. Memoria per mesh: token 9.3 MiB (mediana, float32) contro 1028 byte del vettore; sottocampione 0.98 MiB.

