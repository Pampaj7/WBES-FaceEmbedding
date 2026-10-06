## BFM held-out (100 soggetti standard)

Prima = protocollo attuale, dopo = equalize-support (righe con crop dalla vista equalizzata). Spearman con la GT del dominio (`/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz`), mesh-pair, clean; differenza dopo - prima con CI 95% bootstrap per soggetto appaiato (1000 repliche). `rif. senza crop`: Spearman della stessa metrica sulle 20 coppie senza crop (base), il livello a cui il gap si misura.

### latente BFM+ICT  (rif. senza crop 0.853)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.854 | 0.851 | -0.003 [-0.020, +0.016] | 100 | 9900 |
| crop<->noisy | 0.844 | 0.819 | -0.025 [-0.045, -0.003] | 100 | 9900 |
| crop<->remesh | 0.843 | 0.793 | -0.050 [-0.071, -0.028] | 100 | 9900 |
| crop<->down8k | 0.807 | 0.788 | -0.019 [-0.045, +0.009] | 100 | 9900 |
| crop<->up60k | 0.850 | 0.799 | -0.051 [-0.068, -0.034] | 100 | 9900 |
| crop (tutte) | 0.837 | 0.793 | -0.044 [-0.063, -0.027] | 100 | 49500 |
| senza crop (controllo) | 0.853 | 0.853 | +0.000 [+0.000, +0.000] | 100 | 99000 |
| tutte le cross | 0.847 | 0.830 | -0.017 [-0.024, -0.012] | 100 | 148500 |
| crop (tutte), 19 soggetti fuori dal training del congiunto | 0.794 | 0.706 | -0.088 [-0.178, -0.027] | 19 | 1710 |

### latente BFM-only  (rif. senza crop 0.803)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.779 | 0.777 | -0.002 [-0.027, +0.025] | 100 | 9900 |
| crop<->noisy | 0.762 | 0.720 | -0.042 [-0.080, -0.005] | 100 | 9900 |
| crop<->remesh | 0.742 | 0.702 | -0.040 [-0.063, -0.014] | 100 | 9900 |
| crop<->down8k | 0.716 | 0.741 | +0.025 [-0.011, +0.060] | 100 | 9900 |
| crop<->up60k | 0.770 | 0.755 | -0.015 [-0.037, +0.009] | 100 | 9900 |
| crop (tutte) | 0.753 | 0.736 | -0.017 [-0.041, +0.008] | 100 | 49500 |
| senza crop (controllo) | 0.803 | 0.803 | +0.000 [+0.000, +0.000] | 100 | 99000 |
| tutte le cross | 0.779 | 0.778 | -0.001 [-0.010, +0.009] | 100 | 148500 |

### Chamfer eval (repo)  (rif. senza crop 0.472)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.029 | 0.553 | +0.525 [+0.431, +0.620] | 100 | 9900 |
| crop<->noisy | 0.018 | 0.547 | +0.530 [+0.437, +0.629] | 100 | 9900 |
| crop<->remesh | 0.046 | 0.377 | +0.331 [+0.258, +0.396] | 100 | 9900 |
| crop<->down8k | -0.029 | 0.396 | +0.425 [+0.336, +0.507] | 100 | 9900 |
| crop<->up60k | -0.007 | 0.495 | +0.502 [+0.410, +0.584] | 100 | 9900 |
| crop (tutte) | 0.012 | 0.454 | +0.442 [+0.359, +0.518] | 100 | 49500 |
| senza crop (controllo) | 0.472 | 0.472 | +0.000 [+0.000, +0.000] | 100 | 99000 |
| tutte le cross | 0.237 | 0.445 | +0.208 [+0.177, +0.235] | 100 | 148500 |

### Chamfer (faceBench, 4096 pt)  (rif. senza crop 0.469)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.023 | 0.549 | +0.527 [+0.430, +0.621] | 100 | 9900 |
| crop<->noisy | 0.010 | 0.542 | +0.532 [+0.438, +0.625] | 100 | 9900 |
| crop<->remesh | 0.015 | 0.339 | +0.324 [+0.248, +0.385] | 100 | 9900 |
| crop<->down8k | -0.043 | 0.374 | +0.417 [+0.328, +0.498] | 100 | 9900 |
| crop<->up60k | -0.028 | 0.468 | +0.496 [+0.405, +0.581] | 100 | 9900 |
| crop (tutte) | -0.006 | 0.441 | +0.447 [+0.361, +0.520] | 100 | 49500 |

### Rigid ICP + Chamfer  (rif. senza crop 0.415)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.051 | 0.471 | +0.421 [+0.330, +0.514] | 100 | 9900 |
| crop<->noisy | 0.046 | 0.460 | +0.414 [+0.327, +0.500] | 100 | 9900 |
| crop<->remesh | 0.159 | 0.478 | +0.320 [+0.251, +0.391] | 100 | 9900 |
| crop<->down8k | -0.016 | 0.449 | +0.465 [+0.372, +0.543] | 100 | 9900 |
| crop<->up60k | 0.041 | 0.492 | +0.451 [+0.366, +0.530] | 100 | 9900 |
| crop (tutte) | 0.052 | 0.469 | +0.417 [+0.330, +0.496] | 100 | 49500 |

### Rigid ICP + NICP + P2P  (rif. senza crop 0.388)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.183 | 0.406 | +0.223 [+0.172, +0.274] | 100 | 9900 |
| crop<->noisy | 0.155 | 0.395 | +0.241 [+0.186, +0.297] | 100 | 9900 |
| crop<->remesh | 0.195 | 0.400 | +0.204 [+0.163, +0.245] | 100 | 9900 |
| crop<->down8k | 0.127 | 0.376 | +0.249 [+0.197, +0.298] | 100 | 9900 |
| crop<->up60k | 0.161 | 0.402 | +0.241 [+0.191, +0.296] | 100 | 9900 |
| crop (tutte) | 0.163 | 0.394 | +0.231 [+0.179, +0.281] | 100 | 49500 |

### Rigid ICP + NICP + P2Tri  (rif. senza crop 0.395)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.185 | 0.416 | +0.231 [+0.177, +0.285] | 100 | 9900 |
| crop<->noisy | 0.155 | 0.400 | +0.246 [+0.194, +0.297] | 100 | 9900 |
| crop<->remesh | 0.201 | 0.410 | +0.209 [+0.163, +0.250] | 100 | 9900 |
| crop<->down8k | 0.127 | 0.386 | +0.259 [+0.209, +0.309] | 100 | 9900 |
| crop<->up60k | 0.162 | 0.411 | +0.249 [+0.194, +0.298] | 100 | 9900 |
| crop (tutte) | 0.165 | 0.403 | +0.238 [+0.190, +0.288] | 100 | 49500 |

### Controlli

Righe senza crop del protocollo contro la base (devono coincidere, differenza 0):

| who | n_rows | n_crop_rows | n_nocrop_rows | nocrop_rows_match_base | nocrop_max_abs_diff | crop_gt_max_abs_diff |
| --- | --- | --- | --- | --- | --- | --- |
| joint | 148500 | 49500 | 99000 | True | 0.000000 | 0.000000 |
| bfm_only | 148500 | 49500 | 99000 | True | 0.000000 | 0.000000 |

Nella variante, (crop_a, original_b) contro (original_a, crop_b): la original equalizzata e' il crop, quindi le due righe sono la stessa coppia di crop:

| who | n_pairs | latent_distance_max_abs_diff | latent_distance_max_rel_diff | raw_chamfer_max_abs_diff | raw_chamfer_max_rel_diff |
| --- | --- | --- | --- | --- | --- |
| joint_eqsupport | 4950 | 0.01568500 | 0.02708510 | 0.00004100 | 0.00786276 |
| bfm_only_eqsupport | 4950 | 0.02105400 | 0.05491143 | 0.00004100 | 0.00786276 |

Chamfer eval nei bracci e nei frame contro il riferimento `joint` (deve coincidere: non dipende dal modello ne' da una rotazione):

| arm | colonna | max_rel_diff | n_rows_rel_gt_1e-4 | n_rows |
| --- | --- | --- | --- | --- |
| joint | raw_chamfer_before | 0.00000000 | 0 | 148500 |
| joint | raw_chamfer_after | 0.00000000 | 0 | 148500 |
| bfm_only | raw_chamfer_before | 0.00000000 | 0 | 148500 |
| bfm_only | raw_chamfer_after | 0.00000000 | 0 | 148500 |

Vista equalizzata `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/BFM_ZS/eqsupport_view`: original equalizzata identica al crop bit per bit in 0/500 soggetti; regione letta dal crop diversa dalle facce del crop in 474 soggetti (BFM: vertici di bordo spostati da open3d, vedi eqsupport_view.py). Distanze in frazione della diagonale della original; `pavimento` = la stessa distanza fra la topologia intera e la original (rumore, smoothing), sotto la quale la regione non si puo' distinguere:

| topology | soggetti | facce_tenute | area_su_crop_mediana | area_su_crop_min | area_su_crop_max | eq_su_crop_p99 | crop_su_eq_p99 | crop_su_eq_max | pavimento_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| down8k | 500 | 0.88347 | 0.99970 | 0.99833 | 1.00139 | 0.00014 | 0.00014 | 0.00820 | 0.00013 |
| noisy | 500 | 0.89468 | 2.35461 | 2.22744 | 2.47935 | 0.00584 | 0.00337 | 0.00880 | 0.00585 |
| original | 500 | 0.89468 | 0.99934 | 0.99845 | 1.00000 | 0.00000 | 0.00000 | 0.00760 | 0.00000 |
| remesh | 500 | 0.90783 | 0.96903 | 0.96345 | 0.97344 | 0.00210 | 0.00383 | 0.03288 | 0.00195 |
| up60k | 500 | 0.89279 | 0.99247 | 0.99026 | 0.99657 | 0.00041 | 0.00050 | 0.01226 | 0.00038 |

Sorgenti:

| arm | prima | dopo |
| --- | --- | --- |
| joint | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_bfm/data_97cac8f5a2/joint_topology/zs_zeroshot | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_bfm/data_97cac8f5a2/joint_eqsupport_topology/zs_zeroshot |
| bfm_only | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_bfm/data_97cac8f5a2/bfm_only_topology/zs_zeroshot | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_bfm/data_97cac8f5a2/bfm_only_eqsupport_topology/zs_zeroshot |

