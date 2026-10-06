# Equalize support: righe con crop prima e dopo, per metrica e dominio

Generato da aau/zs3dmm/eqsupport_summarize.sbatch (job 1057191, 2026-10-06 17:27).
Regola di equalizzazione: aau/zs3dmm/eqsupport_view.py. Modelli: congiunto 1019532, BFM-only 1019310.

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


## ICT held-out (controllo della pipeline)

Prima = protocollo attuale, dopo = equalize-support (righe con crop dalla vista equalizzata). Spearman con la GT del dominio (`/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/ICT_ZS/eval_view/gt_matrix.npz`), mesh-pair, clean; differenza dopo - prima con CI 95% bootstrap per soggetto appaiato (1000 repliche). `rif. senza crop`: Spearman della stessa metrica sulle 20 coppie senza crop (base), il livello a cui il gap si misura.

### latente BFM+ICT  (rif. senza crop 0.991)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.988 | 0.983 | -0.005 [-0.008, -0.003] | 100 | 9900 |
| crop<->noisy | 0.986 | 0.972 | -0.013 [-0.019, -0.010] | 100 | 9900 |
| crop<->remesh | 0.987 | 0.951 | -0.036 [-0.046, -0.026] | 100 | 9900 |
| crop<->down8k | 0.984 | 0.889 | -0.095 [-0.122, -0.073] | 100 | 9900 |
| crop<->up60k | 0.984 | 0.967 | -0.018 [-0.024, -0.013] | 100 | 9900 |
| crop (tutte) | 0.986 | 0.943 | -0.043 [-0.054, -0.032] | 100 | 49500 |
| senza crop (controllo) | 0.991 | 0.991 | +0.000 [+0.000, +0.000] | 100 | 99000 |
| tutte le cross | 0.989 | 0.974 | -0.015 [-0.021, -0.012] | 100 | 148500 |
| crop (tutte), 20 soggetti fuori dal training del congiunto | 0.978 | 0.924 | -0.054 [-0.117, -0.020] | 20 | 1900 |

### latente BFM-only  (rif. senza crop 0.259)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.671 | 0.820 | +0.149 [+0.108, +0.192] | 100 | 9900 |
| crop<->noisy | 0.640 | 0.815 | +0.175 [+0.129, +0.226] | 100 | 9900 |
| crop<->remesh | 0.624 | 0.547 | -0.077 [-0.118, -0.031] | 100 | 9900 |
| crop<->down8k | 0.209 | 0.145 | -0.064 [-0.103, -0.022] | 100 | 9900 |
| crop<->up60k | 0.652 | 0.682 | +0.029 [-0.015, +0.074] | 100 | 9900 |
| crop (tutte) | 0.379 | 0.397 | +0.018 [+0.000, +0.036] | 100 | 49500 |
| senza crop (controllo) | 0.259 | 0.259 | +0.000 [+0.000, +0.000] | 100 | 99000 |
| tutte le cross | 0.294 | 0.296 | +0.002 [-0.005, +0.009] | 100 | 148500 |

### Chamfer eval (repo)  (rif. senza crop 0.443)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.492 | 0.923 | +0.431 [+0.362, +0.498] | 100 | 9900 |
| crop<->noisy | 0.447 | 0.914 | +0.467 [+0.404, +0.535] | 100 | 9900 |
| crop<->remesh | 0.800 | 0.831 | +0.031 [+0.003, +0.062] | 100 | 9900 |
| crop<->down8k | 0.565 | 0.365 | -0.200 [-0.240, -0.152] | 100 | 9900 |
| crop<->up60k | 0.719 | 0.873 | +0.154 [+0.120, +0.196] | 100 | 9900 |
| crop (tutte) | 0.472 | 0.668 | +0.195 [+0.175, +0.218] | 100 | 49500 |
| senza crop (controllo) | 0.443 | 0.443 | +0.000 [+0.000, +0.000] | 100 | 99000 |
| tutte le cross | 0.442 | 0.511 | +0.068 [+0.060, +0.078] | 100 | 148500 |

### Chamfer (faceBench, 4096 pt)  (rif. senza crop n/d)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.514 | 0.939 | +0.426 [+0.356, +0.506] | 100 | 9900 |
| crop<->noisy | 0.468 | 0.932 | +0.465 [+0.394, +0.536] | 100 | 9900 |
| crop<->remesh | 0.789 | 0.808 | +0.020 [-0.012, +0.056] | 100 | 9900 |
| crop<->down8k | 0.472 | 0.323 | -0.149 [-0.194, -0.090] | 100 | 9900 |
| crop<->up60k | 0.722 | 0.858 | +0.136 [+0.098, +0.186] | 100 | 9900 |
| crop (tutte) | 0.464 | 0.659 | +0.195 [+0.173, +0.221] | 100 | 49500 |

### Rigid ICP + Chamfer  (rif. senza crop n/d)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.433 | 0.823 | +0.390 [+0.347, +0.439] | 100 | 9900 |
| crop<->noisy | 0.400 | 0.811 | +0.411 [+0.365, +0.459] | 100 | 9900 |
| crop<->remesh | 0.572 | 0.797 | +0.224 [+0.182, +0.266] | 100 | 9900 |
| crop<->down8k | 0.573 | 0.538 | -0.035 [-0.064, -0.003] | 100 | 9900 |
| crop<->up60k | 0.464 | 0.793 | +0.329 [+0.283, +0.373] | 100 | 9900 |
| crop (tutte) | 0.488 | 0.734 | +0.246 [+0.214, +0.278] | 100 | 49500 |

### Rigid ICP + NICP + P2P  (rif. senza crop n/d)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.336 | 0.694 | +0.358 [+0.310, +0.406] | 100 | 9900 |
| crop<->noisy | 0.282 | 0.680 | +0.398 [+0.352, +0.442] | 100 | 9900 |
| crop<->remesh | 0.352 | 0.657 | +0.305 [+0.265, +0.346] | 100 | 9900 |
| crop<->down8k | 0.359 | 0.553 | +0.194 [+0.146, +0.233] | 100 | 9900 |
| crop<->up60k | 0.291 | 0.642 | +0.351 [+0.296, +0.400] | 100 | 8628 |
| crop (tutte) | 0.328 | 0.600 | +0.271 [+0.231, +0.308] | 100 | 48228 |

### Rigid ICP + NICP + P2Tri  (rif. senza crop n/d)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.336 | 0.699 | +0.363 [+0.316, +0.408] | 100 | 9900 |
| crop<->noisy | 0.279 | 0.679 | +0.400 [+0.352, +0.446] | 100 | 9900 |
| crop<->remesh | 0.359 | 0.666 | +0.307 [+0.267, +0.345] | 100 | 9900 |
| crop<->down8k | 0.381 | 0.554 | +0.174 [+0.131, +0.212] | 100 | 9900 |
| crop<->up60k | 0.292 | 0.647 | +0.355 [+0.302, +0.404] | 100 | 8628 |
| crop (tutte) | 0.334 | 0.614 | +0.280 [+0.243, +0.317] | 100 | 48228 |

### Controlli

Righe senza crop del protocollo contro la base (devono coincidere, differenza 0):

| who | n_rows | n_crop_rows | n_nocrop_rows | nocrop_rows_match_base | nocrop_max_abs_diff | crop_gt_max_abs_diff |
| --- | --- | --- | --- | --- | --- | --- |
| joint | 148500 | 49500 | 99000 | True | 0.000000 | 0.000000 |
| bfm_only | 148500 | 49500 | 99000 | True | 0.000000 | 0.000000 |

Nella variante, (crop_a, original_b) contro (original_a, crop_b): la original equalizzata e' il crop, quindi le due righe sono la stessa coppia di crop:

| who | n_pairs | latent_distance_max_abs_diff | latent_distance_max_rel_diff | raw_chamfer_max_abs_diff | raw_chamfer_max_rel_diff |
| --- | --- | --- | --- | --- | --- |
| joint_eqsupport | 4950 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| bfm_only_eqsupport | 4950 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |

Chamfer eval nei bracci e nei frame contro il riferimento `joint` (deve coincidere: non dipende dal modello ne' da una rotazione):

| arm | colonna | max_rel_diff | n_rows_rel_gt_1e-4 | n_rows |
| --- | --- | --- | --- | --- |
| joint | raw_chamfer_before | 0.00000000 | 0 | 148500 |
| joint | raw_chamfer_after | 0.00000000 | 0 | 148500 |
| bfm_only | raw_chamfer_before | 0.00000000 | 0 | 148500 |
| bfm_only | raw_chamfer_after | 0.00000000 | 0 | 148500 |

Vista equalizzata `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/ICT_ZS/eqsupport_view`: original equalizzata identica al crop bit per bit in 500/500 soggetti; regione letta dal crop diversa dalle facce del crop in 0 soggetti (BFM: vertici di bordo spostati da open3d, vedi eqsupport_view.py). Distanze in frazione della diagonale della original; `pavimento` = la stessa distanza fra la topologia intera e la original (rumore, smoothing), sotto la quale la regione non si puo' distinguere:

| topology | soggetti | facce_tenute | area_su_crop_mediana | area_su_crop_min | area_su_crop_max | eq_su_crop_p99 | crop_su_eq_p99 | crop_su_eq_max | pavimento_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| down8k | 500 | 0.83899 | 1.00052 | 0.99623 | 1.00529 | 0.00038 | 0.00037 | 0.03155 | 0.00032 |
| noisy | 500 | 0.91817 | 1.47178 | 1.39018 | 1.53975 | 0.00574 | 0.00438 | 0.00932 | 0.00573 |
| original | 500 | 0.91817 | 1.00000 | 1.00000 | 1.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| remesh | 500 | 0.89002 | 0.97018 | 0.96154 | 0.97703 | 0.00243 | 0.00292 | 0.02291 | 0.00267 |
| up60k | 500 | 0.88444 | 1.00000 | 0.99993 | 1.00002 | 0.00000 | 0.00000 | 0.00110 | 0.00000 |

Sorgenti:

| arm | prima | dopo |
| --- | --- | --- |
| joint | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_ictzs/data_b00b50597f/joint/zs_zeroshot | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_ictzs/data_b00b50597f/joint_eqsupport_topology/zs_zeroshot |
| bfm_only | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_ictzs/data_b00b50597f/bfm_only/zs_zeroshot | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_ictzs/data_b00b50597f/bfm_only_eqsupport_topology/zs_zeroshot |


## HIFI3D

Prima = protocollo attuale, dopo = equalize-support (righe con crop dalla vista equalizzata). Spearman con la GT del dominio (`/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/HIFI3D/eval_view/gt_matrix.npz`), mesh-pair, clean; differenza dopo - prima con CI 95% bootstrap per soggetto appaiato (1000 repliche). `rif. senza crop`: Spearman della stessa metrica sulle 20 coppie senza crop (base), il livello a cui il gap si misura.

### latente BFM+ICT, nativo  (rif. senza crop 0.428)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.113 | 0.586 | +0.473 [+0.396, +0.547] | 100 | 9900 |
| crop<->noisy | 0.081 | 0.545 | +0.463 [+0.389, +0.527] | 100 | 9900 |
| crop<->remesh | 0.038 | 0.413 | +0.375 [+0.312, +0.439] | 100 | 9900 |
| crop<->down8k | 0.008 | 0.204 | +0.196 [+0.146, +0.250] | 100 | 9900 |
| crop<->up60k | 0.037 | 0.436 | +0.399 [+0.327, +0.466] | 100 | 9900 |
| crop (tutte) | 0.049 | 0.392 | +0.342 [+0.284, +0.395] | 100 | 49500 |
| senza crop (controllo) | 0.428 | 0.428 | +0.000 [+0.000, +0.000] | 100 | 99000 |
| tutte le cross | 0.246 | 0.413 | +0.167 [+0.143, +0.192] | 100 | 148500 |

### latente BFM-only, nativo  (rif. senza crop 0.206)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.369 | 0.521 | +0.152 [+0.101, +0.207] | 100 | 9900 |
| crop<->noisy | 0.391 | 0.497 | +0.106 [+0.059, +0.156] | 100 | 9900 |
| crop<->remesh | 0.145 | 0.267 | +0.122 [+0.087, +0.153] | 100 | 9900 |
| crop<->down8k | 0.053 | 0.128 | +0.076 [+0.044, +0.108] | 100 | 9900 |
| crop<->up60k | 0.173 | 0.295 | +0.122 [+0.082, +0.162] | 100 | 9900 |
| crop (tutte) | 0.135 | 0.233 | +0.098 [+0.074, +0.120] | 100 | 49500 |
| senza crop (controllo) | 0.206 | 0.206 | +0.000 [+0.000, +0.000] | 100 | 99000 |
| tutte le cross | 0.180 | 0.208 | +0.029 [+0.021, +0.036] | 100 | 148500 |

### latente BFM+ICT, frame BFM (Rx180 + facce invertite)  (rif. senza crop 0.245)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.207 | 0.531 | +0.324 [+0.264, +0.383] | 100 | 9900 |
| crop<->noisy | 0.220 | 0.503 | +0.282 [+0.223, +0.339] | 100 | 9900 |
| crop<->remesh | 0.163 | 0.375 | +0.213 [+0.161, +0.261] | 100 | 9900 |
| crop<->down8k | 0.091 | 0.192 | +0.101 [+0.056, +0.144] | 100 | 9900 |
| crop<->up60k | 0.210 | 0.354 | +0.145 [+0.094, +0.192] | 100 | 9900 |
| crop (tutte) | 0.143 | 0.279 | +0.136 [+0.107, +0.166] | 100 | 49500 |
| senza crop (controllo) | 0.245 | 0.245 | +0.000 [+0.000, +0.000] | 100 | 99000 |
| tutte le cross | 0.197 | 0.250 | +0.053 [+0.041, +0.065] | 100 | 148500 |

### latente BFM-only, frame BFM (Rx180 + facce invertite)  (rif. senza crop 0.320)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.132 | 0.466 | +0.334 [+0.262, +0.405] | 100 | 9900 |
| crop<->noisy | 0.116 | 0.454 | +0.338 [+0.272, +0.407] | 100 | 9900 |
| crop<->remesh | 0.138 | 0.416 | +0.278 [+0.221, +0.335] | 100 | 9900 |
| crop<->down8k | 0.150 | 0.281 | +0.131 [+0.084, +0.180] | 100 | 9900 |
| crop<->up60k | 0.180 | 0.357 | +0.177 [+0.125, +0.229] | 100 | 9900 |
| crop (tutte) | 0.136 | 0.361 | +0.226 [+0.174, +0.277] | 100 | 49500 |
| senza crop (controllo) | 0.320 | 0.320 | +0.000 [+0.000, +0.000] | 100 | 99000 |
| tutte le cross | 0.217 | 0.328 | +0.112 [+0.091, +0.131] | 100 | 148500 |

### Chamfer eval (repo)  (rif. senza crop 0.372)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.648 | 0.643 | -0.005 [-0.043, +0.039] | 100 | 9900 |
| crop<->noisy | 0.653 | 0.614 | -0.039 [-0.075, -0.000] | 100 | 9900 |
| crop<->remesh | 0.296 | 0.484 | +0.188 [+0.129, +0.249] | 100 | 9900 |
| crop<->down8k | 0.068 | 0.226 | +0.159 [+0.107, +0.206] | 100 | 9900 |
| crop<->up60k | 0.315 | 0.460 | +0.145 [+0.088, +0.201] | 100 | 9900 |
| crop (tutte) | 0.267 | 0.410 | +0.144 [+0.102, +0.180] | 100 | 49500 |
| senza crop (controllo) | 0.372 | 0.372 | +0.000 [+0.000, +0.000] | 100 | 99000 |
| tutte le cross | 0.336 | 0.368 | +0.032 [+0.018, +0.047] | 100 | 148500 |

### Chamfer (faceBench, 4096 pt)  (rif. senza crop 0.325)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.644 | 0.665 | +0.022 [-0.018, +0.070] | 100 | 9900 |
| crop<->noisy | 0.659 | 0.640 | -0.019 [-0.055, +0.019] | 100 | 9900 |
| crop<->remesh | 0.254 | 0.474 | +0.220 [+0.163, +0.271] | 100 | 9900 |
| crop<->down8k | 0.022 | 0.189 | +0.167 [+0.118, +0.217] | 100 | 9900 |
| crop<->up60k | 0.235 | 0.436 | +0.201 [+0.144, +0.252] | 100 | 9900 |
| crop (tutte) | 0.223 | 0.389 | +0.166 [+0.128, +0.200] | 100 | 49500 |

### Rigid ICP + Chamfer  (rif. senza crop 0.355)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.175 | 0.387 | +0.213 [+0.161, +0.266] | 100 | 9900 |
| crop<->noisy | 0.175 | 0.361 | +0.186 [+0.143, +0.230] | 100 | 9900 |
| crop<->remesh | 0.177 | 0.370 | +0.193 [+0.144, +0.244] | 100 | 9900 |
| crop<->down8k | 0.132 | 0.341 | +0.209 [+0.149, +0.267] | 100 | 9900 |
| crop<->up60k | 0.175 | 0.357 | +0.182 [+0.132, +0.233] | 100 | 9900 |
| crop (tutte) | 0.155 | 0.349 | +0.194 [+0.149, +0.240] | 100 | 49500 |

### Rigid ICP + NICP + P2P  (rif. senza crop 0.355)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.265 | 0.386 | +0.121 [+0.080, +0.163] | 100 | 9900 |
| crop<->noisy | 0.265 | 0.361 | +0.096 [+0.063, +0.132] | 100 | 9900 |
| crop<->remesh | 0.282 | 0.373 | +0.090 [+0.056, +0.127] | 100 | 9900 |
| crop<->down8k | 0.255 | 0.342 | +0.088 [+0.042, +0.127] | 100 | 9900 |
| crop<->up60k | 0.277 | 0.355 | +0.078 [+0.039, +0.116] | 100 | 9468 |
| crop (tutte) | 0.254 | 0.348 | +0.094 [+0.060, +0.128] | 100 | 49068 |

### Rigid ICP + NICP + P2Tri  (rif. senza crop 0.389)

| gruppo | prima | dopo | dopo - prima [CI 95%] | n soggetti | n coppie |
| --- | --- | --- | --- | --- | --- |
| crop<->original | 0.278 | 0.403 | +0.125 [+0.086, +0.171] | 100 | 9900 |
| crop<->noisy | 0.279 | 0.372 | +0.093 [+0.060, +0.127] | 100 | 9900 |
| crop<->remesh | 0.296 | 0.390 | +0.094 [+0.057, +0.132] | 100 | 9900 |
| crop<->down8k | 0.271 | 0.382 | +0.110 [+0.068, +0.151] | 100 | 9900 |
| crop<->up60k | 0.289 | 0.368 | +0.079 [+0.040, +0.118] | 100 | 9468 |
| crop (tutte) | 0.271 | 0.371 | +0.100 [+0.066, +0.137] | 100 | 49068 |

### Controlli

Righe senza crop del protocollo contro la base (devono coincidere, differenza 0):

| who | n_rows | n_crop_rows | n_nocrop_rows | nocrop_rows_match_base | nocrop_max_abs_diff | crop_gt_max_abs_diff |
| --- | --- | --- | --- | --- | --- | --- |
| joint | 148500 | 49500 | 99000 | True | 0.000000 | 0.000000 |
| bfm_only | 148500 | 49500 | 99000 | True | 0.000000 | 0.000000 |
| joint_frame-xmymz_flip | 148500 | 49500 | 99000 | True | 0.000000 | 0.000000 |
| bfm_only_frame-xmymz_flip | 148500 | 49500 | 99000 | True | 0.000000 | 0.000000 |

Nella variante, (crop_a, original_b) contro (original_a, crop_b): la original equalizzata e' il crop, quindi le due righe sono la stessa coppia di crop:

| who | n_pairs | latent_distance_max_abs_diff | latent_distance_max_rel_diff | raw_chamfer_max_abs_diff | raw_chamfer_max_rel_diff |
| --- | --- | --- | --- | --- | --- |
| joint_eqsupport | 4950 | 0.00000000 | 0.00000000 | 0.00000100 | 0.00019716 |
| bfm_only_eqsupport | 4950 | 0.00000000 | 0.00000000 | 0.00000100 | 0.00019716 |
| joint_frame-xmymz_flip_eqsupport | 4950 | 0.00000000 | 0.00000000 | 0.00000100 | 0.00019716 |
| bfm_only_frame-xmymz_flip_eqsupport | 4950 | 0.00000000 | 0.00000000 | 0.00000100 | 0.00019716 |

Chamfer eval nei bracci e nei frame contro il riferimento `joint` (deve coincidere: non dipende dal modello ne' da una rotazione):

| arm | colonna | max_rel_diff | n_rows_rel_gt_1e-4 | n_rows |
| --- | --- | --- | --- | --- |
| joint | raw_chamfer_before | 0.00000000 | 0 | 148500 |
| joint | raw_chamfer_after | 0.00000000 | 0 | 148500 |
| bfm_only | raw_chamfer_before | 0.00000000 | 0 | 148500 |
| bfm_only | raw_chamfer_after | 0.00000000 | 0 | 148500 |
| joint_frame-xmymz_flip | raw_chamfer_before | 0.03270677 | 165 | 148500 |
| joint_frame-xmymz_flip | raw_chamfer_after | 0.03270677 | 76 | 148500 |
| bfm_only_frame-xmymz_flip | raw_chamfer_before | 0.03270677 | 165 | 148500 |
| bfm_only_frame-xmymz_flip | raw_chamfer_after | 0.03270677 | 76 | 148500 |

Vista equalizzata `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/HIFI3D/eqsupport_view`: original equalizzata identica al crop bit per bit in 500/500 soggetti; regione letta dal crop diversa dalle facce del crop in 0 soggetti (BFM: vertici di bordo spostati da open3d, vedi eqsupport_view.py). Distanze in frazione della diagonale della original; `pavimento` = la stessa distanza fra la topologia intera e la original (rumore, smoothing), sotto la quale la regione non si puo' distinguere:

| topology | soggetti | facce_tenute | area_su_crop_mediana | area_su_crop_min | area_su_crop_max | eq_su_crop_p99 | crop_su_eq_p99 | crop_su_eq_max | pavimento_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| down8k | 500 | 0.89731 | 0.99965 | 0.99551 | 1.00445 | 0.00025 | 0.00025 | 0.02082 | 0.00022 |
| noisy | 500 | 0.95092 | 1.47069 | 1.41259 | 1.53666 | 0.00577 | 0.00454 | 0.01050 | 0.00576 |
| original | 500 | 0.95092 | 1.00000 | 1.00000 | 1.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| remesh | 500 | 0.93371 | 0.97984 | 0.97402 | 0.98338 | 0.00253 | 0.00276 | 0.01360 | 0.00278 |
| up60k | 500 | 0.92305 | 1.00000 | 0.99995 | 1.00001 | 0.00000 | 0.00000 | 0.00096 | 0.00000 |

Sorgenti:

| arm | prima | dopo |
| --- | --- | --- |
| joint | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/joint/zs_zeroshot | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/joint_eqsupport_topology/zs_zeroshot |
| bfm_only | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/bfm_only/zs_zeroshot | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/bfm_only_eqsupport_topology/zs_zeroshot |
| joint_frame-xmymz_flip | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/joint_frame-xmymz_flip/zs_zeroshot | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/joint_frame-xmymz_flip_eqsupport_topology/zs_zeroshot |
| bfm_only_frame-xmymz_flip | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/bfm_only_frame-xmymz_flip/zs_zeroshot | /home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/ws_hifi3d/data_328f2bfc1a/bfm_only_frame-xmymz_flip_eqsupport_topology/zs_zeroshot |


