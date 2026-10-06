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

