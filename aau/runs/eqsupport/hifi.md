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

