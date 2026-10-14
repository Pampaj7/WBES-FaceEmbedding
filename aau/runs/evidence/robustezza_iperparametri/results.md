# Robustezza agli iperparametri: allineamento per coppia contro la variabilita' da seme del nostro

Generato da `aau/robustezza_iperparametri/ri.py` (regole in `REGOLE.md`, committate prima dei numeri; definizioni nel docstring). Lettura DESCRITTIVA. HIFI3D e FaceScape: Spearman con GT FR sulle righe senza crop del sottocampione del rumore landmark (8%, seme 20261013), una maschera per vista; FaMoS: rank-1 e mAP sulle 150 scansioni di `esterna_famos_id`. 1000 repliche per soggetto, IC 95% percentile.

## Griglie

| metodo | configurazione | manopola | fonte |
|---|---|---|---|
| ICP + Chamfer in mm | default | - | matrice: blmm.icp modo mm (centro, registration_icp punto-punto, soglia 1000 = infinita, criteri di default di Open3D: 30 iterazioni, 1e-6, 1e-6) |
| ICP + Chamfer in mm | it10 | iterazioni | PCL, Registration: max_iterations_ = 10 (default) |
| ICP + Chamfer in mm | it2000 | iterazioni | tutorial ICP di Open3D 0.18: ICPConvergenceCriteria(max_iteration=2000) |
| ICP + Chamfer in mm | thr20 | soglia | tutorial ICP di Open3D 0.18: threshold = 0.02 (m) |
| ICP + Chamfer in mm | thr10 | soglia | tutorial colored ICP di Open3D 0.18: voxel_radius = [0.04, 0.02, 0.01] (m), l'ultimo |
| ICP + Chamfer in mm | init_id | inizializzazione | Open3D registration_icp: init = identita' (default), frame canonico senza centrare |
| ICP + Chamfer in mm | init_lmk | inizializzazione | rigida sui 7 landmark NoW rilevati (Procrustes senza scala, come l'avvio di NoW metrical; facebench icp_align prealign='landmark') |
| ICP + Chamfer in mm | rej_pulli | rigetto | Pulli 1999 in Rusinkiewicz e Levoy 2001: scarta il 10% delle coppie peggiori |
| ICP + Chamfer in mm | rej_masuda | rigetto | Masuda 1996 in Rusinkiewicz e Levoy 2001: scarta oltre 2,5 sigma |
| NICP per coppia in mm | default | - | matrice: facebench nonrigid_icp_align (alpha 50 e^(-0,5 d), soglia 6 x 0,5^d, 3 stadi, gamma 1) |
| NICP per coppia in mm | eps_realy | iterazioni | REALY utils/NICP.py nICP_without_keypoints con eval.py epsilon = 1.0 (soglia 0,5^d, senza x 6) |
| NICP per coppia in mm | cap10 | iterazioni | trimesh 5.1 nricp_amberg: max_iter = 10 per passo (default) |
| NICP per coppia in mm | menpo3d | rigidezza | menpo3d non_rigid_icp: stiffness_weights = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2], eps = 1e-3 (default; 'empirically found to perform well for well rigidly aligned facial meshes') |
| NICP per coppia in mm | menpo3d_lmk | landmark | menpo3d non_rigid_icp con landmark: landmark_weights = [5, 2, 0.5, 0, 0, 0, 0, 0] (default), 7 landmark NoW rilevati |
| NoW metrical (rilevatore) | default | - | matrice: plug-in now_metrical/lmk_det (7 landmark NoW, raggio 1,4, dogleg delta_0 0,1, e_3 1e-4, maxiter 200) |
| NoW metrical (rilevatore) | r12 | raggio | Stirling/ESRC, fg2018-competition compute_vertices_to_mesh_distances.py:46 (raggio 1,2) |
| NoW metrical (rilevatore) | icp0 | iterazioni | Stirling/ESRC (fg2018): solo il Procrustes, nessun ICP |
| NoW metrical (rilevatore) | e3_0 | iterazioni | chumpy minimize_dogleg: e_3 = 0 (default; fino a maxiter = 200 o e_1 / e_2) |
| NoW metrical (rilevatore) | lmk5 | landmark | 5 punti di ArcFace / insightface (centri degli occhi, punta del naso, angoli della bocca), dai 68 rilevati |
| NoW metrical (rilevatore) | lmk51 | landmark | 51 interni iBUG (300-W, senza il contorno: 17-67) |
| NoW metrical (rilevatore) | lmk68 | landmark | 68 iBUG (Sagonas et al.) |

Righe: hifi3d 7880 / 7942 del sottocampione senza crop; facescape 7088 / 7916 del sottocampione senza crop; FaMoS 150 scansioni, 11175 coppie.

## hifi3d, rho_fr

| metodo | configurazione | valore [IC] |
|---|---|---|
| ICP + Chamfer in mm | default | 0.648 [0.573, 0.710] |
| ICP + Chamfer in mm | it10 | 0.652 [0.582, 0.711] |
| ICP + Chamfer in mm | it2000 | 0.642 [0.565, 0.706] |
| ICP + Chamfer in mm | thr20 | 0.648 [0.573, 0.709] |
| ICP + Chamfer in mm | thr10 | 0.648 [0.573, 0.709] |
| ICP + Chamfer in mm | init_id | 0.657 [0.583, 0.718] |
| ICP + Chamfer in mm | init_lmk | 0.645 [0.569, 0.708] |
| ICP + Chamfer in mm | rej_pulli | 0.649 [0.575, 0.710] |
| ICP + Chamfer in mm | rej_masuda | 0.648 [0.573, 0.710] |
| NICP per coppia in mm | default | 0.499 [0.420, 0.579] |
| NICP per coppia in mm | eps_realy | 0.399 [0.307, 0.481] |
| NICP per coppia in mm | cap10 | 0.499 [0.420, 0.579] |
| NICP per coppia in mm | menpo3d | 0.035 [-0.016, 0.083] |
| NICP per coppia in mm | menpo3d_lmk | 0.033 [-0.017, 0.078] |
| NoW metrical (rilevatore) | default | 0.684 [0.615, 0.742] |
| NoW metrical (rilevatore) | r12 | 0.639 [0.568, 0.704] |
| NoW metrical (rilevatore) | icp0 | 0.419 [0.354, 0.480] |
| NoW metrical (rilevatore) | e3_0 | 0.684 [0.615, 0.742] |
| NoW metrical (rilevatore) | lmk5 | 0.684 [0.616, 0.742] |
| NoW metrical (rilevatore) | lmk51 | 0.684 [0.616, 0.742] |
| NoW metrical (rilevatore) | lmk68 | 0.684 [0.615, 0.742] |
| nostro, d_F non cal. | s1234 | 0.644 [0.566, 0.713] |
| nostro, d_F non cal. | s2345 | 0.634 [0.560, 0.703] |
| nostro, d_F cal. | s1234 | 0.754 [0.679, 0.813] |
| nostro, d_F cal. | s2345 | 0.735 [0.658, 0.795] |

Spread fra configurazioni (migliore e peggiore scelte sul punto: post hoc):

| metodo | n | intervallo [IC] | SD [IC] | migliore - peggiore [IC] | default - peggiore [IC] | migliore / peggiore |
|---|---|---|---|---|---|---|
| ICP + Chamfer in mm | 9 | 0.015 [0.007, 0.030] | 0.004 [0.002, 0.009] | +0.015 [+0.006, +0.028] | +0.006 [+0.001, +0.014] | init_id / it2000 |
| NICP per coppia in mm | 5 | 0.466 [0.372, 0.560] | 0.240 [0.187, 0.289] | +0.466 [+0.370, +0.557] | +0.466 [+0.370, +0.557] | default / menpo3d_lmk |
| NoW metrical (rilevatore) | 7 | 0.265 [0.214, 0.320] | 0.099 [0.080, 0.119] | +0.265 [+0.214, +0.320] | +0.265 [+0.214, +0.320] | lmk51 / icp0 |
| nostro, d_F non cal. | 2 | 0.009 [0.001, 0.045] | 0.007 [0.000, 0.032] | +0.009 [-0.024, +0.045] | - | s1234 / s2345 |
| nostro, d_F cal. | 2 | 0.018 [0.002, 0.038] | 0.013 [0.001, 0.027] | +0.018 [-0.003, +0.038] | - | s1234 / s2345 |
| nostro, 4 letture (2 semi x cal./non cal.) | 4 | 0.119 [0.068, 0.179] | 0.062 [0.034, 0.090] | +0.119 [+0.059, +0.177] | - | s1234 d_F cal. / s2345 d_F non cal. |

Intervallo del metodo - scarto fra i nostri semi (stesse repliche):

| metodo | lettura nostra | scarto fra i semi [IC] | intervallo - scarto [IC] | lettura |
|---|---|---|---|---|
| ICP + Chamfer in mm | d_F non cal. | 0.009 [0.001, 0.045] | +0.006 [-0.032, +0.023] | non distinguibile |
| NICP per coppia in mm | d_F non cal. | 0.009 [0.001, 0.045] | +0.457 [+0.353, +0.546] | piu' instabile del seme |
| NoW metrical (rilevatore) | d_F non cal. | 0.009 [0.001, 0.045] | +0.255 [+0.186, +0.307] | piu' instabile del seme |
| ICP + Chamfer in mm | d_F cal. | 0.018 [0.002, 0.038] | -0.003 [-0.025, +0.020] | non distinguibile |
| NICP per coppia in mm | d_F cal. | 0.018 [0.002, 0.038] | +0.448 [+0.350, +0.542] | piu' instabile del seme |
| NoW metrical (rilevatore) | d_F cal. | 0.018 [0.002, 0.038] | +0.247 [+0.193, +0.303] | piu' instabile del seme |

Inversioni (tutte le combinazioni di configurazioni; IC = delta appaiato con IC che esclude 0):

| a | b | combinazioni | a meglio | b meglio | a meglio (IC) | b meglio (IC) | delta min / max |
|---|---|---|---|---|---|---|---|
| ICP + Chamfer in mm | NICP per coppia in mm | 45 | 45 | 0 | 45 | 0 | +0.143 / +0.625 |
| ICP + Chamfer in mm | NoW metrical (rilevatore) | 63 | 18 | 45 | 9 | 40 | -0.042 / +0.238 |
| ICP + Chamfer in mm | nostro, d_F non cal. | 18 | 17 | 1 | 0 | 0 | -0.001 / +0.023 |
| ICP + Chamfer in mm | nostro, d_F cal. | 18 | 0 | 18 | 0 | 18 | -0.111 / -0.078 |
| NICP per coppia in mm | NoW metrical (rilevatore) | 35 | 2 | 33 | 0 | 32 | -0.651 / +0.080 |
| NICP per coppia in mm | nostro, d_F non cal. | 10 | 0 | 10 | 0 | 10 | -0.611 / -0.135 |
| NICP per coppia in mm | nostro, d_F cal. | 10 | 0 | 10 | 0 | 10 | -0.721 / -0.236 |
| NoW metrical (rilevatore) | nostro, d_F non cal. | 14 | 11 | 3 | 0 | 2 | -0.224 / +0.050 |
| NoW metrical (rilevatore) | nostro, d_F cal. | 14 | 0 | 14 | 0 | 14 | -0.334 / -0.051 |

Classifiche dei quattro metodi (sul punto, tutte le combinazioni; * = configurazioni di default e s1234):

- nostro d_F non cal.: 7 classifiche distinte; now > icp > nostro_form > nicp * (67%); icp > nostro_form > now > nicp (14%); icp > now > nostro_form > nicp (7%); icp > nostro_form > nicp > now (5%)
- nostro d_F cal.: 3 classifiche distinte; nostro_form_cal > now > icp > nicp * (71%); nostro_form_cal > icp > now > nicp (23%); nostro_form_cal > icp > nicp > now (6%)

## facescape, rho_fr

| metodo | configurazione | valore [IC] |
|---|---|---|
| ICP + Chamfer in mm | default | 0.464 [0.393, 0.528] |
| ICP + Chamfer in mm | it10 | 0.467 [0.396, 0.531] |
| ICP + Chamfer in mm | it2000 | 0.464 [0.393, 0.528] |
| ICP + Chamfer in mm | thr20 | 0.463 [0.393, 0.528] |
| ICP + Chamfer in mm | thr10 | 0.461 [0.391, 0.525] |
| ICP + Chamfer in mm | init_id | 0.464 [0.393, 0.529] |
| ICP + Chamfer in mm | init_lmk | 0.464 [0.393, 0.528] |
| ICP + Chamfer in mm | rej_pulli | 0.463 [0.392, 0.528] |
| ICP + Chamfer in mm | rej_masuda | 0.461 [0.391, 0.525] |
| NICP per coppia in mm | default | 0.385 [0.302, 0.466] |
| NICP per coppia in mm | eps_realy | 0.313 [0.235, 0.386] |
| NICP per coppia in mm | cap10 | 0.385 [0.302, 0.466] |
| NICP per coppia in mm | menpo3d | 0.001 [-0.042, 0.044] |
| NICP per coppia in mm | menpo3d_lmk | -0.002 [-0.043, 0.042] |
| NoW metrical (rilevatore) | default | 0.676 [0.607, 0.733] |
| NoW metrical (rilevatore) | r12 | 0.635 [0.550, 0.705] |
| NoW metrical (rilevatore) | icp0 | 0.251 [0.195, 0.305] |
| NoW metrical (rilevatore) | e3_0 | 0.676 [0.607, 0.733] |
| NoW metrical (rilevatore) | lmk5 | 0.676 [0.607, 0.733] |
| NoW metrical (rilevatore) | lmk51 | 0.676 [0.607, 0.733] |
| NoW metrical (rilevatore) | lmk68 | 0.676 [0.607, 0.733] |
| nostro, d_F non cal. | s1234 | 0.696 [0.625, 0.765] |
| nostro, d_F non cal. | s2345 | 0.694 [0.611, 0.768] |
| nostro, d_F cal. | s1234 | 0.670 [0.599, 0.736] |
| nostro, d_F cal. | s2345 | 0.665 [0.596, 0.731] |

Spread fra configurazioni (migliore e peggiore scelte sul punto: post hoc):

| metodo | n | intervallo [IC] | SD [IC] | migliore - peggiore [IC] | default - peggiore [IC] | migliore / peggiore |
|---|---|---|---|---|---|---|
| ICP + Chamfer in mm | 9 | 0.006 [0.003, 0.010] | 0.002 [0.001, 0.003] | +0.006 [+0.002, +0.010] | +0.003 [-0.001, +0.007] | it10 / thr10 |
| NICP per coppia in mm | 5 | 0.386 [0.293, 0.487] | 0.200 [0.148, 0.254] | +0.386 [+0.290, +0.485] | +0.386 [+0.290, +0.485] | default / menpo3d_lmk |
| NoW metrical (rilevatore) | 7 | 0.425 [0.351, 0.500] | 0.159 [0.131, 0.187] | +0.425 [+0.351, +0.500] | +0.425 [+0.350, +0.499] | e3_0 / icp0 |
| nostro, d_F non cal. | 2 | 0.002 [0.001, 0.041] | 0.002 [0.000, 0.029] | +0.002 [-0.031, +0.037] | - | s1234 / s2345 |
| nostro, d_F cal. | 2 | 0.005 [0.000, 0.035] | 0.003 [0.000, 0.024] | +0.005 [-0.024, +0.032] | - | s1234 / s2345 |
| nostro, 4 letture (2 semi x cal./non cal.) | 4 | 0.031 [0.016, 0.067] | 0.016 [0.007, 0.030] | +0.031 [-0.003, +0.064] | - | s1234 d_F non cal. / s2345 d_F cal. |

Intervallo del metodo - scarto fra i nostri semi (stesse repliche):

| metodo | lettura nostra | scarto fra i semi [IC] | intervallo - scarto [IC] | lettura |
|---|---|---|---|---|
| ICP + Chamfer in mm | d_F non cal. | 0.002 [0.001, 0.041] | +0.004 [-0.035, +0.007] | non distinguibile |
| NICP per coppia in mm | d_F non cal. | 0.002 [0.001, 0.041] | +0.384 [+0.272, +0.475] | piu' instabile del seme |
| NoW metrical (rilevatore) | d_F non cal. | 0.002 [0.001, 0.041] | +0.423 [+0.334, +0.486] | piu' instabile del seme |
| ICP + Chamfer in mm | d_F cal. | 0.005 [0.000, 0.035] | +0.001 [-0.029, +0.007] | non distinguibile |
| NICP per coppia in mm | d_F cal. | 0.005 [0.000, 0.035] | +0.382 [+0.274, +0.476] | piu' instabile del seme |
| NoW metrical (rilevatore) | d_F cal. | 0.005 [0.000, 0.035] | +0.421 [+0.334, +0.485] | piu' instabile del seme |

Inversioni (tutte le combinazioni di configurazioni; IC = delta appaiato con IC che esclude 0):

| a | b | combinazioni | a meglio | b meglio | a meglio (IC) | b meglio (IC) | delta min / max |
|---|---|---|---|---|---|---|---|
| ICP + Chamfer in mm | NICP per coppia in mm | 45 | 45 | 0 | 45 | 0 | +0.076 / +0.468 |
| ICP + Chamfer in mm | NoW metrical (rilevatore) | 63 | 9 | 54 | 9 | 54 | -0.215 / +0.216 |
| ICP + Chamfer in mm | nostro, d_F non cal. | 18 | 0 | 18 | 0 | 18 | -0.236 / -0.227 |
| ICP + Chamfer in mm | nostro, d_F cal. | 18 | 0 | 18 | 0 | 18 | -0.209 / -0.199 |
| NICP per coppia in mm | NoW metrical (rilevatore) | 35 | 3 | 32 | 2 | 32 | -0.677 / +0.134 |
| NICP per coppia in mm | nostro, d_F non cal. | 10 | 0 | 10 | 0 | 10 | -0.698 / -0.309 |
| NICP per coppia in mm | nostro, d_F cal. | 10 | 0 | 10 | 0 | 10 | -0.671 / -0.281 |
| NoW metrical (rilevatore) | nostro, d_F non cal. | 14 | 0 | 14 | 0 | 2 | -0.446 / -0.018 |
| NoW metrical (rilevatore) | nostro, d_F cal. | 14 | 10 | 4 | 0 | 2 | -0.419 / +0.010 |

Classifiche dei quattro metodi (sul punto, tutte le combinazioni; * = configurazioni di default e s1234):

- nostro d_F non cal.: 3 classifiche distinte; nostro_form > now > icp > nicp * (86%); nostro_form > icp > nicp > now (9%); nostro_form > icp > now > nicp (6%)
- nostro d_F cal.: 4 classifiche distinte; now > nostro_form_cal > icp > nicp * (71%); nostro_form_cal > now > icp > nicp (14%); nostro_form_cal > icp > nicp > now (9%); nostro_form_cal > icp > now > nicp (6%)

## famos, rank1

| metodo | configurazione | valore [IC] |
|---|---|---|
| ICP + Chamfer in mm | default | 0.947 [0.913, 0.980] |
| ICP + Chamfer in mm | it10 | 0.947 [0.907, 0.980] |
| ICP + Chamfer in mm | it2000 | 0.947 [0.913, 0.980] |
| ICP + Chamfer in mm | thr20 | 0.947 [0.913, 0.980] |
| ICP + Chamfer in mm | thr10 | 0.940 [0.900, 0.973] |
| ICP + Chamfer in mm | init_id | 0.947 [0.913, 0.980] |
| ICP + Chamfer in mm | init_lmk | 0.947 [0.913, 0.980] |
| ICP + Chamfer in mm | rej_pulli | 0.953 [0.927, 0.980] |
| ICP + Chamfer in mm | rej_masuda | 0.940 [0.900, 0.973] |
| NICP per coppia in mm | default | 0.947 [0.913, 0.980] |
| NICP per coppia in mm | eps_realy | 0.947 [0.913, 0.973] |
| NICP per coppia in mm | cap10 | 0.947 [0.913, 0.980] |
| NICP per coppia in mm | menpo3d | 0.000 [0.000, 0.000] |
| NICP per coppia in mm | menpo3d_lmk | 0.000 [0.000, 0.000] |
| NoW metrical (rilevatore) | default | 0.967 [0.940, 0.993] |
| NoW metrical (rilevatore) | r12 | 0.933 [0.900, 0.967] |
| NoW metrical (rilevatore) | icp0 | 0.840 [0.793, 0.887] |
| NoW metrical (rilevatore) | e3_0 | 0.967 [0.940, 0.993] |
| NoW metrical (rilevatore) | lmk5 | 0.967 [0.940, 0.993] |
| NoW metrical (rilevatore) | lmk51 | 0.967 [0.940, 0.993] |
| NoW metrical (rilevatore) | lmk68 | 0.967 [0.940, 0.993] |
| nostro, d_F non cal. | s1234 | 0.853 [0.813, 0.900] |
| nostro, d_F non cal. | s2345 | 0.820 [0.767, 0.880] |
| nostro, d_F cal. | s1234 | 0.873 [0.827, 0.927] |
| nostro, d_F cal. | s2345 | 0.853 [0.807, 0.907] |

Spread fra configurazioni (migliore e peggiore scelte sul punto: post hoc):

| metodo | n | intervallo [IC] | SD [IC] | migliore - peggiore [IC] | default - peggiore [IC] | migliore / peggiore |
|---|---|---|---|---|---|---|
| ICP + Chamfer in mm | 9 | 0.013 [0.000, 0.033] | 0.004 [0.000, 0.011] | +0.013 [+0.000, +0.033] | +0.007 [+0.000, +0.020] | rej_pulli / thr10 |
| NICP per coppia in mm | 5 | 0.947 [0.920, 0.980] | 0.519 [0.501, 0.536] | +0.947 [+0.913, +0.980] | +0.947 [+0.913, +0.980] | default / menpo3d |
| NoW metrical (rilevatore) | 7 | 0.127 [0.093, 0.160] | 0.047 [0.035, 0.060] | +0.127 [+0.093, +0.160] | +0.127 [+0.093, +0.160] | default / icp0 |
| nostro, d_F non cal. | 2 | 0.033 [0.000, 0.080] | 0.024 [0.000, 0.057] | +0.033 [-0.007, +0.080] | - | s1234 / s2345 |
| nostro, d_F cal. | 2 | 0.020 [0.000, 0.067] | 0.014 [0.000, 0.047] | +0.020 [-0.020, +0.067] | - | s1234 / s2345 |
| nostro, 4 letture (2 semi x cal./non cal.) | 4 | 0.053 [0.020, 0.113] | 0.022 [0.008, 0.048] | +0.053 [+0.000, +0.113] | - | s1234 d_F cal. / s2345 d_F non cal. |

Intervallo del metodo - scarto fra i nostri semi (stesse repliche):

| metodo | lettura nostra | scarto fra i semi [IC] | intervallo - scarto [IC] | lettura |
|---|---|---|---|---|
| ICP + Chamfer in mm | d_F non cal. | 0.033 [0.000, 0.080] | -0.020 [-0.060, +0.013] | non distinguibile |
| NICP per coppia in mm | d_F non cal. | 0.033 [0.000, 0.080] | +0.913 [+0.860, +0.960] | piu' instabile del seme |
| NoW metrical (rilevatore) | d_F non cal. | 0.033 [0.000, 0.080] | +0.093 [+0.040, +0.140] | piu' instabile del seme |
| ICP + Chamfer in mm | d_F cal. | 0.020 [0.000, 0.067] | -0.007 [-0.047, +0.013] | non distinguibile |
| NICP per coppia in mm | d_F cal. | 0.020 [0.000, 0.067] | +0.927 [+0.867, +0.967] | piu' instabile del seme |
| NoW metrical (rilevatore) | d_F cal. | 0.020 [0.000, 0.067] | +0.107 [+0.060, +0.140] | piu' instabile del seme |

Inversioni (tutte le combinazioni di configurazioni; IC = delta appaiato con IC che esclude 0):

| a | b | combinazioni | a meglio | b meglio | a meglio (IC) | b meglio (IC) | delta min / max |
|---|---|---|---|---|---|---|---|
| ICP + Chamfer in mm | NICP per coppia in mm | 45 | 21 | 6 | 18 | 0 | -0.007 / +0.953 |
| ICP + Chamfer in mm | NoW metrical (rilevatore) | 63 | 18 | 45 | 9 | 0 | -0.027 / +0.113 |
| ICP + Chamfer in mm | nostro, d_F non cal. | 18 | 18 | 0 | 18 | 0 | +0.087 / +0.133 |
| ICP + Chamfer in mm | nostro, d_F cal. | 18 | 18 | 0 | 18 | 0 | +0.067 / +0.100 |
| NICP per coppia in mm | NoW metrical (rilevatore) | 35 | 6 | 29 | 3 | 14 | -0.967 / +0.107 |
| NICP per coppia in mm | nostro, d_F non cal. | 10 | 6 | 4 | 6 | 4 | -0.853 / +0.127 |
| NICP per coppia in mm | nostro, d_F cal. | 10 | 6 | 4 | 6 | 4 | -0.873 / +0.093 |
| NoW metrical (rilevatore) | nostro, d_F non cal. | 14 | 13 | 1 | 12 | 0 | -0.013 / +0.147 |
| NoW metrical (rilevatore) | nostro, d_F cal. | 14 | 12 | 2 | 12 | 0 | -0.033 / +0.113 |

Classifiche dei quattro metodi (sul punto, tutte le combinazioni; * = configurazioni di default e s1234):

- nostro d_F non cal.: 9 classifiche distinte; now > icp > nicp > nostro_form * (33%); now > icp > nostro_form > nicp (29%); icp > nicp > now > nostro_form (10%); now > nicp > icp > nostro_form (10%)
- nostro d_F cal.: 9 classifiche distinte; now > icp > nicp > nostro_form_cal * (33%); now > icp > nostro_form_cal > nicp (29%); now > nicp > icp > nostro_form_cal (10%); icp > nicp > now > nostro_form_cal (7%)

## famos, mAP

| metodo | configurazione | valore [IC] |
|---|---|---|
| ICP + Chamfer in mm | default | 0.822 [0.773, 0.873] |
| ICP + Chamfer in mm | it10 | 0.825 [0.773, 0.879] |
| ICP + Chamfer in mm | it2000 | 0.822 [0.772, 0.873] |
| ICP + Chamfer in mm | thr20 | 0.823 [0.774, 0.874] |
| ICP + Chamfer in mm | thr10 | 0.825 [0.776, 0.877] |
| ICP + Chamfer in mm | init_id | 0.822 [0.772, 0.873] |
| ICP + Chamfer in mm | init_lmk | 0.822 [0.772, 0.873] |
| ICP + Chamfer in mm | rej_pulli | 0.831 [0.781, 0.882] |
| ICP + Chamfer in mm | rej_masuda | 0.826 [0.775, 0.877] |
| NICP per coppia in mm | default | 0.797 [0.740, 0.854] |
| NICP per coppia in mm | eps_realy | 0.786 [0.726, 0.846] |
| NICP per coppia in mm | cap10 | 0.797 [0.740, 0.854] |
| NICP per coppia in mm | menpo3d | 0.099 [0.085, 0.111] |
| NICP per coppia in mm | menpo3d_lmk | 0.111 [0.094, 0.128] |
| NoW metrical (rilevatore) | default | 0.865 [0.812, 0.914] |
| NoW metrical (rilevatore) | r12 | 0.802 [0.747, 0.855] |
| NoW metrical (rilevatore) | icp0 | 0.599 [0.512, 0.689] |
| NoW metrical (rilevatore) | e3_0 | 0.865 [0.812, 0.914] |
| NoW metrical (rilevatore) | lmk5 | 0.865 [0.812, 0.914] |
| NoW metrical (rilevatore) | lmk51 | 0.865 [0.812, 0.914] |
| NoW metrical (rilevatore) | lmk68 | 0.865 [0.812, 0.914] |
| nostro, d_F non cal. | s1234 | 0.664 [0.600, 0.739] |
| nostro, d_F non cal. | s2345 | 0.641 [0.566, 0.724] |
| nostro, d_F cal. | s1234 | 0.651 [0.585, 0.726] |
| nostro, d_F cal. | s2345 | 0.628 [0.549, 0.712] |

Spread fra configurazioni (migliore e peggiore scelte sul punto: post hoc):

| metodo | n | intervallo [IC] | SD [IC] | migliore - peggiore [IC] | default - peggiore [IC] | migliore / peggiore |
|---|---|---|---|---|---|---|
| ICP + Chamfer in mm | 9 | 0.009 [0.006, 0.013] | 0.003 [0.002, 0.004] | +0.009 [+0.006, +0.013] | +0.001 [+0.000, +0.002] | rej_pulli / it2000 |
| NICP per coppia in mm | 5 | 0.698 [0.648, 0.752] | 0.377 [0.350, 0.407] | +0.698 [+0.648, +0.752] | +0.698 [+0.648, +0.752] | default / menpo3d |
| NoW metrical (rilevatore) | 7 | 0.265 [0.211, 0.316] | 0.099 [0.079, 0.118] | +0.265 [+0.211, +0.316] | +0.265 [+0.211, +0.316] | default / icp0 |
| nostro, d_F non cal. | 2 | 0.023 [0.002, 0.047] | 0.016 [0.001, 0.033] | +0.023 [+0.000, +0.047] | - | s1234 / s2345 |
| nostro, d_F cal. | 2 | 0.022 [0.003, 0.044] | 0.016 [0.002, 0.031] | +0.022 [+0.001, +0.044] | - | s1234 / s2345 |
| nostro, 4 letture (2 semi x cal./non cal.) | 4 | 0.036 [0.013, 0.075] | 0.015 [0.006, 0.032] | +0.036 [-0.004, +0.075] | - | s1234 d_F non cal. / s2345 d_F cal. |

Intervallo del metodo - scarto fra i nostri semi (stesse repliche):

| metodo | lettura nostra | scarto fra i semi [IC] | intervallo - scarto [IC] | lettura |
|---|---|---|---|---|
| ICP + Chamfer in mm | d_F non cal. | 0.023 [0.002, 0.047] | -0.014 [-0.036, +0.007] | non distinguibile |
| NICP per coppia in mm | d_F non cal. | 0.023 [0.002, 0.047] | +0.675 [+0.612, +0.738] | piu' instabile del seme |
| NoW metrical (rilevatore) | d_F non cal. | 0.023 [0.002, 0.047] | +0.242 [+0.197, +0.283] | piu' instabile del seme |
| ICP + Chamfer in mm | d_F cal. | 0.022 [0.003, 0.044] | -0.013 [-0.035, +0.007] | non distinguibile |
| NICP per coppia in mm | d_F cal. | 0.022 [0.003, 0.044] | +0.676 [+0.614, +0.738] | piu' instabile del seme |
| NoW metrical (rilevatore) | d_F cal. | 0.022 [0.003, 0.044] | +0.243 [+0.199, +0.287] | piu' instabile del seme |

Inversioni (tutte le combinazioni di configurazioni; IC = delta appaiato con IC che esclude 0):

| a | b | combinazioni | a meglio | b meglio | a meglio (IC) | b meglio (IC) | delta min / max |
|---|---|---|---|---|---|---|---|
| ICP + Chamfer in mm | NICP per coppia in mm | 45 | 45 | 0 | 22 | 0 | +0.025 / +0.732 |
| ICP + Chamfer in mm | NoW metrical (rilevatore) | 63 | 18 | 45 | 9 | 45 | -0.043 / +0.231 |
| ICP + Chamfer in mm | nostro, d_F non cal. | 18 | 18 | 0 | 18 | 0 | +0.157 / +0.189 |
| ICP + Chamfer in mm | nostro, d_F cal. | 18 | 18 | 0 | 18 | 0 | +0.171 / +0.202 |
| NICP per coppia in mm | NoW metrical (rilevatore) | 35 | 3 | 32 | 3 | 29 | -0.766 / +0.198 |
| NICP per coppia in mm | nostro, d_F non cal. | 10 | 6 | 4 | 6 | 4 | -0.566 / +0.156 |
| NICP per coppia in mm | nostro, d_F cal. | 10 | 6 | 4 | 6 | 4 | -0.552 / +0.169 |
| NoW metrical (rilevatore) | nostro, d_F non cal. | 14 | 12 | 2 | 12 | 2 | -0.065 / +0.223 |
| NoW metrical (rilevatore) | nostro, d_F cal. | 14 | 12 | 2 | 12 | 1 | -0.051 / +0.236 |

Classifiche dei quattro metodi (sul punto, tutte le combinazioni; * = configurazioni di default e s1234):

- nostro d_F non cal.: 6 classifiche distinte; now > icp > nicp > nostro_form * (43%); now > icp > nostro_form > nicp (29%); icp > now > nicp > nostro_form (9%); icp > nicp > nostro_form > now (9%)
- nostro d_F cal.: 6 classifiche distinte; now > icp > nicp > nostro_form_cal * (43%); now > icp > nostro_form_cal > nicp (29%); icp > now > nicp > nostro_form_cal (9%); icp > nicp > nostro_form_cal > now (9%)

<!-- da qui in giu': scritto a mano, conservato da ri.py results -->

## Lettura (scritta a mano dopo i numeri; descrittiva)

**Controlli** (`controls/`): default ICP identico alla matrice (0 mm su HIFI3D e FaceScape, 9e-14 su FaMoS); NoW esatto
sulla CPU della matrice (`check_now_node_*`: 0 su tutte le coppie) e su FaMoS; NICP FaceScape entro 8e-5 mm, HIFI3D
esatto sul nodo della matrice per ogni gruppo di coppie (`check_nicp_node_*`: 0 su 4732 + 812; i256-a10: griglia = matrice).
La griglia valutata e' quella calcolata (ICP e NoW su a768-l40s-01, NICP su i256-a10-06, a512-l4-06, i256-a10-10):
scarti dalla matrice <= 2,3e-3 mm su 89/18 coppie (NoW) e <= 7,4e-4 mm (NICP), irrilevanti per i ranghi.

**ICP + Chamfer in mm e' stabile**: intervallo fra 9 configurazioni 0,015 [0,007; 0,030] (HIFI3D), 0,006 [0,003; 0,010]
(FaceScape), rank-1 FaMoS 0,013 [0,000; 0,033], mAP 0,009 [0,006; 0,013]: non distinguibile dallo scarto fra i nostri
due semi (d_F non cal.: 0,009 / 0,002 / 0,033 / 0,023) in tutte le misure.

**NoW metrical e' stabile tranne che su due manopole**: landmark 5/51/68 ed e_3 = 0 di chumpy danno lo stesso valore del
default alla terza cifra; il raggio 1,2 toglie 0,045 (HIFI3D), 0,041 (FaceScape), 0,033 di rank-1; togliere l'ICP (solo
Procrustes, Stirling) toglie 0,265 / 0,425 / 0,127. L'intervallo dichiarato (0,265 [0,214; 0,320] HIFI3D, 0,425
[0,351; 0,500] FaceScape, rank-1 0,127, mAP 0,265) e' tutto dovuto a icp0; senza icp0 (selezione post hoc, sul punto)
resta 0,045 / 0,041 / 0,033, comunque sopra lo scarto fra i semi sulle sintetiche.

**NICP per coppia e' sensibile al programma della rigidezza**: `cap10` coincide col default ovunque (il primo stadio
converge in 2 iterazioni); `eps_realy` toglie 0,100 (HIFI3D) e 0,072 (FaceScape), niente su FaMoS; il programma di
menpo3d (rigidezza finale 0,2) porta il residuo a ~0,004 mm contro ~2 mm del default e la distanza perde l'identita'
(rho 0,035 / 0,001, rank-1 0,000): e' il valore di default di una libreria per volti, citato, ma come lettura di
distanza e' un caso limite, e da solo fa l'intervallo dichiarato (0,466 / 0,386 / 0,947).

**Inversioni**: tutte quelle con IC su entrambi i versi passano per icp0 di NoW o per menpo3d del NICP (ICP contro NoW:
9 contro 40 combinazioni su HIFI3D, 9 contro 54 su FaceScape; su FaMoS NICP contro il nostro 6 contro 4, NoW contro il
nostro in mAP 12 contro 2). Fuori da queste due configurazioni la classifica fra metodi non si inverte con IC; il nostro
d_F cal. resta primo su HIFI3D in tutte le 630 combinazioni, il nostro d_F non cal. primo su FaceScape in tutte, e su
FaMoS il nostro resta sotto l'ICP in tutte le 9 configurazioni e sotto NoW in tutte tranne icp0 (rank-1 0,840 contro 0,853 / 0,820).

**Conclusione**: la critica "il concorrente cambia con gli iperparametri" vale per il NICP e, in misura minore, per il
raggio e l'ICP di NoW; l'ICP rigido in mm e NoW nelle manopole di landmark e iterazioni sono stabili quanto il nostro
seme. Non cambia nessuna conclusione della matrice o di FaMoS se si escludono i due casi limite.
