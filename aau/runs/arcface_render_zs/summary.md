# Protocollo dichiarato prima delle eval: ArcFace su render di sola geometria, zero-shot FaceVerse con espressioni (e HIFI3D)

Dichiarato il 2026-10-06 19:21:40 CEST, prima di generare qualunque render o embedding ArcFace su FaceVerse o HIFI3D
(nessun numero ArcFace su questi domini esiste a quest'ora). Numeri gia' visti e noti in anticipo:
le baseline e i modelli di `aau/runs/ws_faceverse_expr/summary.md` (NICP P2Tri rank-1 0.959,
congiunto BFM 0.680) e l'AUC 1.000 di ArcFace su Multiface (13 soggetti).

**Domanda.** ArcFace su render di sola geometria, con la pipeline di Multiface, riconosce le
identita' FaceVerse con espressioni quanto NICP, o piu' del modello appreso?

**Pipeline (identica a Multiface, `aau/multiface/ws3a_render.py` + `ws3a_perceptual.py`):**
- mesh: `datasets/FACEVERSE_ZS/expr_view/npz`, i 100 soggetti di `select_subjects` (seed 1234), 6 topologie;
- rotazione fissa dal frame del dominio a quello del renderer (alto -y, naso -z), UNA per dominio,
  scelta guardando i render di controllo prima di calcolare gli embedding: FaceVerse ha gia' alto -y
  e naso -z (tabella di aau/runs/ws_frame) -> attesa identita'; HIFI3D alto +y, naso +z -> attesa Rx(180).
  Le facce non si invertono: il renderer ombreggia a due facce, il verso dei triangoli non entra;
- `normalize_maxabs` per mesh, camera unica calcolata su tutte le 600 mesh del dominio, yaw 0, -30, +30,
  512 px, `render_mesh` (grigio ombreggiato);
- crop fisso ricalibrato su QUESTI render: detector di insightface su tutti i render del dominio, mediana
  per yaw dei 5 landmark, `estimate_norm` -> similarita' 2x3 congelata; poi il detector non si usa piu'.
  Conteggio dei fallimenti del detector per topologia riportato;
- ArcFace `w600k_r50.onnx`: embedding per vista, L2, media sulle viste, rinormalizzazione; distanza = 1 - coseno.

**PRIMARIO (identico a `aau/runs/ws_faceverse_expr/protocol.md`, revisione 1):** riconoscimento
d'identita' sulle 5 topologie senza crop. Retrieval: 20 coppie ordinate (t1, t2) x 100 query, galleria
di 100 mesh in t2; rank-1 e mAP (= MRR). Verifica: AUC di -distanza, 1000 coppie stessa persona contro
99.000 persone diverse, 10 coppie non ordinate di topologie. IC 95% bootstrap per soggetto, 1000
repliche, con le STESSE repliche del summary esistente (seme `stable_seed(1234, "expr_recognition")`):
le righe delle baseline devono riprodurre quei numeri, ed e' il controllo. Funzioni di calcolo importate
da `aau/zs3dmm/zs_expr_summarize.py`, non riscritte.
- Riga di riferimento: **ArcFace, 3 viste, ombreggiato**.
- Delta APPAIATI ArcFace - {NICP P2Tri, ICP rigido + Chamfer, Chamfer faceBench, BFM+ICT in convenzione BFM}.
- Crop: le stesse misure, a parte, mai nel primario.

**Lettura fissata ora:** "ArcFace pari a NICP" se il CI del delta ArcFace - NICP P2Tri sul rank-1 contiene
0 o e' positivo; "ArcFace sopra il congiunto" se il CI del delta ArcFace - congiunto (BFM) sul rank-1 e'
tutto sopra 0. Le stesse due letture valgono per l'AUC.

**Ablazioni (secondarie, non sostituiscono la riga di riferimento):** 1 vista (yaw 0) contro 3 viste;
normal map (normali in spazio camera, RGB = (n+1)/2) al posto dell'ombreggiatura, solo se costa meno
di un'ora, con lo STESSO crop dei render ombreggiati (stessa camera, stessa geometria, quindi stesso
riquadro del volto) e anche lei in 1 e 3 viste. Delta appaiati 3 viste - 1 vista e normal map - ombreggiato.

**HIFI3D senza espressioni (se c'e' tempo):** stesse regole su `datasets/HIFI3D/eval_view/npz`, 100
soggetti, neutre. Le baseline faceBench stesso-soggetto (che su HIFI3D mancano) si calcolano con
`zs_bl_same.py` (stessa pipeline, stesso seme). Il congiunto in convenzione BFM completa
(`joint_frame-xmymz_flip`) entra solo se i suoi embedding per mesh sono gia' disponibili o
calcolabili in giornata; altrimenti la riga e' riportata come mancante.

**GPU:** la pipeline ArcFace di Multiface gira su onnxruntime CPU; i job sono CPU-only (`--gres=NONE`).

---

# Risultati: FaceVerse v2 con espressioni casuali

Soggetti: 100 (`select_subjects`, seed 1234), mesh da `datasets/FACEVERSE_ZS/expr_view/npz`, baseline da `aau/runs/ws_faceverse_expr/data_736f96956a/baselines`, congiunto da `aau/runs/ws_faceverse_expr/data_736f96956a/joint_flip_topology/zs_zeroshot`. CI 95% bootstrap per soggetto, 1000 repliche (le stesse per tutte le righe).

Crop fisso ricalibrato su questi render (`aau/runs/arcface_render_zs/fv_expr/shaded/renders/arcface_align.json`): detector su 1800 render, fallito su 443. Per topologia (falliti/render): crop 15/300, down8k 12/300, noisy 300/300, original 9/300, remesh 91/300, up60k 16/300.
Landmark mediani per vista (5 punti, px su 512) e IQR massimo fra i 5 punti:
- yaw +000.00: 457 detection, IQR max 17.9 px, kps [[174.7, 172.9], [335.8, 175.5], [249.3, 264.1], [190.6, 340.8], [318.7, 342.5]]
- yaw +030.00: 430 detection, IQR max 20.6 px, kps [[164.5, 168.7], [298.5, 174.2], [184.1, 256.0], [169.4, 336.1], [275.6, 341.5]]
- yaw -030.00: 470 detection, IQR max 16.6 px, kps [[212.4, 176.9], [346.7, 175.7], [316.8, 260.6], [234.8, 341.1], [338.3, 338.0]]
Camera unica: base_rotation `none`, center [0.0045, -0.0016, 0.1116], scale 2.0307, su 600 mesh.

## PRIMARIO: riconoscimento d'identita', 5 topologie senza crop

Retrieval: 2000 query, galleria di 100 mesh in un'altra topologia; mAP = MRR. Verifica: 1000 coppie stessa persona, 99000 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | 0.750 [0.724, 0.775] | 0.807 [0.785, 0.829] | 0.863 [0.847, 0.879] | 0 |
| ArcFace, ombreggiato, 1 vista | 0.737 [0.712, 0.763] | 0.789 [0.765, 0.813] | 0.843 [0.825, 0.859] | 0 |
| ArcFace, normal map, 3 viste | 0.867 [0.843, 0.889] | 0.911 [0.893, 0.927] | 0.934 [0.921, 0.946] | 0 |
| ArcFace, normal map, 1 vista | 0.827 [0.803, 0.849] | 0.876 [0.857, 0.894] | 0.917 [0.904, 0.930] | 0 |
| BFM+ICT, convenzione BFM | 0.680 [0.634, 0.723] | 0.731 [0.688, 0.770] | 0.875 [0.847, 0.900] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.740 [0.698, 0.783] | 0.775 [0.737, 0.814] | 0.882 [0.853, 0.910] | 0 |
| Rigid ICP + Chamfer | 0.918 [0.895, 0.939] | 0.935 [0.916, 0.953] | 0.986 [0.979, 0.992] | 0 |
| Rigid ICP + NICP + P2P | 0.954 [0.936, 0.971] | 0.964 [0.949, 0.978] | 0.994 [0.991, 0.997] | 0 |
| Rigid ICP + NICP + P2Tri | 0.959 [0.940, 0.975] | 0.968 [0.953, 0.981] | 0.995 [0.992, 0.998] | 0 |
| Chamfer regione stabile | 0.787 [0.748, 0.830] | 0.818 [0.781, 0.856] | 0.894 [0.868, 0.920] | 0 |
| Chamfer intero (stessa implementazione) | 0.751 [0.708, 0.793] | 0.782 [0.743, 0.820] | 0.884 [0.856, 0.911] | 0 |

### Delta appaiati, riferimento ArcFace - baseline (stesse repliche)

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.209 [-0.235, -0.181] (1.000) | -0.161 [-0.183, -0.138] (1.000) | -0.132 [-0.148, -0.117] (1.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) | Rigid ICP + Chamfer | -0.168 [-0.197, -0.136] (1.000) | -0.128 [-0.153, -0.100] (1.000) | -0.123 [-0.140, -0.107] (1.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.010 [-0.034, +0.053] (0.355) | +0.032 [-0.008, +0.073] (0.062) | -0.019 [-0.049, +0.010] (0.873) |
| ArcFace, ombreggiato, 3 viste (riferimento) | BFM+ICT, convenzione BFM | +0.070 [+0.022, +0.118] (0.002) | +0.075 [+0.031, +0.121] (0.000) | -0.012 [-0.040, +0.019] (0.783) |

### Ablazioni (secondarie), delta appaiati

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | ArcFace, ombreggiato, 1 vista | +0.013 [-0.005, +0.030] (0.076) | +0.018 [+0.004, +0.032] (0.001) | +0.020 [+0.013, +0.028] (0.000) |
| ArcFace, normal map, 3 viste | ArcFace, ombreggiato, 3 viste (riferimento) | +0.117 [+0.094, +0.142] (0.000) | +0.104 [+0.085, +0.122] (0.000) | +0.071 [+0.060, +0.083] (0.000) |
| ArcFace, normal map, 1 vista | ArcFace, ombreggiato, 1 vista | +0.090 [+0.064, +0.115] (0.000) | +0.087 [+0.065, +0.111] (0.000) | +0.074 [+0.061, +0.088] (0.000) |
| ArcFace, normal map, 3 viste | ArcFace, normal map, 1 vista | +0.040 [+0.024, +0.056] (0.000) | +0.034 [+0.023, +0.045] (0.000) | +0.017 [+0.012, +0.023] (0.000) |

## A parte: crop (coppie di topologie con crop da un lato)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | 0.881 [0.867, 0.896] | 0.909 [0.897, 0.921] | 0.922 [0.910, 0.932] | 0 |
| ArcFace, ombreggiato, 1 vista | 0.870 [0.854, 0.886] | 0.893 [0.880, 0.908] | 0.908 [0.896, 0.919] | 0 |
| ArcFace, normal map, 3 viste | 0.952 [0.937, 0.964] | 0.966 [0.955, 0.976] | 0.963 [0.954, 0.970] | 0 |
| ArcFace, normal map, 1 vista | 0.924 [0.908, 0.939] | 0.945 [0.932, 0.957] | 0.955 [0.945, 0.963] | 0 |
| BFM+ICT, convenzione BFM | 0.243 [0.205, 0.283] | 0.364 [0.325, 0.405] | 0.801 [0.771, 0.829] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.396 [0.336, 0.457] | 0.479 [0.421, 0.537] | 0.809 [0.781, 0.837] | 0 |
| Rigid ICP + Chamfer | 0.513 [0.455, 0.571] | 0.590 [0.540, 0.640] | 0.777 [0.734, 0.814] | 0 |
| Rigid ICP + NICP + P2P | 0.898 [0.864, 0.929] | 0.924 [0.897, 0.947] | 0.981 [0.971, 0.991] | 0 |
| Rigid ICP + NICP + P2Tri | 0.909 [0.876, 0.938] | 0.934 [0.910, 0.956] | 0.983 [0.974, 0.992] | 0 |
| Chamfer regione stabile | 0.497 [0.442, 0.557] | 0.584 [0.531, 0.637] | 0.808 [0.779, 0.838] | 0 |
| Chamfer intero (stessa implementazione) | 0.442 [0.381, 0.500] | 0.521 [0.462, 0.575] | 0.832 [0.804, 0.860] | 0 |

### Delta appaiati, crop

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.028 [-0.061, +0.008] (0.943) | -0.024 [-0.049, +0.003] (0.961) | -0.061 [-0.075, -0.048] (1.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) | Rigid ICP + Chamfer | +0.368 [+0.312, +0.426] (0.000) | +0.319 [+0.272, +0.371] (0.000) | +0.145 [+0.109, +0.188] (0.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.485 [+0.423, +0.549] (0.000) | +0.431 [+0.372, +0.491] (0.000) | +0.113 [+0.084, +0.142] (0.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) | BFM+ICT, convenzione BFM | +0.638 [+0.599, +0.677] (0.000) | +0.545 [+0.504, +0.585] (0.000) | +0.121 [+0.093, +0.150] (0.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) | ArcFace, ombreggiato, 1 vista | +0.011 [-0.001, +0.024] (0.052) | +0.016 [+0.006, +0.026] (0.000) | +0.014 [+0.009, +0.020] (0.000) |
| ArcFace, normal map, 3 viste | ArcFace, ombreggiato, 3 viste (riferimento) | +0.071 [+0.055, +0.087] (0.000) | +0.057 [+0.045, +0.070] (0.000) | +0.041 [+0.033, +0.049] (0.000) |
| ArcFace, normal map, 1 vista | ArcFace, ombreggiato, 1 vista | +0.054 [+0.035, +0.073] (0.000) | +0.052 [+0.037, +0.067] (0.000) | +0.047 [+0.038, +0.057] (0.000) |
| ArcFace, normal map, 3 viste | ArcFace, normal map, 1 vista | +0.028 [+0.016, +0.040] (0.000) | +0.021 [+0.013, +0.029] (0.000) | +0.008 [+0.005, +0.012] (0.000) |

## Controlli

- riproduzione di `aau/runs/ws_faceverse_expr/recognition.csv` (stesse repliche bootstrap): max |diff| su punto e CI di rank-1, mAP, AUC = 1.11e-16 su 14 righe (chamfer, chamfer_full, chamfer_stable, joint@bfm, nicp_p2p, nicp_p2tri, rigid_icp_chamfer)
- png di controllo: `aau/runs/arcface_render_zs/fv_expr/shaded/control`, `aau/runs/arcface_render_zs/fv_expr/normals/control`; frame: `aau/runs/arcface_render_zs/fv_expr/shaded/frame_check`

---

# Risultati: HIFI3D, neutre (senza espressioni)

Soggetti: 100 (`select_subjects`, seed 1234), mesh da `datasets/HIFI3D/eval_view/npz`, baseline da `aau/runs/ws_hifi3d/data_328f2bfc1a/baselines`, congiunto da `aau/runs/ws_hifi3d/data_328f2bfc1a/joint_frame-xmymz_flip_ranking/zs_zeroshot`. CI 95% bootstrap per soggetto, 1000 repliche (le stesse per tutte le righe).

Crop fisso ricalibrato su questi render (`aau/runs/arcface_render_zs/hifi3d/shaded/renders/arcface_align.json`): detector su 1800 render, fallito su 312. Per topologia (falliti/render): crop 4/300, down8k 0/300, noisy 300/300, original 0/300, remesh 8/300, up60k 0/300.
Landmark mediani per vista (5 punti, px su 512) e IQR massimo fra i 5 punti:
- yaw +000.00: 489 detection, IQR max 12.4 px, kps [[180.4, 166.1], [339.0, 166.9], [257.1, 251.9], [199.8, 336.5], [321.2, 337.5]]
- yaw +030.00: 499 detection, IQR max 11.9 px, kps [[164.6, 167.8], [301.8, 167.0], [195.6, 247.7], [175.0, 334.8], [284.6, 335.7]]
- yaw -030.00: 500 detection, IQR max 12.0 px, kps [[218.3, 169.9], [350.6, 170.0], [319.5, 251.5], [238.6, 335.8], [340.2, 333.9]]
Camera unica: base_rotation `x180`, center [-0.0116, 0.0483, 0.2325], scale 2.1002, su 600 mesh.

## PRIMARIO: riconoscimento d'identita', 5 topologie senza crop

Retrieval: 2000 query, galleria di 100 mesh in un'altra topologia; mAP = MRR. Verifica: 1000 coppie stessa persona, 99000 persone diverse.

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | 0.984 [0.974, 0.992] | 0.991 [0.985, 0.996] | 0.989 [0.985, 0.993] | 0 |
| ArcFace, ombreggiato, 1 vista | 0.931 [0.908, 0.954] | 0.955 [0.939, 0.971] | 0.971 [0.961, 0.979] | 0 |
| BFM+ICT, convenzione BFM | 0.397 [0.363, 0.434] | 0.517 [0.489, 0.549] | 0.755 [0.740, 0.770] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.477 [0.445, 0.513] | 0.564 [0.534, 0.594] | 0.782 [0.767, 0.797] | 0 |
| Rigid ICP + Chamfer | 0.996 [0.991, 0.999] | 0.997 [0.995, 0.999] | 0.999 [0.999, 1.000] | 0 |
| Rigid ICP + NICP + P2P | 0.988 [0.979, 0.996] | 0.989 [0.980, 0.997] | 0.999 [0.999, 1.000] | 1965 |
| Rigid ICP + NICP + P2Tri | 0.990 [0.982, 0.998] | 0.990 [0.982, 0.998] | 1.000 [1.000, 1.000] | 1965 |

### Delta appaiati, riferimento ArcFace - baseline (stesse repliche)

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | -0.006 [-0.019, +0.007] (0.845) | +0.001 [-0.009, +0.012] (0.457) | -0.011 [-0.015, -0.007] (1.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) | Rigid ICP + Chamfer | -0.012 [-0.023, -0.002] (0.995) | -0.006 [-0.012, -0.001] (0.991) | -0.010 [-0.014, -0.007] (1.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.506 [+0.468, +0.541] (0.000) | +0.427 [+0.395, +0.456] (0.000) | +0.207 [+0.191, +0.222] (0.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) | BFM+ICT, convenzione BFM | +0.587 [+0.551, +0.619] (0.000) | +0.474 [+0.443, +0.502] (0.000) | +0.234 [+0.218, +0.249] (0.000) |

### Ablazioni (secondarie), delta appaiati

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | ArcFace, ombreggiato, 1 vista | +0.052 [+0.031, +0.077] (0.000) | +0.036 [+0.022, +0.052] (0.000) | +0.019 [+0.013, +0.025] (0.000) |

## A parte: crop (coppie di topologie con crop da un lato)

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | 0.990 [0.984, 0.996] | 0.994 [0.991, 0.998] | 0.994 [0.992, 0.996] | 0 |
| ArcFace, ombreggiato, 1 vista | 0.960 [0.946, 0.973] | 0.974 [0.965, 0.983] | 0.984 [0.979, 0.989] | 0 |
| BFM+ICT, convenzione BFM | 0.085 [0.057, 0.117] | 0.201 [0.171, 0.233] | 0.716 [0.699, 0.734] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.472 [0.445, 0.499] | 0.536 [0.510, 0.564] | 0.699 [0.684, 0.715] | 0 |
| Rigid ICP + Chamfer | 0.284 [0.237, 0.338] | 0.423 [0.381, 0.470] | 0.888 [0.865, 0.911] | 0 |
| Rigid ICP + NICP + P2P | 0.752 [0.714, 0.788] | 0.834 [0.810, 0.858] | 0.995 [0.993, 0.997] | 1965 |
| Rigid ICP + NICP + P2Tri | 0.777 [0.740, 0.810] | 0.854 [0.829, 0.877] | 0.997 [0.995, 0.998] | 1965 |

### Delta appaiati, crop

| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP: A - B [CI] (P<=0) | AUC: A - B [CI] (P<=0) |
| --- | --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | Rigid ICP + NICP + P2Tri | +0.213 [+0.179, +0.250] (0.000) | +0.141 [+0.118, +0.166] (0.000) | -0.003 [-0.006, -0.001] (0.992) |
| ArcFace, ombreggiato, 3 viste (riferimento) | Rigid ICP + Chamfer | +0.706 [+0.652, +0.755] (0.000) | +0.571 [+0.523, +0.614] (0.000) | +0.107 [+0.082, +0.129] (0.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) | Chamfer (faceBench, 4096 pt) | +0.518 [+0.489, +0.546] (0.000) | +0.458 [+0.430, +0.485] (0.000) | +0.295 [+0.279, +0.310] (0.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) | BFM+ICT, convenzione BFM | +0.905 [+0.874, +0.934] (0.000) | +0.793 [+0.760, +0.823] (0.000) | +0.279 [+0.261, +0.295] (0.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) | ArcFace, ombreggiato, 1 vista | +0.030 [+0.018, +0.043] (0.000) | +0.020 [+0.012, +0.029] (0.000) | +0.010 [+0.006, +0.014] (0.000) |

## Controlli

- riproduzione del summary esistente: nessun `--reference-csv`
- png di controllo: `aau/runs/arcface_render_zs/hifi3d/shaded/control`, `aau/runs/arcface_render_zs/hifi3d/normals/control`; frame: `aau/runs/arcface_render_zs/hifi3d/shaded/frame_check`

---

# Giudizio (scritto il 2026-10-06 22:28:28 CEST, dopo i numeri)

Job: FaceVerse render+embedding 1057301 (finito con `[arcface-zs] OK` ma marcato OUT_OF_MEMORY: 32
processi onnxruntime vicini ai 48G), summary 1057316; HIFI3D render 1057302 (calibrazione bloccata da
oom-kill, TIMEOUT), ripresa 1057317 (96G, un thread per sessione), stesso-soggetto faceBench 1057299,
embedding del congiunto 1057300, summary 1057318.

**Lettura fissata nel protocollo, FaceVerse con espressioni (primario, senza crop):**
- ArcFace pari a NICP? **No.** Delta rank-1 ArcFace - NICP P2Tri -0.209 [-0.235, -0.181]; AUC -0.132 [-0.148, -0.117].
- ArcFace sopra il congiunto? **Si' sul rank-1** (+0.070 [+0.022, +0.118]) e sul mAP; **no sull'AUC**
  (-0.012 [-0.040, +0.019]). ArcFace e' pari a Chamfer faceBench (rank-1 +0.010 [-0.034, +0.053]).
- L'AUC 1.000 di Multiface non si trasferisce: qui ArcFace fa AUC 0.863, sotto ICP rigido + Chamfer.

**HIFI3D, neutre (stesse regole):** ArcFace rank-1 0.984, pari a NICP P2Tri per la regola del
protocollo (-0.006 [-0.019, +0.007]), AUC un po' sotto (-0.011 [-0.015, -0.007]); ben sopra il congiunto
(+0.587). Su NICP pesano 1965 distanze NaN (coppie faceBench fallite, con up60k), contate come +inf:
le righe NICP di HIFI3D sono un limite inferiore.

**Con il crop** ArcFace tiene e i metodi geometrici no: su FaceVerse ArcFace 0.881 contro NICP 0.909
(-0.028 [-0.061, +0.008]); su HIFI3D ArcFace 0.990 contro NICP 0.777.

**Cosa non si puo' separare con questi dati.** Il calo su FaceVerse puo' venire dalle espressioni o
dal dominio: FaceVerse neutro non e' stato fatto. Indizio, non verificato: il detector fallisce su
300/300 render `noisy` in entrambi i domini, e su FaceVerse `noisy` sta in 8 delle 20 coppie del
primario; il blocco crop, dove `noisy` pesa meno, va meglio del primario. Manca un breakdown per
coppia di topologie.

**Ablazioni (secondarie, aggiunte dopo, non cambiano la riga di riferimento):** 3 viste contro 1
vista: +0.013 [-0.005, +0.030] di rank-1 su FaceVerse, +0.052 su HIFI3D. Normal map (solo FaceVerse,
stesso crop): +0.117 [+0.094, +0.142] di rank-1, cioe' 0.867, sempre sotto NICP.
