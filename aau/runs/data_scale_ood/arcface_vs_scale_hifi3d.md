# ArcFace su render contro il run grande (BFM+ICT+GNM 10^5, e036/e072/e108): HIFI3D, stesso protocollo

Soggetti: 100 (`select_subjects`, seed 1234; primi id900001, id900004, id900018), gli stessi dei due summary. CI 95% bootstrap per soggetto, 1000 repliche; P(<=0) = frazione di repliche con differenza <= 0. Script: `aau/zs3dmm/zs_arcface_vs_scale.py` (importa zs_summarize, zs_expr_summarize, zs_arcface_summarize: nessuna misura riscritta). CSV in `aau/runs/data_scale_ood/arcface_vs_scale_hifi3d`.

ArcFace = 1 - coseno della media degli embedding delle 3 viste (yaw 0, +-30), rinormalizzata, dagli `arcface_views.npz` gia' calcolati in `aau/runs/arcface_render_zs/hifi3d` (ombreggiato = riga di riferimento di results_hifi3d.md; normal map = stesso crop, calcolata dopo quel summary e qui alla prima apparizione). Convenzione di results_hifi3d.md invariata: crop fisso ricalibrato (similarita' congelata dalle mediane per yaw dei landmark) applicato a TUTTI i render, detector solo per la calibrazione; su `noisy` il detector fallisce 300/300 e il crop fisso vale come sulle altre. Calibrazione normal map = ombreggiato: True.

Checkpoint: `epoch036/072/108.pth` del run 1060130 (gli stessi di `scale_eNNN_topology`, controllato dagli `eval_key.txt`), frame nativo HIFI3D come in (a) (nessuna rotazione, facce come sono). Per (b) gli embedding vengono da `aau/runs/ws_hifi3d/data_328f2bfc1a/scale_eNNN_embed` (zs_zeroshot.sbatch, WBES_ZS_PART=embed).

## (a) Spearman con la GT `maxabs` (protocollo di `aau/runs/data_scale_ood/hifi/summary.md`)

Righe: le pair_metrics del breakdown (coppie di soggetti diversi, coppie ordinate di topologie, clean); ArcFace letto sulle stesse righe. `chamfer` = Chamfer eval degli script di eval (la colonna del summary (a)), non la Chamfer faceBench. Righe singole: Chamfer ed e0NN col loro seme del summary (riprodotte, vedi Controlli), ArcFace col seme della riga Chamfer (stesse repliche del riferimento 0.372).

| metodo | nocrop_cross (20 coppie ordinate) | subject_pair_mean (clean, 30) |
| --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | 0.245 [0.175, 0.324] | 0.296 [0.210, 0.387] |
| ArcFace, normal map, 3 viste | 0.242 [0.162, 0.330] | 0.270 [0.179, 0.365] |
| BFM+ICT+GNM (10^5), e036 | 0.677 [0.609, 0.737] | 0.767 [0.706, 0.825] |
| BFM+ICT+GNM (10^5), e072 | 0.663 [0.591, 0.725] | 0.792 [0.734, 0.838] |
| BFM+ICT+GNM (10^5), e108 | 0.630 [0.569, 0.689] | 0.795 [0.738, 0.840] |
| Chamfer eval | 0.372 [0.324, 0.422] | 0.743 [0.682, 0.796] |

### (a) Differenze appaiate

Tutte sulle STESSE repliche: il seme della differenza `e108 - Chamfer eval` del summary (a), che qui torna identica (ultima riga). Chamfer eval presa dalle pair_metrics di e108, come nel summary.

| A - B | nocrop_cross [CI] (P<=0) | subject_pair_mean [CI] (P<=0) |
| --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) - BFM+ICT+GNM (10^5), e108 | -0.385 [-0.466, -0.293] (1.000) | -0.499 [-0.591, -0.411] (1.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) - Chamfer eval | -0.127 [-0.214, -0.043] (0.999) | -0.447 [-0.552, -0.350] (1.000) |
| ArcFace, normal map, 3 viste - BFM+ICT+GNM (10^5), e108 | -0.388 [-0.482, -0.286] (1.000) | -0.526 [-0.632, -0.432] (1.000) |
| ArcFace, normal map, 3 viste - Chamfer eval | -0.131 [-0.224, -0.034] (0.996) | -0.473 [-0.589, -0.365] (1.000) |
| BFM+ICT+GNM (10^5), e108 - Chamfer eval | +0.258 [+0.208, +0.301] (0.000) | +0.052 [-0.004, +0.106] (0.039) |

## (b) Riconoscimento d'identita' (protocollo di `aau/runs/arcface_render_zs/results_hifi3d.md`)

5 topologie senza crop: 2000 query di retrieval (galleria di 100 mesh in un'altra topologia; mAP = MRR), verifica su 1000 coppie stessa persona e 99000 diverse. Repliche: `bootstrap_counts` col seme `expr_recognition`, le stesse per tutte le righe e le differenze (come nel summary (b)). `chamfer` qui = Chamfer faceBench 4096 pt (la riga di results_hifi3d.md). Il congiunto e' in convenzione BFM come in (b); i checkpoint su scala nel frame nativo di (a).

| metodo | rank-1 | mAP | AUC verifica | distanze NaN |
| --- | --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) | 0.984 [0.974, 0.992] | 0.991 [0.985, 0.996] | 0.989 [0.985, 0.993] | 0 |
| ArcFace, normal map, 3 viste | 0.998 [0.995, 1.000] | 0.999 [0.998, 1.000] | 0.997 [0.996, 0.998] | 0 |
| BFM+ICT+GNM (10^5), e036 | 0.811 [0.785, 0.839] | 0.861 [0.840, 0.883] | 0.968 [0.960, 0.975] | 0 |
| BFM+ICT+GNM (10^5), e072 | 0.784 [0.758, 0.809] | 0.845 [0.825, 0.864] | 0.954 [0.943, 0.964] | 0 |
| BFM+ICT+GNM (10^5), e108 | 0.782 [0.754, 0.807] | 0.845 [0.822, 0.863] | 0.946 [0.934, 0.957] | 0 |
| Chamfer (faceBench, 4096 pt) | 0.477 [0.445, 0.513] | 0.564 [0.534, 0.594] | 0.782 [0.767, 0.797] | 0 |
| BFM+ICT, convenzione BFM | 0.397 [0.363, 0.434] | 0.517 [0.489, 0.549] | 0.755 [0.740, 0.770] | 0 |
| Rigid ICP + Chamfer | 0.996 [0.991, 0.999] | 0.997 [0.995, 0.999] | 0.999 [0.999, 1.000] | 0 |
| Rigid ICP + NICP + P2P | 0.988 [0.979, 0.996] | 0.989 [0.980, 0.997] | 0.999 [0.999, 1.000] | 1965 |
| Rigid ICP + NICP + P2Tri | 0.990 [0.982, 0.998] | 0.990 [0.982, 0.998] | 1.000 [1.000, 1.000] | 1965 |

### (b) Differenze appaiate (stesse repliche)

| A - B | rank-1 [CI] (P<=0) | mAP [CI] (P<=0) | AUC [CI] (P<=0) |
| --- | --- | --- | --- |
| ArcFace, ombreggiato, 3 viste (riferimento) - BFM+ICT+GNM (10^5), e108 | +0.201 [+0.174, +0.231] (0.000) | +0.146 [+0.126, +0.168] (0.000) | +0.044 [+0.032, +0.055] (0.000) |
| ArcFace, ombreggiato, 3 viste (riferimento) - Chamfer (faceBench, 4096 pt) | +0.506 [+0.468, +0.541] (0.000) | +0.427 [+0.395, +0.456] (0.000) | +0.207 [+0.191, +0.222] (0.000) |
| ArcFace, normal map, 3 viste - BFM+ICT+GNM (10^5), e108 | +0.216 [+0.191, +0.244] (0.000) | +0.154 [+0.136, +0.176] (0.000) | +0.052 [+0.041, +0.063] (0.000) |
| ArcFace, normal map, 3 viste - Chamfer (faceBench, 4096 pt) | +0.520 [+0.486, +0.554] (0.000) | +0.435 [+0.405, +0.465] (0.000) | +0.215 [+0.201, +0.230] (0.000) |
| BFM+ICT+GNM (10^5), e108 - Chamfer (faceBench, 4096 pt) | +0.305 [+0.278, +0.336] (0.000) | +0.280 [+0.258, +0.304] (0.000) | +0.163 [+0.153, +0.174] (0.000) |

## Breakdown sulle coppie di topologie con `noisy` (senza crop)

Per coppia NON ordinata {noisy, X}: in (a) le due coppie ordinate insieme, in (b) retrieval nei due versi (200 query) e verifica su quella coppia. `con noisy` = le 4 coppie insieme, `senza noisy` = le altre 6 (complemento nel protocollo senza crop). Repliche: in (a) un seme per gruppo, uguale per righe e differenze del gruppo; in (b) le stesse di sopra.

### (a) Spearman, GT maxabs

| coppia | ArcFace, ombreggiato, 3 viste (riferimento) | ArcFace, normal map, 3 viste | BFM+ICT+GNM (10^5), e036 | BFM+ICT+GNM (10^5), e072 | BFM+ICT+GNM (10^5), e108 | Chamfer eval | ArcFace ombr. - e108 | ArcFace ombr. - Chamfer eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy / down8k | 0.230 [0.139, 0.319] | 0.236 [0.132, 0.335] | 0.543 [0.457, 0.621] | 0.529 [0.441, 0.616] | 0.479 [0.395, 0.560] | 0.112 [0.076, 0.149] | -0.249 [-0.345, -0.150] (1.000) | +0.118 [+0.021, +0.217] (0.007) |
| noisy / original | 0.236 [0.154, 0.326] | 0.247 [0.148, 0.347] | 0.789 [0.721, 0.842] | 0.811 [0.754, 0.861] | 0.805 [0.753, 0.852] | 0.829 [0.790, 0.864] | -0.569 [-0.655, -0.475] (1.000) | -0.593 [-0.678, -0.493] (1.000) |
| noisy / remesh | 0.232 [0.149, 0.319] | 0.235 [0.147, 0.331] | 0.697 [0.631, 0.759] | 0.710 [0.642, 0.766] | 0.699 [0.636, 0.753] | 0.442 [0.393, 0.501] | -0.467 [-0.551, -0.377] (1.000) | -0.211 [-0.301, -0.122] (1.000) |
| noisy / up60k | 0.234 [0.152, 0.332] | 0.244 [0.159, 0.347] | 0.704 [0.632, 0.763] | 0.685 [0.613, 0.745] | 0.656 [0.584, 0.716] | 0.467 [0.415, 0.515] | -0.422 [-0.506, -0.327] (1.000) | -0.234 [-0.324, -0.124] (1.000) |
| con noisy | 0.233 [0.147, 0.321] | 0.241 [0.146, 0.334] | 0.679 [0.602, 0.739] | 0.666 [0.588, 0.729] | 0.629 [0.556, 0.692] | 0.358 [0.315, 0.399] | -0.396 [-0.486, -0.309] (1.000) | -0.125 [-0.215, -0.037] (0.999) |
| senza noisy | 0.277 [0.196, 0.363] | 0.253 [0.165, 0.352] | 0.676 [0.601, 0.741] | 0.662 [0.591, 0.725] | 0.631 [0.562, 0.691] | 0.399 [0.344, 0.446] | -0.353 [-0.448, -0.251] (1.000) | -0.121 [-0.219, -0.021] (0.989) |

### (b) rank-1

| coppia | ArcFace, ombreggiato, 3 viste (riferimento) | ArcFace, normal map, 3 viste | BFM+ICT+GNM (10^5), e036 | BFM+ICT+GNM (10^5), e072 | BFM+ICT+GNM (10^5), e108 | Chamfer (faceBench, 4096 pt) | ArcFace ombr. - e108 | ArcFace ombr. - Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy / down8k | 0.955 [0.925, 0.980] | 0.990 [0.975, 1.000] | 0.350 [0.285, 0.420] | 0.285 [0.220, 0.350] | 0.340 [0.280, 0.400] | 0.040 [0.015, 0.070] | +0.615 [+0.550, +0.675] (0.000) | +0.915 [+0.875, +0.950] (0.000) |
| noisy / original | 0.955 [0.925, 0.980] | 0.995 [0.985, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | -0.045 [-0.075, -0.020] (1.000) | -0.045 [-0.075, -0.020] (1.000) |
| noisy / remesh | 0.965 [0.935, 0.985] | 1.000 [1.000, 1.000] | 0.915 [0.865, 0.960] | 0.940 [0.900, 0.975] | 0.935 [0.895, 0.965] | 0.590 [0.510, 0.670] | +0.030 [-0.010, +0.075] (0.108) | +0.375 [+0.290, +0.460] (0.000) |
| noisy / up60k | 0.960 [0.935, 0.985] | 0.995 [0.985, 1.000] | 0.905 [0.860, 0.945] | 0.790 [0.735, 0.840] | 0.765 [0.710, 0.815] | 0.405 [0.335, 0.480] | +0.195 [+0.135, +0.255] (0.000) | +0.555 [+0.480, +0.630] (0.000) |
| con noisy | 0.959 [0.934, 0.980] | 0.995 [0.988, 1.000] | 0.792 [0.762, 0.821] | 0.754 [0.726, 0.780] | 0.760 [0.732, 0.787] | 0.509 [0.474, 0.545] | +0.199 [+0.162, +0.234] (0.000) | +0.450 [+0.404, +0.494] (0.000) |
| senza noisy | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.823 [0.795, 0.852] | 0.805 [0.777, 0.832] | 0.797 [0.767, 0.825] | 0.457 [0.422, 0.495] | +0.203 [+0.175, +0.233] (0.000) | +0.543 [+0.505, +0.578] (0.000) |

### (b) AUC verifica

| coppia | ArcFace, ombreggiato, 3 viste (riferimento) | ArcFace, normal map, 3 viste | BFM+ICT+GNM (10^5), e036 | BFM+ICT+GNM (10^5), e072 | BFM+ICT+GNM (10^5), e108 | Chamfer (faceBench, 4096 pt) | ArcFace ombr. - e108 | ArcFace ombr. - Chamfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy / down8k | 0.997 [0.995, 0.999] | 0.999 [0.999, 1.000] | 0.932 [0.910, 0.951] | 0.924 [0.903, 0.944] | 0.919 [0.897, 0.940] | 0.647 [0.628, 0.667] | +0.078 [+0.058, +0.101] (0.000) | +0.351 [+0.331, +0.369] (0.000) |
| noisy / original | 0.998 [0.996, 0.999] | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | -0.002 [-0.004, -0.001] (1.000) | -0.002 [-0.004, -0.001] (1.000) |
| noisy / remesh | 0.998 [0.997, 0.999] | 1.000 [0.999, 1.000] | 0.997 [0.995, 0.999] | 0.999 [0.998, 1.000] | 0.999 [0.998, 1.000] | 0.953 [0.935, 0.967] | -0.001 [-0.002, +0.001] (0.790) | +0.046 [+0.031, +0.063] (0.000) |
| noisy / up60k | 0.998 [0.996, 0.999] | 1.000 [0.999, 1.000] | 0.998 [0.997, 0.999] | 0.995 [0.992, 0.997] | 0.995 [0.991, 0.997] | 0.935 [0.916, 0.953] | +0.003 [+0.000, +0.007] (0.018) | +0.063 [+0.044, +0.081] (0.000) |
| con noisy | 0.998 [0.996, 0.999] | 1.000 [0.999, 1.000] | 0.961 [0.951, 0.969] | 0.944 [0.932, 0.956] | 0.934 [0.920, 0.948] | 0.763 [0.749, 0.776] | +0.064 [+0.050, +0.078] (0.000) | +0.235 [+0.222, +0.248] (0.000) |
| senza noisy | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.974 [0.966, 0.980] | 0.960 [0.951, 0.969] | 0.953 [0.943, 0.963] | 0.805 [0.788, 0.822] | +0.047 [+0.037, +0.057] (0.000) | +0.195 [+0.178, +0.212] (0.000) |

## Controlli

Riproduzione del summary (a) (`table_cells.csv`, `paired.csv`; max |diff| su punto e CI, e P per la differenza; per subject_pair_mean il punto si confronta con `point_check`, cioe' lo Spearman sulle pair_metrics mediate: il punto pubblicato del congiunto viene dallo script di ranking e ne differisce di ~1e-6):

| riga | pubblicato | ricalcolato | max |diff| |
| --- | --- | --- | --- |
| Chamfer eval, nocrop_cross | 0.372 [0.324, 0.422] | 0.372 [0.324, 0.422] | 5.55e-17 |
| BFM+ICT+GNM (10^5), e036, nocrop_cross | 0.677 [0.609, 0.737] | 0.677 [0.609, 0.737] | 0.00e+00 |
| BFM+ICT+GNM (10^5), e072, nocrop_cross | 0.663 [0.591, 0.725] | 0.663 [0.591, 0.725] | 0.00e+00 |
| BFM+ICT+GNM (10^5), e108, nocrop_cross | 0.630 [0.569, 0.689] | 0.630 [0.569, 0.689] | 0.00e+00 |
| e108 - Chamfer eval (appaiata), nocrop_cross | +0.258 [+0.208, +0.301] | +0.258 [+0.208, +0.301] | 5.55e-17 |
| Chamfer eval, subject_pair_mean | 0.743 [0.682, 0.796] | 0.743 [0.682, 0.796] | 0.00e+00 |
| BFM+ICT+GNM (10^5), e036, subject_pair_mean | 0.767 [0.706, 0.825] | 0.767 [0.706, 0.825] | 0.00e+00 |
| BFM+ICT+GNM (10^5), e072, subject_pair_mean | 0.792 [0.734, 0.838] | 0.792 [0.734, 0.838] | 0.00e+00 |
| BFM+ICT+GNM (10^5), e108, subject_pair_mean | 0.795 [0.738, 0.840] | 0.795 [0.738, 0.840] | 0.00e+00 |
| e108 - Chamfer eval (appaiata), subject_pair_mean | +0.052 [-0.004, +0.106] | +0.052 [-0.004, +0.106] | 8.67e-17 |

Riproduzione di results_hifi3d.md (`aau/runs/arcface_render_zs/hifi3d/recognition.csv`, blocco nocrop; max |diff| su punto e CI di rank-1, mAP, AUC):

| riga | pubblicato rank-1 | ricalcolato rank-1 | max |diff| |
| --- | --- | --- | --- |
| Rigid ICP + NICP + P2Tri | 0.990 [0.982, 0.998] | 0.990 [0.982, 0.998] | 1.11e-16 |
| BFM+ICT, convenzione BFM | 0.397 [0.363, 0.434] | 0.397 [0.363, 0.434] | 5.55e-17 |
| Rigid ICP + NICP + P2P | 0.988 [0.979, 0.996] | 0.988 [0.979, 0.996] | 1.11e-16 |
| ArcFace, ombreggiato, 3 viste (riferimento) | 0.984 [0.974, 0.992] | 0.984 [0.974, 0.992] | 1.11e-16 |
| Chamfer (faceBench, 4096 pt) | 0.477 [0.445, 0.513] | 0.477 [0.445, 0.513] | 0.00e+00 |
| Rigid ICP + Chamfer | 0.996 [0.991, 0.999] | 0.996 [0.991, 0.999] | 1.11e-16 |

Altri:

- Chamfer eval delle pair_metrics di joint contro quelle di scale_e108, max |diff|: 0.00e+00
- Chamfer eval delle pair_metrics di scale_e036 contro quelle di scale_e108, max |diff|: 0.00e+00
- Chamfer eval delle pair_metrics di scale_e072 contro quelle di scale_e108, max |diff|: 0.00e+00
- scale_e036: distanze dagli embedding contro latent_distance delle pair_metrics di (a) (148500 righe), max |diff|: 6.12e-07
- scale_e072: distanze dagli embedding contro latent_distance delle pair_metrics di (a) (148500 righe), max |diff|: 6.20e-07
- scale_e108: distanze dagli embedding contro latent_distance delle pair_metrics di (a) (148500 righe), max |diff|: 5.98e-07
- soggetti: pair_metrics, `subjects.json` degli embedding e `select_subjects` coincidono (altrimenti lo script esce)
- embedding ArcFace: tutti finiti e L2 (|norma - 1| <= 1e-3, controllo di `arcface_distances`)
