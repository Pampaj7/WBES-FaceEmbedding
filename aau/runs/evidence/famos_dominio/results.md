# FaMoS come confine di dominio: espressione, qualita' della scansione, taglia

Generato da `aau/famos_dominio/fd.py analyze` (regole in `REGOLE.md`, scritte e committate prima dei numeri). Distanze di `esterna_famos_id` (protocollo 29baaa7), nessuna nuova. Lettura DESCRITTIVA; IC 95% percentile, 1000 repliche per soggetto (seme 1234). Controllo: metriche globali uguali a `esterna_famos_id/metrics.csv` (scarto massimo 0.0e+00).

## Verdetti

- **H1 espressione**: SUPPORTATA; spiega il divario: si.
- **H2 qualita' della scansione**: NON SUPPORTATA; spiega il divario: no.
- **H3 taglia**: PARZIALE; f_ICP(AUC) = 0.53, 0.41 (s1234, s2345).

## H1 espressione (expr_mm, mm)

Terzili t1 = 0.862, t2 = 3.315; scansioni L/M/H 50/50/50; coppie della stessa persona per strato (della scansione piu espressiva) 80/225/370.

AUC_k: stessa persona dello strato k contro tutte le coppie di persone diverse. rank-1_k: query dello strato, galleria completa.

| metodo | AUC L | AUC M | AUC H | rank-1 L | rank-1 M | rank-1 H |
|---|---|---|---|---|---|---|
| nostro s1234, d_F non cal. | 1.000 [1.000, 1.000] | 0.962 [0.922, 0.990] | 0.713 [0.636, 0.796] | 1.000 [1.000, 1.000] | 0.920 [0.854, 0.981] | 0.640 [0.465, 0.765] |
| nostro s1234, d_F cal. | 1.000 [1.000, 1.000] | 0.946 [0.899, 0.982] | 0.697 [0.594, 0.788] | 1.000 [1.000, 1.000] | 0.920 [0.849, 0.980] | 0.700 [0.537, 0.814] |
| nostro s1234, d_P | 1.000 [1.000, 1.000] | 0.957 [0.913, 0.988] | 0.716 [0.639, 0.797] | 1.000 [1.000, 1.000] | 0.900 [0.795, 0.980] | 0.540 [0.382, 0.667] |
| nostro s2345, d_F non cal. | 1.000 [1.000, 1.000] | 0.958 [0.918, 0.990] | 0.681 [0.596, 0.771] | 1.000 [1.000, 1.000] | 0.920 [0.850, 0.981] | 0.540 [0.333, 0.719] |
| nostro s2345, d_F cal. | 1.000 [1.000, 1.000] | 0.943 [0.892, 0.985] | 0.678 [0.563, 0.777] | 1.000 [1.000, 1.000] | 0.900 [0.822, 0.965] | 0.660 [0.488, 0.793] |
| nostro s2345, d_P | 1.000 [1.000, 1.000] | 0.951 [0.910, 0.984] | 0.679 [0.596, 0.768] | 1.000 [1.000, 1.000] | 0.900 [0.833, 0.962] | 0.500 [0.341, 0.625] |
| ICP + Chamfer in mm | 1.000 [1.000, 1.000] | 0.980 [0.952, 0.999] | 0.820 [0.752, 0.885] | 1.000 [1.000, 1.000] | 0.980 [0.940, 1.000] | 0.860 [0.744, 0.952] |
| ICP + Chamfer cs | 1.000 [1.000, 1.000] | 0.979 [0.953, 0.998] | 0.758 [0.684, 0.831] | 1.000 [1.000, 1.000] | 0.980 [0.943, 1.000] | 0.760 [0.673, 0.852] |
| NoW metrical (rilevatore) | 1.000 [1.000, 1.000] | 0.997 [0.994, 1.000] | 0.852 [0.778, 0.919] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.900 [0.818, 0.977] |
| NoW originale (rilevatore) | 1.000 [1.000, 1.000] | 0.995 [0.989, 0.999] | 0.781 [0.711, 0.857] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.820 [0.739, 0.900] |
| ULRSSM faces_xyz per coppia, mm | 1.000 [1.000, 1.000] | 0.962 [0.912, 0.994] | 0.784 [0.678, 0.871] | 1.000 [1.000, 1.000] | 0.960 [0.880, 1.000] | 0.820 [0.667, 0.944] |
| ULRSSM faces_xyz per coppia, cs | 1.000 [0.999, 1.000] | 0.960 [0.924, 0.991] | 0.671 [0.584, 0.758] | 1.000 [1.000, 1.000] | 0.960 [0.917, 1.000] | 0.620 [0.447, 0.750] |
| GW euclid_128_cgw2, mm | 0.985 [0.974, 0.995] | 0.924 [0.866, 0.974] | 0.816 [0.732, 0.883] | 0.940 [0.872, 1.000] | 0.800 [0.692, 0.900] | 0.720 [0.528, 0.868] |
| GW euclid_128_cgw2, cs | 0.979 [0.958, 0.993] | 0.912 [0.869, 0.954] | 0.713 [0.624, 0.795] | 0.900 [0.794, 0.980] | 0.680 [0.523, 0.854] | 0.540 [0.365, 0.694] |
| taglia oracolo, abs(CS_i - CS_j) | 0.956 [0.930, 0.975] | 0.843 [0.764, 0.911] | 0.779 [0.658, 0.860] | 0.260 [0.116, 0.408] | 0.280 [0.159, 0.390] | 0.280 [0.098, 0.455] |

Delta nostro - concorrente per strato (positivo = il nostro meglio) e interazione Delta_H - Delta_L.

| nostro - concorrente | dAUC L | dAUC M | dAUC H | interazione AUC | interazione rank-1 | quota del divario AUC L/M/H |
|---|---|---|---|---|---|---|
| nostro s1234, d_F non cal. - ICP + Chamfer in mm | +0.000 [+0.000, +0.000] | -0.019 [-0.067, +0.018] | -0.108 [-0.167, -0.052] | -0.108 [-0.167, -0.052] | -0.220 [-0.372, -0.091] | -0.00/0.09/0.91 |
| nostro s1234, d_F non cal. - ICP + Chamfer cs | +0.000 [+0.000, +0.000] | -0.017 [-0.059, +0.014] | -0.046 [-0.100, +0.005] | -0.046 [-0.100, +0.005] | -0.120 [-0.306, +0.017] | - |
| nostro s1234, d_F non cal. - NoW metrical (rilevatore) | +0.000 [+0.000, +0.000] | -0.036 [-0.073, -0.009] | -0.139 [-0.192, -0.096] | -0.139 [-0.192, -0.096] | -0.260 [-0.400, -0.156] | -0.00/0.13/0.87 |
| nostro s1234, d_F non cal. - NoW originale (rilevatore) | +0.000 [+0.000, +0.000] | -0.034 [-0.070, -0.008] | -0.068 [-0.121, -0.023] | -0.068 [-0.121, -0.023] | -0.180 [-0.324, -0.062] | - |
| nostro s1234, d_F non cal. - ULRSSM faces_xyz per coppia, mm | +0.000 [+0.000, +0.000] | -0.000 [-0.051, +0.051] | -0.071 [-0.139, -0.011] | -0.071 [-0.139, -0.011] | -0.180 [-0.333, -0.016] | - |
| nostro s1234, d_F non cal. - ULRSSM faces_xyz per coppia, cs | +0.000 [+0.000, +0.001] | +0.002 [-0.039, +0.040] | +0.041 [-0.003, +0.089] | +0.041 [-0.003, +0.088] | +0.020 [-0.116, +0.180] | - |
| nostro s1234, d_F non cal. - GW euclid_128_cgw2, mm | +0.015 [+0.005, +0.026] | +0.037 [-0.019, +0.099] | -0.103 [-0.185, -0.030] | -0.119 [-0.200, -0.045] | -0.140 [-0.261, -0.021] | - |
| nostro s1234, d_F non cal. - GW euclid_128_cgw2, cs | +0.021 [+0.007, +0.042] | +0.050 [+0.016, +0.084] | -0.000 [-0.050, +0.052] | -0.021 [-0.068, +0.030] | +0.000 [-0.166, +0.171] | - |
| nostro s2345, d_F non cal. - ICP + Chamfer in mm | +0.000 [+0.000, +0.000] | -0.023 [-0.065, +0.006] | -0.139 [-0.208, -0.074] | -0.139 [-0.208, -0.074] | -0.320 [-0.500, -0.186] | -0.00/0.09/0.91 |
| nostro s2345, d_F non cal. - ICP + Chamfer cs | +0.000 [+0.000, +0.000] | -0.021 [-0.060, +0.003] | -0.077 [-0.143, -0.020] | -0.077 [-0.143, -0.020] | -0.220 [-0.436, -0.029] | - |
| nostro s2345, d_F non cal. - NoW metrical (rilevatore) | +0.000 [+0.000, +0.000] | -0.040 [-0.077, -0.010] | -0.171 [-0.229, -0.124] | -0.171 [-0.229, -0.124] | -0.360 [-0.535, -0.216] | -0.00/0.12/0.88 |
| nostro s2345, d_F non cal. - NoW originale (rilevatore) | +0.000 [+0.000, +0.000] | -0.038 [-0.073, -0.009] | -0.100 [-0.159, -0.051] | -0.100 [-0.159, -0.051] | -0.280 [-0.469, -0.113] | - |
| nostro s2345, d_F non cal. - ULRSSM faces_xyz per coppia, mm | +0.000 [+0.000, +0.000] | -0.004 [-0.051, +0.037] | -0.103 [-0.167, -0.042] | -0.103 [-0.167, -0.042] | -0.280 [-0.465, -0.130] | - |
| nostro s2345, d_F non cal. - ULRSSM faces_xyz per coppia, cs | +0.000 [+0.000, +0.001] | -0.002 [-0.038, +0.028] | +0.010 [-0.041, +0.055] | +0.010 [-0.041, +0.055] | -0.080 [-0.245, +0.064] | - |
| nostro s2345, d_F non cal. - GW euclid_128_cgw2, mm | +0.015 [+0.005, +0.026] | +0.033 [-0.020, +0.087] | -0.135 [-0.223, -0.059] | -0.150 [-0.238, -0.073] | -0.240 [-0.452, -0.054] | - |
| nostro s2345, d_F non cal. - GW euclid_128_cgw2, cs | +0.021 [+0.007, +0.042] | +0.046 [+0.018, +0.072] | -0.032 [-0.093, +0.028] | -0.053 [-0.113, +0.008] | -0.100 [-0.316, +0.139] | - |

### Sottoinsieme quasi neutro (solo L: 50 scansioni, 13 soggetti con almeno una coppia, 80 coppie della stessa persona)

| metodo | rank-1 | AUC | d rank-1 (s1234 - metodo) | d AUC (s1234 - metodo) |
|---|---|---|---|---|
| nostro s1234, d_F non cal. | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | - | - |
| nostro s1234, d_F cal. | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | - | - |
| nostro s1234, d_P | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | - | - |
| nostro s2345, d_F non cal. | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | - | - |
| nostro s2345, d_F cal. | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | - | - |
| nostro s2345, d_P | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | - | - |
| ICP + Chamfer in mm | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| ICP + Chamfer cs | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| NoW metrical (rilevatore) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| NoW originale (rilevatore) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| ULRSSM faces_xyz per coppia, mm | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.001] |
| ULRSSM faces_xyz per coppia, cs | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.001] |
| GW euclid_128_cgw2, mm | 0.898 [0.783, 0.980] | 0.981 [0.964, 0.996] | +0.102 [+0.020, +0.217] | +0.019 [+0.004, +0.036] |
| GW euclid_128_cgw2, cs | 0.816 [0.641, 0.940] | 0.957 [0.918, 0.986] | +0.184 [+0.060, +0.359] | +0.043 [+0.014, +0.082] |
| taglia oracolo, abs(CS_i - CS_j) | 0.469 [0.295, 0.650] | 0.948 [0.877, 0.988] | +0.531 [+0.350, +0.705] | +0.052 [+0.012, +0.123] |

Secondaria (AUC_k con le sole coppie di persone diverse dello stesso strato), per le letture del verdetto:

| metodo | L | M | H |
|---|---|---|---|
| nostro s1234, d_F non cal. | 1.000 [1.000, 1.000] | 0.943 [0.892, 0.982] | 0.793 [0.730, 0.850] |
| nostro s2345, d_F non cal. | 1.000 [1.000, 1.000] | 0.935 [0.887, 0.980] | 0.771 [0.704, 0.830] |
| ICP + Chamfer in mm | 1.000 [1.000, 1.000] | 0.971 [0.934, 0.998] | 0.881 [0.825, 0.924] |
| NoW metrical (rilevatore) | 1.000 [1.000, 1.000] | 0.995 [0.989, 1.000] | 0.907 [0.852, 0.949] |

## H2 irregolarita' del bordo della patch (q)

Terzili t1 = 1.045, t2 = 1.150; scansioni L/M/H 50/50/50; coppie della stessa persona per strato (della scansione peggiore) 102/228/345.

AUC_k: stessa persona dello strato k contro tutte le coppie di persone diverse. rank-1_k: query dello strato, galleria completa.

| metodo | AUC L | AUC M | AUC H | rank-1 L | rank-1 M | rank-1 H |
|---|---|---|---|---|---|---|
| nostro s1234, d_F non cal. | 0.921 [0.837, 0.976] | 0.898 [0.831, 0.953] | 0.757 [0.689, 0.833] | 0.840 [0.755, 0.919] | 0.880 [0.767, 0.963] | 0.840 [0.741, 0.917] |
| nostro s1234, d_F cal. | 0.905 [0.809, 0.969] | 0.881 [0.808, 0.943] | 0.746 [0.677, 0.818] | 0.900 [0.811, 0.964] | 0.900 [0.778, 0.981] | 0.820 [0.723, 0.900] |
| nostro s1234, d_P | 0.916 [0.832, 0.968] | 0.891 [0.828, 0.945] | 0.764 [0.690, 0.845] | 0.800 [0.710, 0.887] | 0.840 [0.714, 0.941] | 0.800 [0.711, 0.879] |
| nostro s2345, d_F non cal. | 0.915 [0.814, 0.979] | 0.886 [0.817, 0.945] | 0.731 [0.663, 0.808] | 0.840 [0.735, 0.939] | 0.860 [0.738, 0.961] | 0.760 [0.690, 0.861] |
| nostro s2345, d_F cal. | 0.899 [0.787, 0.970] | 0.873 [0.795, 0.937] | 0.732 [0.665, 0.807] | 0.920 [0.822, 0.982] | 0.880 [0.789, 0.957] | 0.760 [0.679, 0.851] |
| nostro s2345, d_P | 0.908 [0.801, 0.972] | 0.876 [0.810, 0.934] | 0.733 [0.660, 0.815] | 0.820 [0.721, 0.918] | 0.800 [0.667, 0.909] | 0.780 [0.681, 0.868] |
| ICP + Chamfer in mm | 0.976 [0.952, 0.996] | 0.938 [0.871, 0.981] | 0.843 [0.788, 0.899] | 1.000 [1.000, 1.000] | 0.960 [0.898, 1.000] | 0.880 [0.812, 0.957] |
| ICP + Chamfer cs | 0.970 [0.936, 0.991] | 0.922 [0.863, 0.968] | 0.788 [0.719, 0.858] | 0.960 [0.889, 1.000] | 0.940 [0.870, 1.000] | 0.840 [0.763, 0.912] |
| NoW metrical (rilevatore) | 0.983 [0.960, 0.997] | 0.963 [0.920, 0.990] | 0.869 [0.811, 0.926] | 1.000 [1.000, 1.000] | 0.960 [0.898, 1.000] | 0.940 [0.884, 1.000] |
| NoW originale (rilevatore) | 0.968 [0.926, 0.994] | 0.941 [0.897, 0.977] | 0.810 [0.745, 0.878] | 0.960 [0.882, 1.000] | 0.940 [0.870, 1.000] | 0.920 [0.857, 0.977] |
| ULRSSM faces_xyz per coppia, mm | 0.967 [0.933, 0.991] | 0.920 [0.841, 0.968] | 0.806 [0.726, 0.882] | 1.000 [1.000, 1.000] | 0.940 [0.875, 1.000] | 0.840 [0.711, 0.968] |
| ULRSSM faces_xyz per coppia, cs | 0.944 [0.898, 0.980] | 0.893 [0.826, 0.949] | 0.709 [0.627, 0.791] | 0.900 [0.824, 0.966] | 0.920 [0.804, 1.000] | 0.760 [0.630, 0.868] |
| GW euclid_128_cgw2, mm | 0.922 [0.872, 0.961] | 0.883 [0.799, 0.938] | 0.850 [0.790, 0.909] | 0.860 [0.745, 0.946] | 0.800 [0.651, 0.922] | 0.800 [0.674, 0.895] |
| GW euclid_128_cgw2, cs | 0.890 [0.833, 0.939] | 0.845 [0.768, 0.907] | 0.765 [0.690, 0.836] | 0.700 [0.480, 0.868] | 0.700 [0.580, 0.814] | 0.720 [0.577, 0.821] |
| taglia oracolo, abs(CS_i - CS_j) | 0.869 [0.794, 0.920] | 0.812 [0.709, 0.889] | 0.813 [0.717, 0.886] | 0.220 [0.068, 0.410] | 0.420 [0.279, 0.579] | 0.180 [0.081, 0.289] |

Delta nostro - concorrente per strato (positivo = il nostro meglio) e interazione Delta_H - Delta_L.

| nostro - concorrente | dAUC L | dAUC M | dAUC H | interazione AUC | interazione rank-1 | quota del divario AUC L/M/H |
|---|---|---|---|---|---|---|
| nostro s1234, d_F non cal. - ICP + Chamfer in mm | -0.055 [-0.131, -0.006] | -0.039 [-0.083, +0.002] | -0.085 [-0.146, -0.032] | -0.030 [-0.089, +0.038] | +0.120 [-0.029, +0.233] | 0.13/0.20/0.67 |
| nostro s1234, d_F non cal. - ICP + Chamfer cs | -0.049 [-0.111, -0.002] | -0.024 [-0.063, +0.011] | -0.030 [-0.076, +0.014] | +0.019 [-0.037, +0.081] | +0.120 [-0.019, +0.231] | - |
| nostro s1234, d_F non cal. - NoW metrical (rilevatore) | -0.061 [-0.132, -0.019] | -0.065 [-0.102, -0.030] | -0.112 [-0.163, -0.068] | -0.050 [-0.104, +0.022] | +0.060 [-0.088, +0.192] | 0.11/0.25/0.65 |
| nostro s1234, d_F non cal. - NoW originale (rilevatore) | -0.047 [-0.110, -0.008] | -0.043 [-0.080, -0.013] | -0.053 [-0.092, -0.016] | -0.006 [-0.053, +0.055] | +0.040 [-0.097, +0.155] | - |
| nostro s1234, d_F non cal. - ULRSSM faces_xyz per coppia, mm | -0.046 [-0.114, +0.001] | -0.022 [-0.063, +0.016] | -0.048 [-0.112, +0.018] | -0.003 [-0.064, +0.073] | +0.160 [-0.023, +0.311] | - |
| nostro s1234, d_F non cal. - ULRSSM faces_xyz per coppia, cs | -0.023 [-0.078, +0.021] | +0.006 [-0.030, +0.040] | +0.048 [+0.008, +0.093] | +0.071 [+0.014, +0.132] | +0.140 [-0.002, +0.282] | - |
| nostro s1234, d_F non cal. - GW euclid_128_cgw2, mm | -0.001 [-0.077, +0.057] | +0.015 [-0.038, +0.071] | -0.093 [-0.176, -0.020] | -0.092 [-0.170, -0.017] | +0.060 [-0.138, +0.221] | - |
| nostro s1234, d_F non cal. - GW euclid_128_cgw2, cs | +0.031 [-0.018, +0.069] | +0.053 [-0.001, +0.109] | -0.007 [-0.052, +0.046] | -0.038 [-0.094, +0.025] | -0.020 [-0.321, +0.191] | - |
| nostro s2345, d_F non cal. - ICP + Chamfer in mm | -0.062 [-0.156, -0.006] | -0.052 [-0.099, -0.006] | -0.112 [-0.172, -0.058] | -0.050 [-0.119, +0.033] | +0.040 [-0.093, +0.162] | 0.11/0.21/0.68 |
| nostro s2345, d_F non cal. - ICP + Chamfer cs | -0.055 [-0.134, -0.001] | -0.036 [-0.081, +0.001] | -0.056 [-0.106, -0.014] | -0.001 [-0.064, +0.072] | +0.040 [-0.095, +0.172] | - |
| nostro s2345, d_F non cal. - NoW metrical (rilevatore) | -0.068 [-0.156, -0.016] | -0.077 [-0.118, -0.039] | -0.138 [-0.190, -0.094] | -0.070 [-0.141, +0.016] | -0.020 [-0.159, +0.116] | 0.10/0.24/0.66 |
| nostro s2345, d_F non cal. - NoW originale (rilevatore) | -0.053 [-0.133, -0.008] | -0.055 [-0.097, -0.022] | -0.079 [-0.120, -0.044] | -0.026 [-0.082, +0.049] | -0.040 [-0.164, +0.084] | - |
| nostro s2345, d_F non cal. - ULRSSM faces_xyz per coppia, mm | -0.052 [-0.135, -0.000] | -0.034 [-0.078, +0.008] | -0.075 [-0.130, -0.020] | -0.023 [-0.089, +0.065] | +0.080 [-0.086, +0.241] | - |
| nostro s2345, d_F non cal. - ULRSSM faces_xyz per coppia, cs | -0.029 [-0.101, +0.019] | -0.007 [-0.047, +0.031] | +0.022 [-0.014, +0.060] | +0.051 [-0.008, +0.122] | +0.060 [-0.052, +0.194] | - |
| nostro s2345, d_F non cal. - GW euclid_128_cgw2, mm | -0.007 [-0.099, +0.052] | +0.003 [-0.052, +0.063] | -0.119 [-0.199, -0.052] | -0.112 [-0.188, -0.029] | -0.020 [-0.216, +0.181] | - |
| nostro s2345, d_F non cal. - GW euclid_128_cgw2, cs | +0.025 [-0.035, +0.066] | +0.040 [-0.017, +0.101] | -0.034 [-0.078, +0.024] | -0.058 [-0.111, +0.014] | -0.100 [-0.368, +0.138] | - |

Secondaria (AUC_k con le sole coppie di persone diverse dello stesso strato), per le letture del verdetto:

| metodo | L | M | H |
|---|---|---|---|
| nostro s1234, d_F non cal. | 0.904 [0.823, 0.970] | 0.870 [0.795, 0.936] | 0.790 [0.732, 0.856] |
| nostro s2345, d_F non cal. | 0.898 [0.799, 0.973] | 0.854 [0.777, 0.926] | 0.769 [0.707, 0.840] |
| ICP + Chamfer in mm | 0.974 [0.955, 0.995] | 0.921 [0.847, 0.975] | 0.863 [0.813, 0.915] |
| NoW metrical (rilevatore) | 0.981 [0.964, 0.999] | 0.952 [0.900, 0.987] | 0.886 [0.830, 0.939] |

## H3 taglia

Predittore a sola taglia (|CS_i - CS_j|, CS della patch in mm):

| rank-1 | mAP | AUC | EER |
|---|---|---|---|
| 0.273 [0.173, 0.387] | 0.312 [0.247, 0.393] | 0.821 [0.742, 0.884] | 0.257 [0.191, 0.324] |

Quota del vantaggio dovuta alla taglia, f = (con taglia - senza) / (concorrente con taglia - nostro):

| nostro | famiglia | metrica | grandezza | valore [IC] |
|---|---|---|---|---|
| nostro s1234, d_F non cal. | ICP | AUC | numeratore (con taglia - senza) | +0.035 [+0.009, +0.066] |
| nostro s1234, d_F non cal. | ICP | AUC | denominatore (concorrente - nostro) | +0.065 [+0.025, +0.110] |
| nostro s1234, d_F non cal. | ICP | AUC | f = quota della taglia | +0.529 [+0.177, +1.089] |
| nostro s1234, d_F non cal. | ICP | rank1 | numeratore (con taglia - senza) | +0.033 [+0.000, +0.073] |
| nostro s1234, d_F non cal. | ICP | rank1 | denominatore (concorrente - nostro) | +0.093 [+0.047, +0.133] |
| nostro s1234, d_F non cal. | ICP | rank1 | f = quota della taglia | +0.357 [+0.000, +0.889] |
| nostro s1234, d_F non cal. | NoW | AUC | numeratore (con taglia - senza) | +0.040 [+0.019, +0.062] |
| nostro s1234, d_F non cal. | NoW | AUC | denominatore (concorrente - nostro) | +0.088 [+0.056, +0.123] |
| nostro s1234, d_F non cal. | NoW | AUC | f = quota della taglia | +0.449 [+0.230, +0.712] |
| nostro s1234, d_F non cal. | NoW | rank1 | numeratore (con taglia - senza) | +0.027 [+0.000, +0.060] |
| nostro s1234, d_F non cal. | NoW | rank1 | denominatore (concorrente - nostro) | +0.113 [+0.073, +0.147] |
| nostro s1234, d_F non cal. | NoW | rank1 | f = quota della taglia | +0.235 [+0.000, +0.538] |
| nostro s2345, d_F non cal. | ICP | AUC | numeratore (con taglia - senza) | +0.035 [+0.009, +0.066] |
| nostro s2345, d_F non cal. | ICP | AUC | denominatore (concorrente - nostro) | +0.084 [+0.040, +0.131] |
| nostro s2345, d_F non cal. | ICP | AUC | f = quota della taglia | +0.412 [+0.136, +0.761] |
| nostro s2345, d_F non cal. | ICP | rank1 | numeratore (con taglia - senza) | +0.033 [+0.000, +0.073] |
| nostro s2345, d_F non cal. | ICP | rank1 | denominatore (concorrente - nostro) | +0.127 [+0.080, +0.173] |
| nostro s2345, d_F non cal. | ICP | rank1 | f = quota della taglia | +0.263 [+0.000, +0.706] |
| nostro s2345, d_F non cal. | NoW | AUC | numeratore (con taglia - senza) | +0.040 [+0.019, +0.062] |
| nostro s2345, d_F non cal. | NoW | AUC | denominatore (concorrente - nostro) | +0.107 [+0.073, +0.142] |
| nostro s2345, d_F non cal. | NoW | AUC | f = quota della taglia | +0.371 [+0.188, +0.581] |
| nostro s2345, d_F non cal. | NoW | rank1 | numeratore (con taglia - senza) | +0.027 [+0.000, +0.060] |
| nostro s2345, d_F non cal. | NoW | rank1 | denominatore (concorrente - nostro) | +0.147 [+0.100, +0.193] |
| nostro s2345, d_F non cal. | NoW | rank1 | f = quota della taglia | +0.182 [+0.000, +0.445] |

## H2 (e H1 continua): Spearman fra statistica per scansione ed eccesso d'errore

x_i = c_i(nostro) - c_i(concorrente), c_i = quota media di coppie di persone diverse piu' vicine delle coppie della stessa persona che contengono i; rho > 0 = la statistica cresce dove il nostro sbaglia di piu'.

| statistica | mediana [p5, p95] | nostro s1234 - ICP + Chamfer in mm | nostro s1234 - NoW metrical (rilevatore) | nostro s2345 - ICP + Chamfer in mm | nostro s2345 - NoW metrical (rilevatore) |
|---|---|---|---|---|---|
| expr_mm | 2.004 [0.3489, 6.433] | +0.38 [+0.23, +0.52] | +0.47 [+0.30, +0.59] | +0.39 [+0.24, +0.55] | +0.48 [+0.34, +0.61] |
| raw_n_vertices | 1.083e+05 [9.459e+04, 1.265e+05] | +0.19 [-0.19, +0.51] | +0.43 [+0.01, +0.70] | +0.19 [-0.18, +0.50] | +0.40 [-0.01, +0.67] |
| raw_n_faces | 2.154e+05 [1.882e+05, 2.51e+05] | +0.19 [-0.19, +0.51] | +0.43 [+0.01, +0.70] | +0.19 [-0.18, +0.50] | +0.40 [+0.00, +0.67] |
| raw_area_mm2 | 1.668e+05 [1.504e+05, 1.864e+05] | +0.25 [-0.12, +0.54] | +0.45 [+0.04, +0.70] | +0.29 [-0.05, +0.57] | +0.45 [+0.11, +0.70] |
| raw_density_per_cm2 | 64.17 [62.66, 67.82] | -0.00 [-0.34, +0.35] | +0.23 [-0.14, +0.52] | -0.06 [-0.43, +0.33] | +0.14 [-0.28, +0.50] |
| raw_components | 3 [1, 11] | +0.18 [-0.02, +0.40] | +0.19 [-0.11, +0.47] | +0.15 [-0.07, +0.42] | +0.18 [-0.11, +0.47] |
| raw_boundary_mm | 2902 [2001, 4721] | -0.01 [-0.28, +0.30] | -0.01 [-0.37, +0.36] | +0.01 [-0.28, +0.36] | +0.02 [-0.34, +0.40] |
| raw_boundary_loops | 17 [11, 27] | +0.11 [-0.12, +0.31] | +0.12 [-0.15, +0.35] | +0.13 [-0.11, +0.38] | +0.17 [-0.08, +0.40] |
| patch_area_mm2 | 3.526e+04 [2.966e+04, 4.174e+04] | +0.23 [-0.10, +0.52] | +0.40 [-0.01, +0.66] | +0.21 [-0.15, +0.54] | +0.37 [-0.07, +0.66] |
| patch_components | 1 [1, 1] | +nan [+nan, +nan] | +nan [+nan, +nan] | +nan [+nan, +nan] | +nan [+nan, +nan] |
| patch_boundary_loops | 8 [3.45, 14] | +0.21 [-0.02, +0.37] | +0.22 [-0.04, +0.43] | +0.29 [+0.05, +0.45] | +0.30 [+0.03, +0.50] |
| patch_boundary_mm | 729.6 [631, 964.5] | +0.29 [+0.06, +0.48] | +0.37 [+0.12, +0.56] | +0.38 [+0.13, +0.58] | +0.46 [+0.21, +0.65] |
| patch_q | 1.083 [0.9738, 1.441] | +0.16 [-0.02, +0.34] | +0.15 [-0.04, +0.30] | +0.26 [+0.04, +0.46] | +0.25 [+0.05, +0.41] |

File: `strata.csv` (metriche, delta, interazioni e scomposizione per strato), `size.csv`, `quality.csv`, `summary.json`, figura `fig_famos_expr.pdf` (`fig_famos_expr.py`).

<!-- da qui in giu': scritto a mano, conservato da fd.py analyze -->
## Lettura (descrittiva, scritta dopo i numeri)

- **H1 espressione: SUPPORTATA secondo la regola.** Interazione Delta_H - Delta_L in AUC, d_F non cal.: s1234 contro
  ICP mm -0,108 [-0,167, -0,052], contro NoW metrical -0,139 [-0,192, -0,096]; s2345 -0,139 [-0,208, -0,074] e
  -0,171 [-0,229, -0,124]. L'87-91% del divario di AUC viene dalle coppie con la scansione piu' espressiva in H (370 delle
  675 coppie della stessa persona). In H l'AUC e' 0,713 (s1234) / 0,681 (s2345), contro 0,820 di ICP mm e 0,852 di NoW
  metrical; in M il divario e' piccolo (s1234: -0,019 [-0,067, +0,018] contro ICP mm, -0,036 [-0,073, -0,009] contro
  NoW metrical).
- **"Spiega il divario: si" va letto come effetto tetto.** In L e nel quasi neutro tutti i metodi, tranne GW e la taglia
  oracolo, hanno AUC e rank-1 = 1,000. Il delta nullo non dice che siamo pari: dice che li' il compito non distingue i
  metodi. La lettura corretta e' "il divario compare solo con espressione marcata", non "a espressione neutra siamo pari".
- **La sensibilita' non e' solo nostra.** Un'interazione cosi' la darebbe anche un metodo semplicemente piu' debole,
  sotto un tetto. In H i metodi senza taglia degradano come il nostro: ICP cs 0,758, ULRSSM cs 0,671, GW cs 0,713, NoW
  originale 0,781. In H il delta del nostro (s1234) e' +0,041 [-0,003, +0,089] contro ULRSSM cs e -0,000 [-0,050, +0,052]
  contro GW cs, ma resta sotto ICP cs: -0,046 [-0,100, +0,005] (s2345 -0,077 [-0,143, -0,020]). La d_F, che ha la taglia,
  non aiuta in H: 0,713 contro 0,716 della d_P (s1234).
- **H3 taglia: PARZIALE.** ICP mm - ICP cs in AUC +0,035 [+0,009, +0,066]; quota del vantaggio di ICP mm dovuta alla
  taglia f = 0,53 [0,18, 1,09] (s1234) e 0,41 [0,14, 0,76] (s2345); per NoW metrical contro originale 0,45 [0,23, 0,71] e
  0,37 [0,19, 0,58]. In rank-1 f e' fra 0,18 e 0,36, con IC da 0. La taglia oracolo da sola arriva ad AUC 0,821
  [0,742, 0,884], come il nostro (0,830), con rank-1 0,273. In H sta sopra il nostro (0,779 contro 0,713): anche lei
  scende con l'espressione (0,956 in L), ma meno della forma.
- **H2 qualita': NON SUPPORTATA.** Interazioni sui terzili dell'irregolarita' q con IC che include 0 in tutte e 4
  (s1234: -0,030 [-0,089, +0,038] e -0,050 [-0,104, +0,022]; s2345: -0,050 [-0,119, +0,033] e -0,070 [-0,141, +0,016]).
  Anche nel terzile migliore il nostro resta sotto: da -0,055 [-0,131, -0,006] a -0,068 [-0,156, -0,016]. Spearman
  descrittivi: l'eccesso d'errore cresce con la lunghezza del bordo della patch (da +0,29 a +0,46) quasi quanto con
  expr_mm (da +0,38 a +0,48). Ma, **post hoc** (non in REGOLE; srun 1080071, solo il punto), rho(expr_mm, bordo della
  patch) = +0,34 e rho(expr_mm, q) = +0,36: la bocca aperta apre un buco nella patch, quindi il debole segnale di
  qualita' e' in parte espressione. La densita' delle grezze (mediana 64 vertici/cm^2, p5-p95 62,7-67,8) non correla
  con l'eccesso d'errore.
- **Sintesi per il paper (confine di dominio).** Il divario su FaMoS sta nelle coppie della stessa persona con
  espressione marcata. Li' la forma senza taglia perde l'identita' per tutti i metodi di forma, il nostro compreso, e i
  metodi in mm reggono meglio anche grazie alla taglia, che conta per circa il 40-50% del loro vantaggio in AUC.
  Su Ava-256 le mesh sono di persone diverse e non ci sono coppie della stessa persona con espressioni diverse: questo
  confine non entra in gioco. Limiti: e' una stratificazione osservazionale su 15 persone, non un intervento. L'effetto
  tetto in L impedisce di dire che a espressione neutra siamo pari. Nessuna ipotesi spiega perche' in H restiamo sotto
  anche ICP cs (in s2345 con IC sotto 0).

## Esecuzione e deviazioni

- Regole `8c1cfd8`, codice `219e8f3` e `3bb6adf`. Collaudo su dati casuali: job 1080035 (uscite cancellate). Analisi:
  job 1080049 (a256-t4-02, prioritized, 3 min 21 s: stats su 150/150 scansioni, analyze, figura). Le metriche globali
  ricalcolate coincidono con `esterna_famos_id/metrics.csv` (scarto 0).
- Dopo le regole e' cambiata solo la figura: l'etichetta comune dell'asse x, su due righe (era piu' larga dei pannelli).
  Nessun numero cambia. Rigenerata con lo stesso script.
- Post hoc, fuori dalle regole: le correlazioni expr_mm - statistiche della patch (sopra), solo come lettura di H2.
