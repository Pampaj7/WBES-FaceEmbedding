# WS3a Multiface: AUC same-vs-different per metrica e topologia

AUC = P(distanza same < distanza different), 0.5 = caso. CI 95% bootstrap su 1000 repliche, ricampionando i 13 soggetti. "CI degenere" = tutte le repliche danno lo stesso valore (separazione perfetta in ognuna): l'intervallo non e' stretto, e' non informativo.

La riga `bbox_proxy` non e' una metrica di forma ma un CONTROLLO: distanza fra due vettori di 4 numeri (centro e diagonale del bounding box) presi sulla mesh **gia' normalizzata maxabs**, cioe' la stessa mesh che vedono chamfer, il latent e i render. Va letta per prima: dove il proxy e' alto quanto una metrica, quella cella e' risolta da informazione banale.

> **Quanto vale il metro.** Quattro numeri sulla mesh normalizzata, per coppia di topologie:
> - `tracked->tracked`: proxy 0.956 sul confronto difficile (b_vs_c); 0/9 metriche non lo battono
> - `tracked->crop`: proxy 0.416 sul confronto difficile (b_vs_c); 0/9 metriche non lo battono
> - `remesh->crop`: proxy 0.416 sul confronto difficile (b_vs_c); 0/9 metriche non lo battono
> - `tracked->noisy`: proxy 0.951 sul confronto difficile (b_vs_c); 3/9 metriche non lo battono (arcface, clip, dinov2)
> - `down->up`: proxy 0.900 sul confronto difficile (b_vs_c); 2/9 metriche non lo battono (currents, varifold)
> - `crop->crop`: proxy 0.953 sul confronto difficile (b_vs_c); 0/9 metriche non lo battono

Classi: (a) stesso soggetto stessa espressione, (b) stesso soggetto espressione diversa, (c) soggetti diversi stessa espressione, (d) tutto diverso.

Coppie di topologie: hard. L'ultima colonna e' AUC(b_vs_c) meno la stessa AUC della stessa metrica su tracked->tracked (il caso pulito): negativa = la perturbazione fa perdere separazione.

| metrica | topologie | ab_vs_cd | b_vs_c | a_vs_c | delta b_vs_c vs clean |
|---|---|---|---|---|---|
| chamfer | tracked->tracked | 0.997 [0.992, 0.999] | 0.993 [0.983, 0.999] | 0.999 [0.998, 1.000] | +0.000 |
| chamfer | tracked->crop | 0.647 [0.390, 0.858] | 0.664 [0.383, 0.885] | 0.681 [0.384, 0.886] | -0.329 |
| chamfer | remesh->crop | 0.579 [0.335, 0.792] | 0.601 [0.330, 0.828] | 0.616 [0.346, 0.835] | -0.393 |
| chamfer | tracked->noisy | 0.996 [0.992, 0.999] | 0.993 [0.984, 0.999] | 0.999 [0.998, 1.000] | -0.001 |
| chamfer | down->up | 0.925 [0.841, 0.989] | 0.903 [0.788, 0.982] | 0.924 [0.810, 0.994] | -0.090 |
| chamfer | crop->crop | 0.987 [0.954, 0.998] | 0.980 [0.923, 0.998] | 0.994 [0.975, 1.000] | -0.014 |
| rigid_icp | tracked->tracked | 0.997 [0.994, 1.000] | 0.995 [0.985, 0.999] | 1.000 [0.999, 1.000] | +0.000 |
| rigid_icp | tracked->crop | 0.727 [0.570, 0.911] | 0.728 [0.554, 0.918] | 0.724 [0.563, 0.916] | -0.267 |
| rigid_icp | remesh->crop | 0.716 [0.554, 0.891] | 0.719 [0.538, 0.906] | 0.716 [0.553, 0.901] | -0.276 |
| rigid_icp | tracked->noisy | 0.997 [0.993, 1.000] | 0.994 [0.982, 0.999] | 1.000 [0.999, 1.000] | -0.001 |
| rigid_icp | down->up | 0.999 [0.998, 1.000] | 0.999 [0.995, 1.000] | 1.000 [1.000, 1.000] | +0.004 |
| rigid_icp | crop->crop | 0.989 [0.977, 0.997] | 0.982 [0.961, 0.994] | 0.996 [0.990, 1.000] | -0.013 |
| varifold | tracked->tracked | 0.995 [0.989, 0.999] | 0.989 [0.973, 0.999] | 1.000 [0.999, 1.000] | +0.000 |
| varifold | tracked->crop | 0.606 [0.405, 0.799] | 0.616 [0.366, 0.827] | 0.622 [0.377, 0.838] | -0.373 |
| varifold | remesh->crop | 0.630 [0.425, 0.818] | 0.639 [0.389, 0.843] | 0.646 [0.411, 0.851] | -0.350 |
| varifold | tracked->noisy | 0.992 [0.985, 0.999] | 0.983 [0.965, 0.997] | 0.999 [0.998, 1.000] | -0.005 |
| varifold | down->up | 0.893 [0.824, 0.977] | 0.878 [0.797, 0.967] | 0.896 [0.812, 0.982] | -0.110 |
| varifold | crop->crop | 1.000 [0.997, 1.000] | 0.999 [0.995, 1.000] | 1.000 [1.000, 1.000] | +0.010 |
| currents | tracked->tracked | 0.994 [0.987, 0.999] | 0.986 [0.968, 0.998] | 1.000 [0.999, 1.000] | +0.000 |
| currents | tracked->crop | 0.586 [0.387, 0.787] | 0.604 [0.353, 0.816] | 0.613 [0.352, 0.814] | -0.382 |
| currents | remesh->crop | 0.593 [0.407, 0.791] | 0.609 [0.383, 0.816] | 0.619 [0.379, 0.812] | -0.378 |
| currents | tracked->noisy | 0.993 [0.986, 0.999] | 0.984 [0.968, 0.998] | 1.000 [0.999, 1.000] | -0.002 |
| currents | down->up | 0.833 [0.756, 0.938] | 0.828 [0.731, 0.937] | 0.840 [0.738, 0.958] | -0.158 |
| currents | crop->crop | 0.999 [0.994, 1.000] | 0.998 [0.990, 1.000] | 1.000 [0.999, 1.000] | +0.012 |
| lpips | tracked->tracked | 1.000 [0.999, 1.000] | 0.999 [0.997, 1.000] | 1.000 [0.999, 1.000] | +0.000 |
| lpips | tracked->crop | 0.839 [0.653, 0.972] | 0.820 [0.605, 0.967] | 0.842 [0.626, 0.974] | -0.180 |
| lpips | remesh->crop | 0.726 [0.472, 0.928] | 0.708 [0.424, 0.914] | 0.731 [0.454, 0.928] | -0.292 |
| lpips | tracked->noisy | 0.991 [0.970, 0.999] | 0.985 [0.956, 0.999] | 0.995 [0.975, 1.000] | -0.014 |
| lpips | down->up | 1.000 [0.999, 1.000] | 0.999 [0.998, 1.000] | 1.000 [0.999, 1.000] | +0.000 |
| lpips | crop->crop | 0.999 [0.996, 1.000] | 0.998 [0.993, 1.000] | 1.000 [0.999, 1.000] | -0.001 |
| arcface | tracked->tracked | 1.000 [CI degenere] | 1.000 [CI degenere] | 1.000 [CI degenere] | +0.000 |
| arcface | tracked->crop | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [CI degenere] | -0.000 |
| arcface | remesh->crop | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [CI degenere] | -0.000 |
| arcface | tracked->noisy | 0.945 [0.869, 0.988] | 0.932 [0.836, 0.983] | 0.954 [0.877, 0.993] | -0.068 |
| arcface | down->up | 1.000 [CI degenere] | 1.000 [CI degenere] | 1.000 [CI degenere] | +0.000 |
| arcface | crop->crop | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [CI degenere] | -0.000 |
| clip | tracked->tracked | 0.986 [0.971, 0.995] | 0.978 [0.950, 0.994] | 0.992 [0.981, 0.998] | +0.000 |
| clip | tracked->crop | 0.939 [0.883, 0.980] | 0.903 [0.802, 0.966] | 0.959 [0.903, 0.994] | -0.075 |
| clip | remesh->crop | 0.791 [0.698, 0.885] | 0.724 [0.599, 0.845] | 0.815 [0.689, 0.913] | -0.254 |
| clip | tracked->noisy | 0.639 [0.494, 0.778] | 0.616 [0.447, 0.764] | 0.657 [0.516, 0.787] | -0.362 |
| clip | down->up | 0.983 [0.963, 0.994] | 0.972 [0.935, 0.991] | 0.990 [0.974, 0.998] | -0.006 |
| clip | crop->crop | 0.980 [0.958, 0.994] | 0.965 [0.923, 0.990] | 0.990 [0.977, 0.998] | -0.013 |
| dinov2 | tracked->tracked | 0.975 [0.952, 0.990] | 0.960 [0.915, 0.985] | 0.986 [0.971, 0.995] | +0.000 |
| dinov2 | tracked->crop | 0.890 [0.770, 0.959] | 0.843 [0.699, 0.943] | 0.918 [0.792, 0.978] | -0.117 |
| dinov2 | remesh->crop | 0.630 [0.463, 0.786] | 0.579 [0.394, 0.747] | 0.650 [0.459, 0.803] | -0.381 |
| dinov2 | tracked->noisy | 0.634 [0.480, 0.750] | 0.621 [0.460, 0.754] | 0.656 [0.474, 0.784] | -0.339 |
| dinov2 | down->up | 0.951 [0.907, 0.982] | 0.926 [0.863, 0.972] | 0.965 [0.929, 0.988] | -0.034 |
| dinov2 | crop->crop | 0.974 [0.949, 0.990] | 0.955 [0.912, 0.982] | 0.983 [0.968, 0.993] | -0.005 |
| latent_v1 | tracked->tracked | 0.999 [0.998, 1.000] | 0.999 [0.997, 1.000] | 1.000 [1.000, 1.000] | +0.000 |
| latent_v1 | tracked->crop | 0.679 [0.460, 0.866] | 0.727 [0.477, 0.885] | 0.726 [0.463, 0.891] | -0.272 |
| latent_v1 | remesh->crop | 0.654 [0.429, 0.847] | 0.698 [0.447, 0.877] | 0.698 [0.433, 0.879] | -0.300 |
| latent_v1 | tracked->noisy | 0.999 [0.998, 1.000] | 0.999 [0.997, 1.000] | 1.000 [0.999, 1.000] | -0.000 |
| latent_v1 | down->up | 0.986 [0.959, 0.999] | 0.977 [0.933, 0.999] | 0.991 [0.966, 1.000] | -0.021 |
| latent_v1 | crop->crop | 0.992 [0.964, 1.000] | 0.989 [0.944, 1.000] | 0.996 [0.982, 1.000] | -0.010 |
| bbox_proxy | tracked->tracked | 0.976 [0.943, 0.995] | 0.956 [0.894, 0.991] | 0.992 [0.978, 0.999] | +0.000 |
| bbox_proxy | tracked->crop | 0.449 [0.263, 0.639] | 0.416 [0.212, 0.650] | 0.427 [0.236, 0.638] | -0.541 |
| bbox_proxy | remesh->crop | 0.446 [0.256, 0.629] | 0.416 [0.207, 0.641] | 0.426 [0.231, 0.628] | -0.540 |
| bbox_proxy | tracked->noisy | 0.971 [0.935, 0.992] | 0.951 [0.888, 0.988] | 0.985 [0.960, 0.998] | -0.005 |
| bbox_proxy | down->up | 0.909 [0.809, 0.978] | 0.900 [0.792, 0.983] | 0.919 [0.803, 0.995] | -0.056 |
| bbox_proxy | crop->crop | 0.967 [0.919, 0.993] | 0.953 [0.890, 0.990] | 0.979 [0.947, 0.997] | -0.003 |
