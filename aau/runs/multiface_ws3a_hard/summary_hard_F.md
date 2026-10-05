# WS3a Multiface: AUC same-vs-different per metrica e topologia

AUC = P(distanza same < distanza different), 0.5 = caso. CI 95% bootstrap su 1000 repliche, ricampionando i 13 soggetti. "CI degenere" = tutte le repliche danno lo stesso valore (separazione perfetta in ognuna): l'intervallo non e' stretto, e' non informativo.

La riga `bbox_proxy` non e' una metrica di forma ma un CONTROLLO: distanza fra due vettori di 4 numeri (centro e diagonale del bounding box) presi sulla mesh **gia' normalizzata maxabs**, cioe' la stessa mesh che vedono chamfer, il latent e i render. Va letta per prima: dove il proxy e' alto quanto una metrica, quella cella e' risolta da informazione banale.

> La riga `bbox_proxy` NON e' stata calcolata in questa corsa: senza di lei la tabella non ha il suo metro, e le AUC qui sotto vanno lette come valori assoluti e basta. Rilancia `ws3a_geometric.sbatch` con `bbox_proxy` fra le metriche.

Classi: (a) stesso soggetto stessa espressione, (b) stesso soggetto espressione diversa, (c) soggetti diversi stessa espressione, (d) tutto diverso.

Coppie di topologie: hard. L'ultima colonna e' AUC(b_vs_c) meno la stessa AUC della stessa metrica su tracked->tracked (il caso pulito): negativa = la perturbazione fa perdere separazione.

| metrica | topologie | ab_vs_cd | b_vs_c | a_vs_c | delta b_vs_c vs clean |
|---|---|---|---|---|---|
| latent_ctrl_s1234 | tracked->tracked | 0.999 [0.998, 1.000] | 0.999 [0.995, 1.000] | 1.000 [1.000, 1.000] | +0.000 |
| latent_ctrl_s1234 | tracked->crop | 0.659 [0.429, 0.851] | 0.697 [0.430, 0.874] | 0.703 [0.417, 0.884] | -0.302 |
| latent_ctrl_s1234 | remesh->crop | 0.653 [0.422, 0.849] | 0.692 [0.415, 0.878] | 0.697 [0.416, 0.879] | -0.307 |
| latent_ctrl_s1234 | tracked->noisy | 0.999 [0.998, 1.000] | 0.999 [0.996, 1.000] | 1.000 [1.000, 1.000] | +0.000 |
| latent_ctrl_s1234 | down->up | 0.987 [0.958, 0.999] | 0.982 [0.936, 0.999] | 0.990 [0.959, 1.000] | -0.017 |
| latent_ctrl_s1234 | crop->crop | 0.994 [0.970, 1.000] | 0.991 [0.950, 1.000] | 0.997 [0.984, 1.000] | -0.008 |
| latent_abl_F | tracked->tracked | 0.993 [0.978, 1.000] | 0.986 [0.956, 0.999] | 0.999 [0.997, 1.000] | +0.000 |
| latent_abl_F | tracked->crop | 0.683 [0.489, 0.838] | 0.711 [0.473, 0.867] | 0.731 [0.499, 0.890] | -0.275 |
| latent_abl_F | remesh->crop | 0.686 [0.476, 0.857] | 0.719 [0.478, 0.891] | 0.734 [0.503, 0.907] | -0.267 |
| latent_abl_F | tracked->noisy | 0.991 [0.964, 0.999] | 0.984 [0.947, 0.998] | 0.996 [0.978, 1.000] | -0.002 |
| latent_abl_F | down->up | 0.986 [0.963, 0.998] | 0.979 [0.944, 0.997] | 0.991 [0.968, 1.000] | -0.008 |
| latent_abl_F | crop->crop | 0.994 [0.968, 1.000] | 0.991 [0.948, 1.000] | 0.997 [0.984, 1.000] | +0.004 |
| latent_abl_B_tokA | tracked->tracked | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | +0.000 |
| latent_abl_B_tokA | tracked->crop | 0.662 [0.471, 0.813] | 0.703 [0.486, 0.860] | 0.729 [0.537, 0.887] | -0.297 |
| latent_abl_B_tokA | remesh->crop | 0.663 [0.449, 0.816] | 0.706 [0.484, 0.871] | 0.727 [0.528, 0.897] | -0.294 |
| latent_abl_B_tokA | tracked->noisy | 0.999 [0.995, 1.000] | 0.999 [0.995, 1.000] | 0.999 [0.998, 1.000] | -0.001 |
| latent_abl_B_tokA | down->up | 0.880 [0.764, 0.982] | 0.871 [0.736, 0.986] | 0.896 [0.772, 0.995] | -0.129 |
| latent_abl_B_tokA | crop->crop | 0.997 [0.984, 1.000] | 0.995 [0.973, 1.000] | 0.999 [0.993, 1.000] | -0.005 |
| latent_abl_B_tok0 | tracked->tracked | 1.000 [0.998, 1.000] | 0.999 [0.998, 1.000] | 1.000 [1.000, 1.000] | +0.000 |
| latent_abl_B_tok0 | tracked->crop | 0.557 [0.370, 0.709] | 0.585 [0.396, 0.768] | 0.608 [0.394, 0.795] | -0.414 |
| latent_abl_B_tok0 | remesh->crop | 0.566 [0.380, 0.729] | 0.598 [0.411, 0.785] | 0.619 [0.414, 0.810] | -0.401 |
| latent_abl_B_tok0 | tracked->noisy | 0.998 [0.993, 1.000] | 0.997 [0.991, 1.000] | 0.999 [0.996, 1.000] | -0.002 |
| latent_abl_B_tok0 | down->up | 0.804 [0.659, 0.932] | 0.765 [0.590, 0.925] | 0.807 [0.644, 0.952] | -0.235 |
| latent_abl_B_tok0 | crop->crop | 0.993 [0.972, 1.000] | 0.989 [0.949, 1.000] | 0.997 [0.985, 1.000] | -0.010 |
