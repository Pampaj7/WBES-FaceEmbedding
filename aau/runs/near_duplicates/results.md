| split | 3DMM | metrica | train / test | soglia (regola) | test sotto soglia | NN test->train: min / p5 / mediana | NN train->train (lascia-uno-fuori): min / p5 / mediana | NN test->test: min / mediana | distanze test-test: p1 / mediana | min NN test->train / soglia |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neurips | BFM | vertex_mean_l2_maxabs | 400 / 100 | 0.0055 (valanga 0.0055) | 0 | 0.0213 / 0.0225 / 0.0256 | 0.0195 / 0.0213 / 0.0261 | 0.0230 / 0.0273 | 0.0269 / 0.0454 | 3.9x |
| neurips | BFM | gt_training (normalized_matrix_distances.npz) | 400 / 100 | 0.0654 (meta' NN minimo del pool) | 0 | 0.1562 / 0.1632 / 0.1928 | 0.1308 / 0.1629 / 0.1963 | 0.1734 / 0.2078 | 0.2076 / 0.3355 | 2.4x |
| joint | BFM | vertex_mean_l2_maxabs | 392 / 108 | 0.0055 (valanga 0.0055) | 0 | 0.0201 / 0.0229 / 0.0259 | 0.0195 / 0.0218 / 0.0260 | 0.0213 / 0.0289 | 0.0290 / 0.0478 | 3.6x |
| joint | BFM | gt_training (gt_matrix.npz) | 392 / 108 | 0.0654 (meta' NN minimo del pool) | 0 | 0.1308 / 0.1681 / 0.1963 | 0.1442 / 0.1631 / 0.1944 | 0.1619 / 0.2203 | 0.2172 / 0.3560 | 2.0x |
| joint | ICT | vertex_mean_l2_maxabs | 4008 / 992 | 0.0055 (valanga 0.0055) | 0 | 0.0115 / 0.0137 / 0.0164 | 0.0111 / 0.0139 / 0.0166 | 0.0126 / 0.0178 | 0.0204 / 0.0400 | 2.1x |
| joint | ICT | gt_training (gt_matrix.npz) | 4008 / 992 | 0.0322 (meta' NN minimo del pool) | 0 | 0.0670 / 0.0799 / 0.0957 | 0.0644 / 0.0808 / 0.0963 | 0.0732 / 0.1035 | 0.1189 / 0.2330 | 2.1x |

Controlli:
- neurips / bfm: Spearman fra vertex-mean-L2 maxabs e GT di training su 124750 coppie = 0.783
- joint / bfm: Spearman fra vertex-mean-L2 maxabs e GT di training su 124750 coppie = 0.783
- joint / ict: Spearman fra vertex-mean-L2 maxabs e GT di training su 12497500 coppie = 1.000
