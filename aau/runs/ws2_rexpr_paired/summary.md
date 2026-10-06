# Espressioni casuali su ICT (riga rexpr di WS2): modello - Chamfer eval appaiato

GT d'identita': vertex-mean-L2 maxabs fra le `original` NEUTRE (`datasets/ICT/train_ready/gt_matrix.npz`). Input: le due mesh di ogni coppia hanno espressioni casuali diverse (k != k', 3-8 blendshape ICT, coefficienti U(0.3, 1.0), `aau/ict/ict_expressions_random.py`), topologia `original` per entrambe (nessuna perturbazione topologica in questa riga). Differenza degli Spearman con la GT, latent - Chamfer eval, CI 95% su 1000 repliche bootstrap per soggetto, le stesse per i due lati.

| modello | vista | protocollo | latent | Chamfer eval | latent - Chamfer [CI 95%] | P(boot <= 0) | soggetti | righe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BFM-only | mixed | mesh_pair | 0.739 | 0.886 | -0.147 [-0.180, -0.114] | 1.000 | 100 | 99000 |
| BFM-only | mixed | subject_pair_mean | 0.816 | 0.930 | -0.114 [-0.148, -0.085] | 1.000 | 100 | 4950 |
| ICT-only | mixed | mesh_pair | 0.918 | 0.854 | +0.064 [+0.047, +0.083] | 0.000 | 95 | 89300 |
| ICT-only | mixed | subject_pair_mean | 0.979 | 0.914 | +0.065 [+0.047, +0.084] | 0.000 | 95 | 4465 |
| ICT-only | neutral | mesh_pair | 0.993 | 0.939 | +0.054 [+0.042, +0.069] | 0.000 | 95 | 4465 |
| ICT-only | neutral | subject_pair_mean | 0.993 | 0.939 | +0.054 [+0.042, +0.071] | 0.000 | 95 | 4465 |
| BFM+ICT | mixed | mesh_pair | 0.881 | 0.838 | +0.044 [+0.024, +0.067] | 0.000 | 89 | 78320 |
| BFM+ICT | mixed | subject_pair_mean | 0.965 | 0.897 | +0.068 [+0.045, +0.096] | 0.000 | 89 | 3916 |
| BFM+ICT | neutral | mesh_pair | 0.992 | 0.929 | +0.064 [+0.043, +0.093] | 0.000 | 89 | 3916 |
| BFM+ICT | neutral | subject_pair_mean | 0.992 | 0.929 | +0.064 [+0.043, +0.094] | 0.000 | 89 | 3916 |

Sorgenti (eval_key.txt): BFM-only/mixed: `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/ICT/rexpr_view`; ICT-only/mixed: `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/WS2_CROSS3DMM/ict_only__rexpr`; ICT-only/neutral: `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/WS2_CROSS3DMM/ict_only__rexpr`; BFM+ICT/mixed: `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/WS2_CROSS3DMM/joint__rexpr`; BFM+ICT/neutral: `/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/datasets/WS2_CROSS3DMM/joint__rexpr`
