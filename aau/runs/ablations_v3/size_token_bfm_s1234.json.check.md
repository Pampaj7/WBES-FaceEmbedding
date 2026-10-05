# Token di taglia, bfm: consistenza fra topologie dello stesso soggetto

Sorgente `datasets/REMESH/npz_data_topo_500`, 3000 mesh. Standardizzazione: rebuild_subject_split(eval_fraction=0.2, seed=1234, max_subjects=0) sui soggetti di npz_data_topo_500 con riga in normalized_matrix_distances.npz: training; 2400 mesh, media 10.92692, std 0.03544.

Std fra soggetti del log r sull'original: 0.01559.

| topologia | n | media Δlog r | std Δlog r | max abs | rapporto raggi | media in std token | max abs in std token |
|---|---|---|---|---|---|---|---|
| remesh | 500 | -0.00340 | 0.00052 | 0.00491 | 0.9966 | -0.096 | 0.139 |
| crop | 500 | -0.08788 | 0.00636 | 0.10276 | 0.9159 | -2.480 | 2.899 |
| noisy | 500 | -0.00583 | 0.00568 | 0.02464 | 0.9942 | -0.164 | 0.695 |
| down8k | 500 | +0.00009 | 0.00009 | 0.00032 | 1.0001 | +0.003 | 0.009 |
| up60k | 500 | +0.00144 | 0.00025 | 0.00198 | 1.0014 | +0.041 | 0.056 |
