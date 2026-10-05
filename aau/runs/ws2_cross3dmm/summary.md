# WS2: tabella cross-3DMM (seed 1234, ricetta v1, operatori ad area unitaria)

Modelli: BFM-only (job 1019310), ICT-only (1019531), BFM+ICT congiunto (1019532). Ogni cella e' valutata SOLO su soggetti held-out del modello valutato (vedi controllo leak). Soggetti per cella: BFM 100/100/108, ICT 100/95/89 (BFM-only/ICT-only/congiunto). GT: BFM `normalized_matrix_distances.npz`, ICT `datasets/ICT/train_ready/gt_matrix.npz` (la GT di `datasets/ICT/gt/` con l'offset id+10000). CI 95% bootstrap subject-level, 1000 ricampionamenti.

## Mesh-pair cross-topology, clean (protocollo dello 0.301 zero-shot)

| modello | BFM latent | BFM chamfer | ICT latent | ICT chamfer |
| --- | --- | --- | --- | --- |
| BFM-only | 0.779 [0.73, 0.82] | 0.237 [0.20, 0.28] | 0.294 [0.24, 0.34] | 0.442 [0.36, 0.52] |
| ICT-only | 0.175 [0.14, 0.21] | 0.237 [0.20, 0.28] | 0.986 [0.98, 0.99] | 0.355 [0.31, 0.40] |
| BFM+ICT | 0.856 [0.83, 0.88] | 0.246 [0.21, 0.28] | 0.981 [0.97, 0.99] | 0.347 [0.27, 0.42] |

### Senza crop (colonna della Tabella 2 del paper)

| modello | BFM latent | BFM chamfer | ICT latent | ICT chamfer |
| --- | --- | --- | --- | --- |
| BFM-only | 0.804 [0.76, 0.84] | 0.472 [0.40, 0.53] | 0.259 [0.21, 0.31] | 0.443 [0.37, 0.51] |
| ICT-only | 0.382 [0.31, 0.44] | 0.472 [0.41, 0.54] | 0.988 [0.98, 0.99] | 0.364 [0.32, 0.41] |
| BFM+ICT | 0.865 [0.84, 0.89] | 0.476 [0.42, 0.53] | 0.985 [0.98, 0.99] | 0.357 [0.29, 0.43] |

## Subject-pair-mean (script di ranking), clean

| modello | BFM latent | BFM chamfer | ICT latent | ICT chamfer |
| --- | --- | --- | --- | --- |
| BFM-only | 0.841 [0.80, 0.87] | 0.485 [0.36, 0.60] | 0.836 [0.79, 0.88] | 0.911 [0.88, 0.94] |
| ICT-only | 0.573 [0.49, 0.65] | 0.485 [0.36, 0.59] | 0.994 [0.99, 1.00] | 0.894 [0.86, 0.92] |
| BFM+ICT | 0.906 [0.88, 0.92] | 0.511 [0.40, 0.60] | 0.993 [0.99, 0.99] | 0.860 [0.79, 0.91] |

## Subject-pair-mean, mixed (solo punto: il breakdown non perturba, niente CI)

| modello | BFM latent | BFM chamfer | ICT latent | ICT chamfer |
| --- | --- | --- | --- | --- |
| BFM-only | 0.804 | 0.448 | 0.821 | 0.886 |
| ICT-only | 0.541 | 0.434 | 0.982 | 0.847 |
| BFM+ICT | 0.878 | 0.468 | 0.977 | 0.811 |

## Espressioni casuali, regime (c) misto, contro la baseline neutra sugli stessi soggetti

| modello | metrica | misto | neutro | delta | n soggetti |
| --- | --- | --- | --- | --- | --- |
| BFM-only | latent | 0.739 [0.68, 0.79] | 0.841 | -0.101 | 100 |
| BFM-only | chamfer | 0.886 [0.85, 0.92] | 0.951 | -0.065 | 100 |
| ICT-only | latent | 0.918 [0.89, 0.94] | 0.993 [0.99, 0.99] | -0.076 | 95 |
| ICT-only | chamfer | 0.854 [0.81, 0.88] | 0.939 [0.92, 0.95] | -0.085 | 95 |
| BFM+ICT | latent | 0.881 [0.83, 0.92] | 0.992 [0.99, 0.99] | -0.111 | 89 |
| BFM+ICT | chamfer | 0.838 [0.78, 0.88] | 0.929 [0.90, 0.95] | -0.091 | 89 |

BFM-only: misto dalle pair_metrics esistenti (job 1019721), neutro dal job 1019710 (solo punto). Il misto e' a una mesh per soggetto (pair_metrics), come in ict_rexpr_summary.


## Controllo leak (sugli output, non sulle viste)

| cell | n_evaluated | evaluated_equals_split_list | n_train_model | n_evaluated_in_train | n_evaluated_in_online_selection | n_json_checked |
| --- | --- | --- | --- | --- | --- | --- |
| bfm_only__bfm | 100 | True | 400 | 0 | 16 | 31 |
| bfm_only__ict | 100 | True | 400 | 0 | 0 | 31 |
| ict_only__bfm | 100 | True | 4000 | 0 | 0 | 31 |
| ict_only__ict | 95 | True | 4000 | 0 | 5 | 31 |
| joint__bfm | 108 | True | 4400 | 0 | 16 | 31 |
| joint__ict | 89 | True | 4400 | 0 | 0 | 31 |
| bfm_only__rexpr | 100 | True | 400 | 0 | 0 | 21 |
| ict_only__rexpr | 95 | True | 4000 | 0 | 5 | 23 |
| joint__rexpr | 89 | True | 4400 | 0 | 0 | 23 |

`n_evaluated_in_online_selection`: soggetti held-out che erano fra i 16 dell'eval online del training, cioe' hanno pesato sulla scelta del checkpoint `best_by_xtopo_mesh_clean` (non sul gradiente). Stesso protocollo di tutte le eval esistenti.


Consistenza: breakdown BFM-only su BFM rifatto con pair_metrics contro quello esistente, max |delta latent| sulle coppie di topologie = 3.35e-04.


## Matrici per coppia di topologie (clean, mesh-pair, Spearman; righe A, colonne B)


### BFM-only su BFM, latent

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.721 | 0.769 | 0.780 | 0.745 | 0.768 |
| down8k | 0.712 | - | 0.811 | 0.789 | 0.798 | 0.779 |
| noisy | 0.756 | 0.807 | - | 0.824 | 0.808 | 0.803 |
| original | 0.779 | 0.799 | 0.834 | - | 0.799 | 0.810 |
| remesh | 0.739 | 0.803 | 0.814 | 0.796 | - | 0.801 |
| up60k | 0.772 | 0.793 | 0.815 | 0.813 | 0.810 | - |

### BFM-only su BFM, chamfer

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | -0.093 | -0.048 | -0.035 | -0.008 | -0.074 |
| down8k | 0.037 | - | 0.472 | 0.407 | 0.436 | 0.611 |
| noisy | 0.083 | 0.313 | - | 0.659 | 0.579 | 0.489 |
| original | 0.093 | 0.240 | 0.627 | - | 0.553 | 0.414 |
| remesh | 0.102 | 0.338 | 0.621 | 0.622 | - | 0.525 |
| up60k | 0.061 | 0.564 | 0.597 | 0.539 | 0.573 | - |

### BFM-only su ICT, latent

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.244 | 0.622 | 0.655 | 0.617 | 0.635 |
| down8k | 0.173 | - | 0.043 | 0.044 | 0.170 | 0.095 |
| noisy | 0.659 | 0.181 | - | 0.839 | 0.485 | 0.667 |
| original | 0.688 | 0.183 | 0.837 | - | 0.506 | 0.676 |
| remesh | 0.632 | 0.273 | 0.448 | 0.472 | - | 0.538 |
| up60k | 0.674 | 0.224 | 0.652 | 0.662 | 0.555 | - |

### BFM-only su ICT, chamfer

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.560 | 0.411 | 0.462 | 0.800 | 0.710 |
| down8k | 0.570 | - | 0.098 | 0.114 | 0.381 | 0.283 |
| noisy | 0.485 | 0.239 | - | 0.941 | 0.683 | 0.762 |
| original | 0.523 | 0.249 | 0.944 | - | 0.709 | 0.788 |
| remesh | 0.801 | 0.437 | 0.681 | 0.712 | - | 0.909 |
| up60k | 0.729 | 0.374 | 0.774 | 0.802 | 0.901 | - |

### ICT-only su BFM, latent

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | -0.133 | -0.112 | -0.118 | -0.101 | -0.132 |
| down8k | 0.102 | - | 0.332 | 0.405 | 0.522 | 0.608 |
| noisy | 0.136 | 0.178 | - | 0.546 | 0.273 | 0.195 |
| original | 0.132 | 0.282 | 0.583 | - | 0.393 | 0.309 |
| remesh | 0.116 | 0.457 | 0.394 | 0.466 | - | 0.546 |
| up60k | 0.104 | 0.591 | 0.351 | 0.425 | 0.592 | - |

### ICT-only su BFM, chamfer

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | -0.093 | -0.048 | -0.035 | -0.008 | -0.074 |
| down8k | 0.037 | - | 0.472 | 0.407 | 0.436 | 0.611 |
| noisy | 0.083 | 0.313 | - | 0.659 | 0.579 | 0.489 |
| original | 0.093 | 0.240 | 0.627 | - | 0.553 | 0.414 |
| remesh | 0.102 | 0.338 | 0.621 | 0.622 | - | 0.525 |
| up60k | 0.061 | 0.564 | 0.597 | 0.539 | 0.573 | - |

### ICT-only su ICT, latent

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.980 | 0.983 | 0.985 | 0.984 | 0.980 |
| down8k | 0.980 | - | 0.986 | 0.987 | 0.988 | 0.982 |
| noisy | 0.983 | 0.987 | - | 0.991 | 0.991 | 0.986 |
| original | 0.985 | 0.987 | 0.991 | - | 0.993 | 0.990 |
| remesh | 0.985 | 0.988 | 0.991 | 0.993 | - | 0.989 |
| up60k | 0.980 | 0.982 | 0.986 | 0.990 | 0.989 | - |

### ICT-only su ICT, chamfer

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.437 | 0.350 | 0.402 | 0.732 | 0.637 |
| down8k | 0.487 | - | 0.182 | 0.193 | 0.376 | 0.306 |
| noisy | 0.353 | 0.120 | - | 0.928 | 0.604 | 0.727 |
| original | 0.399 | 0.129 | 0.927 | - | 0.645 | 0.766 |
| remesh | 0.733 | 0.334 | 0.601 | 0.647 | - | 0.883 |
| up60k | 0.644 | 0.256 | 0.716 | 0.760 | 0.888 | - |

### BFM+ICT su BFM, latent

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.796 | 0.828 | 0.850 | 0.833 | 0.843 |
| down8k | 0.828 | - | 0.875 | 0.872 | 0.850 | 0.828 |
| noisy | 0.846 | 0.874 | - | 0.889 | 0.867 | 0.869 |
| original | 0.864 | 0.871 | 0.890 | - | 0.883 | 0.884 |
| remesh | 0.852 | 0.838 | 0.864 | 0.882 | - | 0.872 |
| up60k | 0.854 | 0.812 | 0.861 | 0.882 | 0.866 | - |

### BFM+ICT su BFM, chamfer

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | -0.005 | 0.041 | 0.060 | 0.075 | 0.009 |
| down8k | 0.021 | - | 0.392 | 0.320 | 0.436 | 0.598 |
| noisy | 0.068 | 0.414 | - | 0.643 | 0.621 | 0.561 |
| original | 0.084 | 0.349 | 0.646 | - | 0.605 | 0.502 |
| remesh | 0.066 | 0.407 | 0.535 | 0.508 | - | 0.560 |
| up60k | 0.028 | 0.584 | 0.517 | 0.448 | 0.580 | - |

### BFM+ICT su ICT, latent

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.972 | 0.976 | 0.980 | 0.979 | 0.973 |
| down8k | 0.971 | - | 0.981 | 0.984 | 0.986 | 0.979 |
| noisy | 0.974 | 0.981 | - | 0.988 | 0.986 | 0.981 |
| original | 0.978 | 0.985 | 0.989 | - | 0.989 | 0.987 |
| remesh | 0.978 | 0.986 | 0.986 | 0.989 | - | 0.984 |
| up60k | 0.972 | 0.981 | 0.984 | 0.988 | 0.985 | - |

### BFM+ICT su ICT, chamfer

| A \ B | crop | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- |
| crop | - | 0.565 | 0.255 | 0.316 | 0.672 | 0.580 |
| down8k | 0.352 | - | -0.058 | -0.046 | 0.199 | 0.094 |
| noisy | 0.477 | 0.342 | - | 0.921 | 0.687 | 0.778 |
| original | 0.510 | 0.348 | 0.912 | - | 0.711 | 0.801 |
| remesh | 0.712 | 0.504 | 0.519 | 0.569 | - | 0.853 |
| up60k | 0.657 | 0.435 | 0.653 | 0.698 | 0.871 | - |
