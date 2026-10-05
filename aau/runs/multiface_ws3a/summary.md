# WS3a Multiface: AUC same-vs-different per metrica e topologia

AUC = P(distanza same < distanza different), 0.5 = caso. CI 95% bootstrap su 1000 repliche, ricampionando i 13 soggetti.

Classi: (a) stesso soggetto stessa espressione, (b) stesso soggetto espressione diversa, (c) soggetti diversi stessa espressione, (d) tutto diverso.

| metrica | topologie | ab_vs_cd | b_vs_c | a_vs_c |
|---|---|---|---|---|
| chamfer | tracked->tracked | 0.997 [0.992, 0.999] | 0.993 [0.983, 0.999] | 0.999 [0.998, 1.000] |
| chamfer | remesh->remesh | 0.997 [0.992, 1.000] | 0.994 [0.982, 1.000] | 1.000 [0.998, 1.000] |
| chamfer | tracked->remesh | 0.988 [0.969, 0.998] | 0.977 [0.933, 0.997] | 0.996 [0.988, 1.000] |
| chamfer | down->down | 0.999 [0.995, 1.000] | 0.998 [0.988, 1.000] | 1.000 [0.999, 1.000] |
| rigid_icp | tracked->tracked | 0.997 [0.994, 1.000] | 0.995 [0.985, 0.999] | 1.000 [0.999, 1.000] |
| rigid_icp | remesh->remesh | 0.998 [0.995, 1.000] | 0.996 [0.987, 1.000] | 1.000 [1.000, 1.000] |
| rigid_icp | tracked->remesh | 0.997 [0.991, 1.000] | 0.993 [0.979, 0.999] | 1.000 [0.998, 1.000] |
| rigid_icp | down->down | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| varifold | tracked->tracked | 0.737 [0.674, 0.813] | 0.722 [0.663, 0.794] | 0.730 [0.674, 0.798] |
| varifold | remesh->remesh | 0.967 [0.941, 0.986] | 0.962 [0.943, 0.984] | 0.967 [0.942, 0.987] |
| varifold | tracked->remesh | 0.758 [0.684, 0.836] | 0.735 [0.672, 0.811] | 0.755 [0.692, 0.826] |
| varifold | down->down | 0.981 [0.962, 0.995] | 0.978 [0.956, 0.994] | 0.984 [0.969, 0.996] |
| lpips | tracked->tracked | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| lpips | remesh->remesh | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| lpips | tracked->remesh | 0.999 [0.993, 1.000] | 0.998 [0.993, 1.000] | 0.999 [0.995, 1.000] |
| lpips | down->down | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| arcface | tracked->tracked | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| arcface | remesh->remesh | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| arcface | tracked->remesh | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| arcface | down->down | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| latent_v1 | tracked->tracked | 0.999 [0.998, 1.000] | 0.999 [0.997, 1.000] | 1.000 [1.000, 1.000] |
| latent_v1 | remesh->remesh | 0.999 [0.998, 1.000] | 0.999 [0.997, 1.000] | 1.000 [0.999, 1.000] |
| latent_v1 | tracked->remesh | 0.990 [0.973, 0.998] | 0.983 [0.952, 0.997] | 0.997 [0.991, 1.000] |
| latent_v1 | down->down | 1.000 [0.998, 1.000] | 0.999 [0.996, 1.000] | 1.000 [0.999, 1.000] |
