# Run grande BFM+ICT+GNM (job 1060130): curva fuori dominio sui checkpoint intermedi

8 ottobre 2026. Checkpoint `epoch036/072/108.pth` (passi 10.548 / 21.096 / 31.644 di 105.480), valutati
con le pipeline e sugli stessi dati dei riferimenti; bracci `scale_eNNN` in `aau/zs3dmm`.
IC 95% bootstrap per soggetto (1000 repliche, seme 1234). Il congiunto e Chamfer sono i numeri
gia' pubblicati nei riepiloghi esistenti (stessi dati, stesse repliche dove indicato).

| | HIFI3D mesh-pair senza crop (Spearman) | HIFI3D subject-pair-mean | FaceVerse espr. rank-1 (conv. BFM) | FaceVerse espr. AUC | NoW tau per immagine (3 metodi) |
|---|---|---|---|---|---|
| e036 | 0.677 [0.61, 0.74] | 0.767 [0.71, 0.83] | 0.518 [0.475, 0.565] | 0.822 [0.794, 0.850] | 0.299 [0.216, 0.398] |
| e072 | 0.663 [0.59, 0.73] | 0.792 [0.73, 0.84] | in coda (job 1061482) | in coda | 0.348 [0.228, 0.474] |
| e108 | 0.630 [0.57, 0.69] | 0.795 [0.74, 0.84] | 0.625 [0.584, 0.668] | 0.868 [0.842, 0.893] | 0.277 [0.137, 0.417] |
| congiunto 1019532 | 0.428 [0.38, 0.48] | 0.720 [0.66, 0.78] | 0.680 [0.633, 0.722] | 0.874 [0.846, 0.900] | 0.131 [-0.014, 0.268] |
| Chamfer | 0.372 [0.32, 0.42] | 0.743 [0.68, 0.80] | 0.740 [0.697, 0.783] | 0.882 [0.853, 0.910] | 0.246 [0.123, 0.373] |

Differenze appaiate contro Chamfer (stesse repliche):
- HIFI3D senza crop: e036 +0.305 [+0.254, +0.349], e072 +0.291 [+0.236, +0.342], e108 +0.258 [+0.208, +0.301]
  (congiunto +0.056 [+0.015, +0.094]). Subject-pair-mean: e036 +0.024 [-0.035, +0.078], e108 +0.052 [-0.004, +0.106].
- FaceVerse rank-1: e036 -0.222 [-0.265, -0.179], e108 -0.115 [-0.155, -0.076]; AUC e108 -0.014 [-0.035, +0.008].
- NoW tau (delta pre-registrato latente - Chamfer): e036 +0.053 [-0.037, +0.153], e072 +0.102 [-0.029, +0.229],
  e108 +0.030 [-0.103, +0.153] (congiunto -0.116 [-0.192, -0.042]).

Fonti: `aau/runs/data_scale_ood/hifi/summary.md`, `aau/runs/data_scale_ood/fvexpr_partial/recognition*.csv`,
`aau/runs/now_eval_scale_e0NN/summary.md`, `concordance_paired.csv`. Job: 1060668 (HIFI3D), 1060669 (FV e036;
e072 fallito per guasto del nodo a768-l40s-03, "unknown userid"), 1060671 (FV e108), 1060670 (NoW),
1060672 / 1061516 (riepiloghi), 1061482 + 1061438 (FV e072 e riepilogo finale, in coda).
Limiti: FaceVerse solo in convenzione BFM (quella del riferimento del congiunto); il run grande ha il
91% dei dati nel frame ICT, quindi la convenzione ICT potrebbe dare numeri diversi (non valutata).
