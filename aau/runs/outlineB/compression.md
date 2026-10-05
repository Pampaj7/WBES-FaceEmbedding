# Compressione delle distanze fra soggetti diversi (OUTLINE_B §4, esperimento 2)

Percentuale trattenuta rispetto a Chamfer raw, CI 95% bootstrap per soggetto (1000 repliche, rapporto appaiato nella stessa replica).

## IQR relativo (IQR/mediana): invariante per scala, la forma corretta (0281a3f)

| set | gruppo | coppie | Rigid ICP + Chamfer | Rigid ICP + NICP + P2P | Rigid ICP + NICP + P2Tri |
|---|---|---|---|---|---|
| bfm_fb100 | same | 4950 | 82.5 [74.5, 91.7] | 44.4 [38.5, 50.0] | 52.6 [46.1, 58.9] |
| bfm_fb100 | tessellation | 59400 | 89.5 [83.1, 96.2] | 46.8 [42.5, 51.3] | 54.8 [50.0, 60.2] |
| bfm_fb100 | perturbation | 39600 | 83.9 [77.0, 91.2] | 42.4 [38.4, 47.1] | 50.3 [45.5, 55.8] |
| bfm_fb100 | nocrop | 99000 | 87.2 [81.2, 94.0] | 44.7 [40.1, 49.0] | 52.7 [47.4, 57.7] |
| bfm_fb100 | all_cross | 148500 | 72.2 [67.5, 76.9] | 36.1 [33.3, 38.8] | 42.1 [38.9, 45.1] |
| bfm_heldout | same | 4950 | 83.9 [74.2, 93.1] | 45.0 [39.0, 53.2] | 53.6 [47.1, 63.1] |
| bfm_heldout | tessellation | 59400 | 88.8 [82.0, 95.9] | 49.7 [44.4, 55.4] | 58.6 [52.4, 65.3] |
| bfm_heldout | perturbation | 39600 | 81.8 [74.3, 88.1] | 44.0 [39.0, 49.9] | 52.8 [46.8, 59.8] |
| bfm_heldout | nocrop | 99000 | 85.9 [78.6, 92.7] | 46.9 [41.7, 52.3] | 55.6 [49.7, 62.1] |
| bfm_heldout | all_cross | 148500 | 68.1 [63.8, 72.2] | 34.0 [31.0, 37.0] | 39.7 [36.4, 43.1] |
| ict_heldout | same | 4950 | 90.2 [78.7, 101.7] | 49.3 [41.8, 57.1] | 57.7 [48.6, 67.0] |
| ict_heldout | tessellation | 57108 | 77.3 [67.4, 85.7] | 40.0 [35.9, 44.0] | 45.8 [41.1, 50.5] |
| ict_heldout | perturbation | 38836 | 56.7 [50.0, 65.1] | 29.1 [26.0, 32.8] | 32.6 [29.3, 37.0] |
| ict_heldout | nocrop | 95944 | 68.2 [59.2, 76.8] | 35.2 [31.3, 39.0] | 39.9 [35.5, 44.2] |
| ict_heldout | all_cross | 144680 | 73.7 [67.6, 80.0] | 39.8 [36.4, 43.0] | 44.8 [41.2, 48.4] |
| bfm_fb100_s2048 | same | 4950 | 80.3 [72.0, 88.7] | 41.3 [36.0, 47.5] | 51.1 [44.7, 57.8] |
| bfm_fb100_s2048 | tessellation | 59400 | 84.6 [78.6, 91.1] | 44.2 [39.6, 48.7] | 53.8 [48.1, 59.4] |
| bfm_fb100_s2048 | perturbation | 39600 | 79.7 [73.9, 87.1] | 41.1 [36.9, 45.4] | 50.1 [45.2, 55.3] |
| bfm_fb100_s2048 | nocrop | 99000 | 82.5 [76.2, 88.8] | 42.8 [38.4, 47.3] | 52.0 [47.0, 57.7] |
| bfm_fb100_s2048 | all_cross | 148500 | 68.1 [63.7, 72.4] | 33.9 [31.2, 36.2] | 40.8 [37.8, 43.5] |
| bfm_heldout_s2048 | same | 4950 | 79.5 [70.2, 86.8] | 43.5 [37.1, 49.8] | 54.0 [46.4, 61.6] |
| bfm_heldout_s2048 | tessellation | 59400 | 83.9 [77.4, 90.3] | 46.8 [41.8, 52.0] | 57.2 [51.2, 63.6] |
| bfm_heldout_s2048 | perturbation | 39600 | 77.8 [70.8, 84.3] | 42.8 [37.9, 48.6] | 52.6 [47.1, 59.6] |
| bfm_heldout_s2048 | nocrop | 99000 | 81.3 [75.0, 87.8] | 44.7 [39.4, 49.9] | 55.1 [48.4, 61.1] |
| bfm_heldout_s2048 | all_cross | 148500 | 64.6 [60.6, 68.2] | 31.3 [28.2, 34.5] | 38.0 [34.3, 41.8] |
| ict_heldout_s2048 | same | 4950 | 88.4 [77.5, 99.6] | 47.7 [41.2, 55.6] | 57.1 [49.1, 67.0] |
| ict_heldout_s2048 | tessellation | 58743 | 73.0 [63.7, 81.3] | 38.2 [34.6, 42.1] | 44.9 [41.0, 49.1] |
| ict_heldout_s2048 | perturbation | 39381 | 55.1 [48.2, 62.6] | 29.7 [27.2, 32.7] | 34.5 [31.6, 37.8] |
| ict_heldout_s2048 | nocrop | 98124 | 65.0 [56.4, 73.0] | 34.4 [31.1, 37.9] | 40.2 [36.5, 44.1] |
| ict_heldout_s2048 | all_cross | 147405 | 72.0 [66.5, 77.3] | 39.8 [37.1, 42.9] | 46.2 [43.0, 49.5] |

## IQR assoluto (la forma del paper, gonfiata dalla riscalatura di prealign_by_bbox)

| set | gruppo | coppie | Rigid ICP + Chamfer | Rigid ICP + NICP + P2P | Rigid ICP + NICP + P2Tri |
|---|---|---|---|---|---|
| bfm_fb100 | same | 4950 | 66.9 [59.9, 74.9] | 28.8 [24.9, 33.0] | 31.0 [26.8, 35.2] |
| bfm_fb100 | tessellation | 59400 | 68.9 [63.5, 74.8] | 28.5 [25.5, 31.8] | 30.4 [27.2, 33.9] |
| bfm_fb100 | perturbation | 39600 | 65.5 [59.7, 71.9] | 27.3 [24.4, 30.8] | 29.4 [26.4, 33.2] |
| bfm_fb100 | nocrop | 99000 | 67.4 [62.0, 73.3] | 27.8 [24.7, 30.8] | 29.8 [26.5, 33.0] |
| bfm_fb100 | all_cross | 148500 | 54.3 [50.7, 58.2] | 20.9 [19.2, 22.6] | 22.4 [20.5, 24.1] |
| bfm_heldout | same | 4950 | 69.5 [61.2, 78.0] | 31.0 [26.5, 37.5] | 33.3 [28.7, 40.1] |
| bfm_heldout | tessellation | 59400 | 68.7 [62.5, 74.9] | 31.4 [27.5, 35.4] | 33.5 [29.5, 37.8] |
| bfm_heldout | perturbation | 39600 | 64.5 [58.2, 70.4] | 29.7 [25.7, 34.3] | 32.2 [27.9, 37.0] |
| bfm_heldout | nocrop | 99000 | 66.9 [60.5, 73.2] | 30.4 [26.8, 34.5] | 32.7 [28.8, 37.0] |
| bfm_heldout | all_cross | 148500 | 51.4 [48.0, 54.8] | 20.2 [18.4, 22.1] | 21.6 [19.7, 23.6] |
| ict_heldout | same | 4950 | 81.2 [69.6, 92.1] | 36.0 [29.8, 42.7] | 38.3 [31.6, 45.4] |
| ict_heldout | tessellation | 57108 | 50.5 [43.6, 56.4] | 20.3 [18.3, 22.3] | 21.2 [18.9, 23.3] |
| ict_heldout | perturbation | 38836 | 42.1 [37.6, 47.2] | 17.5 [16.0, 19.0] | 17.8 [16.3, 19.6] |
| ict_heldout | nocrop | 95944 | 46.1 [40.4, 51.5] | 18.7 [17.0, 20.5] | 19.4 [17.5, 21.2] |
| ict_heldout | all_cross | 144680 | 50.7 [46.5, 55.0] | 21.2 [19.5, 22.9] | 21.9 [20.1, 23.7] |
| bfm_fb100_s2048 | same | 4950 | 67.8 [60.6, 75.4] | 29.7 [25.8, 34.5] | 32.6 [28.2, 37.5] |
| bfm_fb100_s2048 | tessellation | 59400 | 68.2 [62.9, 74.2] | 30.0 [26.6, 33.5] | 32.6 [28.7, 36.4] |
| bfm_fb100_s2048 | perturbation | 39600 | 65.1 [59.6, 71.4] | 29.0 [25.9, 32.4] | 31.4 [28.1, 35.2] |
| bfm_fb100_s2048 | nocrop | 99000 | 66.8 [61.2, 72.6] | 29.5 [26.3, 33.1] | 31.9 [28.5, 35.8] |
| bfm_fb100_s2048 | all_cross | 148500 | 53.7 [50.0, 57.3] | 21.9 [20.2, 23.5] | 23.6 [21.9, 25.4] |
| bfm_heldout_s2048 | same | 4950 | 68.7 [60.4, 75.5] | 32.9 [27.8, 38.1] | 36.0 [30.6, 41.5] |
| bfm_heldout_s2048 | tessellation | 59400 | 68.1 [62.2, 73.7] | 32.8 [29.0, 36.9] | 35.6 [31.5, 40.2] |
| bfm_heldout_s2048 | perturbation | 39600 | 64.3 [58.0, 70.0] | 31.4 [27.5, 35.9] | 34.2 [30.1, 39.2] |
| bfm_heldout_s2048 | nocrop | 99000 | 66.4 [60.5, 72.3] | 32.0 [27.7, 36.0] | 34.9 [30.2, 39.1] |
| bfm_heldout_s2048 | all_cross | 148500 | 51.2 [48.0, 54.3] | 20.7 [18.7, 23.0] | 22.5 [20.3, 24.8] |
| ict_heldout_s2048 | same | 4950 | 81.1 [71.0, 92.0] | 37.4 [31.8, 44.5] | 39.9 [33.8, 47.8] |
| ict_heldout_s2048 | tessellation | 58743 | 50.5 [44.0, 56.7] | 21.7 [19.8, 23.8] | 23.0 [21.0, 25.1] |
| ict_heldout_s2048 | perturbation | 39381 | 42.8 [38.0, 47.6] | 19.5 [18.2, 21.0] | 20.4 [19.0, 22.0] |
| ict_heldout_s2048 | nocrop | 98124 | 46.5 [40.6, 51.9] | 20.5 [18.8, 22.2] | 21.5 [19.8, 23.2] |
| ict_heldout_s2048 | all_cross | 147405 | 52.3 [48.4, 56.4] | 23.7 [22.1, 25.5] | 24.8 [23.1, 26.7] |

## Gate su bfm_fb100_s2048 (30 coppie cross a 2048 punti, l'insieme dell'artefatto del paper)

```
  [OK] rigid_icp_chamfer iqr_ratio: qui 0.5370 vs artefatto 0.5370 (delta -0.0000)
  [OK] rigid_icp_chamfer rel_iqr_ratio: qui 0.6814 vs artefatto 0.6814 (delta +0.0000)
  [OK] nicp_p2p iqr_ratio: qui 0.2187 vs artefatto 0.2187 (delta -0.0000)
  [OK] nicp_p2p rel_iqr_ratio: qui 0.3387 vs artefatto 0.3387 (delta +0.0000)
  [OK] nicp_p2tri iqr_ratio: qui 0.2365 vs artefatto 0.2365 (delta +0.0000)
  [OK] nicp_p2tri rel_iqr_ratio: qui 0.4084 vs artefatto 0.4084 (delta +0.0000)
  max errore relativo sugli IQR per coppia di topologie (30 coppie x 4 metriche): 5.85e-10
```
