# Triplette dello studio umano v2

Generato il 2026-10-09T11:30:35+00:00 da `aau/human_study_v2/select_triplets_v2.py`, seed 1234.

**Stato delle GT: definitivo (E12 concluso)** (frame di GT-F: e12, E12 concluso: True, impronta di cgt.py+gt.py `1a9807b0d0ec`).

| parametro | valore |
|---|---|
| soggetti | 100 (BFM REMESH `original`, held-out della v1) |
| GT su disco | F, F_rig_ls, F_rig_rob, S, EDM, EDM_s, unified, F_centered, F_json, size_only, height_only, maxabs |
| tipi | F_vs_S, F_vs_maxabs, S_vs_maxabs, EDM_vs_F |
| pool completo | 485100 triplette (A, {B, C}) |
| margine relativo | >= 0.10 su entrambe le GT in disaccordo; estrazione fra le 5 x 60 col margine minimo piu' grande, <= 15 comparse per soggetto nei test |
| controlli | 30, unanimi su F,S,EDM,EDM_s,unified,maxabs con margine >= 0.40 |
| prova | 6 triplette unanimi (stessa regola, disgiunte dai controlli), escluse dall'analisi |
| test | 240 |

## Tipi di disaccordo

| tipo | disponibili (margine ok) | scelte | margine minimo: mediana / min nelle scelte |
|---|---:|---:|---|
| `F_vs_S` | 958 | 60 | 0.138 / 0.129 |
| `F_vs_maxabs` | 1029 | 60 | 0.133 / 0.123 |
| `S_vs_maxabs` | 2643 | 60 | 0.167 / 0.154 |
| `EDM_vs_F` | 33992 | 60 | 0.378 / 0.351 |

## Come votano le altre GT dentro ogni tipo

Quota delle triplette del tipo in cui la GT da' la stessa risposta della PRIMA GT del tipo (X in `X_vs_Y`); 1 = sta con X, 0 = sta con Y. Dice che cosa si confronta davvero in ogni strato.

| GT | `F_vs_S` | `F_vs_maxabs` | `S_vs_maxabs` | `EDM_vs_F` |
|---|---:|---:|---:|---:|
| F | 1.00 | 1.00 | 0.80 | 0.00 |
| F_rig_ls | 1.00 | 0.80 | 0.47 | 0.65 |
| F_rig_rob | 0.97 | 0.75 | 0.42 | 0.67 |
| S | 0.00 | 0.95 | 1.00 | 0.00 |
| EDM | 1.00 | 0.63 | 0.12 | 1.00 |
| EDM_s | 0.40 | 0.53 | 0.58 | 0.82 |
| unified | 0.27 | 0.72 | 0.83 | 0.48 |
| F_centered | 1.00 | 0.83 | 0.57 | 0.25 |
| F_json | 1.00 | 1.00 | 0.80 | 0.00 |
| size_only | 1.00 | 0.77 | 0.07 | 0.97 |
| height_only | 0.85 | 0.75 | 0.40 | 0.85 |
| maxabs | 0.87 | 0.00 | 0.00 | 0.00 |

## Margini relativi nelle triplette scelte (mediana)

| GT | test | controlli |
|---|---:|---:|
| F | 0.147 | 0.823 |
| F_rig_ls | 0.156 | 0.822 |
| F_rig_rob | 0.158 | 0.832 |
| S | 0.171 | 0.817 |
| EDM | 0.421 | 0.884 |
| EDM_s | 0.192 | 0.935 |
| unified | 0.136 | 0.811 |
| F_centered | 0.148 | 0.829 |
| F_json | 0.147 | 0.823 |
| size_only | 1.171 | 1.012 |
| height_only | 0.920 | 1.360 |
| maxabs | 0.170 | 0.832 |

## Copertura dei soggetti

- 100 soggetti distinti sui 100; comparse per soggetto: min 1, mediana 7, max 18.

## Pacchetto

- 100 strisce JPEG (frontale, 3/4, profilo) in `docs/human_study_v2/img/` (3.6 MB), camera unica 1.959 px/mm.
- `triplets.json` (con metriche, per l'analisi) e `docs/human_study_v2/triplets.js` (senza).
