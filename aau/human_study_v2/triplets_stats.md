# Triplette dello studio umano v2

Generato il 2026-10-09T11:46:50+00:00 da `aau/human_study_v2/select_triplets_v2.py`, seed 1234.

**Stato delle GT: definitivo (E12 concluso)** (frame di GT-F: e12, E12 concluso: True, impronta di cgt.py+gt.py `1a9807b0d0ec`).

| parametro | valore |
|---|---|
| identita' | 100 GNM Head (100 campionate, seed 1234) |
| GT su disco | F, S, EDM, EDM_s, maxabs, unified, F_rig_ls, F_pure, size_only, height_only |
| strati (prove per sessione) | F_vs_S (36), S_vs_maxabs (24) |
| pool completo | 485100 triplette (A, {B, C}) |
| margine relativo | >= 0.10 su entrambe le GT in disaccordo; estrazione fra le 3 x n col margine minimo piu' grande, <= 15 comparse per soggetto nei test |
| controlli | 30, unanimi su F,S,EDM,unified,maxabs con margine >= 0.40 |
| prova | 6 triplette unanimi (stessa regola, disgiunte dai controlli), escluse dall'analisi |
| test | 200 |

## Tipi di disaccordo

| tipo | disponibili (margine ok) | scelte | margine minimo: mediana / min nelle scelte |
|---|---:|---:|---|
| `S_vs_maxabs` | 37523 | 80 | 0.418 / 0.394 |
| `F_vs_S` | 57506 | 120 | 0.479 / 0.442 |

## Come votano le altre GT dentro ogni tipo

Quota delle triplette del tipo in cui la GT da' la stessa risposta della PRIMA GT del tipo (X in `X_vs_Y`); 1 = sta con X, 0 = sta con Y. Dice che cosa si confronta davvero in ogni strato.

| GT | `F_vs_S` | `S_vs_maxabs` |
|---|---:|---:|
| F | 1.00 | 0.84 |
| S | 0.00 | 1.00 |
| EDM | 1.00 | 0.57 |
| EDM_s | 0.00 | 0.96 |
| maxabs | 0.03 | 0.00 |
| unified | 0.00 | 1.00 |
| F_rig_ls | 1.00 | 0.80 |
| F_pure | 1.00 | 0.79 |
| size_only | 1.00 | 0.42 |
| height_only | 1.00 | 0.42 |

## Margini relativi nelle triplette scelte (mediana)

| GT | test | controlli |
|---|---:|---:|
| F | 0.486 | 0.981 |
| S | 0.486 | 0.921 |
| EDM | 0.851 | 1.115 |
| EDM_s | 0.507 | 0.897 |
| maxabs | 0.434 | 0.910 |
| unified | 0.487 | 0.921 |
| F_rig_ls | 0.497 | 0.993 |
| F_pure | 0.462 | 0.904 |
| size_only | 1.383 | 1.446 |
| height_only | 1.130 | 1.268 |

## Copertura dei soggetti

- 93 soggetti distinti sui 100; comparse per soggetto: min 1, mediana 7, max 19.

## Pacchetto

- 93 strisce JPEG (frontale, 3/4, profilo) in `docs/human_study_v2/img/` (2.9 MB), camera unica 1.745 px/mm.
- `triplets.json` (con metriche, per l'analisi) e `docs/human_study_v2/triplets.js` (senza).
