# Triplette dello studio umano v2

Generato il 2026-10-09T13:38:24+00:00 da `aau/human_study_v2/select_triplets_v2.py`, seed 1234.

**Stato delle GT: definitivo (E12 concluso)** (frame di GT-F: e12, E12 concluso: True, impronta di cgt.py+gt.py `1a9807b0d0ec`).

| parametro | valore |
|---|---|
| identita' | 100 GNM Head (100 campionate, seed 1234) |
| GT su disco | F, S, EDM, EDM_s, maxabs, unified, F_rig_ls, F_pure, size_only, height_only |
| strati (prove per sessione) | F_vs_S (12), F_vs_size (3), S_vs_maxabs (3) |
| pool completo | 485100 triplette (A, {B, C}) |
| vincoli | F_vs_S: nessuno; F_vs_size: |d_size_only(A,B) - d_size_only(A,C)| >= 0.02; S_vs_maxabs: |d_size_only(A,B) - d_size_only(A,C)| <= 0.01, bilanciato 50/50 su F, size_only |
| margine relativo | >= 0.10 su entrambe le GT in disaccordo; estrazione fra le 3 x n col margine minimo piu' grande, <= 25 comparse per soggetto nei test |
| controlli | 30, unanimi su F,S,EDM,unified,maxabs con margine >= 0.40 |
| prova | 6 triplette unanimi (stessa regola, disgiunte dai controlli), escluse dall'analisi |
| test | 360 |

## Tipi di disaccordo

| tipo | disponibili (margine ok) | scelte | margine minimo: mediana / min nelle scelte |
|---|---:|---:|---|
| `S_vs_maxabs` | 6038 | 80 | 0.260 / 0.108 |
| `F_vs_size` | 16853 | 80 | 0.587 / 0.550 |
| `F_vs_S` | 57506 | 200 | 0.459 / 0.432 |

## Come votano le altre GT dentro ogni tipo

Quota delle triplette del tipo in cui la GT da' la stessa risposta della PRIMA GT del tipo (X in `X_vs_Y`); 1 = sta con X, 0 = sta con Y. Dice che cosa si confronta davvero in ogni strato.

| GT | `F_vs_S` | `F_vs_size` | `S_vs_maxabs` |
|---|---:|---:|---:|
| F | 1.00 | 1.00 | 0.50 |
| S | 0.00 | 1.00 | 1.00 |
| EDM | 1.00 | 0.99 | 0.49 |
| EDM_s | 0.00 | 1.00 | 0.88 |
| maxabs | 0.05 | 1.00 | 0.00 |
| unified | 0.00 | 1.00 | 1.00 |
| F_rig_ls | 1.00 | 1.00 | 0.54 |
| F_pure | 1.00 | 1.00 | 0.59 |
| size_only | 1.00 | 0.00 | 0.50 |
| height_only | 0.99 | 0.54 | 0.36 |

## Margini relativi nelle triplette scelte (mediana)

| GT | test | controlli |
|---|---:|---:|
| F | 0.500 | 0.995 |
| S | 0.481 | 0.928 |
| EDM | 0.746 | 1.093 |
| EDM_s | 0.531 | 0.995 |
| maxabs | 0.402 | 0.889 |
| unified | 0.483 | 0.928 |
| F_rig_ls | 0.501 | 1.003 |
| F_pure | 0.467 | 0.948 |
| size_only | 1.206 | 1.215 |
| height_only | 1.009 | 1.322 |

## Copertura dei soggetti

- 100 soggetti distinti sui 100; comparse per soggetto: min 1, mediana 10, max 32.

## Pacchetto

- 100 strisce JPEG (frontale, 3/4, profilo) in `docs/human_study_v2/img/` (4.6 MB), camera unica 2.327 px/mm.
- `triplets.json` (con metriche, per l'analisi) e `docs/human_study_v2/triplets.js` (senza).
