# Studio umano v2: calcolo di potenza (simulazione)

Generato da `aau/human_study_v2/power_v2.py`, 4000 studi simulati per cella, seed 1234. Strati (prove per partecipante / triplette): F_vs_S 36/120, S_vs_maxabs 24/80.

q = quota marginale delle risposte che stanno con X nello strato `X_vs_Y`; differenziale d'accordo fra le due GT nello strato = 2q - 1. Potenza del test a segni ribaltati per partecipante, a due code.

## F_vs_S: 36 prove; sd fra partecipanti 0.5 logit, fra triplette 0.0 logit

| q | differenziale | N per 80% (alfa 0.05) | N per 80% (alfa 0.025) | N per 90% (alfa 0.025) |
|---:|---:|---:|---:|---:|
| 0.550 | +0.10 | 70 | 100 | 120 |
| 0.575 | +0.15 | 35 | 40 | 50 |
| 0.600 | +0.20 | 18 | 25 | 30 |
| 0.650 | +0.30 | 10 | 12 | 15 |
| 0.700 | +0.40 | 8 | 10 | 10 |

## F_vs_S: 36 prove; sd fra partecipanti 0.5 logit, fra triplette 0.8 logit

| q | differenziale | N per 80% (alfa 0.05) | N per 80% (alfa 0.025) | N per 90% (alfa 0.025) |
|---:|---:|---:|---:|---:|
| 0.550 | +0.10 | 70 | 100 | 150 |
| 0.575 | +0.15 | 30 | 40 | 50 |
| 0.600 | +0.20 | 18 | 25 | 30 |
| 0.650 | +0.30 | 10 | 12 | 15 |
| 0.700 | +0.40 | 8 | 8 | 10 |

## S_vs_maxabs: 24 prove; sd fra partecipanti 0.5 logit, fra triplette 0.0 logit

| q | differenziale | N per 80% (alfa 0.05) | N per 80% (alfa 0.025) | N per 90% (alfa 0.025) |
|---:|---:|---:|---:|---:|
| 0.550 | +0.10 | 80 | 100 | 150 |
| 0.575 | +0.15 | 40 | 50 | 60 |
| 0.600 | +0.20 | 20 | 30 | 35 |
| 0.650 | +0.30 | 12 | 15 | 18 |
| 0.700 | +0.40 | 8 | 10 | 12 |

## S_vs_maxabs: 24 prove; sd fra partecipanti 0.5 logit, fra triplette 0.8 logit

| q | differenziale | N per 80% (alfa 0.05) | N per 80% (alfa 0.025) | N per 90% (alfa 0.025) |
|---:|---:|---:|---:|---:|
| 0.550 | +0.10 | 100 | 120 | > 200 |
| 0.575 | +0.15 | 40 | 50 | 70 |
| 0.600 | +0.20 | 20 | 25 | 35 |
| 0.650 | +0.30 | 10 | 12 | 18 |
| 0.700 | +0.40 | 8 | 10 | 10 |

## Taratura sulla v1

7 partecipanti della v1 (36 test ciascuno, triplette dove le metriche litigano). Accordo per risposta e deviazione fra partecipanti oltre la binomiale:

| metrica | accordo | sd fra partecipanti (prob.) | (logit) |
|---|---:|---:|---:|
| gt | 0.488 | 0.095 | 0.38 |
| chamfer | 0.433 | 0.000 | 0.00 |
| lpips | 0.619 | 0.054 | 0.23 |
| latent | 0.492 | 0.053 | 0.21 |

