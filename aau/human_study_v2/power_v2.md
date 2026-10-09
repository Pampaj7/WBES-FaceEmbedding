# Studio umano v2: calcolo di potenza (simulazione)

Generato da `aau/human_study_v2/power_v2.py`, 4000 studi simulati per cella, seed 1234. 9 prove per strato e partecipante, 60 triplette per strato.

q = quota marginale delle risposte che stanno con X nello strato `X_vs_Y`; differenziale d'accordo fra le due GT nello strato = 2q - 1. Potenza del test a segni ribaltati per partecipante, a due code.

## sd fra partecipanti 0.5 logit, sd fra triplette 0.0 logit

| q | differenziale | N per 80% (alfa 0.05) | N per 80% (alfa 0.0125) | N per 90% (alfa 0.0125) |
|---:|---:|---:|---:|---:|
| 0.550 | +0.10 | 140 | 200 | 250 |
| 0.575 | +0.15 | 60 | 90 | 120 |
| 0.600 | +0.20 | 35 | 50 | 70 |
| 0.650 | +0.30 | 20 | 25 | 30 |

## sd fra partecipanti 0.5 logit, sd fra triplette 0.8 logit

| q | differenziale | N per 80% (alfa 0.05) | N per 80% (alfa 0.0125) | N per 90% (alfa 0.0125) |
|---:|---:|---:|---:|---:|
| 0.550 | +0.10 | 200 | 300 | > 400 |
| 0.575 | +0.15 | 70 | 100 | 160 |
| 0.600 | +0.20 | 35 | 50 | 70 |
| 0.650 | +0.30 | 15 | 25 | 30 |

## Taratura sulla v1

7 partecipanti della v1 (36 test ciascuno, triplette dove le metriche litigano). Accordo per risposta e deviazione fra partecipanti oltre la binomiale:

| metrica | accordo | sd fra partecipanti (prob.) | (logit) |
|---|---:|---:|---:|
| gt | 0.488 | 0.095 | 0.38 |
| chamfer | 0.433 | 0.000 | 0.00 |
| lpips | 0.619 | 0.054 | 0.23 |
| latent | 0.492 | 0.053 | 0.21 |

