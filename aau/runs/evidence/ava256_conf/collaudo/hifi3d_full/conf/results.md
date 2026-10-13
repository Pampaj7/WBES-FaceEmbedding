# Prova confermativa: regole R1-R7

Generato da `aau/ava256/conf_eval.py` (spec `aau/ava256/spec/hifi3d_full.json`). Seme delle repliche 990708, 1000 repliche, 100 soggetti, righe 148500 (lettura confermativa: {'nocrop': 99000, 'crop': 49500, 'all_cross': 148500}). Lettura confermativa: righe fallite dei metodi per coppia = +inf; Holm per seme con m = 10.

| regola | seme | delta [IC 95%] | k (p) | Holm i / soglia / rifiuta | non risolto | esito |
|---|---|---|---|---|---|---|
| R1 | s1234 | +0.105 [+0.062, +0.154] | 0 (0.000) | 1 / 0.0025 / si | - | confermata |
| R1 | s2345 | +0.088 [+0.039, +0.138] | 0 (0.000) | 1 / 0.0025 / si | - | confermata |
| R2a | s1234 | +0.134 [+0.085, +0.188] | 0 (0.000) | 2 / 0.0028 / si | - | confermata |
| R2a | s2345 | +0.117 [+0.066, +0.171] | 0 (0.000) | 2 / 0.0028 / si | - | confermata |
| R2b | s1234 | +0.262 [+0.179, +0.350] | 0 (0.000) | 3 / 0.0031 / si | - | confermata |
| R2b | s2345 | +0.245 [+0.160, +0.334] | 0 (0.000) | 3 / 0.0031 / si | - | confermata |
| R3a | - | +0.284 [+0.187, +0.382] | 0 (0.000) | 4 / 0.0036 / si; 4 / 0.0036 / si | - | confermata |
| R3b | - | -0.009 [-0.036, +0.020] | 760 (0.760) | 9 / 0.0125 / no; 9 / 0.0125 / no | - | non confermata |
| R4 | s1234 | +0.246 [+0.200, +0.295] | 0 (0.000) | 5 / 0.0042 / si | - | confermata |
| R4 | s2345 | +0.136 [+0.088, +0.185] | 0 (0.000) | 5 / 0.0042 / si | - | confermata |
| R5a | s1234 | +0.041 [-0.006, +0.091] | 46 (0.046) | 7 / 0.0063 / no | - | non confermata |
| R5a | s2345 | +0.032 [-0.026, +0.087] | 116 (0.116) | 7 / 0.0063 / no | - | non confermata |
| R5b | s1234 | +0.037 [-0.019, +0.093] | 105 (0.105) | 8 / 0.0083 / no | - | non confermata |
| R5b | s2345 | +0.028 [-0.035, +0.092] | 167 (0.167) | 8 / 0.0083 / no | - | non confermata |
| R6 | s1234 | - | - | - | - | non calcolata (collaudo) |
| R6 | s2345 | - | - | - | - | non calcolata (collaudo) |
| R7 | - | +0.281 [+0.194, +0.371] | 0 (0.000) | 6 / 0.0050 / si; 6 / 0.0050 / si | - | confermata |

Delta della versione mascherata, NoW metrical contro d_F (sez. 9.2) e Spearman dei protocolli: `deltas.csv`, `spearman.csv` (descrittivi, non si leggono).

