# Prova confermativa: regole R1-R7

Generato da `aau/ava256/conf_eval.py` (spec `aau/ava256/spec/ava256.json`). Seme delle repliche 20261011, 1000 repliche, 200 soggetti, righe 597000 (lettura confermativa: {'nocrop': 398000, 'crop': 199000, 'all_cross': 597000}). Lettura confermativa: righe fallite dei metodi per coppia = +inf; Holm per seme con m = 9 (R4, gruppo crop: descrittiva, fuori dalla famiglia, in `deltas.csv`).

| regola | seme | delta [IC 95%] | k (p) | Holm i / soglia / rifiuta | non risolto | esito |
|---|---|---|---|---|---|---|
| R1 | s1234 | +0.017 [-0.038, +0.069] | 257 (0.257) | 7 / 0.0083 / no | - | non confermata |
| R1 | s2345 | +0.023 [-0.031, +0.079] | 197 (0.197) | 8 / 0.0125 / no | - | non confermata |
| R2a | s1234 | +0.126 [+0.057, +0.194] | 1 (0.001) | 6 / 0.0063 / si | - | confermata |
| R2a | s2345 | +0.133 [+0.065, +0.197] | 0 (0.000) | 1 / 0.0028 / si | - | confermata |
| R2b | s1234 | +0.197 [+0.136, +0.260] | 0 (0.000) | 1 / 0.0028 / si | righe non finite 9.06% > 5% | non risolta |
| R2b | s2345 | +0.203 [+0.141, +0.269] | 0 (0.000) | 2 / 0.0031 / si | righe non finite 9.06% > 5% | non risolta |
| R3a | - | +0.119 [+0.061, +0.181] | 0 (0.000) | 2 / 0.0031 / si; 3 / 0.0036 / si | - | confermata |
| R3b | - | +0.105 [+0.063, +0.145] | 0 (0.000) | 3 / 0.0036 / si; 4 / 0.0042 / si | righe non finite 9.06% > 5% | non risolta |
| R5a | s1234 | -0.028 [-0.076, +0.013] | 903 (0.903) | 9 / 0.0250 / no | - | non confermata |
| R5a | s2345 | -0.031 [-0.078, +0.011] | 916 (0.916) | 9 / 0.0250 / no | - | non confermata |
| R5b | s1234 | +0.159 [+0.101, +0.214] | 0 (0.000) | 4 / 0.0042 / si | righe non finite 9.06% > 5% | non risolta |
| R5b | s2345 | +0.157 [+0.098, +0.211] | 0 (0.000) | 5 / 0.0050 / si | righe non finite 9.06% > 5% | non risolta |
| R6 | s1234 | -0.000 [-0.024, +0.026] | 492 (0.492) | 8 / 0.0125 / no | - | non confermata |
| R6 | s2345 | +0.020 [-0.004, +0.046] | 50 (0.050) | 7 / 0.0083 / no | - | non confermata |
| R7 | - | +0.127 [+0.088, +0.172] | 0 (0.000) | 5 / 0.0050 / si; 6 / 0.0063 / si | - | confermata |

Delta della versione mascherata, NoW metrical contro d_F (sez. 9.2) e Spearman dei protocolli: `deltas.csv`, `spearman.csv` (descrittivi, non si leggono).

