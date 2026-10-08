# FaMoS nello spazio della GT unificata

95 soggetti (TRAIN 80, TEST 15), forma neutra di riferimento registrata (famos_subsample.py), regione unificata di 1478 vertici, mm.

## Distanza dalle medie dei domini (mm, mediana [p25, p75] sui soggetti)

| dominio | FaMoS tutti | FaMoS TRAIN | FaMoS TEST | media FaMoS -> media del dominio |
| --- | --- | --- | --- | --- |
| flame | 2.94 [2.57, 3.56] | 2.93 | 3.29 | 1.33 |
| bfm | 3.20 [2.79, 3.73] | 3.16 | 3.63 | 1.85 |
| ict | 3.14 [2.79, 3.78] | 3.18 | 3.04 | 1.74 |
| gnm | 2.84 [2.53, 3.39] | 2.82 | 3.03 | 1.14 |
| facescape | 3.90 [3.39, 4.51] | 3.87 | 3.99 | 2.87 |
| hifi3d | 3.85 [3.40, 4.47] | 3.84 | 3.90 | 2.84 |
| faceverse | 3.86 [3.47, 4.50] | 3.86 | 4.05 | 2.89 |
| multiface | 3.26 [2.87, 3.85] | 3.26 | 3.19 | 1.89 |

## Dispersione (mm)

| dominio | n | mediana a coppie | mediana del vicino piu' prossimo |
| --- | --- | --- | --- |
| **famos** | 95 | 3.83 | 2.33 |
| hifi3d | 500 | 4.31 | 2.51 |
| faceverse | 500 | 7.02 | 5.28 |
| bfm | 500 | 3.98 | 2.25 |
| ict | 55000 | 3.59 | 1.74 |
| gnm | 10100 | 4.16 | 2.00 |
| flame | 1000 | 3.90 | 2.09 |
| multiface | 13 | 4.29 | 3.34 |

## Split nello spazio

- Rumore della stima della neutra (meta' pari contro dispari dei primi fotogrammi): mediana 0.229 mm, massimo 0.592 mm.
- Persona di TEST -> TRAIN piu' vicina: mediana 2.38 mm, minimo 2.17 mm (9.5 volte il rumore mediano): nessun duplicato geometrico.
- GT dei 15 soggetti di TEST: mediana 4.28 mm, minimo fuori diagonale 2.18 mm, mm_per_unit 7.758.
