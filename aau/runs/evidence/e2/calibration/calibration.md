# E2, calibrazione della canonicalizzazione (domini di training, soggetti held-out)

Parametri: {"n_src": 4000, "n_src_coarse": 1000, "n_ref": 50000, "n_core": 3000, "core_radius": 1.1, "trim": 0.8, "trim_core": 0.9, "scale_bounds": [0.67, 1.5], "dist_cap": 0.5, "iters_coarse": 30, "iters_fine": 60, "tol": 1e-06, "keep_starts": 2, "ref_seed": 1234}; start: I, Rx180, Ry180, Rz180, Rx+90, Rx-90, Ry+90, Ry-90.
Soggetti: 40 per dominio (ICT, BFM, GNM held-out del run grande), 6 topologie.

## Mesh come sono (frame del dominio: ICT e GNM = identita', BFM = Rx180)

| dominio | topologia | n | residuo mediano | p99 | max | angolo dalla convenzione mediano | max | start scelti | s/mesh mediani |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bfm | crop | 40 | 0.0347 | 0.0428 | 0.0441 | 1.94 | 5.49 | {'Rx180': 28, 'Rx+90': 12} | 4.48 |
| bfm | down8k | 40 | 0.0234 | 0.0287 | 0.0290 | 1.88 | 5.49 | {'Rx180': 37, 'Rx+90': 3} | 4.29 |
| bfm | noisy | 40 | 0.0241 | 0.0289 | 0.0290 | 1.78 | 5.19 | {'Rx180': 37, 'Rx+90': 3} | 4.21 |
| bfm | original | 40 | 0.0234 | 0.0290 | 0.0291 | 1.77 | 5.34 | {'Rx180': 37, 'Rx+90': 3} | 4.31 |
| bfm | remesh | 40 | 0.0235 | 0.0297 | 0.0297 | 1.73 | 5.37 | {'Rx180': 36, 'Rx+90': 4} | 4.28 |
| bfm | up60k | 40 | 0.0230 | 0.0289 | 0.0291 | 1.70 | 5.44 | {'Rx180': 36, 'Rx+90': 4} | 4.31 |
| gnm | crop | 40 | 0.0226 | 0.0476 | 0.0553 | 1.40 | 6.21 | {'I': 29, 'Rx-90': 11} | 3.77 |
| gnm | down8k | 40 | 0.0213 | 0.0417 | 0.0489 | 1.33 | 5.20 | {'I': 38, 'Rx-90': 2} | 3.93 |
| gnm | noisy | 40 | 0.0216 | 0.0427 | 0.0492 | 1.43 | 5.23 | {'I': 39, 'Rx-90': 1} | 3.90 |
| gnm | original | 40 | 0.0211 | 0.0414 | 0.0487 | 1.31 | 5.32 | {'I': 37, 'Rx-90': 3} | 3.88 |
| gnm | remesh | 40 | 0.0212 | 0.0430 | 0.0505 | 1.33 | 5.00 | {'I': 36, 'Rx-90': 4} | 3.82 |
| gnm | up60k | 40 | 0.0211 | 0.0419 | 0.0493 | 1.33 | 5.26 | {'I': 37, 'Rx-90': 3} | 3.91 |
| ict | crop | 40 | 0.0240 | 0.0334 | 0.0343 | 2.33 | 6.54 | {'I': 40} | 5.50 |
| ict | down8k | 40 | 0.0255 | 0.0345 | 0.0346 | 2.36 | 5.93 | {'I': 40} | 5.91 |
| ict | noisy | 40 | 0.0257 | 0.0336 | 0.0336 | 2.35 | 6.28 | {'I': 40} | 5.95 |
| ict | original | 40 | 0.0255 | 0.0348 | 0.0349 | 2.31 | 6.04 | {'I': 40} | 5.90 |
| ict | remesh | 40 | 0.0247 | 0.0338 | 0.0339 | 2.41 | 6.30 | {'I': 40} | 5.80 |
| ict | up60k | 40 | 0.0255 | 0.0346 | 0.0349 | 2.46 | 6.28 | {'I': 40} | 5.89 |

## Consistenza: angolo fra la rotazione di una topologia e quella della `original` dello stesso soggetto

| dominio | topologia | mediana (gradi) | p95 | max |
| --- | --- | --- | --- | --- |
| bfm | crop | 1.02 | 2.53 | 2.83 |
| bfm | down8k | 0.32 | 1.63 | 3.09 |
| bfm | noisy | 0.44 | 1.20 | 1.58 |
| bfm | remesh | 0.30 | 1.45 | 2.92 |
| bfm | up60k | 0.25 | 1.60 | 2.80 |
| gnm | crop | 0.56 | 1.79 | 3.24 |
| gnm | down8k | 0.48 | 1.85 | 2.20 |
| gnm | noisy | 0.50 | 1.51 | 4.64 |
| gnm | remesh | 0.44 | 1.18 | 2.00 |
| gnm | up60k | 0.41 | 1.23 | 1.50 |
| ict | crop | 0.72 | 2.03 | 3.01 |
| ict | down8k | 0.39 | 0.86 | 2.69 |
| ict | noisy | 0.54 | 1.25 | 2.70 |
| ict | remesh | 0.44 | 1.38 | 3.06 |
| ict | up60k | 0.46 | 0.88 | 2.30 |

## Robustezza: mesh ruotate (flip casuale fra i 4 di 180 gradi, poi fino a 30 gradi attorno a un asse casuale)

Errore = angolo fra (R stimata sulla mesh ruotata) x (rotazione applicata) e R stimata sulla mesh com'e'.

| dominio | topologia | n | errore mediano (gradi) | p95 | max | > 5 gradi | residuo mediano |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bfm | crop | 40 | 0.58 | 3.17 | 4.12 | 0 | 0.0347 |
| bfm | down8k | 40 | 0.27 | 2.63 | 2.94 | 0 | 0.0234 |
| bfm | original | 40 | 0.25 | 2.99 | 4.66 | 0 | 0.0236 |
| gnm | crop | 40 | 0.65 | 3.49 | 4.19 | 0 | 0.0231 |
| gnm | down8k | 40 | 0.82 | 7.58 | 8.84 | 6 | 0.0219 |
| gnm | original | 40 | 1.12 | 6.98 | 13.11 | 7 | 0.0217 |
| ict | crop | 40 | 0.24 | 6.47 | 7.29 | 4 | 0.0243 |
| ict | down8k | 40 | 0.33 | 4.21 | 5.70 | 1 | 0.0258 |
| ict | original | 40 | 0.26 | 4.16 | 7.50 | 2 | 0.0256 |

Falliti nella robustezza (> 5 gradi): 20 su 360; residuo min dei falliti 0.0201, sopra la soglia: 0.

Mesh come sono con angolo dalla convenzione >= 15.0 gradi: 0 su 720.

Faccia media GNM (held-out, original) contro faccia media ICT: {"angle_from_identity_deg": 0.6667901701645678, "residual": 0.017146972852915936, "start": "I"}.

## Soglia di fallimento dichiarata

T = 2 x p99 del residuo trimmed delle canonicalizzazioni corrette = 2 x 0.0406 = **0.0811** (unita': raggio RMS della faccia media).
Tempo per mesh (calibrazione, 1 thread, 8 start): mediana 4.47 s, p95 6.52 s.
