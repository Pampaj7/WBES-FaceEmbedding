# E10: tabelle generate da aau/evidence/e10_gpu_ops/summarize.py

## Correttezza: operatori GPU contro CPU (ops_areanorm) su 60 mesh

Campione: BFM e ICT x {original, remesh, down8k, up60k, noisy, crop} x 5 identita'. `cpu_v0` = stessa CPU con un altro vettore iniziale di eigsh (il rumore del solo risolutore CPU). Angoli fra sottospazi M-ortonormali; 'primi k-8' = angolo massimo dei primi k-8 vettori GPU dentro lo span dei k CPU (insensibile al mescolamento al bordo dello spettro). Embedding: checkpoint e108, loader congelato, CPU; dz = |z_var - z_cpu|.

### k = 128, GPU Tesla V100-SXM3-32GB (check_k128_Tesla_V100-SXM3-32GB.json)

Distanza mediana fra coppie di embedding CPU 2.251; distanza dal vicino piu' prossimo: minima 0.092, mediana 0.170.

| variante | autovalori i>=1: err. rel. max (mediana) | lambda_0: err. ass. max | tutti identici in fp32 | angolo max gradi | primi k-8 max gradi | struttura L/grad identica | L, massa max rel | grad max rel | dz max | dz mediana | dz max / dist. mediana | dz max / dist. min vicino | cos min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu_v0 | 0.0e+00 (0.0e+00) | 0.0e+00 | 60/60 | 1.74e-02 | 1.74e-02 | True | 0.0e+00, 0.0e+00 | 0.0e+00 | **1.14e-06** | 9.10e-07 | 5.1e-07 | 1.2e-05 | 0.99999982 |
| gpu_bk | 0.0e+00 (0.0e+00) | 4.6e-09 | 47/60 | 1.74e-02 | 1.74e-02 | True | 0.0e+00, 0.0e+00 | 1.7e-05 | **1.31e-06** | 1.03e-06 | 5.8e-07 | 1.4e-05 | 0.99999988 |
| gpu_bk_1pass | 1.1e-03 (0.0e+00) | 3.1e-09 | 0/60 | 1.48e+01 | 1.33e+00 | True | 0.0e+00, 0.0e+00 | 1.7e-05 | **1.31e-03** | 3.47e-04 | 5.8e-04 | 1.4e-02 | 0.99999988 |
| gpu_bk_loose | 1.3e-01 (4.1e-05) | 4.6e-09 | 0/60 | 8.99e+01 | 4.47e+01 | True | 0.0e+00, 0.0e+00 | 1.7e-05 | **4.56e-02** | 3.09e-02 | 2.0e-02 | 5.0e-01 | 0.99995679 |
| gpu_si | 0.0e+00 (0.0e+00) | 4.0e-09 | 50/60 | 1.74e-02 | 1.74e-02 | True | 0.0e+00, 0.0e+00 | 1.7e-05 | **1.27e-06** | 1.01e-06 | 5.7e-07 | 1.4e-05 | 0.99999988 |

Per categoria (gpu_bk):

| categoria | V mediano | grad max rel | angolo max gradi | dz max | dz max cpu_v0 | iterazioni (solve, colonne) | batch s (B=5; il primo include il riscaldamento) |
|---|---|---|---|---|---|---|---|
| bfm_crop | 21006 | 7.2e-06 | 1.08e-02 | 1.14e-06 | 1.12e-06 | 32, 512 | 16.65 |
| bfm_down8k | 8137 | 1.7e-05 | 1.74e-02 | 1.17e-06 | 1.02e-06 | 32, 512 | 0.32 |
| bfm_noisy | 23470 | 1.7e-05 | 7.98e-03 | 1.31e-06 | 1.14e-06 | 32, 512 | 0.69 |
| bfm_original | 23470 | 2.9e-07 | 1.15e-02 | 1.20e-06 | 1.09e-06 | 32, 512 | 0.68 |
| bfm_remesh | 16502 | 6.7e-07 | 8.47e-03 | 1.19e-06 | 1.13e-06 | 32, 512 | 0.50 |
| bfm_up60k | 60432 | 1.7e-06 | 7.14e-03 | 1.07e-06 | 1.13e-06 | 32, 512 | 1.70 |
| ict_crop | 8655 | 2.4e-07 | 1.22e-02 | 9.79e-07 | 9.03e-07 | 32, 512 | 0.28 |
| ict_down8k | 3281 | 2.1e-07 | 1.30e-02 | 1.04e-06 | 8.46e-07 | 32, 512 | 0.15 |
| ict_noisy | 9409 | 1.3e-05 | 1.15e-02 | 9.79e-07 | 9.01e-07 | 32, 512 | 0.31 |
| ict_original | 9409 | 2.1e-07 | 1.19e-02 | 9.81e-07 | 9.56e-07 | 32, 512 | 0.29 |
| ict_remesh | 6600 | 2.8e-07 | 1.18e-02 | 1.03e-06 | 9.11e-07 | 32, 512 | 0.23 |
| ict_up60k | 24125 | 2.4e-07 | 8.38e-03 | 1.05e-06 | 1.03e-06 | 32, 512 | 0.71 |

### k = 64, GPU Tesla V100-SXM3-32GB (check_k64_Tesla_V100-SXM3-32GB.json)

Distanza mediana fra coppie di embedding CPU 2.232; distanza dal vicino piu' prossimo: minima 0.139, mediana 0.208.

| variante | autovalori i>=1: err. rel. max (mediana) | lambda_0: err. ass. max | tutti identici in fp32 | angolo max gradi | primi k-8 max gradi | struttura L/grad identica | L, massa max rel | grad max rel | dz max | dz mediana | dz max / dist. mediana | dz max / dist. min vicino | cos min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu_v0 | 0.0e+00 (0.0e+00) | 0.0e+00 | 60/60 | 1.61e-02 | 1.57e-02 | True | 0.0e+00, 0.0e+00 | 0.0e+00 | **1.22e-06** | 8.32e-07 | 5.5e-07 | 8.8e-06 | 0.99999994 |
| gpu_bk | 0.0e+00 (0.0e+00) | 4.0e-09 | 46/60 | 1.62e-02 | 1.58e-02 | True | 0.0e+00, 0.0e+00 | 1.7e-05 | **1.40e-06** | 1.04e-06 | 6.3e-07 | 1.0e-05 | 0.99999988 |
| gpu_bk_1pass | 2.2e-02 (3.1e-06) | 3.0e-09 | 0/60 | 8.90e+01 | 7.14e+00 | True | 0.0e+00, 0.0e+00 | 1.7e-05 | **3.47e-02** | 8.49e-03 | 1.6e-02 | 2.5e-01 | 0.99997729 |
| gpu_bk_loose | 3.5e-01 (6.8e-03) | 4.1e-09 | 0/60 | 8.99e+01 | 3.95e+01 | True | 0.0e+00, 0.0e+00 | 1.7e-05 | **1.71e-01** | 1.11e-01 | 7.6e-02 | 1.2e+00 | 0.99940586 |
| gpu_si | 0.0e+00 (0.0e+00) | 4.9e-09 | 50/60 | 1.61e-02 | 1.57e-02 | True | 0.0e+00, 0.0e+00 | 1.7e-05 | **1.34e-06** | 1.07e-06 | 6.0e-07 | 9.6e-06 | 0.99999988 |

Per categoria (gpu_bk):

| categoria | V mediano | grad max rel | angolo max gradi | dz max | dz max cpu_v0 | iterazioni (solve, colonne) | batch s (B=5; il primo include il riscaldamento) |
|---|---|---|---|---|---|---|---|
| bfm_crop | 21006 | 7.2e-06 | 8.80e-03 | 1.21e-06 | 1.12e-06 | 20, 320 | 2.13 |
| bfm_down8k | 8137 | 1.7e-05 | 1.62e-02 | 1.12e-06 | 1.06e-06 | 20, 320 | 0.21 |
| bfm_noisy | 23470 | 1.7e-05 | 6.65e-03 | 1.22e-06 | 1.04e-06 | 20, 320 | 0.48 |
| bfm_original | 23470 | 2.9e-07 | 9.70e-03 | 1.24e-06 | 1.16e-06 | 20, 320 | 0.47 |
| bfm_remesh | 16502 | 6.7e-07 | 7.17e-03 | 1.40e-06 | 1.11e-06 | 20, 320 | 0.35 |
| bfm_up60k | 60432 | 1.7e-06 | 5.40e-03 | 1.21e-06 | 1.22e-06 | 20, 320 | 1.17 |
| ict_crop | 8655 | 2.4e-07 | 1.09e-02 | 1.04e-06 | 8.57e-07 | 20, 320 | 0.20 |
| ict_down8k | 3281 | 2.1e-07 | 1.15e-02 | 1.04e-06 | 1.49e-08 | 20, 320 | 0.11 |
| ict_noisy | 9409 | 1.3e-05 | 9.59e-03 | 1.08e-06 | 8.48e-07 | 20, 320 | 0.20 |
| ict_original | 9409 | 2.1e-07 | 9.60e-03 | 1.04e-06 | 8.39e-07 | 20, 320 | 0.20 |
| ict_remesh | 6600 | 2.8e-07 | 1.01e-02 | 9.83e-07 | 9.64e-07 | 20, 320 | 0.18 |
| ict_up60k | 24125 | 2.4e-07 | 6.92e-03 | 1.04e-06 | 9.20e-07 | 20, 320 | 0.48 |

## Pipeline completa sugli stessi campioni di E9

Lettura e normalizzazione (thread), batch per categoria sulla GPU (piu' lavoratori, uno stream ciascuno), npz scritti su /tmp (thread) e cancellati. CPU = il migliore di E9 sullo stesso campione e lo stesso k (nodo L40S, 100 CPU logiche, build_grad vettorizzato).

| GPU | file | campione | k | lavoratori GPU | CPU nel job | mesh | wall s | GPU mesh/s | convergenti | picco GB | CPU nodo E9 mesh/s (config.) | GPU / nodo CPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tesla V100-SXM3-32GB | e2e_bk_Tesla_V100-SXM3-32GB.json | full200 | 64 | 2 | 16 | 200 | 10.0 | **20.0** | 200/200 | 9.6 | 24.0 (d_vec_k64_full_P64_f2) | 0.83x |
| Tesla V100-SXM3-32GB | e2e_bk_Tesla_V100-SXM3-32GB.json | full200 | 128 | 2 | 16 | 200 | 14.5 | **13.8** | 200/200 | 14.1 | 9.3 (d_vec_k128_full_P100) | 1.49x |
| Tesla V100-SXM3-32GB | e2e_bk_Tesla_V100-SXM3-32GB.json | le10k400 | 64 | 2 | 16 | 400 | 8.6 | **46.3** | 400/400 | 14.4 | 126.1 (d_vec_k64_le10k_P64_f2) | 0.37x |
| Tesla V100-SXM3-32GB | e2e_bk_Tesla_V100-SXM3-32GB.json | le10k400 | 128 | 2 | 16 | 400 | 11.9 | **33.5** | 400/400 | 16.7 | 42.8 (d_vec_k128_le10k_P32_f2) | 0.78x |
| Tesla V100-SXM3-32GB | e2e_bk_Tesla_V100-SXM3-32GB_w3.json | full200 | 64 | 3 | 16 | 200 | 8.7 | **22.9** | 200/200 | 13.9 | 24.0 (d_vec_k64_full_P64_f2) | 0.95x |
| Tesla V100-SXM3-32GB | e2e_bk_Tesla_V100-SXM3-32GB_w3.json | full200 | 128 | 3 | 16 | 200 | 12.2 | **16.4** | 200/200 | 20.6 | 9.3 (d_vec_k128_full_P100) | 1.78x |
| Tesla V100-SXM3-32GB | e2e_bk_Tesla_V100-SXM3-32GB_w3.json | le10k400 | 64 | 3 | 16 | 400 | 7.1 | **56.2** | 400/400 | 15.0 | 126.1 (d_vec_k64_le10k_P64_f2) | 0.45x |
| Tesla V100-SXM3-32GB | e2e_bk_Tesla_V100-SXM3-32GB_w3.json | le10k400 | 128 | 3 | 16 | 400 | 10.5 | **38.2** | 400/400 | 24.7 | 42.8 (d_vec_k128_le10k_P32_f2) | 0.89x |
| Tesla V100-SXM3-32GB | e2e_bk_Tesla_V100-SXM3-32GB_w4c32.json | full200 | 64 | 4 | 32 | 200 | 8.3 | **24.2** | 200/200 | 18.3 | 24.0 (d_vec_k64_full_P64_f2) | 1.01x |
| Tesla V100-SXM3-32GB | e2e_bk_Tesla_V100-SXM3-32GB_w4c32.json | full200 | 128 | 4 | 32 | 200 | 13.0 | **15.4** | 200/200 | 27.2 | 9.3 (d_vec_k128_full_P100) | 1.66x |
| Tesla V100-SXM3-32GB | e2e_bk_Tesla_V100-SXM3-32GB_w4c32.json | le10k400 | 64 | 4 | 32 | 400 | 7.5 | **53.2** | 400/400 | 18.0 | 126.1 (d_vec_k64_le10k_P64_f2) | 0.42x |
| Tesla V100-SXM3-32GB | e2e_bk_Tesla_V100-SXM3-32GB_w4c32.json | le10k400 | 128 | 4 | 32 | 400 | 11.4 | **35.0** | 400/400 | 30.3 | 42.8 (d_vec_k128_le10k_P32_f2) | 0.82x |

## Classi omogenee: Tesla V100-SXM3-32GB (job 1061877, variante bk, classes_bk_Tesla_V100-SXM3-32GB.json)

Mesh gia' lette e normalizzate in RAM; tempo = geometria + autovettori + copia su host, media di 3 batch dopo uno di riscaldamento; un solo stream. Picco = memoria usata sul dispositivo (torch + cuDSS + contesto). CPU = stima E9 per la sola categoria (P / t medio, nodo L40S, 100 CPU logiche).

| classe | V | k | latenza B=1 ms | picco B=1 GB | migliore mesh/s (B) | picco GB al migliore | B max (OOM a) | CPU nodo E9 mesh/s | GPU / CPU |
|---|---|---|---|---|---|---|---|---|---|
| ict_down8k | 3279 | 64 | 52 | 0.57 | **62.8** (32) | 2.7 | 64 (-) | 379.4 | 0.17x |
| ict_down8k | 3279 | 128 | 63 | 0.60 | **43.8** (16) | 2.1 | 64 (-) | 119.2 | 0.37x |
| ict_original | 9409 | 64 | 71 | 0.73 | **29.1** (8) | 1.9 | 64 (-) | 73.1 | 0.40x |
| ict_original | 9409 | 128 | 97 | 0.84 | **19.9** (8) | 2.9 | 64 (-) | 25.7 | 0.77x |
| ict_up60k | 24125 | 64 | 136 | 1.06 | **12.5** (8) | 4.0 | 32 (-) | 22.4 | 0.56x |
| ict_up60k | 24125 | 128 | 167 | 1.33 | **9.2** (4) | 3.1 | 32 (-) | 9.1 | 1.01x |
| bfm_up60k | 60440 | 64 | 243 | 1.60 | **5.0** (8) | 9.2 | 8 (-) | 5.7 | 0.88x |
| bfm_up60k | 60440 | 128 | 309 | 2.16 | **3.7** (8) | 13.7 | 8 (-) | 2.3 | 1.62x |

Curve (B: mesh/s / picco GB; fasi al B migliore in ms per mesh: analisi+fattorizzazione cuDSS, iterazioni, geometria, copia su host):

- ict_down8k k=64: 1: 19.2/0.6, 2: 31.2/0.6, 4: 44.7/0.8, 8: 55.6/1.1, 16: 61.7/1.7, 32: 62.8/2.7, 64: 61.3/4.8 | B=32: fattorizzazione 6, iterazioni 9, geometria 0.4, copia 0.3; convergenti True
- ict_down8k k=128: 1: 15.9/0.6, 2: 23.6/0.7, 4: 33.0/1.0, 8: 39.8/1.5, 16: 43.8/2.1, 32: 42.6/3.8, 64: 42.8/7.1 | B=16: fattorizzazione 6, iterazioni 16, geometria 0.4, copia 0.6; convergenti True
- ict_original k=64: 1: 14.0/0.7, 2: 20.8/0.9, 4: 26.2/1.3, 8: 29.1/1.9, 16: 26.9/3.2, 32: 27.1/5.9, 64: 25.7/11.3 | B=8: fattorizzazione 15, iterazioni 17, geometria 1.1, copia 1.2; convergenti True
- ict_original k=128: 1: 10.3/0.8, 2: 15.8/1.2, 4: 19.2/1.7, 8: 19.9/2.9, 16: 19.5/5.2, 32: 19.7/9.9, 64: 19.3/19.3 | B=8: fattorizzazione 15, iterazioni 29, geometria 1.1, copia 4.3; convergenti True
- ict_up60k k=64: 1: 7.4/1.1, 2: 10.4/1.5, 4: 12.3/2.2, 8: 12.5/4.0, 16: 10.7/7.4, 32: 10.1/14.4 | B=8: fattorizzazione 42, iterazioni 33, geometria 2.4, copia 2.4; convergenti True
- ict_up60k k=128: 1: 6.0/1.3, 2: 7.9/1.8, 4: 9.2/3.1, 8: 8.7/5.8, 16: 8.1/11.0, 32: 7.8/21.5 | B=4: fattorizzazione 40, iterazioni 62, geometria 2.7, copia 3.8; convergenti True
- bfm_up60k k=64: 1: 4.1/1.6, 2: 4.8/2.7, 4: 4.8/4.9, 8: 5.0/9.2 | B=8: fattorizzazione 101, iterazioni 74, geometria 6.1, copia 17.6; convergenti True
- bfm_up60k k=128: 1: 3.2/2.2, 2: 3.3/3.8, 4: 3.6/7.1, 8: 3.7/13.7 | B=8: fattorizzazione 100, iterazioni 132, geometria 5.8, copia 28.5; convergenti True

## Vie alternative per gli autovettori, una mesh: Tesla V100-SXM3-32GB (methods_bk_Tesla_V100-SXM3-32GB.json)

Errore rispetto all'eigh denso fp64 della stessa mesh (stessa A). Tempo = geometria + autovettori, seconda chiamata.

| classe | V | k | via | s per mesh | autovalori err. rel. max (mediana) | note |
|---|---|---|---|---|---|---|
| ict_down8k | 3279 | 64 | dense_fp64 | 0.202 | 4.4e-13 (3.8e-15) |  |
| ict_down8k | 3279 | 128 | dense_fp64 | 0.191 | 5.3e-13 (2.0e-15) |  |
| ict_down8k | 3279 | 128 | dense_fp32 | 0.126 | 1.2e-02 (2.8e-05) |  |
| ict_original | 9409 | 64 | dense_fp64 | 3.007 | 2.4e-12 (3.6e-14) |  |
| ict_original | 9409 | 128 | dense_fp64 | 2.963 | 7.3e-12 (1.3e-14) |  |
| ict_original | 9409 | 128 | dense_fp32 | 2.124 | 2.6e-01 (5.9e-03) |  |
| ict_down8k | 3279 | 64 | torch_lobpcg_jacobi | 4.645 | 8.4e-06 (1.1e-09) |  |
| ict_down8k | 3279 | 128 | torch_lobpcg_jacobi | 3.616 | 8.8e-09 (1.5e-10) |  |
| ict_original | 9409 | 128 | torch_lobpcg_jacobi | 11.849 | 8.8e-03 (4.6e-04) |  |
| ict_down8k | | 128 | cupy_lobpcg_jacobi | | | AttributeError: 'NoneType' object has no attribute 'ndim' |
| ict_down8k | 3279 | 128 | chebyshev_noshift | 0.198 | 1.6e-08 (6.2e-11) | outer 30, converged False |
| ict_original | 9409 | 128 | chebyshev_noshift | 0.280 | 5.6e+01 (1.5e+00) | outer 30, converged False |
| ict_down8k | 3279 | 128 | shiftinvert_chebyshev | 0.067 | 2.2e-13 (2.2e-15) | outer 4, n_solve 16, rhs_cols 3328, converged True |
| ict_original | 9409 | 128 | shiftinvert_chebyshev | 0.129 | 2.3e-12 (6.8e-15) | outer 4, n_solve 16, rhs_cols 3328, converged True |
| ict_down8k | 3279 | 128 | shiftinvert_blockkrylov | 0.061 | 5.9e-13 (1.7e-15) | n_solve 32, rhs_cols 512, converged True |
| ict_original | 9409 | 128 | shiftinvert_blockkrylov | 0.091 | 2.6e-12 (7.1e-15) | n_solve 32, rhs_cols 512, converged True |

Spettro di A = M^-1/2 L M^-1/2 (area 1):

| classe | lambda_1 | lambda_64 | lambda_128 | lambda_max | lambda_max / lambda_128 |
|---|---|---|---|---|---|
| ict_down8k | 10.3 | 675 | 1378 | 1.35e+06 | 9.8e+02 |
| ict_original | 10.2 | 682 | 1420 | 6.05e+07 | 4.3e+04 |

