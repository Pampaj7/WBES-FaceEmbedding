# Costo per confronto: il nostro encoder contro i protocolli per coppia

Generato da `aau/costo_confronto/cc.py report` (definizioni nel docstring di `cc.py`). Tempi MISURATI in questa sessione; l'ultima tabella e' un'estrapolazione lineare dai tempi misurati.

## Protocollo

- Mesh: HIFI3D (vista della matrice), topologie original, down8k, up60k. Coppie: 50 per tipo (original-down8k, original-up60k, down8k-up60k), 150 in tutto, soggetti diversi, estratte con seme 20261013 dal sottocampione di `pr_noise.subsample` (dove esistono i valori di tutti i metodi). Tempi per mesh su 20 mesh per topologia fra quelle delle coppie.
- Ripetizioni: 1 giro di riscaldamento escluso + 3 cronometrati; per elemento la mediana dei giri, poi mediana [IQR p25-p75] fra gli elementi.
- CPU a UN thread per processo (secondi-core); GPU con `torch.cuda.synchronize`. Durante la misura girano in parallelo fino a ~7 processi a un thread sui core del job (stesse condizioni per tutti i metodi).
- Una coppia dei protocolli NoW = i due versi (A predetta, B riferimento), come la matrice; ICP e NICP = un verso, come la matrice; ULRSSM = mappe nei due versi (`pair_rms`).

## Hardware

- Nodo `a768-l40s-04.srv.aau.dk`, job 1072554; CPU: AMD EPYC 9354 32-Core Processor; core del job: 16; GPU: NVIDIA L40S; thread: OMP/MKL/OpenBLAS = 1, torch 1, onnxruntime 1.

## Per mesh (una volta per mesh: landmark, operatori, embedding)

mediana [IQR] per mesh. `totale` = somma delle fasi.

| metodo | fase | disp. | original | down8k | up60k | tutte |
|---|---|---|---|---|---|---|
| nostro | caricamento | cpu | 2.0 ms [2.0 ms, 2.0 ms] | 1.1 ms [1.1 ms, 1.1 ms] | 4.1 ms [4.1 ms, 4.2 ms] | 2.0 ms [1.1 ms, 4.1 ms] |
| nostro | operatori | cpu | 1.40 s [1.40 s, 1.41 s] | 488.9 ms [487.9 ms, 490.6 ms] | 4.51 s [4.49 s, 4.54 s] | 1.40 s [490.7 ms, 4.49 s] |
| nostro | scrittura_operatori | cpu | 12.8 ms [12.7 ms, 12.8 ms] | 4.9 ms [4.8 ms, 4.9 ms] | 32.2 ms [31.9 ms, 32.3 ms] | 12.8 ms [4.9 ms, 31.9 ms] |
| nostro | tabella_scala | cpu | 1.6 ms [1.6 ms, 1.6 ms] | 581.64 us [576.43 us, 585.37 us] | 3.9 ms [3.8 ms, 3.9 ms] | 1.6 ms [585.83 us, 3.8 ms] |
| nostro | lettura_operatori | cpu | 36.7 ms [36.5 ms, 38.9 ms] | 14.6 ms [14.4 ms, 16.6 ms] | 98.5 ms [97.1 ms, 100.1 ms] | 36.7 ms [16.6 ms, 96.7 ms] |
| nostro | trasferimento | gpu | 915.34 us [908.83 us, 928.99 us] | 464.37 us [455.57 us, 481.60 us] | 1.8 ms [1.7 ms, 1.8 ms] | 915.34 us [483.91 us, 1.7 ms] |
| nostro | forward | gpu | 6.3 ms [6.3 ms, 6.3 ms] | 5.8 ms [5.7 ms, 5.9 ms] | 7.6 ms [7.6 ms, 7.7 ms] | 6.3 ms [5.9 ms, 7.6 ms] |
| nostro | totale_cpu | cpu | 1.46 s [1.45 s, 1.46 s] | 510.9 ms [509.4 ms, 512.2 ms] | 4.65 s [4.62 s, 4.68 s] | 1.46 s [512.5 ms, 4.62 s] |
| nostro | totale_gpu | gpu | 7.2 ms [7.2 ms, 7.3 ms] | 6.2 ms [6.2 ms, 6.3 ms] | 9.4 ms [9.3 ms, 9.4 ms] | 7.2 ms [6.3 ms, 9.3 ms] |
| nostro | totale | cpu+gpu | 1.46 s [1.46 s, 1.47 s] | 517.2 ms [515.6 ms, 518.5 ms] | 4.66 s [4.63 s, 4.69 s] | 1.46 s [518.7 ms, 4.63 s] |
| icp | caricamento | cpu | 2.2 ms [2.2 ms, 2.2 ms] | 1.1 ms [1.1 ms, 1.2 ms] | 4.8 ms [4.8 ms, 5.0 ms] | 2.2 ms [1.2 ms, 4.8 ms] |
| icp | totale_cpu | cpu | 2.2 ms [2.2 ms, 2.2 ms] | 1.1 ms [1.1 ms, 1.2 ms] | 4.8 ms [4.8 ms, 5.0 ms] | 2.2 ms [1.2 ms, 4.8 ms] |
| icp | totale | cpu+gpu | 2.2 ms [2.2 ms, 2.2 ms] | 1.1 ms [1.1 ms, 1.2 ms] | 4.8 ms [4.8 ms, 5.0 ms] | 2.2 ms [1.2 ms, 4.8 ms] |
| nicp | caricamento | cpu | 2.2 ms [2.2 ms, 2.2 ms] | 1.1 ms [1.1 ms, 1.1 ms] | 4.8 ms [4.8 ms, 4.8 ms] | 2.2 ms [1.1 ms, 4.8 ms] |
| nicp | totale_cpu | cpu | 2.2 ms [2.2 ms, 2.2 ms] | 1.1 ms [1.1 ms, 1.1 ms] | 4.8 ms [4.8 ms, 4.8 ms] | 2.2 ms [1.1 ms, 4.8 ms] |
| nicp | totale | cpu+gpu | 2.2 ms [2.2 ms, 2.2 ms] | 1.1 ms [1.1 ms, 1.1 ms] | 4.8 ms [4.8 ms, 4.8 ms] | 2.2 ms [1.1 ms, 4.8 ms] |
| now_det | caricamento | cpu | 4.4 ms [4.4 ms, 4.4 ms] | 2.0 ms [2.0 ms, 2.0 ms] | 11.2 ms [11.1 ms, 11.2 ms] | 4.4 ms [2.0 ms, 11.1 ms] |
| now_det | render | cpu | 46.7 ms [46.4 ms, 47.0 ms] | 40.4 ms [40.2 ms, 41.2 ms] | 62.5 ms [62.0 ms, 62.6 ms] | 46.7 ms [41.2 ms, 62.0 ms] |
| now_det | scrittura_png | cpu | 18.1 ms [17.9 ms, 18.9 ms] | 17.8 ms [17.5 ms, 18.3 ms] | 18.9 ms [18.5 ms, 19.3 ms] | 18.4 ms [17.9 ms, 19.1 ms] |
| now_det | lettura_png | cpu | 3.8 ms [3.8 ms, 3.8 ms] | 3.8 ms [3.8 ms, 3.8 ms] | 3.8 ms [3.8 ms, 3.9 ms] | 3.8 ms [3.8 ms, 3.8 ms] |
| now_det | rilevamento | cpu | 350.8 ms [350.1 ms, 352.6 ms] | 351.7 ms [350.4 ms, 353.8 ms] | 351.5 ms [350.3 ms, 354.0 ms] | 351.4 ms [350.3 ms, 353.6 ms] |
| now_det | ripiego | cpu | 6.86 us [6.67 us, 7.09 us] | 6.87 us [6.69 us, 6.99 us] | 6.91 us [6.69 us, 7.15 us] | 6.87 us [6.69 us, 7.07 us] |
| now_det | retroproiezione | cpu | 49.0 ms [46.8 ms, 52.3 ms] | 44.8 ms [42.0 ms, 49.8 ms] | 59.1 ms [56.0 ms, 62.6 ms] | 50.0 ms [46.1 ms, 56.8 ms] |
| now_det | ritaglio | cpu | 433.97 us [430.57 us, 439.55 us] | 199.68 us [194.66 us, 205.81 us] | 962.83 us [958.33 us, 966.74 us] | 433.97 us [206.16 us, 957.98 us] |
| now_det | superficie_A | cpu | 286.40 us [281.01 us, 290.97 us] | 245.45 us [238.49 us, 255.07 us] | 351.47 us [343.45 us, 359.77 us] | 286.40 us [256.63 us, 342.61 us] |
| now_det | totale_cpu | cpu | 474.7 ms [471.8 ms, 479.0 ms] | 461.5 ms [458.1 ms, 466.2 ms] | 510.7 ms [506.2 ms, 512.0 ms] | 474.7 ms [466.4 ms, 505.9 ms] |
| now_det | totale | cpu+gpu | 474.7 ms [471.8 ms, 479.0 ms] | 461.5 ms [458.1 ms, 466.2 ms] | 510.7 ms [506.2 ms, 512.0 ms] | 474.7 ms [466.4 ms, 505.9 ms] |
| ulrssm | caricamento_e_scalari | cpu | 6.2 ms [6.1 ms, 6.2 ms] | 3.1 ms [3.0 ms, 3.1 ms] | 15.4 ms [15.2 ms, 18.2 ms] | 6.2 ms [3.1 ms, 15.2 ms] |
| ulrssm | operatori | cpu | 1.65 s [1.65 s, 1.66 s] | 541.9 ms [537.5 ms, 545.7 ms] | 5.86 s [5.85 s, 5.91 s] | 1.65 s [546.0 ms, 5.84 s] |
| ulrssm | operatori_da_cache | cpu | 23.1 ms [23.1 ms, 23.2 ms] | 11.5 ms [11.5 ms, 11.5 ms] | 54.6 ms [54.3 ms, 54.7 ms] | 23.1 ms [11.6 ms, 54.3 ms] |
| ulrssm | feature | gpu | 30.3 ms [30.2 ms, 30.4 ms] | 17.0 ms [16.9 ms, 17.1 ms] | 65.1 ms [64.5 ms, 65.7 ms] | 30.3 ms [17.1 ms, 64.4 ms] |
| ulrssm | totale_cpu | cpu | 1.68 s [1.68 s, 1.68 s] | 556.6 ms [552.0 ms, 560.2 ms] | 5.93 s [5.92 s, 5.98 s] | 1.68 s [560.6 ms, 5.91 s] |
| ulrssm | totale_gpu | gpu | 30.3 ms [30.2 ms, 30.4 ms] | 17.0 ms [16.9 ms, 17.1 ms] | 65.1 ms [64.5 ms, 65.7 ms] | 30.3 ms [17.1 ms, 64.4 ms] |
| ulrssm | totale | cpu+gpu | 1.71 s [1.71 s, 1.71 s] | 573.7 ms [569.0 ms, 577.3 ms] | 6.00 s [5.98 s, 6.04 s] | 1.71 s [577.7 ms, 5.97 s] |

## Per coppia

mediana [IQR] per coppia non ordinata.

| metodo | fase | disp. | original-down8k | original-up60k | down8k-up60k | tutte |
|---|---|---|---|---|---|---|
| icp | campionamento | cpu | 415.07 us [371.43 us, 420.11 us] | 797.15 us [715.32 us, 842.20 us] | 501.81 us [345.22 us, 511.52 us] | 503.13 us [403.49 us, 710.82 us] |
| icp | allineamento_rigido | cpu | 39.2 ms [35.5 ms, 42.1 ms] | 42.4 ms [33.4 ms, 46.5 ms] | 37.3 ms [34.7 ms, 40.7 ms] | 39.2 ms [34.7 ms, 42.7 ms] |
| icp | distanza | cpu | 2.9 ms [2.8 ms, 2.9 ms] | 3.2 ms [3.1 ms, 3.3 ms] | 2.8 ms [2.7 ms, 2.9 ms] | 2.9 ms [2.8 ms, 3.1 ms] |
| icp | totale_cpu | cpu | 42.5 ms [38.6 ms, 45.5 ms] | 46.2 ms [37.4 ms, 50.4 ms] | 40.4 ms [37.9 ms, 44.0 ms] | 42.6 ms [38.0 ms, 46.4 ms] |
| icp | totale | cpu+gpu | 42.5 ms [38.6 ms, 45.5 ms] | 46.2 ms [37.4 ms, 50.4 ms] | 40.4 ms [37.9 ms, 44.0 ms] | 42.6 ms [38.0 ms, 46.4 ms] |
| nicp | campionamento | cpu | 431.36 us [426.90 us, 436.65 us] | 879.26 us [808.37 us, 891.90 us] | 532.49 us [526.43 us, 541.45 us] | 532.49 us [433.08 us, 802.65 us] |
| nicp | allineamento_rigido | cpu | 38.9 ms [35.4 ms, 42.0 ms] | 41.9 ms [33.0 ms, 45.8 ms] | 37.4 ms [34.4 ms, 40.3 ms] | 38.9 ms [34.4 ms, 42.4 ms] |
| nicp | allineamento_nonrigido | cpu | 1.27 s [1.16 s, 1.39 s] | 1.30 s [1.23 s, 1.86 s] | 1.25 s [998.9 ms, 1.50 s] | 1.28 s [1.18 s, 1.52 s] |
| nicp | distanza | cpu | 128.2 ms [115.1 ms, 130.7 ms] | 134.4 ms [132.9 ms, 135.4 ms] | 122.9 ms [116.6 ms, 130.3 ms] | 130.4 ms [116.8 ms, 133.0 ms] |
| nicp | totale_cpu | cpu | 1.44 s [1.33 s, 1.55 s] | 1.48 s [1.41 s, 2.03 s] | 1.42 s [1.15 s, 1.67 s] | 1.45 s [1.35 s, 1.68 s] |
| nicp | totale | cpu+gpu | 1.44 s [1.33 s, 1.55 s] | 1.48 s [1.41 s, 2.03 s] | 1.42 s [1.15 s, 1.67 s] | 1.45 s [1.35 s, 1.68 s] |
| now_det | procrustes | cpu | 264.39 us [241.82 us, 287.62 us] | 291.61 us [278.22 us, 306.26 us] | 300.50 us [291.35 us, 309.26 us] | 291.02 us [268.35 us, 303.07 us] |
| now_det | icp_now | cpu | 234.8 ms [195.2 ms, 309.4 ms] | 748.2 ms [615.2 ms, 913.1 ms] | 493.6 ms [369.8 ms, 615.2 ms] | 475.3 ms [298.2 ms, 694.8 ms] |
| now_det | distanza | cpu | 21.3 ms [20.4 ms, 22.0 ms] | 71.0 ms [67.1 ms, 74.2 ms] | 44.2 ms [41.9 ms, 45.9 ms] | 44.2 ms [22.0 ms, 67.0 ms] |
| now_det | totale_cpu | cpu | 256.0 ms [217.1 ms, 329.7 ms] | 820.0 ms [682.9 ms, 988.9 ms] | 538.2 ms [413.0 ms, 664.1 ms] | 529.3 ms [322.1 ms, 759.6 ms] |
| now_det | totale | cpu+gpu | 256.0 ms [217.1 ms, 329.7 ms] | 820.0 ms [682.9 ms, 988.9 ms] | 538.2 ms [413.0 ms, 664.1 ms] | 529.3 ms [322.1 ms, 759.6 ms] |
| ulrssm | mappe_e_distanza | gpu | 5.7 ms [5.7 ms, 5.8 ms] | 22.7 ms [22.7 ms, 22.7 ms] | 8.5 ms [8.5 ms, 8.6 ms] | 8.5 ms [5.8 ms, 22.7 ms] |
| ulrssm | totale_gpu | gpu | 5.7 ms [5.7 ms, 5.8 ms] | 22.7 ms [22.7 ms, 22.7 ms] | 8.5 ms [8.5 ms, 8.6 ms] | 8.5 ms [5.8 ms, 22.7 ms] |
| ulrssm | totale | cpu+gpu | 5.7 ms [5.7 ms, 5.8 ms] | 22.7 ms [22.7 ms, 22.7 ms] | 8.5 ms [8.5 ms, 8.6 ms] | 8.5 ms [5.8 ms, 22.7 ms] |

### Nostro: confronti fra vettori (`fact_paired.distances`, d_F e d_P), tutte le coppie di N mesh, CPU a un thread

| N | coppie | tempo (mediana dei giri) | per coppia |
|---|---|---|---|
| 100 | 4,950 | 5.9 ms | 1.19 us |
| 1,000 | 499,500 | 366.7 ms | 734.2 ns |
| 10,000 | 49,995,000 | 37.97 s | 759.4 ns |

Costo fisso di una chiamata (rilegge il checkpoint): 1.8 ms; i blocchi sono di 500,000 coppie, quindi incluso una volta per blocco.

## Per coppia isolata (le due mesh da zero + la coppia)

media per mesh x 2 + media per coppia (per il nostro: 2 embedding + 1 confronto dal blocco N = 10.000).

| metodo | CPU (s-core) | GPU (s) |
|---|---|---|
| nostro | 4.42 s | 15.3 ms |
| icp | 47.4 ms | 0.0 ns |
| nicp | 1.53 s | 0.0 ns |
| now_det | 1.54 s | 0.0 ns |
| ulrssm | 5.45 s | 87.3 ms |

## Scalabilita': tutte le coppie di N mesh (ESTRAPOLAZIONE lineare dai tempi misurati)

costo = N x (media per mesh) + N(N-1)/2 x (media per coppia); per il nostro i confronti sono MISURATI su N mesh (tabella sopra), gli embedding estrapolati. Media e non mediana: stima di una somma. Tempo a muro stimato = s-core CPU / 16 (parallelismo ideale sui 16 core del nodo di misura) + s GPU (GPU in serie, nessuna sovrapposizione).

| N | coppie | metodo | CPU (ore-core) | GPU (ore) | a muro, 16 core + 1 GPU | x nostro |
|---|---|---|---|---|---|---|
| 100 | 4,950 | nostro | 0.0615 | 0.000212 | 14.6 s | 1.00 |
| 100 | 4,950 | icp | 0.0576 | 0 | 13.0 s | 0.89 |
| 100 | 4,950 | nicp | 2.1 | 0 | 7.9 min | 32 |
| 100 | 4,950 | now_det | 0.799 | 0 | 3.0 min | 12 |
| 100 | 4,950 | ulrssm | 0.0757 | 0.018 | 1.4 min | 5.60 |
| 1,000 | 499,500 | nostro | 0.615 | 0.00212 | 2.4 min | 1.00 |
| 1,000 | 499,500 | icp | 5.81 | 0 | 21.8 min | 8.96 |
| 1,000 | 499,500 | nicp | 212 | 0 | 13.2 h | 326 |
| 1,000 | 499,500 | now_det | 79.4 | 0 | 5.0 h | 122 |
| 1,000 | 499,500 | ulrssm | 0.757 | 1.72 | 1.8 h | 44 |
| 10,000 | 49,995,000 | nostro | 6.16 | 0.0212 | 24.4 min | 1.00 |
| 10,000 | 49,995,000 | icp | 581 | 0 | 36.3 h | 89 |
| 10,000 | 49,995,000 | nicp | 2.12e+04 | 0 | 55.1 giorni | 3,259 |
| 10,000 | 49,995,000 | now_det | 7.93e+03 | 0 | 20.7 giorni | 1,222 |
| 10,000 | 49,995,000 | ulrssm | 7.57 | 171 | 7.2 giorni | 423 |

## Controlli di riproduzione

- `cmp`: `{"form_a_blocchi_vs_intera": {"n": 1770, "nan": 0, "max_abs": 0.0, "max_rel": 0.0, "equal": 1770}, "shape_a_blocchi_vs_intera": {"n": 1770, "nan": 0, "max_abs": 0.0, "max_rel": 0.0, "equal": 1770}}`
- `det_render`: `{"render_uguali_a_pr_noise": 183, "render": 183}`
- `detcheck`: `{"render": 24, "lmk2d_cc_vs_pr_noise_detect_stessi_render_px": {"n": 4896, "nan": 0, "max_abs": 0.0, "max_rel": 0.0, "equal": 4896}, "fallback_uguali": true}`
- `detect`: `{"ripieghi": 0, "onnxruntime": "1.23.2", "intra_op_num_threads": 1}`
- `icp`: `{"rigid_icp_chamfer_mm_vs_matrice_blmm": {"n": 150, "nan": 0, "max_abs": 0.0, "max_rel": 0.0, "equal": 150}, "unit_mm": 91.63428668106282, "nan_qui_e_nella_matrice": 0, "nan_solo_qui": 0, "nan_solo_nella_matrice": 0, "fallite": []}`
- `nicp`: `{"nicp_p2tri_mm_vs_matrice_blmm": {"n": 147, "nan": 3, "max_abs": 0.0, "max_rel": 0.0, "equal": 147}, "unit_mm": 91.63428668106282, "nan_qui_e_nella_matrice": 3, "nan_solo_qui": 0, "nan_solo_nella_matrice": 0, "fallite": ["id900020/up60k|id900130/original: RuntimeError: Factor is exactly singular", "id900072/up60k|id900392/original: RuntimeError: Factor is exactly singular", "id900107/up60k|id900441/down8k: RuntimeError: Factor is exactly singular"]}`
- `now`: `{"L68_vs_prep_det_mm": {"n": 37332, "nan": 0, "max_abs": 0.022314973931543136, "max_rel": 0.12654125453096804, "equal": 14001}, "lmk2d_vs_pr_noise_px": {"n": 37332, "nan": 0, "max_abs": 0.03607177734375, "max_rel": 0.0451850718275993, "equal": 13093}, "now_det_mm_vs_plugin_lmk_det": {"n": 150, "nan": 0, "max_abs": 0.0001938472242186684, "max_rel": 9.198386802629333e-05, "equal": 0}, "now_spezzata_vs_now_pair": {"n": 300, "nan": 0, "max_abs": 0.0, "max_rel": 0.0, "equal": 300}, "versi_per_coppia": 2}`
- `ops`: `{"area_mm2_vs_tabella": {"n": 60, "nan": 0, "max_abs": 0.0, "max_rel": 0.0, "equal": 60}}`
- `ours`: `{"Z_vs_collaudo_L40S": {"n": 15420, "nan": 0, "max_abs": 0.0005622655153274536, "max_rel": 2.2736293240982137, "equal": 3}, "Z_vs_sviluppo_A100": {"n": 15420, "nan": 0, "max_abs": 0.0005622655153274536, "max_rel": 2.2736293240982137, "equal": 4}}`
- `ulrssm`: `{"d_pair_fr_vs_plugin_faces_xyz": {"n": 150, "nan": 0, "max_abs": 0.0019021242552712536, "max_rel": 0.0003918957128918812, "equal": 138}, "input_type": "xyz"}`

## Ambiente per passo

- `cmp`: a768-l40s-04.srv.aau.dk, job 1072554, senza GPU, core 16, torch 2.5.0a0+e000cf0ad9.nv24.10 (1 thread), 2026-10-13 16:30:06
- `det_render`: a768-l40s-04.srv.aau.dk, job 1072554, senza GPU, core 16, torch 2.5.0a0+e000cf0ad9.nv24.10 (1 thread), 2026-10-13 16:07:12
- `detcheck`: a768-l40s-04.srv.aau.dk, job 1072554, senza GPU, core 16, torch - (- thread), 2026-10-13 16:31:53
- `detect`: a768-l40s-04.srv.aau.dk, job 1072554, senza GPU, core 16, torch 2.5.0a0+e000cf0ad9.nv24.10 (1 thread), 2026-10-13 16:10:10
- `icp`: a768-l40s-04.srv.aau.dk, job 1072554, senza GPU, core 16, torch 2.5.0a0+e000cf0ad9.nv24.10 (1 thread), 2026-10-13 16:07:12
- `nicp`: a768-l40s-04.srv.aau.dk, job 1072554, senza GPU, core 16, torch 2.5.0a0+e000cf0ad9.nv24.10 (1 thread), 2026-10-13 16:22:07
- `now`: a768-l40s-04.srv.aau.dk, job 1072554, senza GPU, core 16, torch 2.5.0a0+e000cf0ad9.nv24.10 (1 thread), 2026-10-13 16:17:46
- `ops`: a768-l40s-04.srv.aau.dk, job 1072554, senza GPU, core 16, torch 2.5.0a0+e000cf0ad9.nv24.10 (1 thread), 2026-10-13 16:15:29
- `ours`: a768-l40s-04.srv.aau.dk, job 1072554, NVIDIA L40S, core 16, torch 2.5.0a0+e000cf0ad9.nv24.10 (1 thread), 2026-10-13 16:15:49
- `ulrssm`: a768-l40s-04.srv.aau.dk, job 1072554, NVIDIA L40S, core 16, torch 2.5.0a0+e000cf0ad9.nv24.10 (1 thread), 2026-10-13 16:27:28
