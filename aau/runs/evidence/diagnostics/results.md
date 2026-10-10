# Diagnostica D: risultati

Protocollo `PROTOCOL_D.md` (sha256 b9f5378a..., commit 251bdd6) ed emendamento 1. Generato da `aau/diagnostics/summarize.py`; numeri in `d1_spearman.csv`, `d1_delta.csv`, `d2_probe.csv`.

## Letture

- C1 (held-out statici compatibili col codice di D1): **si**
- R1 (varieta' leva forte, FLAME suddiviso): **si**; con FLAME nativo: si
- R2 (limite di lettura sui visti, contro B-FLAME): **no**
- R3 (informazione nell'embedding, FaceScape SR): **si**

| braccio | Delta_gen (suddiviso) | Delta_gen (nativo) | Delta_read | DiD difficolta' |
|---|---|---|---|---|
| fact. s1234 | +0.108 [+0.079, +0.137] | +0.131 [+0.100, +0.162] | +0.166 [+0.139, +0.196] | +0.166 [+0.126, +0.209] |
| fact. s2345 | +0.107 [+0.077, +0.139] | +0.123 [+0.091, +0.154] | +0.166 [+0.139, +0.197] | +0.169 [+0.128, +0.213] |
| C3M e123 | +0.137 [+0.112, +0.162] | +0.209 [+0.177, +0.241] | +0.208 [+0.179, +0.240] | +0.194 [+0.152, +0.241] |

Note del coder, scritte DOPO i numeri (commento, non regole):

- R1 passa sulla soglia: Delta_gen 0.108 e 0.107, ma gli estremi inferiori (0.079, 0.077) stanno sotto 0.10; col FLAME nativo 0.131 e 0.123 (effetto della risoluzione su SR +0.023 e +0.015).
- Il calo su FLAME non e' dell'insieme: B-GNM, incrociato su bfm, ict e FLAME, su FLAME va MEGLIO (-0.056) e la differenza delle differenze e' +0.17 per entrambi i semi.
- R2 no: in distribuzione i bracci superano B di +0.17 (SR); anche B-GNM col prior esatto su gnm arriva a 0.752 contro 0.910. Su FLAME suddiviso i bracci sono pari a B-GNM incrociato (-0.005) e a B-FLAME esatto (-0.024, IC che tocca lo 0); sul nativo sotto B-FLAME esatto (-0.047 e -0.039).
- C3M (10x identita'): +0.03-0.05 in distribuzione, +0.013 su FLAME suddiviso, -0.037 su FLAME nativo rispetto a C3F s1234: piu' identita' degli stessi generatori non chiudono il gap di generatore.
- D2: la sonda e' addestrata su 50 soggetti FaceScape per fold (GT del dominio dev): dice che l'informazione e' decodificabile linearmente dall'embedding, non che esista un metodo. L'IC del protocollo e' spostato in basso come previsto dall'emendamento 1, ma esclude lo 0; la sonda coi bersagli permutati (0.61-0.63) sta sotto d_P. HIFI3D (descrittivo): SR +0.09-0.11 con l'IC del protocollo che tocca lo 0; FR: la sonda non batte d_F calibrata.

## D1

### D1, GT SR (d_P dei bracci, B `vb_sr`)

| insieme | fact. s1234 | fact. s2345 | C3M e123 | B-FLAME | B-GNM | righe |
|---|---|---|---|---|---|---|
| bfm | 0.902 [0.881, 0.922] | 0.905 [0.882, 0.925] | 0.949 [0.936, 0.958] | 0.709 [0.655, 0.755] | 0.671 [0.618, 0.721] | 99000 |
| ict | 0.930 [0.905, 0.948] | 0.934 [0.909, 0.950] | 0.964 [0.950, 0.974] | 0.786 [0.717, 0.837] | 0.840 [0.786, 0.880] | 99000 |
| gnm | 0.910 [0.885, 0.928] | 0.902 [0.873, 0.924] | 0.953 [0.940, 0.963] | 0.748 [0.688, 0.796] | 0.752 [0.693, 0.798] (esatto) | 99000 |
| regen_ict | 0.930 [0.905, 0.948] | 0.933 [0.909, 0.950] | 0.964 [0.950, 0.974] | - | - | 99000 |
| regen_gnm | 0.910 [0.885, 0.928] | 0.902 [0.873, 0.924] | 0.953 [0.940, 0.963] | - | - | 99000 |
| flame2023_s1 | 0.806 [0.778, 0.832] | 0.806 [0.776, 0.832] | 0.819 [0.794, 0.844] | 0.830 [0.800, 0.858] (esatto) | 0.812 [0.781, 0.841] | 398000 |
| flame2023 | 0.783 [0.753, 0.810] | 0.791 [0.761, 0.819] | 0.746 [0.713, 0.778] | 0.830 [0.800, 0.858] (esatto) | 0.803 [0.771, 0.832] | 398000 |

### D1, GT FR (d_F calibrata dei bracci, B `vb_fr`)

| insieme | fact. s1234 | fact. s2345 | C3M e123 | B-FLAME | B-GNM | righe |
|---|---|---|---|---|---|---|
| bfm | 0.825 [0.782, 0.862] | 0.818 [0.772, 0.858] | 0.828 [0.790, 0.861] | 0.644 [0.575, 0.711] | 0.555 [0.476, 0.633] | 99000 |
| ict | 0.895 [0.857, 0.920] | 0.904 [0.872, 0.926] | 0.961 [0.947, 0.971] | 0.772 [0.699, 0.830] | 0.810 [0.744, 0.858] | 99000 |
| gnm | 0.857 [0.818, 0.891] | 0.839 [0.795, 0.877] | 0.939 [0.920, 0.955] | 0.782 [0.719, 0.834] | 0.755 [0.693, 0.812] (esatto) | 99000 |
| regen_ict | 0.895 [0.857, 0.920] | 0.903 [0.872, 0.925] | 0.961 [0.947, 0.971] | - | - | 99000 |
| regen_gnm | 0.857 [0.818, 0.891] | 0.839 [0.795, 0.876] | 0.939 [0.920, 0.955] | - | - | 99000 |
| flame2023_s1 | 0.809 [0.777, 0.838] | 0.804 [0.770, 0.836] | 0.832 [0.805, 0.859] | 0.803 [0.768, 0.838] (esatto) | 0.741 [0.697, 0.787] | 398000 |
| flame2023 | 0.761 [0.725, 0.796] | 0.770 [0.735, 0.804] | 0.773 [0.737, 0.806] | 0.838 [0.806, 0.868] (esatto) | 0.723 [0.679, 0.770] | 398000 |

### D1, delta (stesse righe e repliche dentro un insieme; insiemi diversi indipendenti)

| tipo | braccio | GT | confronto | delta [IC 95%] | P(<= 0) |
|---|---|---|---|---|---|
| gen | fact. s1234 | fr | media visti - flame2023_s1: factorized_s1234|form_cal | +0.050 [+0.015, +0.086] | 0.005 |
| gen | fact. s1234 | fr | media visti - flame2023: factorized_s1234|form_cal | +0.098 [+0.059, +0.139] | 0.000 |
| gen | fact. s2345 | fr | media visti - flame2023_s1: factorized_s2345|form_cal | +0.050 [+0.010, +0.088] | 0.006 |
| gen | fact. s2345 | fr | media visti - flame2023: factorized_s2345|form_cal | +0.084 [+0.043, +0.125] | 0.000 |
| gen | C3M e123 | fr | media visti - flame2023_s1: factorizedc3m_e123|form_cal | +0.077 [+0.047, +0.106] | 0.000 |
| gen | C3M e123 | fr | media visti - flame2023: factorizedc3m_e123|form_cal | +0.136 [+0.101, +0.171] | 0.000 |
| gen | fact. s1234 | sr | media visti - flame2023_s1: factorized_s1234|shape | +0.108 [+0.079, +0.137] | 0.000 |
| gen | fact. s1234 | sr | media visti - flame2023: factorized_s1234|shape | +0.131 [+0.100, +0.162] | 0.000 |
| gen | fact. s2345 | sr | media visti - flame2023_s1: factorized_s2345|shape | +0.107 [+0.077, +0.139] | 0.000 |
| gen | fact. s2345 | sr | media visti - flame2023: factorized_s2345|shape | +0.123 [+0.091, +0.154] | 0.000 |
| gen | C3M e123 | sr | media visti - flame2023_s1: factorizedc3m_e123|shape | +0.137 [+0.112, +0.162] | 0.000 |
| gen | C3M e123 | sr | media visti - flame2023: factorizedc3m_e123|shape | +0.209 [+0.177, +0.241] | 0.000 |
| read | fact. s1234 | fr | media visti: factorized_s1234|form_cal - B-flame2023|vb_fr (incrociato) | +0.126 [+0.097, +0.158] | 0.000 |
| read | fact. s2345 | fr | media visti: factorized_s2345|form_cal - B-flame2023|vb_fr (incrociato) | +0.121 [+0.089, +0.154] | 0.000 |
| read | C3M e123 | fr | media visti: factorizedc3m_e123|form_cal - B-flame2023|vb_fr (incrociato) | +0.177 [+0.145, +0.212] | 0.000 |
| read | fact. s1234 | sr | media visti: factorized_s1234|shape - B-flame2023|vb_sr (incrociato) | +0.166 [+0.139, +0.196] | 0.000 |
| read | fact. s2345 | sr | media visti: factorized_s2345|shape - B-flame2023|vb_sr (incrociato) | +0.166 [+0.139, +0.197] | 0.000 |
| read | C3M e123 | sr | media visti: factorizedc3m_e123|shape - B-flame2023|vb_sr (incrociato) | +0.208 [+0.179, +0.240] | 0.000 |
| difficulty | B-gnm | fr | media {bfm, ict} - flame2023_s1: B-gnm|vb (incrociato) | -0.059 [-0.127, +0.007] | 0.956 |
| difficulty | fact. s1234 | fr | media {bfm, ict} - flame2023_s1: factorized_s1234|form_cal | +0.051 [+0.011, +0.091] | 0.007 |
| difficulty | fact. s1234 | fr | DiD: (factorized_s1234|form_cal) - (B-gnm|vb_fr) | +0.110 [+0.052, +0.167] | 0.000 |
| difficulty | fact. s2345 | fr | media {bfm, ict} - flame2023_s1: factorized_s2345|form_cal | +0.057 [+0.016, +0.099] | 0.004 |
| difficulty | fact. s2345 | fr | DiD: (factorized_s2345|form_cal) - (B-gnm|vb_fr) | +0.116 [+0.058, +0.174] | 0.000 |
| difficulty | C3M e123 | fr | media {bfm, ict} - flame2023_s1: factorizedc3m_e123|form_cal | +0.062 [+0.029, +0.093] | 0.000 |
| difficulty | C3M e123 | fr | DiD: (factorizedc3m_e123|form_cal) - (B-gnm|vb_fr) | +0.121 [+0.064, +0.183] | 0.000 |
| difficulty | B-gnm | sr | media {bfm, ict} - flame2023_s1: B-gnm|vb (incrociato) | -0.056 [-0.107, -0.010] | 0.991 |
| difficulty | fact. s1234 | sr | media {bfm, ict} - flame2023_s1: factorized_s1234|shape | +0.110 [+0.081, +0.140] | 0.000 |
| difficulty | fact. s1234 | sr | DiD: (factorized_s1234|shape) - (B-gnm|vb_sr) | +0.166 [+0.126, +0.209] | 0.000 |
| difficulty | fact. s2345 | sr | media {bfm, ict} - flame2023_s1: factorized_s2345|shape | +0.113 [+0.084, +0.145] | 0.000 |
| difficulty | fact. s2345 | sr | DiD: (factorized_s2345|shape) - (B-gnm|vb_sr) | +0.169 [+0.128, +0.213] | 0.000 |
| difficulty | C3M e123 | sr | media {bfm, ict} - flame2023_s1: factorizedc3m_e123|shape | +0.138 [+0.112, +0.163] | 0.000 |
| difficulty | C3M e123 | sr | DiD: (factorizedc3m_e123|shape) - (B-gnm|vb_sr) | +0.194 [+0.152, +0.241] | 0.000 |
| resolution | fact. s1234 | fr | flame2023_s1 - flame2023: factorized_s1234|form_cal | +0.048 [+0.041, +0.056] | 0.000 |
| resolution | fact. s2345 | fr | flame2023_s1 - flame2023: factorized_s2345|form_cal | +0.034 [+0.028, +0.040] | 0.000 |
| resolution | C3M e123 | fr | flame2023_s1 - flame2023: factorizedc3m_e123|form_cal | +0.059 [+0.049, +0.071] | 0.000 |
| resolution | fact. s1234 | sr | flame2023_s1 - flame2023: factorized_s1234|shape | +0.023 [+0.015, +0.031] | 0.000 |
| resolution | fact. s2345 | sr | flame2023_s1 - flame2023: factorized_s2345|shape | +0.015 [+0.009, +0.021] | 0.000 |
| resolution | C3M e123 | sr | flame2023_s1 - flame2023: factorizedc3m_e123|shape | +0.073 [+0.058, +0.087] | 0.000 |
| within | fact. s1234 | fr | bfm: factorized_s1234|form_cal - B-gnm|vb_fr (incrociato) | +0.271 [+0.192, +0.344] | 0.000 |
| within | fact. s1234 | fr | bfm: factorized_s1234|form_cal - B-flame2023|vb_fr (incrociato) | +0.181 [+0.110, +0.247] | 0.000 |
| within | fact. s1234 | fr | ict: factorized_s1234|form_cal - B-gnm|vb_fr (incrociato) | +0.084 [+0.047, +0.131] | 0.000 |
| within | fact. s1234 | fr | ict: factorized_s1234|form_cal - B-flame2023|vb_fr (incrociato) | +0.123 [+0.079, +0.179] | 0.000 |
| within | fact. s1234 | fr | gnm: factorized_s1234|form_cal - B-gnm|vb_fr (prior esatto) | +0.102 [+0.063, +0.144] | 0.000 |
| within | fact. s1234 | fr | gnm: factorized_s1234|form_cal - B-flame2023|vb_fr (incrociato) | +0.075 [+0.034, +0.122] | 0.000 |
| within | fact. s1234 | fr | flame2023_s1: factorized_s1234|form_cal - B-gnm|vb_fr (incrociato) | +0.067 [+0.035, +0.097] | 0.000 |
| within | fact. s1234 | fr | flame2023_s1: factorized_s1234|form_cal - B-flame2023|vb_fr (prior esatto) | +0.006 [-0.020, +0.032] | 0.321 |
| within | fact. s1234 | fr | flame2023: factorized_s1234|form_cal - B-gnm|vb_fr (incrociato) | +0.037 [+0.006, +0.068] | 0.007 |
| within | fact. s1234 | fr | flame2023: factorized_s1234|form_cal - B-flame2023|vb_fr (prior esatto) | -0.077 [-0.108, -0.046] | 1.000 |
| within | fact. s2345 | fr | bfm: factorized_s2345|form_cal - B-gnm|vb_fr (incrociato) | +0.263 [+0.184, +0.336] | 0.000 |
| within | fact. s2345 | fr | bfm: factorized_s2345|form_cal - B-flame2023|vb_fr (incrociato) | +0.173 [+0.104, +0.243] | 0.000 |
| within | fact. s2345 | fr | ict: factorized_s2345|form_cal - B-gnm|vb_fr (incrociato) | +0.093 [+0.056, +0.140] | 0.000 |
| within | fact. s2345 | fr | ict: factorized_s2345|form_cal - B-flame2023|vb_fr (incrociato) | +0.132 [+0.089, +0.187] | 0.000 |
| within | fact. s2345 | fr | gnm: factorized_s2345|form_cal - B-gnm|vb_fr (prior esatto) | +0.084 [+0.046, +0.125] | 0.000 |
| within | fact. s2345 | fr | gnm: factorized_s2345|form_cal - B-flame2023|vb_fr (incrociato) | +0.057 [+0.015, +0.106] | 0.003 |
| within | fact. s2345 | fr | flame2023_s1: factorized_s2345|form_cal - B-gnm|vb_fr (incrociato) | +0.062 [+0.029, +0.097] | 0.000 |
| within | fact. s2345 | fr | flame2023_s1: factorized_s2345|form_cal - B-flame2023|vb_fr (prior esatto) | +0.001 [-0.028, +0.031] | 0.481 |
| within | fact. s2345 | fr | flame2023: factorized_s2345|form_cal - B-gnm|vb_fr (incrociato) | +0.046 [+0.013, +0.081] | 0.005 |
| within | fact. s2345 | fr | flame2023: factorized_s2345|form_cal - B-flame2023|vb_fr (prior esatto) | -0.068 [-0.101, -0.034] | 1.000 |
| within | C3M e123 | fr | bfm: factorizedc3m_e123|form_cal - B-gnm|vb_fr (incrociato) | +0.273 [+0.193, +0.347] | 0.000 |
| within | C3M e123 | fr | bfm: factorizedc3m_e123|form_cal - B-flame2023|vb_fr (incrociato) | +0.183 [+0.114, +0.254] | 0.000 |
| within | C3M e123 | fr | ict: factorizedc3m_e123|form_cal - B-gnm|vb_fr (incrociato) | +0.150 [+0.107, +0.211] | 0.000 |
| within | C3M e123 | fr | ict: factorizedc3m_e123|form_cal - B-flame2023|vb_fr (incrociato) | +0.189 [+0.137, +0.257] | 0.000 |
| within | C3M e123 | fr | gnm: factorizedc3m_e123|form_cal - B-gnm|vb_fr (prior esatto) | +0.184 [+0.136, +0.235] | 0.000 |
| within | C3M e123 | fr | gnm: factorizedc3m_e123|form_cal - B-flame2023|vb_fr (incrociato) | +0.157 [+0.113, +0.205] | 0.000 |
| within | C3M e123 | fr | flame2023_s1: factorizedc3m_e123|form_cal - B-gnm|vb_fr (incrociato) | +0.091 [+0.056, +0.125] | 0.000 |
| within | C3M e123 | fr | flame2023_s1: factorizedc3m_e123|form_cal - B-flame2023|vb_fr (prior esatto) | +0.030 [+0.002, +0.056] | 0.014 |
| within | C3M e123 | fr | flame2023: factorizedc3m_e123|form_cal - B-gnm|vb_fr (incrociato) | +0.049 [+0.015, +0.083] | 0.003 |
| within | C3M e123 | fr | flame2023: factorizedc3m_e123|form_cal - B-flame2023|vb_fr (prior esatto) | -0.065 [-0.095, -0.033] | 1.000 |
| within | fact. s1234 | sr | bfm: factorized_s1234|shape - B-gnm|vb_sr (incrociato) | +0.231 [+0.180, +0.285] | 0.000 |
| within | fact. s1234 | sr | bfm: factorized_s1234|shape - B-flame2023|vb_sr (incrociato) | +0.193 [+0.147, +0.245] | 0.000 |
| within | fact. s1234 | sr | ict: factorized_s1234|shape - B-gnm|vb_sr (incrociato) | +0.090 [+0.057, +0.130] | 0.000 |
| within | fact. s1234 | sr | ict: factorized_s1234|shape - B-flame2023|vb_sr (incrociato) | +0.145 [+0.103, +0.202] | 0.000 |
| within | fact. s1234 | sr | gnm: factorized_s1234|shape - B-gnm|vb_sr (prior esatto) | +0.158 [+0.119, +0.202] | 0.000 |
| within | fact. s1234 | sr | gnm: factorized_s1234|shape - B-flame2023|vb_sr (incrociato) | +0.161 [+0.121, +0.209] | 0.000 |
| within | fact. s1234 | sr | flame2023_s1: factorized_s1234|shape - B-gnm|vb_sr (incrociato) | -0.005 [-0.031, +0.022] | 0.658 |
| within | fact. s1234 | sr | flame2023_s1: factorized_s1234|shape - B-flame2023|vb_sr (prior esatto) | -0.024 [-0.050, +0.001] | 0.967 |
| within | fact. s1234 | sr | flame2023: factorized_s1234|shape - B-gnm|vb_sr (incrociato) | -0.020 [-0.048, +0.009] | 0.909 |
| within | fact. s1234 | sr | flame2023: factorized_s1234|shape - B-flame2023|vb_sr (prior esatto) | -0.047 [-0.075, -0.019] | 0.998 |
| within | fact. s2345 | sr | bfm: factorized_s2345|shape - B-gnm|vb_sr (incrociato) | +0.234 [+0.183, +0.289] | 0.000 |
| within | fact. s2345 | sr | bfm: factorized_s2345|shape - B-flame2023|vb_sr (incrociato) | +0.196 [+0.150, +0.249] | 0.000 |
| within | fact. s2345 | sr | ict: factorized_s2345|shape - B-gnm|vb_sr (incrociato) | +0.093 [+0.060, +0.135] | 0.000 |
| within | fact. s2345 | sr | ict: factorized_s2345|shape - B-flame2023|vb_sr (incrociato) | +0.148 [+0.106, +0.203] | 0.000 |
| within | fact. s2345 | sr | gnm: factorized_s2345|shape - B-gnm|vb_sr (prior esatto) | +0.150 [+0.109, +0.196] | 0.000 |
| within | fact. s2345 | sr | gnm: factorized_s2345|shape - B-flame2023|vb_sr (incrociato) | +0.154 [+0.111, +0.203] | 0.000 |
| within | fact. s2345 | sr | flame2023_s1: factorized_s2345|shape - B-gnm|vb_sr (incrociato) | -0.005 [-0.035, +0.025] | 0.657 |
| within | fact. s2345 | sr | flame2023_s1: factorized_s2345|shape - B-flame2023|vb_sr (prior esatto) | -0.024 [-0.054, +0.004] | 0.957 |
| within | fact. s2345 | sr | flame2023: factorized_s2345|shape - B-gnm|vb_sr (incrociato) | -0.012 [-0.043, +0.019] | 0.779 |
| within | fact. s2345 | sr | flame2023: factorized_s2345|shape - B-flame2023|vb_sr (prior esatto) | -0.039 [-0.070, -0.010] | 0.994 |
| within | C3M e123 | sr | bfm: factorizedc3m_e123|shape - B-gnm|vb_sr (incrociato) | +0.278 [+0.225, +0.332] | 0.000 |
| within | C3M e123 | sr | bfm: factorizedc3m_e123|shape - B-flame2023|vb_sr (incrociato) | +0.240 [+0.192, +0.294] | 0.000 |
| within | C3M e123 | sr | ict: factorizedc3m_e123|shape - B-gnm|vb_sr (incrociato) | +0.124 [+0.088, +0.172] | 0.000 |
| within | C3M e123 | sr | ict: factorizedc3m_e123|shape - B-flame2023|vb_sr (incrociato) | +0.178 [+0.131, +0.240] | 0.000 |
| within | C3M e123 | sr | gnm: factorizedc3m_e123|shape - B-gnm|vb_sr (prior esatto) | +0.201 [+0.158, +0.253] | 0.000 |
| within | C3M e123 | sr | gnm: factorizedc3m_e123|shape - B-flame2023|vb_sr (incrociato) | +0.205 [+0.159, +0.261] | 0.000 |
| within | C3M e123 | sr | flame2023_s1: factorizedc3m_e123|shape - B-gnm|vb_sr (incrociato) | +0.007 [-0.019, +0.033] | 0.304 |
| within | C3M e123 | sr | flame2023_s1: factorizedc3m_e123|shape - B-flame2023|vb_sr (prior esatto) | -0.012 [-0.039, +0.015] | 0.796 |
| within | C3M e123 | sr | flame2023: factorizedc3m_e123|shape - B-gnm|vb_sr (incrociato) | -0.057 [-0.087, -0.025] | 1.000 |
| within | C3M e123 | sr | flame2023: factorizedc3m_e123|shape - B-flame2023|vb_sr (prior esatto) | -0.084 [-0.116, -0.050] | 1.000 |
| C1 | fact. s1234 | fr | regen_ict - ict: factorized_s1234|form_cal | -0.000 [-0.000, +0.000] | 0.677 |
| C1 | fact. s1234 | fr | regen_gnm - gnm: factorized_s1234|form_cal | -0.000 [-0.000, +0.000] | 0.551 |
| C1 | fact. s2345 | fr | regen_ict - ict: factorized_s2345|form_cal | -0.000 [-0.000, +0.000] | 0.915 |
| C1 | fact. s2345 | fr | regen_gnm - gnm: factorized_s2345|form_cal | -0.000 [-0.000, +0.000] | 0.743 |
| C1 | C3M e123 | fr | regen_ict - ict: factorizedc3m_e123|form_cal | +0.000 [-0.000, +0.001] | 0.200 |
| C1 | C3M e123 | fr | regen_gnm - gnm: factorizedc3m_e123|form_cal | +0.000 [-0.000, +0.000] | 0.221 |
| C1 | fact. s1234 | sr | regen_ict - ict: factorized_s1234|shape | +0.000 [-0.000, +0.001] | 0.285 |
| C1 | fact. s1234 | sr | regen_gnm - gnm: factorized_s1234|shape | +0.000 [-0.000, +0.000] | 0.345 |
| C1 | fact. s2345 | sr | regen_ict - ict: factorized_s2345|shape | -0.000 [-0.000, +0.000] | 0.760 |
| C1 | fact. s2345 | sr | regen_gnm - gnm: factorized_s2345|shape | +0.000 [-0.000, +0.000] | 0.407 |
| C1 | C3M e123 | sr | regen_ict - ict: factorizedc3m_e123|shape | +0.000 [-0.000, +0.001] | 0.320 |
| C1 | C3M e123 | sr | regen_gnm - gnm: factorizedc3m_e123|shape | -0.000 [-0.000, +0.000] | 0.743 |

## D2

### D2, sonda lineare (diagnostica, non un metodo)

| dominio | braccio | GT | sonda [IC] | riferimento [IC] | delta [IC protocollo] | IC sola valutazione | P(<= 0) | permutazione (min, max) | lambda mediano (bordo) | R3 |
|---|---|---|---|---|---|---|---|---|---|---|
| facescape | fact. s1234 | sr (shape) | 0.889 [0.802, 0.903] | 0.745 [0.676, 0.796] | +0.144 [+0.074, +0.163] | [+0.114, +0.180] | 0.000 | +0.626 (+0.512, +0.728) | 177.828 (0.00) | passa |
| facescape | fact. s1234 | fr (form_cal) | 0.822 [0.696, 0.846] | 0.658 [0.586, 0.720] | +0.164 [+0.055, +0.191] | [+0.123, +0.212] | 0.000 | +0.560 (+0.455, +0.647) | 177.828 (0.00) | - |
| facescape | fact. s2345 | sr (shape) | 0.886 [0.791, 0.897] | 0.751 [0.681, 0.807] | +0.135 [+0.055, +0.153] | [+0.100, +0.174] | 0.000 | +0.628 (+0.499, +0.727) | 177.828 (0.00) | passa |
| facescape | fact. s2345 | fr (form_cal) | 0.813 [0.686, 0.837] | 0.666 [0.592, 0.726] | +0.147 [+0.039, +0.173] | [+0.107, +0.196] | 0.003 | +0.558 (+0.459, +0.651) | 177.828 (0.00) | - |
| facescape | C3M e123 | sr (shape) | 0.896 [0.808, 0.903] | 0.744 [0.679, 0.793] | +0.153 [+0.080, +0.170] | [+0.123, +0.189] | 0.000 | +0.614 (+0.507, +0.701) | 100 (0.00) | passa |
| facescape | C3M e123 | fr (form_cal) | 0.837 [0.707, 0.851] | 0.662 [0.584, 0.723] | +0.175 [+0.061, +0.203] | [+0.129, +0.230] | 0.001 | +0.551 (+0.451, +0.628) | 100 (0.00) | - |
| hifi3d | fact. s1234 | sr (shape) | 0.717 [0.573, 0.745] | 0.619 [0.544, 0.696] | +0.098 [-0.025, +0.110] | [+0.058, +0.138] | 0.101 | +0.519 (+0.447, +0.602) | 316.228 (0.00) | no |
| hifi3d | fact. s1234 | fr (form_cal) | 0.721 [0.417, 0.768] | 0.749 [0.675, 0.806] | -0.028 [-0.311, +0.013] | [-0.082, +0.028] | 0.961 | +0.335 (+0.217, +0.392) | 17.7828 (0.00) | - |
| hifi3d | fact. s2345 | sr (shape) | 0.696 [0.559, 0.735] | 0.611 [0.524, 0.690] | +0.085 [-0.026, +0.114] | [+0.052, +0.122] | 0.125 | +0.502 (+0.425, +0.578) | 439.285 (0.00) | no |
| hifi3d | fact. s2345 | fr (form_cal) | 0.673 [0.382, 0.736] | 0.732 [0.661, 0.794] | -0.060 [-0.339, -0.005] | [-0.114, -0.004] | 0.980 | +0.317 (+0.189, +0.381) | 56.2341 (0.05) | - |
| hifi3d | C3M e123 | sr (shape) | 0.699 [0.537, 0.729] | 0.592 [0.502, 0.675] | +0.107 [-0.025, +0.114] | [+0.072, +0.147] | 0.117 | +0.485 (+0.369, +0.555) | 177.828 (0.00) | no |
| hifi3d | C3M e123 | fr (form_cal) | 0.757 [0.459, 0.781] | 0.748 [0.672, 0.810] | +0.008 [-0.288, +0.029] | [-0.035, +0.054] | 0.897 | +0.304 (+0.218, +0.375) | 1.77828 (0.01) | - |

## Controlli

- C1-GT: {"regen_ict": {"sr_max_rel": 1.8606427896083169e-07, "fr_ratio_median": 21.534749583338574, "fr_ratio_rel_spread": 1.8913352387035372e-07, "pass": true}, "regen_gnm": {"sr_max_rel": 1.9306687535567794e-07, "fr_ratio_median": 18.51811052578926, "fr_ratio_rel_spread": 2.0563619930860656e-07, "pass": true}}
- C1-emb regen_ict: fact. s1234 original 6.0e-04, remesh 6.1e-04, down8k 1.8e-03, noisy 5.8e-04, up60k 1.5e-02 (scala 4.20); fact. s2345 original 7.5e-04, remesh 7.3e-04, down8k 2.7e-03, noisy 7.6e-04, up60k 1.1e-02 (scala 4.20); C3M e123 original 1.1e-03, remesh 1.8e-03, down8k 3.1e-03, noisy 1.1e-03, up60k 3.5e-02 (scala 4.20)
- C1-emb regen_gnm: fact. s1234 original 5.6e-04, remesh 6.7e-04, down8k 6.0e-04, noisy 6.1e-04, up60k 4.1e-04 (scala 4.17); fact. s2345 original 6.0e-04, remesh 6.7e-04, down8k 6.4e-04, noisy 6.8e-04, up60k 9.0e-04 (scala 4.17); C3M e123 original 1.2e-03, remesh 1.1e-03, down8k 1.3e-03, noisy 1.2e-03, up60k 1.6e-03 (scala 4.18)
- bfm: 100 soggetti, 99000 righe, 0 fuori maschera; d_P GT fra soggetti mediana 0.0730, IQR [0.0625, 0.0862]
- ict: 100 soggetti, 99000 righe, 0 fuori maschera; d_P GT fra soggetti mediana 0.0648, IQR [0.0542, 0.0784]
- gnm: 100 soggetti, 99000 righe, 0 fuori maschera; d_P GT fra soggetti mediana 0.0757, IQR [0.0636, 0.0901]
- regen_ict: 100 soggetti, 99000 righe, 0 fuori maschera; d_P GT fra soggetti mediana 0.0648, IQR [0.0542, 0.0784]
- regen_gnm: 100 soggetti, 99000 righe, 0 fuori maschera; d_P GT fra soggetti mediana 0.0757, IQR [0.0636, 0.0901]
- flame2023_s1: 200 soggetti, 398000 righe, 0 fuori maschera; d_P GT fra soggetti mediana 0.0706, IQR [0.0608, 0.0827]
- flame2023: 200 soggetti, 398000 righe, 0 fuori maschera; d_P GT fra soggetti mediana 0.0706, IQR [0.0608, 0.0827]
- D2 facescape: 88725 righe, K2 {"sr": {"ratio_median": 0.12943025253086415, "ratio_rel_spread": 2.8258168294178035e-07}, "fr": {"ratio_median": 8.398201600694684, "ratio_rel_spread": 2.4187481228263474e-07}}, K3 max |rho - pubblicato| 2.2e-16
- D2 hifi3d: 98224 righe, K2 {"sr": {"ratio_median": 0.21023924771852878, "ratio_rel_spread": 1.6667509644362946e-07}, "fr": {"ratio_median": 18.14943918386762, "ratio_rel_spread": 1.9652680024562583e-07}}, K3 max |rho - pubblicato| 1.1e-16
- generazione: {"regen_ict": {"original": [9409, 9409], "remesh": [6597, 6607], "down8k": [3272, 3285], "noisy": [9409, 9409], "up60k": [24120, 24128]}, "regen_gnm": {"original": [9022, 9022], "remesh": [6340, 6350], "down8k": [3161, 3179], "noisy": [9022, 9022], "up60k": [23164, 23177]}, "flame2023_s1": {"original": [6986, 6986], "remesh": [4901, 4912], "down8k": [2442, 2452], "noisy": [6986, 6986], "up60k": [17857, 17879]}, "flame2023": {"original": [1787, 1787], "remesh": [1255, 1261], "down8k": [628, 634], "noisy": [1787, 1787], "up60k": [4522, 4529]}}
- B flame2023_flame2023: fallite 0, regione {'n_refs': 200, 'n_vertices': 1446, 'kept_median': 0.7949076664801343, 'median_mm_median': 1.9765039849042032}, 38 s (64 processi), s/mesh mediana per passo {'load': 0.0, 'nicp': 0.87, 'va': 0.33, 'vb': 0.54}
- B flame2023_gnm: fallite 0, regione {'n_refs': 200, 'n_vertices': 7771, 'kept_median': 0.8425515406783418, 'median_mm_median': 2.466028059869748}, 121 s (64 processi), s/mesh mediana per passo {'load': 0.02, 'nicp': 3.66, 'va': 0.39, 'vb': 0.87}
- B flame2023_s1_flame2023: fallite 0, regione {'n_refs': 200, 'n_vertices': 1641, 'kept_median': 0.8606603245663123, 'median_mm_median': 1.546453428863163}, 51 s (64 processi), s/mesh mediana per passo {'load': 0.02, 'nicp': 1.02, 'va': 0.36, 'vb': 0.68}
- B flame2023_s1_gnm: fallite 0, regione {'n_refs': 200, 'n_vertices': 8361, 'kept_median': 0.9052316559521171, 'median_mm_median': 1.817742567436771}, 195 s (64 processi), s/mesh mediana per passo {'load': 0.03, 'nicp': 3.67, 'va': 0.4, 'vb': 1.2}
