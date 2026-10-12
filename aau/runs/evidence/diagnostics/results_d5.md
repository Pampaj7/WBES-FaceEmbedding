# Diagnostica D5: adattamento few-shot della testa (i-e2) a un generatore nuovo, risultati

Protocollo `PROTOCOL_D5.md` (sha256 3c2f7e7f..., commit 484d638), scritto prima dei numeri. FaceScape, HIFI3D e FaceVerse sono campioni di 3DMM non visti dai bracci, non dati reali. Generato da `aau/diagnostics/d5_fewshot.py`; numeri in `d5_delta.csv` (aggregati) e `d5_draws.csv` (per estrazione). Delta = rho(testa) - rho(d_P) dello stesso braccio sulle righe di test di T (viste: righe incrociate dei 100 soggetti valutati; FLAME: righe fra i soggetti 100-199). Punto e IC 95% = delta medio sulle estrazioni, bootstrap per soggetto di test (1000 repliche, condizionato alle estrazioni e ai fit); sd = deviazione standard fra le 10 estrazioni (k = 0 e pool intero: un fit). FLAME ha un pool di 100: nella colonna k = 200 entra con k = 100.

## Lettura preregistrata

- **EQ (primaria, peso uguale per sorgente): NON CONFERMATO**
  - fact. s1234: F(k = 20) SI; media sui 4 SR +0.060 [+0.040, +0.079]; FR per bersaglio FaceScape +0.073, HIFI3D +0.013, FaceVerse neutra +0.045, FLAME 2023 +0.014; k minimo con F: 20; k a meta' guadagno 20 (G(200) = +0.119), sull'incremento 50; bersagli con SR >= +0.05 e IC > 0 a k = 20: 2 su 4
  - fact. s2345: F(k = 20) NO; media sui 4 SR +0.039 [+0.016, +0.064]; FR per bersaglio FaceScape +0.045, HIFI3D +0.021, FaceVerse neutra -0.011, FLAME 2023 +0.015; k minimo con F: 50; k a meta' guadagno 50 (G(200) = +0.101), sull'incremento 50; bersagli con SR >= +0.05 e IC > 0 a k = 20: 2 su 4
- **variante W (bersaglio 1/2, secondaria): NON CONFERMATO**
  - fact. s1234: F(k = 20) SI; media sui 4 SR +0.067 [+0.048, +0.086]; FR per bersaglio FaceScape +0.080, HIFI3D +0.018, FaceVerse neutra +0.047, FLAME 2023 +0.017; k minimo con F: 20; k a meta' guadagno 50 (G(200) = +0.151), sull'incremento 50; bersagli con SR >= +0.05 e IC > 0 a k = 20: 3 su 4
  - fact. s2345: F(k = 20) NO; media sui 4 SR +0.047 [+0.025, +0.070]; FR per bersaglio FaceScape +0.052, HIFI3D +0.025, FaceVerse neutra +0.003, FLAME 2023 +0.017; k minimo con F: 50; k a meta' guadagno 50 (G(200) = +0.120), sull'incremento 50; bersagli con SR >= +0.05 e IC > 0 a k = 20: 2 su 4

Controlli: K0 (k = 0 = LOGO di D3 e2) scarto massimo 0.0e+00, passa True; K1 (d_P) 0.0e+00, passa True; K6 True; fit 784 / 784 con success, iterazioni 159-418, nodi i256-a10-10.srv.aau.dk.

## Guadagno contro d_P per bersaglio e k, EQ, d_P con SR (primaria)

**fact. s1234**

| bersaglio | k = 0 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 | k = 200 |
|---|---|---|---|---|---|---|---|
| FaceScape | +0.034 [+0.004, +0.066] | +0.043 [+0.015, +0.073] sd 0.005 | +0.064 [+0.036, +0.094] sd 0.007 | +0.089 [+0.060, +0.123] sd 0.009 | +0.117 [+0.085, +0.155] sd 0.006 | +0.130 [+0.096, +0.170] sd 0.003 | +0.140 [+0.104, +0.183] |
| HIFI3D | +0.024 [-0.016, +0.064] | +0.034 [-0.004, +0.075] sd 0.007 | +0.041 [+0.005, +0.079] sd 0.018 | +0.062 [+0.024, +0.102] sd 0.011 | +0.108 [+0.067, +0.150] sd 0.013 | +0.132 [+0.088, +0.175] sd 0.008 | +0.152 [+0.105, +0.197] |
| FaceVerse neutra | +0.016 [-0.048, +0.072] | +0.022 [-0.041, +0.079] sd 0.009 | +0.031 [-0.029, +0.087] sd 0.011 | +0.046 [-0.010, +0.097] sd 0.014 | +0.072 [+0.019, +0.119] sd 0.020 | +0.099 [+0.044, +0.146] sd 0.012 | +0.111 [+0.053, +0.163] |
| FLAME 2023 | +0.018 [-0.008, +0.045] | +0.026 [+0.000, +0.053] sd 0.004 | +0.028 [+0.002, +0.056] sd 0.002 | +0.042 [+0.015, +0.072] sd 0.008 | +0.062 [+0.035, +0.093] sd 0.003 | +0.073 [+0.046, +0.105] | +0.073 [+0.046, +0.105] |
| media dei 4 | +0.023 [+0.002, +0.042] | +0.031 [+0.011, +0.051] sd 0.003 | +0.041 [+0.021, +0.061] sd 0.006 | +0.060 [+0.040, +0.079] sd 0.007 | +0.090 [+0.069, +0.109] sd 0.009 | +0.109 [+0.087, +0.128] sd 0.005 | +0.119 [+0.096, +0.140] |

**fact. s2345**

| bersaglio | k = 0 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 | k = 200 |
|---|---|---|---|---|---|---|---|
| FaceScape | +0.016 [-0.012, +0.045] | +0.025 [-0.003, +0.056] sd 0.006 | +0.038 [+0.011, +0.067] sd 0.008 | +0.062 [+0.035, +0.093] sd 0.012 | +0.087 [+0.056, +0.120] sd 0.011 | +0.103 [+0.070, +0.141] sd 0.005 | +0.118 [+0.083, +0.158] |
| HIFI3D | +0.027 [-0.016, +0.078] | +0.041 [-0.000, +0.089] sd 0.011 | +0.049 [+0.008, +0.094] sd 0.016 | +0.069 [+0.027, +0.115] sd 0.016 | +0.105 [+0.064, +0.155] sd 0.014 | +0.133 [+0.086, +0.186] sd 0.008 | +0.151 [+0.102, +0.207] |
| FaceVerse neutra | -0.029 [-0.102, +0.052] | -0.021 [-0.095, +0.058] sd 0.007 | -0.016 [-0.090, +0.061] sd 0.013 | -0.014 [-0.085, +0.061] sd 0.019 | +0.018 [-0.050, +0.088] sd 0.022 | +0.041 [-0.027, +0.108] sd 0.016 | +0.062 [-0.016, +0.139] |
| FLAME 2023 | +0.019 [-0.006, +0.043] | +0.023 [-0.000, +0.046] sd 0.004 | +0.027 [+0.005, +0.049] sd 0.005 | +0.039 [+0.018, +0.061] sd 0.006 | +0.056 [+0.034, +0.078] sd 0.004 | +0.072 [+0.048, +0.096] | +0.072 [+0.048, +0.096] |
| media dei 4 | +0.008 [-0.017, +0.036] | +0.017 [-0.007, +0.043] sd 0.003 | +0.025 [+0.002, +0.050] sd 0.006 | +0.039 [+0.016, +0.064] sd 0.007 | +0.066 [+0.043, +0.091] sd 0.009 | +0.087 [+0.063, +0.114] sd 0.005 | +0.101 [+0.075, +0.130] |

## Guadagno contro d_P per bersaglio e k, EQ, d_F cal. con FR (vincolo)

**fact. s1234**

| bersaglio | k = 0 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 | k = 200 |
|---|---|---|---|---|---|---|---|
| FaceScape | +0.028 [+0.001, +0.056] | +0.037 [+0.010, +0.064] sd 0.007 | +0.052 [+0.023, +0.079] sd 0.010 | +0.073 [+0.043, +0.102] sd 0.010 | +0.096 [+0.062, +0.130] sd 0.007 | +0.106 [+0.069, +0.143] sd 0.007 | +0.116 [+0.077, +0.156] |
| HIFI3D | -0.001 [-0.020, +0.017] | +0.004 [-0.014, +0.023] sd 0.003 | +0.006 [-0.011, +0.025] sd 0.007 | +0.013 [-0.004, +0.034] sd 0.005 | +0.031 [+0.010, +0.054] sd 0.006 | +0.039 [+0.016, +0.065] sd 0.004 | +0.046 [+0.022, +0.074] |
| FaceVerse neutra | +0.027 [-0.029, +0.079] | +0.030 [-0.025, +0.082] sd 0.006 | +0.036 [-0.017, +0.085] sd 0.009 | +0.045 [-0.005, +0.091] sd 0.013 | +0.063 [+0.016, +0.106] sd 0.014 | +0.084 [+0.038, +0.128] sd 0.008 | +0.095 [+0.045, +0.141] |
| FLAME 2023 | +0.002 [-0.012, +0.018] | +0.005 [-0.009, +0.021] sd 0.002 | +0.008 [-0.006, +0.024] sd 0.002 | +0.014 [-0.001, +0.031] sd 0.002 | +0.018 [+0.003, +0.036] sd 0.002 | +0.022 [+0.006, +0.041] | +0.022 [+0.006, +0.041] |
| media dei 4 | +0.014 [-0.001, +0.030] | +0.019 [+0.004, +0.035] sd 0.003 | +0.025 [+0.010, +0.042] sd 0.004 | +0.036 [+0.022, +0.052] sd 0.005 | +0.052 [+0.038, +0.067] sd 0.005 | +0.063 [+0.047, +0.079] sd 0.003 | +0.070 [+0.053, +0.087] |

**fact. s2345**

| bersaglio | k = 0 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 | k = 200 |
|---|---|---|---|---|---|---|---|
| FaceScape | +0.017 [-0.008, +0.042] | +0.020 [-0.003, +0.045] sd 0.006 | +0.028 [+0.005, +0.052] sd 0.008 | +0.045 [+0.021, +0.069] sd 0.010 | +0.064 [+0.039, +0.092] sd 0.008 | +0.078 [+0.049, +0.109] sd 0.007 | +0.088 [+0.058, +0.123] |
| HIFI3D | +0.003 [-0.016, +0.024] | +0.010 [-0.009, +0.031] sd 0.004 | +0.013 [-0.005, +0.034] sd 0.006 | +0.021 [+0.002, +0.045] sd 0.008 | +0.034 [+0.013, +0.061] sd 0.007 | +0.044 [+0.020, +0.074] sd 0.004 | +0.051 [+0.024, +0.081] |
| FaceVerse neutra | -0.024 [-0.092, +0.047] | -0.018 [-0.085, +0.049] sd 0.005 | -0.014 [-0.079, +0.050] sd 0.011 | -0.011 [-0.075, +0.053] sd 0.017 | +0.010 [-0.051, +0.067] sd 0.016 | +0.028 [-0.034, +0.087] sd 0.014 | +0.041 [-0.025, +0.106] |
| FLAME 2023 | +0.009 [-0.003, +0.020] | +0.010 [-0.001, +0.021] sd 0.002 | +0.013 [+0.002, +0.024] sd 0.002 | +0.015 [+0.005, +0.026] sd 0.003 | +0.020 [+0.009, +0.031] sd 0.003 | +0.025 [+0.013, +0.036] | +0.025 [+0.013, +0.036] |
| media dei 4 | +0.001 [-0.018, +0.022] | +0.006 [-0.013, +0.025] sd 0.002 | +0.010 [-0.009, +0.030] sd 0.004 | +0.018 [+0.000, +0.037] sd 0.005 | +0.032 [+0.015, +0.051] sd 0.005 | +0.044 [+0.026, +0.063] sd 0.004 | +0.051 [+0.031, +0.072] |

## Guadagno contro d_P per bersaglio e k, variante W, d_P con SR (primaria)

**fact. s1234**

| bersaglio | k = 0 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 | k = 200 |
|---|---|---|---|---|---|---|---|
| FaceScape | +0.034 [+0.004, +0.066] | +0.046 [+0.019, +0.076] sd 0.006 | +0.069 [+0.043, +0.100] sd 0.009 | +0.099 [+0.070, +0.134] sd 0.010 | +0.132 [+0.098, +0.174] sd 0.006 | +0.152 [+0.116, +0.197] sd 0.004 | +0.165 [+0.127, +0.212] |
| HIFI3D | +0.024 [-0.016, +0.064] | +0.038 [+0.002, +0.075] sd 0.009 | +0.049 [+0.016, +0.084] sd 0.017 | +0.071 [+0.034, +0.112] sd 0.009 | +0.134 [+0.087, +0.179] sd 0.015 | +0.173 [+0.125, +0.223] sd 0.011 | +0.204 [+0.151, +0.260] |
| FaceVerse neutra | +0.016 [-0.048, +0.072] | +0.037 [-0.025, +0.093] sd 0.010 | +0.038 [-0.018, +0.093] sd 0.015 | +0.051 [+0.000, +0.099] sd 0.017 | +0.079 [+0.031, +0.124] sd 0.021 | +0.121 [+0.071, +0.165] sd 0.017 | +0.150 [+0.091, +0.204] |
| FLAME 2023 | +0.018 [-0.008, +0.045] | +0.027 [+0.003, +0.054] sd 0.005 | +0.033 [+0.008, +0.059] sd 0.004 | +0.047 [+0.021, +0.076] sd 0.007 | +0.068 [+0.041, +0.099] sd 0.004 | +0.083 [+0.054, +0.117] | +0.083 [+0.054, +0.117] |
| media dei 4 | +0.023 [+0.002, +0.042] | +0.037 [+0.017, +0.056] sd 0.004 | +0.047 [+0.028, +0.065] sd 0.008 | +0.067 [+0.048, +0.086] sd 0.009 | +0.103 [+0.082, +0.122] sd 0.010 | +0.132 [+0.110, +0.153] sd 0.006 | +0.151 [+0.126, +0.173] |

**fact. s2345**

| bersaglio | k = 0 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 | k = 200 |
|---|---|---|---|---|---|---|---|
| FaceScape | +0.016 [-0.012, +0.045] | +0.026 [-0.001, +0.056] sd 0.007 | +0.041 [+0.013, +0.070] sd 0.011 | +0.072 [+0.043, +0.103] sd 0.012 | +0.108 [+0.075, +0.146] sd 0.012 | +0.133 [+0.095, +0.174] sd 0.006 | +0.147 [+0.108, +0.193] |
| HIFI3D | +0.027 [-0.016, +0.078] | +0.045 [+0.006, +0.091] sd 0.009 | +0.057 [+0.018, +0.102] sd 0.013 | +0.075 [+0.034, +0.121] sd 0.017 | +0.127 [+0.080, +0.179] sd 0.015 | +0.167 [+0.115, +0.224] sd 0.010 | +0.191 [+0.135, +0.252] |
| FaceVerse neutra | -0.029 [-0.102, +0.052] | -0.020 [-0.091, +0.048] sd 0.007 | -0.011 [-0.080, +0.054] sd 0.011 | +0.001 [-0.062, +0.064] sd 0.018 | +0.027 [-0.039, +0.090] sd 0.029 | +0.053 [-0.009, +0.113] sd 0.019 | +0.062 [-0.004, +0.131] |
| FLAME 2023 | +0.019 [-0.006, +0.043] | +0.020 [-0.004, +0.043] sd 0.006 | +0.025 [+0.003, +0.047] sd 0.006 | +0.041 [+0.020, +0.063] sd 0.008 | +0.062 [+0.039, +0.085] sd 0.006 | +0.080 [+0.055, +0.105] | +0.080 [+0.055, +0.105] |
| media dei 4 | +0.008 [-0.017, +0.036] | +0.018 [-0.006, +0.041] sd 0.003 | +0.028 [+0.006, +0.051] sd 0.005 | +0.047 [+0.025, +0.070] sd 0.007 | +0.081 [+0.059, +0.105] sd 0.012 | +0.108 [+0.083, +0.134] sd 0.006 | +0.120 [+0.094, +0.147] |

## Guadagno contro d_P per bersaglio e k, variante W, d_F cal. con FR (vincolo)

**fact. s1234**

| bersaglio | k = 0 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 | k = 200 |
|---|---|---|---|---|---|---|---|
| FaceScape | +0.028 [+0.001, +0.056] | +0.040 [+0.015, +0.066] sd 0.007 | +0.056 [+0.029, +0.084] sd 0.012 | +0.080 [+0.052, +0.111] sd 0.011 | +0.106 [+0.073, +0.142] sd 0.009 | +0.120 [+0.085, +0.158] sd 0.007 | +0.132 [+0.094, +0.173] |
| HIFI3D | -0.001 [-0.020, +0.017] | +0.006 [-0.011, +0.023] sd 0.004 | +0.010 [-0.007, +0.028] sd 0.007 | +0.018 [-0.001, +0.038] sd 0.005 | +0.040 [+0.016, +0.068] sd 0.007 | +0.053 [+0.027, +0.085] sd 0.005 | +0.062 [+0.034, +0.098] |
| FaceVerse neutra | +0.027 [-0.029, +0.079] | +0.041 [-0.017, +0.094] sd 0.007 | +0.040 [-0.013, +0.089] sd 0.014 | +0.047 [-0.002, +0.090] sd 0.016 | +0.065 [+0.019, +0.104] sd 0.015 | +0.096 [+0.051, +0.137] sd 0.011 | +0.119 [+0.066, +0.166] |
| FLAME 2023 | +0.002 [-0.012, +0.018] | +0.007 [-0.007, +0.022] sd 0.003 | +0.011 [-0.003, +0.026] sd 0.002 | +0.017 [+0.002, +0.034] sd 0.002 | +0.022 [+0.005, +0.040] sd 0.002 | +0.026 [+0.009, +0.046] | +0.026 [+0.009, +0.046] |
| media dei 4 | +0.014 [-0.001, +0.030] | +0.023 [+0.008, +0.040] sd 0.003 | +0.029 [+0.014, +0.045] sd 0.006 | +0.040 [+0.025, +0.056] sd 0.007 | +0.058 [+0.044, +0.074] sd 0.004 | +0.074 [+0.058, +0.090] sd 0.003 | +0.085 [+0.067, +0.104] |

**fact. s2345**

| bersaglio | k = 0 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 | k = 200 |
|---|---|---|---|---|---|---|---|
| FaceScape | +0.017 [-0.008, +0.042] | +0.022 [-0.001, +0.047] sd 0.007 | +0.030 [+0.008, +0.055] sd 0.011 | +0.052 [+0.027, +0.078] sd 0.009 | +0.080 [+0.052, +0.110] sd 0.009 | +0.099 [+0.068, +0.136] sd 0.008 | +0.111 [+0.075, +0.152] |
| HIFI3D | +0.003 [-0.016, +0.024] | +0.012 [-0.006, +0.033] sd 0.004 | +0.016 [-0.001, +0.037] sd 0.005 | +0.025 [+0.006, +0.049] sd 0.009 | +0.043 [+0.021, +0.072] sd 0.006 | +0.056 [+0.030, +0.088] sd 0.005 | +0.063 [+0.035, +0.095] |
| FaceVerse neutra | -0.024 [-0.092, +0.047] | -0.012 [-0.075, +0.048] sd 0.007 | -0.006 [-0.065, +0.052] sd 0.010 | +0.003 [-0.054, +0.057] sd 0.017 | +0.017 [-0.039, +0.071] sd 0.024 | +0.032 [-0.021, +0.083] sd 0.015 | +0.031 [-0.023, +0.089] |
| FLAME 2023 | +0.009 [-0.003, +0.020] | +0.011 [+0.001, +0.022] sd 0.002 | +0.013 [+0.003, +0.024] sd 0.003 | +0.017 [+0.008, +0.028] sd 0.004 | +0.023 [+0.012, +0.035] sd 0.003 | +0.030 [+0.018, +0.043] | +0.030 [+0.018, +0.043] |
| media dei 4 | +0.001 [-0.018, +0.022] | +0.008 [-0.010, +0.026] sd 0.003 | +0.014 [-0.003, +0.031] sd 0.005 | +0.024 [+0.008, +0.041] sd 0.004 | +0.041 [+0.024, +0.059] sd 0.007 | +0.054 [+0.037, +0.073] sd 0.004 | +0.059 [+0.041, +0.080] |

## Incremento su k = 0 (media dei 4, SR), EQ e W

| braccio | variante | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 | k = 200 |
|---|---|---|---|---|---|---|---|
| fact. s1234 | EQ | +0.009 [+0.005, +0.013] sd 0.003 | +0.018 [+0.013, +0.024] sd 0.006 | +0.037 [+0.029, +0.046] sd 0.007 | +0.067 [+0.054, +0.081] sd 0.009 | +0.086 [+0.070, +0.102] sd 0.005 | +0.096 [+0.078, +0.115] |
| fact. s1234 | W | +0.014 [+0.008, +0.022] sd 0.004 | +0.024 [+0.017, +0.033] sd 0.008 | +0.044 [+0.033, +0.057] sd 0.009 | +0.081 [+0.064, +0.099] sd 0.010 | +0.110 [+0.090, +0.131] sd 0.006 | +0.128 [+0.106, +0.151] |
| fact. s2345 | EQ | +0.009 [+0.005, +0.012] sd 0.003 | +0.016 [+0.011, +0.022] sd 0.006 | +0.031 [+0.022, +0.041] sd 0.007 | +0.058 [+0.044, +0.073] sd 0.009 | +0.079 [+0.063, +0.097] sd 0.005 | +0.093 [+0.073, +0.115] |
| fact. s2345 | W | +0.009 [+0.001, +0.017] sd 0.003 | +0.020 [+0.010, +0.029] sd 0.005 | +0.039 [+0.026, +0.052] sd 0.007 | +0.073 [+0.054, +0.092] sd 0.012 | +0.100 [+0.079, +0.123] sd 0.006 | +0.112 [+0.088, +0.139] |

## W - EQ (SR), replica per replica

| braccio | bersaglio | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 | k = 200 |
|---|---|---|---|---|---|---|---|
| fact. s1234 | FaceScape | +0.003 [-0.001, +0.008] | +0.005 [+0.002, +0.009] | +0.009 [+0.005, +0.014] | +0.015 [+0.009, +0.022] | +0.022 [+0.014, +0.030] | +0.025 [+0.016, +0.035] |
| fact. s1234 | HIFI3D | +0.004 [-0.004, +0.012] | +0.008 [-0.000, +0.017] | +0.010 [+0.002, +0.019] | +0.027 [+0.015, +0.041] | +0.041 [+0.027, +0.057] | +0.052 [+0.035, +0.071] |
| fact. s1234 | FaceVerse neutra | +0.015 [-0.002, +0.033] | +0.007 [-0.010, +0.025] | +0.005 [-0.014, +0.026] | +0.007 [-0.010, +0.026] | +0.022 [+0.001, +0.044] | +0.039 [+0.011, +0.070] |
| fact. s1234 | FLAME 2023 | +0.002 [-0.002, +0.006] | +0.004 [+0.000, +0.009] | +0.005 [+0.001, +0.009] | +0.006 [+0.001, +0.012] | +0.010 [+0.003, +0.017] | +0.010 [+0.003, +0.017] |
| fact. s1234 | media dei 4 | +0.006 [+0.001, +0.011] | +0.006 [+0.002, +0.012] | +0.007 [+0.002, +0.013] | +0.014 [+0.008, +0.020] | +0.024 [+0.017, +0.031] | +0.032 [+0.023, +0.041] |
| fact. s2345 | FaceScape | +0.001 [-0.003, +0.005] | +0.003 [-0.001, +0.007] | +0.009 [+0.004, +0.015] | +0.021 [+0.013, +0.030] | +0.030 [+0.021, +0.041] | +0.030 [+0.020, +0.042] |
| fact. s2345 | HIFI3D | +0.004 [-0.002, +0.011] | +0.008 [+0.001, +0.014] | +0.007 [-0.000, +0.014] | +0.022 [+0.011, +0.034] | +0.034 [+0.020, +0.048] | +0.040 [+0.025, +0.057] |
| fact. s2345 | FaceVerse neutra | +0.001 [-0.022, +0.023] | +0.005 [-0.019, +0.028] | +0.014 [-0.006, +0.035] | +0.009 [-0.009, +0.030] | +0.012 [-0.012, +0.037] | -0.000 [-0.040, +0.039] |
| fact. s2345 | FLAME 2023 | -0.004 [-0.007, +0.000] | -0.002 [-0.006, +0.001] | +0.002 [-0.001, +0.006] | +0.006 [+0.001, +0.012] | +0.008 [+0.001, +0.015] | +0.008 [+0.001, +0.015] |
| fact. s2345 | media dei 4 | +0.001 [-0.005, +0.006] | +0.003 [-0.003, +0.009] | +0.008 [+0.002, +0.014] | +0.015 [+0.008, +0.021] | +0.021 [+0.013, +0.029] | +0.019 [+0.008, +0.031] |

## Dispersione fra estrazioni (media dei 4, SR, EQ)

| braccio | k | sd | min | max | estrazioni >= +0.05 |
|---|---|---|---|---|---|
| fact. s1234 | 5 | 0.003 | +0.026 | +0.035 | 0 / 10 |
| fact. s1234 | 10 | 0.006 | +0.035 | +0.054 | 1 / 10 |
| fact. s1234 | 20 | 0.007 | +0.050 | +0.073 | 9 / 10 |
| fact. s1234 | 50 | 0.009 | +0.078 | +0.103 | 10 / 10 |
| fact. s1234 | 100 | 0.005 | +0.101 | +0.115 | 10 / 10 |
| fact. s2345 | 5 | 0.003 | +0.011 | +0.019 | 0 / 10 |
| fact. s2345 | 10 | 0.006 | +0.014 | +0.033 | 0 / 10 |
| fact. s2345 | 20 | 0.007 | +0.028 | +0.050 | 0 / 10 |
| fact. s2345 | 50 | 0.009 | +0.055 | +0.080 | 10 / 10 |
| fact. s2345 | 100 | 0.005 | +0.081 | +0.097 | 10 / 10 |

## k = 200 accanto alla testa congiunta di D3 e2 (descrittivo, SR)

La congiunta ha in piu' i pool degli altri bersagli e FLAME 0-99: non e' lo stesso disegno.

| bersaglio | braccio | D5 k = 0 | D5 k = 200 EQ | D5 k = 200 W | congiunta D3 e2 |
|---|---|---|---|---|---|
| FaceScape | fact. s1234 | +0.034 [+0.004, +0.066] | +0.140 [+0.104, +0.183] | +0.165 [+0.127, +0.212] | +0.142 [+0.106, +0.185] |
| FaceScape | fact. s2345 | +0.016 [-0.012, +0.045] | +0.118 [+0.083, +0.158] | +0.147 [+0.108, +0.193] | +0.120 [+0.084, +0.161] |
| HIFI3D | fact. s1234 | +0.024 [-0.016, +0.064] | +0.152 [+0.105, +0.197] | +0.204 [+0.151, +0.260] | +0.153 [+0.106, +0.199] |
| HIFI3D | fact. s2345 | +0.027 [-0.016, +0.078] | +0.151 [+0.102, +0.207] | +0.191 [+0.135, +0.252] | +0.155 [+0.103, +0.210] |
| FaceVerse neutra | fact. s1234 | +0.016 [-0.048, +0.072] | +0.111 [+0.053, +0.163] | +0.150 [+0.091, +0.204] | +0.112 [+0.056, +0.166] |
| FaceVerse neutra | fact. s2345 | -0.029 [-0.102, +0.052] | +0.062 [-0.016, +0.139] | +0.062 [-0.004, +0.131] | +0.059 [-0.010, +0.127] |
| FLAME 2023 | fact. s1234 | +0.018 [-0.008, +0.045] | +0.073 [+0.046, +0.105] | +0.083 [+0.054, +0.117] | +0.075 [+0.046, +0.106] |
| FLAME 2023 | fact. s2345 | +0.019 [-0.006, +0.043] | +0.072 [+0.048, +0.096] | +0.080 [+0.055, +0.105] | +0.072 [+0.048, +0.096] |

## k a meta' guadagno per bersaglio (SR, G(k) >= G(200) / 2)

| braccio | variante | FaceScape | HIFI3D | FaceVerse neutra | FLAME 2023 | media dei 4 |
|---|---|---|---|---|---|---|
| fact. s1234 | EQ | 20 | 50 | 50 | 20 | 20 |
| fact. s1234 | W | 20 | 50 | 50 | 20 | 50 |
| fact. s2345 | EQ | 20 | 50 | 100 | 20 | 50 |
| fact. s2345 | W | 50 | 50 | 100 | 20 | 50 |

## Note del coder, scritte DOPO i numeri (commento, non regole)

- **Verdetto preregistrato: NON CONFERMATO** (EQ; anche la variante W). s1234 passa a k = 20, ma di poco: +0.060
  [+0.040, +0.079], 9 estrazioni su 10 >= +0.05 (minimo +0.050). s2345 non passa: +0.039 [+0.016, +0.064], 0 su 10
  (massimo +0.050). A k = 50 F vale in entrambi i semi (+0.090 / +0.066; W +0.103 / +0.081).
  - **Come leggere k = 50.** 50 e' il k piu' piccolo della griglia per cui il criterio vale in entrambi i semi. La
    soglia vera sta fra 20 e 50, valori non campionati. Il risultato vale per la media dei 4 bersagli, non per ognuno:
    a k = 50 FaceVerse neutra s2345 vale +0.018 [-0.050, +0.088].
- **Il guadagno dato dai soggetti del bersaglio e' reale in entrambi i semi gia' da k = 5.** Incremento su k = 0 (media
  dei 4): k = 5 +0.009 [+0.005, +0.013] / +0.009 [+0.005, +0.012]; k = 20 +0.037 [+0.029, +0.046] / +0.031 [+0.022,
  +0.041]. s2345 non passa per il livello di partenza: il suo LOGO vale +0.008, contro +0.023 per s1234.
- **Il deficit di s2345 viene da FaceVerse neutra**: -0.029 a k = 0, -0.014 a k = 20, +0.062 [-0.016, +0.139] a
  k = 200; l'IC non esclude mai lo 0. Gli altri tre bersagli di s2345 a k = 20: FaceScape +0.062, HIFI3D +0.069,
  FLAME +0.039. La media aritmetica di questi tre punti, +0.057, e' POST HOC e senza IC: non e' una regola.
- **k a meta' guadagno**: s1234 EQ k = 20, ma al limite (G(20) = +0.0597 contro G(200) / 2 = +0.0595); s2345 k = 50;
  sull'incremento k = 50 in entrambi i semi. Lettura: meta' del guadagno raggiungibile chiede 20-50 identita', quasi
  tutto chiede 100-200.
- **k = 200 EQ ~ testa congiunta di D3 e2** (scarto <= 0.004 su ogni bersaglio e braccio). Il guadagno della congiunta
  viene dal pool del bersaglio stesso, non dai pool degli altri bersagli.
- **W contro EQ**: W vince e lo scarto cresce con k. Media dei 4: k = 5 +0.006 [+0.001, +0.011] / +0.001 [-0.005,
  +0.006]; k = 20 +0.007 / +0.008 (IC sopra 0); k = 200 +0.032 / +0.019. E' il contrario dell'attesa ("a k piccolo
  conviene pesare di piu'"): a k piccolo pesare di piu' il bersaglio quasi non sposta nulla.
- **La dispersione fra estrazioni e' piccola**: sd della media dei 4 tra 0.003 e 0.009, contro una semi-ampiezza
  dell'IC di circa 0.02. Per bersaglio la sd arriva a 0.022 (FaceVerse, k = 50). Conta piu' il rumore dei soggetti di
  test che la scelta di quali k soggetti etichettare.
- **FR non peggiora mai**: a k = 20 il delta FR minimo e' -0.011 (FaceVerse, s2345); media dei 4 +0.036 / +0.018.
- **Limiti.** La configurazione (r, lambda) e' quella di D3 e2, ferma al bordo lambda = 1e-6; il critic di D3 ha
  mostrato che a k = 0 estrapola peggio di 1e-5, e qui non c'e' una nuova CV con i dati del bersaglio. Gli embedding
  sono congelati. I bersagli sono campioni di 3DMM, non dati reali. c usa anche il blocco del bersaglio. Gli IC sono
  condizionati alle estrazioni e ai fit, e i soggetti di test sono 100 per bersaglio.
  - **Dipendenza da lambda (POST HOC, dal critic di D5, RISERVE senza difetti nel codice).** Il critic ha rifittato
    con lambda = 1e-5 a k = 0, 20, 50 (EQ, stessi r, stesse estrazioni, stesse righe e repliche); file in
    `/home/create.aau.dk/ga41wf/.critic_tmp/d5lam/`, fuori dal repo; job 1070080 sul nodo i256-a10-10. Le sue teste
    a 1e-6 coincidono con quelle salvate (scarto 0), e i numeri sotto sono ricalcolati dal coder col suo `ana.py` sul
    suo `V.pkl` (job 1070085). Media dei 4, SR, s1234 / s2345, lambda 1e-5 contro 1e-6:
    - k = 0: +0.033 / +0.020 contro +0.023 / +0.008;
    - k = 20: +0.066 [+0.049, +0.083] / +0.047 [+0.028, +0.067] contro +0.060 / +0.039. s2345 resta sotto +0.05,
      quindi il verdetto NON CONFERMATO regge anche a 1e-5;
    - k = 50: nessuna differenza (+0.088 / +0.068 contro +0.090 / +0.066).
  - **FaceVerse neutra s2345: con 1e-5 il deficit si dimezza circa.** k = 0 +0.008 contro -0.029; k = 20 +0.026
    contro -0.014; k = 50 +0.049 [-0.012, +0.102] contro +0.018. L'IC continua a non escludere lo 0.
  - **Avvertenza.** lambda = 1e-5 e' una scelta gia' informata dal test: viene dal LOGO del critic di D3, letto sui
    bersagli. Per questo non sostituisce la configurazione preregistrata. Misura solo quanto conta lambda: a k <= 20
    alza la media dei 4 di 0.006-0.012 (k = 0 +0.010 / +0.012, k = 20 +0.006 / +0.008), a k = 50 non la sposta.
- **Riproducibilita'.** Valutazione rilanciata (job 1070043, stesso nodo) dopo due correzioni d'uscita: l'etichetta del
  seme di FLAME in `d5/controls.json` (diceva il seme di D1; i conteggi usati erano gia' quelli di `cross_half`, come
  conferma K0 = 0) e la "sd 0.000" (in realta' ~1e-17) nelle celle della media dei 4 a k = 0 e k = 200. `d5_delta.csv`,
  `d5_draws.csv` e `d5/readings.json` sono identici byte per byte al primo run (job 1069966).
