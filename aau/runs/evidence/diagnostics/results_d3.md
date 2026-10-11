# Diagnostica D3: risultati (emendamento 1, leave-one-generator-out)

Protocollo `PROTOCOL_D3.md` (sha256 c75619e6..., commit 21afea8) ed emendamento 1 (`PROTOCOL_D3_emendamento_1.md`, sha256 3f75da65..., commit 49fead6), scritti prima dei numeri; le anteprime del critic sono dichiarate nell'emendamento (sez. 1). FaceScape, HIFI3D e FaceVerse sono campioni di 3DMM non visti dai bracci, non dati reali; reale e' solo FaMoS. Generato da `aau/diagnostics/d3_summarize.py`; numeri in `d3_spearman.csv`, `d3_delta.csv`, `d3_curve.csv`, `d3_heads.csv`, `d3_cv.csv`. IC 95% percentile, 1000 repliche per soggetto di test, condizionati alle teste addestrate.

## Letture

- **R_LOGO, testa (i), tutte le sorgenti tranne T (primaria): NO**
  - fact. s1234: NO (n_SR 0 su 4, FR >= -0.03 ovunque: True); facescape SR +0.038 [+0.014, +0.065], FR +0.030; hifi3d SR +0.000 [+0.000, +0.000], FR +0.000; faceverse_neutral SR +0.026 [-0.016, +0.071], FR +0.019; flame2023_s1 SR -0.001 [-0.018, +0.014], FR -0.005
  - fact. s2345: NO (n_SR 0 su 4, FR >= -0.03 ovunque: True); facescape SR +0.000 [+0.000, +0.000], FR +0.007; hifi3d SR +0.000 [+0.000, +0.000], FR +0.001; faceverse_neutral SR -0.003 [-0.047, +0.046], FR -0.000; flame2023_s1 SR +0.000 [+0.000, +0.000], FR +0.002
- **Secondaria: (ii) Mahalanobis intra-soggetto: NO**
  - fact. s1234: NO (n_SR 0 su 4, FR >= -0.03 ovunque: False); facescape SR -0.000 [-0.041, +0.042], FR +0.010; hifi3d SR -0.069 [-0.126, -0.011], FR -0.031; faceverse_neutral SR +0.056 [-0.016, +0.121], FR +0.035; flame2023_s1 SR -0.021 [-0.050, +0.006], FR -0.009
  - fact. s2345: NO (n_SR 0 su 4, FR >= -0.03 ovunque: True); facescape SR +0.012 [-0.022, +0.047], FR +0.014; hifi3d SR -0.059 [-0.111, -0.012], FR -0.026; faceverse_neutral SR +0.025 [-0.033, +0.084], FR +0.009; flame2023_s1 SR -0.013 [-0.038, +0.012], FR +0.000
- **Secondaria: (i) senza FaMoS: NO**
  - fact. s1234: NO (n_SR 0 su 4, FR >= -0.03 ovunque: True); facescape SR +0.000 [+0.000, +0.000], FR +0.006; hifi3d SR +0.000 [+0.000, +0.000], FR +0.000; faceverse_neutral SR +0.024 [-0.016, +0.067], FR +0.016; flame2023_s1 SR -0.013 [-0.030, +0.003], FR -0.006
  - fact. s2345: NO (n_SR 0 su 4, FR >= -0.03 ovunque: True); facescape SR +0.000 [+0.000, +0.000], FR +0.007; hifi3d SR +0.000 [+0.000, +0.000], FR +0.000; faceverse_neutral SR +0.008 [-0.040, +0.059], FR +0.009; flame2023_s1 SR +0.000 [+0.000, +0.000], FR +0.002

Delta medio sui 4 bersagli primari (replica per replica):

| braccio | testa | SR | FR |
|---|---|---|---|
| fact. s1234 | (i) appresa, tutte tranne T | +0.016 [+0.003, +0.029] | +0.011 [-0.000, +0.023] |
| fact. s1234 | (i) senza FaMoS | +0.003 [-0.008, +0.014] | +0.004 [-0.005, +0.014] |
| fact. s1234 | (ii) intra-soggetto, tutte tranne T | -0.009 [-0.036, +0.016] | +0.001 [-0.017, +0.020] |
| fact. s1234 | (ii) senza FaMoS | -0.016 [-0.043, +0.010] | -0.005 [-0.023, +0.014] |
| fact. s2345 | (i) appresa, tutte tranne T | -0.001 [-0.012, +0.011] | +0.002 [-0.007, +0.012] |
| fact. s2345 | (i) senza FaMoS | +0.002 [-0.010, +0.015] | +0.005 [-0.005, +0.015] |
| fact. s2345 | (ii) intra-soggetto, tutte tranne T | -0.009 [-0.032, +0.013] | -0.001 [-0.016, +0.015] |
| fact. s2345 | (ii) senza FaMoS | -0.014 [-0.038, +0.007] | -0.006 [-0.021, +0.010] |
| C3M e123 | (i) appresa, tutte tranne T | +0.007 [-0.006, +0.020] | +0.009 [-0.001, +0.020] |
| C3M e123 | (i) senza FaMoS | -0.002 [-0.015, +0.011] | +0.005 [-0.006, +0.016] |
| C3M e123 | (ii) intra-soggetto, tutte tranne T | -0.015 [-0.037, +0.006] | -0.008 [-0.023, +0.009] |
| C3M e123 | (ii) senza FaMoS | -0.019 [-0.040, +0.003] | -0.010 [-0.026, +0.006] |

Curva sul numero di sorgenti: il guadagno cresce (media non decrescente da k = 1 a 7 su >= 3 bersagli, entrambi i semi)? (i): **non risolto**; (ii): **no**.

Note del coder, scritte DOPO i numeri (commento, non regole):

- R_LOGO e' NO in entrambi i semi: nessun bersaglio arriva a +0.05. Il guadagno piu' grande e' FaceScape s1234, +0.038 [+0.014, +0.065], sotto la soglia; la media sui 4 bersagli e' +0.016 [+0.003, +0.029] (s1234) e -0.001 [-0.012, +0.011] (s2345). FR non peggiora oltre -0.03 da nessuna parte con la testa (i). Verdetto preregistrato, invariato dopo il critic e dopo l'emendamento 2 (POST HOC, sotto).
- La CV dentro le sorgenti sceglie r = 0 (d_P invariata) in 3 impostazioni su 9 per s1234 e in 6 su 9 per s2345; dove sceglie una correzione, lambda e' sempre sul bordo basso (1e-4). **ARTEFATTO DELLA GRIGLIA DI LAMBDA** (critic, emendamento 2): la penalita' agisce su x non standardizzate e con lambda >= 1e-2 azzera W, quindi dei 5 valori preregistrati solo il bordo lasciava una correzione. Con lambda fino a 1e-6 e la perdita riscalata la CV sceglie lambda = 1e-6 in 30 impostazioni su 30 e r = 64 in 27 (r = 32 nelle altre), con guadagno di CV sulle sorgenti da +0.045 a +0.064; di nuovo sul bordo della griglia estesa (riportato, griglia non allargata).
- Sorgenti singole nella catena preregistrata (iperparametri fissi, k = 1): FLAME da solo porta FaceScape a +0.060 (s1234) e +0.086 (C3M), FaMoS da solo a +0.068 e +0.071, HIFI3D da solo a -0.074 (s1234); tutte insieme danno meno della sorgente migliore. **LETTURA CORRETTA** (critic, emendamento 2): ne avevo dedotto che una correzione lineare comune non esiste, ed e' FALSO. La correzione comune esiste se il generatore e' visto: la testa congiunta su tutte le 8 sorgenti, compresi i pool non valutati dei bersagli, migliora i 4 bersagli insieme (media +0.120 [+0.098, +0.142] s1234, +0.102 [+0.076, +0.130] s2345). Fallisce l'estrapolazione a un generatore mai visto, che resta piccola (LOGO con la catena corretta: media +0.025 [+0.004, +0.045] e +0.011 [-0.013, +0.037]). La curva piatta di s2345 e il suo "non risolto" erano un **ARTEFATTO DELLA GRIGLIA** (r = 0 scelto): con la catena corretta il guadagno cresce col numero di sorgenti su tutti i bersagli (una sola sorgente da -0.03 a -0.08 in media, tutte e 7 da -0.03 a +0.06).
- (ii) intra-soggetto stimata sulle sorgenti peggiora HIFI3D (-0.069 e -0.059 in SR con IC sotto 0; FR -0.031 e -0.026: per s1234 oltre la soglia), e' nulla su FaceScape e +0.056 / +0.025 su FaceVerse (IC che tocca lo 0): la covarianza intra-soggetto delle sorgenti non e' quella del bersaglio (l'anteprima P4, +0.05-0.07 su FaceScape, la stimava sul bersaglio stesso). Non ha lambda ne' perdita: l'emendamento 2 non la tocca.
- CORAL (esplorativa, covarianza dal pool non valutato): da -0.12 a -0.21 su FaceScape e HIFI3D, circa 0 su FaceVerse, come l'anteprima P4.
- La media sulle etichette alza d_P su FaceScape (0.747 -> 0.802, s1234) e il guadagno della testa (i) scende da +0.038 (righe incrociate) a +0.019 (mediate). **LETTA MALE** (critic, emendamento 2): non e' riduzione del rumore ne' un effetto della original. Il guadagno viene dall'avere la STESSA discretizzazione ai due lati della riga: d_P FaceScape s1234 remesh-remesh 0.800, up60k-up60k 0.802, original-original 0.778, contro 0.747 incrociate (remesh-remesh - incrociate +0.053 [+0.041, +0.068]); HIFI3D +0.020, FaceVerse +0.012, FLAME +0.026. L'aumento al test da provare e' una ridiscretizzazione canonica (sempre lo stesso remesher ai due lati): colpisce il gap di discretizzazione, non quello di generatore.
- Dato reale (FaMoS): nella catena preregistrata il contributo passava dalla scelta r = 0 / r = 32 (+0.038 su FaceScape s1234, zero altrove): **ARTEFATTO DELLA GRIGLIA**. Con la catena corretta (i-e2) - (i-e2 senza FaMoS) vale +0.025 [+0.015, +0.036] su FaceScape (s1234), +0.023 / +0.013 / +0.031 su FLAME, +0.030 [+0.004, +0.055] su FaceVerse (s1234), circa 0 su HIFI3D: il reale aggiunge poco, ma non zero. FaMoS come bersaglio (secondario): +0.021 (s1234), -0.018 (s2345), +0.026 (C3M) preregistrati; +0.053, +0.029, +0.043 con la catena corretta.
- C3M e123 (descrittivo): media +0.007; su HIFI3D la testa preregistrata peggiora (-0.031 [-0.053, -0.011]), con la catena corretta +0.061 [+0.026, +0.098]. B-GNM resta sopra tutte le teste LOGO su FaceScape SR (0.854).

## POST HOC, emendamento 2: griglia di lambda estesa, perdita riscalata per sorgente, testa congiunta

`PROTOCOL_D3_emendamento_2.md` (sha256 0a6c6367..., commit 36fe43e), scritto DOPO i numeri sopra e dopo il critic (RISERVE). Non cambia il verdetto preregistrato (R_LOGO: NO). (i-e2) = testa (i) con lambda in {1e-6, ..., 1} e la GT di ogni sorgente divisa per il suo alpha0 nella perdita; stessa CV, scelta, rifit, righe e repliche. Numeri in `d3_e2_delta.csv`, `d3_e2_spearman.csv`, `d3_e2_curve.csv`, `d3_e2_heads.csv`, `d3_e2_cv.csv`.

- **(i-e2), regola di R_LOGO applicata POST HOC (descrittiva): NO**
  - fact. s1234: NO (n_SR 0 su 4, FR >= -0.03 ovunque: True); facescape SR +0.034 [+0.004, +0.066], FR +0.028; hifi3d SR +0.024 [-0.016, +0.064], FR -0.001; faceverse_neutral SR +0.016 [-0.048, +0.072], FR +0.027; flame2023_s1 SR +0.028 [+0.010, +0.048], FR +0.005
  - fact. s2345: NO (n_SR 0 su 4, FR >= -0.03 ovunque: True); facescape SR +0.016 [-0.012, +0.045], FR +0.017; hifi3d SR +0.027 [-0.016, +0.078], FR +0.003; faceverse_neutral SR -0.029 [-0.102, +0.052], FR -0.024; flame2023_s1 SR +0.030 [+0.014, +0.048], FR +0.010

Note del coder sull'emendamento 2, scritte DOPO i numeri (commento, non regole):

- Il verdetto preregistrato resta NO e la catena corretta non lo ribalta: n_SR = 0 in entrambi i semi, media sui 4 bersagli +0.025 [+0.004, +0.045] (s1234) e +0.011 [-0.013, +0.037] (s2345). Lettura: **la testa lineare trasferisce poco (+0.01 / +0.03) a un generatore mai visto, sotto la soglia**; non "non trasferisce".
- **La correzione comune esiste se il generatore e' visto**: la testa congiunta (descrittiva, non LOGO) migliora tutti e 4 i bersagli insieme, media +0.120 [+0.098, +0.142] (s1234), +0.102 [+0.076, +0.130] (s2345), +0.122 (C3M); s1234 FaceScape +0.142, HIFI3D +0.153, FaceVerse +0.112, FLAME +0.075, tutti con IC sopra 0 (FaceVerse s2345 +0.059 con IC che tocca lo 0). E' l'estrapolazione a un generatore mai visto che resta piccola.
- Confronto col critic. Alla sua configurazione fissa (r = 32, lambda = 1e-5, perdita riscalata) i 16 delta LOGO e congiunti coincidono coi suoi (scarto 0, tabella sotto). La sua catena (CV ridotta: r in {8, 32, 64}, lambda in {1e-5, 1e-4, 1e-3}, 2 ripetizioni) da' medie piu' alte della nostra (+0.033 [+0.016, +0.051] e +0.022 [+0.001, +0.043] contro +0.025 e +0.011) perche' la nostra CV completa sceglie lambda = 1e-6, il bordo della griglia estesa, che sulle sorgenti vince (+0.045 / +0.064 di CV) ma estrapola peggio di 1e-5: FaceScape s1234 +0.050 a (32, 1e-5) contro +0.034 a (64, 1e-6), FaceVerse s2345 +0.008 contro -0.029. La CV sulle sorgenti premia l'adattamento ai generatori visti, non il trasferimento. In entrambe le catene n_SR = 0. Sulla testa congiunta la nostra scelta (64, 1e-6) da' piu' del critic su FaceScape, HIFI3D e FaceVerse (+0.142 / +0.153 / +0.112 contro +0.133 / +0.122 / +0.091), uguale su FLAME (+0.075 contro +0.076).
- Curva: con la catena corretta il guadagno cresce col numero di sorgenti su tutti i bersagli (verdetto "si"); una sola sorgente in media peggiora (da -0.03 a -0.08), tutte e 7 vanno da -0.03 a +0.06. Il contributo di FaMoS e' piccolo ma positivo su FaceScape (s1234), FLAME e FaceVerse (s1234), nullo su HIFI3D.
- d_P con la stessa discretizzazione ai due lati: su FaceScape tutte le righe con la stessa etichetta (0.767-0.808) stanno sopra le incrociate (0.746-0.753) e la original non e' la migliore (0.778-0.792); remesh-remesh - incrociate +0.050 / +0.062 su FaceScape, +0.02 su HIFI3D, +0.01 / +0.02 su FaceVerse, +0.03 su FLAME. Conferma il punto 4 del critic (i suoi 0.800 e 0.802 per s1234 sono i nostri).
- Bordi e riproducibilita': tutte le scelte (i-e2) hanno lambda = 1e-6 e 27 su 30 r = 64; la CV potrebbe salire ancora con meno cresta (griglia non allargata, come da protocollo). Le CV di s2345 e C3M sono state finite da due aiutanti su altri nodi: i 952 e 903 fit calcolati due volte differiscono fino a 0.008 nel punteggio (BLAS diverse fra nodi; con lambda = 1e-6 la perdita e' quasi senza cresta); la valutazione gira sullo stesso nodo del run preregistrato (K_pre = 0).

LOGO, preregistrata contro post hoc (righe incrociate; delta contro d_P dello stesso braccio):

| bersaglio | braccio | scelta e2 (r, lambda) | SR preregistrata | SR (i-e2) | (i-e2) - preregistrata | FR (i-e2) |
|---|---|---|---|---|---|---|
| facescape | fact. s1234 | 64, 1e-06 | +0.038 [+0.014, +0.065] | +0.034 [+0.004, +0.066] | -0.004 [-0.032, +0.025] | +0.028 [+0.001, +0.056] |
| facescape | fact. s2345 | 64, 1e-06 | +0.000 [+0.000, +0.000] | +0.016 [-0.012, +0.045] | +0.016 [-0.012, +0.045] | +0.017 [-0.008, +0.042] |
| facescape | C3M e123 | 64, 1e-06 | +0.013 [-0.004, +0.030] | +0.024 [-0.005, +0.053] | +0.010 [-0.017, +0.039] | +0.029 [-0.001, +0.059] |
| hifi3d | fact. s1234 | 64, 1e-06 | +0.000 [+0.000, +0.000] | +0.024 [-0.016, +0.064] | +0.024 [-0.016, +0.064] | -0.001 [-0.020, +0.017] |
| hifi3d | fact. s2345 | 64, 1e-06 | +0.000 [+0.000, +0.000] | +0.027 [-0.016, +0.078] | +0.027 [-0.016, +0.078] | +0.003 [-0.016, +0.024] |
| hifi3d | C3M e123 | 64, 1e-06 | -0.031 [-0.053, -0.011] | +0.061 [+0.026, +0.098] | +0.092 [+0.050, +0.134] | +0.018 [+0.003, +0.036] |
| faceverse_neutral | fact. s1234 | 64, 1e-06 | +0.026 [-0.016, +0.071] | +0.016 [-0.048, +0.072] | -0.010 [-0.070, +0.046] | +0.027 [-0.029, +0.079] |
| faceverse_neutral | fact. s2345 | 32, 1e-06 | -0.003 [-0.047, +0.046] | -0.029 [-0.102, +0.052] | -0.026 [-0.098, +0.045] | -0.024 [-0.092, +0.047] |
| faceverse_neutral | C3M e123 | 64, 1e-06 | +0.021 [-0.022, +0.063] | -0.014 [-0.070, +0.040] | -0.035 [-0.088, +0.016] | -0.011 [-0.061, +0.039] |
| flame2023_s1 | fact. s1234 | 32, 1e-06 | -0.001 [-0.018, +0.014] | +0.028 [+0.010, +0.048] | +0.029 [+0.013, +0.046] | +0.005 [-0.005, +0.015] |
| flame2023_s1 | fact. s2345 | 64, 1e-06 | +0.000 [+0.000, +0.000] | +0.030 [+0.014, +0.048] | +0.030 [+0.014, +0.048] | +0.010 [+0.001, +0.017] |
| flame2023_s1 | C3M e123 | 64, 1e-06 | +0.023 [+0.012, +0.035] | +0.025 [+0.010, +0.042] | +0.002 [-0.010, +0.015] | +0.004 [-0.004, +0.013] |
| faceverse | fact. s1234 | 64, 1e-06 | +0.001 [-0.037, +0.041] | +0.002 [-0.051, +0.054] | +0.001 [-0.050, +0.046] | +0.008 [-0.041, +0.053] |
| faceverse | fact. s2345 | 32, 1e-06 | -0.025 [-0.062, +0.014] | -0.051 [-0.118, +0.020] | -0.026 [-0.089, +0.037] | -0.042 [-0.098, +0.017] |
| faceverse | C3M e123 | 64, 1e-06 | -0.004 [-0.038, +0.031] | -0.037 [-0.082, +0.009] | -0.033 [-0.077, +0.007] | -0.027 [-0.064, +0.013] |
| famos | fact. s1234 | 64, 1e-06 | +0.021 [+0.004, +0.040] | +0.053 [+0.027, +0.084] | +0.031 [+0.010, +0.060] | +0.027 [+0.011, +0.047] |
| famos | fact. s2345 | 64, 1e-06 | -0.018 [-0.037, -0.000] | +0.029 [-0.004, +0.069] | +0.047 [+0.019, +0.082] | +0.018 [+0.000, +0.038] |
| famos | C3M e123 | 64, 1e-06 | +0.026 [+0.009, +0.044] | +0.043 [+0.015, +0.076] | +0.017 [-0.003, +0.041] | +0.026 [+0.008, +0.048] |

Medie sui 4 bersagli primari (replica per replica; testa congiunta su FLAME sulle righe 100-199):

| braccio | SR preregistrata | SR (i-e2) | FR (i-e2) | SR congiunta | FR congiunta |
|---|---|---|---|---|---|
| fact. s1234 | +0.016 [+0.003, +0.029] | +0.025 [+0.004, +0.045] | +0.015 [-0.001, +0.031] | +0.120 [+0.098, +0.142] | +0.070 [+0.052, +0.087] |
| fact. s2345 | -0.001 [-0.012, +0.011] | +0.011 [-0.013, +0.037] | +0.001 [-0.018, +0.021] | +0.102 [+0.076, +0.130] | +0.052 [+0.032, +0.073] |
| C3M e123 | +0.007 [-0.006, +0.020] | +0.024 [+0.005, +0.042] | +0.010 [-0.004, +0.025] | +0.122 [+0.099, +0.146] | +0.066 [+0.049, +0.084] |

Testa congiunta (DESCRITTIVA, non LOGO: tutte le 8 sorgenti, compresi i pool non valutati dei bersagli; FLAME soggetti 0-99 in training, righe 100-199 in test con bootstrap sui 100 di test):

| bersaglio | braccio | scelta (r, lambda) | SR [IC] | FR [IC] | congiunta - (i-e2) SR [IC] |
|---|---|---|---|---|---|
| facescape | fact. s1234 | 64, 1e-06 | +0.142 [+0.106, +0.185] | +0.117 [+0.075, +0.158] | +0.108 [+0.084, +0.137] |
| facescape | fact. s2345 | 64, 1e-06 | +0.120 [+0.084, +0.161] | +0.090 [+0.057, +0.126] | +0.104 [+0.074, +0.136] |
| facescape | C3M e123 | 64, 1e-06 | +0.140 [+0.108, +0.176] | +0.123 [+0.088, +0.157] | +0.116 [+0.088, +0.148] |
| hifi3d | fact. s1234 | 64, 1e-06 | +0.153 [+0.106, +0.199] | +0.047 [+0.022, +0.076] | +0.130 [+0.084, +0.186] |
| hifi3d | fact. s2345 | 64, 1e-06 | +0.155 [+0.103, +0.210] | +0.052 [+0.025, +0.084] | +0.128 [+0.080, +0.188] |
| hifi3d | C3M e123 | 64, 1e-06 | +0.199 [+0.137, +0.269] | +0.070 [+0.041, +0.106] | +0.137 [+0.094, +0.191] |
| faceverse_neutral | fact. s1234 | 64, 1e-06 | +0.112 [+0.056, +0.166] | +0.096 [+0.044, +0.143] | +0.096 [+0.051, +0.142] |
| faceverse_neutral | fact. s2345 | 64, 1e-06 | +0.059 [-0.010, +0.127] | +0.041 [-0.021, +0.100] | +0.088 [+0.038, +0.135] |
| faceverse_neutral | C3M e123 | 64, 1e-06 | +0.064 [+0.010, +0.116] | +0.045 [-0.002, +0.092] | +0.078 [+0.035, +0.120] |
| flame2023_s1 | fact. s1234 | 64, 1e-06 | +0.075 [+0.046, +0.106] | +0.021 [+0.005, +0.039] | +0.057 [+0.038, +0.076] |
| flame2023_s1 | fact. s2345 | 64, 1e-06 | +0.072 [+0.048, +0.096] | +0.025 [+0.013, +0.036] | +0.053 [+0.036, +0.070] |
| flame2023_s1 | C3M e123 | 64, 1e-06 | +0.087 [+0.062, +0.113] | +0.026 [+0.014, +0.039] | +0.064 [+0.044, +0.083] |
| faceverse | fact. s1234 | 64, 1e-06 | +0.048 [-0.011, +0.096] | +0.042 [-0.007, +0.085] | +0.046 [+0.006, +0.084] |
| faceverse | fact. s2345 | 64, 1e-06 | -0.008 [-0.068, +0.051] | -0.009 [-0.060, +0.039] | +0.043 [+0.008, +0.078] |
| faceverse | C3M e123 | 64, 1e-06 | +0.006 [-0.040, +0.050] | +0.003 [-0.035, +0.042] | +0.043 [+0.010, +0.077] |

Contributo di FaMoS con la catena e2: (i-e2) - (i-e2 senza FaMoS):

| bersaglio | braccio | SR | FR |
|---|---|---|---|
| facescape | fact. s1234 | +0.025 [+0.015, +0.036] | +0.014 [+0.004, +0.024] |
| facescape | fact. s2345 | -0.001 [-0.012, +0.011] | -0.006 [-0.017, +0.004] |
| facescape | C3M e123 | +0.011 [-0.000, +0.022] | +0.010 [-0.000, +0.021] |
| hifi3d | fact. s1234 | +0.006 [-0.005, +0.016] | +0.005 [+0.001, +0.010] |
| hifi3d | fact. s2345 | +0.001 [-0.011, +0.015] | +0.001 [-0.003, +0.007] |
| hifi3d | C3M e123 | -0.005 [-0.018, +0.009] | -0.001 [-0.006, +0.004] |
| faceverse_neutral | fact. s1234 | +0.030 [+0.004, +0.055] | +0.023 [-0.000, +0.044] |
| faceverse_neutral | fact. s2345 | +0.020 [-0.010, +0.053] | +0.013 [-0.014, +0.038] |
| faceverse_neutral | C3M e123 | +0.020 [-0.006, +0.044] | +0.021 [-0.001, +0.045] |
| flame2023_s1 | fact. s1234 | +0.023 [+0.015, +0.031] | +0.007 [+0.003, +0.012] |
| flame2023_s1 | fact. s2345 | +0.013 [+0.005, +0.021] | +0.002 [-0.001, +0.006] |
| flame2023_s1 | C3M e123 | +0.031 [+0.020, +0.042] | +0.009 [+0.003, +0.014] |

Curva sul numero di sorgenti con (i-e2) (iperparametri di "tutte tranne T"): cresce? **si**

| bersaglio | braccio | k = 1 | k = 2 | k = 4 | k = 7 |
|---|---|---|---|---|---|
| facescape | fact. s1234 | -0.072 [-0.183, +0.024] | -0.035 [-0.150, +0.066] | +0.003 [-0.113, +0.057] | +0.034 |
| facescape | fact. s2345 | -0.057 [-0.147, +0.040] | -0.018 [-0.106, +0.051] | +0.010 [-0.055, +0.053] | +0.016 |
| facescape | C3M e123 | -0.048 [-0.108, +0.039] | -0.016 [-0.080, +0.073] | +0.011 [-0.069, +0.052] | +0.024 |
| hifi3d | fact. s1234 | -0.076 [-0.101, -0.038] | -0.043 [-0.071, -0.006] | -0.005 [-0.044, +0.018] | +0.024 |
| hifi3d | fact. s2345 | -0.066 [-0.124, -0.031] | -0.036 [-0.103, +0.013] | +0.002 [-0.059, +0.034] | +0.027 |
| hifi3d | C3M e123 | -0.033 [-0.096, +0.013] | -0.003 [-0.088, +0.058] | +0.029 [-0.016, +0.067] | +0.061 |
| faceverse | fact. s1234 | -0.025 [-0.138, +0.075] | -0.012 [-0.120, +0.080] | -0.003 [-0.070, +0.054] | +0.016 |
| faceverse | fact. s2345 | -0.069 [-0.171, -0.002] | -0.049 [-0.132, +0.038] | -0.039 [-0.079, +0.013] | -0.029 |
| faceverse | C3M e123 | -0.070 [-0.102, -0.049] | -0.055 [-0.103, +0.021] | -0.039 [-0.071, +0.000] | -0.014 |
| flame2023_s1 | fact. s1234 | -0.047 [-0.144, +0.026] | -0.013 [-0.065, +0.049] | +0.017 [-0.039, +0.056] | +0.028 |
| flame2023_s1 | fact. s2345 | -0.077 [-0.145, -0.008] | -0.032 [-0.119, +0.016] | +0.012 [-0.046, +0.040] | +0.030 |
| flame2023_s1 | C3M e123 | -0.058 [-0.114, +0.008] | -0.020 [-0.089, +0.035] | +0.008 [-0.055, +0.042] | +0.025 |

d_P con la STESSA discretizzazione ai due lati della riga (una riga per coppia di soggetti di test), SR:

| bersaglio | braccio | incrociate | original-original | remesh-remesh | down8k-down8k | noisy-noisy | up60k-up60k | mediate | remesh-remesh - incrociate [IC] |
|---|---|---|---|---|---|---|---|---|---|
| facescape | fact. s1234 | 0.747 | 0.778 | 0.800 | 0.779 | 0.791 | 0.802 | 0.802 | +0.053 [+0.041, +0.068] |
| facescape | fact. s2345 | 0.753 | 0.778 | 0.803 | 0.781 | 0.767 | 0.789 | 0.792 | +0.050 [+0.035, +0.067] |
| facescape | C3M e123 | 0.746 | 0.792 | 0.808 | 0.779 | 0.770 | 0.800 | 0.803 | +0.062 [+0.047, +0.081] |
| hifi3d | fact. s1234 | 0.622 | 0.642 | 0.642 | 0.635 | 0.619 | 0.636 | 0.639 | +0.020 [+0.013, +0.028] |
| hifi3d | fact. s2345 | 0.613 | 0.624 | 0.632 | 0.624 | 0.597 | 0.624 | 0.623 | +0.019 [+0.014, +0.026] |
| hifi3d | C3M e123 | 0.595 | 0.619 | 0.617 | 0.616 | 0.613 | 0.605 | 0.620 | +0.022 [+0.011, +0.035] |
| faceverse_neutral | fact. s1234 | 0.333 | 0.345 | 0.345 | 0.339 | 0.365 | 0.348 | 0.351 | +0.012 [+0.001, +0.022] |
| faceverse_neutral | fact. s2345 | 0.372 | 0.377 | 0.383 | 0.375 | 0.389 | 0.379 | 0.383 | +0.011 [+0.003, +0.020] |
| faceverse_neutral | C3M e123 | 0.382 | 0.400 | 0.398 | 0.395 | 0.432 | 0.427 | 0.417 | +0.017 [-0.000, +0.032] |
| flame2023_s1 | fact. s1234 | 0.806 | 0.810 | 0.832 | 0.819 | 0.801 | 0.827 | 0.825 | +0.026 [+0.019, +0.033] |
| flame2023_s1 | fact. s2345 | 0.806 | 0.809 | 0.836 | 0.814 | 0.794 | 0.814 | 0.818 | +0.030 [+0.023, +0.038] |
| flame2023_s1 | C3M e123 | 0.819 | 0.825 | 0.845 | 0.828 | 0.820 | 0.833 | 0.838 | +0.027 [+0.020, +0.033] |

Confronto col critic alla sua configurazione fissa (r = 32, lambda = 1e-5, perdita riscalata), delta SR puntuale:

| cella | critic | nostro | scarto |
|---|---|---|---|
| factorized_s1234 logo facescape | +0.0501 | +0.0501 | 0.0000 |
| factorized_s1234 logo hifi3d | +0.0217 | +0.0217 | 0.0000 |
| factorized_s1234 logo faceverse | +0.0268 | +0.0268 | 0.0000 |
| factorized_s1234 logo flame2023_s1 | +0.0332 | +0.0332 | 0.0000 |
| factorized_s2345 logo facescape | +0.0312 | +0.0312 | 0.0000 |
| factorized_s2345 logo hifi3d | +0.0262 | +0.0262 | 0.0000 |
| factorized_s2345 logo faceverse | +0.0080 | +0.0080 | 0.0000 |
| factorized_s2345 logo flame2023_s1 | +0.0175 | +0.0175 | 0.0000 |
| factorized_s1234 joint facescape | +0.1326 | +0.1326 | 0.0000 |
| factorized_s1234 joint hifi3d | +0.1224 | +0.1224 | 0.0000 |
| factorized_s1234 joint faceverse | +0.0898 | +0.0898 | 0.0000 |
| factorized_s1234 joint flame2023_s1 | +0.0763 | +0.0763 | 0.0000 |
| factorized_s2345 joint facescape | +0.1043 | +0.1043 | 0.0000 |
| factorized_s2345 joint hifi3d | +0.1236 | +0.1236 | 0.0000 |
| factorized_s2345 joint faceverse | +0.0525 | +0.0525 | 0.0000 |
| factorized_s2345 joint flame2023_s1 | +0.0655 | +0.0655 | 0.0000 |

Controlli: K1 (riferimenti = pubblicati) 3.3e-16, **passa**; K_pre (teste preregistrate rifittate = delta preregistrati, 54 celle, punto e IC) 0.0e+00, **passa**; K_critic max scarto 0.0000, **passa**; K5 0.0e+00; 19350 fit di CV (43 configurazioni); tempo della valutazione 1198 s.

## Delta contro d_P per bersaglio (righe incrociate)

SR: d_h - d_P; FR: d_F,h - d_F calibrata; FR senza taglia: d_h - d_P letti con la GT FR. "fold" = minimo e massimo del delta SR puntuale delle 15 teste di fold (4/5 dei soggetti delle sorgenti).

### FaceScape (3DMM bilineare) (88725 righe, 100 soggetti, seme 796786)

| testa | braccio | rho SR | delta SR [IC] | fold SR | delta FR [IC] | delta FR senza taglia [IC] |
|---|---|---|---|---|---|---|
| (i) appresa, tutte tranne T | fact. s1234 | 0.784 | +0.038 [+0.014, +0.065] | [+0.018, +0.044] | +0.030 [+0.003, +0.055] | +0.027 [-0.002, +0.058] |
| (i) appresa, tutte tranne T | fact. s2345 | 0.753 | +0.000 [+0.000, +0.000] | [-0.000, -0.000] | +0.007 [+0.005, +0.009] | +0.000 [+0.000, +0.000] |
| (i) appresa, tutte tranne T | C3M e123 | 0.759 | +0.013 [-0.004, +0.030] | [+0.001, +0.027] | +0.011 [-0.006, +0.029] | +0.003 [-0.015, +0.021] |
| (i) senza FaMoS | fact. s1234 | 0.747 | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | +0.006 [+0.004, +0.009] | +0.000 [+0.000, +0.000] |
| (i) senza FaMoS | fact. s2345 | 0.753 | +0.000 [+0.000, +0.000] | [-0.000, -0.000] | +0.007 [+0.005, +0.010] | +0.000 [+0.000, +0.000] |
| (i) senza FaMoS | C3M e123 | 0.749 | +0.003 [-0.015, +0.020] | [-0.005, +0.014] | +0.001 [-0.017, +0.019] | -0.008 [-0.027, +0.010] |
| (ii) intra-soggetto, tutte tranne T | fact. s1234 | 0.746 | -0.000 [-0.041, +0.042] | - | +0.010 [-0.028, +0.045] | -0.007 [-0.050, +0.035] |
| (ii) intra-soggetto, tutte tranne T | fact. s2345 | 0.765 | +0.012 [-0.022, +0.047] | - | +0.014 [-0.017, +0.046] | -0.004 [-0.042, +0.035] |
| (ii) intra-soggetto, tutte tranne T | C3M e123 | 0.746 | -0.000 [-0.038, +0.034] | - | +0.006 [-0.026, +0.039] | -0.005 [-0.042, +0.030] |
| (ii) senza FaMoS | fact. s1234 | 0.740 | -0.006 [-0.047, +0.038] | - | +0.004 [-0.035, +0.039] | -0.011 [-0.055, +0.030] |
| (ii) senza FaMoS | fact. s2345 | 0.760 | +0.007 [-0.028, +0.041] | - | +0.009 [-0.021, +0.041] | -0.008 [-0.047, +0.031] |
| (ii) senza FaMoS | C3M e123 | 0.738 | -0.008 [-0.044, +0.024] | - | +0.001 [-0.030, +0.033] | -0.010 [-0.045, +0.024] |
| CORAL (esplorativa) | fact. s1234 | 0.575 | -0.171 [-0.226, -0.119] | - | -0.117 [-0.163, -0.068] | -0.153 [-0.201, -0.104] |
| CORAL (esplorativa) | fact. s2345 | 0.559 | -0.194 [-0.248, -0.142] | - | -0.123 [-0.176, -0.071] | -0.162 [-0.214, -0.111] |
| CORAL (esplorativa) | C3M e123 | 0.534 | -0.212 [-0.269, -0.155] | - | -0.164 [-0.222, -0.101] | -0.195 [-0.250, -0.134] |

| riferimento | SR | FR | FR senza taglia |
|---|---|---|---|
| fact. s1234: d_P / d_F cal. | 0.747 [0.677, 0.804] | 0.661 [0.586, 0.727] | 0.677 [0.596, 0.750] |
| fact. s2345: d_P / d_F cal. | 0.753 [0.687, 0.813] | 0.669 [0.597, 0.733] | 0.684 [0.601, 0.760] |
| C3M e123: d_P / d_F cal. | 0.746 [0.686, 0.799] | 0.665 [0.596, 0.727] | 0.675 [0.603, 0.742] |
| B-GNM (vB) | 0.854 [0.803, 0.894] | 0.678 [0.615, 0.740] | - |
| B-FLAME (vB) | 0.820 [0.757, 0.870] | 0.630 [0.555, 0.695] | - |

### HIFI3D (AI-NEXT) (98224 righe, 100 soggetti, seme 990708)

| testa | braccio | rho SR | delta SR [IC] | fold SR | delta FR [IC] | delta FR senza taglia [IC] |
|---|---|---|---|---|---|---|
| (i) appresa, tutte tranne T | fact. s1234 | 0.622 | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| (i) appresa, tutte tranne T | fact. s2345 | 0.613 | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | +0.001 [+0.000, +0.001] | +0.000 [+0.000, +0.000] |
| (i) appresa, tutte tranne T | C3M e123 | 0.564 | -0.031 [-0.053, -0.011] | [-0.040, -0.018] | -0.005 [-0.014, +0.004] | -0.014 [-0.036, +0.011] |
| (i) senza FaMoS | fact. s1234 | 0.622 | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| (i) senza FaMoS | fact. s2345 | 0.613 | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| (i) senza FaMoS | C3M e123 | 0.560 | -0.035 [-0.059, -0.015] | [-0.046, -0.021] | -0.008 [-0.017, +0.001] | -0.014 [-0.037, +0.011] |
| (ii) intra-soggetto, tutte tranne T | fact. s1234 | 0.553 | -0.069 [-0.126, -0.011] | - | -0.031 [-0.055, -0.005] | -0.035 [-0.108, +0.036] |
| (ii) intra-soggetto, tutte tranne T | fact. s2345 | 0.554 | -0.059 [-0.111, -0.012] | - | -0.026 [-0.049, -0.006] | -0.040 [-0.096, +0.020] |
| (ii) intra-soggetto, tutte tranne T | C3M e123 | 0.549 | -0.046 [-0.100, +0.009] | - | -0.018 [-0.037, +0.004] | -0.024 [-0.080, +0.033] |
| (ii) senza FaMoS | fact. s1234 | 0.548 | -0.074 [-0.134, -0.016] | - | -0.035 [-0.060, -0.008] | -0.038 [-0.111, +0.032] |
| (ii) senza FaMoS | fact. s2345 | 0.553 | -0.060 [-0.112, -0.013] | - | -0.028 [-0.051, -0.009] | -0.040 [-0.097, +0.020] |
| (ii) senza FaMoS | C3M e123 | 0.547 | -0.048 [-0.103, +0.008] | - | -0.019 [-0.039, +0.002] | -0.027 [-0.083, +0.031] |
| CORAL (esplorativa) | fact. s1234 | 0.501 | -0.120 [-0.169, -0.068] | - | -0.053 [-0.087, -0.022] | -0.080 [-0.132, -0.026] |
| CORAL (esplorativa) | fact. s2345 | 0.484 | -0.128 [-0.181, -0.073] | - | -0.056 [-0.097, -0.019] | -0.067 [-0.130, -0.008] |
| CORAL (esplorativa) | C3M e123 | 0.414 | -0.181 [-0.229, -0.130] | - | -0.062 [-0.092, -0.032] | -0.062 [-0.125, -0.002] |

| riferimento | SR | FR | FR senza taglia |
|---|---|---|---|
| fact. s1234: d_P / d_F cal. | 0.622 [0.550, 0.685] | 0.749 [0.673, 0.806] | 0.426 [0.333, 0.510] |
| fact. s2345: d_P / d_F cal. | 0.613 [0.533, 0.688] | 0.731 [0.657, 0.792] | 0.417 [0.315, 0.512] |
| C3M e123: d_P / d_F cal. | 0.595 [0.511, 0.671] | 0.748 [0.672, 0.808] | 0.400 [0.304, 0.492] |
| B-GNM (vB) | 0.607 [0.532, 0.679] | 0.664 [0.570, 0.738] | - |
| B-FLAME (vB) | 0.597 [0.518, 0.673] | 0.686 [0.597, 0.758] | - |

### FaceVerse neutra (99000 righe, 100 soggetti, seme 271049)

| testa | braccio | rho SR | delta SR [IC] | fold SR | delta FR [IC] | delta FR senza taglia [IC] |
|---|---|---|---|---|---|---|
| (i) appresa, tutte tranne T | fact. s1234 | 0.359 | +0.026 [-0.016, +0.071] | [+0.014, +0.037] | +0.019 [-0.019, +0.057] | +0.021 [-0.019, +0.065] |
| (i) appresa, tutte tranne T | fact. s2345 | 0.369 | -0.003 [-0.047, +0.046] | [-0.013, +0.007] | -0.000 [-0.037, +0.036] | +0.001 [-0.041, +0.047] |
| (i) appresa, tutte tranne T | C3M e123 | 0.403 | +0.021 [-0.022, +0.063] | [-0.005, +0.032] | +0.023 [-0.015, +0.060] | +0.018 [-0.026, +0.057] |
| (i) senza FaMoS | fact. s1234 | 0.358 | +0.024 [-0.016, +0.067] | [+0.009, +0.035] | +0.016 [-0.019, +0.054] | +0.019 [-0.018, +0.061] |
| (i) senza FaMoS | fact. s2345 | 0.380 | +0.008 [-0.040, +0.059] | [-0.008, +0.010] | +0.009 [-0.032, +0.048] | +0.012 [-0.034, +0.057] |
| (i) senza FaMoS | C3M e123 | 0.404 | +0.022 [-0.023, +0.064] | [+0.003, +0.029] | +0.023 [-0.016, +0.060] | +0.020 [-0.027, +0.064] |
| (ii) intra-soggetto, tutte tranne T | fact. s1234 | 0.389 | +0.056 [-0.016, +0.121] | - | +0.035 [-0.025, +0.091] | +0.032 [-0.035, +0.093] |
| (ii) intra-soggetto, tutte tranne T | fact. s2345 | 0.396 | +0.025 [-0.033, +0.084] | - | +0.009 [-0.039, +0.060] | +0.006 [-0.053, +0.065] |
| (ii) intra-soggetto, tutte tranne T | C3M e123 | 0.374 | -0.007 [-0.066, +0.054] | - | -0.015 [-0.064, +0.036] | -0.024 [-0.081, +0.033] |
| (ii) senza FaMoS | fact. s1234 | 0.381 | +0.047 [-0.026, +0.113] | - | +0.027 [-0.034, +0.083] | +0.023 [-0.042, +0.084] |
| (ii) senza FaMoS | fact. s2345 | 0.384 | +0.012 [-0.045, +0.070] | - | -0.002 [-0.050, +0.050] | -0.006 [-0.065, +0.051] |
| (ii) senza FaMoS | C3M e123 | 0.369 | -0.012 [-0.073, +0.049] | - | -0.019 [-0.069, +0.033] | -0.029 [-0.086, +0.029] |
| CORAL (esplorativa) | fact. s1234 | 0.345 | +0.012 [-0.061, +0.083] | - | +0.016 [-0.047, +0.077] | +0.020 [-0.050, +0.088] |
| CORAL (esplorativa) | fact. s2345 | 0.348 | -0.024 [-0.100, +0.046] | - | -0.017 [-0.085, +0.042] | -0.015 [-0.092, +0.056] |
| CORAL (esplorativa) | C3M e123 | 0.365 | -0.016 [-0.084, +0.054] | - | +0.006 [-0.055, +0.069] | -0.004 [-0.073, +0.068] |

| riferimento | SR | FR | FR senza taglia |
|---|---|---|---|
| fact. s1234: d_P / d_F cal. | 0.333 [0.251, 0.405] | 0.341 [0.263, 0.416] | 0.321 [0.237, 0.391] |
| fact. s2345: d_P / d_F cal. | 0.372 [0.291, 0.444] | 0.370 [0.298, 0.440] | 0.361 [0.276, 0.439] |
| C3M e123: d_P / d_F cal. | 0.382 [0.293, 0.457] | 0.347 [0.268, 0.419] | 0.363 [0.276, 0.440] |
| B-GNM (vB) | 0.399 [0.315, 0.473] | 0.321 [0.225, 0.400] | - |
| B-FLAME (vB) | 0.382 [0.297, 0.459] | 0.315 [0.223, 0.393] | - |

### FLAME 2023 (D1) (398000 righe, 200 soggetti, seme D1: SeedSequence([20261113, 3]))

| testa | braccio | rho SR | delta SR [IC] | fold SR | delta FR [IC] | delta FR senza taglia [IC] |
|---|---|---|---|---|---|---|
| (i) appresa, tutte tranne T | fact. s1234 | 0.805 | -0.001 [-0.018, +0.014] | [-0.011, +0.003] | -0.005 [-0.013, +0.004] | -0.026 [-0.049, -0.006] |
| (i) appresa, tutte tranne T | fact. s2345 | 0.806 | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | +0.002 [+0.001, +0.004] | +0.000 [+0.000, +0.000] |
| (i) appresa, tutte tranne T | C3M e123 | 0.842 | +0.023 [+0.012, +0.035] | [+0.014, +0.029] | +0.009 [+0.003, +0.015] | +0.005 [-0.010, +0.019] |
| (i) senza FaMoS | fact. s1234 | 0.793 | -0.013 [-0.030, +0.003] | [-0.026, -0.005] | -0.006 [-0.015, +0.003] | -0.033 [-0.054, -0.014] |
| (i) senza FaMoS | fact. s2345 | 0.806 | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | +0.002 [+0.001, +0.004] | +0.000 [+0.000, +0.000] |
| (i) senza FaMoS | C3M e123 | 0.820 | +0.002 [-0.012, +0.014] | [-0.007, +0.010] | +0.002 [-0.004, +0.009] | -0.008 [-0.023, +0.008] |
| (ii) intra-soggetto, tutte tranne T | fact. s1234 | 0.785 | -0.021 [-0.050, +0.006] | - | -0.009 [-0.023, +0.003] | -0.052 [-0.082, -0.022] |
| (ii) intra-soggetto, tutte tranne T | fact. s2345 | 0.793 | -0.013 [-0.038, +0.012] | - | +0.000 [-0.009, +0.009] | -0.031 [-0.059, -0.003] |
| (ii) intra-soggetto, tutte tranne T | C3M e123 | 0.812 | -0.007 [-0.025, +0.011] | - | -0.004 [-0.014, +0.005] | -0.040 [-0.065, -0.017] |
| (ii) senza FaMoS | fact. s1234 | 0.777 | -0.029 [-0.056, -0.003] | - | -0.013 [-0.028, -0.001] | -0.049 [-0.079, -0.021] |
| (ii) senza FaMoS | fact. s2345 | 0.791 | -0.015 [-0.040, +0.009] | - | -0.001 [-0.011, +0.007] | -0.030 [-0.058, -0.003] |
| (ii) senza FaMoS | C3M e123 | 0.810 | -0.008 [-0.026, +0.009] | - | -0.004 [-0.014, +0.005] | -0.036 [-0.059, -0.015] |

| riferimento | SR | FR | FR senza taglia |
|---|---|---|---|
| fact. s1234: d_P / d_F cal. | 0.806 [0.778, 0.832] | 0.809 [0.777, 0.838] | 0.605 [0.544, 0.660] |
| fact. s2345: d_P / d_F cal. | 0.806 [0.776, 0.832] | 0.804 [0.770, 0.836] | 0.601 [0.543, 0.657] |
| C3M e123: d_P / d_F cal. | 0.819 [0.794, 0.844] | 0.832 [0.805, 0.859] | 0.598 [0.538, 0.652] |

### FaceVerse con espressioni (secondaria) (99000 righe, 100 soggetti, seme 271049)

| testa | braccio | rho SR | delta SR [IC] | fold SR | delta FR [IC] | delta FR senza taglia [IC] |
|---|---|---|---|---|---|---|
| (i) appresa, tutte tranne T | fact. s1234 | 0.287 | +0.001 [-0.037, +0.041] | [-0.009, +0.009] | +0.003 [-0.029, +0.038] | +0.004 [-0.031, +0.042] |
| (i) appresa, tutte tranne T | fact. s2345 | 0.288 | -0.025 [-0.062, +0.014] | [-0.037, -0.015] | -0.017 [-0.049, +0.014] | -0.020 [-0.059, +0.018] |
| (i) appresa, tutte tranne T | C3M e123 | 0.309 | -0.004 [-0.038, +0.031] | [-0.025, +0.008] | +0.003 [-0.027, +0.033] | -0.004 [-0.038, +0.031] |
| (i) senza FaMoS | fact. s1234 | 0.285 | -0.001 [-0.038, +0.041] | [-0.009, +0.006] | +0.001 [-0.031, +0.035] | +0.002 [-0.034, +0.041] |
| (i) senza FaMoS | fact. s2345 | 0.294 | -0.019 [-0.061, +0.023] | [-0.031, -0.012] | -0.013 [-0.047, +0.019] | -0.015 [-0.056, +0.024] |
| (i) senza FaMoS | C3M e123 | 0.315 | +0.002 [-0.034, +0.039] | [-0.013, +0.010] | +0.006 [-0.025, +0.038] | +0.001 [-0.035, +0.038] |
| (ii) intra-soggetto, tutte tranne T | fact. s1234 | 0.327 | +0.041 [-0.025, +0.101] | - | +0.021 [-0.031, +0.071] | +0.019 [-0.037, +0.074] |
| (ii) intra-soggetto, tutte tranne T | fact. s2345 | 0.319 | +0.006 [-0.042, +0.055] | - | +0.000 [-0.041, +0.041] | -0.006 [-0.055, +0.042] |
| (ii) intra-soggetto, tutte tranne T | C3M e123 | 0.328 | +0.015 [-0.033, +0.063] | - | +0.005 [-0.035, +0.046] | +0.000 [-0.045, +0.047] |
| (ii) senza FaMoS | fact. s1234 | 0.319 | +0.033 [-0.035, +0.094] | - | +0.014 [-0.040, +0.064] | +0.012 [-0.046, +0.067] |
| (ii) senza FaMoS | fact. s2345 | 0.310 | -0.003 [-0.050, +0.046] | - | -0.008 [-0.047, +0.033] | -0.015 [-0.063, +0.033] |
| (ii) senza FaMoS | C3M e123 | 0.325 | +0.012 [-0.036, +0.060] | - | +0.001 [-0.039, +0.043] | -0.003 [-0.050, +0.045] |
| CORAL (esplorativa) | fact. s1234 | 0.267 | -0.019 [-0.076, +0.040] | - | -0.013 [-0.064, +0.039] | -0.013 [-0.071, +0.047] |
| CORAL (esplorativa) | fact. s2345 | 0.262 | -0.051 [-0.113, +0.009] | - | -0.045 [-0.102, +0.010] | -0.051 [-0.114, +0.010] |
| CORAL (esplorativa) | C3M e123 | 0.276 | -0.037 [-0.096, +0.022] | - | -0.017 [-0.071, +0.038] | -0.033 [-0.092, +0.027] |

| riferimento | SR | FR | FR senza taglia |
|---|---|---|---|
| fact. s1234: d_P / d_F cal. | 0.286 [0.215, 0.352] | 0.303 [0.229, 0.374] | 0.283 [0.210, 0.346] |
| fact. s2345: d_P / d_F cal. | 0.313 [0.237, 0.377] | 0.318 [0.249, 0.379] | 0.308 [0.234, 0.373] |
| C3M e123: d_P / d_F cal. | 0.313 [0.236, 0.386] | 0.289 [0.220, 0.353] | 0.303 [0.228, 0.376] |
| B-GNM (vB) | 0.327 [0.239, 0.402] | 0.292 [0.205, 0.363] | - |
| B-FLAME (vB) | 0.293 [0.209, 0.368] | 0.264 [0.183, 0.336] | - |

### FaMoS TRAIN, reale (secondaria) (63200 righe, 80 soggetti, seme SeedSequence([20261123, 0]))

| testa | braccio | rho SR | delta SR [IC] | fold SR | delta FR [IC] | delta FR senza taglia [IC] |
|---|---|---|---|---|---|---|
| (i) appresa, tutte tranne T | fact. s1234 | 0.860 | +0.021 [+0.004, +0.040] | [+0.011, +0.028] | +0.002 [-0.009, +0.013] | -0.016 [-0.045, +0.010] |
| (i) appresa, tutte tranne T | fact. s2345 | 0.810 | -0.018 [-0.037, -0.000] | [-0.024, -0.012] | -0.005 [-0.016, +0.006] | -0.037 [-0.068, -0.009] |
| (i) appresa, tutte tranne T | C3M e123 | 0.864 | +0.026 [+0.009, +0.044] | [+0.017, +0.026] | +0.008 [-0.003, +0.021] | +0.002 [-0.023, +0.025] |
| (ii) intra-soggetto, tutte tranne T | fact. s1234 | 0.809 | -0.030 [-0.079, +0.014] | - | -0.023 [-0.051, +0.003] | -0.030 [-0.075, +0.017] |
| (ii) intra-soggetto, tutte tranne T | fact. s2345 | 0.805 | -0.023 [-0.063, +0.019] | - | -0.009 [-0.031, +0.011] | -0.010 [-0.059, +0.036] |
| (ii) intra-soggetto, tutte tranne T | C3M e123 | 0.818 | -0.020 [-0.060, +0.020] | - | -0.004 [-0.026, +0.021] | -0.021 [-0.066, +0.026] |

| riferimento | SR | FR | FR senza taglia |
|---|---|---|---|
| fact. s1234: d_P / d_F cal. | 0.839 [0.785, 0.881] | 0.824 [0.775, 0.866] | 0.646 [0.556, 0.731] |
| fact. s2345: d_P / d_F cal. | 0.828 [0.757, 0.880] | 0.815 [0.761, 0.860] | 0.632 [0.535, 0.725] |
| C3M e123: d_P / d_F cal. | 0.838 [0.787, 0.877] | 0.836 [0.785, 0.881] | 0.630 [0.546, 0.709] |

## Contributo del dato reale: (i) - (i) senza FaMoS

Catena preregistrata: ARTEFATTO DELLA GRIGLIA di lambda (vedi le note e la sezione POST HOC, dove si ricalcola con la catena corretta).

| bersaglio | braccio | SR | FR |
|---|---|---|---|
| facescape | fact. s1234 | +0.038 [+0.014, +0.065] | +0.023 [-0.002, +0.049] |
| facescape | fact. s2345 | +0.000 [+0.000, +0.000] | -0.000 [-0.000, -0.000] |
| facescape | C3M e123 | +0.011 [+0.005, +0.017] | +0.010 [+0.003, +0.017] |
| hifi3d | fact. s1234 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| hifi3d | fact. s2345 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.001] |
| hifi3d | C3M e123 | +0.004 [-0.001, +0.010] | +0.003 [+0.000, +0.006] |
| faceverse_neutral | fact. s1234 | +0.002 [-0.013, +0.016] | +0.003 [-0.011, +0.016] |
| faceverse_neutral | fact. s2345 | -0.010 [-0.026, +0.005] | -0.009 [-0.022, +0.005] |
| faceverse_neutral | C3M e123 | -0.001 [-0.013, +0.011] | +0.000 [-0.010, +0.012] |
| flame2023_s1 | fact. s1234 | +0.012 [+0.007, +0.017] | +0.002 [-0.001, +0.004] |
| flame2023_s1 | fact. s2345 | +0.000 [+0.000, +0.000] | -0.000 [-0.000, -0.000] |
| flame2023_s1 | C3M e123 | +0.022 [+0.016, +0.028] | +0.006 [+0.004, +0.010] |

## Invarianza contro pesatura (SR)

Righe incrociate (topologie diverse), original-original e mediate sulle 5 etichette: rho di d_P e delta delle teste. Lettura corretta nelle note: conta avere la stessa discretizzazione ai due lati (tabella delle righe con la stessa etichetta nella sezione POST HOC), non la original ne' la riduzione del rumore.

| bersaglio | braccio | d_P incr. / orig. / mediate | (i) delta incr. / orig. / mediate | (ii) delta incr. / orig. / mediate |
|---|---|---|---|---|
| facescape | fact. s1234 | 0.747 / 0.778 / 0.802 | +0.038 / +0.027 / +0.019 | -0.000 / -0.026 / -0.046 |
| facescape | fact. s2345 | 0.753 / 0.778 / 0.792 | +0.000 / +0.000 / +0.000 | +0.012 / -0.009 / -0.019 |
| facescape | C3M e123 | 0.746 / 0.792 / 0.803 | +0.013 / +0.017 / +0.017 | -0.000 / -0.028 / -0.033 |
| hifi3d | fact. s1234 | 0.622 / 0.642 / 0.639 | +0.000 / +0.000 / +0.000 | -0.069 / -0.085 / -0.080 |
| hifi3d | fact. s2345 | 0.613 / 0.624 / 0.623 | +0.000 / +0.000 / +0.000 | -0.059 / -0.069 / -0.064 |
| hifi3d | C3M e123 | 0.595 / 0.619 / 0.620 | -0.031 / -0.035 / -0.035 | -0.046 / -0.059 / -0.059 |
| faceverse_neutral | fact. s1234 | 0.333 / 0.345 / 0.351 | +0.026 / +0.027 / +0.023 | +0.056 / +0.046 / +0.041 |
| faceverse_neutral | fact. s2345 | 0.372 / 0.377 / 0.383 | -0.003 / -0.006 / -0.006 | +0.025 / +0.020 / +0.016 |
| faceverse_neutral | C3M e123 | 0.382 / 0.400 / 0.417 | +0.021 / +0.019 / +0.018 | -0.007 / -0.024 / -0.033 |
| flame2023_s1 | fact. s1234 | 0.806 / 0.810 / 0.825 | -0.001 / -0.001 / -0.009 | -0.021 / -0.026 / -0.039 |
| flame2023_s1 | fact. s2345 | 0.806 / 0.809 / 0.818 | +0.000 / +0.000 / +0.000 | -0.013 / -0.018 / -0.023 |
| flame2023_s1 | C3M e123 | 0.819 / 0.825 / 0.838 | +0.023 / +0.021 / +0.019 | -0.007 / -0.014 / -0.023 |

## Curva sul numero di sorgenti (delta SR puntuale, righe incrociate)

Testa (i) con gli iperparametri scelti per "tutte tranne T", rifittata su ogni sottoinsieme di k sorgenti, e testa (ii); media [minimo, massimo] sui sottoinsiemi (k = 7: le teste di R_LOGO). Catena preregistrata: le righe piatte a 0 (s2345, e s1234 su HIFI3D) sono un ARTEFATTO DELLA GRIGLIA (r = 0 scelto); la curva con la catena corretta e' nella sezione POST HOC.

| bersaglio | braccio | testa | k = 1 | k = 2 | k = 4 | k = 7 |
|---|---|---|---|---|---|---|
| facescape | fact. s1234 | (i) | +0.004 [-0.074, +0.068] | +0.011 [-0.042, +0.064] | +0.021 [-0.014, +0.056] | +0.038 |
| facescape | fact. s1234 | (ii) | -0.016 [-0.081, +0.031] | -0.005 [-0.058, +0.033] | +0.000 [-0.023, +0.016] | -0.000 |
| facescape | fact. s2345 | (i) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 |
| facescape | fact. s2345 | (ii) | -0.014 [-0.049, +0.021] | +0.003 [-0.030, +0.036] | +0.011 [-0.007, +0.026] | +0.012 |
| facescape | C3M e123 | (i) | +0.026 [-0.022, +0.086] | +0.022 [-0.048, +0.082] | +0.015 [-0.037, +0.065] | +0.013 |
| facescape | C3M e123 | (ii) | -0.020 [-0.082, +0.023] | -0.003 [-0.054, +0.037] | +0.005 [-0.029, +0.029] | -0.000 |
| hifi3d | fact. s1234 | (i) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 |
| hifi3d | fact. s1234 | (ii) | -0.086 [-0.155, -0.050] | -0.071 [-0.114, -0.038] | -0.065 [-0.087, -0.044] | -0.069 |
| hifi3d | fact. s2345 | (i) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 |
| hifi3d | fact. s2345 | (ii) | -0.064 [-0.085, -0.051] | -0.057 [-0.077, -0.043] | -0.055 [-0.070, -0.043] | -0.059 |
| hifi3d | C3M e123 | (i) | -0.012 [-0.044, +0.021] | -0.017 [-0.096, +0.033] | -0.023 [-0.077, +0.026] | -0.031 |
| hifi3d | C3M e123 | (ii) | -0.057 [-0.087, -0.022] | -0.049 [-0.077, -0.018] | -0.045 [-0.063, -0.024] | -0.046 |
| faceverse | fact. s1234 | (i) | +0.023 [-0.015, +0.078] | +0.028 [-0.020, +0.081] | +0.027 [-0.009, +0.058] | +0.026 |
| faceverse | fact. s1234 | (ii) | +0.054 [+0.024, +0.104] | +0.059 [+0.027, +0.088] | +0.060 [+0.030, +0.082] | +0.056 |
| faceverse | fact. s2345 | (i) | -0.001 [-0.052, +0.048] | +0.006 [-0.034, +0.053] | +0.007 [-0.020, +0.035] | -0.003 |
| faceverse | fact. s2345 | (ii) | -0.001 [-0.042, +0.046] | +0.014 [-0.038, +0.060] | +0.023 [-0.017, +0.051] | +0.025 |
| faceverse | C3M e123 | (i) | +0.003 [-0.022, +0.036] | +0.006 [-0.020, +0.034] | +0.013 [-0.014, +0.034] | +0.021 |
| faceverse | C3M e123 | (ii) | -0.033 [-0.065, -0.022] | -0.019 [-0.047, +0.001] | -0.010 [-0.029, +0.004] | -0.007 |
| flame2023_s1 | fact. s1234 | (i) | +0.012 [-0.002, +0.043] | +0.008 [-0.062, +0.052] | +0.001 [-0.044, +0.050] | -0.001 |
| flame2023_s1 | fact. s1234 | (ii) | -0.017 [-0.080, +0.005] | -0.014 [-0.047, +0.013] | -0.016 [-0.042, +0.012] | -0.021 |
| flame2023_s1 | fact. s2345 | (i) | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 |
| flame2023_s1 | fact. s2345 | (ii) | -0.021 [-0.056, -0.004] | -0.011 [-0.042, +0.015] | -0.010 [-0.027, +0.007] | -0.013 |
| flame2023_s1 | C3M e123 | (i) | +0.010 [-0.022, +0.035] | +0.010 [-0.023, +0.043] | +0.013 [-0.030, +0.041] | +0.023 |
| flame2023_s1 | C3M e123 | (ii) | -0.017 [-0.037, +0.006] | -0.004 [-0.027, +0.012] | -0.001 [-0.021, +0.013] | -0.007 |

Sottoinsiemi di una sola sorgente (k = 1), delta SR della testa (i):

| bersaglio | braccio | bfm | ict | gnm | flame2023_s1 | facescape | hifi3d | faceverse | famos |
|---|---|---|---|---|---|---|---|---|---|
| facescape | fact. s1234 | -0.015 | -0.004 | +0.001 | +0.060 | - | -0.074 | -0.008 | +0.068 |
| facescape | fact. s2345 | +0.000 | +0.000 | +0.000 | +0.000 | - | +0.000 | +0.000 | +0.000 |
| facescape | C3M e123 | +0.007 | +0.016 | +0.011 | +0.086 | - | -0.022 | +0.011 | +0.071 |
| hifi3d | fact. s1234 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | - | +0.000 | +0.000 |
| hifi3d | fact. s2345 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | - | +0.000 | +0.000 |
| hifi3d | C3M e123 | -0.007 | -0.006 | +0.021 | +0.018 | -0.044 | - | -0.024 | -0.044 |
| faceverse | fact. s1234 | -0.008 | +0.009 | -0.007 | +0.078 | +0.049 | -0.015 | - | +0.056 |
| faceverse | fact. s2345 | -0.005 | -0.030 | -0.052 | +0.048 | +0.037 | +0.005 | - | -0.011 |
| faceverse | C3M e123 | -0.001 | +0.010 | -0.005 | +0.036 | +0.009 | -0.022 | - | -0.004 |
| flame2023_s1 | fact. s1234 | +0.012 | +0.007 | -0.002 | - | +0.043 | -0.001 | +0.003 | +0.025 |
| flame2023_s1 | fact. s2345 | +0.000 | +0.000 | +0.000 | - | +0.000 | +0.000 | +0.000 | +0.000 |
| flame2023_s1 | C3M e123 | +0.013 | -0.009 | +0.011 | - | +0.035 | -0.022 | +0.009 | +0.031 |

## Teste (i) scelte (CV per soggetto dentro le sorgenti)

| impostazione | braccio | r | lambda | CV testa | CV d_P | bordo | alpha / alpha0 | c | convergenza |
|---|---|---|---|---|---|---|---|---|---|
| facescape|all | fact. s1234 | 32 | 0.0001 | 0.7696 | 0.7666 | True | 0.578 | 0.988 | True (93 it.) |
| facescape|nofamos | fact. s1234 | 0 | 0 | 0.7540 | 0.7540 | False | 1.000 | 0.958 | True (0 it.) |
| hifi3d|all | fact. s1234 | 0 | 0 | 0.7792 | 0.7792 | False | 1.000 | 0.946 | True (0 it.) |
| hifi3d|nofamos | fact. s1234 | 0 | 0 | 0.7687 | 0.7687 | False | 1.000 | 0.939 | True (0 it.) |
| faceverse|all | fact. s1234 | 16 | 0.0001 | 0.8587 | 0.8196 | True | 0.648 | 1.011 | True (147 it.) |
| faceverse|nofamos | fact. s1234 | 16 | 0.0001 | 0.8546 | 0.8158 | True | 0.649 | 1.010 | True (94 it.) |
| flame2023_s1|all | fact. s1234 | 32 | 0.0001 | 0.7620 | 0.7556 | True | 0.479 | 0.995 | True (95 it.) |
| flame2023_s1|nofamos | fact. s1234 | 32 | 0.0001 | 0.7421 | 0.7412 | True | 0.460 | 0.994 | True (91 it.) |
| famos|all | fact. s1234 | 32 | 0.0001 | 0.7563 | 0.7498 | True | 0.477 | 0.992 | True (86 it.) |
| facescape|all | fact. s2345 | 0 | 0 | 0.7635 | 0.7635 | False | 1.000 | 0.961 | True (0 it.) |
| facescape|nofamos | fact. s2345 | 0 | 0 | 0.7516 | 0.7516 | False | 1.000 | 0.958 | True (0 it.) |
| hifi3d|all | fact. s2345 | 0 | 0 | 0.7827 | 0.7827 | False | 1.000 | 0.947 | True (0 it.) |
| hifi3d|nofamos | fact. s2345 | 0 | 0 | 0.7740 | 0.7740 | False | 1.000 | 0.938 | True (0 it.) |
| faceverse|all | fact. s2345 | 16 | 0.0001 | 0.8559 | 0.8199 | True | 0.658 | 1.011 | True (111 it.) |
| faceverse|nofamos | fact. s2345 | 16 | 0.0001 | 0.8554 | 0.8174 | True | 0.656 | 1.009 | True (118 it.) |
| flame2023_s1|all | fact. s2345 | 0 | 0 | 0.7564 | 0.7564 | False | 1.000 | 0.959 | True (0 it.) |
| flame2023_s1|nofamos | fact. s2345 | 0 | 0 | 0.7434 | 0.7434 | False | 1.000 | 0.954 | True (0 it.) |
| famos|all | fact. s2345 | 32 | 0.0001 | 0.7554 | 0.7521 | True | 0.529 | 0.991 | True (84 it.) |
| facescape|all | C3M e123 | 32 | 0.0001 | 0.8035 | 0.7840 | True | 0.707 | 0.993 | True (85 it.) |
| facescape|nofamos | C3M e123 | 32 | 0.0001 | 0.7917 | 0.7743 | True | 0.702 | 0.993 | True (74 it.) |
| hifi3d|all | C3M e123 | 32 | 0.0001 | 0.8105 | 0.7979 | True | 0.632 | 0.997 | True (84 it.) |
| hifi3d|nofamos | C3M e123 | 32 | 0.0001 | 0.7971 | 0.7906 | True | 0.623 | 0.997 | True (73 it.) |
| faceverse|all | C3M e123 | 16 | 0.0001 | 0.8816 | 0.8387 | True | 0.679 | 1.011 | True (141 it.) |
| faceverse|nofamos | C3M e123 | 16 | 0.0001 | 0.8800 | 0.8381 | True | 0.676 | 1.011 | True (132 it.) |
| flame2023_s1|all | C3M e123 | 32 | 0.0001 | 0.7889 | 0.7709 | True | 0.617 | 1.000 | True (80 it.) |
| flame2023_s1|nofamos | C3M e123 | 32 | 0.0001 | 0.7735 | 0.7591 | True | 0.603 | 1.000 | True (89 it.) |
| famos|all | C3M e123 | 32 | 0.0001 | 0.7887 | 0.7674 | True | 0.609 | 0.999 | True (94 it.) |

## Controlli

- K1 (riferimenti dei bracci e B = pubblicati, punto e IC; FLAME = D1): max |scarto| 3.3e-16, righe uguali True, **passa**; celle senza riferimento: ['faceverse_neutral|factorizedc3m_e123|dP|shape|sr', 'faceverse_neutral|factorizedc3m_e123|dP|form|fr']
- K2 (c dei bracci dagli held-out): fact. s1234 0.405267 contro 0.405267, fact. s2345 0.405445 contro 0.405445, C3M e123 0.257327 contro 0.257327, **passa**
- K3 (GT FaMoS TRAIN contro la GT-F di E12): max |scarto| 3.1e-07 mm, **passa**
- K4 (sorgenti: soggetti, mesh): {"bfm": [100, 500], "ict": [100, 500], "gnm": [100, 500], "flame2023_s1": [200, 1000], "facescape": [200, 1000], "hifi3d": [200, 1000], "faceverse": [200, 1000], "famos": [80, 400]}; embedding mancanti {"facescape": [], "hifi3d": [], "faceverse_neutral": [], "faceverse": [], "flame2023_s1": [], "famos": []}
- K5 (r = 0 contro d_P in CV): max |scarto| 0.0e+00
- K6: bersaglio fuori dalle sorgenti della sua testa (settings), sorgenti dei pool disgiunte dai soggetti valutati e di test, soggetti di fit e di validazione disgiunti (d3_head.split) verificati nel codice
- K7 (embedding dei pool contro gli store ufficiali, 5 soggetti di controllo per pool): max |dz| 5.2e-04, **passa**
- Righe per insieme: {"facescape": {"cross": 88725, "orig": 4950, "lavg": 4950}, "hifi3d": {"cross": 98224, "orig": 4950, "lavg": 4950}, "faceverse_neutral": {"cross": 99000, "orig": 4950, "lavg": 4950}, "faceverse": {"cross": 99000}, "flame2023_s1": {"cross": 398000, "orig": 19900, "lavg": 19900}, "famos": {"cross": 63200, "orig": 3160, "lavg": 3160}}
- Tempo: CV 2708 s, totale 3613 s (64 processi)

