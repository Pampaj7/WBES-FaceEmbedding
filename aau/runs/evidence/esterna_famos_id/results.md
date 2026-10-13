# Validazione esterna su FaMoS: riconoscimento dell'identita' su scansioni grezze

Generato da `aau/esterna_famos_id/ei.py eval` (regole in `PROTOCOLLO.md`, scritte e committate prima dei numeri). Lettura DESCRITTIVA: nessun test confermativo; i risultati sfavorevoli sono riportati come gli altri.

## Dati

- 15 soggetti di FaMoS TEST, 150 scansioni grezze (10 sequenze per soggetto, le stesse per tutti: bareteeth, blow_cheeks, disgust, fear, happiness, kissing, mouth_down, rolling_lips, surprise, wrinkle_nose), 11175 coppie (675 stessa persona, 10500 persone diverse). Nessuna registrazione FLAME: topologie tutte diverse.
- Preparazione comune (rilevatore su render frontale, patch NoW in mm): 150/150 patch, ripieghi del rilevatore sulla grezza 0, sulla patch 0; dettagli in `prep_eval.json`.
- IC 95% percentile, 1000 repliche per soggetto (seme 1234, `famos_eval.bootstrap_counts`); retrieval: query ricampionate per soggetto, galleria fissa; verifica: pesi c_s per le coppie della stessa persona, c_a c_b per le altre.

## Metriche per metodo

valore [IC 95%]. rank-1, mAP, AUC: piu' alto = meglio; EER: piu' basso = meglio.

| metodo | rank-1 | mAP | EER | AUC | coppie non finite |
|---|---|---|---|---|---|
| nostro s1234, d_F non cal. | 0.853 [0.813, 0.900] | 0.664 [0.600, 0.739] | 0.242 [0.184, 0.294] | 0.830 [0.783, 0.883] | 0 |
| nostro s1234, d_F cal. | 0.873 [0.827, 0.927] | 0.651 [0.585, 0.726] | 0.265 [0.210, 0.323] | 0.816 [0.760, 0.870] | 0 |
| nostro s1234, d_P | 0.813 [0.753, 0.873] | 0.636 [0.562, 0.716] | 0.241 [0.187, 0.294] | 0.830 [0.781, 0.885] | 0 |
| nostro s2345, d_F non cal. | 0.820 [0.767, 0.880] | 0.641 [0.566, 0.724] | 0.259 [0.201, 0.313] | 0.811 [0.758, 0.869] | 0 |
| nostro s2345, d_F cal. | 0.853 [0.807, 0.907] | 0.628 [0.549, 0.712] | 0.283 [0.218, 0.332] | 0.805 [0.743, 0.862] | 0 |
| nostro s2345, d_P | 0.800 [0.753, 0.847] | 0.614 [0.539, 0.696] | 0.265 [0.204, 0.320] | 0.808 [0.754, 0.869] | 0 |
| ICP + Chamfer in mm | 0.947 [0.913, 0.980] | 0.822 [0.773, 0.873] | 0.173 [0.121, 0.225] | 0.895 [0.858, 0.932] | 0 |
| ICP + Chamfer cs | 0.913 [0.873, 0.947] | 0.753 [0.686, 0.818] | 0.221 [0.164, 0.261] | 0.860 [0.818, 0.907] | 0 |
| NoW metrical (rilevatore) | 0.967 [0.940, 0.993] | 0.865 [0.812, 0.914] | 0.157 [0.101, 0.206] | 0.918 [0.879, 0.955] | 0 |
| NoW originale (rilevatore) | 0.940 [0.907, 0.967] | 0.791 [0.730, 0.850] | 0.196 [0.145, 0.239] | 0.878 [0.837, 0.923] | 0 |
| ULRSSM faces_xyz per coppia, mm | 0.927 [0.867, 0.980] | 0.727 [0.656, 0.795] | 0.225 [0.166, 0.282] | 0.869 [0.809, 0.921] | 0 |
| ULRSSM faces_xyz per coppia, cs | 0.860 [0.813, 0.900] | 0.631 [0.556, 0.707] | 0.273 [0.221, 0.316] | 0.807 [0.755, 0.860] | 0 |
| GW euclid_128_cgw2, mm | 0.820 [0.753, 0.887] | 0.628 [0.554, 0.706] | 0.215 [0.163, 0.261] | 0.872 [0.824, 0.917] | 0 |
| GW euclid_128_cgw2, cs | 0.707 [0.613, 0.787] | 0.449 [0.386, 0.513] | 0.268 [0.222, 0.312] | 0.811 [0.759, 0.861] | 0 |

## Delta appaiati: nostro s1234, d_F non cal. contro ciascun concorrente

positivo = il nostro meglio (rank-1, mAP, AUC: nostro - concorrente; EER: concorrente - nostro); stesse repliche.

| concorrente | rank-1 | mAP | EER | AUC |
|---|---|---|---|---|
| ICP + Chamfer in mm | -0.093 [-0.133, -0.047] | -0.158 [-0.222, -0.098] | -0.068 [-0.120, -0.017] | -0.065 [-0.110, -0.025] |
| ICP + Chamfer cs | -0.060 [-0.107, -0.007] | -0.089 [-0.149, -0.028] | -0.021 [-0.071, +0.016] | -0.031 [-0.067, +0.004] |
| NoW metrical (rilevatore) | -0.113 [-0.147, -0.073] | -0.200 [-0.253, -0.144] | -0.085 [-0.132, -0.044] | -0.088 [-0.123, -0.056] |
| NoW originale (rilevatore) | -0.087 [-0.127, -0.040] | -0.126 [-0.174, -0.071] | -0.046 [-0.086, -0.009] | -0.049 [-0.081, -0.020] |
| ULRSSM faces_xyz per coppia, mm | -0.073 [-0.127, -0.020] | -0.063 [-0.118, -0.010] | -0.017 [-0.071, +0.026] | -0.039 [-0.083, +0.004] |
| ULRSSM faces_xyz per coppia, cs | -0.007 [-0.040, +0.040] | +0.034 [-0.013, +0.081] | +0.031 [-0.009, +0.066] | +0.023 [-0.008, +0.057] |
| GW euclid_128_cgw2, mm | +0.033 [-0.027, +0.100] | +0.037 [-0.037, +0.112] | -0.027 [-0.092, +0.025] | -0.042 [-0.101, +0.010] |
| GW euclid_128_cgw2, cs | +0.147 [+0.067, +0.253] | +0.216 [+0.158, +0.285] | +0.026 [-0.009, +0.065] | +0.019 [-0.018, +0.058] |

## Delta appaiati: nostro s1234, d_F cal. contro ciascun concorrente

positivo = il nostro meglio (rank-1, mAP, AUC: nostro - concorrente; EER: concorrente - nostro); stesse repliche.

| concorrente | rank-1 | mAP | EER | AUC |
|---|---|---|---|---|
| ICP + Chamfer in mm | -0.073 [-0.120, -0.027] | -0.172 [-0.224, -0.119] | -0.092 [-0.145, -0.042] | -0.079 [-0.126, -0.036] |
| ICP + Chamfer cs | -0.040 [-0.093, +0.007] | -0.103 [-0.172, -0.032] | -0.044 [-0.105, +0.001] | -0.045 [-0.095, +0.002] |
| NoW metrical (rilevatore) | -0.093 [-0.133, -0.047] | -0.214 [-0.261, -0.165] | -0.108 [-0.156, -0.067] | -0.102 [-0.140, -0.066] |
| NoW originale (rilevatore) | -0.067 [-0.113, -0.020] | -0.140 [-0.198, -0.072] | -0.069 [-0.117, -0.027] | -0.063 [-0.112, -0.016] |
| ULRSSM faces_xyz per coppia, mm | -0.053 [-0.120, +0.020] | -0.077 [-0.121, -0.030] | -0.040 [-0.090, -0.003] | -0.053 [-0.093, -0.016] |
| ULRSSM faces_xyz per coppia, cs | +0.013 [-0.033, +0.073] | +0.020 [-0.040, +0.084] | +0.008 [-0.041, +0.051] | +0.009 [-0.035, +0.055] |
| GW euclid_128_cgw2, mm | +0.053 [-0.020, +0.133] | +0.023 [-0.039, +0.081] | -0.050 [-0.115, +0.004] | -0.056 [-0.118, -0.003] |
| GW euclid_128_cgw2, cs | +0.167 [+0.080, +0.280] | +0.202 [+0.143, +0.268] | +0.003 [-0.050, +0.053] | +0.005 [-0.048, +0.056] |

## Delta appaiati: nostro s1234, d_P contro ciascun concorrente

positivo = il nostro meglio (rank-1, mAP, AUC: nostro - concorrente; EER: concorrente - nostro); stesse repliche.

| concorrente | rank-1 | mAP | EER | AUC |
|---|---|---|---|---|
| ICP + Chamfer in mm | -0.133 [-0.200, -0.073] | -0.187 [-0.257, -0.116] | -0.068 [-0.124, -0.015] | -0.065 [-0.118, -0.021] |
| ICP + Chamfer cs | -0.100 [-0.147, -0.047] | -0.118 [-0.173, -0.065] | -0.021 [-0.074, +0.018] | -0.030 [-0.067, +0.003] |
| NoW metrical (rilevatore) | -0.153 [-0.214, -0.100] | -0.229 [-0.297, -0.159] | -0.084 [-0.136, -0.042] | -0.088 [-0.130, -0.052] |
| NoW originale (rilevatore) | -0.127 [-0.193, -0.073] | -0.155 [-0.204, -0.098] | -0.045 [-0.092, -0.010] | -0.048 [-0.085, -0.021] |
| ULRSSM faces_xyz per coppia, mm | -0.113 [-0.193, -0.047] | -0.092 [-0.160, -0.026] | -0.016 [-0.079, +0.033] | -0.039 [-0.092, +0.010] |
| ULRSSM faces_xyz per coppia, cs | -0.047 [-0.113, +0.013] | +0.005 [-0.034, +0.045] | +0.031 [-0.013, +0.065] | +0.023 [-0.010, +0.055] |
| GW euclid_128_cgw2, mm | -0.007 [-0.093, +0.080] | +0.008 [-0.079, +0.097] | -0.027 [-0.097, +0.031] | -0.042 [-0.109, +0.014] |
| GW euclid_128_cgw2, cs | +0.107 [+0.027, +0.213] | +0.187 [+0.127, +0.259] | +0.026 [-0.009, +0.061] | +0.019 [-0.018, +0.057] |

## Delta appaiati: nostro s2345, d_F non cal. contro ciascun concorrente

positivo = il nostro meglio (rank-1, mAP, AUC: nostro - concorrente; EER: concorrente - nostro); stesse repliche.

| concorrente | rank-1 | mAP | EER | AUC |
|---|---|---|---|---|
| ICP + Chamfer in mm | -0.127 [-0.173, -0.080] | -0.181 [-0.247, -0.121] | -0.086 [-0.139, -0.039] | -0.084 [-0.131, -0.040] |
| ICP + Chamfer cs | -0.093 [-0.153, -0.027] | -0.112 [-0.177, -0.052] | -0.039 [-0.091, -0.004] | -0.049 [-0.091, -0.014] |
| NoW metrical (rilevatore) | -0.147 [-0.193, -0.100] | -0.223 [-0.281, -0.168] | -0.102 [-0.150, -0.065] | -0.107 [-0.142, -0.073] |
| NoW originale (rilevatore) | -0.120 [-0.180, -0.067] | -0.149 [-0.198, -0.101] | -0.063 [-0.106, -0.029] | -0.067 [-0.103, -0.036] |
| ULRSSM faces_xyz per coppia, mm | -0.107 [-0.153, -0.067] | -0.086 [-0.143, -0.034] | -0.034 [-0.086, +0.009] | -0.058 [-0.100, -0.017] |
| ULRSSM faces_xyz per coppia, cs | -0.040 [-0.080, +0.007] | +0.011 [-0.033, +0.053] | +0.013 [-0.028, +0.042] | +0.005 [-0.027, +0.034] |
| GW euclid_128_cgw2, mm | +0.000 [-0.074, +0.073] | +0.014 [-0.067, +0.092] | -0.044 [-0.110, +0.007] | -0.061 [-0.120, -0.009] |
| GW euclid_128_cgw2, cs | +0.113 [+0.013, +0.240] | +0.193 [+0.128, +0.266] | +0.008 [-0.032, +0.045] | +0.000 [-0.043, +0.041] |

## Delta appaiati: nostro s2345, d_F cal. contro ciascun concorrente

positivo = il nostro meglio (rank-1, mAP, AUC: nostro - concorrente; EER: concorrente - nostro); stesse repliche.

| concorrente | rank-1 | mAP | EER | AUC |
|---|---|---|---|---|
| ICP + Chamfer in mm | -0.093 [-0.133, -0.053] | -0.194 [-0.252, -0.138] | -0.110 [-0.156, -0.052] | -0.090 [-0.140, -0.045] |
| ICP + Chamfer cs | -0.060 [-0.113, -0.013] | -0.125 [-0.201, -0.048] | -0.062 [-0.119, -0.010] | -0.056 [-0.110, -0.005] |
| NoW metrical (rilevatore) | -0.113 [-0.153, -0.067] | -0.236 [-0.290, -0.184] | -0.126 [-0.167, -0.079] | -0.113 [-0.154, -0.074] |
| NoW originale (rilevatore) | -0.087 [-0.133, -0.040] | -0.163 [-0.224, -0.095] | -0.087 [-0.131, -0.038] | -0.074 [-0.129, -0.025] |
| ULRSSM faces_xyz per coppia, mm | -0.073 [-0.120, -0.027] | -0.099 [-0.143, -0.055] | -0.058 [-0.097, -0.016] | -0.064 [-0.104, -0.028] |
| ULRSSM faces_xyz per coppia, cs | -0.007 [-0.047, +0.033] | -0.003 [-0.064, +0.063] | -0.010 [-0.053, +0.038] | -0.002 [-0.053, +0.043] |
| GW euclid_128_cgw2, mm | +0.033 [-0.047, +0.113] | +0.001 [-0.069, +0.063] | -0.068 [-0.125, -0.007] | -0.067 [-0.128, -0.014] |
| GW euclid_128_cgw2, cs | +0.147 [+0.053, +0.260] | +0.180 [+0.107, +0.255] | -0.015 [-0.063, +0.043] | -0.006 [-0.066, +0.049] |

## Delta appaiati: nostro s2345, d_P contro ciascun concorrente

positivo = il nostro meglio (rank-1, mAP, AUC: nostro - concorrente; EER: concorrente - nostro); stesse repliche.

| concorrente | rank-1 | mAP | EER | AUC |
|---|---|---|---|---|
| ICP + Chamfer in mm | -0.147 [-0.193, -0.093] | -0.208 [-0.281, -0.142] | -0.092 [-0.145, -0.041] | -0.087 [-0.142, -0.039] |
| ICP + Chamfer cs | -0.113 [-0.167, -0.067] | -0.139 [-0.196, -0.091] | -0.044 [-0.098, -0.009] | -0.053 [-0.097, -0.018] |
| NoW metrical (rilevatore) | -0.167 [-0.213, -0.120] | -0.250 [-0.313, -0.188] | -0.108 [-0.157, -0.067] | -0.110 [-0.155, -0.069] |
| NoW originale (rilevatore) | -0.140 [-0.193, -0.087] | -0.177 [-0.221, -0.132] | -0.069 [-0.112, -0.034] | -0.071 [-0.107, -0.040] |
| ULRSSM faces_xyz per coppia, mm | -0.127 [-0.180, -0.073] | -0.113 [-0.177, -0.054] | -0.040 [-0.100, +0.009] | -0.061 [-0.113, -0.014] |
| ULRSSM faces_xyz per coppia, cs | -0.060 [-0.100, -0.020] | -0.017 [-0.048, +0.012] | +0.007 [-0.033, +0.038] | +0.001 [-0.034, +0.030] |
| GW euclid_128_cgw2, mm | -0.020 [-0.093, +0.053] | -0.014 [-0.103, +0.077] | -0.050 [-0.118, +0.008] | -0.064 [-0.129, -0.006] |
| GW euclid_128_cgw2, cs | +0.093 [+0.013, +0.187] | +0.166 [+0.099, +0.238] | +0.002 [-0.040, +0.043] | -0.003 [-0.047, +0.041] |

## Controlli di riproduzione (contro la matrice, FaMoS TEST galleria)

- `backproject`: `{"ok": true, "backproject_vs_pr_noise_hifi3d_mm": {"n": 1632, "nan": 0, "max_abs": 0.0}, "render_px_diversi": 0}`
- `embed_factorized_s1234`: `{"ok": true, "Z_galleria_famos_vs_matrice": {"n": 3855, "nan": 0, "max_abs": 0.0004685521125793457, "median_abs": 9.967386722564697e-05, "p95_abs": 0.0002771884202957152, "equal": 0}}`
- `embed_factorized_s2345`: `{"ok": true, "Z_galleria_famos_vs_matrice": {"n": 3855, "nan": 0, "max_abs": 0.0006372630596160889, "median_abs": 9.40561294555664e-05, "p95_abs": 0.00027962327003479, "equal": 0}}`
- `factorized_s1234`: `{}`
- `factorized_s2345`: `{}`
- `gw`: `{"ok": true, "checks": {"fr_famos_galleria": {"n": 12, "nan": 0, "max_abs": 0.0}, "sr_famos_galleria": {"n": 12, "nan": 0, "max_abs": 0.0}}}`
- `icp`: `{"ok": true, "checks": {"famos_galleria_mm": {"n": 180, "nan": 0, "max_abs": 3.863576125695545e-14}, "famos_galleria_cs": {"n": 180, "nan": 0, "max_abs": 0.0}}}`
- `now_metrical`: `{"ok": true, "checks": {"now_original_famos_galleria": {"n": 12, "nan": 0, "max_abs": 1.3493930417496358e-09}, "now_metrical_famos_galleria": {"n": 12, "nan": 0, "max_abs": 8.739254742096136e-08}}}`
- `now_original`: `{"ok": true, "checks": {"now_original_famos_galleria": {"n": 12, "nan": 0, "max_abs": 1.3493930417496358e-09}, "now_metrical_famos_galleria": {"n": 12, "nan": 0, "max_abs": 8.739254742096136e-08}}}`
- `ulrssm`: `{"ok": true, "checks": {"pair_fr_famos_galleria": {"n": 105, "nan": 0, "max_abs": 0.0, "median_abs": 0.0, "p95_abs": 0.0, "equal": 105}, "pair_sr_famos_galleria": {"n": 105, "nan": 0, "max_abs": 0.0, "median_abs": 0.0, "p95_abs": 0.0, "equal": 105}}}`

File: `metrics.csv` (una riga per metodo e metrica), `deltas.csv` (tutti i delta appaiati), `summary.json`, `selection.json`, `prep_eval.json`, `prep_pilot.json`, `controls/`.

<!-- da qui in giu': scritto a mano, conservato da ei.py eval -->

## Lettura (descrittiva, scritta dopo i numeri)

- **Sfavorevole per noi.** Le letture principali sono la d_F non calibrata, s1234 e s2345. In rank-1 e mAP restano
  sotto ICP + Chamfer in mm e cs, NoW metrical e originale col rilevatore, e ULRSSM in mm, con l'IC del delta tutto
  sotto 0 in entrambi i semi. In EER restano sotto ICP mm e i due NoW con IC sotto 0; contro ICP cs e ULRSSM mm l'IC
  include 0 per s1234, mentre per s2345 contro ICP cs e' sotto 0. Il migliore e' NoW metrical: rank-1 0,967, mAP 0,865,
  EER 0,157, contro 0,853 / 0,664 / 0,242 di s1234. Siamo pari (IC che include 0) a ULRSSM cs e GW mm, e sopra GW cs
  in rank-1 e mAP.
- **La taglia aiuta il riconoscimento.** Per ogni concorrente la lettura in mm batte quella a centroid size: ICP rank-1
  0,947 contro 0,913, ULRSSM 0,927 contro 0,860, GW 0,820 contro 0,707. Per noi la d_F supera la d_P in rank-1 (0,853
  contro 0,813 in s1234) e la eguaglia in EER.
- **La calibrazione non aiuta qui.** La d_F calibrata alza il rank-1 (0,873 / 0,853) ma abbassa mAP e AUC e alza l'EER
  in entrambi i semi.
- **Limiti.** 15 persone, quindi IC larghi; un solo sistema di acquisizione; 10 sequenze con espressioni spesso marcate,
  scelte a caso. Il compito misura l'identita' sotto variazione d'espressione, non la forma graduata della GT.

## Esecuzione e deviazioni dal protocollo

- Job: selezione 1072552; pilota della preparazione 1072553, 1072555, 1072560; preparazione 1072561; ICP, NoW e GW
  1072584 (CPU); nostro e ULRSSM 1072583 (L40S); distanze del nostro e metriche 1072921, rifatte nel job 1073887 solo
  per le etichette GW di results.md (stessi numeri).
- Correzioni dopo il protocollo (29baaa7), nessuna regola cambiata: in 730a7ab la cartella della cache di ULRSSM (bug)
  e le statistiche degli scarti dei controlli; poi le etichette GW senza `|` (rompevano la tabella) e questa sezione
  conservata da `ei.py eval`.
- La prova su A10 (job 1072730, 1072836) era solo di collaudo; le sue uscite sono state cancellate prima del job L40S.
  Su A10 il controllo di ULRSSM usciva di tolleranza (0,38 mm al massimo); su L40S, la GPU dei plug-in della matrice,
  lo scarto e' 0 su 105 coppie su 105.
- Tutti i controlli di riproduzione sono entro tolleranza (sopra). Su questi controlli GW ricalcola la centroid size
  delle patch con uno scarto di 3,1e-7 mm dalla cache comune.
