# Baseline geometriche con la rimozione dei disturbi coerente con la GT

## Conclusioni (10 ottobre 2026, sui numeri delle sezioni sotto)

**1. Riproduzione.** Con la normalizzazione maxabs le righe pubblicate tornano identiche.
- 204 numeri di E12 (punto e IC, GT maxabs e F_rig_rob): scarto massimo 8e-17.
- Pipeline ricalcolata contro le matrici pubblicate: Chamfer bit per bit; ICP entro 1e-15; NICP entro 1.5e-6 (unita' maxabs, solo coppie con up60k; FaceVerse entro 5e-11); template identico; riconoscimento entro 1e-16.
- FR contro F_rig_rob di E12: entro 1.3e-6 (FR e' salvata in float32).

**2. HIFI3D, nocrop_cross, GT FR: con la normalizzazione coerente le baseline geometriche salgono e superano e108 (0.194).**
- ICP + Chamfer: da 0.325 a 0.643 [0.572, 0.703], delta +0.318 [+0.220, +0.411].
- NICP su template: da 0.260 a 0.614. NICP per coppia: da 0.367 a 0.495, delta +0.128 [+0.063, +0.197]. Chamfer: da 0.138 a 0.411.
- ICP in mm - e108: +0.449 [+0.357, +0.539].
- **Taglia.** Oracolo 0.736 [0.658, 0.802]. Stimata dalla mesh osservata (centroid size robusta, NON oracolo): 0.600 [0.498, 0.690], sopra ogni metodo non metrico. Su HIFI3D la maggior parte del segnale di FR e' la taglia: con SR l'oracolo scende a 0.074 e la stimata a 0.023.

**3. GT SR: vincono le baseline del modo cs.**
- HIFI3D: NICP per coppia 0.595 [0.546, 0.648], ICP 0.581; e108 0.279; cs NICP - e108 +0.316 [+0.249, +0.386].
- Il modo mm perde con SR: ICP -0.129 contro maxabs; il template mm scende a 0.083, perche' riportato con la rigida misura la taglia.

**4. FaceVerse e FaceScape: effetti piu' piccoli, coerenti con la poca variazione di taglia (CV 1.8% e 1.6%).**
- FaceVerse, FR: ICP mm 0.337 contro maxabs 0.201 (+0.136 [+0.042, +0.226]); e108 0.184; oracolo della taglia 0.209.
- FaceScape dev, FR: ICP mm 0.467 contro 0.398 (+0.069 [+0.039, +0.093]); NICP su template mm 0.544; e108 0.333; oracolo della taglia 0.464.
- FaceScape, taglia stimata: 0.206. Con taglie cosi' compresse la stima non basta, ed e108 la batte (-0.128 [-0.232, -0.028]).

**5. FaMoS TEST (15 persone, IC larghi).**
- scan gallery -> scan, FR: ICP mm 0.739 [0.362, 0.914], NICP mm 0.719, taglia stimata 0.715, oracolo 0.787.
- Riconoscimento scan peak -> scan, rank-1: ICP mm 0.909 contro maxabs 0.813.

**6. Riconoscimento (nocrop): la normalizzazione cambia poco.**
- HIFI3D ICP 0.996 -> 1.000 (gia' saturo); FaceVerse ICP 0.918 -> 0.958; FaceScape con espressioni 0.419 -> 0.444.
- La sola taglia stimata identifica poco: rank-1 0.13 su HIFI3D, con AUC 0.91.

**Avvertenze.**
- **Chamfer non centrata: solo diagnostica, non una baseline.** Sulla posizione assoluta nel frame del 3DMM fa rank-1 1.000 su HIFI3D e FaceScape neutra: le topologie di un soggetto condividono quella posizione, che sulle scansioni reali non si osserva. La Chamfer "in mm" di riferimento e' quella centrata: faceBench senza la scala.
- **NICP di faceBench fallisce ("Factor is exactly singular") in due casi, uguale nei tre modi.**
  - Sorgente up60k: su HIFI3D 199 coppie per coppia di topologie (776 NaN in nocrop, come le righe pubblicate); su FaceScape circa 10.275 NaN su 99.000.
  - Sorgente una patch di registrazione FaMoS: falliscono tutte.
  - Le coppie fallite sono escluse dagli Spearman e contano +inf nel riconoscimento; i blocchi FaMoS con una reg come sorgente sono omessi.
- **NICP non e' invariante alla scala.** Lavora in unita' L_d, una costante per dominio (il max|coordinata| mediano); le distanze si moltiplicano poi per L_d.
- **Semi.** `subject_pair_mean_nocrop` usa il seme 1234 dei bracci (in E12 quel gruppo non c'e'); gli altri gruppi usano i semi di E12.

## Definizioni

Codice in `aau/baselines_mm/` (definizioni in `blmm.py`), numeri qui. Le baseline pubblicate normalizzano ogni mesh per maxabs e l'ICP prealinea col bbox, che scala: sono cieche alla taglia, mentre FR la conserva. Qui la stessa pipeline faceBench in tre normalizzazioni:
- **maxabs** (pubblicata): righe esistenti, o ricalcolate dove mancavano (FaceScape, FaMoS, il template);
- **mm** (coerente con FR): mm nel frame canonico di E12, UNA trasformazione per dominio, nessuna scala per mesh; ICP rigido senza scala (prealineamento solo per traslazione); template NICP riportato con la rigida;
- **cs** (coerente con SR): come mm, ma ogni mesh a centroid size robusta CS_ref (aree della geometria passa-basso, `area_v3` modo smooth).

Unita' di lavoro per dominio (la stessa costante per tutte le mesh, perche' NICP non e' invariante alla scala): hifi3d L = 91.6 mm, CS_ref = 66.7 mm, faceverse L = 84.8 mm, CS_ref = 64.2 mm, facescape L = 108.3 mm, CS_ref = 73.0 mm, famos L = 95.0 mm, CS_ref = 64.6 mm.

Righe, soggetti, gruppi e repliche bootstrap (1000, per soggetto) sono quelli di E12; IC 95%. Colonne = GT: FR e SR di `datasets/CANONICAL_GT/eval`, maxabs per continuita'. Banali NON oracolo: |log x_a - log x_b| dalla mesh osservata. Oracolo: dalla GT (S di FR; altezza della regione dopo la rigida robusta).

Per affiancare i bracci factorized e ctrl-FR (`v3_work/trainer/tools/eval_factorized.py`, `form_spearman.csv`): il loro `mesh_pair` sono le righe di `nocrop_cross` (FaceVerse: `mesh_pair_nocrop`), il loro `subject_pair_mean` e' `subject_pair_mean_nocrop` (stesso seme 1234); le colonne gt `fr`, `sr`, `maxabs` sono le stesse matrici. Tutti i numeri anche in `spearman.csv` (colonne domain, group, method, gt, point, ci_low, ci_high, n_subjects, n_rows), i delta appaiati in `paired.csv`.

## Controlli di riproduzione

| sorgente | confronti | max abs diff |
| --- | --- | --- |
| competitors_hifi3d/recognition.csv | 16 | 1.11e-16 |
| competitors_hifi3d/template_hifi3d.npz | 2 | 0.00e+00 |
| e12/methods_spearman.csv | 204 | 8.33e-17 |
| e12/methods_spearman.csv, F_rig_rob contro FR (float32) | 102 | 1.27e-06 |
| faceverse check_fast_0of2.json: modo maxabs ricalcolato contro le matrici | 30 | 0.00e+00 |
| faceverse check_fast_1of2.json: modo maxabs ricalcolato contro le matrici | 30 | 0.00e+00 |
| faceverse check_nicp_0of1.json: modo maxabs ricalcolato contro le matrici | 30 | 4.48e-11 |
| hifi3d check_fast_0of2.json: modo maxabs ricalcolato contro le matrici | 30 | 9.92e-16 |
| hifi3d check_fast_1of2.json: modo maxabs ricalcolato contro le matrici | 30 | 9.51e-16 |
| hifi3d check_nicp_0of1.json: modo maxabs ricalcolato contro le matrici | 30 | 1.51e-06 |
| ws_faceverse_expr/recognition.csv | 12 | 1.11e-16 |

## hifi3d, nocrop_cross

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) | - | 0.194 [0.110, 0.282] | 0.279 [0.209, 0.343] | 0.630 [0.562, 0.687] | 99000 (0) |
| Chamfer eval | - | 0.155 [0.112, 0.198] | 0.220 [0.177, 0.265] | 0.372 [0.323, 0.421] | 99000 (0) |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.138 [0.102, 0.175] | 0.190 [0.151, 0.229] | 0.325 [0.282, 0.369] | 99000 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.411 [0.349, 0.467] | 0.183 [0.124, 0.242] | 0.187 [0.130, 0.245] | 99000 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.171 [0.132, 0.208] | 0.266 [0.226, 0.301] | 0.224 [0.186, 0.262] | 99000 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.646 [0.548, 0.722] | 0.066 [-0.024, 0.155] | 0.134 [0.033, 0.243] | 99000 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.325 [0.246, 0.403] | 0.495 [0.426, 0.563] | 0.355 [0.267, 0.438] | 99000 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.643 [0.572, 0.703] | 0.366 [0.281, 0.454] | 0.326 [0.236, 0.413] | 99000 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.359 [0.276, 0.439] | 0.581 [0.530, 0.631] | 0.442 [0.372, 0.507] | 99000 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.367 [0.289, 0.449] | 0.552 [0.488, 0.614] | 0.389 [0.303, 0.479] | 98224 (776) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.495 [0.423, 0.569] | 0.503 [0.437, 0.574] | 0.408 [0.328, 0.481] | 98224 (776) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.376 [0.296, 0.465] | 0.595 [0.546, 0.648] | 0.447 [0.371, 0.517] | 98224 (776) |
| NICP su template | maxabs per mesh (pubblicata) | 0.260 [0.163, 0.355] | 0.415 [0.307, 0.523] | 0.351 [0.264, 0.429] | 99000 (0) |
| NICP su template | mm, senza scala (coerente con FR) | 0.614 [0.505, 0.697] | 0.083 [-0.018, 0.185] | 0.142 [0.047, 0.234] | 99000 (0) |
| NICP su template | CS robusta (coerente con SR) | 0.189 [0.104, 0.270] | 0.301 [0.200, 0.393] | 0.250 [0.178, 0.320] | 99000 (0) |
| taglia stimata: centroid size robusta | - | 0.600 [0.498, 0.690] | 0.023 [-0.069, 0.110] | 0.095 [-0.002, 0.188] | 99000 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.578 [0.475, 0.667] | 0.023 [-0.065, 0.104] | 0.089 [-0.008, 0.186] | 99000 (0) |
| altezza stimata (y) | - | 0.439 [0.343, 0.529] | 0.034 [-0.034, 0.107] | 0.123 [0.030, 0.214] | 99000 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.736 [0.658, 0.802] | 0.074 [-0.024, 0.166] | 0.139 [0.032, 0.246] | 99000 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.589 [0.478, 0.680] | 0.188 [0.067, 0.303] | 0.172 [0.063, 0.269] | 99000 (0) |

Delta appaiati, modo coerente - maxabs (stesse repliche):

| baseline | FR (form, mm) | SR (shape) | maxabs |
| --- | --- | --- | --- |
| Chamfer (4096 pt), mm | +0.273 [+0.209, +0.340] | -0.007 [-0.055, +0.041] | -0.138 [-0.203, -0.076] |
| Chamfer (4096 pt), cs | +0.033 [+0.001, +0.066] | +0.076 [+0.046, +0.107] | -0.102 [-0.159, -0.050] |
| ICP + Chamfer, mm | +0.318 [+0.220, +0.411] | -0.129 [-0.212, -0.046] | -0.029 [-0.121, +0.055] |
| ICP + Chamfer, cs | +0.034 [-0.021, +0.087] | +0.086 [+0.040, +0.135] | +0.087 [+0.025, +0.146] |
| ICP + NICP P2Tri (per coppia), mm | +0.128 [+0.063, +0.197] | -0.049 [-0.101, +0.000] | +0.019 [-0.044, +0.079] |
| ICP + NICP P2Tri (per coppia), cs | +0.008 [-0.038, +0.057] | +0.043 [+0.007, +0.081] | +0.057 [-0.001, +0.114] |
| NICP su template, mm | +0.354 [+0.226, +0.486] | -0.332 [-0.440, -0.212] | -0.209 [-0.310, -0.111] |
| NICP su template, cs | -0.071 [-0.132, -0.002] | -0.113 [-0.182, -0.040] | -0.101 [-0.166, -0.028] |

## hifi3d, all_cross

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) | - | 0.165 [0.100, 0.241] | 0.241 [0.188, 0.298] | 0.541 [0.469, 0.603] | 148500 (0) |
| Chamfer eval | - | 0.149 [0.109, 0.194] | 0.213 [0.171, 0.257] | 0.336 [0.290, 0.380] | 148500 (0) |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.131 [0.097, 0.169] | 0.181 [0.145, 0.221] | 0.291 [0.251, 0.329] | 148500 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.348 [0.292, 0.399] | 0.163 [0.114, 0.215] | 0.161 [0.116, 0.209] | 148500 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.150 [0.116, 0.185] | 0.234 [0.199, 0.269] | 0.193 [0.158, 0.230] | 148500 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.645 [0.547, 0.725] | 0.065 [-0.017, 0.158] | 0.132 [0.037, 0.241] | 148500 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.250 [0.190, 0.314] | 0.370 [0.303, 0.429] | 0.263 [0.195, 0.331] | 148500 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.556 [0.488, 0.617] | 0.327 [0.249, 0.404] | 0.287 [0.217, 0.360] | 148500 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.264 [0.200, 0.327] | 0.425 [0.378, 0.472] | 0.322 [0.268, 0.378] | 148500 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.332 [0.260, 0.405] | 0.492 [0.430, 0.548] | 0.346 [0.266, 0.423] | 147530 (970) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.462 [0.388, 0.529] | 0.467 [0.396, 0.531] | 0.375 [0.298, 0.447] | 147530 (970) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.325 [0.246, 0.403] | 0.517 [0.465, 0.567] | 0.386 [0.319, 0.455] | 147530 (970) |
| NICP su template | maxabs per mesh (pubblicata) | 0.146 [0.087, 0.211] | 0.235 [0.172, 0.305] | 0.195 [0.141, 0.244] | 148500 (0) |
| NICP su template | mm, senza scala (coerente con FR) | 0.514 [0.410, 0.601] | 0.066 [-0.013, 0.148] | 0.113 [0.040, 0.195] | 148500 (0) |
| NICP su template | CS robusta (coerente con SR) | 0.095 [0.054, 0.139] | 0.151 [0.101, 0.201] | 0.125 [0.092, 0.156] | 148500 (0) |
| taglia stimata: centroid size robusta | - | 0.353 [0.276, 0.428] | 0.018 [-0.031, 0.072] | 0.061 [0.011, 0.122] | 148500 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.297 [0.233, 0.359] | 0.014 [-0.027, 0.059] | 0.051 [0.006, 0.108] | 148500 (0) |
| altezza stimata (y) | - | 0.425 [0.328, 0.518] | 0.043 [-0.026, 0.115] | 0.119 [0.035, 0.209] | 148500 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.736 [0.659, 0.801] | 0.074 [-0.013, 0.164] | 0.139 [0.040, 0.251] | 148500 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.589 [0.470, 0.691] | 0.188 [0.071, 0.309] | 0.172 [0.067, 0.275] | 148500 (0) |

Delta appaiati, modo coerente - maxabs (stesse repliche):

| baseline | FR (form, mm) | SR (shape) | maxabs |
| --- | --- | --- | --- |
| Chamfer (4096 pt), mm | +0.218 [+0.159, +0.273] | -0.019 [-0.055, +0.020] | -0.129 [-0.180, -0.073] |
| Chamfer (4096 pt), cs | +0.019 [-0.011, +0.047] | +0.053 [+0.022, +0.081] | -0.098 [-0.145, -0.048] |
| ICP + Chamfer, mm | +0.306 [+0.217, +0.383] | -0.044 [-0.114, +0.030] | +0.023 [-0.053, +0.100] |
| ICP + Chamfer, cs | +0.014 [-0.031, +0.065] | +0.054 [+0.016, +0.095] | +0.058 [+0.006, +0.110] |
| ICP + NICP P2Tri (per coppia), mm | +0.130 [+0.064, +0.191] | -0.025 [-0.075, +0.026] | +0.029 [-0.032, +0.087] |
| ICP + NICP P2Tri (per coppia), cs | -0.006 [-0.051, +0.039] | +0.025 [-0.009, +0.063] | +0.041 [-0.011, +0.095] |
| NICP su template, mm | +0.367 [+0.252, +0.464] | -0.168 [-0.251, -0.090] | -0.082 [-0.160, -0.004] |
| NICP su template, cs | -0.051 [-0.093, -0.010] | -0.084 [-0.129, -0.042] | -0.071 [-0.114, -0.029] |

## hifi3d, subject_pair_mean

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) | - | 0.249 [0.151, 0.351] | 0.385 [0.293, 0.468] | 0.795 [0.744, 0.842] | 4950 (0) |
| Chamfer eval | - | 0.301 [0.220, 0.387] | 0.439 [0.346, 0.530] | 0.743 [0.686, 0.796] | 4950 (0) |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.312 [0.229, 0.398] | 0.442 [0.349, 0.537] | 0.751 [0.694, 0.803] | 4950 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.740 [0.669, 0.803] | 0.318 [0.219, 0.426] | 0.323 [0.222, 0.426] | 4950 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.383 [0.300, 0.462] | 0.604 [0.529, 0.670] | 0.499 [0.426, 0.572] | 4950 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.647 [0.555, 0.723] | 0.066 [-0.014, 0.151] | 0.133 [0.034, 0.241] | 4950 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.354 [0.268, 0.439] | 0.523 [0.435, 0.604] | 0.370 [0.270, 0.458] | 4950 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.696 [0.628, 0.761] | 0.407 [0.311, 0.496] | 0.356 [0.269, 0.443] | 4950 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.404 [0.319, 0.492] | 0.659 [0.604, 0.709] | 0.500 [0.422, 0.578] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.403 [0.320, 0.486] | 0.601 [0.529, 0.665] | 0.420 [0.325, 0.508] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.547 [0.469, 0.621] | 0.557 [0.482, 0.626] | 0.447 [0.364, 0.526] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.411 [0.319, 0.507] | 0.658 [0.604, 0.708] | 0.489 [0.405, 0.567] | 4950 (0) |
| NICP su template | maxabs per mesh (pubblicata) | 0.240 [0.139, 0.341] | 0.392 [0.274, 0.519] | 0.335 [0.239, 0.433] | 4950 (0) |
| NICP su template | mm, senza scala (coerente con FR) | 0.616 [0.504, 0.706] | 0.083 [-0.011, 0.183] | 0.141 [0.049, 0.235] | 4950 (0) |
| NICP su template | CS robusta (coerente con SR) | 0.218 [0.120, 0.313] | 0.337 [0.224, 0.446] | 0.305 [0.220, 0.387] | 4950 (0) |
| taglia stimata: centroid size robusta | - | 0.618 [0.512, 0.710] | 0.027 [-0.049, 0.118] | 0.105 [0.018, 0.205] | 4950 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.611 [0.505, 0.704] | 0.026 [-0.056, 0.118] | 0.097 [0.008, 0.203] | 4950 (0) |
| altezza stimata (y) | - | 0.481 [0.373, 0.577] | 0.052 [-0.031, 0.129] | 0.140 [0.044, 0.235] | 4950 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.736 [0.652, 0.806] | 0.074 [-0.019, 0.164] | 0.139 [0.040, 0.252] | 4950 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.589 [0.471, 0.693] | 0.188 [0.057, 0.308] | 0.172 [0.070, 0.274] | 4950 (0) |

Delta appaiati, modo coerente - maxabs (stesse repliche):

| baseline | FR (form, mm) | SR (shape) | maxabs |
| --- | --- | --- | --- |
| Chamfer (4096 pt), mm | +0.428 [+0.333, +0.520] | -0.124 [-0.221, -0.033] | -0.428 [-0.529, -0.323] |
| Chamfer (4096 pt), cs | +0.071 [-0.001, +0.140] | +0.162 [+0.090, +0.238] | -0.252 [-0.328, -0.174] |
| ICP + Chamfer, mm | +0.341 [+0.234, +0.443] | -0.116 [-0.215, -0.011] | -0.014 [-0.118, +0.094] |
| ICP + Chamfer, cs | +0.050 [-0.023, +0.124] | +0.136 [+0.068, +0.205] | +0.130 [+0.053, +0.213] |
| ICP + NICP P2Tri (per coppia), mm | +0.144 [+0.072, +0.211] | -0.043 [-0.109, +0.016] | +0.027 [-0.047, +0.101] |
| ICP + NICP P2Tri (per coppia), cs | +0.008 [-0.048, +0.064] | +0.057 [+0.010, +0.106] | +0.070 [+0.004, +0.141] |
| NICP su template, mm | +0.377 [+0.238, +0.489] | -0.310 [-0.424, -0.207] | -0.194 [-0.304, -0.083] |
| NICP su template, cs | -0.022 [-0.099, +0.060] | -0.055 [-0.143, +0.042] | -0.031 [-0.116, +0.055] |

## hifi3d, subject_pair_mean_nocrop

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) | - | 0.248 [0.152, 0.344] | 0.370 [0.284, 0.454] | 0.792 [0.736, 0.838] | 4950 (0) |
| Chamfer eval | - | 0.296 [0.211, 0.382] | 0.426 [0.336, 0.518] | 0.763 [0.701, 0.814] | 4950 (0) |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.308 [0.223, 0.393] | 0.431 [0.340, 0.526] | 0.776 [0.720, 0.825] | 4950 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.736 [0.662, 0.799] | 0.311 [0.212, 0.413] | 0.325 [0.226, 0.421] | 4950 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.391 [0.312, 0.472] | 0.612 [0.540, 0.679] | 0.520 [0.446, 0.591] | 4950 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.647 [0.554, 0.727] | 0.067 [-0.016, 0.154] | 0.134 [0.039, 0.233] | 4950 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.374 [0.290, 0.458] | 0.567 [0.490, 0.639] | 0.407 [0.309, 0.499] | 4950 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.706 [0.636, 0.764] | 0.400 [0.308, 0.492] | 0.356 [0.265, 0.439] | 4950 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.413 [0.326, 0.498] | 0.671 [0.616, 0.719] | 0.512 [0.434, 0.582] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.405 [0.322, 0.486] | 0.609 [0.541, 0.673] | 0.429 [0.330, 0.521] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.550 [0.471, 0.627] | 0.561 [0.485, 0.633] | 0.455 [0.371, 0.536] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.417 [0.328, 0.504] | 0.663 [0.607, 0.715] | 0.499 [0.415, 0.572] | 4950 (0) |
| NICP su template | maxabs per mesh (pubblicata) | 0.266 [0.169, 0.358] | 0.424 [0.318, 0.517] | 0.360 [0.283, 0.439] | 4950 (0) |
| NICP su template | mm, senza scala (coerente con FR) | 0.618 [0.517, 0.706] | 0.084 [-0.005, 0.180] | 0.144 [0.058, 0.229] | 4950 (0) |
| NICP su template | CS robusta (coerente con SR) | 0.212 [0.124, 0.310] | 0.336 [0.227, 0.438] | 0.284 [0.203, 0.366] | 4950 (0) |
| taglia stimata: centroid size robusta | - | 0.616 [0.510, 0.709] | 0.024 [-0.063, 0.111] | 0.097 [0.010, 0.189] | 4950 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.610 [0.499, 0.704] | 0.025 [-0.061, 0.111] | 0.093 [0.004, 0.192] | 4950 (0) |
| altezza stimata (y) | - | 0.458 [0.355, 0.562] | 0.035 [-0.038, 0.111] | 0.128 [0.026, 0.222] | 4950 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.736 [0.658, 0.801] | 0.074 [-0.014, 0.163] | 0.139 [0.039, 0.244] | 4950 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.589 [0.483, 0.690] | 0.188 [0.071, 0.326] | 0.172 [0.073, 0.286] | 4950 (0) |

Delta appaiati, modo coerente - maxabs (stesse repliche):

| baseline | FR (form, mm) | SR (shape) | maxabs |
| --- | --- | --- | --- |
| Chamfer (4096 pt), mm | +0.428 [+0.330, +0.524] | -0.121 [-0.216, -0.032] | -0.451 [-0.563, -0.340] |
| Chamfer (4096 pt), cs | +0.083 [+0.012, +0.147] | +0.181 [+0.113, +0.251] | -0.256 [-0.336, -0.169] |
| ICP + Chamfer, mm | +0.332 [+0.232, +0.436] | -0.168 [-0.260, -0.070] | -0.051 [-0.152, +0.047] |
| ICP + Chamfer, cs | +0.040 [-0.021, +0.103] | +0.104 [+0.050, +0.160] | +0.105 [+0.038, +0.181] |
| ICP + NICP P2Tri (per coppia), mm | +0.145 [+0.076, +0.216] | -0.049 [-0.105, +0.009] | +0.026 [-0.048, +0.096] |
| ICP + NICP P2Tri (per coppia), cs | +0.013 [-0.037, +0.064] | +0.054 [+0.016, +0.096] | +0.069 [+0.007, +0.140] |
| NICP su template, mm | +0.352 [+0.227, +0.485] | -0.339 [-0.451, -0.236] | -0.216 [-0.321, -0.115] |
| NICP su template, cs | -0.054 [-0.124, +0.021] | -0.088 [-0.162, -0.010] | -0.076 [-0.148, -0.000] |

## faceverse, mesh_pair_nocrop

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) | - | 0.184 [0.130, 0.233] | 0.190 [0.136, 0.240] | 0.265 [0.201, 0.320] | 99000 (0) |
| Chamfer eval | - | 0.195 [0.138, 0.247] | 0.202 [0.143, 0.253] | 0.317 [0.255, 0.368] | 99000 (0) |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.197 [0.136, 0.255] | 0.202 [0.141, 0.257] | 0.319 [0.255, 0.370] | 99000 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.281 [0.216, 0.337] | 0.260 [0.200, 0.316] | 0.314 [0.246, 0.370] | 99000 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.241 [0.186, 0.292] | 0.255 [0.198, 0.305] | 0.317 [0.256, 0.367] | 99000 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.366 [0.293, 0.428] | 0.342 [0.270, 0.404] | 0.406 [0.331, 0.464] | 99000 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.201 [0.125, 0.271] | 0.212 [0.137, 0.280] | 0.259 [0.178, 0.338] | 99000 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.337 [0.262, 0.410] | 0.309 [0.234, 0.380] | 0.364 [0.288, 0.428] | 99000 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.290 [0.218, 0.358] | 0.306 [0.233, 0.376] | 0.361 [0.279, 0.423] | 99000 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.185 [0.118, 0.253] | 0.209 [0.141, 0.277] | 0.194 [0.116, 0.261] | 99000 (0) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.201 [0.128, 0.277] | 0.223 [0.147, 0.298] | 0.175 [0.093, 0.250] | 99000 (0) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.212 [0.131, 0.292] | 0.235 [0.158, 0.310] | 0.197 [0.115, 0.273] | 99000 (0) |
| NICP su template | maxabs per mesh (pubblicata) | 0.207 [0.111, 0.292] | 0.210 [0.115, 0.293] | 0.280 [0.178, 0.369] | 99000 (0) |
| NICP su template | mm, senza scala (coerente con FR) | 0.212 [0.117, 0.308] | 0.152 [0.058, 0.246] | 0.245 [0.152, 0.331] | 99000 (0) |
| NICP su template | CS robusta (coerente con SR) | 0.135 [0.061, 0.204] | 0.147 [0.070, 0.216] | 0.243 [0.159, 0.317] | 99000 (0) |
| taglia stimata: centroid size robusta | - | 0.126 [0.037, 0.214] | 0.080 [-0.002, 0.170] | 0.124 [0.034, 0.210] | 99000 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.123 [0.044, 0.201] | 0.085 [0.012, 0.156] | 0.117 [0.037, 0.200] | 99000 (0) |
| altezza stimata (y) | - | 0.046 [0.002, 0.095] | 0.032 [-0.017, 0.083] | 0.030 [-0.020, 0.091] | 99000 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.209 [0.114, 0.307] | 0.019 [-0.067, 0.106] | 0.024 [-0.063, 0.110] | 99000 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.092 [0.012, 0.184] | 0.091 [0.003, 0.185] | 0.105 [0.012, 0.202] | 99000 (0) |

Delta appaiati, modo coerente - maxabs (stesse repliche):

| baseline | FR (form, mm) | SR (shape) | maxabs |
| --- | --- | --- | --- |
| Chamfer (4096 pt), mm | +0.084 [+0.036, +0.135] | +0.057 [+0.017, +0.100] | -0.005 [-0.054, +0.044] |
| Chamfer (4096 pt), cs | +0.044 [+0.009, +0.082] | +0.052 [+0.018, +0.089] | -0.002 [-0.047, +0.038] |
| ICP + Chamfer, mm | +0.136 [+0.042, +0.226] | +0.097 [+0.013, +0.184] | +0.105 [+0.020, +0.190] |
| ICP + Chamfer, cs | +0.088 [+0.023, +0.153] | +0.094 [+0.033, +0.157] | +0.102 [+0.046, +0.153] |
| ICP + NICP P2Tri (per coppia), mm | +0.016 [-0.026, +0.055] | +0.014 [-0.031, +0.054] | -0.019 [-0.062, +0.022] |
| ICP + NICP P2Tri (per coppia), cs | +0.026 [-0.012, +0.063] | +0.026 [-0.013, +0.063] | +0.003 [-0.033, +0.041] |
| NICP su template, mm | +0.005 [-0.082, +0.096] | -0.058 [-0.137, +0.027] | -0.036 [-0.130, +0.055] |
| NICP su template, cs | -0.072 [-0.133, -0.012] | -0.063 [-0.124, -0.004] | -0.037 [-0.089, +0.013] |

## faceverse, subject_pair_mean_nocrop

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) | - | 0.312 [0.221, 0.404] | 0.315 [0.221, 0.409] | 0.423 [0.329, 0.516] | 4950 (0) |
| Chamfer eval | - | 0.250 [0.144, 0.348] | 0.252 [0.149, 0.348] | 0.426 [0.325, 0.517] | 4950 (0) |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.274 [0.170, 0.376] | 0.277 [0.174, 0.376] | 0.462 [0.367, 0.548] | 4950 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.389 [0.288, 0.477] | 0.362 [0.269, 0.450] | 0.446 [0.346, 0.532] | 4950 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.344 [0.246, 0.427] | 0.360 [0.269, 0.446] | 0.459 [0.367, 0.541] | 4950 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.465 [0.374, 0.543] | 0.431 [0.342, 0.510] | 0.513 [0.429, 0.588] | 4950 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.231 [0.136, 0.322] | 0.244 [0.150, 0.331] | 0.300 [0.202, 0.391] | 4950 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.400 [0.314, 0.482] | 0.367 [0.278, 0.450] | 0.432 [0.353, 0.506] | 4950 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.352 [0.260, 0.442] | 0.372 [0.277, 0.460] | 0.442 [0.347, 0.521] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.229 [0.142, 0.319] | 0.259 [0.171, 0.350] | 0.240 [0.152, 0.331] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.254 [0.153, 0.354] | 0.283 [0.186, 0.379] | 0.221 [0.121, 0.315] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.264 [0.165, 0.361] | 0.293 [0.198, 0.389] | 0.244 [0.147, 0.339] | 4950 (0) |
| NICP su template | maxabs per mesh (pubblicata) | 0.234 [0.131, 0.331] | 0.237 [0.138, 0.332] | 0.312 [0.204, 0.415] | 4950 (0) |
| NICP su template | mm, senza scala (coerente con FR) | 0.232 [0.126, 0.330] | 0.167 [0.063, 0.264] | 0.269 [0.169, 0.364] | 4950 (0) |
| NICP su template | CS robusta (coerente con SR) | 0.158 [0.068, 0.250] | 0.175 [0.085, 0.267] | 0.292 [0.192, 0.388] | 4950 (0) |
| taglia stimata: centroid size robusta | - | 0.146 [0.031, 0.246] | 0.092 [-0.008, 0.196] | 0.148 [0.037, 0.248] | 4950 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.138 [0.054, 0.223] | 0.095 [0.011, 0.183] | 0.129 [0.034, 0.230] | 4950 (0) |
| altezza stimata (y) | - | 0.082 [-0.021, 0.190] | 0.051 [-0.055, 0.161] | 0.049 [-0.056, 0.162] | 4950 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.209 [0.116, 0.302] | 0.019 [-0.069, 0.101] | 0.024 [-0.058, 0.114] | 4950 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.092 [0.010, 0.180] | 0.091 [0.004, 0.183] | 0.105 [0.012, 0.210] | 4950 (0) |

Delta appaiati, modo coerente - maxabs (stesse repliche):

| baseline | FR (form, mm) | SR (shape) | maxabs |
| --- | --- | --- | --- |
| Chamfer (4096 pt), mm | +0.115 [+0.050, +0.185] | +0.085 [+0.024, +0.145] | -0.015 [-0.084, +0.060] |
| Chamfer (4096 pt), cs | +0.070 [+0.015, +0.127] | +0.083 [+0.027, +0.139] | -0.003 [-0.070, +0.060] |
| ICP + Chamfer, mm | +0.169 [+0.055, +0.276] | +0.123 [+0.024, +0.224] | +0.132 [+0.027, +0.239] |
| ICP + Chamfer, cs | +0.121 [+0.041, +0.199] | +0.127 [+0.054, +0.202] | +0.142 [+0.072, +0.215] |
| ICP + NICP P2Tri (per coppia), mm | +0.025 [-0.027, +0.077] | +0.023 [-0.031, +0.077] | -0.020 [-0.073, +0.038] |
| ICP + NICP P2Tri (per coppia), cs | +0.035 [-0.018, +0.084] | +0.034 [-0.015, +0.085] | +0.004 [-0.044, +0.053] |
| NICP su template, mm | -0.002 [-0.097, +0.093] | -0.070 [-0.164, +0.019] | -0.044 [-0.148, +0.063] |
| NICP su template, cs | -0.076 [-0.153, -0.002] | -0.062 [-0.136, +0.011] | -0.020 [-0.082, +0.041] |

## facescape, nocrop_cross

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) | - | 0.333 [0.260, 0.404] | 0.385 [0.322, 0.445] | 0.393 [0.342, 0.445] | 99000 (0) |
| Chamfer eval | - | 0.203 [0.161, 0.244] | 0.238 [0.203, 0.271] | 0.289 [0.264, 0.316] | 99000 (0) |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.162 [0.127, 0.195] | 0.190 [0.159, 0.218] | 0.247 [0.224, 0.270] | 99000 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.211 [0.181, 0.240] | 0.217 [0.186, 0.246] | 0.207 [0.176, 0.236] | 99000 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.193 [0.159, 0.226] | 0.222 [0.190, 0.251] | 0.219 [0.187, 0.249] | 99000 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.608 [0.537, 0.676] | 0.595 [0.518, 0.661] | 0.566 [0.493, 0.638] | 99000 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.398 [0.335, 0.462] | 0.439 [0.378, 0.493] | 0.440 [0.382, 0.495] | 99000 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.467 [0.401, 0.528] | 0.491 [0.427, 0.547] | 0.454 [0.391, 0.517] | 99000 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.429 [0.355, 0.501] | 0.493 [0.427, 0.551] | 0.469 [0.402, 0.528] | 99000 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.317 [0.251, 0.385] | 0.355 [0.285, 0.419] | 0.379 [0.318, 0.438] | 88724 (10276) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.374 [0.305, 0.446] | 0.406 [0.336, 0.472] | 0.391 [0.327, 0.453] | 88726 (10274) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.346 [0.276, 0.419] | 0.398 [0.329, 0.464] | 0.396 [0.331, 0.458] | 88725 (10275) |
| NICP su template | maxabs per mesh (pubblicata) | 0.515 [0.416, 0.609] | 0.574 [0.479, 0.657] | 0.410 [0.330, 0.494] | 99000 (0) |
| NICP su template | mm, senza scala (coerente con FR) | 0.544 [0.469, 0.609] | 0.398 [0.319, 0.473] | 0.281 [0.202, 0.370] | 99000 (0) |
| NICP su template | CS robusta (coerente con SR) | 0.405 [0.326, 0.479] | 0.439 [0.367, 0.510] | 0.377 [0.304, 0.444] | 99000 (0) |
| taglia stimata: centroid size robusta | - | 0.206 [0.135, 0.269] | 0.094 [0.042, 0.146] | 0.094 [0.033, 0.164] | 99000 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.199 [0.127, 0.271] | 0.117 [0.053, 0.177] | 0.090 [0.033, 0.143] | 99000 (0) |
| altezza stimata (y) | - | 0.221 [0.104, 0.330] | 0.202 [0.091, 0.311] | 0.216 [0.123, 0.320] | 99000 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.464 [0.344, 0.573] | 0.118 [0.023, 0.229] | 0.068 [-0.024, 0.169] | 99000 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.525 [0.423, 0.616] | 0.320 [0.223, 0.425] | 0.188 [0.094, 0.292] | 99000 (0) |

Delta appaiati, modo coerente - maxabs (stesse repliche):

| baseline | FR (form, mm) | SR (shape) | maxabs |
| --- | --- | --- | --- |
| Chamfer (4096 pt), mm | +0.050 [+0.028, +0.070] | +0.026 [+0.010, +0.042] | -0.040 [-0.063, -0.018] |
| Chamfer (4096 pt), cs | +0.031 [+0.014, +0.046] | +0.031 [+0.016, +0.045] | -0.029 [-0.050, -0.008] |
| ICP + Chamfer, mm | +0.069 [+0.039, +0.093] | +0.052 [+0.027, +0.075] | +0.015 [-0.009, +0.039] |
| ICP + Chamfer, cs | +0.031 [+0.004, +0.055] | +0.054 [+0.029, +0.076] | +0.029 [+0.005, +0.053] |
| ICP + NICP P2Tri (per coppia), mm | +0.057 [+0.030, +0.087] | +0.051 [+0.025, +0.082] | +0.012 [-0.012, +0.039] |
| ICP + NICP P2Tri (per coppia), cs | +0.029 [+0.009, +0.049] | +0.043 [+0.026, +0.063] | +0.017 [-0.004, +0.037] |
| NICP su template, mm | +0.029 [-0.070, +0.123] | -0.176 [-0.258, -0.090] | -0.129 [-0.208, -0.039] |
| NICP su template, cs | -0.110 [-0.171, -0.049] | -0.135 [-0.190, -0.084] | -0.032 [-0.089, +0.024] |

## facescape, all_cross

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) | - | 0.312 [0.242, 0.383] | 0.361 [0.294, 0.422] | 0.360 [0.301, 0.414] | 148500 (0) |
| Chamfer eval | - | 0.152 [0.118, 0.188] | 0.178 [0.148, 0.207] | 0.201 [0.173, 0.227] | 148500 (0) |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.126 [0.097, 0.155] | 0.149 [0.123, 0.174] | 0.180 [0.155, 0.202] | 148500 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.184 [0.158, 0.211] | 0.189 [0.160, 0.217] | 0.180 [0.151, 0.208] | 148500 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.165 [0.131, 0.199] | 0.189 [0.158, 0.219] | 0.188 [0.157, 0.217] | 148500 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.570 [0.499, 0.641] | 0.557 [0.481, 0.625] | 0.533 [0.455, 0.603] | 148500 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.259 [0.207, 0.307] | 0.290 [0.241, 0.333] | 0.282 [0.233, 0.324] | 148500 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.381 [0.326, 0.434] | 0.401 [0.346, 0.454] | 0.370 [0.312, 0.426] | 148500 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.318 [0.253, 0.385] | 0.370 [0.311, 0.426] | 0.349 [0.289, 0.404] | 148500 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.258 [0.198, 0.318] | 0.295 [0.235, 0.352] | 0.302 [0.245, 0.358] | 135656 (12844) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.342 [0.278, 0.408] | 0.370 [0.306, 0.431] | 0.346 [0.282, 0.406] | 135657 (12843) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.300 [0.238, 0.365] | 0.350 [0.292, 0.410] | 0.341 [0.278, 0.399] | 135658 (12842) |
| NICP su template | maxabs per mesh (pubblicata) | 0.290 [0.225, 0.354] | 0.316 [0.249, 0.374] | 0.225 [0.167, 0.283] | 148500 (0) |
| NICP su template | mm, senza scala (coerente con FR) | 0.266 [0.228, 0.301] | 0.195 [0.154, 0.231] | 0.142 [0.098, 0.187] | 148500 (0) |
| NICP su template | CS robusta (coerente con SR) | 0.197 [0.157, 0.236] | 0.214 [0.172, 0.251] | 0.180 [0.142, 0.218] | 148500 (0) |
| taglia stimata: centroid size robusta | - | 0.092 [0.062, 0.120] | 0.043 [0.018, 0.066] | 0.043 [0.016, 0.073] | 148500 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.088 [0.055, 0.120] | 0.051 [0.020, 0.080] | 0.040 [0.015, 0.067] | 148500 (0) |
| altezza stimata (y) | - | 0.118 [0.056, 0.181] | 0.111 [0.055, 0.170] | 0.116 [0.063, 0.173] | 148500 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.464 [0.347, 0.576] | 0.118 [0.012, 0.235] | 0.068 [-0.032, 0.172] | 148500 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.525 [0.424, 0.619] | 0.320 [0.216, 0.431] | 0.188 [0.085, 0.295] | 148500 (0) |

Delta appaiati, modo coerente - maxabs (stesse repliche):

| baseline | FR (form, mm) | SR (shape) | maxabs |
| --- | --- | --- | --- |
| Chamfer (4096 pt), mm | +0.058 [+0.042, +0.075] | +0.040 [+0.026, +0.053] | +0.000 [-0.020, +0.020] |
| Chamfer (4096 pt), cs | +0.039 [+0.024, +0.052] | +0.041 [+0.026, +0.053] | +0.009 [-0.012, +0.027] |
| ICP + Chamfer, mm | +0.122 [+0.096, +0.145] | +0.111 [+0.090, +0.131] | +0.088 [+0.065, +0.111] |
| ICP + Chamfer, cs | +0.060 [+0.036, +0.082] | +0.080 [+0.059, +0.100] | +0.067 [+0.045, +0.089] |
| ICP + NICP P2Tri (per coppia), mm | +0.084 [+0.050, +0.120] | +0.075 [+0.046, +0.108] | +0.044 [+0.022, +0.066] |
| ICP + NICP P2Tri (per coppia), cs | +0.042 [+0.024, +0.060] | +0.055 [+0.039, +0.073] | +0.039 [+0.022, +0.059] |
| NICP su template, mm | -0.024 [-0.084, +0.040] | -0.121 [-0.172, -0.069] | -0.083 [-0.132, -0.032] |
| NICP su template, cs | -0.094 [-0.133, -0.052] | -0.102 [-0.136, -0.063] | -0.045 [-0.081, -0.006] |

## facescape, subject_pair_mean

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) | - | 0.604 [0.499, 0.697] | 0.711 [0.636, 0.774] | 0.735 [0.678, 0.789] | 4950 (0) |
| Chamfer eval | - | 0.530 [0.418, 0.629] | 0.640 [0.563, 0.713] | 0.767 [0.707, 0.814] | 4950 (0) |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.516 [0.405, 0.618] | 0.627 [0.549, 0.698] | 0.810 [0.756, 0.851] | 4950 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.691 [0.627, 0.750] | 0.703 [0.637, 0.764] | 0.662 [0.588, 0.727] | 4950 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.601 [0.508, 0.680] | 0.708 [0.646, 0.764] | 0.696 [0.626, 0.759] | 4950 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.649 [0.576, 0.715] | 0.633 [0.551, 0.699] | 0.605 [0.516, 0.680] | 4950 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.578 [0.476, 0.666] | 0.652 [0.583, 0.720] | 0.622 [0.544, 0.696] | 4950 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.690 [0.619, 0.752] | 0.722 [0.660, 0.776] | 0.656 [0.580, 0.726] | 4950 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.614 [0.524, 0.692] | 0.714 [0.649, 0.773] | 0.661 [0.585, 0.734] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.544 [0.457, 0.633] | 0.622 [0.541, 0.701] | 0.615 [0.534, 0.690] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.618 [0.536, 0.694] | 0.667 [0.590, 0.739] | 0.611 [0.535, 0.687] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.571 [0.482, 0.655] | 0.665 [0.589, 0.736] | 0.625 [0.545, 0.701] | 4950 (0) |
| NICP su template | maxabs per mesh (pubblicata) | 0.543 [0.431, 0.636] | 0.583 [0.473, 0.677] | 0.401 [0.304, 0.497] | 4950 (0) |
| NICP su template | mm, senza scala (coerente con FR) | 0.581 [0.491, 0.659] | 0.415 [0.324, 0.507] | 0.308 [0.211, 0.408] | 4950 (0) |
| NICP su template | CS robusta (coerente con SR) | 0.472 [0.378, 0.566] | 0.516 [0.427, 0.604] | 0.417 [0.325, 0.507] | 4950 (0) |
| taglia stimata: centroid size robusta | - | 0.298 [0.201, 0.388] | 0.143 [0.050, 0.238] | 0.151 [0.046, 0.262] | 4950 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.305 [0.197, 0.401] | 0.171 [0.059, 0.279] | 0.138 [0.031, 0.237] | 4950 (0) |
| altezza stimata (y) | - | 0.220 [0.083, 0.344] | 0.205 [0.072, 0.323] | 0.211 [0.090, 0.328] | 4950 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.464 [0.330, 0.570] | 0.118 [0.004, 0.219] | 0.068 [-0.034, 0.166] | 4950 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.525 [0.412, 0.609] | 0.320 [0.207, 0.419] | 0.188 [0.083, 0.287] | 4950 (0) |

Delta appaiati, modo coerente - maxabs (stesse repliche):

| baseline | FR (form, mm) | SR (shape) | maxabs |
| --- | --- | --- | --- |
| Chamfer (4096 pt), mm | +0.174 [+0.096, +0.262] | +0.076 [+0.016, +0.133] | -0.149 [-0.216, -0.085] |
| Chamfer (4096 pt), cs | +0.085 [+0.027, +0.145] | +0.081 [+0.029, +0.135] | -0.114 [-0.176, -0.059] |
| ICP + Chamfer, mm | +0.112 [+0.055, +0.172] | +0.069 [+0.028, +0.107] | +0.034 [-0.003, +0.074] |
| ICP + Chamfer, cs | +0.036 [-0.012, +0.084] | +0.062 [+0.016, +0.108] | +0.039 [-0.002, +0.084] |
| ICP + NICP P2Tri (per coppia), mm | +0.073 [+0.026, +0.120] | +0.045 [+0.008, +0.081] | -0.005 [-0.043, +0.032] |
| ICP + NICP P2Tri (per coppia), cs | +0.027 [-0.005, +0.057] | +0.043 [+0.018, +0.069] | +0.009 [-0.023, +0.041] |
| NICP su template, mm | +0.038 [-0.065, +0.149] | -0.168 [-0.260, -0.067] | -0.094 [-0.183, +0.005] |
| NICP su template, cs | -0.071 [-0.147, +0.013] | -0.067 [-0.137, +0.010] | +0.016 [-0.062, +0.097] |

## facescape, subject_pair_mean_nocrop

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| e108 (BFM+ICT+GNM) | - | 0.591 [0.489, 0.691] | 0.696 [0.624, 0.763] | 0.738 [0.681, 0.787] | 4950 (0) |
| Chamfer eval | - | 0.502 [0.391, 0.609] | 0.603 [0.514, 0.680] | 0.776 [0.730, 0.822] | 4950 (0) |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.490 [0.382, 0.599] | 0.591 [0.500, 0.672] | 0.812 [0.769, 0.853] | 4950 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.690 [0.626, 0.750] | 0.707 [0.640, 0.767] | 0.670 [0.598, 0.735] | 4950 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.605 [0.517, 0.689] | 0.712 [0.645, 0.769] | 0.697 [0.631, 0.758] | 4950 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.658 [0.591, 0.725] | 0.642 [0.567, 0.713] | 0.611 [0.530, 0.685] | 4950 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.610 [0.520, 0.690] | 0.672 [0.602, 0.738] | 0.660 [0.591, 0.730] | 4950 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.696 [0.626, 0.759] | 0.728 [0.665, 0.783] | 0.666 [0.591, 0.732] | 4950 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.642 [0.558, 0.721] | 0.738 [0.675, 0.794] | 0.690 [0.622, 0.756] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.539 [0.457, 0.625] | 0.604 [0.527, 0.679] | 0.635 [0.565, 0.699] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.617 [0.535, 0.692] | 0.668 [0.593, 0.738] | 0.630 [0.559, 0.699] | 4950 (0) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.572 [0.489, 0.657] | 0.659 [0.583, 0.726] | 0.641 [0.570, 0.707] | 4950 (0) |
| NICP su template | maxabs per mesh (pubblicata) | 0.541 [0.444, 0.634] | 0.604 [0.514, 0.692] | 0.430 [0.344, 0.516] | 4950 (0) |
| NICP su template | mm, senza scala (coerente con FR) | 0.592 [0.515, 0.662] | 0.434 [0.349, 0.518] | 0.305 [0.213, 0.397] | 4950 (0) |
| NICP su template | CS robusta (coerente con SR) | 0.491 [0.397, 0.581] | 0.533 [0.447, 0.615] | 0.459 [0.374, 0.542] | 4950 (0) |
| taglia stimata: centroid size robusta | - | 0.302 [0.216, 0.385] | 0.135 [0.058, 0.215] | 0.137 [0.043, 0.236] | 4950 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.326 [0.224, 0.421] | 0.192 [0.092, 0.290] | 0.151 [0.063, 0.243] | 4950 (0) |
| altezza stimata (y) | - | 0.240 [0.124, 0.362] | 0.221 [0.104, 0.336] | 0.234 [0.129, 0.336] | 4950 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.464 [0.337, 0.566] | 0.118 [0.017, 0.222] | 0.068 [-0.027, 0.174] | 4950 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.525 [0.414, 0.611] | 0.320 [0.217, 0.424] | 0.188 [0.088, 0.290] | 4950 (0) |

Delta appaiati, modo coerente - maxabs (stesse repliche):

| baseline | FR (form, mm) | SR (shape) | maxabs |
| --- | --- | --- | --- |
| Chamfer (4096 pt), mm | +0.199 [+0.115, +0.278] | +0.116 [+0.049, +0.181] | -0.142 [-0.205, -0.082] |
| Chamfer (4096 pt), cs | +0.114 [+0.050, +0.174] | +0.121 [+0.063, +0.180] | -0.114 [-0.172, -0.065] |
| ICP + Chamfer, mm | +0.086 [+0.035, +0.132] | +0.057 [+0.017, +0.095] | +0.006 [-0.030, +0.039] |
| ICP + Chamfer, cs | +0.032 [-0.014, +0.070] | +0.066 [+0.024, +0.111] | +0.031 [-0.009, +0.072] |
| ICP + NICP P2Tri (per coppia), mm | +0.078 [+0.041, +0.115] | +0.064 [+0.032, +0.099] | -0.004 [-0.039, +0.033] |
| ICP + NICP P2Tri (per coppia), cs | +0.033 [+0.004, +0.062] | +0.055 [+0.031, +0.081] | +0.007 [-0.023, +0.037] |
| NICP su template, mm | +0.051 [-0.049, +0.153] | -0.170 [-0.264, -0.080] | -0.125 [-0.211, -0.032] |
| NICP su template, cs | -0.050 [-0.121, +0.018] | -0.071 [-0.133, -0.009] | +0.029 [-0.037, +0.104] |

## famos, scan gallery -> scan

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.335 [-0.040, 0.701] | 0.483 [0.103, 0.741] | 0.356 [0.048, 0.634] | 105 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.623 [0.199, 0.879] | 0.551 [0.127, 0.823] | 0.348 [0.033, 0.613] | 105 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.349 [-0.009, 0.692] | 0.494 [0.109, 0.734] | 0.342 [0.051, 0.592] | 105 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.668 [0.289, 0.893] | 0.581 [0.200, 0.843] | 0.356 [0.035, 0.606] | 105 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.541 [0.203, 0.780] | 0.567 [0.288, 0.770] | 0.441 [0.162, 0.630] | 105 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.739 [0.362, 0.914] | 0.548 [0.185, 0.799] | 0.322 [0.002, 0.580] | 105 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.601 [0.254, 0.815] | 0.612 [0.306, 0.787] | 0.461 [0.148, 0.673] | 105 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.605 [0.277, 0.832] | 0.802 [0.604, 0.883] | 0.556 [0.262, 0.737] | 105 (0) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.719 [0.418, 0.874] | 0.779 [0.525, 0.897] | 0.517 [0.212, 0.729] | 105 (0) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.584 [0.243, 0.818] | 0.775 [0.540, 0.860] | 0.527 [0.232, 0.702] | 105 (0) |
| taglia stimata: centroid size robusta | - | 0.715 [0.392, 0.885] | 0.333 [-0.178, 0.687] | 0.139 [-0.228, 0.460] | 105 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.545 [0.127, 0.819] | 0.353 [-0.196, 0.728] | 0.135 [-0.274, 0.493] | 105 (0) |
| altezza stimata (y) | - | 0.632 [0.255, 0.874] | 0.277 [-0.257, 0.649] | 0.048 [-0.292, 0.371] | 105 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.787 [0.546, 0.900] | 0.213 [-0.232, 0.573] | 0.058 [-0.290, 0.380] | 105 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.543 [0.228, 0.720] | 0.132 [-0.208, 0.456] | 0.099 [-0.241, 0.403] | 105 (0) |

## famos, reg gallery -> reg

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.363 [0.018, 0.643] | 0.576 [0.246, 0.771] | 0.435 [0.150, 0.657] | 105 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.726 [0.399, 0.919] | 0.678 [0.339, 0.865] | 0.482 [0.215, 0.685] | 105 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.418 [0.057, 0.732] | 0.660 [0.381, 0.836] | 0.532 [0.276, 0.719] | 105 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.711 [0.368, 0.901] | 0.682 [0.369, 0.871] | 0.448 [0.181, 0.685] | 105 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.490 [0.160, 0.721] | 0.698 [0.456, 0.857] | 0.575 [0.310, 0.751] | 105 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.780 [0.446, 0.930] | 0.704 [0.389, 0.876] | 0.462 [0.154, 0.664] | 105 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.554 [0.202, 0.786] | 0.736 [0.491, 0.865] | 0.585 [0.320, 0.746] | 105 (0) |
| taglia stimata: centroid size robusta | - | 0.668 [0.318, 0.868] | 0.373 [-0.086, 0.686] | 0.163 [-0.197, 0.498] | 105 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.483 [0.009, 0.809] | 0.391 [-0.199, 0.774] | 0.171 [-0.269, 0.587] | 105 (0) |
| altezza stimata (y) | - | 0.491 [-0.020, 0.813] | 0.222 [-0.273, 0.606] | 0.017 [-0.349, 0.378] | 105 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.787 [0.546, 0.900] | 0.213 [-0.232, 0.573] | 0.058 [-0.290, 0.380] | 105 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.543 [0.228, 0.720] | 0.132 [-0.208, 0.456] | 0.099 [-0.241, 0.403] | 105 (0) |

## famos, scan gallery -> reg

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.115 [-0.071, 0.341] | 0.135 [-0.000, 0.298] | 0.142 [-0.011, 0.330] | 210 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.409 [0.102, 0.666] | 0.253 [-0.036, 0.521] | 0.175 [-0.028, 0.353] | 210 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.118 [-0.088, 0.349] | 0.145 [-0.012, 0.337] | 0.136 [-0.048, 0.347] | 210 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.659 [0.270, 0.878] | 0.600 [0.233, 0.835] | 0.375 [0.087, 0.614] | 210 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.431 [0.147, 0.671] | 0.503 [0.282, 0.671] | 0.403 [0.155, 0.579] | 210 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.709 [0.397, 0.871] | 0.517 [0.186, 0.743] | 0.327 [0.052, 0.532] | 210 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.404 [0.151, 0.577] | 0.480 [0.268, 0.603] | 0.369 [0.148, 0.526] | 210 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.483 [0.174, 0.687] | 0.685 [0.430, 0.792] | 0.464 [0.187, 0.654] | 210 (0) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.607 [0.258, 0.796] | 0.712 [0.427, 0.831] | 0.455 [0.183, 0.635] | 210 (0) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.519 [0.167, 0.711] | 0.683 [0.406, 0.800] | 0.459 [0.186, 0.640] | 210 (0) |
| taglia stimata: centroid size robusta | - | 0.293 [0.017, 0.522] | 0.141 [-0.102, 0.402] | 0.048 [-0.092, 0.228] | 210 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.409 [0.069, 0.665] | 0.317 [-0.042, 0.593] | 0.164 [-0.062, 0.407] | 210 (0) |
| altezza stimata (y) | - | 0.352 [-0.039, 0.641] | 0.149 [-0.173, 0.481] | 0.005 [-0.209, 0.254] | 210 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.787 [0.546, 0.900] | 0.213 [-0.232, 0.573] | 0.058 [-0.290, 0.380] | 210 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.543 [0.228, 0.720] | 0.132 [-0.208, 0.456] | 0.099 [-0.241, 0.403] | 210 (0) |

## famos, scan peak -> scan

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.187 [0.049, 0.319] | 0.231 [0.089, 0.326] | 0.170 [0.063, 0.266] | 5852 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.444 [0.146, 0.667] | 0.354 [0.083, 0.584] | 0.214 [0.044, 0.388] | 5852 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.204 [0.049, 0.350] | 0.248 [0.086, 0.358] | 0.173 [0.060, 0.273] | 5852 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.483 [0.170, 0.696] | 0.397 [0.134, 0.636] | 0.234 [0.032, 0.425] | 5852 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.339 [0.156, 0.471] | 0.368 [0.222, 0.453] | 0.292 [0.122, 0.417] | 5852 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.604 [0.298, 0.764] | 0.422 [0.136, 0.642] | 0.262 [0.028, 0.447] | 5852 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.362 [0.180, 0.479] | 0.386 [0.216, 0.474] | 0.302 [0.114, 0.435] | 5852 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.324 [0.121, 0.460] | 0.414 [0.249, 0.504] | 0.275 [0.106, 0.400] | 5852 (0) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.423 [0.170, 0.596] | 0.451 [0.225, 0.607] | 0.291 [0.109, 0.459] | 5852 (0) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.310 [0.098, 0.456] | 0.409 [0.222, 0.508] | 0.275 [0.103, 0.409] | 5852 (0) |
| taglia stimata: centroid size robusta | - | 0.642 [0.333, 0.819] | 0.311 [-0.142, 0.634] | 0.136 [-0.201, 0.435] | 5852 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.449 [0.096, 0.692] | 0.302 [-0.124, 0.611] | 0.140 [-0.171, 0.405] | 5852 (0) |
| altezza stimata (y) | - | 0.518 [0.154, 0.764] | 0.227 [-0.158, 0.565] | 0.051 [-0.196, 0.312] | 5852 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.787 [0.546, 0.900] | 0.213 [-0.234, 0.573] | 0.057 [-0.292, 0.379] | 5852 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.542 [0.225, 0.720] | 0.131 [-0.208, 0.454] | 0.098 [-0.241, 0.403] | 5852 (0) |

## famos, scan nearneutral -> scan

| metodo | normalizzazione | FR (form, mm) | SR (shape) | maxabs | righe (NaN) |
| --- | --- | --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.319 [-0.032, 0.656] | 0.473 [0.117, 0.698] | 0.359 [0.091, 0.605] | 5852 (0) |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.623 [0.214, 0.868] | 0.533 [0.122, 0.797] | 0.343 [0.053, 0.590] | 5852 (0) |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.336 [-0.007, 0.656] | 0.489 [0.118, 0.710] | 0.353 [0.089, 0.590] | 5852 (0) |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.667 [0.272, 0.882] | 0.569 [0.189, 0.824] | 0.349 [0.041, 0.587] | 5852 (0) |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.536 [0.220, 0.743] | 0.592 [0.374, 0.738] | 0.470 [0.229, 0.649] | 5852 (0) |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.766 [0.453, 0.912] | 0.551 [0.201, 0.789] | 0.346 [0.029, 0.576] | 5852 (0) |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.568 [0.241, 0.790] | 0.627 [0.361, 0.776] | 0.493 [0.194, 0.679] | 5852 (0) |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.584 [0.235, 0.787] | 0.776 [0.559, 0.857] | 0.529 [0.231, 0.708] | 5852 (0) |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.670 [0.336, 0.848] | 0.746 [0.484, 0.869] | 0.489 [0.213, 0.673] | 5852 (0) |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.562 [0.203, 0.778] | 0.755 [0.499, 0.849] | 0.521 [0.235, 0.699] | 5852 (0) |
| taglia stimata: centroid size robusta | - | 0.710 [0.392, 0.877] | 0.328 [-0.179, 0.684] | 0.140 [-0.232, 0.468] | 5852 (0) |
| taglia stimata: sqrt(area robusta) | - | 0.546 [0.125, 0.815] | 0.343 [-0.186, 0.707] | 0.135 [-0.247, 0.486] | 5852 (0) |
| altezza stimata (y) | - | 0.596 [0.194, 0.855] | 0.239 [-0.258, 0.624] | 0.029 [-0.287, 0.331] | 5852 (0) |
| ORACOLO: solo taglia (S di FR) | - | 0.787 [0.546, 0.900] | 0.213 [-0.234, 0.573] | 0.057 [-0.292, 0.379] | 5852 (0) |
| ORACOLO: solo altezza (regione, FR) | - | 0.542 [0.225, 0.720] | 0.131 [-0.208, 0.454] | 0.098 [-0.241, 0.403] | 5852 (0) |

## Riconoscimento

### hifi3d, nocrop

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.477 [0.445, 0.513] | 0.782 [0.767, 0.797] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.698 [0.676, 0.719] | 0.834 [0.820, 0.848] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.725 [0.704, 0.744] | 0.800 [0.788, 0.811] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.996 [0.991, 0.999] | 0.999 [0.999, 1.000] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.000] |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.999 [0.998, 1.000] | 1.000 [1.000, 1.000] |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.990 [0.982, 0.998] | 1.000 [1.000, 1.000] |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.990 [0.982, 0.998] | 1.000 [1.000, 1.000] |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.990 [0.982, 0.998] | 1.000 [1.000, 1.000] |
| NICP su template | maxabs per mesh (pubblicata) | 0.875 [0.843, 0.905] | 0.998 [0.997, 0.999] |
| NICP su template | mm, senza scala (coerente con FR) | 0.884 [0.855, 0.913] | 0.998 [0.996, 0.999] |
| NICP su template | CS robusta (coerente con SR) | 0.546 [0.510, 0.584] | 0.964 [0.954, 0.972] |
| taglia stimata: centroid size robusta | - | 0.129 [0.104, 0.158] | 0.912 [0.897, 0.925] |
| taglia stimata: sqrt(area robusta) | - | 0.110 [0.086, 0.138] | 0.885 [0.868, 0.899] |
| altezza stimata (y) | - | 0.307 [0.292, 0.326] | 0.888 [0.869, 0.905] |

### hifi3d, crop

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.472 [0.445, 0.499] | 0.699 [0.684, 0.715] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.446 [0.420, 0.470] | 0.710 [0.699, 0.722] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.521 [0.492, 0.552] | 0.744 [0.732, 0.755] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.284 [0.237, 0.338] | 0.888 [0.865, 0.911] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.680 [0.656, 0.705] | 1.000 [1.000, 1.000] |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.567 [0.534, 0.598] | 0.988 [0.982, 0.993] |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.777 [0.740, 0.810] | 0.997 [0.995, 0.998] |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.865 [0.841, 0.887] | 1.000 [1.000, 1.000] |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.724 [0.693, 0.756] | 1.000 [1.000, 1.000] |
| NICP su template | maxabs per mesh (pubblicata) | 0.031 [0.011, 0.054] | 0.640 [0.615, 0.668] |
| NICP su template | mm, senza scala (coerente con FR) | 0.017 [0.001, 0.037] | 0.654 [0.620, 0.688] |
| NICP su template | CS robusta (coerente con SR) | 0.010 [0.000, 0.025] | 0.559 [0.544, 0.577] |
| taglia stimata: centroid size robusta | - | 0.010 [0.000, 0.025] | 0.504 [0.500, 0.508] |
| taglia stimata: sqrt(area robusta) | - | 0.010 [0.000, 0.025] | 0.501 [0.500, 0.503] |
| altezza stimata (y) | - | 0.035 [0.012, 0.061] | 0.710 [0.671, 0.748] |

### faceverse, nocrop

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.740 [0.698, 0.783] | 0.882 [0.853, 0.910] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.829 [0.790, 0.866] | 0.919 [0.895, 0.941] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.825 [0.788, 0.862] | 0.919 [0.895, 0.942] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.942 [0.918, 0.963] | 0.982 [0.973, 0.991] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.918 [0.895, 0.939] | 0.986 [0.979, 0.992] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.958 [0.938, 0.976] | 0.985 [0.975, 0.992] |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.959 [0.936, 0.977] | 0.991 [0.985, 0.996] |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.959 [0.940, 0.975] | 0.995 [0.992, 0.998] |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.967 [0.948, 0.983] | 0.992 [0.987, 0.996] |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.971 [0.953, 0.986] | 0.995 [0.992, 0.998] |
| NICP su template | maxabs per mesh (pubblicata) | 0.440 [0.392, 0.485] | 0.928 [0.914, 0.942] |
| NICP su template | mm, senza scala (coerente con FR) | 0.483 [0.432, 0.536] | 0.932 [0.916, 0.947] |
| NICP su template | CS robusta (coerente con SR) | 0.387 [0.342, 0.438] | 0.870 [0.843, 0.898] |
| taglia stimata: centroid size robusta | - | 0.053 [0.036, 0.074] | 0.767 [0.731, 0.803] |
| taglia stimata: sqrt(area robusta) | - | 0.040 [0.022, 0.061] | 0.770 [0.738, 0.799] |
| altezza stimata (y) | - | 0.022 [0.013, 0.033] | 0.618 [0.591, 0.648] |

### faceverse, crop

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.396 [0.336, 0.457] | 0.809 [0.781, 0.837] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.628 [0.581, 0.673] | 0.893 [0.876, 0.910] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.741 [0.701, 0.778] | 0.917 [0.900, 0.933] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.949 [0.929, 0.967] | 0.982 [0.974, 0.990] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.513 [0.455, 0.571] | 0.777 [0.734, 0.814] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.866 [0.835, 0.894] | 0.994 [0.987, 0.998] |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.892 [0.860, 0.922] | 0.980 [0.970, 0.989] |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.909 [0.876, 0.938] | 0.983 [0.974, 0.992] |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.942 [0.918, 0.963] | 0.989 [0.980, 0.996] |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.948 [0.923, 0.970] | 0.990 [0.982, 0.996] |
| NICP su template | maxabs per mesh (pubblicata) | 0.143 [0.101, 0.188] | 0.773 [0.740, 0.805] |
| NICP su template | mm, senza scala (coerente con FR) | 0.085 [0.051, 0.122] | 0.706 [0.671, 0.741] |
| NICP su template | CS robusta (coerente con SR) | 0.042 [0.022, 0.066] | 0.670 [0.640, 0.705] |
| taglia stimata: centroid size robusta | - | 0.010 [0.000, 0.024] | 0.503 [0.494, 0.512] |
| taglia stimata: sqrt(area robusta) | - | 0.010 [0.000, 0.025] | 0.505 [0.497, 0.513] |
| altezza stimata (y) | - | 0.033 [0.020, 0.048] | 0.626 [0.596, 0.657] |

### facescape, nocrop

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.576 [0.561, 0.590] | 0.690 [0.685, 0.695] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.613 [0.597, 0.629] | 0.693 [0.688, 0.698] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.652 [0.635, 0.670] | 0.694 [0.688, 0.699] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.829 [0.811, 0.845] | 0.924 [0.910, 0.936] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.872 [0.859, 0.882] | 0.964 [0.955, 0.971] |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.860 [0.844, 0.873] | 0.958 [0.949, 0.966] |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.756 [0.737, 0.776] | 0.930 [0.921, 0.939] |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.767 [0.748, 0.788] | 0.946 [0.938, 0.954] |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.763 [0.743, 0.783] | 0.945 [0.936, 0.953] |
| NICP su template | maxabs per mesh (pubblicata) | 0.722 [0.682, 0.760] | 0.988 [0.983, 0.992] |
| NICP su template | mm, senza scala (coerente con FR) | 0.659 [0.614, 0.704] | 0.969 [0.958, 0.978] |
| NICP su template | CS robusta (coerente con SR) | 0.425 [0.386, 0.463] | 0.931 [0.915, 0.945] |
| taglia stimata: centroid size robusta | - | 0.076 [0.059, 0.099] | 0.694 [0.669, 0.719] |
| taglia stimata: sqrt(area robusta) | - | 0.050 [0.030, 0.074] | 0.695 [0.672, 0.718] |
| altezza stimata (y) | - | 0.311 [0.301, 0.324] | 0.831 [0.804, 0.856] |

### facescape, crop

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.127 [0.094, 0.159] | 0.709 [0.694, 0.724] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.471 [0.444, 0.496] | 0.655 [0.646, 0.665] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.313 [0.271, 0.349] | 0.719 [0.706, 0.731] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.095 [0.068, 0.126] | 0.818 [0.791, 0.845] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.541 [0.528, 0.556] | 1.000 [1.000, 1.000] |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.285 [0.245, 0.325] | 0.955 [0.941, 0.966] |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.286 [0.243, 0.326] | 0.958 [0.947, 0.968] |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.565 [0.551, 0.579] | 1.000 [1.000, 1.000] |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.519 [0.504, 0.535] | 0.997 [0.994, 0.998] |
| NICP su template | maxabs per mesh (pubblicata) | 0.024 [0.007, 0.045] | 0.604 [0.581, 0.629] |
| NICP su template | mm, senza scala (coerente con FR) | 0.016 [0.002, 0.035] | 0.527 [0.517, 0.537] |
| NICP su template | CS robusta (coerente con SR) | 0.014 [0.000, 0.031] | 0.555 [0.541, 0.572] |
| taglia stimata: centroid size robusta | - | 0.010 [0.000, 0.025] | 0.501 [0.497, 0.505] |
| taglia stimata: sqrt(area robusta) | - | 0.010 [0.000, 0.025] | 0.500 [0.496, 0.504] |
| altezza stimata (y) | - | 0.010 [0.000, 0.025] | 0.510 [0.501, 0.520] |

### facescape_expr, nocrop

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.290 [0.259, 0.323] | 0.646 [0.634, 0.657] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.280 [0.250, 0.312] | 0.635 [0.624, 0.646] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.297 [0.263, 0.332] | 0.636 [0.625, 0.647] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.738 [0.703, 0.773] | 0.944 [0.930, 0.957] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.419 [0.389, 0.452] | 0.779 [0.755, 0.802] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.444 [0.413, 0.473] | 0.787 [0.762, 0.812] |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.432 [0.399, 0.463] | 0.789 [0.763, 0.812] |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.369 [0.343, 0.400] | 0.766 [0.745, 0.786] |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.382 [0.352, 0.412] | 0.777 [0.756, 0.799] |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.384 [0.354, 0.414] | 0.777 [0.756, 0.798] |
| NICP su template | maxabs per mesh (pubblicata) | 0.191 [0.162, 0.219] | 0.840 [0.817, 0.861] |
| NICP su template | mm, senza scala (coerente con FR) | 0.204 [0.178, 0.232] | 0.799 [0.771, 0.829] |
| NICP su template | CS robusta (coerente con SR) | 0.174 [0.149, 0.203] | 0.790 [0.762, 0.818] |
| taglia stimata: centroid size robusta | - | 0.024 [0.013, 0.039] | 0.606 [0.581, 0.632] |
| taglia stimata: sqrt(area robusta) | - | 0.024 [0.011, 0.040] | 0.617 [0.592, 0.646] |
| altezza stimata (y) | - | 0.040 [0.025, 0.058] | 0.658 [0.626, 0.691] |

### facescape_expr, crop

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.078 [0.058, 0.098] | 0.635 [0.621, 0.650] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.183 [0.154, 0.210] | 0.618 [0.604, 0.631] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.123 [0.099, 0.146] | 0.646 [0.631, 0.662] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.685 [0.638, 0.732] | 0.908 [0.889, 0.927] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.066 [0.047, 0.088] | 0.722 [0.689, 0.753] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.326 [0.296, 0.353] | 0.930 [0.912, 0.947] |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.151 [0.126, 0.178] | 0.851 [0.829, 0.872] |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.149 [0.122, 0.175] | 0.827 [0.802, 0.852] |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.327 [0.297, 0.354] | 0.920 [0.903, 0.937] |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.230 [0.201, 0.259] | 0.883 [0.865, 0.899] |
| NICP su template | maxabs per mesh (pubblicata) | 0.015 [0.005, 0.028] | 0.564 [0.547, 0.583] |
| NICP su template | mm, senza scala (coerente con FR) | 0.005 [0.001, 0.010] | 0.520 [0.509, 0.531] |
| NICP su template | CS robusta (coerente con SR) | 0.010 [0.003, 0.021] | 0.545 [0.529, 0.559] |
| taglia stimata: centroid size robusta | - | 0.010 [0.000, 0.024] | 0.497 [0.491, 0.504] |
| taglia stimata: sqrt(area robusta) | - | 0.010 [0.000, 0.025] | 0.500 [0.495, 0.506] |
| altezza stimata (y) | - | 0.012 [0.004, 0.021] | 0.507 [0.498, 0.516] |

### famos, scan peak -> scan

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.514 [0.432, 0.596] | 0.738 [0.696, 0.782] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.677 [0.601, 0.754] | 0.803 [0.754, 0.855] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.593 [0.517, 0.674] | 0.755 [0.720, 0.795] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.677 [0.614, 0.745] | 0.797 [0.739, 0.852] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.813 [0.755, 0.873] | 0.853 [0.821, 0.890] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.909 [0.875, 0.945] | 0.890 [0.858, 0.924] |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.864 [0.811, 0.909] | 0.851 [0.821, 0.886] |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.801 [0.749, 0.854] | 0.836 [0.812, 0.865] |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.828 [0.765, 0.878] | 0.842 [0.818, 0.872] |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.844 [0.795, 0.885] | 0.836 [0.811, 0.867] |
| taglia stimata: centroid size robusta | - | 0.404 [0.277, 0.537] | 0.866 [0.799, 0.919] |
| taglia stimata: sqrt(area robusta) | - | 0.278 [0.146, 0.437] | 0.783 [0.697, 0.852] |
| altezza stimata (y) | - | 0.232 [0.093, 0.391] | 0.756 [0.630, 0.858] |

### famos, scan nearneutral -> scan

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.998 [0.993, 1.000] | 1.000 [0.999, 1.000] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 1.000 [1.000, 1.000] | 1.000 [0.998, 1.000] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| ICP + Chamfer | CS robusta (coerente con SR) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| taglia stimata: centroid size robusta | - | 0.725 [0.605, 0.835] | 0.971 [0.934, 0.994] |
| taglia stimata: sqrt(area robusta) | - | 0.574 [0.438, 0.705] | 0.953 [0.898, 0.983] |
| altezza stimata (y) | - | 0.443 [0.269, 0.629] | 0.895 [0.790, 0.963] |

### famos, reg peak -> reg

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.574 [0.494, 0.653] | 0.739 [0.693, 0.788] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.694 [0.629, 0.762] | 0.800 [0.748, 0.855] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.622 [0.537, 0.699] | 0.757 [0.714, 0.805] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.689 [0.627, 0.754] | 0.785 [0.726, 0.839] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.873 [0.814, 0.926] | 0.906 [0.867, 0.944] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.947 [0.918, 0.971] | 0.936 [0.904, 0.968] |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.916 [0.844, 0.969] | 0.917 [0.883, 0.951] |
| taglia stimata: centroid size robusta | - | 0.318 [0.192, 0.467] | 0.844 [0.764, 0.909] |
| taglia stimata: sqrt(area robusta) | - | 0.280 [0.160, 0.426] | 0.792 [0.700, 0.867] |
| altezza stimata (y) | - | 0.184 [0.082, 0.310] | 0.721 [0.612, 0.819] |

### famos, scan peak -> reg

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.443 [0.352, 0.558] | 0.685 [0.641, 0.736] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.452 [0.308, 0.604] | 0.720 [0.667, 0.776] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.512 [0.434, 0.601] | 0.691 [0.641, 0.742] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.694 [0.629, 0.761] | 0.791 [0.733, 0.846] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.799 [0.730, 0.871] | 0.874 [0.839, 0.911] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.890 [0.844, 0.935] | 0.898 [0.862, 0.933] |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.734 [0.601, 0.856] | 0.863 [0.828, 0.904] |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 0.754 [0.671, 0.835] | 0.852 [0.818, 0.888] |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 0.751 [0.647, 0.840] | 0.844 [0.809, 0.886] |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 0.754 [0.657, 0.840] | 0.840 [0.808, 0.880] |
| taglia stimata: centroid size robusta | - | 0.093 [0.017, 0.231] | 0.592 [0.511, 0.683] |
| taglia stimata: sqrt(area robusta) | - | 0.177 [0.057, 0.329] | 0.703 [0.583, 0.807] |
| altezza stimata (y) | - | 0.105 [0.012, 0.242] | 0.604 [0.524, 0.694] |

### famos, scan nearneutral -> reg

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.589 [0.388, 0.794] | 0.789 [0.728, 0.869] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.500 [0.279, 0.709] | 0.808 [0.735, 0.880] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.651 [0.444, 0.849] | 0.806 [0.733, 0.896] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 1.000 [1.000, 1.000] | 0.999 [0.998, 1.000] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 1.000 [1.000, 1.000] | 0.997 [0.990, 1.000] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.856 [0.680, 0.990] | 0.993 [0.979, 0.999] |
| ICP + NICP P2Tri (per coppia) | maxabs per mesh (pubblicata) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| ICP + NICP P2Tri (per coppia) | mm, senza scala (coerente con FR) | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] |
| ICP + NICP P2Tri (per coppia) | CS robusta (coerente con SR) | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] |
| taglia stimata: centroid size robusta | - | 0.067 [0.000, 0.201] | 0.554 [0.485, 0.642] |
| taglia stimata: sqrt(area robusta) | - | 0.134 [0.000, 0.335] | 0.664 [0.534, 0.790] |
| altezza stimata (y) | - | 0.108 [0.014, 0.248] | 0.628 [0.527, 0.737] |

### famos, reg peak -> scan

| metodo | normalizzazione | rank-1 | AUC |
| --- | --- | --- | --- |
| Chamfer (4096 pt) | maxabs per mesh (pubblicata) | 0.160 [0.043, 0.315] | 0.628 [0.598, 0.665] |
| Chamfer (4096 pt) | mm, senza scala (coerente con FR) | 0.187 [0.060, 0.353] | 0.703 [0.643, 0.771] |
| Chamfer (4096 pt) | CS robusta (coerente con SR) | 0.237 [0.103, 0.392] | 0.651 [0.611, 0.702] |
| Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore) | mm, senza scala (coerente con FR) | 0.651 [0.592, 0.711] | 0.787 [0.731, 0.841] |
| ICP + Chamfer | maxabs per mesh (pubblicata) | 0.754 [0.675, 0.839] | 0.864 [0.821, 0.909] |
| ICP + Chamfer | mm, senza scala (coerente con FR) | 0.885 [0.835, 0.933] | 0.897 [0.856, 0.942] |
| ICP + Chamfer | CS robusta (coerente con SR) | 0.744 [0.656, 0.837] | 0.830 [0.785, 0.882] |
| taglia stimata: centroid size robusta | - | 0.069 [0.000, 0.204] | 0.541 [0.494, 0.613] |
| taglia stimata: sqrt(area robusta) | - | 0.100 [0.010, 0.240] | 0.593 [0.521, 0.675] |
| altezza stimata (y) | - | 0.158 [0.086, 0.269] | 0.648 [0.562, 0.737] |

