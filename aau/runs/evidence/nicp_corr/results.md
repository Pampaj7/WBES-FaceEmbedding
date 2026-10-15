# NICP per le corrispondenze contro il residuo della matrice

Generato da `aau/nicp_corr/nc.py` (regole in `REGOLE.md`, committate prima del codice e dei numeri). HIFI3D e FaceScape dev, righe di all_cross senza crop del sottocampione del rumore landmark (8%, seme 20261013), una maschera per vista; Spearman pesato con GT FR e SR, 1000 repliche per soggetto col seme della T1, IC 95% percentile, delta appaiati. Coppie primarie: FR con mm, SR con cs.

Righe: hifi3d 7802 / 7942; facescape 6293 / 7916.

Collaudo (`nicp_p2tri` contro la matrice, tolleranza 1e-6 mm):

- hifi3d mm: scarto massimo 1.74e-04 mm, mediano 2.8e-13, NaN 62 sulle stesse coppie: True; coppie di topologie riprodotte qui 18/20, le altre sul nodo d'origine (a768-l40s-05); strutturale min(corr_xy - ICP) = 0.00494 mm
- hifi3d cs: scarto massimo 5.61e-05 mm, mediano 0.0e+00, NaN 62 sulle stesse coppie: True; coppie di topologie riprodotte qui 19/20, le altre sul nodo d'origine (a256-a40-04); strutturale min(corr_xy - ICP) = 0.00436 mm
- facescape mm: scarto massimo 2.69e-06 mm, mediano 0.0e+00, NaN 828 sulle stesse coppie: True; coppie di topologie riprodotte qui 19/20, le altre sul nodo d'origine (i256-a10-07); strutturale min(corr_xy - ICP) = 0.0033 mm
- facescape cs: scarto massimo 4.19e-10 mm, mediano 0.0e+00, NaN 828 sulle stesse coppie: True; coppie di topologie riprodotte qui 20/20, le altre sul nodo d'origine (-); strutturale min(corr_xy - ICP) = 0.00398 mm

## hifi3d

| colonna | rho FR [IC] | rho SR [IC] |
|---|---|---|
| NICP per coppia, residuo P2Tri (matrice) (mm) | 0.499 [0.421, 0.577] | 0.506 [0.435, 0.578] |
| NICP per le corrispondenze (media dei due versi) (mm) | 0.695 [0.615, 0.757] | 0.315 [0.217, 0.414] |
| NICP per le corrispondenze, un verso (quello della matrice) (mm) | 0.633 [0.546, 0.705] | 0.303 [0.202, 0.407] |
| NICP per coppia, residuo P2Tri, media dei due versi (mm) | 0.551 [0.470, 0.629] | 0.562 [0.489, 0.636] |
| ICP + Chamfer (mm) | 0.649 [0.574, 0.710] | 0.370 [0.276, 0.466] |
| NICP su template (mm) | 0.618 [0.504, 0.705] | 0.091 [-0.023, 0.202] |
| NICP per coppia, residuo P2Tri (matrice) (cs) | 0.374 [0.283, 0.470] | 0.600 [0.546, 0.656] |
| NICP per le corrispondenze (media dei due versi) (cs) | 0.383 [0.285, 0.474] | 0.634 [0.570, 0.686] |
| NICP per le corrispondenze, un verso (quello della matrice) (cs) | 0.344 [0.252, 0.431] | 0.571 [0.509, 0.627] |
| NICP per coppia, residuo P2Tri, media dei due versi (cs) | 0.406 [0.309, 0.505] | 0.655 [0.601, 0.711] |
| ICP + Chamfer (cs) | 0.356 [0.266, 0.445] | 0.585 [0.527, 0.637] |
| NICP su template (cs) | 0.181 [0.100, 0.263] | 0.300 [0.192, 0.391] |
| nostro s1234, d_F non cal. | 0.644 [0.566, 0.714] | 0.556 [0.470, 0.637] |
| nostro s2345, d_F non cal. | 0.634 [0.560, 0.703] | 0.539 [0.440, 0.625] |
| nostro s1234, d_P | 0.426 [0.321, 0.518] | 0.629 [0.553, 0.694] |
| nostro s2345, d_P | 0.420 [0.310, 0.525] | 0.614 [0.526, 0.689] |
| REALY (protocollo, landmark esatti) | 0.489 [0.388, 0.576] | 0.703 [0.630, 0.764] |

Delta appaiati (a - b, rho con la GT; * = primaria):

| GT | modo | tipo | a | b | delta [IC] |
|---|---|---|---|---|---|
| FR | mm | D1: corrispondenze - residuo * | NICP per le corrispondenze (media dei due versi) (mm) | NICP per coppia, residuo P2Tri (matrice) (mm) | +0.196 [+0.102, +0.294] |
| FR | mm | D2: corrispondenze - ICP * | NICP per le corrispondenze (media dei due versi) (mm) | ICP + Chamfer (mm) | +0.046 [+0.014, +0.082] |
| FR | mm | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (mm) | nostro s1234, d_F non cal. | +0.051 [-0.015, +0.115] |
| FR | mm | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (mm) | nostro s2345, d_F non cal. | +0.061 [-0.009, +0.124] |
| FR | mm | corrispondenze - NICP su template | NICP per le corrispondenze (media dei due versi) (mm) | NICP su template (mm) | +0.077 [+0.032, +0.130] |
| FR | mm | corrispondenze - REALY | NICP per le corrispondenze (media dei due versi) (mm) | REALY (protocollo, landmark esatti) | +0.206 [+0.082, +0.337] |
| FR | mm | solo la lettura, un verso | NICP per le corrispondenze, un verso (quello della matrice) (mm) | NICP per coppia, residuo P2Tri (matrice) (mm) | +0.134 [+0.031, +0.238] |
| FR | mm | solo la lettura, due versi | NICP per le corrispondenze (media dei due versi) (mm) | NICP per coppia, residuo P2Tri, media dei due versi (mm) | +0.144 [+0.055, +0.239] |
| FR | cs | D1: corrispondenze - residuo | NICP per le corrispondenze (media dei due versi) (cs) | NICP per coppia, residuo P2Tri (matrice) (cs) | +0.009 [-0.035, +0.053] |
| FR | cs | D2: corrispondenze - ICP | NICP per le corrispondenze (media dei due versi) (cs) | ICP + Chamfer (cs) | +0.027 [+0.004, +0.052] |
| FR | cs | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (cs) | nostro s1234, d_F non cal. | -0.261 [-0.347, -0.177] |
| FR | cs | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (cs) | nostro s2345, d_F non cal. | -0.251 [-0.342, -0.165] |
| FR | cs | corrispondenze - NICP su template | NICP per le corrispondenze (media dei due versi) (cs) | NICP su template (cs) | +0.202 [+0.122, +0.282] |
| FR | cs | corrispondenze - REALY | NICP per le corrispondenze (media dei due versi) (cs) | REALY (protocollo, landmark esatti) | -0.105 [-0.184, -0.027] |
| FR | cs | solo la lettura, un verso | NICP per le corrispondenze, un verso (quello della matrice) (cs) | NICP per coppia, residuo P2Tri (matrice) (cs) | -0.031 [-0.074, +0.011] |
| FR | cs | solo la lettura, due versi | NICP per le corrispondenze (media dei due versi) (cs) | NICP per coppia, residuo P2Tri, media dei due versi (cs) | -0.022 [-0.064, +0.016] |
| SR | mm | D1: corrispondenze - residuo | NICP per le corrispondenze (media dei due versi) (mm) | NICP per coppia, residuo P2Tri (matrice) (mm) | -0.191 [-0.286, -0.092] |
| SR | mm | D2: corrispondenze - ICP | NICP per le corrispondenze (media dei due versi) (mm) | ICP + Chamfer (mm) | -0.056 [-0.089, -0.023] |
| SR | mm | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (mm) | nostro s1234, d_P | -0.315 [-0.409, -0.222] |
| SR | mm | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (mm) | nostro s2345, d_P | -0.299 [-0.399, -0.202] |
| SR | mm | corrispondenze - NICP su template | NICP per le corrispondenze (media dei due versi) (mm) | NICP su template (mm) | +0.224 [+0.166, +0.284] |
| SR | mm | corrispondenze - REALY | NICP per le corrispondenze (media dei due versi) (mm) | REALY (protocollo, landmark esatti) | -0.388 [-0.495, -0.284] |
| SR | mm | solo la lettura, un verso | NICP per le corrispondenze, un verso (quello della matrice) (mm) | NICP per coppia, residuo P2Tri (matrice) (mm) | -0.203 [-0.299, -0.103] |
| SR | mm | solo la lettura, due versi | NICP per le corrispondenze (media dei due versi) (mm) | NICP per coppia, residuo P2Tri, media dei due versi (mm) | -0.247 [-0.339, -0.148] |
| SR | cs | D1: corrispondenze - residuo * | NICP per le corrispondenze (media dei due versi) (cs) | NICP per coppia, residuo P2Tri (matrice) (cs) | +0.034 [-0.010, +0.082] |
| SR | cs | D2: corrispondenze - ICP * | NICP per le corrispondenze (media dei due versi) (cs) | ICP + Chamfer (cs) | +0.049 [+0.025, +0.074] |
| SR | cs | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (cs) | nostro s1234, d_P | +0.004 [-0.049, +0.056] |
| SR | cs | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (cs) | nostro s2345, d_P | +0.019 [-0.041, +0.087] |
| SR | cs | corrispondenze - NICP su template | NICP per le corrispondenze (media dei due versi) (cs) | NICP su template (cs) | +0.333 [+0.252, +0.414] |
| SR | cs | corrispondenze - REALY | NICP per le corrispondenze (media dei due versi) (cs) | REALY (protocollo, landmark esatti) | -0.069 [-0.146, +0.007] |
| SR | cs | solo la lettura, un verso | NICP per le corrispondenze, un verso (quello della matrice) (cs) | NICP per coppia, residuo P2Tri (matrice) (cs) | -0.028 [-0.071, +0.020] |
| SR | cs | solo la lettura, due versi | NICP per le corrispondenze (media dei due versi) (cs) | NICP per coppia, residuo P2Tri, media dei due versi (cs) | -0.021 [-0.061, +0.025] |

Verdetto (REGOLE.md sez. 4):

- FR con mm: D1 +0.196 [+0.102, +0.294], D2 +0.046 [+0.014, +0.082]: **sopra l'ICP**
- SR con cs: D1 +0.034 [-0.010, +0.082], D2 +0.049 [+0.025, +0.074]: **cancella ancora l'identita'**

Spearman semplice fra colonne sulle righe della maschera (descrittivo): mm_nicp_corr ~ mm_nicp_p2tri 0.462; mm_nicp_corr ~ mm_rigid_icp_chamfer 0.877; cs_nicp_corr ~ cs_nicp_p2tri 0.753; cs_nicp_corr ~ cs_rigid_icp_chamfer 0.869

## facescape

| colonna | rho FR [IC] | rho SR [IC] |
|---|---|---|
| NICP per coppia, residuo P2Tri (matrice) (mm) | 0.381 [0.297, 0.461] | 0.417 [0.342, 0.493] |
| NICP per le corrispondenze (media dei due versi) (mm) | 0.632 [0.563, 0.695] | 0.657 [0.593, 0.715] |
| NICP per le corrispondenze, un verso (quello della matrice) (mm) | 0.503 [0.435, 0.565] | 0.529 [0.467, 0.586] |
| NICP per coppia, residuo P2Tri, media dei due versi (mm) | 0.515 [0.430, 0.604] | 0.562 [0.482, 0.640] |
| ICP + Chamfer (mm) | 0.460 [0.389, 0.526] | 0.491 [0.424, 0.552] |
| NICP su template (mm) | 0.535 [0.449, 0.606] | 0.392 [0.301, 0.478] |
| NICP per coppia, residuo P2Tri (matrice) (cs) | 0.353 [0.271, 0.436] | 0.410 [0.337, 0.485] |
| NICP per le corrispondenze (media dei due versi) (cs) | 0.586 [0.498, 0.665] | 0.678 [0.615, 0.734] |
| NICP per le corrispondenze, un verso (quello della matrice) (cs) | 0.466 [0.385, 0.542] | 0.540 [0.475, 0.599] |
| NICP per coppia, residuo P2Tri, media dei due versi (cs) | 0.482 [0.390, 0.573] | 0.559 [0.478, 0.634] |
| ICP + Chamfer (cs) | 0.420 [0.340, 0.496] | 0.489 [0.422, 0.552] |
| NICP su template (cs) | 0.407 [0.322, 0.496] | 0.447 [0.364, 0.527] |
| nostro s1234, d_F non cal. | 0.698 [0.624, 0.766] | 0.751 [0.685, 0.807] |
| nostro s2345, d_F non cal. | 0.694 [0.612, 0.769] | 0.745 [0.678, 0.807] |
| nostro s1234, d_P | 0.688 [0.610, 0.760] | 0.754 [0.688, 0.810] |
| nostro s2345, d_P | 0.680 [0.595, 0.758] | 0.748 [0.680, 0.808] |
| REALY (protocollo, landmark esatti) | 0.612 [0.523, 0.690] | 0.716 [0.644, 0.777] |

Delta appaiati (a - b, rho con la GT; * = primaria):

| GT | modo | tipo | a | b | delta [IC] |
|---|---|---|---|---|---|
| FR | mm | D1: corrispondenze - residuo * | NICP per le corrispondenze (media dei due versi) (mm) | NICP per coppia, residuo P2Tri (matrice) (mm) | +0.251 [+0.199, +0.305] |
| FR | mm | D2: corrispondenze - ICP * | NICP per le corrispondenze (media dei due versi) (mm) | ICP + Chamfer (mm) | +0.172 [+0.139, +0.208] |
| FR | mm | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (mm) | nostro s1234, d_F non cal. | -0.066 [-0.124, -0.011] |
| FR | mm | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (mm) | nostro s2345, d_F non cal. | -0.062 [-0.126, +0.002] |
| FR | mm | corrispondenze - NICP su template | NICP per le corrispondenze (media dei due versi) (mm) | NICP su template (mm) | +0.096 [+0.014, +0.178] |
| FR | mm | corrispondenze - REALY | NICP per le corrispondenze (media dei due versi) (mm) | REALY (protocollo, landmark esatti) | +0.020 [-0.043, +0.091] |
| FR | mm | solo la lettura, un verso | NICP per le corrispondenze, un verso (quello della matrice) (mm) | NICP per coppia, residuo P2Tri (matrice) (mm) | +0.122 [+0.071, +0.171] |
| FR | mm | solo la lettura, due versi | NICP per le corrispondenze (media dei due versi) (mm) | NICP per coppia, residuo P2Tri, media dei due versi (mm) | +0.117 [+0.066, +0.167] |
| FR | cs | D1: corrispondenze - residuo | NICP per le corrispondenze (media dei due versi) (cs) | NICP per coppia, residuo P2Tri (matrice) (cs) | +0.232 [+0.178, +0.289] |
| FR | cs | D2: corrispondenze - ICP | NICP per le corrispondenze (media dei due versi) (cs) | ICP + Chamfer (cs) | +0.166 [+0.131, +0.204] |
| FR | cs | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (cs) | nostro s1234, d_F non cal. | -0.112 [-0.172, -0.052] |
| FR | cs | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (cs) | nostro s2345, d_F non cal. | -0.109 [-0.178, -0.041] |
| FR | cs | corrispondenze - NICP su template | NICP per le corrispondenze (media dei due versi) (cs) | NICP su template (cs) | +0.178 [+0.105, +0.253] |
| FR | cs | corrispondenze - REALY | NICP per le corrispondenze (media dei due versi) (cs) | REALY (protocollo, landmark esatti) | -0.026 [-0.089, +0.042] |
| FR | cs | solo la lettura, un verso | NICP per le corrispondenze, un verso (quello della matrice) (cs) | NICP per coppia, residuo P2Tri (matrice) (cs) | +0.112 [+0.068, +0.157] |
| FR | cs | solo la lettura, due versi | NICP per le corrispondenze (media dei due versi) (cs) | NICP per coppia, residuo P2Tri, media dei due versi (cs) | +0.103 [+0.053, +0.156] |
| SR | mm | D1: corrispondenze - residuo | NICP per le corrispondenze (media dei due versi) (mm) | NICP per coppia, residuo P2Tri (matrice) (mm) | +0.240 [+0.188, +0.291] |
| SR | mm | D2: corrispondenze - ICP | NICP per le corrispondenze (media dei due versi) (mm) | ICP + Chamfer (mm) | +0.166 [+0.131, +0.204] |
| SR | mm | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (mm) | nostro s1234, d_P | -0.097 [-0.152, -0.036] |
| SR | mm | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (mm) | nostro s2345, d_P | -0.091 [-0.146, -0.032] |
| SR | mm | corrispondenze - NICP su template | NICP per le corrispondenze (media dei due versi) (mm) | NICP su template (mm) | +0.265 [+0.201, +0.332] |
| SR | mm | corrispondenze - REALY | NICP per le corrispondenze (media dei due versi) (mm) | REALY (protocollo, landmark esatti) | -0.058 [-0.117, +0.003] |
| SR | mm | solo la lettura, un verso | NICP per le corrispondenze, un verso (quello della matrice) (mm) | NICP per coppia, residuo P2Tri (matrice) (mm) | +0.111 [+0.068, +0.156] |
| SR | mm | solo la lettura, due versi | NICP per le corrispondenze (media dei due versi) (mm) | NICP per coppia, residuo P2Tri, media dei due versi (mm) | +0.095 [+0.046, +0.143] |
| SR | cs | D1: corrispondenze - residuo * | NICP per le corrispondenze (media dei due versi) (cs) | NICP per coppia, residuo P2Tri (matrice) (cs) | +0.269 [+0.216, +0.323] |
| SR | cs | D2: corrispondenze - ICP * | NICP per le corrispondenze (media dei due versi) (cs) | ICP + Chamfer (cs) | +0.189 [+0.153, +0.225] |
| SR | cs | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (cs) | nostro s1234, d_P | -0.076 [-0.131, -0.020] |
| SR | cs | corrispondenze - nostro | NICP per le corrispondenze (media dei due versi) (cs) | nostro s2345, d_P | -0.070 [-0.123, -0.017] |
| SR | cs | corrispondenze - NICP su template | NICP per le corrispondenze (media dei due versi) (cs) | NICP su template (cs) | +0.231 [+0.169, +0.299] |
| SR | cs | corrispondenze - REALY | NICP per le corrispondenze (media dei due versi) (cs) | REALY (protocollo, landmark esatti) | -0.038 [-0.091, +0.020] |
| SR | cs | solo la lettura, un verso | NICP per le corrispondenze, un verso (quello della matrice) (cs) | NICP per coppia, residuo P2Tri (matrice) (cs) | +0.131 [+0.090, +0.173] |
| SR | cs | solo la lettura, due versi | NICP per le corrispondenze (media dei due versi) (cs) | NICP per coppia, residuo P2Tri, media dei due versi (cs) | +0.119 [+0.068, +0.170] |

Verdetto (REGOLE.md sez. 4):

- FR con mm: D1 +0.251 [+0.199, +0.305], D2 +0.172 [+0.139, +0.208]: **sopra l'ICP**
- SR con cs: D1 +0.269 [+0.216, +0.323], D2 +0.189 [+0.153, +0.225]: **sopra l'ICP**

Spearman semplice fra colonne sulle righe della maschera (descrittivo): mm_nicp_corr ~ mm_nicp_p2tri 0.498; mm_nicp_corr ~ mm_rigid_icp_chamfer 0.678; cs_nicp_corr ~ cs_nicp_p2tri 0.527; cs_nicp_corr ~ cs_rigid_icp_chamfer 0.684

<!-- da qui in giu': scritto a mano, conservato da nc.py results -->

## Lettura (scritta a mano dopo i numeri)

**Collaudo**: `nicp_p2tri` uguale alla matrice entro 1e-6 mm su 76/80 coppie ordinate di topologie (vista x modo) nel
calcolo principale (a512-l4-06, nv-ai-02, a768-l40s-05, a768-l40s-06); le 4 fuori tolleranza (hifi3d mm down8k>up60k e
original>up60k, max 1,7e-4; hifi3d cs original>up60k, 5,6e-5; facescape mm up60k>remesh, 2,7e-6) danno 0,0 su TUTTE le
loro coppie ricalcolate sul tipo di nodo d'origine (`controls/check_node_*`: a768-l40s-05, a256-a40-04, i256-a10-07; i
tipi candidati sbagliati, a768-l40s-02/03 e nv-ai-03, restano fuori). NaN sulle stesse coppie (62 e 828).

**Il NICP usato per le corrispondenze non cancella l'identita'.** Nelle letture primarie supera il residuo (D1, FR mm:
+0,196 HIFI3D, +0,251 FaceScape) e supera l'ICP (D2: +0,046 [+0,014; +0,082] HIFI3D FR mm, +0,172 FaceScape FR mm,
+0,189 FaceScape SR cs). L'eccezione formale e' HIFI3D SR cs ("cancella ancora", D1 +0,034 [-0,010; +0,082]): li' il
residuo in cs non era il difetto (0,600, gia' sopra ICP cs 0,585), e `nicp_corr` resta sopra l'ICP (+0,049
[+0,025; +0,074]); l'etichetta viene dalla regola, non da una perdita d'identita'.

**Due ingredienti, non uno** (FR mm): la sola lettura (stesso verso, o entrambi i versi) vale +0,134 / +0,144 su HIFI3D
e +0,122 / +0,117 su FaceScape; la media dei due versi vale +0,05-0,06 su HIFI3D e +0,13 su FaceScape (residuo da 0,499 a
0,551 e da 0,381 a 0,515; corrispondenze da 0,633 a 0,695 e da 0,503 a 0,632). La colonna della matrice ha entrambi gli
svantaggi.

**Contro il nostro**: HIFI3D FR mm `nicp_corr` 0,695 contro d_F non cal. 0,644/0,634 (delta +0,051 [-0,015; +0,115],
non distinguibile); FaceScape il nostro resta sopra (FR mm -0,066 [-0,124; -0,011] con s1234; SR cs -0,076 / -0,070).
In SR mm su HIFI3D `nicp_corr` crolla (0,315): senza taglia nella lettura, come l'ICP mm (0,370).

**Righe**: la maschera toglie le coppie in cui uno dei due NICP e' singolare ("Factor is exactly singular"): HIFI3D
7802/7942, FaceScape 6293/7916 (il verso inverso fallisce su 794 coppie in piu' delle 828 della matrice). Stesse righe
per tutte le colonne.
