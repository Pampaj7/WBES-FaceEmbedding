# Gate G0: l'insegnante ArcFace-render come metrica contro le GT FR/SR, risultati

Protocollo `PROTOCOL_G0.md` (sha256 eee7afef..., commit 0b4a426), scritto prima dei numeri. Insegnante: normal map, camera unica per insieme, yaw 0 / -30 / +30, 512 px, crop fisso calibrato sui render ombreggiati di tutte le mesh dell'insieme (dipendenza transduttiva dalle sole geometrie, come il gate), ArcFace `buffalo_l` `w600k_r50`, media delle 3 viste poi L2, d_T = 1 - coseno. Generato da `aau/diagnostics/g0_stats.py`; numeri in `g0_spearman.csv`, `g0_delta.csv`, `g0/controls.json`, `g0/readings.json`, `g0/cost.json`. Spearman pesato per coppia di mesh, IC 95% percentile, 1000 repliche per soggetto coi semi pubblicati (facescape 796786, hifi3d 990708, faceverse_neutral 271049, flame2023_s1 D1: SeedSequence([20261113, 3]), famos SeedSequence([20261123, 0])).

## Lettura preregistrata (sez. 6)

- **Verdetto: NO** e SOTTO (l'insegnante ordina peggio di d_P in almeno 2 famiglie su 3 in entrambi i semi).
  - factorized_s1234: P = False (Delta_SR >= +0.05 con IC > 0 in 0 famiglie []; FaMoS TRAIN False); IC sotto 0 in ['facescape', 'hifi3d', 'faceverse_neutral']; FLAME (descrittivo) False.
  - factorized_s2345: P = False (Delta_SR >= +0.05 con IC > 0 in 0 famiglie []; FaMoS TRAIN False); IC sotto 0 in ['facescape', 'hifi3d', 'faceverse_neutral']; FLAME (descrittivo) False.

Note del coder, scritte DOPO i numeri (commento, non regole):

- Verdetto preregistrato **NO**, e **SOTTO** in entrambi i semi: l'insegnante ordina le coppie peggio di d_P su tutte le famiglie, con tutti gli IC sotto 0. Delta con SR (s1234 / s2345): FaceScape -0.399 / -0.405, HIFI3D -0.401 / -0.393, FaceVerse neutra -0.167 / -0.206, FLAME -0.461 / -0.461, FaMoS TRAIN (reale) -0.444 / -0.433. Sta anche sotto ICP mm (-0.129, -0.145, -0.217) e sotto B (da -0.216 a -0.506). E' l'esito atteso dal precedente dell'8 ottobre (HIFI3D, GT maxabs: 0.242; qui con SR 0.220).
- **Non e' una pipeline rotta** (controllo di sanita' POST HOC, `g0_sanity.py`, `g0/sanity_posthoc.json`): sulle stesse mesh l'insegnante riconosce le persone su ogni insieme, AUC stessa persona contro persone diverse da 0.969 (FaceVerse neutra) a 0.9999 (FaMoS), rank-1 contro le original da 0.98 a 1.00, FLAME e FaMoS compresi. Le distanze fra persone diverse stanno in una banda stretta e lontana da quelle della stessa persona (HIFI3D: mediana 0.595, IQR 0.544-0.643, contro 0.033 per la stessa persona; FaceScape 0.608, IQR 0.548-0.662): ArcFace separa le identita' ma ordina male le persone diverse per somiglianza di forma, che e' cio' che lo Spearman con FR/SR misura. Riconoscere non e' graduare (diario, 6 ottobre), ora con le GT FR/SR e su cinque famiglie.
- FR (descrittivo): da solo l'insegnante, cieco alla taglia, fa 0.13-0.34. La combinazione con la taglia del braccio (c_T = 0.101 dai soli held-out) arriva a 0.61 su HIFI3D, 0.68 su FLAME, 0.63 su FaMoS, ma resta sotto d_F cal. in ogni insieme e seme (da -0.116 a -0.315, IC sotto 0): quel FR viene in gran parte da S, non da d_T.
- Righe col crop (secondario): l'insegnante resta sotto d_P (da -0.105 a -0.324). L'unico delta positivo di G0 e' insegnante - ICP mm sulle righe col crop di FaceScape, +0.077 [+0.021, +0.141]: e' ICP mm che crolla col crop, non l'insegnante che migliora.
- Conseguenza (sez. 6, "cosa cambierebbe"): l'insegnante ArcFace-render non e' una leva per il benchmark FR/SR e la distillazione verso di esso non ha appoggio; resta una baseline di riconoscimento. G0 non dice nulla su una distillazione con obiettivo di riconoscimento, che e' un'altra domanda.
- Detector della calibrazione: fallisce su tutti i render noisy (come nel gate) e, oltre a noisy, su 497 render BFM su 1.200, 120 FaceVerse, 6 FLAME: il crop di quegli insiemi e' calibrato su meno detection. Non ho verificato se un altro crop cambi lo Spearman; con il riconoscimento intatto (BFM AUC 0.974) e margini da -0.17 a -0.46 non e' plausibile che cambi la lettura.
- Costo: 0.71 s per mesh a un thread (render 0.28 s, ArcFace 0.43 s; up60k 1.06 s), su AMD EPYC 7302; una tantum per insieme circa 130 s di calibrazione con 48 processi (detector su tutti i render ombreggiati) e 2-51 s di camera. Per contesto, non rimisurati e da altri nodi: operatori dei bracci 4.1 s per mesh (campione di D4), "ArcFace migliore" 0.57 s (`indomain_recog`).
- Deviazioni dal protocollo. (1) K3: le tabelle di scala non hanno il numero di vertici; ho confrontato per mesh area grezza e maxabs (scarto relativo 0 su 2.900 mesh) e gli intervalli di vertici per etichetta contro `gen.json` di D1 e D3 (uguali; per gli held-out non c'e' riferimento). (2) Il frame check usa il verdetto del detector piu' il vertice piu' vicino alla camera nel terzo centrale del bbox; su HIFI3D x180 passa 2 soggetti su 4 (le altre rotazioni 0); tutti gli 8 insiemi concordano con l'attesa. (3) Il costo e' misurato su un altro nodo (a256-t4-01) rispetto al job dell'insegnante (i256-a10-09): |Z - teacher.npz| 4.2e-5. (4) K2 2.5e-4, entro 1e-3. (5) Aggiunto dopo i numeri: il controllo di sanita'. (6) Restano 2 png di controllo per insieme in `g0/teacher/*/control`, fuori da git; i render sono stati cancellati con /tmp.

## SR: insegnante contro d_P, ICP mm e B (nocrop_cross; FLAME e FaMoS: righe di D1 / D3)

| insieme | righe | insegnante | d_P s1234 | d_P s2345 | ICP mm | B-GNM vb_sr | B-FLAME vb_sr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| facescape | 88725 | **0.348 [0.284, 0.414]** | 0.747 [0.677, 0.804] | 0.753 [0.687, 0.813] | 0.477 [0.414, 0.533] | 0.854 [0.803, 0.894] | 0.820 [0.757, 0.870] |
| hifi3d | 98224 | **0.220 [0.149, 0.293]** | 0.622 [0.550, 0.685] | 0.613 [0.533, 0.688] | 0.365 [0.280, 0.452] | 0.607 [0.532, 0.679] | 0.597 [0.518, 0.673] |
| faceverse_neutral | 99000 | **0.166 [0.122, 0.213]** | 0.333 [0.251, 0.405] | 0.372 [0.291, 0.444] | 0.383 [0.298, 0.456] | 0.399 [0.315, 0.473] | 0.382 [0.297, 0.459] |
| flame2023_s1 | 398000 | **0.345 [0.298, 0.389]** | 0.806 [0.778, 0.832] | 0.806 [0.776, 0.832] | n.d. | 0.812 [0.781, 0.841] | 0.830 [0.800, 0.858] |
| famos | 63200 | **0.395 [0.306, 0.483]** | 0.839 [0.785, 0.881] | 0.828 [0.757, 0.880] | n.d. | n.d. | n.d. |

Delta appaiati, insegnante - confronto (SR):

| insieme | - d_P s1234 (regola) | - d_P s2345 (regola) | - ICP mm | - B-GNM | - B-FLAME |
| --- | --- | --- | --- | --- | --- |
| facescape | **-0.399 [-0.461, -0.328]** | **-0.405 [-0.473, -0.326]** | -0.129 [-0.189, -0.062] | -0.506 [-0.572, -0.431] | -0.472 [-0.546, -0.392] |
| hifi3d | **-0.401 [-0.498, -0.302]** | **-0.393 [-0.484, -0.293]** | -0.145 [-0.259, -0.033] | -0.387 [-0.478, -0.288] | -0.377 [-0.478, -0.269] |
| faceverse_neutral | **-0.167 [-0.252, -0.074]** | **-0.206 [-0.296, -0.107]** | -0.217 [-0.305, -0.118] | -0.233 [-0.315, -0.140] | -0.216 [-0.311, -0.119] |
| flame2023_s1 | **-0.461 [-0.506, -0.413]** | **-0.461 [-0.503, -0.415]** | n.d. | -0.467 [-0.512, -0.419] | -0.485 [-0.533, -0.434] |
| famos | **-0.444 [-0.518, -0.367]** | **-0.433 [-0.503, -0.357]** | n.d. | n.d. | n.d. |

B su FLAME: B-GNM incrociato, B-FLAME prior esatto (solo riferimento). ICP mm su FLAME e FaMoS e B su FaMoS: non esistono e non si sono calcolati (sez. 3).

## FR (descrittivo): insegnante, d_F calibrata e combinazione insegnante + taglia

c_T = 0.1009 (mediana d_P GT 0.0713 / mediana d_T 0.7068, 297000 coppie degli held-out bfm, ict, gnm; per dominio bfm 0.091, ict 0.095, gnm 0.116).

| insieme | insegnante | d_F cal. s1234 | d_F cal. s2345 | comb. s1234 | comb. s2345 | ICP mm | B-GNM vb_fr | B-FLAME vb_fr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| facescape | 0.300 [0.231, 0.370] | 0.661 [0.586, 0.727] | 0.669 [0.597, 0.733] | 0.346 [0.277, 0.415] | 0.356 [0.287, 0.424] | 0.453 [0.389, 0.514] | 0.678 [0.615, 0.740] | 0.630 [0.555, 0.695] |
| hifi3d | 0.134 [0.042, 0.233] | 0.749 [0.673, 0.806] | 0.731 [0.657, 0.792] | 0.609 [0.509, 0.696] | 0.598 [0.495, 0.684] | 0.643 [0.572, 0.702] | 0.664 [0.570, 0.738] | 0.686 [0.597, 0.758] |
| faceverse_neutral | 0.153 [0.110, 0.198] | 0.341 [0.263, 0.416] | 0.370 [0.298, 0.440] | 0.188 [0.117, 0.263] | 0.184 [0.119, 0.259] | 0.409 [0.326, 0.481] | 0.321 [0.225, 0.400] | 0.315 [0.223, 0.393] |
| flame2023_s1 | 0.223 [0.171, 0.277] | 0.809 [0.777, 0.838] | 0.804 [0.770, 0.836] | 0.680 [0.626, 0.733] | 0.688 [0.635, 0.740] | n.d. | 0.741 [0.697, 0.787] | 0.803 [0.768, 0.838] |
| famos | 0.337 [0.261, 0.419] | 0.824 [0.775, 0.866] | 0.815 [0.761, 0.860] | 0.634 [0.550, 0.719] | 0.637 [0.557, 0.715] | n.d. | n.d. | n.d. |

| insieme | insegnante - d_F cal. s1234 | - d_F cal. s2345 | comb. - d_F cal. s1234 | comb. - d_F cal. s2345 | insegnante - ICP mm | - B-GNM | - B-FLAME |
| --- | --- | --- | --- | --- | --- | --- | --- |
| facescape | -0.360 [-0.432, -0.285] | -0.368 [-0.444, -0.292] | -0.315 [-0.393, -0.239] | -0.313 [-0.398, -0.233] | -0.153 [-0.213, -0.086] | -0.378 [-0.455, -0.294] | -0.330 [-0.409, -0.246] |
| hifi3d | -0.615 [-0.720, -0.498] | -0.597 [-0.699, -0.487] | -0.140 [-0.192, -0.092] | -0.133 [-0.183, -0.087] | -0.509 [-0.612, -0.393] | -0.531 [-0.640, -0.411] | -0.552 [-0.659, -0.432] |
| faceverse_neutral | -0.187 [-0.272, -0.095] | -0.217 [-0.305, -0.129] | -0.153 [-0.227, -0.071] | -0.187 [-0.267, -0.103] | -0.255 [-0.337, -0.163] | -0.168 [-0.260, -0.063] | -0.162 [-0.249, -0.051] |
| flame2023_s1 | -0.585 [-0.645, -0.529] | -0.580 [-0.643, -0.527] | -0.129 [-0.164, -0.098] | -0.116 [-0.150, -0.086] | n.d. | -0.518 [-0.587, -0.455] | -0.579 [-0.640, -0.523] |
| famos | -0.487 [-0.559, -0.405] | -0.478 [-0.552, -0.395] | -0.190 [-0.254, -0.129] | -0.178 [-0.239, -0.119] | n.d. | n.d. | n.d. |

## Righe col crop (secondario; maschera comune)

| vista | righe | insegnante SR | d_P s1234 SR | d_P s2345 SR | insegnante - d_P s1234 | insegnante - d_P s2345 | insegnante FR | d_F cal. s1234 FR | insegnante - d_F cal. s1234 | insegnante - ICP mm (SR) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| facescape | 49500 | 0.371 [0.303, 0.441] | 0.518 [0.434, 0.598] | 0.476 [0.390, 0.559] | -0.147 [-0.229, -0.058] | -0.105 [-0.192, -0.013] | 0.310 [0.237, 0.384] | 0.351 [0.272, 0.431] | -0.041 [-0.131, +0.054] | +0.077 [+0.021, +0.141] |
| hifi3d | 49500 | 0.225 [0.154, 0.300] | 0.550 [0.472, 0.623] | 0.514 [0.435, 0.590] | -0.324 [-0.426, -0.211] | -0.289 [-0.383, -0.176] | 0.139 [0.049, 0.239] | 0.663 [0.590, 0.721] | -0.523 [-0.626, -0.402] | -0.041 [-0.137, +0.056] |
| faceverse_neutral | 49500 | 0.196 [0.145, 0.249] | 0.304 [0.224, 0.382] | 0.316 [0.236, 0.389] | -0.108 [-0.205, -0.009] | -0.120 [-0.216, -0.015] | 0.181 [0.130, 0.233] | 0.282 [0.206, 0.360] | -0.101 [-0.188, -0.009] | -0.103 [-0.184, -0.013] |

## Costo dell'insegnante per mesh (sez. 7)

primi 3 soggetti valutati per id x 6 etichette x facescape, hifi3d, faceverse_neutral, un processo a un thread, in memoria; nodo a256-t4-01.srv.aau.dk (AMD EPYC 7302 16-Core Processor). Controllo: |Z - teacher.npz| massimo 4.2e-05.

| voce | mediana s | p10 s | p90 s |
| --- | --- | --- | --- |
| lettura + 3 normal map | 0.280 | 0.186 | 0.632 |
| 3 ritagli + 3 forward ArcFace + media | 0.431 | 0.431 | 0.437 |
| totale per mesh | 0.712 | 0.621 | 1.063 |

| etichetta | totale mediana s | p90 s |
| --- | --- | --- |
| crop | 0.690 | 0.986 |
| down8k | 0.620 | 0.708 |
| noisy | 0.734 | 1.079 |
| original | 0.708 | 1.001 |
| remesh | 0.651 | 0.849 |
| up60k | 1.058 | 1.755 |

Una tantum per insieme e throughput dei job (secondi di parete, N processi):

| insieme | mesh | processi | camera | render ombreggiati | calibrazione | normal map | embedding | totale | detector fallito (noisy, totale) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| facescape | 600 | 48 | 38 | 7 | 131 | 9 | 20 | 204 | 300/300, 300/1800 |
| hifi3d | 600 | 48 | 35 | 6 | 130 | 8 | 20 | 198 | 300/300, 312/1800 |
| faceverse_neutral | 600 | 48 | 51 | 12 | 131 | 17 | 20 | 231 | 300/300, 420/1800 |
| flame2023_s1 | 1000 | 48 | 4 | 8 | 138 | 10 | 28 | 189 | 598/600, 604/3000 |
| famos | 400 | 48 | 2 | 3 | 128 | 4 | 15 | 152 | 239/240, 240/1200 |
| bfm | 500 | 48 | 6 | 9 | 128 | 12 | 17 | 173 | 300/300, 797/1500 |
| ict | 500 | 48 | 3 | 4 | 128 | 6 | 17 | 159 | 300/300, 300/1500 |
| gnm | 500 | 48 | 3 | 5 | 129 | 7 | 18 | 161 | 300/300, 310/1500 |

## Il pilota della distillazione e il suo insegnante (DESCRITTIVO)

Riconoscimento d'identita', rank-1 sulle 5 topologie senza crop, da `aau/runs/distill_pilot/summary.md` (7 ottobre, protocollo `distill_pilot/protocol.md`), copiati come sono: non sono distanze graduate e non entrano nella lettura di G0.

| dominio | studente, conv. BFM (riferimento) | studente, conv. ICT | insegnante, normal map 3 viste | insegnante, ombreggiato | congiunto BFM+ICT | studente / insegnante (conv. BFM) |
| --- | --- | --- | --- | --- | --- | --- |
| FaceVerse con espressioni | 0.332 [0.299, 0.366] | 0.570 [0.520, 0.618] | 0.867 [0.843, 0.889] | 0.750 [0.724, 0.775] | 0.680 [0.634, 0.723] | 0.383 [0.344, 0.422] |
| HIFI3D neutre | 0.224 [0.207, 0.244] | 0.355 [0.322, 0.392] | 0.998 [0.995, 1.000] | 0.984 [0.974, 0.992] | 0.397 [0.363, 0.434] | 0.224 [0.208, 0.245] |

## Controlli

- K1 riferimenti (bracci e B = `d3_spearman.csv` / `d1_spearman.csv`, ICP mm = `factorized_paired.csv`): True, 40 celle, scarto massimo 3.3e-16.
- K2 catena dell'insegnante su HIFI3D contro il gate: True (soggetti uguali True, |E - gate| massimo 2.5e-04).
- K3 mesh rigenerate: flame2023_s1 True (scarto relativo massimo 0.0e+00, up60k accettate 0); famos True (scarto relativo massimo 0.0e+00, up60k accettate 0); heldout True (scarto relativo massimo 0.0e+00, up60k accettate 0).
- K4 embedding: facescape 600, hifi3d 600, faceverse_neutral 600, flame2023_s1 1000, famos 400, bfm 500, ict 500, gnm 500 mesh, norme entro 1.1e-07.
- K5 (insegnante finto = d_P s1234, `g0_test.py`, `g0/test/controls.json`): True, scarto 0.0e+00 su 16 celle; K1 del test True.
- K6 c dei bracci ricalcolata come c_T: True (factorized_s1234 0.4053 contro 0.4053, factorized_s2345 0.4054 contro 0.4054).
- Frame check: facescape x180 (attesa x180), hifi3d x180 (attesa x180), faceverse_neutral none (attesa none), flame2023_s1 x180 (attesa x180), famos x180 (attesa x180), bfm none (attesa none), ict x180 (attesa x180), gnm x180 (attesa x180).

**Licenza.** I pesi di insightface (`buffalo_l`: detector e `w600k_r50`) sono distribuiti per la sola ricerca non commerciale; uno studente distillato da loro ne erediterebbe probabilmente il vincolo.
