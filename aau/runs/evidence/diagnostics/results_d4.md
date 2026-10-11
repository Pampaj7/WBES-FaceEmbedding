# Diagnostica D4: ridiscretizzazione canonica al test, risultati

Protocollo `PROTOCOL_D4.md` (sha256 77a93f61..., commit 5fd3061), scritto prima dei numeri. Remesh isotropico esplicito di pymeshlab 2025.7.post1 a lato 2.0 mm su tutte le 6 topologie dei 100 soggetti valutati; tabella di scala con l'area della mesh canonica, operatori areanorm k 128, embedding degli store. Generato da `aau/diagnostics/d4_stats.py`; numeri in `d4_spearman.csv`, `d4_delta.csv`, `d4/controls.json`, `d4/readings.json`, `d4/remesh_stats.json`, `d4/cost.json`. IC 95% percentile, 1000 repliche per soggetto, semi del gruppo primario ufficiale (facescape 796786, hifi3d 990708, faceverse_neutral 271049).

## Lettura preregistrata (sez. 5)

- **Verdetto: NO** (domini validi per la regola: facescape, hifi3d, faceverse_neutral).
  - factorized_s1234: P = False (d_P con SR >= +0.02 e IC > 0 in 0 domini []; d_F cal. con FR >= -0.02 ovunque: True); dannosa in 0 domini []; facescape SR +0.002 [-0.005, +0.010], FR +0.010 [+0.002, +0.018]; hifi3d SR +0.000 [-0.007, +0.007], FR +0.001 [-0.003, +0.004]; faceverse_neutral SR -0.000 [-0.009, +0.009], FR -0.000 [-0.009, +0.009]
  - factorized_s2345: P = False (d_P con SR >= +0.02 e IC > 0 in 0 domini []; d_F cal. con FR >= -0.02 ovunque: True); dannosa in 0 domini []; facescape SR +0.011 [+0.005, +0.019], FR +0.014 [+0.008, +0.022]; hifi3d SR -0.000 [-0.004, +0.004], FR +0.001 [-0.002, +0.003]; faceverse_neutral SR +0.001 [-0.006, +0.008], FR -0.000 [-0.006, +0.006]

## Lettura corretta dopo il critic (post hoc)

Critic: **RISERVE**. Il NO preregistrato e' vero e non nasce da un difetto del codice: il critic ha riverificato pipeline, scala, normali (coseno della normale media canonica / ingresso 1.000 su tutte le etichette) e righe, e il suo Spearman ricalcolato dagli npz su nocrop_cross rida' i nostri valori (FaceScape s1234 originale 0.7469, canonico 0.7494). Va corretta la lettura del meccanismo. I numeri marcati "critic, post hoc" vengono dai suoi script (`a_stats.py`, `b_geom.py`), copiati con le uscite in `/home/create.aau.dk/ga41wf/critic_scratch_657e6a75/d4c` (fuori dal repo, non versionati); non li ho rieseguiti, coincidono coi nostri dove si sovrappongono (nocrop_cross, stessa topologia, residuo sulle 5 etichette, diagonale per etichetta).

1. **Falso** (nota 2 della versione e45a5dc): "una ridiscretizzazione al test non tocca il gap" e "la sfaccettatura di down8k viene conservata". Il residuo stessa etichetta - incrociate calcolato SOLO sulle etichette che cambiano la discretizzazione (original, down8k, up60k: superficie quasi identica alla original, sotto) va quasi a zero dopo la canonizzazione (d_P con SR, critic, post hoc):

| dominio | residuo prima, s1234 | prima, s2345 | dopo, s1234 | dopo, s2345 |
| --- | --- | --- | --- | --- |
| FaceScape | +0.026 [+0.020, +0.034] | +0.006 [+0.004, +0.008] | +0.003 [+0.001, +0.005] | +0.000 [-0.000, +0.002] |
| HIFI3D | +0.002 [+0.001, +0.004] | +0.002 [+0.001, +0.003] | -0.000 [-0.000, +0.000] | -0.000 [-0.000, +0.000] |
| FaceVerse neutra | +0.003 [+0.001, +0.006] | +0.003 [+0.001, +0.004] | +0.000 [-0.000, +0.000] | +0.000 [-0.000, +0.000] |

   Sulle righe che contengono noisy o remesh il residuo resta: FaceScape +0.050 -> +0.046, HIFI3D +0.014 -> +0.012, FaceVerse +0.022 -> +0.016 (s1234; s2345 +0.036 -> +0.040, +0.005 -> +0.007, +0.014 -> +0.014). Sulle sole righe incrociate fra original, down8k e up60k FaceScape guadagna +0.015 [+0.006, +0.024] (s1234) e +0.007 [+0.003, +0.011] (s2345); HIFI3D +0.007 [+0.000, +0.013] e +0.001, FaceVerse +0.005 e +0.002 con IC che toccano lo 0. Su queste etichette il residuo si chiude in parte perche' sale l'incrociato e in parte perche' scende la stessa etichetta (up60k-up60k 0.802 -> 0.777 su FaceScape s1234).
2. **Su noisy D4 non separa superficie e discretizzazione.** Con featuredeg = 30 (`d4_remesh.canonical`) gli spigoli del rumore superano la soglia di diedro e restano come spigoli vivi: sono invariati il 95.6%, 90.4% e 98.1% dei vertici d'ingresso noisy (FaceScape, HIFI3D, FaceVerse; 2-13% per le altre etichette) e il 45-72% dei vertici della canonica noisy sono vertici d'ingresso; diedro medio della canonica 41-68 gradi per noisy contro 3-7.5 per original (critic, post hoc). Lo mostrava gia' la nostra Hausdorff ingresso -> canonica su noisy, 0.003-0.012 mm: i vertici d'ingresso stanno sulla canonica perche' ci sono ancora.
3. **Il NO preregistrato non dice nulla sul meccanismo.** Il tetto plausibile del delta su nocrop_cross era portare le righe incrociate al livello della migliore cella con la stessa etichetta di D3: per s2345 +0.019 su HIFI3D (remesh-remesh 0.632 contro 0.613) e +0.017 su FaceVerse (noisy-noisy 0.389 contro 0.372), sotto +0.02, quindi P(s2345) e il verdetto UTILE erano irraggiungibili per costruzione. Il protocollo (sez. 5) diceva "esigente", non "irraggiungibile": va dichiarato. Il +0.053 di partenza (D3) era la cella remesh-remesh, cioe' lo stesso smoothing ai due lati, non la stessa discretizzazione.
4. **Geometria** (critic, post hoc): distanza media simmetrica in mm dalla original dello stesso soggetto, mediana su 10 soggetti per vista, ingresso -> canonica (contro la canonica della original):

| dominio | noisy | remesh | down8k | up60k |
| --- | --- | --- | --- | --- |
| FaceScape | 0.435 -> 0.459 | 0.124 -> 0.241 | 0.016 -> 0.014 | 0.000 -> 0.014 |
| HIFI3D | 0.419 -> 0.413 | 0.137 -> 0.204 | 0.019 -> 0.018 | 0.000 -> 0.015 |
| FaceVerse neutra | 0.397 -> 0.382 | 0.060 -> 0.080 | 0.011 -> 0.022 | 0.000 -> 0.019 |

   noisy e remesh cambiano la superficie (0.06-0.44 mm) e il remesh canonico la conserva o la allontana (remesh 0.124 -> 0.241 su FaceScape); down8k e up60k cambiano quasi solo la discretizzazione (<= 0.02 mm).
5. **Nome.** Ne' "gap di discretizzazione" (D3) ne' "gap di vista" (nota 2 sotto): **sensibilita' a perturbazioni di superficie (rumore, smoothing)**. Affermazione positiva: sulle viste che cambiano solo la discretizzazione il residuo e' <= 0.006 (FaceScape s1234 0.026) e il remesh canonico lo azzera (<= 0.003). Leva al test non ancora provata: un filtro di superficie (smoothing o denoising canonico prima degli operatori); il suo tetto e' chiudere il residuo sulle 5 etichette, FaceScape +0.042 / +0.029, HIFI3D +0.013 / +0.007, FaceVerse +0.014 / +0.008 (d_P con SR, s1234 / s2345, tabella del residuo sotto).

Note del coder, scritte DOPO i numeri (commento, non regole; la nota 2 e' corretta dalla sezione sopra):

- Verdetto preregistrato **NO** in entrambi i semi, non dannosa: nessun dominio arriva a +0.02 con d_P contro SR su nocrop_cross. Il delta piu' grande e' FaceScape s2345 +0.011 [+0.005, +0.019] (s1234 +0.002 [-0.005, +0.010]); HIFI3D +0.000 / -0.000 e FaceVerse neutra -0.000 / +0.001, con IC entro +-0.009. FR non peggiora (da -0.000 a +0.014; FaceScape +0.010 / +0.014 con IC sopra 0). Sulle righe di D3 (sensibilita') la lettura e' la stessa.
- Il residuo stessa topologia - nocrop_cross sulle 5 etichette insieme cala poco: d_P con SR, FaceScape +0.042 -> +0.033 (s1234) e +0.029 -> +0.027 (s2345), HIFI3D +0.013 -> +0.014 e +0.007 -> +0.007, FaceVerse +0.014 -> +0.011 e +0.008 -> +0.007; frazione recuperata da -0.04 a 0.39 per factorized. **LETTO MALE** (critic, post hoc; sezione sopra): nella versione e45a5dc ne deducevo che il remesh canonico conserva la superficie di tutte le etichette, sfaccettatura di down8k compresa, che "una ridiscretizzazione al test non tocca" il gap e che andava chiamato "gap di vista". E' FALSO: sulle etichette che cambiano solo la discretizzazione (original, down8k, up60k) il residuo va quasi a zero; resta sulle righe con noisy e remesh (superficie diversa: rumore, smoothing), e su noisy D4 non ridiscretizza (featuredeg = 30).
- Le mesh sono cambiate davvero e l'embedding si e' mosso (tabella dello spostamento, descrittiva): spostamento canonico - originale mediano 0.06-0.13 in u per factorized (9-24% della distanza mediana fra soggetti); la distanza fra la original e le altre topologie dello STESSO soggetto scende su HIFI3D (0.092 -> 0.061, s1234) e FaceVerse (0.107 -> 0.054), quasi invariata su FaceScape (0.154 -> 0.146). La canonizzazione avvicina le topologie dello stesso soggetto ma, sulle 5 etichette insieme, non migliora l'ordinamento fra soggetti diversi, che e' cio' che lo Spearman misura (sulle sole righe incrociate original / down8k / up60k di FaceScape si', +0.015 con s1234: critic, sezione sopra); l'errore dominante non e' la dispersione fra topologie (coerente con D3: il limite e' il generatore).
- Per etichetta (descrittivo): FaceScape up60k-up60k scende (0.802 -> 0.777, s1234; la densa da 32k vertici perde il vantaggio della discretizzazione condivisa), noisy-noisy sale per s2345 (0.767 -> 0.805) e cala per s1234 (0.791 -> 0.786); su HIFI3D e FaceVerse le etichette di factorized si muovono entro +-0.015.
- ctrlfr s1234 (secondario): piccole perdite su nocrop_cross, FaceScape FR -0.012 [-0.024, -0.000], HIFI3D FR -0.006 [-0.012, -0.000] e SR -0.009 [-0.015, -0.003], FaceVerse 0. Righe col crop e all_cross: da -0.007 a +0.024; il remesh non tocca il crop (e' parzialita', non discretizzazione), come atteso.
- Costo: remesh mediana 5.0 s per mesh su tutte le 1.800 (p90 21.6 s; noisy 11-30 s, perche' il rumore gonfia l'area e la mesh canonica ha 1.8-3.5 volte i vertici delle altre); sul campione di 54 la pipeline canonica (remesh + operatori) costa 9.1 s per mesh contro 4.1 s degli operatori sull'originale, rapporto mediano 2.45 (p10 0.90 dove l'originale e' densa: FaceVerse 9.2 contro 10.1 s; p90 7.3 su noisy e down8k). Embedding di HIFI3D 82 s contro 75 s per 600 mesh (stessa A100). Con guadagno nullo il costo non si giustifica.
- Il precedente E3b (rimesh ACVD al test per e108, senza operatori areanorm: -0.174 su HIFI3D) non si ripete: per i bracci C3F il remesh canonico e' neutro, non dannoso. Perche' (operatori, remesher, modello) non e' separato qui.
- Deviazioni e aggiunte. (1) K1: il protocollo scriveva i conteggi di D3 per nocrop_cross (88.725 / 98.224 / 99.000), ma la maschera di D4 (bracci e GT, sez. 3) tiene 99.000 righe su tutte e tre le viste perche' non ha le colonne delle baseline geometriche, che in D3 e fact_paired tolgono 10.275 righe a FaceScape e 776 a HIFI3D. Il primario resta sulle 99.000 righe della sez. 3; K1 si legge sulle righe comuni (clausola della sez. 4, scritta per il caso opposto) e passa con scarto 0; il gruppo "righe di D3" e' una sensibilita' aggiunta dopo la prova su dati finti (`d4_test.py`, che ha mostrato i conteggi diversi) e PRIMA dei numeri canonici. (2) Post hoc, descrittivi: Hausdorff ingresso -> canonica e tabella dello spostamento nell'embedding. (3) Il campione del costo e' stato misurato su un nodo condiviso (a256-t4-02, 4 core chiesti, un thread usato), il remesh di tutte le 1.800 con 63 processi in parallelo (i256-a10-10): i tempi includono la contesa del nodo.

## Delta canonico - originale (stesse righe e repliche)

| dominio | braccio | distanza | GT | nocrop_cross (primario) | stessa topologia | all_cross | righe col crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| facescape | factorized_s1234 | d_P (primaria) | SR | **+0.002 [-0.005, +0.010]** | -0.007 [-0.014, -0.000] | +0.005 [+0.001, +0.010] | +0.008 [+0.000, +0.016] |
| facescape | factorized_s1234 | d_F cal. (vincolo) | FR | +0.010 [+0.002, +0.018] | -0.004 [-0.011, +0.003] | +0.003 [-0.001, +0.007] | -0.007 [-0.013, -0.002] |
| facescape | factorized_s2345 | d_P (primaria) | SR | **+0.011 [+0.005, +0.019]** | +0.009 [+0.003, +0.016] | +0.014 [+0.010, +0.018] | +0.015 [+0.008, +0.023] |
| facescape | factorized_s2345 | d_F cal. (vincolo) | FR | +0.014 [+0.008, +0.022] | +0.005 [-0.002, +0.011] | +0.006 [+0.002, +0.010] | -0.000 [-0.009, +0.007] |
| facescape | ctrlfr_s1234 | ||z|| (secondaria) | FR | -0.012 [-0.024, -0.000] | +0.002 [-0.006, +0.011] | +0.010 [+0.003, +0.017] | +0.015 [-0.005, +0.032] |
| facescape | ctrlfr_s1234 | ||z|| (secondaria) | SR | -0.001 [-0.012, +0.009] | +0.011 [+0.003, +0.018] | +0.017 [+0.011, +0.023] | +0.024 [+0.007, +0.039] |
| hifi3d | factorized_s1234 | d_P (primaria) | SR | **+0.000 [-0.007, +0.007]** | +0.001 [-0.005, +0.007] | +0.006 [+0.001, +0.011] | +0.005 [-0.000, +0.010] |
| hifi3d | factorized_s1234 | d_F cal. (vincolo) | FR | +0.001 [-0.003, +0.004] | +0.002 [-0.002, +0.005] | +0.001 [-0.002, +0.004] | -0.003 [-0.006, +0.000] |
| hifi3d | factorized_s2345 | d_P (primaria) | SR | **-0.000 [-0.004, +0.004]** | +0.000 [-0.004, +0.004] | +0.000 [-0.004, +0.004] | -0.002 [-0.007, +0.002] |
| hifi3d | factorized_s2345 | d_F cal. (vincolo) | FR | +0.001 [-0.002, +0.003] | +0.000 [-0.002, +0.002] | +0.002 [-0.000, +0.004] | +0.001 [-0.001, +0.004] |
| hifi3d | ctrlfr_s1234 | ||z|| (secondaria) | FR | -0.006 [-0.012, -0.000] | +0.006 [+0.001, +0.013] | -0.000 [-0.006, +0.006] | +0.002 [-0.008, +0.012] |
| hifi3d | ctrlfr_s1234 | ||z|| (secondaria) | SR | -0.009 [-0.015, -0.003] | -0.007 [-0.013, -0.001] | -0.007 [-0.012, -0.002] | -0.006 [-0.016, +0.004] |
| faceverse_neutral | factorized_s1234 | d_P (primaria) | SR | **-0.000 [-0.009, +0.009]** | -0.003 [-0.012, +0.006] | -0.001 [-0.009, +0.008] | -0.001 [-0.010, +0.008] |
| faceverse_neutral | factorized_s1234 | d_F cal. (vincolo) | FR | -0.000 [-0.009, +0.009] | -0.002 [-0.010, +0.008] | -0.000 [-0.008, +0.008] | +0.001 [-0.007, +0.009] |
| faceverse_neutral | factorized_s2345 | d_P (primaria) | SR | **+0.001 [-0.006, +0.008]** | -0.000 [-0.008, +0.007] | +0.001 [-0.006, +0.008] | +0.001 [-0.006, +0.008] |
| faceverse_neutral | factorized_s2345 | d_F cal. (vincolo) | FR | -0.000 [-0.006, +0.006] | +0.002 [-0.004, +0.008] | +0.000 [-0.005, +0.006] | -0.000 [-0.006, +0.005] |
| faceverse_neutral | ctrlfr_s1234 | ||z|| (secondaria) | FR | -0.000 [-0.008, +0.006] | -0.001 [-0.009, +0.006] | +0.001 [-0.006, +0.007] | +0.002 [-0.006, +0.010] |
| faceverse_neutral | ctrlfr_s1234 | ||z|| (secondaria) | SR | +0.001 [-0.006, +0.008] | +0.001 [-0.006, +0.008] | +0.002 [-0.005, +0.008] | +0.002 [-0.006, +0.009] |

Sensibilita' (descrittiva, non nel protocollo): nocrop_cross ristretto alle righe di D3 e di fact_paired (la loro maschera toglie le righe con baseline geometriche non finite), delta canonico - originale:

| dominio | braccio | distanza | GT | righe | delta |
| --- | --- | --- | --- | --- | --- |
| facescape | factorized_s1234 | d_P | SR | 88725 | +0.003 [-0.006, +0.011] |
| facescape | factorized_s1234 | d_F cal. | FR | 88725 | +0.009 [+0.000, +0.017] |
| facescape | factorized_s2345 | d_P | SR | 88725 | +0.012 [+0.005, +0.020] |
| facescape | factorized_s2345 | d_F cal. | FR | 88725 | +0.012 [+0.006, +0.020] |
| facescape | ctrlfr_s1234 | ||z|| | FR | 88725 | -0.014 [-0.026, -0.002] |
| facescape | ctrlfr_s1234 | ||z|| | SR | 88725 | -0.003 [-0.015, +0.007] |
| hifi3d | factorized_s1234 | d_P | SR | 98224 | +0.000 [-0.007, +0.007] |
| hifi3d | factorized_s1234 | d_F cal. | FR | 98224 | +0.001 [-0.003, +0.004] |
| hifi3d | factorized_s2345 | d_P | SR | 98224 | -0.000 [-0.004, +0.004] |
| hifi3d | factorized_s2345 | d_F cal. | FR | 98224 | +0.001 [-0.002, +0.003] |
| hifi3d | ctrlfr_s1234 | ||z|| | FR | 98224 | -0.006 [-0.012, -0.000] |
| hifi3d | ctrlfr_s1234 | ||z|| | SR | 98224 | -0.009 [-0.015, -0.003] |
| faceverse_neutral | factorized_s1234 | d_P | SR | 99000 | -0.000 [-0.009, +0.009] |
| faceverse_neutral | factorized_s1234 | d_F cal. | FR | 99000 | -0.000 [-0.009, +0.009] |
| faceverse_neutral | factorized_s2345 | d_P | SR | 99000 | +0.001 [-0.006, +0.008] |
| faceverse_neutral | factorized_s2345 | d_F cal. | FR | 99000 | -0.000 [-0.006, +0.006] |
| faceverse_neutral | ctrlfr_s1234 | ||z|| | FR | 99000 | -0.000 [-0.008, +0.006] |
| faceverse_neutral | ctrlfr_s1234 | ||z|| | SR | 99000 | +0.001 [-0.006, +0.008] |

Valori (rho, originale -> canonico):

| dominio | braccio | distanza | GT | nocrop_cross | stessa topologia | all_cross | righe col crop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| facescape | factorized_s1234 | d_P | SR | 0.747 -> 0.749 | 0.789 -> 0.782 | 0.489 -> 0.494 | 0.518 -> 0.526 |
| facescape | factorized_s1234 | d_F cal. | FR | 0.659 -> 0.669 | 0.705 -> 0.701 | 0.370 -> 0.373 | 0.351 -> 0.344 |
| facescape | factorized_s2345 | d_P | SR | 0.754 -> 0.765 | 0.783 -> 0.792 | 0.464 -> 0.478 | 0.476 -> 0.492 |
| facescape | factorized_s2345 | d_F cal. | FR | 0.667 -> 0.682 | 0.714 -> 0.718 | 0.351 -> 0.358 | 0.278 -> 0.278 |
| facescape | ctrlfr_s1234 | ||z|| | FR | 0.663 -> 0.650 | 0.757 -> 0.760 | 0.334 -> 0.344 | 0.264 -> 0.278 |
| facescape | ctrlfr_s1234 | ||z|| | SR | 0.621 -> 0.619 | 0.707 -> 0.717 | 0.315 -> 0.332 | 0.247 -> 0.271 |
| hifi3d | factorized_s1234 | d_P | SR | 0.622 -> 0.622 | 0.634 -> 0.635 | 0.557 -> 0.563 | 0.550 -> 0.554 |
| hifi3d | factorized_s1234 | d_F cal. | FR | 0.749 -> 0.749 | 0.760 -> 0.761 | 0.709 -> 0.710 | 0.663 -> 0.660 |
| hifi3d | factorized_s2345 | d_P | SR | 0.613 -> 0.612 | 0.620 -> 0.620 | 0.540 -> 0.540 | 0.514 -> 0.512 |
| hifi3d | factorized_s2345 | d_F cal. | FR | 0.731 -> 0.731 | 0.744 -> 0.744 | 0.661 -> 0.663 | 0.552 -> 0.554 |
| hifi3d | ctrlfr_s1234 | ||z|| | FR | 0.757 -> 0.751 | 0.779 -> 0.785 | 0.516 -> 0.515 | 0.236 -> 0.238 |
| hifi3d | ctrlfr_s1234 | ||z|| | SR | 0.339 -> 0.330 | 0.347 -> 0.340 | 0.244 -> 0.237 | 0.164 -> 0.158 |
| faceverse_neutral | factorized_s1234 | d_P | SR | 0.333 -> 0.333 | 0.347 -> 0.344 | 0.323 -> 0.322 | 0.304 -> 0.303 |
| faceverse_neutral | factorized_s1234 | d_F cal. | FR | 0.341 -> 0.340 | 0.352 -> 0.350 | 0.319 -> 0.319 | 0.282 -> 0.283 |
| faceverse_neutral | factorized_s2345 | d_P | SR | 0.372 -> 0.373 | 0.380 -> 0.380 | 0.352 -> 0.353 | 0.316 -> 0.317 |
| faceverse_neutral | factorized_s2345 | d_F cal. | FR | 0.370 -> 0.370 | 0.380 -> 0.382 | 0.335 -> 0.335 | 0.272 -> 0.272 |
| faceverse_neutral | ctrlfr_s1234 | ||z|| | FR | 0.327 -> 0.327 | 0.360 -> 0.359 | 0.316 -> 0.317 | 0.296 -> 0.298 |
| faceverse_neutral | ctrlfr_s1234 | ||z|| | SR | 0.326 -> 0.328 | 0.356 -> 0.357 | 0.314 -> 0.315 | 0.291 -> 0.292 |

Spostamento nell'embedding (descrittivo, post hoc, non nel protocollo): mediane di ||z canonico - z originale|| della stessa mesh, della distanza fra soggetti (original contro original) e della distanza fra la original e le altre 4 topologie senza crop dello stesso soggetto; u per factorized, z per ctrlfr.

| dominio | braccio | spostamento canonico - originale | fra soggetti, originale -> canonico | stesso soggetto fra topologie, originale -> canonico |
| --- | --- | --- | --- | --- |
| facescape | factorized_s1234 | 0.130 | 0.547 -> 0.548 | 0.154 -> 0.146 |
| facescape | factorized_s2345 | 0.084 | 0.547 -> 0.545 | 0.106 -> 0.111 |
| facescape | ctrlfr_s1234 | 0.307 | 0.794 -> 0.792 | 0.241 -> 0.232 |
| hifi3d | factorized_s1234 | 0.085 | 0.640 -> 0.644 | 0.092 -> 0.061 |
| hifi3d | factorized_s2345 | 0.060 | 0.643 -> 0.642 | 0.081 -> 0.052 |
| hifi3d | ctrlfr_s1234 | 0.277 | 1.046 -> 1.050 | 0.196 -> 0.166 |
| faceverse_neutral | factorized_s1234 | 0.093 | 0.906 -> 0.910 | 0.107 -> 0.054 |
| faceverse_neutral | factorized_s2345 | 0.082 | 0.907 -> 0.908 | 0.101 -> 0.048 |
| faceverse_neutral | ctrlfr_s1234 | 0.145 | 1.245 -> 1.248 | 0.195 -> 0.114 |

## Residuo stessa topologia - nocrop_cross, prima e dopo

Frazione recuperata = delta canonico - originale su nocrop_cross / residuo originale (solo dove il residuo originale ha IC sopra 0; descrittiva).

| dominio | braccio | distanza | GT | residuo originale | residuo canonico | canonico - originale | frazione recuperata |
| --- | --- | --- | --- | --- | --- | --- | --- |
| facescape | factorized_s1234 | d_P | SR | +0.042 [+0.034, +0.051] | +0.033 [+0.026, +0.041] | -0.009 [-0.014, -0.005] | 0.06 |
| facescape | factorized_s1234 | d_F cal. | FR | +0.046 [+0.038, +0.056] | +0.032 [+0.025, +0.041] | -0.014 [-0.019, -0.009] | 0.21 |
| facescape | factorized_s2345 | d_P | SR | +0.029 [+0.023, +0.038] | +0.027 [+0.021, +0.034] | -0.003 [-0.006, +0.001] | 0.39 |
| facescape | factorized_s2345 | d_F cal. | FR | +0.046 [+0.038, +0.056] | +0.037 [+0.029, +0.046] | -0.010 [-0.013, -0.006] | 0.31 |
| facescape | ctrlfr_s1234 | ||z|| | FR | +0.095 [+0.078, +0.113] | +0.109 [+0.092, +0.129] | +0.015 [+0.007, +0.023] | -0.13 |
| facescape | ctrlfr_s1234 | ||z|| | SR | +0.086 [+0.071, +0.104] | +0.098 [+0.083, +0.116] | +0.012 [+0.005, +0.019] | -0.02 |
| hifi3d | factorized_s1234 | d_P | SR | +0.013 [+0.010, +0.016] | +0.014 [+0.009, +0.019] | +0.001 [-0.003, +0.005] | 0.01 |
| hifi3d | factorized_s1234 | d_F cal. | FR | +0.011 [+0.009, +0.014] | +0.012 [+0.009, +0.015] | +0.001 [-0.001, +0.002] | 0.08 |
| hifi3d | factorized_s2345 | d_P | SR | +0.007 [+0.005, +0.009] | +0.007 [+0.005, +0.010] | +0.000 [-0.001, +0.002] | -0.04 |
| hifi3d | factorized_s2345 | d_F cal. | FR | +0.014 [+0.011, +0.016] | +0.013 [+0.011, +0.016] | -0.001 [-0.001, +0.000] | 0.04 |
| hifi3d | ctrlfr_s1234 | ||z|| | FR | +0.022 [+0.018, +0.027] | +0.035 [+0.030, +0.041] | +0.013 [+0.010, +0.016] | -0.29 |
| hifi3d | ctrlfr_s1234 | ||z|| | SR | +0.009 [+0.005, +0.012] | +0.011 [+0.006, +0.016] | +0.002 [-0.001, +0.005] | -1.07 |
| faceverse_neutral | factorized_s1234 | d_P | SR | +0.014 [+0.009, +0.019] | +0.011 [+0.006, +0.017] | -0.003 [-0.005, +0.000] | -0.04 |
| faceverse_neutral | factorized_s1234 | d_F cal. | FR | +0.011 [+0.007, +0.015] | +0.010 [+0.005, +0.015] | -0.001 [-0.004, +0.001] | -0.04 |
| faceverse_neutral | factorized_s2345 | d_P | SR | +0.008 [+0.006, +0.011] | +0.007 [+0.004, +0.010] | -0.001 [-0.003, +0.001] | 0.13 |
| faceverse_neutral | factorized_s2345 | d_F cal. | FR | +0.009 [+0.007, +0.012] | +0.011 [+0.008, +0.015] | +0.002 [+0.000, +0.004] | -0.00 |
| faceverse_neutral | ctrlfr_s1234 | ||z|| | FR | +0.033 [+0.023, +0.044] | +0.032 [+0.021, +0.045] | -0.001 [-0.005, +0.004] | -0.01 |
| faceverse_neutral | ctrlfr_s1234 | ||z|| | SR | +0.030 [+0.020, +0.041] | +0.030 [+0.018, +0.042] | -0.000 [-0.005, +0.004] | 0.05 |

Stessa topologia per etichetta (rho originale -> canonico):

| dominio | braccio | distanza | GT | down8k | noisy | original | remesh | up60k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| facescape | factorized_s1234 | d_P | SR | 0.779 -> 0.776 | 0.791 -> 0.786 | 0.778 -> 0.776 | 0.800 -> 0.798 | 0.802 -> 0.777 |
| facescape | factorized_s1234 | d_F cal. | FR | 0.691 -> 0.694 | 0.707 -> 0.710 | 0.694 -> 0.694 | 0.716 -> 0.716 | 0.724 -> 0.695 |
| facescape | factorized_s2345 | d_P | SR | 0.781 -> 0.780 | 0.767 -> 0.805 | 0.778 -> 0.788 | 0.803 -> 0.803 | 0.789 -> 0.785 |
| facescape | factorized_s2345 | d_F cal. | FR | 0.706 -> 0.707 | 0.702 -> 0.727 | 0.706 -> 0.713 | 0.736 -> 0.734 | 0.720 -> 0.711 |
| facescape | ctrlfr_s1234 | ||z|| | FR | 0.752 -> 0.756 | 0.732 -> 0.748 | 0.762 -> 0.760 | 0.780 -> 0.782 | 0.766 -> 0.755 |
| facescape | ctrlfr_s1234 | ||z|| | SR | 0.708 -> 0.713 | 0.672 -> 0.708 | 0.713 -> 0.717 | 0.728 -> 0.739 | 0.719 -> 0.713 |
| hifi3d | factorized_s1234 | d_P | SR | 0.635 -> 0.642 | 0.619 -> 0.610 | 0.642 -> 0.642 | 0.642 -> 0.643 | 0.636 -> 0.640 |
| hifi3d | factorized_s1234 | d_F cal. | FR | 0.758 -> 0.764 | 0.754 -> 0.752 | 0.763 -> 0.763 | 0.764 -> 0.764 | 0.759 -> 0.763 |
| hifi3d | factorized_s2345 | d_P | SR | 0.624 -> 0.623 | 0.597 -> 0.601 | 0.624 -> 0.623 | 0.632 -> 0.630 | 0.624 -> 0.623 |
| hifi3d | factorized_s2345 | d_F cal. | FR | 0.748 -> 0.745 | 0.735 -> 0.739 | 0.746 -> 0.746 | 0.748 -> 0.748 | 0.745 -> 0.745 |
| hifi3d | ctrlfr_s1234 | ||z|| | FR | 0.779 -> 0.793 | 0.768 -> 0.770 | 0.781 -> 0.790 | 0.788 -> 0.785 | 0.781 -> 0.791 |
| hifi3d | ctrlfr_s1234 | ||z|| | SR | 0.350 -> 0.349 | 0.336 -> 0.324 | 0.352 -> 0.345 | 0.356 -> 0.335 | 0.344 -> 0.347 |
| faceverse_neutral | factorized_s1234 | d_P | SR | 0.339 -> 0.347 | 0.365 -> 0.351 | 0.345 -> 0.345 | 0.345 -> 0.343 | 0.348 -> 0.344 |
| faceverse_neutral | factorized_s1234 | d_F cal. | FR | 0.341 -> 0.351 | 0.371 -> 0.362 | 0.346 -> 0.348 | 0.349 -> 0.345 | 0.354 -> 0.348 |
| faceverse_neutral | factorized_s2345 | d_P | SR | 0.375 -> 0.377 | 0.389 -> 0.395 | 0.377 -> 0.376 | 0.383 -> 0.381 | 0.379 -> 0.377 |
| faceverse_neutral | factorized_s2345 | d_F cal. | FR | 0.375 -> 0.379 | 0.387 -> 0.389 | 0.375 -> 0.379 | 0.384 -> 0.385 | 0.379 -> 0.379 |
| faceverse_neutral | ctrlfr_s1234 | ||z|| | FR | 0.354 -> 0.360 | 0.369 -> 0.382 | 0.359 -> 0.355 | 0.358 -> 0.354 | 0.366 -> 0.356 |
| faceverse_neutral | ctrlfr_s1234 | ||z|| | SR | 0.355 -> 0.364 | 0.358 -> 0.372 | 0.358 -> 0.358 | 0.350 -> 0.346 | 0.364 -> 0.357 |

## Remesh: validita' e geometria (sez. 1)

| vista | topologia | vertici prima (mediana) | dopo (mediana, min-max) | lato mm (mediana) | CV del lato | area dopo / prima (mediana, min) | Hausdorff ingresso -> canonica mm (media mediana, max p90) | piu' componenti | con spigoli non-manifold | s per mesh (mediana, p90) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| facescape | crop | 12096 | 10603 (9739-11300) | 2.04 | 0.23 | 0.998 (0.997) | 0.044 (0.60) | 0 | 0 | 5.9 (6.5) |
| facescape | down8k | 4385 | 12822 (11780-13781) | 2.07 | 0.13 | 0.998 (0.998) | 0.040 (0.56) | 0 | 0 | 2.8 (4.9) |
| facescape | noisy | 12596 | 25984 (24733-27578) | 1.91 | 0.29 | 0.990 (0.988) | 0.004 (1.28) | 0 | 0 | 25.0 (27.0) |
| facescape | original | 12596 | 13446 (12392-14432) | 2.05 | 0.21 | 0.998 (0.998) | 0.044 (0.62) | 0 | 0 | 7.0 (7.7) |
| facescape | remesh | 8832 | 12172 (11070-13054) | 2.07 | 0.14 | 0.999 (0.998) | 0.042 (0.80) | 0 | 0 | 3.3 (3.7) |
| facescape | up60k | 32320 | 13446 (12391-14461) | 2.05 | 0.21 | 0.998 (0.998) | 0.035 (0.60) | 0 | 29 | 9.2 (10.0) |
| hifi3d | crop | 9062 | 8762 (6743-10599) | 2.08 | 0.13 | 0.998 (0.997) | 0.047 (0.64) | 0 | 0 | 2.9 (3.2) |
| hifi3d | down8k | 3312 | 10860 (8478-13163) | 2.07 | 0.13 | 0.998 (0.997) | 0.041 (0.56) | 0 | 0 | 2.5 (2.9) |
| hifi3d | noisy | 9518 | 19222 (15960-22601) | 1.91 | 0.28 | 0.988 (0.987) | 0.012 (1.86) | 0 | 0 | 11.4 (13.1) |
| hifi3d | original | 9518 | 10812 (8414-13039) | 2.08 | 0.13 | 0.998 (0.998) | 0.046 (0.64) | 0 | 0 | 3.4 (3.8) |
| hifi3d | remesh | 6679 | 10324 (8018-12518) | 2.08 | 0.13 | 0.998 (0.998) | 0.043 (0.56) | 0 | 0 | 2.7 (3.1) |
| hifi3d | up60k | 24394 | 10934 (8587-13146) | 2.06 | 0.13 | 0.998 (0.997) | 0.033 (0.70) | 0 | 1 | 4.9 (5.3) |
| faceverse_neutral | crop | 26126 | 10014 (9005-11458) | 2.04 | 0.19 | 0.996 (0.991) | 0.031 (2.01) | 0 | 0 | 5.1 (5.7) |
| faceverse_neutral | down8k | 9930 | 10914 (9676-12478) | 2.04 | 0.18 | 0.996 (0.991) | 0.038 (1.99) | 0 | 0 | 3.9 (4.3) |
| faceverse_neutral | noisy | 28632 | 39330 (36425-42324) | 1.76 | 0.36 | 0.980 (0.975) | 0.003 (2.06) | 0 | 0 | 29.8 (32.7) |
| faceverse_neutral | original | 28632 | 11318 (10286-13082) | 2.03 | 0.21 | 0.996 (0.991) | 0.031 (2.01) | 0 | 0 | 6.1 (6.5) |
| faceverse_neutral | remesh | 20109 | 9986 (8617-10895) | 2.06 | 0.14 | 0.998 (0.997) | 0.029 (0.88) | 0 | 0 | 4.1 (4.3) |
| faceverse_neutral | up60k | 73603 | 11430 (10377-13256) | 2.02 | 0.22 | 0.995 (0.991) | 0.027 (2.01) | 0 | 0 | 12.9 (13.7) |

## Costo per mesh (sez. 6)

Campione di 54 mesh (primi 3 soggetti valutati x 6 topologie x 3 viste), un thread, nodo a256-t4-02.srv.aau.dk; remesh su tutte le 1.800: mediana 4.99 s, p90 21.55 s.

| | mediana s | p90 s | media s |
| --- | --- | --- | --- |
| operatori sull'originale (pipeline attuale) | 4.11 | 11.50 | 6.58 |
| remesh | 5.25 | 21.01 | 8.43 |
| operatori sulla canonica | 3.69 | 8.52 | 4.73 |
| remesh + operatori canonici | 9.10 | 29.44 | 13.16 |

Rapporto (remesh + operatori canonici) / operatori originali per mesh: mediana 2.45 (p10 0.90, p90 7.26).

| vista | operatori originali (mediana s) | remesh + operatori canonici (mediana s) |
| --- | --- | --- |
| facescape | 4.11 | 10.98 |
| hifi3d | 2.93 | 6.56 |
| faceverse_neutral | 10.07 | 9.23 |

| topologia | operatori originali (mediana s) | remesh + operatori canonici (mediana s) |
| --- | --- | --- |
| crop | 3.99 | 8.57 |
| down8k | 1.23 | 7.65 |
| noisy | 4.13 | 35.28 |
| original | 4.13 | 10.03 |
| remesh | 2.68 | 7.36 |
| up60k | 11.95 | 14.22 |

Embedding, tempo a parete per 600 mesh nello stesso job (2 code su 2 A100, preload compreso): `emb/facescape/ctrlfr_s1234` 94 s; `emb/facescape/factorized_s1234` 106 s; `emb/facescape/factorized_s2345` 105 s; `emb/faceverse_neutral/ctrlfr_s1234` 101 s; `emb/faceverse_neutral/factorized_s1234` 101 s; `emb/faceverse_neutral/factorized_s2345` 102 s; `emb/hifi3d/ctrlfr_s1234` 82 s; `emb/hifi3d/factorized_s1234` 82 s; `emb/hifi3d/factorized_s2345` 82 s; `emb_check/hifi3d_factorized_s1234` 75 s. Confronto dichiarato: HIFI3D canonica 82 s contro originale (controllo della catena) 75 s.

## Controlli

- K1 (originale = D3, d_P con SR, sulle righe comuni = righe di D3): scarto massimo 0.0e+00, righe uguali True: **passa** (facescape factorized_s1234 0.7465 contro 0.7465, righe 88725 / 88725; facescape factorized_s2345 0.7530 contro 0.7530, righe 88725 / 88725; hifi3d factorized_s1234 0.6217 contro 0.6217, righe 98224 / 98224; hifi3d factorized_s2345 0.6128 contro 0.6128, righe 98224 / 98224; faceverse_neutral factorized_s1234 0.3334 contro 0.3334, righe 99000 / 99000; faceverse_neutral factorized_s2345 0.3719 contro 0.3719, righe 99000 / 99000).
- K2 (catena: factorized s1234 sulla cache dello store di HIFI3D): 600 mesh, scarto massimo 4.8e-07: **passa**.
- K3 (determinismo del remesher, 10 mesh): **passa**.
- K4 (validita'): facescape 600 / 600 valide; hifi3d 600 / 600 valide; faceverse_neutral 600 / 600 valide: **passa**.
- K5 (righe nocrop = fact_paired.rows_for): facescape 99000 righe, chiavi uguali (scarto GT 0.0); hifi3d 99000 righe, chiavi uguali (scarto GT 0.0); faceverse_neutral 99000 righe, chiavi uguali (scarto GT 0.0): **passa**. Semi: facescape 796786 (fact_paired 796786, D3 796786); hifi3d 990708 (fact_paired 990708, D3 990708); faceverse_neutral 271049 (fact_paired 271049, D3 271049).
- Righe per gruppo dopo la maschera: facescape: nocrop_cross 99000, stessa topologia 24750, all_cross 148500, righe col crop 49500, nocrop_cross, righe di D3 88725; hifi3d: nocrop_cross 99000, stessa topologia 24750, all_cross 148500, righe col crop 49500, nocrop_cross, righe di D3 98224; faceverse_neutral: nocrop_cross 99000, stessa topologia 24750, all_cross 148500, righe col crop 49500, nocrop_cross, righe di D3 99000; tolte dalla maschera: facescape 0; hifi3d 0; faceverse_neutral 0.
