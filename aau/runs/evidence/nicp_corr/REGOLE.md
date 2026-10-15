# NICP per le corrispondenze (`nicp_corr`): regole (scritte e committate PRIMA dei numeri)

Domanda dell'utente: il NICP per coppia della matrice (`aau/baselines_mm/blmm.py:311-333`, `pair_metrics` passo `nicp`)
legge il RESIDUO dopo la deformazione: `X_nicp = nonrigid_icp_align(X_rig, Ys)`, distanza = media P2Tri di `X_nicp` su
`Ys`. Di solito il NICP serve a stabilire CORRISPONDENZE, e la distanza si misura fra la mesh NON deformata (allineata
rigidamente) e i punti corrispondenti. Il residuo dopo la deformazione cancella l'identita' quasi per costruzione: un
revisore puo' chiamarlo un concorrente costruito per perdere. Qui si misura la seconda lettura sulle stesse coppie, con
la stessa deformazione, e la si confronta con il residuo, l'ICP e il nostro. Niente Ava-256, niente `paper/`.

Codice: `blmm.pair_metrics` passo `nicp_corr` (nuovo, opzionale: il passo `nicp` e il default non cambiano) e
`aau/nicp_corr/nc.py` (+ `nc.sbatch`, solo CPU, partizione prioritized).

## 1. Convenzione ufficiale: REALY

REALY (`~/ext/protocolli/cloni/REALY`, commit 3f13e9a di `pr_common.CLONE_COMMITS`), `utils/eval.py`:
`:70` punto piu' vicino sulla superficie della predizione per ogni vertice del riferimento; `:77-103` nICP che deforma
il riferimento verso quei punti; `:106-110` nuove corrispondenze dai vertici DEFORMATI; `:114-116` errore = media di
||corrispondenza aggiornata - vertice ORIGINALE del riferimento||. E' la lettura `nicp_corr`: il NICP fissa le
corrispondenze, la distanza si misura sulla geometria non deformata.

Il protocollo REALY completo (gICP sui keypoint, ICP rigido per regione, nICP con keypoint, 4 regioni, media dei due
versi: `pr_metrics.realy_pair` `pr_metrics.py:405-460`, nICP `:339-402`, `pr_common.py:30`) e' GIA' nella matrice come
plug-in `realy` (landmark esatti). Si riporta quella colonna sulle stesse righe, senza reimplementarla (costo zero).
Differenze da `nicp_corr`: keypoint e regioni, punto piu' vicino sulla superficie invece che sul campione, soglia
dell'arresto 0,5^d (epsilon 1) invece di 6 x 0,5^d, triangoli del riferimento invece della Delaunay sul piano xy. Una
variante per coppia "alla REALY" senza landmark (solo soglia e proiezione sulla superficie) NON si aggiunge: la lettura
e' la stessa di `nicp_corr`, e la soglia di REALY raddoppia il costo del NICP (robustezza_iperparametri, `eps_realy`:
8,3 contro 4,1 s per coppia).

## 2. Le letture (`blmm.pair_metrics`, passo `nicp_corr`, modi `mm` e `cs`)

Campioni di 4096 punti `Xs`, `Ys` coi semi della coppia (`seed`, `seed + 1`), `X_rig = blmm.icp(Xs, Ys, modo)` e
`X_nicp = fb.nonrigid_icp_align(X_rig, Ys)`: gli STESSI passi e le stesse chiamate del passo `nicp`, nello stesso ordine.

- `nicp_p2tri`: media `fb.p2tri_distance(X_nicp, Ys, corr)`, `corr = fb.chamfer_correspondence(X_nicp, Ys)`; il residuo
  della matrice, deve riprodurla (sez. 5).
- `nicp_corr_xy`: stessa `corr`, distanza = media ||X_rig[i] - Ys[corr[i]]|| (`fb.p2p_distance(X_rig, Ys, corr)`): la
  mesh rigida originale contro il punto corrispondente del bersaglio.
- `nicp_corr_yx`: lo stesso con i ruoli scambiati sugli stessi campioni: `Y_rig = blmm.icp(Ys, Xs, modo)`,
  `Y_nicp = NICP(Y_rig, Xs)`, media ||Y_rig[i] - Xs[corr'[i]]||, `corr' = chamfer_correspondence(Y_nicp, Xs)`.
- `nicp_corr` = (`nicp_corr_xy` + `nicp_corr_yx`) / 2: simmetrizzata come i protocolli asimmetrici della matrice (media
  dei due versi, `pr_common.py:30`). NaN se uno dei due versi fallisce.
- `nicp_p2tri_yx`: il residuo P2Tri del verso inverso (gratis: stessa deformazione), per la colonna descrittiva
  `nicp_p2tri_sym` = media dei due versi del residuo, che separa l'effetto della lettura da quello della simmetria.

Distanze x L_d (mm; in `cs`, mm alla taglia di riferimento). Nota strutturale: `nicp_corr_xy` >= `rigid_icp_chamfer`
della matrice su ogni coppia (stessa `X_rig`, e il piu' vicino minimizza la distanza punto per punto).

## 3. Righe, colonne, bootstrap

- **HIFI3D e FaceScape dev**, GT FR e SR, SENZA crop. Coppie = il sottocampione del rumore landmark (`pr_noise.subsample`:
  8% delle coppie non ordinate di all_cross, seme 20261013 e nome della vista) ristretto alle coppie senza crop: le stesse
  di `robustezza_iperparametri` (`ri.view_data`), X = mesh del soggetto di indice minore, seme = indice della coppia di
  soggetti in `np.triu_indices` (la regola di `blmm_pairs`). Stesse coppie per tutte le colonne.
- Colonne, per modo `mm` e `cs`: `nicp_p2tri` (ricalcolato), `nicp_corr`, `nicp_corr_xy`, `nicp_p2tri_sym`; dalla
  matrice (nessun ricalcolo, `fm_eval.geo_columns`): ICP + Chamfer (`rigid_icp_chamfer`) e NICP su template
  (`nicp_template`); il nostro factorized s1234 e s2345 (`fact_paired.model_columns`: d_F non cal. e d_P); REALY
  (plug-in `realy`, landmark esatti). Ogni colonna contro GT FR e SR.
- Righe = `bp_paired_e2.all_cross_rows` sulle coppie del sottocampione, soggetti diversi; UNA maschera per vista = righe
  con TUTTE le colonne finite; le righe tolte per colonna in `controls.json`.
- Spearman pesato della matrice (`bp_paired_e3.replicas`), 1000 repliche per soggetto col seme della T1
  (`pr_eval.t1_seed`: HIFI3D 990708, FaceScape 796786), IC 95% percentile, tutte le colonne sulle STESSE repliche. Delta
  appaiati replica per replica.

## 4. Letture primarie e verdetto (fissati ora)

Coppie primarie (normalizzazione coerente con la GT, come in `blmm`): **FR con `mm`**, **SR con `cs`**; le altre
combinazioni sono descrittive. Per vista e coppia primaria:

- D1 = rho(`nicp_corr`) - rho(`nicp_p2tri`): la lettura recupera l'identita' cancellata dal residuo?
- D2 = rho(`nicp_corr`) - rho(ICP + Chamfer, stesso modo): il NICP per corrispondenze vale l'ICP?
- Descrittivi: `nicp_corr` - nostro (d_F non cal. con FR, d_P con SR, s1234 e s2345), - NICP su template, - REALY;
  `nicp_corr_xy` - `nicp_p2tri` (solo la lettura, stesso verso); `nicp_corr` - `nicp_p2tri_sym` (solo la lettura, entrambi
  simmetrici). Spearman semplice fra `nicp_corr` e ICP sulle righe della maschera (quanto `nicp_corr` e' l'ICP).

Verdetto per vista e coppia primaria:
- IC di D1 che include 0 o tutto sotto: **"cancella ancora l'identita'"**;
- D1 sopra 0 e D2 tutto sotto 0: **"recupera in parte, sotto l'ICP"**;
- D1 sopra 0 e IC di D2 che include 0: **"recupera, pari all'ICP"**;
- D2 tutto sopra 0: **"sopra l'ICP"**.

## 5. Collaudo (prima di leggere qualunque GT)

- `nicp_p2tri` ricalcolato (mm e cs) contro `mm_nicp_p2tri` / `cs_nicp_p2tri` della matrice sulle stesse coppie:
  scarto massimo <= **1e-6 mm**, NaN sulle stesse coppie.
- Se il controllo supera la tolleranza per via del nodo (il sistema sparso del NICP dipende dalla CPU:
  robustezza_iperparametri, emendamenti 2 e 4), per ogni coppia ordinata di topologie fuori tolleranza si ricalcola il
  solo `nicp_p2tri` di TUTTE le coppie del sottocampione di quella coppia di topologie su un nodo con la CPU che l'ha
  calcolata nella matrice (log `aau/logs/wbes-blmm-*`; per i job su piu' nodi, i tipi di nodo del job, provati sulle
  coppie peggiori), entro 1e-6 mm e NaN sulle stesse coppie. I valori ricalcolati non entrano nella valutazione, che usa
  il calcolo principale (tutte le colonne di una vista e un modo dallo stesso nodo).
- Strutturale: `nicp_corr_xy` >= ICP + Chamfer della matrice - 1e-9 mm su ogni coppia finita.
- `eval` non parte se un controllo non passa. Se una coppia di topologie non si riproduce su nessun tipo di nodo
  candidato, ci si ferma e si riporta.

## 6. Fuori da questo giro

Crop, FaMoS, FaceVerse, Ava-256; altre configurazioni del NICP (stanno in robustezza_iperparametri); la variante per
coppia alla REALY (sez. 1); qualunque modifica al passo `nicp` della matrice o alle colonne gia' pubblicate.
