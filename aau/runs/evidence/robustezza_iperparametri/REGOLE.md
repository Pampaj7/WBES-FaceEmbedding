# Robustezza agli iperparametri: regole (scritte e committate PRIMA dei numeri)

Domanda del gruppo: i metodi con allineamento per coppia (ICP, NICP, NoW) hanno molte manopole; cambiando valori
ragionevoli il risultato puo' cambiare. Il nostro encoder, a pesi fissi, non ne ha al momento del test, ma ha la
variabilita' del seme di training. Qui si misura quanto cambiano i concorrenti su griglie di configurazioni
ragionevoli, e lo si confronta con lo scarto fra i nostri due semi. Lettura DESCRITTIVA, nessun test confermativo.

Codice: `aau/robustezza_iperparametri/ri.py` (+ `ri.sbatch`), definizioni nel docstring. Le implementazioni sono quelle
della matrice, importate; cambiano solo i parametri. Niente Ava-256.

## 1. Dati e righe

- **HIFI3D e FaceScape dev**, GT FR, SENZA crop. Coppie = il sottocampione del rumore landmark
  (`pr_noise.subsample`: 8% delle coppie non ordinate di soggetti e topologie diverse, seme 20261013 e nome della
  vista), ristretto alle coppie in cui nessuna delle due mesh e' crop. Stesse coppie per TUTTE le configurazioni e
  tutti i metodi. Righe = quelle di `bp_paired_e2.all_cross_rows` su queste coppie; UNA maschera per vista = righe con
  tutte le colonne (tutte le configurazioni di tutti i metodi e le 4 letture del nostro) finite. Le righe tolte per
  colonna si riportano in `controls.json`.
- **FaMoS, riconoscimento dell'identita'** (`aau/esterna_famos_id/`): le stesse 150 patch e 11.175 coppie, le stesse
  regole di `ei.py` (galleria = tutte le altre scansioni, distanza non finita = +inf).
- Orientamento e semi della matrice: X = mesh del soggetto di indice minore, campioni di 4096 punti coi semi della
  coppia (`blmm_pairs`: indice della coppia di soggetti in `np.triu_indices`; FaMoS: `blmm.mesh_seed` dei nomi).
  NoW: media dei due versi.

## 2. Bootstrap e misure

- Sintetiche: Spearman pesato con GT FR, 1000 repliche per soggetto col seme della T1 della vista (`pr_eval.t1_seed`:
  HIFI3D 990708, FaceScape 796786), `bp_paired_e3.replicas` (le repliche della matrice). FaMoS: rank-1 e mAP, 1000
  repliche per soggetto (`famos_eval.bootstrap_counts(15, 1000, 1234)`, query ricampionate per soggetto, galleria
  fissa: come `ei.py`). IC 95% percentile; tutte le configurazioni sulle STESSE repliche.
- Per metodo, sulle sue configurazioni: **intervallo** (max - min) e **deviazione standard** (ddof 1) del valore, con
  IC calcolato replica per replica; **default sbagliato**: migliore - peggiore (scelte sul punto: post hoc), default -
  peggiore e migliore - default, delta appaiati con IC.
- Nostro: le due varianti esistenti, factorized **s1234** e **s2345** (epoch072 EMA), letture d_F non cal.
  (preregistrata) e d_F cal. (post hoc, emendamento 4): **scarto fra i semi** = |rho_s1234 - rho_s2345| per lettura,
  IC per replica; e' la variabilita' da seme, non un iperparametro di test. Si riporta anche l'intervallo sulle 4
  letture.
- **Confronto**: intervallo del metodo - scarto fra i semi (per lettura), per replica. Lettura fissata ora: IC tutto
  sopra 0 = "piu' instabile del seme"; tutto sotto 0 = "piu' stabile del seme"; altrimenti "non distinguibile".
- **Inversioni**: per ogni coppia di metodi (ICP, NICP, NoW, nostro d_F non cal., nostro d_F cal.; non nostro contro
  nostro) su tutte le combinazioni di configurazioni (i semi sono le "configurazioni" del nostro): quante danno a
  meglio di b e quante il contrario, sul punto e con IC del delta appaiato che esclude 0. Inversione = entrambi i
  versi presenti. **Classifiche** dei quattro metodi (ICP, NICP, NoW, nostro per lettura) su tutte le combinazioni:
  numero di classifiche distinte e frequenza di ciascuna; si marca quella delle configurazioni di default con s1234.

## 3. Griglie (una manopola per volta attorno al default della matrice; ogni valore ha una fonte)

Valori in mm convertiti nelle coordinate di lavoro (mm / L_d, `blmm`). Nessun valore scelto per rompere un metodo.

**ICP + Chamfer in mm** (`blmm.pair_metrics` passo `fast`, lettura `rigid_icp_chamfer`):

| configurazione | manopola | valore | fonte |
|---|---|---|---|
| default | - | centro + `registration_icp` punto-punto, soglia 1000 (infinita), 30 iterazioni | matrice (`blmm.icp`, criteri di default di Open3D) |
| it10 | iterazioni | 10 | PCL `Registration::max_iterations_{10}` (default) |
| it2000 | iterazioni | 2000 | tutorial ICP di Open3D 0.18 (`ICPConvergenceCriteria(max_iteration=2000)`) |
| thr20 | soglia | 20 mm | tutorial ICP di Open3D 0.18 (`threshold = 0.02`, m) |
| thr10 | soglia | 10 mm | tutorial colored ICP di Open3D 0.18 (`voxel_radius = [0.04, 0.02, 0.01]`, l'ultimo) |
| init_id | inizializzazione | identita' nel frame canonico (senza centrare) | Open3D, `init` di default |
| init_lmk | inizializzazione | rigida sui 7 landmark NoW RILEVATI (Kabsch senza scala) | avvio di NoW metrical; facebench `prealign='landmark'` |
| rej_pulli | rigetto | scarta il 10% delle coppie peggiori | Pulli 1999, in Rusinkiewicz e Levoy 2001 |
| rej_masuda | rigetto | scarta oltre 2,5 sigma (sigma = RMS delle distanze) | Masuda 1996, in Rusinkiewicz e Levoy 2001 |

Rigetto: a ogni iterazione la soglia di Open3D e' ricalcolata dalla distribuzione delle distanze dal piu' vicino
(una iterazione di Open3D per volta), arresto quando l'RMS di tutte le distanze cambia meno di 1e-6 unita' di lavoro o
dopo 30 iterazioni (il default di Open3D). La lettura resta su tutti i punti.

**NICP per coppia in mm** (`blmm.pair_metrics` passo `nicp`: ICP della matrice, poi NICP, lettura P2Tri): il ciclo di
`facebench.nonrigid_icp_align` con il programma come parametro (stessi aiutanti importati; il default lo riproduce).

| configurazione | manopola | valore | fonte |
|---|---|---|---|
| default | - | alpha 50 e^(-0,5 d), soglia 6 x 0,5^d, d = 0, 1, 2, gamma 1 | matrice (facebench) |
| eps_realy | iterazioni | soglia 0,5^d (senza x 6: piu' iterazioni) | REALY `utils/NICP.py` + `eval.py` (`epsilon = 1.0`) |
| cap10 | iterazioni | al piu' 10 iterazioni per stadio | trimesh 5.1 `nricp_amberg` (`max_iter = 10` per passo) |
| menpo3d | rigidezza | alpha = 50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2; arresto RMS(X - X_prec) < 1e-3; normalizzazione di menpo3d (centro, diagonale 1) | menpo3d `non_rigid_icp` (default, "for well rigidly aligned facial meshes") |
| menpo3d_lmk | landmark | come menpo3d + pesi dei landmark 5, 2, 0.5, 0, ... sui 7 landmark NoW RILEVATI | menpo3d `landmark_weights` (default) |

Nella matrice il NICP per coppia non ha landmark: `menpo3d_lmk` li aggiunge (landmark di X = punto del campione piu'
vicino, come REALY; posizioni = landmark di Y). Di menpo3d si prendono solo il programma, il criterio di arresto, la
normalizzazione e i pesi dei landmark; restano i pesi delle corrispondenze della matrice (non le normali) e niente
similarita' iniziale sui landmark (toglierebbe i mm).

**NoW metrical col rilevatore** (`pr_metrics`; landmark di insightface `1k3d68` su render frontale: `pr_noise`
condizione `det`, patch di `esterna_famos_id` su FaMoS):

| configurazione | manopola | valore | fonte |
|---|---|---|---|
| default | - | 7 landmark NoW, raggio 1,4, dogleg `e_3` 1e-4, `maxiter` 200 | matrice (plug-in `now_metrical/lmk_det`) |
| r12 | raggio | 1,2 | Stirling/ESRC, fg2018 `compute_vertices_to_mesh_distances.py:46` |
| icp0 | iterazioni | nessun ICP, solo il Procrustes | Stirling/ESRC (fg2018) |
| e3_0 | iterazioni | `e_3` = 0 (fino a `maxiter` 200) | chumpy `minimize_dogleg` (default) |
| lmk5 | landmark | 5 punti: centri degli occhi, punta del naso (30), angoli della bocca (48, 54) | ArcFace / insightface (5 punti) |
| lmk51 | landmark | 51 interni iBUG (17-67) | 300-W (senza contorno) |
| lmk68 | landmark | 68 iBUG | Sagonas et al. |

Il ritaglio usa sempre i 7 landmark NoW (cambia solo il fattore del raggio); il sottoinsieme cambia il Procrustes.

## 4. Controlli di riproduzione (prima di leggere qualunque GT)

La configurazione di default deve riprodurre, sulle stesse coppie: sintetiche, `mm_rigid_icp_chamfer` e
`mm_nicp_p2tri` della matrice (tolleranza 1e-6 mm; i NICP falliti, NaN, sulle stesse coppie) e il plug-in
`now_metrical/lmk_det` (1e-4 mm); FaMoS, `icp|mm` (1e-6 mm) e `now_metrical|mm` (1e-4 mm) di `esterna_famos_id`. Il NICP
su FaMoS non ha un riferimento (non e' fra i metodi di `esterna_famos_id`). `eval` non parte se un controllo non passa.

## 4b. Emendamento 1 (dopo il pilota dei tempi, PRIMA di ogni valutazione; nessuna GT letta)

Pilota su 32 coppie / 32 mesh A di HIFI3D (job 1074555; tempi e iterazioni soltanto, distanze cancellate):

- **Correzione del codice, configurazione invariata**: menpo3d controlla l'arresto DOPO ogni iterazione e ne fa almeno
  una per stadio (`non_rigid_icp`: `while True: j += 1; ...; if stop_criterion < eps: break`); la prima versione di
  `ri.nicp` controllava in testa come facebench, e con lo scarto rimasto dal primo stadio saltava gli altri 7 (iterazioni
  mediane 5, 0, 0, ...). Ora `dowhile` per `menpo3d` e `menpo3d_lmk`.
- **Osservazione, nessuna modifica**: nella matrice (facebench) e in REALY il controllo e' in testa: gli stadi 2 e 3
  partono con lo scarto dello stadio 1 e spesso non iterano (mediane 2, 0, 0 per default; 3, 1, 1 per `eps_realy`). Per
  questo `cap10` (tetto di 10 per stadio) coincide col default dove il primo stadio converge in meno di 10 iterazioni;
  resta nella griglia come dichiarata (un valore ragionevole che non cambia nulla e' un dato di robustezza).
- **Osservazione, nessuna modifica**: le due configurazioni di rigetto dell'ICP si fermano quasi sempre al tetto delle
  30 iterazioni (la soglia ricalcolata a ogni passo non lascia l'RMS fermo entro 1e-6).

## 5. Fuori da questo giro

Letture cs, SR, crop, NoW originale, REALY, Stirling; nessuna griglia sui semi di campionamento; nessun valore senza
fonte. Un pilota dei soli tempi (`--limit`, nessuna GT, distanze non riportate e cancellate) puo' precedere i job; se
una configurazione risulta incalcolabile, la modifica si dichiara qui in un emendamento PRIMA della valutazione.
