# Trainer v3: modalita' fattorizzata dimensione + forma

Scritto il 9 ottobre 2026 dal coder (compito del PI: ingresso a scala globale e embedding z = (s, u)). Codice in
`v3_work/trainer/`; i default del trainer sono invariati (sezione 5). Dalle 13:50, su richiesta del PI, i run con le
GT di E12 (FR, SR, S_i) sono in coda/esecuzione: protocollo `factorized_protocol.md` (sha256 registrato prima dei
numeri), sezione 4.

## 1. Cosa fa

**Ingresso a scala globale** (`--input-norm global --scale-table T [--global-unit-mm 100] [--global-ops areanorm]`,
`global_v3.py`). X = ((V - c) f) R_d^T / L0:
- V = vertici del loader congelato (centro dei vertici e maxabs per mesh, come sempre); f = sqrt(area_mm2 / area(V))
  li riporta in mm. `area_mm2` = u_d^2 x area della geometria GREZZA da cui la mesh deriva, per nome di file, dalla
  tabella di `tools/build_scale_table.py` (la scala non sopravvive nello store: gli npz sono ad area 1 e il loader
  divide per maxabs). L'area non dipende da centro e rotazione, quindi f non dipende da come il loader ha inquadrato.
- **u_d e R_d = il frame di GT-F di E12** (`aau/runs/evidence/e12/frames.json`, emendamento 1 del protocollo E12):
  u_d = unita' fisica dichiarata (BFM um, ICT cm, GNM m, FaceScape mm) o 63 mm / IPD della media (HIFI3D, FaceVerse);
  R_d = la rotazione di `datasets/UNIFIED_GT/canonical_transforms.json` (frame FLAME: +y alto, +z fuori dal volto).
  NON la scala s_d = u_d k_d del json, che porta la media di ogni dominio sulla taglia della media FLAME e
  cancellerebbe le differenze di taglia fra domini (k_d = 0.96-1.04). Prima versione con s_d, corretta prima di
  qualunque run quando E12 ha emendato GT-F.
- c = baricentro pesato per AREA ROBUSTA (`--area robust`: aree della geometria passa-basso, come arearobust; con
  `--area on` la massa). Nessuna scala per mesh: L0 = 100 mm e' la stessa costante per tutte.
- Le perturbazioni del training restano con le stesse sigma, quindi in unita' di L0: 0.05-2 mm (vedi limiti).
- **Operatori** (`--global-ops`): restano quelli dello store (area 1). Il loader congelato divide gli autovalori per
  lambda_max e i gradienti per sqrt(lambda_max), quindi diffusione spettrale, gradienti e pesi del pooling sono gia'
  adimensionali: l'unica grandezza con la scala e' xyz. `--global-ops mm` riporta massa e autovettori alle unita' di
  X (massa x k, autovettori / sqrt(k), M-ortonormalita' conservata): stesso embedding (misurato, sezione 5), il flag
  documenta l'invarianza.

**Embedding fattorizzato** (`--head factorized --size-table size.npz [--lambda-size 1] [--scale-aug 0.8,1.25]`,
`factorized_v3.py`). Il modello (EncoderV3 con pooling per area) restituisce [s, u]: u = la proiezione di sempre
(256), s = testa piccola (512 -> 64 -> 1, SiLU) sulle stesse feature aggregate, ultimo strato a zero con bias =
media di log S sul training. Loss = loss di forma scelta da `--loss` (v2: stress + rank + id) su u con la GT shape +
lambda_size x MSE(s, log S_i + log a). Augmentation: ogni mesh moltiplicata per a log-uniforme in [0.8, 1.25] PRIMA
del rumore; le estrazioni dipendono da (seme, rank, passo), quindi la ripresa le riproduce. L'eval online e gli
script di eval leggono solo u (`fz.set_output`); `WBES_V3_FACTORIZED_OUT=full` restituisce [s, u].

## 2. Bersagli e convenzione della distanza form

`tools/build_factorized_targets.py` (venv della GT unificata), per ogni identita' BFM/ICT/GNM: p_i = mappa della
forma neutra nativa sulla regione unificata (1478 punti; BFM original REMESH, ICT e GNM dai pesi, come
`shapes.py`), f_i = u_d R_d p_i + t_d (frame di GT-F), W = pesi d'area di mu normalizzati.
- **S_i** = sqrt(sum_v W_v ||f_i,v - m_i||^2), centroid size in mm; bersaglio di s: log S_i, lo stesso per ogni mesh
  dell'identita' (crop ed espressioni compresi).
- **GT shape: dP_ij = 2 sin(rho_ij / 2) = ||z_i - z_j||_W**, corda fra pre-forme a centroid size UNITARIA
  z_i = (f_i - m_i) / S_i, nel frame di GT-F, **senza rotazione per identita'** (come E12: nessun allineamento per
  identita'). In `gt_shape.npz`: D_orig = kappa x dP, kappa = mediana della GT maxabs / mediana di dP sulle coppie di
  training dentro il dominio (i margini della loss v2, 0.05 e 0.02, sono in unita' di GT); `dp_per_unit` = 1/kappa
  nel json: **dP = ||u_i - u_j|| x dp_per_unit**.
- **Distanza form (convenzione usata): d_F^2 = (S_i - S_j)^2 + S_i S_j dP^2**, con S = exp(s) e dP dalla corda: la
  forma esatta della distanza size-and-shape di Dryden e Mardia (2016; `ssriemdist` del pacchetto R shapes),
  equivalente a S_i^2 + S_j^2 - 2 S_i S_j cos(rho). `factorized_v3.form_distance` / `pair_distances`.
- **Coerenza con E12.** Con dP senza rotazione, d_F coincide ESATTAMENTE con `F_centered` di E12 (stessi punti, pesi,
  frame): identita' algebrica, misurata. `F` di E12 aggiunge la traslazione per identita', che (S, dP) non
  rappresenta; `S` di E12 scala attorno al centroide fisso di mu senza togliere la traslazione, quindi
  CS(mu) x dP ne e' la versione centrata. Con la rotazione ottimizzata per coppia la stessa formula da' la
  size-and-shape di Procrustes (riportata nei controlli). Numeri: sezione 5.

## 3. Lo store: stesso store, piu' una tabella

Gli operatori non cambiano (area 1, come gli altri bracci), e gli xyz globali si ottengono dai vertici dello store
con il fattore per mesh della tabella: **non serve uno store nuovo**. Pero' lo store dei bracci
(`/tmp/wbes_v3_store_c3f` su nv-ai-04, job 1062011) non esiste piu': l'epilogo del nodo ha pulito /tmp quando sono
finiti i custodi (1062017, 1062076). Va ricostruito con la stessa ricetta (`build_store.sbatch`, 2 h 37 la prima
volta su 224 CPU). Gli operatori delle mesh dei tar si ricalcolano: i segni degli autovettori possono cambiare
(ARPACK), quindi **factorized va confrontato con un braccio sullo STESSO store nuovo**: `robal` (arearobust + bal,
i due flag adottati insieme, mai addestrati in combinazione), preparato accanto.

## 4. Run lanciati (protocollo `factorized_protocol.md`) e bracci preparati

Dopo E12 (GT di riferimento FR = F + rigida robusta) la GT di training di u e' **GT-SR di E12** (d_P di Procrustes
fra pre-forme a CS 1, grezza x kappa, kappa = media geometrica dei fattori di taratura per dominio: un valore, cosi'
d_P = ||u_i - u_j|| x dP_per_unit / kappa resta assoluta) e il bersaglio di s e' log S_i di E12
(`centroid_size_bfm_ict_gnm.npz`). I miei bersagli (`tools/build_factorized_targets.py`, d_P senza rotazione) restano
per i dati di prova e per il controllo di coerenza. Job in `ablations/c3f_jobs.txt`:
- `factorized` s1234 / s2345 e `ctrlfr` s1234 / s2345 (testa standard, GT-FR tarata, ingresso globale), C3F, 1 A100
  ciascuno, dopo lo store C3F ricostruito, i dati del braccio e `prepare_fr.sbatch` (attende `READY` di E12, ritaglia
  SR e FR tarata sui 6.993 soggetti, scrive kappa e la spec del seme 2345); eval con `eval_body.sh` (HIFI3D, dev
  FaceScape, FaceVerse + passo `form`: FR, SR, maxabs con `tools/eval_factorized.py`);
- `factorized` C3M (split del run su scala, K 46, T 60.000, DDP su 6 L40S di a768-l40s-06, `slurm/factorized_c3m.sbatch`:
  attende `READY` e la tabella di scala C3M), eval delle epoche 123 e 205 in dipendenza.
Il braccio della prima versione (GT shape mia, confronto `robal`) resta in `train_body.sh` ma non e' lanciato.

### Preparazione originale

- `v3_work/trainer/ablations/c3f/train_body.sh`: bracci `robal` e `factorized` (= robal + `--input-norm global
  --scale-table c3f/scale_table.npz --head factorized --size-table c3f/factorized/size.npz --scale-aug 0.8,1.25`, GT
  `c3f/factorized/gt_shape.npz`); stessi dati, split, spec, ordine, seme, passi (21.096, S 293), EMA, ricetta.
- `build_factorized_data.sbatch`: tabella di scala dello store C3F, tabelle delle viste di eval, GT shape e
  centroid size dei 6.993 soggetti (eseguito, sezione 5).
- `launch_factorized.sh`: dati (se mancano) + store ricostruito + robal + factorized + eval + custode dello store.
- `eval_body.sh`, ramo `factorized*`: `WBES_V3_SCALE_TABLES` sulle viste di eval, gli script ricevono u (stesse
  misure del protocollo: dev FaceScape, HIFI3D maxabs e unificata, FaceVerse); NoW escluso (nessuna tabella di
  scala per le scansioni); passo `form`: embedding [s, u] di HIFI3D, dev FaceScape e FaceVerse e
  `tools/eval_factorized.py` contro TUTTE le GT di E12 presenti in `datasets/CANONICAL_GT/<set>_*.npz` (F,
  F_centered, S, EDM, EDM_s, ...): Spearman di d_F, dP e |delta s| con IC bootstrap per soggetto.
- **Prima del lancio:** un emendamento del protocollo delle ablazioni (confronto factorized contro robal, regola,
  metriche form) con lo sha256 registrato; `tools/ablation_summary.py` giudica ogni braccio contro ctrl
  (`ARMS` fisso) e va esteso al confronto con robal.

## 5. Verifiche (eseguite)

- **Default invariati.** `tests/test_units_cpu.py` 7/7 e `tests/test_eval_hook.py` 9/9 (scarto 0.0) sul codice
  modificato (`factorized_defaults/`); equivalenza L1 v2 = v3 18/18 (scarto 0 di loss e gradienti); L2 traiettoria
  sullo split di sole viste (configurazione dell'evidenza `a100/equivalence_l2_views`, job 1062785): v2 = v3 BIT PER
  BIT (loss, eval online e pesi alle epoche 2-8, scarto 0.0). L'L2 sullo split con i tar (job 1062736) mostra scarti
  1e-3 anche fra run v2 e v3 perche' ogni run rifa' il pre-pass (segni degli autovettori): non e' un confronto valido.
  Packed/groups contro sequenziale in fp32: FALLITO come nell'evidenza dell'8 ottobre, stessi valori (non toccati).
  Nomi dei run dir (hash compreso) dei 6 bracci C3F gia' addestrati ricostruiti identici con i flag nuovi al default.
- **Modalita' nuova** (`tests/test_factorized.py`, `factorized/units.json`, 7/7): X ha area area_mm2 / L0^2 (err rel
  <= 1.3e-8), non dipende dal frame d'ingresso (1.2e-7), coincide con u_d R_d V_grezza / L0 entro 1.5e-5 mm su BFM e
  ICT; `--global-ops mm` = `areanorm` (scarto 5e-8 su |z| 4.1); eval_v3 = training con uscita u e [s, u] (scarto 0.0);
  testa: s iniziale = media, gradiente su testa ed encoder; estrazioni di scala riproducibili; identita' della
  distanza form (1e-13).
- **Tabelle di scala:** 366 viste verificate contro la geometria grezza (maxabs, scarto massimo 1.1e-7).
- **Coerenza con E12** (`factorized/e12_coherence.json`, matrici di E12 su disco): d_F da (S, d_P senza rotazione)
  contro `F_centered` di E12: scarto massimo 1.7e-12 mm su HIFI3D, FaceVerse e FaceScape (identita' esatta); Spearman
  di d_F con F 0.67 / 0.99 / 0.92; d_P con l'unificata 0.90-0.94. Nei dati di prova d_P senza rotazione contro quella
  di Procrustes: Spearman 0.89-0.98.
- **Aggancio di eval** (`tools/eval_factorized.py`) validato su e108: HIFI3D `nocrop_cross` (livello coppia di mesh,
  come E12) FR 0.194 e maxabs 0.630, gli stessi valori di E12.
- **Smoke DDP** della combinazione completa (2 GPU, crash al passo 18 e ripresa): ha trovato un bug (nome `fz`
  oscurato in `Data.__init__` dal json del held-out congelato) e un difetto preesistente della ripresa elastica (un
  temporaneo `.tmp.npz` del pre-pass ucciso letto come mesh al cambio di blocco: `BlockedDataset.stage` ora lo
  toglie). Job 1062807: crash al passo 18, ripresa, cambio di blocco, 40/40 passi, pesi identici sui rank.
- **Run breve di stabilita'** (dati di prova, 2.000 passi, 1 A100, job 1062802, dropout 0.1): nessun NaN, loss
  0.123 -> 0.030, MSE di s 0.020 -> 0.0022, Spearman online 0.70-0.80. **Equivarianza dopo il training** (32 soggetti
  mai visti, `short_1062802/equivariance*.json`): NON ancora appresa. Pesi grezzi: ds/dlog a = 0.70-0.73 (atteso 1),
  ||u(aX) - u(X)|| = 10-29% della distanza mediana fra soggetti; EMA: 0.25-0.28 (al passo 2.000 l'EMA 0.999 pesa
  ancora 13.5% l'inizializzazione). Bias di s: -0.21 in modo eval, ~0 in modo train sulle stesse mesh di training
  -> dropout 0 nei run del protocollo (emendamento 2); verifica con dropout 0 in corso (job 1062946).

## 6. Limiti dichiarati

- Il braccio e' un PACCHETTO (ingresso, testa, GT shape, loss di taglia, augmentation), come loginv: contro robal
  misura il pacchetto, non un singolo fattore.
- Rumore del training: con L0 = 100 mm le sigma valgono 0.05-2 mm per tutti i domini; in arearobust (unita'
  sqrt(area): BFM ~171 mm, GNM ~191, ICT ~280 sulle original) erano 1.7-2.8 volte piu' grandi in mm.
- BFM: le 500 original REMESH sono gia' allineate per similarita' una per una (E12): la taglia BFM e' quasi costante
  (CV sezione 5), quindi il segnale di taglia viene da ICT e GNM.
- Solo `--forward sequential` con la testa fattorizzata (packed e groups chiamano pool_proj).
- Lo stream (`v3_work/stream/`) non e' toccato: il training massivo a streaming non ha ancora l'ingresso globale.
