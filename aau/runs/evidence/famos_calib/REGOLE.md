# FaMoS: calibrazione lineare post hoc per l'invarianza all'espressione — regole (scritte PRIMA dei numeri)

Domanda (PI, 15 ottobre): sul riconoscimento dell'identita' FaMoS (`esterna_famos_id`) il nostro e' sotto ICP + Chamfer
in mm e NoW metrical, e `famos_dominio` attribuisce l'87-91% del divario di AUC alle coppie molto espressive: u si sposta
con l'espressione. Una calibrazione LINEARE, stimata su altre persone con piu' espressioni ciascuna (WCCN, LDA: classiche
nella verifica biometrica), basta a togliere la dipendenza dall'espressione e a chiudere il divario?

Codice: `aau/famos_calib/fcal.py` e `fcal.sbatch` (commit e sha256 nel messaggio del commit di questo file). Le
implementazioni della preparazione, del nostro, di ICP / NoW e delle metriche sono quelle di `esterna_famos_id`,
`robustezza_iperparametri` e `famos_dominio`, importate.

## 0. Etichetta: POST HOC e DESCRITTIVO

FaMoS TEST e' gia' stato visto: tutti i numeri di `esterna_famos_id`, `famos_dominio` (strati d'espressione) e
`robustezza_iperparametri` (le configurazioni di ICP e NoW su FaMoS TEST) sono noti a chi scrive. La domanda stessa nasce
da quei numeri. Quindi tutto cio' che segue e' **post hoc e descrittivo**: nessun test confermativo, nessun p-value; i
risultati sfavorevoli si riportano nella stessa tabella. Su FaMoS TEST non si sceglie NIENTE: ogni scelta (iperparametri,
dimensione, configurazione dei concorrenti, variante primaria) si fa sui soli soggetti di calibrazione.

Stato al commit di questo file:
- **calcolati**: la selezione dei soggetti e dei fotogrammi di calibrazione (`selection.json`; 73 soggetti) e la loro
  preparazione (patch, landmark, operatori), in corso o finita, senza nessuna distanza;
- **inesistenti**: latenti di calibrazione, distanze di calibrazione dei concorrenti, scelte, metriche calibrate.

## 1. Dati di calibrazione

- **Solo soggetti FaMoS TRAIN** (`aau/famos/split.json`, 80), **mai i 15 di TEST** (controllo in `select`).
- **Cosa c'e' su disco** (verificato): le scansioni grezze di TRAIN NON ci sono (cancellate per la quota, e comunque il
  rilascio TEMPEH ha grezzi solo per i 15 di TEST). Ci sono le **registrazioni FaMoS di TRAIN** sottocampionate
  (`datasets/FAMOS/train/<soggetto>.npz`: 1 fotogramma ogni 10 per sequenza, mesh in topologia FLAME, mm, con la neutra
  di riferimento `V_neutral`). Si usano quelle. Nessun download; nessun modello FLAME (ne' 2020 ne' 2023) e nessun
  embedding di landmark FLAME: le registrazioni sono trattate come mesh qualsiasi.
- **Differenza di dominio, dichiarata**: in calibrazione superfici registrate (lisce, chiuse, senza buchi ne' rumore di
  acquisizione), al test scansioni grezze. Una Sigma_w stimata sulle registrazioni puo' non trasferirsi alle grezze.
  Inoltre: pool dei fotogrammi dalle registrazioni sottocampionate (al test: dai fotogrammi con una scansione).
- **Sequenze**: le 10 di `esterna_famos_id` (bareteeth, blow_cheeks, disgust, fear, happiness, kissing, mouth_down,
  rolling_lips, surprise, wrinkle_nose). Soggetti TRAIN senza una di esse fra le registrazioni: **esclusi** (7: 007,
  023, 028, 036, 041, 065, 069). Restano 73 soggetti, 730 mesh.
- **Fotogramma**: la regola del test. Pool per (soggetto, sequenza) = galleria (fotogramma con `expr_mm` minima del
  soggetto, se cade nella sequenza), peak (`expr_mm` massima della sequenza), nearneutral (minima, galleria e peak
  esclusi), con `expr_mm` di `famos_test_view.select` (GT unificata contro `V_neutral`, Procrustes); scelta =
  `ei.pick_frame` (`rng([20261013, numero del soggetto, crc32(sequenza)])`).
- **Fold**: soggetti con tutte le patch, `rng(20261015).permutation`, 5 blocchi (`np.array_split`; `folds.json`). Una
  mesh la cui preparazione fallisce toglie il suo soggetto (dichiarato in `prep_calib.json`).

## 2. Preparazione (identica a `ei.py`)

Render frontale (`pr_noise.render_mesh`), 68 landmark insightface `1k3d68` (`pr_noise_detect.py --dir`, eseguito a
blocchi in parallelo e ricucito: stesso modello e stesse immagini), retroproiezione `ei.backproject` con `front_only`,
`famos_test_view.canonical_patch` (similarita' sul T7, ritaglio `compute_mask`, pulizia, `bfs_orient`, 5215 triangoli),
ritorno in mm, frame canonico; landmark NoW rilevati di nuovo sulla patch (`ei._lmk_task`). Nostro: operatori areanorm
k_eig 128 **su CPU** (lo stesso percorso di `ei.sbatch`; quello GPU di E10 non serve: costa ~1,5 s-core a mesh e qui
sono 730 mesh), tabella di scala in mm (`build_scale_table --domain famos`), latenti [s, u] dei checkpoint di `ei.CKPT`
dentro `eval_v3.py`.

**Riuso**: gli embedding e le distanze di test sono quelli gia' calcolati da `ei.py` (e, per le configurazioni dei
concorrenti, da `robustezza_iperparametri`). Controllo: nello stesso job dei latenti di calibrazione si rifanno i
latenti delle 150 patch di test e si confrontano con quelli riusati, tolleranza `ei.TOL['Z']` = 5e-3 (`controls/`).

## 3. Varianti del nostro (per seme, s1234 e s2345 separatamente)

u = le 256 componenti di forma del latente, S = exp(s) la taglia. Stima sulle 730 u di calibrazione (73 identita').
S_w = covarianza intra-identita' (N - C gradi di liberta'), S_b = fra identita' (pesata per numerosita'),
t = tr(S_w) / 256.

- **(a) WCCN**: d' = || (S_w + lam t I)^(-1/2) (u_i - u_j) ||, lam in {1, 1e-1, 1e-2, 1e-3, 1e-4}.
- **(b) LDA**: i k autovettori principali di S_b v = mu (S_w + lam t I) v (normalizzati: V^T (S_w + lam t I) V = I),
  d' = || V_k^T (u_i - u_j) ||; lam come sopra, k in {2, 4, 8, 16, 32, 48, 63} (63 = soggetti di stima in validazione
  incrociata - 1, per eccesso).
- **(c) taglia + u calibrata**: d_F' = sqrt((S_i - S_j)^2 + S_i S_j (g c0 d')^2) con d' di (a) o (b) alla configurazione
  gia' scelta per (a) o (b); c0 = mediana di d_P / mediana di d' sulle coppie di persone diverse dei soggetti di stima
  (porta d' nelle unita' di d_P), g in {2^(k/2), k = -8 ... 8}.
- **Controllo di (c)**: d_F con u NON calibrata e c = g scelto allo stesso modo su FaMoS TRAIN: separa l'effetto della
  calibrazione di u da quello di riscegliere il peso della taglia sul dominio.
- Di riferimento, invariati: d_F non cal. (lettura preregistrata), d_F cal. (c di `fact_calib`), d_P.

**Scelta**: validazione incrociata a 5 fold sui soggetti di calibrazione: stima su 4 fold, distanze sulle coppie interne
al fold tenuto fuori (~146 mesh), AUC (stessa persona contro persone diverse, `famos_eval.weighted_auc`, pesi 1), media
sui 5 fold; massimo, a pari merito il primo della griglia (lam piu' grande, k piu' piccolo, g piu' piccolo). Poi stima su
tutti i 73 soggetti con la configurazione scelta e applicazione agli embedding dei 150 di test. Tutte le configurazioni
e le AUC per fold vanno in `cv.csv`.

**Variante primaria** (per il verdetto, fissata ora): per ciascun seme, quella fra (a), (b), (c)-WCCN, (c)-LDA con l'AUC
di validazione incrociata piu' alta su FaMoS TRAIN. Le altre si riportano tutte.

## 4. Concorrenti: stessa calibrazione

ICP + Chamfer in mm e NoW metrical ricevono gli stessi soggetti e gli stessi fold per scegliere la configurazione sulla
griglia di `robustezza_iperparametri` (`ri.GRID`: ICP 9 configurazioni, NoW 7; funzioni `ri.run` / `ri.merge`
importate, regola di FaMoS: X = indice minore, seme `blmm.mesh_seed`, landmark rilevati sulla patch, media dei due
versi per NoW). Criterio identico: AUC sulle coppie interne a ciascun fold, media sui 5 fold, massimo (a pari merito il
default, primo della griglia). Non stimano nulla, quindi non serve la stima fuori fold. Al test si usano le distanze gia'
calcolate da `robustezza_iperparametri` per la configurazione scelta. Si riportano anche i numeri senza calibrazione
(default della matrice, `esterna_famos_id`).

## 5. Valutazione (15 soggetti FaMoS TEST)

- Le 150 scansioni e le 11.175 coppie di `esterna_famos_id`, le regole di `ei.evaluate`: rank-1, mAP (galleria = le
  altre 149), AUC, EER; distanza non finita = +inf; IC 95% percentile con 1000 repliche per soggetto,
  `famos_eval.bootstrap_counts(15, 1000, 1234)` (stesso seme); delta appaiati sulle stesse repliche, positivo = la
  variante meglio (EER: riferimento - variante).
- **Strati d'espressione** di `famos_dominio`: terzili di `expr_mm` (0,862 / 3,315 mm), coppia nello strato della
  scansione piu' espressiva; AUC_k = coppie della stessa persona dello strato k contro TUTTE le coppie di persone diverse;
  rank-1_k = query dello strato. Si riportano L, M, H; l'alto e' H.
- Controlli: metriche delle colonne di base ricalcolate = `esterna_famos_id/metrics.csv` (1e-12, altrimenti il passo si
  ferma); d_F ricalcolata dai latenti riusati = distanze di `ei.py`; default della griglia = colonne di `ei.py`.
- Incertezza NON inclusa negli IC: la scelta dei soggetti di calibrazione (gli IC ricampionano solo i 15 di test).

## 6. Lettura (fissata ora, descrittiva)

Per seme, sulla variante primaria, metrica AUC (dove sta il divario), globale e nello strato H:
- **guadagno**: IC del delta (primaria - d_F non cal.) tutto sopra 0;
- **divario chiuso**: IC del delta (primaria - concorrente) che include 0 o sta sopra, contro ICP mm e NoW metrical
  ciascuno nella configurazione scelta su calibrazione E nel default;
- **la calibrazione non basta**: se il delta contro ICP mm o NoW metrical resta con IC tutto sotto 0 in almeno un seme.
Rank-1, mAP ed EER si riportano accanto, senza verdetto.

## 7. Uscite

`aau/runs/evidence/famos_calib/`: `REGOLE.md`, `selection.json`, `folds.json`, `spec_f*.json`, `prep_calib.json`,
`controls/*.json`, `cv.csv`, `metrics.csv`, `deltas.csv`, `summary.json`, `results.md`. Registrazioni, patch, latenti e
distanze in `datasets/FAMOS/calib/` (fuori da git, licenza MPI). Nessuna mesh in git. Correzioni di bug dopo questo
commit: in results.md con commit e motivo.
