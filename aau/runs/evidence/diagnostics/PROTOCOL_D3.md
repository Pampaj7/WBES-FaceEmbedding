# Diagnostica D3: una testa metrica lineare trasferisce fra domini? (protocollo, scritto PRIMA dei numeri)

Scritto l'11 ottobre 2026 dal coder, su richiesta del PI, PRIMA di qualunque Spearman, delta, testa, GT o embedding
nuovo di questo protocollo. Codice in `aau/diagnostics/` (file `d3_*`), evidenze in `aau/runs/evidence/diagnostics/`.
L'impronta sha256 di questo file sta nel messaggio del commit che lo introduce e in `PROTOCOL_D3.sha256`; ogni modifica
va in un emendamento datato. E' una DIAGNOSTICA: dice se una testa metrica lineare sopra l'encoder congelato e' una leva
per il paper; nessun numero di questo protocollo e' un metodo o un risultato del paper.

**Cosa esisteva gia' al momento della scrittura** (nessuno di questi passi calcola uno Spearman, una testa o una GT):
- D1 e D2 (`PROTOCOL_D.md`, emendamento 1, `results.md`, commit 81e5690): embedding e GT degli held-out visti (`bfm`,
  `ict`, `gnm`, quelli della calibrazione di c) e di FLAME 2023 Open suddiviso (`d1/emb/<braccio>/embeddings.npz`,
  `d1/gt_flame2023_s1.npz`: npz fuori da git, ancora su disco); la sonda di D2 (FaceScape SR 0.889 contro d_P 0.745);
- i numeri pubblicati di `trainer_v3/factorized_paired.csv` e `baselines_param/paired_e1.csv` (righe, semi, valori e
  IC dei bracci e di B sulle viste di test): qui servono solo da controllo (sez. 7), non per scegliere;
- sonde tecniche del coder di oggi (job 1068036, 1068037, 1068039; nessun numero di D3): forma dei file di GT di eval
  e degli store di embedding; `sources.FamosSource` carica le 80 persone TRAIN (patch FLAME `face` 1.787 v. / 3.408
  tri. in mm, normale media +z come la media FLAME 2023 di D1), `igl.upsample` la porta a 6.986 v. / 13.632 tri.; le
  original delle viste di test hanno 9.518 (HIFI3D), 12.596 (FaceScape) e 28.632 (FaceVerse) vertici; sklearn 1.5.2
  nel container;
- NON esistono: embedding e GT di FaMoS TRAIN; embedding di C3M e123 su FaceVerse NEUTRA (lo store neutro
  `aau/runs/evidence/faceverse_neutral/embed` ha C3M solo all'epoca 205). Si calcolano qui (sez. 3, 6).

## 1. Domanda

D2 ha mostrato che la forma reale e' decodificabile linearmente dall'embedding congelato DENTRO un dominio (sonda
addestrata sulla GT di FaceScape). D3 chiede se una **testa metrica lineare** appresa con la GT di un dominio X, senza
riaddestrare l'encoder, **ordina meglio di d_P su un dominio Y diverso da X**. Se trasferisce fra domini reali, e' la
leva piu' economica per il paper; se trasferisce solo dentro il dominio, no. Due domande accanto: la varieta' sintetica
nella sola testa basta (d contro c)? Un allineamento NON supervisionato al dominio di test basta (e)?

## 2. Bracci, embedding, distanze di riferimento

| chiave | checkpoint | ruolo |
|---|---|---|
| `factorized_s1234` | C3F, `epoch072_ema.pth` (`fact_calib.checkpoint`) | regole |
| `factorized_s2345` | idem, seme 2345 | regole |
| `factorizedc3m_e123` | C3M, `epoch123_ema.pth` | descrittivo |

Embedding z = [s, u] (257 numeri) degli store ufficiali (o calcolati con la stessa pipeline, sez. 3). **La testa lavora
su u** (256 numeri, il vettore di d_P), in unita' di d_P: x = u x dp_per_unit (`diag.dp_per_unit`: json di `--dist_npz`
/ `gt_scale` del checkpoint). Riferimenti dello stesso braccio sulle stesse righe (`diag.arm_distances`, copia di
`fact_paired.distances`): d_P = ||x_i - x_j|| (`shape`, letta con la GT SR) e d_F calibrata `form_cal` = sqrt((S_i -
S_j)^2 + S_i S_j (c d_P)^2), S = exp(s), c = `c_median` di `trainer_v3/factorized_calibration.csv` (letta con la GT FR).
**B** (variante B, modello nel ciclo, dell'emendamento 1 dei concorrenti parametrici), GNM e FLAME 2023 Open, distanze
fra mesh d'identita' `vb_sr` (con SR) e `vb_fr` (con FR) da `baselines_param/<vista>/<modello>/fit_e1.npz`: solo
riferimento, mai nelle regole.

## 3. Domini di training della testa

| testa | dominio | soggetti x etichette | embedding | GT d_P |
|---|---|---|---|---|
| (a) `facescape` | dev FaceScape, i 100 soggetti valutati (dominio dev) | 100 x 5 | store `form_devfs` (`fact_paired.embeddings`) | `CANONICAL_GT/eval/facescape_sr.npz` x `dP_per_unit` del suo json |
| (b) `famos` | FaMoS TRAIN, le 80 persone di `aau/famos/split.json` | 80 x 5 | NUOVI (sotto) | NUOVA (sotto) |
| (c) `seen` | held-out visti di D1: `bfm`, `ict`, `gnm` | 3 x 100 x 5 | `trainer_v3/factorized/calib_heldout/<braccio>` | `ablations/c3f/gt_sr.npz` x `dP_per_unit` (come D1) |
| (d) `seen_flame` | (c) piu' `flame2023_s1` di D1 (FLAME 2023 Open, N(0, 1), suddiviso) | (c) + 200 x 5 | (c) + `d1/emb/<braccio>` | (c) + `d1/gt_flame2023_s1.npz` (`D_sr`) |

Etichette: original, remesh, down8k, noisy, up60k (niente crop, come le righe di test). Le GT d_P di tutti i domini
sono la stessa grandezza (distanza di Procrustes fra pre-forme a centroid size 1 sulla regione unificata,
`train_fr_sr.py` / `targets.CanonTargets`; per (c) e FLAME il controllo C1-GT di D1).

**(b) FaMoS TRAIN.** `sources.FamosSource` (importato, non modificato): per persona p la neutra di riferimento
registrata FLAME (`V_neutral`, mm) sulla maschera FLAME 2020 `face`, allineata rigidamente senza scala alla patch media
FLAME canonica (`view_mesh(p, rng, expr=False)`); poi **suddivisa 1-a-4 una volta** (`igl.upsample`, la stessa
superficie: la GT non cambia), come `flame2023_s1`, il primario di D1: la patch FLAME nativa ha 1.787 vertici contro i
9.518-28.632 delle original delle viste di test e D1 ha misurato che la sola risoluzione costa ai bracci 0.015-0.023
di SR. E' l'unica differenza dalla pipeline delle viste reali, dichiarata qui. Discretizzazioni `views.discretize`
(le funzioni di `make_ict_topologies.py`, come D1 e le viste reali), seme di `noisy` = `int(SeedSequence([20261120,
NNN]).generate_state(1)[0])`, NNN = numero della persona FaMoS; nomi `id<720000 + NNN>_GTready_<etichetta>.npz`
(fuori da tutti gli intervalli di id del repo). Tabella di scala: area_mm2 = area della mesh discretizzata (gia' in
mm, u = 1), dominio `famos` (`global_v3.EXTRA_FRAMES`: u = 1, R = I; la mesh e' gia' nel frame canonico). Operatori
`areanorm` k_eig 128 (`areanorm_operators.py`), embedding `eval_v3.py -- zs_embed.py` con gli argomenti di
`embed.sbatch` di D1 (`WBES_V3_FACTORIZED_OUT=full`, `WBES_EVAL_SCENARIOS=clean`). GT: `targets.CanonTargets()("famos",
P_p)`, P_p = `FamosSource.neutral_points(p)` (la GT al volo dello stream, lo stesso codice di
`train_fr_sr._factor_chunk`, come D1 per FLAME): d_P = ||sr_i - sr_j|| / sqrt(A), d_FR = ||fr_i - fr_j|| / sqrt(A) in
mm. Nessun file delle persone di TEST viene letto per costruire mesh, GT o embedding (la GT e' per identita': rigida
robusta verso mu, fissa).

**Pesi.** In (c) e (d) le coppie stanno dentro un generatore (come D1) e ogni generatore pesa lo stesso nella perdita
(1/3 o 1/4 del totale), qualunque sia il numero di soggetti (FLAME ne ha 200).

## 4. La testa

**Famiglia.** d_h(i, j)^2 = alpha^2 ||x_i - x_j||^2 + ||W (x_i - x_j)||^2, alpha = e^a > 0, W in R^{r x 256}: una
Mahalanobis M = alpha^2 I + W^T W (= W~^T W~ con W~ = [alpha I; W]), cioe' d_P piu' una correzione di rango r.
Perche' questa: (i) contiene d_P (W = 0 da' gli stessi ranghi), quindi la CV puo' scegliere "nessuna correzione" e il
termine di regolarizzazione restringe verso d_P, non verso 0; (ii) con r fino a 64 copre la capacita' della sonda di
D2 (ridge sui bersagli di 50 soggetti: rango <= 49); (iii) il gradiente in forma chiusa costa O(n^2 r) per
valutazione (la somma sulle coppie scritta col laplaciano della matrice dei coefficienti), quindi la CV costa poco.

**Perdita: sulle DISTANZE (stress), non sui ranghi.**

    L(a, W) = sum_p w_p (d_h,p - g_p)^2 / sum_p w_p g_p^2 + lambda ||W||_F^2 / alpha0^2

Perche': (i) e' la famiglia di perdita dei bracci (stress sulle distanze, `losses_v3` v2); (ii) la GT e' una distanza
(scala di rapporti): la regressione usa tutta l'informazione, i ranghi ne buttano via; (iii) i ranghi entrano dove
contano: la CV sceglie r e lambda sullo Spearman, la metrica di lettura; (iv) gradiente liscio e ottimizzazione
deterministica, mentre una perdita sui ranghi chiede di campionare coppie di coppie (O(P^2)).
- Coppie p: mesh dei soggetti di fit, stesso generatore, soggetti diversi, etichette diverse (`fact_calib.calib_one`);
  g_p = GT d_P dei due soggetti; w_p = 1 / (n_gen |P_gen|).
- alpha0 = sum w g ||dx|| / sum w ||dx||^2 (scala ai minimi quadrati di d_P sulle coppie di fit).
- Inizio: a = log alpha0, W = 0.1 alpha0 V_r, V_r = le prime r direzioni principali (vettori singolari destri) delle x
  delle mesh di fit, centrate per generatore. (W = 0 e' stazionario in W: l'inizio non puo' essere 0.)
- Ottimizzatore: `scipy.optimize.minimize(method="L-BFGS-B")`, gradiente analitico, float64, `maxiter` 2000, gli altri
  valori di default; deterministico. Convergenza e iterazioni si riportano.
- Griglia: r in {2, 4, 8, 16, 32, 64} x lambda in {1e-4, 1e-3, 1e-2, 1e-1, 1}, piu' r = 0 (d_P da sola, alpha = alpha0):
  31 configurazioni.

**CV per soggetto DENTRO il dominio di training.** 5 fold x 3 ripetizioni; ripetizione q: `rng =
default_rng(SeedSequence([20261121, q]))`, per generatore nell'ordine (`bfm`, `ict`, `gnm`, `flame2023_s1`) o per
l'unico dominio, `fold = rng.permutation(n_soggetti) % 5` sui soggetti ordinati (cosi' (c) e (d) hanno gli stessi fold
su bfm, ict, gnm). Fit sui soggetti degli altri 4 fold (alpha0, V_r e coppie dai soli soggetti di fit), punteggio =
Spearman di d_h con la GT d_P sulle coppie di validazione (entrambi i soggetti nel fold, stesse regole); con piu'
generatori, media degli Spearman per generatore. Punteggio di una configurazione = media sui 15 fit. Scelta: il
massimo; entro 1e-3 dal massimo, r minore e poi lambda maggiore. Scelta sul bordo della griglia (lambda 1e-4 o 1, r =
64): si riporta, la griglia non si allarga. Il punteggio di CV e' un punteggio di scelta (ottimista), non una stima.
**Testa finale**: rifit su tutti i soggetti del dominio con la configurazione scelta. Accanto, le 15 "teste di fold"
con la configurazione scelta (descrittive, sez. 6).

**FR.** d_F,h = sqrt((S_i - S_j)^2 + S_i S_j (c_h d_h)^2), S del braccio, c_h = mediana pesata (w) di g / mediana
pesata di d_h sulle coppie dell'intero dominio di training (la regola di `fact_calib`: mediana della GT / mediana del
modello; per (a), (b), (c) i pesi sono uguali e sono le mediane semplici). c_h viene SOLO dal dominio di training.

**Nessuna statistica del dominio di test** entra nelle teste (a)-(d): ne' standardizzazione, ne' PCA, ne' scala, ne'
c, ne' iperparametri; la funzione che addestra riceve solo gli array del dominio di training.

## 5. Variante non supervisionata (e), TRANSDUTTIVA

Usa le mesh del dominio di test (i loro embedding), MAI le loro etichette (ne' GT ne' soggetto).
- Covarianze di Ledoit-Wolf: covarianza a massima verosimiglianza delle u del dominio (centrate sulla loro media)
  ristretta verso mu I con il coefficiente di Ledoit-Wolf (la formula di `sklearn.covariance.ledoit_wolf_shrinkage`).
- Sorgente = le u del braccio sugli held-out visti (`bfm`, `ict`, `gnm`, 500 mesh ciascuno: il dominio su cui sono
  stati tarati d_P e c); Sigma_ref = media delle tre covarianze.
- Dominio di test D: Sigma_D = covarianza delle u delle 500 mesh senza crop dei 100 soggetti della vista (5
  etichette; FaceVerse: le mesh neutre o con espressione della vista).
- **(e) CORAL**: A_D = Sigma_ref^{1/2} Sigma_D^{-1/2} (radici simmetriche dalla decomposizione spettrale), d_e =
  ||A_D (u_i - u_j)|| x dp_per_unit; FR con c_e = mediana(g) / mediana(d_e) sulle coppie della sorgente con ogni
  generatore trasformato dalla sua A_d (la stessa trasformazione applicata alla sorgente, regola di `fact_calib`).
- **(e') sbiancamento** (descrittivo): A_D = Sigma_D^{-1/2}, c_e' allo stesso modo.
- Una sola trasformazione fissa per dominio; dichiarata transduttiva in ogni tabella.

## 6. Domini di test, righe, statistiche

| vista | righe | gruppo | seme del bootstrap | maschera |
|---|---|---|---|---|
| `hifi3d` | 98.224 | `nocrop_cross` | 990708 | comune di `fact_paired.main` |
| `facescape` | 88.725 | `nocrop_cross` | 796786 | comune di `fact_paired.main` |
| `faceverse_neutral` | 99.000 | `mesh_pair_nocrop` (righe di `faceverse`, mesh neutre) | 271049 | bracci, taglia oracolo, B e GT finiti (come `bp_paired`) |
| `faceverse` (con espressioni, secondaria) | 99.000 | `mesh_pair_nocrop` | 271049 | comune di `fact_paired.main` |

Righe, GT (`gt_sr`, `gt_fr`) e semi da `fact_paired.rows_for` (importato; per la neutra `rows_for("faceverse")` e gli
embedding dello store neutro, come `bp_paired`); semi e conteggi delle righe sono quelli pubblicati. Tutte le colonne di
D3 devono essere finite su queste righe (altrimenti errore). **Mai FaMoS TEST, mai Ava-256.**

**Teste per vista.** (a) su `hifi3d`, `faceverse_neutral`, `faceverse` (NON su `facescape`, il suo dominio di
training); (b), (c), (d), (e), (e') su tutte e quattro.

**Colonne** per (vista, braccio): d_P (SR), `form_cal` (FR), per ogni testa h: d_h (SR) e d_F,h (FR); per vista: B-GNM
e B-FLAME `vb_sr` (SR) e `vb_fr` (FR).

**Bootstrap per soggetto**: 1000 repliche coi conteggi di `fact_paired.main` per la vista (`default_rng(seme)`,
`bincount(rng.integers(0, n, n))`, soggetti ordinati): i valori e gli IC dei bracci e di B coincidono con quelli
pubblicati (controllo K1). Spearman pesato c_a c_b (`diag.wspearman`, uguale al campione con le righe ripetute). IC 95%
percentile, P(delta <= 0). **L'IC e' condizionato alla testa addestrata** (ricampiona i soggetti di TEST); la
variabilita' dal lato del training si mostra a parte: il delta puntuale delle 15 teste di fold (addestrate su 4/5 dei
soggetti, configurazione scelta) sulle stesse righe, minimo, mediana e massimo (descrittivo).

**Delta**, tutti appaiati (stesse righe e repliche): d_h - d_P (SR) e d_F,h - `form_cal` (FR) per testa, braccio e
vista; (d) - (c) (SR e FR); testa - B (descrittivo).

## 7. Controlli

- **K1 riferimenti**: rho (punto e IC) di d_P (SR) e `form_cal` (FR) dei tre bracci = `factorized_paired.csv`
  (`hifi3d`, `facescape`, `faceverse`) e `paired_e1.csv` (`faceverse_neutral`; C3M e123 neutro e' una cella nuova, senza
  riferimento); B `vb_sr` / `vb_fr` = `paired_e1.csv`; scarto massimo <= 1e-9, stesse righe.
- **K2 c**: la c dei bracci ricalcolata dagli embedding e dalla GT degli held-out con la regola di `fact_calib` =
  `c_median` (scarto <= 1e-9).
- **K3 GT FaMoS**: d_FR delle persone TRAIN contro il blocco train x train di `CANONICAL_GT/famos_F_rig_rob.npz` (E12;
  si legge solo quel blocco): scarto massimo <= 1e-3 mm.
- **K4 embedding nuovi**: FaMoS 400 mesh x 3 bracci, C3M e123 neutro 600 chiavi; Z finiti; checkpoint =
  `fact_calib.checkpoint(chiave)`.
- **K5 testa** (`test_d3.py`, dati finti, nessun dato valutato): gradiente analitico = differenze finite; recupero di
  una metrica piantata di rango basso (la testa batte d_P in validazione); r = 0 da' gli stessi ranghi di d_P;
  Ledoit-Wolf = sklearn; CORAL con Sigma_D = Sigma_ref e' l'identita'. Sui dati veri: in CV r = 0 riproduce lo Spearman
  di d_P (scarto <= 1e-12).
- **K6 nessuna fuga**: i fit di CV vedono solo i soggetti di fit (verificato nel codice a ogni fit); le teste (a)-(d)
  sono funzioni dei soli array del dominio di training; (e) usa solo gli embedding del dominio di test.

## 8. Letture (dichiarate qui, prima dei numeri)

**Soglie e perche'.**
- **SR: delta >= +0.05 con l'IC 95% che esclude lo 0** (estremo inferiore > 0). E' la soglia di R3 in D2; e' circa
  meta' del guadagno della sonda di D2 dentro il dominio (+0.14) e del gap di generatore di D1 (+0.11): un guadagno
  minore non giustifica un componente nuovo del metodo. L'IC che esclude lo 0 dice che il guadagno non si spiega col
  campionamento dei soggetti di test (condizionato alla testa).
- **FR: delta >= -0.03 sulla stima puntuale.** Una testa che migliora SR e peggiora FR non e' una leva usabile (FR e'
  la lettura principale del paper). 0.03 supera la differenza fra i due semi di `form_cal` sulle stesse viste (0.018
  su HIFI3D, 0.749 / 0.731; 0.008 su FaceScape, 0.661 / 0.669; `factorized_paired.csv`): una perdita maggiore non e'
  rumore di seme. Sulla stima puntuale perche' la richiesta e' "senza peggiorare oltre -0.03", non una non
  inferiorita'; l'IC si riporta.
- Le regole valgono per **entrambi i semi** (s1234, s2345); un seme solo: "non risolto". C3M e123 descrittivo.

**R_A trasferimento reale -> reale** (primaria): testa (a) (FaceScape) su HIFI3D.
- **PASSA**: in entrambi i semi delta SR >= +0.05 con IC > 0 E delta FR >= -0.03;
- **SR SI, FR NO**: la parte SR passa in entrambi i semi, quella FR fallisce in almeno uno;
- **NO**: la parte SR fallisce in entrambi i semi; **NON RISOLTO**: la parte SR passa in un seme solo.
Lettura: PASSA = la testa appresa su un dominio reale etichettato trasferisce (almeno FaceScape -> HIFI3D); NO = si
impara solo dentro il dominio. Accanto, con la stessa regola ma secondarie: (a) su `faceverse_neutral`; (b) (FaMoS)
su `hifi3d` e su `facescape`.

**R_S trasferimento sintetico**: (d) - (c) su HIFI3D, SR. **SI** se in entrambi i semi delta >= +0.05 con IC > 0
("la varieta' nella sola testa basta"); **NO** se in nessuno; altrimenti NON RISOLTO. Descrittivi: lo stesso su
`facescape` e `faceverse_neutral`, con FR, e (c) - d_P, (d) - d_P (atteso: (c) circa uguale a d_P).

**R_E non supervisionata** (TRANSDUTTIVA): (e) CORAL su HIFI3D, con la regola di R_A (PASSA / SR SI, FR NO / NO / NON
RISOLTO). (e') e le altre viste descrittive.

**Cosa cambierebbe** (dichiarato ora): R_A PASSA -> una testa addestrata su dati reali etichettati e' una leva, da
preregistrare come metodo (fuori da D3); R_A NO e R_S SI -> basta la varieta' sintetica nella testa; R_E PASSA ->
basta un allineamento non supervisionato, che pero' chiede le mesh del dominio di test; tutto NO -> il limite non e'
la testa ma l'encoder o la varieta' del suo training (R1 di D1).

## 9. Uscite, calcolo, pulizia

- Codice: `d3_famos.py` (mesh, tabella, GT di FaMoS TRAIN), `d3_head.py` (testa, CV, Ledoit-Wolf, CORAL), `d3_stats.py`
  (dati, CV, teste finali, valutazione, bootstrap, letture), `d3_summarize.py`, `test_d3.py`; `d3.sbatch` (passi CPU,
  partizione `prioritized`), `d3_embed.sbatch` (A100 di nv-ai-04, `--qos=unprivileged --requeue`, job breve e
  ripartibile: embedding di FaMoS TRAIN dei tre bracci; C3M e123 su FaceVerse neutra col passo `fvn` di
  `eval_body.sh`, come C3M e205, uscite in `d3/fvn` con la tabella di scala ufficiale della vista neutra).
- Evidenze in `aau/runs/evidence/diagnostics/`: `d3/` (npz fuori da git: GT ed embedding di FaMoS, teste, repliche;
  json in git), `d3_cv.csv` (punteggi di CV), `d3_heads.csv` (configurazioni scelte, c_h, convergenza),
  `d3_spearman.csv`, `d3_delta.csv`, `results_d3.md`. Dati derivati da FaMoS (licenza MPI): mai in git.
- Mesh di FaMoS in `datasets/DIAG_D3/` e operatori in `datasets/V3_OPS_CACHE/diag_d3/` (fuori da git), cancellati a
  fine lavoro; restano ricetta e semi.

## 10. Fuori da questo protocollo

Training o fine-tuning dell'encoder; teste non lineari; altri bracci (ctrlfr, dual, factorized2, C3M e205); crop;
FaMoS TEST e Ava-256; generatori diversi da quelli di D1; la scelta di un metodo per il paper (D3 e' una
diagnostica); qualunque modifica di `fact_paired.py`, `fact_summary.py`, `v3_work/stream/`, `aau/baselines_param/`,
`paper/`.
