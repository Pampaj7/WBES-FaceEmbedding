# Diagnostica D: cosa limita la metrica appresa (protocollo, scritto PRIMA dei numeri)

Scritto l'11 ottobre 2026 dal coder, su richiesta del PI, PRIMA di qualunque Spearman, delta o embedding nuovo di
questo protocollo. Codice in `aau/diagnostics/`, evidenze in `aau/runs/evidence/diagnostics/`. L'impronta sha256 di
questo file sta nel messaggio del commit che lo introduce e in `PROTOCOL_D.sha256`; ogni modifica va in un emendamento
datato. E' una DIAGNOSTICA: nessun numero di questo protocollo e' un metodo o un risultato del paper.

**Cosa esisteva gia' al momento della scrittura** (nessuno di questi passi calcola uno Spearman o un delta nuovo):
- sonda tecnica sui generatori (job 1067906, `aau/scratch/diag_probe/probe_regen.py`): le sorgenti `ict`, `gnm`,
  `flame2023` di `v3_work/stream/sources.py` (patch di lavoro = patch nativa: ICT 9.409 v. / 18.460 tri., GNM 9.022 /
  17.720, FLAME 2023 1.787 / 3.408; FLAME 2023: frame canonico = m -> mm, R = I, nessun capovolgimento); due held-out
  ICT (id10014, id14921) e due GNM (id110000, id110099) rigenerati dai loro pesi con `views.discretize` coincidono con
  le mesh statiche del training per original, down8k, remesh (stesse facce, max |dV| <= 2.6e-5 unita' native), NON per
  up60k (facce diverse: la decimazione qslim non e' riproducibile bit per bit); FLAME 2023 suddivisa 1-a-4 una volta:
  6.986 v. / 13.632 tri.; down8k nativa 630 v., suddivisa 2.445 v.;
- sonda tecnica D2 (job 1067949, `probe_d2.py`): i vettori GT per soggetto ricostruiti con `cgt.native_points` +
  la fattorizzazione di `train_fr_sr.py` riproducono `datasets/CANONICAL_GT/eval/{facescape,hifi3d}_{fr,sr}.npz`
  (rapporto costante, scarto relativo <= 3.6e-7); `fact_paired.rows_for` da' 99.000 righe `nocrop_cross` (100
  soggetti, coppie non ordinate) per FaceScape e HIFI3D; gli embedding ufficiali hanno le 600 chiavi (con crop);
- i numeri gia' pubblicati nel repo e citati dal PI (E1, `factorized_results.md`, concorrenti parametrici).

## 1. Domande

Testa `factorized`: d_F^2 = (S_i - S_j)^2 + S_i S_j (c d_P)^2, d_P = ||u_i - u_j|| x dp_per_unit (forma).
- **H1 VARIETA'**: il limite e' la copertura dei generatori: i bracci leggono bene la forma dei generatori visti e
  male quella di un generatore mai visto.
- **H2 LETTURA**: l'encoder ammortizzato non estrae la forma fine nemmeno in distribuzione: sui generatori visti un
  fit 3DMM col modello nel ciclo (variante B, con un prior che i bracci non hanno visto) fa meglio.
- **H3 TESTA contro ENCODER**: l'informazione c'e' nell'embedding, ma la distanza addestrata sui sintetici non la usa:
  una sonda lineare dall'embedding congelato alla forma GT ordina meglio di d_P.
Non esclusive: D1 risponde a H1 e H2, D2 a H3.

## 2. Bracci e distanze

| chiave | checkpoint | c (d_F calibrata) |
|---|---|---|
| `factorized_s1234` | C3F, `epoch072_ema.pth` (`fact_calib.checkpoint`) | 0.405 |
| `factorized_s2345` | idem, seme 2345 | 0.405 |
| `factorizedc3m_e123` | C3M, `epoch123_ema.pth` | 0.257 |

c = `c_median` di `aau/runs/evidence/trainer_v3/factorized_calibration.csv` (emendamento 4 di factorized). Distanze
come `fact_paired.distances` (copiata): S = exp(z_0), d_P = ||z_i[1:] - z_j[1:]|| x dp_per_unit (json di `--dist_npz` /
`gt_scale` del checkpoint), d_F calibrata `form_cal` = sqrt((S_i - S_j)^2 + S_i S_j (c d_P)^2). Si legge d_P con la GT
SR e `form_cal` con la GT FR. **Le regole valgono per factorized s1234 e s2345 (servono entrambi i semi); C3M e123 e'
riportato accanto, descrittivo**, come in factorized_protocol. Avvertenza: c e' stata calcolata (mediana del rapporto,
un solo scalare) sugli stessi held-out visti di D1; tocca solo `form_cal`, non d_P.

## 3. D1, gap di generatore (nessun training)

### 3.1 Insiemi

| insieme | generatore | soggetti | ruolo |
|---|---|---|---|
| `bfm`, `ict`, `gnm` | held-out della calibrazione c (`fact_calib.heldout_subjects`: BFM REMESH, ICT-5000, GNM_DISTILL id110000-110099) | 100 ciascuno | **visti** (training di C3F e C3M) |
| `regen_ict`, `regen_gnm` | gli STESSI 100 + 100 held-out, rigenerati col codice di D1 (sez. 3.3) | 100 ciascuno | controllo di compatibilita' C1 |
| `flame2023_s1` | FLAME 2023 Open, suddiviso 1-a-4 una volta | 200 | **mai visto, primario** |
| `flame2023` | FLAME 2023 Open, patch nativa (la topologia di lavoro dello stream) | gli stessi 200 | mai visto, secondario |

Etichette: original, remesh, down8k, noisy, up60k (niente crop), viste neutre. Visti: mesh, operatori ed embedding
della calibrazione (`datasets/V3_OPS_CACHE/heldout_calib`, `aau/runs/evidence/trainer_v3/factorized/calib_heldout/
<chiave>/embeddings.npz`), GT di training di C3F (`ablations/c3f/gt_sr.npz` x `dP_per_unit` per SR, `gt_frcal.npz` per
FR: dentro un dominio e' la FR grezza per una costante, lo Spearman non cambia).

**Perche' FLAME suddiviso e' il primario.** La patch FLAME ha 1.787 vertici contro 9.022-9.409 di GNM e ICT (e down8k
630 contro 3.163-3.281): con la patch nativa il generatore nuovo cambia insieme alla risoluzione, che il training dei
bracci non ha mai coperto cosi' in basso. La suddivisione 1-a-4 a punto medio (`igl.upsample`, una iterazione) non
cambia la superficie (lineare a tratti, stessa GT) e porta la original a 6.986 vertici: isola il generatore. La patch
nativa e' la pipeline letterale dello stream e si riporta accanto; la differenza suddivisa - nativa e' l'effetto della
risoluzione (descrittivo).

### 3.2 FLAME 2023 Open

`sources.MMSource("flame2023")` (maschera `face`, 300 modi d'identita', metri). Identita' k = 0..199:
z_k = `model.sample_identity(rng_k, tails=False, trunc=0.0, purpose="eval")`, rng_k = `default_rng(SeedSequence(
[20261111, k]))`: N(0, 1) non troncata sui 300 modi, **il campionatore dei generatori visti** (ICT-5000, ICT_SCALE,
GNM_DISTILL: N(0, 1) non troncata), non le code pesanti dello stream, che i bracci non hanno mai visto. Mesh di lavoro
V = media + base x z_k (patch, unita' native), facce della sorgente; `flame2023_s1`: (V, F) = `igl.upsample(V, F, 1)`.
Discretizzazioni: `views.discretize(V, F, etichetta, seme)` (le funzioni di `make_ict_topologies.py`, come i dati
di training), seme di `noisy` = `int(SeedSequence([20261112, k]).generate_state(1)[0])`. Nomi `id700000+k` (nativa)
e `id710000+k` (suddivisa), fuori da tutti gli intervalli di id del repo.

GT: `targets.CanonTargets()("flame2023", P_k)` con P_k = `src.neutral_points(z_k)` (la GT al volo dello stream, lo
stesso codice di `train_fr_sr._factor_chunk`): d_FR = ||fr_i - fr_j|| / sqrt(A) in mm, d_P = ||sr_i - sr_j|| / sqrt(A).

Geometria grezza nel frame NATIVO di FLAME (metri; il frame di E12 `flame` ha u = 1000, R = I): tabella di scala come
`fact_calib.stage` (area_mm2 = u^2 x area grezza, dominio `flame` per la rotazione di `global_v3`), operatori
`areanorm_operators.py --k-eig 128`, embedding `eval_v3.py -- zs_embed.py` con gli argomenti di `calib_heldout.sbatch`
(`WBES_V3_FACTORIZED_OUT=full`, `WBES_EVAL_SCENARIOS=clean`). Mesh e operatori fuori da git, cancellati a fine lavoro.

### 3.3 Controllo di compatibilita' C1 (rigenerazione)

I 100 held-out ICT e i 100 GNM della calibrazione, coi loro pesi (`domains.ict_weights`, `domains.gnm_weights`),
passano per lo stesso codice di FLAME (sorgente dello stream, `views.discretize`, tabella, operatori, embedding), coi
semi di `noisy` statici (ICT-5000: `int(id[-4:])`; GNM: `int(SeedSequence([20261011, id - 100000]).generate_state(1)
[0])`), frame nativo (ICT: cm, `ict`; GNM: m, `gnm`), GT da `CanonTargets`. Atteso: original, remesh, down8k, noisy
identiche alle statiche, up60k equivalente ma non identica.
- **C1-GT**: d_P di `CanonTargets` contro `gt_sr.npz` x `dP_per_unit` sulle stesse coppie: scarto relativo massimo
  <= 1e-4; FR contro `gt_frcal.npz`: rapporto costante per dominio entro 1e-4 relativo.
- **C1-emb**: per le etichette deterministiche, max |z_rigenerato - z_statico| (riportato, nessuna soglia).
- **C1-rho**: per ogni braccio e dominio, |rho(regen) - rho(statico)| <= 0.02 con SR (d_P) e con FR (`form_cal`), sulle
  stesse coppie e repliche.
Se C1 passa, gli held-out statici sono compatibili e le letture R1/R2 li usano come visti (come chiesto dal PI). Se non
passa: R1/R2 si rifanno con `regen_ict`, `regen_gnm` come visti (bfm escluso, non rigenerabile) e lo si dichiara.

### 3.4 Righe, statistiche

Righe: per insieme, coppie di mesh i < j (nomi ordinati), soggetti diversi, etichette diverse (`fact_calib.calib_one`).
Spearman (ranghi medi sui pari merito) con bootstrap per soggetto: 1000 repliche, conteggi multinomiali per soggetto
(`bincount(rng.integers(0, n, n))`, come `fact_paired.main`), peso di riga c_a c_b (righe ripetute, come
`fact_paired._rep`), IC 95% percentile. Semi per gruppo di identita', `default_rng(SeedSequence([20261113, g]))`:
g = 0 `bfm`; 1 `ict` e `regen_ict` (stessi conteggi, soggetti nello stesso ordine); 2 `gnm` e `regen_gnm`; 3
`flame2023_s1` e `flame2023` (stessi conteggi per indice k). Delta dentro un insieme: stesse righe e repliche. Delta
fra insiemi: replica b di ciascun insieme (insiemi di soggetti diversi: indipendenti).

### 3.5 Variante B (concorrente parametrico)

`aau/baselines_param` importato, non modificato: variante B senza espressione (`free=False`, come gli held-out
dell'emendamento 2) coi valori congelati `bp.LOOP` dell'emendamento 1 (GNM sigma 2 mm, tau 10 mm, 5 iterazioni;
FLAME 2023 sigma 2 mm, tau 5 mm, 5 iterazioni). Distanze `bp.mesh_distances` sulle mesh d'identita': `vb_sr` con SR,
`vb_fr` con FR.
- Visti: i fit gia' calcolati di `baselines_param/calib_e2/{gnm,flame2023}.npz` (1.500 mesh, 0 fallimenti; regione
  per dominio dalle 100 original), distanze ricalcolate dai beta e dalle regioni salvate.
- FLAME (`flame2023_s1`, `flame2023`): fit nuovi con entrambi i modelli; regione per (insieme, modello) con
  riferimenti = le 200 original dell'insieme (`bp.region`, come l'emendamento 2); L = mediana del maxabs delle
  original in mm (`blmm.mesh_scalars`, come `fact_calib.baselines`); semi `blmm.mesh_seed`.
- Etichette: **B-FLAME su bfm/ict/gnm e B-GNM su bfm/ict/FLAME = incrociati** (prior che i bracci non hanno visto o
  generatore diverso dal prior: confronti equi); **B-GNM su gnm e B-FLAME su FLAME = "prior esatto"**, solo
  riferimento, mai nelle regole.
- Fallimenti (`bp_fit_e1.failure`): le righe della mesh escono per TUTTI i metodi dell'insieme (maschera comune per
  insieme), si contano; oltre il 5% di righe perse si riportano anche i bracci sulle righe complete.

### 3.6 Letture di D1 (dichiarate qui, prima dei numeri)

**R1 VARIETA'.** Delta_gen = media su {bfm, ict, gnm} di rho(d_P, SR) - rho(d_P, SR) su `flame2023_s1`. **La varieta'
e' una leva forte se Delta_gen >= 0.10 e l'IC 95% esclude lo 0 (estremo inferiore > 0), in ENTRAMBI i semi.** Un seme
solo: non risolto, si dice quale. Accanto, descrittivi: lo stesso con `flame2023` nativo; l'effetto della risoluzione
rho(`flame2023_s1`) - rho(`flame2023`) (appaiato per identita'); Delta_gen con FR (`form_cal`); C3M.
Se R1 passa sul nativo e non sul suddiviso, la lettura e' "gap di risoluzione/discretizzazione, non di generatore".

**Controllo di difficolta' (descrittivo).** Uno Spearman piu' basso su FLAME puo' venire dall'insieme (spettro della
forma diverso) e non dalla copertura dei bracci. B-GNM e' incrociato su bfm, ict e FLAME e non e' tarato su nessuno:
Delta_B = media su {bfm, ict} rho(B-GNM `vb_sr`) - rho su `flame2023_s1`; per i bracci Delta' = la stessa differenza
con d_P; differenza delle differenze DiD = Delta' - Delta_B con IC. DiD vicino a 0 = il calo su FLAME e' dell'insieme,
non dei bracci. Riportati anche mediana e IQR di d_P GT per insieme.

**R2 LETTURA.** Delta_read = media su {bfm, ict, gnm} di [rho(d_P, SR) - rho(B-FLAME `vb_sr`, SR)] (appaiato dentro
ogni insieme). **Limite di lettura anche senza cambio di dominio se l'IC 95% sta sotto 0 (estremo superiore < 0) in
entrambi i semi.** Riportati per insieme, e con FR (`form_cal` contro `vb_fr`, descrittivo). Su FLAME: bracci contro
B-GNM (incrociato, descrittivo) e B-FLAME (prior esatto, riferimento).

**Valori assoluti in distribuzione**: rho dei bracci su bfm, ict, gnm con SR e FR (quanto sono buoni sui loro stessi
generatori), accanto a B.

## 4. D2, testa contro encoder (nessun training dei bracci)

**Diagnostica, non un metodo**: la sonda e' addestrata con la GT del dominio dev.

- Domini: **FaceScape dev (primario)**; HIFI3D solo descrittivo e secondario.
- Righe e GT: quelle di `factorized_paired.csv`: `fact_paired.rows_for(vista)` e la maschera comune di
  `fact_paired.main` (colonne di tutti i bracci, baseline e GT finite; importato, non modificato): FaceScape 88.725 delle
  99.000 coppie `nocrop_cross`, HIFI3D 98.224; 100 soggetti; colonne `gt_sr`, `gt_fr`.
- Ingresso: gli embedding degli store ufficiali (`fact_paired.embeddings`, `form_devfs` / `form_hifi`), z = [s, u]
  (257 numeri), le 500 mesh senza crop; standardizzati (media e deviazione delle mesh di training).
- Bersaglio per soggetto: i vettori GT della sonda tecnica (`cgt.native_points` + fattorizzazione di `train_fr_sr`):
  SR = sqrt(w) z_i (4.434 numeri, distanza = d_P GT), FR = sqrt(w) a_i (mm, distanza = d_FR GT); controllo K2: le loro
  distanze riproducono `gt_sr` / `gt_fr` delle righe (rapporto costante entro 1e-5).
- Sonda: PCA dei bersagli dei soggetti di training (tutte le componenti con valore singolare > 1e-10 del massimo:
  equivale alla regressione sul vettore intero), ridge dall'embedding standardizzato ai punteggi, lambda nella griglia
  `logspace(-1, 5, 25)` scelto con CV interna a 5 fold PER SOGGETTO sui soli soggetti di training (minimo dell'errore
  quadratico dei bersagli), poi rifit su tutto il training. Distanza = euclidea fra i bersagli previsti (scala della
  GT); Spearman con la GT sulle righe di test (entrambi i soggetti nel fold di test).
- CV esterna: 2 fold PER SOGGETTO (50 + 50, soggetti mai insieme in training e test), ripetuta R = 50 volte, split
  `default_rng(SeedSequence([20261114, r]))`. Per ripetizione: media dei due fold. Stima puntuale: media sulle 50
  ripetizioni. Riferimento sulle STESSE righe con la stessa media: d_P (con SR) e `form_cal` (con FR) dello stesso
  braccio.
- IC: bootstrap per soggetto dell'INTERA procedura, 1000 repliche `default_rng(SeedSequence([20261115, b]))`: 100
  soggetti estratti con reinserimento, uno split a 2 fold dei soggetti distinti, pesi c_s nel training (standardizzazione,
  PCA, ridge, CV interna) e c_a c_b sulle righe di test; delta = rho(sonda) - rho(riferimento) per replica; IC 95%
  percentile. Lo split cambia a ogni replica: l'IC comprende anche la variabilita' dello split (conservativo rispetto
  alla media su 50 split della stima puntuale).
- Controlli: K1 permutazione (bersagli permutati fra i soggetti di training, 50 ripetizioni `SeedSequence([20261116,
  r])`: rho atteso ~ 0); K2 sopra; K3 rho di d_P e `form_cal` su tutte le righe della maschera = `arm_point` di
  `factorized_paired.csv` (scarto <= 1e-6). Si riporta la frequenza di lambda sul bordo della griglia.

**R3 TESTA.** Delta_probe = rho(sonda SR) - rho(d_P) su FaceScape. **Se Delta_probe >= 0.05 e l'IC 95% esclude lo 0
(estremo inferiore > 0) in entrambi i semi: l'informazione c'e' (linearmente decodificabile), il limite e' la testa o la
sua taratura sintetica. Altrimenti: nessun guadagno decodificabile linearmente con 50 soggetti di training, il limite
e' a livello dell'encoder (per quanto puo' dirlo una sonda lineare).** Semi discordi: non risolto. Con FR (sonda FR
contro `form_cal`), HIFI3D e C3M: descrittivi.

## 5. Uscite

`aau/runs/evidence/diagnostics/`: `d1/` (tabelle di scala, GT, embedding, fit B: npz fuori da git; `controls.json`),
`d1_spearman.csv` (rho per insieme, metodo, GT), `d1_delta.csv` (delta e letture), `d2_probe.csv`, `d2_controls.json`,
`results.md` (tabelle, esito di R1, R2, R3, tempi e job). Mesh grezze in `datasets/DIAG_D1/` (fuori da git),
cancellate a fine lavoro con gli operatori.

## 6. Fuori da questo protocollo

Crop, espressioni, FaceVerse, FaMoS; altri bracci (ctrlfr, dual, factorized2, C3M e205); varifold e le altre
baseline; generatori diversi da FLAME 2023 Open; sonde non lineari; qualunque training o modifica di `v3_work/stream`,
`aau/baselines_param`, `fact_paired.py`, `fact_summary.py`, `paper/`.
