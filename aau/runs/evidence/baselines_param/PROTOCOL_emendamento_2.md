# Concorrenti parametrici, emendamento 2 (POST HOC): B sul crop e su all_cross, composizione forma B + taglia B, costi misurati, sensibilita' a sigma

Scritto il 10 ottobre 2026 dal coder, su richiesta del PI, **DOPO aver visto i numeri** dell'emendamento 1
(`PROTOCOL_emendamento_1.md`, sha256 `3d06ab08...`; risultati in fondo a `results.md`, commit 6c50d3c, ec351f7). E'
un emendamento **post hoc**: la motivazione e' il verdetto RISERVE del critic sull'emendamento 1 (la variante B e'
reale, non un oracolo ne' un bug, ma il confronto e' incompleto), non i numeri in se'. Nulla delle analisi precedenti
cambia: protocollo e emendamento 1 restano come sono; le analisi di qui sono di sensibilita' e si leggono come tali.
L'impronta sha256 di questo file sta nel messaggio del commit che lo introduce e in `PROTOCOL_emendamento_2.sha256`.
I valori che escono da passi SENZA GT di test (k della sez. 2, esito del pilota della sez. 4) si aggiungono nella
sez. 9 con un commit successivo, PRIMA di ogni calcolo che legga una GT dei domini di test (passo `paired` della
sez. 8).

Regola di lettura per tutto l'emendamento: quella della sez. 7 del protocollo (delta a favore del braccio se l'IC 95%
sta sopra 0, contro se sta sotto, altrimenti non risolto; nessuna correzione per confronti multipli, descrittivo).

## 1. Variante B sulla topologia crop e sul gruppo all_cross

**Mesh nuove.** La topologia `crop` dei 100 soggetti valutati sulle viste che la hanno: `hifi3d`, `facescape`,
`faceverse`, `faceverse_neutral` (500 crop per vista nei dati, 100 dei soggetti valutati; FaMoS non ha crop). Entrambi
i modelli. Fit IDENTICO alla variante B dell'emendamento 1 (`bp_fit_e1.inputs`, `bp.fit_registered`, `bp.fit_loop`):
NICP col seme `blmm.mesh_seed(soggetto, "crop")`, regione di `fit.npz` della (vista, modello), variante A come
partenza (senza espressione sulle viste neutre, libera col prior dichiarato su `faceverse`), B coi valori congelati
`bp.LOOP` (sez. 8 dell'emendamento 1: GNM sigma 2 / tau 10 / 5 iterazioni, FLAME sigma 2 / tau 5 / 5). Fallimento come
l'emendamento 1 (eccezione, valori non finiti, RMS > 5 mm, corrispondenze tenute < 25% in un verso). Si salvano anche
l'errore di superficie di B (sez. 3 dell'emendamento 1) e i tempi per passo (sez. 3 qui). Uscita
`<vista>/<modello>/fit_e2_crop.npz`.

**Colonne.** `<modello>_vb_{coef,fr,sr}` sulle 600 mesh (beta di B delle 500 senza crop da `fit_e1.npz` + le 100 crop),
matrici ricalcolate con `bp.mesh_distances` sulle 600. Controllo: sulle coppie senza crop coincidono con le `D_vb_*` di
`fit_e1.npz` (scarto massimo riportato).

**Righe e repliche.** `fact_paired.py` non ha righe `all_cross` (la sua `rows_for` toglie il crop): si ricavano nello
stesso modo, senza il filtro, documentato qui.
- `hifi3d`, `facescape`: il gruppo `all_cross` di E12 (`blmm_eval.e12m.hifi_frames()[0]` / `fs_frames()`,
  `["all_cross"]`), con le colonne e le GT come `fact_paired.rows_for` (stesse chiamate, `be.add_columns`,
  `be.add_gts`) ma senza togliere il crop; seme del gruppo `all_cross` di E12 (quello di e108 - Chamfer eval:
  757683 per HIFI3D, 621096 per FaceScape). 148.500 righe (4.950 coppie di soggetti x 30 coppie di topologie diverse).
- `faceverse`, `faceverse_neutral`: E8/E12 non hanno un gruppo con crop per FaceVerse. Righe = le `pair_metrics` dello
  stage di `methods.fv_frames` (`ws_faceverse_expr/.../scale_e108_flip_topology`, `base.read_pair_metrics`) SENZA il
  filtro `NOCROP` di `zs_expr_summarize.secondary_frame`: 148.500 righe (soggetti diversi, topologie diverse, crop
  compreso); GT con `be.add_gts(df, "faceverse")`; seme `base.stable_seed(1234, "expr_sec", "scale_e108@bfm",
  "all_cross", "latent_distance", "raw_chamfer")` (la ricetta del seme di `mesh_pair_nocrop` col nome del gruppo
  cambiato). La vista neutra usa le stesse righe e GT (`faceverse_neutral/PROTOCOL.md`).
- Controllo: tolto il crop, le chiavi (soggetti, topologie) e le GT delle righe coincidono con quelle di
  `fact_paired.rows_for` della vista.
- Repliche: per soggetto, 1000, `default_rng(seme)`; Spearman e delta con `fact_paired._rep` / `fact_paired.summarize`
  (importati in sola lettura). Maschera COMUNE per dominio (righe con tutte le colonne finite).

**Bracci.** `fact_paired.model_columns(vista, idx)` sulle 600 chiavi (gli embedding dei bracci contengono il crop: 600
mesh per vista, verificato); vista neutra: gli embedding di `aau/runs/evidence/faceverse_neutral/embed` (600 mesh).
Baseline: solo la taglia oracolo (serve alla parziale).

**Cosa si riporta.** Spearman (rho, IC 95%) con FR e SR delle 6 colonne di B e dei bracci di riferimento
(`bp_paired.REF_ARMS`); delta braccio - colonna di B come la sez. 5 dell'emendamento 1 (con FR: factorized s1234 /
s2345 `form_cal` e ctrlfr s1234 / s2345; con SR: factorized `shape` e ctrlfr). Analisi secondaria, descrittiva: lo
stesso sul sottoinsieme delle righe con il crop su almeno un lato (49.500 righe; stesse repliche, stesso seme). Le
righe senza crop sono quelle dell'emendamento 1, gia' riportate.

## 2. Composizione dichiarata forma B + taglia B

Per modello, sulle mesh d'identita' di B (mu + B beta_B sulla regione, pesi d'area w di mu, come `bp.mesh_distances`):
- S_i = centroid size di B: sqrt(sum_v w_v ||a_i,v - c_i||^2 / sum w), c_i baricentro pesato (mm; e' la grandezza
  per cui SR scala ogni mesh);
- d_P,ij = SR_ij / S(mu): la distanza della mesh d'identita' SR di B (`D_vb_sr`, mm alla centroid size di mu) divisa
  per la centroid size di mu (adimensionale, a centroid size 1);
- **d_F,ij = sqrt((S_i - S_j)^2 + S_i S_j (k d_P,ij)^2)**, colonna `<modello>_vb_comp`.
Lo Spearman non dipende dall'unita' di S (i due termini scalano entrambi come S^2): conta solo k.

**k fissato senza test**, come k_ICP / k_NICP dell'emendamento 5 di factorized (`fact_calib.py bl`, `calib-bl`):
- mesh: gli held-out SINTETICI del training (100 soggetti per dominio bfm, ict, gnm, etichette down8k, noisy,
  original, remesh, up60k: 1.500 mesh), messe in scena con `fact_calib.py stage` su /tmp del job (come
  `calib_heldout_bl.sbatch`); millimetri e frame di E12 del dominio (`fact_calib.mesh_mm`, cioe' `frames.json`);
- L_d del dominio da `trainer_v3/factorized/calib_heldout_bl/params.json` (mediana di maxabs delle original
  held-out, la definizione di blmm);
- regione per (dominio, modello): la regola di `bp.region` con riferimenti = le 100 original held-out del dominio,
  ognuna un riferimento (voto >= 50%, come FaMoS; nessuna GT letta);
- fit: NICP col seme `blmm.mesh_seed(soggetto, etichetta)`, variante A senza espressione (le etichette sono forme
  d'identita' neutre: le espressioni `rexpr*` non sono fra le etichette), B coi valori congelati `bp.LOOP`; fallimenti
  come sopra (le coppie che li toccano escono dal calcolo di k, contate);
- coppie: quelle di `fact_calib.heldout_pairs` (stesso dominio, soggetti diversi, etichette diverse; regola copiata);
  d_P,GT = GT-SR di training (`ablations/c3f/gt_sr.npz`, `D_orig`) x `dP_per_unit` del suo json (come `calib_bl`);
- **k = mediana(d_P,GT) / mediana(d_P,B)** sulle coppie dei tre domini insieme, per modello; si riportano anche k_LS
  (minimi quadrati senza intercetta) e k per dominio (informazione, non parametri).
Uscita `calib_e2/<modello>.npz` (beta, S, regioni, tempi) e `calib_e2/k.json`. k entra nella sez. 9 prima del passo
`paired`. Se un modello avesse piu' del 10% di fit falliti sugli held-out, la composizione di quel modello non si
calcola e lo si dichiara.

**Lettura.** Tutte le viste (`hifi3d`, `facescape`, `faceverse`, `faceverse_neutral`, `famos`), righe e repliche del
protocollo (gruppi primari di fact_paired, come `bp_paired.frame`), maschera comune. Lettura dichiarata: **con FR**;
delta braccio - `<modello>_vb_comp` con factorized s1234 / s2345 `form_cal` e ctrlfr s1234 / s2345. Con SR si riporta
il valore (lettura non dichiarata).

**Controllo della formula** (richiesto dal critic): la stessa composizione con la S oracolo (FR) e d_P,GT = GT-SR x
`dP_per_unit` del set di test deve dare rho con FR = 1 (limite per una ricostruzione perfetta); si riporta il punto per
dominio. Dallo stesso passo: rho delle colonne di B sulle sole coppie original - original (punto, con FR e SR).

## 3. Costi misurati

Nessuna stima: ogni numero da un job di questa sezione, con nodo, CPU/GPU e numero di processi dichiarati.

**B (CPU).** Tempi per passo, per mesh, `time.perf_counter` dentro il worker (un processo per mesh, un thread BLAS,
tanti processi quanti core, come i job dell'emendamento 1): (i) lettura e coordinate di lavoro, (ii) NICP
(`bp.enroll_model`), (iii) variante A, (iv) variante B; l'errore di superficie e' escluso (e' diagnostica). Mesh: le
100 crop per vista e modello (sez. 1) e un campione delle senza crop: i primi 20 soggetti valutati
(`blmm.subjects`) x 5 topologie per vista (FaMoS: le 15 scansioni), uscita `<vista>/<modello>/time_e2.npz`. Per
(vista, modello): mediana e p95 con NICP (i-iv) e senza NICP (iii-iv). Si riportano anche i tempi totali per mesh gia'
in `fit_e1.npz` (`seconds`: NICP + A + B + tre errori di superficie), mediana e p95 su tutte le mesh. Controllo: i beta
di B del campione coincidono con quelli di `fit_e1.npz` (scarto massimo).

**B, confronto di una coppia.** coefficienti: ||beta_i - beta_j||; mesh d'identita' FR/SR e composizione: la rigida
verso mu (e la scala per SR) una volta per mesh, poi la distanza pesata sui vertici della regione per coppia. Misurati
con numpy a un thread sulle regioni di HIFI3D (tempo per mesh della rigida, tempo per coppia della distanza).

**Bracci.** Checkpoint di fact_paired: factorized s1234 (d_F calibrata) e ctrlfr s1234, `epoch072_ema.pth`.
- Preprocessing su CPU: il corpo di `v2_work/potential/areanorm_operators.py` (lettura, area, normalizzazione,
  `compute_operators` k_eig 128, scrittura dell'npz su /tmp del nodo), cronometrato per passo, un thread per processo,
  tanti processi quanti core (come i job degli operatori, 31 shard a un thread); mesh: le 600 di HIFI3D (100 soggetti x
  6 topologie, `datasets/HIFI3D/eval_view/npz`). Per topologia e in totale: mediana e p95 per mesh, numero di vertici.
  L'area per la tabella di scala (`build_scale_table`) e' un'area di triangoli: compresa nel passo dell'area.
- Forward su GPU: la catena degli embedding dei bracci (`eval_v3.py -- zs_embed`, stessi argomenti e tabelle di
  scala del passo `form` su HIFI3D: operatori `datasets/V3_OPS_CACHE/8e8f81d5f0204394`), 600 mesh, 10 forward di
  riscaldamento esclusi, `torch.cuda.synchronize` attorno a ogni misura. (a) batch 1: per mesh lettura dell'npz degli
  operatori, copia sul device, forward (con l'aggancio di `eval_v3`), separati; (b) batch tipico: il training dei
  bracci usa `--forward sequential` con 5 soggetti x fino a 6 mesh per passo, cioe' un forward per mesh: si misura il
  forward di gruppi di 30 mesh gia' sul device, tempo per mesh. Il forward a gruppi con padding (`embed_groups`) non e'
  usato da questi bracci e non si misura. GPU: 1 L40S, QoS normal; se nessuna L40S e' libera, 1 A40 o A10, dichiarata.
- Confronto di una coppia: ||z_i - z_j|| (ctrlfr) e d_F calibrata (factorized) con numpy su tutte le coppie di 600
  embedding, tempo per coppia.

**Tabella dei costi**: per iscrizione di una mesh (B: lettura + NICP + A + B; bracci: preprocessing + lettura +
copia + forward) e per confronto di una coppia, con hardware e parallelismo.

## 4. Sensibilita' a sigma sul pilota

Lo stesso pilota della sez. 2 dell'emendamento 1 (gli stessi 200 mesh per modello, stessi semi, stesso criterio S
senza GT) con `LOOP_SIGMA_MM` in {4, 8}, `LOOP_TAU_MM` in {2, 5, 10}, `LOOP_ITERS` in {5, 10}: 12 configurazioni per
modello (`bp_fit_e1.run_pilot` con la griglia passata come argomento, uscita `pilot_e2/<modello>.json`). Controllo: i
punteggi del fit originale e della variante A coincidono con `pilot_e1/<modello>.json`. Si applica la regola di scelta
dell'emendamento 1 (minimo; parita' entro l'1%: iters minore, sigma maggiore, tau maggiore; configurazioni con un fit
fallito escluse) **all'unione delle 30 configurazioni** (18 dell'emendamento 1 + 12).
- Se la scelta ha sigma = 2 (cioe' coincide con `bp.LOOP`): niente altro, lo si dichiara.
- Se la scelta ha sigma diverso: B con la configurazione scelta su tutte le viste valutate (le 500 mesh senza crop
  per vista, FaMoS 15) per quel modello, colonne `<modello>_vbs_{coef,fr,sr}` (uscita `<vista>/<modello>/fit_e2_sens.npz`),
  Spearman e delta come la sez. 5 dell'emendamento 1, **riportati come sensibilita'**: la B congelata (sigma 2) resta
  l'analisi dell'emendamento 1 e non si sostituisce. Il crop e la composizione restano sulla B congelata.

## 5. Correzioni al testo (`conclusions.md`, e quindi `results.md`)

1. Si toglie la spiegazione "FaceScape e' generato da un modello bilineare, un 3DMM ben fittato vi ricostruisce la
   forma": non distingue FaceScape (anche HIFI3D e FaceVerse sono campioni di 3DMM, ricostruiti da B a 0.34-0.58 mm
   con rho_SR 0.61 / 0.40). Al suo posto: la cella FR di FaceScape si legge con la bassa varianza della taglia (CV di
   S e Spearman fra FR e SR delle GT, misurati sulle righe e riportati con la definizione); FaceScape e' il dominio di
   sviluppo dei bracci (vantaggio di casa), quindi la vittoria di B li' pesa di piu'.
2. Le 8 celle HIFI3D SR contro B sono tutte contro ctrlfr, che con SR perde gia' contro NICP cs (0.594) e ICP cs
   (0.580); contro d_P di factorized B resta non risolto.
3. La cella FaceVerse neutra (1 su 240, ctrlfr s2345 contro GNM vB mesh SR) e' fragile: nessuna correzione per
   confronti multipli. Le celle FaceScape SR sono solide (tutti i bracci, entrambi i modelli).
4. L'errore di superficie ingresso -> M e' in parte in-sample: usa gli stessi 4096 punti della mesh su cui B fitta.
5. Il controllo del critic: il limite della formula GT su una ricostruzione perfetta e' rho = 1 (ricalcolato nella
   sez. 2); rho di B sulle sole coppie original - original 0.856 / 0.61 / 0.40 (FaceScape / HIFI3D / FaceVerse, valori
   del critic; ricalcolati qui e riportati accanto).

## 6. Calcolo

CPU su `prioritized` senza GPU (il nodo `cpu` e' in fail), venv con open3d per i fit (`aau/outlineB/run_o3d.sh`),
venv della GT unificata per i delta (`v3_work/unified_gt/run.sh`); preprocessing dei bracci e forward nel venv del
training (`aau/run.sh`). GPU solo per il forward dei bracci (sez. 3). Disco: le uscite stanno sotto
`aau/runs/evidence/baselines_param/` (npz dei beta e delle matrici; nessuna mesh); le mesh held-out e gli operatori di
prova solo su /tmp dei job. Niente dati licenziati in git.

## 7. Controlli (riepilogo)

1. B sulle 600: coppie senza crop = `fit_e1.npz` (scarto massimo).
2. all_cross: tolto il crop, chiavi e GT = `fact_paired.rows_for`.
3. Gruppi primari (sez. 2 e 4): bracci e colonne vB di `paired_e2.csv` = `paired_e1.csv` (stesse righe).
4. Formula della composizione con le GT: rho con FR = 1.
5. Tempi: beta di B del campione = `fit_e1.npz`.
6. Pilota: S del fit originale e della variante A = `pilot_e1`.

## 8. Passi e uscite

`aau/baselines_param/bp_e2.py` (fit e tempi, venv open3d): `crop`, `heldout` (dopo `fact_calib.py stage`), `pilot`,
`time`, `sens`; `aau/baselines_param/bp_paired_e2.py` (venv della GT unificata): `paired` (sez. 1, 2, 4: legge le GT
di test), uscite `paired_e2.csv`, `spearman_e2.csv`, `controls_e2.json` e le sezioni dell'emendamento 2 in fondo a
`results.md`; `aau/baselines_param/bp_cost.py`: `ops` (CPU), `forward` (GPU, dentro `eval_v3.py`), `pairs`, uscite in
`cost_e2/`. Ordine: crop, heldout, pilot, time, costi (nessuna GT di test) -> sez. 9 (commit) -> sens se serve ->
paired.

## 9. Valori congelati

Aggiunta del 10 ottobre 2026, dopo i passi senza GT di test e PRIMA del passo `paired` (le sez. 1-8 sono quelle del
commit 05eceb0, sha256 `0bd93adb...`). Nessuna GT dei domini di test e' stata letta dai job di questa sezione.

**k della composizione (sez. 2)**, job 1067774 (48 core CPU, 12 min 49 s; stage 1.500 mesh, 1.6 GB su /tmp): 0 fit
falliti su 1.500 per modello, 297.000 coppie (le stesse dell'`icp_cs` di factorized_calibration_bl.csv), mediana d_P
GT 0.0713.

| modello | k (mediana) | k_LS | k bfm | k ict | k gnm | mediana d_P B | vertici della regione bfm / ict / gnm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GNM | **1.6981** | 1.5767 | 1.791 | 1.569 | 1.713 | 0.0420 | 7.999 / 8.906 / 8.167 |
| FLAME 2023 Open | **1.6096** | 1.5012 | 1.678 | 1.518 | 1.604 | 0.0443 | 1.599 / 1.736 / 1.549 |

**Pilota della sensibilita' (sez. 4)**, job 1067775 (GNM, 3 min 49 s) e 1067776 (FLAME, 12 min 16 s), 40 core: 0 fit
falliti nelle 12 configurazioni nuove; controllo: S del fit originale e della variante A identici a `pilot_e1`
(scarto 0). Scelta della regola sull'unione delle 30 configurazioni:

| modello | scelta | S | entro l'1% | B congelata (S) |
| --- | --- | --- | --- | --- |
| GNM | **sigma 8 mm, tau 10 mm, 10 iterazioni** | 0.1389 | (8, 10, 10), (8, 5, 10) | sigma 2, tau 10, 5 (0.1566) |
| FLAME 2023 Open | **sigma 8 mm, tau 5 mm, 10 iterazioni** | 0.1766 | solo la scelta | sigma 2, tau 5, 5 (0.2106) |

La scelta ha sigma diversa da `bp.LOOP` per entrambi i modelli: la sez. 4 si applica, B con queste configurazioni su
tutte le viste valutate, colonne `<modello>_vbs_*`, lette come sensibilita'. Da dichiarare con i numeri: il minimo di S per
sigma cala in modo monotono (GNM: 0.156 a sigma 2, 0.149 a 4, 0.139 a 8; FLAME 0.209, 0.198, 0.177) e **la scelta sta di
nuovo sul bordo superiore della griglia** (8 mm): la griglia non si allarga ancora (sarebbe una terza scelta dopo
aver visto il pilota). Col prior piu' forte l'errore di superficie sul pilota sale (GNM mediana 0.42-0.81 mm, FLAME
0.54-0.86 mm, contro 0.3-0.6 mm della B congelata).

**Crop (sez. 1)**, job 1067773 (48 core, 6 min 12 s): 0 fit falliti in A e B su 800 (4 viste x 100 x 2 modelli).
**Riproducibilita' (sez. 3)**: rifatti su altri nodi, i beta di B differiscono da `fit_e1.npz` fino a 8.8e-3 (FLAME
HIFI3D; in unita' di sigma del prior), lo stesso valore su due nodi diversi (job 1067777 e 1067783): non e'
bit-a-bit, coerente con un'architettura di CPU diversa da quella dei job dell'emendamento 1 (i256-a10-06 e -08); il
crop di GNM e FLAME e' stato calcolato su i256-a10-08.
