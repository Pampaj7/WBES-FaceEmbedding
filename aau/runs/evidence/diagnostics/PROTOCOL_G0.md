# Gate G0: l'insegnante ArcFace-render, usato come metrica, ordina le coppie come le GT FR/SR? (protocollo, scritto PRIMA dei numeri)

Scritto il 12 ottobre 2026 dal coder, su richiesta del PI, PRIMA di qualunque render, embedding o Spearman
dell'insegnante di questo protocollo. L'impronta sha256 di questo file sta nel messaggio del commit che lo introduce e in
`PROTOCOL_G0.sha256`; ogni modifica va in un emendamento datato. E' un GATE a costo zero di training: nessun modello
viene addestrato, nessun numero di G0 e' un metodo o un risultato del paper.

## 0. Cosa era gia' visto prima di scrivere

- **Il gate ArcFace-render del 6 ottobre** (`aau/runs/arcface_render_zs/`, riconoscimento, non distanza graduata):
  normal map 3 viste rank-1 FaceVerse con espressioni 0.867, HIFI3D 0.998; ombreggiato 0.750 / 0.984; normal map -
  ombreggiato su FaceVerse +0.117 [+0.094, +0.142].
- **Il pilota della distillazione del 7 ottobre** (`aau/runs/distill_pilot/summary.md`): studente in convenzione BFM
  rank-1 0.332 (FaceVerse) e 0.224 (HIFI3D), 0.383 e 0.224 dell'insegnante a normal map; non passa nessuno dei due
  criteri. La v2 (`aau/distill/*_v2*`) e' stata cancellata senza valutazione.
- **Il precedente piu' vicino (8 ottobre, `aau/runs/data_scale_ood/arcface_vs_scale_hifi3d.md`)**: con la GT di allora
  (`maxabs`, non FR/SR) su HIFI3D nocrop_cross lo Spearman di ArcFace e' 0.242 [0.162, 0.330] a normal map e 0.245
  ombreggiato, contro 0.630 di e108 e 0.372 della Chamfer. WS1 (`paper/BOARD_DIARY.md`): ArcFace 0.29-0.38 nel ranking
  sintetico di allora. **Atteso, dichiarato qui: l'insegnante ordina male le distanze graduate (lettura NO).** GT,
  bracci e righe di G0 sono diversi: il precedente non decide l'esito.
- **I valori dei confronti sulle stesse righe sono pubblicati** (`d3_spearman.csv`, `results.md` di D1,
  `factorized_paired.csv`): d_P con SR, factorized s1234 / s2345: FaceScape 0.747 / 0.753, HIFI3D 0.622 / 0.613,
  FaceVerse neutra 0.333 / 0.372, FLAME suddiviso 0.806 / 0.806, FaMoS TRAIN 0.839 / 0.828; B-GNM `vb_sr` FaceScape
  0.854, HIFI3D 0.607, FaceVerse neutra 0.399. Le soglie (sez. 6) sono relative a questi valori.
- **Embedding dell'insegnante gia' su disco**: HIFI3D, normal map e ombreggiato, le 600 mesh valutate
  (`aau/runs/arcface_render_zs/hifi3d/{normals,shaded}/arcface_views.npz`). Qui servono solo da controllo della catena
  (K2). A quanto risulta dal repo nessuno Spearman dell'insegnante con le GT FR o SR e' mai stato calcolato.
- **Configurazione "ArcFace migliore"** di `aau/runs/indomain_recog` (inquadratura per mesh, normali smussate): vista,
  NON usata (sez. 2).
- Sonde di questa mattina, nessun render ne' embedding: le mesh di FLAME (D1) e di FaMoS TRAIN (D3) sono state
  cancellate a fine D1 / D3 (restano GT, embedding dei bracci, tabelle di scala); `datasets/ABL2X2_D1/in` (ablazione 2x2,
  2.000 mesh FLAME) non si usa (non e' di G0 e non ha controlli di provenienza qui).

## 1. Domanda

La distillazione (`paper/PLAN_DISTILL.md`) dovrebbe dare una supervisione che non dipende dal generatore: un
riconoscitore 2D addestrato su volti reali, applicato a render della sola geometria. D1-D4 e l'ablazione 2x2 dicono che
la metrica appresa cala sui generatori mai visti e che una testa lineare appresa altrove trasferisce poco. **G0 chiede,
senza training: l'insegnante, usato direttamente come metrica (1 - coseno), ordina le coppie di mesh in modo coerente
con le GT FR/SR sulle famiglie mai viste, meglio di d_P?** Se no, uno studente che lo imita non ha nulla da guadagnare
sul benchmark FR/SR (condizione necessaria, non sufficiente: lo studente del pilota ha raggiunto il 22-38%
dell'insegnante nel rank-1, 36-66% nella convenzione secondaria).

## 2. L'insegnante

Pipeline esistente, importata e non riscritta: `aau/zs3dmm/zs_arcface_render.py` (renderer, camera, normal map,
frame check) + `aau/multiface/ws3a_perceptual.py` (`calibrate_arcface`, `stage_embed`) + `aau/baselines/
arcface_fixed.py`. **La configurazione e' quella dell'insegnante del pilota** (`aau/distill/teacher_labels.py`), cioe'
la normal map del gate, che nel gate e nel pilota dava i numeri migliori (sez. 0):

- mesh ruotata dal frame dell'insieme a quello del renderer (alto -y, naso -z) con UNA rotazione propria per insieme,
  poi `normalize_maxabs` per mesh (centro dei vertici, / max |coordinata|: l'insegnante e' cieco alla taglia);
- **camera unica per insieme** (`zs_arcface_render.compute_camera`: centro medio dei bbox, scala che inquadra tutte le
  mesh a tutti gli yaw, margine 1.02), 512 px, **yaw 0, -30, +30**;
- **normal map** (`render_normals`, normali di faccia in spazio camera girate verso la camera, RGB = (n + 1) / 2, stesso
  SSAA e ordine del pittore di `render_mesh`);
- **crop fisso**: detector di insightface (`det_10g` di `buffalo_l`) sui render OMBREGGIATI di TUTTE le mesh
  dell'insieme, mediana per yaw dei 5 landmark, `estimate_norm` -> similarita' 2x3 congelata
  (`calibrate_arcface`), copiata sulle normal map (regola del gate: stessa camera e geometria, stesso riquadro); il
  detector non si usa piu' dopo la calibrazione; i suoi fallimenti per etichetta si contano (atteso: 100% su noisy,
  come nel gate);
- **ArcFace `buffalo_l` / `w600k_r50.onnx`**, onnxruntime CPU, un thread per sessione (`stage_embed`); embedding per
  vista L2; **embedding della mesh = media delle 3 viste, poi L2**; **distanza d_T = 1 - coseno**.

Perche' non "ArcFace migliore" (`indomain_recog`, revisione 1, punto C): non e' l'insegnante del pilota, in dominio non
e' uniformemente migliore (ICT senza crop AUC 0.987 contro 0.994 della camera di dominio, rank-1 col crop ICT 0.980
contro 0.993) e sceglierla ora sarebbe una scelta fra configurazioni gia' viste. Si riporta come fuori protocollo.

**Insiemi di render** (camera e crop sono per insieme: usano le mesh dell'insieme, MAI etichette o GT; e' una
dipendenza transduttiva dalle sole geometrie, la stessa del gate, dichiarata in ogni tabella):

| insieme | mesh | rotazione attesa | perche' |
|---|---|---|---|
| `facescape` | `datasets/DEV_FACESCAPE/eval_view/npz`, 100 soggetti x 6 etichette | `x180` | frame nativo alto +y, naso +z (E12, R circa I) |
| `hifi3d` | `datasets/HIFI3D/eval_view/npz`, 100 x 6 | `x180` | come il gate |
| `faceverse_neutral` | `datasets/FACEVERSE_ZS/eval_view/npz`, 100 x 6 | `none` | FaceVerse gia' alto -y, naso -z (gate) |
| `flame2023_s1` | FLAME 2023 di D1, suddiviso 1-a-4, 200 x 5 | `x180` | frame canonico FLAME (E12: R = I) |
| `famos` | FaMoS TRAIN di D3, 80 x 5 | `x180` | patch FLAME nel frame canonico |
| `bfm`, `ict`, `gnm` | held-out della calibrazione c, 100 x 5 ciascuno | `none`, `x180`, `x180` | come `teacher_labels.DOMAINS` |

Soggetti delle tre viste: i 100 valutati (`d3_pools.subjects(...)["evaluated"]`, gli stessi degli store ufficiali);
per HIFI3D devono coincidere con quelli del gate (controllo K2). **Frame check** (`zs_arcface_render.frame_check`, 4
soggetti per insieme, le 4 rotazioni candidate) PRIMA di ogni embedding: se il verdetto ("dritto" e vertice piu' vicino
alla camera al centro del bbox) contraddice la rotazione attesa, si usa quella del frame check e lo si dichiara.

**Mesh che non esistono piu' su disco**, rigenerate in `/tmp` del job col codice che le ha prodotte (importato):
- `flame2023_s1`: `d1_gen.identities` + `d1_gen.work_mesh` + `views.discretize`, stessi semi di D1;
- `famos`: `d3_famos.work_mesh` + `views.discretize`, stessi semi di D3 (solo le 80 persone TRAIN; nessun file delle
  persone di TEST);
- held-out `bfm`, `ict`, `gnm`: `fact_calib.stage` (le mesh grezze dalle sorgenti della spec, come la calibrazione).
Controllo K3 (sez. 5): numero di vertici e area grezza di ogni mesh = le tabelle di scala con cui i bracci sono stati
valutati (`d1/scale_table.npz`, `d3/famos/scale_table.npz`, `datasets/V3_OPS_CACHE/heldout_calib/scale_table.npz`).

**Licenza.** I pesi di insightface (`buffalo_l`: detector e `w600k_r50`) sono distribuiti per la sola ricerca non
commerciale. G0 li usa per una diagnostica; uno studente distillato da loro ne erediterebbe probabilmente il vincolo,
da tenere presente per il rilascio del modello e del dataset.

## 3. Insiemi di valutazione, righe, confronti

Stessi soggetti e righe della valutazione ufficiale e delle diagnostiche; nessun dato nuovo. **Mai Ava-256, mai FaMoS
TEST.**

| insieme | gruppo | righe | GT | seme del bootstrap | confronti |
|---|---|---|---|---|---|
| FaceScape dev | `nocrop_cross` (primario) | 88.725 | `facescape_{sr,fr}` di `fact_paired.rows_for` | 796786 | d_P, d_F cal. (s1234, s2345), ICP mm, B-GNM, B-FLAME |
| HIFI3D | `nocrop_cross` (primario) | 98.224 | `hifi3d_{sr,fr}` | 990708 | idem |
| FaceVerse neutra | `nocrop_cross` (`mesh_pair_nocrop`, primario) | 99.000 | `faceverse_{sr,fr}` | 271049 | idem (ICP mm delle mesh neutre) |
| le tre viste | righe col crop (secondario) | 49.500 per vista | `bp_paired_e2.all_cross_rows` | lo stesso della vista | idem |
| FLAME 2023 suddiviso | righe di D1 | 398.000 | `d1/gt_flame2023_s1.npz` | `SeedSequence([20261113, 3])` | d_P, d_F cal., B-GNM (incrociato), B-FLAME (prior esatto: solo riferimento) |
| FaMoS TRAIN (reale) | righe di D3 | 63.200 | `d3/famos/gt_famos.npz` | `SeedSequence([20261123, 0])` | d_P, d_F cal. |

- **Righe nocrop delle tre viste** = quelle di D3 (`d3_stats.view_data`, importato: righe e GT di
  `fact_paired.rows_for`, maschera di `fact_paired.main` con B finito), FLAME e FaMoS = `d3_stats.set_data` (righe di
  `diag.rows`, soggetti diversi, etichette diverse). **Righe col crop**: le righe di all_cross con il crop su un lato
  (come D4), maschera comune su tutte le colonne, righe tolte contate.
- **Colonne.** Insegnante d_T (letto con SR e con FR). Bracci factorized s1234 e s2345 (store ufficiali, `d1/emb`,
  `d3/famos/emb`): d_P (`shape`, con SR) e d_F calibrata (`form_cal`, c ufficiale, con FR). ICP mm =
  `mm_rigid_icp_chamfer` (ICP + Chamfer in mm, matrici di `aau/runs/evidence/baselines_mm/<vista>/mm` e
  `faceverse_neutral/blmm/mm`), letta con SR e con FR. B = variante B del fit senza espressione (emendamento 1 dei
  concorrenti parametrici), GNM e FLAME 2023 Open: `vb_sr` con SR, `vb_fr` con FR (`fit_e1.npz` e, per il crop,
  `fit_e2_crop.npz`; FLAME: `d1/bp`). **ICP mm su FLAME e FaMoS, B su FaMoS: non esistono e non si calcolano** (G0 e' a
  costo zero; le celle sono "non disponibile").
- Se una colonna di confronto ha valori non finiti su righe del primario (atteso solo per ICP mm sulla FaceVerse
  neutra, mai controllato), il suo delta si calcola sulle righe dove e' finita, contate; tutte le altre colonne restano
  sulle righe del primario.
- **Spearman** pesato c_a c_b coi conteggi della replica (`diag.wranks` / `diag.wpearson`, uguale al campione con le
  righe ripetute), ranghi medi sui pari merito. **Bootstrap per soggetto**, 1000 repliche, i conteggi dei semi in
  tabella (quelli pubblicati: nessun seme nuovo), le STESSE repliche per tutte le colonne e i gruppi di un insieme.
  IC 95% percentile, P(delta <= 0).
- **Delta appaiati** (stesse righe e repliche): insegnante - d_P (SR) e insegnante - d_F cal. (FR) per seme;
  insegnante - ICP mm (SR e FR); insegnante - B-GNM, insegnante - B-FLAME (`vb_sr` con SR, `vb_fr` con FR).

## 4. Combinazione insegnante + taglia del modello (DESCRITTIVA)

d_TF = sqrt((S_i - S_j)^2 + S_i S_j (c_T d_T)^2), S = exp(z_0) del braccio factorized dello stesso seme (la formula
di `fact_paired.distances`, con d_T al posto di d_P). **c_T si fissa SOLO sugli held-out sintetici** della calibrazione
(`bfm`, `ict`, `gnm`, 100 soggetti x 5 etichette ciascuno), con la regola di `fact_calib.calib_one`: coppie dello stesso
dominio, soggetti diversi, etichette diverse; c_T = mediana(d_P GT) / mediana(d_T), d_P GT = `c3f/gt_sr.npz` x
`dP_per_unit`. Una sola c_T per entrambi i semi (d_T non dipende dal seme). Letta con FR, delta contro d_F cal. dello
stesso seme; mai nella regola.

## 5. Controlli

- **K1 riferimenti**: d_P (SR) e d_F cal. (FR) dei due semi e B (`vb_sr`, `vb_fr`) sulle righe primarie = `d3_spearman.csv`
  (righe `cross`; B su FLAME = `d1_spearman.csv`), scarto <= 1e-9 sul punto e sugli estremi dell'IC, stesso numero di
  righe; ICP mm su FaceScape e HIFI3D = la riga `mm_rigid_icp_chamfer` di `factorized_paired.csv` (scarto <= 1e-9).
- **K2 catena dell'insegnante**: su HIFI3D soggetti = quelli del gate; embedding per vista ricalcolati =
  `arcface_render_zs/hifi3d/normals/arcface_views.npz` (scarto massimo <= 1e-3, come K2 di D4).
- **K3 mesh rigenerate**: per FLAME, FaMoS e held-out, vertici uguali e area grezza con scarto relativo <= 1e-6 rispetto
  alla tabella di scala dei bracci, per mesh. up60k (decimazione qslim) puo' non essere bit per bit riproducibile: se
  K3 fallisce SOLO su up60k con vertici entro l'1% e area entro 1e-3 relativo, la mesh si usa e lo si dichiara; altrimenti
  le righe delle mesh fallite escono da tutte le colonne (contate).
- **K4 embedding**: tutti finiti, norma per vista 1 entro 1e-3, una riga per mesh attesa (1.800 + 1.000 + 400 + 1.500).
- **K5 catena delle statistiche** (`g0_test.py`, prima dei numeri veri): con un "insegnante" finto = d_P di s1234 i
  delta contro d_P s1234 sono 0 su tutte le repliche e K1 passa.
- **K6 c**: la c dei due bracci ricalcolata sugli stessi held-out con lo stesso codice di c_T = `c_median` di
  `factorized_calibration.csv` (scarto <= 1e-9): righe e GT di c_T sono quelle della calibrazione.

## 6. Lettura preregistrata

Famiglie della regola: le **tre del benchmark FR/SR, mai viste in training** (FaceScape dev, HIFI3D, FaceVerse neutra;
FaceScape e' il dominio dev, nessun gradiente), gruppo nocrop_cross; in alternativa **FaMoS TRAIN**, l'unica famiglia
reale. Per ognuno dei due semi factorized (s1234, s2345):

- **P(seme)** vale se Delta_SR = rho(d_T, SR) - rho(d_P, SR) e' **>= +0.05 con l'IC 95% sopra 0** in **almeno 2 delle 3
  famiglie**, OPPURE su **FaMoS TRAIN**.
- **PROMETTENTE** se P vale in entrambi i semi; **NON RISOLTO** se in uno solo; **NO** se in nessuno.
- In piu', descrittivo, **SOTTO** se in entrambi i semi almeno 2 famiglie su 3 hanno Delta_SR con l'IC sotto 0
  (l'insegnante ordina peggio di d_P).

Perche' queste soglie:
- +0.05 e' la soglia di R3 (D2) e di R_A (D3): il guadagno minimo che giustifica un componente nuovo del metodo; e' circa
  meta' del gap di generatore di D1 (+0.11) e piu' del doppio della semi-ampiezza tipica degli IC dei delta appaiati di
  D3-D4 (0.01-0.02), quindi risolvibile.
- Lo studente non eguaglia l'insegnante (pilota: 22-38% del suo rank-1): un insegnante che supera d_P di poco non lascia
  margine a uno studente che deve anche restare invariante alla discretizzazione. Il gate e' necessario, non sufficiente.
- 2 famiglie su 3: un guadagno su una sola famiglia puo' essere un'idiosincrasia del dominio (FaceVerse sta a 0.33-0.40
  per tutti i metodi). FaMoS TRAIN da sola basta perche' e' reale: un guadagno li' direbbe che l'insegnante porta
  informazione di forma dei volti veri che d_P non ha.
- Entrambi i semi: d_P differisce fra i semi fino a 0.04 (FaceVerse 0.333 / 0.372); la regola non deve dipendere dal seme
  piu' debole.
- Solo SR: l'insegnante e' cieco alla taglia (maxabs per mesh) e FR contiene la taglia; FR si legge con la combinazione
  della sez. 4, descrittiva.

**Descrittivi, accanto (non regole)**: FLAME suddiviso con la stessa regola per seme (quarto generatore mai visto,
sintetico); insegnante con FR contro d_F cal.; combinazione contro d_F cal.; insegnante contro ICP mm e B; righe col crop;
i numeri del pilota e del suo insegnante (`distill_pilot/summary.md`), riportati come sono.

**Cosa cambierebbe** (dichiarato ora): PROMETTENTE -> si puo' scrivere un protocollo di distillazione verso FR/SR (fuori
da G0); NO -> l'insegnante ArcFace-render non e' una leva per il benchmark FR/SR e la distillazione verso di esso non si
rilancia; resta una baseline di riconoscimento; NON RISOLTO -> nessuna decisione da G0.

## 7. Costo misurato dell'insegnante per mesh

- **Per mesh** (iscrizione): lettura + 3 normal map + 3 ritagli + 3 forward ArcFace + media, su un campione fisso: i
  primi 3 soggetti valutati per id x 6 etichette x 3 viste del benchmark = 54 mesh, UN processo a UN thread
  (onnxruntime intra-op 1, OMP/MKL/OpenBLAS 1), dopo una mesh di riscaldamento esclusa, su un nodo CPU di `prioritized`
  (host e CPU nel referto). Mediana, p10, p90, per etichetta e totale; voci separate render e embedding.
- **Una tantum per insieme** (camera, render ombreggiati, calibrazione del detector): secondi di parete e numero di
  processi dai job veri, riportati a parte (non sono costo per mesh).
- **Throughput dei job**: secondi di parete per insieme e per fase con N processi (dai log), mesh al secondo.
- Nessun numero di costo dei bracci si rimisura; per contesto si citano quelli gia' misurati (D4 `cost.json`,
  `indomain_recog`), dichiarando che vengono da altri nodi.

## 8. Risorse, uscite, pulizia

- **CPU** su `prioritized` per tutto: ArcFace e il detector girano su onnxruntime CPU come nel gate; gli embedding dei
  bracci esistono gia'. **Nessuna GPU.**
- Codice nuovo in `aau/diagnostics/g0_*.py` (`g0_meshes.py` mesh rigenerate e K3, `g0_teacher.py` frame check, render,
  calibrazione, embedding, costo; `g0_stats.py` righe, colonne, bootstrap, delta, lettura, `results_g0.md`;
  `g0_test.py` K5) e `g0.sbatch`. Importati in sola lettura: `zs_arcface_render.py`, `ws3a_perceptual.py`,
  `arcface_fixed.py`, `diag.py`, `d1_gen.py`, `d3_famos.py`, `d3_stats.py`, `d3_pools.py`, `fact_paired.py`,
  `fact_calib.py`, `bp_paired_e2.py`, `blmm_eval.py`. Non si toccano `fact_paired.py`, `fact_summary.py`,
  `v3_work/stream/`, `paper/` (salvo la voce "12 ottobre, G0" di `paper/BOARD_DIARY.md`), i job altrui.
- Uscite in `aau/runs/evidence/diagnostics/`: `g0/teacher/<insieme>/teacher.npz` (fuori da git), `camera.json`,
  `arcface_align.json`, `timing.json`, `frame_check.json` (in git); `g0/meshes.json` (K3), `g0/controls.json`,
  `g0/readings.json`, `g0/cost.json`, `g0/reps.npz` (fuori da git); `g0_spearman.csv`, `g0_delta.csv`,
  `results_g0.md`.
- **Mesh rigenerate e render in `/tmp` del job** (RAM: `--mem` include i render, circa 0.5 GB ogni 600 mesh), cancellati
  in uscita; nessuna mesh, render, immagine di volto, npz o peso in git. Dati derivati da FaMoS (licenza MPI): mai in
  git. Disco: si controlla di restare con almeno 300 GB liberi nella quota.

## 9. Fuori da questo protocollo

Qualunque training o distillazione; altre configurazioni dell'insegnante (ombreggiato, "ArcFace migliore", altre viste
o risoluzioni) se non come emendamento dichiarato; altri bracci (C3M, ctrlfr, dual); FaceVerse con espressioni; il
riconoscimento d'identita' (gia' misurato dal gate); ICP mm e B dove non esistono; FaMoS TEST e Ava-256.
