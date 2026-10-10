# Concorrenti parametrici (GNM Head, FLAME 2023 Open) e distanza varifold: protocollo

Scritto il 10 ottobre 2026 dal coder, su richiesta del PI, PRIMA di ogni numero di Spearman o di delta di questo
protocollo. Codice in `aau/baselines_param/` (definizioni e costanti in `bp.py`), numeri in questa cartella. L'impronta
sha256 di questo file sta nel messaggio del commit che lo introduce e in `PROTOCOL.sha256`; ogni modifica va in un
emendamento datato.

**Cosa esisteva gia' al momento della scrittura** (nessuno di questi passi legge una GT):
- pilota tecnico su soggetti NON valutati: 3 soggetti del template (`blmm.template_subjects`) x 5 topologie su HIFI3D
  e FaceVerse con espressioni, 3 original su dev FaceScape. Misurati RMS del fit (0.08-0.69 mm), residuo NICP
  (1.8-3.5 mm), tempi, e la separazione stesso soggetto / soggetti diversi delle distanze nuove;
- FaMoS (che non ha soggetti non valutati): regione e fit di 3 scansioni `peak` (solo RMS e tempi), poi `bp_fit.py` e
  `bp_varifold.py` completi sulle 15 scansioni di galleria come prova funzionale, e la costruzione delle colonne di
  `bp_paired.py` controllata solo per forma e NaN (nessuno Spearman calcolato);
- varifold: tempi e precisione float32 contro float64 su 15 mesh di soggetti del template.
Le costanti sotto erano tutte nel codice prima del pilota, tranne il blocco del varifold (8192, solo velocita').

## 1. Domanda

Il revisore chiedera': "perche' un encoder appreso, se il fit di un 3DMM statistico fa lo stesso?" (Amberg, Knothe,
Vetter 2008: NICP robusto, poi distanza nello spazio dei coefficienti). Qui: quanto i bracci (factorized calibrato,
ctrlfr) battono o perdono contro il fit NICP + proiezione d'identita' di due 3DMM, con FR e SR, sulle stesse righe e
repliche dei delta appaiati di `fact_paired.py`. In piu' la distanza varifold (Besnier et al. 2023, arXiv 2306.15762;
baseline di Lee 2025), geometrica e non appresa.

## 2. Modelli

| modello | file e sha256 | licenza | regione | identita' / espressione | stato rispetto al training |
|---|---|---|---|---|---|
| GNM Head v3.0 | `~/data/gnm_head/gnm_head.npz`, `61d78bbf...47eb` (= checksum HF) | Apache-2.0 | `hockey_mask`, quad a ventaglio (9.022 v.) | 170 `head_*` / pool di 350 PCA (parte bassa del volto, occhi), lingua e pupille a zero | **PRIOR VISTO**: GNM e' fra i generatori di C3F (392 BFM, 844 GNM, 4.557 ICT) e di e108 |
| FLAME 2023 Open | `v2_work/genflame/official/FLAME2023Open/flame2023_Open.pkl`, `e75a0990...7623` | CC BY 4.0 (Readme del pacchetto) | maschera `face` di `FLAME_masks.pkl` (1.787 v.) | 300 / 100 PCA + pitch della mandibola | non nel training dei bracci |

Basi d'identita' gia' a +1 sigma in entrambi i file (norme dei modi decrescenti, misurato): il prior e' N(0, 1) e i
coefficienti sono gia' "scalati per autovalore". Caricamento: `v3_work/mm/loaders.py` (`load_gnm`, `load_flame`), in mm.

**Verifica che FLAME sia la versione Open e non la 2020** (che non va usata): il file e' quello del pacchetto
`external_data/FLAME2023Open.zip` (`flame2023_Open.pkl` + `FLAME2023_Open Readme.pdf`, che dichiara "FLAME2023_Open is
available under a Creative Commons Attribution 4.0 International License"); sha256 diverso da quello di FLAME 2020
(`generic_model.pkl`, `efcd14cc...725b`); chiave `supr_expression_metadata` presente solo qui; `v_template` diverso fino
a 6.6 mm e `shapedirs` fino a 21 mm da FLAME 2020, triangoli identici. `FLAME_masks.pkl` (dal sito FLAME, accanto a
FLAME 2020) si usa solo come elenco di indici della regione, valido per la stessa topologia; non si ridistribuisce.

Avvertenze da riportare con ogni numero:
1. **GNM e' un prior visto.** I bracci sono addestrati su mesh generate da GNM: il confronto e' "stesso prior" e va
   letto come tale (puo' favorire il braccio, che ha visto la variabilita' di GNM, o il fit, che ne ha la base esatta).
2. **FLAME e' un prior non visto dai bracci**, ma definisce il frame canonico e la regione unificata della GT di E12
   (media del dominio -> media FLAME). La GT non usa i coefficienti FLAME.
3. **FLAME su FaMoS e' quasi un oracolo**: le registrazioni FaMoS sono mesh FLAME e la GT di FaMoS viene da quelle.
   Il numero si riporta come limite superiore, non come concorrente.

## 3. Procedura di fit (per vista e modello)

Tutto in `bp.py`; l'infrastruttura e' quella di `aau/baselines_mm` (importata): viste, mm nel frame canonico
(`blmm.to_mm`), unita' di lavoro L_d (`blmm.params`), semi, `view_root`.

1. **Regione del modello dentro il dominio** (`bp.region`), una volta per vista. Riferimenti: per le viste a 6
   topologie la media vertice per vertice delle original dei 100 soggetti NON valutati del template
   (`blmm.template_subjects`, gli stessi del NICP su template); per FaMoS le 15 scansioni di galleria (soggetti
   valutati: usate solo come maschera binaria, votata su tutte e 15). La media del modello, nel frame canonico
   (`frames.json`, voci `gnm` / `flame`), si porta su ogni riferimento con traslazione dei baricentri + ICP rigido
   punto-punto (soglia infinita, poi 10 mm). Un vertice vota si' se il vertice piu' vicino del riferimento dista meno
   di 10 mm e non sta sul bordo ne' a un anello dal bordo. Tenuti i vertici col voto >= 50%, i triangoli coi tre
   vertici tenuti, la componente connessa piu' grande.
2. **Iscrizione NICP** (`bp.enroll_model`), per mesh: template = media del modello sulla regione (al piu' 4096 vertici,
   `rng(0)`, come `blmm.build_template`), centrato e diviso per L_d (modo `mm` di blmm, per cui NICP e' tarato); 4096
   punti della mesh (`blmm.work_coords` modo `mm`) col seme `blmm.mesh_seed(soggetto, topologia)` (quello del NICP su
   template); `ir_simicp.similarity_icp` + `facebench.nonrigid_icp_align` del template sulla mesh (la catena di
   `blmm.enroll`, parametri di default); uscita x L_d = vertici del modello registrati sulla mesh, in mm.
3. **Regressione MAP** (`bp.fit_points`): min_{R, t, beta, psi, theta} sum_v ||R x_v(beta, psi, theta) + t - y_v||^2 +
   sigma^2 (||beta||^2 + ||psi||^2), sigma = 1 mm, alternando la rigida (Umeyama SENZA scala: la taglia resta
   nell'identita') e i coefficienti in forma chiusa, al piu' 30 giri, arresto se l'RMS cambia meno di 1e-4 mm.
   **Espressione libera in tutte le viste** (anche nelle neutre, dove puo' assorbire parte dell'identita': e' la
   scelta dichiarata, uguale per tutti i domini). FLAME: x_v include il pitch theta della mandibola (LBS del solo
   giunto 2 con i correttivi di posa, giunto regredito dalla forma, quindi lineare in beta e psi a theta fisso),
   theta in [-0.1, 0.6] rad per ricerca scalare limitata. GNM non ha un giunto della mandibola (e' nella base PCA).
4. Si tengono beta (identita'); psi, theta, R, t si scartano per le distanze.

## 4. Distanze riportate

Per ogni coppia di mesh (i, j):
- **(i) `<modello>_coef`**: ||beta_i - beta_j||, euclidea sui coefficienti d'identita' a +1 sigma (scalati per
  autovalore; e' la distanza di Mahalanobis nello spazio PCA).
- **(ii) `<modello>_fr`, `<modello>_sr`**: sulle mesh d'identita' ricostruite mu + B beta (espressione e mandibola a
  zero), sui vertici della regione del passo 3.1, in mm, pesi = aree baricentriche di mu. FR: ogni mesh portata su mu
  con la rigida ai minimi quadrati pesata (senza scala), poi sqrt(sum_v w_v ||a_i,v - a_j,v||^2 / sum w) (la formula
  di `cgt.Canon.distances`). SR: come FR dopo aver scalato ogni mesh a centroid size di mu attorno al baricentro pesato
  (come `cgt.Canon.shape_S`). Differenza dalla GT: rigida ai minimi quadrati invece che robusta (le mesh d'identita'
  non hanno outlier). Nessuna riscalatura per mesh in FR.
- **varifold** (`bp.varifold_measure`, `bp_varifold.py`): implementazione del repo, non riscritta: kernel di
  `v2_work/phase0/measure_distances.py` nella versione a blocchi di `aau/baselines/geometric_kernel.py` (validata
  contro phase0): k = exp(-||c_i - c_j||^2 / sigma^2) (n_i . n_j)^2 (varifold non orientato), d^2 = somma su sigma di
  ||mu_X - mu_Y||^2. **Larghezze di banda fissate qui: sigma = 10, 20, 40 mm.** Misura: triangoli della mesh in mm
  (`blmm.to_mm`), centrati sul baricentro pesato per area, nessuna rotazione ne' scala per mesh (frame canonico del
  dominio, coerente con FR), aree divise per l'area totale (massa unitaria: con le aree grezze la topologia `noisy`
  ha piu' massa dell'original, misurato in `geometric_matrix.py`), quantizzata su una griglia di lato 2.5 mm (un
  quarto della sigma minore, `geometric_matrix.quantize_measure`, deterministica). Float32 su GPU: scarto da float64
  <= 1e-4 relativo sulle distanze fra soggetti diversi (pilota), fino a qualche % sulle distanze minuscole dello stesso
  soggetto, che non stanno nelle righe.

## 5. Viste, righe, repliche

Le righe, le GT e le repliche sono quelle di `v3_work/trainer/tools/fact_paired.py` (importato da `bp_paired.py`, NON
modificato: lo usa il job 1066515), che a loro volta sono quelle di `aau/baselines_mm`:

| dominio | vista | righe | GT | repliche |
|---|---|---|---|---|
| HIFI3D | `hifi3d` | `nocrop_cross` (soggetti diversi, topologie diverse, niente crop) | FR, SR di `datasets/CANONICAL_GT/eval` | 1000 per soggetto, seme di E12 del gruppo |
| dev FaceScape | `facescape` | `nocrop_cross` | idem | idem |
| FaceVerse con espressioni | `faceverse` | `mesh_pair_nocrop` | idem | idem |
| FaceVerse neutra | `faceverse_neutral` | le stesse righe di FaceVerse, mesh neutre (`faceverse_neutral/PROTOCOL.md`) | le stesse | le stesse di FaceVerse |
| FaMoS TEST | `famos` | scan gallery -> scan (15 soggetti) | `famos_test_{fr,sr}` | `famos_eval.bootstrap_counts(15, 1000, 1234)` |

Mesh iscritte: i 100 soggetti valutati x 5 topologie senza crop (original, remesh, noisy, down8k, up60k), 500 per
vista; FaMoS: le 15 scansioni di galleria. Spearman, parziale dato l'oracolo della taglia e quintile basso con
`fact_paired._rep`, valori e delta con `fact_paired.summarize`; maschera COMUNE per dominio (righe con tutte le
distanze finite, colonne nuove comprese), come `fact_paired.main`. Bracci: le colonne di `fact_paired.model_columns` /
`famos_frame` (ultimo checkpoint EMA); per la vista neutra gli embedding di `aau/runs/evidence/faceverse_neutral/embed`
(e delle baseline di fact_paired solo la taglia oracolo, che e' per identita').

## 6. Fallimenti del fit

Fallito = eccezione (NICP, ICP, algebra), valori non finiti, oppure **RMS finale del fit sui punti registrati > 5 mm**.
Un fit fallito da' coefficienti NaN, le righe che lo toccano escono dalla maschera comune del dominio (per TUTTI i
metodi, come in fact_paired); nessun nuovo tentativo, nessun seme diverso. Si riportano per vista e modello: mesh
fallite e tasso, RMS mediana e massima, residuo NICP mediano, tempi, righe perse. Se le righe perse superano il 5% di un
dominio, il dominio si riporta anche con le sole righe complete per i soli metodi esistenti (controllo).

## 7. Cosa si riporta

- Spearman (rho, IC 95% percentile) con FR e con SR, per dominio, per ognuna delle 7 colonne nuove (`gnm_coef`,
  `gnm_fr`, `gnm_sr`, `flame2023_coef`, `flame2023_fr`, `flame2023_sr`, `varifold`) e, come riferimento sulle stesse
  righe, factorized s1234 / s2345 (d_F calibrata con FR, d_P con SR), ctrlfr s1234 / s2345 (||z||), NICP su template in
  mm, ICP + Chamfer in mm, NICP per coppia cs, taglia oracolo.
- Delta appaiati braccio - colonna nuova (stesse repliche), IC 95% e P(delta <= 0): con FR per factorized s1234 / s2345
  `form_cal` e ctrlfr s1234 / s2345; con SR per factorized s1234 / s2345 `shape` e ctrlfr s1234 / s2345. Tutti gli altri
  delta (altri bracci, parziale, quintile basso) stanno in `paired.csv`.
- Lettura dichiarata: un delta e' a favore del braccio se l'IC 95% sta sopra 0, contro se sta sotto, altrimenti non
  risolto. Nessuna correzione per confronti multipli (descrittivo). Le colonne si leggono con la GT corrispondente
  (`_fr` con FR, `_sr` con SR, `_coef` e `varifold` con entrambe); tutte si riportano con entrambe.
- Tasso di fallimento, tempi, job id.

## 8. Controlli

1. Con la maschera uguale a quella di fact_paired (nessun fit fallito), i valori dei bracci devono coincidere con
   `aau/runs/evidence/trainer_v3/factorized_paired.csv` (stesse righe, stesse repliche): differenza massima riportata.
2. FaMoS: i controlli di `famos_frame` (bracci contro `graded.csv`).
3. Varifold: `bp_varifold.py --check` (stessi prodotti su GPU e CPU), scarto riportato.
4. Regione: frazione di vertici tenuti e distanza mediana per riferimento (`region.json`).

## 9. Fuori da questo protocollo

Topologia crop e gruppi `all_cross` / `subject_pair_mean`; riconoscimento; la variante del fit senza espressione;
MICA, NPHM e gli altri concorrenti di `literature/COMPETITORS_LEARNED_2026-10-10.md`; il collegamento a
`fact_paired.py` / `fact_summary.py` (dopo il job 1066515).
