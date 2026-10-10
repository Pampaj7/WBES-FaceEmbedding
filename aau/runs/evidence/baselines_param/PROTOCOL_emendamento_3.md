# Concorrenti parametrici, emendamento 3 (POST HOC): bracci sulla regione di B, crop in-distribuzione, correzioni al testo

Scritto il 10 ottobre 2026 dal coder, su richiesta del PI, **DOPO aver visto i numeri** dell'emendamento 2
(`PROTOCOL_emendamento_2.md`, sha256 `0bd93adb...` / `91a0fc53...`; risultati in `results.md`, commit f53ba2a). E' un
emendamento **post hoc**: la motivazione e' il verdetto RISERVE del critic sull'emendamento 2 (numeri corretti,
racconto incompleto: il crop e' nella distribuzione di training, le baseline geometriche su all_cross esistono gia',
C3M crolla anch'esso, la fragilita' al crop era nota, B usa una regione che esclude il bordo e i bracci no, il pooling
di all_cross sovrastima il calo). Nulla delle analisi precedenti cambia: protocollo ed emendamenti 1-2 restano come
sono; le analisi di qui sono di sensibilita' e diagnostiche e si leggono come tali. L'impronta sha256 di questo file
sta nel messaggio del commit che lo introduce e in `PROTOCOL_emendamento_3.sha256`, prima di ogni calcolo.

Regola di lettura per i delta: quella della sez. 7 del protocollo (a favore del braccio se l'IC 95% sta sopra 0,
contro se sta sotto, altrimenti non risolto; nessuna correzione per confronti multipli, descrittivo).

## 1. Esperimento 1: i bracci sulla stessa regione di B (controllo di equita')

**Ipotesi.** Sulle righe col crop i bracci cadono (emendamento 2) e B no. B confronta le mesh solo sulla sua regione
per (vista, modello), stimata sui 100 soggetti NON valutati con l'anello di bordo escluso (`bp.region`), e nelle
corrispondenze ingresso -> modello scarta i punti che cadono su triangoli che toccano il bordo della regione
(`bp.loop_correspondences`); i bracci fanno pooling su tutta la superficie, bordo compreso. H_supporto: il calo dei
bracci viene dal supporto diverso (la banda di bordo tolta dal crop); se l'ingresso dei bracci e' ritagliato alla
regione di B, sulle righe col crop i bracci recuperano. H_descrittore: non recuperano (limite del descrittore anche a
supporto uguale).

**Regione (non toccata).** Per (vista, modello) i vertici di `<vista>/<modello>/region.npz` (bp_fit.py). Si ricolloca
la regione come in `bp.region` (stesso codice: `bp.place_canonical`, `bp.references`, `bp.rigid_icp` con soglia 1000 mm
poi `REGION_DIST_MM`): A = media del modello collocata sul riferimento della vista (media delle original dei 100
soggetti del template), superficie della regione R = (A[vertici], `bp.compact_faces`). Controllo: `bp.region` rieseguita
ridà gli stessi vertici di `region.npz`.

**Ritaglio per mesh.** Per ognuna delle 600 mesh valutate per vista (100 soggetti x 6 topologie, crop compreso; il file
dello store dei bracci, `subjects.json` del passo `form`): X = `blmm.to_mm(vista, V)`; R portata su X con
`bp.rigid_icp(R, X, REGION_DIST_MM, identita')`; per ogni vertice di X il punto piu' vicino su R (`bp.Surface`): il
vertice si tiene se dista meno di `REGION_DIST_MM` (10 mm) e il triangolo di R su cui cade non tocca il bordo di R
(`bp.boundary_vertices` sui triangoli di R; la stessa esclusione delle corrispondenze ingresso -> modello di B). Poi
i triangoli coi tre vertici tenuti, la componente connessa piu' grande (`mesh_ops.largest_component`, come
`bp.region`), i vertici non referenziati tolti. Si scrivono le coordinate GREZZE (unita' native, V[tenuti]) con le
facce rimappate; FaceVerse (con espressioni e neutra): facce invertite come `zs_stage.py --flip-faces` (la
convenzione degli store dei bracci). Viste: `hifi3d`, `facescape`, `faceverse`, `faceverse_neutral`; regioni: GNM e
FLAME 2023 Open (due ritagli per mesh: 4 x 2 x 600 = 4.800 mesh). Fallimento: meno di 500 vertici dopo il ritaglio o
eccezione -> la mesh resta NaN per quel ritaglio, contata.

**Diagnostica del ritaglio (senza GT), da riportare perche' decide la lettura.** Per mesh: vertici e area (mm^2) prima
e dopo, frazione d'area tenuta; **copertura della regione da parte della mesh**: frazione dei vertici di R (portata
sulla mesh, pesi d'area della media del modello sulla regione) il cui vertice piu' vicino di X dista meno di
`REGION_DIST_MM` e non e' sul bordo ne' a un anello dal bordo di X (`bp.boundary_ring`, la regola di voto di
`bp.region`). Per (vista, modello, topologia): mediana e p5 della copertura; per soggetto, crop contro original dopo il
ritaglio: log(sqrt(area ritaglio del crop) / sqrt(area ritaglio dell'original)), mediana e IQR. Se la copertura del
crop e' >= 0.95 di mediana, crop e original dopo il ritaglio hanno quasi lo stesso supporto (la regione di B esclude
gia' la banda tolta dal crop): lo si dichiara, e il recupero dei bracci e' allora atteso per costruzione (il test dice
che il calo veniva dalla banda fuori dalla regione di B). Se e' < 0.95, il crop toglie anche parte della regione di B,
e B stesso lavora su dati parziali.

**Catena dei bracci (identica al passo `form`).** Tabella di scala `v3_work/trainer/tools/build_scale_table.py
--view-dir <ritagli> --domain <dominio>` (area_mm2 = u_d^2 x area del ritaglio grezzo; dominio `hifi3d`, `facescape`,
`faceverse`); operatori `v2_work/potential/areanorm_operators.py --k-eig 128` (un thread per processo); embedding
`v3_work/trainer/eval_v3.py -- aau/zs3dmm/zs_embed.py` con gli argomenti di `eval_common.sh` usati dal passo `form`
(`--checkpoint_selector best_by_clean --subject_split all --max_subjects 0 --max_meshes_per_subject_eval 10 --seed
1234 ...`, `--dist_npz` della vista come nell'`eval_key.txt` dello store), `WBES_V3_FACTORIZED_OUT=full`,
`WBES_V3_SCALE_TABLES` = la tabella dei ritagli. Checkpoint: quello scritto nell'`embeddings.npz` dello store della
vista (`fact_paired.embeddings`). Distanze con `fact_paired.distances` e la c di calibrazione di
`factorized_calibration.csv` (`fact_paired.calibration`), invariata. Bracci: factorized s1234 / s2345 (d_F cal., d_P),
ctrlfr s1234 / s2345 (||z||), C3M e123 / e205 (d_F cal., d_P; descrittivo). Controllo della catena: factorized s1234
sugli operatori della cache dello store di HIFI3D (senza ritaglio) ridà gli embedding dello store (scarto massimo;
atteso <= ~1e-3, come i 1.1e-3 dell'emendamento 2).

**Righe, repliche, colonne.** Le righe all_cross dell'emendamento 2 (`bp_paired_e2.all_cross_rows`: HIFI3D e FaceScape
`all_cross` di E12, FaceVerse e neutra le pair_metrics senza filtro del crop; GT con `be.add_gts`), il loro seme, 1000
repliche per soggetto `default_rng(seme)`; maschera COMUNE per vista (righe con tutte le colonne finite). Colonne:
- bracci sull'ingresso ritagliato alla regione del modello m: `<braccio>@<m>|<distanza>`;
- bracci sull'ingresso intero (store, come l'emendamento 2): `<braccio>|<distanza>`;
- B: `<m>_vb_{coef,fr,sr}` sulle 600 mesh (`bp_paired_e2.vb600`);
- baseline geometriche sulle stesse 600 mesh (`blmm_eval.view_distances`): ICP + Chamfer in mm
  (`mm_rigid_icp_chamfer`), Chamfer pura in mm (`mm_chamfer_pure`), NICP su template in mm (`mm_nicp_template`);
- taglia oracolo (per la parziale).

**Gruppi** (stesse repliche per tutti): (a) **senza crop**: le righe di all_cross senza crop (99.000); (b) **all_cross**
(148.500); (c) **righe col crop** su almeno un lato (49.500); (d) **media dentro le coppie di topologie**: in ogni replica
lo Spearman dentro ognuna delle 15 coppie non ordinate di topologie diverse (9.900 righe ciascuna), poi la media sulle
15 ("media 15") e sulle 5 coppie col crop ("media crop 5"). Lettura preferita per il crop: (c) e "media crop 5" (il
pooling di (b) mescola coppie di topologie con scarti sistematici diversi e sovrastima il calo). In (a)-(c) Spearman,
parziale e quintile basso con `fact_paired._rep` / `summarize`; in (d) solo rho.

**Delta dichiarati** (FR: factorized s1234 / s2345 d_F cal., ctrlfr s1234 / s2345; SR: factorized d_P, ctrlfr; sempre
regione di m contro le colonne di m):
- D1: braccio@m - B di m (coefficienti, mesh FR, mesh SR): 4 bracci x 3 colonne x 2 modelli = 24 per GT, vista e
  gruppo, come le tabelle dell'emendamento 2;
- D2: braccio@m - baseline geometriche (3): 4 x 3 x 2 = 24 per GT, vista e gruppo;
- D3: braccio@m - braccio sull'ingresso intero (il recupero): 4 x 2 = 8 per GT, vista e gruppo.
C3M: gli stessi delta, descrittivi (non entrano nei conteggi).

**Criteri di lettura (fissati qui).** Per braccio dichiarato, regione e GT, sulla vista: recupero
F = (rho_c(braccio@m) - rho_c(braccio)) / (rho_a(braccio) - rho_c(braccio)), con rho_c = righe col crop (gruppo c) e
rho_a = senza crop (gruppo a) del braccio sull'ingresso intero (punti); calcolato solo se rho_a - rho_c >= 0.05.
- **dipendenza dal supporto** se D3 sulle righe col crop ha IC sopra 0 e F >= 0.5;
- **limite del descrittore** se D3 sulle righe col crop non ha IC sopra 0 o F < 0.25;
- **misto** altrimenti.
La lettura della vista e' quella della maggioranza dei bracci dichiarati per GT (FR: 4, SR: 4); si riportano tutti.
Va letta con la copertura (sopra): con copertura del crop >= 0.95 il recupero e' atteso per costruzione. Si riporta
anche se il ritaglio costa sulle righe senza crop (D3 nel gruppo a con IC sotto 0) e come cambiano D1 e D2 sulle righe
col crop rispetto all'emendamento 2.

## 2. Esperimento 2: il crop sugli held-out sintetici del training

**Ipotesi.** Il crop e' nella distribuzione di training (C3F e C3M, stesso generatore `mesh_ops.make_crop`, banda di
bordo fissa; anche il run massivo). Se i bracci sono invarianti al crop in-distribuzione, sugli held-out sintetici lo
Spearman e l'AUC delle coppie col crop sono vicini a quelli senza crop e log S non si sposta; se no, l'invarianza manca
gia' in-distribuzione.

**Mesh.** I 300 soggetti held-out di `fact_calib.py` (bfm, ict, gnm; 100 per dominio, `heldout_subjects`, quelli della
calibrazione di c). Il crop: l'etichetta `crop` dei soggetti nelle sorgenti della spec di C3M (`fact_calib.SPEC`,
`data_v3.collect_sources`; 100 per dominio, verificato), messa in scena come `fact_calib.stage` (geometria grezza,
tabella con area_mm2 = u_d^2 x area); operatori areanorm k_eig 128. Le 1.500 senza crop (down8k, noisy, original,
remesh, up60k): gli operatori della cache della calibrazione (`datasets/V3_OPS_CACHE/heldout_calib/ops`) e la sua
tabella, invariati. Embedding delle 1.800 mesh per braccio con la catena di `calib_heldout.sbatch` (eval_v3 +
zs_embed, `--dist_npz` = `c3f/gt_sr.npz`). Bracci: factorized s1234 / s2345, ctrlfr s1234 / s2345, C3M e123 / e205
(checkpoint come nell'esperimento 1). Controllo: per factorized e C3M gli embedding delle 1.500 senza crop coincidono
con `factorized/calib_heldout/<chiave>/embeddings.npz` (scarto massimo).

**GT.** SR: `c3f/gt_sr.npz` (`D_orig`, quella della calibrazione). FR: `c3f/gt_frcal.npz` (FR calibrata per dominio;
dentro un dominio ordina come FR). BFM: la taglia delle original REMESH non e' quella del modello (allineate per
similarita', `gt_sr.json` `bfm_caveat`): FR su BFM si riporta ma non si legge.

**Coppie e misure, per dominio** (stesso dominio, soggetti diversi): (i) senza crop: etichette diverse, entrambe senza
crop (le coppie di `fact_calib.calib_one`, 99.000); (ii) col crop: un lato crop, l'altro senza crop (49.500).
Spearman con FR (d_F cal., ctrlfr) e con SR (d_P, ctrlfr; anche C3M); delta (ii) - (i). AUC di verifica (distanza
minore = stesso soggetto; Mann-Whitney con le parita' a meta'): genuine = stesso soggetto, etichette diverse; impostori
= soggetti diversi; (i) entrambe senza crop, (ii) crop contro senza crop; delta (ii) - (i). Repliche: 1000 per
soggetto, `default_rng(1234)` per dominio, pesi c_a c_b (impostori e righe dello Spearman) e c_s (genuine); IC
percentile. Spostamento (factorized e C3M): d log S = log S(crop) - media di log S delle 5 senza crop, per soggetto;
media, sd, e la sd fra soggetti di log S medio (unita' di sd); per tutti: distanza nell'embedding crop -> senza crop
dello stesso soggetto, fra senza crop dello stesso soggetto, mediana fra soggetti diversi (original), come la
diagnostica dell'emendamento 2.

**Criteri di lettura (fissati qui).** Per braccio dichiarato e dominio (ict e gnm; bfm solo SR):
- **invarianza presente in-distribuzione** se il delta (ii) - (i) dello Spearman ha IC che contiene 0 o |delta| < 0.05,
  e |d log S medio| < 1 sd fra soggetti (factorized);
- **invarianza assente in-distribuzione** se il delta ha IC sotto 0 con |delta| >= 0.10, oppure |d log S medio| >= 2 sd;
- **parziale** altrimenti.
Si confronta con lo stesso calo sui domini di test (righe col crop contro senza crop, esperimento 1, braccio
sull'ingresso intero) e con lo spostamento di log S della diagnostica dell'emendamento 2.

## 3. Correzioni al testo (`conclusions.md`, quindi `results.md`)

1. Al posto di "lo scenario crop e' fuori dalla distribuzione di training dei bracci? non verificato": il crop E' nella
   distribuzione di training (C3F: REMESH 500 / 500, ICT 5000 / 5000, `gnm|crop`; C3M: la sua tabella di scala;
   stesso generatore `v2_work/genict/mesh_ops.py` `make_crop`, banda di bordo fissa), con perdita d'area comparabile
   (log del rapporto sqrt(area) crop / original: training BFM -0.080, ICT -0.104, GNM -0.117; test HIFI3D -0.108,
   FaceScape -0.131, FaceVerse -0.047; valori del critic, da riportare come tali), e anche il run massivo lo ha
   (`v3_work/stream/views.py`, peso 1.0). Piu' l'esito dell'esperimento 2.
2. Baseline geometriche su all_cross (stesse 148.500 righe, `aau/runs/evidence/baselines_mm/spearman.csv`): FaceScape
   FR ICP + Chamfer mm 0.381, Chamfer pura 0.570 [0.499, 0.641]; HIFI3D FR ICP + Chamfer mm 0.556, Chamfer pura 0.645;
   con il crop i bracci arrivano al livello di ICP o sotto. Piu' i delta D2 dell'esperimento 1 (anche sul braccio
   intero).
3. C3M crolla anch'esso sulle righe col crop (d_F cal., FR: HIFI3D 0.739 -> 0.472, FaceScape 0.711 -> 0.449; da
   ricalcolare qui e riportare accanto ai valori del critic). Conseguenza per il run massivo: ha lo stesso crop fisso
   nella distribuzione (stesso generatore, peso 1.0), quindi non c'e' ragione di attendersi che risolva da solo la
   fragilita'.
4. La fragilita' al crop era nota: `aau/runs/evidence/e1/summary.md` (C3F 0.716 -> 0.587, C3M 0.677 -> 0.509, GT
   maxabs, HIFI3D), dev FaceScape e108 0.642 -> 0.335, `paper/REPORT.md` (legge di Weyl sugli autovalori del
   Laplaciano). L'emendamento 2 non l'ha scoperta: l'ha misurata contro B.
5. Asimmetria della regione: B usa una regione per dominio stimata su 100 soggetti NON valutati con l'anello di bordo
   escluso (`bp.py` `region`); i bracci fanno pooling su tutta la superficie, bordo compreso. Piu' l'esito
   dell'esperimento 1.
6. Il pooling di all_cross sovrastima un po' il calo: per il crop si leggono le righe col crop o la media dentro le
   coppie di topologie (gruppi c e d).
Conteggi mancanti da aggiungere: FaceScape SR sulle righe col crop (a favore / contro / non risolte 0 / 19 / 5, da
ricontrollare su `paired_e2.csv`); FaceVerse neutra, le celle contro (7 secondo il critic, da ricontare su
`paired_e2.csv` e riportare col numero ricontato).

## 4. Calcolo

Esperimenti 1 e 2, ritagli, operatori ed embedding: UN job su `nv-ai-04` (`-p aicentre-a100 --qos=unprivileged
--gres=gpu:a100:4`, job breve, eccezione autorizzata; prelazionabile: ogni passo e' ripartibile, le uscite gia' su disco
si saltano), CPU del nodo per ritagli (venv open3d) e operatori (venv del training), quattro code di embedding in
parallelo (una per GPU). Mesh ritagliate e operatori su /tmp del nodo, cancellati all'uscita; su disco restano solo
statistiche del ritaglio (npz/json) ed embedding (npz piccoli) sotto `aau/runs/evidence/baselines_param/e3/`, fuori da
git. Delta: un job CPU su `prioritized` (venv della GT unificata). Prima di ogni allocazione `gpufree` e
`nodesummary`. Nessun job altrui toccato; `fact_paired.py`, `fact_summary.py`, `v3_work/stream/` non si modificano
(importati in sola lettura).

## 5. Passi e uscite

`aau/baselines_param/bp_e3.py` (venv open3d / del training): `crop` (ritagli e diagnostica, uscita
`e3/crop/<vista>_<modello>.npz` e `e3/crop_stats.json`), `stage-heldout` (crop held-out su /tmp e tabella unita),
`check-region`; `aau/baselines_param/bp_e3.sbatch` (job GPU: crop, tabelle, operatori, embedding in
`e3/embed/<vista>/<regione>/<braccio>/embeddings.npz` e `e3/embed_heldout/<braccio>/embeddings.npz`, controllo della
catena); `aau/baselines_param/bp_paired_e3.py` (venv della GT unificata): `paired` (esperimento 1, legge le GT di test)
e `heldout` (esperimento 2, GT sintetiche), uscite `paired_e3.csv`, `spearman_e3.csv`, `heldout_e3.csv`,
`controls_e3.json` e le sezioni dell'emendamento 3 in fondo a `results.md`. Ordine: commit di questo file -> job GPU ->
delta -> testo.
