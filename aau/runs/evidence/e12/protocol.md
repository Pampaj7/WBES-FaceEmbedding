# E12: GT canonica, protocollo (scritto il 9 ottobre 2026, PRIMA di ogni calcolo)

Nessun numero di E12 esiste a quest'ora. Codice in `v3_work/canonical_gt/`, numeri in `aau/runs/evidence/e12/`,
matrici GT in `datasets/CANONICAL_GT/` (fuori da git: derivano dai modelli con licenza).

## 0. Perche'

Tesi del paper (`paper/main_short.tex`, titolo e sezione "Why Alignment Hurts: Distance Compression", righe 22 e 425): un allineamento o una
normalizzazione SPECIFICI per identita' assorbono tratti identitari. Entrambe le GT in uso violano la tesi sulla
scala: la `maxabs` divide ogni mesh per il proprio max|coordinata| (l'estensione verticale in 492 casi su 500 di
HIFI3D), la `unificata` fa un Procrustes di similarita' per identita' (scala = centroid size). La GT canonica
non ha NESSUNA trasformazione per identita'.

## 1. Definizione

Per l'identita' i del dominio d:

1. **Forma nativa.** `V_i`: la forma neutra nel frame e nelle unita' dei DATI del dominio, la stessa che usano
   le GT esistenti: per HIFI3D, FaceVerse e FaceScape dev la mesh del modello dai pesi salvati
   (`v3_work/unified_gt/make_eval_gt.mapped_points`, testa intera); per FaMoS `V_neutral`
   (`datasets/FAMOS/{train,test}/FaMoS_subject_NNN.npz`, topologia FLAME, mm).
2. **Regione comune.** `p_i = mappa_d(V_i)`: la mappa baricentrica del dominio sulla regione unificata
   (`datasets/UNIFIED_GT/unified_space.npz`, `vidx_<d>`, `bary_<d>`, 1478 punti), la stessa della GT unificata.
3. **Frame canonico, UNA trasformazione per dominio.** `c_i = s_d R_d p_i + t_d`, con `(s_d, R_d, t_d)` di
   `datasets/UNIFIED_GT/canonical_transforms.json`. E' la similarita' ai minimi quadrati dal TEMPLATE MEDIO del
   dominio alla media FLAME (`unified.py`): dipende solo dal dominio, e la sua scala contiene gia' il cambio di
   unita' (BFM µm -> s ~ 1e-3, GNM m -> s ~ 1e3). Quindi `s_d = u_d * k_d`: `u_d` = conversione delle unita' in
   mm, `k_d` = rapporto fra la taglia della media FLAME e quella della media del dominio. Verifiche al punto 2.
   FaMoS e' gia' FLAME in mm: `s = 1, R = I, t = 0`.
4. **Distanza.** `g_ij = sqrt( sum_v w_v ||c_i,v - c_j,v||^2 / A )`: RMS pesato per area fra vertici
   corrispondenti, in mm. Pesi `w` = aree baricentriche dei vertici sulla media globale `mu`
   (`unified_space.npz`, `w`), `A = sum w` (22.819 mm^2): gli stessi pesi, uguali per tutte le mesh, della GT
   unificata. Cosi' fra canonica e unificata cambia solo il Procrustes per identita'.
5. **Nessun centraggio, nessuna scala, nessuna rotazione per identita'.**

Dentro un dominio la trasformazione e' comune a tutte le identita': `g` e' la RMS delle differenze delle
coordinate native, moltiplicata per `s_d`. Lo Spearman dentro un dominio non dipende quindi da `(s_d, R_d, t_d)`;
la trasformazione conta per i mm assoluti e per i confronti fra domini.

**Varianti (dichiarate ora, tutte sugli stessi punti, stessi pesi):**
- `canonical_centered` (secondaria, richiesta): `c_i - centroide_w(c_i)`. Isola la traslazione.
- `canonical_rigid` (diagnostica, solo nella sezione delle correlazioni fra GT): rotazione + traslazione per
  identita' verso `mu` (Umeyama pesato, scala 1). Con `unified` (rotazione + traslazione + scala) completa la
  catena canonica -> traslazione -> rotazione -> scala, e serve per FaMoS (sotto).
- `canonical_phys` (controllo delle unita'): `u_d R_d p_i` con la sola conversione fisica delle unita' (punto
  2). Dentro un dominio e' proporzionale alla canonica: lo Spearman deve venire 1 (controllo), cambiano i mm.

**FaMoS: regola fissata ora.** `V_neutral` sta nel frame di cattura (la registrazione FLAME ha la posa della
testa del fotogramma; la neutra e' nel frame del fotogramma medoide). Si misura, per i 95 soggetti, la
rigida verso `mu`: angolo di rotazione e traslazione. **Se il p95 dell'angolo supera 2 gradi o il p95 della
traslazione supera 5 mm, la canonica pura su FaMoS non e' definita** (misurerebbe la posa di cattura) e la GT
canonica di FaMoS e' `canonical_rigid` (rotazione + traslazione per identita', SENZA scala), dichiarata come
deviazione. La canonica pura si riporta comunque. Le altre GT di FaMoS si calcolano sulla neutra senza posa:
- `unified`: quella esistente (`datasets/FAMOS/test_view/gt_matrix.npz`, invariante alla similarita');
- `maxabs`: non esiste; definita qui come la gemella di `build_zs_gt.py`: neutra portata nel frame canonico con la
  rigida sopra, vertici della maschera `face` di FLAME (`v2_work/genflame/official/FLAME_masks.pkl`, 1787,
  l'equivalente della patch del volto delle viste), `normalize_maxabs` per mesh, media per vertice della
  distanza L2 (`vertex_mean_l2_matrix`).

## 2. Verifiche

**(a) La trasformazione e' per dominio.** Si ricalcola `(s_d, R_d, t_d)` dal template medio del dominio con la
stessa procedura di `unified.py` (Umeyama pesato per l'area FLAME, dalla mappa del template alla media FLAME) e
si confronta col json (max |diff|). Domini: tutti quelli caricabili (flame, bfm, ict, gnm, facescape, hifi3d,
faceverse, multiface). In piu': la media delle identita' canoniche di un dominio contro il template canonico del
dominio (mm), e la distribuzione della RMS di ogni identita' canonica dalla media FLAME (deve variare: la
trasformazione non porta le identita' sulla media).

**(b) Unita' native.** Due misure indipendenti dalla similarita' del json:
- **distanza interpupillare (IPD)** sul template medio, in unita' native: centro di ogni occhio = media dei 6
  landmark iBUG del contorno (36-41, 42-47; embedding FLAME 2020 di `famos_common.flame_lmk68`), portati in
  ogni dominio tramite il vertice della regione unificata piu' vicino sulla media FLAME (distanza del
  sostituto dichiarata). Riferimenti: IPD della media FLAME in mm (stessa costruzione) e il valore
  antropometrico 63 mm (adulti, media; atteso 60-65);
- **scala del json** `s_d` e `k_d = s_d / u_d`.
`u_d` = la potenza di 10 (µm, mm, cm, dm, m) piu' vicina, in log, a `63 / IPD_nativa`; se nessuna sta entro il
±15%, l'unita' e' dichiarata arbitraria e i mm passano solo dall'ancoraggio alla media FLAME del json. Si
riportano anche le unita' dichiarate nel codice (`domains.py`).

**(c) Dimensione del volto dentro un dominio.** Per ogni identita', nel frame canonico: centroid size pesata
sulla regione (`sqrt(sum_v W_v ||c_v - centroide||^2)`, W normalizzati) e IPD. CV = dev. std. / media. Atteso
4-8%. Insiemi: i 500 di HIFI3D, FaceVerse, FaceScape dev; i 95 (e i 15 di TEST) di FaMoS; per i domini di training
quelli gia' usati da `shapes.py` che si leggono senza shard pesanti (FLAME 1000 N(0, 1), BFM 500 REMESH, ICT-5000,
GNM_DISTILL 10.100, Multiface 13). Le mesh BFM REMESH sono gia' allineate per similarita' una per una
(`domains.bfm`): ci si attende un CV vicino a zero, da segnalare.

## 3. Correlazioni fra GT

Spearman fra le GT sulle coppie dei soggetti valutati, IC 95% bootstrap per soggetto (1000 repliche, seme 1234,
peso di una coppia = prodotto dei conteggi, come E8 sez. 4), e punto sul pool intero:
- **HIFI3D, FaceVerse, FaceScape dev:** 100 soggetti di `select_subjects(<vista>/npz, 1234)` (4.950 coppie) e
  pool di 500;
- **FaMoS TEST:** i 15 soggetti di `aau/famos/split.json` (105 coppie); punto anche sui 95.
Coppie di GT: canonica contro maxabs e contro unificata; unificata contro maxabs (controllo: HIFI3D 0.543 e
FaceVerse 0.693 di E8); centrata e rigida contro canonica e contro maxabs; `canonical_phys` contro canonica
(controllo = 1); contro la GT `raw` di `build_zs_gt.py` dove c'e' (patch nativa, media delle norme, nessuna
normalizzazione: controllo di coerenza, attesa alta).

Ci si aspetta di ritrovare i numeri del critic (job 1062067): GT in mm grezzi contro maxabs 0.41, contro unificata
0.57. Non e' un criterio: la sua GT non e' documentata.

## 4. Metodi rivalutati

Nessuna distanza ricalcolata: righe, distanze e repliche sono quelle dei summary esistenti; cambia SOLO la colonna
della GT (`zs_summarize.with_gt` per nome di soggetto). GT: `maxabs` (quella delle righe), `unified`, `canonical`,
`canonical_centered`. IC 95% bootstrap per soggetto, 1000 repliche, UN seme per (dominio, gruppo), quello della
differenza pubblicata e108 - Chamfer eval (come `v3_work/unified_gt/eval_methods.py`), cosi' righe, delta fra
metodi e delta fra GT stanno sulle stesse repliche.

- **HIFI3D** (`aau/runs/ws_hifi3d/data_328f2bfc1a`, 100 soggetti, 6 topologie): righe di
  `zs_arcface_vs_scale.frame_a` (e036, e072, e108, congiunto 1019532, Chamfer eval, ArcFace ombreggiato e normal
  map 3 viste), faceBench (Chamfer 4096 pt, ICP + Chamfer, ICP + NICP P2Tri), competitori di
  `aau/runs/competitors_hifi3d` (`comp_summarize.competitor_distances`: NICP su template, Uni3D, OpenShape,
  ShapeDNA k=50 e k=100; righe primarie nel frame nativo; le altre righe del file nei csv). Gruppi:
  `nocrop_cross` (PRIMARIO, 20 coppie ordinate di topologie), `all_cross` (30), `subject_pair_mean` (media per
  coppia di soggetti sulle 30). Seme di `all_cross`: lo stesso token con il gruppo `all_cross`.
- **FaceVerse** (secondario, come E8: `aau/runs/ws_faceverse_expr`, vista con espressioni, GT d'identita'
  neutra): righe di `zs_expr_summarize.secondary_frame`; e036, e072 (ora completo), e108, congiunto nelle due
  convenzioni, Chamfer eval, faceBench, ArcFace. Gruppi `mesh_pair_nocrop`, `subject_pair_mean_nocrop`. Non ci
  sono competitori.
- **FaceScape dev** (`aau/runs/ws_dev_facescape/data_aca84a16c6`, vista neutra): righe di
  `dev_fs_summarize.pair_frame` (e108 dagli embedding, Chamfer eval dal breakdown, Chamfer intera e su regione
  stabile 4096 pt). Gruppi `nocrop_cross`, `all_cross`, `subject_pair_mean`. Seme: quello di
  `devfs_paired` (neutral, maxabs, gruppo, scale_e108, raw_chamfer). Non ci sono faceBench, ArcFace, competitori
  ne' altri bracci.

**Controlli (devono tornare, altrimenti ci si ferma):** con la GT maxabs i punti pubblicati di ogni riga e la
differenza appaiata e108 - Chamfer eval (punto e IC) coincidono con: `data_scale_ood/arcface_vs_scale_hifi3d`,
`data_scale_ood/hifi/{baselines,table_cells,paired}.csv`, `competitors_hifi3d/spearman.csv` (punti, anche con la
GT unificata), `evidence/e8/methods_spearman.csv` (maxabs e unificata, punto e IC: stesso seme),
`ws_faceverse_expr/secondary.csv`, `evidence/dev_facescape/graded.csv`. La GT unificata ricalcolata qui coincide
con quella su disco.

**Tabella finale:** per dominio e gruppo, una riga per metodo, colonne GT maxabs / unificata / canonica (punto e
IC) piu' la centrata; delta appaiati contro e108 e contro Chamfer eval per ogni GT, con P(delta <= 0).

## 5. Fuori scope

Nuovi embedding o distanze; GT di training canonica; studio umano; riconoscimento (non usa la GT).

---

## Emendamento 1 (9 ottobre 2026, ore 13, PRIMA di ogni numero; richiesta del PI)

Stato al momento dell'emendamento: nessun numero di E12 esiste. L'unico job lanciato (1062668) e' rimasto in
coda sul nodo `cpu` in drain ed e' stato annullato con 0 s di esecuzione, senza log ne' uscite. Hash del testo
sopra (sezioni 0-5): `7d911e52...` (`protocol.sha256` della prima versione). Dove l'emendamento e la prima
versione divergono, vale l'emendamento.

**Principio.** La GT dev'essere invariante SOLO a cio' che non si puo' osservare nei dati di destinazione,
tramite un riferimento scelto per identita' (mai per coppia), con stimatori robusti.

### E1. GT costruite

Tutte sulla regione unificata (1478 punti, mappe di `unified_space.npz`), pesi d'area FISSI `w` presi dalla
media globale `mu` (uguali per tutte le mesh: ogni GT per vertici e' una distanza euclidea pesata). Distanza per
vertici = `sqrt(sum_v w_v ||x_i,v - x_j,v||^2 / A)`.

1. **GT-F, forma metrica in mm** (sostituisce la "canonica" della sezione 1, che usava la scala del json).
   - **La similarita' del json contiene una scala** (lo si verifica): `s_d = u_d * k_d`, con `k_d` = taglia
     della media FLAME / taglia della media del dominio. GT-F NON usa `k_d`.
   - **Unita'.** `u_d` = conversione in mm se l'unita' e' NOTA, cioe' dichiarata come unita' fisica in
     `domains.py` (FLAME metri, BFM micrometri, ICT centimetri, GNM metri, FaceScape mm, Multiface mm); per le
     unita' IGNOTE (HIFI3D "unita' del .mat", FaceVerse "unita' del .npy") UN solo fattore per modello,
     `u_d = 63 mm / IPD` della forma media del modello (IPD come in sezione 2b). Per le unita' note l'IPD
     resta un controllo: un disaccordo oltre il ±15% si segnala, non cambia `u_d`.
   - **Frame.** Una sola rigida per dominio, media -> media: Umeyama SENZA scala da `u_d * mappa(template)`
     alla media FLAME in mm (pesi d'area FLAME, come `unified.py`). La rotazione coincide con quella del json
     (la scala non cambia la rotazione di Umeyama); cambia la traslazione.
   - `f_i = u_d R_d p_i + t_d`. Nessuna trasformazione per identita' sui 3DMM.
   - **GT-F su catture reali** (FaMoS; Multiface nel controllo secondario): la posa della testa e' un disturbo
     vero, quindi GT-F = rigida robusta per identita' verso `mu` (sotto). Sostituisce la regola sulla posa
     della sezione 1 (la dispersione della posa si riporta comunque).
   - Varianti: **GT-F-rig-LS** (rigida per identita' verso `mu`, minimi quadrati pesati, scala 1) e
     **GT-F-rig-rob** (resistant fit, sotto); servono a misurare quanto assorbe un allineamento rigido per
     identita' (effetto Pinocchio). **GT-F-centrata** (sola traslazione per identita'): la variante secondaria
     della sezione 1, solo sui 3DMM.
   - **Resistant fit:** IRLS con perdita di Tukey (biweight) sui residui per punto `r_v = ||R x_v + t - mu_v||`;
     pesi `w_v * (1 - (r_v / c)^2)^2` per `r_v < c`, 0 altrimenti; `c = 3 x` mediana pesata (pesi d'area) dei
     residui, ricalcolata a ogni giro (circa 4.6 sigma per residui gaussiani isotropi); partenza dalla rigida
     LS; al massimo 100 giri, fine quando R cambia meno di 1e-9 e t meno di 1e-7 mm. Si riportano giri e frazione
     d'area con peso ridotto.
2. **GT-EDM, senza allineamento** (EDMA, Lele e Richtsmeier 1991). K = 400 punti corrispondenti campionati
   uniformemente per area sulla regione: triangoli della regione su `mu` estratti con probabilita'
   proporzionale all'area, coordinate baricentriche uniformi, seme 1234; ogni punto e' la stessa combinazione
   baricentrica dei vertici della regione in tutte le forme. `FM_i` = le 79.800 distanze interne (a < b) dei
   punti di f_i (o dei punti nativi in mm: e' invariante alla rigida). `GT-EDM_ij = sqrt(mean_ab (FM_i - FM_j)^2)`
   (Frobenius con pesi uniformi, in mm). **GT-EDM-s** (senza scala): `FM_i / gm_i`, con `gm_i` la media
   geometrica delle sue distanze, moltiplicato per `gm(mu)` per leggerlo in mm alla taglia di riferimento.
3. **GT-S, forma senza taglia:** `s_i = m + (f_i - m) * CS(mu) / CS_i`, con `CS_i` la centroid size pesata della
   regione di f_i (attorno al suo centroide) e `m` il centroide pesato di `mu`, fisso: una sola scala per
   identita', attorno a un punto fisso, nessuna rotazione ne' traslazione per identita'. Sulle catture reali
   f_i e' quella con la rigida robusta. Riferimento: **Procrustes completo = GT unificata** (invariata).
4. **Maxabs legacy**, per continuita' (sezione 1). Su FaMoS la definizione della sezione 1 con la rigida
   ROBUSTA al posto della LS; su Multiface sulla regione (la topologia tracked non ha una patch del volto).

**Baseline banali (obbligatorie):** "solo taglia" `|log CS_i - log CS_j|` e "solo altezza" `|log H_i - log H_j|`,
H = estensione verticale (max y - min y) della regione in GT-F. Si riporta lo Spearman di ciascuna con ogni GT
(quanto di una GT spiega un solo scalare), e la loro identificabilita' (sotto) come pavimento.

**Controlli aggiunti:** canonica col json (`s_d` intera) contro GT-F: Spearman 1 dentro ogni dominio
(differiscono per un fattore costante e una traslazione comune).

### E2. Arbitro: identificabilita' su catture reali ripetute

- **FaMoS (primario).** Catture = il PRIMO fotogramma di ogni sequenza (`first_frames` di
  `famos_subsample.py`; le sequenze di FaMoS sono riprese diverse della stessa sessione), registrazioni FLAME
  in mm. Persone: **tutte le 95** registrate (TRAIN + TEST). Qui non si addestra niente: le persone di TRAIN
  servono solo a rendere il compito non saturo (con 15 persone tutto starebbe vicino a 1); le 15 di TEST si
  riportano a parte. **Tutti** i primi fotogrammi, senza filtri: il filtro della neutra di
  `famos_subsample.py` (distanza dal medoide nella GT unificata) favorirebbe le GT simili all'unificata;
  la versione filtrata (`neutral_from`) e' una sensibilita'.
- **Multiface (secondario, fuori dalla regola).** Ogni soggetto ha UNA sola ripresa neutra (10-20 fotogrammi
  dello stesso segmento, circa 15 s): non sono catture ripetute da sequenze diverse. Si riporta come
  controllo della ripetibilita' dentro una ripresa (tracking e posa), 13 persone.
- **Misure per ogni GT** (coppie genuine = stessa persona, sequenze diverse; impostore = persone diverse):
  - AUC di verifica = P(d genuina < d impostore) (Mann-Whitney, pareggi a meta');
  - rank-1 di identificazione: ogni cattura come richiesta contro tutte le altre catture (galleria fissa);
  - rapporto intra / inter = mediana delle distanze genuine / mediana delle distanze impostore.
  IC 95%: bootstrap per persona (1000 repliche, seme 1234, le stesse per tutte le GT): coppia genuina della
  persona p pesata `c_p`, impostore (p, q) pesata `c_p c_q`, richieste di rank-1 pesate `c_p` (galleria fissa);
  differenze appaiate fra GT sulle stesse repliche.
- GT valutate: F (rigida robusta), F-rig-LS, S, EDM, EDM-s, unificata, maxabs, F pura (frame di cattura, per
  mostrare la posa), solo taglia, solo altezza.

**Regola di selezione (fissata ora).** Candidate principiate: **F, S, EDM, EDM-s**. Ordine di semplicita'
(gradi di liberta' tolti per identita' sui 3DMM, poi costruzione): F (0) < S (1) < EDM (6, rigida implicita) <
EDM-s (7).
1. Sulle 95 persone FaMoS, primi fotogrammi senza filtri: la candidata con l'AUC piu' alta. Sono "pari" a lei le
   candidate la cui differenza appaiata d'AUC con lei ha l'IC 95% che contiene 0.
2. Se le pari sono piu' d'una (p.es. AUC saturo), fra loro decide il rapporto intra / inter (piu' basso =
   meglio), con la stessa regola di parita'.
3. Fra le ancora pari, la piu' semplice.
La GT primaria e' quella scelta; si riportano sempre tutte. Il passo 2 e' una mia aggiunta alla regola del PI
("a parita' entro gli IC la piu' semplice"), per non decidere a vuoto se l'AUC satura: si riporta anche l'esito
della regola stretta (AUC, poi semplicita').

### E3. Effetto su sezioni 3 e 4

- **Spearman fra GT (sezione 3):** tutte le GT di E1, piu' raw e le due baseline, sugli stessi soggetti e con gli
  stessi IC.
- **Metodi (sezione 4):** tutti i metodi rivalutati con TUTTE le GT: maxabs, unificata, F, F-centrata,
  F-rig-LS, F-rig-rob, S, EDM, EDM-s. Delta appaiati contro e108 e contro Chamfer eval per ogni GT. Tabella
  principale: maxabs, unificata, F, S, EDM, EDM-s; le varianti di F in una tabella a parte.
