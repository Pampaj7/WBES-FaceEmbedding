# NoW validation: la metrica su ricostruzioni reali (critica 2 del rebuttal)

Scritto il 7 ottobre 2026, PRIMA di qualunque numero: al momento della stesura le
ricostruzioni sono in corso (job 1060226) e nessuna metrica e' stata calcolata. Ogni
deviazione successiva va annotata in fondo, con la data, nella sezione "Deviazioni".

## Dati

- NoW validation: 20 soggetti FaMoS con scansione (mm; frame x a destra, y in su, z in
  avanti) e 7 landmark annotati (`scans_lmks_onlypp`, ordine NoW: ex_r, en_r, en_l, ex_l,
  subnasale, ch_r, ch_l). 352 immagini iPhone da `imagepathsvalidation.txt`: 108
  multiview_neutral, 114 multiview_expressions, 86 multiview_occlusions, 44 selfie; da 12 a
  29 immagini per soggetto, 7 soggetti senza una delle quattro sfide.
- Tutto (immagini, scansioni, mesh ricostruite, render, operatori) sta FUORI dal repo, in
  `~/data/now`, `~/data/now_recon`, `~/data/now_eval_work`: la licenza NoW e' di ricerca
  non commerciale e non consente la redistribuzione. Nel repo entrano solo codice e
  numeri aggregati o per-immagine.

## Metodi di ricostruzione

3DDFA_V2, SynergyNet, PRNet: pesi pubblici, codice dei cloni senza modifiche, gli stessi
runner di WS3b (`aau/recon/*_run.py`), detector FaceBoxes con il box piu' grande. Le
immagini su cui un metodo non trova il volto restano mancanti (contate, non imputate).
DECA / EMOCA / MICA (FLAME) solo se integrabili in tempi ragionevoli; altrimenti lo si
dichiara.

Convenzioni, uguali a WS3b: le mesh escono in spazio immagine (y in giu') e la y si nega
prima di tutto (e' una conversione di mano, non un allineamento). I 7 landmark della
ricostruzione sono gli iBUG 36, 39, 42, 45, **33**, 48, 54 dei 68 che ogni metodo
restituisce: 33 (subnasale) e non 30 (punta) perche' il quinto punto NoW e' il "nose
bottom" (`compute_mask` del codice ufficiale; e' anche la scelta di DECA/MICA). Nelle
scansioni il quinto punto sta circa 48 mm sotto la linea degli occhi, cioe' al subnasale.

## (a) Errore NoW ufficiale

`external/now_evaluation/compute_error.py` (commit 7bd1498), senza modifiche, sulle
predizioni esportate nel layout ufficiale (`<soggetto>/<sfida>/<IMG>.obj` + `.npy` con i
7 landmark). Il protocollo: ritaglio della scansione attorno al centro del volto
(`compute_mask`), similarita' Procrustes dai 7 landmark, raffinamento rigido con scala
che minimizza la distanza scan-to-mesh, distanza punto-superficie dai vertici della
scansione ritagliata. Statistiche come nella challenge: mediana, media e deviazione
standard di TUTTE le distanze concatenate delle immagini di un metodo; piu' la stessa
cosa per sfida. Se l'ambiente ufficiale (psbody-mesh, chumpy, sbody) non si costruisce,
si usa una reimplementazione documentata (stesso ritaglio, stessa Procrustes, ICP
punto-superficie con scala al posto del dogleg di chumpy) e la si dichiara tale.

Classifica primaria sul sottoinsieme di immagini ricostruite da TUTTI i metodi; quella su
tutte le immagini disponibili si riporta accanto.

## (b) Distanza ricostruzione-scansione con le nostre metriche

**Frame canonico e regione volto (identici per scansione e ricostruzione).**
Template T7 = media di Procrustes generalizzata (con scala, riportata alla dimensione di
centroide media in mm) dei 7 landmark delle 20 scansioni. Ogni mesh -- scansione o
ricostruzione -- va nel frame di T7 con la similarita' (senza riflessione) che porta i
SUOI 7 landmark su T7: la ricostruzione non vede mai la scansione del proprio soggetto.
Poi lo stesso ritaglio NoW (`compute_mask`: centro = subnasale + 0.3 (radice del naso -
subnasale), raggio = 1.4 (ex-ex + naso) / 2) calcolato sui landmark della mesh stessa, e
decimazione quadrica (`mesh_ops.decimate_to`) a 5215 triangoli, la risoluzione di
`gt_face` in WS3b. Ne escono `scan_face/<soggetto>` e `recon_face/<metodo>/<nome>`.

Distanze, una riga per ricostruzione, contro `scan_face` dello stesso soggetto:

- `latent_joint`: norma L2 fra i latenti del modello congiunto BFM+ICT
  `x3dmm_joint_bfm_ict_s1234_1019532` (checkpoint `best_by_xtopo_mesh_clean`), operatori
  ad area unitaria k_eig 128 (`v2_work/potential/areanorm_operators.py`), la catena di
  `ws3b_latent.py`.
- `chamfer_raw`: Chamfer simmetrica, ogni mesh centrata e divisa per il proprio maxabs,
  4096 punti per lato (variante facebench di WS3b). Nessun allineamento oltre il frame
  canonico.
- `icp_chamfer_mm`: similarita' ICP (`fg_metrics.rigid_icp_align`, 30 iterazioni, 4096
  punti) della scansione sulla ricostruzione, poi la stessa Chamfer in mm nel frame
  della scansione.
- `arcface_normals`: 1 - coseno fra gli embedding ArcFace medi sulle viste (yaw 0, -30,
  +30, 512 px) dei render a normal map, crop fisso calibrato sui render ombreggiati
  delle stesse mesh (pipeline di `aau/zs3dmm/zs_arcface_render.py` e
  `aau/multiface/ws3a_perceptual.py`).

## (c) Classifica dei metodi e concordanza con NoW

- Punteggio di un metodo per metrica: media delle distanze per ricostruzione (NoW: la
  mediana ufficiale). CI 95% e p_first con 1000 repliche bootstrap sui 20 soggetti, seme
  1234.
- Concordanza 1, classifica: Kendall tau fra l'ordine dei metodi dato da ogni metrica e
  quello dato da NoW. Con tre metodi tau vale solo -1, -1/3, 1/3, 1: e' un'indicazione,
  non un test.
- Concordanza 2, per immagine (PRIMARIA): per ogni immagine ricostruita da tutti i metodi,
  Kendall tau fra l'ordine dei metodi secondo la metrica e secondo l'errore NoW mediano di
  quell'immagine; media sulle immagini, CI bootstrap sui soggetti.
- Concordanza 3, per ricostruzione: Spearman fra metrica ed errore NoW per immagine, su
  tutte le ricostruzioni di tutti i metodi; CI bootstrap sui soggetti.
- Confronto pre-registrato: `latent_joint` contro `chamfer_raw` (le due misure senza
  allineamento alla verita' a terra) sulle concordanze 2 e 3, delta appaiato sulle stesse
  repliche.

Lettura fissata ora. `icp_chamfer_mm` e' quasi la stessa misura di NoW (allineamento
alla scansione, distanza in mm): la sua concordanza e' il tetto atteso, non un risultato.
Se `latent_joint` concorda con NoW almeno quanto `chamfer_raw`, la metrica appresa ordina
le ricostruzioni reali in modo coerente con il benchmark di riferimento senza chiedere
landmark ne' allineamento alla scansione. Se concorda meno, lo si scrive: NoW misura
l'errore geometrico dopo l'allineamento, non l'identita', e la discordanza va letta
insieme a (d). Concordanze vicine a zero per tutte le metriche vorrebbero dire che, su
questi tre metodi e queste immagini, le differenze fra metodi sono sotto il rumore.

## (d) Identita' fra ricostruzioni (nessuna verita' a terra)

Per ogni metodo, separatamente: matrice delle distanze fra tutte le sue ricostruzioni,
per ogni metrica (`latent_joint`, `chamfer_raw`, `icp_chamfer_mm`, `arcface_normals`, le
stesse definizioni di (b) applicate a due ricostruzioni). Protocollo di
`aau/zs3dmm/zs_expr_summarize.py`:

- verifica: AUC di -distanza, coppie i < j, stessa persona contro persone diverse; CI con
  i soggetti ricampionati e coppia pesata come in `recognition_values` (stessa persona:
  il conteggio del soggetto; persone diverse: il prodotto dei conteggi);
- retrieval: ogni ricostruzione e' una query; la distanza query-soggetto e' il minimo
  sulle ALTRE ricostruzioni di quel soggetto (cosi' la galleria ha un solo rilevante per
  query, come in zs_expr); rank-1 e mAP (= MRR) fra i 20 soggetti, caso 1/20 = 0.05;
- blocco secondario: query nelle sfide non neutre, galleria le sole multiview_neutral;
- delta appaiati `latent_joint` - ogni baseline sulle stesse repliche, P(delta <= 0).

Lettura: AUC e rank-1 alti vogliono dire che le ricostruzioni dello stesso soggetto da
foto diverse sono piu' vicine fra loro che a quelle di altri, secondo quella metrica. A
metodo fissato, il confronto fra metriche dice quale misura vede meglio l'identita' nelle
ricostruzioni reali; a metrica fissata, il confronto fra metodi dice quanto ogni metodo
conserva l'identita'. Avvertenza: una metrica puo' riconoscere il soggetto anche da
indizi non anatomici che le ricostruzioni ereditano dalla sessione di ripresa (luce,
occhiali, posa tipica delle foto di quel soggetto); il blocco con galleria neutra lo
limita in parte, non lo elimina.

## Deviazioni

Tutte del 7 ottobre 2026.

1. **MICA aggiunto** (previsto sopra come facoltativo): `aau/recon/mica_run.py`, catena di
   `demo.py` del clone, FLAME2020 licenziato convertito senza chumpy (fuori dal repo). Esce
   la forma canonica neutra in mm, gia' destrorsa: la y NON si nega (`coordinate_system`
   nel json). Il ritaglio FLAME ha meno triangoli del bersaglio: suddivisione 1->4 a punto
   medio prima della decimazione. DECA ed EMOCA non integrati (pytorch3d / rasterizzatore
   da compilare, pesi da recuperare): fuori tempo.
2. **Controllo del codice ufficiale.** La scansione identica coi landmark esatti fa
   produrre NaN al dogleg di chumpy (residuo zero dopo la Procrustes, job 1060237). Il
   controllo usa la scansione spostata con una similarita' nota (metri, 20 gradi) e
   landmark con 2 mm di rumore. Per l'import senza libGL, `psbody.mesh.meshviewer` e'
   sostituito da uno stub (`now_official.py`); nessuna riga del codice ufficiale cambia.
3. **Verso dei triangoli.** Le patch di 3DDFA_V2, SynergyNet e PRNet arrivavano con le
   normali verso l'interno (negazione della y), scansioni e MICA verso l'esterno; gli
   operatori gradiente di DiffusionNet dipendono dal verso. Tutte le patch ora sono
   orientate verso l'esterno (convenzione ICT) dopo la decimazione: vertici identici bit a
   bit (verificato su 1428 patch), cambia solo il latente. Effetto osservato piccolo.
4. **Bug nel ritaglio.** `ws3b_prepare_meshes.crop_region`, riusato all'inizio, scambia gli
   indici dei vertici quando `prepare_open_surface` toglie una componente: colpite le
   scansioni FaMoS_180507_03345_TA e FaMoS_180502_03341_TA e 6 ricostruzioni PRNet (patch
   piene di triangoli spuri; latente ~10 contro ~1.5). Sostituito da `crop_patch`; tutte le
   metriche (b)-(d) ricalcolate. L'errore NoW ufficiale non ne dipende (usa la scansione
   grezza). I numeri della corsa col bug restano in `_superseded_crop_bug/` (e quelli
   prima del punto 3 in `_superseded_winding_misto/`) e NON vanno citati.
5. **Primaria sui 3 metodi pre-registrati** (revisione del critic). La concordanza
   PRIMARIA e il delta pre-registrato si calcolano solo su 3DDFA_V2, SynergyNet e PRNet,
   come scritto sopra; la versione con MICA (deviazione 1) e' secondaria, perche' il segno
   del latente dipende da MICA. Aggiunta l'analisi di sensibilita' sulla storia di
   tassellazione di MICA (Loop x2 al posto del punto medio) su tutte le 352 immagini.
