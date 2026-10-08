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

---

# Risultati

Metodi: 3ddfa_v2, synergynet, prnet, mica. Insieme comune: 352 immagini su 352 (ricostruite da tutti i metodi e con tutte le metriche), 20 soggetti. CI 95% bootstrap sui soggetti, 1000 repliche, seme 1234.

## (a) Errore NoW ufficiale (mm)

| metodo | blocco | immagini | mediana | media | std |
| --- | --- | --- | --- | --- | --- |
| 3ddfa_v2 | comune | 352 | 1.390 | 1.790 | 1.666 |
| 3ddfa_v2 | multiview_neutral | 108 | 1.347 | 1.755 | 1.665 |
| 3ddfa_v2 | multiview_expressions | 114 | 1.437 | 1.827 | 1.689 |
| 3ddfa_v2 | multiview_occlusions | 86 | 1.366 | 1.743 | 1.590 |
| 3ddfa_v2 | selfie | 44 | 1.428 | 1.870 | 1.741 |
| 3ddfa_v2 | tutte (json ufficiale) | 352 | 1.390 | 1.790 | 1.666 |
| synergynet | comune | 352 | 1.350 | 1.687 | 1.425 |
| synergynet | multiview_neutral | 108 | 1.288 | 1.619 | 1.368 |
| synergynet | multiview_expressions | 114 | 1.360 | 1.656 | 1.346 |
| synergynet | multiview_occlusions | 86 | 1.464 | 1.826 | 1.539 |
| synergynet | selfie | 44 | 1.287 | 1.673 | 1.511 |
| synergynet | tutte (json ufficiale) | 352 | 1.350 | 1.687 | 1.425 |
| prnet | comune | 352 | 1.563 | 2.051 | 2.007 |
| prnet | multiview_neutral | 108 | 1.491 | 2.019 | 2.100 |
| prnet | multiview_expressions | 114 | 1.546 | 2.004 | 1.947 |
| prnet | multiview_occlusions | 86 | 1.665 | 2.130 | 1.976 |
| prnet | selfie | 44 | 1.602 | 2.098 | 1.980 |
| prnet | tutte (json ufficiale) | 352 | 1.563 | 2.051 | 2.007 |
| mica | comune | 352 | 0.910 | 1.125 | 0.939 |
| mica | multiview_neutral | 108 | 0.911 | 1.137 | 0.961 |
| mica | multiview_expressions | 114 | 0.912 | 1.134 | 0.951 |
| mica | multiview_occlusions | 86 | 0.892 | 1.092 | 0.898 |
| mica | selfie | 44 | 0.938 | 1.138 | 0.929 |
| mica | tutte (json ufficiale) | 352 | 0.910 | 1.125 | 0.939 |

Controllo del codice ufficiale (scansione come predizione di se stessa): FaMoS_180424_03335_TA mediana 0.0002 mm, max 0.0008 mm; FaMoS_180426_03336_TA mediana 0.0002 mm, max 0.0008 mm.

## (b) Distanza ricostruzione-scansione e (c) classifica dei metodi

Punteggio: NoW = mediana ufficiale sull'insieme comune; le altre = media delle distanze per ricostruzione. Piccolo = meglio. `p_first` = frazione di repliche in cui il metodo e' primo.

| metrica | 1. | 2. | 3. | 4. | Kendall tau con NoW (punto; media bootstrap; P(tau=1)) |
| --- | --- | --- | --- | --- | --- |
| NoW ufficiale (mm) | mica 0.91 [0.84, 0.99] (1.00) | synergynet 1.35 [1.18, 1.54] (0.00) | 3ddfa_v2 1.39 [1.21, 1.62] (0.00) | prnet 1.56 [1.40, 1.77] (0.00) | - |
| latente congiunto BFM+ICT | synergynet 0.5202 [0.4670, 0.5707] (1.00) | 3ddfa_v2 0.5848 [0.5208, 0.6502] (0.00) | mica 0.6042 [0.5535, 0.6500] (0.00) | prnet 0.6062 [0.5425, 0.6716] (0.00) | +0.33; +0.19; 0.00 |
| Chamfer grezza (maxabs) | synergynet 0.0467 [0.0436, 0.0498] (0.93) | mica 0.0482 [0.0446, 0.0526] (0.07) | 3ddfa_v2 0.0490 [0.0455, 0.0528] (0.00) | prnet 0.0502 [0.0473, 0.0532] (0.00) | +0.67; +0.44; 0.05 |
| ICP + Chamfer (mm) | mica 2.71 [2.62, 2.81] (0.74) | synergynet 2.79 [2.58, 2.99] (0.25) | 3ddfa_v2 2.89 [2.67, 3.13] (0.00) | prnet 3.32 [3.03, 3.69] (0.00) | +1.00; +0.84; 0.61 |
| ArcFace su normal map | mica 0.5870 [0.5558, 0.6167] (1.00) | 3ddfa_v2 0.7004 [0.6596, 0.7461] (0.00) | synergynet 0.7219 [0.6787, 0.7631] (0.00) | prnet 0.7863 [0.7612, 0.8123] (0.00) | +0.67; +0.72; 0.17 |
| ArcFace su render ombreggiato (secondaria) | mica 0.5905 [0.5697, 0.6126] (1.00) | 3ddfa_v2 0.7136 [0.6711, 0.7546] (0.00) | synergynet 0.7279 [0.6834, 0.7707] (0.00) | prnet 0.7516 [0.7226, 0.7831] (0.00) | +0.67; +0.70; 0.16 |

### PRIMARIA: concordanza con NoW sui 3 metodi pre-registrati

Metodi: 3ddfa_v2, synergynet, prnet (quelli del protocollo). Per immagine: Kendall tau fra l'ordine dei metodi secondo la metrica e secondo l'errore NoW mediano dell'immagine, medio sulle immagini. Per ricostruzione: Spearman su tutte le ricostruzioni dei metodi; accanto lo Spearman dentro ciascun metodo.

| metrica | tau per immagine [CI] | Spearman [CI] | Spearman 3ddfa_v2 | Spearman synergynet | Spearman prnet |
| --- | --- | --- | --- | --- | --- |
| latente congiunto BFM+ICT | 0.299 [0.216, 0.398] | 0.477 [0.300, 0.627] | 0.515 | 0.515 | 0.348 |
| Chamfer grezza (maxabs) | 0.246 [0.123, 0.373] | 0.466 [0.245, 0.637] | 0.449 | 0.487 | 0.424 |
| ICP + Chamfer (mm) | 0.612 [0.537, 0.686] | 0.866 [0.758, 0.924] | 0.853 | 0.837 | 0.865 |
| ArcFace su normal map | 0.271 [0.145, 0.396] | 0.134 [-0.198, 0.485] | 0.042 | 0.151 | 0.094 |
| ArcFace su render ombreggiato (secondaria) | 0.269 [0.173, 0.365] | 0.234 [-0.084, 0.553] | 0.142 | 0.255 | 0.255 |

Delta appaiato pre-registrato latente congiunto BFM+ICT - Chamfer grezza (maxabs): tau_image +0.053 [-0.037, +0.153] (P<=0 0.153); spearman +0.011 [-0.156, +0.195] (P<=0 0.495).

### Secondaria: con MICA (deviazione dal protocollo)

MICA non era fra i metodi pre-registrati: e' stato aggiunto dopo (deviazione 1). Il segno della concordanza del latente dipende da MICA: il latente mette MICA ULTIMO (distanza dalla scansione piu' grande dei 4 metodi) in 139 immagini su 352, NoW in 4; le altre metriche in 96 (Chamfer grezza (maxabs)), 70 (ICP + Chamfer (mm)), 2 (ArcFace su normal map), 6 (ArcFace su render ombreggiato (secondaria)).

| metrica | tau per immagine [CI] | Spearman [CI] | Spearman 3ddfa_v2 | Spearman synergynet | Spearman prnet | Spearman mica |
| --- | --- | --- | --- | --- | --- | --- |
| latente congiunto BFM+ICT | 0.067 [-0.032, 0.184] | 0.234 [0.069, 0.401] | 0.515 | 0.515 | 0.348 | -0.079 |
| Chamfer grezza (maxabs) | 0.176 [0.041, 0.314] | 0.339 [0.141, 0.494] | 0.449 | 0.487 | 0.424 | 0.103 |
| ICP + Chamfer (mm) | 0.455 [0.311, 0.584] | 0.743 [0.592, 0.843] | 0.853 | 0.837 | 0.865 | 0.515 |
| ArcFace su normal map | 0.488 [0.394, 0.578] | 0.409 [0.182, 0.621] | 0.042 | 0.151 | 0.094 | 0.055 |
| ArcFace su render ombreggiato (secondaria) | 0.479 [0.390, 0.561] | 0.503 [0.294, 0.698] | 0.142 | 0.255 | 0.255 | 0.252 |

Delta appaiato pre-registrato latente congiunto BFM+ICT - Chamfer grezza (maxabs): tau_image -0.109 [-0.254, +0.041] (P<=0 0.922); spearman -0.105 [-0.294, +0.111] (P<=0 0.842).

## (d) Identita' fra ricostruzioni

Retrieval: query = ogni ricostruzione, distanza dal soggetto = minimo sulle altre ricostruzioni di quel soggetto, rank fra 20 soggetti (caso: rank-1 0.05). Verifica: AUC su tutte le coppie i < j. Blocco `neutral_gallery`: query non neutre, galleria e coppie solo verso multiview_neutral.


### Blocco all

| metodo | metrica | rank-1 | mAP | AUC verifica | query / coppie |
| --- | --- | --- | --- | --- | --- |
| 3ddfa_v2 | latente congiunto BFM+ICT | 0.276 [0.216, 0.338] | 0.434 [0.378, 0.487] | 0.574 [0.542, 0.616] | 352 / 61776 |
| 3ddfa_v2 | Chamfer grezza (maxabs) | 0.293 [0.250, 0.337] | 0.474 [0.428, 0.516] | 0.594 [0.558, 0.643] | 352 / 61776 |
| 3ddfa_v2 | ICP + Chamfer (mm) | 0.358 [0.269, 0.436] | 0.540 [0.470, 0.599] | 0.641 [0.596, 0.696] | 352 / 61776 |
| 3ddfa_v2 | ArcFace su normal map | 0.366 [0.296, 0.435] | 0.543 [0.484, 0.600] | 0.688 [0.641, 0.742] | 352 / 61776 |
| 3ddfa_v2 | ArcFace su render ombreggiato (secondaria) | 0.378 [0.299, 0.449] | 0.544 [0.480, 0.597] | 0.684 [0.638, 0.738] | 352 / 61776 |
| synergynet | latente congiunto BFM+ICT | 0.196 [0.136, 0.252] | 0.367 [0.312, 0.414] | 0.562 [0.529, 0.604] | 352 / 61776 |
| synergynet | Chamfer grezza (maxabs) | 0.239 [0.161, 0.306] | 0.397 [0.328, 0.457] | 0.574 [0.540, 0.617] | 352 / 61776 |
| synergynet | ICP + Chamfer (mm) | 0.233 [0.170, 0.289] | 0.414 [0.355, 0.463] | 0.582 [0.540, 0.633] | 352 / 61776 |
| synergynet | ArcFace su normal map | 0.224 [0.156, 0.288] | 0.401 [0.338, 0.465] | 0.594 [0.559, 0.628] | 352 / 61776 |
| synergynet | ArcFace su render ombreggiato (secondaria) | 0.207 [0.130, 0.283] | 0.390 [0.320, 0.458] | 0.597 [0.559, 0.642] | 352 / 61776 |
| prnet | latente congiunto BFM+ICT | 0.173 [0.125, 0.220] | 0.333 [0.282, 0.376] | 0.546 [0.529, 0.564] | 352 / 61776 |
| prnet | Chamfer grezza (maxabs) | 0.202 [0.144, 0.249] | 0.353 [0.298, 0.396] | 0.552 [0.531, 0.577] | 352 / 61776 |
| prnet | ICP + Chamfer (mm) | 0.349 [0.267, 0.416] | 0.518 [0.448, 0.577] | 0.595 [0.561, 0.635] | 352 / 61776 |
| prnet | ArcFace su normal map | 0.372 [0.281, 0.449] | 0.527 [0.446, 0.595] | 0.608 [0.574, 0.644] | 352 / 61776 |
| prnet | ArcFace su render ombreggiato (secondaria) | 0.344 [0.261, 0.414] | 0.507 [0.430, 0.570] | 0.622 [0.587, 0.660] | 352 / 61776 |
| mica | latente congiunto BFM+ICT | 0.812 [0.753, 0.866] | 0.884 [0.848, 0.917] | 0.882 [0.831, 0.922] | 352 / 61776 |
| mica | Chamfer grezza (maxabs) | 0.923 [0.880, 0.958] | 0.954 [0.928, 0.976] | 0.934 [0.905, 0.960] | 352 / 61776 |
| mica | ICP + Chamfer (mm) | 0.977 [0.959, 0.992] | 0.987 [0.976, 0.995] | 0.971 [0.947, 0.986] | 352 / 61776 |
| mica | ArcFace su normal map | 0.977 [0.960, 0.992] | 0.987 [0.977, 0.995] | 0.981 [0.968, 0.994] | 352 / 61776 |
| mica | ArcFace su render ombreggiato (secondaria) | 0.991 [0.981, 1.000] | 0.995 [0.988, 1.000] | 0.983 [0.971, 0.994] | 352 / 61776 |

Delta appaiati latente - baseline:

| metodo | baseline | rank-1 delta (P<=0) | mAP delta (P<=0) | AUC delta (P<=0) |
| --- | --- | --- | --- | --- |
| 3ddfa_v2 | ArcFace su normal map | -0.091 [-0.160, -0.029] (0.998) | -0.109 [-0.161, -0.062] (1.000) | -0.114 [-0.148, -0.086] (1.000) |
| 3ddfa_v2 | ArcFace su render ombreggiato (secondaria) | -0.102 [-0.166, -0.040] (1.000) | -0.110 [-0.154, -0.065] (1.000) | -0.111 [-0.141, -0.085] (1.000) |
| 3ddfa_v2 | Chamfer grezza (maxabs) | -0.017 [-0.062, +0.027] (0.793) | -0.040 [-0.076, -0.007] (0.991) | -0.020 [-0.031, -0.012] (1.000) |
| 3ddfa_v2 | ICP + Chamfer (mm) | -0.082 [-0.164, -0.003] (0.979) | -0.106 [-0.164, -0.054] (1.000) | -0.068 [-0.094, -0.046] (1.000) |
| mica | ArcFace su normal map | -0.165 [-0.218, -0.114] (1.000) | -0.103 [-0.135, -0.074] (1.000) | -0.100 [-0.148, -0.065] (1.000) |
| mica | ArcFace su render ombreggiato (secondaria) | -0.179 [-0.234, -0.128] (1.000) | -0.111 [-0.143, -0.081] (1.000) | -0.101 [-0.149, -0.067] (1.000) |
| mica | Chamfer grezza (maxabs) | -0.111 [-0.161, -0.059] (1.000) | -0.071 [-0.097, -0.042] (1.000) | -0.053 [-0.082, -0.031] (1.000) |
| mica | ICP + Chamfer (mm) | -0.165 [-0.220, -0.114] (1.000) | -0.103 [-0.135, -0.072] (1.000) | -0.089 [-0.125, -0.059] (1.000) |
| prnet | ArcFace su normal map | -0.199 [-0.272, -0.127] (1.000) | -0.194 [-0.261, -0.129] (1.000) | -0.061 [-0.090, -0.033] (1.000) |
| prnet | ArcFace su render ombreggiato (secondaria) | -0.170 [-0.247, -0.098] (1.000) | -0.174 [-0.241, -0.105] (1.000) | -0.075 [-0.104, -0.042] (1.000) |
| prnet | Chamfer grezza (maxabs) | -0.028 [-0.069, +0.012] (0.908) | -0.019 [-0.046, +0.008] (0.919) | -0.005 [-0.021, +0.005] (0.797) |
| prnet | ICP + Chamfer (mm) | -0.176 [-0.231, -0.111] (1.000) | -0.185 [-0.231, -0.134] (1.000) | -0.048 [-0.075, -0.025] (1.000) |
| synergynet | ArcFace su normal map | -0.028 [-0.085, +0.025] (0.853) | -0.034 [-0.078, +0.012] (0.915) | -0.032 [-0.056, -0.002] (0.986) |
| synergynet | ArcFace su render ombreggiato (secondaria) | -0.011 [-0.080, +0.049] (0.653) | -0.023 [-0.072, +0.025] (0.814) | -0.035 [-0.057, -0.009] (0.998) |
| synergynet | Chamfer grezza (maxabs) | -0.043 [-0.088, +0.000] (0.976) | -0.029 [-0.063, +0.005] (0.942) | -0.012 [-0.020, -0.003] (0.998) |
| synergynet | ICP + Chamfer (mm) | -0.037 [-0.081, +0.000] (0.980) | -0.046 [-0.074, -0.020] (1.000) | -0.020 [-0.039, +0.000] (0.974) |

### Blocco neutral_gallery

| metodo | metrica | rank-1 | mAP | AUC verifica | query / coppie |
| --- | --- | --- | --- | --- | --- |
| 3ddfa_v2 | latente congiunto BFM+ICT | 0.176 [0.105, 0.249] | 0.350 [0.285, 0.412] | 0.571 [0.550, 0.603] | 244 / 26352 |
| 3ddfa_v2 | Chamfer grezza (maxabs) | 0.189 [0.129, 0.249] | 0.377 [0.313, 0.439] | 0.586 [0.557, 0.626] | 244 / 26352 |
| 3ddfa_v2 | ICP + Chamfer (mm) | 0.287 [0.219, 0.354] | 0.453 [0.385, 0.511] | 0.640 [0.602, 0.689] | 244 / 26352 |
| 3ddfa_v2 | ArcFace su normal map | 0.266 [0.186, 0.337] | 0.435 [0.361, 0.503] | 0.678 [0.640, 0.721] | 244 / 26352 |
| 3ddfa_v2 | ArcFace su render ombreggiato (secondaria) | 0.254 [0.173, 0.324] | 0.430 [0.358, 0.496] | 0.673 [0.634, 0.719] | 244 / 26352 |
| synergynet | latente congiunto BFM+ICT | 0.156 [0.099, 0.216] | 0.299 [0.244, 0.354] | 0.568 [0.537, 0.608] | 244 / 26352 |
| synergynet | Chamfer grezza (maxabs) | 0.184 [0.112, 0.252] | 0.335 [0.265, 0.400] | 0.579 [0.543, 0.621] | 244 / 26352 |
| synergynet | ICP + Chamfer (mm) | 0.201 [0.130, 0.273] | 0.348 [0.282, 0.409] | 0.588 [0.545, 0.641] | 244 / 26352 |
| synergynet | ArcFace su normal map | 0.189 [0.107, 0.274] | 0.344 [0.263, 0.419] | 0.593 [0.552, 0.631] | 244 / 26352 |
| synergynet | ArcFace su render ombreggiato (secondaria) | 0.168 [0.107, 0.231] | 0.334 [0.268, 0.395] | 0.595 [0.552, 0.643] | 244 / 26352 |
| prnet | latente congiunto BFM+ICT | 0.111 [0.061, 0.172] | 0.278 [0.218, 0.341] | 0.548 [0.529, 0.568] | 244 / 26352 |
| prnet | Chamfer grezza (maxabs) | 0.131 [0.079, 0.186] | 0.289 [0.231, 0.349] | 0.550 [0.529, 0.577] | 244 / 26352 |
| prnet | ICP + Chamfer (mm) | 0.238 [0.181, 0.295] | 0.408 [0.345, 0.463] | 0.594 [0.561, 0.634] | 244 / 26352 |
| prnet | ArcFace su normal map | 0.262 [0.187, 0.338] | 0.443 [0.374, 0.507] | 0.610 [0.579, 0.644] | 244 / 26352 |
| prnet | ArcFace su render ombreggiato (secondaria) | 0.246 [0.170, 0.315] | 0.425 [0.360, 0.485] | 0.618 [0.583, 0.655] | 244 / 26352 |
| mica | latente congiunto BFM+ICT | 0.734 [0.663, 0.802] | 0.828 [0.770, 0.875] | 0.891 [0.844, 0.928] | 244 / 26352 |
| mica | Chamfer grezza (maxabs) | 0.869 [0.800, 0.924] | 0.919 [0.874, 0.953] | 0.938 [0.911, 0.962] | 244 / 26352 |
| mica | ICP + Chamfer (mm) | 0.967 [0.940, 0.988] | 0.980 [0.962, 0.993] | 0.973 [0.953, 0.986] | 244 / 26352 |
| mica | ArcFace su normal map | 0.943 [0.911, 0.971] | 0.968 [0.949, 0.984] | 0.982 [0.967, 0.994] | 244 / 26352 |
| mica | ArcFace su render ombreggiato (secondaria) | 0.971 [0.942, 0.992] | 0.984 [0.969, 0.996] | 0.984 [0.972, 0.994] | 244 / 26352 |

Delta appaiati latente - baseline:

| metodo | baseline | rank-1 delta (P<=0) | mAP delta (P<=0) | AUC delta (P<=0) |
| --- | --- | --- | --- | --- |
| 3ddfa_v2 | ArcFace su normal map | -0.090 [-0.184, -0.004] (0.979) | -0.085 [-0.158, -0.017] (0.997) | -0.107 [-0.136, -0.080] (1.000) |
| 3ddfa_v2 | ArcFace su render ombreggiato (secondaria) | -0.078 [-0.172, +0.008] (0.962) | -0.081 [-0.144, -0.018] (0.999) | -0.102 [-0.130, -0.075] (1.000) |
| 3ddfa_v2 | Chamfer grezza (maxabs) | -0.012 [-0.055, +0.030] (0.762) | -0.027 [-0.055, +0.001] (0.965) | -0.015 [-0.027, -0.005] (0.998) |
| 3ddfa_v2 | ICP + Chamfer (mm) | -0.111 [-0.191, -0.035] (0.999) | -0.103 [-0.163, -0.046] (1.000) | -0.069 [-0.094, -0.045] (1.000) |
| mica | ArcFace su normal map | -0.209 [-0.281, -0.136] (1.000) | -0.140 [-0.194, -0.091] (1.000) | -0.091 [-0.138, -0.057] (1.000) |
| mica | ArcFace su render ombreggiato (secondaria) | -0.238 [-0.304, -0.172] (1.000) | -0.156 [-0.210, -0.111] (1.000) | -0.093 [-0.138, -0.061] (1.000) |
| mica | Chamfer grezza (maxabs) | -0.135 [-0.178, -0.089] (1.000) | -0.091 [-0.122, -0.061] (1.000) | -0.048 [-0.079, -0.026] (1.000) |
| mica | ICP + Chamfer (mm) | -0.234 [-0.299, -0.169] (1.000) | -0.152 [-0.203, -0.106] (1.000) | -0.082 [-0.118, -0.054] (1.000) |
| prnet | ArcFace su normal map | -0.152 [-0.230, -0.068] (1.000) | -0.165 [-0.233, -0.085] (1.000) | -0.062 [-0.087, -0.037] (1.000) |
| prnet | ArcFace su render ombreggiato (secondaria) | -0.135 [-0.217, -0.050] (0.998) | -0.147 [-0.215, -0.063] (1.000) | -0.070 [-0.097, -0.038] (1.000) |
| prnet | Chamfer grezza (maxabs) | -0.020 [-0.054, +0.019] (0.884) | -0.011 [-0.041, +0.023] (0.756) | -0.003 [-0.016, +0.008] (0.646) |
| prnet | ICP + Chamfer (mm) | -0.127 [-0.183, -0.066] (1.000) | -0.130 [-0.181, -0.071] (1.000) | -0.046 [-0.075, -0.022] (1.000) |
| synergynet | ArcFace su normal map | -0.033 [-0.096, +0.032] (0.835) | -0.044 [-0.099, +0.010] (0.947) | -0.025 [-0.049, +0.002] (0.966) |
| synergynet | ArcFace su render ombreggiato (secondaria) | -0.012 [-0.051, +0.027] (0.769) | -0.035 [-0.073, +0.002] (0.966) | -0.027 [-0.051, -0.000] (0.977) |
| synergynet | Chamfer grezza (maxabs) | -0.029 [-0.074, +0.013] (0.909) | -0.036 [-0.072, -0.001] (0.977) | -0.011 [-0.020, -0.002] (0.997) |
| synergynet | ICP + Chamfer (mm) | -0.045 [-0.101, +0.012] (0.940) | -0.049 [-0.085, -0.011] (0.992) | -0.020 [-0.042, +0.002] (0.963) |
