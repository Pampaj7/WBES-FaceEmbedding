# Protocolli di valutazione della comunita' sul nostro benchmark: NoW, NoW metrical (MICA), REALY, Stirling/ESRC

Codice: `aau/competitors/protocolli/` (`pr_common.py`, `pr_landmarks.py`, `pr_metrics.py`, `pr_run.py`, `pr_eval.py`,
`pr_selftest.py`, `pr_official_check.py`, `pr.sbatch`). Definizioni complete nei docstring. Resoconto dei protocolli
con file:riga del codice ufficiale: `literature/PROTOCOLLI_BENCHMARK_2026-10-12.md`. Numeri: `results.md` (generato),
`spearman.csv`, `deltas.csv`; controlli in `checks/`, preparazione in `prep/`.

## Cosa si calcola

Ogni protocollo e' un errore di una coppia ORDINATA (A predetta, B riferimento). La distanza di una coppia della
matrice e' la **media dei due versi**, d(a, b) = (e(A = a, B = b) + e(A = b, B = a)) / 2. Mesh in mm nel frame canonico
(`blmm.to_mm`), nessuna normalizzazione per mesh: ogni protocollo fa il proprio allineamento, come lo fa la comunita'.

| metodo (cartella plug-in) | allineamento | errore |
|---|---|---|
| `now_original` | Procrustes a 7 landmark con scala (riflessione 'best') + ICP scan-to-mesh con s, R, t (dogleg di chumpy) | media delle distanze vertice di B (cerchio NoW, raggio 1.4) -> superficie di A, mm |
| `now_metrical` | Procrustes applicato senza scala (fattore d'unita' 1, traslazione calcolata con la scala, come il codice) + ICP R, t | come sopra |
| `realy` | gICP con scala sui 7 keypoint, ICP rigido per ognuna delle 4 regioni con i keypoint come peso, nICP della regione di B verso A | media per regione di ||corrispondenza aggiornata - vertice di B||, in mm con la scala metrica di B; totale = media delle 4 regioni |
| `stirling` | Procrustes a 7 landmark con scala, nessun ICP | RMSE delle distanze vertice di B (cerchio raggio 1.2) -> superficie di A, mm |

## Formato delle uscite (plug-in della matrice unificata)

`aau/runs/evidence/competitors_new/<metodo>/`: `fm_plugin.json` (etichetta, famiglia, lettura `mm` dichiarata per GT FR
e SR), `<vista>.npz` (fuori da git: `subjects`, `topologies`, `paths`, `D_mm` (n, n) simmetrica, NaN fuori dalle coppie
calcolate; REALY anche `R_<regione>`, NoW anche `Q_median`: chiavi che il loader non legge), `<vista>.json` (costi,
iterazioni, riflessioni, fallimenti, asimmetria fra i versi), `pairs_<vista>.csv` (fuori da git). Viste: `hifi3d`,
`facescape`, `faceverse`, `faceverse_neutral` (righe di all_cross, crop compreso; le righe "stessa topologia" della
matrice NON sono calcolate: NaN, passo a parte in `fm_eval`), `famos` (TEST, 15 scansioni di galleria).

## Landmark (definiti una volta per generatore, `aau/competitors/protocolli/landmarks/<dominio>.json`)

- **Fonte**: i 68 iBUG dell'embedding FLAME 2020 di DECA/MICA (`external/MICA/data/FLAME2020/landmark_embedding.npy`,
  lo stesso delle viste FaMoS) e gli 85 keypoint di REALY su FLAME (`REALY/data/FLAME.txt`; triangoli di
  `REALY/data/FLAME.obj` = FLAME 2020, controllato), portati sul template del generatore con la mappa FLAME -> template
  della GT unificata (`datasets/UNIFIED_GT/corr/<dominio>.npz`, NICP guidato da landmark; residuo sui landmark tenuti
  fuori ~1-1.5 mm, `corr/<dominio>.json`), come `aau/ava256/ava_patch.py:183`; poi il vertice piu' vicino della media
  delle original di 100 soggetti NON valutati. 7 landmark NoW / Stirling = iBUG 36, 39, 42, 45, 33 (subnasale, il "nose
  bottom" di `compute_mask`, come DECA/MICA), 48, 54; REALY usa i suoi 85 (7 = gli stessi indici iBUG, ma sul FLAME di
  REALY il 33 sta 2.7 mm piu' in basso che nell'embedding di MICA).
- **Aggancio** (distanza punto trasferito -> vertice scelto) sui 7: HIFI3D 0.16-0.45 mm, FaceScape 0.07-0.51,
  FaceVerse 0.23-0.83. Frame: media delle original -> template, mediana 0.13-0.29 mm.
- **Controllo indipendente (HIFI3D)**: i keypoint NATIVI di REALY per la topologia HIFI3D (`REALY/data/HIFI3D.txt`)
  contro quelli trasferiti da FLAME: 0.4-3.9 mm sui 7 (ex_l 3.9 mm), mediana 2.8 mm sugli 85.
- **Altre topologie**: punto piu' vicino della mesh perturbata alla posizione sulla original dello stesso soggetto
  (`prep/<vista>.json`: distanza per topologia). **FaceVerse con espressioni**: ogni (identita', topologia) ha
  un'espressione propria (`aau/zs3dmm/make_zs_expr_topologies.py`), quindi la "original dello stesso soggetto" e' qui
  la original con la STESSA espressione della mesh, rigenerata dai pesi d'identita' e dal vettore d'espressione salvati
  (funzioni di quel modulo, importate; rigenerata contro la original su disco: scarto <= 5e-6 mm). **Crop**: nessuno dei
  7 landmark NoW cade fuori dal crop in HIFI3D, FaceScape, FaceVerse neutra (distanza > 0.5 mm); gli 85 di REALY sul
  contorno si' (usato comunque il punto piu' vicino). **FaMoS TEST**: dalla registrazione FLAME dello stesso fotogramma
  (dati grezzi) con la similarita' di `famos_test_view.canonical_patch` (scala ricalcolata = `scale_to_mm` del
  manifest), poi il punto piu' vicino della scansione (distanza fino a 1.9 mm: registrazione != scansione).
- **Landmark oracolo (cautela principale)**: i landmark sono esatti per costruzione (corrispondenza del generatore,
  nessun rumore di rilevatore o annotatore). Nella comunita' vengono da annotazioni manuali (NoW, Stirling) o da un
  rilevatore; i protocolli basati su landmark (Procrustes di NoW, NoW metrical, Stirling, gICP di REALY) sono quindi
  valutati nel loro caso MIGLIORE. Nessuna sensibilita' al rumore dei landmark calcolata qui.

## Regioni di REALY (approssimazione dichiarata)

Le 4 maschere di REALY (naso, bocca, fronte, guancia) sono definite sul template HIFI3D di REALY e distribuite solo
come regioni ritagliate delle scansioni del benchmark: non applicabili alle nostre mesh. Approssimazione: maschere
ufficiali di FLAME (`FLAME_masks.pkl`) trasferite con la stessa mappa dei landmark, naso = `nose`, bocca = `lips`,
fronte = `forehead` U `eye_region` (REALY ancora la fronte ai keypoint di occhi e sopracciglia, iBUG 17-26 e 36-47,
`utils/util.py:31`: con la sola `forehead` di FLAME quei keypoint cadono fuori e il nICP trascina il bordo; controllo
6 di `pr_selftest.py`), guancia = resto della regione del volto FLAME; dentro il disco centrato sulla punta del naso
(paper Sez. 4; raggio non dato: quello di Stirling), componente connessa piu' grande. Riferimento nel frame e nelle
unita' del template HIFI3D di REALY (similarita' sui 7 keypoint, la funzione `fit_icp_RT` di REALY): i parametri del
nICP (soglia `max(2 media, 1)`, epsilon, rigidezza) sono in quelle unita' (~9.4 mm per unita' sulle nostre mesh,
REALY dichiara 6.5-9.9 per le sue scansioni); errore riportato in mm con la scala metrica del riferimento.
**Decimazione**: regioni oltre 1500 vertici decimate (quadriche, open3d) a 1500, per il costo del nICP (sistema 4N x 4N
per passo, ~44 soluzioni per coppia). Controllo su 64 coppie per vista contro un tetto di 3000: Spearman 0.990
(HIFI3D) / 0.993 (FaceVerse), scarto relativo mediano 1.3%, p95 4.6-6.8%.

## Fedelta' ai protocolli: cosa e' identico e cosa e' approssimato

- **NoW originale e metrical**: logica di `scan2mesh_computations{,_metrical}.py` riscritta (Procrustes con
  riflessione 'best', cerchio 1.4, ICP scan-to-mesh con il dogleg di chumpy riscritto: stessi `delta_0 = 0.1`,
  `e_3 = 1e-4`, `maxiter = 200`, termine `s_reg`; il passo di Gauss-Newton 7 x 7 con `np.linalg.solve` invece del `cg`
  a 2000 iterazioni, stessa soluzione). Punto piu' vicino con open3d (Embree) invece dell'AABB di psbody. ICP su
  TUTTI i punti del riferimento ritagliato in tutte le viste. **Controllo contro il codice ufficiale** (clone
  `now_evaluation` 7bd1498, venv `external/venvs/now`, `checks/official_hifi3d.json`): 12 coppie HIFI3D, sulle 10 dove
  il codice ufficiale termina scarto mediano 2e-6 mm, massimo 1.7e-4 mm (originale) e 6.8e-5 mm (metrical), stessi
  punti di riferimento; sulle 2 con una up60k come predetta il codice ufficiale ha residuo NaN dal primo passo
  (triangoli di area nulla) e il suo dogleg non si ferma piu' (rho NaN non riduce il raggio, `max_fevals` infinito):
  tagliato a 600 s. Il nostro port su quelle coppie da' un valore finito.
- **Media, non mediana**: la distanza della coppia e' la media delle distanze (la sfida NoW ordina per mediana; la
  mediana e' in `Q_median` dei npz).
- **NoW metrical, fattore d'unita'**: le nostre mesh sono in mm, fattore 1; la traslazione del Procrustes e' calcolata
  con la scala stimata e applicata senza (`:117-125`), riprodotta com'e' (l'ICP rigido corregge la traslazione).
- **Stirling**: identico a `compute_vertices_to_mesh_distances.py` (distanze da open3d invece che da pymesh), RMSE.
- **REALY**: pipeline identica nei passi e nei parametri (gICP con la scala L1, rICP per regione con pesi dei keypoint
  e arresto a 1e-6 / 100 iterazioni, nICP con e senza keypoint, errore come distanza fra corrispondenze), con tre
  approssimazioni dichiarate sopra: regioni da FLAME, decimazione a 1500 vertici per regione, normalizzazione nel frame
  del template di REALY con la nostra similarita' sui 7 keypoint (REALY la fa sulle scansioni con un procedimento
  iterativo di rilevamento). Il nICP senza keypoint (guancia) non ha tetto di passi nel codice ufficiale: qui 200 per
  decadimento, mai raggiunto (`hit_cap` = 0 in tutte le coppie). Su una coppia identica a meno di similarita' l'errore
  e' 0.08 mm, non 0: il keypoint del riferimento e' il suo vertice piu' vicino nella regione (`eval.py:51`).
- **Simmetria**: ogni protocollo e' asimmetrico; distanza = media dei due versi. Asimmetria relativa mediana fra i
  versi 10-14% (`<vista>.json`, `asymmetry_relative`).
- **FaceVerse con espressioni**: i protocolli confrontano mesh con espressioni diverse (anche fra topologie dello stesso
  soggetto), come ogni altro metodo della matrice; nessuna compensazione dell'espressione (non e' nei protocolli).

## Controlli eseguiti (tutti eseguiti, non solo letti)

- `pr_selftest.py` (8/8): Procrustes, jacobiano di Rodrigues, NoW su una similarita' nota (scala ritrovata 1.08000),
  NoW metrical sulla stessa B riscalata (1.41 mm) e rigida (0.0000 mm), dogleg = Levenberg-Marquardt di scipy
  sullo stesso obiettivo (18682.096 contro 18682.097 mm^2), nICP (splu) = lsqr (7.7e-12), REALY 0.08 mm, Stirling 2e-14.
- `checks/official_hifi3d.json`: sopra.
- `checks/sample_<vista>.json` (64 coppie ordinate a caso): tempi per metodo; ICP di NoW su 4000 punti contro tutti
  (Spearman 0.9989-0.9997, non usato: tutte le viste sono calcolate su tutti i punti); REALY con 1500 contro 3000
  vertici per regione (Spearman 0.990 HIFI3D, 0.994 FaceVerse con espressioni).
- Loader della matrice unificata (`fm_eval.plugin_dirs` + `plugin_columns`, importati in sola lettura): 148.500 /
  148.500 righe di all_cross finite per vista e metodo, FaMoS 15 x 15, letture dichiarate per FR e SR.
- `prep/<vista>.json`: distanze di trasferimento dei landmark per topologia, landmark fuori dal crop, vertici per regione
  prima e dopo la decimazione, scala metrica di REALY, rigenerazione delle original di FaceVerse con espressioni.
- Unione: nessun verso mancante e nessun errore in nessuna vista e metodo. Riflessioni scelte dal Procrustes con
  `reflection='best'` (comportamento del codice ufficiale, non corretto): 0 ovunque tranne FaceVerse con espressioni,
  854 coppie ordinate su 297.000 (0.29%; landmark spostati dall'espressione), dove l'ICP (solo rotazioni) non puo'
  annullarle. Scala stimata per coppia da NoW originale (5-95 percentile): 0.90-1.13 HIFI3D, 0.95-1.06 FaceScape,
  0.95-1.09 FaceVerse, 0.96-1.08 neutra, 0.90-1.15 FaMoS.

## Risultati in breve (tabelle complete con IC in `results.md`)

Spearman con la GT, righe senza crop (IC 95%, 1000 repliche per soggetto). Con GT FR, NoW metrical (Procrustes senza
scala + ICP rigido) e' alla pari o sopra il nostro d_F non calibrato su tutte le famiglie sintetiche (HIFI3D 0.697
contro 0.643, delta appaiato +0.055 [+0.009, +0.105]; FaceScape 0.677 contro 0.687, pari; FaceVerse e neutra pari) e
nettamente sopra con il crop (HIFI3D +0.140, FaceScape +0.174). NoW originale (con la scala per coppia) perde contro
NoW metrical con GT FR in tutti i domini (HIFI3D -0.281 [-0.371, -0.194]; FaceScape -0.053; FaceVerse -0.038; neutra
-0.021 non risolto), e lo supera con GT SR (HIFI3D +0.189): la scala per coppia toglie la taglia. Con GT SR REALY e'
il migliore dei protocolli (HIFI3D 0.715, neutra 0.521) e sopra il nostro d_P su HIFI3D (+0.093) e sulla neutra
(+0.188); NoW originale e' pari al nostro d_P senza crop e pari o sopra con il crop. **Cautela che condiziona tutto**: landmark
oracolo (sopra); i protocolli basati su landmark sono qui nel loro caso migliore.

## Costi e job (12-13 ottobre 2026, solo CPU, partizione prioritized)

Ore-core di calcolo (somma dei tempi per mesh A): NoW originale + metrical + Stirling 82 (HIFI3D), 103 (FaceScape), 410
(FaceVerse), 349 (neutra); REALY 584, 259, 382, 525 (i nodi a512-l4 e nv-ai-02 sono ~2x piu' lenti degli a768 per
REALY). Secondi per coppia ordinata (mediana): NoW 0.3-1.6, Stirling 0.03-0.11, REALY 3-6.5.

Job: collaudo 1070916 (fermato dal selftest REALY fallito, corretto), 1070942 (annullato: codice NoW ufficiale in ciclo
infinito su una up60k), 1070965 (controllo ufficiale, campione, prep); NoW + Stirling 1070966 (HIFI3D), 1071033 (FaMoS
tutti i metodi, FaceScape), 1071059-1071062 (FaceVerse e neutra, due parti ciascuna); REALY 1071025-1071026 (HIFI3D),
1071071 (FaceScape), 1071074-1071077 (FaceVerse e neutra) con gli ausiliari 1071119-1071121 e 1071140 (stesse parti
dall'altro capo, salvataggio per mesh A), riserva 1071072 (annullata, non servita); valutazione 1071118 (prima di
REALY su FaceVerse) e 1071144 (finale). Prove brevi con srun: 1070888, 1070890, 1071036, 1071037, 1071054, 1071145.
