# Ava-256 (Meta): test set CONFERMATIVO

> **CONFERMATIVO: non valutare prima del protocollo confermativo.**
> Nessun modello, baseline o metrica va calcolato su Ava-256, nemmeno per prova, prima che il PI abbia completato
> e congelato `PROTOCOL_CONFERMATIVO_bozza.md`. Ava-256 non va aggiunto a nessuno strumento di valutazione
> esistente (`fact_paired`, `fact_summary`, `baselines_*`, `diagnostics`, `zs_*`). Qui si lavora solo su dati e GT.

Fonte e motivazione della scelta: `literature/TEST_SET_VERGINE_2026-10-11.md`. Codice in questa cartella; dati, GT e
viste in `datasets/AVA256/` (in `.gitignore`, insieme a ogni file con geometria o render). Nel repo solo codice, i
manifest (nomi e hash) e numeri aggregati.

## Licenza e termini d'uso

- **Licenza: CC BY-NC 4.0** ("ava-256 is CC BY-NC 4.0 licensed, as found in the LICENSE file", README del repo;
  "The dataset is licensed under a the CC-by-NC 4.0 license", DATASHEET, sezione Distribution). Testo:
  https://creativecommons.org/licenses/by-nc/4.0/legalcode ; copia nel repo di Meta:
  https://github.com/facebookresearch/ava-256/blob/main/LICENSE (scaricata in `datasets/AVA256/meta/LICENSE`, commit
  `a9d2fbe85c2139d1c072212287b484309f3460e7` del repo, 11 ottobre 2026).
- **Termini specifici di Meta: nessuno oltre la licenza.** Il DATASHEET dichiara: nessuna restrizione di terzi "to
  our knowledge", nessun controllo all'esportazione, nessun compito vietato ("Are there tasks for which the dataset
  should not be used? No."). La pagina https://www.meta.com/emerging-tech/codec-avatars/ava256/ non contiene termini
  (letta l'11 ottobre 2026). Contatto: julietamartinez@meta.com.
- **Cosa comporta per noi** (lettura, non parere legale):
  - NC (sez. 1(i)): solo uso "not primarily intended for or directed towards commercial advantage". Un paper e un
    repo di ricerca vanno bene.
  - BY (sez. 3(a)): se si condivide materiale derivato (GT, viste, figure di mesh), vanno dati credito ai creatori,
    link alla licenza, indicazione che il materiale e' stato modificato. Citazione richiesta dal README: Martinez et
    al., "Codec Avatar Studio: Paired Human Captures for Complete, Driveable, and Generalizable Avatars", NeurIPS
    Datasets and Benchmarks 2024.
  - **Figure di volti.** La sez. 2(b)(1) dice che "publicity, privacy, and/or other similar personality rights" NON
    sono licenziati. Il consenso dei soggetti e' "for both capture and distribution of the data" (DATASHEET), ma il
    testo del consenso non e' pubblico. Quindi la licenza non basta a coprire render riconoscibili di singole persone
    in figura. **Raccomandazione:** in figura solo forme medie o aggregate; per render di singoli soggetti, una conferma
    scritta da Meta (email gia' proposta in `literature/TEST_SET_VERGINE_2026-10-11.md`, sez. 7).
  - Nessuna redistribuzione dei dati da parte nostra e' necessaria: si pubblicano numeri, ID delle catture e script.
    Se si pubblicassero GT o viste, varrebbe la BY sopra.

## Dati scaricati (download minimo)

`ava_download.py`, sul frontend (serve la rete; solo libreria standard). Release `4TB`, bucket S3 pubblico di Meta.
Per ognuna delle 256 catture di `256_ids.csv`: `frame_list.csv` intero, e dallo zip `registration_vertices.zip`
(~441 MB, un PLY non compresso per frame) solo i frame scelti, con Range request. Ogni file e' verificato col CRC32
della directory dello zip; `download_manifest.json` (in git) ha per cattura i conteggi e l'impronta degli sha256,
`datasets/AVA256/manifest_files.json` lo sha256 di ogni file.

## Scelte dichiarate PRIMA di calcolare corrispondenza, neutre, GT e viste (11 ottobre 2026)

Viste prima di queste scelte: due soggetti (AAN112, ADL311) scaricati per intero nei due segmenti neutri, in
`datasets/AVA256/explore/`, per capire formato e topologia. Nessuna GT, nessun numero sugli altri 254.

**Fatti sulla topologia (verificati).** 7306 vertici, 5741 usati da 11.432 triangoli, una sola componente e un solo
anello di bordo (48 vertici, il taglio del collo): testa chiusa, senza buchi per occhi e bocca. Le PLY stanno in un
frame locale della testa (+y in alto, +z avanti, origine vicino al centro della testa). La topologia **estende quella
di Multiface**: i 5471 vertici usati da Multiface sono usati anche qui con gli stessi indici, 10.764 triangoli su
10.936 di Multiface sono identici.

1. **Frame.** Neutra: tutti i frame di `EXP_neutral_peak` (9-27 per soggetto). Neutro ripetuto: 16 frame a
   posizioni equispaziate di `EXP_eye_neutral` (un altro segmento della stessa sessione, a volto neutro con
   movimenti degli occhi). Le catture sono una sola per persona (la seconda cattura del DATASHEET e' col visore e non
   ha mesh): non esistono neutri da sessioni diverse.
2. **Neutra di una cattura** (regola di FaMoS, `aau/famos/famos_subsample.py`): frame portati sulla regione
   unificata, Procrustes di similarita' verso mu, distanze g in mm; medoide; si tengono i frame entro 2 x la mediana
   delle distanze dal medoide; neutra = media dei tenuti allineati rigidamente (senza scala) al medoide sui punti
   della regione pesati per area, trasformazione applicata a tutta la mesh. Rumore: distanza g fra le medie dei
   tenuti pari e dispari. Neutro ripetuto: la stessa regola sui 16 frame di `EXP_eye_neutral`.
3. **Regione del volto per la GT: la stessa degli altri domini.** I 1478 punti della regione unificata
   (`datasets/UNIFIED_GT/unified_space.npz`), portati sulla topologia Ava-256 con **la stessa procedura di
   `v3_work/unified_gt/correspond.py`** usata per gli 8 domini di E8: landmark FaceMesh sui render frontali di FLAME e
   del template, similarita' robusta, NICP della regione FLAME sul template guidato dai landmark, mappa baricentrica
   del punto piu' vicino. Template = per cattura la media dei frame di `EXP_neutral_peak` (rigida sui vertici usati,
   verso il primo frame), poi Procrustes generalizzato fra le 256 catture nel frame della prima (come
   `domains.multiface`). La regione NON si restringe: se qualche punto non e' "coperto" (sul bordo del template o a
   piu' di 2 mm), resta nella GT e si segnala; sensibilita' sulla sola parte coperta (Spearman fra GT).
   `unified_space.npz` e le corrispondenze degli altri domini non si toccano: la mappa di Ava-256 sta in
   `datasets/AVA256/corr/`. **Controllo indipendente:** la mappa di Multiface trasferita per identita' di indice
   (topologia condivisa): distanza fra le due mappe sul template e Spearman fra le due GT.
4. **Unita'.** Come E12 (`v3_work/canonical_gt/gt.py`, sez. 2b): IPD in unita' native dai 12 punti del contorno degli
   occhi (iBUG 36-47 sulla media FLAME, punto piu' vicino della regione unificata), `u` = la potenza di 10 piu'
   vicina a 63 mm / IPD se sta entro il +-15%, altrimenti `u = 63 / IPD` e unita' dichiarata arbitraria. Controlli
   indipendenti: larghezza intercantale (iBUG 39-42) e biorbitale (36-45) contro i valori antropometrici di adulti
   (circa 31-33 e 87-91 mm), centroid size della regione contro gli altri domini in mm (54-59 mm, E12
   `size_cv.csv`).
5. **GT.** FR e SR di `v3_work/canonical_gt/train_fr_sr.py` (`factor`, importato): punti della regione della neutra in
   mm, rigida robusta per identita' verso mu (IRLS Tukey, come FaMoS: catture reali con posa), FR = distanza RMS
   pesata per area in mm, SR = la stessa fra pre-forme a centroid size 1. Formato delle GT di valutazione (`D_orig`
   diviso per il massimo, json con l'unita'), ma in `datasets/AVA256/gt/` e NON in `datasets/CANONICAL_GT/eval/`.
6. **Validita' di un soggetto.** Escluso solo per errore dei dati: meno di 3 frame di `EXP_neutral_peak`, vertici non
   finiti, rigida robusta non convergente. I soggetti anomali (centroid size, IPD, distanza da mu o rumore della
   neutra oltre mediana +- 5 x 1.4826 MAD) restano nella GT e si elencano per il PI.
7. **Viste.** Patch `original` = i triangoli del template nell'impronta della regione FLAME
   (`datasets/UNIFIED_GT/flame_region.npz`: maschera `face` senza l'interno di occhi e bocca, 1671 vertici) deformata
   sul template dallo stesso NICP: un vertice entra se il suo punto piu' vicino sulla regione deformata non e' sul
   bordo ed e' entro 2 mm; triangoli con tre vertici dentro, componente connessa piu' grande; lo stesso insieme di
   indici per tutti. Coordinate: la neutra nel frame nativo dei dati (frame della testa del tracking, mm), SENZA la
   rigida della GT, che non deve passare negli ingressi. Le sei topologie con lo stesso codice di HIFI3D, FaceVerse e
   FaceScape dev (`aau/zs3dmm/make_zs_topologies.py` -> `make_ict_topologies.process_subject`: original, remesh,
   crop, noisy, down8k, up60k), controllo del crop con `zs_check_crop.py`, vista `id<950000 + k>_GTready_<topologia>`
   (k = riga di `256_ids.csv`). Nessun operatore, nessun embedding.

## Modifiche dell'11 ottobre, PRIMA di qualsiasi valutazione (critic: RISERVE; decisioni del PI)

Quando sono state decise e applicate non esisteva nessuna valutazione: nessun modello, embedding, baseline o metrica di
prestazione su Ava-256. Sostituiscono le scelte indicate sopra; il resto della sezione precedente vale ancora. Le scelte
sopra restano com'erano (sha256 delle prime 98 righe: `2c0515af...`, commit `985bed2`).

1. **Viste equivalenti a quelle di sviluppo (scelta 7, viste).** La original era 2-3 volte piu' rada delle original dev
   (lato medio 3,95 mm contro 1,38-2,11), quindi remesh e down8k la perturbavano molto di piu' (spostamento normale dello
   smoothing di remesh 0,465 mm contro 0,08-0,17; lato di down8k 7,4 mm contro 2,3-4,0). Ora la original e' **suddivisa
   1-a-4 a punto medio una volta** (`igl.upsample`, la superficie non cambia; lo stesso trattamento di FLAME 2023 in D1)
   prima delle sei topologie, prodotte con lo stesso codice e gli stessi parametri. Esito sotto: dentro gli intervalli dev,
   nessuna taratura dei parametri.
2. **Patch: solo bordo esterno, bocca e aperture palpebrali (scelta 7, patch).** La patch precedente aveva 9 anelli di
   bordo (narici, frammenti attorno agli occhi, pezzi di calotta oculare) e 64 punti della GT fuori dalla vista.
   `ava_patch.py` toglie solo i triangoli che chiudono le aperture palpebrali (la calotta) e la rima delle labbra: quelli
   col baricentro dentro gli anelli della regione FLAME deformata (la regione della GT), in proiezione frontale. Narici e
   frammenti si chiudono coi triangoli di pelle del template. Gli occhi restano aperti come in HIFI3D e FaceScape dev
   ("niente bulbo oculare"; FaceVerse invece copre gli occhi). Le regole provate e scartate sono nel docstring.
3. **Impronta indipendente dai modelli confrontati (scelta 7, impronta).** Il contorno non e' piu' l'impronta della
   maschera `face` di FLAME (la regione del fit B-FLAME, un vantaggio per costruzione): e' **la sfera NoW delle viste
   FaMoS** (`now_common.now_mask`: centro subnasale + 0,3 (ponte - subnasale), raggio 1,4 (ex-ex + naso) / 2), dai
   landmark iBUG 36, 39, 42, 45, 33 portati sul template con la corrispondenza; restano i triangoli coi tre vertici dentro.
4. **Protocollo eseguibile: split di calibrazione 56 + 200.** Con 256 valutati su 256 il template del NICP e la regione
   del fit B (`blmm.template_subjects`) non avevano soggetti. Split fisso (`ava_common.split`): i primi 56 sid in ordine
   di sha256(`ava256-confermativo-calibrazione-2026-10-11:` + sid) sono di **calibrazione** (template del NICP, regione di
   B, L_d, cs_ref; mai valutati, da nessun metodo), gli altri **200 sono i valutati**. Elenco in `split.json`. Viste
   separate (`eval_view`: 200; `calib_view`: 56), GT della prova sui 200. Tabella di scala del modello con la definizione
   di `build_scale_table.py`, che non si tocca: `ava_scale_table.py`.
5. **Unita' (scelta 4).** **u = 1 (mm), dichiarata** per analogia con Multiface (stesso laboratorio e stessa famiglia di
   topologia, mm in `domains.py`) e per la coerenza della centroid size (57,93 mm contro 56,10 della media FLAME, +3,3%).
   La regola letterale di E12 (63 mm / IPD, con l'IPD mediana) darebbe 0,954: una scala globale entro la tolleranza del
   15%, che dentro il dominio non cambia gli Spearman.
6. **Qualita' (scelta 6).** Tolta la regola MAD sulla distanza fra le meta' pari e dispari della neutra: i frame sono
   consecutivi, quindi non e' informativa. Al suo posto la **stabilita' su tutti i frame** del segmento: distanza g di
   ogni frame dal medoide; e' segnalato chi ne ha uno oltre 1 mm. Le regole MAD su centroid size, IPD e distanza da mu
   restano. I segnalati restano nella GT; i loro id stanno solo fuori da git.
7. **Congelamento.** `ava_freeze.py` registra le impronte del contenuto in `freeze_manifest.json` (in git) e in
   `datasets/AVA256/FROZEN.json`:
   - sha256 di `V` float32 e `F` int32 di ogni mesh;
   - sha256 di `D_orig` e dei nomi delle GT;
   - sha256 di nomi e aree della tabella di scala.

   Poi rende i dati in sola lettura, file e cartelle. Con `FROZEN.json` presente si rifiutano di scrivere
   `ava_build.sbatch` (guardia sui passi), `ava_gt.py`, `ava_views.py` e `ava_scale_table.py`. Rigenerare, che
   cambierebbe `up60k`, richiede una decisione del PI. **`ava_freeze.verify_frozen()` e' la funzione che lo strumento di
   valutazione deve chiamare prima di leggere viste, GT o tabella.**
8. **Repo pubblico.** Tolta la descrizione della barba legata a un id; gli id dei soggetti segnalati non stanno piu' in
   git. La versione precedente del README (commit `82c062d`, gia' pubblicato) contiene ancora id e descrizione: toglierla
   dalla storia richiede una riscrittura forzata, che non ho fatto.
9. **Affidabilita'.** Lo Spearman 0,972 fra la GT della neutra e quella del neutro ripetuto e' una stima del rumore fra
   segmenti diversi della stessa sessione. **Non e' un tetto per gli Spearman dei metodi**: viste e GT vengono dalla stessa
   neutra.

## Esito (dopo le modifiche)

Catena completa in un job (`ava_build.sbatch`, job 1068115; congelamento a parte, sotto). Numeri aggregati in
`corr_summary.json`, `patch_summary.json`, `neutral_summary.json`, `gt_summary.json`, `size_cv.csv`, `views_summary.json`,
`view_qc.json`, `split.json`, `freeze_manifest.json`; nessuna geometria in git.

| Passo | Script | Esito |
|---|---|---|
| download | `ava_download.py` | 256/256 catture, 8.756 file, **781 MB** su disco (870 MB letti in rete), CRC32 tutti corretti. `EXP_neutral_peak` 7-54 frame per persona (mediana 15); frame "dropped" in 21 catture (1-4 neutri in 11; ava0140 ha 6 frame su 16 di `EXP_eye_neutral`) |
| corrispondenza | `ava_corr.py` | 339 landmark, residuo 1,52 -> 0,49 mm (tenuti fuori 0,65 mm), Chamfer 0,208 mm, nessun triangolo girato; **regione unificata coperta 1478/1478** (regione FLAME 1671/1671). Mappa di Multiface per indice: 1,45 mm in mediana dalla nuova |
| patch | `ava_patch.py` | sfera NoW di 98,7 mm (frame FLAME del template), **4 anelli** (87 esterno, 44 bocca, 39 e 39 occhi), nessun buco spurio. Buchi: occhi 25,7 x 10,6 e 27,0 x 10,0 mm, bocca 39,0 x 7,4 mm |
| neutre | `ava_neutral.py` | **256 validi su 256**; neutro ripetuto per tutti; nessun vertice non finito, nessun triangolo degenere |
| unita', split, GT | `ava_gt.py` | u = 1 (mm); split 56 + 200; FR fra persone sui 200 valutati: mediana 6,58 mm |
| viste | `ava_views.py`, `make_zs_topologies.py`, `zs_check_crop.py` | `eval_view` 1.200 mesh (200 x 6, `id950000`-), `calib_view` 336 (56 x 6) |
| tabella di scala | `ava_scale_table.py` | 1.200 + 336 mesh, sqrt(area) mediana della original 189,0 mm (valutati) |
| controlli delle viste | `ava_view_qc.py` | sotto |

**Viste contro i domini dev** (`view_qc.json`, definizioni del critic, mediana su 10 mesh per dominio):

| Quantita' | Ava-256 prima | **Ava-256 ora** | HIFI3D / FaceVerse / FaceScape dev |
|---|---|---|---|
| lato medio della original | 3,95 mm | **1,856 mm** | 2,109 / 1,377 / 1,683 |
| spostamento normale dello smoothing di remesh | 0,465 mm | **0,134 mm** | 0,171 / 0,083 / 0,150 |
| lato medio di down8k | 7,43 mm | **3,457 mm** | 4,007 / 2,314 / 3,549 |
| spostamento tangenziale (non un criterio) | 0,96 mm | 0,368 mm | 0,114 / 0,321 / 0,319 |
| anelli di bordo, tutte e 6 le topologie | 9 (original) | **4** | 5 / 2 / 4 |

Le tre quantita' del criterio stanno nell'intervallo dev; lo spostamento tangenziale supera del 15% il massimo dev, ma
non cambia la superficie. Topologie della vista (mediana): original 12.225 vertici / 24.036 triangoli, remesh
8.591 / 16.825, crop 11.409 / 22.364 (tiene il 91,6-94,6% dei vertici, diverso dalla original per tutti), noisy
12.225 / 24.036, down8k 4.266 / 8.281, up60k 31.367 / 62.109. Pieghe della original (coppie di triangoli a piu' di 120
gradi): mediana 14 su 24.036 triangoli (FaMoS: 22 sulle scansioni e 41 sulle registrazioni, su 5.215).

**Copertura della GT dentro la vista:** 1434/1478 punti su triangoli della vista, 1472/1478 coi tre vertici nella vista.
Il residuo e' di 44 punti, lo **0,35% del peso d'area**. Stanno sui margini palpebrali e sulla rima, dove il NICP ha
steso palpebre e labbra FLAME sulla chiusura di Ava: triangoli lunghi senza vertici interni, quindi interpolano i vertici
dei margini.

**Impronta:** tutti i 1478 punti della GT stanno dentro la sfera (distanza massima 0,92 del raggio). Con i landmark
FaceMesh (indici 33, 133, 362, 263, 2) al posto di quelli trasferiti, i landmark distano 1,1-3,6 mm, il centro della
sfera si sposta di 0,9 mm e il raggio cambia dell'1,1%.

**Unita': mm** (modifica 5). IPD mediana 66,0 mm; con gli stessi sostituti dei landmark, Ava-256 sta sopra la media
FLAME del 2,8% (IPD), del 2,1% (larghezza intercantale) e del 3,3% (biorbitale). La centroid size della regione e' 57,9
mm, contro 54,2-59,0 mm degli altri domini (FaMoS 57,7). La scala della similarita' canonica di `correspond` (0,884)
torna a 0,970 tolta la taglia della prima cattura, su cui e' costruito il template.

**CV della taglia sui 256** (IC 95% bootstrap per soggetto, 1000 repliche, seme 1234):
- centroid size **4,78% [4,36; 5,20]**;
- IPD 5,93% [5,30; 6,55];
- altezza della regione 6,50% [5,86; 7,15];
- correlazione centroid size-IPD 0,71.

Sta nella fascia plausibile di E12 (4-7%: FaMoS 4,0%, Multiface 5,2%, ICT 4,8%, GNM 5,3%).

**Qualita'.** Stabilita' su tutti i frame (modifica 6): la distanza massima di un frame dal medoide ha mediana 0,13 mm e
p95 0,81 mm. **4 soggetti** hanno un frame oltre 1 mm (massimo 1,94 mm). Regole MAD: 1 soggetto segnalato per la distanza
da mu (8,0 mm contro 3,9 di mediana), nessuno per centroid size o IPD. In tutto 5 segnalati, 4 dei quali fra i 200
valutati; restano nella GT. La superficie e' quella visibile: il tracking segue barba e capelli, e un soggetto ha una
barba folta che sposta la mandibola (controllo visivo interno, render fuori da git).

**Affidabilita' della GT** (solo GT contro GT, 256 soggetti; modifica 9):
- con il neutro ripetuto di `EXP_eye_neutral`, Spearman FR 0,972 e SR 0,966 sulle 32.640 coppie;
- distanza FR intra-persona mediana 0,76 mm (p95 2,08) contro 6,68 mm fra persone;
- per 256 persone su 256 il neutro ripetuto ha come neutra piu' vicina la propria;
- le distanze intra-persona alte (fino a 4,7 mm) sono espressione nel segmento ripetuto (sopracciglia alzate,
  mandibola), quindi la stima sovrastima il rumore;
- con la mappa di Multiface invece della nuova: Spearman FR 0,996.

**Riproducibilita'.** Due esecuzioni complete coincidono sulla GT entro 1,3e-4 mm: la GT dei 256 di questa esecuzione ha
la stessa impronta della precedente (`58a3ba2c...`). La mappa coincide entro 0,0025 mm. `up60k` cambia tassellazione a
ogni rigenerazione (la decimazione amplifica differenze di 1 ulp). Per questo GT, viste e tabella di scala sono congelate
per contenuto (modifica 7). Impronte in `freeze_manifest.json`.

**Protocollo:** `PROTOCOL_CONFERMATIVO_bozza.md`, da completare dal PI prima di qualsiasi valutazione.
