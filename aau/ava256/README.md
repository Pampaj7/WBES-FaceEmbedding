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
